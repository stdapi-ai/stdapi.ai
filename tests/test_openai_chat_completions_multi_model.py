"""Multi-model parametrized tests for the OpenAI /v1/chat/completions route.

Covers representative "real-world" usage across one model per provider family
available on AWS Bedrock, including:

  - Basic chat completion and usage tokens
  - Streaming SSE chunk sequence
  - Multi-turn context retention
  - Single-turn tool calling
  - Tool result continuation (two-turn tool use cycle)
  - Full agentic loop with real local tool execution (multi-turn, multi-tool)
  - Structured JSON output

All tests require actual Bedrock access and are therefore marked
``@pytest.mark.expensive``.  Run with::

    pytest --expensive tests/test_openai_chat_completions_multi_model.py

A ``Claude`` model is included in every parametrized list as the reference
baseline.  Feature-gated tests (streaming+tools) carry narrower parametrize
lists reflecting known Bedrock capabilities.

If a model does not support a specific feature (e.g. Mistral or Llama3 cannot
use tools in streaming mode), the test calls ``pytest.skip()`` so the result
is recorded as *skipped* — not as a failure — in the report.
"""

import base64
import json
import struct
import zlib
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion

# ---------------------------------------------------------------------------
# Model lists — one representative per family, prefer fast/cheap variants
# ---------------------------------------------------------------------------

#: One model per provider family for basic/streaming/multi-turn tests.
_BASIC_MODELS = pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-sonnet-4-6",  # Claude (reference)
        "amazon.nova-micro-v1:0",  # Amazon Nova (cheapest)
        # "ai21.jamba-1-5-mini-v1:0",  # AI21 Jamba (SSM/Transformer hybrid, 256k ctx)
        "deepseek.v3-v1:0",  # DeepSeek V3 (fast non-reasoning)
        "google.gemma-3-12b-it",  # Google Gemma
        "meta.llama3-3-70b-instruct-v1:0",  # Meta Llama
        "minimax.minimax-m2.5",  # MiniMax
        "mistral.mistral-7b-instruct-v0:2",  # Mistral (cheapest)
        "mistral.mistral-large-2402-v1:0",  # Mistral Large
        pytest.param(
            "mistral.pixtral-large-2502-v1:0",
            marks=pytest.mark.xfail(
                strict=False,
                reason="Pixtral non-deterministically misidentifies colour of 1x1 PNG",
            ),
        ),  # Mistral Pixtral Large (vision)
        "moonshotai.kimi-k2.5",  # Moonshot Kimi K2.5
        "nvidia.nemotron-nano-3-30b",  # NVIDIA Nemotron Nano 30B
        "qwen.qwen3-32b-v1:0",  # Qwen3 32B
        "qwen.qwen3-vl-235b-a22b",  # Qwen3 VL (vision)
        "writer.palmyra-vision-7b",  # Writer Palmyra Vision
        "writer.palmyra-x5-v1:0",  # Writer Palmyra X5
        "zai.glm-4.7-flash",  # Z.AI GLM-4.7 Flash
    ],
)

#: Models confirmed to support non-streaming tool use.
_TOOL_MODELS = pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-sonnet-4-6",  # Claude (reference)
        "amazon.nova-lite-v1:0",  # Amazon Nova
        "amazon.nova-2-lite-v1:0",  # Amazon Nova 2
        # "ai21.jamba-1-5-mini-v1:0",  # AI21 Jamba Mini
        # "ai21.jamba-1-5-large-v1:0",  # AI21 Jamba Large
        "deepseek.v3-v1:0",  # DeepSeek V3
        "deepseek.v3.2",  # DeepSeek V3.2 (newer revision)
        "meta.llama3-1-70b-instruct-v1:0",  # Meta Llama 3.1 70B
        "minimax.minimax-m2.5",  # MiniMax
        "mistral.mistral-large-2402-v1:0",  # Mistral Large
        "mistral.pixtral-large-2502-v1:0",  # Mistral Pixtral Large
        "moonshotai.kimi-k2.5",  # Moonshot Kimi K2.5
        pytest.param(
            "openai.gpt-oss-20b-1:0",  # OpenAI GPT-OSS 20B (Bedrock)
            marks=pytest.mark.xfail(  # type: ignore[call-overload]
                match="toolUse.*failed to satisfy constraint.*[a-zA-Z0-9_-]",
                reason="Model generates invalid tool names (fails regex [a-zA-Z0-9_-]+)",
            ),
        ),
        "openai.gpt-oss-120b-1:0",  # OpenAI GPT-OSS 120B (Bedrock)
        "qwen.qwen3-32b-v1:0",  # Qwen3 32B
        "writer.palmyra-x4-v1:0",  # Writer Palmyra X4
        "writer.palmyra-x5-v1:0",  # Writer Palmyra X5
        "zai.glm-5",  # Z.AI GLM-5
    ],
)

#: Models confirmed to support tool use in streaming mode.
#: Only Mistral 7B and Llama 3.3 70B are excluded — Bedrock returns 400 for
#: streaming+tools on those specific models.
_STREAMING_TOOL_MODELS = pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-sonnet-4-6",  # Claude (reference)
        "amazon.nova-lite-v1:0",  # Amazon Nova
        "amazon.nova-2-lite-v1:0",  # Amazon Nova 2
        # "ai21.jamba-1-5-mini-v1:0",  # AI21 Jamba Mini
        # "ai21.jamba-1-5-large-v1:0",  # AI21 Jamba Large
        "deepseek.v3-v1:0",  # DeepSeek V3
        "deepseek.v3.2",  # DeepSeek V3.2 (newer revision)
        "meta.llama3-1-70b-instruct-v1:0",  # Meta Llama 3.1 70B
        "minimax.minimax-m2.5",  # MiniMax
        "mistral.mistral-large-2402-v1:0",  # Mistral Large
        "mistral.pixtral-large-2502-v1:0",  # Mistral Pixtral Large
        "moonshotai.kimi-k2.5",  # Moonshot Kimi K2.5
        pytest.param(
            "openai.gpt-oss-20b-1:0",  # OpenAI GPT-OSS 20B (Bedrock)
            marks=pytest.mark.xfail(  # type: ignore[call-overload]
                match="toolUse.*failed to satisfy constraint.*[a-zA-Z0-9_-]",
                reason="Model generates invalid tool names (fails regex [a-zA-Z0-9_-]+)",
            ),
        ),
        "openai.gpt-oss-120b-1:0",  # OpenAI GPT-OSS 120B (Bedrock)
        "qwen.qwen3-32b-v1:0",  # Qwen3 32B
        "writer.palmyra-x4-v1:0",  # Writer Palmyra X4
        "writer.palmyra-x5-v1:0",  # Writer Palmyra X5
        "zai.glm-5",  # Z.AI GLM-5
    ],
)

#: Models for the full agentic loop (excludes llama3-3-70b which outputs raw JSON
#: instead of using the Converse native tool_use block format).
_AGENTIC_MODELS = pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-sonnet-4-6",  # Claude (reference)
        "amazon.nova-lite-v1:0",  # Amazon Nova
        "amazon.nova-2-lite-v1:0",  # Amazon Nova 2
        # "ai21.jamba-1-5-mini-v1:0",  # AI21 Jamba Mini
        # "ai21.jamba-1-5-large-v1:0",  # AI21 Jamba Large
        "deepseek.v3-v1:0",  # DeepSeek V3
        "deepseek.v3.2",  # DeepSeek V3.2 (newer revision)
        "meta.llama3-1-70b-instruct-v1:0",  # Meta Llama 3.1 70B
        "minimax.minimax-m2.5",  # MiniMax
        "mistral.mistral-large-2402-v1:0",  # Mistral Large
        "mistral.pixtral-large-2502-v1:0",  # Mistral Pixtral Large
        "moonshotai.kimi-k2.5",  # Moonshot Kimi K2.5
        pytest.param(
            "openai.gpt-oss-20b-1:0",  # OpenAI GPT-OSS 20B (Bedrock)
            marks=pytest.mark.xfail(  # type: ignore[call-overload]
                match="toolUse.*failed to satisfy constraint.*[a-zA-Z0-9_-]",
                reason="Model generates invalid tool names (fails regex [a-zA-Z0-9_-]+)",
            ),
        ),
        "openai.gpt-oss-120b-1:0",  # OpenAI GPT-OSS 120B (Bedrock)
        "qwen.qwen3-32b-v1:0",  # Qwen3 32B
        "writer.palmyra-x4-v1:0",  # Writer Palmyra X4
        "writer.palmyra-x5-v1:0",  # Writer Palmyra X5
        "zai.glm-5",  # Z.AI GLM-5
    ],
)

# ---------------------------------------------------------------------------
# Shared tool definitions and local tool executor
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, object]] = [
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": "List the files and directories inside a filesystem path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to list"}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the text contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute path to read"}
                },
                "required": ["path"],
            },
        },
    },
]

_PROJECT_ROOT = "/var/opt/projects/stdapi.ai"


def _run_tool(name: str, arguments_json: str) -> str:
    """Execute a local read-only tool and return the string result.

    Args:
        name: Tool name (``list_directory`` or ``read_file``).
        arguments_json: JSON-encoded arguments string from the tool call.

    Returns:
        String result to feed back as tool message content.
    """
    try:
        args: dict[str, str] = json.loads(arguments_json)
    except json.JSONDecodeError:
        return f"Invalid arguments JSON: {arguments_json!r}"

    if name == "list_directory":
        try:
            entries = sorted(
                p.name for p in Path(args.get("path", _PROJECT_ROOT)).iterdir()
            )[:40]
            return "\n".join(entries)
        except OSError as exc:
            return f"Error: {exc}"

    if name == "read_file":
        try:
            return Path(args.get("path", "")).read_text(encoding="utf-8")[:3000]
        except OSError as exc:
            return f"Error: {exc}"

    return f"Unknown tool: {name!r}"


def _message_text(completion: ChatCompletion) -> str:
    """Return the text content of the first assistant choice."""
    if not completion.choices:
        return ""
    return completion.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Tests: basic completion, streaming, multi-turn
# ---------------------------------------------------------------------------


class TestMultiModelChatCompletions:
    """Basic and streaming functionality across all supported model families."""

    @pytest.mark.expensive
    @_BASIC_MODELS
    def test_basic_chat_completion(
        self, model: str, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Non-streaming chat completion has assistant message and usage.

        Validates:
            - Response has at least one choice
            - First choice has ``role == "assistant"``
            - ``finish_reason`` is set
            - ``usage.prompt_tokens > 0`` and ``usage.completion_tokens > 0``
            - Content is non-empty text
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        completion = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Reply with exactly one word: HELLO"}
            ],
        )

        assert len(completion.choices) >= 1
        choice = completion.choices[0]
        assert choice.message.role == "assistant"
        assert choice.finish_reason is not None
        assert completion.usage is not None
        assert completion.usage.prompt_tokens > 0
        assert completion.usage.completion_tokens > 0

        content = choice.message.content or ""
        assert content, f"Expected non-empty content for {model!r}"

    @pytest.mark.expensive
    @_BASIC_MODELS
    def test_streaming_chat_completion(
        self, model: str, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Streaming response delivers content delta chunks and a final chunk.

        Validates:
            - At least one chunk with delta content
            - Last chunk has ``finish_reason`` set
            - Accumulated content is non-empty
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        accumulated = ""
        finish_reasons: list[str | None] = []
        delta_count = 0

        stream = openai_client.chat.completions.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            stream=True,
        )
        for chunk in stream:
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.delta.content:
                    accumulated += choice.delta.content
                    delta_count += 1
                if choice.finish_reason:
                    finish_reasons.append(choice.finish_reason)

        assert delta_count > 0, f"No content deltas received for {model!r}"
        assert accumulated, f"No accumulated content for {model!r}"
        assert finish_reasons, f"No finish_reason received for {model!r}"

    @pytest.mark.expensive
    @_BASIC_MODELS
    def test_multi_turn_context_retention(
        self, model: str, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Model correctly uses conversation history in multi-turn dialogue.

        Validates:
            - Third turn response references information shared in the first turn
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        completion = openai_client.chat.completions.create(
            model=model,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": "The test identifier for this session is ZEBRA99.",
                },
                {
                    "role": "assistant",
                    "content": "Understood, the test identifier is ZEBRA99.",
                },
                {
                    "role": "user",
                    "content": "What is the test identifier for this session?",
                },
            ],
        )

        text = _message_text(completion)
        assert text, "Expected non-empty response"
        assert "ZEBRA99" in text, (
            f"Expected test identifier in response for {model!r}, got: {text[:200]!r}"
        )


# ---------------------------------------------------------------------------
# Tests: tool use (single turn, continuation, streaming, agentic loop)
# ---------------------------------------------------------------------------


class TestMultiModelToolUse:
    """Tool-calling functionality across tool-capable model families."""

    @pytest.mark.expensive
    @_TOOL_MODELS
    def test_tool_call_single_turn(
        self, model: str, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Model invokes a tool when asked to perform a task requiring it.

        Validates:
            - ``finish_reason == "tool_calls"``
            - At least one tool call in ``message.tool_calls``
            - Tool name matches a defined tool
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        completion = openai_client.chat.completions.create(
            model=model,
            max_tokens=512,
            tools=_TOOLS,  # type: ignore[arg-type]
            messages=[
                {"role": "user", "content": f"List the files in {_PROJECT_ROOT}"}
            ],
        )

        assert len(completion.choices) >= 1
        choice = completion.choices[0]
        assert choice.finish_reason == "tool_calls", (
            f"Expected finish_reason='tool_calls', got {choice.finish_reason!r}; "
            f"content: {choice.message.content!r}"
        )
        assert choice.message.tool_calls, "Expected at least one tool call"

        tool_names = {
            t["function"]["name"]  # type: ignore[index]
            for t in _TOOLS
        }
        called_name = choice.message.tool_calls[0].function.name  # type: ignore[union-attr]
        assert called_name in tool_names, f"Unexpected tool name: {called_name!r}"

    @pytest.mark.expensive
    @_TOOL_MODELS
    def test_tool_result_continuation(
        self, model: str, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Two-turn tool-use cycle: model calls tool, receives result, gives answer.

        Validates:
            - First turn: ``finish_reason == "tool_calls"``
            - Second turn (after injecting tool result): ``finish_reason == "stop"``
            - Second turn response contains non-empty text
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        tools = [_TOOLS[0]]  # list_directory only

        # Turn 1: model decides to call the tool
        resp1 = openai_client.chat.completions.create(
            model=model,
            max_tokens=2048,
            tools=tools,  # type: ignore[arg-type]
            messages=[
                {"role": "user", "content": f"What files are in {_PROJECT_ROOT}?"}
            ],
        )
        assert resp1.choices, "Turn 1: no choices"
        choice1 = resp1.choices[0]
        assert choice1.finish_reason == "tool_calls", (
            f"Turn 1: expected finish_reason='tool_calls', got {choice1.finish_reason!r}"
        )
        assert choice1.message.tool_calls, "Turn 1: no tool calls"
        tc = choice1.message.tool_calls[0]

        # Execute the tool locally
        tool_result = _run_tool(tc.function.name, tc.function.arguments)  # type: ignore[union-attr]

        # Turn 2: provide tool result
        resp2 = openai_client.chat.completions.create(
            model=model,
            max_tokens=2048,
            tools=tools,  # type: ignore[arg-type]
            messages=[
                {"role": "user", "content": f"What files are in {_PROJECT_ROOT}?"},
                choice1.message,  # type: ignore[list-item]
                {"role": "tool", "tool_call_id": tc.id, "content": tool_result},
            ],
        )

        assert resp2.choices, "Turn 2: no choices"
        choice2 = resp2.choices[0]
        assert choice2.finish_reason == "stop", (
            f"Turn 2: expected finish_reason='stop', got {choice2.finish_reason!r}"
        )
        assert choice2.message.content, "Turn 2: expected non-empty text response"

    @pytest.mark.expensive
    @_STREAMING_TOOL_MODELS
    def test_streaming_tool_call(
        self, model: str, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Streaming response emits tool_call delta chunks.

        Validates:
            - At least one chunk with ``delta.tool_calls``
            - Final chunk has ``finish_reason == "tool_calls"``
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        tool_call_chunks = 0
        finish_reasons: list[str | None] = []

        try:
            stream = openai_client.chat.completions.create(
                model=model,
                max_tokens=512,
                tools=_TOOLS,  # type: ignore[arg-type]
                messages=[
                    {"role": "user", "content": f"List the files in {_PROJECT_ROOT}"}
                ],
                stream=True,
            )
            for chunk in stream:
                if chunk.choices:  # type: ignore[union-attr]
                    c = chunk.choices[0]  # type: ignore[union-attr]
                    if c.delta.tool_calls:
                        tool_call_chunks += 1
                    if c.finish_reason:
                        finish_reasons.append(c.finish_reason)
        except BadRequestError as exc:
            if "streaming mode" in str(exc).lower():
                pytest.skip(f"Model does not support streaming with tools: {exc}")
            raise

        assert tool_call_chunks > 0, (
            f"Expected tool_call delta chunks for {model!r}, got 0"
        )
        assert "tool_calls" in finish_reasons, (
            f"Expected finish_reason='tool_calls', got: {finish_reasons}"
        )

    @pytest.mark.expensive
    @_AGENTIC_MODELS
    def test_agentic_loop_directory_and_file(
        self, model: str, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Full agentic loop: model lists directory then reads pyproject.toml.

        Simulates a realistic "code assistant" workflow where the model must
        use two different tools across multiple turns to answer a question.

        Validates:
            - Model calls at least two tools in the loop
            - Final answer (``finish_reason == "stop"``) contains the project name
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        messages: list[object] = [
            {
                "role": "user",
                "content": (
                    f"Use tools to: 1) list {_PROJECT_ROOT}, "
                    "2) read pyproject.toml to find the project name, "
                    "3) report what you found."
                ),
            }
        ]

        tools_used: list[str] = []
        final_text = ""

        for _ in range(8):
            response = openai_client.chat.completions.create(
                model=model,
                max_tokens=4096,
                tools=_TOOLS,  # type: ignore[arg-type]
                messages=messages,  # type: ignore[arg-type]
            )
            assert response.choices
            choice = response.choices[0]

            # Always append assistant message to history
            messages.append(choice.message)

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    tools_used.append(tc.function.name)  # type: ignore[union-attr]
                    result = _run_tool(tc.function.name, tc.function.arguments)  # type: ignore[union-attr]
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result}
                    )
            else:
                final_text = choice.message.content or ""
                break

        assert len(tools_used) >= 2, (
            f"Expected ≥2 tool calls in agentic loop, got: {tools_used}"
        )
        assert final_text, "Expected non-empty final answer from model"
        assert "stdapi" in final_text.lower(), (
            f"Expected project name 'stdapi' in final answer, got: {final_text[:300]!r}"
        )


# ---------------------------------------------------------------------------
# Tests: structured JSON output
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    """Models can reliably produce JSON-structured output when asked."""

    @pytest.mark.expensive
    @pytest.mark.parametrize(
        "model",
        [
            "anthropic.claude-sonnet-4-6",
            "amazon.nova-lite-v1:0",
            "deepseek.v3-v1:0",
            "minimax.minimax-m2.5",
        ],
    )
    def test_json_output_parseable(
        self, model: str, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Model produces valid JSON when the system prompt requests it.

        Validates:
            - Response text can be parsed as JSON
            - JSON object contains at least one expected key
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        completion = openai_client.chat.completions.create(
            model=model,
            max_tokens=256,
            messages=[
                {
                    "role": "system",
                    "content": "You are a JSON API. Respond ONLY with a valid JSON object, no other text.",
                },
                {
                    "role": "user",
                    "content": 'Return a JSON object with keys "language" (value: "Python") and "version" (value: "3.12").',
                },
            ],
        )

        raw = _message_text(completion)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(
                line for line in lines if not line.strip().startswith("```")
            )

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            pytest.fail(
                f"Response is not valid JSON for {model!r}: {exc}\nRaw: {raw!r}"
            )

        assert isinstance(data, dict), f"Expected JSON object, got {type(data)}"
        assert "language" in data or "version" in data, (
            f"Expected 'language' or 'version' key in JSON, got: {data}"
        )


# ---------------------------------------------------------------------------
# Tests: vision / image input
# ---------------------------------------------------------------------------


def _make_1x1_red_png_b64() -> str:
    """Return a base64-encoded minimal valid 1x1 red PNG."""

    def _chunk(name: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        return length + name + data + crc

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = _chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
    iend = _chunk(b"IEND", b"")
    return base64.b64encode(signature + ihdr + idat + iend).decode()


#: Vision-capable models tested on the OpenAI route.
_VISION_MODELS = pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-sonnet-4-6",  # Claude (reference)
        "amazon.nova-lite-v1:0",  # Amazon Nova
        pytest.param(
            "mistral.pixtral-large-2502-v1:0",
            marks=pytest.mark.xfail(
                strict=False,
                reason="Pixtral non-deterministically misidentifies colour of 1x1 PNG",
            ),
        ),  # Mistral Pixtral Large
        "qwen.qwen3-vl-235b-a22b",  # Qwen3 VL 235B
        "writer.palmyra-vision-7b",  # Writer Palmyra Vision 7B
    ],
)


class TestVision:
    """Vision-capable models correctly identify the color of a simple image."""

    @pytest.mark.expensive
    @_VISION_MODELS
    def test_image_color_recognition(
        self, model: str, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Vision model identifies the color of a 1x1 red PNG via image_url.

        Uses a locally generated minimal PNG encoded as a data URI.

        Validates:
            - Response contains non-empty text
            - Response correctly identifies "red" as the image color
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        completion = openai_client.chat.completions.create(
            model=model,
            max_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{_make_1x1_red_png_b64()}"
                            },
                        },
                        {
                            "type": "text",
                            "text": "What is the color of this image? Reply in one word.",
                        },
                    ],
                }
            ],
        )

        text = _message_text(completion)
        assert text, f"Expected non-empty response from {model!r}"
        assert any(color in text.lower() for color in ("red", "orange")), (
            f"Expected 'red' or 'orange' in response for {model!r}, got: {text!r}"
        )
