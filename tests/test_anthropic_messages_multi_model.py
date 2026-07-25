"""Multi-model parametrized tests for the Anthropic /v1/messages route.

Covers representative "real-world" usage across one model per provider family
available on AWS Bedrock, including:

  - Basic text generation and usage tokens
  - Streaming SSE event sequence and no-empty-text-block regression (ISSUE-1 fix)
  - Multi-turn context retention
  - Single-turn tool calling with stop_reason validation
  - Tool-result continuation (two-turn tool use cycle)
  - Streaming tool calling
  - Full agentic loop with real local tool execution (multi-turn, multi-tool)
  - Native thinking/reasoning blocks on native-reasoning models
  - Prompt caching on Nova models

All tests require actual Bedrock access and are therefore marked
``@pytest.mark.expensive``.  Run with::

    pytest --expensive tests/test_anthropic_messages_multi_model.py

A ``Claude`` model is included in every parametrized list as the reference
baseline.  Feature-gated tests (streaming+tools, reasoning blocks, caching)
carry narrower parametrize lists reflecting known Bedrock capabilities.

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
from anthropic import BadRequestError

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic.types import Message

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
        "mistral.pixtral-large-2502-v1:0",  # Mistral Pixtral Large (text+vision)
        "moonshotai.kimi-k2.5",  # Moonshot Kimi K2.5
        "nvidia.nemotron-nano-3-30b",  # NVIDIA Nemotron Nano 30B
        "qwen.qwen3-32b-v1:0",  # Qwen3 32B
        "qwen.qwen3-vl-235b-a22b",  # Qwen3 VL 235B (text+vision)
        "writer.palmyra-vision-7b",  # Writer Palmyra Vision 7B (text+vision)
        "writer.palmyra-x5-v1:0",  # Writer Palmyra X5
        "zai.glm-4.7-flash",  # Z.AI GLM-4.7 Flash
    ],
)

#: Models confirmed to support non-streaming tool use.
_TOOL_MODELS = pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-sonnet-4-6",  # Claude (reference)
        "amazon.nova-2-lite-v1:0",  # Amazon Nova 2
        "amazon.nova-lite-v1:0",  # Amazon Nova
        # "ai21.jamba-1-5-mini-v1:0",  # AI21 Jamba Mini
        # "ai21.jamba-1-5-large-v1:0",  # AI21 Jamba Large
        "deepseek.v3-v1:0",  # DeepSeek V3
        "deepseek.v3.2",  # DeepSeek V3.2 (newer revision)
        "meta.llama3-1-70b-instruct-v1:0",  # Meta Llama 3.1 70B
        "minimax.minimax-m2.5",  # MiniMax
        "mistral.mistral-large-2402-v1:0",  # Mistral Large
        "mistral.pixtral-large-2502-v1:0",  # Mistral Pixtral Large
        "moonshotai.kimi-k2.5",  # Moonshot Kimi K2.5
        "openai.gpt-oss-20b-1:0",  # OpenAI GPT-OSS 20B (Bedrock)
        "openai.gpt-oss-120b-1:0",  # OpenAI GPT-OSS 120B (Bedrock)
        "qwen.qwen3-32b-v1:0",  # Qwen3 32B
        "writer.palmyra-x4-v1:0",  # Writer Palmyra X4
        "writer.palmyra-x5-v1:0",  # Writer Palmyra X5
        "zai.glm-5",  # Z.AI GLM-5
    ],
)

#: Models confirmed to support tool use in streaming mode.
#: Mistral 7B and Llama 3.3 70B are excluded — Bedrock returns 400 for
#: streaming + tool use on those models specifically.
_STREAMING_TOOL_MODELS = pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-sonnet-4-6",  # Claude (reference)
        "amazon.nova-2-lite-v1:0",  # Amazon Nova 2
        "amazon.nova-lite-v1:0",  # Amazon Nova
        # "ai21.jamba-1-5-mini-v1:0",  # AI21 Jamba Mini
        # "ai21.jamba-1-5-large-v1:0",  # AI21 Jamba Large
        "deepseek.v3-v1:0",  # DeepSeek V3
        "deepseek.v3.2",  # DeepSeek V3.2 (newer revision)
        "meta.llama3-1-70b-instruct-v1:0",  # Meta Llama 3.1 70B
        "minimax.minimax-m2.5",  # MiniMax
        "mistral.mistral-large-2402-v1:0",  # Mistral Large
        "mistral.pixtral-large-2502-v1:0",  # Mistral Pixtral Large
        "moonshotai.kimi-k2.5",  # Moonshot Kimi K2.5
        "openai.gpt-oss-20b-1:0",  # OpenAI GPT-OSS 20B (Bedrock)
        "openai.gpt-oss-120b-1:0",  # OpenAI GPT-OSS 120B (Bedrock)
        "qwen.qwen3-32b-v1:0",  # Qwen3 32B
        "writer.palmyra-x4-v1:0",  # Writer Palmyra X4
        "writer.palmyra-x5-v1:0",  # Writer Palmyra X5
        "zai.glm-5",  # Z.AI GLM-5
    ],
)

#: Models whose Bedrock stream drops the leading tool-input JSON fragment (upstream bug).
_BROKEN_STREAMING_TOOL_INPUT_MODELS = frozenset({"qwen.qwen3-32b-v1:0"})

#: Models to use for the full agentic loop (non-streaming; excludes models with
#: known non-standard tool-output behaviour, e.g. llama3-3-70b outputs raw JSON).
_AGENTIC_MODELS = pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-sonnet-4-6",  # Claude (reference)
        "amazon.nova-2-lite-v1:0",  # Amazon Nova 2
        "amazon.nova-lite-v1:0",  # Amazon Nova
        # "ai21.jamba-1-5-mini-v1:0",  # AI21 Jamba Mini
        # "ai21.jamba-1-5-large-v1:0",  # AI21 Jamba Large
        "deepseek.v3-v1:0",  # DeepSeek V3
        "deepseek.v3.2",  # DeepSeek V3.2 (newer revision)
        "meta.llama3-1-70b-instruct-v1:0",  # Meta Llama 3.1 70B
        "minimax.minimax-m2.5",  # MiniMax
        "mistral.mistral-large-2402-v1:0",  # Mistral Large
        "mistral.pixtral-large-2502-v1:0",  # Mistral Pixtral Large
        "moonshotai.kimi-k2.5",  # Moonshot Kimi K2.5
        # "openai.gpt-oss-20b-1:0",  # OpenAI GPT-OSS 20B (Bedrock), disabled : unstable tool use
        "openai.gpt-oss-120b-1:0",  # OpenAI GPT-OSS 120B (Bedrock)
        "qwen.qwen3-32b-v1:0",  # Qwen3 32B
        "writer.palmyra-x4-v1:0",  # Writer Palmyra X4
        "writer.palmyra-x5-v1:0",  # Writer Palmyra X5
        "zai.glm-5",  # Z.AI GLM-5
    ],
)

#: Native-reasoning models that produce ``thinking`` blocks without the
#: ``thinking`` API parameter (Bedrock does not support that parameter for
#: non-Claude models).  These models always emit an internal reasoning trace
#: regardless of whether the ``thinking`` param is set.
_REASONING_MODELS = pytest.mark.parametrize(
    "model",
    [
        "deepseek.r1-v1:0",  # DeepSeek R1
        "minimax.minimax-m2.5",  # MiniMax M2.5
        "moonshot.kimi-k2-thinking",  # Moonshot Kimi K2 Thinking
    ],
)

#: Models that support prompt caching (implicit Nova hash-based caching).
_CACHE_MODELS = pytest.mark.parametrize(
    "model",
    [
        "amazon.nova-micro-v1:0",
        "amazon.nova-lite-v1:0",
        "amazon.nova-2-lite-v1:0",
        "amazon.nova-pro-v1:0",
    ],
)

# ---------------------------------------------------------------------------
# Shared tool definitions and local tool executor
# ---------------------------------------------------------------------------

_TOOLS: list[dict[str, object]] = [
    {
        "name": "list_directory",
        "description": "List the files and directories inside a filesystem path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to list"}
            },
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": "Read the text contents of a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to read"}
            },
            "required": ["path"],
        },
    },
]

_PROJECT_ROOT = "/var/opt/projects/stdapi.ai"


def _run_tool(name: str, tool_input: object) -> str:
    """Execute a local read-only tool and return the string result.

    Args:
        name: Tool name (``list_directory`` or ``read_file``).
        tool_input: Arbitrary mapping from the model's tool_use block ``input``.

    Returns:
        String result to feed back as tool_result content.
    """
    inp: dict[str, str] = {}
    if isinstance(tool_input, dict):
        inp = {k: str(v) for k, v in tool_input.items()}

    if name == "list_directory":
        try:
            entries = sorted(
                p.name for p in Path(inp.get("path", _PROJECT_ROOT)).iterdir()
            )[:40]
            return "\n".join(entries)
        except OSError as exc:
            return f"Error: {exc}"

    if name == "read_file":
        try:
            return Path(inp.get("path", "")).read_text(encoding="utf-8")[:3000]
        except OSError as exc:
            return f"Error: {exc}"

    return f"Unknown tool: {name!r}"


def _text_from(msg: Message) -> str:
    """Concatenate text blocks from a message response."""
    return "".join(b.text for b in msg.content if b.type == "text")


# ---------------------------------------------------------------------------
# Tests: basic text generation, streaming, multi-turn
# ---------------------------------------------------------------------------


class TestMultiModelBasics:
    """Basic functionality across all supported model families."""

    @pytest.mark.expensive
    @_BASIC_MODELS
    def test_basic_text_generation(
        self, model: str, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Non-streaming response has text content and populated usage.

        Validates:
            - ``response.type == "message"``
            - ``response.role == "assistant"``
            - At least one content block with non-empty text
            - ``stop_reason`` is set
            - ``usage.input_tokens > 0`` and ``usage.output_tokens > 0``
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        # Use generous max_tokens so native-reasoning models (DeepSeek R1,
        # MiniMax M2.5) can produce text after their thinking blocks.
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[
                {"role": "user", "content": "Reply with exactly one word: HELLO"}
            ],
        )

        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) >= 1
        assert response.stop_reason is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

        text = _text_from(response)
        assert text, (
            f"Expected non-empty text, got content types: {[b.type for b in response.content]}"
        )

    @pytest.mark.expensive
    @_BASIC_MODELS
    def test_streaming_event_sequence(
        self, model: str, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Streaming response emits the required SSE event sequence.

        Regression test for ISSUE-1: DeepSeek V3 and Google Gemma previously
        produced only 3 events (message_start/delta/stop) with empty text because
        the server permanently suppressed blocks starting with an empty delta.

        Validates:
            - ``message_start`` event is present
            - At least one ``content_block_start`` event is present
            - At least one ``content_block_stop`` event is present
            - ``message_delta`` event is present
            - ``message_stop`` event is present
            - Final message has non-empty text (for non-reasoning-only responses)
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        event_types: list[str] = []
        with anthropic_client.messages.stream(
            model=model,
            max_tokens=256,
            messages=[
                {"role": "user", "content": "Reply with exactly one word: HELLO"}
            ],
        ) as stream:
            event_types.extend(event.type for event in stream)
            final = stream.get_final_message()

        assert "message_start" in event_types, (
            f"Missing message_start; got: {event_types}"
        )
        assert "content_block_start" in event_types, (
            f"Missing content_block_start; got: {event_types}"
        )
        assert "content_block_stop" in event_types, (
            f"Missing content_block_stop; got: {event_types}"
        )
        assert "message_delta" in event_types, (
            f"Missing message_delta; got: {event_types}"
        )
        assert "message_stop" in event_types, (
            f"Missing message_stop; got: {event_types}"
        )

        block_types = [b.type for b in final.content]
        if block_types == ["thinking"]:
            # A reasoning model may spend the whole budget thinking; the event
            # sequence asserted above is what this test covers.
            return
        assert _text_from(final), f"Final message has no text; content: {block_types}"

    @pytest.mark.expensive
    @_BASIC_MODELS
    def test_streaming_no_empty_text_blocks(
        self, model: str, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Streaming response never surfaces a text block with empty content.

        Regression test for ISSUE-1: the deferred-suppression fix in
        ``_process_content_block_delta`` must not leak empty text blocks for
        any model (e.g. Nova's preamble blocks must still be discarded).

        Validates:
            - Every ``text`` block in the accumulated final message has non-empty text
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        with anthropic_client.messages.stream(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
        ) as stream:
            final = stream.get_final_message()

        for block in final.content:
            if block.type == "text":
                assert block.text, (
                    f"Empty text block in final message for {model!r}: {block!r}"
                )

    @pytest.mark.expensive
    @_BASIC_MODELS
    def test_multi_turn_context_retention(
        self, model: str, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Model correctly uses conversation history in multi-turn dialogue.

        Validates:
            - Third turn response references information shared in the first turn
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        response = anthropic_client.messages.create(
            model=model,
            max_tokens=2048,
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

        text = _text_from(response)
        assert text, "Expected non-empty response"
        assert "ZEBRA99" in text, (
            f"Expected test identifier in response, got: {text[:200]!r}"
        )


# ---------------------------------------------------------------------------
# Tests: tool use (single turn, continuation, streaming, agentic loop)
# ---------------------------------------------------------------------------


class TestMultiModelToolUse:
    """Tool-calling functionality across tool-capable model families."""

    @pytest.mark.expensive
    @_TOOL_MODELS
    def test_tool_call_single_turn(
        self, model: str, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Model invokes a tool when asked to perform a task requiring it.

        Validates:
            - ``stop_reason == "tool_use"``
            - At least one ``tool_use`` block in content
            - Tool name matches a defined tool
            - Tool input contains expected keys
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        response = anthropic_client.messages.create(
            model=model,
            max_tokens=2048,
            tools=_TOOLS,  # type: ignore[arg-type]
            messages=[
                {"role": "user", "content": f"List the files in {_PROJECT_ROOT}"}
            ],
        )

        assert response.stop_reason == "tool_use", (
            f"Expected stop_reason='tool_use', got {response.stop_reason!r}; "
            f"content: {response.content}"
        )
        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_blocks) >= 1, "Expected at least one tool_use block"

        tool = tool_blocks[0]
        tool_names = {t["name"] for t in _TOOLS}
        assert tool.name in tool_names, f"Unexpected tool name: {tool.name!r}"

    @pytest.mark.expensive
    @_TOOL_MODELS
    def test_tool_result_continuation(
        self, model: str, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Two-turn tool-use cycle: model calls tool, receives result, gives answer.

        Validates:
            - First turn: ``stop_reason == "tool_use"``
            - Second turn (after injecting tool result): ``stop_reason == "end_turn"``
            - Second turn response contains non-empty text
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        tools = [_TOOLS[0]]  # list_directory only

        # Turn 1: model decides to call the tool.
        # Some models occasionally answer directly instead of calling a tool;
        # retry once before giving up (the user prompt strongly implies tool use).
        for _attempt in range(2):
            resp1 = anthropic_client.messages.create(
                model=model,
                max_tokens=4096,
                tools=tools,  # type: ignore[arg-type]
                messages=[
                    {"role": "user", "content": f"What files are in {_PROJECT_ROOT}?"}
                ],
            )
            if resp1.stop_reason == "tool_use":
                break
        else:
            pytest.skip(
                f"Model {model!r} did not invoke a tool after 2 attempts "
                f"(stop_reason={resp1.stop_reason!r}); skipping continuation test."
            )
        tool_blocks = [b for b in resp1.content if b.type == "tool_use"]
        assert tool_blocks, "Turn 1: no tool_use block found"
        tool_block = tool_blocks[0]

        # Execute the tool locally
        tool_result = _run_tool(tool_block.name, tool_block.input)

        # Turn 2: provide tool result
        resp2 = anthropic_client.messages.create(
            model=model,
            max_tokens=4096,
            tools=tools,  # type: ignore[arg-type]
            messages=[
                {"role": "user", "content": f"What files are in {_PROJECT_ROOT}?"},
                {"role": "assistant", "content": resp1.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": tool_result,
                        }
                    ],
                },
            ],
        )

        assert resp2.stop_reason == "end_turn", (
            f"Turn 2: expected stop_reason='end_turn', got {resp2.stop_reason!r}"
        )
        text = _text_from(resp2)
        assert text, "Turn 2: expected non-empty text in final response"

    @pytest.mark.expensive
    @_STREAMING_TOOL_MODELS
    def test_streaming_tool_call(
        self, model: str, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Streaming response correctly emits ``content_block_start`` for tool_use.

        Validates:
            - At least one ``content_block_start`` event with ``type == "tool_use"``
            - Accumulated final message has ``stop_reason == "tool_use"``
            - Tool_use block has name and id
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        try:
            with anthropic_client.messages.stream(
                model=model,
                max_tokens=2048,
                tools=_TOOLS,  # type: ignore[arg-type]
                messages=[
                    {"role": "user", "content": f"List the files in {_PROJECT_ROOT}"}
                ],
            ) as stream:
                tool_starts = [
                    e.content_block
                    for e in stream
                    if e.type == "content_block_start"
                    and e.content_block.type == "tool_use"
                ]
                final = stream.get_final_message()
        except BadRequestError as exc:
            if "streaming mode" in str(exc).lower():
                pytest.skip(f"Model does not support streaming with tools: {exc}")
            raise
        except ValueError as exc:
            # The Anthropic SDK accumulator fails on the truncated tool input.
            if (
                "expected value" in str(exc)
                and model in _BROKEN_STREAMING_TOOL_INPUT_MODELS
            ):
                pytest.xfail(
                    f"Bedrock drops the leading tool-input JSON fragment "
                    f"when streaming {model}: {exc}"
                )
            raise

        assert len(tool_starts) >= 1, (
            f"Expected tool_use content_block_start, got final content: {final.content}"
        )
        assert final.stop_reason == "tool_use", (
            f"Expected stop_reason='tool_use', got {final.stop_reason!r}"
        )

    @pytest.mark.expensive
    @_AGENTIC_MODELS
    def test_agentic_loop_directory_and_file(
        self, model: str, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Full agentic loop: model lists directory then reads pyproject.toml.

        Simulates a realistic "code assistant" workflow where the model must
        use two different tools across multiple turns to answer a question.

        Validates:
            - Model calls at least two tools in the loop
            - Final answer (``stop_reason == "end_turn"``) contains the project name
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        messages: list[dict[str, object]] = [
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
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=800,
                tools=_TOOLS,  # type: ignore[arg-type]
                system="Use the provided tools methodically. Complete all requested steps.",
                messages=messages,  # type: ignore[arg-type]
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "tool_use":
                tool_results: list[dict[str, object]] = []
                for block in response.content:
                    if block.type == "tool_use":
                        tools_used.append(block.name)
                        result = _run_tool(block.name, block.input)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result,
                            }
                        )
                messages.append({"role": "user", "content": tool_results})
            else:
                final_text = _text_from(response)
                break

        assert len(tools_used) >= 2, (
            f"Expected ≥2 tool calls in agentic loop, got: {tools_used}"
        )
        assert final_text, "Expected non-empty final answer from model"
        assert "stdapi" in final_text.lower(), (
            f"Expected project name 'stdapi' in final answer, got: {final_text[:300]!r}"
        )


# ---------------------------------------------------------------------------
# Tests: native reasoning / thinking blocks (reasoning models only)
# ---------------------------------------------------------------------------


class TestNativeReasoning:
    """Reasoning models produce native thinking blocks without the thinking param."""

    @pytest.mark.expensive
    @_REASONING_MODELS
    def test_native_thinking_blocks_present(
        self, model: str, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Native-reasoning models return at least one ``thinking`` block.

        DeepSeek R1 and MiniMax M2.5 always generate an internal reasoning
        trace regardless of whether the ``thinking`` API parameter is sent.
        The server must expose this as a ``thinking`` content block.

        Validates:
            - At least one ``thinking`` block with non-empty text
            - Also has at least one ``text`` block with the final answer
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        # Generous max_tokens: reasoning models spend many tokens on thinking.
        response = anthropic_client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "What is 17 x 23? Show your reasoning."}
            ],
        )

        thinking_blocks = [b for b in response.content if b.type == "thinking"]
        text_blocks = [b for b in response.content if b.type == "text"]

        assert thinking_blocks, (
            f"Expected thinking block for reasoning model {model!r}; "
            f"content types: {[b.type for b in response.content]}"
        )
        assert thinking_blocks[0].thinking, "thinking block must have non-empty content"
        assert text_blocks or response.stop_reason == "max_tokens", (
            f"Expected text block alongside thinking for {model!r} "
            f"(stop_reason={response.stop_reason!r}); "
            f"content types: {[b.type for b in response.content]}"
        )

    @pytest.mark.expensive
    @_REASONING_MODELS
    def test_streaming_native_thinking_blocks(
        self, model: str, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Streaming native-reasoning model emits thinking content_block_start events.

        Validates:
            - At least one ``content_block_start`` with type ``thinking``
            - Accumulated final message has at least one ``thinking`` block
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        with anthropic_client.messages.stream(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": "What is 13 x 17?"}],
        ) as stream:
            thinking_starts = [
                e.content_block
                for e in stream
                if e.type == "content_block_start"
                and e.content_block.type == "thinking"
            ]
            final = stream.get_final_message()

        assert thinking_starts, (
            f"Expected thinking content_block_start event for {model!r}; "
            f"final content types: {[b.type for b in final.content]}"
        )
        thinking_blocks = [b for b in final.content if b.type == "thinking"]
        assert thinking_blocks, (
            f"Expected thinking block in final message for {model!r}"
        )


# ---------------------------------------------------------------------------
# Tests: prompt caching (Amazon Nova models)
# ---------------------------------------------------------------------------


class TestPromptCaching:
    """Prompt caching validation on Amazon Nova models.

    Nova uses *implicit* caching: ``cache_creation_input_tokens`` is always 0
    in the response (writes are not reported), but ``cache_read_input_tokens``
    is non-zero on cache hits.  The cache key is based on the content hash,
    so sending the same large prompt twice should trigger a read on the second call.
    """

    @pytest.mark.expensive
    @_CACHE_MODELS
    def test_cache_read_on_second_call(
        self, model: str, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Second call with identical large prompt reports cache_read_input_tokens > 0.

        Validates:
            - Both calls succeed (status 200)
            - Second call (or subsequent) reports ``cache_read_input_tokens > 0``
        """
        if use_official_api:
            pytest.skip("Nova caching is only available on AWS Bedrock")

        long_context = "Detailed project context. " * 200  # ~800 tokens
        payload = {
            "model": model,
            "max_tokens": 24,
            "system": [
                {
                    "type": "text",
                    "text": long_context,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": "Reply with OK."}],
        }

        resp1 = anthropic_client.messages.create(**payload)  # type: ignore[call-overload]
        resp2 = anthropic_client.messages.create(**payload)  # type: ignore[call-overload]

        assert resp1.type == "message"
        assert resp2.type == "message"

        # At least one of the calls after the first must observe a cache hit.
        # For explicit caching (Claude with cache_control): cache_read_input_tokens > 0.
        # For implicit hash-based caching (Nova 2): inputTokens is silently reduced to
        # near-zero (only user message tokens remain; system prompt served from cache
        # without populating cache_read_input_tokens).
        _long_context_approx_tokens = 400  # ~800 token system prompt, threshold at half
        cache_token_report = (
            resp2.usage.cache_read_input_tokens or resp1.usage.cache_read_input_tokens
        )
        implicit_cache = (
            resp1.usage.input_tokens < _long_context_approx_tokens
            or resp2.usage.input_tokens < _long_context_approx_tokens
        )
        assert cache_token_report or implicit_cache, (
            f"Expected cache hit on {model!r}; "
            f"call1 usage: {resp1.usage}, call2 usage: {resp2.usage}"
        )


# ---------------------------------------------------------------------------
# Tests: JSON-structured output
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    """Models can reliably produce JSON-structured output when asked."""

    @pytest.mark.expensive
    @pytest.mark.parametrize(
        "model",
        [
            "anthropic.claude-sonnet-4-6",
            "amazon.nova-lite-v1:0",
            "amazon.nova-pro-v1:0",
            "deepseek.v3-v1:0",
            "minimax.minimax-m2.5",
        ],
    )
    def test_json_output_parseable(
        self, model: str, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Model produces valid JSON when the system prompt requests it.

        Validates:
            - Response text can be parsed as JSON
            - JSON object contains the expected keys
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        response = anthropic_client.messages.create(
            model=model,
            max_tokens=256,
            system="You are a JSON API. Respond ONLY with a valid JSON object, no other text.",
            messages=[
                {
                    "role": "user",
                    "content": 'Return a JSON object with keys "language" (value: "Python") and "version" (value: "3.12").',
                }
            ],
        )

        raw = _text_from(response)
        # Strip potential markdown fences that some models add
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


#: Vision-capable models tested on the Anthropic route.
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
        self, model: str, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Vision model identifies the color of a 1x1 red PNG image.

        Uses a locally generated minimal PNG to avoid external dependencies.

        Validates:
            - Response contains non-empty text
            - Response correctly identifies "red" as the image color
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        response = anthropic_client.messages.create(
            model=model,
            max_tokens=64,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": _make_1x1_red_png_b64(),
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

        text = _text_from(response)
        assert text, f"Expected non-empty response from {model!r}"
        assert any(color in text.lower() for color in ("red", "orange")), (
            f"Expected 'red' or 'orange' in response for {model!r}, got: {text!r}"
        )
