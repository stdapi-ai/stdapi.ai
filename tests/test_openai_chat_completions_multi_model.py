"""OpenAI /v1/chat/completions exercised across one model per Bedrock provider family.

The gateway funnels every family through the same Converse adapter, so these tests are
the cross-model regression net for the shared envelope: response/chunk shape, the
Bedrock ``stopReason`` → ``finish_reason`` map, usage accounting, tool-call plumbing and
image input.  A Claude model heads every parametrize list as the reference baseline, and the
tool roster is narrower than the basic one because Mistral 7B and Llama 3.3 70B reject
streaming with tools and Llama 3.3 70B emits raw JSON instead of ``toolUse`` blocks.

Model-specific behaviour lives in the per-family modules; only vendor-neutral behaviour
is asserted here, because live model text is not reproducible.

Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
     stdapi/routes/openai_chat_completions.py:create_chat_completion
     stdapi/models/chat/_adapters/_openai_chat_completion.py:format_response
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError

from tests._helpers import red_png_b64
from tests._multi_model import VISION_MODELS_OPENAI, with_marks
from tests.conftest import REPO_ROOT

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
        "anthropic.claude-haiku-4-5-20251001-v1:0",  # Claude (reference)
        "amazon.nova-micro-v1:0",  # Amazon Nova (cheapest)
        # "ai21.jamba-1-5-mini-v1:0",  # AI21 Jamba (SSM/Transformer hybrid, 256k ctx)
        "deepseek.v3-v1:0",  # DeepSeek V3 (fast non-reasoning)
        "google.gemma-3-12b-it",  # Google Gemma
        "meta.llama3-3-70b-instruct-v1:0",  # Meta Llama
        "minimax.minimax-m2.5",  # MiniMax
        "mistral.mistral-7b-instruct-v0:2",  # Mistral (cheapest)
        "mistral.mistral-large-3-675b-instruct",  # Mistral Large 3
        "moonshotai.kimi-k2.5",  # Moonshot Kimi K2.5
        "nvidia.nemotron-nano-3-30b",  # NVIDIA Nemotron Nano 30B
        "qwen.qwen3-32b-v1:0",  # Qwen3 32B
        "qwen.qwen3-vl-235b-a22b",  # Qwen3 VL (vision)
        "writer.palmyra-vision-7b",  # Writer Palmyra Vision
        "writer.palmyra-x5-v1:0",  # Writer Palmyra X5
        "zai.glm-4.7-flash",  # Z.AI GLM-4.7 Flash
    ],
)

#: Models confirmed to support tool use, in both buffered and streaming mode.
#: One roster drives the single-turn, continuation, streaming and agentic tool tests:
#: Mistral 7B and Llama 3.3 70B are absent because Bedrock returns 400 for
#: streaming+tools on them and Llama 3.3 70B emits raw JSON instead of ``toolUse``.
_TOOL_MODELS = pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-haiku-4-5-20251001-v1:0",  # Claude (reference)
        "amazon.nova-lite-v1:0",  # Amazon Nova
        "amazon.nova-2-lite-v1:0",  # Amazon Nova 2
        # "ai21.jamba-1-5-mini-v1:0",  # AI21 Jamba Mini
        # "ai21.jamba-1-5-large-v1:0",  # AI21 Jamba Large
        "deepseek.v3-v1:0",  # DeepSeek V3
        "deepseek.v3.2",  # DeepSeek V3.2 (newer revision)
        "meta.llama3-1-70b-instruct-v1:0",  # Meta Llama 3.1 70B
        "minimax.minimax-m2.5",  # MiniMax
        "mistral.mistral-large-3-675b-instruct",  # Mistral Large 3
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

#: Alias kept for readability at the streaming tool-use tests.
_STREAMING_TOOL_MODELS = _TOOL_MODELS

#: Alias kept for readability at the agentic tool-loop test.
_AGENTIC_MODELS = _TOOL_MODELS

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

#: Directory the tool tests ask the models to list and read from.
_PROJECT_ROOT = str(REPO_ROOT)

#: finish_reason values the OpenAI Chat Completions reference defines.
_FINISH_REASONS = frozenset({"stop", "length", "content_filter", "tool_calls"})

#: Tool names declared in ``_TOOLS``.
_TOOL_NAMES = frozenset({"list_directory", "read_file"})


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


@pytest.fixture(autouse=True)
def _skip_official_api(use_official_api: bool) -> None:
    """Skip every test in this module when a remote official API is targeted.

    The matrix is keyed by Bedrock model ids, which the official OpenAI API does
    not serve, so no test here has a meaningful remote counterpart.
    """
    if use_official_api:
        pytest.skip("Multi-model tests only run against the local server")


# ---------------------------------------------------------------------------
# Tests: basic completion, streaming, multi-turn
# ---------------------------------------------------------------------------


class TestMultiModelChatCompletions:
    """Basic and streaming functionality across all supported model families.

    Ref: https://developers.openai.com/api/reference/resources/chat.md
         stdapi/models/chat/_default.py:ChatModel.create_completion
    """

    @pytest.mark.expensive
    @_BASIC_MODELS
    def test_basic_chat_completion(self, model: str, openai_client: OpenAI) -> None:
        """Every family answers with the same ``chat.completion`` envelope.

        The id prefix, the ``chat.completion`` literal and ``total_tokens`` as the sum of
        the two counters are produced by the gateway, not the model, so they must hold
        identically for all families.  The prompt is pinned to one word so the text can
        be checked without depending on style.

        Ref: https://developers.openai.com/api/reference/resources/chat.md
             stdapi/models/chat/_adapters/_openai_chat_completion.py:format_response
        """
        completion = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Reply with exactly one word: HELLO"}
            ],
        )

        assert completion.object == "chat.completion"
        assert completion.id.startswith("chatcmpl-")
        assert completion.created > 0
        assert len(completion.choices) >= 1
        choice = completion.choices[0]
        assert choice.index == 0
        assert choice.message.role == "assistant"
        assert choice.finish_reason in _FINISH_REASONS, (
            f"Unexpected finish_reason for {model!r}: {choice.finish_reason!r}"
        )
        assert completion.usage is not None
        assert completion.usage.prompt_tokens > 0
        assert completion.usage.completion_tokens > 0
        assert (
            completion.usage.total_tokens
            == completion.usage.prompt_tokens + completion.usage.completion_tokens
        )

        content = choice.message.content or ""
        assert content, f"Expected non-empty content for {model!r}"
        assert "hello" in content.lower(), (
            f"Expected the pinned word for {model!r}, got: {content[:200]!r}"
        )

    @pytest.mark.expensive
    @_BASIC_MODELS
    def test_streaming_chat_completion(self, model: str, openai_client: OpenAI) -> None:
        """Streaming delivers a role chunk, text deltas and exactly one stop chunk.

        Chat Completions has a single SSE event type, ``chat.completion.chunk``.  The
        gateway prepends a synthetic ``delta={"role": "assistant"}`` chunk, emits one
        finish-reason chunk, and — without ``stream_options.include_usage`` — never
        attaches usage.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
             stdapi/models/chat/_adapters/_openai_chat_completion.py:format_stream
        """
        accumulated = ""
        finish_reasons: list[str | None] = []
        delta_count = 0
        first_delta_role: str | None = None

        stream = openai_client.chat.completions.create(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": "Count from 1 to 5."}],
            stream=True,
        )
        for index, chunk in enumerate(stream):
            assert chunk.object == "chat.completion.chunk"
            assert chunk.usage is None, (
                "usage must only be streamed when stream_options.include_usage is set"
            )
            if index == 0 and chunk.choices:
                first_delta_role = chunk.choices[0].delta.role
            if chunk.choices:
                choice = chunk.choices[0]
                if choice.delta.content:
                    accumulated += choice.delta.content
                    delta_count += 1
                if choice.finish_reason:
                    finish_reasons.append(choice.finish_reason)

        assert first_delta_role == "assistant", (
            f"Stream for {model!r} must open with the synthetic role-only chunk"
        )
        assert delta_count > 0, f"No content deltas received for {model!r}"
        assert accumulated, f"No accumulated content for {model!r}"
        assert len(finish_reasons) == 1, (
            f"Expected exactly one finish chunk for {model!r}, got: {finish_reasons}"
        )
        assert finish_reasons[0] in _FINISH_REASONS

    @pytest.mark.expensive
    @_BASIC_MODELS
    def test_multi_turn_context_retention(
        self, model: str, openai_client: OpenAI
    ) -> None:
        """Prior turns reach the model, so a first-turn identifier can be recalled.

        ``map_messages`` translates the alternating user/assistant history into Bedrock
        ``messages``, merging adjacent same-role turns; a dropped or reordered history
        would leave the model unable to echo ``ZEBRA99``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/models/chat/_adapters/_openai_chat_completion.py:map_messages
        """
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

        assert completion.choices[0].finish_reason in _FINISH_REASONS
        assert completion.usage is not None
        assert completion.usage.prompt_tokens > 0
        text = _message_text(completion)
        assert text, "Expected non-empty response"
        assert "ZEBRA99" in text, (
            f"Expected test identifier in response for {model!r}, got: {text[:200]!r}"
        )


# ---------------------------------------------------------------------------
# Tests: tool use (single turn, continuation, streaming, agentic loop)
# ---------------------------------------------------------------------------


class TestMultiModelToolUse:
    """Tool-calling functionality across tool-capable model families.

    Ref: https://developers.openai.com/api/docs/guides/function-calling
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
         stdapi/models/chat/_adapters/_openai_chat_completion.py:build_tool_config
    """

    @pytest.mark.expensive
    @_TOOL_MODELS
    def test_tool_call_single_turn(self, model: str, openai_client: OpenAI) -> None:
        """A ``toolUse`` answer becomes a ``tool_calls`` message with JSON arguments.

        Bedrock's ``tool_use`` stop reason maps to ``finish_reason="tool_calls"``, the
        Bedrock ``toolUseId`` becomes the OpenAI call id, and the structured ``input`` is
        re-serialized to the ``arguments`` JSON string the OpenAI contract requires.

        Ref: https://developers.openai.com/api/docs/guides/function-calling
             stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_tool_calls
        """
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

        call = choice.message.tool_calls[0]
        assert call.type == "function"
        assert call.id, "Expected a tool call id derived from the Bedrock toolUseId"
        called_name = call.function.name
        assert called_name in _TOOL_NAMES, f"Unexpected tool name: {called_name!r}"
        arguments = json.loads(call.function.arguments)
        assert isinstance(arguments, dict), (
            f"Expected a JSON object in arguments, got: {call.function.arguments!r}"
        )

    @pytest.mark.expensive
    @_TOOL_MODELS
    def test_tool_result_continuation(self, model: str, openai_client: OpenAI) -> None:
        """A ``tool`` message closes the tool cycle and the model answers from it.

        The second request replays the assistant ``tool_calls`` message and a ``tool``
        message keyed by ``tool_call_id``; ``_extract_tool_blocks`` turns those into
        Bedrock ``toolUse``/``toolResult`` blocks, which Converse rejects if the pairing
        is lost.

        Ref: https://developers.openai.com/api/docs/guides/function-calling
             stdapi/models/chat/_adapters/_openai_chat_completion.py:_extract_tool_blocks
        """
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
        assert choice2.message.tool_calls is None, (
            f"Turn 2: expected no further tool call, got {choice2.message.tool_calls}"
        )
        assert resp2.usage is not None
        assert resp2.usage.prompt_tokens > 0, (
            "Turn 2 must bill the replayed tool result"
        )

    @pytest.mark.expensive
    @_STREAMING_TOOL_MODELS
    def test_streaming_tool_call(self, model: str, openai_client: OpenAI) -> None:
        """Streaming tool calls arrive as deltas indexed from zero, then a stop chunk.

        Bedrock content-block indices are remapped to contiguous OpenAI
        ``tool_calls[].index`` positions, and the first delta of a call carries its id and
        name while later deltas carry only argument fragments.  Bedrock refuses streaming
        with tools on a few models; those raise a 400 mentioning streaming mode and are
        skipped rather than failed.

        Ref: https://developers.openai.com/api/docs/guides/function-calling
             stdapi/models/chat/_adapters/_openai_chat_completion.py:_stream_delta_chunk
        """
        tool_call_chunks = 0
        finish_reasons: list[str | None] = []
        first_call_index: int | None = None
        first_call_name: str | None = None
        first_call_id: str | None = None

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
                        if tool_call_chunks == 0:
                            first = c.delta.tool_calls[0]
                            first_call_index = first.index
                            first_call_id = first.id
                            first_call_name = (
                                first.function.name if first.function else None
                            )
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
        assert first_call_index == 0, (
            f"First tool call must be remapped to index 0, got {first_call_index!r}"
        )
        assert first_call_id, "First tool_call delta must carry the call id"
        assert first_call_name in _TOOL_NAMES, (
            f"First tool_call delta must name a declared tool, got {first_call_name!r}"
        )
        assert "tool_calls" in finish_reasons, (
            f"Expected finish_reason='tool_calls', got: {finish_reasons}"
        )

    @pytest.mark.expensive
    @pytest.mark.agentic
    @_AGENTIC_MODELS
    def test_agentic_loop_directory_and_file(
        self, model: str, openai_client: OpenAI
    ) -> None:
        """A multi-turn loop with two distinct tools reaches a grounded final answer.

        Each iteration replays the whole growing history — assistant ``tool_calls``
        messages interleaved with ``tool`` results — so this is the cross-model check
        that repeated tool round-trips stay accepted by Converse.  ``read_file`` must be
        among the calls: the project name lives in ``pyproject.toml``, and a directory
        listing alone would let a model guess it.

        Ref: https://developers.openai.com/api/docs/guides/function-calling
             stdapi/models/chat/_adapters/_openai_chat_completion.py:map_messages
        """
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
        assert "read_file" in tools_used, (
            f"Expected the model to read pyproject.toml, tools used: {tools_used}"
        )
        assert final_text, "Expected non-empty final answer from model"
        assert "stdapi" in final_text.lower(), (
            f"Expected project name 'stdapi' in final answer, got: {final_text[:300]!r}"
        )


# ---------------------------------------------------------------------------
# Tests: structured JSON output
# ---------------------------------------------------------------------------


class TestStructuredOutput:
    """Models can reliably produce JSON-structured output when asked.

    Ref: https://developers.openai.com/api/docs/guides/structured-outputs
         stdapi/models/chat/_adapters/_openai_chat_completion.py:build_output_config
    """

    @pytest.mark.expensive
    @pytest.mark.parametrize(
        "model",
        [
            "anthropic.claude-haiku-4-5-20251001-v1:0",
            "amazon.nova-lite-v1:0",
            "deepseek.v3-v1:0",
            "minimax.minimax-m2.5",
        ],
    )
    def test_json_output_parseable(self, model: str, openai_client: OpenAI) -> None:
        """Prompted JSON output round-trips through the gateway unaltered.

        No ``response_format`` is sent: the JSON contract here is prompt-only, which is
        the fallback for families whose Bedrock profile has no ``outputConfig`` support.
        The point of the test is that the adapter returns the model text verbatim —
        Markdown fences are the models' own, so they are stripped before parsing.

        Ref: https://developers.openai.com/api/docs/guides/structured-outputs
             stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_output_text
        """
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

        assert completion.choices[0].finish_reason in _FINISH_REASONS
        assert isinstance(data, dict), f"Expected JSON object, got {type(data)}"
        assert "language" in data or "version" in data, (
            f"Expected 'language' or 'version' key in JSON, got: {data}"
        )
        assert str(data.get("language", "Python")).lower() == "python", (
            f"Expected the pinned 'language' value for {model!r}, got: {data}"
        )


# ---------------------------------------------------------------------------
# Tests: vision / image input
# ---------------------------------------------------------------------------


#: Vision-capable models tested on the OpenAI route.
_VISION_MODELS = pytest.mark.parametrize(
    "model",
    with_marks(
        VISION_MODELS_OPENAI,
        {
            "mistral.pixtral-large-2502-v1:0": pytest.mark.xfail(
                strict=False,
                reason="Pixtral non-deterministically misidentifies colour of 1x1 PNG",
            )
        },
    ),
)


class TestVision:
    """Vision-capable models correctly identify the color of a simple image.

    Ref: https://developers.openai.com/api/reference/resources/chat.md
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
    """

    @pytest.mark.expensive
    @_VISION_MODELS
    def test_image_color_recognition(self, model: str, openai_client: OpenAI) -> None:
        """A base64 ``image_url`` data URI reaches vision models as a Bedrock image block.

        ``_convert_content_part`` decodes the data URI and emits a Converse ``image``
        block with the format taken from the MIME type; the picture is a locally built
        1x1 red PNG so the expected answer needs no fixture.  "orange" is accepted
        because models disagree on naming a single saturated pixel.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
             https://docs.aws.amazon.com/nova/latest/userguide/modalities-image.html
             stdapi/models/chat/_adapters/_openai_chat_completion.py:_convert_content_part
        """
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
                                "url": f"data:image/png;base64,{red_png_b64()}"
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

        assert completion.choices[0].finish_reason in _FINISH_REASONS
        assert completion.usage is not None
        assert completion.usage.prompt_tokens > 0, (
            "image input must be billed as prompt tokens"
        )
        text = _message_text(completion)
        assert text, f"Expected non-empty response from {model!r}"
        assert any(color in text.lower() for color in ("red", "orange")), (
            f"Expected 'red' or 'orange' in response for {model!r}, got: {text!r}"
        )
