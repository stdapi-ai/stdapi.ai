"""Anthropic /v1/messages exercised across one model per Bedrock provider family.

Every family is funnelled through the same Converse adapter, so this module is the
cross-model regression net for the shared envelope: the ``Message`` shape, the Bedrock
``stopReason`` → ``stop_reason`` map, usage accounting, the SSE event sequence, tool-use
plumbing, native reasoning blocks, Nova prompt caching and image input.  Only
vendor-neutral behavior is asserted, because live model text is not reproducible;
per-family specifics live in the dedicated modules.

A Claude model heads every parametrize list as the reference baseline, and the narrower
lists reflect Bedrock capabilities (Mistral 7B and Llama 3.3 70B reject streaming with
tools; Llama 3.3 70B emits raw JSON instead of ``toolUse`` blocks).  Where Bedrock refuses
a combination outright the test skips rather than fails, so the report distinguishes a
missing capability from a broken mapping.

Tool-use and reasoning matrices are marked ``expensive``; the latency-bound matrices
(basics, vision, structured output, caching) are marked ``slow``.  The markers are
conjunctive, so the whole file needs::

    pytest --expensive --slow tests/test_anthropic_messages_multi_model.py

Ref: https://platform.claude.com/docs/en/api/messages
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
     stdapi/routes/anthropic_messages.py:create_message
     stdapi/models/chat/_adapters/_anthropic_message.py:format_response
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from anthropic import BadRequestError

from tests._helpers import red_png_b64
from tests.conftest import REPO_ROOT

if TYPE_CHECKING:
    from anthropic import Anthropic
    from anthropic.types import Message


@pytest.fixture(autouse=True)
def _skip_on_official_api(use_official_api: bool) -> None:
    """Skip the whole module when a remote Anthropic-compatible target is selected.

    The matrices name Bedrock model IDs, which only the local gateway serves.
    """
    if use_official_api:
        pytest.skip("Multi-model tests only run against the local server")


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

#: Models confirmed to support tool use, streaming and non-streaming alike.
_TOOL_MODEL_IDS = (
    "anthropic.claude-haiku-4-5-20251001-v1:0",  # Claude (reference)
    "amazon.nova-2-lite-v1:0",  # Amazon Nova 2
    "amazon.nova-lite-v1:0",  # Amazon Nova
    # "ai21.jamba-1-5-mini-v1:0",  # AI21 Jamba Mini
    # "ai21.jamba-1-5-large-v1:0",  # AI21 Jamba Large
    "deepseek.v3-v1:0",  # DeepSeek V3
    "deepseek.v3.2",  # DeepSeek V3.2 (newer revision)
    "meta.llama3-1-70b-instruct-v1:0",  # Meta Llama 3.1 70B
    "minimax.minimax-m2.5",  # MiniMax
    "mistral.mistral-large-3-675b-instruct",  # Mistral Large 3
    "moonshotai.kimi-k2.5",  # Moonshot Kimi K2.5
    "openai.gpt-oss-20b-1:0",  # OpenAI GPT-OSS 20B (Bedrock)
    "openai.gpt-oss-120b-1:0",  # OpenAI GPT-OSS 120B (Bedrock)
    "qwen.qwen3-32b-v1:0",  # Qwen3 32B
    "writer.palmyra-x4-v1:0",  # Writer Palmyra X4
    "writer.palmyra-x5-v1:0",  # Writer Palmyra X5
    "zai.glm-5",  # Z.AI GLM-5
)

#: Model IDs excluded from the agentic loop for non-standard tool-output behaviour.
_UNSTABLE_AGENTIC_MODEL_IDS = frozenset({"openai.gpt-oss-20b-1:0"})

#: Models confirmed to support non-streaming tool use.
_TOOL_MODELS = pytest.mark.parametrize("model", _TOOL_MODEL_IDS)

#: Models confirmed to support tool use in streaming mode.
_STREAMING_TOOL_MODELS = pytest.mark.parametrize("model", _TOOL_MODEL_IDS)

#: Models whose Bedrock stream drops the leading tool-input JSON fragment (upstream bug).
_BROKEN_STREAMING_TOOL_INPUT_MODELS = frozenset({"qwen.qwen3-32b-v1:0"})

#: Models to use for the full agentic loop (non-streaming).
_AGENTIC_MODELS = pytest.mark.parametrize(
    "model",
    [
        model_id
        for model_id in _TOOL_MODEL_IDS
        if model_id not in _UNSTABLE_AGENTIC_MODEL_IDS
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
    ["amazon.nova-micro-v1:0", "amazon.nova-lite-v1:0", "amazon.nova-2-lite-v1:0"],
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

#: Declared tool names; a model may only ever call one of these.
_TOOL_NAMES = frozenset({"list_directory", "read_file"})

#: ``stop_reason`` values the Anthropic Messages reference defines.
_STOP_REASONS = frozenset(
    {
        "end_turn",
        "max_tokens",
        "stop_sequence",
        "tool_use",
        "pause_turn",
        "refusal",
        "model_context_window_exceeded",
    }
)

#: Checkout root the model is asked to explore; must be a real readable path.
_PROJECT_ROOT = str(REPO_ROOT)


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
    """Text generation, streaming and multi-turn history across all model families.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/models/chat/_default.py:ChatModel.create_message
    """

    @pytest.mark.slow
    @_BASIC_MODELS
    def test_basic_text_generation(
        self, model: str, anthropic_client: Anthropic
    ) -> None:
        """Every family answers with the same ``message`` envelope and billed usage.

        The ``msg_`` id prefix, the ``message`` / ``assistant`` literals, the echoed model
        id and the ``stop_reason`` vocabulary come from the gateway rather than the model,
        so they must hold identically for every family.  The prompt is pinned to one word
        so the text can be checked without depending on style.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_adapters/_anthropic_message.py:format_response
        """
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
        assert response.id.startswith("msg_"), f"Unexpected id: {response.id!r}"
        assert response.model == model, "The requested model id must be echoed back"
        assert len(response.content) >= 1
        assert response.stop_reason is not None
        assert response.stop_reason in _STOP_REASONS, (
            f"Unexpected stop_reason for {model!r}: {response.stop_reason!r}"
        )
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

        text = _text_from(response)
        assert text, (
            f"Expected non-empty text, got content types: {[b.type for b in response.content]}"
        )
        assert "hello" in text.lower(), (
            f"Expected the pinned word for {model!r}, got: {text[:200]!r}"
        )

    @pytest.mark.slow
    @_BASIC_MODELS
    def test_streaming_event_sequence(
        self, model: str, anthropic_client: Anthropic
    ) -> None:
        """Streaming response emits the required SSE event sequence.

        Anthropic's taxonomy is ``message_start`` → per block ``content_block_start`` /
        deltas / ``content_block_stop`` → ``message_delta`` → ``message_stop``.  Bedrock
        sends no ``contentBlockStart`` for plain text and several families open with an
        empty delta, so the gateway has to synthesize the start event; families that only
        ever emitted ``message_start``/``delta``/``stop`` were losing their whole answer.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_adapters/_anthropic_message.py:format_stream
        """
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
        assert event_types[0] == "message_start", (
            f"Stream must open with message_start; got: {event_types[:3]}"
        )
        assert event_types[-1] == "message_stop", (
            f"Stream must end with message_stop; got: {event_types[-3:]}"
        )
        assert event_types.index("content_block_start") < event_types.index(
            "content_block_stop"
        ), f"Block events out of order; got: {event_types}"
        assert event_types.index("message_delta") < event_types.index("message_stop"), (
            f"message_delta must precede message_stop; got: {event_types}"
        )
        assert final.model == model, "The requested model id must be echoed back"

        block_types = [b.type for b in final.content]
        if block_types == ["thinking"]:
            # A reasoning model may spend the whole budget thinking; the event
            # sequence asserted above is what this test covers.
            return
        assert _text_from(final), f"Final message has no text; content: {block_types}"

    @pytest.mark.slow
    @_BASIC_MODELS
    def test_streaming_no_empty_text_blocks(
        self, model: str, anthropic_client: Anthropic
    ) -> None:
        """Streaming response never surfaces a text block with empty content.

        Suppression of an empty first delta is deferred to ``contentBlockStop``: a block
        that only ever carried ``{"text": ""}`` (Nova's preamble) is dropped, while a block
        that later receives real text is emitted.  Both halves are checked here — no empty
        text block, and still a text block for a model that answers in text.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_adapters/_anthropic_message.py:_process_content_block_delta
        """
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
        block_types = [b.type for b in final.content]
        assert "text" in block_types or block_types == ["thinking"], (
            f"Expected a text block for {model!r}; content: {block_types}"
        )

    @pytest.mark.slow
    @_BASIC_MODELS
    def test_multi_turn_context_retention(
        self, model: str, anthropic_client: Anthropic
    ) -> None:
        """Prior turns reach the model, so a first-turn identifier can be recalled.

        ``_prepare_messages_and_system`` translates the alternating user/assistant history
        into Bedrock ``messages``; a dropped, reordered or merged-away turn would leave the
        model unable to echo ``ZEBRA99``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/models/chat/_adapters/_anthropic_message.py:translate_request
        """
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

        assert response.usage.input_tokens > 0, "The replayed history must be billed"
        assert response.stop_reason in _STOP_REASONS, (
            f"Unexpected stop_reason for {model!r}: {response.stop_reason!r}"
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
    """Tool-calling functionality across tool-capable model families.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_build_tool_config
    """

    @pytest.mark.expensive
    @_TOOL_MODELS
    def test_tool_call_single_turn(
        self, model: str, anthropic_client: Anthropic
    ) -> None:
        """A Bedrock ``toolUse`` answer becomes a ``tool_use`` block and ``stop_reason``.

        The Bedrock ``tool_use`` stop reason maps onto Anthropic's, the ``toolUseId`` is
        re-prefixed ``toolu_`` and the structured ``input`` is passed through as an object —
        all three come from the gateway, so they hold for every tool-capable family.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             stdapi/models/chat/_adapters/_anthropic_message.py:format_response
        """
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
        assert tool.name in _TOOL_NAMES, f"Unexpected tool name: {tool.name!r}"
        assert tool.id.startswith("toolu_"), (
            f"Expected a toolu_ prefixed id derived from the Bedrock toolUseId, "
            f"got: {tool.id!r}"
        )
        assert isinstance(tool.input, dict), (
            f"Expected a JSON object as tool input, got: {tool.input!r}"
        )

    @pytest.mark.expensive
    @_TOOL_MODELS
    def test_tool_result_continuation(
        self, model: str, anthropic_client: Anthropic
    ) -> None:
        """A ``tool_result`` message closes the tool cycle and the model answers from it.

        The second request replays the assistant ``tool_use`` block and a user
        ``tool_result`` keyed by ``tool_use_id``; ``_map_tool_result_to_bedrock`` strips the
        ``toolu_`` prefix again, and Converse rejects the conversation if that pairing is
        lost.  A model that answers without calling the tool is skipped, not failed.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_tool_result_to_bedrock
        """
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
        assert tool_block.name in _TOOL_NAMES, (
            f"Turn 1: unexpected tool name: {tool_block.name!r}"
        )
        assert tool_block.id.startswith("toolu_"), (
            f"Turn 1: expected toolu_ prefix, got: {tool_block.id!r}"
        )

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
        assert not [b for b in resp2.content if b.type == "tool_use"], (
            "Turn 2: expected no further tool_use block after end_turn"
        )
        assert resp2.usage.input_tokens > 0, "Turn 2 must bill the replayed tool result"

    @pytest.mark.expensive
    @_STREAMING_TOOL_MODELS
    def test_streaming_tool_call(self, model: str, anthropic_client: Anthropic) -> None:
        """Streaming a tool call emits a ``content_block_start`` carrying id and name.

        Bedrock puts the ``toolUseId`` and name in ``contentBlockStart`` and streams the
        arguments as ``input_json_delta`` fragments, so the start event must already be
        fully identified.  Bedrock refuses streaming with tools on some models (400
        mentioning streaming mode) and truncates the first JSON fragment on others, which
        breaks the SDK accumulator — both are recorded as skip/xfail, not failures.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_adapters/_anthropic_message.py:_process_content_block_start
        """
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
        first_start = tool_starts[0]
        assert first_start.name in _TOOL_NAMES, (
            f"First tool_use start must name a declared tool, got: {first_start.name!r}"
        )
        assert first_start.id.startswith("toolu_"), (
            f"First tool_use start must carry the mapped call id, got: {first_start.id!r}"
        )
        final_tool_blocks = [b for b in final.content if b.type == "tool_use"]
        assert final_tool_blocks, "Accumulated message must contain the tool_use block"
        assert isinstance(final_tool_blocks[0].input, dict), (
            f"Streamed tool input must accumulate to an object, got: "
            f"{final_tool_blocks[0].input!r}"
        )

    @pytest.mark.expensive
    @pytest.mark.agentic
    @_AGENTIC_MODELS
    def test_agentic_loop_directory_and_file(
        self, model: str, anthropic_client: Anthropic
    ) -> None:
        """A multi-turn loop with two distinct tools reaches a grounded final answer.

        Every iteration replays the whole growing history — assistant ``tool_use`` blocks
        interleaved with user ``tool_result`` blocks — so this is the cross-model check that
        repeated tool round-trips stay acceptable to Converse.  ``read_file`` must be among
        the calls: the project name lives in ``pyproject.toml`` and a directory listing
        alone would let a model guess it from the path.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             stdapi/models/chat/_adapters/_anthropic_message.py:translate_request
        """
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
        assert "read_file" in tools_used, (
            f"Expected the model to read pyproject.toml, tools used: {tools_used}"
        )
        assert final_text, "Expected non-empty final answer from model"
        assert "stdapi" in final_text.lower(), (
            f"Expected project name 'stdapi' in final answer, got: {final_text[:300]!r}"
        )


# ---------------------------------------------------------------------------
# Tests: native reasoning / thinking blocks (reasoning models only)
# ---------------------------------------------------------------------------


class TestNativeReasoning:
    """Reasoning models produce native thinking blocks without the thinking param.

    Bedrock does not accept Anthropic's ``thinking`` parameter for non-Claude models, yet
    these models always emit a reasoning trace; the gateway turns each Bedrock
    ``reasoningContent`` block into an Anthropic ``thinking`` block.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlock.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_content_block_from_bedrock
    """

    @pytest.mark.expensive
    @_REASONING_MODELS
    def test_native_thinking_blocks_present(
        self, model: str, anthropic_client: Anthropic
    ) -> None:
        """Native-reasoning models return at least one ``thinking`` block.

        No ``thinking`` parameter is sent — the trace is the model's own — so a missing
        block means the ``reasoningContent`` mapping was lost rather than the model staying
        silent.  The answer text may be absent when the budget is spent on reasoning, which
        Bedrock reports as ``max_tokens``.

        Ref: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
             stdapi/models/chat/_adapters/_anthropic_message.py:format_response
        """
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
        self, model: str, anthropic_client: Anthropic
    ) -> None:
        """Streaming a native-reasoning model emits ``thinking`` block start events.

        Bedrock streams the trace as ``reasoningContent`` deltas; the gateway opens a
        ``thinking`` block for them and feeds ``thinking_delta`` events, so the accumulated
        message must end up with a populated thinking block.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_delta
        """
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
        assert thinking_blocks[0].thinking, (
            "Accumulated thinking block must carry the streamed reasoning text"
        )


# ---------------------------------------------------------------------------
# Tests: prompt caching (Amazon Nova models)
# ---------------------------------------------------------------------------


class TestPromptCaching:
    """Prompt caching on Amazon Nova models via ``cache_control`` on a system block.

    ``_build_cache_point`` turns the ephemeral ``cache_control`` into a Bedrock
    ``cachePoint``.  Nova then reports a hit in one of two ways: the documented
    ``cache_read_input_tokens`` counter, or — with hash-based implicit caching — a silently
    reduced ``input_tokens`` with no cache counter at all.  Below a model's minimum
    cacheable length nothing is cached and no error is raised.

    Ref: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
         https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_build_cache_point
    """

    @pytest.mark.slow
    @_CACHE_MODELS
    def test_cache_read_on_second_call(
        self, model: str, anthropic_client: Anthropic
    ) -> None:
        """Repeating a cached system prompt is served from the cache on the second call.

        The system block is ~800 tokens so it can clear Nova's minimum cacheable length;
        both hit signals are accepted because which one Bedrock uses is model-dependent.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_system_blocks
        """
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
        assert resp1.usage.input_tokens > 0
        assert resp2.usage.input_tokens > 0

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
    """Models can reliably produce JSON-structured output when asked.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/models/chat/_adapters/_anthropic_message.py:format_response
    """

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "model",
        [
            "anthropic.claude-haiku-4-5-20251001-v1:0",
            "amazon.nova-lite-v1:0",
            "amazon.nova-pro-v1:0",
            "deepseek.v3-v1:0",
            "minimax.minimax-m2.5",
        ],
    )
    def test_json_output_parseable(
        self, model: str, anthropic_client: Anthropic
    ) -> None:
        """Prompted JSON output round-trips through the gateway unaltered.

        No ``output_config`` is sent: the JSON contract is prompt-only, which is the
        fallback for families whose Bedrock profile has no constrained decoding.  The point
        is that the adapter returns the model text verbatim — the Markdown fences some
        models add are their own, so they are stripped before parsing.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_content_block_from_bedrock
        """
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
        assert str(data.get("language", "Python")).lower() == "python", (
            f"Expected the pinned 'language' value for {model!r}, got: {data}"
        )
        assert response.stop_reason in _STOP_REASONS, (
            f"Unexpected stop_reason for {model!r}: {response.stop_reason!r}"
        )


# ---------------------------------------------------------------------------
# Tests: vision / image input
# ---------------------------------------------------------------------------


#: Vision-capable models tested on the Anthropic route.
_VISION_MODELS = pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-haiku-4-5-20251001-v1:0",  # Claude (reference)
        "amazon.nova-lite-v1:0",  # Amazon Nova
        "mistral.ministral-3-8b-instruct",  # Mistral Ministral 3 8B
        "qwen.qwen3-vl-235b-a22b",  # Qwen3 VL 235B
        "writer.palmyra-vision-7b",  # Writer Palmyra Vision 7B
    ],
)


class TestVision:
    """Vision-capable models correctly identify the color of a simple image.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_image_to_bedrock
    """

    @pytest.mark.slow
    @_VISION_MODELS
    def test_image_color_recognition(
        self, model: str, anthropic_client: Anthropic
    ) -> None:
        """A base64 image source reaches vision models as a Bedrock ``image`` block.

        ``_map_image_to_bedrock`` resolves the source and derives the Bedrock image format
        from ``media_type``; the picture is a locally built 1x1 red PNG so the expected
        answer needs no fixture.  Nova rescales inputs, and models disagree on naming a
        single saturated pixel, so "orange" is accepted too.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/modalities-image.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_image_to_bedrock
        """
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
                                "data": red_png_b64(),
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

        assert response.usage.input_tokens > 0, (
            "image input must be billed as input tokens"
        )
        text = _text_from(response)
        assert text, f"Expected non-empty response from {model!r}"
        assert any(color in text.lower() for color in ("red", "orange")), (
            f"Expected 'red' or 'orange' in response for {model!r}, got: {text!r}"
        )
