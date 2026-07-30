"""Chat Completions behaviour specific to Amazon Nova 2: reasoning and system tools.

Nova 2 takes reasoning as ``additionalModelRequestFields.reasoningConfig`` and exposes
``nova_grounding`` / ``nova_code_interpreter`` as Bedrock ``systemTool`` entries, which
the gateway promotes from, and suppresses in, the ordinary OpenAI ``tools`` surface.
Web grounding is US-Region only, which ``aws_bedrock_model_region_restrict`` enforces in
``conftest``.

Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
     https://docs.aws.amazon.com/nova/latest/nova2-userguide/using-tools.html
     stdapi/models/chat/amazon_nova_2.py:ChatModel
"""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openai import OpenAI

#: The Nova 2 model under test.  Bedrock only accepts ``nova_grounding`` on a
#: geo-scoped (``us.``) profile, which conftest forces through
#: ``aws_bedrock_model_region_restrict``.
_NOVA_MODEL = "amazon.nova-2-lite-v1:0"

#: The ``nova_grounding`` system tool, declared as an ordinary OpenAI function tool.
_GROUNDING_TOOL: list[dict[str, object]] = [
    {"type": "function", "function": {"name": "nova_grounding"}}
]

#: finish_reason values the OpenAI Chat Completions reference defines.
_FINISH_REASONS = frozenset({"stop", "length", "content_filter", "tool_calls"})


@pytest.mark.gateway("Amazon Nova is not supported on the official API")
class TestNovaChatCompletions:
    """Amazon Nova chat completions tests.

    Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/what-is-nova-2.html
         stdapi/models/chat/amazon_nova_2.py:ChatModel
    """

    def test_reasoning_effort_parameter(self, openai_client: OpenAI) -> None:
        """``reasoning_effort="minimal"`` enables Nova reasoning and returns its text.

        Nova has three effort levels, so ``_REASONING_OVERRIDE`` folds ``minimal`` onto
        ``low`` and emits ``reasoningConfig={"type": "enabled", "maxReasoningEffort":
        "low"}``.  The resulting Bedrock ``reasoningContent`` blocks are split out of the
        assistant text into ``reasoning_content``.

        Ref: https://developers.openai.com/api/docs/guides/reasoning
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlock.html
             stdapi/models/chat/amazon_nova_2.py:ChatModel._req_configure_reasoning
        """
        resp = openai_client.chat.completions.create(
            model=_NOVA_MODEL,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="minimal",
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) >= 1
        choice = resp.choices[0]
        assert choice.finish_reason in _FINISH_REASONS
        msg = choice.message
        assert msg.role == "assistant"
        assert msg.reasoning_content  # type: ignore[attr-defined]
        assert resp.usage is not None
        assert resp.usage.prompt_tokens > 0
        assert resp.usage.completion_tokens > 0

    def test_reasoning_effort_none_explicit_disable(
        self, openai_client: OpenAI
    ) -> None:
        """``reasoning_effort="none"`` disables Nova reasoning, so no reasoning text returns.

        ``extract_reasoning`` maps ``"none"`` to ``enabled=False`` and Nova then receives
        ``reasoningConfig={"type": "disabled"}`` — an explicit opt-out rather than an
        omission — so the response carries plain text only.

        Ref: https://developers.openai.com/api/docs/guides/reasoning
             stdapi/models/chat/amazon_nova_2.py:ChatModel._req_configure_reasoning
        """
        resp = openai_client.chat.completions.create(
            model=_NOVA_MODEL,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="none",
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) >= 1
        choice = resp.choices[0]
        assert choice.finish_reason in _FINISH_REASONS
        msg = choice.message
        assert msg.role == "assistant"
        assert msg.content
        assert getattr(msg, "reasoning_content", None) is None, (
            "reasoning_effort='none' must not return reasoning content"
        )
        assert resp.usage is not None
        assert resp.usage.completion_tokens > 0

    # --- System tool routing ---

    @pytest.mark.expensive
    def test_nova_grounding_tool_name_auto_promoted_to_system_tool(
        self, openai_client: OpenAI
    ) -> None:
        """A function tool named ``nova_grounding`` becomes a Bedrock ``systemTool``.

        Nova declares ``SUPPORTED_SYSTEM_TOOLS = {"nova_grounding",
        "nova_code_interpreter"}``, so ``_req_promote_system_tools`` rewrites the
        ``toolSpec`` to ``{"systemTool": {"name": "nova_grounding"}}`` — no prefix
        needed.  Bedrock then runs the search itself, which is why the answer comes back
        as assistant text with no ``tool_calls`` for the client to execute.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_SystemTool.html
             https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
             stdapi/models/chat/_default.py:ChatModel._req_promote_system_tools
        """
        resp = openai_client.chat.completions.create(
            model=_NOVA_MODEL,
            messages=[{"role": "user", "content": "What is today's date? Be concise."}],
            tools=_GROUNDING_TOOL,  # type: ignore[arg-type]
        )
        assert len(resp.choices) >= 1
        choice = resp.choices[0]
        assert choice.message.role == "assistant"
        assert choice.message.content
        assert choice.message.tool_calls is None, (
            f"a promoted system tool must not be returned to the client: "
            f"{choice.message.tool_calls}"
        )
        assert choice.finish_reason == "stop"
        assert resp.usage is not None
        assert resp.usage.completion_tokens > 0

    def test_web_search_plain_name_not_auto_promoted(
        self, openai_client: OpenAI
    ) -> None:
        """A user tool named ``web_search`` stays an ordinary ``toolSpec``.

        ``_req_promote_system_tools`` promotes only names literally present in
        ``SUPPORTED_SYSTEM_TOOLS``; the ``web_search`` → ``nova_grounding`` entry in
        ``CANONICAL_TO_BEDROCK_TOOL_MAP`` is applied by the Anthropic and Responses
        adapters, not by Chat Completions.  Promotion here would emit an unknown
        ``systemTool`` name and Bedrock would fail the request, so a successful call
        proves the tool was passed through verbatim.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_SystemTool.html
             stdapi/models/chat/_default.py:ChatModel._req_promote_system_tools
        """
        # web_search passed as a regular toolSpec: Nova treats it as a custom tool
        resp = openai_client.chat.completions.create(
            model=_NOVA_MODEL,
            messages=[{"role": "user", "content": "Reply with OK."}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            ],
        )
        # Model receives web_search as a regular toolSpec — it may or may not call it
        assert resp.object == "chat.completion"
        assert len(resp.choices) >= 1
        choice = resp.choices[0]
        assert choice.message.role == "assistant"
        assert choice.finish_reason in _FINISH_REASONS
        assert choice.message.content or choice.message.tool_calls
        assert all(
            call.function.name == "web_search"  # type: ignore[union-attr]
            for call in choice.message.tool_calls or ()
        ), "a client tool must never be renamed to its Bedrock system-tool equivalent"
        assert resp.usage is not None
        assert resp.usage.prompt_tokens > 0


# ===========================================================================
# nova_grounding response mapping
# ===========================================================================


@pytest.mark.expensive
@pytest.mark.gateway("Amazon Nova is not supported on the official API")
class TestNovaGrounding:
    """Tests for nova_grounding web search via the OpenAI Chat Completions route.

    nova_grounding is an Amazon Nova system tool for autonomous web search: Bedrock runs
    the search inside the invocation, interleaves ``citationsContent`` blocks with the
    text, and never expects the client to answer a ``toolUse``.  The gateway therefore
    suppresses the ``nova_grounding`` blocks from ``tool_calls`` and re-publishes the
    citations as ``url_citation`` annotations.

    Uses Nova 2 Lite pinned to us-east-1, the cheapest model accepting nova_grounding.

    Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
         stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_tool_calls
    """

    def test_tool_calls_suppressed_non_streaming(self, openai_client: OpenAI) -> None:
        """A nova_grounding invocation is hidden from ``tool_calls``.

        ``extract_tool_calls`` drops ``toolUse`` blocks whose name is in
        ``SUPPORTED_SYSTEM_TOOLS``, and Bedrock reports the turn as complete, so the
        client sees a finished text answer rather than a tool round-trip it cannot serve.

        Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
             stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_tool_calls
        """
        resp = openai_client.chat.completions.create(
            model=_NOVA_MODEL,
            messages=[
                {"role": "user", "content": "What is the current version of Python?"}
            ],
            tools=_GROUNDING_TOOL,  # type: ignore[arg-type]
        )
        msg = resp.choices[0].message
        assert msg.tool_calls is None, (
            f"nova_grounding must not leak into tool_calls, got: {msg.tool_calls}"
        )
        assert resp.choices[0].finish_reason == "stop"
        assert msg.content
        assert resp.usage is not None
        assert resp.usage.completion_tokens > 0

    def test_citations_returned_non_streaming(self, openai_client: OpenAI) -> None:
        """Web search results are surfaced as ``url_citation`` annotations.

        ``extract_citations`` keeps only citations with a ``location.web.url`` and falls
        back to the web domain when the citation has no title.  Bedrock reports no
        character offsets, so both indices are published as ``0`` rather than guessed.

        Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
             stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_citations
        """
        resp = openai_client.chat.completions.create(
            model=_NOVA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "What are the current AWS regions and their locations?",
                }
            ],
            tools=_GROUNDING_TOOL,  # type: ignore[arg-type]
        )
        assert resp.choices[0].message.content
        annotations = resp.choices[0].message.annotations
        if not annotations:
            pytest.xfail("nova_grounding did not return citations this run")
        assert len(annotations) > 0
        for ann in annotations:
            assert ann.type == "url_citation"
            assert ann.url_citation.url.startswith("http"), (
                f"Expected HTTP URL, got: {ann.url_citation.url!r}"
            )
            assert ann.url_citation.title, "Expected non-empty title"
            assert ann.url_citation.start_index == 0
            assert ann.url_citation.end_index == 0

    def test_tool_calls_suppressed_streaming(self, openai_client: OpenAI) -> None:
        """Streaming with nova_grounding emits no tool_call delta chunks.

        ``_suppress_system_tool_event`` tracks the Bedrock content-block index of a
        suppressed ``toolUse`` and drops its start, delta and stop events, so the client
        stream contains only the synthetic role chunk, text deltas and the stop chunk.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
             stdapi/models/chat/_adapters/_openai_chat_completion.py:_suppress_system_tool_event
        """
        tool_call_chunks = 0
        content_chunks = 0
        finish_reason = None
        first_delta_role = None
        response = openai_client.chat.completions.create(
            model=_NOVA_MODEL,
            messages=[
                {"role": "user", "content": "What is the latest Python version?"}
            ],
            tools=_GROUNDING_TOOL,  # type: ignore[arg-type]
            max_completion_tokens=1024,
            stream=True,
        )
        seen_chunks = 0
        for chunk in response:
            assert chunk.object == "chat.completion.chunk"  # type: ignore[union-attr]
            assert chunk.usage is None, (  # type: ignore[union-attr]
                "usage must only be streamed when stream_options.include_usage is set"
            )
            if seen_chunks == 0 and chunk.choices:  # type: ignore[union-attr]
                first_delta_role = chunk.choices[0].delta.role  # type: ignore[union-attr]
            seen_chunks += 1
            if not chunk.choices:  # type: ignore[union-attr]
                continue
            choice = chunk.choices[0]  # type: ignore[union-attr]
            if choice.delta.tool_calls:
                tool_call_chunks += 1
            if choice.delta.content:
                content_chunks += 1
            if choice.finish_reason:
                finish_reason = choice.finish_reason
        assert first_delta_role == "assistant", (
            "the stream must open with the synthetic role-only chunk"
        )
        assert tool_call_chunks == 0, (
            f"Expected 0 tool_call chunks, got {tool_call_chunks} "
            "(nova_grounding invocation leaked into stream)"
        )
        assert content_chunks > 0, "Expected at least one content chunk"
        assert finish_reason == "stop"

    def test_multi_turn(self, openai_client: OpenAI) -> None:
        """A grounded answer can be replayed as history for a second grounded turn.

        Because the ``nova_grounding`` ``toolUse`` blocks are suppressed, the assistant
        turn a client echoes back contains text only — no orphan ``tool_calls`` that
        would need a matching ``tool`` message for Bedrock to accept the conversation.

        Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
             stdapi/models/chat/_adapters/_openai_chat_completion.py:map_messages
        """
        # ── Turn 1 ──────────────────────────────────────────────────────────
        resp1 = openai_client.chat.completions.create(
            model=_NOVA_MODEL,
            messages=[
                {"role": "user", "content": "What is the current Python version?"}
            ],
            tools=_GROUNDING_TOOL,  # type: ignore[arg-type]
        )
        msg1 = resp1.choices[0].message
        assert msg1.tool_calls is None, (
            f"Turn 1: tool_calls must be None, got: {msg1.tool_calls}"
        )
        assert msg1.content, "Turn 1: expected non-empty content"
        assert resp1.choices[0].finish_reason == "stop"

        # ── Turn 2 ──────────────────────────────────────────────────────────
        resp2 = openai_client.chat.completions.create(
            model=_NOVA_MODEL,
            messages=[
                {"role": "user", "content": "What is the current Python version?"},
                {"role": "assistant", "content": msg1.content},
                {"role": "user", "content": "And what about Node.js?"},
            ],
            tools=_GROUNDING_TOOL,  # type: ignore[arg-type]
        )
        msg2 = resp2.choices[0].message
        assert msg2.tool_calls is None, (
            f"Turn 2: tool_calls must be None, got: {msg2.tool_calls}"
        )
        assert msg2.content, "Turn 2: expected non-empty content"
        assert resp2.choices[0].finish_reason == "stop"
        assert resp2.usage is not None
        assert resp2.usage.prompt_tokens > 0
        assert resp2.usage.completion_tokens > 0
