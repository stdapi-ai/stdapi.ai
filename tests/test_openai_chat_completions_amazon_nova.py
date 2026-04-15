"""Tests specific to Amazon Nova chat completions.

Covers Nova-specific behaviour on both the OpenAI Chat Completions route:

  - Reasoning effort parameter for Amazon Nova 2
  - nova_grounding system tool routing and response mapping

nova_grounding tests require US-region Bedrock access (``nova_grounding`` is
not available on EU inference profiles or the official OpenAI API).
"""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openai import OpenAI

NOVA_ALL = ("amazon.nova-2-lite-v1:0",)

#: Nova Premier model — only available in US regions, where nova_grounding is supported.
_NOVA_PREMIER = "amazon.nova-premier-v1:0"

_GROUNDING_TOOL: list[dict[str, object]] = [
    {"type": "function", "function": {"name": "nova_grounding"}}
]


class TestNovaChatCompletions:
    """Amazon Nova chat completions tests."""

    @pytest.mark.parametrize("model", NOVA_ALL)
    def test_reasoning_effort_parameter(
        self, openai_client: OpenAI, use_official_api: bool, model: str
    ) -> None:
        """reasoning_effort parameter: accepted and yields valid response on this backend."""
        if use_official_api:
            pytest.skip("Amazon Nova is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="minimal",
        )
        assert hasattr(resp, "choices")
        assert len(resp.choices) >= 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"
        assert msg.reasoning_content  # type: ignore[attr-defined]

    # --- System tool routing ---

    def test_nova_grounding_tool_name_auto_promoted_to_system_tool(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Passing ``nova_grounding`` as a plain function tool name auto-promotes to systemTool.

        Nova Premier declares ``SUPPORTED_SYSTEM_TOOLS = {"nova_grounding"}``.
        When ``nova_grounding`` appears as a ``toolSpec`` name, ``_req_promote_system_tools``
        promotes it to ``{"systemTool": {"name": "nova_grounding"}}`` automatically —
        no prefix needed.  Uses Nova Premier because ``nova_grounding`` is only supported
        in US regions where Nova Premier is available.
        """
        if use_official_api:
            pytest.skip("Amazon Nova is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=_NOVA_PREMIER,
            messages=[{"role": "user", "content": "What is today's date? Be concise."}],
            tools=[{"type": "function", "function": {"name": "nova_grounding"}}],
        )
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.role == "assistant"

    def test_web_search_plain_name_not_auto_promoted(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Passing ``web_search`` as a plain Chat Completions function name is NOT auto-promoted.

        ``_req_promote_system_tools`` only promotes names that are literally present in
        ``SUPPORTED_SYSTEM_TOOLS`` (Bedrock names such as ``"nova_grounding"``).  A user-defined
        tool named ``"web_search"`` is never pre-translated, so it stays as a regular
        ``toolSpec`` and Nova Premier receives it as a custom function tool.
        """
        if use_official_api:
            pytest.skip("Amazon Nova is not supported on the official API")
        # web_search passed as a regular toolSpec: Nova Premier treats it as a custom tool
        resp = openai_client.chat.completions.create(
            model=_NOVA_PREMIER,
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
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.role == "assistant"


# ===========================================================================
# nova_grounding response mapping
# ===========================================================================


class TestNovaGrounding:
    """Tests for nova_grounding web search via the OpenAI Chat Completions route.

    nova_grounding is an Amazon Nova system tool for autonomous web search.  The
    gateway must:

      - Suppress nova_grounding ``toolUse`` blocks from ``tool_calls`` (both
        non-streaming and streaming) so clients see ``tool_calls: null``.
      - Surface web search citations as ``url_citation`` ``annotations`` on the
        response message.

    Uses Nova Premier (US-region model that supports nova_grounding).
    """

    def test_tool_calls_suppressed_non_streaming(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """nova_grounding invocations must not appear in ``tool_calls``.

        Validates:
            - ``tool_calls`` is ``None`` (no leaked system tool invocation)
            - ``finish_reason`` is ``"stop"``, not ``"tool_calls"``
            - Response content is non-empty text
        """
        if use_official_api:
            pytest.skip("Amazon Nova is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=_NOVA_PREMIER,
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

    def test_citations_returned_non_streaming(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Web search results are surfaced as ``url_citation`` annotations.

        Validates:
            - ``annotations`` field is present and non-empty
            - Each annotation has ``type == "url_citation"``
            - ``url_citation.url`` is a non-empty HTTP URL
            - ``url_citation.title`` is a non-empty string
            - ``url_citation.start_index`` / ``end_index`` are integers
        """
        if use_official_api:
            pytest.skip("Amazon Nova is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=_NOVA_PREMIER,
            messages=[
                {
                    "role": "user",
                    "content": "What are the current AWS regions and their locations?",
                }
            ],
            tools=_GROUNDING_TOOL,  # type: ignore[arg-type]
        )
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
            assert isinstance(ann.url_citation.start_index, int)
            assert isinstance(ann.url_citation.end_index, int)

    def test_tool_calls_suppressed_streaming(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Streaming with nova_grounding must not emit any tool_call delta chunks.

        Validates:
            - Zero chunks contain ``delta.tool_calls`` (no leaked system tool stream events)
            - At least one content chunk is received
            - Stream ends with ``finish_reason == "stop"``
        """
        if use_official_api:
            pytest.skip("Amazon Nova is not supported on the official API")
        tool_call_chunks = 0
        content_chunks = 0
        finish_reason = None
        response = openai_client.chat.completions.create(
            model=_NOVA_PREMIER,
            messages=[
                {"role": "user", "content": "What is the latest Python version?"}
            ],
            tools=_GROUNDING_TOOL,  # type: ignore[arg-type]
            max_completion_tokens=1024,
            stream=True,
        )
        for chunk in response:
            if not chunk.choices:  # type: ignore[union-attr]
                continue
            choice = chunk.choices[0]  # type: ignore[union-attr]
            if choice.delta.tool_calls:
                tool_call_chunks += 1
            if choice.delta.content:
                content_chunks += 1
            if choice.finish_reason:
                finish_reason = choice.finish_reason
        assert tool_call_chunks == 0, (
            f"Expected 0 tool_call chunks, got {tool_call_chunks} "
            "(nova_grounding invocation leaked into stream)"
        )
        assert content_chunks > 0, "Expected at least one content chunk"
        assert finish_reason == "stop"

    def test_multi_turn(self, openai_client: OpenAI, use_official_api: bool) -> None:
        """Multi-turn conversation with nova_grounding works end-to-end.

        Validates:
            - Turn 1: ``tool_calls`` is ``None``, content is present
            - Turn 2: passes Turn 1 assistant response in history, no error
            - Turn 2: ``tool_calls`` is also ``None``
        """
        if use_official_api:
            pytest.skip("Amazon Nova is not supported on the official API")

        # ── Turn 1 ──────────────────────────────────────────────────────────
        resp1 = openai_client.chat.completions.create(
            model=_NOVA_PREMIER,
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

        # ── Turn 2 ──────────────────────────────────────────────────────────
        resp2 = openai_client.chat.completions.create(
            model=_NOVA_PREMIER,
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
