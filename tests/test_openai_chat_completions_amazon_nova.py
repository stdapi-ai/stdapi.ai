"""Tests specific to Amazon Nova chat completions."""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openai import OpenAI

NOVA_ALL = ("amazon.nova-2-lite-v1:0",)

#: Nova Premier model used for system-tool tests — only available in US regions,
#: where nova_grounding is supported.
_NOVA_PREMIER = "amazon.nova-premier-v1:0"


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

    def test_system_tool_prefix_still_works_for_nova_grounding(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """``systemTool_nova_grounding`` is also accepted via the explicit prefix path.

        The ``systemTool_`` prefix unconditionally promotes the entry to a raw Bedrock
        ``systemTool`` via ``_req_promote_system_tools``, independent of
        ``SUPPORTED_SYSTEM_TOOLS``.  Uses Nova Premier because ``nova_grounding`` is only
        supported in US regions where Nova Premier is available.
        """
        if use_official_api:
            pytest.skip("Amazon Nova is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=_NOVA_PREMIER,
            messages=[{"role": "user", "content": "What is today's date? Be concise."}],
            tools=[
                {"type": "function", "function": {"name": "systemTool_nova_grounding"}}
            ],
        )
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.role == "assistant"

    def test_nova_grounding_streaming(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Streaming with ``systemTool_nova_grounding`` on Nova Premier completes without error.

        Validates:
            - Stream completes
            - At least one chunk is received
        """
        if use_official_api:
            pytest.skip("Amazon Nova is not supported on the official API")
        tools: list[dict[str, object]] = [
            {"type": "function", "function": {"name": "systemTool_nova_grounding"}}
        ]
        chunks = []
        response = openai_client.chat.completions.create(
            model=_NOVA_PREMIER,
            messages=[
                {"role": "user", "content": "What is the current weather in Seattle?"}
            ],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=2048,
            stream=True,
        )
        for chunk in response:
            if isinstance(chunk, str) and chunk == "[DONE]":
                break
            chunks.append(chunk)
        assert len(chunks) > 0

    def test_web_search_plain_name_not_auto_promoted(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Passing ``web_search`` as a plain tool name is NOT auto-promoted to a system tool.

        ``web_search`` is the Anthropic-side key in ``ANTHROPIC_TOOL_NAME_MAP``, not the
        Bedrock value (``nova_grounding``).  Only Bedrock names listed in
        ``SUPPORTED_SYSTEM_TOOLS`` trigger auto-promotion.
        ``web_search`` is passed as a regular ``toolSpec`` and Nova Premier responds normally.
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
