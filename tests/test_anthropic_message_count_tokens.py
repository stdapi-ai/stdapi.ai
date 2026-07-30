"""Bedrock ``CountTokens`` request building for /v1/messages/count_tokens (no AWS calls).

Ref: https://platform.claude.com/docs/en/api/messages/count_tokens
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html
     stdapi/models/chat/_adapters/_anthropic_message.py:count_tokens_via_bedrock
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

import stdapi.models.chat._adapters._anthropic_message as anthropic_message_adapter
from stdapi.models.chat import get_chat_model
from stdapi.models.chat._adapters._anthropic_message import (
    count_tokens_via_bedrock,
    extract_reasoning,
)
from stdapi.types.anthropic_messages import (
    MessageCountTokensParams,
    MessageParam,
    ThinkingConfigEnabledParam,
    ToolParam,
    WebSearchToolParam,
)

if TYPE_CHECKING:
    from stdapi.models.chat._default import ChatModel
    from stdapi.types import JsonMapping

pytestmark = pytest.mark.local

#: Claude model used to exercise the reasoning and prompt-caching hooks.
_CLAUDE_MODEL = "anthropic.claude-opus-5"

#: Non-Claude model serving ``web_search`` through a Bedrock ``systemTool``.
_NOVA_MODEL = "amazon.nova-2-lite-v1:0"


def _chat_model(model_id: str) -> ChatModel:
    """Return the Converse chat model implementation selected for *model_id*."""
    return cast("ChatModel", get_chat_model(model_id))


class _FakeCountTokensClient:
    """Records the ``count_tokens`` call instead of hitting AWS."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def count_tokens(
        self,
        *,
        modelId: str,  # noqa: N803 (mirrors the boto3 client's camelCase kwarg)
        input: JsonMapping,  # noqa: A002
    ) -> JsonMapping:
        self.calls.append({"modelId": modelId, "input": input})
        return {"inputTokens": 42}


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> _FakeCountTokensClient:
    """Patch ``get_client`` in the adapter module to return a recording fake."""
    client = _FakeCountTokensClient()
    monkeypatch.setattr(
        anthropic_message_adapter, "get_client", lambda _service, _region=None: client
    )
    return client


def test_extract_reasoning_works_on_count_tokens_params() -> None:
    """``extract_reasoning`` accepts count-tokens params (no ``max_tokens`` field).

    The count-tokens body is the Messages body minus ``max_tokens``, so the shared
    extractor must read that field defensively and report it as unset.

    Ref: stdapi/models/chat/_adapters/_anthropic_message.py:extract_reasoning
    """
    request = MessageCountTokensParams(
        model=_CLAUDE_MODEL,
        messages=[MessageParam(role="user", content="hi")],
        thinking=ThinkingConfigEnabledParam(type="enabled", budget_tokens=1024),
    )
    reasoning = extract_reasoning(request)
    assert reasoning is not None
    assert reasoning["enabled"] is True
    assert reasoning["budget_tokens"] == 1024
    assert reasoning["max_tokens"] is None
    assert reasoning["reasoning_effort"] is None
    assert not hasattr(request, "max_tokens")


async def test_count_tokens_forwards_reasoning(
    fake_client: _FakeCountTokensClient,
) -> None:
    """The reasoning configuration reaches the counted request.

    Reasoning is forwarded through the model's own hook, matching what
    ``create_message`` sends, because thinking tokens change the counted prompt.

    Ref: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
         stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_reasoning
    """
    request = MessageCountTokensParams(
        model=_CLAUDE_MODEL,
        messages=[MessageParam(role="user", content="hi")],
        thinking=ThinkingConfigEnabledParam(type="enabled", budget_tokens=1024),
    )
    tokens = await count_tokens_via_bedrock(
        request, _CLAUDE_MODEL, "us-east-1", _chat_model(_CLAUDE_MODEL)
    )
    assert tokens == 42, "the Bedrock inputTokens value is returned unchanged"
    (call,) = fake_client.calls
    assert call["modelId"] == _CLAUDE_MODEL
    additional_fields = call["input"]["converse"]["additionalModelRequestFields"]
    assert additional_fields["reasoning_config"] == {
        "type": "enabled",
        "budget_tokens": 1024,
    }


async def test_count_tokens_promotes_server_tools_to_system_tools(
    fake_client: _FakeCountTokensClient,
) -> None:
    """A server tool is renamed and promoted exactly as ``create_message`` does.

    Counting a plain ``toolSpec`` stub instead of the ``systemTool`` entry the
    model really receives would report a different token count; Anthropic's
    ``web_search`` maps onto Amazon Nova's ``nova_grounding`` system tool.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_SystemTool.html
         https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_handle_system_tool
    """
    request = MessageCountTokensParams(
        model=_NOVA_MODEL,
        messages=[MessageParam(role="user", content="hi")],
        tools=[
            ToolParam(name="lookup", input_schema={}),  # type: ignore[arg-type]
            WebSearchToolParam(type="web_search_20250305", name="web_search"),
        ],
    )
    await count_tokens_via_bedrock(
        request, _NOVA_MODEL, "us-east-1", _chat_model(_NOVA_MODEL)
    )
    (call,) = fake_client.calls
    tools = call["input"]["converse"]["toolConfig"]["tools"]
    assert {"systemTool": {"name": "nova_grounding"}} in tools
    assert not any(
        entry.get("toolSpec", {}).get("name") == "web_search" for entry in tools
    )
    assert [entry.get("toolSpec", {}).get("name") for entry in tools] == [
        "lookup",
        None,
    ], "the client-side tool must be kept, and only the server tool promoted"


async def test_count_tokens_without_extras_sends_no_additional_fields(
    fake_client: _FakeCountTokensClient,
) -> None:
    """A request with no reasoning and no server tool adds no model request fields.

    The counted body must stay a bare Converse input: an empty
    ``additionalModelRequestFields`` would still be a Bedrock validation error.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         stdapi/models/chat/_adapters/_anthropic_message.py:count_tokens_via_bedrock
    """
    request = MessageCountTokensParams(
        model=_CLAUDE_MODEL, messages=[MessageParam(role="user", content="hi")]
    )
    await count_tokens_via_bedrock(
        request, _CLAUDE_MODEL, "us-east-1", _chat_model(_CLAUDE_MODEL)
    )
    (call,) = fake_client.calls
    assert call["input"]["converse"] == {
        "messages": [{"role": "user", "content": [{"text": "hi"}]}]
    }
    assert "additionalModelRequestFields" not in call["input"]["converse"]
