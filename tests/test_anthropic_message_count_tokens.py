"""Unit tests for count_tokens_via_bedrock request building (no AWS calls)."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

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
        self.calls: list[JsonMapping] = []

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
    """``extract_reasoning`` accepts count-tokens params (no ``max_tokens`` field)."""
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


async def test_count_tokens_forwards_reasoning(
    fake_client: _FakeCountTokensClient,
) -> None:
    """The reasoning configuration reaches the counted request.

    It is forwarded via the model's own hook, matching what ``create_message``
    would actually send.
    """
    request = MessageCountTokensParams(
        model=_CLAUDE_MODEL,
        messages=[MessageParam(role="user", content="hi")],
        thinking=ThinkingConfigEnabledParam(type="enabled", budget_tokens=1024),
    )
    tokens = await count_tokens_via_bedrock(
        request, _CLAUDE_MODEL, "us-east-1", _chat_model(_CLAUDE_MODEL)
    )
    assert tokens == 42
    (call,) = fake_client.calls
    additional_fields = call["input"]["converse"]["additionalModelRequestFields"]
    assert additional_fields["reasoning_config"]["budget_tokens"] == 1024


async def test_count_tokens_promotes_server_tools_to_system_tools(
    fake_client: _FakeCountTokensClient,
) -> None:
    """A server tool is renamed and promoted exactly as ``create_message`` does.

    Counting a plain ``toolSpec`` stub instead of the ``systemTool`` entry the
    model really receives would report a different token count.
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


async def test_count_tokens_without_extras_sends_no_additional_fields(
    fake_client: _FakeCountTokensClient,
) -> None:
    """A request with no reasoning and no server tool adds no model request fields."""
    request = MessageCountTokensParams(
        model=_CLAUDE_MODEL, messages=[MessageParam(role="user", content="hi")]
    )
    await count_tokens_via_bedrock(
        request, _CLAUDE_MODEL, "us-east-1", _chat_model(_CLAUDE_MODEL)
    )
    (call,) = fake_client.calls
    assert "additionalModelRequestFields" not in call["input"]["converse"]
