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
    ToolBashParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
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


async def test_count_tokens_claude_server_tool_with_custom_tool_history_no_duplicate(
    fake_client: _FakeCountTokensClient,
) -> None:
    """A Claude server tool and a custom tool from history never collide (issue #97).

    ``count_tokens_via_bedrock`` builds its own Converse-shaped request through the
    same ``_req_configure_tools`` hook ``create_message`` uses, so ``bash`` must be
    natively promoted here too, while a custom tool the assistant already invoked
    earlier in the conversation keeps a ``toolConfig`` stub instead of vanishing —
    and the two must never end up naming the same tool.

    A server-tool-*only* turn-2 conversation (no custom tool anywhere in
    history) takes the other branch and is covered by
    ``test_count_tokens_server_tool_only_history_keeps_the_stub``.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool
         stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
         stdapi/models/chat/_adapters/_anthropic_message.py:count_tokens_via_bedrock
    """
    request = MessageCountTokensParams(
        model=_CLAUDE_MODEL,
        messages=[
            MessageParam(role="user", content="What time is it, then list files?"),
            MessageParam(
                role="assistant",
                content=[
                    ToolUseBlockParam(
                        type="tool_use", id="tooluse_1", name="get_time", input={}
                    )
                ],
            ),
            MessageParam(
                role="user",
                content=[
                    ToolResultBlockParam(
                        type="tool_result", tool_use_id="tooluse_1", content="12:00"
                    )
                ],
            ),
        ],
        tools=[
            ToolParam(name="get_time", input_schema={}),  # type: ignore[arg-type]
            ToolBashParam(type="bash_20250124", name="bash"),
        ],
    )
    await count_tokens_via_bedrock(
        request, _CLAUDE_MODEL, "us-east-1", _chat_model(_CLAUDE_MODEL)
    )
    (call,) = fake_client.calls
    converse = call["input"]["converse"]
    native_names = {
        tool["name"]
        for tool in converse.get("additionalModelRequestFields", {}).get("tools", [])
    }
    config_names = {
        entry["toolSpec"]["name"]
        for entry in converse.get("toolConfig", {}).get("tools", [])
    }
    assert native_names == {"bash"}
    assert config_names == {"get_time"}
    assert not (native_names & config_names)


async def test_count_tokens_server_tool_only_history_keeps_the_stub(
    fake_client: _FakeCountTokensClient,
) -> None:
    """A server-tool-*only* turn 2 keeps the stub and skips native promotion.

    History references only ``bash``, so nothing else is left to populate a
    ``toolConfig``.  Two backend rules meet here and cannot both be satisfied:
    Bedrock rejects a request whose history carries ``toolUse``/``toolResult``
    blocks unless a ``toolConfig`` is present, and Anthropic rejects the same
    tool name appearing in both the ``toolConfig`` and the native tool list.
    Sending only the native definition was measured against a live gateway on
    2026-07-31 and returned ``400 The toolConfig field must be defined when
    using toolUse and toolResult content blocks``, so the stub wins this turn
    and the native definition is deferred; the count must reflect that exactly,
    or it would bill for a request shape Bedrock never accepts.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
    """
    request = MessageCountTokensParams(
        model=_CLAUDE_MODEL,
        messages=[
            MessageParam(role="user", content="List files in /tmp"),
            MessageParam(
                role="assistant",
                content=[
                    ToolUseBlockParam(
                        type="tool_use",
                        id="tooluse_1",
                        name="bash",
                        input={"command": "ls"},
                    )
                ],
            ),
            MessageParam(
                role="user",
                content=[
                    ToolResultBlockParam(
                        type="tool_result", tool_use_id="tooluse_1", content="file.txt"
                    )
                ],
            ),
        ],
        tools=[ToolBashParam(type="bash_20250124", name="bash")],
    )
    await count_tokens_via_bedrock(
        request, _CLAUDE_MODEL, "us-east-1", _chat_model(_CLAUDE_MODEL)
    )
    (call,) = fake_client.calls
    converse = call["input"]["converse"]
    native_names = {
        tool["name"]
        for tool in converse.get("additionalModelRequestFields", {}).get("tools", [])
    }
    assert not native_names, (
        "bash cannot be promoted natively: nothing else would populate toolConfig"
    )
    config_names = {
        entry["toolSpec"]["name"] for entry in converse["toolConfig"]["tools"]
    }
    assert config_names == {"bash"}, (
        "the stub must stay so Bedrock sees a toolConfig for the tool in history"
    )
    (stub,) = converse["toolConfig"]["tools"]
    schema = stub["toolSpec"]["inputSchema"]["json"]
    assert set(schema.get("properties", {})) == {"command", "restart"}, (
        "the retained stub must carry the documented bash input schema, exactly "
        "as create_message sends it, so the count matches the created request"
    )


async def test_count_tokens_omitted_tools_synthesizes_config_from_history(
    fake_client: _FakeCountTokensClient,
) -> None:
    """A later turn omitting ``tools`` still gets a ``toolConfig`` when history needs one.

    OpenAI-style clients (and some Anthropic ones) routinely drop ``tools`` on
    a round-trip turn once the model has already made its choice; if history
    still carries a ``toolUse``/``toolResult`` pair for a plain custom tool,
    the counted request must synthesize a permissive stub for it, mirroring
    the fallback ``create_message`` applies via ``_prepare_converse_request``
    -- otherwise the count would silently omit tokens Bedrock actually charges
    for the missing ``toolConfig``, or Bedrock could reject the request outright.

    Ref: stdapi/models/chat/_adapters/_anthropic_message.py:_synthesize_tool_config_from_history
         stdapi/models/chat/_adapters/_anthropic_message.py:count_tokens_via_bedrock
    """
    request = MessageCountTokensParams(
        model=_CLAUDE_MODEL,
        messages=[
            MessageParam(role="user", content="What time is it?"),
            MessageParam(
                role="assistant",
                content=[
                    ToolUseBlockParam(
                        type="tool_use", id="tooluse_1", name="get_time", input={}
                    )
                ],
            ),
            MessageParam(
                role="user",
                content=[
                    ToolResultBlockParam(
                        type="tool_result", tool_use_id="tooluse_1", content="12:00"
                    )
                ],
            ),
        ],
        # No `tools` on this turn -- a common round-trip pattern.
    )
    await count_tokens_via_bedrock(
        request, _CLAUDE_MODEL, "us-east-1", _chat_model(_CLAUDE_MODEL)
    )
    (call,) = fake_client.calls
    converse = call["input"]["converse"]
    assert converse["toolConfig"] == {
        "tools": [
            {
                "toolSpec": {
                    "name": "get_time",
                    "inputSchema": {"json": {"type": "object"}},
                }
            }
        ]
    }


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


async def test_count_tokens_counts_a_replayed_mcp_conversation(
    fake_client: _FakeCountTokensClient,
) -> None:
    """Replayed MCP blocks are counted as the tool use and result they are.

    ``count_tokens`` shares the Messages body, so the same conversation must be
    countable; the synthesized ``toolConfig`` is what makes Bedrock accept a
    history carrying tool blocks with no ``tools`` array.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/mcp-connector
         stdapi/models/chat/_adapters/_anthropic_message.py:_synthesize_tool_config_from_history
    """
    request = MessageCountTokensParams.model_validate(
        {
            "model": _CLAUDE_MODEL,
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "mcp_tool_use",
                            "id": "mcptoolu_01ABCdefGHIjklMNOpqrST",
                            "name": "lookup",
                            "server_name": "example",
                            "input": {"query": "hello"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "mcp_tool_result",
                            "tool_use_id": "mcptoolu_01ABCdefGHIjklMNOpqrST",
                            "content": [{"type": "text", "text": "hi"}],
                        }
                    ],
                },
            ],
        }
    )
    await count_tokens_via_bedrock(
        request, _CLAUDE_MODEL, "us-east-1", _chat_model(_CLAUDE_MODEL)
    )
    (call,) = fake_client.calls
    converse = call["input"]["converse"]
    assert [spec["toolSpec"]["name"] for spec in converse["toolConfig"]["tools"]] == [
        "lookup"
    ]
    use_id = converse["messages"][0]["content"][0]["toolUse"]["toolUseId"]
    assert use_id == converse["messages"][1]["content"][0]["toolResult"]["toolUseId"]
