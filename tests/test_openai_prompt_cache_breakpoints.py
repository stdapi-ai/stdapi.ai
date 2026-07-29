"""Unit tests for explicit OpenAI prompt cache breakpoints (no AWS calls)."""

from __future__ import annotations

from typing import Any

import pytest

from stdapi.models.chat._adapters._openai_chat_completion import map_messages
from stdapi.models.chat._adapters._openai_common import (
    cap_cache_points,
    drop_tool_turn_cache_points,
    parse_prompt_cache_key,
    resolve_cache_ttl,
)
from stdapi.models.chat._adapters._openai_responses import (
    _build_response_object,
    map_input,
)
from stdapi.models.chat._default import ChatModel
from stdapi.types.openai_chat_completions import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    CompletionCreateParams,
    PromptCacheBreakpoint,
    PromptCacheOptions,
)
from stdapi.types.openai_responses import (
    EasyInputMessage,
    ResponseCreateParams,
    ResponseInputText,
)
from stdapi.types.openai_responses import (
    PromptCacheBreakpoint as ResponsesPromptCacheBreakpoint,
)
from stdapi.types.openai_responses import (
    PromptCacheOptions as ResponsesPromptCacheOptions,
)

pytestmark = pytest.mark.local

#: Bedrock cache point block emitted with the default TTL.
_CACHE_POINT = {"cachePoint": {"type": "default"}}

#: Explicit breakpoint mark set on a content part.
_BREAKPOINT = PromptCacheBreakpoint()

#: Explicit breakpoint mark set on a Responses input content part.
_RESPONSES_BREAKPOINT = ResponsesPromptCacheBreakpoint()


def _text(
    text: str, *, breakpoint_: bool = False
) -> ChatCompletionContentPartTextParam:
    """Return a Chat Completions text part, optionally cache-marked.

    Args:
        text: Part text.
        breakpoint_: Whether the part carries a cache breakpoint.

    Returns:
        The content part.
    """
    return ChatCompletionContentPartTextParam(
        type="text",
        text=text,
        prompt_cache_breakpoint=_BREAKPOINT if breakpoint_ else None,
    )


def _input_text(text: str, *, breakpoint_: bool = False) -> ResponseInputText:
    """Return a Responses input text part, optionally cache-marked.

    Args:
        text: Part text.
        breakpoint_: Whether the part carries a cache breakpoint.

    Returns:
        The input content part.
    """
    return ResponseInputText(
        type="input_text",
        text=text,
        prompt_cache_breakpoint=_RESPONSES_BREAKPOINT if breakpoint_ else None,
    )


# ---------------------------------------------------------------------------
# Breakpoint -> cachePoint placement
# ---------------------------------------------------------------------------


async def test_chat_cache_point_inserted_after_marked_part() -> None:
    """A marked Chat Completions part is followed by a cachePoint block."""
    messages, _ = await map_messages(
        [
            ChatCompletionUserMessageParam(
                role="user", content=[_text("a", breakpoint_=True), _text("b")]
            )
        ],
        allow_explicit_caching=True,
    )
    assert messages == [
        {"role": "user", "content": [{"text": "a"}, _CACHE_POINT, {"text": "b"}]}
    ]


async def test_chat_cache_point_uses_requested_ttl() -> None:
    """The cache TTL derived from the request is carried by the cachePoint."""
    messages, _ = await map_messages(
        [
            ChatCompletionUserMessageParam(
                role="user", content=[_text("a", breakpoint_=True)]
            )
        ],
        allow_explicit_caching=True,
        cache_ttl="1h",
    )
    assert messages[0]["content"][1] == {"cachePoint": {"type": "default", "ttl": "1h"}}


async def test_chat_system_and_assistant_cache_points() -> None:
    """Breakpoints are honored on system parts and assistant text parts."""
    messages, system_blocks = await map_messages(
        [
            ChatCompletionSystemMessageParam(
                role="system", content=[_text("sys", breakpoint_=True), _text("more")]
            ),
            ChatCompletionUserMessageParam(role="user", content="hi"),
            ChatCompletionAssistantMessageParam(
                role="assistant", content=[_text("ok", breakpoint_=True)]
            ),
        ],
        allow_explicit_caching=True,
    )
    assert system_blocks == [{"text": "sys"}, _CACHE_POINT, {"text": "more"}]
    assert messages[1]["content"] == [{"text": "ok"}, _CACHE_POINT]


async def test_chat_system_empty_parts_never_emit_stray_cache_points() -> None:
    """Marked but empty system parts yield no leading nor duplicated cachePoint."""
    _, system_blocks = await map_messages(
        [
            ChatCompletionSystemMessageParam(
                role="system",
                content=[
                    _text("", breakpoint_=True),
                    _text("sys", breakpoint_=True),
                    _text("", breakpoint_=True),
                ],
            )
        ],
        allow_explicit_caching=True,
    )
    assert system_blocks == [{"text": "sys"}, _CACHE_POINT]


async def test_responses_cache_point_inserted_after_marked_part() -> None:
    """A marked Responses input part is followed by a cachePoint block."""
    messages, _ = await map_input(
        [
            EasyInputMessage(
                role="user",
                content=[_input_text("a", breakpoint_=True), _input_text("b")],
            )
        ],
        None,
        allow_explicit_caching=True,
    )
    assert messages == [
        {"role": "user", "content": [{"text": "a"}, _CACHE_POINT, {"text": "b"}]}
    ]


async def test_responses_system_cache_point() -> None:
    """A marked system input part splits the system blocks around a cachePoint."""
    _, system_blocks = await map_input(
        [
            EasyInputMessage(
                role="system",
                content=[_input_text("sys", breakpoint_=True), _input_text("more")],
            )
        ],
        None,
        allow_explicit_caching=True,
    )
    assert system_blocks == [{"text": "sys"}, _CACHE_POINT, {"text": "more"}]


async def test_responses_system_parts_joined_without_breakpoint() -> None:
    """Unmarked system parts keep the single joined system block."""
    _, system_blocks = await map_input(
        [
            EasyInputMessage(
                role="system", content=[_input_text("sys"), _input_text("more")]
            )
        ],
        None,
        allow_explicit_caching=True,
    )
    assert system_blocks == [{"text": "sys more"}]


# ---------------------------------------------------------------------------
# Models without prompt caching support
# ---------------------------------------------------------------------------


async def test_chat_breakpoint_ignored_without_caching_support() -> None:
    """Breakpoints are accepted and ignored on models without prompt caching."""
    messages, _ = await map_messages(
        [
            ChatCompletionUserMessageParam(
                role="user", content=[_text("a", breakpoint_=True), _text("b")]
            )
        ]
    )
    assert messages == [{"role": "user", "content": [{"text": "a"}, {"text": "b"}]}]


async def test_responses_breakpoint_ignored_without_caching_support() -> None:
    """Breakpoints are accepted and ignored on models without prompt caching."""
    messages, _ = await map_input(
        [EasyInputMessage(role="user", content=[_input_text("a", breakpoint_=True)])],
        None,
    )
    assert messages == [{"role": "user", "content": [{"text": "a"}]}]


# ---------------------------------------------------------------------------
# Explicit mode heuristic suppression
# ---------------------------------------------------------------------------


def test_explicit_mode_disables_prompt_cache_key_heuristic() -> None:
    """``mode="explicit"`` disables the ``prompt_cache_key`` driven placement."""
    assert (
        parse_prompt_cache_key("system.tools", PromptCacheOptions(mode="explicit"))
        == set()
    )
    assert (
        parse_prompt_cache_key(
            "system.tools", ResponsesPromptCacheOptions(mode="explicit")
        )
        == set()
    )


def test_implicit_mode_keeps_prompt_cache_key_heuristic() -> None:
    """``implicit`` mode and no options keep the key-driven placement."""
    assert parse_prompt_cache_key("system", PromptCacheOptions(mode="implicit")) == {
        "system"
    }
    assert parse_prompt_cache_key("system", PromptCacheOptions(ttl="30m")) == {"system"}
    assert parse_prompt_cache_key("system") == {"system"}


def test_prompt_cache_options_ttl_resolves_to_bedrock_ttl() -> None:
    """``prompt_cache_options.ttl`` applies when no retention is requested."""
    assert resolve_cache_ttl(None, ResponsesPromptCacheOptions(ttl="30m")) == "1h"
    assert resolve_cache_ttl("5m", ResponsesPromptCacheOptions(ttl="30m")) == "5m"
    assert resolve_cache_ttl(None, ResponsesPromptCacheOptions(mode="explicit")) is None
    assert resolve_cache_ttl(None) is None


# ---------------------------------------------------------------------------
# Cache point limits
# ---------------------------------------------------------------------------


def test_cap_cache_points_drops_the_oldest() -> None:
    """Only the latest cache points survive the Bedrock per-request limit."""
    system_blocks = [{"text": "sys"}, _CACHE_POINT]
    tool_config = {"tools": [{"toolSpec": {"name": "t"}}, _CACHE_POINT]}
    messages = [
        {"role": "user", "content": [{"text": str(index)}, _CACHE_POINT]}
        for index in range(4)
    ]
    cap_cache_points(system_blocks, tool_config, messages, 4)  # type: ignore[arg-type]
    # The two oldest cache points (system, then tools) are the ones dropped.
    assert system_blocks == [{"text": "sys"}]
    assert tool_config == {"tools": [{"toolSpec": {"name": "t"}}]}
    assert [len(message["content"]) for message in messages] == [2, 2, 2, 2]


def test_cap_cache_points_drops_the_oldest_message_breakpoints() -> None:
    """Message cache points beyond the limit are dropped oldest first."""
    messages = [
        {"role": "user", "content": [{"text": str(index)}, _CACHE_POINT]}
        for index in range(6)
    ]
    cap_cache_points(None, None, messages, 4)  # type: ignore[arg-type]
    assert [len(message["content"]) for message in messages] == [1, 1, 2, 2, 2, 2]


def test_cap_cache_points_keeps_requests_within_the_limit() -> None:
    """Requests at or below the limit are left untouched."""
    messages = [{"role": "user", "content": [{"text": "a"}, _CACHE_POINT]}]
    cap_cache_points(None, None, messages, 4)  # type: ignore[arg-type]
    assert messages == [{"role": "user", "content": [{"text": "a"}, _CACHE_POINT]}]


def test_drop_tool_turn_cache_points() -> None:
    """Cache points are removed from turns carrying tool blocks."""
    messages = [
        {"role": "user", "content": [{"text": "a"}, _CACHE_POINT]},
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": "1", "content": []}},
                _CACHE_POINT,
            ],
        },
    ]
    drop_tool_turn_cache_points(messages)  # type: ignore[arg-type]
    assert messages[0]["content"] == [{"text": "a"}, _CACHE_POINT]
    assert messages[1]["content"] == [{"toolResult": {"toolUseId": "1", "content": []}}]


class _CachingModel(ChatModel):
    """Chat model caching everything but tool turns, with a two cache-point limit."""

    PROMPT_CACHING_SUPPORTED = True
    MAX_CACHE_BLOCKS = 2


class _ToolCachingModel(_CachingModel):
    """Chat model also caching tool turns."""

    PROMPT_CACHING_TOOL_SUPPORTED = True


def _limit_fixture() -> tuple[list[Any], dict[str, Any], list[Any]]:
    """Build system blocks, a tool config and messages each carrying a cache point."""
    return (
        [{"text": "sys"}, dict(_CACHE_POINT)],
        {"tools": [{"toolSpec": {"name": "t"}}, dict(_CACHE_POINT)]},
        [
            {
                "role": "assistant",
                "content": [{"toolUse": {"name": "t"}}, _CACHE_POINT],
            },
            {"role": "user", "content": [{"text": "u"}, _CACHE_POINT]},
        ],
    )


def test_req_limit_cache_points_drops_tool_turns_then_caps() -> None:
    """Without tool caching, tool-turn breakpoints go first, then the oldest ones."""
    system_blocks, tool_config, messages = _limit_fixture()
    _CachingModel("model")._req_limit_cache_points(  # noqa: SLF001
        system_blocks,
        tool_config,  # type: ignore[arg-type]
        messages,
    )
    assert system_blocks == [{"text": "sys"}]
    assert tool_config == {"tools": [{"toolSpec": {"name": "t"}}, _CACHE_POINT]}
    assert messages[0]["content"] == [{"toolUse": {"name": "t"}}]
    assert messages[1]["content"] == [{"text": "u"}, _CACHE_POINT]


def test_req_limit_cache_points_keeps_tool_turns_when_supported() -> None:
    """With tool caching, the tool turn keeps its breakpoint and older ones are cut."""
    system_blocks, tool_config, messages = _limit_fixture()
    _ToolCachingModel("model")._req_limit_cache_points(  # noqa: SLF001
        system_blocks,
        tool_config,  # type: ignore[arg-type]
        messages,
    )
    assert system_blocks == [{"text": "sys"}]
    assert tool_config == {"tools": [{"toolSpec": {"name": "t"}}]}
    assert messages[0]["content"] == [{"toolUse": {"name": "t"}}, _CACHE_POINT]
    assert messages[1]["content"] == [{"text": "u"}, _CACHE_POINT]


def test_req_limit_cache_points_noop_without_prompt_caching() -> None:
    """A model without prompt caching never emits cache points, so nothing is edited."""
    system_blocks, tool_config, messages = _limit_fixture()
    ChatModel("model")._req_limit_cache_points(  # noqa: SLF001
        system_blocks,
        tool_config,  # type: ignore[arg-type]
        messages,
    )
    assert system_blocks == [{"text": "sys"}, _CACHE_POINT]
    assert messages[0]["content"] == [{"toolUse": {"name": "t"}}, _CACHE_POINT]


# ---------------------------------------------------------------------------
# Response echo
# ---------------------------------------------------------------------------


def test_response_echoes_prompt_cache_options() -> None:
    """``prompt_cache_options`` is echoed back on the Responses API response."""
    options = ResponsesPromptCacheOptions(mode="explicit", ttl="30m")
    response = _build_response_object(
        "resp-1",
        0.0,
        "model-id",
        [],
        "completed",
        None,
        None,
        None,
        ResponseCreateParams(model="model-id", prompt_cache_options=options),
    )
    assert response.prompt_cache_options == options


def test_response_without_prompt_cache_options() -> None:
    """No ``prompt_cache_options`` is echoed when the request did not set it."""
    response = _build_response_object(
        "resp-1",
        0.0,
        "model-id",
        [],
        "completed",
        None,
        None,
        None,
        ResponseCreateParams(model="model-id"),
    )
    assert response.prompt_cache_options is None


# ---------------------------------------------------------------------------
# Bedrock Mantle passthrough
# ---------------------------------------------------------------------------


def test_breakpoints_survive_request_serialization() -> None:
    """Parsed breakpoints are kept when the request is dumped for Mantle."""
    chat_request = CompletionCreateParams.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "a",
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        }
                    ],
                }
            ],
        }
    )
    dumped = chat_request.model_dump(mode="json", by_alias=True, exclude_unset=True)
    assert dumped["messages"][0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }

    responses_request = ResponseCreateParams.model_validate(
        {
            "model": "m",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "a",
                            "prompt_cache_breakpoint": {"mode": "explicit"},
                        }
                    ],
                }
            ],
            "prompt_cache_options": {"mode": "explicit"},
        }
    )
    dumped = responses_request.model_dump(
        mode="json", by_alias=True, exclude_unset=True
    )
    assert dumped["input"][0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }
    assert dumped["prompt_cache_options"] == {"mode": "explicit"}


class TestCacheTTLCapability:
    """_cache_ttl: extended TTLs reach Bedrock only on models supporting them."""

    def test_ttl_kept_on_supporting_models(self) -> None:
        """A TTL-capable model resolves the requested extended TTL."""

        class _TTLModel(_CachingModel):
            PROMPT_CACHING_TTL_SUPPORTED = True

        assert _TTLModel("m")._cache_ttl("24h") == "1h"  # noqa: SLF001

    def test_ttl_dropped_on_other_models(self) -> None:
        """Bedrock rejects extended TTLs outside Anthropic models: fall back to default."""
        assert _CachingModel("m")._cache_ttl("24h") is None  # noqa: SLF001
        options = PromptCacheOptions(mode="explicit", ttl="30m")
        assert _CachingModel("m")._cache_ttl(None, options) is None  # noqa: SLF001
