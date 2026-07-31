"""Offline unit tests for the OpenAI Chat Completions Bedrock adapter (no AWS calls).

Ref: https://developers.openai.com/api/reference/resources/chat.md
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
     stdapi/models/chat/_adapters/_openai_chat_completion.py
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest
from pybase64 import b64decode

from stdapi.aws_bedrock import PROMPT_CACHING
from stdapi.config import SETTINGS
from stdapi.models.chat._adapters._openai_chat_completion import (
    _FINISH_REASONS,
    _LEGACY_FUNCTION,
    _get_or_generate_audio,
    build_output_config,
    extract_output_text,
    format_response,
    format_stream,
    map_bedrock_stop_reason,
    map_messages,
    translate_request,
)
from stdapi.models.chat._adapters._openai_common import (
    CACHE_TTL,
    JSON_OBJECT_SYSTEM_INSTRUCTION,
    enforce_json_object,
    map_service_tier,
    parse_prompt_cache_key,
    resolve_cache_ttl,
)
from stdapi.types.openai import (
    JSONSchema,
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
    ResponseFormatText,
)
from stdapi.types.openai_chat_completions import (
    Audio,
    ChatCompletionAssistantMessageParam,
    ChatCompletionAudioParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
    CompletionCreateParams,
    CompletionUsage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator

pytestmark = pytest.mark.local


@pytest.fixture(autouse=True)
def _adapter_call_context(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    """Bind the per-request state the adapter reads outside a real request.

    ``format_response``/``format_stream`` read the ``_LEGACY_FUNCTION`` contextvar
    (normally set by ``translate_request``) and log request params through
    ``SETTINGS``; binding both here keeps the contextvar from leaking between
    tests and keeps the offline runs silent.

    Ref: stdapi/models/chat/_adapters/_openai_chat_completion.py:_LEGACY_FUNCTION
    """
    monkeypatch.setattr(SETTINGS, "log_request_params", False)
    token = _LEGACY_FUNCTION.set(False)
    try:
        yield
    finally:
        _LEGACY_FUNCTION.reset(token)


async def _stub_stream(events: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    """Yield the given Bedrock Converse stream event dicts one by one.

    Args:
        events: Converse stream event dicts to replay.

    Yields:
        Each event dict, in order.
    """
    for event in events:
        yield event


async def _collect_chunks(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run ``format_stream`` over stub events and return the decoded chunk payloads.

    Args:
        events: Converse stream event dicts to replay.

    Returns:
        The JSON-decoded chunks, excluding the ``[DONE]`` sentinel.
    """
    sse_events = [
        event
        async for event in format_stream(
            completion_id="chatcmpl-1",
            created=0,
            model_id="model",
            stream=_stub_stream(events),  # type: ignore[arg-type]
            service_tier=None,
        )
    ]
    assert sse_events[-1].data == "[DONE]", "the stream must end with the sentinel"
    return [
        json.loads(event.data)
        for event in sse_events
        if isinstance(event.data, str) and event.data != "[DONE]"
    ]


class TestMapMessagesRoleAlternation:
    """Consecutive messages with the same Bedrock role are merged into one turn.

    OpenAI allows any message sequence, while Converse expects alternating
    user/assistant turns, so adjacent same-role messages must become extra
    content blocks of a single turn rather than extra turns.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         stdapi/models/chat/_adapters/_openai_chat_completion.py:_append_or_merge
    """

    async def test_consecutive_user_messages_are_merged(self) -> None:
        """Two user messages produce a single Bedrock user turn."""
        messages, _ = await map_messages(
            [
                ChatCompletionUserMessageParam(role="user", content="a"),
                ChatCompletionUserMessageParam(role="user", content="b"),
            ]
        )
        assert messages == [{"role": "user", "content": [{"text": "a"}, {"text": "b"}]}]

    async def test_consecutive_assistant_messages_are_merged(self) -> None:
        """Two assistant messages produce a single Bedrock assistant turn."""
        messages, _ = await map_messages(
            [
                ChatCompletionUserMessageParam(role="user", content="q"),
                ChatCompletionAssistantMessageParam(role="assistant", content="a"),
                ChatCompletionAssistantMessageParam(role="assistant", content="b"),
            ]
        )
        assert messages == [
            {"role": "user", "content": [{"text": "q"}]},
            {"role": "assistant", "content": [{"text": "a"}, {"text": "b"}]},
        ]

    async def test_tool_message_merges_with_following_user_message(self) -> None:
        """A tool result and the next user message share one Bedrock user turn.

        Converse has no ``tool`` role: a ``tool`` message becomes a
        ``toolResult`` block on a user turn, which then absorbs the next user
        message instead of opening a second consecutive user turn.
        """
        messages, _ = await map_messages(
            [
                ChatCompletionToolMessageParam(
                    role="tool", content="ok", tool_call_id="call_1"
                ),
                ChatCompletionUserMessageParam(role="user", content="next"),
            ]
        )
        assert messages == [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call_1",
                            "content": [{"text": "ok"}],
                        }
                    },
                    {"text": "next"},
                ],
            }
        ]

    async def test_consecutive_tool_messages_are_merged(self) -> None:
        """Consecutive tool results stay merged into a single user turn.

        Each result keeps its own ``toolUseId``, which is what lets the model
        pair parallel tool calls with their outputs.
        """
        messages, _ = await map_messages(
            [
                ChatCompletionToolMessageParam(
                    role="tool", content="r1", tool_call_id="call_1"
                ),
                ChatCompletionToolMessageParam(
                    role="tool", content="r2", tool_call_id="call_2"
                ),
            ]
        )
        assert messages == [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call_1",
                            "content": [{"text": "r1"}],
                        }
                    },
                    {
                        "toolResult": {
                            "toolUseId": "call_2",
                            "content": [{"text": "r2"}],
                        }
                    },
                ],
            }
        ]

    async def test_mid_conversation_system_message_does_not_split_user_turn(
        self,
    ) -> None:
        """A system message between two user messages leaves a single user turn.

        Converse carries system instructions in a top-level ``system`` array,
        so a mid-conversation ``system`` message is lifted out of the turn
        sequence entirely.
        """
        messages, system_blocks = await map_messages(
            [
                ChatCompletionUserMessageParam(role="user", content="a"),
                ChatCompletionSystemMessageParam(role="system", content="rules"),
                ChatCompletionUserMessageParam(role="user", content="b"),
            ]
        )
        assert system_blocks == [{"text": "rules"}]
        assert messages == [{"role": "user", "content": [{"text": "a"}, {"text": "b"}]}]


class TestFormatResponseCacheWriteTokens:
    """Cache-write tokens are reported in ``prompt_tokens_details``.

    Bedrock reports ``cacheReadInputTokens``/``cacheWriteInputTokens`` outside
    ``inputTokens``, while OpenAI's ``prompt_tokens`` includes both buckets, so
    the adapter has to add them back into ``prompt_tokens``.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
         https://developers.openai.com/api/reference/resources/chat.md
         stdapi/models/chat/_adapters/_openai_chat_completion.py:format_response
    """

    @staticmethod
    def _converse_response(cache_read: int, cache_write: int) -> dict[str, Any]:
        """Build a minimal Converse response with the given cache token counts.

        Args:
            cache_read: ``cacheReadInputTokens`` value.
            cache_write: ``cacheWriteInputTokens`` value.

        Returns:
            A Converse response payload.
        """
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 5,
                "cacheReadInputTokens": cache_read,
                "cacheWriteInputTokens": cache_write,
            },
        }

    async def _usage(self, cache_read: int, cache_write: int) -> CompletionUsage:
        """Format a response with the given cache token counts and return its usage.

        Args:
            cache_read: ``cacheReadInputTokens`` value.
            cache_write: ``cacheWriteInputTokens`` value.

        Returns:
            The usage of the formatted completion.
        """
        completion = await format_response(
            completion_id="chatcmpl-1",
            created=0,
            model_id="model",
            responses=[self._converse_response(cache_read, cache_write)],  # type: ignore[list-item]
            service_tier=None,
            audio_params=None,
            modalities=["text"],
        )
        assert completion.usage is not None
        return completion.usage

    async def test_cache_write_tokens_are_reported(self) -> None:
        """A positive cacheWriteInputTokens is exposed as cache_write_tokens."""
        usage = await self._usage(cache_read=0, cache_write=7)
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cache_write_tokens == 7
        assert usage.prompt_tokens_details.cached_tokens == 0
        assert usage.prompt_tokens == 17, (
            "prompt_tokens must include the cache-write bucket Bedrock reports apart"
        )
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 22

    async def test_cache_read_and_write_tokens_are_reported(self) -> None:
        """Both cache buckets are reported together."""
        usage = await self._usage(cache_read=3, cache_write=7)
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 3
        assert usage.prompt_tokens_details.cache_write_tokens == 7
        assert usage.prompt_tokens == 20, (
            "prompt_tokens must include both cache buckets, unlike Bedrock inputTokens"
        )
        assert usage.total_tokens == 25

    async def test_details_omitted_without_cache_usage(self) -> None:
        """No cache usage leaves prompt_tokens_details unset."""
        usage = await self._usage(cache_read=0, cache_write=0)
        assert usage.prompt_tokens_details is None
        assert usage.prompt_tokens == 10
        assert usage.total_tokens == 15


class TestLegacyFunctionDetection:
    """Legacy function format detected from history survives ``translate_request``.

    The flag decides the whole response shape: a legacy request gets a single
    ``function_call`` and the ``function_call`` finish reason instead of
    ``tool_calls``.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
         stdapi/models/chat/_adapters/_openai_chat_completion.py:map_bedrock_stop_reason
    """

    @staticmethod
    async def _legacy_flag(payload: dict[str, Any]) -> bool:
        """Map the payload messages then translate it, returning the legacy flag.

        Args:
            payload: Raw chat completion request payload.

        Returns:
            The legacy function format flag seen by the response formatters.
        """
        request = CompletionCreateParams.model_validate(payload)
        await map_messages(request.messages)
        translate_request(request, "model")
        return _LEGACY_FUNCTION.get()

    async def test_function_message_history_enables_legacy_format(self) -> None:
        """A replayed `function` message keeps the legacy format without `functions`."""
        legacy = await self._legacy_flag(
            {
                "model": "model",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "function", "name": "f", "content": "ok"},
                ],
            }
        )
        assert legacy
        assert map_bedrock_stop_reason("tool_use", legacy_function=legacy) == (
            "function_call"
        )

    async def test_declared_tools_disable_legacy_format(self) -> None:
        """Declared `tools` win over legacy history detection."""
        legacy = await self._legacy_flag(
            {
                "model": "model",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "function", "name": "f", "content": "ok"},
                ],
                "tools": [
                    {"type": "function", "function": {"name": "f", "parameters": {}}}
                ],
            }
        )
        assert not legacy
        assert (
            map_bedrock_stop_reason("tool_use", legacy_function=legacy) == "tool_calls"
        )

    async def test_plain_request_keeps_tool_calls_format(self) -> None:
        """A request without legacy history or `functions` stays on tool_calls format."""
        legacy = await self._legacy_flag(
            {"model": "model", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert not legacy
        assert (
            map_bedrock_stop_reason("tool_use", legacy_function=legacy) == "tool_calls"
        )


class TestExtractOutputText:
    """Non-streaming text matches the concatenation of the streamed deltas.

    Text and reasoning are accumulated in separate buffers, and neither picks
    up a separator or content from the other, so a client that concatenated
    stream deltas sees the same strings.

    Ref: https://developers.openai.com/api/reference/resources/chat.md
         stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_output_text
    """

    def test_multiple_text_blocks_are_concatenated_without_separator(self) -> None:
        """Two text blocks join exactly like their streamed deltas would."""
        content, reasoning = extract_output_text(
            [{"text": "A"}, {"citationsContent": {}}, {"text": "B"}]
        )
        assert content == "AB"
        assert reasoning is None

    def test_multiple_reasoning_blocks_are_concatenated_without_separator(self) -> None:
        """Two reasoning blocks join exactly like their streamed deltas would."""
        content, reasoning = extract_output_text(
            [
                {"reasoningContent": {"reasoningText": {"text": "A"}}},
                {"reasoningContent": {"reasoningText": {"text": "B"}}},
            ]
        )
        assert reasoning == "AB"
        assert content is None, "reasoning text must not leak into the message content"


class TestMapMessagesEmptyContent:
    """Messages yielding no content block never reach Bedrock as empty messages.

    Converse rejects a message with an empty ``content`` array, so a turn that
    maps to nothing is dropped, and the surrounding same-role turns merge.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         stdapi/models/chat/_adapters/_openai_chat_completion.py:_append_or_merge
    """

    async def test_assistant_audio_reference_only_message_is_dropped(self) -> None:
        """An assistant turn with only an ``audio`` reference emits no Bedrock message.

        Ref: https://developers.openai.com/api/docs/guides/audio
        """
        messages, _ = await map_messages(
            [
                ChatCompletionUserMessageParam(role="user", content="hi"),
                ChatCompletionAssistantMessageParam(
                    role="assistant", content=None, audio=Audio(id="audio-x")
                ),
                ChatCompletionUserMessageParam(role="user", content="and now?"),
            ]
        )
        assert messages == [
            {"role": "user", "content": [{"text": "hi"}, {"text": "and now?"}]}
        ]

    async def test_empty_assistant_content_message_is_dropped(self) -> None:
        """An assistant turn with empty string content emits no Bedrock message."""
        messages, _ = await map_messages(
            [
                ChatCompletionUserMessageParam(role="user", content="hi"),
                ChatCompletionAssistantMessageParam(role="assistant", content=""),
            ]
        )
        assert messages == [{"role": "user", "content": [{"text": "hi"}]}]


class TestStreamToolCallIndex:
    """Streamed tool call indices are 0-based positions in the tool_calls array.

    Bedrock numbers ``contentBlockIndex`` across every content block, including
    text and reasoning ones, while OpenAI clients accumulate tool call deltas by
    their position in ``tool_calls``; the adapter therefore remaps Bedrock
    indices to contiguous positions starting at 0.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
         https://developers.openai.com/api/docs/guides/function-calling#streaming
         stdapi/models/chat/_adapters/_openai_chat_completion.py:format_stream
    """

    @staticmethod
    def _indices(chunks: list[dict[str, Any]]) -> list[int]:
        """Return every streamed tool call index, in emission order.

        Args:
            chunks: Decoded ChatCompletionChunk payloads.

        Returns:
            The ``index`` of each streamed tool call delta.
        """
        return [
            tool_call["index"]
            for chunk in chunks
            for choice in chunk["choices"]
            for tool_call in choice["delta"].get("tool_calls", ())
        ]

    async def test_tool_call_index_ignores_preceding_content_blocks(self) -> None:
        """A tool call following a reasoning block still streams with index 0."""
        chunks = await _collect_chunks(
            [
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"reasoningContent": {"text": "think"}},
                    }
                },
                {
                    "contentBlockStart": {
                        "contentBlockIndex": 1,
                        "start": {"toolUse": {"toolUseId": "t1", "name": "f1"}},
                    }
                },
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 1,
                        "delta": {"toolUse": {"input": '{"a":'}},
                    }
                },
                {"messageStop": {"stopReason": "tool_use"}},
            ]
        )
        assert self._indices(chunks) == [0, 0]
        assert len(chunks) == 5
        assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
        assert chunks[1]["choices"][0]["delta"]["reasoning_content"] == "think"
        assert chunks[2]["choices"][0]["delta"]["tool_calls"] == [
            {"index": 0, "id": "t1", "type": "function", "function": {"name": "f1"}}
        ]
        assert chunks[3]["choices"][0]["delta"]["tool_calls"] == [
            {"index": 0, "type": "function", "function": {"arguments": '{"a":'}}
        ]
        assert chunks[4]["choices"][0]["finish_reason"] == "tool_calls"
        assert not any("usage" in chunk for chunk in chunks), (
            "usage is only emitted when stream_options.include_usage is set"
        )

    async def test_parallel_tool_calls_are_numbered_contiguously(self) -> None:
        """Two toolUse blocks after a text block stream as indices 0 and 1."""
        chunks = await _collect_chunks(
            [
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"text": "hi"},
                    }
                },
                {
                    "contentBlockStart": {
                        "contentBlockIndex": 1,
                        "start": {"toolUse": {"toolUseId": "t1", "name": "f1"}},
                    }
                },
                {
                    "contentBlockStart": {
                        "contentBlockIndex": 2,
                        "start": {"toolUse": {"toolUseId": "t2", "name": "f2"}},
                    }
                },
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 2,
                        "delta": {"toolUse": {"input": "{}"}},
                    }
                },
                {"messageStop": {"stopReason": "tool_use"}},
            ]
        )
        assert self._indices(chunks) == [0, 1, 1]
        assert len(chunks) == 6
        assert chunks[1]["choices"][0]["delta"]["content"] == "hi"
        assert chunks[2]["choices"][0]["delta"]["tool_calls"] == [
            {"index": 0, "id": "t1", "type": "function", "function": {"name": "f1"}}
        ]
        assert chunks[3]["choices"][0]["delta"]["tool_calls"] == [
            {"index": 1, "id": "t2", "type": "function", "function": {"name": "f2"}}
        ]
        assert chunks[4]["choices"][0]["delta"]["tool_calls"] == [
            {"index": 1, "type": "function", "function": {"arguments": "{}"}}
        ]
        assert chunks[5]["choices"][0]["finish_reason"] == "tool_calls"
        assert {chunk["id"] for chunk in chunks} == {"chatcmpl-1"}


class TestMapBedrockStopReason:
    """Every Bedrock ``stopReason`` maps to a documented OpenAI ``finish_reason``.

    The OpenAI enum is exactly ``stop``/``length``/``tool_calls``/``content_filter``
    (plus the legacy ``function_call``), so Bedrock's wider stop-reason vocabulary
    is folded into it, and anything unknown degrades to ``stop``.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         stdapi/models/chat/_adapters/_openai_chat_completion.py:map_bedrock_stop_reason
    """

    @pytest.mark.parametrize(
        ("stop_reason", "expected"),
        [
            ("max_tokens", "length"),
            ("model_context_window_exceeded", "length"),
            ("incomplete", "length"),
            ("content_filtered", "content_filter"),
            ("guardrail_intervened", "content_filter"),
            ("malformed_model_output", "content_filter"),
            ("malformed_tool_use", "content_filter"),
            ("tool_use", "tool_calls"),
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("a-reason-bedrock-has-not-shipped-yet", "stop"),
            (None, "stop"),
        ],
    )
    def test_stop_reason_maps_to_finish_reason(
        self, stop_reason: str | None, expected: str
    ) -> None:
        """A truncated or filtered turn is never reported as a completed one.

        ``model_context_window_exceeded`` and the non-standard ``incomplete``
        must surface as ``length``, and the four guardrail/malformed reasons as
        ``content_filter``; collapsing any of them to ``stop`` would let a client
        treat a censored or cut-off answer as a finished one.
        """
        assert map_bedrock_stop_reason(stop_reason, legacy_function=False) == expected

    def test_every_mapping_table_entry_is_covered(self) -> None:
        """The parametrized cases enumerate the whole ``_FINISH_REASONS`` table."""
        covered = {
            "max_tokens",
            "model_context_window_exceeded",
            "incomplete",
            "content_filtered",
            "guardrail_intervened",
            "malformed_model_output",
            "malformed_tool_use",
            "tool_use",
        }
        assert set(_FINISH_REASONS) == covered

    @pytest.mark.parametrize(
        "stop_reason", ["max_tokens", "content_filtered", "end_turn", None]
    )
    def test_legacy_flag_only_rewrites_tool_calls(
        self, stop_reason: str | None
    ) -> None:
        """``legacy_function`` rewrites only ``tool_calls``, never the other reasons.

        Ref: https://developers.openai.com/api/reference/resources/chat.md
        """
        assert map_bedrock_stop_reason(
            stop_reason, legacy_function=True
        ) == map_bedrock_stop_reason(stop_reason, legacy_function=False)


class TestServiceTierMapping:
    """``service_tier`` maps to the Bedrock tier applied and the tier echoed back.

    Bedrock only knows ``priority``/``flex``/``reserved``; the remaining OpenAI
    values leave the Converse request untouched and are echoed as ``default``.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
         https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
         stdapi/models/chat/_adapters/_openai_common.py:map_service_tier
    """

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("priority", ("priority", "priority")),
            ("flex", ("flex", "flex")),
            ("reserved", ("reserved", "reserved")),
            ("auto", (None, "default")),
            ("default", (None, "default")),
            ("scale", (None, "default")),
            (None, (None, None)),
        ],
    )
    def test_service_tier_maps_only_bedrock_backed_values(
        self, value: str | None, expected: tuple[str | None, str | None]
    ) -> None:
        """Paid tiers reach Bedrock; the rest collapse to the standard tier.

        The first element is what goes on the Converse request, so a regression
        that stopped forwarding ``priority``/``flex`` would silently downgrade a
        paid request while the echoed value stayed unchanged.
        """
        assert map_service_tier(value) == expected  # type: ignore[arg-type]


class TestPromptCacheRetention:
    """``prompt_cache_retention`` resolves to the Bedrock ``cachePoint`` TTL.

    Bedrock's longest TTL is ``1h``, so OpenAI's ``24h`` is clamped to it, while
    ``in_memory`` means "Bedrock default" and emits no explicit TTL.

    Ref: https://developers.openai.com/api/docs/guides/prompt-caching
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CachePointBlock.html
         stdapi/models/chat/_adapters/_openai_common.py:resolve_cache_ttl
    """

    @pytest.mark.parametrize(
        ("retention", "expected"),
        [("in_memory", None), ("24h", "1h"), ("1h", "1h"), ("5m", "5m"), (None, None)],
    )
    def test_retention_maps_to_bedrock_ttl(
        self, retention: str | None, expected: str | None
    ) -> None:
        """The two upstream values (``in_memory``, ``24h``) resolve without error."""
        assert resolve_cache_ttl(retention) == expected  # type: ignore[arg-type]

    def test_every_retention_value_is_covered(self) -> None:
        """The parametrized cases enumerate the whole ``CACHE_TTL`` table."""
        assert set(CACHE_TTL) == {"in_memory", "24h", "1h", "5m"}


class TestPromptCacheKeySelector:
    """``prompt_cache_key`` doubles as a dot-separated cache-component selector.

    Upstream treats the key as an opaque cache bucket; this gateway additionally
    reads recognised tokens as the set of prompt sections to mark with a Bedrock
    ``cachePoint``, which directly changes what is billed as a cache write.

    Ref: https://developers.openai.com/api/docs/guides/prompt-caching
         stdapi/models/chat/_adapters/_openai_common.py:parse_prompt_cache_key
    """

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("system.tools", {"system", "tools"}),
            ("messages", {"messages"}),
            ("system.messages.tools", {"system", "messages", "tools"}),
            ("system.not-a-component", {"system"}),
        ],
    )
    def test_named_components_are_selected(self, key: str, expected: set[str]) -> None:
        """Recognised tokens select exactly their components; others are dropped."""
        assert parse_prompt_cache_key(key) == expected

    @pytest.mark.parametrize("key", ["opaque-hash", "default"])
    def test_unrecognised_key_enables_every_component(self, key: str) -> None:
        """A key with no recognisable token falls back to caching everything.

        This is the branch an upstream client hits, since OpenAI's own
        ``prompt_cache_key`` is an arbitrary bucketing string.
        """
        assert parse_prompt_cache_key(key) == PROMPT_CACHING

    @pytest.mark.parametrize("key", [None, ""])
    def test_absent_key_disables_caching(self, key: str | None) -> None:
        """No key means no cache point is written at all."""
        assert parse_prompt_cache_key(key) == set()


class TestBuildOutputConfig:
    """``response_format`` becomes the Bedrock ``outputConfig`` JSON schema string.

    Bedrock takes the schema as a serialized string, so a mapping is encoded once
    and an already-serialized schema is passed through instead of being encoded
    twice.

    Ref: https://developers.openai.com/api/docs/guides/structured-outputs
         stdapi/models/chat/_adapters/_openai_chat_completion.py:build_output_config
    """

    def test_mapping_schema_is_serialized_once(self) -> None:
        """A dict schema is emitted as its compact JSON serialization."""
        response_format = ResponseFormatJSONSchema.model_validate(
            {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": {"type": "object"}},
            }
        )
        assert build_output_config(response_format) == {
            "schema": '{"type":"object"}',
            "name": "answer",
        }

    def test_string_schema_is_passed_through_verbatim(self) -> None:
        """A pre-serialized schema string reaches Bedrock unaltered.

        ``JSONSchema.schema_`` is typed as a mapping, so this branch is defensive:
        it is reachable only from a model built without validation, and re-encoding
        the string would hand Bedrock a quoted string instead of a schema.
        """
        response_format = ResponseFormatJSONSchema.model_construct(
            type="json_schema",
            json_schema=JSONSchema.model_construct(
                name="answer", schema_='{"type":"object"}'
            ),
        )
        assert build_output_config(response_format) == {
            "schema": '{"type":"object"}',
            "name": "answer",
        }

    def test_json_object_and_text_formats(self) -> None:
        """``json_object`` and plain text both send no outputConfig schema.

        Bedrock's strict structured output has no schema for "any JSON object": an
        empty schema is rejected, and the only closed alternative admits only
        ``{}`` (issue #96). Skipping outputConfig for ``json_object`` avoids
        constraining the model to an empty response.
        """
        assert build_output_config(ResponseFormatJSONObject(type="json_object")) is None
        assert build_output_config(ResponseFormatText(type="text")) is None
        assert build_output_config(None) is None


class TestEnforceJsonObject:
    """``enforce_json_object`` appends a JSON-only system instruction on request.

    Since Bedrock's ``outputConfig`` has no schema for ``json_object`` (issue
    #96), this system-prompt nudge is the substitute enforcement -- and must
    never disturb an existing system prompt.

    Ref: stdapi/models/chat/_adapters/_openai_common.py:enforce_json_object
    """

    def test_appends_instruction_when_requested(self) -> None:
        """A new block is appended, leaving the existing prompt untouched."""
        system_blocks: list[Any] = [{"text": "You are a helpful assistant."}]
        enforce_json_object(system_blocks, requested=True)
        assert system_blocks == [
            {"text": "You are a helpful assistant."},
            JSON_OBJECT_SYSTEM_INSTRUCTION,
        ]

    def test_no_op_when_not_requested(self) -> None:
        """``requested=False`` leaves ``system_blocks`` unchanged (e.g. json_schema)."""
        system_blocks: list[Any] = [{"text": "You are a helpful assistant."}]
        enforce_json_object(system_blocks, requested=False)
        assert system_blocks == [{"text": "You are a helpful assistant."}]

    def test_appends_to_empty_system_blocks(self) -> None:
        """No prior system prompt still gets the instruction appended."""
        system_blocks: list[Any] = []
        enforce_json_object(system_blocks, requested=True)
        assert system_blocks == [JSON_OBJECT_SYSTEM_INSTRUCTION]


class TestTopKForwarding:
    """``top_k`` is a declared field routed through the inference configuration.

    ``top_k`` is a Qwen-compatible gateway extra; being declared (rather than an
    unknown key) is what keeps it out of ``model_extra`` and routes it through
    ``set_inference_configuration``.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         stdapi/models/chat/_adapters/_openai_chat_completion.py:translate_request
    """

    @staticmethod
    def _translate(payload: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        """Translate a request and return its inference config and extra fields.

        Args:
            payload: Raw chat completion request payload.

        Returns:
            The Bedrock ``inferenceConfig`` and ``additionalModelRequestFields``.
        """
        request = CompletionCreateParams.model_validate(payload)
        assert request.model_extra == {}, "top_k must be a declared field"
        inference_config, additional_fields, *_ = translate_request(
            request, "anthropic.claude-haiku-4-5-20251001-v1:0"
        )
        return inference_config, additional_fields

    def test_top_k_reaches_the_model_request(self) -> None:
        """``top_k`` is forwarded as an ``additionalModelRequestFields`` entry.

        Converse has no ``topK`` in ``inferenceConfig``, so the value travels in
        the model-specific fields rather than being dropped.
        """
        _, additional_fields = self._translate(
            {
                "model": "model",
                "messages": [{"role": "user", "content": "hi"}],
                "top_k": 7,
            }
        )
        assert additional_fields["top_k"] == 7

    def test_top_k_absent_adds_no_field(self) -> None:
        """Omitting ``top_k`` leaves the model request fields untouched."""
        _, additional_fields = self._translate(
            {"model": "model", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert "top_k" not in additional_fields


#: Bedrock output content block carrying model-native audio.
_NATIVE_AUDIO_BLOCK: dict[str, Any] = {
    "audio": {"format": "mp3", "source": {"bytes": b"RAWAUDIO"}}
}


class TestAudioOutputEnvelope:
    """The ``ChatCompletionAudio`` envelope identifies and dates each audio choice.

    Ref: https://developers.openai.com/api/docs/guides/audio
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
         stdapi/models/chat/_adapters/_openai_chat_completion.py:_get_or_generate_audio
    """

    @staticmethod
    def _audio_params() -> ChatCompletionAudioParam:
        """Return an MP3/alloy audio request parameter object.

        Returns:
            The audio output configuration.
        """
        return ChatCompletionAudioParam(voice="alloy", format="mp3")

    async def test_model_native_audio_block_is_returned_verbatim(self) -> None:
        """A Bedrock ``audio`` content block bypasses TTS and is returned as-is.

        No speech synthesis is reachable in this test process, so a fall-through
        to the Polly path would fail rather than silently substitute a different
        waveform.
        """
        contents: list[Any] = [{"text": "hello"}, _NATIVE_AUDIO_BLOCK]
        audio = await _get_or_generate_audio(
            self._audio_params(), contents, "chatcmpl-1", "hello", 1234, 0
        )
        assert b64decode(audio.data) == b"RAWAUDIO"
        assert audio.transcript == "hello", "the text output stays the transcript"

    async def test_audio_id_is_unique_per_choice_and_expiry_is_the_request_time(
        self,
    ) -> None:
        """Each choice gets its own ``audio-<completion id>-<index>`` handle.

        ``expires_at`` is set to the completion's ``created`` timestamp, so a
        client replaying an audio id by that field always sees an elapsed expiry.
        """
        contents: list[Any] = [_NATIVE_AUDIO_BLOCK]
        first = await _get_or_generate_audio(
            self._audio_params(), contents, "chatcmpl-1", "hello", 1234, 0
        )
        second = await _get_or_generate_audio(
            self._audio_params(), contents, "chatcmpl-1", "hello", 1234, 1
        )
        assert first.id == "audio-chatcmpl-1-0"
        assert second.id == "audio-chatcmpl-1-1"
        assert first.id != second.id, "n>1 choices must not share an audio id"
        assert first.expires_at == 1234
