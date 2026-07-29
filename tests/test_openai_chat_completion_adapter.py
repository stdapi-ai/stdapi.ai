"""Offline unit tests for the OpenAI Chat Completions Bedrock adapter (no AWS calls)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from stdapi.config import SETTINGS
from stdapi.models.chat._adapters._openai_chat_completion import (
    _LEGACY_FUNCTION,
    extract_output_text,
    format_response,
    format_stream,
    map_messages,
    translate_request,
)
from stdapi.types.openai_chat_completions import (
    Audio,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
    CompletionCreateParams,
    CompletionUsage,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.local


async def _stub_stream(events: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    """Yield the given Bedrock Converse stream event dicts one by one.

    Args:
        events: Converse stream event dicts to replay.

    Yields:
        Each event dict, in order.
    """
    for event in events:
        yield event


async def _collect_chunks(
    monkeypatch: pytest.MonkeyPatch, events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Run ``format_stream`` over stub events and return the decoded chunk payloads.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        events: Converse stream event dicts to replay.

    Returns:
        The JSON-decoded chunks, excluding the ``[DONE]`` sentinel.
    """
    monkeypatch.setattr(SETTINGS, "log_request_params", False)
    token = _LEGACY_FUNCTION.set(False)
    try:
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
    finally:
        _LEGACY_FUNCTION.reset(token)
    return [json.loads(event.data) for event in sse_events if event.data != "[DONE]"]


class TestMapMessagesRoleAlternation:
    """Consecutive messages with the same Bedrock role are merged into one turn."""

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
        """A tool result and the next user message share one Bedrock user turn."""
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
        """Consecutive tool results stay merged into a single user turn."""
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
        assert len(messages) == 1
        assert len(messages[0]["content"]) == 2

    async def test_mid_conversation_system_message_does_not_split_user_turn(
        self,
    ) -> None:
        """A system message between two user messages leaves a single user turn."""
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
    """Cache-write tokens are reported in ``prompt_tokens_details``."""

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

    async def _usage(
        self, monkeypatch: pytest.MonkeyPatch, cache_read: int, cache_write: int
    ) -> CompletionUsage:
        """Format a response with the given cache token counts and return its usage.

        Args:
            monkeypatch: Pytest monkeypatch fixture.
            cache_read: ``cacheReadInputTokens`` value.
            cache_write: ``cacheWriteInputTokens`` value.

        Returns:
            The usage of the formatted completion.
        """
        monkeypatch.setattr(SETTINGS, "log_request_params", False)
        token = _LEGACY_FUNCTION.set(False)
        try:
            completion = await format_response(
                completion_id="chatcmpl-1",
                created=0,
                model_id="model",
                responses=[self._converse_response(cache_read, cache_write)],  # type: ignore[list-item]
                service_tier=None,
                audio_params=None,
                modalities=["text"],
            )
        finally:
            _LEGACY_FUNCTION.reset(token)
        assert completion.usage is not None
        return completion.usage

    async def test_cache_write_tokens_are_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A positive cacheWriteInputTokens is exposed as cache_write_tokens."""
        usage = await self._usage(monkeypatch, cache_read=0, cache_write=7)
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cache_write_tokens == 7

    async def test_cache_read_and_write_tokens_are_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Both cache buckets are reported together."""
        usage = await self._usage(monkeypatch, cache_read=3, cache_write=7)
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 3
        assert usage.prompt_tokens_details.cache_write_tokens == 7

    async def test_details_omitted_without_cache_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No cache usage leaves prompt_tokens_details unset."""
        usage = await self._usage(monkeypatch, cache_read=0, cache_write=0)
        assert usage.prompt_tokens_details is None


class TestLegacyFunctionDetection:
    """Legacy function format detected from history survives ``translate_request``."""

    @staticmethod
    async def _legacy_flag(payload: dict[str, Any]) -> bool:
        """Map the payload messages then translate it, returning the legacy flag.

        Args:
            payload: Raw chat completion request payload.

        Returns:
            The legacy function format flag seen by the response formatters.
        """
        request = CompletionCreateParams.model_validate(payload)
        token = _LEGACY_FUNCTION.set(False)
        try:
            await map_messages(request.messages)
            translate_request(request, "model")
            return _LEGACY_FUNCTION.get()
        finally:
            _LEGACY_FUNCTION.reset(token)

    async def test_function_message_history_enables_legacy_format(self) -> None:
        """A replayed `function` message keeps the legacy format without `functions`."""
        assert await self._legacy_flag(
            {
                "model": "model",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "function", "name": "f", "content": "ok"},
                ],
            }
        )

    async def test_declared_tools_disable_legacy_format(self) -> None:
        """Declared `tools` win over legacy history detection."""
        assert not await self._legacy_flag(
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

    async def test_plain_request_keeps_tool_calls_format(self) -> None:
        """A request without legacy history or `functions` stays on tool_calls format."""
        assert not await self._legacy_flag(
            {"model": "model", "messages": [{"role": "user", "content": "hi"}]}
        )


class TestExtractOutputText:
    """Non-streaming text matches the concatenation of the streamed deltas."""

    def test_multiple_text_blocks_are_concatenated_without_separator(self) -> None:
        """Two text blocks join exactly like their streamed deltas would."""
        content, reasoning = extract_output_text(
            [{"text": "A"}, {"citationsContent": {}}, {"text": "B"}]  # type: ignore[list-item]
        )
        assert content == "AB"
        assert reasoning is None

    def test_multiple_reasoning_blocks_are_concatenated_without_separator(self) -> None:
        """Two reasoning blocks join exactly like their streamed deltas would."""
        _, reasoning = extract_output_text(
            [
                {"reasoningContent": {"reasoningText": {"text": "A"}}},
                {"reasoningContent": {"reasoningText": {"text": "B"}}},
            ]  # type: ignore[list-item]
        )
        assert reasoning == "AB"


class TestMapMessagesEmptyContent:
    """Messages yielding no content block never reach Bedrock as empty messages."""

    async def test_assistant_audio_reference_only_message_is_dropped(self) -> None:
        """An assistant turn with only an ``audio`` reference emits no Bedrock message."""
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
    """Streamed tool call indices are 0-based positions in the tool_calls array."""

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

    async def test_tool_call_index_ignores_preceding_content_blocks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tool call following a reasoning block still streams with index 0."""
        chunks = await _collect_chunks(
            monkeypatch,
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
            ],
        )
        assert self._indices(chunks) == [0, 0]

    async def test_parallel_tool_calls_are_numbered_contiguously(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two toolUse blocks after a text block stream as indices 0 and 1."""
        chunks = await _collect_chunks(
            monkeypatch,
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
            ],
        )
        assert self._indices(chunks) == [0, 1, 1]
