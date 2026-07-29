"""Offline unit tests for the OpenAI Chat Completions Bedrock adapter (no AWS calls)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from stdapi.config import SETTINGS
from stdapi.models.chat._adapters._openai_chat_completion import (
    _LEGACY_FUNCTION,
    format_stream,
    map_messages,
)
from stdapi.types.openai_chat_completions import (
    Audio,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
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
