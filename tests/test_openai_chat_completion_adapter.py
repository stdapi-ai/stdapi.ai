"""Offline unit tests for the OpenAI Chat Completions Bedrock adapter (no AWS calls)."""

from __future__ import annotations

import pytest

from stdapi.models.chat._adapters._openai_chat_completion import map_messages
from stdapi.types.openai_chat_completions import (
    Audio,
    ChatCompletionAssistantMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)

pytestmark = pytest.mark.local


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
