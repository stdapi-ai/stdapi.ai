"""Unit tests for mid-conversation system message preparation (no AWS calls)."""

from __future__ import annotations

import pytest

from stdapi.models.chat._adapters._anthropic_message import _prepare_messages_and_system
from stdapi.types.anthropic_messages import MessageParam, TextBlockParam

pytestmark = pytest.mark.local


def _message(role: str, text: str) -> MessageParam:
    """Return a message with a single string content."""
    return MessageParam.model_validate({"role": role, "content": text})


def _conversation() -> list[MessageParam]:
    """Return a conversation with a historical and a trailing system directive."""
    return [
        _message("user", "Hello."),
        _message("system", "Answer in one word."),
        _message("assistant", "Hi."),
        _message("system", "Now be verbose."),
        _message("user", "How are you?"),
    ]


class TestPrepareMessagesAndSystem:
    """Placement-aware split between forwarded and folded system messages."""

    def test_historical_directive_is_forwarded(self) -> None:
        """A ``user -> system -> assistant`` directive stays in the message list."""
        messages, system = _prepare_messages_and_system(
            _conversation(), "Be helpful.", system_message_as_messages=True
        )
        assert [message.role for message in messages] == [
            "user",
            "system",
            "assistant",
            "user",
        ]
        assert system == [
            TextBlockParam(type="text", text="Be helpful."),
            TextBlockParam(type="text", text="Now be verbose."),
        ]

    def test_trailing_directive_is_folded(self) -> None:
        """A directive not followed by an assistant turn folds into the system field."""
        messages, system = _prepare_messages_and_system(
            [
                _message("user", "How are you?"),
                _message("system", "Answer in one word."),
            ],
            None,
            system_message_as_messages=True,
        )
        assert [message.role for message in messages] == ["user"]
        assert system == [TextBlockParam(type="text", text="Answer in one word.")]

    def test_leading_directive_is_folded(self) -> None:
        """A directive placed before the first user turn folds into the system field."""
        messages, system = _prepare_messages_and_system(
            [
                _message("system", "Answer in one word."),
                _message("user", "Hello."),
                _message("assistant", "Hi."),
                _message("user", "How are you?"),
            ],
            None,
            system_message_as_messages=True,
        )
        assert [message.role for message in messages] == ["user", "assistant", "user"]
        assert system == [TextBlockParam(type="text", text="Answer in one word.")]

    def test_unsupported_model_folds_every_directive(self) -> None:
        """Without native support every system message merges into the system field."""
        messages, system = _prepare_messages_and_system(
            _conversation(), "Be helpful.", system_message_as_messages=False
        )
        assert [message.role for message in messages] == ["user", "assistant", "user"]
        assert system == [
            TextBlockParam(type="text", text="Be helpful."),
            TextBlockParam(type="text", text="Answer in one word."),
            TextBlockParam(type="text", text="Now be verbose."),
        ]

    def test_forwarded_directive_keeps_its_block_content(self) -> None:
        """A forwarded directive is passed through untouched, blocks included."""
        directive = MessageParam.model_validate(
            {
                "role": "system",
                "content": [{"type": "text", "text": "Answer in one word."}],
            }
        )
        messages, system = _prepare_messages_and_system(
            [
                _message("user", "Hello."),
                directive,
                _message("assistant", "Hi."),
                _message("user", "How are you?"),
            ],
            None,
            system_message_as_messages=True,
        )
        assert messages[1] is directive
        assert system is None
