"""Unit tests for the non-streaming Anthropic messages response adapter (no AWS calls)."""

from __future__ import annotations

import pytest

from stdapi.models.chat._adapters._anthropic_message import _map_stop_reason
from stdapi.types.anthropic_messages import Message, Usage

pytestmark = pytest.mark.local


def test_map_stop_reason_preserves_context_window_exceeded() -> None:
    """Bedrock's ``model_context_window_exceeded`` stop reason is preserved.

    It must not be collapsed into ``max_tokens``, so clients can distinguish
    context exhaustion from the output cap.
    """
    assert _map_stop_reason("model_context_window_exceeded") == (
        "model_context_window_exceeded"
    )


def test_message_accepts_context_window_exceeded_stop_reason() -> None:
    """The ``Message`` response model validates the new stop reason.

    It must not raise a ``literal_error`` for ``model_context_window_exceeded``.
    """
    message = Message(
        id="msg_1",
        type="message",
        role="assistant",
        content=[],
        model="model-x",
        stop_reason="model_context_window_exceeded",
        usage=Usage(input_tokens=1, output_tokens=0),
    )
    assert message.stop_reason == "model_context_window_exceeded"
