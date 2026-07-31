"""TwelveLabs Pegasus video understanding through the Anthropic /v1/messages route.

Pegasus has no Converse support: the gateway builds the Converse request as usual, then
translates it into the native InvokeModel body ``{inputPrompt, mediaSource, temperature,
maxOutputTokens}`` and maps the flat ``{message, finishReason}`` answer back onto an
Anthropic ``Message``.  One video plus one prompt per call is the whole contract, so
system prompts, tools, reasoning and prompt caching have nowhere to go and are dropped
rather than rejected.

Videos are passed as an ``image`` block with ``media_type: "video/mp4"``; the gateway's
``InputFile`` type accepts the data URI produced by ``sample_video_file_base64`` and
resolves it to a Bedrock ``video`` block.

Ref: https://platform.claude.com/docs/en/api/messages
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
     stdapi/routes/anthropic_messages.py:create_message
     stdapi/models/chat/twelvelabs_pegasus.py:ChatModel
"""

from typing import TYPE_CHECKING

import pytest
from anthropic import Anthropic, BadRequestError

from tests.conftest import PEGASUS_MODEL

if TYPE_CHECKING:
    from collections.abc import Callable

#: Skip the module when a remote Anthropic-compatible target is selected.
pytestmark = pytest.mark.gateway("Pegasus is not supported on the official API")

#: Anthropic stop_reason values reachable from Pegasus's ``finishReason`` (stop, length).
_PEGASUS_STOP_REASONS = frozenset({"end_turn", "max_tokens"})


@pytest.fixture
def video_message(sample_video_file_base64: str) -> Callable[..., dict[str, object]]:
    """Build the user message carrying the sample video.

    Videos travel as an ``image`` block with ``media_type: "video/mp4"``; the
    optional prompt becomes the trailing text block Pegasus uses as ``inputPrompt``.
    """

    def _build(prompt: str | None = None) -> dict[str, object]:
        content: list[dict[str, object]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "video/mp4",
                    "data": sample_video_file_base64,
                },
            }
        ]
        if prompt is not None:
            content.append({"type": "text", "text": prompt})
        return {"role": "user", "content": content}

    return _build


class TestTwelveLabsPegasusAnthropicMessages:
    """Anthropic Messages API tests for TwelveLabs Pegasus video model.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
         stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._build_pegasus_body
    """

    @pytest.mark.expensive
    def test_video_basic(
        self,
        anthropic_client: Anthropic,
        video_message: Callable[..., dict[str, object]],
    ) -> None:
        """A video block plus a prompt returns a single assistant text block.

        Pegasus answers with one ``message`` string, so the gateway always produces
        exactly one ``text`` block — never the multi-block content a Converse model can
        return.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._converse
        """
        response = anthropic_client.messages.create(
            model=PEGASUS_MODEL,
            max_tokens=256,
            messages=[video_message("Describe what happens in this video.")],  # type: ignore[list-item]
        )
        assert response.content
        assert response.content[0].type == "text"
        assert response.content[0].text
        assert len(response.content) == 1, (
            f"Pegasus returns one message string: {[b.type for b in response.content]}"
        )
        assert response.role == "assistant"
        assert response.model == PEGASUS_MODEL
        assert response.stop_reason in _PEGASUS_STOP_REASONS, (
            f"Unexpected stop_reason: {response.stop_reason!r}"
        )

    @pytest.mark.expensive
    def test_video_streaming(
        self,
        anthropic_client: Anthropic,
        video_message: Callable[..., dict[str, object]],
    ) -> None:
        """Video input with streaming yields delta events.

        Pegasus streams its own ``{message, stopReason}`` chunks, which
        ``_format_converse_stream`` re-frames as a single Bedrock content block so the
        Anthropic layer can emit the standard event sequence; the concatenated
        ``text_delta`` payloads must therefore reproduce the final message exactly.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._format_converse_stream
        """
        with anthropic_client.messages.stream(
            model=PEGASUS_MODEL,
            max_tokens=256,
            messages=[video_message("Describe what happens in this video.")],  # type: ignore[list-item]
        ) as stream:
            events = list(stream)
            final = stream.get_final_message()
        has_content_delta = any(
            hasattr(event, "type")
            and event.type == "content_block_delta"
            and hasattr(event.delta, "text")
            for event in events
        )
        assert has_content_delta

        has_message_stop = any(
            hasattr(event, "type") and event.type == "message_stop" for event in events
        )
        assert has_message_stop

        event_types = [event.type for event in events]
        assert event_types[0] == "message_start", (
            f"Stream must open with message_start, got: {event_types[:3]}"
        )
        assert event_types[-1] == "message_stop", (
            f"Stream must end with message_stop, got: {event_types[-3:]}"
        )
        assert "content_block_start" in event_types
        assert "content_block_stop" in event_types
        streamed_text = "".join(
            event.delta.text
            for event in events
            if event.type == "content_block_delta" and event.delta.type == "text_delta"
        )
        assert streamed_text, "Expected at least one non-empty text delta"
        assert streamed_text == "".join(
            block.text for block in final.content if block.type == "text"
        ), "Accumulated deltas must reproduce the final message text"
        assert final.stop_reason in _PEGASUS_STOP_REASONS, (
            f"Unexpected stop_reason: {final.stop_reason!r}"
        )

    @pytest.mark.expensive
    def test_video_in_assistant_history(
        self,
        anthropic_client: Anthropic,
        video_message: Callable[..., dict[str, object]],
    ) -> None:
        """A follow-up turn reuses the video sent in an earlier user message.

        ``_extract_latest_video`` walks the conversation backwards, so the video survives
        later turns, while ``_extract_latest_user_text`` only keeps the trailing run of
        user messages — the follow-up question alone becomes ``inputPrompt``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
             stdapi/models/chat/twelvelabs_pegasus.py:_extract_latest_video
        """
        response = anthropic_client.messages.create(
            model=PEGASUS_MODEL,
            max_tokens=256,
            messages=[
                video_message(),  # type: ignore[list-item]
                {"role": "assistant", "content": "I see a video."},
                {"role": "user", "content": "What else do you notice?"},
            ],
        )
        assert response.content
        assert response.content[0].type == "text"
        assert response.content[0].text
        assert response.stop_reason in _PEGASUS_STOP_REASONS, (
            f"Unexpected stop_reason: {response.stop_reason!r}"
        )

    def test_no_video_returns_400(self, anthropic_client: Anthropic) -> None:
        """A text-only request is rejected because Pegasus requires a video.

        ``mediaSource`` is mandatory in the Pegasus body, so the gateway fails the request
        itself rather than letting Bedrock reject an incomplete payload.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._build_pegasus_body
             stdapi/api_providers/anthropic.py:_format_error
        """
        with pytest.raises(BadRequestError) as exc_info:
            anthropic_client.messages.create(
                model=PEGASUS_MODEL,
                max_tokens=256,
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.type == "invalid_request_error", (
            f"Expected an invalid_request_error envelope, got: {exc_info.value.type!r}"
        )
        assert "video" in str(exc_info.value).lower(), (
            f"Error must name the missing video: {exc_info.value}"
        )

    @pytest.mark.expensive
    def test_max_tokens_forwarded(
        self,
        anthropic_client: Anthropic,
        video_message: Callable[..., dict[str, object]],
    ) -> None:
        """``max_tokens`` reaches Pegasus as ``maxOutputTokens`` and truncates the answer.

        A budget of 8 tokens cannot hold a video description, so Pegasus either reports
        ``finishReason: "length"`` — mapped to ``stop_reason="max_tokens"`` — or returns a
        text far shorter than an unbounded answer.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._build_pegasus_body
        """
        response = anthropic_client.messages.create(
            model=PEGASUS_MODEL,
            max_tokens=8,
            messages=[video_message("Describe this video briefly.")],  # type: ignore[list-item]
        )
        assert response.content
        assert response.content[0].type == "text"
        assert response.content[0].text
        text = "".join(b.text for b in response.content if b.type == "text")
        assert response.stop_reason == "max_tokens" or len(text) <= 120, (
            f"max_tokens=8 was not honored (stop_reason={response.stop_reason!r}, "
            f"{len(text)} characters): {text[:200]!r}"
        )

    @pytest.mark.expensive
    def test_tools_silently_ignored(
        self,
        anthropic_client: Anthropic,
        video_message: Callable[..., dict[str, object]],
    ) -> None:
        """``tools`` are dropped instead of failing the request on Pegasus.

        The Pegasus body has no tool field, so the ``toolConfig`` built by the shared
        adapter is discarded: the answer is plain text and can never carry a ``tool_use``
        block the client would have to serve.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._build_pegasus_body
        """
        response = anthropic_client.messages.create(
            model=PEGASUS_MODEL,
            max_tokens=256,
            messages=[video_message("Describe this video.")],  # type: ignore[list-item]
            tools=[
                {
                    "name": "test_func",
                    "description": "A test function",
                    "input_schema": {"type": "object", "properties": {}},
                }
            ],
        )
        assert response.content
        assert response.content[0].type == "text"
        assert response.content[0].text
        assert [b.type for b in response.content] == ["text"], (
            f"Expected text only, got: {[b.type for b in response.content]}"
        )
        assert response.stop_reason in _PEGASUS_STOP_REASONS, (
            f"Pegasus cannot request a tool: {response.stop_reason!r}"
        )

    @pytest.mark.expensive
    def test_system_prompt_silently_ignored(
        self,
        anthropic_client: Anthropic,
        video_message: Callable[..., dict[str, object]],
    ) -> None:
        """A ``system`` prompt is dropped instead of failing the request on Pegasus.

        ``SYSTEM_PROMPT_SUPPORTED = False`` plus the default
        ``drop_unsupported_system_prompt`` keeps the block out of the Converse request,
        and the Pegasus body has no field for it either — only ``inputPrompt``, built from
        the trailing user text, reaches the model.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel
             stdapi/config.py:Settings.drop_unsupported_system_prompt
        """
        response = anthropic_client.messages.create(
            model=PEGASUS_MODEL,
            max_tokens=256,
            system="You are a helpful assistant.",
            messages=[video_message("Describe this video.")],  # type: ignore[list-item]
        )
        assert response.content
        assert response.content[0].type == "text"
        assert response.content[0].text
        assert response.stop_reason in _PEGASUS_STOP_REASONS, (
            f"Unexpected stop_reason: {response.stop_reason!r}"
        )
