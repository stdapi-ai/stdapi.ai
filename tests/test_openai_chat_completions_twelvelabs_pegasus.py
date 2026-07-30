"""Chat Completions on TwelveLabs Pegasus, a video-understanding InvokeModel model.

Pegasus has no Converse support: the gateway builds the full Converse request and then
translates it into the native body ``{inputPrompt, mediaSource, temperature,
maxOutputTokens, responseFormat}``.  One video and one prompt per call; system prompts,
tools, reasoning and prompt caching have no place in that body and are dropped.

Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
     stdapi/models/chat/twelvelabs_pegasus.py:ChatModel
"""

import base64
import json
from typing import Any

import pytest
from openai import BadRequestError, OpenAI

PEGASUS_MODEL = "twelvelabs.pegasus-1-2-v1:0"
PEGASUS_INLINE_BYTES = 18_874_368

#: finish_reason values reachable from Pegasus' ``stop`` / ``length`` finishReason.
#: Deliberately narrower than the Chat Completions enum: Pegasus reports only these two.
_FINISH_REASONS = frozenset({"stop", "length"})


def _video_messages(video_url: str, text: str) -> list[Any]:
    """Build the one user turn Pegasus accepts: one video part plus one text part.

    Args:
        video_url: Video data URI, sent as an ``image_url`` part.
        text: Prompt text, which becomes Pegasus' ``inputPrompt``.

    Returns:
        A single-message ``messages`` list.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": video_url}},
                {"type": "text", "text": text},
            ],
        }
    ]


class TestTwelveLabsPegasusChatCompletions:
    """Chat completions tests for TwelveLabs Pegasus video model.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
         stdapi/models/chat/twelvelabs_pegasus.py:ChatModel
    """

    @pytest.fixture(autouse=True)
    def _skip_official_api(self, use_official_api: bool) -> None:
        """Skip the whole class: Pegasus is a Bedrock-only model."""
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")

    @pytest.fixture
    def video_url(self, sample_video_file_base64: str) -> str:
        """The shared sample video as a data URI, skipping when it is unavailable."""
        if not sample_video_file_base64:
            pytest.skip("No sample video available")
        return sample_video_file_base64

    @pytest.mark.expensive
    def test_video_basic(self, openai_client: OpenAI, video_url: str) -> None:
        """A data-URI video plus a text part yields one assistant text choice.

        Pegasus returns a flat ``{"message", "finishReason"}`` body; ``_converse``
        re-shapes it into a Converse response so the OpenAI adapter can emit the normal
        envelope, with ``finishReason`` mapped through ``_STOP_MAP``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._converse
        """
        resp = openai_client.chat.completions.create(
            model=PEGASUS_MODEL,
            messages=_video_messages(video_url, "Describe what happens in this video."),
        )
        assert resp.object == "chat.completion"
        assert resp.id.startswith("chatcmpl-")
        assert len(resp.choices) >= 1
        choice = resp.choices[0]
        assert choice.index == 0
        assert choice.message.role == "assistant"
        assert choice.message.content
        assert choice.message.tool_calls is None
        assert choice.finish_reason in _FINISH_REASONS
        assert resp.usage is not None
        assert resp.usage.completion_tokens > 0
        assert (
            resp.usage.total_tokens
            == resp.usage.prompt_tokens + resp.usage.completion_tokens
        )

    @pytest.mark.expensive
    def test_video_streaming(self, openai_client: OpenAI, video_url: str) -> None:
        """Streaming a video prompt yields a role chunk, text deltas and one stop chunk.

        ``_format_converse_stream`` synthesises the Bedrock ``messageStart`` /
        ``contentBlockDelta`` / ``messageStop`` sequence from Pegasus' delta stream, and
        ``format_stream`` prepends the role-only chunk.  Without
        ``stream_options.include_usage`` no chunk carries usage.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._format_converse_stream
             stdapi/models/chat/_adapters/_openai_chat_completion.py:format_stream
        """
        chunks = list(
            openai_client.chat.completions.create(
                model=PEGASUS_MODEL,
                messages=_video_messages(
                    video_url, "Describe what happens in this video."
                ),
                stream=True,
            )
        )
        assert len(chunks) >= 2
        assert all(chunk.object == "chat.completion.chunk" for chunk in chunks)
        assert all(chunk.usage is None for chunk in chunks), (
            "usage must only be streamed when stream_options.include_usage is set"
        )
        assert chunks[0].choices[0].delta.role == "assistant"
        assert chunks[0].choices[0].delta.content is None
        assert "".join(
            chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices
        ), "expected at least one text delta"
        finish_reasons = [
            chunk.choices[0].finish_reason
            for chunk in chunks
            if chunk.choices and chunk.choices[0].finish_reason
        ]
        assert len(finish_reasons) == 1, (
            f"expected one stop chunk, got {finish_reasons}"
        )
        assert finish_reasons[0] in _FINISH_REASONS

    @pytest.mark.expensive
    def test_video_in_assistant_history(
        self, openai_client: OpenAI, video_url: str
    ) -> None:
        """A video from an earlier turn is reused for a later text-only question.

        ``_extract_latest_video`` scans the whole conversation in reverse, so the video
        need not be in the final message; ``_extract_latest_user_text`` stops at the
        assistant turn, so only the trailing question becomes ``inputPrompt``.  Were the
        video lookup limited to the last message this call would fail with the
        "Pegasus requires exactly one video" 400.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
             stdapi/models/chat/twelvelabs_pegasus.py:_extract_latest_video
        """
        resp = openai_client.chat.completions.create(
            model=PEGASUS_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": video_url}}],
                },
                {"role": "assistant", "content": "I see a video."},
                {"role": "user", "content": "What else do you notice?"},
            ],
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.role == "assistant"
        assert resp.choices[0].message.content
        assert resp.choices[0].finish_reason in _FINISH_REASONS
        assert resp.usage is not None
        assert resp.usage.completion_tokens > 0

    @pytest.mark.expensive
    def test_concat_consecutive_user_text(
        self, openai_client: OpenAI, video_url: str
    ) -> None:
        """Two trailing user messages are answered by a single Pegasus call.

        Pegasus accepts exactly one ``inputPrompt``, so
        ``_extract_latest_user_text`` newline-joins the text of every message in the
        trailing user run instead of rejecting the extra turn.  The join order itself is
        not observable through the response.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
             stdapi/models/chat/twelvelabs_pegasus.py:_extract_latest_user_text
        """
        resp = openai_client.chat.completions.create(
            model=PEGASUS_MODEL,
            messages=[
                *_video_messages(video_url, "Watch this video."),
                {"role": "user", "content": "Summarize the video."},
            ],
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.role == "assistant"
        assert resp.choices[0].message.content
        assert resp.choices[0].finish_reason in _FINISH_REASONS
        assert resp.usage is not None
        assert resp.usage.completion_tokens > 0

    @pytest.mark.expensive
    def test_temperature_and_max_tokens_forwarded(
        self, openai_client: OpenAI, video_url: str
    ) -> None:
        """``max_tokens`` caps the answer through ``maxOutputTokens``.

        ``_build_pegasus_body`` copies the Converse ``inferenceConfig`` fields
        ``maxTokens`` and ``temperature`` onto ``maxOutputTokens`` and ``temperature``.
        A "describe in detail" prompt limited to 8 tokens must therefore come back
        truncated; the bound is loose because Bedrock's reported count can drift from
        the requested cap.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._build_pegasus_body
        """
        resp = openai_client.chat.completions.create(
            model=PEGASUS_MODEL,
            messages=_video_messages(video_url, "Describe this video in detail."),
            temperature=0,
            max_tokens=8,
        )
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.content
        assert resp.choices[0].finish_reason in _FINISH_REASONS
        # Loose bound for estimation drift
        assert resp.usage is not None
        assert resp.usage.completion_tokens <= 16

    @pytest.mark.expensive
    def test_response_format_json_schema(
        self, openai_client: OpenAI, video_url: str
    ) -> None:
        """``response_format=json_schema`` reaches Pegasus as ``responseFormat.jsonSchema``.

        Pegasus returns structured output as a JSON string inside its ``message`` field,
        so the OpenAI ``content`` is the serialized object and must satisfy the schema —
        including the ``description`` property declared ``required``.

        Ref: https://developers.openai.com/api/docs/guides/structured-outputs
             https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._build_pegasus_body
        """
        resp = openai_client.chat.completions.create(
            model=PEGASUS_MODEL,
            messages=_video_messages(video_url, "Describe this video."),
            extra_body={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "result",
                        "schema": {
                            "type": "object",
                            "properties": {"description": {"type": "string"}},
                            "required": ["description"],
                        },
                    },
                }
            },
        )
        assert len(resp.choices) >= 1
        assert resp.choices[0].finish_reason in _FINISH_REASONS
        content = resp.choices[0].message.content
        assert content
        data = json.loads(content)
        assert isinstance(data, dict), f"expected a JSON object, got {type(data)}"
        assert isinstance(data.get("description"), str), (
            f"schema requires a string 'description', got: {data}"
        )

    def test_no_video_returns_400(self, openai_client: OpenAI) -> None:
        """A text-only conversation is rejected as ``invalid_request_error``.

        Pegasus' ``mediaSource`` is mandatory, so ``_build_pegasus_body`` raises before
        any Bedrock call when no ``video`` content block is present anywhere in the
        conversation.  The gateway reports it with OpenAI's 400 envelope, whose ``code``
        is set explicitly by the raiser.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
             https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._build_pegasus_body
        """
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.chat.completions.create(
                model=PEGASUS_MODEL, messages=[{"role": "user", "content": "Hello"}]
            )

        error = excinfo.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        assert body["code"] == "invalid_request_error"
        assert "video" in str(body["message"]).lower(), (
            f"expected the missing-video message, got: {body['message']!r}"
        )

    @pytest.mark.expensive
    def test_system_prompt_silently_ignored(
        self, openai_client: OpenAI, video_url: str
    ) -> None:
        """A ``system`` message neither errors nor blocks the video answer.

        ``SYSTEM_PROMPT_SUPPORTED = False`` drops the Converse ``system`` blocks, and the
        native Pegasus body has no field to carry them either, so the instruction is
        discarded rather than rejected.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel
        """
        resp = openai_client.chat.completions.create(
            model=PEGASUS_MODEL,
            messages=[
                {"role": "system", "content": "You are a test assistant."},
                *_video_messages(video_url, "Describe this video."),
            ],
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.role == "assistant"
        assert resp.choices[0].message.content
        assert resp.choices[0].finish_reason in _FINISH_REASONS
        assert resp.usage is not None
        assert resp.usage.completion_tokens > 0

    @pytest.mark.expensive
    def test_tools_silently_ignored(
        self, openai_client: OpenAI, video_url: str
    ) -> None:
        """A ``tools`` array is accepted but can never produce ``tool_calls``.

        The Converse ``toolConfig`` the adapter builds is simply not read by
        ``_build_pegasus_body``, so the request is answered as plain video Q&A instead of
        failing the way Bedrock would for an unsupported ``toolConfig``.

        Ref: https://developers.openai.com/api/docs/guides/function-calling
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._build_pegasus_body
        """
        resp = openai_client.chat.completions.create(
            model=PEGASUS_MODEL,
            messages=_video_messages(video_url, "Describe this video."),
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "test_func",
                        "description": "A test",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.content
        assert resp.choices[0].message.tool_calls is None
        assert resp.choices[0].finish_reason in _FINISH_REASONS
        assert resp.choices[0].finish_reason != "tool_calls"

    @pytest.mark.expensive
    def test_large_video_auto_s3(self, openai_client: OpenAI, video_url: str) -> None:
        """A video above the inline limit is uploaded to S3 and passed by reference.

        Bedrock caps an invocation payload at 25 MB, i.e. ~18.75 MB of raw bytes once
        base64-encoded, so ``_video_to_media_source`` switches from ``base64String`` to a
        temporary ``s3Location`` carrying the caller's account id as ``bucketOwner``.
        The test is skipped when the shared sample video is below that threshold.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
             stdapi/models/chat/twelvelabs_pegasus.py:_video_to_media_source
        """
        # Skip if raw bytes are under the threshold
        _prefix = "data:video/mp4;base64,"
        raw_bytes = base64.b64decode(video_url[len(_prefix) :])
        if len(raw_bytes) <= PEGASUS_INLINE_BYTES:
            pytest.skip("Sample video is too small to test auto-S3")

        resp = openai_client.chat.completions.create(
            model=PEGASUS_MODEL,
            messages=_video_messages(video_url, "Describe this video."),
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.content
        assert resp.choices[0].finish_reason in _FINISH_REASONS
        assert resp.usage is not None
        assert resp.usage.completion_tokens > 0
