"""TwelveLabs Pegasus video understanding on the OpenAI /v1/responses route.

Pegasus answers InvokeModel only: the gateway resolves the request through the
normal Converse pipeline, then rewrites it into Pegasus's
``{inputPrompt, mediaSource}`` body and maps the ``{message, finishReason}``
reply back onto a Responses object.

Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
     stdapi/models/chat/twelvelabs_pegasus.py:ChatModel
"""

import json
from typing import TYPE_CHECKING, cast

import pytest
from openai import BadRequestError, OpenAI

if TYPE_CHECKING:
    from openai import OpenAI, Stream
    from openai.types.responses import ResponseStreamEvent

PEGASUS_MODEL = "twelvelabs.pegasus-1-2-v1:0"


class TestTwelveLabsPegasusResponses:
    """Video input, streaming, structured output and unsupported-feature handling."""

    @pytest.mark.expensive
    @pytest.mark.slow
    def test_video_basic(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """A video part plus a text prompt returns one assistant message item.

        The Responses input union has no video content part, so the video is sent
        as an ``input_image`` data URL and the gateway forwards it as Pegasus's
        ``mediaSource.base64String``.  Pegasus replies with a single ``message``
        string, hence exactly one output item with one ``output_text`` part.

        Ref: https://developers.openai.com/api/docs/guides/file-inputs
             stdapi/models/chat/twelvelabs_pegasus.py:_video_to_media_source
        """
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")
        if not sample_video_file_base64:
            pytest.skip("No sample video available")

        video_url = sample_video_file_base64
        response = openai_client.responses.create(
            model=PEGASUS_MODEL,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [  # type: ignore[misc, list-item]
                        {"type": "input_image", "image_url": video_url},
                        {
                            "type": "input_text",
                            "text": "Describe what happens in this video.",
                        },
                    ],
                }
            ],
        )
        assert response.output_text
        assert response.error is None

        messages = [item for item in response.output if item.type == "message"]
        assert len(messages) == 1, (
            f"Pegasus returns a single message, got: {[i.type for i in response.output]}"
        )
        assert messages[0].role == "assistant"
        assert messages[0].status == "completed"
        text_part = messages[0].content[0]
        assert text_part.type == "output_text"
        assert response.output_text == text_part.text

        assert response.usage is not None
        assert response.usage.total_tokens == (
            response.usage.input_tokens + response.usage.output_tokens
        )

    @pytest.mark.expensive
    @pytest.mark.slow
    def test_video_streaming(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """Streaming a video prompt emits the full lifecycle envelope with text deltas.

        Pegasus has no ConverseStream: its delta chunks are re-shaped into Bedrock
        ``contentBlockDelta``/``messageStop`` events, which the Responses adapter
        renders as ``response.created`` → deltas → ``response.completed`` with a
        gap-free ``sequence_number`` series.  Token counts are not asserted: the
        Pegasus stream reports 0 input tokens by design.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._format_converse_stream
             stdapi/models/chat/_adapters/_openai_responses.py:format_stream
        """
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")
        if not sample_video_file_base64:
            pytest.skip("No sample video available")

        video_url = sample_video_file_base64
        stream = cast(
            "Stream[ResponseStreamEvent]",
            openai_client.responses.create(
                model=PEGASUS_MODEL,
                input=[
                    {
                        "type": "message",
                        "role": "user",
                        "content": [  # type: ignore[misc, list-item]
                            {"type": "input_image", "image_url": video_url},
                            {
                                "type": "input_text",
                                "text": "Describe what happens in this video.",
                            },
                        ],
                    }
                ],
                stream=True,
            ),
        )
        chunks = list(stream)
        accumulated = "".join(
            event.delta
            for event in chunks
            if event.type == "response.output_text.delta"
        )
        assert accumulated, "No response.output_text.delta events received"
        assert chunks[0].type == "response.created"
        assert [event.sequence_number for event in chunks] == list(
            range(len(chunks))
        ), "sequence_number must start at 0 and increase by one per event"

        terminal = chunks[-1]
        assert terminal.type == "response.completed"
        assert terminal.response.status == "completed"
        assert terminal.response.output_text == accumulated
        assert terminal.response.usage is not None

    def test_no_video_returns_400(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """A text-only request is rejected as an ``invalid_request_error``.

        Pegasus takes exactly one video per call, so the gateway scans the resolved
        messages for a video block and raises before reaching Bedrock.  This is a
        gateway-side rejection: no InvokeModel call is billed.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._build_pegasus_body
        """
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")

        with pytest.raises(BadRequestError) as excinfo:
            openai_client.responses.create(
                model=PEGASUS_MODEL,
                input=[{"type": "message", "role": "user", "content": "Hello"}],
            )
        assert excinfo.value.status_code == 400
        assert excinfo.value.type == "invalid_request_error"
        assert excinfo.value.code == "invalid_request_error"
        assert "video" in str(excinfo.value).lower(), (
            f"Error does not mention the missing video: {excinfo.value}"
        )

    @pytest.mark.expensive
    @pytest.mark.slow
    def test_response_format_json_schema(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """``text.format=json_schema`` makes Pegasus return an object matching the schema.

        The gateway maps ``text.format`` onto Pegasus's
        ``responseFormat.jsonSchema``; the model answers with the JSON document as
        a plain string in ``message``, so the schema is checked by parsing
        ``output_text``.

        Ref: https://developers.openai.com/api/docs/guides/structured-outputs
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._build_pegasus_body
        """
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")
        if not sample_video_file_base64:
            pytest.skip("No sample video available")

        video_url = sample_video_file_base64
        response = openai_client.responses.create(  # type: ignore[call-overload]
            model=PEGASUS_MODEL,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": video_url},
                        {"type": "input_text", "text": "Describe this video."},
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "result",
                    "schema": {
                        "type": "object",
                        "properties": {"description": {"type": "string"}},
                        "required": ["description"],
                    },
                }
            },
        )
        assert response.status == "completed"
        assert response.output_text
        payload = json.loads(response.output_text)
        assert isinstance(payload, dict), f"Expected a JSON object, got {payload!r}"
        assert isinstance(payload.get("description"), str), (
            f"Required schema property missing from {payload!r}"
        )

    @pytest.mark.expensive
    @pytest.mark.slow
    def test_tools_silently_ignored(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """Function tools are dropped rather than rejected, and never called.

        Pegasus has no tool-use support, so the gateway keeps cross-model client
        code working by ignoring ``tools`` instead of returning a 400: the request
        succeeds and the output holds a message item only, never a
        ``function_call``.

        Ref: https://developers.openai.com/api/docs/guides/function-calling
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._build_pegasus_body
        """
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")
        if not sample_video_file_base64:
            pytest.skip("No sample video available")

        video_url = sample_video_file_base64
        response = openai_client.responses.create(
            model=PEGASUS_MODEL,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [  # type: ignore[misc, list-item]
                        {"type": "input_image", "image_url": video_url},
                        {"type": "input_text", "text": "Describe this video."},
                    ],
                }
            ],
            tools=[
                {  # type: ignore[list-item]
                    "type": "function",
                    "name": "test_func",
                    "description": "A test function",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        )
        assert response.output_text
        item_types = [item.type for item in response.output]
        assert "message" in item_types, f"No message item in output: {item_types}"
        assert "function_call" not in item_types, (
            f"Pegasus cannot call tools, got: {item_types}"
        )
