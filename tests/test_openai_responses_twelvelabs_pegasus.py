"""Tests for TwelveLabs Pegasus via OpenAI Responses API."""

import json
from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError, OpenAI

if TYPE_CHECKING:
    from openai import OpenAI

PEGASUS_MODEL = "twelvelabs.pegasus-1-2-v1:0"


class TestTwelveLabsPegasusResponses:
    """OpenAI Responses API tests for TwelveLabs Pegasus video model."""

    @pytest.mark.expensive
    @pytest.mark.slow
    def test_video_basic(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """Basic video input returns a valid text response."""
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

    @pytest.mark.expensive
    @pytest.mark.slow
    def test_video_streaming(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """Video input with streaming yields chunks."""
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")
        if not sample_video_file_base64:
            pytest.skip("No sample video available")

        video_url = sample_video_file_base64
        chunks = list(
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
            )
        )
        has_text_delta = any(
            getattr(event, "type", None) == "response.output_text.delta"
            and getattr(event, "delta", None)
            for event in chunks
        )
        assert has_text_delta

    def test_no_video_returns_400(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Text-only message without video returns 400 error."""
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")

        with pytest.raises(BadRequestError):
            openai_client.responses.create(
                model=PEGASUS_MODEL,
                input=[{"type": "message", "role": "user", "content": "Hello"}],
            )

    @pytest.mark.expensive
    @pytest.mark.slow
    def test_response_format_json_schema(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """Response format with json_schema returns valid JSON."""
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
        assert response.output_text
        # Verify it's valid JSON
        json.loads(response.output_text)

    @pytest.mark.expensive
    @pytest.mark.slow
    def test_tools_silently_ignored(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """Tools are silently ignored for Pegasus."""
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
