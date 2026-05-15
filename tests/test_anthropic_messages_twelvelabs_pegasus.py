"""Tests for TwelveLabs Pegasus via Anthropic Messages API."""

import pytest
from anthropic import Anthropic, BadRequestError

PEGASUS_MODEL = "twelvelabs.pegasus-1-2-v1:0"


class TestTwelveLabsPegasusAnthropicMessages:
    """Anthropic Messages API tests for TwelveLabs Pegasus video model."""

    @pytest.mark.expensive
    def test_video_basic(
        self,
        anthropic_client: Anthropic,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """Basic video input returns a valid text response."""
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")
        if not sample_video_file_base64:
            pytest.skip("No sample video available")

        response = anthropic_client.messages.create(
            model=PEGASUS_MODEL,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {  # type: ignore[list-item]
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "video/mp4",
                                "data": sample_video_file_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe what happens in this video.",
                        },
                    ],
                }
            ],
        )
        assert response.content
        assert response.content[0].type == "text"
        assert response.content[0].text

    @pytest.mark.expensive
    def test_video_streaming(
        self,
        anthropic_client: Anthropic,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """Video input with streaming yields delta events."""
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")
        if not sample_video_file_base64:
            pytest.skip("No sample video available")

        with anthropic_client.messages.stream(
            model=PEGASUS_MODEL,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {  # type: ignore[list-item]
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "video/mp4",
                                "data": sample_video_file_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": "Describe what happens in this video.",
                        },
                    ],
                }
            ],
        ) as stream:
            events = list(stream)
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

    @pytest.mark.expensive
    def test_video_in_assistant_history(
        self,
        anthropic_client: Anthropic,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """Video in first user message with assistant history allows follow-up."""
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")
        if not sample_video_file_base64:
            pytest.skip("No sample video available")

        response = anthropic_client.messages.create(
            model=PEGASUS_MODEL,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {  # type: ignore[list-item]
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "video/mp4",
                                "data": sample_video_file_base64,
                            },
                        }
                    ],
                },
                {"role": "assistant", "content": "I see a video."},
                {"role": "user", "content": "What else do you notice?"},
            ],
        )
        assert response.content
        assert response.content[0].type == "text"
        assert response.content[0].text

    @pytest.mark.expensive
    def test_no_video_returns_400(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Text-only message without video returns 400 error."""
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")

        with pytest.raises(BadRequestError):
            anthropic_client.messages.create(
                model=PEGASUS_MODEL,
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
            )

    @pytest.mark.expensive
    def test_max_tokens_forwarded(
        self,
        anthropic_client: Anthropic,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """max_tokens parameter is forwarded to Pegasus."""
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")
        if not sample_video_file_base64:
            pytest.skip("No sample video available")

        response = anthropic_client.messages.create(
            model=PEGASUS_MODEL,
            max_tokens=8,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {  # type: ignore[list-item]
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "video/mp4",
                                "data": sample_video_file_base64,
                            },
                        },
                        {"type": "text", "text": "Describe this video briefly."},
                    ],
                }
            ],
        )
        assert response.content
        assert response.content[0].type == "text"
        assert response.content[0].text

    @pytest.mark.expensive
    def test_tools_silently_ignored(
        self,
        anthropic_client: Anthropic,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """Tools are silently ignored for Pegasus."""
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")
        if not sample_video_file_base64:
            pytest.skip("No sample video available")

        response = anthropic_client.messages.create(
            model=PEGASUS_MODEL,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {  # type: ignore[list-item]
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "video/mp4",
                                "data": sample_video_file_base64,
                            },
                        },
                        {"type": "text", "text": "Describe this video."},
                    ],
                }
            ],
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

    @pytest.mark.expensive
    def test_system_prompt_silently_ignored(
        self,
        anthropic_client: Anthropic,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """System prompt is silently ignored for Pegasus."""
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")
        if not sample_video_file_base64:
            pytest.skip("No sample video available")

        response = anthropic_client.messages.create(
            model=PEGASUS_MODEL,
            max_tokens=1024,
            system="You are a helpful assistant.",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {  # type: ignore[list-item]
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "video/mp4",
                                "data": sample_video_file_base64,
                            },
                        },
                        {"type": "text", "text": "Describe this video."},
                    ],
                }
            ],
        )
        assert response.content
        assert response.content[0].type == "text"
        assert response.content[0].text
