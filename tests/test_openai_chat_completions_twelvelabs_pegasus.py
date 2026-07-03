"""Tests for TwelveLabs Pegasus chat completions."""

import base64
import json

import pytest
from openai import BadRequestError, OpenAI

PEGASUS_MODEL = "twelvelabs.pegasus-1-2-v1:0"
PEGASUS_INLINE_BYTES = 18_874_368


class TestTwelveLabsPegasusChatCompletions:
    """Chat completions tests for TwelveLabs Pegasus video model."""

    @pytest.mark.expensive
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
        resp = openai_client.chat.completions.create(
            model=PEGASUS_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": video_url}},
                        {
                            "type": "text",
                            "text": "Describe what happens in this video.",
                        },
                    ],
                }
            ],
        )
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.role == "assistant"
        assert resp.choices[0].message.content
        assert resp.choices[0].finish_reason in {"stop", "length"}
        assert resp.usage is not None
        assert resp.usage.total_tokens >= 0

    @pytest.mark.expensive
    def test_video_streaming(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """Video input with streaming yields delta chunks."""
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")
        if not sample_video_file_base64:
            pytest.skip("No sample video available")

        video_url = sample_video_file_base64
        chunks = list(
            openai_client.chat.completions.create(
                model=PEGASUS_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": video_url}},
                            {
                                "type": "text",
                                "text": "Describe what happens in this video.",
                            },
                        ],
                    }
                ],
                stream=True,
            )
        )
        assert len(chunks) >= 1
        has_delta_content = any(
            chunk.choices[0].delta.content for chunk in chunks if chunk.choices
        )
        assert has_delta_content
        has_finish_reason = any(
            chunk.choices[0].finish_reason for chunk in chunks if chunk.choices
        )
        assert has_finish_reason

    @pytest.mark.expensive
    def test_video_in_assistant_history(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """Video in first user message with assistant history allows follow-up."""
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")
        if not sample_video_file_base64:
            pytest.skip("No sample video available")

        video_url = sample_video_file_base64
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
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.content

    @pytest.mark.expensive
    def test_concat_consecutive_user_text(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """Consecutive user messages are concatenated with the video message."""
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")
        if not sample_video_file_base64:
            pytest.skip("No sample video available")

        video_url = sample_video_file_base64
        resp = openai_client.chat.completions.create(
            model=PEGASUS_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "image_url", "image_url": {"url": video_url}}],
                },
                {"role": "user", "content": "Summarize the video."},
            ],
        )
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.content

    @pytest.mark.expensive
    def test_temperature_and_max_tokens_forwarded(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """Temperature and max_tokens parameters are forwarded to Pegasus."""
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")
        if not sample_video_file_base64:
            pytest.skip("No sample video available")

        video_url = sample_video_file_base64
        resp = openai_client.chat.completions.create(
            model=PEGASUS_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": video_url}},
                        {"type": "text", "text": "Describe this video in detail."},
                    ],
                }
            ],
            temperature=0,
            max_tokens=8,
        )
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.content
        # Loose bound for estimation drift
        assert resp.usage is not None
        assert resp.usage.completion_tokens <= 16

    @pytest.mark.expensive
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
        resp = openai_client.chat.completions.create(
            model=PEGASUS_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": video_url}},
                        {"type": "text", "text": "Describe this video."},
                    ],
                }
            ],
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
        content = resp.choices[0].message.content
        assert content
        # Verify it's valid JSON
        json.loads(content)

    @pytest.mark.expensive
    def test_no_video_returns_400(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Text-only message without video returns 400 error."""
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")

        with pytest.raises(BadRequestError):
            openai_client.chat.completions.create(
                model=PEGASUS_MODEL, messages=[{"role": "user", "content": "Hello"}]
            )

    @pytest.mark.expensive
    def test_system_prompt_silently_ignored(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """System prompt is silently ignored for Pegasus."""
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")
        if not sample_video_file_base64:
            pytest.skip("No sample video available")

        video_url = sample_video_file_base64
        resp = openai_client.chat.completions.create(
            model=PEGASUS_MODEL,
            messages=[
                {"role": "system", "content": "You are a test assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": video_url}},
                        {"type": "text", "text": "Describe this video."},
                    ],
                },
            ],
        )
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.content

    @pytest.mark.expensive
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
        resp = openai_client.chat.completions.create(
            model=PEGASUS_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": video_url}},
                        {"type": "text", "text": "Describe this video."},
                    ],
                }
            ],
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
        assert not resp.choices[0].message.tool_calls

    @pytest.mark.expensive
    def test_large_video_auto_s3(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_video_file_base64: str,
    ) -> None:
        """Large videos automatically use S3 upload."""
        if use_official_api:
            pytest.skip("Pegasus is not supported on the official API")
        if not sample_video_file_base64:
            pytest.skip("No sample video available")

        # Skip if raw bytes are under the threshold
        _prefix = "data:video/mp4;base64,"
        raw_bytes = base64.b64decode(sample_video_file_base64[len(_prefix) :])
        if len(raw_bytes) <= PEGASUS_INLINE_BYTES:
            pytest.skip("Sample video is too small to test auto-S3")

        video_url = sample_video_file_base64
        resp = openai_client.chat.completions.create(
            model=PEGASUS_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": video_url}},
                        {"type": "text", "text": "Describe this video."},
                    ],
                }
            ],
        )
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.content
