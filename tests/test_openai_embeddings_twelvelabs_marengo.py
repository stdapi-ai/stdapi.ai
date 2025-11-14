"""Basic tests for TwelveLabs Marengo embedding models via OpenAI-compatible API."""

import pytest
from openai import BadRequestError, OpenAI

MARANGO_V2 = "twelvelabs.marengo-embed-2-7-v1:0"
MARANGO_V3 = "twelvelabs.marengo-embed-3-0-v1:0"

MARANGO_ALL = (MARANGO_V2, MARANGO_V3)
MARANGO_SAMPLE = (MARANGO_V3,)


class TestTwelveLabsMarengoEmbeddings:
    """Basic behavior checks for TwelveLabs Marengo embeddings."""

    @pytest.mark.parametrize("model_id", MARANGO_ALL)
    def test_text_single(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Text input returns a valid embedding vector."""
        if use_openai_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
            )
        response = openai_client.embeddings.create(
            model=model_id, input="Hello from TwelveLabs Marengo embeddings."
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", MARANGO_ALL)
    def test_image_single(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """Image data URI returns a valid embedding vector."""
        if use_openai_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
            )
        response = openai_client.embeddings.create(
            model=model_id, input=sample_image_file_base64
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", MARANGO_ALL)
    def test_video_single(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_video_file_base64: str,
        model_id: str,
    ) -> None:
        """Video data URI returns a valid embedding vector."""
        if use_openai_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
            )
        if not sample_video_file_base64:
            pytest.skip(
                "Missing video sample file. Skipping test. Add a MP4 file to 'tests/.cache/video.mp4'."
            )
        response = openai_client.embeddings.create(
            model=model_id, input=sample_video_file_base64
        )
        assert response.object == "list"
        assert len(response.data) >= 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) > 0

    @pytest.mark.parametrize("model_id", [MARANGO_V2])
    def test_text_extra_params_text_truncate(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Extra body parameter textTruncate is forwarded to provider.

        Not part of the OpenAI Embeddings API; this project forwards it via body.
        """
        if use_openai_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
            )
        response = openai_client.embeddings.create(
            model=model_id,
            input="Hello from TwelveLabs Marengo embeddings.",
            extra_body={"textTruncate": "end"},
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) > 0

    @pytest.mark.parametrize("model_id", MARANGO_SAMPLE)
    def test_dimensions_unsupported_error(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Requesting dimensions must fail (model does not support this option)."""
        if use_openai_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
            )
        with pytest.raises(BadRequestError):
            openai_client.embeddings.create(
                model=model_id, input="Dims not supported.", dimensions=128
            )

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", MARANGO_ALL)
    def test_force_s3_data_with_small_image(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """Force S3 upload for small image using force_s3_data parameter.

        This tests that small files (under 6MB) can be forced to use S3 and
        async invocation via the force_s3_data extra parameter.
        """
        if use_openai_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
            )

        response = openai_client.embeddings.create(
            model=model_id,
            input=sample_image_file_base64,
            extra_body={"force_s3_data": True},
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", MARANGO_ALL)
    def test_force_s3_data_with_video(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_video_file_base64: str,
        model_id: str,
    ) -> None:
        """Force S3 upload for video using force_s3_data parameter."""
        if use_openai_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
            )
        if not sample_video_file_base64:
            pytest.skip(
                "Missing video sample file. Skipping test. Add a MP4 file to 'tests/.cache/video.mp4'."
            )

        response = openai_client.embeddings.create(
            model=model_id,
            input=sample_video_file_base64,
            extra_body={"force_s3_data": True},
        )
        assert response.object == "list"
        assert len(response.data) >= 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", MARANGO_ALL)
    def test_force_s3_data_with_audio(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_audio_mp3_file_base64: str,
        model_id: str,
    ) -> None:
        """Force S3 upload for audio using force_s3_data parameter."""
        if use_openai_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
            )

        response = openai_client.embeddings.create(
            model=model_id,
            input=sample_audio_mp3_file_base64,
            extra_body={"force_s3_data": True},
        )
        assert response.object == "list"
        assert len(response.data) >= 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", MARANGO_ALL)
    def test_force_s3_data_with_mixed_batch(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_image_file_base64: str,
        sample_video_file_base64: str,
        model_id: str,
    ) -> None:
        """Force S3 upload for mixed batch with force_s3_data parameter.

        This tests that the force_s3_data parameter works correctly with
        mixed input types (text, image, video) in a single batch.
        """
        if use_openai_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
            )
        if not sample_video_file_base64:
            pytest.skip(
                "Missing video sample file. Skipping test. Add a MP4 file to 'tests/.cache/video.mp4'."
            )

        inputs = [
            "Text description.",
            sample_image_file_base64,
            sample_video_file_base64,
        ]
        response = openai_client.embeddings.create(
            model=model_id, input=inputs, extra_body={"force_s3_data": True}
        )
        assert response.object == "list"
        # Video may return multiple segments
        assert len(response.data) >= len(inputs)
        for item in response.data:
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", [MARANGO_V3])
    def test_text_image_pair_v3(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """Text+image pair automatically uses text_image mode (v3 only).

        When exactly 2 inputs are provided where one is text and one is image,
        v3 models automatically combine them into a single text_image embedding.
        """
        if use_openai_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
            )

        # Test with text first, then image
        inputs = ["A beautiful sunset over the ocean.", sample_image_file_base64]
        response = openai_client.embeddings.create(model=model_id, input=inputs)

        assert response.object == "list"
        assert len(response.data) == 1  # Combined into single text_image embedding
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", [MARANGO_V2])
    def test_text_image_pair_not_combined_v2(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """Text+image pair NOT combined in v2 (no text_image support).

        v2 models do not support text_image mode, so text and image
        are embedded independently, returning 2 separate embeddings.
        """
        if use_openai_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
            )

        inputs = ["A beautiful sunset over the ocean.", sample_image_file_base64]
        response = openai_client.embeddings.create(model=model_id, input=inputs)

        assert response.object == "list"
        assert len(response.data) == 2  # NOT combined in v2, returns 2 embeddings
        for item in response.data:
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) > 0
