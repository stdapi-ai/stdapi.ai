"""Basic tests for Amazon Nova multimodal embedding models via OpenAI-compatible API."""

import pytest
from openai import OpenAI

NOVA_V1 = "amazon.nova-2-multimodal-embeddings-v1:0"

NOVA_ALL = (NOVA_V1,)
NOVA_SAMPLE = (NOVA_V1,)


class TestAmazonNovaEmbeddings:
    """Basic behavior checks for Amazon Nova multimodal embeddings."""

    @pytest.mark.parametrize("model_id", NOVA_ALL)
    def test_text_single(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Text input returns a valid embedding vector."""
        if use_openai_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )
        response = openai_client.embeddings.create(
            model=model_id, input="Hello from Amazon Nova multimodal embeddings."
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) > 0

    @pytest.mark.parametrize("model_id", NOVA_ALL)
    def test_text_batch(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Batch of text inputs returns one embedding per item."""
        if use_openai_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        inputs = [
            "First text input for embedding.",
            "Second different sentence.",
            "Third entry to complete batch.",
        ]
        response = openai_client.embeddings.create(model=model_id, input=inputs)
        assert response.object == "list"
        assert len(response.data) == len(inputs)
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) > 0

    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_dimensions(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Dimensions parameter is accepted and returns correct size."""
        if use_openai_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        dimensions = 256
        response = openai_client.embeddings.create(
            model=model_id,
            input="Test sentence for dimensions parameter.",
            dimensions=dimensions,
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) == dimensions

    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_extra_params_embedding_purpose(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Extra body parameter embeddingPurpose is forwarded to provider.

        Nova supports GENERIC_INDEX (default), CLASSIFICATION, and CLUSTERING
        as embedding purposes. Not part of OpenAI Embeddings API.
        """
        if use_openai_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        response = openai_client.embeddings.create(
            model=model_id,
            input="Classification test sentence.",
            extra_body={"embeddingPurpose": "CLASSIFICATION"},
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_ALL)
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
                "Amazon Nova models are not available on the official OpenAI API"
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
    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_image_batch(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """Batch of image data URIs returns embeddings for all items."""
        if use_openai_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        inputs = [sample_image_file_base64, sample_image_file_base64]
        response = openai_client.embeddings.create(model=model_id, input=inputs)
        assert response.object == "list"
        assert len(response.data) == len(inputs)
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_mixed_text_image_batch(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """Mixed batch of text and image inputs returns valid embeddings.

        Nova supports multimodal embeddings in a unified semantic space,
        allowing text and images to be embedded together.
        """
        if use_openai_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        inputs = [
            "A sample text description.",
            sample_image_file_base64,
            "Another text input.",
        ]
        response = openai_client.embeddings.create(model=model_id, input=inputs)
        assert response.object == "list"
        assert len(response.data) == len(inputs)
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_ALL)
    def test_audio_single(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_audio_mp3_file_base64: str,
        model_id: str,
    ) -> None:
        """Audio data URI returns a valid embedding vector.

        Nova supports audio embeddings up to 30 seconds in duration.
        """
        if use_openai_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )
        response = openai_client.embeddings.create(
            model=model_id, input=sample_audio_mp3_file_base64
        )
        assert response.object == "list"
        assert len(response.data) >= 1  # May return multiple embeddings for segments
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_ALL)
    def test_video_single(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_video_file_base64: str,
        model_id: str,
    ) -> None:
        """Video data URI returns a valid embedding vector.

        Nova supports video embeddings up to 30 seconds in duration.
        This test is skipped if no sample video file is available.
        """
        if use_openai_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )
        if not sample_video_file_base64:
            pytest.skip(
                "Missing video sample file. Skipping test. Add a MP4 file to 'tests/.cache/video.mp4'."
            )
        response = openai_client.embeddings.create(
            model=model_id, input=sample_video_file_base64
        )
        assert response.object == "list"
        assert len(response.data) >= 1  # May return multiple embeddings for segments
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_multimodal_batch(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_image_file_base64: str,
        sample_audio_mp3_file_base64: str,
        model_id: str,
    ) -> None:
        """Batch with multiple modalities returns valid embeddings.

        Nova's unified semantic space allows embedding text, images, and audio
        in the same batch for cross-modal retrieval and comparison.
        """
        if use_openai_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        inputs = [
            "Text description of the content.",
            sample_image_file_base64,
            sample_audio_mp3_file_base64,
        ]
        response = openai_client.embeddings.create(model=model_id, input=inputs)
        assert response.object == "list"
        # Audio may return multiple segments, so total may be >= len(inputs)
        assert len(response.data) >= len(inputs)
        for item in response.data:
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_image_with_dimensions(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """Image embedding with custom dimensions parameter works correctly."""
        if use_openai_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        dimensions = 256
        response = openai_client.embeddings.create(
            model=model_id, input=sample_image_file_base64, dimensions=dimensions
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) == dimensions

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_clustering_purpose(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Clustering embedding purpose is accepted and returns valid embeddings."""
        if use_openai_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        inputs = [
            "First document for clustering.",
            "Second document for clustering.",
            "Third document for clustering.",
        ]
        response = openai_client.embeddings.create(
            model=model_id, input=inputs, extra_body={"embeddingPurpose": "CLUSTERING"}
        )
        assert response.object == "list"
        assert len(response.data) == len(inputs)
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
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
                "Amazon Nova models are not available on the official OpenAI API"
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
    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
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
                "Amazon Nova models are not available on the official OpenAI API"
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
    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
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
                "Amazon Nova models are not available on the official OpenAI API"
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
    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_force_s3_data_with_multimodal_batch(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_image_file_base64: str,
        sample_audio_mp3_file_base64: str,
        model_id: str,
    ) -> None:
        """Force S3 upload for multimodal batch with force_s3_data parameter.

        This tests that the force_s3_data parameter works correctly with
        mixed input types (text, image, audio) in a single batch.
        """
        if use_openai_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        inputs = [
            "Text description of the content.",
            sample_image_file_base64,
            sample_audio_mp3_file_base64,
        ]
        response = openai_client.embeddings.create(
            model=model_id, input=inputs, extra_body={"force_s3_data": True}
        )
        assert response.object == "list"
        # Audio may return multiple segments, so total may be >= len(inputs)
        assert len(response.data) >= len(inputs)
        for item in response.data:
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) > 0
