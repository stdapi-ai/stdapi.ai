"""Basic tests for Cohere embedding models via OpenAI-compatible embeddings API."""

import pytest
from openai import OpenAI

COHERE_V3 = "cohere.embed-english-v3"
COHERE_V4 = "cohere.embed-v4:0"

COHERE_ALL = (COHERE_V3, COHERE_V4)
COHERE_SAMPLE = (COHERE_V4,)


class TestCohereEmbeddings:
    """Basic behavior checks for Cohere embeddings family (V4)."""

    @pytest.mark.parametrize("model_id", COHERE_SAMPLE)
    def test_text_extra_params_truncate(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Extra body parameter "truncate" is forwarded to provider.

        Not part of OpenAI Embeddings API, but accepted here as an extra body field.
        """
        if use_openai_api:
            pytest.skip("Cohere models are not available on the official OpenAI API")

        truncate_value = "START" if model_id.endswith("v3") else "LEFT"
        response = openai_client.embeddings.create(
            model=model_id,
            input="The quick brown fox.",
            extra_body={"truncate": truncate_value},
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) > 0

    @pytest.mark.parametrize("model_id", COHERE_ALL)
    def test_text_single(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Text input returns a valid embedding."""
        if use_openai_api:
            pytest.skip("Cohere models are not available on the official OpenAI API")

        response = openai_client.embeddings.create(
            model=model_id, input="The quick brown fox jumps over the lazy dog."
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) > 0

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", COHERE_ALL)
    def test_image_single(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """Image input (data URI) returns a valid embedding."""
        if use_openai_api:
            pytest.skip("Cohere models are not available on the official OpenAI API")

        response = openai_client.embeddings.create(
            model=model_id, input=sample_image_file_base64
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) > 0

    @pytest.mark.parametrize("model_id", COHERE_ALL)
    def test_text_batch(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Batch of text inputs returns one embedding per item."""
        if use_openai_api:
            pytest.skip("Cohere models are not available on the official OpenAI API")

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

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", COHERE_SAMPLE)
    def test_image_batch(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """Batch of image data URIs returns embeddings for all items."""
        if use_openai_api:
            pytest.skip("Cohere models are not available on the official OpenAI API")

        inputs = [
            sample_image_file_base64,
            sample_image_file_base64,
            sample_image_file_base64,
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
    @pytest.mark.parametrize("model_id", COHERE_SAMPLE)
    def test_mixed_text_image_batch(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """Mixed batch of text and image inputs should be handled.

        Some backends may not support mixed batches and can return 400. In that
        case, this is accepted behavior.
        """
        if use_openai_api:
            pytest.skip("Cohere models are not available on the official OpenAI API")

        inputs = ["A sample image.", sample_image_file_base64]
        response = openai_client.embeddings.create(model=model_id, input=inputs)
        assert response.object == "list"
        assert len(response.data) == len(inputs)
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) > 0

    @pytest.mark.parametrize("model_id", COHERE_SAMPLE)
    def test_dimensions_supported_when_valid(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Dimensions parameter is honored when supported; otherwise 400 is acceptable.

        The Cohere implementation supports output_dimension; for an unsupported
        value the server may raise 400, which aligns with OpenAI behavior.
        """
        if use_openai_api:
            pytest.skip("Cohere models are not available on the official OpenAI API")

        dimensions = 512
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
        assert len(item.embedding) > 0
        # If respected exactly, length must match; otherwise just ensure non-empty
        assert len(item.embedding) == dimensions
