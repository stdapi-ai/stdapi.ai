"""Basic tests for Amazon Titan embedding models via OpenAI-compatible API."""

import pytest
from openai import OpenAI

TITAN_V1_TEXT = "amazon.titan-embed-text-v1"
TITAN_V1_IMAGE = "amazon.titan-embed-image-v1"
TITAN_V2_TEXT = "amazon.titan-embed-text-v2:0"

TITAN_TEXT_ALL = (TITAN_V1_TEXT, TITAN_V2_TEXT)
TITAN_TEXT_SAMPLE = (TITAN_V2_TEXT,)
TITAN_IMAGE_SAMPLE = (TITAN_V1_IMAGE,)


class TestAmazonTitanEmbeddings:
    """Text-focused checks for Titan embeddings."""

    @pytest.mark.parametrize("model_id", TITAN_TEXT_SAMPLE)
    def test_text_extra_params_normalize(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Extra body parameters are passed to the backend (normalize for v2).

        OpenAI API does not support such extra parameters; this project forwards
        them to the underlying provider as extra body fields.
        """
        if use_openai_api:
            pytest.skip(
                "Amazon Titan models are not available on the official OpenAI API"
            )
        response = openai_client.embeddings.create(
            model=model_id,
            input="Hello from Titan text embeddings.",
            extra_body={"normalize": True},
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) > 0

    @pytest.mark.parametrize("model_id", TITAN_TEXT_ALL)
    def test_text_single(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Basic text embedding works and returns non-empty vector."""
        if use_openai_api:
            pytest.skip(
                "Amazon Titan models are not available on the official OpenAI API"
            )
        response = openai_client.embeddings.create(
            model=model_id, input="Hello from Titan text embeddings."
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) > 0

    @pytest.mark.parametrize("model_id", TITAN_TEXT_SAMPLE)
    def test_text_dimensions(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Dimensions parameter is accepted on text-v2 when valid (400 otherwise)."""
        if use_openai_api:
            pytest.skip(
                "Amazon Titan models are not available on the official OpenAI API"
            )

        dimensions = 256
        response = openai_client.embeddings.create(
            model=model_id, input="Dimensions parameter test.", dimensions=dimensions
        )
        assert response.object == "list"
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) == dimensions

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_IMAGE_SAMPLE)
    def test_image_single(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """Image data URI embeds successfully on the image model."""
        if use_openai_api:
            pytest.skip(
                "Amazon Titan models are not available on the official OpenAI API"
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
