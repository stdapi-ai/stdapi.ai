"""Basic tests for TwelveLabs Marengo embedding models via OpenAI-compatible API."""

import pytest
from openai import BadRequestError, OpenAI

MARANGO_V2 = "twelvelabs.marengo-embed-2-7-v1:0"

MARANGO_ALL = (MARANGO_V2,)
MARANGO_SAMPLE = (MARANGO_V2,)


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

    @pytest.mark.parametrize("model_id", MARANGO_SAMPLE)
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
