"""Basic tests for Cohere embedding models via OpenAI-compatible embeddings API."""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from stdapi.api_errors import ApiError
from stdapi.input_file import InputFileUrl
from stdapi.models import InvokeResult
from stdapi.models.embedding.cohere_embed import EmbeddingModel

if TYPE_CHECKING:
    from openai import OpenAI

COHERE_V3 = "cohere.embed-english-v3"
COHERE_V4 = "cohere.embed-v4:0"

COHERE_ALL = (COHERE_V3, COHERE_V4)
COHERE_SAMPLE = (COHERE_V4,)

#: Minimal data URI, sufficient to exercise the image branch without a real image.
_SAMPLE_IMAGE_DATA_URI = "data:image/png;base64,AAAA"


class TestCohereEmbeddings:
    """Basic behavior checks for Cohere embeddings family (V4)."""

    @pytest.mark.parametrize("model_id", COHERE_SAMPLE)
    def test_text_extra_params_truncate(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """Extra body parameter "truncate" is forwarded to provider.

        Not part of OpenAI Embeddings API, but accepted here as an extra body field.
        """
        if use_official_api:
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
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """Text input returns a valid embedding."""
        if use_official_api:
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
        use_official_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """Image input (data URI) returns a valid embedding."""
        if use_official_api:
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
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """Batch of text inputs returns one embedding per item."""
        if use_official_api:
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
        use_official_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """Batch of image data URIs returns embeddings for all items."""
        if use_official_api:
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
        use_official_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """Mixed batch of text and image inputs should be handled.

        Some backends may not support mixed batches and can return 400. In that
        case, this is accepted behavior.
        """
        if use_official_api:
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
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """Dimensions parameter is honored when supported; otherwise 400 is acceptable.

        The Cohere implementation supports output_dimension; for an unsupported
        value the server may raise 400, which aligns with OpenAI behavior.
        """
        if use_official_api:
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


@pytest.mark.local
class TestCohereEmbedFusedInputsGuard:
    """Offline checks for mixed text+image handling in the model layer."""

    async def test_mixed_input_on_v3_model_raises_clear_error(self) -> None:
        """A fused text+image request on a V3 model raises a 400 ApiError."""
        model = EmbeddingModel(COHERE_V3)
        model.invoke = AsyncMock()  # type: ignore[method-assign]
        with pytest.raises(ApiError, match="Cohere Embed v4") as exc_info:
            await model.embed_text(
                ["A sample text.", InputFileUrl(_SAMPLE_IMAGE_DATA_URI)],
                dimensions=None,
                extra_params={},
            )
        assert exc_info.value.status == 400
        model.invoke.assert_not_called()

    async def test_mixed_input_on_v4_model_builds_fused_body(self) -> None:
        """A fused text+image request on the V4 model builds the ``inputs`` body."""
        model = EmbeddingModel(COHERE_V4)
        model.invoke = AsyncMock(  # type: ignore[method-assign]
            return_value=InvokeResult(response={"embeddings": [[0.1]]})
        )
        await model.embed_text(
            ["A sample text.", InputFileUrl(_SAMPLE_IMAGE_DATA_URI)],
            dimensions=None,
            extra_params={},
        )
        request = model.invoke.call_args.args[0]
        assert request["inputs"] == [
            {"content": [{"type": "text", "text": "A sample text."}]},
            {
                "content": [
                    {"type": "image_url", "image_url": {"url": _SAMPLE_IMAGE_DATA_URI}}
                ]
            },
        ]
