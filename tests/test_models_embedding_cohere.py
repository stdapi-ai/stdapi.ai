"""Tests for the Cohere embedding model's Bedrock response parsing."""

from typing import Any

import pytest

from stdapi.api_errors import ApiError
from stdapi.models import InvokeResult
from stdapi.models.embedding.cohere_embed import EmbeddingModel


@pytest.mark.local
class TestCohereEmbeddingModelImages:
    """EmbeddingModel.embed_text: parsing of the Bedrock `images` metadata field."""

    async def test_images_metadata_is_surfaced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Bedrock response with `images` is parsed into `EmbeddingResponse.images`."""
        model = EmbeddingModel("cohere.embed-v4:0")

        async def _invoke(
            _body: dict[str, Any],
            **_kwargs: Any,  # noqa: ANN401
        ) -> InvokeResult[Any]:
            return InvokeResult(
                response={
                    "id": "req-1",
                    "response_type": "embeddings_floats",
                    "embeddings": [[0.1, 0.2]],
                    "texts": [],
                    "images": [
                        {"format": "png", "width": 10, "height": 20, "bit_depth": 8}
                    ],
                },
                input_tokens=3,
                output_tokens=0,
            )

        monkeypatch.setattr(model, "invoke", _invoke)

        response = await model.embed_text(
            ["dummy-image"], dimensions=None, extra_params={}
        )

        assert response.images is not None
        assert [image.model_dump() for image in response.images] == [
            {"format": "png", "width": 10, "height": 20, "bit_depth": 8}
        ]

    async def test_no_images_metadata_leaves_field_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Bedrock response without `images` leaves `EmbeddingResponse.images` unset."""
        model = EmbeddingModel("cohere.embed-multilingual-v3")

        async def _invoke(
            _body: dict[str, Any],
            **_kwargs: Any,  # noqa: ANN401
        ) -> InvokeResult[Any]:
            return InvokeResult(
                response={
                    "id": "req-2",
                    "response_type": "embeddings_floats",
                    "embeddings": [[0.1, 0.2]],
                    "texts": ["hello"],
                    "images": [],
                },
                input_tokens=1,
                output_tokens=0,
            )

        monkeypatch.setattr(model, "invoke", _invoke)

        response = await model.embed_text(["hello"], dimensions=None, extra_params={})

        assert response.images is None


@pytest.mark.local
class TestCohereEmbeddingModelQuantizedTypes:
    """EmbeddingModel.embed_text: parsing of by-type quantized embeddings."""

    async def test_by_type_response_is_surfaced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dict `embeddings` response is split into `embeddings`/`embeddings_by_type`."""
        model = EmbeddingModel("cohere.embed-v4:0")

        async def _invoke(
            _body: dict[str, Any],
            **_kwargs: Any,  # noqa: ANN401
        ) -> InvokeResult[Any]:
            return InvokeResult(
                response={
                    "id": "req-3",
                    "response_type": "embeddings_floats",
                    "embeddings": {"float": [[0.1, 0.2]], "int8": [[1, 2]]},
                    "texts": ["hello"],
                    "images": [],
                },
                input_tokens=1,
                output_tokens=0,
            )

        monkeypatch.setattr(model, "invoke", _invoke)

        response = await model.embed_text(["hello"], dimensions=None, extra_params={})

        assert response.embeddings == [[0.1, 0.2]]
        assert response.embeddings_by_type == {"float": [[0.1, 0.2]], "int8": [[1, 2]]}

    async def test_by_type_response_without_float_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dict `embeddings` response without a `float` key still errors.

        The Cohere-compatible routes always request `float` alongside other
        types, so a missing `float` key signals an unsupported request made
        through another route's extra-params passthrough.
        """
        model = EmbeddingModel("cohere.embed-v4:0")

        async def _invoke(
            _body: dict[str, Any],
            **_kwargs: Any,  # noqa: ANN401
        ) -> InvokeResult[Any]:
            return InvokeResult(
                response={
                    "id": "req-4",
                    "response_type": "embeddings_floats",
                    "embeddings": {"int8": [[1, 2]]},
                    "texts": ["hello"],
                    "images": [],
                },
                input_tokens=1,
                output_tokens=0,
            )

        monkeypatch.setattr(model, "invoke", _invoke)

        with pytest.raises(ApiError, match="float"):
            await model.embed_text(["hello"], dimensions=None, extra_params={})
