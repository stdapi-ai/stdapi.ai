"""Cohere Embed backend: Bedrock InvokeModel request body and response parsing.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html
     stdapi/models/embedding/cohere_embed.py:EmbeddingModel
"""

from typing import Any

import pytest

from stdapi.api_errors import ApiError
from stdapi.models import InvokeResult
from stdapi.models.embedding.cohere_embed import EmbeddingModel


@pytest.mark.local
class TestCohereEmbeddingModelImages:
    """EmbeddingModel.embed_text: parsing of the Bedrock `images` metadata field.

    Bedrock echoes an `images` array describing each embedded image; it is
    present but empty for text-only requests, which must not surface as image
    metadata.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html
         stdapi/models/embedding/cohere_embed.py:EmbeddingModel.embed_text
    """

    async def test_images_metadata_is_surfaced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Bedrock response with `images` is parsed into `EmbeddingResponse.images`."""
        model = EmbeddingModel("cohere.embed-v4:0")
        bodies: list[dict[str, Any]] = []

        async def _invoke(
            body: dict[str, Any],
            **_kwargs: Any,  # noqa: ANN401
        ) -> InvokeResult[Any]:
            bodies.append(body)
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

        assert bodies == [
            {"input_type": "search_document", "texts": ["dummy-image"]}
        ], "the backend defaults `input_type` and sends string inputs as `texts`"
        assert response.images is not None
        assert [image.model_dump() for image in response.images] == [
            {"format": "png", "width": 10, "height": 20, "bit_depth": 8}
        ]
        assert response.embeddings == [[0.1, 0.2]]
        assert response.embeddings_by_type is None
        assert response.prompt_tokens == 3

    async def test_no_images_metadata_leaves_field_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A Bedrock response without `images` leaves `EmbeddingResponse.images` unset.

        Bedrock returns an empty `images` list for text-only requests; the
        backend must map that to `None` so the Cohere routes omit the key.
        """
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
        assert response.embeddings == [[0.1, 0.2]]
        assert response.prompt_tokens == 1


@pytest.mark.local
class TestCohereEmbeddingModelQuantizedTypes:
    """EmbeddingModel.embed_text: parsing of by-type quantized embeddings.

    Bedrock returns `embeddings` as a flat list of float vectors, or as an
    object keyed by embedding type when `embedding_types` was sent.

    Ref: https://docs.cohere.com/v1/reference/embed
         stdapi/models/embedding/cohere_embed.py:EmbeddingModel.embed_text
    """

    async def test_by_type_response_is_surfaced(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dict `embeddings` response is split into `embeddings`/`embeddings_by_type`.

        The provider-neutral `embeddings` field keeps the `float` vectors so
        non-Cohere routes stay unaffected by a by-type request.
        """
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
        assert response.prompt_tokens == 1
        assert response.total_tokens == 1

    async def test_by_type_response_without_float_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dict `embeddings` response without a `float` key is a 400 `ApiError`.

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

        with pytest.raises(ApiError) as excinfo:
            await model.embed_text(["hello"], dimensions=None, extra_params={})

        assert excinfo.value.status == 400
        assert "`float`" in str(excinfo.value)
        assert "supported" in str(excinfo.value)


@pytest.mark.local
class TestCohereEmbeddingModelImageSources:
    """EmbeddingModel.embed_text: non-``data:`` image inputs are re-encoded before sending.

    Bedrock Cohere Embed only accepts ``image/jpeg`` or ``image/png`` data URIs,
    so the documented URL and S3-URI extension is implemented by fetching the
    bytes and re-encoding them; only the transport is stubbed here.

    Ref: https://docs.cohere.com/reference/embed
         stdapi/models/embedding/cohere_embed.py:EmbeddingModel.embed_text
         stdapi/input_file.py:_HttpSource
    """

    async def test_image_https_url_is_converted_to_a_data_uri(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ``https://`` image reaches Bedrock as a ``data:image/png;base64,`` URI.

        The Bedrock body carries no URL locator, so a regression in the fetch or
        re-encode path would send an unusable value (or the raw URL) instead.
        """
        from base64 import b64encode  # noqa: PLC0415

        from stdapi import input_file  # noqa: PLC0415
        from stdapi.input_file import InputFileUrl  # noqa: PLC0415

        png = b"\x89PNG\r\n\x1a\nstub"

        async def _resolve_metadata(source: Any) -> None:  # noqa: ANN401
            source._content_type = "image/png"  # noqa: SLF001
            source._size = len(png)  # noqa: SLF001
            source._filename = "image.png"  # noqa: SLF001

        async def _read(_source: Any) -> bytes:  # noqa: ANN401
            return png

        monkeypatch.setattr(
            input_file._HttpSource,  # noqa: SLF001
            "_resolve_metadata",
            _resolve_metadata,
        )
        monkeypatch.setattr(input_file._HttpSource, "_read", _read)  # noqa: SLF001

        model = EmbeddingModel("cohere.embed-multilingual-v3")
        bodies: list[dict[str, Any]] = []

        async def _invoke(
            body: dict[str, Any],
            **_kwargs: Any,  # noqa: ANN401
        ) -> InvokeResult[Any]:
            bodies.append(body)
            return InvokeResult(
                response={
                    "id": "req-5",
                    "response_type": "embeddings_floats",
                    "embeddings": [[0.1, 0.2]],
                    "texts": [],
                    "images": [],
                },
                input_tokens=1,
                output_tokens=0,
            )

        monkeypatch.setattr(model, "invoke", _invoke)

        response = await model.embed_text(
            [InputFileUrl("https://example.invalid/image.png")],
            dimensions=None,
            extra_params={},
        )

        (body,) = bodies
        assert body["images"] == [f"data:image/png;base64,{b64encode(png).decode()}"], (
            "the image must be inlined as a data URI, not forwarded as a URL"
        )
        assert body["input_type"] == "image", (
            "an all-image batch switches a v3 model to the image input type"
        )
        assert response.embeddings == [[0.1, 0.2]]
