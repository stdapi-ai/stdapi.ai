"""Cohere Embed backend: Bedrock InvokeModel request body and response parsing.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html
     stdapi/models/embedding/cohere_embed.py:EmbeddingModel
"""

from typing import TYPE_CHECKING, Any

import pytest

import stdapi.main  # noqa: F401  (registers every route's ROUTE_CAPABILITIES entry)
from stdapi.api_errors import ApiError
from stdapi.models import (
    InvokeResult,
    _advertised_input_modalities,
    _compute_model_capabilities,
)
from stdapi.models.embedding.cohere_embed import EmbeddingModel
from stdapi.pricing import Dimension
from stdapi.usage import USAGE, init_model_state, init_usage
from tests._helpers import make_model_details

if TYPE_CHECKING:
    from collections.abc import Generator

#: The Cohere Embed models Bedrock lists as text-only.
COHERE_EMBED_V3_MODELS = ("cohere.embed-english-v3", "cohere.embed-multilingual-v3")


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

        monkeypatch.setattr(type(model), "invoke", staticmethod(_invoke))

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

        monkeypatch.setattr(type(model), "invoke", staticmethod(_invoke))

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

        monkeypatch.setattr(type(model), "invoke", staticmethod(_invoke))

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

        monkeypatch.setattr(type(model), "invoke", staticmethod(_invoke))

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

        monkeypatch.setattr(type(model), "invoke", staticmethod(_invoke))

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


@pytest.mark.local
class TestCohereEmbeddingAdvertisedInputModalities:
    """Every Cohere Embed model advertises IMAGE input, whatever Bedrock lists.

    Bedrock declares the Embed v3 models text-only, yet both embed an image and
    are billed for it, so the listing alone would have the catalog deny a path
    the gateway serves.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html
         stdapi/models/__init__.py:_advertised_input_modalities
    """

    @pytest.mark.parametrize("model_id", COHERE_EMBED_V3_MODELS)
    def test_text_only_listing_gains_the_image_modality(self, model_id: str) -> None:
        """A text-only Bedrock listing is published with IMAGE appended.

        Ref: stdapi/models/__init__.py:_advertised_input_modalities
        """
        assert _advertised_input_modalities(model_id, ["TEXT"]) == ["TEXT", "IMAGE"]

    def test_a_declared_modality_is_not_duplicated(self) -> None:
        """Embed v4 already declares IMAGE and keeps its listing unchanged.

        Ref: stdapi/models/__init__.py:_advertised_input_modalities
        """
        assert _advertised_input_modalities("cohere.embed-v4:0", ["TEXT", "IMAGE"]) == [
            "TEXT",
            "IMAGE",
        ]

    def test_other_models_keep_their_listed_input_modalities(self) -> None:
        """A model class declaring nothing undeclared publishes the listing as it stands.

        Ref: stdapi/models/__init__.py:_advertised_input_modalities
        """
        assert _advertised_input_modalities(
            "amazon.titan-embed-text-v2:0", ["TEXT"]
        ) == ["TEXT"]

    @pytest.mark.parametrize("model_id", COHERE_EMBED_V3_MODELS)
    def test_the_image_modality_advertises_no_extra_route(self, model_id: str) -> None:
        """IMAGE input adds no route: the image routes need an IMAGE output too.

        `stdapi.main` is imported at module level so `ROUTE_CAPABILITIES` is
        really populated here, whatever ran before this test in the worker;
        otherwise both sides below compute against an empty registry and the
        comparison holds trivially, whatever the computation actually does.

        Ref: stdapi/models/__init__.py:_compute_model_capabilities
        """
        model = make_model_details(
            model_id,
            input_modalities=["TEXT", "IMAGE"],
            output_modalities=["EMBEDDING"],
        )
        routes, _tools = _compute_model_capabilities(model_id, model)
        assert routes, (
            "the registry must be populated for this comparison to mean anything"
        )
        assert "/v1/embeddings" in routes
        assert (
            routes
            == _compute_model_capabilities(
                model_id, make_model_details(model_id, output_modalities=["EMBEDDING"])
            )[0]
        )
        assert not [route for route in routes if "images" in route]


@pytest.mark.local
class TestCohereEmbeddingV3UsageRequests:
    """EmbeddingModel._record_invoke_usage: one Bedrock invocation is one usage record.

    Embed v3 bills an embedded image as its own quantity, apart from the token
    count, but both come from the same `InvokeModel` call and share the same
    `UsageKey` (same service/model/operation/region/tier/routing/context), so
    they must aggregate into one record rather than double-count `requests`.

    Ref: stdapi/usage.py:UsageKey
         stdapi/usage.py:_record_usage
         stdapi/models/embedding/cohere_embed.py:EmbeddingModel._record_invoke_usage
    """

    @pytest.fixture(autouse=True)
    def _usage_state(self) -> Generator[None]:
        """Install a fresh per-test usage map, matching request-time setup."""
        usage_token = init_usage()
        init_model_state()
        yield
        USAGE.reset(usage_token)

    def test_v3_image_beside_a_token_count_bills_one_request(self) -> None:
        """A v3 answer carrying both a token count and `images` is one record.

        Before the fix, the image quantity was recorded through a second
        `record_bedrock_usage` call sharing the base call's `UsageKey`, so
        `requests` (and therefore the reported invocation count on this
        model) came out as 2 for a single Bedrock call.
        """
        model = EmbeddingModel("cohere.embed-multilingual-v3")
        response = {
            "id": "req-usage",
            "response_type": "embeddings_floats",
            "embeddings": [[0.1, 0.2]],
            "texts": [],
            "images": [{"format": "png", "width": 1, "height": 1, "bit_depth": 8}],
        }

        model._record_invoke_usage(  # noqa: SLF001
            7, 0, response, region="us-east-1", tier=None, routing=""
        )

        (record,) = USAGE.get().values()
        assert record.requests == 1, "one Bedrock invocation must be one usage record"
        assert record.quantities[Dimension.INPUT_TOKENS] == 7
        assert record.quantities[Dimension.INPUT_IMAGES] == 1

    def test_v4_never_records_a_separate_image_quantity(self) -> None:
        """Embed v4 bills the image inside the token count, so no image quantity is added.

        `images` in the response is a v3-only billing signal; on a non-v3
        model the same field must not add an `INPUT_IMAGES` quantity.
        """
        model = EmbeddingModel("cohere.embed-v4:0")
        response = {
            "id": "req-usage-v4",
            "response_type": "embeddings_floats",
            "embeddings": [[0.1, 0.2]],
            "texts": [],
            "images": [{"format": "png", "width": 1, "height": 1, "bit_depth": 8}],
        }

        model._record_invoke_usage(  # noqa: SLF001
            9, 0, response, region="us-east-1", tier=None, routing=""
        )

        (record,) = USAGE.get().values()
        assert record.requests == 1
        assert Dimension.INPUT_IMAGES not in record.quantities
        assert record.quantities[Dimension.INPUT_TOKENS] == 9


@pytest.mark.local
class TestCohereEmbeddingV3MultiImageForwarding:
    """EmbeddingModel._build_request: no local image-count limit on Embed v3.

    Cohere Embed v3 documents a one-image-per-request limit, but nothing in
    `_build_request` enforces it locally -- Bedrock's own limit is what
    answers a request naming more than one. This pins the gateway's honest
    pass-through so a future local cap (e.g. silently trimming to the first
    image) would be caught here; whether Bedrock itself accepts or rejects
    more than one image on v3 is a live-only question this offline test
    cannot answer.

    Ref: docs/api_cohere_embed.md (model table, "one image per request")
         stdapi/models/embedding/cohere_embed.py:EmbeddingModel._build_request
    """

    async def test_v3_forwards_every_image_it_is_given(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two images sent to a v3 model reach Bedrock as two, not one."""
        from stdapi.input_file import InputFile, InputFileUrl  # noqa: PLC0415

        async def _to_data_uri(_self: InputFile) -> str:
            return "data:image/png;base64,AAA="

        monkeypatch.setattr(InputFile, "to_data_uri", _to_data_uri)

        model = EmbeddingModel("cohere.embed-english-v3")
        bodies: list[dict[str, Any]] = []

        async def _invoke(
            body: dict[str, Any],
            **_kwargs: Any,  # noqa: ANN401
        ) -> InvokeResult[Any]:
            bodies.append(body)
            return InvokeResult(
                response={
                    "id": "req-multi",
                    "response_type": "embeddings_floats",
                    "embeddings": [[0.1, 0.2], [0.3, 0.4]],
                    "texts": [],
                    "images": [
                        {"format": "png", "width": 1, "height": 1, "bit_depth": 8},
                        {"format": "png", "width": 1, "height": 1, "bit_depth": 8},
                    ],
                },
                input_tokens=2,
                output_tokens=0,
            )

        monkeypatch.setattr(type(model), "invoke", staticmethod(_invoke))

        await model.embed_text(
            [
                InputFileUrl("https://example.invalid/a.png"),
                InputFileUrl("https://example.invalid/b.png"),
            ],
            dimensions=None,
            extra_params={},
        )

        (body,) = bodies
        assert body["images"] == [
            "data:image/png;base64,AAA=",
            "data:image/png;base64,AAA=",
        ], (
            "the gateway does not cap images locally; Bedrock's own limit would answer this"
        )
        assert body["input_type"] == "image"
