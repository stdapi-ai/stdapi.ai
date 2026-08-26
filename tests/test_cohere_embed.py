"""Cohere-compatible /v1/embed and /v2/embed routes (unit and live).

Ref: https://docs.cohere.com/reference/embed
     https://docs.cohere.com/v1/reference/embed
     stdapi/routes/cohere_embed.py:embed
     stdapi/routes/cohere_embed_v1.py:embed_v1
"""

from asyncio import run
from base64 import b64decode, b64encode
from os import getenv
from struct import unpack
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import httpx
import pytest
from cohere.errors import BadRequestError
from cohere.types import (
    EmbedImageUrl,
    EmbedInput,
    ImageUrlEmbedContent,
    TextEmbedContent,
)

from stdapi.api_errors import UnsupportedModelError
from stdapi.config import SETTINGS
from stdapi.input_file import InputFile, InputFileUrl
from stdapi.models import InvokeResult
from stdapi.models.embedding import EmbeddingImageDescription, EmbeddingResponse
from stdapi.models.embedding.cohere_embed import EmbeddingModel
from stdapi.routes import cohere_embed, cohere_embed_v1
from tests._helpers import make_model_details
from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from collections.abc import Generator

    import cohere
    from cohere.types import EmbedByTypeResponse
    from starlette.testclient import TestClient

    from stdapi.models import ModelDetails

#: Model aliases resolved by the stubbed ``validate_model``.
_MODEL_ALIASES = {"embed-multilingual": "cohere.embed-multilingual-v3"}

#: Text embedded by the live tests, echoed back in the response ``texts`` field.
_SAMPLE_TEXT = "The quick brown fox jumps over the lazy dog."

#: Output dimensions accepted by Cohere Embed v4 (default 1536).
_COHERE_V4_DIMENSIONS = frozenset({256, 512, 1024, 1536})

#: Default output dimension of Amazon Titan Text Embeddings V2.
_TITAN_V2_DIMENSIONS = 1024

#: Minimal 1x1 PNG image bytes served by the stubbed URL and S3 sources.
_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
    b"\xff?\x00\x05\xfe\x02\xfe\xa75\x81\x84\x00\x00\x00\x00IEND\xaeB`\x82"
)

#: The data URI Bedrock Cohere Embed expects for the 1x1 PNG.
_PNG_DATA_URI = f"data:image/png;base64,{b64encode(_PNG).decode()}"

#: Bucket name allow-listed by the stubbed S3 image-input test.
_S3_BUCKET = "embed-inputs"

#: Inputs a Cohere Embed model accepts per request, deliberately not enforced here.
_COHERE_MAX_INPUTS = 96


def _serve_png(monkeypatch: pytest.MonkeyPatch, source_name: str) -> None:
    """Make the named ``InputFile`` source backend yield ``_PNG`` without any I/O.

    Args:
        monkeypatch: Patch context.
        source_name: ``_HttpSource`` or ``_S3Source``.
    """
    from stdapi import input_file  # noqa: PLC0415

    async def _resolve_metadata(source: Any) -> None:  # noqa: ANN401
        source._content_type = "image/png"  # noqa: SLF001
        source._size = len(_PNG)  # noqa: SLF001
        source._filename = "image.png"  # noqa: SLF001

    async def _read(_source: Any) -> bytes:  # noqa: ANN401
        return _PNG

    source = getattr(input_file, source_name)
    monkeypatch.setattr(source, "_resolve_metadata", _resolve_metadata)
    monkeypatch.setattr(source, "_read", _read)


class _StubEmbeddingModel:
    """Stub backend recording the embed call and returning fixed vectors."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        #: Image metadata to return from the next `embed_text` call, if set.
        self.images: list[EmbeddingImageDescription] | None = None
        #: By-type embeddings to return from the next `embed_text` call, if set.
        self.embeddings_by_type: dict[str, list[list[float | int]]] | None = None

    async def embed_text(
        self, inputs: list[Any], dimensions: int | None, extra_params: dict[str, Any]
    ) -> EmbeddingResponse:
        """Record the call and return one vector per input."""
        self.calls.append(
            {"inputs": inputs, "dimensions": dimensions, "extra_params": extra_params}
        )
        return EmbeddingResponse(
            embeddings=[[0.1, 0.2]] * len(inputs),
            embeddings_by_type=self.embeddings_by_type,
            prompt_tokens=7,
            total_tokens=7,
            images=self.images,
        )


@pytest.fixture
def embed_backend(monkeypatch: pytest.MonkeyPatch) -> _StubEmbeddingModel:
    """Stub model validation and the embedding backend."""

    async def _validate_model(
        model_id: str, modality: str, *, route: str
    ) -> ModelDetails:
        assert modality == "EMBEDDING"
        # Asserted, not swallowed: the scope a wildcard model name is resolved
        # in comes from the route, so a call site that stopped passing it would
        # widen the match set silently.
        assert route in {"cohere_embed", "cohere_embed_v1"}
        if model_id == "unknown-model":
            raise UnsupportedModelError(model_id)
        resolved_id = _MODEL_ALIASES.get(model_id, model_id)
        return make_model_details(resolved_id, output_modalities=["EMBEDDING"])

    stub = _StubEmbeddingModel()
    for module in (cohere_embed, cohere_embed_v1):
        monkeypatch.setattr(module, "validate_model", _validate_model)
        monkeypatch.setattr(module, "get_embedding_model", lambda _model_id: stub)
    return stub


@pytest.fixture
def validated_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub model validation only, keeping the real embedding model classes."""

    async def _validate_model(
        model_id: str, modality: str, *, route: str
    ) -> ModelDetails:
        assert modality == "EMBEDDING"
        assert route == "cohere_embed"
        return make_model_details(model_id, output_modalities=["EMBEDDING"])

    monkeypatch.setattr(cohere_embed, "validate_model", _validate_model)


@pytest.mark.local
class TestCohereEmbedRoute:
    """POST /cohere/v2/embed: response shape and parameter mapping.

    The embedding backend is stubbed, so the assertions on
    ``embed_backend.calls`` pin the exact parameters the route forwards to
    Bedrock for each model family.

    Ref: https://docs.cohere.com/reference/embed
         stdapi/routes/cohere_embed.py:embed
    """

    def test_embed_texts_success(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """A valid request returns the Cohere v2 response shape.

        v2 always keys embeddings by type — `float` is produced even though the
        request omitted `embedding_types` — and echoes the submitted `texts`.

        Ref: stdapi/types/cohere_embed.py:build_embeddings_by_type
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-multilingual-v3",
                "input_type": "search_document",
                "texts": ["hello", "world"],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"]
        assert response.headers["x-request-id"] == body["id"]
        assert body["response_type"] == "embeddings_by_type"
        assert body["embeddings"] == {"float": [[0.1, 0.2], [0.1, 0.2]]}
        assert body["texts"] == ["hello", "world"]
        assert body["meta"] == {
            "api_version": {"version": "2"},
            "billed_units": {"input_tokens": 7},
        }
        (call,) = embed_backend.calls
        assert call["inputs"] == ["hello", "world"]
        assert call["dimensions"] is None
        assert call["extra_params"] == {"input_type": "search_document"}

    def test_cohere_params_forwarded_for_cohere_models(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """input_type/truncate/max_tokens and output_dimension are forwarded.

        `output_dimension` travels as the provider-neutral `dimensions`
        argument, which the Cohere backend renames back to `output_dimension`.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v4.html
             stdapi/models/embedding/cohere_embed.py:EmbeddingModel.embed_text
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_query",
                "texts": ["q"],
                "truncate": "START",
                "max_tokens": 128,
                "output_dimension": 512,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["embeddings"] == {"float": [[0.1, 0.2]]}
        (call,) = embed_backend.calls
        assert call["dimensions"] == 512
        assert call["extra_params"] == {
            "input_type": "search_query",
            "truncate": "START",
            "max_tokens": 128,
        }

    def test_cohere_params_dropped_for_other_providers(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Cohere-specific fields are not forwarded to non-Cohere models.

        Titan Embed rejects unknown body fields, so `input_type`, `truncate`
        and `max_tokens` must all be consumed by the route instead of reaching
        InvokeModel and turning a valid request into a ValidationException.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
             stdapi/routes/cohere_embed.py:embed
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "amazon.titan-embed-text-v2:0",
                "input_type": "search_document",
                "texts": ["hello"],
                "truncate": "START",
                "max_tokens": 128,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["embeddings"] == {"float": [[0.1, 0.2]]}
        (call,) = embed_backend.calls
        assert call["extra_params"] == {}

    def test_embedding_vectors_are_not_written_to_the_request_log(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """The logged response keeps the envelope but never the embedding vectors.

        ``log_request_params`` is enabled in this test environment. Dropping the
        route's ``exclude`` argument would put every vector of every request
        into the structured log, blowing up log volume and leaking content.

        Ref: https://stdapi.ai/api_cohere_embed/
             stdapi/routes/cohere_embed.py:embed
             stdapi/monitoring.py:log_response_params
        """
        from stdapi import monitoring  # noqa: PLC0415

        written: list[dict[str, Any]] = []
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(monitoring, "write_log_event", written.append)
            response = app_client.post(
                "/cohere/v2/embed",
                json={
                    "model": "cohere.embed-multilingual-v3",
                    "input_type": "search_document",
                    "texts": ["hello"],
                },
            )

        assert response.status_code == 200, response.text
        assert response.json()["embeddings"] == {"float": [[0.1, 0.2]]}
        (log,) = [entry for entry in written if entry.get("type") == "request"]
        logged = log["request_response"]
        assert "embeddings" not in logged
        assert logged["meta"]["billed_units"]["input_tokens"] == 7

    def test_image_data_uri_input(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Image data URIs are parsed into file inputs and billed as images."""
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "image",
                "images": ["data:image/png;base64,aGVsbG8="],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["response_type"] == "embeddings_by_type"
        assert body["embeddings"] == {"float": [[0.1, 0.2]]}
        assert body["meta"]["billed_units"] == {"input_tokens": 7, "images": 1}
        assert "texts" not in body
        (call,) = embed_backend.calls
        assert len(call["inputs"]) == 1
        assert isinstance(call["inputs"][0], InputFile)
        assert call["extra_params"]["input_type"] == "image"

    def test_mixed_texts_and_images_input(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Texts and images in one request are concatenated, texts first.

        Only Cohere Embed v4 accepts a mixed batch; the backend turns it into
        the fused `inputs` Bedrock field.

        Ref: https://docs.cohere.com/docs/embeddings
             stdapi/models/embedding/cohere_embed.py:EmbeddingModel.embed_text
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "texts": ["hello", "world"],
                "images": ["data:image/png;base64,aGVsbG8="],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["texts"] == ["hello", "world"]
        assert body["embeddings"] == {"float": [[0.1, 0.2]] * 3}
        assert body["meta"]["billed_units"] == {"input_tokens": 7, "images": 1}
        (call,) = embed_backend.calls
        assert len(call["inputs"]) == 3
        assert call["inputs"][:2] == ["hello", "world"]
        assert isinstance(call["inputs"][2], InputFile)

    def test_https_image_url_is_accepted_and_re_encoded_as_a_data_uri(
        self,
        app_client: TestClient,
        embed_backend: _StubEmbeddingModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An `https://` image is accepted and rendered as the data URI Bedrock requires.

        Cohere's own API only takes data URIs in `images`; this gateway also
        accepts URLs, which is only usable if the fetched body is re-encoded
        with the content type resolved from the response.

        Ref: https://docs.cohere.com/reference/embed
             stdapi/types/cohere_embed.py:_EmbedRequestBase
             stdapi/models/embedding/cohere_embed.py:EmbeddingModel.embed_text
        """
        _serve_png(monkeypatch, "_HttpSource")

        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "image",
                "images": ["https://example.invalid/image.png"],
            },
        )

        assert response.status_code == 200, response.text
        assert response.json()["meta"]["billed_units"] == {
            "input_tokens": 7,
            "images": 1,
        }
        (call,) = embed_backend.calls
        (image,) = call["inputs"]
        assert isinstance(image, InputFile)
        assert run(image.to_data_uri()) == _PNG_DATA_URI

    def test_s3_image_uri_is_accepted_and_re_encoded_as_a_data_uri(
        self,
        app_client: TestClient,
        embed_backend: _StubEmbeddingModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An `s3://` image is accepted and rendered as the data URI Bedrock requires.

        The content type comes from the stored object metadata rather than
        from a data-URI header, so this is a distinct resolution path.

        Ref: https://docs.cohere.com/reference/embed
             stdapi/types/cohere_embed.py:_EmbedRequestBase
             stdapi/input_file.py:_S3Source
        """
        from stdapi import input_file  # noqa: PLC0415

        monkeypatch.setattr(input_file, "_ACCEPTED_BUCKETS", frozenset({_S3_BUCKET}))
        monkeypatch.setattr(input_file, "BUCKET_TO_REGION", {_S3_BUCKET: "us-east-1"})
        _serve_png(monkeypatch, "_S3Source")

        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "image",
                "images": [f"s3://{_S3_BUCKET}/image.png"],
            },
        )

        assert response.status_code == 200, response.text
        (call,) = embed_backend.calls
        (image,) = call["inputs"]
        assert isinstance(image, InputFile)
        assert run(image.to_data_uri()) == _PNG_DATA_URI

    def test_no_input_is_rejected(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """A request without texts, images or inputs fails validation.

        The message names every field that would satisfy the endpoint, `inputs`
        included, so a client is not told to send one it already sent.

        Ref: stdapi/types/cohere_embed.py:_EmbedRequestBase
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={"model": "cohere.embed-v4:0", "input_type": "search_document"},
        )
        assert response.status_code == 400
        body = response.json()
        assert set(body) == {"message", "id"}
        assert "texts" in body["message"]
        assert "images" in body["message"]
        assert "inputs" in body["message"]
        assert not embed_backend.calls

    def test_non_float_embedding_types_are_rejected_for_unsupported_backends(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """embedding_types other than float/base64 are rejected for non-Cohere/Titan models.

        Ref: stdapi/types/cohere_embed.py:resolve_embedding_types
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "some-vendor.embed-v1",
                "input_type": "search_document",
                "texts": ["a"],
                "embedding_types": ["int8"],
            },
        )
        assert response.status_code == 400
        body = response.json()
        assert set(body) == {"message", "id"}
        assert "int8" in body["message"]
        assert "not supported" in body["message"]
        assert not embed_backend.calls

    def test_int8_embedding_type_is_forwarded_and_returned_for_cohere_model(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """embedding_types=["int8"] is forwarded to Bedrock Cohere Embed and returned as-is.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html
             stdapi/types/cohere_embed.py:resolve_embedding_types
        """
        embed_backend.embeddings_by_type = {"int8": [[1, 2]]}
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "texts": ["a"],
                "embedding_types": ["int8"],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["response_type"] == "embeddings_by_type"
        assert body["embeddings"] == {"int8": [[1, 2]]}, (
            "only the requested types are returned, matching the Cohere API"
        )
        (call,) = embed_backend.calls
        # `float` is always requested alongside other types (model-layer safety net).
        assert call["extra_params"]["embedding_types"] == ["float", "int8"]

    def test_binary_embedding_type_is_forwarded_for_titan_model(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """embedding_types=["binary"] is forwarded to Titan Embed v2 as `embeddingTypes`.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
             stdapi/types/cohere_embed.py:resolve_embedding_types
        """
        embed_backend.embeddings_by_type = {"binary": [[1, 0]]}
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "amazon.titan-embed-text-v2:0",
                "input_type": "search_document",
                "texts": ["a"],
                "embedding_types": ["binary"],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["embeddings"] == {"binary": [[1, 0]]}
        (call,) = embed_backend.calls
        assert call["extra_params"]["embeddingTypes"] == ["binary", "float"]
        assert "embedding_types" not in call["extra_params"], (
            "the Cohere spelling must not be sent to a Titan model"
        )

    def test_int8_embedding_type_is_rejected_for_titan_model(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Titan Embed v2 only supports float/binary; int8 returns 400.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
             stdapi/types/cohere_embed.py:resolve_embedding_types
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "amazon.titan-embed-text-v2:0",
                "input_type": "search_document",
                "texts": ["a"],
                "embedding_types": ["int8"],
            },
        )
        assert response.status_code == 400
        body = response.json()
        assert set(body) == {"message", "id"}
        assert "int8" in body["message"]
        assert not embed_backend.calls

    def test_binary_embedding_type_is_rejected_for_titan_v1_model(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Titan Embed G1 (v1) has no `embeddingTypes` support; binary returns 400.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
             stdapi/types/cohere_embed.py:resolve_embedding_types
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "amazon.titan-embed-text-v1",
                "input_type": "search_document",
                "texts": ["a"],
                "embedding_types": ["binary"],
            },
        )
        assert response.status_code == 400
        body = response.json()
        assert set(body) == {"message", "id"}
        assert "binary" in body["message"]
        assert not embed_backend.calls

    def test_embedding_types_are_not_forwarded_to_titan_multimodal_model(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Titan Multimodal Embeddings G1 has no `embeddingTypes` field at all.

        `float` is the only type this backend can serve, so it is accepted but
        satisfied from the plain `embedding` response field.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-mm.html
             stdapi/types/cohere_embed.py:resolve_embedding_types
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "amazon.titan-embed-image-v1",
                "input_type": "search_document",
                "texts": ["a"],
                "embedding_types": ["float"],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["embeddings"] == {"float": [[0.1, 0.2]]}
        (call,) = embed_backend.calls
        assert "embeddingTypes" not in call["extra_params"]

    def test_base64_embedding_type_is_computed_client_side(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """embedding_types=["base64"] is computed from `float`, not forwarded to Bedrock.

        Ref: stdapi/types/cohere_embed.py:_encode_base64
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "texts": ["a"],
                "embedding_types": ["base64"],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert set(body["embeddings"]) == {"base64"}
        (encoded,) = body["embeddings"]["base64"]
        assert unpack("<2f", b64decode(encoded)) == pytest.approx((0.1, 0.2))
        (call,) = embed_backend.calls
        # `float` is requested from the backend to compute `base64` locally.
        assert call["extra_params"]["embedding_types"] == ["float"]

    def test_base64_and_float_embedding_types_both_returned(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Requesting both `float` and `base64` returns both keys.

        Ref: stdapi/types/cohere_embed.py:build_embeddings_by_type
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "texts": ["a"],
                "embedding_types": ["float", "base64"],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert set(body["embeddings"]) == {"float", "base64"}
        assert body["embeddings"]["float"] == [[0.1, 0.2]]
        (encoded,) = body["embeddings"]["base64"]
        assert unpack("<2f", b64decode(encoded)) == pytest.approx((0.1, 0.2)), (
            "`base64` must encode the same vector as `float`"
        )
        (call,) = embed_backend.calls
        assert call["extra_params"]["embedding_types"] == ["float"]

    def test_fused_input_is_forwarded_as_one_grouped_entry(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """An `inputs` entry with several content parts embeds into one vector.

        The parts of one entry travel to the backend grouped together, in the
        order they were sent, which is what makes the result a single fused
        embedding instead of one embedding per part.

        Ref: https://docs.cohere.com/reference/embed
             stdapi/types/cohere_embed.py:EmbedRequest
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "inputs": [
                    {
                        "content": [
                            {"type": "text", "text": "a red bicycle"},
                            {"type": "image_url", "image_url": {"url": _PNG_DATA_URI}},
                        ]
                    }
                ],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["embeddings"] == {"float": [[0.1, 0.2]]}, "one vector per entry"
        assert body["meta"]["billed_units"] == {"input_tokens": 7, "images": 1}
        (call,) = embed_backend.calls
        (entry,) = call["inputs"]
        text_part, image_part = entry
        assert text_part == "a red bicycle"
        assert isinstance(image_part, InputFile)
        assert call["extra_params"] == {"input_type": "search_document"}

    def test_fused_inputs_are_appended_after_texts_and_images_in_order(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """`texts`, `images` and `inputs` are embedded in that order.

        The response returns one vector per input in request order, so a client
        combining the three fields has to be able to map `embeddings[i]` back to
        what it sent.

        Ref: https://stdapi.ai/api_cohere_embed/
             stdapi/routes/cohere_embed.py:embed
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "texts": ["plain text"],
                "images": [_PNG_DATA_URI],
                "inputs": [
                    {
                        "content": [
                            {"type": "text", "text": "fused text"},
                            {"type": "image_url", "image_url": {"url": _PNG_DATA_URI}},
                        ]
                    },
                    {"content": [{"type": "text", "text": "single part"}]},
                ],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["embeddings"] == {"float": [[0.1, 0.2]] * 4}
        assert body["texts"] == ["plain text"], "only `texts` is echoed back"
        assert body["meta"]["billed_units"] == {"input_tokens": 7, "images": 2}
        (call,) = embed_backend.calls
        plain, image, fused, single = call["inputs"]
        assert plain == "plain text"
        assert isinstance(image, InputFile)
        assert fused[0] == "fused text"
        assert isinstance(fused[1], InputFile)
        assert single == "single part", (
            "a one-part entry is equivalent to the same `texts` entry"
        )

    @pytest.mark.parametrize(
        "model", ["cohere.embed-v4:0", "amazon.titan-embed-text-v2:0"]
    )
    def test_single_part_text_input_is_equivalent_to_a_text_entry(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel, model: str
    ) -> None:
        """A one-part `inputs` entry is embedded like the same `texts` entry.

        Flattening it makes `inputs` usable on every embedding model, not only
        on the ones that can fuse several parts into one vector. The equality
        stops at the vector: the response `texts` array echoes the `texts`
        field alone, so a text sent through `inputs` is not echoed.

        Ref: https://docs.cohere.com/reference/embed
             stdapi/types/cohere_embed.py:EmbedRequest
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": model,
                "input_type": "search_document",
                "inputs": [{"content": [{"type": "text", "text": "hello"}]}],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["embeddings"] == {"float": [[0.1, 0.2]]}
        assert "texts" not in body, "only the `texts` field is echoed back"
        (call,) = embed_backend.calls
        assert call["inputs"] == ["hello"]

    def test_fused_inputs_are_not_forwarded_as_extra_model_parameters(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """`inputs` is a declared field, so it never reaches the backend as an extra.

        Unknown request fields are forwarded as additional model parameters; a
        second, unshaped copy of `inputs` would reach the model beside the one
        the route builds.

        Ref: stdapi/aws_bedrock.py:get_extra_model_parameters
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "inputs": [{"content": [{"type": "text", "text": "a"}]}],
            },
        )
        assert response.status_code == 200, response.text
        (call,) = embed_backend.calls
        assert call["extra_params"] == {"input_type": "search_document"}

    def test_inputs_alone_satisfies_the_input_requirement(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """A request carrying only `inputs` is accepted.

        Ref: stdapi/types/cohere_embed.py:_EmbedRequestBase
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "inputs": [{"content": [{"type": "text", "text": "a"}]}],
            },
        )
        assert response.status_code == 200, response.text
        assert embed_backend.calls

    @pytest.mark.parametrize("field", ["texts", "inputs"])
    def test_more_inputs_than_the_cohere_limit_are_forwarded(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel, field: str
    ) -> None:
        """The 96-input Cohere limit is a model limit and is not enforced here.

        Models that embed one input per call take far more than 96, so a cap on
        `texts` or `inputs` would reject a legitimate request to Titan, Nova or
        Marengo; the resolved model refuses what it cannot take.

        Ref: https://docs.cohere.com/reference/embed
             stdapi/types/cohere_embed.py:EmbedRequest
        """
        count = _COHERE_MAX_INPUTS + 1
        entries: list[Any] = (
            [f"chunk {index}" for index in range(count)]
            if field == "texts"
            else [
                {"content": [{"type": "text", "text": f"chunk {index}"}]}
                for index in range(count)
            ]
        )
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "amazon.titan-embed-text-v2:0",
                "input_type": "search_document",
                field: entries,
            },
        )
        assert response.status_code == 200, response.text
        (call,) = embed_backend.calls
        assert len(call["inputs"]) == count

    def test_empty_input_content_is_rejected(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """An `inputs` entry with no content part is rejected.

        An empty entry has nothing to embed but would still claim a position in
        the returned vector list.

        Ref: stdapi/types/cohere_embed.py:EmbedInput
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "inputs": [{"content": []}],
            },
        )
        assert response.status_code == 400
        body = response.json()
        assert set(body) == {"message", "id"}
        assert "content" in body["message"]
        assert not embed_backend.calls

    def test_unknown_input_content_type_is_rejected(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """A content part of an unknown type is rejected rather than ignored.

        Ref: https://docs.cohere.com/reference/embed
             stdapi/types/cohere_embed.py:EmbedInputContent
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "inputs": [{"content": [{"type": "audio_url", "text": "a"}]}],
            },
        )
        assert response.status_code == 400
        body = response.json()
        assert set(body) == {"message", "id"}
        assert "audio_url" in body["message"]
        assert not embed_backend.calls

    def test_every_text_part_of_a_fused_input_is_guarded(
        self,
        app_client: TestClient,
        embed_backend: _StubEmbeddingModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Guardrails screen the text inside `inputs`, not only `texts`.

        A text part reaching the model unscreened would be a guardrail bypass
        that a client can trigger by moving its text into a fused entry.

        Ref: https://stdapi.ai/api_cohere_embed/
             stdapi/aws_bedrock.py:apply_guardrail_to_texts
        """
        seen: list[Any] = []

        async def _guard(items: list[Any], *, source: str) -> list[Any]:
            assert source == "INPUT"
            seen.extend(item for item in items if isinstance(item, str))
            return [f"[{item}]" if isinstance(item, str) else item for item in items]

        monkeypatch.setattr(cohere_embed, "apply_guardrail_to_texts", _guard)

        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "texts": ["plain"],
                "inputs": [
                    {
                        "content": [
                            {"type": "image_url", "image_url": {"url": _PNG_DATA_URI}},
                            {"type": "text", "text": "second part"},
                        ]
                    },
                    {"content": [{"type": "text", "text": "third input"}]},
                ],
            },
        )

        assert response.status_code == 200, response.text
        assert seen == ["plain", "second part", "third input"]
        (call,) = embed_backend.calls
        plain, fused, single = call["inputs"]
        assert plain == "[plain]"
        assert isinstance(fused[0], InputFile), "image parts pass through unchanged"
        assert fused[1] == "[second part]"
        assert single == "[third input]"

    def test_priority_is_ignored(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """The Cohere priority hint is accepted but not forwarded to AWS.

        Ref: stdapi/types/cohere_embed.py:EmbedRequest
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "texts": ["a"],
                "priority": 0,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["embeddings"] == {"float": [[0.1, 0.2]]}
        (call,) = embed_backend.calls
        assert call["extra_params"] == {"input_type": "search_document"}

    def test_null_inputs_is_treated_as_absent(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """An explicit `inputs: null` alongside `texts` is accepted and not forwarded.

        Unknown request fields are passed through to Bedrock as extra model
        parameters, so a null `inputs` must be dropped by the validator instead.

        Ref: stdapi/types/cohere_embed.py:_EmbedRequestBase
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "texts": ["a"],
                "inputs": None,
            },
        )
        assert response.status_code == 200, response.text
        (call,) = embed_backend.calls
        assert call["extra_params"] == {"input_type": "search_document"}

    def test_request_params_override_operator_defaults(
        self,
        app_client: TestClient,
        embed_backend: _StubEmbeddingModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit request values win over `default_model_params` defaults.

        Operator defaults are merged first, then the route overwrites each
        Cohere field that the request set explicitly.

        Ref: stdapi/aws_bedrock.py:get_extra_model_parameters
        """
        monkeypatch.setitem(
            SETTINGS.default_model_params,
            "cohere.embed-multilingual-v3",
            {"input_type": "clustering", "truncate": "NONE", "max_tokens": 64},
        )
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-multilingual-v3",
                "input_type": "search_query",
                "texts": ["q"],
                "truncate": "START",
                "max_tokens": 128,
            },
        )
        assert response.status_code == 200, response.text
        (call,) = embed_backend.calls
        assert call["extra_params"] == {
            "input_type": "search_query",
            "truncate": "START",
            "max_tokens": 128,
        }

    def test_alias_resolution_drives_cohere_params(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Cohere params follow the resolved model ID, not the requested alias.

        The alias `embed-multilingual` does not carry the `cohere.` prefix, so a
        route keying off the requested ID would drop `input_type` entirely.

        Ref: stdapi/routes/cohere_embed.py:embed
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "embed-multilingual",
                "input_type": "search_document",
                "texts": ["hello"],
            },
        )
        assert response.status_code == 200, response.text
        (call,) = embed_backend.calls
        assert call["extra_params"] == {"input_type": "search_document"}

    def test_unknown_model_returns_cohere_error_envelope(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Model errors use the Cohere `{"message": ...}` envelope.

        Cohere never documents an error schema, so the gateway emits its own
        `{message, id}` body and keeps the 404 status of the model lookup.

        Ref: https://docs.cohere.com/reference/errors
             stdapi/api_providers/cohere.py:_format_error
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "unknown-model",
                "input_type": "search_document",
                "texts": ["a"],
            },
        )
        assert response.status_code == 404
        body = response.json()
        assert set(body) == {"message", "id"}
        assert "unknown-model" in body["message"]
        assert body["id"] == response.headers["x-request-id"]
        assert not embed_backend.calls

    def test_images_metadata_is_echoed_in_wire_shape(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Image metadata parsed by the model is echoed with the Cohere `images` shape."""
        embed_backend.images = [
            EmbeddingImageDescription(format="png", width=10, height=20, bit_depth=8)
        ]
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "image",
                "images": ["data:image/png;base64,aGVsbG8="],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["images"] == [
            {"width": 10, "height": 20, "format": "png", "bit_depth": 8}
        ]
        assert body["meta"]["billed_units"]["images"] == 1

    def test_images_metadata_absent_when_not_reported(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """No `images` key is emitted when the model does not report image metadata."""
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-multilingual-v3",
                "input_type": "search_document",
                "texts": ["hello"],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert "images" not in body
        assert "images" not in body["meta"]["billed_units"]
        assert body["texts"] == ["hello"]


@pytest.mark.local
@pytest.mark.usefixtures("validated_model")
class TestCohereFusedInputSupport:
    """POST /cohere/v2/embed: fused `inputs` entries on models that cannot fuse.

    These requests are refused by the resolved model itself, before any backend
    call, so they run against the real embedding model classes.

    Ref: https://docs.cohere.com/reference/embed
         stdapi/models/embedding/__init__.py:EmbeddingModelBase._single_part_inputs
    """

    @pytest.mark.parametrize(
        "model",
        [
            "cohere.embed-multilingual-v3",
            "cohere.embed-english-v3",
            "amazon.titan-embed-text-v2:0",
            "amazon.titan-embed-image-v1",
            "amazon.nova-2-multimodal-embeddings-v1:0",
            "twelvelabs.marengo-embed-3-0-v1:0",
        ],
    )
    def test_multi_part_input_is_rejected(
        self, app_client: TestClient, model: str
    ) -> None:
        """A model that cannot fuse content parts refuses the request.

        Splitting the entry would answer with two vectors where one was asked
        for, silently shifting every following index.

        Ref: stdapi/models/embedding/__init__.py:EmbeddingModelBase._single_part_inputs
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": model,
                "input_type": "search_document",
                "inputs": [
                    {
                        "content": [
                            {"type": "text", "text": "a red bicycle"},
                            {"type": "image_url", "image_url": {"url": _PNG_DATA_URI}},
                        ]
                    }
                ],
            },
        )
        assert response.status_code == 400, response.text
        body = response.json()
        assert set(body) == {"message", "id"}
        assert "Embed v4" in body["message"]

    def test_texts_and_images_on_v3_name_no_backend_identifier(
        self, app_client: TestClient
    ) -> None:
        """The mixed-content refusal names the model generation, not a backend ID.

        A client calling `embed-multilingual-v3.0` never sent a Bedrock model
        identifier and cannot look one up in Cohere's catalogue.

        Ref: stdapi/models/embedding/cohere_embed.py:EmbeddingModel._build_request
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-multilingual-v3",
                "input_type": "search_document",
                "texts": ["a red bicycle"],
                "images": [_PNG_DATA_URI],
            },
        )
        assert response.status_code == 400, response.text
        message = response.json()["message"]
        assert "Cohere Embed v4" in message
        assert "cohere.embed" not in message, "no backend model identifier"


@pytest.mark.local
class TestCohereFusedRequestBody:
    """The Cohere Embed v4 request body a fused `inputs` entry produces.

    The route hands the model one nested list per fused entry; this is where
    that grouping becomes a single `inputs` element. Splitting it would answer
    with one vector per content part instead of one per entry, shifting every
    following index without any error.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v4.html
         stdapi/models/embedding/cohere_embed.py:EmbeddingModel._to_input_contents
    """

    async def test_fused_entry_becomes_one_grouped_inputs_element(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A multi-part entry is sent as one `inputs` element carrying both parts.

        Ref: stdapi/models/embedding/cohere_embed.py:EmbeddingModel._to_input_contents
        """
        model = EmbeddingModel("cohere.embed-v4:0")
        monkeypatch.setattr(
            type(model),
            "invoke",
            AsyncMock(
                return_value=InvokeResult(response={"embeddings": [[0.1], [0.2]]})
            ),
        )
        response = await model.embed_text(
            [["a red bicycle", InputFileUrl(_PNG_DATA_URI)], "Bonjour le monde"],
            dimensions=None,
            extra_params={},
        )
        request = model.invoke.call_args.args[0]  # type: ignore[attr-defined]
        assert request["inputs"] == [
            {
                "content": [
                    {"type": "text", "text": "a red bicycle"},
                    {"type": "image_url", "image_url": {"url": _PNG_DATA_URI}},
                ]
            },
            {"content": [{"type": "text", "text": "Bonjour le monde"}]},
        ], "one entry per input, the fused one keeping both parts"
        assert "texts" not in request
        assert "images" not in request
        assert response.embeddings == [[0.1], [0.2]]


@pytest.mark.local
class TestCohereEmbedV1Route:
    """POST /cohere/v1/embed: legacy v1 response shapes and error envelopes.

    v1 has two mutually exclusive envelopes: `embeddings_floats` (a flat list of
    float vectors) when `embedding_types` is omitted, and `embeddings_by_type`
    when it is sent.

    Ref: https://docs.cohere.com/v1/reference/embed
         stdapi/routes/cohere_embed_v1.py:embed_v1
    """

    def test_embed_texts_floats_shape_by_default(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Without embedding_types the legacy embeddings_floats shape is returned.

        Ref: stdapi/types/cohere_embed.py:EmbedV1FloatsResponse
        """
        response = app_client.post(
            "/cohere/v1/embed",
            json={"model": "cohere.embed-multilingual-v3", "texts": ["hello", "world"]},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"]
        assert response.headers["x-request-id"] == body["id"]
        assert body == {
            "response_type": "embeddings_floats",
            "id": body["id"],
            "embeddings": [[0.1, 0.2], [0.1, 0.2]],
            "texts": ["hello", "world"],
            "meta": {
                "api_version": {"version": "1"},
                "billed_units": {"input_tokens": 7},
            },
        }
        (call,) = embed_backend.calls
        assert call["inputs"] == ["hello", "world"]
        assert call["dimensions"] is None
        assert call["extra_params"] == {}, (
            "v1 leaves `input_type` unset when the request omits it"
        )

    def test_embedding_types_float_returns_by_type_shape(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """embedding_types=["float"] switches to the embeddings_by_type shape.

        Ref: stdapi/types/cohere_embed.py:resolve_embedding_types
        """
        response = app_client.post(
            "/cohere/v1/embed",
            json={
                "model": "cohere.embed-multilingual-v3",
                "texts": ["hello"],
                "embedding_types": ["float"],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["response_type"] == "embeddings_by_type"
        assert body["embeddings"] == {"float": [[0.1, 0.2]]}
        assert body["texts"] == ["hello"]
        assert body["meta"] == {
            "api_version": {"version": "1"},
            "billed_units": {"input_tokens": 7},
        }
        (call,) = embed_backend.calls
        assert call["extra_params"]["embedding_types"] == ["float"]

    def test_binary_embedding_type_is_forwarded_for_titan_model(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """v1 forwards Titan's native `embeddingTypes`, matching the v2 endpoint.

        The v1 route carries its own copy of the Titan branch; without this the
        two could drift and a v1 client asking for `binary` would silently get
        float-only vectors.

        Ref: https://docs.cohere.com/v1/reference/embed
             https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
             stdapi/routes/cohere_embed_v1.py:embed_v1
        """
        embed_backend.embeddings_by_type = {"binary": [[1, 0]]}
        response = app_client.post(
            "/cohere/v1/embed",
            json={
                "model": "amazon.titan-embed-text-v2:0",
                "texts": ["a"],
                "embedding_types": ["binary"],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["response_type"] == "embeddings_by_type"
        assert body["embeddings"] == {"binary": [[1, 0]]}
        (call,) = embed_backend.calls
        assert call["extra_params"]["embeddingTypes"] == ["binary", "float"]
        assert "embedding_types" not in call["extra_params"], (
            "the Cohere spelling must not be sent to a Titan model"
        )

    def test_non_float_embedding_types_are_rejected_for_unsupported_backends(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """embedding_types other than float/base64 are rejected for non-Cohere/Titan models.

        Ref: stdapi/types/cohere_embed.py:resolve_embedding_types
        """
        response = app_client.post(
            "/cohere/v1/embed",
            json={
                "model": "some-vendor.embed-v1",
                "texts": ["a"],
                "embedding_types": ["int8"],
            },
        )
        assert response.status_code == 400
        body = response.json()
        assert set(body) == {"message", "id"}
        assert "int8" in body["message"]
        assert "not supported" in body["message"]
        assert not embed_backend.calls

    def test_int8_embedding_type_is_forwarded_for_cohere_model(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """embedding_types=["int8"] is forwarded to Bedrock Cohere Embed and returned as-is.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html
             stdapi/types/cohere_embed.py:resolve_embedding_types
        """
        embed_backend.embeddings_by_type = {"int8": [[1, 2]]}
        response = app_client.post(
            "/cohere/v1/embed",
            json={
                "model": "cohere.embed-multilingual-v3",
                "texts": ["a"],
                "embedding_types": ["int8"],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["response_type"] == "embeddings_by_type"
        assert body["embeddings"] == {"int8": [[1, 2]]}
        (call,) = embed_backend.calls
        # `float` is always requested alongside other types (model-layer safety net).
        assert call["extra_params"]["embedding_types"] == ["float", "int8"]

    def test_base64_embedding_type_is_computed_client_side(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """embedding_types=["base64"] is computed from `float`, not forwarded to Bedrock.

        Ref: stdapi/types/cohere_embed.py:_encode_base64
        """
        response = app_client.post(
            "/cohere/v1/embed",
            json={
                "model": "cohere.embed-multilingual-v3",
                "texts": ["a"],
                "embedding_types": ["base64"],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["response_type"] == "embeddings_by_type"
        assert set(body["embeddings"]) == {"base64"}
        (encoded,) = body["embeddings"]["base64"]
        assert unpack("<2f", b64decode(encoded)) == pytest.approx((0.1, 0.2))
        (call,) = embed_backend.calls
        assert call["extra_params"]["embedding_types"] == ["float"]

    def test_input_type_and_truncate_forwarded_for_cohere_models(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """input_type and truncate are forwarded to Cohere models when provided.

        v1 has no `max_tokens` or `output_dimension`, so nothing else may reach
        the backend.

        Ref: https://docs.cohere.com/v1/reference/embed
        """
        response = app_client.post(
            "/cohere/v1/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_query",
                "texts": ["q"],
                "truncate": "END",
            },
        )
        assert response.status_code == 200, response.text
        (call,) = embed_backend.calls
        assert call["extra_params"] == {"input_type": "search_query", "truncate": "END"}
        assert call["dimensions"] is None

    def test_cohere_params_dropped_for_other_providers(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Cohere-specific fields are not forwarded to non-Cohere models.

        Titan Embed rejects unknown body fields, so both `input_type` and
        `truncate` — every Cohere-only field the v1 schema declares — must be
        consumed by the route instead of reaching InvokeModel.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
             stdapi/routes/cohere_embed_v1.py:embed_v1
        """
        response = app_client.post(
            "/cohere/v1/embed",
            json={
                "model": "amazon.titan-embed-text-v2:0",
                "input_type": "search_document",
                "texts": ["hello"],
                "truncate": "START",
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["embeddings"] == [[0.1, 0.2]]
        (call,) = embed_backend.calls
        assert call["extra_params"] == {}

    @pytest.mark.parametrize("embedding_types", [None, ["float"]])
    def test_embedding_vectors_are_not_written_to_the_request_log(
        self,
        app_client: TestClient,
        embed_backend: _StubEmbeddingModel,
        embedding_types: list[str] | None,
    ) -> None:
        """Neither v1 envelope carries its embedding vectors into the request log.

        v1 builds two different response models — the legacy floats shape and
        the by-type shape — each with its own `exclude`; losing either would
        copy every vector of every request into the structured log.

        Ref: https://docs.cohere.com/v1/reference/embed
             stdapi/routes/cohere_embed_v1.py:embed_v1
             stdapi/monitoring.py:log_response_params
        """
        from stdapi import monitoring  # noqa: PLC0415

        body: dict[str, Any] = {
            "model": "cohere.embed-multilingual-v3",
            "texts": ["hello"],
        }
        if embedding_types is not None:
            body["embedding_types"] = embedding_types
        written: list[dict[str, Any]] = []
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(monitoring, "write_log_event", written.append)
            response = app_client.post("/cohere/v1/embed", json=body)

        assert response.status_code == 200, response.text
        assert response.json()["embeddings"]
        (log,) = [entry for entry in written if entry.get("type") == "request"]
        logged = log["request_response"]
        assert "embeddings" not in logged
        assert logged["texts"] == ["hello"], "only `embeddings` is excluded"
        assert logged["meta"]["billed_units"]["input_tokens"] == 7

    def test_image_data_uri_input(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Image data URIs are parsed into file inputs and billed as images.

        An image-only v1 request still uses the floats envelope and omits
        `texts` entirely.
        """
        response = app_client.post(
            "/cohere/v1/embed",
            json={
                "model": "cohere.embed-v4:0",
                "images": ["data:image/png;base64,aGVsbG8="],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["response_type"] == "embeddings_floats"
        assert body["embeddings"] == [[0.1, 0.2]]
        assert body["meta"]["billed_units"] == {"input_tokens": 7, "images": 1}
        assert "texts" not in body
        (call,) = embed_backend.calls
        assert len(call["inputs"]) == 1
        assert isinstance(call["inputs"][0], InputFile)

    def test_no_input_is_rejected(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """A request without texts or images fails with the Cohere envelope.

        Ref: stdapi/types/cohere_embed.py:_EmbedRequestBase
             stdapi/api_providers/cohere.py:_format_error
        """
        response = app_client.post(
            "/cohere/v1/embed", json={"model": "cohere.embed-multilingual-v3"}
        )
        assert response.status_code == 400
        body = response.json()
        assert set(body) == {"message", "id"}
        assert "texts" in body["message"]
        assert "images" in body["message"]
        assert "inputs" not in body["message"], (
            "v1 must not offer a field it does not accept"
        )
        assert not embed_backend.calls

    def test_fused_inputs_are_rejected(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """The v2-style fused multimodal `inputs` field is rejected with a clear error.

        Ref: stdapi/types/cohere_embed.py:_EmbedRequestBase
        """
        response = app_client.post(
            "/cohere/v1/embed",
            json={
                "model": "cohere.embed-v4:0",
                "inputs": [{"content": [{"type": "text", "text": "a"}]}],
            },
        )
        assert response.status_code == 400
        body = response.json()
        assert set(body) == {"message", "id"}
        assert "inputs" in body["message"]
        assert not embed_backend.calls

    def test_null_inputs_is_treated_as_absent(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """An explicit `inputs: null` alongside `texts` is accepted and not forwarded.

        Ref: stdapi/types/cohere_embed.py:_EmbedRequestBase
        """
        response = app_client.post(
            "/cohere/v1/embed",
            json={"model": "cohere.embed-v4:0", "texts": ["a"], "inputs": None},
        )
        assert response.status_code == 200, response.text
        (call,) = embed_backend.calls
        assert call["extra_params"] == {}

    def test_unknown_model_returns_cohere_error_envelope(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Model errors use the Cohere `{"message": ...}` envelope.

        Ref: https://docs.cohere.com/reference/errors
             stdapi/api_providers/cohere.py:_format_error
        """
        response = app_client.post(
            "/cohere/v1/embed", json={"model": "unknown-model", "texts": ["a"]}
        )
        assert response.status_code == 404
        body = response.json()
        assert set(body) == {"message", "id"}
        assert "unknown-model" in body["message"]
        assert body["id"] == response.headers["x-request-id"]
        assert not embed_backend.calls

    def test_images_metadata_is_echoed_in_wire_shape(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """Image metadata parsed by the model is echoed with the Cohere `images` shape.

        The `images` metadata is carried by the floats envelope too, not only by
        the by-type one.
        """
        embed_backend.images = [
            EmbeddingImageDescription(format="png", width=10, height=20, bit_depth=8)
        ]
        response = app_client.post(
            "/cohere/v1/embed",
            json={
                "model": "cohere.embed-v4:0",
                "images": ["data:image/png;base64,aGVsbG8="],
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["response_type"] == "embeddings_floats"
        assert body["images"] == [
            {"width": 10, "height": 20, "format": "png", "bit_depth": 8}
        ]
        assert body["meta"]["billed_units"]["images"] == 1


@pytest.fixture
def live_client(
    server_url: str | None, test_client: TestClient | None, api_key: str
) -> Generator[httpx.Client]:
    """Authenticated client for the local live server or the --server-url target."""
    if test_client is not None:
        test_client.headers["authorization"] = f"Bearer {api_key}"
        # Starlette types its TestClient against httpx2; the alias in conftest makes
        # it an httpx.Client at runtime, which is what this fixture promises.
        yield test_client  # type: ignore[misc]
        del test_client.headers["authorization"]
    elif server_url:
        with httpx.Client(
            base_url=server_url,
            headers={"authorization": f"Bearer {getenv('OPENAI_API_KEY', '')}"},
            timeout=60.0,
        ) as client:
            yield client
    else:
        pytest.skip("Cohere-compatible routes are not part of the official API")


class TestCohereEmbedIntegration:
    """Live /v2/embed calls through the official Cohere SDK.

    Ref: https://docs.cohere.com/reference/embed
         stdapi/routes/cohere_embed.py:embed
    """

    @staticmethod
    def _embed(
        cohere_client: cohere.ClientV2,
        model_id: str,
        **params: Any,  # noqa: ANN401
    ) -> EmbedByTypeResponse:
        """Embed one short text with *model_id* and return the response."""
        return cohere_client.embed(
            model=model_id,
            input_type="search_document",
            texts=[_SAMPLE_TEXT],
            embedding_types=["float"],
            **params,
        )

    def test_embed_texts_cohere_model(
        self, cohere_client: cohere.ClientV2, cohere_embed_multilingual_model: str
    ) -> None:
        """A Cohere model returns one float vector per text with the v2 shape."""
        response = self._embed(cohere_client, cohere_embed_multilingual_model)
        assert response.id
        assert response.response_type == "embeddings_by_type"
        assert response.embeddings.float_ is not None
        (vector,) = response.embeddings.float_
        assert len(vector) > 0
        assert all(isinstance(value, float) for value in vector)
        assert response.texts == [_SAMPLE_TEXT]
        assert response.meta is not None
        assert response.meta.api_version is not None
        assert response.meta.api_version.version == "2"
        assert response.meta.billed_units is not None
        assert (response.meta.billed_units.input_tokens or 0) > 0
        assert not response.meta.billed_units.images, (
            "a text-only request must not be billed as an image"
        )

    @pytest.mark.gateway("Bedrock-specific model")
    def test_embed_texts_non_cohere_model(
        self, cohere_client: cohere.ClientV2, embedding_model: str
    ) -> None:
        """The required input_type is accepted (and dropped) for non-Cohere models.

        Titan Embed rejects unknown InvokeModel body fields, so a successful
        call is itself the evidence that `input_type` never reached Bedrock.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
        """
        response = self._embed(cohere_client, embedding_model)
        assert response.response_type == "embeddings_by_type"
        assert response.embeddings.float_ is not None
        (vector,) = response.embeddings.float_
        if embedding_model.startswith("amazon.titan-embed-text-v2"):
            assert len(vector) == _TITAN_V2_DIMENSIONS, (
                "Titan Text Embeddings V2 defaults to 1024 dimensions"
            )
        else:
            assert len(vector) > 0
        assert all(isinstance(value, float) for value in vector)
        assert response.texts == [_SAMPLE_TEXT]
        assert response.meta is not None
        assert response.meta.billed_units is not None
        assert (response.meta.billed_units.input_tokens or 0) > 0

    def test_output_dimension(
        self, cohere_client: cohere.ClientV2, cohere_embed_v4_model: str
    ) -> None:
        """output_dimension controls the returned vector size.

        512 is one of the four widths Cohere Embed v4 accepts and differs from
        its 1536 default, so the length is a real echo of the parameter.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v4.html
        """
        response = self._embed(
            cohere_client, cohere_embed_v4_model, output_dimension=512
        )
        assert response.embeddings.float_ is not None
        (vector,) = response.embeddings.float_
        assert len(vector) == 512
        assert all(isinstance(value, float) for value in vector)
        assert response.texts == [_SAMPLE_TEXT]

    def test_quantized_and_base64_embedding_types(
        self, cohere_client: cohere.ClientV2, cohere_embed_v4_model: str
    ) -> None:
        """int8/base64 embedding_types are returned, and float is omitted when unrequested.

        `base64` is computed by the gateway from the `float` vectors it always
        requests from Bedrock, so it is present even though `float` is not.

        Ref: https://docs.cohere.com/docs/embeddings
             stdapi/types/cohere_embed.py:build_embeddings_by_type
        """
        response = cohere_client.embed(
            model=cohere_embed_v4_model,
            input_type="search_document",
            texts=[_SAMPLE_TEXT],
            embedding_types=["int8", "base64"],
        )
        assert response.embeddings.float_ is None
        assert response.embeddings.int8 is not None
        (int8_vector,) = response.embeddings.int8
        assert len(int8_vector) in _COHERE_V4_DIMENSIONS
        assert all(-128 <= value <= 127 for value in int8_vector), (
            "int8 embeddings must fit in a signed byte"
        )
        assert response.embeddings.base64 is not None
        (encoded,) = response.embeddings.base64
        decoded = b64decode(encoded)
        assert len(decoded) % 4 == 0, "base64 embeddings are packed float32 values"
        assert len(decoded) // 4 in _COHERE_V4_DIMENSIONS

    @pytest.mark.gateway("Bedrock-specific model")
    def test_binary_embedding_type_for_titan_model(
        self, cohere_client: cohere.ClientV2, embedding_model: str
    ) -> None:
        """embedding_types=["binary"] is served by Titan Embed v2 as `binary`.

        The gateway asks Bedrock for `float` as well but must return only the
        requested `binary` type.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
             stdapi/types/cohere_embed.py:build_embeddings_by_type
        """
        response = cohere_client.embed(
            model=embedding_model,
            input_type="search_document",
            texts=[_SAMPLE_TEXT],
            embedding_types=["binary"],
        )
        assert response.embeddings.binary is not None
        (vector,) = response.embeddings.binary
        assert len(vector) > 0
        assert all(isinstance(value, int) for value in vector)
        assert response.embeddings.float_ is None, (
            "`float` was not requested and must not be returned"
        )
        assert response.meta is not None
        assert response.meta.billed_units is not None
        assert (response.meta.billed_units.input_tokens or 0) > 0

    def test_embed_image(
        self,
        cohere_client: cohere.ClientV2,
        cohere_embed_v4_model: str,
        sample_image_file_base64: str,
    ) -> None:
        """An image data URI is embedded, billed as one image, and its metadata echoed.

        Cohere Embed v3 accepts a single image per call and v4 takes several;
        one is enough to pin the echo of the format and pixel size the model
        reports, which the route returns as `images`.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html
        """
        response = cohere_client.embed(
            model=cohere_embed_v4_model,
            input_type="image",
            images=[sample_image_file_base64],
            embedding_types=["float"],
        )
        assert response.response_type == "embeddings_by_type"
        assert response.embeddings.float_ is not None
        (vector,) = response.embeddings.float_
        assert len(vector) in _COHERE_V4_DIMENSIONS
        assert response.texts in (None, [])
        assert response.meta is not None
        assert response.meta.billed_units is not None
        assert response.meta.billed_units.images == 1
        assert response.images is not None
        (image,) = response.images
        assert image.width > 0
        assert image.height > 0
        assert image.format
        assert "png" in image.format.lower(), "the submitted data URI is a PNG"

    def test_embed_fused_text_and_image(
        self,
        cohere_client: cohere.ClientV2,
        cohere_embed_v4_model: str,
        sample_image_file_base64: str,
        use_official_api: bool,
    ) -> None:
        """A text and an image sent as one input embed into a single vector.

        The point of `inputs` is that the caption and the picture share one
        vector: one entry in, one vector out, and the image's tokens inside the
        same billed count.

        Ref: https://docs.cohere.com/reference/embed
             stdapi/models/embedding/cohere_embed.py:EmbeddingModel._build_request
        """
        response = cohere_client.embed(
            model=cohere_embed_v4_model,
            input_type="search_document",
            inputs=[
                EmbedInput(
                    content=[
                        TextEmbedContent(text=_SAMPLE_TEXT),
                        ImageUrlEmbedContent(
                            image_url=EmbedImageUrl(url=sample_image_file_base64)
                        ),
                    ]
                )
            ],
            embedding_types=["float"],
        )
        assert response.embeddings.float_ is not None
        (vector,) = response.embeddings.float_
        assert len(vector) in _COHERE_V4_DIMENSIONS
        assert response.meta is not None
        assert response.meta.billed_units is not None
        assert (response.meta.billed_units.input_tokens or 0) > 0, (
            "the image is billed inside the input token count"
        )
        if not use_official_api:
            assert response.meta.billed_units.images == 1, (
                "the image part is reported as one submitted image"
            )
            assert not response.images, (
                "no image metadata is reported for a fused input"
            )

    def test_single_part_image_input_echoes_its_metadata(
        self,
        cohere_client: cohere.ClientV2,
        cohere_embed_v4_model: str,
        sample_image_file_base64: str,
        use_official_api: bool,
    ) -> None:
        """A one-part image `inputs` entry reports metadata, unlike a fused one.

        The entry is submitted exactly as the same `images` entry would be, so
        the model still describes the image -- which is the difference a client
        reading `images` has to know about.

        Ref: https://docs.cohere.com/reference/embed
             stdapi/types/cohere_embed.py:EmbedRequest.embed_inputs
        """
        response = cohere_client.embed(
            model=cohere_embed_v4_model,
            input_type="search_document",
            inputs=[
                EmbedInput(
                    content=[
                        ImageUrlEmbedContent(
                            image_url=EmbedImageUrl(url=sample_image_file_base64)
                        )
                    ]
                )
            ],
            embedding_types=["float"],
        )
        assert response.embeddings.float_ is not None
        (vector,) = response.embeddings.float_
        assert len(vector) in _COHERE_V4_DIMENSIONS
        if not use_official_api:
            assert response.images is not None
            (image,) = response.images
            assert image.width > 0
            assert image.height > 0
            assert image.format
            assert "png" in image.format.lower(), "the submitted data URI is a PNG"

    def test_fused_inputs_return_one_vector_per_entry(
        self,
        cohere_client: cohere.ClientV2,
        cohere_embed_v4_model: str,
        sample_image_file_base64: str,
    ) -> None:
        """Two fused inputs return two distinct vectors, in request order.

        Ref: https://docs.cohere.com/reference/embed
        """
        response = cohere_client.embed(
            model=cohere_embed_v4_model,
            input_type="search_document",
            inputs=[
                EmbedInput(
                    content=[
                        TextEmbedContent(text=_SAMPLE_TEXT),
                        ImageUrlEmbedContent(
                            image_url=EmbedImageUrl(url=sample_image_file_base64)
                        ),
                    ]
                ),
                EmbedInput(content=[TextEmbedContent(text="Bonjour le monde")]),
            ],
            embedding_types=["float"],
        )
        assert response.embeddings.float_ is not None
        first, second = response.embeddings.float_
        assert len(first) == len(second)
        assert first != second

    def test_texts_combined_with_inputs_return_one_vector_each(
        self,
        cohere_client: cohere.ClientV2,
        cohere_embed_v4_model: str,
        sample_image_file_base64: str,
        use_official_api: bool,
    ) -> None:
        """`texts` sent together with `inputs` embeds both, texts first.

        Cohere documents a maximum per field but never says whether the fields
        may be combined, so this is the test that checks the vendor accepts the
        combination the endpoint advertises rather than only that this gateway
        does. A failure here is a product bug, not a lane to relax.

        Ref: https://docs.cohere.com/reference/embed
             stdapi/types/cohere_embed.py:EmbedRequest.embed_inputs
        """
        response = cohere_client.embed(
            model=cohere_embed_v4_model,
            input_type="search_document",
            texts=[_SAMPLE_TEXT],
            inputs=[
                EmbedInput(
                    content=[
                        TextEmbedContent(text="A red bicycle leaning on a wall"),
                        ImageUrlEmbedContent(
                            image_url=EmbedImageUrl(url=sample_image_file_base64)
                        ),
                    ]
                ),
                EmbedInput(content=[TextEmbedContent(text="Bonjour le monde")]),
            ],
            embedding_types=["float"],
        )
        assert response.embeddings.float_ is not None
        plain, fused, single = response.embeddings.float_
        assert len({len(plain), len(fused), len(single)}) == 1, (
            "one vector per input, all of the model's width"
        )
        assert plain != fused
        assert fused != single
        assert single != plain
        if not use_official_api:
            assert response.texts == [_SAMPLE_TEXT], "only `texts` is echoed back"
            assert not response.images, (
                "a request carrying a text returns no image metadata"
            )
            assert response.meta is not None
            assert response.meta.billed_units is not None
            assert response.meta.billed_units.images == 1

    def test_more_texts_than_the_cohere_limit_are_refused(
        self, cohere_client: cohere.ClientV2, cohere_embed_multilingual_model: str
    ) -> None:
        """A Cohere model embeds 96 texts and refuses 97 with a 400.

        The endpoint deliberately forwards any number of `texts`, because other
        embedding models take far more than 96, so the documented cap rests
        entirely on the model refusing the excess -- the half never covered.
        The refusal names no count, so the boundary itself is what pins the
        limit; the over-long half is free, being refused before inference with
        no token count reported at all.

        Ref: https://docs.cohere.com/reference/embed
             stdapi/types/cohere_embed.py:EmbedRequest
        """
        texts = [f"chunk {index}" for index in range(_COHERE_MAX_INPUTS + 1)]
        accepted = cohere_client.embed(
            model=cohere_embed_multilingual_model,
            input_type="search_document",
            texts=texts[:_COHERE_MAX_INPUTS],
            embedding_types=["float"],
        )
        assert accepted.embeddings.float_ is not None
        assert len(accepted.embeddings.float_) == _COHERE_MAX_INPUTS
        with pytest.raises(BadRequestError) as refused:
            cohere_client.embed(
                model=cohere_embed_multilingual_model,
                input_type="search_document",
                texts=texts,
                embedding_types=["float"],
            )
        assert refused.value.status_code == 400

    @pytest.mark.gateway("Bedrock-specific model")
    def test_single_part_input_on_a_non_cohere_model(
        self, cohere_client: cohere.ClientV2, embedding_model: str
    ) -> None:
        """A one-part `inputs` entry works on a model that cannot fuse parts.

        Ref: stdapi/types/cohere_embed.py:EmbedRequest
        """
        response = cohere_client.embed(
            model=embedding_model,
            input_type="search_document",
            inputs=[EmbedInput(content=[TextEmbedContent(text=_SAMPLE_TEXT)])],
            embedding_types=["float"],
        )
        assert response.embeddings.float_ is not None
        (vector,) = response.embeddings.float_
        assert len(vector) > 0

    @pytest.mark.local
    def test_embed_v3_image_records_the_billed_image(
        self,
        cohere_client: cohere.ClientV2,
        cohere_embed_multilingual_model: str,
        sample_image_file_base64: str,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """An image embedded on Embed v3 is reported as one billed image.

        Embed v3 meters images as their own billed unit and reports no token
        count for them, so a request recording nothing would be attributed at
        zero cost while AWS bills it.

        Ref: https://aws.amazon.com/bedrock/pricing/
             stdapi/models/embedding/cohere_embed.py:EmbeddingModel._record_invoke_usage
        """
        capfd.readouterr()
        response = cohere_client.embed(
            model=cohere_embed_multilingual_model,
            input_type="image",
            images=[sample_image_file_base64],
            embedding_types=["float"],
        )
        assert response.embeddings.float_ is not None
        (entry,) = logged_usage_entries(
            capfd.readouterr().out, model=cohere_embed_multilingual_model
        )
        assert entry["input_images"] == 1

    @pytest.mark.slow
    @pytest.mark.xdist_group("moderations_guardrail")
    def test_guardrail_blocks_a_text_part_of_a_fused_input(
        self, live_client: httpx.Client, live_guardrail: str, cohere_embed_v4_model: str
    ) -> None:
        """A blocked word inside a fused entry stops the request.

        The text of a fused entry is screened exactly like a `texts` entry, so
        moving it into `inputs` is not a way around the configured guardrail.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-content-filters.html
             stdapi/routes/cohere_embed.py:embed
        """
        response = live_client.post(
            "/cohere/v2/embed",
            headers={
                "X-Amzn-Bedrock-GuardrailIdentifier": live_guardrail,
                "X-Amzn-Bedrock-GuardrailVersion": "DRAFT",
            },
            json={
                "model": cohere_embed_v4_model,
                "input_type": "search_document",
                "inputs": [
                    {"content": [{"type": "text", "text": _SAMPLE_TEXT}]},
                    {
                        "content": [
                            {"type": "text", "text": "harmless"},
                            {"type": "text", "text": "BLOCKWORDXYZ"},
                        ]
                    },
                ],
            },
        )
        assert response.status_code == 400, response.text
        body = response.json()
        assert set(body) == {"message", "id"}
        assert body["message"] == "Blocked by test guardrail.", (
            "the guardrail's own blocked-input messaging is returned"
        )

    @pytest.mark.slow
    def test_every_advertised_model_embeds(
        self, cohere_client: cohere.ClientV2, live_client: httpx.Client
    ) -> None:
        """Every model advertising the route serves a text embed request.

        Ref: stdapi/models/capabilities.py:register_route_capability
        """
        response = live_client.get("/search_models", params={"route": "cohere_embed"})
        assert response.status_code == 200, response.text
        model_ids = [model["id"] for model in response.json()]
        assert model_ids
        for model_id in model_ids:
            embedded = self._embed(cohere_client, model_id)
            assert embedded.response_type == "embeddings_by_type", model_id
            assert embedded.embeddings.float_ is not None, model_id
            (vector,) = embedded.embeddings.float_
            assert len(vector) > 0, model_id
            assert embedded.texts == [_SAMPLE_TEXT], model_id


class TestCohereEmbedV1Integration:
    """Live /v1/embed calls through the official Cohere SDK.

    Ref: https://docs.cohere.com/v1/reference/embed
         stdapi/routes/cohere_embed_v1.py:embed_v1
    """

    def test_embed_texts_floats_shape(
        self, cohere_client_v1: cohere.Client, cohere_embed_multilingual_model: str
    ) -> None:
        """Without embedding_types (or input_type) the legacy floats shape is returned.

        Ref: stdapi/types/cohere_embed.py:EmbedV1FloatsResponse
        """
        response = cohere_client_v1.embed(
            model=cohere_embed_multilingual_model, texts=[_SAMPLE_TEXT]
        )
        assert response.response_type == "embeddings_floats"
        assert response.id
        (vector,) = response.embeddings
        assert len(vector) > 0
        assert all(isinstance(value, float) for value in vector)
        assert response.texts == [_SAMPLE_TEXT]
        assert response.meta is not None
        assert response.meta.api_version is not None
        assert response.meta.api_version.version == "1"
        assert response.meta.billed_units is not None
        assert (response.meta.billed_units.input_tokens or 0) > 0
        assert not response.meta.billed_units.images, (
            "a text-only request must not be billed as an image"
        )

    def test_embedding_types_float_returns_by_type_shape(
        self, cohere_client_v1: cohere.Client, cohere_embed_multilingual_model: str
    ) -> None:
        """embedding_types=["float"] returns the by-type shape with v1 metadata.

        Ref: stdapi/types/cohere_embed.py:resolve_embedding_types
        """
        response = cohere_client_v1.embed(
            model=cohere_embed_multilingual_model,
            input_type="search_document",
            texts=[_SAMPLE_TEXT],
            embedding_types=["float"],
        )
        assert response.response_type == "embeddings_by_type"
        assert response.embeddings.float_ is not None
        (vector,) = response.embeddings.float_
        assert len(vector) > 0
        assert all(isinstance(value, float) for value in vector)
        assert response.texts == [_SAMPLE_TEXT]
        assert response.meta is not None
        assert response.meta.api_version is not None
        assert response.meta.api_version.version == "1"
        assert response.meta.billed_units is not None
        assert (response.meta.billed_units.input_tokens or 0) > 0

    @pytest.mark.gateway("Bedrock-specific model")
    def test_embed_texts_non_cohere_model(
        self, cohere_client_v1: cohere.Client, embedding_model: str
    ) -> None:
        """A non-Cohere model works without the Cohere-specific parameters.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
        """
        response = cohere_client_v1.embed(model=embedding_model, texts=[_SAMPLE_TEXT])
        assert response.response_type == "embeddings_floats"
        (vector,) = response.embeddings
        if embedding_model.startswith("amazon.titan-embed-text-v2"):
            assert len(vector) == _TITAN_V2_DIMENSIONS, (
                "Titan Text Embeddings V2 defaults to 1024 dimensions"
            )
        else:
            assert len(vector) > 0
        assert response.texts == [_SAMPLE_TEXT]
        assert response.meta is not None
        assert response.meta.api_version is not None
        assert response.meta.api_version.version == "1"
        assert response.meta.billed_units is not None
        assert (response.meta.billed_units.input_tokens or 0) > 0

    def test_embed_image(
        self,
        cohere_client_v1: cohere.Client,
        cohere_embed_v4_model: str,
        sample_image_file_base64: str,
    ) -> None:
        """An image data URI is embedded and billed as one image.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html
        """
        response = cohere_client_v1.embed(
            model=cohere_embed_v4_model,
            input_type="image",
            images=[sample_image_file_base64],
        )
        assert response.response_type == "embeddings_floats"
        (vector,) = response.embeddings
        assert len(vector) in _COHERE_V4_DIMENSIONS
        assert response.texts in (None, [])
        assert response.meta is not None
        assert response.meta.billed_units is not None
        assert response.meta.billed_units.images == 1
