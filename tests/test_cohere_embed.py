"""Cohere-compatible /v1/embed and /v2/embed routes (unit and live).

Ref: https://docs.cohere.com/reference/embed
     https://docs.cohere.com/v1/reference/embed
     stdapi/routes/cohere_embed.py:embed
     stdapi/routes/cohere_embed_v1.py:embed_v1
"""

from base64 import b64decode
from os import getenv
from struct import unpack
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from stdapi.api_errors import UnsupportedModelError
from stdapi.config import SETTINGS
from stdapi.input_file import InputFile
from stdapi.models.embedding import EmbeddingImageDescription, EmbeddingResponse
from stdapi.routes import cohere_embed, cohere_embed_v1
from tests._helpers import make_model_details

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

    async def _validate_model(model_id: str, modality: str) -> ModelDetails:
        assert modality == "EMBEDDING"
        if model_id == "unknown-model":
            raise UnsupportedModelError(model_id)
        resolved_id = _MODEL_ALIASES.get(model_id, model_id)
        return make_model_details(resolved_id, output_modalities=["EMBEDDING"])

    stub = _StubEmbeddingModel()
    for module in (cohere_embed, cohere_embed_v1):
        monkeypatch.setattr(module, "validate_model", _validate_model)
        monkeypatch.setattr(module, "get_embedding_model", lambda _model_id: stub)
    return stub


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

    def test_no_input_is_rejected(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """A request without texts or images fails validation.

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

    def test_fused_inputs_are_rejected(
        self, app_client: TestClient, embed_backend: _StubEmbeddingModel
    ) -> None:
        """The v2 fused multimodal `inputs` field is rejected with a clear error.

        Ref: https://docs.cohere.com/reference/embed
             stdapi/types/cohere_embed.py:_EmbedRequestBase
        """
        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-v4:0",
                "input_type": "search_document",
                "inputs": [{"content": [{"type": "text", "text": "a"}]}],
            },
        )
        assert response.status_code == 400
        body = response.json()
        assert set(body) == {"message", "id"}
        assert "inputs" in body["message"]
        assert not embed_backend.calls

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

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
        """
        response = app_client.post(
            "/cohere/v1/embed",
            json={
                "model": "amazon.titan-embed-text-v2:0",
                "input_type": "search_document",
                "texts": ["hello"],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["embeddings"] == [[0.1, 0.2]]
        (call,) = embed_backend.calls
        assert call["extra_params"] == {}

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
        yield test_client
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

        Bedrock accepts at most one image per Cohere Embed call and reports its
        format and pixel size, which the route echoes as `images`.

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
