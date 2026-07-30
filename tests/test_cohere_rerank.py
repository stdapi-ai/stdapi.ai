"""Cohere-compatible rerank surface: POST /v1/rerank and POST /v2/rerank.

Both routes are served by the Bedrock Agent Runtime Rerank API, so v1-only and
v2-only Cohere parameters are either translated, ignored or rejected here.

Ref: https://docs.cohere.com/v2/reference/rerank
     https://docs.cohere.com/v1/reference/rerank
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html
     stdapi/routes/cohere_rerank.py:rerank
     stdapi/routes/cohere_rerank_v1.py:rerank_v1
"""

from os import getenv
from typing import TYPE_CHECKING, Any

import cohere
import pytest

from stdapi.api_errors import UnsupportedModelError
from stdapi.config import SETTINGS
from stdapi.models import RERANKING_MODALITY, ModelDetails
from stdapi.models.rerank import RerankedDocument, RerankResponse
from stdapi.routes import cohere_rerank, cohere_rerank_v1
from tests.test_models_rerank import RERANK_MODELS

if TYPE_CHECKING:
    from starlette.testclient import TestClient


class _StubRerankModel:
    """Stub backend recording the rerank call and returning a fixed response."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def rerank(
        self,
        query: str,
        documents: list[str | dict[str, Any]],
        *,
        top_n: int | None,
        extra_params: dict[str, Any],
    ) -> RerankResponse:
        """Record the call and return one reranked document."""
        self.calls.append(
            {
                "query": query,
                "documents": documents,
                "top_n": top_n,
                "extra_params": extra_params,
            }
        )
        return RerankResponse(
            results=[RerankedDocument(index=1, relevance_score=0.98)], search_units=1
        )


@pytest.fixture
def rerank_backend(monkeypatch: pytest.MonkeyPatch) -> _StubRerankModel:
    """Stub model validation and the rerank backend."""

    async def _validate_model(
        model_id: str, *_args: object, **_kwargs: object
    ) -> ModelDetails:
        if model_id == "unknown-model":
            raise UnsupportedModelError(model_id)
        return ModelDetails(
            id=model_id,
            name=model_id,
            provider="Vendor",
            input_modalities=["TEXT"],
            output_modalities=[RERANKING_MODALITY],
            regions=["us-east-1"],
        )

    stub = _StubRerankModel()
    for module in (cohere_rerank, cohere_rerank_v1):
        monkeypatch.setattr(module, "validate_model", _validate_model)
        monkeypatch.setattr(module, "get_rerank_model", lambda _model_id: stub)
    return stub


@pytest.mark.local
class TestCohereRerankRoute:
    """POST /cohere/v2/rerank: response shape and error envelopes.

    Ref: https://docs.cohere.com/v2/reference/rerank
         stdapi/routes/cohere_rerank.py:rerank
         stdapi/types/cohere_rerank.py:RerankRequest
    """

    def test_rerank_success(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """A valid request returns ``{id, results, meta}`` with ``api_version`` "2".

        The response ``id`` is the gateway request ID, also echoed in the
        ``x-request-id`` header, and ``meta.billed_units.search_units`` carries the
        backend's billed search units.

        Ref: stdapi/types/cohere_rerank.py:RerankResponse
             stdapi/types/cohere.py:ApiMeta
        """
        response = app_client.post(
            "/cohere/v2/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "capital of the USA",
                "documents": ["Carson City", "Washington, D.C."],
                "top_n": 1,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert set(body) == {"id", "results", "meta"}
        assert body["id"]
        assert body["results"] == [{"index": 1, "relevance_score": 0.98}]
        assert body["meta"] == {
            "api_version": {"version": "2"},
            "billed_units": {"search_units": 1},
        }
        assert response.headers["x-request-id"] == body["id"]
        (call,) = rerank_backend.calls
        assert call["query"] == "capital of the USA"
        assert call["documents"] == ["Carson City", "Washington, D.C."]
        assert call["top_n"] == 1
        assert call["extra_params"] == {}

    def test_max_tokens_per_doc_forwarded(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """max_tokens_per_doc is forwarded to the backend as an extra parameter.

        The Rerank API has no truncation field of its own, so v2's
        ``max_tokens_per_doc`` (Cohere default 4096) can only reach the model
        through ``additionalModelRequestFields``.

        Ref: https://docs.cohere.com/v2/reference/rerank
             stdapi/models/rerank/bedrock_rerank.py:RerankModel
        """
        response = app_client.post(
            "/cohere/v2/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": ["a"],
                "max_tokens_per_doc": 512,
            },
        )
        assert response.status_code == 200, response.text
        (call,) = rerank_backend.calls
        assert call["extra_params"] == {"max_tokens_per_doc": 512}

    def test_unknown_body_field_is_forwarded_as_an_extra_param(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """An undeclared body field reaches the backend as an additional model parameter.

        Undeclared fields land in the request model's ``model_extra`` and are
        merged over the operator's ``default_model_params``, so a client can
        drive ``additionalModelRequestFields`` without a gateway change.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html
             stdapi/aws_bedrock.py:get_extra_model_parameters
             stdapi/routes/cohere_rerank.py:rerank
        """
        response = app_client.post(
            "/cohere/v2/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": ["a"],
                "custom_knob": 3,
            },
        )
        assert response.status_code == 200, response.text
        (call,) = rerank_backend.calls
        assert call["extra_params"] == {"custom_knob": 3}

    def test_operator_default_model_params_are_merged_and_overridable(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """Operator defaults apply to rerank models and lose to a same-named body field.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html
             stdapi/aws_bedrock.py:get_extra_model_parameters
        """
        with pytest.MonkeyPatch.context() as patch:
            patch.setitem(
                SETTINGS.default_model_params,
                "cohere.rerank-v3-5:0",
                {"custom_knob": 1, "operator_only": "kept"},
            )
            response = app_client.post(
                "/cohere/v2/rerank",
                json={
                    "model": "cohere.rerank-v3-5:0",
                    "query": "q",
                    "documents": ["a"],
                    "custom_knob": 3,
                },
            )

        assert response.status_code == 200, response.text
        (call,) = rerank_backend.calls
        assert call["extra_params"] == {"custom_knob": 3, "operator_only": "kept"}

    def test_reranked_documents_are_not_written_to_the_request_log(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """The logged response keeps ``meta`` but never the reranked ``results``.

        ``log_request_params`` is enabled in this test environment; without the
        route's ``exclude`` argument every ranked customer document would be
        copied into the structured request log.

        Ref: https://stdapi.ai/api_cohere_rerank/
             stdapi/routes/cohere_rerank.py:rerank
             stdapi/monitoring.py:log_response_params
        """
        from stdapi import monitoring  # noqa: PLC0415

        written: list[dict[str, Any]] = []
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(monitoring, "write_log_event", written.append)
            response = app_client.post(
                "/cohere/v2/rerank",
                json={
                    "model": "cohere.rerank-v3-5:0",
                    "query": "q",
                    "documents": ["a", "b"],
                },
            )

        assert response.status_code == 200, response.text
        assert response.json()["results"] == [{"index": 1, "relevance_score": 0.98}]
        (log,) = [entry for entry in written if entry.get("type") == "request"]
        logged = log["request_response"]
        assert "results" not in logged
        assert logged["meta"]["billed_units"]["search_units"] == 1

    @pytest.mark.parametrize("priority", [5, 1000])
    def test_priority_is_ignored(
        self, app_client: TestClient, rerank_backend: _StubRerankModel, priority: int
    ) -> None:
        """The Cohere priority hint is accepted (no upper bound) but not forwarded to AWS.

        Request scheduling priority has no Bedrock equivalent, so the field is a
        declared-but-ignored compatibility knob rather than an extra parameter.

        Ref: stdapi/types/cohere_rerank.py:RerankRequest
        """
        response = app_client.post(
            "/cohere/v2/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": ["a"],
                "priority": priority,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["results"] == [{"index": 1, "relevance_score": 0.98}]
        (call,) = rerank_backend.calls
        assert "priority" not in call["extra_params"]

    @pytest.mark.parametrize("return_documents", [True, False])
    def test_return_documents_is_ignored(
        self,
        app_client: TestClient,
        rerank_backend: _StubRerankModel,
        return_documents: bool,
    ) -> None:
        """The v1-only return_documents field is accepted but never echoes documents.

        v2 results reference the input documents by ``index`` only, so the field is
        accepted for v1 compatibility and dropped.

        Ref: https://docs.cohere.com/v2/reference/rerank
             stdapi/types/cohere_rerank.py:RerankRequest
        """
        response = app_client.post(
            "/cohere/v2/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": ["a"],
                "return_documents": return_documents,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["results"] == [{"index": 1, "relevance_score": 0.98}]
        (call,) = rerank_backend.calls
        assert "return_documents" not in call["extra_params"]

    def test_unknown_model_returns_cohere_error_envelope(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """An unknown model yields a 404 in the Cohere ``{message, id}`` envelope.

        Cohere never documents its error JSON shape, so the gateway's envelope is
        ``{"message": ..., "id": <request id>}`` with no ``type``/``code`` field.

        Ref: https://docs.cohere.com/reference/errors
             stdapi/api_providers/cohere.py:_format_error
             stdapi/api_errors.py:UnsupportedModelError
        """
        response = app_client.post(
            "/cohere/v2/rerank",
            json={"model": "unknown-model", "query": "q", "documents": ["a"]},
        )
        assert response.status_code == 404
        body = response.json()
        assert set(body) == {"message", "id"}
        assert "unknown-model" in body["message"]
        assert "does not exist" in body["message"]
        assert response.headers["x-request-id"] == body["id"]
        assert not rerank_backend.calls

    def test_resolved_model_not_rerank_capable_returns_cohere_error_envelope(
        self,
        app_client: TestClient,
        rerank_backend: _StubRerankModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A model that resolves but has no rerank backend returns a Cohere 404.

        ``validate_model`` only checks the model exists with the RERANKING output
        modality; the registry lookup is the second gate and its
        ``UnsupportedModelError`` must reach the client in the Cohere envelope too.

        Ref: stdapi/models/rerank/__init__.py:get_rerank_model
             stdapi/api_providers/cohere.py:_format_error
        """

        def _get_rerank_model(model_id: str) -> _StubRerankModel:
            raise UnsupportedModelError(model_id)

        monkeypatch.setattr(cohere_rerank, "get_rerank_model", _get_rerank_model)

        response = app_client.post(
            "/cohere/v2/rerank",
            json={
                "model": "anthropic.claude-3-5-haiku-20241022-v1:0",
                "query": "q",
                "documents": ["a"],
            },
        )

        assert response.status_code == 404
        body = response.json()
        assert set(body) == {"message", "id"}
        assert "anthropic.claude-3-5-haiku-20241022-v1:0" in body["message"]
        assert response.headers["x-request-id"] == body["id"]
        assert not rerank_backend.calls

    def test_missing_query_is_rejected(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """A request without ``query`` is a 400 naming the offending body field.

        ``query`` is required by Cohere; the gateway surfaces the Pydantic failure
        as ``Validation error at body.<field>: <reason>`` inside the Cohere envelope.

        Ref: https://docs.cohere.com/v2/reference/rerank
             stdapi/main.py:handle_validation_exception
        """
        response = app_client.post(
            "/cohere/v2/rerank",
            json={"model": "cohere.rerank-v3-5:0", "documents": ["a"]},
        )
        assert response.status_code == 400
        body = response.json()
        assert set(body) == {"message", "id"}
        assert body["message"].startswith("Validation error at body.query:")
        assert not rerank_backend.calls

    def test_empty_documents_are_rejected(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """An empty ``documents`` list is a 400 naming ``body.documents``.

        The Rerank API requires between 1 and 1000 sources, so the gateway rejects
        the empty list before any AWS call.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html
             stdapi/types/cohere_rerank.py:RerankRequest
        """
        response = app_client.post(
            "/cohere/v2/rerank",
            json={"model": "cohere.rerank-v3-5:0", "query": "q", "documents": []},
        )
        assert response.status_code == 400
        body = response.json()
        assert set(body) == {"message", "id"}
        assert body["message"].startswith("Validation error at body.documents:")
        assert not rerank_backend.calls


@pytest.mark.local
class TestCohereRerankV1Route:
    """POST /cohere/v1/rerank: legacy v1 response shape and error envelopes.

    Ref: https://docs.cohere.com/v1/reference/rerank
         stdapi/routes/cohere_rerank_v1.py:rerank_v1
         stdapi/types/cohere_rerank.py:RerankV1Request
    """

    def test_rerank_success_with_string_documents(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """A valid v1 request returns ``{id, results, meta}`` with ``api_version`` "1".

        Ref: stdapi/types/cohere_rerank.py:RerankV1Result
             stdapi/types/cohere.py:ApiMeta
        """
        response = app_client.post(
            "/cohere/v1/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "capital of the USA",
                "documents": ["Carson City", "Washington, D.C."],
                "top_n": 1,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"]
        assert response.headers["x-request-id"] == body["id"]
        assert body == {
            "id": body["id"],
            "results": [{"index": 1, "relevance_score": 0.98}],
            "meta": {
                "api_version": {"version": "1"},
                "billed_units": {"search_units": 1},
            },
        }
        (call,) = rerank_backend.calls
        assert call["query"] == "capital of the USA"
        assert call["documents"] == ["Carson City", "Washington, D.C."]
        assert call["top_n"] == 1
        assert call["extra_params"] == {}

    def test_unknown_body_field_is_forwarded_as_an_extra_param(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """v1 forwards undeclared body fields as extra model parameters, like v2.

        The two routes carry independent parameter-mapping code, so the v1 copy
        needs its own proof that ``model_extra`` reaches the backend.

        Ref: https://docs.cohere.com/v1/reference/rerank
             stdapi/aws_bedrock.py:get_extra_model_parameters
             stdapi/routes/cohere_rerank_v1.py:rerank_v1
        """
        response = app_client.post(
            "/cohere/v1/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": ["a"],
                "custom_knob": 3,
            },
        )
        assert response.status_code == 200, response.text
        (call,) = rerank_backend.calls
        assert call["extra_params"] == {"custom_knob": 3}

    def test_single_key_text_object_documents_are_passed_through(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """Single-key ``{"text": ...}`` object documents reach the backend unchanged.

        v1 accepts objects as well as strings; without ``rank_fields`` no projection
        happens, so the backend decides the Bedrock source shape.

        Ref: https://docs.cohere.com/v1/reference/rerank
             stdapi/routes/cohere_rerank_v1.py:_project_document
        """
        response = app_client.post(
            "/cohere/v1/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": [{"text": "Carson City"}, {"text": "Washington, D.C."}],
            },
        )
        assert response.status_code == 200, response.text
        (call,) = rerank_backend.calls
        assert call["documents"] == [
            {"text": "Carson City"},
            {"text": "Washington, D.C."},
        ]

    def test_multi_field_object_documents_are_passed_through(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """Multi-field object documents reach the backend as-is without rank_fields.

        Ref: stdapi/routes/cohere_rerank_v1.py:_project_document
        """
        document = {"title": "Nevada", "body": "Carson City is its capital."}
        response = app_client.post(
            "/cohere/v1/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": [document],
            },
        )
        assert response.status_code == 200, response.text
        (call,) = rerank_backend.calls
        assert call["documents"] == [document]

    def test_rank_fields_projects_object_documents(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """rank_fields keeps only the requested keys of object documents.

        Keys absent from the document are skipped rather than materialised as null,
        and unlisted keys are dropped before the document reaches Bedrock.

        Ref: https://docs.cohere.com/v1/reference/rerank
             stdapi/routes/cohere_rerank_v1.py:_project_document
        """
        response = app_client.post(
            "/cohere/v1/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": [
                    {"title": "Nevada", "body": "Carson City.", "other": "ignored"}
                ],
                "rank_fields": ["title", "body", "missing"],
            },
        )
        assert response.status_code == 200, response.text
        (call,) = rerank_backend.calls
        assert call["documents"] == [{"title": "Nevada", "body": "Carson City."}]

    def test_rank_fields_do_not_affect_string_documents(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """rank_fields is only meaningful for object documents; strings pass through.

        Ref: stdapi/routes/cohere_rerank_v1.py:_project_document
        """
        response = app_client.post(
            "/cohere/v1/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": ["a"],
                "rank_fields": ["title"],
            },
        )
        assert response.status_code == 200, response.text
        (call,) = rerank_backend.calls
        assert call["documents"] == ["a"]

    def test_return_documents_echoes_original_multi_field_document(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """return_documents echoes the original object, not the rank_fields projection.

        Cohere does not document how object documents are rendered back as
        ``document.text``; the gateway joins every original field as
        ``key: value`` lines, so the ``rank_fields=["title"]`` projection used for
        ranking is not what the client sees.

        Ref: stdapi/routes/cohere_rerank_v1.py:_echo_document_text
        """
        response = app_client.post(
            "/cohere/v1/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": ["a", {"title": "Nevada", "body": "Carson City."}],
                "rank_fields": ["title"],
                "return_documents": True,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["results"] == [
            {
                "document": {"text": "title: Nevada\nbody: Carson City."},
                "index": 1,
                "relevance_score": 0.98,
            }
        ]

    def test_return_documents_echoes_documents(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """return_documents=true echoes the input text back in each result.

        A single-key ``{"text": ...}`` document echoes its value verbatim, exactly
        like a plain string document.

        Ref: https://docs.cohere.com/v1/reference/rerank
             stdapi/routes/cohere_rerank_v1.py:_echo_document_text
        """
        response = app_client.post(
            "/cohere/v1/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": ["Carson City", {"text": "Washington, D.C."}],
                "return_documents": True,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["results"] == [
            {
                "document": {"text": "Washington, D.C."},
                "index": 1,
                "relevance_score": 0.98,
            }
        ]

    def test_documents_are_not_echoed_by_default(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """Without return_documents, results carry only index and score.

        ``return_documents`` defaults to false, and the omitted ``document`` field is
        stripped from the payload rather than serialised as null.

        Ref: https://docs.cohere.com/v1/reference/rerank
             stdapi/types/cohere_rerank.py:RerankV1Result
        """
        response = app_client.post(
            "/cohere/v1/rerank",
            json={"model": "cohere.rerank-v3-5:0", "query": "q", "documents": ["a"]},
        )
        assert response.status_code == 200, response.text
        assert response.json()["results"] == [{"index": 1, "relevance_score": 0.98}]

    def test_default_rank_fields_value_is_accepted(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """rank_fields=["text"] leaves string documents untouched.

        ``["text"]`` is Cohere's documented default; projection only applies to
        object documents, so a string document reaches the backend unchanged.

        Ref: https://docs.cohere.com/v1/reference/rerank
             stdapi/routes/cohere_rerank_v1.py:_project_document
        """
        response = app_client.post(
            "/cohere/v1/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": ["a"],
                "rank_fields": ["text"],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["results"] == [{"index": 1, "relevance_score": 0.98}]
        (call,) = rerank_backend.calls
        assert call["documents"] == ["a"]

    @pytest.mark.parametrize("rank_fields", [["title"], ["title", "text"], []])
    def test_custom_rank_fields_are_accepted(
        self,
        app_client: TestClient,
        rerank_backend: _StubRerankModel,
        rank_fields: list[str],
    ) -> None:
        """rank_fields selects which object fields are ranked, in the listed order.

        An empty ``rank_fields`` list projects the document down to an empty object
        rather than falling back to ranking every field.

        Ref: https://docs.cohere.com/v1/reference/rerank
             stdapi/routes/cohere_rerank_v1.py:_project_document
        """
        response = app_client.post(
            "/cohere/v1/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": [{"title": "a", "text": "b"}],
                "rank_fields": rank_fields,
            },
        )
        assert response.status_code == 200, response.text
        expected: dict[tuple[str, ...], list[dict[str, str]]] = {
            ("title",): [{"title": "a"}],
            ("title", "text"): [{"title": "a", "text": "b"}],
            (): [{}],
        }
        (call,) = rerank_backend.calls
        assert call["documents"] == expected[tuple(rank_fields)]

    def test_max_chunks_per_doc_is_rejected(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """max_chunks_per_doc has no Bedrock equivalent and is rejected with a 400.

        v1's chunking knob (Cohere default 10) has no Rerank API counterpart, so the
        gateway fails loudly instead of silently ignoring it — unlike v2's
        ``max_tokens_per_doc``, which is forwarded.

        Ref: https://docs.cohere.com/v1/reference/rerank
             stdapi/routes/cohere_rerank_v1.py:rerank_v1
        """
        response = app_client.post(
            "/cohere/v1/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "q",
                "documents": ["a"],
                "max_chunks_per_doc": 10,
            },
        )
        assert response.status_code == 400
        body = response.json()
        assert set(body) == {"message", "id"}
        assert (
            body["message"]
            == "'max_chunks_per_doc' is not supported on this implementation."
        )
        assert response.headers["x-request-id"] == body["id"]
        assert not rerank_backend.calls

    def test_unknown_model_returns_cohere_error_envelope(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """An unknown model yields a 404 in the Cohere ``{message, id}`` envelope.

        Ref: https://docs.cohere.com/reference/errors
             stdapi/api_providers/cohere.py:_format_error
        """
        response = app_client.post(
            "/cohere/v1/rerank",
            json={"model": "unknown-model", "query": "q", "documents": ["a"]},
        )
        assert response.status_code == 404
        body = response.json()
        assert set(body) == {"message", "id"}
        assert "unknown-model" in body["message"]
        assert "does not exist" in body["message"]
        assert response.headers["x-request-id"] == body["id"]
        assert not rerank_backend.calls

    def test_empty_documents_are_rejected(
        self, app_client: TestClient, rerank_backend: _StubRerankModel
    ) -> None:
        """An empty ``documents`` list is a 400 naming ``body.documents``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html
             stdapi/types/cohere_rerank.py:RerankV1Request
        """
        response = app_client.post(
            "/cohere/v1/rerank",
            json={"model": "cohere.rerank-v3-5:0", "query": "q", "documents": []},
        )
        assert response.status_code == 400
        body = response.json()
        assert set(body) == {"message", "id"}
        assert body["message"].startswith("Validation error at body.documents:")
        assert not rerank_backend.calls


#: Documents with exactly one relevant answer (index 1) for the live tests.
_LIVE_DOCUMENTS = [
    "Carson City is the capital city of Nevada.",
    "Washington, D.C. is the capital of the United States.",
    "Capital punishment has existed in the United States since colonial times.",
]


@pytest.fixture(params=RERANK_MODELS)
def live_rerank_model(
    request: pytest.FixtureRequest, use_official_api: bool, cohere_rerank_model: str
) -> str:
    """Every Bedrock rerank model, or the official Cohere model once."""
    if use_official_api:
        if request.param != RERANK_MODELS[0]:
            pytest.skip("a single model covers the official Cohere API")
        return cohere_rerank_model
    return str(request.param)


class TestCohereRerankIntegration:
    """Live /v2/rerank calls through the official Cohere SDK.

    Ref: https://docs.cohere.com/v2/reference/rerank
         stdapi/routes/cohere_rerank.py:rerank
    """

    def test_rerank_orders_documents(
        self, cohere_client: cohere.ClientV2, live_rerank_model: str
    ) -> None:
        """The most relevant document ranks first and top_n limits results.

        Results reference the input documents by index, are ordered by decreasing
        relevance and carry scores normalised to [0, 1]; index 1 is the only
        document that answers the query.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html
        """
        response = cohere_client.rerank(
            model=live_rerank_model,
            query="What is the capital of the United States?",
            documents=_LIVE_DOCUMENTS,
            top_n=2,
        )
        assert response.id
        assert len(response.results) == 2
        assert response.results[0].index == 1
        indexes = [result.index for result in response.results]
        assert len(set(indexes)) == 2
        assert set(indexes) <= set(range(len(_LIVE_DOCUMENTS)))
        scores = [result.relevance_score for result in response.results]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= score <= 1.0 for score in scores)
        assert response.meta is not None
        assert response.meta.api_version is not None
        assert response.meta.api_version.version == "2"
        assert response.meta.billed_units is not None
        assert response.meta.billed_units.search_units == 1

    def test_unsupported_extra_field_returns_clean_error(
        self, cohere_client: cohere.ClientV2, use_official_api: bool
    ) -> None:
        """AWS's rejection of an unsupported model field surfaces as a Cohere 400.

        ``amazon.rerank-v1:0`` has no ``max_tokens_per_doc`` parameter, so Bedrock
        answers the forwarded ``additionalModelRequestFields`` with a
        ``ValidationException`` that the gateway maps to a 400 carrying the Cohere
        ``{message, id}`` envelope rather than a 5xx.

        Ref: https://docs.cohere.com/reference/errors
             stdapi/aws_bedrock.py:AWS_ERROR_MAP
             stdapi/api_providers/cohere.py:_format_error
        """
        if use_official_api:
            pytest.skip("Bedrock-specific model")
        with pytest.raises(cohere.BadRequestError) as excinfo:
            cohere_client.rerank(
                model="amazon.rerank-v1:0",
                query="q",
                documents=["a"],
                max_tokens_per_doc=512,
            )
        assert excinfo.value.status_code == 400
        body = excinfo.value.body
        assert isinstance(body, dict), body
        assert set(body) == {"message", "id"}
        assert body["message"], "the AWS validation message must reach the client"
        assert body["id"]

    def test_over_one_hundred_documents_bill_two_units(
        self, cohere_client: cohere.ClientV2, live_rerank_model: str
    ) -> None:
        """A query with more than 100 documents is billed two search units.

        One search unit covers a query with up to 100 documents, so 120 documents
        start a second batch.

        Ref: stdapi/models/rerank/bedrock_rerank.py:RerankModel
        """
        response = cohere_client.rerank(
            model=live_rerank_model,
            query="the capital of the United States",
            documents=[f"Fact {i}: water is wet." for i in range(120)],
            top_n=1,
        )
        assert len(response.results) == 1
        assert response.meta is not None
        assert response.meta.billed_units is not None
        assert response.meta.billed_units.search_units == 2


@pytest.fixture(scope="session")
def cohere_client_v1(
    request: pytest.FixtureRequest, test_client: TestClient | None, api_key: str
) -> cohere.Client:
    """Create a Cohere v1 client for either local or official API testing."""
    if test_client:
        return cohere.Client(
            api_key=api_key,
            base_url="http://testserver/cohere",
            httpx_client=test_client,
        )
    if request.config.getoption("--use-official-api"):
        if not getenv("CO_API_KEY"):
            pytest.skip("CO_API_KEY is required to test the official Cohere API")
        return cohere.Client()
    return cohere.Client(
        api_key=getenv("OPENAI_API_KEY", ""),
        base_url=f"{request.config.getoption('--server-url').rstrip('/')}/cohere",
    )


class TestCohereRerankV1Integration:
    """Live /v1/rerank calls through the official Cohere SDK.

    Ref: https://docs.cohere.com/v1/reference/rerank
         stdapi/routes/cohere_rerank_v1.py:rerank_v1
    """

    def test_rerank_orders_documents(
        self, cohere_client_v1: cohere.Client, live_rerank_model: str
    ) -> None:
        """The most relevant document ranks first and top_n limits results.

        The v1 surface reports ``meta.api_version.version`` "1" while sharing the
        v2 backend and its search-unit billing.

        Ref: stdapi/types/cohere.py:ApiMeta
        """
        response = cohere_client_v1.rerank(
            model=live_rerank_model,
            query="What is the capital of the United States?",
            documents=_LIVE_DOCUMENTS,
            top_n=2,
        )
        assert response.id
        assert len(response.results) == 2
        assert response.results[0].index == 1
        indexes = [result.index for result in response.results]
        assert len(set(indexes)) == 2
        assert set(indexes) <= set(range(len(_LIVE_DOCUMENTS)))
        scores = [result.relevance_score for result in response.results]
        assert scores == sorted(scores, reverse=True)
        assert all(0.0 <= score <= 1.0 for score in scores)
        assert response.meta is not None
        assert response.meta.api_version is not None
        assert response.meta.api_version.version == "1"
        assert response.meta.billed_units is not None
        assert response.meta.billed_units.search_units == 1

    def test_return_documents_echoes_documents(
        self, cohere_client_v1: cohere.Client, live_rerank_model: str
    ) -> None:
        """return_documents=true echoes the document text back in each result.

        The echoed text is looked up by the result's ``index`` in the original
        request, so it must match the input document at that position.

        Ref: https://docs.cohere.com/v1/reference/rerank
             stdapi/routes/cohere_rerank_v1.py:_echo_document_text
        """
        response = cohere_client_v1.rerank(
            model=live_rerank_model,
            query="What is the capital of the United States?",
            documents=[{"text": text} for text in _LIVE_DOCUMENTS],
            top_n=1,
            return_documents=True,
        )
        assert len(response.results) == 1
        result = response.results[0]
        assert result.document is not None
        assert result.document.text == _LIVE_DOCUMENTS[result.index]

    def test_multi_field_object_documents_with_rank_fields(
        self, cohere_client_v1: cohere.Client, live_rerank_model: str
    ) -> None:
        """Multi-field object documents are sent as jsonDocument and ranked correctly.

        Bedrock ranks the structured document natively; the gateway echoes the
        original object back as ``key: value`` lines, which is gateway-defined since
        Cohere never documents its own join for object documents.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html
             stdapi/models/rerank/bedrock_rerank.py:RerankModel
             stdapi/routes/cohere_rerank_v1.py:_echo_document_text
        """
        documents = [
            {"title": "Nevada", "text": _LIVE_DOCUMENTS[0]},
            {"title": "United States", "text": _LIVE_DOCUMENTS[1]},
            {"title": "United States history", "text": _LIVE_DOCUMENTS[2]},
        ]
        response = cohere_client_v1.rerank(
            model=live_rerank_model,
            query="What is the capital of the United States?",
            documents=documents,
            rank_fields=["title", "text"],
            top_n=1,
            return_documents=True,
        )
        assert len(response.results) == 1
        result = response.results[0]
        assert result.index == 1
        assert result.document is not None
        assert (
            result.document.text == "title: United States\ntext: " + _LIVE_DOCUMENTS[1]
        )
