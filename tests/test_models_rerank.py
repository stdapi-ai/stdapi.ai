"""Bedrock rerank models: dispatch, capabilities, Rerank API calls, usage.

Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html
     stdapi/models/rerank/bedrock_rerank.py:RerankModel
     stdapi/models/rerank/__init__.py:get_rerank_model
"""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError

import stdapi.aws
import stdapi.main
import stdapi.models
import stdapi.region_routing
from stdapi.api_errors import UnsupportedModelError
from stdapi.config import SETTINGS
from stdapi.models import (
    RERANKING_MODALITY,
    ModelDetails,
    _advertised_output_modalities,
    _compute_model_capabilities,
)
from stdapi.models.capabilities import Capability
from stdapi.models.rerank import bedrock_rerank, get_rerank_model
from stdapi.models.rerank.bedrock_rerank import RerankModel
from stdapi.monitoring import REQUEST_ID, REQUEST_LOG, EventLog
from stdapi.pricing import Dimension
from stdapi.usage import USAGE, init_model_state, init_usage

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from types_aiobotocore_bedrock.literals import RegionName


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


#: Both Bedrock rerank model families reachable through the Rerank API.
RERANK_MODELS = ("amazon.rerank-v1:0", "cohere.rerank-v3-5:0")


class TestRerankModelDispatch:
    """Rerank model IDs must resolve to the Rerank API backend class.

    Ref: stdapi/models/rerank/__init__.py:get_rerank_model
         stdapi/models/rerank/bedrock_rerank.py:RerankModel
    """

    @pytest.mark.parametrize("model_id", RERANK_MODELS)
    def test_matcher_dispatch(self, model_id: str) -> None:
        """The registry resolves rerank IDs to a cached RERANK-capable backend.

        ``RerankModel.MATCHER`` covers both the Amazon and Cohere rerank families,
        and the registry caches one instance per model ID.

        Ref: stdapi/models/rerank/bedrock_rerank.py:RerankModel
        """
        model = get_rerank_model(model_id)
        assert type(model) is RerankModel
        assert model.get_supported_operations() == Capability.RERANK
        assert get_rerank_model(model_id) is model

    def test_non_rerank_model_is_rejected(self) -> None:
        """A chat model ID has no rerank backend and raises a 404 model_not_found.

        Ref: stdapi/api_errors.py:UnsupportedModelError
             stdapi/models/rerank/__init__.py:get_rerank_model
        """
        with pytest.raises(UnsupportedModelError) as excinfo:
            get_rerank_model("anthropic.claude-3-5-haiku-20241022-v1:0")
        assert excinfo.value.status == 404
        assert excinfo.value.code == "model_not_found"
        assert "anthropic.claude-3-5-haiku-20241022-v1:0" in str(excinfo.value)


class TestRerankSupportedRoutes:
    """Rerank models advertise the rerank route only; text models never do.

    Bedrock's ModelModality enum has no rerank value, so the gateway derives its
    own RERANKING modality from the model ID.

    Ref: stdapi/models/__init__.py:_advertised_output_modalities
         stdapi/models/__init__.py:_compute_model_capabilities
    """

    @staticmethod
    def _details(model_id: str, output_modalities: list[str]) -> ModelDetails:
        return ModelDetails(
            id=model_id,
            name=model_id,
            provider="Vendor",
            input_modalities=["TEXT"],
            output_modalities=output_modalities,
            regions=["us-east-1"],
        )

    @pytest.mark.parametrize("model_id", RERANK_MODELS)
    def test_reranking_output_modality_advertised(self, model_id: str) -> None:
        """The TEXT output modality from the Bedrock listing becomes RERANKING.

        Ref: stdapi/models/__init__.py:_advertised_output_modalities
        """
        assert _advertised_output_modalities(model_id, ["TEXT"]) == [RERANKING_MODALITY]

    def test_text_models_keep_listed_output_modalities(self) -> None:
        """Non-rerank models keep the modalities from the Bedrock listing.

        Ref: stdapi/models/__init__.py:_advertised_output_modalities
        """
        model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"
        assert _advertised_output_modalities(model_id, ["TEXT"]) == ["TEXT"]

    @pytest.mark.parametrize("model_id", RERANK_MODELS)
    def test_rerank_route_advertised(self, model_id: str) -> None:
        """A RERANKING model advertises both rerank routes and nothing else.

        A TEXT-in / RERANKING-out model matches exactly the two Cohere rerank
        routes, so no chat, responses or messages route is advertised.

        Ref: stdapi/routes/cohere_rerank.py:rerank
             stdapi/routes/cohere_rerank_v1.py:rerank_v1
        """
        routes, tools = _compute_model_capabilities(
            model_id, self._details(model_id, [RERANKING_MODALITY])
        )
        prefix = SETTINGS.cohere_routes_prefix
        assert routes == [f"{prefix}/v1/rerank", f"{prefix}/v2/rerank"]
        assert tools == ["cohere_rerank", "cohere_rerank_v1"]

    def test_text_models_do_not_advertise_rerank(self) -> None:
        """A TEXT/TEXT model without the RERANK capability skips both rerank routes.

        The chat tools stay advertised, so the exclusion comes from the missing
        RERANKING output modality and not from an empty capability computation.

        Ref: stdapi/models/__init__.py:_compute_model_capabilities
        """
        model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"
        routes, tools = _compute_model_capabilities(
            model_id, self._details(model_id, ["TEXT"])
        )
        assert "cohere_rerank" not in tools
        assert "cohere_rerank_v1" not in tools
        assert not any(route.endswith("rerank") for route in routes)
        assert "openai_chat_completion" in tools


class _StubAgentRuntimeClient:
    """Stub bedrock-agent-runtime client returning pre-defined result pages."""

    def __init__(self, pages: list[dict[str, Any] | Exception]) -> None:
        self._pages = list(pages)
        self.requests: list[dict[str, Any]] = []

    async def rerank(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        """Record the request and return (or raise) the next page."""
        self.requests.append(params)
        page = self._pages.pop(0)
        if isinstance(page, Exception):
            raise page
        return page


def _bedrock_config(request: dict[str, Any]) -> dict[str, Any]:
    """Return the Bedrock-specific block of one recorded Rerank request.

    Args:
        request: A request recorded by ``_StubAgentRuntimeClient``.

    Returns:
        ``rerankingConfiguration.bedrockRerankingConfiguration``.
    """
    return request["rerankingConfiguration"]["bedrockRerankingConfiguration"]  # type: ignore[no-any-return]


def _throttling_error() -> ClientError:
    response: Any = {
        "Error": {"Code": "ThrottlingException", "Message": "Throttled"},
        "ResponseMetadata": {"HTTPStatusCode": 429},
    }
    return ClientError(response, "Rerank")


def _new_log() -> EventLog:
    return EventLog(
        type="request",
        level="info",
        date=datetime.now(UTC),
        server_id="test",
        server_version="0.0.0",
    )


class TestRerankCall:
    """RerankModel.rerank: request building, pagination, usage recording.

    Rerank runs on bedrock-agent-runtime (POST /rerank), not bedrock-runtime, and
    is billed per search unit rather than per token.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html
         stdapi/models/rerank/bedrock_rerank.py:RerankModel
    """

    @pytest.fixture(autouse=True)
    def _request_context(self) -> Generator[None]:
        """Provide request ID/log and fresh usage state for each test."""
        id_token = REQUEST_ID.set("req1")
        log_token = REQUEST_LOG.set(_new_log())
        usage_token = init_usage()
        init_model_state()
        yield
        USAGE.reset(usage_token)
        REQUEST_LOG.reset(log_token)
        REQUEST_ID.reset(id_token)

    @pytest.fixture
    def rerank_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> Callable[..., _StubAgentRuntimeClient]:
        """Return a factory that stubs the agent-runtime client for one pinned region.

        Returns:
            ``make(pages, region="us-east-1")`` -> the stub whose ``requests``
            list records every Rerank call the model issued.
        """

        def _make(
            pages: list[dict[str, Any] | Exception], region: str = "us-east-1"
        ) -> _StubAgentRuntimeClient:
            client = _StubAgentRuntimeClient(pages)

            async def _candidates(_model_id: str) -> list[RegionName]:
                return [region]  # type: ignore[list-item]

            monkeypatch.setattr(
                bedrock_rerank, "compute_candidate_regions", _candidates
            )
            monkeypatch.setattr(
                bedrock_rerank, "get_client", lambda _service, _region: client
            )
            return client

        return _make

    async def test_rerank_maps_results_and_bills_one_unit(
        self, rerank_client: Callable[..., _StubAgentRuntimeClient]
    ) -> None:
        """Results map to (index, relevance_score) and one search unit is billed.

        The AWS request carries exactly one ``queries`` entry, one INLINE source per
        document and the region-scoped foundation-model ARN; the camelCase
        ``relevanceScore`` becomes the snake_case ``relevance_score``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html
             stdapi/usage.py:record_bedrock_usage
        """
        client = rerank_client([{"results": [{"index": 1, "relevanceScore": 0.9}]}])

        response = await RerankModel(RERANK_MODELS[1]).rerank(
            "query", ["a", "b"], top_n=None, extra_params={}
        )

        assert [(r.index, r.relevance_score) for r in response.results] == [(1, 0.9)]
        assert response.search_units == 1
        (request,) = client.requests
        assert "nextToken" not in request
        assert request["queries"] == [{"type": "TEXT", "textQuery": {"text": "query"}}]
        assert [source["type"] for source in request["sources"]] == ["INLINE", "INLINE"]
        assert [
            source["inlineDocumentSource"]["textDocument"]["text"]
            for source in request["sources"]
        ] == ["a", "b"]
        assert request["rerankingConfiguration"]["type"] == "BEDROCK_RERANKING_MODEL", (
            "the reranking configuration must be discriminated for Bedrock models"
        )
        configuration = _bedrock_config(request)
        assert configuration["numberOfResults"] == 2
        assert configuration["modelConfiguration"] == {
            "modelArn": f"arn:aws:bedrock:us-east-1::foundation-model/{RERANK_MODELS[1]}"
        }
        records = list(USAGE.get().values())
        assert len(records) == 1
        assert records[0].quantities == {Dimension.SEARCH_UNITS: 1}

    async def test_top_n_caps_number_of_results(
        self, rerank_client: Callable[..., _StubAgentRuntimeClient]
    ) -> None:
        """NumberOfResults is min(top_n, document count).

        Cohere's ``top_n`` merely limits the returned results and may exceed the
        number of documents, so the gateway clamps it before sending it as
        ``numberOfResults``.

        Ref: https://docs.cohere.com/v2/reference/rerank
             stdapi/models/rerank/bedrock_rerank.py:RerankModel
        """
        client = rerank_client([{"results": []}, {"results": []}])
        model = RerankModel(RERANK_MODELS[0])

        await model.rerank("q", ["a", "b", "c"], top_n=2, extra_params={})
        await model.rerank("q", ["a", "b"], top_n=9, extra_params={})

        counts = [
            _bedrock_config(request)["numberOfResults"] for request in client.requests
        ]
        assert counts == [2, 2]

    async def test_top_n_zero_requests_zero_results(
        self, rerank_client: Callable[..., _StubAgentRuntimeClient]
    ) -> None:
        """top_n=0 is honoured literally; only None means all documents.

        Reachable only from internal callers: both Cohere request models constrain
        ``top_n`` to ``>= 1``, and AWS's ``numberOfResults`` minimum is 1.

        Ref: stdapi/types/cohere_rerank.py:RerankRequest
             botocore/data/bedrock-agent-runtime/2023-07-26/service-2.json
        """
        client = rerank_client([{"results": []}])

        await RerankModel(RERANK_MODELS[0]).rerank(
            "q", ["a", "b"], top_n=0, extra_params={}
        )

        configuration = _bedrock_config(client.requests[0])
        assert configuration["numberOfResults"] == 0

    async def test_throttled_region_fails_over_via_no_retry_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A throttled region escalates to the next one through the no-retry client pool.

        With region routing enabled the failover budget belongs to
        ``route_and_execute``, so the call must resolve the ``.no-retry``
        bedrock-agent-runtime pool and bill usage against the serving region.

        Ref: stdapi/models/__init__.py:route_and_execute
             stdapi/aws.py:_NO_RETRY_SERVICES
             stdapi/models/rerank/bedrock_rerank.py:agent_runtime_client
        """
        regions = ["us-east-1", "us-west-2"]
        throttled = _StubAgentRuntimeClient([_throttling_error()])
        serving = _StubAgentRuntimeClient(
            [{"results": [{"index": 0, "relevanceScore": 0.7}]}]
        )

        async def _candidates(_model_id: str) -> list[RegionName]:
            return list(regions)  # type: ignore[arg-type]

        monkeypatch.setattr(bedrock_rerank, "compute_candidate_regions", _candidates)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_region_routing", "ordered")
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", regions)
        monkeypatch.setattr(
            stdapi.models, "REGION_ROUTER", stdapi.region_routing.RegionRouter()
        )
        # Same seam as the chat no-retry tests: clients resolve through the real
        # get_client over injected _CLIENTS pools. The full-retry pool holds
        # non-client sentinels so any use of it fails loudly.
        monkeypatch.setitem(
            stdapi.aws._CLIENTS,  # noqa: SLF001
            "bedrock-agent-runtime",
            dict.fromkeys(regions, object()),
        )
        monkeypatch.setitem(
            stdapi.aws._CLIENTS,  # noqa: SLF001
            "bedrock-agent-runtime.no-retry",
            {"us-east-1": throttled, "us-west-2": serving},
        )

        response = await RerankModel(RERANK_MODELS[0]).rerank(
            "q", ["a"], top_n=None, extra_params={}
        )

        assert [(r.index, r.relevance_score) for r in response.results] == [(0, 0.7)]
        assert response.search_units == 1
        assert len(throttled.requests) == 1
        assert len(serving.requests) == 1
        assert throttled.requests[0]["sources"] == serving.requests[0]["sources"]
        (record,) = USAGE.get().values()
        assert record.region == "us-west-2"
        assert record.quantities == {Dimension.SEARCH_UNITS: 1}

    async def test_pagination_follows_next_token(
        self, rerank_client: Callable[..., _StubAgentRuntimeClient]
    ) -> None:
        """Paginated results are concatenated and billed once.

        The Rerank API returns ``nextToken`` when the results do not fit in one
        response; the follow-up call replays the same query and sources with the
        token added, and billing stays per query, not per page.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html
        """
        client = rerank_client(
            [
                {
                    "results": [{"index": 0, "relevanceScore": 0.8}],
                    "nextToken": "token",
                },
                {"results": [{"index": 1, "relevanceScore": 0.2}]},
            ]
        )

        response = await RerankModel(RERANK_MODELS[1]).rerank(
            "q", ["a", "b"], top_n=None, extra_params={}
        )

        assert [(r.index, r.relevance_score) for r in response.results] == [
            (0, 0.8),
            (1, 0.2),
        ]
        assert "nextToken" not in client.requests[0]
        assert client.requests[1]["nextToken"] == "token"
        assert client.requests[1]["queries"] == client.requests[0]["queries"]
        assert client.requests[1]["sources"] == client.requests[0]["sources"]
        assert response.search_units == 1
        (record,) = USAGE.get().values()
        assert record.quantities == {Dimension.SEARCH_UNITS: 1}

    async def test_search_units_scale_with_document_count(
        self, rerank_client: Callable[..., _StubAgentRuntimeClient]
    ) -> None:
        """One search unit is billed per started batch of 100 documents.

        Billing is derived from the document count, not from the number of returned
        results, so ``top_n=1`` over 150 documents still costs two units.

        Ref: stdapi/models/rerank/bedrock_rerank.py:RerankModel
             stdapi/usage.py:record_bedrock_usage
        """
        client = rerank_client([{"results": []}])

        response = await RerankModel(RERANK_MODELS[1]).rerank(
            "q", [f"doc{i}" for i in range(150)], top_n=1, extra_params={}
        )

        assert response.search_units == 2
        assert len(client.requests[0]["sources"]) == 150
        assert _bedrock_config(client.requests[0])["numberOfResults"] == 1
        records = list(USAGE.get().values())
        assert records[0].quantities == {Dimension.SEARCH_UNITS: 2}

    async def test_extra_params_forwarded_as_additional_fields(
        self, rerank_client: Callable[..., _StubAgentRuntimeClient]
    ) -> None:
        """Extra model parameters land in additionalModelRequestFields.

        The Rerank API exposes no per-model knobs of its own, so options such as
        v2's ``max_tokens_per_doc`` travel alongside the model ARN inside
        ``modelConfiguration``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html
        """
        client = rerank_client([{"results": []}])

        await RerankModel(RERANK_MODELS[1]).rerank(
            "q", ["a"], top_n=None, extra_params={"max_tokens_per_doc": 512}
        )

        configuration = _bedrock_config(client.requests[0])["modelConfiguration"]
        assert configuration["additionalModelRequestFields"] == {
            "max_tokens_per_doc": 512
        }
        assert configuration["modelArn"].endswith(
            f"foundation-model/{RERANK_MODELS[1]}"
        )

    async def test_single_key_text_object_uses_text_document_fast_path(
        self, rerank_client: Callable[..., _StubAgentRuntimeClient]
    ) -> None:
        """A ``{"text": ...}`` object document keeps the plain textDocument wire shape.

        v1 lets clients wrap a single text in an object; unwrapping it keeps the
        Bedrock request identical to the plain-string form.

        Ref: stdapi/models/rerank/bedrock_rerank.py:_document_source
        """
        client = rerank_client([{"results": []}])

        await RerankModel(RERANK_MODELS[1]).rerank(
            "q", [{"text": "a"}], top_n=None, extra_params={}
        )

        (source,) = client.requests[0]["sources"]
        assert source["inlineDocumentSource"] == {
            "type": "TEXT",
            "textDocument": {"text": "a"},
        }

    async def test_multi_field_object_uses_json_document_source(
        self, rerank_client: Callable[..., _StubAgentRuntimeClient]
    ) -> None:
        """A multi-field object document is sent as a Bedrock jsonDocument source.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html
             stdapi/models/rerank/bedrock_rerank.py:_document_source
        """
        client = rerank_client([{"results": []}])
        document: dict[str, Any] = {
            "title": "Nevada",
            "body": "Carson City is its capital.",
        }

        await RerankModel(RERANK_MODELS[1]).rerank(
            "q", [document], top_n=None, extra_params={}
        )

        (source,) = client.requests[0]["sources"]
        assert source["inlineDocumentSource"] == {
            "type": "JSON",
            "jsonDocument": document,
        }

    async def test_object_without_text_key_uses_json_document_source(
        self, rerank_client: Callable[..., _StubAgentRuntimeClient]
    ) -> None:
        """An object document without a ``text`` key is sent as a jsonDocument source.

        Ref: stdapi/models/rerank/bedrock_rerank.py:_document_source
        """
        client = rerank_client([{"results": []}])

        await RerankModel(RERANK_MODELS[1]).rerank(
            "q", [{"title": "Nevada"}], top_n=None, extra_params={}
        )

        (source,) = client.requests[0]["sources"]
        assert source["inlineDocumentSource"] == {
            "type": "JSON",
            "jsonDocument": {"title": "Nevada"},
        }

    async def test_mixed_string_and_object_documents(
        self, rerank_client: Callable[..., _StubAgentRuntimeClient]
    ) -> None:
        """String and object documents in the same request build distinct sources.

        The source type is decided per document and the request order matches the
        input order, which is what result ``index`` values refer to.

        Ref: stdapi/models/rerank/bedrock_rerank.py:_document_source
        """
        client = rerank_client([{"results": []}])

        await RerankModel(RERANK_MODELS[1]).rerank(
            "q", ["a", {"title": "b"}], top_n=None, extra_params={}
        )

        sources = [
            source["inlineDocumentSource"] for source in client.requests[0]["sources"]
        ]
        assert sources == [
            {"type": "TEXT", "textDocument": {"text": "a"}},
            {"type": "JSON", "jsonDocument": {"title": "b"}},
        ]

    async def test_model_arn_uses_region_partition(
        self, rerank_client: Callable[..., _StubAgentRuntimeClient]
    ) -> None:
        """The model ARN partition follows the serving region.

        The European Sovereign Cloud lives in the ``aws-eusc`` partition, so a
        hard-coded ``aws`` partition would make every rerank call fail there.

        Ref: stdapi/pricing.py:partition_of_region
             stdapi/models/rerank/bedrock_rerank.py:RerankModel
        """
        client = rerank_client([{"results": []}], region="eusc-de-east-1")

        await RerankModel(RERANK_MODELS[1]).rerank(
            "q", ["a"], top_n=None, extra_params={}
        )

        arn = _bedrock_config(client.requests[0])["modelConfiguration"]["modelArn"]
        assert arn == (
            f"arn:aws-eusc:bedrock:eusc-de-east-1::foundation-model/{RERANK_MODELS[1]}"
        )
