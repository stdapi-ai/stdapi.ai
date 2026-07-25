"""Bedrock rerank models: dispatch, capabilities, Rerank API calls, usage."""

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
from stdapi.models.rerank import bedrock_rerank, get_rerank_model
from stdapi.models.rerank.bedrock_rerank import RerankModel
from stdapi.monitoring import REQUEST_ID, REQUEST_LOG, EventLog
from stdapi.pricing import Dimension
from stdapi.usage import USAGE, init_model_state, init_usage

if TYPE_CHECKING:
    from collections.abc import Generator

    from types_aiobotocore_bedrock.literals import RegionName


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


RERANK_MODELS = ("amazon.rerank-v1:0", "cohere.rerank-v3-5:0")


class TestRerankModelDispatch:
    """Rerank model IDs must resolve to the Rerank API backend class."""

    @pytest.mark.parametrize("model_id", RERANK_MODELS)
    def test_matcher_dispatch(self, model_id: str) -> None:
        """The registry resolves rerank IDs to the rerank backend class."""
        assert type(get_rerank_model(model_id)) is RerankModel

    def test_non_rerank_model_is_rejected(self) -> None:
        """Non-rerank model IDs have no rerank backend."""
        with pytest.raises(UnsupportedModelError):
            get_rerank_model("anthropic.claude-3-5-haiku-20241022-v1:0")


class TestRerankSupportedRoutes:
    """Rerank models advertise the rerank route only; text models never do."""

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
        """The TEXT output modality from the Bedrock listing becomes RERANKING."""
        assert _advertised_output_modalities(model_id, ["TEXT"]) == [RERANKING_MODALITY]

    def test_text_models_keep_listed_output_modalities(self) -> None:
        """Non-rerank models keep the modalities from the Bedrock listing."""
        model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"
        assert _advertised_output_modalities(model_id, ["TEXT"]) == ["TEXT"]

    @pytest.mark.parametrize("model_id", RERANK_MODELS)
    def test_rerank_route_advertised(self, model_id: str) -> None:
        """supported_routes includes /v2/rerank and excludes text routes."""
        routes, tools = _compute_model_capabilities(
            model_id, self._details(model_id, [RERANKING_MODALITY])
        )
        assert any(route.endswith("/v2/rerank") for route in routes)
        assert "cohere_rerank" in tools
        assert not any("chat/completions" in route for route in routes)
        assert not any("responses" in route for route in routes)
        assert not any("messages" in route for route in routes)
        assert "openai_chat_completion" not in tools

    def test_text_models_do_not_advertise_rerank(self) -> None:
        """A TEXT/TEXT model without the RERANK capability skips the route."""
        model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"
        _, tools = _compute_model_capabilities(
            model_id, self._details(model_id, ["TEXT"])
        )
        assert "cohere_rerank" not in tools


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
    """RerankModel.rerank: request building, pagination, usage recording."""

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

    @staticmethod
    def _patch_infra(
        monkeypatch: pytest.MonkeyPatch,
        client: _StubAgentRuntimeClient,
        region: str = "us-east-1",
    ) -> None:
        """Pin the candidate region and stub the agent-runtime client."""

        async def _candidates(_model_id: str) -> list[RegionName]:
            return [region]  # type: ignore[list-item]

        monkeypatch.setattr(bedrock_rerank, "compute_candidate_regions", _candidates)
        monkeypatch.setattr(
            bedrock_rerank, "get_client", lambda _service, _region: client
        )

    async def test_rerank_maps_results_and_bills_one_unit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Results map to (index, relevance_score) and one search unit is billed."""
        client = _StubAgentRuntimeClient(
            [{"results": [{"index": 1, "relevanceScore": 0.9}]}]
        )
        self._patch_infra(monkeypatch, client)

        response = await RerankModel(RERANK_MODELS[1]).rerank(
            "query", ["a", "b"], top_n=None, extra_params={}
        )

        assert [(r.index, r.relevance_score) for r in response.results] == [(1, 0.9)]
        assert response.search_units == 1
        (request,) = client.requests
        assert request["queries"] == [{"type": "TEXT", "textQuery": {"text": "query"}}]
        assert [
            source["inlineDocumentSource"]["textDocument"]["text"]
            for source in request["sources"]
        ] == ["a", "b"]
        configuration = request["rerankingConfiguration"][
            "bedrockRerankingConfiguration"
        ]
        assert configuration["numberOfResults"] == 2
        assert configuration["modelConfiguration"] == {
            "modelArn": f"arn:aws:bedrock:us-east-1::foundation-model/{RERANK_MODELS[1]}"
        }
        records = list(USAGE.get().values())
        assert len(records) == 1
        assert records[0].quantities == {Dimension.SEARCH_UNITS: 1}

    async def test_top_n_caps_number_of_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """NumberOfResults is min(top_n, document count)."""
        client = _StubAgentRuntimeClient([{"results": []}, {"results": []}])
        self._patch_infra(monkeypatch, client)
        model = RerankModel(RERANK_MODELS[0])

        await model.rerank("q", ["a", "b", "c"], top_n=2, extra_params={})
        await model.rerank("q", ["a", "b"], top_n=9, extra_params={})

        counts = [
            request["rerankingConfiguration"]["bedrockRerankingConfiguration"][
                "numberOfResults"
            ]
            for request in client.requests
        ]
        assert counts == [2, 2]

    async def test_top_n_zero_requests_zero_results(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """top_n=0 is honoured literally; only None means all documents."""
        client = _StubAgentRuntimeClient([{"results": []}])
        self._patch_infra(monkeypatch, client)

        await RerankModel(RERANK_MODELS[0]).rerank(
            "q", ["a", "b"], top_n=0, extra_params={}
        )

        configuration = client.requests[0]["rerankingConfiguration"][
            "bedrockRerankingConfiguration"
        ]
        assert configuration["numberOfResults"] == 0

    async def test_throttled_region_fails_over_via_no_retry_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A throttled region escalates to the next one through the no-retry client pool."""
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
        assert len(throttled.requests) == 1
        assert len(serving.requests) == 1
        (record,) = USAGE.get().values()
        assert record.region == "us-west-2"
        assert record.quantities == {Dimension.SEARCH_UNITS: 1}

    async def test_pagination_follows_next_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Paginated results are concatenated and billed once."""
        client = _StubAgentRuntimeClient(
            [
                {
                    "results": [{"index": 0, "relevanceScore": 0.8}],
                    "nextToken": "token",
                },
                {"results": [{"index": 1, "relevanceScore": 0.2}]},
            ]
        )
        self._patch_infra(monkeypatch, client)

        response = await RerankModel(RERANK_MODELS[1]).rerank(
            "q", ["a", "b"], top_n=None, extra_params={}
        )

        assert [r.index for r in response.results] == [0, 1]
        assert "nextToken" not in client.requests[0]
        assert client.requests[1]["nextToken"] == "token"
        assert len(list(USAGE.get().values())) == 1

    async def test_search_units_scale_with_document_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One search unit is billed per started batch of 100 documents."""
        client = _StubAgentRuntimeClient([{"results": []}])
        self._patch_infra(monkeypatch, client)

        response = await RerankModel(RERANK_MODELS[1]).rerank(
            "q", [f"doc{i}" for i in range(150)], top_n=1, extra_params={}
        )

        assert response.search_units == 2
        records = list(USAGE.get().values())
        assert records[0].quantities == {Dimension.SEARCH_UNITS: 2}

    async def test_extra_params_forwarded_as_additional_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Extra model parameters land in additionalModelRequestFields."""
        client = _StubAgentRuntimeClient([{"results": []}])
        self._patch_infra(monkeypatch, client)

        await RerankModel(RERANK_MODELS[1]).rerank(
            "q", ["a"], top_n=None, extra_params={"max_tokens_per_doc": 512}
        )

        configuration = client.requests[0]["rerankingConfiguration"][
            "bedrockRerankingConfiguration"
        ]["modelConfiguration"]
        assert configuration["additionalModelRequestFields"] == {
            "max_tokens_per_doc": 512
        }

    async def test_model_arn_uses_region_partition(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The model ARN partition follows the serving region."""
        client = _StubAgentRuntimeClient([{"results": []}])
        self._patch_infra(monkeypatch, client, region="eusc-de-east-1")

        await RerankModel(RERANK_MODELS[1]).rerank(
            "q", ["a"], top_n=None, extra_params={}
        )

        arn = client.requests[0]["rerankingConfiguration"][
            "bedrockRerankingConfiguration"
        ]["modelConfiguration"]["modelArn"]
        assert arn.startswith("arn:aws-eusc:bedrock:eusc-de-east-1::")
