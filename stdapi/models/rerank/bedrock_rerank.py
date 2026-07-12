"""Bedrock rerank models (Amazon Rerank, Cohere Rerank) served via the Rerank API.

AWS bills rerank by search unit: one unit covers a query with up to 100
documents.
"""

from functools import partial
from re import compile as re_compile
from typing import TYPE_CHECKING

from stdapi.aws import get_client
from stdapi.aws_bedrock import handle_bedrock_client_error
from stdapi.models import (
    compute_candidate_regions,
    route_and_execute,
    set_effective_region,
)
from stdapi.models.rerank import RerankedDocument, RerankModelBase, RerankResponse
from stdapi.pricing import partition_of_region
from stdapi.usage import record_bedrock_usage

if TYPE_CHECKING:
    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_agent_runtime.client import (
        AgentsforBedrockRuntimeClient,
    )
    from types_aiobotocore_bedrock_agent_runtime.type_defs import (
        BedrockRerankingModelConfigurationTypeDef,
        RerankRequestTypeDef,
        RerankResultTypeDef,
    )

    from stdapi.types import JsonMapping


class RerankModel(RerankModelBase):
    """Bedrock rerank model served via the Bedrock Rerank API."""

    MATCHER = re_compile(r"(?:amazon|cohere)\.rerank")

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None,
        extra_params: JsonMapping,
    ) -> RerankResponse:
        """Rank documents by relevance to the query.

        Args:
            query: The search query.
            documents: Texts to compare to the query.
            top_n: Maximum number of results to return, or None for all.
            extra_params: Extra model parameters, passed as
                ``additionalModelRequestFields``.

        Returns:
            Rerank response with billed search units.
        """
        results, region = await route_and_execute(
            self._model_id,
            await compute_candidate_regions(self._model_id),
            partial(self._rerank_in_region, query, documents, top_n, extra_params),
        )
        search_units = (len(documents) + 99) // 100
        record_bedrock_usage(self._model_id, region=region, search_units=search_units)
        return RerankResponse(
            results=[
                RerankedDocument(
                    index=result["index"], relevance_score=result["relevanceScore"]
                )
                for result in results
            ],
            search_units=search_units,
        )

    async def _rerank_in_region(
        self,
        query: str,
        documents: list[str],
        top_n: int | None,
        extra_params: JsonMapping,
        region: RegionName,
    ) -> tuple[list[RerankResultTypeDef], RegionName]:
        """Run one Rerank API call in *region*, following result pagination.

        Args:
            query: The search query.
            documents: Texts to compare to the query.
            top_n: Maximum number of results to return, or None for all.
            extra_params: Extra model parameters.
            region: AWS region to target.

        Returns:
            Tuple of (raw rerank results, region that served the call).
        """
        set_effective_region(self._model_id, region)
        client: AgentsforBedrockRuntimeClient = get_client(
            "bedrock-agent-runtime", region
        )
        model_configuration: BedrockRerankingModelConfigurationTypeDef = {
            "modelArn": (
                f"arn:{partition_of_region(region)}:bedrock:{region}::"
                f"foundation-model/{self._model_id}"
            )
        }
        if extra_params:
            # type-ignore: the stub is narrower than the API's JSON document values.
            model_configuration["additionalModelRequestFields"] = extra_params  # type: ignore[typeddict-item]
        request: RerankRequestTypeDef = {
            "queries": [{"type": "TEXT", "textQuery": {"text": query}}],
            "sources": [
                {
                    "type": "INLINE",
                    "inlineDocumentSource": {
                        "type": "TEXT",
                        "textDocument": {"text": text},
                    },
                }
                for text in documents
            ],
            "rerankingConfiguration": {
                "type": "BEDROCK_RERANKING_MODEL",
                "bedrockRerankingConfiguration": {
                    "numberOfResults": min(top_n or len(documents), len(documents)),
                    "modelConfiguration": model_configuration,
                },
            },
        }
        results: list[RerankResultTypeDef] = []
        with handle_bedrock_client_error():
            while True:
                response = await client.rerank(**request)
                results.extend(response["results"])
                if not (token := response.get("nextToken")):
                    return results, region
                request["nextToken"] = token
