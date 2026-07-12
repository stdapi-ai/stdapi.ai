"""Bedrock rerank models (Amazon Rerank, Cohere Rerank) served via Converse.

AWS bills rerank by search unit (one unit covers a query with up to 100
documents), not by tokens: each Converse call is recorded as one search
unit on top of any token usage AWS reports.
"""

from re import compile as re_compile
from typing import TYPE_CHECKING

from stdapi.models.chat._default import ChatModel as _BaseChatModel
from stdapi.usage import record_bedrock_usage

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime.literals import ServiceTierTypeType
    from types_aiobotocore_bedrock_runtime.type_defs import ConverseResponseTypeDef

    from stdapi.pricing import Routing


class ChatModel(_BaseChatModel):
    """Bedrock rerank model served via Converse, billed per search unit."""

    MATCHER = re_compile(r"(?:amazon|cohere)\.rerank")
    # Converse rejects rerank models (confirmed live): keep them out of the
    # advertised text routes until AWS enables it or a Rerank endpoint exists.
    UNSUPPORTED_OPERATIONS = frozenset(
        {
            "openai_chat_completion",
            "openai_completion",
            "openai_response",
            "openai_response_input_tokens",
            "anthropic_message",
            "anthropic_message_count_tokens",
        }
    )

    def _record_converse_usage(
        self,
        response: ConverseResponseTypeDef,
        grounding_requests: int = 0,
        *,
        region: str = "",
        requested_tier: ServiceTierTypeType | None = None,
        routing: Routing | None = None,
    ) -> None:
        """Record token usage plus one billed search unit for this call.

        Args:
            response: Converse API response with usage metrics.
            grounding_requests: See the base implementation.
            region: See the base implementation.
            requested_tier: See the base implementation.
            routing: See the base implementation.
        """
        super()._record_converse_usage(
            response,
            grounding_requests,
            region=region,
            requested_tier=requested_tier,
            routing=routing,
        )
        # Queries with >100 documents are under-counted: the document count
        # isn't visible at this layer.
        record_bedrock_usage(
            self._model_id,
            tier=(response.get("serviceTier") or {}).get("type") or requested_tier,
            region=region,
            routing=routing,
            search_units=1,
        )
