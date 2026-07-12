"""Bedrock rerank models: chat-class dispatch and search-unit usage recording."""

from typing import TYPE_CHECKING, cast

import openai
import pytest

from stdapi.models import ModelDetails, _compute_model_capabilities
from stdapi.models.chat import get_chat_model
from stdapi.models.chat.bedrock_rerank import ChatModel as RerankChatModel
from stdapi.pricing import Dimension
from stdapi.usage import USAGE, init_model_state, init_usage

if TYPE_CHECKING:
    from openai import OpenAI
    from types_aiobotocore_bedrock_runtime.type_defs import ConverseResponseTypeDef

RERANK_MODELS = ("amazon.rerank-v1:0", "cohere.rerank-v3-5:0")


class TestRerankChatModelDispatch:
    """Rerank model IDs must resolve to the search-unit-recording class."""

    @pytest.mark.parametrize("model_id", RERANK_MODELS)
    def test_matcher_dispatch(self, model_id: str) -> None:
        """The registry resolves rerank IDs to the rerank override class."""
        assert type(get_chat_model(model_id)) is RerankChatModel


class TestRerankSupportedRoutes:
    """Rerank models must not advertise Converse-served text routes."""

    @pytest.mark.parametrize("model_id", RERANK_MODELS)
    def test_no_converse_routes_advertised(self, model_id: str) -> None:
        """supported_routes excludes chat/completions/responses/messages."""
        details = ModelDetails(
            id=model_id,
            name=model_id,
            provider="Vendor",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
            regions=["us-east-1"],
        )
        routes, tools = _compute_model_capabilities(model_id, details)
        assert not any("chat/completions" in route for route in routes)
        assert not any("responses" in route for route in routes)
        assert not any("messages" in route for route in routes)
        assert "openai_chat_completion" not in tools


class TestRerankUsageRecording:
    """Each rerank Converse call records one billed search unit."""

    @pytest.mark.parametrize("model_id", RERANK_MODELS)
    def test_converse_usage_records_search_unit(self, model_id: str) -> None:
        """Token usage and one search unit are recorded per call."""
        usage_token = init_usage()
        init_model_state()
        try:
            RerankChatModel(model_id)._record_converse_usage(  # noqa: SLF001
                {"usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 12}}  # type: ignore[typeddict-item]
            )
            records = list(USAGE.get().values())
        finally:
            USAGE.reset(usage_token)
        assert len(records) == 1
        assert records[0].quantities[Dimension.SEARCH_UNITS] == 1
        assert records[0].quantities[Dimension.INPUT_TOKENS] == 10

    @pytest.mark.parametrize("model_id", RERANK_MODELS)
    def test_search_unit_recorded_without_token_usage(self, model_id: str) -> None:
        """A call whose response lacks a usage block still bills one search unit."""
        usage_token = init_usage()
        init_model_state()
        try:
            RerankChatModel(model_id)._record_converse_usage(  # noqa: SLF001
                cast("ConverseResponseTypeDef", {})
            )
            records = list(USAGE.get().values())
        finally:
            USAGE.reset(usage_token)
        assert len(records) == 1
        assert records[0].quantities == {Dimension.SEARCH_UNITS: 1}


class TestRerankChatCompletions:
    """End-to-end rerank via the chat completions route."""

    @pytest.mark.parametrize("model", RERANK_MODELS[:1])
    def test_rerank_via_chat_completions(
        self, openai_client: OpenAI, use_official_api: bool, model: str
    ) -> None:
        """Rerank via chat completions: succeeds, or fails with AWS's clean rejection.

        Confirmed live (2026-07): the Bedrock Converse action currently
        rejects rerank models ("This action doesn't support the model...").
        The dispatch and billing are in place for when AWS enables it; until
        then the app must surface AWS's validation error as a clean 400.
        """
        if use_official_api:
            pytest.skip("Rerank models are not supported on the official API")
        try:
            resp = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Rank: a, b, c for query 'b'."}],
            )
        except openai.BadRequestError as exc:
            error_message = str(exc)
        else:
            assert hasattr(resp, "choices")
            return
        assert "support the model" in error_message
