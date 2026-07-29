"""Tests for the Amazon Titan embedding model's Bedrock response parsing."""

from typing import Any

import pytest

from stdapi.models import InvokeResult
from stdapi.models.embedding.amazon_titan_embed import EmbeddingModel


@pytest.mark.local
class TestTitanEmbeddingModelQuantizedTypes:
    """EmbeddingModel.embed_text: aggregation of `embeddingsByType` across invokes."""

    async def test_embeddings_by_type_is_aggregated_per_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each per-input `embeddingsByType` vector is aggregated into a combined list."""
        model = EmbeddingModel("amazon.titan-embed-text-v2:0")
        responses = {
            "hello": {
                "embedding": [0.1, 0.2],
                "embeddingsByType": {"float": [0.1, 0.2], "binary": [1, 0]},
                "inputTextTokenCount": 2,
            },
            "world": {
                "embedding": [0.3, 0.4],
                "embeddingsByType": {"float": [0.3, 0.4], "binary": [0, 1]},
                "inputTextTokenCount": 3,
            },
        }

        async def _invoke(body: dict[str, Any]) -> InvokeResult[Any]:
            return InvokeResult(
                response=responses[body["inputText"]],
                input_tokens=None,
                output_tokens=0,
            )

        monkeypatch.setattr(model, "invoke", _invoke)

        response = await model.embed_text(
            ["hello", "world"],
            dimensions=None,
            extra_params={"embeddingTypes": ["float", "binary"]},
        )

        assert response.embeddings == [[0.1, 0.2], [0.3, 0.4]]
        assert response.embeddings_by_type == {
            "float": [[0.1, 0.2], [0.3, 0.4]],
            "binary": [[1, 0], [0, 1]],
        }
        assert response.prompt_tokens == 5

    async def test_embeddings_by_type_is_none_when_not_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without `embeddingTypes` in the request, `embeddings_by_type` stays unset."""
        model = EmbeddingModel("amazon.titan-embed-text-v2:0")

        async def _invoke(_body: dict[str, Any]) -> InvokeResult[Any]:
            return InvokeResult(
                response={"embedding": [0.1, 0.2], "inputTextTokenCount": 2},
                input_tokens=None,
                output_tokens=0,
            )

        monkeypatch.setattr(model, "invoke", _invoke)

        response = await model.embed_text(["hello"], dimensions=None, extra_params={})

        assert response.embeddings == [[0.1, 0.2]]
        assert response.embeddings_by_type is None
