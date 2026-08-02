"""Amazon Titan embedding backend: Bedrock request body and response aggregation.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
     stdapi/models/embedding/amazon_titan_embed.py:EmbeddingModel
"""

from typing import Any

import pytest

from stdapi.models import InvokeResult
from stdapi.models.embedding.amazon_titan_embed import EmbeddingModel


@pytest.mark.local
class TestTitanEmbeddingModelQuantizedTypes:
    """EmbeddingModel.embed_text: aggregation of `embeddingsByType` across invokes.

    Titan Embed accepts a single `inputText` per InvokeModel call, so a batched
    request fans out into one call per input and the per-call vectors must be
    stitched back into one list per embedding type.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
         stdapi/models/embedding/amazon_titan_embed.py:EmbeddingModel.embed_text
    """

    async def test_embeddings_by_type_is_aggregated_per_input(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each per-input `embeddingsByType` vector is aggregated into a combined list.

        Titan V2 returns `embeddingsByType` with at least `float` when
        `embeddingTypes` is sent, and `inputTextTokenCount` per call, which the
        backend sums into `prompt_tokens`.
        """
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
        bodies: list[dict[str, Any]] = []

        async def _invoke(body: dict[str, Any]) -> InvokeResult[Any]:
            bodies.append(body)
            return InvokeResult(
                response=responses[body["inputText"]],
                input_tokens=None,
                output_tokens=0,
            )

        monkeypatch.setattr(type(model), "invoke", staticmethod(_invoke))

        response = await model.embed_text(
            ["hello", "world"],
            dimensions=None,
            extra_params={"embeddingTypes": ["float", "binary"]},
        )

        assert len(bodies) == 2, "Titan embeds one input text per InvokeModel call"
        assert {body["inputText"] for body in bodies} == {"hello", "world"}
        assert all(body["embeddingTypes"] == ["float", "binary"] for body in bodies)
        assert all("dimensions" not in body for body in bodies), (
            "no `dimensions` must be sent when the caller did not ask for one"
        )
        assert response.embeddings == [[0.1, 0.2], [0.3, 0.4]]
        assert response.embeddings_by_type == {
            "float": [[0.1, 0.2], [0.3, 0.4]],
            "binary": [[1, 0], [0, 1]],
        }
        assert response.prompt_tokens == 5
        assert response.total_tokens == 5

    async def test_embeddings_by_type_is_none_when_not_requested(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without `embeddingTypes` in the request, `embeddings_by_type` stays unset."""
        model = EmbeddingModel("amazon.titan-embed-text-v2:0")
        bodies: list[dict[str, Any]] = []

        async def _invoke(body: dict[str, Any]) -> InvokeResult[Any]:
            bodies.append(body)
            return InvokeResult(
                response={"embedding": [0.1, 0.2], "inputTextTokenCount": 2},
                input_tokens=None,
                output_tokens=0,
            )

        monkeypatch.setattr(type(model), "invoke", staticmethod(_invoke))

        response = await model.embed_text(["hello"], dimensions=None, extra_params={})

        assert bodies == [{"inputText": "hello"}]
        assert response.embeddings == [[0.1, 0.2]]
        assert response.embeddings_by_type is None
        assert response.prompt_tokens == 2
        assert response.total_tokens == 2
