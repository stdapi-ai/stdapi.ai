"""A Haystack retrieval stack driven end to end against the gateway.

This is the lane's only client that reranks. Coding agents never call a rerank
route, and n8n provably cannot reach this one: its Cohere Reranker node builds
its client from an API key and a model alone, and the ``cohereApi`` credential's
base URL field is hidden and unread. Haystack's ``CohereRanker`` takes an
``api_base_url``, so ``/cohere/v2/rerank`` gets end-to-end coverage here or
nowhere.

The pipeline chains three routes the way a production retrieval stack does:

- ``/v1/embeddings`` twice, once for the corpus and once for the query, through a
  second client implementation -- Haystack batches the whole corpus into one
  request and the OpenAI SDK asks for ``encoding_format=base64``, neither of
  which n8n's embeddings node does;
- ``/cohere/v2/rerank``, over the documents the vector search returned;
- ``/v1/chat/completions``, answering from the reranked context alone.

What makes the rerank assertion mean something is that the two rankings disagree
by construction: the corpus decoys restate the question's own wording while the
planted document answers it in synonyms, so pure embedding retrieval leaves the
planted document below rank 0. A gateway answering the rerank call with a 200 the
client cannot use -- indices that do not address the request's ``documents``,
results out of order, the score in the wrong field -- leaves that ranking
untouched and fails here.

It is a documentation-regression test too: ``docs/use_cases.md`` tells retrieval
frameworks to point their OpenAI-compatible embedder at ``/v1`` and the Cohere SDK
at ``/cohere`` for reranking, which is exactly the wiring the pipeline uses.

Requires ``--agentic``, podman, and Bedrock credentials.

Ref: https://docs.haystack.deepset.ai/docs/cohereranker
     https://docs.cohere.com/reference/rerank
     stdapi/routes/cohere_rerank.py:rerank
     docs/use_cases.md
     tests/agentic/rag_pipeline.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ._runner import ModelConfig, assert_result, log_metrics, run_agent
from ._tools import HAYSTACK, HAYSTACK_PLANTED_NAME, AgenticTool, haystack_record

if TYPE_CHECKING:
    from pathlib import Path

    from ._server import AgenticServer

pytestmark = pytest.mark.agentic

TOOL: AgenticTool = HAYSTACK

#: Seconds allowed for one pipeline run: Haystack builds its component registry
#: before the first request, which costs several seconds by itself.
_TIMEOUT = 600

#: Embedding model both the corpus and the query are vectorised with.
_EMBED_MODEL = "amazon.titan-embed-text-v2:0"

#: Reranking model the Cohere route resolves.
_RERANK_MODEL = "cohere.rerank-v3-5:0"

#: Question the corpus is built around.
#:
#: Its wording is the decoys' wording -- "array", "lock" and "outage" all appear in
#: documents that do not answer it -- which is what makes embedding retrieval rank
#: the planted document below them.
_QUERY = "Which unit brought the array back into lock after the outage?"

#: Components the pipeline declares; a complete run reports output from all five.
_PIPELINE_COMPONENTS = 5

#: Requests one run makes, by route.
#:
#: Two embedding calls, not one: the corpus is indexed in a single batch and the
#: query is embedded separately, which is what makes the assertion that the two
#: vectors share a space worth anything.
_EXPECTED_CALLS = {
    "/v1/embeddings": 2,
    "/cohere/v2/rerank": 1,
    "/v1/chat/completions": 1,
}

#: The chat model under test, the only one the shared identity check can see.
#:
#: That check attributes positionally and ignores whatever precedes the first
#: request to the model it was given, so the model under test has to be the
#: pipeline's last call. The embedding and reranking models are pinned by the
#: explicit per-route assertions instead.
_MODEL_CONFIGS = [
    # Reference baseline: the only model here that is natively Anthropic.
    pytest.param(
        ModelConfig(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            timeout=_TIMEOUT,
            extra_env={"EMBED_MODEL": _EMBED_MODEL, "RERANK_MODEL": _RERANK_MODEL},
        ),
        id="claude-haiku-4-5",
    ),
    pytest.param(
        ModelConfig(
            model="amazon.nova-2-lite-v1:0",
            timeout=_TIMEOUT,
            extra_env={"EMBED_MODEL": _EMBED_MODEL, "RERANK_MODEL": _RERANK_MODEL},
        ),
        id="nova-2-lite",
    ),
]


def _ranking(record: dict[str, object], key: str) -> list[dict[str, object]]:
    """Return one of the two rankings the pipeline reported.

    Args:
        record: Decoded pipeline record.
        key: ``retrieved`` for the embedding order, ``reranked`` for the rerank
            order.

    Returns:
        The documents in that order.
    """
    ranked = record[key]
    assert isinstance(ranked, list), f"{key} is not a ranking: {ranked!r}"
    return [document for document in ranked if isinstance(document, dict)]


def _requests_by_route(
    server: AgenticServer, start: int
) -> dict[str, list[dict[str, object]]]:
    """Return the requests logged since *start*, grouped by route.

    Args:
        server: Gateway the pipeline was pointed at.
        start: Log index captured before the run.

    Returns:
        The logged request entries keyed by path, restricted to the routes under
        test. Empty when the gateway's log is not observable, which is the case
        for the external deployment ``--server-url`` selects, so a caller may
        treat an empty mapping as "skip the route assertions".
    """
    if server.process is None:
        return {}
    grouped: dict[str, list[dict[str, object]]] = {path: [] for path in _EXPECTED_CALLS}
    for entry in server.log_entries(start):
        if entry.get("type") != "request":
            continue
        path = str(entry.get("path") or "")
        if path in grouped:
            grouped[path].append(entry)
    return grouped


@pytest.mark.parametrize("model_config", _MODEL_CONFIGS)
class TestHaystackRag:
    """Embedding retrieval, Cohere reranking and generation over one corpus.

    Ref: https://docs.haystack.deepset.ai/docs/cohereranker
         stdapi/routes/cohere_rerank.py:rerank
         stdapi/routes/openai_embeddings.py:create_embedding
    """

    def test_rerank_promotes_the_planted_document(
        self,
        request: pytest.FixtureRequest,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """The reranker moves the answering document from below rank 0 to rank 0.

        Two halves of one statement: embedding retrieval ranked a decoy first,
        and after ``/cohere/v2/rerank`` the document that actually answers the
        question is first. A rerank reply the client could not consume leaves the
        embedding order in place, so a broken translation of Bedrock's response
        surfaces here rather than in a status code.

        The rest of the record is asserted in the same test because one run costs
        three billed calls and every assertion reads a different part of it.

        Ref: stdapi/types/cohere_rerank.py:RerankResult
             stdapi/models/rerank/bedrock_rerank.py:BedrockRerankModel
        """
        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=TOOL,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_QUERY,
            workdir=agentic_workdir,
            test_name=request.node.originalname or request.node.name,
        )
        log_metrics(
            TOOL, result, model_config, "test_rerank_promotes_the_planted_document"
        )
        assert_result(
            result,
            config=model_config,
            contains=HAYSTACK_PLANTED_NAME,
            min_steps=_PIPELINE_COMPONENTS,
        )

        record = haystack_record(agentic_workdir)
        planted = record["planted_id"]
        retrieved = _ranking(record, "retrieved")
        reranked = _ranking(record, "reranked")

        assert planted in {document["id"] for document in retrieved}, (
            f"the planted document was never retrieved: {retrieved}"
        )
        assert retrieved[0]["id"] != planted, (
            "embedding retrieval already ranked the planted document first, so "
            f"the rerank had nothing to move: {retrieved}"
        )
        assert reranked[0]["id"] == planted, (
            f"the reranker did not promote the planted document: {reranked}"
        )
        scores = [float(document["score"]) for document in reranked]  # type: ignore[arg-type]
        assert scores == sorted(scores, reverse=True), f"unranked results: {scores}"
        assert scores[0] > scores[-1], "every document scored identically"

        self._assert_shared_embedding_space(record)
        self._assert_routes(agentic_server, log_start, model_config)

    @staticmethod
    def _assert_shared_embedding_space(record: dict[str, object]) -> None:
        """Assert both embedding calls returned vectors of the same dimension.

        The corpus is embedded in one batched request and the query in another,
        so a gateway resolving a different vector length between the two would
        leave the retriever ranking on noise.

        Args:
            record: Decoded pipeline record.

        Ref: stdapi/types/openai_embeddings.py:CreateEmbeddingResponse
        """
        document_dimension = record["document_embedding_dim"]
        assert isinstance(document_dimension, int)
        assert document_dimension > 0, "the gateway returned empty document vectors"
        assert record["query_embedding_dim"] == document_dimension

    @staticmethod
    def _assert_routes(
        server: AgenticServer, log_start: int, model_config: ModelConfig
    ) -> None:
        """Assert each stage reached its own route once, on its own model.

        The shared identity check only sees the chat model, because it attributes
        positionally and the chat call is last. This is what pins the other two:
        an embedding or rerank request that silently resolved a different model
        would leave the rankings intact and pass every other assertion.

        Args:
            server: Gateway the pipeline was pointed at.
            log_start: Log index captured before the run.
            model_config: Model under test.

        Ref: stdapi/monitoring.py:EventLog
        """
        requests = _requests_by_route(server, log_start)
        if not requests:
            return  # External server: its log is not observable here.

        counts = {path: len(entries) for path, entries in requests.items()}
        assert counts == _EXPECTED_CALLS, f"unexpected gateway traffic: {counts}"

        for path, model in (
            ("/v1/embeddings", _EMBED_MODEL),
            ("/cohere/v2/rerank", _RERANK_MODEL),
            ("/v1/chat/completions", model_config.model),
        ):
            resolved = {str(entry.get("model_id") or "") for entry in requests[path]}
            assert resolved == {model}, f"{path} resolved {resolved}, expected {model}"
