# mypy: disable-error-code="import-not-found"
"""Retrieval-augmented generation pipeline driven against the gateway.

Executed by ``python /work/rag_pipeline.py`` inside the ``rag`` image group's
container and never on the host: Haystack and the Cohere client live in that
image alone. Everything varying per run arrives through the environment, so the
file is copied in verbatim rather than rendered.

Three gateway routes are chained the way a real retrieval stack chains them:
``/v1/embeddings`` indexes the corpus and embeds the query, ``/cohere/v2/rerank``
reorders what the vector search returned, and ``/v1/chat/completions`` answers
from the reordered context. The corpus is written so the two rankings disagree --
the decoys restate the query's own wording while the planted document answers it
in different words -- so a rerank response that was requested but not applied
leaves the planted document below rank 0.

Prints exactly one JSON object, and writes the same object next to itself:
``tests/agentic/_tools.py:_haystack_parse`` normalises the printed copy into the
lane's shared result, while the assertions need the whole record and read the
file back through ``tests/agentic/_tools.py:haystack_record``.

Ref: https://docs.haystack.deepset.ai/docs/openaidocumentembedder
     https://docs.haystack.deepset.ai/docs/cohereranker
     tests/agentic/test_rag_haystack.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from haystack import Document, Pipeline
from haystack.components.builders import ChatPromptBuilder
from haystack.components.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.utils import Secret
from haystack_integrations.components.rankers.cohere import CohereRanker

#: Gateway under test, without a route prefix.
BASE_URL = os.environ["STDAPI_BASE_URL"]

#: Question asked of the corpus, chosen by the test.
QUERY = os.environ["QUERY"]

#: Name only the planted document carries, injected so the test owns the value.
PLANTED_NAME = os.environ["PLANTED_NAME"]

#: Identifier of the planted document, reported so the test needs no corpus copy.
PLANTED_ID = "planted"

#: Documents handed to the reranker, and to the generator as context.
TOP_K = 3

#: File the record is written to, in the working directory the test can read.
RECORD_PATH = Path(__file__).resolve().parent / "rag_run.json"

#: Corpus: one document answering :data:`QUERY`, six restating it without doing so.
#:
#: The decoys repeat the question's own wording -- "the array came back into lock
#: after the outage" is nearly the question itself -- which is what a bi-encoder
#: scores on, yet none of them names what restored the array. The planted document
#: is the only one that does, and it says so in a different register ("recovered
#: lock ... once mains power returned"), so it lands mid-table on vector similarity
#: and first once a cross-encoder reads the question.
DOCUMENTS = [
    Document(
        id=PLANTED_ID,
        content=(
            f"Field note 88: at Fennmoor the {PLANTED_NAME} was the unit that "
            f"recovered lock on the antenna assembly once mains power returned, "
            f"in nine seconds."
        ),
    ),
    Document(
        content=(
            "Site checklist AR-12 lists what the crew must verify before the "
            "array is declared back in lock after an outage."
        )
    ),
    Document(
        content=(
            "The outage began at 02:14 UTC and lasted eleven minutes; the array "
            "stayed out of lock for the whole window."
        )
    ),
    Document(
        content=(
            "The weekly summary sheet recorded the outage and the array's loss "
            "of lock as a single event."
        )
    ),
    Document(
        content=(
            "The array came back into lock after the outage; lock acquisition is "
            "specified to complete in under thirty seconds."
        )
    ),
    Document(
        content=(
            "The Calderwood damping unit was serviced in March; its vibration "
            "logs are archived with the array mast records."
        )
    ),
    Document(
        content=(
            "Unit designations at the Fennmoor site are printed on the cabinet "
            "doors; the outage affected cabinets 3 and 4 of the array hall."
        )
    ),
]

#: Prompt the generator answers from, rendered over the reranked context.
TEMPLATE = """\
Answer the question using only the context below, in one sentence. Spell any
name exactly as the context spells it.

Context:
{% for document in documents %}
- {{ document.content }}
{% endfor %}

Question: {{ question }}
Answer:"""


def index_corpus() -> tuple[InMemoryDocumentStore, int]:
    """Embed the corpus through the gateway and load it into a document store.

    Returns:
        The populated store, and the dimension of the document vectors.

    Raises:
        RuntimeError: If the gateway returned no embedding for a document.
    """
    embedder = OpenAIDocumentEmbedder(
        api_key=Secret.from_env_var("OPENAI_API_KEY"),
        model=os.environ["EMBED_MODEL"],
        api_base_url=f"{BASE_URL}/v1",
        # One request for the whole corpus, so the log carries exactly one
        # indexing call and the test can count what the pipeline produced.
        batch_size=len(DOCUMENTS),
        raise_on_failure=True,
    )
    embedded = embedder.run(documents=list(DOCUMENTS))["documents"]
    if not embedded or embedded[0].embedding is None:
        msg = "the gateway returned no document embeddings"
        raise RuntimeError(msg)
    store = InMemoryDocumentStore()
    store.write_documents(embedded)
    return store, len(embedded[0].embedding)


def build_pipeline(store: InMemoryDocumentStore) -> Pipeline:
    """Assemble the query pipeline: embed, retrieve, rerank, prompt, generate.

    Args:
        store: Document store holding the indexed corpus.

    Returns:
        The connected pipeline.
    """
    pipeline = Pipeline()
    pipeline.add_component(
        "text_embedder",
        OpenAITextEmbedder(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model=os.environ["EMBED_MODEL"],
            api_base_url=f"{BASE_URL}/v1",
        ),
    )
    # Every document reaches the reranker, so the two rankings are comparable
    # over the same set and a move to rank 0 is the reranker's own doing.
    pipeline.add_component(
        "retriever", InMemoryEmbeddingRetriever(store, top_k=len(DOCUMENTS))
    )
    pipeline.add_component(
        "ranker",
        CohereRanker(
            api_key=Secret.from_env_var("COHERE_API_KEY"),
            model=os.environ["RERANK_MODEL"],
            # The Cohere client appends "v2/rerank" to its base URL, which is how
            # the gateway's Cohere-compatible route is reached.
            api_base_url=f"{BASE_URL}/cohere",
            top_k=TOP_K,
        ),
    )
    pipeline.add_component(
        "prompt_builder", ChatPromptBuilder(template=[ChatMessage.from_user(TEMPLATE)])
    )
    pipeline.add_component(
        "llm",
        OpenAIChatGenerator(
            api_key=Secret.from_env_var("OPENAI_API_KEY"),
            model=os.environ["CHAT_MODEL"],
            api_base_url=f"{BASE_URL}/v1",
        ),
    )
    pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "ranker.documents")
    pipeline.connect("ranker.documents", "prompt_builder.documents")
    pipeline.connect("prompt_builder.prompt", "llm.messages")
    return pipeline


def ranking(outputs: dict[str, Any], component: str) -> list[dict[str, Any]]:
    """Return one ``{id, score, content}`` entry per document *component* emitted.

    Args:
        outputs: Result of the pipeline run.
        component: Name of the component whose documents are read.

    Returns:
        The documents in the order the component ranked them.
    """
    return [
        {"id": document.id, "score": document.score, "content": document.content}
        for document in outputs.get(component, {}).get("documents", [])
    ]


def main() -> None:
    """Run the pipeline and print the single JSON record the test parses."""
    store, document_dimension = index_corpus()
    pipeline = build_pipeline(store)
    outputs = pipeline.run(
        {
            "text_embedder": {"text": QUERY},
            "ranker": {"query": QUERY},
            "prompt_builder": {"question": QUERY},
        },
        include_outputs_from={"text_embedder", "retriever", "ranker", "prompt_builder"},
    )
    replies = outputs["llm"]["replies"]
    usage = replies[0].meta.get("usage") or {} if replies else {}
    record = json.dumps(
        {
            "components": sorted(outputs),
            "planted_id": PLANTED_ID,
            "document_embedding_dim": document_dimension,
            "query_embedding_dim": len(outputs["text_embedder"]["embedding"]),
            "retrieved": ranking(outputs, "retriever"),
            "reranked": ranking(outputs, "ranker"),
            "answer": replies[0].text if replies else "",
            "usage": dict(usage),
        }
    )
    RECORD_PATH.write_text(record, encoding="utf-8")
    print(record)  # noqa: T201


if __name__ == "__main__":
    main()
