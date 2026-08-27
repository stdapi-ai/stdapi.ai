"""Ollama-compatible POST /api/embed and POST /api/embeddings, against Bedrock.

The two are different shapes rather than aliases: ``/api/embed`` takes one or
several ``input`` values and answers ``embeddings``, while the superseded
``/api/embeddings`` takes a single ``prompt`` and answers one ``embedding``.
Both are driven through the official ``ollama`` client, which has a call for
each of them.

Ollama Cloud serves neither, which is why this module is gateway-only rather
than untested: it hosts **no embedding model at all** -- all 18 models it
publishes advertise ``completion`` and none advertises ``embedding`` --
``/api/embed`` answers 401 to a cloud API key whichever model is named, and
``/api/embeddings`` is not a path it serves at all (404, ``path not found``).
The gateway is the more complete of the two here.

Ref: https://docs.ollama.com/api/embed
     stdapi/routes/ollama_embed.py:embed
     stdapi/routes/ollama_embed.py:embeddings
"""

from typing import TYPE_CHECKING

import ollama
import pytest

if TYPE_CHECKING:
    import httpx


pytestmark = pytest.mark.gateway(
    "Ollama Cloud hosts no embedding model at all -- all 18 models it publishes "
    "advertise 'completion' only -- /api/embed answers 401 to a cloud API key "
    "whichever model is named, and /api/embeddings is not a path it serves. "
    "The gateway marking is a fact about the vendor, not an untested route"
)

#: An Ollama-shaped embedding model name this server does not offer.
UNKNOWN_MODEL = "nomic-embed-text:latest"

#: Vector width asked for by the truncation test; Titan v2 serves 256.
_REDUCED_DIMENSIONS = 256


def test_embed_one_input(
    ollama_client: ollama.Client, ollama_embedding_model: str
) -> None:
    """A single string answers one vector, and the input token count.

    Ref: https://docs.ollama.com/api/embed
    """
    answer = ollama_client.embed(model=ollama_embedding_model, input="Hello world")
    assert answer.model == ollama_embedding_model
    assert len(answer.embeddings) == 1
    assert len(answer.embeddings[0]) > 1
    assert any(answer.embeddings[0])
    assert answer.prompt_eval_count
    assert answer.prompt_eval_count > 0
    assert answer.total_duration
    assert answer.total_duration > 0
    assert answer.load_duration is None


def test_embed_several_inputs_in_request_order(
    ollama_client: ollama.Client, ollama_embedding_model: str
) -> None:
    """A list answers one vector per input, in the order they were sent.

    Ref: https://docs.ollama.com/openapi.yaml (EmbedRequest.input)
    """
    answer = ollama_client.embed(
        model=ollama_embedding_model, input=["first text", "a different one"]
    )
    assert len(answer.embeddings) == 2
    assert answer.embeddings[0] != answer.embeddings[1]


def test_embed_honours_the_requested_dimensions(
    ollama_client: ollama.Client, ollama_embedding_model: str
) -> None:
    """`dimensions` selects the vector width where the model supports it.

    Ref: https://docs.ollama.com/openapi.yaml (EmbedRequest.dimensions)
    """
    answer = ollama_client.embed(
        model=ollama_embedding_model,
        input="Hello world",
        dimensions=_REDUCED_DIMENSIONS,
    )
    assert len(answer.embeddings[0]) == _REDUCED_DIMENSIONS


def test_embed_ignores_the_residency_and_runner_options(
    ollama_client: ollama.Client, ollama_embedding_model: str
) -> None:
    """`keep_alive` and `truncate` are accepted and ignored, not refused.

    The caller still gets the embeddings it asked for, which is the accept-and-
    ignore case rather than the reject one.

    Ref: stdapi/types/ollama.py:EmbedRequest
    """
    answer = ollama_client.embed(
        model=ollama_embedding_model,
        input="Hello world",
        keep_alive="5m",
        truncate=True,
    )
    assert answer.embeddings


def test_legacy_embeddings_answers_a_single_vector(
    ollama_client: ollama.Client, ollama_http: httpx.Client, ollama_embedding_model: str
) -> None:
    """The superseded endpoint keeps its own singular shape.

    The client parses it into ``EmbeddingsResponse``, whose only field is the
    single ``embedding``. That model ignores anything else the body carries, so
    the wire shape is checked over raw HTTP as well: the two endpoints have to
    stay different shapes rather than one growing into the other.

    Ref: https://github.com/ollama/ollama/blob/main/docs/api.md
    """
    answer = ollama_client.embeddings(
        model=ollama_embedding_model, prompt="Hello world"
    )
    assert isinstance(answer, ollama.EmbeddingsResponse)
    assert len(answer.embedding) > 1
    body = ollama_http.post(
        "/api/embeddings", json={"model": ollama_embedding_model, "prompt": "Hello"}
    ).json()
    assert set(body) == {"embedding"}


def test_embed_refuses_an_unknown_model(ollama_client: ollama.Client) -> None:
    """An unknown model answers 404 in the Ollama error envelope.

    Ref: https://docs.ollama.com/openapi.yaml (ErrorResponse)
    """
    with pytest.raises(ollama.ResponseError) as raised:
        ollama_client.embed(model=UNKNOWN_MODEL, input="hi")
    assert raised.value.status_code == 404
    assert UNKNOWN_MODEL in raised.value.error
