"""Ollama-compatible model management: /api/pull and the four refused verbs.

``pull`` is the only one of the five whose post-condition this server can meet:
its contract is "after this returns, the model can be used", which is already
true of every model the catalogue lists. The other four mutate a model store
that does not exist here, and answering them 200 would tell a caller that state
changed when nothing did.

Ollama Cloud refuses all five to a cloud API key with 401 -- it resolves the
model name first, so an unknown one still answers 404 -- which is why only the
unknown-model case runs on both targets.

Ref: https://docs.ollama.com/api/pull
     stdapi/routes/ollama_model_management.py
"""

from typing import TYPE_CHECKING

import ollama
import pytest

from tests.test_ollama_chat import NDJSON, ndjson_lines

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx
    from starlette.testclient import TestClient


#: An Ollama-shaped name neither target can make available.
UNKNOWN_MODEL = "llama3.2:3b"

#: Every route of this dialect, with the method a client reaches it by.
OLLAMA_ROUTES: list[tuple[str, str]] = [
    ("post", "/api/chat"),
    ("post", "/api/generate"),
    ("post", "/api/embed"),
    ("post", "/api/embeddings"),
    ("get", "/api/tags"),
    ("post", "/api/show"),
    ("get", "/api/ps"),
    ("get", "/api/version"),
    ("post", "/api/pull"),
    ("post", "/api/create"),
    ("post", "/api/copy"),
    ("post", "/api/push"),
    ("delete", "/api/delete"),
    # The two liveness probes are out of the schema, so nothing else lists them.
    ("head", "/api/tags"),
    ("head", "/api/version"),
]


@pytest.mark.gateway("Ollama Cloud refuses /api/pull to a cloud API key with 401")
def test_pull_reports_success_for_an_available_model(
    ollama_client: ollama.Client, ollama_chat_model: str
) -> None:
    """A catalogued model is already usable, so the pull succeeds immediately.

    Several clients call this before their first chat and abort on a non-200.

    Ref: https://docs.ollama.com/api/pull
    """
    progress = ollama_client.pull(ollama_chat_model, stream=False)
    assert isinstance(progress, ollama.ProgressResponse)
    assert progress.status == "success"


@pytest.mark.gateway("Ollama Cloud refuses /api/pull to a cloud API key with 401")
def test_pull_streams_its_status(
    ollama_client: ollama.Client, ollama_http: httpx.Client, ollama_chat_model: str
) -> None:
    """The streamed form ends on the success status, with no invented progress.

    The client yields one ``ProgressResponse`` per line; the media type it never
    inspects is checked over raw HTTP.

    Ref: https://docs.ollama.com/openapi.yaml (StatusEvent)
    """
    events = list(ollama_client.pull(ollama_chat_model, stream=True))
    assert [event.status for event in events] == ["success"]
    assert events[-1].completed is None
    assert events[-1].total is None
    response = ollama_http.post(
        "/api/pull", json={"model": ollama_chat_model, "stream": True}
    )
    assert response.headers["content-type"].startswith(NDJSON)
    assert ndjson_lines(response) == [{"status": "success"}]


def test_pull_refuses_a_model_this_server_does_not_offer(
    ollama_client: ollama.Client,
) -> None:
    """A model outside the catalogue cannot be made available, so it answers 404.

    Both targets resolve the name before anything else, so this is the one pull
    case Ollama Cloud answers the same way.

    Ref: stdapi/routes/ollama_model_management.py:pull
    """
    with pytest.raises(ollama.ResponseError) as raised:
        ollama_client.pull(UNKNOWN_MODEL, stream=False)
    assert raised.value.status_code == 404
    assert UNKNOWN_MODEL in raised.value.error


@pytest.mark.gateway("Ollama Cloud refuses these to a cloud API key with 401")
@pytest.mark.parametrize("verb", ["create", "copy", "push", "delete"])
def test_the_store_mutating_verbs_are_refused(
    ollama_client: ollama.Client, verb: str
) -> None:
    """Each answers 400 and says what is unavailable and where to look instead.

    Driven through the client's own four calls, so the refusal reaches a caller
    as a ``ResponseError`` rather than as an unparseable body.

    Ref: stdapi/routes/ollama_model_management.py:_refuse
    """
    calls: dict[str, Callable[[], object]] = {
        "create": lambda: ollama_client.create(model="mine", from_="base"),
        "copy": lambda: ollama_client.copy("a", "b"),
        "push": lambda: ollama_client.push("mine", stream=False),
        "delete": lambda: ollama_client.delete("mine"),
    }
    with pytest.raises(ollama.ResponseError) as raised:
        calls[verb]()
    assert raised.value.status_code == 400
    assert "does not store models" in raised.value.error
    assert "model list endpoint" in raised.value.error


@pytest.mark.local
@pytest.mark.parametrize(("method", "path"), OLLAMA_ROUTES)
def test_every_ollama_route_requires_authentication(
    enforced_auth_client: TestClient, method: str, path: str
) -> None:
    """No endpoint of this dialect is reachable without credentials.

    A refusal is still an authenticated endpoint: the four that only ever answer
    an error are the ones most easily written without the dependency, and the
    two ``HEAD`` liveness probes are the ones most easily opened on purpose,
    since a local Ollama answers them to a client holding no credential. Driven
    over raw HTTP because the point is the absent credential, which the client
    always supplies once it has one.

    Ref: stdapi/routes/ollama_model_management.py
    """
    response = enforced_auth_client.request(method.upper(), path, json={})
    assert response.status_code == 401
