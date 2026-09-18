"""Ollama-compatible model management: /api/pull and the five refused verbs.

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

from hashlib import sha256
from typing import TYPE_CHECKING

import ollama
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from stdapi.api_errors import NoModelStoreError
from stdapi.config import SETTINGS
from stdapi.routes import ollama_model_management
from tests._helpers import ollama_route
from tests.test_ollama_chat import NDJSON, ndjson_lines

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import httpx


#: An Ollama-shaped name neither target can make available.
UNKNOWN_MODEL = "llama3.2:3b"

#: Digest of the empty blob, spelled as the official client spells one.
BLOB_DIGEST = f"sha256:{sha256(b'').hexdigest()}"

#: Every route of this dialect, with the method a client reaches it by.
OLLAMA_ROUTES: list[tuple[str, str]] = [
    ("post", ollama_route("/api/chat")),
    ("post", ollama_route("/api/generate")),
    ("post", ollama_route("/api/embed")),
    ("post", ollama_route("/api/embeddings")),
    ("get", ollama_route("/api/tags")),
    ("post", ollama_route("/api/show")),
    ("get", ollama_route("/api/ps")),
    ("get", ollama_route("/api/version")),
    ("post", ollama_route("/api/pull")),
    ("post", ollama_route("/api/create")),
    ("post", ollama_route("/api/copy")),
    ("post", ollama_route("/api/push")),
    ("delete", ollama_route("/api/delete")),
    ("post", ollama_route(f"/api/blobs/{BLOB_DIGEST}")),
    # The probes are out of the schema, so nothing else lists them.
    ("head", ollama_route("/api/tags")),
    ("head", ollama_route("/api/version")),
    ("head", ollama_route(f"/api/blobs/{BLOB_DIGEST}")),
    ("get", ollama_route("")),
    ("get", ollama_route("/")),
    ("head", ollama_route("")),
    ("head", ollama_route("/")),
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
    """Each answers 403 and says what is unavailable and where to look instead.

    A 403 rather than a 400: the request is well-formed, the server simply
    will not perform it. Driven through the client's own four calls, so the
    refusal reaches a caller as a ``ResponseError`` rather than as an
    unparseable body.

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
    assert raised.value.status_code == 403
    assert "does not store models" in raised.value.error
    assert "model list endpoint" in raised.value.error


@pytest.mark.local
def test_a_blob_upload_is_refused_rather_than_404(app_client: TestClient) -> None:
    """A blob push answers the refusal the store-mutating verbs give.

    A blob is uploaded only as the first step of building a model from local
    files, which `/api/create` already refuses, so the upload has nowhere to
    go. Answering the router's bare 404 instead would tell a client that
    streamed a multi-gigabyte file that it had the address wrong. Nothing is
    invoked, so the in-process app answers this without a backend.

    Ref: https://raw.githubusercontent.com/ollama/ollama/main/docs/api.md (Push a Blob)
         stdapi/routes/ollama_model_management.py:push_blob
    """
    response = app_client.post(
        ollama_route(f"/api/blobs/{BLOB_DIGEST}"), content=b"data"
    )

    assert response.status_code == 403
    assert "does not store models" in response.json()["error"]
    assert "model list endpoint" in response.json()["error"]


@pytest.mark.local
def test_a_blob_is_never_present(app_client: TestClient) -> None:
    """`HEAD` answers 404: no blob is ever stored here, which is Ollama's own answer.

    It is mounted only so that the upload verb on the same path does not turn
    an absent blob into a `405`.

    Ref: https://raw.githubusercontent.com/ollama/ollama/main/docs/api.md (Check if a Blob Exists)
         stdapi/routes/ollama_model_management.py:head_blob
    """
    response = app_client.head(ollama_route(f"/api/blobs/{BLOB_DIGEST}"))

    assert response.status_code == 404


def test_the_blob_upload_declares_no_request_body() -> None:
    """The upload takes no declared body, which is what keeps it out of memory.

    An ``Annotated[bytes, Body()]`` parameter is the obvious way to type an
    octet-stream payload for the schema, and it buffers the whole GGUF before
    the refusal is reached -- on a deployment sized in gigabytes, that is the
    request that ends it.

    Ref: stdapi/routes/ollama_model_management.py:push_blob
    """
    app = FastAPI()
    app.include_router(ollama_model_management.router)

    operation = app.openapi()["paths"][ollama_route("/api/blobs/{digest}")]["post"]

    assert "requestBody" not in operation


def test_a_streamed_blob_upload_is_read_out_before_it_is_refused(
    monkeypatch: pytest.MonkeyPatch, api_key: str
) -> None:
    """The body is drained in chunks, so the refusal is an answer, not a broken pipe.

    Refusing before the body is consumed makes the ASGI server close the
    connection mid-upload, and the client sees a transport error instead of the
    `403`. Driven against a bare app holding this router alone: the refusal is
    raised rather than formatted, which is what makes the drain assertable. That
    app carries no monitoring middleware, so request-param logging is off for it.

    Args:
        monkeypatch: Silences the request-param logging the bare app cannot do.
        api_key: Credential the authenticating route is reached with.

    Ref: stdapi/routes/ollama_model_management.py:_discard_body
    """
    monkeypatch.setattr(SETTINGS, "log_request_params", False)
    chunk = b"\0" * (1024 * 1024)
    sent = 0

    def body() -> Iterator[bytes]:
        """Stream the upload a chunk at a time, counting what the server took.

        Yields:
            One chunk of the blob.
        """
        nonlocal sent
        for _ in range(8):
            sent += len(chunk)
            yield chunk

    app = FastAPI()
    app.include_router(ollama_model_management.router)

    # Credentials are sent because the route authenticates first: whether the
    # handler is armed depends on what else ran, and an unauthenticated 401
    # would answer before the body was ever read.
    with (
        TestClient(app, headers={"Authorization": f"Bearer {api_key}"}) as client,
        pytest.raises(NoModelStoreError),
    ):
        client.post(ollama_route(f"/api/blobs/{BLOB_DIGEST}"), content=body())

    assert sent == 8 * len(chunk)


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
