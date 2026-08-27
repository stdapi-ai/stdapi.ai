"""Open WebUI wired to the gateway exactly as ``docs/use_cases_openwebui.md`` says.

This is the lane's first client shaped as a server rather than as a command: it is
started once per module by :func:`~tests.agentic._podman.start_service_container`
and driven over its own HTTP API, with no browser anywhere.

Its value is not a new gateway route -- n8n and Haystack already reach every one of
them -- but that the environment block below is the documentation's, verbatim. A
setting the docs promise and the gateway no longer honours fails here, at the layer
users actually configure. All six of the doc's sections are exercised in one boot:

- Core Connection -> ``/v1/chat/completions``;
- Ollama Connection -> the ``/api/*`` dialect, driven through Open WebUI's own
  Ollama proxy and admin surface;
- Text to Speech and Speech to Text -> ``/v1/audio/speech`` fed straight back into
  ``/v1/audio/transcriptions``, so the round trip needs no sample file;
- Image Generation -> ``/v1/images/generations``, decoded to a raster;
- RAG Embeddings **and** RAG Reranking -> ``/v1/embeddings`` and the Cohere-dialect
  ``/cohere/v2/rerank`` the doc prescribes, chained by one document upload.

Those settings are ``PersistentConfig`` values: Open WebUI reads them out of the
environment on the **first** boot only and stores them in its SQLite database, so
the data directory is created fresh per module and never reused.

Requires ``--agentic``, podman, and Bedrock credentials.

Ref: https://docs.openwebui.com/getting-started/env-configuration/
     docs/use_cases_openwebui.md
     stdapi/routes/cohere_rerank.py:rerank
     stdapi/routes/openai_chat_completions.py:create_chat_completion
     stdapi/routes/ollama_chat.py:chat
     tests/agentic/_podman.py:start_service_container
"""

from __future__ import annotations

import re
from json import dumps
from secrets import token_hex
from time import monotonic, sleep
from typing import TYPE_CHECKING

import httpx
import pytest

from tests.conftest import smallest_image_size

from ._podman import start_service_container, stop_service_container
from ._server import find_free_port

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from ._server import AgenticServer

pytestmark = pytest.mark.agentic

#: Image driven here. The slim variant is the right one for this configuration:
#: every model is remote, so the torch and Sentence-Transformers layers the full
#: image carries would only be downloaded and never loaded.
_IMAGE = "ghcr.io/open-webui/open-webui:main-slim"

#: Command started in the container.
#:
#: The image's own ``CMD`` is ``bash start.sh``, relative to a working directory
#: the sandbox replaces with ``/work``, so the entry point is named absolutely.
_ARGV = ("bash", "/app/backend/start.sh")

#: Seconds allowed for the service to answer ``/health`` after launch.
_STARTUP_TIMEOUT = 300

#: Seconds one request to Open WebUI may take; a Bedrock call runs behind each.
_REQUEST_TIMEOUT = 600.0

#: Chat model, the doc's ``TASK_MODEL_EXTERNAL`` and default chat backend.
_CHAT_MODEL = "amazon.nova-2-lite-v1:0"

#: Second chat model: Claude replays its own reasoning under a signature.
_CLAUDE_CHAT_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"

#: Namespace the Ollama connection's models are listed under.
#:
#: Both connection types are enabled at once here, and they serve the same
#: catalogue, so without a prefix each model id would arrive twice and the merged
#: list would keep whichever the merge saw last -- silently rerouting the Core
#: Connection tests above through the Ollama one. ``prefix_id`` is Open WebUI's
#: own answer to that, and ``docs/use_cases_openwebui.md`` prescribes it for
#: exactly this configuration.
_OLLAMA_PREFIX = "ollama"

#: The chat model as the Ollama connection lists it, prefix included.
_OLLAMA_CHAT_MODEL = f"{_OLLAMA_PREFIX}.{_CHAT_MODEL}"

#: Shape of the API version Open WebUI refuses a connection without.
_VERSION = re.compile(r"^\d+\.\d+\.\d+")

#: Embedding model the corpus and every query are vectorised with.
_EMBEDDING_MODEL = "cohere.embed-v4:0"

#: Reranking model reached through the Cohere-compatible route.
_RERANK_MODEL = "cohere.rerank-v3-5:0"

#: Image generation model.
_IMAGE_MODEL = "stability.stable-image-core-v1:1"

#: Speech-to-text model.
_STT_MODEL = "amazon.transcribe"

#: Text-to-speech model.
_TTS_MODEL = "amazon.polly-neural"

#: Account ``WEBUI_AUTH=False`` mints on the first sign-in; the value is hardcoded
#: by Open WebUI itself and only ever reachable on the container's own loopback.
_ADMIN_EMAIL = "admin@localhost"

#: Password of that account, likewise fixed by Open WebUI.
_ADMIN_PASSWORD = "admin"  # noqa: S105

#: Characters per retrieval chunk, small enough that each corpus paragraph below
#: becomes one chunk; the default of 1000 would fold the whole file into one and
#: leave the reranker a single document to order.
_CHUNK_SIZE = "220"

#: Overlap between chunks, off so no chunk carries a neighbour's wording.
_CHUNK_OVERLAP = "0"

#: Sentence synthesised and transcribed back, chosen for phonetic breadth.
_SPOKEN_SENTENCE = "The quick brown fox jumps over the lazy dog."

#: Words the transcript of that sentence must carry.
_SPOKEN_WORDS = ("quick", "brown", "fox", "lazy", "dog")

#: Word the chat model is asked to answer with, absent from the request otherwise.
_CHAT_KEYWORD = "MARMALADE"

#: Identifier only the answering paragraph of the corpus carries.
_PLANTED_NAME = "Quillon-7"

#: Question asked of the uploaded document.
#:
#: Its wording -- "array", "lock", "outage" -- is the decoys' wording, and the
#: answering paragraph avoids it, which is what makes vector retrieval rank that
#: paragraph last and a consumed rerank reply move it to the front.
_RAG_QUERY = "Which unit recovered lock on the array after the outage?"

#: Document uploaded, one paragraph per retrieval chunk.
_RAG_DOCUMENT = f"""\
Site checklist AR-12 lists everything the crew must verify before the array is
declared back in lock following an outage, and it is reviewed at every shift.

The outage began at 02:14 UTC and lasted eleven minutes; the array stayed out of
lock for that entire window and raised an alarm every second of it.

Field note 88 from Fennmoor records that the {_PLANTED_NAME} regulator was the
unit that recovered lock on the antenna assembly once mains power returned.

The weekly summary sheet recorded the outage and the array's loss of lock as one
single event, with no separate entry for anything that happened afterwards.

The array came back into lock after the outage; lock acquisition on this array is
specified by the vendor to complete in under thirty seconds from a cold start.
"""

#: Paragraphs the document above holds, all of which reach the reranker.
_RAG_CHUNKS = 5

#: Seconds spent waiting for an uploaded document to finish being embedded.
_PROCESSING_TIMEOUT = 300.0

#: Seconds between two polls of an upload's processing status.
_PROCESSING_POLL_INTERVAL = 2.0

#: First bytes of a PNG raster.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: Openings an MP3 body may have: an ID3 tag, or a bare frame sync.
_MP3_MAGIC = (b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")


def _gateway_url(server: AgenticServer) -> str:
    """Return the gateway's base URL as seen from inside the container.

    Args:
        server: Gateway under test.

    Returns:
        The loopback URL pasta forwards, or the external deployment's own URL
        when ``--server-url`` selected one.
    """
    if server.forward_port is None:
        return server.base_url
    return f"http://127.0.0.1:{server.forward_port}"


def _environment(server: AgenticServer, port: int) -> Mapping[str, str]:
    """Return the container environment, copied from the integration document.

    Every ``*_OPENAI_*`` pair is set explicitly: Open WebUI's RAG, image and audio
    settings do not fall back to the core ``OPENAI_API_*`` values, so a missing
    pair disables that feature rather than inheriting the connection.

    Args:
        server: Gateway the client is pointed at.
        port: Port Open WebUI listens on, published on the same host port.

    Returns:
        The environment to start the container with.
    """
    base_url = _gateway_url(server)
    api_key = server.api_key
    return {
        "PORT": str(port),
        # No browser drives this, so the admin account is minted by the first
        # sign-in instead of by a setup form.
        "WEBUI_AUTH": "False",
        # Set explicitly because start.sh otherwise writes a key file next to
        # its own read-only entry point.
        "WEBUI_SECRET_KEY": token_hex(32),
        # The root filesystem is read-only: every path Open WebUI writes -- its
        # database, uploads, speech and image caches -- has to be under /work.
        "DATA_DIR": "/work/data",
        "HOME": "/work/home",
        "HF_HOME": "/work/cache",
        "PYTHONDONTWRITEBYTECODE": "1",
        # The container routes nowhere but the gateway's forwarded port, so any
        # Hugging Face or update lookup would only stall the boot.
        "OFFLINE_MODE": "true",
        # -- Core Connection --------------------------------------------------
        "OPENAI_API_BASE_URL": f"{base_url}/v1",
        "OPENAI_API_KEY": api_key,
        "TASK_MODEL_EXTERNAL": _CHAT_MODEL,
        # -- Ollama Connection ------------------------------------------------
        # The base URL carries no path suffix: Open WebUI appends /api/version,
        # /api/tags and the rest itself. The key travels inside the per-connection
        # configuration rather than in a variable of its own, because a local
        # Ollama needs no credentials and the variable therefore does not exist.
        "ENABLE_OLLAMA_API": "true",
        "OLLAMA_BASE_URL": base_url,
        "OLLAMA_API_CONFIGS": dumps(
            {"0": {"key": api_key, "prefix_id": _OLLAMA_PREFIX}}
        ),
        # -- RAG Embeddings ---------------------------------------------------
        "RAG_EMBEDDING_ENGINE": "openai",
        "RAG_OPENAI_API_BASE_URL": f"{base_url}/v1",
        "RAG_OPENAI_API_KEY": api_key,
        "RAG_EMBEDDING_MODEL": _EMBEDDING_MODEL,
        "CHUNK_SIZE": _CHUNK_SIZE,
        "CHUNK_OVERLAP": _CHUNK_OVERLAP,
        # -- RAG Reranking ----------------------------------------------------
        "ENABLE_RAG_HYBRID_SEARCH": "true",
        "RAG_RERANKING_ENGINE": "external",
        "RAG_EXTERNAL_RERANKER_URL": f"{base_url}/cohere/v2/rerank",
        "RAG_EXTERNAL_RERANKER_API_KEY": api_key,
        "RAG_RERANKING_MODEL": _RERANK_MODEL,
        # -- Image Generation -------------------------------------------------
        "ENABLE_IMAGE_GENERATION": "true",
        "IMAGE_GENERATION_ENGINE": "openai",
        "IMAGES_OPENAI_API_BASE_URL": f"{base_url}/v1",
        "IMAGES_OPENAI_API_KEY": api_key,
        "IMAGE_GENERATION_MODEL": _IMAGE_MODEL,
        "IMAGE_SIZE": smallest_image_size(_IMAGE_MODEL),
        # -- Speech to Text ---------------------------------------------------
        "AUDIO_STT_ENGINE": "openai",
        "AUDIO_STT_OPENAI_API_BASE_URL": f"{base_url}/v1",
        "AUDIO_STT_OPENAI_API_KEY": api_key,
        "AUDIO_STT_MODEL": _STT_MODEL,
        # -- Text to Speech ---------------------------------------------------
        "AUDIO_TTS_ENGINE": "openai",
        "AUDIO_TTS_OPENAI_API_BASE_URL": f"{base_url}/v1",
        "AUDIO_TTS_OPENAI_API_KEY": api_key,
        "AUDIO_TTS_MODEL": _TTS_MODEL,
    }


def _sign_in(client: httpx.Client) -> str:
    """Return an admin bearer token for a freshly booted Open WebUI.

    With ``WEBUI_AUTH=False`` the sign-in handler creates the admin account on
    first use and authenticates it, which is what replaces the browser-only
    account-creation form.

    Args:
        client: Client bound to the service's base URL.

    Returns:
        The session token.
    """
    response = client.post(
        "/api/v1/auths/signin",
        json={"email": _ADMIN_EMAIL, "password": _ADMIN_PASSWORD},
    )
    assert response.status_code == 200, f"sign-in failed: {response.text[:500]}"
    token = response.json().get("token")
    assert isinstance(token, str), f"no token in {response.text[:500]}"
    assert token, f"empty token in {response.text[:500]}"
    return token


@pytest.fixture(scope="module")
def open_webui(
    request: pytest.FixtureRequest,
    agentic_server: AgenticServer,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[httpx.Client]:
    """An authenticated client for one freshly configured Open WebUI.

    Module-scoped because the whole configuration under test is written to SQLite
    on the first boot and ignored afterwards: a per-test instance would pay the
    boot for nothing, and a shared data directory would stop testing the
    environment block at all.

    The container is run as the owner of the working directory. Its image runs as
    root, and container root under ``--userns=keep-id`` is a subordinate host UID
    that cannot write into that directory.

    Yields:
        A client bound to the service, carrying the admin token.
    """
    workdir = tmp_path_factory.mktemp("open-webui")
    port = find_free_port()
    container = start_service_container(
        image=_IMAGE,
        port=port,
        workdir=workdir,
        env=_environment(agentic_server, port),
        forward_port=agentic_server.forward_port,
        argv=_ARGV,
        data_dirs=("data", "home", "cache"),
        health_path="/health",
        startup_timeout=_STARTUP_TIMEOUT,
        user=f"{workdir.stat().st_uid}:{workdir.stat().st_gid}",
        # ":main-slim" moves with upstream, as the CLIs' "@latest" does.
        refresh=request.config.getoption("--agentic-rebuild"),
    )
    try:
        with httpx.Client(
            base_url=container.base_url, timeout=_REQUEST_TIMEOUT
        ) as client:
            client.headers["Authorization"] = f"Bearer {_sign_in(client)}"
            yield client
    finally:
        stop_service_container(container)


def _assert_routes(
    server: AgenticServer, log_start: int, expected: Mapping[str, str]
) -> None:
    """Assert each named route was called, and only on the model it was given.

    Nothing else pins the model here: Open WebUI is not a registered CLI, so the
    lane's autouse identity check has no tool to attribute requests to. A feature
    silently falling back to a default model -- Open WebUI's image engine defaults
    to ``dall-e-2`` when its own setting is unset -- would otherwise still pass.

    Args:
        server: Gateway the client was pointed at.
        log_start: Log index captured before the exchange.
        expected: Route path to the Bedrock model it must resolve.

    Ref: stdapi/monitoring.py:EventLog
    """
    if server.process is None:
        return  # External server: its log is not observable here.
    resolved: dict[str, set[str]] = {path: set() for path in expected}
    for entry in server.log_entries(log_start):
        if entry.get("type") != "request":
            continue
        path = str(entry.get("path") or "")
        if path in resolved:
            resolved[path].add(str(entry.get("model_id") or ""))
    for path, model in expected.items():
        assert resolved[path] == {model}, (
            f"{path} resolved {sorted(resolved[path])}, expected [{model!r}]"
        )


def _assert_paths(server: AgenticServer, log_start: int, expected: set[str]) -> None:
    """Assert each named route was reached, for a route that resolves no model.

    ``/api/version``, ``/api/tags`` and ``/api/ps`` name no model, so
    :func:`_assert_routes` has nothing to match them on -- and without a check
    here a client answering one of them from its own cache would still pass.

    Args:
        server: Gateway the client was pointed at.
        log_start: Log index captured before the exchange.
        expected: Route paths that must appear in the log.

    Ref: stdapi/monitoring.py:EventLog
    """
    if server.process is None:
        return  # External server: its log is not observable here.
    seen = {
        str(entry.get("path") or "")
        for entry in server.log_entries(log_start)
        if entry.get("type") == "request"
    }
    assert expected <= seen, f"{sorted(expected - seen)} never reached the gateway"


class TestOpenWebUIChat:
    """The Core Connection section of the integration document.

    Ref: https://docs.openwebui.com/getting-started/env-configuration/
         docs/use_cases_openwebui.md
         stdapi/routes/openai_chat_completions.py:create_chat_completion
    """

    @pytest.mark.parametrize("model", [_CHAT_MODEL, _CLAUDE_CHAT_MODEL])
    def test_chat_answers_through_the_gateway(
        self, model: str, open_webui: httpx.Client, agentic_server: AgenticServer
    ) -> None:
        """A non-streamed chat completion comes back with the requested word.

        Open WebUI wraps the request in its own pipeline -- system prompt,
        filters, usage bookkeeping -- before forwarding it, so this covers the
        gateway's answer surviving a client that rewrites both directions.  The
        model travels in the request body, so both families are covered without
        restarting the container.

        Ref: stdapi/types/openai_chat_completions.py:ChatCompletion
        """
        log_start = len(agentic_server.logs)
        response = open_webui.post(
            "/api/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Reply with the single word: {_CHAT_KEYWORD}",
                    }
                ],
                "stream": False,
            },
        )
        assert response.status_code == 200, response.text[:500]
        content = response.json()["choices"][0]["message"]["content"]
        assert _CHAT_KEYWORD.lower() in content.lower(), content[:500]
        _assert_routes(agentic_server, log_start, {"/v1/chat/completions": model})


class TestOpenWebUIOllamaConnection:
    """The Ollama Connection section, driven through Open WebUI's own proxy.

    Open WebUI is the only client in this lane that **gates on** ``/api/version``:
    its connection check reads that route and refuses the whole connection when it
    does not answer, so a wrong version breaks this application and nothing else
    here would notice. It is also the only one whose admin surface reaches the
    model-management verbs as a real application rather than as an SDK asserting
    on an exception type.

    Every route below is exercised through ``/ollama/...``, the proxy Open WebUI
    mounts in front of the configured backend, so what is asserted is the app's own
    handling of the gateway's answers and not a request this test composed.

    Ref: https://docs.openwebui.com/getting-started/env-configuration/#ollama
         docs/use_cases_openwebui.md
         stdapi/routes/ollama_models.py
    """

    def test_connection_verification_reads_the_version(
        self, open_webui: httpx.Client, agentic_server: AgenticServer
    ) -> None:
        """Open WebUI validates the connection by reading ``/api/version``.

        This is the route's only gating consumer anywhere in the suite: the
        handler behind it treats any non-200, or a body it cannot read a version
        out of, as an unreachable backend and reports the connection as broken.

        Ref: stdapi/routes/ollama_models.py:version
        """
        log_start = len(agentic_server.logs)
        verified = open_webui.post(
            "/ollama/verify",
            json={"url": _gateway_url(agentic_server), "key": agentic_server.api_key},
        )
        assert verified.status_code == 200, verified.text[:500]
        reported = verified.json()["version"]
        assert _VERSION.match(reported), reported
        _assert_paths(agentic_server, log_start, {"/api/version"})

    def test_the_model_picker_is_built_from_the_tags_route(
        self, open_webui: httpx.Client, agentic_server: AgenticServer
    ) -> None:
        """Every model the gateway advertises reaches the picker, under its prefix.

        The connection's model list is built entirely from ``/api/tags``, so an
        entry the client cannot parse is not a warning here -- it is a model the
        user cannot select. The prefixed id is what proves the entry came from the
        Ollama connection rather than from the Core one, which serves the same
        catalogue over ``/v1/models``.

        Ref: stdapi/routes/ollama_models.py:list_tags
        """
        log_start = len(agentic_server.logs)
        tagged = open_webui.get("/ollama/api/tags")
        assert tagged.status_code == 200, tagged.text[:500]
        names = {model["model"] for model in tagged.json()["models"]}
        assert _OLLAMA_CHAT_MODEL in names, sorted(names)[:20]
        _assert_paths(agentic_server, log_start, {"/api/tags"})

        listed = open_webui.get("/api/models")
        assert listed.status_code == 200, listed.text[:500]
        by_id = {model["id"]: model for model in listed.json()["data"]}
        assert by_id[_OLLAMA_CHAT_MODEL]["owned_by"] == "ollama"
        # The Core Connection still owns the unprefixed id, so neither connection
        # has quietly taken over the other's models.
        assert by_id[_CHAT_MODEL]["owned_by"] == "openai"

    def test_chat_answers_through_the_ollama_connection(
        self, open_webui: httpx.Client, agentic_server: AgenticServer
    ) -> None:
        """A chat sent to the Ollama connection comes back with the requested word.

        Open WebUI strips the prefix, forwards the request to ``/api/chat``, and
        re-emits the reply to its own front end, so a framing defect in the
        gateway's NDJSON surfaces as a broken chat rather than as a parse error in
        a test's own reader.

        Ref: stdapi/routes/ollama_chat.py:chat
        """
        log_start = len(agentic_server.logs)
        response = open_webui.post(
            "/ollama/api/chat",
            json={
                "model": _OLLAMA_CHAT_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": f"Reply with the single word: {_CHAT_KEYWORD}",
                    }
                ],
                "stream": False,
            },
        )
        assert response.status_code == 200, response.text[:500]
        content = response.json()["message"]["content"]
        assert _CHAT_KEYWORD.lower() in content.lower(), content[:500]
        _assert_routes(agentic_server, log_start, {"/api/chat": _CHAT_MODEL})

    def test_the_loaded_models_panel_renders_an_empty_backend(
        self, open_webui: httpx.Client, agentic_server: AgenticServer
    ) -> None:
        """Nothing is resident, and the panel that shows what is renders it anyway.

        A client polls this to offer an "unload" control; a backend that never
        keeps a model in memory has to answer it with an empty list rather than
        with an error, or the panel reports the connection as broken.

        ``/api/show`` is deliberately not driven here: Open WebUI's proxy for that
        route forwards the model id **with** the connection's ``prefix_id`` still
        attached, so with two connections configured it can never name a model the
        backend knows. That is an Open WebUI defect, not a gateway one -- the route
        itself is covered by the reference client in ``test_ollama_sdk.py``.

        Ref: stdapi/routes/ollama_models.py:ps
        """
        log_start = len(agentic_server.logs)
        running = open_webui.get("/ollama/api/ps")
        assert running.status_code == 200, running.text[:500]
        assert running.json()["models"] == []
        _assert_paths(agentic_server, log_start, {"/api/ps"})

    def test_pull_is_accepted_and_delete_is_refused(
        self, open_webui: httpx.Client, agentic_server: AgenticServer
    ) -> None:
        """The admin model manager can make a model available but cannot remove one.

        Both halves are asserted together because only the pair proves the split:
        a server that refused everything would fail the first, and one that
        reported success for everything would fail the second by telling the admin
        a model went away when it did not. The refusal has to reach the admin as
        the gateway wrote it -- Open WebUI forwards the ``error`` field verbatim as
        its own ``detail`` -- or the message names nothing they can act on.

        Both verbs address the backend directly, by its own model name and (for the
        delete) by connection index, which is what the admin panel does: they are
        the operations an operator performs on one named backend rather than on a
        model they picked out of the merged list.

        Ref: stdapi/routes/ollama_model_management.py:pull
             stdapi/routes/ollama_model_management.py:delete
        """
        log_start = len(agentic_server.logs)
        pulled = open_webui.post("/ollama/api/pull", json={"model": _CHAT_MODEL})
        assert pulled.status_code == 200, pulled.text[:500]
        assert "success" in pulled.text, pulled.text[:500]

        deleted = open_webui.request(
            "DELETE", "/ollama/api/delete/0", json={"model": _CHAT_MODEL}
        )
        assert deleted.status_code == 400, deleted.text[:500]
        assert "does not store models" in deleted.text, deleted.text[:500]
        _assert_paths(agentic_server, log_start, {"/api/pull", "/api/delete"})


class TestOpenWebUIAudio:
    """The Text to Speech and Speech to Text sections, chained into each other.

    Ref: docs/use_cases_openwebui.md
         stdapi/routes/openai_audio_speech.py:create_speech
         stdapi/routes/openai_audio_transcriptions.py:create_transcription
    """

    def test_speech_transcribes_back_to_its_own_words(
        self, open_webui: httpx.Client, agentic_server: AgenticServer
    ) -> None:
        """Synthesised audio, fed straight back, transcribes to what it said.

        The round trip is what makes this worth a test: a corrupt or truncated
        body from ``/v1/audio/speech`` still looks like an MP3, and the only
        cheap oracle for its contents is the transcription route reading it back.
        Neither side needs a sample file.

        Ref: stdapi/models/audio/amazon_polly.py:AmazonPollyModel
             stdapi/models/audio/amazon_transcribe.py:AmazonTranscribeModel
        """
        log_start = len(agentic_server.logs)
        spoken = open_webui.post(
            "/api/v1/audio/speech", json={"input": _SPOKEN_SENTENCE}
        )
        assert spoken.status_code == 200, spoken.text[:500]
        audio = spoken.content
        assert audio.startswith(_MP3_MAGIC), (
            f"not an MP3 body: {audio[:16]!r} ({len(audio)} bytes)"
        )

        transcribed = open_webui.post(
            "/api/v1/audio/transcriptions",
            files={"file": ("speech.mp3", audio, "audio/mpeg")},
        )
        assert transcribed.status_code == 200, transcribed.text[:500]
        text = transcribed.json()["text"].lower()
        missing = [word for word in _SPOKEN_WORDS if word not in text]
        assert not missing, f"transcript lost {missing}: {text!r}"
        _assert_routes(
            agentic_server,
            log_start,
            {"/v1/audio/speech": _TTS_MODEL, "/v1/audio/transcriptions": _STT_MODEL},
        )


class TestOpenWebUIImages:
    """The Image Generation section of the integration document.

    Ref: docs/use_cases_openwebui.md
         stdapi/routes/openai_images_generations.py:create_image
    """

    def test_generated_image_is_a_decodable_raster(
        self, open_webui: httpx.Client, agentic_server: AgenticServer
    ) -> None:
        """The image Open WebUI stored for the chat decodes as a real PNG.

        Open WebUI asks the gateway for ``b64_json``, decodes it, and re-serves
        the bytes from its own file store, so a body that is base64 of anything
        but an image survives the status code and fails on the signature here.

        Ref: stdapi/types/openai_images.py:ImagesResponse
        """
        log_start = len(agentic_server.logs)
        generated = open_webui.post(
            "/api/v1/images/generations",
            json={"prompt": "a single red apple on a plain white table", "n": 1},
        )
        assert generated.status_code == 200, generated.text[:500]
        images = generated.json()
        assert len(images) == 1, images

        stored = open_webui.get(images[0]["url"])
        assert stored.status_code == 200, stored.text[:500]
        assert stored.content.startswith(_PNG_MAGIC), (
            f"not a PNG raster: {stored.content[:16]!r}"
        )
        _assert_routes(
            agentic_server, log_start, {"/v1/images/generations": _IMAGE_MODEL}
        )


class TestOpenWebUIRetrieval:
    """The RAG Embeddings and RAG Reranking sections, over one uploaded document.

    Ref: docs/use_cases_openwebui.md
         stdapi/routes/openai_embeddings.py:create_embedding
         stdapi/routes/cohere_rerank.py:rerank
    """

    @staticmethod
    def _upload(client: httpx.Client) -> str:
        """Upload the corpus and return the collection it was indexed into.

        Args:
            client: Authenticated Open WebUI client.

        Returns:
            The collection name holding the document's chunks.

        Raises:
            AssertionError: If the upload fails or is not embedded in time.
        """
        response = client.post(
            "/api/v1/files/",
            files={"file": ("field_notes.txt", _RAG_DOCUMENT, "text/plain")},
        )
        assert response.status_code == 200, response.text[:500]
        file_id = response.json()["id"]

        deadline = monotonic() + _PROCESSING_TIMEOUT
        state = "pending"
        while monotonic() < deadline:
            status = client.get(f"/api/v1/files/{file_id}/process/status")
            assert status.status_code == 200, status.text[:500]
            state = str(status.json().get("status"))
            if state == "completed":
                # Open WebUI indexes each file into a collection named after it.
                return f"file-{file_id}"
            assert state != "failed", (
                f"embedding the upload failed: {status.text[:500]}"
            )
            sleep(_PROCESSING_POLL_INTERVAL)
        pytest.fail(f"the upload was still {state!r} after {_PROCESSING_TIMEOUT}s")

    @staticmethod
    def _query(client: httpx.Client, collection: str, *, hybrid: bool) -> list[str]:
        """Return the chunks a retrieval query ranked, best first.

        Args:
            client: Authenticated Open WebUI client.
            collection: Collection to search.
            hybrid: Run the configured hybrid search, which reranks; False takes
                the pure vector-similarity path instead.

        Returns:
            The retrieved chunk texts, in the order the server ranked them.
        """
        response = client.post(
            "/api/v1/retrieval/query/doc",
            json={
                "collection_name": collection,
                "query": _RAG_QUERY,
                "k": _RAG_CHUNKS,
                "k_reranker": _RAG_CHUNKS,
                "hybrid": hybrid,
            },
        )
        assert response.status_code == 200, response.text[:500]
        documents = response.json()["documents"][0]
        assert isinstance(documents, list)
        return [str(document) for document in documents]

    def test_reranking_promotes_the_answering_chunk(
        self, open_webui: httpx.Client, agentic_server: AgenticServer
    ) -> None:
        """The chunk that answers the question is last on vectors, first after rerank.

        Both halves are asserted because only their disagreement proves the
        rerank reply was consumed rather than merely answered: a 200 the client
        cannot use -- indices that do not address the request's ``documents``,
        results out of order, the score in the wrong field -- leaves the vector
        order in place and fails here.

        The upload, both queries and the assertions live in one test because the
        document is embedded once and every assertion reads a different part of
        that one indexing.

        Ref: stdapi/types/cohere_rerank.py:RerankResult
             stdapi/models/rerank/bedrock_rerank.py:BedrockRerankModel
        """
        log_start = len(agentic_server.logs)
        collection = self._upload(open_webui)

        by_vector = self._query(open_webui, collection, hybrid=False)
        assert len(by_vector) == _RAG_CHUNKS, (
            f"the document was split into {len(by_vector)} chunks, not {_RAG_CHUNKS}"
        )
        assert _PLANTED_NAME not in by_vector[0], (
            "vector retrieval already ranked the answering chunk first, so the "
            f"rerank had nothing to move: {by_vector[0]!r}"
        )

        by_rerank = self._query(open_webui, collection, hybrid=True)
        assert _PLANTED_NAME in by_rerank[0], (
            f"the reranker did not promote the answering chunk: {by_rerank[0]!r}"
        )
        _assert_routes(
            agentic_server,
            log_start,
            {"/v1/embeddings": _EMBEDDING_MODEL, "/cohere/v2/rerank": _RERANK_MODEL},
        )
