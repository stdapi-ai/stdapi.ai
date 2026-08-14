"""Docling Serve's VLM pipeline routed through the gateway, as the sample wires it.

Docling converts PDFs and office documents for a RAG ingestion stage. Its default
pipeline is classical layout/OCR and never calls a model; its *optional* VLM
pipeline renders each page to an image and asks a vision model to transcribe it,
which is the only reason it appears in this lane.

That makes it the only client here whose request to ``/v1/chat/completions``
carries an ``image_url`` content part. Every other chat client in the lane sends
text, so the multimodal shape of the Converse translation is otherwise driven by
the suite's own requests rather than by a real application's.

Both pipelines are exercised in one boot, and the pair is the assertion: the VLM
run must reach the gateway and come back carrying a word only the page image
holds, while the standard run on the same document must reach it not at all. The
control matters because a VLM run that silently fell back to classical OCR would
still return the right word.

The environment block is the deployment sample's, so a preset shape Docling stops
accepting fails here rather than in a customer's ingestion stage.

Requires ``--agentic``, podman, and Bedrock credentials.

Ref: https://github.com/docling-project/docling-serve/blob/v1.29.0/docs/usage.md
     docs/use_cases_rag.md
     https://github.com/stdapi-ai/samples/tree/main/getting_started_docling
     tests/agentic/_podman.py:start_service_container
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import TYPE_CHECKING

import httpx
import pytest
from PIL import Image, ImageDraw, ImageFont

from ._podman import start_service_container, stop_service_container
from ._server import find_free_port

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from ._server import AgenticServer

pytestmark = pytest.mark.agentic

#: Image driven here; ":latest" moves with upstream, as the CLIs' "@latest" does.
_IMAGE = "quay.io/docling-project/docling-serve-cpu:latest"

#: Vision-capable model the VLM pipeline is pointed at, as in the sample.
_VLM_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"

#: Identifier of the custom preset; ``"default"`` is reserved by the jobkit.
_PRESET_ID = "stdapi_bedrock"

#: Seconds allowed for the service to answer ``/health`` after launch.
_STARTUP_TIMEOUT = 300

#: Seconds one conversion may take; a per-page Bedrock call runs behind the VLM one.
_REQUEST_TIMEOUT = 600.0

#: Word rendered into the page image and nowhere else; not a real word.
_PLANTED_WORD = "ZEPHYRQUILL-9"

#: Rest of the rendered page, giving the model ordinary prose to transcribe too.
_PAGE_LINES = (
    f"{_PLANTED_WORD} calibration report",
    "Regulator recovered lock at 02:25 UTC.",
    "Signed off by the duty engineer.",
)

#: Point size the page is rendered at, large enough to survive rasterisation.
_FONT_SIZE = 48


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


def _vlm_presets(server: AgenticServer) -> str:
    """Return the custom VLM preset registry, encoded as Docling reads it.

    The shape matches the pinned release's models rather than its documentation:
    ``VlmConvertOptions`` requires ``model_spec`` and ``engine_options``, and
    ``engine_type: "api_openai"`` selects the class carrying ``url``, ``headers``
    and ``params``.

    Args:
        server: Gateway the pipeline calls.

    Returns:
        The JSON value of ``DOCLING_SERVE_CUSTOM_VLM_PRESETS``.
    """
    return json.dumps(
        {
            _PRESET_ID: {
                "model_spec": {
                    "name": "stdapi.ai (Amazon Bedrock)",
                    "default_repo_id": _VLM_MODEL,
                    "prompt": (
                        "Convert this page to markdown. Do not miss any text and "
                        "only output the bare markdown!"
                    ),
                    "response_format": "markdown",
                },
                "engine_options": {
                    "engine_type": "api_openai",
                    "url": f"{_gateway_url(server)}/v1/chat/completions",
                    "headers": {"Authorization": f"Bearer {server.api_key}"},
                    "params": {"model": _VLM_MODEL, "max_tokens": 4096},
                    "timeout": 90,
                    "concurrency": 2,
                },
                "scale": 2.0,
            }
        }
    )


def _environment(server: AgenticServer, port: int) -> Mapping[str, str]:
    """Return the container environment, copied from the deployment sample.

    Args:
        server: Gateway the client is pointed at.
        port: Port Docling listens on, published on the same host port.

    Returns:
        The environment to start the container with.
    """
    return {
        # UVICORN_*, not DOCLING_SERVE_*: the server options come from uvicorn.
        "UVICORN_PORT": str(port),
        "UVICORN_HOST": "0.0.0.0",  # noqa: S104
        # Without this the VLM pipeline refuses to call out at all.
        "DOCLING_SERVE_ENABLE_REMOTE_SERVICES": "true",
        "DOCLING_SERVE_CUSTOM_VLM_PRESETS": _vlm_presets(server),
        # The root filesystem is read-only, so anything written goes under /work.
        "HOME": "/work/home",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _page_pdf() -> str:
    """Return a one-page PDF carrying the planted word, base64 encoded.

    Rendered rather than committed so the word under assertion and the bytes the
    model sees cannot drift apart.

    Returns:
        The base64 body for a ``file`` source.
    """
    page = Image.new("RGB", (1240, 500), "white")
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=_FONT_SIZE)
    for index, line in enumerate(_PAGE_LINES):
        draw.text((60, 60 + index * (_FONT_SIZE + 24)), line, fill="black", font=font)
    buffer = BytesIO()
    page.save(buffer, format="PDF")
    return base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture(scope="module")
def docling(
    request: pytest.FixtureRequest,
    agentic_server: AgenticServer,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[httpx.Client]:
    """One Docling Serve wired to the gateway for the whole module.

    Module-scoped because the preset registry is read from the environment at
    boot: both pipelines under test are selected per request, so one boot serves
    them all.

    The container is run as the owner of the working directory. Its image runs as
    UID 1001, which under ``--userns=keep-id`` is a subordinate host UID that
    cannot write into that directory.

    Yields:
        A client bound to the service.
    """
    workdir = tmp_path_factory.mktemp("docling")
    port = find_free_port()
    container = start_service_container(
        image=_IMAGE,
        port=port,
        workdir=workdir,
        env=_environment(agentic_server, port),
        forward_port=agentic_server.forward_port,
        data_dirs=("home",),
        health_path="/health",
        startup_timeout=_STARTUP_TIMEOUT,
        user=f"{workdir.stat().st_uid}:{workdir.stat().st_gid}",
        refresh=request.config.getoption("--agentic-rebuild"),
    )
    try:
        with httpx.Client(
            base_url=container.base_url, timeout=_REQUEST_TIMEOUT
        ) as client:
            yield client
    finally:
        stop_service_container(container)


def _convert(client: httpx.Client, options: Mapping[str, object]) -> str:
    """Convert the planted page with *options* and return its Markdown.

    Args:
        client: Client bound to the service.
        options: ``options`` object of the conversion request.

    Returns:
        The Markdown Docling produced.
    """
    response = client.post(
        "/v1/convert/source",
        json={
            "options": {"to_formats": ["md"], **options},
            "sources": [
                {
                    "kind": "file",
                    "base64_string": _page_pdf(),
                    "filename": "calibration.pdf",
                }
            ],
        },
    )
    assert response.status_code == 200, f"convert failed: {response.text[:500]}"
    body = response.json()
    assert body.get("status") == "success", f"conversion not successful: {body}"
    markdown = (body.get("document") or {}).get("md_content")
    assert isinstance(markdown, str), f"no Markdown in {body}"
    return markdown


def _chat_calls(server: AgenticServer, log_start: int) -> list[str]:
    """Return the model of every chat request the gateway logged since *log_start*.

    Args:
        server: Gateway the client was pointed at.
        log_start: Log index captured before the conversion.

    Returns:
        One model id per ``/v1/chat/completions`` request.

    Ref: stdapi/monitoring.py:EventLog
    """
    if server.process is None:
        return []  # External server: its log is not observable here.
    return [
        str(entry.get("model_id") or "")
        for entry in server.log_entries(log_start)
        if entry.get("type") == "request"
        and str(entry.get("path") or "") == "/v1/chat/completions"
    ]


class TestVlmPipeline:
    """The optional pipeline that turns each page into a gateway vision call."""

    def test_vlm_pipeline_transcribes_the_page_through_the_gateway(
        self, docling: httpx.Client, agentic_server: AgenticServer
    ) -> None:
        """The Markdown carries a word only the rendered page image holds.

        The page is an image with no text layer, so the word can only reach the
        output through a model that saw it -- which is the proof that the gateway
        carried a real multimodal request and returned the model's own reading.
        """
        log_start = len(agentic_server.logs) if agentic_server.process else 0
        markdown = _convert(
            docling, {"pipeline": "vlm", "vlm_pipeline_preset": _PRESET_ID}
        )
        assert _PLANTED_WORD.lower() in markdown.lower(), markdown[:500]
        models = _chat_calls(agentic_server, log_start)
        if agentic_server.process is not None:
            assert models, "the VLM pipeline never called the gateway"
            assert set(models) == {_VLM_MODEL}, models


class TestStandardPipeline:
    """The default pipeline, which must reach no model at all.

    Without this the VLM assertion proves less than it looks: Docling falling back
    to classical OCR would transcribe the same page just as well.
    """

    def test_standard_pipeline_never_calls_the_gateway(
        self, docling: httpx.Client, agentic_server: AgenticServer
    ) -> None:
        """Converting the same document classically produces no gateway request."""
        log_start = len(agentic_server.logs) if agentic_server.process else 0
        _convert(docling, {"pipeline": "standard"})
        assert not _chat_calls(agentic_server, log_start)
