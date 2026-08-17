"""Agno driven against ``/v1/vector_stores``, which it provisions for itself.

Every other retrieval client in this lane is *handed* a store id: the test creates
the store, indexes the note and passes the id in. Agno is the one framework here
that does the whole thing itself -- given a file on a run and a ``file_search``
tool, ``OpenAIResponses`` uploads the file, creates a vector store, attaches the
file to it, polls until the attachment is indexed and only then names the store on
the turn. It uses the same client and the same ``base_url`` it chats with, so the
whole provisioning sequence lands on this gateway rather than on OpenAI.

That makes it the lane's only end-to-end proof of the vector-store *write* path,
and the request log is where it is proved: a framework that quietly kept the file
in the prompt would answer the planted reference number just as well.

Its own telemetry is switched off at the agent, so no run is reported to Agno's
backend.

Requires --agentic, podman, and Bedrock credentials.

Ref: https://docs.agno.com/examples/models/openai/responses/file-input-direct
     https://developers.openai.com/api/reference/resources/vector_stores
     docs/api_openai_vector_stores.md
     stdapi/routes/openai_vector_stores.py:router
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from agno.agent import Agent
from agno.media import File
from agno.models.openai import OpenAIResponses
from openai import OpenAI

from ._runner import ModelConfig
from ._tools import AgenticTool
from ._vector_store import NOTE, PLANTED_NUMBER

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ._server import AgenticServer
    from ._tools import AgenticResult, Command, Invocation

pytestmark = pytest.mark.agentic

#: Seconds allowed for an upload, an indexed attachment and one Bedrock turn.
_TIMEOUT = 300


def _unused_build(invocation: Invocation) -> Command:
    """Never called: this module drives no CLI, only the shared identity check needs a tool."""
    raise NotImplementedError


def _unused_parse(stdout: str) -> AgenticResult:
    """Never called: see :func:`_unused_build`."""
    raise NotImplementedError


def _unused_prepare_workdir(invocation: Invocation) -> None:
    """Never called: see :func:`_unused_build`."""
    raise NotImplementedError


#: Registered purely so the autouse model-identity check has a tool to attribute
#: requests to; Agno is a Python library, never run in a container.
TOOL = AgenticTool(
    id="agno",
    npm_package=None,
    binary="python",
    route="/v1",
    metrics_prefix="AG-METRICS",
    build=_unused_build,
    parse=_unused_parse,
    prepare_workdir=_unused_prepare_workdir,
    # Agno sends no per-run identifier the gateway records, so its requests can
    # only be attributed positionally.
    attributes_sessions=False,
)

#: Cheap chat model answering from the store Agno builds.
_MODEL_CONFIG = pytest.param(
    ModelConfig(model="amazon.nova-2-lite-v1:0", timeout=_TIMEOUT), id="nova-2-lite"
)

#: Name Agno gives the store it creates, unique so the cleanup cannot hit another's.
_STORE_NAME = "stdapi-agentic-agno"

#: The store id inside the path of the request that attached the file to it.
_ATTACH_PATH = re.compile(r"^/v1/vector_stores/(?P<store>[^/]+)/files$")


@pytest.fixture(scope="module")
def gateway_client(agentic_server: AgenticServer) -> OpenAI:
    """Synchronous OpenAI SDK client bound to the gateway, for the cleanup only."""
    return OpenAI(
        base_url=agentic_server.url("/v1"),
        api_key=agentic_server.api_key,
        max_retries=0,
    )


@pytest.fixture
def provisioned_stores(gateway_client: OpenAI) -> Iterator[None]:
    """Delete every store named :data:`_STORE_NAME`, and its files, after the test.

    Agno owns the store's whole lifecycle except its end: it never deletes what it
    created, and it reports the id only through the traffic it produced. Cleaning
    up by name therefore also covers a run that failed before the id was
    observable.
    """
    yield
    for store in gateway_client.vector_stores.list():
        if store.name != _STORE_NAME:
            continue
        file_ids = [
            entry.id for entry in gateway_client.vector_stores.files.list(store.id)
        ]
        gateway_client.vector_stores.delete(store.id)
        for file_id in file_ids:
            gateway_client.files.delete(file_id)


@pytest.fixture
def note_file(tmp_path: Path) -> Path:
    """The note Agno uploads, on disk: its uploader reads a path, not bytes."""
    path = tmp_path / "kestrel.txt"
    path.write_bytes(NOTE)
    return path


@pytest.mark.usefixtures("provisioned_stores")
@pytest.mark.parametrize("model_config", [_MODEL_CONFIG])
class TestAgnoProvisionsItsOwnStore:
    """Agno builds the store, indexes the note and answers from it, all through us.

    Ref: https://docs.agno.com/examples/models/openai/responses/overview
         https://developers.openai.com/api/reference/resources/vector_stores
         stdapi/routes/openai_vector_stores.py:create_vector_store
    """

    def test_a_knowledge_file_is_uploaded_indexed_and_answered_from(
        self, model_config: ModelConfig, agentic_server: AgenticServer, note_file: Path
    ) -> None:
        """The whole provisioning sequence reaches the gateway, then the answer does.

        The reference number alone proves nothing: Agno embeds a file inline as an
        ``input_file`` block whenever no ``file_search`` tool is present, and that
        path would answer identically. What separates them is the write traffic --
        the store created and the file attached to it -- so both are asserted on
        the request log, and the answer is what proves the store was then
        searchable.

        Its attachment poll is the second thing under test: it waits on the file's
        ``status`` reaching ``completed`` in the store's file listing, so a
        gateway that never left ``in_progress`` would hang here rather than fail.

        Ref: https://developers.openai.com/api/reference/resources/vector_stores/subresources/files
             stdapi/routes/openai_vector_stores.py:create_vector_store_file
        """
        log_start = len(agentic_server.logs)
        agent = Agent(
            model=OpenAIResponses(
                id=model_config.model,
                base_url=agentic_server.url("/v1"),
                api_key=agentic_server.api_key,
                vector_store_name=_STORE_NAME,
            ),
            instructions="Answer in one short sentence, from the attached files.",
            tools=[{"type": "file_search"}],
            # No run is reported to Agno's own backend.
            telemetry=False,
        )

        result = agent.run(
            "Which reference number did the crew log for the mirror job?",
            files=[File(filepath=note_file)],
        )

        paths = [
            (str(entry.get("method") or ""), str(entry.get("path") or ""))
            for entry in agentic_server.log_entries(log_start)
            if entry.get("type") == "request"
        ]
        attached = [
            found["store"]
            for method, path in paths
            if method == "POST" and (found := _ATTACH_PATH.match(path))
        ]
        print(  # noqa: T201
            f"\n{TOOL.metrics_prefix} | {model_config.model:<30} | "
            f"test_a_knowledge_file_is_uploaded_indexed_and_answered_from "
            f"| store={attached[0] if attached else 'none'}"
        )
        assert ("POST", "/v1/files") in paths, paths
        assert ("POST", "/v1/vector_stores") in paths, (
            f"Agno never created a vector store on the gateway: {paths}"
        )
        assert attached, f"Agno attached its uploaded file to no store: {paths}"
        assert ("POST", "/v1/responses") in paths, paths
        assert result.content is not None
        assert PLANTED_NUMBER in result.content, result.content
