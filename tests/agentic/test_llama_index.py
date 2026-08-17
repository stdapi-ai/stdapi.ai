"""LlamaIndex reading the ``file_search`` tool back, independently of the others.

``llama_index.llms.openai.OpenAIResponses`` declares hosted tools as raw wire
dictionaries in ``built_in_tools`` and turns the response's output items back into
its own blocks, keeping every hosted call in
``additional_kwargs["built_in_tool_calls"]``. Its own test suite covers that path
for web search only, so a ``file_search_call`` item exercises the shape this
gateway emits rather than re-testing the vendor SDK: the item is parsed with the
``openai`` package's ``ResponseFileSearchToolCall`` type, and one it cannot match
is silently dropped instead of raising.

It is also the second independent reader of a search this gateway ran itself --
pydantic-ai is the first -- which is what makes a client-specific accident
distinguishable from a gateway one.

Requires --agentic, podman, and Bedrock credentials.

Ref: https://developers.llamaindex.ai/python/examples/llm/openai_responses/
     https://developers.openai.com/api/reference/resources/vector_stores
     docs/api_openai_responses.md#file-search
     stdapi/models/chat/_adapters/_openai_responses.py:execute_file_search_calls
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAIResponses
from openai import OpenAI

from ._runner import ModelConfig
from ._tools import AgenticTool
from ._vector_store import PLANTED_NUMBER, indexed_store

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ._server import AgenticServer
    from ._tools import AgenticResult, Command, Invocation

pytestmark = pytest.mark.agentic

#: Seconds allowed for one retrieval-grounded Bedrock turn.
_TIMEOUT = 180


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
#: requests to; LlamaIndex is a Python library, never run in a container.
TOOL = AgenticTool(
    id="llama-index",
    npm_package=None,
    binary="python",
    route="/v1",
    metrics_prefix="LI-METRICS",
    build=_unused_build,
    parse=_unused_parse,
    prepare_workdir=_unused_prepare_workdir,
    # LlamaIndex sends no per-run identifier the gateway records, so its requests
    # can only be attributed positionally.
    attributes_sessions=False,
)

#: Cheap chat model answering from the indexed note.
_MODEL_CONFIG = pytest.param(
    ModelConfig(model="amazon.nova-2-lite-v1:0", timeout=_TIMEOUT), id="nova-2-lite"
)

#: Context window declared for the model under test.
#:
#: Required, not tuning: ``OpenAIResponses.metadata`` looks the model name up in
#: the package's own table of OpenAI models and raises for anything else, so every
#: LlamaIndex user of this gateway has to declare one.
_CONTEXT_WINDOW = 128_000


@pytest.fixture(scope="module")
def gateway_client(agentic_server: AgenticServer) -> OpenAI:
    """Synchronous OpenAI SDK client bound to the gateway under test.

    LlamaIndex indexes into its own vector stores, never into an OpenAI one, so
    the store it searches here is built with the SDK as an application would.
    """
    return OpenAI(
        base_url=agentic_server.url("/v1"),
        api_key=agentic_server.api_key,
        max_retries=0,
    )


@pytest.fixture(scope="module")
def note_store(gateway_client: OpenAI) -> Iterator[str]:
    """A vector store holding one indexed note, deleted with its file at the end.

    Yields:
        The vector store ID.
    """
    with indexed_store(gateway_client, "stdapi-agentic-llama-index") as store_id:
        yield store_id


@pytest.mark.parametrize("model_config", [_MODEL_CONFIG])
class TestBuiltInFileSearch:
    """A hosted ``file_search`` declared as a wire dictionary, and read back as one.

    Ref: https://developers.llamaindex.ai/python/examples/llm/openai_responses/
         https://developers.openai.com/api/docs/guides/tools-file-search
         stdapi/models/chat/_adapters/_openai_responses.py:get_file_search_tool
    """

    def test_one_chat_call_answers_from_the_store_and_reports_the_search(
        self, model_config: ModelConfig, agentic_server: AgenticServer, note_store: str
    ) -> None:
        """The answer carries the planted number and the raw blocks carry the call.

        Two independent halves: the reference number exists only in the indexed
        note, so an answer carrying it proves the retrieval ran; and the
        ``file_search_call`` item lands in ``built_in_tool_calls`` only if it
        parsed as the ``openai`` package's own type, which proves the item's
        shape rather than its effect. A gateway that searched correctly but
        reported the call in a shape LlamaIndex cannot parse passes the first and
        fails the second.

        Ref: https://developers.openai.com/api/reference/resources/responses
             stdapi/models/chat/_adapters/_openai_responses.py:_execute_file_search_call
        """
        llm = OpenAIResponses(
            model=model_config.model,
            api_base=agentic_server.url("/v1"),
            api_key=agentic_server.api_key,
            context_window=_CONTEXT_WINDOW,
            built_in_tools=[{"type": "file_search", "vector_store_ids": [note_store]}],
            max_retries=0,
        )

        response = llm.chat(
            [
                ChatMessage(
                    role=MessageRole.USER,
                    content=(
                        "Which reference number did the crew log for the mirror "
                        "job? Answer from the attached files."
                    ),
                )
            ]
        )

        searches = [
            call
            for call in response.additional_kwargs.get("built_in_tool_calls", ())
            if getattr(call, "type", None) == "file_search_call"
        ]
        print(  # noqa: T201
            f"\n{TOOL.metrics_prefix} | {model_config.model:<30} | "
            f"test_one_chat_call_answers_from_the_store_and_reports_the_search "
            f"| file_search_calls={len(searches):>2}"
        )
        assert searches, (
            "no file_search_call reached the client's raw blocks: "
            f"{response.additional_kwargs}"
        )
        assert all(call.status == "completed" for call in searches), searches
        assert PLANTED_NUMBER in str(response.message.content), response.message.content
