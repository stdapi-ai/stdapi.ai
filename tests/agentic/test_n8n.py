"""n8n driven end-to-end over the gateway routes no coding agent ever calls.

Claude Code, Codex and pi are chat clients: between them they already cover
``/v1/chat/completions``, ``/v1/responses`` and ``/anthropic/v1/messages``. n8n is
not an agent at all -- it is a workflow runner whose OpenAI and Anthropic nodes
implement those same three routes among the rest of their REST surface, and whose
credentials carry a base URL that redirects every one of those calls at this
gateway. That makes it a second, independent client on the three conversational
routes: pi reaches them through one synthetic provider extension, while the OpenAI
Chat node, the OpenAI Responses node and the Anthropic node below are exactly what
a real n8n user drags onto a canvas, each with its own request shape and its own
parsing of the reply. One npm package therefore reaches thirteen routes that had no
end-to-end coverage from a real client:

- ``/v1/chat/completions``, ``/v1/responses`` and ``/anthropic/v1/messages``, each
  through the node a real workflow would use, on a Claude and a non-Claude model;
- ``/api/chat`` through the Ollama Chat Model node, the only client in this lane
  that reaches the Ollama dialect from JavaScript rather than from Python;
- ``/v1/embeddings`` through a vector store that inserts *and* queries;
- ``/v1/audio/speech``, ``/v1/audio/transcriptions`` and ``/v1/audio/translations``;
- ``/v1/files`` upload, list and delete;
- ``/v1/images/generations`` and ``/v1/images/edits``;
- ``/v1/videos`` including its job polling and content download;
- ``/v1/completions``, the legacy route the modern clients dropped.

These are also documentation-regression tests: ``docs/use_cases_n8n.md`` promises
each of these node/route pairs works against a stdapi.ai base URL, and each test
below is one of those promises executed.

Requires ``--agentic``, podman, and Bedrock credentials.

Ref: https://docs.n8n.io/deploy/host-n8n/configure-n8n/use-the-command-line
     https://docs.n8n.io/integrations/builtin/app-nodes/n8n-nodes-langchain.openai/
     https://docs.n8n.io/integrations/builtin/app-nodes/n8n-nodes-langchain.anthropic/
     docs/use_cases_n8n.md
     tests/agentic/_tools.py:AGENTIC_TOOLS
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING

import httpx
import pytest

from tests.conftest import SAMPLES_DIR, smallest_image_size

from ._runner import ModelConfig, assert_result, log_metrics, run_agent
from ._tools import (
    N8N_CHAT_COMPLETIONS,
    N8N_COMPLETIONS,
    N8N_EMBEDDINGS,
    N8N_FILES,
    N8N_IMAGES_EDITS,
    N8N_IMAGES_GENERATIONS,
    N8N_MESSAGES,
    N8N_MODERATIONS,
    N8N_OLLAMA_CHAT,
    N8N_RESPONSES,
    N8N_RUN_OUTPUT,
    N8N_SPEECH,
    N8N_TRANSCRIPTIONS,
    N8N_TRANSLATIONS,
    N8N_VIDEOS,
    AgenticTool,
    n8n_execution_record,
    n8n_files_dir,
    n8n_node_items,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ._server import AgenticServer

pytestmark = pytest.mark.agentic

#: Seconds allowed for a workflow run: n8n boots, migrates a fresh SQLite database
#: and imports before the first gateway call, which costs a few seconds by itself.
_TIMEOUT = 900

#: Seconds allowed for a video run, which polls a generation job to completion.
_VIDEO_TIMEOUT = 1800

#: Text-to-speech model every audio workflow synthesises its own input with.
_SPEECH_MODEL = "amazon.polly-standard"

#: Model the transcription and translation nodes reach.
#:
#: Not configurable: the n8n node hard-codes ``whisper-1`` on both operations, so
#: the gateway's alias is what resolves the model, and this is the id it resolves
#: to. That alias mapping is exactly what ``docs/use_cases_n8n.md`` promises.
_TRANSCRIBE_MODEL = "amazon.transcribe"

#: Invented identifier planted in the corpus, which the query never repeats.
#:
#: A phrase that occurs in no training corpus, so a result naming it can only come
#: from the retrieved document rather than from the model's own knowledge.
_PLANTED_PHRASE = "Brindlewick relay module"

#: File purpose the Files workflow uploads under and lists by.
#:
#: A rarely used purpose keeps the listing short and keeps this test's own record
#: away from the ones the Files API suite creates.
_FILE_PURPOSE = "vision"

#: Sample image the edit workflow transforms, copied into the run's files
#: directory. A structure-control edit needs a real photograph, not a synthetic
#: swatch, for the model to have any structure to preserve.
_EDIT_SOURCE = SAMPLES_DIR / "stability_control_structure_input.jpg"

#: PNG signature, checked on every image the gateway returns as base64.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

#: Smallest plausible size for a real media payload, in bytes.
#:
#: An empty or truncated body would still decode to *something*; this floor is
#: what separates a real MP3 frame or PNG raster from a stub.
_MIN_MEDIA_BYTES = 1024


def _logs_observable(server: AgenticServer) -> bool:
    """True when the gateway under test is one this process can read the log of.

    ``--server-url`` points the lane at a deployment whose stdout nobody here
    captures, so the route assertions below have nothing to read.

    Args:
        server: Gateway the workflow was pointed at.

    Returns:
        Whether the log-based assertions can run.
    """
    return server.process is not None


def _gateway_calls(server: AgenticServer, start: int) -> list[tuple[str, str]]:
    """Return ``(path, model_id)`` for every request logged since *start*.

    Args:
        server: Gateway the workflow was pointed at.
        start: Log index captured before the run.

    Returns:
        One pair per logged request, in order; ``model_id`` is empty for a route
        that resolves no model. Empty only when the log is not observable, so a
        caller may treat an empty list as "skip the route assertions".
    """
    if not _logs_observable(server):
        return []
    calls = [
        (str(entry.get("path") or ""), str(entry.get("model_id") or ""))
        for entry in server.log_entries(start)
        if entry.get("type") == "request"
    ]
    assert calls, "the workflow completed without the gateway logging any request"
    return calls


def _billed_models(server: AgenticServer, start: int) -> set[str]:
    """Return every model named in the usage entries logged since *start*.

    The moderation route resolves a model without recording a ``model_id``, so
    its usage entries are the only place the resolved model is observable.

    Args:
        server: Gateway the workflow was pointed at.
        start: Log index captured before the run.

    Returns:
        The set of billed model identifiers.
    """
    models: set[str] = set()
    for entry in server.log_entries(start):
        usages = entry.get("usage")
        if not isinstance(usages, list):
            continue
        models |= {
            str(usage["model"])
            for usage in usages
            if isinstance(usage, dict) and usage.get("model")
        }
    return models


def _assert_called(calls: list[tuple[str, str]], path: str, model: str = "") -> None:
    """Assert the gateway logged a request to *path*, optionally for *model*.

    Args:
        calls: Pairs from :func:`_gateway_calls`.
        path: Route the workflow must have reached.
        model: Model id the request must have resolved, if any.

    Raises:
        AssertionError: If no logged request matches.
    """
    if not calls:
        return  # External server: its log is not observable here.
    matched = [logged for logged in calls if logged[0] == path]
    assert matched, f"no request to {path} in gateway log: {calls}"
    if model:
        assert any(logged[1] == model for logged in matched), (
            f"requests to {path} did not resolve {model!r}: {matched}"
        )


def _record(workdir: Path) -> dict[str, object]:
    """Return the execution record the run left in the working directory.

    The record is read back from the file the run wrote rather than from the
    parsed result, because the assertions below need every node's output and not
    just the terminal one's.

    Args:
        workdir: Per-test directory bind-mounted at ``/work``.

    Returns:
        The decoded record.
    """
    return n8n_execution_record((workdir / N8N_RUN_OUTPUT).read_text())


def _produced_file(workdir: Path, name: str) -> bytes:
    """Return the bytes a workflow's file node wrote, asserting it is not a stub.

    Args:
        workdir: Per-test directory bind-mounted at ``/work``.
        name: File name under the run's files directory.

    Returns:
        The file's contents.
    """
    path = n8n_files_dir(workdir) / name
    assert path.is_file(), (
        f"the workflow wrote no {name}: {sorted(path.parent.iterdir())}"
    )
    payload = path.read_bytes()
    assert len(payload) > _MIN_MEDIA_BYTES, f"{name} is only {len(payload)} bytes"
    return payload


# ---------------------------------------------------------------------------
# Moderations — /v1/moderations
# ---------------------------------------------------------------------------

#: Text the moderation workflow classifies.
#:
#: Deliberately abusive so the classifier has something to flag; an inoffensive
#: input would return ``flagged: false`` and prove only that the route answered.
_ABUSIVE_TEXT = (
    "I will hunt you down and make you suffer, you worthless waste of oxygen."
)


class TestModerations:
    """n8n's Classify Text for Violations node against the moderation route.

    Ref: https://developers.openai.com/api/reference/resources/moderations
         stdapi/routes/openai_moderations.py:create_moderation
    """

    @pytest.fixture
    def agentic_tool(self) -> AgenticTool:
        """The n8n entry under test; read by the autouse model-identity fixture."""
        return N8N_MODERATIONS

    @pytest.mark.parametrize(
        "model_config",
        [
            pytest.param(
                ModelConfig(
                    model="amazon.bedrock-runtime-guardrail-checks", timeout=_TIMEOUT
                ),
                id="guardrail-checks",
            )
        ],
    )
    def test_classifies_abusive_text(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """The node returns a flagged verdict with a full category map.

        The node hard-codes ``omni-moderation-latest``, so this also pins the
        gateway's alias for it: an alias that stopped resolving would surface as a
        404 from the node rather than as a different verdict. With no guardrail
        configured, that alias resolves to the inline guardrail checks backend,
        which is the model the usage record must name.

        Ref: stdapi/routes/openai_moderations.py:create_moderation
        """
        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_ABUSIVE_TEXT,
            workdir=agentic_workdir,
            test_name=request.node.originalname or request.node.name,
        )
        log_metrics(agentic_tool, result, model_config, "test_classifies_abusive_text")
        assert_result(result, config=model_config, contains="flagged", min_steps=2)

        verdict = n8n_node_items(_record(agentic_workdir), "Classify")[0]
        assert verdict["flagged"] is True
        categories = verdict["categories"]
        assert isinstance(categories, dict)
        assert categories["harassment"] is True
        scores = verdict["category_scores"]
        assert isinstance(scores, dict)
        assert set(scores) == set(categories)

        calls = _gateway_calls(agentic_server, log_start)
        _assert_called(calls, "/v1/moderations")
        if calls:
            assert model_config.model in _billed_models(agentic_server, log_start)


# ---------------------------------------------------------------------------
# Embeddings — /v1/embeddings
# ---------------------------------------------------------------------------


class TestEmbeddings:
    """n8n's Embeddings OpenAI sub-node, embedding a corpus and a query.

    A vector store is the only n8n consumer of an embeddings model, which makes
    this the strongest available assertion: the planted document is retrieved by
    meaning alone, so the document vectors and the query vector must have come
    back in the same space and been usable together.

    Ref: https://developers.openai.com/api/reference/resources/embeddings
         stdapi/routes/openai_embeddings.py:create_embedding
    """

    @pytest.fixture
    def agentic_tool(self) -> AgenticTool:
        """The n8n entry under test; read by the autouse model-identity fixture."""
        return N8N_EMBEDDINGS

    @pytest.mark.parametrize(
        "model_config",
        [
            pytest.param(
                ModelConfig(model="amazon.titan-embed-text-v2:0", timeout=_TIMEOUT),
                id="titan-embed-text-v2",
            )
        ],
    )
    def test_semantic_search_ranks_the_planted_document_first(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """The document naming the planted token outranks five unrelated ones.

        The query never repeats the planted words, so a store filled with
        malformed vectors -- wrong dimension, wrong order, all zeros -- would
        still return three documents but not this one first.

        Ref: stdapi/types/openai_embeddings.py:CreateEmbeddingResponse
        """
        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt="Which component keeps its timing with a quartz crystal?",
            workdir=agentic_workdir,
            test_name=request.node.originalname or request.node.name,
        )
        log_metrics(
            agentic_tool,
            result,
            model_config,
            "test_semantic_search_ranks_the_planted_document_first",
        )
        assert_result(result, config=model_config, contains="score", min_steps=6)

        ranked = n8n_node_items(_record(agentic_workdir), "Query")
        assert len(ranked) == 3, f"expected the top 3 documents, got {len(ranked)}"
        best = ranked[0]["document"]
        assert isinstance(best, dict)
        assert _PLANTED_PHRASE in str(best["pageContent"])
        scores = [float(item["score"]) for item in ranked]  # type: ignore[arg-type]
        assert scores == sorted(scores, reverse=True), f"unranked results: {scores}"
        assert scores[0] > scores[-1], "every document scored identically"

        _assert_called(
            _gateway_calls(agentic_server, log_start),
            "/v1/embeddings",
            model_config.model,
        )


# ---------------------------------------------------------------------------
# Legacy completions — /v1/completions
# ---------------------------------------------------------------------------


class TestLegacyCompletions:
    """n8n's OpenAI Model sub-node, the only client left driving Completions.

    Ref: https://developers.openai.com/api/reference/resources/completions
         stdapi/routes/openai_completions.py:create_completion
    """

    @pytest.fixture
    def agentic_tool(self) -> AgenticTool:
        """The n8n entry under test; read by the autouse model-identity fixture."""
        return N8N_COMPLETIONS

    @pytest.mark.parametrize(
        "model_config",
        [
            pytest.param(
                ModelConfig(model="amazon.nova-micro-v1:0", timeout=_TIMEOUT),
                id="nova-micro",
            ),
            pytest.param(
                # A Claude model rides every agentic suite: it is the family with
                # the most translation-specific behaviour behind it, and the one
                # whose quirks -- signed reasoning, temperature-or-top_p, its own
                # system-block shape -- no other family exercises.
                ModelConfig(
                    model="anthropic.claude-haiku-4-5-20251001-v1:0", timeout=_TIMEOUT
                ),
                id="claude-haiku-4-5",
            ),
        ],
    )
    def test_completes_a_prompt(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """A prompt-completion chain answers over the legacy route.

        The node sends a bare ``prompt`` string rather than a message list, which
        is the shape only this route accepts -- a gateway that silently rerouted
        it to Chat Completions would be caught by the logged path.

        Ref: stdapi/types/openai_completions.py:CompletionCreateParams
        """
        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=("Name the capital city of Japan. Answer with the city name only."),
            workdir=agentic_workdir,
            test_name=request.node.originalname or request.node.name,
        )
        log_metrics(agentic_tool, result, model_config, "test_completes_a_prompt")
        assert_result(result, config=model_config, contains="tokyo", min_steps=2)

        _assert_called(
            _gateway_calls(agentic_server, log_start),
            "/v1/completions",
            model_config.model,
        )


# ---------------------------------------------------------------------------
# Conversational routes — /v1/chat/completions, /v1/responses and
# /anthropic/v1/messages
# ---------------------------------------------------------------------------

#: Prompt every conversational-route test asks, with an unambiguous right answer.
_CAPITAL_PROMPT = "Name the capital city of Japan. Answer with the city name only."

#: Models exercised on every conversational-route test.
#:
#: A Claude model is required: the point of exercising these three routes through
#: n8n -- on top of the coverage pi's provider extension already gives them -- is
#: Claude's reasoning-signature round trip through the node a real n8n user would
#: actually pick. Amazon Nova covers a second, native Converse family so a
#: Claude-only quirk cannot be mistaken for a route bug.
_CHAT_MODEL_CONFIGS = [
    pytest.param(
        ModelConfig(model="amazon.nova-micro-v1:0", timeout=_TIMEOUT), id="nova-micro"
    ),
    pytest.param(
        ModelConfig(model="anthropic.claude-haiku-4-5-20251001-v1:0", timeout=_TIMEOUT),
        id="claude-haiku-4-5",
    ),
]

#: Bedrock Mantle model verified (tests/probes/results/) to accept
#: /v1/chat/completions; other Mantle models in this module reject that route.
_MANTLE_CHAT_COMPLETIONS_MODEL = pytest.param(
    ModelConfig(model="qwen.qwen3-32b", timeout=_TIMEOUT), id="qwen3-32b"
)

#: Bedrock Mantle model verified (tests/probes/results/) to accept /v1/responses;
#: it rejects /v1/chat/completions outright.
_MANTLE_RESPONSES_MODEL = pytest.param(
    ModelConfig(model="openai.gpt-5.6-luna", timeout=_TIMEOUT), id="gpt-5.6-luna"
)

#: Bedrock Mantle model verified (tests/probes/results/) to accept
#: /anthropic/v1/messages; it rejects /v1/chat/completions outright.
_MANTLE_MESSAGES_MODEL = pytest.param(
    ModelConfig(model="google.gemma-4-31b", timeout=_TIMEOUT), id="gemma-4-31b"
)


class TestChatCompletions:
    """n8n's OpenAI node, Message a Model operation, against Chat Completions.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
         stdapi/routes/openai_chat_completions.py:create_chat_completion
    """

    @pytest.fixture
    def agentic_tool(self) -> AgenticTool:
        """The n8n entry under test; read by the autouse model-identity fixture."""
        return N8N_CHAT_COMPLETIONS

    @pytest.mark.parametrize(
        "model_config", [*_CHAT_MODEL_CONFIGS, _MANTLE_CHAT_COMPLETIONS_MODEL]
    )
    def test_completes_a_chat_prompt(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """The node's answer round-trips through ``POST /v1/chat/completions``.

        The node sends a ``messages`` array to this route rather than the bare
        ``prompt`` string the legacy Completions node sends, so a gateway that
        silently rerouted it would be caught by the logged path.

        Ref: stdapi/types/openai_chat_completions.py:CreateChatCompletionParams
        """
        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_CAPITAL_PROMPT,
            workdir=agentic_workdir,
            test_name=request.node.originalname or request.node.name,
        )
        log_metrics(agentic_tool, result, model_config, "test_completes_a_chat_prompt")
        assert_result(result, config=model_config, contains="tokyo", min_steps=2)

        entry = n8n_node_items(_record(agentic_workdir), "Chat")[0]
        message = entry["message"]
        assert isinstance(message, dict)
        assert "tokyo" in str(message["content"]).lower()

        _assert_called(
            _gateway_calls(agentic_server, log_start),
            "/v1/chat/completions",
            model_config.model,
        )


class TestResponses:
    """n8n's OpenAI node, Generate a Model Response operation, against Responses.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/routes/openai_responses.py:create_response
    """

    @pytest.fixture
    def agentic_tool(self) -> AgenticTool:
        """The n8n entry under test; read by the autouse model-identity fixture."""
        return N8N_RESPONSES

    @pytest.mark.parametrize(
        "model_config", [*_CHAT_MODEL_CONFIGS, _MANTLE_RESPONSES_MODEL]
    )
    def test_completes_a_response(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """The node's answer round-trips through ``POST /v1/responses``.

        The node sends its prompt as an ``input`` item list and reads the answer
        back out of ``output[].content[].text``, the Responses-specific shape
        Chat Completions does not have.

        Ref: stdapi/types/openai_responses.py:CreateResponseParams
        """
        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_CAPITAL_PROMPT,
            workdir=agentic_workdir,
            test_name=request.node.originalname or request.node.name,
        )
        log_metrics(agentic_tool, result, model_config, "test_completes_a_response")
        assert_result(result, config=model_config, contains="tokyo", min_steps=2)

        entry = n8n_node_items(_record(agentic_workdir), "Respond")[0]
        output = entry["output"]
        assert isinstance(output, list)
        assert output, "the Responses node produced no output message"
        message = output[0]
        assert isinstance(message, dict)
        content = message["content"]
        assert isinstance(content, list)
        assert content, "the Responses node's message carried no content part"
        part = content[0]
        assert isinstance(part, dict)
        assert "tokyo" in str(part["text"]).lower()

        _assert_called(
            _gateway_calls(agentic_server, log_start),
            "/v1/responses",
            model_config.model,
        )


class TestAnthropicMessages:
    """n8n's Anthropic node, Message a Model operation, against Anthropic Messages.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/routes/anthropic_messages.py:create_message
    """

    @pytest.fixture
    def agentic_tool(self) -> AgenticTool:
        """The n8n entry under test; read by the autouse model-identity fixture."""
        return N8N_MESSAGES

    @pytest.mark.parametrize(
        "model_config", [*_CHAT_MODEL_CONFIGS, _MANTLE_MESSAGES_MODEL]
    )
    def test_completes_a_message(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """The node's answer round-trips through ``POST /anthropic/v1/messages``.

        The node authenticates with the Anthropic ``x-api-key`` header rather than
        the OpenAI nodes' bearer token, over a credential of its own pointed at the
        gateway's ``/anthropic`` prefix -- so this also pins that second credential
        wiring, not just the route.

        Ref: stdapi/types/anthropic_messages.py:CreateMessageParams
        """
        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_CAPITAL_PROMPT,
            workdir=agentic_workdir,
            test_name=request.node.originalname or request.node.name,
        )
        log_metrics(agentic_tool, result, model_config, "test_completes_a_message")
        assert_result(result, config=model_config, contains="tokyo", min_steps=2)

        entry = n8n_node_items(_record(agentic_workdir), "Message")[0]
        assert "tokyo" in str(entry["merged_response"]).lower()

        _assert_called(
            _gateway_calls(agentic_server, log_start),
            "/anthropic/v1/messages",
            model_config.model,
        )


class TestOllamaChat:
    """n8n's Ollama Chat Model node, the lane's only JavaScript Ollama client.

    Every other client on the ``/api/*`` dialect is the official Python library.
    This node builds ``@langchain/ollama``'s ``ChatOllama``, which speaks to the
    same routes through the ``ollama`` **npm** package -- a second, independent
    implementation of the wire format, reading our replies with a different
    parser, in a different language.

    Its credential is also the only one in the lane that carries a bearer token
    for this dialect: the node reads ``apiKey`` off the ``ollamaApi`` credential
    and turns it into an ``Authorization: Bearer`` header, which is what makes a
    proxied Ollama -- this gateway -- reachable at all.

    Ref: https://docs.n8n.io/integrations/builtin/cluster-nodes/sub-nodes/n8n-nodes-langchain.lmchatollama/
         https://docs.ollama.com/api/chat
         stdapi/routes/ollama_chat.py:chat
    """

    @pytest.fixture
    def agentic_tool(self) -> AgenticTool:
        """The n8n entry under test; read by the autouse model-identity fixture."""
        return N8N_OLLAMA_CHAT

    @pytest.mark.parametrize("model_config", _CHAT_MODEL_CONFIGS)
    def test_completes_a_chat_prompt(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """The chain's answer round-trips through ``POST /api/chat``.

        The committed workflow pins the node's whole option block --
        ``temperature``, ``numPredict``, ``numCtx`` and ``keepAlive`` -- so what
        the gateway receives is a real client's full request rather than the
        minimum a test would have written. ``numCtx`` and ``keepAlive`` describe a
        local model server's memory and have no meaning here, and are accepted and
        ignored; a gateway that rejected them would refuse every request this node
        makes, since the node sends them from its defaults.

        Ref: stdapi/types/ollama.py:ChatRequest
             stdapi/models/chat/_adapters/_ollama.py:_apply_options
        """
        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_CAPITAL_PROMPT,
            workdir=agentic_workdir,
            test_name=request.node.originalname or request.node.name,
        )
        log_metrics(agentic_tool, result, model_config, "test_completes_a_chat_prompt")
        assert_result(result, config=model_config, contains="tokyo", min_steps=2)

        entry = n8n_node_items(_record(agentic_workdir), "Chain")[0]
        assert "tokyo" in str(entry["text"]).lower()

        _assert_called(
            _gateway_calls(agentic_server, log_start), "/api/chat", model_config.model
        )


# ---------------------------------------------------------------------------
# Audio — /v1/audio/speech, /v1/audio/transcriptions, /v1/audio/translations
# ---------------------------------------------------------------------------

#: Sentence the speech workflows synthesise.
#:
#: Ordinary words a speech-to-text model transcribes reliably; the assertion is on
#: the gateway's round trip, not on the model's handling of an unusual vocabulary.
_SPOKEN_SENTENCE = "The quarterly audit report is ready for review this morning."

#: French sentence the translation workflow synthesises, with its English content.
#:
#: Translation must return English regardless of the spoken language, so the input
#: has to be non-English for the route to be doing anything at all.
_SPOKEN_FRENCH = "Le rapport trimestriel est prêt pour la réunion de ce matin."


class TestAudio:
    """n8n's Generate Audio, Transcribe a Recording and Translate a Recording.

    Each workflow synthesises its own input through the gateway rather than
    shipping a fixture, so the recording under test is produced by the same
    deployment that then consumes it.

    Ref: https://developers.openai.com/api/reference/resources/audio
         stdapi/routes/openai_audio_speech.py:create_speech
    """

    @pytest.mark.parametrize(
        ("agentic_tool", "model_config"),
        [
            pytest.param(
                N8N_SPEECH,
                ModelConfig(model=_SPEECH_MODEL, timeout=_TIMEOUT),
                id="polly-standard",
            )
        ],
    )
    def test_speech_writes_a_decodable_mp3(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """The synthesised audio reaches disk as a real MPEG stream.

        The node asks for ``response_format=mp3`` and hands the body straight to
        a file node, so a gateway answering with the wrong container -- or with a
        JSON error body the client stores verbatim -- fails on the frame header
        rather than on the byte count alone.

        Ref: stdapi/routes/openai_audio_speech.py:create_speech
        """
        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_SPOKEN_SENTENCE,
            workdir=agentic_workdir,
            test_name=request.node.originalname or request.node.name,
        )
        log_metrics(
            agentic_tool, result, model_config, "test_speech_writes_a_decodable_mp3"
        )
        assert_result(result, config=model_config, contains="audio/", min_steps=3)

        audio = _produced_file(agentic_workdir, "speech.mp3")
        # An MPEG audio stream starts either with an ID3 tag or with a frame
        # sync word; anything else is not the container that was requested.
        assert audio.startswith(b"ID3") or audio[0] == 0xFF, (
            f"not an MPEG stream: {audio[:8]!r}"
        )

        _assert_called(
            _gateway_calls(agentic_server, log_start),
            "/v1/audio/speech",
            model_config.model,
        )

    @pytest.mark.parametrize(
        ("agentic_tool", "model_config"),
        [
            pytest.param(
                N8N_TRANSCRIPTIONS,
                ModelConfig(
                    model=_TRANSCRIBE_MODEL,
                    timeout=_TIMEOUT,
                    extra_env={"SPEECH_MODEL": _SPEECH_MODEL},
                ),
                id="transcribe",
            )
        ],
    )
    def test_transcription_recovers_the_synthesised_sentence(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """Speech synthesised by the gateway transcribes back to its own words.

        A closed loop: the same deployment produces the audio and reads it, so a
        failure is in one of the two routes rather than in a fixture recording
        that may simply be hard to transcribe.

        Ref: stdapi/routes/openai_audio_transcriptions.py:create_transcription
        """
        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_SPOKEN_SENTENCE,
            workdir=agentic_workdir,
            test_name=request.node.originalname or request.node.name,
        )
        log_metrics(
            agentic_tool,
            result,
            model_config,
            "test_transcription_recovers_the_synthesised_sentence",
        )
        assert_result(result, config=model_config, contains="audit", min_steps=3)

        transcript = str(
            n8n_node_items(_record(agentic_workdir), "Transcribe")[0]["text"]
        )
        assert "quarterly" in transcript.lower()

        calls = _gateway_calls(agentic_server, log_start)
        _assert_called(calls, "/v1/audio/speech", _SPEECH_MODEL)
        _assert_called(calls, "/v1/audio/transcriptions", model_config.model)

    @pytest.mark.parametrize(
        ("agentic_tool", "model_config"),
        [
            pytest.param(
                N8N_TRANSLATIONS,
                ModelConfig(
                    model=_TRANSCRIBE_MODEL,
                    timeout=_TIMEOUT,
                    extra_env={"SPEECH_MODEL": _SPEECH_MODEL},
                ),
                id="transcribe",
            )
        ],
    )
    def test_translation_returns_english_for_french_speech(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """French speech comes back as English text, not as a French transcript.

        This is the whole difference between the translation route and the
        transcription route, and the only assertion that separates them: a
        gateway wiring translations to plain transcription returns the French
        words and fails here.

        Ref: stdapi/routes/openai_audio_translations.py:create_translation
        """
        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_SPOKEN_FRENCH,
            workdir=agentic_workdir,
            test_name=request.node.originalname or request.node.name,
        )
        log_metrics(
            agentic_tool,
            result,
            model_config,
            "test_translation_returns_english_for_french_speech",
        )
        assert_result(result, config=model_config, min_steps=3)

        translated = str(
            n8n_node_items(_record(agentic_workdir), "Translate")[0]["text"]
        ).lower()
        assert any(word in translated for word in ("report", "quarterly", "meeting")), (
            f"translation is not English: {translated!r}"
        )
        assert "rapport" not in translated, f"translation stayed French: {translated!r}"

        calls = _gateway_calls(agentic_server, log_start)
        _assert_called(calls, "/v1/audio/speech", _SPEECH_MODEL)
        _assert_called(calls, "/v1/audio/translations", model_config.model)


# ---------------------------------------------------------------------------
# Files — /v1/files
# ---------------------------------------------------------------------------


@pytest.mark.xdist_group("openai_files")
class TestFiles:
    """n8n's Upload, List and Delete File operations, in that order.

    The three operations only mean anything together: an upload nobody can list
    and a delete nobody can verify each pass in isolation. The workflow chains
    them so the uploaded id has to survive into the listing and back out again.

    Ref: https://developers.openai.com/api/reference/resources/files
         stdapi/routes/openai_files.py:upload
    """

    @pytest.fixture
    def agentic_tool(self) -> AgenticTool:
        """The n8n entry under test; read by the autouse model-identity fixture."""
        return N8N_FILES

    @pytest.mark.parametrize(
        "model_config",
        [
            # The Files API stores bytes and invokes no model, so there is no
            # model id for the shared identity check to hold the run to; the
            # route assertions below are what carry this test.
            pytest.param(
                ModelConfig(
                    model="", timeout=_TIMEOUT, extra_env={"PURPOSE": _FILE_PURPOSE}
                ),
                id="no-model",
            )
        ],
    )
    def test_uploads_then_lists_then_deletes(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """The uploaded file appears in the listing and deletes cleanly.

        Ref: stdapi/types/openai_files.py:FileObject
        """
        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt="Uploaded by the stdapi.ai agentic lane; safe to delete.",
            workdir=agentic_workdir,
            test_name=request.node.originalname or request.node.name,
        )
        log_metrics(
            agentic_tool, result, model_config, "test_uploads_then_lists_then_deletes"
        )
        assert_result(result, config=model_config, contains="deleted", min_steps=6)

        record = _record(agentic_workdir)
        uploaded = n8n_node_items(record, "Upload")[0]
        assert str(uploaded["id"]).startswith("file-")
        assert uploaded["purpose"] == _FILE_PURPOSE
        assert uploaded["filename"] == "stdapi-agentic-note.txt"

        listed = {str(item["id"]) for item in n8n_node_items(record, "List")}
        assert uploaded["id"] in listed, (
            f"{uploaded['id']} is missing from a listing of {len(listed)} files"
        )

        deleted = n8n_node_items(record, "Delete")[0]
        assert deleted["id"] == uploaded["id"]
        assert deleted["deleted"] is True

        calls = _gateway_calls(agentic_server, log_start)
        paths = [path for path, _ in calls]
        _assert_called(calls, "/v1/files")
        if calls:
            assert f"/v1/files/{uploaded['id']}" in paths, (
                f"the uploaded file was never deleted by id: {paths}"
            )


# ---------------------------------------------------------------------------
# Images — /v1/images/generations and /v1/images/edits
# ---------------------------------------------------------------------------


@pytest.mark.expensive
class TestImages:
    """n8n's Generate an Image and Edit an Image operations.

    Ref: https://developers.openai.com/api/reference/resources/images
         stdapi/routes/openai_images_generations.py:create_image
    """

    @pytest.mark.parametrize(
        ("agentic_tool", "model_config"),
        [
            pytest.param(
                N8N_IMAGES_GENERATIONS,
                ModelConfig(
                    model="stability.stable-image-core-v1:1",
                    timeout=_TIMEOUT,
                    extra_env={
                        "SIZE": smallest_image_size("stability.stable-image-core-v1:1")
                    },
                ),
                id="stable-image-core",
            )
        ],
    )
    def test_generation_writes_a_decodable_png(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """The node's base64 image decodes to a real PNG raster on disk.

        The node asks for ``response_format=b64_json`` and decodes it itself, so
        a gateway returning a URL where base64 was requested, or base64 of
        something that is not an image, fails on the signature.

        Ref: stdapi/types/openai_images.py:ImagesResponse
        """
        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt="A single red maple leaf on flat white paper, studio lighting",
            workdir=agentic_workdir,
            test_name=request.node.originalname or request.node.name,
        )
        log_metrics(
            agentic_tool, result, model_config, "test_generation_writes_a_decodable_png"
        )
        assert_result(result, config=model_config, contains="image/", min_steps=3)

        assert _produced_file(agentic_workdir, "generated.png").startswith(_PNG_MAGIC)

        _assert_called(
            _gateway_calls(agentic_server, log_start),
            "/v1/images/generations",
            model_config.model,
        )

    @pytest.mark.parametrize(
        ("agentic_tool", "model_config"),
        [
            pytest.param(
                N8N_IMAGES_EDITS,
                ModelConfig(
                    model="stability.stable-image-control-structure-v1:0",
                    timeout=_TIMEOUT,
                    extra_env={
                        "SIZE": smallest_image_size(
                            "stability.stable-image-control-structure-v1:0"
                        )
                    },
                ),
                id="stable-image-control-structure",
            )
        ],
    )
    def test_edit_returns_a_retrievable_image(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """A photograph uploaded as multipart comes back as a downloadable image.

        The node sends no ``response_format`` for a non-OpenAI model, so this
        also pins the route's default: the answer has to carry a URL, and that
        URL has to serve the image it claims to.

        Ref: stdapi/routes/openai_images_edits.py:edit_image
        """
        source_dir = n8n_files_dir(agentic_workdir)
        source_dir.mkdir(exist_ok=True)
        shutil.copyfile(_EDIT_SOURCE, source_dir / "source.jpg")

        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt="The same scene rendered as a watercolour painting",
            workdir=agentic_workdir,
            test_name=request.node.originalname or request.node.name,
        )
        log_metrics(
            agentic_tool, result, model_config, "test_edit_returns_a_retrievable_image"
        )
        assert_result(result, config=model_config, contains="http", min_steps=3)

        url = str(n8n_node_items(_record(agentic_workdir), "Edit")[0]["url"])
        downloaded = httpx.get(url, timeout=60.0)
        assert downloaded.status_code == 200, f"edited image URL returned {url}"
        assert len(downloaded.content) > _MIN_MEDIA_BYTES

        _assert_called(
            _gateway_calls(agentic_server, log_start),
            "/v1/images/edits",
            model_config.model,
        )


# ---------------------------------------------------------------------------
# Videos — /v1/videos
# ---------------------------------------------------------------------------


@pytest.mark.video
@pytest.mark.expensive
@pytest.mark.slow
class TestVideos:
    """n8n's Generate a Video operation, job polling included.

    The node is the lane's only client that drives the three-call video flow --
    create, poll until ``completed``, download the content -- so it covers the
    job lifecycle no single-request test reaches.

    Ref: https://developers.openai.com/api/reference/resources/videos
         stdapi/routes/openai_videos.py:create_video
    """

    @pytest.fixture
    def agentic_tool(self) -> AgenticTool:
        """The n8n entry under test; read by the autouse model-identity fixture."""
        return N8N_VIDEOS

    @pytest.mark.parametrize(
        "model_config",
        [
            pytest.param(
                ModelConfig(
                    model="luma.ray-v2:0",
                    timeout=_VIDEO_TIMEOUT,
                    # The shortest clip the model produces, at its smallest
                    # landscape resolution: video is billed per second of output.
                    extra_env={"SECONDS": "5", "SIZE": "1280x720"},
                ),
                id="luma-ray-2",
            )
        ],
    )
    def test_generation_polls_then_downloads_a_clip(
        self,
        request: pytest.FixtureRequest,
        agentic_tool: AgenticTool,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """The finished job's content downloads as an ISO base media file.

        Ref: stdapi/types/openai_videos.py:Video
        """
        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=agentic_tool,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt="A slow aerial pan over a calm lake at sunrise",
            workdir=agentic_workdir,
            test_name=request.node.originalname or request.node.name,
        )
        log_metrics(
            agentic_tool,
            result,
            model_config,
            "test_generation_polls_then_downloads_a_clip",
        )
        assert_result(result, config=model_config, contains="video/", min_steps=3)

        clip = _produced_file(agentic_workdir, "clip.mp4")
        # Every ISO base media file names its brand in the first box.
        assert clip[4:8] == b"ftyp", f"not an MP4 container: {clip[:16]!r}"

        calls = _gateway_calls(agentic_server, log_start)
        paths = [path for path, _ in calls]
        _assert_called(calls, "/v1/videos", model_config.model)
        if calls:
            assert any(path.endswith("/content") for path in paths), (
                f"the finished clip was never downloaded: {paths}"
            )
            # Create, at least one status poll, then the content download.
            assert sum(path.startswith("/v1/videos") for path in paths) >= 3, (
                f"the job was downloaded without being polled: {paths}"
            )
