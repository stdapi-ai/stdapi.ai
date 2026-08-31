"""The official ``ollama`` Python client driven against a real gateway process.

This is the lane's only client on the Ollama ``/api/*`` dialect, and it is the
dialect's own reference implementation: the library every Ollama integration is
built on, pointed at ``agentic_server`` over a real socket with a real bearer
token.

Its **pydantic models are the assertion**, the argument that earns the ``ollama``
client its place here. ``ListResponse.Model.modified_at`` and ``ShowResponse.modified_at``
are typed ``datetime``, ``size`` is a ``ByteSize``, and ``model_info`` reaches
the client through a field alias (``ShowResponse.modelinfo``) -- so a shape the
gateway gets wrong surfaces as a ``ValidationError`` in the client rather than
as a key some future test forgot to look at.

What only a real gateway process proves, on top of what ``tests/test_ollama_*``
already assert against the ASGI application:

- **NDJSON over a real HTTP/1.1 connection.** The unit lane hands the client an
  in-process transport, which never chunks a body; here the reference parser
  consumes the wire framing the gateway actually writes.
- **Bearer authentication on every route**, including the ones with no body.
- **The whole live catalogue deserialises**, not one model a fixture named: every
  entry of ``/api/tags`` is parsed by the client's own types.
- **The model-management split**, which no other client in this lane can express:
  ``pull`` is accepted and the four verbs that would write to a model store are
  refused, as ``ollama.ResponseError``.

Requires ``--agentic``, podman, and Bedrock credentials.

Ref: https://github.com/ollama/ollama-python
     https://docs.ollama.com/api/chat
     https://docs.ollama.com/openapi.yaml
     stdapi/routes/ollama_models.py
     stdapi/routes/ollama_model_management.py
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import TYPE_CHECKING

import ollama
import pytest
from pydantic import ByteSize

from ._runner import ModelConfig
from ._tools import AgenticTool

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from ._server import AgenticServer
    from ._tools import AgenticResult, Command, Invocation

pytestmark = pytest.mark.agentic


def _unused_build(invocation: Invocation) -> Command:
    """Never called: this module drives no CLI, only the shared identity check needs a tool."""
    raise NotImplementedError


def _unused_parse(stdout: str) -> AgenticResult:
    """Never called: see :func:`_unused_build`."""
    raise NotImplementedError


def _unused_prepare_workdir(invocation: Invocation) -> None:
    """Never called: see :func:`_unused_build`."""
    raise NotImplementedError


#: The official Ollama client, against the gateway's Ollama-dialect routes.
TOOL = AgenticTool(
    id="ollama-python",
    npm_package=None,
    binary="python",
    route="/api",
    metrics_prefix="OL-METRICS",
    build=_unused_build,
    parse=_unused_parse,
    prepare_workdir=_unused_prepare_workdir,
    # The client sends no per-run identifier the gateway records, so its requests
    # can only be attributed positionally.
    attributes_sessions=False,
)

#: Seconds one request may take; a Bedrock call runs behind each.
_TIMEOUT = 600.0

#: Cheapest model serving the chat and generate routes.
#:
#: The same one ``tests/conftest.py:OLLAMA_MODEL_MAPPINGS`` names for the unit
#: lane, so a divergence between the two lanes is never a difference of model.
_CHAT_MODEL = "amazon.nova-micro-v1:0"

#: Second chat family, so a Converse-native quirk cannot pass for a dialect bug.
_CLAUDE_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"

#: Cheapest model serving the embed routes; embeddings are not a chat capability.
_EMBEDDING_MODEL = "amazon.titan-embed-text-v2:0"

#: Chat-route parametrization, read by the lane's autouse identity check.
_CHAT_MODEL_CONFIG = pytest.param(ModelConfig(model=_CHAT_MODEL), id="nova-micro")

#: Second chat-route parametrization, on the Claude family.
_CLAUDE_MODEL_CONFIG = pytest.param(
    ModelConfig(model=_CLAUDE_MODEL), id="claude-haiku-4-5"
)

#: Embedding-route parametrization.
_EMBEDDING_MODEL_CONFIG = pytest.param(
    ModelConfig(model=_EMBEDDING_MODEL), id="titan-embed-text-v2"
)

#: Prompt whose one-word answer is cheap to assert without matching model prose.
_PLANET_PROMPT = (
    "Name the largest planet in the solar system. Answer with just its name."
)

#: Value only the tool call can reveal, so the model cannot answer without calling it.
_MAGIC_NUMBER = 4817

#: Tool the chat round trip declares, in Ollama's own tool shape.
_MAGIC_TOOL = {
    "type": "function",
    "function": {
        "name": "magic_number",
        "description": "Look up the registered magic number for a key.",
        "parameters": {
            "type": "object",
            "properties": {"key": {"type": "string", "description": "Lookup key."}},
            "required": ["key"],
        },
    },
}

#: Shape of a digest the gateway derives from a model name: a SHA-256 hex string.
_DIGEST = re.compile(r"^[0-9a-f]{64}$")

#: Shape of the API version clients gate their feature set on.
_VERSION = re.compile(r"^\d+\.\d+\.\d+")

#: Name a refused model-management verb must never create.
_UNWRITABLE_MODEL = "stdapi-agentic-copy"


@pytest.fixture(scope="module")
def ollama_client(agentic_server: AgenticServer) -> Iterator[ollama.Client]:
    """The official client, pointed at the lane's gateway and carrying its key.

    Module-scoped: the client is a connection pool, and every test here spends
    its time in Bedrock rather than in the handshake.

    Yields:
        A client bound to the gateway under test.
    """
    with ollama.Client(
        host=agentic_server.base_url,
        headers={"Authorization": f"Bearer {agentic_server.api_key}"},
        timeout=_TIMEOUT,
    ) as client:
        yield client


class TestChat:
    """``POST /api/chat``, buffered and streamed, through the reference client.

    Ref: https://docs.ollama.com/api/chat
         stdapi/routes/ollama_chat.py:chat
    """

    @pytest.mark.parametrize("model_config", [_CHAT_MODEL_CONFIG, _CLAUDE_MODEL_CONFIG])
    def test_chat_returns_a_typed_response(
        self,
        model_config: ModelConfig,
        ollama_client: ollama.Client,
        agentic_server: AgenticServer,
    ) -> None:
        """The answer parses as a ``ChatResponse`` and carries the model's real content.

        The client validates the whole envelope, so ``done``, ``done_reason`` and
        the token counts are asserted by the parse itself; what is left to check
        is that the content is the model's rather than an empty well-formed reply.

        Ref: https://docs.ollama.com/api/chat
             stdapi/types/ollama.py:ChatResponse
        """
        response = ollama_client.chat(
            model=model_config.model,
            messages=[{"role": "user", "content": _PLANET_PROMPT}],
            options={"temperature": 0, "num_predict": 64},
            keep_alive="5m",
            stream=False,
        )
        assert isinstance(response, ollama.ChatResponse)
        assert response.model == model_config.model
        assert response.done is True
        assert "jupiter" in (response.message.content or "").lower()

    @pytest.mark.parametrize("model_config", [_CHAT_MODEL_CONFIG])
    def test_chat_streams_ndjson_over_a_real_connection(
        self,
        model_config: ModelConfig,
        ollama_client: ollama.Client,
        agentic_server: AgenticServer,
    ) -> None:
        """The reference parser reassembles the gateway's NDJSON off the wire.

        The in-process lane hands the client an ASGI transport that never frames
        a body, so this is the only place the client's line-oriented parser meets
        the chunked encoding the gateway actually writes. More than one part is
        required: a single part means nothing streamed.

        Ref: https://docs.ollama.com/api/streaming
             stdapi/api_providers/ollama.py
        """
        parts = list(
            ollama_client.chat(
                model=model_config.model,
                messages=[{"role": "user", "content": _PLANET_PROMPT}],
                options={"temperature": 0, "num_predict": 64},
                stream=True,
            )
        )
        assert len(parts) > 1, "a single part means the gateway did not stream"
        assert parts[-1].done is True
        assert all(part.done is False for part in parts[:-1])
        text = "".join(part.message.content or "" for part in parts)
        assert "jupiter" in text.lower()

    @pytest.mark.parametrize("model_config", [_CHAT_MODEL_CONFIG])
    def test_chat_completes_a_tool_round_trip(
        self,
        model_config: ModelConfig,
        ollama_client: ollama.Client,
        agentic_server: AgenticServer,
    ) -> None:
        """The model calls the declared tool and reports the value it returned.

        Ollama's tool calls carry no identifier, so the result is correlated by
        ``tool_name`` alone -- the only shape an Ollama client can send. The final
        answer has to name a number that exists nowhere in the conversation until
        the tool result is replayed.

        Ref: https://docs.ollama.com/api/chat#tools
             stdapi/models/chat/_adapters/_ollama.py
        """
        messages: list[ollama.Message] = [
            ollama.Message(
                role="user",
                content=(
                    "Call the magic_number tool with key='zephyr', then state the "
                    "result in your answer."
                ),
            )
        ]
        first = ollama_client.chat(
            model=model_config.model,
            messages=messages,
            tools=[_MAGIC_TOOL],
            stream=False,
        )
        calls = first.message.tool_calls
        assert calls, "the model never called magic_number"
        assert calls[0].function.name == "magic_number"

        messages.append(first.message)
        messages.append(
            ollama.Message(
                role="tool", tool_name="magic_number", content=str(_MAGIC_NUMBER)
            )
        )
        final = ollama_client.chat(
            model=model_config.model,
            messages=messages,
            tools=[_MAGIC_TOOL],
            stream=False,
        )
        assert str(_MAGIC_NUMBER) in (final.message.content or "")


class TestGenerate:
    """``POST /api/generate``, the single-prompt route Chat Completions has no twin for.

    Ref: https://docs.ollama.com/api/generate
         stdapi/routes/ollama_generate.py:generate
    """

    @pytest.mark.parametrize("model_config", [_CHAT_MODEL_CONFIG])
    def test_generate_returns_a_typed_response(
        self,
        model_config: ModelConfig,
        ollama_client: ollama.Client,
        agentic_server: AgenticServer,
    ) -> None:
        """The answer parses as a ``GenerateResponse`` carrying the model's text.

        ``system`` and ``options`` travel as the route's own fields rather than
        as a message list, which is the whole difference from ``/api/chat``.

        Ref: https://docs.ollama.com/api/generate
             stdapi/types/ollama.py:GenerateResponse
        """
        response = ollama_client.generate(
            model=model_config.model,
            prompt=_PLANET_PROMPT,
            system="Answer with a single word.",
            options={"temperature": 0, "num_predict": 64},
            keep_alive="5m",
            stream=False,
        )
        assert isinstance(response, ollama.GenerateResponse)
        assert response.model == model_config.model
        assert response.done is True
        assert "jupiter" in (response.response or "").lower()


class TestEmbed:
    """Both embedding routes: the current one and the legacy single-vector one.

    Ref: https://docs.ollama.com/api/embed
         stdapi/routes/ollama_embed.py
    """

    @pytest.mark.parametrize("model_config", [_EMBEDDING_MODEL_CONFIG])
    def test_embed_returns_one_vector_per_input(
        self,
        model_config: ModelConfig,
        ollama_client: ollama.Client,
        agentic_server: AgenticServer,
    ) -> None:
        """A batch of two inputs comes back as two non-zero vectors, in order.

        Ref: https://docs.ollama.com/api/embed
             stdapi/types/ollama.py:EmbedResponse
        """
        response = ollama_client.embed(
            model=model_config.model,
            input=["stdapi.ai gateway integration test", "a second, different text"],
        )
        assert isinstance(response, ollama.EmbedResponse)
        assert len(response.embeddings) == 2
        assert all(any(vector) for vector in response.embeddings)
        assert response.embeddings[0] != response.embeddings[1]

    @pytest.mark.parametrize("model_config", [_EMBEDDING_MODEL_CONFIG])
    def test_legacy_embeddings_returns_a_flat_vector(
        self,
        model_config: ModelConfig,
        ollama_client: ollama.Client,
        agentic_server: AgenticServer,
    ) -> None:
        """``/api/embeddings`` answers the flat shape its own response model demands.

        The deprecated route returns ``embedding`` rather than ``embeddings``, and
        the client has a separate type for it -- so a gateway answering the modern
        shape here would fail the parse rather than pass unnoticed.

        Ref: https://docs.ollama.com/openapi.yaml (EmbeddingsResponse)
             stdapi/routes/ollama_embed.py:embeddings
        """
        response = ollama_client.embeddings(
            model=model_config.model, prompt="stdapi.ai gateway integration test"
        )
        assert isinstance(response, ollama.EmbeddingsResponse)
        assert response.embedding
        assert any(response.embedding)


class TestModelDiscovery:
    """``/api/tags``, ``/api/show``, ``/api/ps`` and ``/api/version``.

    None of these resolves a model the lane's identity check can attribute, so
    none is parametrized on ``model_config``: what they assert is the client's own
    deserialisation of the live catalogue.

    Ref: https://docs.ollama.com/api/tags
         stdapi/routes/ollama_models.py
    """

    def test_every_catalogue_entry_deserialises(
        self, ollama_client: ollama.Client, agentic_server: AgenticServer
    ) -> None:
        """The whole live model list parses into the client's own types.

        ``list()`` raising is the real assertion: ``modified_at`` is a ``datetime``
        and ``size`` a ``ByteSize`` in the client, so the gateway's epoch fallback
        for an unknown release date and its always-zero size are both parsed here
        rather than merely returned. Checking the types back afterwards is what
        proves the fields arrived at all instead of defaulting to ``None``.

        Ref: https://docs.ollama.com/api/tags
             stdapi/routes/ollama_models.py:format_model_summary
        """
        listed = ollama_client.list()
        assert isinstance(listed, ollama.ListResponse)
        assert listed.models, "the gateway advertised no model at all"
        for entry in listed.models:
            assert entry.model, listed
            assert isinstance(entry.modified_at, datetime), entry
            assert isinstance(entry.size, ByteSize), entry
            assert entry.digest is not None, entry
            assert _DIGEST.match(entry.digest), entry
            assert entry.details is not None, entry
            assert entry.details.family, entry

    def test_show_carries_model_info_through_its_field_alias(
        self, ollama_client: ollama.Client, agentic_server: AgenticServer
    ) -> None:
        """``model_info`` arrives as a mapping under the client's aliased field.

        The client names the field ``modelinfo`` and reads it from ``model_info``
        on the wire, so a gateway sending ``null`` -- or omitting the key -- leaves
        it ``None`` here. An empty mapping is what a hosted model has to report,
        and this is the only place that convention is observable to a client.

        Ref: https://docs.ollama.com/openapi.yaml (ShowResponse)
             stdapi/types/ollama.py:ShowResponse
        """
        shown = ollama_client.show(_CHAT_MODEL)
        assert isinstance(shown, ollama.ShowResponse)
        assert shown.modelinfo == {}
        assert isinstance(shown.modified_at, datetime)
        assert shown.capabilities is not None
        assert "completion" in shown.capabilities
        assert shown.details is not None
        assert shown.details.family

    def test_ps_reports_nothing_resident(
        self, ollama_client: ollama.Client, agentic_server: AgenticServer
    ) -> None:
        """No model is ever loaded, and the empty list still parses.

        A client polls this to decide whether to warm a model up; an entry here
        would make it wait for an unload that never comes.

        Ref: https://docs.ollama.com/api/ps
             stdapi/routes/ollama_models.py:ps
        """
        running = ollama_client.ps()
        assert isinstance(running, ollama.ProcessResponse)
        assert list(running.models) == []

    def test_version_reports_a_compatibility_claim(
        self, ollama_client: ollama.Client, agentic_server: AgenticServer
    ) -> None:
        """``/api/version`` answers a dotted version clients can gate features on.

        The client has no method for this route, so its own HTTP client is
        borrowed rather than a second one built: that keeps the request on the
        same connection, headers and credentials as every other call here.
        Open WebUI is the client that actually refuses a connection whose version
        it cannot read -- see ``test_open_webui.py``.

        Ref: https://docs.ollama.com/openapi.yaml (/api/version)
             stdapi/routes/ollama_models.py:version
        """
        response = ollama_client._client.get("/api/version")  # noqa: SLF001
        assert response.status_code == 200, response.text[:500]
        reported = response.json()["version"]
        assert _VERSION.match(reported), reported


class TestModelManagement:
    """The one verb whose post-condition can be met, and the four that cannot.

    Nothing else in this lane has a method for these routes, so the gateway's
    accept-versus-refuse decision is otherwise unvalidated by any real client.

    Ref: https://docs.ollama.com/api/pull
         stdapi/routes/ollama_model_management.py
    """

    @pytest.mark.parametrize("model_config", [_CHAT_MODEL_CONFIG])
    def test_pull_reports_success_without_transferring_anything(
        self,
        model_config: ModelConfig,
        ollama_client: ollama.Client,
        agentic_server: AgenticServer,
    ) -> None:
        """A catalogued model is already available, so the pull succeeds at once.

        Several Ollama clients pull before their first chat and abort on anything
        but a success status, so this is a precondition for those clients working
        at all rather than a route with a caller of its own.

        Ref: https://docs.ollama.com/api/pull
             stdapi/routes/ollama_model_management.py:pull
        """
        response = ollama_client.pull(model_config.model, stream=False)
        assert isinstance(response, ollama.ProgressResponse)
        assert response.status == "success"

    @pytest.mark.parametrize("model_config", [_CHAT_MODEL_CONFIG])
    def test_pull_streams_its_terminal_status(
        self,
        model_config: ModelConfig,
        ollama_client: ollama.Client,
        agentic_server: AgenticServer,
    ) -> None:
        """The streamed pull ends on the same success status, framed as NDJSON.

        A client that pulls with progress reporting reads this stream, and stops
        on its terminal event; an unterminated one leaves it waiting forever.

        Ref: https://docs.ollama.com/openapi.yaml (StatusEvent)
             stdapi/routes/ollama_model_management.py:pull
        """
        events = list(ollama_client.pull(model_config.model, stream=True))
        assert events, "the streamed pull produced no event"
        assert events[-1].status == "success"

    def test_delete_is_refused(
        self, ollama_client: ollama.Client, agentic_server: AgenticServer
    ) -> None:
        """Deleting a hosted model is refused rather than reported as done.

        Reporting success would tell the caller a model went away when it did
        not, and the next listing would contradict it.

        Ref: https://docs.ollama.com/api/delete
             stdapi/routes/ollama_model_management.py:delete
        """
        self._assert_refused(lambda: ollama_client.delete(_CHAT_MODEL))

    def test_copy_is_refused(
        self, ollama_client: ollama.Client, agentic_server: AgenticServer
    ) -> None:
        """Copying a model is refused: there is no model store to copy into.

        Ref: https://docs.ollama.com/api/copy
             stdapi/routes/ollama_model_management.py:copy
        """
        self._assert_refused(lambda: ollama_client.copy(_CHAT_MODEL, _UNWRITABLE_MODEL))

    def test_create_is_refused(
        self, ollama_client: ollama.Client, agentic_server: AgenticServer
    ) -> None:
        """Creating a model is refused, and the refusal is not a stream of progress.

        Ref: https://docs.ollama.com/api/create
             stdapi/routes/ollama_model_management.py:create
        """
        self._assert_refused(
            lambda: ollama_client.create(
                _UNWRITABLE_MODEL, from_=_CHAT_MODEL, stream=False
            )
        )

    def test_push_is_refused(
        self, ollama_client: ollama.Client, agentic_server: AgenticServer
    ) -> None:
        """Publishing a model is refused: this server publishes none.

        Ref: https://docs.ollama.com/api/push
             stdapi/routes/ollama_model_management.py:push
        """
        self._assert_refused(lambda: ollama_client.push(_CHAT_MODEL, stream=False))

    @staticmethod
    def _assert_refused(call: Callable[[], object]) -> None:
        """Assert *call* fails as a clean, actionable Ollama error.

        A bare ``ResponseError`` would also be raised by a 404 or a 500, so the
        status and the message both have to be checked: what the gateway owes the
        caller is a refusal they can act on, not merely a failure.

        Args:
            call: Zero-argument callable performing the refused operation.
        """
        with pytest.raises(ollama.ResponseError) as refusal:
            call()
        assert refusal.value.status_code == 400, refusal.value
        assert "does not store models" in refusal.value.error, refusal.value
