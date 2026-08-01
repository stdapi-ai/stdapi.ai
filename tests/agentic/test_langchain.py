"""langchain-openai and langchain-anthropic driven against the gateway.

langchain-openai is the library's own OpenAI-compatible client, so it is the
closest thing in this lane to a second, independent implementation of the
``/v1/chat/completions`` and ``/v1/embeddings`` wire formats -- useful in its own
right, but also the vehicle for a real client-side trap: ``OpenAIEmbeddings``
tokenizes its input with ``tiktoken`` by default and sends token arrays, which
this backend explicitly rejects rather than silently embedding a different
artifact than the caller asked for (see the Ref block). langchain-anthropic adds
a second wire format, ``ChatAnthropic``, pointed at the gateway's ``/anthropic``
route through its documented ``base_url`` constructor argument.

Requires --agentic, podman, and Bedrock credentials.

Ref: https://docs.langchain.com/oss/python/integrations/chat/openai
     https://docs.langchain.com/oss/python/integrations/chat/anthropic
     https://docs.langchain.com/oss/python/integrations/text_embedding/openai
     stdapi/types/openai_embeddings.py:64-112 (EmbeddingCreateParams._unsupported)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import BadRequestError
from pydantic import BaseModel, Field, SecretStr

from ._runner import ModelConfig
from ._tools import AgenticTool

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

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


#: langchain-openai, against the gateway's OpenAI-compatible chat and embeddings routes.
_TOOL_OPENAI = AgenticTool(
    id="langchain-openai",
    npm_package=None,
    binary="python",
    route="/v1",
    metrics_prefix="LC-METRICS",
    build=_unused_build,
    parse=_unused_parse,
    prepare_workdir=_unused_prepare_workdir,
    # langchain-openai sends no per-run identifier the gateway records, so its
    # requests can only be attributed positionally.
    attributes_sessions=False,
)

#: langchain-anthropic, against the gateway's Anthropic Messages route.
_TOOL_ANTHROPIC = AgenticTool(
    id="langchain-anthropic",
    npm_package=None,
    binary="python",
    route="/anthropic",
    metrics_prefix="LC-METRICS",
    build=_unused_build,
    parse=_unused_parse,
    prepare_workdir=_unused_prepare_workdir,
    attributes_sessions=False,
)


@pytest.fixture
def agentic_tool() -> AgenticTool:
    """The langchain-openai entry; overridden by :class:`TestChatAnthropic`."""
    return _TOOL_OPENAI


#: Models exercised on every chat test in this module, including `TestChatAnthropic`.
_MODEL_CONFIGS = [
    pytest.param(
        ModelConfig(model="anthropic.claude-haiku-4-5-20251001-v1:0"),
        id="claude-haiku-4-5",
    ),
    pytest.param(ModelConfig(model="amazon.nova-2-lite-v1:0"), id="nova-2-lite"),
]

#: `_MODEL_CONFIGS` plus a Bedrock Mantle model, for the `ChatOpenAI` classes only.
#:
#: qwen.qwen3-32b is verified (tests/probes/results/) to accept
#: /v1/chat/completions with tool calls, parallel tool calls and JSON mode, which
#: is what `TestToolCalling` and `TestStructuredOutput` need; it is not added to
#: `_MODEL_CONFIGS` itself because that list also drives `TestChatAnthropic` over
#: /anthropic, a route this model was not verified against.
_CHAT_COMPLETIONS_MODEL_CONFIGS = [
    *_MODEL_CONFIGS,
    pytest.param(ModelConfig(model="qwen.qwen3-32b"), id="qwen3-32b"),
]

#: Models whose backend decodes a JSON schema, which `with_structured_output` needs.
#:
#: `json_schema` structured output is model-specific (see the request-field table
#: in `docs/api_openai_chat_completions.md`): Converse serves it through
#: `outputConfig`, which only some model families implement.
_STRUCTURED_OUTPUT_MODEL_CONFIGS = [
    pytest.param(
        ModelConfig(model="anthropic.claude-haiku-4-5-20251001-v1:0"),
        id="claude-haiku-4-5",
    ),
    pytest.param(ModelConfig(model="qwen.qwen3-32b"), id="qwen3-32b"),
]

#: Model whose backend has no schema decoder, used for the refusal case.
_NO_STRUCTURED_OUTPUT_MODEL_CONFIG = pytest.param(
    ModelConfig(model="amazon.nova-2-lite-v1:0"), id="nova-2-lite"
)

#: Small Bedrock-native embedding model; embeddings are not a Claude capability.
_EMBEDDING_MODEL_CONFIG = pytest.param(
    ModelConfig(model="amazon.titan-embed-text-v2:0"), id="titan-embed-text-v2"
)

#: Prompt whose one-word answer is cheap to assert on without matching model prose.
_PLANET_PROMPT = (
    "Name the largest planet in the solar system. Answer with just its name."
)


def _chat_model(server: AgenticServer, config: ModelConfig) -> ChatOpenAI:
    """Build a `ChatOpenAI` pointed at the gateway's OpenAI-compatible chat route.

    Args:
        server: Gateway the model talks to.
        config: Model under test.

    Returns:
        A client ready to invoke.
    """
    return ChatOpenAI(
        model=config.model,
        base_url=server.url("/v1"),
        api_key=SecretStr(server.api_key),
    )


@pytest.mark.parametrize("model_config", _CHAT_COMPLETIONS_MODEL_CONFIGS)
class TestChatCompletion:
    """Non-streaming chat, the baseline every other feature in this module builds on."""

    def test_invoke_returns_real_content(
        self,
        model_config: ModelConfig,
        agentic_tool: AgenticTool,
        agentic_server: AgenticServer,
    ) -> None:
        """`.invoke()` returns an `AIMessage` carrying the model's real answer."""
        response = _chat_model(agentic_server, model_config).invoke(_PLANET_PROMPT)
        assert isinstance(response, AIMessage)
        assert "jupiter" in str(response.content).lower()


@pytest.mark.parametrize("model_config", _CHAT_COMPLETIONS_MODEL_CONFIGS)
class TestStreaming:
    """`.stream()` reassembles into the same real content `.invoke()` returns."""

    def test_stream_yields_chunks_that_reassemble(
        self,
        model_config: ModelConfig,
        agentic_tool: AgenticTool,
        agentic_server: AgenticServer,
    ) -> None:
        """Multiple chunks arrive and concatenate into the expected answer."""
        chunks = list(_chat_model(agentic_server, model_config).stream(_PLANET_PROMPT))
        assert len(chunks) > 1, (
            "a single chunk means the gateway did not actually stream"
        )
        text = "".join(str(chunk.content) for chunk in chunks)
        assert "jupiter" in text.lower()


#: Value only the tool call can reveal, so the model cannot answer without calling it.
_MAGIC_NUMBER = 4817


@tool
def magic_number(key: str) -> int:
    """Look up the registered magic number for a key."""
    del key
    return _MAGIC_NUMBER


@pytest.mark.parametrize("model_config", _CHAT_COMPLETIONS_MODEL_CONFIGS)
class TestToolCalling:
    """`bind_tools` round trip: the model calls the tool and reports its result."""

    def test_bind_tools_round_trip(
        self,
        model_config: ModelConfig,
        agentic_tool: AgenticTool,
        agentic_server: AgenticServer,
    ) -> None:
        """The final answer reflects a value only the tool call could reveal."""
        model = _chat_model(agentic_server, model_config).bind_tools([magic_number])
        messages: list[BaseMessage] = [
            HumanMessage(
                "Call the magic_number tool with key='zephyr', then state the "
                "result in your answer."
            )
        ]
        response = model.invoke(messages)
        assert isinstance(response, AIMessage)
        assert response.tool_calls, "the model never called magic_number"
        messages.append(response)
        for call in response.tool_calls:
            result = magic_number.invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        final = model.invoke(messages)
        assert str(_MAGIC_NUMBER) in str(final.content)


class _NumericAnswer(BaseModel):
    """A single numeric answer."""

    value: int = Field(description="The numeric result.")


class TestStructuredOutput:
    """`with_structured_output` parses the reply into a Pydantic model.

    LangChain turns the Pydantic model into `response_format: json_schema`, which
    the gateway serves from the backend's own schema decoder rather than emulating.
    Support is therefore per model, and both outcomes are asserted here: the parse
    on a backend that decodes schemas, and a clear refusal on one that does not.

    Ref: https://reference.langchain.com/python/langchain_core/runnables/#langchain_core.runnables.base.Runnable.with_structured_output
         docs/api_openai_chat_completions.md
         stdapi/models/chat/_adapters/_openai_chat_completion.py:build_output_config
    """

    @pytest.mark.parametrize("model_config", _STRUCTURED_OUTPUT_MODEL_CONFIGS)
    def test_with_structured_output_parses_pydantic_model(
        self,
        model_config: ModelConfig,
        agentic_tool: AgenticTool,
        agentic_server: AgenticServer,
    ) -> None:
        """The parsed model carries the arithmetic answer as a real `int`."""
        model = _chat_model(agentic_server, model_config).with_structured_output(
            _NumericAnswer
        )
        result = model.invoke(
            "What is 12 plus 30? Respond with only the numeric result."
        )
        assert isinstance(result, _NumericAnswer)
        assert result.value == 42

    @pytest.mark.parametrize("model_config", [_NO_STRUCTURED_OUTPUT_MODEL_CONFIG])
    def test_model_without_a_schema_decoder_is_refused_clearly(
        self,
        model_config: ModelConfig,
        agentic_tool: AgenticTool,
        agentic_server: AgenticServer,
    ) -> None:
        """A backend with no schema decoder refuses, naming the field it cannot take.

        The request is forwarded rather than pre-screened, so the message the
        caller sees is the backend's own. It has to stay specific enough to act
        on: a bare 400 would leave a LangChain user with no way to tell an
        unsupported model from a malformed schema.
        """
        model = _chat_model(agentic_server, model_config).with_structured_output(
            _NumericAnswer
        )
        with pytest.raises(BadRequestError) as refusal:
            model.invoke("What is 12 plus 30? Respond with only the numeric result.")
        assert "outputConfig" in str(refusal.value), refusal.value


@pytest.mark.parametrize("model_config", [_EMBEDDING_MODEL_CONFIG])
class TestEmbeddings:
    """The token-array default, rejected, and the plain-string workaround, accepted.

    Ref: stdapi/types/openai_embeddings.py:64-112 (EmbeddingCreateParams._unsupported)
    """

    def test_default_tokenization_is_rejected(
        self,
        model_config: ModelConfig,
        agentic_tool: AgenticTool,
        agentic_server: AgenticServer,
    ) -> None:
        """`OpenAIEmbeddings` tokenizes client-side by default and is rejected.

        It sends token arrays -- the exact artifact this backend's embeddings
        endpoint rejects with a 400 rather than silently embedding something else.
        """
        embeddings = OpenAIEmbeddings(
            model=model_config.model,
            base_url=agentic_server.url("/v1"),
            api_key=SecretStr(agentic_server.api_key),
        )
        with pytest.raises(BadRequestError):
            embeddings.embed_query("stdapi.ai gateway integration test text")

    def test_check_embedding_ctx_length_false_sends_plain_strings(
        self,
        model_config: ModelConfig,
        agentic_tool: AgenticTool,
        agentic_server: AgenticServer,
    ) -> None:
        """Disabling client-side tokenization makes the same call succeed."""
        embeddings = OpenAIEmbeddings(
            model=model_config.model,
            base_url=agentic_server.url("/v1"),
            api_key=SecretStr(agentic_server.api_key),
            check_embedding_ctx_length=False,
        )
        vector = embeddings.embed_query("stdapi.ai gateway integration test text")
        assert len(vector) > 0
        assert any(vector)


@pytest.mark.parametrize("model_config", _MODEL_CONFIGS)
class TestChatAnthropic:
    """The same chat round trip through langchain-anthropic's Messages wire format.

    ``base_url`` is a pydantic alias for the ``anthropic_api_url`` field (verified
    against the installed package's ``ChatAnthropic.model_fields``), so it points
    `ChatAnthropic` at a non-Anthropic endpoint; the gateway's Anthropic Messages
    route accepts every model it serves, not only Claude, mirroring
    ``PI_MESSAGES`` in ``test_pi.py``.
    """

    @pytest.fixture
    def agentic_tool(self) -> AgenticTool:
        """Override: this class drives the gateway's ``/anthropic`` route instead."""
        return _TOOL_ANTHROPIC

    def test_invoke_returns_real_content(
        self,
        model_config: ModelConfig,
        agentic_tool: AgenticTool,
        agentic_server: AgenticServer,
    ) -> None:
        """`.invoke()` over Anthropic Messages returns the model's real answer."""
        model = ChatAnthropic(
            model_name=model_config.model,
            base_url=agentic_server.url("/anthropic"),
            api_key=SecretStr(agentic_server.api_key),
            timeout=None,
            stop=None,
        )
        response = model.invoke(_PLANET_PROMPT)
        assert "jupiter" in str(response.content).lower()
