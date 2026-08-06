"""litellm driven against the gateway's OpenAI-compatible routes.

litellm is here for one behaviour no other client in this lane has: it puts
*its own* control parameters on the wire. ``extra_body`` is merged into the
request body at the top level, so ``extra_body={"drop_params": True}`` -- the
literal call RAGFlow's embedding client makes -- arrives as a body field named
``drop_params``, which is a LiteLLM convention and a model parameter for no
provider at all. Verified against the installed package by capturing the request
it sends; the same merge happens on ``completion`` and ``embedding`` alike.

That is the client-side half of ``EXTRA_MODEL_PARAMS_DENYLIST``: the gateway
strips those names from the extra model parameters instead of forwarding them to
Bedrock as inference fields, where they fail with ``ValidationException``. Both
halves are asserted here -- the denylisted name gets through, and a name that is
genuinely not a model parameter still fails -- because a denylist that swallowed
everything would pass the first assertion just as well.

litellm also forwards *unknown* keyword arguments regardless of its own
``drop_params`` flag (that flag only removes parameters it knows a provider
rejects), so reaching the gateway with an unrecognized field is a single keyword
argument away for any litellm user.

Requires --agentic, podman, and Bedrock credentials.

Ref: https://docs.litellm.ai/docs/providers/openai_compatible
     https://docs.litellm.ai/docs/completion/input
     https://docs.litellm.ai/docs/embedding/supported_embedding
     docs/use_cases_ragflow.md
     stdapi/aws_bedrock.py:filter_extra_model_parameters
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import litellm
import pytest
from litellm.exceptions import BadRequestError

from ._runner import ModelConfig
from ._tools import AgenticTool

if TYPE_CHECKING:
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


#: litellm, against the gateway's OpenAI-compatible chat and embeddings routes.
_TOOL = AgenticTool(
    id="litellm",
    npm_package=None,
    binary="python",
    route="/v1",
    metrics_prefix="LL-METRICS",
    build=_unused_build,
    parse=_unused_parse,
    prepare_workdir=_unused_prepare_workdir,
    # litellm sends no per-run identifier the gateway records, so its requests can
    # only be attributed positionally.
    attributes_sessions=False,
)

TOOL = _TOOL


#: Models exercised on the chat tests in this module.
_MODEL_CONFIGS = [
    pytest.param(
        ModelConfig(model="anthropic.claude-haiku-4-5-20251001-v1:0"),
        id="claude-haiku-4-5",
    ),
    pytest.param(ModelConfig(model="amazon.nova-2-lite-v1:0"), id="nova-2-lite"),
]

#: Single cheap model for the assertions that are about the body, not the backend.
_CHEAP_MODEL_CONFIG = pytest.param(
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

#: Body field name that is neither a model parameter nor a litellm control one.
#:
#: The denylist has to keep rejecting this: a gateway that dropped every
#: unrecognized field would hide a caller's typo instead of reporting it.
_NOT_A_PARAMETER = "not_a_model_parameter"


def _model_name(config: ModelConfig) -> str:
    """Model id prefixed for litellm's OpenAI-compatible provider.

    The ``openai/`` prefix selects the provider; litellm strips it before sending,
    so the gateway receives the bare model id.

    Args:
        config: Model under test.

    Returns:
        The prefixed model name litellm expects.
    """
    return f"openai/{config.model}"


@pytest.mark.parametrize("model_config", _MODEL_CONFIGS)
class TestChatCompletion:
    """The baseline round trip every other assertion in this module builds on."""

    def test_completion_returns_real_content(
        self, model_config: ModelConfig, agentic_server: AgenticServer
    ) -> None:
        """`completion()` returns the model's real answer."""
        response = litellm.completion(
            model=_model_name(model_config),
            api_base=agentic_server.url("/v1"),
            api_key=agentic_server.api_key,
            messages=[{"role": "user", "content": _PLANET_PROMPT}],
        )
        assert "jupiter" in str(response.choices[0].message.content).lower()

    def test_stream_yields_chunks_that_reassemble(
        self, model_config: ModelConfig, agentic_server: AgenticServer
    ) -> None:
        """Multiple SSE chunks arrive and concatenate into the expected answer."""
        chunks = list(
            litellm.completion(
                model=_model_name(model_config),
                api_base=agentic_server.url("/v1"),
                api_key=agentic_server.api_key,
                messages=[{"role": "user", "content": _PLANET_PROMPT}],
                stream=True,
            )
        )
        assert len(chunks) > 1, (
            "a single chunk means the gateway did not actually stream"
        )
        text = "".join(chunk.choices[0].delta.content or "" for chunk in chunks)
        assert "jupiter" in text.lower()


class TestClientControlParameters:
    """``extra_body`` control names are stripped; a real unknown field still fails.

    Ref: docs/operations_configuration.md (EXTRA_MODEL_PARAMS_DENYLIST)
         stdapi/aws_bedrock.py:filter_extra_model_parameters
    """

    @pytest.mark.parametrize("model_config", [_CHEAP_MODEL_CONFIG])
    def test_drop_params_in_extra_body_is_stripped_on_chat(
        self, model_config: ModelConfig, agentic_server: AgenticServer
    ) -> None:
        """A body carrying `drop_params` still answers instead of failing."""
        response = litellm.completion(
            model=_model_name(model_config),
            api_base=agentic_server.url("/v1"),
            api_key=agentic_server.api_key,
            messages=[{"role": "user", "content": _PLANET_PROMPT}],
            extra_body={"drop_params": True},
        )
        assert "jupiter" in str(response.choices[0].message.content).lower()

    @pytest.mark.parametrize("model_config", [_EMBEDDING_MODEL_CONFIG])
    def test_drop_params_in_extra_body_is_stripped_on_embeddings(
        self, model_config: ModelConfig, agentic_server: AgenticServer
    ) -> None:
        """RAGFlow's exact embedding call embeds instead of failing.

        RAGFlow hardcodes ``extra_body={"drop_params": True}`` on every embeddings
        request, which is what made this the route the denylist was written for.
        """
        response = litellm.embedding(
            model=_model_name(model_config),
            api_base=agentic_server.url("/v1"),
            api_key=agentic_server.api_key,
            input=["stdapi.ai gateway integration test text"],
            extra_body={"drop_params": True},
        )
        vector = response.data[0]["embedding"]
        assert len(vector) > 0
        assert any(vector)

    @pytest.mark.parametrize("model_config", [_CHEAP_MODEL_CONFIG])
    def test_unrecognized_model_parameter_is_still_refused(
        self, model_config: ModelConfig, agentic_server: AgenticServer
    ) -> None:
        """A field that is not a control name reaches Bedrock and is rejected.

        This is the half that proves the denylist is a denylist: the gateway keeps
        forwarding every extra parameter it was not told to strip, so a caller
        misspelling a real inference field still learns about it.
        """
        with pytest.raises(BadRequestError):
            litellm.completion(
                model=_model_name(model_config),
                api_base=agentic_server.url("/v1"),
                api_key=agentic_server.api_key,
                messages=[{"role": "user", "content": _PLANET_PROMPT}],
                extra_body={_NOT_A_PARAMETER: "x"},
            )


@pytest.mark.parametrize("model_config", [_EMBEDDING_MODEL_CONFIG])
class TestEmbeddings:
    """litellm embeds plain strings, where `OpenAIEmbeddings` sends token arrays.

    The contrast with ``test_langchain.py`` is the point: two OpenAI-compatible
    Python clients disagree on what ``input`` carries, and only one of them is
    accepted by this backend. litellm additionally names ``encoding_format:
    "float"`` on every request, which the gateway has to keep accepting.
    """

    def test_embedding_returns_a_real_vector(
        self, model_config: ModelConfig, agentic_server: AgenticServer
    ) -> None:
        """The default call succeeds and returns a populated vector."""
        response = litellm.embedding(
            model=_model_name(model_config),
            api_base=agentic_server.url("/v1"),
            api_key=agentic_server.api_key,
            input=["stdapi.ai gateway integration test text"],
        )
        vector = response.data[0]["embedding"]
        assert len(vector) > 0
        assert any(vector)
