"""Unit tests for the Responses ``prompt`` parameter (Bedrock Prompt Management).

Covers the prompt ARN matcher, the ``prompt.variables`` -> Converse
``promptVariables`` mapping, the request-level parameter rejections, the
``GetPrompt`` model resolution and its TTL cache, and the region/``modelId``
pinning of the resulting Converse call.  Entirely offline: the ``bedrock-agent``
client and the model catalog are stubbed.
"""

from typing import TYPE_CHECKING, Any

import pytest
from sse_starlette import EventSourceResponse

import stdapi.models as models_module
from stdapi.api_errors import ApiError
from stdapi.aws_bedrock import BEDROCK_PROMPT_VAR, BedrockPrompt
from stdapi.config import SETTINGS
from stdapi.models import _PROMPTS, ModelDetails, resolve_bedrock_prompt
from stdapi.models.chat._adapters._openai_responses import map_prompt_variables
from stdapi.models.chat._default import ChatModel
from stdapi.monitoring import REQUEST_ID, REQUEST_LOG
from stdapi.types.openai_responses import ResponseCreateParams, ResponsePrompt
from stdapi.utils import match_bedrock_prompt_arn

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Prompt ARN used across the tests, without version suffix.
_PROMPT_ARN = "arn:aws:bedrock:us-east-1:123456789012:prompt/ABCDE12345"

#: Model the stubbed prompt variant is bound to.
_PROMPT_MODEL = "vendor.model-v1"


class _StubBedrockAgentClient:
    """Minimal ``bedrock-agent`` client returning a canned ``GetPrompt`` payload."""

    def __init__(self, prompt: dict[str, Any]) -> None:
        """Store the payload and initialize the call counter.

        Args:
            prompt: ``GetPrompt`` response to return.
        """
        self.prompt = prompt
        self.calls: list[dict[str, Any]] = []

    async def get_prompt(self, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        """Record the call and return the canned payload.

        Args:
            **kwargs: ``GetPrompt`` request parameters.

        Returns:
            The canned ``GetPrompt`` response.
        """
        self.calls.append(kwargs)
        return self.prompt


def _text_prompt(model_id: str | None = _PROMPT_MODEL) -> dict[str, Any]:
    """Build a ``GetPrompt`` response for a TEXT prompt.

    Args:
        model_id: Model bound to the variant, or ``None`` for an unbound prompt.

    Returns:
        A ``GetPrompt`` response payload.
    """
    variant: dict[str, Any] = {
        "name": "v1",
        "templateType": "TEXT",
        "templateConfiguration": {"text": {"text": "Hello {{name}}"}},
    }
    if model_id is not None:
        variant["modelId"] = model_id
    return {"arn": _PROMPT_ARN, "variants": [variant]}


async def _empty_stream() -> AsyncGenerator[dict[str, Any]]:
    """Yield no Converse stream event."""
    return
    yield {}


def _model_details() -> ModelDetails:
    """Build the catalog entry the stubbed prompt resolves to."""
    return ModelDetails(
        id=_PROMPT_MODEL,
        name="Model",
        provider="Vendor",
        input_modalities=["TEXT"],
        output_modalities=["TEXT"],
        regions=["us-east-1"],
    )


@pytest.fixture
def prompt_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[_StubBedrockAgentClient]:
    """Enable prompt ARNs and stub both ``GetPrompt`` and the model catalog."""
    monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_prompt_arn", True)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1"])
    client = _StubBedrockAgentClient(_text_prompt())
    monkeypatch.setattr(models_module, "get_client", lambda *_args: client)

    async def fake_validate_model(model_id: str, **_kwargs: Any) -> ModelDetails:  # noqa: ANN401
        if model_id != _PROMPT_MODEL:
            msg = f"unknown model {model_id}"
            raise ApiError(msg)
        return _model_details()

    monkeypatch.setattr(models_module, "validate_model", fake_validate_model)
    _PROMPTS.clear()
    yield client
    _PROMPTS.clear()


class TestPromptArnMatcher:
    """The prompt ARN matcher must accept only Prompt Management ARNs."""

    @pytest.mark.parametrize(
        ("arn", "version"),
        [
            (_PROMPT_ARN, None),
            (f"{_PROMPT_ARN}:1", "1"),
            (f"{_PROMPT_ARN}:12345", "12345"),
            (
                "arn:aws-us-gov:bedrock:us-gov-west-1:123456789012:prompt/ABCDE12345",
                None,
            ),
        ],
    )
    def test_accepts_prompt_arn(self, arn: str, version: str | None) -> None:
        """A prompt ARN matches, with its optional version captured."""
        result = match_bedrock_prompt_arn(arn)
        assert result is not None
        assert result.group("version") == version
        assert result.group("region") in {"us-east-1", "us-gov-west-1"}

    @pytest.mark.parametrize(
        "arn",
        [
            "arn:aws:bedrock:us-east-1:123456789012:prompt/ABCDE12345:1 ",
            "arn:aws:bedrock:us-east-1:123456789012:prompt/ABCDE12345:DRAFT",
            "arn:aws:bedrock:us-east-1:123456789012:prompt/SHORT",
            "arn:aws:bedrock:us-east-1:123456789012:default-prompt-router/my-router",
            "arn:aws:bedrock:us-east-1:123456789012:inference-profile/p1",
            "pmpt_abc123",
        ],
    )
    def test_rejects_other_ids(self, arn: str) -> None:
        """Non-prompt identifiers, and trailing data, do not match."""
        assert match_bedrock_prompt_arn(arn) is None


class TestPromptVariables:
    """``prompt.variables`` map onto Bedrock ``promptVariables`` text blocks."""

    def test_maps_string_values(self) -> None:
        """String values are wrapped in a ``text`` block."""
        assert map_prompt_variables({"genre": "pop", "number": "3"}) == {
            "genre": {"text": "pop"},
            "number": {"text": "3"},
        }

    def test_no_variables(self) -> None:
        """Missing variables map to an empty dict."""
        assert map_prompt_variables(None) == {}

    def test_rejects_content_part_values(self) -> None:
        """Bedrock prompt variables only carry text, so content parts are rejected."""
        variables = {"doc": {"type": "input_text", "text": "hello"}}
        with pytest.raises(ApiError, match="must be a string"):
            map_prompt_variables(variables)  # type: ignore[arg-type]


class TestPromptRequestValidation:
    """A ``prompt`` request cannot carry what the prompt template provides."""

    def test_prompt_alone_is_accepted(self) -> None:
        """``prompt`` is no longer rejected outright."""
        request = ResponseCreateParams(
            model=_PROMPT_MODEL,
            prompt=ResponsePrompt(id=_PROMPT_ARN, variables={"a": "b"}),
        )
        assert request.prompt is not None
        assert request.prompt.id == _PROMPT_ARN

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("input", "hello"),
            ("instructions", "be brief"),
            ("temperature", 0.5),
            ("top_p", 0.5),
            ("max_output_tokens", 16),
            ("tools", []),
            ("tool_choice", "auto"),
            ("previous_response_id", "resp-1"),
        ],
    )
    def test_rejects_incompatible_parameters(self, field: str, value: Any) -> None:  # noqa: ANN401
        """Request-level equivalents of the prompt's own content are rejected."""
        with pytest.raises(ApiError, match="cannot be used with 'prompt'"):
            ResponseCreateParams(
                model=_PROMPT_MODEL,
                prompt=ResponsePrompt(id=_PROMPT_ARN),
                **{field: value},
            )

    def test_allows_orthogonal_parameters(self) -> None:
        """Runtime-level parameters unrelated to the template stay allowed."""
        request = ResponseCreateParams(
            model=_PROMPT_MODEL,
            prompt=ResponsePrompt(id=_PROMPT_ARN),
            stream=True,
            store=True,
            metadata={"k": "v"},
        )
        assert request.stream is True


class TestResolveBedrockPrompt:
    """``resolve_bedrock_prompt`` gates, validates and resolves the prompt."""

    async def test_disabled_by_configuration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The feature gate returns a clear 400 instead of an opaque rejection."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_prompt_arn", False)
        with pytest.raises(ApiError, match="not allowed by server configuration"):
            await resolve_bedrock_prompt(_PROMPT_ARN, None)

    @pytest.mark.usefixtures("prompt_enabled")
    async def test_rejects_hosted_prompt_id(self) -> None:
        """An OpenAI-hosted prompt template ID cannot exist on this gateway."""
        with pytest.raises(ApiError, match="not an Amazon Bedrock Prompt Management"):
            await resolve_bedrock_prompt("pmpt_abc123", None)

    @pytest.mark.usefixtures("prompt_enabled")
    async def test_rejects_unconfigured_region(self) -> None:
        """An ARN region that is not configured must not reach get_client()."""
        arn = "arn:aws:bedrock:ap-south-1:123456789012:prompt/ABCDE12345"
        with pytest.raises(ApiError, match="not a configured Bedrock region"):
            await resolve_bedrock_prompt(arn, None)

    @pytest.mark.usefixtures("prompt_enabled")
    @pytest.mark.parametrize("version", ["DRAFT", "", "١٢", "123456"])
    async def test_rejects_non_numeric_version(self, version: str) -> None:
        """Only the ASCII digits Bedrock accepts as an ARN version suffix are allowed."""
        with pytest.raises(ApiError, match="Invalid prompt version"):
            await resolve_bedrock_prompt(_PROMPT_ARN, version)

    @pytest.mark.usefixtures("prompt_enabled")
    async def test_rejects_version_conflict(self) -> None:
        """A version suffix disagreeing with ``prompt.version`` is ambiguous."""
        with pytest.raises(ApiError, match="conflicts with the version"):
            await resolve_bedrock_prompt(f"{_PROMPT_ARN}:2", "3")

    async def test_rejects_chat_prompt(
        self, prompt_enabled: _StubBedrockAgentClient
    ) -> None:
        """Only TEXT prompts are supported by this implementation."""
        prompt_enabled.prompt["variants"][0]["templateType"] = "CHAT"
        with pytest.raises(ApiError, match="is not a TEXT prompt"):
            await resolve_bedrock_prompt(_PROMPT_ARN, None)

    async def test_rejects_prompt_without_model(
        self, prompt_enabled: _StubBedrockAgentClient
    ) -> None:
        """A prompt with no bound model has no dispatch/billing model."""
        prompt_enabled.prompt = _text_prompt(None)
        with pytest.raises(ApiError, match="is not bound to a model"):
            await resolve_bedrock_prompt(_PROMPT_ARN, None)

    async def test_rejects_unservable_model(
        self, prompt_enabled: _StubBedrockAgentClient
    ) -> None:
        """A prompt model missing from the catalog cannot be served."""
        prompt_enabled.prompt = _text_prompt("vendor.unknown-v1")
        with pytest.raises(ApiError, match="unknown model"):
            await resolve_bedrock_prompt(_PROMPT_ARN, None)

    @pytest.mark.parametrize(
        "model_id",
        [
            f"us.{_PROMPT_MODEL}",
            f"arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.{_PROMPT_MODEL}",
            f"arn:aws:bedrock:us-east-1::foundation-model/{_PROMPT_MODEL}",
        ],
    )
    async def test_resolves_prompt_bound_to_an_inference_profile(
        self,
        prompt_enabled: _StubBedrockAgentClient,
        monkeypatch: pytest.MonkeyPatch,
        model_id: str,
    ) -> None:
        """A prompt may name an inference profile or an ARN instead of the model."""
        monkeypatch.setitem(
            models_module._ALL_MODELS,  # noqa: SLF001
            _PROMPT_MODEL,
            _model_details(),
        )
        prompt_enabled.prompt = _text_prompt(model_id)
        resolved = await resolve_bedrock_prompt(_PROMPT_ARN, None)
        assert resolved.model_id == _PROMPT_MODEL

    async def test_resolves_versioned_arn(
        self, prompt_enabled: _StubBedrockAgentClient
    ) -> None:
        """``prompt.version`` is appended to the ARN and read via GetPrompt."""
        resolved = await resolve_bedrock_prompt(_PROMPT_ARN, "2")
        assert resolved == BedrockPrompt(
            arn=f"{_PROMPT_ARN}:2", region="us-east-1", model_id=_PROMPT_MODEL
        )
        assert prompt_enabled.calls == [
            {"promptIdentifier": _PROMPT_ARN, "promptVersion": "2"}
        ]

    async def test_version_from_arn_suffix(
        self, prompt_enabled: _StubBedrockAgentClient
    ) -> None:
        """A version already in the ARN is used when ``prompt.version`` is unset."""
        resolved = await resolve_bedrock_prompt(f"{_PROMPT_ARN}:7", None)
        assert resolved.arn == f"{_PROMPT_ARN}:7"
        assert prompt_enabled.calls == [
            {"promptIdentifier": _PROMPT_ARN, "promptVersion": "7"}
        ]

    async def test_draft_prompt_without_version(
        self, prompt_enabled: _StubBedrockAgentClient
    ) -> None:
        """Without a version, the working draft is read and the ARN kept bare."""
        resolved = await resolve_bedrock_prompt(_PROMPT_ARN, None)
        assert resolved.arn == _PROMPT_ARN
        assert prompt_enabled.calls == [{"promptIdentifier": _PROMPT_ARN}]

    async def test_get_prompt_is_cached(
        self, prompt_enabled: _StubBedrockAgentClient
    ) -> None:
        """Repeated resolutions of the same version hit the TTL cache."""
        await resolve_bedrock_prompt(_PROMPT_ARN, "1")
        await resolve_bedrock_prompt(_PROMPT_ARN, "1")
        await resolve_bedrock_prompt(_PROMPT_ARN, "2")
        assert [call["promptVersion"] for call in prompt_enabled.calls] == ["1", "2"]


class TestPromptConverseRequest:
    """The Converse call of a prompt request is pinned to the prompt resource."""

    @pytest.fixture
    def prompt_context(self) -> Iterator[BedrockPrompt]:
        """Install a resolved prompt and a minimal request scope."""
        REQUEST_ID.set("test-request")
        REQUEST_LOG.set({"level": "info"})  # type: ignore[typeddict-item]
        prompt = BedrockPrompt(
            arn=f"{_PROMPT_ARN}:1", region="us-east-1", model_id=_PROMPT_MODEL
        )
        token = BEDROCK_PROMPT_VAR.set(prompt)
        yield prompt
        BEDROCK_PROMPT_VAR.reset(token)

    async def test_body_carries_only_prompt_variables(
        self, monkeypatch: pytest.MonkeyPatch, prompt_context: BedrockPrompt
    ) -> None:
        """The whole request body is the prompt variables (plus the modelId)."""
        captured: dict[str, Any] = {}

        async def fake_converse(
            _self: ChatModel, request: ConverseRequestBaseTypeDef
        ) -> dict[str, Any]:
            captured.update(request)
            return {
                "output": {"message": {"role": "assistant", "content": []}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            }

        monkeypatch.setattr(ChatModel, "converse", fake_converse)
        request = ResponseCreateParams(
            model=_PROMPT_MODEL,
            prompt=ResponsePrompt(id=prompt_context.arn, variables={"genre": "pop"}),
        )
        await ChatModel(_PROMPT_MODEL).create_response(request, "resp-1", 0.0)
        assert set(captured) == {"modelId", "promptVariables"}
        assert captured["promptVariables"] == {"genre": {"text": "pop"}}

    async def test_no_variables_omits_the_field(
        self, monkeypatch: pytest.MonkeyPatch, prompt_context: BedrockPrompt
    ) -> None:
        """A prompt without variables sends no ``promptVariables`` at all."""
        captured: dict[str, Any] = {}

        async def fake_converse(
            _self: ChatModel, request: ConverseRequestBaseTypeDef
        ) -> dict[str, Any]:
            captured.update(request)
            return {
                "output": {"message": {"role": "assistant", "content": []}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            }

        monkeypatch.setattr(ChatModel, "converse", fake_converse)
        request = ResponseCreateParams(
            model=_PROMPT_MODEL, prompt=ResponsePrompt(id=prompt_context.arn)
        )
        await ChatModel(_PROMPT_MODEL).create_response(request, "resp-1", 0.0)
        assert set(captured) == {"modelId"}

    async def test_streaming_uses_converse_stream(
        self, monkeypatch: pytest.MonkeyPatch, prompt_context: BedrockPrompt
    ) -> None:
        """A streaming prompt request opens ConverseStream with the same body."""
        captured: dict[str, Any] = {}

        async def fake_converse_stream(
            _self: ChatModel, request: ConverseRequestBaseTypeDef
        ) -> dict[str, Any]:
            captured.update(request)
            return {"stream": _empty_stream()}

        monkeypatch.setattr(ChatModel, "converse_stream", fake_converse_stream)
        request = ResponseCreateParams(
            model=_PROMPT_MODEL,
            prompt=ResponsePrompt(id=prompt_context.arn, variables={"genre": "pop"}),
            stream=True,
        )
        result = await ChatModel(_PROMPT_MODEL).create_response(request, "resp-1", 0.0)
        assert isinstance(result, EventSourceResponse)
        assert captured["promptVariables"] == {"genre": {"text": "pop"}}

    async def test_candidate_regions_pinned_to_prompt(
        self, prompt_context: BedrockPrompt
    ) -> None:
        """A region-bound prompt ARN disables cross-region failover."""
        model = ChatModel(_PROMPT_MODEL)
        assert await model._converse_candidate_regions() == [  # noqa: SLF001
            prompt_context.region
        ]

    async def test_model_id_is_the_prompt_arn(
        self, prompt_context: BedrockPrompt
    ) -> None:
        """The prompt ARN replaces the resolved model/profile ID."""
        request: ConverseRequestBaseTypeDef = {"modelId": ""}
        await ChatModel(_PROMPT_MODEL)._prepare_converse_request_for_region(  # noqa: SLF001
            request, "us-east-1"
        )
        assert request["modelId"] == prompt_context.arn
