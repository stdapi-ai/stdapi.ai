"""Unit tests for the Responses ``prompt`` parameter (Bedrock Prompt Management).

This ``prompt`` parameter is not OpenAI's reusable prompt object: it resolves to
an Amazon Bedrock Prompt Management ARN (optionally ``:version``) and its
``variables`` become Converse ``promptVariables``.  Bedrock renders the
messages, system prompt, tools and inference config from the stored prompt
version, so the request must not supply request-level equivalents, and the
prompt ARN replaces the ``modelId`` of a region-pinned Converse call.
Entirely offline: the ``bedrock-agent`` client and the model catalog are stubbed.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-management.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
     https://developers.openai.com/api/reference/resources/responses/methods/create
     stdapi/types/openai_responses.py:ResponsePrompt
     stdapi/models/__init__.py:resolve_bedrock_prompt
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
from stdapi.monitoring import REQUEST_ID
from stdapi.types.openai_responses import Response, ResponseCreateParams, ResponsePrompt
from stdapi.utils import match_bedrock_prompt_arn

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator

    from starlette.testclient import TestClient

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


def _capturing_converse(
    captured: dict[str, Any],
) -> Callable[[ChatModel, ConverseRequestBaseTypeDef], Awaitable[dict[str, Any]]]:
    """Build a ``ChatModel.converse`` replacement recording the request body.

    Args:
        captured: Mapping updated with the Converse request body.

    Returns:
        A coroutine function returning a canned one-token Converse response.
    """

    async def fake_converse(
        _self: ChatModel, request: ConverseRequestBaseTypeDef
    ) -> dict[str, Any]:
        captured.update(request)
        return {
            "output": {"message": {"role": "assistant", "content": []}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }

    return fake_converse


def _capturing_converse_stream(
    captured: dict[str, Any],
) -> Callable[[ChatModel, ConverseRequestBaseTypeDef], Awaitable[dict[str, Any]]]:
    """Build a ``ChatModel.converse_stream`` replacement recording the request body.

    Args:
        captured: Mapping updated with the ConverseStream request body.

    Returns:
        A coroutine function returning an empty Converse event stream.
    """

    async def fake_converse_stream(
        _self: ChatModel, request: ConverseRequestBaseTypeDef
    ) -> dict[str, Any]:
        captured.update(request)
        return {"stream": _empty_stream()}

    return fake_converse_stream


class TestPromptArnMatcher:
    """The prompt ARN matcher accepts only Prompt Management ARNs.

    Ref: stdapi/utils.py:match_bedrock_prompt_arn
    """

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
        """A prompt ARN matches, splitting off its region and optional version.

        The ``base`` group is what the resolver sends to ``GetPrompt``, so the
        version suffix must not be part of it; non-``aws`` partitions such as
        ``aws-us-gov`` are accepted too.
        """
        result = match_bedrock_prompt_arn(arn)
        assert result is not None
        assert result.group("version") == version
        assert result.group("region") == arn.split(":")[3]
        assert result.group("base") == (
            arn.removesuffix(f":{version}") if version else arn
        )

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
        """Non-prompt identifiers, and trailing data, do not match.

        The matcher is end-anchored, so a trailing space or a non-numeric
        version suffix must not be absorbed into the prompt ID; prompt routers
        and inference profiles are distinct resource types.
        """
        assert match_bedrock_prompt_arn(arn) is None


class TestPromptVariables:
    """``prompt.variables`` map onto Bedrock ``promptVariables`` text blocks.

    Ref: stdapi/models/chat/_adapters/_openai_responses.py:map_prompt_variables
    """

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
        with pytest.raises(ApiError, match="must be a string") as excinfo:
            map_prompt_variables(variables)  # type: ignore[arg-type]
        assert excinfo.value.status == 400
        assert "'doc'" in str(excinfo.value), "the offending variable is named"


class _StubPromptChatModel:
    """Stub chat backend recording generation calls and the prompt in scope."""

    IS_MANTLE = False

    def __init__(self) -> None:
        self.requests: list[ResponseCreateParams] = []
        self.prompts: list[BedrockPrompt | None] = []

    def native_store_supported(self) -> bool:
        """Local-store stub: no Mantle native storage."""
        return False

    async def create_response(
        self,
        request: ResponseCreateParams,
        response_id: str,
        created_at: float,
        moderation_builder: Any = None,  # noqa: ANN401
    ) -> Response:
        """Record the request and the pinned prompt, and return a canned response."""
        self.requests.append(request)
        self.prompts.append(BEDROCK_PROMPT_VAR.get(None))
        return Response(
            id=response_id,
            created_at=int(created_at),
            model=request.model,
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )


class TestPromptRouteGuards:
    """POST /v1/responses refuses a ``prompt`` the target model cannot serve.

    The prompt version carries its own model, so a request naming another model
    would silently run somewhere else — different capabilities, price and region
    pinning. Bedrock Mantle endpoints cannot resolve a prompt ARN at all.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-management.html
         stdapi/routes/openai_responses.py:_apply_prompt_template
    """

    @pytest.fixture
    def backend(self, monkeypatch: pytest.MonkeyPatch) -> _StubPromptChatModel:
        """Stub model validation and the generation backend of the route."""
        from stdapi.routes import openai_responses  # noqa: PLC0415

        async def _validate_model(
            model_id: str, *_args: object, **_kwargs: object
        ) -> ModelDetails:
            del model_id
            return _model_details()

        stub = _StubPromptChatModel()
        monkeypatch.setattr(openai_responses, "validate_model", _validate_model)
        monkeypatch.setattr(openai_responses, "get_chat_model", lambda _model_id: stub)
        return stub

    @staticmethod
    def _resolving_to(
        monkeypatch: pytest.MonkeyPatch, model_id: str
    ) -> list[tuple[str, str | None]]:
        """Stub ``resolve_bedrock_prompt`` and record its calls.

        Args:
            monkeypatch: Patcher applied to the route module.
            model_id: Model the stubbed prompt version is bound to.

        Returns:
            The mutable list of ``(prompt id, version)`` resolution calls.
        """
        from stdapi.routes import openai_responses  # noqa: PLC0415

        calls: list[tuple[str, str | None]] = []

        async def _resolve(prompt_id: str, version: str | None) -> BedrockPrompt:
            calls.append((prompt_id, version))
            return BedrockPrompt(arn=prompt_id, region="us-east-1", model_id=model_id)

        monkeypatch.setattr(openai_responses, "resolve_bedrock_prompt", _resolve)
        return calls

    def test_prompt_model_mismatch_is_rejected(
        self,
        app_client: TestClient,
        backend: _StubPromptChatModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A prompt bound to another model is a 400 naming both models.

        Nothing is generated, so the caller is never billed on a model it did
        not ask for.
        """
        calls = self._resolving_to(monkeypatch, "vendor.other-v1")
        response = app_client.post(
            "/v1/responses",
            json={"model": _PROMPT_MODEL, "prompt": {"id": _PROMPT_ARN}},
        )
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert "vendor.other-v1" in error["message"]
        assert _PROMPT_MODEL in error["message"]
        assert error["type"] == "invalid_request_error"
        assert calls == [(_PROMPT_ARN, None)]
        assert not backend.requests, "generation ran on a mismatched prompt model"

    def test_prompt_on_a_mantle_model_is_rejected_before_resolution(
        self,
        app_client: TestClient,
        backend: _StubPromptChatModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A Mantle-served model rejects ``prompt`` without calling GetPrompt.

        Mantle's OpenAI-shaped API has no notion of a Bedrock prompt resource,
        so resolving the ARN would be wasted work.
        """
        backend.IS_MANTLE = True
        calls = self._resolving_to(monkeypatch, _PROMPT_MODEL)
        response = app_client.post(
            "/v1/responses",
            json={"model": _PROMPT_MODEL, "prompt": {"id": _PROMPT_ARN}},
        )
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert "does not support the 'prompt' parameter" in error["message"]
        assert _PROMPT_MODEL in error["message"]
        assert calls == [], "the prompt was resolved for a model that cannot use it"
        assert not backend.requests

    def test_matching_prompt_pins_the_request(
        self,
        app_client: TestClient,
        backend: _StubPromptChatModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A prompt bound to the requested model reaches generation, pinned in scope.

        ``BEDROCK_PROMPT_VAR`` is what makes the Converse call use the prompt
        ARN as its ``modelId``, so it must be set before the backend runs.
        """
        self._resolving_to(monkeypatch, _PROMPT_MODEL)
        response = app_client.post(
            "/v1/responses",
            json={
                "model": _PROMPT_MODEL,
                "prompt": {"id": _PROMPT_ARN, "version": "3"},
            },
        )
        assert response.status_code == 200, response.text
        (pinned,) = backend.prompts
        assert pinned is not None
        assert pinned.arn == _PROMPT_ARN
        assert pinned.model_id == _PROMPT_MODEL


class TestPromptRequestValidation:
    """A ``prompt`` request cannot carry what the prompt template provides.

    Ref: stdapi/types/openai_responses.py:ResponseCreateParams
    """

    def test_prompt_alone_is_accepted(self) -> None:
        """A ``prompt`` reference with variables is accepted on its own."""
        request = ResponseCreateParams(
            model=_PROMPT_MODEL,
            prompt=ResponsePrompt(id=_PROMPT_ARN, variables={"a": "b"}),
        )
        assert request.prompt is not None
        assert request.prompt.id == _PROMPT_ARN
        assert request.prompt.variables == {"a": "b"}
        assert request.input is None, "the template supplies the conversation"

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
        """Request-level equivalents of the prompt's own content are rejected.

        Bedrock renders these from the stored prompt version, so accepting and
        dropping them would silently ignore what the caller asked for; the
        error names the offending field.
        """
        with pytest.raises(ApiError, match="cannot be used with 'prompt'") as excinfo:
            ResponseCreateParams(
                model=_PROMPT_MODEL,
                prompt=ResponsePrompt(id=_PROMPT_ARN),
                **{field: value},
            )
        assert excinfo.value.status == 400
        assert f"'{field}'" in str(excinfo.value)

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
        assert request.store is True
        assert request.metadata == {"k": "v"}


class TestResolveBedrockPrompt:
    """``resolve_bedrock_prompt`` gates, validates and resolves the prompt.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-management.html
         stdapi/models/__init__.py:_get_prompt_model_id
    """

    async def test_disabled_by_configuration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The feature gate rejects prompt ARNs with a 400 before any AWS call."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_prompt_arn", False)
        with pytest.raises(
            ApiError, match="not allowed by server configuration"
        ) as excinfo:
            await resolve_bedrock_prompt(_PROMPT_ARN, None)
        assert excinfo.value.status == 400

    async def test_rejects_hosted_prompt_id(
        self, prompt_enabled: _StubBedrockAgentClient
    ) -> None:
        """An OpenAI-hosted prompt template ID cannot exist on this gateway.

        ``prompt.id`` is an Amazon Bedrock ARN here, so upstream's ``pmpt_``
        identifiers are unresolvable and are rejected before ``GetPrompt``.
        """
        with pytest.raises(
            ApiError, match="not an Amazon Bedrock Prompt Management"
        ) as excinfo:
            await resolve_bedrock_prompt("pmpt_abc123", None)
        assert excinfo.value.status == 400
        assert "pmpt_abc123" in str(excinfo.value)
        assert prompt_enabled.calls == []

    async def test_rejects_unconfigured_region(
        self, prompt_enabled: _StubBedrockAgentClient
    ) -> None:
        """An ARN region that is not configured must not reach ``GetPrompt``.

        The region is parsed from client-supplied data and used as a client
        lookup key, so an unconfigured region is rejected instead of raising an
        unhandled ``KeyError``.

        Ref: stdapi/models/__init__.py:_validate_bedrock_region
        """
        arn = "arn:aws:bedrock:ap-south-1:123456789012:prompt/ABCDE12345"
        with pytest.raises(
            ApiError, match="not a configured Bedrock region"
        ) as excinfo:
            await resolve_bedrock_prompt(arn, None)
        assert excinfo.value.status == 400
        assert "ap-south-1" in str(excinfo.value)
        assert prompt_enabled.calls == []

    @pytest.mark.parametrize("version", ["DRAFT", "", "١٢", "123456"])
    async def test_rejects_non_numeric_version(
        self, prompt_enabled: _StubBedrockAgentClient, version: str
    ) -> None:
        """Only the ASCII digits Bedrock accepts as an ARN version suffix are allowed.

        ``١٢`` is an Arabic-Indic pair that ``str.isdigit()`` accepts, so the
        check is ASCII-restricted; ``123456`` exceeds the five-digit suffix
        Bedrock allows.
        """
        with pytest.raises(ApiError, match="Invalid prompt version") as excinfo:
            await resolve_bedrock_prompt(_PROMPT_ARN, version)
        assert excinfo.value.status == 400
        assert prompt_enabled.calls == []

    async def test_rejects_version_conflict(
        self, prompt_enabled: _StubBedrockAgentClient
    ) -> None:
        """A version suffix disagreeing with ``prompt.version`` is ambiguous."""
        with pytest.raises(ApiError, match="conflicts with the version") as excinfo:
            await resolve_bedrock_prompt(f"{_PROMPT_ARN}:2", "3")
        assert excinfo.value.status == 400
        assert "'3'" in str(excinfo.value)
        assert "'2'" in str(excinfo.value)
        assert prompt_enabled.calls == []

    async def test_rejects_chat_prompt(
        self, prompt_enabled: _StubBedrockAgentClient
    ) -> None:
        """Only TEXT prompts are supported by this implementation.

        A CHAT template carries its own message list, which the ``prompt``
        request has no way to complete.
        """
        prompt_enabled.prompt["variants"][0]["templateType"] = "CHAT"
        with pytest.raises(ApiError, match="is not a TEXT prompt") as excinfo:
            await resolve_bedrock_prompt(_PROMPT_ARN, None)
        assert excinfo.value.status == 400
        assert _PROMPT_ARN in str(excinfo.value)

    async def test_rejects_prompt_without_model(
        self, prompt_enabled: _StubBedrockAgentClient
    ) -> None:
        """A prompt with no bound model has no dispatch/billing model."""
        prompt_enabled.prompt = _text_prompt(None)
        with pytest.raises(ApiError, match="is not bound to a model") as excinfo:
            await resolve_bedrock_prompt(_PROMPT_ARN, None)
        assert excinfo.value.status == 400
        assert _PROMPT_ARN in str(excinfo.value)

    async def test_rejects_unservable_model(
        self, prompt_enabled: _StubBedrockAgentClient
    ) -> None:
        """A prompt model missing from the catalog cannot be served.

        The variant's ``modelId`` is validated against the catalog, so the
        rejection carries the model read from the prompt, not the request's.
        """
        prompt_enabled.prompt = _text_prompt("vendor.unknown-v1")
        with pytest.raises(ApiError, match="unknown model") as excinfo:
            await resolve_bedrock_prompt(_PROMPT_ARN, None)
        assert "vendor.unknown-v1" in str(excinfo.value)
        assert prompt_enabled.calls == [{"promptIdentifier": _PROMPT_ARN}]

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
        """A prompt may name an inference profile or an ARN instead of the model.

        The catalog is keyed by bare model ID, so a variant bound to a ``us.``
        cross-Region profile or to a full ARN must be folded back to that key
        before the model is validated.
        """
        monkeypatch.setitem(
            models_module._ALL_MODELS,  # noqa: SLF001
            _PROMPT_MODEL,
            _model_details(),
        )
        prompt_enabled.prompt = _text_prompt(model_id)
        resolved = await resolve_bedrock_prompt(_PROMPT_ARN, None)
        assert resolved.model_id == _PROMPT_MODEL
        assert resolved.arn == _PROMPT_ARN
        assert resolved.region == "us-east-1"

    async def test_resolves_versioned_arn(
        self, prompt_enabled: _StubBedrockAgentClient
    ) -> None:
        """``prompt.version`` is appended to the ARN and read via GetPrompt.

        Bedrock accepts the versioned ARN as a ``modelId``, so the version is
        folded into the ARN rather than kept as a separate field.
        """
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
        assert resolved.model_id == _PROMPT_MODEL
        assert prompt_enabled.calls == [
            {"promptIdentifier": _PROMPT_ARN, "promptVersion": "7"}
        ], "GetPrompt takes the bare ARN plus an explicit promptVersion"

    async def test_draft_prompt_without_version(
        self, prompt_enabled: _StubBedrockAgentClient
    ) -> None:
        """Without a version, the working draft is read and the ARN kept bare."""
        resolved = await resolve_bedrock_prompt(_PROMPT_ARN, None)
        assert resolved.arn == _PROMPT_ARN
        assert resolved.model_id == _PROMPT_MODEL
        assert prompt_enabled.calls == [{"promptIdentifier": _PROMPT_ARN}], (
            "no promptVersion is sent, so GetPrompt returns the draft"
        )

    async def test_get_prompt_is_cached(
        self, prompt_enabled: _StubBedrockAgentClient
    ) -> None:
        """Repeated resolutions of the same version hit the TTL cache.

        The cache is keyed per ARN *and* version, so a second version still
        calls ``GetPrompt``.
        """
        first = await resolve_bedrock_prompt(_PROMPT_ARN, "1")
        cached = await resolve_bedrock_prompt(_PROMPT_ARN, "1")
        await resolve_bedrock_prompt(_PROMPT_ARN, "2")
        assert [call["promptVersion"] for call in prompt_enabled.calls] == ["1", "2"]
        assert cached == first


class TestPromptConverseRequest:
    """The Converse call of a prompt request is pinned to the prompt resource.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
         stdapi/models/__init__.py:_prepare_converse_request_for_region
    """

    @pytest.fixture
    def prompt_context(self, request_log: dict[str, Any]) -> Iterator[BedrockPrompt]:
        """Install a resolved prompt and a minimal request scope.

        Every context variable set here is reset on teardown so the request
        scope does not leak into the tests that follow.
        """
        del request_log
        request_id_token = REQUEST_ID.set("test-request")
        prompt = BedrockPrompt(
            arn=f"{_PROMPT_ARN}:1", region="us-east-1", model_id=_PROMPT_MODEL
        )
        token = BEDROCK_PROMPT_VAR.set(prompt)
        yield prompt
        BEDROCK_PROMPT_VAR.reset(token)
        REQUEST_ID.reset(request_id_token)

    async def test_body_carries_only_prompt_variables(
        self, monkeypatch: pytest.MonkeyPatch, prompt_context: BedrockPrompt
    ) -> None:
        """The whole request body is the prompt variables (plus the modelId).

        Bedrock materializes ``messages``, ``system`` and ``inferenceConfig``
        from the stored prompt, so sending any of them would be rejected
        upstream.
        """
        captured: dict[str, Any] = {}
        monkeypatch.setattr(ChatModel, "converse", _capturing_converse(captured))
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
        """A prompt without variables sends no ``promptVariables`` at all.

        Bedrock rejects an empty ``promptVariables`` map, so the field is
        omitted rather than sent as ``{}``.
        """
        captured: dict[str, Any] = {}
        monkeypatch.setattr(ChatModel, "converse", _capturing_converse(captured))
        request = ResponseCreateParams(
            model=_PROMPT_MODEL, prompt=ResponsePrompt(id=prompt_context.arn)
        )
        await ChatModel(_PROMPT_MODEL).create_response(request, "resp-1", 0.0)
        assert set(captured) == {"modelId"}

    async def test_streaming_uses_converse_stream(
        self, monkeypatch: pytest.MonkeyPatch, prompt_context: BedrockPrompt
    ) -> None:
        """A streaming prompt request opens ConverseStream with the same body.

        ``stream`` is one of the parameters a prompt request may still set, so
        the SSE path must carry the prompt variables just like Converse does.
        """
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            ChatModel, "converse_stream", _capturing_converse_stream(captured)
        )
        request = ResponseCreateParams(
            model=_PROMPT_MODEL,
            prompt=ResponsePrompt(id=prompt_context.arn, variables={"genre": "pop"}),
            stream=True,
        )
        result = await ChatModel(_PROMPT_MODEL).create_response(request, "resp-1", 0.0)
        assert isinstance(result, EventSourceResponse)
        assert set(captured) == {"modelId", "promptVariables"}
        assert captured["promptVariables"] == {"genre": {"text": "pop"}}

    async def test_candidate_regions_pinned_to_prompt(
        self, monkeypatch: pytest.MonkeyPatch, prompt_context: BedrockPrompt
    ) -> None:
        """A region-bound prompt ARN disables cross-region failover.

        The prompt ARN embeds its own region, so retrying the call in one of the
        model's other candidate regions could only fail; the candidate list
        collapses to the prompt's region even when the model has more.
        """

        async def _model_regions(_model_id: str, **_kwargs: Any) -> list[str]:  # noqa: ANN401
            return ["us-west-2", "eu-west-1"]

        monkeypatch.setattr(models_module, "compute_candidate_regions", _model_regions)
        model = ChatModel(_PROMPT_MODEL)
        assert await model._converse_candidate_regions() == [  # noqa: SLF001
            prompt_context.region
        ]

    async def test_model_id_is_the_prompt_arn(
        self, prompt_context: BedrockPrompt
    ) -> None:
        """The prompt ARN replaces the resolved model/profile ID.

        Bedrock keys inference on the ``modelId``, so the versioned prompt ARN
        goes there instead of the catalog model or its inference profile.
        """
        request: ConverseRequestBaseTypeDef = {"modelId": ""}
        await ChatModel(_PROMPT_MODEL)._prepare_converse_request_for_region(  # noqa: SLF001
            request, "us-east-1"
        )
        assert request["modelId"] == prompt_context.arn
        assert request["modelId"].endswith(":1"), "the resolved version is kept"
