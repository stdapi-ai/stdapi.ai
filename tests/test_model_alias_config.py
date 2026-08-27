"""Model aliases that carry configuration alongside their target model.

``MODEL_ALIASES`` accepts either a plain target model ID
or an object carrying a service tier, a guardrail, request metadata and extra
model parameters. Those values resolve as one chain -- request, then alias,
then general configuration -- with the security- and cost-bearing knobs gated
by their ``*_allow_*_override`` setting.

Ref: https://stdapi.ai/operations_configuration/#model-aliases
     stdapi/config.py:ModelAliasConfig
     stdapi/aws_bedrock.py:resolve_service_tier
     stdapi/models/__init__.py:_populate_model_aliases
"""

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError
from pydantic_core import from_json

from stdapi import aws_bedrock, models
from stdapi.aws_bedrock import (
    GUARDRAIL_CONFIG_VAR,
    GUARDRAIL_TRACE_VAR,
    MODEL_ALIAS_OVERLAY_RESOLVED_VAR,
    MODEL_ALIAS_OVERLAY_VAR,
    PERFORMANCE_CONFIG_VAR,
    build_alias_overlay,
    get_extra_model_parameters,
    resolve_service_tier,
    set_guardrail_configuration,
    set_inference_configuration,
)
from stdapi.config import SETTINGS, ModelAliasConfig, _Settings
from stdapi.exceptions import ServerError
from stdapi.models import (
    MANTLE_SERVICE,
    MODEL_ALIAS_OVERLAYS,
    MODEL_ALIASES,
    ModelDetails,
)
from stdapi.models.chat._adapters import _openai_common
from stdapi.models.chat._adapters import _openai_responses as openai_responses_adapter
from stdapi.models.chat._mantle import _convert as mantle_convert
from stdapi.monitoring import REQUEST_ID, EventLog
from stdapi.routes import openai_chat_completions, openai_responses
from stdapi.routes._moderation import apply_request_moderation
from stdapi.types import BaseModelRequestWithExtra
from stdapi.types.openai import RequestModeration
from stdapi.types.openai_chat_completions import ChatCompletion
from stdapi.types.openai_chat_completions import (
    CompletionCreateParams as ChatCompletionCreateParams,
)
from stdapi.types.openai_completions import CompletionCreateParams
from stdapi.types.openai_responses import Response, ResponseCreateParams

if TYPE_CHECKING:
    from collections.abc import Generator

    from openai import OpenAI
    from starlette.testclient import TestClient

#: Aliases are a server-side configuration feature, exercised in-process.
pytestmark = pytest.mark.local

#: Target model of every alias under test.
_TARGET = "amazon.nova-micro-v1:0"

#: Alias name used by the resolution tests.
_ALIAS = "test-configured-alias"

#: Bedrock Prompt Management ARN used by the prompt-template resolution test.
_PROMPT_ARN = "arn:aws:bedrock:us-east-1:123456789012:prompt/ABCDE12345"


def _headers(**values: str) -> dict[str, str]:
    """Return request headers for :func:`set_guardrail_configuration`.

    Args:
        **values: Header values keyed by their short name.

    Returns:
        Header mapping using the Bedrock guardrail header names.
    """
    names = {
        "identifier": "X-Amzn-Bedrock-GuardrailIdentifier",
        "version": "X-Amzn-Bedrock-GuardrailVersion",
    }
    return {names[key]: value for key, value in values.items()}


@pytest.fixture
def alias_overlay() -> Generator[None]:
    """Reset the per-request alias overlay around each test."""
    token = MODEL_ALIAS_OVERLAY_VAR.set(None)
    resolved = MODEL_ALIAS_OVERLAY_RESOLVED_VAR.set(False)
    yield
    MODEL_ALIAS_OVERLAY_RESOLVED_VAR.reset(resolved)
    MODEL_ALIAS_OVERLAY_VAR.reset(token)


def _settings(aliases: dict[str, Any], **overrides: Any) -> _Settings:  # noqa: ANN401
    """Load the settings with *aliases* as ``MODEL_ALIASES`` would be parsed.

    Mantle preferences are cleared unless a caller sets them: an alias guardrail
    is refused alongside them, which would make every guardrail case here fail
    for a reason that is not about aliases.

    Args:
        aliases: Raw alias map, in either supported form.
        **overrides: Further settings to apply.

    Returns:
        The validated settings.
    """
    overrides.setdefault("aws_bedrock_mantle_preferred_models", [])
    return _Settings(model_aliases=aliases, **overrides)


def _chat_request(**fields: Any) -> ChatCompletionCreateParams:  # noqa: ANN401
    """Build a minimal Chat Completions request.

    Args:
        **fields: Extra request fields to set.

    Returns:
        The validated request.
    """
    return ChatCompletionCreateParams.model_validate(
        {"model": _TARGET, "messages": [{"role": "user", "content": "hello"}], **fields}
    )


def _start_event() -> EventLog:
    """Return a startup event log, as the lifespan passes to the model refresh."""
    return EventLog(
        type="start",
        level="info",
        date=SETTINGS.now(),
        server_id="test",
        server_version="test",
    )


def _apply(config: ModelAliasConfig) -> None:
    """Resolve *config* into an overlay and apply it to the current request.

    Args:
        config: Alias configuration to apply.
    """
    aws_bedrock.apply_alias_overlay(build_alias_overlay(_ALIAS, config))


class TestModelAliasSettings:
    """``MODEL_ALIASES`` accepts both the string and the object form.

    Both forms must validate; the object form is checked at startup so a typo
    fails the deployment instead of every request.

    Ref: stdapi/config.py:ModelAliasConfig
         stdapi/config.py:_Settings.model_aliases
    """

    def test_string_form_is_preserved(self) -> None:
        """An alias mapped to a bare model ID stays a string."""
        settings = _settings({"my-model": _TARGET})
        assert settings.model_aliases == {"my-model": _TARGET}

    def test_object_form_is_validated(self) -> None:
        """An alias mapped to an object becomes a ``ModelAliasConfig``."""
        settings = _settings(
            {
                "my-model": {
                    "model": _TARGET,
                    "service_tier": "flex",
                    "guardrail_id": "gr-1",
                    "guardrail_version": "DRAFT",
                    "metadata": {"team": "research"},
                    "extra_params": {"temperature": 0.1},
                }
            }
        )
        config = settings.model_aliases["my-model"]
        assert isinstance(config, ModelAliasConfig)
        assert config.model == _TARGET
        assert config.service_tier == "flex"
        assert config.guardrail_identifier == "gr-1"
        assert config.metadata == {"team": "research"}
        assert config.extra_params == {"temperature": 0.1}

    def test_both_forms_mix_in_one_map(self) -> None:
        """A deployment may configure some aliases plainly and others with configuration."""
        settings = _settings({"plain": _TARGET, "rich": {"model": _TARGET}})
        assert settings.model_aliases["plain"] == _TARGET
        assert isinstance(settings.model_aliases["rich"], ModelAliasConfig)

    def test_unknown_key_fails_startup(self) -> None:
        """A misspelled alias key is rejected, naming the offending key."""
        with pytest.raises(ValidationError) as error:
            _settings({"rich": {"model": _TARGET, "servic_tier": "flex"}})
        message = str(error.value)
        assert "servic_tier" in message
        assert "Extra inputs are not permitted" in message
        # The union does not also report the string branch: the error stays readable.
        assert "1 validation error" in message

    def test_target_model_is_required(self) -> None:
        """An alias carrying configuration but no target model fails startup."""
        with pytest.raises(ValidationError, match="model"):
            _settings({"rich": {"service_tier": "flex"}})

    def test_guardrail_needs_both_identifier_and_version(self) -> None:
        """A half-configured guardrail fails startup instead of being ignored."""
        with pytest.raises(ValidationError, match="guardrail_version"):
            _settings({"rich": {"model": _TARGET, "guardrail_id": "gr-1"}})

    def test_alias_guardrail_keeps_the_override_gate_closed(self) -> None:
        """An alias-only guardrail is still an operator guardrail: requests cannot override it.

        Without any guardrail configured the gate opens automatically, so an
        alias-borne guardrail must count as configuration for that rule.
        """
        settings = _settings(
            {
                "rich": {
                    "model": _TARGET,
                    "guardrail_id": "gr-1",
                    "guardrail_version": "1",
                }
            }
        )
        assert settings.aws_bedrock_allow_guardrail_override is False

    def test_no_guardrail_anywhere_opens_the_gate(self) -> None:
        """With no guardrail configured at all, request headers stay allowed."""
        settings = _settings({"plain": _TARGET})
        assert settings.aws_bedrock_allow_guardrail_override is True

    def test_guardrail_trace_requires_a_guardrail(self) -> None:
        """A trace level without a guardrail to trace fails startup."""
        with pytest.raises(ValidationError, match="guardrail_trace") as error:
            _settings({"rich": {"model": _TARGET, "guardrail_trace": "enabled"}})
        assert "requires guardrail_identifier" in str(error.value)

    def test_alias_guardrail_rejects_the_mantle_service_header(self) -> None:
        """A per-request Mantle transport cannot coexist with an alias guardrail.

        The header lets a caller pick the Mantle transport, where Amazon Bedrock
        Guardrails do not apply -- the same bypass the server-wide guardrail
        already blocks.

        Ref: stdapi/config.py:_Settings.aws_bedrock_mantle_service_header
        """
        with pytest.raises(ValidationError) as error:
            _Settings(
                aws_bedrock_mantle_service_header=True,
                model_aliases={
                    "rich": ModelAliasConfig(
                        model=_TARGET,
                        guardrail_identifier="gr-1",
                        guardrail_version="1",
                    )
                },
            )
        message = str(error.value)
        assert "aws_bedrock_mantle_service_header" in message
        assert "incompatible with Amazon Bedrock Guardrails" in message

    def test_invalid_extra_params_fail_at_startup(self) -> None:
        """An out-of-range inference parameter stops the deployment, not each request.

        ``extra_params`` are validated where the overlay is built, at startup,
        rather than eagerly in the settings model: the inference-parameter
        schema lives with the Bedrock request builder, which imports the
        settings and so cannot be imported by them.
        """
        with pytest.raises(ServerError, match=_ALIAS):
            build_alias_overlay(
                _ALIAS,
                ModelAliasConfig(model=_TARGET, extra_params={"temperature": -1}),
            )

    def test_valid_extra_params_build_an_overlay(self) -> None:
        """A provider-specific extra is not an inference parameter and stays free-form."""
        overlay = build_alias_overlay(
            _ALIAS, ModelAliasConfig(model=_TARGET, extra_params={"anything": "goes"})
        )
        assert overlay.model_params == {"anything": "goes"}


class TestServiceTierPrecedence:
    """Service tier resolves request, then alias, then general configuration.

    Ref: stdapi/aws_bedrock.py:resolve_service_tier
         stdapi/config.py:_Settings.aws_bedrock_allow_service_tier_override
    """

    @pytest.fixture(autouse=True)
    def _no_configured_tier(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Start from a server that configures no tier for the target model."""
        monkeypatch.setattr(SETTINGS, "default_model_service_tiers", {})

    @pytest.mark.usefixtures("alias_overlay")
    def test_request_beats_alias(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A request-supplied tier wins over the alias' tier by default."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_service_tier_override", True)
        _apply(ModelAliasConfig(model=_TARGET, service_tier="flex"))
        assert resolve_service_tier(_TARGET, "priority") == "priority"

    @pytest.mark.usefixtures("alias_overlay")
    def test_alias_beats_general_configuration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no request tier, the alias' tier wins over the per-model default."""
        monkeypatch.setattr(
            SETTINGS, "default_model_service_tiers", {_TARGET: "priority"}
        )
        _apply(ModelAliasConfig(model=_TARGET, service_tier="flex"))
        assert resolve_service_tier(_TARGET, None) == "flex"

    @pytest.mark.usefixtures("alias_overlay")
    def test_unset_falls_through_to_general_configuration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An alias that sets no tier leaves the per-model default in place."""
        monkeypatch.setattr(SETTINGS, "default_model_service_tiers", {_TARGET: "flex"})
        _apply(ModelAliasConfig(model=_TARGET))
        assert resolve_service_tier(_TARGET, None) == "flex"

    @pytest.mark.usefixtures("alias_overlay")
    def test_gate_closed_keeps_the_alias_tier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the override gate closed, a request cannot displace the alias' tier."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_service_tier_override", False)
        _apply(ModelAliasConfig(model=_TARGET, service_tier="flex"))
        assert resolve_service_tier(_TARGET, "priority") == "flex"

    def test_gate_closed_still_honors_an_unconfigured_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate only protects a configured tier: otherwise the request is honored."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_service_tier_override", False)
        assert resolve_service_tier(_TARGET, "priority") == "priority"

    def test_gate_closed_keeps_the_general_configuration_tier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the gate closed, a request cannot displace ``default_model_service_tiers``.

        This is the case the setting leads with: no alias involved, just the
        per-model tier the administrator configured server-wide.
        """
        monkeypatch.setattr(SETTINGS, "default_model_service_tiers", {_TARGET: "flex"})
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_service_tier_override", False)
        assert resolve_service_tier(_TARGET, "priority") == "flex"

    def test_default_preserves_the_previous_behaviour(self) -> None:
        """The gate ships open, so an existing deployment sees no change."""
        assert _Settings().aws_bedrock_allow_service_tier_override is True


class TestServiceTierEchoStaysTheRequestedTier:
    """The response echoes the tier the request asked for.

    The adapters translate only the request's own ``service_tier``: the alias
    and server-configured tiers -- and the tier header -- resolve where the
    Bedrock request is built. Resolving in the adapter would mask the header
    (it is not visible there), and a configured Bedrock-only ``reserved`` tier
    would be echoed into a response schema that does not define it, failing
    every request for that model.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
         stdapi/models/chat/_adapters/_openai_common.py:map_service_tier
         stdapi/models/chat/_adapters/_openai_responses.py:_build_response_object
    """

    def test_the_request_value_is_echoed(self) -> None:
        """The mapping is a plain request translation, with no configured layer."""
        assert _openai_common.map_service_tier("flex") == ("flex", "flex")
        assert _openai_common.map_service_tier("auto") == (None, "default")
        assert _openai_common.map_service_tier(None) == (None, None)

    def test_a_configured_reserved_tier_does_not_break_the_responses_echo(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model configured with the ``reserved`` tier still answers on ``/v1/responses``.

        ``reserved`` exists in Bedrock but not in the Responses ``service_tier``
        literal, so an echo reporting the configured tier would fail response
        validation on every request -- as a 500 non-streaming, mid-SSE streaming.
        """
        monkeypatch.setattr(
            SETTINGS, "default_model_service_tiers", {_TARGET: "reserved"}
        )
        response = openai_responses_adapter._build_response_object(  # noqa: SLF001
            response_id="resp_1",
            created_at=0.0,
            model_id=_TARGET,
            output_items=[],
            status="completed",
            incomplete_details=None,
            error=None,
            usage=None,
            request=ResponseCreateParams.model_validate(
                {"model": _TARGET, "input": "hello"}
            ),
        )
        assert response.service_tier is None


class TestGuardrailPrecedence:
    """An alias guardrail sits between the request headers and the global one.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails.html
         stdapi/aws_bedrock.py:apply_alias_overlay
         stdapi/config.py:_Settings.aws_bedrock_allow_guardrail_override
    """

    @pytest.fixture(autouse=True)
    def _guardrail_scope(self, monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
        """Isolate the guardrail and alias context variables per test."""
        overlay = MODEL_ALIAS_OVERLAY_VAR.set(None)
        guardrail = GUARDRAIL_CONFIG_VAR.set(None)  # type: ignore[arg-type]
        trace = GUARDRAIL_TRACE_VAR.set({})
        request_override = aws_bedrock.GUARDRAIL_REQUEST_OVERRIDE_VAR.set(False)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", None)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", None)
        yield
        aws_bedrock.GUARDRAIL_REQUEST_OVERRIDE_VAR.reset(request_override)
        GUARDRAIL_TRACE_VAR.reset(trace)
        GUARDRAIL_CONFIG_VAR.reset(guardrail)
        MODEL_ALIAS_OVERLAY_VAR.reset(overlay)

    def test_alias_guardrail_applies_without_a_request_guardrail(self) -> None:
        """A guardrail-bearing alias installs its guardrail for the request."""
        set_guardrail_configuration({})  # type: ignore[arg-type]
        _apply(
            ModelAliasConfig(
                model=_TARGET,
                guardrail_identifier="gr-alias",
                guardrail_version="1",
                guardrail_trace="enabled",
            )
        )
        assert GUARDRAIL_CONFIG_VAR.get() == {
            "guardrailIdentifier": "gr-alias",
            "guardrailVersion": "1",
            "trace": "enabled",
        }

    def test_alias_guardrail_beats_the_global_guardrail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The alias layer overrides the operator's server-wide guardrail."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", "gr-global")
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", "3")
        set_guardrail_configuration({})  # type: ignore[arg-type]
        _apply(
            ModelAliasConfig(
                model=_TARGET, guardrail_identifier="gr-alias", guardrail_version="1"
            )
        )
        assert GUARDRAIL_CONFIG_VAR.get()["guardrailIdentifier"] == "gr-alias"

    def test_request_guardrail_beats_the_alias_when_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the override gate open, request headers win over the alias."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", True)
        set_guardrail_configuration(_headers(identifier="gr-request", version="2"))  # type: ignore[arg-type]
        _apply(
            ModelAliasConfig(
                model=_TARGET, guardrail_identifier="gr-alias", guardrail_version="1"
            )
        )
        assert GUARDRAIL_CONFIG_VAR.get()["guardrailIdentifier"] == "gr-request"

    def test_request_guardrail_cannot_displace_the_alias_when_gated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the override gate closed, the alias' guardrail holds.

        This is the multi-tenant case: an alias is the operator's policy, and a
        client naming its own guardrail must not escape it.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", False)
        set_guardrail_configuration(_headers(identifier="gr-request", version="2"))  # type: ignore[arg-type]
        _apply(
            ModelAliasConfig(
                model=_TARGET, guardrail_identifier="gr-alias", guardrail_version="1"
            )
        )
        assert GUARDRAIL_CONFIG_VAR.get()["guardrailIdentifier"] == "gr-alias"

    def test_alias_without_a_guardrail_leaves_the_global_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An alias that sets no guardrail falls through to the configured one."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", "gr-global")
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", "3")
        set_guardrail_configuration({})  # type: ignore[arg-type]
        _apply(ModelAliasConfig(model=_TARGET))
        assert GUARDRAIL_CONFIG_VAR.get()["guardrailIdentifier"] == "gr-global"

    def test_moderation_reporting_follows_the_alias_guardrail(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``moderation`` reports on the guardrail that actually applies.

        The moderation parameter asks for the applied guardrail's assessments,
        so it must land on the alias' guardrail rather than replace it with the
        server-wide one -- which, with the override gate closed, would be a way
        to downgrade the safeguard the alias exists to enforce.

        Ref: stdapi/routes/_moderation.py:apply_request_moderation
             stdapi/aws_bedrock.py:resolve_guardrail_model
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", False)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", "gr-global")
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", "3")
        set_guardrail_configuration({})  # type: ignore[arg-type]
        _apply(
            ModelAliasConfig(
                model=_TARGET, guardrail_identifier="gr-alias", guardrail_version="1"
            )
        )
        apply_request_moderation(RequestModeration(model="omni-moderation-latest"))
        assert GUARDRAIL_CONFIG_VAR.get() == {
            "guardrailIdentifier": "gr-alias",
            "guardrailVersion": "1",
            "trace": "enabled_full",
        }


class TestExtraParametersPrecedence:
    """Extra model parameters merge request over alias over general configuration.

    Ref: stdapi/aws_bedrock.py:set_inference_configuration
         stdapi/aws_bedrock.py:get_extra_model_parameters
    """

    @pytest.mark.usefixtures("alias_overlay")
    def test_inference_parameters_chain(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Temperature comes from the request, top_p from the alias, max_tokens from settings."""
        monkeypatch.setattr(
            SETTINGS,
            "default_model_params",
            {_TARGET: {"temperature": 0.9, "top_p": 0.9, "max_tokens": 16}},
        )
        _apply(
            ModelAliasConfig(
                model=_TARGET, extra_params={"temperature": 0.5, "top_p": 0.2}
            )
        )
        config = set_inference_configuration(_TARGET, {}, temperature=0.1)
        assert config == {"temperature": 0.1, "topP": 0.2, "maxTokens": 16}

    @pytest.mark.usefixtures("alias_overlay")
    def test_provider_specific_extras_chain(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Alias extras reach ``additionalModelRequestFields`` and lose to request extras."""
        monkeypatch.setattr(
            SETTINGS,
            "default_model_params",
            {_TARGET: {"from_settings": 1, "shared": "settings"}},
        )
        _apply(
            ModelAliasConfig(
                model=_TARGET, extra_params={"from_alias": 2, "shared": "alias"}
            )
        )
        fields: dict[str, Any] = {}
        set_inference_configuration(_TARGET, fields, shared="request")
        assert fields == {"from_settings": 1, "from_alias": 2, "shared": "request"}

    @pytest.mark.usefixtures("alias_overlay")
    def test_non_chat_routes_see_the_alias_parameters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shared non-chat helper applies the same three-layer merge."""
        monkeypatch.setattr(
            SETTINGS, "default_model_params", {_TARGET: {"a": 1, "b": 1}}
        )
        _apply(ModelAliasConfig(model=_TARGET, extra_params={"b": 2, "c": 2}))
        request = BaseModelRequestWithExtra(c=3)  # type: ignore[call-arg]
        assert get_extra_model_parameters(_TARGET, request) == {"a": 1, "b": 2, "c": 3}

    @pytest.mark.usefixtures("alias_overlay")
    def test_alias_parameters_do_not_leak_to_other_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without an alias overlay, only the general configuration applies."""
        monkeypatch.setattr(SETTINGS, "default_model_params", {_TARGET: {"a": 1}})
        fields: dict[str, Any] = {}
        set_inference_configuration(_TARGET, fields)
        assert fields == {"a": 1}


class TestAliasResolution:
    """Resolving an alias installs its configuration for the rest of the request.

    Ref: stdapi/models/__init__.py:validate_model
         stdapi/models/__init__.py:_populate_model_aliases
    """

    @pytest.fixture
    def catalog(self, monkeypatch: pytest.MonkeyPatch) -> ModelDetails:
        """Register the alias, its overlay and a target model in the catalog.

        Returns:
            The catalog entry the alias points at.
        """
        details = ModelDetails(
            id=_TARGET,
            name="Target",
            provider="Amazon",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
            regions=["us-east-1"],
        )
        monkeypatch.setitem(models._MODELS, _TARGET, details)  # noqa: SLF001
        monkeypatch.setitem(MODEL_ALIASES, _ALIAS, _TARGET)
        return details

    @pytest.mark.usefixtures("catalog", "request_log", "alias_overlay")
    async def test_validate_model_installs_the_overlay(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Requesting the alias makes its service tier the effective one."""
        monkeypatch.setitem(
            MODEL_ALIAS_OVERLAYS,
            _ALIAS,
            build_alias_overlay(
                _ALIAS, ModelAliasConfig(model=_TARGET, service_tier="flex")
            ),
        )
        model = await models.validate_model(_ALIAS)
        assert model.id == _TARGET
        assert resolve_service_tier(_TARGET, None) == "flex"

    @pytest.mark.usefixtures("catalog", "request_log", "alias_overlay")
    async def test_target_model_id_carries_no_overlay(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Requesting the target model directly leaves the alias configuration off.

        The overlay table is non-empty and pre-seeded with another alias'
        overlay, so a resolution that finds nothing has to clear it rather than
        inherit whatever the context happened to carry.
        """
        monkeypatch.setitem(
            MODEL_ALIAS_OVERLAYS,
            "another-alias",
            build_alias_overlay(
                _ALIAS, ModelAliasConfig(model=_TARGET, service_tier="flex")
            ),
        )
        MODEL_ALIAS_OVERLAY_VAR.set(
            build_alias_overlay(
                _ALIAS, ModelAliasConfig(model=_TARGET, service_tier="priority")
            )
        )
        await models.validate_model(_TARGET)
        assert MODEL_ALIAS_OVERLAY_VAR.get() is None

    @pytest.mark.usefixtures("catalog", "request_log", "alias_overlay")
    async def test_a_second_resolution_does_not_clear_the_overlay(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the model the request named installs an overlay.

        A route may resolve a second model within the same request -- a prompt
        template's model, a built-in tool's model. That resolution must not
        drop the configuration the request's own alias installed.
        """
        monkeypatch.setitem(
            MODEL_ALIAS_OVERLAYS,
            _ALIAS,
            build_alias_overlay(
                _ALIAS, ModelAliasConfig(model=_TARGET, service_tier="flex")
            ),
        )
        await models.validate_model(_ALIAS)
        await models.validate_model(_TARGET)
        assert resolve_service_tier(_TARGET, None) == "flex"

    @pytest.mark.usefixtures("catalog", "request_log", "alias_overlay")
    async def test_a_latest_tagged_alias_still_installs_its_overlay(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An alias asked for as ``<alias>:latest`` is served with its own guardrail.

        An Ollama client names the current version of a model with a
        ``:latest`` tag, and the catalog lookup falls back to the untagged
        name, so the alias resolves and is served. Its configuration has to
        follow the same fallback: looked up on the tagged name alone it finds
        nothing, and the target is then served with neither the alias'
        guardrail nor its parameters, silently.

        Ref: https://docs.ollama.com/api/chat
             stdapi/models/__init__.py:_alias_overlay
        """
        monkeypatch.setitem(
            MODEL_ALIAS_OVERLAYS,
            _ALIAS,
            build_alias_overlay(
                _ALIAS,
                ModelAliasConfig(
                    model=_TARGET,
                    service_tier="flex",
                    guardrail_identifier="gr-alias",
                    guardrail_version="1",
                ),
            ),
        )
        guardrail = GUARDRAIL_CONFIG_VAR.set(None)  # type: ignore[arg-type]
        try:
            model = await models.validate_model(f"{_ALIAS}:latest")
            assert model.id == _TARGET
            assert resolve_service_tier(_TARGET, None) == "flex"
            assert GUARDRAIL_CONFIG_VAR.get()["guardrailIdentifier"] == "gr-alias"
        finally:
            GUARDRAIL_CONFIG_VAR.reset(guardrail)

    def test_populate_builds_overlays_for_configured_aliases(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The alias table and the overlay table are rebuilt together.

        Both forms end up in ``MODEL_ALIASES`` so every existing consumer --
        the model listing, the pricing route -- keeps seeing a plain target ID.
        """
        monkeypatch.setattr(models, "MODEL_ALIASES", {})
        monkeypatch.setattr(models, "MODEL_ALIAS_OVERLAYS", {})
        monkeypatch.setattr(models, "_GLOBAL_MODEL_REGISTRY", [])
        monkeypatch.setattr(
            SETTINGS,
            "model_aliases",
            {
                "plain": _TARGET,
                "rich": ModelAliasConfig(model=_TARGET, service_tier="flex"),
            },
        )
        models._populate_model_aliases({})  # noqa: SLF001
        assert models.MODEL_ALIASES == {"plain": _TARGET, "rich": _TARGET}
        assert set(models.MODEL_ALIAS_OVERLAYS) == {"rich"}
        assert models.MODEL_ALIAS_OVERLAYS["rich"].service_tier == "flex"

    @pytest.mark.usefixtures("catalog")
    def test_chat_request_is_checked_by_the_alias_guardrail(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A chat request naming the alias is generated under the alias' guardrail.

        The route resolves the model before reading the request's own guardrail
        selection, so the layer a client cannot override is the one in force
        when generation starts.

        Ref: stdapi/routes/openai_chat_completions.py:create_chat_completion
             stdapi/aws_bedrock.py:apply_alias_overlay
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", False)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", "gr-global")
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", "3")
        monkeypatch.setitem(
            MODEL_ALIAS_OVERLAYS,
            _ALIAS,
            build_alias_overlay(
                _ALIAS,
                ModelAliasConfig(
                    model=_TARGET,
                    guardrail_identifier="gr-alias",
                    guardrail_version="1",
                ),
            ),
        )
        seen: list[Any] = []

        class _StubChatModel:
            """Chat model recording the guardrail in force when it is called."""

            async def create_completion(
                self,
                request: Any,  # noqa: ANN401
                completion_id: str,
                created: int,
            ) -> ChatCompletion:
                """Record the guardrail and return a canned completion."""
                seen.append(GUARDRAIL_CONFIG_VAR.get(None))
                return ChatCompletion.model_validate(
                    {
                        "id": completion_id,
                        "created": created,
                        "model": request.model,
                        "object": "chat.completion",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "hi"},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                )

        monkeypatch.setattr(
            openai_chat_completions,
            "get_chat_model",
            lambda _model_id: _StubChatModel(),
        )
        response = app_client.post(
            "/v1/chat/completions",
            json={
                "model": _ALIAS,
                "messages": [{"role": "user", "content": "hello"}],
                "moderation": {"model": "omni-moderation-latest"},
            },
            headers=_headers(identifier="gr-request", version="9"),
        )
        assert response.status_code == 200, response.text
        assert seen == [
            {
                "guardrailIdentifier": "gr-alias",
                "guardrailVersion": "1",
                "trace": "enabled_full",
            }
        ]

    @pytest.mark.usefixtures("catalog")
    def test_responses_request_is_checked_by_the_alias_guardrail(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Responses route resolves the model before the request's moderation.

        Same ordering as its Chat Completions twin: ``moderation`` reports on
        the guardrail that actually applies, which the alias may have replaced.

        Ref: stdapi/routes/openai_responses.py:create_response
             stdapi/aws_bedrock.py:apply_alias_overlay
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", False)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", "gr-global")
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", "3")
        monkeypatch.setitem(
            MODEL_ALIAS_OVERLAYS,
            _ALIAS,
            build_alias_overlay(
                _ALIAS,
                ModelAliasConfig(
                    model=_TARGET,
                    guardrail_identifier="gr-alias",
                    guardrail_version="1",
                ),
            ),
        )
        seen: list[Any] = []

        class _StubChatModel:
            """Chat model recording the guardrail in force when it is called."""

            @staticmethod
            def native_store_supported() -> bool:
                """Use the local response store, like a classic Bedrock model."""
                return False

            async def create_response(
                self,
                request: Any,  # noqa: ANN401
                response_id: str,
                created_at: float,
                moderation_builder: Any = None,  # noqa: ANN401
            ) -> Response:
                """Record the guardrail and return a canned response."""
                seen.append(GUARDRAIL_CONFIG_VAR.get(None))
                return Response.model_validate(
                    {
                        "id": response_id,
                        "created_at": created_at,
                        "model": request.model,
                        "object": "response",
                        "status": "completed",
                        "output": [],
                        "parallel_tool_calls": True,
                        "tool_choice": "auto",
                        "tools": [],
                    }
                )

        monkeypatch.setattr(
            openai_responses, "get_chat_model", lambda _model_id: _StubChatModel()
        )
        response = app_client.post(
            "/v1/responses",
            json={
                "model": _ALIAS,
                "input": "hello",
                "store": False,
                "moderation": {"model": "omni-moderation-latest"},
            },
            headers=_headers(identifier="gr-request", version="9"),
        )
        assert response.status_code == 200, response.text
        assert seen == [
            {
                "guardrailIdentifier": "gr-alias",
                "guardrailVersion": "1",
                "trace": "enabled_full",
            }
        ]


class TestMantleServedModels:
    """What an alias can and cannot configure on a Bedrock Mantle-served model.

    Mantle requests go to an OpenAI-compatible endpoint rather than the Bedrock
    Converse/InvokeModel APIs. Amazon Bedrock Guardrails cannot be attached to
    them, and that is fatal at startup. The service tier is not applied there
    either -- the request's own value is forwarded as sent -- but that is silent,
    the way an alias' ``metadata`` and ``extra_params`` already are.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
         stdapi/models/__init__.py:_mantle_guardrail_aliases
         stdapi/models/chat/_mantle/_convert.py:chat_completions_payload
    """

    @staticmethod
    def _catalog(service: str | None) -> dict[str, ModelDetails]:
        """Build a one-model catalog served by *service*.

        Args:
            service: Value of ``ModelDetails.service``.

        Returns:
            The catalog, keyed by model ID.
        """
        details = ModelDetails(
            id=_TARGET,
            name="Target",
            provider="Amazon",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
            regions=["us-east-1"],
        )
        if service is not None:
            details.service = service
        return {_TARGET: details}

    def test_a_guardrail_alias_on_a_mantle_model_is_detected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An alias guardrail pointing at a Mantle-served model is reported."""
        monkeypatch.setattr(
            SETTINGS,
            "model_aliases",
            {
                _ALIAS: ModelAliasConfig(
                    model=_TARGET, guardrail_identifier="gr-1", guardrail_version="1"
                )
            },
        )
        assert models._mantle_guardrail_aliases(  # noqa: SLF001
            self._catalog(MANTLE_SERVICE)
        ) == [_ALIAS]

    def test_a_guardrail_alias_on_a_runtime_model_is_fine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same alias on a bedrock-runtime model raises nothing."""
        monkeypatch.setattr(
            SETTINGS,
            "model_aliases",
            {
                _ALIAS: ModelAliasConfig(
                    model=_TARGET, guardrail_identifier="gr-1", guardrail_version="1"
                )
            },
        )
        assert models._mantle_guardrail_aliases(self._catalog(None)) == []  # noqa: SLF001

    def test_a_tier_only_alias_on_a_mantle_model_is_fine(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only a guardrail is fatal: a tier is ignored, not a safeguard gap."""
        monkeypatch.setattr(
            SETTINGS,
            "model_aliases",
            {_ALIAS: ModelAliasConfig(model=_TARGET, service_tier="flex")},
        )
        assert (
            models._mantle_guardrail_aliases(  # noqa: SLF001
                self._catalog(MANTLE_SERVICE)
            )
            == []
        )

    def test_startup_fails_naming_the_alias(self) -> None:
        """Startup stops rather than serve unfiltered content under a guardrail name."""
        with pytest.raises(ServerError, match=_ALIAS) as error:
            models._reject_mantle_guardrail_aliases([_ALIAS], _start_event())  # noqa: SLF001
        assert "Bedrock Mantle" in str(error.value)

    def test_a_later_refresh_only_warns(self, request_log: dict[str, Any]) -> None:
        """A running deployment is not killed by a lazy catalog refresh.

        The mismatch is still reported: the request log is raised to a warning
        naming the offending alias.
        """
        models._reject_mantle_guardrail_aliases([_ALIAS], None)  # noqa: SLF001
        assert request_log["level"] == "warning"
        assert any(_ALIAS in str(detail) for detail in request_log["error_detail"])

    @pytest.mark.usefixtures("alias_overlay")
    @pytest.mark.parametrize("gate", [True, False])
    async def test_no_configured_tier_reaches_the_mantle_chat_payload(
        self, monkeypatch: pytest.MonkeyPatch, gate: bool
    ) -> None:
        """Neither the alias' tier nor the server-wide one is sent to Mantle.

        Both layers are resolved where the Bedrock request is built, which a
        Mantle request never goes through. Sending one here would forward a
        Bedrock-only tier name (``reserved``) that this surface does not
        define, and would beat the tier header even with the gate open.
        """
        monkeypatch.setattr(SETTINGS, "default_model_service_tiers", {_TARGET: "flex"})
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_service_tier_override", gate)
        _apply(ModelAliasConfig(model=_TARGET, service_tier="reserved"))
        payload = await mantle_convert.chat_completions_payload(
            _chat_request(), _TARGET
        )
        assert "service_tier" not in payload

    @pytest.mark.usefixtures("alias_overlay")
    async def test_the_request_tier_still_passes_through_to_mantle(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The client's own tier reaches Mantle unchanged, configuration or not."""
        monkeypatch.setattr(SETTINGS, "default_model_service_tiers", {_TARGET: "flex"})
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_service_tier_override", False)
        _apply(ModelAliasConfig(model=_TARGET, service_tier="flex"))
        payload = await mantle_convert.chat_completions_payload(
            _chat_request(service_tier="priority"), _TARGET
        )
        assert payload["service_tier"] == "priority"

    @pytest.mark.usefixtures("alias_overlay")
    async def test_the_mantle_responses_payload_follows_the_same_contract(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Responses passthrough drops no tier and adds none either.

        ``reserved`` has no place in the Responses ``service_tier`` literal, so
        a configured tier landing here would fail every request for the model.
        """
        monkeypatch.setattr(
            SETTINGS, "default_model_service_tiers", {_TARGET: "reserved"}
        )
        _apply(ModelAliasConfig(model=_TARGET, service_tier="reserved"))
        payload, _region = await mantle_convert.responses_payload(
            ResponseCreateParams.model_validate({"model": _TARGET, "input": "hello"}),
            _TARGET,
        )
        assert "service_tier" not in payload
        payload, _region = await mantle_convert.responses_payload(
            ResponseCreateParams.model_validate(
                {"model": _TARGET, "input": "hello", "service_tier": "flex"}
            ),
            _TARGET,
        )
        assert payload["service_tier"] == "flex"

    @pytest.mark.usefixtures("alias_overlay")
    async def test_the_completions_payload_carries_only_the_request_tier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``/v1/completions`` forwards the client's tier and no configured one.

        ``service_tier`` is part of that surface's request, so the legacy route
        copies it into the Chat Completions payload it builds -- as the two
        other Mantle payload builders do for their own requests.
        """
        monkeypatch.setattr(SETTINGS, "default_model_service_tiers", {_TARGET: "flex"})
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_service_tier_override", False)
        _apply(ModelAliasConfig(model=_TARGET, service_tier="flex"))
        payload = await mantle_convert.text_completion_as_chat_payload(
            CompletionCreateParams.model_validate(
                {"model": _TARGET, "prompt": "hello", "service_tier": "priority"}
            ),
            _TARGET,
        )
        assert payload["service_tier"] == "priority"
        payload = await mantle_convert.text_completion_as_chat_payload(
            CompletionCreateParams.model_validate(
                {"model": _TARGET, "prompt": "hello"}
            ),
            _TARGET,
        )
        assert "service_tier" not in payload


@pytest.mark.usefixtures("alias_overlay", "usage_scope", "request_log")
class TestAliasConfigurationReachesTheBackendRequest:
    """The resolved alias configuration lands in the outgoing model request.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         stdapi/models/__init__.py:_build_invoke_kwargs
         stdapi/models/__init__.py:ModelBase._prepare_converse_request_for_region
    """

    @pytest.fixture(autouse=True)
    def _stubs(self, monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
        """Resolve model IDs and content blocks without touching the catalog or AWS."""

        async def resolve(model_id: str, _region: str, **_kwargs: object) -> str:
            """Return the model ID unchanged."""
            return model_id

        async def resolve_blocks(_region: str, **_kwargs: object) -> None:
            """Skip content-block resolution."""

        monkeypatch.setattr(models, "resolve_routed_model_id", resolve)
        monkeypatch.setattr(
            models, "resolve_all_bedrock_content_blocks", resolve_blocks
        )
        token = REQUEST_ID.set("req-alias")
        yield
        REQUEST_ID.reset(token)

    async def test_invoke_request_carries_the_alias_tier_and_metadata(self) -> None:
        """An alias' tier and metadata reach the ``InvokeModel`` request."""
        _apply(
            ModelAliasConfig(
                model=_TARGET, service_tier="flex", metadata={"team": "research"}
            )
        )
        kwargs = await models._build_invoke_kwargs(  # noqa: SLF001
            _TARGET, {"prompt": "hi"}, "us-east-1", inference_profile=False
        )
        assert kwargs["serviceTier"] == "flex"
        assert from_json(kwargs["requestMetadata"])["team"] == "research"

    async def test_converse_request_carries_the_alias_tier_and_metadata(self) -> None:
        """An alias' tier and metadata reach the ``Converse`` request."""
        _apply(
            ModelAliasConfig(
                model=_TARGET, service_tier="flex", metadata={"team": "research"}
            )
        )
        request: dict[str, Any] = {}
        await models.ModelBase(_TARGET)._prepare_converse_request_for_region(  # noqa: SLF001
            request,  # type: ignore[arg-type]
            "us-east-1",
        )
        assert request["serviceTier"] == {"type": "flex"}
        assert request["requestMetadata"]["team"] == "research"
        assert request["requestMetadata"]["stdapi-ai.request_id"] == "req-alias"

    async def test_the_tier_header_wins_over_a_configured_tier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the gate open, ``X-Amzn-Bedrock-Service-Tier`` displaces a configured tier.

        The header and the ``service_tier`` body parameter are two spellings of
        the same request-level choice, so both pass through the same override
        gate where the Bedrock request is built. A configured tier must not
        mask the header -- that would silently change latency and cost for
        existing deployments that select tiers per request.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_service_tier_override", True)
        monkeypatch.setattr(SETTINGS, "default_model_service_tiers", {_TARGET: "flex"})
        token = PERFORMANCE_CONFIG_VAR.set((None, "priority"))
        try:
            request: dict[str, Any] = {}
            await models.ModelBase(_TARGET)._prepare_converse_request_for_region(  # noqa: SLF001
                request,  # type: ignore[arg-type]
                "us-east-1",
            )
        finally:
            PERFORMANCE_CONFIG_VAR.reset(token)
        assert request["serviceTier"] == {"type": "priority"}

    async def test_the_gate_closed_discards_the_tier_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the gate closed, the header cannot displace the configured tier."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_service_tier_override", False)
        monkeypatch.setattr(SETTINGS, "default_model_service_tiers", {_TARGET: "flex"})
        token = PERFORMANCE_CONFIG_VAR.set((None, "priority"))
        try:
            request: dict[str, Any] = {}
            await models.ModelBase(_TARGET)._prepare_converse_request_for_region(  # noqa: SLF001
                request,  # type: ignore[arg-type]
                "us-east-1",
            )
        finally:
            PERFORMANCE_CONFIG_VAR.reset(token)
        assert request["serviceTier"] == {"type": "flex"}

    async def test_request_metadata_wins_over_the_alias(self) -> None:
        """A client-supplied metadata key overrides the alias' value for that key."""
        _apply(
            ModelAliasConfig(
                model=_TARGET, metadata={"team": "research", "tier": "internal"}
            )
        )
        request: dict[str, Any] = {"requestMetadata": {"team": "support"}}
        await models.ModelBase(_TARGET)._prepare_converse_request_for_region(  # noqa: SLF001
            request,  # type: ignore[arg-type]
            "us-east-1",
        )
        assert request["requestMetadata"]["team"] == "support"
        assert request["requestMetadata"]["tier"] == "internal"

    async def test_a_prompt_template_keeps_the_alias_configuration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A prompt ARN resolves its own model without dropping the alias' configuration.

        The Responses ``prompt`` path resolves the prompt's model as a second
        model of the same request. That model is a plain ID, so a resolution
        that installed its (absent) configuration would silently strip the
        tier, guardrail and metadata the named alias carries.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-management.html
             stdapi/models/__init__.py:resolve_bedrock_prompt
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_prompt_arn", True)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1"])
        monkeypatch.setitem(
            models._MODELS,  # noqa: SLF001
            _TARGET,
            ModelDetails(
                id=_TARGET,
                name="Target",
                provider="Amazon",
                input_modalities=["TEXT"],
                output_modalities=["TEXT"],
                regions=["us-east-1"],
            ),
        )
        monkeypatch.setitem(MODEL_ALIASES, _ALIAS, _TARGET)
        monkeypatch.setitem(
            MODEL_ALIAS_OVERLAYS,
            _ALIAS,
            build_alias_overlay(
                _ALIAS,
                ModelAliasConfig(
                    model=_TARGET, service_tier="flex", metadata={"team": "research"}
                ),
            ),
        )

        async def prompt_model_id(*_args: object, **_kwargs: object) -> str:
            """Return the model the stored prompt variant is bound to."""
            return _TARGET

        monkeypatch.setattr(models, "_get_prompt_model_id", prompt_model_id)

        await models.validate_model(_ALIAS)
        prompt = await models.resolve_bedrock_prompt(_PROMPT_ARN, None)
        assert prompt.model_id == _TARGET

        request: dict[str, Any] = {}
        await models.ModelBase(_TARGET)._prepare_converse_request_for_region(  # noqa: SLF001
            request,  # type: ignore[arg-type]
            "us-east-1",
        )
        assert request["serviceTier"] == {"type": "flex"}
        assert request["requestMetadata"]["team"] == "research"


@pytest.mark.slow
@pytest.mark.xdist_group("moderations_guardrail")
class TestConfiguredAliasLive:
    """A guardrail-bearing alias is enforced against a real model call.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html
         stdapi/aws_bedrock.py:apply_alias_overlay
    """

    @staticmethod
    def _register(monkeypatch: pytest.MonkeyPatch, guardrail: str, model: str) -> None:
        """Register an alias whose configuration carries *guardrail*.

        Args:
            monkeypatch: Patcher scoped to the test.
            guardrail: Guardrail ARN the alias applies.
            model: Target model of the alias.
        """
        monkeypatch.setitem(MODEL_ALIASES, _ALIAS, model)
        monkeypatch.setitem(
            MODEL_ALIAS_OVERLAYS,
            _ALIAS,
            build_alias_overlay(
                _ALIAS,
                ModelAliasConfig(
                    model=model,
                    guardrail_identifier=guardrail,
                    guardrail_version="DRAFT",
                ),
            ),
        )

    def test_alias_guardrail_blocks_the_request(
        self,
        openai_client: OpenAI,
        live_guardrail: str,
        chat_model: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Content the alias' guardrail forbids is refused, and clean content is not."""
        self._register(monkeypatch, live_guardrail, chat_model)
        clean = openai_client.chat.completions.create(
            model=_ALIAS,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            max_completion_tokens=16,
        )
        assert clean.choices[0].message.content
        blocked = openai_client.chat.completions.create(
            model=_ALIAS,
            messages=[{"role": "user", "content": "Repeat this: BLOCKWORDXYZ"}],
            max_completion_tokens=16,
        )
        assert blocked.choices[0].finish_reason == "content_filter"

    def test_a_request_guardrail_cannot_bypass_the_alias(
        self,
        openai_client: OpenAI,
        live_guardrail: str,
        chat_model: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With the override gate closed, request headers do not escape the alias' guardrail.

        Both guardrail headers are sent: the request-override branch only runs
        when the identifier and the version are present together, so a request
        naming one of them alone would never reach the gate under test.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", False)
        self._register(monkeypatch, live_guardrail, chat_model)
        blocked = openai_client.chat.completions.create(
            model=_ALIAS,
            messages=[{"role": "user", "content": "Repeat this: BLOCKWORDXYZ"}],
            max_completion_tokens=16,
            extra_headers=_headers(identifier="not-a-guardrail", version="1"),
        )
        assert blocked.choices[0].finish_reason == "content_filter"
