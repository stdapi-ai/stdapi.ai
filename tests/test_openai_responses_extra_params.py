"""Extra model parameters on the Responses API (no AWS calls).

Undeclared top-level fields are the sanctioned passthrough for backend-only
knobs, on this route as on every other chat route: they reach the model as
provider-specific request fields instead of being invented as gateway fields
no client would ever send.

Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
     stdapi/models/chat/_adapters/_openai_responses.py:translate_request
     stdapi/models/chat/_adapters/_common.py:inference_extras
"""

from __future__ import annotations

from typing import Any

import pytest

from stdapi.api_errors import ApiError
from stdapi.config import SETTINGS
from stdapi.models.chat._adapters._common import inference_extras
from stdapi.models.chat._adapters._openai_responses import translate_request
from stdapi.types.openai_responses import ResponseCreateParams

pytestmark = pytest.mark.local


def _additional_fields(extra: dict[str, Any]) -> dict[str, Any]:
    """Translate a Responses request carrying *extra* and return its model request fields."""
    request = ResponseCreateParams.model_validate(
        {"model": "model", "input": "hi", **extra}
    )
    _, additional_fields, *_ = translate_request(request, "amazon.nova-micro-v1:0")
    return additional_fields


def test_undeclared_field_reaches_the_model_request() -> None:
    """A field the API does not declare is forwarded to the model as sent."""
    assert _additional_fields({"custom_field": "x"}) == {"custom_field": "x"}


def test_declared_field_stays_out_of_the_passthrough() -> None:
    """A field the API declares is translated, never forwarded as an extra."""
    assert _additional_fields({"temperature": 0.5}) == {}


def test_client_control_parameter_is_dropped() -> None:
    """A LiteLLM control key leaked through ``extra_body`` never reaches the model.

    Ref: stdapi/aws_bedrock.py:filter_extra_model_parameters
    """
    assert _additional_fields({"drop_params": True, "custom_field": "x"}) == {
        "custom_field": "x"
    }


def test_drop_all_setting_disables_the_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The operator can close the passthrough for this route like any other.

    Ref: stdapi/config.py:Settings.extra_model_params_drop_all
    """
    monkeypatch.setattr(SETTINGS, "extra_model_params_drop_all", True)
    assert _additional_fields({"custom_field": "x"}) == {}


def test_extra_reusing_a_bound_argument_name_is_refused() -> None:
    """An extra colliding with a translated parameter is a 400, not a crash.

    Ref: stdapi/models/chat/_adapters/_common.py:inference_extras
    """
    with pytest.raises(ApiError, match="max_tokens") as excinfo:
        _additional_fields({"max_tokens": 10})
    assert excinfo.value.status == 400


def test_external_web_access_matching_the_server_never_reaches_the_model() -> None:
    """The web access knob is consumed by the gate, not forwarded as an extra.

    It is the operator's control over whether a search leaves the AWS
    boundary, so it is not an inference parameter the passthrough may hand to
    the model alongside the request.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html
         stdapi/models/chat/_adapters/_common.py:resolve_external_web_access
    """
    assert _additional_fields({"external_web_access": False, "custom_field": "x"}) == {
        "custom_field": "x"
    }


def test_external_web_access_extra_cannot_switch_web_access_on() -> None:
    """A request cannot reach the external web the operator kept closed.

    The passthrough accepts any undeclared field, so without a gate on this
    one a client would decide the deployment's data-governance boundary for
    itself.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html
         stdapi/config.py:Settings.aws_bedrock_external_web_access
         stdapi/models/chat/_adapters/_common.py:resolve_external_web_access
    """
    with pytest.raises(ApiError, match="external_web_access") as excinfo:
        _additional_fields({"external_web_access": True})
    assert excinfo.value.status == 400
    assert "disabled" in str(excinfo.value)


def test_external_web_access_extra_refused_even_where_the_override_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that takes no web access choice refuses one instead of dropping it.

    Allowing the override lets a request choose only where its searches
    actually run with what it chose; answering with a search made under the
    server's own web access would give the client something it did not ask
    for.

    Ref: stdapi/config.py:Settings.aws_bedrock_allow_external_web_access_override
         stdapi/models/chat/_adapters/_common.py:resolve_external_web_access
    """
    monkeypatch.setattr(
        SETTINGS, "aws_bedrock_allow_external_web_access_override", True
    )
    with pytest.raises(ApiError, match="not available with this model") as excinfo:
        _additional_fields({"external_web_access": True})
    assert excinfo.value.status == 400


def test_external_web_access_is_gated_for_every_chat_route() -> None:
    """The gate sits on the extras helper every chat adapter forwards through.

    Chat Completions, legacy Completions and Anthropic Messages hand their
    undeclared fields to the same helper, so a gate placed on one route only
    would leave the parameter ungated on the other three.

    Ref: stdapi/models/chat/_adapters/_common.py:inference_extras
         stdapi/models/chat/_adapters/_common.py:resolve_external_web_access
    """
    assert inference_extras({"external_web_access": False}, frozenset()) == {}
    with pytest.raises(ApiError, match="external_web_access") as excinfo:
        inference_extras({"external_web_access": True}, frozenset())
    assert excinfo.value.status == 400


def test_non_boolean_external_web_access_is_refused() -> None:
    """Only a boolean can decide web access; anything else is a 400.

    Ref: stdapi/models/chat/_adapters/_common.py:resolve_external_web_access
    """
    with pytest.raises(ApiError, match="must be a boolean") as excinfo:
        _additional_fields({"external_web_access": "yes"})
    assert excinfo.value.status == 400
