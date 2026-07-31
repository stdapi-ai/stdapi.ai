"""Unit tests for the Responses ``text.format`` -> Bedrock ``outputConfig`` mapping.

Structured Outputs are served by Bedrock's native ``outputConfig.textFormat``
JSON schema definition rather than by prompt-level instructions, so the schema
name and description supplied by the client must reach Bedrock verbatim.

Ref: https://developers.openai.com/api/docs/guides/structured-outputs
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
     stdapi/models/chat/_adapters/_openai_responses.py:_build_output_config
"""

from __future__ import annotations

import pytest

from stdapi.models.chat._adapters._openai_responses import _build_output_config
from stdapi.types.openai_responses import ResponseTextConfig

pytestmark = pytest.mark.local


def test_build_output_config_returns_none_for_json_object() -> None:
    """``text.format={"type": "json_object"}`` builds no Bedrock outputConfig.

    Bedrock's strict structured output has no schema for "any JSON object": an
    empty schema is rejected and the only closed alternative,
    ``{"type": "object", "additionalProperties": false}``, admits only ``{}``,
    so the adapter must skip outputConfig entirely rather than constrain the
    model to an empty response (issue #96).
    """
    text = ResponseTextConfig.model_validate({"format": {"type": "json_object"}})
    assert _build_output_config(text) is None


def test_build_output_config_forwards_json_schema_name_and_description() -> None:
    """The client-supplied schema, name and description reach Bedrock's jsonSchema."""
    text = ResponseTextConfig.model_validate(
        {
            "format": {
                "type": "json_schema",
                "name": "weather_report",
                "description": "A weather report.",
                "schema": {"type": "object"},
            }
        }
    )
    output_config = _build_output_config(text)
    assert output_config is not None
    assert output_config["name"] == "weather_report"
    assert output_config["description"] == "A weather report."
    assert output_config["schema"] == '{"type":"object"}', (
        "the schema is forwarded as a serialised JSON string"
    )


def test_build_output_config_omits_description_when_unset() -> None:
    """An unset schema description is not forwarded to Bedrock's jsonSchema."""
    text = ResponseTextConfig.model_validate(
        {
            "format": {
                "type": "json_schema",
                "name": "weather_report",
                "schema": {"type": "object"},
            }
        }
    )
    output_config = _build_output_config(text)
    assert output_config is not None
    assert "description" not in output_config
    assert output_config["name"] == "weather_report"
