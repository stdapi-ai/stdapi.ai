"""Unit tests for Responses API tool-config and structured-output mapping (no AWS calls)."""

from __future__ import annotations

import pytest

from stdapi.models.chat._adapters._openai_responses import _build_output_config
from stdapi.types.openai_responses import FunctionTool, ResponseTextConfig

pytestmark = pytest.mark.local


def _tool(**kwargs: object) -> FunctionTool:
    """Build a validated function tool with the given overrides."""
    base: dict[str, object] = {
        "type": "function",
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {}},
    }
    base.update(kwargs)
    return FunctionTool.model_validate(base)


def test_build_output_config_forwards_json_schema_name_and_description() -> None:
    """The client-supplied schema name and description reach Bedrock's jsonSchema."""
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
