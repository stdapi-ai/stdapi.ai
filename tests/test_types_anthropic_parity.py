"""Tests for Anthropic SDK schema-parity fields (unit)."""

import pytest
from pydantic import ValidationError

from stdapi.types.anthropic_messages import (
    MessageCreateParams,
    ThinkingConfigAdaptiveParam,
    ToolInputSchema,
)

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Minimal valid request body, extended per-test with a ``top_p`` value.
_BASE_REQUEST = {"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]}


class TestToolInputSchemaParity:
    """Arbitrary JSON Schema keywords on ToolInputSchema."""

    def test_extra_json_schema_keywords_are_accepted(self) -> None:
        """Unknown JSON Schema keywords ($schema, $defs, additionalProperties) validate."""
        schema = ToolInputSchema.model_validate(
            {
                "type": "object",
                "properties": {"x": {"type": "string"}},
                "$schema": "http://json-schema.org/draft-07/schema#",
                "$defs": {"y": {"type": "integer"}},
                "additionalProperties": False,
            }
        )
        assert schema.type == "object"

    def test_extra_json_schema_keywords_survive_model_dump(self) -> None:
        """Extra keywords are forwarded upstream via model_dump."""
        schema = ToolInputSchema.model_validate(
            {
                "type": "object",
                "$schema": "http://json-schema.org/draft-07/schema#",
                "$defs": {"y": {"type": "integer"}},
                "additionalProperties": False,
            }
        )
        dumped = schema.model_dump()
        assert dumped["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert dumped["$defs"] == {"y": {"type": "integer"}}
        assert dumped["additionalProperties"] is False


class TestThinkingConfigAdaptiveParamParity:
    """Unknown client fields (e.g. display) on ThinkingConfigAdaptiveParam."""

    def test_unknown_field_is_accepted(self) -> None:
        """An unknown 'display' field validates without error."""
        config = ThinkingConfigAdaptiveParam.model_validate(
            {"type": "adaptive", "display": "compact"}
        )
        assert config.type == "adaptive"

    def test_unknown_field_survives_model_dump(self) -> None:
        """The unknown field is retained on the model (available via model_dump)."""
        config = ThinkingConfigAdaptiveParam.model_validate(
            {"type": "adaptive", "display": "compact"}
        )
        assert config.model_dump()["display"] == "compact"


class TestTopPRange:
    """``top_p`` is bounded to the valid nucleus-sampling range [0.0, 1.0]."""

    def test_top_p_above_one_is_rejected(self) -> None:
        """``top_p=1.5`` fails validation instead of reaching the backend."""
        with pytest.raises(ValidationError):
            MessageCreateParams.model_validate({**_BASE_REQUEST, "top_p": 1.5})

    def test_top_p_of_one_is_accepted(self) -> None:
        """``top_p=1.0`` (the upper bound) validates successfully."""
        params = MessageCreateParams.model_validate({**_BASE_REQUEST, "top_p": 1.0})
        assert params.top_p == 1.0
