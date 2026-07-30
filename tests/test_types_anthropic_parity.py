"""Anthropic request-model parity with the Anthropic SDK types (no AWS calls).

The gateway's Pydantic models mirror ``anthropic.types.*``; fields the SDK carries
but the mirror does not declare must survive validation as extras so newer clients
are not rejected, while documented numeric bounds must still be enforced.

Ref: https://platform.claude.com/docs/en/api/messages
     https://github.com/anthropics/anthropic-sdk-python/tree/main/src/anthropic/types
     stdapi/types/anthropic_messages.py:MessageCreateParams
"""

import pytest
from pydantic import ValidationError

from stdapi.types.anthropic_messages import (
    MessageCreateParams,
    ThinkingConfigAdaptiveParam,
    ThinkingConfigEnabledParam,
    ToolInputSchema,
)

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Minimal valid request body, extended per-test with a ``top_p`` value.
_BASE_REQUEST = {"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]}


class TestToolInputSchemaParity:
    """Arbitrary JSON Schema keywords on ToolInputSchema.

    A tool's ``input_schema`` is an arbitrary JSON Schema, so keywords the model does
    not declare must be retained rather than dropped or rejected.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
         stdapi/types/anthropic_messages.py:ToolInputSchema
    """

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
        assert schema.properties == {"x": {"type": "string"}}
        assert schema.model_extra == {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "$defs": {"y": {"type": "integer"}},
            "additionalProperties": False,
        }

    def test_extra_json_schema_keywords_survive_model_dump(self) -> None:
        """Extra keywords are forwarded to Bedrock via model_dump.

        ``_map_tool_spec`` dumps the schema straight into
        ``toolSpec.inputSchema.json``, so a dropped keyword would silently change the
        contract the model is given.
        """
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
    """Unknown client fields (e.g. display) on ThinkingConfigAdaptiveParam.

    Adaptive thinking is the current upstream shape and keeps gaining fields, so
    unmodelled ones must be preserved instead of failing the request.

    Ref: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
         stdapi/types/anthropic_messages.py:ThinkingConfigAdaptiveParam
    """

    def test_unknown_field_is_accepted(self) -> None:
        """An unknown 'display' field validates without error."""
        config = ThinkingConfigAdaptiveParam.model_validate(
            {"type": "adaptive", "display": "compact"}
        )
        assert config.type == "adaptive"
        assert config.model_extra == {"display": "compact"}

    def test_unknown_field_survives_model_dump(self) -> None:
        """The unknown field is retained on the model (available via model_dump)."""
        config = ThinkingConfigAdaptiveParam.model_validate(
            {"type": "adaptive", "display": "compact"}
        )
        assert config.model_dump()["display"] == "compact"


class TestThinkingConfigEnabledParamParity:
    """The upstream 'display' field (summarized/omitted) on ThinkingConfigEnabledParam.

    The deprecated budget-based shape accepts the same ``display`` field as the
    adaptive one, and accepting it must not disturb ``budget_tokens``.

    Ref: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
         stdapi/types/anthropic_messages.py:ThinkingConfigEnabledParam
    """

    def test_display_field_is_accepted(self) -> None:
        """A 'display' field validates without error, matching the adaptive variant."""
        config = ThinkingConfigEnabledParam.model_validate(
            {"type": "enabled", "budget_tokens": 1024, "display": "omitted"}
        )
        assert config.type == "enabled"
        assert config.budget_tokens == 1024
        assert config.model_extra == {"display": "omitted"}

    def test_display_field_survives_model_dump(self) -> None:
        """The 'display' field is retained on the model (available via model_dump)."""
        config = ThinkingConfigEnabledParam.model_validate(
            {"type": "enabled", "budget_tokens": 1024, "display": "omitted"}
        )
        assert config.model_dump()["display"] == "omitted"


class TestTopPRange:
    """``top_p`` is bounded to the valid nucleus-sampling range [0.0, 1.0].

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/types/anthropic_messages.py:MessageCreateParams
    """

    def test_top_p_above_one_is_rejected(self) -> None:
        """``top_p=1.5`` fails validation instead of reaching the backend.

        The bound is enforced locally so Bedrock's own ``ValidationException`` is
        never reached; the reported error must point at ``top_p`` and not at some
        unrelated field of the request body.
        """
        with pytest.raises(ValidationError) as excinfo:
            MessageCreateParams.model_validate({**_BASE_REQUEST, "top_p": 1.5})
        (error,) = excinfo.value.errors()
        assert error["loc"] == ("top_p",)
        assert error["type"] == "less_than_equal"
        assert "less than or equal to 1" in error["msg"]

    def test_top_p_of_one_is_accepted(self) -> None:
        """``top_p=1.0`` (the upper bound) validates successfully."""
        params = MessageCreateParams.model_validate({**_BASE_REQUEST, "top_p": 1.0})
        assert params.top_p == 1.0
