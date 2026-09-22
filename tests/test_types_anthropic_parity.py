"""Anthropic model parity with the Anthropic SDK types (no AWS calls).

The gateway's Pydantic models mirror ``anthropic.types.*``; fields the SDK carries
but the mirror does not declare must survive validation as extras so newer clients
are not rejected, while documented numeric bounds, discriminators and enumerations
must still match the SDK.

Ref: https://platform.claude.com/docs/en/api/messages
     https://github.com/anthropics/anthropic-sdk-python/tree/main/src/anthropic/types
     stdapi/types/anthropic_messages.py:MessageCreateParams
"""

from typing import Required, get_args, get_origin, get_type_hints

import pytest
from anthropic.types import ContainerParams as SdkContainerParams
from anthropic.types import SkillParams as SdkSkillParams
from anthropic.types import WebSearchToolResultError as SdkWebSearchToolResultError
from anthropic.types.tool_result_block_param import (
    ToolResultBlockParam as SdkToolResultBlockParam,
)
from pydantic import ValidationError

from stdapi.types.anthropic_messages import (
    ContainerParams,
    MessageCreateParams,
    SkillParams,
    ThinkingConfigAdaptiveParam,
    ThinkingConfigEnabledParam,
    ToolInputSchema,
    ToolResultBlockParam,
    WebSearchToolRequestErrorParam,
    WebSearchToolResultError,
)

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Minimal valid request body, extended per-test with the field under test.
_BASE_REQUEST = {"model": "claude-x", "messages": [{"role": "user", "content": "hi"}]}

#: An Agent Skills request body, as the Agent Skills quickstart documents it.
_SKILLS_CONTAINER = {
    "skills": [{"type": "anthropic", "skill_id": "pptx", "version": "latest"}]
}


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
    """Unknown client fields on ThinkingConfigAdaptiveParam.

    Adaptive thinking is the current upstream shape and keeps gaining fields, so
    unmodelled ones must be preserved instead of failing the request.

    Ref: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
         stdapi/types/anthropic_messages.py:ThinkingConfigAdaptiveParam
    """

    def test_unknown_field_is_accepted(self) -> None:
        """An unknown field validates without error."""
        config = ThinkingConfigAdaptiveParam.model_validate(
            {"type": "adaptive", "future_field": "compact"}
        )
        assert config.type == "adaptive"
        assert config.model_extra == {"future_field": "compact"}

    def test_unknown_field_survives_model_dump(self) -> None:
        """The unknown field is retained on the model (available via model_dump)."""
        config = ThinkingConfigAdaptiveParam.model_validate(
            {"type": "adaptive", "future_field": "compact"}
        )
        assert config.model_dump()["future_field"] == "compact"


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
        assert config.display == "omitted"

    def test_display_field_survives_model_dump(self) -> None:
        """The 'display' field is retained on the model (available via model_dump)."""
        config = ThinkingConfigEnabledParam.model_validate(
            {"type": "enabled", "budget_tokens": 1024, "display": "omitted"}
        )
        assert config.model_dump()["display"] == "omitted"


class TestToolResultContentParity:
    """``content`` is optional on a ``tool_result`` block, as upstream declares it.

    A tool called only for its side effect answers with an empty result — a block
    carrying just ``type`` and ``tool_use_id`` — which Anthropic documents and the
    SDK types as valid.  Rejecting it kills a conformant agent loop mid-turn.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls
         https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/tool_result_block_param.py
         stdapi/types/anthropic_messages.py:ToolResultBlockParam
    """

    def test_required_fields_match_the_sdk(self) -> None:
        """Only the keys the SDK marks ``Required`` are mandatory on the mirror.

        The SDK's own ``__required_keys__`` reports nothing on a ``total=False``
        TypedDict here, so the annotations are read instead.
        """
        hints = get_type_hints(SdkToolResultBlockParam, include_extras=True)
        sdk_required = {
            name for name, hint in hints.items() if get_origin(hint) is Required
        }
        assert sdk_required, "the SDK annotations must state which keys are required"
        required = {
            name
            for name, field in ToolResultBlockParam.model_fields.items()
            if field.is_required()
        }
        assert required == sdk_required

    def test_a_block_without_content_validates(self) -> None:
        """A block carrying only ``type`` and ``tool_use_id`` validates."""
        block = ToolResultBlockParam.model_validate(
            {"type": "tool_result", "tool_use_id": "toolu_1"}
        )
        assert block.tool_use_id == "toolu_1"
        assert block.content is None, "an omitted content stays absent, not invented"

    def test_a_request_replaying_an_empty_tool_result_validates(self) -> None:
        """A whole request whose tool turn is answered with an empty result validates.

        The content-block union is resolved left to right, so a block failing
        ``ToolResultBlockParam`` is reported as a 422 against the whole request
        instead of reaching a model — which is what an agent loop hits when a tool
        returns nothing.
        """
        params = MessageCreateParams.model_validate(
            {
                **_BASE_REQUEST,
                "messages": [
                    {"role": "user", "content": "record the event"},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "record_event",
                                "input": {"name": "signup"},
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": "toolu_1"}],
                    },
                ],
            }
        )
        content = params.messages[2].content
        assert isinstance(content, list)
        (block,) = content
        assert isinstance(block, ToolResultBlockParam), (
            "the block must resolve to a tool result, not to another union member"
        )
        assert block.content is None


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


class TestWebSearchToolResultErrorParity:
    """The error variant of a ``web_search_tool_result`` block matches the SDK.

    This block is published in the gateway's OpenAPI schema, so a client generated
    from it resolves the union by the ``type`` discriminator: a value the API never
    sends yields a variant that can never match.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
         stdapi/types/anthropic_messages.py:WebSearchToolResultError
    """

    def test_type_discriminator_matches_the_sdk(self) -> None:
        """``type`` is ``web_search_tool_result_error``, as the SDK and the request side spell it."""
        assert get_args(WebSearchToolResultError.model_fields["type"].annotation) == (
            get_args(SdkWebSearchToolResultError.model_fields["type"].annotation)
        )
        assert get_args(
            WebSearchToolRequestErrorParam.model_fields["type"].annotation
        ) == (get_args(WebSearchToolResultError.model_fields["type"].annotation))

    def test_error_code_is_narrowed_to_the_sdk_literals(self) -> None:
        """``error_code`` publishes the SDK's enumeration instead of an open string."""
        assert get_args(
            WebSearchToolResultError.model_fields["error_code"].annotation
        ) == get_args(SdkWebSearchToolResultError.model_fields["error_code"].annotation)

    def test_a_real_error_payload_validates(self) -> None:
        """A block Anthropic actually sends round-trips through the mirror."""
        error = WebSearchToolResultError.model_validate(
            {"type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}
        )
        assert error.model_dump() == {
            "type": "web_search_tool_result_error",
            "error_code": "max_uses_exceeded",
        }


class TestContainerParity:
    """``container`` takes the object form Agent Skills requests are written with.

    Upstream types the field as ``ContainerParams | str``, and the object form is
    the only way to name Agent Skills.  No container runs here, so the value is
    accepted and ignored — but a client sending the documented shape must be
    answered, not refused on the whole request before anything else is read.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/agent-skills/quickstart
         https://platform.claude.com/docs/en/api/messages
         stdapi/types/anthropic_messages.py:ContainerParams
    """

    def test_a_skills_container_validates(self) -> None:
        """The quickstart's ``container.skills`` body is read into the mirror.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/agent-skills/quickstart
             stdapi/types/anthropic_messages.py:SkillParams
        """
        params = MessageCreateParams.model_validate(
            {**_BASE_REQUEST, "container": _SKILLS_CONTAINER}
        )

        container = params.container
        assert isinstance(container, ContainerParams), (
            "the object form must resolve to the container model, not to the string arm"
        )
        assert container.skills is not None
        (skill,) = container.skills
        assert (skill.type, skill.skill_id, skill.version) == (
            "anthropic",
            "pptx",
            "latest",
        )

    def test_an_identifier_only_container_validates(self) -> None:
        """``{"container": {"id": …}}`` — the object form of a reused container.

        Ref: https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/container_params.py
             stdapi/types/anthropic_messages.py:ContainerParams
        """
        params = MessageCreateParams.model_validate(
            {**_BASE_REQUEST, "container": {"id": "container_1"}}
        )

        container = params.container
        assert isinstance(container, ContainerParams)
        assert container.id == "container_1"
        assert container.skills is None, "an omitted skill list stays absent"

    def test_the_string_form_still_validates(self) -> None:
        """The other arm of the union keeps working, unchanged and uncoerced.

        Ref: https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/message_create_params_container_param.py
             stdapi/types/anthropic_messages.py:MessageCreateParams
        """
        params = MessageCreateParams.model_validate(
            {**_BASE_REQUEST, "container": "container_1"}
        )

        assert params.container == "container_1"

    def test_container_fields_match_the_sdk(self) -> None:
        """The mirror declares the SDK's container keys, all of them optional.

        Ref: https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/container_params.py
             stdapi/types/anthropic_messages.py:ContainerParams
        """
        sdk_fields = set(get_type_hints(SdkContainerParams))
        assert sdk_fields, "the SDK annotations must state the container keys"
        assert set(ContainerParams.model_fields) == sdk_fields
        assert not [
            name
            for name, field in ContainerParams.model_fields.items()
            if field.is_required()
        ]

    def test_skill_fields_and_required_keys_match_the_sdk(self) -> None:
        """A skill entry carries the SDK's keys, required where the SDK requires them.

        Ref: https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/skill_params.py
             stdapi/types/anthropic_messages.py:SkillParams
        """
        hints = get_type_hints(SdkSkillParams, include_extras=True)
        sdk_required = {
            name for name, hint in hints.items() if get_origin(hint) is Required
        }
        assert sdk_required, "the SDK annotations must state which keys are required"
        assert set(SkillParams.model_fields) == set(hints)
        required = {
            name
            for name, field in SkillParams.model_fields.items()
            if field.is_required()
        }
        assert required == sdk_required

    def test_skill_type_matches_the_sdk_literals(self) -> None:
        """``skills[].type`` publishes the SDK's enumeration, not an open string.

        Ref: https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/skill_params.py
             stdapi/types/anthropic_messages.py:SkillParams
        """
        (sdk_type,) = get_args(
            get_type_hints(SdkSkillParams, include_extras=True)["type"]
        )

        assert get_args(SkillParams.model_fields["type"].annotation) == get_args(
            sdk_type
        )

    @pytest.mark.parametrize(
        "container", [_SKILLS_CONTAINER, {"id": "container_1"}, "container_1"]
    )
    def test_a_container_never_reaches_a_backend(self, container: object) -> None:
        """No serialization of the request carries ``container``, in either form.

        The Anthropic-shaped passthrough payload is built by dumping the validated
        request, so a container left in the dump would be forwarded — asking a
        backend for skills the gateway states it does not run.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/agent-skills/quickstart
             stdapi/models/chat/_mantle/_convert.py:messages_payload
        """
        params = MessageCreateParams.model_validate(
            {**_BASE_REQUEST, "container": container}
        )

        dumped = params.model_dump(mode="json", by_alias=True, exclude_unset=True)
        assert "container" not in dumped
        assert "container" not in params.model_dump(mode="json")


class TestInferenceGeoIsNeverForwarded:
    """``inference_geo`` is read off the request and goes no further.

    Where inference runs is the deployment's choice, so the documented behaviour is
    that a per-request geography is accepted and ignored.  A serialization that
    still carried it would let a caller move inference on the very path the
    documentation promises it cannot.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/types/anthropic_messages.py:MessageCreateParams
    """

    def test_an_inference_geo_validates(self) -> None:
        """A request naming a geography is answered rather than refused.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/types/anthropic_messages.py:MessageCreateParams
        """
        params = MessageCreateParams.model_validate(
            {**_BASE_REQUEST, "inference_geo": "us"}
        )

        assert params.inference_geo == "us"

    def test_an_inference_geo_never_reaches_a_backend(self) -> None:
        """No serialization of the request carries ``inference_geo``.

        Ref: stdapi/models/chat/_mantle/_convert.py:messages_payload
        """
        params = MessageCreateParams.model_validate(
            {**_BASE_REQUEST, "inference_geo": "us"}
        )

        dumped = params.model_dump(mode="json", by_alias=True, exclude_unset=True)
        assert "inference_geo" not in dumped
        assert "inference_geo" not in params.model_dump(mode="json")
