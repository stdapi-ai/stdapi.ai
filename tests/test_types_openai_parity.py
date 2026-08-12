"""Unit tests for OpenAI SDK schema-parity fields on the local request/response types.

Every request model here derives from ``BaseModelRequest`` (``extra`` is ignored
or forbidden), so a field that is not declared is dropped instead of surviving
as an extra: asserting that a value round-trips is therefore an assertion that
the field exists in the gateway's schema.

Ref: https://github.com/openai/openai-python
     https://developers.openai.com/api/reference/resources/responses/methods/create
     stdapi/types/openai_responses.py
"""

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from stdapi.api_errors import UnsupportedParameterError
from stdapi.types import BaseModelRequestWithFormExtra
from stdapi.types.openai_chat_completions import (
    ChatCompletionList,
    ChatCompletionStoreMessageList,
    CompletionCreateParams,
)
from stdapi.types.openai_responses import (
    AdditionalTools,
    CallerProgram,
    CompactParams,
    FunctionCallInput,
    FunctionTool,
    InputTokenCountParams,
    Mcp,
    Reasoning,
    ResponseApplyPatchToolCall,
    ResponseApplyPatchToolCallOutput,
    ResponseCreateParams,
    ResponseCustomToolCall,
    ResponseCustomToolCallOutput,
    ResponseCustomToolCallOutputItem,
    ResponseError,
    ResponseFunctionShellToolCall,
    ResponseFunctionShellToolCallOutput,
    ResponseFunctionToolCall,
    ResponseFunctionToolCallOutputItem,
    ResponseItemList,
    ResponseOutputItem,
)
from stdapi.types.openai_videos import Video, VideoList

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


class TestInputTokenCountParity:
    """personality and reasoning.context are accepted on POST /v1/responses/input_tokens.

    Both fields exist only so newer SDKs validate; neither can change the
    Bedrock ``CountTokens`` result, which counts the rendered input only.

    Ref: https://developers.openai.com/api/reference/resources/responses/subresources/input_tokens
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html
         stdapi/types/openai_responses.py:InputTokenCountParams
    """

    def test_personality_is_accepted_and_ignored(self) -> None:
        """The personality parameter is accepted for compatibility."""
        params = InputTokenCountParams(model="m", input="x", personality="friendly")
        assert params.personality == "friendly"

    def test_reasoning_context_is_accepted_and_ignored(self) -> None:
        """reasoning.context is accepted for compatibility."""
        params = InputTokenCountParams(
            model="m", input="x", reasoning=Reasoning(context="auto")
        )
        assert params.reasoning is not None
        assert params.reasoning.context == "auto"

    def test_unsupported_parameter_still_rejected(self) -> None:
        """Parameters that would change the count remain rejected.

        ``previous_response_id`` is one of
        ``InputTokenCountParams._UNSUPPORTED``; it is surfaced as a 400
        ``unsupported_parameter`` naming the offending field rather than being
        silently ignored like ``personality``.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
             stdapi/api_errors.py:UnsupportedParameterError
        """
        with pytest.raises(
            UnsupportedParameterError, match="previous_response_id"
        ) as excinfo:
            InputTokenCountParams(model="m", input="x", previous_response_id="resp_1")

        error = excinfo.value
        assert error.status == 400
        assert error.code == "unsupported_parameter"
        assert error.param == "previous_response_id"
        assert "not supported" in str(error)

    def test_conversation_is_accepted(self) -> None:
        """A conversation reference is counted, matching the upstream API.

        Ref: https://developers.openai.com/api/reference/resources/responses/subresources/input_tokens
             stdapi/routes/openai_responses.py:count_input_tokens
        """
        params = InputTokenCountParams(model="m", input="x", conversation="conv_1")
        assert params.conversation == "conv_1"

    def test_reasoning_effort_still_accepted(self) -> None:
        """reasoning.effort remains accepted."""
        params = InputTokenCountParams(
            model="m", input="x", reasoning=Reasoning(effort="low")
        )
        assert params.reasoning is not None
        assert params.reasoning.effort == "low"


class TestResponseCreateParity:
    """reasoning.context and reasoning.mode are accepted on POST /v1/responses.

    Ref: https://developers.openai.com/api/docs/guides/reasoning
         stdapi/types/openai_responses.py:Reasoning
    """

    def test_reasoning_context_is_accepted_and_ignored(self) -> None:
        """reasoning.context is accepted for compatibility."""
        params = ResponseCreateParams(
            model="m", input="x", reasoning=Reasoning(context="all_turns")
        )
        assert params.reasoning is not None
        assert params.reasoning.context == "all_turns"

    @pytest.mark.parametrize("mode", ["pro", "future-mode"])
    def test_reasoning_mode_is_accepted_and_ignored(self, mode: str) -> None:
        """reasoning.mode is accepted for compatibility, open like the SDK type."""
        params = ResponseCreateParams(
            model="m", input="x", reasoning=Reasoning(mode=mode)
        )
        assert params.reasoning is not None
        assert params.reasoning.mode == mode


class TestAdditionalToolsItem:
    """additional_tools output items parse through the output item union.

    Ref: https://developers.openai.com/api/docs/guides/tools
         stdapi/types/openai_responses.py:AdditionalTools
    """

    def test_union_parses_additional_tools(self) -> None:
        """A canned additional_tools payload parses to AdditionalTools."""
        item = TypeAdapter[ResponseOutputItem](ResponseOutputItem).validate_python(
            {
                "id": "at-1",
                "role": "assistant",
                "type": "additional_tools",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": {"type": "object"},
                        "strict": True,
                    }
                ],
            }
        )
        assert isinstance(item, AdditionalTools)
        assert item.tools[0].name == "get_weather"  # type: ignore[union-attr]


class TestClientCompatibilityFields:
    """Fields sent by newer OpenAI clients are accepted and ignored.

    Rejecting an unknown tool type at parse time would surface as a schema
    error; the gateway keeps the type in its union so the request reaches the
    route, which then raises the "unsupported on this implementation" 400.

    Ref: https://developers.openai.com/api/docs/guides/tools
         stdapi/types/openai_responses.py:ResponseCreateParams
    """

    def test_client_metadata_accepted(self) -> None:
        """client_metadata (sent by Codex) is accepted and ignored."""
        params = ResponseCreateParams(
            model="m", input="x", client_metadata={"editor": "pycharm"}
        )
        assert params.client_metadata == {"editor": "pycharm"}

    def test_unsupported_tools_accepted_and_parsed(self) -> None:
        """Hosted tool types without a backend equivalent parse cleanly.

        Ref: stdapi/types/openai_responses.py:NamespaceTool
        """
        params = ResponseCreateParams(
            model="m",
            input="x",
            tools=[
                {"type": "namespace", "name": "ns", "description": "d", "tools": []}  # type: ignore[list-item]
            ],
        )
        assert params.tools is not None
        assert len(params.tools) == 1
        assert params.tools[0].type == "namespace"
        assert params.tools[0].name == "ns"


class TestReasoningEffortParity:
    """reasoning.effort accepts the upstream SDK's `max` literal.

    Ref: https://developers.openai.com/api/docs/guides/reasoning
         stdapi/types/openai_responses.py:Reasoning
    """

    def test_max_effort_is_accepted(self) -> None:
        """reasoning.effort="max" validates like the other SDK literals."""
        params = ResponseCreateParams(
            model="m", input="x", reasoning=Reasoning(effort="max")
        )
        assert params.reasoning is not None
        assert params.reasoning.effort == "max"


class TestToolAllowedCallersParity:
    """allowed_callers/output_schema/tunnel_id round-trip on tool definitions.

    These fields belong to programmatic tool calling; dropping them would
    silently change which callers a tool accepts.

    Ref: https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling
         https://developers.openai.com/api/docs/guides/tools-connectors-mcp
         stdapi/types/openai_responses.py:FunctionTool
    """

    def test_function_tool_accepts_allowed_callers_and_output_schema(self) -> None:
        """FunctionTool keeps allowed_callers and output_schema instead of dropping them."""
        tool = FunctionTool(
            name="f",
            type="function",
            allowed_callers=["direct", "programmatic"],
            output_schema={"type": "object"},
        )
        assert tool.allowed_callers == ["direct", "programmatic"]
        assert tool.output_schema == {"type": "object"}

    def test_mcp_accepts_allowed_callers_and_tunnel_id(self) -> None:
        """Mcp keeps allowed_callers and tunnel_id instead of dropping them."""
        tool = Mcp(
            server_label="s", type="mcp", allowed_callers=["direct"], tunnel_id="t-1"
        )
        assert tool.allowed_callers == ["direct"]
        assert tool.tunnel_id == "t-1"


class TestResponseErrorCodeParity:
    """ResponseErrorCode includes the full upstream error-code enum.

    A stored response with ``status="failed"`` echoes the upstream code back,
    so a missing literal would make the response object unserializable.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/retrieve
         stdapi/types/openai_responses.py:ResponseError
    """

    @pytest.mark.parametrize("code", ["data_residency_mismatch", "bio_policy"])
    def test_upstream_only_codes_are_accepted(self, code: str) -> None:
        """Error codes documented upstream validate here too."""
        error = ResponseError(code=code, message="m")  # type: ignore[arg-type]
        assert error.code == code


class TestToolCallerParity:
    """Tool-call items keep the `caller` provenance field, as input and as output.

    Upstream declares `caller` on both directions of a tool call: the input
    item echoed back from a previous response, and the output item the model
    just produced. A field added only on the input side would silently drop
    `caller` from every function/custom tool call in a fresh response.

    Ref: https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling
         stdapi/types/openai_responses.py:FunctionCallInput
         stdapi/types/openai_responses.py:ResponseFunctionToolCall
         stdapi/types/openai_responses.py:ResponseCustomToolCall
    """

    def test_function_call_input_accepts_program_caller(self) -> None:
        """A function_call input item retains a program caller instead of dropping it."""
        item = FunctionCallInput(
            arguments="{}",
            call_id="c1",
            name="f",
            type="function_call",
            caller=CallerProgram(type="program", caller_id="p1"),
        )
        assert item.caller is not None
        assert item.caller.type == "program"
        assert item.caller.caller_id == "p1"

    def test_response_function_tool_call_accepts_program_caller(self) -> None:
        """A function_call output item retains a program caller instead of dropping it."""
        item = ResponseFunctionToolCall(
            arguments="{}",
            call_id="c1",
            name="f",
            type="function_call",
            caller=CallerProgram(type="program", caller_id="p1"),
        )
        assert item.caller is not None
        assert item.caller.type == "program"
        assert item.caller.caller_id == "p1"

    def test_response_custom_tool_call_accepts_program_caller(self) -> None:
        """A custom_tool_call output item retains a program caller instead of dropping it."""
        item = ResponseCustomToolCall(
            call_id="c1",
            input="x",
            name="f",
            type="custom_tool_call",
            caller=CallerProgram(type="program", caller_id="p1"),
        )
        assert item.caller is not None
        assert item.caller.type == "program"
        assert item.caller.caller_id == "p1"

    @pytest.mark.parametrize(
        ("item_type", "payload"),
        [
            (
                ResponseFunctionToolCallOutputItem,
                {
                    "id": "o1",
                    "call_id": "c1",
                    "output": "ok",
                    "status": "completed",
                    "type": "function_call_output",
                },
            ),
            (
                ResponseCustomToolCallOutput,
                {"call_id": "c1", "output": "ok", "type": "custom_tool_call_output"},
            ),
            (
                ResponseCustomToolCallOutputItem,
                {
                    "id": "o1",
                    "call_id": "c1",
                    "output": "ok",
                    "status": "completed",
                    "type": "custom_tool_call_output",
                },
            ),
            (
                ResponseFunctionShellToolCall,
                {
                    "id": "s1",
                    "call_id": "c1",
                    "action": {"commands": ["ls"]},
                    "status": "completed",
                    "type": "shell_call",
                },
            ),
            (
                ResponseFunctionShellToolCallOutput,
                {
                    "id": "o1",
                    "call_id": "c1",
                    "output": [
                        {
                            "outcome": {"type": "exit", "exit_code": 0},
                            "stdout": "",
                            "stderr": "",
                        }
                    ],
                    "status": "completed",
                    "type": "shell_call_output",
                },
            ),
            (
                ResponseApplyPatchToolCall,
                {
                    "id": "p1",
                    "call_id": "c1",
                    "operation": {"type": "create_file", "path": "a.txt", "diff": "+x"},
                    "status": "completed",
                    "type": "apply_patch_call",
                },
            ),
            (
                ResponseApplyPatchToolCallOutput,
                {
                    "id": "o1",
                    "call_id": "c1",
                    "status": "completed",
                    "type": "apply_patch_call_output",
                },
            ),
        ],
    )
    def test_stored_response_items_keep_a_program_caller(
        self, item_type: type[BaseModel], payload: dict[str, object]
    ) -> None:
        """Every response-side tool item retains `caller` instead of erroring on it.

        These are the shapes the local store replays through the input_items
        listing: a missing `caller` declaration would reject (``extra="forbid"``)
        or drop the provenance a program-driven tool call carries.
        """
        item = item_type.model_validate(
            {**payload, "caller": {"type": "program", "caller_id": "p1"}}
        )
        caller = item.caller  # type: ignore[attr-defined]
        assert caller is not None
        assert caller.type == "program"
        assert caller.caller_id == "p1"


class TestUnsupportedNullIsAccepted:
    """Explicitly-null unsupported parameters validate like omission.

    SDKs and proxies routinely serialise unset optionals as ``null``; the Chat
    Completions twin already treats ``null``/``false`` as requesting the
    supported default behaviour, so the Responses bodies must not 400 on them.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/types/openai_chat_completions.py:CompletionCreateParams._unsupported
         stdapi/types/openai_responses.py:ResponseCreateParams
         stdapi/types/openai_responses.py:InputTokenCountParams
    """

    @pytest.mark.parametrize(
        "key", ["context_management", "conversation", "max_tool_calls", "truncation"]
    )
    def test_create_params_accept_an_explicit_null(self, key: str) -> None:
        """A null unsupported field on ResponseCreateParams validates."""
        params = ResponseCreateParams.model_validate(
            {"model": "m", "input": "x", key: None}
        )
        assert getattr(params, key) is None

    @pytest.mark.parametrize(
        "key", ["text", "truncation", "previous_response_id", "conversation"]
    )
    def test_count_params_accept_an_explicit_null(self, key: str) -> None:
        """A null unsupported field on InputTokenCountParams validates."""
        params = InputTokenCountParams.model_validate(
            {"model": "m", "input": "x", key: None}
        )
        assert getattr(params, key) is None

    def test_a_real_value_is_still_rejected(self) -> None:
        """Setting an unsupported parameter to a value still 400s."""
        with pytest.raises(UnsupportedParameterError, match="truncation"):
            ResponseCreateParams.model_validate(
                {"model": "m", "input": "x", "truncation": "auto"}
            )


class TestPromptCacheRetentionParity:
    """prompt_cache_retention uses the OpenAI SDK literal `in_memory`.

    The hyphenated spelling is not an accepted alias: it must fail validation
    so a typo cannot silently select Bedrock's default TTL.

    Ref: https://developers.openai.com/api/docs/guides/prompt-caching#prompt-cache-retention
         stdapi/models/chat/_adapters/_openai_common.py:resolve_cache_ttl
    """

    @pytest.mark.parametrize(
        "params_type", [ResponseCreateParams, CompactParams, CompletionCreateParams]
    )
    def test_sdk_literal_accepted(self, params_type: type) -> None:
        """The SDK value `in_memory` is accepted; the hyphen form is not."""
        extra = (
            {"messages": [{"role": "user", "content": "x"}]}
            if params_type is CompletionCreateParams
            else {"input": "x"}
        )
        params = params_type(model="m", prompt_cache_retention="in_memory", **extra)
        assert params.prompt_cache_retention == "in_memory"
        with pytest.raises(ValidationError, match="prompt_cache_retention"):
            params_type(model="m", prompt_cache_retention="in-memory", **extra)


class TestPromptCacheOptionsParity:
    """prompt_cache_options is accepted on the Responses create/compact request bodies.

    ``mode="explicit"`` is what suppresses the ``prompt_cache_key`` heuristic,
    so the object must survive parsing rather than being dropped.

    Ref: https://developers.openai.com/api/docs/guides/prompt-caching#prompt-cache-breakpoints
         https://developers.openai.com/api/reference/resources/responses/methods/compact
         stdapi/models/chat/_adapters/_openai_common.py:parse_prompt_cache_key
    """

    @pytest.mark.parametrize("params_type", [ResponseCreateParams, CompactParams])
    def test_prompt_cache_options_is_accepted(self, params_type: type) -> None:
        """The explicit prompt_cache_options object round-trips instead of being dropped."""
        params = params_type(
            model="m",
            input="x",
            prompt_cache_options={"mode": "explicit", "ttl": "30m"},
        )
        assert params.prompt_cache_options is not None
        assert params.prompt_cache_options.mode == "explicit"
        assert params.prompt_cache_options.ttl == "30m"


class TestPaginatedListEnvelopeParity:
    """Paginated list responses share a common envelope base without changing wire keys.

    ``PaginatedListEnvelope`` factors out has_more/first_id/last_id; the dumped
    key set proves the shared base neither reorders nor renames the wire fields,
    and that ``object`` still defaults to ``"list"`` per subclass.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/list
         https://developers.openai.com/api/reference/resources/responses/subresources/input_items/methods/list
         stdapi/types/openai.py:PaginatedListEnvelope
    """

    @pytest.mark.parametrize(
        ("list_type", "extra"),
        [
            (ChatCompletionList, {}),
            (ChatCompletionStoreMessageList, {}),
            (VideoList, {}),
            (ResponseItemList, {"object": "list"}),
        ],
    )
    def test_envelope_keys_unchanged(
        self, list_type: type, extra: dict[str, str]
    ) -> None:
        """object, data, has_more, first_id, and last_id all round-trip via model_dump."""
        instance = list_type(
            data=[], has_more=False, first_id=None, last_id=None, **extra
        )
        dumped = instance.model_dump()
        assert dumped.keys() == {"object", "data", "has_more", "first_id", "last_id"}
        assert dumped["object"] == "list"
        assert dumped["data"] == []
        assert dumped["has_more"] is False
        assert dumped["first_id"] is None
        assert dumped["last_id"] is None


class TestAdditionalToolsInputEcho:
    """A previously emitted `additional_tools` output item can be echoed back as input.

    Clients replay the whole output list as the next request's input, so an
    output item that does not also parse as an input item breaks multi-turn use.

    Ref: https://developers.openai.com/api/docs/guides/tools
         stdapi/types/openai_responses.py:AdditionalToolsInput
    """

    def test_additional_tools_accepted_as_input(self) -> None:
        """additional_tools parses as a ResponseCreateParams.input item without error."""
        params = ResponseCreateParams(
            model="m",
            input=[
                {
                    "id": "at-1",
                    "role": "assistant",
                    "type": "additional_tools",
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        }
                    ],
                }  # type: ignore[list-item]
            ],
        )
        assert params.input

    def test_additional_tools_id_is_optional(self) -> None:
        """additional_tools without `id` still validates, matching the upstream shape."""
        params = ResponseCreateParams(
            model="m",
            input=[
                {
                    "role": "developer",
                    "type": "additional_tools",
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_weather",
                            "parameters": {"type": "object"},
                        }
                    ],
                }  # type: ignore[list-item]
            ],
        )
        assert params.input


class TestVideoParity:
    """remixed_from_video_id exists on Video objects and stays null.

    Bedrock async invocation has no remix operation, so the field is declared
    for schema parity only and is excluded from the emitted payload.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         https://stdapi.ai/api_openai_videos/
         stdapi/types/openai_videos.py:Video
    """

    def test_remixed_from_video_id_defaults_to_none(self) -> None:
        """The field exists for schema parity and defaults to null."""
        video = Video(
            id="video_x",
            model="luma.ray-v2:0",
            status="queued",
            created_at=1,
            seconds="5",
            size="1280x720",
        )
        assert video.remixed_from_video_id is None
        assert "remixed_from_video_id" not in video.model_dump(exclude_none=True)


class _FormParams(BaseModelRequestWithFormExtra):
    """Minimal multipart-form request type for extra-parameter coercion tests."""

    model: str
    prompt: str = ""


class TestFormExtraParameterParity:
    """Extra model parameters get the same typing over multipart as over JSON.

    A JSON body already delivers non-string extra values (numbers, booleans) as
    their native Python type; a multipart/form-data body only has strings.
    ``BaseModelRequestWithFormExtra._deserialize_forms`` JSON-decodes extra
    (undeclared) field values so both surfaces agree, without touching declared
    string fields, which must reach the model verbatim even if they look like
    JSON (a `prompt` of literal text "null" must stay the string "null").

    Ref: https://platform.openai.com/docs/api-reference/images/createVariation
         (multipart/form-data; `extra_body` values are form-encoded strings)
         stdapi/types/__init__.py:BaseModelRequestWithFormExtra._deserialize_forms
    """

    def test_extra_numeric_string_is_json_decoded(self) -> None:
        """A numeric extra parameter arriving as a form string becomes a float."""
        params = _FormParams.model_validate({"model": "m", "strength": "0.7"})
        assert params.model_extra == {"strength": 0.7}

    def test_extra_boolean_string_is_json_decoded(self) -> None:
        """A boolean extra parameter arriving as a form string becomes a bool."""
        params = _FormParams.model_validate({"model": "m", "flag": "true"})
        assert params.model_extra == {"flag": True}

    def test_declared_string_field_is_not_coerced(self) -> None:
        """A declared string field keeps a JSON-look-alike value verbatim."""
        params = _FormParams.model_validate({"model": "m", "prompt": "null"})
        assert params.prompt == "null"

    def test_non_json_extra_string_stays_a_string(self) -> None:
        """An extra value that is not valid JSON is kept as-is, not dropped."""
        params = _FormParams.model_validate({"model": "m", "note": "hello there"})
        assert params.model_extra == {"note": "hello there"}
