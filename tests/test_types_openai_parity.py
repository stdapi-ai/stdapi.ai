"""Tests for OpenAI SDK schema-parity fields (unit)."""

import pytest
from pydantic import TypeAdapter, ValidationError

from stdapi.api_errors import UnsupportedParameterError
from stdapi.types.openai_chat_completions import (
    ChatCompletionList,
    ChatCompletionStoreMessageList,
    CompletionCreateParams,
)
from stdapi.types.openai_responses import (
    AdditionalTools,
    CompactParams,
    FunctionCallInput,
    FunctionTool,
    InputTokenCountParams,
    Mcp,
    Reasoning,
    ResponseCreateParams,
    ResponseError,
    ResponseItemList,
    ResponseOutputItem,
)
from stdapi.types.openai_videos import Video, VideoList

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


class TestInputTokenCountParity:
    """personality and reasoning.context on POST /v1/responses/input_tokens."""

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
        """Parameters that would change the count remain rejected."""
        with pytest.raises(UnsupportedParameterError, match="conversation"):
            InputTokenCountParams(model="m", input="x", conversation="conv_1")

    def test_reasoning_effort_still_accepted(self) -> None:
        """reasoning.effort remains accepted."""
        params = InputTokenCountParams(
            model="m", input="x", reasoning=Reasoning(effort="low")
        )
        assert params.reasoning is not None
        assert params.reasoning.effort == "low"


class TestResponseCreateParity:
    """reasoning.context on POST /v1/responses."""

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
    """additional_tools output items parse through the output item union."""

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
    """Fields sent by newer OpenAI clients are accepted and ignored."""

    def test_client_metadata_accepted(self) -> None:
        """client_metadata (sent by Codex) is accepted and ignored."""
        params = ResponseCreateParams(
            model="m", input="x", client_metadata={"editor": "pycharm"}
        )
        assert params.client_metadata == {"editor": "pycharm"}

    def test_unsupported_tools_accepted_and_parsed(self) -> None:
        """Hosted tool types without a backend equivalent parse cleanly."""
        params = ResponseCreateParams(
            model="m",
            input="x",
            tools=[
                {"type": "namespace", "name": "ns", "description": "d", "tools": []}  # type: ignore[list-item]
            ],
        )
        assert params.tools


class TestReasoningEffortParity:
    """reasoning.effort accepts the upstream SDK's `max` literal."""

    def test_max_effort_is_accepted(self) -> None:
        """reasoning.effort="max" validates like the other SDK literals."""
        params = ResponseCreateParams(
            model="m", input="x", reasoning=Reasoning(effort="max")
        )
        assert params.reasoning is not None
        assert params.reasoning.effort == "max"


class TestToolAllowedCallersParity:
    """allowed_callers/output_schema/tunnel_id round-trip on tool definitions."""

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
    """ResponseErrorCode includes the full upstream error-code enum."""

    @pytest.mark.parametrize("code", ["data_residency_mismatch", "bio_policy"])
    def test_upstream_only_codes_are_accepted(self, code: str) -> None:
        """Error codes present upstream but previously missing here validate."""
        error = ResponseError(code=code, message="m")  # type: ignore[arg-type]
        assert error.code == code


class TestToolCallerParity:
    """Tool-call items keep the `caller` provenance field when echoed as input."""

    def test_function_call_input_accepts_program_caller(self) -> None:
        """A function_call input item retains a program caller instead of dropping it."""
        item = FunctionCallInput(
            arguments="{}",
            call_id="c1",
            name="f",
            type="function_call",
            caller={"type": "program", "caller_id": "p1"},
        )
        assert item.caller is not None
        assert item.caller.type == "program"
        assert item.caller.caller_id == "p1"  # type: ignore[union-attr]


class TestPromptCacheRetentionParity:
    """prompt_cache_retention uses the OpenAI SDK literal `in_memory`."""

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
    """prompt_cache_options is accepted on the Responses create/compact request bodies."""

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
    """Paginated list responses share a common envelope base without changing wire keys."""

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
    """A previously emitted `additional_tools` output item can be echoed back as input."""

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
    """remixed_from_video_id on Video objects."""

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
