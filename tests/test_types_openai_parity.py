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
    InputTokenCountParams,
    Reasoning,
    ResponseCreateParams,
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
