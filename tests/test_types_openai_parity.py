"""Tests for OpenAI SDK schema-parity fields (unit)."""

import pytest
from pydantic import TypeAdapter, ValidationError

from stdapi.api_errors import UnsupportedParameterError
from stdapi.types.openai_chat_completions import CompletionCreateParams
from stdapi.types.openai_responses import (
    AdditionalTools,
    CompactParams,
    InputTokenCountParams,
    Reasoning,
    ResponseCreateParams,
    ResponseOutputItem,
)
from stdapi.types.openai_videos import Video

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
        item = TypeAdapter(ResponseOutputItem).validate_python(
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
