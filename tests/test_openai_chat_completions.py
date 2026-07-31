"""Tests for the OpenAI Chat Completions route (``POST /v1/chat/completions``).

Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
     https://developers.openai.com/api/reference/resources/chat.md
     stdapi/routes/openai_chat_completions.py:create_chat_completion
"""

import base64
import json as _json
from asyncio import sleep
from contextlib import contextmanager
from secrets import token_hex
from typing import TYPE_CHECKING, Any, ClassVar

import httpx
import pytest
from aiobotocore.session import get_session
from openai import APIError, BadRequestError, NotFoundError, OpenAI
from pybase64 import b64encode
from pydantic import ValidationError

from stdapi.config import SETTINGS
from stdapi.models.chat._adapters._openai_chat_completion import (
    _LEGACY_FUNCTION,
    format_response,
    format_stream,
)
from stdapi.models.chat._adapters._openai_common import (
    JSON_OBJECT_SYSTEM_INSTRUCTION,
    extract_stream_usage,
)
from stdapi.models.chat._default import ChatModel
from stdapi.models.deprecation import DEPRECATED_MODELS
from stdapi.types.openai_chat_completions import CompletionCreateParams
from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Iterator

    from openai.types.chat import ChatCompletion
    from starlette.testclient import TestClient as TestClientType

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef


def _strip_code_fence(text: str) -> str:
    """Strip a wrapping Markdown code fence (e.g. ` ```json `) from model output.

    Args:
        text: Raw model output, possibly fenced.

    Returns:
        ``text`` with a leading/trailing triple-backtick fence removed, or
        ``text`` stripped of surrounding whitespace when it is not fenced.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    lines = lines[1:-1] if len(lines) > 1 and lines[-1].strip() == "```" else lines[1:]
    return "\n".join(lines).strip()


#: Tool declaration reused verbatim so a repeated request can hit the prompt cache.
_CACHE_TOOLS: list[Any] = [
    {
        "type": "function",
        "function": {
            "name": "search_database",
            "description": "Search a database for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "limit": {"type": "integer", "description": "Maximum results"},
                },
                "required": ["query"],
            },
        },
    }
]

#: System-plus-user prompt reused verbatim so a repeated request can hit the cache.
_CACHE_MESSAGES: list[Any] = [
    {
        "role": "system",
        "content": (
            "You are an AI assistant. You are a highly capable, thoughtful, "
            "and nuanced conversational AI with strong reasoning abilities. "
            "Your purpose is to be helpful, harmless, and honest in all interactions."
        ),
    },
    {"role": "user", "content": "What is 2 + 2?"},
]

#: ``max_completion_tokens`` for the official lane, whose model bills reasoning against it.
_OFFICIAL_TOKEN_BUDGET = 1024

#: Third-party HTTPS image used to exercise the gateway's own image downloader.
_REMOTE_IMAGE_URL = (
    "https://raw.githubusercontent.com/JGoutin/asus-s14na-u12-uefi/"
    "refs/heads/master/data/block_diagram.png"
)


@contextmanager
def _xfail_on_invalid_tool_use() -> Iterator[None]:
    """Xfail instead of erroring when the model emits a malformed ``toolUse`` block.

    Bedrock surfaces that model-side failure as a 500 (or, mid-stream, as a relayed
    ``APIError``), which is not something the gateway can prevent or the test can
    retry away.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html
    """
    try:
        yield
    except APIError as exc:
        if "Model produced invalid sequence as part of ToolUse" in str(exc):
            pytest.xfail(str(exc))
        raise


@pytest.fixture(scope="module")
def envelope_completion(openai_client: OpenAI, chat_model: str) -> ChatCompletion:
    """One cheap completion shared by the request-independent envelope assertions.

    ``id``, ``object`` and ``created`` are minted by the gateway rather than by the
    model, so a single billable call is enough to assert all of them.

    Ref: https://developers.openai.com/api/reference/resources/chat.md
         stdapi/routes/openai_chat_completions.py:create_chat_completion
    """
    return openai_client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "user", "content": "Say OK."}],
        max_completion_tokens=16,
    )


def _gather_legacy_stream_info(
    response: Iterable[object],
) -> tuple[list[object], bool, list[str], bool]:
    """Collect streaming info for legacy function_call deltas.

    Args:
        response: Stream of chat completion chunks or DONE tokens.

    Returns:
        A tuple of (chunks, saw_function_delta, args_fragments, has_finish).
    """
    chunks: list[object] = []
    saw_function_delta = False
    args_fragments: list[str] = []
    has_finish = False

    for chunk in response:
        if isinstance(chunk, str) and chunk == "[DONE]":
            break
        chunks.append(chunk)
        choices = getattr(chunk, "choices", None)
        if choices:
            c0 = choices[0]
            delta = c0.delta
            fc = getattr(delta, "function_call", None)
            if fc is not None:
                saw_function_delta = True
                if getattr(fc, "arguments", None):
                    args_fragments.append(fc.arguments)
            if c0.finish_reason is not None:
                has_finish = True

    return chunks, saw_function_delta, args_fragments, has_finish


class TestChatCompletions:
    """Live coverage of ``POST /v1/chat/completions`` served by Bedrock Converse.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         stdapi/models/chat/_adapters/_openai_chat_completion.py
    """

    def test_basic_chat_completion(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """A minimal request returns a single assistant choice with billed usage.

        ``safety_identifier`` is the successor of the deprecated ``user`` field
        and is accepted with the otherwise-default parameter set.

        Ref: https://developers.openai.com/api/docs/guides/safety-best-practices#implement-safety-identifiers
             stdapi/models/chat/_adapters/_openai_chat_completion.py:format_response
        """
        response = openai_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            safety_identifier="test-chat-completion",
        )

        # Validate response structure
        assert response.object == "chat.completion"
        assert len(response.choices) == 1
        assert response.choices[0].index == 0
        assert response.choices[0].message.role == "assistant"
        assert isinstance(response.choices[0].message.content, str)
        assert len(response.choices[0].message.content) > 0
        assert response.choices[0].message.tool_calls is None
        assert response.choices[0].finish_reason in ("stop", "length")

        # Validate usage information
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert (
            response.usage.total_tokens
            == response.usage.prompt_tokens + response.usage.completion_tokens
        )

    def test_multiple_choices_parameter(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """``n=2`` returns two contiguously indexed assistant choices.

        Bedrock Converse has no ``n`` parameter: the gateway issues one Converse
        call per choice and merges their token counts into a single ``usage``.

        Ref: stdapi/models/chat/_adapters/_openai_chat_completion.py:format_response
        """
        response = openai_client.chat.completions.create(
            model=chat_model, messages=[{"role": "user", "content": "Say hello."}], n=2
        )

        # Validate multiple choices
        assert len(response.choices) == 2

        for i, choice in enumerate(response.choices):
            assert choice.index == i
            assert choice.message.role == "assistant"
            assert isinstance(choice.message.content, str)
            assert choice.finish_reason in ("stop", "length")

        # Usage covers both choices
        assert response.usage is not None
        assert response.usage.completion_tokens > 0
        assert (
            response.usage.total_tokens
            == response.usage.prompt_tokens + response.usage.completion_tokens
        )

    def test_system_message_handling(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """A multi-turn history with a ``system`` message and content-part turns is accepted.

        ``system`` is not a Bedrock message role: it is extracted into the
        Converse ``system`` field, while the assistant turn expressed as a text
        content-part array maps to the same Bedrock message shape as a plain
        string turn.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/models/chat/_adapters/_openai_chat_completion.py:map_messages
        """
        response = openai_client.chat.completions.create(
            model=chat_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a pirate. Always respond like a pirate.",
                },
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Ahoy there, matey!"},
                {"role": "user", "content": "How are you today?"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Very well, matey."}],
                },
                {"role": "user", "content": "let's go"},
            ],
        )

        # Validate the whole history was accepted and answered
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
        assert isinstance(response.choices[0].message.content, str)
        assert response.choices[0].message.tool_calls is None
        assert response.choices[0].finish_reason in ("stop", "length")
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert (
            response.usage.total_tokens
            == response.usage.prompt_tokens + response.usage.completion_tokens
        )

    def test_streaming_basic_functionality(
        self, openai_client: OpenAI, chat_model: str, use_official_api: bool
    ) -> None:
        """Streaming starts with a role-only delta and continues with content deltas.

        The gateway emits a synthetic first chunk carrying
        ``delta={"role": "assistant"}`` before relaying any Bedrock event; every
        chunk repeats the same completion id and the ``chat.completion.chunk``
        object type.  The answer is bounded with ``max_completion_tokens`` rather
        than a chunk counter, so the whole stream is consumed and the single
        terminal chunk is observable.  The budget is larger on the official lane
        because its model bills reasoning tokens against ``max_completion_tokens``
        and would otherwise stop on ``length`` before emitting any content delta.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
             https://developers.openai.com/api/docs/guides/reasoning
             stdapi/models/chat/_adapters/_openai_chat_completion.py:format_stream
        """
        response = openai_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Count to 5 slowly."}],
            max_completion_tokens=_OFFICIAL_TOKEN_BUDGET if use_official_api else 64,
            stream=True,
        )

        chunks = []
        accumulated_content = ""
        finish_reasons = []

        for chunk in response:
            # Skip the final "[DONE]" string message
            if isinstance(chunk, str) and chunk == "[DONE]":
                break
            chunks.append(chunk)
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    accumulated_content += delta.content
                if chunk.choices[0].finish_reason is not None:
                    finish_reasons.append(chunk.choices[0].finish_reason)

        # Validate streaming behavior
        assert len(chunks) > 0, "No streaming chunks received"
        assert len(accumulated_content) > 0, "No content accumulated from stream"
        assert chunks[0].choices[0].delta.role == "assistant", (
            "First chunk must announce the assistant role"
        )
        assert chunks[0].choices[0].index == 0
        for chunk in chunks:
            assert chunk.object == "chat.completion.chunk"
            assert chunk.id == chunks[0].id, "All chunks share the completion id"
        assert len(finish_reasons) == 1, (
            f"exactly one terminal chunk is expected, got {finish_reasons}"
        )
        assert finish_reasons[0] in {"stop", "length"}

    def test_stop_sequences_functionality(
        self, openai_client: OpenAI, chat_legacy_model: str
    ) -> None:
        """``stop`` is honored both as a bare string and as a sequence list.

        A single string is wrapped into Bedrock's ``stopSequences`` list, and a
        run stopped by a sequence reports ``finish_reason="stop"`` because
        Bedrock's ``stop_sequence`` reason has no dedicated OpenAI value.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/models/chat/_adapters/_openai_chat_completion.py:_FINISH_REASONS
        """
        # Test single stop string
        response = openai_client.chat.completions.create(
            model=chat_legacy_model,
            messages=[
                {"role": "user", "content": "Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"}
            ],
            stop="5",
        )

        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "stop"
        assert isinstance(response.choices[0].message.content, str)

        # Test multiple stop sequences
        response = openai_client.chat.completions.create(
            model=chat_legacy_model,
            messages=[
                {"role": "user", "content": "List colors: red, blue, green, yellow"}
            ],
            stop=["green", "yellow"],
        )

        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "stop"
        assert isinstance(response.choices[0].message.content, str)

    def test_tools_calling(self, openai_client: OpenAI, chat_vision_model: str) -> None:
        """A full tool round trip: forced call, tool result, then a text answer.

        ``tool_choice="required"`` maps to Bedrock ``toolChoice.any`` and a named
        tool_choice to ``toolChoice.tool``, so both must produce a tool call.
        The follow-up turns cover JSON and plain-text tool results, which the
        gateway sends as Bedrock ``toolResult`` ``json`` and ``text`` blocks.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
             stdapi/models/chat/_adapters/_openai_common.py:parse_tool_content
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City and state",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        response = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=chat_vision_model,
            messages=[{"role": "user", "content": "What's the weather in New York?"}],
            tools=tools,
            tool_choice="required",
        )

        assert len(response.choices) == 1
        required_call = response.choices[0]
        assert required_call.message.tool_calls, (
            "tool_choice='required' must produce a tool call"
        )
        assert required_call.message.tool_calls[0].function.name == "get_weather"
        assert required_call.finish_reason in ("tool_calls", "stop")

        # Force a specific function to be called to make the behavior deterministic across providers
        forced_tool_choice = {"type": "function", "function": {"name": "get_weather"}}

        response = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=chat_vision_model,
            messages=[{"role": "user", "content": "What's the weather in New York?"}],
            tools=tools,
            tool_choice=forced_tool_choice,
        )

        assert len(response.choices) == 1
        choice = response.choices[0]

        # When a tool is called, content is typically None and tool_calls is populated
        assert choice.message.role == "assistant"
        assert choice.finish_reason in ("tool_calls", "stop")
        assert choice.message.tool_calls is not None
        assert isinstance(choice.message.tool_calls, list)
        assert len(choice.message.tool_calls) >= 1
        first_call = choice.message.tool_calls[0]
        assert first_call.type == "function"
        assert first_call.id
        assert first_call.function.name == "get_weather"
        assert isinstance(first_call.function.arguments, str)
        # Arguments should be valid JSON string
        args_dict = _json.loads(first_call.function.arguments)
        assert isinstance(args_dict, dict)
        assert "location" in args_dict, "The required schema property must be filled"
        assert isinstance(args_dict["location"], str)

        # Simulate tool execution and send tool result back to the model
        tool_result = {
            "location": "New York",
            "temperature_c": 20,
            "condition": "sunny",
        }
        followup_messages = [
            {"role": "user", "content": "What's the weather in New York?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": first_call.id,
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": first_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": first_call.id,
                "content": _json.dumps(tool_result),
            },
        ]

        final = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=followup_messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
        )
        assert len(final.choices) == 1
        final_choice = final.choices[0]
        assert final_choice.message.role == "assistant"
        assert isinstance(final_choice.message.content, str)
        assert final_choice.message.content is not None
        assert final_choice.message.tool_calls is None
        assert final_choice.finish_reason == "stop"

        # Test tool result with non-JSON content (plain text)
        # This validates _req_parse_tool_content handles non-JSON correctly
        followup_messages_plain_text = [
            {"role": "user", "content": "What's the weather in New York?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": first_call.id,
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": first_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": first_call.id,
                "content": "Weather is sunny, 20 degrees",  # Plain text, not JSON
            },
        ]

        final_plain = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=followup_messages_plain_text,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
        )
        assert final_plain.choices[0].message.content is not None
        assert final_plain.choices[0].message.tool_calls is None
        assert final_plain.usage is not None
        assert final_plain.usage.prompt_tokens > 0

    def test_legacy_functions_parameter(
        self, openai_client: OpenAI, chat_legacy_model: str
    ) -> None:
        """The deprecated ``functions``/``function_call`` pair round-trips in legacy shape.

        ``functions`` and ``tools`` share one Bedrock ``toolConfig``, but a
        request that used ``functions`` is tracked so the response reports
        ``message.function_call`` (never ``tool_calls``) and the ``tool_use``
        stop reason becomes ``finish_reason="function_call"``.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             stdapi/models/chat/_adapters/_openai_chat_completion.py:map_bedrock_stop_reason
        """
        functions = [
            {
                "name": "calculate_sum",
                "description": "Calculate sum of two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            }
        ]

        response = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=chat_legacy_model,
            messages=[{"role": "user", "content": "What is 2 + 3?"}],
            functions=functions,
            function_call="auto",
        )

        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.message.role == "assistant"
        assert choice.message.tool_calls is None, (
            "A legacy `functions` request must never report `tool_calls`"
        )

        # Force a specific legacy function call
        response = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=chat_legacy_model,
            messages=[{"role": "user", "content": "What is 2 + 3?"}],
            functions=functions,
            function_call={"name": "calculate_sum"},
        )
        assert len(response.choices) == 1
        choice = response.choices[0]
        # With legacy flow, function_call field is populated
        # finish_reason may be "function_call" (OpenAI)
        # Some models may return a regular assistant message with content (finish_reason 'stop'/'length').
        assert choice.finish_reason in ("function_call", "stop")
        fc = choice.message.function_call
        assert fc is not None, choice.message
        assert choice.message.tool_calls is None
        assert fc.name == "calculate_sum"
        assert isinstance(fc.arguments, str)
        assert isinstance(_json.loads(fc.arguments), dict), (
            "Legacy function arguments must be a JSON object string"
        )

        args = {"a": 2, "b": 3}
        tool_answer = _json.dumps({"result": args["a"] + args["b"]})

        # Build follow-up messages using legacy "function" role message
        followup_messages = [
            {"role": "user", "content": "What is 2 + 3?"},
            {
                "role": "assistant",
                "function_call": {
                    "name": "calculate_sum",
                    "arguments": _json.dumps(args),
                },
            },
            {"role": "function", "name": "calculate_sum", "content": tool_answer},
        ]
        final = openai_client.chat.completions.create(
            model=chat_legacy_model,
            messages=followup_messages,  # type: ignore[arg-type]
            functions=functions,  # type: ignore[arg-type]
            max_completion_tokens=100,
        )
        assert len(final.choices) == 1
        final_choice = final.choices[0]
        assert final_choice.message.role == "assistant"
        assert isinstance(final_choice.message.content, str)
        assert final_choice.message.content is not None
        assert final_choice.message.function_call is None
        assert final_choice.message.tool_calls is None
        assert final_choice.finish_reason in ("stop", "length")

    def test_legacy_functions_streaming(
        self, openai_client: OpenAI, chat_vision_model: str, use_official_api: bool
    ) -> None:
        """Streaming a forced legacy function emits ``delta.function_call`` fragments.

        In legacy mode the gateway streams the tool call as ``function_call``
        argument fragments instead of indexed ``tool_calls`` entries; the
        fragments concatenate into the complete JSON argument object.  The
        official lane gets a larger budget because its model bills reasoning
        tokens against ``max_completion_tokens`` and would otherwise stop on
        ``length`` before emitting the forced call.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#streaming
             https://developers.openai.com/api/docs/guides/reasoning
             stdapi/models/chat/_adapters/_openai_chat_completion.py:format_stream
        """
        functions = [
            {
                "name": "calculate_sum",
                "description": "Calculate sum of two numbers",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
            }
        ]

        response = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=chat_vision_model,
            messages=[{"role": "user", "content": "What is 2 + 3?"}],
            functions=functions,
            function_call={"name": "calculate_sum"},
            stream=True,
            max_completion_tokens=_OFFICIAL_TOKEN_BUDGET if use_official_api else 60,
        )

        chunks, saw_function_delta, args_fragments, has_finish = (
            _gather_legacy_stream_info(response)
        )

        assert len(chunks) > 0, "No streaming chunks received for legacy functions"
        assert saw_function_delta, (
            "A forced legacy function must stream function_call deltas"
        )
        assert getattr(chunks[0], "object", None) == "chat.completion.chunk"
        assert has_finish, "the stream must be consumed to its terminal chunk"

        if args_fragments:
            args_joined = "".join(args_fragments)
            assert isinstance(args_joined, str)
            assert len(args_joined) > 0

            # Validate the joined arguments are parseable
            # With the change to _resp_stream_get_content_block_delta,
            # arguments should still be valid when accumulated
            try:
                parsed_args = _json.loads(args_joined)
                # Should be a valid dict with expected keys
                assert isinstance(parsed_args, dict)
                # For calculate_sum, we expect 'a' and 'b' keys (or empty if streaming incomplete)
                if parsed_args:  # Only check if not empty
                    assert "a" in parsed_args or "b" in parsed_args
            except _json.JSONDecodeError:
                # If not valid JSON, it might be incomplete streaming
                # Accept partial JSON if it starts correctly
                assert args_joined.startswith(("{", '"'))

    def test_empty_messages_error(self, openai_client: OpenAI, chat_model: str) -> None:
        """An empty ``messages`` array is rejected with a 400 before any model call.

        ``messages`` is declared with ``min_length=1``, so the request fails
        request-body validation and the error names the field.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai_chat_completions.py:CompletionCreateParams
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(model=chat_model, messages=[])

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert "messages" in error_body["message"].lower()

    def test_invalid_model_error(self, openai_client: OpenAI) -> None:
        """An unknown model ID is a 404 carrying the ``model_not_found`` code.

        The gateway's status → ``error.type`` table has no 404 entry, so a
        not-found model keeps the default ``invalid_request_error`` type.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
             stdapi/api_providers/openai.py:_STATUS
        """
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.chat.completions.create(
                model="invalid-nonexistent-model",
                messages=[{"role": "user", "content": "Hello"}],
            )

        error = exc_info.value
        assert error.status_code == 404
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert error_body["code"] == "model_not_found"
        assert "model" in error_body["message"].lower()
        assert (
            "exist" in error_body["message"].lower()
            or "access" in error_body["message"].lower()
        )

    @pytest.mark.parametrize("temperature", [-0.1, 3.0])
    def test_invalid_temperature_error(
        self, openai_client: OpenAI, chat_model: str, temperature: float
    ) -> None:
        """Out-of-range ``temperature`` values are rejected with a 400.

        The gateway only enforces ``temperature >= 0`` itself; the upper bound is
        model-specific, so a too-high value is rejected downstream by Bedrock and
        still comes back as a 400 naming the field.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/aws_bedrock.py:AWS_ERROR_MAP
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hello"}],
                temperature=temperature,
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert "temperature" in error_body["message"].lower()

    @pytest.mark.parametrize(
        ("kwargs", "message_token"),
        [
            pytest.param({"top_p": 1.5}, "top", id="top_p"),
            pytest.param({"max_completion_tokens": 0}, "max_", id="max_tokens"),
            pytest.param(
                {"frequency_penalty": 2.5}, "frequency_penalty", id="frequency_penalty"
            ),
            pytest.param(
                {"presence_penalty": -2.5}, "presence_penalty", id="presence_penalty"
            ),
            pytest.param({"logit_bias": {"100": 105}}, "bias", id="logit_bias"),
        ],
    )
    def test_out_of_range_parameter_error(
        self,
        openai_client: OpenAI,
        chat_model: str,
        kwargs: dict[str, Any],
        message_token: str,
    ) -> None:
        """An out-of-range sampling parameter comes back as a 400 naming the field.

        Only ``max_completion_tokens`` is bounded by the gateway itself (declared
        ``ge=1``).  ``top_p`` is clamped by Bedrock's ``inferenceConfig``, while the
        penalties and ``logit_bias`` travel in ``additionalModelRequestFields`` and
        are rejected by the model provider — all three paths must surface as the
        same OpenAI 400 envelope.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html
             stdapi/aws_bedrock.py:set_inference_configuration
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hello"}],
                **kwargs,
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert message_token in error_body["message"].lower()

    def test_streaming_with_tool_calls(
        self, openai_client: OpenAI, chat_vision_model: str
    ) -> None:
        """A streamed tool call arrives as indexed ``tool_calls`` deltas and a finish chunk.

        Bedrock content-block indices are remapped to contiguous OpenAI
        ``tool_calls[].index`` values, ``id``/``function.name`` appear on the
        opening delta and the arguments follow as string fragments. The tool takes
        no parameters, which the gateway declares to Bedrock with the
        ``{"type": "object"}`` fallback schema.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
             stdapi/models/chat/_adapters/_openai_chat_completion.py:format_stream
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]

        response = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=[{"role": "user", "content": "What time is it?"}],
            tools=tools,  # type: ignore[arg-type]
            stream=True,
        )

        chunks = list(response)
        assert len(chunks) > 0
        # Validate streamed chunks have expected structure
        has_tool_delta = False
        has_finish = False
        for ch in chunks:
            # Each chunk should have choices
            choices = getattr(ch, "choices", None)
            if not choices:
                continue
            assert getattr(ch, "object", None) == "chat.completion.chunk"
            assert getattr(ch, "id", None) == getattr(chunks[0], "id", None)
            assert len(choices) >= 1
            c0 = choices[0]
            # role may appear only once as a delta; tolerate None
            if getattr(c0.delta, "tool_calls", None):
                has_tool_delta = True
                # Validate tool call delta
                t = c0.delta.tool_calls[0]
                assert t.index >= 0
                # Type may stream as None on some chunks; when present it must be 'function'
                assert t.type in (None, "function")
                # id or function fields may stream partially
                assert (t.id is not None) or (t.function is not None)

                # Validate function arguments format in delta
                if t.function is not None and hasattr(t.function, "arguments"):
                    func_args = t.function.arguments
                    if func_args:
                        # Arguments in streaming should be a string (delta fragment) or dict/object
                        # With the change, arguments should be the raw input value, not JSON-encoded
                        assert isinstance(func_args, (str, dict)) or func_args is None
            if c0.finish_reason is not None:
                has_finish = True
                # finish reason must be tool_calls or stop/length depending on stage
                assert c0.finish_reason in ("tool_calls", "stop", "length")
        assert has_tool_delta or any(
            (
                getattr(getattr(ch, "choices", [None])[0].delta, "function_call", None)
                is not None
            )
            for ch in chunks
        )
        # The whole stream was consumed, so the finish chunk must have been seen.
        assert has_finish, "Stream ended without a finish_reason chunk"

    def test_multiple_tool_calls_flow(
        self, openai_client: OpenAI, chat_vision_model: str
    ) -> None:
        """Two tool calls in one assistant turn with their two results are accepted.

        Bedrock requires every ``toolUse`` block to be answered by a matching
        ``toolResult`` block in the next turn; the gateway groups both results into
        a single user message, and mixing a JSON and a plain-text result exercises
        both ``toolResult`` content shapes.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
             stdapi/models/chat/_adapters/_openai_common.py:parse_tool_content
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_news",
                    "description": "Get latest headline",
                    "parameters": {
                        "type": "object",
                        "properties": {"topic": {"type": "string"}},
                        "required": ["topic"],
                    },
                },
            },
        ]

        # Pre-generate stable IDs for the two calls
        call1_id = "call_1"
        call2_id = "call_2"

        # Build a conversation where the assistant previously called two tools
        messages = [
            {
                "role": "user",
                "content": "Using tools, summarize the weather in Paris and the top news about technology.",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call1_id,
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": _json.dumps({"location": "Paris"}),
                        },
                    },
                    {
                        "id": call2_id,
                        "type": "function",
                        "function": {
                            "name": "get_news",
                            "arguments": _json.dumps({"topic": "technology"}),
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call1_id,
                "content": _json.dumps(
                    {"location": "Paris", "temperature_c": 18, "condition": "cloudy"}
                ),
            },
            {
                "role": "tool",
                "tool_call_id": call2_id,
                "content": _json.dumps(
                    {"headline": "Breakthrough in AI chips announced."}
                ),
            },
        ]

        final = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=120,
        )

        assert len(final.choices) == 1
        choice = final.choices[0]
        assert choice.message.role == "assistant"
        assert choice.message.tool_calls is None
        assert isinstance(choice.message.content, str)
        assert choice.finish_reason in ("stop", "length")
        assert final.usage is not None
        assert final.usage.prompt_tokens > 0, "Both tool results are billed as input"

        # Test mixed JSON and non-JSON tool results
        # This validates _req_parse_tool_content handles both formats
        messages_mixed = [
            {
                "role": "user",
                "content": "Using tools, check weather and get a simple status.",
            },
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_weather",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": _json.dumps({"location": "Paris"}),
                        },
                    },
                    {
                        "id": "call_status",
                        "type": "function",
                        "function": {
                            "name": "get_news",
                            "arguments": "{}",  # Empty JSON object as string
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_weather",
                "content": _json.dumps({"location": "Paris", "temperature_c": 18}),
            },
            {
                "role": "tool",
                "tool_call_id": "call_status",
                "content": "System operational",  # Plain text, not JSON
            },
        ]

        mixed_result = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=messages_mixed,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=120,
        )
        assert len(mixed_result.choices) == 1
        assert mixed_result.choices[0].message.content is not None
        assert mixed_result.choices[0].finish_reason in ("stop", "length", "tool_calls")

    def test_tool_arguments_edge_cases(
        self, openai_client: OpenAI, chat_vision_model: str
    ) -> None:
        """Plain-text and malformed-JSON tool results are replayed as Bedrock text blocks.

        ``parse_tool_content`` only emits a Bedrock ``toolResult`` ``json`` block
        for a JSON object; anything else (plain prose, a broken JSON fragment)
        becomes a ``text`` block, so Bedrock accepts the turn instead of rejecting
        the conversation.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
             stdapi/models/chat/_adapters/_openai_common.py:parse_tool_content
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "process_data",
                    "description": "Process some data",
                    "parameters": {
                        "type": "object",
                        "properties": {"data": {"type": "string"}},
                    },
                },
            }
        ]

        # Test 1: Valid JSON string arguments with plain text result
        response = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=[
                {"role": "user", "content": "Process this"},
                {
                    "role": "assistant",
                    "content": "",  # Explicit content field, but empty
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "process_data",
                                "arguments": '{"data": "test"}',  # Valid JSON string
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "Processed successfully",  # Plain text result
                },
            ],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=256,
        )
        assert response.choices[0].message.content is not None
        assert response.choices[0].message.role == "assistant"
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0

        # Test 2: Tool result with invalid JSON (should be treated as plain text)
        response2 = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=[
                {"role": "user", "content": "What happened?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "process_data",
                                "arguments": '{"data": "test2"}',
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_2",
                    "content": "Error: Invalid {bracket",  # Invalid JSON
                },
            ],
            tools=tools,  # type: ignore[arg-type]
            # Room for a follow-up tool call: truncating one mid-JSON makes Bedrock
            # fail the turn with "Model produced invalid sequence as part of ToolUse".
            max_completion_tokens=256,
        )
        assert response2.choices[0].message.content is not None
        assert response2.choices[0].message.role == "assistant"
        assert response2.usage is not None
        assert response2.usage.prompt_tokens > 0

    def test_conflicting_tools_and_functions_error(
        self, openai_client: OpenAI, chat_vision_model: str
    ) -> None:
        """Sending both ``tools`` and legacy ``functions`` is a 400 naming both parameters.

        ``functions`` is the deprecated spelling of ``tools`` and both map to the
        same Bedrock ``toolConfig``, so the gateway refuses the ambiguous request
        rather than silently picking one.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             stdapi/types/openai_chat_completions.py:CompletionCreateParams
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]
        functions = [
            {
                "name": "get_time",
                "description": "Get current time",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        ]

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_vision_model,
                messages=[{"role": "user", "content": "What time is it?"}],
                tools=tools,  # type: ignore[arg-type]
                functions=functions,  # type: ignore[arg-type]
            )

        error = exc_info.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        assert body.get("code") is None
        # Message should mention both parameters
        msg = body["message"].lower()
        assert "functions" in msg
        assert "tools" in msg

    def test_system_message_with_text_parts(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """A ``system`` message given as text content parts is accepted.

        Each part becomes its own Bedrock ``system`` content block, so a
        multi-part system prompt needs no client-side concatenation.

        Ref: https://developers.openai.com/api/reference/resources/chat.md
             stdapi/models/chat/_adapters/_openai_chat_completion.py:_extract_system_content_blocks
        """
        response = openai_client.chat.completions.create(
            model=chat_model,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a concise assistant."},
                        {"type": "text", "text": "Answer briefly."},
                    ],
                },
                {"role": "user", "content": "Explain what an API is in one sentence."},
            ],
        )
        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.message.role == "assistant"
        assert isinstance(choice.message.content, str)
        assert len(choice.message.content) > 0
        assert choice.message.tool_calls is None
        assert choice.finish_reason in ("stop", "length")
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert (
            response.usage.total_tokens
            == response.usage.prompt_tokens + response.usage.completion_tokens
        )

    def test_allowed_tools_auto(
        self, openai_client: OpenAI, chat_vision_model: str, use_official_api: bool
    ) -> None:
        """An ``allowed_tools`` tool_choice is rejected with a 400 by this gateway.

        Upstream documents ``tool_choice.type="allowed_tools"`` as a way to
        restrict the callable subset, but Bedrock's ``toolChoice`` union only has
        ``auto``/``any``/``tool``, so the gateway refuses it instead of silently
        widening the choice. Against the official API the same request succeeds.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
             stdapi/types/openai_chat_completions.py:CompletionCreateParams._validate_tool_choice
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather information",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]
        tool_choice = {
            "type": "allowed_tools",
            "allowed_tools": {
                "mode": "auto",
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
            },
        }

        if not use_official_api:
            with pytest.raises(BadRequestError) as exc_info:
                openai_client.chat.completions.create(  # type: ignore[call-overload]
                    model=chat_vision_model,
                    messages=[
                        {"role": "user", "content": "What is the weather in Paris?"}
                    ],
                    tools=tools,
                    tool_choice=tool_choice,
                )
            error = exc_info.value
            assert error.status_code == 400
            body = error.body
            assert isinstance(body, dict)
            assert body["type"] == "invalid_request_error"
            assert "allowed_tools" in body["message"].lower()
            assert body["code"] is None
            return

        response = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=chat_vision_model,
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=tools,
            tool_choice=tool_choice,
        )
        assert hasattr(response, "choices")
        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.message.role == "assistant"
        if choice.message.tool_calls is not None:
            assert isinstance(choice.message.tool_calls, list)
            assert len(choice.message.tool_calls) >= 1
            first_call = choice.message.tool_calls[0]
            assert first_call.type == "function"
            assert first_call.id
            assert first_call.function.name in ("get_weather", "get_time")
            assert isinstance(first_call.function.arguments, str)
        else:
            # No tool call; ensure assistant content present
            assert isinstance(choice.message.content, str)
            assert len(choice.message.content) > 0
        assert choice.finish_reason in ("stop", "tool_calls", "length")
        assert response.usage is not None

    def test_custom_tool_choice_supported(
        self, openai_client: OpenAI, chat_vision_model: str, use_official_api: bool
    ) -> None:
        """A ``custom`` tool with a matching ``tool_choice`` is rejected with a 400 here.

        Custom (free-form / grammar) tools have no Bedrock ``toolSpec``
        equivalent, so both the tool definition and the custom ``tool_choice`` are
        refused. Against the official API the call succeeds and may return a
        ``type="custom"`` tool call.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#custom-tools
             stdapi/types/openai_chat_completions.py:CompletionCreateParams._validate_no_custom_tools
        """
        tools = [
            {
                "type": "custom",
                "custom": {
                    "name": "my_custom_tool",
                    "description": "Custom tool for demonstration",
                    "format": {"type": "text"},
                },
            }
        ]
        tool_choice = {"type": "custom", "custom": {"name": "my_custom_tool"}}

        if not use_official_api:
            with pytest.raises(BadRequestError) as exc_info:
                openai_client.chat.completions.create(  # type: ignore[call-overload]
                    model=chat_vision_model,
                    messages=[
                        {
                            "role": "user",
                            "content": "Call the custom tool with a short input.",
                        }
                    ],
                    tools=tools,
                    tool_choice=tool_choice,
                )
            error = exc_info.value
            assert error.status_code == 400
            body = error.body
            assert isinstance(body, dict)
            assert body["type"] == "invalid_request_error"
            assert "custom" in body["message"].lower()
            assert body["code"] is None
            return

        response = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=chat_vision_model,
            messages=[
                {"role": "user", "content": "Call the custom tool with a short input."}
            ],
            tools=tools,
            tool_choice=tool_choice,
        )
        assert hasattr(response, "choices")
        assert len(response.choices) >= 1
        choice = response.choices[0]
        assert choice.message.role == "assistant"
        if choice.message.tool_calls is not None:
            assert isinstance(choice.message.tool_calls, list)
            assert len(choice.message.tool_calls) >= 1
            tc = choice.message.tool_calls[0]
            # For custom tool, type must be 'custom' and fields present
            assert tc.type == "custom"
            assert tc.id
            assert tc.custom.name == "my_custom_tool"
            assert isinstance(tc.custom.input, str)
            assert len(tc.custom.input) >= 0
        else:
            # No tool call; ensure assistant content present
            assert isinstance(choice.message.content, str)
            assert len(choice.message.content) > 0
        assert choice.finish_reason in ("stop", "tool_calls", "length")
        assert response.usage is not None

    def test_multimodal_with_https_image_url(
        self, openai_client: OpenAI, chat_vision_model: str
    ) -> None:
        """An ``image_url`` pointing at an HTTPS image is fetched and sent as an image block.

        OpenAI passes the URL to the model provider; Bedrock Converse accepts only
        inline bytes or S3, so the gateway downloads the image itself and converts
        it into a Bedrock ``image`` content block.

        The URL is fetched here first: a third-party outage or a repository rename
        must skip rather than report a gateway failure.

        Ref: https://developers.openai.com/api/reference/resources/chat.md
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
             stdapi/input_file.py:InputFile.to_bedrock_content_block
        """
        try:
            probe = httpx.get(_REMOTE_IMAGE_URL, timeout=15, follow_redirects=True)
            probe.raise_for_status()
        except httpx.HTTPError as exc:
            pytest.skip(f"Remote image is unreachable: {exc}")
        assert probe.headers["content-type"].startswith("image/"), (
            "the fixture URL must still serve an image"
        )

        response = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {"type": "image_url", "image_url": {"url": _REMOTE_IMAGE_URL}},
                    ],
                }
            ],
        )

        # Validate successful assistant response structure
        assert len(response.choices) >= 1
        choice = response.choices[0]
        assert choice.message.role == "assistant"
        assert isinstance(choice.message.content, str)
        assert len(choice.message.content) > 0
        assert choice.finish_reason in ("stop", "length")
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0, "The image is billed as input tokens"

    def test_multimodal_with_data_url_base64_success(
        self,
        openai_client: OpenAI,
        chat_vision_model: str,
        sample_image_file_base64: str,
    ) -> None:
        """An ``image_url`` holding a base64 ``data:`` URL is decoded and sent to the model.

        Ref: https://developers.openai.com/api/reference/resources/chat.md
             stdapi/input_file.py:InputFile
        """
        response = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this tiny image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": sample_image_file_base64},
                        },
                    ],
                }
            ],
        )
        assert len(response.choices) >= 1
        assert response.choices[0].message.role == "assistant"
        assert isinstance(response.choices[0].message.content, str)
        assert len(response.choices[0].message.content) > 0
        assert response.choices[0].finish_reason in ("stop", "length")
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0

    def test_file_part_audio(
        self, openai_client: OpenAI, chat_audio_model: str, sample_audio_mp3_file: bytes
    ) -> None:
        """An ``input_audio`` content part is forwarded as a Bedrock audio block.

        The declared ``format`` ("mp3") is turned into the audio content type, so
        an audio-capable model answers about the recording instead of the request
        being rejected as an unsupported media type.

        Ref: https://developers.openai.com/api/docs/guides/audio
             stdapi/models/chat/_adapters/_openai_chat_completion.py:_convert_content_part
        """
        b64 = b64encode(sample_audio_mp3_file).decode("utf-8")
        response = openai_client.chat.completions.create(
            model=chat_audio_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": b64, "format": "mp3"},
                        },
                        {"type": "text", "text": "What's in this file?"},
                    ],
                }
            ],
            max_completion_tokens=100,
        )
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content
        assert response.choices[0].finish_reason in ("stop", "length")
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0, "The audio is billed as input tokens"

    @pytest.mark.gateway("Audio file handling as file not supported by OpenAI API")
    def test_file_part_audio_as_file(
        self,
        openai_client: OpenAI,
        chat_audio_model: str,
        sample_audio_mp3_file_base64: str,
    ) -> None:
        """An audio payload sent through a ``file`` part is routed to the audio block.

        Upstream reserves ``file`` parts for documents, but the gateway detects the
        MIME type of the payload and builds the matching Bedrock block, so an MP3
        supplied as a file is treated exactly like an ``input_audio`` part.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
             stdapi/input_file.py:InputFile.to_bedrock_content_block
        """
        response = openai_client.chat.completions.create(
            model=chat_audio_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file": {
                                "file_data": sample_audio_mp3_file_base64,
                                "filename": "audio.mp3",
                            },
                        },
                        {"type": "text", "text": "What's in this file?"},
                    ],
                }
            ],
            max_completion_tokens=100,
        )
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content
        assert response.choices[0].finish_reason in ("stop", "length")
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0

    def test_multimodal_with_http_image_url_error(
        self, openai_client: OpenAI, chat_vision_model: str
    ) -> None:
        """An unreachable ``image_url`` host is reported as a 400, not a 5xx.

        Because the gateway (not the model provider) downloads the image, a failed
        fetch is treated as a client-side problem and reported in the OpenAI
        envelope instead of leaking a transport error as a 5xx.

        Ref: stdapi/input_file.py:_HttpSource
             stdapi/security.py:ssrf_blocked_status
        """
        http_image = f"https://{token_hex(16)}.eu-west-3.amazonaws.com/{token_hex(16)}"

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is in this image?"},
                            {"type": "image_url", "image_url": {"url": http_image}},
                        ],
                    }
                ],
                max_completion_tokens=32,
            )

        error = exc_info.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        message = body["message"].lower()
        assert any(token in message for token in ("download", "image", "url")), (
            f"Unexpected rejection reason: {body['message']}"
        )

    def test_multimodal_with_invalid_data_url_base64_error(
        self, openai_client: OpenAI, chat_vision_model: str
    ) -> None:
        """A ``data:`` image URL with undecodable base64 is a 400 naming the payload.

        The gateway decodes the payload itself, so the failure is caught before
        any Bedrock call and reported without the raw payload.

        Ref: stdapi/input_file.py:_DataUriSource
             stdapi/utils.py:b64decode
        """
        invalid_data_url = "data:image/png;base64,@@@not-base64@@@"

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Check this image."},
                            {
                                "type": "image_url",
                                "image_url": {"url": invalid_data_url},
                            },
                        ],
                    }
                ],
                max_completion_tokens=32,
            )

        error = exc_info.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        assert "base64" in body["message"].lower() or "data" in body["message"].lower()

    @pytest.mark.gateway("File content part shape is implementation-specific here")
    @pytest.mark.parametrize("bad_b64", ["@@@", "!", "==?"])
    def test_file_part_invalid_base64_error(
        self, openai_client: OpenAI, chat_model: str, bad_b64: str
    ) -> None:
        """A ``file`` part whose ``file_data`` is undecodable base64 is a 400.

        MIME sniffing decodes only a prefix and tolerates stray characters, so the
        payload must sniff as a supported document type to reach the strict decode:
        a valid ``text/plain`` payload with a non-base64 suffix is accepted by the
        sniffer and then rejected by the full decode. Skipped against the official
        API, where this file-part shape does not exist.

        Ref: stdapi/input_file.py:_Base64Source._read
             stdapi/utils.py:b64decode
        """
        file_data = b64encode(b"A short text document.\n").decode("utf-8") + bad_b64
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "file",
                                "file": {"file_data": file_data, "filename": "bad.txt"},
                            }
                        ],
                    }
                ],
                max_completion_tokens=16,
            )
        error = exc_info.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        message = body["message"].lower()
        assert "base64" in message, f"Unexpected rejection reason: {body['message']}"
        assert bad_b64 not in body["message"], "Payload echoed in the error message"

    @pytest.mark.gateway("File content part shape is implementation-specific here")
    def test_file_part_unsupported_mime_error(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """A file whose sniffed MIME type is not a Bedrock document format is a 400.

        The payload is a minimal glTF binary, which detects as
        ``model/gltf-binary``: not an image, video or audio block and not one of
        Bedrock's ``DocumentBlock`` formats, so the gateway rejects it up front
        instead of letting Bedrock fail the call.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html
             stdapi/input_file.py:InputFile.to_bedrock_content_block
        """
        # Create a minimal glTF binary header to trigger model/gltf-binary detection
        # Magic: glTF (0x46546C67), version: 2, length: 20 bytes
        gltf_bytes = (
            b"glTF"  # Magic
            b"\x02\x00\x00\x00"  # Version 2
            b"\x14\x00\x00\x00"  # Total length: 20 bytes
            b"\x00\x00\x00\x00"  # Chunk length: 0
        )
        b64 = b64encode(gltf_bytes).decode("utf-8")
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "file",
                                "file": {"file_data": b64, "filename": "model.glb"},
                            }
                        ],
                    }
                ],
                max_completion_tokens=16,
            )
        error = exc_info.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        assert (
            "mime" in body["message"].lower()
            or "unsupported" in body["message"].lower()
        )
        assert body["code"] is None

    def test_parallel_tool_calls_false_is_accepted(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """``parallel_tool_calls=False`` completes rather than failing the request.

        Upstream never rejects the flag, and the Responses API here has always
        accepted it, so refusing it on Chat Completions alone broke requests that
        are valid upstream. Models able to constrain tool use honor it; the others
        ignore it, and the response still reports the tool calls actually made, so
        a client depending on sequential tool use can detect that it did not get
        it. Runs on the official lane too, which is what pins the upstream half of
        that claim.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#parallel-function-calling
             stdapi/types/openai_chat_completions.py:CompletionCreateParams
        """
        response = openai_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Hello"}],
            # Upstream rejects the flag outright when no tools are declared
            # ("'parallel_tool_calls' is only allowed when 'tools' are
            # specified"), so a tool is required to exercise the value itself.
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            parallel_tool_calls=False,
            max_completion_tokens=16,
        )
        assert response.choices[0].message.role == "assistant"

    @pytest.mark.gateway("Project-specific restriction: stream with n>1 unsupported")
    def test_validation_stream_n_gt1_error(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """``n>1`` together with ``stream=True`` is rejected with a 400.

        Multiple choices are produced by issuing one Converse call per choice,
        which cannot be interleaved into a single SSE stream, so the combination is
        refused instead of silently returning one choice.

        Ref: stdapi/types/openai_chat_completions.py:CompletionCreateParams._unsupported
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hello"}],
                n=2,
                stream=True,
            )
        error = exc_info.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        assert "stream" in body["message"].lower()
        assert body["code"] is None

    def test_response_format_json_object(
        self, openai_client: OpenAI, chat_reasoning_model: str, use_official_api: bool
    ) -> None:
        """``response_format={"type": "json_object"}`` returns the prompted JSON object.

        The word "json" must appear in the input or OpenAI rejects the request
        with a 400.  OpenAI's JSON mode constrains the syntax only, so the object
        carries whatever the prompt asked for.  Upstream never wraps ``json_object``
        output in a Markdown code fence, so the official lane requires the raw
        prefix; the gateway's Bedrock-backed models are not constrained the same
        way, so that lane tolerates a fence -- a common, harmless way models wrap
        JSON -- rather than requiring an exact-prefix match.  The token budget is
        larger on the official lane because ``gpt-5-nano`` bills reasoning tokens
        against ``max_completion_tokens`` and would otherwise stop on ``length``
        before emitting any content.

        Ref: https://developers.openai.com/api/docs/guides/structured-outputs
             https://developers.openai.com/api/docs/guides/reasoning
             stdapi/models/chat/_adapters/_openai_chat_completion.py:build_output_config
        """
        response = openai_client.chat.completions.create(
            model=chat_reasoning_model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Reply in json with exactly this object and nothing "
                        'else: {"status": "ok"}'
                    ),
                }
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=_OFFICIAL_TOKEN_BUDGET if use_official_api else 64,
        )
        content = response.choices[0].message.content
        assert content
        unfenced = content if use_official_api else _strip_code_fence(content)
        assert unfenced.startswith("{"), (
            f"json_object output must not be wrapped in prose: {content!r}"
        )
        parsed = _json.loads(unfenced)
        assert isinstance(parsed, dict), f"json_object must be an object: {parsed!r}"
        assert "status" in parsed, (
            f"json_object must carry the prompted content: {parsed!r}"
        )

    @pytest.mark.gateway("Unsupported fields are project-specific here")
    def test_unsupported_seed_error(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """``seed`` is accepted by the gateway but rejected by the model as a 400.

        ``seed`` has no Bedrock ``inferenceConfig`` slot, so it is passed through in
        ``additionalModelRequestFields``; a model that does not declare it answers
        with a ``ValidationException``, which the gateway maps to a 400.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html
             stdapi/aws_bedrock.py:AWS_ERROR_MAP
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_model, messages=[{"role": "user", "content": "Hi"}], seed=123
            )
        assert exc_info.value.status_code == 400
        body = exc_info.value.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        assert "the model returned" in str(body["message"]).lower(), (
            "The rejection must come from the model, not from request validation"
        )

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            pytest.param("verbosity", "high", id="verbosity"),
            pytest.param(
                "web_search_options",
                {
                    "search_context_size": "low",
                    "user_location": {
                        "type": "approximate",
                        "approximate": {"city": "x", "country": "US"},
                    },
                },
                id="web_search_options",
            ),
            pytest.param(
                "prediction", {"type": "content", "content": "abc"}, id="prediction"
            ),
        ],
    )
    @pytest.mark.gateway("Unsupported fields are project-specific here")
    def test_unsupported_parameter_error(
        self, openai_client: OpenAI, chat_model: str, param: str, value: object
    ) -> None:
        """Blocklisted upstream parameters are refused up front, naming the field.

        ``verbosity``, ``web_search_options`` and ``prediction`` are documented by
        OpenAI but have no Bedrock counterpart, so the gateway rejects them rather
        than ignoring them — dropping ``prediction`` in particular would still bill
        the rejected prediction tokens.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             https://developers.openai.com/api/docs/guides/predicted-outputs
             stdapi/types/openai_chat_completions.py:CompletionCreateParams._UNSUPPORTED
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(  # type: ignore[call-overload]
                model=chat_model,
                messages=[{"role": "user", "content": "Hi"}],
                **{param: value},
            )
        assert exc_info.value.status_code == 400
        body = exc_info.value.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        assert body["code"] == "unsupported_parameter"
        assert body["param"] == param
        assert "unsupported parameter" in body["message"].lower()
        assert body.keys() >= {"message", "type", "param", "code"}, (
            "The gateway envelope always carries all four keys"
        )

    @pytest.mark.gateway("Unsupported fields are project-specific here")
    def test_unsupported_top_logprobs_error(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """``top_logprobs`` is forwarded to the model, which rejects it as a 400.

        Unlike ``logprobs`` the gateway does not blocklist ``top_logprobs``: it
        travels in ``additionalModelRequestFields`` and only models that declare
        the field accept it, so this model answers with a ``ValidationException``.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             stdapi/aws_bedrock.py:AWS_ERROR_MAP
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hi"}],
                top_logprobs=5,
            )
        assert exc_info.value.status_code == 400
        body = exc_info.value.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        assert "the model returned" in str(body["message"]).lower(), (
            "The rejection must come from the model, not from request validation"
        )

    def test_prompt_cache_key_with_long_system_prompt(
        self, openai_client: OpenAI, chat_model: str, use_official_api: bool
    ) -> None:
        """A repeated request under the same ``prompt_cache_key`` reports cached prompt tokens.

        ``prompt_cache_key`` is not an opaque bucket here: it is parsed as a
        dot-separated selector of the cacheable sections, and any unrecognized
        non-empty value (``"default"``) enables all of them, so the gateway inserts
        Bedrock ``cachePoint`` blocks after the system prompt and the tool config.
        Bedrock excludes cache reads from ``inputTokens``, so the gateway adds them
        back to keep OpenAI's "``prompt_tokens`` includes cached tokens" contract.

        Ref: https://developers.openai.com/api/docs/guides/prompt-caching#improve-cache-hit-rates-with-a-prompt-cache-key
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
             stdapi/models/chat/_adapters/_openai_common.py:parse_prompt_cache_key
        """
        # First request - should cache the prompt
        with _xfail_on_invalid_tool_use():
            response1 = openai_client.chat.completions.create(
                model=chat_model,
                messages=_CACHE_MESSAGES,
                tools=_CACHE_TOOLS,
                prompt_cache_key="default",
                max_completion_tokens=2048,
            )

        assert len(response1.choices) == 1
        assert response1.choices[0].message.role == "assistant"
        assert response1.usage is not None
        assert response1.usage.prompt_tokens > 0

        # Second request - should use cached prompt
        with _xfail_on_invalid_tool_use():
            response2 = openai_client.chat.completions.create(
                model=chat_model,
                messages=_CACHE_MESSAGES,
                tools=_CACHE_TOOLS,
                prompt_cache_key="default",
                max_completion_tokens=2048,
            )

        assert len(response2.choices) == 1
        assert response2.usage is not None
        usage_details = getattr(response2.usage, "prompt_tokens_details", None)
        assert usage_details is not None, "prompt_tokens_details not found in usage"
        cached_tokens = getattr(usage_details, "cached_tokens", 0)
        if use_official_api and cached_tokens == 0:
            pytest.xfail(
                "Cached tokens may not be available when testing against OpenAI API"
            )
        assert cached_tokens > 0, f"Expected cached_tokens > 0, got {cached_tokens}"
        assert response2.usage.prompt_tokens >= cached_tokens, (
            "prompt_tokens must include the cached prefix"
        )

    def test_prompt_cache_key_with_long_system_prompt_streaming(
        self, openai_client: OpenAI, chat_model: str, use_official_api: bool
    ) -> None:
        """Streaming reports the same cached prompt tokens in its trailing usage chunk.

        Streaming usage is rebuilt from the Bedrock ``metadata`` event rather than
        from a Converse response, so the cache-read accounting has to be applied a
        second time in that code path.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStreamMetadataEvent.html
             stdapi/models/chat/_adapters/_openai_common.py:extract_stream_usage
        """
        # First streaming request - should cache the prompt
        with _xfail_on_invalid_tool_use():
            stream1 = openai_client.chat.completions.create(
                model=chat_model,
                messages=_CACHE_MESSAGES,
                tools=_CACHE_TOOLS,
                prompt_cache_key="default",
                stream=True,
                stream_options={"include_usage": True},
                max_completion_tokens=50,
            )

            # Consume the stream and keep the last chunk, which carries usage
            last_chunk1 = None
            for chunk in stream1:
                if isinstance(chunk, str) and chunk == "[DONE]":
                    break
                last_chunk1 = chunk

        # Validate first stream was successful
        assert last_chunk1 is not None
        assert last_chunk1.object == "chat.completion.chunk"

        # Second streaming request - should use cached prompt
        with _xfail_on_invalid_tool_use():
            stream2 = openai_client.chat.completions.create(
                model=chat_model,
                messages=_CACHE_MESSAGES,
                tools=_CACHE_TOOLS,
                prompt_cache_key="default",
                stream=True,
                stream_options={"include_usage": True},
                max_completion_tokens=50,
            )

            # Consume the stream and keep the last chunk, which carries usage
            last_chunk2 = None
            for chunk in stream2:
                if isinstance(chunk, str) and chunk == "[DONE]":
                    break
                last_chunk2 = chunk

        # Validate second stream uses cache
        assert last_chunk2 is not None
        assert last_chunk2.choices == [], (
            "Usage is reported in its own chunk with empty choices"
        )
        usage2 = getattr(last_chunk2, "usage", None)
        assert usage2 is not None

        # Check that cached tokens were used
        usage_details = getattr(usage2, "prompt_tokens_details", None)
        if use_official_api and usage_details is None:
            pytest.xfail(
                "Prompt tokens details not available when testing against OpenAI API"
            )
        assert usage_details is not None, "prompt_tokens_details not found in usage"
        cached_tokens = getattr(usage_details, "cached_tokens", 0)
        if use_official_api and cached_tokens == 0:
            pytest.xfail("Cached tokens not available when testing against OpenAI API")
        assert cached_tokens > 0, f"Expected cached_tokens > 0, got {cached_tokens}"
        assert usage2.prompt_tokens >= cached_tokens, (
            "prompt_tokens must include the cached prefix"
        )

    @pytest.mark.gateway(
        "gpt-5-nano rejects prompt_cache_options with "
        "'prompt_cache_options is not supported on this model'"
    )
    def test_prompt_cache_explicit_breakpoint(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """An explicit ``prompt_cache_breakpoint`` caches the marked prompt prefix.

        With ``prompt_cache_options.mode="explicit"`` the gateway stops deriving
        cache points from ``prompt_cache_key`` and only honors marked content
        parts. The system text is repeated 200 times to clear Bedrock's per-model
        minimum, below which a cache point is silently not written. ``ttl="30m"``
        is mapped to Bedrock's closest supported TTL (1h).

        Ref: https://developers.openai.com/api/docs/guides/prompt-caching#prompt-cache-breakpoints
             https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
             stdapi/models/chat/_adapters/_openai_common.py:resolve_cache_ttl
        """
        messages: list[Any] = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a meticulous assistant. "
                            "Follow these operating instructions exactly. " * 200
                        ),
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    }
                ],
            },
            {"role": "user", "content": "What is 2 + 2?"},
        ]
        for _ in range(2):
            response = openai_client.chat.completions.create(
                model=chat_model,
                messages=messages,
                prompt_cache_options={"mode": "explicit", "ttl": "30m"},
                max_completion_tokens=32,
            )
        assert response.usage is not None
        usage_details = response.usage.prompt_tokens_details
        assert usage_details is not None
        assert usage_details.cached_tokens
        assert response.usage.prompt_tokens >= usage_details.cached_tokens, (
            "prompt_tokens must include the cached prefix"
        )

    @pytest.mark.gateway("requestMetadata limits are a Bedrock-specific feature")
    def test_metadata_over_bedrock_limits_rejected(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """Metadata breaking the Bedrock ``requestMetadata`` limits returns a clean 400.

        ``metadata`` is forwarded as Bedrock ``requestMetadata``, which allows at
        most 16 pairs and a restricted value charset — and the gateway itself adds
        ``stdapi-ai.*`` tracing keys to that budget. Both violations surface as a
        400 with Bedrock's ``ValidationException`` code rather than a 5xx.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/monitoring.py:build_metadata
        """
        with pytest.raises(BadRequestError) as too_many:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hi"}],
                metadata={f"key-{index}": "value" for index in range(17)},
            )
        assert too_many.value.status_code == 400
        too_many_body = too_many.value.body
        assert isinstance(too_many_body, dict)
        assert too_many_body["type"] == "invalid_request_error"
        assert "validation error detected" in str(too_many_body["message"]).lower()

        with pytest.raises(BadRequestError) as bad_charset:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hi"}],
                metadata={"key": "forbidden!char"},
            )
        assert bad_charset.value.status_code == 400
        bad_charset_body = bad_charset.value.body
        assert isinstance(bad_charset_body, dict)
        assert bad_charset_body["type"] == "invalid_request_error"
        assert "validation error detected" in str(bad_charset_body["message"]).lower()

    @pytest.mark.gateway(
        "gpt-5-nano rejects reasoning_effort='max' with an "
        "unsupported_value error: it only accepts 'minimal', 'low', "
        "'medium' and 'high'"
    )
    def test_reasoning_effort_max_parameter(
        self, openai_client: OpenAI, chat_reasoning_model: str
    ) -> None:
        """The upstream ``max`` effort level is accepted and answered.

        On Claude, ``reasoning_effort`` selects the adaptive reasoning effort:
        only ``minimal`` and ``low`` are downgraded, so ``max`` is forwarded
        unchanged instead of being rejected as an unknown value.

        Ref: https://developers.openai.com/api/docs/guides/reasoning#reasoning-effort
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_reasoning
        """
        response = openai_client.chat.completions.create(
            model=chat_reasoning_model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="max",
            max_completion_tokens=2048,
        )
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].finish_reason in ("stop", "length")
        assert response.usage is not None
        assert response.usage.completion_tokens > 0

    def test_disabled_logprobs_accepted(
        self, openai_client: OpenAI, chat_model: str, use_official_api: bool
    ) -> None:
        """``logprobs: false`` requests the default behavior and is not rejected.

        ``logprobs`` is on the gateway's unsupported list, but a ``false``/``null``
        value is treated as omission rather than as a request for the feature, so
        the call succeeds and returns no logprobs.  The official lane gets a
        larger budget because its model bills reasoning tokens against
        ``max_completion_tokens`` and would otherwise return empty content.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             https://developers.openai.com/api/docs/guides/reasoning
             stdapi/types/openai_chat_completions.py:CompletionCreateParams._unsupported
        """
        response = openai_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Say OK."}],
            logprobs=False,
            max_completion_tokens=_OFFICIAL_TOKEN_BUDGET if use_official_api else 16,
        )
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content
        assert response.choices[0].logprobs is None

    @pytest.mark.gateway("Custom tools unsupported only on this backend")
    def test_custom_tools_in_tools_unsupported(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """A ``custom`` tool definition in ``tools`` is rejected with a 400.

        Bedrock ``toolSpec`` only models JSON-schema function tools, so free-form
        custom tools cannot be translated.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#custom-tools
             stdapi/types/openai_chat_completions.py:CompletionCreateParams._validate_no_custom_tools
        """
        tools = [
            {
                "type": "custom",
                "custom": {
                    "name": "my_custom",
                    "description": "desc",
                    "format": {"type": "text"},
                },
            }
        ]
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Test"}],
                tools=tools,  # type: ignore[arg-type]
            )
        assert exc_info.value.status_code == 400
        body = exc_info.value.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        assert "custom" in body["message"].lower()

    @pytest.mark.gateway("Custom tool_choice unsupported only on this backend")
    def test_tool_choice_custom_unsupported(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """A ``custom`` ``tool_choice`` is rejected with a 400 even without custom tools.

        The check is on the choice itself, so naming a custom tool fails before the
        request is compared against the declared ``tools``.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#custom-tools
             stdapi/types/openai_chat_completions.py:CompletionCreateParams._validate_no_custom_tools
        """
        tool_choice = {"type": "custom", "custom": {"name": "my_custom"}}
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(  # type: ignore[call-overload]
                model=chat_model,
                messages=[{"role": "user", "content": "Hi"}],
                tool_choice=tool_choice,
            )
        assert exc_info.value.status_code == 400
        body = exc_info.value.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        assert "custom" in body["message"].lower()

    def test_service_tier(
        self, openai_client: OpenAI, chat_model: str, use_official_api: bool
    ) -> None:
        """A requested ``service_tier`` is echoed, while the Bedrock headers are not.

        Only ``priority``, ``flex`` and the Bedrock-only ``reserved`` map to a real
        Bedrock tier; every other requested tier resolves to an effective
        ``default``, which is what the response reports. The
        ``X-Amzn-Bedrock-*`` headers configure Bedrock directly and deliberately
        do not populate the OpenAI ``service_tier`` field.  The official lane gets
        a larger budget because its model bills reasoning tokens against
        ``max_completion_tokens`` and would otherwise return empty content.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             https://developers.openai.com/api/docs/guides/reasoning
             https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
             stdapi/models/chat/_adapters/_openai_common.py:map_service_tier
        """
        response = openai_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Say hi again"}],
            service_tier="default",
            max_completion_tokens=_OFFICIAL_TOKEN_BUDGET if use_official_api else 32,
        )
        assert getattr(response, "service_tier", None) == "default"
        assert response.choices[0].message.content

        # Test Bedrock headers
        if not use_official_api:
            response = openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Say hi again"}],
                max_completion_tokens=32,
                extra_headers={
                    "X-Amzn-Bedrock-PerformanceConfig-Latency": "standard",
                    "X-Amzn-Bedrock-Service-Tier": "default",
                },
            )
            assert not getattr(response, "service_tier", None)
            assert response.choices[0].message.content

    def test_reasoning_effort_parameter(
        self, openai_client: OpenAI, chat_reasoning_model: str
    ) -> None:
        """``reasoning_effort="minimal"`` is accepted and answered by a reasoning model.

        Claude has no ``minimal`` effort level, so the gateway downgrades it to
        ``low`` instead of forwarding an unknown value.

        Ref: https://developers.openai.com/api/docs/guides/reasoning#reasoning-effort
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel.REASONING_OVERRIDE
        """
        resp = openai_client.chat.completions.create(
            model=chat_reasoning_model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="minimal",
        )
        assert len(resp.choices) >= 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"
        # Either assistant text or a tool call depending on model/tooling
        assert isinstance(msg.content, str) or msg.tool_calls is not None
        assert resp.choices[0].finish_reason in ("stop", "length", "tool_calls")
        assert resp.usage is not None
        assert resp.usage.completion_tokens > 0

    @pytest.mark.gateway(
        "Qwen thinking response parameter is not supported on the official API"
    )
    def test_qwen_thinking_effort_parameter(
        self, openai_client: OpenAI, chat_reasoning_model: str
    ) -> None:
        """The Qwen ``enable_thinking``/``thinking_budget`` extras drive native reasoning.

        These are non-OpenAI fields accepted for Qwen compatibility: on a Claude
        model the explicit budget becomes a budget-based ``reasoning_config``
        instead of the adaptive effort form.

        Ref: stdapi/types/openai_chat_completions.py:CompletionCreateParams
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_reasoning
        """
        resp = openai_client.chat.completions.create(
            model=chat_reasoning_model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            extra_body={"enable_thinking": True, "thinking_budget": 1100},
        )

        assert len(resp.choices) >= 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"
        # Either assistant text or a tool call depending on model/tooling
        assert isinstance(msg.content, str) or msg.tool_calls is not None
        assert resp.choices[0].finish_reason in ("stop", "length", "tool_calls")
        assert resp.usage is not None
        assert resp.usage.completion_tokens > 0

    @pytest.mark.gateway(
        "Qwen thinking response parameter is not supported on the official API"
    )
    def test_unsupported_thinking_param_combinations(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """Contradictory thinking parameters are each rejected with a naming 400.

        ``thinking_budget`` is mutually exclusive with ``reasoning_effort`` (two
        ways to size the same budget), requires ``enable_thinking=true``, and is
        refused by models that only accept a categorical effort level such as
        DeepSeek V3.

        Ref: stdapi/types/openai_chat_completions.py:CompletionCreateParams._validate_thinking_options
             stdapi/models/chat/deepseek_v3.py:ChatModel._req_configure_reasoning
        """
        # reasoning_effort + thinking_budget
        with pytest.raises(BadRequestError) as both_budgets:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Reply with OK."}],
                reasoning_effort="minimal",
                extra_body={"enable_thinking": True, "thinking_budget": 1100},
            )
        assert both_budgets.value.status_code == 400
        both_budgets_body = both_budgets.value.body
        assert isinstance(both_budgets_body, dict)
        assert both_budgets_body["type"] == "invalid_request_error"
        assert "reasoning_effort" in both_budgets_body["message"]
        assert "thinking_budget" in both_budgets_body["message"]

        # thinking_budget + enable_thinking=False
        with pytest.raises(BadRequestError) as thinking_disabled:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Reply with OK."}],
                extra_body={"enable_thinking": False, "thinking_budget": 1100},
            )
        assert thinking_disabled.value.status_code == 400
        thinking_disabled_body = thinking_disabled.value.body
        assert isinstance(thinking_disabled_body, dict)
        assert thinking_disabled_body["type"] == "invalid_request_error"
        assert "enable_thinking" in thinking_disabled_body["message"]

        # thinking_budget unsupported by "deepseek.v3-v1:0"
        with pytest.raises(BadRequestError) as budget_unsupported:
            openai_client.chat.completions.create(
                model="deepseek.v3-v1:0",
                messages=[{"role": "user", "content": "Reply with OK."}],
                extra_body={"enable_thinking": True, "thinking_budget": 1100},
            )
        assert budget_unsupported.value.status_code == 400
        budget_unsupported_body = budget_unsupported.value.body
        assert isinstance(budget_unsupported_body, dict)
        assert budget_unsupported_body["type"] == "invalid_request_error"
        assert "thinking_budget" in budget_unsupported_body["message"]

    @pytest.mark.slow
    @pytest.mark.gateway(
        "Deepseek reasoning response parameter is not supported on the official API"
    )
    def test_deepseek_reasoning_response_parameter(self, openai_client: OpenAI) -> None:
        """DeepSeek reasoning maps to a categorical effort and round-trips ``reasoning_content``.

        DeepSeek takes a string ``reasoning_config`` rather than a token budget, and
        R1 always reasons: its thinking text is surfaced as the non-OpenAI
        ``reasoning_content`` field, which can be sent back on an assistant turn
        either as text or as content parts.

        Ref: stdapi/models/chat/deepseek_v3.py:ChatModel._req_configure_reasoning
             stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_output_text
        """
        # Test reasoning effort
        resp = openai_client.chat.completions.create(
            model="deepseek.v3-v1:0",
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="minimal",
        )
        msg = resp.choices[0].message
        assert msg.role == "assistant"
        assert resp.usage is not None
        assert resp.usage.completion_tokens > 0

        # Test reasoning content returned
        resp = openai_client.chat.completions.create(
            model="deepseek.r1-v1:0",
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_completion_tokens=512,
        )
        msg = resp.choices[0].message
        assert msg.role == "assistant"
        assert msg.reasoning_content  # type: ignore[attr-defined]
        assert isinstance(msg.reasoning_content, str)  # type: ignore[attr-defined]

        # Check sending reasoning content
        resp = openai_client.chat.completions.create(
            model="deepseek.v3-v1:0",
            messages=[
                {"role": "user", "content": "Reply with OK."},
                {  # type: ignore[list-item]
                    "role": "assistant",
                    "content": msg.content,
                    "reasoning_content": msg.reasoning_content,  # type: ignore[attr-defined]
                },
                {"role": "user", "content": "Reply with OK."},
                {  # type: ignore[list-item]
                    "role": "assistant",
                    "content": "ok",
                    "reasoning_content": [
                        {
                            "type": "text",
                            "text": "The user want only a simple OK response",
                        }
                    ],
                },
                {"role": "user", "content": "Reply with OK."},
            ],
        )
        msg = resp.choices[0].message
        assert msg.role == "assistant"
        assert resp.usage is not None
        assert resp.usage.prompt_tokens > 0, (
            "The replayed reasoning turns are billed as input"
        )

    def test_tool_choice_none_no_tool_calls(
        self, openai_client: OpenAI, chat_vision_model: str, use_official_api: bool
    ) -> None:
        """``tool_choice="none"`` with tools declared returns text and no tool calls.

        OpenAI ``none`` means "behave as if no tools were passed"; Bedrock has no
        ``none`` value in its ``toolChoice`` union, so the gateway implements it by
        omitting the tool configuration entirely.  The official lane gets a larger
        budget because its model bills reasoning tokens against
        ``max_completion_tokens`` and would otherwise return empty content.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
             https://developers.openai.com/api/docs/guides/reasoning
             stdapi/models/chat/_adapters/_openai_chat_completion.py:build_tool_config
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "echo",
                    "description": "Echo text",
                    "parameters": {
                        "type": "object",
                        "properties": {"text": {"type": "string"}},
                    },
                },
            }
        ]
        resp = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=chat_vision_model,
            messages=[{"role": "user", "content": "Hello"}],
            tools=tools,
            tool_choice="none",
            max_completion_tokens=_OFFICIAL_TOKEN_BUDGET if use_official_api else 64,
        )
        assert resp.choices[0].message.tool_calls is None
        assert isinstance(resp.choices[0].message.content, str)
        assert resp.choices[0].message.content, "The model must answer with text"
        assert resp.choices[0].finish_reason in ("stop", "length")

    def test_functions_function_call_none_no_function_call(
        self, openai_client: OpenAI, chat_legacy_model: str
    ) -> None:
        """With legacy ``functions`` and ``function_call="auto"`` a small talk turn stays text.

        ``auto`` is the documented default once functions are present, and the
        parameterless ``sum`` function is irrelevant to the prompt, so the model
        answers in plain text: neither ``function_call`` nor ``tool_calls`` is set.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             stdapi/models/chat/_adapters/_openai_chat_completion.py:_map_function_call
        """
        functions = [
            {
                "name": "sum",
                "description": "sum",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        resp = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=chat_legacy_model,
            messages=[{"role": "user", "content": "Hi"}],
            functions=functions,
            function_call="auto",
            max_completion_tokens=64,
        )
        msg = resp.choices[0].message
        assert msg.function_call is None
        assert msg.tool_calls is None or msg.tool_calls == []
        assert isinstance(msg.content, str)
        assert resp.choices[0].finish_reason in ("stop", "length")

    def test_stream_include_usage_final_chunk(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """``stream_options.include_usage`` adds a final usage-only chunk.

        Per the OpenAI contract that extra chunk carries the whole request's usage
        and an empty ``choices`` array, and it is emitted after the finish chunk and
        before ``[DONE]``.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             stdapi/models/chat/_adapters/_openai_chat_completion.py:format_stream
        """
        stream = openai_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Short reply please"}],
            stream=True,
            stream_options={"include_usage": True},
            max_completion_tokens=32,
        )
        last_chunk = None
        for item in stream:
            if isinstance(item, str) and item == "[DONE]":
                break
            last_chunk = item
        assert last_chunk is not None
        usage = getattr(last_chunk, "usage", None)
        assert usage is not None
        assert last_chunk.choices == [], "The usage chunk carries no choices"
        assert usage.completion_tokens > 0
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    def test_file_part_pdf(
        self,
        openai_client: OpenAI,
        chat_vision_model: str,
        sample_pdf_file_data_uri: str,
    ) -> None:
        """A PDF ``file`` part is sent as a Bedrock document block.

        ``file_data`` holding a PDF data URI is detected as ``application/pdf``,
        one of Bedrock's ``DocumentBlock`` formats, and the ``filename`` is
        sanitised into the block's ``name``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html
             stdapi/input_file.py:InputFile.to_bedrock_content_block
        """
        resp = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=[
                {  # type: ignore[unused-ignore,misc,list-item]
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Respond with OK only."},
                        {
                            "type": "file",
                            "file": {
                                "file_data": sample_pdf_file_data_uri,
                                "filename": "test.pdf",
                            },
                        },
                    ],
                }
            ],
            max_completion_tokens=16,
        )
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.role == "assistant"
        assert isinstance(resp.choices[0].message.content, str)
        assert resp.usage is not None
        assert resp.usage.prompt_tokens > 0, "The document is billed as input tokens"

    def test_developer_role_system_like(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """A ``developer`` message is routed to the Bedrock ``system`` field.

        Bedrock knows only ``user`` and ``assistant`` message roles; ``developer``
        (the successor of ``system``) is recognised as a system role and extracted
        alongside it, so it is not sent as a conversation turn.

        Ref: https://developers.openai.com/api/reference/resources/chat.md
             stdapi/models/chat/_adapters/_openai_chat_completion.py:_SYSTEM_ROLES
        """
        resp = openai_client.chat.completions.create(
            model=chat_model,
            messages=[
                {
                    "role": "developer",
                    "content": [{"type": "text", "text": "Respond with OK only."}],
                },
                {"role": "user", "content": "Say anything"},
            ],
            max_completion_tokens=16,
        )
        assert len(resp.choices) == 1
        assert resp.choices[0].message.role == "assistant"
        assert isinstance(resp.choices[0].message.content, str)
        assert resp.choices[0].finish_reason in ("stop", "length")
        assert resp.usage is not None
        assert resp.usage.prompt_tokens > 0

    def test_assistant_refusal_part_handling(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """An assistant turn made of a ``refusal`` content part is accepted as history.

        Bedrock has no refusal block, so the refusal text is replayed as a plain
        assistant text block instead of being dropped (which would break the
        user/assistant alternation Bedrock requires).

        Ref: https://developers.openai.com/api/reference/resources/chat.md
             stdapi/models/chat/_adapters/_openai_chat_completion.py:_extract_assistant_blocks
        """
        resp = openai_client.chat.completions.create(
            model=chat_model,
            messages=[
                {"role": "user", "content": "State a sensitive request"},
                {
                    "role": "assistant",
                    "content": [{"type": "refusal", "refusal": "I must refuse."}],
                },
                {"role": "user", "content": "Ok, proceed"},
            ],
            max_completion_tokens=64,
        )
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.role == "assistant"
        assert isinstance(resp.choices[0].message.content, str)
        assert resp.usage is not None
        assert resp.usage.prompt_tokens > 0, "The refusal turn is billed as input"

    def test_audio_output_mp3_format(
        self, openai_client: OpenAI, chat_audio_model: str
    ) -> None:
        """Audio output returns decodable base64 audio plus its transcript.

        Bedrock Converse never returns audio, so for a text-only model the gateway
        synthesizes the reply with Polly and reports the generating text as the
        ``transcript``.

        Ref: https://developers.openai.com/api/docs/guides/audio
             stdapi/models/chat/_adapters/_openai_chat_completion.py:_get_or_generate_audio
        """
        resp = openai_client.chat.completions.create(
            model=chat_audio_model,
            messages=[{"role": "user", "content": "Reply with OK"}],
            audio={"voice": "echo", "format": "mp3"},
            modalities=["text", "audio"],
            max_completion_tokens=16,
        )
        assert len(resp.choices) == 1
        audio = resp.choices[0].message.audio
        assert audio is not None
        assert isinstance(audio.data, str)
        assert audio.transcript
        # Verify base64 encoded
        try:
            decoded = base64.b64decode(audio.data)
        except (ValueError, TypeError) as error:
            pytest.fail(f"Audio data is not valid base64: {error}")
        assert decoded, "Audio payload decodes to no bytes"

    def test_audio_output_with_modalities_audio_only_unsupported(
        self, openai_client: OpenAI, chat_audio_model: str, use_official_api: bool
    ) -> None:
        """``modalities=["audio"]`` without ``text`` is rejected with a 400.

        The gateway derives audio from the generated text, so ``text`` can never be
        dropped: only ``["text"]`` and ``["text", "audio"]`` are accepted.

        Ref: https://developers.openai.com/api/docs/guides/audio
             stdapi/types/openai_chat_completions.py:CompletionCreateParams._validate_audio_modalities
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_audio_model,
                messages=[{"role": "user", "content": "Say hello"}],
                audio={"voice": "alloy", "format": "wav"},
                modalities=["audio"],
            )
        error = exc_info.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        assert "modalities" in body["message"].lower()
        if not use_official_api:
            assert body["code"] is None

    def test_audio_with_streaming_unsupported(
        self, openai_client: OpenAI, chat_audio_model: str, use_official_api: bool
    ) -> None:
        """Audio output combined with ``stream=True`` is rejected with a 400.

        The audio is synthesized from the complete text once generation is done, so
        it cannot be interleaved into the SSE stream.

        Ref: https://developers.openai.com/api/docs/guides/audio
             stdapi/types/openai_chat_completions.py:CompletionCreateParams._validate_audio_modalities
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_audio_model,
                messages=[{"role": "user", "content": "Reply with OK"}],
                audio={"voice": "echo", "format": "mp3"},
                modalities=["text", "audio"],
                stream=True,
            )
        error = exc_info.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        assert "stream" in body["message"].lower()
        if not use_official_api:
            assert body["code"] is None

    def test_audio_without_details_unsupported(
        self, openai_client: OpenAI, chat_audio_model: str, use_official_api: bool
    ) -> None:
        """Requesting the audio modality without the ``audio`` config is a 400.

        Voice and container format have no defaults, so the ``audio`` object is
        mandatory as soon as ``modalities`` includes ``audio``.

        Ref: https://developers.openai.com/api/docs/guides/audio
             stdapi/types/openai_chat_completions.py:CompletionCreateParams._validate_audio_modalities
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_audio_model,
                messages=[{"role": "user", "content": "Reply with OK"}],
                modalities=["text", "audio"],
            )
        error = exc_info.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        assert "audio" in body["message"].lower()
        if not use_official_api:
            assert body["code"] is None

    def test_audio_output_no_audio_with_tool_calls(
        self, openai_client: OpenAI, chat_audio_model: str
    ) -> None:
        """Audio is only produced for a turn that has text: a pure tool call gets none.

        The gateway synthesizes speech from the assistant text, so a choice whose
        content is empty (a tool-call-only turn) carries no ``audio`` object even
        though audio output was requested.

        Ref: https://developers.openai.com/api/docs/guides/audio
             stdapi/models/chat/_adapters/_openai_chat_completion.py:format_response
        """
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"],
                    },
                },
            }
        ]
        resp = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=chat_audio_model,
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=tools,
            audio={"voice": "alloy", "format": "mp3"},
            modalities=["text", "audio"],
            max_completion_tokens=128,
        )
        choice = resp.choices[0]
        assert bool(choice.message.audio) == bool(choice.message.content), (
            "Audio is generated exactly when the choice carries text"
        )
        if choice.message.tool_calls is not None and len(choice.message.tool_calls) > 0:
            # If model chose to call a tool, no audio should be generated
            # (or minimal audio from any brief text response)
            # This validates audio is not generated for tool call responses
            assert choice.finish_reason == "tool_calls"

    @pytest.mark.slow
    @pytest.mark.gateway("Application inference profiles are AWS Bedrock specific")
    async def test_inference_profile_as_model(
        self, openai_client: OpenAI, aws_region: str, aws_account_id: str
    ) -> None:
        """An application inference profile ARN is accepted as the ``model`` value.

        Bedrock accepts an inference-profile ARN wherever a model ID is expected,
        which is how callers attach cost-allocation tags; the gateway resolves the
        ARN to its copied-from model instead of rejecting the unknown ID.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-use.html
             stdapi/models/__init__.py:validate_model
        """
        inference_profile_arn = None
        async with get_session().create_client("bedrock") as bedrock:
            try:
                # Create the application inference profile
                inference_profile_arn = (
                    await bedrock.create_inference_profile(
                        inferenceProfileName=f"test-profile-{token_hex(8)}",
                        description="Test inference profile for automated testing",
                        modelSource={
                            "copyFrom": f"arn:aws:bedrock:{aws_region}:{aws_account_id}:inference-profile/global.anthropic.claude-haiku-4-5-20251001-v1:0"
                        },
                    )
                )["inferenceProfileArn"]

                max_wait = 30  # seconds
                wait_interval = 1  # second
                elapsed = 0
                while elapsed < max_wait:
                    if (
                        await bedrock.get_inference_profile(
                            inferenceProfileIdentifier=inference_profile_arn
                        )
                    )["status"] == "ACTIVE":
                        break
                    await sleep(wait_interval)
                    elapsed += wait_interval
                else:
                    pytest.fail(
                        f"Inference profile did not become active within {max_wait} seconds"
                    )

                # Test using the inference profile as model parameter
                response = openai_client.chat.completions.create(
                    model=inference_profile_arn,
                    messages=[{"role": "user", "content": "Say OK"}],
                    max_completion_tokens=10,
                )

                # Validate response structure
                assert len(response.choices) == 1
                assert response.choices[0].message.role == "assistant"
                assert isinstance(response.choices[0].message.content, str)
                assert len(response.choices[0].message.content) > 0
                assert response.usage is not None
                assert response.usage.completion_tokens > 0

            finally:
                if inference_profile_arn:
                    await bedrock.delete_inference_profile(
                        inferenceProfileIdentifier=inference_profile_arn
                    )

    @pytest.mark.gateway("Prompt routers are AWS Bedrock specific")
    async def test_prompt_router_as_model(
        self, openai_client: OpenAI, aws_region: str, aws_account_id: str
    ) -> None:
        """A default prompt-router ARN works as a ``model``, and a bogus ARN is a 400.

        Prompt routers pick the model per request, so the gateway forwards the ARN
        as the Converse ``modelId``. An ARN of a known Bedrock resource type that
        does not resolve is reported as an invalid-ARN 400 rather than as a 404 or a
        5xx.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/models/__init__.py:_validate_model_from_arn
        """
        # Test using the prompt router as model parameter
        response = openai_client.chat.completions.create(
            model=f"arn:aws:bedrock:{aws_region}:{aws_account_id}:default-prompt-router/amazon.nova:1",
            messages=[{"role": "user", "content": "Say OK"}],
            max_completion_tokens=10,
        )

        # Validate response structure
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
        assert isinstance(response.choices[0].message.content, str)
        assert len(response.choices[0].message.content) > 0
        assert response.usage is not None
        assert response.usage.completion_tokens > 0

        # Test bad ARN
        for bad_arn in (
            f"arn:aws:bedrock:{aws_region}:{aws_account_id}:prompt-router/not.exists",
            f"arn:aws:bedrock:{aws_region}:{aws_account_id}:foundation-model/not.exists",
        ):
            with pytest.raises(BadRequestError) as exc_info:
                openai_client.chat.completions.create(
                    model=bad_arn,
                    messages=[{"role": "user", "content": "Say OK"}],
                    max_completion_tokens=10,
                )
            assert exc_info.value.status_code == 400
            body = exc_info.value.body
            assert isinstance(body, dict)
            assert body["type"] == "invalid_request_error"
            assert "ARN does not match a valid" in body["message"]

    # --- Response metadata fields ---

    def test_response_id_format(self, envelope_completion: ChatCompletion) -> None:
        """The completion id is the ``chatcmpl-`` prefix plus a per-request identifier.

        An unstored completion is identified by ``chatcmpl-{request id}``; a stored
        one uses the session id instead. Either way the prefix is what clients
        match on.

        Ref: https://developers.openai.com/api/reference/resources/chat.md
             stdapi/routes/openai_chat_completions.py:create_chat_completion
        """
        response = envelope_completion
        assert response.id.startswith("chatcmpl-")
        assert len(response.id) > len("chatcmpl-"), "The id carries a request suffix"

    def test_response_object_and_created_fields(
        self, envelope_completion: ChatCompletion
    ) -> None:
        """``object`` is ``chat.completion`` and ``created`` is a Unix-seconds timestamp.

        ``created`` comes from the gateway's own request timestamp, so it must be in
        seconds — a millisecond value would be roughly a thousand times too large.
        The range is asserted instead of a delta against the local clock, which would
        make the test fail on a slow or clock-skewed runner.

        Ref: https://developers.openai.com/api/reference/resources/chat.md
             stdapi/routes/openai_chat_completions.py:create_chat_completion
        """
        response = envelope_completion
        assert response.object == "chat.completion"
        assert isinstance(response.created, int)
        assert 1_700_000_000 < response.created < 4_000_000_000, (
            "created must be a Unix timestamp in seconds"
        )

    def test_streaming_finish_reason_stop(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """A stream that ends naturally reports ``finish_reason="stop"``.

        Bedrock's ``end_turn`` stop reason has no OpenAI counterpart, so it falls
        through the mapping table to ``stop``.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
             stdapi/models/chat/_adapters/_openai_chat_completion.py:map_bedrock_stop_reason
        """
        response = openai_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Say OK."}],
            stream=True,
        )

        finish_reason = None
        for chunk in response:
            if isinstance(chunk, str) and chunk == "[DONE]":
                break
            if chunk.choices and chunk.choices[0].finish_reason is not None:
                finish_reason = chunk.choices[0].finish_reason

        assert finish_reason == "stop"

    def test_streaming_chunk_object_field(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """The first streamed chunk is a ``chat.completion.chunk`` announcing the role.

        Ref: https://developers.openai.com/api/reference/resources/chat.md
             stdapi/models/chat/_adapters/_openai_chat_completion.py:format_stream
        """
        response = openai_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Say OK."}],
            stream=True,
            max_completion_tokens=16,
        )

        for chunk in response:
            if isinstance(chunk, str) and chunk == "[DONE]":
                break
            assert chunk.object == "chat.completion.chunk"
            assert chunk.choices[0].delta.role == "assistant"
            assert chunk.id.startswith("chatcmpl-")
            break  # Only need to check first chunk

    def test_streaming_with_stop_sequences(
        self, openai_client: OpenAI, chat_legacy_model: str
    ) -> None:
        """A stream cut short by a stop sequence still reports ``finish_reason="stop"``.

        The 200-token budget is far above what the truncated answer needs, so a
        ``length`` finish would mean the stop sequence was not applied.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             stdapi/models/chat/_adapters/_openai_chat_completion.py:map_bedrock_stop_reason
        """
        response = openai_client.chat.completions.create(
            model=chat_legacy_model,
            messages=[
                {"role": "user", "content": "Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"}
            ],
            stop=["5"],
            stream=True,
            max_completion_tokens=200,
        )

        finish_reason = None
        for chunk in response:
            if isinstance(chunk, str) and chunk == "[DONE]":
                break
            if chunk.choices and chunk.choices[0].finish_reason is not None:
                finish_reason = chunk.choices[0].finish_reason

        assert finish_reason == "stop"

    def test_user_parameter_accepted(
        self, openai_client: OpenAI, chat_model: str, use_official_api: bool
    ) -> None:
        """The deprecated ``user`` field is still accepted as an end-user identifier.

        It is superseded by ``safety_identifier`` and ``prompt_cache_key`` but is
        kept for compatibility: the gateway uses it as the request log's user id.
        The official lane gets a larger budget because its model bills reasoning
        tokens against ``max_completion_tokens`` and would otherwise return empty
        content.

        Ref: https://developers.openai.com/api/docs/guides/safety-best-practices#implement-safety-identifiers
             https://developers.openai.com/api/docs/guides/reasoning
             stdapi/routes/openai_chat_completions.py:create_chat_completion
        """
        response = openai_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Say OK."}],
            user="test-user-123",
            max_completion_tokens=_OFFICIAL_TOKEN_BUDGET if use_official_api else 16,
        )
        assert len(response.choices) >= 1
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content
        assert response.usage is not None
        assert response.usage.completion_tokens > 0

    def test_frequency_penalty_accepted(
        self, openai_client: OpenAI, chat_legacy_model: str, use_official_api: bool
    ) -> None:
        """``frequency_penalty=1.0`` is accepted on a model that supports it.

        The gateway forwards the value untouched, so support is model-specific:
        the test only runs against the official API, where the mapped model
        implements it.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             stdapi/aws_bedrock.py:set_inference_configuration
        """
        if not use_official_api:
            pytest.skip(
                "frequency_penalty is model-dependent and may not be supported"
                " on all Bedrock models"
            )
        response = openai_client.chat.completions.create(
            model=chat_legacy_model,
            messages=[{"role": "user", "content": "Say OK."}],
            frequency_penalty=1.0,
            max_completion_tokens=16,
        )
        assert len(response.choices) >= 1
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].finish_reason in ("stop", "length")
        assert response.usage is not None
        assert response.usage.completion_tokens > 0

    def test_presence_penalty_accepted(
        self, openai_client: OpenAI, chat_legacy_model: str, use_official_api: bool
    ) -> None:
        """``presence_penalty=1.0`` is accepted on a model that supports it.

        As with ``frequency_penalty`` the value is forwarded untouched, so the
        check runs only against the official API.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             stdapi/aws_bedrock.py:set_inference_configuration
        """
        if not use_official_api:
            pytest.skip(
                "presence_penalty is model-dependent and may not be supported"
                " on all Bedrock models"
            )
        response = openai_client.chat.completions.create(
            model=chat_legacy_model,
            messages=[{"role": "user", "content": "Say OK."}],
            presence_penalty=1.0,
            max_completion_tokens=16,
        )
        assert len(response.choices) >= 1
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].finish_reason in ("stop", "length")
        assert response.usage is not None
        assert response.usage.completion_tokens > 0

    def test_max_completion_tokens_limits_output(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """``max_completion_tokens`` caps the output and yields ``finish_reason="length"``.

        The budget becomes Bedrock's ``inferenceConfig.maxTokens``, and its
        ``max_tokens`` stop reason maps to OpenAI's ``length``.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             stdapi/models/chat/_adapters/_openai_chat_completion.py:_FINISH_REASONS
        """
        max_tokens = 100
        response = openai_client.chat.completions.create(
            model=chat_model,
            messages=[
                {
                    "role": "user",
                    "content": "Write a very long detailed essay about the universe.",
                }
            ],
            max_completion_tokens=max_tokens,
        )
        assert response.choices[0].finish_reason == "length"
        assert response.usage
        assert response.usage.completion_tokens <= max_tokens

    @pytest.mark.gateway(
        "Deprecated model fallback is not available on the official OpenAI API"
    )
    def test_deprecated_model_fallback(self, openai_client: OpenAI) -> None:
        """A deprecated model ID is transparently served by its replacement model.

        ``amazon.titan-text-lite-v1`` no longer exists in the Bedrock catalogue; the
        deprecation chain resolves it to its recommended replacement, and the
        response reports the model that actually ran. The legacy ``max_tokens``
        alias is used here, which the gateway still accepts.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html
             stdapi/models/__init__.py:_resolve_deprecated
        """
        response = openai_client.chat.completions.create(
            model="amazon.titan-text-lite-v1",
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=16,
        )
        assert len(response.choices) > 0
        assert response.choices[0].message.content
        assert response.choices[0].finish_reason in ("stop", "length")
        assert response.model == DEPRECATED_MODELS["amazon.titan-text-lite-v1"], (
            "The response must name the replacement model that served the request"
        )

    @pytest.mark.gateway(
        "Bedrock outputConfig is not available on the official OpenAI API"
    )
    def test_response_format_json_schema(
        self, openai_client: OpenAI, chat_reasoning_model: str
    ) -> None:
        """A multi-property ``json_schema`` response format is enforced by the model.

        The schema is passed to Bedrock as the Converse ``outputConfig``, so both
        required properties come back and the answer is machine-readable without
        any prose wrapper. Only Anthropic models accept ``outputConfig``.

        Ref: https://developers.openai.com/api/docs/guides/structured-outputs#supported-schemas
             stdapi/models/chat/_adapters/_openai_chat_completion.py:build_output_config
        """
        schema = {
            "type": "object",
            "properties": {
                "capital": {"type": "string"},
                "country": {"type": "string"},
            },
            "required": ["capital", "country"],
            "additionalProperties": False,
        }
        response = openai_client.chat.completions.create(
            model=chat_reasoning_model,
            messages=[
                {
                    "role": "user",
                    "content": "What is the capital of France? Reply using the provided schema.",
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "capital_info", "schema": schema},
            },
            max_completion_tokens=128,
        )
        content = response.choices[0].message.content
        assert content
        parsed = _json.loads(content)
        assert "capital" in parsed
        assert "country" in parsed
        assert isinstance(parsed["capital"], str)
        assert "paris" in parsed["capital"].lower()

    @pytest.mark.gateway("requestMetadata is a Bedrock-specific feature")
    def test_metadata_accepted(self, openai_client: OpenAI, chat_model: str) -> None:
        """``metadata`` is forwarded as Bedrock ``requestMetadata`` and echoed back.

        The pairs are attached to the Bedrock invocation for log filtering, and the
        route copies them onto the response after the adapter has run, so the
        client sees exactly what it sent.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/routes/openai_chat_completions.py:create_chat_completion
        """
        metadata = {"test-key": "test-value", "session": "unit-test"}
        response = openai_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Say OK."}],
            metadata=metadata,
            max_completion_tokens=16,
        )
        assert response.choices[0].message.content
        assert getattr(response, "metadata", None) == metadata


class TestChatCompletionsUsage:
    """Billed-usage logging for ``POST /v1/chat/completions``.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
         stdapi/usage.py:record_bedrock_usage
    """

    def test_chat_completion_usage_logged(
        self,
        test_client: TestClientType | None,
        chat_model: str,
        api_key: str,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A non-streaming completion logs one ``bedrock-runtime`` usage record.

        The record is keyed by the route path and the resolved model so billing can
        be attributed per endpoint, and it carries the real Bedrock token counts.

        Ref: stdapi/usage.py:record_bedrock_usage
             stdapi/usage.py:usage_log_entries
        """
        if test_client is None:
            pytest.skip("Requires local test server")
        capfd.readouterr()
        response = test_client.post(
            "/v1/chat/completions",
            json={
                "model": chat_model,
                "messages": [{"role": "user", "content": "Say OK."}],
                "max_completion_tokens": 16,
            },
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert response.status_code == 200
        entries = logged_usage_entries(
            capfd.readouterr().out,
            service="bedrock-runtime",
            operation="/v1/chat/completions",
            model=chat_model,
        )
        assert entries, "Expected a bedrock chat usage log entry"
        assert entries[0]["input_tokens"] > 0
        assert entries[0]["output_tokens"] > 0
        assert len(entries) == 1, "One request must be billed exactly once"
        assert (
            response.json()["usage"]["completion_tokens"]
            == (entries[0]["output_tokens"])
        ), "The logged usage must match the usage returned to the client"

    def test_chat_completion_streaming_usage_logged(
        self,
        test_client: TestClientType | None,
        chat_model: str,
        api_key: str,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A streaming completion logs usage exactly once, in the stream log event.

        A streamed request produces two log events (request and stream), so the
        usage must be attached to only one of them or the tokens would be billed
        twice; the route path has to be propagated into that second event.

        Ref: stdapi/monitoring.py:log_request_sse_stream_event
             stdapi/routes/openai_chat_completions.py:create_chat_completion
        """
        if test_client is None:
            pytest.skip("Requires local test server")
        capfd.readouterr()
        with test_client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": chat_model,
                "messages": [{"role": "user", "content": "Say OK."}],
                "max_completion_tokens": 16,
                "stream": True,
            },
            headers={"Authorization": f"Bearer {api_key}"},
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())
        assert "data: [DONE]" in body, "The SSE body must end with the DONE sentinel"
        entries = logged_usage_entries(
            capfd.readouterr().out,
            service="bedrock-runtime",
            operation="/v1/chat/completions",
            model=chat_model,
        )
        assert entries, "Expected a bedrock chat usage log entry for streaming"
        assert sum(entry["output_tokens"] for entry in entries) > 0
        assert len(entries) == 1, (
            f"Usage logged {len(entries)} times for one streaming request; "
            "expected exactly one (no double-counting)"
        )


class TestPromptTokensDetailsGate:
    """``prompt_tokens_details`` is set only when cache tokens are reported (unit).

    Bedrock reports ``cacheReadInputTokens``/``cacheWriteInputTokens`` outside
    ``inputTokens``, while OpenAI's ``prompt_tokens`` includes cached tokens, so
    the adapter has to add them back on both the buffered and the streaming path.

    The zero-cache case is owned by
    ``tests/test_openai_chat_completion_adapter.py:TestFormatResponseCacheWriteTokens``,
    which covers both cache buckets through the same ``format_response`` call.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
         https://developers.openai.com/api/reference/resources/chat.md
    """

    pytestmark = pytest.mark.local

    @pytest.fixture(autouse=True)
    def _adapter_call_context(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        """Bind the per-request state ``format_response`` reads outside a request.

        Ref: stdapi/models/chat/_adapters/_openai_chat_completion.py:_LEGACY_FUNCTION
        """
        monkeypatch.setattr(SETTINGS, "log_request_params", False)
        token = _LEGACY_FUNCTION.set(False)
        try:
            yield
        finally:
            _LEGACY_FUNCTION.reset(token)

    @staticmethod
    def _converse_response(cache_read_input_tokens: int) -> dict[str, Any]:
        """Build a minimal Converse response with the given cached-token count."""
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
            "stopReason": "end_turn",
            "usage": {
                "inputTokens": 10,
                "outputTokens": 5,
                "cacheReadInputTokens": cache_read_input_tokens,
            },
        }

    async def test_format_response_sets_details_when_cache_read_is_positive(
        self,
    ) -> None:
        """A positive ``cacheReadInputTokens`` is reported and folded into ``prompt_tokens``.

        Ref: stdapi/models/chat/_adapters/_openai_chat_completion.py:format_response
        """
        completion = await format_response(
            completion_id="chatcmpl-1",
            created=0,
            model_id="model",
            responses=[self._converse_response(3)],  # type: ignore[list-item]
            service_tier=None,
            audio_params=None,
            modalities=["text"],
        )
        assert completion.usage is not None
        assert completion.usage.prompt_tokens_details is not None
        assert completion.usage.prompt_tokens_details.cached_tokens == 3
        assert completion.usage.prompt_tokens == 13, (
            "OpenAI prompt_tokens includes the cached tokens Bedrock reports apart"
        )
        assert completion.usage.total_tokens == 18

    def test_extract_stream_usage_omits_details_when_cache_read_is_zero(self) -> None:
        """A zero ``cacheReadInputTokens`` in the metadata event omits the details object.

        Ref: stdapi/models/chat/_adapters/_openai_common.py:extract_stream_usage
        """
        event: dict[str, Any] = {
            "metadata": {
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 5,
                    "cacheReadInputTokens": 0,
                }
            }
        }
        usage = extract_stream_usage(event)  # type: ignore[arg-type]
        assert usage is not None
        assert usage.prompt_tokens_details is None
        assert usage.prompt_tokens == 10
        assert usage.total_tokens == 15

    def test_extract_stream_usage_sets_details_when_cache_read_is_positive(
        self,
    ) -> None:
        """A positive ``cacheReadInputTokens`` is reported and added to ``prompt_tokens``.

        Ref: stdapi/models/chat/_adapters/_openai_common.py:extract_stream_usage
        """
        event: dict[str, Any] = {
            "metadata": {
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 5,
                    "cacheReadInputTokens": 4,
                }
            }
        }
        usage = extract_stream_usage(event)  # type: ignore[arg-type]
        assert usage is not None
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 4
        assert usage.prompt_tokens == 14
        assert usage.total_tokens == 19

    def test_extract_stream_usage_reports_cache_write_tokens(self) -> None:
        """Cache writes are reported separately and also counted in ``prompt_tokens``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
             stdapi/models/chat/_adapters/_openai_common.py:extract_stream_usage
        """
        event: dict[str, Any] = {
            "metadata": {
                "usage": {
                    "inputTokens": 10,
                    "outputTokens": 5,
                    "cacheReadInputTokens": 0,
                    "cacheWriteInputTokens": 7,
                }
            }
        }
        usage = extract_stream_usage(event)  # type: ignore[arg-type]
        assert usage is not None
        assert usage.prompt_tokens == 17
        assert usage.prompt_tokens_details is not None
        assert usage.prompt_tokens_details.cached_tokens == 0
        assert usage.prompt_tokens_details.cache_write_tokens == 7


#: Stubbed Bedrock Converse stream for one text turn: a delta, the stop, then usage metadata.
_STUB_STREAM_EVENTS: list[dict[str, Any]] = [
    {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"text": "Hi"}}},
    {"messageStop": {"stopReason": "end_turn"}},
    {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
]


async def _stub_converse_stream(
    events: list[dict[str, Any]],
) -> AsyncIterator[dict[str, Any]]:
    """Yield the given Bedrock Converse stream event dicts one by one.

    Args:
        events: Converse stream event dicts to replay.

    Yields:
        Each event dict, in order.
    """
    for event in events:
        yield event


class TestFormatStreamSentinelAndUsage:
    """``format_stream`` chunk ordering, usage chunk and ``[DONE]`` sentinel (unit).

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
         stdapi/models/chat/_adapters/_openai_chat_completion.py:format_stream
    """

    pytestmark = pytest.mark.local

    @pytest.fixture(autouse=True)
    def _adapter_call_context(self, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        """Bind the per-request state ``format_stream`` reads outside a request.

        Ref: stdapi/models/chat/_adapters/_openai_chat_completion.py:_LEGACY_FUNCTION
        """
        monkeypatch.setattr(SETTINGS, "log_request_params", False)
        token = _LEGACY_FUNCTION.set(False)
        try:
            yield
        finally:
            _LEGACY_FUNCTION.reset(token)

    @staticmethod
    async def _run(*, include_usage: bool) -> list[Any]:
        """Drive format_stream over the stub events and collect every SSE event.

        Args:
            include_usage: Value forwarded to format_stream's include_usage.

        Returns:
            All SSE events yielded by format_stream, in order.
        """
        return [
            event
            async for event in format_stream(
                completion_id="chatcmpl-1",
                created=0,
                model_id="model",
                stream=_stub_converse_stream(_STUB_STREAM_EVENTS),  # type: ignore[arg-type]
                service_tier=None,
                include_usage=include_usage,
            )
        ]

    async def test_stream_ends_with_done_sentinel(self) -> None:
        """The raw SSE body has exactly one ``data: [DONE]`` event, positioned last.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
        """
        events = await self._run(include_usage=False)
        raw = "".join(event.encode().decode() for event in events)
        assert raw.count("data: [DONE]") == 1
        assert raw.rstrip("\r\n").endswith("data: [DONE]")

    async def test_stream_usage_in_separate_final_chunk(self) -> None:
        """With ``include_usage``, usage arrives in its own empty-choices chunk after the finish chunk.

        The stub stream reports 10 input and 5 output tokens, so the trailing chunk
        must total 15; the finish chunk itself must stay usage-free.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
        """
        events = await self._run(include_usage=True)
        assert events[-1].data == "[DONE]"

        first_chunk = _json.loads(events[0].data)
        assert first_chunk["choices"][0]["delta"] == {"role": "assistant"}
        assert first_chunk["object"] == "chat.completion.chunk"

        usage_chunk = _json.loads(events[-2].data)
        assert usage_chunk["choices"] == []
        assert usage_chunk["usage"] == {
            "completion_tokens": 5,
            "prompt_tokens": 10,
            "total_tokens": 15,
        }

        finish_chunk = _json.loads(events[-3].data)
        assert finish_chunk["choices"][0]["finish_reason"] == "stop"
        assert "usage" not in finish_chunk

    async def test_no_chunk_carries_stream_obfuscation(self) -> None:
        """No emitted chunk carries an obfuscation field.

        ``stream_options.include_obfuscation`` is accepted for upstream
        compatibility but never read by ``format_stream``, so the padding upstream
        adds by default is simply absent here.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             stdapi/types/openai_chat_completions.py:ChatCompletionStreamOptionsParam
        """
        events = await self._run(include_usage=True)
        for event in events[:-1]:
            chunk = _json.loads(event.data)
            assert "obfuscation" not in chunk
            for choice in chunk["choices"]:
                assert "obfuscation" not in choice
                assert "obfuscation" not in choice["delta"]

    async def test_stream_without_include_usage_omits_usage(self) -> None:
        """Without ``include_usage`` no chunk carries usage and the stream still ends with ``[DONE]``.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
        """
        events = await self._run(include_usage=False)
        assert events[-1].data == "[DONE]"
        for event in events[:-1]:
            assert "usage" not in _json.loads(event.data)


class TestStopSequenceValidation:
    """Whitespace-only stop sequences are rejected before dispatch (unit).

    AWS Bedrock rejects a blank ``stopSequences`` entry with a raw
    ``ValidationException`` while upstream OpenAI accepts it; this backend
    surfaces a clean 400 instead. Validation happens before any model dispatch or
    AWS call, so the rejection test runs against an app instance without the
    AWS-touching lifespan.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         stdapi/types/openai_chat_completions.py:CompletionCreateParams._validate_stop_sequences
    """

    pytestmark = pytest.mark.local

    def test_whitespace_only_stop_sequence_is_rejected(
        self, app_client: TestClientType
    ) -> None:
        r"""``stop=["\n"]`` is rejected with a clean 400 naming the whitespace rule.

        The model is never resolved, so the rejection is independent of the model
        catalogue and of any AWS call.

        Ref: stdapi/types/openai_chat_completions.py:CompletionCreateParams._validate_stop_sequences
        """
        response = app_client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stop": ["\n"],
            },
        )
        assert response.status_code == 400, response.text
        error_body = response.json()
        assert error_body["error"]["type"] == "invalid_request_error"
        assert "whitespace" in error_body["error"]["message"].lower()
        assert error_body["error"].keys() >= {"message", "type", "param", "code"}, (
            "The OpenAI envelope always carries all four keys"
        )

    def test_non_blank_stop_sequence_is_still_accepted(self) -> None:
        """A non-blank ``stop`` sequence is still accepted and preserved verbatim.

        Ref: stdapi/types/openai_chat_completions.py:CompletionCreateParams._validate_stop_sequences
        """
        request = CompletionCreateParams.model_validate(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stop": ["5"],
            }
        )
        assert request.stop == ["5"]


class TestOpenAIResponseHeaders:
    """Every OpenAI-tagged route echoes the OpenAI-compatible response headers.

    ``openai-version`` is a fixed REST API version string, ``openai-processing-ms``
    is the gateway's own timing, and ``openai-organization`` is echoed back only
    when the client sent it.  They are attached by the response middleware, so a
    request rejected during validation carries them too.

    Ref: https://developers.openai.com/api/reference/overview
         stdapi/api_providers/openai.py:set_openai_headers
    """

    pytestmark = pytest.mark.local

    #: Request body rejected before any AWS call, so the headers can be read for free.
    _REJECTED_BODY: ClassVar[dict[str, Any]] = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hi"}],
        "stop": ["\n"],
    }

    def test_headers_are_attached_and_organization_is_echoed(
        self, app_client: TestClientType
    ) -> None:
        """``openai-version`` is ``2020-10-01`` and the sent organization comes back."""
        response = app_client.post(
            "/v1/chat/completions",
            json=self._REJECTED_BODY,
            headers={"OpenAI-Organization": "org-test"},
        )
        assert response.status_code == 400, response.text
        assert response.headers["openai-version"] == "2020-10-01"
        assert response.headers["openai-organization"] == "org-test"
        assert int(response.headers["openai-processing-ms"]) >= 0
        assert response.headers["x-request-id"], (
            "clients correlate a failed call by its request id"
        )

    def test_organization_header_is_omitted_when_not_sent(
        self, app_client: TestClientType
    ) -> None:
        """``openai-organization`` is absent rather than empty when unset."""
        response = app_client.post("/v1/chat/completions", json=self._REJECTED_BODY)
        assert response.status_code == 400, response.text
        assert "openai-organization" not in response.headers
        assert response.headers["openai-version"] == "2020-10-01"

    def test_legacy_completions_route_carries_the_same_headers(
        self, app_client: TestClientType
    ) -> None:
        """The legacy ``/v1/completions`` route shares the OpenAI header handler."""
        response = app_client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "hi", "stop": ["\n"]},
            headers={"OpenAI-Organization": "org-test"},
        )
        assert response.status_code == 400, response.text
        assert response.headers["openai-version"] == "2020-10-01"
        assert response.headers["openai-organization"] == "org-test"


class TestBedrockRequestFieldAliases:
    """Bedrock/vendor-shaped request keys populate the declared OpenAI fields.

    ``BaseModelRequestWithExtra`` accepts unknown keys, so a dropped alias would
    not error: the value would land in ``model_extra`` and be forwarded as an
    ``additionalModelRequestFields`` entry instead of an inference-config field,
    silently not applying the limit.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
         stdapi/types/openai_chat_completions.py:CompletionCreateParams
    """

    pytestmark = pytest.mark.local

    def test_bedrock_field_aliases_populate_the_declared_fields(self) -> None:
        """``maxTokens``/``topP``/``stopSequences`` land on the declared fields."""
        request = CompletionCreateParams.model_validate(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "maxTokens": 16,
                "topP": 0.5,
                "stopSequences": ["X"],
            }
        )
        assert request.max_tokens == 16
        assert request.top_p == 0.5
        assert request.stop == ["X"]
        assert request.model_extra == {}, (
            "an alias that stopped being declared would leak into model_extra"
        )

    def test_snake_case_stop_sequences_alias(self) -> None:
        """``stop_sequences`` is accepted as a third spelling of ``stop``."""
        request = CompletionCreateParams.model_validate(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stop_sequences": ["Y"],
            }
        )
        assert request.stop == ["Y"]
        assert request.model_extra == {}


class TestJsonObjectSystemInstruction:
    """``response_format={"type": "json_object"}`` appends a JSON-only system block.

    Bedrock's ``outputConfig`` has no schema for "any JSON object" (issue #96),
    so ``create_completion`` enforces the contract with a system-prompt
    instruction instead, appended after any explicit system prompt without
    altering it.

    Ref: stdapi/models/chat/_default.py:ChatModel.create_completion
         stdapi/models/chat/_adapters/_openai_common.py:enforce_json_object
    """

    pytestmark = pytest.mark.local

    @staticmethod
    async def _captured_system_blocks(
        monkeypatch: pytest.MonkeyPatch,
        request: CompletionCreateParams,
        request_log: dict[str, Any],
    ) -> list[Any] | None:
        """Run ``create_completion`` against a stub Converse call and return ``system``.

        Args:
            monkeypatch: Fixture used to stub ``ChatModel.converse``.
            request: Chat completion request to translate.
            request_log: Bound ``REQUEST_LOG`` context, read by ``format_response``.

        Returns:
            The ``system`` field of the captured Converse request body, or
            ``None`` if the request carried no system blocks.
        """
        del request_log
        captured: dict[str, Any] = {}

        async def fake_converse(
            _self: ChatModel, bedrock_request: ConverseRequestBaseTypeDef
        ) -> dict[str, Any]:
            captured.update(bedrock_request)
            return {
                "output": {"message": {"role": "assistant", "content": []}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            }

        monkeypatch.setattr(ChatModel, "converse", fake_converse)
        await ChatModel("amazon.nova-2-lite-v1:0").create_completion(
            request, "chatcmpl-1", 0
        )
        return captured.get("system")

    async def test_instruction_is_appended_after_the_user_system_prompt(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """An explicit system prompt is preserved, with the instruction appended."""
        request = CompletionCreateParams.model_validate(
            {
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": "You only speak in French."},
                    {"role": "user", "content": "Reply in json."},
                ],
                "response_format": {"type": "json_object"},
            }
        )
        system = await self._captured_system_blocks(monkeypatch, request, request_log)
        assert system == [
            {"text": "You only speak in French."},
            JSON_OBJECT_SYSTEM_INSTRUCTION,
        ]

    async def test_no_instruction_for_plain_text(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """The default (text) ``response_format`` sends no extra system block."""
        request = CompletionCreateParams.model_validate(
            {"model": "test-model", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert (
            await self._captured_system_blocks(monkeypatch, request, request_log)
            is None
        )

    async def test_no_instruction_for_json_schema(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """``json_schema`` output is already constrained, so no nudge is added."""
        request = CompletionCreateParams.model_validate(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Reply in json."}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": {"type": "object"}},
                },
            }
        )
        assert (
            await self._captured_system_blocks(monkeypatch, request, request_log)
            is None
        )


class TestIdentifierFieldLengthBounds:
    """``prompt_cache_key``/``safety_identifier``/``user`` are bounded to 1..255 chars.

    These values flow into Bedrock ``requestMetadata`` and into the request log, so
    the gateway bounds them itself; the ceiling is deliberately higher than
    upstream's 64-character ``safety_identifier`` limit.

    Ref: https://developers.openai.com/api/docs/guides/safety-best-practices#implement-safety-identifiers
         stdapi/types/openai_chat_completions.py:CompletionCreateParams
    """

    pytestmark = pytest.mark.local

    @staticmethod
    def _validate(field: str, value: str) -> CompletionCreateParams:
        """Validate a minimal request carrying *field* set to *value*.

        Args:
            field: Identifier field name.
            value: Value to assign to it.

        Returns:
            The validated request.
        """
        return CompletionCreateParams.model_validate(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                field: value,
            }
        )

    @pytest.mark.parametrize("field", ["prompt_cache_key", "safety_identifier", "user"])
    def test_empty_identifier_is_rejected(self, field: str) -> None:
        """An empty identifier fails validation instead of reaching Bedrock."""
        with pytest.raises(ValidationError) as exc_info:
            self._validate(field, "")
        assert exc_info.value.errors()[0]["type"] == "string_too_short"

    @pytest.mark.parametrize("field", ["prompt_cache_key", "safety_identifier", "user"])
    def test_over_long_identifier_is_rejected(self, field: str) -> None:
        """256 characters exceeds the bound, so the value never reaches metadata."""
        with pytest.raises(ValidationError) as exc_info:
            self._validate(field, "x" * 256)
        assert exc_info.value.errors()[0]["type"] == "string_too_long"

    @pytest.mark.parametrize("field", ["prompt_cache_key", "safety_identifier", "user"])
    def test_maximum_length_identifier_is_accepted(self, field: str) -> None:
        """255 characters is the largest accepted value."""
        request = self._validate(field, "x" * 255)
        assert getattr(request, field) == "x" * 255


class TestStreamObfuscationOption:
    """``stream_options.include_obfuscation`` is accepted for upstream compatibility.

    Upstream includes obfuscation padding by default and lets clients turn it off;
    this gateway never emits it, but must keep accepting the flag so unmodified
    OpenAI SDK clients are not rejected.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
         stdapi/types/openai_chat_completions.py:ChatCompletionStreamOptionsParam
    """

    pytestmark = pytest.mark.local

    def test_include_obfuscation_is_accepted(self) -> None:
        """The flag validates alongside ``include_usage`` instead of 400-ing."""
        request = CompletionCreateParams.model_validate(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "stream_options": {"include_usage": True, "include_obfuscation": True},
            }
        )
        assert request.stream_options is not None
        assert request.stream_options.include_obfuscation is True
        assert request.stream_options.include_usage is True

    def test_include_obfuscation_defaults_to_false(self) -> None:
        """Omitting the flag leaves it false, matching the gateway's behaviour."""
        request = CompletionCreateParams.model_validate(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        assert request.stream_options is not None
        assert request.stream_options.include_obfuscation is False
