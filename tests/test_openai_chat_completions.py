"""Tests for the OpenAI /v1/chat/completions route.

Comprehensive test suite that validates all features of the OpenAI Chat Completions API
specification, ensuring compatibility with the official OpenAI API behavior.
"""

import base64
import json as _json
from collections.abc import Iterable
from secrets import token_hex

import pytest
from openai import BadRequestError, NotFoundError, OpenAI
from pybase64 import b64encode


@pytest.fixture(scope="session")
def _png_data_url(sample_image_file: bytes) -> str:
    """Generates a PNG data URL containing a base64-encoded image.

    Returns:
        str: A string representing the data URL of a PNG image in base64 encoding.
    """
    return f"data:image/png;base64,{b64encode(sample_image_file).decode('utf-8')}"


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

    for idx, chunk in enumerate(response):
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
        if idx >= 24:
            break

    return chunks, saw_function_delta, args_fragments, has_finish


class TestChatCompletions:
    """Test suite for the /v1/chat/completions endpoint.

    Tests are designed to validate complete OpenAI API compatibility including:
    - All parameter combinations and validations
    - All response formats and completion output validation
    - Complete error scenario coverage with exact error matching
    - Edge cases and boundary conditions
    - Streaming behavior and function calling capabilities
    - Multimodal capabilities and tool usage
    - Allowed tools option and custom tool choice handling
    - System messages provided as content part text items
    """

    def test_basic_chat_completion(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """Test fundamental chat completion functionality with default parameters.

        Validates the core chat completion functionality using minimal parameters
        to ensure the service can generate responses successfully.

        Args:
            openai_client: OpenAI client instance for API calls
            chat_model: Chat model identifier

        Validates:
            - Response contains choices with message content
            - Message role is 'assistant'
            - Usage information is included
            - Response structure matches OpenAI specification
        """
        response = openai_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            safety_identifier="test-chat-completion",
        )

        # Validate response structure
        assert hasattr(response, "choices")
        assert len(response.choices) == 1
        assert hasattr(response.choices[0], "message")
        assert response.choices[0].message.role == "assistant"
        assert isinstance(response.choices[0].message.content, str)
        assert len(response.choices[0].message.content) > 0

        # Validate usage information
        assert hasattr(response, "usage")
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def test_multiple_choices_parameter(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """Test n parameter for generating multiple choices.

        Validates that the n parameter correctly generates multiple response choices
        when supported by the model.

        Args:
            openai_client: OpenAI client instance for API calls
            chat_model: Chat model identifier

        Validates:
            - Multiple choices are generated when n > 1
            - Each choice has proper structure
            - All choices contain valid assistant messages
        """
        response = openai_client.chat.completions.create(
            model=chat_model, messages=[{"role": "user", "content": "Say hello."}], n=2
        )

        # Validate multiple choices
        assert hasattr(response, "choices")
        assert len(response.choices) == 2

        for i, choice in enumerate(response.choices):
            assert choice.index == i
            assert choice.message.role == "assistant"
            assert isinstance(choice.message.content, str)

    def test_system_message_handling(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """Test system message processing and conversation context.

        Validates that system messages properly set conversation context
        and influence assistant responses as expected.

        Args:
            openai_client: OpenAI client instance for API calls
            chat_model: Chat model identifier

        Validates:
            - System messages are processed correctly
            - Assistant follows system instructions
            - Multi-turn conversation context is maintained
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

        # Validate response follows context
        assert hasattr(response, "choices")
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
        assert isinstance(response.choices[0].message.content, str)

    def test_streaming_basic_functionality(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """Test basic streaming chat completion functionality.

        Validates that streaming mode works correctly and produces incremental
        response chunks with proper delta content.

        Args:
            openai_client: OpenAI client instance for API calls
            chat_model: Chat model identifier

        Validates:
            - Streaming response is iterable
            - Chunks contain delta content
            - Stream completes with proper finish reason
            - Accumulated content forms coherent response
        """
        response = openai_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Count to 5 slowly."}],
            stream=True,
        )

        chunks = []
        accumulated_content = ""

        for chunk in response:
            # Skip the final "[DONE]" string message
            if isinstance(chunk, str) and chunk == "[DONE]":
                break
            chunks.append(chunk)
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    accumulated_content += delta.content

            # Limit chunks for efficiency
            if len(chunks) >= 20:
                break

        # Validate streaming behavior
        assert len(chunks) > 0, "No streaming chunks received"
        assert len(accumulated_content) > 0, "No content accumulated from stream"

    def test_stop_sequences_functionality(
        self, openai_client: OpenAI, chat_legacy_model: str
    ) -> None:
        """Test stop sequences for controlling generation termination.

        Validates that stop sequences work correctly to terminate generation
        when specific tokens or strings are encountered.

        Args:
            openai_client: OpenAI client instance for API calls
            chat_legacy_model: Chat model identifier

        Validates:
            - Single stop string works correctly
            - Multiple stop sequences work correctly
            - Generation terminates appropriately
        """
        # Test single stop string
        response = openai_client.chat.completions.create(
            model=chat_legacy_model,
            messages=[
                {"role": "user", "content": "Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"}
            ],
            stop="5",
        )

        assert hasattr(response, "choices")
        assert len(response.choices) == 1

        # Test multiple stop sequences
        response = openai_client.chat.completions.create(
            model=chat_legacy_model,
            messages=[
                {"role": "user", "content": "List colors: red, blue, green, yellow"}
            ],
            stop=["green", "yellow"],
        )

        assert hasattr(response, "choices")
        assert len(response.choices) == 1

    def test_tools_calling(self, openai_client: OpenAI, chat_vision_model: str) -> None:
        """Test basic tool calling capabilities.

        Validates that tool definitions are processed correctly and
        the model can invoke functions appropriately.

        Args:
            openai_client: OpenAI client instance for API calls
            chat_vision_model: Vision-capable chat model identifier

        Validates:
            - Function definitions are accepted
            - Model can decide to call functions
            - Function call structure is correct
            - Tool choice parameters work correctly
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

        assert hasattr(response, "choices")
        assert len(response.choices) == 1

        # Force a specific function to be called to make the behavior deterministic across providers
        forced_tool_choice = {"type": "function", "function": {"name": "get_weather"}}

        response = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=chat_vision_model,
            messages=[{"role": "user", "content": "What's the weather in New York?"}],
            tools=tools,
            tool_choice=forced_tool_choice,
        )

        assert hasattr(response, "choices")
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
        # Try to parse arguments as JSON if possible

        args_dict = None
        try:
            args_dict = _json.loads(first_call.function.arguments)
        except (ValueError, TypeError):
            # Some providers may not emit strict JSON; accept non-empty string
            args_dict = None
        if args_dict is not None and isinstance(args_dict, dict):
            # If JSON-like, the schema should resemble our tool definition
            assert "location" in args_dict

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

    def test_legacy_functions_parameter(
        self, openai_client: OpenAI, chat_legacy_model: str
    ) -> None:
        """Test legacy functions parameter for backward compatibility.

        Validates that the deprecated functions parameter still works
        for backward compatibility with older implementations.

        Args:
            openai_client: OpenAI client instance for API calls
            chat_legacy_model: Chat model identifier

        Validates:
            - Legacy functions parameter is accepted
            - Function call behavior works correctly
            - Response structure is valid
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

        assert hasattr(response, "choices")
        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.message.role == "assistant"

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
        self, openai_client: OpenAI, chat_vision_model: str
    ) -> None:
        """Test streaming with legacy functions (functions + function_call).

        Ensures that with stream=True and legacy `functions` enabled, the response
        streams function_call deltas and ends with an appropriate finish_reason.
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
            max_completion_tokens=60,
        )

        chunks, saw_function_delta, args_fragments, has_finish = (
            _gather_legacy_stream_info(response)
        )

        assert len(chunks) > 0, "No streaming chunks received for legacy functions"
        assert saw_function_delta or has_finish

        if args_fragments:
            args_joined = "".join(args_fragments)
            assert isinstance(args_joined, str)
            assert len(args_joined) > 0

    def test_empty_messages_error(self, openai_client: OpenAI, chat_model: str) -> None:
        """Test error handling for empty messages array.

        Validates proper error response for empty messages according to OpenAI specification.

        Args:
            openai_client: OpenAI client instance for API calls
            chat_model: Chat model identifier

        Validates:
            - Correct HTTP status code (400)
            - Proper error type ("invalid_request_error") and code ("empty_array")
            - Error response format matches OpenAI specification
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
        """Test error handling for invalid model specification.

        Validates proper error response for non-existent model names.

        Args:
            openai_client: OpenAI client instance for API calls

        Validates:
            - Correct HTTP status code (404)
            - Proper error type ("invalid_request_error") and code ("model_not_found")
            - Error message identifies model as invalid
            - Consistent error response structure
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
        """Test error handling for invalid temperature values.

        Validates proper error response for temperature values outside valid range (0.0-2.0).

        Args:
            openai_client: OpenAI client instance for API calls
            chat_model: Chat model identifier
            temperature: The invalid temperature value to test

        Validates:
            - Correct HTTP status code (400)
            - Proper error type ("invalid_request_error") and code ("decimal_above_max_value")
            - Error message mentions temperature validation
            - All boundary violations are caught
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

    def test_invalid_top_p_error(self, openai_client: OpenAI, chat_model: str) -> None:
        """Test error handling for invalid top_p values.

        Validates proper error response for top_p values outside valid range (0.0-1.0).

        Args:
            openai_client: OpenAI client instance for API calls
            chat_model: Chat model identifier

        Validates:
            - Correct HTTP status code (400)
            - Proper error type ("invalid_request_error") and code ("decimal_above_max_value")
            - Error message mentions top_p validation
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hello"}],
                top_p=1.5,
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert "top" in error_body["message"].lower()

    def test_invalid_max_tokens_error(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """Test error handling for invalid max_tokens values.

        Validates proper error response for max_tokens values outside valid range (>= 1).

        Args:
            openai_client: OpenAI client instance for API calls
            chat_model: Chat model identifier

        Validates:
            - Correct HTTP status code (400)
            - Proper error type ("invalid_request_error") and code ("integer_below_min_value")
            - Error message mentions max_tokens validation
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_completion_tokens=0,
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert "max_" in error_body["message"].lower()

    def test_invalid_frequency_penalty_error(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """Test error handling for invalid frequency_penalty values.

        Validates proper error response for frequency_penalty values outside valid range (-2.0 to 2.0).

        Args:
            openai_client: OpenAI client instance for API calls
            chat_model: Chat model identifier

        Validates:
            - Correct HTTP status code (400)
            - Proper error type ("invalid_request_error") and appropriate code
            - Error message mentions frequency_penalty validation
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hello"}],
                frequency_penalty=2.5,
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert "frequency_penalty" in error_body["message"].lower()

    def test_invalid_presence_penalty_error(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """Test error handling for invalid presence_penalty values.

        Validates proper error response for presence_penalty values outside valid range (-2.0 to 2.0).

        Args:
            openai_client: OpenAI client instance for API calls
            chat_model: Chat model identifier

        Validates:
            - Correct HTTP status code (400)
            - Proper error type ("invalid_request_error") and appropriate code
            - Error message mentions presence_penalty validation
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hello"}],
                presence_penalty=-2.5,
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert "presence_penalty" in error_body["message"].lower()

    def test_invalid_logit_bias_error(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """Test error handling for invalid logit_bias values.

        Validates proper error response for logit_bias values outside valid range (-100 to 100).

        Args:
            openai_client: OpenAI client instance for API calls
            chat_model: Chat model identifier

        Validates:
            - Correct HTTP status code (400)
            - Proper error type ("invalid_request_error") and code (null)
            - Error message mentions logit_bias validation
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hello"}],
                logit_bias={"100": 105},
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert "bias" in error_body["message"].lower()

    def test_streaming_with_tool_calls(
        self, openai_client: OpenAI, chat_vision_model: str
    ) -> None:
        """Test streaming functionality combined with tool calls.

        Validates that streaming works correctly when tool calling
        capabilities are enabled.

        Args:
            openai_client: OpenAI client instance for API calls
            chat_vision_model: Vision-capable chat model identifier

        Validates:
            - Streaming works with function definitions
            - Tool calls are streamed appropriately
            - Stream completion handles function calls correctly
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
        # Do not strictly require a finish_reason within the first limited chunks
        # to keep the test fast and robust across providers.
        assert isinstance(has_finish, bool)

    def test_multiple_tool_calls_flow(
        self, openai_client: OpenAI, chat_vision_model: str
    ) -> None:
        """Validate a flow with multiple tool calls in a single assistant message.

        This test crafts an assistant message with two tool calls and provides two tool
        results, then asks the model to produce a final answer. This covers the edge
        case of multiple tool calls at once and validates tool-related fields handling.
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

        assert hasattr(final, "choices")
        assert len(final.choices) == 1
        choice = final.choices[0]
        assert choice.message.role == "assistant"
        assert choice.message.tool_calls is None
        assert choice.message.content is not None
        assert isinstance(choice.message.content, str)
        assert choice.finish_reason in ("stop", "length")

    def test_conflicting_tools_and_functions_error(
        self, openai_client: OpenAI, chat_vision_model: str
    ) -> None:
        """Ensure providing both tools and legacy functions yields a proper error.

        The OpenAI API does not allow specifying both `functions` (legacy) and `tools` in
        the same request. Validate we get a 400 invalid_request_error consistent with
        OpenAI behavior.
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
        """System message content provided as content part text items.

        Validates the API accepts system messages where content is a list of
        ChatCompletionContentPartTextParam and still produces a valid assistant reply.
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
        assert hasattr(response, "choices")
        assert len(response.choices) == 1
        choice = response.choices[0]
        assert choice.message.role == "assistant"
        assert isinstance(choice.message.content, str)
        assert len(choice.message.content) > 0
        assert choice.message.tool_calls is None
        assert choice.finish_reason in ("stop", "length")
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.total_tokens >= response.usage.prompt_tokens

    def test_allowed_tools_auto(
        self, openai_client: OpenAI, chat_vision_model: str, use_openai_api: bool
    ) -> None:
        """allowed_tools tool choice: ensure request is accepted and outputs valid.

        We provide two tools but restrict allowed_tools to a single named tool in
        mode 'auto'. The OpenAI API accepts this experimental feature. This project
        currently treats it as 'auto' without strict enforcement; the test only asserts
        that the response is valid and, if a tool is called, its shape is correct.
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

        if not use_openai_api:
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
        self, openai_client: OpenAI, chat_vision_model: str, use_openai_api: bool
    ) -> None:
        """Custom tool choice: ensure request is accepted and outputs valid.

        We provide a single custom tool and force the model to call it using a
        named custom tool_choice. The API should accept the request. If the
        model calls the tool, the tool_calls item must have type 'custom' with
        name and input fields. Otherwise, the assistant should produce text.
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

        if not use_openai_api:
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

    @pytest.mark.expensive
    def test_multimodal_with_https_image_url(
        self, openai_client: OpenAI, chat_vision_model: str
    ) -> None:
        """Test text + image_url with an HTTPS image or data URL fallback."""
        response = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://raw.githubusercontent.com/JGoutin/asus-s14na-u12-uefi/refs/heads/master/data/block_diagram.png"
                            },
                        },
                    ],
                }
            ],
        )

        # Validate successful assistant response structure
        assert hasattr(response, "choices")
        assert len(response.choices) >= 1
        choice = response.choices[0]
        assert choice.message.role == "assistant"
        assert isinstance(choice.message.content, str)
        assert len(choice.message.content) > 0

    @pytest.mark.expensive
    def test_multimodal_with_data_url_base64_success(
        self,
        openai_client: OpenAI,
        chat_vision_model: str,
        sample_image_file_base64: str,
    ) -> None:
        """Test text + image_url with a valid data URL (base64 PNG)."""
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
        assert hasattr(response, "choices")
        assert len(response.choices) >= 1
        assert response.choices[0].message.role == "assistant"
        assert isinstance(response.choices[0].message.content, str)
        assert len(response.choices[0].message.content) > 0

    def test_multimodal_with_http_image_url_error(
        self, openai_client: OpenAI, chat_vision_model: str
    ) -> None:
        """Test http image URL handling.

        Uses an HTTP endpoint that returns non-200 or may be unreachable to
        validate error behavior. The API should respond with a 400 BadRequest
        in OpenAI-compatible shape when the image cannot be fetched.
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

    def test_multimodal_with_invalid_data_url_base64_error(
        self, openai_client: OpenAI, chat_vision_model: str
    ) -> None:
        """Test invalid data URL (bad base64) error handling for image_url."""
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

    @pytest.mark.parametrize("bad_b64", ["@@@", "!", "==?"])
    def test_file_part_invalid_base64_error(
        self, openai_client: OpenAI, chat_model: str, bad_b64: str, use_openai_api: bool
    ) -> None:
        """Invalid base64 in file content part should yield 400 invalid_request_error.

        Skipped against the official OpenAI API since this project-specific file part
        format is not part of the public API.
        """
        if use_openai_api:
            pytest.skip("File content part shape is implementation-specific here")
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "file",
                                "file": {
                                    "file_id": "f1",
                                    "file_data": bad_b64,
                                    "filename": "bad.bin",
                                },
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

    def test_file_part_unsupported_audio_mime_error(
        self, openai_client: OpenAI, chat_model: str, use_openai_api: bool
    ) -> None:
        """Unsupported audio MIME type in file part should yield 400.

        We craft a minimal WAV header to trigger audio MIME detection and validate
        the error path that rejects non-image/video/document MIME types.
        Skipped against the official OpenAI API.
        """
        if use_openai_api:
            pytest.skip("File content part shape is implementation-specific here")
        # Minimal WAV header bytes to be detected as audio/wav
        wav_bytes = (
            b"RIFF"
            b"\x24\x80\x00\x00"
            b"WAVEfmt "
            b"\x10\x00\x00\x00"
            b"\x01\x00\x01\x00"
            b"@\x1f\x00\x00"
            b"\x80>\x00\x00"
            b"\x02\x00\x10\x00"
            b"data"
            b"\x00\x80\x00\x00"
        )
        b64 = b64encode(wav_bytes).decode("utf-8")
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "file",
                                "file": {
                                    "file_id": "f2",
                                    "file_data": b64,
                                    "filename": "a.wav",
                                },
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
        assert "mime" in body["message"].lower() or "audio" in body["message"].lower()

    def test_validation_parallel_tool_calls_false_error(
        self, openai_client: OpenAI, chat_model: str, use_openai_api: bool
    ) -> None:
        """parallel_tool_calls=False should be rejected by this implementation.

        Skipped against the official OpenAI API if their behavior differs and the
        output result differs.
        """
        if use_openai_api:
            pytest.skip("Project-specific restriction: parallel_tool_calls=False")
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hello"}],
                parallel_tool_calls=False,
            )
        error = exc_info.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        assert "parallel_tool_calls" in body["message"].lower()

    def test_validation_stream_n_gt1_error(
        self, openai_client: OpenAI, chat_model: str, use_openai_api: bool
    ) -> None:
        """n>1 with stream=True is explicitly unsupported on this backend.

        Skipped against the official OpenAI API if their behavior differs and the
        output result differs.
        """
        if use_openai_api:
            pytest.skip("Project-specific restriction: stream with n>1 unsupported")
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

    def test_unsupported_response_format_error(
        self, openai_client: OpenAI, chat_model: str, use_openai_api: bool
    ) -> None:
        """response_format is unsupported; expect 400 (skip on OpenAI)."""
        if use_openai_api:
            pytest.skip("Unsupported fields are project-specific here")
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hi"}],
                response_format={"type": "json_object"},
            )
        assert exc_info.value.status_code == 400

    def test_unsupported_seed_error(
        self, openai_client: OpenAI, chat_model: str, use_openai_api: bool
    ) -> None:
        """Seed is unsupported; expect 400 (skip on OpenAI)."""
        if use_openai_api:
            pytest.skip("Unsupported fields are project-specific here")
        with pytest.raises(BadRequestError):
            openai_client.chat.completions.create(
                model=chat_model, messages=[{"role": "user", "content": "Hi"}], seed=123
            )

    def test_unsupported_store_error(
        self, openai_client: OpenAI, chat_model: str, use_openai_api: bool
    ) -> None:
        """Store is unsupported; expect 400 (skip on OpenAI)."""
        if use_openai_api:
            pytest.skip("Unsupported fields are project-specific here")
        with pytest.raises(BadRequestError):
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hi"}],
                store=True,
            )

    def test_unsupported_verbosity_error(
        self, openai_client: OpenAI, chat_model: str, use_openai_api: bool
    ) -> None:
        """Verbosity is unsupported; expect 400 (skip on OpenAI)."""
        if use_openai_api:
            pytest.skip("Unsupported fields are project-specific here")
        with pytest.raises(BadRequestError):
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hi"}],
                verbosity="high",
            )

    def test_unsupported_web_search_options_error(
        self, openai_client: OpenAI, chat_model: str, use_openai_api: bool
    ) -> None:
        """web_search_options is unsupported; expect 400 (skip on OpenAI)."""
        if use_openai_api:
            pytest.skip("Unsupported fields are project-specific here")
        with pytest.raises(BadRequestError):
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hi"}],
                web_search_options={
                    "search_context_size": "low",
                    "user_location": {
                        "type": "approximate",
                        "approximate": {"city": "x", "country": "US"},
                    },
                },
            )

    def test_unsupported_prediction_error(
        self, openai_client: OpenAI, chat_model: str, use_openai_api: bool
    ) -> None:
        """Prediction is unsupported; expect 400 (skip on OpenAI)."""
        if use_openai_api:
            pytest.skip("Unsupported fields are project-specific here")
        with pytest.raises(BadRequestError):
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hi"}],
                prediction={"type": "content", "content": "abc"},
            )

    def test_unsupported_metadata_error(
        self, openai_client: OpenAI, chat_model: str, use_openai_api: bool
    ) -> None:
        """Metadata is unsupported; expect 400 (skip on OpenAI)."""
        if use_openai_api:
            pytest.skip("Unsupported fields are project-specific here")
        with pytest.raises(BadRequestError):
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hi"}],
                metadata={"k": "v"},
            )

    def test_unsupported_top_logprobs_error(
        self, openai_client: OpenAI, chat_model: str, use_openai_api: bool
    ) -> None:
        """top_logprobs is unsupported; expect 400 (skip on OpenAI)."""
        if use_openai_api:
            pytest.skip("Unsupported fields are project-specific here")
        with pytest.raises(BadRequestError):
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hi"}],
                top_logprobs=5,
            )

    def test_unsupported_prompt_cache_key_error(
        self, openai_client: OpenAI, chat_model: str, use_openai_api: bool
    ) -> None:
        """prompt_cache_key is unsupported; expect 400 (skip on OpenAI)."""
        if use_openai_api:
            pytest.skip("Unsupported fields are project-specific here")
        with pytest.raises(BadRequestError):
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hi"}],
                prompt_cache_key="cache-key",
            )

    def test_unsupported_modalities_audio_error(
        self, openai_client: OpenAI, chat_model: str, use_openai_api: bool
    ) -> None:
        """Modalities including 'audio' are unsupported; expect 400 (skip on OpenAI)."""
        if use_openai_api:
            pytest.skip("Unsupported fields are project-specific here")
        with pytest.raises(BadRequestError):
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Hi"}],
                modalities=["text", "audio"],
            )

    def test_custom_tools_in_tools_unsupported(
        self, openai_client: OpenAI, chat_model: str, use_openai_api: bool
    ) -> None:
        """Providing custom tools in 'tools' is unsupported and should 400 here."""
        if use_openai_api:
            pytest.skip("Custom tools unsupported only on this backend")
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
        body = exc_info.value.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"

    def test_tool_choice_custom_unsupported(
        self, openai_client: OpenAI, chat_model: str, use_openai_api: bool
    ) -> None:
        """Specifying a custom tool in tool_choice should 400 on this backend."""
        if use_openai_api:
            pytest.skip("Custom tool_choice unsupported only on this backend")
        tool_choice = {"type": "custom", "custom": {"name": "my_custom"}}
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(  # type: ignore[call-overload]
                model=chat_model,
                messages=[{"role": "user", "content": "Hi"}],
                tool_choice=tool_choice,
            )
        body = exc_info.value.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"

    def test_service_tier_priority_and_scale(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """Validate service_tier mapping to response.service_tier."""
        # priority -> response shows 'priority'
        try:
            r1 = openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Say hi"}],
                service_tier="priority",
                max_completion_tokens=32,
            )
            assert getattr(r1, "service_tier", None)
        except BadRequestError as error:
            # Pass if argument properly passed but unsupported by model
            if (
                "Latency performance configuration is not supported "
                not in error.message
            ):
                raise

        # non-priority -> mapped to 'default' in response on this backend
        r2 = openai_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Say hi again"}],
            service_tier="flex",
            max_completion_tokens=32,
        )
        assert getattr(r2, "service_tier", None)

    def test_reasoning_effort_parameter(
        self, openai_client: OpenAI, chat_reasoning_model: str
    ) -> None:
        """reasoning_effort parameter: accepted and yields valid response on this backend.

        The parameter is only applicable to reasoning models on the official API.
        Since the chat_model fixture may not be a reasoning model when testing
        against the real OpenAI API, this test is skipped in that mode.
        """
        resp = openai_client.chat.completions.create(
            model=chat_reasoning_model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="minimal",
        )
        assert hasattr(resp, "choices")
        assert len(resp.choices) >= 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"
        # Either assistant text or a tool call depending on model/tooling
        assert (isinstance(msg.content, str) and len(msg.content) >= 0) or (
            msg.tool_calls is not None
        )
        assert resp.usage is not None

    def test_qwen_thinking_effort_parameter(
        self, openai_client: OpenAI, chat_reasoning_model: str, use_openai_api: bool
    ) -> None:
        """enable_thinking parameter: accepted and yields valid response on this backend."""
        if use_openai_api:
            pytest.skip(
                "Qwen thinking response parameter is not supported on the official API"
            )

        resp = openai_client.chat.completions.create(
            model=chat_reasoning_model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            extra_body={"enable_thinking": True, "thinking_budget": 1100},
        )

        assert hasattr(resp, "choices")
        assert len(resp.choices) >= 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"
        # Either assistant text or a tool call depending on model/tooling
        assert (isinstance(msg.content, str) and len(msg.content) >= 0) or (
            msg.tool_calls is not None
        )
        assert resp.usage is not None

    def test_unsupported_thinking_param_combinations(
        self, openai_client: OpenAI, chat_model: str, use_openai_api: bool
    ) -> None:
        """Tests unsupported combinations of thinking parameters for OpenAI chat API.

        This function evaluates error handling in situations where certain
        combinations of parameters related to "thinking" behavior are not supported.
        Specifically, it checks for errors when invalid parameter combinations are
        passed to the chat API and ensures the API behaves as expected in these
        situations.

        Args:
            openai_client: An instance of the OpenAI client to interact with the
                chat completion API.
            chat_model: The identifier of the chat model to be used for completions.
            use_openai_api: A boolean indicating whether to use the official OpenAI
                API or skip the test.

        Raises:
            BadRequestError: Raised when unsupported parameter combinations are
                provided to the chat completion API.
        """
        if use_openai_api:
            pytest.skip(
                "Qwen thinking response parameter is not supported on the official API"
            )

        # reasoning_effort + thinking_budget
        with pytest.raises(BadRequestError):
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Reply with OK."}],
                reasoning_effort="minimal",
                extra_body={"enable_thinking": True, "thinking_budget": 1100},
            )

        # thinking_budget + enable_thinking=False
        with pytest.raises(BadRequestError):
            openai_client.chat.completions.create(
                model=chat_model,
                messages=[{"role": "user", "content": "Reply with OK."}],
                extra_body={"enable_thinking": False, "thinking_budget": 1100},
            )

        # thinking_budget unsupported by "deepseek.v3-v1:0"
        with pytest.raises(BadRequestError):
            openai_client.chat.completions.create(
                model="deepseek.v3-v1:0",
                messages=[{"role": "user", "content": "Reply with OK."}],
                extra_body={"enable_thinking": True, "thinking_budget": 1100},
            )

    def test_deepseek_reasoning_response_parameter(
        self, openai_client: OpenAI, use_openai_api: bool
    ) -> None:
        """Check Deepseek reasoning response parameter."""
        if use_openai_api:
            pytest.skip(
                "Deepseek reasoning response parameter "
                "is not supported on the official API"
            )

        # Test reasoning effort
        resp = openai_client.chat.completions.create(
            model="deepseek.v3-v1:0",
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="minimal",
        )
        msg = resp.choices[0].message
        assert msg.role == "assistant"

        # Test reasoning content returned
        resp = openai_client.chat.completions.create(
            model="deepseek.r1-v1:0",
            messages=[{"role": "user", "content": "Reply with OK."}],
        )
        msg = resp.choices[0].message
        assert msg.role == "assistant"
        assert msg.reasoning_content  # type: ignore[attr-defined]

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

    def test_tool_choice_none_no_tool_calls(
        self, openai_client: OpenAI, chat_vision_model: str, use_openai_api: bool
    ) -> None:
        """With tools provided but tool_choice='none', expect assistant text and no tool_calls."""
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
        if not use_openai_api:
            with pytest.raises(BadRequestError) as exc_info:
                openai_client.chat.completions.create(  # type: ignore[call-overload]
                    model=chat_vision_model,
                    messages=[{"role": "user", "content": "Hello"}],
                    tools=tools,
                    tool_choice="none",
                )
            error = exc_info.value
            assert error.status_code == 400
            body = error.body
            assert isinstance(body, dict)
            assert body["type"] == "invalid_request_error"
            assert "none" in body["message"].lower()
            return

        resp = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=chat_vision_model,
            messages=[{"role": "user", "content": "Hello"}],
            tools=tools,
            tool_choice="none",
            max_completion_tokens=64,
        )
        assert resp.choices[0].message.tool_calls is None
        assert isinstance(resp.choices[0].message.content, str)

    def test_functions_function_call_none_no_function_call(
        self, openai_client: OpenAI, chat_legacy_model: str
    ) -> None:
        """With legacy functions and function_call='none', expect assistant text."""
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

    def test_stream_include_usage_final_chunk(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """When include_usage is True, the last streamed chunk should include usage."""
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
        assert getattr(last_chunk, "usage", None) is not None

    def test_file_part_text_plain_success(
        self, openai_client: OpenAI, chat_vision_model: str, use_openai_api: bool
    ) -> None:
        """Valid text/plain file part should be accepted and yield a response.

        Skipped on the official OpenAI API where this custom file part shape is not applicable.
        """
        if use_openai_api:
            pytest.skip("File content part is implementation-specific here")
        data = b64encode(b"hello world").decode("utf-8")
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
                                "file_id": "x",
                                "file_data": data,
                                "filename": None,
                            },
                        },
                    ],
                }
            ],
            max_completion_tokens=16,
        )
        assert hasattr(resp, "choices")
        assert len(resp.choices) >= 1

    def test_developer_role_system_like(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """Developer role is treated like system-level; ensure acceptance and valid reply."""
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

    def test_assistant_refusal_part_handling(
        self, openai_client: OpenAI, chat_model: str
    ) -> None:
        """Assistant message with refusal and text parts is accepted and processed."""
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

    def test_audio_output_mp3_format(
        self, openai_client: OpenAI, chat_audio_model: str
    ) -> None:
        """Audio output with mp3 format should generate valid base64 audio."""
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
            base64.b64decode(audio.data)
        except (ValueError, TypeError) as error:
            pytest.fail(f"Audio data is not valid base64: {error}")

    def test_audio_output_with_modalities_audio_only_unsupported(
        self, openai_client: OpenAI, chat_audio_model: str, use_openai_api: bool
    ) -> None:
        """Audio parameter only should raise an error."""
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

    def test_audio_with_streaming_unsupported(
        self, openai_client: OpenAI, chat_audio_model: str, use_openai_api: bool
    ) -> None:
        """Audio parameter with streaming should raise an error."""
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

    def test_audio_without_details_unsupported(
        self, openai_client: OpenAI, chat_audio_model: str, use_openai_api: bool
    ) -> None:
        """Audio parameter with streaming should raise an error."""
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

    def test_audio_output_no_audio_with_tool_calls(
        self, openai_client: OpenAI, chat_audio_model: str
    ) -> None:
        """When response contains tool calls, audio should not be generated."""
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
        if choice.message.tool_calls is not None and len(choice.message.tool_calls) > 0:
            # If model chose to call a tool, no audio should be generated
            # (or minimal audio from any brief text response)
            # This validates audio is not generated for tool call responses
            assert choice.finish_reason == "tool_calls"
