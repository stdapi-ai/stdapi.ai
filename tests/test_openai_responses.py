"""Tests for the OpenAI /v1/responses route.

Comprehensive test suite that validates all features of the OpenAI Responses API
specification, ensuring compatibility with the official OpenAI API behavior.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError, NotFoundError, OpenAI

from stdapi import usage
from stdapi.config import SETTINGS
from stdapi.usage import record_bedrock_usage

if TYPE_CHECKING:
    from starlette.testclient import TestClient as TestClientType


class TestResponses:
    """Test suite for the /v1/responses endpoint.

    Tests are designed to validate complete OpenAI Responses API compatibility
    including:
    - Core response generation and output structure validation
    - All parameter combinations and validations
    - Function tool calling (basic and advanced)
    - Built-in tool support (web search)
    - Multi-turn conversation via previous_response_id
    - Multimodal (image) inputs
    - Streaming and stream manager behavior
    - Response lifecycle (retrieve, input items listing)
    - Text format configuration (plain text, JSON object, JSON schema)
    - Error handling and input validation
    """

    # ---------------------------------------------------------------------------
    # Group 1: Core Functionality
    # ---------------------------------------------------------------------------

    def test_basic_response(self, openai_client: OpenAI, responses_model: str) -> None:
        """Test fundamental response creation with minimal parameters.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - Response contains a non-empty string id
            - Response output is a non-empty list
            - First output item is an assistant message with role 'assistant'
            - output_text convenience property returns non-empty string
        """
        response = openai_client.responses.create(
            model=responses_model, input="Say hello."
        )

        assert response.id
        assert isinstance(response.id, str)
        assert len(response.output) > 0
        # Reasoning models produce a reasoning item before the message
        msg = next((item for item in response.output if item.type == "message"), None)
        assert msg is not None, "Expected a message item in response output"
        assert msg.role == "assistant"
        assert isinstance(response.output_text, str)
        assert len(response.output_text) > 0

    def test_response_object_fields(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test all top-level Response object fields are present and correctly typed.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - All required Response fields are present after creation
            - Field types match the OpenAI Responses API specification
            - The model field is a non-empty string (may be a versioned alias)
        """
        response = openai_client.responses.create(model=responses_model, input="Hello.")

        assert isinstance(response.id, str)
        assert isinstance(response.created_at, float)
        # The API may return a versioned model name (e.g. 'gpt-5-nano-2025-08-07')
        assert isinstance(response.model, str)
        assert len(response.model) > 0
        assert response.object == "response"
        assert isinstance(response.output, list)
        assert response.status is not None
        assert response.usage is not None
        assert isinstance(response.tools, list)
        assert response.tool_choice is not None
        assert isinstance(response.parallel_tool_calls, bool)

    def test_instructions_system_prompt(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test the instructions parameter acts as a system-level prompt.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - instructions parameter is accepted alongside input
            - Response is generated successfully with instructions context
            - Output text is non-empty
        """
        response = openai_client.responses.create(
            model=responses_model,
            input="How should I respond?",
            instructions=(
                "You are a concise assistant. Always keep responses under 10 words."
            ),
        )

        assert len(response.output) > 0
        assert isinstance(response.output_text, str)
        assert len(response.output_text) > 0

    def test_structured_input_array(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test input provided as an array of message objects with roles.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - Input as a list of role/content dicts is accepted
            - Conversation history in the input array is processed correctly
            - The model uses prior assistant context to answer the follow-up
        """
        response = openai_client.responses.create(
            model=responses_model,
            input=[
                {"role": "user", "content": "My name is Alice."},
                {
                    "role": "assistant",
                    "content": "Hello, Alice! How can I help you today?",
                },
                {"role": "user", "content": "What is my name?"},
            ],
        )

        assert len(response.output) > 0
        assert isinstance(response.output_text, str)
        assert "alice" in response.output_text.lower(), (
            f"Expected name 'Alice' in response but got: {response.output_text!r}"
        )

    def test_output_text_property(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test the output_text convenience property aggregates text from output items.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - output_text property returns a non-empty string
            - output_text matches text manually extracted from output items
        """
        response = openai_client.responses.create(
            model=responses_model, input="Reply with exactly: hello world"
        )

        # Manually extract all output_text parts
        manual_text = ""
        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if part.type == "output_text":
                        manual_text += part.text

        assert response.output_text == manual_text
        assert len(response.output_text) > 0

    # ---------------------------------------------------------------------------
    # Group 2: Generation Parameters
    # ---------------------------------------------------------------------------

    def test_temperature_parameter(
        self, openai_client: OpenAI, chat_legacy_model: str
    ) -> None:
        """Test temperature parameter is accepted across valid range.

        Uses a non-reasoning model (gpt-4o-mini) since reasoning models reject
        the temperature parameter.

        Args:
            openai_client: OpenAI client instance for API calls
            chat_legacy_model: Non-reasoning model identifier (gpt-4o-mini)

        Validates:
            - temperature=0.0 (fully deterministic) is accepted
            - temperature=0.5 (mid-range) is accepted
            - temperature=1.0 is accepted
            - Responses are generated successfully for all values
        """
        for temperature in (0.0, 0.5, 1.0):
            response = openai_client.responses.create(
                model=chat_legacy_model, input="Say 'ok'.", temperature=temperature
            )
            assert len(response.output_text) > 0

    def test_top_p_parameter(
        self, openai_client: OpenAI, chat_legacy_model: str
    ) -> None:
        """Test top_p nucleus sampling parameter is accepted and response is generated.

        Uses a non-reasoning model (gpt-4o-mini) since reasoning models reject
        the top_p parameter.

        Args:
            openai_client: OpenAI client instance for API calls
            chat_legacy_model: Non-reasoning model identifier (gpt-4o-mini)

        Validates:
            - top_p=0.5 is accepted
            - top_p=1.0 (maximum / no filtering) is accepted
            - Response is non-empty for both values
        """
        for top_p in (0.5, 1.0):
            response = openai_client.responses.create(
                model=chat_legacy_model, input="Say 'hello'.", top_p=top_p
            )
            assert len(response.output_text) > 0

    def test_max_output_tokens_limits_output(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test max_output_tokens restricts the number of output tokens generated.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - max_output_tokens is accepted
            - Response usage.output_tokens is within the limit, or status is 'incomplete'
        """
        response = openai_client.responses.create(
            model=responses_model,
            input=(
                "Write a very long essay about the complete history of the world, "
                "covering every century in detail."
            ),
            max_output_tokens=20,
        )

        assert response.usage is not None
        # Either the output stays within the token limit, or the response is truncated
        assert response.usage.output_tokens <= 20 or response.status == "incomplete"

    def test_metadata_parameter(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test metadata key-value pairs are stored and returned on the response.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - Metadata dict is accepted
            - Metadata values are returned unchanged on the response object
        """
        test_metadata = {"session_id": "test-abc-123", "test_type": "automated"}
        response = openai_client.responses.create(
            model=responses_model, input="Say 'ok'.", metadata=test_metadata
        )

        assert response.metadata is not None
        assert response.metadata.get("session_id") == "test-abc-123"
        assert response.metadata.get("test_type") == "automated"

    def test_user_parameter(self, openai_client: OpenAI, responses_model: str) -> None:
        """Test user parameter is accepted without error.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - user parameter is accepted
            - Response is generated successfully
        """
        response = openai_client.responses.create(
            model=responses_model, input="Say 'hi'.", user="test-user-identifier-123"
        )

        assert len(response.output_text) > 0

    def test_service_tier_parameter(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test service_tier parameter is accepted without error.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - service_tier='default' is accepted
            - Response is generated successfully
        """
        response = openai_client.responses.create(
            model=responses_model, input="Say 'hello'.", service_tier="default"
        )

        assert len(response.output_text) > 0

    # ---------------------------------------------------------------------------
    # Group 3: Response Metadata & Structure
    # ---------------------------------------------------------------------------

    def test_response_id_format(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test response ID has the correct 'resp-' prefix format.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - Response id starts with 'resp_' (official API) or 'resp-' (local)
        """
        response = openai_client.responses.create(model=responses_model, input="Hello.")

        assert response.id.startswith("resp"), (
            f"Response ID '{response.id}' should start with 'resp'"
        )

    def test_response_object_type_field(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test the object field identifies the response as a 'response' object.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - response.object == 'response'
        """
        response = openai_client.responses.create(model=responses_model, input="Hello.")

        assert response.object == "response"

    def test_usage_token_counts(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test usage token counts are accurate and internally consistent.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - usage.input_tokens > 0
            - usage.output_tokens > 0
            - usage.total_tokens == input_tokens + output_tokens
            - usage.input_tokens_details is present
            - usage.output_tokens_details is present
        """
        response = openai_client.responses.create(
            model=responses_model, input="Say exactly: hello world."
        )

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.input_tokens + response.usage.output_tokens
        )
        assert response.usage.input_tokens_details is not None
        assert response.usage.output_tokens_details is not None

    def test_response_status_completed(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test that a normal synchronous response has status 'completed'.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - response.status == 'completed' for successful non-streaming requests
        """
        response = openai_client.responses.create(
            model=responses_model, input="Say 'done'."
        )

        assert response.status == "completed"

    # ---------------------------------------------------------------------------
    # Group 4: Text Format Configuration
    # ---------------------------------------------------------------------------

    def test_text_format_text(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test explicit plain text format configuration produces text output.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - text.format type='text' is accepted
            - Response output is a non-empty string
        """
        response = openai_client.responses.create(
            model=responses_model,
            input="Describe a red apple briefly.",
            text={"format": {"type": "text"}},
        )

        assert isinstance(response.output_text, str)
        assert len(response.output_text) > 0

    def test_text_format_json_object(
        self, openai_client: OpenAI, responses_json_output_model: str
    ) -> None:
        """Test json_object format produces valid parseable JSON output.

        Uses a model known to support Bedrock ``outputConfig`` (structured
        output is enforced natively — no system-prompt injection fallback).

        Args:
            openai_client: OpenAI client instance for API calls
            responses_json_output_model: Responses model supporting Bedrock
                ``outputConfig``.

        Validates:
            - text.format type='json_object' produces parseable JSON
            - Parsed output is a valid dictionary
        """
        response = openai_client.responses.create(
            model=responses_json_output_model,
            input=(
                'Reply with a JSON object containing the key "result" '
                'and value "success".'
            ),
            text={"format": {"type": "json_object"}},
        )

        output_text = response.output_text
        assert isinstance(output_text, str)
        parsed = json.loads(output_text)
        assert isinstance(parsed, dict)

    def test_text_format_json_schema(
        self, openai_client: OpenAI, responses_json_output_model: str
    ) -> None:
        """Test json_schema format produces output that conforms to the given schema.

        Uses a model known to support Bedrock ``outputConfig`` (structured
        output is enforced natively — no system-prompt injection fallback).

        Args:
            openai_client: OpenAI client instance for API calls
            responses_json_output_model: Responses model supporting Bedrock
                ``outputConfig``.

        Validates:
            - text.format type='json_schema' is accepted with name and schema
            - Response output parses as JSON conforming to the specified schema
        """
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer", "confidence"],
            "additionalProperties": False,
        }
        response = openai_client.responses.create(
            model=responses_json_output_model,
            input="What is 2 + 2? Reply with an answer and confidence score.",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "MathAnswer",
                    "schema": schema,
                    "strict": True,
                }
            },
        )

        output_text = response.output_text
        parsed = json.loads(output_text)
        assert isinstance(parsed, dict)
        assert "answer" in parsed
        assert "confidence" in parsed
        assert isinstance(parsed["answer"], str)
        assert isinstance(parsed["confidence"], (int, float))

    # ---------------------------------------------------------------------------
    # Group 5: Function Tool Calling
    # ---------------------------------------------------------------------------

    def test_function_tool_call_basic(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test basic function tool calling in the Responses API.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - Function tool definition is accepted in responses API format
            - Model produces a function_tool_call output item
            - Tool call includes the correct function name
            - Tool call arguments are valid JSON
        """
        tools = [
            {
                "type": "function",
                "name": "get_current_weather",
                "description": "Get the current weather for a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country, e.g. 'London, UK'",
                        }
                    },
                    "required": ["location"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

        response = openai_client.responses.create(  # type: ignore[call-overload]
            model=responses_model,
            input="What is the weather in Paris, France right now?",
            tools=tools,
            tool_choice="required",
        )

        tool_calls = [item for item in response.output if item.type == "function_call"]
        assert len(tool_calls) >= 1, "Expected at least one function_call in output"

        tool_call = tool_calls[0]
        assert tool_call.name == "get_current_weather"
        assert tool_call.arguments is not None
        args = json.loads(tool_call.arguments)
        assert isinstance(args, dict)

    def test_tool_choice_required(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test tool_choice='required' forces the model to call at least one tool.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - tool_choice='required' causes at least one function_tool_call in output
        """
        tools = [
            {
                "type": "function",
                "name": "calculate_sum",
                "description": "Add two numbers together",
                "parameters": {
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

        response = openai_client.responses.create(  # type: ignore[call-overload]
            model=responses_model,
            input="What is 5 plus 3?",
            tools=tools,
            tool_choice="required",
        )

        tool_calls = [item for item in response.output if item.type == "function_call"]
        assert len(tool_calls) >= 1

    def test_tool_choice_none(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test tool_choice='none' prevents the model from calling any tools.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - tool_choice='none' produces no function_tool_call items in output
            - Response contains a text message instead of a tool call
        """
        tools = [
            {
                "type": "function",
                "name": "search_web",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

        response = openai_client.responses.create(  # type: ignore[call-overload]
            model=responses_model,
            input="Search for news about AI.",
            tools=tools,
            tool_choice="none",
        )

        tool_calls = [item for item in response.output if item.type == "function_call"]
        assert len(tool_calls) == 0, (
            "Expected no function tool calls when tool_choice='none'"
        )
        assert len(response.output_text) > 0

    def test_tool_choice_specific_function(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test tool_choice forcing a specific named function to be called.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - tool_choice with specific function name forces that exact function
            - The forced function name matches the tool_choice specification
        """
        tools = [
            {
                "type": "function",
                "name": "get_time",
                "description": "Get the current time for a timezone",
                "parameters": {
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}},
                    "required": ["timezone"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ]

        response = openai_client.responses.create(  # type: ignore[call-overload]
            model=responses_model,
            input="Tell me about London.",
            tools=tools,
            tool_choice={"type": "function", "name": "get_time"},
        )

        tool_calls = [item for item in response.output if item.type == "function_call"]
        assert len(tool_calls) >= 1
        assert tool_calls[0].name == "get_time"

    def test_parallel_tool_calls_parameter(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test parallel_tool_calls parameter is accepted and reflected in response.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - parallel_tool_calls=True is accepted and reflected in response
            - parallel_tool_calls=False is accepted and reflected in response
        """
        tools = [
            {
                "type": "function",
                "name": "lookup",
                "description": "Look up information",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

        for parallel in (True, False):
            response = openai_client.responses.create(
                model=responses_model,
                input="Hello.",
                tools=tools,  # type: ignore[arg-type]
                parallel_tool_calls=parallel,
            )
            assert response.parallel_tool_calls == parallel

    def test_multiple_function_tools(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test multiple function tools can be defined simultaneously.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - Multiple tools can be provided in the tools list
            - Model selects an appropriate tool from the available options
            - At least one tool call is produced when tool_choice='required'
        """
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_population",
                "description": "Get the population of a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
            {
                "type": "function",
                "name": "get_timezone",
                "description": "Get the timezone for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        ]

        response = openai_client.responses.create(  # type: ignore[call-overload]
            model=responses_model,
            input="What is the weather in Tokyo?",
            tools=tools,
            tool_choice="required",
        )

        tool_calls = [item for item in response.output if item.type == "function_call"]
        assert len(tool_calls) >= 1
        assert tool_calls[0].name in {"get_weather", "get_population", "get_timezone"}

    # ---------------------------------------------------------------------------
    # Group 6: Built-in Tools
    # ---------------------------------------------------------------------------

    @pytest.mark.expensive
    def test_web_search_tool(
        self,
        openai_client: OpenAI,
        responses_web_search_model: str,
        use_official_api: bool,
    ) -> None:
        """Test the built-in web search tool produces search-augmented output.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_web_search_model: Model with web search support (Nova 2 locally,
                gpt-5-nano on the official API)
            use_official_api: Whether we are testing against the official OpenAI API

        Validates:
            - web_search_preview tool type is accepted
            - Response status is ``"completed"``
            - Response output contains a non-empty text message
            - At least one ``web_search_call`` output item is present (both locally and
              on the official API).  Locally ``web_search_preview`` maps to the
              ``nova_grounding`` Bedrock system tool; the gateway synthesises a
              ``web_search_call`` item from the query and any ``citationsContent``
              sources returned by Bedrock.
            - No bare ``function_call`` items leak from the suppressed system tool
        """
        try:
            response = openai_client.responses.create(
                model=responses_web_search_model,
                input="What is today's top headline news story?",
                tools=[{"type": "web_search_preview"}],
            )
        except BadRequestError as exc:
            if "nova_grounding is not supported" in str(exc):
                pytest.xfail("nova_grounding unavailable in cross-region routing")
            raise

        assert response.status == "completed"
        assert len(response.output) > 0
        assert isinstance(response.output_text, str)
        assert len(response.output_text) > 0

        web_search_calls = [
            item for item in response.output if item.type == "web_search_call"
        ]
        assert len(web_search_calls) >= 1, (
            "Expected at least one web_search_call output item"
        )

        # No bare function_call items should leak from the nova_grounding system tool.
        function_calls = [
            item for item in response.output if item.type == "function_call"
        ]
        assert function_calls == [], (
            f"function_call items must not leak from nova_grounding: {function_calls}"
        )

    @pytest.mark.expensive
    def test_web_search_tool_streaming(
        self,
        openai_client: OpenAI,
        responses_web_search_model: str,
        use_official_api: bool,
    ) -> None:
        """Streaming web search emits web_search_call lifecycle events and text deltas.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_web_search_model: Model with web search support
            use_official_api: Whether we are testing against the official OpenAI API

        Validates:
            - At least one ``response.web_search_call.in_progress`` event
            - At least one ``response.web_search_call.completed`` event
            - At least one ``response.output_text.delta`` event
            - No ``response.function_call_arguments.delta`` events (no function_call leaks)
            - Stream ends with ``response.completed``
        """
        ws_in_progress = 0
        ws_completed = 0
        text_delta_count = 0
        func_call_delta_count = 0
        completed = False

        try:
            stream = openai_client.responses.create(
                model=responses_web_search_model,
                input="What is today's top headline news story?",
                tools=[{"type": "web_search_preview"}],
                stream=True,
            )
        except BadRequestError as exc:
            if "nova_grounding is not supported" in str(exc):
                pytest.xfail("nova_grounding unavailable in cross-region routing")
            raise
        for event in stream:
            match event.type:
                case "response.web_search_call.in_progress":
                    ws_in_progress += 1
                case "response.web_search_call.completed":
                    ws_completed += 1
                case "response.output_text.delta":
                    text_delta_count += 1
                case "response.function_call_arguments.delta":
                    func_call_delta_count += 1
                case "response.completed":
                    completed = True

        assert ws_in_progress >= 1, (
            "Expected response.web_search_call.in_progress event"
        )
        assert ws_completed >= 1, "Expected response.web_search_call.completed event"
        assert text_delta_count >= 1, "Expected at least one output_text.delta event"
        assert func_call_delta_count == 0, (
            f"function_call_arguments.delta must not leak: {func_call_delta_count} events"
        )
        assert completed, "Expected response.completed event"

    @pytest.mark.expensive
    def test_web_search_type_tool(
        self, openai_client: OpenAI, responses_web_search_model: str
    ) -> None:
        """web_search tool type returns a web_search_call item.

        ``{"type": "web_search"}`` is the current official tool format.  Both
        the official API and stdapi (via ``nova_grounding``) should return a
        ``web_search_call`` output item and a text message.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_web_search_model: Model with web search support

        Validates:
            - Response status is ``"completed"``
            - At least one ``web_search_call`` output item present
            - No bare ``function_call`` items leak
            - Non-empty output text
        """
        try:
            resp = openai_client.responses.create(
                model=responses_web_search_model,
                input="What is the current version of Python?",
                tools=[{"type": "web_search"}],
            )
        except BadRequestError as exc:
            if "nova_grounding is not supported" in str(exc):
                pytest.xfail("nova_grounding unavailable in cross-region routing")
            raise
        assert resp.status == "completed"
        assert resp.output_text, "Expected non-empty text response"
        web_search_calls = [
            item for item in resp.output if item.type == "web_search_call"
        ]
        assert len(web_search_calls) >= 1, (
            "Expected at least one web_search_call output item"
        )
        function_calls = [item for item in resp.output if item.type == "function_call"]
        assert function_calls == [], (
            f"function_call items must not leak: {function_calls}"
        )

    @pytest.mark.expensive
    def test_web_search_type_tool_streaming(
        self, openai_client: OpenAI, responses_web_search_model: str
    ) -> None:
        """Streaming web_search tool type emits web_search_call events and text deltas.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_web_search_model: Model with web search support

        Validates:
            - At least one ``response.web_search_call.in_progress`` event
            - At least one ``response.web_search_call.completed`` event
            - At least one ``response.output_text.delta`` event
            - No ``response.function_call_arguments.delta`` events
            - Stream ends with ``response.completed``
        """
        ws_in_progress = 0
        ws_completed = 0
        text_delta_count = 0
        func_call_delta_count = 0
        completed = False

        try:
            stream = openai_client.responses.create(
                model=responses_web_search_model,
                input="What is the latest Python version?",
                tools=[{"type": "web_search"}],
                stream=True,
            )
        except BadRequestError as exc:
            if "nova_grounding is not supported" in str(exc):
                pytest.xfail("nova_grounding unavailable in cross-region routing")
            raise
        for event in stream:
            match event.type:
                case "response.web_search_call.in_progress":
                    ws_in_progress += 1
                case "response.web_search_call.completed":
                    ws_completed += 1
                case "response.output_text.delta":
                    text_delta_count += 1
                case "response.function_call_arguments.delta":
                    func_call_delta_count += 1
                case "response.completed":
                    completed = True

        assert ws_in_progress >= 1, (
            "Expected response.web_search_call.in_progress event"
        )
        assert ws_completed >= 1, "Expected response.web_search_call.completed event"
        assert text_delta_count >= 1, "Expected at least one output_text.delta event"
        assert func_call_delta_count == 0, (
            f"function_call_arguments.delta must not leak: {func_call_delta_count} events"
        )
        assert completed, "Expected response.completed event"

    # ---------------------------------------------------------------------------
    # Group 7: Multi-turn Conversation
    # ---------------------------------------------------------------------------

    @pytest.mark.skip(reason="Response storage not implemented (store=false)")
    def test_previous_response_id(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test previous_response_id enables stateful multi-turn conversation.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - First response is created and stored with store=True
            - Second response using previous_response_id is created successfully
            - Second response has a different id than the first
            - Both responses have status 'completed'
        """
        first_response = openai_client.responses.create(
            model=responses_model,
            input="Remember this number: 42. Just say 'Noted.'",
            store=True,
        )
        assert first_response.id
        assert first_response.status == "completed"

        second_response = openai_client.responses.create(
            model=responses_model,
            input="What number did I ask you to remember?",
            previous_response_id=first_response.id,
            store=True,
        )

        assert second_response.id != first_response.id
        assert second_response.status == "completed"
        assert len(second_response.output_text) > 0

    @pytest.mark.skip(reason="Response storage not implemented (store=false)")
    def test_multi_turn_context_maintained(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test that context from a previous response is available in follow-up.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - Model can recall information established in a prior response
            - The second response correctly references prior conversation context
        """
        first = openai_client.responses.create(
            model=responses_model,
            input="My favorite color is blue. Just say you noted it.",
            instructions="You are a helpful assistant with memory.",
            store=True,
        )

        second = openai_client.responses.create(
            model=responses_model,
            input="What is my favorite color?",
            previous_response_id=first.id,
            store=True,
        )

        assert "blue" in second.output_text.lower(), (
            f"Expected 'blue' in response but got: {second.output_text!r}"
        )

    # ---------------------------------------------------------------------------
    # Group 8: Multimodal Input
    # ---------------------------------------------------------------------------

    def test_image_base64_input(
        self,
        openai_client: OpenAI,
        chat_vision_model: str,
        sample_image_file_base64: str,
    ) -> None:
        """Test base64-encoded image data URL can be provided as input.

        Args:
            openai_client: OpenAI client instance for API calls
            chat_vision_model: Vision-capable model identifier
            sample_image_file_base64: Base64-encoded PNG image data URL

        Validates:
            - Input with type='input_image' and base64 data URL is accepted
            - Model generates a text response about the image
        """
        response = openai_client.responses.create(
            model=chat_vision_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": sample_image_file_base64,
                            "detail": "low",
                        },
                        {
                            "type": "input_text",
                            "text": "Describe this image briefly in one sentence.",
                        },
                    ],
                }
            ],
        )

        assert len(response.output_text) > 0

    def test_image_url_input(
        self, openai_client: OpenAI, chat_vision_model: str
    ) -> None:
        """Test HTTPS image URL can be provided as input to a vision-capable model.

        Args:
            openai_client: OpenAI client instance for API calls
            chat_vision_model: Vision-capable model identifier

        Validates:
            - Input with type='input_image' and HTTPS URL is accepted
            - Model generates a description of the image
        """
        response = openai_client.responses.create(
            model=chat_vision_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": (
                                "https://raw.githubusercontent.com/JGoutin/asus-s14na-u12-uefi"
                                "/refs/heads/master/data/block_diagram.png"
                            ),
                            "detail": "low",
                        },
                        {
                            "type": "input_text",
                            "text": "What do you see in this image? One sentence.",
                        },
                    ],
                }
            ],
        )

        assert isinstance(response.output_text, str)
        assert len(response.output_text) > 0

    # ---------------------------------------------------------------------------
    # Group 9: Streaming
    # ---------------------------------------------------------------------------

    def test_streaming_basic(self, openai_client: OpenAI, responses_model: str) -> None:
        """Test basic streaming response returns events and produces text output.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - stream=True returns an iterable of ResponseStreamEvent objects
            - Events are received from the stream
            - Accumulated text from delta events is non-empty
        """
        stream = openai_client.responses.create(
            model=responses_model, input="Count to five.", stream=True
        )

        events = []
        accumulated_text = ""

        for event in stream:
            events.append(event)
            if event.type == "response.output_text.delta":
                accumulated_text += event.delta

        assert len(events) > 0, "No streaming events received"
        assert len(accumulated_text) > 0, "No text accumulated from stream"

    def test_streaming_text_delta_events(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test streaming produces output_text.delta and output_text.done events.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - response.output_text.delta events appear with string delta values
            - response.output_text.done event appears with final aggregated text
            - Text from done event matches accumulated delta text
        """
        stream = openai_client.responses.create(
            model=responses_model, input="Write a short two-word greeting.", stream=True
        )

        delta_events = []
        done_event = None
        accumulated_text = ""

        for event in stream:
            if event.type == "response.output_text.delta":
                delta_events.append(event)
                accumulated_text += event.delta
            elif event.type == "response.output_text.done":
                done_event = event

        assert len(delta_events) > 0, "Expected at least one output_text.delta event"
        assert done_event is not None, "Expected a response.output_text.done event"
        assert done_event.text == accumulated_text

    def test_streaming_lifecycle_events(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test streaming produces the expected lifecycle events.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - response.created event appears with a non-empty response id
            - response.completed event appears with status 'completed'
            - The completed response contains output
        """
        stream = openai_client.responses.create(
            model=responses_model, input="Say 'done'.", stream=True
        )

        created_event = None
        completed_event = None

        for event in stream:
            if event.type == "response.created":
                created_event = event
            elif event.type == "response.completed":
                completed_event = event

        assert created_event is not None, "Expected response.created event"
        assert created_event.response is not None
        assert created_event.response.id

        assert completed_event is not None, "Expected response.completed event"
        assert completed_event.response.status == "completed"
        assert len(completed_event.response.output) > 0

    def test_streaming_function_call_events(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test streaming produces function call argument events when a tool is called.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - response.function_call_arguments.done event appears for a forced tool call
            - Arguments in the done event are valid JSON
        """
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

        stream = openai_client.responses.create(  # type: ignore[call-overload]
            model=responses_model,
            input="What is the weather in Tokyo?",
            tools=tools,
            tool_choice="required",
            stream=True,
        )

        args_done_event = None

        for event in stream:
            if event.type == "response.function_call_arguments.done":
                args_done_event = event

        assert args_done_event is not None, (
            "Expected response.function_call_arguments.done event"
        )
        args = json.loads(args_done_event.arguments)
        assert isinstance(args, dict)

    def test_streaming_with_stream_manager(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test the high-level stream manager context manager interface.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - responses.stream() context manager is entered successfully
            - get_final_response() returns a complete Response object
            - Final response has expected fields and status 'completed'
        """
        with openai_client.responses.stream(
            model=responses_model, input="Write a haiku about coding."
        ) as stream:
            for _ in stream:
                pass
            final_response = stream.get_final_response()

        assert final_response is not None
        assert final_response.id
        assert final_response.status == "completed"
        assert len(final_response.output_text) > 0

    # ---------------------------------------------------------------------------
    # Group 10: Response Lifecycle
    # ---------------------------------------------------------------------------

    @pytest.mark.skip(reason="Response storage not implemented (store=false)")
    def test_retrieve_response(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test retrieving a previously created stored response by ID.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - A response created with store=True can be retrieved by id
            - Retrieved response id and model match the original
            - Retrieved response has status 'completed'
        """
        original = openai_client.responses.create(
            model=responses_model, input="Say 'hello for retrieval test'.", store=True
        )
        assert original.id

        retrieved = openai_client.responses.retrieve(original.id)

        assert retrieved.id == original.id
        assert retrieved.model == original.model
        assert retrieved.status == "completed"

    @pytest.mark.skip(reason="Response storage not implemented (store=false)")
    def test_list_input_items(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test listing input items for a stored response.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - input_items.list() is accepted for a stored response
            - Returns a page object with a non-empty data list
            - Items in the list have expected structure
        """
        response = openai_client.responses.create(
            model=responses_model, input="Say 'hello for input items test'.", store=True
        )

        items = openai_client.responses.input_items.list(response.id)

        assert items is not None
        assert hasattr(items, "data")
        assert len(items.data) > 0

        first_item = items.data[0]
        # Input items have a role (message) or type attribute
        assert hasattr(first_item, "role") or hasattr(first_item, "type")

    def test_include_logprobs(
        self, openai_client: OpenAI, chat_legacy_model: str, use_official_api: bool
    ) -> None:
        """Test including log probabilities in the response output.

        Uses a non-reasoning model (gpt-4o-mini) since reasoning models do not
        support logprobs.

        Args:
            openai_client: OpenAI client instance for API calls
            chat_legacy_model: Non-reasoning model identifier (gpt-4o-mini)
            use_official_api: Whether tests run against the official OpenAI API

        Validates:
            - include=['message.output_text.logprobs'] is accepted
            - top_logprobs parameter is accepted alongside include
            - Output text content parts contain populated logprobs data
        """
        if not use_official_api:
            pytest.skip("Logprobs are not supported by Bedrock models")
        response = openai_client.responses.create(
            model=chat_legacy_model,
            input="Say 'yes'.",
            include=["message.output_text.logprobs"],
            top_logprobs=3,
        )

        msg = next((item for item in response.output if item.type == "message"), None)
        assert msg is not None, "Expected a message item in output"

        text_part = next(
            (part for part in msg.content if part.type == "output_text"), None
        )
        assert text_part is not None, "Expected an output_text content part"
        assert text_part.logprobs is not None, "Expected logprobs to be populated"
        assert len(text_part.logprobs) > 0

    # ---------------------------------------------------------------------------
    # Group 11: Advanced Features
    # ---------------------------------------------------------------------------

    def test_developer_role_input(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test developer role in input array acts as a system-level instruction.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - Input array containing a 'developer' role message is accepted
            - The developer instruction influences the model response
        """
        response = openai_client.responses.create(
            model=responses_model,
            input=[
                {
                    "role": "developer",
                    "content": "You are a helpful assistant. Always end replies with 'OK'.",
                },
                {"role": "user", "content": "Say hi."},
            ],
        )

        assert len(response.output) > 0
        assert len(response.output_text) > 0

    @pytest.mark.skip(reason="Response storage not implemented (store=false)")
    def test_function_tool_call_round_trip(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test a complete function tool call round-trip with result submission.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - First turn: model makes a function_call
            - Tool call includes a call_id
            - Second turn: function_call_output item with matching call_id is accepted
            - Final response contains a text message incorporating the tool result
        """
        tools = [
            {
                "type": "function",
                "name": "get_current_weather",
                "description": "Get the current weather for a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]

        # First turn: trigger a tool call
        first = openai_client.responses.create(  # type: ignore[call-overload]
            model=responses_model,
            input="What is the weather in Madrid?",
            tools=tools,
            tool_choice="required",
            store=True,
        )
        tool_calls = [item for item in first.output if item.type == "function_call"]
        assert len(tool_calls) >= 1
        tool_call = tool_calls[0]
        assert tool_call.call_id

        # Second turn: submit the function result
        second = openai_client.responses.create(
            model=responses_model,
            input=[
                {
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": '{"temperature": "22 degrees", "condition": "sunny"}',
                }
            ],
            tools=tools,  # type: ignore[arg-type]
            previous_response_id=first.id,
            store=True,
        )

        assert second.status == "completed"
        msg = next((item for item in second.output if item.type == "message"), None)
        assert msg is not None, "Expected a message item in second response"
        assert len(second.output_text) > 0

    @pytest.mark.skip(reason="Response storage not implemented (store=false)")
    def test_delete_response(self, openai_client: OpenAI, responses_model: str) -> None:
        """Test deleting a stored response removes it from the API.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - delete() is accepted for a stored response and returns None
            - The deleted response can no longer be retrieved (404 error)
        """
        response = openai_client.responses.create(
            model=responses_model, input="Say 'hello for delete test'.", store=True
        )
        assert response.id

        openai_client.responses.delete(response.id)

        with pytest.raises(NotFoundError):
            openai_client.responses.retrieve(response.id)

    def test_reasoning_output_item_and_tokens(
        self, openai_client: OpenAI, responses_model: str, use_official_api: bool
    ) -> None:
        """Test that reasoning models include a reasoning item and reasoning tokens.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier
            use_official_api: Whether tests are running against the official OpenAI API

        Validates:
            - Output contains a 'reasoning' type item (reasoning models only)
            - usage.output_tokens_details.reasoning_tokens > 0
        """
        if not use_official_api:
            pytest.skip("Only relevant for reasoning models (gpt-5-nano)")

        response = openai_client.responses.create(
            model=responses_model,
            input="What is 47 * 83?",
            reasoning={"effort": "medium"},
        )

        reasoning_items = [item for item in response.output if item.type == "reasoning"]
        assert len(reasoning_items) >= 1, "Expected a reasoning item in output"

        assert response.usage is not None
        assert response.usage.output_tokens_details is not None
        # With effort="medium" the model allocates a non-trivial reasoning budget
        assert response.usage.output_tokens_details.reasoning_tokens > 0

    # ---------------------------------------------------------------------------
    # Group 12: Error Handling & Validation
    # ---------------------------------------------------------------------------

    def test_invalid_model_error(self, openai_client: OpenAI) -> None:
        """Test that using a non-existent model raises an error.

        Args:
            openai_client: OpenAI client instance for API calls

        Validates:
            - NotFoundError (HTTP 404) or BadRequestError (HTTP 400) is raised
              for unknown model identifiers (official API returns 400; local
              implementation may return 404)
        """
        with pytest.raises((NotFoundError, BadRequestError)):
            openai_client.responses.create(
                model="definitely-not-a-valid-model-xyz-123", input="Hello."
            )

    def test_invalid_temperature_too_high(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test that temperature above maximum (2.0) raises a validation error.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - BadRequestError is raised for temperature=3.0 (above max of 2.0)
        """
        with pytest.raises(BadRequestError):
            openai_client.responses.create(
                model=responses_model, input="Hello.", temperature=3.0
            )

    def test_invalid_temperature_negative(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test that negative temperature raises a validation error.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - BadRequestError is raised for temperature=-1.0 (below minimum of 0)
        """
        with pytest.raises(BadRequestError):
            openai_client.responses.create(
                model=responses_model, input="Hello.", temperature=-1.0
            )

    def test_invalid_top_p_error(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test that top_p outside the valid range raises a validation error.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - BadRequestError is raised for top_p=2.0 (above maximum of 1.0)
        """
        with pytest.raises(BadRequestError):
            openai_client.responses.create(
                model=responses_model, input="Hello.", top_p=2.0
            )

    def test_invalid_max_output_tokens_error(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test that max_output_tokens below minimum raises a validation error.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - BadRequestError is raised for max_output_tokens=0 (minimum is 1)
        """
        with pytest.raises(BadRequestError):
            openai_client.responses.create(
                model=responses_model, input="Hello.", max_output_tokens=0
            )

    def test_invalid_top_logprobs_error(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Test that top_logprobs above the maximum raises a validation error.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_model: Responses model identifier

        Validates:
            - BadRequestError is raised for top_logprobs=21 (above maximum of 20)
        """
        with pytest.raises(BadRequestError):
            openai_client.responses.create(
                model=responses_model, input="Hello.", top_logprobs=21
            )

    def test_reasoning_parameter_accepted(
        self, openai_client: OpenAI, chat_reasoning_model: str
    ) -> None:
        """Test the reasoning parameter is accepted for reasoning-capable models.

        Args:
            openai_client: OpenAI client instance for API calls
            chat_reasoning_model: Reasoning-capable model identifier

        Validates:
            - reasoning={'effort': 'low'} parameter is accepted
            - Response is generated successfully with reasoning enabled
        """
        response = openai_client.responses.create(
            model=chat_reasoning_model,
            input="What is 15 * 8?",
            reasoning={"effort": "low"},
        )

        assert len(response.output) > 0
        assert isinstance(response.output_text, str)
        assert len(response.output_text) > 0


# ---------------------------------------------------------------------------
# Unsupported features — validated at the Pydantic layer before the API runs
# ---------------------------------------------------------------------------


class TestUnsupportedFeatures:
    """Unsupported parameters and tools are rejected before reaching the backend.

    All tests are skipped against the official OpenAI API, where these features
    are natively supported — the restrictions are gateway-specific.
    """

    def test_unsupported_tools_are_ignored(
        self, openai_client: OpenAI, responses_model: str, use_official_api: bool
    ) -> None:
        """Tool types without a backend equivalent are accepted and dropped.

        Validates:
            - The request succeeds with every hosted tool type present
            - The model still answers (the dropped tools impose no constraint)
        """
        if use_official_api:
            pytest.skip(
                "official API supports these tools; the drop is gateway-specific"
            )
        response = openai_client.responses.create(
            model=responses_model,
            input="Reply with OK.",
            tools=[
                {"type": "file_search", "vector_store_ids": ["vs_123"]},
                {"type": "computer"},
                {
                    "type": "computer_use_preview",
                    "display_height": 768,
                    "display_width": 1024,
                    "environment": "linux",
                },
                {"type": "mcp", "server_label": "my_server"},
                {"type": "local_shell"},
                {"type": "shell"},
                {"type": "custom", "name": "my_custom"},
                {"type": "namespace", "name": "ns", "description": "d", "tools": []},
                {"type": "tool_search"},
                {"type": "apply_patch"},
            ],
        )
        assert response.status == "completed"
        assert response.output_text

    @pytest.mark.parametrize(
        "extra",
        [
            {"tools": [{"type": "programmatic_tool_calling"}]},
            {"tool_choice": {"type": "programmatic_tool_calling"}},
        ],
        ids=["tool", "tool_choice"],
    )
    def test_programmatic_tool_calling_accepted_and_dropped(
        self,
        openai_client: OpenAI,
        responses_model: str,
        use_official_api: bool,
        extra: dict[str, object],
    ) -> None:
        """Programmatic tool calling is accepted and dropped on Converse models.

        Validates:
            - The request succeeds and the model answers directly
            - No program/program_output items are emitted
        """
        if use_official_api:
            pytest.skip(
                "official API serves programmatic tool calling on capable models"
            )
        response = openai_client.responses.create(  # type: ignore[call-overload]
            model=responses_model, input="Reply with OK.", **extra
        )
        assert response.status == "completed"
        assert not [
            item
            for item in response.output
            if item.type in ("program", "program_output")
        ]

    @pytest.mark.parametrize(("param", "value"), [("truncation", "auto")])
    def test_unsupported_param_returns_400(
        self,
        openai_client: OpenAI,
        responses_model: str,
        use_official_api: bool,
        param: str,
        value: object,
    ) -> None:
        """Unsupported request parameters are rejected with a 400 error.

        Validates:
            - BadRequestError is raised for each unsupported parameter
        """
        if use_official_api:
            pytest.skip(
                "official API supports these params; restriction is gateway-specific"
            )
        with pytest.raises(BadRequestError):
            openai_client.responses.create(  # type: ignore[call-overload]
                model=responses_model, input="Hello.", **{param: value}
            )


# ---------------------------------------------------------------------------
# image_generation integrated tool — model-agnostic (works for all text models)
# ---------------------------------------------------------------------------

#: Text models to exercise the image_generation tool against (one per provider family).
_IMAGE_GEN_TEXT_MODELS = (
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "amazon.nova-micro-v1:0",
)


class TestImageGenerationTool:
    """image_generation integrated tool works for all text models via the base gateway path.

    The gateway intercepts the ``image_generation`` tool, replaces it with a synthetic
    function tool the LLM calls with structured parameters, executes the actual image
    generation against a Bedrock image model, and returns an ``ImageGenerationCall``
    output item to the client — matching OpenAI's server-side image generation contract.

    When running against the official OpenAI API (``--use-official-api``), parametrized
    model variants are collapsed to a single run using ``responses_model`` (gpt-5-nano),
    and the tool definition omits the ``model`` field (OpenAI handles image generation
    server-side without a client-specified image model).
    """

    @pytest.mark.expensive
    @pytest.mark.parametrize("chat_model", _IMAGE_GEN_TEXT_MODELS)
    def test_image_generation_returns_image_call(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        chat_model: str,
        image_generation_model: str,
        responses_model: str,
    ) -> None:
        """image_generation tool produces an ImageGenerationCall item with base64 result.

        The gateway must suppress the ``function_call`` items from the LLM and replace
        them with ``image_generation_call`` output items containing the generated image.
        The official OpenAI API generates the image server-side and also returns an
        ``image_generation_call`` item.

        Validates:
            - At least one output item has type ``"image_generation_call"``
            - That item has ``status == "completed"`` and a non-empty base64 ``result``
            - No ``function_call`` items leak through to the client
            - Response status is ``"completed"``
        """
        if use_official_api and chat_model != _IMAGE_GEN_TEXT_MODELS[0]:
            pytest.skip("official API: collapsing parametrized variants to one run")
        if use_official_api:
            effective_model = responses_model
            tool: dict[str, object] = {"type": "image_generation"}
        else:
            effective_model = chat_model
            tool = {"type": "image_generation", "model": image_generation_model}
        try:
            resp = openai_client.responses.create(  # type: ignore[call-overload]
                model=effective_model,
                input="Generate a small red square image.",
                tools=[tool],
                tool_choice="required",
            )
        except BadRequestError as exc:
            if "does not exist" in str(exc) or "not available" in str(exc):
                pytest.xfail("Model not available in this environment")
            raise
        function_calls = [item for item in resp.output if item.type == "function_call"]
        image_calls = [
            item for item in resp.output if item.type == "image_generation_call"
        ]
        assert function_calls == [], (
            f"function_call items must not leak: {function_calls}"
        )
        assert len(image_calls) >= 1, (
            "Expected at least one image_generation_call output item"
        )
        assert image_calls[0].status == "completed"
        assert image_calls[0].result, "Expected non-empty base64 image result"
        assert resp.status == "completed"

    @pytest.mark.expensive
    @pytest.mark.parametrize("chat_model", _IMAGE_GEN_TEXT_MODELS)
    def test_image_generation_streaming_emits_image_call_item(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        chat_model: str,
        image_generation_model: str,
        responses_model: str,
    ) -> None:
        """Streaming image_generation emits an output_item.done event for the image call.

        The gateway must suppress ``response.function_call_arguments.*`` events and
        emit ``response.output_item.done`` with type ``"image_generation_call"`` at the
        end of the stream (after image generation completes).  The official OpenAI API
        emits the same event types.

        Validates:
            - Zero ``response.function_call_arguments.delta`` events
            - Zero ``response.function_call_arguments.done`` events
            - At least one ``response.output_item.done`` event with type ``"image_generation_call"``
            - Stream ends with ``response.completed``
        """
        if use_official_api and chat_model != _IMAGE_GEN_TEXT_MODELS[0]:
            pytest.skip("official API: collapsing parametrized variants to one run")
        if use_official_api:
            effective_model = responses_model
            tool: dict[str, object] = {"type": "image_generation"}
        else:
            effective_model = chat_model
            tool = {"type": "image_generation", "model": image_generation_model}
        func_delta_count = 0
        func_done_count = 0
        image_done_count = 0
        completed = False

        stream = openai_client.responses.create(  # type: ignore[call-overload]
            model=effective_model,
            input="Generate a small blue circle image.",
            tools=[tool],
            tool_choice="required",
            stream=True,
        )
        for event in stream:
            if event.type == "response.function_call_arguments.delta":
                func_delta_count += 1
            elif event.type == "response.function_call_arguments.done":
                func_done_count += 1
            elif event.type == "response.output_item.done":
                if getattr(event.item, "type", None) == "image_generation_call":
                    image_done_count += 1
            elif event.type == "response.completed":
                completed = True

        assert func_delta_count == 0, (
            f"function_call_arguments.delta leaked: {func_delta_count} events"
        )
        assert func_done_count == 0, (
            f"function_call_arguments.done leaked: {func_done_count} events"
        )
        assert image_done_count >= 1, (
            "Expected at least one image_generation_call output_item.done"
        )
        assert completed, "Expected response.completed event"


# ---------------------------------------------------------------------------
# input_tokens endpoint
# ---------------------------------------------------------------------------


class TestOpenAIInputTokens:
    """Test suite for POST /v1/responses/input_tokens (OpenAI Responses API).

    Validates token counting for the Responses API:
      - basic input
      - with `instructions` (system equivalent)
      - with `tools`
      - multi-turn message-array input
      - longer content yields more tokens
      - invalid model returns 400/404
      - structured input_text content blocks
    """

    def test_input_tokens_basic(
        self, openai_client: OpenAI, responses_input_tokens_model: str
    ) -> None:
        """Test basic token counting with a simple string input.

        Validates:
            - Response contains input_tokens field
            - Token count is a positive integer
        """
        response = openai_client.responses.input_tokens.count(
            model=responses_input_tokens_model, input="Hello, how are you?"
        )

        assert response.input_tokens > 0
        assert response.object == "response.input_tokens"

    def test_input_tokens_with_instructions(
        self, openai_client: OpenAI, responses_input_tokens_model: str
    ) -> None:
        """Test token counting includes instruction tokens.

        Validates:
            - Instructions contribute to token count
            - Token count with instructions is greater than without
        """
        response_without = openai_client.responses.input_tokens.count(
            model=responses_input_tokens_model, input="Hello"
        )

        response_with = openai_client.responses.input_tokens.count(
            model=responses_input_tokens_model,
            input="Hello",
            instructions="You are a very detailed and verbose assistant that always provides comprehensive answers.",
        )

        assert response_with.input_tokens > response_without.input_tokens

    def test_input_tokens_with_tools(
        self, openai_client: OpenAI, responses_input_tokens_model: str
    ) -> None:
        """Test token counting includes tool definition tokens.

        Validates:
            - Tool definitions contribute to token count
            - Token count with tools is greater than without
        """
        response_without = openai_client.responses.input_tokens.count(
            model=responses_input_tokens_model, input="What is the weather?"
        )

        response_with = openai_client.responses.input_tokens.count(
            model=responses_input_tokens_model,
            input="What is the weather?",
            tools=[
                {  # type: ignore[list-item]
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                }
            ],
        )

        assert response_with.input_tokens > response_without.input_tokens

    def test_input_tokens_multi_turn(
        self, openai_client: OpenAI, responses_input_tokens_model: str
    ) -> None:
        """Test token counting with multi-turn conversation.

        Validates:
            - Multi-turn messages are counted
            - More messages result in higher token count
        """
        response_single = openai_client.responses.input_tokens.count(
            model=responses_input_tokens_model, input="Hello"
        )

        response_multi = openai_client.responses.input_tokens.count(
            model=responses_input_tokens_model,
            input=[
                {"type": "message", "role": "user", "content": "Hello"},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": "Hi there! How can I help you?",
                },
                {
                    "type": "message",
                    "role": "user",
                    "content": "Tell me about Python programming.",
                },
            ],
        )

        assert response_multi.input_tokens > response_single.input_tokens

    def test_input_tokens_longer_content_more_tokens(
        self, openai_client: OpenAI, responses_input_tokens_model: str
    ) -> None:
        """Test that longer content produces more tokens.

        Validates:
            - Longer messages result in higher token counts
        """
        response_short = openai_client.responses.input_tokens.count(
            model=responses_input_tokens_model, input="Hi"
        )

        response_long = openai_client.responses.input_tokens.count(
            model=responses_input_tokens_model,
            input="Please explain the theory of relativity in great detail, "
            "covering both special and general relativity, their mathematical "
            "foundations, key experiments that confirmed them, and their "
            "implications for modern physics and cosmology.",
        )

        assert response_long.input_tokens > response_short.input_tokens

    def test_input_tokens_invalid_model(self, openai_client: OpenAI) -> None:
        """Test token counting with an invalid model returns an error.

        Validates:
            - Invalid model ID raises BadRequestError.
        """
        with pytest.raises(BadRequestError):
            openai_client.responses.input_tokens.count(
                model="nonexistent-model-xyz", input="Hello"
            )

    def test_input_tokens_input_text_blocks(
        self, openai_client: OpenAI, responses_input_tokens_model: str
    ) -> None:
        """Test token counting with input_text content blocks.

        Validates:
            - Input message with input_text blocks is accepted for counting
            - Returns a valid token count
        """
        response = openai_client.responses.input_tokens.count(
            model=responses_input_tokens_model,
            input=[
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello, how are you?"}],
                }
            ],
        )

        assert response.input_tokens > 0
        assert response.object == "response.input_tokens"


# ---------------------------------------------------------------------------
# code_interpreter integrated tool
# ---------------------------------------------------------------------------

#: Nova models exercised for code_interpreter (autonomous server-side execution).
_CODE_INTERP_MODELS = ("amazon.nova-2-lite-v1:0",)


class TestCodeInterpreterTool:
    """code_interpreter integrated tool tests via the Responses API.

    Local: Amazon Nova 2 Lite — ``code_interpreter`` maps to the
    ``nova_code_interpreter`` Bedrock system tool, which executes code
    autonomously in a single call.  The invocation is suppressed from output
    and the result appears in ``output_text``.

    Official API: ``gpt-5-nano`` — OpenAI executes Python code natively and
    returns a ``code_interpreter_call`` output item alongside the text result.

    .. note::
        Claude models previously mapped ``code_interpreter`` → ``bash`` (a
        server tool that requires a follow-up turn from the client), which does
        not match the OpenAI reference behaviour of autonomous single-turn
        execution.  That mapping has been removed; ``code_interpreter`` is only
        supported on Nova 2 (locally) and the official OpenAI API.
    """

    @pytest.mark.expensive
    @pytest.mark.parametrize("chat_model", _CODE_INTERP_MODELS)
    def test_code_interpreter_executes_and_returns_result(
        self,
        openai_client: OpenAI,
        responses_code_interpreter_model: str,
        use_official_api: bool,
        chat_model: str,
    ) -> None:
        """code_interpreter executes code autonomously and returns the result in output text.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_code_interpreter_model: Model for official API runs (gpt-5-nano)
            use_official_api: Whether we are testing against the official OpenAI API
            chat_model: Local model under test (parametrized)

        Validates:
            - Response status is ``"completed"``
            - Output text contains the expected numeric result (391 = 17 * 23)
            - Official API: at least one ``code_interpreter_call`` output item present
        """
        effective_model = (
            responses_code_interpreter_model if use_official_api else chat_model
        )
        tool: dict[str, object] = {
            "type": "code_interpreter",
            "container": {"type": "auto"},
        }
        resp = openai_client.responses.create(  # type: ignore[call-overload]
            model=effective_model,
            input=(
                "Use the code interpreter to calculate 17 * 23. "
                "Output only the numeric result, nothing else."
            ),
            tools=[tool],
            tool_choice="required",
        )
        assert resp.status == "completed"
        assert isinstance(resp.output_text, str)
        assert "391" in resp.output_text, (
            f"Expected '391' in output; got: {resp.output_text!r}"
        )
        if use_official_api:
            code_calls = [
                item for item in resp.output if item.type == "code_interpreter_call"
            ]
            assert len(code_calls) >= 1, (
                "Expected at least one code_interpreter_call output item from official API"
            )

    @pytest.mark.expensive
    @pytest.mark.parametrize("chat_model", _CODE_INTERP_MODELS)
    def test_code_interpreter_streaming(
        self,
        openai_client: OpenAI,
        responses_code_interpreter_model: str,
        use_official_api: bool,
        chat_model: str,
    ) -> None:
        """Streaming code_interpreter produces text delta events and completes.

        Args:
            openai_client: OpenAI client instance for API calls
            responses_code_interpreter_model: Model for official API runs (gpt-5-nano)
            use_official_api: Whether we are testing against the official OpenAI API
            chat_model: Local model under test (parametrized)

        Validates:
            - At least one ``response.output_text.delta`` event is emitted
            - Stream ends with ``response.completed``
            - Official API: at least one ``response.output_item.done`` event with type
              ``"code_interpreter_call"``
        """
        effective_model = (
            responses_code_interpreter_model if use_official_api else chat_model
        )

        text_delta_count = 0
        code_interp_done_count = 0
        completed = False

        tool: dict[str, object] = {
            "type": "code_interpreter",
            "container": {"type": "auto"},
        }
        stream = openai_client.responses.create(  # type: ignore[call-overload]
            model=effective_model,
            input="Calculate 8 + 7 using the code interpreter. Output only the number.",
            tools=[tool],
            tool_choice="required",
            stream=True,
        )
        for event in stream:
            if event.type == "response.output_text.delta":
                text_delta_count += 1
            elif event.type == "response.output_item.done":
                if getattr(event.item, "type", None) == "code_interpreter_call":
                    code_interp_done_count += 1
            elif event.type == "response.completed":
                completed = True

        assert text_delta_count >= 1, "Expected at least one output_text.delta event"
        assert completed, "Expected response.completed event"
        if use_official_api:
            assert code_interp_done_count >= 1, (
                "Expected at least one code_interpreter_call output_item.done from official API"
            )


class TestUsageLogging:
    """Tests for usage logging to stdout."""

    def test_response_usage_logged(
        self, test_client: TestClientType | None, responses_model: str, api_key: str
    ) -> None:
        """Test that usage is recorded and logged in API response and stdout."""
        if test_client is None:
            pytest.skip("Requires local test server")

        response = test_client.post(
            "/v1/responses",
            json={"model": responses_model, "input": "Say hello."},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert response.status_code == 200

        response_data = response.json()
        assert "usage" in response_data, (
            f"Response missing usage: {response_data.keys()}"
        )
        api_usage = response_data["usage"]
        assert api_usage is not None, "Response usage is None"
        assert api_usage.get("input_tokens", 0) > 0, "Expected input_tokens > 0"
        assert api_usage.get("output_tokens", 0) > 0, "Expected output_tokens > 0"

        assert "input_tokens" in api_usage
        assert "output_tokens" in api_usage
        assert "total_tokens" in api_usage


class TestUsageAggregation:
    """Tests for usage aggregation across multiple requests."""

    def test_multiple_requests_aggregate_usage(
        self, test_client: TestClientType | None, chat_legacy_model: str, api_key: str
    ) -> None:
        """Test that multiple requests to same model produce valid usage."""
        if test_client is None:
            pytest.skip("Requires local test server")

        for _ in range(3):
            response = test_client.post(
                "/v1/responses",
                json={"model": chat_legacy_model, "input": "Say hi."},
                headers={"Authorization": f"Bearer {api_key}"},
            )
            assert response.status_code == 200

            api_usage = response.json()["usage"]
            assert api_usage["input_tokens"] > 0
            assert api_usage["output_tokens"] > 0

    @pytest.mark.local
    def test_record_usage_twice_sums_values(self) -> None:
        """Test that calling record_usage twice produces one entry with summed values."""
        token = usage.init_usage()
        try:
            record_bedrock_usage("test-model", input_tokens=100, output_tokens=50)
            record_bedrock_usage("test-model", input_tokens=200, output_tokens=75)

            entries = list(usage.usage_log_entries())
        finally:
            usage.USAGE.reset(token)
        assert len(entries) == 1, (
            "Expected exactly one usage entry after two record_bedrock_usage calls"
        )

        entry = entries[0]
        assert entry["service"] == "bedrock-runtime"
        assert entry["model"] == "test-model"
        assert entry["input_tokens"] == 300, "input_tokens should be summed"
        assert entry["output_tokens"] == 125, "output_tokens should be summed"

    @pytest.mark.local
    def test_cache_write_tokens_by_ttl_logged(self) -> None:
        """Test that cache_write_tokens_by_ttl appears in usage entries."""
        token = usage.init_usage()
        try:
            record_bedrock_usage(
                "test-model",
                input_tokens=1000,
                output_tokens=100,
                cache_write_tokens_by_ttl={"5m": 500, "1h": 200},
            )

            entries = list(usage.usage_log_entries())
        finally:
            usage.USAGE.reset(token)
        assert len(entries) == 1, "Expected exactly one usage entry"

        entry = entries[0]
        assert "cache_write_tokens_by_ttl" in entry, (
            "Expected cache_write_tokens_by_ttl in entry"
        )
        cache_tokens = entry["cache_write_tokens_by_ttl"]
        assert cache_tokens["5m"] == 500
        assert cache_tokens["1h"] == 200


class TestUsageEMF:
    """Tests for CloudWatch EMF (Embedded Metric Format) usage emission."""

    def test_emf_metrics_emitted_when_enabled(
        self,
        test_client: TestClientType | None,
        responses_model: str,
        api_key: str,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """Test that EMF metrics are emitted when cloudwatch_metrics=True."""
        if test_client is None:
            pytest.skip("Requires local test server")

        capfd.readouterr()

        response = test_client.post(
            "/v1/responses",
            json={"model": responses_model, "input": "Hello."},
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert response.status_code == 200

        captured = capfd.readouterr()
        emf_lines = []
        for line in captured.out.split("\n"):
            if line.strip() and '"_aws"' in line:
                try:
                    emf_lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        assert emf_lines, "Expected at least one EMF line in stdout"

        emf = emf_lines[0]
        assert "_aws" in emf, "EMF line missing _aws key"
        aws = emf["_aws"]
        assert "CloudWatchMetrics" in aws, "EMF missing CloudWatchMetrics"
        assert "Timestamp" in aws, "EMF missing Timestamp"

        metrics_spec = aws["CloudWatchMetrics"][0]
        dimensions = metrics_spec["Dimensions"]
        assert ["Model"] in dimensions, "EMF dimensions should include ['Model']"
        for dimension_set in dimensions:
            for name in dimension_set:
                assert name in emf, (
                    f"EMF declares dimension {name!r} with no matching field"
                )
        assert "Metrics" in metrics_spec, "EMF missing Metrics"

        assert emf.get("Model") == responses_model, "EMF missing or wrong Model"
        assert "operation" in emf, "EMF missing operation field"

        metric_names = [m["Name"] for m in metrics_spec["Metrics"]]
        assert "InputTokens" in metric_names, "Expected InputTokens metric"
        assert "OutputTokens" in metric_names, "Expected OutputTokens metric"

        assert emf.get("InputTokens", 0) > 0, "InputTokens should be > 0"
        assert emf.get("OutputTokens", 0) > 0, "OutputTokens should be > 0"

    @pytest.mark.local
    def test_no_emf_when_cloudwatch_metrics_disabled(
        self, capfd: pytest.CaptureFixture[str]
    ) -> None:
        """Test that EMF metrics are NOT emitted when cloudwatch_metrics=False."""
        token = usage.init_usage()
        try:
            record_bedrock_usage("test-model", input_tokens=100, output_tokens=50)

            capfd.readouterr()
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(SETTINGS, "cloudwatch_metrics", False)
                usage.emit_usage_metrics()
        finally:
            usage.USAGE.reset(token)

        captured = capfd.readouterr()
        emf_lines = []
        for line in captured.out.split("\n"):
            if line.strip() and '"_aws"' in line:
                try:
                    emf_lines.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        assert not emf_lines, "Expected no EMF lines when cloudwatch_metrics=False"


@pytest.mark.local
class TestDeprecation:
    """Tests for the SETTINGS.deprecated() helper."""

    def test_tokens_estimation_deprecated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Setting a deprecated field is reported by deprecated()."""
        monkeypatch.setattr(SETTINGS, "__pydantic_fields_set__", {"tokens_estimation"})

        assert SETTINGS.deprecated() == {"tokens_estimation"}

    def test_no_deprecated_setting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No deprecated setting is reported when none was explicitly set."""
        monkeypatch.setattr(SETTINGS, "__pydantic_fields_set__", set())

        assert SETTINGS.deprecated() == set()
