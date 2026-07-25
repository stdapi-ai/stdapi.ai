"""Tests for the Anthropic /v1/messages route.

Comprehensive test suite that validates all features of the Anthropic Messages API
specification, ensuring compatibility with the official Anthropic API behavior.
"""

import base64
import json as _json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicError,
    BadRequestError,
    NotFoundError,
)
from starlette.testclient import TestClient

import stdapi.models as _models_mod
from stdapi.models import ModelDetails
from stdapi.routes import anthropic_messages

#: Non-Anthropic model used to validate that extended thinking is rejected for non-Claude models.
NON_ANTHROPIC_THINKING = "amazon.nova-2-lite-v1:0"


class TestAnthropicMessages:
    """Test suite for the /v1/messages endpoint (Anthropic API).

    Tests are designed to validate complete Anthropic Messages API compatibility including:
    - Basic message creation and response validation
    - Streaming behavior
    - Tool calling capabilities
    - System prompt handling
    - Multi-turn conversations
    - Extended thinking
    - Image/multimodal inputs
    - Parameter validation and error handling
    - Stop sequences
    - Temperature and sampling parameters
    """

    # --- Basic functionality ---

    def test_basic_message(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test fundamental message creation with default parameters.

        Validates:
            - Response contains content with text
            - Role is 'assistant'
            - Usage information is included
            - Response structure matches Anthropic specification
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )

        assert response.type == "message"
        assert response.role == "assistant"
        assert response.id is not None
        assert len(response.id) > 0
        assert response.model is not None
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert isinstance(response.content[0].text, str)
        assert len(response.content[0].text) > 0
        assert response.stop_reason == "end_turn"
        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    def test_multi_turn_conversation(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test multi-turn conversation with alternating user/assistant messages.

        Validates:
            - Multi-turn messages are accepted
            - Model responds coherently to conversation context
            - Response structure is valid
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
                {"role": "user", "content": "What is my name?"},
            ],
        )

        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert "Alice" in response.content[0].text

    def test_content_as_block_list(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test message with content provided as a list of content blocks.

        Validates:
            - Content blocks list format is accepted
            - TextBlockParam works correctly
            - Response is valid
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "Say hi."}]}
            ],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"

    # --- System prompt ---

    def test_system_prompt_string(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test system prompt provided as a plain string.

        Validates:
            - System prompt as string is accepted
            - Model follows system instructions
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            system="You are a pirate. Always respond with 'Arrr!'.",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"

    def test_system_prompt_text_blocks(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test system prompt provided as a list of text blocks.

        Validates:
            - System prompt as list of TextBlockParam is accepted
            - Response is valid
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            system=[
                {"type": "text", "text": "You are a helpful assistant."},
                {"type": "text", "text": "Be concise."},
            ],
            messages=[{"role": "user", "content": "Say hi."}],
        )

        assert response.type == "message"
        assert len(response.content) >= 1

    def test_system_role_in_messages(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_official_api: bool,
    ) -> None:
        """Test system prompt provided as a message with role='system'.

        Validates:
            - A message with role='system' is extracted as the system prompt
            - The response is valid and the system instruction is honoured
        """
        if use_official_api:
            pytest.skip("system-role messages in `messages` are a stdapi extension")
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Be concise.",
                },
                {"role": "user", "content": "Say hi."},
            ],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"

    def test_system_role_merged_with_system_field(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_official_api: bool,
    ) -> None:
        """Test that system-role messages are merged with the top-level system field.

        Validates:
            - Content from both sources is accepted without error
            - Response is valid
        """
        if use_official_api:
            pytest.skip("system-role messages in `messages` are a stdapi extension")
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            system="You are a helpful assistant.",
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Say hi."},
            ],
        )

        assert response.type == "message"
        assert len(response.content) >= 1

    def test_system_role_list_content_in_messages(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_official_api: bool,
    ) -> None:
        """Test system-role message with list-of-blocks content is extracted correctly.

        Validates:
            - System message whose content is a list of TextBlockParams is accepted
            - Non-TextBlockParam blocks in the list are silently dropped without error
            - Response is valid
        """
        if use_official_api:
            pytest.skip("system-role messages in `messages` are a stdapi extension")
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant. Be concise.",
                        }
                    ],
                },
                {"role": "user", "content": "Say hi."},
            ],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"

    def test_system_role_passthrough_as_message(
        self,
        anthropic_client: Anthropic,
        anthropic_system_as_messages_model: str,
        use_official_api: bool,
    ) -> None:
        """Test a mid-conversation system message on Claude Opus 4.8+.

        The first message must always be role='user'; a system-role message may only
        appear immediately after a user turn and must be the last entry (or be followed
        by an assistant turn). When SYSTEM_MESSAGE_AS_MESSAGES_SUPPORTED is True the
        message is forwarded to Bedrock as a native mid-conversation system instruction;
        while Bedrock lacks support the flag stays False and the message is extracted
        into the system field. Either way the request must succeed.

        Validates:
            - A mid-conversation system message is accepted after the last user turn
            - Response is valid (the instruction is applied as a system instruction)
        """
        if use_official_api:
            pytest.skip("system-role messages in `messages` are a stdapi extension")
        response = anthropic_client.messages.create(
            model=anthropic_system_as_messages_model,
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Hello."},
                {"role": "assistant", "content": "Hi! How can I help you?"},
                {"role": "user", "content": "How are you?"},
                {"role": "system", "content": "From now on, respond only in one word."},
                {"role": "user", "content": "Do you like the weather today?"},
            ],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert any(c.type == "text" for c in response.content)

    # --- Streaming ---

    def test_streaming_basic(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test basic streaming functionality.

        Validates:
            - Streaming response produces events
            - Events include message_start, content_block_start, content_block_delta,
              content_block_stop, message_delta, message_stop
            - Accumulated text forms a coherent response
        """
        event_types: list[str] = []
        accumulated_text = ""

        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Count to 3."}],
            stream=True,
        )

        for event in response:
            event_types.append(event.type)
            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                accumulated_text += event.delta.text

        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "content_block_stop" in event_types
        assert "message_delta" in event_types
        assert "message_stop" in event_types
        assert len(accumulated_text) > 0

    def test_streaming_with_create(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test streaming using create() with stream=True.

        Validates:
            - stream=True returns an iterable of raw events
            - Events can be iterated
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello."}],
            stream=True,
        )

        event_types: list[str] = []
        for event in response:
            event_types.append(event.type)
            if len(event_types) >= 20:
                break

        assert len(event_types) > 0
        assert "message_start" in event_types

    # --- Stop sequences ---

    def test_stop_sequences(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test stop sequences for controlling generation termination.

        Validates:
            - Stop sequences cause generation to stop
            - stop_reason reflects stop_sequence when triggered
            - Content before stop sequence is returned
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=200,
            messages=[
                {"role": "user", "content": "Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"}
            ],
            stop_sequences=["5"],
        )

        assert response.type == "message"
        assert response.stop_reason == "stop_sequence"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        # Content should include numbers before the stop sequence
        assert len(response.content[0].text) > 0

    # --- Temperature and sampling ---

    def test_temperature_parameter(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test temperature parameter for controlling randomness.

        Validates:
            - Temperature 0.0 is accepted (deterministic)
            - Temperature 1.0 is accepted (maximum randomness)
            - Response is valid in both cases
        """
        for temp in (0.0, 1.0):
            response = anthropic_client.messages.create(
                model=anthropic_chat_model,
                max_tokens=50,
                messages=[{"role": "user", "content": "Say hi."}],
                temperature=temp,
            )
            assert response.type == "message"
            assert len(response.content) >= 1

    def test_top_p_parameter(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test top_p nucleus sampling parameter.

        Validates:
            - top_p parameter is accepted
            - Response is valid
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hi."}],
            top_p=0.9,
        )
        assert response.type == "message"
        assert len(response.content) >= 1

    def test_top_k_parameter(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test top_k sampling parameter.

        Validates:
            - top_k parameter is accepted
            - Response is valid
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hi."}],
            top_k=40,
        )
        assert response.type == "message"
        assert len(response.content) >= 1

    # --- Tool calling ---

    def test_tool_calling_basic(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """Test basic tool calling capabilities.

        Validates:
            - Tool definitions are accepted
            - Model can decide to call tools
            - Tool use block structure is correct
        """
        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather information for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City and state"}
                    },
                    "required": ["location"],
                },
            }
        ]

        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_vision_model,
            max_tokens=300,
            messages=[{"role": "user", "content": "What's the weather in New York?"}],
            tools=tools,
            tool_choice={"type": "any"},
        )

        assert response.type == "message"
        assert response.stop_reason == "tool_use"

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_use_blocks) >= 1
        tool_block = tool_use_blocks[0]
        assert tool_block.name == "get_weather"
        assert tool_block.id is not None
        assert tool_block.id.startswith("toolu_")
        assert isinstance(tool_block.input, dict)
        assert "location" in tool_block.input

    def test_tool_calling_with_result(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """Test full tool calling flow with tool result.

        Validates:
            - Tool call followed by tool result produces final response
            - Model incorporates tool result in its response
        """
        tools = [
            {
                "name": "get_weather",
                "description": "Get current weather information",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ]

        # First call: get tool use
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_vision_model,
            max_tokens=300,
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=tools,
            tool_choice={"type": "any"},
        )

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_use_blocks) >= 1
        tool_block = tool_use_blocks[0]
        assert tool_block.id.startswith("toolu_")

        # Second call: provide tool result
        final = anthropic_client.messages.create(
            model=anthropic_chat_vision_model,
            max_tokens=300,
            messages=[
                {"role": "user", "content": "What's the weather in Paris?"},
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": _json.dumps(
                                {"temperature": "22C", "condition": "sunny"}
                            ),
                        }
                    ],
                },
            ],
            tools=tools,  # type: ignore[arg-type]
        )

        assert final.type == "message"
        assert final.stop_reason == "end_turn"
        text_blocks = [b for b in final.content if b.type == "text"]
        assert len(text_blocks) >= 1

    def test_tool_choice_auto(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """Test tool_choice auto - model decides whether to use tools.

        Validates:
            - tool_choice auto is accepted
            - Model can choose not to use tools when unnecessary
        """
        tools = [
            {
                "name": "calculator",
                "description": "Perform math calculations",
                "input_schema": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            }
        ]

        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_vision_model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=tools,
            tool_choice={"type": "auto"},
        )

        assert response.type == "message"
        assert response.stop_reason in ("end_turn", "tool_use")

    def test_tool_choice_specific_tool(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """Test tool_choice forcing a specific tool.

        Validates:
            - tool_choice with type=tool and name forces that tool
            - Model uses the specified tool
        """
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
            {
                "name": "get_time",
                "description": "Get current time",
                "input_schema": {
                    "type": "object",
                    "properties": {"timezone": {"type": "string"}},
                    "required": ["timezone"],
                },
            },
        ]

        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_vision_model,
            max_tokens=300,
            messages=[
                {"role": "user", "content": "What's the weather and time in Tokyo?"}
            ],
            tools=tools,
            tool_choice={"type": "tool", "name": "get_weather"},
        )

        assert response.type == "message"
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_use_blocks) >= 1
        # The forced tool must be present (Bedrock may return extra tools)
        assert tool_use_blocks[0].name == "get_weather"

    def test_tool_calling_streaming(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """Test tool calling with streaming.

        Validates:
            - Tool calls are properly streamed
            - Events include content_block_start with tool_use type
            - Input JSON is streamed as deltas
        """
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ]

        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_vision_model,
            max_tokens=300,
            messages=[{"role": "user", "content": "What's the weather in London?"}],
            tools=tools,
            tool_choice={"type": "any"},
            stream=True,
        )

        event_types = [event.type for event in response]

        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "message_stop" in event_types

    def test_extended_thinking_enabled(
        self, anthropic_client: Anthropic, anthropic_chat_reasoning_model: str
    ) -> None:
        """Test extended thinking with enabled configuration.

        Validates:
            - Thinking config is accepted
            - Response includes a thinking block
            - Response includes text content with the answer
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_reasoning_model,
            max_tokens=4000,
            messages=[{"role": "user", "content": "What is 15 * 27?"}],
            thinking={"type": "enabled", "budget_tokens": 1024},
        )

        assert response.type == "message"
        assert len(response.content) >= 1

        thinking_blocks = [b for b in response.content if b.type == "thinking"]
        text_blocks = [b for b in response.content if b.type == "text"]
        assert len(thinking_blocks) >= 1
        assert len(thinking_blocks[0].thinking) > 0
        assert len(text_blocks) >= 1
        assert "405" in text_blocks[0].text

    def test_extended_thinking_non_claude_enabled(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Test extended thinking with enabled configuration with a model that is not Claude.

        Validates:
            - Thinking config is accepted
            - Response includes a thinking block
            - Response includes text content with the answer
        """
        if use_official_api:
            pytest.skip("Only Claude models are supported by official API")

        response = anthropic_client.messages.create(
            model=NON_ANTHROPIC_THINKING,
            max_tokens=4000,
            messages=[{"role": "user", "content": "What is 15 * 27?"}],
            thinking={"type": "adaptive"},
        )

        assert response.type == "message"
        assert len(response.content) >= 1

        thinking_blocks = [b for b in response.content if b.type == "thinking"]
        text_blocks = [b for b in response.content if b.type == "text"]
        assert len(thinking_blocks) >= 1
        assert len(thinking_blocks[0].thinking) > 0
        assert len(text_blocks) >= 1
        assert "405" in text_blocks[0].text

    def test_output_config_effort_without_thinking(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Test output_config with effort but without thinking field.

        Validates:
            - output_config.effort is accepted without thinking field
            - Response includes a thinking block
            - Response includes text content with the answer
        """
        if use_official_api:
            pytest.skip("Only Claude models are supported by official API")

        response = anthropic_client.messages.create(
            model=NON_ANTHROPIC_THINKING,
            max_tokens=4000,
            messages=[{"role": "user", "content": "What is 15 * 27?"}],
            output_config={"effort": "medium"},
        )

        assert response.type == "message"
        assert len(response.content) >= 1

        thinking_blocks = [b for b in response.content if b.type == "thinking"]
        text_blocks = [b for b in response.content if b.type == "text"]
        assert len(thinking_blocks) >= 1
        assert len(thinking_blocks[0].thinking) > 0
        assert len(text_blocks) >= 1
        assert "405" in text_blocks[0].text

    def test_extended_thinking_streaming(
        self, anthropic_client: Anthropic, anthropic_chat_reasoning_model: str
    ) -> None:
        """Test extended thinking with streaming.

        Validates:
            - Thinking events are streamed
            - Both thinking_delta and text_delta events are received
        """
        events = list(
            anthropic_client.messages.create(
                model=anthropic_chat_reasoning_model,
                max_tokens=4000,
                messages=[{"role": "user", "content": "What is 15 * 27?"}],
                thinking={"type": "enabled", "budget_tokens": 1024},
                stream=True,
            )
        )

        event_types = [e.type for e in events]
        delta_types = {e.delta.type for e in events if e.type == "content_block_delta"}

        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "message_stop" in event_types
        assert "thinking_delta" in delta_types
        assert "text_delta" in delta_types

    def test_extended_thinking_non_claude_streaming(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Test extended thinking with streaming with a model that is not Claude.

        Validates:
            - Thinking events are streamed
            - Both thinking_delta and text_delta events are received
        """
        if use_official_api:
            pytest.skip("Only Claude models are supported by official API")

        events = list(
            anthropic_client.messages.create(
                model=NON_ANTHROPIC_THINKING,
                max_tokens=4000,
                messages=[{"role": "user", "content": "What is 15 * 27?"}],
                thinking={"type": "adaptive"},
                stream=True,
            )
        )

        event_types = [e.type for e in events]
        delta_types = {e.delta.type for e in events if e.type == "content_block_delta"}

        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "message_stop" in event_types
        assert "thinking_delta" in delta_types
        assert "text_delta" in delta_types

    # --- Image/multimodal input ---

    def test_image_base64_input(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_vision_model: str,
        sample_image_file: bytes,
    ) -> None:
        """Test image input via base64 encoding.

        Validates:
            - Base64 image content block is accepted
            - Model can describe the image
        """
        b64_data = base64.b64encode(sample_image_file).decode("utf-8")

        response = anthropic_client.messages.create(
            model=anthropic_chat_vision_model,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_data,
                            },
                        },
                        {"type": "text", "text": "What do you see in this image?"},
                    ],
                }
            ],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert len(response.content[0].text) > 0

    # --- Max tokens ---

    def test_max_tokens_limit(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test that max_tokens limits the response length.

        Validates:
            - Small max_tokens produces short response
            - stop_reason may be max_tokens
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=5,
            messages=[
                {
                    "role": "user",
                    "content": "Write a very long essay about the history of computing.",
                }
            ],
        )

        assert response.type == "message"
        assert response.usage.output_tokens <= 10  # Allow small margin

    # --- Metadata ---

    def test_metadata_user_id(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test metadata with user_id parameter.

        Validates:
            - Metadata with user_id is accepted
            - Response is valid
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hi."}],
            metadata={"user_id": "test-user-123"},
        )

        assert response.type == "message"
        assert len(response.content) >= 1

    # --- Error handling ---

    def test_empty_messages_error(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test that empty messages list produces an error.

        Validates:
            - Empty messages list is rejected
            - Appropriate error is returned
        """
        with pytest.raises(BadRequestError):
            anthropic_client.messages.create(
                model=anthropic_chat_model, max_tokens=100, messages=[]
            )

    def test_invalid_model_error(self, anthropic_client: Anthropic) -> None:
        """Test that an invalid model name produces an error.

        Validates:
            - Invalid model is rejected
            - Appropriate error type is returned (BadRequestError on official API, NotFoundError on local)
        """
        with pytest.raises((BadRequestError, NotFoundError)):
            anthropic_client.messages.create(
                model="nonexistent-model-xyz",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}],
            )

    def test_invalid_temperature_error(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test that invalid temperature values produce errors.

        Validates:
            - Temperature > 1.0 is rejected
            - Temperature < 0.0 is rejected
        """
        with pytest.raises(BadRequestError):
            anthropic_client.messages.create(
                model=anthropic_chat_model,
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}],
                temperature=2.0,
            )

    def test_invalid_max_tokens_error(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_official_api: bool,
    ) -> None:
        """Test that invalid max_tokens value produces an error.

        Validates:
            - max_tokens of 0 is rejected
        """
        if use_official_api:
            pytest.skip("the AWS-hosted official endpoint accepts max_tokens=0")
        with pytest.raises(BadRequestError):
            anthropic_client.messages.create(
                model=anthropic_chat_model,
                max_tokens=0,
                messages=[{"role": "user", "content": "Hello"}],
            )

    def test_invalid_top_p_error(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test that invalid top_p values produce errors.

        Validates:
            - top_p > 1.0 is rejected
        """
        with pytest.raises(BadRequestError):
            anthropic_client.messages.create(
                model=anthropic_chat_model,
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}],
                top_p=1.5,
            )

    # --- Multiple content blocks in response ---

    def test_tool_use_with_text_response(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """Test that tool use responses can include both text and tool_use blocks.

        Validates:
            - Response may contain mixed content block types
            - Both text and tool_use blocks are valid
        """
        tools = [
            {
                "name": "search",
                "description": "Search for information",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]

        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_vision_model,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": "Search for the capital of France and tell me about it.",
                }
            ],
            tools=tools,
            tool_choice={"type": "auto"},
        )

        assert response.type == "message"
        # Response should have at least one content block
        assert len(response.content) >= 1
        # All blocks should be valid types
        for block in response.content:
            assert block.type in ("text", "tool_use")

    # --- Streaming message_start event ---

    def test_streaming_message_start_has_usage(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test that streaming message_start event includes usage info.

        Validates:
            - message_start event contains message with usage
            - input_tokens is a non-negative integer (may be 0 for Bedrock-backed
              gateways where usage is reported only at message_delta)
        """
        message_start_event = None

        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Hi."}],
            stream=True,
        )

        for event in response:
            if event.type == "message_start":
                message_start_event = event
                break

        assert message_start_event is not None
        assert hasattr(message_start_event, "message")
        assert hasattr(message_start_event.message, "usage")
        assert message_start_event.message.usage.input_tokens >= 0

    def test_streaming_message_delta_has_usage(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test that streaming message_delta event includes output usage.

        Validates:
            - message_delta event contains usage with output_tokens
        """
        message_delta_event = None

        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Hi."}],
            stream=True,
        )

        for event in response:
            if event.type == "message_delta":
                message_delta_event = event

        assert message_delta_event is not None
        assert hasattr(message_delta_event, "usage")
        assert message_delta_event.usage.output_tokens > 0

    # --- Multiple tools ---

    def test_multiple_tools_defined(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """Test defining multiple tools.

        Validates:
            - Multiple tool definitions are accepted
            - Model can select the appropriate tool
        """
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather for a location",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
            {
                "name": "get_stock_price",
                "description": "Get stock price for a ticker symbol",
                "input_schema": {
                    "type": "object",
                    "properties": {"ticker": {"type": "string"}},
                    "required": ["ticker"],
                },
            },
        ]

        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_vision_model,
            max_tokens=300,
            messages=[{"role": "user", "content": "What's the weather in Berlin?"}],
            tools=tools,
            tool_choice={"type": "any"},
        )

        assert response.type == "message"
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_use_blocks) >= 1

    # --- Tool result with error ---

    def test_tool_result_with_is_error(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """Test sending a tool result with is_error=True.

        Validates:
            - Tool result with is_error flag is accepted
            - Model handles error results gracefully
        """
        tools = [
            {
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ]

        # First get a tool call
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_vision_model,
            max_tokens=300,
            messages=[{"role": "user", "content": "What's the weather in Mars?"}],
            tools=tools,
            tool_choice={"type": "any"},
        )

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_use_blocks) >= 1
        tool_block = tool_use_blocks[0]
        assert tool_block.id.startswith("toolu_")

        # Send error result
        final = anthropic_client.messages.create(
            model=anthropic_chat_vision_model,
            max_tokens=300,
            messages=[
                {"role": "user", "content": "What's the weather in Mars?"},
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": "Error: Location not found",
                            "is_error": True,
                        }
                    ],
                },
            ],
            tools=tools,  # type: ignore[arg-type]
        )

        assert final.type == "message"
        assert len(final.content) >= 1

    # --- Service tier ---

    def test_service_tier_parameter(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test service_tier parameter.

        Validates:
            - service_tier parameter is accepted
            - Response is valid
        """
        if isinstance(anthropic_client, AnthropicBedrock):
            pytest.xfail("Bedrock does not support service_tier parameter")

        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hi."}],
            service_tier="auto",
        )

        assert response.type == "message"
        assert len(response.content) >= 1

    # --- Streaming with system prompt ---

    def test_streaming_with_system_prompt(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test streaming with a system prompt.

        Validates:
            - System prompt works correctly in streaming mode
            - Events are properly received
        """
        accumulated_text = ""

        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            system="You are a helpful assistant. Be very concise.",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            stream=True,
        )

        for event in response:
            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                accumulated_text += event.delta.text

        assert len(accumulated_text) > 0

    # --- Content block index in streaming ---

    def test_streaming_content_block_indices(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test that streaming events have correct content block indices.

        Validates:
            - content_block_start events have index field
            - content_block_delta events have index field
            - content_block_stop events have index field
        """
        indices: dict[str, list[int]] = {"start": [], "delta": [], "stop": []}

        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hi."}],
            stream=True,
        )

        for event in response:
            if event.type == "content_block_start":
                indices["start"].append(event.index)
            elif event.type == "content_block_delta":
                indices["delta"].append(event.index)
            elif event.type == "content_block_stop":
                indices["stop"].append(event.index)

        assert len(indices["start"]) >= 1
        assert len(indices["stop"]) >= 1
        # First content block should have index 0
        assert indices["start"][0] == 0

    # --- Long conversation ---

    def test_long_multi_turn_conversation(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test a longer multi-turn conversation.

        Validates:
            - Multiple turns of conversation are handled
            - Response remains coherent
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Remember the number 42."},
                {"role": "assistant", "content": "I'll remember the number 42."},
                {"role": "user", "content": "Now add 8 to it."},
                {"role": "assistant", "content": "42 + 8 = 50."},
                {"role": "user", "content": "What number did we start with?"},
            ],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert hasattr(response.content[0], "text")
        assert "42" in response.content[0].text

    # --- Stop reason validation ---

    def test_stop_reason_end_turn(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test that normal completion has end_turn stop reason.

        Validates:
            - Normal completion returns end_turn stop_reason
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Say OK."}],
        )

        assert response.stop_reason == "end_turn"

    def test_stop_reason_max_tokens(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test that hitting max_tokens returns max_tokens stop reason.

        Validates:
            - Hitting token limit returns max_tokens stop_reason
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=1,
            messages=[
                {
                    "role": "user",
                    "content": "Write a very long detailed essay about the universe.",
                }
            ],
        )

        assert response.stop_reason == "max_tokens"

    def test_stop_reason_tool_use(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """Test that tool use returns tool_use stop reason.

        Validates:
            - Tool use returns tool_use stop_reason
        """
        tools = [
            {
                "name": "lookup",
                "description": "Look up information",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]

        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_vision_model,
            max_tokens=300,
            messages=[{"role": "user", "content": "Look up Python."}],
            tools=tools,
            tool_choice={"type": "any"},
        )

        assert response.stop_reason == "tool_use"

    # --- Model field in response ---

    def test_response_model_field(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """Test that response includes the model field matching the requested model.

        Validates:
            - Response model field is present and non-empty
            - On local gateway: response model echoes the requested model name exactly
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Hi."}],
        )

        assert response.model is not None
        assert len(response.model) > 0
        if not use_anthropic_api:
            # Our gateway must echo back the exact requested model name
            assert response.model == anthropic_chat_model

    # --- Response ID format ---

    def test_response_id_format(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test that response ID has expected format.

        Validates:
            - Response ID starts with 'msg_'
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Hi."}],
        )

        assert response.id.startswith("msg_")

    # --- Streaming final message ---

    def test_streaming_get_final_message(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test getting the final assembled message from a stream.

        Validates:
            - Streaming produces all required events
            - Final state contains complete message info
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello."}],
            stream=True,
        )

        message_start = None
        message_delta = None
        has_content = False

        for event in response:
            if event.type == "message_start":
                message_start = event.message
            elif event.type == "message_delta":
                message_delta = event
            elif event.type == "content_block_start":
                has_content = True

        assert message_start is not None
        assert message_start.type == "message"
        assert message_start.role == "assistant"
        assert has_content
        assert message_delta is not None
        assert message_delta.delta.stop_reason is not None
        assert message_delta.usage.output_tokens > 0

    # --- Streaming get_final_text ---

    def test_streaming_get_final_text(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test getting the final text from a stream.

        Validates:
            - Accumulated text from streaming is coherent
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello."}],
            stream=True,
        )

        final_text = ""
        for event in response:
            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                final_text += event.delta.text

        assert isinstance(final_text, str)
        assert len(final_text) > 0

    # --- Multiple user content blocks ---

    def test_multiple_text_blocks_in_user_message(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test user message with multiple text content blocks.

        Validates:
            - Multiple text blocks in a single user message are accepted
            - Model processes all blocks
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "First part: My name is Bob."},
                        {"type": "text", "text": "Second part: What is my name?"},
                    ],
                }
            ],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert hasattr(response.content[0], "text")
        assert "Bob" in response.content[0].text

    # --- Prompt caching (cache_control) ---

    def test_cache_control_on_user_message_block(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test cache_control on a user message content block.

        Validates:
            - cache_control ephemeral on a text block is accepted
            - Response is valid
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Say hello in one word.",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert len(response.content[0].text) > 0

    def test_cache_control_on_system_prompt_block(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test cache_control on a system prompt text block.

        Validates:
            - cache_control ephemeral on a system text block is accepted
            - Model follows system instructions
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            system=[
                {
                    "type": "text",
                    "text": "You are a pirate. Always respond with 'Arrr'.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": "Say hello."}],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"

    def test_cache_control_on_tool(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """Test cache_control on a tool definition.

        Validates:
            - cache_control ephemeral on a tool is accepted
            - Tool calling still works correctly
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_vision_model,
            max_tokens=200,
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get the weather for a location.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"],
                    },
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        )

        assert response.type == "message"
        assert len(response.content) >= 1

    def test_cache_control_with_ttl(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test cache_control with explicit TTL value.

        Validates:
            - cache_control with ttl parameter is accepted
            - Response is valid
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Say hello in one word.",
                            "cache_control": {"type": "ephemeral", "ttl": "5m"},
                        }
                    ],
                }
            ],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert len(response.content[0].text) > 0

    def test_cache_control_streaming(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test cache_control works with streaming responses.

        Validates:
            - cache_control is accepted in streaming mode
            - Stream produces valid events
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Say hello in one word.",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
            stream=True,
        )

        final_text = ""
        for event in response:
            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                final_text += event.delta.text

        assert len(final_text) > 0

    def test_automatic_cache_control(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test automatic caching with top-level cache_control.

        Validates:
            - Top-level cache_control is accepted
            - System automatically applies cache breakpoint to last cacheable block
            - Response is valid
            - Explicit cache_control on blocks is ignored when automatic caching is enabled
        """
        try:
            response = anthropic_client.messages.create(
                model=anthropic_chat_model,
                max_tokens=100,
                system="You are a helpful assistant that remembers our conversation.",
                messages=[
                    {
                        "role": "user",
                        "content": "My name is Alex. I work on machine learning.",
                    },
                    {
                        "role": "assistant",
                        "content": "Nice to meet you, Alex! How can I help with your ML work today?",
                    },
                    {"role": "user", "content": "What did I say I work on?"},
                ],
                cache_control={"type": "ephemeral"},
            )
        except BadRequestError:
            if isinstance(anthropic_client, AnthropicBedrock):
                pytest.xfail("Bedrock does not support cache_control parameter")
            raise

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert len(response.content[0].text) > 0

    def test_automatic_cache_control_with_ttl(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test automatic caching with custom TTL.

        Validates:
            - Top-level cache_control with ttl parameter is accepted
            - Response is valid
        """
        try:
            response = anthropic_client.messages.create(
                model=anthropic_chat_model,
                max_tokens=100,
                system="You are a helpful assistant.",
                messages=[{"role": "user", "content": "Say hello in one word."}],
                cache_control={"type": "ephemeral", "ttl": "5m"},
            )
        except BadRequestError:
            if isinstance(anthropic_client, AnthropicBedrock):
                pytest.xfail("Bedrock does not support cache_control parameter")
            raise

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert len(response.content[0].text) > 0

    def test_output_config_json_schema(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test output_config with json_schema format.

        Validates:
            - output_config parameter with json_schema type is accepted
            - Response is valid and contains JSON content
        """
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
            "additionalProperties": False,
        }

        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": "Return a JSON object with name set to 'Alice' and age set to 30.",
                }
            ],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        result = _json.loads(response.content[0].text)
        assert result["name"] == "Alice"
        assert result["age"] == 30

    # --- Extended thinking multi-turn ---

    @pytest.mark.expensive
    def test_extended_thinking_multi_turn(
        self, anthropic_client: Anthropic, anthropic_chat_reasoning_model: str
    ) -> None:
        """Test multi-turn conversation that sends thinking blocks back.

        Validates:
            - Thinking blocks from a first response can be sent back in a follow-up
            - ThinkingBlockParam and content are correctly handled
            - Model produces a valid response in the second turn
        """
        first = anthropic_client.messages.create(
            model=anthropic_chat_reasoning_model,
            max_tokens=4000,
            messages=[{"role": "user", "content": "What is 15 * 27?"}],
            thinking={"type": "enabled", "budget_tokens": 1024},
        )

        assert first.type == "message"
        assert len(first.content) >= 1

        # Send the full assistant content (including thinking blocks) back
        second = anthropic_client.messages.create(
            model=anthropic_chat_reasoning_model,
            max_tokens=4000,
            messages=[
                {"role": "user", "content": "What is 15 * 27?"},
                {"role": "assistant", "content": first.content},
                {"role": "user", "content": "Now multiply that result by 2."},
            ],
            thinking={"type": "enabled", "budget_tokens": 1024},
        )

        assert second.type == "message"
        text_blocks = [b for b in second.content if b.type == "text"]
        assert len(text_blocks) >= 1
        assert len(text_blocks[0].text) > 0

    @pytest.mark.expensive
    def test_extended_thinking_streaming_content(
        self, anthropic_client: Anthropic, anthropic_chat_reasoning_model: str
    ) -> None:
        """Test extended thinking streaming with detailed content verification.

        Validates:
            - Thinking deltas are received during streaming
            - Both thinking and text content block types appear
            - Signature delta is present for thinking blocks
        """
        events = list(
            anthropic_client.messages.create(
                model=anthropic_chat_reasoning_model,
                max_tokens=4000,
                messages=[{"role": "user", "content": "What is 15 * 27?"}],
                thinking={"type": "enabled", "budget_tokens": 1024},
                stream=True,
            )
        )

        event_types = [e.type for e in events]
        assert "content_block_start" in event_types
        assert "content_block_delta" in event_types
        assert "message_delta" in event_types

        # Verify thinking deltas are present
        delta_types = set()
        for e in events:
            if e.type == "content_block_delta":
                delta_types.add(e.delta.type)

        assert "thinking_delta" in delta_types
        assert "text_delta" in delta_types

    def test_tool_choice_any(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """Test tool_choice with 'any' type forces tool use.

        Validates:
            - tool_choice any is accepted
            - Model is forced to use a tool
            - stop_reason is tool_use
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_vision_model,
            max_tokens=300,
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            tools=[
                {
                    "name": "greet",
                    "description": "Generate a greeting",
                    "input_schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    },
                }
            ],
            tool_choice={"type": "any"},
        )

        assert response.type == "message"
        assert response.stop_reason == "tool_use"
        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_blocks) >= 1
        assert all(b.id.startswith("toolu_") for b in tool_blocks)

    def test_streaming_tool_calling_events(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """Test streaming tool calling with detailed event verification.

        Validates:
            - Tool use content_block_start events contain tool info
            - input_json_delta events are received for tool input
            - content_block_stop events are received
        """
        events = list(
            anthropic_client.messages.create(
                model=anthropic_chat_vision_model,
                max_tokens=300,
                messages=[{"role": "user", "content": "What's the weather in Paris?"}],
                tools=[
                    {
                        "name": "get_weather",
                        "description": "Get weather for a location",
                        "input_schema": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    }
                ],
                tool_choice={"type": "any"},
                stream=True,
            )
        )

        event_types = [e.type for e in events]
        assert "content_block_start" in event_types
        assert "content_block_stop" in event_types

        # Verify input_json_delta is present
        delta_types = set()
        for e in events:
            if e.type == "content_block_delta":
                delta_types.add(e.delta.type)

        assert "input_json_delta" in delta_types

    def test_stop_reason_stop_sequence(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test that stop_reason is stop_sequence when a stop sequence is hit.

        Validates:
            - stop_reason is 'stop_sequence' when the model hits a stop sequence
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=200,
            messages=[
                {"role": "user", "content": "Count from 1 to 10, separated by commas."}
            ],
            stop_sequences=["5"],
        )

        assert response.type == "message"
        assert response.stop_reason == "stop_sequence"

    def test_anthropic_beta_header_passthrough(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test that the anthropic-beta header is passed through to Bedrock.

        Validates:
            - Request with anthropic-beta header succeeds
            - Response structure is valid
            - The header is accepted and does not cause errors
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            extra_headers={"anthropic-beta": "context-management-2025-06-27"},
        )

        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert isinstance(response.content[0].text, str)
        assert len(response.content[0].text) > 0
        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    def test_anthropic_beta_header_passthrough_filter(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test that the unsupported anthropic-beta header is filtered.

        Validates:
            - Request with unsupported anthropic-beta header succeeds
            - The header is accepted and does not cause errors
        """
        anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            extra_headers={"anthropic-beta": "claude-code-20250219"},
        )

    def test_document_plain_text(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test document block with plain text source."""
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "The capital of France is Paris.",
                            },
                            "title": "France Facts",
                        },
                        {
                            "type": "text",
                            "text": "What is the capital of France according to the document?",
                        },
                    ],
                }
            ],
        )
        assert response.type == "message"
        assert len(response.content) >= 1
        assert hasattr(response.content[0], "text")
        text = response.content[0].text
        assert "Paris" in text

    def test_document_plain_text_with_citations(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test document block with citations enabled on plain text.

        Note: Bedrock Converse may only support citations on PDF documents.
        This test uses content block source which maps to txt format.
        """
        try:
            response = anthropic_client.messages.create(
                model=anthropic_chat_model,
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "document",
                                "source": {
                                    "type": "content",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "Python was created by Guido van Rossum in 1991. It is a high-level programming language known for its readability and versatility.",
                                        }
                                    ],
                                },
                                "title": "PythonHistory",
                                "citations": {"enabled": True},
                            },
                            {"type": "text", "text": "Who created Python and when?"},
                        ],
                    }
                ],
            )
        except BadRequestError:
            pytest.xfail(
                "Citations on plain text documents may not be supported by Bedrock"
            )
            return
        assert response.type == "message"
        text_blocks = [b for b in response.content if b.type == "text"]
        assert len(text_blocks) >= 1

    def test_document_base64_pdf(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        sample_pdf_file: bytes,
    ) -> None:
        """Test document block with base64 PDF source."""
        pdf_b64 = base64.b64encode(sample_pdf_file).decode("utf-8")
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": pdf_b64,
                            },
                            "title": "Test PDF",
                        },
                        {"type": "text", "text": "What text is in this PDF document?"},
                    ],
                }
            ],
        )
        assert response.type == "message"
        assert len(response.content) >= 1

    def test_document_content_block_source(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test document block with content block source."""
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "content",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "The speed of light is 299,792,458 m/s.",
                                    }
                                ],
                            },
                            "title": "Physics Facts",
                        },
                        {"type": "text", "text": "What is the speed of light?"},
                    ],
                }
            ],
        )
        assert response.type == "message"
        assert len(response.content) >= 1

    def test_document_with_context(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test document block with context field."""
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "Revenue: $1.5B. Profit: $200M.",
                            },
                            "title": "Q4 Report",
                            "context": "This is a quarterly financial report.",
                        },
                        {"type": "text", "text": "What was the revenue?"},
                    ],
                }
            ],
        )
        assert response.type == "message"
        text_blocks = [b for b in response.content if b.type == "text"]
        assert len(text_blocks) >= 1

    # --- Tool choice none ---

    def test_tool_choice_none_raises_error(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """Test that tool_choice 'none' raises an error on this implementation.

        The official Anthropic API supports tool_choice 'none', but this
        implementation (Bedrock Converse) does not.
        """
        if use_anthropic_api:
            pytest.skip("tool_choice 'none' is supported on the official API")

        with pytest.raises(BadRequestError):
            anthropic_client.messages.create(
                model=anthropic_chat_model,
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}],
                tools=[
                    {
                        "name": "test_tool",
                        "description": "A test tool",
                        "input_schema": {"type": "object", "properties": {}},
                    }
                ],
                tool_choice={"type": "none"},
            )

    # --- Cache creation input tokens ---

    def test_cache_creation_input_tokens_in_usage(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test that usage includes cache_creation_input_tokens field."""
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hi."}],
        )
        assert response.usage is not None
        cache_creation = response.usage.cache_creation_input_tokens
        assert cache_creation is None or isinstance(cache_creation, int)

    # --- Search result block input ---

    def test_search_result_block_input(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test search result block as input content."""
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "search_result",
                            "source": "https://example.com/article",
                            "title": "Example Article",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "The population of Tokyo is approximately 14 million.",
                                }
                            ],
                        },
                        {"type": "text", "text": "What is the population of Tokyo?"},
                    ],
                }
            ],
        )
        assert response.type == "message"
        text_blocks = [b for b in response.content if b.type == "text"]
        assert len(text_blocks) >= 1

    # --- Streaming with document ---

    def test_streaming_with_document(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test streaming response with document input."""
        collected_text = []
        with anthropic_client.messages.stream(
            model=anthropic_chat_model,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "text",
                                "media_type": "text/plain",
                                "data": "The answer to everything is 42.",
                            },
                            "title": "Guide",
                        },
                        {"type": "text", "text": "What is the answer to everything?"},
                    ],
                }
            ],
        ) as stream:
            collected_text = list(stream.text_stream)

        full_text = "".join(collected_text)
        assert len(full_text) > 0
        assert "42" in full_text

    # --- Streaming with stop sequences ---

    def test_streaming_with_stop_sequences(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test streaming with stop_sequences produces stop_sequence stop reason.

        Validates:
            - Streaming + stop_sequences works together
            - message_delta reports stop_reason as stop_sequence
        """
        stop_reason = None

        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=200,
            messages=[
                {"role": "user", "content": "Count: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10"}
            ],
            stop_sequences=["5"],
            stream=True,
        )

        for event in response:
            if event.type == "message_delta":
                stop_reason = event.delta.stop_reason

        assert stop_reason == "stop_sequence"

    # --- Error format validation ---

    def test_invalid_model_error_format(
        self, anthropic_client: Anthropic, use_anthropic_api: bool
    ) -> None:
        """Test that error responses match Anthropic error format.

        Validates:
            - Error body has {"type": "error", "error": {"type": "...", "message": "..."}}
        """
        if use_anthropic_api:
            pytest.skip("Error format varies on official API")

        with pytest.raises(AnthropicError) as exc_info:
            anthropic_client.messages.create(
                model="nonexistent-model-xyz",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}],
            )
        error = exc_info.value
        assert hasattr(error, "body")
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "error"
        assert "error" in body
        assert "type" in body["error"]
        assert "message" in body["error"]

    # --- Negative temperature ---

    def test_invalid_negative_temperature_error(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test that negative temperature value produces an error.

        Validates:
            - Temperature below 0 is rejected
        """
        with pytest.raises(BadRequestError):
            anthropic_client.messages.create(
                model=anthropic_chat_model,
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}],
                temperature=-0.5,
            )

    # --- Thinking disabled explicitly ---

    def test_thinking_disabled_explicitly(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Test thinking=disabled produces normal response without thinking blocks.

        Validates:
            - thinking type disabled is accepted
            - Response contains no thinking blocks
            - Response contains normal text content
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            messages=[{"role": "user", "content": "What is 2+2?"}],
            thinking={"type": "disabled"},
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        thinking_blocks = [b for b in response.content if b.type == "thinking"]
        assert len(thinking_blocks) == 0
        text_blocks = [b for b in response.content if b.type == "text"]
        assert len(text_blocks) >= 1

    # --- Model alias resolution ---

    def test_model_alias_resolution(
        self, anthropic_client: Anthropic, use_anthropic_api: bool
    ) -> None:
        """Test that Anthropic-style model aliases are resolved correctly.

        Validates:
            - An alias like 'claude-haiku-4-5-20251001' (without Bedrock prefix)
              resolves to the correct model on the local gateway
        """
        if use_anthropic_api:
            pytest.skip("Alias resolution is a local gateway feature")

        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hi."}],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"


class TestAnthropicCountTokens:
    """Test suite for the /v1/messages/count_tokens endpoint (Anthropic API).

    Tests are designed to validate the token counting functionality including:
    - Basic token counting
    - Token counting with system prompts
    - Token counting with tools
    - Error handling for invalid models
    """

    def test_count_tokens_basic(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """Test basic token counting with a simple message.

        Validates:
            - Response contains input_tokens field
            - Token count is a positive integer
        """
        try:
            response = anthropic_client.messages.count_tokens(
                model=anthropic_count_tokens_model,
                messages=[{"role": "user", "content": "Hello, how are you?"}],
            )
        except AnthropicError as exc:
            if isinstance(
                anthropic_client, AnthropicBedrock
            ) and "not supported" in str(exc):
                pytest.xfail("Token counting is not supported in Bedrock yet")
            raise

        assert response.input_tokens > 0

    def test_count_tokens_with_system_prompt(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """Test token counting includes system prompt tokens.

        Validates:
            - System prompt contributes to token count
            - Token count with system prompt is greater than without
        """
        try:
            response_without = anthropic_client.messages.count_tokens(
                model=anthropic_count_tokens_model,
                messages=[{"role": "user", "content": "Hello"}],
            )
        except AnthropicError as exc:
            if isinstance(
                anthropic_client, AnthropicBedrock
            ) and "not supported" in str(exc):
                pytest.xfail("Token counting is not supported in Bedrock yet")
            raise

        response_with = anthropic_client.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=[{"role": "user", "content": "Hello"}],
            system="You are a very detailed and verbose assistant that always provides comprehensive answers.",
        )

        assert response_with.input_tokens > response_without.input_tokens

    def test_count_tokens_with_tools(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """Test token counting includes tool definitions.

        Validates:
            - Tool definitions contribute to token count
            - Token count with tools is greater than without
        """
        try:
            response_without = anthropic_client.messages.count_tokens(
                model=anthropic_count_tokens_model,
                messages=[{"role": "user", "content": "What is the weather?"}],
            )
        except AnthropicError as exc:
            if isinstance(
                anthropic_client, AnthropicBedrock
            ) and "not supported" in str(exc):
                pytest.xfail("Token counting is not supported in Bedrock yet")
            raise

        response_with = anthropic_client.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=[{"role": "user", "content": "What is the weather?"}],
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                }
            ],
        )

        assert response_with.input_tokens > response_without.input_tokens

    def test_count_tokens_multi_turn(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """Test token counting with multi-turn conversation.

        Validates:
            - Multi-turn messages are counted
            - More messages result in higher token count
        """
        try:
            response_single = anthropic_client.messages.count_tokens(
                model=anthropic_count_tokens_model,
                messages=[{"role": "user", "content": "Hello"}],
            )
        except AnthropicError as exc:
            if isinstance(
                anthropic_client, AnthropicBedrock
            ) and "not supported" in str(exc):
                pytest.xfail("Token counting is not supported in Bedrock yet")
            raise

        response_multi = anthropic_client.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there! How can I help you?"},
                {"role": "user", "content": "Tell me about Python programming."},
            ],
        )

        assert response_multi.input_tokens > response_single.input_tokens

    def test_count_tokens_longer_content_more_tokens(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """Test that longer content produces more tokens.

        Validates:
            - Longer messages result in higher token counts
        """
        try:
            response_short = anthropic_client.messages.count_tokens(
                model=anthropic_count_tokens_model,
                messages=[{"role": "user", "content": "Hi"}],
            )
        except AnthropicError as exc:
            if isinstance(
                anthropic_client, AnthropicBedrock
            ) and "not supported" in str(exc):
                pytest.xfail("Token counting is not supported in Bedrock yet")
            raise

        response_long = anthropic_client.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=[
                {
                    "role": "user",
                    "content": "Please explain the theory of relativity in great detail, "
                    "covering both special and general relativity, their mathematical "
                    "foundations, key experiments that confirmed them, and their "
                    "implications for modern physics and cosmology.",
                }
            ],
        )

        assert response_long.input_tokens > response_short.input_tokens

    def test_count_tokens_invalid_model(self, anthropic_client: Anthropic) -> None:
        """Test token counting with an invalid model returns an error.

        Validates:
            - Invalid model ID raises NotFoundError (matching official Anthropic API).
        """
        try:
            with pytest.raises(NotFoundError):
                anthropic_client.messages.count_tokens(
                    model="nonexistent-model-xyz",
                    messages=[{"role": "user", "content": "Hello"}],
                )
        except AnthropicError as exc:
            if isinstance(
                anthropic_client, AnthropicBedrock
            ) and "not supported" in str(exc):
                pytest.xfail("Token counting is not supported in Bedrock yet")
            raise

    def test_count_tokens_content_blocks(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """Test token counting with content block list format.

        Validates:
            - Content blocks list format is accepted for counting
            - Returns a valid token count
        """
        try:
            response = anthropic_client.messages.count_tokens(
                model=anthropic_count_tokens_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is the meaning of life?"}
                        ],
                    }
                ],
            )
        except AnthropicError as exc:
            if isinstance(
                anthropic_client, AnthropicBedrock
            ) and "not supported" in str(exc):
                pytest.xfail("Token counting is not supported in Bedrock yet")
            raise

        assert response.input_tokens > 0

    def test_count_tokens_web_search_tool_ignored(
        self,
        anthropic_client: Anthropic,
        anthropic_count_tokens_model: str,
        use_official_api: bool,
    ) -> None:
        """Test that web search tools are ignored during token counting.

        Validates:
            - Token counting succeeds when web_search tool is present
            - The system tool does not cause an error
        """
        if use_official_api:
            pytest.skip("the official API rejects server tools in count_tokens")
        try:
            response = anthropic_client.messages.count_tokens(
                model=anthropic_count_tokens_model,
                messages=[{"role": "user", "content": "Hello"}],
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
            )
        except AnthropicError as exc:
            if isinstance(
                anthropic_client, AnthropicBedrock
            ) and "not supported" in str(exc):
                pytest.xfail("Token counting is not supported in Bedrock yet")
            raise

        assert response.input_tokens > 0

    def test_count_tokens_with_system_blocks(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """Test token counting with system prompt as text blocks.

        Validates:
            - System prompt as list of text blocks is accepted
            - Returns a valid token count
        """
        try:
            response = anthropic_client.messages.count_tokens(
                model=anthropic_count_tokens_model,
                messages=[{"role": "user", "content": "Hello"}],
                system=[{"type": "text", "text": "You are a helpful assistant."}],
            )
        except AnthropicError as exc:
            if isinstance(
                anthropic_client, AnthropicBedrock
            ) and "not supported" in str(exc):
                pytest.xfail("Token counting is not supported in Bedrock yet")
            raise

        assert response.input_tokens > 0

    def test_count_tokens_system_role_in_messages(
        self,
        anthropic_client: Anthropic,
        anthropic_count_tokens_model: str,
        use_official_api: bool,
    ) -> None:
        """Test that a system-role message contributes to the token count.

        Validates:
            - System-role message content is counted (not silently dropped)
            - Token count with system-role message exceeds count without it
        """
        if use_official_api:
            pytest.skip("system-role messages in `messages` are a stdapi extension")
        try:
            response_without = anthropic_client.messages.count_tokens(
                model=anthropic_count_tokens_model,
                messages=[{"role": "user", "content": "Hello"}],
            )
        except AnthropicError as exc:
            if isinstance(
                anthropic_client, AnthropicBedrock
            ) and "not supported" in str(exc):
                pytest.xfail("Token counting is not supported in Bedrock yet")
            raise

        response_with = anthropic_client.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a very detailed and verbose assistant.",
                },
                {"role": "user", "content": "Hello"},
            ],
        )

        assert response_with.input_tokens > response_without.input_tokens

    def test_count_tokens_system_role_equivalent_to_system_field(
        self,
        anthropic_client: Anthropic,
        anthropic_count_tokens_model: str,
        use_official_api: bool,
    ) -> None:
        """Test that system-role message and top-level system field yield the same token count.

        Validates:
            - Both paths for providing a system prompt produce an equivalent token count
        """
        if use_official_api:
            pytest.skip("system-role messages in `messages` are a stdapi extension")
        system_text = "You are a helpful assistant."
        try:
            response_field = anthropic_client.messages.count_tokens(
                model=anthropic_count_tokens_model,
                messages=[{"role": "user", "content": "Hello"}],
                system=system_text,
            )
        except AnthropicError as exc:
            if isinstance(
                anthropic_client, AnthropicBedrock
            ) and "not supported" in str(exc):
                pytest.xfail("Token counting is not supported in Bedrock yet")
            raise

        response_role = anthropic_client.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": "Hello"},
            ],
        )

        assert response_role.input_tokens == response_field.input_tokens


class TestAnthropicCountTokensDispatch:
    """Offline unit tests for count_tokens dispatch to the classic vs Mantle counter.

    Registers fake models directly in the model registry so the real
    `serves_via_mantle` dispatch condition runs unmocked; only the two
    counting functions are replaced with recorders.
    """

    @pytest.fixture
    def client(self, api_key: str) -> TestClient:
        """Test client without lifespan (no AWS startup), pre-authenticated."""
        from stdapi.main import app  # noqa: PLC0415

        return TestClient(
            app, headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        )

    @pytest.fixture
    def runtime_model(self, monkeypatch: pytest.MonkeyPatch) -> ModelDetails:
        """Register a fake Bedrock Runtime model in the model registry."""
        details = ModelDetails(
            id="test.count-tokens-runtime-model",
            name="Runtime Count Tokens Test",
            provider="Vendor",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
            regions=["us-east-1"],
        )
        monkeypatch.setitem(_models_mod._MODELS, details.id, details)  # noqa: SLF001
        monkeypatch.setitem(_models_mod._ALL_MODELS, details.id, details)  # noqa: SLF001
        return details

    @pytest.fixture
    def mantle_model(self, monkeypatch: pytest.MonkeyPatch) -> ModelDetails:
        """Register a fake Bedrock Mantle model in the model registry."""
        details = ModelDetails(
            id="test.count-tokens-mantle-model",
            name="Mantle Count Tokens Test",
            provider="Vendor",
            service="AWS Bedrock Mantle",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
            regions=["us-east-1"],
        )
        monkeypatch.setitem(_models_mod._MODELS, details.id, details)  # noqa: SLF001
        monkeypatch.setitem(_models_mod._ALL_MODELS, details.id, details)  # noqa: SLF001
        return details

    def test_runtime_model_uses_classic_bedrock_counter(
        self,
        client: TestClient,
        runtime_model: ModelDetails,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-Mantle model routes count_tokens to the classic Bedrock counter."""
        classic = AsyncMock(return_value=7)
        mantle = AsyncMock(return_value=99)
        monkeypatch.setattr(anthropic_messages, "count_tokens_via_bedrock", classic)
        monkeypatch.setattr(anthropic_messages, "_count_tokens_via_mantle", mantle)

        response = client.post(
            "/anthropic/v1/messages/count_tokens",
            json={
                "model": runtime_model.id,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["input_tokens"] == 7
        classic.assert_awaited_once()
        assert classic.await_args is not None
        assert runtime_model.id in classic.await_args.args
        mantle.assert_not_awaited()

    def test_mantle_served_model_uses_mantle_counter(
        self,
        client: TestClient,
        mantle_model: ModelDetails,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A Mantle-served model routes count_tokens to the Mantle counter."""
        classic = AsyncMock(return_value=7)
        mantle = AsyncMock(return_value=99)
        monkeypatch.setattr(anthropic_messages, "count_tokens_via_bedrock", classic)
        monkeypatch.setattr(anthropic_messages, "_count_tokens_via_mantle", mantle)

        response = client.post(
            "/anthropic/v1/messages/count_tokens",
            json={
                "model": mantle_model.id,
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["input_tokens"] == 99
        mantle.assert_awaited_once()
        assert mantle.await_args is not None
        assert mantle_model.id in mantle.await_args.args
        classic.assert_not_awaited()


class TestAnthropicMessagesUnknownModel:
    """Offline tests pinning the Anthropic-parity 404 for unknown models."""

    pytestmark = pytest.mark.local

    @pytest.fixture
    def client(self, api_key: str) -> TestClient:
        """Test client without lifespan (no AWS startup), pre-authenticated."""
        from stdapi.main import app  # noqa: PLC0415

        return TestClient(
            app, headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        )

    @pytest.fixture
    def _offline_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Seed one model and disable the AWS registry refresh on cache miss."""
        details = ModelDetails(
            id="test.some-registered-model",
            name="Registered Test Model",
            provider="Vendor",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
            regions=["us-east-1"],
        )
        monkeypatch.setitem(_models_mod._MODELS, details.id, details)  # noqa: SLF001
        monkeypatch.setitem(_models_mod._ALL_MODELS, details.id, details)  # noqa: SLF001
        monkeypatch.setattr(
            _models_mod, "initialize_bedrock_models", AsyncMock(return_value=None)
        )

    @pytest.mark.usefixtures("_offline_registry")
    def test_messages_unknown_model_returns_404(self, client: TestClient) -> None:
        """An unknown model returns 404 not_found_error like the official API."""
        response = client.post(
            "/anthropic/v1/messages",
            json={
                "model": "nonexistent-model-xyz",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 404, response.text
        body = response.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "not_found_error"

    @pytest.mark.usefixtures("_offline_registry")
    def test_count_tokens_unknown_model_returns_404(self, client: TestClient) -> None:
        """count_tokens with an unknown model returns 404 not_found_error."""
        response = client.post(
            "/anthropic/v1/messages/count_tokens",
            json={
                "model": "nonexistent-model-xyz",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 404, response.text
        assert response.json()["error"]["type"] == "not_found_error"


class TestAnthropicMessagesMaxTokensOptional:
    """Offline unit test pinning that ``max_tokens`` stays optional on /v1/messages.

    Intentional divergence from the official Anthropic API (which requires
    ``max_tokens``): when omitted here, the underlying model's default output
    length applies. Validation and dispatch are exercised against an app
    instance without the AWS-touching lifespan, with the model call stubbed.
    """

    pytestmark = pytest.mark.local

    @pytest.fixture
    def client(self, api_key: str) -> TestClient:
        """Test client without lifespan (no AWS startup), pre-authenticated."""
        from stdapi.main import app  # noqa: PLC0415

        return TestClient(
            app, headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        )

    def test_missing_max_tokens_is_accepted(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A /v1/messages request without ``max_tokens`` is accepted (200)."""
        details = ModelDetails(
            id="test.max-tokens-optional-model",
            name="Max Tokens Optional Test",
            provider="Vendor",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
            regions=["us-east-1"],
        )
        monkeypatch.setitem(_models_mod._MODELS, details.id, details)  # noqa: SLF001
        monkeypatch.setitem(_models_mod._ALL_MODELS, details.id, details)  # noqa: SLF001
        fake_message = {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
            "model": details.id,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        fake_model = SimpleNamespace(
            create_message=AsyncMock(return_value=fake_message)
        )
        monkeypatch.setattr(anthropic_messages, "get_chat_model", lambda _: fake_model)

        response = client.post(
            "/anthropic/v1/messages",
            json={"model": details.id, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200, response.text
        assert response.json()["content"][0]["text"] == "hi"
