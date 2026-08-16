"""Anthropic Messages surface: POST /v1/messages and POST /v1/messages/count_tokens.

The gateway serves both routes from AWS Bedrock Converse (or Bedrock Mantle for
Mantle-only models), so every test here pins Anthropic-compatible behavior that
the gateway has to reconstruct from Bedrock primitives.

Ref: https://platform.claude.com/docs/en/api/messages
     https://platform.claude.com/docs/en/api/messages/count_tokens
     stdapi/routes/anthropic_messages.py:create_message
     stdapi/routes/anthropic_messages.py:count_tokens
"""

import base64
import json as _json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal
from unittest.mock import AsyncMock

import httpx
import pytest
from anthropic import (
    Anthropic,
    AnthropicBedrock,
    AnthropicError,
    APIStatusError,
    BadRequestError,
    NotFoundError,
)

import stdapi.models as _models_mod
from stdapi.api_errors import ApiError
from stdapi.aws_bedrock import GUARDRAIL_CONFIG_VAR, PERFORMANCE_CONFIG_VAR
from stdapi.aws_bedrock_mantle import mantle_request_headers, validate_pruning_extras
from stdapi.config import SETTINGS
from stdapi.models import ModelDetails
from stdapi.models.chat._adapters._anthropic_message import translate_request
from stdapi.models.chat._default import ChatModel
from stdapi.routes import anthropic_messages
from stdapi.types.anthropic_messages import (
    Message,
    MessageCountTokensParams,
    MessageCreateParams,
    MessageDelta,
    MessageDeltaUsage,
    MessageParam,
)

if TYPE_CHECKING:
    from starlette.testclient import TestClient

#: Non-Anthropic model used to validate that extended thinking is rejected for non-Claude models.
NON_ANTHROPIC_THINKING = "amazon.nova-2-lite-v1:0"

#: Third-party HTTPS image used to exercise the ``{"type": "url"}`` image source.
_REMOTE_IMAGE_URL = (
    "https://raw.githubusercontent.com/JGoutin/asus-s14na-u12-uefi/"
    "refs/heads/master/data/block_diagram.png"
)

#: Single-parameter custom tool used by every tool-calling test on this route.
_WEATHER_TOOL: dict[str, object] = {
    "name": "get_weather",
    "description": "Get weather for a location",
    "input_schema": {
        "type": "object",
        "properties": {"location": {"type": "string"}},
        "required": ["location"],
    },
}


def _register_test_model(
    monkeypatch: pytest.MonkeyPatch, model_id: str, name: str, **extra: object
) -> ModelDetails:
    """Register a fake TEXT-in/TEXT-out model in the in-process model registry.

    Both registry dicts are seeded because the routes read the per-region
    ``_MODELS`` map while model resolution falls back to ``_ALL_MODELS``.

    Args:
        monkeypatch: Fixture used to undo the registry mutation after the test.
        model_id: Identifier the test sends as ``model``.
        name: Human-readable model name.
        **extra: Extra ``ModelDetails`` fields, e.g. ``service``.

    Returns:
        The registered model details.
    """
    details = ModelDetails(
        id=model_id,
        name=name,
        provider="Vendor",
        input_modalities=["TEXT"],
        output_modalities=["TEXT"],
        regions=["us-east-1"],
        **extra,  # type: ignore[arg-type]
    )
    monkeypatch.setitem(_models_mod._MODELS, details.id, details)  # noqa: SLF001
    monkeypatch.setitem(_models_mod._ALL_MODELS, details.id, details)  # noqa: SLF001
    return details


class TestAnthropicMessages:
    """POST /v1/messages: request mapping, response shape, streaming, tools and errors.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/models/chat/_adapters/_anthropic_message.py:translate_request
         stdapi/models/chat/_adapters/_anthropic_message.py:format_response
    """

    # --- Basic functionality ---

    def test_basic_message(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """A minimal request returns an assistant ``message`` with text content and usage.

        The gateway mints the response id as ``msg_{request_id}`` and rebuilds
        ``usage`` from Bedrock's ``TokenUsage``, whose ``inputTokens`` excludes
        cache tokens.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
             stdapi/models/chat/_adapters/_anthropic_message.py:format_response
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello in one word."}],
        )

        assert response.type == "message"
        assert response.role == "assistant"
        assert response.id is not None
        assert len(response.id) > 0
        assert response.id.startswith("msg_")
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
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """Earlier turns stay in context: the model answers from a prior user message.

        Alternating user/assistant turns map 1:1 onto Bedrock Converse
        ``messages``, so a fact stated in the first turn must still be
        recoverable in the third.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_messages
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
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
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """``content`` given as a ``TextBlockParam`` list behaves like the string shorthand.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_messages
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Reply with exactly the word TEAL and nothing else.",
                        }
                    ],
                }
            ],
        )

        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert "teal" in response.content[0].text.lower()

    # --- System prompt ---

    def test_system_prompt_string(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """A ``system`` string is honored: its instruction shapes the reply.

        Anthropic accepts ``system`` as a string; the gateway turns it into a
        single Bedrock Converse ``system`` text block.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_system_blocks
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
            max_tokens=100,
            system="Whatever the user writes, reply with exactly the word TEAL and nothing else.",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert "teal" in response.content[0].text.lower()

    def test_system_prompt_text_blocks(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """Every block of a ``system`` block list reaches the model, not just the first.

        ``system`` accepts an array of ``TextBlockParam``; the instruction that
        pins the answer is in the second block, so a mapping that kept only one
        block would fail here.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_system_blocks
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
            max_tokens=100,
            system=[
                {"type": "text", "text": "You are a helpful assistant."},
                {
                    "type": "text",
                    "text": "Whatever the user writes, reply with exactly the word TEAL and nothing else.",
                },
            ],
            messages=[{"role": "user", "content": "Say hi."}],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert "teal" in response.content[0].text.lower()

    @pytest.mark.gateway("system-role messages in `messages` are a stdapi extension")
    def test_system_role_in_messages(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """A leading ``role: "system"`` message is hoisted into the system prompt.

        Bedrock Converse has no system role inside ``messages``, so the gateway
        extracts such a turn into the Converse ``system`` blocks; the
        instruction it carries must still reach the model.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_adapters/_anthropic_message.py:_extract_system_messages
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
            max_tokens=100,
            messages=[
                {
                    "role": "system",
                    "content": "Whatever the user writes, reply with exactly the word TEAL and nothing else.",
                },
                {"role": "user", "content": "Say hi."},
            ],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert "teal" in response.content[0].text.lower()

    @pytest.mark.gateway("system-role messages in `messages` are a stdapi extension")
    def test_system_role_merged_with_system_field(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """A system-role message is appended after the top-level ``system`` field.

        Both sources are merged into one Converse ``system`` block list, the
        top-level field first; here each half of the instruction lives in a
        different source, so dropping either one changes the answer.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_adapters/_anthropic_message.py:_merge_system_content
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
            max_tokens=100,
            system="Whatever the user writes, reply with exactly one word and nothing else.",
            messages=[
                {"role": "system", "content": "That word is TEAL."},
                {"role": "user", "content": "Say hi."},
            ],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert "teal" in response.content[0].text.lower()

    @pytest.mark.gateway("system-role messages in `messages` are a stdapi extension")
    def test_system_role_list_content_in_messages(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """A system-role message whose ``content`` is a block list is hoisted block by block.

        ``_extract_system_messages`` keeps the ``TextBlockParam`` entries of such
        a list, so the instruction they carry must still reach the model.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_adapters/_anthropic_message.py:_extract_system_messages
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
            max_tokens=100,
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful assistant."},
                        {
                            "type": "text",
                            "text": "Whatever the user writes, reply with exactly the word TEAL and nothing else.",
                        },
                    ],
                },
                {"role": "user", "content": "Say hi."},
            ],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert "teal" in response.content[0].text.lower()

    @pytest.mark.gateway("system-role messages in `messages` are a stdapi extension")
    def test_system_role_passthrough_as_message(
        self, anthropic_client: Anthropic, anthropic_system_as_messages_model: str
    ) -> None:
        """A system-role message not followed by an assistant turn is folded into ``system``.

        Bedrock Converse requires the last turn to be a user one, so the
        "ends the array" placement Anthropic allows is unreachable: the gateway
        falls back to extracting the directive into the ``system`` blocks instead
        of forwarding it, and the request must still be accepted.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_is_historical_directive
        """
        response = anthropic_client.messages.create(
            model=anthropic_system_as_messages_model,
            # Room for reasoning models to think AND answer with text.
            max_tokens=512,
            messages=[
                {"role": "user", "content": "Hello."},
                {"role": "assistant", "content": "Hi! How can I help you?"},
                {"role": "user", "content": "How are you?"},
                {"role": "system", "content": "From now on, respond only in one word."},
                {"role": "user", "content": "Do you like the weather today?"},
            ],
        )

        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) >= 1
        assert any(c.type == "text" and c.text.strip() for c in response.content), (
            "folded system directive must still yield a text answer"
        )
        assert response.stop_reason in {"end_turn", "max_tokens"}

    @pytest.mark.gateway("system-role messages in `messages` are a stdapi extension")
    def test_system_role_forwarded_between_user_and_assistant_turns(
        self, anthropic_client: Anthropic, anthropic_system_as_messages_model: str
    ) -> None:
        """A user -> system -> assistant placement is forwarded to Bedrock as a system turn.

        On a model with ``SYSTEM_MESSAGE_AS_MESSAGES_SUPPORTED`` the directive
        stays in the message list and reaches Bedrock as a native
        ``role: "system"`` turn, so this exercises the forwarding path end to end
        rather than the folding fallback; Bedrock rejects the turn if the
        placement rules are not respected.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_prepare_messages_and_system
        """
        response = anthropic_client.messages.create(
            model=anthropic_system_as_messages_model,
            max_tokens=100,
            messages=[
                {"role": "user", "content": "Hello."},
                {"role": "system", "content": "From now on, respond only in one word."},
                {"role": "assistant", "content": "Hi."},
                {"role": "user", "content": "How are you?"},
            ],
        )

        assert response.type == "message"
        assert response.role == "assistant"
        assert any(c.type == "text" and c.text.strip() for c in response.content), (
            "forwarded system turn must still yield a text answer"
        )
        assert response.stop_reason in {"end_turn", "max_tokens"}

    # --- Streaming ---

    def test_streaming_basic(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """The SSE stream follows Anthropic's documented event order.

        Order is ``message_start`` -> per block ``content_block_start`` /
        ``content_block_delta`` / ``content_block_stop`` -> ``message_delta`` ->
        ``message_stop``, which the gateway reconstructs from the Bedrock
        ``ConverseStream`` events.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html
             stdapi/models/chat/_adapters/_anthropic_message.py:format_stream
        """
        event_types: list[str] = []
        accumulated_text = ""

        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
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
        assert event_types[0] == "message_start"
        assert event_types[-1] == "message_stop"
        assert (
            event_types.index("content_block_start")
            < event_types.index("content_block_delta")
            < event_types.index("content_block_stop")
            < event_types.index("message_delta")
        ), f"events out of documented order: {event_types}"

    def test_streaming_with_create(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """``stream=True`` on ``create`` yields raw events whose types are all documented.

        Only the event names of Anthropic's streaming taxonomy may appear; the
        stream is abandoned early, so this covers the opening frames rather than
        the full message.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_adapters/_anthropic_message.py:_make_message_start_event
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
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
        assert event_types[0] == "message_start"
        assert set(event_types) <= {
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
            "ping",
        }, f"unexpected event names: {sorted(set(event_types))}"

    # --- Stop sequences ---

    def test_stop_sequences(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A matched stop sequence ends the turn with ``stop_reason`` ``stop_sequence``.

        ``stop_sequences`` maps onto Converse ``inferenceConfig.stopSequences``;
        Bedrock does not report which sequence matched, so the gateway leaves
        ``stop_sequence`` null on Converse-served models.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_stop_reason
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
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """Both ends of the documented ``temperature`` range generate a normal message.

        Anthropic restricts ``temperature`` to 0.0-1.0 and the gateway forwards
        it as Converse ``inferenceConfig.temperature``; the sampling effect
        itself is not observable, so only acceptance and a well-formed
        completion are asserted.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        """
        for temp in (0.0, 1.0):
            response = anthropic_client.messages.create(
                model=anthropic_chat_basic_model,
                max_tokens=50,
                messages=[{"role": "user", "content": "Say hi."}],
                temperature=temp,
            )
            assert response.type == "message"
            assert len(response.content) >= 1
            assert response.content[0].type == "text"
            assert response.content[0].text.strip()
            assert response.usage.output_tokens > 0
            assert response.stop_reason in {"end_turn", "max_tokens"}

    def test_top_p_parameter(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """``top_p`` is accepted and generation still completes normally.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hi."}],
            top_p=0.9,
        )
        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert response.content[0].text.strip()
        assert response.usage.output_tokens > 0

    def test_top_k_parameter(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """``top_k`` is accepted even though Converse has no such field.

        ``inferenceConfig`` has no ``top_k``, so the gateway passes it through
        ``additionalModelRequestFields`` under the name the model expects; a
        wrong field name would make Bedrock reject the request.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
             stdapi/types/anthropic_messages.py:MessageCreateParams
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hi."}],
            top_k=40,
        )
        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert response.content[0].text.strip()
        assert response.usage.output_tokens > 0

    # --- Tool calling ---

    def test_tool_calling_basic(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """A forced tool call returns ``stop_reason`` ``tool_use`` and a ``toolu_`` block.

        Bedrock mints its own ``toolUseId``; the gateway rewrites it to
        Anthropic's ``toolu_`` form and rebuilds ``input`` as a JSON object.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
             stdapi/models/chat/_adapters/_anthropic_message.py:format_response
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
        location = str(tool_block.input["location"]).lower()
        assert any(hint in location for hint in ("new york", "nyc")), (
            f"tool input lost the requested location: {location!r}"
        )

    def test_tool_calling_with_result(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """A ``tool_result`` turn closes the tool loop and the answer uses its payload.

        The assistant turn is replayed verbatim (``tool_use`` block included) and
        answered with ``{"type": "tool_result", "tool_use_id", "content"}``; the
        gateway maps that onto a Bedrock ``toolResult`` block keyed by the same id.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_tool_result_to_bedrock
        """
        tools = [_WEATHER_TOOL]

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
        final_text = " ".join(b.text for b in text_blocks).lower()
        assert any(hint in final_text for hint in ("22", "sunny")), (
            f"answer ignored the tool result: {final_text!r}"
        )

    def test_tool_choice_auto(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """With ``tool_choice`` ``auto`` the model may skip the tool, and ``stop_reason`` follows.

        ``auto`` maps to Converse ``toolChoice: {"auto": {}}``. Which branch the
        model takes is its own decision, so the invariant asserted here is the
        agreement between ``stop_reason`` and the presence of ``tool_use`` blocks,
        plus the fact that only declared tools can be called.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools#forcing-tool-use
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_tool_choice
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
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        assert (response.stop_reason == "tool_use") == bool(tool_use_blocks), (
            "stop_reason must agree with the presence of tool_use blocks"
        )
        assert all(b.name == "calculator" for b in tool_use_blocks)
        assert all(b.type in ("text", "tool_use") for b in response.content)

    def test_tool_choice_specific_tool(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_vision_model: str,
        use_official_api: bool,
    ) -> None:
        """``tool_choice`` ``tool`` always calls that tool; only the gateway suppresses the others.

        Two tools are offered and the prompt asks for both. Upstream, ``tool`` only forces
        the named tool to be used, so Claude may call ``get_time`` in the same turn. The
        gateway sends Converse ``toolChoice: {"tool": {"name": ...}}`` and additionally
        drops every ``tool_use`` block for another tool through its ``forced_tool`` filter,
        so only ``get_weather`` blocks can survive there.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools#forcing-tool-use
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_is_suppressed_tool
        """
        tools = [
            _WEATHER_TOOL,
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
        assert response.stop_reason == "tool_use"
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_use_blocks) >= 1
        names = [b.name for b in tool_use_blocks]
        assert "get_weather" in names, "the forced tool must be called"
        if use_official_api:
            # Upstream forces the named tool without forbidding the other ones.
            assert set(names) <= {"get_weather", "get_time"}
        else:
            # Blocks for the other tool are filtered out by the forced-tool guard.
            assert names == ["get_weather"] * len(names)
        assert all(b.id.startswith("toolu_") for b in tool_use_blocks)

    def test_tool_calling_streaming(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """A streamed tool call opens a ``tool_use`` block and ends with ``stop_reason`` ``tool_use``.

        The ``content_block_start`` frame carries the fully formed ``tool_use``
        block (id and name), while the arguments arrive later as
        ``input_json_delta`` fragments.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_adapters/_anthropic_message.py:_handle_block_start
        """
        tools = [_WEATHER_TOOL]

        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_vision_model,
            max_tokens=300,
            messages=[{"role": "user", "content": "What's the weather in London?"}],
            tools=tools,
            tool_choice={"type": "any"},
            stream=True,
        )

        events = list(response)
        event_types = [event.type for event in events]

        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "message_stop" in event_types
        started_blocks = [
            event.content_block
            for event in events
            if event.type == "content_block_start"
        ]
        tool_starts = [block for block in started_blocks if block.type == "tool_use"]
        assert len(tool_starts) >= 1, f"no tool_use block was streamed: {event_types}"
        assert tool_starts[0].name == "get_weather"
        assert tool_starts[0].id.startswith("toolu_")
        stop_reasons = [
            event.delta.stop_reason for event in events if event.type == "message_delta"
        ]
        assert stop_reasons[-1:] == ["tool_use"]

    def test_extended_thinking_enabled(
        self, anthropic_client: Anthropic, anthropic_chat_reasoning_model: str
    ) -> None:
        """``thinking`` enabled with a 1,024-token budget yields a thinking block plus the answer.

        1,024 is the documented minimum ``budget_tokens`` and must stay below
        ``max_tokens``; the gateway maps it onto Bedrock's ``reasoningConfig`` and
        turns ``reasoningContent`` back into Anthropic ``thinking`` blocks.

        Ref: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlock.html
             stdapi/models/chat/_adapters/_anthropic_message.py:extract_reasoning
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_reasoning_model,
            max_tokens=1500,
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

    @pytest.mark.gateway("Only Claude models are supported by official API")
    def test_extended_thinking_non_claude_enabled(
        self, anthropic_client: Anthropic
    ) -> None:
        """``thinking`` adaptive works on a non-Claude Bedrock model too.

        Adaptive thinking carries no ``budget_tokens``; the gateway translates it
        into the reasoning configuration the target model expects, so a Nova model
        answers with ``thinking`` blocks on the Anthropic route as well.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlock.html
             stdapi/models/chat/_adapters/_anthropic_message.py:extract_reasoning
        """
        response = anthropic_client.messages.create(
            model=NON_ANTHROPIC_THINKING,
            max_tokens=2048,
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

    @pytest.mark.gateway("Only Claude models are supported by official API")
    def test_output_config_effort_without_thinking(
        self, anthropic_client: Anthropic
    ) -> None:
        """``output_config.effort`` alone enables reasoning, with no ``thinking`` field.

        The gateway derives its reasoning configuration from either ``thinking`` or
        ``output_config.effort``; the effort form is the one Anthropic moved to,
        and it must produce thinking blocks on its own.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
             stdapi/types/anthropic_messages.py:OutputConfigParam
             stdapi/models/chat/_adapters/_anthropic_message.py:extract_reasoning
        """
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
        """A streamed thinking turn emits ``thinking_delta`` frames before ``text_delta`` ones.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             https://platform.claude.com/docs/en/build-with-claude/extended-thinking
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_delta
        """
        events = list(
            anthropic_client.messages.create(
                model=anthropic_chat_reasoning_model,
                max_tokens=1500,
                messages=[{"role": "user", "content": "What is 15 * 27?"}],
                thinking={"type": "enabled", "budget_tokens": 1024},
                stream=True,
            )
        )

        event_types = [e.type for e in events]
        delta_types = [e.delta.type for e in events if e.type == "content_block_delta"]

        assert "message_start" in event_types
        assert "content_block_start" in event_types
        assert "message_stop" in event_types
        assert "thinking_delta" in delta_types
        assert "text_delta" in delta_types
        assert delta_types.index("thinking_delta") < delta_types.index("text_delta"), (
            "thinking must be streamed before the answer text"
        )

    @pytest.mark.gateway("Only Claude models are supported by official API")
    def test_extended_thinking_non_claude_streaming(
        self, anthropic_client: Anthropic
    ) -> None:
        """Adaptive thinking streams ``thinking_delta`` and ``text_delta`` on a non-Claude model.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_delta
        """
        events = list(
            anthropic_client.messages.create(
                model=NON_ANTHROPIC_THINKING,
                max_tokens=2048,
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
        """A base64 ``image`` block is forwarded and billed as input tokens.

        The gateway decodes the source and rebuilds it as a Bedrock ``image``
        content block. The fixture image is model-generated, so its subject is
        not asserted — only that the vision model consumed it and answered.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_image_to_bedrock
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
        assert response.usage.input_tokens > 0, "image tokens must be counted"

    def test_image_url_source_is_fetched(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """A ``{"type": "url"}`` image source is resolved and billed as input tokens.

        Upstream hands the URL to the model provider, whereas Bedrock Converse takes
        only inline bytes or an S3 location, so the gateway downloads the image
        itself.  A dropped download would still produce a plausible answer to the
        question, which is why the input-token count is the assertion that matters.

        The URL is probed first: a third-party outage or a repository rename must
        skip rather than be reported as a gateway failure.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_image_to_bedrock
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

        response = anthropic_client.messages.create(
            model=anthropic_chat_vision_model,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "url", "url": _REMOTE_IMAGE_URL},
                        },
                        {"type": "text", "text": "What do you see in this image?"},
                    ],
                }
            ],
        )

        assert response.type == "message"
        assert response.content
        assert response.content[0].type == "text"
        assert response.content[0].text
        assert response.usage.input_tokens > 100, (
            f"the fetched image must dominate the prompt cost, got "
            f"{response.usage.input_tokens} input tokens"
        )

    # --- Max tokens ---

    def test_max_tokens_limit(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """``max_tokens`` caps generation and the turn ends with ``stop_reason`` ``max_tokens``.

        ``max_tokens`` becomes Converse ``inferenceConfig.maxTokens``; the margin
        on the assertion absorbs providers that count the truncated token.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_stop_reason
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
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
        assert response.stop_reason == "max_tokens"

    # --- Metadata ---

    def test_metadata_user_id(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """``metadata.user_id`` is accepted and does not reach the response.

        Anthropic wants an opaque identifier there; the gateway only records it in
        its request log, so the round trip is invisible to the client and the
        request must simply succeed unchanged.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/routes/anthropic_messages.py:create_message
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hi."}],
            metadata={"user_id": "test-user-123"},
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert response.content[0].text.strip()
        assert response.usage.output_tokens > 0

    # --- Error handling ---

    def test_empty_messages_error(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_basic_model: str,
        use_official_api: bool,
    ) -> None:
        """An empty ``messages`` array is rejected with HTTP 400.

        Nothing in the gateway's own schema forbids the empty array, so the
        rejection comes from Bedrock Converse and is re-dressed as Anthropic's
        ``invalid_request_error`` envelope. The wording is AWS's, hence only the
        status and the error type are pinned.

        Ref: https://platform.claude.com/docs/en/api/errors
             https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html
             stdapi/api_providers/anthropic.py:_format_error
        """
        with pytest.raises(BadRequestError) as excinfo:
            anthropic_client.messages.create(
                model=anthropic_chat_basic_model, max_tokens=100, messages=[]
            )

        assert excinfo.value.status_code == 400
        if not use_official_api:
            assert excinfo.value.type == "invalid_request_error"

    def test_invalid_model_error(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """An unknown model id is rejected, as ``not_found_error`` on this gateway.

        The gateway raises ``UnsupportedModelError`` (404 ``not_found_error``) and
        names the rejected id in the message; the official endpoints answer 400 or
        404 depending on the backend, so both statuses are accepted there.

        Ref: https://platform.claude.com/docs/en/api/errors
             stdapi/api_errors.py:UnsupportedModelError
             stdapi/api_providers/anthropic.py:_STATUS
        """
        with pytest.raises((BadRequestError, NotFoundError)) as excinfo:
            anthropic_client.messages.create(
                model="nonexistent-model-xyz",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert excinfo.value.status_code in (400, 404)
        assert "nonexistent-model-xyz" in str(excinfo.value)
        if not use_official_api:
            assert excinfo.value.status_code == 404
            assert excinfo.value.type == "not_found_error"

    def test_invalid_temperature_error(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_basic_model: str,
        use_official_api: bool,
    ) -> None:
        """``temperature`` above the documented 1.0 ceiling is rejected with HTTP 400.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://platform.claude.com/docs/en/api/errors
             stdapi/types/anthropic_messages.py:MessageCreateParams
        """
        with pytest.raises(BadRequestError) as excinfo:
            anthropic_client.messages.create(
                model=anthropic_chat_basic_model,
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}],
                temperature=2.0,
            )

        assert excinfo.value.status_code == 400
        assert "temperature" in str(excinfo.value).lower()
        if not use_official_api:
            assert excinfo.value.type == "invalid_request_error"

    @pytest.mark.gateway("the AWS-hosted official endpoint accepts max_tokens=0")
    def test_invalid_max_tokens_error(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """``max_tokens: 0`` is rejected by this gateway with HTTP 400.

        A deliberate divergence: Anthropic documents ``max_tokens: 0`` as valid
        (it pre-warms the prompt cache without generating), but the gateway's
        request model constrains the field to ``>= 1``, so Pydantic validation
        turns it into an ``invalid_request_error``.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/types/anthropic_messages.py:MessageCreateParams
             stdapi/main.py:handle_validation_exception
        """
        with pytest.raises(BadRequestError) as excinfo:
            anthropic_client.messages.create(
                model=anthropic_chat_basic_model,
                max_tokens=0,
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert excinfo.value.status_code == 400
        assert excinfo.value.type == "invalid_request_error"
        assert "max_tokens" in str(excinfo.value)

    def test_invalid_top_p_error(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_basic_model: str,
        use_official_api: bool,
    ) -> None:
        """``top_p`` above 1.0 is rejected with HTTP 400.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://platform.claude.com/docs/en/api/errors
             stdapi/types/anthropic_messages.py:MessageCreateParams
        """
        with pytest.raises(BadRequestError) as excinfo:
            anthropic_client.messages.create(
                model=anthropic_chat_basic_model,
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}],
                top_p=1.5,
            )

        assert excinfo.value.status_code == 400
        assert "top_p" in str(excinfo.value).lower()
        if not use_official_api:
            assert excinfo.value.type == "invalid_request_error"

    # --- Multiple content blocks in response ---

    def test_tool_use_with_text_response(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """A turn may mix ``text`` and ``tool_use`` blocks, and nothing else.

        Bedrock returns preamble text alongside a ``toolUse`` block; the gateway
        preserves that order and maps each block to its Anthropic counterpart, so
        no other block type may appear for a client-tool turn.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             stdapi/models/chat/_adapters/_anthropic_message.py:format_response
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
        assert all(
            b.name == "search" for b in response.content if b.type == "tool_use"
        ), "only the declared tool may be called"
        assert response.stop_reason in ("end_turn", "tool_use")
        assert (response.stop_reason == "tool_use") == any(
            b.type == "tool_use" for b in response.content
        ), "stop_reason must agree with the presence of tool_use blocks"

    # --- Streaming message_start event ---

    def test_streaming_message_start_has_usage(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """``message_start`` carries an empty-content ``Message`` shell with a usage object.

        Bedrock reports token usage only in its trailing metadata event, so the
        gateway opens the stream with zeroed counters and fills them in on
        ``message_delta``; the shell itself must already be a valid ``Message``.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStreamMetadataEvent.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_make_message_start_event
        """
        message_start_event = None

        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
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
        message = message_start_event.message
        assert message.type == "message"
        assert message.role == "assistant"
        assert message.content == [], "message_start must open with empty content"
        assert message.stop_reason is None
        assert message.id.startswith("msg_")

    def test_streaming_message_delta_has_usage(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """``message_delta`` reports cumulative output usage.

        Anthropic specifies the ``message_delta`` usage counters as cumulative
        rather than incremental, so the values may only grow across events and the
        last one is the total for the message.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_adapters/_anthropic_message.py:_make_message_delta_event
        """
        message_delta_event = None
        output_token_counts: list[int] = []

        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Hi."}],
            stream=True,
        )

        for event in response:
            if event.type == "message_delta":
                message_delta_event = event
                output_token_counts.append(event.usage.output_tokens)

        assert message_delta_event is not None
        assert hasattr(message_delta_event, "usage")
        assert message_delta_event.usage.output_tokens > 0
        assert output_token_counts == sorted(output_token_counts), (
            f"message_delta usage must be cumulative: {output_token_counts}"
        )

    # --- Multiple tools ---

    def test_multiple_tools_defined(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """With several tools declared and ``tool_choice`` ``any``, the matching tool is called.

        Both definitions are sent as Converse ``toolSpec`` entries; the prompt only
        fits one of them, so the forced call must land on ``get_weather``.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             stdapi/models/chat/_adapters/_anthropic_message.py:_build_tool_config
        """
        tools = [
            _WEATHER_TOOL,
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
        assert response.stop_reason == "tool_use"
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_use_blocks) >= 1
        assert tool_use_blocks[0].name == "get_weather"
        assert all(
            b.name in ("get_weather", "get_stock_price") for b in tool_use_blocks
        )

    # --- Tool result with error ---

    def test_tool_result_with_is_error(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """A ``tool_result`` marked ``is_error`` is accepted and answered with text.

        Bedrock's ``toolResult`` carries a ``status`` field; the gateway sets it to
        ``error`` for ``is_error: true`` so the model is told the call failed
        instead of being handed the message as data.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_tool_result_to_bedrock
        """
        tools = [_WEATHER_TOOL]

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
        text_blocks = [b for b in final.content if b.type == "text"]
        assert len(text_blocks) >= 1, "the error result must still be answered in text"
        assert text_blocks[0].text.strip()
        assert final.usage.output_tokens > 0

    # --- Service tier ---

    @pytest.mark.parametrize("service_tier", ["auto", "standard_only"])
    def test_service_tier_parameter(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_basic_model: str,
        service_tier: Literal["auto", "standard_only"],
    ) -> None:
        """Both documented ``service_tier`` values are accepted and leave generation unaffected.

        ``auto`` has no Bedrock counterpart, so the gateway sends no explicit tier;
        ``standard_only`` is translated to the tier Bedrock spells ``default``. Neither
        is echoed in the response, so the observable contract on this route is that
        the backend accepts the translated value instead of rejecting the request.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
             stdapi/types/anthropic_messages.py:ServiceTiers
             stdapi/models/chat/_adapters/_anthropic_message.py:_SERVICES_TIERS
        """
        if isinstance(anthropic_client, AnthropicBedrock):
            pytest.xfail("Bedrock does not support service_tier parameter")

        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hi."}],
            service_tier=service_tier,
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert response.content[0].text.strip()
        assert response.usage.output_tokens > 0

    # --- Streaming with system prompt ---

    def test_streaming_with_system_prompt(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """A ``system`` prompt is honored in streaming mode too.

        The system blocks take the same path as in the buffered case, so the
        instruction must still shape the text assembled from ``text_delta``
        fragments.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_system_blocks
        """
        accumulated_text = ""

        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
            max_tokens=100,
            system="Answer with digits only, no words.",
            messages=[{"role": "user", "content": "What is 2+2?"}],
            stream=True,
        )

        for event in response:
            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                accumulated_text += event.delta.text

        assert len(accumulated_text) > 0
        assert "4" in accumulated_text

    # --- Content block index in streaming ---

    def test_streaming_content_block_indices(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """Streamed block indices start at 0 and every started block is stopped once.

        The gateway keeps its own index bookkeeping while remapping Bedrock's
        content-block indices, so ``content_block_delta`` frames may only refer to
        an index that was started, and the stop frames must mirror the start ones.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_adapters/_anthropic_message.py:_process_stream_events
        """
        indices: dict[str, list[int]] = {"start": [], "delta": [], "stop": []}

        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
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
        assert indices["stop"] == indices["start"], (
            "each started block must be stopped exactly once, in order"
        )
        assert set(indices["delta"]) <= set(indices["start"]), (
            "deltas may only target a started block"
        )

    # --- Long conversation ---

    def test_long_multi_turn_conversation(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """A five-turn history is preserved: the model recovers a fact from the first turn.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_messages
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
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
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """A completion that finishes on its own reports ``stop_reason`` ``end_turn``.

        Bedrock's ``end_turn`` is the only stop reason mapped to Anthropic's
        ``end_turn``; anything unmapped would also fall back to it, hence the
        companion tests pinning ``max_tokens``, ``stop_sequence`` and ``tool_use``.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_stop_reason
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Say OK."}],
        )

        assert response.stop_reason == "end_turn"
        assert response.usage.output_tokens > 0

    def test_stop_reason_max_tokens(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """Truncation at the token ceiling reports ``stop_reason`` ``max_tokens``.

        Bedrock also emits the non-standard ``incomplete`` stop reason, which the
        gateway folds into ``max_tokens``.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_stop_reason
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
            max_tokens=1,
            messages=[
                {
                    "role": "user",
                    "content": "Write a very long detailed essay about the universe.",
                }
            ],
        )

        assert response.stop_reason == "max_tokens"
        assert response.usage.output_tokens <= 5, (
            "generation must stop at the 1-token ceiling"
        )

    def test_stop_reason_tool_use(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """A turn that ends in a tool call reports ``stop_reason`` ``tool_use``.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_stop_reason
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
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        assert len(tool_use_blocks) >= 1, (
            "stop_reason tool_use requires at least one tool_use block"
        )
        assert tool_use_blocks[0].name == "lookup"

    # --- Model field in response ---

    def test_response_model_field(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_basic_model: str,
        use_official_api: bool,
    ) -> None:
        """The response ``model`` echoes the requested identifier verbatim.

        The gateway passes ``request.model`` straight into the response rather than
        the resolved Bedrock model or inference profile, so aliases come back
        exactly as sent.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_adapters/_anthropic_message.py:format_response
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Hi."}],
        )

        assert response.model is not None
        assert len(response.model) > 0
        if not use_official_api:
            # Our gateway must echo back the exact requested model name
            assert response.model == anthropic_chat_basic_model

    # --- Response ID format ---

    def test_response_id_format(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """The response id uses Anthropic's ``msg_`` prefix and is not a bare request id.

        The gateway builds it as ``msg_{request_id}``, which keeps the Anthropic
        prefix contract while remaining traceable in the request log.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/routes/anthropic_messages.py:create_message
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Hi."}],
        )

        assert response.id.startswith("msg_")
        assert len(response.id) > len("msg_"), "id must carry a request identifier"

    # --- Streaming final message ---

    def test_streaming_get_final_message(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """The stream carries everything needed to assemble the final message.

        ``message_start`` provides the envelope, ``message_delta`` the terminal
        ``stop_reason`` and the cumulative usage that Bedrock only reveals in its
        trailing metadata event.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStreamMetadataEvent.html
             stdapi/models/chat/_adapters/_anthropic_message.py:format_stream
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
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
        assert message_start.content == []
        assert message_start.id.startswith("msg_")
        assert has_content
        assert message_delta is not None
        assert message_delta.delta.stop_reason is not None
        assert message_delta.delta.stop_reason in {
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            "pause_turn",
            "refusal",
            "model_context_window_exceeded",
        }
        assert message_delta.usage.output_tokens > 0

    # --- Streaming get_final_text ---

    def test_streaming_get_final_text(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """Concatenated ``text_delta`` fragments reproduce the model's answer.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_delta
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
            max_tokens=100,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly the word TEAL and nothing else.",
                }
            ],
            stream=True,
        )

        final_text = ""
        for event in response:
            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                final_text += event.delta.text

        assert isinstance(final_text, str)
        assert len(final_text) > 0
        assert "teal" in final_text.lower()

    # --- Multiple user content blocks ---

    def test_multiple_text_blocks_in_user_message(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """Every text block of a user message is forwarded, not just the first.

        The question lives in the second block, so an answer that uses it proves
        both blocks reached the model as separate Bedrock content blocks.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_messages
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
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
        """``cache_control`` on a user text block inserts a Bedrock ``cachePoint``.

        A prompt below the model's cacheable minimum is silently not cached and no
        error is raised, so the usage counters must come back unset or zero rather
        than reporting a cache write.

        Ref: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CachePointBlock.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_build_cache_point
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
        assert response.usage.cache_creation_input_tokens in (None, 0), (
            "a prompt below the model minimum must not be cached"
        )
        assert response.usage.cache_read_input_tokens in (None, 0)

    def test_cache_control_on_system_prompt_block(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A cache-marked ``system`` block is still applied as an instruction.

        The gateway appends the ``cachePoint`` after the system text block, so
        marking a block for caching must not stop its content from reaching the
        model.

        Ref: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_system_blocks
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=100,
            system=[
                {
                    "type": "text",
                    "text": "Whatever the user writes, reply with exactly the word TEAL and nothing else.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": "Say hello."}],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert "teal" in response.content[0].text.lower()

    def test_cache_control_on_tool(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """``cache_control`` on a tool definition is accepted and tools stay callable.

        In Converse a cache point inside ``toolConfig.tools`` is a list element of
        its own, not a field on the tool, so a malformed translation would make
        Bedrock reject the request outright.

        Ref: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CachePointBlock.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_build_tool_config
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
        assert all(b.type in ("text", "tool_use") for b in response.content)
        assert all(
            b.name == "get_weather" for b in response.content if b.type == "tool_use"
        )
        assert response.usage.input_tokens > 0

    def test_cache_control_with_ttl(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """``cache_control`` with an explicit ``5m`` TTL is accepted.

        ``5m`` is Anthropic's default lifetime and the only other legal value is
        ``1h``; the gateway forwards the TTL onto the Bedrock cache point when the
        model supports it.

        Ref: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
             stdapi/types/anthropic_messages.py:CacheControlEphemeralParam
             stdapi/models/chat/_adapters/_anthropic_message.py:_build_cache_point
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
        assert response.usage.cache_creation_input_tokens in (None, 0), (
            "a prompt below the model minimum must not be cached"
        )

    def test_cache_control_streaming(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """``cache_control`` is accepted in streaming mode and the stream stays well formed.

        Ref: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
             https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_adapters/_anthropic_message.py:format_stream
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
        event_types: list[str] = []
        for event in response:
            event_types.append(event.type)
            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                final_text += event.delta.text

        assert len(final_text) > 0
        assert event_types[0] == "message_start"
        assert event_types[-1] == "message_stop"
        assert "message_delta" in event_types

    def test_automatic_cache_control(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A top-level ``cache_control`` places the cache breakpoint automatically.

        This request-level field is a gateway extension: instead of requiring
        per-block markers it inserts the cache point after the last cacheable
        block. Short prompts are still not cached, so only acceptance and the
        untouched conversation behavior are observable here.

        Ref: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
             stdapi/types/anthropic_messages.py:MessageCreateParams
             stdapi/models/chat/_default.py:_req_enable_prompt_caching
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
        answer = response.content[0].text.lower()
        assert any(hint in answer for hint in ("machine learning", "ml")), (
            f"the cached conversation prefix must stay visible: {answer!r}"
        )

    def test_automatic_cache_control_with_ttl(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A top-level ``cache_control`` accepts a ``ttl`` alongside ``type``.

        Ref: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
             stdapi/types/anthropic_messages.py:CacheControlEphemeralParam
             stdapi/models/chat/_default.py:_req_enable_prompt_caching
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
        assert response.usage.cache_creation_input_tokens in (None, 0), (
            "a prompt below the model minimum must not be cached"
        )

    def test_output_config_json_schema(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """``output_config.format`` with a JSON schema constrains the reply to that schema.

        Structured output is Anthropic-only on Bedrock (Converse ``outputConfig``),
        so the answer must parse as JSON and match the requested fields exactly.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
             stdapi/types/anthropic_messages.py:JSONOutputFormatParam
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
        assert set(result) == {"name", "age"}, (
            f"additionalProperties: false must be honored, got {sorted(result)}"
        )
        assert response.stop_reason == "end_turn"

    # --- Extended thinking multi-turn ---

    def test_extended_thinking_multi_turn(
        self, anthropic_client: Anthropic, anthropic_chat_reasoning_model: str
    ) -> None:
        """Thinking blocks can be replayed in the next turn, signature included.

        Bedrock requires ``reasoningContent.signature`` to come back byte-identical
        with all prior messages unchanged, so a round trip that drops or rewrites
        the signature makes the second call fail.

        Ref: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlock.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_thinking_to_bedrock
        """
        first = anthropic_client.messages.create(
            model=anthropic_chat_reasoning_model,
            max_tokens=1500,
            messages=[{"role": "user", "content": "What is 15 * 27?"}],
            thinking={"type": "enabled", "budget_tokens": 1024},
        )

        assert first.type == "message"
        assert len(first.content) >= 1
        first_thinking = [b for b in first.content if b.type == "thinking"]
        assert len(first_thinking) >= 1
        assert first_thinking[0].signature, (
            "a thinking block must carry the signature required for replay"
        )

        # Send the full assistant content (including thinking blocks) back
        second = anthropic_client.messages.create(
            model=anthropic_chat_reasoning_model,
            max_tokens=1500,
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
        assert "810" in " ".join(b.text for b in text_blocks), (
            "the replayed turn must keep the first result (405) in context"
        )

    def test_extended_thinking_streaming_content(
        self, anthropic_client: Anthropic, anthropic_chat_reasoning_model: str
    ) -> None:
        """A streamed thinking block ends with a ``signature_delta`` before its stop frame.

        Bedrock sends the reasoning signature as its own delta at the end of the
        reasoning block; the gateway forwards it as Anthropic's ``signature_delta``,
        which is what makes the block replayable in a later turn.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlock.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_delta
        """
        events = list(
            anthropic_client.messages.create(
                model=anthropic_chat_reasoning_model,
                max_tokens=1500,
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
        deltas = [e.delta for e in events if e.type == "content_block_delta"]
        delta_types = {d.type for d in deltas}
        signatures = [d.signature for d in deltas if d.type == "signature_delta"]

        assert "thinking_delta" in delta_types
        assert "text_delta" in delta_types
        assert "signature_delta" in delta_types
        assert all(signatures), "signature_delta must carry a non-empty signature"

    def test_tool_choice_any(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """``tool_choice`` ``any`` forces a tool call even on a conversational prompt.

        ``any`` becomes Converse ``toolChoice: {"any": {}}``; the greeting would
        otherwise be answered in plain text, so a ``tool_use`` turn proves the
        constraint was applied.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools#forcing-tool-use
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_tool_choice
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
        assert all(b.name == "greet" for b in tool_blocks)
        assert all(isinstance(b.input, dict) for b in tool_blocks)

    def test_streaming_tool_calling_events(
        self, anthropic_client: Anthropic, anthropic_chat_vision_model: str
    ) -> None:
        """Streamed tool arguments arrive as ``input_json_delta`` fragments forming valid JSON.

        Anthropic sends the arguments as partial JSON strings while the final
        ``tool_use.input`` is always an object, so the concatenation of every
        fragment must parse back into a JSON object.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_delta
        """
        events = list(
            anthropic_client.messages.create(  # type: ignore[call-overload]
                model=anthropic_chat_vision_model,
                max_tokens=300,
                messages=[{"role": "user", "content": "What's the weather in Paris?"}],
                tools=[_WEATHER_TOOL],
                tool_choice={"type": "any"},
                stream=True,
            )
        )

        event_types = [e.type for e in events]
        assert "content_block_start" in event_types
        assert "content_block_stop" in event_types

        # Verify input_json_delta is present
        deltas = [e.delta for e in events if e.type == "content_block_delta"]
        delta_types = {d.type for d in deltas}
        partial_json = "".join(
            d.partial_json for d in deltas if d.type == "input_json_delta"
        )

        assert "input_json_delta" in delta_types
        assert isinstance(_json.loads(partial_json), dict), (
            f"streamed tool input is not a JSON object: {partial_json!r}"
        )
        started_blocks = [
            e.content_block for e in events if e.type == "content_block_start"
        ]
        tool_starts = [block for block in started_blocks if block.type == "tool_use"]
        assert len(tool_starts) >= 1
        assert tool_starts[0].name == "get_weather"

    def test_stop_reason_stop_sequence(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Hitting a stop sequence reports ``stop_reason`` ``stop_sequence``.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_stop_reason
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
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """A supported ``anthropic-beta`` header leaves the request working.

        ``context-management-2025-06-27`` is on the gateway's Bedrock beta
        allowlist. On a Claude model the header is turned into the
        ``anthropic_beta`` body field; on the Nova model used here there is no
        passthrough mapping, so the header is simply ignored and generation must
        proceed normally.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel
             stdapi/config.py:_Settings
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
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
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """An ``anthropic-beta`` flag outside the allowlist does not break the request.

        ``claude-code-20250219`` is not a Bedrock-supported beta: rather than
        letting Bedrock reject the call, the gateway drops unknown flags (logging a
        warning) and serves the request normally.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel
             stdapi/config.py:_Settings
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
            max_tokens=100,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            extra_headers={"anthropic-beta": "claude-code-20250219"},
        )

        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert response.content[0].text.strip()
        assert response.usage.output_tokens > 0

    def test_document_plain_text(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A plain-text ``document`` block is readable by the model.

        Bedrock has no text source type, so the gateway encodes the text as a
        ``txt``-format ``DocumentBlock``; the answer must come from the document
        content.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_document_to_bedrock
        """
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
        """``citations.enabled`` on a text document either works or is rejected outright.

        Bedrock Converse may restrict citations to some document formats; when it
        rejects the combination the test xfails, otherwise the document must still
        be answered from.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_document_to_bedrock
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
        answer = " ".join(b.text for b in text_blocks).lower()
        assert "guido" in answer or "1991" in answer, (
            f"answer did not use the cited document: {answer!r}"
        )

    def test_document_base64_pdf(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        sample_pdf_file: bytes,
    ) -> None:
        """A base64 PDF ``document`` block is decoded and its text reaches the model.

        The fixture PDF contains the single string "Hello World"; the gateway
        forwards the raw bytes as a Bedrock ``pdf`` document and the model reads
        them.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_document_to_bedrock
        """
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
        text_blocks = [b for b in response.content if b.type == "text"]
        assert len(text_blocks) >= 1
        assert "hello" in " ".join(b.text for b in text_blocks).lower(), (
            "the PDF text was not readable by the model"
        )

    def test_document_content_block_source(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A ``document`` block whose source is a content-block list is flattened to text.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_document_to_bedrock
        """
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
        text_blocks = [b for b in response.content if b.type == "text"]
        assert len(text_blocks) >= 1
        assert "299" in " ".join(b.text for b in text_blocks), (
            "the answer must come from the document content"
        )

    def test_document_with_context(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A ``document`` block's ``context`` field is forwarded alongside its content.

        ``context`` maps to the Bedrock ``DocumentBlock.context`` field, so an
        unsupported value would be rejected rather than ignored.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_document_to_bedrock
        """
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
        assert "1.5" in " ".join(b.text for b in text_blocks), (
            "the answer must come from the document content"
        )

    # --- Tool choice none ---

    def test_tool_choice_none_keeps_tools_declared_but_unused(
        self, anthropic_client: Anthropic, anthropic_chat_basic_model: str
    ) -> None:
        """``tool_choice`` ``none`` answers normally without calling a tool.

        Anthropic documents ``none`` as the way to keep tools declared but unused.
        Converse has no equivalent ``toolChoice``, so the gateway drops the whole
        tool config instead; the model layer restores a permissive one when the
        history still carries ``toolUse``/``toolResult`` blocks, which is what
        makes dropping it safe. The behavior is shared with the official API, so
        no lane is skipped.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools#forcing-tool-use
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_build_tool_config
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_basic_model,
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

        assert not [block for block in response.content if block.type == "tool_use"], (
            "tool_choice 'none' must not produce a tool call"
        )
        assert response.stop_reason != "tool_use"

    # --- Cache creation input tokens ---

    def test_cache_creation_input_tokens_in_usage(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """``usage`` exposes the cache counters, unset when no cache point is requested.

        Bedrock only reports cache tokens when a ``cachePoint`` was sent, so an
        uncached request must leave both counters absent or zero while the plain
        input/output counters are populated.

        Ref: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
             stdapi/models/chat/_adapters/_anthropic_message.py:format_response
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hi."}],
        )
        assert response.usage is not None
        cache_creation = response.usage.cache_creation_input_tokens
        assert cache_creation is None or isinstance(cache_creation, int)
        assert cache_creation in (None, 0), (
            "no cache point was requested, so nothing may be written to the cache"
        )
        assert response.usage.cache_read_input_tokens in (None, 0)
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    # --- Search result block input ---

    def test_search_result_block_input(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A ``search_result`` input block is readable by the model.

        The block maps onto Bedrock's ``searchResult`` content block, keeping
        source, title and the text items.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_search_result_to_bedrock
        """
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
        assert "14" in " ".join(b.text for b in text_blocks), (
            "the answer must come from the supplied search result"
        )

    # --- Streaming with document ---

    def test_streaming_with_document(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A document input works through the SDK's streaming helper.

        ``messages.stream`` assembles the events itself, so this covers the
        document path and the SSE text stream together.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_adapters/_anthropic_message.py:format_stream
        """
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
        """A stop sequence hit while streaming surfaces on the ``message_delta`` frame.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_adapters/_anthropic_message.py:_make_message_delta_event
        """
        stop_reason = None
        streamed_text = ""

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
            elif event.type == "content_block_delta" and hasattr(event.delta, "text"):
                streamed_text += event.delta.text

        assert stop_reason == "stop_sequence"
        assert streamed_text, "content generated before the stop sequence must stream"

    # --- Error format validation ---

    def test_invalid_model_error_format(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Errors use Anthropic's envelope: ``type``, nested ``error`` and ``request_id``.

        The gateway rebuilds that envelope for every Anthropic-tagged route and
        derives ``error.type`` from the HTTP status, so an unknown model yields a
        404 ``not_found_error``.

        Ref: https://platform.claude.com/docs/en/api/errors
             stdapi/api_providers/anthropic.py:_format_error
             stdapi/api_providers/anthropic.py:_STATUS
        """
        if use_official_api:
            pytest.skip("Error format varies on official API")

        with pytest.raises(AnthropicError) as exc_info:
            anthropic_client.messages.create(
                model="nonexistent-model-xyz",
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}],
            )
        error = exc_info.value
        assert isinstance(error, APIStatusError)
        assert error.status_code == 404
        assert hasattr(error, "body")
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "error"
        assert "error" in body
        assert "type" in body["error"]
        assert "message" in body["error"]
        assert body["error"]["type"] == "not_found_error"
        assert "nonexistent-model-xyz" in body["error"]["message"]
        assert "request_id" in body

    # --- Negative temperature ---

    def test_invalid_negative_temperature_error(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_basic_model: str,
        use_official_api: bool,
    ) -> None:
        """A negative ``temperature`` is rejected with HTTP 400.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://platform.claude.com/docs/en/api/errors
             stdapi/types/anthropic_messages.py:MessageCreateParams
        """
        with pytest.raises(BadRequestError) as excinfo:
            anthropic_client.messages.create(
                model=anthropic_chat_basic_model,
                max_tokens=100,
                messages=[{"role": "user", "content": "Hello"}],
                temperature=-0.5,
            )

        assert excinfo.value.status_code == 400
        assert "temperature" in str(excinfo.value).lower()
        if not use_official_api:
            assert excinfo.value.type == "invalid_request_error"

    # --- Thinking disabled explicitly ---

    def test_thinking_disabled_explicitly(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """``thinking`` disabled suppresses thinking blocks and still answers.

        The gateway sends an explicitly disabled reasoning configuration to models
        that accept one, so the response must contain plain text only.

        Ref: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
             stdapi/types/anthropic_messages.py:ThinkingConfigDisabledParam
             stdapi/models/chat/_adapters/_anthropic_message.py:extract_reasoning
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
        assert "4" in " ".join(b.text for b in text_blocks)

    # --- Model alias resolution ---

    def test_model_alias_resolution(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """An Anthropic-style model alias resolves to its Bedrock model and is echoed back.

        ``claude-haiku-4-5-20251001`` carries neither the ``anthropic.`` prefix nor
        a Bedrock version suffix; an unresolvable id would 404, and the response
        echoes the alias exactly as requested rather than the resolved id.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/__init__.py:validate_model
             stdapi/models/chat/_adapters/_anthropic_message.py:format_response
        """
        if use_official_api:
            pytest.skip("Alias resolution is a local gateway feature")

        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{"role": "user", "content": "Say hi."}],
        )

        assert response.type == "message"
        assert len(response.content) >= 1
        assert response.content[0].type == "text"
        assert response.content[0].text.strip()
        assert response.model == "claude-haiku-4-5-20251001"
        assert response.usage.output_tokens > 0


class TestAnthropicCountTokens:
    """POST /v1/messages/count_tokens: the ``{input_tokens}`` count for a request body.

    Anthropic's Token Counting API does not exist on legacy Bedrock, so the
    gateway serves it from the Bedrock Runtime ``CountTokens`` operation (or the
    Mantle count_tokens path), building the same Converse input that
    ``create_message`` would send. The ``AnthropicBedrock`` client refuses the
    route client-side, hence the class-wide xfail on that lane.

    Ref: https://platform.claude.com/docs/en/api/messages/count_tokens
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html
         stdapi/routes/anthropic_messages.py:count_tokens
         stdapi/models/chat/_adapters/_anthropic_message.py:count_tokens_via_bedrock
    """

    @pytest.fixture(autouse=True)
    def _xfail_on_bedrock(self, is_bedrock_direct: bool) -> None:
        """Xfail the whole class when the client talks to Bedrock directly.

        ``AnthropicBedrock`` raises client-side for ``/v1/messages/count_tokens``,
        so the route is unreachable rather than broken.
        """
        if is_bedrock_direct:
            pytest.xfail("Token counting is not supported in Bedrock yet")

    def test_count_tokens_basic(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """A short message is counted as a small positive ``input_tokens`` value.

        The response body has a single field, so the count itself is the only
        assertable behavior; the upper bound catches a count that reflects
        something other than this prompt.

        Ref: https://platform.claude.com/docs/en/api/messages/count_tokens
             stdapi/types/anthropic_messages.py:MessageTokensCount
        """
        response = anthropic_client.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )

        assert response.input_tokens > 0
        assert response.input_tokens < 100, (
            f"a six-word prompt cannot cost {response.input_tokens} tokens"
        )

    def test_count_tokens_with_system_prompt(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """A ``system`` prompt raises the counted ``input_tokens``.

        The count must account for every input, so the same messages counted with
        and without a long system prompt cannot come out equal.

        Ref: https://platform.claude.com/docs/en/api/messages/count_tokens
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_system_blocks
        """
        response_without = anthropic_client.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=[{"role": "user", "content": "Hello"}],
        )

        response_with = anthropic_client.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=[{"role": "user", "content": "Hello"}],
            system="You are a very detailed and verbose assistant that always provides comprehensive answers.",
        )

        assert response_without.input_tokens > 0
        assert response_with.input_tokens > response_without.input_tokens

    def test_count_tokens_with_tools(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """Tool definitions raise the counted ``input_tokens``.

        ``tools`` are part of the counted input: the gateway builds the Converse
        ``toolConfig`` for the count exactly as it would for generation.

        Ref: https://platform.claude.com/docs/en/api/messages/count_tokens
             stdapi/models/chat/_adapters/_anthropic_message.py:_build_tool_config
        """
        response_without = anthropic_client.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=[{"role": "user", "content": "What is the weather?"}],
        )

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

        assert response_without.input_tokens > 0
        assert response_with.input_tokens > response_without.input_tokens

    def test_count_tokens_multi_turn(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """Every turn of a conversation is counted, not just the last one.

        Ref: https://platform.claude.com/docs/en/api/messages/count_tokens
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_messages
        """
        response_single = anthropic_client.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=[{"role": "user", "content": "Hello"}],
        )

        response_multi = anthropic_client.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there! How can I help you?"},
                {"role": "user", "content": "Tell me about Python programming."},
            ],
        )

        assert response_single.input_tokens > 0
        assert response_multi.input_tokens > response_single.input_tokens

    def test_count_tokens_longer_content_more_tokens(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """A longer message costs more tokens than a short one.

        Ref: https://platform.claude.com/docs/en/api/messages/count_tokens
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html
        """
        response_short = anthropic_client.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=[{"role": "user", "content": "Hi"}],
        )

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

        assert response_short.input_tokens > 0
        assert response_long.input_tokens > response_short.input_tokens

    def test_count_tokens_invalid_model(self, anthropic_client: Anthropic) -> None:
        """An unknown model on count_tokens returns 404 ``not_found_error``.

        Model validation happens before any counting, so the route answers with the
        same ``UnsupportedModelError`` envelope as ``create_message``.

        Ref: https://platform.claude.com/docs/en/api/errors
             stdapi/routes/anthropic_messages.py:count_tokens
             stdapi/api_errors.py:UnsupportedModelError
        """
        with pytest.raises(NotFoundError) as excinfo:
            anthropic_client.messages.count_tokens(
                model="nonexistent-model-xyz",
                messages=[{"role": "user", "content": "Hello"}],
            )
        assert excinfo.value.status_code == 404
        assert excinfo.value.type == "not_found_error"
        assert "nonexistent-model-xyz" in str(excinfo.value)

    def test_count_tokens_content_blocks(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """A block-list ``content`` is accepted by count_tokens and counted.

        Ref: https://platform.claude.com/docs/en/api/messages/count_tokens
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_messages
        """
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

        assert response.input_tokens > 0
        assert response.input_tokens < 100, (
            f"a one-sentence prompt cannot cost {response.input_tokens} tokens"
        )

    def test_count_tokens_web_search_tool_rejected(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """A ``web_search`` server tool is rejected with HTTP 400 on count_tokens.

        The official Anthropic API's own ``count_tokens`` endpoint does not
        support server tools either, regardless of backend; the gateway mirrors
        that contract here instead of silently counting a request that could
        never be generated.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html
             stdapi/models/chat/_adapters/_anthropic_message.py:count_tokens_via_bedrock
        """
        with pytest.raises(BadRequestError) as excinfo:
            anthropic_client.messages.count_tokens(
                model=anthropic_count_tokens_model,
                messages=[{"role": "user", "content": "Hello"}],
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
            )
        assert excinfo.value.status_code == 400
        assert excinfo.value.type == "invalid_request_error"

    def test_count_tokens_with_system_blocks(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """A ``system`` block list is accepted by count_tokens and counted.

        Ref: https://platform.claude.com/docs/en/api/messages/count_tokens
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_system_blocks
        """
        response = anthropic_client.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=[{"role": "user", "content": "Hello"}],
            system=[{"type": "text", "text": "You are a helpful assistant."}],
        )

        assert response.input_tokens > 0
        assert response.input_tokens < 100, (
            f"a one-sentence system prompt cannot cost {response.input_tokens} tokens"
        )

    @pytest.mark.gateway("system-role messages in `messages` are a stdapi extension")
    def test_count_tokens_system_role_in_messages(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """A system-role message inside ``messages`` is counted, not dropped.

        The count path runs the same system-message hoisting as generation, so the
        extracted directive must show up in ``input_tokens``.

        Ref: https://platform.claude.com/docs/en/api/messages/count_tokens
             stdapi/models/chat/_adapters/_anthropic_message.py:_prepare_messages_and_system
        """
        response_without = anthropic_client.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=[{"role": "user", "content": "Hello"}],
        )

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

        assert response_without.input_tokens > 0
        assert response_with.input_tokens > response_without.input_tokens

    @pytest.mark.gateway("system-role messages in `messages` are a stdapi extension")
    def test_count_tokens_system_role_equivalent_to_system_field(
        self, anthropic_client: Anthropic, anthropic_count_tokens_model: str
    ) -> None:
        """The two ways of giving a system prompt produce the same token count.

        A system-role message is folded into the ``system`` blocks, so counting it
        must be indistinguishable from passing the same text in ``system``.

        Ref: https://platform.claude.com/docs/en/api/messages/count_tokens
             stdapi/models/chat/_adapters/_anthropic_message.py:_merge_system_content
        """
        system_text = "You are a helpful assistant."
        response_field = anthropic_client.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=[{"role": "user", "content": "Hello"}],
            system=system_text,
        )

        response_role = anthropic_client.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": "Hello"},
            ],
        )

        assert response_field.input_tokens > 0
        assert response_role.input_tokens == response_field.input_tokens


class TestAnthropicCountTokensDispatch:
    """Offline unit tests for count_tokens dispatch to the classic vs Mantle counter.

    Registers fake models directly in the model registry so the real
    `serves_via_mantle` dispatch condition runs unmocked; only the two
    counting functions are replaced with recorders.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html
         stdapi/routes/anthropic_messages.py:count_tokens
    """

    pytestmark = pytest.mark.local

    @pytest.fixture
    def runtime_model(self, monkeypatch: pytest.MonkeyPatch) -> ModelDetails:
        """Register a fake Bedrock Runtime model in the model registry."""
        return _register_test_model(
            monkeypatch, "test.count-tokens-runtime-model", "Runtime Count Tokens Test"
        )

    @pytest.fixture
    def mantle_model(self, monkeypatch: pytest.MonkeyPatch) -> ModelDetails:
        """Register a fake Bedrock Mantle model in the model registry."""
        return _register_test_model(
            monkeypatch,
            "test.count-tokens-mantle-model",
            "Mantle Count Tokens Test",
            service="AWS Bedrock Mantle",
        )

    def test_runtime_model_uses_classic_bedrock_counter(
        self,
        anthropic_app_client: TestClient,
        runtime_model: ModelDetails,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-Mantle model routes count_tokens to the classic Bedrock counter.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html
             stdapi/models/chat/_adapters/_anthropic_message.py:count_tokens_via_bedrock
        """
        classic = AsyncMock(return_value=7)
        mantle = AsyncMock(return_value=99)
        monkeypatch.setattr(anthropic_messages, "count_tokens_via_bedrock", classic)
        monkeypatch.setattr(anthropic_messages, "_count_tokens_via_mantle", mantle)

        response = anthropic_app_client.post(
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
        anthropic_app_client: TestClient,
        mantle_model: ModelDetails,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A Mantle-served model routes count_tokens to the Mantle counter.

        Mantle-only models are unreachable through Bedrock Runtime ``CountTokens``,
        so the count is proxied to the Mantle Anthropic count_tokens path instead.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html
             stdapi/routes/anthropic_messages.py:_count_tokens_via_mantle
        """
        classic = AsyncMock(return_value=7)
        mantle = AsyncMock(return_value=99)
        monkeypatch.setattr(anthropic_messages, "count_tokens_via_bedrock", classic)
        monkeypatch.setattr(anthropic_messages, "_count_tokens_via_mantle", mantle)

        response = anthropic_app_client.post(
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
    """Offline tests pinning the Anthropic-parity 404 for unknown models.

    Ref: https://platform.claude.com/docs/en/api/errors
         stdapi/api_providers/anthropic.py:_format_error
    """

    pytestmark = pytest.mark.local

    @pytest.fixture
    def _offline_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Seed one model and disable the AWS registry refresh on cache miss."""
        _register_test_model(
            monkeypatch, "test.some-registered-model", "Registered Test Model"
        )
        monkeypatch.setattr(
            _models_mod, "initialize_bedrock_models", AsyncMock(return_value=None)
        )

    @pytest.mark.usefixtures("_offline_registry")
    def test_messages_unknown_model_returns_404(
        self, anthropic_app_client: TestClient
    ) -> None:
        """An unknown model returns 404 not_found_error like the official API.

        Ref: https://platform.claude.com/docs/en/api/errors
             stdapi/api_errors.py:UnsupportedModelError
        """
        response = anthropic_app_client.post(
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
        assert "nonexistent-model-xyz" in body["error"]["message"]
        assert "request_id" in body
        assert response.headers["request-id"] == body["request_id"]

    @pytest.mark.usefixtures("_offline_registry")
    def test_count_tokens_unknown_model_returns_404(
        self, anthropic_app_client: TestClient
    ) -> None:
        """count_tokens with an unknown model returns 404 not_found_error.

        Ref: https://platform.claude.com/docs/en/api/errors
             stdapi/routes/anthropic_messages.py:count_tokens
        """
        response = anthropic_app_client.post(
            "/anthropic/v1/messages/count_tokens",
            json={
                "model": "nonexistent-model-xyz",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 404, response.text
        body = response.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "not_found_error"
        assert "nonexistent-model-xyz" in body["error"]["message"]


class TestAnthropicMessagesMaxTokensOptional:
    """Offline unit test pinning that ``max_tokens`` stays optional on /v1/messages.

    Intentional divergence from the official Anthropic API (which requires
    ``max_tokens``): when omitted here, the underlying model's default output
    length applies. Validation and dispatch are exercised against an app
    instance without the AWS-touching lifespan, with the model call stubbed.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/types/anthropic_messages.py:MessageCreateParams
         stdapi/routes/anthropic_messages.py:create_message
    """

    pytestmark = pytest.mark.local

    def test_missing_max_tokens_is_accepted(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A /v1/messages request without ``max_tokens`` is accepted and forwards it unset.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/types/anthropic_messages.py:MessageCreateParams
        """
        details = _register_test_model(
            monkeypatch, "test.max-tokens-optional-model", "Max Tokens Optional Test"
        )
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

        response = anthropic_app_client.post(
            "/anthropic/v1/messages",
            json={"model": details.id, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200, response.text
        assert response.json()["content"][0]["text"] == "hi"
        assert fake_model.create_message.await_args is not None
        forwarded = fake_model.create_message.await_args.args[0]
        assert forwarded.max_tokens is None, (
            "an omitted max_tokens must stay unset rather than get a default"
        )


class TestCountTokensViaMantleSingleRegion:
    """Offline unit tests for the ``single_region`` flag passed to Mantle's retry.

    ``route_and_execute`` only retries across regions when the region router is
    enabled and there is more than one candidate; otherwise it calls the first
    candidate exactly once, so ``_count_tokens_via_mantle``'s own in-region retry
    (gated by ``single_region``) must cover that case instead.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html
         stdapi/routes/anthropic_messages.py:_count_tokens_via_mantle
    """

    pytestmark = pytest.mark.local

    @staticmethod
    async def _run(monkeypatch: pytest.MonkeyPatch, *, region_router: object) -> bool:
        """Invoke ``_count_tokens_via_mantle`` and return the ``single_region`` it used."""
        monkeypatch.setattr(anthropic_messages, "REGION_ROUTER", region_router)
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_mantle_regions", ["us-east-1", "eu-west-1"]
        )
        monkeypatch.setattr(
            anthropic_messages, "messages_payload", AsyncMock(return_value={})
        )
        monkeypatch.setattr(
            anthropic_messages, "set_effective_region", lambda *_a, **_k: None
        )

        captured: dict[str, object] = {}

        async def _fake_invoke(
            _region: object,
            _path: object,
            _payload: object,
            *,
            single_region: bool,
            **_kw: object,
        ) -> dict[str, int]:
            captured["single_region"] = single_region
            return {"input_tokens": 5}

        async def _fake_route_and_execute(
            _model_id: object, regions: list[object], fn: object
        ) -> object:
            return await fn(regions[0])  # type: ignore[operator]

        monkeypatch.setattr(anthropic_messages, "invoke", _fake_invoke)
        monkeypatch.setattr(
            anthropic_messages, "route_and_execute", _fake_route_and_execute
        )

        request = MessageCountTokensParams(
            model="test.fake-mantle-model",
            messages=[MessageParam(role="user", content="hi")],
        )
        tokens = await anthropic_messages._count_tokens_via_mantle(  # noqa: SLF001
            request, "test.fake-mantle-model"
        )
        assert tokens == 5
        return bool(captured["single_region"])

    async def test_single_region_true_with_region_router_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With region routing disabled and several regions, retries stay in-region.

        Ref: stdapi/routes/anthropic_messages.py:_count_tokens_via_mantle
        """
        assert await self._run(monkeypatch, region_router=None) is True

    async def test_single_region_false_with_region_router_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With region routing enabled, ``route_and_execute`` handles the retry.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html
             stdapi/routes/anthropic_messages.py:_count_tokens_via_mantle
        """
        assert await self._run(monkeypatch, region_router=object()) is False


class TestTranslateRequestParameters:
    """Offline unit tests for the request parameters ``translate_request`` maps to Bedrock.

    Converse's ``inferenceConfig`` only carries temperature / topP / maxTokens /
    stopSequences, so every other generation knob — including ``top_k`` and any
    caller-supplied model extra — has to travel in
    ``additionalModelRequestFields``.

    Ref: https://platform.claude.com/docs/en/api/messages
         https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
         stdapi/models/chat/_adapters/_anthropic_message.py:translate_request
    """

    pytestmark = pytest.mark.local

    @staticmethod
    async def _translate(
        **fields: object,
    ) -> tuple[dict[str, object], dict[str, object], str | None]:
        """Translate a minimal request and return (inference config, extras, tier)."""
        request = MessageCreateParams.model_validate(
            {
                "model": "test.translate-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                **fields,
            }
        )
        (
            _messages,
            _system,
            inference_config,
            additional_request_fields,
            _tool_config,
            service_tier,
            *_rest,
        ) = await translate_request(
            request,
            "test.translate-model",
            prompt_caching_supported=False,
            prompt_caching_tool_supported=False,
        )
        return dict(inference_config), dict(additional_request_fields), service_tier

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [
            ("standard_only", "default"),
            ("priority", "priority"),
            ("flex", "flex"),
            ("reserved", "reserved"),
        ],
    )
    async def test_service_tier_maps_to_a_bedrock_tier(
        self, requested: str, expected: str
    ) -> None:
        """Each accepted ``service_tier`` resolves to its Bedrock service tier.

        Anthropic's baseline value is ``standard_only`` while Converse spells the
        same tier ``default``; ``priority`` / ``flex`` / ``reserved`` are
        Bedrock-only extensions the gateway forwards unchanged.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_SERVICES_TIERS
        """
        _config, _extras, tier = await self._translate(service_tier=requested)
        assert tier == expected

    async def test_service_tier_auto_selects_no_bedrock_tier(self) -> None:
        """``auto`` is deliberately unmapped so Bedrock applies the account default.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_adapters/_anthropic_message.py:_SERVICES_TIERS
        """
        _config, _extras, tier = await self._translate(service_tier="auto")
        assert tier is None

    async def test_top_k_goes_to_additional_model_request_fields(self) -> None:
        """``top_k`` is forwarded as a model extra, not as an inference-config field.

        Converse's ``inferenceConfig`` has no ``top_k``, so putting it there would
        make Bedrock reject the request.

        Ref: https://platform.claude.com/docs/en/api/messages
             https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
             stdapi/models/chat/_adapters/_anthropic_message.py:translate_request
        """
        config, extras, _tier = await self._translate(top_k=40)
        assert extras["top_k"] == 40
        assert "top_k" not in config
        assert config["maxTokens"] == 16

    async def test_unknown_body_field_reaches_additional_model_request_fields(
        self,
    ) -> None:
        """An unmodelled body field is forwarded verbatim as a model-specific extra.

        Accepting extra keys is a documented gateway extension: they are collected
        in ``model_extra`` and merged into ``additionalModelRequestFields`` so a
        caller can reach a parameter the Anthropic schema does not model.

        Ref: stdapi/types/anthropic_messages.py:MessageCreateParams
             stdapi/aws_bedrock.py:set_inference_configuration
        """
        config, extras, _tier = await self._translate(vendor_specific_knob=7)
        assert extras["vendor_specific_knob"] == 7
        assert "vendor_specific_knob" not in config

    @pytest.mark.parametrize("key", ["model_id", "additional_request_fields"])
    async def test_reserved_extra_is_rejected(self, key: str) -> None:
        """A body key named like a ``set_inference_configuration`` argument is a 400.

        Extras are splatted into that function, so one spelled like its own
        arguments would bind twice and raise ``TypeError``, which no handler
        maps: the caller would get a 500 for a body the schema accepts.

        Ref: stdapi/models/chat/_adapters/_common.py:inference_extras
             stdapi/aws_bedrock.py:set_inference_configuration
        """
        with pytest.raises(ApiError) as exc_info:
            await self._translate(**{key: "x"})
        assert exc_info.value.status == 400
        assert key in str(exc_info.value)

    def test_camelcase_aliases_are_accepted(self) -> None:
        """``maxTokens`` / ``topP`` / ``stopSequences`` validate as their snake_case fields.

        These aliases are a gateway extension; if one were dropped the camelCase
        spelling would land in ``model_extra`` and be forwarded to Bedrock as an
        unknown extra instead of configuring generation.

        Ref: stdapi/types/anthropic_messages.py:MessageCreateParams
        """
        request = MessageCreateParams.model_validate(
            {
                "model": "test.translate-model",
                "messages": [{"role": "user", "content": "hi"}],
                "maxTokens": 10,
                "topP": 0.5,
                "stopSequences": ["x"],
            }
        )
        assert request.max_tokens == 10
        assert request.top_p == 0.5
        assert request.stop_sequences == ["x"]
        assert not request.model_extra, (
            f"an alias leaked into model_extra: {request.model_extra}"
        )

    async def test_inference_geo_and_container_are_accepted_and_unused(self) -> None:
        """``inference_geo`` and ``container`` validate but reach no Converse field.

        Both are declared so an SDK client sending them is not rejected with a 422;
        the Converse path has nowhere to put them, and forwarding them as model
        extras would make Bedrock reject the request.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/types/anthropic_messages.py:MessageCreateParams
        """
        request = MessageCreateParams.model_validate(
            {
                "model": "test.translate-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "inference_geo": "us",
                "container": "container_123",
            }
        )
        assert request.inference_geo == "us"
        assert request.container == "container_123"
        config, extras, _tier = await self._translate(
            inference_geo="us", container="container_123"
        )
        assert "inference_geo" not in extras
        assert "container" not in extras
        assert "inference_geo" not in config
        assert "container" not in config


class TestCountTokensViaMantlePayload:
    """Offline unit tests for the payload ``_count_tokens_via_mantle`` sends.

    ``messages_payload`` injects the generation-only ``max_tokens`` default that
    the Messages API needs, so the count_tokens caller has to strip it again: the
    count_tokens body is the Messages body *minus* ``max_tokens``.

    Ref: https://platform.claude.com/docs/en/api/messages/count_tokens
         https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html
         stdapi/routes/anthropic_messages.py:_count_tokens_via_mantle
    """

    pytestmark = pytest.mark.local

    @staticmethod
    async def _capture(
        monkeypatch: pytest.MonkeyPatch, **fields: object
    ) -> dict[str, object]:
        """Run the Mantle counter with ``invoke`` stubbed and return what it received."""
        captured: dict[str, object] = {}

        async def _fake_invoke(
            region: object, path: str, payload: dict[str, object], **kwargs: object
        ) -> dict[str, int]:
            captured.update(
                region=region, path=path, payload=payload, headers=kwargs.get("headers")
            )
            return {"input_tokens": 3}

        async def _fake_route_and_execute(
            _model_id: object, regions: list[object], fn: object
        ) -> object:
            return await fn(regions[0])  # type: ignore[operator]

        monkeypatch.setattr(anthropic_messages, "invoke", _fake_invoke)
        monkeypatch.setattr(
            anthropic_messages, "route_and_execute", _fake_route_and_execute
        )
        monkeypatch.setattr(
            anthropic_messages, "set_effective_region", lambda *_a, **_k: None
        )
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_regions", ["us-east-1"])

        request = MessageCountTokensParams.model_validate(
            {
                "model": "test.fake-mantle-model",
                "messages": [{"role": "user", "content": "hi"}],
                **fields,
            }
        )
        tokens = await anthropic_messages._count_tokens_via_mantle(  # noqa: SLF001
            request, "test.fake-mantle-model"
        )
        assert tokens == 3
        return captured

    async def test_payload_drops_max_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The forwarded body carries no ``max_tokens``, which count_tokens does not accept.

        ``messages_payload`` defaults ``max_tokens`` when unset, so the pop is
        load-bearing even for a request that never mentioned the field.
        """
        captured = await self._capture(monkeypatch)
        payload = captured["payload"]
        assert isinstance(payload, dict)
        assert "max_tokens" not in payload
        assert payload["model"] == "test.fake-mantle-model"
        assert payload["messages"] == [{"role": "user", "content": "hi"}]

    async def test_payload_targets_the_count_tokens_path_with_messages_headers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The call goes to the Messages count_tokens path with the Messages Mantle headers.

        Mantle serves the Anthropic-native surface under its own path prefix, and
        the ``anthropic-version`` header replaces the body field the Bedrock
        InvokeModel surface expects.
        """
        captured = await self._capture(monkeypatch)
        assert isinstance(captured["path"], str)
        assert captured["path"].endswith("/messages/count_tokens")
        headers = captured["headers"]
        assert isinstance(headers, dict)
        assert headers == mantle_request_headers("messages")


class TestMessagesRequestLogging:
    """Offline unit tests for what the Messages route records in the request log.

    ``metadata.user_id`` has no Converse equivalent, so the documented behavior on
    this path is exactly "it is logged" — per-user cost attribution reads it from
    the structured request log.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/routes/anthropic_messages.py:create_message
    """

    pytestmark = pytest.mark.local

    @staticmethod
    def _stub_model(monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace model resolution and invocation with offline stubs."""
        details = SimpleNamespace(id="test.logging-model")
        monkeypatch.setattr(
            anthropic_messages, "REQUEST_ID", SimpleNamespace(get=lambda: "req-1")
        )
        monkeypatch.setattr(
            anthropic_messages, "validate_model", AsyncMock(return_value=details)
        )
        monkeypatch.setattr(
            anthropic_messages,
            "get_chat_model",
            lambda _: SimpleNamespace(create_message=AsyncMock(return_value={})),
        )

    async def test_metadata_user_id_is_logged(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, object]
    ) -> None:
        """``metadata.user_id`` is recorded as ``request_user_id`` in the request log."""
        self._stub_model(monkeypatch)
        request = MessageCreateParams.model_validate(
            {
                "model": "test.logging-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "metadata": {"user_id": "user-42"},
            }
        )
        await anthropic_messages.create_message(request)
        assert request_log["request_user_id"] == "user-42"

    async def test_no_metadata_leaves_the_user_id_unset(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, object]
    ) -> None:
        """A request without ``metadata`` records no ``request_user_id`` key."""
        self._stub_model(monkeypatch)
        request = MessageCreateParams.model_validate(
            {
                "model": "test.logging-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        await anthropic_messages.create_message(request)
        assert "request_user_id" not in request_log


class TestMessagesBedrockHeaders:
    """Offline unit tests for the Bedrock passthrough headers on ``POST /v1/messages``.

    The headers are documented as a feature of this route but are applied by the
    shared app middleware, so only an end-to-end request through the Anthropic
    route proves they are honored here and not just on the OpenAI surfaces.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html
         stdapi/main.py:_middleware
         stdapi/aws_bedrock.py:set_guardrail_configuration
         stdapi/aws_bedrock.py:set_performance_configuration
    """

    pytestmark = pytest.mark.local

    @staticmethod
    def _capture_context_vars(
        monkeypatch: pytest.MonkeyPatch,
        details: ModelDetails,
        captured: dict[str, object],
    ) -> None:
        """Stub the chat model so it records the header-derived context vars."""

        async def _create_message(
            *_args: object, **_kwargs: object
        ) -> dict[str, object]:
            captured["guardrail"] = GUARDRAIL_CONFIG_VAR.get(None)
            captured["performance"] = PERFORMANCE_CONFIG_VAR.get(None)
            return {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "hi"}],
                "model": details.id,
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }

        monkeypatch.setattr(
            anthropic_messages,
            "get_chat_model",
            lambda _: SimpleNamespace(create_message=_create_message),
        )

    def test_performance_headers_reach_the_bedrock_call(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The latency and service-tier headers become the request's performance config.

        Unlike the guardrail ones these headers need no opt-in setting: they only
        select a latency and billing tier for the caller's own request.  Both values
        travel unchanged and are kept apart, because Bedrock spells the baseline
        differently in each -- ``standard`` for the latency knob, ``default`` for the
        service tier.
        """
        details = _register_test_model(
            monkeypatch, "test.performance-headers-model", "Performance Headers Test"
        )
        captured: dict[str, object] = {}
        self._capture_context_vars(monkeypatch, details, captured)

        response = anthropic_app_client.post(
            "/anthropic/v1/messages",
            json={"model": details.id, "messages": [{"role": "user", "content": "hi"}]},
            headers={
                "X-Amzn-Bedrock-PerformanceConfig-Latency": "optimized",
                "X-Amzn-Bedrock-Service-Tier": "priority",
            },
        )
        assert response.status_code == 200, response.text
        assert captured["performance"] == ("optimized", "priority")

    def test_performance_headers_are_absent_without_them(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request sending neither header selects neither latency nor service tier.

        The context var is bound on every request, so a missing header has to resolve
        to ``None`` rather than leave the previous request's tier in place -- a leak
        would bill an unrelated caller at the priority tier.
        """
        details = _register_test_model(
            monkeypatch, "test.performance-headers-off-model", "Performance Headers Off"
        )
        captured: dict[str, object] = {}
        self._capture_context_vars(monkeypatch, details, captured)

        response = anthropic_app_client.post(
            "/anthropic/v1/messages",
            json={"model": details.id, "messages": [{"role": "user", "content": "hi"}]},
        )
        assert response.status_code == 200, response.text
        assert captured["performance"] == (None, None)

    def test_guardrail_headers_reach_the_bedrock_call(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The three guardrail headers become the request's Bedrock guardrail config.

        The override is opt-in: without ``aws_bedrock_allow_guardrail_override`` the
        headers are ignored in favor of the server-wide configuration, so the
        setting is enabled here to exercise the header branch.
        """
        details = _register_test_model(
            monkeypatch, "test.guardrail-headers-model", "Guardrail Headers Test"
        )
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", True)
        captured: dict[str, object] = {}
        self._capture_context_vars(monkeypatch, details, captured)

        response = anthropic_app_client.post(
            "/anthropic/v1/messages",
            json={"model": details.id, "messages": [{"role": "user", "content": "hi"}]},
            headers={
                "X-Amzn-Bedrock-GuardrailIdentifier": "gr-123",
                "X-Amzn-Bedrock-GuardrailVersion": "2",
                "X-Amzn-Bedrock-Trace": "enabled",
            },
        )
        assert response.status_code == 200, response.text
        assert captured["guardrail"] == {
            "guardrailIdentifier": "gr-123",
            "guardrailVersion": "2",
            "trace": "enabled",
        }

    def test_guardrail_headers_are_ignored_without_the_override_setting(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Caller-supplied guardrail headers are dropped when the override is disabled.

        Letting any caller pick the guardrail would defeat a server-enforced one,
        so the header branch is gated behind an explicit opt-in.
        """
        details = _register_test_model(
            monkeypatch, "test.guardrail-headers-off-model", "Guardrail Headers Off"
        )
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", False)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", None)
        captured: dict[str, object] = {}
        self._capture_context_vars(monkeypatch, details, captured)

        response = anthropic_app_client.post(
            "/anthropic/v1/messages",
            json={"model": details.id, "messages": [{"role": "user", "content": "hi"}]},
            headers={
                "X-Amzn-Bedrock-GuardrailIdentifier": "gr-123",
                "X-Amzn-Bedrock-GuardrailVersion": "2",
            },
        )
        assert response.status_code == 200, response.text
        assert captured["guardrail"] is None


def _mantle_refusal_payload() -> dict[str, Any]:
    """Return the ``Message`` payload a Bedrock Mantle refusal answers with.

    A fresh dict per call: the passthrough validator prunes in place.
    """
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": "claude-x",
        "stop_reason": "refusal",
        "stop_details": {
            "type": "refusal",
            "category": "cyber",
            "explanation": "The request could enable cyber harm.",
        },
        "usage": {
            "input_tokens": 12,
            "output_tokens": 340,
            "output_tokens_details": {"thinking_tokens": 128},
            "service_tier": "standard",
        },
    }


class TestMessagesRefusalStopDetails:
    """Offline unit tests for ``stop_details`` on the Messages response types.

    ``stop_reason: "refusal"`` alone does not say which policy stopped the
    generation; the category and its explanation live in ``stop_details``. A
    passthrough response carrying one must keep it rather than have it pruned as
    an unknown extra.

    Ref: https://platform.claude.com/docs/en/api/messages
         anthropic.types.refusal_stop_details.RefusalStopDetails
         stdapi/types/anthropic_messages.py:Message
         stdapi/models/chat/_mantle/_default.py:create_message
    """

    pytestmark = pytest.mark.local

    def test_passthrough_message_keeps_the_refusal_category(self) -> None:
        """A passthrough refusal keeps ``stop_details`` instead of losing it to pruning.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/aws_bedrock_mantle.py:validate_pruning_extras
        """
        message = validate_pruning_extras(Message, _mantle_refusal_payload())

        assert message.stop_reason == "refusal"
        assert message.stop_details is not None
        assert message.stop_details.type == "refusal"
        assert message.stop_details.category == "cyber"
        assert (
            message.stop_details.explanation == "The request could enable cyber harm."
        )

    def test_message_delta_carries_the_refusal_details(self) -> None:
        """The streaming ``message_delta`` declares the same ``stop_details`` object.

        Ref: https://platform.claude.com/docs/en/api/messages
             anthropic.types.raw_message_delta_event.RawMessageDeltaEvent.Delta
             stdapi/types/anthropic_messages.py:MessageDelta
        """
        delta = MessageDelta.model_validate(
            {
                "stop_reason": "refusal",
                "stop_details": {"type": "refusal", "category": "bio"},
            }
        )

        assert delta.stop_details is not None
        assert delta.stop_details.category == "bio"
        assert delta.stop_details.explanation is None

    def test_an_unknown_refusal_category_is_refused(self) -> None:
        """The category is the documented enumeration, not an open string.

        Ref: anthropic.types.refusal_stop_details.RefusalStopDetails
             stdapi/types/anthropic_messages.py:RefusalStopDetails
        """
        payload = _mantle_refusal_payload()
        payload["stop_details"]["category"] = "not-a-category"

        with pytest.raises(ApiError) as excinfo:
            validate_pruning_extras(Message, payload)
        assert excinfo.value.status == 502


class TestMessagesOutputTokensDetails:
    """Offline unit tests for ``usage.output_tokens_details``.

    ``output_tokens`` stays the authoritative billed total; the breakdown says
    how many of those tokens were spent on internal reasoning. Bedrock's
    ``TokenUsage`` has no reasoning split, so the Converse path leaves it unset
    rather than inventing one.

    Ref: https://platform.claude.com/docs/en/api/messages
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
         anthropic.types.output_tokens_details.OutputTokensDetails
         stdapi/types/anthropic_messages.py:Usage
    """

    pytestmark = pytest.mark.local

    def test_passthrough_usage_keeps_the_thinking_token_count(self) -> None:
        """A passthrough response keeps the reasoning-token breakdown.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/aws_bedrock_mantle.py:validate_pruning_extras
        """
        message = validate_pruning_extras(Message, _mantle_refusal_payload())

        assert message.usage.output_tokens_details is not None
        assert message.usage.output_tokens_details.thinking_tokens == 128

    def test_message_delta_usage_carries_the_breakdown(self) -> None:
        """The trailing ``message_delta`` usage declares the same breakdown.

        Ref: anthropic.types.message_delta_usage.MessageDeltaUsage
             stdapi/types/anthropic_messages.py:MessageDeltaUsage
        """
        usage = MessageDeltaUsage.model_validate(
            {"output_tokens": 340, "output_tokens_details": {"thinking_tokens": 128}}
        )

        assert usage.output_tokens_details is not None
        assert usage.output_tokens_details.thinking_tokens == 128


def _converse_response(**extra: object) -> dict[str, Any]:
    """Return a minimal Bedrock ``Converse`` response, extended with *extra*."""
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": "hi"}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 5, "outputTokens": 3, "totalTokens": 8},
        **extra,
    }


class TestMessagesConverseUsageAttribution:
    """Offline unit tests for the usage fields the Converse path attributes itself.

    Upstream always reports which tier served a request and, when a prompt was
    cached, how the cache-creation tokens split across TTLs. Bedrock reports both
    — the tier on the ``Converse`` response, the split in ``TokenUsage.cacheDetails``
    — so a response served this way carries them too instead of a null.

    Ref: https://platform.claude.com/docs/en/api/messages
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ServiceTier.html
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CacheDetail.html
         stdapi/models/chat/_adapters/_anthropic_message.py:format_response
    """

    pytestmark = pytest.mark.local

    @staticmethod
    async def _serve(
        monkeypatch: pytest.MonkeyPatch,
        response: dict[str, Any],
        *,
        requested_tier: str | None = None,
    ) -> Message:
        """Serve one non-streaming message from a canned Converse *response*."""
        bedrock_request: dict[str, Any] = {"modelId": ""}
        if requested_tier is not None:
            bedrock_request["serviceTier"] = {"type": requested_tier}
        monkeypatch.setattr(
            ChatModel,
            "build_message_request",
            AsyncMock(return_value=(bedrock_request, None)),
        )
        monkeypatch.setattr(ChatModel, "converse", AsyncMock(return_value=response))
        request = MessageCreateParams.model_validate(
            {
                "model": "test.usage-model",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        message = await ChatModel("test.usage-model").create_message(request, "msg_1")
        assert isinstance(message, Message)
        return message

    async def test_service_tier_reports_the_tier_bedrock_served(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, object]
    ) -> None:
        """The tier the backend reports serving wins over the one that was requested.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ServiceTier.html
             stdapi/models/chat/_adapters/_anthropic_message.py:map_response_service_tier
        """
        message = await self._serve(
            monkeypatch,
            _converse_response(serviceTier={"type": "priority"}),
            requested_tier="default",
        )

        assert message.usage.service_tier == "priority"

    async def test_service_tier_falls_back_to_the_requested_tier(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, object]
    ) -> None:
        """A response that reports no tier is attributed to the tier that was asked for.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/models/chat/_default.py:create_message
        """
        message = await self._serve(
            monkeypatch, _converse_response(), requested_tier="default"
        )

        assert message.usage.service_tier == "standard"

    async def test_a_tier_anthropic_cannot_name_stays_unset(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, object]
    ) -> None:
        """``flex`` has no Anthropic equivalent, so no tier is claimed at all.

        Ref: anthropic.types.usage.Usage
             stdapi/models/chat/_adapters/_anthropic_message.py:map_response_service_tier
        """
        message = await self._serve(
            monkeypatch,
            _converse_response(serviceTier={"type": "flex"}),
            requested_tier="flex",
        )

        assert message.usage.service_tier is None

    async def test_cache_creation_is_split_by_ttl(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, object]
    ) -> None:
        """``cacheDetails`` becomes the per-TTL ``cache_creation`` breakdown.

        AWS prices a 5-minute and a 1-hour cache write differently, and reports
        each bucket separately; the flat total stays the authoritative figure.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CacheDetail.html
             stdapi/models/chat/_adapters/_anthropic_message.py:format_response
        """
        message = await self._serve(
            monkeypatch,
            _converse_response(
                usage={
                    "inputTokens": 5,
                    "outputTokens": 3,
                    "totalTokens": 8,
                    "cacheWriteInputTokens": 1500,
                    "cacheDetails": [
                        {"ttl": "1h", "inputTokens": 1000},
                        {"ttl": "5m", "inputTokens": 500},
                    ],
                }
            ),
        )

        assert message.usage.cache_creation_input_tokens == 1500
        assert message.usage.cache_creation is not None
        assert message.usage.cache_creation.ephemeral_1h_input_tokens == 1000
        assert message.usage.cache_creation.ephemeral_5m_input_tokens == 500

    async def test_no_cache_write_leaves_the_breakdown_unset(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, object]
    ) -> None:
        """Without a reported per-TTL split, no breakdown is invented.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
             stdapi/models/chat/_adapters/_anthropic_message.py:format_response
        """
        message = await self._serve(monkeypatch, _converse_response())

        assert message.usage.cache_creation is None
        assert message.usage.output_tokens_details is None
