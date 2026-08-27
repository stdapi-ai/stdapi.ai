"""Offline unit tests for the Bedrock Mantle wire-format conversion helpers.

Covers request, response and stream conversion between the three Mantle wire
shapes (:mod:`stdapi.models.chat._mantle._convert`) and the passthrough
payload builders, all without any network or AWS call.

A Mantle model serves only a subset of these wire shapes, so the gateway converts
client-side whenever the inbound API is not one the model supports. Cross-shape
conversion always composes through the Chat Completions shape.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
     https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
     https://developers.openai.com/api/reference/resources/responses/methods/create
     https://platform.claude.com/docs/en/api/messages
     stdapi/models/chat/_mantle/_convert.py
"""

from __future__ import annotations

from base64 import b64encode
from hashlib import sha256
from json import dumps, loads
from typing import TYPE_CHECKING, Any

import pytest
from sse_starlette import ServerSentEvent

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock_mantle import MantleError, encode_mantle_response_id
from stdapi.config import SETTINGS
from stdapi.models.chat._mantle import _convert as mantle_convert
from stdapi.types.anthropic_messages import MessageCreateParams
from stdapi.types.openai_chat_completions import (
    CompletionCreateParams as ChatCompletionCreateParams,
)
from stdapi.types.openai_completions import (
    CompletionCreateParams as LegacyCompletionCreateParams,
)
from stdapi.types.openai_responses import (
    ResponseCreateParams,
    WebSearchPreviewTool,
    WebSearchTool,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from pydantic import BaseModel
    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.aws_http import SseEvent

pytestmark = pytest.mark.local


def _mantle_region() -> RegionName:
    """Return a region configured for Mantle in the test settings."""
    return SETTINGS.aws_bedrock_mantle_regions[0]


def _data_uri(data: bytes, media_type: str) -> str:
    """Build a base64 ``data:`` URI carrying *data*."""
    return f"data:{media_type};base64,{b64encode(data).decode()}"


async def _agen(events: list[SseEvent]) -> AsyncGenerator[SseEvent]:
    """Yield pre-built SSE events as a fake upstream stream."""
    for item in events:
        yield item


async def _collect(stream: AsyncGenerator[SseEvent]) -> list[SseEvent]:
    """Drain an SSE event stream into a list of (event name, data) pairs."""
    return [event async for event in stream]


async def _drain_into(
    stream: AsyncGenerator[SseEvent], collected: list[SseEvent]
) -> None:
    """Append every event from *stream* into *collected*, propagating errors.

    Used to inspect the events emitted before an in-band error aborts a stream.
    """
    async for event in stream:
        # Appends one by one: a comprehension would discard partial results on error.
        collected.append(event)  # noqa: PERF401


def _payloads(events: list[SseEvent]) -> list[dict[str, Any]]:
    """Parse the JSON data payload of each collected SSE event."""
    return [loads(data) for _, data in events]


def _names(events: list[SseEvent]) -> list[str | None]:
    """Return the event names of a collected SSE event list."""
    return [name for name, _ in events]


# ---------------------------------------------------------------------------
# 1. chat -> responses request
# ---------------------------------------------------------------------------


class TestChatToResponsesRequest:
    """Chat Completions request payloads converted to the Responses shape.

    Ref: https://developers.openai.com/api/docs/guides/migrate-to-responses
         stdapi/models/chat/_mantle/_convert.py:_chat_to_responses_request
    """

    def test_system_and_string_user_message_produce_instructions_and_input(
        self,
    ) -> None:
        """System text becomes instructions; a string user message stays a string.

        Responses has no ``system`` role, so that text is hoisted to the top-level
        ``instructions`` field. A string ``content`` stays a string instead of being
        expanded into an ``input_text`` part list.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_messages_to_input
        """
        payload = {
            "model": "m",
            "messages": [
                {"role": "system", "content": "Be nice."},
                {"role": "user", "content": "Hello"},
            ],
        }
        out = mantle_convert.convert_payload("chat_completions", "responses", payload)
        assert out["instructions"] == "Be nice."
        assert out["input"] == [{"role": "user", "content": "Hello"}]

    def test_user_part_list_with_image_data_uri_becomes_input_image(self) -> None:
        """A user text+image part list converts to input_text/input_image parts.

        Responses flattens the nested ``image_url`` object: the URL becomes the part's
        ``image_url`` string and ``detail`` becomes a sibling field.

        Ref: stdapi/models/chat/_mantle/_convert.py:_input_parts_from_chat
        """
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,QUJD",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        }
        out = mantle_convert.convert_payload("chat_completions", "responses", payload)
        assert out["input"] == [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "look"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,QUJD",
                        "detail": "high",
                    },
                ],
            }
        ]

    def test_assistant_message_with_tool_calls_and_content(self) -> None:
        """An assistant message yields a text item and a function_call item.

        Responses has no per-message ``tool_calls`` array: every call becomes its own
        top-level ``function_call`` item, emitted after the assistant text item.

        Ref: https://developers.openai.com/api/docs/guides/function-calling
             stdapi/models/chat/_mantle/_convert.py:_input_items_from_chat_assistant
        """
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Checking...",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Paris"}',
                            },
                        }
                    ],
                }
            ],
        }
        out = mantle_convert.convert_payload("chat_completions", "responses", payload)
        assert out["input"] == [
            {"role": "assistant", "content": "Checking..."},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
            },
        ]

    def test_tool_message_becomes_function_call_output(self) -> None:
        """A tool-role message becomes a function_call_output item.

        Ref: https://developers.openai.com/api/docs/guides/function-calling
             stdapi/models/chat/_mantle/_convert.py:_chat_messages_to_input
        """
        payload = {
            "model": "m",
            "messages": [{"role": "tool", "tool_call_id": "call_1", "content": "72F"}],
        }
        out = mantle_convert.convert_payload("chat_completions", "responses", payload)
        assert out["input"] == [
            {"type": "function_call_output", "call_id": "call_1", "output": "72F"}
        ]

    def test_function_tool_converted_non_function_tool_dropped(self) -> None:
        """Only ``function``-typed tools survive conversion to Responses tools.

        A Chat Completions ``custom`` tool has no Responses analogue, so it is dropped
        rather than forwarded with a type upstream would reject.

        Ref: stdapi/models/chat/_mantle/_convert.py:_responses_tools_from_chat
        """
        payload = {
            "model": "m",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    },
                },
                {"type": "custom", "custom": {"name": "x"}},
            ],
        }
        out = mantle_convert.convert_payload("chat_completions", "responses", payload)
        assert out["tools"] == [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object"},
            }
        ]

    @pytest.mark.parametrize(
        ("tool_choice", "expected"),
        [
            ("auto", "auto"),
            (
                {"type": "function", "function": {"name": "f"}},
                {"type": "function", "name": "f"},
            ),
        ],
    )
    def test_tool_choice_forms(
        self, tool_choice: str | dict[str, Any], expected: str | dict[str, Any]
    ) -> None:
        """``auto`` passes through; a named function choice becomes flat.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
             stdapi/models/chat/_mantle/_convert.py:_responses_tool_choice_from_chat
        """
        payload: dict[str, Any] = {
            "model": "m",
            "messages": [],
            "tool_choice": tool_choice,
        }
        out = mantle_convert.convert_payload("chat_completions", "responses", payload)
        assert out["tool_choice"] == expected

    def test_response_format_json_object(self) -> None:
        """``response_format: json_object`` becomes ``text.format: json_object``.

        Ref: https://developers.openai.com/api/docs/guides/structured-outputs
             stdapi/models/chat/_mantle/_convert.py:_text_format_from_response_format
        """
        payload = {
            "model": "m",
            "messages": [],
            "response_format": {"type": "json_object"},
        }
        out = mantle_convert.convert_payload("chat_completions", "responses", payload)
        assert out["text"]["format"] == {"type": "json_object"}

    def test_response_format_json_schema(self) -> None:
        """``response_format: json_schema`` becomes a flat ``text.format`` value.

        Responses drops the ``json_schema`` wrapper object: ``name``, ``schema`` and
        ``strict`` sit directly on ``text.format`` beside its ``type``.

        Ref: https://developers.openai.com/api/docs/guides/structured-outputs
             stdapi/models/chat/_mantle/_convert.py:_text_format_from_response_format
        """
        payload = {
            "model": "m",
            "messages": [],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather",
                    "schema": {"type": "object"},
                    "strict": True,
                },
            },
        }
        out = mantle_convert.convert_payload("chat_completions", "responses", payload)
        assert out["text"]["format"] == {
            "type": "json_schema",
            "name": "weather",
            "schema": {"type": "object"},
            "strict": True,
        }


# ---------------------------------------------------------------------------
# 2. chat -> messages request (rich)
# ---------------------------------------------------------------------------


class TestChatToMessagesRequestRich:
    """Chat Completions request payloads converted to the Anthropic shape.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/models/chat/_mantle/_convert.py:_chat_to_messages_request
    """

    def test_system_and_developer_messages_join_as_system_text(self) -> None:
        """System and developer message text join into the Anthropic system.

        Anthropic has neither a ``system`` nor a ``developer`` message role, so both are
        hoisted into the top-level ``system`` field and joined by a blank line.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_to_messages_request
        """
        payload = {
            "model": "m",
            "messages": [
                {"role": "system", "content": "Be terse."},
                {"role": "developer", "content": "Follow policy."},
                {"role": "user", "content": "Hi"},
            ],
        }
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["system"] == "Be terse.\n\nFollow policy."
        assert out["messages"] == [
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]}
        ]

    def test_user_part_list_with_image_data_uri(self) -> None:
        """A user image_url data URI part becomes a base64 Anthropic image block.

        Anthropic takes the media type and the raw base64 as separate ``source`` members,
        so the ``data:`` URI has to be split rather than forwarded.

        Ref: stdapi/models/chat/_mantle/_convert.py:_anthropic_image_block
        """
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,QUJD"},
                        },
                    ],
                }
            ],
        }
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["messages"][0]["content"] == [
            {"type": "text", "text": "look"},
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/png", "data": "QUJD"},
            },
        ]

    def test_assistant_message_with_tool_calls_and_text(self) -> None:
        """Assistant text and tool calls both convert into Anthropic blocks.

        ``tool_calls[].function.arguments`` is a JSON *string* on the OpenAI wire while
        Anthropic's ``tool_use.input`` is a decoded object.

        Ref: stdapi/models/chat/_mantle/_convert.py:_anthropic_blocks_from_chat_assistant
             stdapi/models/chat/_mantle/_convert.py:_json_object
        """
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": "Sure",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": '{"x": 1}'},
                        }
                    ],
                }
            ],
        }
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["messages"][0] == {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Sure"},
                {"type": "tool_use", "id": "call_1", "name": "f", "input": {"x": 1}},
            ],
        }

    def test_tool_message_becomes_tool_result_block(self) -> None:
        """A tool-role message becomes a user turn carrying a tool_result block.

        Anthropic has no ``tool`` role: results are ``tool_result`` blocks inside a user
        turn.

        Ref: stdapi/models/chat/_mantle/_convert.py:_anthropic_messages_from_chat
        """
        payload = {
            "model": "m",
            "messages": [
                {"role": "tool", "tool_call_id": "call_1", "content": "result text"}
            ],
        }
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["messages"][0] == {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "result text",
                }
            ],
        }

    def test_consecutive_same_role_messages_merge_into_one_turn(self) -> None:
        """A user message directly followed by a tool message merge into one turn.

        Anthropic requires alternating roles, and a tool result is a user-turn block, so
        it has to be appended to the preceding user turn instead of opening a new one.

        Ref: stdapi/models/chat/_mantle/_convert.py:_append_anthropic_turn
        """
        payload = {
            "model": "m",
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "tool", "tool_call_id": "c1", "content": "result"},
            ],
        }
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["messages"] == [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hi"},
                    {"type": "tool_result", "tool_use_id": "c1", "content": "result"},
                ],
            }
        ]

    def test_temperature_clamped_to_one(self) -> None:
        """A temperature above the Anthropic range is clamped to 1.0.

        Chat Completions accepts ``temperature`` up to 2.0 while Anthropic caps it at 1.0;
        the value is clamped so a legal inbound request is not turned into an upstream 400.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             stdapi/models/chat/_mantle/_convert.py:_chat_to_messages_request
        """
        payload = {"model": "m", "messages": [], "temperature": 1.5}
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["temperature"] == 1.0

    def test_stop_string_becomes_single_element_list(self) -> None:
        """A single stop string becomes a one-element ``stop_sequences`` list.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_to_messages_request
        """
        payload = {"model": "m", "messages": [], "stop": "STOP"}
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["stop_sequences"] == ["STOP"]

    def test_stop_list_passthrough(self) -> None:
        """A stop list passes through unchanged as ``stop_sequences``.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_to_messages_request
        """
        payload = {"model": "m", "messages": [], "stop": ["A", "B"]}
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["stop_sequences"] == ["A", "B"]

    def test_user_field_becomes_metadata_user_id(self) -> None:
        """The ``user`` field maps to Anthropic ``metadata.user_id``.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_to_messages_request
        """
        payload = {"model": "m", "messages": [], "user": "user-123"}
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["metadata"] == {"user_id": "user-123"}

    def test_tool_without_parameters_gets_default_object_schema(self) -> None:
        """A function tool without ``parameters`` gets a default object schema.

        ``input_schema`` is required by Anthropic, so a parameterless function tool gets a
        bare object schema rather than being dropped.

        Ref: stdapi/models/chat/_mantle/_convert.py:_anthropic_tools_from_chat
        """
        payload = {
            "model": "m",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "f"}}],
        }
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["tools"] == [{"name": "f", "input_schema": {"type": "object"}}]

    @pytest.mark.parametrize(
        ("tool_choice", "expected"),
        [
            ("auto", {"type": "auto"}),
            ("required", {"type": "any"}),
            ("none", {"type": "none"}),
            (
                {"type": "function", "function": {"name": "f"}},
                {"type": "tool", "name": "f"},
            ),
        ],
    )
    def test_tool_choice_forms(
        self, tool_choice: str | dict[str, Any], expected: dict[str, Any]
    ) -> None:
        """Each Chat Completions tool choice form maps to its Anthropic shape.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools#forcing-tool-use
             stdapi/models/chat/_mantle/_convert.py:_anthropic_tool_choice_from_chat
        """
        payload: dict[str, Any] = {
            "model": "m",
            "messages": [],
            "tool_choice": tool_choice,
        }
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["tool_choice"] == expected

    @pytest.mark.parametrize(
        ("tool_choice", "expected"),
        [
            (None, {"type": "auto", "disable_parallel_tool_use": True}),
            ("auto", {"type": "auto", "disable_parallel_tool_use": True}),
            ("required", {"type": "any", "disable_parallel_tool_use": True}),
            # Anthropic's `none` choice has no disable_parallel_tool_use field.
            ("none", {"type": "none"}),
            (
                {"type": "function", "function": {"name": "f"}},
                {"type": "tool", "name": "f", "disable_parallel_tool_use": True},
            ),
        ],
    )
    def test_parallel_tool_calls_false_disables_parallel_tool_use(
        self, tool_choice: str | dict[str, Any] | None, expected: dict[str, Any]
    ) -> None:
        """``parallel_tool_calls: false`` maps to ``disable_parallel_tool_use``.

        Anthropic expresses the constraint as a flag on ``tool_choice`` instead of a
        top-level field, so a choice has to be synthesised when the caller sent none.
        Anthropic's ``none`` choice has no such flag, which is why it is parametrised apart.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools#forcing-tool-use
             stdapi/models/chat/_mantle/_convert.py:_anthropic_tool_choice_from_chat
        """
        payload: dict[str, Any] = {
            "model": "m",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "parallel_tool_calls": False,
        }
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["tool_choice"] == expected

    def test_parallel_tool_calls_false_without_tools_adds_no_choice(self) -> None:
        """Without tools the synthesised ``auto`` tool choice is not emitted.

        A ``tool_choice`` without ``tools`` is rejected upstream, so the synthesised choice
        is suppressed entirely.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools#forcing-tool-use
             stdapi/models/chat/_mantle/_convert.py:_anthropic_tool_choice_from_chat
        """
        payload = {"model": "m", "messages": [], "parallel_tool_calls": False}
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert "tool_choice" not in out

    def test_unknown_role_ignored(self) -> None:
        """A message with an unrecognized role produces no turn.

        Skipping keeps the turn list valid: forwarding an unknown role would make Anthropic
        reject the whole request.

        Ref: stdapi/models/chat/_mantle/_convert.py:_anthropic_messages_from_chat
        """
        payload = {"model": "m", "messages": [{"role": "foo", "content": "bar"}]}
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["messages"] == []

    def test_unmappable_content_part_is_dropped_without_losing_the_turn(self) -> None:
        """An audio part has no Anthropic equivalent and is dropped, not forwarded.

        Anthropic's content-block union has no audio member, so forwarding the
        part verbatim would make upstream reject the whole request; the
        surrounding text must survive so the turn still carries the question.

        Ref: https://platform.claude.com/docs/en/api/messages
             stdapi/models/chat/_mantle/_convert.py:_anthropic_block_from_chat_part
        """
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe this."},
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "QUJD", "format": "wav"},
                        },
                    ],
                }
            ],
        }
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["messages"][0]["content"] == [
            {"type": "text", "text": "Transcribe this."}
        ]

    def test_http_image_url_becomes_url_source(self) -> None:
        """A non-data-URI image URL becomes an Anthropic ``url`` source.

        Ref: stdapi/models/chat/_mantle/_convert.py:_anthropic_image_block
        """
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/pic.png"},
                        }
                    ],
                }
            ],
        }
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["messages"][0]["content"] == [
            {
                "type": "image",
                "source": {"type": "url", "url": "https://example.com/pic.png"},
            }
        ]

    def test_max_tokens_defaults_when_absent(self) -> None:
        """Missing token limits fall back to the default max_tokens.

        ``max_tokens`` is required by Anthropic but optional for Chat Completions, so the
        converter injects the module default instead of letting upstream reject the request.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_to_messages_request
             stdapi/models/chat/_mantle/_convert.py:_DEFAULT_MAX_TOKENS
        """
        payload = {"model": "m", "messages": []}
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["max_tokens"] == 4096


# ---------------------------------------------------------------------------
# 3. messages -> chat request (edge arms)
# ---------------------------------------------------------------------------


class TestMessagesToChatRequestEdges:
    """Anthropic Messages request payloads converted to the Chat Completions shape.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
         stdapi/models/chat/_mantle/_convert.py:_messages_to_chat_request
    """

    def test_system_block_list_becomes_chat_system_message(self) -> None:
        """A list-of-blocks system prompt joins into a single system message.

        Block texts are concatenated without a separator and emitted as the single leading
        ``system`` message.

        Ref: stdapi/models/chat/_mantle/_convert.py:_messages_to_chat_request
             stdapi/models/chat/_mantle/_convert.py:_anthropic_text
        """
        payload = {
            "model": "m",
            "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "messages": [],
        }
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert out["messages"][0] == {"role": "system", "content": "ab"}

    def test_user_turn_with_mixed_blocks_keeps_all_parts(self) -> None:
        """User turns keep text, base64 image, URL image and base64 document parts.

        Base64 sources are re-encoded as ``data:`` URIs, the only inline form Chat
        Completions accepts, while a URL source is forwarded as the URL itself.

        Ref: https://developers.openai.com/api/docs/guides/file-inputs
             stdapi/models/chat/_mantle/_convert.py:_chat_part_from_anthropic_image
             stdapi/models/chat/_mantle/_convert.py:_chat_part_from_anthropic_document
        """
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "QUJD",
                            },
                        },
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://x/pic.png"},
                        },
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": "UERG",
                            },
                        },
                    ],
                }
            ],
        }
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert out["messages"][0] == {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,QUJD"},
                },
                {"type": "image_url", "image_url": {"url": "https://x/pic.png"}},
                {
                    "type": "file",
                    "file": {"file_data": "data:application/pdf;base64,UERG"},
                },
            ],
        }

    def test_document_url_source_has_no_equivalent_and_is_dropped(self) -> None:
        """A ``document`` block with a URL source has no mapping and is dropped.

        A Chat Completions ``file`` part carries inline ``file_data`` or a stored
        ``file_id``, never a URL, so the block is unmappable and the emptied turn is
        dropped with it.

        Ref: https://developers.openai.com/api/docs/guides/file-inputs
             stdapi/models/chat/_mantle/_convert.py:_chat_part_from_anthropic_document
        """
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "url", "url": "https://x/doc.pdf"},
                        }
                    ],
                }
            ],
        }
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert out["messages"] == []

    def test_assistant_turn_with_tool_use_and_text(self) -> None:
        """An assistant turn's text and tool_use block convert to message+tool_calls.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_messages_from_anthropic_turn
        """
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Here"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "f",
                            "input": {"a": 1},
                        },
                    ],
                }
            ],
        }
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert out["messages"][0] == {
            "role": "assistant",
            "content": "Here",
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {"name": "f", "arguments": '{"a":1}'},
                }
            ],
        }

    def test_assistant_turn_with_image_block_drops_non_text_content(self) -> None:
        """Assistant turns keep only text; image/document parts are dropped.

        Chat Completions assistant messages carry text and refusals only; image and
        document parts are legal for user messages.

        Ref: stdapi/models/chat/_mantle/_convert.py:_assemble_chat_message
        """
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Here"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": "QUJD",
                            },
                        },
                    ],
                }
            ],
        }
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert out["messages"][0] == {"role": "assistant", "content": "Here"}

    def test_tool_result_block_becomes_tool_message_before_turn(self) -> None:
        """A tool_result block emits a ``tool`` message ahead of the rest of the turn.

        Chat Completions requires the tool output in its own ``tool`` message, so a turn
        mixing a result with text splits into two messages, result first.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_messages_from_anthropic_turn
        """
        payload = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "42",
                        },
                        {"type": "text", "text": "continue"},
                    ],
                }
            ],
        }
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert out["messages"] == [
            {"role": "tool", "tool_call_id": "toolu_1", "content": "42"},
            {"role": "user", "content": [{"type": "text", "text": "continue"}]},
        ]

    def test_metadata_user_id_over_64_chars_is_hashed(self) -> None:
        """A user ID over 64 characters is replaced by its SHA-256 hex digest.

        OpenAI caps ``user`` at 64 characters while Anthropic's ``metadata.user_id`` is not
        capped; hashing keeps the value stable per original ID instead of truncating it
        into a collision.

        Ref: https://developers.openai.com/api/docs/guides/safety-best-practices#implement-safety-identifiers
             stdapi/models/chat/_mantle/_convert.py:_openai_user
        """
        long_id = "u" * 70
        payload = {"model": "m", "messages": [], "metadata": {"user_id": long_id}}
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert out["user"] != long_id
        assert out["user"] == sha256(long_id.encode()).hexdigest()
        assert len(out["user"]) == 64

    def test_metadata_user_id_under_64_chars_passthrough(self) -> None:
        """A user ID within the OpenAI limit passes through unchanged.

        Ref: stdapi/models/chat/_mantle/_convert.py:_openai_user
        """
        payload = {"model": "m", "messages": [], "metadata": {"user_id": "u1"}}
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert out["user"] == "u1"

    def test_tool_without_input_schema_is_skipped(self) -> None:
        """A tool missing the ``input_schema`` key is dropped from the output.

        ``input_schema`` is mandatory on an Anthropic tool, so a tool lacking it is
        malformed and skipped rather than forwarded with a fabricated schema.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_tools_from_anthropic
        """
        payload = {
            "model": "m",
            "messages": [],
            "tools": [
                {"name": "no_schema"},
                {"name": "good", "input_schema": {"type": "object"}},
            ],
        }
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert [tool["function"]["name"] for tool in out["tools"]] == ["good"]

    def test_tool_description_forwarded(self) -> None:
        """A tool's description is forwarded to the Chat Completions shape.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_tools_from_anthropic
        """
        payload = {
            "model": "m",
            "messages": [],
            "tools": [
                {
                    "name": "f",
                    "input_schema": {"type": "object"},
                    "description": "does f",
                }
            ],
        }
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert out["tools"][0] == {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {"type": "object"},
                "description": "does f",
            },
        }

    @pytest.mark.parametrize(
        ("tool_choice", "expected"),
        [
            ({"type": "auto"}, "auto"),
            ({"type": "any"}, "required"),
            ({"type": "none"}, "none"),
            (
                {"type": "tool", "name": "f"},
                {"type": "function", "function": {"name": "f"}},
            ),
        ],
    )
    def test_tool_choice_forms(
        self, tool_choice: dict[str, Any], expected: str | dict[str, Any]
    ) -> None:
        """Each Anthropic tool choice type maps to its Chat Completions shape.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
             stdapi/models/chat/_mantle/_convert.py:_chat_tool_choice_from_anthropic
        """
        payload = {"model": "m", "messages": [], "tool_choice": tool_choice}
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert out["tool_choice"] == expected

    def test_disable_parallel_tool_use_becomes_parallel_tool_calls(self) -> None:
        """``disable_parallel_tool_use`` maps to ``parallel_tool_calls: false``.

        Anthropic hangs the flag off ``tool_choice`` while Chat Completions has a
        top-level field, so the single inbound object produces two outbound fields.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_tool_choice_from_anthropic
        """
        payload = {
            "model": "m",
            "messages": [],
            "tool_choice": {
                "type": "tool",
                "name": "f",
                "disable_parallel_tool_use": True,
            },
        }
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert out["tool_choice"] == {"type": "function", "function": {"name": "f"}}
        assert out["parallel_tool_calls"] is False

    def test_parallel_tool_use_left_enabled(self) -> None:
        """Without ``disable_parallel_tool_use`` no ``parallel_tool_calls`` is set.

        Parallel tool use is the default on both APIs, so nothing is emitted when the
        inbound flag is absent.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_tool_choice_from_anthropic
        """
        payload = {"model": "m", "messages": [], "tool_choice": {"type": "auto"}}
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert "parallel_tool_calls" not in out

    def test_stop_sequences_become_stop(self) -> None:
        """``stop_sequences`` maps to the Chat Completions ``stop`` field.

        Ref: stdapi/models/chat/_mantle/_convert.py:_messages_to_chat_request
        """
        payload = {"model": "m", "messages": [], "stop_sequences": ["END"]}
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert out["stop"] == ["END"]


# ---------------------------------------------------------------------------
# 4. responses -> chat request (branches)
# ---------------------------------------------------------------------------


class TestResponsesToChatRequestBranches:
    """Responses API request payloads converted to the Chat Completions shape.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/models/chat/_mantle/_convert.py:_responses_to_chat_request
    """

    def test_instructions_and_string_input_become_messages(self) -> None:
        """Instructions become a system message; a string input becomes user text.

        Ref: stdapi/models/chat/_mantle/_convert.py:_responses_to_chat_request
        """
        payload = {"model": "m", "instructions": "Be terse.", "input": "Hi"}
        out = mantle_convert._responses_to_chat_request(payload)  # noqa: SLF001
        assert out["messages"] == [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hi"},
        ]

    def test_function_call_item_becomes_assistant_tool_call(self) -> None:
        """A ``function_call`` input item becomes an assistant tool_calls message.

        Ref: https://developers.openai.com/api/docs/guides/function-calling
             stdapi/models/chat/_mantle/_convert.py:_chat_messages_from_input_item
        """
        payload = {
            "model": "m",
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "f",
                    "arguments": '{"a": 1}',
                }
            ],
        }
        out = mantle_convert._responses_to_chat_request(payload)  # noqa: SLF001
        assert out["messages"] == [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "f", "arguments": '{"a": 1}'},
                    }
                ],
            }
        ]

    def test_parallel_function_calls_share_one_assistant_message(self) -> None:
        """Consecutive ``function_call`` items merge into one assistant message.

        Responses lists parallel calls as separate items while Chat Completions groups them
        in one assistant message; without coalescing, the second call would open a new
        assistant message and the tool outputs would no longer follow their calls.

        Ref: stdapi/models/chat/_mantle/_convert.py:_append_chat_message
        """
        payload = {
            "model": "m",
            "input": [
                {"role": "user", "content": "Hi"},
                {
                    "type": "function_call",
                    "call_id": "call_a",
                    "name": "f",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "call_id": "call_b",
                    "name": "g",
                    "arguments": "{}",
                },
                {"type": "function_call_output", "call_id": "call_a", "output": "a"},
                {"type": "function_call_output", "call_id": "call_b", "output": "b"},
            ],
        }
        out = mantle_convert._responses_to_chat_request(payload)  # noqa: SLF001
        assert [message["role"] for message in out["messages"]] == [
            "user",
            "assistant",
            "tool",
            "tool",
        ]
        assert [call["id"] for call in out["messages"][1]["tool_calls"]] == [
            "call_a",
            "call_b",
        ]

    def test_function_call_output_item_becomes_tool_message(self) -> None:
        """A ``function_call_output`` input item becomes a ``tool`` message.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_messages_from_input_item
        """
        payload = {
            "model": "m",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "result",
                }
            ],
        }
        out = mantle_convert._responses_to_chat_request(payload)  # noqa: SLF001
        assert out["messages"] == [
            {"role": "tool", "tool_call_id": "call_1", "content": "result"}
        ]

    def test_function_call_output_with_part_list_output(self) -> None:
        """A part-list ``function_call_output.output`` is flattened to plain text.

        Chat Completions ``tool`` message content is a plain string, so a part list has to
        be collapsed rather than forwarded.

        Ref: stdapi/models/chat/_mantle/_convert.py:_responses_text
        """
        payload = {
            "model": "m",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": [{"type": "output_text", "text": "42"}],
                }
            ],
        }
        out = mantle_convert._responses_to_chat_request(payload)  # noqa: SLF001
        assert out["messages"][0]["content"] == "42"

    def test_input_image_and_input_file_parts_become_chat_parts(self) -> None:
        """``input_image``/``input_file`` parts convert to ``image_url``/``file``.

        Ref: https://developers.openai.com/api/docs/guides/file-inputs
             stdapi/models/chat/_mantle/_convert.py:_chat_part_from_input
        """
        payload = {
            "model": "m",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,QUJD",
                            "detail": "high",
                        },
                        {
                            "type": "input_file",
                            "file_data": "data:application/pdf;base64,UERG",
                            "filename": "doc.pdf",
                        },
                    ],
                }
            ],
        }
        out = mantle_convert._responses_to_chat_request(payload)  # noqa: SLF001
        assert out["messages"][0]["content"] == [
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,QUJD", "detail": "high"},
            },
            {
                "type": "file",
                "file": {
                    "file_data": "data:application/pdf;base64,UERG",
                    "filename": "doc.pdf",
                },
            },
        ]

    def test_input_file_without_file_data_is_dropped(self) -> None:
        """An ``input_file`` part without inline ``file_data`` has no mapping.

        A bare ``file_id`` refers to an OpenAI-hosted file that Mantle cannot resolve, so
        the part is unmappable and the emptied turn is dropped with it.

        Ref: https://developers.openai.com/api/docs/guides/file-inputs
             stdapi/models/chat/_mantle/_convert.py:_chat_part_from_input
        """
        payload = {
            "model": "m",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_file", "file_id": "file-1"}],
                }
            ],
        }
        out = mantle_convert._responses_to_chat_request(payload)  # noqa: SLF001
        assert out["messages"] == []

    def test_max_output_tokens_becomes_max_completion_tokens(self) -> None:
        """``max_output_tokens`` maps to ``max_completion_tokens``.

        Ref: stdapi/models/chat/_mantle/_convert.py:_responses_to_chat_request
        """
        payload = {"model": "m", "input": "hi", "max_output_tokens": 500}
        out = mantle_convert._responses_to_chat_request(payload)  # noqa: SLF001
        assert out["max_completion_tokens"] == 500

    def test_reasoning_effort_forwarded(self) -> None:
        """``reasoning.effort`` maps to the flat ``reasoning_effort`` field.

        Ref: https://developers.openai.com/api/docs/guides/reasoning
             stdapi/models/chat/_mantle/_convert.py:_responses_to_chat_request
        """
        payload = {"model": "m", "input": "hi", "reasoning": {"effort": "low"}}
        out = mantle_convert._responses_to_chat_request(payload)  # noqa: SLF001
        assert out["reasoning_effort"] == "low"

    def test_text_format_json_object(self) -> None:
        """``text.format: json_object`` maps to ``response_format: json_object``.

        Ref: https://developers.openai.com/api/docs/guides/structured-outputs
             stdapi/models/chat/_mantle/_convert.py:_response_format_from_text
        """
        payload = {
            "model": "m",
            "input": "hi",
            "text": {"format": {"type": "json_object"}},
        }
        out = mantle_convert._responses_to_chat_request(payload)  # noqa: SLF001
        assert out["response_format"] == {"type": "json_object"}

    def test_text_format_json_schema(self) -> None:
        """``text.format: json_schema`` maps to a nested ``response_format``.

        Chat Completions keeps the ``json_schema`` wrapper that Responses flattens, so the
        name, schema, strict and description fields move back down one level.

        Ref: https://developers.openai.com/api/docs/guides/structured-outputs
             stdapi/models/chat/_mantle/_convert.py:_response_format_from_text
        """
        payload = {
            "model": "m",
            "input": "hi",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "n",
                    "schema": {"type": "object"},
                    "strict": True,
                    "description": "d",
                }
            },
        }
        out = mantle_convert._responses_to_chat_request(payload)  # noqa: SLF001
        assert out["response_format"] == {
            "type": "json_schema",
            "json_schema": {
                "name": "n",
                "schema": {"type": "object"},
                "strict": True,
                "description": "d",
            },
        }

    def test_tools_and_named_tool_choice(self) -> None:
        """Flat Responses tools/tool_choice map to their nested Chat shapes.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
             stdapi/models/chat/_mantle/_convert.py:_chat_tools_from_responses
             stdapi/models/chat/_mantle/_convert.py:_chat_tool_choice_from_responses
        """
        payload = {
            "model": "m",
            "input": "hi",
            "tools": [
                {
                    "type": "function",
                    "name": "f",
                    "description": "d",
                    "parameters": {"type": "object"},
                    "strict": True,
                }
            ],
            "tool_choice": {"type": "function", "name": "f"},
        }
        out = mantle_convert._responses_to_chat_request(payload)  # noqa: SLF001
        assert out["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "f",
                    "description": "d",
                    "parameters": {"type": "object"},
                    "strict": True,
                },
            }
        ]
        assert out["tool_choice"] == {"type": "function", "function": {"name": "f"}}


# ---------------------------------------------------------------------------
# 5. Non-stream response conversion
# ---------------------------------------------------------------------------


class TestNonStreamResponseConversions:
    """Complete-response conversion between the three Mantle wire shapes.

    Ref: stdapi/models/chat/_mantle/_convert.py:convert_response
    """

    def test_chat_to_responses_response(self) -> None:
        """Chat text, tool calls and usage convert to Responses output items.

        The upstream ID token is reused so the two shapes stay correlatable:
        ``chatcmpl-abc123`` becomes ``resp_abc123`` and every output item ID is derived
        from it.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_to_responses_response
             stdapi/models/chat/_mantle/_convert.py:_id_token
        """
        raw = {
            "id": "chatcmpl-abc123",
            "created": 1000,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Paris"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        }
        out = mantle_convert.convert_response("chat_completions", "responses", raw)
        assert out["id"] == "resp_abc123"
        assert out["status"] == "completed"
        message_item, call_item = out["output"]
        assert message_item == {
            "type": "message",
            "id": "resp_abc123-msg-0",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Hello", "annotations": []}],
        }
        assert call_item == {
            "type": "function_call",
            "id": "resp_abc123-fc-call_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city": "Paris"}',
            "status": "completed",
        }
        assert out["usage"] == {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 2, "cache_write_tokens": 0},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 15,
        }

    def test_reasoning_tokens_survive_both_usage_directions(self) -> None:
        """Reasoning tokens are carried over in both usage conversions.

        The two usage converters round-trip the fields both shapes carry; the added
        ``cached_tokens: 0`` is the Chat Completions shape's mandatory counterpart to
        ``input_tokens_details``.

        Ref: stdapi/models/chat/_mantle/_convert.py:_responses_usage_from_chat
             stdapi/models/chat/_mantle/_convert.py:_chat_usage_from_responses
        """
        chat_usage = {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "completion_tokens_details": {"reasoning_tokens": 4},
        }
        responses_usage = mantle_convert._responses_usage_from_chat(chat_usage)  # noqa: SLF001
        assert responses_usage["output_tokens_details"] == {"reasoning_tokens": 4}
        assert mantle_convert._chat_usage_from_responses(responses_usage) == {  # noqa: SLF001
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 4},
        }

    def test_chat_to_messages_response(self) -> None:
        """Chat text, tool calls and usage convert to Anthropic content blocks.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_to_messages_response
             stdapi/models/chat/_mantle/_convert.py:_FINISH_TO_STOP
        """
        raw = {
            "id": "chatcmpl-abc123",
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        out = mantle_convert.convert_response("chat_completions", "messages", raw)
        assert out["id"] == "msg_abc123"
        assert out["stop_reason"] == "tool_use"
        assert out["content"] == [
            {"type": "text", "text": "Hello"},
            {"type": "tool_use", "id": "call_1", "name": "f", "input": {}},
        ]
        assert out["usage"] == {
            "input_tokens": 10,
            "output_tokens": 5,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

    def test_messages_to_chat_response_with_tool_use_and_cache_usage(self) -> None:
        """Anthropic text, tool_use and cache usage convert to CC choice/usage.

        Anthropic reports cache reads and writes outside ``input_tokens`` while OpenAI's
        ``prompt_tokens`` includes cached tokens, so the counters are added back in
        (8 + 4 + 1 = 13) and the cached share is echoed in ``prompt_tokens_details``.

        Ref: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
             stdapi/models/chat/_mantle/_convert.py:_chat_usage_from_messages
        """
        raw = {
            "id": "msg_xyz789",
            "model": "m",
            "content": [
                {"type": "text", "text": "Hi"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 8,
                "output_tokens": 3,
                "cache_read_input_tokens": 4,
                "cache_creation_input_tokens": 1,
            },
        }
        out = mantle_convert.convert_response("messages", "chat_completions", raw)
        assert out["id"] == "chatcmpl-xyz789"
        choice = out["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] == "Hi"
        assert choice["message"]["tool_calls"] == [
            {
                "id": "toolu_1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
            }
        ]
        assert out["usage"] == {
            "prompt_tokens": 13,
            "completion_tokens": 3,
            "total_tokens": 16,
            "prompt_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 1},
        }

    def test_messages_to_chat_response_stop_reason_mapped_without_tool_use(
        self,
    ) -> None:
        """Without tool_use blocks, the finish reason follows ``stop_reason``.

        Ref: stdapi/models/chat/_mantle/_convert.py:_messages_to_chat_response
             stdapi/models/chat/_mantle/_convert.py:_STOP_TO_FINISH
        """
        raw = {
            "id": "msg_1",
            "model": "m",
            "content": [{"type": "text", "text": "Hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
        out = mantle_convert.convert_response("messages", "chat_completions", raw)
        assert out["choices"][0]["finish_reason"] == "stop"


class TestFinishFromResponse:
    """Chat Completions finish reason derivation from a Responses response.

    Ref: stdapi/models/chat/_mantle/_convert.py:_finish_from_response
    """

    def test_function_call_output_item_forces_tool_calls(self) -> None:
        """A ``function_call`` output item forces the ``tool_calls`` finish reason.

        A pending call in the output overrides the response status: a completed Responses
        response carrying a tool call would otherwise finish as ``stop``, and the client
        would never run the tool.

        Ref: stdapi/models/chat/_mantle/_convert.py:_finish_from_response
        """
        response = {"output": [{"type": "function_call", "name": "f"}]}
        result = mantle_convert._finish_from_response(  # noqa: SLF001
            response, has_tool_calls=False
        )
        assert result == "tool_calls"

    def test_incomplete_max_output_tokens_maps_to_length(self) -> None:
        """An incomplete response with ``max_output_tokens`` maps to ``length``.

        Ref: stdapi/models/chat/_mantle/_convert.py:_finish_from_response
             stdapi/models/chat/_mantle/_convert.py:_INCOMPLETE_TO_FINISH
        """
        response = {
            "output": [],
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
        }
        result = mantle_convert._finish_from_response(  # noqa: SLF001
            response, has_tool_calls=False
        )
        assert result == "length"


# ---------------------------------------------------------------------------
# 6. Stream conversion
# ---------------------------------------------------------------------------


class TestChatToResponsesStream:
    """Chat Completions SSE chunks converted to a Responses SSE stream.

    Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
         stdapi/models/chat/_mantle/_convert.py:_chat_stream_to_responses
    """

    def _cc_chunks(self) -> list[SseEvent]:
        """Build role, text-delta and tool-call CC chunks ending on a finish chunk."""
        return [
            (
                None,
                dumps(
                    {
                        "id": "chatcmpl-1",
                        "created": 100,
                        "model": "m",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": None,
                            }
                        ],
                    }
                ),
            ),
            (
                None,
                dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "Hel"},
                                "finish_reason": None,
                            }
                        ]
                    }
                ),
            ),
            (
                None,
                dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "lo"},
                                "finish_reason": None,
                            }
                        ]
                    }
                ),
            ),
            (
                None,
                dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_1",
                                            "type": "function",
                                            "function": {
                                                "name": "get_weather",
                                                "arguments": "",
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                ),
            ),
            (
                None,
                dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": '{"city":'},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                ),
            ),
            (
                None,
                dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": '"Paris"}'},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                ),
            ),
            (
                None,
                dumps(
                    {
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                        ]
                    }
                ),
            ),
        ]

    async def test_full_event_sequence_ends_with_completed_and_increasing_sequence(
        self,
    ) -> None:
        """Text then a tool call produce the full close/open event sequence.

        ``sequence_number`` is a stream-wide counter starting at 0, and switching from text
        to a tool call must close the message item (text done, part done, item done) before
        the function-call item opens.

        Ref: stdapi/models/chat/_mantle/_convert.py:_responses_chunk_events
             stdapi/models/chat/_mantle/_convert.py:_responses_event
        """
        chunks = [
            *self._cc_chunks(),
            (
                None,
                dumps(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                ),
            ),
        ]
        events = await _collect(
            mantle_convert.convert_stream(
                "chat_completions", "responses", _agen(chunks)
            )
        )
        assert _names(events) == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
            "response.output_item.done",
            "response.output_item.added",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.output_item.done",
            "response.completed",
        ]
        seqs = [payload["sequence_number"] for payload in _payloads(events)]
        assert seqs == list(range(len(events)))
        completed = _payloads(events)[-1]
        assert completed["response"]["usage"]["output_tokens"] == 5

    async def test_in_progress_follows_created_with_same_response(self) -> None:
        """``response.in_progress`` follows ``response.created`` as upstream does.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_stream_to_responses
        """
        events = await _collect(
            mantle_convert.convert_stream(
                "chat_completions", "responses", _agen(self._cc_chunks())
            )
        )
        payloads = _payloads(events)
        assert _names(events)[:2] == ["response.created", "response.in_progress"]
        assert payloads[1]["response"]["id"] == payloads[0]["response"]["id"]
        assert payloads[1]["response"]["status"] == "in_progress"

    async def test_stream_ending_without_usage_chunk_still_emits_completed(
        self,
    ) -> None:
        """A stream ending without a usage chunk still closes with zero usage.

        Usage is requested upstream, but a stream that ends without it must still terminate
        with a well-formed ``response.completed`` rather than leaving the client hanging.

        Ref: stdapi/models/chat/_mantle/_convert.py:_responses_stream_tail
        """
        events = await _collect(
            mantle_convert.convert_stream(
                "chat_completions", "responses", _agen(self._cc_chunks())
            )
        )
        assert events[-1][0] == "response.completed"
        completed = _payloads(events)[-1]
        assert completed["response"]["usage"]["output_tokens"] == 0

    async def test_length_finish_reason_emits_incomplete_event_not_completed(
        self,
    ) -> None:
        """A "length" finish reason terminates with response.incomplete.

        Matches the sibling Converse adapter's wire grammar: the event name
        itself (not just the nested ``status``) marks a truncated response.

        Ref: stdapi/models/chat/_mantle/_convert.py:_responses_completed
             stdapi/models/chat/_mantle/_convert.py:_FINISH_TO_INCOMPLETE
        """
        chunk = dumps(
            {
                "id": "chatcmpl-1",
                "created": 1,
                "model": "m",
                "choices": [
                    {"index": 0, "delta": {"content": "hi"}, "finish_reason": "length"}
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 7,
                    "total_tokens": 10,
                },
            }
        )
        events = await _collect(
            mantle_convert.convert_stream(
                "chat_completions", "responses", _agen([(None, chunk)])
            )
        )
        assert "response.completed" not in _names(events)
        assert events[-1][0] == "response.incomplete"
        terminal = _payloads(events)[-1]
        assert terminal["type"] == "response.incomplete"
        assert terminal["response"]["status"] == "incomplete"
        assert (
            terminal["response"]["incomplete_details"]["reason"] == "max_output_tokens"
        )
        assert terminal["response"]["usage"]["output_tokens"] == 7

    async def test_error_chunk_raises_mantle_error_without_fabricated_tail(
        self,
    ) -> None:
        """An in-band chat-shaped error chunk aborts the stream: no fabricated tail.

        Raising lets the route surface the upstream failure; synthesising the terminal
        events would make a failed generation look like a successful completion.

        Ref: stdapi/models/chat/_mantle/_convert.py:_stream_error_message
             stdapi/aws_bedrock_mantle.py:MantleError
        """
        chunks: list[SseEvent] = [
            *self._cc_chunks()[:2],
            (None, dumps({"error": {"message": "upstream exploded"}})),
        ]
        stream = mantle_convert.convert_stream(
            "chat_completions", "responses", _agen(chunks)
        )
        collected: list[SseEvent] = []
        with pytest.raises(MantleError, match="upstream exploded") as exc_info:
            await _drain_into(stream, collected)
        assert exc_info.value.status == 502
        assert "response.completed" not in _names(collected)

    async def test_malformed_frame_is_skipped_without_aborting_the_stream(self) -> None:
        """A non-JSON upstream frame is skipped; the rest still converts.

        An unparseable frame is dropped rather than raised on, so a single bad frame cannot
        lose the text already streamed nor the terminal event.

        Ref: stdapi/models/chat/_mantle/_convert.py:_parsed_chunk
        """
        chunks: list[SseEvent] = [
            *self._cc_chunks()[:2],
            (None, "{not json"),
            *self._cc_chunks()[2:],
        ]
        events = await _collect(
            mantle_convert.convert_stream(
                "chat_completions", "responses", _agen(chunks)
            )
        )
        assert _names(events) == _names(
            await _collect(
                mantle_convert.convert_stream(
                    "chat_completions", "responses", _agen(self._cc_chunks())
                )
            )
        ), "the bad frame must not add or remove a single event"
        deltas = {
            name: "".join(
                payload["delta"]
                for (event_name, _data), payload in zip(
                    events, _payloads(events), strict=True
                )
                if event_name == name
            )
            for name in (
                "response.output_text.delta",
                "response.function_call_arguments.delta",
            )
        }
        assert deltas["response.output_text.delta"] == "Hello"
        assert deltas["response.function_call_arguments.delta"] == '{"city":"Paris"}'

    async def test_route_assigned_id_is_carried_by_every_response_event(self) -> None:
        """The plumbed route ID replaces the minted one on all events.

        A minted ``resp_`` ID is parsed by the route as a Mantle-tagged ID
        and always 404s: the streamed ID must be the retrievable one.

        Ref: stdapi/models/chat/_mantle/_convert.py:convert_stream
             stdapi/models/chat/_mantle/_convert.py:_ResponsesStreamState
        """
        chunks = [
            *self._cc_chunks(),
            (
                None,
                dumps(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                ),
            ),
        ]
        events = await _collect(
            mantle_convert.convert_stream(
                "chat_completions", "responses", _agen(chunks), "resp-request-id"
            )
        )
        payloads = _payloads(events)
        with_response = {
            payload["type"]: payload["response"]["id"]
            for payload in payloads
            if "response" in payload
        }
        assert with_response == {
            "response.created": "resp-request-id",
            "response.in_progress": "resp-request-id",
            "response.completed": "resp-request-id",
        }
        item_ids = {
            payload["item"]["id"] for payload in payloads if "item" in payload
        } | {payload["item_id"] for payload in payloads if "item_id" in payload}
        assert item_ids
        assert all(item_id.startswith("resp-request-id-") for item_id in item_ids)
        assert not any("resp_" in data for _, data in events)

    async def test_minted_id_is_used_when_no_route_id_is_plumbed(self) -> None:
        """Without a route ID the converter still mints a synthetic response ID.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_stream_to_responses
        """
        events = await _collect(
            mantle_convert.convert_stream(
                "chat_completions", "responses", _agen(self._cc_chunks())
            )
        )
        payloads = _payloads(events)
        minted = payloads[0]["response"]["id"]
        assert minted.startswith("resp_")
        assert {
            payload["response"]["id"] for payload in payloads if "response" in payload
        } == {minted}, "the minted ID must be stable across the whole stream"
        assert all(
            payload["item"]["id"].startswith(f"{minted}-")
            for payload in payloads
            if "item" in payload
        )


class TestChatToMessagesStream:
    """Chat Completions SSE chunks converted to an Anthropic Messages SSE stream.

    Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
         stdapi/models/chat/_mantle/_convert.py:_chat_stream_to_messages
    """

    def _cc_chunks(self) -> list[SseEvent]:
        """Build role, text-delta and tool-call CC chunks without a finish chunk."""
        return [
            (
                None,
                dumps(
                    {
                        "id": "chatcmpl-1",
                        "model": "m",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": ""},
                                "finish_reason": None,
                            }
                        ],
                    }
                ),
            ),
            (
                None,
                dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "Hel"},
                                "finish_reason": None,
                            }
                        ]
                    }
                ),
            ),
            (
                None,
                dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "lo"},
                                "finish_reason": None,
                            }
                        ]
                    }
                ),
            ),
            (
                None,
                dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "toolu_1",
                                            "type": "function",
                                            "function": {
                                                "name": "get_weather",
                                                "arguments": "",
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                ),
            ),
            (
                None,
                dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": '{"city":'},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                ),
            ),
            (
                None,
                dumps(
                    {
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "function": {"arguments": '"Paris"}'},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ]
                    }
                ),
            ),
        ]

    async def test_full_event_sequence_with_usage_and_finish(self) -> None:
        """Text, a tool call, a finish chunk and usage produce the full sequence.

        Anthropic indexes content blocks, so the text block is stopped before the tool-use
        block starts, and the terminal counters land on ``message_delta`` rather than
        ``message_stop``.

        Ref: stdapi/models/chat/_mantle/_convert.py:_messages_chunk_events
        """
        chunks = [
            *self._cc_chunks(),
            (
                None,
                dumps(
                    {
                        "choices": [
                            {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                        ]
                    }
                ),
            ),
            (
                None,
                dumps(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                ),
            ),
        ]
        events = await _collect(
            mantle_convert.convert_stream("chat_completions", "messages", _agen(chunks))
        )
        assert _names(events) == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        message_delta = _payloads(events)[-2]
        assert message_delta["delta"]["stop_reason"] == "tool_use"
        assert message_delta["usage"]["output_tokens"] == 5

    async def test_response_id_is_ignored_when_converting_to_messages(self) -> None:
        """Anthropic message IDs stay minted: they are not retrievable.

        The route-assigned ID identifies a stored Responses object, so leaking it into
        an Anthropic ``message_start`` would advertise an ID no Messages endpoint can
        resolve.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_stream_to_messages
        """
        events = await _collect(
            mantle_convert.convert_stream(
                "chat_completions", "messages", _agen(self._cc_chunks()), "resp-req-id"
            )
        )
        assert _payloads(events)[0]["message"]["id"].startswith("msg_")
        assert not any("resp-req-id" in data for _, data in events)

    async def test_early_end_without_finish_or_usage_closes_open_block(self) -> None:
        """A stream ending mid tool-call still closes the block and defaults usage.

        Anthropic clients track open blocks, so an unterminated stream must still emit the
        matching ``content_block_stop`` and a complete usage object.

        Ref: stdapi/models/chat/_mantle/_convert.py:_messages_stream_tail
        """
        events = await _collect(
            mantle_convert.convert_stream(
                "chat_completions", "messages", _agen(self._cc_chunks())
            )
        )
        assert _names(events)[-3:] == [
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        message_delta = _payloads(events)[-2]
        assert message_delta["delta"]["stop_reason"] == "end_turn"
        assert message_delta["usage"] == {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }

    async def test_error_chunk_raises_mantle_error_without_fabricated_tail(
        self,
    ) -> None:
        """An in-band chat-shaped error chunk aborts the stream: no fabricated tail.

        Raising lets the route surface the upstream failure; synthesising the terminal
        events would make a failed generation look like a completed message.

        Ref: stdapi/models/chat/_mantle/_convert.py:_stream_error_message
             stdapi/aws_bedrock_mantle.py:MantleError
        """
        chunks: list[SseEvent] = [
            *self._cc_chunks()[:2],
            (None, dumps({"error": {"message": "upstream exploded"}})),
        ]
        stream = mantle_convert.convert_stream(
            "chat_completions", "messages", _agen(chunks)
        )
        collected: list[SseEvent] = []
        with pytest.raises(MantleError, match="upstream exploded") as exc_info:
            await _drain_into(stream, collected)
        assert exc_info.value.status == 502
        assert "message_stop" not in _names(collected)


class TestMessagesToChatStream:
    """Anthropic Messages SSE events converted to Chat Completions SSE chunks.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
         stdapi/models/chat/_mantle/_convert.py:_messages_stream_to_chat
    """

    async def test_full_event_sequence_with_tool_use_and_text_deltas(self) -> None:
        """A tool_use, arguments, a text delta, a thinking delta and ping convert.

        ``thinking_delta`` has no Chat Completions field and ``ping`` is keep-alive only, so
        neither yields a chunk; usage arrives as a final choice-less chunk.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_delta_from_messages
        """
        events_in: list[SseEvent] = [
            (
                "message_start",
                dumps(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_1",
                            "model": "m",
                            "usage": {"input_tokens": 10, "output_tokens": 0},
                        },
                    }
                ),
            ),
            (
                "content_block_start",
                dumps(
                    {
                        "type": "content_block_start",
                        "content_block": {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                        },
                    }
                ),
            ),
            (
                "content_block_delta",
                dumps(
                    {
                        "type": "content_block_delta",
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": '{"city": "Paris"}',
                        },
                    }
                ),
            ),
            (
                "content_block_delta",
                dumps(
                    {
                        "type": "content_block_delta",
                        "delta": {"type": "text_delta", "text": "hello"},
                    }
                ),
            ),
            (
                "content_block_delta",
                dumps(
                    {
                        "type": "content_block_delta",
                        "delta": {"type": "thinking_delta", "thinking": "pondering"},
                    }
                ),
            ),
            (
                "message_delta",
                dumps(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "tool_use"},
                        "usage": {"output_tokens": 5},
                    }
                ),
            ),
            ("ping", dumps({"type": "ping"})),
        ]
        events = await _collect(
            mantle_convert.convert_stream(
                "messages", "chat_completions", _agen(events_in)
            )
        )
        chunks = _payloads(events)
        assert len(chunks) == 6  # thinking_delta and ping produce no chunk
        assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}
        assert (
            chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
            == "get_weather"
        )
        assert (
            chunks[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
            == '{"city": "Paris"}'
        )
        assert chunks[3]["choices"][0]["delta"] == {"content": "hello"}
        assert chunks[4]["choices"][0]["finish_reason"] == "tool_calls"
        assert chunks[5]["usage"] == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
        }

    async def test_unnamed_error_type_payload_raises_mantle_error(self) -> None:
        """An unnamed ``{"type": "error"}`` data event aborts the stream (502).

        The Anthropic wire marks errors with ``type: error``, so a frame arriving without
        its SSE event name must still be recognised as an error.

        Ref: stdapi/models/chat/_mantle/_convert.py:_stream_error_message
        """
        events_in: list[SseEvent] = [
            (None, dumps({"type": "error", "message": "boom problem"}))
        ]
        stream = mantle_convert.convert_stream(
            "messages", "chat_completions", _agen(events_in)
        )
        with pytest.raises(MantleError, match="boom problem") as exc_info:
            await _collect(stream)
        assert exc_info.value.status == 502


class TestResponsesToChatStreamExtension:
    """Responses SSE events converted to Chat Completions SSE chunks.

    Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
         stdapi/models/chat/_mantle/_convert.py:_responses_stream_to_chat
    """

    async def test_function_call_item_and_argument_delta_with_unknown_event_ignored(
        self,
    ) -> None:
        """A function_call item, argument deltas and completion convert; unknown is skipped.

        Unknown ``response.*`` events are ignored so a newer upstream event grammar cannot
        break the conversion.

        Ref: stdapi/models/chat/_mantle/_convert.py:_responses_stream_to_chat
        """
        events_in: list[SseEvent] = [
            (
                "response.created",
                dumps({"response": {"id": "resp_1", "created_at": 100, "model": "m"}}),
            ),
            (
                "response.output_item.added",
                dumps(
                    {
                        "item": {
                            "type": "function_call",
                            "call_id": "call_9",
                            "name": "foo",
                            "arguments": "",
                        }
                    }
                ),
            ),
            ("response.function_call_arguments.delta", dumps({"delta": '{"x": 1}'})),
            ("response.some_unknown_event", dumps({"foo": "bar"})),
            (
                "response.completed",
                dumps(
                    {
                        "response": {
                            "id": "resp_1",
                            "status": "completed",
                            "output": [],
                            "usage": {
                                "input_tokens": 1,
                                "output_tokens": 1,
                                "total_tokens": 2,
                            },
                        }
                    }
                ),
            ),
        ]
        events = await _collect(
            mantle_convert.convert_stream(
                "responses", "chat_completions", _agen(events_in)
            )
        )
        chunks = _payloads(events)
        assert len(chunks) == 5  # unknown event yields no chunk
        assert chunks[1]["choices"][0]["delta"]["tool_calls"][0]["id"] == "call_9"
        assert (
            chunks[1]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"]
            == "foo"
        )
        assert (
            chunks[2]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"]
            == '{"x": 1}'
        )
        assert chunks[3]["choices"][0]["finish_reason"] == "tool_calls"
        assert chunks[4]["usage"] == {
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        }

    async def test_response_incomplete_event_emits_finish_and_usage(self) -> None:
        """A named ``response.incomplete`` event ends the stream like ``response.completed``.

        Without this, a truncated Responses stream converted to Chat
        Completions would end with neither a finish reason nor billed usage.

        Ref: stdapi/models/chat/_mantle/_convert.py:_responses_stream_to_chat
             stdapi/models/chat/_mantle/_convert.py:_finish_from_response
        """
        events_in: list[SseEvent] = [
            (
                "response.created",
                dumps({"response": {"id": "resp_1", "created_at": 100, "model": "m"}}),
            ),
            ("response.output_text.delta", dumps({"delta": "hi"})),
            (
                "response.incomplete",
                dumps(
                    {
                        "response": {
                            "id": "resp_1",
                            "status": "incomplete",
                            "incomplete_details": {"reason": "max_output_tokens"},
                            "output": [],
                            "usage": {
                                "input_tokens": 3,
                                "output_tokens": 7,
                                "total_tokens": 10,
                            },
                        }
                    }
                ),
            ),
        ]
        events = await _collect(
            mantle_convert.convert_stream(
                "responses", "chat_completions", _agen(events_in)
            )
        )
        chunks = _payloads(events)
        assert chunks[-2]["choices"][0]["finish_reason"] == "length"
        assert chunks[-1]["usage"]["completion_tokens"] == 7


# ---------------------------------------------------------------------------
# 7. Passthrough payload builders
# ---------------------------------------------------------------------------


class TestChatCompletionsPayloadBuilder:
    """Validated Chat Completions requests dumped to the Mantle passthrough shape.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions-mantle.html
         stdapi/models/chat/_mantle/_convert.py:chat_completions_payload
    """

    async def test_inline_parts_resolved_model_overridden_extension_fields_stripped(
        self,
    ) -> None:
        """Image/audio/file parts are inlined, the model is overridden, store dropped.

        The caller's model name is replaced by the resolved Mantle model ID, gateway-only
        extension fields such as ``store`` never reach Mantle, and ``input_audio.data`` is
        reduced to bare base64 because that part carries the format separately.

        Ref: stdapi/models/chat/_mantle/_convert.py:chat_completions_payload
             stdapi/models/chat/_mantle/_convert.py:_resolve_chat_part
             stdapi/models/chat/_mantle/_convert.py:sanitize_tool_schema
        """
        image_uri = _data_uri(b"PNGDATA", "image/png")
        audio_uri = _data_uri(b"AUDIODATA", "audio/wav")
        audio_b64 = b64encode(b"AUDIODATA").decode()
        file_uri = _data_uri(b"PDFDATA", "application/pdf")
        request = ChatCompletionCreateParams.model_validate(
            {
                "model": "ignored",
                "store": True,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_uri}},
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_uri, "format": "wav"},
                            },
                            {
                                "type": "file",
                                "file": {"file_data": file_uri, "filename": "doc.pdf"},
                            },
                        ],
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "f",
                            "parameters": {
                                "$schema": "http://json-schema.org/draft-07/schema#",
                                "type": "object",
                                "properties": {
                                    "a": {"propertyNames": {"pattern": "^x"}}
                                },
                            },
                        },
                    }
                ],
            }
        )
        payload = await mantle_convert.chat_completions_payload(request, "model-id")
        assert payload["model"] == "model-id"
        assert "store" not in payload
        content = payload["messages"][0]["content"]
        assert content[0]["image_url"]["url"] == image_uri
        assert content[1]["input_audio"]["data"] == audio_b64
        assert content[2]["file"]["file_data"] == file_uri
        assert content[2]["file"]["filename"] == "doc.pdf"
        parameters = payload["tools"][0]["function"]["parameters"]
        assert "propertyNames" not in parameters["properties"]["a"]
        # sanitize_tool_schema only strips "propertyNames": "$schema" is left as-is.
        assert parameters["$schema"] == "http://json-schema.org/draft-07/schema#"

    async def test_text_parts_travel_untouched_beside_the_inlined_ones(self) -> None:
        """A text part in a multimodal message is forwarded byte for byte.

        The resolver walks the validated parts and their dumped twins in lockstep,
        touching only the file-backed ones; a text part carries no ``InputFile``,
        so it must pass through with its position and its content intact rather
        than being dropped or re-encoded.

        Ref: stdapi/models/chat/_mantle/_convert.py:_resolve_chat_part
             stdapi/models/chat/_mantle/_convert.py:_resolve_chat_message_files
        """
        image_uri = _data_uri(b"PNGDATA", "image/png")
        request = ChatCompletionCreateParams.model_validate(
            {
                "model": "ignored",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What is in this picture?"},
                            {"type": "image_url", "image_url": {"url": image_uri}},
                            {"type": "text", "text": "Answer in one word."},
                        ],
                    }
                ],
            }
        )
        payload = await mantle_convert.chat_completions_payload(request, "model-id")
        content = payload["messages"][0]["content"]
        assert [part["type"] for part in content] == ["text", "image_url", "text"]
        assert content[0]["text"] == "What is in this picture?"
        assert content[2]["text"] == "Answer in one word."
        assert content[1]["image_url"]["url"] == image_uri

    async def test_named_tool_choice_forwarded_verbatim(self) -> None:
        """A named-function ``tool_choice`` reaches the upstream payload unchanged.

        This is the passthrough path, so the nested Chat Completions shape must survive
        untouched: a cross-API conversion would flatten it.

        Ref: stdapi/models/chat/_mantle/_convert.py:chat_completions_payload
        """
        request = ChatCompletionCreateParams.model_validate(
            {
                "model": "ignored",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "get_weather"}}],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
            }
        )
        payload = await mantle_convert.chat_completions_payload(request, "model-id")
        assert payload["tool_choice"] == {
            "type": "function",
            "function": {"name": "get_weather"},
        }

    async def test_reasoning_object_is_consumed_and_sent_as_reasoning_effort(
        self,
    ) -> None:
        """``reasoning``/``include_reasoning`` are replaced by ``reasoning_effort``.

        Upstream documents the flat field only, so the objects this surface also
        accepts must not travel: they are normalized on the way in and stripped
        from the passthrough payload.

        Ref: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
             stdapi/models/chat/_mantle/_convert.py:_CHAT_EXTENSION_FIELDS
        """
        request = ChatCompletionCreateParams.model_validate(
            {
                "model": "ignored",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning": {"effort": "high"},
                "include_reasoning": False,
            }
        )
        payload = await mantle_convert.chat_completions_payload(request, "model-id")
        assert payload["reasoning_effort"] == "high"
        assert "reasoning" not in payload
        assert "include_reasoning" not in payload
        converted = mantle_convert._chat_to_responses_request(payload)  # noqa: SLF001
        assert converted["reasoning"] == {"effort": "high"}, (
            "the Responses conversion reads the normalized field"
        )

    async def test_replayed_reasoning_alias_travels_under_the_upstream_name(
        self,
    ) -> None:
        """An assistant turn replaying ``reasoning`` is sent back as ``reasoning``.

        The alias is normalized to ``reasoning_content`` at validation time, and
        ``_restore_reasoning_field`` renames it back, so a client replaying
        either name reaches upstream with the one name upstream knows.

        Ref: stdapi/models/chat/_mantle/_convert.py:_restore_reasoning_field
             stdapi/types/openai_chat_completions.py:ChatCompletionAssistantMessageParam
        """
        request = ChatCompletionCreateParams.model_validate(
            {
                "model": "ignored",
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a", "reasoning": "because"},
                    {"role": "user", "content": "and?"},
                ],
            }
        )
        payload = await mantle_convert.chat_completions_payload(request, "model-id")
        assert payload["messages"][1]["reasoning"] == "because"
        assert "reasoning_content" not in payload["messages"][1]

    async def test_external_web_access_is_gated_and_never_passed_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The web access knob is answered here, not handed to the backend.

        This passthrough forwards every undeclared field verbatim, so a knob
        the operator owns has to be taken off the payload rather than travel
        with it -- and this API serves no web search, so a request asking for
        something else than the server does is refused rather than accepted
        and ignored.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html
             stdapi/models/chat/_mantle/_convert.py:chat_completions_payload
             stdapi/models/chat/_adapters/_common.py:resolve_external_web_access
        """
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_allow_external_web_access_override", True
        )
        body = {
            "model": "ignored",
            "messages": [{"role": "user", "content": "hi"}],
            "external_web_access": False,
        }
        payload = await mantle_convert.chat_completions_payload(
            ChatCompletionCreateParams.model_validate(body), "model-id"
        )
        assert "external_web_access" not in payload

        with pytest.raises(ApiError, match="external_web_access") as exc_info:
            await mantle_convert.chat_completions_payload(
                ChatCompletionCreateParams.model_validate(
                    body | {"external_web_access": True}
                ),
                "model-id",
            )
        assert exc_info.value.status == 400


class TestMessagesPayloadBuilder:
    """Validated Anthropic Messages requests dumped to the Mantle passthrough shape.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html
         stdapi/models/chat/_mantle/_convert.py:messages_payload
    """

    async def test_inline_sources_default_max_tokens_system_folding(self) -> None:
        """Base64 sources stay inline, max_tokens defaults, inline system folds in.

        On Mantle the Anthropic version is an HTTP header, so ``anthropic_version`` is
        stripped from the body. A ``system``-role message is not part of the Anthropic wire
        format and folds into the ``system`` field.

        Ref: stdapi/models/chat/_mantle/_convert.py:messages_payload
             stdapi/models/chat/_mantle/_convert.py:_fold_inline_system
        """
        request = MessageCreateParams.model_validate(
            {
                "model": "ignored",
                "anthropic_version": "2023-06-01",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "QUJD",
                                },
                            },
                            {
                                "type": "document",
                                "source": {
                                    "type": "base64",
                                    "media_type": "application/pdf",
                                    "data": "UERG",
                                },
                            },
                        ],
                    },
                    {"role": "system", "content": "Inline system note."},
                ],
                "tools": [
                    {
                        "name": "f",
                        "input_schema": {
                            "type": "object",
                            "properties": {"a": {"propertyNames": {}}},
                        },
                    }
                ],
            }
        )
        payload = await mantle_convert.messages_payload(request, "model-id")
        assert payload["model"] == "model-id"
        assert "anthropic_version" not in payload
        assert payload["max_tokens"] == 4096
        assert payload["system"] == "Inline system note."
        assert len(payload["messages"]) == 1
        content = payload["messages"][0]["content"]
        assert content[0]["source"]["data"] == "QUJD"
        assert content[1]["source"]["data"] == "UERG"
        schema = payload["tools"][0]["input_schema"]
        assert "propertyNames" not in schema["properties"]["a"]

    async def test_external_web_access_is_gated_and_never_passed_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """This route answers the web access knob instead of forwarding it.

        The Anthropic Messages passthrough sends undeclared fields as they
        arrive, so the operator's control has to be resolved before the
        payload is built here too.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html
             stdapi/models/chat/_mantle/_convert.py:messages_payload
             stdapi/models/chat/_adapters/_common.py:resolve_external_web_access
        """
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_allow_external_web_access_override", True
        )
        body = {
            "model": "ignored",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "external_web_access": False,
        }
        payload = await mantle_convert.messages_payload(
            MessageCreateParams.model_validate(body), "model-id"
        )
        assert "external_web_access" not in payload

        with pytest.raises(ApiError, match="external_web_access") as exc_info:
            await mantle_convert.messages_payload(
                MessageCreateParams.model_validate(
                    body | {"external_web_access": True}
                ),
                "model-id",
            )
        assert exc_info.value.status == 400

    async def test_system_messages_forwarded_when_natively_supported(self) -> None:
        """Mid-conversation system messages stay in place for capable models.

        ``system_message_as_messages`` is set for models that accept a ``system`` role
        inside ``messages``; the top-level ``system`` field is then left as the caller sent
        it.

        Ref: stdapi/models/chat/_mantle/_convert.py:_fold_inline_system
             stdapi/models/chat/_mantle/_convert.py:_is_native_system_placement
        """
        request = MessageCreateParams.model_validate(
            {
                "model": "ignored",
                "max_tokens": 16,
                "system": "Be helpful.",
                "messages": [
                    {"role": "user", "content": "Hello."},
                    {"role": "system", "content": "Answer in one word."},
                    {"role": "assistant", "content": "Hi."},
                    {"role": "user", "content": "And now?"},
                ],
            }
        )
        payload = await mantle_convert.messages_payload(
            request, "model-id", system_message_as_messages=True
        )
        assert payload["system"] == "Be helpful."
        assert [message["role"] for message in payload["messages"]] == [
            "user",
            "system",
            "assistant",
            "user",
        ]

    async def test_misplaced_system_messages_fold_when_natively_supported(self) -> None:
        """Placements the model rejects keep folding into the ``system`` field.

        Native support is per-position: a directive must follow a user turn and either end
        the list or precede an assistant turn. A leading directive and one followed by
        another user turn both fail that test and fold into ``system`` instead.

        Ref: stdapi/models/chat/_mantle/_convert.py:_is_native_system_placement
        """
        request = MessageCreateParams.model_validate(
            {
                "model": "ignored",
                "max_tokens": 16,
                "messages": [
                    {"role": "system", "content": "Be terse."},
                    {"role": "user", "content": "Hello."},
                    {"role": "system", "content": "Answer in one word."},
                    {"role": "user", "content": "And now?"},
                ],
            }
        )
        payload = await mantle_convert.messages_payload(
            request, "model-id", system_message_as_messages=True
        )
        assert payload["system"] == "Be terse.\n\nAnswer in one word."
        assert [message["role"] for message in payload["messages"]] == ["user", "user"]

    async def test_trailing_system_message_is_forwarded(self) -> None:
        """A directive ending the message list is accepted by the model as-is.

        Ending the list is an accepted placement, so nothing folds and the ``system`` field
        stays absent.

        Ref: stdapi/models/chat/_mantle/_convert.py:_is_native_system_placement
        """
        request = MessageCreateParams.model_validate(
            {
                "model": "ignored",
                "max_tokens": 16,
                "messages": [
                    {"role": "user", "content": "Hello."},
                    {"role": "system", "content": "Answer in one word."},
                ],
            }
        )
        payload = await mantle_convert.messages_payload(
            request, "model-id", system_message_as_messages=True
        )
        assert "system" not in payload
        assert [message["role"] for message in payload["messages"]] == [
            "user",
            "system",
        ]

    async def test_consecutive_system_messages_are_forwarded(self) -> None:
        """Consecutive directives are one section, placed as a whole.

        Placement is evaluated for the whole run of consecutive directives, not per
        message, so a pair between a user and an assistant turn is forwarded intact.

        Ref: stdapi/models/chat/_mantle/_convert.py:_is_native_system_placement
        """
        request = MessageCreateParams.model_validate(
            {
                "model": "ignored",
                "max_tokens": 16,
                "messages": [
                    {"role": "user", "content": "Hello."},
                    {"role": "system", "content": "Answer in one word."},
                    {"role": "system", "content": "Stay factual."},
                    {"role": "assistant", "content": "Hi."},
                    {"role": "user", "content": "And now?"},
                ],
            }
        )
        payload = await mantle_convert.messages_payload(
            request, "model-id", system_message_as_messages=True
        )
        assert "system" not in payload
        assert [message["role"] for message in payload["messages"]] == [
            "user",
            "system",
            "system",
            "assistant",
            "user",
        ]

    async def test_system_messages_folded_when_not_natively_supported(self) -> None:
        """Mid-conversation system messages fold into ``system`` by default.

        Folding appends to any caller-provided ``system`` text rather than replacing it.

        Ref: stdapi/models/chat/_mantle/_convert.py:_fold_inline_system
        """
        request = MessageCreateParams.model_validate(
            {
                "model": "ignored",
                "max_tokens": 16,
                "system": "Be helpful.",
                "messages": [
                    {"role": "user", "content": "Hello."},
                    {"role": "system", "content": "Answer in one word."},
                    {"role": "assistant", "content": "Hi."},
                    {"role": "user", "content": "And now?"},
                ],
            }
        )
        payload = await mantle_convert.messages_payload(request, "model-id")
        assert payload["system"] == "Be helpful.\n\nAnswer in one word."
        assert [message["role"] for message in payload["messages"]] == [
            "user",
            "assistant",
            "user",
        ]


class TestResponsesPayloadBuilder:
    """Validated Responses requests dumped to the Mantle passthrough shape.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/models/chat/_mantle/_convert.py:responses_payload
    """

    #: Every ``type`` the Responses API accepts for a web search tool.
    _WEB_SEARCH_TOOL_TYPES = (
        "web_search",
        "web_search_2025_08_26",
        "web_search_preview",
        "web_search_preview_2025_03_11",
    )

    async def test_inline_files_web_search_forced_off_tool_sanitized_pinned_region(
        self,
    ) -> None:
        """Files inline, web_search access follows the default, and the region pins.

        ``external_web_access`` defaults to the configured value, off, so a request
        can never reach the public web implicitly. The region is read back from the
        response-ID tag because a stored Mantle response can only be chained in the
        Region that created it.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
             https://developers.openai.com/api/docs/guides/tools-web-search
             stdapi/models/chat/_mantle/_convert.py:responses_payload
             stdapi/models/chat/_mantle/_convert.py:_pin_previous_response
        """
        region = _mantle_region()
        tagged_id = encode_mantle_response_id(region, "resp_native123")
        image_uri = _data_uri(b"PNGDATA", "image/png")
        file_uri = _data_uri(b"PDFDATA", "application/pdf")
        request = ResponseCreateParams.model_validate(
            {
                "model": "ignored",
                "previous_response_id": tagged_id,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": image_uri},
                            {
                                "type": "input_file",
                                "file_data": file_uri,
                                "filename": "doc.pdf",
                            },
                        ],
                    }
                ],
                "tools": [
                    {"type": "web_search"},
                    {
                        "type": "function",
                        "name": "f",
                        "parameters": {
                            "type": "object",
                            "properties": {"a": {"propertyNames": {}}},
                        },
                    },
                ],
            }
        )
        payload, pinned_region = await mantle_convert.responses_payload(
            request, "model-id"
        )
        assert payload["model"] == "model-id"
        assert payload["previous_response_id"] == "resp_native123"
        assert pinned_region == region
        web_search_tool = next(
            tool for tool in payload["tools"] if tool["type"] == "web_search"
        )
        assert web_search_tool["external_web_access"] is False
        function_tool = next(
            tool for tool in payload["tools"] if tool["type"] == "function"
        )
        assert "propertyNames" not in function_tool["parameters"]["properties"]["a"]
        content = payload["input"][0]["content"]
        assert content[0]["image_url"] == image_uri
        assert content[1]["file_data"] == file_uri
        assert content[1]["filename"] == "doc.pdf"

    @staticmethod
    def _web_search_request(
        *, external_web_access: object = None, tool_type: str = "web_search"
    ) -> ResponseCreateParams:
        """Build a Responses request carrying one web search tool."""
        body: dict[str, Any] = {
            "model": "ignored",
            "input": "Who won?",
            "tools": [{"type": tool_type}],
        }
        if external_web_access is not None:
            body["external_web_access"] = external_web_access
        return ResponseCreateParams.model_validate(body)

    async def test_external_web_access_enabled_by_configuration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator that allows external web access gets it on every request.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html
             stdapi/models/chat/_adapters/_common.py:resolve_external_web_access
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_external_web_access", True)
        payload, _ = await mantle_convert.responses_payload(
            self._web_search_request(), "model-id"
        )
        assert payload["tools"][0]["external_web_access"] is True

    def test_external_web_access_is_not_a_field_of_the_web_search_tool(self) -> None:
        """The knob is a request extra, not a field of the tool.

        Upstream's tool carries only ``filters``, ``search_context_size`` and
        ``user_location``; a request field this gateway alone declares is one
        no client will ever send.

        Ref: openai.types.responses.web_search_tool.WebSearchTool
             openai.types.responses.web_search_preview_tool.WebSearchPreviewTool
             stdapi/types/openai_responses.py:WebSearchTool
        """
        assert "external_web_access" not in WebSearchTool.model_fields
        assert "external_web_access" not in WebSearchPreviewTool.model_fields

    @pytest.mark.parametrize("tool", [WebSearchTool, WebSearchPreviewTool])
    def test_the_web_search_tool_schema_names_the_parameter(
        self, tool: type[BaseModel]
    ) -> None:
        """The published schema is where a client learns the parameter exists.

        Carrying it as an extra keeps it out of the tool's fields, so the tool
        description is the only thing an agent reading the OpenAPI or MCP
        schema can discover it from.

        Ref: https://developers.openai.com/api/docs/guides/tools-web-search
             stdapi/types/openai_responses.py:WebSearchTool
        """
        description = tool.model_json_schema()["description"]

        assert "external_web_access" in description
        assert "server" in description

    def test_the_override_setting_points_at_the_parameter_that_works(self) -> None:
        """The override setting's reference names the extra model parameter.

        The tool field it could have named exists on neither tool schema, so a
        description pointing there would send an operator to a knob that
        changes nothing.

        Ref: stdapi/config.py:_Settings.aws_bedrock_allow_external_web_access_override
             stdapi/models/chat/_adapters/_common.py:resolve_external_web_access
        """
        description = (
            type(SETTINGS)
            .model_fields["aws_bedrock_allow_external_web_access_override"]
            .description
        )

        assert description is not None
        assert "extra model parameter" in description
        assert "tool" not in description

    async def test_external_web_access_extra_is_not_forwarded_upstream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The extra parameter is consumed here, never sent as a request field.

        Ref: stdapi/models/chat/_adapters/_common.py:resolve_external_web_access
        """
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_allow_external_web_access_override", True
        )
        payload, _ = await mantle_convert.responses_payload(
            self._web_search_request(external_web_access=True), "model-id"
        )
        assert "external_web_access" not in payload
        assert payload["tools"][0]["external_web_access"] is True

    async def test_a_request_that_searches_nothing_keeps_its_choice_harmlessly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no web search tool the choice is accepted, and applies to nothing.

        Nothing is searched, so the answer is the one the caller asked for and
        the knob is incidental: refusing it would break a client that sets the
        parameter once for every request it sends. The value still never
        travels upstream.

        Ref: stdapi/models/chat/_mantle/_convert.py:responses_payload
             stdapi/models/chat/_adapters/_common.py:resolve_external_web_access
        """
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_allow_external_web_access_override", True
        )
        request = ResponseCreateParams.model_validate(
            {"model": "ignored", "input": "Who won?", "external_web_access": True}
        )

        payload, _ = await mantle_convert.responses_payload(request, "model-id")

        assert "external_web_access" not in payload
        assert "tools" not in payload

    async def test_non_boolean_external_web_access_is_rejected(self) -> None:
        """A value that is not a boolean cannot decide web access.

        Ref: stdapi/models/chat/_adapters/_common.py:resolve_external_web_access
        """
        with pytest.raises(ApiError, match="must be a boolean") as exc_info:
            await mantle_convert.responses_payload(
                self._web_search_request(external_web_access="yes"), "model-id"
            )
        assert exc_info.value.status == 400

    @pytest.mark.parametrize("configured", [False, True])
    async def test_external_web_access_request_rejected_when_override_disabled(
        self, monkeypatch: pytest.MonkeyPatch, configured: bool
    ) -> None:
        """Asking for external web access the server forbids is refused, not dropped.

        Silently answering from the cached index would hide that the request left the
        configured data-governance boundary unchanged, so the explicit ask is rejected.
        A server that allows web access refuses the reverse ask the same way: a client
        cannot quietly opt back into the boundary it was not given.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html
             stdapi/models/chat/_adapters/_common.py:resolve_external_web_access
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_external_web_access", configured)
        with pytest.raises(ApiError, match="external_web_access") as exc_info:
            await mantle_convert.responses_payload(
                self._web_search_request(external_web_access=not configured), "model-id"
            )
        assert exc_info.value.status == 400
        assert f"set it to {str(configured).lower()}" in str(exc_info.value)
        assert ("enabled" if configured else "disabled") in str(exc_info.value)

    @staticmethod
    def _nested_web_search_request(
        item_type: str = "additional_tools",
        *,
        external_web_access: object = None,
        tool_type: str = "web_search",
    ) -> ResponseCreateParams:
        """Build a Responses request carrying a web search tool inside an input item."""
        item: dict[str, Any] = {"type": item_type, "tools": [{"type": tool_type}]}
        if item_type == "additional_tools":
            item["role"] = "user"
        body: dict[str, Any] = {
            "model": "ignored",
            "input": [item, {"role": "user", "content": "Who won?"}],
        }
        if external_web_access is not None:
            body["external_web_access"] = external_web_access
        return ResponseCreateParams.model_validate(body)

    @pytest.mark.parametrize("item_type", ["additional_tools", "tool_search_output"])
    async def test_external_web_access_resolved_on_tool_inside_input_item(
        self, item_type: str
    ) -> None:
        """A nested tool that omits the field is pinned to the configured value.

        The field is optional upstream and defaults to enabled there, so leaving it
        unset must not hand the nested tool more access than the operator allows.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html
             stdapi/models/chat/_mantle/_convert.py:_apply_nested_external_web_access
        """
        payload, _ = await mantle_convert.responses_payload(
            self._nested_web_search_request(item_type), "model-id"
        )
        assert payload["input"][0]["tools"][0]["external_web_access"] is False

    @staticmethod
    def _tool_choice_web_search_request(
        *, tool_type: str = "web_search"
    ) -> ResponseCreateParams:
        """Build a Responses request referencing a web search tool in ``tool_choice``."""
        return ResponseCreateParams.model_validate(
            {
                "model": "ignored",
                "input": "Who won?",
                "tool_choice": {
                    "type": "allowed_tools",
                    "mode": "auto",
                    "tools": [{"type": tool_type}],
                },
            }
        )

    async def test_external_web_access_request_matching_configuration_accepted(
        self,
    ) -> None:
        """Asking for exactly what the server does is accepted, override or not.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html
             stdapi/models/chat/_adapters/_common.py:resolve_external_web_access
        """
        payload, _ = await mantle_convert.responses_payload(
            self._web_search_request(external_web_access=False), "model-id"
        )
        assert payload["tools"][0]["external_web_access"] is False

    async def test_external_web_access_request_wins_when_override_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the override allowed, the request decides in both directions.

        Ref: stdapi/config.py:Settings.aws_bedrock_allow_external_web_access_override
             stdapi/models/chat/_adapters/_common.py:resolve_external_web_access
        """
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_allow_external_web_access_override", True
        )
        payload, _ = await mantle_convert.responses_payload(
            self._web_search_request(external_web_access=True), "model-id"
        )
        assert payload["tools"][0]["external_web_access"] is True

        monkeypatch.setattr(SETTINGS, "aws_bedrock_external_web_access", True)
        payload, _ = await mantle_convert.responses_payload(
            self._web_search_request(external_web_access=False), "model-id"
        )
        assert payload["tools"][0]["external_web_access"] is False

    @pytest.mark.parametrize("item_type", ["additional_tools", "tool_search_output"])
    async def test_external_web_access_nested_request_wins_when_override_allowed(
        self, monkeypatch: pytest.MonkeyPatch, item_type: str
    ) -> None:
        """The override reaches the nested carriers, not only the top-level tools.

        Those carriers are where a replayed conversation echoes its own tool
        definitions back, so a gate that sanitized them unconditionally would
        reject every replay on a server that deliberately allows the override.

        Ref: stdapi/config.py:Settings.aws_bedrock_allow_external_web_access_override
             stdapi/models/chat/_mantle/_convert.py:_apply_nested_external_web_access
        """
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_allow_external_web_access_override", True
        )
        payload, _ = await mantle_convert.responses_payload(
            self._nested_web_search_request(item_type, external_web_access=True),
            "model-id",
        )
        assert payload["input"][0]["tools"][0]["external_web_access"] is True

    async def test_tool_choice_reference_is_forwarded_as_sent(self) -> None:
        """A ``tool_choice`` entry that omits the field keeps the shape it was sent in.

        ``allowed_tools`` narrows the set declared in ``tools``; its entries are
        references, not definitions, so writing ``external_web_access`` into one
        would add a field that grants nothing to a tool the request never
        defined that way.

        Ref: openai.types.responses.tool_choice_allowed.ToolChoiceAllowed
             stdapi/models/chat/_mantle/_convert.py:_apply_nested_external_web_access
        """
        payload, _ = await mantle_convert.responses_payload(
            self._tool_choice_web_search_request(), "model-id"
        )
        assert payload["tool_choice"]["tools"] == [{"type": "web_search"}]

    @pytest.mark.parametrize("tool_type", _WEB_SEARCH_TOOL_TYPES)
    async def test_external_web_access_pinned_for_every_tool_spelling(
        self, tool_type: str
    ) -> None:
        """Each spelling of the tool is pinned to the configured value.

        The type is versioned upstream and has a preview twin, so a gate keyed on
        the bare ``web_search`` alone would hand every other spelling the upstream
        default, which is external access enabled.

        Ref: openai.types.responses.web_search_tool.WebSearchTool
             openai.types.responses.web_search_preview_tool.WebSearchPreviewTool
             stdapi/models/chat/_mantle/_convert.py:responses_payload
             stdapi/models/chat/_mantle/_convert.py:_web_search_tools
        """
        payload, _ = await mantle_convert.responses_payload(
            self._web_search_request(tool_type=tool_type), "model-id"
        )
        assert payload["tools"][0]["external_web_access"] is False
        payload, _ = await mantle_convert.responses_payload(
            self._nested_web_search_request(tool_type=tool_type), "model-id"
        )
        assert payload["input"][0]["tools"][0]["external_web_access"] is False

    @pytest.mark.parametrize(
        "build",
        [_web_search_request, _nested_web_search_request],
        ids=["tools", "input_item"],
    )
    async def test_external_web_access_rejected_wherever_the_tool_travels(
        self, build: Callable[..., ResponseCreateParams]
    ) -> None:
        """The gate answers before any carrier is served.

        The gate is what keeps a request from reaching web access the operator
        forbade, so it must not depend on where the tool it applies to travels.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html
             stdapi/models/chat/_adapters/_common.py:resolve_external_web_access
        """
        with pytest.raises(ApiError, match="external_web_access") as exc_info:
            await mantle_convert.responses_payload(
                build(external_web_access=True), "model-id"
            )
        assert exc_info.value.status == 400

    async def test_undecodable_previous_response_id_raises_api_error(self) -> None:
        """A non-Mantle ``previous_response_id`` fails with a 400 ApiError.

        Only IDs this gateway minted carry the region tag; forwarding an unknown ID would
        reach an arbitrary Region and fail upstream with a confusing error.

        Ref: https://developers.openai.com/api/docs/guides/conversation-state#passing-context-from-the-previous-response
             stdapi/models/chat/_mantle/_convert.py:_pin_previous_response
        """
        request = ResponseCreateParams.model_validate(
            {"model": "ignored", "previous_response_id": "resp_@@@invalid@@@"}
        )
        with pytest.raises(
            ApiError, match="only responses created by this server"
        ) as exc_info:
            await mantle_convert.responses_payload(request, "model-id")
        assert exc_info.value.status == 400

    async def test_null_previous_response_id_removed_from_payload(self) -> None:
        """An explicitly-null ``previous_response_id`` is dropped, not forwarded.

        An explicit ``null`` is not the same as an absent field upstream, so a falsy value
        is popped instead of being serialized.

        Ref: stdapi/models/chat/_mantle/_convert.py:_pin_previous_response
        """
        request = ResponseCreateParams.model_validate(
            {"model": "ignored", "previous_response_id": "resp-local", "input": "hi"}
        ).model_copy(update={"previous_response_id": None})
        payload, pinned_region = await mantle_convert.responses_payload(
            request, "model-id"
        )
        assert "previous_response_id" not in payload
        assert pinned_region is None

    async def test_named_tool_choice_forwarded_verbatim(self) -> None:
        """A named-function ``tool_choice`` reaches the upstream payload unchanged.

        This is the passthrough path, so the flat Responses shape must survive untouched.

        Ref: stdapi/models/chat/_mantle/_convert.py:responses_payload
        """
        request = ResponseCreateParams.model_validate(
            {
                "model": "ignored",
                "input": "hi",
                "tools": [{"type": "function", "name": "get_weather"}],
                "tool_choice": {"type": "function", "name": "get_weather"},
            }
        )
        payload, _region = await mantle_convert.responses_payload(request, "model-id")
        assert payload["tool_choice"] == {"type": "function", "name": "get_weather"}

    async def test_reasoning_mode_forwarded_verbatim(self) -> None:
        """``reasoning.mode`` reaches the upstream payload unchanged.

        ``mode`` is honored only by models that support it, and ignored
        otherwise -- but on the Mantle passthrough path it is always forwarded:
        ``responses_payload`` dumps the request with ``exclude_unset=True`` and
        only strips ``_RESPONSES_EXTENSION_FIELDS`` (``moderation``), so ``mode``
        reaches the upstream ``reasoning`` object like every other unrecognized
        field, leaving the model to honor or ignore it.

        Ref: stdapi/models/chat/_mantle/_convert.py:responses_payload
             stdapi/types/openai_responses.py:Reasoning.mode
        """
        request = ResponseCreateParams.model_validate(
            {
                "model": "ignored",
                "input": "hi",
                "reasoning": {"effort": "medium", "mode": "pro"},
            }
        )
        payload, _region = await mantle_convert.responses_payload(request, "model-id")
        assert payload["reasoning"] == {"effort": "medium", "mode": "pro"}


class TestEnableStreamUsage:
    """Forcing streaming with usage reporting on upstream request payloads.

    Ref: stdapi/models/chat/_mantle/_convert.py:enable_stream_usage
    """

    def test_chat_merges_existing_stream_options(self) -> None:
        """Existing ``stream_options`` keys are preserved alongside include_usage.

        Usage is always requested because the gateway meters the request even when the
        caller did not ask for token counts.

        Ref: stdapi/models/chat/_mantle/_convert.py:enable_stream_usage
        """
        payload = {"stream_options": {"foo": "bar"}}
        out = mantle_convert.enable_stream_usage("chat_completions", payload)
        assert out["stream"] is True
        assert out["stream_options"] == {"foo": "bar", "include_usage": True}

    def test_chat_creates_stream_options_when_absent(self) -> None:
        """``stream_options`` is created when the payload has none.

        Ref: stdapi/models/chat/_mantle/_convert.py:enable_stream_usage
        """
        out = mantle_convert.enable_stream_usage("chat_completions", {})
        assert out["stream_options"] == {"include_usage": True}

    def test_non_chat_api_only_forces_stream(self) -> None:
        """Non-chat APIs get ``stream: True`` without a ``stream_options`` field.

        Responses and Messages report usage in their terminal event unconditionally, so
        they have no ``stream_options`` equivalent to set.

        Ref: stdapi/models/chat/_mantle/_convert.py:enable_stream_usage
        """
        out = mantle_convert.enable_stream_usage("responses", {"model": "m"})
        assert out["stream"] is True
        assert "stream_options" not in out


# ---------------------------------------------------------------------------
# 8. Legacy text completions
# ---------------------------------------------------------------------------


class TestTextCompletionAsChatPayload:
    """Legacy completion requests converted to a Chat Completions payload.

    Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
         stdapi/models/chat/_mantle/_convert.py:text_completion_as_chat_payload
    """

    async def test_string_prompt_copies_supported_fields(self) -> None:
        """A string prompt and supported sampling fields are copied through.

        Ref: stdapi/models/chat/_mantle/_convert.py:text_completion_as_chat_payload
        """
        request = LegacyCompletionCreateParams.model_validate(
            {
                "model": "ignored",
                "prompt": "Hello",
                "max_tokens": 50,
                "stop": ["END"],
                "temperature": 0.5,
                "user": "u1",
            }
        )
        payload = await mantle_convert.text_completion_as_chat_payload(
            request, "model-id"
        )
        assert payload["model"] == "model-id"
        assert payload["messages"] == [{"role": "user", "content": "Hello"}]
        assert payload["max_tokens"] == 50
        assert payload["stop"] == ["END"]
        assert payload["temperature"] == 0.5
        assert payload["user"] == "u1"

    async def test_one_element_list_prompt_unwrapped(self) -> None:
        """A single-element prompt list is unwrapped to its string element.

        The legacy API accepts a batch of prompts; a one-element batch is equivalent to a
        single prompt and needs no multi-choice handling.

        Ref: stdapi/models/chat/_mantle/_convert.py:text_completion_as_chat_payload
        """
        request = LegacyCompletionCreateParams.model_validate(
            {"model": "ignored", "prompt": ["Solo"]}
        )
        payload = await mantle_convert.text_completion_as_chat_payload(
            request, "model-id"
        )
        assert payload["messages"] == [{"role": "user", "content": "Solo"}]

    async def test_n_is_copied_through_without_rejection(self) -> None:
        """``n`` is copied unchanged; only the target-API conversion rejects n>1.

        Mantle Chat Completions models can serve several choices, so the ``n>1`` rejection
        belongs to the Responses and Messages converters, not to this payload builder.

        Ref: stdapi/models/chat/_mantle/_convert.py:text_completion_as_chat_payload
             stdapi/models/chat/_mantle/_convert.py:_ensure_single_choice
        """
        request = LegacyCompletionCreateParams.model_validate(
            {"model": "ignored", "prompt": "Hi", "n": 2}
        )
        payload = await mantle_convert.text_completion_as_chat_payload(
            request, "model-id"
        )
        assert payload["n"] == 2

    @pytest.mark.parametrize(
        ("field", "value"), [("echo", True), ("suffix", "S"), ("logprobs", 1)]
    )
    async def test_unsupported_option_rejected(self, field: str, value: object) -> None:
        """``echo``, ``suffix`` and ``logprobs`` are rejected with a 400 ApiError.

        None of the three has a Chat Completions equivalent, and silently dropping them
        would change the response shape the caller expects.

        Ref: stdapi/models/chat/_mantle/_convert.py:text_completion_as_chat_payload
        """
        request = LegacyCompletionCreateParams.model_validate(
            {"model": "ignored", "prompt": "Hi", field: value}
        )
        with pytest.raises(ApiError, match=f"`{field}` is not supported") as exc_info:
            await mantle_convert.text_completion_as_chat_payload(request, "model-id")
        assert exc_info.value.status == 400

    async def test_multi_prompt_list_rejected(self) -> None:
        """A multi-element prompt list is rejected: only one prompt is supported.

        One upstream call serves one prompt; the legacy multi-prompt batch would need a
        fan-out this passthrough path does not implement.

        Ref: stdapi/models/chat/_mantle/_convert.py:text_completion_as_chat_payload
        """
        request = LegacyCompletionCreateParams.model_validate(
            {"model": "ignored", "prompt": ["a", "b"]}
        )
        with pytest.raises(ApiError, match="Multiple prompts") as exc_info:
            await mantle_convert.text_completion_as_chat_payload(request, "model-id")
        assert exc_info.value.status == 400

    async def test_url_prompt_rejected_as_file_prompt(self) -> None:
        """A non-inline (URL) prompt is rejected: file prompts are unsupported.

        The request model parses a non-inline prompt into an ``InputFileUrl`` rather than a
        string, and a file cannot be used as a Chat Completions prompt.

        Ref: stdapi/types/openai_completions.py:CompletionCreateParams.prompt
             stdapi/models/chat/_mantle/_convert.py:text_completion_as_chat_payload
        """
        request = LegacyCompletionCreateParams.model_validate(
            {"model": "ignored", "prompt": "https://example.com/prompt.txt"}
        )
        assert not isinstance(request.prompt, str), (
            "a URL prompt must not validate as inline text"
        )
        with pytest.raises(ApiError, match="File prompts") as exc_info:
            await mantle_convert.text_completion_as_chat_payload(request, "model-id")
        assert exc_info.value.status == 400


class TestChatResponseAsTextCompletion:
    """Chat Completions responses converted to the legacy ``Completion`` shape.

    Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
         stdapi/models/chat/_mantle/_convert.py:chat_response_as_text_completion
    """

    def test_tool_calls_finish_reason_maps_to_stop_and_usage_extras_filtered(
        self,
    ) -> None:
        """A ``tool_calls`` finish maps to ``stop``; unknown usage keys are dropped.

        The legacy shape has no tool-call finish reason, so ``tool_calls`` collapses onto
        ``stop``. Usage is filtered through ``CompletionUsage``'s fields so provider extras
        cannot leak into the response model.

        Ref: stdapi/models/chat/_mantle/_convert.py:_text_finish
             stdapi/models/chat/_mantle/_convert.py:chat_response_as_text_completion
        """
        raw = {
            "id": "chatcmpl-1",
            "created": 100,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "tool_calls",
                    "logprobs": None,
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
                "cache_read_input_tokens": 99,
            },
        }
        result = mantle_convert.chat_response_as_text_completion(raw, "cmpl-1")
        assert result.id == "cmpl-1"
        assert result.object == "text_completion"
        assert result.choices[0].text == "hi"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 2
        assert result.usage.total_tokens == 7
        assert "cache_read_input_tokens" not in result.usage.model_dump()


class TestChatStreamAsTextCompletionCompact:
    """SSE wrapper converting a Chat Completions stream to text-completion chunks.

    Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
         stdapi/models/chat/_mantle/_convert.py:chat_stream_as_text_completion
    """

    async def test_done_and_non_string_data_pass_through(self) -> None:
        """The ``[DONE]`` sentinel and a non-string ``data`` event pass through.

        ``[DONE]`` is the stream terminator and must not be parsed as a chunk; an event
        carrying no data has nothing to convert.

        Ref: stdapi/models/chat/_mantle/_convert.py:chat_stream_as_text_completion
        """

        async def source() -> AsyncGenerator[ServerSentEvent]:
            yield ServerSentEvent(data="[DONE]")
            yield ServerSentEvent()

        events = [
            event
            async for event in mantle_convert.chat_stream_as_text_completion(
                source(), "cmpl-1"
            )
        ]
        assert [event.data for event in events] == ["[DONE]", None]

    async def test_unnamed_error_chunk_raises_mantle_error(self) -> None:
        """An unnamed in-band error chunk aborts the stream instead of being dropped.

        Ref: stdapi/models/chat/_mantle/_convert.py:_stream_error_message
             stdapi/aws_bedrock_mantle.py:MantleError
        """

        async def source() -> AsyncGenerator[ServerSentEvent]:
            yield ServerSentEvent(data=dumps({"error": {"message": "boom problem"}}))

        stream = mantle_convert.chat_stream_as_text_completion(source(), "cmpl-1")
        with pytest.raises(MantleError, match="boom problem") as exc_info:
            async for _ in stream:
                pass
        assert exc_info.value.status == 502

    async def test_named_error_event_still_passes_through_unchanged(self) -> None:
        """A named error event (e.g. relayed) is passed through, not raised on.

        A named ``error`` event is already a relayed upstream error frame: raising on it
        would replace a faithful passthrough with a synthesised 502.

        Ref: stdapi/models/chat/_mantle/_convert.py:chat_stream_as_text_completion
        """

        async def source() -> AsyncGenerator[ServerSentEvent]:
            yield ServerSentEvent(
                data=dumps({"error": {"message": "boom"}}), event="error"
            )

        events = [
            event
            async for event in mantle_convert.chat_stream_as_text_completion(
                source(), "cmpl-1"
            )
        ]
        assert len(events) == 1
        assert events[0].event == "error"


# ---------------------------------------------------------------------------
# 9. Small helpers and dispatch/composition
# ---------------------------------------------------------------------------


class TestJsonObjectHelper:
    """Tool-arguments JSON string parsing.

    Tool-call arguments arrive as a JSON string that a model may leave empty or
    truncated, so parsing never raises: the Anthropic and Responses shapes both need
    an object.

    Ref: stdapi/models/chat/_mantle/_convert.py:_json_object
    """

    def test_invalid_json_returns_empty_dict(self) -> None:
        """Malformed JSON falls back to an empty object."""
        assert mantle_convert._json_object("not json") == {}  # noqa: SLF001

    def test_non_object_json_returns_empty_dict(self) -> None:
        """A valid but non-object JSON value falls back to an empty object."""
        assert mantle_convert._json_object("[1, 2, 3]") == {}  # noqa: SLF001

    def test_empty_string_returns_empty_dict(self) -> None:
        """An empty arguments string falls back to an empty object."""
        assert mantle_convert._json_object("") == {}  # noqa: SLF001

    def test_valid_object_parsed(self) -> None:
        """A valid JSON object string parses as-is."""
        assert mantle_convert._json_object('{"a": 1}') == {"a": 1}  # noqa: SLF001


class TestSplitDataUri:
    """Base64 ``data:`` URI splitting.

    Ref: stdapi/models/chat/_mantle/_convert.py:_split_data_uri
    """

    def test_non_data_uri_returns_none(self) -> None:
        """A plain URL is not a data URI."""
        assert mantle_convert._split_data_uri("https://example.com/x.png") is None  # noqa: SLF001

    def test_data_uri_without_base64_marker_returns_none(self) -> None:
        """A data URI without ``;base64,`` is not recognized."""
        assert mantle_convert._split_data_uri("data:text/plain,hello") is None  # noqa: SLF001

    def test_parameterized_media_type_is_stripped(self) -> None:
        """Media type parameters (e.g. ``;charset=``) are stripped from the type."""
        uri = "data:text/plain;charset=utf-8;base64,QUJD"
        assert mantle_convert._split_data_uri(uri) == ("text/plain", "QUJD")  # noqa: SLF001


class TestTextExtractors:
    """Plain-text extraction from the three wire content shapes.

    Ref: stdapi/models/chat/_mantle/_convert.py:_chat_text
    """

    def test_chat_text_from_string(self) -> None:
        """A plain string is returned as-is.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_text
        """
        assert mantle_convert._chat_text("hi") == "hi"  # noqa: SLF001

    def test_chat_text_from_part_list(self) -> None:
        """Text and refusal parts concatenate.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_text
        """
        parts = [{"type": "text", "text": "a"}, {"type": "refusal", "refusal": "b"}]
        assert mantle_convert._chat_text(parts) == "ab"  # noqa: SLF001

    def test_chat_text_from_none(self) -> None:
        """``None`` content yields an empty string.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_text
        """
        assert mantle_convert._chat_text(None) == ""  # noqa: SLF001

    def test_anthropic_text_from_string(self) -> None:
        """A plain string is returned as-is.

        Ref: stdapi/models/chat/_mantle/_convert.py:_anthropic_text
        """
        assert mantle_convert._anthropic_text("hi") == "hi"  # noqa: SLF001

    def test_anthropic_text_from_block_list(self) -> None:
        """Only ``text`` blocks contribute; other block types are ignored.

        Ref: stdapi/models/chat/_mantle/_convert.py:_anthropic_text
        """
        blocks = [{"type": "text", "text": "a"}, {"type": "tool_use", "id": "x"}]
        assert mantle_convert._anthropic_text(blocks) == "a"  # noqa: SLF001

    def test_anthropic_text_from_none(self) -> None:
        """``None`` content yields an empty string.

        Ref: stdapi/models/chat/_mantle/_convert.py:_anthropic_text
        """
        assert mantle_convert._anthropic_text(None) == ""  # noqa: SLF001

    def test_responses_text_from_string(self) -> None:
        """A plain string is returned as-is.

        Ref: stdapi/models/chat/_mantle/_convert.py:_responses_text
        """
        assert mantle_convert._responses_text("hi") == "hi"  # noqa: SLF001

    def test_responses_text_from_part_list(self) -> None:
        """Textual part types concatenate.

        Ref: stdapi/models/chat/_mantle/_convert.py:_responses_text
        """
        parts = [
            {"type": "input_text", "text": "a"},
            {"type": "output_text", "text": "b"},
        ]
        assert mantle_convert._responses_text(parts) == "ab"  # noqa: SLF001

    def test_responses_text_from_none(self) -> None:
        """``None`` content yields an empty string.

        Ref: stdapi/models/chat/_mantle/_convert.py:_responses_text
        """
        assert mantle_convert._responses_text(None) == ""  # noqa: SLF001


class TestEnsureSingleChoice:
    """Multi-choice request rejection for single-choice-only APIs.

    Responses and Anthropic Messages have no ``n``: a multi-choice request would
    silently return a single choice, so it is rejected up front.

    Ref: stdapi/models/chat/_mantle/_convert.py:_ensure_single_choice
    """

    def test_n_greater_than_one_raises(self) -> None:
        """``n=2`` is rejected with a 400 ApiError naming the unsupported option."""
        with pytest.raises(ApiError, match=r"Multiple choices \(n>1\)") as exc_info:
            mantle_convert._ensure_single_choice({"n": 2})  # noqa: SLF001
        assert exc_info.value.status == 400

    def test_n_one_or_absent_does_not_raise(self) -> None:
        """``n=1`` and a missing ``n`` are accepted and leave the payload untouched.

        The guard is a pure precondition check: an accepted payload must come back
        byte-for-byte identical, so it can be handed straight to the converter.
        """
        single: dict[str, Any] = {"model": "m", "n": 1}
        absent: dict[str, Any] = {"model": "m"}
        mantle_convert._ensure_single_choice(single)  # noqa: SLF001
        mantle_convert._ensure_single_choice(absent)  # noqa: SLF001
        assert single == {"model": "m", "n": 1}
        assert absent == {"model": "m"}


class TestSanitizeToolSchema:
    """Recursive stripping of unsupported JSON Schema keywords.

    Some open-weight tool templates emit an empty generation when a schema contains
    ``propertyNames``, so it is stripped in place at any depth.

    Ref: stdapi/models/chat/_mantle/_convert.py:sanitize_tool_schema
    """

    def test_removes_property_names_recursively_in_nested_dict_and_anyof_list(
        self,
    ) -> None:
        """``propertyNames`` is stripped at any nesting depth, including in a list."""
        schema = {
            "type": "object",
            "properties": {"a": {"propertyNames": {"pattern": "^x"}}},
            "anyOf": [{"propertyNames": {}}, {"type": "string"}],
        }
        result = mantle_convert.sanitize_tool_schema(schema)
        assert result is schema
        assert "propertyNames" not in result["properties"]["a"]
        assert "propertyNames" not in result["anyOf"][0]
        assert result["anyOf"][1] == {"type": "string"}


class TestStreamErrorMessageHelper:
    """In-band stream error message extraction.

    The helper decides whether an in-band SSE frame aborts the stream, so anything that
    is not recognisably an error must return ``None``.

    Ref: stdapi/models/chat/_mantle/_convert.py:_stream_error_message
    """

    def test_non_json_returns_none(self) -> None:
        """Non-JSON data carries no error message."""
        assert mantle_convert._stream_error_message("not json") is None  # noqa: SLF001

    def test_scalar_json_returns_none(self) -> None:
        """A scalar JSON value (not an object) carries no error message."""
        assert mantle_convert._stream_error_message("42") is None  # noqa: SLF001

    def test_non_error_dict_returns_none(self) -> None:
        """A JSON object without an ``error`` field and non-error type is not an error."""
        payload = dumps({"foo": "bar"})
        assert mantle_convert._stream_error_message(payload) is None  # noqa: SLF001


class TestIdentityDispatchAndComposition:
    """Same-shape conversions are no-ops; cross-shape ones compose through chat.

    Ref: stdapi/models/chat/_mantle/_convert.py:convert_payload
    """

    def test_convert_payload_same_api_returns_same_object(self) -> None:
        """A same-shape payload conversion returns the identical object.

        Ref: stdapi/models/chat/_mantle/_convert.py:convert_payload
        """
        payload = {"a": 1}
        result = mantle_convert.convert_payload(
            "chat_completions", "chat_completions", payload
        )
        assert result is payload

    def test_convert_response_same_api_returns_same_object(self) -> None:
        """A same-shape response conversion returns the identical object.

        Ref: stdapi/models/chat/_mantle/_convert.py:convert_response
        """
        raw = {"a": 1}
        result = mantle_convert.convert_response("responses", "responses", raw)
        assert result is raw

    def test_convert_stream_same_api_returns_same_generator(self) -> None:
        """A same-shape stream conversion returns the identical generator.

        Returning the caller's generator unchanged keeps a passthrough stream free of an
        extra async wrapper.

        Ref: stdapi/models/chat/_mantle/_convert.py:convert_stream
        """
        gen = _agen([])
        result = mantle_convert.convert_stream("messages", "messages", gen)
        assert result is gen

    def test_messages_to_responses_composes_through_chat(self) -> None:
        """A Messages payload converts to Responses via the Chat Completions shape.

        There is no direct Messages-to-Responses converter: the pair is composed from the
        two Chat Completions converters, so the intermediate shape's losses apply.

        Ref: stdapi/models/chat/_mantle/_convert.py:convert_payload
        """
        payload = {
            "model": "m",
            "system": "Be terse.",
            "messages": [{"role": "user", "content": "Hi there"}],
            "max_tokens": 100,
        }
        out = mantle_convert.convert_payload("messages", "responses", payload)
        assert out["instructions"] == "Be terse."
        assert out["input"] == [{"role": "user", "content": "Hi there"}]


# ---------------------------------------------------------------------------
# 10. Documented drop lists
# ---------------------------------------------------------------------------


class TestChatToResponsesDropList:
    """Chat Completions fields without a Responses equivalent are dropped.

    Silently dropping is deliberate: forwarding a field Responses does not define
    would make upstream reject an otherwise valid request.

    Ref: https://developers.openai.com/api/docs/guides/migrate-to-responses
         stdapi/models/chat/_mantle/_convert.py:_chat_to_responses_request
    """

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("seed", 42),
            ("presence_penalty", 0.5),
            ("frequency_penalty", 0.3),
            ("logit_bias", {"123": 1}),
            ("audio", {"voice": "x"}),
            ("modalities", ["text", "audio"]),
        ],
    )
    def test_field_dropped(self, field: str, value: object) -> None:
        """The documented unmapped field never appears in the Responses payload."""
        out = mantle_convert._chat_to_responses_request(  # noqa: SLF001
            {"model": "m", "messages": [], field: value}
        )
        assert field not in out


class TestChatToMessagesDropList:
    """Chat Completions fields without an Anthropic equivalent are dropped.

    Silently dropping is deliberate: forwarding a field Anthropic does not define
    would make upstream reject an otherwise valid request.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/models/chat/_mantle/_convert.py:_chat_to_messages_request
    """

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("seed", 42),
            ("presence_penalty", 0.5),
            ("frequency_penalty", 0.3),
            ("logit_bias", {"123": 1}),
            ("response_format", {"type": "json_object"}),
            ("metadata", {"k": "v"}),
        ],
    )
    def test_field_dropped(self, field: str, value: object) -> None:
        """The documented unmapped field never appears in the Anthropic payload."""
        out = mantle_convert._chat_to_messages_request(  # noqa: SLF001
            {"model": "m", "messages": [], field: value}
        )
        assert field not in out


class TestMessagesToChatDropList:
    """Anthropic fields without a Chat Completions equivalent are dropped.

    Silently dropping is deliberate: forwarding a field Chat Completions does not
    define would make upstream reject an otherwise valid request.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
         stdapi/models/chat/_mantle/_convert.py:_messages_to_chat_request
    """

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("top_k", 5),
            ("thinking", {"type": "enabled", "budget_tokens": 1024}),
            ("container", "abc"),
            ("cache_control", {"type": "ephemeral"}),
        ],
    )
    def test_field_dropped(self, field: str, value: object) -> None:
        """The documented unmapped field never appears in the Chat payload."""
        out = mantle_convert._messages_to_chat_request(  # noqa: SLF001
            {"model": "m", "messages": [], field: value}
        )
        assert field not in out


# ---------------------------------------------------------------------------
# 9. Reasoning text surfacing
# ---------------------------------------------------------------------------


class TestRenameReasoningField:
    """Upstream ``reasoning`` text renamed to the ``reasoning_content`` field.

    Reasoning models served upstream return their chain of thought under
    ``reasoning``; the Chat Completions surface of this gateway exposes it as
    ``reasoning_content``, the DeepSeek-compatible field, so the rename is what
    keeps the text from being pruned out of the validated response.

    Ref: https://api-docs.deepseek.com/api/create-chat-completion
         stdapi/models/chat/_mantle/_convert.py:rename_reasoning_field
    """

    def test_message_and_delta_reasoning_are_both_promoted(self) -> None:
        """The non-streaming message and the streamed delta are both renamed."""
        payload: dict[str, Any] = {
            "choices": [
                {"message": {"role": "assistant", "reasoning": "why"}},
                {"delta": {"reasoning": "wh"}},
            ]
        }

        mantle_convert.rename_reasoning_field(payload)

        assert payload["choices"][0]["message"] == {
            "role": "assistant",
            "reasoning_content": "why",
        }
        assert payload["choices"][1]["delta"] == {"reasoning_content": "wh"}

    def test_an_existing_reasoning_content_wins(self) -> None:
        """A payload already using this surface's name is left as it arrived."""
        payload: dict[str, Any] = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "reasoning": "other",
                        "reasoning_content": "kept",
                    }
                }
            ]
        }

        mantle_convert.rename_reasoning_field(payload)

        assert payload["choices"][0]["message"]["reasoning_content"] == "kept"
        assert payload["choices"][0]["message"]["reasoning"] == "other", (
            "the unknown key is pruned by validation, never merged"
        )

    def test_target_field_renames_toward_the_upstream_name(self) -> None:
        """``field="reasoning"`` moves ``reasoning_content`` the other way.

        The streaming relay passes the operator-configured field name, so the
        rename must work toward either spelling.

        Ref: stdapi/config.py:_Settings.chat_completions_reasoning_field
             stdapi/models/chat/_mantle/_default.py:_rename_stream_reasoning
        """
        payload: dict[str, Any] = {
            "choices": [
                {"delta": {"reasoning_content": "step"}},
                {"delta": {"reasoning": "kept"}},
            ]
        }

        mantle_convert.rename_reasoning_field(payload, field="reasoning")

        assert payload["choices"][0]["delta"] == {"reasoning": "step"}
        assert payload["choices"][1]["delta"] == {"reasoning": "kept"}

    def test_excluding_drops_the_text_under_either_name(self) -> None:
        """``exclude`` removes the chain of thought instead of renaming it.

        Ref: stdapi/types/openai_chat_completions.py:CompletionCreateParams.suppress_reasoning
        """
        payload: dict[str, Any] = {
            "choices": [
                {"message": {"role": "assistant", "content": "hi", "reasoning": "why"}},
                {"delta": {"reasoning_content": "wh"}},
            ]
        }

        mantle_convert.rename_reasoning_field(payload, exclude=True)

        assert payload["choices"][0]["message"] == {
            "role": "assistant",
            "content": "hi",
        }
        assert payload["choices"][1]["delta"] == {}

    def test_a_non_text_reasoning_value_keeps_its_own_name(self) -> None:
        """Only text is promoted; any other shape stays an unknown field.

        ``reasoning_content`` is declared as text, so renaming a structured
        value into it turns a field that was harmlessly pruned into a validation
        failure -- and, mid-stream, into a response that dies after its headers.

        Ref: stdapi/types/openai_chat_completions.py:ChatCompletionMessage
        """
        payload = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        "reasoning": [{"type": "text", "text": "t"}],
                    }
                }
            ]
        }

        mantle_convert.rename_reasoning_field(payload)

        message = payload["choices"][0]["message"]
        assert "reasoning_content" not in message
        assert message["reasoning"] == [{"type": "text", "text": "t"}]

    def test_reasoning_content_is_sent_back_under_the_upstream_name(self) -> None:
        """A replayed assistant turn carries its thinking text upstream again.

        The OpenAI SDK idiom for a follow-up turn is to append the message
        object the API just returned, so the field the gateway emits is the
        field the gateway receives -- and upstream only knows ``reasoning``.

        Ref: stdapi/models/chat/_mantle/_convert.py:_restore_reasoning_field
        """
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": "45",
                "reasoning_content": "Let total be T.",
            },
            {
                "role": "assistant",
                "content": "45",
                "reasoning_content": [{"type": "text", "text": "in parts"}],
            },
        ]

        mantle_convert._restore_reasoning_field({"messages": messages})  # noqa: SLF001

        assert messages[1] == {
            "role": "assistant",
            "content": "45",
            "reasoning": "Let total be T.",
        }
        assert messages[2]["reasoning"] == "in parts", (
            "text split into parts is flattened, since upstream takes one string"
        )
        assert "reasoning_content" not in messages[2]

    def test_non_streaming_message_renamed(self) -> None:
        """``choices[].message.reasoning`` becomes ``reasoning_content``."""
        payload = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": " 45",
                        "reasoning": " Let total be T.",
                    },
                }
            ]
        }
        out = mantle_convert.rename_reasoning_field(payload)
        message = out["choices"][0]["message"]
        assert message["reasoning_content"] == " Let total be T."
        assert "reasoning" not in message

    def test_streaming_delta_renamed(self) -> None:
        """``choices[].delta.reasoning`` becomes ``reasoning_content``."""
        payload = {"choices": [{"index": 0, "delta": {"reasoning": "step"}}]}
        out = mantle_convert.rename_reasoning_field(payload)
        delta = out["choices"][0]["delta"]
        assert delta["reasoning_content"] == "step"
        assert "reasoning" not in delta

    def test_existing_reasoning_content_left_alone(self) -> None:
        """An upstream ``reasoning_content`` is never overwritten by the rename."""
        payload = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        "reasoning": "ignored",
                        "reasoning_content": "kept",
                    },
                }
            ]
        }
        out = mantle_convert.rename_reasoning_field(payload)
        message = out["choices"][0]["message"]
        assert message["reasoning_content"] == "kept"
        assert message["reasoning"] == "ignored"

    def test_payload_without_reasoning_unchanged(self) -> None:
        """A payload carrying no reasoning text is returned untouched."""
        payload = {
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi"}}
            ],
            "usage": {"completion_tokens_details": {"reasoning_tokens": 3}},
        }
        assert mantle_convert.rename_reasoning_field(payload) == {
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi"}}
            ],
            "usage": {"completion_tokens_details": {"reasoning_tokens": 3}},
        }

    def test_malformed_choices_tolerated(self) -> None:
        """Non-object choices and deltas are skipped instead of raising."""
        payload: dict[str, Any] = {"choices": ["oops", {"message": None}]}
        assert mantle_convert.rename_reasoning_field(payload) is payload


class TestChatReasoningToResponses:
    """Chat Completions reasoning text becomes a Responses ``reasoning`` item.

    The item precedes the assistant message, as on the gateway's own Converse
    path, and its ID derives from the response ID like every other output item.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/models/chat/_mantle/_convert.py:_chat_to_responses_response
    """

    def test_reasoning_item_precedes_message(self) -> None:
        """A reasoning item carrying the thinking summary is emitted first."""
        raw = {
            "id": "chatcmpl-abc123",
            "created": 1000,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": " 45",
                        "reasoning_content": " Let total be T.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 70, "completion_tokens": 342},
        }
        out = mantle_convert.convert_response("chat_completions", "responses", raw)
        assert [item["type"] for item in out["output"]] == ["reasoning", "message"]
        reasoning = out["output"][0]
        assert reasoning["id"] == "resp_abc123-rs-0"
        assert reasoning["status"] == "completed"
        assert reasoning["content"] == [
            {"type": "reasoning_text", "text": " Let total be T."}
        ]

    def test_no_reasoning_item_without_reasoning_content(self) -> None:
        """A response without reasoning text produces no reasoning item."""
        raw = {
            "id": "chatcmpl-abc123",
            "model": "m",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi"}}
            ],
        }
        out = mantle_convert.convert_response("chat_completions", "responses", raw)
        assert [item["type"] for item in out["output"]] == ["message"]

    def test_reasoning_not_surfaced_on_messages_conversion(self) -> None:
        """The Anthropic shape carries no thinking block for the reasoning text.

        An Anthropic ``thinking`` block requires a signature that cannot be
        produced from a Chat Completions response, and an invalid one breaks
        replay, so the text is dropped on this conversion.
        """
        raw = {
            "id": "chatcmpl-abc123",
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "hi",
                        "reasoning_content": "thinking",
                    },
                }
            ],
        }
        out = mantle_convert.convert_response("chat_completions", "messages", raw)
        assert out["content"] == [{"type": "text", "text": "hi"}]


class TestChatReasoningToResponsesStream:
    """Chat Completions reasoning deltas become Responses summary events.

    Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
         stdapi/models/chat/_mantle/_convert.py:_responses_reasoning_delta
         stdapi/models/chat/_mantle/_convert.py:_close_responses_reasoning
    """

    def _chunks(self) -> list[SseEvent]:
        """Build CC chunks streaming two reasoning deltas then the answer."""
        return [
            (
                None,
                dumps(
                    {
                        "id": "chatcmpl-1",
                        "created": 100,
                        "model": "m",
                        "choices": [
                            {"index": 0, "delta": {"reasoning_content": "Let "}}
                        ],
                    }
                ),
            ),
            (
                None,
                dumps(
                    {"choices": [{"index": 0, "delta": {"reasoning_content": "T=x."}}]}
                ),
            ),
            (None, dumps({"choices": [{"index": 0, "delta": {"content": "45"}}]})),
            (
                None,
                dumps(
                    {
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 70, "completion_tokens": 342},
                    }
                ),
            ),
        ]

    async def test_reasoning_events_precede_message_events(self) -> None:
        """The reasoning item opens, streams and closes before the message item."""
        events = await _collect(
            mantle_convert.convert_stream(
                "chat_completions", "responses", _agen(self._chunks()), None
            )
        )
        names = _names(events)
        assert "response.content_part.added" in names
        assert names.index("response.reasoning_text.done") < names.index(
            "response.output_text.delta"
        )
        deltas = [
            loads(data)["delta"]
            for name, data in events
            if name == "response.reasoning_text.delta"
        ]
        assert deltas == ["Let ", "T=x."]

    async def test_completed_response_carries_the_reasoning_item(self) -> None:
        """The terminal event's output lists the completed reasoning item first."""
        events = await _collect(
            mantle_convert.convert_stream(
                "chat_completions", "responses", _agen(self._chunks()), "resp_route1"
            )
        )
        completed = next(
            loads(data) for name, data in events if name == "response.completed"
        )
        output = completed["response"]["output"]
        assert [item["type"] for item in output] == ["reasoning", "message"]
        assert output[0]["id"] == "resp_route1-rs-0"
        assert output[0]["content"] == [{"type": "reasoning_text", "text": "Let T=x."}]
        assert output[0]["status"] == "completed"


class TestRefusalResponseConversions:
    """A model refusal survives every non-streaming response conversion.

    An OpenAI-compatible upstream reports a structured-output refusal in
    ``message.refusal`` (Chat Completions) or in a ``refusal`` output content
    part (Responses). Anthropic has no refusal content block, so the text
    becomes a text block and the ``refusal`` stop reason carries the signal.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
         https://developers.openai.com/api/reference/resources/responses/methods/create
         https://platform.claude.com/docs/en/api/messages
         stdapi/models/chat/_mantle/_convert.py:_chat_to_responses_response
         stdapi/models/chat/_mantle/_convert.py:_chat_to_messages_response
         stdapi/models/chat/_mantle/_convert.py:_responses_to_chat_response
    """

    #: Refusal text shared by the conversions under test.
    REFUSAL = "I'm sorry, I can't help with that."

    def _chat_refusal(self) -> dict[str, Any]:
        """Build a Chat Completions response whose only content is a refusal."""
        return {
            "id": "chatcmpl-abc123",
            "created": 1000,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "refusal": self.REFUSAL,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

    def test_chat_refusal_becomes_a_responses_refusal_part(self) -> None:
        """The Responses message item carries a ``refusal`` content part."""
        out = mantle_convert.convert_response(
            "chat_completions", "responses", self._chat_refusal()
        )
        assert out["output"] == [
            {
                "type": "message",
                "id": "resp_abc123-msg-0",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": self.REFUSAL}],
            }
        ]
        assert out["status"] == "completed"

    def test_chat_text_and_refusal_share_one_message_item(self) -> None:
        """Text and refusal parts are both emitted, text first."""
        raw = self._chat_refusal()
        raw["choices"][0]["message"]["content"] = "Here is what I can say."
        out = mantle_convert.convert_response("chat_completions", "responses", raw)
        assert [part["type"] for part in out["output"][0]["content"]] == [
            "output_text",
            "refusal",
        ]

    def test_chat_refusal_becomes_an_anthropic_text_block_and_stop_reason(self) -> None:
        """The Anthropic content is non-empty and the stop reason is ``refusal``."""
        out = mantle_convert.convert_response(
            "chat_completions", "messages", self._chat_refusal()
        )
        assert out["content"] == [{"type": "text", "text": self.REFUSAL}]
        assert out["stop_reason"] == "refusal"

    def test_responses_refusal_part_becomes_the_chat_refusal_field(self) -> None:
        """A Responses refusal part lands in ``message.refusal``, not in the content."""
        raw = {
            "id": "resp_abc123",
            "created_at": 1000,
            "model": "m",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "refusal", "refusal": self.REFUSAL}],
                }
            ],
        }
        out = mantle_convert.convert_response("responses", "chat_completions", raw)
        message = out["choices"][0]["message"]
        assert message["refusal"] == self.REFUSAL
        assert message["content"] is None

    def test_responses_refusal_round_trips_through_the_chat_pivot(self) -> None:
        """Responses to Anthropic composes both halves without losing the refusal."""
        raw = {
            "id": "resp_abc123",
            "model": "m",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "refusal", "refusal": self.REFUSAL}],
                }
            ],
        }
        out = mantle_convert.convert_response("responses", "messages", raw)
        assert out["content"] == [{"type": "text", "text": self.REFUSAL}]
        assert out["stop_reason"] == "refusal"


class TestRefusalStreamConversions:
    """A model refusal survives every streaming response conversion.

    Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
         https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
         stdapi/models/chat/_mantle/_convert.py:_responses_refusal_delta
         stdapi/models/chat/_mantle/_convert.py:_close_responses_refusal
         stdapi/models/chat/_mantle/_convert.py:_messages_chunk_events
         stdapi/models/chat/_mantle/_convert.py:_responses_stream_to_chat
    """

    def _chunks(self) -> list[SseEvent]:
        """Build CC chunks streaming a refusal in two deltas then a finish chunk."""
        return [
            (
                None,
                dumps(
                    {
                        "id": "chatcmpl-1",
                        "created": 100,
                        "model": "m",
                        "choices": [{"index": 0, "delta": {"refusal": "I'm sorry, "}}],
                    }
                ),
            ),
            (
                None,
                dumps({"choices": [{"index": 0, "delta": {"refusal": "I can't."}}]}),
            ),
            (
                None,
                dumps(
                    {
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    }
                ),
            ),
        ]

    async def test_refusal_deltas_become_responses_refusal_events(self) -> None:
        """The message item opens with a refusal part and streams refusal deltas."""
        events = await _collect(
            mantle_convert.convert_stream(
                "chat_completions", "responses", _agen(self._chunks()), "resp_route1"
            )
        )
        names = _names(events)
        assert "response.refusal.delta" in names
        assert names.index("response.refusal.done") < names.index("response.completed")
        deltas = [
            loads(data)["delta"]
            for name, data in events
            if name == "response.refusal.delta"
        ]
        assert deltas == ["I'm sorry, ", "I can't."]
        added = next(
            loads(data)
            for name, data in events
            if name == "response.content_part.added"
        )
        assert added["part"] == {"type": "refusal", "refusal": ""}

    async def test_completed_response_carries_the_refusal_part(self) -> None:
        """The terminal event's output lists the message item with its refusal part."""
        events = await _collect(
            mantle_convert.convert_stream(
                "chat_completions", "responses", _agen(self._chunks()), "resp_route1"
            )
        )
        completed = next(
            loads(data) for name, data in events if name == "response.completed"
        )
        assert completed["response"]["output"] == [
            {
                "type": "message",
                "id": "resp_route1-msg-0",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "refusal", "refusal": "I'm sorry, I can't."}],
            }
        ]

    async def test_refusal_deltas_become_anthropic_text_and_refusal_stop(self) -> None:
        """The Anthropic stream emits a text block and stops with ``refusal``."""
        events = await _collect(
            mantle_convert.convert_stream(
                "chat_completions", "messages", _agen(self._chunks())
            )
        )
        texts = [
            payload["delta"]["text"]
            for name, payload in zip(_names(events), _payloads(events), strict=True)
            if name == "content_block_delta"
        ]
        assert texts == ["I'm sorry, ", "I can't."]
        message_delta = _payloads(events)[-2]
        assert message_delta["delta"]["stop_reason"] == "refusal"

    async def test_responses_refusal_deltas_become_chat_refusal_deltas(self) -> None:
        """``response.refusal.delta`` events convert to ``delta.refusal`` chunks."""
        events: list[SseEvent] = [
            (
                "response.created",
                dumps(
                    {
                        "type": "response.created",
                        "response": {"id": "resp_1", "created_at": 100, "model": "m"},
                    }
                ),
            ),
            (
                "response.refusal.delta",
                dumps({"type": "response.refusal.delta", "delta": "I can't."}),
            ),
            (
                "response.completed",
                dumps(
                    {
                        "type": "response.completed",
                        "response": {"status": "completed", "output": [], "usage": {}},
                    }
                ),
            ),
        ]
        chunks = _payloads(
            await _collect(
                mantle_convert.convert_stream(
                    "responses", "chat_completions", _agen(events)
                )
            )
        )
        assert chunks[1]["choices"][0]["delta"] == {"refusal": "I can't."}


class TestCacheWriteTokenUsage:
    """Cache-write tokens survive every usage conversion direction.

    The Converse path always reports ``input_tokens_details.cache_write_tokens``,
    so the Mantle path must report the same field rather than leaving a client
    reading two different usage shapes depending on the backend.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         https://platform.claude.com/docs/en/build-with-claude/prompt-caching
         stdapi/models/chat/_adapters/_openai_responses.py:InputTokensDetails
         stdapi/models/chat/_mantle/_convert.py:_responses_usage_from_chat
         stdapi/models/chat/_mantle/_convert.py:_messages_usage_from_chat
    """

    def test_chat_cache_write_tokens_reach_the_responses_usage(self) -> None:
        """``prompt_tokens_details`` cache writes become ``input_tokens_details`` ones."""
        raw = {
            "id": "chatcmpl-abc123",
            "model": "m",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi"}}
            ],
            "usage": {
                "prompt_tokens": 13,
                "completion_tokens": 3,
                "total_tokens": 16,
                "prompt_tokens_details": {"cached_tokens": 4, "cache_write_tokens": 1},
            },
        }
        out = mantle_convert.convert_response("chat_completions", "responses", raw)
        assert out["usage"]["input_tokens_details"] == {
            "cached_tokens": 4,
            "cache_write_tokens": 1,
        }

    async def test_streamed_responses_usage_reports_cache_write_tokens(self) -> None:
        """The terminal streaming event carries the same ``cache_write_tokens``."""
        chunks: list[SseEvent] = [
            (
                None,
                dumps(
                    {
                        "id": "chatcmpl-1",
                        "created": 100,
                        "model": "m",
                        "choices": [{"index": 0, "delta": {"content": "hi"}}],
                    }
                ),
            ),
            (
                None,
                dumps(
                    {
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {
                            "prompt_tokens": 13,
                            "completion_tokens": 3,
                            "prompt_tokens_details": {
                                "cached_tokens": 4,
                                "cache_write_tokens": 1,
                            },
                        },
                    }
                ),
            ),
        ]
        events = await _collect(
            mantle_convert.convert_stream(
                "chat_completions", "responses", _agen(chunks), "resp_route1"
            )
        )
        completed = next(
            loads(data) for name, data in events if name == "response.completed"
        )
        assert completed["response"]["usage"]["input_tokens_details"] == {
            "cached_tokens": 4,
            "cache_write_tokens": 1,
        }

    def test_anthropic_cache_creation_survives_the_chat_pivot(self) -> None:
        """``cache_creation_input_tokens`` round-trips through ``cache_write_tokens``."""
        anthropic_usage = {
            "input_tokens": 8,
            "output_tokens": 3,
            "cache_read_input_tokens": 4,
            "cache_creation_input_tokens": 1,
        }
        chat_usage = mantle_convert._chat_usage_from_messages(anthropic_usage)  # noqa: SLF001
        assert chat_usage["prompt_tokens_details"]["cache_write_tokens"] == 1
        assert mantle_convert._messages_usage_from_chat(chat_usage) == anthropic_usage  # noqa: SLF001
