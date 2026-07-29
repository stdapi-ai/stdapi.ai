"""Offline unit tests for the Bedrock Mantle wire-format conversion helpers.

Covers request, response and stream conversion between the three Mantle wire
shapes (:mod:`stdapi.models.chat._mantle._convert`) and the passthrough
payload builders, all without any network or AWS call.
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
from stdapi.types.openai_responses import ResponseCreateParams

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.aws_bedrock_mantle import SseEvent

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
    """Chat Completions request payloads converted to the Responses shape."""

    def test_system_and_string_user_message_produce_instructions_and_input(
        self,
    ) -> None:
        """System text becomes instructions; a string user message stays a string."""
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
        """A user text+image part list converts to input_text/input_image parts."""
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
        """An assistant message yields a text item and a function_call item."""
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
        """A tool-role message becomes a function_call_output item."""
        payload = {
            "model": "m",
            "messages": [{"role": "tool", "tool_call_id": "call_1", "content": "72F"}],
        }
        out = mantle_convert.convert_payload("chat_completions", "responses", payload)
        assert out["input"] == [
            {"type": "function_call_output", "call_id": "call_1", "output": "72F"}
        ]

    def test_function_tool_converted_non_function_tool_dropped(self) -> None:
        """Only ``function``-typed tools survive conversion to Responses tools."""
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
        """``auto`` passes through; a named function choice becomes flat."""
        payload: dict[str, Any] = {
            "model": "m",
            "messages": [],
            "tool_choice": tool_choice,
        }
        out = mantle_convert.convert_payload("chat_completions", "responses", payload)
        assert out["tool_choice"] == expected

    def test_response_format_json_object(self) -> None:
        """``response_format: json_object`` becomes ``text.format: json_object``."""
        payload = {
            "model": "m",
            "messages": [],
            "response_format": {"type": "json_object"},
        }
        out = mantle_convert.convert_payload("chat_completions", "responses", payload)
        assert out["text"]["format"] == {"type": "json_object"}

    def test_response_format_json_schema(self) -> None:
        """``response_format: json_schema`` becomes a flat ``text.format`` value."""
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
    """Chat Completions request payloads converted to the Anthropic shape."""

    def test_system_and_developer_messages_join_as_system_text(self) -> None:
        """System and developer message text join into the Anthropic system."""
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
        """A user image_url data URI part becomes a base64 Anthropic image block."""
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
        """Assistant text and tool calls both convert into Anthropic blocks."""
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
        """A tool-role message becomes a user turn carrying a tool_result block."""
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
        """A user message directly followed by a tool message merge into one turn."""
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
        """A temperature above the Anthropic range is clamped to 1.0."""
        payload = {"model": "m", "messages": [], "temperature": 1.5}
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["temperature"] == 1.0

    def test_stop_string_becomes_single_element_list(self) -> None:
        """A single stop string becomes a one-element ``stop_sequences`` list."""
        payload = {"model": "m", "messages": [], "stop": "STOP"}
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["stop_sequences"] == ["STOP"]

    def test_stop_list_passthrough(self) -> None:
        """A stop list passes through unchanged as ``stop_sequences``."""
        payload = {"model": "m", "messages": [], "stop": ["A", "B"]}
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["stop_sequences"] == ["A", "B"]

    def test_user_field_becomes_metadata_user_id(self) -> None:
        """The ``user`` field maps to Anthropic ``metadata.user_id``."""
        payload = {"model": "m", "messages": [], "user": "user-123"}
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["metadata"] == {"user_id": "user-123"}

    def test_tool_without_parameters_gets_default_object_schema(self) -> None:
        """A function tool without ``parameters`` gets a default object schema."""
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
        """Each Chat Completions tool choice form maps to its Anthropic shape."""
        payload: dict[str, Any] = {
            "model": "m",
            "messages": [],
            "tool_choice": tool_choice,
        }
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["tool_choice"] == expected

    def test_unknown_role_ignored(self) -> None:
        """A message with an unrecognized role produces no turn."""
        payload = {"model": "m", "messages": [{"role": "foo", "content": "bar"}]}
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["messages"] == []

    def test_http_image_url_becomes_url_source(self) -> None:
        """A non-data-URI image URL becomes an Anthropic ``url`` source."""
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
        """Missing token limits fall back to the default max_tokens."""
        payload = {"model": "m", "messages": []}
        out = mantle_convert._chat_to_messages_request(payload)  # noqa: SLF001
        assert out["max_tokens"] == 4096


# ---------------------------------------------------------------------------
# 3. messages -> chat request (edge arms)
# ---------------------------------------------------------------------------


class TestMessagesToChatRequestEdges:
    """Anthropic Messages request payloads converted to the Chat Completions shape."""

    def test_system_block_list_becomes_chat_system_message(self) -> None:
        """A list-of-blocks system prompt joins into a single system message."""
        payload = {
            "model": "m",
            "system": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
            "messages": [],
        }
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert out["messages"][0] == {"role": "system", "content": "ab"}

    def test_user_turn_with_mixed_blocks_keeps_all_parts(self) -> None:
        """User turns keep text, base64 image, URL image and base64 document parts."""
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
        """A ``document`` block with a URL source has no mapping and is dropped."""
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
        """An assistant turn's text and tool_use block convert to message+tool_calls."""
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
                    "function": {"name": "f", "arguments": dumps({"a": 1})},
                }
            ],
        }

    def test_assistant_turn_with_image_block_drops_non_text_content(self) -> None:
        """Assistant turns keep only text; image/document parts are dropped."""
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
        """A tool_result block emits a ``tool`` message ahead of the rest of the turn."""
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
        """A user ID over 64 characters is replaced by its SHA-256 hex digest."""
        long_id = "u" * 70
        payload = {"model": "m", "messages": [], "metadata": {"user_id": long_id}}
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert out["user"] != long_id
        assert out["user"] == sha256(long_id.encode()).hexdigest()
        assert len(out["user"]) == 64

    def test_metadata_user_id_under_64_chars_passthrough(self) -> None:
        """A user ID within the OpenAI limit passes through unchanged."""
        payload = {"model": "m", "messages": [], "metadata": {"user_id": "u1"}}
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert out["user"] == "u1"

    def test_tool_without_input_schema_is_skipped(self) -> None:
        """A tool missing the ``input_schema`` key is dropped from the output."""
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
        """A tool's description is forwarded to the Chat Completions shape."""
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
        """Each Anthropic tool choice type maps to its Chat Completions shape."""
        payload = {"model": "m", "messages": [], "tool_choice": tool_choice}
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert out["tool_choice"] == expected

    def test_disable_parallel_tool_use_has_no_chat_equivalent(self) -> None:
        """``disable_parallel_tool_use`` has no Chat Completions field and is dropped."""
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

    def test_stop_sequences_become_stop(self) -> None:
        """``stop_sequences`` maps to the Chat Completions ``stop`` field."""
        payload = {"model": "m", "messages": [], "stop_sequences": ["END"]}
        out = mantle_convert._messages_to_chat_request(payload)  # noqa: SLF001
        assert out["stop"] == ["END"]


# ---------------------------------------------------------------------------
# 4. responses -> chat request (branches)
# ---------------------------------------------------------------------------


class TestResponsesToChatRequestBranches:
    """Responses API request payloads converted to the Chat Completions shape."""

    def test_instructions_and_string_input_become_messages(self) -> None:
        """Instructions become a system message; a string input becomes user text."""
        payload = {"model": "m", "instructions": "Be terse.", "input": "Hi"}
        out = mantle_convert._responses_to_chat_request(payload)  # noqa: SLF001
        assert out["messages"] == [
            {"role": "system", "content": "Be terse."},
            {"role": "user", "content": "Hi"},
        ]

    def test_function_call_item_becomes_assistant_tool_call(self) -> None:
        """A ``function_call`` input item becomes an assistant tool_calls message."""
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
        """Consecutive ``function_call`` items merge into one assistant message."""
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
        """A ``function_call_output`` input item becomes a ``tool`` message."""
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
        """A part-list ``function_call_output.output`` is flattened to plain text."""
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
        """``input_image``/``input_file`` parts convert to ``image_url``/``file``."""
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
        """An ``input_file`` part without inline ``file_data`` has no mapping."""
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
        """``max_output_tokens`` maps to ``max_completion_tokens``."""
        payload = {"model": "m", "input": "hi", "max_output_tokens": 500}
        out = mantle_convert._responses_to_chat_request(payload)  # noqa: SLF001
        assert out["max_completion_tokens"] == 500

    def test_reasoning_effort_forwarded(self) -> None:
        """``reasoning.effort`` maps to the flat ``reasoning_effort`` field."""
        payload = {"model": "m", "input": "hi", "reasoning": {"effort": "low"}}
        out = mantle_convert._responses_to_chat_request(payload)  # noqa: SLF001
        assert out["reasoning_effort"] == "low"

    def test_text_format_json_object(self) -> None:
        """``text.format: json_object`` maps to ``response_format: json_object``."""
        payload = {
            "model": "m",
            "input": "hi",
            "text": {"format": {"type": "json_object"}},
        }
        out = mantle_convert._responses_to_chat_request(payload)  # noqa: SLF001
        assert out["response_format"] == {"type": "json_object"}

    def test_text_format_json_schema(self) -> None:
        """``text.format: json_schema`` maps to a nested ``response_format``."""
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
        """Flat Responses tools/tool_choice map to their nested Chat shapes."""
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
    """Complete-response conversion between the three Mantle wire shapes."""

    def test_chat_to_responses_response(self) -> None:
        """Chat text, tool calls and usage convert to Responses output items."""
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
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 15,
        }

    def test_chat_to_messages_response(self) -> None:
        """Chat text, tool calls and usage convert to Anthropic content blocks."""
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
        """Anthropic text, tool_use and cache usage convert to CC choice/usage."""
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
                "function": {
                    "name": "get_weather",
                    "arguments": dumps({"city": "Paris"}),
                },
            }
        ]
        assert out["usage"] == {
            "prompt_tokens": 13,
            "completion_tokens": 3,
            "total_tokens": 16,
            "prompt_tokens_details": {"cached_tokens": 4},
        }

    def test_messages_to_chat_response_stop_reason_mapped_without_tool_use(
        self,
    ) -> None:
        """Without tool_use blocks, the finish reason follows ``stop_reason``."""
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
    """Chat Completions finish reason derivation from a Responses response."""

    def test_function_call_output_item_forces_tool_calls(self) -> None:
        """A ``function_call`` output item forces the ``tool_calls`` finish reason."""
        response = {"output": [{"type": "function_call", "name": "f"}]}
        result = mantle_convert._finish_from_response(  # noqa: SLF001
            response, has_tool_calls=False
        )
        assert result == "tool_calls"

    def test_incomplete_max_output_tokens_maps_to_length(self) -> None:
        """An incomplete response with ``max_output_tokens`` maps to ``length``."""
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
    """Chat Completions SSE chunks converted to a Responses SSE stream."""

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
        """Text then a tool call produce the full close/open event sequence."""
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

    async def test_stream_ending_without_usage_chunk_still_emits_completed(
        self,
    ) -> None:
        """A stream ending without a usage chunk still closes with zero usage."""
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
        """An in-band chat-shaped error chunk aborts the stream: no fabricated tail."""
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
        """A non-JSON upstream frame is skipped; the rest still converts."""
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
        assert "response.created" in _names(events)


class TestChatToMessagesStream:
    """Chat Completions SSE chunks converted to an Anthropic Messages SSE stream."""

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
        """Text, a tool call, a finish chunk and usage produce the full sequence."""
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

    async def test_early_end_without_finish_or_usage_closes_open_block(self) -> None:
        """A stream ending mid tool-call still closes the block and defaults usage."""
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
        """An in-band chat-shaped error chunk aborts the stream: no fabricated tail."""
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
    """Anthropic Messages SSE events converted to Chat Completions SSE chunks."""

    async def test_full_event_sequence_with_tool_use_and_text_deltas(self) -> None:
        """A tool_use, arguments, a text delta, a thinking delta and ping convert."""
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
            "prompt_tokens_details": {"cached_tokens": 0},
        }

    async def test_unnamed_error_type_payload_raises_mantle_error(self) -> None:
        """An unnamed ``{"type": "error"}`` data event aborts the stream (502)."""
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
    """Responses SSE events converted to Chat Completions SSE chunks."""

    async def test_function_call_item_and_argument_delta_with_unknown_event_ignored(
        self,
    ) -> None:
        """A function_call item, argument deltas and completion convert; unknown is skipped."""
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
            "prompt_tokens_details": {"cached_tokens": 0},
        }

    async def test_response_incomplete_event_emits_finish_and_usage(self) -> None:
        """A named ``response.incomplete`` event ends the stream like ``response.completed``.

        Without this, a truncated Responses stream converted to Chat
        Completions would end with neither a finish reason nor billed usage.
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
    """Validated Chat Completions requests dumped to the Mantle passthrough shape."""

    async def test_inline_parts_resolved_model_overridden_extension_fields_stripped(
        self,
    ) -> None:
        """Image/audio/file parts are inlined, the model is overridden, store dropped."""
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

    async def test_named_tool_choice_forwarded_verbatim(self) -> None:
        """A named-function ``tool_choice`` reaches the upstream payload unchanged."""
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


class TestMessagesPayloadBuilder:
    """Validated Anthropic Messages requests dumped to the Mantle passthrough shape."""

    async def test_inline_sources_default_max_tokens_system_folding(self) -> None:
        """Base64 sources stay inline, max_tokens defaults, inline system folds in."""
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


class TestResponsesPayloadBuilder:
    """Validated Responses requests dumped to the Mantle passthrough shape."""

    async def test_inline_files_web_search_forced_off_tool_sanitized_pinned_region(
        self,
    ) -> None:
        """Files inline, web_search access is forced off, and the region pins."""
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

    async def test_undecodable_previous_response_id_raises_api_error(self) -> None:
        """A non-Mantle ``previous_response_id`` fails with a 400 ApiError."""
        request = ResponseCreateParams.model_validate(
            {"model": "ignored", "previous_response_id": "resp_@@@invalid@@@"}
        )
        with pytest.raises(ApiError) as exc_info:
            await mantle_convert.responses_payload(request, "model-id")
        assert exc_info.value.status == 400

    async def test_null_previous_response_id_removed_from_payload(self) -> None:
        """An explicitly-null ``previous_response_id`` is dropped, not forwarded."""
        request = ResponseCreateParams.model_validate(
            {"model": "ignored", "previous_response_id": "resp-local", "input": "hi"}
        ).model_copy(update={"previous_response_id": None})
        payload, pinned_region = await mantle_convert.responses_payload(
            request, "model-id"
        )
        assert "previous_response_id" not in payload
        assert pinned_region is None

    async def test_named_tool_choice_forwarded_verbatim(self) -> None:
        """A named-function ``tool_choice`` reaches the upstream payload unchanged."""
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


class TestEnableStreamUsage:
    """Forcing streaming with usage reporting on upstream request payloads."""

    def test_chat_merges_existing_stream_options(self) -> None:
        """Existing ``stream_options`` keys are preserved alongside include_usage."""
        payload = {"stream_options": {"foo": "bar"}}
        out = mantle_convert.enable_stream_usage("chat_completions", payload)
        assert out["stream"] is True
        assert out["stream_options"] == {"foo": "bar", "include_usage": True}

    def test_chat_creates_stream_options_when_absent(self) -> None:
        """``stream_options`` is created when the payload has none."""
        out = mantle_convert.enable_stream_usage("chat_completions", {})
        assert out["stream_options"] == {"include_usage": True}

    def test_non_chat_api_only_forces_stream(self) -> None:
        """Non-chat APIs get ``stream: True`` without a ``stream_options`` field."""
        out = mantle_convert.enable_stream_usage("responses", {"model": "m"})
        assert out["stream"] is True
        assert "stream_options" not in out


# ---------------------------------------------------------------------------
# 8. Legacy text completions
# ---------------------------------------------------------------------------


class TestTextCompletionAsChatPayload:
    """Legacy completion requests converted to a Chat Completions payload."""

    async def test_string_prompt_copies_supported_fields(self) -> None:
        """A string prompt and supported sampling fields are copied through."""
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
        """A single-element prompt list is unwrapped to its string element."""
        request = LegacyCompletionCreateParams.model_validate(
            {"model": "ignored", "prompt": ["Solo"]}
        )
        payload = await mantle_convert.text_completion_as_chat_payload(
            request, "model-id"
        )
        assert payload["messages"] == [{"role": "user", "content": "Solo"}]

    async def test_n_is_copied_through_without_rejection(self) -> None:
        """``n`` is copied unchanged; only the target-API conversion rejects n>1."""
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
        """``echo``, ``suffix`` and ``logprobs`` are rejected with a 400 ApiError."""
        request = LegacyCompletionCreateParams.model_validate(
            {"model": "ignored", "prompt": "Hi", field: value}
        )
        with pytest.raises(ApiError) as exc_info:
            await mantle_convert.text_completion_as_chat_payload(request, "model-id")
        assert exc_info.value.status == 400

    async def test_multi_prompt_list_rejected(self) -> None:
        """A multi-element prompt list is rejected: only one prompt is supported."""
        request = LegacyCompletionCreateParams.model_validate(
            {"model": "ignored", "prompt": ["a", "b"]}
        )
        with pytest.raises(ApiError, match="Multiple prompts"):
            await mantle_convert.text_completion_as_chat_payload(request, "model-id")

    async def test_url_prompt_rejected_as_file_prompt(self) -> None:
        """A non-inline (URL) prompt is rejected: file prompts are unsupported."""
        request = LegacyCompletionCreateParams.model_validate(
            {"model": "ignored", "prompt": "https://example.com/prompt.txt"}
        )
        with pytest.raises(ApiError, match="File prompts"):
            await mantle_convert.text_completion_as_chat_payload(request, "model-id")


class TestChatResponseAsTextCompletion:
    """Chat Completions responses converted to the legacy ``Completion`` shape."""

    def test_tool_calls_finish_reason_maps_to_stop_and_usage_extras_filtered(
        self,
    ) -> None:
        """A ``tool_calls`` finish maps to ``stop``; unknown usage keys are dropped."""
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
        assert result.choices[0].finish_reason == "stop"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 5
        assert result.usage.completion_tokens == 2
        assert result.usage.total_tokens == 7


class TestChatStreamAsTextCompletionCompact:
    """SSE wrapper converting a Chat Completions stream to text-completion chunks."""

    async def test_done_and_non_string_data_pass_through(self) -> None:
        """The ``[DONE]`` sentinel and a non-string ``data`` event pass through."""

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
        """An unnamed in-band error chunk aborts the stream instead of being dropped."""

        async def source() -> AsyncGenerator[ServerSentEvent]:
            yield ServerSentEvent(data=dumps({"error": {"message": "boom problem"}}))

        stream = mantle_convert.chat_stream_as_text_completion(source(), "cmpl-1")
        with pytest.raises(MantleError, match="boom problem") as exc_info:
            async for _ in stream:
                pass
        assert exc_info.value.status == 502

    async def test_named_error_event_still_passes_through_unchanged(self) -> None:
        """A named error event (e.g. relayed) is passed through, not raised on."""

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
    """Tool-arguments JSON string parsing."""

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
    """Base64 ``data:`` URI splitting."""

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
    """Plain-text extraction from the three wire content shapes."""

    def test_chat_text_from_string(self) -> None:
        """A plain string is returned as-is."""
        assert mantle_convert._chat_text("hi") == "hi"  # noqa: SLF001

    def test_chat_text_from_part_list(self) -> None:
        """Text and refusal parts concatenate."""
        parts = [{"type": "text", "text": "a"}, {"type": "refusal", "refusal": "b"}]
        assert mantle_convert._chat_text(parts) == "ab"  # noqa: SLF001

    def test_chat_text_from_none(self) -> None:
        """``None`` content yields an empty string."""
        assert mantle_convert._chat_text(None) == ""  # noqa: SLF001

    def test_anthropic_text_from_string(self) -> None:
        """A plain string is returned as-is."""
        assert mantle_convert._anthropic_text("hi") == "hi"  # noqa: SLF001

    def test_anthropic_text_from_block_list(self) -> None:
        """Only ``text`` blocks contribute; other block types are ignored."""
        blocks = [{"type": "text", "text": "a"}, {"type": "tool_use", "id": "x"}]
        assert mantle_convert._anthropic_text(blocks) == "a"  # noqa: SLF001

    def test_anthropic_text_from_none(self) -> None:
        """``None`` content yields an empty string."""
        assert mantle_convert._anthropic_text(None) == ""  # noqa: SLF001

    def test_responses_text_from_string(self) -> None:
        """A plain string is returned as-is."""
        assert mantle_convert._responses_text("hi") == "hi"  # noqa: SLF001

    def test_responses_text_from_part_list(self) -> None:
        """Textual part types concatenate."""
        parts = [
            {"type": "input_text", "text": "a"},
            {"type": "output_text", "text": "b"},
        ]
        assert mantle_convert._responses_text(parts) == "ab"  # noqa: SLF001

    def test_responses_text_from_none(self) -> None:
        """``None`` content yields an empty string."""
        assert mantle_convert._responses_text(None) == ""  # noqa: SLF001


class TestEnsureSingleChoice:
    """Multi-choice request rejection for single-choice-only APIs."""

    def test_n_greater_than_one_raises(self) -> None:
        """``n=2`` is rejected with a 400 ApiError."""
        with pytest.raises(ApiError) as exc_info:
            mantle_convert._ensure_single_choice({"n": 2})  # noqa: SLF001
        assert exc_info.value.status == 400

    def test_n_one_or_absent_does_not_raise(self) -> None:
        """``n=1`` and a missing ``n`` are both accepted."""
        mantle_convert._ensure_single_choice({"n": 1})  # noqa: SLF001
        mantle_convert._ensure_single_choice({})  # noqa: SLF001


class TestSanitizeToolSchema:
    """Recursive stripping of unsupported JSON Schema keywords."""

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
    """In-band stream error message extraction."""

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
    """Same-shape conversions are no-ops; cross-shape ones compose through chat."""

    def test_convert_payload_same_api_returns_same_object(self) -> None:
        """A same-shape payload conversion returns the identical object."""
        payload = {"a": 1}
        result = mantle_convert.convert_payload(
            "chat_completions", "chat_completions", payload
        )
        assert result is payload

    def test_convert_response_same_api_returns_same_object(self) -> None:
        """A same-shape response conversion returns the identical object."""
        raw = {"a": 1}
        result = mantle_convert.convert_response("responses", "responses", raw)
        assert result is raw

    def test_convert_stream_same_api_returns_same_generator(self) -> None:
        """A same-shape stream conversion returns the identical generator."""
        gen = _agen([])
        result = mantle_convert.convert_stream("messages", "messages", gen)
        assert result is gen

    def test_messages_to_responses_composes_through_chat(self) -> None:
        """A Messages payload converts to Responses via the Chat Completions shape."""
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
    """Chat Completions fields without a Responses equivalent are dropped."""

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
    """Chat Completions fields without an Anthropic equivalent are dropped."""

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
    """Anthropic fields without a Chat Completions equivalent are dropped."""

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
