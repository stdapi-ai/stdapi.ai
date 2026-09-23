"""Unknown request fields: ignored exactly where OpenAI ignores them.

Probed on 2026-09-23 with raw requests: Chat Completions ignores an unknown
field on a user, system, assistant, tool or function message, on a text, image
or refusal part and its ``image_url``, on a tool call and its ``function`` or
``custom``, and on a tool definition and its ``function``; it refuses one on a
developer message or its parts, on a file or input-audio part and its object, on
an assistant's ``audio``, on ``response_format`` and at the top level. Moderations ignores one at the top
level and on every input item; Responses ignores one on a function tool and
refuses one on an input item or its parts. The suite runs with
``strict_input_validation``, so the refusals below are the strict ones.

Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
     https://developers.openai.com/api/reference/resources/moderations/methods/create
     stdapi/types/__init__.py:BaseModelRequestIgnoringExtra
"""

from __future__ import annotations

import struct
import zlib
from base64 import b64encode
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from openai import BadRequestError

from stdapi.types.openai_moderations import ModerationCreateParams

if TYPE_CHECKING:
    from openai import OpenAI

#: The unknown field every request below carries.
_EXTRA: dict[str, Any] = {"x_vendor_extension": 1}


def _png_data_uri() -> str:
    """Return a 16x16 red PNG as a data URI."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    rows = b"".join(b"\x00" + b"\xff\x00\x00" * 16 for _ in range(16))
    header = struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + b64encode(png).decode()


def _budget(use_official_api: bool) -> dict[str, Any]:
    """Return an output budget the target's chat model answers within.

    The official model reasons by default, and runs out of a small budget
    before it answers unless told to reason minimally.
    """
    if use_official_api:
        return {"reasoning_effort": "minimal", "max_completion_tokens": 256}
    return {"max_completion_tokens": 16}


class TestChatCompletionsIgnoresWhereOpenAIDoes:
    """Chat Completions ignores unknown fields inside messages and tools, as OpenAI does.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
    """

    def test_messages_parts_tool_calls_and_tools_ignore_unknown_fields(
        self, openai_client: OpenAI, chat_model: str, use_official_api: bool
    ) -> None:
        """Every surface OpenAI ignores an unknown field on carries one, and the call answers."""
        response = openai_client.post(
            "/chat/completions",
            body={
                "model": chat_model,
                "messages": [
                    {"role": "system", "content": "Answer briefly.", **_EXTRA},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "What is 2+2?", **_EXTRA}],
                        **_EXTRA,
                    },
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "add",
                                    "arguments": '{"a": 2, "b": 2}',
                                    **_EXTRA,
                                },
                                **_EXTRA,
                            }
                        ],
                        **_EXTRA,
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_1",
                        "content": "4",
                        **_EXTRA,
                    },
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "add",
                            "parameters": {"type": "object", "properties": {}},
                            **_EXTRA,
                        },
                        **_EXTRA,
                    }
                ],
                **_budget(use_official_api),
            },
            cast_to=httpx.Response,
        )

        assert response.status_code == 200, response.text
        assert response.json()["choices"], response.text

    def test_image_parts_ignore_unknown_fields(
        self, openai_client: OpenAI, chat_vision_model: str, use_official_api: bool
    ) -> None:
        """An image part and its ``image_url`` both carry one, and the call answers."""
        response = openai_client.post(
            "/chat/completions",
            body={
                "model": chat_vision_model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Name the colour."},
                            {
                                "type": "image_url",
                                "image_url": {"url": _png_data_uri(), **_EXTRA},
                                **_EXTRA,
                            },
                        ],
                    }
                ],
                **_budget(use_official_api),
            },
            cast_to=httpx.Response,
        )

        assert response.status_code == 200, response.text

    def test_refusal_parts_and_function_messages_ignore_unknown_fields(
        self, openai_client: OpenAI, chat_legacy_model: str
    ) -> None:
        """A replayed refusal part and a legacy function message both carry one, and the call answers.

        The legacy chat model is used: OpenAI's reasoning models refuse the
        ``function`` role outright.
        """
        response = openai_client.post(
            "/chat/completions",
            body={
                "model": chat_legacy_model,
                "messages": [
                    {"role": "user", "content": "Say something unsafe."},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "refusal", "refusal": "I can't.", **_EXTRA}
                        ],
                    },
                    {"role": "user", "content": "Call f."},
                    {
                        "role": "assistant",
                        "content": None,
                        "function_call": {"name": "f", "arguments": "{}"},
                    },
                    {"role": "function", "name": "f", "content": "ok", **_EXTRA},
                    {"role": "user", "content": "Say OK."},
                ],
                "max_tokens": 16,
            },
            cast_to=httpx.Response,
        )

        assert response.status_code == 200, response.text

    @pytest.mark.parametrize(
        ("body", "param"),
        [
            pytest.param(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "file",
                                    "file": {
                                        "file_data": "data:application/pdf;base64,JVBERi0=",
                                        "filename": "a.pdf",
                                    },
                                    **_EXTRA,
                                }
                            ],
                        }
                    ]
                },
                "messages[0].content[0].x_vendor_extension",
                id="file_part",
            ),
            pytest.param(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": "AAAA",
                                        "format": "wav",
                                        **_EXTRA,
                                    },
                                }
                            ],
                        }
                    ]
                },
                "messages[0].content[0].input_audio.x_vendor_extension",
                id="input_audio",
            ),
            pytest.param(
                {
                    "messages": [
                        {"role": "user", "content": "x"},
                        {"role": "assistant", "audio": {"id": "audio_x", **_EXTRA}},
                        {"role": "user", "content": "x"},
                    ]
                },
                "messages[1].audio.x_vendor_extension",
                id="assistant_audio",
            ),
            pytest.param(
                {"messages": [{"role": "developer", "content": "x", **_EXTRA}]},
                "messages[0].x_vendor_extension",
                id="developer_message",
            ),
            pytest.param(
                {
                    "messages": [
                        {
                            "role": "developer",
                            "content": [{"type": "text", "text": "x", **_EXTRA}],
                        }
                    ]
                },
                "messages[0].content[0].x_vendor_extension",
                id="developer_part",
            ),
            pytest.param(
                {
                    "messages": [{"role": "user", "content": "x"}],
                    "response_format": {"type": "text", "x_vendor_extension": 1},
                },
                "response_format.x_vendor_extension",
                id="response_format",
            ),
        ],
    )
    def test_the_surfaces_openai_refuses_still_refuse(
        self, openai_client: OpenAI, chat_model: str, body: dict[str, Any], param: str
    ) -> None:
        """Where OpenAI refuses an unknown field, the refusal names it."""
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.post(
                "/chat/completions",
                body={"model": chat_model, **body},
                cast_to=httpx.Response,
            )

        error = excinfo.value.body
        assert isinstance(error, dict)
        assert (error["param"], error["code"]) == (param, "unknown_parameter")


class TestModerationsIgnoresUnknownFields:
    """Moderations ignores unknown fields at the top level and on every input item.

    The model differs per target: OpenAI's own, and the gateway's toxicity
    detection, which needs no guardrail.

    Ref: https://developers.openai.com/api/reference/resources/moderations/methods/create
         stdapi/types/openai_moderations.py:ModerationCreateParams
    """

    def test_a_request_and_its_items_ignore_unknown_fields(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """A text item carrying an unknown field is classified like any other."""
        model = (
            "omni-moderation-latest"
            if use_official_api
            else "amazon.comprehend-toxicity"
        )
        response = openai_client.post(
            "/moderations",
            body={
                "model": model,
                "input": [{"type": "text", "text": "The weather is nice.", **_EXTRA}],
                **_EXTRA,
            },
            cast_to=httpx.Response,
        )

        assert response.status_code == 200, response.text
        assert len(response.json()["results"]) == 1

    @pytest.mark.local
    def test_an_image_item_and_its_url_ignore_unknown_fields(self) -> None:
        """An image item and its ``image_url`` carrying one validate, as OpenAI accepts them.

        Checked on the request model: classifying an image needs a guardrail,
        which a target without one cannot run.
        """
        params = ModerationCreateParams.model_validate(
            {
                "input": [
                    {
                        "type": "image_url",
                        "image_url": {"url": _png_data_uri(), **_EXTRA},
                        **_EXTRA,
                    }
                ],
                **_EXTRA,
            }
        )

        assert "x_vendor_extension" not in params.model_dump_json()


class TestResponsesIgnoresWhereOpenAIDoes:
    """Responses ignores an unknown field on a function tool, and nowhere in its input.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/types/openai_responses.py:FunctionTool
    """

    def test_a_function_tool_ignores_unknown_fields(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """A function tool carrying an unknown field is accepted, and the call answers."""
        response = openai_client.post(
            "/responses",
            body={
                "model": responses_model,
                "input": "Say OK.",
                "max_output_tokens": 16,
                "tools": [
                    {
                        "type": "function",
                        "name": "noop",
                        "parameters": {"type": "object", "properties": {}},
                        **_EXTRA,
                    }
                ],
            },
            cast_to=httpx.Response,
        )

        assert response.status_code == 200, response.text

    def test_an_input_item_still_refuses_unknown_fields(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """An unknown field on a function call output item is refused, naming it."""
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.post(
                "/responses",
                body={
                    "model": responses_model,
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "c",
                            "output": "x",
                            **_EXTRA,
                        }
                    ],
                },
                cast_to=httpx.Response,
            )

        error = excinfo.value.body
        assert isinstance(error, dict)
        assert (error["param"], error["code"]) == (
            "input[0].x_vendor_extension",
            "unknown_parameter",
        )
