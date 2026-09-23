"""Request-validation 400s, worded as each API words the same fault.

The same invalid body is sent to every target, so the official lane proves the
expectation is the vendor's. Probed on 2026-09-23 with raw requests: OpenAI's
Chat Completions, Responses and Images routes name the parameter in ``param``
(``messages[0].content``) and the failure class in ``code``; its Moderations and
Audio routes relay their validator's error list, Embeddings writes a sentence
per field and Uploads and Files use JSON-schema wording, all with ``param`` and
``code`` null. Anthropic reads ``<loc>: <msg>``, the Pydantic location without a
request-part prefix.

Ref: https://developers.openai.com/api/docs/guides/error-codes
     https://platform.claude.com/docs/en/api/errors
     stdapi/validation_errors.py:openai_validation_error
"""

from __future__ import annotations

import struct
import zlib
from typing import TYPE_CHECKING, Any, NamedTuple

import httpx
import pytest
from anthropic import BadRequestError as AnthropicBadRequestError
from openai import BadRequestError

if TYPE_CHECKING:
    from anthropic import Anthropic
    from openai import OpenAI

#: A one-message conversation, valid on every target.
_MESSAGES = [{"role": "user", "content": "hi"}]


def _png() -> bytes:
    """Return a valid 16x16 red PNG, small enough to cost nothing to send."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data))
        )

    rows = b"".join(b"\x00" + b"\xff\x00\x00" * 16 for _ in range(16))
    header = struct.pack(">IIBBBBB", 16, 16, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


class _Case(NamedTuple):
    """One invalid OpenAI request and the refusal it earns."""

    #: Route under ``/v1``.
    route: str
    #: Body fields beside ``model``, which the test fills per target.
    body: dict[str, Any]
    #: Expected ``error.param``.
    param: str
    #: Expected ``error.code``.
    code: str
    #: Expected ``error.message``, or its beginning and end around the parts upstream may reorder.
    message: str | tuple[str, str]


#: Invalid OpenAI requests, one or more per failure class.
_OPENAI_CASES = {
    "unknown_parameter": _Case(
        "responses",
        {"input": "hi", "reasoning": {"effort": "low", "bogus": 1}},
        "reasoning.bogus",
        "unknown_parameter",
        "Unknown parameter: 'reasoning.bogus'.",
    ),
    "unknown_parameter_in_an_item": _Case(
        "responses",
        {"input": [{"role": "user", "content": "hi", "bogus": 1}]},
        "input[0].bogus",
        "unknown_parameter",
        "Unknown parameter: 'input[0].bogus'.",
    ),
    "missing_required_parameter": _Case(
        "chat/completions",
        {},
        "messages",
        "missing_required_parameter",
        "Missing required parameter: 'messages'.",
    ),
    "missing_required_parameter_in_an_item": _Case(
        "chat/completions",
        {"messages": [{"content": "hi"}]},
        "messages[0].role",
        "missing_required_parameter",
        "Missing required parameter: 'messages[0].role'.",
    ),
    "missing_required_parameter_in_a_nested_object": _Case(
        "chat/completions",
        {"messages": _MESSAGES, "tools": [{"type": "function", "function": {}}]},
        "tools[0].function.name",
        "missing_required_parameter",
        "Missing required parameter: 'tools[0].function.name'.",
    ),
    "invalid_type": _Case(
        "chat/completions",
        {"messages": _MESSAGES, "max_completion_tokens": "abc"},
        "max_completion_tokens",
        "invalid_type",
        "Invalid type for 'max_completion_tokens': expected an integer, but got a "
        "string instead.",
    ),
    "invalid_type_of_a_decimal_number": _Case(
        "chat/completions",
        {"messages": _MESSAGES, "seed": 1.5},
        "seed",
        "invalid_type",
        "Invalid type for 'seed': expected an integer, but got a decimal number "
        "instead.",
    ),
    "invalid_type_in_an_item": _Case(
        "chat/completions",
        {"messages": [*_MESSAGES, {"role": "user", "content": 123}]},
        "messages[1].content",
        "invalid_type",
        (
            "Invalid type for 'messages[1].content': expected one of a string or ",
            ", but got an integer instead.",
        ),
    ),
    "invalid_value": _Case(
        "chat/completions",
        {"messages": _MESSAGES, "response_format": {"type": "bogus"}},
        "response_format.type",
        "invalid_value",
        ("Invalid value: 'bogus'. Supported values are: ", "'."),
    ),
    "invalid_value_in_an_item": _Case(
        "chat/completions",
        {"messages": [{"role": "bogus", "content": "hi"}]},
        "messages[0].role",
        "invalid_value",
        ("Invalid value: 'bogus'. Supported values are: ", "'."),
    ),
    "integer_below_min_value": _Case(
        "chat/completions",
        {"messages": _MESSAGES, "n": 0},
        "n",
        "integer_below_min_value",
        "Invalid 'n': integer below minimum value. Expected a value >= 1, but got 0 "
        "instead.",
    ),
    "integer_above_max_value": _Case(
        "responses",
        {"input": "hi", "top_logprobs": 50},
        "top_logprobs",
        "integer_above_max_value",
        "Invalid 'top_logprobs': integer above maximum value. Expected a value <= 20, "
        "but got 50 instead.",
    ),
    "empty_array": _Case(
        "chat/completions",
        {"messages": []},
        "messages",
        "empty_array",
        "Invalid 'messages': empty array. Expected an array with minimum length 1, "
        "but got an empty array instead.",
    ),
}


def _assert_message(message: str, expected: str | tuple[str, str]) -> None:
    """Assert *message* is *expected*, or starts and ends as it says."""
    if isinstance(expected, str):
        assert message == expected
    else:
        assert message.startswith(expected[0]), message
        assert message.endswith(expected[1]), message


class TestOpenAIValidationErrors:
    """OpenAI routes name the parameter and the failure class of a malformed request.

    Ref: https://developers.openai.com/api/docs/guides/error-codes
         stdapi/validation_errors.py:_chat_completions_wording
    """

    @pytest.mark.parametrize("case", _OPENAI_CASES.values(), ids=_OPENAI_CASES)
    def test_the_refusal_names_the_parameter(
        self, openai_client: OpenAI, chat_model: str, responses_model: str, case: _Case
    ) -> None:
        """The 400 carries upstream's ``param``, ``code`` and message for the class.

        Where upstream lists accepted values, only the frame of the message is
        compared: each target lists the values it accepts.
        """
        model = responses_model if case.route == "responses" else chat_model
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.post(
                f"/{case.route}",
                body={"model": model, **case.body},
                cast_to=httpx.Response,
            )

        assert excinfo.value.status_code == 400
        error = excinfo.value.body
        assert isinstance(error, dict)
        assert error["type"] == "invalid_request_error"
        assert (error["param"], error["code"]) == (case.param, case.code), error
        _assert_message(error["message"], case.message)

    def test_an_image_request_names_the_parameter(
        self, openai_client: OpenAI, image_generation_model: str
    ) -> None:
        """Beyond the chat routes, a request refused before generating names its field.

        Ref: https://developers.openai.com/api/reference/resources/images/methods/generate
        """
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.post(
                "/images/generations",
                body={"model": image_generation_model, "prompt": "x", "n": "abc"},
                cast_to=httpx.Response,
            )

        error = excinfo.value.body
        assert isinstance(error, dict)
        assert (error["param"], error["code"]) == ("n", "invalid_type"), error
        assert error["message"] == (
            "Invalid type for 'n': expected an integer, but got a string instead."
        )


#: Invalid JSON requests to the routes with a wording of their own: model fixture, body, expected refusal.
_ROUTE_CASES = {
    "moderations_error_list": (
        "/moderations",
        None,
        {},
        (
            "[{'type': 'missing', 'loc': ('body', 'input'), 'msg': 'Field required'}]",
            None,
            None,
        ),
    ),
    "speech_error_list": (
        "/audio/speech",
        "speech_standard_model",
        {"voice": "alloy", "input": 5},
        (
            (
                "[{'type': 'string_type', 'loc': ('body', 'input'), 'msg': 'Input "
                "should be a valid string'}]"
            ),
            None,
            None,
        ),
    ),
    "chat_missing_model": (
        "/chat/completions",
        None,
        {"messages": _MESSAGES},
        ("you must provide a model parameter", None, None),
    ),
    "embeddings_missing_input": (
        "/embeddings",
        "embedding_model",
        {},
        ("Please submit an `input`.", None, None),
    ),
    "embeddings_missing_model": (
        "/embeddings",
        None,
        {"input": "hi"},
        ("you must provide a model parameter", None, None),
    ),
    "uploads_json_schema_type": (
        "/uploads",
        None,
        {
            "filename": "a.txt",
            "purpose": "assistants",
            "bytes": "abc",
            "mime_type": "text/plain",
        },
        ("'abc' is not of type 'integer' - 'bytes'", None, None),
    ),
}


class TestOpenAIRouteWordings:
    """Routes OpenAI words differently keep their own wording, ``param`` and ``code`` null.

    Ref: https://developers.openai.com/api/docs/guides/error-codes
         stdapi/validation_errors.py:openai_validation_error
    """

    @pytest.mark.parametrize(
        ("path", "model_fixture", "body", "expected"),
        _ROUTE_CASES.values(),
        ids=_ROUTE_CASES,
    )
    def test_the_route_words_the_refusal_as_openai_does(
        self,
        openai_client: OpenAI,
        request: pytest.FixtureRequest,
        path: str,
        model_fixture: str | None,
        body: dict[str, Any],
        expected: tuple[str, str | None, str | None],
    ) -> None:
        """The message, ``param`` and ``code`` are those of OpenAI's route."""
        if model_fixture is not None:
            body = {"model": request.getfixturevalue(model_fixture), **body}
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.post(path, body=body, cast_to=httpx.Response)

        error = excinfo.value.body
        assert isinstance(error, dict)
        assert (error["message"], error["param"], error["code"]) == expected

    def test_a_model_of_the_wrong_type_is_an_unparsable_body(
        self, openai_client: OpenAI
    ) -> None:
        """A route reading ``model`` first refuses a non-string one as unparsable."""
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.post(
                "/chat/completions",
                body={"model": 5, "messages": _MESSAGES},
                cast_to=httpx.Response,
            )

        error = excinfo.value.body
        assert isinstance(error, dict)
        assert error["message"].startswith(
            "We could not parse the JSON body of your request."
        )
        assert (error["param"], error["code"]) == (None, None)

    def test_a_refused_purpose_is_named_in_json_schema_terms(
        self, openai_client: OpenAI
    ) -> None:
        """Uploads name ``purpose`` in ``param`` with no code; each lists its own purposes."""
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.post(
                "/uploads",
                body={
                    "filename": "a.txt",
                    "purpose": "bogus",
                    "bytes": 3,
                    "mime_type": "text/plain",
                },
                cast_to=httpx.Response,
            )

        error = excinfo.value.body
        assert isinstance(error, dict)
        assert (error["param"], error["code"]) == ("purpose", None)
        assert error["message"].startswith("'bogus' is not one of ['"), error
        assert error["message"].endswith("'] - 'purpose'"), error

    def test_a_query_below_its_minimum_is_worded_in_json_schema_terms(
        self, openai_client: OpenAI
    ) -> None:
        """The Files listing states the bound its ``limit`` fell below."""
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.get(
                "/files", options={"params": {"limit": 0}}, cast_to=httpx.Response
            )

        error = excinfo.value.body
        assert isinstance(error, dict)
        assert error["message"].startswith("0 is less than the minimum of 1"), error
        assert (error["param"], error["code"]) == (None, None)

    def test_a_missing_upload_is_listed_with_no_value(
        self, openai_client: OpenAI, transcription_model: str
    ) -> None:
        """A transcription form without a file lists the missing field, input null."""
        response = openai_client._client.post(  # noqa: SLF001
            f"{openai_client.base_url}audio/transcriptions",
            data={"model": transcription_model},
            files={"unrelated": (None, "")},
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )

        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["message"] == (
            "[{'type': 'missing', 'loc': ('body', 'file'), 'msg': 'Field required', "
            "'input': None}]"
        )
        assert (error["param"], error["code"]) == (None, None)

    @pytest.mark.parametrize("position", [0, 1])
    def test_a_text_value_among_images_names_its_position(
        self, openai_client: OpenAI, image_edit_model: str, position: int
    ) -> None:
        """A text part posted as ``image[]`` is refused at its index among the images.

        Ref: https://developers.openai.com/api/reference/resources/images/methods/edit
             stdapi/routes/openai_images_edits.py:_merge_image_parameters
        """
        images = [("image[]", ("a.png", _png(), "image/png"))] * position
        response = openai_client._client.post(  # noqa: SLF001
            f"{openai_client.base_url}images/edits",
            data={"model": image_edit_model, "prompt": "x"},
            files=[*images, ("image[]", (None, "not_a_file"))],
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )

        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert (error["param"], error["code"]) == (f"image[{position}]", "invalid_type")
        assert error["message"] == (
            f"Invalid type for 'image[{position}]': expected a file, but got a string "
            "instead."
        )


class TestAnthropicValidationErrors:
    """Anthropic routes read ``<loc>: <msg>`` with no request-part prefix.

    Ref: https://platform.claude.com/docs/en/api/errors
         stdapi/validation_errors.py:anthropic_validation_message
    """

    @pytest.fixture(autouse=True)
    def _skip_bedrock(self, is_bedrock_direct: bool) -> None:
        """Skip on Bedrock: its own validation words the refusals differently."""
        if is_bedrock_direct:
            pytest.skip("the Bedrock-hosted client words validation errors itself")

    @pytest.mark.parametrize(
        ("body", "expected"),
        [
            pytest.param({"max_tokens": 5}, "messages: Field required", id="missing"),
            pytest.param(
                {
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": [{"type": "text"}]}],
                },
                "messages.0.content.0.text.text: Field required",
                id="missing_in_an_item",
            ),
            pytest.param(
                {
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": "hi", "bogus": 1}],
                },
                "messages.0.bogus: Extra inputs are not permitted",
                id="unknown_in_an_item",
            ),
            pytest.param(
                {"max_tokens": "abc", "messages": _MESSAGES},
                "max_tokens: Input should be a valid integer",
                id="wrong_type",
            ),
            pytest.param(
                {"max_tokens": 5, "messages": _MESSAGES, "service_tier": "bogus"},
                "service_tier: Input should be 'auto'",
                id="invalid_enum",
            ),
            pytest.param(
                {
                    "max_tokens": 5,
                    "messages": _MESSAGES,
                    "tool_choice": {"type": "bogus"},
                },
                "tool_choice: Input tag 'bogus' found using 'type' does not match any "
                "of the expected tags: 'auto', 'any', 'tool', 'none'",
                id="invalid_tag",
            ),
            pytest.param(
                {"max_tokens": 5, "messages": _MESSAGES, "temperature": 5},
                "temperature: ",
                id="out_of_range",
            ),
        ],
    )
    def test_the_refusal_starts_with_the_location(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_basic_model: str,
        body: dict[str, Any],
        expected: str,
    ) -> None:
        """The message is the location then the reason, as Anthropic answers it.

        Anthropic's messages are Pydantic's, so they begin identically on both
        targets; the tail may carry extra detail (the gateway adds why a string
        does not parse, and lists the service tiers it accepts).
        """
        with pytest.raises(AnthropicBadRequestError) as excinfo:
            anthropic_client.post(
                "/v1/messages",
                body={"model": anthropic_chat_basic_model, **body},
                cast_to=httpx.Response,
            )

        assert excinfo.value.status_code == 400
        envelope = excinfo.value.body
        assert isinstance(envelope, dict)
        assert envelope["error"]["type"] == "invalid_request_error"
        assert envelope["error"]["message"].startswith(expected), envelope
