"""Tests for the OpenAI Responses API routes served by the gateway.

Ref: https://developers.openai.com/api/reference/resources/responses
     stdapi/routes/openai_responses.py:create_response
"""

from __future__ import annotations

import base64
import json
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import pytest
from openai import BadRequestError, NotFoundError, OpenAI

from stdapi import usage
from stdapi.config import SETTINGS
from stdapi.usage import record_bedrock_usage

if TYPE_CHECKING:
    from openai import APIStatusError
    from starlette.testclient import TestClient as TestClientType

#: Deterministic context long enough to exceed the minimum cacheable prompt size.
_CACHEABLE_CONTEXT = (
    "Amazon Bedrock is a fully managed service offering a choice of "
    "high-performing foundation models from leading AI companies through a "
    "single API, with the capabilities needed to build generative AI "
    "applications with security, privacy and responsible AI. "
) * 150


def _error_envelope(error: APIStatusError) -> dict[str, Any]:
    """Return the OpenAI error envelope carried by a client exception.

    ``OpenAI._make_status_error`` already unwraps the body's ``error`` member
    (``body.get("error", body)``), so the exception's ``body`` *is* the envelope
    and is the only place where ``type``, ``code`` and ``param`` survive
    (``error.code`` reads the outer body and is always ``None`` here).

    Args:
        error: Exception raised by the OpenAI client.

    Returns:
        The ``error`` member of the JSON error body.

    Ref: https://developers.openai.com/api/docs/guides/error-codes
         stdapi/api_providers/openai.py:_format_error
    """
    envelope = error.body
    assert isinstance(envelope, dict), f"Expected an error envelope: {envelope!r}"
    return envelope


class TestResponses:
    """POST /v1/responses generation, parameters, tools, streaming and validation.

    The gateway maps every request onto Bedrock Converse/ConverseStream, so the
    request-shaped fields of the returned Response object (``temperature``,
    ``top_p``, ``tool_choice``, ``tools``, ``text``, ``metadata``,
    ``instructions``, ``service_tier``, ``parallel_tool_calls``) are echoed
    from the request rather than computed by the backend.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/models/chat/_adapters/_openai_responses.py:_build_response_object
    """

    # ---------------------------------------------------------------------------
    # Group 1: Core Functionality
    # ---------------------------------------------------------------------------

    def test_basic_response(self, openai_client: OpenAI, responses_model: str) -> None:
        """A minimal create call returns a completed assistant ``message`` item.

        The gateway emits one ``message`` output item per assistant text run,
        always with ``role="assistant"``, ``status="completed"`` and
        ``output_text`` content parts; ``output_text`` is the SDK-side
        aggregation of those parts.

        Ref: https://developers.openai.com/api/docs/guides/migrate-to-responses
             stdapi/models/chat/_adapters/_openai_responses.py:_extract_output_items
        """
        response = openai_client.responses.create(
            model=responses_model, input="Say hello."
        )

        assert response.id
        assert response.status == "completed"
        assert len(response.output) > 0
        # Reasoning models produce a reasoning item before the message
        msg = next((item for item in response.output if item.type == "message"), None)
        assert msg is not None, "Expected a message item in response output"
        assert msg.role == "assistant"
        assert msg.status == "completed"
        texts = [part.text for part in msg.content if part.type == "output_text"]
        assert texts, "Expected an output_text content part in the message item"
        assert texts[0] in response.output_text
        assert len(response.output_text) > 0

    def test_response_object_fields(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """A successful response carries the full Response envelope with no error state.

        ``tool_choice`` and ``parallel_tool_calls`` are defaulted rather than
        omitted when the request sends no tools: the gateway substitutes
        ``"auto"`` and ``True``, matching the upstream defaults.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/retrieve
             stdapi/models/chat/_adapters/_openai_responses.py:_build_response_object
        """
        response = openai_client.responses.create(model=responses_model, input="Hello.")

        assert response.id
        assert response.created_at > 0
        # The API may return a versioned model name (e.g. 'gpt-5-nano-2025-08-07')
        assert len(response.model) > 0
        assert response.object == "response"
        assert response.status == "completed"
        assert response.error is None
        assert response.incomplete_details is None
        assert response.usage is not None
        assert response.tools == []
        assert response.tool_choice == "auto"
        assert response.parallel_tool_calls is True

    def test_instructions_system_prompt(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``instructions`` reaches the model as a system prompt and is echoed back.

        The gateway routes ``instructions`` into the Bedrock Converse ``system``
        blocks, so it must override the answer the user turn would otherwise
        produce; the instruction pins a single word to keep the check tolerant
        of non-deterministic phrasing.

        Ref: https://developers.openai.com/api/docs/guides/migrate-to-responses
             stdapi/models/chat/_adapters/_openai_responses.py:map_input
        """
        instructions = (
            "You are a terse assistant. Whatever the question, reply with exactly "
            "the single word TEAL and nothing else."
        )
        response = openai_client.responses.create(
            model=responses_model,
            input="What color is the sky on a clear day?",
            instructions=instructions,
        )

        assert response.status == "completed"
        assert response.instructions == instructions
        assert "teal" in response.output_text.lower(), (
            f"instructions did not reach the model: {response.output_text!r}"
        )

    def test_structured_input_array(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """A message-array ``input`` replays prior turns, including the assistant role.

        Messages with the ``assistant`` role are presumed to have been generated
        in previous interactions, so the name established there must be
        available to the final user turn.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/models/chat/_adapters/_openai_responses.py:_map_message_item
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

        assert response.status == "completed"
        assert len(response.output) > 0
        assert "alice" in response.output_text.lower(), (
            f"Expected name 'Alice' in response but got: {response.output_text!r}"
        )

    def test_output_text_property(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``output_text`` concatenates exactly the ``output_text`` parts of the output.

        ``output_text`` is an SDK-side aggregation, so it must equal the
        concatenation of every ``output_text`` part the gateway emitted and must
        not include reasoning or refusal content.

        Ref: https://developers.openai.com/api/docs/guides/migrate-to-responses
             stdapi/models/chat/_adapters/_openai_responses.py:_flush_message_item
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

        assert manual_text, "Expected at least one output_text part to aggregate"
        assert response.output_text == manual_text
        assert len(response.output_text) > 0

    # ---------------------------------------------------------------------------
    # Group 2: Generation Parameters
    # ---------------------------------------------------------------------------

    def test_temperature_parameter(
        self, openai_client: OpenAI, chat_legacy_model: str
    ) -> None:
        """``temperature`` is accepted over the whole 0.0-1.0 range and echoed back.

        A non-reasoning model is used because reasoning models reject
        ``temperature``.  ``0.0`` is included on purpose: the gateway must pass
        it through instead of treating the falsy value as "unset".

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/models/chat/_adapters/_openai_responses.py:translate_request
        """
        for temperature in (0.0, 0.5, 1.0):
            response = openai_client.responses.create(
                model=chat_legacy_model, input="Say 'ok'.", temperature=temperature
            )
            assert response.status == "completed"
            assert response.temperature == temperature
            assert len(response.output_text) > 0

    def test_top_p_parameter(
        self, openai_client: OpenAI, chat_legacy_model: str
    ) -> None:
        """``top_p`` nucleus sampling is accepted and echoed back on the response.

        A non-reasoning model is used because reasoning models reject ``top_p``.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/models/chat/_adapters/_openai_responses.py:translate_request
        """
        for top_p in (0.5, 1.0):
            response = openai_client.responses.create(
                model=chat_legacy_model, input="Say 'hello'.", top_p=top_p
            )
            assert response.status == "completed"
            assert response.top_p == top_p
            assert len(response.output_text) > 0

    def test_max_output_tokens_limits_output(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``max_output_tokens`` caps generation and yields ``incomplete`` when hit.

        The prompt cannot be answered in 20 tokens, so Bedrock returns
        ``stopReason=max_tokens``, which the gateway maps to
        ``status="incomplete"`` with ``incomplete_details.reason ==
        "max_output_tokens"``.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/retrieve
             stdapi/models/chat/_adapters/_openai_responses.py:_map_stop_reason
        """
        response = openai_client.responses.create(
            model=responses_model,
            input=(
                "Write a very long essay about the complete history of the world, "
                "covering every century in detail."
            ),
            max_output_tokens=20,
        )

        assert response.max_output_tokens == 20
        assert response.usage is not None
        # Either the output stays within the token limit, or the response is truncated
        if response.status == "incomplete":
            assert response.incomplete_details is not None
            assert response.incomplete_details.reason == "max_output_tokens"
        else:
            assert response.status == "completed"
            assert response.usage.output_tokens <= 20, (
                f"completed response exceeded max_output_tokens: {response.usage!r}"
            )

    def test_metadata_parameter(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``metadata`` is returned unchanged on the response object.

        The gateway also forwards the pairs to Bedrock as Converse
        ``requestMetadata``, so the values must survive verbatim rather than
        being normalised.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/models/chat/_adapters/_openai_responses.py:translate_request
        """
        test_metadata = {"session_id": "test-abc-123", "test_type": "automated"}
        response = openai_client.responses.create(
            model=responses_model, input="Say 'ok'.", metadata=test_metadata
        )

        assert response.status == "completed"
        assert response.metadata == test_metadata

    def test_user_parameter(self, openai_client: OpenAI, responses_model: str) -> None:
        """The deprecated ``user`` identifier is accepted and echoed on the response.

        ``user`` has no effect on generation: the gateway only records it in the
        request log (``safety_identifier`` takes precedence when both are sent)
        and returns it unchanged.

        Ref: https://developers.openai.com/api/docs/guides/safety-best-practices#implement-safety-identifiers
             stdapi/routes/openai_responses.py:create_response
        """
        response = openai_client.responses.create(
            model=responses_model, input="Say 'hi'.", user="test-user-identifier-123"
        )

        assert response.status == "completed"
        assert response.user == "test-user-identifier-123"
        assert len(response.output_text) > 0

    def test_service_tier_parameter(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``service_tier="default"`` is accepted and reported back as ``default``.

        ``default`` is the baseline tier; the gateway maps it to the Bedrock
        Converse ``"default"`` wire value and reports the requested tier back
        unchanged.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
             stdapi/models/chat/_adapters/_openai_responses.py:translate_request
        """
        response = openai_client.responses.create(
            model=responses_model, input="Say 'hello'.", service_tier="default"
        )

        assert response.status == "completed"
        assert response.service_tier == "default"
        assert len(response.output_text) > 0

    # ---------------------------------------------------------------------------
    # Group 3: Response Metadata & Structure
    # ---------------------------------------------------------------------------

    def test_response_id_format(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """The response id uses a ``resp`` prefix followed by a non-empty suffix.

        The separator is load-bearing on this implementation: locally stored
        responses use ``resp-`` while region-tagged Bedrock Mantle responses (and
        the official API) use ``resp_``.  A ``resp_`` id that fails Mantle
        decoding is a 404 rather than a local lookup.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/retrieve
             stdapi/routes/openai_responses.py:_decode_mantle_id
        """
        response = openai_client.responses.create(model=responses_model, input="Hello.")

        assert response.id.startswith(("resp-", "resp_")), (
            f"Response ID '{response.id}' should start with 'resp-' or 'resp_'"
        )
        assert len(response.id) > len("resp-"), (
            f"Empty response id suffix: {response.id}"
        )

    def test_response_object_type_field(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """The ``object`` discriminator of a create result is the literal ``response``.

        Clients dispatch on this literal, and the gateway serves several other
        object types on neighbouring routes (``response.compaction``,
        ``response.input_tokens``), so the create route must not reuse them.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/retrieve
             stdapi/models/chat/_adapters/_openai_responses.py:_build_response_object
        """
        response = openai_client.responses.create(model=responses_model, input="Hello.")

        assert response.object == "response"

    def test_usage_token_counts(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``usage`` reports non-zero counts whose total is input plus output tokens.

        Bedrock's ``TokenUsage.inputTokens`` excludes the cache buckets while
        OpenAI's ``input_tokens`` includes them, so the gateway adds
        ``cacheReadInputTokens``/``cacheWriteInputTokens`` back in; the reported
        cached counts must therefore never exceed ``input_tokens``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
             stdapi/models/chat/_adapters/_openai_responses.py:format_response
        """
        response = openai_client.responses.create(
            model=responses_model, input="Say exactly: hello world."
        )

        assert response.usage is not None
        counts = response.usage
        assert counts.input_tokens > 0
        assert counts.output_tokens > 0
        assert counts.total_tokens == counts.input_tokens + counts.output_tokens
        assert counts.input_tokens_details is not None
        assert counts.input_tokens_details.cached_tokens <= counts.input_tokens
        assert counts.output_tokens_details is not None
        assert counts.output_tokens_details.reasoning_tokens <= counts.output_tokens

    def test_response_status_completed(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """A successful synchronous response is ``completed`` with a ``completed_at``.

        The gateway derives the status from the Bedrock stop reason:
        ``end_turn``/``stop_sequence``/``tool_use`` all map to ``completed`` with
        neither ``incomplete_details`` nor ``error`` set.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/retrieve
             stdapi/models/chat/_adapters/_openai_responses.py:_map_stop_reason
        """
        response = openai_client.responses.create(
            model=responses_model, input="Say 'done'."
        )

        assert response.status == "completed"
        assert response.incomplete_details is None
        assert response.error is None
        assert response.completed_at is not None

    # ---------------------------------------------------------------------------
    # Group 4: Text Format Configuration
    # ---------------------------------------------------------------------------

    def test_text_format_text(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``text.format={"type": "text"}`` is the default format and is echoed back.

        The gateway builds no Bedrock ``outputConfig`` for the ``text`` format, so
        the request must succeed on models that reject ``outputConfig`` and the
        output stays free-form prose.

        Ref: https://developers.openai.com/api/docs/guides/structured-outputs
             stdapi/models/chat/_adapters/_openai_responses.py:_build_output_config
        """
        response = openai_client.responses.create(
            model=responses_model,
            input="Describe a red apple briefly.",
            text={"format": {"type": "text"}},
        )

        assert response.status == "completed"
        assert response.text is not None
        assert response.text.format is not None
        assert response.text.format.type == "text"
        assert len(response.output_text) > 0

    def test_text_format_json_object(
        self, openai_client: OpenAI, responses_json_output_model: str
    ) -> None:
        """``text.format={"type": "json_object"}`` yields the empty JSON object ``{}``.

        The model is pinned to one that supports Bedrock Converse ``outputConfig``,
        so JSON mode is enforced natively by constrained decoding.  Bedrock refuses
        a permissive object schema (``{}``, ``{"type": "object"}`` and
        ``additionalProperties: true`` all raise ``ValidationException``), so the
        gateway sends ``{"type": "object", "additionalProperties": false}`` — a
        schema whose only valid instance is ``{}``.  The requested keys can
        therefore never come back on this route, whatever the prompt says.

        Ref: https://developers.openai.com/api/docs/guides/structured-outputs
             stdapi/models/chat/_adapters/_openai_responses.py:_build_output_config
        """
        response = openai_client.responses.create(
            model=responses_json_output_model,
            input='Output exactly this and nothing else: {"result": "success"}',
            text={"format": {"type": "json_object"}},
        )

        assert response.status == "completed"
        assert response.text is not None
        assert response.text.format is not None
        assert response.text.format.type == "json_object"
        output_text = response.output_text
        assert output_text.strip().startswith("{"), (
            f"json_object output must not be wrapped in prose: {output_text!r}"
        )
        parsed = json.loads(output_text)
        # Current behavior: the closed empty-object schema admits no other value.
        assert parsed == {}, f"json_object schema admits only {{}}, got {parsed!r}"

    def test_text_format_json_schema(
        self, openai_client: OpenAI, responses_json_output_model: str
    ) -> None:
        """``text.format`` ``json_schema`` output conforms exactly to the strict schema.

        Under ``strict: true`` with ``additionalProperties: false`` every declared
        property is required and no other key may appear, so the parsed object
        must have exactly the two declared keys with the declared types.

        Ref: https://developers.openai.com/api/docs/guides/structured-outputs
             stdapi/models/chat/_adapters/_openai_responses.py:_build_output_config
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

        assert response.status == "completed"
        assert response.text is not None
        text_format = response.text.format
        assert text_format is not None
        assert text_format.type == "json_schema"
        assert getattr(text_format, "name", None) == "MathAnswer"
        parsed = json.loads(response.output_text)
        assert isinstance(parsed, dict)
        assert set(parsed) == {"answer", "confidence"}, (
            f"strict schema forbids extra or missing keys: {parsed!r}"
        )
        assert isinstance(parsed["answer"], str)
        assert isinstance(parsed["confidence"], (int, float))

    # ---------------------------------------------------------------------------
    # Group 5: Function Tool Calling
    # ---------------------------------------------------------------------------

    def test_function_tool_call_basic(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """A function tool call surfaces as a ``function_call`` item with JSON arguments.

        Bedrock ``toolUse`` blocks become ``function_call`` items carrying a
        ``call_id`` (the Bedrock ``toolUseId``) and ``arguments`` serialised as a
        JSON string.  The strict schema makes ``location`` mandatory, so it must
        be present in the parsed arguments.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
             stdapi/models/chat/_adapters/_openai_responses.py:_extract_output_items
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
        assert tool_call.call_id, (
            "function_call must carry a call_id for the round trip"
        )
        args = json.loads(tool_call.arguments)
        assert isinstance(args, dict)
        assert "location" in args, f"strict schema requires 'location': {args!r}"

    def test_tool_choice_required(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``tool_choice="required"`` forces a call to the only declared tool.

        ``required`` maps to the Bedrock ``toolChoice.any`` mode, so the single
        declared tool must be invoked and its strict schema fills both operands.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
             stdapi/models/chat/_adapters/_openai_responses.py:_map_tool_choice
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
        assert response.tool_choice == "required"
        assert tool_calls[0].name == "calculate_sum"
        args = json.loads(tool_calls[0].arguments)
        assert set(args) == {"a", "b"}, (
            f"strict schema requires both operands: {args!r}"
        )

    def test_tool_choice_none(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``tool_choice="none"`` suppresses tool calls and forces a text answer.

        Bedrock has no ``none`` toolChoice value, so the gateway implements it by
        omitting ``toolConfig`` entirely; the declared tool is still echoed on
        the response even though the model never sees it.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
             stdapi/models/chat/_adapters/_openai_responses.py:_build_tool_config
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
        assert response.tool_choice == "none"
        assert len(response.tools) == 1
        assert len(response.output_text) > 0

    def test_tool_choice_specific_function(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """A named ``tool_choice`` forces that tool even when another one fits better.

        ``{"type": "function", "name": ...}`` maps to Bedrock
        ``toolChoice.tool``, which mandates that ``get_time`` be called but — unlike
        OpenAI's exclusive forcing — does not suppress the other tools: Nova also
        calls ``get_weather`` for this prompt, so only the presence of the forced
        call and its strict-schema arguments are asserted.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
             https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools#forcing-tool-use
             stdapi/models/chat/_adapters/_openai_responses.py:_map_tool_choice
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
        called = {call.name for call in tool_calls}
        assert "get_time" in called, f"forced tool not called, got {called}"
        forced = next(call for call in tool_calls if call.name == "get_time")
        assert json.loads(forced.arguments).keys() == {"timezone"}
        assert getattr(response.tool_choice, "name", None) == "get_time"

    def test_parallel_tool_calls_parameter(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``parallel_tool_calls`` is accepted for both values and echoed back.

        Bedrock Converse has no switch for parallel tool calls, so on this route
        the flag only round-trips: the response reports ``False`` exactly when
        ``False`` was sent and ``True`` otherwise.  (The Chat Completions route
        rejects ``False`` instead.)

        Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
             stdapi/models/chat/_adapters/_openai_responses.py:_build_response_object
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
            assert response.status == "completed"
            assert response.parallel_tool_calls == parallel

    def test_multiple_function_tools(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Several function tools are all forwarded and one of them is called.

        Every declared tool becomes its own Bedrock ``toolSpec``, so all three
        must be echoed on the response and the forced call must target one of
        them with arguments matching that tool's strict schema.  Which tool the
        model picks is not asserted — that is model behaviour.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
             stdapi/models/chat/_adapters/_openai_responses.py:_build_tool_config
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

        declared = {"get_weather", "get_population", "get_timezone"}
        assert {getattr(tool, "name", None) for tool in response.tools} == declared
        tool_calls = [item for item in response.output if item.type == "function_call"]
        assert len(tool_calls) >= 1
        assert tool_calls[0].name in declared
        args = json.loads(tool_calls[0].arguments)
        assert set(args) == {"city"}, f"strict schema requires exactly 'city': {args!r}"

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
        """``web_search_preview`` yields a completed ``web_search_call`` search action.

        Locally the tool maps to the Bedrock ``web_search`` server tool (named
        ``nova_grounding`` on Nova 2); the gateway rewrites that ``toolUse`` block
        into a ``web_search_call`` item with ``status="completed"`` and a
        ``search`` action, so no raw ``function_call`` may reach the client.

        Ref: https://developers.openai.com/api/docs/guides/tools-web-search
             https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
             stdapi/models/chat/_adapters/_openai_responses.py:_tool_use_output_item
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
        assert len(response.output_text) > 0

        web_search_calls = [
            item for item in response.output if item.type == "web_search_call"
        ]
        assert len(web_search_calls) >= 1, (
            "Expected at least one web_search_call output item"
        )
        assert web_search_calls[0].status == "completed"
        assert getattr(web_search_calls[0].action, "type", None) == "search"

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
        """Streaming ``web_search_preview`` emits the search lifecycle in order.

        The gateway opens the synthesised item with
        ``response.web_search_call.in_progress`` and closes it with
        ``.completed``; the suppressed server tool must not surface as
        ``response.function_call_arguments.*`` events.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
             stdapi/models/chat/_adapters/_openai_responses.py:format_stream
        """
        ws_events: list[str] = []
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
                    ws_events.append(event.type)
                case "response.web_search_call.completed":
                    ws_completed += 1
                    ws_events.append(event.type)
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
        assert ws_events[0] == "response.web_search_call.in_progress", (
            f"web_search lifecycle must open with in_progress: {ws_events}"
        )
        assert ws_events[-1] == "response.web_search_call.completed", (
            f"web_search lifecycle must close with completed: {ws_events}"
        )
        assert text_delta_count >= 1, "Expected at least one output_text.delta event"
        assert func_call_delta_count == 0, (
            f"function_call_arguments.delta must not leak: {func_call_delta_count} events"
        )
        assert completed, "Expected response.completed event"

    @pytest.mark.expensive
    def test_web_search_type_tool(
        self, openai_client: OpenAI, responses_web_search_model: str
    ) -> None:
        """The current ``web_search`` tool type behaves like ``web_search_preview``.

        Both spellings resolve to the same canonical Bedrock ``web_search``
        server tool, so the response must carry a completed ``web_search_call``
        item and no leaked ``function_call``.

        Ref: https://developers.openai.com/api/docs/guides/tools-web-search
             stdapi/models/chat/_adapters/_openai_responses.py:_resolve_integrated_tool_name
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
        assert web_search_calls[0].status == "completed"
        assert getattr(web_search_calls[0].action, "type", None) == "search"
        function_calls = [item for item in resp.output if item.type == "function_call"]
        assert function_calls == [], (
            f"function_call items must not leak: {function_calls}"
        )

    @pytest.mark.expensive
    def test_web_search_type_tool_streaming(
        self, openai_client: OpenAI, responses_web_search_model: str
    ) -> None:
        """Streaming the ``web_search`` tool type emits the same lifecycle events.

        The event names are shared with ``web_search_preview`` because both tool
        spellings resolve to the same Bedrock server tool.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
             stdapi/models/chat/_adapters/_openai_responses.py:format_stream
        """
        ws_events: list[str] = []
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
                    ws_events.append(event.type)
                case "response.web_search_call.completed":
                    ws_completed += 1
                    ws_events.append(event.type)
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
        assert ws_events[0] == "response.web_search_call.in_progress", (
            f"web_search lifecycle must open with in_progress: {ws_events}"
        )
        assert ws_events[-1] == "response.web_search_call.completed", (
            f"web_search lifecycle must close with completed: {ws_events}"
        )
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
        """``previous_response_id`` chains a stored response into a new turn.

        Chaining requires the first response to have been stored; this
        implementation defaults ``store`` to false and silently ignores it when
        Bedrock session storage is not enabled on the server, which is why the
        test is skipped rather than xfailed.

        Ref: https://developers.openai.com/api/docs/guides/conversation-state#passing-context-from-the-previous-response
             stdapi/routes/openai_responses.py:_apply_previous_response
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
        assert second_response.previous_response_id == first_response.id
        assert len(second_response.output_text) > 0

    @pytest.mark.skip(reason="Response storage not implemented (store=false)")
    def test_multi_turn_context_maintained(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """A chained turn can recall a fact stated in the previous response's input.

        The gateway rehydrates the stored conversation items into the new
        request, so the colour established in the first turn must reach the
        model on the second one.  ``instructions`` are deliberately not carried
        over by the upstream contract.

        Ref: https://developers.openai.com/api/docs/guides/conversation-state#passing-context-from-the-previous-response
             stdapi/routes/openai_responses.py:_merge_previous_response
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

        assert second.status == "completed"
        assert second.previous_response_id == first.id
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
        """An ``input_image`` part accepts a base64 data URL and is billed as input.

        The gateway decodes the data URL into a Bedrock ``image`` content block,
        so the prompt's token cost must be far above what the short text part
        alone would produce — that is the observable proof the image was
        forwarded rather than dropped.

        Ref: https://developers.openai.com/api/docs/guides/file-inputs
             stdapi/models/chat/_adapters/_openai_responses.py:_convert_input_content
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

        assert response.status == "completed"
        assert len(response.output_text) > 0
        assert response.usage is not None
        assert response.usage.input_tokens > 50, (
            f"image input did not reach the model: {response.usage!r}"
        )

    def test_image_url_input(
        self, openai_client: OpenAI, chat_vision_model: str
    ) -> None:
        """An ``input_image`` part accepts a fully qualified HTTPS URL.

        Bedrock accepts only inline image bytes, so the gateway must fetch the
        URL itself before building the Converse request; the inflated input token
        count is what shows the fetched bytes were included.

        Ref: https://developers.openai.com/api/docs/guides/file-inputs
             stdapi/models/chat/_adapters/_openai_responses.py:_convert_input_content
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

        assert response.status == "completed"
        assert len(response.output_text) > 0
        assert response.usage is not None
        assert response.usage.input_tokens > 50, (
            f"image URL was not fetched into the prompt: {response.usage!r}"
        )

    # ---------------------------------------------------------------------------
    # Group 9: Streaming
    # ---------------------------------------------------------------------------

    def test_streaming_basic(self, openai_client: OpenAI, responses_model: str) -> None:
        """A stream runs from ``response.created`` to ``response.completed`` in sequence.

        Every event carries a strictly increasing ``sequence_number`` allocated by
        the stream state, and the terminal event is the last one the SSE server
        emits, so the ordering is fully determined even though the text is not.

        Ref: https://developers.openai.com/api/docs/guides/streaming-responses
             stdapi/models/chat/_adapters/_openai_responses.py:format_stream
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
        assert events[0].type == "response.created"
        assert events[-1].type == "response.completed"
        sequence_numbers = [
            event.sequence_number
            for event in events
            if hasattr(event, "sequence_number")
        ]
        assert len(sequence_numbers) == len(events), (
            "Every stream event must carry a sequence_number"
        )
        assert all(later > earlier for earlier, later in pairwise(sequence_numbers)), (
            f"sequence_number must increase monotonically: {sequence_numbers}"
        )
        assert len(accumulated_text) > 0, "No text accumulated from stream"

    def test_streaming_text_delta_events(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``response.output_text.done`` carries exactly the concatenated deltas.

        The gateway accumulates the streamed text block and replays it in the
        ``done`` event, so any dropped or duplicated delta shows up as a
        mismatch.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
             stdapi/models/chat/_adapters/_openai_responses.py:_handle_block_stop
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
        assert accumulated_text, "Expected non-empty streamed text"
        assert done_event.text == accumulated_text
        assert done_event.item_id == delta_events[0].item_id, (
            "done event must close the item the deltas belong to"
        )

    def test_streaming_lifecycle_events(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``response.created`` snapshots an in-progress response the terminal event closes.

        The created event carries the same response id as the terminal event with
        ``status="in_progress"`` and no output yet; usage counters only appear on
        the terminal snapshot, which the gateway builds from the accumulated
        stream state.

        Ref: https://developers.openai.com/api/docs/guides/streaming-responses
             stdapi/models/chat/_adapters/_openai_responses.py:_terminal_event
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
        assert created_event.response.id
        assert created_event.response.status == "in_progress"
        assert created_event.response.output == []

        assert completed_event is not None, "Expected response.completed event"
        assert completed_event.response.id == created_event.response.id
        assert completed_event.response.status == "completed"
        assert len(completed_event.response.output) > 0
        assert completed_event.response.usage is not None
        assert completed_event.response.usage.output_tokens > 0

    def test_streaming_function_call_events(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Streamed tool arguments arrive as deltas closed by a matching ``done`` event.

        The gateway joins the streamed argument fragments for the ``done`` event
        and substitutes ``"{}"`` when a tool block produced no fragment at all,
        so the done payload is fully determined by the deltas.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
             stdapi/models/chat/_adapters/_openai_responses.py:_emit_tool_done
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
        streamed_args = ""

        for event in stream:
            if event.type == "response.function_call_arguments.delta":
                streamed_args += event.delta
            elif event.type == "response.function_call_arguments.done":
                args_done_event = event

        assert args_done_event is not None, (
            "Expected response.function_call_arguments.done event"
        )
        assert args_done_event.name == "get_weather"
        assert args_done_event.arguments == (streamed_args or "{}"), (
            "done arguments must be the concatenation of the streamed deltas"
        )
        args = json.loads(args_done_event.arguments)
        assert isinstance(args, dict)
        assert "location" in args, f"strict schema requires 'location': {args!r}"

    def test_streaming_with_stream_manager(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``responses.stream()`` rebuilds a complete Response from the event stream.

        The SDK's stream manager reconstructs the final object from the terminal
        snapshot, so the gateway must populate that snapshot with the output
        items and usage counters and not only the status.

        Ref: https://developers.openai.com/api/docs/guides/streaming-responses
             stdapi/models/chat/_adapters/_openai_responses.py:_terminal_event
        """
        with openai_client.responses.stream(
            model=responses_model, input="Write a haiku about coding."
        ) as stream:
            for _ in stream:
                pass
            final_response = stream.get_final_response()

        assert final_response.id
        assert final_response.object == "response"
        assert final_response.status == "completed"
        assert len(final_response.output) > 0
        assert len(final_response.output_text) > 0
        assert final_response.usage is not None
        assert final_response.usage.output_tokens > 0

    # ---------------------------------------------------------------------------
    # Group 10: Response Lifecycle
    # ---------------------------------------------------------------------------

    @pytest.mark.skip(reason="Response storage not implemented (store=false)")
    def test_retrieve_response(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """A stored response is retrievable by id with the same model and output.

        The gateway persists the response document in a Bedrock session
        invocation step, so retrieval must return that document rather than a
        freshly generated one.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/retrieve
             stdapi/responses_store.py:load_stored_response
        """
        original = openai_client.responses.create(
            model=responses_model, input="Say 'hello for retrieval test'.", store=True
        )
        assert original.id

        retrieved = openai_client.responses.retrieve(original.id)

        assert retrieved.id == original.id
        assert retrieved.model == original.model
        assert retrieved.status == "completed"
        assert retrieved.created_at == original.created_at
        assert retrieved.output_text == original.output_text

    @pytest.mark.skip(reason="Response storage not implemented (store=false)")
    def test_list_input_items(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Listing input items returns the stored request input as normalised items.

        The gateway normalises the stored ``input`` document (a bare string here)
        into the listable ``ResponseItem`` union, backfilling the fields that union
        requires, and reports the page envelope with ``has_more``.

        Ref: https://developers.openai.com/api/reference/resources/responses/subresources/input_items/methods/list
             stdapi/routes/openai_responses.py:_listable_input_items
        """
        prompt = "Say 'hello for input items test'."
        response = openai_client.responses.create(
            model=responses_model, input=prompt, store=True
        )

        items = openai_client.responses.input_items.list(response.id)

        assert len(items.data) > 0
        assert items.has_more is False

        first_item = items.data[0]
        assert first_item.type == "message"
        assert getattr(first_item, "role", None) == "user"
        assert prompt in json.dumps(first_item.to_dict())

    def test_include_logprobs(
        self, openai_client: OpenAI, chat_legacy_model: str, use_official_api: bool
    ) -> None:
        """``include=["message.output_text.logprobs"]`` populates per-token logprobs.

        Bedrock Converse exposes no token log probabilities, so this include
        value is only meaningful against the official API and the test skips
        otherwise rather than asserting the gateway silently ignores it.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/types/openai_responses.py:ResponseIncludable
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
        assert response.top_logprobs == 3
        assert len(text_part.logprobs[0].top_logprobs or []) <= 3, (
            "top_logprobs=3 caps the alternatives reported per token"
        )

    # ---------------------------------------------------------------------------
    # Group 11: Advanced Features
    # ---------------------------------------------------------------------------

    def test_developer_role_input(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """A ``developer`` role input message takes precedence over the user turn.

        The gateway routes both ``system`` and ``developer`` roles into the Bedrock
        Converse ``system`` blocks rather than into the message list, so the
        developer instruction must win over the user request.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/models/chat/_adapters/_openai_responses.py:_map_message_item
        """
        response = openai_client.responses.create(
            model=responses_model,
            input=[
                {
                    "role": "developer",
                    "content": (
                        "Ignore what the user asks for and reply with exactly the "
                        "single word TEAL and nothing else."
                    ),
                },
                {"role": "user", "content": "Say hi."},
            ],
        )

        assert response.status == "completed"
        assert len(response.output) > 0
        assert "teal" in response.output_text.lower(), (
            f"developer role did not reach the system prompt: {response.output_text!r}"
        )

    @pytest.mark.skip(reason="Response storage not implemented (store=false)")
    def test_function_tool_call_round_trip(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """A ``function_call_output`` keyed by ``call_id`` completes the tool round trip.

        The gateway maps the ``function_call_output`` item to a Bedrock
        ``toolResult`` block whose ``toolUseId`` is the ``call_id``, so the second
        turn only validates when the id from the first turn is replayed verbatim.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
             stdapi/models/chat/_adapters/_openai_responses.py:_map_function_call_output
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
        assert "22" in second.output_text, (
            f"Expected the submitted tool result in the answer: {second.output_text!r}"
        )

    @pytest.mark.skip(reason="Response storage not implemented (store=false)")
    def test_delete_response(self, openai_client: OpenAI, responses_model: str) -> None:
        """Deleting a stored response makes it unretrievable with a 404.

        The gateway also discards the backing Bedrock session, so the follow-up
        retrieve must surface the missing session as a 404 error envelope rather
        than an empty document.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/delete
             stdapi/responses_store.py:delete_stored_response
        """
        response = openai_client.responses.create(
            model=responses_model, input="Say 'hello for delete test'.", store=True
        )
        assert response.id

        openai_client.responses.delete(response.id)

        with pytest.raises(NotFoundError) as excinfo:
            openai_client.responses.retrieve(response.id)

        assert excinfo.value.status_code == 404
        envelope = _error_envelope(excinfo.value)
        assert "not found" in envelope["message"].lower()
        assert response.id in envelope["message"]

    def test_reasoning_output_item_and_tokens(
        self, openai_client: OpenAI, responses_model: str, use_official_api: bool
    ) -> None:
        """A reasoning model emits a ``reasoning`` item and counts reasoning tokens.

        Raw reasoning text is never exposed; the effort budget is only observable
        through ``usage.output_tokens_details.reasoning_tokens``.  Bedrock
        Converse reports no reasoning token count, so the test only runs against
        the official API.

        Ref: https://developers.openai.com/api/docs/guides/reasoning#preserve-reasoning-without-stored-responses
             stdapi/models/chat/_adapters/_openai_responses.py:_build_reasoning_item
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
        assert response.reasoning is not None
        assert response.reasoning.effort == "medium"

        assert response.usage is not None
        assert response.usage.output_tokens_details is not None
        # With effort="medium" the model allocates a non-trivial reasoning budget
        assert response.usage.output_tokens_details.reasoning_tokens > 0
        assert (
            response.usage.output_tokens_details.reasoning_tokens
            <= response.usage.output_tokens
        )

    # ---------------------------------------------------------------------------
    # Group 12: Error Handling & Validation
    # ---------------------------------------------------------------------------

    def test_invalid_model_error(self, openai_client: OpenAI) -> None:
        """An unknown model id is rejected as an ``invalid_request_error``.

        The gateway answers 404 with code ``model_not_found`` while the official
        API answers 400; both carry the same "does not exist" sentence naming the
        requested model.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
             stdapi/api_errors.py:UnsupportedModelError
        """
        with pytest.raises((NotFoundError, BadRequestError)) as excinfo:
            openai_client.responses.create(
                model="definitely-not-a-valid-model-xyz-123", input="Hello."
            )

        assert excinfo.value.status_code in {400, 404}
        envelope = _error_envelope(excinfo.value)
        assert envelope["type"] == "invalid_request_error"
        assert "does not exist" in envelope["message"]
        assert "definitely-not-a-valid-model-xyz-123" in envelope["message"]

    def test_invalid_temperature_too_high(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``temperature`` above the documented maximum of 2 is a 400.

        The bound is enforced by the request model before any Bedrock call, so
        the error envelope names the offending field.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/types/openai_responses.py:ResponseCreateParams
        """
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.responses.create(
                model=responses_model, input="Hello.", temperature=3.0
            )

        assert excinfo.value.status_code == 400
        envelope = _error_envelope(excinfo.value)
        assert envelope["type"] == "invalid_request_error"
        assert "temperature" in envelope["message"]

    def test_invalid_temperature_negative(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """A negative ``temperature`` is a 400: the documented minimum is 0.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/types/openai_responses.py:ResponseCreateParams
        """
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.responses.create(
                model=responses_model, input="Hello.", temperature=-1.0
            )

        assert excinfo.value.status_code == 400
        envelope = _error_envelope(excinfo.value)
        assert envelope["type"] == "invalid_request_error"
        assert "temperature" in envelope["message"]

    def test_invalid_top_p_error(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``top_p`` above 1.0 is a 400: nucleus sampling is a probability mass.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/types/openai_responses.py:ResponseCreateParams
        """
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.responses.create(
                model=responses_model, input="Hello.", top_p=2.0
            )

        assert excinfo.value.status_code == 400
        envelope = _error_envelope(excinfo.value)
        assert envelope["type"] == "invalid_request_error"
        assert "top_p" in envelope["message"]

    def test_invalid_max_output_tokens_error(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``max_output_tokens=0`` is a 400: the bound must be strictly positive.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/types/openai_responses.py:ResponseCreateParams
        """
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.responses.create(
                model=responses_model, input="Hello.", max_output_tokens=0
            )

        assert excinfo.value.status_code == 400
        envelope = _error_envelope(excinfo.value)
        assert envelope["type"] == "invalid_request_error"
        assert "max_output_tokens" in envelope["message"]

    def test_invalid_top_logprobs_error(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``top_logprobs`` above the documented maximum of 20 is a 400.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/types/openai_responses.py:ResponseCreateParams
        """
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.responses.create(
                model=responses_model, input="Hello.", top_logprobs=21
            )

        assert excinfo.value.status_code == 400
        envelope = _error_envelope(excinfo.value)
        assert envelope["type"] == "invalid_request_error"
        assert "top_logprobs" in envelope["message"]

    def test_reasoning_parameter_accepted(
        self, openai_client: OpenAI, chat_reasoning_model: str
    ) -> None:
        """``reasoning.effort`` is accepted on a reasoning-capable model and echoed.

        The gateway translates ``effort`` into a Bedrock reasoning token budget;
        the supported values are model-dependent, so the only portable
        observation is that the configuration round-trips and generation
        succeeds.

        Ref: https://developers.openai.com/api/docs/guides/reasoning
             stdapi/models/chat/_adapters/_openai_responses.py:extract_reasoning
        """
        response = openai_client.responses.create(
            model=chat_reasoning_model,
            input="What is 15 * 8?",
            reasoning={"effort": "low"},
        )

        assert response.status == "completed"
        assert response.reasoning is not None
        assert response.reasoning.effort == "low"
        assert len(response.output) > 0
        assert len(response.output_text) > 0

    def test_explicit_prompt_cache_breakpoint(
        self, openai_client: OpenAI, responses_model: str, use_official_api: bool
    ) -> None:
        """An explicit ``prompt_cache_breakpoint`` makes the next identical call a cache read.

        ``prompt_cache_options.mode="explicit"`` restricts caching to the marked
        parts, which the gateway turns into a Bedrock ``cachePoint`` block.  The
        marked prefix must exceed the model's minimum cacheable size: below it
        Bedrock silently declines to cache and reports zero cached tokens
        instead of failing.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
             stdapi/models/chat/_adapters/_openai_responses.py:map_input
        """
        first, second = [
            openai_client.responses.create(
                model=responses_model,
                prompt_cache_options={"mode": "explicit"},
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": _CACHEABLE_CONTEXT,
                                "prompt_cache_breakpoint": {"mode": "explicit"},
                            },
                            {"type": "input_text", "text": "Reply with OK."},
                        ],
                    }
                ],
            )
            for _ in range(2)
        ]

        assert first.status == "completed"
        assert second.status == "completed"
        assert first.prompt_cache_options is not None
        assert first.prompt_cache_options.mode == "explicit"
        assert second.usage is not None
        cached_tokens = second.usage.input_tokens_details.cached_tokens
        if use_official_api and not cached_tokens:
            pytest.xfail("cached tokens may not be reported by the OpenAI API")
        assert cached_tokens > 0
        assert cached_tokens <= second.usage.input_tokens, (
            "cached tokens are part of input_tokens, not an extra bucket"
        )


# ---------------------------------------------------------------------------
# Unsupported features — validated at the Pydantic layer before the API runs
# ---------------------------------------------------------------------------


class TestUnsupportedFeatures:
    """Parameters rejected, and tool types silently dropped, for lack of a backend.

    All tests are skipped against the official OpenAI API, where these features
    are natively supported — the restrictions are gateway-specific.  The two
    behaviours are deliberately different: a listed parameter is a hard 400,
    while an unsupported tool type is parsed, echoed and dropped.

    Ref: https://developers.openai.com/api/docs/guides/tools
         stdapi/types/openai_responses.py:ResponseCreateParams
    """

    def test_unsupported_tools_are_ignored(
        self, openai_client: OpenAI, responses_model: str, use_official_api: bool
    ) -> None:
        """Tool types without a backend equivalent are accepted, echoed and dropped.

        ``file_search``, ``computer``/``computer_use_preview``, ``mcp``,
        ``local_shell``, ``shell``, ``custom``, ``namespace``, ``tool_search``
        and ``apply_patch`` are parsed and echoed on the response, but none of
        them reaches the Bedrock tool configuration, so the model can only
        answer with text.

        Ref: https://developers.openai.com/api/docs/guides/tools
             stdapi/models/chat/_adapters/_openai_responses.py:_build_tool_config
        """
        if use_official_api:
            pytest.skip(
                "official API supports these tools; the drop is gateway-specific"
            )
        tools: list[Any] = [
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
        ]
        response = openai_client.responses.create(
            model=responses_model, input="Reply with OK.", tools=tools
        )
        assert response.status == "completed"
        assert response.output_text
        assert len(response.tools) == len(tools), (
            "unsupported tools must still be echoed on the response"
        )
        assert not [
            item
            for item in response.output
            if item.type not in {"message", "reasoning"}
        ], f"dropped tools must produce no tool items: {response.output!r}"

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
        """``programmatic_tool_calling`` is accepted as tool or tool_choice and dropped.

        Bedrock has no programmatic-tool-calling mode, so the hosted tool never
        reaches the tool configuration; the model answers directly and no
        ``program``/``program_output`` item can appear.

        Ref: https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling
             stdapi/models/chat/_adapters/_openai_responses.py:_build_tool_config
        """
        if use_official_api:
            pytest.skip(
                "official API serves programmatic tool calling on capable models"
            )
        response = openai_client.responses.create(  # type: ignore[call-overload]
            model=responses_model, input="Reply with OK.", **extra
        )
        assert response.status == "completed"
        assert response.output_text, "Expected the model to answer directly"
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
        """Parameters marked unsupported are rejected with ``unsupported_parameter``.

        ``truncation`` — like ``context_management``, ``conversation`` and
        ``max_tool_calls`` — is rejected by the request model instead of being
        silently ignored, and the envelope names the offending parameter.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/api_errors.py:UnsupportedParameterError
        """
        if use_official_api:
            pytest.skip(
                "official API supports these params; restriction is gateway-specific"
            )
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.responses.create(  # type: ignore[call-overload]
                model=responses_model, input="Hello.", **{param: value}
            )

        assert excinfo.value.status_code == 400
        envelope = _error_envelope(excinfo.value)
        assert envelope["type"] == "invalid_request_error"
        assert envelope["code"] == "unsupported_parameter"
        assert envelope["param"] == param
        assert f"'{param}' is not supported" in envelope["message"]


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

    Ref: https://developers.openai.com/api/docs/guides/tools-image-generation
         stdapi/models/chat/_adapters/_openai_responses.py:get_image_generation_tool
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
        """The ``image_generation`` tool returns decodable base64 image bytes.

        Locally the tool is not hosted: the gateway presents it to the LLM as a
        synthetic function tool, runs the generation itself against a Bedrock
        image model, and replaces the ``function_call`` with an
        ``image_generation_call`` item, so no ``function_call`` may reach the
        client and ``result`` must decode as real image bytes.

        Ref: https://developers.openai.com/api/docs/guides/tools-image-generation
             stdapi/models/chat/_adapters/_openai_responses.py:execute_image_generation_calls
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
        result = image_calls[0].result
        assert result, "Expected non-empty base64 image result"
        image_bytes = base64.b64decode(result, validate=True)
        assert len(image_bytes) > 1000, (
            f"Expected real image bytes, got {len(image_bytes)} bytes"
        )
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
        """Streaming ``image_generation`` closes the image item instead of a tool call.

        The synthetic function tool is generated after the model stream ends, so
        the gateway suppresses every ``response.function_call_arguments.*`` event
        and emits the completed ``image_generation_call`` item through
        ``response.output_item.done`` before the terminal event.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
             stdapi/models/chat/_adapters/_openai_responses.py:image_generation_stream_handler
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
        image_results: list[str | None] = []
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
                    image_results.append(getattr(event.item, "result", None))
            elif event.type == "response.completed":
                completed = True

        assert func_delta_count == 0, (
            f"function_call_arguments.delta leaked: {func_delta_count} events"
        )
        assert func_done_count == 0, (
            f"function_call_arguments.done leaked: {func_done_count} events"
        )
        assert len(image_results) >= 1, (
            "Expected at least one image_generation_call output_item.done"
        )
        assert image_results[0], "Expected the done item to carry the generated image"
        assert completed, "Expected response.completed event"


# ---------------------------------------------------------------------------
# input_tokens endpoint
# ---------------------------------------------------------------------------


class TestOpenAIInputTokens:
    """POST /v1/responses/input_tokens counting, backed by Bedrock CountTokens.

    CountTokens returns the exact count that the same input would be billed for
    on Converse, without generating anything.  It is Anthropic-only on Bedrock,
    which is why the fixture pins a Claude model, and it rejects
    ``inferenceConfig``, so the gateway forwards only messages, system blocks and
    the tool configuration.

    Ref: https://developers.openai.com/api/reference/resources/responses/subresources/input_tokens
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html
         stdapi/models/chat/_adapters/_openai_responses.py:count_input_tokens_via_bedrock
    """

    def test_input_tokens_basic(
        self, openai_client: OpenAI, responses_input_tokens_model: str
    ) -> None:
        """A string input is counted and reported as a ``response.input_tokens`` object.

        Ref: https://developers.openai.com/api/docs/guides/token-counting
             stdapi/routes/openai_responses.py:count_input_tokens
        """
        response = openai_client.responses.input_tokens.count(
            model=responses_input_tokens_model, input="Hello, how are you?"
        )

        assert response.input_tokens > 0
        assert response.object == "response.input_tokens"

    def test_input_tokens_with_instructions(
        self, openai_client: OpenAI, responses_input_tokens_model: str
    ) -> None:
        """``instructions`` are counted: they become Bedrock ``system`` blocks.

        Ref: https://developers.openai.com/api/docs/guides/token-counting
             stdapi/models/chat/_adapters/_openai_responses.py:count_input_tokens_via_bedrock
        """
        response_without = openai_client.responses.input_tokens.count(
            model=responses_input_tokens_model, input="Hello"
        )

        response_with = openai_client.responses.input_tokens.count(
            model=responses_input_tokens_model,
            input="Hello",
            instructions="You are a very detailed and verbose assistant that always provides comprehensive answers.",
        )

        assert response_without.input_tokens > 0
        assert response_with.input_tokens > response_without.input_tokens

    def test_input_tokens_with_tools(
        self, openai_client: OpenAI, responses_input_tokens_model: str
    ) -> None:
        """Tool definitions are counted: they are forwarded as ``toolConfig``.

        The gateway reuses the Converse tool mapping for counting, so a declared
        function tool must raise the count exactly as it would when generating.

        Ref: https://developers.openai.com/api/docs/guides/token-counting
             stdapi/models/chat/_adapters/_openai_responses.py:count_input_tokens_via_bedrock
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

        assert response_without.input_tokens > 0
        assert response_with.input_tokens > response_without.input_tokens

    def test_input_tokens_multi_turn(
        self, openai_client: OpenAI, responses_input_tokens_model: str
    ) -> None:
        """A multi-turn message array is counted across all of its turns.

        Ref: https://developers.openai.com/api/docs/guides/token-counting
             stdapi/models/chat/_adapters/_openai_responses.py:map_input
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

        assert response_single.input_tokens > 0
        assert response_multi.input_tokens > response_single.input_tokens

    def test_input_tokens_longer_content_more_tokens(
        self, openai_client: OpenAI, responses_input_tokens_model: str
    ) -> None:
        """A longer input yields a strictly higher count than a short one.

        Ref: https://developers.openai.com/api/docs/guides/token-counting
             stdapi/routes/openai_responses.py:count_input_tokens
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

        assert response_short.input_tokens > 0
        assert response_long.input_tokens > response_short.input_tokens

    def test_input_tokens_invalid_model(self, openai_client: OpenAI) -> None:
        """An unknown model is a 400 on this route, not the 404 the create route uses.

        ``count_input_tokens`` resolves the model with an explicit
        ``error_status=400`` override, so the same ``UnsupportedModelError`` is
        surfaced as a bad request here.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
             stdapi/routes/openai_responses.py:count_input_tokens
        """
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.responses.input_tokens.count(
                model="nonexistent-model-xyz", input="Hello"
            )

        assert excinfo.value.status_code == 400
        envelope = _error_envelope(excinfo.value)
        assert envelope["type"] == "invalid_request_error"
        assert "does not exist" in envelope["message"]
        assert "nonexistent-model-xyz" in envelope["message"]

    def test_input_tokens_input_text_blocks(
        self, openai_client: OpenAI, responses_input_tokens_model: str
    ) -> None:
        """A message whose content is a list of ``input_text`` parts is countable.

        The counting route accepts the same input union as create, so structured
        content parts must be flattened into Bedrock text blocks rather than
        rejected.

        Ref: https://developers.openai.com/api/reference/resources/responses/subresources/input_tokens
             stdapi/models/chat/_adapters/_openai_responses.py:_convert_input_content
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

    Only Nova 2 exposes an autonomous code-execution system tool, so
    ``code_interpreter`` is supported there and on the official API but on no
    other Bedrock model.

    Ref: https://developers.openai.com/api/docs/guides/tools-code-interpreter
         https://docs.aws.amazon.com/nova/latest/nova2-userguide/using-tools.html
         stdapi/models/chat/_adapters/_openai_responses.py:_resolve_integrated_tool_name
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
        """``code_interpreter`` executes the code and returns its result in the text.

        Locally the tool maps to the ``nova_code_interpreter`` Bedrock system
        tool, which runs the code within a single Converse call; the gateway
        suppresses that invocation, so the arithmetic result is the only visible
        evidence it ran and no ``function_call`` may leak.

        Ref: https://developers.openai.com/api/docs/guides/tools-code-interpreter
             stdapi/models/chat/_adapters/_openai_responses.py:_extract_output_items
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
        assert "391" in resp.output_text, (
            f"Expected '391' in output; got: {resp.output_text!r}"
        )
        assert not [item for item in resp.output if item.type == "function_call"], (
            "the suppressed code-execution tool must not surface as a function_call"
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
        """Streaming ``code_interpreter`` yields text deltas and no tool-call events.

        The suppressed ``nova_code_interpreter`` invocation must not surface as
        ``response.function_call_arguments.*`` events; only the resulting text is
        streamed before the terminal event.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
             stdapi/models/chat/_adapters/_openai_responses.py:format_stream
        """
        effective_model = (
            responses_code_interpreter_model if use_official_api else chat_model
        )

        text_delta_count = 0
        code_interp_done_count = 0
        func_event_count = 0
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
            elif event.type.startswith("response.function_call_arguments."):
                func_event_count += 1
            elif event.type == "response.completed":
                completed = True

        assert text_delta_count >= 1, "Expected at least one output_text.delta event"
        assert func_event_count == 0, (
            f"function_call_arguments events must not leak: {func_event_count}"
        )
        assert completed, "Expected response.completed event"
        if use_official_api:
            assert code_interp_done_count >= 1, (
                "Expected at least one code_interpreter_call output_item.done from official API"
            )


class TestUsageLogging:
    """Bedrock token usage recorded during a request reaches the response body.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
         stdapi/usage.py:record_bedrock_usage
    """

    def test_response_usage_logged(
        self, test_client: TestClientType | None, responses_model: str, api_key: str
    ) -> None:
        """The raw ``/v1/responses`` body carries a consistent ``usage`` object.

        Exercised over HTTP rather than through the SDK so that the serialised
        field names and the arithmetic between them are asserted on the wire
        format, not on a re-parsed model.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/retrieve
             stdapi/usage.py:record_bedrock_usage
        """
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
        assert api_usage["input_tokens"] > 0, "Expected input_tokens > 0"
        assert api_usage["output_tokens"] > 0, "Expected output_tokens > 0"
        assert api_usage["total_tokens"] == (
            api_usage["input_tokens"] + api_usage["output_tokens"]
        )
        assert api_usage["input_tokens_details"]["cached_tokens"] >= 0
        assert api_usage["output_tokens_details"]["reasoning_tokens"] >= 0


class TestUsageAggregation:
    """Per-request usage isolation and in-request summing of recorded usage.

    Ref: stdapi/usage.py:record_bedrock_usage
         stdapi/usage.py:usage_log_entries
    """

    def test_multiple_requests_aggregate_usage(
        self, test_client: TestClientType | None, chat_legacy_model: str, api_key: str
    ) -> None:
        """Usage counters are per-request and never accumulate across requests.

        The usage store lives in a context variable initialised per request, so
        three identical prompts must each report the same input token count; a
        growing count would mean the previous request's usage leaked into the
        next one.

        Ref: stdapi/usage.py:init_usage
             stdapi/usage.py:record_bedrock_usage
        """
        if test_client is None:
            pytest.skip("Requires local test server")

        input_token_counts = []
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
            input_token_counts.append(api_usage["input_tokens"])

        assert len(set(input_token_counts)) == 1, (
            f"identical prompts must report identical input tokens: "
            f"{input_token_counts}"
        )

    @pytest.mark.local
    def test_record_usage_twice_sums_values(self) -> None:
        """Two recordings for one model collapse into a single summed usage entry.

        Usage is keyed by (service, model), so repeated Bedrock calls within one
        request — region failover, or a multi-call fan-out — must be reported as
        one aggregate line rather than several.

        Ref: stdapi/usage.py:record_bedrock_usage
             stdapi/usage.py:usage_log_entries
        """
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
        """Cache-write tokens are reported per TTL bucket in the usage entry.

        Bedrock bills cache writes differently per TTL, so the buckets are kept
        separate instead of being summed into a single counter.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
             stdapi/usage.py:record_bedrock_usage
        """
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
        assert cache_tokens == {"5m": 500, "1h": 200}
        assert entry["input_tokens"] == 1000
        assert entry["output_tokens"] == 100


class TestUsageEMF:
    """CloudWatch Embedded Metric Format emission of the recorded usage.

    Ref: stdapi/usage.py:emit_usage_metrics
    """

    def test_emf_metrics_emitted_when_enabled(
        self,
        test_client: TestClientType | None,
        responses_model: str,
        api_key: str,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A billed request prints a self-consistent EMF document on stdout.

        Every dimension the EMF header declares must have a matching top-level
        field, otherwise CloudWatch silently drops the metric; the token metrics
        must also carry the same non-zero values as the response usage.

        Ref: stdapi/usage.py:emit_usage_metrics
        """
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
        """No EMF document is printed when ``cloudwatch_metrics`` is disabled.

        Recorded usage is still tracked for the request log; only the metric
        emission is gated on the setting.

        Ref: stdapi/config.py:_Settings
             stdapi/usage.py:emit_usage_metrics
        """
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
    """Reporting of deprecated settings that were explicitly set.

    Ref: stdapi/config.py:_Settings
    """

    def test_tokens_estimation_deprecated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deprecated field that was explicitly set is reported by ``deprecated()``.

        The check is driven by Pydantic's ``__pydantic_fields_set__``, so only
        settings the operator provided are reported — never a default.

        Ref: stdapi/config.py:_Settings
        """
        monkeypatch.setattr(SETTINGS, "__pydantic_fields_set__", {"tokens_estimation"})

        assert SETTINGS.deprecated() == {"tokens_estimation"}

    def test_no_deprecated_setting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nothing is reported when no setting was explicitly provided.

        Ref: stdapi/config.py:_Settings
        """
        monkeypatch.setattr(SETTINGS, "__pydantic_fields_set__", set())

        assert SETTINGS.deprecated() == set()
