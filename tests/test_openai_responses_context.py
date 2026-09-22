"""Context-window management on POST /v1/responses: overflow, truncation, compaction.

These tests encode the upstream contract and run unchanged against the official
API (``--use-official-api``) and the gateway:

- an input larger than the model's context window is refused with
  ``context_length_exceeded`` (``param: input``), in-stream when streaming;
- ``truncation: "auto"`` drops the oldest items instead, and reports only the
  tokens that were kept;
- ``context_management`` compacts the conversation once it crosses
  ``compact_threshold``: the compaction item leads ``output`` and supersedes
  every earlier item when it is sent back;
- a trailing ``compaction_trigger`` item compacts on demand.

Overflow tests use each lane's cheapest Responses model with the smallest
context window, so the rejected upload is small and a truncated answer cheap.

Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
     https://developers.openai.com/api/docs/guides/compaction
     https://developers.openai.com/api/reference/resources/responses/streaming-events
"""

from json import loads
from typing import TYPE_CHECKING, Any

import pytest
from openai import BadRequestError

from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from openai import OpenAI

#: Bedrock Mantle model serving the Responses API natively, compaction included.
_NATIVE_MODEL = "openai.gpt-5.6-luna"

#: Bedrock Mantle model serving the Responses API natively, without compaction.
_NATIVE_MODEL_WITHOUT_COMPACTION = "google.gemma-4-e2b"

#: Bedrock Mantle model answering the Responses API through a conversion.
_CONVERTED_MODEL = "google.gemma-3-4b-it"


#: Filler vocabulary: every tokenizer measured reads each word as about one token.
_WORDS = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
]

#: Per lane (official API or not): the cheapest Responses model with the smallest window, and the input tokens it takes.
_OVERFLOW_MODELS: dict[bool, tuple[str, int]] = {
    # 128k output tokens are reserved from gpt-5-nano's 400k window.
    True: ("gpt-5-nano", 272_000),
    False: ("meta.llama3-8b-instruct-v1:0", 8_192),
}

#: The OpenAI API's message refusing a Responses input over the context window.
_OVERFLOW_MESSAGE = (
    "Your input exceeds the context window of this model. Please adjust your input "
    "and try again."
)

#: Token threshold the compaction tests configure, well under their filler.
_COMPACT_THRESHOLD = 3000

#: Words of assistant-held filler that put the compaction input above the threshold.
_COMPACT_FILLER_WORDS = 6000


def _filler(words: int, tag: str) -> str:
    """Return text of about *words* tokens, starting with a readable tag.

    Args:
        words: Number of filler words.
        tag: Label placed at the start of the text.

    Returns:
        The filler text.
    """
    return f"[{tag}] " + " ".join(_WORDS[i % len(_WORDS)] for i in range(words))


def _overflow_input(window: int) -> Any:  # noqa: ANN401 - an SDK input value
    """Build a three-turn conversation about 1.25 times larger than *window*.

    Every turn but the last holds a third of the filler, so dropping the two
    oldest turns makes the rest fit.

    Args:
        window: Input tokens the model's context window takes.

    Returns:
        The input items.
    """
    chunk = window * 5 // 12
    items: list[dict[str, Any]] = []
    for index in range(3):
        items.append({"role": "user", "content": _filler(chunk, f"chunk {index}")})
        items.append({"role": "assistant", "content": f"Noted chunk {index}."})
    items.append({"role": "user", "content": "Reply with the single word OK."})
    return items


def _compaction_input() -> Any:  # noqa: ANN401 - an SDK input value
    """Build a conversation above the compaction threshold, with a fact to recall.

    The filler sits in an assistant item: upstream keeps user messages across a
    compaction, so user-held filler would stay above the threshold and compact
    again and again until the output budget runs out.

    Returns:
        The input items.
    """
    return [
        {
            "role": "user",
            "content": "Remember: the code word is TEAL. Then write a long text.",
        },
        {"role": "assistant", "content": _filler(_COMPACT_FILLER_WORDS, "long text")},
        {"role": "user", "content": "What is the code word? Answer in one word."},
    ]


def _compaction(threshold: int = _COMPACT_THRESHOLD) -> Any:  # noqa: ANN401 - an SDK parameter value
    """Return a ``context_management`` value compacting above *threshold* tokens.

    Args:
        threshold: The ``compact_threshold`` to configure.

    Returns:
        The parameter value.
    """
    return [{"type": "compaction", "compact_threshold": threshold}]


def _stream_events(client: OpenAI, **params: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """Stream a response and return every event payload, read off the raw frames.

    The raw body is read rather than the SDK's parsed events, so the wire
    sequence is asserted and an ``error`` event cannot be turned into an
    exception on one lane and an event on the other.

    Args:
        client: OpenAI client bound to the target.
        **params: Response creation parameters.

    Returns:
        The decoded ``data`` payloads, in wire order.
    """
    with client.responses.with_streaming_response.create(
        stream=True, **params
    ) as response:
        assert response.http_response.status_code == 200
        return [
            loads(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]


def _error_event_fields(
    event: dict[str, Any], *, use_official_api: bool
) -> dict[str, Any]:
    """Return the ``code``/``param``/``message`` of a streamed ``error`` event.

    The official API nests them under ``error``; the gateway sends them flat,
    the shape the SDK's ``ResponseErrorEvent`` types.

    Args:
        event: The ``error`` event payload.
        use_official_api: Whether the official API sent it.

    Returns:
        The mapping carrying the error fields.
    """
    return event["error"] if use_official_api else event


def _error_envelope(error: BadRequestError) -> dict[str, Any]:
    """Return the error envelope of a client exception (the SDK already unwrapped it).

    Args:
        error: Exception raised by the OpenAI client.

    Returns:
        The error envelope.
    """
    assert isinstance(error.body, dict)
    return error.body


@pytest.fixture(scope="module")
def overflow_model(use_official_api: bool) -> tuple[str, int]:
    """Return the lane's smallest-window Responses model and the input it takes."""
    return _OVERFLOW_MODELS[use_official_api]


class TestTruncation:
    """``truncation``: refuse an overflowing input, or drop its oldest items.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
    """

    def test_auto_on_an_input_that_fits_is_answered_and_echoed(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``truncation: "auto"`` is accepted, answered, and reported back.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        response = openai_client.responses.create(
            model=responses_model, input="Reply with OK.", truncation="auto"
        )
        assert response.status == "completed"
        assert response.truncation == "auto"
        assert response.output_text

    @pytest.mark.slow
    def test_overflow_is_context_length_exceeded(
        self, openai_client: OpenAI, overflow_model: tuple[str, int]
    ) -> None:
        """An input over the context window is a 400 ``context_length_exceeded``.

        Rejected before generation, so the call costs nothing; ``slow`` for the
        upload.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             https://developers.openai.com/api/docs/guides/error-codes
        """
        model, window = overflow_model
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.responses.create(
                model=model, input=_overflow_input(window), max_output_tokens=16
            )
        envelope = _error_envelope(excinfo.value)
        assert envelope == {
            "type": "invalid_request_error",
            "code": "context_length_exceeded",
            "param": "input",
            "message": _OVERFLOW_MESSAGE,
        }

    @pytest.mark.slow
    def test_streamed_overflow_ends_in_a_failed_response(
        self,
        openai_client: OpenAI,
        overflow_model: tuple[str, int],
        use_official_api: bool,
    ) -> None:
        """A streamed overflow is an ``error`` event, then ``response.failed``.

        The stream opens (200) and reports the refusal in-stream with the same
        error as the non-streamed 400, never as a retryable server error. The
        failed response has no output and no usage (upstream: ``usage: null``).

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
        """
        model, window = overflow_model
        events = _stream_events(
            openai_client,
            model=model,
            input=_overflow_input(window),
            max_output_tokens=16,
        )
        types = [event["type"] for event in events]
        assert types[0] == "response.created"
        assert types[-1] == "response.failed"
        error = next(event for event in events if event["type"] == "error")
        fields = _error_event_fields(error, use_official_api=use_official_api)
        assert fields["code"] == "context_length_exceeded"
        assert fields["param"] == "input"
        assert types.index("error") < types.index("response.failed")
        failed = events[-1]["response"]
        assert failed["status"] == "failed"
        assert failed["error"] == {
            "code": "context_length_exceeded",
            "message": _OVERFLOW_MESSAGE,
        }
        assert failed["output"] == []
        assert failed.get("usage") is None

    @pytest.mark.slow
    def test_auto_drops_the_oldest_items_of_an_overflowing_input(
        self, openai_client: OpenAI, overflow_model: tuple[str, int]
    ) -> None:
        """``truncation: "auto"`` answers an overflowing input from what still fits.

        The reported usage is the input that was kept, which is below the
        window the full input exceeded. About $0.01 on either lane.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        model, window = overflow_model
        response = openai_client.responses.create(
            model=model,
            input=_overflow_input(window),
            truncation="auto",
            max_output_tokens=16,
        )
        assert response.status in {"completed", "incomplete"}, response.error
        assert response.truncation == "auto"
        assert response.usage is not None
        assert 0 < response.usage.input_tokens < window

    @pytest.mark.slow
    def test_streamed_auto_drops_the_oldest_items(
        self, openai_client: OpenAI, overflow_model: tuple[str, int]
    ) -> None:
        """A streamed ``truncation: "auto"`` request completes on the kept input.

        On a backend that refuses the input only once the stream has opened,
        the refusal still has to be caught before the response starts.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
        """
        model, window = overflow_model
        events = _stream_events(
            openai_client,
            model=model,
            input=_overflow_input(window),
            truncation="auto",
            max_output_tokens=16,
        )
        terminal = events[-1]
        assert terminal["type"] in {"response.completed", "response.incomplete"}
        assert not [event for event in events if event["type"] == "error"]
        assert terminal["response"]["truncation"] == "auto"
        assert 0 < terminal["response"]["usage"]["input_tokens"] < window

    @pytest.mark.slow
    def test_input_token_count_honours_auto(
        self, openai_client: OpenAI, responses_input_tokens_model: str
    ) -> None:
        """Counting with ``truncation: "auto"`` counts what a response would keep.

        The input is larger than any counting model's window (200k tokens for
        the local counter, 128k upstream), so the count can only come back
        below it if the oldest items were dropped. Counting is free.

        Ref: https://developers.openai.com/api/reference/resources/responses/subresources/input_tokens/methods/count
        """
        items: Any = [
            item
            for index in range(4)
            for item in (
                {"role": "user", "content": _filler(60_000, f"chunk {index}")},
                {"role": "assistant", "content": "Noted."},
            )
        ]
        items.append({"role": "user", "content": "Reply OK."})
        counted = openai_client.responses.input_tokens.count(
            model=responses_input_tokens_model, input=items, truncation="auto"
        )
        assert 0 < counted.input_tokens < 200_000


class TestContextManagementValidation:
    """``context_management`` is validated with the upstream messages and codes.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
    """

    @pytest.mark.parametrize(
        ("value", "param", "code", "message"),
        [
            (
                [{"type": "foo"}],
                "context_management",
                None,
                "Unsupported context_management type: 'foo'.",
            ),
            (
                [{"type": "compaction", "compact_threshold": 0}],
                "context_management[0].compact_threshold",
                "integer_below_min_value",
                "Invalid 'context_management[0].compact_threshold': integer below minimum value. Expected a value >= 1000, but got 0 instead.",
            ),
            (
                [{"type": "compaction", "compact_threshold": 999}],
                "context_management[0].compact_threshold",
                "integer_below_min_value",
                "Invalid 'context_management[0].compact_threshold': integer below minimum value. Expected a value >= 1000, but got 999 instead.",
            ),
            (
                [],
                "context_management",
                "empty_array",
                "Invalid 'context_management': empty array. Expected an array with minimum length 1, but got an empty array instead.",
            ),
            (
                [{"type": "compaction", "compact_threshold": 1000.5}],
                "context_management[0].compact_threshold",
                "invalid_type",
                "Invalid type for 'context_management[0].compact_threshold': expected an integer, but got a decimal number instead.",
            ),
            (
                [{"type": "compaction", "compact_threshold": 1000.0}],
                "context_management[0].compact_threshold",
                "invalid_type",
                "Invalid type for 'context_management[0].compact_threshold': expected an integer, but got a decimal number instead.",
            ),
            (
                [{"type": "compaction", "compact_threshold": "1000"}],
                "context_management[0].compact_threshold",
                "invalid_type",
                "Invalid type for 'context_management[0].compact_threshold': expected an integer, but got a string instead.",
            ),
            (
                [{"type": "compaction", "compact_threshold": True}],
                "context_management[0].compact_threshold",
                "invalid_type",
                "Invalid type for 'context_management[0].compact_threshold': expected an integer, but got a boolean instead.",
            ),
            (
                [{"type": "compaction", "compact_threshold": [1000]}],
                "context_management[0].compact_threshold",
                "invalid_type",
                "Invalid type for 'context_management[0].compact_threshold': expected an integer, but got an array instead.",
            ),
            (
                {"type": "compaction", "compact_threshold": 1000},
                "context_management",
                "invalid_type",
                "Invalid type for 'context_management': expected an array of objects, but got an object instead.",
            ),
            (
                "compaction",
                "context_management",
                "invalid_type",
                "Invalid type for 'context_management': expected an array of objects, but got a string instead.",
            ),
            (
                ["compaction"],
                "context_management[0]",
                "invalid_type",
                "Invalid type for 'context_management[0]': expected an object, but got a string instead.",
            ),
            (
                [{"compact_threshold": 1000}],
                "context_management[0].type",
                "missing_required_parameter",
                "Missing required parameter: 'context_management[0].type'.",
            ),
            (
                [{"type": "compaction", "compact_threshold": 1000, "foo": 1}],
                "context_management[0].foo",
                "unknown_parameter",
                "Unknown parameter: 'context_management[0].foo'.",
            ),
        ],
        ids=[
            "unknown-type",
            "zero",
            "below-1000",
            "empty",
            "decimal",
            "integral-decimal",
            "string-threshold",
            "boolean-threshold",
            "array-threshold",
            "object",
            "string",
            "string-entry",
            "no-type",
            "unknown-key",
        ],
    )
    def test_invalid_entries_are_refused(
        self,
        openai_client: OpenAI,
        responses_model: str,
        value: Any,  # noqa: ANN401 - an SDK parameter value
        param: str,
        code: str | None,
        message: str,
    ) -> None:
        """Each malformed entry is a 400 naming the offending field, in upstream's words.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.responses.create(
                model=responses_model, input="Reply with OK.", context_management=value
            )
        envelope = _error_envelope(excinfo.value)
        assert envelope["type"] == "invalid_request_error"
        assert envelope["param"] == param
        assert envelope["code"] == code
        assert envelope["message"] == message

    def test_the_minimum_threshold_is_accepted(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """``compact_threshold: 1000`` is the smallest accepted value.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        response = openai_client.responses.create(
            model=responses_model,
            input="Reply with OK.",
            context_management=_compaction(1000),
        )
        assert response.status == "completed"
        assert [
            item.type for item in response.output if item.type == "compaction"
        ] == []


class TestCompaction:
    """``context_management`` compacts the conversation above its threshold.

    Ref: https://developers.openai.com/api/docs/guides/compaction
    """

    def test_compaction_leads_the_output_above_the_threshold(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Above the threshold, the compaction item comes first, then the answer.

        ``max_output_tokens`` is left unset: upstream draws the compaction pass
        from it, and a small budget ends the response after that item alone.

        Ref: https://developers.openai.com/api/docs/guides/compaction
        """
        response = openai_client.responses.create(
            model=responses_model,
            input=_compaction_input(),
            context_management=_compaction(),
        )
        assert response.status == "completed", response.error
        types = [item.type for item in response.output]
        assert types[0] == "compaction"
        assert "message" in types[1:]
        compaction = response.output[0].model_dump()
        assert compaction["encrypted_content"]
        assert compaction["id"].startswith("cmp")
        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.input_tokens + response.usage.output_tokens
        )

    def test_a_tool_loop_is_compacted(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """A single task run through tool calls compacts too, and is answered.

        The filler sits in a tool output, the shape an agent's context grows in.

        Ref: https://developers.openai.com/api/docs/guides/compaction
             https://developers.openai.com/api/docs/guides/function-calling
        """
        tools: Any = [
            {
                "type": "function",
                "name": "lookup",
                "description": "Look a record up by its identifier.",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
            }
        ]
        items: Any = [
            {
                "role": "user",
                "content": "Look up records 1 and 2, then give the code word of "
                "record 1 in one word.",
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "lookup",
                "arguments": '{"id": "1"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Record 1: the code word is TEAL. "
                + _filler(_COMPACT_FILLER_WORDS, "notes"),
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "lookup",
                "arguments": '{"id": "2"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": "Record 2: nothing notable.",
            },
        ]
        response = openai_client.responses.create(
            model=responses_model,
            input=items,
            tools=tools,
            context_management=_compaction(),
        )
        assert response.status == "completed", response.error
        assert response.output[0].type == "compaction"
        assert len(response.output) > 1

    def test_below_the_threshold_nothing_is_compacted(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Below the threshold, the response carries no compaction item.

        Ref: https://developers.openai.com/api/docs/guides/compaction
        """
        response = openai_client.responses.create(
            model=responses_model,
            input=_compaction_input(),
            context_management=_compaction(100_000),
            max_output_tokens=256,
        )
        assert "compaction" not in [item.type for item in response.output]

    def test_streamed_compaction_is_the_first_output_item(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Streamed, the compaction item is emitted at index 0 before the answer.

        Every later item is shifted after it, and the terminal snapshot carries
        it first with the usage of the whole response.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
             https://developers.openai.com/api/docs/guides/compaction
        """
        events = _stream_events(
            openai_client,
            model=responses_model,
            input=_compaction_input(),
            context_management=_compaction(),
        )
        assert [event["type"] for event in events[:2]] == [
            "response.created",
            "response.in_progress",
        ]
        added = [e for e in events if e["type"] == "response.output_item.added"]
        assert added[0]["item"]["type"] == "compaction"
        assert added[0]["output_index"] == 0
        done = next(
            e
            for e in events
            if e["type"] == "response.output_item.done"
            and e["item"]["type"] == "compaction"
        )
        assert done["output_index"] == 0
        assert done["item"]["encrypted_content"]
        assert all(event["output_index"] >= 1 for event in added[1:])
        sequence = [event["sequence_number"] for event in events]
        assert (
            sequence
            == sorted(sequence)
            == list(range(sequence[0], sequence[0] + len(sequence)))
        )
        terminal = events[-1]
        assert terminal["type"] == "response.completed"
        output = terminal["response"]["output"]
        assert output[0]["type"] == "compaction"
        assert "message" in [item["type"] for item in output[1:]]
        assert terminal["response"]["usage"]["input_tokens"] > 0

    def test_a_stored_compaction_bounds_the_next_turn(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """A turn continuing a compacted response starts from the compaction.

        Upstream measured 10,018 input tokens on the compacting turn and 793 on
        the next: everything before the compaction item is gone for good.

        Ref: https://developers.openai.com/api/docs/guides/compaction
             https://developers.openai.com/api/docs/guides/conversation-state
        """
        first = openai_client.responses.create(
            model=responses_model,
            input=_compaction_input(),
            context_management=_compaction(),
            store=True,
        )
        assert first.output[0].type == "compaction"
        second = openai_client.responses.create(
            model=responses_model,
            previous_response_id=first.id,
            input="Repeat the code word once more, in one word.",
            store=True,
            max_output_tokens=256,
        )
        assert first.usage is not None
        assert second.usage is not None
        assert second.usage.input_tokens * 3 < first.usage.input_tokens

    def test_a_compaction_item_supersedes_every_earlier_item(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Items sent before a compaction item a response produced are not read.

        Replaying the whole history ahead of that item costs what replaying
        from the item costs (upstream: 1,608 input tokens either way).

        Ref: https://developers.openai.com/api/docs/guides/compaction
        """
        history: Any = [
            *_compaction_input()[:2],
            {"role": "user", "content": "Reply with the single word Noted."},
        ]
        first = openai_client.responses.create(
            model=responses_model, input=history, context_management=_compaction()
        )
        output: Any = [item.model_dump(exclude_none=True) for item in first.output]
        assert output[0]["type"] == "compaction"
        question: Any = {"role": "user", "content": "What is the code word? One word."}
        full = openai_client.responses.create(
            model=responses_model,
            input=[*history, *output, question],
            max_output_tokens=256,
        )
        pruned = openai_client.responses.create(
            model=responses_model, input=[*output, question], max_output_tokens=256
        )
        assert full.usage is not None
        assert pruned.usage is not None
        assert abs(full.usage.input_tokens - pruned.usage.input_tokens) <= max(
            50, pruned.usage.input_tokens // 10
        )

    def test_a_compact_endpoint_item_does_not_supersede_earlier_items(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """An item from the compact endpoint adds to the history instead.

        Upstream: 11,661 input tokens with the original history ahead of it,
        1,876 without.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/compact
             https://developers.openai.com/api/docs/guides/compaction
        """
        history: Any = _compaction_input()[:2]
        compacted = openai_client.responses.compact(
            model=responses_model, input=history
        )
        item = next(part for part in compacted.output if part.type == "compaction")
        replay: Any = {
            "type": "compaction",
            "id": item.id,
            "encrypted_content": item.encrypted_content,
        }
        question: Any = {"role": "user", "content": "What is the code word? One word."}
        full = openai_client.responses.create(
            model=responses_model,
            input=[*history, replay, question],
            max_output_tokens=256,
        )
        alone = openai_client.responses.create(
            model=responses_model, input=[replay, question], max_output_tokens=256
        )
        assert full.usage is not None
        assert alone.usage is not None
        assert full.usage.input_tokens > alone.usage.input_tokens + 3000

    @pytest.mark.parametrize(
        ("item_id", "subject"),
        [("cmp_123", "for item cmp_123"), (None, "garb...bage")],
        ids=["with-id", "without-id"],
    )
    def test_a_compaction_item_it_cannot_read_is_refused(
        self,
        openai_client: OpenAI,
        responses_model: str,
        item_id: str | None,
        subject: str,
    ) -> None:
        """Compaction content the API did not produce is ``invalid_encrypted_content``.

        Refused before generation, so it costs nothing.

        Ref: https://developers.openai.com/api/docs/guides/compaction
             https://developers.openai.com/api/docs/guides/error-codes
        """
        item: dict[str, Any] = {"type": "compaction", "encrypted_content": "garbage"}
        if item_id:
            item["id"] = item_id
        items: Any = [item, {"role": "user", "content": "Reply with OK."}]
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.responses.create(
                model=responses_model, input=items, max_output_tokens=16
            )
        assert _error_envelope(excinfo.value) == {
            "type": "invalid_request_error",
            "code": "invalid_encrypted_content",
            "param": None,
            "message": f"The encrypted content {subject} could not be verified. "
            "Reason: Encrypted content could not be decrypted or parsed.",
        }


class TestCompactionTrigger:
    """A trailing ``compaction_trigger`` input item compacts on demand.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         https://developers.openai.com/api/docs/guides/compaction
    """

    @staticmethod
    def _input() -> Any:  # noqa: ANN401 - an SDK input value
        return [
            {"role": "user", "content": "Remember: the code word is TEAL."},
            {"role": "assistant", "content": _filler(800, "long text")},
            {"type": "compaction_trigger"},
        ]

    def test_trigger_answers_with_compaction_items_only(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """The response holds compaction items and no answer.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        response = openai_client.responses.create(
            model=responses_model, input=self._input()
        )
        assert response.status == "completed", response.error
        types = [item.type for item in response.output]
        assert types
        assert set(types) == {"compaction"}
        assert response.usage is not None
        assert response.usage.input_tokens > 0

    def test_streamed_trigger_reports_the_compacting_step(
        self, openai_client: OpenAI, responses_model: str
    ) -> None:
        """Streamed, ``response.compaction.compacting`` sits inside the item's lifecycle.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
        """
        events = _stream_events(
            openai_client, model=responses_model, input=self._input()
        )
        types = [event["type"] for event in events]
        added = types.index("response.output_item.added")
        compacting = types.index("response.compaction.compacting")
        done = types.index("response.output_item.done")
        assert added < compacting < done
        assert events[compacting]["output_index"] == events[added]["output_index"]
        assert events[added]["item"]["type"] == "compaction"
        assert types[-1] == "response.completed"
        assert {item["type"] for item in events[-1]["response"]["output"]} == {
            "compaction"
        }

    @pytest.mark.parametrize(
        ("change", "param", "message"),
        [
            (
                {"max_output_tokens": 1000},
                "max_output_tokens",
                "requires 'max_output_tokens' to be at least 20000",
            ),
            ({"reorder": True}, "input", "must be the final input item"),
            (
                {"repeat": True},
                "input",
                "Only one 'compaction_trigger' item may be provided.",
            ),
        ],
        ids=["small-output-budget", "not-final", "two-triggers"],
    )
    def test_trigger_misuse_is_refused(
        self,
        openai_client: OpenAI,
        responses_model: str,
        change: dict[str, Any],
        param: str,
        message: str,
    ) -> None:
        """A trigger with a small output budget, not last, or repeated is a 400.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        items = self._input()
        params: dict[str, Any] = {}
        if change.pop("reorder", False):
            items = [items[-1], *items[:-1]]
        if change.pop("repeat", False):
            items = [*items, items[-1]]
        params.update(change)
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.responses.create(model=responses_model, input=items, **params)
        envelope = _error_envelope(excinfo.value)
        assert envelope["type"] == "invalid_request_error"
        assert envelope["param"] == param
        assert message in envelope["message"]


@pytest.mark.gateway("gateway billing and Bedrock Mantle native models")
class TestGatewayContext:
    """What only the gateway exposes: the billing of a compaction, and Mantle.

    Ref: https://developers.openai.com/api/docs/guides/compaction
         stdapi/routes/_responses_context.py:generate
    """

    @pytest.mark.usefixtures("local_test_client")
    def test_both_calls_of_a_compacting_response_are_billed(
        self,
        openai_client: OpenAI,
        responses_model: str,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """The summary and the answer are both recorded, as the response reports.

        Ref: stdapi/routes/_responses_context.py:summarize
        """
        capfd.readouterr()
        response = openai_client.responses.create(
            model=responses_model,
            input=_compaction_input(),
            context_management=_compaction(),
        )
        assert response.output[0].type == "compaction"
        assert response.usage is not None
        entries = logged_usage_entries(capfd.readouterr().out, model=responses_model)
        assert sum(entry["input_tokens"] for entry in entries) == (
            response.usage.input_tokens
        )
        assert sum(entry["output_tokens"] for entry in entries) == (
            response.usage.output_tokens
        )

    def test_a_converted_model_compacts_and_answers(
        self, openai_client: OpenAI
    ) -> None:
        """A Mantle model without the Responses API is compacted by the gateway.

        Bedrock Mantle refuses this server's compaction items, so the answer must
        reach it expanded.

        Ref: https://developers.openai.com/api/docs/guides/compaction
             stdapi/routes/_responses_context.py:_Call.open
        """
        response = openai_client.responses.create(
            model=_CONVERTED_MODEL,
            input=_compaction_input(),
            context_management=_compaction(),
            max_output_tokens=64,
        )
        assert response.status in {"completed", "incomplete"}, response.error
        types = [item.type for item in response.output]
        assert types[0] == "compaction"
        assert "message" in types[1:]

    def test_a_native_model_honours_both_parameters(
        self, openai_client: OpenAI
    ) -> None:
        """A model serving the Responses API natively takes both parameters as sent.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
        """
        response = openai_client.responses.create(
            model=_NATIVE_MODEL,
            input="Reply with OK.",
            truncation="auto",
            context_management=_compaction(1000),
            max_output_tokens=500,
        )
        assert response.status in {"completed", "incomplete"}
        assert response.truncation == "auto"

    def test_a_native_model_refusing_compaction_says_so(
        self, openai_client: OpenAI
    ) -> None:
        """A native model without compaction refuses it rather than ignoring it.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
        """
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.responses.create(
                model=_NATIVE_MODEL_WITHOUT_COMPACTION,
                input="Reply with OK.",
                context_management=_compaction(1000),
                max_output_tokens=16,
            )
        assert "compaction" in _error_envelope(excinfo.value)["message"]
