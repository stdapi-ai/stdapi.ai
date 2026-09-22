"""Unit tests of the Responses context-window helpers and their stream rendering.

Overflow recognition runs over the refusals Bedrock was recorded returning on
2026-09-22 for each model family; the rest runs on built items, with no call.

Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
     https://developers.openai.com/api/docs/guides/compaction
     https://developers.openai.com/api/reference/resources/responses/streaming-events
     stdapi/models/chat/_adapters/_responses_context.py
     stdapi/models/chat/_adapters/_openai_responses.py:expand_compaction_items
"""

from base64 import urlsafe_b64encode
from contextlib import suppress
from json import dumps, loads
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import EventStreamError
from sse_starlette import ServerSentEvent

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock_mantle import MantleError
from stdapi.models.chat._adapters._openai_responses import (
    COMPACTION_CONTENT_PREFIX,
    encode_compaction_content,
    encode_compaction_state,
    expand_compaction_items,
    format_compaction_stream,
    format_failed_stream,
    format_stream,
    join_summaries,
    map_input,
    with_user_text,
)
from stdapi.models.chat._adapters._responses_context import (
    ContextLengthExceededError,
    ContextOverflow,
    collect_stream_open_errors,
    context_overflow,
    estimate_tokens,
    record_stream_open_error,
    split_everything,
    split_for_compaction,
    truncate_input,
)
from stdapi.monitoring import SseHandledStreamError
from stdapi.types.openai_responses import (
    CompactionItemParam,
    EasyInputMessage,
    FunctionCallInput,
    FunctionCallOutput,
    InputMessage,
    ResponseCreateParams,
    ResponseInputImage,
    ResponseInputItem,
    ResponseInputText,
    ResponseReasoningItemInput,
    ResponseUsage,
)
from tests._helpers import make_client_error

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator

    from sse_starlette import EventSourceResponse

pytestmark = pytest.mark.usefixtures("request_log")

#: Refusals recorded from Converse on 2026-09-22, with the sizes each states.
_RECORDED_OVERFLOWS: list[tuple[str, str, int | None, int | None]] = [
    (
        "nova-micro",
        (
            "The model returned the following errors: Input Tokens Exceeded: Number "
            "of input tokens exceeds maximum length. Please update the input to try "
            "again."
        ),
        None,
        None,
    ),
    (
        "claude-haiku-4.5",
        (
            "The model returned the following errors: prompt is too long: 286028 "
            "tokens > 200000 maximum"
        ),
        286028,
        200000,
    ),
    (
        "gpt-oss-20b",
        (
            'The model returned the following errors: {"error":{"code":'
            '"validation_error","message":"ErrorEvent { error: APIError { type: '
            '\\"BadRequestError\\", code: Some(400), message: \\"Input length '
            '(195071) exceeds model\'s maximum context length (131072).\\" } }"}}'
        ),
        195071,
        131072,
    ),
    (
        "gpt-oss-20b-stream",
        (
            "Mantle streaming error for requestId 9293f38b-0000: ErrorEvent { error: "
            "APIError { message: \"Input length (195071) exceeds model's maximum "
            'context length (131072)." } }'
        ),
        195071,
        131072,
    ),
    (
        "gemma-3-4b",
        (
            "The model returned the following errors: This model's maximum context "
            "length is 131072 tokens. However, you requested 5 output tokens and your "
            "prompt contains at least 242008 input tokens, for a total of at least "
            "242013 tokens."
        ),
        242008,
        131072,
    ),
    (
        "llama-3-8b",
        (
            "The model returned the following errors: This model's maximum context "
            "length is 8192 tokens. Please reduce the length of the prompt"
        ),
        None,
        None,
    ),
]


def _user(text: str) -> EasyInputMessage:
    return EasyInputMessage(role="user", content=text)


def _assistant(text: str) -> EasyInputMessage:
    return EasyInputMessage(role="assistant", content=text)


def _call(call_id: str) -> FunctionCallInput:
    return FunctionCallInput(
        type="function_call", call_id=call_id, name="lookup", arguments="{}"
    )


def _output(call_id: str, text: str = "result") -> FunctionCallOutput:
    return FunctionCallOutput(type="function_call_output", call_id=call_id, output=text)


def _texts(items: list[ResponseInputItem]) -> list[str]:
    texts: list[str] = []
    for item in items:
        content = getattr(item, "content", None)
        if isinstance(content, list):
            texts.append(" | ".join(str(getattr(part, "text", "")) for part in content))
        else:
            texts.append(str(content or getattr(item, "call_id", "")))
    return texts


class TestContextOverflow:
    """A backend's context-window refusal is recognized, whatever its wording.

    Ref: stdapi/models/chat/_adapters/_responses_context.py:context_overflow
    """

    @pytest.mark.parametrize(
        ("message", "observed", "limit"),
        [
            (message, observed, limit)
            for _, message, observed, limit in _RECORDED_OVERFLOWS
        ],
        ids=[family for family, *_ in _RECORDED_OVERFLOWS],
    )
    def test_recorded_refusals_are_overflows(
        self, message: str, observed: int | None, limit: int | None
    ) -> None:
        """Each family's refusal is an overflow, with the sizes it states.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        """
        error = make_client_error("ValidationException", "Converse", message=message)
        assert context_overflow(error) == ContextOverflow(observed, limit)

    def test_an_in_stream_refusal_is_an_overflow(self) -> None:
        """The first-event refusal of a stream, lowercase code, is recognized.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html
        """
        error = EventStreamError(
            {
                "Error": {
                    "Code": "validationException",
                    "Message": _RECORDED_OVERFLOWS[0][1],
                }
            },
            "ConverseStream",
        )
        assert context_overflow(error) == ContextOverflow()

    def test_a_relayed_refusal_without_a_code_is_read_by_its_wording(self) -> None:
        """A converted backend's 400 carrying only the wording is recognized, sized.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:context_overflow
        """
        error = MantleError(_RECORDED_OVERFLOWS[4][1], status=400)
        assert error.code is None
        assert context_overflow(error) == ContextOverflow(242008, 131072)

    def test_an_upstream_shaped_error_is_an_overflow(self) -> None:
        """A relayed OpenAI-shaped refusal is recognized by its code.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
        """
        error = MantleError("Your input exceeds the context window.", status=400)
        error.code = "context_length_exceeded"
        assert context_overflow(error) == ContextOverflow()
        assert context_overflow(ContextLengthExceededError()) == ContextOverflow()

    @pytest.mark.parametrize(
        "error",
        [
            make_client_error("ValidationException", message="Malformed input."),
            make_client_error("ThrottlingException", message="prompt is too long"),
            ApiError("prompt is too long", status=500),
            ValueError("prompt is too long"),
        ],
        ids=["other-validation", "throttle", "server-error", "not-an-api-error"],
    )
    def test_other_failures_are_not_overflows(self, error: Exception) -> None:
        """Only a validation refusal naming the window is an overflow.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:context_overflow
        """
        assert context_overflow(error) is None

    def test_the_error_is_upstreams(self) -> None:
        """The refusal reaches the client with upstream's code, param and message.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        error = ContextLengthExceededError()
        assert (error.status, error.code, error.param) == (
            400,
            "context_length_exceeded",
            "input",
        )
        assert "exceeds the context window" in str(error)

    def test_stream_open_errors_reach_the_collector_only(self) -> None:
        """An error is collected while a collector is open, and dropped after.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:collect_stream_open_errors
        """
        first, second = ValueError("a"), ValueError("b")
        with collect_stream_open_errors() as errors:
            record_stream_open_error(first)
        record_stream_open_error(second)
        assert errors == [first]


class TestTruncateInput:
    """``truncation: "auto"`` drops whole turns oldest first, then cuts text.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/models/chat/_adapters/_responses_context.py:truncate_input
    """

    def test_oldest_turns_go_first_and_tool_pairs_stay_together(self) -> None:
        """An unsized refusal keeps half the input, dropping whole turns.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:truncate_input
        """
        items: list[ResponseInputItem] = [
            _user("u1 " * 100),
            _call("c1"),
            _output("c1", "o1 " * 100),
            _user("u2 " * 100),
            _assistant("a2 " * 100),
            _user("u3 " * 100),
            _assistant("a3"),
            _user("question"),
        ]
        trimmed = truncate_input(items, ContextOverflow(), 0)
        assert trimmed is not None
        assert trimmed == items[5:]

    def test_system_and_developer_messages_are_kept(self) -> None:
        """A configuring message inside a dropped turn stays in place.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:truncate_input
        """
        developer = InputMessage(
            role="developer", content=[ResponseInputText(type="input_text", text="d")]
        )
        items: list[ResponseInputItem] = [
            developer,
            _user("u1 " * 300),
            _assistant("a1"),
            _user("question"),
        ]
        trimmed = truncate_input(items, ContextOverflow(), 0)
        assert trimmed == [developer, items[-1]]

    def test_sized_refusal_drops_only_what_is_needed(self) -> None:
        """A refusal stating its sizes drops turns only down to the target.

        10% over the window, aiming at 90% of it: the oldest of four equal
        turns goes, the others stay.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:truncate_input
        """
        items: list[ResponseInputItem] = [
            item
            for index in range(4)
            for item in (_user(f"u{index} " * 200), _assistant(f"a{index}"))
        ]
        trimmed = truncate_input(items, ContextOverflow(110, 100), 0)
        assert trimmed == items[2:]

    def test_the_latest_turn_is_not_cut_once_older_turns_went(self) -> None:
        """Dropping older turns leaves the latest one whole, however large.

        Whatever is still too large is left to the next refusal to size.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:truncate_input
        """
        items: list[ResponseInputItem] = [
            item
            for index in range(3)
            for item in (_user(f"u{index} " + "x" * 3000), _assistant(f"a{index}"))
        ]
        latest = _user("y" * 20_000 + "QUESTION")
        items.append(latest)
        assert truncate_input(items, ContextOverflow(), 0) == [latest]
        assert isinstance(latest.content, str)
        assert latest.content.endswith("QUESTION")

    def test_each_retry_aims_lower(self) -> None:
        """A later retry keeps a smaller share of the same refused input.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:truncate_input
        """
        items: list[ResponseInputItem] = [
            item
            for index in range(10)
            for item in (_user(f"u{index} " * 200), _assistant(f"a{index}"))
        ]
        overflow = ContextOverflow(110, 100)
        first = truncate_input(items, overflow, 0)
        later = truncate_input(items, overflow, 2)
        assert first is not None
        assert later is not None
        assert len(later) < len(first)

    def test_a_single_oversized_text_keeps_its_beginning(self) -> None:
        """With no turn left to drop, the largest text is cut at its end.

        Upstream kept the head of an item larger than the window and cut its tail.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        text = "HEAD " + "x" * 10_000 + " TAIL"
        trimmed = truncate_input(text, ContextOverflow(), 0)
        assert trimmed is not None
        (message,) = trimmed
        assert isinstance(message, EasyInputMessage)
        assert isinstance(message.content, str)
        assert message.content.startswith("HEAD ")
        assert "TAIL" not in message.content
        assert len(message.content) < len(text)

    def test_text_parts_are_cut_in_place(self) -> None:
        """A text part of a content list is shortened, its neighbours kept.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:truncate_input
        """
        image = ResponseInputImage(
            type="input_image", image_url="https://example.com/a.png"
        )
        item = InputMessage(
            role="user",
            content=[image, ResponseInputText(type="input_text", text="y" * 5000)],
        )
        trimmed = truncate_input([item], ContextOverflow(), 0)
        assert trimmed is not None
        (message,) = trimmed
        assert isinstance(message, InputMessage)
        assert message.content[0] == image
        assert isinstance(text := message.content[1], ResponseInputText)
        assert 0 < len(text.text) < 5000

    def test_inline_media_does_not_count_against_the_budget(self) -> None:
        """Old text turns are dropped even when the latest turn carries a large image.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:truncate_input
        """
        latest = InputMessage(
            role="user",
            content=[
                ResponseInputText(type="input_text", text="What is on this image?"),
                ResponseInputImage(
                    type="input_image",
                    image_url="data:image/png;base64," + "A" * 200_000,
                ),
            ],
        )
        items: list[ResponseInputItem] = [
            item
            for index in range(4)
            for item in (_user(f"u{index} " * 2000), _assistant(f"a{index}"))
        ]
        trimmed = truncate_input([*items, latest], ContextOverflow(286028, 200000), 0)
        assert trimmed == [*items[4:], latest]

    def test_a_turn_too_large_after_the_drops_is_left_to_the_next_refusal(self) -> None:
        """Once older turns are dropped, the latest turn is not also cut.

        Many small tool outputs cannot absorb a cut, and a question at the end
        of a long text would lose its question: the next refusal sizes the rest.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:truncate_input
        """
        latest: list[ResponseInputItem] = [
            _user("task"),
            *(
                item
                for index in range(20)
                for item in (_call(f"c{index}"), _output(f"c{index}", "o" * 1000))
            ),
        ]
        items: list[ResponseInputItem] = [
            _user("u0 " * 1700),
            _assistant("a0"),
            _user("u1 " * 1700),
            _assistant("a1"),
            *latest,
        ]
        assert truncate_input(items, ContextOverflow(), 0) == latest

    def test_a_reasoning_item_is_never_cut(self) -> None:
        """The largest text of a signed reasoning item stays whole.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:truncate_input
        """
        reasoning: Any = ResponseReasoningItemInput.model_validate(
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "r" * 10000}],
                "encrypted_content": "signed",
            }
        )
        question = _user("q" * 8000)
        trimmed = truncate_input([question, reasoning], ContextOverflow(110, 100), 0)
        assert trimmed is not None
        assert trimmed[1] == reasoning
        cut = trimmed[0]
        assert isinstance(cut, EasyInputMessage)
        assert isinstance(cut.content, str)
        assert len(cut.content) < 8000

    def test_an_input_without_text_to_cut_is_refused(self) -> None:
        """Media alone cannot be cut, so there is no retry to plan.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:truncate_input
        """
        item = InputMessage(
            role="user",
            content=[
                ResponseInputImage(
                    type="input_image", image_url="data:image/png;base64,AA"
                )
            ],
        )
        assert truncate_input([item], ContextOverflow(), 0) is None
        assert truncate_input(None, ContextOverflow(), 0) is None


class TestSplitForCompaction:
    """What a compaction summarizes, and what it keeps verbatim.

    Ref: https://developers.openai.com/api/docs/guides/compaction
         stdapi/models/chat/_adapters/_responses_context.py:split_for_compaction
    """

    def test_a_conversation_keeps_its_latest_turn(self) -> None:
        """Earlier turns are summarized; the latest turn and system prompts stay.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:split_for_compaction
        """
        system = EasyInputMessage(role="system", content="be brief")
        items: list[ResponseInputItem] = [
            system,
            _user("u1"),
            _assistant("a1"),
            _user("u2"),
        ]
        split = split_for_compaction(items)
        assert split is not None
        assert split.summarized == items[:3]
        assert split.before == [system]
        assert split.after == [items[3]]

    def test_a_tool_loop_keeps_its_task_and_latest_step(self) -> None:
        """A single task run through tool steps keeps the task and the latest step.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:split_for_compaction
        """
        task = _user("task")
        items: list[ResponseInputItem] = [
            task,
            _call("c1"),
            _output("c1"),
            _call("c2"),
            _call("c3"),
            _output("c2"),
            _output("c3"),
        ]
        split = split_for_compaction(items)
        assert split is not None
        assert split.summarized == items[:3]
        assert split.before == [task]
        assert split.after == items[3:]

    def test_nothing_before_the_latest_step_is_not_compacted(self) -> None:
        """A lone question, or one tool step, leaves nothing to summarize.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:split_for_compaction
        """
        assert split_for_compaction("hello") is None
        assert split_for_compaction([_user("u"), _call("c"), _output("c")]) is None
        assert split_for_compaction(None) is None

    def test_a_trigger_compacts_everything_but_system_prompts(self) -> None:
        """An explicit compaction keeps configuring messages only.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:split_everything
        """
        developer = EasyInputMessage(role="developer", content="rules")
        items: list[ResponseInputItem] = [developer, _user("u"), _assistant("a")]
        split = split_everything(items)
        assert split is not None
        assert split.before == [developer]
        assert split.after == []
        assert split_everything([developer]) is None


class TestEstimateTokens:
    """The fallback threshold measure counts text only, and biases low.

    Ref: stdapi/models/chat/_adapters/_responses_context.py:estimate_tokens
    """

    def test_text_is_counted_below_four_characters_a_token(self) -> None:
        """660 characters of text estimate under the 165 tokens four a token gives.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:estimate_tokens
        """
        estimate = estimate_tokens([_user("x" * 600)], "y" * 60)
        assert 0 < estimate < 165

    def test_media_is_not_counted(self) -> None:
        """An inline image weighs nothing in the estimate.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:estimate_tokens
        """
        image = InputMessage(
            role="user",
            content=[
                ResponseInputImage(
                    type="input_image",
                    image_url="data:image/png;base64," + "A" * 60_000,
                )
            ],
        )
        assert estimate_tokens([image]) == 0


class TestCompactionCodec:
    """Compaction items round-trip, and a response's item supersedes the history.

    Ref: https://developers.openai.com/api/docs/guides/compaction
         stdapi/models/chat/_adapters/_openai_responses.py:expand_compaction_items
    """

    @staticmethod
    def _item(content: str) -> CompactionItemParam:
        return CompactionItemParam(
            id="cmp_1", encrypted_content=content, type="compaction"
        )

    async def test_a_response_item_replaces_everything_before_it(self) -> None:
        """Items ahead of a response-produced item are dropped when it is expanded.

        Upstream counted the same input tokens with and without the replayed
        history ahead of such an item.

        Ref: https://developers.openai.com/api/docs/guides/compaction
        """
        item = self._item(
            await encode_compaction_state(
                "SUMMARY", [_user("kept before")], [_user("after")]
            )
        )
        expanded = await expand_compaction_items(
            [_user("old"), _assistant("old answer"), item, _user("next")]
        )
        assert _texts(expanded) == [
            "kept before",
            "Summary of the earlier conversation:\nSUMMARY",
            "after",
            "next",
        ]

    async def test_the_expansion_keeps_each_summary_apart(self) -> None:
        """The expansion compactions read never folds a summary into a kept message.

        Folding there made each compaction keep the previous summary inside the
        task it keeps verbatim, so summaries piled up in the kept items.

        Ref: stdapi/models/chat/_adapters/_openai_responses.py:expand_compaction_items
        """
        item = self._item(
            await encode_compaction_state("S", [_user("task")], [_call("c1")])
        )
        expanded = await expand_compaction_items([item, _output("c1")])
        assert _texts(expanded) == [
            "task",
            "Summary of the earlier conversation:\nS",
            "c1",
            "c1",
        ]

    async def test_a_summary_joins_the_user_message_beside_it_when_sent(self) -> None:
        """For a backend refusing two user messages in a row, a summary is folded.

        It joins the following user message, else the preceding one, else stays.

        Ref: stdapi/models/chat/_adapters/_openai_responses.py:join_summaries
        """
        item = self._item(
            await encode_compaction_state("S", [_user("task")], [_call("c1")])
        )
        joined = join_summaries(await expand_compaction_items([item, _output("c1")]))
        assert _texts(joined) == [
            "task | Summary of the earlier conversation:\nS",
            "c1",
            "c1",
        ]
        state = await encode_compaction_state("S", [], [_user("next")])
        following = join_summaries(await expand_compaction_items([self._item(state)]))
        assert _texts(following) == ["Summary of the earlier conversation:\nS | next"]
        alone = join_summaries(
            await expand_compaction_items(
                [self._item(await encode_compaction_state("S", [], [_assistant("a")]))]
            )
        )
        assert _texts(alone) == ["Summary of the earlier conversation:\nS", "a"]

    def test_a_trailing_text_joins_a_final_user_message(self) -> None:
        """The summarisation directive never makes a second user message in a row.

        Ref: stdapi/models/chat/_adapters/_openai_responses.py:with_user_text
        """
        assert _texts(with_user_text([_user("S")], "DO")) == ["S | DO"]
        assert _texts(with_user_text([_user("q"), _assistant("a")], "DO")) == [
            "q",
            "a",
            "DO",
        ]
        assert _texts(with_user_text([], "DO")) == ["DO"]

    async def test_a_compact_endpoint_item_adds_to_the_history(self) -> None:
        """An item from the compact endpoint keeps the items before it.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/compact
        """
        item = self._item(encode_compaction_content("SUMMARY"))
        expanded = await expand_compaction_items([_user("old"), item])
        assert _texts(expanded) == [
            "old",
            "Summary of the earlier conversation:\nSUMMARY",
        ]

    async def test_the_latest_response_item_wins(self) -> None:
        """Of two response-produced items, the later one stands for the history.

        Ref: https://developers.openai.com/api/docs/guides/compaction
        """
        first = self._item(await encode_compaction_state("ONE", [], []))
        second = self._item(await encode_compaction_state("TWO", [], [_user("tail")]))
        expanded = await expand_compaction_items([first, _user("x"), second])
        assert _texts(expanded) == ["Summary of the earlier conversation:\nTWO", "tail"]

    async def test_expanded_items_reach_the_model_as_messages(self) -> None:
        """The Converse mapping replays the expansion, merged into one user turn.

        Ref: stdapi/models/chat/_adapters/_openai_responses.py:map_input
        """
        item = self._item(await encode_compaction_state("S", [_user("task")], []))
        messages, _ = await map_input([_user("dropped"), item, _user("q")], None)
        assert [message["role"] for message in messages] == ["user"]
        texts = [block["text"] for block in messages[0]["content"] if "text" in block]
        assert "dropped" not in texts
        assert texts[0] == "task"
        assert texts[-1] == "q"

    @pytest.mark.parametrize(
        "content",
        [
            f"{COMPACTION_CONTENT_PREFIX[1]}!!!",
            COMPACTION_CONTENT_PREFIX[1]
            + urlsafe_b64encode(b'{"summary": 1, "before": [], "after": []}').decode(),
            COMPACTION_CONTENT_PREFIX[1]
            + urlsafe_b64encode(b'{"summary": "s"}').decode(),
        ],
        ids=["not-base64", "summary-not-text", "missing-items"],
    )
    async def test_malformed_state_is_refused(self, content: str) -> None:
        """A marked but malformed state is a 400, never a 500.

        Ref: stdapi/models/chat/_adapters/_openai_responses.py:expand_compaction_items
        """
        with pytest.raises(ApiError, match="could not be verified") as excinfo:
            await expand_compaction_items([self._item(content)])
        assert excinfo.value.status == 400
        assert excinfo.value.code == "invalid_encrypted_content"


async def _collect(events: AsyncIterator[Any]) -> list[dict[str, Any]]:
    """Drain an event stream into its payloads, ending at a handled error.

    Args:
        events: The stream.

    Returns:
        Every payload, in order.
    """
    payloads: list[dict[str, Any]] = []
    with suppress(SseHandledStreamError):
        async for event in events:
            payloads.append(loads(event.data))  # noqa: PERF401 - kept up to a failure
    return payloads


class TestStreamRendering:
    """Overflow and compaction render as upstream's event sequences.

    Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
    """

    @staticmethod
    def _request() -> ResponseCreateParams:
        return ResponseCreateParams(model="m", input="hello", stream=True)

    async def test_a_first_event_refusal_is_collected_and_reported(self) -> None:
        """A stream refused on its first event reports it, and tells its opener.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
             stdapi/models/chat/_adapters/_openai_responses.py:format_stream
        """
        refusal = EventStreamError(
            {
                "Error": {
                    "Code": "validationException",
                    "Message": _RECORDED_OVERFLOWS[5][1],
                }
            },
            "ConverseStream",
        )

        async def _refused() -> AsyncGenerator[Any]:
            raise refusal
            yield  # pragma: no cover - makes this an async generator

        with collect_stream_open_errors() as errors:
            events = await _collect(
                format_stream("resp-1", 0.0, "m", _refused(), self._request())
            )
        assert errors == [refusal]
        assert [event["type"] for event in events] == [
            "response.created",
            "response.in_progress",
            "error",
            "response.failed",
        ]
        assert events[2]["code"] == "context_length_exceeded"
        assert events[2]["param"] == "input"
        assert "8192" not in events[2]["message"], "the backend text stays internal"
        assert events[3]["response"]["error"]["code"] == "context_length_exceeded"

    async def test_a_refusal_before_the_stream_is_streamed_as_upstream_does(
        self,
    ) -> None:
        """A request refused before any event still streams its failure.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
             stdapi/models/chat/_adapters/_openai_responses.py:format_failed_stream
        """
        events = await _collect(
            format_failed_stream(
                "resp-1", 0.0, "m", self._request(), ContextLengthExceededError()
            )
        )
        assert [event["type"] for event in events] == [
            "response.created",
            "response.in_progress",
            "error",
            "response.failed",
        ]
        assert [event["sequence_number"] for event in events] == [0, 1, 2, 3]

    async def test_a_compaction_leads_the_answer_it_precedes(self) -> None:
        """The answer's events follow the compaction item, shifted and renumbered.

        Ref: https://developers.openai.com/api/docs/guides/compaction
             stdapi/models/chat/_adapters/_openai_responses.py:format_compaction_stream
        """
        usage = ResponseUsage.model_validate(
            {
                "input_tokens": 100,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 10,
                "output_tokens_details": {"reasoning_tokens": 3},
                "total_tokens": 110,
            }
        )
        answer_usage = {
            "input_tokens": 5,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 7,
        }
        answered: list[CompactionItemParam] = []

        async def _answer_events() -> AsyncGenerator[ServerSentEvent]:
            for payload in (
                {"type": "response.created", "sequence_number": 0},
                {"type": "response.in_progress", "sequence_number": 1},
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "sequence_number": 2,
                    "item": {"type": "message"},
                },
                {
                    "type": "response.completed",
                    "sequence_number": 3,
                    "response": {
                        "output": [{"type": "message"}],
                        "usage": answer_usage,
                    },
                },
            ):
                yield ServerSentEvent(data=dumps(payload), event=str(payload["type"]))

        async def _compact() -> tuple[str, ResponseUsage | None]:
            return "v2:state", usage

        async def _answer(item: CompactionItemParam) -> EventSourceResponse:
            from sse_starlette import EventSourceResponse  # noqa: PLC0415

            answered.append(item)
            return EventSourceResponse(_answer_events())

        events = await _collect(
            format_compaction_stream(
                "resp-1", 0.0, "m", self._request(), "cmp_1", _compact, _answer
            )
        )
        assert [event["type"] for event in events] == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.output_item.done",
            "response.output_item.added",
            "response.completed",
        ]
        assert [event["sequence_number"] for event in events] == list(range(6))
        assert events[3]["item"]["encrypted_content"] == "v2:state"
        assert events[4]["output_index"] == 1
        assert answered[0].encrypted_content == "v2:state"
        final = events[-1]["response"]
        assert [item["type"] for item in final["output"]] == ["compaction", "message"]
        assert final["usage"]["input_tokens"] == 105
        assert final["usage"]["total_tokens"] == 117
        assert final["usage"]["output_tokens_details"]["reasoning_tokens"] == 5

    @staticmethod
    def _answer_stream(*payloads: dict[str, Any]) -> Any:  # noqa: ANN401
        """Return an answer opener streaming *payloads*.

        Args:
            *payloads: The answer's event payloads.

        Returns:
            The opener.
        """

        async def _events() -> AsyncGenerator[ServerSentEvent]:
            for payload in payloads:
                yield ServerSentEvent(data=dumps(payload), event=str(payload["type"]))

        async def _answer(_item: CompactionItemParam) -> EventSourceResponse:
            from sse_starlette import EventSourceResponse  # noqa: PLC0415

            return EventSourceResponse(_events())

        return _answer

    @staticmethod
    async def _compact() -> tuple[str, ResponseUsage | None]:
        return "v2:state", None

    async def test_a_failed_answer_after_a_compaction_fails_once(self) -> None:
        """An answer failing in-stream relays one error, then one failed snapshot.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
        """
        answer = self._answer_stream(
            {"type": "response.created", "sequence_number": 0},
            {"type": "response.in_progress", "sequence_number": 1},
            {"type": "error", "code": "rate_limit_exceeded", "sequence_number": 2},
            {
                "type": "response.failed",
                "sequence_number": 3,
                "response": {"output": [], "status": "failed"},
            },
        )
        events = await _collect(
            format_compaction_stream(
                "resp-1", 0.0, "m", self._request(), "cmp_1", self._compact, answer
            )
        )
        types = [event["type"] for event in events]
        assert types.count("error") == 1
        assert types[-1] == "response.failed"
        assert [event["sequence_number"] for event in events] == list(
            range(len(events))
        )
        assert events[-1]["response"]["output"][0]["type"] == "compaction"

    async def test_an_answer_refused_after_the_compaction_fails_the_response(
        self,
    ) -> None:
        """An answer refused at open still reports the finished compaction item.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
        """

        async def _refused(_item: CompactionItemParam) -> EventSourceResponse:
            raise ContextLengthExceededError

        events = await _collect(
            format_compaction_stream(
                "resp-1", 0.0, "m", self._request(), "cmp_1", self._compact, _refused
            )
        )
        assert [event["type"] for event in events][-2:] == ["error", "response.failed"]
        assert events[-2]["code"] == "context_length_exceeded"
        assert events[-1]["response"]["output"][0]["type"] == "compaction"

    async def test_a_trigger_reports_compacting_and_ends_with_the_item(self) -> None:
        """Without an answer, the compacting step is reported and the item ends it.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
        """

        async def _compact() -> tuple[str, ResponseUsage | None]:
            return "v2:state", None

        events = await _collect(
            format_compaction_stream(
                "resp-1", 0.0, "m", self._request(), "cmp_1", _compact, None
            )
        )
        assert [event["type"] for event in events] == [
            "response.created",
            "response.in_progress",
            "response.output_item.added",
            "response.compaction.compacting",
            "response.output_item.done",
            "response.completed",
        ]
        assert events[-1]["response"]["output"][0]["id"] == "cmp_1"

    async def test_a_failed_compaction_fails_the_response(self) -> None:
        """A summarisation failure ends the stream with ``response.failed``.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
        """

        async def _compact() -> tuple[str, ResponseUsage | None]:
            raise ContextLengthExceededError

        events = await _collect(
            format_compaction_stream(
                "resp-1", 0.0, "m", self._request(), "cmp_1", _compact, None
            )
        )
        assert [event["type"] for event in events][-2:] == ["error", "response.failed"]
        assert events[-2]["code"] == "context_length_exceeded"


class TestRequestValidation:
    """The request body refuses what upstream refuses, with upstream's param and code.

    The same matrix runs live against both targets in
    test_openai_responses_context.py; this is its offline twin.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/types/openai_responses.py:reject_invalid_context_management
    """

    @pytest.mark.parametrize(
        ("value", "param", "code", "message"),
        [
            ([{"type": "foo"}], "context_management", None, "type: 'foo'"),
            (
                [{"type": "compaction", "compact_threshold": 10}],
                "context_management[0].compact_threshold",
                "integer_below_min_value",
                "but got 10 instead",
            ),
            ([], "context_management", "empty_array", "empty array"),
            (
                [
                    {"type": "compaction"},
                    {"type": "compaction", "compact_threshold": 0.5},
                ],
                "context_management[1].compact_threshold",
                "invalid_type",
                "decimal number",
            ),
            (
                [{"type": "compaction", "compact_threshold": 1000.0}],
                "context_management[0].compact_threshold",
                "invalid_type",
                (
                    "Invalid type for 'context_management[0].compact_threshold': "
                    "expected an integer, but got a decimal number instead."
                ),
            ),
            (
                [{"type": "compaction", "compact_threshold": "1000"}],
                "context_management[0].compact_threshold",
                "invalid_type",
                "expected an integer, but got a string instead.",
            ),
            (
                [{"type": "compaction", "compact_threshold": True}],
                "context_management[0].compact_threshold",
                "invalid_type",
                "expected an integer, but got a boolean instead.",
            ),
            (
                [{"type": "compaction", "compact_threshold": [1000]}],
                "context_management[0].compact_threshold",
                "invalid_type",
                "expected an integer, but got an array instead.",
            ),
            (
                {"type": "compaction"},
                "context_management",
                "invalid_type",
                (
                    "Invalid type for 'context_management': expected an array of "
                    "objects, but got an object instead."
                ),
            ),
            (
                "compaction",
                "context_management",
                "invalid_type",
                "expected an array of objects, but got a string instead.",
            ),
            (
                ["compaction"],
                "context_management[0]",
                "invalid_type",
                (
                    "Invalid type for 'context_management[0]': expected an object, but "
                    "got a string instead."
                ),
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
    def test_context_management_is_refused_as_upstream_does(
        self, value: object, param: str, code: str | None, message: str
    ) -> None:
        """Each malformed value names its path and upstream's code.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        with pytest.raises(ApiError) as excinfo:
            ResponseCreateParams.model_validate(
                {"model": "m", "input": "x", "context_management": value}
            )
        assert (excinfo.value.param, excinfo.value.code) == (param, code)
        assert message in str(excinfo.value)

    def test_valid_context_management_is_accepted(self) -> None:
        """A threshold of 1000, or none, is accepted.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        params = ResponseCreateParams.model_validate(
            {
                "model": "m",
                "input": "x",
                "context_management": [
                    {"type": "compaction", "compact_threshold": 1000},
                    {"type": "compaction"},
                ],
            }
        )
        assert params.context_management is not None
        assert [entry.compact_threshold for entry in params.context_management] == [
            1000,
            None,
        ]

    @pytest.mark.parametrize(
        ("items", "extra", "param", "message"),
        [
            (
                [{"type": "compaction_trigger"}, "q"],
                {},
                "input",
                "The 'compaction_trigger' item must be the final input item.",
            ),
            (
                ["q", {"type": "compaction_trigger"}, {"type": "compaction_trigger"}],
                {"max_output_tokens": 100},
                "input",
                "Only one 'compaction_trigger' item may be provided.",
            ),
            (
                [{"type": "compaction_trigger"}, "q", {"type": "compaction_trigger"}],
                {},
                "input",
                "Only one 'compaction_trigger' item may be provided.",
            ),
            (
                ["q", {"type": "compaction_trigger"}],
                {"max_output_tokens": 100},
                "max_output_tokens",
                (
                    "'compaction_trigger' requires 'max_output_tokens' to be at least "
                    "20000 when specified."
                ),
            ),
        ],
        ids=["not-last", "twice", "twice-first", "small-output-budget"],
    )
    def test_trigger_misuse_is_refused(
        self, items: list[object], extra: dict[str, object], param: str, message: str
    ) -> None:
        """A trigger must come last, once, with room for the compaction.

        Upstream checks the count first, then the position, then the budget.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        body_input = [
            {"role": "user", "content": item} if isinstance(item, str) else item
            for item in items
        ]
        with pytest.raises(ApiError) as excinfo:
            ResponseCreateParams.model_validate(
                {"model": "m", "input": body_input, **extra}
            )
        assert excinfo.value.param == param
        assert str(excinfo.value) == message

    def test_a_last_trigger_with_room_is_accepted(self) -> None:
        """A final trigger with no output budget, or a large one, is valid.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        for extra in ({}, {"max_output_tokens": 20000}):
            ResponseCreateParams.model_validate(
                {
                    "model": "m",
                    "input": [
                        {"role": "user", "content": "q"},
                        {"type": "compaction_trigger"},
                    ],
                    **extra,
                }
            )
