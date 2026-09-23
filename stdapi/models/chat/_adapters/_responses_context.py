"""Context-window handling for the Responses API: overflow, truncation, compaction.

Backends expose no context-window size ahead of a call, so an overflow is only
known from the refusal it earns. This module recognizes that refusal, plans the
input a ``truncation: "auto"`` retry sends instead, and picks what a compaction
summarizes and what it keeps verbatim.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from itertools import pairwise
from re import IGNORECASE
from re import compile as re_compile
from typing import TYPE_CHECKING, Any, Final

from botocore.exceptions import ClientError
from pydantic import BaseModel

from stdapi.api_errors import ApiError
from stdapi.types.openai_responses import (
    CompactionItemParam,
    EasyInputMessage,
    InputMessage,
    ResponseInputItem,
    ResponseOutputMessage,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator, Sequence
    from re import Pattern

#: Upstream's message for an input larger than the model's context window.
_CONTEXT_LENGTH_EXCEEDED: Final = (
    "Your input exceeds the context window of this model. "
    "Please adjust your input and try again."
)

#: Backend refusals of an input larger than the model's context window.
_OVERFLOW_PATTERN: Final = re_compile(
    r"input tokens exceeded|prompt is too long"
    r"|exceeds (?:the )?model's maximum context length"
    r"|maximum context length is \d+|input is too long|exceeds the context window"
    r"|exceed model maximum|exceeds the max_model_len",
    IGNORECASE,
)

#: Refusal of output tokens that leave the input no room in the context window.
OUTPUT_BUDGET_TOO_LARGE: Final = (
    "The requested maximum output tokens leave no room for the input in the "
    "context window of this model. Lower the maximum output tokens."
)

#: Refusals stating the input size first, then the window.
_OBSERVED_LIMIT_PATTERNS: Final = (
    re_compile(r"(\d+) tokens > (\d+) maximum"),
    re_compile(r"Input length \((\d+)\) exceeds [^(]*\((\d+)\)"),
    re_compile(r"prompt tokens \((\d+)\) exceed model maximum \((\d+)\)"),
    re_compile(r"prompt length (\d+) exceeds the max_model_len (\d+)"),
)

#: Refusals stating the window in the OpenAI wording.
_WINDOW_PATTERN: Final = re_compile(
    r"maximum context length is (\d+) tokens", IGNORECASE
)

#: Refusals stating the prompt's tokens, or a lower bound of them, after the window.
_PROMPT_TOKENS_PATTERN: Final = re_compile(
    r"contains (at least )?(\d+) input tokens", IGNORECASE
)

#: Refusals naming the output tokens requested.
_REQUESTED_OUTPUT_PATTERN: Final = re_compile(
    r"requested (\d+) output tokens", IGNORECASE
)

#: Refusals splitting the tokens requested between the messages and the completion.
_MESSAGES_COMPLETION_PATTERN: Final = re_compile(
    r"(\d+) in the messages, (\d+) in the completion", IGNORECASE
)

#: Refusals stating the input tokens the window leaves beside the output requested.
_INPUT_BOUND_PATTERN: Final = re_compile(
    r"upper bound for (\d+) input tokens", IGNORECASE
)

#: Heading of the user message a compaction summary is replayed as.
SUMMARY_HEADING: Final = "Summary of the earlier conversation:\n"

#: Roles whose messages configure the model and are never dropped or summarized away.
_PINNED_ROLES: Final = frozenset({"system", "developer"})

#: Share of the context window a trimmed input aims for on its first retry.
_TRIM_TARGET: Final = 0.9

#: How much lower each further retry aims, since a refusal may understate the input.
_TRIM_TIGHTENING: Final = 0.2

#: Lowest share of the context window a trimmed input aims for.
_TRIM_TARGET_FLOOR: Final = 0.3

#: Share of the input kept when the refusal did not say by how much it overflowed.
_UNSIZED_KEEP_SHARE: Final = 0.5

#: Characters per token a low-biased estimate assumes; English text runs nearer four.
_CHARS_PER_TOKEN_LOW: Final = 6

#: Keys whose values are media, identifiers or opaque state rather than text the model reads.
_UNCOUNTED_KEYS: Final = frozenset(
    {
        "annotations",
        "call_id",
        "detail",
        "encrypted_content",
        "file_data",
        "file_id",
        "file_url",
        "filename",
        "id",
        "image_url",
        "logprobs",
        "role",
        "signature",
        "status",
        "type",
    }
)

#: Errors response streams raised before their first event, for the caller that opened them.
_STREAM_OPEN_ERRORS: ContextVar[list[BaseException] | None] = ContextVar(
    "responses_stream_open_errors", default=None
)


class ContextLengthExceededError(ApiError):
    """The input does not fit the model's context window."""

    code: str | None = "context_length_exceeded"
    disclosed = True

    def __init__(
        self,
        overflow: ContextOverflow | None = None,
        *,
        message: str = _CONTEXT_LENGTH_EXCEEDED,
        param: str | None = "input",
    ) -> None:
        """Create the error, worded as the caller's API words it.

        Args:
            overflow: The refusal it reports, with the sizes the backend stated.
            message: The message, the Responses API's by default.
            param: The parameter holding the input, if the API names one.
        """
        super().__init__(message)
        self.overflow = overflow or ContextOverflow()
        self.param = param


@dataclass(frozen=True, slots=True)
class ContextOverflow:
    """A context-window refusal, with the sizes the backend stated, if any."""

    #: Input tokens the backend counted before refusing.
    observed: int | None = None
    #: Tokens the context window takes.
    limit: int | None = None
    #: Output tokens the request asked for, when the refusal states them.
    output: int | None = None
    #: Whether ``observed`` is only a lower bound, the backend having stopped counting.
    bounded: bool = False

    @property
    def sized(self) -> bool:
        """Whether the stated input alone exceeds the stated window."""
        return (
            self.observed is not None
            and self.limit is not None
            and self.observed > self.limit
        )

    @property
    def combined(self) -> bool:
        """Whether the stated input fits the window alone, but not with the output."""
        return (
            self.observed is not None
            and self.limit is not None
            and self.output is not None
            and self.observed <= self.limit < self.observed + self.output
        )


@dataclass(frozen=True, slots=True)
class CompactionSplit:
    """How a compaction divides an input."""

    #: Items the summary is written from, including the ones kept verbatim.
    summarized: list[ResponseInputItem]
    #: Items kept verbatim ahead of the summary.
    before: list[ResponseInputItem]
    #: Items kept verbatim after the summary.
    after: list[ResponseInputItem]


def _validation_message(exc: BaseException) -> str | None:
    """Return the message of a backend's validation refusal.

    Args:
        exc: The exception a model call raised.

    Returns:
        The message, or None when the exception is no validation refusal.
    """
    if isinstance(exc, ClientError):
        error = exc.response.get("Error") or {}
        if str(error.get("Code") or "").lower() != "validationexception":
            return None
        return str(error.get("Message") or "")
    if isinstance(exc, ApiError) and exc.status == 400 and exc.args:
        return str(exc.args[0])
    return None


def _refusal_message(exc: BaseException) -> str | None:
    """Return the message of a backend's context-window refusal.

    Args:
        exc: The exception a model call raised.

    Returns:
        The message, or None when the exception is no such refusal.
    """
    if (message := _validation_message(exc)) is None:
        return None
    if isinstance(exc, ApiError) and exc.code == "context_length_exceeded":
        return message
    return message if _OVERFLOW_PATTERN.search(message) else None


def context_overflow(exc: BaseException) -> ContextOverflow | None:
    """Recognize a backend's refusal of an input the context window cannot hold.

    The input overflows alone, or only with the output tokens requested: either
    way a shorter input fits. A refusal of output tokens filling the window
    alone is not one, since no input trimming answers it.

    Args:
        exc: The exception a model call raised.

    Returns:
        The overflow, with the sizes the refusal stated, or None when the
        exception is anything else.
    """
    if isinstance(exc, ContextLengthExceededError):
        return exc.overflow
    if (message := _refusal_message(exc)) is None or _output_fills_window(message):
        return None
    overflow = _overflow_sizes(message)
    if overflow.observed is not None and overflow.limit is not None:
        # Stated sizes that fit the window do not describe this refusal.
        return overflow if overflow.sized or overflow.combined else None
    return overflow


def capped_output_budget(overflow: ContextOverflow | None) -> int | None:
    """Return the output budget that fits beside an input the window holds alone.

    Upstream Responses and Messages answer such a request with its output
    capped, so a refusal of it is retried once with this budget.

    Args:
        overflow: The refusal, if the error was one.

    Returns:
        The tokens the window leaves beside the input, or None unless the
        refusal stated both sizes and the input fits the window alone.
    """
    # vLLM counts only up to what the output leaves, so a bound sizes nothing.
    if (
        overflow is None
        or overflow.bounded
        or overflow.observed is None
        or overflow.limit is None
        or not overflow.combined
    ):
        return None
    budget = overflow.limit - overflow.observed
    return budget if budget > 0 else None


def output_budget_exceeded(exc: BaseException) -> bool:
    """Whether a backend refused output tokens that alone fill the context window.

    Args:
        exc: The exception a model call raised.

    Returns:
        True for a context-window refusal no input trimming answers.
    """
    if isinstance(exc, ContextLengthExceededError):
        return False
    message = _refusal_message(exc)
    return message is not None and _output_fills_window(message)


def _output_fills_window(message: str) -> bool:
    """Whether a refusal states output tokens leaving the input no room.

    Args:
        message: The backend's refusal message.

    Returns:
        True when the output requested takes the whole window.
    """
    if _number(_INPUT_BOUND_PATTERN, message) == 0:
        return True
    sizes = _overflow_sizes(message)
    return (
        sizes.output is not None
        and sizes.limit is not None
        and sizes.output >= sizes.limit
    )


def _overflow_sizes(message: str) -> ContextOverflow:
    """Read the input size, the window and the output out of a refusal message.

    Args:
        message: The backend's refusal message.

    Returns:
        The overflow, with whichever sizes the message states.
    """
    for pattern in _OBSERVED_LIMIT_PATTERNS:
        if match := pattern.search(message):
            return ContextOverflow(int(match[1]), int(match[2]))
    output = _number(_REQUESTED_OUTPUT_PATTERN, message)
    observed, bounded = None, False
    if match := _PROMPT_TOKENS_PATTERN.search(message):
        observed, bounded = int(match[2]), bool(match[1])
    if match := _MESSAGES_COMPLETION_PATTERN.search(message):
        observed, output = int(match[1]), int(match[2])
    return ContextOverflow(observed, _number(_WINDOW_PATTERN, message), output, bounded)


def _number(pattern: Pattern[str], message: str) -> int | None:
    """Return the number a pattern's first group captures in a message.

    Args:
        pattern: A pattern capturing digits.
        message: The message.

    Returns:
        The number, or None when the pattern does not match.
    """
    return int(match[1]) if (match := pattern.search(message)) else None


def record_stream_open_error(exc: BaseException) -> None:
    """Report an error a response stream raised before its first event.

    Args:
        exc: The error, handed to whichever caller is collecting them.
    """
    if (errors := _STREAM_OPEN_ERRORS.get()) is not None:
        errors.append(exc)


@contextmanager
def collect_stream_open_errors() -> Generator[list[BaseException]]:
    """Collect the errors response streams raise before their first event.

    Yields:
        The list the errors are appended to, while the context is open.
    """
    errors: list[BaseException] = []
    token = _STREAM_OPEN_ERRORS.set(errors)
    try:
        yield errors
    finally:
        _STREAM_OPEN_ERRORS.reset(token)


def _role(item: object) -> str | None:
    """Return a message item's role, or None for any other item.

    Args:
        item: An input item.

    Returns:
        The role.
    """
    role = getattr(item, "role", None)
    return role if isinstance(role, str) else None


def _pinned(item: object) -> bool:
    """Whether an item configures the model, so it is never dropped or summarized away.

    Args:
        item: An input item.

    Returns:
        True for a system or developer message.
    """
    return _role(item) in _PINNED_ROLES


def _starts_turn(item: object) -> bool:
    """Whether an item opens a turn: a user message, or a compaction item.

    Args:
        item: An input item.

    Returns:
        True when the item opens a turn.
    """
    return isinstance(item, CompactionItemParam) or (
        isinstance(item, EasyInputMessage | InputMessage) and item.role == "user"
    )


def is_summary(item: object) -> bool:
    """Whether an item is a compaction summary replayed as a user message.

    Args:
        item: An input item.

    Returns:
        True for a user message opening with the summary heading.
    """
    return (
        isinstance(item, EasyInputMessage)
        and isinstance(item.content, str)
        and item.content.startswith(SUMMARY_HEADING)
    )


def _is_tool_output(item: object) -> bool:
    """Whether an item carries a tool's output back to the model.

    Args:
        item: An input item.

    Returns:
        True for any ``*_output`` item.
    """
    return str(getattr(item, "type", None) or "").endswith("_output")


def _ends_step(item: object) -> bool:
    """Whether a model step ends with this item: a tool output or an assistant message.

    Args:
        item: An input item.

    Returns:
        True when the next non-output item starts a new step.
    """
    return (
        _is_tool_output(item)
        or isinstance(item, ResponseOutputMessage)
        or (_role(item) == "assistant")
    )


def _as_items(value: str | Sequence[ResponseInputItem]) -> list[ResponseInputItem]:
    """Return an ``input`` value as a list of items.

    Args:
        value: The request's ``input``.

    Returns:
        The items, a bare string becoming one user message.
    """
    if isinstance(value, str):
        return [EasyInputMessage(role="user", content=value)]
    return list(value)


def truncate_input(
    value: str | Sequence[ResponseInputItem] | None,
    overflow: ContextOverflow,
    attempt: int,
) -> list[ResponseInputItem] | None:
    """Plan the input a ``truncation: "auto"`` retry sends after an overflow.

    Whole turns are dropped oldest first, a turn being a user message and
    everything up to the next one, so a tool call is never separated from its
    output. System and developer messages, and the latest turn, are always
    kept. When no turn is left to drop, the latest turn's largest text is
    cut, keeping its beginning; a reasoning item's signed text never is.

    Args:
        value: The refused request's ``input``.
        overflow: The refusal.
        attempt: How many retries already failed, each one aiming lower.

    Returns:
        The input to retry with, or None when nothing is left to remove.
    """
    if not value:
        return None
    items = _as_items(value)
    # Text only: inline media would swamp the share the cut has to remove.
    sizes = [_text_length(item) for item in items]
    total = sum(sizes)
    budget = int(total * _keep_share(overflow, attempt))
    starts = sorted({0, *(i for i, item in enumerate(items) if _starts_turn(item))})
    dropped: set[int] = set()
    kept = total
    for start, end in pairwise(starts):
        if kept <= budget:
            break
        for index in range(start, end):
            if not _pinned(items[index]):
                dropped.add(index)
                kept -= sizes[index]
    trimmed = [item for index, item in enumerate(items) if index not in dropped]
    if dropped:
        # Whatever is still too large is left to the next refusal to size.
        return trimmed
    return _cut_largest_text(trimmed, kept - budget) if kept > budget else None


def _keep_share(overflow: ContextOverflow, attempt: int) -> float:
    """Return the share of the refused input a retry keeps.

    Args:
        overflow: The refusal.
        attempt: How many retries already failed.

    Returns:
        A share between 0 and 1.
    """
    observed, limit, output = overflow.observed, overflow.limit, overflow.output or 0
    if observed and limit and output < limit < observed + output:
        # The input fits what the window leaves beside the output requested.
        target = max(_TRIM_TARGET - _TRIM_TIGHTENING * attempt, _TRIM_TARGET_FLOOR)
        return target * (limit - output) / observed
    return _UNSIZED_KEEP_SHARE


def _text_slots(dumped: dict[str, Any]) -> Iterator[tuple[dict[str, Any], str]]:
    """Yield where a dumped item holds text: its container and key.

    Args:
        dumped: An input item in JSON form.

    Yields:
        ``(container, key)`` pairs whose value is a string the model reads.
    """
    for field in ("content", "output"):
        value = dumped.get(field)
        if isinstance(value, str):
            yield dumped, field
        elif isinstance(value, list):
            for part in value:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    yield part, "text"


def _cut_largest_text(
    items: list[ResponseInputItem], excess: int
) -> list[ResponseInputItem] | None:
    """Cut the end of the largest text an input carries.

    Args:
        items: The input, already stripped of the turns that could go.
        excess: Characters to remove.

    Returns:
        The input with that text shortened, or None when no text is long
        enough to absorb the cut.
    """
    best: tuple[int, int, dict[str, Any], dict[str, Any], str] | None = None
    for index, item in enumerate(items):
        # A reasoning item's text is signed, and a cut would void its signature.
        if _pinned(item) or getattr(item, "type", None) == "reasoning":
            continue
        dumped = item.model_dump(mode="json", by_alias=True, exclude_none=True)
        for container, key in _text_slots(dumped):
            length = len(container[key])
            if best is None or length > best[0]:
                best = (length, index, dumped, container, key)
    if best is None or best[0] <= excess:
        return None
    length, index, dumped, container, key = best
    container[key] = container[key][: length - excess]
    cut = type(items[index]).model_validate(dumped)
    return [*items[:index], cut, *items[index + 1 :]]


def split_for_compaction(
    value: str | Sequence[ResponseInputItem] | None,
) -> CompactionSplit | None:
    """Divide an input into what a compaction summarizes and what it keeps.

    The latest step stays verbatim: the latest user message and what follows
    it, or, when a single user message opened a run of tool steps, that
    message and the latest step. System and developer messages are kept too.
    Everything before is summarized, earlier summaries included: a summary is
    never the user message kept, so kept items never accumulate summaries.

    Args:
        value: The request's ``input``, compaction items already expanded.

    Returns:
        The split, or None when nothing precedes the latest step.
    """
    if not value:
        return None
    items = _as_items(value)
    last_user = max(
        (
            i
            for i, item in enumerate(items)
            if _starts_turn(item) and not is_summary(item)
        ),
        default=None,
    )
    first_step = 0 if last_user is None else last_user + 1
    step_starts = [
        i
        for i in range(first_step + 1, len(items))
        if _ends_step(items[i - 1]) and not _is_tool_output(items[i])
    ]
    if step_starts:
        cut = step_starts[-1]
    elif last_user is not None and not all(map(_pinned, items[:last_user])):
        cut = last_user
    else:
        return None
    return CompactionSplit(
        summarized=items[:cut],
        before=[
            item
            for i, item in enumerate(items[:cut])
            if _pinned(item) or i == last_user
        ],
        after=items[cut:],
    )


def split_everything(items: Sequence[ResponseInputItem]) -> CompactionSplit | None:
    """Divide an input for a compaction of all of it, as a trigger asks for.

    Args:
        items: The input before the trigger, compaction items already expanded.

    Returns:
        The split keeping only system and developer messages verbatim, or None
        when nothing else is there to compact.
    """
    before = [item for item in items if _pinned(item)]
    if len(before) == len(items):
        return None
    return CompactionSplit(summarized=list(items), before=before, after=[])


def estimate_tokens(*values: object) -> int:
    """Estimate, biased low, how many tokens request content takes.

    Only text the model reads is counted, at a generous number of characters
    per token: media, identifiers and opaque state are skipped.

    Args:
        *values: Request content: input items, instructions, tools.

    Returns:
        The estimate.
    """
    return _text_length(values) // _CHARS_PER_TOKEN_LOW


def _text_length(value: object) -> int:
    """Return how many characters of text a value holds.

    Args:
        value: A string, model, mapping or sequence.

    Returns:
        The character count.
    """
    match value:
        case str():
            return len(value)
        case BaseModel():
            return _text_length(value.model_dump(mode="json", exclude_none=True))
        case dict():
            return sum(
                _text_length(item)
                for key, item in value.items()
                if key not in _UNCOUNTED_KEYS
            )
        case list() | tuple():
            return sum(_text_length(item) for item in value)
        case _:
            return 0
