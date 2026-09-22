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
from re import DOTALL, IGNORECASE
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

#: Upstream's message for an input larger than the model's context window.
_CONTEXT_LENGTH_EXCEEDED: Final = (
    "Your input exceeds the context window of this model. "
    "Please adjust your input and try again."
)

#: Backend refusals of an input larger than the model's context window.
_OVERFLOW_PATTERN: Final = re_compile(
    r"input tokens exceeded|prompt is too long"
    r"|exceeds (?:the )?model's maximum context length"
    r"|maximum context length is \d+|input is too long|exceeds the context window",
    IGNORECASE,
)

#: Refusals stating the input size first, then the window.
_OBSERVED_LIMIT_PATTERNS: Final = (
    re_compile(r"(\d+) tokens > (\d+) maximum"),
    re_compile(r"Input length \((\d+)\) exceeds [^(]*\((\d+)\)"),
)

#: Refusals stating the window first, then the input size.
_LIMIT_OBSERVED_PATTERN: Final = re_compile(
    r"maximum context length is (\d+) tokens.*?contains (?:at least )?(\d+) input tokens",
    DOTALL,
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

    code = "context_length_exceeded"
    param = "input"
    disclosed = True

    def __init__(self) -> None:
        """Create the error with upstream's message."""
        super().__init__(_CONTEXT_LENGTH_EXCEEDED)


@dataclass(frozen=True, slots=True)
class ContextOverflow:
    """A context-window refusal, with the sizes the backend stated, if any."""

    #: Input tokens the backend counted before refusing.
    observed: int | None = None
    #: Tokens the context window takes.
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class CompactionSplit:
    """How a compaction divides an input."""

    #: Items the summary is written from, including the ones kept verbatim.
    summarized: list[ResponseInputItem]
    #: Items kept verbatim ahead of the summary.
    before: list[ResponseInputItem]
    #: Items kept verbatim after the summary.
    after: list[ResponseInputItem]


def context_overflow(exc: BaseException) -> ContextOverflow | None:
    """Recognize a backend's refusal of an input larger than the context window.

    Args:
        exc: The exception a model call raised.

    Returns:
        The overflow, with the sizes the refusal stated, or None when the
        exception is anything else.
    """
    if isinstance(exc, ContextLengthExceededError):
        return ContextOverflow()
    if isinstance(exc, ClientError):
        error = exc.response.get("Error") or {}
        if str(error.get("Code") or "").lower() != "validationexception":
            return None
        message = str(error.get("Message") or "")
    elif isinstance(exc, ApiError) and exc.status == 400 and exc.args:
        message = str(exc.args[0])
        if exc.code == "context_length_exceeded":
            return _overflow_sizes(message)
    else:
        return None
    return _overflow_sizes(message) if _OVERFLOW_PATTERN.search(message) else None


def _overflow_sizes(message: str) -> ContextOverflow:
    """Read the input size and the window out of a refusal message.

    Args:
        message: The backend's refusal message.

    Returns:
        The overflow, with whichever sizes the message states.
    """
    for pattern in _OBSERVED_LIMIT_PATTERNS:
        if match := pattern.search(message):
            return ContextOverflow(int(match[1]), int(match[2]))
    if match := _LIMIT_OBSERVED_PATTERN.search(message):
        return ContextOverflow(int(match[2]), int(match[1]))
    return ContextOverflow()


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
    if overflow.observed and overflow.limit and overflow.observed > overflow.limit:
        target = max(_TRIM_TARGET - _TRIM_TIGHTENING * attempt, _TRIM_TARGET_FLOOR)
        return target * overflow.limit / overflow.observed
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
