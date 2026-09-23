"""Token counting where the backend's counter refuses what upstream counts.

An input over the context window is counted in pieces the counter accepts, a
server tool the counter does not take is counted as a stub of its name plus the
tokens its definition adds upstream, and a server tool call or result replayed
in history is counted as its text plus the tokens upstream counts beyond it.
A model no counter serves gets an approximation that errs high, never low.
"""

from asyncio import Semaphore, TaskGroup
from dataclasses import dataclass, field
from http import HTTPStatus
from io import BytesIO
from itertools import accumulate
from math import ceil, sumprod
from re import IGNORECASE, Pattern
from re import compile as re_compile
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from botocore.exceptions import BotoCoreError, ClientError
from PIL import Image
from pydantic import BaseModel

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock import handle_bedrock_client_error
from stdapi.aws_bedrock_mantle import (
    MantleApiUnsupportedError,
    MantleError,
    MantleSurfaceUnsupportedError,
)
from stdapi.input_file import InputFile, resolve_inline_bedrock_content_blocks
from stdapi.models import catalog_model
from stdapi.models.chat._adapters._responses_context import (
    ContextOverflow,
    context_overflow,
)
from stdapi.monitoring import log_error_details
from stdapi.types.anthropic_messages import TextBlockParam
from stdapi.utils import b64decode, to_json_str

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_runtime.client import BedrockRuntimeClient
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ContentBlockTypeDef,
        ConverseTokensRequestTypeDef,
        MessageTypeDef,
    )

    from stdapi.models import ModelDetails
    from stdapi.types.anthropic_messages import MessageParam

#: Tokens upstream counts for a server tool beyond a ``{"type": "object"}`` stub of its name, by type (Claude 4.5 and 4.6).
SERVER_TOOL_TOKENS: Final = MappingProxyType(
    {
        "web_search": MappingProxyType(
            {"web_search_20250305": 1678, "web_search_20260209": 3844}
        ),
        "web_fetch": MappingProxyType(
            {"web_fetch_20250910": 493, "web_fetch_20260209": 2925}
        ),
        "code_execution": MappingProxyType(
            {
                "code_execution_20250522": 392,
                "code_execution_20250825": 1708,
                "code_execution_20260120": 1708,
            }
        ),
        "tool_search_tool_bm25": MappingProxyType(
            {"tool_search_tool_bm25_20251119": 144}
        ),
        "tool_search_tool_regex": MappingProxyType(
            {"tool_search_tool_regex_20251119": 173}
        ),
    }
)

#: Server tool types, the tool's family captured.
_SERVER_TOOL_TYPE: Final = re_compile(
    rf"^({'|'.join(SERVER_TOOL_TOKENS)})(?:_\d{{8}})?$"
)

#: Message leading every piece after the first, since a conversation starts with a user turn.
_LEAD: Final[MessageTypeDef] = {"role": "user", "content": [{"text": "."}]}

#: Most counter calls one request spends counting past the context window.
_MAX_CALLS: Final = 64

#: Counter calls one request runs at once counting past the context window.
_CONCURRENCY: Final = 4

#: Content blocks a user message holds as they stand.
_PLAIN_BLOCKS: Final = frozenset({"text", "image", "document", "video"})

#: Content block types recording a server tool's call or result.
_SERVER_TOOL_BLOCKS: Final = frozenset(
    {
        "server_tool_use",
        "web_search_tool_result",
        "web_fetch_tool_result",
        "code_execution_tool_result",
        "bash_code_execution_tool_result",
        "text_editor_code_execution_tool_result",
        "tool_search_tool_result",
    }
)

#: Tokens upstream counts for a server tool call and its result beyond their JSON text (Claude Haiku 4.5).
_SERVER_TOOL_RESULT_TOKENS: Final = 31

#: Tokens upstream counts for a web search result beyond its title and URL, besides its page.
_SEARCH_RESULT_TOKENS: Final = 12

#: Tokens upstream counts per character of a web search result's encrypted page.
_SEARCH_PAGE_TOKENS_PER_CHAR: Final = 0.296

#: Citation type pointing into a web search result.
_SEARCH_CITATION: Final = "web_search_result_location"

#: Tokens upstream counts for a web search citation.
_SEARCH_CITATION_TOKENS: Final = 16

#: Refusal of a model its backend's counter does not count.
_UNCOUNTABLE_MESSAGE: Final = "support counting tokens"

#: Models whose backend refused to count their tokens.
UNCOUNTABLE_MODELS: Final[set[str]] = set()

#: Models whose backend counted their tokens.
_COUNTABLE_MODELS: Final[set[str]] = set()

#: Refusals worded by the counter itself rather than by the model, naming internals.
_COUNTER_WORDING: Final = re_compile(
    r"token counting|counting tokens|Input tag '[^']*' found", IGNORECASE
)

#: Answer to a request the counter refused, its own wording kept to the server log.
_REFUSED: Final = (
    "The request could not be counted. Check it against the API reference and "
    "try again."
)

#: Countable Claude model whose count stands for the Claude models no counter serves.
PROXY_MODEL_ID: Final = "anthropic.claude-haiku-4-5-20251001-v1:0"

#: Claude model identifiers.
_CLAUDE: Final = re_compile(r"anthropic\.claude-")

#: Claude models sharing the proxy's tokenizer (Claude 3 to 4.6), counted by it unscaled.
_PROXY_TOKENIZER: Final = re_compile(
    r"anthropic\.claude-(?:3|(?:haiku|sonnet|opus)-4(?:-[0-6])?(?:-\d{8}|-v\d|:|$))"
)

#: Ratio a Claude 4.7+ count takes over the proxy's on prose, code, JSON and tool definitions (1.515 at most); the character estimate covers what runs higher, such as identifiers.
_PROXY_RATIO: Final = 1.52

#: Tokens a Claude 4.7+ image takes beyond the proxy's scaled count (4,756 vs 1.52 x 1,592 at 2576 px).
_PROXY_IMAGE_TOKENS: Final = 2400

#: Estimate coefficients where no tokenizer family matches: fixed tokens; per ASCII letter, whitespace, digit, punctuation, random-run character, two-byte, other three-byte, CJK/kana/Hangul and four-byte character; tokens a model's template adds when the request offers tools; the most tokens one image takes.
_ESTIMATE_DEFAULT: Final = (
    65.0,
    0.336,
    0.123,
    1.157,
    0.473,
    0.931,
    1.062,
    1.323,
    1.671,
    4.415,
    800.0,
    16500.0,
)

#: Estimate coefficients of the Claude models, as ``_ESTIMATE_DEFAULT``.
_ESTIMATE_CLAUDE: Final = (
    16.0,
    0.549,
    0.062,
    0.579,
    0.558,
    0.997,
    0.763,
    0.966,
    1.5,
    3.275,
    550.0,
    4800.0,
)

#: Estimate coefficients per tokenizer family, as ``_ESTIMATE_DEFAULT``, each above every model of the family measured.
_ESTIMATE_FAMILIES: Final[tuple[tuple[Pattern[str], tuple[float, ...]], ...]] = (
    (re_compile(r"anthropic\.claude-"), _ESTIMATE_CLAUDE),
    (
        re_compile(r"^amazon\.nova"),
        (
            65.0,
            0.336,
            0.123,
            1.157,
            0.473,
            0.922,
            0.531,
            0.908,
            1.671,
            3.231,
            800.0,
            16500.0,
        ),
    ),
    (
        re_compile(r"^meta\.llama3"),
        (
            65.0,
            0.319,
            0.083,
            0.579,
            0.468,
            0.733,
            0.531,
            0.662,
            0.836,
            3.489,
            0.0,
            16500.0,
        ),
    ),
    (
        re_compile(
            r"^mistral\.(?:mistral-7b|mixtral|mistral-large-2402|mistral-small-2402)"
        ),
        (
            65.0,
            0.336,
            0.123,
            1.157,
            0.473,
            0.931,
            1.062,
            1.323,
            1.536,
            2.689,
            0.0,
            16500.0,
        ),
    ),
    (
        re_compile(r"^mistral\."),
        (
            65.0,
            0.277,
            0.123,
            1.157,
            0.473,
            0.918,
            0.531,
            1.323,
            0.909,
            4.415,
            0.0,
            16500.0,
        ),
    ),
    (
        re_compile(r"^openai\.gpt-"),
        (
            65.0,
            0.336,
            0.071,
            0.621,
            0.473,
            0.718,
            0.531,
            0.662,
            0.836,
            2.351,
            0.0,
            16500.0,
        ),
    ),
    (
        re_compile(r"^qwen\."),
        (
            65.0,
            0.303,
            0.114,
            1.13,
            0.473,
            0.916,
            0.531,
            1.149,
            0.842,
            2.208,
            170.0,
            16500.0,
        ),
    ),
    (
        re_compile(r"^google\.gemma"),
        (
            65.0,
            0.336,
            0.091,
            1.157,
            0.473,
            0.917,
            0.531,
            0.662,
            0.836,
            2.208,
            0.0,
            16500.0,
        ),
    ),
    (
        re_compile(r"^deepseek\."),
        (
            65.0,
            0.276,
            0.09,
            0.671,
            0.473,
            0.73,
            0.531,
            0.662,
            0.836,
            2.208,
            20.0,
            16500.0,
        ),
    ),
    (
        re_compile(r"^minimax\."),
        (
            65.0,
            0.24,
            0.08,
            0.858,
            0.473,
            0.683,
            0.531,
            1.094,
            0.836,
            2.208,
            50.0,
            16500.0,
        ),
    ),
    (
        re_compile(r"^moonshot"),
        (
            65.0,
            0.336,
            0.123,
            0.579,
            0.473,
            0.695,
            1.062,
            0.871,
            1.023,
            2.633,
            0.0,
            16500.0,
        ),
    ),
    (
        re_compile(r"^nvidia\."),
        (
            65.0,
            0.336,
            0.097,
            1.115,
            0.473,
            0.92,
            0.531,
            1.323,
            1.671,
            4.415,
            90.0,
            16500.0,
        ),
    ),
    (
        re_compile(r"^writer\."),
        (
            65.0,
            0.304,
            0.119,
            1.118,
            0.466,
            0.907,
            0.531,
            1.13,
            0.836,
            2.208,
            0.0,
            16500.0,
        ),
    ),
    (
        re_compile(r"^zai\."),
        (
            65.0,
            0.26,
            0.123,
            0.735,
            0.465,
            0.734,
            0.531,
            1.075,
            0.92,
            2.208,
            35.0,
            16500.0,
        ),
    ),
)

#: Long runs of ASCII word characters, random-looking when they mix letters and digits.
_RANDOM_RUN: Final = re_compile(r"[A-Za-z0-9+/=_-]{24,}")

#: ASCII digits.
_DIGITS: Final = b"0123456789"

#: ASCII letters.
_LETTERS: Final = bytes(range(65, 91)) + bytes(range(97, 123))

#: ASCII bytes.
_ASCII: Final = bytes(range(128))

#: ASCII whitespace.
_SPACE: Final = b" \t\n\r"

#: UTF-8 lead bytes of a two-byte character: Latin extensions, Greek, Cyrillic, Hebrew, Arabic.
_TWO_BYTE_LEADS: Final = bytes(range(0xC2, 0xE0))

#: UTF-8 lead bytes of a three-byte character below U+3000: Indic, Thai, symbols.
_THREE_BYTE_LEADS: Final = bytes(range(0xE0, 0xE3))

#: UTF-8 lead bytes of a three-byte character from U+3000: CJK, kana, Hangul, full-width forms.
_CJK_LEADS: Final = bytes(range(0xE3, 0xF0))

#: UTF-8 lead bytes of a four-byte character: emoji and rare scripts.
_FOUR_BYTE_LEADS: Final = bytes(range(0xF0, 0xF8))

#: Keys of request content no model reads as text.
_UNCOUNTED_KEYS: Final = frozenset({"signature"})

#: Fewest tokens an image is estimated at (Amazon Nova Lite took 2,610 for 1024 px).
_IMAGE_MIN_TOKENS: Final = 2700

#: Most tokens an image is estimated at before its family's own cap, and an unreadable one.
_IMAGE_MAX_TOKENS: Final = 16500

#: Image pixels per estimated token, above the minimum (Pixtral Large took 4,160 for 1024 px: 252).
_PIXELS_PER_TOKEN: Final = 240

#: Tokens a Claude image the counter cannot read is counted at (Claude Haiku 4.5 took 1,592 at most).
_CLAUDE_IMAGE_TOKENS: Final = 1650

#: Tokens a PDF page is estimated at.
_PDF_PAGE_TOKENS: Final = 3000

#: Fewest tokens a file given by reference, or unreadable, is estimated at.
_FILE_TOKENS: Final = 30000

#: Tokens per byte a file given by reference is estimated at, above its minimum.
_FILE_TOKENS_PER_BYTE: Final = 1

#: A page object in a PDF.
_PDF_PAGE: Final = re_compile(rb"/Type\s*/Page(?![A-Za-z])")

#: Counts one Converse request.
type _Counter = Callable[[ConverseTokensRequestTypeDef], Awaitable[int]]


def server_tool_tokens(tool_type: object) -> int | None:
    """Return the tokens a server tool's definition adds beyond a stub of its name.

    Args:
        tool_type: The tool's ``type``.

    Returns:
        The tokens, those of the family's latest known version for an unknown
        version, or None when the type names no server tool.
    """
    if not isinstance(tool_type, str) or not (
        match := _SERVER_TOOL_TYPE.match(tool_type)
    ):
        return None
    versions = SERVER_TOOL_TOKENS[match[1]]
    return versions.get(tool_type) or versions[max(versions)]


def stub_server_tools(tools: list[Any]) -> int:
    """Replace the server tools of an Anthropic ``tools`` list by stubs of their names.

    Args:
        tools: The request's tool definitions, changed in place.

    Returns:
        The tokens the replaced definitions add upstream.
    """
    tokens = 0
    for index, tool in enumerate(tools):
        if (
            isinstance(tool, dict)
            and (added := server_tool_tokens(tool.get("type"))) is not None
        ):
            name = tool.get("name")
            tools[index] = {
                "name": name,
                "description": name,
                "input_schema": {"type": "object"},
            }
            tokens += added
    return tokens


def plain_server_tool_history(
    messages: list[MessageParam],
) -> tuple[list[MessageParam], int]:
    """Replace the server tool calls, results and citations a history replays by text.

    Args:
        messages: The request's messages.

    Returns:
        The messages, and the tokens upstream counts beyond that text.
    """
    tokens = 0
    plain: list[MessageParam] = []
    for message in messages:
        if isinstance(message.content, str):
            plain.append(message)
            continue
        content: list[Any] = []
        for block in message.content:
            replaced, added = _plain_history_block(block)
            content.append(replaced)
            tokens += added
        changed = any(
            new is not old for new, old in zip(content, message.content, strict=True)
        )
        plain.append(
            message.model_copy(update={"content": content}) if changed else message
        )
    return plain, tokens


def _plain_history_block(block: Any) -> tuple[Any, int]:  # noqa: ANN401
    """Return a history block as the counter takes it, and the tokens it misses.

    Args:
        block: A content block.

    Returns:
        The block, or its text for a server tool call or result, or the text
        block without its web search citations; and the tokens upstream counts
        beyond it.
    """
    block_type = getattr(block, "type", None)
    if block_type == "text":
        citations = block.citations or ()
        kept = [c for c in citations if c.type != _SEARCH_CITATION]
        if len(kept) == len(citations):
            return block, 0
        return block.model_copy(update={"citations": kept or None}), (
            len(citations) - len(kept)
        ) * _SEARCH_CITATION_TOKENS
    if block_type not in _SERVER_TOOL_BLOCKS:
        return block, 0
    dumped = block.model_dump(mode="json", exclude_none=True)
    content = dumped.get("content")
    if block_type == "server_tool_use":
        text = to_json_str({"name": dumped["name"], "input": dumped["input"]})
        tokens = 0
    elif block_type == "web_search_tool_result" and isinstance(content, list):
        # Upstream reads each page, which only its encrypted form sizes.
        text = "\n".join(f"{result['title']}\n{result['url']}" for result in content)
        tokens = _SERVER_TOOL_RESULT_TOKENS + sum(
            _SEARCH_RESULT_TOKENS
            + round(len(result["encrypted_content"]) * _SEARCH_PAGE_TOKENS_PER_CHAR)
            for result in content
        )
    else:
        text = to_json_str(content)
        tokens = _SERVER_TOOL_RESULT_TOKENS
    return TextBlockParam(type="text", text=text or "."), tokens


def uncountable(exc: ClientError) -> bool:
    """Whether the backend's counter refused a model it does not count.

    Args:
        exc: The counter's error.

    Returns:
        True for that refusal.
    """
    return _UNCOUNTABLE_MESSAGE in str(
        (exc.response.get("Error") or {}).get("Message") or ""
    )


async def count_or_approximate[T](
    model: ModelDetails,
    request: BaseModel,
    exact: Callable[[], Awaitable[T]] | None,
    proxy: Callable[[ModelDetails], Awaitable[T]],
    wrap: Callable[[int], T],
    remap: Callable[[T, Callable[[int], int]], T],
    probe: Callable[[], Awaitable[object]] | None = None,
) -> T:
    """Count a request's tokens exactly where a counter serves the model, else approximate high.

    A Claude model no counter serves is counted by a countable Claude model,
    scaled by the highest ratio measured between their tokenizers; any other
    model is estimated locally, with weights above every tokenizer measured.
    Whether the counter serves a model is learned from a trivial request only,
    never from a caller's own request, and remembered for every caller.

    Args:
        model: The model the request names.
        request: The request, for the local estimate.
        exact: Counts the request with the model's own counter, if it has one.
        proxy: Counts the request with another model's counter.
        wrap: Builds the answer from a token count.
        remap: Applies a function to every token count of an answer.
        probe: Counts a trivial request with the model's own counter.

    Returns:
        The answer.

    Raises:
        ApiError: When the counter refuses the request itself.
    """
    if exact is not None and await _serves(model.id, probe):
        try:
            return await _counted(exact)
        except _UncountableError as exc:
            if probe is not None:
                # The probe proved the model countable: this request is the cause.
                log_error_details(str(exc.__cause__), status=400)
                raise ApiError(_REFUSED) from exc
    # Estimated first: the proxy's count consumes the request's inline files.
    estimate = await estimate_request_tokens(request, model.id)
    if _CLAUDE.search(model.id) and (proxy_model := catalog_model(PROXY_MODEL_ID)):
        try:
            counted = await proxy(proxy_model)
        except (ApiError, ClientError, BotoCoreError) as exc:
            log_error_details(
                f"Token counting through {PROXY_MODEL_ID} failed ({exc}); the "
                f"count for {model.id} is estimated locally.",
                level="warning",
            )
        else:
            if _PROXY_TOKENIZER.search(model.id):
                return counted
            extra = _image_count(request) * _PROXY_IMAGE_TOKENS
            return remap(
                counted,
                lambda tokens: max(ceil(tokens * _PROXY_RATIO) + extra, estimate),
            )
    return wrap(estimate)


async def _serves(model_id: str, probe: Callable[[], Awaitable[object]] | None) -> bool:
    """Whether the model's own counter serves it, learned once from a trivial request.

    Args:
        model_id: The model.
        probe: Counts a trivial request with the model's own counter.

    Returns:
        False once the counter refused the trivial request, True otherwise.
    """
    if model_id in UNCOUNTABLE_MODELS:
        return False
    if probe is None or model_id in _COUNTABLE_MODELS:
        return True
    try:
        await _counted(probe)
    except _UncountableError:
        UNCOUNTABLE_MODELS.add(model_id)
        return False
    _COUNTABLE_MODELS.add(model_id)
    return True


class _UncountableError(Exception):
    """The backend's counter does not count the model."""


async def _counted[T](exact: Callable[[], Awaitable[T]]) -> T:
    """Count with the model's own counter, never forwarding the counter's own wording.

    The model's validation of the request is answered as it words it, as a
    generation answers it.

    Args:
        exact: Counts the request.

    Returns:
        The answer.

    Raises:
        _UncountableError: When the counter does not count the model.
        ApiError: When the counter refuses the request in its own terms.
    """
    try:
        return await exact()
    except ClientError as exc:
        if uncountable(exc):
            raise _UncountableError from exc
        message = str((exc.response.get("Error") or {}).get("Message") or "")
        if not _COUNTER_WORDING.search(message):
            raise
        # The counter's own wording names internals: logged, not sent.
        log_error_details(message, status=400)
        raise ApiError(_REFUSED) from exc
    except (MantleApiUnsupportedError, MantleSurfaceUnsupportedError) as exc:
        raise _UncountableError from exc
    except MantleError as exc:
        if exc.status != HTTPStatus.BAD_REQUEST or not _COUNTER_WORDING.search(
            str(exc)
        ):
            raise
        log_error_details(str(exc), status=400)
        raise ApiError(_REFUSED) from exc


async def estimate_request_tokens(request: BaseModel, model_id: str) -> int:
    """Estimate a request's input tokens, above every tokenizer of the model's family measured.

    Args:
        request: The request.
        model_id: The model, whose tokenizer family selects the coefficients.

    Returns:
        The estimate.
    """
    coefficients = next(
        (table for pattern, table in _ESTIMATE_FAMILIES if pattern.search(model_id)),
        _ESTIMATE_DEFAULT,
    )
    media: list[int] = []
    content = await _plain_content(request, media, coefficients)
    if isinstance(content, dict):
        content.pop("model", None)
    tools = coefficients[10] if getattr(request, "tools", None) else 0
    return (
        ceil(coefficients[0] + tools)
        + sum(media)
        + estimate_text_tokens(to_json_str(content), coefficients)
    )


def estimate_text_tokens(
    text: str, coefficients: Sequence[float] = _ESTIMATE_DEFAULT
) -> int:
    """Estimate a text's tokens from the characters of each class, in linear time.

    Args:
        text: The text.
        coefficients: Tokens per character of each class, after the fixed tokens.

    Returns:
        The estimate, without the fixed tokens.
    """
    runs = "".join(
        run[0] for run in _RANDOM_RUN.finditer(text) if _is_random(run[0])
    ).encode()
    encoded = text.encode()
    letters = _byte_count(encoded, _LETTERS) - _byte_count(runs, _LETTERS)
    digits = _byte_count(encoded, _DIGITS) - _byte_count(runs, _DIGITS)
    spaces = _byte_count(encoded, _SPACE)
    counts = (
        letters,
        spaces,
        digits,
        _byte_count(encoded, _ASCII) - letters - digits - spaces - len(runs),
        len(runs),
        _byte_count(encoded, _TWO_BYTE_LEADS),
        _byte_count(encoded, _THREE_BYTE_LEADS),
        _byte_count(encoded, _CJK_LEADS),
        _byte_count(encoded, _FOUR_BYTE_LEADS),
    )
    return ceil(sumprod(counts, coefficients[1:10]))


def _byte_count(encoded: bytes, values: bytes) -> int:
    """Count the bytes of data among some values.

    Args:
        encoded: The data.
        values: The byte values.

    Returns:
        The count.
    """
    return len(encoded) - len(encoded.translate(None, values))


def _is_random(run: str) -> bool:
    """Whether a long run of ASCII word characters mixes letters and digits.

    Args:
        run: The run.

    Returns:
        True for identifiers, hashes and encoded data rather than words.
    """
    return any(char.isdigit() for char in run) and any(char.isalpha() for char in run)


async def _plain_content(
    value: object, media: list[int], coefficients: Sequence[float]
) -> Any:  # noqa: ANN401
    """Return request content as JSON data, its images and files replaced by their allowance.

    Args:
        value: The request, or part of it.
        media: Receives the tokens each image or file is estimated at.
        coefficients: The estimate coefficients, for text files.

    Returns:
        The content without media or signatures.
    """
    match value:
        case BaseModel():
            fields = {
                name: getattr(value, name)
                for name in type(value).model_fields
                if name not in _UNCOUNTED_KEYS
            } | (value.model_extra or {})
            return await _plain_content(fields, media, coefficients)
        case dict():
            if (tokens := await _media_tokens(value, coefficients)) is not None:
                media.append(tokens)
                return None
            return {
                str(key): await _plain_content(item, media, coefficients)
                for key, item in value.items()
                if item is not None
            }
        case list() | tuple():
            return [await _plain_content(item, media, coefficients) for item in value]
        case str() | int() | float() | bool() | None:
            return value
        case _:
            return str(value)


async def _media_tokens(
    block: dict[str, Any], coefficients: Sequence[float]
) -> int | None:
    """Estimate the tokens of an image or file block.

    Args:
        block: A content block's fields.
        coefficients: The estimate coefficients, for text files.

    Returns:
        The estimate, or None for any other block, or a document given as text.
    """
    match block.get("type"):
        case "image" | "input_image":
            return min(await _image_tokens(block), ceil(coefficients[11]))
        case "document" | "input_file":
            return await _file_tokens(block, coefficients)
        case _:
            return None


async def _image_tokens(block: Mapping[str, Any]) -> int:
    """Estimate an image's tokens from its pixels, or at the most any model takes.

    Args:
        block: An image content block's fields.

    Returns:
        The estimate.
    """
    data = await _inline_data(block)
    return _IMAGE_MAX_TOKENS if data is None else _pixel_tokens(data)


def _pixel_tokens(data: bytes) -> int:
    """Estimate an image's tokens from its pixels.

    Args:
        data: The image file.

    Returns:
        The estimate, the most any model takes for an unreadable image.
    """
    try:
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
    except OSError, Image.DecompressionBombError:
        return _IMAGE_MAX_TOKENS
    return min(
        _IMAGE_MAX_TOKENS, max(_IMAGE_MIN_TOKENS, width * height // _PIXELS_PER_TOKEN)
    )


async def _file_tokens(
    block: Mapping[str, Any], coefficients: Sequence[float]
) -> int | None:
    """Estimate a file's tokens: a PDF by its pages, a text file by its text.

    Args:
        block: A document or file content block's fields.
        coefficients: The estimate coefficients, for text files.

    Returns:
        The estimate, or None for a document given as text, counted as such.
    """
    source = block.get("source")
    if getattr(source, "type", None) in {"text", "content"}:
        return None
    if (data := await _inline_data(block)) is None:
        return await _referenced_file_tokens(block)
    return _data_file_tokens(data, coefficients)


def _data_file_tokens(data: bytes, coefficients: Sequence[float]) -> int:
    """Estimate the tokens of a file's content.

    Args:
        data: The content.
        coefficients: The estimate coefficients, for text files.

    Returns:
        A PDF's pages at their allowance, or a text file's estimate.
    """
    if data.startswith(b"%PDF"):
        return max(1, len(_PDF_PAGE.findall(data))) * _PDF_PAGE_TOKENS
    try:
        return estimate_text_tokens(data.decode(), coefficients)
    except UnicodeDecodeError:
        return reference_tokens(len(data))


def reference_tokens(size: int) -> int:
    """Estimate the tokens of a file known only by its size.

    Args:
        size: The size in bytes, zero when unknown.

    Returns:
        The estimate.
    """
    return max(_FILE_TOKENS, size * _FILE_TOKENS_PER_BYTE)


async def _referenced_file_tokens(block: Mapping[str, Any]) -> int:
    """Estimate the tokens of a file the request refers to, from its metadata only.

    Args:
        block: A document or file content block's fields.

    Returns:
        The estimate.
    """
    source = block.get("source")
    for value in (getattr(source, "data", None), getattr(source, "url", None)):
        if isinstance(value, InputFile):
            try:
                return reference_tokens(await value.get_size())
            except ApiError, ClientError, BotoCoreError, OSError:
                break
    return _FILE_TOKENS


async def _inline_data(block: Mapping[str, Any]) -> bytes | None:
    """Return the content an image or file block carries inline, never fetching it.

    Args:
        block: The block's fields.

    Returns:
        The content, or None for a block referring to it.
    """
    source = block.get("source")
    for value in (
        getattr(source, "data", None),
        block.get("image_url"),
        block.get("file_data"),
    ):
        if isinstance(value, InputFile):
            return await value.peek_inline()
        if isinstance(value, str) and value.startswith("data:") and ";base64," in value:
            try:
                return await b64decode(value.partition(";base64,")[2])
            except ValueError:
                return None
    return None


def _image_count(value: object) -> int:
    """Count the image blocks in request content.

    Args:
        value: The request, or part of it.

    Returns:
        The number of images.
    """
    match value:
        case BaseModel():
            if getattr(value, "type", None) in {"image", "input_image"}:
                return 1
            return sum(
                _image_count(getattr(value, name)) for name in type(value).model_fields
            )
        case dict():
            if value.get("type") in {"image", "input_image"}:
                return 1
            return sum(_image_count(item) for item in value.values())
        case list() | tuple():
            return sum(_image_count(item) for item in value)
        case _:
            return 0


@dataclass(slots=True)
class CallBudget:
    """Counter calls one request may spend counting past the context window."""

    #: Counter calls left.
    calls: int = field(default_factory=lambda: _MAX_CALLS)
    #: Bounds the calls in flight.
    semaphore: Semaphore = field(default_factory=lambda: Semaphore(_CONCURRENCY))


async def countable_media(
    request: ConverseTokensRequestTypeDef, region: RegionName
) -> int:
    """Prepare a Converse input's media for the counter, returning the tokens it cannot count.

    The counter takes inline images only: documents, and files the request only
    refers to, are removed and estimated instead. A referenced file is never
    downloaded, only its size is read.

    Args:
        request: The Converse input, changed in place.
        region: Target AWS region.

    Returns:
        The estimated tokens of the removed blocks.
    """
    tokens = sum(
        _CLAUDE_IMAGE_TOKENS if kind == "image" else reference_tokens(size)
        for kind, size in await resolve_inline_bedrock_content_blocks(region)
    )
    for message in request["messages"]:
        content = list(message["content"])
        tokens += _strip_uncountable(content)
        message["content"] = content or [{"text": "."}]  # type: ignore[arg-type]
    return tokens


def _strip_uncountable(content: list[Any]) -> int:
    """Remove the blocks the counter refuses from a content list, tool results included.

    Args:
        content: The content blocks, changed in place.

    Returns:
        The estimated tokens of the removed documents; referenced files are
        estimated by the caller.
    """
    tokens = 0
    kept = []
    for block in content:
        if "toolResult" in block:
            inner = list(block["toolResult"].get("content", ()))
            tokens += _strip_uncountable(inner)
            block["toolResult"]["content"] = inner or [{"text": "."}]
        media = next(
            (block[key] for key in ("image", "document", "video") if key in block), None
        )
        if media is None:
            kept.append(block)
        elif data := media.get("source", {}).get("bytes"):
            if "image" in block:
                kept.append(block)
            else:
                tokens += _data_file_tokens(data, _ESTIMATE_CLAUDE)
    content[:] = kept
    return tokens


async def count_converse_tokens(
    client: BedrockRuntimeClient,
    model_id: str,
    request: ConverseTokensRequestTypeDef,
    *,
    past_window: bool = True,
    budget: CallBudget | None = None,
) -> int:
    """Count a Converse request's input tokens.

    An input the counter refuses as larger than the context window is counted
    in pieces it accepts, less the fixed tokens each extra piece adds. A piece
    that cannot be split further, or left once the calls run out, counts as the
    larger of the counter's stated figure and a local estimate erring high, and
    the total is never below the window.

    Args:
        client: The Bedrock Runtime client.
        model_id: The model.
        request: The Converse input.
        past_window: Count an input over the context window instead of refusing it.
        budget: The request's call budget, shared by all its counts.

    Returns:
        The input tokens.

    Raises:
        ContextLengthExceededError: When the input is over the context window
            and ``past_window`` is false.
    """

    async def count(converse: ConverseTokensRequestTypeDef) -> int:
        """Count one request."""
        response = await client.count_tokens(
            modelId=model_id, input={"converse": converse}
        )
        return response["inputTokens"]

    with handle_bedrock_client_error():
        if not past_window:
            return await count(request)
        try:
            return await count(request)
        except ClientError as exc:
            if (overflow := context_overflow(exc)) is None:
                raise
        return await _count_in_pieces(count, request, overflow, budget or CallBudget())


async def _count_in_pieces(
    count: _Counter,
    request: ConverseTokensRequestTypeDef,
    overflow: ContextOverflow,
    budget: CallBudget,
) -> int:
    """Count an input over the context window as the sum of pieces the counter accepts.

    Args:
        count: Counts one request.
        request: The refused request.
        overflow: Its refusal.
        budget: The request's call budget.

    Returns:
        The input tokens, above the window.
    """
    floor = max(overflow.observed or 0, (overflow.limit or 0) + 1)
    pieces = _Pieces(
        count,
        {key: value for key, value in request.items() if key != "messages"},
        budget,
    )
    try:
        pieces.lead = await pieces.call([_LEAD])
    except ClientError as exc:
        if context_overflow(exc) is None:
            raise
        # The system prompt and tools alone overflow: no piece can be smaller.
        return max(floor, _converse_estimate(request))
    messages: list[MessageTypeDef] = list(request["messages"])  # type: ignore[arg-type]
    # The whole was just refused: splitting starts at once.
    return max(floor, await pieces.split(messages, overflow, first=True))


@dataclass(slots=True)
class _Pieces:
    """Counts messages piece by piece, each piece sent with the request's other fields."""

    #: Counts one request.
    count: _Counter
    #: The request without its messages.
    frame: dict[str, Any]
    #: The request's call budget.
    budget: CallBudget
    #: Tokens of the frame and the lead message alone.
    lead: int = 0

    async def call(self, messages: list[MessageTypeDef]) -> int:
        """Count the frame with some messages.

        Args:
            messages: The messages.

        Returns:
            The counter's answer.
        """
        self.budget.calls -= 1
        async with self.budget.semaphore:
            return await self.count(self.frame | {"messages": messages})  # type: ignore[arg-type]

    async def tokens(self, messages: list[MessageTypeDef], *, first: bool) -> int:
        """Return the tokens messages add to the count, splitting them while refused.

        Args:
            messages: The messages.
            first: Whether they start the conversation, so no lead is needed.

        Returns:
            The tokens.
        """
        if self.budget.calls <= 0:
            return _converse_estimate(messages)
        try:
            counted = await self.call(messages if first else [_LEAD, *messages])
        except ClientError as exc:
            if (overflow := context_overflow(exc)) is None:
                raise
            return await self.split(messages, overflow, first=first)
        return counted if first else counted - self.lead

    async def split(
        self, messages: list[MessageTypeDef], overflow: ContextOverflow, *, first: bool
    ) -> int:
        """Return the tokens of refused messages, counted in two halves.

        Args:
            messages: The refused messages.
            overflow: Their refusal.
            first: Whether they start the conversation.

        Returns:
            The tokens; an estimate erring high where no split is left.
        """
        if self.budget.calls < 2 or (halves := _halves(messages)) is None:
            stated = (overflow.observed or 0) - (0 if first else self.lead)
            return max(stated, _converse_estimate(messages))
        try:
            async with TaskGroup() as group:
                left = group.create_task(self.tokens(halves[0], first=first))
                right = group.create_task(self.tokens(halves[1], first=False))
        except ExceptionGroup as error:
            # A failed half cancels the other; its own error reaches the caller.
            raise error.exceptions[0] from None
        return left.result() + right.result()


def _converse_estimate(value: object) -> int:
    """Estimate Converse content's tokens erring high, as the Claude models count.

    Args:
        value: Converse messages, or a whole Converse input.

    Returns:
        The estimate.
    """
    media: list[int] = []
    text = to_json_str(_converse_text(value, media))
    return sum(media) + estimate_text_tokens(text, _ESTIMATE_CLAUDE)


def _converse_text(value: object, media: list[int]) -> Any:  # noqa: ANN401
    """Return Converse content as JSON data, its images and files replaced by their allowance.

    Args:
        value: Converse content.
        media: Receives the tokens each image or file is estimated at.

    Returns:
        The content without media or raw bytes.
    """
    match value:
        case {"image": {"source": {"bytes": bytes() as data}}}:
            media.append(min(_pixel_tokens(data), ceil(_ESTIMATE_CLAUDE[11])))
            return None
        case {"document": {"source": {"bytes": bytes() as data}}}:
            media.append(_data_file_tokens(data, _ESTIMATE_CLAUDE))
            return None
        case dict():
            return {
                str(key): _converse_text(item, media) for key, item in value.items()
            }
        case list() | tuple():
            return [_converse_text(item, media) for item in value]
        case bytes():
            return None
        case _:
            return value


def _halves(
    messages: list[MessageTypeDef],
) -> tuple[list[MessageTypeDef], list[MessageTypeDef]] | None:
    """Split messages in two halves of about the same size the counter accepts apart.

    A split between messages never parts a tool result from its call; where
    none is possible, the messages' content is split as plain user content.

    Args:
        messages: The messages.

    Returns:
        The halves, or None when the content is a single block that cannot be split.
    """
    cut = _balanced_cut(
        messages,
        [index for index in range(1, len(messages)) if _starts_piece(messages[index])],
    )
    if cut is not None:
        return messages[:cut], messages[cut:]
    blocks = [
        plain
        for message in messages
        for block in message["content"]
        for plain in _plain(block)
    ]
    if (cut := _balanced_cut(blocks, range(1, len(blocks)))) is not None:
        return _as_user(blocks[:cut]), _as_user(blocks[cut:])
    text = blocks[0].get("text") if blocks else None
    if not isinstance(text, str) or len(text) < 2:
        return None
    middle = len(text) // 2
    cut = max(text.rfind(" ", 0, middle), text.rfind("\n", 0, middle)) + 1
    if not text[:cut].strip():
        cut = middle
    return _as_user([{"text": text[:cut]}]), _as_user([{"text": text[cut:]}])


def _balanced_cut(items: Sequence[object], cuts: Iterable[int]) -> int | None:
    """Pick the cut that parts items into two halves of the closest size.

    Args:
        items: The items.
        cuts: The indexes a cut may be made before.

    Returns:
        The index, or None when no cut is allowed.
    """
    prefix = list(
        accumulate((_converse_estimate(item) + 1 for item in items), initial=0)
    )
    return min(
        cuts, key=lambda index: abs(2 * prefix[index] - prefix[-1]), default=None
    )


def _starts_piece(message: MessageTypeDef) -> bool:
    """Whether a piece may start at this message without parting a tool result from its call.

    Args:
        message: The message.

    Returns:
        True for an assistant message, or a user message carrying no tool result.
    """
    return message["role"] == "assistant" or not any(
        "toolResult" in block for block in message["content"]
    )


def _plain(block: Mapping[str, Any]) -> list[Any]:
    """Return a content block as blocks a user message holds without any pairing.

    Args:
        block: The block.

    Returns:
        The block itself when it is text or media, the content of a tool
        result, the text of a reasoning block, nothing for a cache point, and
        the JSON of anything else.
    """
    if block.keys() & _PLAIN_BLOCKS:
        return [block]
    if "toolResult" in block:
        return [
            item if item.keys() & _PLAIN_BLOCKS else {"text": to_json_str(item)}
            for item in block["toolResult"].get("content", ())
        ]
    if "reasoningContent" in block:
        text = block["reasoningContent"].get("reasoningText", {}).get("text")
        return [{"text": text}] if text else []
    if "cachePoint" in block:
        return []
    return [{"text": to_json_str(block)}]


def _as_user(blocks: list[ContentBlockTypeDef]) -> list[MessageTypeDef]:
    """Wrap content blocks in a single user message.

    Args:
        blocks: The blocks.

    Returns:
        The message, alone in a list.
    """
    return [{"role": "user", "content": blocks}]
