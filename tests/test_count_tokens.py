"""Token counting past the context window and with server tools (no AWS calls).

A fake counter stands in for Bedrock CountTokens: it enforces the rules the
real one was measured to apply (a tool result must follow its call, a tool call
must be answered in the next message), refuses a request over its window with
an understated figure as the real one does, and otherwise counts additively,
so the exact size of any conversation is known.

Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html
     https://platform.claude.com/docs/en/api/messages/count_tokens
     stdapi/models/chat/_adapters/_count_tokens.py
"""

from __future__ import annotations

from asyncio import gather, sleep
from json import loads
from math import ceil
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock_mantle import MantleApiUnsupportedError, MantleError
from stdapi.input_file import InputFile
from stdapi.models.chat._adapters import _count_tokens
from stdapi.models.chat._adapters._count_tokens import (
    PROXY_MODEL_ID,
    CallBudget,
    count_converse_tokens,
    count_or_approximate,
    countable_media,
    estimate_request_tokens,
    plain_server_tool_history,
    server_tool_tokens,
    stub_server_tools,
)
from stdapi.models.chat._adapters._responses_context import ContextLengthExceededError
from stdapi.types.anthropic_messages import MessageCountTokensParams, MessageParam
from stdapi.types.openai_responses import InputTokenCountParams
from stdapi.utils import b64encode
from tests._helpers import make_client_error, make_model_details, red_png_b64
from tests.conftest import REPO_ROOT

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from stdapi.models import ModelDetails

pytestmark = pytest.mark.local

#: The fake counter's context window.
_WINDOW = 1000

#: Tokens the fake counter adds once per request.
_FIXED = 10

#: Tokens the fake counter adds per message.
_PER_MESSAGE = 3


def _text_tokens(text: str) -> int:
    """Return the fake count of a text: a token per four characters, as English runs."""
    return ceil(len(text) / 4)


def _block_tokens(block: dict[str, Any]) -> int:
    """Return the fake count of one content block."""
    if "text" in block:
        return _text_tokens(block["text"])
    if "toolUse" in block:
        return 5 + _text_tokens(str(block["toolUse"]["input"]))
    if "toolResult" in block:
        return 5 + sum(_block_tokens(item) for item in block["toolResult"]["content"])
    if "image" in block:
        return len(block["image"]["source"]["bytes"])
    return 0


def _tokens(request: dict[str, Any]) -> int:
    """Return the fake count of a request."""
    return (
        _FIXED
        + sum(_text_tokens(block["text"]) for block in request.get("system", ()))
        + sum(
            _PER_MESSAGE + sum(_block_tokens(block) for block in message["content"])
            for message in request["messages"]
        )
    )


class _FakeCounter:
    """Counts like Bedrock CountTokens, refusing what it refuses."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    async def count_tokens(
        self,
        *,
        modelId: str,  # noqa: N803 (mirrors the boto3 client's camelCase kwarg)
        input: dict[str, Any],  # noqa: A002
    ) -> dict[str, int]:
        request = input["converse"]
        self.requests.append(request)
        messages = request["messages"]
        for index, message in enumerate(messages):
            results = {
                block["toolResult"]["toolUseId"]
                for block in message["content"]
                if "toolResult" in block
            }
            calls = {
                block["toolUse"]["toolUseId"]
                for block in (messages[index - 1]["content"] if index else ())
                if "toolUse" in block
            }
            if results != calls and message["role"] == "user":
                unpaired = make_client_error(
                    "ValidationException",
                    "CountTokens",
                    message=f"Expected toolResult blocks at messages.{index}.content",
                )
                raise unpaired
        tokens = _tokens(request)
        if tokens > _WINDOW:
            # The real counter stops early: its figure understates the input.
            stated = min(tokens, _WINDOW + 37)
            too_long = make_client_error(
                "ValidationException",
                "CountTokens",
                message=f"prompt is too long: {stated} tokens > {_WINDOW} maximum",
            )
            raise too_long
        return {"inputTokens": tokens}


def _user(text: str) -> dict[str, Any]:
    """Return a user text message."""
    return {"role": "user", "content": [{"text": text}]}


def _round_trip(index: int, result: str) -> list[dict[str, Any]]:
    """Return a tool call and its result."""
    tool_id = f"t{index}"
    return [
        {
            "role": "assistant",
            "content": [
                {"text": "reading"},
                {"toolUse": {"toolUseId": tool_id, "name": "read", "input": {}}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"toolResult": {"toolUseId": tool_id, "content": [{"text": result}]}}
            ],
        },
    ]


async def _count(request: dict[str, Any], **kwargs: bool) -> tuple[int, _FakeCounter]:
    """Count a request with the fake counter."""
    client = _FakeCounter()
    tokens = await count_converse_tokens(client, "model", request, **kwargs)  # type: ignore[arg-type]
    return tokens, client


async def test_an_input_the_window_holds_is_counted_in_one_call() -> None:
    """An input under the window is the counter's own answer, in a single call.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:count_converse_tokens
    """
    request = {"messages": [_user("hello")]}
    tokens, client = await _count(request)
    assert tokens == _tokens(request)
    assert len(client.requests) == 1


async def test_an_agent_conversation_over_the_window_is_the_sum_of_its_pieces() -> None:
    """Tool round trips over the window are counted exactly, never parted.

    The fake counter refuses any piece that parts a tool result from its call,
    so an exact total proves every piece kept them together.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:_halves
    """
    messages = [_user("start")]
    for index in range(6):
        messages += _round_trip(index, "x" * 1600)
    request = {"system": [{"text": "be brief"}], "messages": messages}
    tokens, client = await _count(request)
    assert _tokens(request) > _WINDOW
    assert tokens == _tokens(request)
    assert all(piece["system"] == request["system"] for piece in client.requests)
    assert all(piece["messages"][0]["role"] == "user" for piece in client.requests)


@pytest.mark.parametrize(
    "messages",
    [
        pytest.param([_user("word " * 2800)], id="one text"),
        pytest.param([_user("hi"), *_round_trip(0, "word " * 2800)], id="tool result"),
    ],
)
async def test_a_message_over_the_window_is_split_within(
    messages: list[dict[str, Any]],
) -> None:
    """A single message larger than the window is split inside, not refused.

    Its pieces carry a few tokens of framing the whole does not, and a tool
    call and result become plain text; the total stays within 2% of the size.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:_plain
    """
    request = {"messages": messages}
    tokens, _ = await _count(request)
    assert tokens > _WINDOW
    assert abs(tokens - _tokens(request)) <= _tokens(request) * 0.02


async def test_a_split_message_keeps_its_readable_content() -> None:
    """Split inside, a round trip keeps its reasoning and data, never its cache marks.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:_plain
    """
    request = {
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {"reasoningText": {"text": "think"}}},
                    {"reasoningContent": {"redactedContent": b"opaque"}},
                    {"toolUse": {"toolUseId": "t0", "name": "read", "input": {}}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "t0",
                            "content": [{"text": "word " * 1200}, {"json": {"k": 1}}],
                        }
                    },
                    {"text": "word " * 1200},
                    {"cachePoint": {"type": "default"}},
                ],
            },
        ]
    }
    tokens, client = await _count(request)
    assert tokens > _WINDOW
    split = [
        piece["messages"]
        for piece in client.requests
        if all(message["role"] == "user" for message in piece["messages"])
    ]
    blocks = [
        block for piece in split for message in piece for block in message["content"]
    ]
    assert {"text": "think"} in blocks
    assert {"text": '{"json":{"k":1}}'} in blocks
    assert not any("cachePoint" in block or "toolUse" in block for block in blocks)


async def test_an_unsplittable_block_is_estimated_high() -> None:
    """A single block no split can shrink is estimated, never at the understated figure.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:_Pieces.split
    """
    image = {"image": {"format": "png", "source": {"bytes": b"\0" * 2000}}}
    request = {"messages": [{"role": "user", "content": [image]}]}
    tokens, _ = await _count(request)
    assert tokens >= _tokens(request) > _WINDOW + 37


async def test_a_system_prompt_over_the_window_counts_the_stated_figure() -> None:
    """When the fields sent with every piece overflow alone, nothing is split.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:_count_in_pieces
    """
    request = {"system": [{"text": "s" * 8000}], "messages": [_user("hi")]}
    tokens, client = await _count(request)
    assert tokens >= _tokens(request)
    assert len(client.requests) == 2, "the refused request, then the frame alone"


async def test_the_count_is_never_below_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the calls run out, what is left is estimated high, never below the input.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:_Pieces.tokens
    """
    monkeypatch.setattr(_count_tokens, "_MAX_CALLS", 2)
    request = {"messages": [_user("a " * 8000)]}
    tokens, _ = await _count(request)
    assert tokens >= _tokens(request)


@pytest.mark.parametrize("windows", [3, 30])
async def test_an_input_many_windows_long_is_never_counted_low(
    monkeypatch: pytest.MonkeyPatch, windows: int
) -> None:
    """However far past the window, the answer stays above the input's size.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:_Pieces.tokens
    """
    monkeypatch.setattr(_count_tokens, "_MAX_CALLS", 8)
    messages = [_user("start")]
    for index in range(windows * 3):
        messages += _round_trip(index, "word " * 300)
    request = {"messages": messages}
    tokens, _ = await _count(request)
    assert tokens >= _tokens(request) > windows * _WINDOW


async def test_the_calls_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """An input many times the window stops splitting once its calls run out.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:_MAX_CALLS
    """
    monkeypatch.setattr(_count_tokens, "_MAX_CALLS", 8)
    _, client = await _count({"messages": [_user("word " * 80_000)]})
    assert len(client.requests) <= 8 + 2


async def test_strict_counting_refuses_an_input_over_the_window() -> None:
    """Without ``past_window``, the refusal is raised for truncation to act on.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:count_converse_tokens
    """
    with pytest.raises(ContextLengthExceededError):
        await _count({"messages": [_user("a" * 8000)]}, past_window=False)


async def test_another_refusal_is_raised() -> None:
    """A refusal that is not about the window is never split around.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:count_converse_tokens
    """
    with pytest.raises(ClientError, match="Expected toolResult"):
        await _count({"messages": _round_trip(0, "x")[1:]})


@pytest.mark.parametrize(
    ("tool_type", "tokens"),
    [
        ("web_search_20250305", 1678),
        ("web_fetch_20260209", 2925),
        ("tool_search_tool_bm25", 144),
        ("web_search_20990101", 3844),
        ("custom", None),
        (None, None),
        ("bash_20250124", None),
    ],
)
def test_server_tool_tokens(tool_type: str | None, tokens: int | None) -> None:
    """Each server tool type has its definition's tokens; an unknown version the latest's.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:server_tool_tokens
    """
    assert server_tool_tokens(tool_type) == tokens


def test_server_tools_are_replaced_by_stubs() -> None:
    """Server tools become stubs of their names; other tools stay as they are.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:stub_server_tools
    """
    function = {"name": "web_search", "input_schema": {"type": "object"}}
    tools: list[Any] = [
        function,
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 2},
        {"type": "bash_20250124", "name": "bash"},
    ]
    assert stub_server_tools(tools) == 1678
    assert tools == [
        function,
        {
            "name": "web_search",
            "description": "web_search",
            "input_schema": {"type": "object"},
        },
        {"type": "bash_20250124", "name": "bash"},
    ]


def _replayed(*content: dict[str, Any]) -> list[MessageParam]:
    """Return a conversation replaying an assistant turn with *content*."""
    return [
        MessageParam.model_validate({"role": "user", "content": "Search."}),
        MessageParam.model_validate({"role": "assistant", "content": list(content)}),
        MessageParam.model_validate({"role": "user", "content": "Thanks."}),
    ]


#: A replayed web search call.
_SEARCH_CALL = {
    "type": "server_tool_use",
    "id": "srvtoolu_01",
    "name": "web_search",
    "input": {"query": "python"},
}


def test_a_replayed_web_search_is_counted_as_text_plus_its_pages() -> None:
    """A search call and result become text; each page adds what its size implies.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:plain_server_tool_history
    """
    result = {
        "type": "web_search_tool_result",
        "tool_use_id": "srvtoolu_01",
        "content": [
            {
                "type": "web_search_result",
                "title": "Python",
                "url": "https://python.org",
                "encrypted_content": "x" * 1000,
            }
        ],
    }
    cited = {
        "type": "text",
        "text": "Python is a language.",
        "citations": [
            {
                "type": "web_search_result_location",
                "cited_text": "Python",
                "encrypted_index": "abc",
                "title": "Python",
                "url": "https://python.org",
            }
        ],
    }
    messages, tokens = plain_server_tool_history(_replayed(_SEARCH_CALL, result, cited))
    content = messages[1].content
    assert isinstance(content, list)
    assert [block.type for block in content] == ["text", "text", "text"]
    assert [getattr(block, "text", None) for block in content] == [
        '{"name":"web_search","input":{"query":"python"}}',
        "Python\nhttps://python.org",
        "Python is a language.",
    ]
    assert content[2].citations is None  # type: ignore[union-attr]
    assert tokens == 31 + 12 + 296 + 16
    assert messages[0].content == "Search."


def test_another_server_tool_result_is_counted_as_its_json() -> None:
    """Any other server tool result is its content's JSON, plus a fixed framing.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:_plain_history_block
    """
    result = {
        "type": "bash_code_execution_tool_result",
        "tool_use_id": "srvtoolu_01",
        "content": {
            "type": "bash_code_execution_result",
            "stdout": "4",
            "stderr": "",
            "return_code": 0,
            "content": [],
        },
    }
    messages, tokens = plain_server_tool_history(_replayed(_SEARCH_CALL, result))
    content = messages[1].content
    assert isinstance(content, list)
    assert '"stdout":"4"' in getattr(content[1], "text", "")
    assert tokens == 31


def test_a_history_without_server_tools_is_left_as_is() -> None:
    """Messages carrying no server tool block are returned unchanged.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:plain_server_tool_history
    """
    original = _replayed({"type": "text", "text": "hello"})
    messages, tokens = plain_server_tool_history(original)
    assert tokens == 0
    assert all(new is old for new, old in zip(messages, original, strict=True))


#: Every tokenizer's count of each content sample, measured 2026-09-23.
_ESTIMATE_SAMPLES: dict[str, Any] = loads(
    (REPO_ROOT / "tests/fixtures/token_estimates.json").read_text()
)


#: Model the estimate is made for, per reference tokenizer the fixture names.
_GATEWAY_MODEL = {
    "gpt-4o-mini": "openai.gpt-6-luna",
    **{
        f"claude-{name}": f"anthropic.claude-{name}"
        for name in (
            "fable-5",
            "opus-4-7",
            "opus-4-8",
            "opus-5",
            "opus-5-5",
            "sonnet-5",
        )
    },
    "claude-haiku-4-5-20251001": "anthropic.claude-haiku-4-5",
}


@pytest.mark.parametrize("sample", sorted(_ESTIMATE_SAMPLES))
async def test_the_local_estimate_is_above_every_tokenizer_measured(
    sample: str,
) -> None:
    """The estimate is never below the count of the model it is made for.

    Each sample was counted by seven Claude models on the Anthropic API,
    ``gpt-4o-mini`` on the OpenAI API, and the prompt tokens nineteen Bedrock
    models billed for it; each estimate uses the coefficients of the
    reference's tokenizer family, on both routes.

    Ref: https://platform.claude.com/docs/en/api/messages/count_tokens
         https://developers.openai.com/api/reference/resources/responses/subresources/input_tokens/methods/count
         stdapi/models/chat/_adapters/_count_tokens.py:estimate_request_tokens
    """
    recorded = _ESTIMATE_SAMPLES[sample]
    anthropic = MessageCountTokensParams.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": recorded["text"]}]}
    )
    openai = InputTokenCountParams.model_validate(
        {"model": "m", "input": recorded["text"]}
    )
    for reference, tokens in recorded["tokens"].items():
        model = _GATEWAY_MODEL.get(reference, reference)
        assert await estimate_request_tokens(anthropic, model) >= tokens, reference
        assert await estimate_request_tokens(openai, model) >= tokens, reference


async def test_an_unknown_family_gets_the_envelope_of_all() -> None:
    """A model of no measured family is estimated above every family's estimate.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:_ESTIMATE_DEFAULT
    """
    recorded = _ESTIMATE_SAMPLES["prose"]
    request = MessageCountTokensParams.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": recorded["text"]}]}
    )
    unknown = await estimate_request_tokens(request, "vendor.unmeasured-model")
    for reference in recorded["tokens"]:
        if not reference.startswith("claude-"):
            model = _GATEWAY_MODEL.get(reference, reference)
            assert unknown >= await estimate_request_tokens(request, model)


async def test_an_image_is_estimated_by_its_pixels_or_at_the_most() -> None:
    """An inline image counts its pixels, at least a floor; one by reference the most.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:_image_tokens
    """

    def request(source: dict[str, Any]) -> MessageCountTokensParams:
        return MessageCountTokensParams.model_validate(
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": [{"type": "image", "source": source}]}
                ],
            }
        )

    inline = await estimate_request_tokens(
        request({"type": "base64", "media_type": "image/png", "data": red_png_b64()}),
        "amazon.nova-lite-v1:0",
    )
    by_url = await estimate_request_tokens(
        request({"type": "url", "url": "https://example.com/a.png"}),
        "amazon.nova-lite-v1:0",
    )
    assert 2700 <= inline < 3000, "a one-pixel image counts the floor"
    assert by_url >= 16500


async def test_a_pdf_is_estimated_by_its_pages() -> None:
    """A PDF counts a fixed allowance per page, not its bytes.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:_file_tokens
    """
    pdf = b"%PDF-1.4\n" + b"<< /Type /Page >>\n" * 3 + b"<< /Type /Pages >>\n"
    request = MessageCountTokensParams.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": await b64encode(pdf),
                            },
                        }
                    ],
                }
            ],
        }
    )
    assert (
        9000 <= await estimate_request_tokens(request, "amazon.nova-lite-v1:0") < 9200
    )


class _Counted:
    """Records which counting path answered."""

    def __init__(self, exact: Exception | int, proxy: Exception | int = 1000) -> None:
        self.exact_result = exact
        self.proxy_result = proxy
        self.calls: list[str] = []

    async def exact(self) -> int:
        self.calls.append("exact")
        if isinstance(self.exact_result, Exception):
            raise self.exact_result
        return self.exact_result

    async def proxy(self, counter: ModelDetails) -> int:
        self.calls.append(f"proxy:{counter.id}")
        if isinstance(self.proxy_result, Exception):
            raise self.proxy_result
        return self.proxy_result


#: A short request, for the paths that do not read its content.
_REQUEST = MessageCountTokensParams.model_validate(
    {"model": "m", "messages": [{"role": "user", "content": "Hello"}]}
)

#: The countable Claude model standing for the others.
_PROXY = make_model_details(PROXY_MODEL_ID)


@pytest.fixture
def catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the proxy model and forget which models were counted or refused."""
    monkeypatch.setattr(_count_tokens, "catalog_model", lambda _id: _PROXY)
    monkeypatch.setattr(_count_tokens, "UNCOUNTABLE_MODELS", set())
    monkeypatch.setattr(_count_tokens, "_COUNTABLE_MODELS", set())


@pytest.mark.usefixtures("catalog")
async def test_a_trivial_request_learns_whether_the_counter_serves_the_model() -> None:
    """The counter is probed once per model, so the request itself is read once.

    A refused probe sends the request to the fallback without counting it;
    an accepted one is remembered, and later counts skip it.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:count_or_approximate
    """
    probes: list[str] = []

    async def refused() -> int:
        probes.append("refused")
        raise _UNCOUNTABLE

    async def accepted() -> int:
        probes.append("accepted")
        return 1

    refused_model = _Counted(exact=42)
    tokens = await count_or_approximate(
        make_model_details("amazon.nova-micro-v1:0"),
        _REQUEST,
        refused_model.exact,
        refused_model.proxy,
        int,
        lambda tokens, scale: scale(tokens),
        refused,
    )
    assert tokens == await estimate_request_tokens(_REQUEST, "amazon.nova-micro-v1:0")
    assert not refused_model.calls, "the request itself was never counted"
    counted = _Counted(exact=42)
    for _ in range(2):
        assert (
            await count_or_approximate(
                make_model_details("anthropic.claude-sonnet-4-6"),
                _REQUEST,
                counted.exact,
                counted.proxy,
                int,
                lambda tokens, scale: scale(tokens),
                accepted,
            )
            == 42
        )
    assert probes == ["refused", "accepted"]


async def _approximate(
    model_id: str,
    counted: _Counted,
    *,
    exact: bool = True,
    request: MessageCountTokensParams = _REQUEST,
    probe: Callable[[], Awaitable[int]] | None = None,
) -> int:
    """Count through ``count_or_approximate`` with int answers."""
    return await count_or_approximate(
        make_model_details(model_id),
        request,
        counted.exact if exact else None,
        counted.proxy,
        int,
        lambda tokens, scale: scale(tokens),
        probe,
    )


#: The counter's refusal of a model it does not count.
_UNCOUNTABLE = make_client_error(
    "ValidationException",
    "CountTokens",
    message="The provided model doesn't support counting tokens.",
)


@pytest.mark.usefixtures("catalog")
async def test_a_countable_model_is_counted_exactly() -> None:
    """The model's own counter answers when it counts the model.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:count_or_approximate
    """
    counted = _Counted(exact=42)
    assert await _approximate("anthropic.claude-sonnet-4-6", counted) == 42
    assert counted.calls == ["exact"]


async def _refused() -> int:
    """Refuse the trivial request, as the counter refuses a model it does not count."""
    raise _UNCOUNTABLE


@pytest.mark.usefixtures("catalog")
@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-opus-5-5",
        "anthropic.claude-opus-4-7",
        "us.anthropic.claude-opus-4-7-v1:0",
    ],
)
async def test_a_newer_claude_is_counted_by_the_proxy_scaled_up(model_id: str) -> None:
    """Claude 4.7+ counts through the proxy times the ratio, never below the estimate.

    The trivial request's refusal is remembered, so the next count skips the
    counter; an image adds what the newer model's high-resolution vision may take.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:count_or_approximate
    """
    counted = _Counted(exact=42, proxy=1000)
    for _ in range(2):
        assert await _approximate(model_id, counted, probe=_refused) == 1520
    assert counted.calls == [f"proxy:{PROXY_MODEL_ID}"] * 2
    with_image = MessageCountTokensParams.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {"type": "url", "url": "https://x/a.png"},
                        }
                    ],
                }
            ],
        }
    )
    tokens = await _approximate(model_id, counted, request=with_image)
    estimate = await estimate_request_tokens(with_image, model_id)
    assert tokens == max(1520 + 2400, estimate)


@pytest.mark.parametrize(
    ("model_id", "scaled"),
    [
        ("anthropic.claude-sonnet-4-6", False),
        ("anthropic.claude-opus-4-1-20250805-v1:0", False),
        ("anthropic.claude-sonnet-4-20250514-v1:0", False),
        ("us.anthropic.claude-haiku-4-5-20251001-v1:0", False),
        ("anthropic.claude-opus-4-7", True),
        ("us.anthropic.claude-opus-4-7-v1:0", True),
        ("anthropic.claude-fable-5", True),
    ],
)
def test_the_proxy_tokenizer_boundary_is_claude_4_6(
    model_id: str, scaled: bool
) -> None:
    """Claude up to 4.6 shares the proxy's tokenizer; 4.7 and later are scaled.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:_PROXY_TOKENIZER
    """
    assert bool(_count_tokens._PROXY_TOKENIZER.search(model_id)) is not scaled  # noqa: SLF001


@pytest.mark.parametrize("sample", sorted(_ESTIMATE_SAMPLES))
async def test_a_newer_claude_is_never_counted_below_upstream(sample: str) -> None:
    """On every fixture sample, the Claude 4.7+ answer is above every 4.7+ model's count.

    The proxy's count is taken as Claude Haiku 4.5's own, which the backend
    counter runs slightly above; the answer is the scaled count or the
    character estimate, whichever is higher.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:count_or_approximate
    """
    recorded = _ESTIMATE_SAMPLES[sample]
    tokens = recorded["tokens"]
    request = MessageCountTokensParams.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": recorded["text"]}]}
    )
    for reference, count in tokens.items():
        if reference.startswith("claude-") and "haiku" not in reference:
            model_id = f"anthropic.{reference}"
            answer = max(
                ceil(tokens["claude-haiku-4-5-20251001"] * _count_tokens._PROXY_RATIO),  # noqa: SLF001
                await estimate_request_tokens(request, model_id),
            )
            assert answer >= count, reference


@pytest.mark.usefixtures("catalog")
async def test_a_caller_s_request_never_marks_a_model_uncountable() -> None:
    """Only the trivial request decides a model is uncountable, never a caller's own.

    A request the counter refuses as if the model were uncountable, after the
    trivial request was counted, is refused for that caller alone.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:_serves
    """
    hostile = _Counted(
        exact=MantleApiUnsupportedError("does not support the 'x' API", status=400)
    )
    with pytest.raises(ApiError) as excinfo:
        await _approximate("anthropic.claude-sonnet-4-6", hostile, probe=hostile_probe)
    assert excinfo.value.status == 400
    assert "does not support" not in str(excinfo.value)
    assert "anthropic.claude-sonnet-4-6" not in _count_tokens.UNCOUNTABLE_MODELS
    honest = _Counted(exact=42)
    assert await _approximate("anthropic.claude-sonnet-4-6", honest) == 42


async def hostile_probe() -> int:
    """Count the trivial request, as the counter counts a model it serves."""
    return 1


@pytest.mark.usefixtures("catalog")
async def test_a_claude_sharing_the_proxy_tokenizer_is_counted_unscaled() -> None:
    """A Claude 4.6-or-older model no counter serves directly counts as the proxy does.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:_PROXY_TOKENIZER
    """
    counted = _Counted(exact=0, proxy=1000)
    assert (
        await _approximate("anthropic.claude-haiku-4-5", counted, exact=False) == 1000
    )


@pytest.mark.usefixtures("catalog")
@pytest.mark.parametrize(
    "refusal",
    [_UNCOUNTABLE, MantleApiUnsupportedError("no", status=400)],
    ids=["counter", "endpoint"],
)
async def test_another_model_is_estimated(refusal: Exception) -> None:
    """A model neither its counter nor a Claude counter serves is estimated locally.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:estimate_request_tokens
    """
    counted = _Counted(exact=refusal)
    tokens = await _approximate("amazon.nova-micro-v1:0", counted)
    assert tokens == await estimate_request_tokens(_REQUEST, "amazon.nova-micro-v1:0")
    assert counted.calls == ["exact"]
    assert "amazon.nova-micro-v1:0" not in _count_tokens.UNCOUNTABLE_MODELS, (
        "without a trivial request, nothing is remembered"
    )


@pytest.mark.usefixtures("catalog")
async def test_a_failing_proxy_falls_back_to_the_estimate() -> None:
    """When the proxy cannot count either, the local estimate answers.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:count_or_approximate
    """
    counted = _Counted(exact=42, proxy=ApiError("unavailable", status=503))
    tokens = await _approximate("anthropic.claude-sonnet-5", counted, probe=_refused)
    assert tokens == await estimate_request_tokens(
        _REQUEST, "anthropic.claude-sonnet-5"
    )


@pytest.mark.usefixtures("catalog")
@pytest.mark.parametrize(
    "refusal",
    [
        make_client_error(
            "ValidationException",
            "CountTokens",
            message="S3 input not supported for token counting",
        ),
        MantleError(
            "tools.0: Input tag 'web_search_20250305' found using 'type' does not "
            "match any of the expected tags: 'bash_20250124', 'custom'",
            status=400,
        ),
    ],
    ids=["counter", "endpoint"],
)
async def test_a_refusal_in_the_counter_s_words_never_reaches_the_caller(
    refusal: Exception,
) -> None:
    """A refusal worded by the counter itself is answered in the gateway's own words.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:_counted
    """
    with pytest.raises(ApiError) as excinfo:
        await _approximate("anthropic.claude-sonnet-4-6", _Counted(exact=refusal))
    assert excinfo.value.status == 400
    assert "token counting" not in str(excinfo.value)
    assert "expected tags" not in str(excinfo.value)


@pytest.mark.usefixtures("catalog")
async def test_the_model_s_own_validation_is_answered_as_worded() -> None:
    """A request the model itself refuses gets that refusal, as a generation would.

    Ref: https://platform.claude.com/docs/en/api/errors
         stdapi/models/chat/_adapters/_count_tokens.py:_counted
    """
    refusal = make_client_error(
        "ValidationException",
        "CountTokens",
        message="context_management: clear_thinking_20251015 requires `thinking`",
    )
    with pytest.raises(ClientError) as excinfo:
        await _approximate("anthropic.claude-sonnet-4-6", _Counted(exact=refusal))
    assert excinfo.value is refusal


#: Requests offering tools, with the prompt tokens each model billed for them (2026-09-23).
_TOOL_SAMPLES: dict[str, Any] = loads(
    (REPO_ROOT / "tests/fixtures/token_estimates_tools.json").read_text()
)


@pytest.mark.parametrize("sample", sorted(_TOOL_SAMPLES))
async def test_a_request_offering_tools_is_estimated_above_what_is_billed(
    sample: str,
) -> None:
    """Tools and a system prompt add a template the estimate still covers, on both routes.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:estimate_request_tokens
    """
    recorded = _TOOL_SAMPLES[sample]
    anthropic = MessageCountTokensParams.model_validate(
        {"model": "m", **recorded["messages"]}
    )
    openai = InputTokenCountParams.model_validate(
        {"model": "m", **recorded["responses"]}
    )
    for model_id, tokens in recorded["tokens"].items():
        assert await estimate_request_tokens(anthropic, model_id) >= tokens, model_id
        assert await estimate_request_tokens(openai, model_id) >= tokens, model_id


async def test_a_referenced_file_is_never_downloaded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An image or document given by URL is estimated from its size, never read.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:_inline_data
    """

    async def _no_read(*_args: object, **_kwargs: object) -> bytes:
        pytest.fail("the file was read")

    async def _size(_self: object) -> int:
        return 50_000

    monkeypatch.setattr(InputFile, "to_bytes", _no_read)
    monkeypatch.setattr(InputFile, "get_size", _size)
    url = "https://example.com/file"
    request = MessageCountTokensParams.model_validate(
        {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": url,
                            },
                        },
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": url,
                            },
                        },
                    ],
                }
            ],
        }
    )
    tokens = await estimate_request_tokens(request, "amazon.nova-lite-v1:0")
    assert tokens >= 16_500 + 50_000


async def test_the_counter_is_sent_only_what_it_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Documents and referenced files leave the counted input and are estimated instead.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:countable_media
    """

    async def _resolved(_region: str) -> list[tuple[str, int]]:
        return [("image", 0), ("document", 50_000)]

    monkeypatch.setattr(
        _count_tokens, "resolve_inline_bedrock_content_blocks", _resolved
    )
    image = {"image": {"format": "png", "source": {"bytes": b"png"}}}
    pdf = b"%PDF-1.4\n<< /Type /Page >>\n<< /Type /Page >>\n"
    request: Any = {
        "messages": [
            {
                "role": "user",
                "content": [
                    image,
                    {
                        "document": {
                            "format": "pdf",
                            "name": "a",
                            "source": {"bytes": pdf},
                        }
                    },
                    {"image": {"format": "png", "source": {}}},
                ],
            },
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "t", "name": "f", "input": {}}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "t",
                            "content": [
                                {
                                    "document": {
                                        "format": "pdf",
                                        "name": "b",
                                        "source": {},
                                    }
                                }
                            ],
                        }
                    }
                ],
            },
        ]
    }
    tokens = await countable_media(request, "us-east-1")
    assert tokens == 1650 + 50_000 + 2 * 3000
    assert request["messages"][0]["content"] == [image]
    assert request["messages"][2]["content"][0]["toolResult"]["content"] == [
        {"text": "."}
    ]


async def test_one_request_s_counts_share_one_call_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The counts of one request, edited and unedited, spend one budget together.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:CallBudget
    """
    monkeypatch.setattr(_count_tokens, "_MAX_CALLS", 6)
    client = _FakeCounter()
    budget = CallBudget()
    request: Any = {"messages": [_user("word " * 80_000)]}
    await gather(
        count_converse_tokens(client, "model", request, budget=budget),  # type: ignore[arg-type]
        count_converse_tokens(client, "model", request, budget=budget),  # type: ignore[arg-type]
    )
    assert len(client.requests) <= 6 + 2, "two refused wholes, then one budget"


async def test_a_failed_piece_stops_its_sibling() -> None:
    """A piece failing for another reason fails the count, and its sibling stops.

    Ref: stdapi/models/chat/_adapters/_count_tokens.py:_Pieces.split
    """
    throttled = make_client_error("ThrottlingException", "CountTokens")

    class _Throttling(_FakeCounter):
        """Throttles every piece holding the last message."""

        async def count_tokens(self, **kwargs: Any) -> dict[str, int]:  # noqa: ANN401
            messages = kwargs["input"]["converse"]["messages"]
            if len(self.requests) >= 2 and "stop" in str(messages[-1]):
                self.requests.append(kwargs["input"]["converse"])
                raise throttled
            return await super().count_tokens(**kwargs)

    client = _Throttling()
    messages = [_user("start")]
    for index in range(6):
        messages += _round_trip(index, "word " * 400)
    messages.append(_user("stop " * 2000))
    request: Any = {"messages": messages}
    with pytest.raises(ClientError) as excinfo:
        await count_converse_tokens(client, "model", request)  # type: ignore[arg-type]
    assert excinfo.value is throttled
    calls = len(client.requests)
    await sleep(0.05)
    assert len(client.requests) == calls, "no piece kept counting after the failure"
