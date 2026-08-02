"""Regression guards for the pydantic_core JSON codec swaps (issue #39).

The gateway replaces stdlib ``json`` with pydantic_core's Rust codec on every
hot path: SSE event building, HTTP response rendering, the Mantle passthrough
codec, manual request-body parsing, and botocore's rest-json request
serializer. Each swap is pinned here by an equivalence corpus (so a pydantic
or botocore upgrade that changes wire behavior fails immediately) and, where
applicable, an installed-seam check (so an upgrade that silently bypasses the
swap fails too).

Ref: stdapi/utils.py
     stdapi/aws.py
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from botocore import serialize as botocore_serialize
from botocore.loaders import Loader
from botocore.model import ServiceModel
from botocore.utils import parse_timestamp
from fastapi.datastructures import DefaultPlaceholder
from fastapi.responses import JSONResponse as FastAPIJSONResponse
from pydantic import BaseModel
from pydantic_core import from_json
from sse_starlette import JSONServerSentEvent

from stdapi.aws import CONFIG, PydanticRestJSONSerializer, parse_aws_timestamp
from stdapi.config import AWS_SESSION
from stdapi.models.chat._mantle._convert import _json_object
from stdapi.models.chat._mantle._default import _scrub_error_event, _try_loads
from stdapi.routes.cohere_rerank_v1 import _echo_document_text
from stdapi.utils import JSONResponse, json_sse, to_json_bytes, to_json_str

pytestmark = pytest.mark.local

#: JSON documents whose pydantic_core encoding is byte-identical to compact stdlib.
_BYTE_IDENTICAL_CORPUS = [
    None,
    True,
    False,
    0,
    -1,
    2**53 + 1,
    2**63,
    10**40,
    0.1,
    -0.0,
    1.5e300,
    3.141592653589793,
    "",
    "ascii",
    "héllo wörld",
    "emoji 😀 and non-BMP 𝄞",
    'escapes " \\ \n \t ',
    {},
    [],
    {"nested": {"list": [1, 2.5, None, "☃"], "empty": {}, "flag": False}},
    {"kéy": "ordre", "a": 1, "Z": 2},
    ["mixed", 1, None, {"deep": [[[]]]}],
]

#: Small-exponent floats: pydantic_core formats the exponent differently
#: (``0.00001`` vs ``1e-05``) — equal JSON numbers, not equal bytes.
_SEMANTIC_ONLY_CORPUS = [1e-5, -1e-7, {"tiny": [1e-6]}]

#: JSON texts that must parse identically through stdlib and pydantic_core.
_PARSE_CORPUS = [
    '{"a":1,"a":2}',  # duplicate keys: last one wins in both
    '"\\ud83d\\ude00"',  # surrogate-pair escape
    '"\\u00e9 raw é"',
    '{"n":1e2,"m":-0.5,"big":9007199254740993}',
    "[1,2,3,[4,[5]]]",
    '{"nested":{"unicode":"😀","null":null,"bool":true}}',
    '  {"ws": 1}  ',
]

#: Malformed JSON texts: stdlib raises JSONDecodeError, pydantic_core ValueError.
_INVALID_JSON = ["{", "", "not json", '{"a":}', "[1,"]


class _SsePayload(BaseModel):
    """Representative streamed-chunk payload for the SSE wire-format tests."""

    id: str
    kind: str = "chat.completion.chunk"
    created: int = 1714999999
    choices: list[dict[str, object]] = []
    usage: dict[str, int] | None = None
    text: str | None = None


def test_to_json_matches_stdlib_wire_format() -> None:
    """pydantic_core encoding is byte-identical to compact stdlib encoding.

    Pins the wire-format contract of every ``to_json`` call site: compact
    separators, raw UTF-8, stdlib float/int formatting. A pydantic upgrade
    changing any of it fails here before it can change API responses.

    Ref: stdapi/utils.py:to_json_str
    """
    for value in _BYTE_IDENTICAL_CORPUS:
        expected = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        assert to_json_bytes(value) == expected.encode(), value
        assert to_json_str(value) == expected, value
    for value in _SEMANTIC_ONLY_CORPUS:
        assert json.loads(to_json_str(value)) == json.loads(json.dumps(value)), value


def test_to_json_lone_surrogate_falls_back_to_stdlib() -> None:
    r"""Strings with lone surrogates encode via the stdlib fallback, not a 500.

    Stdlib ``json.loads`` accepts ``"\ud800"`` escapes, so client input can
    carry lone surrogates; pydantic_core rejects them with ``ValueError``. The
    helpers must keep such payloads encodable (ASCII-escaped) like today.

    Ref: stdapi/utils.py:to_json_str
    """
    value = {"text": "lone \ud800 surrogate"}
    encoded = to_json_bytes(value)
    assert encoded == json.dumps(value, separators=(",", ":")).encode()
    assert to_json_str(value) == json.dumps(value, separators=(",", ":"))
    assert json.loads(encoded) == value


def test_from_json_matches_stdlib_parse() -> None:
    """pydantic_core parsing produces the same structures as ``json.loads``.

    Pins duplicate-key handling, unicode escapes, and number parsing for every
    ``from_json`` call site (Mantle frames, pricing, request bodies).

    Ref: stdapi/models/chat/_mantle/_default.py:_try_loads
    """
    for text in _PARSE_CORPUS:
        assert from_json(text) == json.loads(text), text
    for text in _INVALID_JSON:
        with pytest.raises(ValueError):  # noqa: PT011 (contract: any ValueError)
            from_json(text)
        with pytest.raises(ValueError):  # noqa: PT011 (JSONDecodeError subclass)
            json.loads(text)
    # Known divergence the encoders' fallback exists for: stdlib accepts a
    # lone-surrogate escape, pydantic_core rejects it.
    assert json.loads('"\\ud800"') == "\ud800"
    with pytest.raises(ValueError):  # noqa: PT011
        from_json('"\\ud800"')


def test_json_sse_wire_bytes_identical_to_double_encode() -> None:
    """Single-pass ``json_sse`` emits byte-identical SSE frames.

    The old path dumped the model to python objects and let
    ``JSONServerSentEvent`` re-encode them with stdlib ``json``; the new path
    encodes once with pydantic_core. Both wire formats must match exactly,
    including unicode and ``exclude_none`` semantics.

    Ref: stdapi/utils.py:json_sse
    """
    payload = _SsePayload(
        id="chatcmpl-é😀",
        choices=[{"index": 0, "delta": {"content": "héllo ☃ 𝄞\nline"}}],
    )
    for event in ("response.output_text.delta", None):
        legacy = JSONServerSentEvent(
            payload.model_dump(mode="json", exclude_none=True), event=event
        )
        assert json_sse(event, payload).encode() == legacy.encode()


def test_sse_starlette_does_not_reencode_str_payloads() -> None:
    """``ServerSentEvent`` relays a pre-encoded JSON string verbatim.

    Guards the sse-starlette behavior ``json_sse`` relies on: string data is
    emitted as-is (no re-quoting), and ``JSONServerSentEvent`` keeps compact
    separators with raw UTF-8 so both paths stay byte-identical.

    Ref: stdapi/utils.py:json_sse
    """
    wire = json_sse("delta", _SsePayload(id="x", usage={"total_tokens": 1})).encode()
    assert wire == (
        b"event: delta\r\ndata: "
        b'{"id":"x","kind":"chat.completion.chunk","created":1714999999,'
        b'"choices":[],"usage":{"total_tokens":1}}\r\n\r\n'
    )
    assert JSONServerSentEvent({"é": "☃"}).encode() == (
        b'data: {"\xc3\xa9":"\xe2\x98\x83"}\r\n\r\n'
    )


def test_json_response_render_matches_stdlib() -> None:
    """The app's response class renders the same bytes as FastAPI's default.

    FastAPI's ``JSONResponse`` uses stdlib ``json.dumps`` with compact
    separators and ``ensure_ascii=False``; the pydantic_core renderer must stay
    byte-identical over the edge-case corpus and keep the JSON media type.

    Ref: stdapi/utils.py:JSONResponse
    """
    for value in _BYTE_IDENTICAL_CORPUS:
        assert JSONResponse(value).body == FastAPIJSONResponse(value).body, value
    for value in _SEMANTIC_ONLY_CORPUS:
        assert json.loads(JSONResponse(value).body) == json.loads(
            FastAPIJSONResponse(value).body
        ), value
    response = JSONResponse({"ok": True})
    assert response.media_type == "application/json"
    assert response.headers["content-type"].startswith("application/json")
    assert response.body == b'{"ok":true}'


def test_json_response_installed_as_app_default() -> None:
    """The FastAPI app renders every route with the pydantic_core response class.

    A FastAPI upgrade or app refactor that drops ``default_response_class``
    would silently fall back to stdlib rendering; this pins both the app
    default and its resolution on a registered route.

    Ref: stdapi/main.py
    """
    from stdapi.main import app  # noqa: PLC0415 (import cost deferred to this test)

    assert app.router.default_response_class is JSONResponse
    context = next(
        ctx
        for router in app.routes
        for ctx in router.effective_route_contexts()  # type: ignore[attr-defined]
        if ctx.path.endswith("/chat/completions")
    )
    response_class = context.response_class
    if isinstance(response_class, DefaultPlaceholder):
        response_class = response_class.value
    assert response_class is JSONResponse


def _converse_operation_model() -> object:
    """Load the real bedrock-runtime ``Converse`` operation model (offline)."""
    model = Loader().load_service_model("bedrock-runtime", "service-2")
    return ServiceModel(model, "bedrock-runtime").operation_model("Converse")


#: Representative Converse request bodies exercising unicode, tools and numbers.
_CONVERSE_PARAMS = [
    {
        "modelId": "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "messages": [{"role": "user", "content": [{"text": "héllo 😀 " + "x" * 64}]}],
        "system": [{"text": "Tu es un assistant émoji ☃ 𝄞"}],
        "inferenceConfig": {"maxTokens": 512, "temperature": 0.5, "topP": 0.9},
    },
    {
        "modelId": "amazon.nova-pro-v1:0",
        "messages": [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "t-1",
                            "name": "lookup",
                            "input": {"query": "μ-law", "limit": 3, "deep": {"a": []}},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "t-1",
                            "content": [{"json": {"rows": [1, 2.5, None, True]}}],
                        }
                    }
                ],
            },
        ],
        "toolConfig": {
            "tools": [
                {
                    "toolSpec": {
                        "name": "lookup",
                        "inputSchema": {"json": {"type": "object"}},
                    }
                }
            ]
        },
    },
]


def test_rest_json_serializer_body_equivalence() -> None:
    """The pydantic_core rest-json serializer matches botocore's output.

    For representative Converse bodies the swapped serializer must be
    byte-identical to ``json.dumps(separators=(",", ":"), ensure_ascii=False)``
    and semantically identical to botocore's default output (URL, method and
    headers unchanged). A botocore upgrade changing the shape walk or the
    stdlib encoding contract fails here.

    Ref: stdapi/aws.py:PydanticRestJSONSerializer
    """
    operation = _converse_operation_model()
    baseline_serializer = botocore_serialize.RestJSONSerializer()
    fast_serializer = PydanticRestJSONSerializer()
    for params in _CONVERSE_PARAMS:
        baseline = baseline_serializer.serialize_to_request(params, operation)
        fast = fast_serializer.serialize_to_request(params, operation)
        assert fast["url_path"] == baseline["url_path"]
        assert fast["method"] == baseline["method"]
        assert fast["headers"] == baseline["headers"]
        assert json.loads(fast["body"]) == json.loads(baseline["body"])
        assert (
            fast["body"]
            == json.dumps(
                json.loads(baseline["body"]), separators=(",", ":"), ensure_ascii=False
            ).encode()
        )


def test_rest_json_serializer_lone_surrogate_fallback() -> None:
    r"""A body with a lone surrogate falls back to botocore's stdlib encoding.

    ``json.loads`` accepts ``"\ud800"`` escapes in client requests;
    pydantic_core raises ``ValueError`` on them. The serializer must produce
    botocore's exact default bytes instead of failing the request.

    Ref: stdapi/aws.py:PydanticRestJSONSerializer
    """
    operation = _converse_operation_model()
    params = {
        "modelId": "m",
        "messages": [{"role": "user", "content": [{"text": "bad \ud800 text"}]}],
    }
    baseline = botocore_serialize.RestJSONSerializer().serialize_to_request(
        params, operation
    )
    fast = PydanticRestJSONSerializer().serialize_to_request(params, operation)
    assert fast["body"] == baseline["body"]


async def test_rest_json_serializer_installed_on_fresh_client() -> None:
    """A freshly created pooled-style client carries the pydantic_core serializer.

    The swap is installed in botocore's serializer registry at import time; a
    botocore/aiobotocore upgrade that stops resolving serializers through the
    registry would silently revert to stdlib. Creating a real client the same
    way ``stdapi.aws`` does must yield the swapped class.

    Ref: stdapi/aws.py:PydanticRestJSONSerializer
    """
    assert (
        botocore_serialize.SERIALIZERS["rest-json"] is PydanticRestJSONSerializer  # type: ignore[comparison-overlap]
    )
    created = botocore_serialize.create_serializer(
        "rest-json", include_validation=False
    )
    assert isinstance(created, PydanticRestJSONSerializer)
    async with AWS_SESSION.create_client(
        "bedrock-runtime", region_name="us-east-1", config=CONFIG
    ) as client:
        assert isinstance(client._serializer, PydanticRestJSONSerializer)  # noqa: SLF001


#: Real-world AWS timestamp shapes: ISO8601 variants, epoch numbers, RFC 822.
_TIMESTAMP_CORPUS = [
    "2024-05-06T12:34:56Z",
    "2024-05-06T12:34:56.123Z",
    "2024-05-06T12:34:56.123456Z",
    "2024-05-06T12:34:56+02:00",
    "2024-05-06T12:34:56-07:00",
    "2024-05-06T12:34:56.500+00:00",
    "2024-05-06T12:34:56",
    "20240506T123456Z",
    1714999999,
    1714999999.5,
    "Sun, 06 Nov 1994 08:49:37 GMT",
]


def test_timestamp_parser_matches_botocore() -> None:
    """The fromisoformat fast path parses every AWS format like botocore.

    Equality covers timezone handling (including naive offset-less strings),
    sub-second precision, epoch numbers and the RFC 822 fallback, so a
    botocore or CPython change in either parser fails here.

    Ref: stdapi/aws.py:parse_aws_timestamp
    """
    for value in _TIMESTAMP_CORPUS:
        expected = parse_timestamp(value)
        parsed = parse_aws_timestamp(value)
        assert parsed == expected, value
        assert isinstance(parsed, datetime)
        assert (parsed.tzinfo is None) == (expected.tzinfo is None), value


async def test_timestamp_parser_installed_on_fresh_client() -> None:
    """A freshly created client's response parsing uses the fast timestamp parser.

    The parser defaults are installed on the shared session's
    ``response_parser_factory`` component; botocore's Endpoint creates the
    parser that handles every response from that factory. An upgrade that
    stops routing parser creation through the session component fails here.

    Ref: stdapi/aws.py:parse_aws_timestamp
    """
    factory = AWS_SESSION.get_component("response_parser_factory")
    parser = factory.create_parser("rest-json")
    assert parser._timestamp_parser is parse_aws_timestamp  # noqa: SLF001
    async with AWS_SESSION.create_client(
        "bedrock", region_name="us-east-1", config=CONFIG
    ) as client:
        endpoint_factory = client._endpoint._response_parser_factory  # noqa: SLF001
        assert endpoint_factory is factory
        fresh = endpoint_factory.create_parser("rest-json")
        assert fresh._timestamp_parser is parse_aws_timestamp  # noqa: SLF001


def test_mantle_json_object_tolerates_invalid_arguments() -> None:
    """Invalid tool-argument JSON still defaults to an empty object.

    The pydantic_core swap must keep the passthrough converter's tolerance:
    malformed or non-object arguments become ``{}`` instead of raising.

    Ref: stdapi/models/chat/_mantle/_convert.py:_json_object
    """
    assert _json_object('{"a": 1}') == {"a": 1}
    assert _json_object("{oops") == {}
    assert _json_object("[1]") == {}
    assert _json_object("") == {}
    assert _json_object(None) == {}


def test_mantle_frame_parsing_tolerates_malformed_frames() -> None:
    """Malformed relayed SSE frames are still passed through, not fatal.

    ``_try_loads`` returns ``None`` (frame relayed unmodified) and
    ``_scrub_error_event`` returns the payload unchanged when the upstream
    frame is not valid JSON — the exact stdlib-era behavior.

    Ref: stdapi/models/chat/_mantle/_default.py:_try_loads
    """
    assert _try_loads("not json") is None
    assert _try_loads("[1, 2]") is None
    assert _try_loads('{"ok": 1}') == {"ok": 1}
    assert _scrub_error_event("not json") == "not json"
    scrubbed = _scrub_error_event(
        '{"error": {"message": "boom arn:aws:bedrock:eu-west-1:123456789012:x"}}'
    )
    parsed = json.loads(scrubbed)
    assert "123456789012" not in scrubbed
    assert parsed["error"]["message"].startswith("boom")


def test_cohere_rerank_echo_renders_json_values() -> None:
    """Object documents echo non-string values as JSON, one field per line.

    Pins the pydantic_core-rendered echo text: booleans and nulls render as
    JSON literals and containers render compactly.

    Ref: stdapi/routes/cohere_rerank_v1.py:_echo_document_text
    """
    assert _echo_document_text("plain") == "plain"
    assert _echo_document_text({"text": "verbatim"}) == "verbatim"
    assert (
        _echo_document_text(
            {"title": "Résumé", "ok": True, "count": 2, "tags": ["a", 1], "gone": None}
        )
        == 'title: Résumé\nok: true\ncount: 2\ntags: ["a",1]\ngone: null'
    )
