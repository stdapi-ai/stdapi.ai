"""Offline unit tests for the AWS Bedrock Mantle support.

Covers the transport helpers (:mod:`stdapi.aws_bedrock_mantle`), the Mantle
chat family API selection (:mod:`stdapi.models.chat._mantle._default`), and
the Mantle configuration validation — all without any AWS call.

Mantle is plain HTTPS/JSON with no botocore service model, so wire details
beyond the user guide are only defined by the gateway implementation; those
tests cite the implementing symbol rather than an upstream document.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
     stdapi/aws_bedrock_mantle.py
     stdapi/models/chat/_mantle/_default.py
"""

from __future__ import annotations

from asyncio import CancelledError, Event, create_task, wait_for
from base64 import b32encode, b64decode, urlsafe_b64encode
from binascii import crc32
from contextlib import contextmanager
from dataclasses import dataclass
from gc import collect as gc_collect
from json import JSONDecodeError, dumps, loads
from typing import TYPE_CHECKING, Any, NoReturn, cast
from urllib.parse import unquote

import pytest
from aiohttp import ClientError as AiohttpClientError
from aiohttp import ClientSession, ConnectionTimeoutError, SocketTimeoutError
from aiohttp.http_exceptions import LineTooLong
from pydantic import BaseModel, ConfigDict, ValidationError
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.datastructures import Headers

from stdapi import aws_bedrock_mantle
from stdapi import models as stdapi_models
from stdapi.api_errors import ApiError
from stdapi.aws_bedrock_mantle import (
    API_PATHS,
    MANTLE_PROJECT_VAR,
    MantleApiUnsupportedError,
    MantleError,
    MantleSurfaceUnsupportedError,
    _map_error,
    decode_mantle_response_id,
    encode_mantle_response_id,
    mantle_request_headers,
    response_web_search_queries,
    set_mantle_project,
    usage_from_chat_completion,
    usage_from_message,
    usage_from_response,
    validate_pruning_extras,
    web_search_queries,
)
from stdapi.config import AWS_SESSION, SETTINGS, _Settings
from stdapi.models import MANTLE_MODELS, MANTLE_SERVICE, ModelDetails
from stdapi.models.chat import get_chat_model, serves_via_mantle
from stdapi.models.chat._adapters._openai_responses import encode_compaction_content
from stdapi.models.chat._anthropic_claude import AnthropicClaudeChatModel
from stdapi.models.chat._mantle import _convert as mantle_convert
from stdapi.models.chat._mantle import _default as mantle_default
from stdapi.models.chat._mantle import get_mantle_chat_model
from stdapi.models.chat._mantle._default import _scrub_error_event
from stdapi.models.chat._mantle.google_gemma4 import ChatModel as GemmaChatModel
from stdapi.models.chat._mantle.open_weight import ChatModel as OpenWeightChatModel
from stdapi.models.chat._mantle.openai_gpt5 import ChatModel as GptChatModel
from stdapi.models.chat._mantle.openai_gpt_oss import ChatModel as GptOssChatModel
from stdapi.models.chat._mantle.xai_grok import ChatModel as GrokChatModel
from stdapi.models.chat.openai_gpt import ChatModel as OpenAiGptChatModel
from stdapi.monitoring import REQUEST, REQUEST_ID, EventLog
from stdapi.pricing import Service
from stdapi.region_routing import RegionRouter
from stdapi.routes.openai_responses import _decode_mantle_id, _require_local_response_id
from stdapi.types.anthropic_messages import Message, MessageCreateParams, MessageParam
from stdapi.types.openai import ModerationResult, ResponseModeration
from stdapi.types.openai_chat_completions import ChatCompletionUserMessageParam
from stdapi.types.openai_chat_completions import (
    CompletionCreateParams as ChatCompletionCreateParams,
)
from stdapi.types.openai_completions import CompletionCreateParams
from stdapi.types.openai_responses import Response, ResponseCreateParams
from tests._helpers import make_event_log, make_model_details

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator, Mapping
    from typing import Self

    from aiohttp import ClientResponse
    from fastapi import Request
    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.aws_bedrock_mantle import SseEvent

pytestmark = [pytest.mark.local, pytest.mark.usefixtures("request_log")]


@pytest.fixture(autouse=True)
def _isolated_learned_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test its own learned API/surface routing caches.

    Both dicts are process-global: a binding learned by one test would otherwise
    decide which API or surface the next test probes first.
    """
    monkeypatch.setattr(mantle_default, "_LEARNED_APIS", {})
    monkeypatch.setattr(mantle_default, "_LEARNED_SURFACE", {})


def _mantle_region() -> RegionName:
    """Return a region configured for Mantle in the test settings."""
    return SETTINGS.aws_bedrock_mantle_regions[0]


class TestMantleResponseIdCodec:
    """Region-tagged Mantle response ID encoding and decoding.

    Mantle stores responses in the source Region only, so the gateway's public
    ID embeds a crc32 fingerprint of the serving Region and decoding resolves
    it against the configured Mantle Regions.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
         stdapi/aws_bedrock_mantle.py:encode_mantle_response_id
         stdapi/aws_bedrock_mantle.py:decode_mantle_response_id
    """

    def test_round_trip(self) -> None:
        """Encoding then decoding returns the original region and native ID.

        The public form is ``resp_`` plus unpadded lowercase base32, so it
        survives case-insensitive handling and carries no base32 padding.
        """
        region = _mantle_region()
        native_id = "resp_abc123XYZ-native"
        public_id = encode_mantle_response_id(region, native_id)
        assert public_id.startswith("resp_")
        assert public_id == public_id.lower()
        assert "=" not in public_id, "base32 padding must be stripped"
        assert decode_mantle_response_id(public_id) == (region, native_id)

    def test_unknown_region_fingerprint_returns_none(self) -> None:
        """An ID tagged with an unconfigured region does not decode."""
        region = next(
            r
            for r in ("ap-southeast-2", "sa-east-1", "ca-central-1")
            if r not in SETTINGS.aws_bedrock_mantle_regions
        )
        public_id = encode_mantle_response_id(cast("RegionName", region), "resp_abc")
        assert decode_mantle_response_id(public_id) is None

    def test_local_stored_id_returns_none(self) -> None:
        """Native stdapi stored-response IDs (``resp-``) are not Mantle IDs."""
        assert decode_mantle_response_id("resp-0123456789abcdef") is None

    def test_invalid_base32_returns_none(self) -> None:
        """Arbitrary non-base32 identifiers do not decode."""
        assert decode_mantle_response_id("resp_@@@invalid@@@") is None
        assert decode_mantle_response_id("chatcmpl-0123456789") is None

    def test_missing_prefix_returns_none(self) -> None:
        """A valid base32 payload without the ``resp_`` prefix does not decode."""
        tagged = encode_mantle_response_id(_mantle_region(), "abc")
        assert decode_mantle_response_id(tagged.removeprefix("resp_")) is None

    def test_non_ascii_native_id_returns_none(self) -> None:
        """A payload whose native ID bytes are not ASCII does not decode."""
        payload = crc32(_mantle_region().encode()).to_bytes(4, "big") + b"\xff\xfe"
        public_id = "resp_" + b32encode(payload).decode("ascii").lower().rstrip("=")
        assert decode_mantle_response_id(public_id) is None

    @pytest.mark.parametrize(
        "native_id",
        [
            "resp_abc/../../secret",
            "resp_abc?x=y",
            "resp_abc#frag",
            "resp_abc def",
            "resp_abc\x00null",
        ],
    )
    def test_hostile_native_id_returns_none(self, native_id: str) -> None:
        """A native ID outside the safe charset does not decode.

        Prevents a crafted public ID from steering upstream request URLs
        (path traversal, injected query/fragment) via the interpolated ID.
        """
        public_id = encode_mantle_response_id(_mantle_region(), native_id)
        assert decode_mantle_response_id(public_id) is None

    def test_legitimate_native_id_round_trips(self) -> None:
        """A realistic hex-shaped native response ID still round-trips."""
        region = _mantle_region()
        native_id = "resp_" + "0123456789abcdef" * 2
        public_id = encode_mantle_response_id(region, native_id)
        assert decode_mantle_response_id(public_id) == (region, native_id)


class TestMantleUsageExtractors:
    """Usage extraction from the three Mantle wire formats.

    Bedrock bills fresh input tokens separately from cache reads and writes,
    while the OpenAI wire formats fold cached tokens into ``prompt_tokens`` /
    ``input_tokens``; the OpenAI-shaped extractors therefore subtract the
    cached share and the Anthropic one does not.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_TokenUsage.html
         https://developers.openai.com/api/docs/guides/prompt-caching
         https://platform.claude.com/docs/en/build-with-claude/prompt-caching
         stdapi/aws_bedrock_mantle.py:_openai_usage
         stdapi/aws_bedrock_mantle.py:usage_from_message
    """

    def test_chat_completion_subtracts_cached_share(self) -> None:
        """OpenAI prompt_tokens includes cached tokens; the share is subtracted."""
        assert usage_from_chat_completion(
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {
                    "cached_tokens": 30,
                    "cache_write_tokens": 10,
                },
            }
        ) == {
            "input_tokens": 60,
            "output_tokens": 20,
            "total_tokens": 120,
            "cached_tokens": 30,
            "cache_write_tokens": 10,
        }

    def test_chat_completion_without_details(self) -> None:
        """Missing prompt_tokens_details means no cached share to subtract."""
        assert usage_from_chat_completion(
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        ) == {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
        }

    def test_response_subtracts_cached_share(self) -> None:
        """Responses input_tokens includes cached tokens; the share is subtracted."""
        assert usage_from_response(
            {
                "input_tokens": 100,
                "output_tokens": 40,
                "total_tokens": 140,
                "input_tokens_details": {"cached_tokens": 25},
            }
        ) == {
            "input_tokens": 75,
            "output_tokens": 40,
            "total_tokens": 140,
            "cached_tokens": 25,
            "cache_write_tokens": 0,
        }

    def test_message_keeps_input_tokens_as_is(self) -> None:
        """Anthropic input_tokens already excludes cache reads and writes."""
        assert usage_from_message(
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 3,
                "cache_creation_input_tokens": 2,
            }
        ) == {
            "input_tokens": 10,
            "output_tokens": 5,
            "cached_tokens": 3,
            "cache_write_tokens": 2,
        }

    def test_chat_completion_explicit_nulls_yield_zero_usage(self) -> None:
        """Explicit JSON nulls in a Chat Completions usage block do not raise."""
        assert usage_from_chat_completion(
            {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "prompt_tokens_details": None,
            }
        ) == {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
        }

    def test_response_explicit_nulls_yield_zero_usage(self) -> None:
        """Explicit JSON nulls in a Responses usage block do not raise."""
        assert usage_from_response(
            {
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "input_tokens_details": None,
            }
        ) == {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
        }

    def test_message_explicit_nulls_yield_zero_usage(self) -> None:
        """Explicit JSON nulls in an Anthropic usage block do not raise."""
        assert usage_from_message(
            {
                "input_tokens": None,
                "output_tokens": None,
                "cache_read_input_tokens": None,
                "cache_creation_input_tokens": None,
            }
        ) == {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "cache_write_tokens": 0,
        }


def _error_body(message: str) -> str:
    """Build a Mantle JSON error body carrying *message*."""
    return dumps({"error": {"message": message}})


class TestMapError:
    """Mantle HTTP error mapping (:func:`stdapi.aws_bedrock_mantle._map_error`).

    The marker strings the mapping keys on are empirical Mantle behavior: a
    model/surface mismatch answers either with "isn't supported on this route"
    on any status or with a misleading 401 "is not enabled", which is textually
    close to a genuine 403 permission denial.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html
         stdapi/aws_bedrock_mantle.py:_map_error
         stdapi/aws_bedrock_mantle.py:MantleError
    """

    @pytest.mark.parametrize("status", [429, 500, 503])
    def test_throttling_and_5xx_are_failover(self, status: int) -> None:
        """Throttling and server errors are retryable in another region."""
        error = _map_error(status, _error_body("Too busy"), "us-east-1")
        assert type(error) is MantleError
        assert error.failover is True
        assert error.status == status
        assert "Too busy" in str(error)

    def test_unsupported_api_marker(self) -> None:
        """The quoted API binding mismatch phrase maps to the demotion error."""
        error = _map_error(
            400,
            _error_body("The model does not support the 'responses' API."),
            "us-east-1",
        )
        assert isinstance(error, MantleApiUnsupportedError)
        assert error.status == 400
        assert error.failover is False

    def test_parameter_error_not_misclassified_as_unsupported_api(self) -> None:
        """A parameter-support message must not demote the model's API binding."""
        error = _map_error(
            400,
            _error_body("This model does not support the 'temperature' parameter."),
            "us-east-1",
        )
        assert type(error) is MantleError
        assert error.status == 400

    def test_structured_fields_propagated(self) -> None:
        """Upstream ``code`` and ``param`` fields survive the error mapping."""
        body = dumps(
            {
                "error": {
                    "message": "Unsupported value for 'temperature'.",
                    "code": "unsupported_value",
                    "param": "temperature",
                }
            }
        )
        error = _map_error(400, body, "us-east-1")
        assert error.code == "unsupported_value"
        assert error.param == "temperature"

    def test_structured_fields_absent_stay_none(self) -> None:
        """Errors without ``code``/``param`` keep the class defaults."""
        error = _map_error(400, _error_body("max_tokens is too large"), "us-east-1")
        assert error.code is None
        assert error.param is None

    @pytest.mark.parametrize(
        ("status", "message"),
        [
            (400, "This model isn't supported on this route."),
            (404, "This model isn't supported on this route."),
            (401, "Berm is not enabled for this account."),
        ],
    )
    def test_unsupported_surface_markers(self, status: int, message: str) -> None:
        """Surface routing mismatch markers map to the surface error."""
        error = _map_error(status, _error_body(message), "us-east-1")
        assert isinstance(error, MantleSurfaceUnsupportedError)
        assert error.status == 400
        assert error.failover is False

    def test_permission_error_is_not_a_surface_mismatch(self) -> None:
        """A 403 'is not enabled' permission error is the deployment's, not a 400."""
        error = _map_error(
            403, _error_body("Model access is not enabled."), "us-east-1"
        )
        assert type(error) is MantleError
        assert error.status == 503
        assert error.code == "feature_unavailable"

    @pytest.mark.parametrize("status", [401, 403])
    def test_credential_errors_are_answered_as_a_feature_the_deployment_lacks(
        self, status: int
    ) -> None:
        """Upstream auth failures answer 503 ``feature_unavailable``, not 500.

        A denial of the server's own role is a deployment that is missing a
        permission, which every other backend answers this way -- a 500 asks the
        caller to retry something no retry can fix. Missing
        ``bedrock-mantle:CountTokens`` reached a client as "The request could not
        be completed. Retry the request." with a 500, and reads as an outage.

        Ref: stdapi/api_errors.py:FeatureUnavailableError
        """
        error = _map_error(status, _error_body("Invalid bearer token"), "us-east-1")
        assert type(error) is MantleError
        assert error.status == 503
        assert error.code == "feature_unavailable"
        assert error.failover is False

    @pytest.mark.parametrize("status", [401, 403])
    def test_credential_errors_do_not_take_the_upstream_error_code(
        self, status: int
    ) -> None:
        """An upstream ``code`` must not displace ``feature_unavailable``."""
        body = dumps(
            {"error": {"message": "Invalid bearer token", "code": "upstream_code"}}
        )
        error = _map_error(status, body, "us-east-1")
        assert error.code == "feature_unavailable"

    @pytest.mark.parametrize("status", [401, 403])
    def test_credential_errors_evict_cached_token(
        self, monkeypatch: pytest.MonkeyPatch, status: int
    ) -> None:
        """A mapped 401/403 evicts the region's cached bearer token."""
        monkeypatch.setitem(aws_bedrock_mantle._TOKENS, "us-east-1", ("stale", 1e18))  # noqa: SLF001
        _map_error(status, _error_body("Invalid bearer token"), "us-east-1")
        assert "us-east-1" not in aws_bedrock_mantle._TOKENS  # noqa: SLF001

    @pytest.mark.parametrize("status", [401, 403])
    def test_credential_error_message_is_generic(self, status: int) -> None:
        """The raw upstream credential message is not forwarded to the client."""
        message = "User: arn:aws:iam::123456789012:role/x is not authorized"
        error = _map_error(status, _error_body(message), "us-east-1")
        assert "arn:aws:iam" not in str(error)
        assert "123456789012" not in str(error)

    def test_client_error_passthrough(self) -> None:
        """Plain 4xx errors keep their status and message without failover."""
        error = _map_error(400, _error_body("max_tokens is too large"), "us-east-1")
        assert type(error) is MantleError
        assert error.status == 400
        assert error.failover is False

    @pytest.mark.parametrize("body", ["not json at all", "[1, 2, 3]", "null"])
    def test_unparsable_body_uses_default_message(self, body: str) -> None:
        """Non-JSON or non-object bodies fall back to a generic message."""
        error = _map_error(418, body, "us-east-1")
        assert "HTTP 418" in str(error)
        assert error.status == 418

    def test_string_error_body_preserves_message(self) -> None:
        """A bare-string ``error`` field maps cleanly, keeping the message."""
        error = _map_error(500, '{"error":"backend exploded"}', "us-east-1")
        assert type(error) is MantleError
        assert error.failover is True
        assert error.status == 500
        assert "backend exploded" in str(error)


class TestRequestTimeoutFailover:
    """Connect-phase failures fail over; post-send read timeouts never do.

    A ``sock_read`` timeout fires after the request reached Mantle, so the
    invocation is already billed: retrying it in another region would
    double-bill it (the Converse-side ``route_and_execute`` applies the same
    rule to botocore's ``ReadTimeoutError``).

    Ref: https://docs.aiohttp.org/en/stable/client_reference.html#hierarchy-of-exceptions
         stdapi/aws_bedrock_mantle.py:_request
         stdapi/models/__init__.py:_region_failover_label
    """

    @staticmethod
    def _stub_transport(monkeypatch: pytest.MonkeyPatch, error: Exception) -> None:
        """Stub the pooled session and bearer token so the send raises *error*."""

        class FakeSession:
            """Session whose request coroutine always raises."""

            async def request(self, *args: object, **kwargs: object) -> NoReturn:
                raise error

        async def fake_token(region: RegionName) -> str:  # noqa: ARG001
            return "token"

        monkeypatch.setattr(aws_bedrock_mantle, "_SESSION", FakeSession())
        monkeypatch.setattr(aws_bedrock_mantle, "bearer_token", fake_token)

    async def test_read_timeout_is_not_failover(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A post-send read timeout raises a 503 that blocks region failover."""
        self._stub_transport(monkeypatch, SocketTimeoutError("read timed out"))
        with pytest.raises(MantleError) as exc_info:
            await aws_bedrock_mantle._request(  # noqa: SLF001
                "us-east-1", "/v1/chat/completions", b"{}", None
            )
        assert exc_info.value.failover is False
        assert exc_info.value.status == 503

    @pytest.mark.parametrize(
        "error",
        [
            ConnectionTimeoutError("connect timed out"),
            AiohttpClientError("connection refused"),
            TimeoutError("timed out"),
        ],
    )
    async def test_other_transport_failures_keep_failing_over(
        self, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        """Errors that do not prove a billed invocation stay failover-eligible."""
        self._stub_transport(monkeypatch, error)
        with pytest.raises(MantleError) as exc_info:
            await aws_bedrock_mantle._request(  # noqa: SLF001
                "us-east-1", "/v1/chat/completions", b"{}", None
            )
        assert exc_info.value.failover is True
        assert exc_info.value.status == 503


class TestIterSseLineTooLong:
    """Oversized SSE lines are mapped to a shaped 502 :class:`MantleError`.

    A single Mantle SSE event can carry the whole response JSON, so the reader
    catches ``aiohttp``'s ``LineTooLong`` — an ``HttpProcessingError`` that
    sits outside ``ClientError`` and would otherwise escape unmapped.

    Ref: stdapi/aws_bedrock_mantle.py:_iter_sse
    """

    async def test_line_too_long_maps_to_502(self) -> None:
        """An ``aiohttp`` ``LineTooLong`` mid-stream raises a 502 ``MantleError``."""

        class _RaisingContent:
            """Fake response content raising ``LineTooLong`` on first iteration."""

            def __aiter__(self) -> Self:
                return self

            async def __anext__(self) -> bytes:
                oversized_line, limit = b"x", "1024"
                raise LineTooLong(oversized_line, limit)

        class _FakeResponse:
            """Fake streaming response supporting ``async with`` and iteration."""

            content = _RaisingContent()

            async def __aenter__(self) -> Self:
                return self

            async def __aexit__(self, *exc_info: object) -> bool:
                return False

        with pytest.raises(MantleError) as exc_info:
            async for _ in aws_bedrock_mantle._iter_sse(  # noqa: SLF001
                cast("ClientResponse", _FakeResponse())
            ):
                pass
        assert exc_info.value.status == 502


class _Inner(BaseModel):
    """Nested payload model rejecting unknown fields."""

    model_config = ConfigDict(extra="forbid")

    value: int


class _Outer(BaseModel):
    """Top-level payload model rejecting unknown fields."""

    model_config = ConfigDict(extra="forbid")

    name: str
    inner: _Inner


def _message_payload() -> dict[str, Any]:
    """Build an Anthropic Message payload with an extra in a tool_use block."""
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "test.model",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {"city": "Paris"},
                "billing_meta": {"cost": 1},
            }
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }


class TestMantleModelClassResolution:
    """Model IDs, including future versions, bind to the right Mantle class.

    The bound class is what seeds the routing surface and the API set, and no
    Mantle API reports either: ``/openai/v1`` serves the newer Mantle-only
    models while ``/v1`` serves the legacy catalog, so a mis-bound family
    probes the wrong prefix first.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-56-cyber.html
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-daybreak-blue-56-sol.html
         stdapi/models/chat/_mantle/__init__.py:get_mantle_chat_model
         stdapi/models/chat/_mantle/_default.py:ChatModel._api_paths
    """

    @pytest.mark.parametrize(
        "model_id", ["openai.gpt-5.6-sol", "openai.gpt-6-nova", "openai.gpt-10.1-vega"]
    )
    def test_numbered_gpt_versions_use_the_gpt_class(self, model_id: str) -> None:
        """GPT-5 and future numbered GPT versions resolve to the GPT class."""
        model = get_mantle_chat_model(model_id)
        assert isinstance(model, GptChatModel)
        assert model._api_paths("responses")[0] == "/openai/v1/responses"  # noqa: SLF001
        assert model.native_store_supported() is True

    @pytest.mark.parametrize(
        "model_id",
        [
            "openai.gpt-5.6-cyber",
            "openai.gpt-daybreak-blue-5.6-sol",
            "openai.gpt-daybreak-red-6-cyber",
        ],
    )
    def test_daybreak_editions_use_the_gpt_class(self, model_id: str) -> None:
        """Daybreak-qualified GPT IDs resolve to the GPT class, image input included.

        AWS names Daybreak Red ``openai.gpt-5.6-cyber`` but Daybreak Blue
        ``openai.gpt-daybreak-blue-5.6-sol``, so the qualifier sits where the
        version number is expected; unmatched, the model falls back to the
        generic Mantle class and is advertised as text-only.
        """
        model = get_mantle_chat_model(model_id)
        assert isinstance(model, GptChatModel)
        assert model._api_paths("responses")[0] == "/openai/v1/responses"  # noqa: SLF001
        assert model.native_store_supported() is True
        assert model.INPUT_MODALITIES == ("TEXT", "IMAGE")

    def test_gpt_oss_is_not_matched_by_the_numbered_gpt_class(self) -> None:
        """gpt-oss models keep resolving to their own class."""
        model = get_mantle_chat_model("openai.gpt-oss-120b")
        assert isinstance(model, GptOssChatModel)
        assert not isinstance(model, GptChatModel)
        assert model._api_paths("responses")[0] == "/v1/responses"  # noqa: SLF001

    @pytest.mark.parametrize(
        "model_id",
        ["google.gemma-4-e2b", "google.gemma-5-e4b", "google.gemma-12-large"],
    )
    def test_gemma_4_and_later_use_the_gemma_class(self, model_id: str) -> None:
        """Gemma 4 and future Gemma versions resolve to the Gemma class."""
        model = get_mantle_chat_model(model_id)
        assert isinstance(model, GemmaChatModel)
        assert (
            model._api_paths("chat_completions")[0]  # noqa: SLF001
            == "/openai/v1/chat/completions"
        )
        assert model.native_store_supported() is True

    @pytest.mark.parametrize(
        "model_id", ["google.gemma-3-4b-it", "google.gemma-3n-e2b"]
    )
    def test_gemma_3_stays_on_the_open_weight_class(self, model_id: str) -> None:
        """Gemma 3 (including 3n) keeps resolving to the open-weight class."""
        model = get_mantle_chat_model(model_id)
        assert isinstance(model, OpenWeightChatModel)
        assert not isinstance(model, GemmaChatModel)
        assert (
            model._api_paths("chat_completions")[0] == "/v1/chat/completions"  # noqa: SLF001
        )
        assert model.native_store_supported() is False

    @pytest.mark.parametrize(
        "model_id", ["xai.grok-4.3", "xai.grok-5", "xai.grok-6-fast-reasoning"]
    )
    def test_grok_versions_use_the_grok_class(self, model_id: str) -> None:
        """Grok IDs resolve to the Grok class, on both APIs and with image input.

        The matcher is the bare ``xai.`` prefix, so no version pins it.
        Unmatched, Grok falls back to the generic Mantle class, which is
        Responses-only, probes ``/v1`` first and is advertised as text-only.
        """
        model = get_mantle_chat_model(model_id)
        assert isinstance(model, GrokChatModel)
        assert not isinstance(model, OpenWeightChatModel)
        assert (
            model._api_paths("chat_completions")[0]  # noqa: SLF001
            == "/openai/v1/chat/completions"
        )
        assert model._api_paths("responses")[0] == "/openai/v1/responses"  # noqa: SLF001
        assert model.native_store_supported() is True
        assert model.INPUT_MODALITIES == ("TEXT", "IMAGE")


class TestMantleSystemMessageAsMessages:
    """Native mid-conversation system message support is scoped per model family.

    Anthropic's Messages API takes the system prompt as a top-level field, so a
    mid-conversation ``system`` role must be folded inline unless the served
    model handles it natively.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/models/chat/_mantle/_default.py:ChatModel._system_message_as_messages
         stdapi/models/chat/_mantle/anthropic_claude.py:ChatModel
    """

    @pytest.mark.parametrize(
        "model_id",
        [
            "anthropic.claude-opus-4-8",
            "anthropic.claude-opus-5",
            "anthropic.claude-opus-5-1-20260724-v1:0",
            "anthropic.claude-fable-5",
            "anthropic.claude-mythos-5",
            "anthropic.claude-mythos-preview",
            "anthropic.claude-sonnet-5-20260501-v1:0",
            "anthropic.claude-haiku-5",
        ],
    )
    def test_opus_48_and_later_forward_system_messages(self, model_id: str) -> None:
        """Opus 4.8+, Fable and Mythos handle system-role messages natively."""
        model = cast("mantle_default.ChatModel", get_mantle_chat_model(model_id))
        assert model._system_message_as_messages() is True  # noqa: SLF001

    @pytest.mark.parametrize(
        "model_id",
        [
            "anthropic.claude-opus-4-7",
            "anthropic.claude-opus-4-6",
            "anthropic.claude-haiku-4-5-20251001-v1:0",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "openai.gpt-5.6-sol",
        ],
    )
    def test_other_models_keep_folding_system_messages(self, model_id: str) -> None:
        """Models without native support keep the folding fallback."""
        model = cast("mantle_default.ChatModel", get_mantle_chat_model(model_id))
        assert model._system_message_as_messages() is False  # noqa: SLF001


class TestValidatePruningExtras:
    """Extra-field pruning validation for upstream passthrough payloads.

    Passthrough responses are validated against ``extra="forbid"`` models, so
    provider extensions (e.g. a ``billing`` block) would break relaying; they
    are dropped in place over bounded re-validation rounds while any other
    validation failure becomes a 502.

    Ref: stdapi/aws_bedrock_mantle.py:validate_pruning_extras
         stdapi/aws_bedrock_mantle.py:_resolve_error_loc
    """

    def test_prunes_top_level_and_nested_extras(self) -> None:
        """Unknown top-level and nested fields are dropped, then validated."""
        raw = {
            "name": "ok",
            "billing": {"cost": 1},
            "inner": {"value": 3, "debug": "x"},
        }
        result = validate_pruning_extras(_Outer, raw)
        assert result.name == "ok"
        assert result.inner.value == 3
        assert "billing" not in raw
        assert "debug" not in raw["inner"]

    def test_valid_payload_untouched(self) -> None:
        """A payload matching the schema validates on the first round."""
        raw = {"name": "ok", "inner": {"value": 1}}
        assert validate_pruning_extras(_Outer, raw).inner.value == 1

    def test_genuine_type_error_maps_to_shaped_error(self) -> None:
        """Validation errors that are not extra fields surface as a 502."""
        raw = {"name": "ok", "extra": True, "inner": {"value": "not-an-int"}}
        with pytest.raises(MantleError) as exc_info:
            validate_pruning_extras(_Outer, raw)
        assert exc_info.value.status == 502
        assert isinstance(exc_info.value.__cause__, ValidationError)

    def test_union_nested_extra_pruned(self) -> None:
        """Extras inside a union content block are pruned despite tag-label locs."""
        raw = _message_payload()
        message = validate_pruning_extras(Message, raw)
        assert message.content[0].type == "tool_use"
        assert "billing_meta" not in raw["content"][0]

    def test_union_nested_genuine_error_maps_to_shaped_error(self) -> None:
        """A wrong-typed required field still fails (502) after union pruning."""
        raw = _message_payload()
        raw["usage"] = {"input_tokens": "not-an-int", "output_tokens": 2}
        with pytest.raises(MantleError) as exc_info:
            validate_pruning_extras(Message, raw)
        assert exc_info.value.status == 502


class TestSelectApi:
    """Mantle upstream API selection and learned-binding behavior.

    No Mantle API reports which of Responses / Chat Completions / Messages a
    model serves, so the binding is seeded per family and demoted along
    ``responses -> chat_completions -> messages`` from upstream rejections.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
         stdapi/models/chat/_mantle/_default.py:ChatModel._select_api
         stdapi/models/chat/_mantle/_default.py:_API_FALLBACK_ORDER
    """

    def test_inbound_api_preferred_when_supported(self) -> None:
        """The inbound API is used directly when the model supports it."""
        model = mantle_default.ChatModel("test.unknown-model")
        assert model._select_api("responses", set()) == "responses"  # noqa: SLF001

    def test_fallback_order_responses_first(self) -> None:
        """Unsupported inbound APIs fall back along the documented chain."""
        model = mantle_default.ChatModel("test.unknown-model")
        # The default family only supports Responses: messages fall back to it.
        assert model._select_api("messages", set()) == "responses"  # noqa: SLF001
        # A chat-completions-only family falls back to Chat Completions.
        gemma3 = OpenWeightChatModel("google.gemma-3-4b-it")
        assert gemma3._select_api("responses", set()) == "chat_completions"  # noqa: SLF001

    def test_optimistic_probe_when_supported_apis_exhausted(self) -> None:
        """Untried APIs are probed once the learned set is exhausted."""
        model = mantle_default.ChatModel("test.unknown-model")
        assert model._select_api("chat_completions", {"responses"}) == (  # noqa: SLF001
            "chat_completions"
        )

    def test_all_apis_tried_raises(self) -> None:
        """A request no Mantle API can serve fails with a 400 error."""
        model = mantle_default.ChatModel("test.unknown-model")
        with pytest.raises(ApiError) as exc_info:
            model._select_api(  # noqa: SLF001
                "responses", {"responses", "chat_completions", "messages"}
            )
        assert exc_info.value.status == 400
        assert "cannot serve this request on Bedrock Mantle" in str(exc_info.value)

    def test_learned_apis_override_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A learned binding takes precedence over the seeded NATIVE_APIS."""
        model_id = "test.learned-model"
        model = mantle_default.ChatModel(model_id)
        monkeypatch.setitem(
            mantle_default._LEARNED_APIS,  # noqa: SLF001
            model_id,
            frozenset({"messages"}),
        )
        assert model._select_api("responses", set()) == "messages"  # noqa: SLF001
        assert model.native_store_supported() is False

    def test_native_store_follows_supported_apis(self) -> None:
        """Native storage is advertised only for Responses-serving models."""
        assert mantle_default.ChatModel("test.unknown-model").native_store_supported()
        assert not OpenWeightChatModel("google.gemma-3-4b-it").native_store_supported()


class TestSurfaceCacheLru:
    """Bounded LRU behavior of the stored-response surface cache.

    A stored response must be retrieved on the same surface that created it,
    so its native ID is remembered; the map is bounded and evicts by insertion
    order, with a read refreshing recency.

    Ref: stdapi/aws_bedrock_mantle.py:cache_response_surface
         stdapi/aws_bedrock_mantle.py:cached_response_surface
    """

    def _reset(self, monkeypatch: pytest.MonkeyPatch, max_size: int) -> None:
        """Reset the surface cache to empty with a small capacity for the test."""
        monkeypatch.setattr(aws_bedrock_mantle, "_SURFACE_CACHE", {})
        monkeypatch.setattr(aws_bedrock_mantle, "_SURFACE_CACHE_MAX", max_size)

    def test_miss_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unknown native ID has no cached surface."""
        self._reset(monkeypatch, 3)
        assert aws_bedrock_mantle.cached_response_surface("unknown") is None

    def test_update_of_cached_key_at_capacity_evicts_nothing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Rewriting an already-cached key at capacity evicts no other entry."""
        self._reset(monkeypatch, 3)
        for key in ("a", "b", "c"):
            aws_bedrock_mantle.cache_response_surface(key, "/v1")
        aws_bedrock_mantle.cache_response_surface("b", "/openai/v1")
        assert set(aws_bedrock_mantle._SURFACE_CACHE) == {"a", "b", "c"}  # noqa: SLF001
        assert aws_bedrock_mantle._SURFACE_CACHE["b"] == "/openai/v1"  # noqa: SLF001

    def test_new_key_at_capacity_evicts_oldest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inserting a new key at capacity evicts only the oldest entry."""
        self._reset(monkeypatch, 3)
        for key in ("a", "b", "c"):
            aws_bedrock_mantle.cache_response_surface(key, "/v1")
        aws_bedrock_mantle.cache_response_surface("d", "/v1")
        assert set(aws_bedrock_mantle._SURFACE_CACHE) == {"b", "c", "d"}  # noqa: SLF001

    def test_cache_hit_refreshes_recency(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reading an entry protects it from the next eviction."""
        self._reset(monkeypatch, 3)
        for key in ("a", "b", "c"):
            aws_bedrock_mantle.cache_response_surface(key, "/v1")
        assert aws_bedrock_mantle.cached_response_surface("a") == "/v1"
        aws_bedrock_mantle.cache_response_surface("d", "/v1")
        assert set(aws_bedrock_mantle._SURFACE_CACHE) == {"a", "c", "d"}  # noqa: SLF001


class TestApiPathsSelfHeal:
    """Candidate request paths self-heal via a same-request alternate surface.

    The known surface is tried first and the other one kept as a fallback, so a
    stale learned or seeded surface recovers within the request instead of
    failing until restart. The Messages API is not surface-relative: its path
    is the absolute third prefix ``/anthropic/v1/messages``.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html
         stdapi/models/chat/_mantle/_default.py:ChatModel._api_paths
         stdapi/aws_bedrock_mantle.py:API_PATHS
    """

    def test_learned_surface_first_alternate_second(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A learned surface is tried first; the alternate stays as fallback."""
        model = mantle_default.ChatModel("test.learned-surface-model")
        monkeypatch.setitem(
            mantle_default._LEARNED_SURFACE,  # noqa: SLF001
            "test.learned-surface-model",
            "/openai/v1",
        )
        assert model._api_paths("chat_completions") == [  # noqa: SLF001
            "/openai/v1/chat/completions",
            "/v1/chat/completions",
        ]

    def test_class_seeded_surface_first_alternate_second(self) -> None:
        """A class-seeded surface is tried first; the alternate stays as fallback."""
        model = OpenWeightChatModel("google.gemma-3-4b-it")
        assert model._api_paths("chat_completions") == [  # noqa: SLF001
            "/v1/chat/completions",
            "/openai/v1/chat/completions",
        ]

    def test_no_known_surface_probes_openai_first(self) -> None:
        """With no learned or seeded surface, ``/openai/v1`` is probed before ``/v1``."""
        model = mantle_default.ChatModel("test.unseeded-surface-model")
        assert model._api_paths("chat_completions") == [  # noqa: SLF001
            "/openai/v1/chat/completions",
            "/v1/chat/completions",
        ]

    def test_messages_api_has_single_path(self) -> None:
        """The Anthropic Messages API always uses its single fixed path."""
        model = mantle_default.ChatModel("test.unseeded-surface-model")
        assert model._api_paths("messages") == ["/anthropic/v1/messages"]  # noqa: SLF001
        assert API_PATHS["messages"] == "/anthropic/v1/messages"


class TestMantleSettings:
    """Mantle configuration fields and validation.

    Bedrock Guardrails are a bedrock-runtime feature and do not apply to
    Mantle-served requests, so letting a caller pick the Mantle transport
    per request would be a guardrail bypass and is rejected at load time.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
         stdapi/config.py:_Settings
    """

    def test_mantle_enabled_by_default(self) -> None:
        """Mantle support is enabled unless explicitly turned off."""
        assert SETTINGS.aws_bedrock_mantle_enabled is True

    def test_service_header_requires_mantle_enabled(self) -> None:
        """The per-request service header cannot be enabled without Mantle."""
        with pytest.raises(ValidationError) as exc_info:
            _Settings(
                aws_bedrock_mantle_service_header=True, aws_bedrock_mantle_enabled=False
            )
        assert (
            "aws_bedrock_mantle_service_header requires aws_bedrock_mantle_enabled"
            in str(exc_info.value)
        )

    def test_service_header_incompatible_with_guardrails(self) -> None:
        """The per-request service header cannot bypass configured guardrails."""
        with pytest.raises(ValidationError) as exc_info:
            _Settings(
                aws_bedrock_mantle_service_header=True,
                aws_bedrock_guardrail_identifier="test-guardrail",
                aws_bedrock_guardrail_version="1",
            )
        message = str(exc_info.value)
        assert "aws_bedrock_mantle_service_header" in message
        assert "incompatible with Amazon Bedrock Guardrails" in message

    def test_service_header_valid_with_mantle_enabled(self) -> None:
        """The service header validates when Mantle is enabled, guardrails off."""
        settings = _Settings(aws_bedrock_mantle_service_header=True)
        assert settings.aws_bedrock_mantle_service_header is True

    def test_mantle_regions_default_drops_regions_without_an_endpoint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unset Mantle regions inherit only the Bedrock regions that serve Mantle.

        Bedrock Mantle is offered in fewer regions than classic Bedrock, and
        where it is not offered ``bedrock-mantle.<region>.api.aws`` has no DNS
        record at all. Inheriting every Bedrock region therefore makes a
        deployment probe an address that cannot exist, spending its connection
        budget and warning at every refresh about a permanent configuration
        fact.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
             stdapi/config.py:_Settings._validate
        """
        monkeypatch.delenv("aws_bedrock_mantle_regions", raising=False)
        monkeypatch.delenv("AWS_BEDROCK_MANTLE_REGIONS", raising=False)

        settings = _Settings(
            aws_bedrock_regions=["us-east-1", "eu-west-3", "eu-west-1"]
        )

        assert settings.aws_bedrock_mantle_regions == ["us-east-1", "eu-west-1"], (
            "eu-west-3 offers no bedrock-mantle endpoint"
        )

    def test_mantle_regions_explicit_value_preserved(self) -> None:
        """An explicit Mantle region list is kept as-is, filtering included.

        A region AWS starts serving after this release is unknown to the
        default filter, so an explicit list must pass through untouched or that
        region could not be used at all until the next release.
        """
        regions: list[RegionName] = ["eu-west-3"]
        settings = _Settings(aws_bedrock_mantle_regions=regions)
        assert settings.aws_bedrock_mantle_regions == regions


class TestMantleCatalogRobustness:
    """Mantle model catalog fetch tolerates malformed entries and region failures.

    The catalog comes from Mantle's ``GET /models`` per Region; a Region that
    cannot be listed (permissions, outage) is recorded as a failed Region so
    startup still serves the models the other Regions returned.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions-mantle.html
         stdapi/models/__init__.py:_get_mantle_models_from_region
         stdapi/models/__init__.py:_collect_mantle_models
    """

    async def test_null_data_returns_no_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A catalog response with a null ``data`` field yields no models."""

        async def fake_request_json(
            region: RegionName,  # noqa: ARG001
            method: str,  # noqa: ARG001
            path: str,  # noqa: ARG001
        ) -> dict[str, Any]:
            return {"data": None}

        monkeypatch.setattr(stdapi_models, "mantle_request_json", fake_request_json)
        models = await stdapi_models._get_mantle_models_from_region(  # noqa: SLF001
            _mantle_region()
        )
        assert models == []

    async def test_malformed_entries_are_skipped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Entries missing or with an empty ``id`` are skipped; valid ones are kept."""

        async def fake_request_json(
            region: RegionName,  # noqa: ARG001
            method: str,  # noqa: ARG001
            path: str,  # noqa: ARG001
        ) -> dict[str, Any]:
            return {"data": [{"no_id": True}, {"id": ""}, {"id": "prov.good-model"}]}

        monkeypatch.setattr(stdapi_models, "mantle_request_json", fake_request_json)
        models = await stdapi_models._get_mantle_models_from_region(  # noqa: SLF001
            _mantle_region()
        )
        assert [model.id for model in models] == ["prov.good-model"]

    async def test_region_failure_recorded_others_still_collected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-``ApiError`` region failure is recorded; other regions still collect."""
        regions: list[RegionName] = ["us-east-1", "us-west-2"]
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_regions", regions)

        async def fake_get(region: RegionName) -> list[ModelDetails]:
            if region == "us-east-1":
                error_message = "boom"
                raise KeyError(error_message)
            return [
                make_model_details(
                    "prov.ok-model",
                    name="ok-model",
                    provider="Prov",
                    service=MANTLE_SERVICE,
                    regions=[region],
                )
            ]

        monkeypatch.setattr(stdapi_models, "_get_mantle_models_from_region", fake_get)
        failed_regions: dict[str, str] = {}
        models = await stdapi_models._collect_mantle_models(failed_regions, {})  # noqa: SLF001
        assert "us-east-1 (Mantle)" in failed_regions
        assert "KeyError" in failed_regions["us-east-1 (Mantle)"]
        assert set(models) == {"prov.ok-model"}


def _model_details(model_id: str, service: str | None = None) -> ModelDetails:
    """Build minimal model details for alias tests."""
    details = make_model_details(model_id, provider="OpenAI")
    if service:
        details.service = service
    return details


async def _fake_stream(events: list[SseEvent]) -> AsyncGenerator[SseEvent]:
    """Yield pre-built SSE events as a fake upstream Mantle stream."""
    for item in events:
        yield item


def _event_data(event: ServerSentEvent) -> str:
    """Return a relayed event's data payload, narrowed to ``str``."""
    assert isinstance(event.data, str)
    return event.data


def _capture_usage_records(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Patch ``record_bedrock_usage`` in the Mantle chat module, capturing calls."""
    records: list[dict[str, Any]] = []

    def fake_record(model: str, **kwargs: object) -> None:
        records.append({"model": model, **kwargs})

    monkeypatch.setattr(mantle_default, "record_bedrock_usage", fake_record)
    return records


def _responses_stream_events() -> list[SseEvent]:
    """Build a minimal responses-shaped SSE stream carrying usage."""
    response = {"id": "resp_native1", "created_at": 123, "model": "test.model"}
    completed = {
        **response,
        "status": "completed",
        "output": [],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }
    return [
        ("response.created", dumps({"response": response})),
        ("response.output_text.delta", dumps({"delta": "Hello"})),
        ("response.completed", dumps({"response": completed})),
    ]


class TestMantleStreamRelay:
    """Stream relaying: conversion billing and stored-ID rewrites.

    Usage is observed on the raw upstream events before any wire conversion, so
    a request served by a different API than the inbound route is still billed
    exactly once and with the upstream usage keys.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
         stdapi/models/chat/_mantle/_default.py:ChatModel._relay_stream
         stdapi/models/chat/_mantle/_default.py:ChatModel._observe_stream
    """

    @pytest.mark.parametrize("strip_usage_chunk", [False, True])
    async def test_converted_stream_records_usage_once(
        self, monkeypatch: pytest.MonkeyPatch, strip_usage_chunk: bool
    ) -> None:
        """A responses-to-chat converted stream bills exactly once.

        Usage is recorded from the raw responses-shaped events even when the
        client-facing usage chunk is stripped; relayed chunks keep their
        Chat Completions shape and untouched ``chatcmpl-`` IDs, and the
        stream ends with the ``[DONE]`` sentinel.
        """
        records = _capture_usage_records(monkeypatch)
        model = mantle_default.ChatModel("test.stream-model")
        events = [
            event
            async for event in model._relay_stream(  # noqa: SLF001
                "responses",
                "chat_completions",
                _fake_stream(_responses_stream_events()),
                "us-east-1",
                strip_usage_chunk=strip_usage_chunk,
            )
        ]
        assert len(records) == 1
        assert records[0]["input_tokens"] == 10
        assert records[0]["output_tokens"] == 5
        assert events[-1].data == "[DONE]"
        chunks = [loads(_event_data(event)) for event in events[:-1]]
        assert chunks
        assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
        assert all(chunk["id"] == "chatcmpl-native1" for chunk in chunks)
        has_usage_chunk = any(chunk.get("usage") for chunk in chunks)
        assert has_usage_chunk is not strip_usage_chunk

    async def test_seeded_id_rewrites_applied_to_relayed_events(self) -> None:
        """Seeded chained-ID rewrites replace native IDs in relayed events."""
        response = {
            "id": "resp_native123",
            "previous_response_id": "resp_native123",
            "model": "test.model",
        }
        model = mantle_default.ChatModel("test.stream-model")
        events = [
            event
            async for event in model._relay_stream(  # noqa: SLF001
                "responses",
                "responses",
                _fake_stream([("response.created", dumps({"response": response}))]),
                "us-east-1",
                strip_usage_chunk=False,
                id_rewrites={"resp_native123": "resp_public456"},
            )
        ]
        assert len(events) == 1
        assert events[0].event == "response.created"
        assert "resp_public456" in _event_data(events[0])
        assert "resp_native123" not in _event_data(events[0])


class TestObserveStreamChatCompletionsUsage:
    """Chat Completions stream usage is billed once, using the last cumulative value.

    Streamed usage chunks may repeat cumulative counters, so summing them would
    over-bill; billing is deferred to the end of the stream and uses the last
    value seen.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
         stdapi/models/chat/_mantle/_default.py:ChatModel._tap_usage
    """

    async def test_last_cumulative_usage_wins(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the final chunk's usage is recorded despite earlier cumulative values."""
        records = _capture_usage_records(monkeypatch)
        model = mantle_default.ChatModel("test.cumulative-usage-model")
        events: list[SseEvent] = [
            (
                None,
                dumps(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                ),
            ),
            (
                None,
                dumps(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 3,
                            "completion_tokens": 2,
                            "total_tokens": 5,
                        },
                    }
                ),
            ),
            (
                None,
                dumps(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 5,
                            "completion_tokens": 3,
                            "total_tokens": 8,
                        },
                    }
                ),
            ),
        ]
        async for _ in model._observe_stream(  # noqa: SLF001
            "chat_completions", _fake_stream(events), "us-east-1", {}
        ):
            pass
        assert len(records) == 1
        assert records[0]["input_tokens"] == 5
        assert records[0]["output_tokens"] == 3
        assert records[0]["total_tokens"] == 8


#: Web search output items as the Responses API reports them, one turn's worth.
_WEB_SEARCH_ITEMS: list[dict[str, Any]] = [
    {
        "type": "web_search_call",
        "id": "ws_1",
        "status": "completed",
        "action": {
            "type": "search",
            "query": "who won",
            "queries": ["who won", "final score"],
        },
    },
    {
        "type": "web_search_call",
        "id": "ws_2",
        "status": "completed",
        "action": {"type": "search", "query": "who won 2026"},
    },
    {
        "type": "web_search_call",
        "id": "ws_3",
        "status": "completed",
        "action": {"type": "open_page", "url": "https://example.invalid/report"},
    },
    {"type": "message", "id": "msg_1", "role": "assistant", "content": []},
]


class TestWebSearchQueryCounting:
    """Built-in web search is metered per query, not per tool call.

    One tool call may run several queries at once, and page reads are a
    separate operation with no published per-query rate, so neither counting
    the output items nor counting the tool calls matches what AWS bills.

    Every observed item reported ``status: "completed"``; whether a refused
    search (no ``bedrock-websearch:InvokeSearch`` permission) still reports the
    queries it attempted is unverified, so the count deliberately reads the
    action rather than the status.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html
         https://aws.amazon.com/bedrock/pricing/
         stdapi/aws_bedrock_mantle.py:web_search_queries
    """

    @pytest.mark.parametrize(
        ("index", "expected"), [(0, 2), (1, 1), (2, 0), (3, 0)], ids=str
    )
    def test_item_query_counts(self, index: int, expected: int) -> None:
        """Each output item contributes the number of queries it actually ran."""
        assert web_search_queries(_WEB_SEARCH_ITEMS[index]) == expected

    def test_response_totals_every_search_call(self) -> None:
        """A complete response sums the queries of all its search calls."""
        assert response_web_search_queries({"output": _WEB_SEARCH_ITEMS}) == 3

    def test_response_without_web_search_counts_nothing(self) -> None:
        """A turn that ran no search records no billable query."""
        assert response_web_search_queries({"output": [_WEB_SEARCH_ITEMS[-1]]}) == 0


class TestObserveStreamWebSearchUsage:
    """Streamed web search queries are billed from the item's terminal event.

    The query list is only complete on ``response.output_item.done``, which
    arrives before the usage event that ends the stream, so counting there
    bills a stream exactly like the equivalent non-streaming turn.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html
         stdapi/models/chat/_mantle/_default.py:_event_web_search_queries
    """

    @staticmethod
    def _capture(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, str]]:
        """Patch ``record_web_search_usage``, capturing (queries, region)."""
        calls: list[tuple[int, str]] = []
        monkeypatch.setattr(
            mantle_default,
            "record_web_search_usage",
            lambda queries, *, region: calls.append((queries, region)),
        )
        return calls

    async def test_streamed_queries_billed_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the terminal item event counts: not the added one, not the recap.

        A real stream repeats every output item in the ``response.completed``
        payload, so counting anything but ``response.output_item.done`` would
        bill each search twice.
        """
        calls = self._capture(monkeypatch)
        model = mantle_default.ChatModel("test.web-search-model")
        events: list[SseEvent] = [
            ("response.created", dumps({"response": {"id": "resp_native1"}}))
        ]
        for item in _WEB_SEARCH_ITEMS:
            events.append(("response.output_item.added", dumps({"item": item})))
            events.append(("response.output_item.done", dumps({"item": item})))
        events.append(
            (
                "response.completed",
                dumps(
                    {"response": {"id": "resp_native1", "output": _WEB_SEARCH_ITEMS}}
                ),
            )
        )
        async for _ in model._observe_stream(  # noqa: SLF001
            "responses", _fake_stream(events), "us-east-1", {}
        ):
            pass
        assert calls == [(3, "us-east-1")]

    async def test_abandoned_stream_still_bills_the_searches_it_ran(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stream cut short after a search still records that search."""
        calls = self._capture(monkeypatch)
        model = mantle_default.ChatModel("test.web-search-model")
        events: list[SseEvent] = [
            ("response.output_item.done", dumps({"item": _WEB_SEARCH_ITEMS[0]})),
            ("response.output_text.delta", dumps({"delta": "partial"})),
        ]
        stream = model._observe_stream(  # noqa: SLF001
            "responses", _fake_stream(events), "us-east-1", {}
        )
        await anext(stream)
        await stream.aclose()
        assert calls == [(2, "us-east-1")]


class TestObserveStreamMalformedFrames:
    """Malformed relayed frames are tolerated instead of crashing the stream.

    Ref: stdapi/models/chat/_mantle/_default.py:ChatModel._observe_stream
         stdapi/models/chat/_mantle/_default.py:_is_usage_chunk
    """

    async def test_malformed_usage_frame_relayed_without_crash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-JSON frame containing "usage" is relayed untouched, no usage taken."""
        records = _capture_usage_records(monkeypatch)
        model = mantle_default.ChatModel("test.malformed-frame-model")
        events: list[SseEvent] = [
            (None, 'not json but has "usage" in it'),
            (
                None,
                dumps(
                    {
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 1,
                            "completion_tokens": 1,
                            "total_tokens": 2,
                        },
                    }
                ),
            ),
        ]
        relayed = [
            event
            async for event in model._observe_stream(  # noqa: SLF001
                "chat_completions", _fake_stream(events), "us-east-1", {}
            )
        ]
        assert relayed == events
        assert len(records) == 1
        assert records[0]["total_tokens"] == 2

    def test_is_usage_chunk_returns_false_for_malformed_data(self) -> None:
        """A non-JSON payload containing "usage" is not treated as a usage chunk."""
        assert (
            mantle_default._is_usage_chunk('not json but has "usage" in it') is False  # noqa: SLF001
        )


class TestServeValidatedBilling:
    """Non-streaming conversion billing (:meth:`ChatModel._serve_validated`).

    Ref: stdapi/models/chat/_mantle/_default.py:ChatModel._serve_validated
         stdapi/models/chat/_mantle/_convert.py:convert_response
    """

    async def test_usage_recorded_from_upstream_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Usage is extracted in the upstream shape before wire conversion."""
        raw = {
            "id": "resp_full1",
            "object": "response",
            "created_at": 123,
            "model": "test.model",
            "status": "completed",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "hi"}]}
            ],
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "service_tier": "default",
        }

        async def fake_serve(
            self: mantle_default.ChatModel,  # noqa: ARG001
            inbound: str,  # noqa: ARG001
            payload: dict[str, Any],  # noqa: ARG001
            *,
            stream: bool,  # noqa: ARG001
            region: str | None = None,  # noqa: ARG001
        ) -> tuple[str, str, dict[str, Any]]:
            return "responses", "us-east-1", raw

        monkeypatch.setattr(mantle_default.ChatModel, "_serve", fake_serve)
        records = _capture_usage_records(monkeypatch)
        model = mantle_default.ChatModel("test.serve-model")
        api, region, out = await model._serve_validated(  # noqa: SLF001
            "chat_completions", {"messages": []}
        )
        assert (api, region) == ("responses", "us-east-1")
        assert len(records) == 1
        assert records[0]["input_tokens"] == 10
        assert records[0]["output_tokens"] == 5
        assert records[0]["tier"] == "default"
        assert out["object"] == "chat.completion"
        assert out["choices"][0]["message"]["content"] == "hi"
        assert out["usage"]["prompt_tokens"] == 10

    async def test_web_search_queries_billed_beside_the_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Web search is billed per query on top of the turn's tokens.

        The upstream usage block never reports the searches, so a turn billed
        from tokens alone under-reports its real cost.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html
             stdapi/models/chat/_mantle/_default.py:ChatModel._serve_validated
        """
        raw = {
            "id": "resp_ws1",
            "object": "response",
            "created_at": 123,
            "model": "test.model",
            "status": "completed",
            "output": _WEB_SEARCH_ITEMS,
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }

        async def fake_serve(
            self: mantle_default.ChatModel,  # noqa: ARG001
            inbound: str,  # noqa: ARG001
            payload: dict[str, Any],  # noqa: ARG001
            *,
            stream: bool,  # noqa: ARG001
            region: str | None = None,  # noqa: ARG001
        ) -> tuple[str, str, dict[str, Any]]:
            return "responses", "us-west-2", raw

        monkeypatch.setattr(mantle_default.ChatModel, "_serve", fake_serve)
        _capture_usage_records(monkeypatch)
        searches: list[tuple[int, str]] = []
        monkeypatch.setattr(
            mantle_default,
            "record_web_search_usage",
            lambda queries, *, region: searches.append((queries, region)),
        )
        model = mantle_default.ChatModel("test.serve-model")
        await model._serve_validated("responses", {"input": "hi"})  # noqa: SLF001
        assert searches == [(3, "us-west-2")]


def _capture_log_response_params(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Patch ``log_response_params`` in the Mantle chat module, capturing calls."""
    calls: list[object] = []

    def fake_log(response: object, exclude: object = None) -> object:  # noqa: ARG001
        calls.append(response)
        return response

    monkeypatch.setattr(mantle_default, "log_response_params", fake_log)
    return calls


class TestNonStreamResponseLogging:
    """Non-streaming Mantle serve paths log their response, like the classic adapters.

    Ref: stdapi/models/chat/_mantle/_default.py:ChatModel.create_completion
         stdapi/models/chat/_mantle/_default.py:ChatModel.create_text_completion
         stdapi/models/chat/_mantle/_default.py:ChatModel.create_response
    """

    async def test_create_completion_logs_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-streamed chat completion is passed through ``log_response_params``."""
        raw = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

        async def fake_invoke_api(
            self: mantle_default.ChatModel,  # noqa: ARG001
            api: str,  # noqa: ARG001
            payload: dict[str, Any],  # noqa: ARG001
            *,
            stream: bool,  # noqa: ARG001
            region: str | None = None,  # noqa: ARG001
        ) -> tuple[str, Any]:
            return "us-east-1", raw

        monkeypatch.setattr(mantle_default.ChatModel, "_invoke_api", fake_invoke_api)
        calls = _capture_log_response_params(monkeypatch)
        model = mantle_default.ChatModel("test.log-completion-model")
        request = ChatCompletionCreateParams.model_validate(
            {
                "model": "test.log-completion-model",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        result = await model.create_completion(request, "chatcmpl-1", 0)
        assert calls == [result]

    async def test_create_text_completion_logs_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-streamed legacy completion is passed through ``log_response_params``."""
        raw = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

        async def fake_invoke_api(
            self: mantle_default.ChatModel,  # noqa: ARG001
            api: str,  # noqa: ARG001
            payload: dict[str, Any],  # noqa: ARG001
            *,
            stream: bool,  # noqa: ARG001
            region: str | None = None,  # noqa: ARG001
        ) -> tuple[str, Any]:
            return "us-east-1", raw

        monkeypatch.setattr(mantle_default.ChatModel, "_invoke_api", fake_invoke_api)
        calls = _capture_log_response_params(monkeypatch)
        model = mantle_default.ChatModel("test.log-text-completion-model")
        request = CompletionCreateParams(
            model="test.log-text-completion-model", prompt="hi"
        )
        result = await model.create_text_completion(request, "cmpl-1", 0)
        assert calls == [result]

    async def test_create_response_logs_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-streamed response is passed through ``log_response_params``."""
        raw = {
            "id": "resp_1",
            "object": "response",
            "created_at": 123,
            "model": "test.log-response-model",
            "status": "completed",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }

        async def fake_invoke_api(
            self: mantle_default.ChatModel,  # noqa: ARG001
            api: str,  # noqa: ARG001
            payload: dict[str, Any],  # noqa: ARG001
            *,
            stream: bool,  # noqa: ARG001
            region: str | None = None,  # noqa: ARG001
        ) -> tuple[str, Any]:
            return "us-east-1", raw

        monkeypatch.setattr(mantle_default.ChatModel, "_invoke_api", fake_invoke_api)
        calls = _capture_log_response_params(monkeypatch)
        model = mantle_default.ChatModel("test.log-response-model")
        request = ResponseCreateParams(model="test.log-response-model", input="hi")
        result = await model.create_response(request, "resp_public", 0.0)
        assert calls == [result]


def _failed_response_raw(
    message: str = "The model produced an invalid tool call.",
) -> dict[str, Any]:
    """Build a terminal ``status="failed"`` Responses payload with empty output."""
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 123,
        "model": "test.failed-response-model",
        "status": "failed",
        "error": {"code": "server_error", "message": message},
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 1,
            "output_tokens": 0,
            "total_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }


class TestSyncResponseFailureSurfaced:
    """A synchronous Responses failure surfaces as an error, not a 200 empty body.

    Upstream reports a terminal ``status="failed"`` with the reason in ``error``
    and no usable output; a background request must keep that state for polling
    while a synchronous one must not answer 200 with empty output.

    Ref: https://developers.openai.com/api/docs/guides/background
         stdapi/models/chat/_mantle/_default.py:_failed_response_error
    """

    async def test_failed_status_raises_for_synchronous_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``status="failed"`` upstream result raises a 502 instead of relaying it."""
        raw = _failed_response_raw()

        async def fake_invoke_api(
            self: mantle_default.ChatModel,  # noqa: ARG001
            api: str,  # noqa: ARG001
            payload: dict[str, Any],  # noqa: ARG001
            *,
            stream: bool,  # noqa: ARG001
            region: str | None = None,  # noqa: ARG001
        ) -> tuple[str, Any]:
            return "us-east-1", raw

        monkeypatch.setattr(mantle_default.ChatModel, "_invoke_api", fake_invoke_api)
        model = mantle_default.ChatModel("test.failed-response-model")
        request = ResponseCreateParams(model="test.failed-response-model", input="hi")
        with pytest.raises(ApiError) as exc_info:
            await model.create_response(request, "resp_public", 0.0)
        assert exc_info.value.status == 502
        assert "invalid tool call" in str(exc_info.value)

    async def test_failed_status_relayed_for_background_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A background request keeps its failed terminal state (200) for polling."""
        raw = _failed_response_raw()

        async def fake_invoke_api(
            self: mantle_default.ChatModel,  # noqa: ARG001
            api: str,  # noqa: ARG001
            payload: dict[str, Any],  # noqa: ARG001
            *,
            stream: bool,  # noqa: ARG001
            region: str | None = None,  # noqa: ARG001
        ) -> tuple[str, Any]:
            return "us-east-1", raw

        monkeypatch.setattr(mantle_default.ChatModel, "_invoke_api", fake_invoke_api)
        model = mantle_default.ChatModel("test.failed-response-model")
        request = ResponseCreateParams(
            model="test.failed-response-model", input="hi", background=True
        )
        response = await model.create_response(request, "resp_public", 0.0)
        assert isinstance(response, Response)
        assert response.status == "failed"


async def _fake_invoke_api_demoting_responses(
    self: mantle_default.ChatModel,  # noqa: ARG001
    api: str,
    payload: dict[str, Any],  # noqa: ARG001
    *,
    stream: bool,  # noqa: ARG001
    region: str | None = None,  # noqa: ARG001
) -> tuple[str, Any]:
    """Fake ``_invoke_api``: rejects Responses, succeeds on Chat Completions."""
    if api == "responses":
        msg = "The model does not support the 'responses' API."
        raise MantleApiUnsupportedError(msg, status=400)
    return "us-east-1", {"id": "chatcmpl-1", "choices": [], "usage": None}


class TestServeStoreFallbackWarning:
    """``_serve`` drops ``store`` with a warning when it falls back off Responses.

    Native storage only exists on the Responses API; agent harnesses set
    ``store`` unconditionally, so demoting the binding drops it with a warning
    rather than failing the request.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
         stdapi/models/chat/_mantle/_default.py:ChatModel._serve
    """

    async def test_store_dropped_and_warned_on_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Falling back from Responses drops ``store`` and logs a single warning."""
        warnings: list[dict[str, Any]] = []

        def fake_log_error_details(*args: object, **kwargs: object) -> None:
            warnings.append({"args": args, "kwargs": kwargs})

        monkeypatch.setattr(mantle_default, "log_error_details", fake_log_error_details)
        monkeypatch.setattr(
            mantle_default.ChatModel, "_invoke_api", _fake_invoke_api_demoting_responses
        )
        model = mantle_default.ChatModel("test.store-fallback-model")
        payload: dict[str, Any] = {"model": "m", "input": "x", "store": True}
        api, _region, _result = await model._serve(  # noqa: SLF001
            "responses", payload, stream=False
        )
        assert api == "chat_completions"
        assert "store" not in payload
        assert len(warnings) == 1
        assert "store" in warnings[0]["args"][0]
        assert warnings[0]["kwargs"]["level"] == "warning"

    async def test_store_absent_no_warning_no_pop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No ``store`` key means no drop and no warning during the same fallback."""
        warnings: list[dict[str, Any]] = []

        def fake_log_error_details(*args: object, **kwargs: object) -> None:
            warnings.append({"args": args, "kwargs": kwargs})

        monkeypatch.setattr(mantle_default, "log_error_details", fake_log_error_details)
        monkeypatch.setattr(
            mantle_default.ChatModel, "_invoke_api", _fake_invoke_api_demoting_responses
        )
        model = mantle_default.ChatModel("test.store-fallback-model-nostore")
        payload: dict[str, Any] = {"model": "m", "input": "x"}
        api, _region, _result = await model._serve(  # noqa: SLF001
            "responses", payload, stream=False
        )
        assert api == "chat_completions"
        assert "store" not in payload
        assert warnings == []


class TestResponsesRouteGuards:
    """Stored-response route helpers guarding Mantle-form identifiers.

    The ID prefix is load-bearing: locally stored responses use ``resp-`` while
    region-tagged Mantle responses use ``resp_``, and Mantle decoding is gated
    on Mantle being enabled.

    Ref: stdapi/routes/openai_responses.py:_require_local_response_id
         stdapi/routes/openai_responses.py:_decode_mantle_id
    """

    def test_undecodable_mantle_form_id_is_not_found(self) -> None:
        """An undecodable ``resp_`` ID never reaches the local store (404)."""
        with pytest.raises(ApiError) as exc_info:
            _require_local_response_id("resp_notdecodable")
        assert exc_info.value.status == 404
        assert "resp_notdecodable" in str(exc_info.value)

    def test_decode_gated_on_mantle_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A well-formed tagged ID does not decode when Mantle is disabled."""
        public_id = encode_mantle_response_id(_mantle_region(), "resp_native")
        assert _decode_mantle_id(public_id) is not None
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_enabled", False)
        assert _decode_mantle_id(public_id) is None


class TestMantleCompactionItemGuard:
    """Compaction input items on the Mantle Responses passthrough payload.

    ``encrypted_content`` is opaque to the client but not to the gateway: a
    locally produced compaction item carries the gateway's own marker and is
    meaningless upstream, while an upstream-produced one must round-trip byte
    for byte.

    Ref: https://developers.openai.com/api/docs/guides/compaction
         stdapi/models/chat/_mantle/_convert.py:responses_payload
         stdapi/models/chat/_adapters/_openai_responses.py:encode_compaction_content
    """

    async def test_local_marker_prefixed_item_is_rejected(self) -> None:
        """A locally-produced compaction item fails with 400 before upstream."""
        request = ResponseCreateParams.model_validate(
            {
                "model": "ignored",
                "input": [
                    {
                        "type": "compaction",
                        "encrypted_content": encode_compaction_content("summary"),
                    },
                    {"role": "user", "content": "next question"},
                ],
            }
        )
        with pytest.raises(ApiError, match="Compact the conversation again") as exc:
            await mantle_convert.responses_payload(request, "model-id")
        assert exc.value.status == 400

    async def test_unmarked_item_passes_through_verbatim(self) -> None:
        """An upstream-produced compaction item is forwarded unchanged."""
        content = urlsafe_b64encode(b"opaque upstream ciphertext").decode()
        request = ResponseCreateParams.model_validate(
            {
                "model": "ignored",
                "input": [{"type": "compaction", "encrypted_content": content}],
            }
        )
        payload, region = await mantle_convert.responses_payload(request, "model-id")
        assert region is None
        (item,) = payload["input"]
        assert item["type"] == "compaction"
        assert item["encrypted_content"] == content


class TestMantleModerationParamGuard:
    """The stdapi ``moderation`` parameter on Mantle passthrough payloads.

    The gateway's ``moderation`` extension is implemented with Bedrock
    Guardrails, which are a bedrock-runtime feature: on a Mantle-served model
    it is rejected rather than silently ignored.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
         stdapi/models/chat/_mantle/_convert.py:chat_completions_payload
         stdapi/models/chat/_mantle/_convert.py:responses_payload
    """

    async def test_chat_moderation_param_is_rejected(self) -> None:
        """A Chat Completions request with ``moderation`` fails with 400."""
        request = ChatCompletionCreateParams.model_validate(
            {
                "model": "ignored",
                "messages": [{"role": "user", "content": "hi"}],
                "moderation": {"model": "gr123"},
            }
        )
        with pytest.raises(ApiError, match="not available with this model") as exc:
            await mantle_convert.chat_completions_payload(request, "model-id")
        assert exc.value.status == 400

    async def test_responses_moderation_param_is_rejected(self) -> None:
        """A Responses request with ``moderation`` fails with 400."""
        request = ResponseCreateParams.model_validate(
            {"model": "ignored", "input": "hi", "moderation": {"model": "gr123"}}
        )
        with pytest.raises(ApiError, match="not available with this model") as exc:
            await mantle_convert.responses_payload(request, "model-id")
        assert exc.value.status == 400

    async def test_chat_without_moderation_builds_the_payload(self) -> None:
        """Without ``moderation`` the chat payload still builds normally."""
        request = ChatCompletionCreateParams.model_validate(
            {"model": "ignored", "messages": [{"role": "user", "content": "hi"}]}
        )
        payload = await mantle_convert.chat_completions_payload(request, "model-id")
        assert payload["model"] == "model-id"
        assert "moderation" not in payload


class TestStreamErrorEvents:
    """In-band upstream stream errors raise a shaped 502 during conversion.

    A converted stream cannot relay a foreign-shaped error frame to the client,
    so the error is raised instead; detection is keyed on the frame carrying an
    ``error`` member, never on the word appearing in generated text.

    Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
         https://platform.claude.com/docs/en/build-with-claude/streaming
         stdapi/models/chat/_mantle/_convert.py:convert_stream
         stdapi/models/chat/_mantle/_convert.py:_stream_error_message
    """

    async def _consume(self, stream: AsyncGenerator[SseEvent]) -> None:
        """Drain *stream*, discarding its events."""
        async for _ in stream:
            pass

    async def test_responses_failed_event_raises(self) -> None:
        """A Responses ``response.failed`` event aborts the converted stream."""
        events = _fake_stream(
            [
                ("response.created", dumps({"response": {"id": "resp_1"}})),
                (
                    "response.failed",
                    dumps(
                        {
                            "type": "response.failed",
                            "response": {
                                "id": "resp_1",
                                "error": {
                                    "code": "server_error",
                                    "message": "upstream failed",
                                },
                            },
                        }
                    ),
                ),
            ]
        )
        stream = mantle_convert.convert_stream("responses", "chat_completions", events)
        with pytest.raises(MantleError) as exc_info:
            await self._consume(stream)
        assert exc_info.value.status == 502
        assert "upstream failed" in str(exc_info.value)

    async def test_responses_error_event_raises(self) -> None:
        """A Responses ``error`` event aborts the converted stream."""
        events = _fake_stream(
            [("error", dumps({"type": "error", "message": "boom", "code": "x"}))]
        )
        stream = mantle_convert.convert_stream("responses", "chat_completions", events)
        with pytest.raises(MantleError) as exc_info:
            await self._consume(stream)
        assert exc_info.value.status == 502
        assert "boom" in str(exc_info.value)

    async def test_unnamed_error_payload_raises(self) -> None:
        """A chat-shaped ``{"error": ...}`` data payload aborts the stream."""
        events = _fake_stream([(None, dumps({"error": {"message": "bad thing"}}))])
        stream = mantle_convert.convert_stream("responses", "chat_completions", events)
        with pytest.raises(MantleError) as exc_info:
            await self._consume(stream)
        assert exc_info.value.status == 502
        assert "bad thing" in str(exc_info.value)

    async def test_messages_error_event_raises(self) -> None:
        """An Anthropic ``error`` event aborts the converted stream."""
        events = _fake_stream(
            [
                (
                    "error",
                    dumps(
                        {
                            "type": "error",
                            "error": {"type": "overloaded_error", "message": "busy"},
                        }
                    ),
                )
            ]
        )
        stream = mantle_convert.convert_stream("messages", "chat_completions", events)
        with pytest.raises(MantleError) as exc_info:
            await self._consume(stream)
        assert exc_info.value.status == 502
        assert "busy" in str(exc_info.value)

    async def test_text_delta_containing_error_word_is_not_an_error(self) -> None:
        """Regular deltas whose text mentions "error" stream through fine."""
        text = 'the "error" word is fine'
        events = _fake_stream(
            [
                ("response.created", dumps({"response": {"id": "resp_1"}})),
                ("response.output_text.delta", dumps({"delta": text})),
            ]
        )
        stream = mantle_convert.convert_stream("responses", "chat_completions", events)
        chunks = [loads(data) async for _, data in stream]
        assert len(chunks) == 2
        assert chunks[-1]["choices"][0]["delta"]["content"] == text

    async def test_malformed_responses_frame_is_skipped_not_fatal(self) -> None:
        """A malformed frame interleaved in a converted Responses stream is skipped."""
        events = _fake_stream(
            [
                ("response.created", dumps({"response": {"id": "resp_1"}})),
                (None, "not json at all"),
                ("response.output_text.delta", dumps({"delta": "hi"})),
            ]
        )
        stream = mantle_convert.convert_stream("responses", "chat_completions", events)
        chunks = [loads(data) async for _, data in stream]
        assert len(chunks) == 2
        assert chunks[-1]["choices"][0]["delta"]["content"] == "hi"

    async def test_malformed_messages_frame_is_skipped_not_fatal(self) -> None:
        """A malformed frame interleaved in a converted Messages stream is skipped."""
        events = _fake_stream(
            [
                ("message_start", dumps({"message": {"id": "msg_1"}})),
                (None, "not json at all"),
                (
                    "content_block_delta",
                    dumps({"delta": {"type": "text_delta", "text": "hi"}}),
                ),
            ]
        )
        stream = mantle_convert.convert_stream("messages", "chat_completions", events)
        chunks = [loads(data) async for _, data in stream]
        assert len(chunks) == 2
        assert chunks[-1]["choices"][0]["delta"]["content"] == "hi"

    async def test_response_incomplete_event_emits_finish_and_usage(self) -> None:
        """A named ``response.incomplete`` event ends the converted stream like completed."""
        response = {
            "id": "resp_1",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "usage": {"input_tokens": 3, "output_tokens": 7, "total_tokens": 10},
        }
        events = _fake_stream(
            [
                ("response.created", dumps({"response": {"id": "resp_1"}})),
                ("response.incomplete", dumps({"response": response})),
            ]
        )
        stream = mantle_convert.convert_stream("responses", "chat_completions", events)
        chunks = [loads(data) async for _, data in stream]
        finish_chunk = next(c for c in chunks if c["choices"][0].get("finish_reason"))
        assert finish_chunk["choices"][0]["finish_reason"] == "length"
        usage_chunk = next(c for c in chunks if c.get("usage"))
        assert usage_chunk["usage"]["completion_tokens"] == 7


class TestConversionFieldLists:
    """Cross-shape field allow/strip lists.

    Ref: https://developers.openai.com/api/docs/guides/safety-best-practices#implement-safety-identifiers
         stdapi/models/chat/_mantle/_convert.py:_OPENAI_COMMON_FIELDS
         stdapi/models/chat/_mantle/_convert.py:_CHAT_EXTENSION_FIELDS
    """

    def test_safety_identifier_is_a_common_field(self) -> None:
        """``safety_identifier`` survives Chat Completions <-> Responses."""
        assert "safety_identifier" in mantle_convert._OPENAI_COMMON_FIELDS  # noqa: SLF001
        out = mantle_convert._chat_to_responses_request(  # noqa: SLF001
            {"model": "m", "messages": [], "safety_identifier": "caller-1"}
        )
        assert out["safety_identifier"] == "caller-1"

    async def test_store_stripped_from_chat_passthrough(self) -> None:
        """``store`` is stripped: chat completions are persisted locally."""
        assert "store" in mantle_convert._CHAT_EXTENSION_FIELDS  # noqa: SLF001
        request = ChatCompletionCreateParams.model_validate(
            {
                "model": "ignored",
                "messages": [{"role": "user", "content": "hi"}],
                "store": True,
            }
        )
        payload = await mantle_convert.chat_completions_payload(request, "model-id")
        assert "store" not in payload


class TestServiceTierAndEffortMapping:
    """service_tier and reasoning effort mapping across wire shapes.

    Only ``auto`` exists on both sides: Anthropic's request-side tier set is
    ``auto`` / ``standard_only`` while OpenAI's is ``auto`` / ``default`` /
    ``flex`` / ``priority``, so every other value is dropped rather than
    guessed. Reasoning effort maps onto Anthropic's ``output_config.effort``,
    whose ``format`` member Mantle rejects and is therefore never emitted.

    Ref: https://platform.claude.com/docs/en/api/messages
         https://developers.openai.com/api/docs/guides/reasoning
         stdapi/models/chat/_mantle/_convert.py:_chat_to_messages_request
         stdapi/models/chat/_mantle/_convert.py:_messages_to_chat_request
    """

    def test_auto_tier_kept_toward_messages(self) -> None:
        """OpenAI ``service_tier=auto`` maps verbatim to the Anthropic shape."""
        out = mantle_convert._chat_to_messages_request(  # noqa: SLF001
            {"model": "m", "messages": [], "service_tier": "auto"}
        )
        assert out["service_tier"] == "auto"

    @pytest.mark.parametrize("tier", ["flex", "default", "priority"])
    def test_other_tiers_dropped_toward_messages(self, tier: str) -> None:
        """Non-``auto`` OpenAI tiers have no Anthropic equivalent."""
        out = mantle_convert._chat_to_messages_request(  # noqa: SLF001
            {"model": "m", "messages": [], "service_tier": tier}
        )
        assert "service_tier" not in out

    def test_auto_tier_kept_toward_chat(self) -> None:
        """Anthropic ``service_tier=auto`` maps verbatim to the OpenAI shape."""
        out = mantle_convert._messages_to_chat_request(  # noqa: SLF001
            {"model": "m", "messages": [], "service_tier": "auto"}
        )
        assert out["service_tier"] == "auto"

    def test_standard_only_tier_dropped_toward_chat(self) -> None:
        """Anthropic ``standard_only`` has no OpenAI equivalent."""
        out = mantle_convert._messages_to_chat_request(  # noqa: SLF001
            {"model": "m", "messages": [], "service_tier": "standard_only"}
        )
        assert "service_tier" not in out

    @pytest.mark.parametrize(
        ("effort", "expected"),
        [
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "high"),
            ("max", "high"),
        ],
    )
    def test_effort_mapped_toward_messages(self, effort: str, expected: str) -> None:
        """OpenAI reasoning efforts map to Anthropic ``output_config.effort``."""
        out = mantle_convert._chat_to_messages_request(  # noqa: SLF001
            {"model": "m", "messages": [], "reasoning_effort": effort}
        )
        assert out["output_config"] == {"effort": expected}

    def test_none_effort_omitted_toward_messages(self) -> None:
        """``reasoning_effort=none`` emits no ``output_config``."""
        out = mantle_convert._chat_to_messages_request(  # noqa: SLF001
            {"model": "m", "messages": [], "reasoning_effort": "none"}
        )
        assert "output_config" not in out

    def test_effort_mapped_toward_chat(self) -> None:
        """Anthropic ``output_config.effort`` maps to ``reasoning_effort``."""
        out = mantle_convert._messages_to_chat_request(  # noqa: SLF001
            {"model": "m", "messages": [], "output_config": {"effort": "high"}}
        )
        assert out["reasoning_effort"] == "high"

    def test_json_schema_format_not_emitted_toward_messages(self) -> None:
        """``response_format`` is dropped: Mantle rejects ``output_config.format``."""
        out = mantle_convert._chat_to_messages_request(  # noqa: SLF001
            {
                "model": "m",
                "messages": [],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "r", "schema": {"type": "object"}},
                },
            }
        )
        assert "output_config" not in out

    def test_output_config_format_still_read_toward_chat(self) -> None:
        """Anthropic ``output_config.format`` still maps to ``response_format``."""
        out = mantle_convert._messages_to_chat_request(  # noqa: SLF001
            {
                "model": "m",
                "messages": [],
                "output_config": {
                    "effort": "low",
                    "format": {"type": "json_schema", "schema": {"type": "object"}},
                },
            }
        )
        assert out["response_format"]["type"] == "json_schema"
        assert out["reasoning_effort"] == "low"


class TestAnthropicServerTools:
    """Anthropic server tools rejection during conversion toward OpenAI shapes.

    Anthropic-hosted server tools have no Chat Completions or Responses
    equivalent, so a request mixing them with client tools is rejected instead
    of losing the server tool silently.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
         stdapi/models/chat/_mantle/_convert.py:_messages_to_chat_request
    """

    def test_server_tool_raises(self) -> None:
        """A server tool in the request fails with a clear 400."""
        with pytest.raises(ApiError, match="server tools") as exc_info:
            mantle_convert._messages_to_chat_request(  # noqa: SLF001
                {
                    "model": "m",
                    "messages": [],
                    "tools": [
                        {"type": "web_search_20250305", "name": "web_search"},
                        {"name": "custom", "input_schema": {"type": "object"}},
                    ],
                }
            )
        assert exc_info.value.status == 400

    def test_client_tools_convert_without_server_tools(self) -> None:
        """Plain client tools still convert when no server tool is present."""
        out = mantle_convert._messages_to_chat_request(  # noqa: SLF001
            {
                "model": "m",
                "messages": [],
                "tools": [{"name": "custom", "input_schema": {"type": "object"}}],
            }
        )
        assert out["tools"][0]["function"]["name"] == "custom"


class TestChatStreamAsTextCompletion:
    """SSE wrapper converting a Chat Completions stream to text-completion chunks.

    Mantle has no legacy completions endpoint, so the legacy route is always
    served as chat and re-shaped on the way out: the leading role-only chunk has
    no legacy equivalent and is dropped, while usage-only chunks keep their
    empty ``choices``.

    Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
         stdapi/models/chat/_mantle/_convert.py:chat_stream_as_text_completion
    """

    async def _collect(
        self, events: list[ServerSentEvent], completion_id: str = "cmpl-1"
    ) -> list[ServerSentEvent]:
        """Run *events* through the wrapper and collect the resulting events."""

        async def source() -> AsyncGenerator[ServerSentEvent]:
            for event in events:
                yield event

        return [
            event
            async for event in mantle_convert.chat_stream_as_text_completion(
                source(), completion_id
            )
        ]

    async def test_named_event_without_choices_or_usage_passes_through(self) -> None:
        """A named event carrying neither choices nor usage is relayed unchanged."""
        event = ServerSentEvent(data=dumps({"foo": "bar"}), event="error")
        assert await self._collect([event]) == [event]

    async def test_usage_only_chunk_emits_empty_choices(self) -> None:
        """A usage-only chunk becomes a text_completion chunk with empty choices."""
        usage = {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
        event = ServerSentEvent(
            data=dumps({"choices": [], "usage": usage}), event="usage-event"
        )
        [result] = await self._collect([event])
        assert result.event == "usage-event"
        payload = loads(_event_data(result))
        assert payload["choices"] == []
        assert payload["usage"] == usage

    async def test_unnamed_role_only_chunk_is_dropped(self) -> None:
        """An unnamed chunk with only a role delta (no content) is dropped."""
        event = ServerSentEvent(
            data=dumps({"choices": [{"index": 0, "delta": {"role": "assistant"}}]}),
            event=None,
        )
        assert await self._collect([event]) == []

    async def test_content_chunk_uses_given_completion_id(self) -> None:
        """A content chunk converts to a text_completion chunk with the given ID."""
        event = ServerSentEvent(
            data=dumps(
                {
                    "created": 100,
                    "model": "test.model",
                    "choices": [
                        {"index": 0, "delta": {"content": "hi"}, "finish_reason": None}
                    ],
                }
            ),
            event=None,
        )
        [result] = await self._collect([event], completion_id="cmpl-xyz")
        payload = loads(_event_data(result))
        assert payload["id"] == "cmpl-xyz"
        assert payload["object"] == "text_completion"
        assert payload["choices"][0]["text"] == "hi"


class TestScrubErrorEvent:
    """Passthrough error event scrubbing (:func:`_default._scrub_error_event`).

    Relayed error events reach the client verbatim, so an upstream message that
    names the task role's IAM ARN must be rewritten while the payload shape and
    error ``type`` stay intact.

    Ref: stdapi/models/chat/_mantle/_default.py:_scrub_error_event
         stdapi/utils.py:hide_security_details
    """

    def test_arn_redacted_in_error_message(self) -> None:
        """ARNs in an upstream error message are redacted before relaying."""
        data = dumps(
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "role arn:aws:iam::123456789012:role/x failed",
                },
            }
        )
        scrubbed = mantle_default._scrub_error_event(data)  # noqa: SLF001
        assert "arn:aws" not in scrubbed
        assert loads(scrubbed)["error"]["type"] == "api_error"

    def test_non_error_payload_unchanged(self) -> None:
        """Payloads without a top-level ``error`` are relayed verbatim."""
        data = dumps({"delta": 'contains the "error" word'})
        assert mantle_default._scrub_error_event(data) == data  # noqa: SLF001


def test_alias_collision_prefers_runtime_service() -> None:
    """A dual-named model keeps its bedrock-runtime ID for the shared alias.

    The same model can be listed on both endpoints under different IDs; the
    short alias resolves to the bedrock-runtime one so guardrails, Converse
    features and cross-Region profiles stay available by default.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
         stdapi/models/__init__.py:ModelBase.get_aliases
         stdapi/models/__init__.py:_order_ids_mantle_first
    """
    aliases = OpenAiGptChatModel.get_aliases(
        {
            "openai.gpt-oss-120b": _model_details(
                "openai.gpt-oss-120b", MANTLE_SERVICE
            ),
            "openai.gpt-oss-120b-1:0": _model_details("openai.gpt-oss-120b-1:0"),
            "openai.gpt-5.6-luna": _model_details(
                "openai.gpt-5.6-luna", MANTLE_SERVICE
            ),
        }
    )
    assert aliases["gpt-oss-120b"] == "openai.gpt-oss-120b-1:0"
    assert aliases["gpt-5.6-luna"] == "openai.gpt-5.6-luna"


def test_claude_alias_prefers_runtime_over_mantle_undated() -> None:
    """A Mantle-held direct alias yields to a competing bedrock-runtime dated model.

    Mantle lists Claude under undated IDs while bedrock-runtime uses dated ones;
    both derive the same short alias, and the dated bedrock-runtime ID wins so
    the alias keeps the Converse feature set.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
         stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel.get_aliases
    """
    mantle_id = "anthropic.claude-haiku-4-5"
    runtime_id = "anthropic.claude-haiku-4-5-20251001-v1:0"
    aliases = AnthropicClaudeChatModel.get_aliases(
        {
            mantle_id: _model_details(mantle_id, MANTLE_SERVICE),
            runtime_id: _model_details(runtime_id),
        }
    )
    assert aliases["claude-haiku-4-5"] == runtime_id
    assert aliases["claude-haiku-4-5-20251001"] == runtime_id


def test_claude_alias_uses_mantle_id_without_runtime_competitor() -> None:
    """A lone Mantle-held undated ID keeps the direct alias.

    Ref: stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel.get_aliases
    """
    mantle_id = "anthropic.claude-haiku-4-5"
    aliases = AnthropicClaudeChatModel.get_aliases(
        {mantle_id: _model_details(mantle_id, MANTLE_SERVICE)}
    )
    assert aliases["claude-haiku-4-5"] == mantle_id


def test_map_error_nonstring_message_coerced() -> None:
    """A structured (non-string) upstream message maps without raising.

    Ref: stdapi/aws_bedrock_mantle.py:_map_error
    """
    error = _map_error(400, '{"error": {"message": {"detail": "nested"}}}', "us-east-1")
    assert error.status == 400
    assert "nested" in str(error)


def test_scrub_error_event_responses_shapes() -> None:
    """Responses-shaped and response.failed error payloads are scrubbed.

    Ref: stdapi/models/chat/_mantle/_default.py:_scrub_error_event
    """
    flat = _scrub_error_event(
        '{"type": "error", "code": "server_error",'
        ' "message": "boom arn:aws:iam::123456789012:role/x"}'
    )
    assert "123456789012" not in flat
    nested = _scrub_error_event(
        '{"type": "response.failed", "response": {"error":'
        ' {"message": "boom arn:aws:iam::123456789012:role/x"}}}'
    )
    assert "123456789012" not in nested


def test_scrub_error_event_structured_message() -> None:
    """A structured (non-string) error message is serialized and scrubbed.

    Ref: stdapi/models/chat/_mantle/_default.py:_scrub_error_event
    """
    scrubbed = _scrub_error_event(
        '{"error": {"message": {"detail": "boom arn:aws:iam::123456789012:role/x"}}}'
    )
    assert "123456789012" not in scrubbed


class _StubRequest:
    """Minimal stand-in for ``fastapi.Request`` exposing only ``.headers``."""

    def __init__(self, headers: Mapping[str, str]) -> None:
        self.headers = headers


@contextmanager
def _request_context(headers: Mapping[str, str]) -> Iterator[None]:
    """Bind the ``REQUEST`` contextvar to a stub request carrying *headers*."""
    token = REQUEST.set(cast("Request", _StubRequest(headers)))
    try:
        yield
    finally:
        REQUEST.reset(token)


class TestServesViaMantleHeaderDispatch:
    """Per-request ``x-stdapi-service`` header dispatch to Bedrock Mantle.

    The header only selects between the two Bedrock inference endpoints for a
    model that both host; it is inert unless explicitly enabled, and can never
    reach a model missing from the Mantle catalog.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
         stdapi/models/chat/__init__.py:serves_via_mantle
    """

    def test_header_disabled_returns_false_even_with_matching_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The header is inert for a dual-homed model when the setting is off."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_service_header", False)
        model_id = "test.header-disabled-model"
        monkeypatch.setitem(
            MANTLE_MODELS, model_id, _model_details(model_id, MANTLE_SERVICE)
        )
        with _request_context({"x-stdapi-service": "bedrock-mantle"}):
            assert serves_via_mantle(model_id) is False

    def test_header_enabled_matching_header_and_mantle_model_returns_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A matching header routes a dual-homed model to Mantle."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_service_header", True)
        model_id = "test.header-enabled-model"
        monkeypatch.setitem(
            MANTLE_MODELS, model_id, _model_details(model_id, MANTLE_SERVICE)
        )
        with _request_context({"x-stdapi-service": "bedrock-mantle"}):
            assert serves_via_mantle(model_id) is True

    def test_wrong_header_value_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A header present with the wrong value does not route to Mantle."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_service_header", True)
        model_id = "test.header-wrong-value-model"
        monkeypatch.setitem(
            MANTLE_MODELS, model_id, _model_details(model_id, MANTLE_SERVICE)
        )
        with _request_context({"x-stdapi-service": "bedrock-runtime"}):
            assert serves_via_mantle(model_id) is False

    def test_model_not_in_mantle_models_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A matching header cannot route a model absent from the Mantle catalog."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_service_header", True)
        with _request_context({"x-stdapi-service": "bedrock-mantle"}):
            assert serves_via_mantle("test.not-in-mantle-catalog") is False

    def test_no_request_context_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no HTTP request bound, the header path never activates."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_service_header", True)
        model_id = "test.no-request-context-model"
        monkeypatch.setitem(
            MANTLE_MODELS, model_id, _model_details(model_id, MANTLE_SERVICE)
        )
        assert serves_via_mantle(model_id) is False

    def test_get_chat_model_classic_model_is_not_mantle(self) -> None:
        """A classic bedrock-runtime model id resolves outside the Mantle family.

        Ref: stdapi/models/chat/__init__.py:get_chat_model
        """
        model = get_chat_model("amazon.nova-micro-v1:0")
        assert not getattr(model, "IS_MANTLE", False)
        assert not isinstance(model, mantle_default.ChatModel)


class TestParallelToolCallsFalseAccepted:
    """``parallel_tool_calls: false`` is accepted for every model.

    Upstream never rejects the flag and the Responses API accepts it, so
    rejecting it on Chat Completions alone would refuse requests that are valid
    upstream and split the two sibling surfaces. Mantle honors it by mapping the
    flag onto Anthropic's ``disable_parallel_tool_use`` (covered in
    ``tests/test_aws_bedrock_mantle_convert.py``); models that cannot constrain
    tool use ignore it, and the client still sees which tool calls were made.

    Ref: https://developers.openai.com/api/docs/guides/function-calling#parallel-function-calling
         stdapi/types/openai_chat_completions.py:CompletionCreateParams
         stdapi/models/chat/_mantle/_convert.py:_anthropic_tool_choice_from_chat
    """

    def test_mantle_served_model_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model the Mantle catalog reports as Mantle-served accepts the flag."""
        model_id = "test.mantle-parallel-tool-calls-model"
        monkeypatch.setitem(
            stdapi_models._ALL_MODELS,  # noqa: SLF001
            model_id,
            _model_details(model_id, MANTLE_SERVICE),
        )
        request = ChatCompletionCreateParams(
            model=model_id,
            messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
            parallel_tool_calls=False,
        )
        assert request.parallel_tool_calls is False

    @pytest.mark.parametrize(
        "model",
        [
            pytest.param("amazon.nova-micro-v1:0", id="classic-bedrock-runtime"),
            pytest.param("test.unregistered-model", id="absent-from-the-catalog"),
        ],
    )
    def test_non_mantle_model_is_accepted(self, model: str) -> None:
        """A model that cannot honor the flag accepts it rather than failing.

        The response still reports the tool calls the model made, so a client
        that depends on sequential tool use can detect that it did not get it --
        which is what makes ignoring the flag safe rather than silent.
        """
        request = ChatCompletionCreateParams(
            model=model,
            messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
            parallel_tool_calls=False,
        )
        assert request.parallel_tool_calls is False

    def test_true_is_unchanged(self) -> None:
        """The default value keeps round-tripping untouched."""
        request = ChatCompletionCreateParams(
            model="amazon.nova-micro-v1:0",
            messages=[ChatCompletionUserMessageParam(role="user", content="Hello")],
            parallel_tool_calls=True,
        )
        assert request.parallel_tool_calls is True


class TestPreviousResponseIdFallback:
    """Serving demoted off the Responses API rejects a chained conversation.

    The stored history behind ``previous_response_id`` lives upstream on the
    Responses API only, so converting the request to another API would silently
    drop the conversation; the request fails instead.

    Ref: https://developers.openai.com/api/docs/guides/conversation-state#passing-context-from-the-previous-response
         stdapi/models/chat/_mantle/_default.py:ChatModel._serve
    """

    async def test_previous_response_id_cannot_be_honored_on_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Demoting off Responses with ``previous_response_id`` set fails with 400."""

        async def fake_invoke_api(
            self: mantle_default.ChatModel,  # noqa: ARG001
            api: str,
            payload: dict[str, Any],  # noqa: ARG001
            *,
            stream: bool,  # noqa: ARG001
            region: str | None = None,  # noqa: ARG001
        ) -> tuple[str, Any]:
            if api == "responses":
                msg = "The model does not support the 'responses' API."
                raise MantleApiUnsupportedError(msg, status=400)
            msg = "must not fall back past the guard"
            raise AssertionError(msg)  # pragma: no cover

        monkeypatch.setattr(mantle_default.ChatModel, "_invoke_api", fake_invoke_api)
        model = mantle_default.ChatModel("test.previous-response-fallback-model")
        payload: dict[str, Any] = {
            "model": "m",
            "input": "hi",
            "previous_response_id": "resp_abc",
        }
        with pytest.raises(ApiError) as exc_info:
            await model._serve("responses", payload, stream=False)  # noqa: SLF001
        assert exc_info.value.status == 400
        assert "previous_response_id cannot be honored" in str(exc_info.value)


class TestLearnedBindingSkipsSecondProbe:
    """A learned API binding is reused directly, skipping the failed probe.

    Ref: stdapi/models/chat/_mantle/_default.py:_LEARNED_APIS
         stdapi/models/chat/_mantle/_default.py:ChatModel._serve
    """

    async def test_learned_binding_used_directly_on_second_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After learning chat_completions, a later call skips the responses probe."""
        calls: list[str] = []

        async def fake_invoke_api(
            self: mantle_default.ChatModel,  # noqa: ARG001
            api: str,
            payload: dict[str, Any],  # noqa: ARG001
            *,
            stream: bool,  # noqa: ARG001
            region: str | None = None,  # noqa: ARG001
        ) -> tuple[str, Any]:
            calls.append(api)
            if api == "responses":
                msg = "The model does not support the 'responses' API."
                raise MantleApiUnsupportedError(msg, status=400)
            return "us-east-1", {"id": "chatcmpl-1", "choices": [], "usage": None}

        monkeypatch.setattr(mantle_default.ChatModel, "_invoke_api", fake_invoke_api)
        model_id = "test.learned-binding-probe-model"
        model = mantle_default.ChatModel(model_id)
        api, _region, _raw = await model._serve(  # noqa: SLF001
            "responses", {"model": model_id, "input": "hi"}, stream=False
        )
        assert api == "chat_completions"
        assert calls == ["responses", "chat_completions"]
        assert mantle_default._LEARNED_APIS[model_id] == frozenset(  # noqa: SLF001
            {"chat_completions"}
        )

        calls.clear()
        api2, _region2, _raw2 = await model._serve(  # noqa: SLF001
            "responses", {"model": model_id, "input": "hi"}, stream=False
        )
        assert api2 == "chat_completions"
        assert calls == ["chat_completions"]


class TestMantleDisabled:
    """Behavior when Bedrock Mantle support is disabled.

    Ref: stdapi/models/__init__.py:_merge_mantle_models
         stdapi/models/chat/__init__.py:serves_via_mantle
         stdapi/config.py:_Settings
    """

    async def test_merge_skips_catalog_fetch_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Model collection never starts Mantle discovery when Mantle is disabled.

        ``_collect_all_models`` is the seam that decides whether the Mantle
        discovery task is created at all; with the setting off, the collector
        must never be called.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_enabled", False)

        async def fail_if_called(
            failed_regions: dict[str, str],  # noqa: ARG001
            regions_without_endpoint: dict[str, str],  # noqa: ARG001
        ) -> dict[str, ModelDetails]:
            msg = "must not be called when Mantle is disabled"
            raise AssertionError(msg)

        async def no_candidates(
            failed_regions: dict[str, str],  # noqa: ARG001
        ) -> dict[str, Any]:
            return {}

        async def no_models(
            candidates: dict[str, Any],  # noqa: ARG001
            unavailable_models: dict[str, dict[str, list[str]]],  # noqa: ARG001
        ) -> dict[str, ModelDetails]:
            return {}

        monkeypatch.setattr(stdapi_models, "_collect_mantle_models", fail_if_called)
        monkeypatch.setattr(stdapi_models, "_collect_region_candidates", no_candidates)
        monkeypatch.setattr(stdapi_models, "_check_candidates", no_models)
        all_models, _ = await stdapi_models._collect_all_models({}, {}, {})  # noqa: SLF001
        assert all_models == {}

    def test_serves_via_mantle_false_when_disabled_and_catalog_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With Mantle disabled (and its catalog never populated), nothing dispatches.

        ``serves_via_mantle`` has no explicit disablement check of its own; the
        guarantee comes from ``_merge_mantle_models`` (tested above) never
        populating ``MANTLE_MODELS``/the model catalog while disabled.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_enabled", False)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_service_header", False)
        monkeypatch.setattr(stdapi_models, "MANTLE_MODELS", {})
        assert serves_via_mantle("test.anything-model") is False


class TestCollectAllModelsCancellation:
    """Cancelling model collection also cancels the in-flight Mantle task.

    Ref: stdapi/models/__init__.py:_collect_all_models
    """

    async def test_cancelled_collection_cancels_the_mantle_task(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cancel while awaiting the Mantle catalog stops the fetch task too.

        Awaiting a task never forwards cancellation into it, so without the
        explicit cancel-and-await cleanup the Mantle catalog fetch would keep
        running detached after the caller (e.g. a shutting-down lifespan) is
        cancelled, ending as an un-retrieved task warning.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_enabled", True)
        fetch_started = Event()
        fetch_cancelled = Event()

        async def blocking_mantle(
            failed_regions: dict[str, str],  # noqa: ARG001
            regions_without_endpoint: dict[str, str],  # noqa: ARG001
        ) -> dict[str, ModelDetails]:
            fetch_started.set()
            try:
                await Event().wait()
            except CancelledError:
                fetch_cancelled.set()
                raise
            return {}

        async def no_candidates(
            failed_regions: dict[str, str],  # noqa: ARG001
        ) -> dict[str, Any]:
            return {}

        async def no_models(
            candidates: dict[str, Any],  # noqa: ARG001
            unavailable_models: dict[str, dict[str, list[str]]],  # noqa: ARG001
        ) -> dict[str, ModelDetails]:
            return {}

        monkeypatch.setattr(stdapi_models, "_collect_mantle_models", blocking_mantle)
        monkeypatch.setattr(stdapi_models, "_collect_region_candidates", no_candidates)
        monkeypatch.setattr(stdapi_models, "_check_candidates", no_models)

        collection = create_task(stdapi_models._collect_all_models({}, {}, {}))  # noqa: SLF001
        await wait_for(fetch_started.wait(), timeout=5)
        collection.cancel()
        with pytest.raises(CancelledError):
            await collection
        await wait_for(fetch_cancelled.wait(), timeout=5)


class TestMantleRegionsPinning:
    """Region candidate selection: pinned region vs. catalog vs. default list.

    A stored response can only be read in the Region that created it, so a
    pinned Region must short-circuit the candidate list instead of being merely
    preferred.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
         stdapi/models/chat/_mantle/_default.py:ChatModel._mantle_regions
    """

    def test_pinned_region_is_returned_as_sole_candidate(self) -> None:
        """A pinned region short-circuits to a single-element candidate list."""
        model = mantle_default.ChatModel("test.unknown-region-model")
        assert model._mantle_regions("eu-west-1") == [  # noqa: SLF001
            "eu-west-1"
        ]

    def test_unpinned_known_model_uses_its_catalog_regions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unpinned request for a cataloged model uses its own region list."""
        model_id = "test.catalog-regions-model"
        regions: list[RegionName] = ["eu-west-1", "ap-northeast-1"]
        monkeypatch.setitem(
            MANTLE_MODELS,
            model_id,
            make_model_details(
                model_id, provider="Test", service=MANTLE_SERVICE, regions=regions
            ),
        )
        model = mantle_default.ChatModel(model_id)
        assert model._mantle_regions(None) == regions  # noqa: SLF001

    def test_unpinned_unknown_model_falls_back_to_settings_regions(self) -> None:
        """An unpinned request for an uncataloged model uses the configured default."""
        model = mantle_default.ChatModel("test.totally-unknown-region-model")
        assert (
            model._mantle_regions(None) == SETTINGS.aws_bedrock_mantle_regions  # noqa: SLF001
        )


class TestRouteAndExecuteMantleFailover:
    """``MantleError.failover`` drives (or blocks) cross-region retry directly.

    Throttling and capacity errors are the only ones worth another Region; a
    400 is deterministic and retrying it would multiply the latency and the
    upstream load for the same failure.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html
         stdapi/models/__init__.py:route_and_execute
    """

    async def test_failover_error_retries_next_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failover-eligible ``MantleError`` is retried by a further call."""
        # Without a router every request is region-pinned, whatever the
        # candidate list holds, and no failover can happen at all.
        monkeypatch.setattr(stdapi_models, "REGION_ROUTER", RegionRouter())
        calls: list[RegionName] = []

        async def fn(region: RegionName) -> str:
            calls.append(region)
            if len(calls) == 1:
                msg = "busy"
                raise MantleError(msg, status=429, failover=True)
            return "ok"

        candidates: list[RegionName] = ["us-east-1", "us-west-2"]
        result = await stdapi_models.route_and_execute(
            "test.route-failover-model", candidates, fn
        )
        assert result == "ok"
        assert len(calls) == 2

    async def test_non_failover_error_raises_after_single_call(self) -> None:
        """A non-failover ``MantleError`` is not retried in another region."""
        calls: list[RegionName] = []

        async def fn(region: RegionName) -> str:
            calls.append(region)
            msg = "bad request"
            raise MantleError(msg, status=400, failover=False)

        candidates: list[RegionName] = ["us-east-1", "us-west-2"]
        with pytest.raises(MantleError) as exc_info:
            await stdapi_models.route_and_execute(
                "test.route-non-failover-model", candidates, fn
            )
        assert exc_info.value.status == 400
        assert len(calls) == 1


class TestInvokeApiSurfaceLearningWritePath:
    """The learned-surface cache self-heals and is then reused on later calls.

    Nothing upstream reports whether a model answers on ``/v1`` or
    ``/openai/v1``, so the surface is discovered by probing and remembered per
    model to keep the extra round trip to the first request only.

    Ref: stdapi/models/chat/_mantle/_default.py:_LEARNED_SURFACE
         stdapi/models/chat/_mantle/_default.py:ChatModel._invoke_api
    """

    async def test_first_surface_fails_second_succeeds_and_is_learned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A rejected first surface self-heals via the alternate, learning it."""
        model_id = "test.surface-write-model"
        calls: list[str] = []

        async def fake_invoke(
            region: RegionName,  # noqa: ARG001
            path: str,
            payload: Mapping[str, Any],  # noqa: ARG001
            *,
            single_region: bool,  # noqa: ARG001
            headers: Mapping[str, str] | None = None,  # noqa: ARG001
        ) -> dict[str, Any]:
            calls.append(path)
            if path.startswith("/openai/v1"):
                msg = "This model isn't supported on this route."
                raise MantleSurfaceUnsupportedError(msg, status=400)
            return {"id": "chatcmpl-1", "choices": [], "usage": None}

        monkeypatch.setattr(mantle_default, "invoke", fake_invoke)
        model = mantle_default.ChatModel(model_id)
        _region, result = await model._invoke_api(  # noqa: SLF001
            "chat_completions", {"model": model_id, "messages": []}, stream=False
        )
        assert result == {"id": "chatcmpl-1", "choices": [], "usage": None}
        assert calls == ["/openai/v1/chat/completions", "/v1/chat/completions"]
        assert mantle_default._LEARNED_SURFACE[model_id] == "/v1"  # noqa: SLF001

        # The learned surface is tried first on the next call and succeeds at once:
        # the rejected surface is skipped rather than probed again.
        calls.clear()
        await model._invoke_api(  # noqa: SLF001
            "chat_completions", {"model": model_id, "messages": []}, stream=False
        )
        assert calls == ["/v1/chat/completions"]


class TestInvokeApiRetriesWhenRouterDisabled:
    """Router-disabled multi-region calls still get in-region retries.

    Ref: stdapi/models/chat/_mantle/_default.py:ChatModel._invoke_api
         stdapi/aws_bedrock_mantle.py:_request_with_retry
         stdapi/models/__init__.py:route_and_execute
    """

    async def test_router_disabled_multi_region_retries_in_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the region router disabled and >1 candidate, failures are retried.

        ``route_and_execute`` calls only the first candidate region once when
        the router is disabled, so the in-region retry in
        ``_request_with_retry`` must cover it instead of silently returning
        zero retries.
        """
        monkeypatch.setattr(mantle_default, "REGION_ROUTER", None)
        monkeypatch.setattr(stdapi_models, "REGION_ROUTER", None)
        regions: list[RegionName] = ["us-east-1", "us-west-2"]
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_regions", regions)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_max_retries", 2)
        calls = 0

        async def fake_request(
            region: RegionName,  # noqa: ARG001
            path: str,  # noqa: ARG001
            body: bytes | None,  # noqa: ARG001
            headers: Mapping[str, str] | None,  # noqa: ARG001
            method: str = "POST",  # noqa: ARG001
        ) -> ClientResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                msg = "busy"
                raise MantleError(msg, status=503, failover=True)
            return cast("ClientResponse", "ok")

        async def fake_sleep(delay: float) -> None:
            pass

        async def fake_read_json(response: ClientResponse) -> dict[str, Any]:  # noqa: ARG001
            return {"id": "chatcmpl-1", "choices": [], "usage": None}

        monkeypatch.setattr(aws_bedrock_mantle, "_request", fake_request)
        monkeypatch.setattr(aws_bedrock_mantle, "sleep", fake_sleep)
        monkeypatch.setattr(aws_bedrock_mantle, "_read_json", fake_read_json)
        model = mantle_default.ChatModel("test.router-disabled-model")
        _region, result = await model._invoke_api(  # noqa: SLF001
            "chat_completions", {"model": "m", "messages": []}, stream=False
        )
        assert result == {"id": "chatcmpl-1", "choices": [], "usage": None}
        assert calls == 2


class TestInvokeApiSurfaceLearning:
    """Surface-learning side effects of ``_invoke_api``.

    Only the OpenAI-shaped APIs are surface-relative: the Messages API has one
    absolute path, so serving it must not record a surface, and a model rejected
    on both OpenAI surfaces must not leave a learned one behind either.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html
         stdapi/models/chat/_mantle/_default.py:ChatModel._invoke_api
    """

    async def test_messages_api_success_does_not_write_learned_surface(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A successful Messages-API call never touches the OpenAI surface cache."""

        async def fake_invoke(
            region: RegionName,  # noqa: ARG001
            path: str,  # noqa: ARG001
            payload: Mapping[str, Any],  # noqa: ARG001
            *,
            single_region: bool,  # noqa: ARG001
            headers: Mapping[str, str] | None = None,  # noqa: ARG001
        ) -> dict[str, Any]:
            return {"id": "msg_1", "type": "message"}

        monkeypatch.setattr(mantle_default, "invoke", fake_invoke)
        model_id = "test.messages-surface-model"
        model = mantle_default.ChatModel(model_id)
        _region, result = await model._invoke_api(  # noqa: SLF001
            "messages", {"model": model_id, "messages": []}, stream=False
        )
        assert result == {"id": "msg_1", "type": "message"}
        assert model_id not in mantle_default._LEARNED_SURFACE  # noqa: SLF001

    async def test_both_surfaces_unsupported_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every OpenAI surface rejects the model, the last error propagates."""
        model_id = "test.both-surfaces-unsupported-model"
        calls: list[str] = []

        async def fake_invoke(
            region: RegionName,  # noqa: ARG001
            path: str,
            payload: Mapping[str, Any],  # noqa: ARG001
            *,
            single_region: bool,  # noqa: ARG001
            headers: Mapping[str, str] | None = None,  # noqa: ARG001
        ) -> dict[str, Any]:
            calls.append(path)
            msg = "This model isn't supported on this route."
            raise MantleSurfaceUnsupportedError(msg, status=400)

        monkeypatch.setattr(mantle_default, "invoke", fake_invoke)
        model = mantle_default.ChatModel(model_id)
        with pytest.raises(MantleSurfaceUnsupportedError) as exc_info:
            await model._invoke_api(  # noqa: SLF001
                "chat_completions", {"model": "m", "messages": []}, stream=False
            )
        assert exc_info.value.status == 400
        assert "isn't supported on this route" in str(exc_info.value)
        assert calls == ["/openai/v1/chat/completions", "/v1/chat/completions"]
        assert model_id not in mantle_default._LEARNED_SURFACE  # noqa: SLF001


class TestCreateResponseModerationBuilder:
    """The ``moderation_builder`` callback sets the non-streamed response moderation.

    Ref: stdapi/models/chat/_mantle/_default.py:ChatModel.create_response
    """

    async def test_moderation_builder_result_attached_to_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-streamed response gets its moderation field from the builder."""
        raw = {
            "id": "resp_1",
            "object": "response",
            "created_at": 123,
            "model": "test.moderation-model",
            "status": "completed",
            "output": [],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }

        async def fake_invoke_api(
            self: mantle_default.ChatModel,  # noqa: ARG001
            api: str,  # noqa: ARG001
            payload: dict[str, Any],  # noqa: ARG001
            *,
            stream: bool,  # noqa: ARG001
            region: str | None = None,  # noqa: ARG001
        ) -> tuple[str, Any]:
            return "us-east-1", raw

        monkeypatch.setattr(mantle_default.ChatModel, "_invoke_api", fake_invoke_api)
        _capture_usage_records(monkeypatch)
        model = mantle_default.ChatModel("test.moderation-model")
        request = ResponseCreateParams(model="test.moderation-model", input="hi")
        result = ModerationResult(
            flagged=False,
            categories={},
            category_scores={},
            category_applied_input_types={},
            model="gr123",
        )
        moderation = ResponseModeration(input=result, output=result)
        response = await model.create_response(
            request, "resp_public", 0.0, moderation_builder=lambda: moderation
        )
        assert isinstance(response, Response)
        assert response.moderation is moderation


class TestCreateResponseConvertedId:
    """Converted (non-native) responses carry the server-assigned response ID.

    A model served by Chat Completions has no upstream stored response, so the
    result must keep the route's local-store ``resp-`` ID: returning the
    upstream ``chatcmpl-`` ID would make GET/DELETE and chaining unusable.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/retrieve
         stdapi/models/chat/_mantle/_default.py:ChatModel.create_response
    """

    async def test_converted_response_uses_local_response_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A chat-served model returns the route ID so local storage works."""
        raw = {
            "id": "chatcmpl-upstream",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

        async def fake_invoke_api(
            self: mantle_default.ChatModel,  # noqa: ARG001
            api: str,
            payload: dict[str, Any],  # noqa: ARG001
            *,
            stream: bool,  # noqa: ARG001
            region: str | None = None,  # noqa: ARG001
        ) -> tuple[str, Any]:
            assert api == "chat_completions"
            return "us-east-1", raw

        monkeypatch.setattr(OpenWeightChatModel, "_invoke_api", fake_invoke_api)
        model = OpenWeightChatModel("qwen.converted-id-model")
        request = ResponseCreateParams(model="qwen.converted-id-model", input="hi")
        response = await model.create_response(request, "resp-localstore1", 0.0)
        assert isinstance(response, Response)
        assert response.id == "resp-localstore1"

    async def test_local_previous_response_id_stripped_for_converted_serving(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A restored local previous_response_id does not block chat-served models.

        The route merges a local store ID's conversation inline and restores
        the ID on the request only for response echoing: it must be stripped
        before the payload build instead of being rejected as non-Mantle.
        """
        raw = {
            "id": "chatcmpl-upstream",
            "object": "chat.completion",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

        async def fake_invoke_api(
            self: mantle_default.ChatModel,  # noqa: ARG001
            api: str,
            payload: dict[str, Any],
            *,
            stream: bool,  # noqa: ARG001
            region: str | None = None,  # noqa: ARG001
        ) -> tuple[str, Any]:
            assert api == "chat_completions"
            assert "previous_response_id" not in payload
            return "us-east-1", raw

        monkeypatch.setattr(OpenWeightChatModel, "_invoke_api", fake_invoke_api)
        model = OpenWeightChatModel("qwen.converted-id-model")
        request = ResponseCreateParams(
            model="qwen.converted-id-model",
            input="What number did I ask you to remember?",
            previous_response_id="resp-abc123def",
        )
        response = await model.create_response(request, "resp-localstore2", 0.0)
        assert isinstance(response, Response)
        assert response.id == "resp-localstore2"


class TestStreamWrapBranches:
    """Streaming legacy-completion and message routes reach their wrap branch.

    Neither route exists on Mantle for the default family: the legacy prompt is
    converted to messages and both are served on the probed OpenAI surface with
    streaming forced on upstream.

    Ref: stdapi/models/chat/_mantle/_default.py:ChatModel._stream_serve
         stdapi/models/chat/_mantle/_convert.py:text_completion_as_chat_payload
         stdapi/models/chat/_mantle/_convert.py:enable_stream_usage
    """

    async def test_create_text_completion_stream_returns_event_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A streaming legacy completion returns a wrapped ``EventSourceResponse``."""
        sent: list[tuple[str, Mapping[str, Any]]] = []

        async def fake_invoke_stream(
            region: RegionName,  # noqa: ARG001
            path: str,
            payload: Mapping[str, Any],
            *,
            single_region: bool,  # noqa: ARG001
            headers: Mapping[str, str] | None = None,  # noqa: ARG001
        ) -> AsyncGenerator[SseEvent]:
            sent.append((path, payload))
            return _fake_stream(
                [
                    (
                        None,
                        dumps(
                            {
                                "id": "chatcmpl-1",
                                "created": 1,
                                "model": "test.model",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": "hi"},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        ),
                    )
                ]
            )

        monkeypatch.setattr(mantle_default, "invoke_stream", fake_invoke_stream)
        model = mantle_default.ChatModel("test.text-completion-stream-model")
        request = CompletionCreateParams(
            model="test.text-completion-stream-model", prompt="hi", stream=True
        )
        result = await model.create_text_completion(request, "cmpl-1", 0)
        assert isinstance(result, EventSourceResponse)
        (path, payload) = sent[0]
        assert path == "/openai/v1/responses"
        assert payload["stream"] is True
        assert "prompt" not in payload, (
            "the legacy prompt is converted, never forwarded"
        )
        assert payload["input"] == [{"role": "user", "content": "hi"}]

    async def test_create_message_stream_returns_event_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A streaming Anthropic message request returns an ``EventSourceResponse``."""
        sent: list[tuple[str, Mapping[str, Any]]] = []

        async def fake_invoke_stream(
            region: RegionName,  # noqa: ARG001
            path: str,
            payload: Mapping[str, Any],
            *,
            single_region: bool,  # noqa: ARG001
            headers: Mapping[str, str] | None = None,  # noqa: ARG001
        ) -> AsyncGenerator[SseEvent]:
            sent.append((path, payload))
            return _fake_stream(
                [("message_start", dumps({"message": {"id": "msg_1"}}))]
            )

        monkeypatch.setattr(mantle_default, "invoke_stream", fake_invoke_stream)
        model = mantle_default.ChatModel("test.message-stream-model")
        request = MessageCreateParams(
            model="test.message-stream-model",
            messages=[MessageParam(role="user", content="hi")],
            stream=True,
        )
        result = await model.create_message(request, "msg_public")
        assert isinstance(result, EventSourceResponse)
        (path, payload) = sent[0]
        assert path == "/openai/v1/responses"
        assert payload["stream"] is True


class TestStreamedResponseIdPlumbing:
    """A converted Responses stream carries the route-assigned response ID.

    A converted stream has no upstream stored response, so minting a ``resp_``
    ID would advertise a region-tagged Mantle object that does not exist; the
    route's local ``resp-`` ID is stamped on the synthesized events instead.

    Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
         stdapi/models/chat/_mantle/_default.py:ChatModel.create_response
         stdapi/models/chat/_mantle/_convert.py:convert_stream
    """

    async def test_create_response_stream_uses_the_route_id(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The streamed ID is the route-assigned one, not a minted ``resp_`` ID."""

        async def fake_invoke_stream(
            region: RegionName,  # noqa: ARG001
            path: str,  # noqa: ARG001
            payload: Mapping[str, Any],  # noqa: ARG001
            *,
            single_region: bool,  # noqa: ARG001
            headers: Mapping[str, str] | None = None,  # noqa: ARG001
        ) -> AsyncGenerator[SseEvent]:
            return _fake_stream(
                [
                    (
                        None,
                        dumps(
                            {
                                "id": "chatcmpl-native1",
                                "created": 1,
                                "model": "qwen.stream-response-id-model",
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": "hi"},
                                        "finish_reason": "stop",
                                    }
                                ],
                                "usage": {
                                    "prompt_tokens": 10,
                                    "completion_tokens": 5,
                                    "total_tokens": 15,
                                },
                            }
                        ),
                    )
                ]
            )

        monkeypatch.setattr(mantle_default, "invoke_stream", fake_invoke_stream)
        model = OpenWeightChatModel("qwen.stream-response-id-model")
        request = ResponseCreateParams(
            model="qwen.stream-response-id-model", input="hi", stream=True
        )
        result = await model.create_response(request, "resp-localstore3", 0.0)
        assert isinstance(result, EventSourceResponse)
        token = REQUEST_ID.set("req-stream-id")
        try:
            events = cast(
                "list[ServerSentEvent]", [event async for event in result.body_iterator]
            )
        finally:
            REQUEST_ID.reset(token)
        assert [event.event for event in events][:2] == [
            "response.created",
            "response.in_progress",
        ]
        assert all("resp_" not in _event_data(event) for event in events)
        assert {
            loads(_event_data(event))["response"]["id"]
            for event in events
            if "response" in loads(_event_data(event))
        } == {"resp-localstore3"}


class TestScrubErrorEventResidualBranches:
    """Residual ``_scrub_error_event`` shapes not covered by the classes above.

    Ref: stdapi/models/chat/_mantle/_default.py:_scrub_error_event
    """

    def test_non_json_data_returned_unchanged(self) -> None:
        """A payload that fails to parse as JSON is relayed verbatim."""
        data = "not json at all"
        assert _scrub_error_event(data) == data

    def test_json_list_payload_returned_unchanged(self) -> None:
        """A top-level JSON array (not an object) is relayed verbatim."""
        data = dumps([1, 2, 3])
        assert _scrub_error_event(data) == data

    def test_plain_string_error_message_scrubbed(self) -> None:
        """A bare-string top-level ``error`` field is scrubbed for ARNs."""
        data = dumps({"error": "boom arn:aws:iam::123456789012:role/x"})
        scrubbed = _scrub_error_event(data)
        assert "123456789012" not in scrubbed
        assert loads(scrubbed)["error"] != "boom arn:aws:iam::123456789012:role/x"


class TestEventUsageMessagesAccumulation:
    """``_event_usage`` accumulates Anthropic input usage across stream events.

    Anthropic reports input tokens on ``message_start`` and output tokens on
    ``message_delta``, so neither event alone can be billed: the two are merged
    before recording.

    Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
         stdapi/models/chat/_mantle/_default.py:_event_usage
    """

    def test_message_start_then_delta_merges_usage(self) -> None:
        """``message_start`` seeds input usage; ``message_delta`` merges and returns it."""
        input_usage: dict[str, Any] = {}
        start_parsed = {"message": {"usage": {"input_tokens": 7, "output_tokens": 0}}}
        assert (
            mantle_default._event_usage(  # noqa: SLF001
                "messages", "message_start", start_parsed, input_usage
            )
            is None
        )
        assert input_usage == {"input_tokens": 7, "output_tokens": 0}
        delta_parsed = {"usage": {"output_tokens": 5}}
        merged = mantle_default._event_usage(  # noqa: SLF001
            "messages", "message_delta", delta_parsed, input_usage
        )
        assert merged == {"input_tokens": 7, "output_tokens": 5}

    def test_unrelated_event_returns_none(self) -> None:
        """An event matching no case (e.g. ``content_block_delta``) returns ``None``."""
        assert (
            mantle_default._event_usage(  # noqa: SLF001
                "messages", "content_block_delta", {}, {}
            )
            is None
        )


class TestEventUsageResponsesTerminalEvents:
    """``_event_usage`` extracts usage from every terminal Responses event.

    A Responses stream can end on ``completed``, ``incomplete`` or ``failed``,
    and all three carry the usage block: billing only ``completed`` would lose
    the tokens of every truncated or failed generation.

    Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
         stdapi/models/chat/_mantle/_default.py:_event_usage
    """

    @pytest.mark.parametrize(
        "event", ["response.completed", "response.incomplete", "response.failed"]
    )
    def test_terminal_event_returns_usage(self, event: str) -> None:
        """A streamed Responses call ending on any terminal event is billed."""
        parsed = {"response": {"usage": {"input_tokens": 4, "output_tokens": 6}}}
        assert mantle_default._event_usage("responses", event, parsed, {}) == {  # noqa: SLF001
            "input_tokens": 4,
            "output_tokens": 6,
        }

    def test_non_terminal_event_returns_none(self) -> None:
        """A non-terminal Responses event (e.g. a delta) carries no usage."""
        parsed = {"response": {"usage": {"input_tokens": 4, "output_tokens": 6}}}
        assert (
            mantle_default._event_usage(  # noqa: SLF001
                "responses", "response.output_text.delta", parsed, {}
            )
            is None
        )


class TestObserveStreamResponsesIncompleteBilling:
    """A native Responses stream truncated at max_output_tokens is still billed.

    Ref: stdapi/models/chat/_mantle/_default.py:ChatModel._observe_stream
         stdapi/models/chat/_mantle/_default.py:_event_usage
    """

    async def test_incomplete_terminal_event_bills_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``response.incomplete`` triggers billing like ``response.completed`` does."""
        records = _capture_usage_records(monkeypatch)
        model = mantle_default.ChatModel("test.incomplete-stream-model")
        response = {
            "id": "resp_incomplete1",
            "status": "incomplete",
            "usage": {"input_tokens": 4, "output_tokens": 6, "total_tokens": 10},
        }
        events: list[SseEvent] = [
            ("response.created", dumps({"response": {"id": "resp_incomplete1"}})),
            ("response.incomplete", dumps({"response": response})),
        ]
        events_out = [
            event
            async for event in model._relay_stream(  # noqa: SLF001
                "responses",
                "responses",
                _fake_stream(events),
                "us-east-1",
                strip_usage_chunk=False,
            )
        ]
        assert len(events_out) == len(events)
        assert len(records) == 1
        assert records[0]["input_tokens"] == 4
        assert records[0]["output_tokens"] == 6


class TestRelayStreamErrorScrubbing:
    """``_relay_stream`` scrubs security details from error-named events.

    Ref: stdapi/models/chat/_mantle/_default.py:ChatModel._relay_stream
         stdapi/models/chat/_mantle/_default.py:_scrub_error_event
    """

    async def test_error_event_is_scrubbed(self) -> None:
        """An in-band ``error`` event has its message scrubbed before relay."""
        model = mantle_default.ChatModel("test.relay-error-model")
        error_data = dumps(
            {
                "type": "error",
                "error": {
                    "type": "api_error",
                    "message": "role arn:aws:iam::123456789012:role/x failed",
                },
            }
        )
        events = [
            event
            async for event in model._relay_stream(  # noqa: SLF001
                "responses",
                "responses",
                _fake_stream([("error", error_data)]),
                "us-east-1",
                strip_usage_chunk=False,
            )
        ]
        assert len(events) == 1
        assert events[0].event == "error"
        assert "123456789012" not in _event_data(events[0])


class TestReasoningFieldSurfacing:
    """A reasoning model's thinking text reaches the client as ``reasoning_content``.

    Upstream returns it under ``reasoning``; the gateway's Chat Completions
    surface declares the DeepSeek-compatible ``reasoning_content`` field, so an
    unrenamed key is pruned out of the validated response and the caller is
    billed for text it never receives.

    Ref: https://api-docs.deepseek.com/api/create-chat-completion
         stdapi/models/chat/_mantle/_default.py:ChatModel._serve_validated
         stdapi/models/chat/_mantle/_default.py:_rename_stream_reasoning
    """

    async def test_non_streaming_response_renamed_before_conversion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The raw Chat Completions message exposes ``reasoning_content``."""
        raw = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "test.model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": " 45",
                        "reasoning": " Let total be T.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 70, "completion_tokens": 342},
        }

        async def fake_serve(
            self: mantle_default.ChatModel,  # noqa: ARG001
            inbound: str,  # noqa: ARG001
            payload: dict[str, Any],  # noqa: ARG001
            *,
            stream: bool,  # noqa: ARG001
            region: str | None = None,  # noqa: ARG001
        ) -> tuple[str, str, dict[str, Any]]:
            return "chat_completions", "us-east-1", raw

        monkeypatch.setattr(mantle_default.ChatModel, "_serve", fake_serve)
        _capture_usage_records(monkeypatch)
        model = mantle_default.ChatModel("test.reasoning-model")
        _, _, out = await model._serve_validated(  # noqa: SLF001
            "chat_completions", {"messages": []}
        )
        message = out["choices"][0]["message"]
        assert message["reasoning_content"] == " Let total be T."
        assert "reasoning" not in message

    async def test_non_streaming_response_reaches_the_responses_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Renaming before conversion lets the Responses shape carry the text."""
        raw = {
            "id": "chatcmpl-1",
            "model": "test.model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": " 45",
                        "reasoning": " Let total be T.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 70, "completion_tokens": 342},
        }

        async def fake_serve(
            self: mantle_default.ChatModel,  # noqa: ARG001
            inbound: str,  # noqa: ARG001
            payload: dict[str, Any],  # noqa: ARG001
            *,
            stream: bool,  # noqa: ARG001
            region: str | None = None,  # noqa: ARG001
        ) -> tuple[str, str, dict[str, Any]]:
            return "chat_completions", "us-east-1", raw

        monkeypatch.setattr(mantle_default.ChatModel, "_serve", fake_serve)
        _capture_usage_records(monkeypatch)
        model = mantle_default.ChatModel("test.reasoning-model")
        _, _, out = await model._serve_validated(  # noqa: SLF001
            "responses", {"input": "x"}
        )
        assert out["output"][0]["type"] == "reasoning"
        assert out["output"][0]["content"][0]["text"] == " Let total be T."

    async def test_excluded_reasoning_is_dropped_from_the_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``exclude_reasoning`` removes the text instead of surfacing it.

        The client asked for the completion without its chain of thought
        (``include_reasoning: false``), so neither name may reach it.

        Ref: stdapi/types/openai_chat_completions.py:CompletionCreateParams.suppress_reasoning
        """
        raw = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "test.model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": " 45",
                        "reasoning": " Let total be T.",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 70, "completion_tokens": 342},
        }

        async def fake_serve(
            self: mantle_default.ChatModel,  # noqa: ARG001
            inbound: str,  # noqa: ARG001
            payload: dict[str, Any],  # noqa: ARG001
            *,
            stream: bool,  # noqa: ARG001
            region: str | None = None,  # noqa: ARG001
        ) -> tuple[str, str, dict[str, Any]]:
            return "chat_completions", "us-east-1", raw

        monkeypatch.setattr(mantle_default.ChatModel, "_serve", fake_serve)
        usage_records = _capture_usage_records(monkeypatch)
        model = mantle_default.ChatModel("test.reasoning-model")
        _, _, out = await model._serve_validated(  # noqa: SLF001
            "chat_completions", {"messages": []}, exclude_reasoning=True
        )
        message = out["choices"][0]["message"]
        assert "reasoning_content" not in message
        assert "reasoning" not in message
        assert message["content"] == " 45"
        assert usage_records, "the generated tokens are still billed"

    async def test_excluded_reasoning_is_dropped_from_the_stream(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reasoning deltas carry no text once the client opted out."""
        _capture_usage_records(monkeypatch)
        model = mantle_default.ChatModel("test.reasoning-stream-model")
        chunks: list[SseEvent] = [
            (None, dumps({"choices": [{"index": 0, "delta": {"reasoning": "step"}}]})),
            (
                None,
                dumps(
                    {"choices": [{"index": 0, "delta": {"reasoning_content": "step"}}]}
                ),
            ),
        ]
        events = [
            event
            async for event in model._relay_stream(  # noqa: SLF001
                "chat_completions",
                "chat_completions",
                _fake_stream(chunks),
                "us-east-1",
                strip_usage_chunk=False,
                exclude_reasoning=True,
            )
        ]
        deltas = [
            loads(_event_data(event))["choices"][0]["delta"] for event in events[:-1]
        ]
        assert deltas == [{}, {}]

    async def test_streaming_delta_renamed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A chunk carrying ``delta.reasoning`` is relayed as ``reasoning_content``."""
        _capture_usage_records(monkeypatch)
        model = mantle_default.ChatModel("test.reasoning-stream-model")
        chunk = dumps(
            {
                "id": "chatcmpl-1",
                "choices": [{"index": 0, "delta": {"reasoning": "step"}}],
            }
        )
        events = [
            event
            async for event in model._relay_stream(  # noqa: SLF001
                "chat_completions",
                "chat_completions",
                _fake_stream([(None, chunk)]),
                "us-east-1",
                strip_usage_chunk=False,
            )
        ]
        relayed = loads(_event_data(events[0]))
        assert relayed["choices"][0]["delta"] == {"reasoning_content": "step"}
        assert events[-1].data == "[DONE]"

    async def test_frames_without_reasoning_pass_through_byte_identical(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Frames not mentioning the key are relayed without a parse round trip.

        The relay deliberately avoids re-serialising every frame, so a payload
        with unusual spacing must come back exactly as it was sent.
        """
        _capture_usage_records(monkeypatch)
        model = mantle_default.ChatModel("test.reasoning-passthrough-model")
        spaced = '{"id": "chatcmpl-1",  "choices": [{"index": 0, "delta": {"content":  "hi"}}]}'
        malformed = 'not json but has "reasoning" in it'
        events = [
            event
            async for event in model._relay_stream(  # noqa: SLF001
                "chat_completions",
                "chat_completions",
                _fake_stream([(None, spaced), (None, malformed)]),
                "us-east-1",
                strip_usage_chunk=False,
            )
        ]
        assert [_event_data(event) for event in events[:-1]] == [spaced, malformed]

    async def test_streaming_reasoning_reaches_the_responses_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Renaming before conversion feeds the Responses reasoning events."""
        _capture_usage_records(monkeypatch)
        model = mantle_default.ChatModel("test.reasoning-convert-model")
        chunks: list[SseEvent] = [
            (
                None,
                dumps(
                    {
                        "id": "chatcmpl-1",
                        "model": "test.model",
                        "choices": [{"index": 0, "delta": {"reasoning": "step"}}],
                    }
                ),
            ),
            (
                None,
                dumps(
                    {
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                    }
                ),
            ),
        ]
        events = [
            event
            async for event in model._relay_stream(  # noqa: SLF001
                "chat_completions",
                "responses",
                _fake_stream(chunks),
                "us-east-1",
                strip_usage_chunk=False,
                response_id="resp_route1",
            )
        ]
        deltas = [
            loads(_event_data(event))["delta"]
            for event in events
            if event.event == "response.reasoning_text.delta"
        ]
        assert deltas == ["step"]

    @pytest.mark.parametrize(
        ("setting", "expected_delta"),
        [
            ("reasoning_content", {"reasoning_content": "step"}),
            ("reasoning", {"reasoning": "step"}),
            ("none", {}),
        ],
    )
    async def test_stream_honors_the_configured_reasoning_field(
        self,
        monkeypatch: pytest.MonkeyPatch,
        setting: str,
        expected_delta: dict[str, Any],
    ) -> None:
        """Passthrough deltas carry the field the operator setting promises.

        The non-streaming path applies the setting when the validated response
        is serialized, and the configuration doc promises the stream and the
        final message never disagree, so the relayed raw frames must honor it
        too — whichever of the two names upstream used.

        Ref: stdapi/config.py:_Settings.chat_completions_reasoning_field
             stdapi/types/openai_chat_completions.py:_rename_emitted_reasoning
             stdapi/models/chat/_mantle/_default.py:_rename_stream_reasoning
        """
        monkeypatch.setattr(SETTINGS, "chat_completions_reasoning_field", setting)
        _capture_usage_records(monkeypatch)
        model = mantle_default.ChatModel("test.reasoning-stream-model")
        chunks: list[SseEvent] = [
            (None, dumps({"choices": [{"index": 0, "delta": {"reasoning": "step"}}]})),
            (
                None,
                dumps(
                    {"choices": [{"index": 0, "delta": {"reasoning_content": "step"}}]}
                ),
            ),
        ]
        events = [
            event
            async for event in model._relay_stream(  # noqa: SLF001
                "chat_completions",
                "chat_completions",
                _fake_stream(chunks),
                "us-east-1",
                strip_usage_chunk=False,
            )
        ]
        deltas = [
            loads(_event_data(event))["choices"][0]["delta"] for event in events[:-1]
        ]
        assert deltas == [expected_delta, expected_delta]

    @pytest.mark.parametrize("setting", ["reasoning", "none"])
    async def test_setting_does_not_leak_into_converted_streams(
        self, monkeypatch: pytest.MonkeyPatch, setting: str
    ) -> None:
        """The setting governs only the Chat Completions surface.

        A Chat Completions upstream converted to the Responses shape must keep
        its reasoning events: the normalization pass feeding the converter is
        independent from the operator's Chat Completions field choice.

        Ref: stdapi/models/chat/_mantle/_default.py:ChatModel._relay_stream
        """
        monkeypatch.setattr(SETTINGS, "chat_completions_reasoning_field", setting)
        _capture_usage_records(monkeypatch)
        model = mantle_default.ChatModel("test.reasoning-convert-model")
        chunks: list[SseEvent] = [
            (
                None,
                dumps(
                    {
                        "id": "chatcmpl-1",
                        "model": "test.model",
                        "choices": [{"index": 0, "delta": {"reasoning": "step"}}],
                    }
                ),
            ),
            (
                None,
                dumps(
                    {
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 2},
                    }
                ),
            ),
        ]
        events = [
            event
            async for event in model._relay_stream(  # noqa: SLF001
                "chat_completions",
                "responses",
                _fake_stream(chunks),
                "us-east-1",
                strip_usage_chunk=False,
                response_id="resp_route1",
            )
        ]
        deltas = [
            loads(_event_data(event))["delta"]
            for event in events
            if event.event == "response.reasoning_text.delta"
        ]
        assert deltas == ["step"]


class TestEndpointUrl:
    """Mantle endpoint URL resolution: configured template vs. default.

    Mantle lives on its own host family, ``bedrock-mantle.{region}.api.aws``,
    not on the ``bedrock-runtime.{region}.amazonaws.com`` endpoint.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
         stdapi/aws_bedrock_mantle.py:endpoint_url
    """

    def test_configured_template_formatted_and_stripped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A configured endpoint template is formatted and trailing-slash stripped."""
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_mantle_endpoint_url", "https://example.com/{region}/"
        )
        assert (
            aws_bedrock_mantle.endpoint_url("us-east-1")
            == "https://example.com/us-east-1"
        )

    def test_default_template_used_when_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default Mantle endpoint template is used when unconfigured."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_endpoint_url", None)
        assert (
            aws_bedrock_mantle.endpoint_url("us-east-1")
            == "https://bedrock-mantle.us-east-1.api.aws"
        )


async def _fake_bearer_token(region: RegionName) -> str:  # noqa: ARG001
    """Return a stub bearer token bypassing real credential resolution."""
    return "test-token"


class TestBearerTokenNoCredentials:
    """Bearer token minting fails cleanly without AWS credentials.

    A missing credential chain is a server-side misconfiguration, so it maps to
    a 500 rather than surfacing as an unhandled exception or a 401.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html
         stdapi/aws_bedrock_mantle.py:bearer_token
    """

    async def test_no_credentials_raises_api_error_500(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Absent credentials map to a 500, not an unhandled exception."""

        async def fake_get_credentials() -> None:
            return None

        monkeypatch.setattr(AWS_SESSION, "get_credentials", fake_get_credentials)
        monkeypatch.setattr(aws_bedrock_mantle, "_TOKENS", {})
        with pytest.raises(ApiError) as exc_info:
            await aws_bedrock_mantle.bearer_token("us-east-1")
        assert exc_info.value.status == 500
        assert "No AWS credentials available" in str(exc_info.value)


@dataclass(slots=True)
class _FakeFrozenCredentials:
    """Minimal frozen-credentials stand-in for ``bearer_token``."""

    access_key: str
    secret_key: str
    token: str | None = None


class _FakeCredentials:
    """Minimal credentials stand-in exposing ``get_frozen_credentials``."""

    def __init__(self, frozen: _FakeFrozenCredentials) -> None:
        self._frozen = frozen

    async def get_frozen_credentials(self) -> _FakeFrozenCredentials:
        """Return the stand-in frozen credentials."""
        return self._frozen


class TestBearerTokenMintingAndCaching:
    """Bearer token minting shape, caching, and TTL-based refresh.

    The token is not an AWS-issued API key: it is a locally presigned SigV4
    query URL for ``POST https://bedrock.amazonaws.com/?Action=CallWithBearerToken``,
    base64-encoded with ``&Version=1`` appended, reproducing what
    ``aws-bedrock-token-generator`` emits. It is cached per Region and re-minted
    on expiry so rotated session credentials are picked up.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html
         stdapi/aws_bedrock_mantle.py:bearer_token
    """

    def _install_fake_credentials(self, monkeypatch: pytest.MonkeyPatch) -> list[int]:
        """Install fake AWS credentials; returns a mutable mint-call counter."""
        mint_calls = [0]

        async def fake_get_credentials() -> _FakeCredentials:
            mint_calls[0] += 1
            return _FakeCredentials(
                _FakeFrozenCredentials(
                    "AKIAFAKEACCESSKEY", "fakesecretkey", "session-tok"
                )
            )

        monkeypatch.setattr(AWS_SESSION, "get_credentials", fake_get_credentials)
        monkeypatch.setattr(aws_bedrock_mantle, "_TOKENS", {})
        return mint_calls

    async def test_token_has_expected_prefix_shape(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A minted token carries the Bedrock API-key prefix."""
        self._install_fake_credentials(monkeypatch)
        token = await aws_bedrock_mantle.bearer_token("us-east-1")
        assert token.startswith("bedrock-api-key-")
        decoded = b64decode(token.removeprefix("bedrock-api-key-")).decode()
        assert decoded.startswith("bedrock.amazonaws.com/?")
        assert "Action=CallWithBearerToken" in decoded
        assert "X-Amz-Signature=" in decoded
        assert decoded.endswith("&Version=1")
        assert "fakesecretkey" not in decoded, "the secret key is never in the token"

    async def test_cached_token_reused_without_reminting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second call within the cache TTL reuses the token without re-minting."""
        mint_calls = self._install_fake_credentials(monkeypatch)
        first = await aws_bedrock_mantle.bearer_token("us-east-1")
        second = await aws_bedrock_mantle.bearer_token("us-east-1")
        assert first == second
        assert mint_calls[0] == 1

    async def test_token_reminted_after_ttl_expiry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A call after the cache TTL elapses mints a fresh token."""
        mint_calls = self._install_fake_credentials(monkeypatch)
        clock = [1_000.0]
        monkeypatch.setattr(aws_bedrock_mantle, "monotonic", lambda: clock[0])
        await aws_bedrock_mantle.bearer_token("us-east-1")
        clock[0] += aws_bedrock_mantle._TOKEN_TTL + 1  # noqa: SLF001
        await aws_bedrock_mantle.bearer_token("us-east-1")
        assert mint_calls[0] == 2


class TestBearerTokenSurvivesCredentialRotation:
    """A rotated credential reaches Mantle, which is what long uptime depends on.

    A gateway outlives the session credentials it started with: STS rotates them
    every few hours. Minting a fresh token is not enough -- the fresh token has
    to be signed with the *new* credentials, or every Mantle request 403s from
    the first rotation until the process restarts. That is invisible to a test
    that only counts how often the token was re-minted.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html
         stdapi/aws_bedrock_mantle.py:bearer_token
    """

    @staticmethod
    def _access_key_in(token: str) -> str:
        """Return the access key the presigned token was signed with.

        Args:
            token: A minted bearer token.

        Returns:
            The key id from the presigned URL's ``X-Amz-Credential`` scope.
        """
        decoded = b64decode(token.removeprefix("bedrock-api-key-")).decode()
        credential = decoded.split("X-Amz-Credential=", 1)[1].split("&", 1)[0]
        return unquote(credential).split("/", 1)[0]

    async def test_the_new_credential_signs_the_next_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After the TTL, the token is signed with the rotated key, not the old one."""
        keys = ["AKIAOLDACCESSKEY00001", "AKIANEWACCESSKEY00002"]

        async def fake_get_credentials() -> _FakeCredentials:
            return _FakeCredentials(
                _FakeFrozenCredentials(keys[0], "fakesecretkey", "session-tok")
            )

        monkeypatch.setattr(AWS_SESSION, "get_credentials", fake_get_credentials)
        monkeypatch.setattr(aws_bedrock_mantle, "_TOKENS", {})
        clock = [1_000.0]
        monkeypatch.setattr(aws_bedrock_mantle, "monotonic", lambda: clock[0])

        before = await aws_bedrock_mantle.bearer_token("us-east-1")
        assert self._access_key_in(before) == keys[0]

        keys[0] = keys[1]
        clock[0] += aws_bedrock_mantle._TOKEN_TTL + 1  # noqa: SLF001
        after = await aws_bedrock_mantle.bearer_token("us-east-1")
        assert self._access_key_in(after) == keys[1], (
            "a token minted after rotation must carry the new credential"
        )

    def test_the_cache_expires_well_inside_a_credential_lifetime(self) -> None:
        """The cache TTL is far shorter than the signature's own validity.

        The presigned URL claims ``_TOKEN_EXPIRY`` seconds of validity, but AWS
        rejects it as soon as the signing session credentials expire, whichever
        comes first. Re-minting on a much shorter clock is what keeps the cached
        token inside the credentials' remaining life.
        """
        ttl = aws_bedrock_mantle._TOKEN_TTL  # noqa: SLF001
        assert 0 < ttl <= aws_bedrock_mantle._TOKEN_EXPIRY / 4  # noqa: SLF001

    async def test_each_region_keeps_its_own_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tokens are scoped per Region, since the signature names one.

        A token minted for one Region must never be served for another: the
        signing scope is part of what AWS validates.
        """

        async def fake_get_credentials() -> _FakeCredentials:
            return _FakeCredentials(
                _FakeFrozenCredentials("AKIAFAKEACCESSKEY", "fakesecretkey", "tok")
            )

        monkeypatch.setattr(AWS_SESSION, "get_credentials", fake_get_credentials)
        monkeypatch.setattr(aws_bedrock_mantle, "_TOKENS", {})
        east = await aws_bedrock_mantle.bearer_token("us-east-1")
        west = await aws_bedrock_mantle.bearer_token("us-west-2")
        assert east != west
        assert "us-east-1" in b64decode(east.removeprefix("bedrock-api-key-")).decode()
        assert "us-west-2" in b64decode(west.removeprefix("bedrock-api-key-")).decode()


class TestRequestConnectionFailure:
    """``_request`` maps aiohttp connection failures to a failover-eligible error.

    Ref: stdapi/aws_bedrock_mantle.py:_request
    """

    async def test_connection_error_maps_to_503_failover(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A raised ``AiohttpClientError`` maps to a 503 Mantle error with failover."""

        class _FailingSession:
            async def request(self, *args: object, **kwargs: object) -> ClientResponse:
                raise AiohttpClientError

        monkeypatch.setattr(aws_bedrock_mantle, "_SESSION", _FailingSession())
        monkeypatch.setattr(aws_bedrock_mantle, "bearer_token", _fake_bearer_token)
        with pytest.raises(MantleError) as exc_info:
            await aws_bedrock_mantle._request(  # noqa: SLF001
                "us-east-1", "/v1/chat/completions", None, None
            )
        assert exc_info.value.status == 503
        assert exc_info.value.failover is True


class TestRequestErrorBodyReadFailure:
    """``_request`` degrades gracefully when the error body cannot be read.

    Ref: stdapi/aws_bedrock_mantle.py:_request
         stdapi/aws_bedrock_mantle.py:_map_error
    """

    async def test_unreadable_error_body_maps_with_empty_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A response whose ``.text()`` raises still maps to a shaped error."""

        class _FakeResponse:
            status = 500

            async def text(self) -> str:
                raise AiohttpClientError

            def release(self) -> None:
                pass

        class _FakeSession:
            async def request(self, *args: object, **kwargs: object) -> _FakeResponse:
                return _FakeResponse()

        monkeypatch.setattr(aws_bedrock_mantle, "_SESSION", _FakeSession())
        monkeypatch.setattr(aws_bedrock_mantle, "bearer_token", _fake_bearer_token)
        with pytest.raises(MantleError) as exc_info:
            await aws_bedrock_mantle._request(  # noqa: SLF001
                "us-east-1", "/v1/chat/completions", None, None
            )
        assert exc_info.value.status == 500
        assert "HTTP 500" in str(exc_info.value)


class TestRequestWithRetry:
    """In-region retry behavior of ``_request_with_retry`` for single-region calls.

    Mantle has no botocore retry layer, so the transport owns the retry budget:
    throttling and capacity errors are retried in-region with a sleep between
    attempts, deterministic errors are not.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting-api-error-codes.html
         stdapi/aws_bedrock_mantle.py:_request_with_retry
    """

    async def test_failover_errors_retried_until_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failover-eligible errors are retried in-region up to the configured max."""
        calls = 0
        sleeps: list[float] = []

        async def fake_request(
            region: RegionName,  # noqa: ARG001
            path: str,  # noqa: ARG001
            body: bytes | None,  # noqa: ARG001
            headers: Mapping[str, str] | None,  # noqa: ARG001
            method: str = "POST",  # noqa: ARG001
        ) -> ClientResponse:
            nonlocal calls
            calls += 1
            if calls <= 2:
                msg = "busy"
                raise MantleError(msg, status=503, failover=True)
            return cast("ClientResponse", "ok")

        async def fake_sleep(delay: float) -> None:
            sleeps.append(delay)

        monkeypatch.setattr(aws_bedrock_mantle, "_request", fake_request)
        monkeypatch.setattr(aws_bedrock_mantle, "sleep", fake_sleep)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_max_retries", 3)
        result = await aws_bedrock_mantle._request_with_retry(  # noqa: SLF001
            "us-east-1", "/v1/chat/completions", {}, None, single_region=True
        )
        assert cast("Any", result) == "ok"
        assert calls == 3
        assert len(sleeps) == 2

    async def test_non_failover_error_raises_after_single_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-failover error is not retried."""
        calls = 0

        async def fake_request(
            region: RegionName,  # noqa: ARG001
            path: str,  # noqa: ARG001
            body: bytes | None,  # noqa: ARG001
            headers: Mapping[str, str] | None,  # noqa: ARG001
            method: str = "POST",  # noqa: ARG001
        ) -> ClientResponse:
            nonlocal calls
            calls += 1
            msg = "bad request"
            raise MantleError(msg, status=400, failover=False)

        monkeypatch.setattr(aws_bedrock_mantle, "_request", fake_request)
        with pytest.raises(MantleError) as exc_info:
            await aws_bedrock_mantle._request_with_retry(  # noqa: SLF001
                "us-east-1", "/v1/chat/completions", {}, None, single_region=True
            )
        assert exc_info.value.status == 400
        assert calls == 1


class TestReadJsonFailure:
    """``_read_json`` maps unparsable bodies to a shaped 502.

    Ref: stdapi/aws_bedrock_mantle.py:_read_json
    """

    async def test_json_decode_error_maps_to_502(self) -> None:
        """A response whose ``.json()`` raises ``JSONDecodeError`` maps to 502."""

        class _FakeResponse:
            async def json(
                self, content_type: str | None = None, loads: object = None
            ) -> NoReturn:
                msg = "bad"
                raise JSONDecodeError(msg, "doc", 0)

            async def __aenter__(self) -> Self:
                return self

            async def __aexit__(self, *exc_info: object) -> bool:
                return False

        with pytest.raises(MantleError) as exc_info:
            await aws_bedrock_mantle._read_json(  # noqa: SLF001
                cast("ClientResponse", _FakeResponse())
            )
        assert exc_info.value.status == 502


class _LinesContent:
    """Fake response content yielding preset raw byte lines."""

    def __init__(self, lines: list[bytes]) -> None:
        self._lines = iter(lines)

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._lines)
        except StopIteration:
            raise StopAsyncIteration from None


class _LinesResponse:
    """Fake streaming response supporting ``async with`` over preset lines."""

    def __init__(self, lines: list[bytes]) -> None:
        self.content = _LinesContent(lines)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> bool:
        return False


class TestIterSseParsing:
    """Line-level SSE parsing edge cases in ``_iter_sse``.

    Only ``data:`` and ``event:`` fields are meaningful to the relay; the
    ``[DONE]`` sentinel is suppressed here because each inbound wire format
    decides on its own whether to emit one.

    Ref: stdapi/aws_bedrock_mantle.py:_iter_sse
    """

    async def test_done_sentinel_yields_no_events(self) -> None:
        """A ``[DONE]`` data line yields no event, matching upstream's own sentinel."""
        events = [
            event
            async for event in aws_bedrock_mantle._iter_sse(  # noqa: SLF001
                cast("ClientResponse", _LinesResponse([b"data: [DONE]\n", b"\n"]))
            )
        ]
        assert events == []

    async def test_comment_and_id_lines_ignored(self) -> None:
        """Comment (``:``) and ``id:`` lines are ignored; only data/event matter."""
        lines = [
            b": a comment\n",
            b"id: 1\n",
            b"event: message\n",
            b"data: hello\n",
            b"\n",
        ]
        events = [
            event
            async for event in aws_bedrock_mantle._iter_sse(  # noqa: SLF001
                cast("ClientResponse", _LinesResponse(lines))
            )
        ]
        assert events == [("message", "hello")]

    async def test_non_utf8_line_maps_to_502(self) -> None:
        """A non-UTF-8 line mid-stream maps to the shaped 502, not a raw crash."""
        lines = [b"data: ok\n", b"\n", b"data: \xff\xfe bad\n"]
        with pytest.raises(MantleError) as exc_info:
            async for _ in aws_bedrock_mantle._iter_sse(  # noqa: SLF001
                cast("ClientResponse", _LinesResponse(lines))
            ):
                pass
        assert exc_info.value.status == 502


class TestInvokeStreamAbandonedGenerator:
    """``invoke_stream`` releases the upstream connection even if never iterated.

    Ref: stdapi/aws_bedrock_mantle.py:invoke_stream
    """

    async def test_unstarted_generator_gc_closes_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A generator garbage-collected before its first iteration closes the response.

        The ``async with`` guarding the response only runs once the generator
        body starts executing, so an abandoned, never-iterated generator
        would otherwise leave the connection open until the response's own
        ``__del__`` eventually reclaims it.
        """
        closed = False

        class _FakeResponse:
            content = _LinesContent([b"data: hi\n", b"\n"])

            def close(self) -> None:
                nonlocal closed
                closed = True

            async def __aenter__(self) -> Self:
                return self

            async def __aexit__(self, *exc_info: object) -> bool:
                return False

        async def fake_request_with_retry(
            *args: object,  # noqa: ARG001
            **kwargs: object,  # noqa: ARG001
        ) -> ClientResponse:
            return cast("ClientResponse", _FakeResponse())

        monkeypatch.setattr(
            aws_bedrock_mantle, "_request_with_retry", fake_request_with_retry
        )
        generator = await aws_bedrock_mantle.invoke_stream(
            "us-east-1", "/v1/chat/completions", {}, single_region=True
        )
        del generator
        gc_collect()
        assert closed is True


class TestResolveErrorLocResidual:
    """Additional unresolvable path shapes for ``_resolve_error_loc``.

    Ref: stdapi/aws_bedrock_mantle.py:_resolve_error_loc
    """

    def test_list_index_out_of_range_returns_none(self) -> None:
        """An out-of-range list index does not resolve."""
        assert (
            aws_bedrock_mantle._resolve_error_loc({"a": [1]}, ["a", 5]) is None  # noqa: SLF001
        )

    def test_scalar_segment_returns_none(self) -> None:
        """A path segment past a scalar value does not resolve."""
        assert (
            aws_bedrock_mantle._resolve_error_loc({"a": 1}, ["a", "b"]) is None  # noqa: SLF001
        )


class TestCatalog403Degradation:
    """A permission-mapped Mantle catalog failure degrades gracefully.

    Missing ``bedrock-mantle`` permissions must not fail startup: the Region is
    reported as failed and its models are simply absent from the catalog.

    Ref: stdapi/models/__init__.py:_collect_mantle_models
         stdapi/aws_bedrock_mantle.py:_map_error
    """

    async def test_permission_failure_recorded_in_failed_regions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ``MantleError`` from one region's catalog fetch is recorded, not raised."""
        regions: list[RegionName] = ["us-east-1", "us-west-2"]
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_regions", regions)

        async def fake_get(region: RegionName) -> list[ModelDetails]:
            if region == "us-east-1":
                msg = "Model access is not enabled."
                raise MantleError(msg, status=500)
            return []

        monkeypatch.setattr(stdapi_models, "_get_mantle_models_from_region", fake_get)
        failed_regions: dict[str, str] = {}
        models = await stdapi_models._collect_mantle_models(failed_regions, {})  # noqa: SLF001
        assert "us-east-1 (Mantle)" in failed_regions
        assert "MantleError" in failed_regions["us-east-1 (Mantle)"]
        assert models == {}


async def _connection_failure(url: str) -> MantleError:
    """Build the ``MantleError`` a real failed connection to *url* produces.

    The transport maps every connection failure onto the same sanitised
    client-facing message, chaining the real ``aiohttp`` exception as the
    cause; the exception is produced by an actual connection attempt so the
    chain is the one the runtime builds rather than a hand-made stand-in.

    Args:
        url: URL to attempt a connection to.

    Returns:
        The mapped error, with its real cause chain attached.
    """
    async with ClientSession() as session:
        try:
            await session.get(url)
        except AiohttpClientError as error:
            msg = "The service is temporarily unavailable. Retry the request."
            mapped = MantleError(msg, status=503, failover=True)
            mapped.__cause__ = error
            return mapped
    msg = f"{url} unexpectedly answered"
    raise AssertionError(msg)


class TestMantleStartupDiagnostic:
    """An unreachable Mantle region names its real cause, endpoint and setting.

    At startup the sanitised client-facing message is the operator's only
    signal, and it renders six unrelated conditions identically. The server log
    is operator-facing, so the warning carries the exception chain, the endpoint
    it failed against, and -- when the region simply has no Mantle endpoint --
    the setting to change instead of a transient-sounding outage.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
         https://docs.aiohttp.org/en/stable/client_reference.html
         stdapi/models/__init__.py:_collect_mantle_models
         stdapi/aws_bedrock_mantle.py:format_exception_chain
    """

    @staticmethod
    def _fail_first_region(
        monkeypatch: pytest.MonkeyPatch, error: BaseException
    ) -> None:
        """Make the first of two configured Mantle regions fail with *error*."""
        regions: list[RegionName] = ["us-east-1", "us-west-2"]
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_regions", regions)

        async def fake_get(region: RegionName) -> list[ModelDetails]:
            if region == "us-east-1":
                raise error
            return []

        monkeypatch.setattr(stdapi_models, "_get_mantle_models_from_region", fake_get)

    async def test_refused_connection_reports_the_real_cause(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refused connection is reported with its cause chain and endpoint.

        ``MantleError: The service is temporarily unavailable`` alone cannot be
        told apart from a DNS failure, a TLS error or a read timeout, so the
        warning appends every exception the error was raised from.
        """
        # Port 1 on loopback is reserved and never bound, so the connection is
        # refused without racing another process for an ephemeral port.
        self._fail_first_region(
            monkeypatch, await _connection_failure("http://127.0.0.1:1/v1/models")
        )
        failed_regions: dict[str, str] = {}
        regions_without_endpoint: dict[str, str] = {}

        await stdapi_models._collect_mantle_models(  # noqa: SLF001
            failed_regions, regions_without_endpoint
        )

        detail = failed_regions["us-east-1 (Mantle)"]
        assert regions_without_endpoint == {}, "a refused connection is not a NXDOMAIN"
        assert "https://bedrock-mantle.us-east-1.api.aws/v1/models" in detail, (
            "the operator needs the endpoint the failure happened against"
        )
        assert "MantleError" in detail
        assert "ConnectionRefusedError" in detail, (
            "the sanitised message must not replace the real cause"
        )
        assert "Errno 111" in detail

    async def test_region_without_endpoint_reported_apart_from_outages(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A region with no Mantle endpoint is a misconfiguration, not an outage.

        ``bedrock-mantle.<region>.api.aws`` does not resolve where Bedrock
        Mantle is not served, which no retry can fix: it is reported apart from
        the transient failures, naming the setting to change.
        """
        monkeypatch.setattr(
            SETTINGS,
            "aws_bedrock_mantle_endpoint_url",
            "https://bedrock-mantle.{region}.invalid",
        )
        self._fail_first_region(
            monkeypatch,
            await _connection_failure(
                "https://bedrock-mantle.us-east-1.invalid/v1/models"
            ),
        )
        failed_regions: dict[str, str] = {}
        regions_without_endpoint: dict[str, str] = {}

        await stdapi_models._collect_mantle_models(  # noqa: SLF001
            failed_regions, regions_without_endpoint
        )

        assert failed_regions == {}, (
            "a permanent misconfiguration must not read as a transient failure"
        )
        detail = regions_without_endpoint["us-east-1"]
        assert "https://bedrock-mantle.us-east-1.invalid" in detail
        assert "AWS_BEDROCK_MANTLE_ENDPOINT_URL" in detail, (
            "an overridden endpoint URL is the setting at fault when it is set"
        )

    def test_no_endpoint_detail_names_the_region_setting(self) -> None:
        """Without an endpoint override, the region list is the setting at fault."""
        detail = stdapi_models._no_mantle_endpoint_detail("eu-west-3")  # noqa: SLF001

        assert "https://bedrock-mantle.eu-west-3.api.aws" in detail
        assert "AWS_BEDROCK_MANTLE_REGIONS" in detail, (
            "the operator needs the name of the setting to change"
        )

    async def test_unresolvable_proxy_is_not_blamed_on_the_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A DNS failure on another hostname is not a missing Mantle endpoint.

        These sessions follow the deployment's proxy environment, so an
        unresolvable proxy name fails with the same exception type, against the
        proxy's hostname. Reading that as "this region has no endpoint" would
        send the operator to drop a region that is not at fault.
        """
        self._fail_first_region(
            monkeypatch,
            await _connection_failure("https://no-such-proxy.invalid/v1/models"),
        )
        failed_regions: dict[str, str] = {}
        regions_without_endpoint: dict[str, str] = {}

        await stdapi_models._collect_mantle_models(  # noqa: SLF001
            failed_regions, regions_without_endpoint
        )

        assert regions_without_endpoint == {}
        assert "no-such-proxy.invalid" in failed_regions["us-east-1 (Mantle)"]

    async def test_catalog_fetch_is_bounded(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One silent region cannot spend the whole response timeout of startup.

        The shared session allows a full ``AI_RESPONSE_TIMEOUT`` to read a
        response, so a region that accepts connections and never answers would
        hold startup for minutes; the catalog fetch gets its own budget.
        """
        monkeypatch.setattr(SETTINGS, "aws_connect_timeout", 1)
        monkeypatch.setattr(SETTINGS, "ai_response_timeout", 600)
        started = Event()

        async def never_answers(
            region: RegionName,  # noqa: ARG001
            method: str,  # noqa: ARG001
            path: str,  # noqa: ARG001
        ) -> dict[str, Any]:
            started.set()
            await Event().wait()
            return {}

        monkeypatch.setattr(stdapi_models, "mantle_request_json", never_answers)

        with pytest.raises(TimeoutError) as failure:
            await wait_for(
                stdapi_models._get_mantle_models_from_region("us-east-1"),  # noqa: SLF001
                timeout=30,
            )

        assert started.is_set()
        assert "catalog" in str(failure.value), (
            "the timeout must say what it gave up on and after how long"
        )

    def test_regions_without_endpoint_warned_separately_at_startup(self) -> None:
        """The no-endpoint regions get their own startup warning key."""
        start_event = make_event_log(type="start")

        stdapi_models._warn_bedrock_refresh_issues(  # noqa: SLF001
            start_event, {}, {"eu-west-3": "no endpoint"}, {}, {}, set()
        )

        warnings = start_event.get("server_warnings", [])
        assert warnings == [
            {"bedrock_mantle_regions_without_endpoint": {"eu-west-3": "no endpoint"}}
        ]
        assert start_event["level"] == "warning"

    def test_no_configured_mantle_region_warns_at_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Mantle enabled with no region serving it is reported, not silent.

        The default region list is filtered to the regions that serve Mantle, so
        a deployment confined to regions without it would otherwise lose every
        Mantle model with nothing said about it.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_enabled", True)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_regions", [])
        start_event = make_event_log(type="start")

        stdapi_models._warn_bedrock_refresh_issues(start_event, {}, {}, {}, {}, set())  # noqa: SLF001

        warnings = [str(warning) for warning in start_event.get("server_warnings", [])]
        assert any("AWS_BEDROCK_MANTLE_REGIONS" in warning for warning in warnings)
        assert start_event["level"] == "warning"


class TestClaudePreFourLatestAlias:
    """Claude < 4 ``-latest`` alias picks the most recently dated runtime ID.

    Pre-4 Claude models get a ``-latest`` alias instead of a bare date-stripped
    one, matching Anthropic's own naming for those generations.

    Ref: https://platform.claude.com/docs/en/api/models/list
         stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel.get_aliases
    """

    def test_latest_alias_prefers_newest_date(self) -> None:
        """The ``-latest`` alias resolves to the newest of two dated runtime IDs."""
        newer_id = "anthropic.claude-3-7-sonnet-20250219-v1:0"
        older_id = "anthropic.claude-3-7-sonnet-20240307-v1:0"
        aliases = AnthropicClaudeChatModel.get_aliases(
            {newer_id: _model_details(newer_id), older_id: _model_details(older_id)}
        )
        assert aliases["claude-3-7-sonnet-latest"] == newer_id
        assert "claude-3-7-sonnet" not in aliases


class TestGuardrailMantleStartupWarning:
    """Startup warns when Bedrock Guardrails are configured with Mantle-served models.

    Guardrails are a bedrock-runtime feature, so a configured guardrail silently
    does not apply to any Mantle-served model; the operator is warned at startup
    rather than discovering it from unfiltered traffic.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
         stdapi/models/__init__.py:_warn_bedrock_refresh_issues
    """

    def _start_event(self) -> EventLog:
        """Build a minimal "start" event log for the warning helper."""
        return make_event_log(type="start")

    async def test_guardrail_configured_with_mantle_models_warns_at_startup(
        self,
    ) -> None:
        """A non-zero Mantle-served model count under guardrails warns at startup."""
        start_event = self._start_event()
        stdapi_models._warn_bedrock_refresh_issues(  # noqa: SLF001
            start_event, {}, {}, {}, {}, set(), mantle_guardrail_models=3
        )
        warnings = start_event.get("server_warnings", [])
        assert any(
            "Bedrock Guardrails do not apply" in str(warning) for warning in warnings
        )
        assert start_event["level"] == "warning"

    async def test_no_mantle_guardrail_models_no_warning(self) -> None:
        """A zero count adds no guardrail-related warning."""
        start_event = self._start_event()
        stdapi_models._warn_bedrock_refresh_issues(  # noqa: SLF001
            start_event, {}, {}, {}, {}, set(), mantle_guardrail_models=0
        )
        assert "server_warnings" not in start_event


class TestServeValidatedBillingService:
    """``_serve_validated`` billing always tags the Mantle service and forwards tier.

    Mantle traffic is priced and quota-tracked separately from bedrock-runtime,
    so usage must be attributed to the Mantle service and carry the tier the
    response reports as served.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
         stdapi/models/chat/_mantle/_default.py:ChatModel._record_usage
    """

    async def test_usage_recorded_with_mantle_service_and_tier(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Billed usage carries ``Service.BEDROCK_MANTLE`` and the raw service tier."""
        raw = {
            "id": "resp_full2",
            "object": "response",
            "created_at": 123,
            "model": "test.model",
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            "service_tier": "priority",
        }

        async def fake_serve(
            self: mantle_default.ChatModel,  # noqa: ARG001
            inbound: str,  # noqa: ARG001
            payload: dict[str, Any],  # noqa: ARG001
            *,
            stream: bool,  # noqa: ARG001
            region: str | None = None,  # noqa: ARG001
        ) -> tuple[str, str, dict[str, Any]]:
            return "responses", "us-east-1", raw

        monkeypatch.setattr(mantle_default.ChatModel, "_serve", fake_serve)
        records = _capture_usage_records(monkeypatch)
        model = mantle_default.ChatModel("test.serve-model-tier")
        await model._serve_validated("chat_completions", {"messages": []})  # noqa: SLF001
        assert records[0]["service"] is Service.BEDROCK_MANTLE
        assert records[0]["tier"] == "priority"


class TestMantleProject:
    """Mantle project/workspace selection and outbound header injection (offline).

    Projects and Workspaces are one Bedrock resource with two header names:
    ``anthropic-workspace`` on ``/anthropic/v1/messages`` and ``OpenAI-Project``
    on the OpenAI surfaces, carrying the same identifier. Honoring an inbound
    header is gated, since the project is a cost-attribution and access-control
    boundary.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/projects.html
         https://docs.aws.amazon.com/bedrock/latest/userguide/workspaces.html
         https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html
         stdapi/aws_bedrock_mantle.py:set_mantle_project
         stdapi/aws_bedrock_mantle.py:mantle_request_headers
    """

    @pytest.fixture(autouse=True)
    def _reset_project(self) -> Iterator[None]:
        """Reset the request-scoped Mantle project var around each test."""
        token = MANTLE_PROJECT_VAR.set("")
        yield
        MANTLE_PROJECT_VAR.reset(token)

    def test_headers_messages_keep_version_without_project(self) -> None:
        """Messages headers keep anthropic-version and omit the project when unset."""
        assert mantle_request_headers("messages") == {"anthropic-version": "2023-06-01"}

    def test_headers_openai_none_without_project(self) -> None:
        """OpenAI-compatible calls send no headers when no project is selected."""
        assert mantle_request_headers("chat_completions") is None
        assert mantle_request_headers("responses") is None

    def test_headers_inject_project_per_api(self) -> None:
        """The project is sent as anthropic-workspace on Messages, OpenAI-Project otherwise."""
        MANTLE_PROJECT_VAR.set("proj_abc123")
        assert mantle_request_headers("messages") == {
            "anthropic-version": "2023-06-01",
            "anthropic-workspace": "proj_abc123",
        }
        assert mantle_request_headers("responses") == {"OpenAI-Project": "proj_abc123"}
        assert mantle_request_headers("chat_completions") == {
            "OpenAI-Project": "proj_abc123"
        }

    def test_default_project_used_without_header(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The configured default project is selected when the request sends no header."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_project", "proj_default1")
        set_mantle_project(Headers({}))
        assert MANTLE_PROJECT_VAR.get() == "proj_default1"

    def test_header_override_honored_when_allowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request header overrides the default project when override is enabled."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_project", "proj_default1")
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_mantle_project_override", True)
        set_mantle_project(Headers({"OpenAI-Project": "proj_req9"}))
        assert MANTLE_PROJECT_VAR.get() == "proj_req9"

    def test_header_ignored_when_override_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request header is ignored when override is disabled and a default is set."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_project", "proj_default1")
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_allow_mantle_project_override", False
        )
        set_mantle_project(Headers({"anthropic-workspace": "proj_req9"}))
        assert MANTLE_PROJECT_VAR.get() == "proj_default1"

    def test_header_honored_without_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request header is always honored when no default project is configured."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_project", None)
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_allow_mantle_project_override", False
        )
        set_mantle_project(Headers({"anthropic-workspace": "proj_req9"}))
        assert MANTLE_PROJECT_VAR.get() == "proj_req9"

    def test_malformed_request_project_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malformed request-supplied project identifier raises a 400 error."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_project", None)
        with pytest.raises(ApiError) as exc:
            set_mantle_project(Headers({"OpenAI-Project": "bad id!"}))
        assert exc.value.status == 400
        assert "Invalid Bedrock Mantle project identifier" in str(exc.value)
        assert MANTLE_PROJECT_VAR.get() == ""


class TestMantleHttpSessionOwnership:
    """Only the first ``mantle_http_session`` opener owns the shared session.

    ``AWSConnectionManager.__aenter__`` warms the Mantle session in the same
    wave as the AWS clients, so a warmup that fails on a client (or any
    secondary manager, as unit tests create) enters and exits the context
    while the server's session is live — its exit must not close or clear the
    session it did not create, or every later Mantle call answers 503
    "not initialized".

    Ref: stdapi/aws_bedrock_mantle.py:mantle_http_session
         stdapi/aws.py:AWSConnectionManager.__aenter__
    """

    @pytest.mark.local
    async def test_secondary_open_reuses_and_preserves_the_owner_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A nested open yields the live session and leaves it live on exit."""
        monkeypatch.setattr(aws_bedrock_mantle, "_SESSION", None)
        async with aws_bedrock_mantle.mantle_http_session() as owner:
            async with aws_bedrock_mantle.mantle_http_session() as inner:
                assert inner is owner
            assert aws_bedrock_mantle._SESSION is owner  # noqa: SLF001
            assert not owner.closed
        assert aws_bedrock_mantle._SESSION is None  # noqa: SLF001
        assert owner.closed

    @pytest.mark.local
    async def test_session_honours_the_proxy_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Mantle session reads ``HTTPS_PROXY``/``HTTP_PROXY``/``NO_PROXY``.

        aiohttp reads those variables only when ``trust_env`` is set, while the
        AWS SDK honours them unconditionally. Without it, a deployment whose
        only egress is a proxy reaches every classic Bedrock region and no
        Mantle endpoint at all, which surfaces as the Mantle models simply
        being absent from the catalog.

        Ref: https://docs.aiohttp.org/en/stable/client_advanced.html#proxy-support
             https://docs.aiohttp.org/en/stable/client_reference.html
             stdapi/aws_bedrock_mantle.py:mantle_http_session
        """
        monkeypatch.setattr(aws_bedrock_mantle, "_SESSION", None)
        async with aws_bedrock_mantle.mantle_http_session() as session:
            assert session.trust_env is True
