"""Offline unit tests for the AWS Bedrock Mantle support.

Covers the transport helpers (:mod:`stdapi.aws_bedrock_mantle`), the Mantle
chat family API selection (:mod:`stdapi.models.chat._mantle._default`), and
the Mantle configuration validation — all without any AWS call.
"""

from __future__ import annotations

from base64 import b32encode, urlsafe_b64encode
from binascii import crc32
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from gc import collect as gc_collect
from json import JSONDecodeError, dumps, loads
from typing import TYPE_CHECKING, Any, NoReturn, cast

import pytest
from aiohttp import ClientError as AiohttpClientError
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
    set_mantle_project,
    usage_from_chat_completion,
    usage_from_message,
    usage_from_response,
    validate_pruning_extras,
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
from stdapi.models.chat.openai_gpt import ChatModel as OpenAiGptChatModel
from stdapi.monitoring import REQUEST, REQUEST_LOG, EventLog
from stdapi.pricing import Service
from stdapi.routes.openai_responses import _decode_mantle_id, _require_local_response_id
from stdapi.types.anthropic_messages import Message, MessageCreateParams, MessageParam
from stdapi.types.openai import ModerationResult, ResponseModeration
from stdapi.types.openai_chat_completions import (
    CompletionCreateParams as ChatCompletionCreateParams,
)
from stdapi.types.openai_completions import CompletionCreateParams
from stdapi.types.openai_responses import Response, ResponseCreateParams

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator, Mapping
    from typing import Self

    from aiohttp import ClientResponse
    from fastapi import Request
    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.aws_bedrock_mantle import SseEvent

pytestmark = pytest.mark.local


@pytest.fixture(autouse=True)
def _request_log_context() -> Iterator[None]:
    """Provide the request log context required by ``set_effective_region`` et al."""
    token = REQUEST_LOG.set({"level": "info"})  # type: ignore[typeddict-item]
    yield
    REQUEST_LOG.reset(token)


def _mantle_region() -> RegionName:
    """Return a region configured for Mantle in the test settings."""
    return SETTINGS.aws_bedrock_mantle_regions[0]


class TestMantleResponseIdCodec:
    """Region-tagged Mantle response ID encoding and decoding."""

    def test_round_trip(self) -> None:
        """Encoding then decoding returns the original region and native ID."""
        region = _mantle_region()
        native_id = "resp_abc123XYZ-native"
        public_id = encode_mantle_response_id(region, native_id)
        assert public_id.startswith("resp_")
        assert public_id == public_id.lower()
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
    """Usage extraction from the three Mantle wire formats."""

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
    """Mantle HTTP error mapping (:func:`stdapi.aws_bedrock_mantle._map_error`)."""

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
        """A 403 'is not enabled' permission error maps to a 500, not a 400."""
        error = _map_error(
            403, _error_body("Model access is not enabled."), "us-east-1"
        )
        assert type(error) is MantleError
        assert error.status == 500

    @pytest.mark.parametrize("status", [401, 403])
    def test_credential_errors_map_to_server_error(self, status: int) -> None:
        """Upstream auth failures are never the caller's fault (500, no failover)."""
        error = _map_error(status, _error_body("Invalid bearer token"), "us-east-1")
        assert type(error) is MantleError
        assert error.status == 500
        assert error.failover is False

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


class TestIterSseLineTooLong:
    """Oversized SSE lines are mapped to a shaped 502 :class:`MantleError`."""

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
    """Model IDs, including future versions, bind to the right Mantle class."""

    @pytest.mark.parametrize(
        "model_id", ["openai.gpt-5.6-sol", "openai.gpt-6-nova", "openai.gpt-10.1-vega"]
    )
    def test_numbered_gpt_versions_use_the_gpt_class(self, model_id: str) -> None:
        """GPT-5 and future numbered GPT versions resolve to the GPT class."""
        assert isinstance(get_mantle_chat_model(model_id), GptChatModel)

    def test_gpt_oss_is_not_matched_by_the_numbered_gpt_class(self) -> None:
        """gpt-oss models keep resolving to their own class."""
        model = get_mantle_chat_model("openai.gpt-oss-120b")
        assert isinstance(model, GptOssChatModel)
        assert not isinstance(model, GptChatModel)

    @pytest.mark.parametrize(
        "model_id",
        ["google.gemma-4-e2b", "google.gemma-5-e4b", "google.gemma-12-large"],
    )
    def test_gemma_4_and_later_use_the_gemma_class(self, model_id: str) -> None:
        """Gemma 4 and future Gemma versions resolve to the Gemma class."""
        assert isinstance(get_mantle_chat_model(model_id), GemmaChatModel)

    @pytest.mark.parametrize(
        "model_id", ["google.gemma-3-4b-it", "google.gemma-3n-e2b"]
    )
    def test_gemma_3_stays_on_the_open_weight_class(self, model_id: str) -> None:
        """Gemma 3 (including 3n) keeps resolving to the open-weight class."""
        model = get_mantle_chat_model(model_id)
        assert isinstance(model, OpenWeightChatModel)
        assert not isinstance(model, GemmaChatModel)


class TestValidatePruningExtras:
    """Extra-field pruning validation for upstream passthrough payloads."""

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
    """Mantle upstream API selection and learned-binding behavior."""

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
    """Bounded LRU behavior of the stored-response surface cache."""

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
    """Candidate request paths self-heal via a same-request alternate surface."""

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
        assert model._api_paths("messages") == [API_PATHS["messages"]]  # noqa: SLF001


class TestMantleSettings:
    """Mantle configuration fields and validation."""

    def test_mantle_enabled_by_default(self) -> None:
        """Mantle support is enabled unless explicitly turned off."""
        assert SETTINGS.aws_bedrock_mantle_enabled is True

    def test_service_header_requires_mantle_enabled(self) -> None:
        """The per-request service header cannot be enabled without Mantle."""
        with pytest.raises(ValidationError, match="aws_bedrock_mantle_service_header"):
            _Settings(
                aws_bedrock_mantle_service_header=True, aws_bedrock_mantle_enabled=False
            )

    def test_service_header_incompatible_with_guardrails(self) -> None:
        """The per-request service header cannot bypass configured guardrails."""
        with pytest.raises(ValidationError, match="aws_bedrock_mantle_service_header"):
            _Settings(
                aws_bedrock_mantle_service_header=True,
                aws_bedrock_guardrail_identifier="test-guardrail",
                aws_bedrock_guardrail_version="1",
            )

    def test_service_header_valid_with_mantle_enabled(self) -> None:
        """The service header validates when Mantle is enabled, guardrails off."""
        settings = _Settings(aws_bedrock_mantle_service_header=True)
        assert settings.aws_bedrock_mantle_service_header is True

    def test_mantle_regions_default_to_bedrock_regions(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unset Mantle regions inherit the Bedrock region list."""
        monkeypatch.delenv("aws_bedrock_mantle_regions", raising=False)
        monkeypatch.delenv("AWS_BEDROCK_MANTLE_REGIONS", raising=False)
        settings = _Settings()
        assert settings.aws_bedrock_mantle_regions == settings.aws_bedrock_regions

    def test_mantle_regions_explicit_value_preserved(self) -> None:
        """An explicit Mantle region list is kept as-is."""
        regions: list[RegionName] = ["eu-west-1"]
        settings = _Settings(aws_bedrock_mantle_regions=regions)
        assert settings.aws_bedrock_mantle_regions == regions


class TestMantleCatalogRobustness:
    """Mantle model catalog fetch tolerates malformed entries and region failures."""

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
                ModelDetails(
                    id="prov.ok-model",
                    name="ok-model",
                    provider="Prov",
                    service=MANTLE_SERVICE,
                    input_modalities=["TEXT"],
                    output_modalities=["TEXT"],
                    regions=[region],
                )
            ]

        monkeypatch.setattr(stdapi_models, "_get_mantle_models_from_region", fake_get)
        failed_regions: dict[str, str] = {}
        models = await stdapi_models._collect_mantle_models(failed_regions)  # noqa: SLF001
        assert "us-east-1 (Mantle)" in failed_regions
        assert failed_regions["us-east-1 (Mantle)"].startswith("KeyError")
        assert set(models) == {"prov.ok-model"}


def _model_details(model_id: str, service: str | None = None) -> ModelDetails:
    """Build minimal model details for alias tests."""
    details = ModelDetails(
        id=model_id,
        name=model_id,
        provider="OpenAI",
        input_modalities=["TEXT"],
        output_modalities=["TEXT"],
        regions=["us-east-1"],
    )
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
    """Stream relaying: conversion billing and stored-ID rewrites."""

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
    """Chat Completions stream usage is billed once, using the last cumulative value."""

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


class TestObserveStreamMalformedFrames:
    """Malformed relayed frames are tolerated instead of crashing the stream."""

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
    """Non-streaming conversion billing (:meth:`ChatModel._serve_validated`)."""

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


def _capture_log_response_params(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Patch ``log_response_params`` in the Mantle chat module, capturing calls."""
    calls: list[object] = []

    def fake_log(response: object, exclude: object = None) -> object:  # noqa: ARG001
        calls.append(response)
        return response

    monkeypatch.setattr(mantle_default, "log_response_params", fake_log)
    return calls


class TestNonStreamResponseLogging:
    """Non-streaming Mantle serve paths log their response, like the classic adapters."""

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
    """A synchronous Responses failure surfaces as an error, not a 200 empty body."""

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
    """``_serve`` drops ``store`` with a warning when it falls back off Responses."""

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
        try:
            api, _region, _result = await model._serve(  # noqa: SLF001
                "responses", payload, stream=False
            )
            assert api == "chat_completions"
            assert "store" not in payload
            assert len(warnings) == 1
            assert "store" in warnings[0]["args"][0]
            assert warnings[0]["kwargs"]["level"] == "warning"
        finally:
            mantle_default._LEARNED_APIS.pop("test.store-fallback-model", None)  # noqa: SLF001

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
        try:
            api, _region, _result = await model._serve(  # noqa: SLF001
                "responses", payload, stream=False
            )
            assert api == "chat_completions"
            assert "store" not in payload
            assert warnings == []
        finally:
            mantle_default._LEARNED_APIS.pop(  # noqa: SLF001
                "test.store-fallback-model-nostore", None
            )


class TestResponsesRouteGuards:
    """Stored-response route helpers guarding Mantle-form identifiers."""

    def test_undecodable_mantle_form_id_is_not_found(self) -> None:
        """An undecodable ``resp_`` ID never reaches the local store (404)."""
        with pytest.raises(ApiError) as exc_info:
            _require_local_response_id("resp_notdecodable")
        assert exc_info.value.status == 404

    def test_decode_gated_on_mantle_enabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A well-formed tagged ID does not decode when Mantle is disabled."""
        public_id = encode_mantle_response_id(_mantle_region(), "resp_native")
        assert _decode_mantle_id(public_id) is not None
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_enabled", False)
        assert _decode_mantle_id(public_id) is None


class TestMantleCompactionItemGuard:
    """Compaction input items on the Mantle Responses passthrough payload."""

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
        with pytest.raises(ApiError, match="compact the conversation again") as exc:
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
    """The stdapi ``moderation`` parameter on Mantle passthrough payloads."""

    async def test_chat_moderation_param_is_rejected(self) -> None:
        """A Chat Completions request with ``moderation`` fails with 400."""
        request = ChatCompletionCreateParams.model_validate(
            {
                "model": "ignored",
                "messages": [{"role": "user", "content": "hi"}],
                "moderation": {"model": "gr123"},
            }
        )
        with pytest.raises(ApiError, match="not available on Bedrock") as exc:
            await mantle_convert.chat_completions_payload(request, "model-id")
        assert exc.value.status == 400

    async def test_responses_moderation_param_is_rejected(self) -> None:
        """A Responses request with ``moderation`` fails with 400."""
        request = ResponseCreateParams.model_validate(
            {"model": "ignored", "input": "hi", "moderation": {"model": "gr123"}}
        )
        with pytest.raises(ApiError, match="not available on Bedrock") as exc:
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
    """In-band upstream stream errors raise a shaped 502 during conversion."""

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
        with pytest.raises(MantleError, match="boom"):
            await self._consume(stream)

    async def test_unnamed_error_payload_raises(self) -> None:
        """A chat-shaped ``{"error": ...}`` data payload aborts the stream."""
        events = _fake_stream([(None, dumps({"error": {"message": "bad thing"}}))])
        stream = mantle_convert.convert_stream("responses", "chat_completions", events)
        with pytest.raises(MantleError, match="bad thing"):
            await self._consume(stream)

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
        with pytest.raises(MantleError, match="busy"):
            await self._consume(stream)

    async def test_text_delta_containing_error_word_is_not_an_error(self) -> None:
        """Regular deltas whose text mentions "error" stream through fine."""
        events = _fake_stream(
            [
                ("response.created", dumps({"response": {"id": "resp_1"}})),
                (
                    "response.output_text.delta",
                    dumps({"delta": 'the "error" word is fine'}),
                ),
            ]
        )
        stream = mantle_convert.convert_stream("responses", "chat_completions", events)
        chunks = [loads(data) async for _, data in stream]
        assert len(chunks) == 2

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
    """Cross-shape field allow/strip lists."""

    def test_safety_identifier_is_a_common_field(self) -> None:
        """``safety_identifier`` survives Chat Completions <-> Responses."""
        assert "safety_identifier" in mantle_convert._OPENAI_COMMON_FIELDS  # noqa: SLF001

    def test_store_stripped_from_chat_passthrough(self) -> None:
        """``store`` is stripped: chat completions are persisted locally."""
        assert "store" in mantle_convert._CHAT_EXTENSION_FIELDS  # noqa: SLF001


class TestServiceTierAndEffortMapping:
    """service_tier and reasoning effort mapping across wire shapes."""

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
    """Anthropic server tools rejection during conversion toward OpenAI shapes."""

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
    """SSE wrapper converting a Chat Completions stream to text-completion chunks."""

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
    """Passthrough error event scrubbing (:func:`_default._scrub_error_event`)."""

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
    """A dual-named model keeps its bedrock-runtime ID for the shared alias."""
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
    """A Mantle-held direct alias yields to a competing bedrock-runtime dated model."""
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
    """A lone Mantle-held undated ID keeps the direct alias."""
    mantle_id = "anthropic.claude-haiku-4-5"
    aliases = AnthropicClaudeChatModel.get_aliases(
        {mantle_id: _model_details(mantle_id, MANTLE_SERVICE)}
    )
    assert aliases["claude-haiku-4-5"] == mantle_id


def test_map_error_nonstring_message_coerced() -> None:
    """A structured (non-string) upstream message maps without raising."""
    error = _map_error(400, '{"error": {"message": {"detail": "nested"}}}', "us-east-1")
    assert error.status == 400
    assert "nested" in str(error)


def test_scrub_error_event_responses_shapes() -> None:
    """Responses-shaped and response.failed error payloads are scrubbed."""
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
    """A structured (non-string) error message is serialized and scrubbed."""
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
    """Per-request ``x-stdapi-service`` header dispatch to Bedrock Mantle."""

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
        """A classic bedrock-runtime model id resolves outside the Mantle family."""
        model = get_chat_model("amazon.nova-micro-v1:0")
        assert not getattr(model, "IS_MANTLE", False)


class TestPreviousResponseIdFallback:
    """Serving demoted off the Responses API rejects a chained conversation."""

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
        try:
            with pytest.raises(ApiError) as exc_info:
                await model._serve("responses", payload, stream=False)  # noqa: SLF001
            assert exc_info.value.status == 400
            assert "previous_response_id cannot be honored" in str(exc_info.value)
        finally:
            mantle_default._LEARNED_APIS.pop(  # noqa: SLF001
                "test.previous-response-fallback-model", None
            )


class TestLearnedBindingSkipsSecondProbe:
    """A learned API binding is reused directly, skipping the failed probe."""

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
        try:
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
        finally:
            mantle_default._LEARNED_APIS.pop(model_id, None)  # noqa: SLF001


class TestMantleDisabled:
    """Behavior when Bedrock Mantle support is disabled."""

    async def test_merge_skips_catalog_fetch_when_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_merge_mantle_models`` no-ops without fetching when Mantle is disabled."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_enabled", False)

        async def fail_if_called(
            failed_regions: dict[str, str],  # noqa: ARG001
        ) -> dict[str, ModelDetails]:
            msg = "must not be called when Mantle is disabled"
            raise AssertionError(msg)

        monkeypatch.setattr(stdapi_models, "_collect_mantle_models", fail_if_called)
        all_models: dict[str, ModelDetails] = {}
        await stdapi_models._merge_mantle_models(all_models, {})  # noqa: SLF001
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


class TestMantleRegionsPinning:
    """Region candidate selection: pinned region vs. catalog vs. default list."""

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
            ModelDetails(
                id=model_id,
                name=model_id,
                provider="Test",
                service=MANTLE_SERVICE,
                input_modalities=["TEXT"],
                output_modalities=["TEXT"],
                regions=regions,
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
    """``MantleError.failover`` drives (or blocks) cross-region retry directly."""

    async def test_failover_error_retries_next_call(self) -> None:
        """A failover-eligible ``MantleError`` is retried by a further call."""
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
    """The learned-surface cache self-heals and is then reused on later calls."""

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
        try:
            _region, result = await model._invoke_api(  # noqa: SLF001
                "chat_completions", {"model": model_id, "messages": []}, stream=False
            )
            assert result == {"id": "chatcmpl-1", "choices": [], "usage": None}
            assert calls == ["/openai/v1/chat/completions", "/v1/chat/completions"]
            assert mantle_default._LEARNED_SURFACE[model_id] == "/v1"  # noqa: SLF001

            # The learned surface is tried first on the next call and
            # succeeds immediately: the previously-failing surface is
            # skipped rather than probed again.
            calls.clear()
            await model._invoke_api(  # noqa: SLF001
                "chat_completions", {"model": model_id, "messages": []}, stream=False
            )
            assert calls == ["/v1/chat/completions"]
        finally:
            mantle_default._LEARNED_SURFACE.pop(model_id, None)  # noqa: SLF001


class TestInvokeApiRetriesWhenRouterDisabled:
    """Router-disabled multi-region calls still get in-region retries (see F5)."""

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
    """Surface-learning side effects of ``_invoke_api``."""

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

        async def fake_invoke(
            region: RegionName,  # noqa: ARG001
            path: str,  # noqa: ARG001
            payload: Mapping[str, Any],  # noqa: ARG001
            *,
            single_region: bool,  # noqa: ARG001
            headers: Mapping[str, str] | None = None,  # noqa: ARG001
        ) -> dict[str, Any]:
            msg = "This model isn't supported on this route."
            raise MantleSurfaceUnsupportedError(msg, status=400)

        monkeypatch.setattr(mantle_default, "invoke", fake_invoke)
        model = mantle_default.ChatModel("test.both-surfaces-unsupported-model")
        with pytest.raises(MantleSurfaceUnsupportedError):
            await model._invoke_api(  # noqa: SLF001
                "chat_completions", {"model": "m", "messages": []}, stream=False
            )


class TestCreateResponseModerationBuilder:
    """The ``moderation_builder`` callback sets the non-streamed response moderation."""

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
    """Converted (non-native) responses carry the server-assigned response ID."""

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
    """Streaming legacy-completion and message routes reach their wrap branch."""

    async def test_create_text_completion_stream_returns_event_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A streaming legacy completion returns a wrapped ``EventSourceResponse``."""

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

    async def test_create_message_stream_returns_event_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A streaming Anthropic message request returns an ``EventSourceResponse``."""

        async def fake_invoke_stream(
            region: RegionName,  # noqa: ARG001
            path: str,  # noqa: ARG001
            payload: Mapping[str, Any],  # noqa: ARG001
            *,
            single_region: bool,  # noqa: ARG001
            headers: Mapping[str, str] | None = None,  # noqa: ARG001
        ) -> AsyncGenerator[SseEvent]:
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


class TestScrubErrorEventResidualBranches:
    """Residual ``_scrub_error_event`` shapes not covered by the classes above."""

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
    """``_event_usage`` accumulates Anthropic input usage across stream events."""

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
    """``_event_usage`` extracts usage from every terminal Responses event."""

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
    """A native Responses stream truncated at max_output_tokens is still billed."""

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
    """``_relay_stream`` scrubs security details from error-named events."""

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


class TestEndpointUrl:
    """Mantle endpoint URL resolution: configured template vs. default."""

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
    """Bearer token minting fails cleanly without AWS credentials."""

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
    """Bearer token minting shape, caching, and TTL-based refresh."""

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


class TestRequestConnectionFailure:
    """``_request`` maps aiohttp connection failures to a failover-eligible error."""

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
    """``_request`` degrades gracefully when the error body cannot be read."""

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
    """In-region retry behavior of ``_request_with_retry`` for single-region calls."""

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
    """``_read_json`` maps unparsable bodies to a shaped 502."""

    async def test_json_decode_error_maps_to_502(self) -> None:
        """A response whose ``.json()`` raises ``JSONDecodeError`` maps to 502."""

        class _FakeResponse:
            async def json(self, content_type: str | None = None) -> NoReturn:
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
    """Line-level SSE parsing edge cases in ``_iter_sse``."""

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
    """``invoke_stream`` releases the upstream connection even if never iterated."""

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
    """Additional unresolvable path shapes for ``_resolve_error_loc``."""

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
    """A permission-mapped Mantle catalog failure degrades gracefully."""

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
        models = await stdapi_models._collect_mantle_models(failed_regions)  # noqa: SLF001
        assert "us-east-1 (Mantle)" in failed_regions
        assert failed_regions["us-east-1 (Mantle)"].startswith("MantleError")
        assert models == {}


class TestClaudePreFourLatestAlias:
    """Claude < 4 ``-latest`` alias picks the most recently dated runtime ID."""

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
    """Startup warns when Bedrock Guardrails are configured with Mantle-served models."""

    def _start_event(self) -> EventLog:
        """Build a minimal "start" event log for the warning helper."""
        return EventLog(
            type="start",
            level="info",
            date=datetime.now(UTC),
            server_id="test",
            server_version="0.0.0",
        )

    async def test_guardrail_configured_with_mantle_models_warns_at_startup(
        self,
    ) -> None:
        """A non-zero Mantle-served model count under guardrails warns at startup."""
        start_event = self._start_event()
        stdapi_models._warn_bedrock_refresh_issues(  # noqa: SLF001
            start_event, {}, {}, {}, set(), mantle_guardrail_models=3
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
            start_event, {}, {}, {}, set(), mantle_guardrail_models=0
        )
        assert "server_warnings" not in start_event


class TestServeValidatedBillingService:
    """``_serve_validated`` billing always tags the Mantle service and forwards tier."""

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
    """Mantle project/workspace selection and outbound header injection (offline)."""

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
