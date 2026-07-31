"""Live tests for chat models served by the AWS Bedrock Mantle endpoint.

Exercises the Mantle passthrough and conversion paths against real AWS:
model discovery, chat completions (native and converted), the Responses API
with native storage, the Anthropic Messages route, legacy completions, and
the count_tokens proxy. Uses the cheapest verified models; the expensive
Responses-only reasoning model (``openai.gpt-5.6-luna``) appears only where
a Responses conversion target is required.

Mantle serves Responses, Chat Completions and Messages (never Converse), each
model on exactly one of the three disjoint path prefixes ``/v1``,
``/openai/v1`` and ``/anthropic/v1/messages``. No API reports which prefix or
which of the three wire formats a model accepts, so the gateway probes and
learns both, and translates a request that arrived on one API into another when
the model does not serve it natively.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
     stdapi/aws_bedrock_mantle.py
     stdapi/models/chat/_mantle/_default.py:ChatModel
     stdapi/models/chat/_mantle/_convert.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from anthropic import BadRequestError as AnthropicBadRequestError
from openai import BadRequestError, NotFoundError, OpenAI

from tests.conftest import logged_usage_entries

#: The learned-routing caches are process-global, so these tests must not be split
#: across xdist workers that would each learn a different routing surface.
#: Bedrock Mantle is an AWS capability reached through the gateway; the models it
#: serves (Gemma, the Mantle-hosted OpenAI builds) do not exist upstream, so the
#: whole module is meaningless against an official API but valid against a
#: deployed gateway.
pytestmark = [pytest.mark.xdist_group("mantle_live"), pytest.mark.gateway]

if TYPE_CHECKING:
    from anthropic import Anthropic
    from starlette.testclient import TestClient as TestClientType

#: Cheap chat-completions-only Mantle model on the legacy /v1 surface.
_GEMMA3 = "google.gemma-3-4b-it"

#: Mantle model serving Chat Completions and Responses on /openai/v1 (native store).
_GEMMA4 = "google.gemma-4-e2b"

#: Responses-only reasoning model (expensive; keep usage minimal).
_LUNA = "openai.gpt-5.6-luna"

#: Mantle-only undated Claude ID (Messages passthrough; keep usage minimal).
_CLAUDE_MANTLE = "anthropic.claude-haiku-4-5"

#: Dual-homed model kept on bedrock-runtime (service-header routing target).
_GEMMA3_DUAL = "google.gemma-3-12b-it"

#: Service name recorded for Mantle-served usage entries.
_MANTLE_USAGE_SERVICE = "bedrock-mantle"

#: Output-token budget for the reasoning model (needs headroom to answer).
_LUNA_MAX_TOKENS = 500


@pytest.fixture(scope="module")
def listed_model_ids(openai_client: OpenAI) -> set[str]:
    """Fetch the served model IDs once for all discovery tests."""
    return {model.id for model in openai_client.models.list()}


class TestMantleModelDiscovery:
    """Mantle model discovery and registry merge behavior.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
         stdapi/models/__init__.py:is_mantle_served
         stdapi/models/__init__.py:is_mantle_preferred
    """

    def test_mantle_models_registered(
        self, test_client: TestClientType | None, listed_model_ids: set[str]
    ) -> None:
        """Discovered Mantle models are Mantle-served and appear in ``GET /v1/models``.

        Gemma 3 is dual-homed (present on both endpoints), so it is Mantle-served
        only because the test environment lists it in
        ``aws_bedrock_mantle_preferred_models``.
        """
        if test_client is None:
            pytest.skip("Requires local test server")
        from stdapi.models import (  # noqa: PLC0415
            MANTLE_MODELS,
            is_mantle_preferred,
            is_mantle_served,
        )

        assert MANTLE_MODELS, "Expected Mantle model discovery to populate the registry"
        for model_id in (_GEMMA3, _GEMMA4, _LUNA):
            assert model_id in MANTLE_MODELS
            assert is_mantle_served(model_id), model_id
            assert model_id in listed_model_ids
        assert is_mantle_preferred(_GEMMA3)

    def test_dual_homed_models_stay_on_runtime(
        self, test_client: TestClientType | None
    ) -> None:
        """Models on both endpoints are served by bedrock-runtime by default.

        Mantle discovery must not steal a model that bedrock-runtime can serve:
        only an explicit ``aws_bedrock_mantle_preferred_models`` entry flips the
        priority, so every dual-homed entry left on runtime is non-preferred.
        """
        if test_client is None:
            pytest.skip("Requires local test server")
        from stdapi.models import (  # noqa: PLC0415
            MANTLE_MODELS,
            is_mantle_preferred,
            is_mantle_served,
        )

        dual_homed = [mid for mid in MANTLE_MODELS if not is_mantle_served(mid)]
        assert dual_homed, "Expected dual-homed models to keep runtime priority"
        assert all(not is_mantle_preferred(mid) for mid in dual_homed)
        # Non-preferred Gemma 3 siblings are dual-homed and stay on runtime.
        sibling = "google.gemma-3-12b-it"
        if sibling in MANTLE_MODELS:
            assert not is_mantle_served(sibling)


class TestMantleChatCompletions:
    """Chat Completions served by Mantle (passthrough and conversions).

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions-mantle.html
         https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
         stdapi/models/chat/_mantle/_default.py:ChatModel._select_api
    """

    def test_gemma3_passthrough(self, openai_client: OpenAI) -> None:
        """Chat completion passthrough returns assistant content and a coherent usage split.

        Gemma 3 serves Chat Completions natively, so the request is relayed
        untranslated and the upstream usage object is passed through.
        """
        response = openai_client.chat.completions.create(
            model=_GEMMA3,
            messages=[{"role": "user", "content": "Say OK."}],
            max_completion_tokens=32,
        )
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens >= (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

    def test_gemma4_openai_v1_surface(self, openai_client: OpenAI) -> None:
        """Chat completion passthrough works for a Mantle-only Gemma 4 model.

        Gemma 4 is reachable only on the ``/openai/v1`` prefix, so this covers
        the non-default surface of the probe-and-learn selection (the learned
        value itself is asserted by ``test_gemma4_surface_self_heal``).
        """
        response = openai_client.chat.completions.create(
            model=_GEMMA4,
            messages=[{"role": "user", "content": "Say OK."}],
            max_completion_tokens=32,
        )
        assert response.choices[0].message.content
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    def test_gemma3_streaming_usage_stripped(
        self,
        openai_client: OpenAI,
        test_client: TestClientType | None,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """Streaming without include_usage strips the forced usage chunk.

        The gateway always asks Mantle for a trailing usage chunk so it can bill
        the request, then removes it from the client stream when the caller did
        not set ``stream_options.include_usage``.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
        """
        if test_client is None:
            pytest.skip("Requires local test server")
        capfd.readouterr()
        stream = openai_client.chat.completions.create(
            model=_GEMMA3,
            messages=[{"role": "user", "content": "Say OK."}],
            max_completion_tokens=32,
            stream=True,
        )
        chunks = [chunk for chunk in stream if not isinstance(chunk, str)]
        assert chunks, "No streaming chunks received"
        assert all(chunk.usage is None for chunk in chunks), (
            "Usage chunk must be stripped when include_usage is not requested"
        )
        content = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in chunks
            if chunk.choices and chunk.choices[0].delta
        )
        assert content
        entries = logged_usage_entries(
            capfd.readouterr().out, service=_MANTLE_USAGE_SERVICE, model=_GEMMA3
        )
        assert entries, "Expected a Mantle usage log entry for the stream"
        assert sum(entry["output_tokens"] for entry in entries) > 0

    def test_gemma3_streaming_usage_included(
        self,
        openai_client: OpenAI,
        test_client: TestClientType | None,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """Streaming with include_usage relays the final usage chunk exactly once.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
        """
        if test_client is None:
            pytest.skip("Requires local test server")
        capfd.readouterr()
        stream = openai_client.chat.completions.create(
            model=_GEMMA3,
            messages=[{"role": "user", "content": "Say OK."}],
            max_completion_tokens=32,
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = [chunk for chunk in stream if not isinstance(chunk, str)]
        usage_chunks = [chunk for chunk in chunks if chunk.usage is not None]
        assert usage_chunks, "Expected the final usage chunk to be relayed"
        assert usage_chunks[-1].usage is not None
        assert usage_chunks[-1].usage.total_tokens > 0
        entries = logged_usage_entries(
            capfd.readouterr().out, service=_MANTLE_USAGE_SERVICE, model=_GEMMA3
        )
        assert entries, "Expected a Mantle usage log entry for the stream"
        assert len(entries) == 1, "Usage must be recorded exactly once"
        assert entries[0]["output_tokens"] > 0

    def test_gemma3_upstream_unsupported_parameter_error(
        self, openai_client: OpenAI
    ) -> None:
        """``logprobs`` is refused with a 400 naming the offending parameter.

        The gateway rejects ``logprobs`` in request validation for every model,
        so the Mantle model is never called; the point of the check is that the
        envelope carries ``param``/``code`` and not a bare 500.

        Ref: stdapi/types/openai_chat_completions.py:CompletionCreateParams._unsupported
             stdapi/api_errors.py:UnsupportedParameterError
        """
        with pytest.raises(BadRequestError) as bad_request:
            openai_client.chat.completions.create(
                model=_GEMMA3,
                messages=[{"role": "user", "content": "Say OK."}],
                max_completion_tokens=16,
                logprobs=True,
                top_logprobs=3,
            )
        assert bad_request.value.status_code == 400
        body = bad_request.value.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        assert body["param"] == "logprobs"
        assert body["code"] == "unsupported_parameter"
        assert "logprobs" in body["message"]
        assert "not supported" in body["message"]

    def test_claude_parallel_tool_calls_disabled_upstream(
        self, openai_client: OpenAI
    ) -> None:
        """``parallel_tool_calls: false`` converts to a tool choice Mantle accepts.

        Anthropic carries the switch as ``disable_parallel_tool_use`` on the tool
        choice, so the conversion synthesises the default ``auto`` choice and
        sets the flag on it. A two-city question would otherwise produce two
        parallel calls, so a single call is the observable consequence.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             stdapi/types/openai_chat_completions.py:CompletionCreateParams._unsupported
             stdapi/models/chat/_mantle/_convert.py:_anthropic_tool_choice_from_chat
        """
        response = openai_client.chat.completions.create(
            model=_CLAUDE_MANTLE,
            messages=[
                {
                    "role": "user",
                    "content": "What is the weather in Paris and in Tokyo?",
                }
            ],
            max_completion_tokens=128,
            parallel_tool_calls=False,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get the weather of a city.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        )
        tool_calls = response.choices[0].message.tool_calls or []
        assert len(tool_calls) == 1, (
            "parallel_tool_calls=false must collapse the two-city question "
            "into a single tool call"
        )
        call = tool_calls[0]
        assert call.type == "function"
        assert call.function.name == "get_weather"

    @pytest.mark.slow
    def test_luna_converted_to_responses(
        self,
        openai_client: OpenAI,
        test_client: TestClientType | None,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A chat completion on a Responses-only model is served via Responses, and billed once.

        Luna serves the Responses API only, so the request is translated and the
        Responses usage object (``input_tokens``/``output_tokens``) is mapped
        back onto Chat Completions' ``prompt_tokens``/``completion_tokens``. The
        server-side billing assertions ride on this same request: Luna is the most
        expensive model in the suite, so the conversion is paid for exactly once.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_usage_from_responses
             stdapi/aws_bedrock_mantle.py:usage_from_response
        """
        capfd.readouterr()
        response = openai_client.chat.completions.create(
            model=_LUNA,
            messages=[{"role": "user", "content": "Reply with the single word: hi"}],
            max_completion_tokens=_LUNA_MAX_TOKENS,
        )
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].message.content
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens >= (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )
        if test_client is None:
            return
        # OpenAI-shaped input counts include cached tokens, so the extraction
        # subtracts them before billing.
        entries = logged_usage_entries(
            capfd.readouterr().out, service=_MANTLE_USAGE_SERVICE, model=_LUNA
        )
        assert len(entries) == 1, "Usage must be recorded exactly once"
        assert entries[0]["input_tokens"] > 0
        assert entries[0]["output_tokens"] > 0

    @pytest.mark.slow
    def test_luna_converted_streaming(self, openai_client: OpenAI) -> None:
        """A streamed chat completion on a Responses-only model converts back.

        Upstream Responses SSE events are re-emitted as Chat Completions chunks,
        with the trailing usage chunk synthesised from the Responses usage.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_usage_from_responses
        """
        stream = openai_client.chat.completions.create(
            model=_LUNA,
            messages=[{"role": "user", "content": "Reply with the single word: hi"}],
            max_completion_tokens=_LUNA_MAX_TOKENS,
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks = [chunk for chunk in stream if not isinstance(chunk, str)]
        content = "".join(
            chunk.choices[0].delta.content or ""
            for chunk in chunks
            if chunk.choices and chunk.choices[0].delta
        )
        assert content
        usage_chunks = [chunk for chunk in chunks if chunk.usage is not None]
        assert usage_chunks, "Expected the converted usage chunk to be relayed"
        assert usage_chunks[-1].usage is not None
        assert usage_chunks[-1].usage.completion_tokens > 0


class TestMantleServiceHeader:
    """Per-request Mantle routing via the gated ``x-stdapi-service`` header.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
         stdapi/models/chat/__init__.py:serves_via_mantle
    """

    def test_header_routes_dual_homed_model_to_mantle(
        self,
        openai_client: OpenAI,
        test_client: TestClientType | None,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """The header flips a runtime-served dual-homed model onto Mantle.

        The two endpoints have independent per-model token quotas, so the proof
        that the header took effect is the service recorded on the billed usage
        entry, not merely a successful generation.
        """
        if test_client is None:
            pytest.skip("Requires local test server")
        from stdapi.config import SETTINGS  # noqa: PLC0415
        from stdapi.models import MANTLE_MODELS, is_mantle_served  # noqa: PLC0415

        if _GEMMA3_DUAL not in MANTLE_MODELS or is_mantle_served(_GEMMA3_DUAL):
            pytest.skip("Requires a dual-homed runtime-served model")
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_service_header", True)
        capfd.readouterr()
        response = openai_client.chat.completions.create(
            model=_GEMMA3_DUAL,
            messages=[{"role": "user", "content": "Say OK."}],
            max_completion_tokens=32,
            extra_headers={"x-stdapi-service": "bedrock-mantle"},
        )
        assert response.choices[0].message.content
        entries = logged_usage_entries(
            capfd.readouterr().out, service=_MANTLE_USAGE_SERVICE, model=_GEMMA3_DUAL
        )
        assert entries, "Expected the request to bill under bedrock-mantle"
        assert sum(entry["output_tokens"] for entry in entries) > 0


class TestMantleRecordedBilling:
    """Server-side usage recording on the converted (non-passthrough) paths.

    A converted request has no client-visible usage object to relay, so the
    gateway must extract usage from the upstream wire format itself.

    Ref: stdapi/aws_bedrock_mantle.py:usage_from_response
         stdapi/aws_bedrock_mantle.py:usage_from_chat_completion
    """

    def test_gemma3_streamed_conversion_records_usage(
        self,
        openai_client: OpenAI,
        test_client: TestClientType | None,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A streamed responses-to-chat conversion records billed usage.

        The Chat Completions stream that actually served the request carries its
        usage in a trailing chunk, which the converter must consume even though
        the client only ever sees Responses events.
        """
        if test_client is None:
            pytest.skip("Requires local test server")
        capfd.readouterr()
        stream = openai_client.responses.create(
            model=_GEMMA3,
            input="Say OK.",
            store=False,
            stream=True,
            max_output_tokens=64,
        )
        events = list(stream)
        assert any(event.type == "response.completed" for event in events)
        entries = logged_usage_entries(
            capfd.readouterr().out, service=_MANTLE_USAGE_SERVICE, model=_GEMMA3
        )
        assert entries, "Expected a Mantle usage log entry for the stream"
        assert sum(entry["output_tokens"] for entry in entries) > 0


class TestMantleResponses:
    """Responses API served by Mantle, including native storage.

    Mantle stored responses are region-local (30-day retention in the source
    Region) and project-scoped, so the gateway's public ``resp_`` IDs embed a
    crc32 fingerprint of the Region and are decoded back on every read.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
         https://developers.openai.com/api/reference/resources/responses
         stdapi/aws_bedrock_mantle.py:encode_mantle_response_id
    """

    def test_gemma4_stored_response_lifecycle(
        self, openai_client: OpenAI, test_client: TestClientType | None
    ) -> None:
        """store=True uses Mantle native storage with region-tagged IDs.

        The full lifecycle is covered in one pass: the created ID decodes to the
        Region that served it, retrieval and chaining both work through the
        public ID, and after DELETE the same ID is a 404.

        Ref: stdapi/aws_bedrock_mantle.py:decode_mantle_response_id
             stdapi/routes/openai_responses.py:_mantle_stored_response
        """
        first = openai_client.responses.create(
            model=_GEMMA4,
            input="Remember this number: 42. Just say 'Noted.'",
            store=True,
            max_output_tokens=64,
        )
        second = None
        try:
            assert first.id.startswith("resp_")
            assert first.status == "completed"
            if test_client is not None:
                from stdapi.aws_bedrock_mantle import (  # noqa: PLC0415
                    decode_mantle_response_id,
                )

                decoded = decode_mantle_response_id(first.id)
                assert decoded is not None
                assert decoded[0] == "us-east-1"

            retrieved = openai_client.responses.retrieve(first.id)
            assert retrieved.id == first.id
            assert retrieved.status == "completed"

            second = openai_client.responses.create(
                model=_GEMMA4,
                input="What number did I ask you to remember? Digits only.",
                previous_response_id=first.id,
                store=True,
                max_output_tokens=64,
            )
            assert second.id != first.id
            assert "42" in second.output_text
        finally:
            if second is not None:
                openai_client.responses.delete(second.id)
            openai_client.responses.delete(first.id)
        with pytest.raises(NotFoundError) as deleted:
            openai_client.responses.retrieve(first.id)
        assert deleted.value.status_code == 404

    def test_gemma4_streamed_chained_public_previous_id(
        self, openai_client: OpenAI, test_client: TestClientType | None
    ) -> None:
        """A streamed chained request echoes the public ID, never the native one.

        The native Mantle ID must stay server-side: leaking it would let a client
        address the upstream store directly, bypassing the Region fingerprint.

        Ref: stdapi/routes/openai_responses.py:_with_public_mantle_ids
        """
        first = openai_client.responses.create(
            model=_GEMMA4,
            input="Remember this number: 7. Just say 'Noted.'",
            store=True,
            max_output_tokens=64,
        )
        assert first.id.startswith("resp_")
        try:
            stream = openai_client.responses.create(
                model=_GEMMA4,
                input="What number did I ask you to remember? Digits only.",
                previous_response_id=first.id,
                store=False,
                stream=True,
                max_output_tokens=64,
            )
            data = "".join(event.model_dump_json() for event in stream)
            assert first.id in data, "Public previous_response_id must be echoed"
            if test_client is not None:
                from stdapi.aws_bedrock_mantle import (  # noqa: PLC0415
                    decode_mantle_response_id,
                )

                decoded = decode_mantle_response_id(first.id)
                assert decoded is not None
                assert decoded[1] not in data, "Native Mantle ID must not leak"
        finally:
            openai_client.responses.delete(first.id)

    def test_gemma4_cancel_completed_stored_response(
        self, openai_client: OpenAI
    ) -> None:
        """Cancelling a completed Mantle-stored response proxies cleanly upstream.

        The Mantle native store answers the cancel with the completed response
        unchanged (200), so the proxy must relay that provider-shaped body
        instead of failing with a server error.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/cancel
             stdapi/routes/openai_responses.py:cancel_response
        """
        first = openai_client.responses.create(
            model=_GEMMA4, input="Say OK.", store=True, max_output_tokens=32
        )
        try:
            cancelled = openai_client.responses.cancel(first.id)
            assert cancelled.id == first.id
            assert cancelled.object == "response"
            assert cancelled.status == "completed"
        finally:
            openai_client.responses.delete(first.id)

    def test_gemma3_converted_to_chat_completions(self, openai_client: OpenAI) -> None:
        """Responses on a chat-completions-only model are converted upstream.

        The Chat Completions usage object is translated back into the Responses
        shape, so ``input_tokens``/``output_tokens`` must be populated.

        Ref: stdapi/models/chat/_mantle/_convert.py:_responses_usage_from_chat
        """
        response = openai_client.responses.create(
            model=_GEMMA3, input="Say OK.", store=False, max_output_tokens=64
        )
        assert response.status == "completed"
        assert response.output_text
        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    def test_gemma3_local_store_lifecycle_after_learned_binding(
        self, openai_client: OpenAI
    ) -> None:
        """A chat-bound Mantle model stores responses in the local session store.

        Once the model is known to serve Chat Completions only, Mantle has no
        native store to use, so the gateway falls back to its own session store.
        The ``resp-`` prefix (hyphen) is load-bearing: ``resp_`` IDs are parsed as
        region-tagged Mantle IDs and are never looked up locally.

        Ref: stdapi/routes/openai_responses.py:_require_local_response_id
             stdapi/models/chat/_mantle/_default.py:ChatModel.native_store_supported
        """
        openai_client.responses.create(
            model=_GEMMA3, input="Say OK.", store=False, max_output_tokens=32
        )
        first = openai_client.responses.create(
            model=_GEMMA3,
            input="Remember this number: 11. Just say 'Noted.'",
            store=True,
            max_output_tokens=64,
        )
        try:
            assert first.id.startswith("resp-")
            retrieved = openai_client.responses.retrieve(first.id)
            assert retrieved.id == first.id
            second = openai_client.responses.create(
                model=_GEMMA3,
                input="What number did I ask you to remember? Digits only.",
                previous_response_id=first.id,
                store=False,
                max_output_tokens=64,
            )
            assert "11" in second.output_text
        finally:
            openai_client.responses.delete(first.id)

    def test_gemma3_streamed_conversion_uses_the_route_response_id(
        self, openai_client: OpenAI
    ) -> None:
        """A converted stream announces the route ID, not a minted Mantle-form one.

        A ``resp_`` ID minted by the converter would be parsed as region-tagged and
        could then only 404 on retrieval, so every event of the stream must carry
        the single ``resp-`` ID the route assigned.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
             stdapi/routes/openai_responses.py:_require_local_response_id
        """
        events = list(
            openai_client.responses.create(
                model=_GEMMA3,
                input="Say OK.",
                store=False,
                stream=True,
                max_output_tokens=32,
            )
        )
        assert [event.type for event in events][:2] == [
            "response.created",
            "response.in_progress",
        ]
        ids = {
            response.id
            for event in events
            if (response := getattr(event, "response", None)) is not None
        }
        assert len(ids) == 1
        assert ids.pop().startswith("resp-")

    def test_gemma3_store_dropped_on_api_fallback(
        self,
        openai_client: OpenAI,
        test_client: TestClientType | None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """store=True on a Responses-to-chat fallback is served without storage.

        The learned API binding is cleared so the optimistic Responses probe runs
        and fails live; ``store`` must then be dropped rather than turned into a
        400, and the response cannot carry a region-tagged Mantle ID because
        nothing was stored upstream.

        Ref: stdapi/models/chat/_mantle/_default.py:ChatModel._select_api
        """
        if test_client is None:
            pytest.skip("Requires local test server")
        from stdapi.aws_bedrock_mantle import decode_mantle_response_id  # noqa: PLC0415
        from stdapi.models.chat._mantle import _default  # noqa: PLC0415

        # Force the optimistic Responses probe so the fallback happens live; the
        # cache is process-global, so the deletion is undone after the test.
        monkeypatch.delitem(_default._LEARNED_APIS, _GEMMA3, raising=False)  # noqa: SLF001
        response = openai_client.responses.create(
            model=_GEMMA3, input="Say OK.", store=True, max_output_tokens=64
        )
        assert response.status == "completed"
        assert response.output_text
        assert decode_mantle_response_id(response.id) is None

    def test_gemma4_input_items_not_available(self, openai_client: OpenAI) -> None:
        """Input item listing of a Mantle-stored response yields a clear 404.

        Bedrock Mantle native storage does not serve input item listings on
        either routing surface; the proxy reports it explicitly instead of a
        bare not-found.

        Ref: https://developers.openai.com/api/reference/resources/responses/subresources/input_items/methods/list
             stdapi/routes/openai_responses.py:list_response_input_items
        """
        first = openai_client.responses.create(
            model=_GEMMA4, input="Say OK.", store=True, max_output_tokens=32
        )
        try:
            with pytest.raises(NotFoundError) as missing:
                openai_client.responses.input_items.list(first.id)
            assert missing.value.status_code == 404
            assert "input item listings" in str(missing.value)
        finally:
            openai_client.responses.delete(first.id)

    def test_gemma4_surface_self_heal(
        self,
        openai_client: OpenAI,
        test_client: TestClientType | None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stale learned routing surface self-heals on the next request.

        Hitting the wrong Mantle surface does not return a clean 404 — it returns
        a "isn't supported on this route" message or a misleading 401 — so the
        cache is poisoned to ``/v1`` to check the request still succeeds and the
        learned value is corrected back to ``/openai/v1``.

        Ref: stdapi/aws_bedrock_mantle.py:_map_error
             stdapi/models/chat/_mantle/_default.py:ChatModel._api_paths
        """
        if test_client is None:
            pytest.skip("Requires local test server")
        from stdapi.models.chat._mantle import _default  # noqa: PLC0415

        monkeypatch.setitem(_default._LEARNED_SURFACE, _GEMMA4, "/v1")  # noqa: SLF001
        response = openai_client.chat.completions.create(
            model=_GEMMA4,
            messages=[{"role": "user", "content": "Say OK."}],
            max_completion_tokens=32,
        )
        assert response.choices[0].message.content
        assert _default._LEARNED_SURFACE[_GEMMA4] == "/openai/v1"  # noqa: SLF001


class TestMantleMessages:
    """Anthropic Messages route served by Mantle, natively and via conversions.

    Mantle's Messages path ``/anthropic/v1/messages`` is absolute rather than
    relative to a surface, and takes the version as the HTTP header
    ``anthropic-version: 2023-06-01`` (bedrock-runtime instead wants the body
    field ``anthropic_version: bedrock-2023-05-31``).

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-messages-api.html
         https://platform.claude.com/docs/en/api/messages
         stdapi/aws_bedrock_mantle.py:API_PATHS
    """

    def test_claude_messages_passthrough(self, anthropic_client: Anthropic) -> None:
        """Messages passthrough on a Mantle-only Claude model returns content.

        The undated Claude IDs exist only in the Mantle catalog, so this
        exercises the native ``/anthropic/v1/messages`` upstream path.
        """
        message = anthropic_client.messages.create(
            model=_CLAUDE_MANTLE,
            max_tokens=32,
            messages=[{"role": "user", "content": "Say OK."}],
        )
        assert message.role == "assistant"
        assert message.content
        assert message.content[0].type == "text"
        assert message.content[0].text
        assert message.stop_reason
        assert message.usage.input_tokens > 0
        assert message.usage.output_tokens > 0

    def test_claude_messages_passthrough_streaming(
        self, anthropic_client: Anthropic
    ) -> None:
        """Streaming Messages passthrough relays the native event grammar.

        The SDK's ``stream`` helper only reassembles a final message when the
        relayed SSE events form a valid Anthropic event sequence.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/aws_bedrock_mantle.py:_iter_sse
        """
        with anthropic_client.messages.stream(
            model=_CLAUDE_MANTLE,
            max_tokens=32,
            messages=[{"role": "user", "content": "Say OK."}],
        ) as stream:
            text = "".join(stream.text_stream)
            final = stream.get_final_message()
        assert text
        assert final.stop_reason
        assert final.usage.output_tokens > 0

    def test_gemma3_converted_to_chat_completions(
        self, anthropic_client: Anthropic
    ) -> None:
        """Messages on a chat-completions-only model are converted upstream.

        Anthropic's ``input_tokens`` already excludes cache reads, so the Chat
        Completions counts are mapped across without the subtraction the
        OpenAI-shaped paths need.

        Ref: stdapi/models/chat/_mantle/_convert.py:_messages_usage_from_chat
        """
        message = anthropic_client.messages.create(
            model=_GEMMA3,
            max_tokens=64,
            messages=[{"role": "user", "content": "Say OK."}],
        )
        assert message.role == "assistant"
        assert message.content
        assert message.content[0].type == "text"
        assert message.content[0].text
        assert message.usage.input_tokens > 0
        assert message.usage.output_tokens > 0

    def test_gemma3_long_user_id_hashed_not_rejected(
        self, anthropic_client: Anthropic
    ) -> None:
        """A metadata.user_id above the OpenAI 64-char cap is hashed, not rejected.

        OpenAI caps ``user`` at 64 characters while Anthropic allows longer IDs,
        so the conversion substitutes the SHA-256 hex digest (exactly 64 chars,
        deterministic per ID) instead of letting upstream 400 the request.

        Ref: stdapi/models/chat/_mantle/_convert.py:_openai_user
        """
        message = anthropic_client.messages.create(
            model=_GEMMA3,
            max_tokens=32,
            messages=[{"role": "user", "content": "Say OK."}],
            metadata={"user_id": "u" * 150},
        )
        assert message.role == "assistant"
        assert message.content
        assert message.content[0].type == "text"
        assert message.content[0].text

    def test_gemma4_property_names_tool_schema_sanitized(
        self, anthropic_client: Anthropic
    ) -> None:
        """A tool schema using `propertyNames` no longer empties the generation.

        Open-weight tool templates emit an empty generation — not an error — when
        the schema carries this keyword, so it is stripped recursively before the
        request leaves the gateway.

        Ref: stdapi/models/chat/_mantle/_convert.py:sanitize_tool_schema
        """
        message = anthropic_client.messages.create(
            model=_GEMMA4,
            max_tokens=64,
            messages=[{"role": "user", "content": "Reply with exactly: hi"}],
            tools=[
                {
                    "name": "record_metadata",
                    "description": "Record free-form metadata.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "metadata": {
                                "type": "object",
                                "propertyNames": {"type": "string"},
                            }
                        },
                    },
                }
            ],
        )
        assert message.content, "Generation must not be silently empty"
        assert any(
            block.type == "tool_use" or (block.type == "text" and bool(block.text))
            for block in message.content
        )

    def test_gemma3_server_tool_rejected_on_conversion(
        self, anthropic_client: Anthropic
    ) -> None:
        """An Anthropic server tool on a conversion path yields a clean 400.

        Anthropic server tools such as ``web_search`` have no Chat Completions
        equivalent, so a request that must be converted is refused up front
        rather than silently losing the tool.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
             stdapi/models/chat/_mantle/_convert.py:_chat_tools_from_anthropic
        """
        with pytest.raises(AnthropicBadRequestError) as bad_request:
            anthropic_client.messages.create(
                model=_GEMMA3,
                max_tokens=32,
                messages=[{"role": "user", "content": "Say OK."}],
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
            )
        assert bad_request.value.status_code == 400
        assert "server tools are not supported" in str(bad_request.value)

    @pytest.mark.slow
    def test_luna_converted_to_responses(self, anthropic_client: Anthropic) -> None:
        """Messages on a Responses-only model are converted upstream.

        Reasoning output must not be surfaced as the only content: an Anthropic
        text block has to survive the Responses-to-Messages translation.

        Ref: stdapi/models/chat/_mantle/_convert.py:messages_payload
        """
        message = anthropic_client.messages.create(
            model=_LUNA,
            max_tokens=_LUNA_MAX_TOKENS,
            messages=[{"role": "user", "content": "Reply with the single word: hi"}],
        )
        assert message.role == "assistant"
        text_blocks = [block for block in message.content if block.type == "text"]
        assert text_blocks
        assert text_blocks[0].text
        assert message.usage.output_tokens > 0


class TestMantleCompletions:
    """Legacy /v1/completions served by Mantle (always converted).

    Mantle serves no completions API, so the prompt is always folded into a chat
    message; the legacy surface's ``finish_reason`` set is therefore limited to
    ``stop``/``length``.

    Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
         stdapi/models/chat/_mantle/_convert.py:text_completion_as_chat_payload
    """

    def test_gemma3_legacy_completion(self, openai_client: OpenAI) -> None:
        """Legacy completions convert the prompt to chat messages upstream."""
        response = openai_client.completions.create(
            model=_GEMMA3, prompt="Say OK.", max_tokens=16
        )
        assert response.choices
        assert response.choices[0].text
        assert response.choices[0].finish_reason in ("stop", "length")
        assert response.usage is not None
        assert response.usage.completion_tokens > 0

    def test_gemma3_echo_rejected(self, openai_client: OpenAI) -> None:
        """The unsupported `echo` option is rejected with the documented 400.

        ``echo`` requires the prompt tokens back in the completion, which no
        chat-shaped upstream can produce, so it is refused instead of ignored.

        Ref: stdapi/models/chat/_mantle/_convert.py:text_completion_as_chat_payload
        """
        with pytest.raises(BadRequestError) as bad_request:
            openai_client.completions.create(
                model=_GEMMA3, prompt="Say OK.", max_tokens=16, echo=True
            )
        assert bad_request.value.status_code == 400
        assert "`echo` is not supported" in str(bad_request.value)


class TestMantleResponsesSiblingGuards:
    """Guard rails on the Responses sibling routes for Mantle models and IDs.

    Ref: stdapi/routes/openai_responses.py:count_input_tokens
         stdapi/routes/openai_responses.py:_decode_mantle_id
    """

    def test_input_tokens_and_undecodable_id_guards(
        self, openai_client: OpenAI
    ) -> None:
        """input_tokens rejects Mantle models (400); undecodable IDs are 404.

        Token counting needs Bedrock ``CountTokens``, which exists on
        bedrock-runtime only. And a ``resp_`` ID whose Region fingerprint does not
        match any configured Mantle Region must be a 404, never a 500 or a
        fall-through to the local store.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html
             stdapi/aws_bedrock_mantle.py:decode_mantle_response_id
        """
        with pytest.raises(BadRequestError) as bad_request:
            openai_client.responses.input_tokens.count(model=_GEMMA3, input="Hello")
        assert bad_request.value.status_code == 400
        assert "not supported for Bedrock Mantle" in str(bad_request.value)
        with pytest.raises(NotFoundError) as undecodable:
            openai_client.responses.retrieve("resp_notdecodable")
        assert undecodable.value.status_code == 404


class TestMantleCountTokens:
    """Anthropic count_tokens proxied to the Mantle endpoint.

    Ref: https://platform.claude.com/docs/en/api/messages/count_tokens
         stdapi/aws_bedrock_mantle.py:_map_error
    """

    def test_count_tokens_upstream_error_shape(
        self, anthropic_client: Anthropic
    ) -> None:
        """Models unsupported by the upstream counter yield a clean 400 error.

        The failure is mapped into Anthropic's ``{"type": "error", "error": {...}}``
        envelope rather than being relayed as a Mantle-shaped body or a 500.

        Ref: https://platform.claude.com/docs/en/api/errors
        """
        with pytest.raises(AnthropicBadRequestError) as exc_info:
            anthropic_client.messages.count_tokens(
                model=_GEMMA3, messages=[{"role": "user", "content": "Hello"}]
            )
        error = exc_info.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["message"]
