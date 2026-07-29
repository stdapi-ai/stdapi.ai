"""Live tests for chat models served by the AWS Bedrock Mantle endpoint.

Exercises the Mantle passthrough and conversion paths against real AWS:
model discovery, chat completions (native and converted), the Responses API
with native storage, the Anthropic Messages route, legacy completions, and
the count_tokens proxy. Uses the cheapest verified models; the expensive
Responses-only reasoning model (``openai.gpt-5.6-luna``) appears only where
a Responses conversion target is required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from anthropic import BadRequestError as AnthropicBadRequestError
from openai import BadRequestError, NotFoundError, OpenAI

from tests.conftest import logged_usage_entries

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
    """Mantle model discovery and registry merge behavior."""

    def test_mantle_models_registered(
        self, test_client: TestClientType | None, listed_model_ids: set[str]
    ) -> None:
        """Mantle models are discovered, registered, and listed.

        Validates:
            - MANTLE_MODELS is populated from the Mantle catalog
            - The verified test models are present and Mantle-served
            - Mantle-only models appear in the public model listing
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
        # Gemma 3 is dual-homed: it is Mantle-served only because the test
        # environment lists it in aws_bedrock_mantle_preferred_models.
        assert is_mantle_preferred(_GEMMA3)

    def test_dual_homed_models_stay_on_runtime(
        self, test_client: TestClientType | None
    ) -> None:
        """Models on both endpoints are served by bedrock-runtime by default.

        Validates:
            - Some Mantle catalog entries stay runtime-served (dual-homed)
            - None of them are Mantle-preferred (which would flip priority)
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
    """Chat Completions served by Mantle (passthrough and conversions)."""

    def test_gemma3_passthrough(self, openai_client: OpenAI) -> None:
        """Chat completion passthrough on the /v1 surface returns content and usage."""
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
        assert response.usage.total_tokens > 0

    def test_gemma4_openai_v1_surface(self, openai_client: OpenAI) -> None:
        """Chat completion passthrough on the /openai/v1 surface works."""
        response = openai_client.chat.completions.create(
            model=_GEMMA4,
            messages=[{"role": "user", "content": "Say OK."}],
            max_completion_tokens=32,
        )
        assert response.choices[0].message.content
        assert response.usage is not None
        assert response.usage.completion_tokens > 0

    def test_gemma3_streaming_usage_stripped(
        self,
        openai_client: OpenAI,
        test_client: TestClientType | None,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """Streaming without include_usage strips the forced usage chunk.

        Usage is still recorded server-side from the stripped chunk.
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
        """Streaming with include_usage relays the final usage chunk."""
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
        """An upstream-rejected parameter surfaces as a clean OpenAI-shaped 400.

        Validates:
            - Passthrough relays the upstream 4xx instead of a 500
            - The error envelope carries the upstream message, param, and code
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

        Anthropic carries the switch as ``disable_parallel_tool_use`` on the
        tool choice, so the conversion synthesises one; this checks the
        upstream Messages API accepts it instead of returning a 400.

        Validates:
            - The converted request succeeds on the Anthropic upstream
            - At most one tool call comes back
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
        assert len(response.choices[0].message.tool_calls or []) <= 1

    def test_luna_converted_to_responses(self, openai_client: OpenAI) -> None:
        """Chat completion on a Responses-only model is converted upstream.

        Validates content and the Responses-to-Chat-Completions usage mapping.
        """
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

    def test_luna_converted_streaming(self, openai_client: OpenAI) -> None:
        """A streamed chat completion on a Responses-only model converts back.

        Validates the chat-inbound converted streaming cell: Responses events
        from upstream are relayed as Chat Completions chunks with usage.
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
    """Per-request Mantle routing via the gated ``x-stdapi-service`` header."""

    def test_header_routes_dual_homed_model_to_mantle(
        self,
        openai_client: OpenAI,
        test_client: TestClientType | None,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """The header flips a runtime-served dual-homed model onto Mantle.

        Validates:
            - Generation succeeds through the Mantle endpoint
            - Billed usage is recorded under the bedrock-mantle service
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


class TestMantleRecordedBilling:
    """Server-side usage recording on the converted (non-passthrough) paths."""

    @pytest.mark.slow
    def test_luna_conversion_records_usage(
        self,
        openai_client: OpenAI,
        test_client: TestClientType | None,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A non-streaming chat-to-responses conversion records billed usage."""
        if test_client is None:
            pytest.skip("Requires local test server")
        capfd.readouterr()
        response = openai_client.chat.completions.create(
            model=_LUNA,
            messages=[{"role": "user", "content": "Reply with the single word: hi"}],
            max_completion_tokens=_LUNA_MAX_TOKENS,
        )
        assert response.usage is not None
        entries = logged_usage_entries(
            capfd.readouterr().out, service=_MANTLE_USAGE_SERVICE, model=_LUNA
        )
        assert len(entries) == 1, "Usage must be recorded exactly once"
        assert entries[0]["input_tokens"] > 0
        assert entries[0]["output_tokens"] > 0

    def test_gemma3_streamed_conversion_records_usage(
        self,
        openai_client: OpenAI,
        test_client: TestClientType | None,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A streamed responses-to-chat conversion records billed usage."""
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
    """Responses API served by Mantle, including native storage."""

    def test_gemma4_stored_response_lifecycle(
        self, openai_client: OpenAI, test_client: TestClientType | None
    ) -> None:
        """store=True uses Mantle native storage with region-tagged IDs.

        Validates:
            - Created response ID is a region-tagged Mantle ID
            - GET retrieves the stored response by public ID
            - Chaining via previous_response_id recalls earlier info
            - DELETE removes the stored response (404 afterwards)
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
        with pytest.raises(NotFoundError):
            openai_client.responses.retrieve(first.id)

    def test_gemma4_streamed_chained_public_previous_id(
        self, openai_client: OpenAI, test_client: TestClientType | None
    ) -> None:
        """A streamed chained request echoes the public ID, never the native one."""
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
        """Responses on a chat-completions-only model are converted upstream."""
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

        Validates:
            - Once the chat binding is learned, store=True returns a local
              ``resp-`` ID retrievable through the local response store
            - previous_response_id continues the conversation via local merge
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

        Validates:
            - ``response.created`` is followed by ``response.in_progress``
            - Every event carries the same route-assigned ``resp-`` ID (a
              minted ``resp_`` one is parsed as Mantle-tagged and only 404s)
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
            event.response.id for event in events if getattr(event, "response", None)
        }
        assert len(ids) == 1
        assert ids.pop().startswith("resp-")

    def test_gemma3_store_dropped_on_api_fallback(
        self, openai_client: OpenAI, test_client: TestClientType | None
    ) -> None:
        """store=True on a Responses-to-chat fallback is served without storage.

        Validates:
            - The request succeeds (no 400) despite store being unsupported
            - The returned ID is not a region-tagged Mantle stored ID
        """
        if test_client is None:
            pytest.skip("Requires local test server")
        from stdapi.aws_bedrock_mantle import decode_mantle_response_id  # noqa: PLC0415
        from stdapi.models.chat._mantle import _default  # noqa: PLC0415

        # Force the optimistic Responses probe so the fallback happens live.
        _default._LEARNED_APIS.pop(_GEMMA3, None)  # noqa: SLF001
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
        """
        first = openai_client.responses.create(
            model=_GEMMA4, input="Say OK.", store=True, max_output_tokens=32
        )
        try:
            with pytest.raises(NotFoundError) as missing:
                openai_client.responses.input_items.list(first.id)
            assert "input item listings" in str(missing.value)
        finally:
            openai_client.responses.delete(first.id)

    def test_gemma4_surface_self_heal(
        self, openai_client: OpenAI, test_client: TestClientType | None
    ) -> None:
        """A stale learned routing surface self-heals on the next request.

        Validates:
            - A request still succeeds after the learned surface is poisoned
            - The learned surface is corrected back by the fallback probe
        """
        if test_client is None:
            pytest.skip("Requires local test server")
        from stdapi.models.chat._mantle import _default  # noqa: PLC0415

        _default._LEARNED_SURFACE[_GEMMA4] = "/v1"  # noqa: SLF001 (poisoned)
        response = openai_client.chat.completions.create(
            model=_GEMMA4,
            messages=[{"role": "user", "content": "Say OK."}],
            max_completion_tokens=32,
        )
        assert response.choices[0].message.content
        assert _default._LEARNED_SURFACE[_GEMMA4] == "/openai/v1"  # noqa: SLF001


class TestMantleMessages:
    """Anthropic Messages route served by Mantle via conversions."""

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
        """Streaming Messages passthrough relays the native event grammar."""
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
        """Messages on a chat-completions-only model are converted upstream."""
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

        The conversion replaces over-long IDs with their SHA-256 digest so the
        upstream `user` field stays valid (previously a live 400).
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

        The unsupported keyword is stripped before reaching Mantle, which
        previously returned a silently empty completion.
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
        """An Anthropic server tool on a conversion path yields a clean 400."""
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
        """Messages on a Responses-only model are converted upstream."""
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
    """Legacy /v1/completions served by Mantle (always converted)."""

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
        """The unsupported `echo` option is rejected with the documented 400."""
        with pytest.raises(BadRequestError) as bad_request:
            openai_client.completions.create(
                model=_GEMMA3, prompt="Say OK.", max_tokens=16, echo=True
            )
        assert bad_request.value.status_code == 400
        assert "`echo` is not supported" in str(bad_request.value)


class TestMantleResponsesSiblingGuards:
    """Guard rails on the Responses sibling routes for Mantle models and IDs."""

    def test_input_tokens_and_undecodable_id_guards(
        self, openai_client: OpenAI
    ) -> None:
        """input_tokens rejects Mantle models (400); undecodable IDs are 404.

        Validates:
            - POST /v1/responses/input_tokens on a Mantle-only model returns a
              clean 400 with the documented message
            - GET /v1/responses/{id} with an undecodable ``resp_`` ID returns
              404 instead of a server error
        """
        with pytest.raises(BadRequestError) as bad_request:
            openai_client.responses.input_tokens.count(model=_GEMMA3, input="Hello")
        assert bad_request.value.status_code == 400
        assert "not supported for Bedrock Mantle" in str(bad_request.value)
        with pytest.raises(NotFoundError):
            openai_client.responses.retrieve("resp_notdecodable")


class TestMantleCountTokens:
    """Anthropic count_tokens proxied to the Mantle endpoint."""

    def test_count_tokens_upstream_error_shape(
        self, anthropic_client: Anthropic
    ) -> None:
        """Models unsupported by the upstream counter yield a clean 400 error."""
        with pytest.raises(AnthropicBadRequestError) as exc_info:
            anthropic_client.messages.count_tokens(
                model=_GEMMA3, messages=[{"role": "user", "content": "Hello"}]
            )
        error = exc_info.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "error"
        assert body["error"]["message"]
