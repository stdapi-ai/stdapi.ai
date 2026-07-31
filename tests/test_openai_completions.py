"""Tests for the OpenAI /v1/completions route (legacy Completions API).

The legacy surface has no ``messages``, tools, ``response_format`` or
``max_completion_tokens``, and its ``finish_reason`` enum is only
``stop``/``length``/``content_filter``.  On top of that upstream contract this
gateway accepts prompt shapes OpenAI never defined (URL, ``s3://``, ``data:``
and ``file-id:`` references, plus a one-text-plus-files multimodal packing) and
fans a batch prompt out into one Bedrock Converse call per prompt.

Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
     https://developers.openai.com/api/docs/guides/completions
     stdapi/routes/openai_completions.py:create_completion
     stdapi/models/chat/_adapters/_openai_completion.py
"""

import io
import json
from typing import TYPE_CHECKING, Any, cast

import pytest
from openai import BadRequestError, NotFoundError, OpenAI
from pybase64 import b64encode
from starlette.requests import Request

from stdapi.aws_bedrock import PROMPT_CACHING_DEFAULT
from stdapi.models.chat._adapters._openai_completion import (
    _FINISH_REASONS as _LEGACY_FINISH_REASONS,
)
from stdapi.models.chat._adapters._openai_completion import (
    _map_finish_reason,
    build_user_messages,
    format_stream,
    translate_request,
)
from stdapi.models.chat._mantle._convert import chat_response_as_text_completion
from stdapi.models.chat.anthropic_claude_46 import ChatModel as ClaudeChatModel
from stdapi.monitoring import REQUEST
from stdapi.types.openai_completions import Completion, CompletionCreateParams

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Generator

    from starlette.testclient import TestClient as TestClientType
    from types_aiobotocore_bedrock_runtime.type_defs import ConverseResponseTypeDef

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef
    from stdapi.types.openai_chat_completions import ServiceTiers

#: Every ``finish_reason`` the legacy adapter can emit for a successful request.
_TERMINAL_REASONS = {"stop", "length"}

#: PNG file signature, the shortest byte string typed as an image by the file detector.
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class TestCompletions:
    """Live tests for ``POST /v1/completions`` against the configured model.

    Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
         stdapi/models/chat/_adapters/_openai_completion.py:format_response
    """

    def test_basic_single_prompt_returns_text(
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """A single string prompt returns one ``text_completion`` choice with usage.

        ``max_tokens`` is forwarded as the Bedrock ``maxTokens`` inference field,
        so the reported completion tokens cannot exceed it.

        Ref: stdapi/models/chat/_adapters/_openai_completion.py:translate_request
        """
        response = openai_client.completions.create(
            model=completion_model, prompt="The capital of France is", max_tokens=20
        )

        assert response.object == "text_completion"
        assert response.id.startswith("cmpl-")
        assert len(response.choices) == 1
        assert response.choices[0].index == 0
        assert response.choices[0].text
        assert response.choices[0].finish_reason in _TERMINAL_REASONS

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.completion_tokens <= 20, "max_tokens was not applied"
        assert (
            response.usage.total_tokens
            == response.usage.prompt_tokens + response.usage.completion_tokens
        )

    def test_list_prompt_returns_one_choice_per_prompt(
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """A list prompt returns one choice per prompt, indexed in input order.

        Each prompt is a separate Bedrock Converse call, so the per-call usage is
        summed and ``max_tokens`` applies to every prompt independently.

        Ref: stdapi/models/chat/_adapters/_openai_completion.py:build_user_messages
        """
        response = openai_client.completions.create(
            model=completion_model,
            prompt=["One plus one is", "Two plus two is"],
            max_tokens=5,
        )

        assert len(response.choices) == 2
        assert [choice.index for choice in response.choices] == [0, 1]
        assert all(choice.text for choice in response.choices)
        assert all(
            choice.finish_reason in _TERMINAL_REASONS for choice in response.choices
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens <= 2 * 5, (
            "usage must be the sum of the two per-prompt completions"
        )
        assert (
            response.usage.total_tokens
            == response.usage.prompt_tokens + response.usage.completion_tokens
        )

    def test_n_gt_1_returns_multiple_choices(
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """``n=2`` returns two independently indexed choices for one prompt.

        Bedrock Converse has no ``n``, so the gateway issues the request twice
        and both prompts are billed: the summed prompt tokens cover both calls.

        Ref: stdapi/models/chat/_default.py:ChatModel.create_text_completion
        """
        response = openai_client.completions.create(
            model=completion_model, prompt="Hello", n=2, max_tokens=3
        )

        assert len(response.choices) == 2
        assert [choice.index for choice in response.choices] == [0, 1]
        assert all(choice.text is not None for choice in response.choices)
        assert all(
            choice.finish_reason in _TERMINAL_REASONS for choice in response.choices
        )

        assert response.usage is not None
        assert response.usage.completion_tokens <= 2 * 3
        assert (
            response.usage.total_tokens
            == response.usage.prompt_tokens + response.usage.completion_tokens
        )

    def test_streaming_yields_text_deltas_and_terminal_finish_reason(
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """Streaming yields text deltas, then exactly one terminal ``finish_reason``.

        ``choices[0].text`` is a delta the client concatenates, and the
        finish-reason chunk is the last one because ``include_usage`` was not
        requested — every chunk therefore carries no ``usage``.

        Ref: stdapi/models/chat/_adapters/_openai_completion.py:format_stream
        """
        response = openai_client.completions.create(
            model=completion_model,
            prompt="Tell me a short story",
            max_tokens=30,
            stream=True,
        )

        chunks = list(response)

        assert chunks
        assert all(len(chunk.choices) == 1 for chunk in chunks), (
            "each completion chunk carries exactly one choice"
        )

        text_fragments = [
            chunk.choices[0].text for chunk in chunks if chunk.choices[0].text
        ]
        finish_reasons = [
            chunk.choices[0].finish_reason
            for chunk in chunks
            if chunk.choices[0].finish_reason is not None
        ]

        assert all(chunk.object == "text_completion" for chunk in chunks)
        assert all(chunk.choices[0].index == 0 for chunk in chunks)
        assert all(chunk.usage is None for chunk in chunks), (
            "usage must only be sent when stream_options.include_usage is set"
        )
        assert "".join(text_fragments) != ""
        assert len(finish_reasons) == 1
        assert finish_reasons[0] in _TERMINAL_REASONS
        assert chunks[-1].choices[0].finish_reason is not None

    def test_streaming_include_usage_final_chunk_has_usage(
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """``stream_options.include_usage`` populates usage on the final chunk only.

        Usage is aggregated from the Bedrock ``metadata`` stream events and
        attached to the last terminal chunk; all preceding chunks keep a null
        ``usage``.

        Ref: stdapi/models/chat/_adapters/_openai_common.py:extract_stream_usage
        """
        response = openai_client.completions.create(
            model=completion_model,
            prompt="Hello world",
            max_tokens=20,
            stream=True,
            stream_options={"include_usage": True},
        )

        chunk_list = list(response)

        assert chunk_list
        assert sum(1 for chunk in chunk_list if chunk.usage is not None) == 1, (
            "usage must be sent once, on the final chunk"
        )
        assert all(chunk.usage is None for chunk in chunk_list[:-1])

        usage = chunk_list[-1].usage
        assert usage is not None
        assert usage.prompt_tokens > 0
        assert usage.completion_tokens > 0
        assert usage.completion_tokens <= 20
        assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens

    def test_streaming_multi_prompt_interleaves_choices(
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """Streaming with a list prompt produces chunks for each prompt index.

        Each SSE chunk carries a single choice whose ``index`` identifies the
        originating prompt.  Every prompt must receive at least one delta
        chunk and exactly one terminal chunk with ``finish_reason`` set.

        Ref: stdapi/models/chat/_adapters/_openai_completion.py:format_stream
        """
        prompts = ["One plus one is", "The sky color is"]
        response = openai_client.completions.create(
            model=completion_model, prompt=prompts, max_tokens=10, stream=True
        )

        deltas_per_index: dict[int, list[str]] = {i: [] for i in range(len(prompts))}
        finish_per_index: dict[int, str | None] = dict.fromkeys(range(len(prompts)))

        for chunk in response:
            assert chunk.object == "text_completion"
            choice = chunk.choices[0]
            assert 0 <= choice.index < len(prompts)
            if choice.text:
                deltas_per_index[choice.index].append(choice.text)
            if choice.finish_reason is not None:
                assert finish_per_index[choice.index] is None
                finish_per_index[choice.index] = choice.finish_reason

        for index in range(len(prompts)):
            assert deltas_per_index[index], f"no deltas for prompt {index}"
            assert finish_per_index[index] in _TERMINAL_REASONS

    def test_streaming_n_gt_1_yields_one_terminal_chunk_per_choice(
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """Streaming with ``n > 1`` returns deltas for each choice index.

        Each choice gets a distinct ``choices[0].index`` (``0..n-1``) and
        exactly one terminal chunk with ``finish_reason`` per index.

        Ref: stdapi/models/chat/_adapters/_openai_completion.py:format_stream
        """
        n = 2
        response = openai_client.completions.create(
            model=completion_model, prompt="Hello", n=n, max_tokens=5, stream=True
        )

        deltas_per_index: dict[int, list[str]] = {i: [] for i in range(n)}
        finish_per_index: dict[int, str | None] = dict.fromkeys(range(n))

        for chunk in response:
            choice = chunk.choices[0]
            assert 0 <= choice.index < n
            if choice.text:
                deltas_per_index[choice.index].append(choice.text)
            if choice.finish_reason is not None:
                assert finish_per_index[choice.index] is None
                finish_per_index[choice.index] = choice.finish_reason

        for index in range(n):
            assert deltas_per_index[index], f"no deltas for choice {index}"
            assert finish_per_index[index] in _TERMINAL_REASONS

    def test_stop_sequences_truncate_generation(
        self, openai_client: OpenAI, completion_model: str, use_official_api: bool
    ) -> None:
        """A stop sequence halts generation at its first occurrence.

        The prompt asks for several sentences within a generous token budget, so a
        completion holding at most one ``"."`` proves the model stopped on the
        sequence rather than on ``max_tokens``.  OpenAI honors its documented
        contract and strips the matched sequence, so the returned text contains no
        ``"."`` at all; Bedrock instead keeps the matched sequence in the text and
        reports ``stopReason="end_turn"``, mapped to ``finish_reason`` ``"stop"``.

        Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/types/openai_completions.py:CompletionCreateParams
             stdapi/models/chat/_adapters/_openai_completion.py:_map_finish_reason
        """
        response = openai_client.completions.create(
            model=completion_model,
            prompt="Write three complete sentences about cats",
            stop=["."],
            max_tokens=500,
        )

        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "stop"
        text = response.choices[0].text
        expected_occurrences = 0 if use_official_api else 1
        assert text.count(".") == expected_occurrences, (
            f"generation must halt at the first stop sequence: {text!r}"
        )
        if use_official_api:
            assert text, f"the truncated completion must still carry text: {text!r}"
        else:
            assert text.endswith("."), (
                f"Bedrock keeps the matched stop sequence in the text: {text!r}"
            )
        assert response.usage is not None
        assert response.usage.completion_tokens < 500

    def test_max_tokens_limit_yields_length_finish_reason(
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """``max_tokens=1`` truncates the completion and reports ``finish_reason="length"``.

        Bedrock reports ``stopReason="max_tokens"``, which the legacy adapter
        maps to ``length`` (the legacy enum has no ``tool_calls``).

        Ref: stdapi/models/chat/_adapters/_openai_completion.py:_map_finish_reason
        """
        response = openai_client.completions.create(
            model=completion_model, prompt="Tell me a long story", max_tokens=1
        )

        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == "length"
        assert response.usage is not None
        assert response.usage.completion_tokens == 1

    def test_unknown_model_returns_404(self, openai_client: OpenAI) -> None:
        """An unknown model is a 404 ``model_not_found`` before any inference call.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
             stdapi/api_errors.py:UnsupportedModelError
        """
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.completions.create(model="does-not-exist-xyz", prompt="Hello")

        error = exc_info.value
        assert error.status_code == 404
        assert error.code == "model_not_found"
        assert error.type == "invalid_request_error"
        assert "does not exist" in error.message
        assert "does-not-exist-xyz" in error.message

    def test_unsupported_params_are_silently_accepted(
        self, openai_client: OpenAI, completion_model: str, use_official_api: bool
    ) -> None:
        """Upstream-only legacy params are accepted instead of returning a 400.

        ``best_of``, ``logprobs`` and ``suffix`` have no Bedrock Converse
        equivalent and are silently ignored -- ``logprobs`` in particular
        never produces a ``logprobs`` object on a choice. The penalties and
        ``logit_bias`` do reach the model, via ``additionalModelRequestFields``,
        exactly like on ``/v1/chat/completions``; a model that honors them
        accepts the in-range values used here without error.

        Ref: stdapi/types/openai_completions.py:CompletionCreateParams
             stdapi/models/chat/_adapters/_openai_completion.py:translate_request
        """
        response = openai_client.completions.create(
            model=completion_model,
            prompt="Test prompt",
            max_tokens=10,
            best_of=1,
            logprobs=0,
            suffix=" end",
            frequency_penalty=0.1,
            presence_penalty=0.1,
            logit_bias={"50256": -100},
        )

        assert response.object == "text_completion"
        assert len(response.choices) == 1
        assert response.choices[0].text
        assert response.choices[0].finish_reason in _TERMINAL_REASONS
        assert response.usage is not None
        assert response.usage.completion_tokens <= 10
        if not use_official_api:
            assert response.choices[0].logprobs is None, (
                "logprobs is documented UNSUPPORTED and must be ignored, not honored"
            )

    def test_input_file_url_prompt_via_file_id(
        self, openai_client: OpenAI, completion_model: str, use_official_api: bool
    ) -> None:
        """Reference an uploaded file via the ``file-id:`` URI scheme.

        Uploads a small text file via the Files API, then references it using
        ``file-id:<file-id>`` as the prompt.  stdapi forwards the file to the
        model as a ``document`` block (its detected modality).  The default
        completion model does not support document inputs, so the upstream
        model rejects the request with ``ValidationException``, mapped to a 400
        ``invalid_request_error`` — proving the file-id resolver and the adapter
        passed the document through instead of failing earlier.

        Ref: stdapi/input_file.py:InputFileUrl.to_bedrock_content_block
             stdapi/aws_bedrock.py:handle_bedrock_client_error
        """
        if use_official_api:
            pytest.skip("file-id: is a project-local URI scheme")

        text_content = b"Hello, this is a test file content."
        uploaded = openai_client.files.create(
            file=("test.txt", io.BytesIO(text_content), "text/plain"),
            purpose="assistants",
        )
        try:
            with pytest.raises(BadRequestError) as exc_info:
                openai_client.completions.create(
                    model=completion_model,
                    prompt=f"file-id:{uploaded.id}",
                    max_tokens=20,
                )
            error = exc_info.value
            assert error.status_code == 400
            assert error.type == "invalid_request_error"
            assert "document" in str(error).lower()
        finally:
            openai_client.files.delete(uploaded.id)

    @pytest.mark.parametrize(
        "with_text",
        [
            pytest.param(True, id="text_plus_image"),
            pytest.param(False, id="image_only"),
        ],
    )
    def test_multimodal_prompt_returns_single_choice(
        self,
        openai_client: OpenAI,
        chat_vision_model: str,
        sample_image_file_base64: str,
        use_official_api: bool,
        with_text: bool,
    ) -> None:
        """A text+image prompt and a lone image both yield exactly one choice.

        The list form (one question plus one ``data:image/png;base64,...`` URI)
        collapses into a single multimodal Bedrock message, and the bare image is
        a single ``image`` block; a per-element fan-out would have produced two
        choices for the first case.

        Ref: stdapi/models/chat/_adapters/_openai_completion.py:build_user_messages
        """
        if use_official_api:
            pytest.skip(
                "multimodal input on v1/completions is a stdapi extension; "
                "the official legacy Completions API rejects chat/vision models"
            )
        prompt: str | list[str] = (
            [
                "Describe what is shown in this image in one short sentence:",
                sample_image_file_base64,
            ]
            if with_text
            else sample_image_file_base64
        )
        response = openai_client.completions.create(
            model=chat_vision_model, prompt=prompt, max_tokens=80
        )
        assert response.object == "text_completion"
        assert len(response.choices) == 1
        assert response.choices[0].index == 0
        assert response.choices[0].text
        assert response.choices[0].finish_reason in _TERMINAL_REASONS
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0


class TestStopSequenceValidation:
    """Offline unit tests: whitespace-only stop sequences are rejected before dispatch.

    AWS Bedrock rejects a blank ``stopSequences`` entry with a raw
    ``ValidationException``; this backend surfaces a clean 400 instead.
    Validation happens before any model dispatch or AWS call, so the
    rejection test runs against an app instance without the AWS-touching
    lifespan.

    Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
         stdapi/types/openai_completions.py:CompletionCreateParams._validate_stop_sequences
    """

    pytestmark = pytest.mark.local

    def test_whitespace_only_stop_sequence_is_rejected(
        self, app_client: TestClientType
    ) -> None:
        r"""stop=["\n"] is rejected with a clean 400 (Bedrock cannot honor blank stops).

        Upstream OpenAI accepts a newline stop sequence, so this 400 is
        gateway-specific.  The envelope always carries all four OpenAI error
        keys, ``param``/``code`` included and null here because the rejection
        comes from request validation rather than from a typed ``ApiError``.

        Ref: stdapi/api_providers/openai.py:_format_error
             stdapi/main.py:handle_validation_exception
        """
        response = app_client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "hi", "stop": ["\n"]},
        )
        assert response.status_code == 400, response.text
        error_body = response.json()
        assert error_body.keys() == {"error"}
        assert error_body["error"].keys() == {"message", "type", "param", "code"}
        assert error_body["error"]["type"] == "invalid_request_error"
        assert "whitespace" in error_body["error"]["message"].lower()


class TestLegacyFinishReasonMapping:
    """Bedrock stop reasons map onto the narrower legacy ``finish_reason`` enum.

    The legacy Completions enum has no ``tool_calls``/``function_call``, so this
    surface keeps a table of its own; an unknown reason degrades to ``stop``.

    Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         stdapi/models/chat/_adapters/_openai_completion.py:_map_finish_reason
    """

    pytestmark = pytest.mark.local

    @pytest.mark.parametrize(
        ("stop_reason", "expected"),
        [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("max_tokens", "length"),
            ("model_context_window_exceeded", "length"),
            ("incomplete", "length"),
            ("content_filtered", "content_filter"),
            ("guardrail_intervened", "content_filter"),
            ("malformed_model_output", "content_filter"),
        ],
    )
    def test_every_bedrock_stop_reason_maps_to_its_legacy_value(
        self, stop_reason: str, expected: str
    ) -> None:
        """A filtered or truncated legacy completion is never reported as ``stop``.

        Guardrail interventions and malformed model output must surface as
        ``content_filter``, and a context-window overflow as ``length``, so a
        client can tell a censored or cut-off answer from a finished one.
        """
        assert _map_finish_reason(stop_reason) == expected

    @pytest.mark.parametrize("stop_reason", ["tool_use", "not-a-real-reason", None, ""])
    def test_unknown_stop_reason_falls_back_to_stop(
        self, stop_reason: str | None
    ) -> None:
        """Reasons absent from the legacy table (``tool_use`` included) become ``stop``.

        ``tool_use`` is deliberately unmapped: the legacy enum has no
        ``tool_calls`` value.
        """
        assert _map_finish_reason(stop_reason) == "stop"

    def test_table_contains_no_chat_only_finish_reason(self) -> None:
        """The legacy table never emits ``tool_calls`` or ``function_call``."""
        assert set(_LEGACY_FINISH_REASONS.values()) == {
            "stop",
            "length",
            "content_filter",
        }


class TestLegacyRequestTranslation:
    """``translate_request`` maps the legacy request onto Converse primitives.

    Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
         stdapi/models/chat/_adapters/_openai_completion.py:translate_request
    """

    pytestmark = pytest.mark.local

    #: Bedrock model the inference config is clamped against.
    _MODEL_ID = "anthropic.claude-haiku-4-5-20251001-v1:0"

    def _translate(self, **kwargs: object) -> tuple[Any, ...]:
        """Validate a minimal completion request and translate it.

        Args:
            **kwargs: Extra request fields merged onto ``model``/``prompt``.

        Returns:
            The tuple returned by ``translate_request``.
        """
        request = CompletionCreateParams.model_validate(
            {"model": "model", "prompt": "hi", **kwargs}
        )
        return translate_request(request, self._MODEL_ID)

    @pytest.mark.parametrize(
        ("service_tier", "expected"),
        [
            ("priority", ("priority", "priority")),
            ("flex", ("flex", "flex")),
            ("reserved", ("reserved", "reserved")),
            ("auto", (None, "default")),
            ("default", (None, "default")),
            ("scale", (None, "default")),
        ],
    )
    def test_service_tier_is_forwarded_and_echoed(
        self, service_tier: str, expected: tuple[str | None, str]
    ) -> None:
        """Only the Bedrock-backed tiers reach Converse; the rest echo ``default``.

        The first element is applied to the Converse call and the second is what
        the ``Completion`` echoes back, so losing the first would silently
        downgrade a paid ``priority`` request without changing the response.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html
             stdapi/models/chat/_adapters/_openai_common.py:map_service_tier
        """
        bedrock_tier, openai_tier = self._translate(service_tier=service_tier)[2:4]
        assert (bedrock_tier, openai_tier) == expected

    def test_no_service_tier_leaves_both_sides_unset(self) -> None:
        """Omitting ``service_tier`` echoes nothing rather than ``default``."""
        assert self._translate()[2:4] == (None, None)

    @pytest.mark.parametrize(
        ("fields", "expected"),
        [
            ({"user": "u-1"}, {"user": "u-1"}),
            ({"safety_identifier": "s-1"}, {"user": "s-1"}),
            ({"user": "u-1", "safety_identifier": "s-1"}, {"user": "s-1"}),
            ({}, None),
        ],
    )
    def test_request_metadata_prefers_safety_identifier(
        self, fields: dict[str, str], expected: dict[str, str] | None
    ) -> None:
        """``safety_identifier`` wins over the deprecated ``user`` in requestMetadata.

        This is the only legacy-route hook putting a client-supplied identifier
        into Bedrock ``requestMetadata``, so it is omitted entirely when neither
        field is set rather than sent as an empty value.

        Ref: https://developers.openai.com/api/docs/guides/safety-best-practices#implement-safety-identifiers
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        """
        assert self._translate(**fields)[5] == expected

    def test_sampling_knobs_are_accepted_and_dropped(self) -> None:
        """``frequency_penalty``, ``presence_penalty``, ``logit_bias`` and ``seed`` drop.

        None of the four has a Bedrock ``inferenceConfig`` slot, so forwarding
        them would mean putting them in ``additionalModelRequestFields``, where
        the text-completion models reject them outright: measured live on
        2026-07-31, a request carrying ``frequency_penalty`` returns
        ``400 Malformed input request: #: extraneous key [frequency_penalty] is
        not permitted``. The route therefore accepts and drops them, which keeps
        an otherwise-valid request working. This diverges from the twin
        ``/v1/chat/completions`` adapter on purpose; the divergence is the
        backend's, not the gateway's.

        Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
             stdapi/models/chat/_adapters/_openai_completion.py:translate_request
        """
        additional_request_fields = self._translate(
            frequency_penalty=0.5, presence_penalty=-0.5, logit_bias={100: 5}, seed=42
        )[1]
        for knob in ("frequency_penalty", "presence_penalty", "logit_bias", "seed"):
            assert knob not in additional_request_fields


class TestLegacyPromptFanOut:
    """``build_user_messages`` decides how many Bedrock calls a prompt becomes.

    A list holding exactly one string plus one or more file references collapses
    into a single multimodal message; anything else fans out one message — and
    therefore one choice — per element.

    Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
         stdapi/models/chat/_adapters/_openai_completion.py:build_user_messages
    """

    pytestmark = pytest.mark.local

    #: Minimal PNG (signature bytes only) as a data URI, enough to type the block.
    _IMAGE_URI = f"data:image/png;base64,{b64encode(_PNG_SIGNATURE).decode()}"

    async def _messages(self, prompt: object) -> list[Any]:
        """Validate a prompt then build its Bedrock user messages.

        Args:
            prompt: Raw ``prompt`` value from the request body.

        Returns:
            The Bedrock user messages.
        """
        request = CompletionCreateParams.model_validate(
            {"model": "model", "prompt": prompt}
        )
        return list(await build_user_messages(request.prompt))

    async def test_files_only_prompt_array_fans_out_one_message_per_file(self) -> None:
        """Two file references with no text produce one message — one choice — each.

        This is the negative case proving the collapse is not over-eager: it fires
        only when the array carries exactly one string.
        """
        messages = await self._messages([self._IMAGE_URI, self._IMAGE_URI])
        assert len(messages) == 2
        assert all(len(message["content"]) == 1 for message in messages)
        assert all("image" in message["content"][0] for message in messages)

    async def test_one_text_plus_files_collapses_into_one_message(self) -> None:
        """One string plus one file becomes a single multimodal message."""
        messages = await self._messages(["describe this", self._IMAGE_URI])
        assert len(messages) == 1
        assert [sorted(block) for block in messages[0]["content"]] == [
            ["text"],
            ["image"],
        ]

    async def test_two_texts_fan_out(self) -> None:
        """Two strings stay two messages, matching the batch-prompt contract."""
        messages = await self._messages(["one", "two"])
        assert len(messages) == 2
        assert [message["content"][0]["text"] for message in messages] == ["one", "two"]


class TestTokenArrayPrompt:
    """Token-array prompts are rejected with a 400 before any model dispatch.

    OpenAI's legacy surface accepts token and token-array prompts; this backend
    speaks Converse text blocks only.  The purpose-written "Token array prompts
    are not supported" message is unreachable — ``prompt`` is typed as
    ``InputFileUrl | str | list[InputFileUrl | str]``, so union validation fails
    before ``_validate_prompt_and_streaming`` runs — and the client sees the
    generic union errors instead.

    Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
         stdapi/types/openai_completions.py:CompletionCreateParams._validate_prompt_and_streaming
    """

    pytestmark = pytest.mark.local

    def test_token_array_prompt_is_rejected(self, app_client: TestClientType) -> None:
        """``prompt=[[15496, 11]]`` is a 400 naming ``prompt``, not a 500."""
        response = app_client.post(
            "/v1/completions", json={"model": "test-model", "prompt": [[15496, 11]]}
        )
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "prompt" in error["message"]

    def test_flat_token_prompt_is_rejected(self, app_client: TestClientType) -> None:
        """A flat token list is rejected too — integers are not valid prompt parts."""
        response = app_client.post(
            "/v1/completions", json={"model": "test-model", "prompt": [15496, 11]}
        )
        assert response.status_code == 400, response.text
        assert response.json()["error"]["type"] == "invalid_request_error"


class TestMantleTextCompletionPassthrough:
    """Bedrock Mantle chat responses converted to a legacy ``Completion``.

    Mantle serves OpenAI-shaped chat responses; the legacy route reshapes them
    into a ``Completion``.  ``system_fingerprint`` has no other producer in this
    codebase, so this passthrough is the only way it ever reaches a client.

    Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
         https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions-mantle.html
         stdapi/models/chat/_mantle/_convert.py:chat_response_as_text_completion
    """

    pytestmark = pytest.mark.local

    @staticmethod
    def _raw(**extra: object) -> dict[str, Any]:
        """Build a minimal Mantle chat completion payload.

        Args:
            **extra: Optional top-level fields to add.

        Returns:
            A Chat Completions response dict.
        """
        return {
            "id": "chatcmpl-mantle",
            "object": "chat.completion",
            "created": 1_700_000_000,
            "model": "openai.gpt-oss-20b-1:0",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            **extra,
        }

    def test_fingerprint_and_service_tier_survive_the_conversion(self) -> None:
        """``system_fingerprint`` and ``service_tier`` are copied onto the Completion.

        ``system_fingerprint`` is what upstream tells clients to watch alongside
        ``seed`` for backend-configuration changes.
        """
        completion = chat_response_as_text_completion(
            self._raw(system_fingerprint="fp_abc123", service_tier="flex"), "cmpl-1"
        )
        assert completion.system_fingerprint == "fp_abc123"
        assert completion.service_tier == "flex"
        assert completion.id == "cmpl-1", "the legacy id replaces the Mantle chat id"
        assert completion.object == "text_completion"
        assert completion.choices[0].text == "hi"

    def test_absent_fields_are_not_invented(self) -> None:
        """A payload without the two fields yields a Completion without them."""
        completion = chat_response_as_text_completion(self._raw(), "cmpl-2")
        assert completion.system_fingerprint is None
        assert completion.service_tier is None


#: Bedrock Converse response the capturing model answers with, one text block.
_CANNED_CONVERSE_RESPONSE: dict[str, Any] = {
    "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 5, "outputTokens": 1, "totalTokens": 6},
}

#: Claude model identifier: the family enabling prompt caching with extended TTLs.
_CACHING_MODEL_ID = "anthropic.claude-sonnet-4-6-20260210-v1:0"


class _CapturingChatModel(ClaudeChatModel):
    """Claude chat model recording Converse requests instead of calling Bedrock."""

    def __init__(self, model_id: str) -> None:
        """Initialize the model with an empty capture buffer.

        Args:
            model_id: Bedrock model identifier.
        """
        super().__init__(model_id)
        self.requests: list[ConverseRequestBaseTypeDef] = []

    async def converse(
        self, request: ConverseRequestBaseTypeDef
    ) -> ConverseResponseTypeDef:
        """Record *request* and answer with a canned response.

        Args:
            request: Bedrock Converse request payload.

        Returns:
            A minimal successful Converse response.
        """
        self.requests.append(request)
        return cast("ConverseResponseTypeDef", _CANNED_CONVERSE_RESPONSE)


class TestLegacyPromptCaching:
    """``prompt_cache_key`` marks every fanned-out prompt with a cache point.

    The legacy route turns a batch prompt into one model call per element and
    applies caching to each of them independently.  A cache point is what a cache
    write is billed on, so whether one is written -- and with which retention --
    is the whole feature.

    Ref: https://developers.openai.com/api/docs/guides/prompt-caching
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CachePointBlock.html
         stdapi/models/chat/_default.py:ChatModel.create_text_completion
    """

    pytestmark = pytest.mark.local

    @pytest.fixture(autouse=True)
    @staticmethod
    def _http_request() -> Generator[None]:
        """Bind a header-less HTTP request for the model's header passthrough.

        Ref: stdapi/models/chat/_default.py:ChatModel._get_passthrough_header_fields
        """
        token = REQUEST.set(
            Request({"type": "http", "method": "POST", "headers": [], "path": "/"})
        )
        try:
            yield
        finally:
            REQUEST.reset(token)

    @staticmethod
    async def _converse_requests(**kwargs: object) -> list[ConverseRequestBaseTypeDef]:
        """Run a two-prompt completion and return the captured Converse requests.

        Args:
            **kwargs: Extra request fields merged onto ``model``/``prompt``.

        Returns:
            One Bedrock Converse request per prompt of the batch.
        """
        model = _CapturingChatModel(_CACHING_MODEL_ID)
        request = CompletionCreateParams.model_validate(
            {"model": "model", "prompt": ["first prompt", "second prompt"], **kwargs}
        )
        completion = await model.create_text_completion(request, "cmpl-1", 0)
        assert isinstance(completion, Completion)
        assert len(model.requests) == 2, "one call per prompt of the batch"
        return model.requests

    @staticmethod
    def _cache_points(request: ConverseRequestBaseTypeDef) -> list[Any]:
        """Return the cache-point blocks a Converse request carries.

        Args:
            request: Captured Bedrock Converse request payload.

        Returns:
            Every ``cachePoint`` block found in the request messages.
        """
        return [
            block
            for message in request["messages"]
            for block in message["content"]
            if "cachePoint" in block
        ]

    async def test_no_cache_key_writes_no_cache_point(self) -> None:
        """Omitting ``prompt_cache_key`` leaves every prompt uncached.

        Caching is opt-in on this route, so an unmarked request must never be
        billed for a cache write.
        """
        requests = await self._converse_requests()
        assert [self._cache_points(request) for request in requests] == [[], []]

    @pytest.mark.parametrize("key", ["messages", "opaque-client-hash"])
    async def test_cache_key_marks_each_prompt_of_the_batch(self, key: str) -> None:
        """Every fanned-out prompt gets its own cache point, not just the first.

        ``messages`` selects the component explicitly and an opaque key -- what an
        upstream client sends, since OpenAI treats the value as a bucket name --
        falls through to caching every component, so both mark the messages.
        """
        requests = await self._converse_requests(prompt_cache_key=key)
        assert [self._cache_points(request) for request in requests] == [
            [PROMPT_CACHING_DEFAULT],
            [PROMPT_CACHING_DEFAULT],
        ]

    async def test_system_selector_finds_nothing_to_cache(self) -> None:
        """``prompt_cache_key="system"`` writes nothing: the route has no system prompt.

        This is the selector's negative case: an unrelated component must not fall
        back to caching the messages.
        """
        requests = await self._converse_requests(prompt_cache_key="system")
        assert [self._cache_points(request) for request in requests] == [[], []]

    @pytest.mark.parametrize(
        ("retention", "ttl"), [("24h", "1h"), ("1h", "1h"), ("5m", "5m")]
    )
    async def test_retention_sets_the_cache_point_ttl(
        self, retention: str, ttl: str
    ) -> None:
        """``prompt_cache_retention`` reaches the cache point of every prompt.

        ``24h`` is clamped to the longest retention this backend offers, so a
        client asking for a day of retention still gets a valid request rather
        than a rejected one.
        """
        requests = await self._converse_requests(
            prompt_cache_key="messages", prompt_cache_retention=retention
        )
        assert [self._cache_points(request) for request in requests] == [
            [{"cachePoint": {"type": "default", "ttl": ttl}}],
            [{"cachePoint": {"type": "default", "ttl": ttl}}],
        ]

    async def test_in_memory_retention_keeps_the_default_cache_point(self) -> None:
        """``in_memory`` asks for no explicit retention and emits no TTL."""
        requests = await self._converse_requests(
            prompt_cache_key="messages", prompt_cache_retention="in_memory"
        )
        assert [self._cache_points(request) for request in requests] == [
            [PROMPT_CACHING_DEFAULT],
            [PROMPT_CACHING_DEFAULT],
        ]


async def _stub_stream(events: list[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    """Yield the given Bedrock Converse stream event dicts one by one.

    Args:
        events: Converse stream event dicts to replay.

    Yields:
        Each event dict, in order.
    """
    for event in events:
        yield event


class TestLegacyStreamChunks:
    """Every legacy SSE chunk carries the echoed tier and a terminal finish reason.

    The streaming path formats its chunks independently of the non-streaming one,
    so the service tier echo and the stop-reason mapping have to hold on both.  A
    filtered generation is only observable here through ``finish_reason``: the
    chunk text is already gone.

    Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
         stdapi/models/chat/_adapters/_openai_completion.py:format_stream
    """

    pytestmark = pytest.mark.local

    @staticmethod
    async def _chunks(
        events: list[dict[str, Any]], service_tier: ServiceTiers | None = None
    ) -> list[dict[str, Any]]:
        """Format one stubbed Bedrock stream and decode the emitted chunks.

        Args:
            events: Converse stream event dicts to replay.
            service_tier: Effective OpenAI service tier to echo.

        Returns:
            The JSON-decoded chunks, excluding the ``[DONE]`` sentinel.
        """
        sse_events = [
            event
            async for event in format_stream(
                "cmpl-1",
                0,
                "model",
                [_stub_stream(events)],  # type: ignore[list-item]
                service_tier,
                include_usage=False,
            )
        ]
        assert sse_events[-1].data == "[DONE]", "the stream must end with the sentinel"
        return [
            json.loads(event.data)
            for event in sse_events
            if isinstance(event.data, str) and event.data != "[DONE]"
        ]

    async def test_every_chunk_echoes_the_effective_service_tier(self) -> None:
        """The tier is repeated on the delta chunks and on the terminal one.

        A client reading the tier off the last chunk and one reading it off the
        first must agree, so it cannot be attached to the terminal chunk only.
        """
        chunks = await self._chunks(
            [
                {"contentBlockDelta": {"delta": {"text": "hi"}}},
                {"messageStop": {"stopReason": "end_turn"}},
            ],
            service_tier="flex",
        )
        assert len(chunks) == 2
        assert {chunk["service_tier"] for chunk in chunks} == {"flex"}

    async def test_absent_service_tier_is_omitted_from_the_chunks(self) -> None:
        """No requested tier means no ``service_tier`` key rather than a null one."""
        chunks = await self._chunks([{"contentBlockDelta": {"delta": {"text": "hi"}}}])
        assert all("service_tier" not in chunk for chunk in chunks)

    @pytest.mark.parametrize(
        ("stop_reason", "expected"),
        [
            ("guardrail_intervened", "content_filter"),
            ("content_filtered", "content_filter"),
            ("model_context_window_exceeded", "length"),
            ("end_turn", "stop"),
        ],
    )
    async def test_terminal_chunk_reports_the_mapped_finish_reason(
        self, stop_reason: str, expected: str
    ) -> None:
        """A filtered or truncated stream says so instead of claiming completion.

        The delta chunks carry no ``finish_reason``; only the terminal chunk does,
        and it is the client's sole signal that the text it concatenated is not
        the whole answer.
        """
        chunks = await self._chunks(
            [
                {"contentBlockDelta": {"delta": {"text": "hi"}}},
                {"messageStop": {"stopReason": stop_reason}},
            ]
        )
        assert "finish_reason" not in chunks[0]["choices"][0]
        assert chunks[-1]["choices"][0]["finish_reason"] == expected
        assert chunks[-1]["choices"][0]["text"] == "", "the terminal chunk adds no text"

    async def test_stream_without_message_stop_still_terminates(self) -> None:
        """A stream ending without ``messageStop`` still gets a terminal chunk.

        The chunk falls back to ``stop`` so the client is never left waiting for a
        finish reason that will not come.
        """
        chunks = await self._chunks([{"contentBlockDelta": {"delta": {"text": "hi"}}}])
        assert [chunk["choices"][0].get("finish_reason") for chunk in chunks] == [
            None,
            "stop",
        ]
