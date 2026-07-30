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
from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError, NotFoundError, OpenAI

if TYPE_CHECKING:
    from starlette.testclient import TestClient as TestClientType

#: Every ``finish_reason`` the legacy adapter can emit for a successful request.
_TERMINAL_REASONS = {"stop", "length"}


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
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """A stop sequence halts generation at its first occurrence, which stays in the text.

        The prompt asks for several sentences within a generous token budget, so a
        completion holding a single ``"."`` proves the model stopped on the sequence
        rather than on ``max_tokens``.  Contrary to the OpenAI contract ("the returned
        text will not contain the stop sequence"), Bedrock keeps the matched sequence in
        the text and reports ``stopReason="end_turn"``, mapped to ``finish_reason``
        ``"stop"``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
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
        assert text.count(".") == 1, (
            f"generation must halt at the first stop sequence: {text!r}"
        )
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

        ``best_of``, ``logprobs``, ``suffix``, the penalties and ``logit_bias``
        have no Bedrock Converse equivalent.  The gateway declares them
        UNSUPPORTED but keeps them valid so unmodified OpenAI clients work, and
        silently ignores them — ``logprobs`` in particular never produces a
        ``logprobs`` object on a choice.

        Ref: stdapi/types/openai_completions.py:CompletionCreateParams
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

    def test_single_text_plus_files_returns_single_choice(
        self,
        openai_client: OpenAI,
        chat_vision_model: str,
        sample_image_file_base64: str,
        use_official_api: bool,
    ) -> None:
        """One text + one image data URI collapse into a single multimodal request.

        Sends a list prompt containing one question and one ``data:image/png;base64,...``
        image URI to a vision-capable model (Claude).  stdapi builds a single
        multimodal Bedrock message (one ``text`` block + one ``image`` block)
        and returns exactly one completion choice — the natural "analyse this
        image" workflow.  A per-element fan-out would have produced two choices.

        Ref: stdapi/models/chat/_adapters/_openai_completion.py:build_user_messages
        """
        if use_official_api:
            pytest.skip(
                "multimodal input on v1/completions is a stdapi extension; "
                "the official legacy Completions API rejects chat/vision models"
            )
        response = openai_client.completions.create(
            model=chat_vision_model,
            prompt=[
                "Describe what is shown in this image in one short sentence:",
                sample_image_file_base64,
            ],
            max_tokens=80,
        )
        assert len(response.choices) == 1
        assert response.choices[0].index == 0
        assert response.choices[0].text
        assert response.choices[0].finish_reason in _TERMINAL_REASONS
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0

    def test_image_only_prompt_returns_single_choice(
        self,
        openai_client: OpenAI,
        chat_vision_model: str,
        sample_image_file_base64: str,
        use_official_api: bool,
    ) -> None:
        """A lone image data URI is sent as a single multimodal request.

        With no accompanying text instruction, the image is forwarded to the
        model as an ``image`` block; the model decides how to respond.

        Ref: stdapi/models/chat/_adapters/_openai_completion.py:build_user_messages
        """
        if use_official_api:
            pytest.skip(
                "multimodal input on v1/completions is a stdapi extension; "
                "the official legacy Completions API rejects chat/vision models"
            )
        response = openai_client.completions.create(
            model=chat_vision_model, prompt=sample_image_file_base64, max_tokens=80
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

    @pytest.fixture
    def client(self, api_key: str) -> TestClientType:
        """Test client without lifespan (no AWS startup), pre-authenticated."""
        from starlette.testclient import TestClient  # noqa: PLC0415

        from stdapi.main import app  # noqa: PLC0415

        return TestClient(app, headers={"Authorization": f"Bearer {api_key}"})

    def test_whitespace_only_stop_sequence_is_rejected(
        self, client: TestClientType
    ) -> None:
        r"""stop=["\n"] is rejected with a clean 400 (Bedrock cannot honor blank stops).

        Upstream OpenAI accepts a newline stop sequence, so this 400 is
        gateway-specific.  The envelope always carries all four OpenAI error
        keys, ``param``/``code`` included and null here because the rejection
        comes from request validation rather than from a typed ``ApiError``.

        Ref: stdapi/api_providers/openai.py:_format_error
             stdapi/main.py:handle_validation_exception
        """
        response = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "hi", "stop": ["\n"]},
        )
        assert response.status_code == 400, response.text
        error_body = response.json()
        assert error_body.keys() == {"error"}
        assert error_body["error"].keys() == {"message", "type", "param", "code"}
        assert error_body["error"]["type"] == "invalid_request_error"
        assert "whitespace" in error_body["error"]["message"].lower()
