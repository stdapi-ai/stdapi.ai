"""Tests for the OpenAI /v1/completions route (Completions API).

Comprehensive test suite that validates the ``/v1/completions`` API specification,
ensuring compatibility with OpenAI SDK clients and the stdapi-specific extensions.
"""

import io
from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError, NotFoundError, OpenAI

if TYPE_CHECKING:
    from starlette.testclient import TestClient as TestClientType


class TestCompletions:
    """Test suite for the ``/v1/completions`` endpoint.

    Covers:
    - Single and batch prompt handling
    - Multiple choice generation
    - Streaming responses
    - Parameter validation and error handling
    - File-ID URI scheme support
    """

    def test_basic_single_prompt_returns_text(
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """Test basic single prompt completion returns non-empty text.

        Validates the core completion functionality using a simple prompt
        and verifies the response contains generated text with proper finish reason.

        Args:
            openai_client: OpenAI client instance for API calls
            completion_model: Model identifier for completion
        """
        response = openai_client.completions.create(
            model=completion_model, prompt="The capital of France is", max_tokens=20
        )

        assert hasattr(response, "choices")
        assert len(response.choices) == 1
        assert response.choices[0].text is not None
        assert len(response.choices[0].text) > 0
        assert response.choices[0].finish_reason in {"stop", "length"}
        assert response.object == "text_completion"

        assert hasattr(response, "usage")
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

    def test_list_prompt_returns_one_choice_per_prompt(
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """Test list of prompts returns one choice per prompt.

        Validates that passing multiple prompts as a list generates
        a separate completion choice for each prompt.

        Args:
            openai_client: OpenAI client instance for API calls
            completion_model: Model identifier for completion
        """
        response = openai_client.completions.create(
            model=completion_model,
            prompt=["One plus one is", "Two plus two is"],
            max_tokens=5,
        )

        assert hasattr(response, "choices")
        assert len(response.choices) == 2
        assert response.choices[0].index == 0
        assert response.choices[1].index == 1
        assert response.choices[0].text is not None
        assert response.choices[1].text is not None

    def test_n_gt_1_returns_multiple_choices(
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """Test n parameter returns multiple choices for single prompt.

        Validates that setting n>1 generates multiple completion choices
        for a single prompt.

        Args:
            openai_client: OpenAI client instance for API calls
            completion_model: Model identifier for completion
        """
        response = openai_client.completions.create(
            model=completion_model, prompt="Hello", n=2, max_tokens=3
        )

        assert hasattr(response, "choices")
        assert len(response.choices) == 2
        assert response.choices[0].index == 0
        assert response.choices[1].index == 1
        assert response.choices[0].text is not None
        assert response.choices[1].text is not None

    def test_streaming_yields_text_deltas_and_terminal_finish_reason(
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """Test streaming yields text deltas and one chunk with finish_reason.

        Validates that streaming mode produces incremental text chunks,
        each with proper object type, and exactly one chunk has finish_reason set.

        Args:
            openai_client: OpenAI client instance for API calls
            completion_model: Model identifier for completion
        """
        response = openai_client.completions.create(
            model=completion_model,
            prompt="Tell me a short story",
            max_tokens=30,
            stream=True,
        )

        chunks: list[object] = []
        text_fragments: list[str] = []
        finish_count = 0

        for chunk in response:
            if hasattr(chunk, "choices"):
                chunks.append(chunk)
                assert chunk.object == "text_completion"
                if chunk.choices and chunk.choices[0].text:
                    text_fragments.append(chunk.choices[0].text)
                if chunk.choices and chunk.choices[0].finish_reason is not None:
                    finish_count += 1

        assert len(chunks) > 0
        assert "".join(text_fragments) != ""
        assert finish_count == 1

    def test_streaming_include_usage_final_chunk_has_usage(
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """Test streaming with include_usage returns usage in final chunk.

        Validates that when stream_options.include_usage is True,
        the final chunk contains populated usage information.

        Args:
            openai_client: OpenAI client instance for API calls
            completion_model: Model identifier for completion
        """
        response = openai_client.completions.create(
            model=completion_model,
            prompt="Hello world",
            max_tokens=20,
            stream=True,
            stream_options={"include_usage": True},
        )

        chunk_list = list(response)

        # Last chunk should have usage
        assert len(chunk_list) > 0
        last_chunk = chunk_list[-1]
        assert hasattr(last_chunk, "usage")
        assert last_chunk.usage is not None
        assert last_chunk.usage.prompt_tokens > 0
        assert last_chunk.usage.completion_tokens > 0
        assert last_chunk.usage.total_tokens > 0

    def test_streaming_multi_prompt_interleaves_choices(
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """Streaming with a list prompt produces chunks for each prompt index.

        Each SSE chunk carries a single choice whose ``index`` identifies the
        originating prompt.  Every prompt must receive at least one delta
        chunk and exactly one terminal chunk with ``finish_reason`` set.
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
            assert finish_per_index[index] in {"stop", "length"}

    def test_streaming_n_gt_1_yields_one_terminal_chunk_per_choice(
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """Streaming with ``n > 1`` returns deltas for each choice index.

        Each choice gets a distinct ``choices[0].index`` (``0..n-1``) and
        exactly one terminal chunk with ``finish_reason`` per index.
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
            assert finish_per_index[index] in {"stop", "length"}

    def test_stop_sequences_truncate_generation(
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """Stop sequences terminate generation and are excluded from the text.

        Uses a prompt that naturally produces the stop sequence within a
        generous token budget, so the model stops on the sequence rather
        than on ``max_tokens``.
        """
        response = openai_client.completions.create(
            model=completion_model,
            prompt="Write one complete sentence about cats",
            stop=["."],
            max_tokens=500,
        )

        assert len(response.choices) > 0
        assert response.choices[0].finish_reason == "stop"

    def test_max_tokens_limit_yields_length_finish_reason(
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """Test max_tokens=1 yields length finish_reason.

        Validates that when max_tokens is very limited, the response
        has finish_reason "length".

        Args:
            openai_client: OpenAI client instance for API calls
            completion_model: Model identifier for completion
        """
        response = openai_client.completions.create(
            model=completion_model, prompt="Tell me a long story", max_tokens=1
        )

        assert len(response.choices) > 0
        assert response.choices[0].finish_reason == "length"

    def test_unknown_model_returns_404(self, openai_client: OpenAI) -> None:
        """Test unknown model returns NotFoundError with model_not_found code.

        Validates that requesting a non-existent model returns proper
        error handling.

        Args:
            openai_client: OpenAI client instance for API calls
        """
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.completions.create(model="does-not-exist-xyz", prompt="Hello")
        assert exc_info.value.code == "model_not_found"

    def test_unsupported_params_are_silently_accepted(
        self, openai_client: OpenAI, completion_model: str
    ) -> None:
        """Test unsupported params are silently accepted without error.

        Validates that certain unsupported parameters are accepted
        (per the adapter's "accept silently" policy) and the request succeeds.

        Args:
            openai_client: OpenAI client instance for API calls
            completion_model: Model identifier for completion
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

        # Request should succeed
        assert hasattr(response, "choices")
        assert len(response.choices) > 0

    def test_input_file_url_prompt_via_file_id(
        self, openai_client: OpenAI, completion_model: str, use_official_api: bool
    ) -> None:
        """Reference an uploaded file via the ``file-id:`` URI scheme.

        Uploads a small text file via the Files API, then references it using
        ``file-id:<file-id>`` as the prompt.  stdapi forwards the file to the
        model as a ``document`` block (its detected modality).  The default
        completion model does not support document inputs, so the upstream
        model rejects the request with ``ValidationException`` — the
        assertion here verifies that the file-id resolver and the adapter
        correctly pass the document through to the model.
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
            assert "document" in str(exc_info.value).lower()
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
        image" workflow.
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
        assert response.choices[0].finish_reason in {"stop", "length"}

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
        """
        if use_official_api:
            pytest.skip(
                "multimodal input on v1/completions is a stdapi extension; "
                "the official legacy Completions API rejects chat/vision models"
            )
        response = openai_client.completions.create(
            model=chat_vision_model, prompt=sample_image_file_base64, max_tokens=80
        )
        assert len(response.choices) == 1
        assert response.choices[0].index == 0
        assert response.choices[0].text
        assert response.choices[0].finish_reason in {"stop", "length"}


class TestStopSequenceValidation:
    """Offline unit tests: whitespace-only stop sequences are rejected before dispatch.

    AWS Bedrock rejects a blank ``stopSequences`` entry with a raw
    ``ValidationException``; this backend surfaces a clean 400 instead.
    Validation happens before any model dispatch or AWS call, so the
    rejection test runs against an app instance without the AWS-touching
    lifespan.
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
        r"""stop=["\n"] is rejected with a clean 400 (Bedrock cannot honor blank stops)."""
        response = client.post(
            "/v1/completions",
            json={"model": "test-model", "prompt": "hi", "stop": ["\n"]},
        )
        assert response.status_code == 400, response.text
        error_body = response.json()
        assert error_body["error"]["type"] == "invalid_request_error"
        assert "whitespace" in error_body["error"]["message"].lower()
