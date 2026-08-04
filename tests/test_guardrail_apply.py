"""Guardrail enforcement via ApplyGuardrail on routes without native support (unit).

When a server guardrail is configured, routes whose AWS backend has no native
guardrail mechanism (embeddings, rerank, images, videos, audio) apply it via the
Bedrock ApplyGuardrail API: client-supplied text is checked as ``INPUT`` before
the backend call and generated text as ``OUTPUT`` after it.

Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-independent-api.html
     stdapi/aws_bedrock.py:apply_guardrail_to_text
     stdapi/aws_bedrock.py:apply_guardrail_to_texts
"""

from typing import TYPE_CHECKING, Any

import pytest

from stdapi import aws_bedrock, models
from stdapi.aws_bedrock import (
    GUARDRAIL_CONFIG_VAR,
    GuardrailInterventionError,
    apply_guardrail_to_text,
    apply_guardrail_to_texts,
)
from stdapi.config import SETTINGS
from stdapi.input_file import InputFile
from stdapi.models.audio.amazon_transcribe import AudioModel
from stdapi.models.embedding import EmbeddingResponse
from stdapi.models.embedding.amazon_titan_embed import EmbeddingModel as TitanEmbedModel
from stdapi.models.rerank import RerankedDocument, RerankResponse
from stdapi.routes import (
    cohere_embed,
    cohere_rerank,
    openai_audio_speech,
    openai_embeddings,
    openai_images_generations,
    openai_videos,
)
from stdapi.routes.openai_audio_transcriptions import _guarded_transcript_events
from stdapi.types.openai_audio import (
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
)
from tests._helpers import make_model_details
from tests.test_openai_audio_transcriptions import (
    _STUB_TRANSCRIPT_DATA,
    _stub_transcribe,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator

    from starlette.testclient import TestClient

#: Per-policy units ApplyGuardrail reports, deliberately not ceil(len(text)/1000).
_USAGE: dict[str, int] = {
    "topicPolicyUnits": 5,
    "contentPolicyUnits": 5,
    "wordPolicyUnits": 5,
    "sensitiveInformationPolicyUnits": 5,
    "sensitiveInformationPolicyFreeUnits": 0,
    "contextualGroundingPolicyUnits": 0,
    "contentPolicyImageUnits": 0,
    "automatedReasoningPolicyUnits": 0,
    "automatedReasoningPolicies": 0,
}

#: A guardrail response without any policy hit.
_CLEAN_RESPONSE: dict[str, Any] = {"action": "NONE", "assessments": [], "usage": _USAGE}

#: Messaging configured on the guardrail, returned by a blocking intervention.
_BLOCKED_MESSAGING = "Blocked by the acme-guardrail policy."

#: A guardrail response blocking the content (content filter hit).
_BLOCKED_RESPONSE: dict[str, Any] = {
    "action": "GUARDRAIL_INTERVENED",
    "assessments": [
        {
            "contentPolicy": {
                "filters": [
                    {"type": "VIOLENCE", "confidence": "HIGH", "action": "BLOCKED"}
                ]
            }
        }
    ],
    "outputs": [{"text": _BLOCKED_MESSAGING}],
    "usage": _USAGE,
}

#: The masked text returned by a masking-only intervention.
_MASKED_TEXT = "Contact {EMAIL} for details"

#: A guardrail response that only anonymized sensitive information.
_MASKED_RESPONSE: dict[str, Any] = {
    "action": "GUARDRAIL_INTERVENED",
    "assessments": [
        {
            "sensitiveInformationPolicy": {
                "piiEntities": [{"type": "EMAIL", "action": "ANONYMIZED"}]
            }
        }
    ],
    "outputs": [{"text": _MASKED_TEXT}],
}


class _StubGuardrailClient:
    """Stub bedrock-runtime client recording apply_guardrail calls."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.requests: list[dict[str, Any]] = []
        self._response = response

    async def apply_guardrail(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        """Record the request and return the canned assessment."""
        self.requests.append(params)
        return self._response


def _stub_guardrail(
    monkeypatch: pytest.MonkeyPatch, response: dict[str, Any]
) -> _StubGuardrailClient:
    """Stub the bedrock-runtime client used by the ApplyGuardrail helper."""
    stub = _StubGuardrailClient(response)
    monkeypatch.setattr(aws_bedrock, "get_client", lambda _service, _region: stub)
    return stub


def _forbid_guardrail_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any guardrail client acquisition fail the test."""

    def _get_client(_service: str, _region: str) -> Any:  # noqa: ANN401
        pytest.fail("ApplyGuardrail client requested without a configured guardrail")

    monkeypatch.setattr(aws_bedrock, "get_client", _get_client)


def _record_usage_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace the guardrail usage recorder, capturing its calls."""
    calls: list[dict[str, Any]] = []

    def _record(usage: dict[str, Any], *, region: str = "") -> None:
        calls.append({"usage": dict(usage), "region": region})

    monkeypatch.setattr(aws_bedrock, "record_guardrail_policy_usage", _record)
    return calls


@pytest.fixture
def guardrail_context() -> Iterator[None]:
    """Set the request guardrail context var directly (no HTTP request)."""
    token = GUARDRAIL_CONFIG_VAR.set(
        {"guardrailIdentifier": "gr123", "guardrailVersion": "1"}
    )
    yield
    GUARDRAIL_CONFIG_VAR.reset(token)


@pytest.mark.local
class TestApplyGuardrailHelper:
    """ApplyGuardrail helper behavior: no-op, blocking, masking and metering.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
         stdapi/aws_bedrock.py:apply_guardrail_to_text
    """

    async def test_unconfigured_is_zero_cost(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a configured guardrail no AWS client is even acquired."""
        _forbid_guardrail_calls(monkeypatch)

        assert await apply_guardrail_to_text("hello", source="INPUT") == "hello"
        items: list[Any] = ["hello", {"title": "doc"}]
        assert await apply_guardrail_to_texts(items, source="INPUT") == items

    @pytest.mark.usefixtures("guardrail_context")
    async def test_clean_text_passes_and_records_usage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A clean result returns the text unchanged and meters what AWS reports.

        The response reports 5 units for a 1,500-character input, which a
        ceil(len(text)/1000) estimate would have metered as 2: the units
        billed are the ones the API returns, per policy, never a guess.

        Ref: stdapi/usage.py:record_guardrail_policy_usage
        """
        stub = _stub_guardrail(monkeypatch, _CLEAN_RESPONSE)
        usage = _record_usage_calls(monkeypatch)
        text = "a" * 1_500

        assert await apply_guardrail_to_text(text, source="INPUT") == text

        (request,) = stub.requests
        assert request["guardrailIdentifier"] == "gr123"
        assert request["guardrailVersion"] == "1"
        assert request["source"] == "INPUT"
        assert request["content"] == [{"text": {"text": text}}]
        (entry,) = usage
        assert entry["usage"] == _USAGE
        assert entry["region"] == SETTINGS.aws_bedrock_regions[0]

    @pytest.mark.usefixtures("guardrail_context")
    async def test_empty_text_is_not_sent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty text short-circuits: ApplyGuardrail rejects empty content."""
        _forbid_guardrail_calls(monkeypatch)

        assert await apply_guardrail_to_text("", source="INPUT") == ""

    @pytest.mark.usefixtures("guardrail_context")
    async def test_blocked_raises_with_guardrail_messaging(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A blocking intervention fails with 400 and the guardrail messaging.

        The error surfaces the same ``content_filter`` code that the chat
        routes map ``guardrail_intervened`` to, and carries the guardrail's
        configured blocked messaging (what a Converse client would receive).

        Ref: stdapi/models/chat/_adapters/_openai_chat_completion.py
        """
        _stub_guardrail(monkeypatch, _BLOCKED_RESPONSE)
        _record_usage_calls(monkeypatch)

        with pytest.raises(GuardrailInterventionError) as exc_info:
            await apply_guardrail_to_text("bad text", source="INPUT")

        assert exc_info.value.status == 400
        assert exc_info.value.code == "content_filter"
        assert str(exc_info.value) == _BLOCKED_MESSAGING

    @pytest.mark.usefixtures("guardrail_context")
    async def test_blocked_without_messaging_uses_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A blocking intervention without outputs falls back to a fixed message."""
        response = {**_BLOCKED_RESPONSE, "outputs": []}
        _stub_guardrail(monkeypatch, response)
        _record_usage_calls(monkeypatch)

        with pytest.raises(GuardrailInterventionError, match="blocked"):
            await apply_guardrail_to_text("bad text", source="INPUT")

    @pytest.mark.usefixtures("guardrail_context")
    async def test_masking_only_returns_masked_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A masking-only intervention substitutes the anonymized output text."""
        _stub_guardrail(monkeypatch, _MASKED_RESPONSE)
        _record_usage_calls(monkeypatch)

        result = await apply_guardrail_to_text(
            "Contact me@example.com for details", source="OUTPUT"
        )

        assert result == _MASKED_TEXT

    @pytest.mark.usefixtures("guardrail_context")
    async def test_masking_when_not_maskable_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Masking fails the request when the caller cannot carry masked text."""
        _stub_guardrail(monkeypatch, _MASKED_RESPONSE)
        _record_usage_calls(monkeypatch)

        with pytest.raises(GuardrailInterventionError, match="masked"):
            await apply_guardrail_to_text(
                "Contact me@example.com", source="OUTPUT", maskable=False
            )

    @pytest.mark.usefixtures("guardrail_context")
    async def test_texts_guards_strings_and_passes_mappings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only string items are guarded; structured items pass through as-is."""
        stub = _stub_guardrail(monkeypatch, _CLEAN_RESPONSE)
        _record_usage_calls(monkeypatch)
        items: list[Any] = ["first", {"title": "doc"}, "second"]

        assert await apply_guardrail_to_texts(items, source="INPUT") == items

        assert [request["content"][0]["text"]["text"] for request in stub.requests] == [
            "first",
            "second",
        ]


class _StubEmbeddingBackend:
    """Stub embedding backend recording the inputs it receives."""

    def __init__(self) -> None:
        self.calls: list[list[Any]] = []

    async def embed_text(
        self, inputs: list[Any], dimensions: int | None, extra_params: dict[str, Any]
    ) -> EmbeddingResponse:
        """Record the inputs and return one vector per input."""
        del dimensions, extra_params
        self.calls.append(list(inputs))
        return EmbeddingResponse(
            embeddings=[[0.1, 0.2]] * len(inputs), prompt_tokens=3, total_tokens=3
        )


async def _validate_any_model(model_id: str, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
    """Resolve any model ID to canned details, bypassing the Bedrock catalog."""
    del args, kwargs
    return make_model_details(model_id)


@pytest.mark.local
class TestNativeGuardrailSupport:
    """InvokeModel guardrail kwargs are suppressed where AWS rejects them.

    Live-verified: InvokeModel returns "Guardrail is not supported with the
    chosen model" for embedding (amazon.titan-embed-text-v2:0) and image
    generation (amazon.nova-canvas-v1:0) models, so the configured guardrail
    must not ride their InvokeModel calls — the ApplyGuardrail route wrapping
    covers them instead. Text generation models keep the native kwargs.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-supported.html
         stdapi/models/__init__.py:ModelBase.invoke
    """

    @pytest.mark.parametrize(
        ("model", "expected"),
        [
            (models.ModelBase("mistral.pixtral-large-2502-v1:0"), "gr123"),
            (TitanEmbedModel("amazon.titan-embed-text-v2:0"), None),
        ],
        ids=["text-generation-native", "embedding-suppressed"],
    )
    async def test_context_var_fallback_honors_the_class_flag(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model: models.ModelBase[Any, Any],
        expected: str | None,
    ) -> None:
        """The configured guardrail reaches InvokeModel only on supporting models."""
        captured: dict[str, Any] = {}

        async def _fake_candidates(model_id: str, **kwargs: Any) -> list[str]:  # noqa: ANN401
            del model_id, kwargs
            return ["us-east-1"]

        async def _fake_route_and_execute(
            model_id: str,
            candidates: Any,  # noqa: ANN401
            fn: Any,  # noqa: ANN401
        ) -> Any:  # noqa: ANN401
            del model_id, candidates
            captured["guardrail"] = fn.keywords["guardrail"]
            msg = "short-circuit"
            raise RuntimeError(msg)

        monkeypatch.setattr(models, "compute_candidate_regions", _fake_candidates)
        monkeypatch.setattr(models, "route_and_execute", _fake_route_and_execute)
        token = GUARDRAIL_CONFIG_VAR.set(
            {"guardrailIdentifier": "gr123", "guardrailVersion": "1"}
        )
        try:
            with pytest.raises(RuntimeError, match="short-circuit"):
                await model.invoke({})
        finally:
            GUARDRAIL_CONFIG_VAR.reset(token)
        guardrail = captured["guardrail"]
        assert (guardrail["guardrailIdentifier"] if guardrail else None) == expected


@pytest.fixture
def embedding_backend(monkeypatch: pytest.MonkeyPatch) -> _StubEmbeddingBackend:
    """Stub model validation and the embedding backend on both embed routes."""
    stub = _StubEmbeddingBackend()
    for module in (openai_embeddings, cohere_embed):
        monkeypatch.setattr(module, "validate_model", _validate_any_model)
        monkeypatch.setattr(module, "get_embedding_model", lambda _model_id: stub)
    return stub


@pytest.mark.local
@pytest.mark.usefixtures("configured_guardrail")
class TestEmbeddingsGuardrail:
    """POST /v1/embeddings and /cohere/v2/embed: INPUT guardrail enforcement.

    Ref: https://developers.openai.com/api/reference/resources/embeddings
         https://docs.cohere.com/reference/embed
         stdapi/routes/openai_embeddings.py:create_embeddings
         stdapi/routes/cohere_embed.py:embed
    """

    def test_masked_input_reaches_the_backend(
        self,
        app_client: TestClient,
        embedding_backend: _StubEmbeddingBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A masking-only intervention embeds the anonymized text (Converse parity)."""
        stub = _stub_guardrail(monkeypatch, _MASKED_RESPONSE)

        response = app_client.post(
            "/v1/embeddings",
            json={"model": "amazon.titan-embed-text-v2:0", "input": "me@example.com"},
        )

        assert response.status_code == 200, response.text
        assert embedding_backend.calls == [[_MASKED_TEXT]]
        assert stub.requests[0]["source"] == "INPUT"

    def test_blocked_input_fails_as_content_filter(
        self,
        app_client: TestClient,
        embedding_backend: _StubEmbeddingBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A blocking intervention fails with 400/content_filter before the backend."""
        _stub_guardrail(monkeypatch, _BLOCKED_RESPONSE)

        response = app_client.post(
            "/v1/embeddings",
            json={"model": "amazon.titan-embed-text-v2:0", "input": ["bad text"]},
        )

        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["code"] == "content_filter"
        assert error["message"] == _BLOCKED_MESSAGING
        assert not embedding_backend.calls

    def test_cohere_embed_texts_are_guarded(
        self,
        app_client: TestClient,
        embedding_backend: _StubEmbeddingBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The Cohere v2 embed route guards its ``texts`` the same way.

        Cohere routes report errors in the Cohere envelope (``message``/``id``)
        rather than the OpenAI ``error`` object.
        """
        _stub_guardrail(monkeypatch, _BLOCKED_RESPONSE)

        response = app_client.post(
            "/cohere/v2/embed",
            json={
                "model": "cohere.embed-multilingual-v3",
                "input_type": "search_document",
                "texts": ["bad text"],
            },
        )

        assert response.status_code == 400, response.text
        assert response.json()["message"] == _BLOCKED_MESSAGING
        assert not embedding_backend.calls


@pytest.mark.local
class TestEmbeddingsWithoutGuardrail:
    """No configured guardrail: routes must not make any ApplyGuardrail call.

    Ref: stdapi/aws_bedrock.py:apply_guardrail_to_texts
    """

    def test_no_guardrail_call_without_configuration(
        self,
        app_client: TestClient,
        embedding_backend: _StubEmbeddingBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Embeddings pass the input through untouched with zero guardrail calls."""
        _forbid_guardrail_calls(monkeypatch)

        response = app_client.post(
            "/v1/embeddings",
            json={"model": "amazon.titan-embed-text-v2:0", "input": "hello"},
        )

        assert response.status_code == 200, response.text
        assert embedding_backend.calls == [["hello"]]


class _StubRerankBackend:
    """Stub rerank backend recording the query and documents it receives."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def rerank(
        self,
        query: str,
        documents: list[Any],
        *,
        top_n: int | None,
        extra_params: dict[str, Any],
    ) -> RerankResponse:
        """Record the call and rank the first document only."""
        del top_n, extra_params
        self.calls.append({"query": query, "documents": documents})
        return RerankResponse(
            results=[RerankedDocument(index=0, relevance_score=0.9)], search_units=1
        )


@pytest.fixture
def rerank_backend(monkeypatch: pytest.MonkeyPatch) -> _StubRerankBackend:
    """Stub model validation and the rerank backend on the v2 rerank route."""
    stub = _StubRerankBackend()
    monkeypatch.setattr(cohere_rerank, "validate_model", _validate_any_model)
    monkeypatch.setattr(cohere_rerank, "get_rerank_model", lambda _model_id: stub)
    return stub


@pytest.mark.local
@pytest.mark.usefixtures("configured_guardrail")
class TestRerankGuardrail:
    """POST /cohere/v2/rerank: INPUT guardrail on the query and documents.

    Ref: https://docs.cohere.com/reference/rerank
         stdapi/routes/cohere_rerank.py:rerank
    """

    def test_masked_query_and_documents_reach_the_backend(
        self,
        app_client: TestClient,
        rerank_backend: _StubRerankBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Masked query and document texts are forwarded to the Bedrock call."""
        stub = _stub_guardrail(monkeypatch, _MASKED_RESPONSE)

        response = app_client.post(
            "/cohere/v2/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "me@example.com",
                "documents": ["me@example.com too"],
            },
        )

        assert response.status_code == 200, response.text
        (call,) = rerank_backend.calls
        assert call["query"] == _MASKED_TEXT
        assert call["documents"] == [_MASKED_TEXT]
        assert {request["source"] for request in stub.requests} == {"INPUT"}

    def test_blocked_document_fails_as_content_filter(
        self,
        app_client: TestClient,
        rerank_backend: _StubRerankBackend,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A blocking intervention on any document fails the whole request.

        Cohere routes report errors in the Cohere envelope (``message``/``id``)
        rather than the OpenAI ``error`` object.
        """
        _stub_guardrail(monkeypatch, _BLOCKED_RESPONSE)

        response = app_client.post(
            "/cohere/v2/rerank",
            json={
                "model": "cohere.rerank-v3-5:0",
                "query": "capital of France",
                "documents": ["bad text"],
            },
        )

        assert response.status_code == 400, response.text
        assert response.json()["message"] == _BLOCKED_MESSAGING
        assert not rerank_backend.calls


class _StubImageJob:
    """Stub image job satisfying ``build_images_response``."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    output_format = "png"
    width = 1024
    height = 1024
    quality = "medium"

    async def generate_images(self) -> list[Any]:
        """Return a single canned base64 image result."""
        from stdapi.models.image import ImageGenerationResponse  # noqa: PLC0415

        return [ImageGenerationResponse(image="aW1n", index=0)]


class _StubImageModel:
    """Stub image model recording the prompts of created generation jobs."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def get_image_generation_job(self, *, prompt: str, **kwargs: Any) -> _StubImageJob:  # noqa: ANN401
        """Record the prompt and return a canned job."""
        del kwargs
        self.prompts.append(prompt)
        return _StubImageJob()


@pytest.fixture
def image_backend(monkeypatch: pytest.MonkeyPatch) -> _StubImageModel:
    """Stub model validation and the image backend on the generations route."""
    stub = _StubImageModel()
    monkeypatch.setattr(
        openai_images_generations, "validate_model", _validate_any_model
    )
    monkeypatch.setattr(
        openai_images_generations, "get_image_model", lambda _model_id: stub
    )
    return stub


@pytest.mark.local
@pytest.mark.usefixtures("configured_guardrail")
class TestImagesGuardrail:
    """POST /v1/images/generations: INPUT guardrail on the prompt.

    Ref: https://developers.openai.com/api/reference/resources/images
         stdapi/routes/openai_images_generations.py:create_images
    """

    def test_masked_prompt_reaches_the_model(
        self,
        app_client: TestClient,
        image_backend: _StubImageModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A masking-only intervention generates from the anonymized prompt."""
        _stub_guardrail(monkeypatch, _MASKED_RESPONSE)

        response = app_client.post(
            "/v1/images/generations",
            json={
                "model": "amazon.nova-canvas-v1:0",
                "prompt": "a portrait of me@example.com",
                "size": "1024x1024",
                "response_format": "b64_json",
            },
        )

        assert response.status_code == 200, response.text
        assert image_backend.prompts == [_MASKED_TEXT]

    def test_blocked_prompt_fails_as_content_filter(
        self,
        app_client: TestClient,
        image_backend: _StubImageModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A blocking intervention fails with 400/content_filter before the model."""
        _stub_guardrail(monkeypatch, _BLOCKED_RESPONSE)

        response = app_client.post(
            "/v1/images/generations",
            json={
                "model": "amazon.nova-canvas-v1:0",
                "prompt": "bad prompt",
                "size": "1024x1024",
            },
        )

        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "content_filter"
        assert not image_backend.prompts


class _StubVideoModel:
    """Stub video model recording the prompts of started generation jobs."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def start_video_generation(self, prompt: str, **kwargs: Any) -> Any:  # noqa: ANN401
        """Record the prompt and return a canned generation start."""
        from stdapi.models.video import VideoGenerationStart  # noqa: PLC0415

        del kwargs
        self.prompts.append(prompt)
        return VideoGenerationStart(
            invocation_arn="arn:aws:bedrock:us-east-1:123456789012:async-invoke/x",
            seconds=6,
            size="1280x720",
        )


@pytest.fixture
def video_backend(monkeypatch: pytest.MonkeyPatch) -> _StubVideoModel:
    """Stub model validation and the video backend on the videos route."""
    stub = _StubVideoModel()
    monkeypatch.setattr(openai_videos, "validate_model", _validate_any_model)
    monkeypatch.setattr(openai_videos, "get_video_model", lambda _model_id: stub)
    return stub


@pytest.mark.local
@pytest.mark.usefixtures("configured_guardrail")
class TestVideosGuardrail:
    """POST /v1/videos: INPUT guardrail on the generation prompt.

    Ref: https://developers.openai.com/api/reference/resources/videos
         stdapi/routes/openai_videos.py:create_video
    """

    def test_masked_prompt_reaches_the_model(
        self,
        app_client: TestClient,
        video_backend: _StubVideoModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A masking-only intervention generates from the anonymized prompt."""
        _stub_guardrail(monkeypatch, _MASKED_RESPONSE)

        response = app_client.post(
            "/v1/videos",
            json={
                "model": "amazon.nova-reel-v1:1",
                "prompt": "a portrait of me@example.com",
            },
        )

        assert response.status_code == 200, response.text
        assert video_backend.prompts == [_MASKED_TEXT]

    def test_blocked_prompt_fails_as_content_filter(
        self,
        app_client: TestClient,
        video_backend: _StubVideoModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A blocking intervention fails with 400/content_filter before the model."""
        _stub_guardrail(monkeypatch, _BLOCKED_RESPONSE)

        response = app_client.post(
            "/v1/videos",
            json={"model": "amazon.nova-reel-v1:1", "prompt": "bad prompt"},
        )

        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "content_filter"
        assert not video_backend.prompts


class _StubTtsModel:
    """Stub TTS backend recording synthesized texts."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def tts(
        self,
        *,
        text: str,
        voice: str,
        resp_format: str,
        speed: float = 1.0,
        extra_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record the text and return a canned audio stream."""
        del voice, resp_format, speed, extra_params
        self.texts.append(text)

        async def _stream() -> AsyncGenerator[bytes]:
            yield b"audio"

        return {"audio_stream": _stream(), "input_tokens": 5, "output_tokens": 0}


@pytest.fixture
def tts_backend(monkeypatch: pytest.MonkeyPatch) -> _StubTtsModel:
    """Stub model validation and the TTS backend on the speech route."""
    stub = _StubTtsModel()
    monkeypatch.setattr(openai_audio_speech, "validate_model", _validate_any_model)
    monkeypatch.setattr(openai_audio_speech, "get_audio_model", lambda _model_id: stub)
    return stub


@pytest.mark.local
@pytest.mark.usefixtures("configured_guardrail")
class TestSpeechGuardrail:
    """POST /v1/audio/speech: INPUT guardrail on the synthesized text.

    Ref: https://developers.openai.com/api/reference/resources/audio/subresources/speech
         stdapi/routes/openai_audio_speech.py:create_speech
    """

    def test_masked_input_is_synthesized(
        self,
        app_client: TestClient,
        tts_backend: _StubTtsModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A masking-only intervention synthesizes the anonymized text."""
        _stub_guardrail(monkeypatch, _MASKED_RESPONSE)

        response = app_client.post(
            "/v1/audio/speech",
            json={
                "model": "amazon.polly-standard",
                "input": "call me@example.com",
                "voice": "alloy",
            },
        )

        assert response.status_code == 200, response.text
        assert response.content == b"audio"
        assert tts_backend.texts == [_MASKED_TEXT]

    def test_blocked_input_fails_as_content_filter(
        self,
        app_client: TestClient,
        tts_backend: _StubTtsModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A blocking intervention fails with 400/content_filter before synthesis."""
        _stub_guardrail(monkeypatch, _BLOCKED_RESPONSE)

        response = app_client.post(
            "/v1/audio/speech",
            json={"model": "amazon.polly-standard", "input": "bad", "voice": "alloy"},
        )

        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "content_filter"
        assert not tts_backend.texts


@pytest.mark.local
@pytest.mark.usefixtures("guardrail_context", "request_log", "usage_scope")
class TestTranscriptionOutputGuardrail:
    """Transcription output guarded as ``OUTPUT`` after the backend call.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
         stdapi/models/audio/amazon_transcribe.py:AudioModel._format_transcription_response
    """

    async def test_blocked_transcript_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A blocking intervention on the transcript fails the request."""
        stub = _stub_guardrail(monkeypatch, _BLOCKED_RESPONSE)

        with pytest.raises(GuardrailInterventionError) as exc_info:
            await AudioModel._format_transcription_response(  # noqa: SLF001
                _STUB_TRANSCRIPT_DATA, "json", 2, 2.0
            )

        assert exc_info.value.status == 400
        assert stub.requests[0]["source"] == "OUTPUT"

    async def test_masked_transcript_returned_for_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Masking is honored on the plain json format: masked text is returned."""
        _stub_guardrail(monkeypatch, _MASKED_RESPONSE)

        response = await AudioModel._format_transcription_response(  # noqa: SLF001
            _STUB_TRANSCRIPT_DATA, "json", 2, 2.0
        )

        assert not isinstance(response, str)
        assert response.text == _MASKED_TEXT  # type: ignore[union-attr]

    async def test_masked_transcript_fails_for_subtitles(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Subtitle formats cannot carry masked text, so masking blocks them."""
        _stub_guardrail(monkeypatch, _MASKED_RESPONSE)
        transcript_data = {
            **_STUB_TRANSCRIPT_DATA,
            "subtitle_content": "1\n00:00:00,000 --> 00:00:02,000\nhello world\n",
        }

        with pytest.raises(GuardrailInterventionError, match="masked"):
            await AudioModel._format_transcription_response(  # noqa: SLF001
                transcript_data, "srt", 2, 2.0
            )

    async def test_stream_blocked_transcript_raises_before_any_event(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The buffered stream wrapper fails without emitting any event."""
        _stub_guardrail(monkeypatch, _BLOCKED_RESPONSE)

        async def _events() -> AsyncGenerator[
            TranscriptionTextDeltaEvent | TranscriptionTextDoneEvent
        ]:
            yield TranscriptionTextDeltaEvent(
                delta="hello world", type="transcript.text.delta"
            )
            yield TranscriptionTextDoneEvent(
                text="hello world", type="transcript.text.done"
            )

        with pytest.raises(GuardrailInterventionError):
            _ = [event async for event in _guarded_transcript_events(_events())]

    async def test_stream_masked_transcript_replaces_deltas(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Masking collapses the stream into one masked delta plus a masked done."""
        _stub_guardrail(monkeypatch, _MASKED_RESPONSE)

        async def _events() -> AsyncGenerator[
            TranscriptionTextDeltaEvent | TranscriptionTextDoneEvent
        ]:
            yield TranscriptionTextDeltaEvent(delta="me@", type="transcript.text.delta")
            yield TranscriptionTextDeltaEvent(
                delta="example.com", type="transcript.text.delta"
            )
            yield TranscriptionTextDoneEvent(
                text="me@example.com", type="transcript.text.done"
            )

        events = [event async for event in _guarded_transcript_events(_events())]

        assert [type(event) for event in events] == [
            TranscriptionTextDeltaEvent,
            TranscriptionTextDoneEvent,
        ]
        assert events[0].delta == _MASKED_TEXT  # type: ignore[union-attr]
        assert events[1].text == _MASKED_TEXT  # type: ignore[union-attr]

    async def test_stream_clean_transcript_passes_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A clean transcript streams the original events unchanged."""
        _stub_guardrail(monkeypatch, _CLEAN_RESPONSE)
        source_events = [
            TranscriptionTextDeltaEvent(delta="hello", type="transcript.text.delta"),
            TranscriptionTextDoneEvent(text="hello", type="transcript.text.done"),
        ]

        async def _events() -> AsyncGenerator[
            TranscriptionTextDeltaEvent | TranscriptionTextDoneEvent
        ]:
            for event in source_events:
                yield event

        events = [event async for event in _guarded_transcript_events(_events())]

        assert events == source_events


@pytest.mark.local
@pytest.mark.usefixtures("guardrail_context", "request_log", "usage_scope")
class TestTranslationOutputGuardrail:
    """Translation output guarded as ``OUTPUT`` after transcription+translation.

    The stubbed transcript is English, so AWS Translate short-circuits and the
    guarded text is the transcript itself.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
         stdapi/models/audio/amazon_transcribe.py:AudioModel.stt_translate
    """

    @staticmethod
    def _audio() -> InputFile:
        """Return a tiny data-URI audio input; it is never read.

        Returns:
            An ``InputFile`` pointing at inline base64 audio.
        """
        return InputFile("data:audio/wav;base64,AAAA")

    async def test_blocked_translation_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A blocking intervention on the translated text fails the request."""
        _stub_transcribe(monkeypatch)
        stub = _stub_guardrail(monkeypatch, _BLOCKED_RESPONSE)

        with pytest.raises(GuardrailInterventionError):
            await AudioModel("amazon.transcribe").stt_translate(
                self._audio(), "json", prompt=None
            )

        assert stub.requests[0]["source"] == "OUTPUT"

    async def test_masked_translation_returned_for_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Masking is honored on the plain json format: masked text is returned."""
        _stub_transcribe(monkeypatch)
        _stub_guardrail(monkeypatch, _MASKED_RESPONSE)

        response = await AudioModel("amazon.transcribe").stt_translate(
            self._audio(), "json", prompt=None
        )

        assert not isinstance(response, str)
        assert response.text == _MASKED_TEXT  # type: ignore[union-attr]
