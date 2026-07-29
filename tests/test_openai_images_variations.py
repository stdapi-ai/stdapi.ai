"""Tests for OpenAI Images API /v1/images/variations endpoint.

This module contains comprehensive tests for the /v1/images/variations endpoint,
validating functionality, error handling, and compliance with OpenAI API specification.
"""

import pytest
from openai import BadRequestError, NotFoundError, OpenAI
from starlette.testclient import TestClient

from stdapi.api_errors import UnsupportedModelError
from stdapi.routes import openai_images_variations

# Import validation helpers from generations tests
from .test_openai_images_generations import (
    validate_base64_image,
    validate_error_response,
    validate_timestamp,
    validate_url_format,
)


class TestImagesVariationsBasic:
    """Basic tests for /v1/images/variations endpoint.

    These tests focus on variation-specific features (no prompt, image-only input)
    that differ from image generations.
    """

    @pytest.mark.expensive
    def test_create_variation_basic(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Test basic image variation creation (no prompt, unique to variations)."""
        response = openai_client.images.create_variation(
            image=sample_image_file,
            model="stability.sd3-5-large-v1:0",
            size="512x512",
            n=1,
        )

        assert response.created is not None
        validate_timestamp(response.created)
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].url is not None
        validate_url_format(response.data[0].url)
        # Stability generates at its own resolution but keeps the 1:1 ratio asked for.
        assert response.size is not None
        width, height = (int(value) for value in response.size.split("x"))
        assert width == height

        # Validate usage tracking
        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.input_tokens_details.image_tokens > 0
        assert response.usage.output_tokens == 1
        assert response.usage.total_tokens > 0

    @pytest.mark.expensive
    def test_create_variations_multiple(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Test creating multiple variations."""
        response = openai_client.images.create_variation(
            image=sample_image_file,
            model="stability.sd3-5-large-v1:0",
            size="512x512",
            n=2,
        )

        assert response.data is not None
        assert len(response.data) == 2
        for img_data in response.data:
            assert img_data.url is not None
            validate_url_format(img_data.url)

        # Validate usage for multiple images
        assert response.usage is not None
        assert response.usage.output_tokens == 2

    @pytest.mark.expensive
    def test_create_variation_b64_json(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Test creating variations with base64 response format."""
        response = openai_client.images.create_variation(
            image=sample_image_file,
            model="stability.sd3-5-large-v1:0",
            size="512x512",
            n=1,
            response_format="b64_json",
        )

        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.data[0].url is None
        validate_base64_image(response.data[0].b64_json)


class TestImagesVariationsErrors:
    """Error handling tests for /v1/images/variations endpoint.

    Focus on variation-specific errors (missing image, unsupported models).
    """

    def test_invalid_model(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Test error handling for invalid model.

        Official API returns 404 (NotFoundError) for unknown models; the
        local implementation returns 400 (BadRequestError).
        """
        with pytest.raises((BadRequestError, NotFoundError)) as exc_info:
            openai_client.images.create_variation(
                image=sample_image_file, model="invalid-model"
            )

        if isinstance(exc_info.value, BadRequestError):
            validate_error_response(exc_info.value)

    def test_unsupported_model_for_variations(
        self, openai_client: OpenAI, sample_image_file: bytes, use_official_api: bool
    ) -> None:
        """Test error for models that don't support variations (model-specific check)."""
        if use_official_api:
            pytest.skip(
                "Bedrock model catalog (generation/editing-only models) is "
                "gateway-specific; no equivalent model exists on the official API"
            )
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.create_variation(
                image=sample_image_file,
                model="stability.stable-fast-upscale-v1:0",  # Only supports generation and editing
            )

        error_msg = str(exc_info.value).lower()
        assert "not supported" in error_msg or "invalid model" in error_msg

    def test_non_image_model_for_variations(
        self, openai_client: OpenAI, sample_image_file: bytes, use_official_api: bool
    ) -> None:
        """Test error when using non-image model for variations (default error path)."""
        if use_official_api:
            pytest.skip(
                "Bedrock chat model catalog is gateway-specific; no equivalent "
                "model exists on the official API"
            )
        # Use a chat model which doesn't support image variations at all
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.create_variation(
                image=sample_image_file,
                model="anthropic.claude-3-5-sonnet-20241022-v2:0",  # Chat model, not image model
            )

        error_msg = str(exc_info.value).lower()
        assert "does not support" in error_msg or "invalid model" in error_msg


class TestImagesVariationsProviderParams:
    """Tests for provider-specific parameters in variations.

    Test unique variation parameters like Stability's strength and negative_prompt.
    """

    @pytest.fixture(autouse=True)
    def _skip_non_local(self, use_official_api: bool) -> None:
        """Skip web search tests when running against the official Anthropic API."""
        if use_official_api:
            pytest.skip("Unittest only for local tests.")

    @pytest.mark.expensive
    def test_variation_with_strength(
        self, openai_client: OpenAI, sample_image_file_base64: str
    ) -> None:
        """Test variation with Stability's image-to-image `strength` parameter.

        Sent as a JSON body: over ``multipart/form-data`` every extra parameter
        reaches the model as a string, which Bedrock rejects for numeric fields.
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}images/variations",
            json={
                "model": "stability.sd3-5-large-v1:0",
                "image": {"image_url": sample_image_file_base64},
                "size": "512x512",
                "n": 1,
                "strength": 0.7,
            },
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        assert len(response.json()["data"]) == 1

    @pytest.mark.expensive
    def test_variation_with_negative_prompt(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Test variation with Stability's `negative_prompt` parameter."""
        response = openai_client.images.create_variation(
            image=sample_image_file,
            model="stability.sd3-5-large-v1:0",
            size="512x512",
            n=1,
            extra_body={"negative_prompt": "blurry, low quality"},
        )

        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].url is not None


class TestImagesVariationsJsonBody:
    """Tests for /v1/images/variations endpoint with application/json request body."""

    @pytest.fixture(autouse=True)
    def _skip_on_official_api(self, use_official_api: bool) -> None:
        """Skip when running against the official API (variations removed, model unavailable)."""
        if use_official_api:
            pytest.skip(
                "Variations endpoint removed from official OpenAI API; "
                "stability.sd3-5-large-v1:0 not available there."
            )

    @pytest.mark.expensive
    def test_variation_with_image_url(
        self, openai_client: OpenAI, sample_image_file_base64: str
    ) -> None:
        """Test image variation via JSON body using a data URL image reference.

        Validates:
            - 200 response with valid image data
            - Response structure matches ImagesResponse schema
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}images/variations",
            json={
                "model": "stability.sd3-5-large-v1:0",
                "image": {"image_url": sample_image_file_base64},
                "response_format": "b64_json",
                "size": "512x512",
                "n": 1,
            },
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        body = response.json()
        assert body.get("created") is not None
        assert body.get("data") is not None
        assert len(body["data"]) == 1
        assert body["data"][0].get("b64_json") is not None
        validate_base64_image(body["data"][0]["b64_json"])

    @pytest.mark.expensive
    def test_variation_with_file_id(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Test image variation via JSON body using a Files API file_id reference.

        Validates:
            - Upload succeeds and returns a file-* identifier
            - JSON body variation with that file_id returns 200 with image data
        """
        uploaded = openai_client.files.create(
            file=("image.png", sample_image_file, "image/png"), purpose="assistants"
        )
        try:
            http_client = openai_client._client  # noqa: SLF001
            response = http_client.post(
                f"{openai_client.base_url}images/variations",
                json={
                    "model": "stability.sd3-5-large-v1:0",
                    "image": {"file_id": uploaded.id},
                    "response_format": "b64_json",
                    "size": "512x512",
                    "n": 1,
                },
                headers={"Authorization": f"Bearer {openai_client.api_key}"},
            )
            assert response.status_code == 200, (
                f"Expected 200, got {response.status_code}: {response.text}"
            )
            body = response.json()
            assert body.get("data") is not None
            assert len(body["data"]) == 1
            assert body["data"][0].get("b64_json") is not None
            validate_base64_image(body["data"][0]["b64_json"])
        finally:
            openai_client.files.delete(uploaded.id)

    def test_missing_image_returns_400(self, openai_client: OpenAI) -> None:
        """Test that a JSON body without an 'image' field returns 400.

        Validates:
            - 400 status code
            - Error type is invalid_request_error
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}images/variations",
            json={"model": "stability.sd3-5-large-v1:0"},
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 400
        error = response.json().get("error", {})
        assert error.get("type") == "invalid_request_error"


@pytest.mark.local
class TestImagesVariationsModelField:
    """Unit tests for the 'model' field regardless of request encoding (no AWS calls)."""

    @pytest.fixture
    def client(self, api_key: str) -> TestClient:
        """Test client without lifespan (no AWS startup), pre-authenticated."""
        from stdapi.main import app  # noqa: PLC0415

        return TestClient(app, headers={"Authorization": f"Bearer {api_key}"})

    @pytest.fixture
    def probed_model_ids(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Stub validate_model to record the requested model id and fail fast."""
        calls: list[str] = []

        async def _validate_model(
            model_id: str, *_args: object, **_kwargs: object
        ) -> None:
            calls.append(model_id)
            raise UnsupportedModelError(model_id, status=400)

        monkeypatch.setattr(openai_images_variations, "validate_model", _validate_model)
        return calls

    def test_json_body_model_field_reaches_model_resolution(
        self, client: TestClient, probed_model_ids: list[str]
    ) -> None:
        """A model supplied only in the JSON body reaches model resolution.

        Regression: the unused, form-only 'model' Form parameter carried
        'min_length=1' with an empty-string default, which rejected every
        JSON-body request with a 422 before the JSON 'model' field was read.
        """
        response = client.post(
            "/v1/images/variations",
            json={
                "model": "probe-model-id",
                "image": {"image_url": "data:image/png;base64,AA=="},
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"
        assert probed_model_ids == ["probe-model-id"]

    def test_multipart_form_model_field_still_works(
        self, client: TestClient, probed_model_ids: list[str]
    ) -> None:
        """A model supplied via multipart form data still reaches model resolution unchanged."""
        response = client.post(
            "/v1/images/variations",
            data={"model": "probe-model-id"},
            files={"image": ("image.png", b"fake-bytes", "image/png")},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"
        assert probed_model_ids == ["probe-model-id"]

    def test_multipart_missing_image_returns_400(self, client: TestClient) -> None:
        """A multipart request without an 'image' file returns 400, not an unhandled error.

        Regression: the ValidationError for the missing 'image' file was raised
        outside validation_error_handler(), bypassing FastAPI's RequestValidationError
        conversion and producing an unhandled 500 instead of a JSON 400 envelope.
        """
        response = client.post(
            "/v1/images/variations", data={"model": "stability.sd3-5-large-v1:0"}
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"
