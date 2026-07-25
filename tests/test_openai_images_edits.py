"""Tests for OpenAI Images API /v1/images/edits endpoint.

This module contains comprehensive tests for the /v1/images/edits endpoint,
validating functionality, error handling, and compliance with OpenAI API specification.
"""

import pytest
from openai import BadRequestError, OpenAI
from starlette.testclient import TestClient

from stdapi.api_errors import UnsupportedModelError
from stdapi.routes import openai_images_edits

# Import validation helpers from generations tests
from .test_openai_images_generations import (
    validate_base64_image,
    validate_error_response,
    validate_streaming_image_response,
    validate_timestamp,
    validate_url_format,
)


class TestImagesEditsBasic:
    """Basic tests for /v1/images/edits endpoint.

    These tests focus on new code paths and features specific to image edits
    (mask support, editing operations) compared to image generations tests.
    """

    @pytest.mark.expensive
    def test_edit_image_with_mask(
        self, openai_client: OpenAI, sample_image_file: bytes, sample_mask_file: bytes
    ) -> None:
        """Test image editing with explicit mask (unique to edits endpoint)."""
        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_mask_file,
            prompt="A blue circle in the masked area",
            model="amazon.nova-canvas-v1:0",
            size="512x512",
            n=1,
        )

        assert response.created is not None
        validate_timestamp(response.created)
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].url is not None
        validate_url_format(response.data[0].url)

        # Validate usage metadata
        assert hasattr(response, "usage")
        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        assert response.usage.total_tokens > 0

        # Check input tokens details - edits should have both text and image tokens
        assert hasattr(response.usage, "input_tokens_details")
        assert response.usage.input_tokens_details.text_tokens >= 0
        assert response.usage.input_tokens_details.image_tokens > 0

    @pytest.mark.expensive
    def test_edit_image_b64_json_with_mask(
        self, openai_client: OpenAI, sample_image_file: bytes, sample_mask_file: bytes
    ) -> None:
        """Test masked editing with base64 response format."""
        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_mask_file,
            prompt="A colorful pattern",
            model="amazon.nova-canvas-v1:0",
            size="512x512",
            n=1,
            response_format="b64_json",
        )

        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.data[0].url is None

        image_format = validate_base64_image(response.data[0].b64_json)
        assert image_format == "png"

    def test_invalid_model(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Test error with invalid model."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                prompt="A test image",
                model="invalid-model-name",
                size="512x512",
            )
        validate_error_response(exc_info.value)

    def test_invalid_mask_format(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Test error with invalid mask format (unique to edits endpoint)."""
        invalid_mask = b"not a valid image"

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                mask=invalid_mask,
                prompt="A test image",
                model="amazon.nova-canvas-v1:0",
                size="512x512",
            )
        validate_error_response(exc_info.value)

    def test_image_array_notation_accepted(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Test that 'image[]' field name is recognized, not treated as missing image.

        Regression: image[] was silently ignored after the JSON body support was added,
        causing a 400 "at least 1 image" error instead of reaching the model.
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}images/edits",
            files={"image[]": ("image.png", sample_image_file, "image/png")},
            data={"prompt": "test", "model": "invalid-model-name", "size": "512x512"},
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        # image[] was parsed → error is about the model, not missing image
        assert response.status_code == 400
        assert "model" in response.json()["error"]["message"].lower()

    def test_image_array_notation_invalid_type(self, openai_client: OpenAI) -> None:
        """Test that a non-file value for 'image[]' returns 400."""
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}images/edits",
            data={
                "prompt": "test",
                "model": "amazon.nova-canvas-v1:0",
                "size": "512x512",
                "image[]": "not_a_file",
            },
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"

    @pytest.mark.skip("Currently no models to use for this test case")
    def test_model_not_supporting_image_editing(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Test error when model doesn't support image editing."""
        # Use a model that only supports text-to-image generation (not editing)
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                prompt="Edit this image",
                model="",  # Only supports generation and variations
                size="512x512",
            )

        error_msg = str(exc_info.value).lower()
        assert (
            "not supported" in error_msg
            or "editing" in error_msg
            or "edit" in error_msg
        )

    @pytest.mark.xfail(
        reason="Multiple images are miss-interpreted as 'image[]' by FastAPI instead of 'image'."
        "Currently, there is no model that support multiple images, so this is not an issue."
    )
    def test_multiple_images_error(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Test error when multiple images are provided to edit endpoint."""
        # Pass a sequence of images to trigger the "exactly one image" validation
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=[sample_image_file, sample_image_file],
                prompt="Edit this image",
                model="amazon.nova-canvas-v1:0",
                size="512x512",
            )

        error_msg = str(exc_info.value).lower()
        assert "exactly one image" in error_msg or "one image" in error_msg

    def test_mask_not_supported_error(
        self,
        openai_client: OpenAI,
        sample_image_file: bytes,
        sample_mask_file: bytes,
        use_official_api: bool,
    ) -> None:
        """Test error when mask is provided to a model that doesn't support it."""
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        # Use a model that doesn't support masks (like background removal or upscale)
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                mask=sample_mask_file,
                prompt="Remove background",
                model="stability.stable-image-remove-background-v1:0",
                size="512x512",
            )

        error_msg = str(exc_info.value).lower()
        assert "mask" in error_msg
        assert "not supported" in error_msg or "not allowed" in error_msg

    def test_mask_required_error(
        self, openai_client: OpenAI, sample_image_file: bytes, use_official_api: bool
    ) -> None:
        """Test error when mask is required but not provided."""
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        # Use VIRTUAL_TRY_ON taskType which requires a mask
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                prompt="Try on garment",
                model="amazon.nova-canvas-v1:0",
                size="1024x1024",
                extra_body={"taskType": "VIRTUAL_TRY_ON"},
            )

        error_msg = str(exc_info.value).lower()
        assert "mask" in error_msg
        assert "required" in error_msg

    @pytest.mark.expensive
    def test_edit_with_streaming(
        self, openai_client: OpenAI, sample_image_file: bytes, sample_mask_file: bytes
    ) -> None:
        """Test image editing with streaming support."""
        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_mask_file,
            prompt="Add colorful elements",
            model="amazon.nova-canvas-v1:0",
            size="512x512",
            n=1,
            stream=True,
        )

        # Validate the streaming response structure
        assert response is not None
        validate_streaming_image_response(response)

    @pytest.mark.expensive
    def test_image_parameter_aliases(
        self, openai_client: OpenAI, sample_image_file: bytes, sample_mask_file: bytes
    ) -> None:
        """Test that both 'image' and 'image[]' parameter names work correctly."""
        # Derive HTTP client from OpenAI client
        http_client = openai_client._client  # noqa: SLF001

        # Prepare common headers with authentication
        headers = {
            "Authorization": f"Bearer {openai_client.api_key}",
            "OpenAI-Organization": openai_client.organization or "",
        }

        # Test with 'image' parameter name
        files_image = {
            "image": ("image.png", sample_image_file, "image/png"),
            "mask": ("mask.png", sample_mask_file, "image/png"),
        }
        data_image = {
            "prompt": "A red square",
            "model": "amazon.nova-canvas-v1:0",
            "size": "512x512",
            "n": "1",
        }

        response_image = http_client.post(
            f"{openai_client.base_url}images/edits",
            files=files_image,
            data=data_image,
            headers=headers,
        )
        assert response_image.status_code == 200, (
            f"Expected 200, got {response_image.status_code}: {response_image.text}"
        )
        json_response_image = response_image.json()
        assert json_response_image.get("created") is not None
        validate_timestamp(json_response_image["created"])
        assert json_response_image.get("data") is not None
        assert len(json_response_image["data"]) == 1
        assert json_response_image["data"][0].get("url") is not None
        validate_url_format(json_response_image["data"][0]["url"])

        # Test with 'image[]' parameter name (array notation)
        files_image_array = {
            "image[]": ("image.png", sample_image_file, "image/png"),
            "mask": ("mask.png", sample_mask_file, "image/png"),
        }
        data_image_array = {
            "prompt": "A blue circle",
            "model": "amazon.nova-canvas-v1:0",
            "size": "512x512",
            "n": "1",
        }

        response_image_array = http_client.post(
            f"{openai_client.base_url}images/edits",
            files=files_image_array,
            data=data_image_array,
            headers=headers,
        )
        assert response_image_array.status_code == 200, (
            f"Expected 200, got {response_image_array.status_code}: {response_image_array.text}"
        )
        json_response_array = response_image_array.json()
        assert json_response_array.get("created") is not None
        validate_timestamp(json_response_array["created"])
        assert json_response_array.get("data") is not None
        assert len(json_response_array["data"]) == 1
        assert json_response_array["data"][0].get("url") is not None
        validate_url_format(json_response_array["data"][0]["url"])

        # Test error case: no image provided
        data_no_image = {
            "prompt": "A test image",
            "model": "amazon.nova-canvas-v1:0",
            "size": "512x512",
            "n": "1",
        }

        response_no_image = http_client.post(
            f"{openai_client.base_url}images/edits", data=data_no_image, headers=headers
        )
        assert response_no_image.status_code == 400
        error_response = response_no_image.json()
        assert "error" in error_response
        assert error_response["error"]["type"] == "invalid_request_error"

        # Verify the error details match Pydantic's min_length validation format
        assert "message" in error_response["error"]
        error_message = error_response["error"]["message"]
        assert "image" in error_message.lower()
        assert (
            "validation" in error_message.lower() or "at least" in error_message.lower()
        )

        # Test error case: invalid type for image[] (string instead of file)
        data_invalid_type = {
            "prompt": "A test image",
            "model": "amazon.nova-canvas-v1:0",
            "size": "512x512",
            "n": "1",
            "image[]": "not_a_file",  # String instead of file upload
        }

        response_invalid_type = http_client.post(
            f"{openai_client.base_url}images/edits",
            data=data_invalid_type,
            headers=headers,
        )
        assert response_invalid_type.status_code == 400
        error_response_invalid = response_invalid_type.json()
        assert "error" in error_response_invalid
        assert error_response_invalid["error"]["type"] == "invalid_request_error"

        # Verify the error message indicates type mismatch
        assert "message" in error_response_invalid["error"]
        error_message_invalid = error_response_invalid["error"]["message"]
        assert "image" in error_message_invalid.lower()


class TestImagesEditsJsonBody:
    """Tests for /v1/images/edits endpoint with application/json request body.

    Covers the structured JSON format where images are referenced by URL or
    Files API identifier instead of binary multipart uploads.
    """

    @pytest.fixture(autouse=True)
    def _skip_on_official_api(self, use_official_api: bool) -> None:
        """Skip when running against the official API (model not available there)."""
        if use_official_api:
            pytest.skip(
                "amazon.nova-canvas-v1:0 not available on the official OpenAI API"
            )

    @pytest.mark.expensive
    def test_edit_with_image_url(
        self, openai_client: OpenAI, sample_image_file_base64: str
    ) -> None:
        """Test image edit via JSON body using a data URL for the image reference.

        Validates:
            - 200 response with valid image data
            - Response structure matches ImagesResponse schema
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}images/edits",
            json={
                "model": "amazon.nova-canvas-v1:0",
                "prompt": "Make it look like a painting",
                "images": [{"image_url": sample_image_file_base64}],
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
    def test_edit_with_file_id(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Test image edit via JSON body using a Files API file_id reference.

        Validates:
            - Upload succeeds and returns a file-* identifier
            - JSON body edit with that file_id returns 200 with image data
        """
        uploaded = openai_client.files.create(
            file=("image.png", sample_image_file, "image/png"), purpose="assistants"
        )
        try:
            http_client = openai_client._client  # noqa: SLF001
            response = http_client.post(
                f"{openai_client.base_url}images/edits",
                json={
                    "model": "amazon.nova-canvas-v1:0",
                    "prompt": "Add a subtle vignette effect",
                    "images": [{"file_id": uploaded.id}],
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

    def test_missing_images_returns_400(self, openai_client: OpenAI) -> None:
        """Test that a JSON body without an 'images' field returns 400.

        Validates:
            - 400 status code
            - Error type is invalid_request_error
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}images/edits",
            json={"model": "amazon.nova-canvas-v1:0", "prompt": "Make it darker"},
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 400
        error = response.json().get("error", {})
        assert error.get("type") == "invalid_request_error"

    def test_empty_image_ref_returns_400(self, openai_client: OpenAI) -> None:
        """Test that an ImageRef with neither file_id nor image_url returns 400.

        Validates:
            - 400 status code
            - Error type is invalid_request_error
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}images/edits",
            json={
                "model": "amazon.nova-canvas-v1:0",
                "prompt": "Make it darker",
                "images": [{}],
            },
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 400
        error = response.json().get("error", {})
        assert error.get("type") == "invalid_request_error"


@pytest.mark.local
class TestImagesEditsModelField:
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

        monkeypatch.setattr(openai_images_edits, "validate_model", _validate_model)
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
            "/v1/images/edits",
            json={
                "model": "probe-model-id",
                "prompt": "Make it darker",
                "images": [{"image_url": "data:image/png;base64,AA=="}],
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
            "/v1/images/edits",
            data={"model": "probe-model-id", "prompt": "test"},
            files={"image": ("image.png", b"fake-bytes", "image/png")},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"
        assert probed_model_ids == ["probe-model-id"]
