"""Tests for OpenAI Images API /v1/images/edits endpoint.

This module contains comprehensive tests for the /v1/images/edits endpoint,
validating functionality, error handling, and compliance with OpenAI API specification.
"""

import pytest
from openai import BadRequestError, OpenAI

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
        use_openai_api: bool,
    ) -> None:
        """Test error when mask is provided to a model that doesn't support it."""
        if use_openai_api:
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
        self, openai_client: OpenAI, sample_image_file: bytes, use_openai_api: bool
    ) -> None:
        """Test error when mask is required but not provided."""
        if use_openai_api:
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
