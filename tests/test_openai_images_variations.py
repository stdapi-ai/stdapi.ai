"""Tests for OpenAI Images API /v1/images/variations endpoint.

This module contains comprehensive tests for the /v1/images/variations endpoint,
validating functionality, error handling, and compliance with OpenAI API specification.
"""

import pytest
from openai import BadRequestError, OpenAI

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
            model="amazon.titan-image-generator-v2:0",
            size="512x512",
            n=1,
        )

        assert response.created is not None
        validate_timestamp(response.created)
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].url is not None
        validate_url_format(response.data[0].url)
        assert response.size == "512x512"  # type: ignore[comparison-overlap]

        # Validate usage tracking - variations have only image tokens, no text
        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.input_tokens_details.image_tokens > 0
        assert response.usage.input_tokens_details.text_tokens == 0  # No prompt
        assert response.usage.output_tokens == 1
        assert response.usage.total_tokens > 0

    @pytest.mark.expensive
    def test_create_variations_multiple(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Test creating multiple variations."""
        response = openai_client.images.create_variation(
            image=sample_image_file,
            model="amazon.titan-image-generator-v2:0",
            size="512x512",
            n=3,
        )

        assert response.data is not None
        assert len(response.data) == 3
        for img_data in response.data:
            assert img_data.url is not None
            validate_url_format(img_data.url)

        # Validate usage for multiple images
        assert response.usage is not None
        assert response.usage.output_tokens == 3

    @pytest.mark.expensive
    def test_create_variation_b64_json(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Test creating variations with base64 response format."""
        response = openai_client.images.create_variation(
            image=sample_image_file,
            model="amazon.titan-image-generator-v2:0",
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
        """Test error handling for invalid model."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.create_variation(
                image=sample_image_file, model="invalid-model"
            )

        validate_error_response(exc_info.value)

    def test_unsupported_model_for_variations(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Test error for models that don't support variations (model-specific check)."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.create_variation(
                image=sample_image_file,
                model="stability.stable-fast-upscale-v1:0",  # Only supports generation and editing
            )

        error_msg = str(exc_info.value).lower()
        assert "not supported" in error_msg or "invalid model" in error_msg

    def test_non_image_model_for_variations(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Test error when using non-image model for variations (default error path)."""
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

    Test unique variation parameters like similarityStrength and text hints.
    """

    @pytest.mark.expensive
    def test_variation_with_similarity_strength(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Test variation with similarity strength parameter (unique to variations)."""
        response = openai_client.images.create_variation(
            image=sample_image_file,
            model="amazon.titan-image-generator-v2:0",
            size="512x512",
            n=1,
            extra_body={"imageVariationParams": {"similarityStrength": 0.7}},
        )

        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].url is not None

    @pytest.mark.expensive
    def test_variation_with_text_hint(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Test variation with text hint (unique to variations API)."""
        response = openai_client.images.create_variation(
            image=sample_image_file,
            model="amazon.titan-image-generator-v2:0",
            size="512x512",
            n=1,
            extra_body={
                "imageVariationParams": {
                    "text": "Keep the overall composition but change colors",
                    "similarityStrength": 0.8,
                }
            },
        )

        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].url is not None
