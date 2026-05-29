"""OpenAI-compatible tests for Amazon Nova Canvas image variations."""

from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError

if TYPE_CHECKING:
    from openai import OpenAI

NOVA_CANVAS_V1 = "amazon.nova-canvas-v1:0"

NOVA_CANVAS_ALL = (NOVA_CANVAS_V1,)
NOVA_CANVAS_SAMPLE = (NOVA_CANVAS_V1,)


class TestAmazonNovaCanvasVariations:
    """Model-specific tests for Amazon Nova Canvas image variations.

    Focus on Nova Canvas-specific variation parameters and features.
    """

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_ALL)
    def test_variation_b64_single(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        model_id: str,
    ) -> None:
        """Test basic variation with base64 response format."""
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        response = openai_client.images.create_variation(
            image=sample_image_file,
            model=model_id,
            size="512x512",
            response_format="b64_json",
        )

        assert response.created > 0
        assert response.size is not None
        assert response.data is not None
        assert len(response.data) == 1
        img = response.data[0]
        assert img.b64_json is not None
        assert img.url is None

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.input_tokens_details.image_tokens > 0
        assert response.usage.input_tokens_details.text_tokens == 0
        assert response.usage.output_tokens == 1

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_variation_with_text_image_task_type(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        model_id: str,
    ) -> None:
        """Test variation with TEXT_IMAGE taskType (condition image generation)."""
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        response = openai_client.images.create_variation(
            image=sample_image_file,
            model=model_id,
            size="512x512",
            n=1,
            response_format="b64_json",
            extra_body={"taskType": "TEXT_IMAGE"},
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.size == "512x512"  # type: ignore[comparison-overlap]

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_variation_with_color_guided_task_type(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        model_id: str,
    ) -> None:
        """Test variation with COLOR_GUIDED_GENERATION taskType."""
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        response = openai_client.images.create_variation(
            image=sample_image_file,
            model=model_id,
            size="512x512",
            n=1,
            response_format="b64_json",
            extra_body={
                "taskType": "COLOR_GUIDED_GENERATION",
                "colorGuidedGenerationParams": {
                    "colors": ["#FF5733", "#33FF57", "#3357FF"]
                },
            },
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.size == "512x512"  # type: ignore[comparison-overlap]

    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_variation_with_invalid_task_type(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        model_id: str,
    ) -> None:
        """Test that invalid taskType raises BadRequestError."""
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.create_variation(
                image=sample_image_file,
                model=model_id,
                size="512x512",
                extra_body={"taskType": "INVALID_TASK_TYPE"},
            )

        assert (
            "taskType" in str(exc_info.value).lower()
            or "IMAGE_VARIATION" in str(exc_info.value)
            or "TEXT_IMAGE" in str(exc_info.value)
            or "COLOR_GUIDED_GENERATION" in str(exc_info.value)
        )

    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_variation_color_guided_missing_colors(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        model_id: str,
    ) -> None:
        """Test that COLOR_GUIDED_GENERATION without colors raises BadRequestError."""
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.create_variation(
                image=sample_image_file,
                model=model_id,
                size="512x512",
                extra_body={
                    "taskType": "COLOR_GUIDED_GENERATION",
                    "colorGuidedGenerationParams": {},
                },
            )

        assert "colorGuidedGenerationParams.colors" in str(exc_info.value)


pytest.skip("Amazon Nova Canvas is deprecated", allow_module_level=True)
