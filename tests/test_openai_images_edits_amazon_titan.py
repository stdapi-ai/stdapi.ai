"""OpenAI-compatible tests for Amazon Titan Image Generator editing."""

from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError

if TYPE_CHECKING:
    from openai import OpenAI

TITAN_V2 = "amazon.titan-image-generator-v2:0"

TITAN_ALL = (TITAN_V2,)
TITAN_SAMPLE = (TITAN_V2,)


class TestAmazonTitanEditing:
    """Model-specific tests for Amazon Titan image editing.

    Focus on Titan-specific parameters and features unique to this provider.
    """

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_edit_with_extra_parameters(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        sample_mask_file: bytes,
        model_id: str,
    ) -> None:
        """Test editing with Titan-specific negativeText parameter."""
        if use_official_api:
            pytest.skip("Amazon Titan is not available on the official OpenAI API")

        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_mask_file,
            prompt="A blue circle in the center",
            model=model_id,
            size="512x512",
            n=1,
            response_format="b64_json",
            extra_body={
                "inpaintingParams": {"negativeText": "blurry, distorted, low quality"},
                "imageGenerationConfig": {"seed": 42, "quality": "premium"},
            },
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.size == "512x512"  # type: ignore[comparison-overlap]

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_ALL)
    def test_edit_b64_single(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        sample_mask_file: bytes,
        model_id: str,
    ) -> None:
        """Test basic editing with base64 response format."""
        if use_official_api:
            pytest.skip("Amazon Titan is not available on the official OpenAI API")

        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_mask_file,
            prompt="A green square",
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
        assert response.usage.input_tokens_details.text_tokens >= 0
        assert response.usage.output_tokens == 1

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_edit_with_outpainting_task_type(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        sample_mask_file: bytes,
        model_id: str,
    ) -> None:
        """Test editing with OUTPAINTING taskType."""
        if use_official_api:
            pytest.skip("Amazon Titan is not available on the official OpenAI API")

        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_mask_file,
            prompt="Extend the scene with a forest",
            model=model_id,
            size="512x512",
            n=1,
            response_format="b64_json",
            extra_body={"taskType": "OUTPAINTING"},
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.size == "512x512"  # type: ignore[comparison-overlap]

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_edit_with_background_removal_task_type(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        model_id: str,
    ) -> None:
        """Test editing with BACKGROUND_REMOVAL taskType (no mask supported)."""
        if use_official_api:
            pytest.skip("Amazon Titan is not available on the official OpenAI API")

        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="Remove background",
            model=model_id,
            size="512x512",
            n=1,
            response_format="b64_json",
            extra_body={"taskType": "BACKGROUND_REMOVAL"},
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.size == "512x512"  # type: ignore[comparison-overlap]

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_edit_inpainting_without_mask(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        model_id: str,
    ) -> None:
        """Test INPAINTING without mask (mask is optional)."""
        if use_official_api:
            pytest.skip("Amazon Titan is not available on the official OpenAI API")

        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="A blue square in the center",
            model=model_id,
            size="512x512",
            n=1,
            response_format="b64_json",
            extra_body={
                "taskType": "INPAINTING",
                "inPaintingParams": {"maskPrompt": "llama"},
            },
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.size == "512x512"  # type: ignore[comparison-overlap]

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_edit_outpainting_without_mask(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        model_id: str,
    ) -> None:
        """Test OUTPAINTING without mask (mask is optional)."""
        if use_official_api:
            pytest.skip("Amazon Titan is not available on the official OpenAI API")

        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="Extend with ocean view",
            model=model_id,
            size="512x512",
            n=1,
            response_format="b64_json",
            extra_body={
                "taskType": "OUTPAINTING",
                "outPaintingParams": {"maskPrompt": "llama"},
            },
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.size == "512x512"  # type: ignore[comparison-overlap]

    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_edit_with_invalid_task_type(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        sample_mask_file: bytes,
        model_id: str,
    ) -> None:
        """Test that invalid taskType raises BadRequestError."""
        if use_official_api:
            pytest.skip("Amazon Titan is not available on the official OpenAI API")

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                mask=sample_mask_file,
                prompt="A test prompt",
                model=model_id,
                size="512x512",
                extra_body={"taskType": "INVALID_TASK_TYPE"},
            )

        assert (
            "taskType" in str(exc_info.value).lower()
            or "INPAINTING" in str(exc_info.value)
            or "OUTPAINTING" in str(exc_info.value)
            or "BACKGROUND_REMOVAL" in str(exc_info.value)
        )
