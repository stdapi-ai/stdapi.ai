"""OpenAI-compatible tests for Stability AI image variations."""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openai import OpenAI

# Stable Diffusion Models
STABLE_DIFFUSION_3_5_LARGE = "stability.sd3-5-large-v1:0"

# Test groups
STABILITY_ALL = (STABLE_DIFFUSION_3_5_LARGE,)


class TestStabilityVariations:
    """Model-specific tests for Stability AI image variations.

    Focus on Stability-specific variation parameters and image-to-image generation.
    """

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", STABILITY_ALL)
    def test_variation_b64_single(
        self,
        openai_client: OpenAI,
        use_openai_api: bool,
        sample_image_file: bytes,
        model_id: str,
    ) -> None:
        """Test basic variation with base64 response format."""
        if use_openai_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        response = openai_client.images.create_variation(
            image=sample_image_file, model=model_id, response_format="b64_json"
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
