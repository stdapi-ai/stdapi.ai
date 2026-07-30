"""Stability AI backend of the OpenAI-compatible ``/v1/images/variations`` endpoint.

Stability has no dedicated variation operation: the gateway maps a variation
request to an ``image-to-image`` generation seeded with a fixed default prompt.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/stable-image-services.html
     stdapi/models/image/stability_stable_diffusion.py:TextToImageJob
"""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openai import OpenAI

# Stable Diffusion Models
STABLE_DIFFUSION_3_5_LARGE = "stability.sd3-5-large-v1:0"

# Test groups
STABILITY_ALL = (STABLE_DIFFUSION_3_5_LARGE,)


class TestStabilityVariations:
    """Variations served by Stability image-to-image generation."""

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", STABILITY_ALL)
    def test_variation_b64_single(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        model_id: str,
    ) -> None:
        """A default-size variation returns one inline PNG with image-only usage.

        Stability takes an ``aspect_ratio`` rather than a pixel size, so the
        default 1024x1024 request becomes 1:1 and the echoed ``size`` is the
        square image actually produced. The prompt is supplied internally, hence
        ``text_tokens == 0``.

        Ref: stdapi/models/image/_stability.py:StabilityImageGenerationJobBase._get_aspect_ratio
             stdapi/routes/_images_common.py:build_images_response
        """
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        response = openai_client.images.create_variation(
            image=sample_image_file, model=model_id, response_format="b64_json"
        )

        assert response.created > 0
        assert response.size is not None
        width, height = (int(value) for value in response.size.split("x"))
        assert width == height, "a 1:1 request must produce a square image"
        assert response.output_format == "png"
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
        assert (
            response.usage.total_tokens
            == response.usage.input_tokens + response.usage.output_tokens
        )
