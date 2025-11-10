"""OpenAI-compatible tests for Stability image generation models."""

import pytest
from openai import BadRequestError, OpenAI
from pybase64 import b64decode

STABILITY_CORE = "stability.stable-image-core-v1:1"
STABILITY_SD35 = "stability.sd3-5-large-v1:0"
STABILITY_ULTRA = "stability.stable-image-ultra-v1:1"

STABILITY_ALL = (STABILITY_CORE, STABILITY_SD35, STABILITY_ULTRA)
STABILITY_SAMPLE = (STABILITY_CORE,)


class TestStabilityImages:
    """Basic behavior checks for Stability image generation models."""

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", STABILITY_ALL)
    def test_generate_b64_single(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Simple prompt returns one base64 image when response_format=b64_json."""
        if use_openai_api:
            pytest.skip("Stability models are not available on the official OpenAI API")

        response = openai_client.images.generate(
            model=model_id,
            prompt="A charcoal sketch of a city skyline.",
            response_format="b64_json",
            size="1024x1024",
            output_format="jpeg",
        )
        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        img = response.data[0]
        assert img.b64_json is not None
        assert img.url is None

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", STABILITY_SAMPLE)
    def test_extra_params_negative_prompt(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Extra body parameter negative_prompt is forwarded to provider.

        Not part of OpenAI Images API; accepted here as provider-specific field.
        """
        if use_openai_api:
            pytest.skip("Stability models are not available on the official OpenAI API")

        response = openai_client.images.generate(
            model=model_id,
            prompt="A charcoal sketch of a city skyline.",
            response_format="b64_json",
            size="1024x1024",
            extra_body={"negative_prompt": "blurry, low quality"},
        )
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", STABILITY_SAMPLE)
    def test_multiple_images(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Requesting multiple images with b64_json works."""
        if use_openai_api:
            pytest.skip("Stability models are not available on the official OpenAI API")

        response = openai_client.images.generate(
            model=model_id,
            prompt="Two poster variants of mountains.",
            response_format="b64_json",
            n=2,
            size="1024x1024",
        )
        assert response.data is not None
        assert len(response.data) == 2
        for item in response.data:
            assert item.b64_json is not None

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", STABILITY_SAMPLE)
    def test_generate_and_convert_to_webp(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Generate a format not supported natively by the model."""
        if use_openai_api:
            pytest.skip("Stability models are not available on the official OpenAI API")

        response = openai_client.images.generate(
            model=model_id,
            prompt="A siamese cat.",
            response_format="b64_json",
            size="1024x1024",
            output_format="webp",
        )
        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        img = response.data[0]
        assert img.b64_json is not None
        assert img.url is None
        image_bytes = b64decode(img.b64_json)
        assert image_bytes.startswith(b"RIFF")
        assert b"WEBP" in image_bytes[:12]

    @pytest.mark.parametrize("model_id", STABILITY_SAMPLE)
    def test_quality_unsupported_raises(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Passing quality should raise 400 as the backend forbids it."""
        if use_openai_api:
            pytest.skip("Stability models are not available on the official OpenAI API")

        with pytest.raises(BadRequestError):
            openai_client.images.generate(
                model=model_id,
                prompt="A landscape painting.",
                response_format="b64_json",
                quality="high",
            )

    @pytest.mark.parametrize("model_id", STABILITY_SAMPLE)
    def test_style_unsupported_raises(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Passing style should raise 400 as the backend forbids it."""
        if use_openai_api:
            pytest.skip("Stability models are not available on the official OpenAI API")

        with pytest.raises(BadRequestError):
            openai_client.images.generate(
                model=model_id,
                prompt="A portrait.",
                response_format="b64_json",
                style="natural",
            )
