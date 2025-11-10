"""OpenAI-compatible tests for Amazon Nova Canvas image generation models."""

import pytest
from openai import BadRequestError, OpenAI

NOVA_CANVAS_V1 = "amazon.nova-canvas-v1:0"

NOVA_CANVAS_ALL = (NOVA_CANVAS_V1,)
NOVA_CANVAS_SAMPLE = (NOVA_CANVAS_V1,)


class TestAmazonNovaCanvas:
    """Basic behavior checks for Amazon Nova Canvas image generation."""

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_extra_params_negative_text(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Extra body parameter textToImageParams.negativeText is forwarded.

        Not part of OpenAI Images API; accepted here as provider-specific field.
        """
        if use_openai_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )
        response = openai_client.images.generate(
            model=model_id,
            prompt="A watercolor of a red fox in a forest, soft digital painting.",
            response_format="b64_json",
            size="1024x1024",
            extra_body={
                "textToImageParams": {"negativeText": "blurry"},
                "imageGenerationConfig": {"seed": 12},
            },
        )
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_ALL)
    def test_generate_b64_single(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Simple prompt returns one base64 image when response_format=b64_json."""
        if use_openai_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        response = openai_client.images.generate(
            model=model_id,
            prompt="A watercolor of a red fox in a forest, soft digital painting.",
            response_format="b64_json",
            size="1024x1024",
        )
        assert response.created > 0
        assert response.size is not None
        assert response.data is not None
        assert len(response.data) == 1
        img = response.data[0]
        assert img.b64_json is not None
        assert img.url is None

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_generate_url_multiple_images(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Requesting multiple images with URL response works."""
        if use_openai_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        response = openai_client.images.generate(
            model=model_id,
            prompt="Logo concept of a lighthouse.",
            response_format="url",
            n=2,
            size="512x512",
        )
        assert response.data is not None
        assert len(response.data) == 2
        for item in response.data:
            assert item.url is not None
            assert item.b64_json is None

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_quality_is_accepted(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Nova Canvas supports quality; request should succeed."""
        if use_openai_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        response = openai_client.images.generate(
            model=model_id,
            prompt="A photorealistic portrait of a golden retriever.",
            response_format="b64_json",
            size="1024x1024",
            quality="high",
        )
        assert response.data is not None
        assert len(response.data) == 1

    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_invalid_style_raises(
        self, openai_client: OpenAI, use_openai_api: bool, model_id: str
    ) -> None:
        """Unknown style should result in a 400 from the backend."""
        if use_openai_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        with pytest.raises(BadRequestError):
            openai_client.images.generate(
                model=model_id,
                prompt="A logo of a tree.",
                response_format="b64_json",
                size="512x512",
                style="vivid",
            )
