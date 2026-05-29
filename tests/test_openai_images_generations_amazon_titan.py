"""OpenAI-compatible tests for Amazon Titan Image Generator models.

These tests mirror the embeddings tests structure and validate the
/v1/images/generations endpoint using the OpenAI Python client.
"""

import pytest
from openai import BadRequestError, OpenAI

TITAN_V2 = "amazon.titan-image-generator-v2:0"

TITAN_ALL = (TITAN_V2,)
TITAN_SAMPLE = (TITAN_V2,)


class TestAmazonTitanImageGenerator:
    """Basic behavior checks for Amazon Titan Image Generator family."""

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_ALL)
    def test_generate_b64_single(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """Simple prompt returns one base64 image when response_format=b64_json."""
        if use_official_api:
            pytest.skip(
                "Amazon Titan models are not available on the official OpenAI API"
            )

        response = openai_client.images.generate(
            model=model_id,
            prompt="A simple watercolor of a mountain.",
            response_format="b64_json",
            size="512x512",
        )
        assert response.created > 0
        assert response.size is not None
        assert response.output_format in {"png", "jpeg", "webp", None}
        assert response.quality in {"low", "medium", "high", None}
        assert response.data is not None
        assert len(response.data) == 1
        img = response.data[0]
        assert img.b64_json is not None
        assert img.url is None

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_extra_params_cfg_scale(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """Extra body parameter imageGenerationConfig.cfgScale is forwarded.

        Not part of OpenAI Images API; this project forwards provider-specific
        fields through the request body.
        """
        if use_official_api:
            pytest.skip(
                "Amazon Titan models are not available on the official OpenAI API"
            )
        response = openai_client.images.generate(
            model=model_id,
            prompt="A watercolor of a mountain.",
            response_format="b64_json",
            size="512x512",
            extra_body={"imageGenerationConfig": {"cfgScale": 7.5}},
        )
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_multiple_images(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """Requesting n>1 returns the requested number of images."""
        if use_official_api:
            pytest.skip(
                "Amazon Titan models are not available on the official OpenAI API"
            )

        response = openai_client.images.generate(
            model=model_id,
            prompt="Three variations of a sunset over the ocean.",
            response_format="b64_json",
            n=2,
            size="512x512",
        )
        assert response.data is not None
        assert len(response.data) == 2
        for item in response.data:
            assert item.b64_json is not None
            assert item.url is None

    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_style_unsupported(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """Passing style is not supported by Titan model (backend raises 400)."""
        if use_official_api:
            pytest.skip(
                "Amazon Titan models are not available on the official OpenAI API"
            )

        with pytest.raises(BadRequestError):
            openai_client.images.generate(
                model=model_id,
                prompt="Portrait photo of a cat",
                style="natural",
                response_format="b64_json",
            )

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_generate_with_color_guided_task_type(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """Test generation with COLOR_GUIDED_GENERATION taskType."""
        if use_official_api:
            pytest.skip(
                "Amazon Titan models are not available on the official OpenAI API"
            )

        response = openai_client.images.generate(
            model=model_id,
            prompt="A vibrant sunset with these specific colors",
            response_format="b64_json",
            size="512x512",
            extra_body={
                "taskType": "COLOR_GUIDED_GENERATION",
                "colorGuidedGenerationParams": {
                    "colors": ["#FF6B6B", "#FFA500", "#FFD700"],
                    "negativeText": "dark, gloomy",
                },
            },
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.size == "512x512"  # type: ignore[comparison-overlap]

    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_generate_with_invalid_task_type(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """Test that invalid taskType raises BadRequestError."""
        if use_official_api:
            pytest.skip(
                "Amazon Titan models are not available on the official OpenAI API"
            )

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(
                model=model_id,
                prompt="A test prompt",
                response_format="b64_json",
                size="512x512",
                extra_body={"taskType": "INVALID_TASK_TYPE"},
            )

        assert (
            "taskType" in str(exc_info.value).lower()
            or "TEXT_IMAGE" in str(exc_info.value)
            or "COLOR_GUIDED_GENERATION" in str(exc_info.value)
        )

    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_generate_color_guided_missing_colors(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """Test that COLOR_GUIDED_GENERATION without colors raises BadRequestError."""
        if use_official_api:
            pytest.skip(
                "Amazon Titan models are not available on the official OpenAI API"
            )

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(
                model=model_id,
                prompt="A test prompt",
                response_format="b64_json",
                size="512x512",
                extra_body={"taskType": "COLOR_GUIDED_GENERATION"},
            )

        assert "colorGuidedGenerationParams.colors" in str(exc_info.value)


pytest.skip("Amazon Titan Image Generator is deprecated", allow_module_level=True)
