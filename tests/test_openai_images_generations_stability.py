"""OpenAI-compatible /v1/images/generations tests for Stability AI models on Bedrock.

Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-stable-ultra-text-image-request-response.html
     stdapi/routes/openai_images_generations.py:create_images
     stdapi/models/image/_stability.py:StabilityImageGenerationJobBase
"""

import pytest
from openai import BadRequestError, OpenAI
from pybase64 import b64decode

from tests.conftest import smallest_image_size

STABILITY_CORE = "stability.stable-image-core-v1:1"
STABILITY_SD35 = "stability.sd3-5-large-v1:0"
STABILITY_ULTRA = "stability.stable-image-ultra-v1:1"

STABILITY_ALL = (STABILITY_CORE, STABILITY_SD35, STABILITY_ULTRA)
STABILITY_SAMPLE = (STABILITY_CORE,)


@pytest.fixture(autouse=True)
def _skip_on_official_api(use_official_api: bool) -> None:
    """Skip every test here: the Stability models have no official OpenAI equivalent."""
    if use_official_api:
        pytest.skip("Stability AI is not available on the official OpenAI API")


class TestStabilityImages:
    """Text-to-image generation with the Stability AI models on Bedrock."""

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", STABILITY_ALL)
    def test_generate_b64_single(self, openai_client: OpenAI, model_id: str) -> None:
        """A prompt returns exactly one base64 JPEG when ``output_format="jpeg"``.

        The Stability request carries ``output_format`` verbatim for the three
        formats the models emit natively (png/jpeg/webp), and the gateway echoes
        the effective format on the response, so the decoded bytes must be JPEG.

        Ref: stdapi/models/image/_stability.py:StabilityImageGenerationJobBase._finalize_request
             stdapi/routes/_images_common.py:build_images_response
        """
        response = openai_client.images.generate(
            model=model_id,
            prompt="A charcoal sketch of a city skyline.",
            response_format="b64_json",
            output_format="jpeg",
        )
        assert response.created > 0
        assert response.output_format == "jpeg"
        assert response.data is not None
        assert len(response.data) == 1
        img = response.data[0]
        assert img.b64_json is not None
        assert img.url is None
        assert b64decode(img.b64_json).startswith(b"\xff\xd8\xff"), (
            "output_format=jpeg must yield JPEG bytes"
        )

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", STABILITY_SAMPLE)
    def test_extra_params_negative_prompt(
        self, openai_client: OpenAI, model_id: str
    ) -> None:
        """``negative_prompt`` is accepted as a provider extra and reaches the model.

        ``negative_prompt`` is not an OpenAI Images field: the gateway collects
        unknown body fields as provider-specific extras and merges them into the
        Stability payload, so a name Stability does not accept would come back as
        a 400 ``ValidationException`` instead of an image.

        Ref: stdapi/aws_bedrock.py:get_extra_model_parameters
             stdapi/models/image/_stability.py:StabilityImageGenerationJobBase._finalize_request
        """
        response = openai_client.images.generate(
            model=model_id,
            prompt="A charcoal sketch of a city skyline.",
            response_format="b64_json",
            extra_body={"negative_prompt": "blurry, low quality"},
        )
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.data[0].url is None

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", STABILITY_SAMPLE)
    def test_generate_and_convert_to_webp(
        self, openai_client: OpenAI, model_id: str
    ) -> None:
        """``output_format="webp"`` yields WebP bytes and is echoed on the response.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._ensure_image_output_format
        """
        response = openai_client.images.generate(
            model=model_id,
            prompt="A siamese cat.",
            response_format="b64_json",
            size=smallest_image_size(model_id),
            output_format="webp",
        )
        assert response.created > 0
        assert response.output_format == "webp"
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
        self, openai_client: OpenAI, model_id: str
    ) -> None:
        """``quality`` is rejected with a 400 naming the parameter.

        Stability text-to-image has no quality knob, so the job rejects any
        resolved quality before invoking Bedrock. The gateway envelope carries no
        ``param``/``code`` for this class of error, only the message.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._validate_no_quality
             stdapi/api_providers/openai.py:_format_error
        """
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.images.generate(
                model=model_id,
                prompt="A landscape painting.",
                response_format="b64_json",
                quality="high",
            )

        error = excinfo.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict), f"Unexpected error body: {body!r}"
        assert body["type"] == "invalid_request_error"
        assert body["message"] == '"quality" parameter is not supported by this model.'

    @pytest.mark.parametrize("model_id", STABILITY_SAMPLE)
    def test_style_unsupported_raises(
        self, openai_client: OpenAI, model_id: str
    ) -> None:
        """``style`` is rejected with a 400 naming the parameter.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._validate_no_style
             stdapi/api_providers/openai.py:_format_error
        """
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.images.generate(
                model=model_id,
                prompt="A portrait.",
                response_format="b64_json",
                style="natural",
            )

        error = excinfo.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict), f"Unexpected error body: {body!r}"
        assert body["type"] == "invalid_request_error"
        assert body["message"] == '"style" parameter is not supported by this model.'
