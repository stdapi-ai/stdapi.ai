"""OpenAI-compatible /v1/images/generations tests for Amazon Titan Image Generator.

The whole module is skipped: Titan Image Generator is deprecated and the gateway
remaps it to Amazon Nova Canvas.

Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
     stdapi/routes/openai_images_generations.py:create_images
     stdapi/models/image/amazon_titan_image_generator.py:_ImageGenerationJob
"""

import pytest
from openai import BadRequestError, OpenAI

from tests.conftest import smallest_image_size

TITAN_V2 = "amazon.titan-image-generator-v2:0"

TITAN_ALL = (TITAN_V2,)
TITAN_SAMPLE = (TITAN_V2,)

#: Cheapest size accepted by Titan, requested wherever the size is incidental.
TITAN_SIZE = smallest_image_size(TITAN_V2)


#: Every test in this module is reported as skipped: the model is deprecated.
pytestmark = pytest.mark.skip(reason="Amazon Titan Image Generator is deprecated")


@pytest.fixture(autouse=True)
def _skip_on_official_api(use_official_api: bool) -> None:
    """Skip every test here: Titan Image Generator has no official OpenAI equivalent."""
    if use_official_api:
        pytest.skip(
            "Amazon Titan Image Generator is not available on the official OpenAI API"
        )


class TestAmazonTitanImageGenerator:
    """Text-to-image generation with the Amazon Titan Image Generator family."""

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_ALL)
    def test_generate_b64_single(self, openai_client: OpenAI, model_id: str) -> None:
        """A prompt returns one base64 PNG at the requested size with default quality.

        Titan always emits PNG and honors the requested ``width``/``height``
        exactly, so the gateway reports the requested size back. With
        ``quality`` left at ``auto`` no ``imageGenerationConfig.quality`` is sent
        and the response reports the neutral ``medium`` level.

        Ref: stdapi/models/image/amazon_titan_image_generator.py:_ImageGenerationJob._invoke_and_process_response
             stdapi/routes/_images_common.py:build_images_response
        """
        response = openai_client.images.generate(
            model=model_id,
            prompt="A simple watercolor of a mountain.",
            response_format="b64_json",
            size=TITAN_SIZE,
        )
        assert response.created > 0
        assert response.size == TITAN_SIZE
        assert response.output_format == "png"
        assert response.quality == "medium"
        assert response.data is not None
        assert len(response.data) == 1
        img = response.data[0]
        assert img.b64_json is not None
        assert img.url is None

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_extra_params_cfg_scale(self, openai_client: OpenAI, model_id: str) -> None:
        """``imageGenerationConfig.cfgScale`` is accepted as a provider extra.

        ``cfgScale`` is not an OpenAI Images field: the gateway collects unknown
        body fields as provider extras and merges the ``imageGenerationConfig``
        sub-object into the Titan payload, so a value Titan rejects surfaces as a
        400 ``ValidationException`` instead of an image.

        Ref: stdapi/aws_bedrock.py:get_extra_model_parameters
             stdapi/models/image/amazon_titan_image_generator.py:_ImageGenerationJob._set_extra_config
        """
        response = openai_client.images.generate(
            model=model_id,
            prompt="A watercolor of a mountain.",
            response_format="b64_json",
            size=TITAN_SIZE,
            extra_body={"imageGenerationConfig": {"cfgScale": 7.5}},
        )
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.data[0].url is None

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_multiple_images(self, openai_client: OpenAI, model_id: str) -> None:
        """``n=2`` returns two base64 images from a single Titan invocation.

        ``n`` maps to ``imageGenerationConfig.numberOfImages``, so both images
        come back from one call and share the requested size.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
             stdapi/models/image/amazon_titan_image_generator.py:_ImageGenerationJob._generate_images_from_text
        """
        response = openai_client.images.generate(
            model=model_id,
            prompt="Three variations of a sunset over the ocean.",
            response_format="b64_json",
            n=2,
            size=TITAN_SIZE,
        )
        assert response.data is not None
        assert len(response.data) == 2
        assert response.size == TITAN_SIZE
        for item in response.data:
            assert item.b64_json is not None
            assert item.url is None

    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_style_unsupported(self, openai_client: OpenAI, model_id: str) -> None:
        """``style`` is rejected with a 400 naming the parameter.

        Titan has no style knob, so the job rejects any style before invoking
        Bedrock. The gateway envelope carries no ``param``/``code`` for this
        class of error, only the message.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._validate_no_style
             stdapi/api_providers/openai.py:_format_error
        """
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.images.generate(
                model=model_id,
                prompt="Portrait photo of a cat",
                style="natural",
                response_format="b64_json",
            )

        error = excinfo.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict), f"Unexpected error body: {body!r}"
        assert body["type"] == "invalid_request_error"
        assert body["message"] == '"style" parameter is not supported by this model.'

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_generate_with_color_guided_task_type(
        self, openai_client: OpenAI, model_id: str
    ) -> None:
        """``taskType=COLOR_GUIDED_GENERATION`` generates from a hex color palette.

        The gateway accepts the Titan task type as a provider extra on the
        text-to-image route and moves the prompt into
        ``colorGuidedGenerationParams.text``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
             stdapi/models/image/amazon_titan_image_generator.py:_ImageGenerationJob._get_request_color_guided_generation
        """
        response = openai_client.images.generate(
            model=model_id,
            prompt="A vibrant sunset with these specific colors",
            response_format="b64_json",
            size=TITAN_SIZE,
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
        assert response.data[0].url is None
        assert response.output_format == "png"
        assert response.size == TITAN_SIZE

    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_generate_with_invalid_task_type(
        self, openai_client: OpenAI, model_id: str
    ) -> None:
        """An unknown ``taskType`` is rejected with the list of legal values.

        Only the two text-to-image task types are reachable from
        /v1/images/generations; the edit-only ones are rejected here even though
        Titan itself supports them.

        Ref: stdapi/models/image/amazon_titan_image_generator.py:_ImageGenerationJob._generate_images_from_text
             stdapi/api_providers/openai.py:_format_error
        """
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.images.generate(
                model=model_id,
                prompt="A test prompt",
                response_format="b64_json",
                size=TITAN_SIZE,
                extra_body={"taskType": "INVALID_TASK_TYPE"},
            )

        error = excinfo.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict), f"Unexpected error body: {body!r}"
        assert body["type"] == "invalid_request_error"
        assert (
            body["message"]
            == '"taskType" value must be "TEXT_IMAGE" or "COLOR_GUIDED_GENERATION".'
        )

    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_generate_color_guided_missing_colors(
        self, openai_client: OpenAI, model_id: str
    ) -> None:
        """``COLOR_GUIDED_GENERATION`` without ``colors`` is rejected before invoking Bedrock.

        ``colors`` has no OpenAI equivalent, so the gateway cannot default it and
        names the missing provider field in the error message.

        Ref: stdapi/models/image/amazon_titan_image_generator.py:_ImageGenerationJob._get_request_color_guided_generation
             stdapi/api_providers/openai.py:_format_error
        """
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.images.generate(
                model=model_id,
                prompt="A test prompt",
                response_format="b64_json",
                size=TITAN_SIZE,
                extra_body={"taskType": "COLOR_GUIDED_GENERATION"},
            )

        error = excinfo.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict), f"Unexpected error body: {body!r}"
        assert body["type"] == "invalid_request_error"
        assert body["message"] == (
            "Required parameter for COLOR_GUIDED_GENERATION: "
            "colorGuidedGenerationParams.colors"
        )
