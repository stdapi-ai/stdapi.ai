"""OpenAI-compatible /v1/images/generations tests for Amazon Nova Canvas.

The whole module is skipped: Nova Canvas is deprecated (EOL September 30, 2026).

Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
     stdapi/routes/openai_images_generations.py:create_images
     stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob
"""

import pytest
from openai import BadRequestError, OpenAI

NOVA_CANVAS_V1 = "amazon.nova-canvas-v1:0"

NOVA_CANVAS_ALL = (NOVA_CANVAS_V1,)
NOVA_CANVAS_SAMPLE = (NOVA_CANVAS_V1,)


class TestAmazonNovaCanvas:
    """Text-to-image generation with Amazon Nova Canvas."""

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_extra_params_negative_text(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """``textToImageParams.negativeText`` and a seed are accepted as provider extras.

        Neither field exists in the OpenAI Images API: the gateway collects
        unknown body fields as provider extras and merges the ``textToImageParams``
        and ``imageGenerationConfig`` sub-objects into the Nova Canvas payload, so
        a field Nova Canvas rejects surfaces as a 400 instead of an image.

        Ref: stdapi/aws_bedrock.py:get_extra_model_parameters
             stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob._apply_extra_params
        """
        if use_official_api:
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
        assert response.data[0].url is None
        assert response.size == "1024x1024"

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_ALL)
    def test_generate_b64_single(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """A prompt returns one base64 PNG at the requested size with default quality.

        Nova Canvas always emits PNG and honors the requested width/height, so the
        gateway reports the requested size back. With ``quality`` left at ``auto``
        no ``imageGenerationConfig.quality`` is sent and the response reports the
        neutral ``medium`` level.

        Ref: stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob._invoke_and_process_response
             stdapi/routes/_images_common.py:build_images_response
        """
        if use_official_api:
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
        assert response.size == "1024x1024"
        assert response.output_format == "png"
        assert response.quality == "medium"
        assert response.data is not None
        assert len(response.data) == 1
        img = response.data[0]
        assert img.b64_json is not None
        assert img.url is None

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_generate_url_multiple_images(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """``n=2`` with ``response_format="url"`` returns two distinct presigned URLs.

        Each image is uploaded under its own 1-based indexed S3 key, so the two
        entries are different objects rather than the same URL repeated.

        Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html
             stdapi/models/image/__init__.py:ImageGenerationJobBase._get_image_url
        """
        if use_official_api:
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
            assert item.url.startswith("https://")
            assert item.b64_json is None
        assert response.data[0].url != response.data[1].url, (
            "each generated image must get its own presigned URL"
        )

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_quality_is_accepted(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """OpenAI ``quality="high"`` maps to Nova Canvas ``premium`` and echoes back ``high``.

        The gateway normalizes OpenAI's quality vocabulary to Nova Canvas'
        ``standard``/``premium`` pair, then reports the level actually used;
        ``premium`` is the only one surfaced as ``high``.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
             stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob._apply_quality_and_style
        """
        if use_official_api:
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
        assert response.data[0].b64_json is not None
        assert response.quality == "high"

    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_invalid_style_raises(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """An OpenAI style value Nova Canvas does not define is rejected by Bedrock.

        The gateway does not validate ``style`` locally: it upper-cases the value
        into ``textToImageParams.style``, so ``vivid`` reaches Bedrock, which
        rejects it with a ``ValidationException`` mapped to a 400.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
             stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob._apply_quality_and_style
             stdapi/aws_bedrock.py:AWS_ERROR_MAP
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        with pytest.raises(BadRequestError) as excinfo:
            openai_client.images.generate(
                model=model_id,
                prompt="A logo of a tree.",
                response_format="b64_json",
                size="512x512",
                style="vivid",
            )

        error = excinfo.value
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict), f"Unexpected error body: {body!r}"
        assert body["type"] == "invalid_request_error"
        assert body["code"] == "ValidationException"
        assert body["message"]

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_generate_with_color_guided_task_type(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """``taskType=COLOR_GUIDED_GENERATION`` generates from a hex color palette.

        The gateway accepts the Nova Canvas task type as a provider extra on the
        text-to-image route and moves the prompt into
        ``colorGuidedGenerationParams.text``.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
             stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob._get_request_color_guided_generation
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        response = openai_client.images.generate(
            model=model_id,
            prompt="A beautiful landscape with sunset tones",
            response_format="b64_json",
            size="1024x1024",
            extra_body={
                "taskType": "COLOR_GUIDED_GENERATION",
                "colorGuidedGenerationParams": {
                    "colors": ["#FF6347", "#FFD700", "#FF4500"],
                    "negativeText": "dark, cold colors",
                },
            },
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.data[0].url is None
        assert response.output_format == "png"
        assert response.size == "1024x1024"

    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_generate_with_invalid_task_type(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """An unknown ``taskType`` is rejected with the list of legal values.

        Only the two text-to-image task types are reachable from
        /v1/images/generations; the edit-only ones are rejected here even though
        Nova Canvas itself supports them.

        Ref: stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob._generate_images_from_text
             stdapi/api_providers/openai.py:_format_error
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        with pytest.raises(BadRequestError) as excinfo:
            openai_client.images.generate(
                model=model_id,
                prompt="A test prompt",
                response_format="b64_json",
                size="1024x1024",
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

    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_generate_color_guided_missing_colors(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """``COLOR_GUIDED_GENERATION`` without ``colors`` is rejected before invoking Bedrock.

        ``colors`` has no OpenAI equivalent, so the gateway cannot default it and
        names the missing provider field in the error message.

        Ref: stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob._get_request_color_guided_generation
             stdapi/api_providers/openai.py:_format_error
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        with pytest.raises(BadRequestError) as excinfo:
            openai_client.images.generate(
                model=model_id,
                prompt="A test prompt",
                response_format="b64_json",
                size="1024x1024",
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


pytest.skip("Amazon Nova Canvas is deprecated", allow_module_level=True)
