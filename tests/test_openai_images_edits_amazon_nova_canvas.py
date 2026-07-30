"""``/v1/images/edits`` backed by Amazon Nova Canvas.

Nova Canvas is the only Bedrock image backend offering ``VIRTUAL_TRY_ON`` and a
quality/style mapping. The whole module is skipped: the model reaches end of
life on 2026-09-30.

Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     https://stdapi.ai/api_openai_images_edits/
     https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
     stdapi/routes/openai_images_edits.py:edit_images
     https://github.com/stdapi-ai/stdapi.ai/issues/93
"""

import base64
from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError

from tests.conftest import OUTPUT_DIR, SAMPLES_DIR, smallest_image_size

#: Every test in this module is reported as skipped: the model is deprecated.
pytestmark = pytest.mark.skip(reason="Amazon Nova Canvas is deprecated")

if TYPE_CHECKING:
    from openai import OpenAI

NOVA_CANVAS_V1 = "amazon.nova-canvas-v1:0"

NOVA_CANVAS_ALL = (NOVA_CANVAS_V1,)
NOVA_CANVAS_SAMPLE = (NOVA_CANVAS_V1,)

#: Cheapest size accepted by Nova Canvas, requested wherever the size is incidental.
NOVA_CANVAS_SIZE = smallest_image_size(NOVA_CANVAS_V1)


def _decoded_png(b64_json: str | None) -> bytes:
    """Decode a base64 image payload and assert it carries a PNG signature.

    Args:
        b64_json: Base64-encoded image data from an ``ImagesResponse``.

    Returns:
        The decoded image bytes.
    """
    assert b64_json is not None
    data = base64.b64decode(b64_json, validate=True)
    assert data.startswith(b"\x89PNG\r\n\x1a\n"), "expected a PNG image payload"
    return data


@pytest.fixture(autouse=True)
def _skip_on_official_api(use_official_api: bool) -> None:
    """Skip every test here: Nova Canvas has no official OpenAI equivalent."""
    if use_official_api:
        pytest.skip("Amazon Nova Canvas is not available on the official OpenAI API")


class TestAmazonNovaCanvasEditing:
    """Nova Canvas ``taskType`` dispatch and provider extras on the edits route.

    Ref: stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob._edit_image
         stdapi/routes/_images_common.py:build_images_response
    """

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_edit_with_extra_parameters(
        self,
        openai_client: OpenAI,
        sample_image_file: bytes,
        sample_mask_file: bytes,
        model_id: str,
    ) -> None:
        """Provider extras are merged into the ``taskType``'s own parameter block.

        ``negativeText`` and ``seed`` have no OpenAI equivalent. Bedrock rejects
        the invocation with a ``ValidationException`` when they are not nested
        under ``inPaintingParams`` / ``imageGenerationConfig``, so a successful
        image is the observable proof that the merge happened.

        Ref: stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob._apply_extra_params
             stdapi/aws_bedrock.py:get_extra_model_parameters
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_mask_file,
            prompt="A blue circle in the center",
            model=model_id,
            size=NOVA_CANVAS_SIZE,
            n=1,
            response_format="b64_json",
            extra_body={
                "inPaintingParams": {"negativeText": "blurry, distorted, low quality"},
                "imageGenerationConfig": {"seed": 42},
            },
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        _decoded_png(response.data[0].b64_json)
        assert response.data[0].url is None
        assert response.size == NOVA_CANVAS_SIZE
        assert response.output_format == "png"
        assert response.background == "opaque"

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_ALL)
    def test_edit_b64_single(
        self,
        openai_client: OpenAI,
        sample_image_file: bytes,
        sample_mask_file: bytes,
        model_id: str,
    ) -> None:
        """A masked edit returns one base64 PNG, no URL, and a split token usage.

        Supplying a mask selects the ``INPAINTING`` task type. The fixture mask
        is a plain black/white PNG with no alpha channel, so it reaches Bedrock
        unchanged. Both the source image and the mask count as input images, so
        at most two of the input tokens are attributed to images.

        Ref: stdapi/utils.py:alpha_mask_to_bw
             https://docs.aws.amazon.com/nova/latest/userguide/image-gen-access.html
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_mask_file,
            prompt="A green square",
            model=model_id,
            size=NOVA_CANVAS_SIZE,
            response_format="b64_json",
        )

        assert response.created > 0
        assert response.size == NOVA_CANVAS_SIZE
        assert response.output_format == "png"
        assert response.background == "opaque"
        assert response.data is not None
        assert len(response.data) == 1
        img = response.data[0]
        assert img.b64_json is not None
        assert img.url is None
        _decoded_png(img.b64_json)

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.input_tokens_details.image_tokens > 0
        assert response.usage.input_tokens_details.image_tokens <= 2, (
            "only the source image and the mask are input images"
        )
        assert response.usage.input_tokens_details.text_tokens >= 0
        assert (
            response.usage.input_tokens_details.image_tokens
            + response.usage.input_tokens_details.text_tokens
            == response.usage.input_tokens
        )
        assert response.usage.output_tokens == 1
        assert (
            response.usage.total_tokens
            == response.usage.input_tokens + response.usage.output_tokens
        )

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_ALL)
    def test_edit_b64_single_without_mask(
        self, openai_client: OpenAI, sample_image_file: bytes, model_id: str
    ) -> None:
        """A mask-less edit still returns one base64 PNG, with a single input image.

        Nova Canvas defaults to ``INPAINTING`` only when a mask is supplied;
        without one the source image is sent as ``textToImageParams``'
        ``conditionImage``. With no mask uploaded, only one input image is
        accounted for in the usage split.

        Ref: stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob._edit_image
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="A green square",
            model=model_id,
            size=NOVA_CANVAS_SIZE,
            response_format="b64_json",
        )

        assert response.created > 0
        assert response.size == NOVA_CANVAS_SIZE
        assert response.output_format == "png"
        assert response.data is not None
        assert len(response.data) == 1
        img = response.data[0]
        assert img.b64_json is not None
        assert img.url is None
        _decoded_png(img.b64_json)

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.input_tokens_details.image_tokens > 0
        assert response.usage.input_tokens_details.image_tokens == 1, (
            "the source image is the only input image"
        )
        assert response.usage.input_tokens_details.text_tokens >= 0
        assert (
            response.usage.input_tokens_details.image_tokens
            + response.usage.input_tokens_details.text_tokens
            == response.usage.input_tokens
        )
        assert response.usage.output_tokens == 1

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_edit_with_outpainting_task_type(
        self,
        openai_client: OpenAI,
        sample_image_file: bytes,
        sample_mask_file: bytes,
        model_id: str,
    ) -> None:
        """An explicit ``OUTPAINTING`` task type overrides the mask-driven default.

        The fixture mask is already a pure black/white PNG, so it is forwarded
        verbatim; outpainting regenerates the white pixels where inpainting
        would have regenerated the black ones.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-access.html
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_mask_file,
            prompt="Extend the scene with mountains in the background",
            model=model_id,
            size=NOVA_CANVAS_SIZE,
            n=1,
            response_format="b64_json",
            extra_body={"taskType": "OUTPAINTING"},
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        _decoded_png(response.data[0].b64_json)
        assert response.data[0].url is None
        assert response.size == NOVA_CANVAS_SIZE
        assert response.output_format == "png"

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_edit_with_outpainting_task_type_and_alpha_mask(
        self,
        openai_client: OpenAI,
        sample_image_file: bytes,
        sample_alpha_mask_file: bytes,
        model_id: str,
    ) -> None:
        """``OUTPAINTING`` accepts an OpenAI-style alpha-transparency mask.

        Bedrock only takes an alpha-less black/white mask, so the RGBA mask must
        be converted — with the outpainting polarity (white = generate) rather
        than inpainting's (black = edit) — for the invocation to succeed. The
        resulting pixel values are asserted at unit level in
        ``tests/test_models_image.py``; here the accepted request and the
        returned image are the observable evidence.

        Ref: stdapi/utils.py:alpha_mask_to_bw
             https://docs.aws.amazon.com/nova/latest/userguide/image-gen-access.html
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_alpha_mask_file,
            prompt="Extend the scene with mountains in the background",
            model=model_id,
            size=NOVA_CANVAS_SIZE,
            n=1,
            response_format="b64_json",
            extra_body={"taskType": "OUTPAINTING"},
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        _decoded_png(response.data[0].b64_json)
        assert response.data[0].url is None
        assert response.size == NOVA_CANVAS_SIZE
        assert response.output_format == "png"
        assert response.usage is not None
        assert response.usage.input_tokens_details.image_tokens <= 2, (
            "only the source image and the mask are input images"
        )

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_edit_with_background_removal_task_type(
        self, openai_client: OpenAI, sample_image_file: bytes, model_id: str
    ) -> None:
        """``BACKGROUND_REMOVAL`` needs no mask and ignores the prompt.

        The Bedrock request carries only ``backgroundRemovalParams.image``; a
        mask would be rejected by the gateway before the invocation.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
             stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob._get_request_background_removal
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="Remove the background",
            model=model_id,
            size=NOVA_CANVAS_SIZE,
            n=1,
            response_format="b64_json",
            extra_body={"taskType": "BACKGROUND_REMOVAL"},
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        _decoded_png(response.data[0].b64_json)
        assert response.data[0].url is None
        assert response.size == NOVA_CANVAS_SIZE
        assert response.output_format == "png"

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    @pytest.mark.parametrize("mask_type", ["IMAGE", "PROMPT", "GARMENT"])
    def test_edit_with_virtual_try_on(
        self,
        openai_client: OpenAI,
        chat_vision_judge_model: str,
        model_id: str,
        mask_type: str,
    ) -> None:
        """``VIRTUAL_TRY_ON`` dresses the source person with the reference garment.

        The task maps the OpenAI fields onto Nova Canvas's own: ``image`` becomes
        ``sourceImage``, ``mask`` becomes ``referenceImage``, and ``prompt`` feeds
        whichever mask sub-object the ``maskType`` selects — ``maskPrompt`` for
        ``PROMPT``, ``garmentClass`` for ``GARMENT``, ``maskImage`` for ``IMAGE``
        (supplied through ``virtualTryOnParams`` here because the OpenAI client
        caps ``prompt`` at 1024 characters).

        Ref: stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob._get_request_virtual_try_on
             https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
        """
        source_image = (SAMPLES_DIR / "vto_upper_body_source.jpg").read_bytes()
        reference_image = (SAMPLES_DIR / "vto_upper_body_reference.jpg").read_bytes()

        # Build the request based on maskType
        # Each maskType uses the prompt differently:
        # - PROMPT: prompt is used for maskPrompt (default behavior)
        # - GARMENT: prompt is used for garmentClass
        # - IMAGE: prompt is used for maskImage (base64 encoded mask)
        extra_body = {"taskType": "VIRTUAL_TRY_ON"}

        if mask_type == "PROMPT":
            # For PROMPT maskType, let default values be used
            # The prompt will be used for maskPrompt automatically
            # Describe the reference garment (pink/salmon button-up shirt)
            prompt = "pink button-up shirt on upper body"
        elif mask_type == "GARMENT":
            # For GARMENT maskType, prompt is used for garmentClass
            prompt = "UPPER_BODY"
            extra_body["virtualTryOnParams"] = {"maskType": "GARMENT"}  # type: ignore[assignment]
        elif mask_type == "IMAGE":
            # For IMAGE maskType, we need to provide a base64 encoded mask image
            # We'll use a simple mask that covers the upper body area
            # For this test, we'll pass the reference image as the mask in the prompt field

            # Python Openai library limits the prompt field to 1024 characters, so pass maskImage instead
            prompt = "ignored"
            extra_body["virtualTryOnParams"] = {  # type: ignore[assignment]
                "maskType": "IMAGE",
                "imageBasedMask": {
                    "maskImage": base64.b64encode(
                        (SAMPLES_DIR / "vto_upper_body_mask.jpg").read_bytes()
                    ).decode("utf-8")
                },
            }

        else:
            msg = f"Invalid maskType: {mask_type}"
            raise ValueError(msg)

        response = openai_client.images.edit(
            image=source_image,
            mask=reference_image,
            prompt=prompt,
            model=model_id,
            size=NOVA_CANVAS_SIZE,
            n=1,
            response_format="b64_json",
            extra_body=extra_body,
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.data[0].url is None
        assert response.size == NOVA_CANVAS_SIZE
        assert response.output_format == "png"

        # Save the output image for manual inspection
        output_image_data = _decoded_png(response.data[0].b64_json)
        output_path = OUTPUT_DIR / f"vto_result_{mask_type.lower()}.jpg"
        output_path.write_bytes(output_image_data)

        # Use VLM to validate the result
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_judge_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image which is the result of a virtual try-on operation. "
                                "The reference garment is a pink/salmon colored button-up shirt. "
                                "Answer the following questions:\n"
                                "1. Is there a person wearing upper body clothing in this image?\n"
                                "2. Is the upper body clothing pink or salmon colored?\n"
                                "3. Does it appear to be a button-up shirt or similar style?\n"
                                "If all conditions are met, respond with only 'YES'. "
                                "If any condition fails, respond with a brief explanation of what you see and what failed."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{response.data[0].b64_json}"
                            },
                        },
                    ],
                }
            ],
        )

        vlm_response = validation_response.choices[0].message.content
        assert vlm_response is not None
        assert "YES" in vlm_response.upper(), (
            f"VLM validation failed for maskType={mask_type}. "
            f"Expected person wearing pink/salmon button-up shirt. "
            f"Response: {vlm_response}"
        )

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_edit_inpainting_without_mask(
        self, openai_client: OpenAI, sample_image_file: bytes, model_id: str
    ) -> None:
        """``INPAINTING`` accepts a ``maskPrompt`` instead of an uploaded mask.

        Nova Canvas requires exactly one of ``maskPrompt`` or ``maskImage``, so
        the request is only valid because no mask file was uploaded.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="A red circle in the center",
            model=model_id,
            size=NOVA_CANVAS_SIZE,
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
        _decoded_png(response.data[0].b64_json)
        assert response.data[0].url is None
        assert response.size == NOVA_CANVAS_SIZE
        assert response.output_format == "png"

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_edit_outpainting_without_mask(
        self, openai_client: OpenAI, sample_image_file: bytes, model_id: str
    ) -> None:
        """``OUTPAINTING`` accepts a ``maskPrompt`` instead of an uploaded mask.

        The prompt-derived mask replaces ``maskImage``, which is mutually
        exclusive with it.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="Extend with mountains",
            model=model_id,
            size=NOVA_CANVAS_SIZE,
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
        _decoded_png(response.data[0].b64_json)
        assert response.data[0].url is None
        assert response.size == NOVA_CANVAS_SIZE
        assert response.output_format == "png"

    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_edit_with_invalid_task_type(
        self,
        openai_client: OpenAI,
        sample_image_file: bytes,
        sample_mask_file: bytes,
        model_id: str,
    ) -> None:
        """An unknown ``taskType`` is rejected locally as an ``invalid_request_error``.

        The gateway dispatches on the ``taskType`` extra itself, so the request
        never reaches Bedrock and the message lists the edit task types Nova
        Canvas supports.

        Ref: stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob._edit_image
             stdapi/api_providers/openai.py:_format_error
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                mask=sample_mask_file,
                prompt="A test prompt",
                model=model_id,
                size=NOVA_CANVAS_SIZE,
                extra_body={"taskType": "INVALID_TASK_TYPE"},
            )

        assert exc_info.value.status_code == 400
        # The OpenAI client unwraps the envelope: body is already the error object.
        body = exc_info.value.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        message = body["message"]
        assert "taskType" in message
        assert "VIRTUAL_TRY_ON" in message, (
            f"expected the Nova Canvas edit task types, got: {message}"
        )

    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_edit_with_invalid_mask_type(
        self, openai_client: OpenAI, sample_image_file: bytes, model_id: str
    ) -> None:
        """An unknown ``virtualTryOnParams.maskType`` is rejected before invoking Bedrock.

        Only ``PROMPT``, ``GARMENT`` and ``IMAGE`` select a mask sub-object; any
        other value has no request shape to build.

        Ref: stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob._get_request_virtual_try_on
        """
        source_image = (SAMPLES_DIR / "vto_upper_body_source.jpg").read_bytes()
        reference_image = (SAMPLES_DIR / "vto_upper_body_reference.jpg").read_bytes()

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=source_image,
                mask=reference_image,
                prompt="test",
                model=model_id,
                size=NOVA_CANVAS_SIZE,
                extra_body={
                    "taskType": "VIRTUAL_TRY_ON",
                    "virtualTryOnParams": {"maskType": "INVALID_MASK_TYPE"},
                },
            )

        assert exc_info.value.status_code == 400
        body = exc_info.value.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        message = body["message"]
        assert "virtualTryOnParams.maskType" in message
        assert "INVALID_MASK_TYPE" in message, (
            f"expected the rejected maskType to be echoed, got: {message}"
        )
