"""``/v1/images/edits`` backed by Amazon Titan Image Generator V2.

Titan caps ``text``/``negativeText`` at 512 characters and inputs at 1408 px on
the longer side, which is why the tests stay at the model's cheapest size. The
whole module is skipped: the model is deprecated.

Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     https://stdapi.ai/api_openai_images_edits/
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
     stdapi/routes/openai_images_edits.py:edit_images
"""

import base64
from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError

from tests.conftest import TITAN_ALL, TITAN_SAMPLE, TITAN_V2, smallest_image_size

#: Every test in this module is reported as skipped: the model is deprecated.
pytestmark = pytest.mark.skip(reason="Amazon Titan Image Generator is deprecated")

if TYPE_CHECKING:
    from openai import OpenAI

#: Cheapest size accepted by Titan, requested wherever the size is incidental.
TITAN_SIZE = smallest_image_size(TITAN_V2)


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


class TestAmazonTitanEditing:
    """Titan ``taskType`` dispatch and provider extras on the edits route.

    Ref: stdapi/models/image/amazon_titan_image_generator.py:_ImageGenerationJob._edit_image
         stdapi/routes/_images_common.py:build_images_response
    """

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_edit_with_extra_parameters(
        self,
        openai_client: OpenAI,
        sample_image_file: bytes,
        sample_mask_file: bytes,
        model_id: str,
    ) -> None:
        """Provider extras are merged into the ``taskType``'s own parameter block.

        ``negativeText``, ``seed`` and the AWS ``quality`` bucket have no OpenAI
        equivalent; Bedrock rejects the invocation with a ``ValidationException``
        when they are not nested under ``inPaintingParams`` /
        ``imageGenerationConfig``. The echoed ``quality`` stays ``medium``
        because the edits route never forwards an OpenAI quality tier, so the
        AWS bucket passed as an extra does not leak into the response.

        Ref: stdapi/models/image/amazon_titan_image_generator.py:_ImageGenerationJob._set_extra_config
             stdapi/aws_bedrock.py:get_extra_model_parameters
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_mask_file,
            prompt="A blue circle in the center",
            model=model_id,
            size=TITAN_SIZE,
            n=1,
            response_format="b64_json",
            extra_body={
                "inPaintingParams": {"negativeText": "blurry, distorted, low quality"},
                "imageGenerationConfig": {"seed": 42, "quality": "premium"},
            },
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        _decoded_png(response.data[0].b64_json)
        assert response.data[0].url is None
        assert response.size == TITAN_SIZE
        assert response.output_format == "png"
        assert response.background == "opaque"
        assert response.quality == "medium"

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_ALL)
    def test_edit_b64_single(
        self,
        openai_client: OpenAI,
        sample_image_file: bytes,
        sample_mask_file: bytes,
        model_id: str,
    ) -> None:
        """A masked edit returns one base64 PNG, no URL, and a split token usage.

        Titan defaults every edit to the ``INPAINTING`` task type. The fixture
        mask is a plain black/white PNG with no alpha channel, so it reaches
        Bedrock unchanged. Both the source image and the mask count as input
        images, so at most two of the input tokens are attributed to images.

        Ref: stdapi/utils.py:alpha_mask_to_bw
             https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_mask_file,
            prompt="A green square",
            model=model_id,
            size=TITAN_SIZE,
            response_format="b64_json",
        )

        assert response.created > 0
        assert response.size == TITAN_SIZE
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
    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_edit_with_outpainting_task_type(
        self,
        openai_client: OpenAI,
        sample_image_file: bytes,
        sample_mask_file: bytes,
        model_id: str,
    ) -> None:
        """An explicit ``OUTPAINTING`` task type overrides the ``INPAINTING`` default.

        The fixture mask is already a pure black/white PNG, so it is forwarded
        verbatim; outpainting regenerates the white pixels where inpainting
        would have regenerated the black ones.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_mask_file,
            prompt="Extend the scene with a forest",
            model=model_id,
            size=TITAN_SIZE,
            n=1,
            response_format="b64_json",
            extra_body={"taskType": "OUTPAINTING"},
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        _decoded_png(response.data[0].b64_json)
        assert response.data[0].url is None
        assert response.size == TITAN_SIZE
        assert response.output_format == "png"

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_edit_with_outpainting_task_type_and_alpha_mask(
        self,
        openai_client: OpenAI,
        sample_image_file: bytes,
        sample_alpha_mask_file: bytes,
        model_id: str,
    ) -> None:
        """``OUTPAINTING`` accepts an OpenAI-style alpha-transparency mask.

        Titan only takes an alpha-less mask whose pixels are (0,0,0) or
        (255,255,255), so the RGBA mask must be converted — with the outpainting
        polarity (white = generate) rather than inpainting's (black = edit) — for
        the invocation to succeed. The resulting pixel values are asserted at
        unit level in ``tests/test_models_image.py``; here the accepted request
        and the returned image are the observable evidence.

        Ref: stdapi/utils.py:alpha_mask_to_bw
             https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_alpha_mask_file,
            prompt="Extend the scene with a forest",
            model=model_id,
            size=TITAN_SIZE,
            n=1,
            response_format="b64_json",
            extra_body={"taskType": "OUTPAINTING"},
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        _decoded_png(response.data[0].b64_json)
        assert response.data[0].url is None
        assert response.size == TITAN_SIZE
        assert response.output_format == "png"
        assert response.usage is not None
        assert response.usage.input_tokens_details.image_tokens <= 2, (
            "only the source image and the mask are input images"
        )

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_edit_with_background_removal_task_type(
        self, openai_client: OpenAI, sample_image_file: bytes, model_id: str
    ) -> None:
        """``BACKGROUND_REMOVAL`` needs no mask and ignores the prompt.

        The Bedrock request carries only ``backgroundRemovalParams.image``; a
        mask would be rejected by the gateway before the invocation. This task
        type exists on Titan V2 only.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
             stdapi/models/image/amazon_titan_image_generator.py:_ImageGenerationJob._get_request_background_removal
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="Remove background",
            model=model_id,
            size=TITAN_SIZE,
            n=1,
            response_format="b64_json",
            extra_body={"taskType": "BACKGROUND_REMOVAL"},
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        _decoded_png(response.data[0].b64_json)
        assert response.data[0].url is None
        assert response.size == TITAN_SIZE
        assert response.output_format == "png"

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_edit_inpainting_without_mask(
        self, openai_client: OpenAI, sample_image_file: bytes, model_id: str
    ) -> None:
        """``INPAINTING`` accepts a ``maskPrompt`` instead of an uploaded mask.

        Titan requires exactly one of ``maskPrompt`` or ``maskImage``, so the
        request is only valid because no mask file was uploaded.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="A blue square in the center",
            model=model_id,
            size=TITAN_SIZE,
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
        assert response.size == TITAN_SIZE
        assert response.output_format == "png"

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_edit_outpainting_without_mask(
        self, openai_client: OpenAI, sample_image_file: bytes, model_id: str
    ) -> None:
        """``OUTPAINTING`` accepts a ``maskPrompt`` instead of an uploaded mask.

        The prompt-derived mask replaces ``maskImage``, which is mutually
        exclusive with it.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="Extend with ocean view",
            model=model_id,
            size=TITAN_SIZE,
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
        assert response.size == TITAN_SIZE
        assert response.output_format == "png"

    @pytest.mark.parametrize("model_id", TITAN_SAMPLE)
    def test_edit_with_invalid_task_type(
        self,
        openai_client: OpenAI,
        sample_image_file: bytes,
        sample_mask_file: bytes,
        model_id: str,
    ) -> None:
        """An unknown ``taskType`` is rejected locally as an ``invalid_request_error``.

        The gateway dispatches on the ``taskType`` extra itself, so the request
        never reaches Bedrock and the message lists the edit task types Titan
        supports.

        Ref: stdapi/models/image/amazon_titan_image_generator.py:_ImageGenerationJob._edit_image
             stdapi/api_providers/openai.py:_format_error
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                mask=sample_mask_file,
                prompt="A test prompt",
                model=model_id,
                size=TITAN_SIZE,
                extra_body={"taskType": "INVALID_TASK_TYPE"},
            )

        assert exc_info.value.status_code == 400
        # The OpenAI client unwraps the envelope: body is already the error object.
        body = exc_info.value.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        message = body["message"]
        assert "taskType" in message
        assert "BACKGROUND_REMOVAL" in message, (
            f"expected the Titan edit task types, got: {message}"
        )
