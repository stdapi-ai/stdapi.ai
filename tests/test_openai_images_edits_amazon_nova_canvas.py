"""OpenAI-compatible tests for Amazon Nova Canvas image editing."""

import base64
from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError

from tests.conftest import OUTPUT_DIR, SAMPLES_DIR

if TYPE_CHECKING:
    from openai import OpenAI

NOVA_CANVAS_V1 = "amazon.nova-canvas-v1:0"

NOVA_CANVAS_ALL = (NOVA_CANVAS_V1,)
NOVA_CANVAS_SAMPLE = (NOVA_CANVAS_V1,)


class TestAmazonNovaCanvasEditing:
    """Model-specific tests for Amazon Nova Canvas image editing.

    Focus on Nova Canvas-specific parameters and features unique to this provider.
    """

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_edit_with_extra_parameters(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        sample_mask_file: bytes,
        model_id: str,
    ) -> None:
        """Test editing with Nova Canvas-specific negativeText parameter."""
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_mask_file,
            prompt="A blue circle in the center",
            model=model_id,
            size="512x512",
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
        assert response.data[0].b64_json is not None
        assert response.size == "512x512"  # type: ignore[comparison-overlap]

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_ALL)
    def test_edit_b64_single(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        sample_mask_file: bytes,
        model_id: str,
    ) -> None:
        """Test basic editing with base64 response format."""
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_mask_file,
            prompt="A green square",
            model=model_id,
            size="512x512",
            response_format="b64_json",
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
        assert response.usage.input_tokens_details.text_tokens >= 0
        assert response.usage.output_tokens == 1

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_ALL)
    def test_edit_b64_single_without_mask(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        model_id: str,
    ) -> None:
        """Test basic editing with base64 response format."""
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="A green square",
            model=model_id,
            size="512x512",
            response_format="b64_json",
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
        assert response.usage.input_tokens_details.text_tokens >= 0
        assert response.usage.output_tokens == 1

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_edit_with_outpainting_task_type(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        sample_mask_file: bytes,
        model_id: str,
    ) -> None:
        """Test editing with OUTPAINTING taskType."""
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_mask_file,
            prompt="Extend the scene with mountains in the background",
            model=model_id,
            size="512x512",
            n=1,
            response_format="b64_json",
            extra_body={"taskType": "OUTPAINTING"},
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.size == "512x512"  # type: ignore[comparison-overlap]

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_edit_with_background_removal_task_type(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        model_id: str,
    ) -> None:
        """Test editing with BACKGROUND_REMOVAL taskType (no mask supported)."""
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="Remove the background",
            model=model_id,
            size="512x512",
            n=1,
            response_format="b64_json",
            extra_body={"taskType": "BACKGROUND_REMOVAL"},
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.size == "512x512"  # type: ignore[comparison-overlap]

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    @pytest.mark.parametrize("mask_type", ["IMAGE", "PROMPT", "GARMENT"])
    def test_edit_with_virtual_try_on(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        chat_vision_model: str,
        model_id: str,
        mask_type: str,
    ) -> None:
        """Test editing with VIRTUAL_TRY_ON taskType using all maskType options."""
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

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
            size="1024x1024",
            n=1,
            response_format="b64_json",
            extra_body=extra_body,
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.size == "1024x1024"

        # Save the output image for manual inspection
        output_image_data = base64.b64decode(response.data[0].b64_json)
        output_path = OUTPUT_DIR / f"vto_result_{mask_type.lower()}.jpg"
        output_path.write_bytes(output_image_data)

        # Use VLM to validate the result
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_model,
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
                                "url": f"data:image/jpeg;base64,{response.data[0].b64_json}"
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
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        model_id: str,
    ) -> None:
        """Test INPAINTING without mask (mask is optional)."""
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="A red circle in the center",
            model=model_id,
            size="512x512",
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
        assert response.data[0].b64_json is not None
        assert response.size == "512x512"  # type: ignore[comparison-overlap]

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_edit_outpainting_without_mask(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        model_id: str,
    ) -> None:
        """Test OUTPAINTING without mask (mask is optional)."""
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="Extend with mountains",
            model=model_id,
            size="512x512",
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
        assert response.data[0].b64_json is not None
        assert response.size == "512x512"  # type: ignore[comparison-overlap]

    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_edit_with_invalid_task_type(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        sample_mask_file: bytes,
        model_id: str,
    ) -> None:
        """Test that invalid taskType raises BadRequestError."""
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                mask=sample_mask_file,
                prompt="A test prompt",
                model=model_id,
                size="512x512",
                extra_body={"taskType": "INVALID_TASK_TYPE"},
            )

        assert (
            "taskType" in str(exc_info.value).lower()
            or "INPAINTING" in str(exc_info.value)
            or "OUTPAINTING" in str(exc_info.value)
            or "BACKGROUND_REMOVAL" in str(exc_info.value)
        )

    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_edit_with_invalid_mask_type(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        model_id: str,
    ) -> None:
        """Test that invalid virtualTryOnParams.maskType raises BadRequestError."""
        if use_official_api:
            pytest.skip(
                "Amazon Nova Canvas is not available on the official OpenAI API"
            )

        source_image = (SAMPLES_DIR / "vto_upper_body_source.jpg").read_bytes()
        reference_image = (SAMPLES_DIR / "vto_upper_body_reference.jpg").read_bytes()

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=source_image,
                mask=reference_image,
                prompt="test",
                model=model_id,
                size="1024x1024",
                extra_body={
                    "taskType": "VIRTUAL_TRY_ON",
                    "virtualTryOnParams": {"maskType": "INVALID_MASK_TYPE"},
                },
            )

        assert "maskType" in str(exc_info.value) or "PROMPT" in str(exc_info.value)
