"""Coverage of /v1/images/edits against the Stability AI models on Amazon Bedrock.

Every model here reaches the endpoint through the same edit route, but each one
consumes a different subset of the OpenAI parameters: some ignore ``prompt``,
some require a mask, some repurpose ``mask`` as a second input image, and some
require a provider extra passed through ``extra_body``. The assertions pin the
gateway-side contract (payload format, response metadata, parameter validation);
image content is left to the vision judge.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/stable-image-services.html
     https://stdapi.ai/api_openai_images_edits/
     stdapi/models/image/_stability.py:StabilityImageGenerationJobBase
"""

import base64
import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError

if TYPE_CHECKING:
    from openai import OpenAI

#: Stable Diffusion 3.5 Large, used here in its image-to-image edit mode
STABILITY_SD3_5 = "stability.sd3-5-large-v1:0"

#: Fast 4x upscaler: takes no prompt and no mask
STABILITY_FAST_UPSCALE = "stability.stable-fast-upscale-v1:0"
#: Prompt-guided creative upscaler
STABILITY_CREATIVE_UPSCALE = "stability.stable-creative-upscale-v1:0"
#: Detail-preserving upscaler
STABILITY_CONSERVATIVE_UPSCALE = "stability.stable-conservative-upscale-v1:0"

#: Inpainting model: the mask is optional and derived from alpha when omitted
STABILITY_INPAINT = "stability.stable-image-inpaint-v1:0"
#: Outpainting model: extends the image beyond its borders, rejects masks
STABILITY_OUTPAINT = "stability.stable-outpaint-v1:0"
#: Recolors the region selected by the required ``select_prompt`` extra
STABILITY_SEARCH_RECOLOR = "stability.stable-image-search-recolor-v1:0"
#: Replaces the object named by the required ``search_prompt`` extra
STABILITY_SEARCH_REPLACE = "stability.stable-image-search-replace-v1:0"
#: Removes the masked object; requires a mask and ignores the prompt
STABILITY_ERASE = "stability.stable-image-erase-object-v1:0"
#: Automatic background removal; rejects masks and ignores the prompt
STABILITY_REMOVE_BG = "stability.stable-image-remove-background-v1:0"

#: Sketch-guided generation
STABILITY_CONTROL_SKETCH = "stability.stable-image-control-sketch-v1:0"
#: Structure-preserving generation
STABILITY_CONTROL_STRUCTURE = "stability.stable-image-control-structure-v1:0"

#: Applies the style of the input image to the prompt
STABILITY_STYLE_GUIDE = "stability.stable-image-style-guide-v1:0"
#: Style transfer: needs a second image as ``mask`` or as the ``style_image`` extra
STABILITY_STYLE_TRANSFER = "stability.stable-style-transfer-v1:0"

#: Models exercised by the generic image-to-image edit test
STABILITY_ALL = (STABILITY_SD3_5,)

#: PNG file signature, the format Stability returns unless another is requested
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
#: JPEG start-of-image marker
_JPEG_MAGIC = b"\xff\xd8\xff"
#: Shape of the ``size`` field built by ``build_images_response`` ("WIDTHxHEIGHT")
_SIZE_PATTERN = re.compile(r"^\d+x\d+$")


class TestStabilityEditing:
    """Image-to-image editing on Stable Diffusion 3.5 Large.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-3-5-large.html
         stdapi/models/image/stability_stable_diffusion.py:TextToImageJob
    """

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", STABILITY_ALL)
    def test_edit_b64_single(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file: bytes,
        model_id: str,
    ) -> None:
        """An edit request without a mask runs as image-to-image and returns one PNG.

        The gateway sends ``mode=image-to-image`` with a default ``strength`` of
        0.35, so the source image is required but no mask is. Usage counts the
        single source image as an input image and ``total_tokens`` is the sum of
        the two counters.

        Ref: stdapi/routes/_images_common.py:build_images_response
        """
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="Transform into a vibrant painting",
            model=model_id,
            response_format="b64_json",
        )

        assert response.created > 0
        assert response.size is not None
        assert _SIZE_PATTERN.match(response.size), response.size
        assert response.output_format == "png"

        data = response.data
        assert data is not None
        assert len(data) == 1
        b64_json = data[0].b64_json
        assert b64_json is not None
        assert data[0].url is None
        assert base64.b64decode(b64_json).startswith(_PNG_MAGIC)

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.input_tokens_details.text_tokens >= 0
        assert response.usage.input_tokens_details.image_tokens > 0
        assert response.usage.total_tokens > 0
        assert (
            response.usage.total_tokens
            == response.usage.input_tokens + response.usage.output_tokens
        )
        assert (
            response.usage.input_tokens_details.image_tokens
            + response.usage.input_tokens_details.text_tokens
            == response.usage.input_tokens
        )


class TestStabilityUpscaleModels:
    """Upscaling models reached through /v1/images/edits.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/stable-image-services.html
         stdapi/models/image/stability_stable_fast_upscale.py:_FastUpscaleJob
         stdapi/models/image/stability_stable_image_edit.py:_SimpleEditJob
    """

    @pytest.mark.expensive
    def test_fast_upscale(
        self, openai_client: OpenAI, use_official_api: bool, sample_image_file: bytes
    ) -> None:
        """Fast upscale ignores the prompt and honors ``output_format=jpeg``.

        The backend request carries only the image and the output format, so the
        prompt the OpenAI client insists on is dropped; ``jpeg`` is a native
        Stability output format and is therefore returned without a re-encode,
        and the response metadata reports it.

        Ref: stdapi/models/image/_stability.py:StabilityImageGenerationJobBase._finalize_request
        """
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="ignored",  # doesn't use prompt, but Openai Python library requires it
            model=STABILITY_FAST_UPSCALE,
            response_format="b64_json",
            output_format="jpeg",
        )

        assert response.created > 0
        assert response.output_format == "jpeg"

        data = response.data
        assert data is not None
        assert len(data) == 1
        b64_json = data[0].b64_json
        assert b64_json is not None
        assert base64.b64decode(b64_json).startswith(_JPEG_MAGIC)

    @pytest.mark.expensive
    @pytest.mark.parametrize(
        "model_id", [STABILITY_CREATIVE_UPSCALE, STABILITY_CONSERVATIVE_UPSCALE]
    )
    def test_creative_upscale(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        chat_vision_judge_model: str,
        model_id: str,
    ) -> None:
        """Prompt-guided upscaling returns a JPEG that still depicts the source scene.

        The prompt guides the added detail, so the AWS documentation sample image
        of Big Ben is used and the result is checked by a vision judge; the
        deterministic part of the contract is the requested ``jpeg`` output.

        Ref: stdapi/models/image/stability_stable_image_edit.py:_SimpleEditJob
        """
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        # Load the upscale example image from AWS documentation
        test_dir = Path(__file__).parent
        samples_dir = test_dir / "samples"
        output_dir = test_dir / "output"
        output_dir.mkdir(exist_ok=True)

        input_image = (samples_dir / "stability_upscale_input.jpg").read_bytes()

        try:
            response = openai_client.images.edit(
                image=input_image,
                prompt="This dreamlike digital art captures a vibrant, kaleidoscopic Big Ben in London",
                model=model_id,
                response_format="b64_json",
                output_format="jpeg",
            )
        except Exception as exc:
            if "unexpected error" in str(exc):
                pytest.xfail(str(exc))
            raise

        assert response.created > 0
        assert response.output_format == "jpeg"

        data = response.data
        assert data is not None
        assert len(data) == 1
        b64_json = data[0].b64_json
        assert b64_json is not None
        output_data = base64.b64decode(b64_json)
        assert output_data.startswith(_JPEG_MAGIC)

        # Save output for manual inspection
        model_name = "creative" if "creative" in model_id else "conservative"
        (output_dir / f"stability_{model_name}_upscale_result.jpg").write_bytes(
            output_data
        )

        # VLM validation
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_judge_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this upscaled image of Big Ben in London. "
                                "The image should show Big Ben with enhanced details and higher resolution. "
                                "Does the image show Big Ben with good quality and detail? "
                                "If yes, respond with only 'YES'. "
                                "If no, respond with 'NO' followed by a brief explanation of what failed."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_json}"},
                        },
                    ],
                }
            ],
        )

        vlm_response = validation_response.choices[0].message.content
        assert vlm_response is not None
        assert "YES" in vlm_response.upper(), (
            f"VLM validation failed for {model_name} upscale. Response: {vlm_response}"
        )


class TestStabilityEditModels:
    """Stability models that edit a region of the source image.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/stable-image-services.html
         https://stdapi.ai/api_openai_images_edits/
    """

    @pytest.mark.expensive
    def test_search_recolor(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        chat_vision_judge_model: str,
    ) -> None:
        """Search-and-recolor recolors the region named by the ``select_prompt`` extra.

        ``select_prompt`` has no OpenAI counterpart, so it travels as a provider
        extra: ``prompt`` describes the target colour while ``select_prompt``
        selects what to recolor.

        Ref: stdapi/models/image/stability_search_recolor.py:_SearchRecolorJob
        """
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        # Load the search and recolor example image from AWS documentation
        test_dir = Path(__file__).parent
        samples_dir = test_dir / "samples"
        output_dir = test_dir / "output"
        output_dir.mkdir(exist_ok=True)

        input_image = (samples_dir / "stability_search_recolor_input.jpg").read_bytes()

        response = openai_client.images.edit(
            image=input_image,
            prompt="pink jacket",
            model=STABILITY_SEARCH_RECOLOR,
            response_format="b64_json",
            extra_body={"select_prompt": "jacket"},
        )

        assert response.created > 0
        assert response.output_format == "png"

        data = response.data
        assert data is not None
        assert len(data) == 1
        b64_json = data[0].b64_json
        assert b64_json is not None
        output_data = base64.b64decode(b64_json)
        assert output_data.startswith(_PNG_MAGIC)

        # Save output for manual inspection
        (output_dir / "stability_search_recolor_result.jpg").write_bytes(output_data)

        # VLM validation
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_judge_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image of a person wearing a jacket. "
                                "The jacket should have been recolored to pink. "
                                "Does the image show a person wearing a pink jacket? "
                                "If yes, respond with only 'YES'. "
                                "If no, respond with 'NO' followed by a brief explanation of what failed."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_json}"},
                        },
                    ],
                }
            ],
        )

        vlm_response = validation_response.choices[0].message.content
        assert vlm_response is not None
        assert "YES" in vlm_response.upper(), (
            f"VLM validation failed for search and recolor. Response: {vlm_response}"
        )

    def test_search_recolor_missing_select_prompt(
        self, openai_client: OpenAI, use_official_api: bool, sample_image_file: bytes
    ) -> None:
        """Search-and-recolor without ``select_prompt`` is a 400 naming the parameter.

        Ref: stdapi/models/image/stability_search_recolor.py:_SearchRecolorJob
        """
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                prompt="pink jacket",
                model=STABILITY_SEARCH_RECOLOR,
                response_format="b64_json",
            )

        assert exc_info.value.type == "invalid_request_error"
        assert '"select_prompt" parameter is required for this model.' in str(
            exc_info.value
        )

    @pytest.mark.expensive
    def test_search_replace(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        chat_vision_judge_model: str,
    ) -> None:
        """Search-and-replace swaps the object named by the ``search_prompt`` extra.

        ``prompt`` describes the replacement while the ``search_prompt`` provider
        extra selects the object to replace.

        Ref: stdapi/models/image/stability_search_replace.py:_SearchReplaceJob
        """
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        # Load the search and replace example image from AWS documentation
        test_dir = Path(__file__).parent
        samples_dir = test_dir / "samples"
        output_dir = test_dir / "output"
        output_dir.mkdir(exist_ok=True)

        input_image = (samples_dir / "stability_search_replace_input.jpg").read_bytes()

        response = openai_client.images.edit(
            image=input_image,
            prompt="jacket",
            model=STABILITY_SEARCH_REPLACE,
            response_format="b64_json",
            extra_body={"search_prompt": "sweater"},
        )

        assert response.created > 0
        assert response.output_format == "png"

        data = response.data
        assert data is not None
        assert len(data) == 1
        b64_json = data[0].b64_json
        assert b64_json is not None
        output_data = base64.b64decode(b64_json)
        assert output_data.startswith(_PNG_MAGIC)

        # Save output for manual inspection
        (output_dir / "stability_search_replace_result.jpg").write_bytes(output_data)

        # VLM validation
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_judge_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image of a female model. "
                                "The original sweater should have been replaced with a jacket. "
                                "Does the image show a person wearing a jacket instead of a sweater? "
                                "If yes, respond with only 'YES'. "
                                "If no, respond with 'NO' followed by a brief explanation of what failed."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_json}"},
                        },
                    ],
                }
            ],
        )

        vlm_response = validation_response.choices[0].message.content
        assert vlm_response is not None
        assert "YES" in vlm_response.upper(), (
            f"VLM validation failed for search and replace. Response: {vlm_response}"
        )

    def test_search_replace_missing_search_prompt(
        self, openai_client: OpenAI, use_official_api: bool, sample_image_file: bytes
    ) -> None:
        """Search-and-replace without ``search_prompt`` is a 400 naming the parameter.

        Ref: stdapi/models/image/stability_search_replace.py:_SearchReplaceJob
        """
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                prompt="jacket",
                model=STABILITY_SEARCH_REPLACE,
                response_format="b64_json",
            )

        assert exc_info.value.type == "invalid_request_error"
        assert '"search_prompt" parameter is required for this model.' in str(
            exc_info.value
        )

    @pytest.mark.expensive
    def test_inpaint(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        chat_vision_judge_model: str,
    ) -> None:
        """Inpainting regenerates the masked area from the prompt.

        The AWS sample mask is a plain black/white PNG with no alpha channel, so
        the gateway forwards it untouched; Stability reads white pixels as
        maximum inpaint strength, the opposite of the Titan/Nova convention where
        black marks the area to edit.

        Ref: stdapi/models/image/stability_stable_image_inpaint.py:_InpaintJob
             stdapi/utils.py:alpha_mask_to_bw
        """
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        # Load the inpaint example images from AWS documentation
        test_dir = Path(__file__).parent
        samples_dir = test_dir / "samples"
        output_dir = test_dir / "output"
        output_dir.mkdir(exist_ok=True)

        input_image = (samples_dir / "stability_inpaint_input.jpg").read_bytes()
        mask_image = (samples_dir / "stability_inpaint_mask.png").read_bytes()

        response = openai_client.images.edit(
            image=input_image,
            mask=mask_image,
            prompt="artificer of time and space",
            model=STABILITY_INPAINT,
            response_format="b64_json",
        )

        assert response.created > 0
        assert response.output_format == "png"

        data = response.data
        assert data is not None
        assert len(data) == 1
        b64_json = data[0].b64_json
        assert b64_json is not None
        output_data = base64.b64decode(b64_json)
        assert output_data.startswith(_PNG_MAGIC)

        # Save output for manual inspection
        (output_dir / "stability_inpaint_result.jpg").write_bytes(output_data)

        # VLM validation
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_judge_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image which is the result of an inpainting operation on a man in an urban setting. "
                                "The masked area should now contain a mystical or sci-fi themed element ('artificer of time and space'). "
                                "Does the image show a modified urban scene with the inpainted area seamlessly integrated? "
                                "If yes, respond with only 'YES'. "
                                "If no, respond with 'NO' followed by a brief explanation of what failed."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_json}"},
                        },
                    ],
                }
            ],
        )

        vlm_response = validation_response.choices[0].message.content
        assert vlm_response is not None
        assert "YES" in vlm_response.upper(), (
            f"VLM validation failed for inpaint. Response: {vlm_response}"
        )

    @pytest.mark.expensive
    def test_inpaint_without_mask(
        self, openai_client: OpenAI, use_official_api: bool, sample_image_file: bytes
    ) -> None:
        """Inpainting accepts a request with no mask at all.

        The ``mask`` key is only added to the backend request when one is
        supplied; Stability then derives the mask from the input image's alpha
        channel, so an opaque source image is edited as a whole.

        Ref: stdapi/models/image/stability_stable_image_inpaint.py:_InpaintJob
        """
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="add magical elements to the scene",
            model=STABILITY_INPAINT,
            response_format="b64_json",
        )

        assert response.created > 0
        assert response.output_format == "png"

        data = response.data
        assert data is not None
        assert len(data) == 1
        b64_json = data[0].b64_json
        assert b64_json is not None
        assert base64.b64decode(b64_json).startswith(_PNG_MAGIC)

    @pytest.mark.expensive
    def test_erase(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        chat_vision_judge_model: str,
    ) -> None:
        """Object erasure consumes the mask and ignores the prompt.

        ``EraseRequest`` carries only the image and the mask, so the prompt the
        OpenAI client requires is dropped before the backend call.

        Ref: stdapi/models/image/stability_stable_image_erase_object.py:_EraseJob
        """
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        # Load the erase example images from AWS documentation
        test_dir = Path(__file__).parent
        samples_dir = test_dir / "samples"
        output_dir = test_dir / "output"
        output_dir.mkdir(exist_ok=True)

        input_image = (samples_dir / "stability_erase_input.jpg").read_bytes()
        mask_image = (samples_dir / "stability_erase_mask.png").read_bytes()

        response = openai_client.images.edit(
            image=input_image,
            mask=mask_image,
            prompt="ignored",  # doesn't use prompt, but Openai Python library requires it
            model=STABILITY_ERASE,
            response_format="b64_json",
        )

        assert response.created > 0
        assert response.output_format == "png"

        data = response.data
        assert data is not None
        assert len(data) == 1
        b64_json = data[0].b64_json
        assert b64_json is not None
        output_data = base64.b64decode(b64_json)
        assert output_data.startswith(_PNG_MAGIC)

        # Save output for manual inspection
        (output_dir / "stability_erase_result.jpg").write_bytes(output_data)

    @pytest.mark.expensive
    def test_remove_background(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        chat_vision_judge_model: str,
    ) -> None:
        """Background removal isolates the subject from a prompt-free request.

        Ref: stdapi/models/image/stability_stable_image_remove_background.py:_RemoveBackgroundJob
        """
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        # Load the remove background example image from AWS documentation
        test_dir = Path(__file__).parent
        samples_dir = test_dir / "samples"
        output_dir = test_dir / "output"
        output_dir.mkdir(exist_ok=True)

        input_image = (samples_dir / "stability_remove_bg_input.jpg").read_bytes()

        response = openai_client.images.edit(
            image=input_image,
            prompt="ignored",  # doesn't use prompt, but Openai Python library requires it
            model=STABILITY_REMOVE_BG,
            response_format="b64_json",
        )

        assert response.created > 0
        assert response.output_format == "png"

        data = response.data
        assert data is not None
        assert len(data) == 1
        b64_json = data[0].b64_json
        assert b64_json is not None
        output_data = base64.b64decode(b64_json)
        assert output_data.startswith(_PNG_MAGIC)

        # Save output for manual inspection
        (output_dir / "stability_remove_bg_result.png").write_bytes(output_data)

        # VLM validation
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_judge_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image which should show a person with the background removed. "
                                "The background should be transparent or removed, leaving only the subject (person). "
                                "Does the image show a person isolated from the background? "
                                "If yes, respond with only 'YES'. "
                                "If no, respond with 'NO' followed by a brief explanation of what failed."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_json}"},
                        },
                    ],
                }
            ],
        )

        vlm_response = validation_response.choices[0].message.content
        assert vlm_response is not None
        assert "YES" in vlm_response.upper(), (
            f"VLM validation failed for remove background. Response: {vlm_response}"
        )


class TestStabilityControlModels:
    """Control models: the input image constrains the generated structure.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/stable-image-services.html
         stdapi/models/image/stability_stable_image_edit.py:_SimpleEditJob
    """

    @pytest.mark.expensive
    def test_control_sketch(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        chat_vision_judge_model: str,
    ) -> None:
        """Control-sketch turns a sketch into a rendered scene described by the prompt."""
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        # Load the control sketch example image from AWS documentation
        test_dir = Path(__file__).parent
        samples_dir = test_dir / "samples"
        output_dir = test_dir / "output"
        output_dir.mkdir(exist_ok=True)

        input_image = (samples_dir / "stability_control_sketch_input.jpg").read_bytes()

        response = openai_client.images.edit(
            image=input_image,
            prompt="a house with background of mountains and river flowing nearby",
            model=STABILITY_CONTROL_SKETCH,
            response_format="b64_json",
        )

        assert response.created > 0
        assert response.output_format == "png"

        data = response.data
        assert data is not None
        assert len(data) == 1
        b64_json = data[0].b64_json
        assert b64_json is not None
        output_data = base64.b64decode(b64_json)
        assert output_data.startswith(_PNG_MAGIC)

        # Save output for manual inspection
        (output_dir / "stability_control_sketch_result.jpg").write_bytes(output_data)

        # VLM validation
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_judge_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image which should be a detailed landscape scene based on a sketch. "
                                "The image should show a house with mountains and a river in the background. "
                                "Does the image show a landscape with a house, mountains, and a river? "
                                "If yes, respond with only 'YES'. "
                                "If no, respond with 'NO' followed by a brief explanation of what failed."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_json}"},
                        },
                    ],
                }
            ],
        )

        vlm_response = validation_response.choices[0].message.content
        assert vlm_response is not None
        assert "YES" in vlm_response.upper(), (
            f"VLM validation failed for control sketch. Response: {vlm_response}"
        )

    @pytest.mark.expensive
    def test_control_structure(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        chat_vision_judge_model: str,
    ) -> None:
        """Control-structure keeps the source composition while restyling it."""
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        # Load the control structure example image from AWS documentation
        test_dir = Path(__file__).parent
        samples_dir = test_dir / "samples"
        output_dir = test_dir / "output"
        output_dir.mkdir(exist_ok=True)

        input_image = (
            samples_dir / "stability_control_structure_input.jpg"
        ).read_bytes()

        response = openai_client.images.edit(
            image=input_image,
            prompt="surreal structure with motion generated sparks lighting the scene",
            model=STABILITY_CONTROL_STRUCTURE,
            response_format="b64_json",
        )

        assert response.created > 0
        assert response.output_format == "png"

        data = response.data
        assert data is not None
        assert len(data) == 1
        b64_json = data[0].b64_json
        assert b64_json is not None
        output_data = base64.b64decode(b64_json)
        assert output_data.startswith(_PNG_MAGIC)

        # Save output for manual inspection
        (output_dir / "stability_control_structure_result.jpg").write_bytes(output_data)

        # VLM validation
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_judge_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image which should show a surreal structure with sparks and lighting effects. "
                                "The original structural composition should be maintained but with added motion-generated sparks. "
                                "Does the image show a structure with lighting/spark effects? "
                                "If yes, respond with only 'YES'. "
                                "If no, respond with 'NO' followed by a brief explanation of what failed."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_json}"},
                        },
                    ],
                }
            ],
        )

        vlm_response = validation_response.choices[0].message.content
        assert vlm_response is not None
        assert "YES" in vlm_response.upper(), (
            f"VLM validation failed for control structure. Response: {vlm_response}"
        )


class TestStabilityStyleModels:
    """Style models, including the two ways of passing a second input image.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/stable-image-services.html
         stdapi/models/image/stability_stable_style_transfer.py:_StyleTransferJob
    """

    @pytest.mark.expensive
    def test_style_guide(
        self, openai_client: OpenAI, use_official_api: bool, sample_image_file: bytes
    ) -> None:
        """Style-guide takes a single image as the style reference for the prompt.

        Ref: stdapi/models/image/stability_stable_image_edit.py:_SimpleEditJob
        """
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="Generate in the style of the reference",
            model=STABILITY_STYLE_GUIDE,
            response_format="b64_json",
        )

        assert response.created > 0
        assert response.output_format == "png"

        data = response.data
        assert data is not None
        assert len(data) == 1
        b64_json = data[0].b64_json
        assert b64_json is not None
        assert base64.b64decode(b64_json).startswith(_PNG_MAGIC)

    @pytest.mark.expensive
    def test_style_transfer_with_mask_as_style_image(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        chat_vision_judge_model: str,
    ) -> None:
        """Style transfer reads the OpenAI ``mask`` upload as its style image.

        Style transfer needs two images but the endpoint only has one binary
        image field, so the mask slot is repurposed as ``style_image`` — it is
        not used as a mask at all.

        Ref: stdapi/models/image/stability_stable_style_transfer.py:_StyleTransferJob
        """
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        # Load the style transfer example images from AWS documentation
        test_dir = Path(__file__).parent
        samples_dir = test_dir / "samples"
        output_dir = test_dir / "output"
        output_dir.mkdir(exist_ok=True)

        input_image = (samples_dir / "stability_style_transfer_input.jpg").read_bytes()
        style_image = (samples_dir / "stability_style_transfer_style.jpg").read_bytes()

        response = openai_client.images.edit(
            image=input_image,
            mask=style_image,  # mask parameter is used for style image
            prompt="statue",
            model=STABILITY_STYLE_TRANSFER,
            response_format="b64_json",
        )

        assert response.created > 0
        assert response.output_format == "png"

        data = response.data
        assert data is not None
        assert len(data) == 1
        b64_json = data[0].b64_json
        assert b64_json is not None
        output_data = base64.b64decode(b64_json)
        assert output_data.startswith(_PNG_MAGIC)

        # Save output for manual inspection
        (output_dir / "stability_style_transfer_result.jpg").write_bytes(output_data)

        # VLM validation
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_judge_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image which should be a statue with applied style transfer. "
                                "The statue content should remain recognizable but styled with bright, colorful lighting effects. "
                                "Does the image show a statue with artistic style effects applied? "
                                "If yes, respond with only 'YES'. "
                                "If no, respond with 'NO' followed by a brief explanation of what failed."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_json}"},
                        },
                    ],
                }
            ],
        )

        vlm_response = validation_response.choices[0].message.content
        assert vlm_response is not None
        assert "YES" in vlm_response.upper(), (
            f"VLM validation failed for style transfer. Response: {vlm_response}"
        )

    @pytest.mark.expensive
    def test_style_transfer_with_style_image_parameter(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        chat_vision_judge_model: str,
    ) -> None:
        """A base64 ``style_image`` extra replaces the mask upload for style transfer.

        This is the JSON-friendly form of the same input: the provider extra is
        forwarded as ``style_image``, so no mask upload is needed.

        Ref: stdapi/models/image/stability_stable_style_transfer.py:_StyleTransferJob
        """
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        # Load the style transfer example images from AWS documentation
        test_dir = Path(__file__).parent
        samples_dir = test_dir / "samples"
        output_dir = test_dir / "output"
        output_dir.mkdir(exist_ok=True)

        input_image = (samples_dir / "stability_style_transfer_input.jpg").read_bytes()
        style_image_bytes = (
            samples_dir / "stability_style_transfer_style.jpg"
        ).read_bytes()
        style_image_b64 = base64.b64encode(style_image_bytes).decode("utf-8")

        response = openai_client.images.edit(
            image=input_image,
            prompt="statue",
            model=STABILITY_STYLE_TRANSFER,
            response_format="b64_json",
            extra_body={"style_image": style_image_b64},
        )

        assert response.created > 0
        assert response.output_format == "png"

        data = response.data
        assert data is not None
        assert len(data) == 1
        b64_json = data[0].b64_json
        assert b64_json is not None
        output_data = base64.b64decode(b64_json)
        assert output_data.startswith(_PNG_MAGIC)

        # Save output for manual inspection
        (output_dir / "stability_style_transfer_extra_body_result.jpg").write_bytes(
            output_data
        )

        # VLM validation
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_judge_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Analyze this image which should be a statue with applied style transfer. "
                                "The statue content should remain recognizable but styled with bright, colorful lighting effects. "
                                "Does the image show a statue with artistic style effects applied? "
                                "If yes, respond with only 'YES'. "
                                "If no, respond with 'NO' followed by a brief explanation of what failed."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64_json}"},
                        },
                    ],
                }
            ],
        )

        vlm_response = validation_response.choices[0].message.content
        assert vlm_response is not None
        assert "YES" in vlm_response.upper(), (
            f"VLM validation failed for style transfer with style_image parameter. Response: {vlm_response}"
        )

    def test_style_transfer_missing_style_image(
        self, openai_client: OpenAI, use_official_api: bool, sample_image_file: bytes
    ) -> None:
        """Style transfer with neither ``mask`` nor ``style_image`` is a 400.

        The error names both spellings of the missing input, because either one
        satisfies the model.

        Ref: stdapi/models/image/stability_stable_style_transfer.py:_StyleTransferJob
        """
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                prompt="statue",
                model=STABILITY_STYLE_TRANSFER,
                response_format="b64_json",
            )

        assert exc_info.value.type == "invalid_request_error"
        assert '"mask" parameter is required by this model' in str(exc_info.value)
        assert "style_image" in str(exc_info.value)
