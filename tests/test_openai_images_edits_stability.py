"""OpenAI-compatible tests for Stability AI image editing."""

import base64
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError

if TYPE_CHECKING:
    from openai import OpenAI

# Text-to-Image / Image-to-Image Models
STABILITY_SD3_5 = "stability.sd3-5-large-v1:0"

# Upscale Models
STABILITY_FAST_UPSCALE = "stability.stable-fast-upscale-v1:0"
STABILITY_CREATIVE_UPSCALE = "stability.stable-creative-upscale-v1:0"
STABILITY_CONSERVATIVE_UPSCALE = "stability.stable-conservative-upscale-v1:0"

# Edit Models
STABILITY_INPAINT = "stability.stable-image-inpaint-v1:0"
STABILITY_OUTPAINT = "stability.stable-outpaint-v1:0"
STABILITY_SEARCH_RECOLOR = "stability.stable-image-search-recolor-v1:0"
STABILITY_SEARCH_REPLACE = "stability.stable-image-search-replace-v1:0"
STABILITY_ERASE = "stability.stable-image-erase-object-v1:0"
STABILITY_REMOVE_BG = "stability.stable-image-remove-background-v1:0"

# Control Models
STABILITY_CONTROL_SKETCH = "stability.stable-image-control-sketch-v1:0"
STABILITY_CONTROL_STRUCTURE = "stability.stable-image-control-structure-v1:0"

# Style Models
STABILITY_STYLE_GUIDE = "stability.stable-image-style-guide-v1:0"
STABILITY_STYLE_TRANSFER = "stability.stable-style-transfer-v1:0"

STABILITY_ALL = (STABILITY_SD3_5,)


class TestStabilityEditing:
    """Model-specific tests for Stability AI image editing.

    Focus on Stability-specific parameters like strength and negative_prompt.
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
        """Test basic image-to-image editing with base64 response format."""
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
        assert len(response.data) == 1  # type: ignore[arg-type]
        img = response.data[0]  # type: ignore[index]
        assert img.b64_json is not None
        assert img.url is None

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.input_tokens_details.text_tokens >= 0
        assert response.usage.input_tokens_details.image_tokens > 0
        assert response.usage.total_tokens > 0


class TestStabilityUpscaleModels:
    """Tests for Stability AI upscale models."""

    @pytest.mark.expensive
    def test_fast_upscale(
        self, openai_client: OpenAI, use_official_api: bool, sample_image_file: bytes
    ) -> None:
        """Test fast upscale model (no prompt needed)."""
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
        assert len(response.data) == 1  # type: ignore[arg-type]
        assert response.data[0].b64_json is not None  # type: ignore[index]

    @pytest.mark.expensive
    @pytest.mark.parametrize(
        "model_id", [STABILITY_CREATIVE_UPSCALE, STABILITY_CONSERVATIVE_UPSCALE]
    )
    def test_creative_upscale(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        chat_vision_model: str,
        model_id: str,
    ) -> None:
        """Test creative/conservative upscale models using AWS documentation example."""
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
        assert len(response.data) == 1  # type: ignore[arg-type]
        assert response.data[0].b64_json is not None  # type: ignore[index]

        # Save output for manual inspection
        model_name = "creative" if "creative" in model_id else "conservative"
        output_data = base64.b64decode(response.data[0].b64_json)  # type: ignore[index]
        (output_dir / f"stability_{model_name}_upscale_result.jpg").write_bytes(
            output_data
        )

        # VLM validation
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=[
                {  # type: ignore[misc,list-item]
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
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{response.data[0].b64_json}"  # type: ignore[index]
                            },
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
    """Tests for Stability AI specialized edit models."""

    @pytest.mark.expensive
    def test_search_recolor(
        self, openai_client: OpenAI, use_official_api: bool, chat_vision_model: str
    ) -> None:
        """Test search and recolor model using AWS documentation example."""
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
        assert len(response.data) == 1  # type: ignore[arg-type]
        assert response.data[0].b64_json is not None  # type: ignore[index]

        # Save output for manual inspection
        output_data = base64.b64decode(response.data[0].b64_json)  # type: ignore[index]
        (output_dir / "stability_search_recolor_result.jpg").write_bytes(output_data)

        # VLM validation
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=[
                {  # type: ignore[misc,list-item]
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
                            "image_url": {
                                "url": f"data:image/png;base64,{response.data[0].b64_json}"  # type: ignore[index]
                            },
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
        """Test that search-recolor without select_prompt raises BadRequestError."""
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                prompt="pink jacket",
                model=STABILITY_SEARCH_RECOLOR,
                response_format="b64_json",
            )

        assert "select_prompt" in str(exc_info.value)

    @pytest.mark.expensive
    def test_search_replace(
        self, openai_client: OpenAI, use_official_api: bool, chat_vision_model: str
    ) -> None:
        """Test search and replace model using AWS documentation example."""
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
        assert len(response.data) == 1  # type: ignore[arg-type]
        assert response.data[0].b64_json is not None  # type: ignore[index]

        # Save output for manual inspection
        output_data = base64.b64decode(response.data[0].b64_json)  # type: ignore[index]
        (output_dir / "stability_search_replace_result.jpg").write_bytes(output_data)

        # VLM validation
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=[
                {  # type: ignore[misc,list-item]
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
                            "image_url": {
                                "url": f"data:image/png;base64,{response.data[0].b64_json}"  # type: ignore[index]
                            },
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
        """Test that search-replace without search_prompt raises BadRequestError."""
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                prompt="jacket",
                model=STABILITY_SEARCH_REPLACE,
                response_format="b64_json",
            )

        assert "search_prompt" in str(exc_info.value)

    @pytest.mark.expensive
    def test_inpaint(
        self, openai_client: OpenAI, use_official_api: bool, chat_vision_model: str
    ) -> None:
        """Test inpaint model with mask using AWS documentation example."""
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
        assert len(response.data) == 1  # type: ignore[arg-type]
        assert response.data[0].b64_json is not None  # type: ignore[index]

        # Save output for manual inspection
        output_data = base64.b64decode(response.data[0].b64_json)  # type: ignore[index]
        (output_dir / "stability_inpaint_result.jpg").write_bytes(output_data)

        # VLM validation
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=[
                {  # type: ignore[misc,list-item]
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
                            "image_url": {
                                "url": f"data:image/png;base64,{response.data[0].b64_json}"  # type: ignore[index]
                            },
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
        """Test inpaint model without mask (mask is optional)."""
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="add magical elements to the scene",
            model=STABILITY_INPAINT,
            response_format="b64_json",
        )

        assert response.created > 0
        assert len(response.data) == 1  # type: ignore[arg-type]
        assert response.data[0].b64_json is not None  # type: ignore[index]

    @pytest.mark.expensive
    def test_erase(
        self, openai_client: OpenAI, use_official_api: bool, chat_vision_model: str
    ) -> None:
        """Test erase model with mask using AWS documentation example."""
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
        assert len(response.data) == 1  # type: ignore[arg-type]
        assert response.data[0].b64_json is not None  # type: ignore[index]

        # Save output for manual inspection
        output = response.data[0].b64_json  # type: ignore[index]
        output_data = base64.b64decode(output)
        (output_dir / "stability_erase_result.jpg").write_bytes(output_data)

    @pytest.mark.expensive
    def test_remove_background(
        self, openai_client: OpenAI, use_official_api: bool, chat_vision_model: str
    ) -> None:
        """Test background removal model using AWS documentation example."""
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
        assert len(response.data) == 1  # type: ignore[arg-type]
        assert response.data[0].b64_json is not None  # type: ignore[index]

        # Save output for manual inspection
        output_data = base64.b64decode(response.data[0].b64_json)  # type: ignore[index]
        (output_dir / "stability_remove_bg_result.png").write_bytes(output_data)

        # VLM validation
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=[
                {  # type: ignore[misc,list-item]
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
                            "image_url": {
                                "url": f"data:image/png;base64,{response.data[0].b64_json}"  # type: ignore[index]
                            },
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
    """Tests for Stability AI control models."""

    @pytest.mark.expensive
    def test_control_sketch(
        self, openai_client: OpenAI, use_official_api: bool, chat_vision_model: str
    ) -> None:
        """Test control sketch model using AWS documentation example."""
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
        assert len(response.data) == 1  # type: ignore[arg-type]
        assert response.data[0].b64_json is not None  # type: ignore[index]

        # Save output for manual inspection
        output_data = base64.b64decode(response.data[0].b64_json)  # type: ignore[index]
        (output_dir / "stability_control_sketch_result.jpg").write_bytes(output_data)

        # VLM validation
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=[
                {  # type: ignore[misc,list-item]
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
                            "image_url": {
                                "url": f"data:image/png;base64,{response.data[0].b64_json}"  # type: ignore[index]
                            },
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
        self, openai_client: OpenAI, use_official_api: bool, chat_vision_model: str
    ) -> None:
        """Test control structure model using AWS documentation example."""
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
        assert len(response.data) == 1  # type: ignore[arg-type]
        assert response.data[0].b64_json is not None  # type: ignore[index]

        # Save output for manual inspection
        output_data = base64.b64decode(response.data[0].b64_json)  # type: ignore[index]
        (output_dir / "stability_control_structure_result.jpg").write_bytes(output_data)

        # VLM validation
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=[
                {  # type: ignore[misc,list-item]
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
                            "image_url": {
                                "url": f"data:image/png;base64,{response.data[0].b64_json}"  # type: ignore[index]
                            },
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
    """Tests for Stability AI style models."""

    @pytest.mark.expensive
    def test_style_guide(
        self, openai_client: OpenAI, use_official_api: bool, sample_image_file: bytes
    ) -> None:
        """Test style-guide model."""
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="Generate in the style of the reference",
            model=STABILITY_STYLE_GUIDE,
            response_format="b64_json",
        )

        assert response.created > 0
        assert len(response.data) == 1  # type: ignore[arg-type]
        assert response.data[0].b64_json is not None  # type: ignore[index]

    @pytest.mark.expensive
    def test_style_transfer_with_mask_as_style_image(
        self, openai_client: OpenAI, use_official_api: bool, chat_vision_model: str
    ) -> None:
        """Test style-transfer model with mask parameter used as style image."""
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
        assert len(response.data) == 1  # type: ignore[arg-type]
        assert response.data[0].b64_json is not None  # type: ignore[index]

        # Save output for manual inspection
        output_data = base64.b64decode(response.data[0].b64_json)  # type: ignore[index]
        (output_dir / "stability_style_transfer_result.jpg").write_bytes(output_data)

        # VLM validation
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=[
                {  # type: ignore[misc,list-item]
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
                            "image_url": {
                                "url": f"data:image/png;base64,{response.data[0].b64_json}"  # type: ignore[index]
                            },
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
        self, openai_client: OpenAI, use_official_api: bool, chat_vision_model: str
    ) -> None:
        """Test style-transfer model with style_image in extra_body."""
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
        assert len(response.data) == 1  # type: ignore[arg-type]
        assert response.data[0].b64_json is not None  # type: ignore[index]

        # Save output for manual inspection
        output_data = base64.b64decode(response.data[0].b64_json)  # type: ignore[index]
        (output_dir / "stability_style_transfer_extra_body_result.jpg").write_bytes(
            output_data
        )

        # VLM validation
        validation_response = openai_client.chat.completions.create(
            model=chat_vision_model,
            messages=[
                {  # type: ignore[misc,list-item]
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
                            "image_url": {
                                "url": f"data:image/png;base64,{response.data[0].b64_json}"  # type: ignore[index]
                            },
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
        """Test that style-transfer without mask or style_image raises BadRequestError."""
        if use_official_api:
            pytest.skip("Stability AI is not available on the official OpenAI API")

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                prompt="statue",
                model=STABILITY_STYLE_TRANSFER,
                response_format="b64_json",
            )

        assert "mask" in str(exc_info.value) or "style_image" in str(exc_info.value)
