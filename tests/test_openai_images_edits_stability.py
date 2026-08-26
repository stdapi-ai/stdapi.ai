"""Coverage of /v1/images/edits against the Stability AI models on Amazon Bedrock.

Every model here reaches the endpoint through the same edit route, but each one
consumes a different subset of the OpenAI parameters: some ignore ``prompt``,
some require a mask, some repurpose ``mask`` as a second input image, and some
require a provider extra passed through ``extra_body``. The assertions pin the
gateway-side contract (payload format, response metadata, parameter validation).
Image content is measured on the pixels wherever the claim admits a measurement,
and left to a vision judge only where it does not.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/stable-image-services.html
     https://stdapi.ai/api_openai_images_edits/
     stdapi/models/image/_stability.py:StabilityImageGenerationJobBase
"""

import base64
import re
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from openai import BadRequestError
from PIL import Image, ImageChops, ImageStat
from pydantic_core import from_json

from stdapi.models.image import ImageGenerationResponse
from stdapi.models.image._stability import StabilityImageGenerationJobBase
from stdapi.models.image.stability_stable_image_erase_object import _EraseJob
from stdapi.models.image.stability_stable_image_inpaint import _InpaintJob
from tests._helpers import decoded_png

from .test_openai_images_generations import validate_image_usage

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

#: JPEG start-of-image marker
_JPEG_MAGIC = b"\xff\xd8\xff"
#: Shape of the ``size`` field built by ``build_images_response`` ("WIDTHxHEIGHT")
_SIZE_PATTERN = re.compile(r"^\d+x\d+$")

#: Every test here: the Stability models have no official OpenAI equivalent.
pytestmark = pytest.mark.gateway(
    "Stability AI is not available on the official OpenAI API"
)


#: Input ceiling of the fast upscale model, which multiplies its input by four.
_FAST_UPSCALE_MAX_PIXELS = 1024 * 1024


def _within_pixel_budget(png: bytes, max_pixels: int) -> bytes:
    """Shrink a PNG to at most *max_pixels*, keeping its aspect ratio.

    The shared sample image is sized for generation, not for upscaling: at the
    model's default size it is already over the fast upscale input ceiling, and
    Stability rejects it before the gateway's own behavior is observable.

    Args:
        png: Source PNG bytes.
        max_pixels: Largest pixel count the model accepts.

    Returns:
        The original bytes when they already fit, a re-encoded PNG otherwise.
    """
    with BytesIO(png) as buffer, Image.open(buffer) as image:
        if image.width * image.height <= max_pixels:
            return png
        scale = (max_pixels / (image.width * image.height)) ** 0.5
        resized = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
        with BytesIO() as out:
            resized.save(out, format="PNG")
            return out.getvalue()


#: Hue byte range (0-255 spans 360°) covering the pink/magenta family.
_PINK_HUE_RANGE: tuple[int, int] = (198, 248)

#: Saturation and value floors keeping near-grey and near-black pixels uncounted.
_COLOURED_PIXEL_FLOORS: tuple[int, int] = (90, 64)

#: Share of the frame the recolored region must gain; the sample measures ~0.22.
_MIN_RECOLORED_SHARE: float = 0.05

#: Saturation below which a pixel is grey enough to carry no colour claim.
_GREY_SATURATION_CEILING: int = 30

#: Pixel-count multiple an upscale must reach; the sample measures ~13.8x.
_MIN_UPSCALE_PIXEL_RATIO: float = 4.0

#: Aspect-ratio drift an upscale may introduce; the sample measures ~0.0015.
_MAX_UPSCALE_ASPECT_DRIFT: float = 0.02

#: Mask level reading as "certainly edit"; the sample mask is antialiased, not binary.
_MASK_EDIT_LEVEL: int = 200

#: Mask level still reading as "certainly preserve", leaving the soft edge uncounted.
_MASK_PRESERVE_LEVEL: int = 55

#: Times more the masked region must move than the preserved one; sample ~55x.
_MIN_INPAINT_DELTA_RATIO: float = 8.0

#: Absolute channel movement the masked region must show; the sample measures ~51.
_MIN_INPAINT_EDIT_DELTA: float = 10.0

#: Share of the frame's outer ring that must be fully transparent; sample ~0.89.
_MIN_TRANSPARENT_BORDER_SHARE: float = 0.6

#: Share of the frame that must stay fully opaque, so a blank cut-out fails; sample ~0.028.
_MIN_OPAQUE_SHARE: float = 0.005

#: Mean saturation a rendered scene must reach; the greyscale sketch measures 0.0.
_MIN_RENDERED_SATURATION: float = 40.0

#: Grey share a rendered scene must stay under; the sketch is 1.0 and the sample ~0.09.
_MAX_RENDERED_GREY_SHARE: float = 0.5

#: Mean channel movement a restyled frame must show against its source; sample ~77.
_MIN_RESTYLE_DELTA: float = 15.0

#: Mean saturation a style transfer must add to its content image; sample ~+102.
_MIN_STYLE_SATURATION_GAIN: float = 40.0


def _pink_pixel_share(image_bytes: bytes) -> float:
    """Return the share of an image's pixels sitting in the pink/magenta hue band.

    Args:
        image_bytes: Encoded image bytes in any format Pillow reads.
    """
    low, high = _PINK_HUE_RANGE
    min_saturation, min_value = _COLOURED_PIXEL_FLOORS
    with BytesIO(image_bytes) as buffer, Image.open(buffer) as image:
        # Pillow types the per-band case, but an HSV image flattens to triples.
        pixels: tuple[tuple[int, int, int], ...] = (
            image.convert("RGB").convert("HSV").get_flattened_data()  # type: ignore[assignment]
        )
        return sum(
            low <= hue <= high and sat > min_saturation and val > min_value
            for hue, sat, val in pixels
        ) / (image.width * image.height)


def _dimensions(image_bytes: bytes) -> tuple[int, int]:
    """Return the ``(width, height)`` of an encoded image."""
    with BytesIO(image_bytes) as buffer, Image.open(buffer) as image:
        return image.width, image.height


def _mean_saturation(image_bytes: bytes) -> float:
    """Return the mean HSV saturation (0-255) of an image.

    Args:
        image_bytes: Encoded image bytes in any format Pillow reads.
    """
    with BytesIO(image_bytes) as buffer, Image.open(buffer) as image:
        return ImageStat.Stat(image.convert("RGB").convert("HSV")).mean[1]


def _grey_share(image_bytes: bytes) -> float:
    """Return the share of pixels whose saturation is under the grey ceiling.

    Args:
        image_bytes: Encoded image bytes in any format Pillow reads.
    """
    with BytesIO(image_bytes) as buffer, Image.open(buffer) as image:
        histogram = image.convert("RGB").convert("HSV").getchannel("S").histogram()
    return sum(histogram[:_GREY_SATURATION_CEILING]) / sum(histogram)


def _mean_abs_delta(
    image_bytes: bytes, other_bytes: bytes, region: Image.Image | None = None
) -> float:
    """Return the mean absolute per-channel difference between two images.

    Args:
        image_bytes: Encoded image bytes in any format Pillow reads.
        other_bytes: The image to compare against, at the same resolution.
        region: Optional Pillow mask restricting the comparison to its set pixels.

    Returns:
        The mean over the three RGB channels, on the 0-255 scale.
    """
    with (
        BytesIO(image_bytes) as buffer,
        Image.open(buffer) as image,
        BytesIO(other_bytes) as other_buffer,
        Image.open(other_buffer) as other,
    ):
        difference = ImageChops.difference(image.convert("RGB"), other.convert("RGB"))
        statistics = ImageStat.Stat(difference, region)
    return sum(statistics.mean) / len(statistics.mean)


def _mask_regions(mask_bytes: bytes) -> tuple[Image.Image, Image.Image]:
    """Split a mask into its unambiguous edit and preserve regions.

    Args:
        mask_bytes: Encoded mask bytes, white marking the region to edit.

    Returns:
        The edit mask and the preserve mask, both usable with
        :func:`_mean_abs_delta`; the soft edge between them belongs to neither.
    """
    with BytesIO(mask_bytes) as buffer, Image.open(buffer) as mask:
        levels = mask.convert("L")
        return (
            levels.point(lambda level: level >= _MASK_EDIT_LEVEL, mode="1"),
            levels.point(lambda level: level <= _MASK_PRESERVE_LEVEL, mode="1"),
        )


def _alpha_shares(png: bytes) -> tuple[float, float]:
    """Measure how much of a PNG is cut out and how much is left untouched.

    Args:
        png: Encoded PNG bytes.

    Returns:
        The share of the one-pixel outer ring that is fully transparent, and the
        share of the whole frame that is fully opaque.
    """
    with BytesIO(png) as buffer, Image.open(buffer) as image:
        alpha = image.convert("RGBA").getchannel("A")
        frame = alpha.histogram()
        inner = alpha.crop((1, 1, alpha.width - 1, alpha.height - 1)).histogram()
    border = [total - interior for total, interior in zip(frame, inner, strict=True)]
    return border[0] / sum(border), frame[255] / sum(frame)


def _assert_style_applied(size: str | None, content: bytes, output: bytes) -> None:
    """Assert a style-transfer answer reports its own bytes and carries the style.

    Args:
        size: The ``size`` field the response reported.
        content: The content image the request sent as ``image``.
        output: The decoded answer.
    """
    width, height = _dimensions(output)
    assert size == f"{width}x{height}"

    gained = _mean_saturation(output) - _mean_saturation(content)
    assert gained >= _MIN_STYLE_SATURATION_GAIN, (
        f"the style image was not applied: mean saturation moved by {gained:.1f} "
        f"against the content image, under the {_MIN_STYLE_SATURATION_GAIN:.0f} floor"
    )


def _rgba_mask_b64() -> str:
    """Build a base64 2x1 RGBA PNG mask: one transparent pixel, one opaque pixel."""
    image = Image.new("RGBA", (2, 1), (0, 0, 0, 255))
    image.putpixel((0, 0), (0, 0, 0, 0))  # transparent -> OpenAI-style edit region
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


async def _build_request(
    monkeypatch: pytest.MonkeyPatch, job: _InpaintJob | _EraseJob, mask: str
) -> dict[str, Any]:
    """Run ``_edit_image`` and return the request it built, without invoking Bedrock."""
    captured: dict[str, Any] = {}

    async def _fake_get_image_from_response(
        _self: object, request: bytes, index: int
    ) -> ImageGenerationResponse:
        captured.update(from_json(request))
        return ImageGenerationResponse(image="", index=index)

    monkeypatch.setattr(
        StabilityImageGenerationJobBase,
        "_get_image_from_response",
        _fake_get_image_from_response,
    )
    for coroutine in await job._edit_image(["c291cmNl"], mask):  # noqa: SLF001
        await coroutine
    return captured


class TestPixelMeasurementsOffline:
    """The pixel measurements the paid tests below assert on, proved without Bedrock.

    A threshold is only worth what its measurement is worth: a helper that
    returned a constant would let every calibrated floor pass against any image.
    These cases pin each helper against images whose answer is known in
    advance -- the committed greyscale sketch, the committed inpaint mask, and
    two synthesised frames -- so a broken measurement fails here, for free,
    instead of being "confirmed" by an expensive run.

    Ref: https://github.com/stdapi-ai/stdapi.ai/issues/167
    """

    pytestmark = pytest.mark.local

    @staticmethod
    def _png(image: Image.Image) -> bytes:
        """Encode a Pillow image as PNG bytes."""
        with BytesIO() as buffer:
            image.save(buffer, format="PNG")
            return buffer.getvalue()

    def test_mean_saturation_reads_greyscale_line_art_as_colourless(self) -> None:
        """The committed sketch carries no colour at all, on both colour measures."""
        sketch = (
            Path(__file__).parent / "samples" / "stability_control_sketch_input.jpg"
        ).read_bytes()

        assert _mean_saturation(sketch) == 0.0
        assert _grey_share(sketch) == 1.0

    def test_mean_abs_delta_of_an_image_against_itself_is_zero(self) -> None:
        """An image compared with itself moves by nothing."""
        frame = self._png(Image.new("RGB", (4, 2), (30, 90, 200)))

        assert _mean_abs_delta(frame, frame) == 0.0

    def test_mean_abs_delta_measures_only_the_region_it_is_given(self) -> None:
        """A change confined to the masked half is invisible outside it."""
        source = self._png(Image.new("RGB", (4, 2), (0, 0, 0)))
        changed = Image.new("RGB", (4, 2), (0, 0, 0))
        changed.paste((255, 255, 255), (0, 0, 2, 2))
        mask = Image.new("L", (4, 2), 0)
        mask.paste(255, (0, 0, 2, 2))

        edit_region, preserve_region = _mask_regions(self._png(mask))
        edited = self._png(changed)

        assert _mean_abs_delta(source, edited, edit_region) == 255.0
        assert _mean_abs_delta(source, edited, preserve_region) == 0.0
        assert _mean_abs_delta(source, edited) == 127.5

    def test_mask_regions_splits_the_committed_inpaint_mask(self) -> None:
        """The sample mask's white and black regions come back separated.

        The mask is antialiased over 224 grey levels, so the two regions are
        deliberately smaller than the frame: the soft edge belongs to neither.
        """
        mask = (
            Path(__file__).parent / "samples" / "stability_inpaint_mask.png"
        ).read_bytes()

        edit_region, preserve_region = _mask_regions(mask)
        pixels = edit_region.width * edit_region.height
        edit_share = sum(edit_region.convert("L").histogram()[255:]) / pixels
        preserve_share = sum(preserve_region.convert("L").histogram()[255:]) / pixels

        assert 0.15 <= edit_share <= 0.25, edit_share
        assert 0.75 <= preserve_share <= 0.85, preserve_share
        # Disjoint by construction, so this cannot exceed 1.0; the floor is the
        # real check -- it fails if the mask stops being near-binary, since the
        # soft edge (neither region) would then claim more of the frame.
        assert 0.99 <= edit_share + preserve_share < 1.0, edit_share + preserve_share

    def test_alpha_shares_separates_a_cut_out_ring_from_its_subject(self) -> None:
        """A transparent frame around an opaque centre reads as a clean cut-out."""
        cut_out = Image.new("RGBA", (6, 6), (0, 0, 0, 0))
        cut_out.paste((10, 20, 30, 255), (2, 2, 4, 4))

        transparent_border, opaque = _alpha_shares(self._png(cut_out))

        assert transparent_border == 1.0
        assert opaque == 4 / 36

    def test_alpha_shares_reads_an_opaque_frame_as_uncut(self) -> None:
        """An image with nothing removed has no transparent ring to report."""
        transparent_border, opaque = _alpha_shares(
            self._png(Image.new("RGBA", (6, 6), (10, 20, 30, 255)))
        )

        assert transparent_border == 0.0
        assert opaque == 1.0

    def test_pink_pixel_share_reads_the_hue_band_and_the_floors(self) -> None:
        """The helper backing the recolor floor is pinned against known answers.

        A frame split between hot pink (hue 233) and saturated blue (hue 170)
        reads as half pink; the committed greyscale sketch, already proved
        colourless by ``_mean_saturation``/``_grey_share`` above, carries none.
        """
        half_pink_half_blue = Image.new("RGB", (4, 2), (255, 105, 180))
        for x, y in ((0, 0), (1, 0), (0, 1), (1, 1)):
            half_pink_half_blue.putpixel((x, y), (0, 0, 255))

        assert _pink_pixel_share(self._png(half_pink_half_blue)) == 0.5

        sketch = (
            Path(__file__).parent / "samples" / "stability_control_sketch_input.jpg"
        ).read_bytes()
        assert _pink_pixel_share(sketch) == 0.0


class TestStabilityMaskPolarityOffline:
    """Alpha-mask conversion for the inpaint/erase models, exercised without Bedrock.

    Stability reads mask polarity the opposite of Nova/Titan: white marks the region
    to edit/erase and black the region to preserve (confirmed by ``test_inpaint``'s
    docstring below and by Stability's own API reference), so an OpenAI-style alpha
    mask must be converted with ``invert=True`` -- the same call Nova/Titan's
    outpainting conversion uses, but for the opposite reason.

    Ref: stdapi/utils.py:alpha_mask_to_bw
         stdapi/models/image/stability_stable_image_inpaint.py:_InpaintJob
         stdapi/models/image/stability_stable_image_erase_object.py:_EraseJob
         https://platform.stability.ai/docs/api-reference#tag/Edit/paths/~1v2beta~1stable-image~1edit~1inpaint/post
         https://github.com/stdapi-ai/stdapi.ai/issues/75
    """

    pytestmark = pytest.mark.local

    @staticmethod
    def _mask_pixels(mask_b64: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Decode a base64 PNG mask and return its two RGB pixels."""
        with BytesIO(base64.b64decode(mask_b64)) as buffer, Image.open(buffer) as image:
            assert image.mode == "RGB", "the alpha channel must be dropped"
            pixel_0 = image.getpixel((0, 0))
            pixel_1 = image.getpixel((1, 0))
        assert isinstance(pixel_0, tuple)
        assert isinstance(pixel_1, tuple)
        return pixel_0, pixel_1

    async def test_inpaint_converts_alpha_mask_to_white_edit_polarity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_InpaintJob`` converts an alpha mask to white=edit, black=preserve."""
        job = _InpaintJob(
            model=None,  # type: ignore[arg-type]
            prompt="prompt",
            count=1,
            width=2,
            height=1,
            quality=None,
            style=None,
            output_format=None,
            output_compression=100,
            extra_params={},
        )
        request = await _build_request(monkeypatch, job, _rgba_mask_b64())

        edit_pixel, preserve_pixel = self._mask_pixels(request["mask"])
        assert edit_pixel == (255, 255, 255), "transparent pixel must map to white"
        assert preserve_pixel == (0, 0, 0), "opaque pixel must map to black"

    async def test_erase_converts_alpha_mask_to_white_edit_polarity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_EraseJob`` converts an alpha mask to white=erase, black=preserve."""
        job = _EraseJob(
            model=None,  # type: ignore[arg-type]
            prompt="prompt",
            count=1,
            width=2,
            height=1,
            quality=None,
            style=None,
            output_format=None,
            output_compression=100,
            extra_params={},
        )
        request = await _build_request(monkeypatch, job, _rgba_mask_b64())

        edit_pixel, preserve_pixel = self._mask_pixels(request["mask"])
        assert edit_pixel == (255, 255, 255), "transparent pixel must map to white"
        assert preserve_pixel == (0, 0, 0), "opaque pixel must map to black"


class TestStabilityEditing:
    """Image-to-image editing on Stable Diffusion 3.5 Large.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-diffusion-3-5-large.html
         stdapi/models/image/stability_stable_diffusion.py:TextToImageJob
    """

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", STABILITY_ALL)
    def test_edit_b64_single(
        self, openai_client: OpenAI, sample_image_file: bytes, model_id: str
    ) -> None:
        """An edit request without a mask runs as image-to-image and returns one PNG.

        The gateway sends ``mode=image-to-image`` with a default ``strength`` of
        0.35, so the source image is required but no mask is. Usage counts the
        single source image as an input image and ``total_tokens`` is the sum of
        the two counters.

        Ref: stdapi/routes/_images_common.py:build_images_response
        """
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
        assert data[0].url is None
        decoded_png(data[0].b64_json)

        validate_image_usage(response.usage)


class TestStabilityUpscaleModels:
    """Upscaling models reached through /v1/images/edits.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/stable-image-services.html
         stdapi/models/image/stability_stable_fast_upscale.py:_FastUpscaleJob
         stdapi/models/image/stability_stable_image_edit.py:_SimpleEditJob
    """

    @pytest.mark.expensive
    def test_fast_upscale(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Fast upscale ignores the prompt and honors ``output_format=jpeg``.

        The backend request carries only the image and the output format, so the
        prompt the OpenAI client insists on is dropped; ``jpeg`` is a native
        Stability output format and is therefore returned without a re-encode,
        and the response metadata reports it.

        Upscaling is a claim about size, so it is measured: the answer must hold
        several times the source pixel count and keep the source aspect ratio,
        which a crop or a re-encode would not. The sample comes back at ~16x the
        pixel count with no measurable aspect drift.

        Ref: stdapi/models/image/_stability.py:StabilityImageGenerationJobBase._finalize_request
        """
        input_image = _within_pixel_budget(sample_image_file, _FAST_UPSCALE_MAX_PIXELS)

        response = openai_client.images.edit(
            image=input_image,
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
        output_data = base64.b64decode(b64_json)
        assert output_data.startswith(_JPEG_MAGIC)

        source_width, source_height = _dimensions(input_image)
        width, height = _dimensions(output_data)
        pixel_ratio = (width * height) / (source_width * source_height)
        assert pixel_ratio >= _MIN_UPSCALE_PIXEL_RATIO, (
            f"the upscaled image holds {pixel_ratio:.1f}x the source pixels, "
            f"under the {_MIN_UPSCALE_PIXEL_RATIO:.0f}x floor"
        )
        aspect_drift = abs((width / height) / (source_width / source_height) - 1)
        assert aspect_drift <= _MAX_UPSCALE_ASPECT_DRIFT, (
            f"upscaling must not reframe the image: the aspect ratio moved by "
            f"{aspect_drift:.2%}"
        )

    @pytest.mark.expensive
    # Both upscale models are served by stability_stable_image_edit.py, the same
    # job the control-sketch, control-structure, style-guide and outpaint tests
    # already exercise, so one of the pair proves the path; it is the cheaper one.
    @pytest.mark.parametrize("model_id", [STABILITY_CONSERVATIVE_UPSCALE])
    def test_creative_upscale(self, openai_client: OpenAI, model_id: str) -> None:
        """Prompt-guided upscaling returns a larger JPEG of the same framing.

        Upscaling is a claim about size, so it is measured: the answer must be
        bigger on both axes and keep the source aspect ratio, which a crop or a
        re-encode would not. The sample comes back at ~13.8x the pixel count
        with 0.15% aspect drift, and the reported ``size`` must match those
        bytes.

        Ref: stdapi/models/image/stability_stable_image_edit.py:_SimpleEditJob
        """
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

        source_width, source_height = _dimensions(input_image)
        width, height = _dimensions(output_data)
        enlarged = f"{source_width}x{source_height} came back as {width}x{height}"
        assert width > source_width, f"upscaling must widen the image: {enlarged}"
        assert height > source_height, f"upscaling must heighten the image: {enlarged}"

        pixel_ratio = (width * height) / (source_width * source_height)
        assert pixel_ratio >= _MIN_UPSCALE_PIXEL_RATIO, (
            f"the upscaled image holds {pixel_ratio:.1f}x the source pixels, "
            f"under the {_MIN_UPSCALE_PIXEL_RATIO:.0f}x floor"
        )
        aspect_drift = abs((width / height) / (source_width / source_height) - 1)
        assert aspect_drift <= _MAX_UPSCALE_ASPECT_DRIFT, (
            f"upscaling must not reframe the image: the aspect ratio moved by "
            f"{aspect_drift:.2%}"
        )
        assert response.size == f"{width}x{height}"


class TestStabilityEditModels:
    """Stability models that edit a region of the source image.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/stable-image-services.html
         https://stdapi.ai/api_openai_images_edits/
    """

    @pytest.mark.expensive
    def test_search_recolor(self, openai_client: OpenAI) -> None:
        """Search-and-recolor recolors the region named by the ``select_prompt`` extra.

        ``select_prompt`` has no OpenAI counterpart, so it travels as a provider
        extra: ``prompt`` describes the target colour while ``select_prompt``
        selects what to recolor.

        What the gateway owes is that both parameters reach the model and that
        the answer is a PNG at the source resolution whose reported ``size``
        matches the bytes. The colour claim is measured on the pixels: the frame
        has to gain a substantial pink/magenta region — the sample gains ~22% of
        it against a 0.005% baseline.

        Ref: stdapi/models/image/stability_search_recolor.py:_SearchRecolorJob
        """
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
        output_data = decoded_png(data[0].b64_json)

        # Save output for manual inspection
        (output_dir / "stability_search_recolor_result.jpg").write_bytes(output_data)

        width, height = _dimensions(output_data)
        assert (width, height) == _dimensions(input_image), (
            "search-and-recolor must answer at the source resolution"
        )
        assert response.size == f"{width}x{height}"

        recolored = _pink_pixel_share(output_data) - _pink_pixel_share(input_image)
        assert recolored >= _MIN_RECOLORED_SHARE, (
            f"the selected region was not recolored: the pink/magenta share of the "
            f"frame grew by {recolored:.3%}, under the {_MIN_RECOLORED_SHARE:.0%} floor"
        )

    def test_search_recolor_missing_select_prompt(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Search-and-recolor without ``select_prompt`` is a 400 naming the parameter.

        Ref: stdapi/models/image/stability_search_recolor.py:_SearchRecolorJob
        """
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
        self, openai_client: OpenAI, chat_vision_judge_model: str
    ) -> None:
        """Search-and-replace swaps the object named by the ``search_prompt`` extra.

        ``prompt`` describes the replacement while the ``search_prompt`` provider
        extra selects the object to replace.

        The gateway owes a PNG at the source resolution whose reported ``size``
        matches the bytes, and that is asserted on the pixels. The remaining
        claim is object identity -- a jacket where a sweater was -- which no
        measurement stands in for, so it is the one place in this file where a
        vision judge is still the honest check (issue #163). The question put to
        it is coarse and single-fact, because a judge asked to grade quality
        rejects correct results.

        Ref: stdapi/models/image/stability_search_replace.py:_SearchReplaceJob
        """
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
        output_data = decoded_png(b64_json)

        # Save output for manual inspection
        (output_dir / "stability_search_replace_result.jpg").write_bytes(output_data)

        width, height = _dimensions(output_data)
        assert (width, height) == _dimensions(input_image), (
            "search-and-replace must answer at the source resolution"
        )
        assert response.size == f"{width}x{height}"

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
                                "Does this image show a person wearing a jacket? "
                                "If yes, respond with only 'YES'. "
                                "If no, respond with 'NO' followed by a brief explanation."
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
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Search-and-replace without ``search_prompt`` is a 400 naming the parameter.

        Ref: stdapi/models/image/stability_search_replace.py:_SearchReplaceJob
        """
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
    def test_inpaint(self, openai_client: OpenAI) -> None:
        """Inpainting regenerates the masked area and leaves the rest alone.

        The AWS sample mask is an antialiased black-and-white PNG with no alpha
        channel, so the gateway forwards it untouched; Stability reads white as
        maximum inpaint strength, the opposite of the Titan/Nova convention where
        black marks the area to edit. That polarity is the claim, and it is
        measured: the frame is compared with its source inside and outside the
        mask, and the masked region has to move by far more. The sample moves by
        ~51 levels inside against ~0.9 outside, a ratio of ~55. A ratio rather
        than an equality outside, because the answer is re-encoded rather than
        returned untouched.

        Ref: stdapi/models/image/stability_stable_image_inpaint.py:_InpaintJob
             stdapi/utils.py:alpha_mask_to_bw
        """
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
        output_data = decoded_png(data[0].b64_json)

        # Save output for manual inspection
        (output_dir / "stability_inpaint_result.jpg").write_bytes(output_data)

        width, height = _dimensions(output_data)
        assert (width, height) == _dimensions(input_image), (
            "inpainting must answer at the source resolution"
        )
        assert response.size == f"{width}x{height}"

        edit_region, preserve_region = _mask_regions(mask_image)
        edited = _mean_abs_delta(input_image, output_data, edit_region)
        preserved = _mean_abs_delta(input_image, output_data, preserve_region)
        assert edited >= _MIN_INPAINT_EDIT_DELTA, (
            f"the masked region was not repainted: it moved by {edited:.1f} levels, "
            f"under the {_MIN_INPAINT_EDIT_DELTA:.0f} floor"
        )
        assert edited >= _MIN_INPAINT_DELTA_RATIO * preserved, (
            f"the mask was not honored: the masked region moved by {edited:.1f} "
            f"levels and the preserved region by {preserved:.1f}, under the "
            f"{_MIN_INPAINT_DELTA_RATIO:.0f}x ratio the polarity requires"
        )

    @pytest.mark.expensive
    def test_inpaint_without_mask(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Inpainting accepts a request with no mask at all.

        The ``mask`` key is only added to the backend request when one is
        supplied; Stability then derives the mask from the input image's alpha
        channel, so an opaque source image is edited as a whole.

        Ref: stdapi/models/image/stability_stable_image_inpaint.py:_InpaintJob
        """
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
        decoded_png(data[0].b64_json)

    @pytest.mark.expensive
    def test_erase(self, openai_client: OpenAI) -> None:
        """Object erasure consumes the mask and ignores the prompt.

        ``EraseRequest`` carries only the image and the mask, so the prompt the
        OpenAI client requires is dropped before the backend call.

        Ref: stdapi/models/image/stability_stable_image_erase_object.py:_EraseJob
        """
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
        output_data = decoded_png(data[0].b64_json)

        # Save output for manual inspection
        (output_dir / "stability_erase_result.jpg").write_bytes(output_data)

    @pytest.mark.expensive
    def test_remove_background(self, openai_client: OpenAI) -> None:
        """Background removal isolates the subject from a prompt-free request.

        "The background is gone" is a transparency claim, so it is measured on
        the alpha channel: the opaque source comes back with an alpha channel,
        its outer ring is overwhelmingly transparent, and enough of the frame
        stays fully opaque that a blank cut-out fails. The sample answers with
        89% of the ring transparent and 2.8% of the frame opaque.

        Ref: stdapi/models/image/stability_stable_image_remove_background.py:_RemoveBackgroundJob
        """
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
        output_data = decoded_png(data[0].b64_json)

        # Save output for manual inspection
        (output_dir / "stability_remove_bg_result.png").write_bytes(output_data)

        width, height = _dimensions(output_data)
        assert (width, height) == _dimensions(input_image), (
            "background removal must answer at the source resolution"
        )
        assert response.size == f"{width}x{height}"

        with BytesIO(output_data) as buffer, Image.open(buffer) as image:
            mode = image.mode
        assert mode in {"RGBA", "LA"}, (
            f"a cut-out subject needs an alpha channel, the answer is {mode}"
        )

        transparent_border, opaque = _alpha_shares(output_data)
        assert transparent_border >= _MIN_TRANSPARENT_BORDER_SHARE, (
            f"the background was not removed: {transparent_border:.1%} of the "
            f"frame's outer ring is transparent, under the "
            f"{_MIN_TRANSPARENT_BORDER_SHARE:.0%} floor"
        )
        assert opaque >= _MIN_OPAQUE_SHARE, (
            f"the subject was removed with the background: only {opaque:.2%} of "
            f"the frame is opaque, under the {_MIN_OPAQUE_SHARE:.1%} floor"
        )


class TestStabilityControlModels:
    """Control models: the input image constrains the generated structure.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/stable-image-services.html
         stdapi/models/image/stability_stable_image_edit.py:_SimpleEditJob
    """

    @pytest.mark.expensive
    def test_control_sketch(self, openai_client: OpenAI) -> None:
        """Control-sketch turns a sketch into a rendered scene described by the prompt.

        The input is greyscale line art -- mean saturation 0.0, every pixel grey
        -- so "it was rendered" is measurable: the answer has to carry real
        colour, at the sketch's own resolution. The sample comes back at mean
        saturation 116 with 9% of it still grey.

        Ref: stdapi/models/image/stability_stable_image_edit.py:_SimpleEditJob
        """
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
        output_data = decoded_png(data[0].b64_json)

        # Save output for manual inspection
        (output_dir / "stability_control_sketch_result.jpg").write_bytes(output_data)

        width, height = _dimensions(output_data)
        assert (width, height) == _dimensions(input_image), (
            "control-sketch must answer at the source resolution"
        )
        assert response.size == f"{width}x{height}"

        saturation = _mean_saturation(output_data)
        assert saturation >= _MIN_RENDERED_SATURATION, (
            f"the sketch was not rendered: mean saturation {saturation:.1f}, "
            f"under the {_MIN_RENDERED_SATURATION:.0f} floor"
        )
        grey = _grey_share(output_data)
        assert grey <= _MAX_RENDERED_GREY_SHARE, (
            f"the answer is still mostly line art: {grey:.1%} of it is grey, "
            f"over the {_MAX_RENDERED_GREY_SHARE:.0%} ceiling"
        )

    @pytest.mark.expensive
    def test_control_structure(self, openai_client: OpenAI) -> None:
        """Control-structure answers with a restyled frame at the source resolution.

        The restyling is measured as movement away from the source: an answer
        that merely echoed the input, the failure this test exists to catch,
        would move by nothing. The sample moves by ~77 levels per channel, and
        the reported ``size`` must match the returned bytes.

        Documented limitation: that the answer keeps the source *composition* is
        **not** verified here. The obvious structural measure carries no signal
        on this sample — a 64x64 luminance correlation reads 0.064 against the
        source and 0.107 against an unrelated image — so the test would pass on
        an image having nothing to do with the input, and asserting the claim
        would be asserting something unmeasured.

        Ref: stdapi/models/image/stability_stable_image_edit.py:_SimpleEditJob
        """
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
        output_data = decoded_png(data[0].b64_json)

        # Save output for manual inspection
        (output_dir / "stability_control_structure_result.jpg").write_bytes(output_data)

        width, height = _dimensions(output_data)
        assert (width, height) == _dimensions(input_image), (
            "control-structure must answer at the source resolution"
        )
        assert response.size == f"{width}x{height}"

        restyled = _mean_abs_delta(input_image, output_data)
        assert restyled >= _MIN_RESTYLE_DELTA, (
            f"the source was echoed rather than restyled: it moved by "
            f"{restyled:.1f} levels, under the {_MIN_RESTYLE_DELTA:.0f} floor"
        )


class TestStabilityStyleModels:
    """Style models, including the two ways of passing a second input image.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/stable-image-services.html
         stdapi/models/image/stability_stable_style_transfer.py:_StyleTransferJob
    """

    @pytest.mark.expensive
    def test_style_guide(self, openai_client: OpenAI, sample_image_file: bytes) -> None:
        """Style-guide takes a single image as the style reference for the prompt.

        Ref: stdapi/models/image/stability_stable_image_edit.py:_SimpleEditJob
        """
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
        decoded_png(data[0].b64_json)

    @pytest.mark.expensive
    def test_style_transfer_with_mask_as_style_image(
        self, openai_client: OpenAI
    ) -> None:
        """Style transfer reads the OpenAI ``mask`` upload as its style image.

        Style transfer needs two images but the endpoint only has one binary
        image field, so the mask slot is repurposed as ``style_image`` — it is
        not used as a mask at all. That the second image arrived is measured on
        the saturation the style carries: the content image sits at mean
        saturation 84 and the style image at 197, so an answer that ignored the
        style stays near the content. The sample gains ~106.

        Ref: stdapi/models/image/stability_stable_style_transfer.py:_StyleTransferJob
        """
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
        output_data = decoded_png(data[0].b64_json)

        # Save output for manual inspection
        (output_dir / "stability_style_transfer_result.jpg").write_bytes(output_data)

        _assert_style_applied(response.size, input_image, output_data)

    @pytest.mark.expensive
    def test_style_transfer_with_style_image_parameter(
        self, openai_client: OpenAI
    ) -> None:
        """A base64 ``style_image`` extra replaces the mask upload for style transfer.

        This is the JSON-friendly form of the same input: the provider extra is
        forwarded as ``style_image``, so no mask upload is needed. Same inputs,
        same model and same measured claim as the mask form above, so both
        assert one shared floor; the sample gains ~102.

        Ref: stdapi/models/image/stability_stable_style_transfer.py:_StyleTransferJob
        """
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
        output_data = decoded_png(data[0].b64_json)

        # Save output for manual inspection
        (output_dir / "stability_style_transfer_extra_body_result.jpg").write_bytes(
            output_data
        )

        _assert_style_applied(response.size, input_image, output_data)

    def test_style_transfer_missing_style_image(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Style transfer with neither ``mask`` nor ``style_image`` is a 400.

        The error names both spellings of the missing input, because either one
        satisfies the model.

        Ref: stdapi/models/image/stability_stable_style_transfer.py:_StyleTransferJob
        """
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
