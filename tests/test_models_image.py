"""Image-model internals: pricing-spec buckets, mask polarity, invoke billing, streaming.

Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase
     stdapi/models/image/amazon_titan_image_generator.py:image_spec
     stdapi/usage.py:record_bedrock_usage
"""

from asyncio import CancelledError, Event, sleep, wait_for
from decimal import Decimal
from io import BytesIO
from typing import TYPE_CHECKING, Any, cast

import pytest
from PIL import Image
from pybase64 import b64decode as pybase64_b64decode
from pybase64 import b64encode

import stdapi.models
import stdapi.models.image
from stdapi import usage
from stdapi.api_errors import ApiError
from stdapi.models import InvokeResult
from stdapi.models.image import (
    ImageGenerationJobBase,
    ImageGenerationResponse,
    ImageModelBase,
)
from stdapi.models.image import amazon_nova_canvas as nova_canvas
from stdapi.models.image._stability import (
    StabilityImageGenerationJobBase,
    StabilityImageModelBase,
)
from stdapi.models.image.amazon_titan_image_generator import (
    _ImageGenerationConfig,
    _ImageGenerationJob,
    _Request,
    _TextToImageParams,
    image_spec,
)
from stdapi.monitoring import REQUEST_ID
from stdapi.pricing import Dimension, Price, PriceKey, Service, _state
from stdapi.usage import IMAGE_SPEC, compute_costs
from stdapi.utils import convert_base64_image
from tests.conftest import set_test_price

if TYPE_CHECKING:
    from collections.abc import Awaitable, Generator


#: Local, in-process tests metering into the per-request usage scopes.
pytestmark = [pytest.mark.local, pytest.mark.usefixtures("usage_scope")]


class TestImageSpecTitanTiers:
    """Titan Image Generator pricing buckets: two doubling tiers from a 512 px low tier.

    Ref: stdapi/models/image/amazon_titan_image_generator.py:image_spec
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
    """

    @pytest.mark.parametrize(
        ("width", "height", "expected_resolution"),
        [
            (320, 320, 512),
            (512, 512, 512),
            (512, 320, 512),
            (513, 512, 1024),
            (768, 768, 1024),
            (1024, 1024, 1024),
            # Above the highest tier still bills at the largest published bucket.
            (2048, 2048, 1024),
        ],
    )
    def test_max_dimension_selects_smallest_covering_tier(
        self, width: int, height: int, expected_resolution: int
    ) -> None:
        """The larger of width/height picks the smallest tier that covers it.

        With the Titan defaults there are only two buckets (512 and 1024), so
        anything above 1024 px still bills at the largest published one.
        """
        assert image_spec(width, height, "standard") == (
            f"{expected_resolution}:standard"
        )


class TestImageSpecNovaCanvasTiers:
    """Nova Canvas pricing buckets: three doubling tiers from a 1024 px low tier.

    Nova Canvas accepts sides up to 4096 px, so its invoke path calls
    ``image_spec`` with ``low_tier_max=1024, tiers=3`` instead of Titan's
    defaults.

    Ref: stdapi/models/image/amazon_titan_image_generator.py:image_spec
         stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob._invoke_and_process_response
         https://docs.aws.amazon.com/nova/latest/userguide/image-gen-access.html
    """

    @pytest.mark.parametrize(
        ("width", "height", "expected_resolution"),
        [
            (1024, 1024, 1024),
            (2048, 1024, 2048),
            (2048, 2048, 2048),
            # Intermediate sizes round up to the next covering tier.
            (1500, 1024, 2048),
            (2049, 1024, 4096),
            (4096, 4096, 4096),
        ],
    )
    def test_third_tier_covers_dimensions_above_2048(
        self, width: int, height: int, expected_resolution: int
    ) -> None:
        """With tiers=3, images above 2048px resolve to the 4096 pricing tier."""
        assert image_spec(width, height, "standard", low_tier_max=1024, tiers=3) == (
            f"{expected_resolution}:standard"
        )


class TestImageSpecQuality:
    """Quality propagation into the ``"<resolution>:<quality>"`` pricing label.

    Ref: stdapi/models/image/amazon_titan_image_generator.py:image_spec
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
    """

    @pytest.mark.parametrize("quality", ["standard", "premium"])
    def test_quality_is_propagated_into_the_spec_label(self, quality: str) -> None:
        """The resolved AWS quality is appended verbatim after the resolution."""
        assert image_spec(1024, 1024, quality) == f"1024:{quality}"  # type: ignore[arg-type]

    def test_missing_quality_defaults_to_standard(self) -> None:
        """A None quality falls back to the "standard" pricing label."""
        assert image_spec(512, 512, None) == "512:standard"


class TestTitanResponseQualityEcho:
    """The echoed response quality mirrors the requested tier, not the AWS bucket.

    Ref: stdapi/models/image/amazon_titan_image_generator.py:_ImageGenerationJob._set_extra_config
         stdapi/models/image/amazon_titan_image_generator.py:AMZ_QUALITY_MAP
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
    """

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [("low", "low"), ("medium", "medium"), ("high", "high")],
    )
    def test_response_quality_matches_request(
        self, requested: str, expected: str
    ) -> None:
        """Both ``low`` and ``medium`` map to AWS ``standard`` but echo distinctly.

        The Bedrock request only carries the two AWS buckets, so the requested
        OpenAI tier has to be remembered separately for the response.
        """
        job = _ImageGenerationJob(
            model=cast("Any", None),
            prompt="a cat",
            count=1,
            width=512,
            height=512,
            quality=requested,
            style=None,
            output_format=None,
            output_compression=0,
            extra_params={},
        )
        request = _Request(
            taskType="TEXT_IMAGE",
            textToImageParams=_TextToImageParams(text="a cat"),
            imageGenerationConfig=_ImageGenerationConfig(
                width=512, height=512, numberOfImages=1
            ),
        )
        job._set_extra_config(request, "textToImageParams")  # noqa: SLF001
        assert job.quality == expected
        assert request["imageGenerationConfig"]["quality"] == (
            "premium" if requested == "high" else "standard"
        ), "Bedrock must receive the AWS bucket, not the OpenAI quality tier"


class TestNovaCanvasResponseQualityEcho:
    """The echoed response quality mirrors the requested tier, not the collapsed AWS bucket.

    Nova Canvas only exposes two AWS quality tiers ("standard"/"premium"), so
    without re-deriving the echo from the originally requested tier, a
    ``low`` request would come back mislabeled as ``medium`` (both collapse
    to "standard" on the AWS side).

    Ref: stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob._apply_quality_and_style
         stdapi/models/image/amazon_titan_image_generator.py:AMZ_QUALITY_MAP
         https://docs.aws.amazon.com/nova/latest/userguide/image-gen-access.html
    """

    @pytest.mark.parametrize(
        ("requested", "expected", "amz_tier"),
        [
            ("low", "low", "standard"),
            ("medium", "medium", "standard"),
            ("high", "high", "premium"),
            ("premium", "high", "premium"),
        ],
    )
    def test_response_quality_matches_request(
        self, requested: str, expected: str, amz_tier: str
    ) -> None:
        """``low`` and ``medium`` both collapse to AWS "standard" but echo distinctly.

        The AWS-side tier written into the request must stay driven by
        ``get_amz_quality`` alone; only the echoed ``response_quality`` is
        re-derived from the originally requested tier.
        """
        job = nova_canvas._ImageGenerationJob(  # noqa: SLF001
            model=cast("Any", None),
            prompt="a cat",
            count=1,
            width=1024,
            height=1024,
            quality=requested,
            style=None,
            output_format=None,
            output_compression=0,
            extra_params={},
        )
        request = nova_canvas._Request(  # noqa: SLF001
            taskType="TEXT_IMAGE",
            textToImageParams=nova_canvas._TextToImageParams(text="a cat"),  # noqa: SLF001
            imageGenerationConfig=nova_canvas._ImageGenerationConfig(  # noqa: SLF001
                width=1024, height=1024, numberOfImages=1
            ),
        )
        job._apply_quality_and_style(request, "textToImageParams")  # noqa: SLF001
        assert job.quality == expected
        assert request["imageGenerationConfig"]["quality"] == amz_tier, (
            "Bedrock must receive the AWS bucket, not the OpenAI quality tier"
        )


def _b64_rgba_png(alpha: int) -> str:
    """Encode a 1x1 RGBA PNG with the given alpha as a base64 string."""
    buffer = BytesIO()
    Image.new("RGBA", (1, 1), (1, 2, 3, alpha)).save(buffer, format="PNG")
    return b64encode(buffer.getvalue()).decode()


def _b64_rgb_png() -> str:
    """Encode a 1x1 black RGB (no-alpha) PNG as a base64 string."""
    buffer = BytesIO()
    Image.new("RGB", (1, 1), (0, 0, 0)).save(buffer, format="PNG")
    return b64encode(buffer.getvalue()).decode()


class TestMaskAlphaConversion:
    """An OpenAI alpha-transparency mask becomes the Bedrock pure black/white RGB mask.

    OpenAI marks the region to edit with alpha 0, while Titan and Nova Canvas
    take an alpha-less mask where black is inside the mask; outpainting then
    alters the white pixels instead, hence the inverted polarity. Both backends
    implement the conversion separately, so each case runs against both.

    Ref: stdapi/utils.py:alpha_mask_to_bw
         https://docs.aws.amazon.com/nova/latest/userguide/image-gen-access.html
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html
         stdapi/models/image/amazon_titan_image_generator.py:_ImageGenerationJob
         stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob
    """

    @pytest.fixture(
        params=[
            pytest.param(
                (_ImageGenerationJob, _ImageGenerationConfig, 512), id="titan"
            ),
            pytest.param(
                (
                    nova_canvas._ImageGenerationJob,  # noqa: SLF001
                    nova_canvas._ImageGenerationConfig,  # noqa: SLF001
                    1024,
                ),
                id="nova_canvas",
            ),
        ]
    )
    def job_and_config(self, request: pytest.FixtureRequest) -> tuple[Any, Any]:
        """Build a 1-image edit job and its config at the backend's native size.

        Returns:
            The job and the matching generation config.
        """
        job_cls, config_cls, size = request.param
        job = job_cls(
            model=cast("Any", None),
            prompt="a cat",
            count=1,
            width=size,
            height=size,
            quality=None,
            style=None,
            output_format=None,
            output_compression=0,
            extra_params={},
        )
        return job, config_cls(width=size, height=size, numberOfImages=1)

    async def test_inpainting_mask_with_alpha_is_converted_to_bw(
        self, job_and_config: tuple[Any, Any]
    ) -> None:
        """A transparent inpainting mask is converted to pure black."""
        job, config = job_and_config
        request = await job._get_request_inpainting(config, "image", _b64_rgba_png(0))  # noqa: SLF001

        assert request["taskType"] == "INPAINTING"
        assert request["inPaintingParams"]["image"] == "image"
        mask = request["inPaintingParams"]["maskImage"]
        assert mask != _b64_rgba_png(0), "an alpha mask must not be forwarded as-is"

        with BytesIO(pybase64_b64decode(mask)) as buffer, Image.open(buffer) as image:
            assert image.mode == "RGB"
            assert image.getpixel((0, 0)) == (0, 0, 0)

    async def test_outpainting_mask_without_alpha_passes_through(
        self, job_and_config: tuple[Any, Any]
    ) -> None:
        """An outpainting mask with no alpha channel is forwarded unchanged.

        A pure black/white PNG is already a native Bedrock mask, so the
        conversion is a no-op passthrough.
        """
        job, config = job_and_config
        mask_input = _b64_rgb_png()
        request = await job._get_request_outpainting(config, "image", mask_input)  # noqa: SLF001

        assert request["taskType"] == "OUTPAINTING"
        assert request["outPaintingParams"]["image"] == "image"
        assert request["outPaintingParams"]["maskImage"] == mask_input

    async def test_outpainting_mask_with_alpha_uses_inverted_polarity(
        self, job_and_config: tuple[Any, Any]
    ) -> None:
        """A transparent outpainting mask is converted to pure white.

        Outpainting uses the opposite mask polarity from inpainting: white
        marks the region to generate, black the region to preserve.
        """
        job, config = job_and_config
        request = await job._get_request_outpainting(config, "image", _b64_rgba_png(0))  # noqa: SLF001

        assert request["taskType"] == "OUTPAINTING"
        mask = request["outPaintingParams"]["maskImage"]

        with BytesIO(pybase64_b64decode(mask)) as buffer, Image.open(buffer) as image:
            assert image.mode == "RGB"
            assert image.getpixel((0, 0)) == (255, 255, 255)


class _FakeBody:
    """Minimal async body mimicking a botocore streaming response body."""

    def __init__(self, payload: bytes) -> None:
        """Store the raw JSON payload to return from read()."""
        self._payload = payload

    async def read(self) -> bytes:
        """Return the stored payload."""
        return self._payload


class TestInvokeImageBilling:
    """A stubbed InvokeModel response drives per-image billing at the spec rate.

    Ref: stdapi/models/image/__init__.py:ImageModelBase._record_invoke_usage
         stdapi/usage.py:record_bedrock_usage
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html
    """

    @pytest.fixture(autouse=True)
    def _request_context(self, request_log: dict[str, Any]) -> Generator[None]:
        """Provide the request ID the invoke request metadata is built from."""
        id_token = REQUEST_ID.set("img1")
        yield
        REQUEST_ID.reset(id_token)

    async def test_stubbed_invoke_bills_images_by_spec_region_and_price(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two returned images bill ``output_images == 2`` in the serving region.

        The number of images actually billed comes from the response ``images``
        list, and the per-image rate is looked up with the ``IMAGE_SPEC``
        resolution/quality bucket rather than the model's flat rate.
        """

        class _FakeInvokeClient:
            """Fake Bedrock runtime client returning a Nova Canvas-style body."""

            async def invoke_model(self, **_kwargs: object) -> dict[str, Any]:
                """Return two images and an input-token-count header."""
                return {
                    "ResponseMetadata": {
                        "HTTPHeaders": {"x-amzn-bedrock-input-token-count": "10"}
                    },
                    "body": _FakeBody(b'{"images": ["aaa", "bbb"]}'),
                }

        async def _fake_candidates(_model_id: str, **_kwargs: object) -> list[str]:
            return ["eu-west-1"]

        async def _fake_resolve(model_id: str, _region: str, **_kwargs: object) -> str:
            return model_id

        monkeypatch.setattr(
            stdapi.models, "compute_candidate_regions", _fake_candidates
        )
        monkeypatch.setattr(stdapi.models, "resolve_routed_model_id", _fake_resolve)
        monkeypatch.setattr(
            stdapi.models,
            "bedrock_client",
            lambda _region, **_kwargs: _FakeInvokeClient(),
        )

        set_test_price(
            "canvasstylemodel", "eu-west-1", Dimension.INPUT_TOKENS, "0.000001", "USD"
        )
        # A different flat per-image rate proves the spec-keyed lookup is used.
        set_test_price(
            "canvasstylemodel", "eu-west-1", Dimension.OUTPUT_IMAGES, "0.9", "USD"
        )
        _state.price_index[
            PriceKey(
                Service.BEDROCK,
                "canvasstylemodel",
                "eu-west-1",
                Dimension.OUTPUT_IMAGES,
                "standard",
                "",
                "",
                "1024:standard",
            )
        ] = Price(Decimal("0.06"), "USD")

        IMAGE_SPEC.set("1024:standard")
        model: ImageModelBase[Any, Any, Any] = ImageModelBase("canvasstylemodel")
        result = await model.invoke({"taskType": "TEXT_IMAGE"})

        assert result.input_tokens == 10
        records = usage.USAGE.get()
        assert len(records) == 1
        key, record = next(iter(records.items()))
        assert key.region == "eu-west-1"
        assert key.model == "canvasstylemodel"
        assert record.quantities[Dimension.INPUT_TOKENS] == 10
        assert record.quantities[Dimension.OUTPUT_IMAGES] == 2
        assert record.output_images_by_spec == {"1024:standard": 2}
        # The override clears IMAGE_SPEC to avoid leaking into later calls.
        assert IMAGE_SPEC.get() == ""
        compute_costs()
        # 10 * 0.000001 (tokens) + 2 * 0.06 (spec rate, not the 0.9 flat rate).
        assert record.cost == Decimal("0.120010")


class TestStreamCancelsAbandonedJobs:
    """An abandoned image stream cancels the jobs that are still in flight.

    Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._stream_completed_images
    """

    async def test_early_close_cancels_pending_tasks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closing the stream after one image cancels the remaining jobs.

        An image completing after the last usage drain would record billed
        usage that nothing ever reads, so the pending tasks must be cancelled.
        """
        cancelled = Event()
        first_image = ImageGenerationResponse(image="aaa", index=0)

        async def _fast() -> ImageGenerationResponse:
            return first_image

        async def _never() -> ImageGenerationResponse:
            try:
                await sleep(3600)
            except CancelledError:
                cancelled.set()
                raise
            return ImageGenerationResponse(image="bbb", index=1)

        monkeypatch.setattr(
            ImageGenerationJobBase, "_ensure_image_output_format", _passthrough_format
        )
        job = object.__new__(ImageGenerationJobBase)
        stream = job._stream_completed_images([_fast(), _never()])  # noqa: SLF001
        assert await stream.__anext__() is first_image
        await stream.aclose()
        await wait_for(cancelled.wait(), timeout=5)


class TestStabilityMultiImageTokenAccumulation:
    """Stability ``n>1`` fan-out sums token counts across the per-image calls.

    Stability models generate one image per invocation, unlike the single-invoke
    Amazon models, so token counts are accumulated instead of overwritten.

    Ref: stdapi/models/image/_stability.py:StabilityImageGenerationJobBase._get_image_from_response
    """

    async def test_tokens_sum_across_per_image_invocations(self) -> None:
        """Three per-image invoke() calls, one reporting no usage, still sum correctly."""
        results = iter(
            [
                InvokeResult(
                    response={"images": ["a"]}, input_tokens=10, output_tokens=20
                ),
                # A call reporting no usage must not zero the accumulator.
                InvokeResult(
                    response={"images": ["b"]}, input_tokens=None, output_tokens=None
                ),
                InvokeResult(
                    response={"images": ["c"]}, input_tokens=5, output_tokens=7
                ),
            ]
        )

        class _FakeModel:
            """Fake Stability model returning one queued InvokeResult per call."""

            async def invoke(self, _request: object) -> InvokeResult[dict[str, Any]]:
                """Return the next queued per-image invocation result."""
                return next(results)

        job = object.__new__(StabilityImageGenerationJobBase)
        job._model = cast("StabilityImageModelBase", _FakeModel())  # noqa: SLF001
        job._input_tokens = None  # noqa: SLF001
        job._output_tokens = None  # noqa: SLF001

        images = [
            await job._get_image_from_response({}, index)  # type: ignore[arg-type]  # noqa: SLF001
            for index in range(3)
        ]

        assert [image.image for image in images] == ["a", "b", "c"]
        assert [image.index for image in images] == [0, 1, 2]
        assert job.input_tokens == 15
        assert job.output_tokens == 27


class _StreamingJob(ImageGenerationJobBase[Any]):
    """Job whose backend returns images resolving in a controlled order."""

    async def _generate_images_from_text(
        self,
    ) -> list[Awaitable[ImageGenerationResponse]]:
        """Return a slow first image and an immediate second one."""
        return [_slow_image(0), _immediate_image(1)]

    async def _edit_image(
        self, images: list[str], mask: str | None
    ) -> list[Awaitable[ImageGenerationResponse]]:
        """Return one image per source, echoing the mask into the payload."""

        async def _edited(index: int) -> ImageGenerationResponse:
            return ImageGenerationResponse(image=f"{images[index]}-{mask}", index=index)

        return [_edited(index) for index in range(len(images))]


async def _passthrough_format(
    _self: object,
    response: Awaitable[ImageGenerationResponse] | ImageGenerationResponse,
) -> ImageGenerationResponse:
    """Resolve an image response without converting its format.

    The real method takes either an awaitable or an already-resolved response,
    so the stub standing in for it accepts both too.

    Args:
        _self: The job, unused.
        response: The awaitable or resolved image response.

    Returns:
        The resolved image response.
    """
    if isinstance(response, ImageGenerationResponse):
        return response
    return await response


async def _slow_image(index: int) -> ImageGenerationResponse:
    """Resolve after a yield to the event loop, so it completes second."""
    await sleep(0)
    return ImageGenerationResponse(image="slow", index=index)


async def _immediate_image(index: int) -> ImageGenerationResponse:
    """Resolve without suspending, so it completes first."""
    return ImageGenerationResponse(image="fast", index=index)


class TestGenericImageStreamFallback:
    """Models without a native streaming API still serve ``stream=true``.

    No image backend streams partial images, so both streaming routes run on
    the generic fallback: the per-image jobs are started together and each
    image is emitted as it completes, rather than after the slowest one.

    Ref: https://developers.openai.com/api/docs/api-reference/images/create
         stdapi/models/image/__init__.py:ImageGenerationJobBase._generate_images_stream
         stdapi/models/image/__init__.py:ImageGenerationJobBase._edit_images_stream
    """

    @staticmethod
    def _job() -> _StreamingJob:
        """Build a streaming job that leaves the backend format untouched."""
        return _StreamingJob(
            model=cast("Any", None),
            prompt="a cat",
            count=2,
            width=64,
            height=64,
            quality=None,
            style=None,
            output_format=None,
            output_compression=0,
            extra_params={},
        )

    async def test_generation_stream_emits_images_in_completion_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The image finishing first is streamed first, not the one requested first."""
        monkeypatch.setattr(
            ImageGenerationJobBase, "_ensure_image_output_format", _passthrough_format
        )

        images = [image async for image in self._job().generate_images_stream()]

        assert [image.image for image in images] == ["fast", "slow"]
        assert [image.index for image in images] == [1, 0]

    async def test_edit_stream_forwards_the_sources_and_the_mask(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The edit stream reaches ``_edit_image`` with both the images and the mask."""
        monkeypatch.setattr(
            ImageGenerationJobBase, "_ensure_image_output_format", _passthrough_format
        )

        images = [
            image
            async for image in self._job().edit_images_stream(["src0", "src1"], "mask")
        ]

        assert sorted(image.image for image in images) == ["src0-mask", "src1-mask"]


class _ConvertingStreamJob(ImageGenerationJobBase[Any]):
    """Job whose backend answers with one real PNG, so conversion runs for real."""

    async def _generate_images_from_text(
        self,
    ) -> list[Awaitable[ImageGenerationResponse]]:
        """Return a single already-resolved PNG image."""

        async def _png() -> ImageGenerationResponse:
            return ImageGenerationResponse(image=_b64_noisy_rgb_png(), index=0)

        return [_png()]


class TestStreamConvertsEachImageOnce:
    """A streamed image is re-encoded once, not once per pipeline stage.

    The conversion belongs to the stage that resolves the backend jobs, so the
    public stream must not repeat it: a second pass would re-encode an image
    that already carries the requested format, losing quality on every lossy
    output and spending the encoder twice per image.

    Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase.generate_images_stream
         stdapi/models/image/__init__.py:ImageGenerationJobBase._stream_completed_images
    """

    async def test_a_streamed_image_is_encoded_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Exactly one ``convert_base64_image`` call serves one streamed image."""
        calls: list[str] = []

        async def _counting_convert(
            image: str, output_format: str, compression: int
        ) -> tuple[str, int, int]:
            """Record the requested format, then convert for real."""
            calls.append(output_format)
            return await convert_base64_image(
                image, output_format=cast("Any", output_format), compression=compression
            )

        monkeypatch.setattr(
            "stdapi.models.image.convert_base64_image", _counting_convert
        )
        job = _ConvertingStreamJob(
            model=cast("Any", None),
            prompt="a cat",
            count=1,
            width=64,
            height=64,
            quality=None,
            style=None,
            output_format="jpeg",
            output_compression=50,
            extra_params={},
        )

        images = [image async for image in job.generate_images_stream()]

        assert calls == ["jpeg"]
        assert pybase64_b64decode(images[0].image).startswith(b"\xff\xd8\xff")


class TestUrlResponseFormatUpload:
    """``response_format="url"`` uploads the image and returns its link instead.

    Ref: https://developers.openai.com/api/docs/api-reference/images/create
         stdapi/models/image/__init__.py:ImageGenerationJobBase._get_image_url
    """

    @pytest.fixture(autouse=True)
    def _request_context(self, request_log: dict[str, Any]) -> Generator[None]:
        """Provide the request ID the uploaded object key is built from."""
        id_token = REQUEST_ID.set("img7")
        yield
        REQUEST_ID.reset(id_token)

    @staticmethod
    def _job(output_format: str | None) -> ImageGenerationJobBase[Any]:
        """Build a URL-returning job producing *output_format*."""
        return ImageGenerationJobBase(
            model=cast("Any", None),
            prompt="a cat",
            count=1,
            width=64,
            height=64,
            quality=None,
            style=None,
            output_format=cast("Any", output_format),
            output_compression=0,
            extra_params={},
            is_url=True,
        )

    @staticmethod
    def _capture_upload(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Replace the S3 upload with a recorder returning a fixed URL."""
        captured: dict[str, Any] = {}

        async def _fake_put(data: bytes, content_type: str, key: str) -> str:
            captured["data"] = data
            captured["content_type"] = content_type
            captured["key"] = key
            return f"https://example.invalid/{key}"

        monkeypatch.setattr(stdapi.models.image, "put_object_and_get_url", _fake_put)
        return captured

    async def test_base64_payload_is_replaced_by_the_uploaded_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The response carries the URL, and the upload the decoded image bytes.

        The base64 string never reaches the client in URL mode, and what is
        stored is the binary image, not its encoding.
        """
        captured = self._capture_upload(monkeypatch)
        source = _b64_noisy_rgb_png()

        result = await self._job(None)._ensure_image_output_format(  # noqa: SLF001
            ImageGenerationResponse(image=source, index=0)
        )

        assert result.image == f"https://example.invalid/{captured['key']}"
        assert captured["data"] == pybase64_b64decode(source)
        assert captured["data"].startswith(b"\x89PNG")
        assert captured["content_type"] == "image/png"

    async def test_object_key_is_scoped_to_the_request_and_1_based(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The key namespaces the request ID and numbers images from 001.

        Two concurrent requests must never write the same object, and the
        JPEG extension is ``.jpg`` while the content type stays ``image/jpeg``.
        """
        captured = self._capture_upload(monkeypatch)

        await self._job("jpeg")._ensure_image_output_format(  # noqa: SLF001
            ImageGenerationResponse(image=_b64_noisy_rgb_png(), index=0)
        )

        assert captured["key"] == "img7/image-img7-001.jpg"
        assert captured["content_type"] == "image/jpeg"
        assert captured["data"].startswith(b"\xff\xd8\xff"), "not a JPEG payload"


class TestStabilityContentFilter:
    """A Stability response reporting a filter reason is a client-visible refusal.

    Stability answers 200 with a ``finish_reasons`` entry instead of failing the
    invocation, so an unchecked response would return a blank image as success.

    Ref: https://platform.stability.ai/docs/api-reference
         stdapi/models/image/_stability.py:StabilityImageGenerationJobBase._get_image_from_response
    """

    @staticmethod
    def _job(response: dict[str, Any]) -> StabilityImageGenerationJobBase:
        """Build a job whose single invocation returns *response*."""

        class _FakeModel:
            """Fake Stability model returning a fixed invocation result."""

            async def invoke(self, _request: object) -> InvokeResult[dict[str, Any]]:
                """Return the canned response with no usage reported."""
                return InvokeResult(
                    response=response, input_tokens=None, output_tokens=None
                )

        job = object.__new__(StabilityImageGenerationJobBase)
        job._model = cast("StabilityImageModelBase", _FakeModel())  # noqa: SLF001
        job._input_tokens = None  # noqa: SLF001
        job._output_tokens = None  # noqa: SLF001
        return job

    async def test_filtered_response_raises_naming_the_reason(self) -> None:
        """A non-empty finish reason aborts with a 400 quoting it."""
        job = self._job({"images": [""], "finish_reasons": ["CONTENT_FILTERED"]})

        with pytest.raises(ApiError, match="Request was filtered") as exc_info:
            await job._get_image_from_response({}, 0)  # type: ignore[arg-type]  # noqa: SLF001

        assert exc_info.value.status == 400
        assert "CONTENT_FILTERED" in str(exc_info.value)

    async def test_null_finish_reason_is_a_successful_image(self) -> None:
        """``finish_reasons: [null]`` is how a clean generation reports itself."""
        job = self._job({"images": ["aaa"], "finish_reasons": [None]})

        result = await job._get_image_from_response({}, 3)  # type: ignore[arg-type]  # noqa: SLF001

        assert result.image == "aaa"
        assert result.index == 3


class TestStabilityRequestShaping:
    """Stability takes an aspect ratio and its own output format, not a pixel size.

    Ref: https://platform.stability.ai/docs/api-reference
         stdapi/models/image/_stability.py:StabilityImageGenerationJobBase
    """

    @staticmethod
    def _job(**overrides: Any) -> StabilityImageGenerationJobBase:  # noqa: ANN401
        """Build a text-to-image Stability job with the given overrides."""
        params: dict[str, Any] = {
            "model": cast("Any", None),
            "prompt": "a cat",
            "count": 1,
            "width": 1024,
            "height": 1024,
            "quality": None,
            "style": None,
            "output_format": None,
            "output_compression": 0,
            "extra_params": {},
        }
        params.update(overrides)
        return StabilityImageGenerationJobBase(**params)

    @pytest.mark.parametrize(
        ("width", "height", "expected"),
        [
            (1024, 1024, "1:1"),
            (1920, 1080, "16:9"),
            (1080, 1920, "9:16"),
            (1536, 1024, "3:2"),
            # An unsupported ratio snaps to the nearest supported one.
            (1000, 1010, "1:1"),
            (2000, 900, "21:9"),
        ],
    )
    def test_size_is_mapped_to_the_closest_supported_aspect_ratio(
        self, width: int, height: int, expected: str
    ) -> None:
        """Any requested size resolves to a supported ratio instead of failing."""
        assert (
            StabilityImageGenerationJobBase._get_aspect_ratio(width, height)  # noqa: SLF001
            == expected
        )

    def test_unsupported_quality_and_style_are_dropped_with_a_warning(
        self, request_log: dict[str, Any]
    ) -> None:
        """Quality and style steer the backend; an unusable one must not fail the request.

        Every OpenAI client sends a default ``quality``, so refusing it would
        break them all; the request proceeds and the response echoes what was
        actually produced.
        """
        job = self._job(quality="high", style="vivid")

        request = job._build_text_to_image_base_request()  # noqa: SLF001

        assert request == {"prompt": "a cat", "output_format": "png"}
        warnings = "".join(map(str, request_log["error_detail"]))
        assert '"quality" is not supported' in warnings
        assert '"style" is not supported' in warnings

    def test_no_requested_format_is_produced_as_png(self) -> None:
        """Without an explicit format the backend is asked for PNG, the lossless one."""
        job = self._job(output_format=None)

        request = job._build_text_to_image_base_request()  # noqa: SLF001

        assert request["output_format"] == "png"
        assert job._response_output_format == "png"  # noqa: SLF001

    def test_supported_output_format_is_requested_from_the_backend(self) -> None:
        """A natively supported format is asked for, avoiding a re-encode."""
        job = self._job(output_format="jpeg")

        request = job._build_text_to_image_base_request()  # noqa: SLF001

        assert request["output_format"] == "jpeg"
        assert job._response_output_format == "jpeg"  # noqa: SLF001

    def test_extra_params_are_merged_into_the_request(self) -> None:
        """Model-specific extras (seed, negative_prompt) reach the backend request."""
        job = self._job(extra_params={"seed": 42, "negative_prompt": "blurry"})

        request = job._build_text_to_image_base_request()  # noqa: SLF001

        assert request["seed"] == 42
        assert request["negative_prompt"] == "blurry"


def _b64_noisy_rgb_png(size: int = 64) -> str:
    """Encode a deterministic noisy RGB PNG, whose JPEG size tracks the quality."""
    image = Image.new("RGB", (size, size))
    image.putdata(
        [
            ((x * 37) % 256, (y * 91) % 256, (x * y) % 256)
            for y in range(size)
            for x in range(size)
        ]
    )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return b64encode(buffer.getvalue()).decode()


class TestOutputCompressionReEncoding:
    """``output_compression`` is applied when the image is re-encoded, and only then.

    The parameter is documented as supported over 1-100% on generations and
    edits, but it only reaches the encoder through the format conversion: a
    backend already producing the requested format streams its bytes through
    untouched. Asserting the produced payload, rather than the value handed to
    the job, is what distinguishes a forwarded parameter from an applied one.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/models/image/__init__.py:ImageGenerationJobBase._ensure_image_output_format
         stdapi/utils.py:convert_base64_image
    """

    @staticmethod
    def _job(
        output_format: str, output_compression: int
    ) -> ImageGenerationJobBase[Any]:
        """Build a job converting into *output_format* at *output_compression*."""
        return ImageGenerationJobBase(
            model=cast("Any", None),
            prompt="a cat",
            count=1,
            width=64,
            height=64,
            quality=None,
            style=None,
            output_format=cast("Any", output_format),
            output_compression=output_compression,
            extra_params={},
        )

    async def test_lower_compression_produces_a_smaller_jpeg(self) -> None:
        """The same source image encodes to fewer bytes at a lower compression level."""
        source = _b64_noisy_rgb_png()

        best = await self._job("jpeg", 100)._ensure_image_output_format(  # noqa: SLF001
            ImageGenerationResponse(image=source, index=0)
        )
        lossy = await self._job("jpeg", 10)._ensure_image_output_format(  # noqa: SLF001
            ImageGenerationResponse(image=source, index=0)
        )

        best_bytes = pybase64_b64decode(best.image)
        lossy_bytes = pybase64_b64decode(lossy.image)
        assert best_bytes.startswith(b"\xff\xd8\xff"), "not a JPEG payload"
        assert lossy_bytes.startswith(b"\xff\xd8\xff"), "not a JPEG payload"
        assert len(lossy_bytes) < len(best_bytes)

    async def test_conversion_reports_the_size_of_the_produced_image(self) -> None:
        """The echoed size is measured on the re-encoded image, not on the request."""
        job = self._job("jpeg", 50)

        await job._ensure_image_output_format(  # noqa: SLF001
            ImageGenerationResponse(image=_b64_noisy_rgb_png(), index=0)
        )

        assert (job.width, job.height) == (64, 64)
        assert job.output_format == "jpeg"

    async def test_no_conversion_leaves_the_payload_untouched(self) -> None:
        """A backend image already in the requested format is not re-encoded.

        ``output_compression`` therefore has no effect on it, which is why a
        compression assertion has to be made on a converted payload.
        """
        source = _b64_noisy_rgb_png()
        job = self._job("png", 10)

        result = await job._ensure_image_output_format(  # noqa: SLF001
            ImageGenerationResponse(image=source, index=0)
        )

        assert result.image == source
        assert (job.width, job.height) == (64, 64)
