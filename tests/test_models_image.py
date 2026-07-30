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
from stdapi import usage
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
from tests.conftest import set_test_price

if TYPE_CHECKING:
    from collections.abc import Awaitable, Generator


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


@pytest.fixture(autouse=True)
def _usage_scope() -> Generator[None]:
    """Install fresh per-request usage/model-state/image-spec scopes for each test."""
    usage_token = usage.init_usage()
    state_token = usage.init_model_state()
    image_spec_token = IMAGE_SPEC.set("")
    yield
    usage.USAGE.reset(usage_token)
    usage.MODEL_STATE.reset(state_token)
    IMAGE_SPEC.reset(image_spec_token)


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

        async def _passthrough(
            _self: object, response: Awaitable[ImageGenerationResponse]
        ) -> ImageGenerationResponse:
            return await response

        monkeypatch.setattr(
            ImageGenerationJobBase, "_ensure_image_output_format", _passthrough
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
