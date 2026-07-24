"""Unit tests for image models: ``image_spec`` pricing buckets and invoke-path billing."""

from asyncio import CancelledError, Event, sleep, wait_for
from decimal import Decimal
from typing import TYPE_CHECKING, Any

import pytest

import stdapi.models
from stdapi import usage
from stdapi.models import InvokeResult
from stdapi.models.image import (
    ImageGenerationJobBase,
    ImageGenerationResponse,
    ImageModelBase,
)
from stdapi.models.image._stability import StabilityImageGenerationJobBase
from stdapi.models.image.amazon_titan_image_generator import image_spec
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
    """Titan Image Generator default bucketing (tiers=2, low_tier_max=512)."""

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
        """The larger of width/height picks the smallest tier that covers it."""
        assert image_spec(width, height, "standard") == (
            f"{expected_resolution}:standard"
        )


class TestImageSpecNovaCanvasTiers:
    """Nova Canvas bucketing (tiers=3, low_tier_max=1024, up to 4096px)."""

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
    """Quality propagation into the "<resolution>:<quality>" label."""

    @pytest.mark.parametrize("quality", ["standard", "premium"])
    def test_quality_is_propagated_into_the_spec_label(self, quality: str) -> None:
        """The resolved AWS quality is appended verbatim after the resolution."""
        assert image_spec(1024, 1024, quality) == f"1024:{quality}"  # type: ignore[arg-type]

    def test_missing_quality_defaults_to_standard(self) -> None:
        """A None quality falls back to the "standard" pricing label."""
        assert image_spec(512, 512, None) == "512:standard"


class _FakeBody:
    """Minimal async body mimicking a botocore streaming response body."""

    def __init__(self, payload: bytes) -> None:
        """Store the raw JSON payload to return from read()."""
        self._payload = payload

    async def read(self) -> bytes:
        """Return the stored payload."""
        return self._payload


class TestInvokeImageBilling:
    """ImageModelBase.invoke(): a stubbed InvokeModel response drives image billing."""

    async def test_stubbed_invoke_bills_images_by_spec_region_and_price(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two returned images bill output_images == 2 in the serving region at the spec rate."""

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
        assert record.quantities[Dimension.OUTPUT_IMAGES] == 2
        assert record.output_images_by_spec == {"1024:standard": 2}
        # The override clears IMAGE_SPEC to avoid leaking into later calls.
        assert IMAGE_SPEC.get() == ""
        compute_costs()
        # 10 * 0.000001 (tokens) + 2 * 0.06 (spec rate, not the 0.9 flat rate).
        assert record.cost == Decimal("0.120010")


class TestStreamCancelsAbandonedJobs:
    """_stream_completed_images: an early close cancels the in-flight image jobs."""

    async def test_early_close_cancels_pending_tasks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closing the stream after one image must cancel the remaining jobs."""
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
    """StabilityImageGenerationJobBase: n>1 fan-out sums tokens across per-image calls."""

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
        job._model = _FakeModel()  # noqa: SLF001
        job._input_tokens = None  # noqa: SLF001
        job._output_tokens = None  # noqa: SLF001

        for index in range(3):
            await job._get_image_from_response({}, index)  # type: ignore[arg-type]  # noqa: SLF001

        assert job.input_tokens == 15
        assert job.output_tokens == 27
