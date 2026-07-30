"""Unit tests for input-media usage recording in the Nova and Marengo embedding models.

Covers the ``record_bedrock_usage(input_images=..., input_seconds=..., media_spec=...)``
call sites in ``amazon_nova_embed.EmbeddingModel`` and
``twelvelabs_marengo_embed.EmbeddingModel`` -- exercised either through the real
embed methods with a stubbed ``invoke``/``invoke_async``, or directly against the
extracted usage-recording helpers when the surrounding method also touches
S3/region plumbing unrelated to billing.

Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
     stdapi/usage.py:record_bedrock_usage
     stdapi/pricing.py:Dimension
"""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from stdapi import usage
from stdapi.models import InvokeResult
from stdapi.models.embedding.amazon_nova_embed import (
    EmbeddingModel as NovaEmbeddingModel,
)
from stdapi.models.embedding.amazon_nova_embed import (
    _EmbeddingParams,
    _ImageInput,
    _MediaSource,
    _SegmentedEmbeddingData,
    _SegmentedEmbeddingParams,
    _SegmentMetadata,
)
from stdapi.models.embedding.twelvelabs_marengo_embed import (
    EmbeddingModel as MarengoEmbeddingModel,
)
from stdapi.pricing import Dimension

if TYPE_CHECKING:
    from collections.abc import Generator

    from stdapi.usage import UsageRecord


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


_NOVA_MODEL_ID = "amazon.nova-2-multimodal-embeddings-v1:0"
_MARENGO_MODEL_ID = "twelvelabs.marengo-embed-3-0-v1:0"


@pytest.fixture(autouse=True)
def _usage_scope() -> Generator[None]:
    """Install fresh per-request usage/model-state scopes for each test."""
    usage_token = usage.init_usage()
    state_token = usage.init_model_state()
    yield
    usage.USAGE.reset(usage_token)
    usage.MODEL_STATE.reset(state_token)


def _only_record() -> UsageRecord:
    """Return the single usage record recorded so far, failing if not exactly one."""
    records = list(usage.USAGE.get().values())
    assert len(records) == 1
    return records[0]


class TestNovaSingleEmbeddingImageUsage:
    """``_embed_single``: image inputs record ``INPUT_IMAGES`` usage.

    Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
         stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel._record_media_usage
    """

    async def test_standard_image_records_one_input_image_with_no_spec(self) -> None:
        """A plain image (no detailLevel) bills as a flat input image.

        Ref: stdapi/usage.py:record_bedrock_usage
        """
        model = NovaEmbeddingModel(_NOVA_MODEL_ID)
        model.invoke = AsyncMock(  # type: ignore[method-assign]
            return_value=InvokeResult(response={"embeddings": [{"embedding": [0.1]}]})
        )
        await model._embed_single(  # noqa: SLF001
            value="base64data",
            media_type="image",
            file_format="png",
            base_params=_EmbeddingParams(embeddingPurpose="GENERIC_INDEX"),
            extra_params={},
            region=None,
        )
        record = _only_record()
        assert record.model == _NOVA_MODEL_ID
        assert record.quantities[Dimension.INPUT_IMAGES] == 1
        assert Dimension.INPUT_SECONDS not in record.quantities
        assert record.input_images_by_spec == {}

    async def test_document_image_records_document_spec(self) -> None:
        """``detailLevel=DOCUMENT_IMAGE`` bills under the "document" spec bucket.

        Document images are priced separately from ordinary images, so the extra
        per-media parameter has to reach the usage record as a spec bucket.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel._add_extra_params
        """
        model = NovaEmbeddingModel(_NOVA_MODEL_ID)
        model.invoke = AsyncMock(  # type: ignore[method-assign]
            return_value=InvokeResult(response={"embeddings": [{"embedding": [0.1]}]})
        )
        await model._embed_single(  # noqa: SLF001
            value="base64data",
            media_type="image",
            file_format="png",
            base_params=_EmbeddingParams(embeddingPurpose="GENERIC_INDEX"),
            extra_params={"image": {"detailLevel": "DOCUMENT_IMAGE"}},
            region=None,
        )
        record = _only_record()
        assert record.quantities[Dimension.INPUT_IMAGES] == 1
        assert record.input_images_by_spec == {"document": 1}

    async def test_text_input_records_no_input_images(self) -> None:
        """Non-image media types never record ``INPUT_IMAGES``.

        Text embedding is billed on tokens reported by Bedrock, so the media
        accounting must stay empty even though the invocation happened.

        Ref: stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel._record_media_usage
        """
        model = NovaEmbeddingModel(_NOVA_MODEL_ID)
        model.invoke = AsyncMock(  # type: ignore[method-assign]
            return_value=InvokeResult(response={"embeddings": [{"embedding": [0.1]}]})
        )
        await model._embed_single(  # noqa: SLF001
            value="hello",
            media_type="text",
            file_format="plain",
            base_params=_EmbeddingParams(embeddingPurpose="GENERIC_INDEX"),
            extra_params={},
            region=None,
        )
        model.invoke.assert_awaited_once()
        assert not usage.USAGE.get()


class TestNovaSegmentedEmbeddingUsage:
    """``_record_media_usage``: billed input-media quantities for SEGMENTED_EMBEDDING jobs.

    Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/embeddings.html
         stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel._embed_segmented
    """

    def test_image_records_one_input_image_with_no_spec(self) -> None:
        """A segmented image without detailLevel bills as a flat input image.

        Ref: stdapi/usage.py:record_bedrock_usage
        """
        model = NovaEmbeddingModel(_NOVA_MODEL_ID)
        params = _SegmentedEmbeddingParams(
            embeddingPurpose="GENERIC_INDEX",
            image=_ImageInput(source=_MediaSource(bytes="abc"), format="png"),
        )
        model._record_media_usage("image", params, [])  # noqa: SLF001
        record = _only_record()
        assert record.model == _NOVA_MODEL_ID
        assert record.quantities[Dimension.INPUT_IMAGES] == 1
        assert Dimension.INPUT_SECONDS not in record.quantities
        assert record.input_images_by_spec == {}

    def test_explicit_region_wins_over_shared_model_state(self) -> None:
        """The media record carries its own call's region, not a sibling's overwrite.

        Concurrent calls to the same model share one model-state entry, so the
        serving Region has to be passed per call rather than read back from it.

        Ref: stdapi/usage.py:record_bedrock_usage
        """
        model = NovaEmbeddingModel(_NOVA_MODEL_ID)
        usage.get_model_state(_NOVA_MODEL_ID).region = "eu-west-3"
        params = _SegmentedEmbeddingParams(
            embeddingPurpose="GENERIC_INDEX",
            image=_ImageInput(source=_MediaSource(bytes="abc"), format="png"),
        )
        model._record_media_usage(  # noqa: SLF001
            "image", params, [], region="us-east-1", routing=""
        )
        (key,) = usage.USAGE.get()
        assert key.region == "us-east-1"
        assert _only_record().quantities[Dimension.INPUT_IMAGES] == 1

    def test_document_image_records_document_spec(self) -> None:
        """A segmented image with ``detailLevel=DOCUMENT_IMAGE`` bills under "document".

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
        """
        model = NovaEmbeddingModel(_NOVA_MODEL_ID)
        params = _SegmentedEmbeddingParams(
            embeddingPurpose="GENERIC_INDEX",
            image=_ImageInput(
                source=_MediaSource(bytes="abc"),
                format="png",
                detailLevel="DOCUMENT_IMAGE",
            ),
        )
        model._record_media_usage("image", params, [])  # noqa: SLF001
        record = _only_record()
        assert record.quantities[Dimension.INPUT_IMAGES] == 1
        assert record.input_images_by_spec == {"document": 1}

    @pytest.mark.parametrize("media_type", ["audio", "video"])
    def test_media_bills_ceil_of_max_segment_end_seconds(self, media_type: str) -> None:
        """Billed seconds are the ceiling of the latest segment's end time.

        A segmented job returns one JSONL entry per segment with its own
        ``segmentEndSeconds``; the media duration is the last segment's end time,
        not the sum of the segments, and partial seconds are rounded up.

        Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/embeddings.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel._record_media_usage
        """
        model = NovaEmbeddingModel(_NOVA_MODEL_ID)
        entries = [
            _SegmentedEmbeddingData(
                embedding=[0.1],
                segmentMetadata=_SegmentMetadata(segmentIndex=0, segmentEndSeconds=4.2),
                status="SUCCESS",
            ),
            _SegmentedEmbeddingData(
                embedding=[0.1],
                segmentMetadata=_SegmentMetadata(segmentIndex=1, segmentEndSeconds=9.1),
                status="SUCCESS",
            ),
        ]
        model._record_media_usage(  # noqa: SLF001
            media_type,  # type: ignore[arg-type]
            _SegmentedEmbeddingParams(embeddingPurpose="GENERIC_INDEX"),
            entries,
        )
        record = _only_record()
        assert record.quantities[Dimension.INPUT_SECONDS] == 10  # ceil(9.1)
        assert Dimension.INPUT_IMAGES not in record.quantities
        assert record.input_seconds_by_spec == {media_type: 10}

    def test_media_without_segment_end_seconds_records_nothing(self) -> None:
        """No ``segmentEndSeconds`` in any entry means no usage is recorded.

        Text segments carry character positions instead of timings, and a failed
        segment may carry no metadata at all, so an unknown duration must not be
        billed as zero seconds.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel._record_media_usage
        """
        model = NovaEmbeddingModel(_NOVA_MODEL_ID)
        entries = [
            _SegmentedEmbeddingData(
                embedding=[0.1],
                segmentMetadata=_SegmentMetadata(segmentIndex=0),
                status="SUCCESS",
            )
        ]
        model._record_media_usage(  # noqa: SLF001
            "audio",
            _SegmentedEmbeddingParams(embeddingPurpose="GENERIC_INDEX"),
            entries,
        )
        assert not usage.USAGE.get()


class TestMarengoTextImageUsage:
    """``_embed_text_image``: the text_image combined mode records one input image.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
         stdapi/models/embedding/twelvelabs_marengo_embed.py:EmbeddingModel._embed_text_image
    """

    async def test_text_image_records_one_input_image(self) -> None:
        """The image half of a text_image pair bills as a flat input image.

        The fused ``text_image`` call embeds one image and one caption, so it is
        billed exactly like a plain image call despite the different input type.

        Ref: stdapi/usage.py:record_bedrock_usage
        """
        model = MarengoEmbeddingModel(_MARENGO_MODEL_ID)
        model.invoke = AsyncMock(  # type: ignore[method-assign]
            return_value=InvokeResult(response={"data": [{"embedding": [0.1]}]})
        )
        await model._embed_text_image(  # noqa: SLF001
            value="base64image",  # type: ignore[arg-type]
            image_text="a sunset",
            extra_params={},
        )
        record = _only_record()
        assert record.model == _MARENGO_MODEL_ID
        assert record.quantities[Dimension.INPUT_IMAGES] == 1
        assert Dimension.INPUT_SECONDS not in record.quantities
        assert record.input_images_by_spec == {}


class TestMarengoRecordMediaUsage:
    """``_record_media_usage``: billed input-media quantities per response shape.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html
         stdapi/models/embedding/twelvelabs_marengo_embed.py:EmbeddingModel._record_media_usage
    """

    def test_image_records_one_input_image(self) -> None:
        """An image response records one flat input image.

        Ref: stdapi/usage.py:record_bedrock_usage
        """
        model = MarengoEmbeddingModel(_MARENGO_MODEL_ID)
        model._record_media_usage(  # noqa: SLF001
            "image", {"data": [{"embedding": [0.1]}]}
        )
        record = _only_record()
        assert record.model == _MARENGO_MODEL_ID
        assert record.quantities[Dimension.INPUT_IMAGES] == 1
        assert Dimension.INPUT_SECONDS not in record.quantities
        assert record.input_images_by_spec == {}

    @pytest.mark.parametrize("media_type", ["audio", "video"])
    def test_media_bills_ceil_of_max_end_sec(self, media_type: str) -> None:
        """Billed seconds are the ceiling of the latest entry's ``endSec``.

        Marengo returns one entry per clip with ``startSec``/``endSec``, so the
        billable duration is the last clip's end time rounded up -- not the sum of
        the clip lengths, which would double-count overlapping scopes.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
             stdapi/models/embedding/twelvelabs_marengo_embed.py:EmbeddingModel._record_media_usage
        """
        model = MarengoEmbeddingModel(_MARENGO_MODEL_ID)
        response = {
            "data": [
                {"embedding": [0.1], "startSec": 0.0, "endSec": 5.5},
                {"embedding": [0.1], "startSec": 5.5, "endSec": 12.3},
            ]
        }
        model._record_media_usage(media_type, response)  # type: ignore[arg-type] # noqa: SLF001
        record = _only_record()
        assert record.quantities[Dimension.INPUT_SECONDS] == 13  # ceil(12.3)
        assert Dimension.INPUT_IMAGES not in record.quantities
        assert record.input_seconds_by_spec == {media_type: 13}

    def test_media_without_end_sec_records_nothing(self) -> None:
        """A response with no ``endSec`` on any entry records no usage.

        Asset-scoped embeddings can come back without clip timings, and an unknown
        duration must not be billed as zero seconds.

        Ref: stdapi/models/embedding/twelvelabs_marengo_embed.py:EmbeddingModel._record_media_usage
        """
        model = MarengoEmbeddingModel(_MARENGO_MODEL_ID)
        response = {"data": [{"embedding": [0.1]}]}
        model._record_media_usage("audio", response)  # type: ignore[arg-type] # noqa: SLF001
        assert not usage.USAGE.get()

    def test_explicit_region_wins_over_shared_model_state(self) -> None:
        """The media record carries its own call's region, not a sibling's overwrite.

        Concurrent calls to the same model share one model-state entry, so the
        serving Region has to be passed per call rather than read back from it.

        Ref: stdapi/usage.py:record_bedrock_usage
        """
        model = MarengoEmbeddingModel(_MARENGO_MODEL_ID)
        usage.get_model_state(_MARENGO_MODEL_ID).region = "eu-west-3"
        model._record_media_usage(  # noqa: SLF001
            "image", {"data": [{"embedding": [0.1]}]}, region="us-east-1", routing=""
        )
        (key,) = usage.USAGE.get()
        assert key.region == "us-east-1"
        assert _only_record().quantities[Dimension.INPUT_IMAGES] == 1
