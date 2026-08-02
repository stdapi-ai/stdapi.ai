"""AWS S3 multipart helpers run their independent calls concurrently (unit).

Stubbed S3 clients record call overlap and order in-process: no AWS call is
made. Each blocking stub only releases once the expected number of calls is
in flight, so a regression to sequential awaits fails the test's timeout
instead of hanging the run.

Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/mpuoverview.html
     https://docs.aws.amazon.com/AmazonS3/latest/API/API_UploadPartCopy.html
     stdapi/aws_s3.py
"""

from asyncio import Event, wait_for
from typing import Any, cast

import pytest
from botocore.exceptions import ClientError

from stdapi import aws_s3
from stdapi.aws_s3 import S3Object, copy_s3_object, multipart_copy_parts
from stdapi.utils import async_iter

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Timeout failing a test whose stub never sees the expected concurrency.
_OVERLAP_TIMEOUT: float = 5.0


class _BarrierCopyClient:
    """Stub S3 client whose part copies all block until *expected* are in flight."""

    def __init__(self, expected: int, *, size: int = 0) -> None:
        self.expected = expected
        self.size = size
        self.in_flight = 0
        self.max_in_flight = 0
        self.copy_ranges: dict[int, str] = {}
        self.completed_parts: list[dict[str, Any]] | None = None
        self.aborted = False
        self._all_started = Event()

    async def head_object(self, **_kwargs: object) -> dict[str, Any]:
        return {"ContentLength": self.size}

    async def create_multipart_upload(self, **_kwargs: object) -> dict[str, Any]:
        return {"UploadId": "mpu-1"}

    async def upload_part_copy(
        self,
        *,
        PartNumber: int,  # noqa: N803
        CopySourceRange: str,  # noqa: N803
        **_kwargs: object,
    ) -> dict[str, Any]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self.in_flight >= self.expected:
            self._all_started.set()
        # Times out (instead of hanging) if copies are issued one at a time.
        await wait_for(self._all_started.wait(), timeout=_OVERLAP_TIMEOUT)
        self.copy_ranges[PartNumber] = CopySourceRange
        self.in_flight -= 1
        return {"CopyPartResult": {"ETag": f'"etag-{PartNumber}"'}}

    async def complete_multipart_upload(
        self,
        *,
        MultipartUpload: dict[str, Any],  # noqa: N803
        **_kwargs: object,
    ) -> dict[str, Any]:
        self.completed_parts = MultipartUpload["Parts"]
        return {}

    async def abort_multipart_upload(self, **_kwargs: object) -> dict[str, Any]:
        self.aborted = True
        return {}


class _FailingCopyClient(_BarrierCopyClient):
    """``_BarrierCopyClient`` whose second part copy fails with a ClientError."""

    async def upload_part_copy(
        self,
        *,
        PartNumber: int,  # noqa: N803
        CopySourceRange: str,  # noqa: N803
        **kwargs: object,
    ) -> dict[str, Any]:
        if PartNumber == 2:
            raise ClientError(
                {"Error": {"Code": "SlowDown", "Message": "busy"}}, "UploadPartCopy"
            )
        return await super().upload_part_copy(
            PartNumber=PartNumber, CopySourceRange=CopySourceRange, **kwargs
        )


class TestMultipartCopyParts:
    """Ranged server-side copies fan out concurrently, bounded, in part order.

    Ref: https://docs.aws.amazon.com/AmazonS3/latest/API/API_UploadPartCopy.html
         stdapi/aws_s3.py:multipart_copy_parts
    """

    async def test_part_copies_overlap_and_stay_ordered(self) -> None:
        """All ranged copies run concurrently; the result stays part-ordered.

        Each stubbed copy blocks until all three are in flight, so this test
        times out if the parts regress to sequential awaits. The returned
        completion list must follow part numbers, which follow byte ranges.
        """
        stub = _BarrierCopyClient(expected=3)
        parts = await wait_for(
            multipart_copy_parts(
                cast("Any", stub),
                bucket="dest",
                key="dk",
                upload_id="mpu-1",
                copy_source={"Bucket": "src", "Key": "sk"},
                size=25,
                part_size=10,
            ),
            timeout=_OVERLAP_TIMEOUT,
        )
        assert parts == [
            {"PartNumber": 1, "ETag": '"etag-1"'},
            {"PartNumber": 2, "ETag": '"etag-2"'},
            {"PartNumber": 3, "ETag": '"etag-3"'},
        ]
        assert stub.copy_ranges == {
            1: "bytes=0-9",
            2: "bytes=10-19",
            3: "bytes=20-24",
        }, "each part number must map to its own byte range"
        assert stub.max_in_flight == 3

    async def test_part_copy_concurrency_is_bounded(self) -> None:
        """More parts than the bound never exceed the concurrency cap.

        The stub only releases once the cap is reached, so the recorded
        maximum proves the width is exactly ``MULTIPART_COPY_CONCURRENCY``:
        neither sequential (below) nor unbounded (above).
        """
        stub = _BarrierCopyClient(expected=aws_s3.MULTIPART_COPY_CONCURRENCY)
        parts = await wait_for(
            multipart_copy_parts(
                cast("Any", stub),
                bucket="dest",
                key="dk",
                upload_id="mpu-1",
                copy_source={"Bucket": "src", "Key": "sk"},
                size=200,
                part_size=10,
            ),
            timeout=_OVERLAP_TIMEOUT,
        )
        assert [part["PartNumber"] for part in parts] == list(range(1, 21))
        assert stub.max_in_flight == aws_s3.MULTIPART_COPY_CONCURRENCY


class TestCopyS3ObjectMultipart:
    """``copy_s3_object`` above the single-copy limit uses concurrent parts.

    Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/copy-object.html
         stdapi/aws_s3.py:copy_s3_object
    """

    @pytest.fixture(autouse=True)
    def _small_parts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Shrink the size thresholds so tests do not handle 5 GiB objects."""
        monkeypatch.setattr(aws_s3, "_COPY_OBJECT_MAX_BYTES", 20)
        monkeypatch.setattr(aws_s3, "_MULTIPART_COPY_PART_SIZE", 10)

    async def test_multipart_copy_completes_with_ordered_parts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The multipart branch overlaps part copies and completes in order."""
        stub = _BarrierCopyClient(expected=3, size=25)
        monkeypatch.setattr(aws_s3, "get_client", lambda *_: stub)
        result = await wait_for(
            copy_s3_object("src", "sk", dest_bucket="dest", dest_key="dk"),
            timeout=_OVERLAP_TIMEOUT,
        )
        assert result == S3Object(bucket="dest", key="dk")
        assert stub.completed_parts is not None
        assert [part["PartNumber"] for part in stub.completed_parts] == [1, 2, 3]
        assert stub.max_in_flight == 3
        assert stub.aborted is False

    async def test_part_failure_aborts_upload_and_propagates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing part copy aborts the multipart upload and re-raises as-is."""
        stub = _FailingCopyClient(expected=1, size=25)
        monkeypatch.setattr(aws_s3, "get_client", lambda *_: stub)
        with pytest.raises(ClientError) as exc_info:
            await wait_for(
                copy_s3_object("src", "sk", dest_bucket="dest", dest_key="dk"),
                timeout=_OVERLAP_TIMEOUT,
            )
        assert exc_info.value.response["Error"]["Code"] == "SlowDown"
        assert stub.aborted is True
        assert stub.completed_parts is None


class _PipelinedUploadClient:
    """Stub S3 client proving upload/read overlap for ``_multipart_upload``."""

    def __init__(self) -> None:
        self.part_bodies: dict[int, bytes] = {}
        self.in_flight = 0
        self.max_in_flight = 0
        self.completed_parts: list[dict[str, Any]] | None = None
        self.aborted = False
        self.window_full = Event()

    async def create_multipart_upload(self, **_kwargs: object) -> dict[str, Any]:
        return {"UploadId": "mpu-1"}

    async def upload_part(
        self,
        *,
        PartNumber: int,  # noqa: N803
        Body: bytes,  # noqa: N803
        **_kwargs: object,
    ) -> dict[str, Any]:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        if self.in_flight >= 2:
            self.window_full.set()
        if PartNumber == 1:
            # Completes only once part 2 also uploads: the old sequential
            # read-upload-read-upload loop times out here.
            await wait_for(self.window_full.wait(), timeout=_OVERLAP_TIMEOUT)
        self.part_bodies[PartNumber] = bytes(Body)
        self.in_flight -= 1
        return {"ETag": f'"etag-{PartNumber}"'}

    async def complete_multipart_upload(
        self,
        *,
        MultipartUpload: dict[str, Any],  # noqa: N803
        **_kwargs: object,
    ) -> dict[str, Any]:
        self.completed_parts = MultipartUpload["Parts"]
        return {}

    async def abort_multipart_upload(self, **_kwargs: object) -> dict[str, Any]:
        self.aborted = True
        return {}


class _FailingUploadClient(_PipelinedUploadClient):
    """``_PipelinedUploadClient`` whose second part upload fails."""

    async def upload_part(
        self,
        *,
        PartNumber: int,  # noqa: N803
        Body: bytes,  # noqa: N803
        **kwargs: object,
    ) -> dict[str, Any]:
        if PartNumber == 2:
            # Release part 1's barrier before failing so only the error path
            # is under test.
            self.window_full.set()
            raise ClientError(
                {"Error": {"Code": "InternalError", "Message": "boom"}}, "UploadPart"
            )
        return await super().upload_part(PartNumber=PartNumber, Body=Body, **kwargs)


class TestMultipartUploadPipelining:
    """``_multipart_upload`` reads ahead while parts upload, window-bounded.

    Ref: https://docs.aws.amazon.com/AmazonS3/latest/API/API_UploadPart.html
         stdapi/aws_s3.py:_multipart_upload
    """

    async def test_reads_next_chunk_while_part_uploads(self) -> None:
        """The next chunk is read while the previous part uploads.

        Part 1 completes only once part 2 is in flight, so the old sequential
        loop deadlocks (times out) here. The completion list stays part-number
        ordered even though part 2 finishes before part 1, and the in-flight
        window never exceeds the documented bound of 2.
        """
        stub = _PipelinedUploadClient()
        await wait_for(
            aws_s3._multipart_upload(  # noqa: SLF001
                cast("Any", stub),
                "bucket",
                "key",
                async_iter(b"aa", b"bb", b"cc", b"dd"),
            ),
            timeout=_OVERLAP_TIMEOUT,
        )
        assert stub.completed_parts == [
            {"PartNumber": number, "ETag": f'"etag-{number}"'}
            for number in (1, 2, 3, 4)
        ]
        assert stub.part_bodies == {1: b"aa", 2: b"bb", 3: b"cc", 4: b"dd"}, (
            "part numbers must follow the chunk read order"
        )
        assert stub.max_in_flight == aws_s3._UPLOAD_PARTS_IN_FLIGHT  # noqa: SLF001
        assert stub.aborted is False

    async def test_part_failure_aborts_upload_and_propagates(self) -> None:
        """A failing part upload aborts the multipart upload and re-raises as-is."""
        stub = _FailingUploadClient()
        with pytest.raises(ClientError) as exc_info:
            await wait_for(
                aws_s3._multipart_upload(  # noqa: SLF001
                    cast("Any", stub), "bucket", "key", async_iter(b"aa", b"bb", b"cc")
                ),
                timeout=_OVERLAP_TIMEOUT,
            )
        assert exc_info.value.response["Error"]["Code"] == "InternalError"
        assert stub.aborted is True
        assert stub.completed_parts is None


class TestBytesChunks:
    """``_bytes_chunks`` slices in-memory payloads into botocore-safe parts.

    Ref: https://docs.aws.amazon.com/AmazonS3/latest/API/API_UploadPart.html
         stdapi/aws_s3.py:_bytes_chunks
    """

    async def test_yields_real_bytes_parts_losslessly(self) -> None:
        """Every chunk is exact ``bytes`` and the parts reassemble the payload.

        botocore's flexible-checksum body wrapper only accepts ``bytes`` or
        ``bytearray`` part bodies (a ``memoryview`` crashes at send time with
        ``AttributeError: 'memoryview' object has no attribute 'read'``), so
        the chunk type is a hard contract, not an implementation detail.
        """
        data = bytes(range(256)) * 40
        chunks = [chunk async for chunk in aws_s3._bytes_chunks(data, 4096)]  # noqa: SLF001
        assert all(type(chunk) is bytes for chunk in chunks)
        assert [len(chunk) for chunk in chunks] == [4096, 4096, 2048]
        assert b"".join(chunks) == data
