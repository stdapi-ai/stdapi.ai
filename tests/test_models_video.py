"""Bedrock video generation models: dispatch, capabilities, async invoke, usage."""

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError

import stdapi.main  # noqa: F401  (registers every route capability)
from stdapi.api_errors import ApiError, UnsupportedModelError
from stdapi.config import SETTINGS
from stdapi.models import ModelDetails, _compute_model_capabilities, video
from stdapi.models.video import (
    ReferenceImage,
    VideoJob,
    delete_video_output,
    get_video_job,
    get_video_model,
    open_video_content,
)
from stdapi.models.video.amazon_nova_reel import VideoModel as NovaReelModel
from stdapi.models.video.luma_ray import VideoModel as LumaRayModel
from stdapi.monitoring import REQUEST_ID, REQUEST_LOG, EventLog
from stdapi.pricing import Dimension
from stdapi.usage import USAGE, init_model_state, init_usage

if TYPE_CHECKING:
    from collections.abc import Generator

    from types_aiobotocore_bedrock.literals import RegionName


#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


VIDEO_MODELS = ("amazon.nova-reel-v1:0", "amazon.nova-reel-v1:1", "luma.ray-v2:0")

#: A well-formed async invocation ARN in the primary test region.
_ARN = "arn:aws:bedrock:us-east-1:000000000000:async-invoke/abc123xyz"


class TestVideoModelDispatch:
    """Video model IDs must resolve to their provider backend class."""

    @pytest.mark.parametrize(
        ("model_id", "model_class"),
        [
            ("amazon.nova-reel-v1:0", NovaReelModel),
            ("amazon.nova-reel-v1:1", NovaReelModel),
            ("luma.ray-v2:0", LumaRayModel),
        ],
    )
    def test_matcher_dispatch(self, model_id: str, model_class: type) -> None:
        """The registry resolves video model IDs to the provider class."""
        assert type(get_video_model(model_id)) is model_class

    def test_non_video_model_is_rejected(self) -> None:
        """Non-video model IDs have no video backend."""
        with pytest.raises(UnsupportedModelError):
            get_video_model("anthropic.claude-3-5-haiku-20241022-v1:0")


class TestVideoSupportedRoutes:
    """Video models advertise the videos route only; text models never do."""

    @pytest.mark.parametrize("model_id", VIDEO_MODELS)
    def test_video_route_advertised(self, model_id: str) -> None:
        """supported_routes includes /v1/videos and excludes text routes."""
        details = ModelDetails(
            id=model_id,
            name=model_id,
            provider="Vendor",
            input_modalities=["TEXT", "IMAGE"],
            output_modalities=["VIDEO"],
            regions=["us-east-1"],
        )
        routes, tools = _compute_model_capabilities(model_id, details)
        assert any(route.endswith("/v1/videos") for route in routes)
        assert "openai_video_generation" in tools
        assert not any("chat/completions" in route for route in routes)
        assert not any("images" in route for route in routes)

    def test_text_models_do_not_advertise_videos(self) -> None:
        """A model without VIDEO output modality skips the route."""
        model_id = "anthropic.claude-3-5-haiku-20241022-v1:0"
        details = ModelDetails(
            id=model_id,
            name=model_id,
            provider="Vendor",
            input_modalities=["TEXT"],
            output_modalities=["TEXT"],
            regions=["us-east-1"],
        )
        _, tools = _compute_model_capabilities(model_id, details)
        assert "openai_video_generation" not in tools


class TestNovaReelInput:
    """Nova Reel modelInput building and parameter validation."""

    @staticmethod
    def _build(model_id: str = "amazon.nova-reel-v1:1", **kwargs: Any) -> Any:  # noqa: ANN401
        params: dict[str, Any] = {
            "seconds": 6,
            "size": "1280x720",
            "reference_image": None,
            "extra_params": {},
        }
        params.update(kwargs)
        return NovaReelModel(model_id).build_generation_input("a cat", **params)

    def test_single_shot_text_to_video(self) -> None:
        """A 6-second request builds a TEXT_VIDEO task."""
        body = self._build()
        assert body == {
            "taskType": "TEXT_VIDEO",
            "textToVideoParams": {"text": "a cat"},
            "videoGenerationConfig": {
                "durationSeconds": 6,
                "fps": 24,
                "dimension": "1280x720",
            },
        }

    def test_multi_shot_for_longer_durations(self) -> None:
        """Durations over 6 seconds build a MULTI_SHOT_AUTOMATED task."""
        body = self._build(seconds=24)
        assert body["taskType"] == "MULTI_SHOT_AUTOMATED"
        assert body["multiShotAutomatedParams"] == {"text": "a cat"}
        assert body["videoGenerationConfig"]["durationSeconds"] == 24

    @pytest.mark.parametrize("seconds", [1, 7, 11, 126])
    def test_unsupported_durations_are_rejected(self, seconds: int) -> None:
        """Durations that are not a multiple of 6 up to 120 are rejected."""
        with pytest.raises(ApiError, match="multiple of 6"):
            self._build(seconds=seconds)

    def test_v1_0_single_shot_duration_is_accepted(self) -> None:
        """A 6-second request on v1:0 builds a TEXT_VIDEO task."""
        body = self._build(model_id="amazon.nova-reel-v1:0")
        assert body["taskType"] == "TEXT_VIDEO"

    def test_v1_0_multi_shot_duration_is_rejected(self) -> None:
        """v1:0 has no multi-shot support; only 6-second videos are allowed."""
        with pytest.raises(
            ApiError, match=r"'seconds' must be 6 for model 'amazon\.nova-reel-v1:0'"
        ):
            self._build(model_id="amazon.nova-reel-v1:0", seconds=12)

    def test_v1_1_multi_shot_duration_is_accepted(self) -> None:
        """A 12-second request on v1:1 builds a MULTI_SHOT_AUTOMATED task."""
        body = self._build(seconds=12)
        assert body["taskType"] == "MULTI_SHOT_AUTOMATED"

    @pytest.mark.parametrize(
        "model_id", ["amazon.nova-reel-v1:0", "amazon.nova-reel-v1:1"]
    )
    def test_unsupported_size_is_rejected(self, model_id: str) -> None:
        """Both versions reject any size other than 1280x720."""
        with pytest.raises(ApiError, match="'size' must be '1280x720'"):
            self._build(model_id=model_id, size="1920x1080")

    @pytest.mark.parametrize(
        ("media_type", "image_format"), [("image/png", "png"), ("image/jpeg", "jpeg")]
    )
    def test_reference_image(self, media_type: str, image_format: str) -> None:
        """A PNG or JPEG reference image becomes the starting keyframe."""
        body = self._build(reference_image=ReferenceImage(media_type, "aGVsbG8="))
        assert body["textToVideoParams"]["images"] == [
            {"format": image_format, "source": {"bytes": "aGVsbG8="}}
        ]

    def test_unsupported_reference_image_format(self) -> None:
        """Non PNG/JPEG reference images are rejected."""
        with pytest.raises(ApiError, match="PNG or JPEG"):
            self._build(reference_image=ReferenceImage("image/webp", "aGVsbG8="))

    def test_reference_image_rejected_for_multi_shot(self) -> None:
        """A reference image cannot be combined with a multi-shot duration."""
        with pytest.raises(ApiError, match="input_reference"):
            self._build(
                seconds=12, reference_image=ReferenceImage("image/png", "aGVsbG8=")
            )

    def test_extra_params_extend_generation_config(self) -> None:
        """Extra parameters land in videoGenerationConfig without overriding it."""
        body = self._build(extra_params={"seed": 7, "durationSeconds": 99})
        assert body["videoGenerationConfig"]["seed"] == 7
        assert body["videoGenerationConfig"]["durationSeconds"] == 6


class TestLumaRayInput:
    """Luma Ray modelInput building and parameter validation."""

    @staticmethod
    def _build(**kwargs: Any) -> Any:  # noqa: ANN401
        params: dict[str, Any] = {
            "seconds": 5,
            "size": "1280x720",
            "reference_image": None,
            "extra_params": {},
        }
        params.update(kwargs)
        return LumaRayModel("luma.ray-v2:0").build_generation_input("a cat", **params)

    @pytest.mark.parametrize(
        ("size", "resolution", "aspect_ratio"),
        [
            ("1280x720", "720p", "16:9"),
            ("720x1280", "720p", "9:16"),
            ("960x540", "540p", "16:9"),
            ("540x540", "540p", "1:1"),
            ("1680x720", "720p", "21:9"),
        ],
    )
    def test_size_maps_to_resolution_and_aspect_ratio(
        self, size: str, resolution: str, aspect_ratio: str
    ) -> None:
        """The size selects the resolution and aspect ratio."""
        body = self._build(size=size, seconds=9)
        assert body == {
            "prompt": "a cat",
            "duration": "9s",
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
        }

    @pytest.mark.parametrize("size", ["1024x1024", "640x480", "1920x1080"])
    def test_unsupported_sizes_are_rejected(self, size: str) -> None:
        """Sizes without a matching resolution or aspect ratio are rejected."""
        with pytest.raises(ApiError, match="'size'"):
            self._build(size=size)

    @pytest.mark.parametrize("seconds", [4, 6, 8, 12])
    def test_unsupported_durations_are_rejected(self, seconds: int) -> None:
        """Durations other than 5 or 9 seconds are rejected."""
        with pytest.raises(ApiError, match="'seconds'"):
            self._build(seconds=seconds)

    def test_reference_image_becomes_first_keyframe(self) -> None:
        """A reference image becomes the frame0 keyframe."""
        body = self._build(reference_image=ReferenceImage("image/png", "aGVsbG8="))
        assert body["keyframes"] == {
            "frame0": {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "aGVsbG8=",
                },
            }
        }

    def test_extra_params_extend_payload(self) -> None:
        """Extra parameters land in the payload without overriding it."""
        body = self._build(extra_params={"loop": True, "duration": "1h"})
        assert body["loop"] is True
        assert body["duration"] == "5s"


class _StubRuntimeClient:
    """Stub bedrock-runtime client recording async invoke calls."""

    def __init__(self, get_response: dict[str, Any] | None = None) -> None:
        self.requests: list[dict[str, Any]] = []
        self._get_response = get_response or {}

    async def start_async_invoke(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        """Record the request and return a fixed invocation ARN."""
        self.requests.append(params)
        return {"invocationArn": _ARN}

    async def get_async_invoke(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        """Record the request and return the pre-defined job state."""
        self.requests.append(params)
        return self._get_response


class _StubStreamingBody:
    """Stub S3 object body exposing the chunk iterator used for streaming."""

    def iter_chunks(self) -> list[bytes]:
        """Return the stub content chunks."""
        return [b"mp4-bytes"]


class _StubS3Client:
    """Stub S3 client recording object listing, download and deletion calls."""

    def __init__(self, keys: list[str] | None = None, *, missing: bool = False) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self._keys = keys or []
        self._missing = missing

    async def list_objects_v2(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        """Return the pre-defined object keys."""
        self.requests.append(("list_objects_v2", params))
        return {"Contents": [{"Key": key} for key in self._keys]} if self._keys else {}

    async def delete_objects(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        """Record the deletion request."""
        self.requests.append(("delete_objects", params))
        return {}

    async def get_object(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        """Return a stub body, or raise NoSuchKey when configured missing."""
        self.requests.append(("get_object", params))
        if self._missing:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject"
            )
        return {"Body": _StubStreamingBody()}


def _new_log() -> EventLog:
    return EventLog(
        type="request",
        level="info",
        date=datetime.now(UTC),
        server_id="test",
        server_version="0.0.0",
    )


class TestStartVideoGeneration:
    """VideoModelBase.start_video_generation: async invoke request and usage."""

    @pytest.fixture(autouse=True)
    def _request_context(self) -> Generator[None]:
        """Provide request ID/log and fresh usage state for each test."""
        id_token = REQUEST_ID.set("req1")
        log_token = REQUEST_LOG.set(_new_log())
        usage_token = init_usage()
        init_model_state()
        yield
        USAGE.reset(usage_token)
        REQUEST_LOG.reset(log_token)
        REQUEST_ID.reset(id_token)

    @staticmethod
    def _patch_infra(
        monkeypatch: pytest.MonkeyPatch, client: _StubRuntimeClient
    ) -> None:
        """Pin the candidate region and stub the AWS dependencies."""

        async def _candidates(
            _model_id: str, *, s3_required: bool = False
        ) -> list[RegionName]:
            assert s3_required
            return ["us-east-1"]

        async def _resolve(model_id: str, _region: str, **_kwargs: object) -> str:
            return model_id

        monkeypatch.setattr(video, "_compute_candidate_regions", _candidates)
        monkeypatch.setattr(video, "_resolve_routed_model_id", _resolve)
        monkeypatch.setattr(video, "require_s3_bucket_for_region", lambda _r: "bucket")
        monkeypatch.setattr(video, "get_client", lambda _service, _region: client)

    async def test_start_records_usage_and_targets_regional_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The job writes to the regional bucket and bills the video duration."""
        client = _StubRuntimeClient()
        self._patch_infra(monkeypatch, client)

        start = await get_video_model("amazon.nova-reel-v1:0").start_video_generation(
            "a cat", seconds=None, size=None, reference_image=None, extra_params={}
        )

        assert start.invocation_arn == _ARN
        assert start.seconds == 6
        assert start.size == "1280x720"
        (request,) = client.requests
        assert request["modelId"] == "amazon.nova-reel-v1:0"
        assert request["modelInput"]["taskType"] == "TEXT_VIDEO"
        assert request["outputDataConfig"] == {
            "s3OutputDataConfig": {
                "s3Uri": f"s3://bucket/{SETTINGS.aws_s3_videos_prefix}"
            }
        }
        tags = {tag["key"]: tag["value"] for tag in request["tags"]}
        assert tags["stdapi-ai.seconds"] == "6"
        assert tags["stdapi-ai.size"] == "1280x720"
        records = list(USAGE.get().values())
        assert len(records) == 1
        assert records[0].quantities == {Dimension.OUTPUT_SECONDS: 6}
        assert records[0].region == "us-east-1"

    async def test_luma_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Luma defaults to a 5-second 720p video and bills 5 seconds."""
        client = _StubRuntimeClient()
        self._patch_infra(monkeypatch, client)

        start = await get_video_model("luma.ray-v2:0").start_video_generation(
            "a cat", seconds=None, size=None, reference_image=None, extra_params={}
        )

        assert start.seconds == 5
        assert client.requests[0]["modelInput"]["duration"] == "5s"
        records = list(USAGE.get().values())
        assert records[0].quantities == {Dimension.OUTPUT_SECONDS: 5}


class TestVideoJobAccess:
    """ARN-addressed job retrieval, content access, and output deletion."""

    @pytest.fixture(autouse=True)
    def _regions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pin the configured Bedrock regions and their videos bucket."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1"])
        monkeypatch.setattr(
            SETTINGS, "aws_s3_regional_buckets", {"us-east-1": "bucket"}
        )

    @staticmethod
    def _job() -> VideoJob:
        return VideoJob(
            invocation_arn=_ARN,
            model_id="amazon.nova-reel-v1:0",
            status="completed",
            created_at=1,
            output_bucket="bucket",
            output_prefix="videos/abc123xyz",
        )

    async def test_get_video_job_maps_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The Bedrock job state maps to a VideoJob."""
        client = _StubRuntimeClient(
            {
                "invocationArn": _ARN,
                "modelArn": (
                    "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-reel-v1:0"
                ),
                "status": "Completed",
                "submitTime": datetime(2026, 7, 11, tzinfo=UTC),
                "endTime": datetime(2026, 7, 11, 0, 2, tzinfo=UTC),
                "outputDataConfig": {
                    "s3OutputDataConfig": {"s3Uri": "s3://bucket/videos/abc123xyz"}
                },
            }
        )
        monkeypatch.setattr(video, "get_client", lambda _service, _region: client)

        job = await get_video_job(_ARN)

        assert job.model_id == "amazon.nova-reel-v1:0"
        assert job.status == "completed"
        assert job.created_at == int(datetime(2026, 7, 11, tzinfo=UTC).timestamp())
        assert job.completed_at == int(
            datetime(2026, 7, 11, 0, 2, tzinfo=UTC).timestamp()
        )
        assert job.output_bucket == "bucket"
        assert job.output_prefix == "videos/abc123xyz"

    @pytest.mark.parametrize(
        ("bedrock_status", "failure_message", "expected_status"),
        [
            ("InProgress", None, "in_progress"),
            ("Failed", "content filters blocked the prompt", "failed"),
        ],
    )
    async def test_get_video_job_maps_non_completed_statuses(
        self,
        monkeypatch: pytest.MonkeyPatch,
        bedrock_status: str,
        failure_message: str | None,
        expected_status: str,
    ) -> None:
        """InProgress/Failed Bedrock statuses map to the API job status."""
        state: dict[str, Any] = {
            "invocationArn": _ARN,
            "modelArn": (
                "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-reel-v1:0"
            ),
            "status": bedrock_status,
            "submitTime": datetime(2026, 7, 11, tzinfo=UTC),
            "outputDataConfig": {
                "s3OutputDataConfig": {"s3Uri": "s3://bucket/videos/abc123xyz"}
            },
        }
        if failure_message is not None:
            state["failureMessage"] = failure_message
        client = _StubRuntimeClient(state)
        monkeypatch.setattr(video, "get_client", lambda _service, _region: client)

        job = await get_video_job(_ARN)

        assert job.status == expected_status
        assert job.failure_message == failure_message
        assert job.completed_at is None

    @pytest.mark.parametrize(
        "invocation_arn",
        [
            "not-an-arn",
            "arn:aws:bedrock:us-east-1:000000000000:model/other",
            # Region not configured on this server.
            "arn:aws:bedrock:ap-south-1:000000000000:async-invoke/abc123xyz",
        ],
    )
    async def test_invalid_invocation_arns_are_not_found(
        self, invocation_arn: str
    ) -> None:
        """Malformed or unconfigured-region ARNs surface as 404."""
        with pytest.raises(ApiError, match="not found") as exc_info:
            await get_video_job(invocation_arn)
        assert exc_info.value.status == 404

    async def test_foreign_job_output_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job whose output lies outside this server's videos prefix is rejected."""
        client = _StubRuntimeClient(
            {
                "invocationArn": _ARN,
                "modelArn": (
                    "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-reel-v1:0"
                ),
                "status": "Completed",
                "submitTime": datetime(2026, 7, 11, tzinfo=UTC),
                "outputDataConfig": {
                    # Same bucket, but a different application's prefix.
                    "s3OutputDataConfig": {"s3Uri": "s3://bucket/other-app/abc123xyz"}
                },
            }
        )
        monkeypatch.setattr(video, "get_client", lambda _service, _region: client)

        with pytest.raises(ApiError, match="not found") as exc_info:
            await get_video_job(_ARN)
        assert exc_info.value.status == 404

    async def test_sibling_prefix_job_output_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A slash-less configured prefix must not false-accept a sibling prefix."""
        monkeypatch.setattr(SETTINGS, "aws_s3_videos_prefix", "videos")
        client = _StubRuntimeClient(
            {
                "invocationArn": _ARN,
                "modelArn": (
                    "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-reel-v1:0"
                ),
                "status": "Completed",
                "submitTime": datetime(2026, 7, 11, tzinfo=UTC),
                "outputDataConfig": {
                    # Shares the "videos" string prefix but is a distinct folder.
                    "s3OutputDataConfig": {
                        "s3Uri": "s3://bucket/videos-other/abc123xyz"
                    }
                },
            }
        )
        monkeypatch.setattr(video, "get_client", lambda _service, _region: client)

        with pytest.raises(ApiError, match="not found") as exc_info:
            await get_video_job(_ARN)
        assert exc_info.value.status == 404

    async def test_job_in_unconfigured_bucket_region_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job in a region without a configured bucket is rejected outright."""
        monkeypatch.setattr(SETTINGS, "aws_s3_regional_buckets", {})
        client = _StubRuntimeClient(
            {
                "invocationArn": _ARN,
                "modelArn": (
                    "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-reel-v1:0"
                ),
                "status": "Completed",
                "submitTime": datetime(2026, 7, 11, tzinfo=UTC),
                "outputDataConfig": {
                    "s3OutputDataConfig": {"s3Uri": "s3://bucket/videos/abc123xyz"}
                },
            }
        )
        monkeypatch.setattr(video, "get_client", lambda _service, _region: client)

        with pytest.raises(ApiError, match="not found") as exc_info:
            await get_video_job(_ARN)
        assert exc_info.value.status == 404

    async def test_open_video_content_targets_output_file(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The MP4 is read from the job output folder."""
        s3 = _StubS3Client()
        monkeypatch.setattr(video, "get_client", lambda _service, _region: s3)

        await open_video_content(self._job())

        assert s3.requests == [
            ("get_object", {"Bucket": "bucket", "Key": "videos/abc123xyz/output.mp4"})
        ]

    async def test_open_video_content_missing_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deleted output surfaces as 404."""
        s3 = _StubS3Client(missing=True)
        monkeypatch.setattr(video, "get_client", lambda _service, _region: s3)

        with pytest.raises(ApiError, match="not found") as exc_info:
            await open_video_content(self._job())
        assert exc_info.value.status == 404

    async def test_video_expires_at_disabled_by_default(self) -> None:
        """Without a retention setting no expiry is reported."""
        job = self._job().model_copy(update={"completed_at": 1000})
        assert video.video_expires_at(job) is None

    async def test_video_expires_at_from_completion(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The expiry is the completion time plus the configured retention."""
        monkeypatch.setattr(SETTINGS, "aws_s3_videos_expires_after", 3600)
        job = self._job().model_copy(update={"completed_at": 1000})
        assert video.video_expires_at(job) == 4600
        assert video.video_expires_at(self._job()) is None

    async def test_open_video_content_expired_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Content of an expired video surfaces as 404 without reaching S3."""
        monkeypatch.setattr(SETTINGS, "aws_s3_videos_expires_after", 3600)
        s3 = _StubS3Client()
        monkeypatch.setattr(video, "get_client", lambda _service, _region: s3)
        job = self._job().model_copy(update={"completed_at": 1})

        with pytest.raises(ApiError, match="not found") as exc_info:
            await open_video_content(job)
        assert exc_info.value.status == 404
        assert s3.requests == []

    async def test_delete_video_output_removes_job_folder(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All objects under the job folder are deleted."""
        s3 = _StubS3Client(
            keys=["videos/abc123xyz/output.mp4", "videos/abc123xyz/manifest.json"]
        )
        monkeypatch.setattr(video, "get_client", lambda _service, _region: s3)

        await delete_video_output(self._job())

        assert s3.requests[0] == (
            "list_objects_v2",
            {"Bucket": "bucket", "Prefix": "videos/abc123xyz/"},
        )
        assert s3.requests[1] == (
            "delete_objects",
            {
                "Bucket": "bucket",
                "Delete": {
                    "Objects": [
                        {"Key": "videos/abc123xyz/output.mp4"},
                        {"Key": "videos/abc123xyz/manifest.json"},
                    ]
                },
            },
        )

    async def test_delete_video_output_with_no_objects(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An already-empty job folder deletes nothing."""
        s3 = _StubS3Client()
        monkeypatch.setattr(video, "get_client", lambda _service, _region: s3)

        await delete_video_output(self._job())

        assert [name for name, _ in s3.requests] == ["list_objects_v2"]


class TestToVideoJob:
    """_to_video_job: mapping of the Bedrock ``s3Uri`` into bucket/prefix."""

    @staticmethod
    def _state(s3_uri: str) -> dict[str, Any]:
        return {
            "modelArn": (
                "arn:aws:bedrock:us-east-1::foundation-model/amazon.nova-reel-v1:0"
            ),
            "status": "Completed",
            "submitTime": datetime(2026, 7, 11, tzinfo=UTC),
            "outputDataConfig": {"s3OutputDataConfig": {"s3Uri": s3_uri}},
        }

    def test_key_present_splits_bucket_and_prefix(self) -> None:
        """A normal ``s3://bucket/prefix`` URI splits into bucket and prefix."""
        job = video._to_video_job(  # noqa: SLF001
            _ARN, self._state("s3://bucket/videos/abc123xyz")
        )
        assert job.output_bucket == "bucket"
        assert job.output_prefix == "videos/abc123xyz"

    def test_bucket_root_with_trailing_slash_yields_empty_prefix(self) -> None:
        """A bucket-root URI with a trailing slash yields an empty prefix."""
        job = video._to_video_job(_ARN, self._state("s3://bucket/"))  # noqa: SLF001
        assert job.output_bucket == "bucket"
        assert job.output_prefix == ""

    def test_bucket_only_without_slash_yields_empty_prefix(self) -> None:
        """A bucket-only URI with no trailing slash does not raise."""
        job = video._to_video_job(_ARN, self._state("s3://bucket"))  # noqa: SLF001
        assert job.output_bucket == "bucket"
        assert job.output_prefix == ""


class _StubListClient:
    """Stub client serving list_async_invokes pages and resource tags."""

    def __init__(
        self, pages: list[dict[str, Any]], tags: dict[str, dict[str, str]] | None = None
    ) -> None:
        self.requests: list[dict[str, Any]] = []
        self._pages = list(pages)
        self._tags = tags or {}

    async def list_async_invokes(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        """Record the request and serve the next pre-defined page."""
        self.requests.append(params)
        return self._pages.pop(0)

    async def list_tags_for_resource(self, resourceARN: str) -> dict[str, Any]:  # noqa: N803
        """Return the pre-defined tags for the resource."""
        return {
            "tags": [
                {"key": k, "value": v}
                for k, v in self._tags.get(resourceARN, {}).items()
            ]
        }


def _arn(suffix: str) -> str:
    return f"arn:aws:bedrock:us-east-1:000000000000:async-invoke/{suffix}"


def _summary(
    suffix: str,
    submit: int,
    model: str = "amazon.nova-reel-v1:0",
    s3_uri: str | None = None,
) -> dict[str, Any]:
    """Build a ListAsyncInvokes summary for a completed job."""
    return {
        "invocationArn": _arn(suffix),
        "modelArn": f"arn:aws:bedrock:us-east-1::foundation-model/{model}",
        "status": "Completed",
        "submitTime": datetime.fromtimestamp(submit, tz=UTC),
        "endTime": datetime.fromtimestamp(submit + 90, tz=UTC),
        "outputDataConfig": {
            "s3OutputDataConfig": {"s3Uri": s3_uri or f"s3://bucket/videos/{suffix}"}
        },
    }


class TestListVideoJobs:
    """Cross-region job listing: filtering, ordering, cursor, and details."""

    @pytest.fixture(autouse=True)
    def _regions(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pin one region with a configured bucket."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["us-east-1"])
        monkeypatch.setattr(video, "get_s3_bucket_for_region", lambda _region: "bucket")

    @staticmethod
    def _patch_client(monkeypatch: pytest.MonkeyPatch, client: _StubListClient) -> None:
        monkeypatch.setattr(video, "get_client", lambda _service, _region: client)

    async def test_lists_own_jobs_with_tag_details(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only jobs under the videos prefix are listed; tags set duration/size."""
        client = _StubListClient(
            [
                {
                    "asyncInvokeSummaries": [
                        _summary("aaa111", 100),
                        _summary("bbb222", 200),
                        _summary("foreign", 300, s3_uri="s3://other/videos/x"),
                    ]
                }
            ],
            tags={
                _arn("bbb222"): {
                    "stdapi-ai.seconds": "12",
                    "stdapi-ai.size": "1280x720",
                }
            },
        )
        self._patch_client(monkeypatch, client)

        listings, has_more = await video.list_video_jobs()

        assert not has_more
        assert [listing.job.invocation_arn for listing in listings] == [
            _arn("bbb222"),
            _arn("aaa111"),
        ]
        # Tagged job uses its tags; untagged one falls back to model defaults.
        assert (listings[0].seconds, listings[0].size) == ("12", "1280x720")
        assert (listings[1].seconds, listings[1].size) == ("6", "1280x720")
        assert listings[0].job.status == "completed"
        assert listings[0].job.completed_at == 290

    async def test_after_cursor_and_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The page starts strictly after the cursor and respects the limit."""
        summaries = [
            _summary(s, t) for s, t in (("aaa", 100), ("bbb", 200), ("ccc", 300))
        ]
        self._patch_client(
            monkeypatch, _StubListClient([{"asyncInvokeSummaries": summaries}])
        )

        listings, has_more = await video.list_video_jobs(after_arn=_arn("ccc"), limit=1)

        assert [listing.job.invocation_arn for listing in listings] == [_arn("bbb")]
        assert has_more

    async def test_unknown_after_cursor_yields_empty_page(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unknown cursor returns an empty page instead of restarting."""
        self._patch_client(
            monkeypatch,
            _StubListClient([{"asyncInvokeSummaries": [_summary("aaa", 100)]}]),
        )

        listings, has_more = await video.list_video_jobs(after_arn=_arn("zzz"))

        assert listings == []
        assert not has_more

    async def test_ascending_order(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """order=asc returns oldest jobs first."""
        summaries = [_summary("aaa", 200), _summary("bbb", 100)]
        self._patch_client(
            monkeypatch, _StubListClient([{"asyncInvokeSummaries": summaries}])
        )

        listings, _ = await video.list_video_jobs(order="asc")

        assert [listing.job.created_at for listing in listings] == [100, 200]

    async def test_scan_follows_pagination(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The region scan follows nextToken across pages."""
        client = _StubListClient(
            [
                {"asyncInvokeSummaries": [_summary("aaa", 100)], "nextToken": "page2"},
                {"asyncInvokeSummaries": [_summary("bbb", 200)]},
            ]
        )
        self._patch_client(monkeypatch, client)

        listings, _ = await video.list_video_jobs()

        assert len(listings) == 2
        assert client.requests[1]["nextToken"] == "page2"

    async def test_unknown_model_without_tags_is_dropped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Jobs from unknown models with no tags are dropped from the page."""
        self._patch_client(
            monkeypatch,
            _StubListClient(
                [{"asyncInvokeSummaries": [_summary("aaa", 100, model="foo.bar-v9:0")]}]
            ),
        )

        listings, has_more = await video.list_video_jobs()

        assert listings == []
        assert not has_more
