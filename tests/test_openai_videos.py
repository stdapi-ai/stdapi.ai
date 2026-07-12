"""Tests for the OpenAI-compatible /v1/videos routes (unit and live)."""

import time
from contextlib import suppress
from io import BytesIO
from os import getenv
from typing import TYPE_CHECKING, Any, Literal

import httpx
import pytest
from openai import BadRequestError
from PIL import Image
from starlette.testclient import TestClient

from stdapi.config import SETTINGS
from stdapi.models import ModelDetails
from stdapi.models.video import VideoGenerationStart, VideoJob, VideoListing
from stdapi.routes import openai_videos

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.video import Video

#: A well-formed async invocation ARN in the primary test region.
_ARN = "arn:aws:bedrock:us-east-1:000000000000:async-invoke/abc123xyz"


class _StubVideoModel:
    """Stub backend recording the generation call and returning a fixed job."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def start_video_generation(
        self,
        prompt: str,
        *,
        seconds: int | None,
        size: str | None,
        reference_image: object,
        extra_params: dict[str, Any],
    ) -> VideoGenerationStart:
        """Record the call and return a fixed started job."""
        self.calls.append(
            {
                "prompt": prompt,
                "seconds": seconds,
                "size": size,
                "reference_image": reference_image,
                "extra_params": extra_params,
            }
        )
        return VideoGenerationStart(
            invocation_arn=_ARN, seconds=seconds or 6, size=size or "1280x720"
        )


def _job(
    status: Literal["in_progress", "completed", "failed"],
    failure_message: str | None = None,
) -> VideoJob:
    return VideoJob(
        invocation_arn=_ARN,
        model_id="amazon.nova-reel-v1:0",
        status=status,
        created_at=1752000000,
        completed_at=1752000090 if status == "completed" else None,
        failure_message=failure_message,
        output_bucket="bucket",
        output_prefix="videos/abc123xyz",
    )


@pytest.fixture
def client(api_key: str) -> TestClient:
    """Test client without lifespan (no AWS startup), pre-authenticated."""
    from stdapi.main import app  # noqa: PLC0415

    return TestClient(app, headers={"Authorization": f"Bearer {api_key}"})


@pytest.fixture
def video_backend(monkeypatch: pytest.MonkeyPatch) -> _StubVideoModel:
    """Stub model validation and the video generation backend."""

    async def _validate_model(
        model_id: str, *_args: object, **_kwargs: object
    ) -> ModelDetails:
        return ModelDetails(
            id=model_id,
            name=model_id,
            provider="Vendor",
            input_modalities=["TEXT", "IMAGE"],
            output_modalities=["VIDEO"],
            regions=["us-east-1"],
        )

    stub = _StubVideoModel()
    monkeypatch.setattr(openai_videos, "validate_model", _validate_model)
    monkeypatch.setattr(openai_videos, "get_video_model", lambda _model_id: stub)
    return stub


def _stub_job(
    monkeypatch: pytest.MonkeyPatch,
    status: Literal["in_progress", "completed", "failed"],
    failure_message: str | None = None,
) -> VideoJob:
    """Stub the job retrieval to return a job with the given status."""
    job = _job(status, failure_message)

    async def _get_video_job(_invocation_arn: str) -> VideoJob:
        return job

    monkeypatch.setattr(openai_videos, "get_video_job", _get_video_job)
    return job


@pytest.mark.local
class TestOpenAIVideoRoutes:
    """POST/GET/DELETE /v1/videos: response shapes and error envelopes."""

    def test_create_json(
        self, client: TestClient, video_backend: _StubVideoModel
    ) -> None:
        """A JSON creation request returns a queued Video job."""
        response = client.post(
            "/v1/videos",
            json={
                "model": "amazon.nova-reel-v1:0",
                "prompt": "a cat",
                "seconds": "6",
                "size": "1280x720",
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["object"] == "video"
        assert body["status"] == "queued"
        assert body["progress"] == 0
        assert body["model"] == "amazon.nova-reel-v1:0"
        assert body["seconds"] == "6"
        assert body["size"] == "1280x720"
        assert body["prompt"] == "a cat"
        assert openai_videos._decode_video_id(body["id"]) == (  # noqa: SLF001
            _ARN,
            "6",
            "1280x720",
        )
        (call,) = video_backend.calls
        assert call["seconds"] == 6
        assert call["reference_image"] is None

    def test_create_json_with_reference_and_extra_params(
        self, client: TestClient, video_backend: _StubVideoModel
    ) -> None:
        """A data URI reference and extra body fields reach the backend."""
        response = client.post(
            "/v1/videos",
            json={
                "model": "amazon.nova-reel-v1:0",
                "prompt": "a cat",
                "input_reference": "data:image/png;base64,aGVsbG8=",
                "seed": 7,
            },
        )
        assert response.status_code == 200, response.text
        (call,) = video_backend.calls
        assert call["reference_image"] is not None
        assert call["extra_params"] == {"seed": 7}
        assert call["seconds"] is None

    def test_create_multipart(
        self, client: TestClient, video_backend: _StubVideoModel
    ) -> None:
        """A multipart request with a binary reference image succeeds."""
        response = client.post(
            "/v1/videos",
            data={
                "model": "luma.ray-v2:0",
                "prompt": "a cat",
                "seconds": "5",
                "size": "960x540",
                "loop": "true",
            },
            files={"input_reference": ("frame.png", b"png-bytes", "image/png")},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["seconds"] == "5"
        assert body["size"] == "960x540"
        (call,) = video_backend.calls
        assert call["reference_image"] is not None
        assert call["extra_params"] == {"loop": True}

    def test_create_without_prompt_is_rejected(
        self, client: TestClient, video_backend: _StubVideoModel
    ) -> None:
        """A request without a prompt fails validation."""
        response = client.post("/v1/videos", json={"model": "amazon.nova-reel-v1:0"})
        assert response.status_code == 400
        assert "prompt" in response.json()["error"]["message"]
        assert not video_backend.calls

    def test_create_with_non_integer_seconds_is_rejected(
        self, client: TestClient, video_backend: _StubVideoModel
    ) -> None:
        """A non-integer duration fails the seconds pattern validation."""
        response = client.post(
            "/v1/videos",
            json={
                "model": "amazon.nova-reel-v1:0",
                "prompt": "a cat",
                "seconds": "4.5",
            },
        )
        assert response.status_code == 400
        assert "seconds" in response.json()["error"]["message"]
        assert not video_backend.calls

    def test_retrieve_completed(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A completed job maps to a completed Video with progress 100."""
        _stub_job(monkeypatch, "completed")
        video_id = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001
        response = client.get(f"/v1/videos/{video_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "completed"
        assert body["progress"] == 100
        assert body["completed_at"] == 1752000090
        assert body["seconds"] == "6"
        assert body["size"] == "1280x720"
        assert body["model"] == "amazon.nova-reel-v1:0"
        assert "error" not in body
        assert "expires_at" not in body

    def test_retrieve_reports_expires_at_when_retention_set(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a retention period configured the Video reports expires_at."""
        monkeypatch.setattr(SETTINGS, "aws_s3_videos_expires_after", 3600)
        _stub_job(monkeypatch, "completed")
        video_id = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001
        response = client.get(f"/v1/videos/{video_id}")
        assert response.status_code == 200, response.text
        assert response.json()["expires_at"] == 1752000090 + 3600

    def test_retrieve_failed(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed job carries the failure reason in the error payload."""
        _stub_job(monkeypatch, "failed", "content filters blocked the prompt")
        video_id = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001
        response = client.get(f"/v1/videos/{video_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "failed"
        assert body["error"]["message"] == "content filters blocked the prompt"

    def test_retrieve_bad_id_is_not_found(self, client: TestClient) -> None:
        """An undecodable video ID surfaces as an OpenAI 404 error."""
        response = client.get("/v1/videos/video_bogus")
        assert response.status_code == 404
        assert "video_bogus" in response.json()["error"]["message"]

    def test_list_videos(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The list endpoint pages the backend listings as Video objects."""
        calls: list[dict[str, Any]] = []

        async def _list_video_jobs(**kwargs: Any) -> tuple[list[VideoListing], bool]:  # noqa: ANN401
            calls.append(kwargs)
            return [VideoListing(_job("completed"), "6", "1280x720")], True

        monkeypatch.setattr(openai_videos, "list_video_jobs", _list_video_jobs)
        after = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001

        response = client.get(f"/v1/videos?limit=5&order=asc&after={after}")

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["object"] == "list"
        assert body["has_more"] is True
        (video,) = body["data"]
        assert video["status"] == "completed"
        assert openai_videos._decode_video_id(video["id"]) == (  # noqa: SLF001
            _ARN,
            "6",
            "1280x720",
        )
        assert body["first_id"] == body["last_id"] == video["id"]
        assert calls == [{"order": "asc", "after_arn": _ARN, "limit": 5}]

    def test_list_videos_bad_cursor_is_not_found(self, client: TestClient) -> None:
        """An undecodable after cursor surfaces as an OpenAI 404 error."""
        response = client.get("/v1/videos?after=video_bogus")
        assert response.status_code == 404
        assert "video_bogus" in response.json()["error"]["message"]

    def test_content_streams_video(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A completed job's MP4 is streamed with the video/mp4 content type."""
        _stub_job(monkeypatch, "completed")

        async def _open_video_content(_job: VideoJob) -> Any:  # noqa: ANN401
            return iter([b"mp4-", b"bytes"])

        monkeypatch.setattr(openai_videos, "open_video_content", _open_video_content)
        video_id = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001
        response = client.get(f"/v1/videos/{video_id}/content")
        assert response.status_code == 200
        assert response.headers["content-type"] == "video/mp4"
        assert response.content == b"mp4-bytes"

    def test_content_not_ready(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Downloading an unfinished job's content is a 404, like upstream."""
        _stub_job(monkeypatch, "in_progress")
        video_id = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001
        response = client.get(f"/v1/videos/{video_id}/content")
        assert response.status_code == 404
        assert "not ready" in response.json()["error"]["message"]

    def test_content_variant_not_available(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the video variant is available."""
        _stub_job(monkeypatch, "completed")
        video_id = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001
        response = client.get(f"/v1/videos/{video_id}/content?variant=thumbnail")
        assert response.status_code == 400

    def test_delete_completed(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deleting a completed job removes its output and confirms deletion."""
        _stub_job(monkeypatch, "completed")
        deleted: list[VideoJob] = []

        async def _delete_video_output(job: VideoJob) -> None:
            deleted.append(job)

        monkeypatch.setattr(openai_videos, "delete_video_output", _delete_video_output)
        video_id = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001
        response = client.delete(f"/v1/videos/{video_id}")
        assert response.status_code == 200, response.text
        assert response.json() == {
            "id": video_id,
            "object": "video.deleted",
            "deleted": True,
        }
        assert len(deleted) == 1

    def test_delete_in_progress_is_rejected(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job still being processed cannot be deleted, like upstream."""
        _stub_job(monkeypatch, "in_progress")
        video_id = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001
        response = client.delete(f"/v1/videos/{video_id}")
        assert response.status_code == 400
        assert "still being processed" in response.json()["error"]["message"]


def _wait_for_completion(
    openai_client: OpenAI, video_id: str, timeout: float = 900.0
) -> Video:
    """Poll a video job until completion, failing on error or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        video = openai_client.videos.retrieve(video_id)
        if video.status == "completed":
            return video
        assert video.status in ("queued", "in_progress"), video.error
        time.sleep(10)
    pytest.fail(f"Video '{video_id}' did not complete within {timeout}s")


def _reference_frame(size: str) -> bytes:
    """Build a plain PNG frame of the given "<width>x<height>" size."""
    width, height = map(int, size.split("x"))
    buffer = BytesIO()
    Image.new("RGB", (width, height), (10, 60, 120)).save(buffer, "PNG")
    return buffer.getvalue()


def _advertised_model_ids(
    request: pytest.FixtureRequest, test_client: TestClient | None, api_key: str
) -> list[str]:
    """Return the model IDs advertising the video generation route.

    Queries the in-process test client when available, or the ``--server-url``
    target otherwise.
    """
    if test_client is not None:
        response: httpx.Response = test_client.get(
            "/search_models",
            params={"route": "openai_video_generation"},
            headers={"authorization": f"Bearer {api_key}"},
        )
    else:
        server_url: str = request.config.getoption("--server-url")
        response = httpx.get(
            f"{server_url.rstrip('/')}/search_models",
            params={"route": "openai_video_generation"},
            headers={"authorization": f"Bearer {getenv('OPENAI_API_KEY', '')}"},
            timeout=60.0,
        )
    assert response.status_code == 200, response.text
    return [model["id"] for model in response.json()]


class TestOpenAIVideoIntegration:
    """Live /v1/videos lifecycle through the OpenAI SDK (local or official API)."""

    @pytest.mark.expensive
    def test_video_generation_lifecycle(
        self, openai_client: OpenAI, video_generation_model: str
    ) -> None:
        """Create, poll, download, and delete a video with the default model."""
        video = openai_client.videos.create(
            model=video_generation_model, prompt="A calico cat playing a piano"
        )
        try:
            assert video.object == "video"
            assert video.status in ("queued", "in_progress")

            video = _wait_for_completion(openai_client, video.id)
            assert video.progress == 100
            assert video.completed_at

            content = openai_client.videos.download_content(video.id).read()
            assert content[4:8] == b"ftyp"

            deleted = openai_client.videos.delete(video.id)
            assert deleted.deleted is True
            assert deleted.object == "video.deleted"
        finally:
            # Best-effort cleanup when an assertion fails mid-lifecycle.
            with suppress(Exception):
                openai_client.videos.delete(video.id)

    @pytest.mark.expensive
    def test_video_generation_from_reference_image(
        self, openai_client: OpenAI, video_generation_model: str
    ) -> None:
        """An uploaded reference image drives image-to-video generation."""
        size = "1280x720"
        video = openai_client.videos.create(
            model=video_generation_model,
            prompt="The camera slowly zooms in",
            size=size,  # type: ignore[arg-type]
            input_reference=("frame.png", _reference_frame(size), "image/png"),
        )
        try:
            video = _wait_for_completion(openai_client, video.id)
            assert video.size == size
        finally:
            with suppress(Exception):
                openai_client.videos.delete(video.id)

    @pytest.mark.expensive
    def test_every_advertised_model_generates(
        self,
        request: pytest.FixtureRequest,
        openai_client: OpenAI,
        test_client: TestClient | None,
        api_key: str,
        use_official_api: bool,
        video_generation_model: str,
    ) -> None:
        """Every model advertising the route serves the full lifecycle.

        With ``AWS_BEDROCK_LEGACY`` enabled this also covers the legacy Amazon
        Nova Reel models.
        """
        if use_official_api:
            pytest.skip("/search_models is not part of the official API")
        model_ids = _advertised_model_ids(request, test_client, api_key)
        assert model_ids
        for model_id in model_ids:
            if model_id == video_generation_model:
                continue  # Covered by test_video_generation_lifecycle.
            video = openai_client.videos.create(
                model=model_id, prompt="Waves rolling onto a sandy beach"
            )
            try:
                video = _wait_for_completion(openai_client, video.id)
                content = openai_client.videos.download_content(video.id).read()
                assert content[4:8] == b"ftyp", model_id
            finally:
                with suppress(Exception):
                    openai_client.videos.delete(video.id)

    @pytest.mark.expensive
    def test_nova_reel_multi_shot_duration(
        self,
        request: pytest.FixtureRequest,
        openai_client: OpenAI,
        test_client: TestClient | None,
        api_key: str,
        use_official_api: bool,
    ) -> None:
        """Nova Reel serves durations beyond 6s through automated multi-shot."""
        if use_official_api:
            pytest.skip("/search_models is not part of the official API")
        if "amazon.nova-reel-v1:1" not in _advertised_model_ids(
            request, test_client, api_key
        ):
            pytest.skip("Nova Reel is hidden unless AWS_BEDROCK_LEGACY is enabled")
        video = openai_client.videos.create(
            model="amazon.nova-reel-v1:1",
            prompt="A tour of a colorful coral reef",
            seconds="12",
        )
        try:
            assert video.seconds == "12"
            video = _wait_for_completion(openai_client, video.id)
            content = openai_client.videos.download_content(video.id).read()
            assert content[4:8] == b"ftyp"
        finally:
            with suppress(Exception):
                openai_client.videos.delete(video.id)

    def test_unsupported_duration_returns_clean_error(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """A duration unsupported by the model is rejected before invocation."""
        if use_official_api:
            pytest.skip("Bedrock video models require a stdapi server")
        with pytest.raises(BadRequestError, match="seconds"):
            openai_client.videos.create(
                model="luma.ray-v2:0", prompt="a cat", seconds="4"
            )

    def test_unsupported_size_returns_clean_error(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """A size unsupported by the model is rejected before invocation."""
        if use_official_api:
            pytest.skip("Bedrock video models require a stdapi server")
        with pytest.raises(BadRequestError, match="size"):
            openai_client.videos.create(
                model="luma.ray-v2:0",
                prompt="a cat",
                size="1024x1024",  # type: ignore[arg-type]
            )

    def test_list_videos(self, openai_client: OpenAI, use_official_api: bool) -> None:
        """Previously generated jobs are listed with duration and size."""
        if use_official_api:
            pytest.skip("listing scope differs on the official API")
        page = openai_client.videos.list(limit=5)
        # AWS retains async job records for a limited time; a fresh account
        # may legitimately have nothing to list.
        videos = list(page.data)
        assert all(video.id.startswith("video_") for video in videos)
        if videos:
            assert videos[0].seconds
            assert videos[0].size
            assert videos[0].created_at >= videos[-1].created_at
