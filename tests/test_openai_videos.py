"""Tests for the OpenAI-compatible /v1/videos routes (unit and live).

The video ID encodes the Bedrock invocation ARN plus the effective duration
and size, so retrieve/download/delete need no server-side state. AWS async
invocation reports only InProgress/Completed/Failed, so OpenAI's ``queued``
status and ``progress`` are synthesized, and the accepted ``seconds``/``size``
values are the model's, not OpenAI's 4/8/12 and 720x1280.

Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     https://stdapi.ai/api_openai_videos/
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_GetAsyncInvoke.html
     stdapi/routes/openai_videos.py:create_video
"""

import time
from base64 import b64encode
from contextlib import suppress
from io import BytesIO
from os import getenv
from typing import TYPE_CHECKING, Any, Literal

import httpx
import pytest
from openai import BadRequestError
from PIL import Image

from stdapi.api_errors import ApiError
from stdapi.config import SETTINGS
from stdapi.models.video import VideoGenerationStart, VideoJob, VideoListing
from stdapi.routes import openai_videos
from tests._helpers import make_model_details

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.video import Video
    from starlette.testclient import TestClient

    from stdapi.models import ModelDetails

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


#: A validly-encoded video ID, used where a path ID is required but must never resolve.
_DUMMY_VIDEO_ID = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001


@pytest.fixture
def video_backend(monkeypatch: pytest.MonkeyPatch) -> _StubVideoModel:
    """Stub model validation and the video generation backend."""

    async def _validate_model(
        model_id: str, *_args: object, **_kwargs: object
    ) -> ModelDetails:
        return make_model_details(
            model_id, input_modalities=["TEXT", "IMAGE"], output_modalities=["VIDEO"]
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
    """POST/GET/DELETE /v1/videos: response shapes and error envelopes.

    The model layer is stubbed, so these cover the route contract only: body
    parsing, the Video/VideoList/VideoDeleted projections and the OpenAI error
    envelope.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/types/openai_videos.py:Video
    """

    def test_create_json(
        self, app_client: TestClient, video_backend: _StubVideoModel
    ) -> None:
        """A JSON creation request returns a queued Video job echoing its parameters.

        AWS has no queued state, so ``queued`` with ``progress`` 0 is synthesized
        for the create response; the returned ID decodes back to the invocation
        ARN with the effective duration and size.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_GetAsyncInvoke.html
             stdapi/routes/openai_videos.py:_encode_video_id
        """
        response = app_client.post(
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
        assert call["prompt"] == "a cat"
        assert call["seconds"] == 6, "the string duration is passed as an integer"
        assert call["size"] == "1280x720"
        assert call["reference_image"] is None
        assert call["extra_params"] == {}

    def test_create_json_with_reference_and_extra_params(
        self, app_client: TestClient, video_backend: _StubVideoModel
    ) -> None:
        """A data URI reference and extra body fields reach the backend.

        Unknown JSON fields are forwarded as provider parameters rather than
        rejected, and omitting seconds/size leaves the model defaults to the
        backend, which reports the effective values back.

        Ref: https://stdapi.ai/api_openai_videos/
             stdapi/aws_bedrock.py:get_extra_model_parameters
        """
        response = app_client.post(
            "/v1/videos",
            json={
                "model": "amazon.nova-reel-v1:0",
                "prompt": "a cat",
                "input_reference": "data:image/png;base64,aGVsbG8=",
                "seed": 7,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "queued"
        assert (body["seconds"], body["size"]) == ("6", "1280x720"), (
            "the effective duration and size reported by the backend are echoed"
        )
        (call,) = video_backend.calls
        assert call["reference_image"] is not None
        assert call["extra_params"] == {"seed": 7}
        assert call["seconds"] is None
        assert call["size"] is None

    def test_create_multipart(
        self, app_client: TestClient, video_backend: _StubVideoModel
    ) -> None:
        """A multipart request with a binary reference image succeeds.

        Multipart values are strings, so extras are JSON-decoded before reaching
        the model ("true" becomes ``True``), and the uploaded file becomes the
        reference image.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_videos.py:_decode_form_extras
        """
        response = app_client.post(
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
        assert body["model"] == "luma.ray-v2:0"
        assert body["seconds"] == "5"
        assert body["size"] == "960x540"
        (call,) = video_backend.calls
        assert (call["seconds"], call["size"]) == (5, "960x540")
        assert call["reference_image"] is not None
        assert call["extra_params"] == {"loop": True}, (
            "form values are JSON-decoded, so 'true' must reach the model as True"
        )

    def test_create_multipart_with_flattened_input_reference(
        self, app_client: TestClient, video_backend: _StubVideoModel
    ) -> None:
        """The SDK's flattened `input_reference[image_url]` key becomes the reference.

        The OpenAI SDK flattens the ``input_reference`` object into bracketed form
        keys; those must be consumed as the reference image instead of leaking
        into the provider parameters.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_videos.py:_REFERENCE_FORM_KEYS
        """
        data_uri = (
            "data:image/png;base64," + b64encode(_reference_frame("64x64")).decode()
        )
        response = app_client.post(
            "/v1/videos",
            data={
                "model": "luma.ray-v2:0",
                "prompt": "a cat",
                "input_reference[image_url]": data_uri,
            },
        )
        assert response.status_code == 200, response.text
        (call,) = video_backend.calls
        assert call["reference_image"] is not None
        assert call["extra_params"] == {}, (
            "the bracketed key must not leak into the provider parameters"
        )

    def test_create_without_prompt_is_rejected(
        self, app_client: TestClient, video_backend: _StubVideoModel
    ) -> None:
        """A request without a prompt fails validation before any model call.

        ``prompt`` is the only required field of the upstream body, and the
        rejection is reported as a request-validation error naming the field.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai_videos.py:VideoCreateParams
        """
        response = app_client.post(
            "/v1/videos", json={"model": "amazon.nova-reel-v1:0"}
        )
        assert response.status_code == 400
        err = response.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert err["message"].startswith("Validation error at prompt")
        assert not video_backend.calls

    def test_create_with_non_integer_seconds_is_rejected(
        self, app_client: TestClient, video_backend: _StubVideoModel
    ) -> None:
        """A non-integer duration fails the seconds pattern validation.

        ``seconds`` is a digits-only string (OpenAI sends "4"/"8"/"12"), so a
        fractional value is rejected by the schema rather than reaching ``int()``.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai_videos.py:VideoCreateParams
        """
        response = app_client.post(
            "/v1/videos",
            json={
                "model": "amazon.nova-reel-v1:0",
                "prompt": "a cat",
                "seconds": "4.5",
            },
        )
        assert response.status_code == 400
        err = response.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert err["message"].startswith("Validation error at seconds")
        assert not video_backend.calls

    def test_create_with_zero_size_is_rejected(
        self, app_client: TestClient, video_backend: _StubVideoModel
    ) -> None:
        """A "0x0" size fails the size pattern validation instead of reaching gcd(0, 0).

        The size pattern requires both dimensions to start with a non-zero digit,
        so the Luma backend never divides by ``gcd(0, 0)``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-luma.html
             stdapi/types/openai_videos.py:VideoCreateParams
        """
        response = app_client.post(
            "/v1/videos",
            json={"model": "amazon.nova-reel-v1:0", "prompt": "a cat", "size": "0x0"},
        )
        assert response.status_code == 400
        err = response.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert err["message"].startswith("Validation error at size")
        assert not video_backend.calls

    def test_create_with_malformed_json_body_is_rejected(
        self, app_client: TestClient, video_backend: _StubVideoModel
    ) -> None:
        """A malformed JSON body is rejected as a validation error, not a 500.

        The body is parsed by the route itself (``model_validate_json``), so the
        pydantic ``json_invalid`` error has to be converted into the same
        request-validation envelope FastAPI would produce.

        Ref: stdapi/utils.py:validation_error_handler
             stdapi/main.py:handle_validation_exception
        """
        response = app_client.post(
            "/v1/videos", content=b"{", headers={"content-type": "application/json"}
        )
        assert response.status_code == 400, response.text
        err = response.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert "Invalid JSON" in err["message"]
        assert not video_backend.calls

    def test_create_then_retrieve_roundtrip(
        self,
        app_client: TestClient,
        video_backend: _StubVideoModel,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The public id from a create response resolves through GET.

        The ID is the only thing carried between the two calls, and the prompt is
        echoed only by create: a retrieved job is rebuilt from the AWS job state
        plus the ID, which holds no prompt.

        Ref: https://stdapi.ai/api_openai_videos/
             stdapi/routes/openai_videos.py:_decode_video_id
        """
        create_response = app_client.post(
            "/v1/videos", json={"model": "amazon.nova-reel-v1:0", "prompt": "a cat"}
        )
        assert create_response.status_code == 200, create_response.text
        created = create_response.json()
        video_id = created["id"]
        assert created["prompt"] == "a cat"

        _stub_job(monkeypatch, "completed")
        retrieve_response = app_client.get(f"/v1/videos/{video_id}")

        assert retrieve_response.status_code == 200, retrieve_response.text
        body = retrieve_response.json()
        assert body["id"] == video_id
        assert body["status"] == "completed"
        assert (body["seconds"], body["size"]) == (
            created["seconds"],
            created["size"],
        ), "duration and size survive the roundtrip inside the ID"
        assert "prompt" not in body, "the prompt is echoed only by create"

    def test_retrieve_propagates_model_layer_not_found(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A 404 ApiError raised by get_video_job propagates as an OpenAI 404 envelope.

        Ref: stdapi/api_providers/openai.py:_format_error
             stdapi/models/video/__init__.py:get_video_job
        """

        async def _get_video_job(_invocation_arn: str) -> VideoJob:
            msg = "Video not found."
            raise ApiError(msg, status=404)

        monkeypatch.setattr(openai_videos, "get_video_job", _get_video_job)
        video_id = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001
        response = app_client.get(f"/v1/videos/{video_id}")
        assert response.status_code == 404
        err = response.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert err["message"] == "Video not found."

    def test_retrieve_completed(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A completed job maps to a completed Video with progress 100.

        ``progress`` is derived from the status (AWS reports no percentage), and
        the unset ``error``/``expires_at`` members are omitted rather than sent as
        null.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_videos.py:_to_video
        """
        _stub_job(monkeypatch, "completed")
        video_id = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001
        response = app_client.get(f"/v1/videos/{video_id}")
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
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With a retention period configured the Video reports expires_at.

        Retention is a server setting, so the expiry is the job completion time
        plus that period rather than OpenAI's fixed window.

        Ref: https://stdapi.ai/api_openai_videos/
             stdapi/models/video/__init__.py:video_expires_at
        """
        monkeypatch.setattr(SETTINGS, "aws_s3_videos_expires_after", 3600)
        _stub_job(monkeypatch, "completed")
        video_id = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001
        response = app_client.get(f"/v1/videos/{video_id}")
        assert response.status_code == 200, response.text
        assert response.json()["expires_at"] == 1752000090 + 3600

    def test_retrieve_failed(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed job carries the AWS failure reason in the error payload.

        ``failureMessage`` from the AWS job state becomes the Video error message
        under a fixed machine-readable code, and a failed job stays at progress 0.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_GetAsyncInvoke.html
             stdapi/routes/openai_videos.py:_to_video
        """
        _stub_job(monkeypatch, "failed", "content filters blocked the prompt")
        video_id = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001
        response = app_client.get(f"/v1/videos/{video_id}")
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "failed"
        assert body["progress"] == 0
        assert body["error"] == {
            "code": "video_generation_failed",
            "message": "content filters blocked the prompt",
        }

    def test_retrieve_bad_id_is_not_found(self, app_client: TestClient) -> None:
        """An undecodable video ID surfaces as an OpenAI 404 error.

        The ID is self-describing, so a syntactically valid but undecodable one is
        answered locally without any AWS lookup.

        Ref: stdapi/routes/openai_videos.py:_decode_video_id
        """
        response = app_client.get("/v1/videos/video_bogus")
        assert response.status_code == 404
        err = response.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert "video_bogus" in err["message"]

    def test_list_videos(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The list endpoint pages the backend listings as Video objects.

        The ``after`` cursor is a video ID that must be decoded to the invocation
        ARN the model layer pages on, and first_id/last_id come from the page.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_videos.py:list_videos
        """
        calls: list[dict[str, Any]] = []

        async def _list_video_jobs(**kwargs: Any) -> tuple[list[VideoListing], bool]:  # noqa: ANN401
            calls.append(kwargs)
            return [VideoListing(_job("completed"), 6, "1280x720")], True

        monkeypatch.setattr(openai_videos, "list_video_jobs", _list_video_jobs)
        after = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001

        response = app_client.get(f"/v1/videos?limit=5&order=asc&after={after}")

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

    def test_list_videos_bad_cursor_is_not_found(self, app_client: TestClient) -> None:
        """An undecodable after cursor surfaces as an OpenAI 404 error.

        Ref: stdapi/routes/openai_videos.py:list_videos
        """
        response = app_client.get("/v1/videos?after=video_bogus")
        assert response.status_code == 404
        err = response.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert "video_bogus" in err["message"]

    def test_content_streams_video(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A completed job's MP4 is streamed with the video/mp4 content type.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_videos.py:get_video_content
        """
        _stub_job(monkeypatch, "completed")

        async def _open_video_content(_job: VideoJob) -> Any:  # noqa: ANN401
            return iter([b"mp4-", b"bytes"])

        monkeypatch.setattr(openai_videos, "open_video_content", _open_video_content)
        video_id = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001
        response = app_client.get(f"/v1/videos/{video_id}/content")
        assert response.status_code == 200
        assert response.headers["content-type"] == "video/mp4"
        assert response.content == b"mp4-bytes"

    def test_content_not_ready(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Downloading an unfinished job's content is a 404, like upstream.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_videos.py:get_video_content
        """
        _stub_job(monkeypatch, "in_progress")
        video_id = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001
        response = app_client.get(f"/v1/videos/{video_id}/content")
        assert response.status_code == 404
        err = response.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert "not ready" in err["message"]

    def test_content_variant_not_available(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A resolvable job with an unsupported variant is a 400.

        OpenAI offers video/thumbnail/spritesheet variants; Bedrock produces only
        the MP4, so the other variants are refused instead of silently returning
        the video.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_videos.py:get_video_content
        """
        _stub_job(monkeypatch, "completed")
        video_id = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001
        response = app_client.get(f"/v1/videos/{video_id}/content?variant=thumbnail")
        assert response.status_code == 400
        err = response.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert "variant" in err["message"]

    def test_content_unknown_id_with_variant_is_not_found(
        self, app_client: TestClient
    ) -> None:
        """An unresolvable video ID with variant=thumbnail is a 404, not the variant 400.

        The ID is resolved before the variant is checked, so an unknown video is
        reported as missing whatever variant was asked for.

        Ref: stdapi/routes/openai_videos.py:get_video_content
        """
        response = app_client.get("/v1/videos/video_bogus/content?variant=thumbnail")
        assert response.status_code == 404
        assert "video_bogus" in response.json()["error"]["message"], (
            "the ID must be resolved before the variant is validated"
        )

    def test_delete_completed(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deleting a completed job removes its output and confirms deletion.

        Only the S3 output is removed: the AWS invocation record stays until AWS
        expires it, so a deleted video remains listable.

        Ref: https://stdapi.ai/api_openai_videos/
             stdapi/models/video/__init__.py:delete_video_output
        """
        _stub_job(monkeypatch, "completed")
        deleted: list[VideoJob] = []

        async def _delete_video_output(job: VideoJob) -> None:
            deleted.append(job)

        monkeypatch.setattr(openai_videos, "delete_video_output", _delete_video_output)
        video_id = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001
        response = app_client.delete(f"/v1/videos/{video_id}")
        assert response.status_code == 200, response.text
        assert response.json() == {
            "id": video_id,
            "object": "video.deleted",
            "deleted": True,
        }
        assert [job.invocation_arn for job in deleted] == [_ARN]

    def test_delete_in_progress_is_rejected(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job still being processed cannot be deleted, like upstream.

        Bedrock async invocations cannot be cancelled, so an in-progress job is
        refused rather than reported as deleted.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_GetAsyncInvoke.html
             stdapi/routes/openai_videos.py:delete_video
        """
        _stub_job(monkeypatch, "in_progress")
        video_id = openai_videos._encode_video_id(_ARN, 6, "1280x720")  # noqa: SLF001
        response = app_client.delete(f"/v1/videos/{video_id}")
        assert response.status_code == 400
        err = response.json()["error"]
        assert err["type"] == "invalid_request_error"
        assert "still being processed" in err["message"]


@pytest.mark.local
class TestOpenAIVideoAuthRejection:
    """A wrong bearer token is rejected with a 401 OpenAI envelope before any model call.

    Uses the session-wide ``test_client`` (lifespan-started, unlike the
    lifespan-free ``app_client`` fixture) so the auth handler is actually
    initialized and able to reject a bad token.

    Ref: stdapi/auth.py:authenticate
         stdapi/api_providers/openai.py:_format_error
    """

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("post", "/v1/videos"),
            ("get", f"/v1/videos/{_DUMMY_VIDEO_ID}"),
            ("get", "/v1/videos"),
            ("get", f"/v1/videos/{_DUMMY_VIDEO_ID}/content"),
            ("delete", f"/v1/videos/{_DUMMY_VIDEO_ID}"),
        ],
        ids=["create", "retrieve", "list", "content", "delete"],
    )
    def test_wrong_bearer_token_is_rejected(
        self,
        test_client: TestClient,
        video_backend: _StubVideoModel,
        monkeypatch: pytest.MonkeyPatch,
        method: str,
        path: str,
    ) -> None:
        """A wrong bearer token yields 401 without reaching the model layer.

        Every /v1/videos method authenticates before any AWS call, and the 401
        carries the full OpenAI error envelope with type
        ``authentication_error``.

        Ref: stdapi/routes/openai_videos.py:authenticate
        """
        model_calls: list[object] = []

        async def _get_video_job(invocation_arn: str) -> VideoJob:
            model_calls.append(invocation_arn)
            return _job("completed")

        async def _list_video_jobs(**kwargs: Any) -> tuple[list[VideoListing], bool]:  # noqa: ANN401
            model_calls.append(kwargs)
            return [], False

        monkeypatch.setattr(openai_videos, "get_video_job", _get_video_job)
        monkeypatch.setattr(openai_videos, "list_video_jobs", _list_video_jobs)

        kwargs: dict[str, Any] = (
            {"json": {"model": "amazon.nova-reel-v1:0", "prompt": "a cat"}}
            if method == "post"
            else {}
        )
        response = getattr(test_client, method)(
            path, headers={"Authorization": "Bearer wrong-key"}, **kwargs
        )

        assert response.status_code == 401
        body = response.json()
        assert set(body.keys()) == {"error"}
        err = body["error"]
        assert set(err.keys()) == {"message", "type", "param", "code"}
        assert err["type"] == "authentication_error"
        assert not model_calls
        assert not video_backend.calls


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
        # Starlette types its TestClient against httpx2; the alias in conftest makes
        # the response it returns at runtime the httpx one this annotation names.
        response: httpx.Response = test_client.get(  # type: ignore[assignment]
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
    """Live /v1/videos lifecycle through the OpenAI SDK (local or official API).

    Generation is billed per output second, so these tests keep to the shortest
    supported clip and assert only what a real job guarantees: the synthesized
    status progression, the echoed parameters and a downloadable MP4.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         https://stdapi.ai/api_openai_videos/
    """

    @pytest.mark.video
    @pytest.mark.slow
    def test_video_generation_lifecycle(
        self, openai_client: OpenAI, video_generation_model: str, use_official_api: bool
    ) -> None:
        """Create from a reference frame, poll, download, and delete a video.

        A fresh job starts in a non-terminal status, reaches ``progress`` 100 with
        a completion timestamp, then serves an MP4 (its 5th-8th bytes are the
        ISO-BMFF ``ftyp`` box) until it is deleted.

        The job is started from a multipart ``input_reference`` frame, generated
        at the requested size because both backends constrain the first frame's
        dimensions. That is the richer of the two request payloads, and a clip
        costs five seconds of generation, so this covers the plain prompt-only
        payload too -- which ``tests/test_models_video.py`` pins offline. Whether
        the video actually starts from that frame is not observable here.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-luma.html
             stdapi/routes/openai_videos.py:get_video_content
        """
        # Luma bills 540p at half the 720p rate; Sora has no 540p size.
        size = "1280x720" if use_official_api else "960x540"
        video = openai_client.videos.create(
            model=video_generation_model,
            prompt="A calico cat playing a piano",
            size=size,  # type: ignore[arg-type]
            input_reference=("frame.png", _reference_frame(size), "image/png"),
        )
        video_id = video.id
        try:
            assert video.object == "video"
            assert video.status in ("queued", "in_progress")
            assert video.size == size
            assert video.error is None

            video = _wait_for_completion(openai_client, video_id)
            assert video.id == video_id
            assert video.progress == 100
            assert video.completed_at
            assert video.error is None

            content = openai_client.videos.download_content(video_id).read()
            assert content[4:8] == b"ftyp"
            assert len(content) > 1024, "a generated clip is never that small"

            deleted = openai_client.videos.delete(video_id)
            assert deleted.deleted is True
            assert deleted.object == "video.deleted"
            assert deleted.id == video_id
        finally:
            # Best-effort cleanup when an assertion fails mid-lifecycle.
            with suppress(Exception):
                openai_client.videos.delete(video.id)

    @pytest.mark.video
    @pytest.mark.slow
    @pytest.mark.gateway("/search_models is not part of the official API")
    def test_every_advertised_model_generates(
        self,
        request: pytest.FixtureRequest,
        openai_client: OpenAI,
        test_client: TestClient | None,
        api_key: str,
        video_generation_model: str,
    ) -> None:
        """Every model advertising the route serves the full lifecycle.

        With ``AWS_BEDROCK_LEGACY`` enabled this also covers the legacy Amazon
        Nova Reel models. Each job is created with no seconds/size so every model
        runs at its own default, which is also its cheapest clip.

        Ref: stdapi/models/__init__.py:_compute_model_capabilities
             stdapi/routes/openai_videos.py:create_video
        """
        model_ids = _advertised_model_ids(request, test_client, api_key)
        assert model_ids
        for model_id in model_ids:
            if model_id == video_generation_model:
                continue  # Covered by test_video_generation_lifecycle.
            video = openai_client.videos.create(
                model=model_id, prompt="Waves rolling onto a sandy beach"
            )
            try:
                assert video.model == model_id
                assert video.status in ("queued", "in_progress"), model_id
                video = _wait_for_completion(openai_client, video.id)
                assert video.progress == 100, model_id
                content = openai_client.videos.download_content(video.id).read()
                assert content[4:8] == b"ftyp", model_id
                assert len(content) > 1024, model_id
            finally:
                with suppress(Exception):
                    openai_client.videos.delete(video.id)

    @pytest.mark.video
    @pytest.mark.slow
    @pytest.mark.gateway("/search_models is not part of the official API")
    def test_nova_reel_multi_shot_duration(
        self,
        request: pytest.FixtureRequest,
        openai_client: OpenAI,
        test_client: TestClient | None,
        api_key: str,
    ) -> None:
        """Nova Reel serves durations beyond 6s through automated multi-shot.

        12 s is two 6-second shots, which Nova Reel only renders through the
        MULTI_SHOT_AUTOMATED task on ``amazon.nova-reel-v1:1``; the duration and
        the model's only size are echoed for the whole job lifetime.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/video-gen-code-examples2.html
             https://docs.aws.amazon.com/nova/latest/userguide/video-generation.html
        """
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
            assert video.size == "1280x720"
            assert video.model == "amazon.nova-reel-v1:1"
            video = _wait_for_completion(openai_client, video.id)
            assert video.seconds == "12"
            content = openai_client.videos.download_content(video.id).read()
            assert content[4:8] == b"ftyp"
            assert len(content) > 1024
        finally:
            with suppress(Exception):
                openai_client.videos.delete(video.id)

    @pytest.mark.gateway("Bedrock video models require a stdapi server")
    def test_unsupported_duration_returns_clean_error(
        self, openai_client: OpenAI
    ) -> None:
        """A duration unsupported by the model is rejected with a clean 400.

        Luma Ray renders 5 s or 9 s only. The model input is built before the
        async invocation, so the rejection carries the gateway's own message
        instead of an AWS ValidationException.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-luma.html
             stdapi/models/video/luma_ray.py:VideoModel.build_generation_input
        """
        with pytest.raises(BadRequestError, match="seconds") as exc_info:
            openai_client.videos.create(
                model="luma.ray-v2:0", prompt="a cat", seconds="4"
            )
        assert exc_info.value.status_code == 400
        body = exc_info.value.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        assert "[5, 9]" in body["message"], (
            "the model layer must name the durations it supports"
        )

    @pytest.mark.gateway("Bedrock video models require a stdapi server")
    def test_unsupported_size_returns_clean_error(self, openai_client: OpenAI) -> None:
        """A size unsupported by the model is rejected with a clean 400.

        1024x1024 is a supported aspect ratio at an unsupported resolution, so it
        passes the request schema and is refused by the model layer before any
        async invocation is started.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-luma.html
             stdapi/models/video/luma_ray.py:VideoModel.build_generation_input
        """
        with pytest.raises(BadRequestError, match="size") as exc_info:
            openai_client.videos.create(
                model="luma.ray-v2:0",
                prompt="a cat",
                size="1024x1024",  # type: ignore[arg-type]
            )
        assert exc_info.value.status_code == 400
        body = exc_info.value.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        assert "1024x1024" in body["message"]

    @pytest.mark.gateway("listing scope differs on the official API")
    def test_list_videos(self, openai_client: OpenAI) -> None:
        """Previously generated jobs are listed newest first with duration and size.

        Duration and size are recovered from the invocation tags (AWS job summaries
        carry neither), while the prompt is not stored anywhere and so is never
        reported on a listed job.

        Ref: https://stdapi.ai/api_openai_videos/
             stdapi/models/video/__init__.py:list_video_jobs
        """
        page = openai_client.videos.list(limit=5)
        # AWS retains async job records for a limited time; a fresh account
        # may legitimately have nothing to list.
        videos = list(page.data)
        assert len(videos) <= 5, "the limit must be honored"
        assert all(video.id.startswith("video_") for video in videos)
        assert all(video.object == "video" for video in videos)
        assert all(video.prompt is None for video in videos), (
            "the prompt is echoed only by create"
        )
        if videos:
            assert videos[0].seconds
            assert videos[0].size
            assert videos[0].created_at >= videos[-1].created_at
