"""Video generation models base classes and dynamic registry.

Modules of this package define a ``VideoModel`` class with a ``MATCHER``
(string prefix or compiled regex) matching the Bedrock model identifier, and
are auto-loaded once on import.

Video generation runs asynchronously on AWS Bedrock: a request starts an
async invocation whose MP4 output lands in the regional S3 bucket. AWS keeps
all job state, addressed by invocation ARN, so this server stays stateless.
"""

from abc import abstractmethod
from asyncio import gather
from contextlib import suppress
from functools import partial
from re import compile as re_compile
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple

from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel

from stdapi.api_errors import ApiError, UnsupportedModelError
from stdapi.aws import get_client
from stdapi.aws_bedrock import handle_bedrock_client_error
from stdapi.aws_s3 import get_s3_bucket_for_region, require_s3_bucket_for_region
from stdapi.config import SETTINGS
from stdapi.models import (
    ModelBase,
    compute_candidate_regions,
    get_model,
    load_model_plugins,
    resolve_routed_model_id,
    route_and_execute,
)
from stdapi.models.capabilities import Capability
from stdapi.monitoring import REQUEST_LOG, build_metadata, log_error_details
from stdapi.usage import record_bedrock_usage
from stdapi.utils import now_utc_timestamp

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Mapping
    from re import Pattern

    from types_aiobotocore_bedrock.client import BedrockClient
    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_runtime.client import BedrockRuntimeClient
    from types_aiobotocore_s3.client import S3Client

    from stdapi.input_file import InputFile
    from stdapi.types import JsonMapping

#: Bedrock async invocation ARN, capturing the region for client routing.
_INVOCATION_ARN_PATTERN = re_compile(
    r"^arn:aws[a-z-]*:bedrock:([a-z0-9-]+):\d{12}:async-invoke/[0-9a-z]+$"
)

#: Video generation job status.
_JobStatus = Literal["in_progress", "completed", "failed"]

#: Bedrock async invocation status mapped to the API job status.
_JOB_STATUS: dict[str, _JobStatus] = {
    "InProgress": "in_progress",
    "Completed": "completed",
    "Failed": "failed",
}

#: GetAsyncInvoke error codes meaning the job is gone, foreign, or invalid.
_JOB_NOT_FOUND_ERRORS = frozenset(
    {"ValidationException", "AccessDeniedException", "ResourceNotFoundException"}
)

#: Resource tag key carrying the effective video duration, set at job start.
_SECONDS_TAG = "stdapi-ai.seconds"

#: Resource tag key carrying the effective video size, set at job start.
_SIZE_TAG = "stdapi-ai.size"

#: Maximum async invocations scanned per region when listing videos.
_LIST_SCAN_LIMIT: int = 1000

#: The feature name a caller reads when no bucket can hold this deployment's videos.
_VIDEO_FEATURE: str = "Video generation"


class ReferenceImage(NamedTuple):
    """Resolved reference image passed to model input builders.

    Attributes:
        media_type: Image MIME type (e.g. "image/png").
        base64_data: Base64-encoded image bytes.
    """

    media_type: str
    base64_data: str


class VideoGenerationStart(BaseModel):
    """Started video generation job.

    Attributes:
        invocation_arn: Bedrock async invocation ARN identifying the job.
        seconds: Effective video duration in seconds.
        size: Effective video size as "<width>x<height>".
    """

    invocation_arn: str
    seconds: int
    size: str


class VideoJob(BaseModel):
    """State of a video generation job, as reported by AWS Bedrock.

    Attributes:
        invocation_arn: Bedrock async invocation ARN identifying the job.
        model_id: Bedrock model identifier that serves the job.
        status: Current job status.
        created_at: Job submission time as a Unix timestamp.
        completed_at: Job end time as a Unix timestamp, once finished.
        failure_message: Failure reason when the job failed.
        output_bucket: S3 bucket holding the job output.
        output_prefix: S3 key prefix (job folder) holding the job output.
    """

    invocation_arn: str
    model_id: str
    status: _JobStatus
    created_at: int
    completed_at: int | None = None
    failure_message: str | None = None
    output_bucket: str
    output_prefix: str


class VideoModelBase(ModelBase[Any, Any]):
    """Base class for provider-specific video generation models."""

    __slots__ = ()

    #: Video duration in seconds used when the request does not specify one.
    DEFAULT_SECONDS: ClassVar[int]

    #: Video size used when the request does not specify one.
    DEFAULT_SIZE: ClassVar[str]

    @classmethod
    def get_supported_operations(cls) -> Capability:
        """Return capability flags for route-based model matching.

        Returns:
            Capability flags.
        """
        return Capability.VIDEO_GENERATION

    @abstractmethod
    def build_generation_input(
        self,
        prompt: str,
        *,
        seconds: int,
        size: str,
        reference_image: ReferenceImage | None,
        extra_params: JsonMapping,
    ) -> JsonMapping:
        """Build the provider-specific async invocation input.

        Args:
            prompt: Text prompt describing the video.
            seconds: Video duration in seconds.
            size: Video size as "<width>x<height>".
            reference_image: Optional starting keyframe image.
            extra_params: Extra model parameters.

        Returns:
            The ``modelInput`` payload for ``StartAsyncInvoke``.

        Raises:
            ApiError: When a parameter is not supported by the model.
        """

    def output_seconds_spec(self, _size: str) -> str:
        """Return this model's pricing spec bucket for a video size.

        Overridden by models whose AWS pricing distinguishes resolutions
        (e.g. a "hd" bucket); the default is the undifferentiated bucket.

        Args:
            _size: Effective video size as "<width>x<height>".

        Returns:
            The spec bucket AWS prices this size under, or "" when this
            model's pricing has no resolution-differentiated bucket.
        """
        return ""

    async def start_video_generation(
        self,
        prompt: str,
        *,
        seconds: int | None,
        size: str | None,
        reference_image: InputFile | None,
        extra_params: JsonMapping,
    ) -> VideoGenerationStart:
        """Start an asynchronous video generation job.

        Args:
            prompt: Text prompt describing the video.
            seconds: Video duration in seconds, or None for the model default.
            size: Video size as "<width>x<height>", or None for the model default.
            reference_image: Optional starting keyframe image.
            extra_params: Extra model parameters.

        Returns:
            The started job with its effective duration and size.
        """
        seconds = seconds if seconds is not None else self.DEFAULT_SECONDS
        size = size or self.DEFAULT_SIZE
        image = (
            ReferenceImage(
                await reference_image.get_content_type(),
                await reference_image.to_base64(),
            )
            if reference_image is not None
            else None
        )
        body = self.build_generation_input(
            prompt,
            seconds=seconds,
            size=size,
            reference_image=image,
            extra_params=extra_params,
        )
        candidates = await compute_candidate_regions(self._model_id, s3_required=True)
        invocation_arn, region = await route_and_execute(
            self._model_id,
            candidates,
            partial(
                self._start_in_region,
                body,
                seconds,
                size,
                single_region=len(candidates) == 1,
            ),
        )
        # Billed at submission, not completion: job state is not tracked here, so
        # a later failure is never reconciled against this recorded usage.
        record_bedrock_usage(
            self._model_id,
            region=region,
            output_seconds=seconds,
            output_seconds_spec=self.output_seconds_spec(size),
        )
        return VideoGenerationStart(
            invocation_arn=invocation_arn, seconds=seconds, size=size
        )

    async def _start_in_region(
        self,
        body: JsonMapping,
        seconds: int,
        size: str,
        region: RegionName,
        *,
        single_region: bool,
    ) -> tuple[str, RegionName]:
        """Start the async invocation in *region*, writing output to its regional bucket.

        The effective duration and size are stored as resource tags so that
        listing can recover them (they are not part of the AWS job state).

        Args:
            body: The ``modelInput`` payload.
            seconds: Effective video duration in seconds.
            size: Effective video size as "<width>x<height>".
            region: Target AWS region.
            single_region: Selects the full-retry or no-retry botocore client,
                mirroring :func:`stdapi.aws_bedrock.bedrock_client`.

        Returns:
            Tuple of (invocation ARN, region that served the call).
        """
        bucket = require_s3_bucket_for_region(region, feature=_VIDEO_FEATURE)
        resolved_model_id = await resolve_routed_model_id(
            self._model_id, region, inference_profile=False
        )
        client: BedrockRuntimeClient = get_client(
            (
                "bedrock-runtime"
                if single_region or SETTINGS.aws_bedrock_region_routing == "disabled"
                else "bedrock-runtime.no-retry"
            ),
            region,
        )
        tags = build_metadata(apn=True) | {_SECONDS_TAG: str(seconds), _SIZE_TAG: size}
        with handle_bedrock_client_error():
            response = await client.start_async_invoke(
                modelId=resolved_model_id,
                modelInput=body,
                outputDataConfig={
                    "s3OutputDataConfig": {
                        "s3Uri": f"s3://{bucket}/{SETTINGS.aws_s3_videos_prefix}"
                    }
                },
                tags=[{"key": k, "value": v} for k, v in tags.items()],
            )
        return response["invocationArn"], region


def _invocation_region(invocation_arn: str) -> RegionName:
    """Extract and validate the region from an async invocation ARN.

    Args:
        invocation_arn: Bedrock async invocation ARN.

    Returns:
        The AWS region embedded in the ARN.

    Raises:
        ApiError: 404 when the ARN is malformed or targets an unconfigured region.
    """
    match = _INVOCATION_ARN_PATTERN.match(invocation_arn)
    if match is None or match.group(1) not in SETTINGS.aws_bedrock_regions:
        msg = "Video not found."
        raise ApiError(msg, status=404)
    return match.group(1)  # type: ignore[return-value]


def _region_videos_uri_prefix(region: RegionName) -> str | None:
    """Return this server's S3 URI prefix for video outputs in *region*.

    Args:
        region: AWS region identifier.

    Returns:
        The ``s3://bucket/prefix`` URI prefix this server's jobs are written
        under in *region*, or ``None`` when the region has no configured bucket.
    """
    bucket = get_s3_bucket_for_region(region)
    if not bucket:
        return None
    prefix = SETTINGS.aws_s3_videos_prefix
    # A non-empty prefix must end in "/": otherwise this ownership check would
    # also match an unrelated sibling prefix (e.g. "videos" matching "videos-other").
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return f"s3://{bucket}/{prefix}"


async def get_video_job(invocation_arn: str) -> VideoJob:
    """Fetch the current state of a video generation job from AWS Bedrock.

    Only jobs writing under this server's configured videos bucket/prefix are
    served; any other async invocation in the AWS account (started by this
    server's other regions notwithstanding, or by another application
    entirely) is reported as not found, matching the scope enforced by
    :func:`list_video_jobs`.

    Args:
        invocation_arn: Bedrock async invocation ARN.

    Returns:
        The job state.

    Raises:
        ApiError: 404 when the ARN is invalid, targets an unconfigured region,
            references an aged-out, deleted, or foreign-account invocation, or
            its output falls outside this server's videos bucket/prefix;
            on AWS client errors otherwise.
    """
    region = _invocation_region(invocation_arn)
    client: BedrockRuntimeClient = get_client("bedrock-runtime", region)
    try:
        with handle_bedrock_client_error():
            response = await client.get_async_invoke(invocationArn=invocation_arn)
    except ClientError as exc:
        # Aged-out, deleted, and foreign-account invocations get the same 404
        # as malformed IDs, avoiding an existence oracle.
        if exc.response["Error"]["Code"] in _JOB_NOT_FOUND_ERRORS:
            msg = "Video not found."
            raise ApiError(msg, status=404) from exc
        raise
    s3_uri = response["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
    uri_prefix = _region_videos_uri_prefix(region)
    if uri_prefix is None or not s3_uri.startswith(uri_prefix):
        msg = "Video not found."
        raise ApiError(msg, status=404)
    return _to_video_job(invocation_arn, response)


def _to_video_job(invocation_arn: str, state: Mapping[str, Any]) -> VideoJob:
    """Map a Bedrock async invocation state (get or list item) to a ``VideoJob``.

    Args:
        invocation_arn: Bedrock async invocation ARN.
        state: ``GetAsyncInvoke`` response or ``ListAsyncInvokes`` summary.

    Returns:
        The job state.
    """
    bucket, _, prefix = (
        state["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"]
        .removeprefix("s3://")
        .partition("/")
    )
    return VideoJob(
        invocation_arn=invocation_arn,
        model_id=state["modelArn"].rsplit("/", 1)[-1],
        # An unknown Bedrock status is assumed non-terminal (still running).
        status=_JOB_STATUS.get(state["status"], "in_progress"),
        created_at=int(state["submitTime"].timestamp()),
        completed_at=int(end.timestamp()) if (end := state.get("endTime")) else None,
        failure_message=state.get("failureMessage"),
        output_bucket=bucket,
        output_prefix=prefix,
    )


class VideoListing(NamedTuple):
    """A listed video job with its effective duration and size.

    Attributes:
        job: The video generation job state.
        seconds: Video duration in seconds.
        size: Video size as "<width>x<height>".
    """

    job: VideoJob
    seconds: int
    size: str


async def _scan_region_jobs(
    region: RegionName, sort_order: Literal["Ascending", "Descending"]
) -> list[VideoJob]:
    """List this server's video generation jobs in *region*.

    Only invocations writing under the configured videos prefix of the
    region's bucket are returned; other async invocations in the AWS account
    are ignored. The scan is capped at ``_LIST_SCAN_LIMIT`` invocations.

    Args:
        region: AWS region to scan.
        sort_order: Bedrock sort order by submission time.

    Returns:
        Matching jobs, in AWS submission-time order.
    """
    uri_prefix = _region_videos_uri_prefix(region)
    if uri_prefix is None:
        return []
    client: BedrockRuntimeClient = get_client("bedrock-runtime", region)
    jobs: list[VideoJob] = []
    scanned = 0
    token: str | None = None
    with handle_bedrock_client_error():
        while scanned < _LIST_SCAN_LIMIT:
            response = await client.list_async_invokes(
                maxResults=100,
                sortBy="SubmissionTime",
                sortOrder=sort_order,
                **({"nextToken": token} if token else {}),  # type: ignore[arg-type]
            )
            summaries = response.get("asyncInvokeSummaries", ())
            scanned += len(summaries)
            jobs.extend(
                _to_video_job(summary["invocationArn"], summary)
                for summary in summaries
                if summary["outputDataConfig"]["s3OutputDataConfig"][
                    "s3Uri"
                ].startswith(uri_prefix)
            )
            token = response.get("nextToken")
            if not token:
                break
    return jobs


async def _listing_details(job: VideoJob) -> VideoListing | None:
    """Resolve a job's duration and size from its resource tags.

    Falls back to the model defaults when the tags are absent (jobs started
    by older server versions), and drops the job when its model is unknown.

    Args:
        job: The video generation job.

    Returns:
        The listing, or ``None`` when the model is not a known video model.
    """
    client: BedrockClient = get_client(
        "bedrock", _invocation_region(job.invocation_arn)
    )
    tags: dict[str, str] = {}
    with suppress(ClientError):
        response = await client.list_tags_for_resource(resourceARN=job.invocation_arn)
        tags = {tag["key"]: tag["value"] for tag in response.get("tags", ())}
    seconds, size = tags.get(_SECONDS_TAG, ""), tags.get(_SIZE_TAG, "")
    if not seconds.isdigit() or not size:
        try:
            model = get_video_model(job.model_id)
        except UnsupportedModelError:
            return None
        seconds = seconds if seconds.isdigit() else str(model.DEFAULT_SECONDS)
        size = size or model.DEFAULT_SIZE
    return VideoListing(job, int(seconds), size)


async def list_video_jobs(
    *,
    order: Literal["asc", "desc"] = "desc",
    after_arn: str | None = None,
    limit: int = 20,
) -> tuple[list[VideoListing], bool]:
    """List this server's video generation jobs across all configured regions.

    Jobs are visible while AWS Bedrock retains their async invocation record;
    the listing is independent of whether the S3 output still exists. A region
    whose scan fails with an AWS error is skipped (surfaced as a request-log
    warning) instead of failing the whole listing; the listing only fails when
    every configured region fails.

    Args:
        order: Sort order by creation time.
        after_arn: Return only jobs strictly after this invocation ARN, or
            ``None`` to start from the first job. An unknown ARN yields an
            empty page.
        limit: Maximum number of jobs to return.

    Returns:
        Tuple of (page of listings, whether more jobs remain).

    Raises:
        BotoCoreError: When every configured region fails (first error).
        ClientError: When every configured region fails (first error).
    """
    sort_order: Literal["Ascending", "Descending"] = (
        "Ascending" if order == "asc" else "Descending"
    )
    regions = list(SETTINGS.aws_bedrock_regions)
    region_results = await gather(
        *(_scan_region_jobs(region, sort_order) for region in regions),
        return_exceptions=True,
    )
    jobs: list[VideoJob] = []
    errors: list[BaseException] = []
    failed_regions: dict[str, str] = {}
    for region, result in zip(regions, region_results, strict=True):
        if isinstance(result, BaseException):
            if not isinstance(result, (BotoCoreError, ClientError)):
                raise result
            errors.append(result)
            failed_regions[region] = f"{type(result).__name__}: {result}"
            continue
        jobs.extend(result)
    if errors and len(errors) == len(regions):
        raise errors[0]
    if failed_regions and REQUEST_LOG.get(None) is not None:
        log_error_details(
            {"unreachable_bedrock_regions": failed_regions},  # type: ignore[dict-item]
            level="warning",
        )
    jobs.sort(
        key=lambda job: (job.created_at, job.invocation_arn), reverse=order == "desc"
    )
    if after_arn is not None:
        index = next(
            (i for i, job in enumerate(jobs) if job.invocation_arn == after_arn), None
        )
        jobs = jobs[index + 1 :] if index is not None else []
    return await _resolve_listing_page(jobs, limit)


async def _resolve_listing_page(
    jobs: list[VideoJob], limit: int
) -> tuple[list[VideoListing], bool]:
    """Resolve listing details for a page, skipping unknown-model jobs.

    Keeps resolving scanned jobs (in order) until ``limit`` resolvable
    listings are collected or ``jobs`` is exhausted, so unknown-model jobs
    dropped by :func:`_listing_details` never shrink the page below its
    limit while more resolvable jobs remain -- doing this after truncating
    to ``jobs[:limit]`` could yield a short or empty page with
    ``has_more=True``, stalling SDK pagers that stop on an empty page.

    Args:
        jobs: Scanned jobs, already ordered and cursor-filtered.
        limit: Maximum number of listings to return.

    Returns:
        Tuple of (page of listings, whether more resolvable jobs remain).
    """
    listings: list[VideoListing] = []
    index = 0
    while index < len(jobs) and len(listings) < limit:
        batch = jobs[index : index + (limit - len(listings))]
        index += len(batch)
        listings.extend(
            listing
            for listing in await gather(*map(_listing_details, batch))
            if listing is not None
        )
    has_more = index < len(jobs)
    return listings, has_more


def video_expires_at(job: VideoJob) -> int | None:
    """Return the expiry timestamp of a job's video output, if any.

    Args:
        job: The video generation job.

    Returns:
        Unix timestamp of expiry, or ``None`` when no retention period is
        configured or the job has not completed.
    """
    if SETTINGS.aws_s3_videos_expires_after is None or job.completed_at is None:
        return None
    return job.completed_at + SETTINGS.aws_s3_videos_expires_after


async def open_video_content(job: VideoJob) -> AsyncIterator[bytes]:
    """Open a byte-chunk stream over a completed job's MP4 output.

    Args:
        job: The completed video generation job.

    Returns:
        Async iterator over the MP4 content.

    Raises:
        ApiError: 404 when the output has expired or no longer exists (e.g. deleted).
    """
    expires_at = video_expires_at(job)
    if expires_at is not None and now_utc_timestamp() >= expires_at:
        msg = "Video content not found."
        raise ApiError(msg, status=404)
    s3: S3Client = get_client("s3", _invocation_region(job.invocation_arn))
    try:
        response = await s3.get_object(
            Bucket=job.output_bucket, Key=f"{job.output_prefix}/output.mp4"
        )
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code == "AccessDenied" and REQUEST_LOG.get(None) is not None:
            # Without s3:ListBucket, S3 reports a missing key as AccessDenied.
            log_error_details(
                "S3 AccessDenied on video content download: the object is gone"
                " or the server lacks S3 permissions on the videos bucket.",
                level="warning",
            )
        if code in ("NoSuchKey", "404", "AccessDenied"):
            msg = "Video content not found."
            raise ApiError(msg, status=404) from exc
        raise
    return response["Body"].iter_chunks()


async def delete_video_output(job: VideoJob) -> None:
    """Delete all S3 objects produced by a video generation job.

    Deletion is best-effort: keys S3 fails to delete are logged as a warning
    and left to the bucket lifecycle expiry.

    Args:
        job: The video generation job to clean up.
    """
    s3: S3Client = get_client("s3", _invocation_region(job.invocation_arn))
    failed_keys = 0
    token: str | None = None
    while True:
        listed = await s3.list_objects_v2(
            Bucket=job.output_bucket,
            Prefix=f"{job.output_prefix}/",
            **({"ContinuationToken": token} if token else {}),  # type: ignore[arg-type]
        )
        if keys := [{"Key": obj["Key"]} for obj in listed.get("Contents", ())]:
            deleted = await s3.delete_objects(
                Bucket=job.output_bucket,
                Delete={"Objects": keys},  # type: ignore[typeddict-item]
            )
            failed_keys += len(deleted.get("Errors", ()))
        token = listed.get("NextContinuationToken")
        if not token:
            break
    # Deletion stays best-effort: the S3 lifecycle expiry covers stragglers.
    if failed_keys and REQUEST_LOG.get(None) is not None:
        log_error_details(
            f"Video output deletion left {failed_keys} object(s) behind.",
            level="warning",
        )


#: Model ID (or pattern) to video model class registry.
_MODEL_REGISTRY: list[tuple[str | Pattern[str], type[VideoModelBase]]] = []
#: Instantiated video models by model ID.
_MODEL_CACHE: dict[str, VideoModelBase] = {}


def get_video_model(model_id: str) -> VideoModelBase:
    """Resolve the video model class matching the provided identifier.

    Args:
        model_id: The provider model identifier (e.g., "amazon.nova-reel-v1:0").

    Returns:
        The video model associated to the ``model_id``.

    Raises:
        UnsupportedModelError: If no registered video model matches ``model_id``.
    """
    return get_model(model_id, _MODEL_CACHE, _MODEL_REGISTRY, __name__)


load_model_plugins(
    class_type=VideoModelBase,  # type: ignore[type-abstract]
    package_name=__name__,
    registry=_MODEL_REGISTRY,
)
