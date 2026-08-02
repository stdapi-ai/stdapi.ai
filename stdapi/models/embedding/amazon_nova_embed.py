"""Amazon Nova multimodal embedding model.

- amazon.nova-2-multimodal-embeddings-v1:0
"""

from asyncio import gather
from math import ceil
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

from pydantic_core import from_json

from stdapi.api_errors import ApiError
from stdapi.aws import get_client
from stdapi.aws_bedrock import BEDROCK_BODY_SIZE_LIMIT, MIME_TYPES_TO_VIDEO_TYPE
from stdapi.aws_s3 import (
    BUCKET_TO_REGION,
    S3Object,
    put_s3_object,
    track_temporary_s3_objects,
)
from stdapi.input_file import InputFile
from stdapi.models import InvokeResult
from stdapi.models.embedding import EmbeddingModelBase, EmbeddingResponse
from stdapi.usage import record_bedrock_usage

if TYPE_CHECKING:
    from collections.abc import Sequence

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_s3.client import S3Client

    from stdapi.input_file import InputFileUrl
    from stdapi.pricing import Routing
    from stdapi.types import JsonMapping

_TaskType = Literal["SINGLE_EMBEDDING", "SEGMENTED_EMBEDDING"]
_EmbeddingPurpose = Literal[
    "GENERIC_INDEX",
    "GENERIC_RETRIEVAL",
    "CLASSIFICATION",
    "CLUSTERING",
    "TEXT_RETRIEVAL",
    "IMAGE_RETRIEVAL",
    "VIDEO_RETRIEVAL",
    "DOCUMENT_RETRIEVAL",
    "AUDIO_RETRIEVAL",
]
_EmbeddingType = Literal["TEXT", "IMAGE", "VIDEO", "AUDIO", "AUDIO_VIDEO_COMBINED"]
_EmbeddingDimension = Literal[256, 384, 1024, 3072]
_VideoEmbeddingMode = Literal["AUDIO_VIDEO_COMBINED", "AUDIO_VIDEO_SEPARATE"]
_TextTruncationMode = Literal["START", "END", "NONE"]
_ImageDetailLevel = Literal["STANDARD_IMAGE", "DOCUMENT_IMAGE"]
_FailureReasons = Literal[
    "RAI_VIOLATION_INPUT_TEXT_DEFLECTION",
    "RAI_VIOLATION_INPUT_IMAGE_DEFLECTION",
    "INVALID_CONTENT",
    "RATE_LIMIT_EXCEEDED",
    "INTERNAL_SERVER_EXCEPTION",
]
_MediaTypes = Literal["video", "text", "audio", "image"]

#: Fields that user can't overwrite
_RESERVED_MEDIA_PARAMS = frozenset({"source", "format", "value"})

#: Synchronous invoke limit size for S3 files
_SYNC_LIMIT_SIZES: dict[_MediaTypes, int] = {
    "image": 50_000_000,
    "video": 100_000_000,  # 30 seconds; 100 MB
    "audio": 100_000_000,  # 30 seconds; 100 MB
    "text": 50_000,  # 1 MB; 50,000 characters
}

#: Default segmented-embedding video mode when not otherwise specified.
_DEFAULT_VIDEO_EMBEDDING_MODE: _VideoEmbeddingMode = "AUDIO_VIDEO_COMBINED"
#: Default segmented-embedding text truncation mode when not otherwise specified.
_DEFAULT_TEXT_TRUNCATION_MODE: _TextTruncationMode = "END"
#: Maximum text length accepted by the synchronous (non-segmented) embedding path.
_TEXT_SIZE_LIMIT = 8192


class _S3Location(TypedDict):
    """S3 location for media sources."""

    uri: str


class _TextSource(TypedDict):
    """Text source for S3 URIs."""

    s3Location: _S3Location


class _MediaSource(TypedDict):
    """Media source for image, video, and audio inputs."""

    bytes: NotRequired[str]  # Base64 encoded
    s3Location: NotRequired[_S3Location]


class _TextInput(TypedDict):
    """Text input parameters."""

    truncationMode: _TextTruncationMode
    value: NotRequired[str]
    source: NotRequired[_TextSource]


class _ImageInput(TypedDict):
    """Image input parameters."""

    format: Literal["png", "jpeg", "gif", "webp"]
    source: _MediaSource
    detailLevel: NotRequired[_ImageDetailLevel]


class _VideoInput(TypedDict):
    """Video input parameters."""

    format: Literal[
        "mkv", "mov", "mp4", "webm", "flv", "mpeg", "mpg", "wmv", "three_gp"
    ]
    source: _MediaSource
    embeddingMode: _VideoEmbeddingMode


class _AudioInput(TypedDict):
    """Audio input parameters."""

    format: Literal["mp3", "wav", "ogg"]
    source: _MediaSource


class _EmbeddingParams(TypedDict):
    """embedding parameters."""

    embeddingPurpose: _EmbeddingPurpose
    embeddingDimension: NotRequired[_EmbeddingDimension]


class _SingleEmbeddingParams(_EmbeddingParams):
    """Single embedding parameters."""

    text: NotRequired[_TextInput]
    image: NotRequired[_ImageInput]
    video: NotRequired[_VideoInput]
    audio: NotRequired[_AudioInput]


class _TextSegmentationConfig(TypedDict):
    """Text segmentation configuration."""

    maxLengthChars: NotRequired[int]  # 800-50000, default 32000


class _MediaSegmentationConfig(TypedDict):
    """Audio/Video segmentation configuration."""

    durationSeconds: int  # 1-30, default 5


class _SegmentedTextInput(TypedDict):
    """Text input for segmented embedding."""

    truncationMode: _TextTruncationMode
    value: NotRequired[str]
    source: NotRequired[_TextSource]
    segmentationConfig: _TextSegmentationConfig


class _SegmentedVideoInput(TypedDict):
    """Video input for segmented embedding."""

    format: Literal[
        "mkv", "mov", "mp4", "webm", "flv", "mpeg", "mpg", "wmv", "three_gp"
    ]
    source: _MediaSource
    embeddingMode: _VideoEmbeddingMode
    segmentationConfig: _MediaSegmentationConfig


class _SegmentedAudioInput(TypedDict):
    """Audio input for segmented embedding."""

    format: Literal["mp3", "wav", "ogg"]
    source: _MediaSource
    segmentationConfig: _MediaSegmentationConfig


class _SegmentedEmbeddingParams(_EmbeddingParams):
    """Segmented embedding parameters."""

    text: NotRequired[_SegmentedTextInput]
    image: NotRequired[_ImageInput]
    video: NotRequired[_SegmentedVideoInput]
    audio: NotRequired[_SegmentedAudioInput]


class _Request(TypedDict):
    """Amazon Nova embedding request parameters."""

    schemaVersion: NotRequired[Literal["nova-multimodal-embed-v1"]]
    taskType: _TaskType
    singleEmbeddingParams: NotRequired[_SingleEmbeddingParams]
    segmentedEmbeddingParams: NotRequired[_SegmentedEmbeddingParams]


class _EmbeddingData(TypedDict):
    """Embedding data in response."""

    embeddingType: _EmbeddingType
    embedding: list[float]
    truncatedCharLength: NotRequired[int]


class _SegmentMetadata(TypedDict):
    """Metadata for a segment in the embedding JSONL files."""

    segmentIndex: int
    segmentStartCharPosition: NotRequired[int]  # Text only
    segmentEndCharPosition: NotRequired[int]  # Text only
    truncatedCharLength: NotRequired[int]  # Only when text gets truncated
    segmentStartSeconds: NotRequired[float]  # Audio/video only
    segmentEndSeconds: NotRequired[float]  # Audio/video only


class _SegmentedEmbeddingData(TypedDict):
    """Segmented embedding entry from embedding-modality.jsonl files."""

    embedding: list[float]
    segmentMetadata: _SegmentMetadata
    status: Literal["SUCCESS", "FAILURE"]
    failureReason: NotRequired[_FailureReasons]
    message: NotRequired[str]


class _SegmentedEmbeddingResult(TypedDict):
    """Result entry in segmented-embedding-result.json."""

    status: Literal["SUCCESS", "FAILED", "PARTIAL_SUCCESS"]
    embeddingType: str
    outputFileUri: str
    message: NotRequired[str]
    failureReason: NotRequired[_FailureReasons]


class _SegmentedEmbeddingResultResponse(TypedDict):
    """Structure of segmented-embedding-result.json."""

    sourceFileUri: str
    embeddingDimension: int
    embeddingResults: list[_SegmentedEmbeddingResult]


class _Response(TypedDict):
    """Amazon Nova embedding response parameters."""

    embeddings: list[_EmbeddingData] | list[_SegmentedEmbeddingData]


class EmbeddingModel(EmbeddingModelBase[_Request, _Response]):
    """Amazon Nova multimodal embedding model."""

    __slots__ = ()

    MATCHER = "amazon.nova-2-multimodal-embeddings"

    async def embed_text(
        self,
        inputs: list[InputFileUrl | str],
        dimensions: int | None,
        extra_params: JsonMapping,
    ) -> EmbeddingResponse:
        """Get embeddings for text.

        Args:
            inputs: Texts to embed.
            dimensions: Number of dimensions.
            extra_params: Extra model parameters.

        Returns:
            Embedding response.
        """
        force_s3_data = bool(extra_params.pop("force_s3_data", False))
        base_params = _EmbeddingParams(
            embeddingPurpose=extra_params.pop("embeddingPurpose", "GENERIC_INDEX")  # type:ignore[typeddict-item]
        )
        if dimensions is not None:
            base_params["embeddingDimension"] = dimensions  # type:ignore[typeddict-item]

        embeddings: list[list[float]] = []
        input_tokens = 0
        output_tokens = 0
        for result in await gather(
            *(
                self._embed(
                    value=value,
                    base_params=base_params,
                    extra_params=extra_params,
                    force_s3_data=force_s3_data,
                )
                for value in inputs
            )
        ):
            embeddings.extend(
                item["embedding"] for item in result.response["embeddings"]
            )
            input_tokens += result.input_tokens or 0
            output_tokens += result.output_tokens or 0

        return EmbeddingResponse(
            embeddings=embeddings,
            prompt_tokens=input_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    async def _embed(
        self,
        value: InputFile | str,
        base_params: _EmbeddingParams,
        extra_params: JsonMapping,
        *,
        force_s3_data: bool = False,
    ) -> InvokeResult[_Response]:
        """Handle media input with automatic S3 upload for large files.

        Args:
            value: Plain text, base64, data URI, or S3 URI.
            base_params: Base embedding parameters.
            extra_params: Optional extra parameters for the input constructor.
            force_s3_data: Force S3 upload regardless of size.

        Returns:
            InvokeResult wrapping the model response and billing attribution.
        """
        if isinstance(value, str):
            media_type: _MediaTypes = "text"
            file_format = "plain"
            size = len(value)
        else:
            size = await value.get_size()
            media_type, file_format = await value.get_content_type_tuple()  # type: ignore[assignment]

        if (
            force_s3_data
            or (isinstance(value, InputFile) and value.is_s3)
            or size
            > (_TEXT_SIZE_LIMIT if media_type == "text" else BEDROCK_BODY_SIZE_LIMIT)
        ):
            # Large file or already S3 file requires to be passed using S3
            region = await self.select_region(s3_required=True)
            val: str | S3Object = await (
                put_s3_object(
                    value.encode(),
                    content_type="text/plain",
                    region=region,
                    temporary=True,
                )
                if isinstance(value, str)
                else value.to_s3(region=region)
            )
        elif isinstance(value, InputFile):
            # Require raw base64 content
            val = await value.to_base64()
            region = None
        else:
            # str input value
            val = value
            region = None

        if isinstance(val, S3Object) and size > _SYNC_LIMIT_SIZES[media_type]:
            return await self._embed_segmented(
                value=val,
                media_type=media_type,
                file_format=file_format,
                base_params=base_params,
                extra_params=extra_params,
            )
        return await self._embed_single(
            value=val,
            media_type=media_type,
            file_format=file_format,
            base_params=base_params,
            extra_params=extra_params,
            region=region,
        )

    async def _embed_single(
        self,
        value: str | S3Object,
        media_type: _MediaTypes,
        file_format: str,
        base_params: _EmbeddingParams,
        extra_params: JsonMapping,
        region: RegionName | None,
    ) -> InvokeResult[_Response]:
        """Build a SINGLE_EMBEDDING request for one media value and invoke it.

        Args:
            value: The media value or S3 URI.
            media_type: The type of media being processed.
            file_format: The format or MIME type of the media (e.g., MP4, PNG).
            base_params: The base embedding parameters used across all types of media.
            extra_params: Additional parameters specific to the media type.
                These parameters are merged into the base settings.
            region: When set, locks the Bedrock invocation to this region (S3
                inputs were already placed there); ``None`` applies full
                multi-region retry.

        Returns:
            InvokeResult wrapping the model response and token usage for this
            single embedding call.
        """
        source = (
            _MediaSource(s3Location=_S3Location(uri=value.uri))
            if isinstance(value, S3Object)
            else _MediaSource(bytes=value)
        )
        match media_type:
            case "image":
                params = _SingleEmbeddingParams(
                    image=_ImageInput(source=source, format=file_format),  # type: ignore[typeddict-item]
                    **base_params,
                )
            case "audio":
                params = _SingleEmbeddingParams(
                    audio=_AudioInput(source=source, format=file_format),  # type: ignore[typeddict-item]
                    **base_params,
                )
            case "video":
                params = _SingleEmbeddingParams(
                    video=_VideoInput(
                        source=source,
                        format=MIME_TYPES_TO_VIDEO_TYPE.get(file_format, file_format),  # type: ignore[arg-type]
                        embeddingMode=_DEFAULT_VIDEO_EMBEDDING_MODE,
                    ),
                    **base_params,
                )
            case _:
                # Default to text content
                params = _SingleEmbeddingParams(
                    text=(
                        _TextInput(
                            source=_TextSource(s3Location=_S3Location(uri=value.uri)),
                            truncationMode=_DEFAULT_TEXT_TRUNCATION_MODE,
                        )
                        if isinstance(value, S3Object)
                        else _TextInput(
                            value=value, truncationMode=_DEFAULT_TEXT_TRUNCATION_MODE
                        )
                    ),
                    **base_params,
                )
        self._add_extra_params(extra_params, media_type, params)

        result = await self.invoke(
            _Request(taskType="SINGLE_EMBEDDING", singleEmbeddingParams=params),
            region=region,
        )
        self._record_media_usage(
            media_type, params, region=result.region, routing=result.routing
        )
        return result

    async def _embed_segmented(
        self,
        value: S3Object,
        media_type: _MediaTypes,
        file_format: str,
        base_params: _EmbeddingParams,
        extra_params: JsonMapping,
    ) -> InvokeResult[_Response]:
        """Handles asynchronous segmented media embeddings.

        Args:
            value: The S3 URI pointing to the media source.
            media_type: The type of media being processed.
            file_format: The format or MIME type of the media (e.g., MP4, PNG).
            base_params: The base embedding parameters used across all types of media.
            extra_params: Additional parameters specific to the media type.
                These parameters are merged into the base settings.

        Returns:
            InvokeResult wrapping the embeddings aggregated from the processed
            media segments (segmented calls report no token usage).

        Raises:
            ApiError: When a segment of the embedding result reports a failure,
                detailing its reason and message.
        """
        s3_source = _MediaSource(s3Location=_S3Location(uri=value.uri))
        match media_type:
            case "image":
                params = _SegmentedEmbeddingParams(
                    image=_ImageInput(
                        source=s3_source,
                        format=file_format,  # type: ignore[typeddict-item]
                    ),
                    **base_params,
                )
            case "audio":
                params = _SegmentedEmbeddingParams(
                    audio=_SegmentedAudioInput(
                        source=s3_source,
                        format=file_format,  # type: ignore[typeddict-item]
                        segmentationConfig=_MediaSegmentationConfig(durationSeconds=5),
                    ),
                    **base_params,
                )
            case "video":
                params = _SegmentedEmbeddingParams(
                    video=_SegmentedVideoInput(
                        source=s3_source,
                        format=MIME_TYPES_TO_VIDEO_TYPE.get(file_format, file_format),  # type: ignore[arg-type]
                        segmentationConfig=_MediaSegmentationConfig(durationSeconds=5),
                        embeddingMode=_DEFAULT_VIDEO_EMBEDDING_MODE,
                    ),
                    **base_params,
                )
            case _:
                # Default to text content
                params = _SegmentedEmbeddingParams(
                    text=_SegmentedTextInput(
                        source=_TextSource(s3Location=_S3Location(uri=value.uri)),
                        segmentationConfig=_TextSegmentationConfig(),
                        truncationMode=_DEFAULT_TEXT_TRUNCATION_MODE,
                    ),
                    **base_params,
                )
        self._add_extra_params(extra_params, media_type, params)

        invoke_result = await self.invoke_async(
            _Request(taskType="SEGMENTED_EMBEDDING", segmentedEmbeddingParams=params),
            output_file="segmented-embedding-result.json",
        )
        embedding_result: _SegmentedEmbeddingResultResponse = invoke_result.response  # type: ignore[assignment]

        results: list[tuple[str, str]] = []
        errors: list[str] = []
        for result in embedding_result["embeddingResults"]:
            if result["status"] == "SUCCESS":
                bucket, key = result["outputFileUri"].replace("s3://", "").split("/", 1)
                results.append((bucket, key))
                track_temporary_s3_objects(bucket, key)
            else:
                errors.append(f"{result['failureReason']}: {result['message']}")

        if errors:
            msg = f"Error in segmented embedding results: {'; '.join(errors)}."
            raise ApiError(msg)

        s3_client: S3Client = get_client("s3", BUCKET_TO_REGION.get(results[0][0]))
        entries = [
            entry
            for sublist in await gather(
                *(
                    self._fetch_and_parse_embedding_jsonl(s3_client, bucket, key)
                    for bucket, key in results
                )
            )
            for entry in sublist
        ]
        self._record_media_usage(
            media_type,
            params,
            entries,
            region=invoke_result.region,
            routing=invoke_result.routing,
        )
        return InvokeResult(
            response=_Response(embeddings=entries),
            region=invoke_result.region,
            routing=invoke_result.routing,
        )

    def _record_media_usage(
        self,
        media_type: _MediaTypes,
        params: _SingleEmbeddingParams | _SegmentedEmbeddingParams,
        entries: Sequence[_SegmentedEmbeddingData] = (),
        *,
        region: str = "",
        routing: Routing | None = None,
    ) -> None:
        """Record billed input-media usage for a completed embedding call.

        Args:
            media_type: The type of media that was processed.
            params: The request parameters that were sent (for image detail level).
            entries: Segmented-job JSONL entries carrying segment timings;
                empty for the synchronous path (no billable duration known).
            region: Region that served the call (per-call, race-free
                attribution -- see :func:`stdapi.usage.record_bedrock_usage`).
            routing: Serving profile of the call.
        """
        if media_type == "image":
            record_bedrock_usage(
                self._model_id,
                region=region,
                routing=routing,
                input_images=1,
                media_spec=(
                    "document"
                    if params["image"].get("detailLevel") == "DOCUMENT_IMAGE"
                    else ""
                ),
            )
        elif media_type in ("audio", "video"):
            end_seconds = [
                metadata["segmentEndSeconds"]
                for entry in entries
                if "segmentEndSeconds" in (metadata := entry["segmentMetadata"])
            ]
            if end_seconds:
                record_bedrock_usage(
                    self._model_id,
                    region=region,
                    routing=routing,
                    input_seconds=ceil(max(end_seconds)),
                    media_spec=media_type,
                )

    @staticmethod
    async def _fetch_and_parse_embedding_jsonl(
        s3_client: S3Client, bucket: str, key: str
    ) -> tuple[_SegmentedEmbeddingData, ...]:
        """Fetch and parse a single embedding JSONL file from S3.

        Args:
            s3_client: S3 client.
            bucket: S3 bucket name.
            key: S3 object key.

        Returns:
            Tuple of parsed embedding data from the JSONL file.
        """
        return tuple(
            from_json(line)
            for line in (
                await (await s3_client.get_object(Bucket=bucket, Key=key))[
                    "Body"
                ].read()
            ).splitlines()
            if line
        )

    @staticmethod
    def _add_extra_params(
        extra_params: JsonMapping,
        media_type: _MediaTypes,
        params: _SegmentedEmbeddingParams | _SingleEmbeddingParams,
    ) -> None:
        """Adds extra parameters to `params` for a specific media type if conditions are met.

        Args:
            extra_params: A dictionary containing additional parameters
                categorized by media types. Each key corresponds to a media type and its value
                is a dictionary of parameters for that media type.
            media_type: Specifies the media type for which the extra parameters should be added.
            params: The parameter dictionary that will be updated with additional parameters
                for the specified media type.
        """
        if extra_params and isinstance(
            media_extra := extra_params.get(media_type), dict
        ):
            params[media_type].update(
                {  # type: ignore[typeddict-item]
                    k: v
                    for k, v in media_extra.items()
                    if k not in _RESERVED_MEDIA_PARAMS
                }
            )
