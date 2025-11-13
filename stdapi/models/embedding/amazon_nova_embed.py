"""Amazon Nova multimodal embedding model.

- amazon.nova-2-multimodal-embeddings-v1:0
"""

from asyncio import create_task, gather
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

from fastapi import BackgroundTasks
from pydantic import JsonValue
from pydantic_core import from_json

from stdapi.aws import get_client
from stdapi.aws_bedrock import BEDROCK_BODY_SIZE_LIMIT, MIME_TYPES_TO_VIDEO_TYPE
from stdapi.aws_s3 import aws_s3_cleanup
from stdapi.models import get_content_type_and_size, put_to_s3
from stdapi.models.embedding import EmbeddingModelBase, EmbeddingResponse
from stdapi.monitoring import REQUEST_ID
from stdapi.openai_exceptions import OpenaiError
from stdapi.tokenizer import estimate_token_count

if TYPE_CHECKING:
    from types_aiobotocore_s3.client import S3Client

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

# Default constants
_DEFAULT_VIDEO_EMBEDDING_MODE: _VideoEmbeddingMode = "AUDIO_VIDEO_COMBINED"
_DEFAULT_TEXT_TRUNCATION_MODE: _TextTruncationMode = "END"
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

    MATCHER = "amazon.nova-2-multimodal-embeddings"

    async def embed_text(
        self,
        inputs: list[str],
        dimensions: int | None,
        extra_params: dict[str, JsonValue],
        background_tasks: BackgroundTasks,
    ) -> EmbeddingResponse:
        """Get embeddings for text.

        Args:
            inputs: Texts to embed.
            dimensions: Number of dimensions.
            extra_params: Extra model parameters.
            background_tasks: FastAPI background tasks.

        Returns:
            Embedding response.
        """
        force_s3_data = bool(extra_params.pop("force_s3_data", False))
        base_params = _EmbeddingParams(
            embeddingPurpose=extra_params.pop("embeddingPurpose", "GENERIC_INDEX")  # type:ignore[typeddict-item]
        )
        if dimensions is not None:
            base_params["embeddingDimension"] = dimensions  # type:ignore[typeddict-item]

        token_task = create_task(estimate_token_count(*inputs))
        embeddings: list[list[float]] = []

        for response in await gather(
            *(
                self._embed(
                    value=value,
                    base_params=base_params,
                    extra_params=extra_params,
                    background_tasks=background_tasks,
                    force_s3_data=force_s3_data,
                )
                for value in inputs
            )
        ):
            embeddings.extend(item["embedding"] for item in response["embeddings"])

        estimated_tokens = await token_task or 0
        return EmbeddingResponse(
            embeddings=embeddings,
            prompt_tokens=estimated_tokens,
            total_tokens=estimated_tokens,
        )

    async def _embed(
        self,
        value: str,
        base_params: _EmbeddingParams,
        extra_params: dict[str, JsonValue],
        background_tasks: BackgroundTasks,
        *,
        force_s3_data: bool = False,
    ) -> _Response:
        """Handle media input with automatic S3 upload for large files.

        Args:
            value: Base64-encoded media data (without data URI prefix).
            base_params: Base embedding parameters.
            extra_params: Optional extra parameters for the input constructor.
            background_tasks: FastAPI background tasks.
            force_s3_data: Force S3 upload regardless of size.

        Returns:
            Response from the model.
        """
        s3_tmp_objects = []
        try:
            content_type, size = await get_content_type_and_size(value, self.model)
            content_type_split = content_type.split("/", 1)
            media_type: _MediaTypes = content_type_split[0]  # type: ignore[assignment]
            file_format = content_type_split[1]
            s3_uri = value.startswith("s3://")

            if not s3_uri:
                if force_s3_data or size > (
                    _TEXT_SIZE_LIMIT
                    if media_type == "text"
                    else BEDROCK_BODY_SIZE_LIMIT
                ):
                    # Large files require to be passed using S3
                    s3_uri = True
                    s3_bucket, s3_key = await put_to_s3(
                        value, content_type=content_type, model=self.model
                    )
                    s3_tmp_objects.append((s3_bucket, s3_key))
                    value = f"s3://{s3_bucket}/{s3_key}"

                elif value.startswith("data:"):
                    # Require raw base64 content
                    value = value.split(",", 1)[1]

            if s3_uri and size > _SYNC_LIMIT_SIZES[media_type]:
                return await self._embed_segmented(
                    value=value,
                    media_type=media_type,
                    file_format=file_format,
                    base_params=base_params,
                    extra_params=extra_params,
                    background_tasks=background_tasks,
                )
            return await self._embed_single(
                value=value,
                media_type=media_type,
                file_format=file_format,
                base_params=base_params,
                extra_params=extra_params,
            )

        finally:
            if s3_tmp_objects:
                background_tasks.add_task(
                    aws_s3_cleanup,
                    get_client("s3", region_name=self.model.region),
                    s3_tmp_objects,
                    REQUEST_ID.get(),
                )

    async def _embed_single(
        self,
        value: str,
        media_type: _MediaTypes,
        file_format: str,
        base_params: _EmbeddingParams,
        extra_params: dict[str, JsonValue],
    ) -> _Response:
        """Handles synchronous single media embeddings.

        Args:
            value: The media value or S3 URI.
            media_type: The type of media being processed.
            file_format: The format or MIME type of the media (e.g., MP4, PNG).
            base_params: The base embedding parameters used across all types of media.
            extra_params: Additional parameters specific to the media type.
                These parameters are merged into the base settings.

        Returns:
            A response object containing the embeddings aggregated from the
            processed media segments.

        Raises:
            OpenaiError: If any part of the segmented embedding result indicates a
            failure, an exception is raised detailing the error reason and message.
        """
        s3_source = value.startswith("s3://")
        if media_type == "image":
            params = _SingleEmbeddingParams(
                image=_ImageInput(
                    source=(
                        _MediaSource(s3Location=_S3Location(uri=value))
                        if s3_source
                        else _MediaSource(bytes=value)
                    ),
                    format=file_format,  # type: ignore[typeddict-item]
                ),
                **base_params,
            )
        elif media_type == "audio":
            params = _SingleEmbeddingParams(
                audio=_AudioInput(
                    source=(
                        _MediaSource(s3Location=_S3Location(uri=value))
                        if s3_source
                        else _MediaSource(bytes=value)
                    ),
                    format=file_format,  # type: ignore[typeddict-item]
                ),
                **base_params,
            )
        elif media_type == "video":
            params = _SingleEmbeddingParams(
                video=_VideoInput(
                    source=(
                        _MediaSource(s3Location=_S3Location(uri=value))
                        if s3_source
                        else _MediaSource(bytes=value)
                    ),
                    format=MIME_TYPES_TO_VIDEO_TYPE.get(file_format, file_format),  # type: ignore[arg-type]
                    embeddingMode=_DEFAULT_VIDEO_EMBEDDING_MODE,
                ),
                **base_params,
            )
        else:
            # Default to text content
            params = _SingleEmbeddingParams(
                text=(
                    _TextInput(
                        source=_TextSource(s3Location=_S3Location(uri=value)),
                        truncationMode=_DEFAULT_TEXT_TRUNCATION_MODE,
                    )
                    if s3_source
                    else _TextInput(
                        value=value, truncationMode=_DEFAULT_TEXT_TRUNCATION_MODE
                    )
                ),
                **base_params,
            )
        self._add_extra_params(extra_params, media_type, params)

        return await self.invoke(
            _Request(taskType="SINGLE_EMBEDDING", singleEmbeddingParams=params)
        )

    async def _embed_segmented(
        self,
        value: str,
        media_type: _MediaTypes,
        file_format: str,
        base_params: _EmbeddingParams,
        extra_params: dict[str, JsonValue],
        background_tasks: BackgroundTasks,
    ) -> _Response:
        """Handles asynchronous segmented media embeddings.

        Args:
            value: The S3 URI pointing to the media source.
            media_type: The type of media being processed.
            file_format: The format or MIME type of the media (e.g., MP4, PNG).
            base_params: The base embedding parameters used across all types of media.
            extra_params: Additional parameters specific to the media type.
                These parameters are merged into the base settings.
            background_tasks: A FastAPI background tasks object to
                manage the cleanup of temporary S3 files post-processing.

        Returns:
            A response object containing the embeddings aggregated from the
            processed media segments.

        Raises:
            OpenaiError: If any part of the segmented embedding result indicates a
            failure, an exception is raised detailing the error reason and message.
        """
        if media_type == "image":
            params = _SegmentedEmbeddingParams(
                image=_ImageInput(
                    source=_MediaSource(s3Location=_S3Location(uri=value)),
                    format=file_format,  # type: ignore[typeddict-item]
                ),
                **base_params,
            )
        elif media_type == "audio":
            params = _SegmentedEmbeddingParams(
                audio=_SegmentedAudioInput(
                    source=_MediaSource(s3Location=_S3Location(uri=value)),
                    format=file_format,  # type: ignore[typeddict-item]
                    segmentationConfig=_MediaSegmentationConfig(durationSeconds=5),
                ),
                **base_params,
            )
        elif media_type == "video":
            params = _SegmentedEmbeddingParams(
                video=_SegmentedVideoInput(
                    source=_MediaSource(s3Location=_S3Location(uri=value)),
                    format=MIME_TYPES_TO_VIDEO_TYPE.get(file_format, file_format),  # type: ignore[arg-type]
                    segmentationConfig=_MediaSegmentationConfig(durationSeconds=5),
                    embeddingMode=_DEFAULT_VIDEO_EMBEDDING_MODE,
                ),
                **base_params,
            )
        else:
            # Default to text content
            params = _SegmentedEmbeddingParams(
                text=_SegmentedTextInput(
                    source=_TextSource(s3Location=_S3Location(uri=value)),
                    segmentationConfig=_TextSegmentationConfig(),
                    truncationMode=_DEFAULT_TEXT_TRUNCATION_MODE,
                ),
                **base_params,
            )
        self._add_extra_params(extra_params, media_type, params)

        embedding_result: _SegmentedEmbeddingResultResponse = await self.invoke_async(  # type: ignore[assignment]
            _Request(taskType="SEGMENTED_EMBEDDING", segmentedEmbeddingParams=params),
            background_tasks=background_tasks,
            output_file="segmented-embedding-result.json",
        )

        s3_client: S3Client = get_client("s3", self.model.region)
        s3_tmp_objects: list[tuple[str, str]] = []
        errors: list[str] = []
        try:
            for result in embedding_result["embeddingResults"]:
                if result["status"] == "SUCCESS":
                    bucket, key = (
                        result["outputFileUri"].replace("s3://", "").split("/", 1)
                    )
                    s3_tmp_objects.append((bucket, key))
                else:
                    errors.append(f"{result['failureReason']}: {result['message']}")

            if errors:
                msg = f"Error in segmented embedding results: {'; '.join(errors)}."
                raise OpenaiError(msg)

            return _Response(
                embeddings=[
                    embedding
                    for sublist in await gather(
                        *(
                            self._fetch_and_parse_embedding_jsonl(
                                s3_client, bucket, key
                            )
                            for bucket, key in s3_tmp_objects
                        )
                    )
                    for embedding in sublist
                ]
            )
        finally:
            background_tasks.add_task(
                aws_s3_cleanup, s3_client, s3_tmp_objects, REQUEST_ID.get()
            )

    @staticmethod
    async def _fetch_and_parse_embedding_jsonl(
        s3_client: "S3Client", bucket: str, key: str
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
            )
            .strip()
            .splitlines()
            if line
        )

    @staticmethod
    def _add_extra_params(
        extra_params: dict[str, JsonValue],
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
        if (
            extra_params
            and media_type in extra_params
            and isinstance(extra_params[media_type], dict)
        ):
            params[media_type].update(
                {  # type: ignore[typeddict-item]
                    k: v
                    for k, v in extra_params[media_type].items()  # type: ignore[union-attr]
                    if k not in _RESERVED_MEDIA_PARAMS
                }
            )
