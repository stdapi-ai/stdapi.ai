"""TwelveLabs Marengo embedding models.

- twelvelabs.marengo-embed-2-7-v1:0
"""

from asyncio import create_task, gather
from typing import Literal, NotRequired, TypedDict

from fastapi import BackgroundTasks, HTTPException
from pydantic import JsonValue

from stdapi.aws import AWS_ACCOUNT_INFO, get_client
from stdapi.aws_bedrock import BEDROCK_BODY_SIZE_LIMIT
from stdapi.aws_s3 import aws_s3_cleanup
from stdapi.models import get_content_type_and_size, put_to_s3
from stdapi.models.embedding import EmbeddingModelBase, EmbeddingResponse
from stdapi.monitoring import REQUEST_ID
from stdapi.tokenizer import estimate_token_count

_EmbeddingOption = Literal["visual-text", "visual-image", "audio"]
_MediaTypes = Literal["video", "text", "audio", "image"]

#: Fields that user can't overwrite
_RESERVED_MEDIA_PARAMS = frozenset({"inputType", "inputText", "mediaSource"})

#: Media types that can only be processed async
_ASYNC_MEDIA_TYPES = frozenset({"video", "audio"})


class _MediaSourceS3Location(TypedDict):
    """S3 location for media sources."""

    uri: str
    bucketOwner: str


class _MediaSource(TypedDict):
    """Media source description for image, video, and audio inputs."""

    base64String: NotRequired[str]
    s3Location: NotRequired[_MediaSourceS3Location]


class _Request(TypedDict):
    """Text request parameters."""


class _TextRequest(_Request):
    """Text request parameters."""

    inputType: Literal["text"]
    inputText: str
    textTruncate: NotRequired[Literal["end", "none"]]


class _ImageRequest(_Request):
    """Image request parameters."""

    inputType: Literal["image"]
    mediaSource: NotRequired[_MediaSource]


class _VideoRequest(_Request):
    """Video request parameters."""

    inputType: Literal["video"]
    mediaSource: NotRequired[_MediaSource]
    startSec: NotRequired[float]
    lengthSec: NotRequired[float]
    useFixedLengthSec: NotRequired[float]
    minClipSec: NotRequired[int]
    embeddingOption: NotRequired[list[_EmbeddingOption]]


class _AudioRequest(_Request):
    """Audio request parameters."""

    inputType: Literal["audio"]
    mediaSource: NotRequired[_MediaSource]
    startSec: NotRequired[float]
    lengthSec: NotRequired[float]
    useFixedLengthSec: NotRequired[float]


class _ResponseData(TypedDict):
    """TwelveLabs Marengo response data parameters."""

    embedding: list[float]
    embeddingOption: NotRequired[_EmbeddingOption]
    startSec: NotRequired[float]
    endSec: NotRequired[float]


class _Response(TypedDict):
    """TwelveLabs Marengo response parameters."""

    data: list[_ResponseData]


class EmbeddingModel(EmbeddingModelBase[_Request, _Response]):
    """TwelveLabs Marengo embedding model."""

    MATCHER = "twelvelabs.marengo-embed-"

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
        if dimensions is not None:
            raise HTTPException(
                status_code=400,
                detail="'dimensions' option is not supported by TwelveLabs Marengo embedding models.",
            )

        force_s3_data = bool(extra_params.pop("force_s3_data", False))
        token_task = create_task(estimate_token_count(*inputs))
        embeddings: list[list[float]] = []

        for response in await gather(
            *(
                self._embed(
                    value=value,
                    extra_params=extra_params,
                    background_tasks=background_tasks,
                    force_s3_data=force_s3_data,
                )
                for value in inputs
            )
        ):
            embeddings.extend(vector["embedding"] for vector in response["data"])

        estimated_tokens = await token_task or 0
        return EmbeddingResponse(
            embeddings=embeddings,
            prompt_tokens=estimated_tokens,
            total_tokens=estimated_tokens,
        )

    async def _embed(
        self,
        value: str,
        extra_params: dict[str, JsonValue],
        background_tasks: BackgroundTasks,
        *,
        force_s3_data: bool = False,
    ) -> _Response:
        """Handle media input with automatic S3 upload for large files.

        Args:
            value: Base64-encoded media data (without data URI prefix).
            extra_params: Optional extra parameters for the input constructor.
            background_tasks: FastAPI background tasks.
            force_s3_data: Force S3 upload regardless of size.

        Returns:
            Response from the model.
        """
        s3_tmp_objects = []
        try:
            content_type, size = await get_content_type_and_size(value, self.model)
            media_type: _MediaTypes = content_type.split("/", 1)[0]  # type: ignore[assignment]
            s3_uri = value.startswith("s3://")

            if not s3_uri and media_type != "text":
                if force_s3_data or size > BEDROCK_BODY_SIZE_LIMIT or media_type:
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

            if media_type == "image":
                request: _Request = _ImageRequest(
                    inputType="image",
                    mediaSource=_MediaSource(
                        s3Location=_MediaSourceS3Location(
                            uri=value, bucketOwner=AWS_ACCOUNT_INFO["account_id"]
                        )
                    )
                    if s3_uri
                    else _MediaSource(base64String=value),
                )
            elif media_type == "video":
                request = _VideoRequest(
                    inputType="video",
                    mediaSource=_MediaSource(
                        s3Location=_MediaSourceS3Location(
                            uri=value, bucketOwner=AWS_ACCOUNT_INFO["account_id"]
                        )
                    )
                    if s3_uri
                    else _MediaSource(base64String=value),
                )
            elif media_type == "audio":
                request = _AudioRequest(
                    inputType="audio",
                    mediaSource=_MediaSource(
                        s3Location=_MediaSourceS3Location(
                            uri=value, bucketOwner=AWS_ACCOUNT_INFO["account_id"]
                        )
                    )
                    if s3_uri
                    else _MediaSource(base64String=value),
                )
            else:
                # Default to text content
                request = _TextRequest(inputType="text", inputText=value)
            self._add_extra_params(extra_params, request)

            if s3_uri or media_type in _ASYNC_MEDIA_TYPES:
                return await self.invoke_async(
                    request, background_tasks=background_tasks, inference_profile=False
                )
            return await self.invoke(request)

        finally:
            if s3_tmp_objects:
                background_tasks.add_task(
                    aws_s3_cleanup,
                    get_client("s3", region_name=self.model.region),
                    s3_tmp_objects,
                    REQUEST_ID.get(),
                )

    @staticmethod
    def _add_extra_params(extra_params: dict[str, JsonValue], params: _Request) -> None:
        """Adds extra parameters.

        Args:
            extra_params: A dictionary containing additional parameters.
            params: The parameter dictionary that will be updated.
        """
        if extra_params:
            params.update(
                {  # type: ignore[typeddict-item]
                    k: v
                    for k, v in extra_params.items()
                    if k not in _RESERVED_MEDIA_PARAMS
                }
            )
