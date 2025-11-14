"""TwelveLabs Marengo embedding models.

- twelvelabs.marengo-embed-2-7-v1:0
- twelvelabs.marengo-embed-3-0-v1:0
"""

from asyncio import create_task, gather
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
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

_EmbeddingOption_V2 = Literal["visual-text", "visual-image", "audio"]
_EmbeddingOption = Literal["visual", "audio", "transcription"]

_MediaTypes = Literal["video", "text", "audio", "image", "text_image"]

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


class _DynamicSegmentation(TypedDict):
    """Dynamic segmentation parameters."""

    minDurationSec: NotRequired[int]  # Range: 1-5, Default: 4


class _FixedSegmentation(TypedDict):
    """Fixed segmentation parameters."""

    durationSec: NotRequired[int]  # Range: 1-10, Default: 6


class _Segmentation(TypedDict):
    """Segmentation parameters."""

    method: NotRequired[Literal["dynamic", "fixed"]]
    dynamic: NotRequired[_DynamicSegmentation]
    fixed: NotRequired[_FixedSegmentation]


class _Request(TypedDict):
    """Base request parameters."""


class _V2TextRequest(_Request):
    """Legacy V2 text request parameters."""

    inputType: Literal["text"]
    inputText: str
    textTruncate: NotRequired[Literal["end", "none"]]


class _V2ImageRequest(_Request):
    """Legacy V2 image request parameters."""

    inputType: Literal["image"]
    mediaSource: NotRequired[_MediaSource]


class _V2VideoRequest(_Request):
    """Legacy V2 video request parameters."""

    inputType: Literal["video"]
    mediaSource: NotRequired[_MediaSource]
    startSec: NotRequired[float]
    lengthSec: NotRequired[float]
    useFixedLengthSec: NotRequired[float]
    minClipSec: NotRequired[int]
    embeddingOption: NotRequired[list[_EmbeddingOption_V2]]


class _V2AudioRequest(_Request):
    """Legacy V2 audio request parameters."""

    inputType: Literal["audio"]
    mediaSource: NotRequired[_MediaSource]
    startSec: NotRequired[float]
    lengthSec: NotRequired[float]
    useFixedLengthSec: NotRequired[float]


class _TextPayload(TypedDict):
    """Text payload (nested under 'text' key)."""

    inputText: str


class _ImagePayload(TypedDict):
    """Image payload (nested under 'image' key)."""

    mediaSource: _MediaSource


class _TextImagePayload(TypedDict):
    """Text+image payload (nested under 'text_image' key)."""

    inputText: str
    mediaSource: _MediaSource


class _VideoPayload(TypedDict):
    """Video payload (nested under 'video' key)."""

    mediaSource: _MediaSource
    startSec: NotRequired[float]
    endSec: NotRequired[float]
    embeddingOption: NotRequired[list[_EmbeddingOption]]
    embeddingScope: NotRequired[list[Literal["clip", "asset"]]]
    segmentation: NotRequired[_Segmentation]
    inferenceId: NotRequired[str]


class _AudioPayload(TypedDict):
    """Audio payload (nested under 'audio' key)."""

    mediaSource: _MediaSource
    startSec: NotRequired[float]
    endSec: NotRequired[float]
    embeddingOption: NotRequired[list[_EmbeddingOption]]
    embeddingScope: NotRequired[list[Literal["clip", "asset"]]]
    segmentation: NotRequired[_Segmentation]
    inferenceId: NotRequired[str]


class _TextRequest(_Request):
    """Text request parameters."""

    inputType: Literal["text"]
    text: _TextPayload


class _ImageRequest(_Request):
    """Image request parameters."""

    inputType: Literal["image"]
    image: _ImagePayload


class _TextImageRequest(_Request):
    """Text+image request parameters."""

    inputType: Literal["text_image"]
    text_image: _TextImagePayload


class _VideoRequest(_Request):
    """Video request parameters."""

    inputType: Literal["video"]
    video: _VideoPayload


class _AudioRequest(_Request):
    """Audio request parameters."""

    inputType: Literal["audio"]
    audio: _AudioPayload


class _ResponseData(TypedDict):
    """TwelveLabs Marengo response data parameters."""

    embedding: list[float]
    embeddingOption: NotRequired[_EmbeddingOption_V2 | _EmbeddingOption]
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

        content_types_and_sizes = await gather(
            *(get_content_type_and_size(value, self.model) for value in inputs)
        )

        text_image = self._get_text_image_input(inputs, content_types_and_sizes)
        if text_image:
            embeddings.extend(
                vector["embedding"]
                for vector in (
                    await self._embed_text_image(
                        image_text=text_image[0],
                        value=text_image[1],
                        content_type=text_image[2],
                        size=text_image[3],
                        extra_params=extra_params,
                        background_tasks=background_tasks,
                        force_s3_data=force_s3_data,
                    )
                )["data"]
            )
        else:
            for response in await gather(
                *(
                    self._embed(
                        value=value,
                        content_type=content_type,
                        size=size,
                        extra_params=extra_params,
                        background_tasks=background_tasks,
                        force_s3_data=force_s3_data,
                    )
                    for value, (content_type, size) in zip(
                        inputs, content_types_and_sizes, strict=False
                    )
                )
            ):
                embeddings.extend(vector["embedding"] for vector in response["data"])

        estimated_tokens = await token_task or 0
        return EmbeddingResponse(
            embeddings=embeddings,
            prompt_tokens=estimated_tokens,
            total_tokens=estimated_tokens,
        )

    def _build_request(
        self,
        media_type: _MediaTypes,
        value: str,
        extra_params: dict[str, JsonValue],
        *,
        s3_uri: bool = False,
        image_text: str = "",
    ) -> _Request:
        """Build a request for V3+ models.

        Args:
            media_type: Type of media.
            value: Text value or media identifier (for text_image, this is the image).
            extra_params: Extra parameters to add.
            s3_uri: True if the media source is an S3 URI.
            image_text: Image text for text_image mode.

        Returns:
            Request object.
        """
        if media_type == "text_image":
            request: _Request = _TextImageRequest(
                inputType="text_image",
                text_image=_TextImagePayload(
                    inputText=image_text,
                    mediaSource=self._media_source(value, s3_uri=s3_uri),
                ),
            )
        elif media_type == "image":
            request = _ImageRequest(
                inputType="image",
                image=_ImagePayload(
                    mediaSource=self._media_source(value, s3_uri=s3_uri)
                ),
            )
        elif media_type == "video":
            request = _VideoRequest(
                inputType="video",
                video=_VideoPayload(
                    mediaSource=self._media_source(value, s3_uri=s3_uri)
                ),
            )
        elif media_type == "audio":
            request = _AudioRequest(
                inputType="audio",
                audio=_AudioPayload(
                    mediaSource=self._media_source(value, s3_uri=s3_uri)
                ),
            )
        else:
            # Default to text
            request = _TextRequest(inputType="text", text=_TextPayload(inputText=value))

        request[request["inputType"]].update(  # type: ignore[literal-required,typeddict-item]
            {k: v for k, v in extra_params.items() if k not in _RESERVED_MEDIA_PARAMS}
        )
        return request

    def _build_v2_request(
        self,
        media_type: _MediaTypes,
        value: str,
        extra_params: dict[str, JsonValue],
        *,
        s3_uri: bool = False,
    ) -> _Request:
        """Build a request for legacy V2 models.

        Args:
            media_type: Type of media.
            value: Text value or media identifier.
            extra_params: Extra parameters to add.
            s3_uri: True if the media source is an S3 URI.

        Returns:
            Request object.
        """
        if media_type == "image":
            request: _Request = _V2ImageRequest(
                inputType="image", mediaSource=self._media_source(value, s3_uri=s3_uri)
            )
        elif media_type == "video":
            request = _V2VideoRequest(
                inputType="video", mediaSource=self._media_source(value, s3_uri=s3_uri)
            )
        elif media_type == "audio":
            request = _V2AudioRequest(
                inputType="audio", mediaSource=self._media_source(value, s3_uri=s3_uri)
            )
        else:
            # Default to text
            request = _V2TextRequest(inputType="text", inputText=value)

        if extra_params:
            request.update(
                {  # type: ignore[typeddict-item]
                    k: v
                    for k, v in extra_params.items()
                    if k not in _RESERVED_MEDIA_PARAMS
                }
            )
        return request

    async def _embed(
        self,
        value: str,
        content_type: str,
        size: int,
        extra_params: dict[str, JsonValue],
        background_tasks: BackgroundTasks,
        *,
        force_s3_data: bool = False,
    ) -> _Response:
        """Handle media input with automatic S3 upload for large files.

        Args:
            value: Base64-encoded media data (without data URI prefix).
            content_type: Pre-computed content type.
            size: Pre-computed size.
            extra_params: Optional extra parameters for the input constructor.
            background_tasks: FastAPI background tasks.
            force_s3_data: Force S3 upload regardless of size.

        Returns:
            Response from the model.
        """
        media_type: _MediaTypes = content_type.split("/", 1)[0]  # type: ignore[assignment]
        async with self._process_media_value(
            value,
            content_type=content_type,
            size=size,
            background_tasks=background_tasks,
            force_s3_data=force_s3_data,
            is_text=(media_type == "text"),
        ) as (processed_value, s3_uri):
            request = (
                self._build_v2_request if self._is_v2() else self._build_request
            )(media_type, processed_value, extra_params, s3_uri=s3_uri)

            if s3_uri or media_type in _ASYNC_MEDIA_TYPES:
                return await self.invoke_async(
                    request, background_tasks=background_tasks, inference_profile=False
                )
            return await self.invoke(request)

    async def _embed_text_image(
        self,
        value: str,
        content_type: str,
        size: int,
        image_text: str,
        extra_params: dict[str, JsonValue],
        background_tasks: BackgroundTasks,
        *,
        force_s3_data: bool = False,
    ) -> _Response:
        """Embed text and image together as text_image type (v3 only).

        Args:
            value: Image data (base64 or S3 URI).
            content_type: Pre-computed content type of image.
            size: Pre-computed size of image.
            image_text: Text caption of the image.
            extra_params: Optional extra parameters.
            background_tasks: FastAPI background tasks.
            force_s3_data: Force S3 upload regardless of size.

        Returns:
            Response from the model.
        """
        async with self._process_media_value(
            value,
            content_type=content_type,
            size=size,
            force_s3_data=force_s3_data,
            background_tasks=background_tasks,
        ) as (processed_value, s3_uri):
            return await self.invoke(
                self._build_request(
                    media_type="text_image",
                    value=processed_value,
                    extra_params=extra_params,
                    s3_uri=s3_uri,
                    image_text=image_text,
                )
            )

    def _is_v2(self) -> bool:
        """Check if the model is version 3+."""
        return "-2-" in self.model.id

    def _get_text_image_input(
        self, inputs: list[str], content_type_results: list[tuple[str, int]]
    ) -> tuple[str, str, str, int] | None:
        """Detect if inputs are exactly one text and one image.

        Args:
            inputs: List of input values.
            content_type_results: Pre-computed content types and sizes.

        Returns:
            Tuple of (text_value, image_value, image_content_type, image_size) if detected, None otherwise.
        """
        if len(inputs) == 2 and not self._is_v2():
            media_types = tuple(ct.split("/", 1)[0] for ct, _ in content_type_results)
            if set(media_types) == {"image", "text"}:
                image_index = media_types.index("image")
                content_type, size = content_type_results[image_index]
                return (
                    inputs[media_types.index("text")],
                    inputs[image_index],
                    content_type,
                    size,
                )
        return None

    @staticmethod
    def _media_source(value: str, *, s3_uri: bool) -> _MediaSource:
        """Creates and returns an appropriate _MediaSource object.

        Args:
            value: Input value to be used as either an S3 URI or a base64 string.
            s3_uri: Determines whether the `value` parameter should be interpreted
                as an S3 URI (True) or a base64 string (False).

        Returns:
            An instance of `MediaSource` configured with either an S3
                location or a base64 string, depending on the value of `s3_uri`.
        """
        return (
            _MediaSource(
                s3Location=_MediaSourceS3Location(
                    uri=value, bucketOwner=AWS_ACCOUNT_INFO["account_id"]
                )
            )
            if s3_uri
            else _MediaSource(base64String=value)
        )

    @asynccontextmanager
    async def _process_media_value(
        self,
        value: str,
        content_type: str,
        size: int,
        background_tasks: BackgroundTasks,
        *,
        force_s3_data: bool = False,
        is_text: bool = False,
    ) -> AsyncGenerator[tuple[str, bool]]:
        """Process media value and handle S3 upload if needed.

        Args:
            value: Media value (base64, data URI, or S3 URI).
            content_type: Pre-computed content type (optional).
            size: Pre-computed size (optional).
            background_tasks: FastAPI Background tasks.
            force_s3_data: Force S3 upload regardless of size.
            is_text: Whether this is plain text (no processing needed).

        Yields:
            Tuple of (processed_value, is_s3_uri).
        """
        s3_tmp_objects: list[tuple[str, str]] = []
        try:
            s3_uri = value.startswith("s3://")

            if not is_text and not s3_uri:
                if force_s3_data or size > BEDROCK_BODY_SIZE_LIMIT:
                    # Large files require S3
                    s3_uri = True
                    s3_bucket, s3_key = await put_to_s3(
                        value, content_type=content_type, model=self.model
                    )
                    s3_tmp_objects.append((s3_bucket, s3_key))
                    value = f"s3://{s3_bucket}/{s3_key}"

                elif value.startswith("data:"):
                    # Require raw base64 content
                    value = value.split(",", 1)[1]

            yield value, s3_uri

        finally:
            if s3_tmp_objects:
                background_tasks.add_task(
                    aws_s3_cleanup,
                    get_client("s3", region_name=self.model.region),
                    s3_tmp_objects,
                    REQUEST_ID.get(),
                )
