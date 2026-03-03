"""TwelveLabs Marengo embedding models.

- twelvelabs.marengo-embed-2-7-v1:0
- twelvelabs.marengo-embed-3-0-v1:0
"""

from asyncio import create_task, gather
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

from stdapi.api_errors import ApiError
from stdapi.aws import AWS_ACCOUNT_INFO
from stdapi.aws_bedrock import BEDROCK_BODY_SIZE_LIMIT
from stdapi.aws_s3 import S3Object
from stdapi.input_file import InputFileUrl, prefetch_all_content_types
from stdapi.models.embedding import EmbeddingModelBase, EmbeddingResponse
from stdapi.tokenizer import estimate_token_count

if TYPE_CHECKING:
    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.types import JsonMapping


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
        if dimensions is not None:
            msg = "'dimensions' option is not supported by TwelveLabs Marengo embedding models."
            raise ApiError(msg)

        force_s3_data = bool(extra_params.pop("force_s3_data", False))
        token_task = create_task(estimate_token_count(*inputs))
        embeddings: list[list[float]] = []

        await prefetch_all_content_types()

        text_image = await self._get_text_image_input(inputs)
        if text_image:
            embeddings.extend(
                vector["embedding"]
                for vector in (
                    await self._embed_text_image(
                        image_text=text_image[0],
                        value=text_image[1],
                        extra_params=extra_params,
                        force_s3_data=force_s3_data,
                    )
                )["data"]
            )
        else:
            for response in await gather(
                *(
                    self._embed(
                        value=value,
                        extra_params=extra_params,
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

    def _build_request(
        self,
        media_type: _MediaTypes,
        value: str | S3Object,
        extra_params: JsonMapping,
        *,
        image_text: str = "",
    ) -> _Request:
        """Build a request for V3+ models.

        Args:
            media_type: Type of media.
            value: Text value or media identifier (for text_image, this is the image).
            extra_params: Extra parameters to add.
            image_text: Image text for text_image mode.

        Returns:
            Request object.
        """
        match media_type:
            case "text_image":
                request: _Request = _TextImageRequest(
                    inputType="text_image",
                    text_image=_TextImagePayload(
                        inputText=image_text, mediaSource=self._media_source(value)
                    ),
                )
            case "image":
                request = _ImageRequest(
                    inputType="image",
                    image=_ImagePayload(mediaSource=self._media_source(value)),
                )
            case "video":
                request = _VideoRequest(
                    inputType="video",
                    video=_VideoPayload(mediaSource=self._media_source(value)),
                )
            case "audio":
                request = _AudioRequest(
                    inputType="audio",
                    audio=_AudioPayload(mediaSource=self._media_source(value)),
                )
            case _ if isinstance(value, str):
                request = _TextRequest(
                    inputType="text", text=_TextPayload(inputText=value)
                )
            case _:
                msg = f"Unsupported media type: {media_type}"
                raise ApiError(msg)

        request[request["inputType"]].update(  # type: ignore[literal-required,typeddict-item]
            {k: v for k, v in extra_params.items() if k not in _RESERVED_MEDIA_PARAMS}
        )
        return request

    def _build_v2_request(
        self, media_type: _MediaTypes, value: str | S3Object, extra_params: JsonMapping
    ) -> _Request:
        """Build a request for legacy V2 models.

        Args:
            media_type: Type of media.
            value: Text value or media identifier.
            extra_params: Extra parameters to add.

        Returns:
            Request object.
        """
        match media_type:
            case "image":
                request: _Request = _V2ImageRequest(
                    inputType="image", mediaSource=self._media_source(value)
                )
            case "video":
                request = _V2VideoRequest(
                    inputType="video", mediaSource=self._media_source(value)
                )
            case "audio":
                request = _V2AudioRequest(
                    inputType="audio", mediaSource=self._media_source(value)
                )
            case _ if isinstance(value, str):
                request = _V2TextRequest(inputType="text", inputText=value)
            case _:
                msg = f"Unsupported media type: {media_type}"
                raise ApiError(msg)

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
        value: str | InputFileUrl,
        extra_params: JsonMapping,
        *,
        force_s3_data: bool = False,
    ) -> _Response:
        """Handle media input with automatic S3 upload for large files.

        Args:
            value: Base64-encoded media data (without data URI prefix).
            extra_params: Optional extra parameters for the input constructor.
            force_s3_data: Force S3 upload regardless of size.

        Returns:
            Response from the model.
        """
        media_type: _MediaTypes = (
            "text"
            if isinstance(value, str)
            else (await value.get_content_type_tuple())[0]  # type: ignore[assignment]
        )
        region = await self._select_fixed_region(value, force_s3_data=force_s3_data)
        data = await self._process_media_value(
            value, region, force_s3_data=force_s3_data
        )
        request = (self._build_v2_request if self._is_v2() else self._build_request)(
            media_type, data, extra_params
        )

        if isinstance(data, S3Object) or media_type in _ASYNC_MEDIA_TYPES:
            return await self.invoke_async(request, inference_profile=False)
        return await self.invoke(request, region=region)

    async def _embed_text_image(
        self,
        value: InputFileUrl,
        image_text: str,
        extra_params: JsonMapping,
        *,
        force_s3_data: bool = False,
    ) -> _Response:
        """Embed text and image together as text_image type (v3 only).

        Args:
            value: Image data (base64 or S3 URI).
            image_text: Text caption of the image.
            extra_params: Optional extra parameters.
            force_s3_data: Force S3 upload regardless of size.

        Returns:
            Response from the model.
        """
        region = await self._select_fixed_region(value, force_s3_data=force_s3_data)
        return await self.invoke(
            self._build_request(
                media_type="text_image",
                value=await self._process_media_value(
                    value, region, force_s3_data=force_s3_data
                ),
                extra_params=extra_params,
                image_text=image_text,
            ),
            region=region,
        )

    async def _select_fixed_region(
        self, value: InputFileUrl | str, *, force_s3_data: bool
    ) -> RegionName | None:
        """Selects the region based on model and input requirements.

        Args:
            value: The input value.
            force_s3_data: Force S3 upload regardless of size.

        Returns:
            The determined fixed region if S3 is required, otherwise `None`.
        """
        return (
            await self.select_region(s3_required=True)
            if isinstance(value, InputFileUrl)
            and (
                force_s3_data
                or value.is_s3
                or await value.get_size() > BEDROCK_BODY_SIZE_LIMIT
            )
            else None
        )

    def _is_v2(self) -> bool:
        """Check if the model is version 3+."""
        return "-2-" in self.model.id

    async def _get_text_image_input(
        self, inputs: list[InputFileUrl | str]
    ) -> tuple[str, InputFileUrl] | None:
        """Detect if inputs are exactly one text and one image.

        Args:
            inputs: List of input values.

        Returns:
            Tuple of (text_value, image_value, image_content_type, image_size) if detected, None otherwise.
        """
        if len(inputs) == 2 and not self._is_v2():
            media_types = [
                "text"
                if isinstance(data, str)
                else (await data.get_content_type_tuple())[0]
                for data in inputs
            ]
            if set(media_types) == {"image", "text"}:
                return (  # type: ignore[return-value]
                    inputs[text_index := media_types.index("text")],
                    inputs[1 - text_index],
                )
        return None

    @staticmethod
    def _media_source(value: str | S3Object) -> _MediaSource:
        """Creates and returns an appropriate _MediaSource object.

        Args:
            value: Input value to be used as either an S3 URI or a base64 string.

        Returns:
            An instance of `MediaSource` configured with either an S3
                location or a base64 string, depending on the value of `s3_uri`.
        """
        return (
            _MediaSource(
                s3Location=_MediaSourceS3Location(
                    uri=value.uri, bucketOwner=AWS_ACCOUNT_INFO["account_id"]
                )
            )
            if isinstance(value, S3Object)
            else _MediaSource(base64String=value)
        )

    @staticmethod
    async def _process_media_value(
        value: InputFileUrl | str,
        region: RegionName | None,
        *,
        force_s3_data: bool = False,
    ) -> str | S3Object:
        """Process media value and handle S3 upload if needed.

        Args:
            value: Media value (base64, data URI, or S3 URI).
            region: S3 region to use for uploads. ``None`` means no S3 is
                required and the value is returned as base64.
            force_s3_data: Force S3 upload regardless of size.

        Returns:
            Processed_value.
        """
        if not isinstance(value, InputFileUrl):
            return value
        return await (
            value.to_s3(region=region)
            if (
                (
                    force_s3_data
                    or value.is_s3
                    or await value.get_size() > BEDROCK_BODY_SIZE_LIMIT
                )
                and region
            )
            else value.to_base64()
        )
