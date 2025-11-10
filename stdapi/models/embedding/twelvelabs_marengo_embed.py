"""TwelveLabs Marengo embedding models.

- twelvelabs.marengo-embed-2-7-v1:0
"""

from asyncio import create_task, gather
from collections.abc import Awaitable
from typing import Literal, NotRequired, TypedDict

from fastapi import BackgroundTasks, HTTPException
from pydantic import JsonValue

from stdapi.models.embedding import EmbeddingModelBase, EmbeddingResponse
from stdapi.openai_exceptions import OpenaiError
from stdapi.tokenizer import estimate_token_count
from stdapi.utils import get_data_uri_type, guess_media_type

_EmbeddingOption = Literal["visual-text", "visual-image", "audio"]


class _MediaSourceS3Location(TypedDict):
    """S3 location for media sources."""

    uri: str
    bucketOwner: NotRequired[str]


class _MediaSource(TypedDict):
    """Media source description for image, video, and audio inputs."""

    base64String: NotRequired[str]
    s3Location: NotRequired[_MediaSourceS3Location]


class _Request(TypedDict):
    """TwelveLabs Marengo request parameters."""

    inputType: Literal["video", "text", "audio", "image"]
    inputText: NotRequired[str]
    startSec: NotRequired[float]
    lengthSec: NotRequired[float]
    useFixedLengthSec: NotRequired[float]
    textTruncate: NotRequired[Literal["end", "none"]]
    embeddingOption: NotRequired[list[_EmbeddingOption]]
    mediaSource: NotRequired[_MediaSource]
    minClipSec: NotRequired[int]


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
        token_task = create_task(estimate_token_count(*inputs))
        embeddings: list[list[float]] = []
        tasks = []
        for value in inputs:
            data_type = get_data_uri_type(value)
            if data_type == "text/plain":
                if value.startswith("s3://"):
                    tasks.append(
                        self._handle_s3_media(value, extra_params, background_tasks)
                    )
                else:
                    tasks.append(self._handle_text(value, extra_params))
            elif data_type.startswith("image"):
                tasks.append(self._handle_image(value, extra_params))
            else:
                tasks.append(
                    self.handle_async_media(
                        value, data_type, extra_params, background_tasks
                    )
                )

        for response in await gather(*tasks):
            embeddings.extend(vector["embedding"] for vector in response["data"])
        estimated_tokens = await token_task or 0
        return EmbeddingResponse(
            embeddings=embeddings,
            prompt_tokens=estimated_tokens,
            total_tokens=estimated_tokens,
        )

    def _handle_text(
        self, value: str, extra_params: dict[str, JsonValue]
    ) -> Awaitable[_Response]:
        """Handles an text input.

        Args:
            value: Text input.
            extra_params: Extra model parameters.

        Returns:
            An awaitable response object corresponding
            to the processed text request.
        """
        request = _Request(inputType="text", inputText=value)
        request.update(extra_params)  # type:ignore[typeddict-item]
        return self.invoke(request)

    def _handle_image(
        self, value: str, extra_params: dict[str, JsonValue]
    ) -> Awaitable[_Response]:
        """Handles an image input.

        Args:
            value: A base64-encoded string representing the image input.
            extra_params: Extra model parameters.

        Returns:
            An awaitable response object corresponding
            to the processed image request.
        """
        request = _Request(
            inputType="image",
            mediaSource=_MediaSource(base64String=value.split(",", 1)[1]),
        )
        request.update(extra_params)  # type:ignore[typeddict-item]
        return self.invoke(request)

    def handle_async_media(
        self,
        value: str,
        data_type: str,
        extra_params: dict[str, JsonValue],
        background_tasks: BackgroundTasks,
    ) -> Awaitable[_Response]:
        """Handles media requiring asynchronous processing.

        Args:
            value: The media content as a string, typically in base64 encoded format.
            data_type: The type of the input media. It determines how the media data is processed.
            extra_params: Extra model parameters.
            background_tasks: The BackgroundTasks instance to manage asynchronous tasks.

        Returns:
            An awaitable object that resolves to the response of the media processing request.
        """
        request = _Request(
            inputType=data_type.split("/", 1)[0],  # type: ignore[typeddict-item]
            mediaSource=_MediaSource(base64String=value.split(",", 1)[1]),
        )
        request.update(extra_params)  # type:ignore[typeddict-item]
        return self.invoke_async(
            request, background_tasks=background_tasks, inference_profile=False
        )

    def _handle_s3_media(
        self,
        value: str,
        extra_params: dict[str, JsonValue],
        background_tasks: BackgroundTasks,
    ) -> Awaitable[_Response]:
        """Handles the processing of S3 media asynchronously.

        Args:
            value: S3 URI of the media to process.
            extra_params: Extra model parameters.
            background_tasks: Background task manager to schedule
                and handle the asynchronous execution.

        Returns:
            An awaitable result of the asynchronous media
                processing operation.

        Raises:
            OpenaiError: If the media type is unable to be determined due to an invalid
                or unexpected input.
        """
        try:
            input_type = guess_media_type(value)
        except ValueError as error:
            raise OpenaiError(error.args[0]) from error

        request = _Request(
            inputType=input_type,
            mediaSource=_MediaSource(s3Location=_MediaSourceS3Location(uri=value)),
        )
        request.update(extra_params)  # type:ignore[typeddict-item]
        return self.invoke_async(
            request, background_tasks=background_tasks, inference_profile=False
        )
