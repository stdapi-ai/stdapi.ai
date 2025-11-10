"""Amazon Titan embedding models.

- amazon.titan-embed-image-v1
- amazon.titan-embed-text-v1
- amazon.titan-embed-text-v2:0
"""

from contextlib import suppress
from typing import Literal, NotRequired, TypedDict

from fastapi import BackgroundTasks
from pydantic import JsonValue

from stdapi.models.embedding import EmbeddingModelBase, EmbeddingResponse
from stdapi.utils import is_data_uri


class _EmbeddingConfig(TypedDict):
    """AmazonTitan embedding config."""

    outputEmbeddingLength: int


class _Request(TypedDict):
    """AmazonTitan request parameters."""

    inputText: NotRequired[str]

    # amazon.titan-embed-text-v2:0
    dimensions: NotRequired[int]
    normalize: NotRequired[bool]
    embeddingTypes: NotRequired[list[Literal["float", "binary"]]]

    # amazon.titan-embed-image-v1
    inputImage: NotRequired[str]  # base64
    embeddingConfig: NotRequired[_EmbeddingConfig]


class _EmbeddingTypes(TypedDict):
    """AmazonTitan embedding types."""

    binary: NotRequired[list[int]]
    float: NotRequired[list[float]]


class _Response(TypedDict):
    """AmazonTitan response parameters."""

    embedding: list[float]
    inputTextTokenCount: NotRequired[int]

    # amazon.titan-embed-text-v2:0
    embeddingsByType: NotRequired[_EmbeddingTypes]

    # amazon.titan-embed-image-v1
    message: NotRequired[str]


class EmbeddingModel(EmbeddingModelBase[_Request, _Response]):
    """Amazon Titan embedding model."""

    MATCHER = "amazon.titan-embed-"

    async def embed_text(
        self,
        inputs: list[str],
        dimensions: int | None,
        extra_params: dict[str, JsonValue],
        background_tasks: BackgroundTasks,  # noqa: ARG002
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
        request = _Request()
        request.update(extra_params)  # type:ignore[typeddict-item]
        if dimensions:
            request["dimensions"] = dimensions
        input_tokens = 0
        embeddings = []
        for response in await self.batch_invoke(
            _Request(inputImage=value.split(",", 1)[1], **request)
            if is_data_uri(value)
            else _Request(inputText=value, **request)
            for value in inputs
        ):
            embeddings.append(response["embedding"])
            with suppress(KeyError):
                input_tokens += int(response["inputTextTokenCount"])
        return EmbeddingResponse(embeddings=embeddings, prompt_tokens=input_tokens)
