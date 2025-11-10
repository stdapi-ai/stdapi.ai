"""Cohere embedding models.

- cohere.embed-english-v3
- cohere.embed-multilingual-v3
- cohere.embed-v4
"""

from asyncio import create_task
from typing import Literal, NotRequired, TypedDict

from fastapi import BackgroundTasks
from pydantic import JsonValue

from stdapi.models.embedding import EmbeddingModelBase, EmbeddingResponse
from stdapi.tokenizer import estimate_token_count
from stdapi.utils import is_data_uri

_EmbeddingType = Literal["float", "int8", "uint8", "binary", "ubinary"]


class _InputUrl(TypedDict):
    """Cohere input URLs."""

    url: str


class _InputContentImageUrl(TypedDict):
    """Cohere input content for image URLs."""

    type: Literal["image_url"]
    image_url: _InputUrl  # Base64 image URI


class _InputContentText(TypedDict):
    """Cohere input content for texts."""

    type: Literal["text"]
    text: str


class _InputContent(TypedDict):
    """Cohere input content."""

    content: list[_InputContentImageUrl | _InputContentText]


class _Request(TypedDict):
    """Cohere request parameters.

    Supported in Cohere V3 only:
        - input_type=image
        - images with more than 1 image
        - truncate=START or END (Replaced by LEFT, RIGHT in V4)
    """

    input_type: Literal[
        "search_document", "search_query", "classification", "clustering", "image"
    ]
    texts: NotRequired[list[str]]
    images: NotRequired[list[str]]  # Base64 image URI
    truncate: NotRequired[Literal["NONE", "START", "END", "LEFT", "RIGHT"]]
    embedding_types: NotRequired[_EmbeddingType]

    # New in Cohere V4
    inputs: NotRequired[list[_InputContent]]
    max_tokens: NotRequired[int]
    output_dimension: NotRequired[int]


class _ImageDescription(TypedDict):
    """Cohere image description."""

    format: str
    width: int
    height: int
    bit_depth: int


class _Response(TypedDict):
    """Cohere response parameters."""

    embeddings: list[list[float]] | dict[_EmbeddingType, list[list[float | int]]]
    id: str
    response_type: Literal["embeddings_floats"]
    texts: list[str]
    images: list[_ImageDescription]


class EmbeddingModel(EmbeddingModelBase[_Request, _Response]):
    """Cohere embedding model."""

    MATCHER = "cohere.embed-"

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
        request = _Request(input_type="search_document")
        request.update(extra_params)  # type:ignore[typeddict-item]
        if dimensions is not None:
            request["output_dimension"] = dimensions

        is_data: tuple[bool, ...] = tuple(
            is_data_uri(input_str) for input_str in inputs
        )
        if all(is_data):
            request["images"] = inputs
            if self._model_id.endswith("v3"):
                request["input_type"] = "image"
        elif any(is_data):
            request["inputs"] = [
                _InputContent(
                    content=[
                        _InputContentImageUrl(
                            type="image_url", image_url=_InputUrl(url=value)
                        )
                        if is_data[index]
                        else _InputContentText(type="text", text=value)
                    ]
                )
                for index, value in enumerate(inputs)
            ]
        else:
            request["texts"] = inputs

        token_task = create_task(estimate_token_count(*inputs))
        embeddings = (await self.invoke(request))["embeddings"]
        if isinstance(embeddings, dict):
            embeddings = embeddings["float"]
        estimated_tokens = await token_task or 0
        return EmbeddingResponse(
            embeddings=embeddings,
            prompt_tokens=estimated_tokens,
            total_tokens=estimated_tokens,
        )
