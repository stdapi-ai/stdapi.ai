"""Cohere embedding models.

- cohere.embed-english-v3
- cohere.embed-multilingual-v3
- cohere.embed-v4
"""

from asyncio import gather
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

from stdapi.api_errors import ApiError
from stdapi.input_file import InputFile, InputFileUrl
from stdapi.models.embedding import (
    EmbeddingImageDescription,
    EmbeddingModelBase,
    EmbeddingResponse,
)

if TYPE_CHECKING:
    from stdapi.types import JsonMapping

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
    embedding_types: NotRequired[list[_EmbeddingType]]

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

    __slots__ = ()

    MATCHER = "cohere.embed-"

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
        request = _Request(input_type="search_document")
        request.update(extra_params)  # type:ignore[typeddict-item]
        if dimensions is not None:
            request["output_dimension"] = dimensions

        is_data: tuple[bool, ...] = tuple(isinstance(v, InputFile) for v in inputs)
        if all(is_data):
            image_inputs: list[InputFile] = inputs  # type: ignore[assignment]
            request["images"] = await gather(
                *(image.to_data_uri() for image in image_inputs)
            )
            if self._model_id.endswith("v3"):
                request["input_type"] = "image"
        elif any(is_data):
            if "v4" not in self._model_id:
                msg = (
                    "Mixing texts and images in a single request is only supported by "
                    "Cohere Embed v4. Use separate requests, or a single content type "
                    "(all texts or all images), with cohere.embed-english-v3 or "
                    "cohere.embed-multilingual-v3."
                )
                raise ApiError(msg)
            request["inputs"] = [
                _InputContent(content=[content])
                for content in (
                    await gather(*(self._to_input_content(value) for value in inputs))
                )
            ]
        else:
            texts: list[str] = inputs  # type: ignore[assignment]
            request["texts"] = texts

        result = await self.invoke(request)
        resp = result.response["embeddings"]
        if isinstance(resp, dict) and "float" not in resp:
            # Cohere routes always request `float` alongside other types, so a
            # missing key means an unsupported type reached the backend directly.
            msg = "Only `float` embeddings are supported on this backend."
            raise ApiError(msg)
        input_tokens = result.input_tokens or 0
        images = result.response.get("images")
        # A fresh dict: `resp` is keyed by the narrower `_EmbeddingType` literal,
        # but `embeddings_by_type` is invariantly typed as `dict[str, ...]`.
        embeddings_by_type: dict[str, list[list[float | int]]] | None = (
            dict(resp.items()) if isinstance(resp, dict) else None
        )
        return EmbeddingResponse(
            embeddings=resp["float"] if isinstance(resp, dict) else resp,
            embeddings_by_type=embeddings_by_type,
            prompt_tokens=input_tokens,
            total_tokens=input_tokens + (result.output_tokens or 0),
            images=(
                [EmbeddingImageDescription(**image) for image in images]
                if images
                else None
            ),
        )

    @staticmethod
    async def _to_input_content(
        value: InputFileUrl | str,
    ) -> _InputContentImageUrl | _InputContentText:
        """Converts the given value into an appropriate input content type asynchronously.

        Args:
            value: The value to be converted.

        Returns:
            The converted input content object corresponding to the type and value provided.
        """
        return (
            _InputContentImageUrl(
                type="image_url", image_url=_InputUrl(url=await value.to_data_uri())
            )
            if isinstance(value, InputFile)
            else _InputContentText(type="text", text=value)
        )
