"""Amazon Titan embedding models.

- amazon.titan-embed-image-v1
- amazon.titan-embed-text-v1
- amazon.titan-embed-text-v2:0
"""

from asyncio import gather
from typing import TYPE_CHECKING, Literal, NotRequired, TypedDict

from stdapi.input_file import InputFile, InputFileUrl
from stdapi.models.embedding import EmbeddingModelBase, EmbeddingResponse

if TYPE_CHECKING:
    from stdapi.models import InvokeResult
    from stdapi.types import JsonMapping


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
        request = _Request()
        request.update(extra_params)  # type:ignore[typeddict-item]
        if dimensions:
            request["dimensions"] = dimensions

        input_tokens = 0
        output_tokens = 0
        embeddings = []
        for result in await gather(*(self._invoke(request, v) for v in inputs)):
            embeddings.append(result.response["embedding"])
            input_tokens += (
                result.response.get("inputTextTokenCount") or result.input_tokens or 0
            )
            output_tokens += result.output_tokens or 0

        return EmbeddingResponse(
            embeddings=embeddings,
            prompt_tokens=input_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    async def _invoke(
        self, request: _Request, value: InputFileUrl | str
    ) -> InvokeResult[_Response]:
        """Call the model with an input file.

        Args:
            request: The request object.
            value: The input value to be used with the request.

        Returns:
            InvokeResult containing the model response and token counts.
        """
        return await self.invoke(
            _Request(inputImage=await value.to_base64(), **request)
            if isinstance(value, InputFile)
            else _Request(inputText=value, **request)
        )
