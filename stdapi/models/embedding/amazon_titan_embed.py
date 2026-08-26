"""Amazon Titan embedding models.

- amazon.titan-embed-image-v1
- amazon.titan-embed-text-v1
- amazon.titan-embed-text-v2:0
"""

from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

from stdapi.input_file import InputFile, InputFileUrl
from stdapi.models.embedding import EmbeddingModelBase, EmbeddingResponse

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stdapi.models import InvokeResult
    from stdapi.models.embedding import EmbedInputValue
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

    __slots__ = ()

    MATCHER = "amazon.titan-embed-"

    async def embed_text(
        self,
        inputs: Sequence[EmbedInputValue],
        dimensions: int | None,
        extra_params: JsonMapping,
    ) -> EmbeddingResponse:
        """Get embeddings for text.

        Args:
            inputs: Texts and images to embed, one vector each.
            dimensions: Number of dimensions.
            extra_params: Extra model parameters.

        Returns:
            Embedding response.

        Raises:
            ApiError: When an input groups several content parts.
        """
        values = self._single_part_inputs(inputs)
        request = self._base_request(dimensions, extra_params)

        input_tokens = 0
        output_tokens = 0
        embeddings = []
        embeddings_by_type: dict[str, list[list[float | int]]] | None = (
            {} if "embeddingTypes" in request else None
        )
        for result in await self._gather_bounded(
            self._invoke(request, value) for value in values
        ):
            embeddings.append(result.response["embedding"])
            input_tokens += (
                result.response.get("inputTextTokenCount") or result.input_tokens or 0
            )
            output_tokens += result.output_tokens or 0
            if embeddings_by_type is not None:
                # Titan invokes once per input; aggregate each call's by-type
                # vectors into the combined per-type lists.
                by_type = result.response.get("embeddingsByType") or {}
                for embedding_type in ("float", "binary"):
                    if embedding_type in by_type:
                        vector: list[float | int] = list(by_type[embedding_type])
                        embeddings_by_type.setdefault(embedding_type, []).append(vector)

        return EmbeddingResponse(
            embeddings=embeddings,
            embeddings_by_type=embeddings_by_type,
            prompt_tokens=input_tokens,
            total_tokens=input_tokens + output_tokens,
        )

    @staticmethod
    def _base_request(dimensions: int | None, extra_params: JsonMapping) -> _Request:
        """Return the request parameters shared by every input of one call.

        Args:
            dimensions: Number of dimensions.
            extra_params: Extra model parameters.

        Returns:
            The request, without its input.
        """
        request = _Request()
        request.update(extra_params)  # type:ignore[typeddict-item]
        if dimensions:
            request["dimensions"] = dimensions
        return request

    @staticmethod
    async def _with_input(request: _Request, value: InputFileUrl | str) -> _Request:
        """Return *request* carrying one input.

        Args:
            request: The request parameters shared by every input.
            value: The input value to embed.

        Returns:
            The complete request body.
        """
        return (
            _Request(inputImage=await value.to_base64(), **request)
            if isinstance(value, InputFile)
            else _Request(inputText=value, **request)
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
        return await self.invoke(await self._with_input(request, value))

    async def build_batch_request(
        self,
        inputs: Sequence[EmbedInputValue],
        dimensions: int | None,
        extra_params: JsonMapping,
    ) -> dict[str, Any]:
        """Build the request body of one embedding, without sending it.

        Args:
            inputs: The single text or image to embed.
            dimensions: Number of dimensions.
            extra_params: Extra model parameters.

        Returns:
            The request body.

        Raises:
            ApiError: When the request carries more than one input, or an
                input grouping several content parts.
        """
        return await self._with_input(  # type: ignore[return-value]
            self._base_request(dimensions, extra_params), self._one_input(inputs)
        )

    def read_batch_response(self, output: JsonMapping) -> EmbeddingResponse:
        """Read an embedding answer out of a model's own response body.

        Args:
            output: The response body the model produced.

        Returns:
            Embedding response.
        """
        count = output.get("inputTextTokenCount")
        tokens = int(count) if isinstance(count, (int, float)) else 0
        embedding = output.get("embedding")
        return EmbeddingResponse(
            embeddings=[embedding] if isinstance(embedding, list) else [],  # type: ignore[list-item]
            prompt_tokens=tokens,
            total_tokens=tokens,
        )
