"""Cohere embedding models.

- cohere.embed-english-v3
- cohere.embed-multilingual-v3
- cohere.embed-v4:0
"""

from asyncio import gather
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict

from stdapi.api_errors import ApiError
from stdapi.input_file import InputFile, InputFileUrl
from stdapi.models._cohere import COHERE_ALIAS_SUBSTITUTIONS, COHERE_EMBED_ALIAS_MATCHER
from stdapi.models.embedding import (
    FUSED_INPUTS_UNSUPPORTED,
    EmbeddingImageDescription,
    EmbeddingModelBase,
    EmbeddingResponse,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stdapi.models.embedding import EmbedInputValue
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
        - truncate=START or END (Replaced by LEFT, RIGHT in V4)

    V3 accepts a single image per call; V4 takes several, one vector each.
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
    """Cohere embedding model.

    Each model is also published under the name Cohere's own API uses, so an
    application already calling Cohere only changes its base URL:
    ``cohere.embed-english-v3`` answers to ``embed-english-v3.0`` and
    ``cohere.embed-v4:0`` to ``embed-v4.0``.

    Ref: https://docs.cohere.com/docs/models
    """

    __slots__ = ()

    MATCHER = "cohere.embed-"
    ALIAS_MATCHER = COHERE_EMBED_ALIAS_MATCHER
    ALIAS_SUBSTITUTIONS = COHERE_ALIAS_SUBSTITUTIONS

    #: Characters a v3 text input accepts; longer texts are rejected, never truncated.
    _V3_MAX_INPUT_CHARACTERS = 2048

    @property
    def max_input_characters(self) -> int:
        """Characters accepted in one text input, or 0 when none is documented."""
        # Only v3 documents a ceiling; later versions must not inherit it.
        return self._V3_MAX_INPUT_CHARACTERS if self._is_v3 else 0

    @property
    def _is_v3(self) -> bool:
        """Whether this is an Embed v3 model, the generation before `inputs`."""
        return "-v3" in self._model_id

    async def embed_text(
        self,
        inputs: Sequence[EmbedInputValue],
        dimensions: int | None,
        extra_params: JsonMapping,
    ) -> EmbeddingResponse:
        """Get embeddings for text.

        Args:
            inputs: Texts and images to embed, one vector per entry; an entry
                listing several values embeds them together into one vector.
            dimensions: Number of dimensions.
            extra_params: Extra model parameters.

        Returns:
            Embedding response.

        Raises:
            ApiError: When the model cannot embed the given inputs.
        """
        request = await self._build_request(inputs, dimensions, extra_params)
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

    async def _build_request(
        self,
        inputs: Sequence[EmbedInputValue],
        dimensions: int | None,
        extra_params: JsonMapping,
    ) -> _Request:
        """Build the request body embedding *inputs*.

        Args:
            inputs: Texts and images to embed, one vector per entry; an entry
                listing several values embeds them together into one vector.
            dimensions: Number of dimensions.
            extra_params: Extra model parameters.

        Returns:
            The request body.

        Raises:
            ApiError: When an Embed v3 model is asked for texts and images
                together, or for an entry fusing several content parts.
        """
        request = _Request(input_type="search_document")
        request.update(extra_params)  # type:ignore[typeddict-item]
        if dimensions is not None:
            request["output_dimension"] = dimensions

        fused = any(isinstance(value, list) for value in inputs)
        is_data: tuple[bool, ...] = tuple(isinstance(v, InputFile) for v in inputs)
        if not fused and all(is_data):
            image_inputs: Sequence[InputFile] = inputs  # type: ignore[assignment]
            request["images"] = await gather(
                *(image.to_data_uri() for image in image_inputs)
            )
            if self._is_v3:
                request["input_type"] = "image"
        elif not fused and not any(is_data):
            texts: list[str] = inputs  # type: ignore[assignment]
            request["texts"] = texts
        elif self._is_v3:
            # Both remaining shapes need the `inputs` field, which v3 has not.
            if fused:
                raise ApiError(FUSED_INPUTS_UNSUPPORTED)
            msg = (
                "Cohere Embed v3 models embed either texts or images in a request, "
                "not both. Send them in separate requests, or select a model that "
                "takes both at once: Cohere Embed v4, Amazon Nova 2 Multimodal "
                "Embeddings, Amazon Titan Embed Image, or TwelveLabs Marengo Embed."
            )
            raise ApiError(msg)
        else:
            request["inputs"] = [
                _InputContent(content=content)
                for content in await gather(
                    *(self._to_input_contents(value) for value in inputs)
                )
            ]
        return request

    async def build_batch_request(
        self,
        inputs: Sequence[EmbedInputValue],
        dimensions: int | None,
        extra_params: JsonMapping,
    ) -> dict[str, Any]:
        """Build the request body of one embedding, without sending it.

        Args:
            inputs: Texts and images to embed, one vector per entry; an entry
                listing several values embeds them together into one vector.
            dimensions: Number of dimensions.
            extra_params: Extra model parameters.

        Returns:
            The request body.

        Raises:
            ApiError: When the model cannot embed the given inputs.
        """
        return await self._build_request(inputs, dimensions, extra_params)  # type: ignore[return-value]

    def read_batch_response(self, output: JsonMapping) -> EmbeddingResponse:
        """Read an embedding answer out of a model's own response body.

        Args:
            output: The response body the model produced.

        Returns:
            Embedding response.
        """
        embeddings = output.get("embeddings")
        if isinstance(embeddings, dict):
            embeddings = embeddings.get("float")
        return EmbeddingResponse(
            embeddings=embeddings if isinstance(embeddings, list) else []  # type: ignore[arg-type]
        )

    @classmethod
    async def _to_input_contents(
        cls, value: EmbedInputValue
    ) -> list[_InputContentImageUrl | _InputContentText]:
        """Convert one input into the content parts of its fused entry.

        Args:
            value: One text or image, or the values fused into one entry.

        Returns:
            The content parts of the entry, in order.
        """
        values = value if isinstance(value, list) else [value]
        return list(await gather(*(cls._to_input_content(item) for item in values)))

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
