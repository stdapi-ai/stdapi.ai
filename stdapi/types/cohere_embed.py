"""Local Cohere-compatible embed types (Cohere v1 and v2 Embed APIs)."""

from base64 import b64encode
from struct import pack
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal, Self

from pydantic import Field, model_validator

from stdapi.api_errors import ApiError
from stdapi.input_file import InputFileUrl
from stdapi.types import BaseModelRequest, BaseModelRequestWithExtra, BaseModelResponse
from stdapi.types.cohere import ApiMeta

if TYPE_CHECKING:
    from collections.abc import Sequence

    from stdapi.models.embedding import EmbeddingResponse, EmbedInputValue

#: Cohere embed `input_type` values.
_InputType = Literal[
    "search_document", "search_query", "classification", "clustering", "image"
]

#: Embedding types natively supported by Bedrock Cohere Embed models.
_COHERE_NATIVE_EMBEDDING_TYPES = frozenset(
    {"float", "int8", "uint8", "binary", "ubinary"}
)
#: Model ID prefix of the only Titan Embed model accepting `embeddingTypes`.
TITAN_EMBED_V2_PREFIX = "amazon.titan-embed-text-v2"
#: Embedding types natively supported by Bedrock Titan Embed v2 models.
_TITAN_NATIVE_EMBEDDING_TYPES = frozenset({"float", "binary"})
#: Embedding types natively supported by every other Bedrock embedding model.
_DEFAULT_NATIVE_EMBEDDING_TYPES = frozenset({"float"})


class EmbedInputImageUrl(BaseModelRequest):
    """Image of an input content part."""

    url: InputFileUrl = Field(
        description=(
            "Image data URI. URLs and S3 URIs are also accepted (beyond the "
            "original Cohere API)."
        )
    )


class EmbedInputContentText(BaseModelRequest):
    """Text part of an input."""

    type: Literal["text"] = Field(description="Content part type.")
    text: str = Field(description="Text embedded as part of this input.")


class EmbedInputContentImageUrl(BaseModelRequest):
    """Image part of an input."""

    type: Literal["image_url"] = Field(description="Content part type.")
    image_url: EmbedInputImageUrl = Field(
        description="Image embedded as part of this input."
    )


#: One content part of an input, discriminated on its `type` field.
EmbedInputContent = Annotated[
    EmbedInputContentText | EmbedInputContentImageUrl, Field(discriminator="type")
]


class EmbedInput(BaseModelRequest):
    """One input to embed, made of one or more content parts."""

    content: list[EmbedInputContent] = Field(
        min_length=1,
        description=(
            "Content parts embedded together into a single vector. Two or more "
            "parts require a Cohere Embed v4 model; a single part is accepted "
            "by every embedding model."
        ),
    )


class _EmbedRequestBase(BaseModelRequestWithExtra):
    """Shared request fields and validation for the v1 and v2 Embed APIs."""

    #: Input fields this Embed API accepts, named in the "no input" error.
    _INPUT_FIELDS: ClassVar[str] = "`texts` or `images`"

    model: str = Field(
        description="ID of the model to use.", min_length=1, max_length=255
    )
    texts: list[str] | None = Field(
        default=None,
        min_length=1,
        description="An array of strings for the model to embed.",
    )
    images: list[InputFileUrl] | None = Field(
        default=None,
        min_length=1,
        description=(
            "An array of image data URIs for the model to embed. "
            "URLs and S3 URIs are also accepted (beyond the original Cohere API)."
        ),
    )
    embedding_types: (
        list[Literal["float", "int8", "uint8", "binary", "ubinary", "base64"]] | None
    ) = Field(
        default=None,
        description=(
            "Embedding types to return. `int8`/`uint8`/`binary`/`ubinary` are "
            "supported by Cohere Embed models, `binary` also by Titan "
            "Embed v2; other combinations return 400. `base64` is always "
            "accepted and computed client-side (little-endian float32 bytes) "
            "from the `float` embedding."
        ),
    )
    truncate: Literal["NONE", "START", "END"] | None = Field(
        default=None,
        description=(
            "How to handle inputs longer than the maximum token length. "
            "Supported by Cohere models only."
        ),
    )

    def embed_inputs(self) -> list[EmbedInputValue]:
        """Return every input to embed, in the order they are returned.

        Returns:
            The texts of the request, then its images.
        """
        return [*(self.texts or ()), *(self.images or ())]

    @model_validator(mode="after")
    def _unsupported(self) -> Self:
        """Validate unsupported or incompatible embed options.

        Raises:
            ValueError: When no input is provided or an unsupported option is requested.
        """
        if "inputs" in (self.model_extra or {}):
            if self.model_extra["inputs"] is not None:  # type: ignore[index]
                msg = (
                    "The `inputs` field is not part of this Embed API version. "
                    "Use `texts` and `images`, or the v2 Embed endpoint."
                )
                raise ValueError(msg)
            # An explicit null is treated as absent, not forwarded to the model.
            self.model_extra.pop("inputs")  # type: ignore[union-attr]
        if not self.embed_inputs():
            msg = f"Provide at least one of {self._INPUT_FIELDS}."
            raise ValueError(msg)
        return self


class EmbedRequest(_EmbedRequestBase):
    """Request body for creating embeddings."""

    _INPUT_FIELDS: ClassVar[str] = "`texts`, `images` or `inputs`"

    inputs: list[EmbedInput] | None = Field(
        default=None,
        min_length=1,
        description=(
            "Inputs to embed, one vector each, as objects carrying a `content` "
            "array of `text` and `image_url` parts. Embedded after `texts` and "
            "`images` when those are sent too."
        ),
    )
    input_type: _InputType = Field(
        ...,
        description=(
            "Specifies the type of input passed to the model. Applied to Cohere "
            "models; other embedding models have no equivalent and the "
            "value is ignored."
        ),
    )
    output_dimension: int | None = Field(
        default=None,
        ge=1,
        le=8192,
        description=(
            "The number of dimensions of the output embedding. "
            "Supported by some models only."
        ),
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description=(
            "The maximum number of tokens to embed per input. "
            "Supported by some models only."
        ),
    )
    priority: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Accepted for compatibility and ignored. Cohere API request "
            "scheduling priority is not applicable on this implementation."
        ),
    )

    def embed_inputs(self) -> list[EmbedInputValue]:
        """Return every input to embed, in the order they are returned.

        Returns:
            The texts of the request, then its images, then its `inputs`
            entries; an entry of several content parts is returned as the list
            of its values, a single-part entry as the value itself.
        """
        values = super().embed_inputs()
        for entry in self.inputs or ():
            parts: list[InputFileUrl | str] = [
                part.text if part.type == "text" else part.image_url.url
                for part in entry.content
            ]
            values.append(parts[0] if len(parts) == 1 else parts)
        return values

    def count_images(self) -> int:
        """Return how many images the request submits.

        Returns:
            The `images` entries plus the image parts of every `inputs` entry.
        """
        return len(self.images or ()) + sum(
            part.type == "image_url"
            for entry in self.inputs or ()
            for part in entry.content
        )


class EmbedV1Request(_EmbedRequestBase):
    """Request body for creating embeddings (Cohere v1 Embed API)."""

    input_type: _InputType | None = Field(
        default=None,
        description=(
            "Specifies the type of input passed to the model. Optional in the "
            "v1 API: forwarded to Cohere models when provided (defaults to "
            "`search_document` otherwise); other embedding models have no "
            "equivalent and the value is ignored."
        ),
    )


class EmbeddingsByType(BaseModelResponse):
    """Embeddings keyed by embedding type.

    Only the requested embedding types are populated, matching the Cohere API.
    """

    float_: list[list[float]] | None = Field(
        default=None,
        serialization_alias="float",
        description="Float embedding vectors, one per input.",
    )
    int8: list[list[int]] | None = Field(
        default=None,
        description="Int8-quantized embedding vectors, one per input. Cohere models only.",
    )
    uint8: list[list[int]] | None = Field(
        default=None,
        description="Uint8-quantized embedding vectors, one per input. Cohere models only.",
    )
    binary: list[list[int]] | None = Field(
        default=None,
        description=(
            "Bit-packed signed-binary embedding vectors, one per input. "
            "Cohere and Titan Embed v2 models."
        ),
    )
    ubinary: list[list[int]] | None = Field(
        default=None,
        description=(
            "Bit-packed unsigned-binary embedding vectors, one per input. Cohere models only."
        ),
    )
    base64: list[str] | None = Field(
        default=None,
        description=(
            "Base64-encoded float32 embedding vectors (little-endian byte order), "
            "one per input, computed client-side from the `float` embedding."
        ),
    )


class ImageDescription(BaseModelResponse):
    """Metadata of an embedded image, echoed back for image inputs."""

    width: int = Field(description="Image width in pixels.")
    height: int = Field(description="Image height in pixels.")
    format: str = Field(description="Image format (e.g. `png`, `jpeg`).")
    bit_depth: int = Field(description="Image bit depth.")


class EmbedResponse(BaseModelResponse):
    """Embed response model."""

    response_type: Literal["embeddings_by_type"] = Field(
        default="embeddings_by_type",
        description="Response envelope type, always `embeddings_by_type`.",
    )
    id: str = Field(description="Unique identifier of the request.")
    embeddings: EmbeddingsByType = Field(
        description="Embeddings grouped by embedding type."
    )
    texts: list[str] | None = Field(
        default=None, description="The text entries for which embeddings were returned."
    )
    images: list[ImageDescription] | None = Field(
        default=None,
        description="Metadata of the image entries for which embeddings were returned.",
    )
    meta: ApiMeta = Field(description="Response metadata.")


def resolve_embedding_types(
    model_id: str, embedding_types: Sequence[str] | None
) -> list[str] | None:
    """Resolve the native Bedrock embedding types to request for a model.

    Args:
        model_id: Resolved Bedrock model identifier.
        embedding_types: Client-requested Cohere embedding types, if any.

    Returns:
        The embedding types to forward to the backend, or `None` to keep the
        default float-only backend behavior. `base64` is never forwarded: it
        is computed client-side from the `float` embedding. `float` is always
        added to a non-empty result, so the backend keeps returning it
        alongside the requested types.

    Raises:
        ApiError: If a requested type is not supported by the resolved model.
    """
    if not embedding_types:
        return None
    if model_id.startswith("cohere."):
        native_types = _COHERE_NATIVE_EMBEDDING_TYPES
    elif model_id.startswith(TITAN_EMBED_V2_PREFIX):
        native_types = _TITAN_NATIVE_EMBEDDING_TYPES
    else:
        native_types = _DEFAULT_NATIVE_EMBEDDING_TYPES
    requested = set(embedding_types)
    unsupported = requested - native_types - {"base64"}
    if unsupported:
        msg = (
            f"Embedding types {sorted(unsupported)} are not supported on this backend."
        )
        raise ApiError(msg)
    forwarded = requested & native_types
    if forwarded or "base64" in requested:
        forwarded.add("float")
    return sorted(forwarded) or None


def build_embeddings_by_type(
    response: EmbeddingResponse, embedding_types: Sequence[str] | None
) -> EmbeddingsByType:
    """Build the Cohere `embeddings_by_type` response from a model response.

    Args:
        response: Embedding model response (float vectors and/or a
            provider-native `embeddings_by_type` dict).
        embedding_types: Client-requested Cohere embedding types, if any
            (defaults to `["float"]` when unset).

    Returns:
        The embeddings grouped by the requested types, base64-encoding the
        float embedding client-side when `base64` was requested.
    """
    by_type = response.embeddings_by_type or {}
    float_vectors = by_type.get("float", response.embeddings)
    requested = set(embedding_types) if embedding_types else {"float"}
    # Bedrock only ever returns integers for the quantized types below (only
    # "float" uses floats), so this narrows the shared `float | int` element
    # type declared on `EmbeddingResponse.embeddings_by_type`.
    quantized: dict[str, list[list[int]]] = by_type  # type: ignore[assignment]
    return EmbeddingsByType(
        float_=float_vectors if "float" in requested else None,
        int8=quantized.get("int8") if "int8" in requested else None,
        uint8=quantized.get("uint8") if "uint8" in requested else None,
        binary=quantized.get("binary") if "binary" in requested else None,
        ubinary=quantized.get("ubinary") if "ubinary" in requested else None,
        base64=(
            [_encode_base64(vector) for vector in float_vectors]
            if "base64" in requested
            else None
        ),
    )


def _encode_base64(vector: list[float]) -> str:
    """Encode a float embedding vector as base64 little-endian float32 bytes.

    Args:
        vector: Float embedding vector.

    Returns:
        Base64-encoded little-endian float32 bytes, matching the Cohere API
        `base64` embedding encoding.
    """
    return b64encode(pack(f"<{len(vector)}f", *vector)).decode("ascii")


class EmbedV1FloatsResponse(BaseModelResponse):
    """Embed response model (legacy v1 `embeddings_floats` format)."""

    response_type: Literal["embeddings_floats"] = Field(
        default="embeddings_floats",
        description="Response envelope type, always `embeddings_floats`.",
    )
    id: str = Field(description="Unique identifier of the request.")
    embeddings: list[list[float]] = Field(
        description="Float embedding vectors, one per input."
    )
    texts: list[str] | None = Field(
        default=None, description="The text entries for which embeddings were returned."
    )
    images: list[ImageDescription] | None = Field(
        default=None,
        description="Metadata of the image entries for which embeddings were returned.",
    )
    meta: ApiMeta = Field(description="Response metadata.")
