"""Local Cohere-compatible embed types (Cohere v1 and v2 Embed APIs)."""

from typing import Literal, Self

from pydantic import Field, model_validator

from stdapi.input_file import InputFileUrl
from stdapi.types import BaseModelRequestWithExtra, BaseModelResponse
from stdapi.types.cohere import ApiMeta

#: Cohere embed `input_type` values.
_InputType = Literal[
    "search_document", "search_query", "classification", "clustering", "image"
]


class _EmbedRequestBase(BaseModelRequestWithExtra):
    """Shared request fields and validation for the v1 and v2 Embed APIs."""

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
        description="Embedding types to return. Only `float` is supported.",
    )
    truncate: Literal["NONE", "START", "END"] | None = Field(
        default=None,
        description=(
            "How to handle inputs longer than the maximum token length. "
            "Supported by Cohere models only."
        ),
    )

    @model_validator(mode="after")
    def _unsupported(self) -> Self:
        """Validate unsupported or incompatible embed options.

        Raises:
            ValueError: When no input is provided or an unsupported option is requested.
        """
        if "inputs" in (self.model_extra or {}):
            if self.model_extra["inputs"] is not None:  # type: ignore[index]
                msg = "Fused multimodal `inputs` are not supported. Use `texts` or `images` instead."
                raise ValueError(msg)
            # An explicit null is treated as absent, not forwarded to the model.
            self.model_extra.pop("inputs")  # type: ignore[union-attr]
        if not self.texts and not self.images:
            msg = "Provide at least one of `texts` or `images`."
            raise ValueError(msg)
        if self.embedding_types and set(self.embedding_types) != {"float"}:
            msg = "Only `float` embeddings are supported on this backend."
            raise ValueError(msg)
        return self


class EmbedRequest(_EmbedRequestBase):
    """Request body for creating embeddings."""

    input_type: _InputType = Field(
        ...,
        description=(
            "Specifies the type of input passed to the model. Applied to Cohere "
            "models; other Bedrock embedding models have no equivalent and the "
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
            "scheduling priority is not applicable on AWS Bedrock."
        ),
    )


class EmbedV1Request(_EmbedRequestBase):
    """Request body for creating embeddings (Cohere v1 Embed API)."""

    input_type: _InputType | None = Field(
        default=None,
        description=(
            "Specifies the type of input passed to the model. Optional in the "
            "v1 API: forwarded to Cohere models when provided (the backend "
            "defaults to `search_document` otherwise); other Bedrock embedding "
            "models have no equivalent and the value is ignored."
        ),
    )


class EmbeddingsByType(BaseModelResponse):
    """Embeddings keyed by embedding type."""

    float_: list[list[float]] = Field(
        serialization_alias="float",
        description="Float embedding vectors, one per input.",
    )


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
    meta: ApiMeta = Field(description="Response metadata.")


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
    meta: ApiMeta = Field(description="Response metadata.")
