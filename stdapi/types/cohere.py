"""Shared Cohere-compatible response metadata types."""

from pydantic import Field

from stdapi.types import BaseModelResponse


class ApiVersion(BaseModelResponse):
    """API version metadata."""

    version: str = Field(default="2", description="The Cohere API version.")


class BilledUnits(BaseModelResponse):
    """Billed units metadata."""

    input_tokens: int | None = Field(default=None, description="Billed input tokens.")
    images: int | None = Field(default=None, description="Billed input images.")
    search_units: int | None = Field(
        default=None,
        description="Billed search units (one unit covers a query with up to 100 documents).",
    )


class ApiMeta(BaseModelResponse):
    """Response metadata."""

    api_version: ApiVersion = Field(
        default_factory=ApiVersion, description="API version information."
    )
    billed_units: BilledUnits = Field(description="Billing information.")
