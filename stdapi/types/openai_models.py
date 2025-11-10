"""Local OpenAI-compatible model types."""

from typing import Literal

from pydantic import Field

from stdapi.types import BaseModelResponse


# Ref: openai.types.Model
class Model(BaseModelResponse):
    """OpenAI-compatible Model."""

    id: str = Field(description="The unique identifier for the model.", min_length=1)
    object: Literal["model"] = Field(
        default="model", description='The object type. Always "model".'
    )
    created: int = Field(
        ge=0, description="The Unix timestamp when the model was created."
    )
    owned_by: str = Field(
        description="The organization that owns the model.", min_length=1
    )
