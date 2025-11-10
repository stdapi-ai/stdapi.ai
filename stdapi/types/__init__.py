"""Local types for the stdapi.ai project.

This package hosts OpenAI-compatible type definitions used by the routes to
avoid hard runtime dependencies on the official openai.types package.
"""

from pydantic import BaseModel, ConfigDict, JsonValue

from stdapi.config import SETTINGS


class BaseModelRequest(BaseModel):
    """Pydantic Basemodel request."""

    model_config = ConfigDict(
        extra="forbid" if SETTINGS.strict_input_validation else "allow", frozen=True
    )


class BaseModelRequestWithExtra(BaseModel):
    """Pydantic Basemodel request storing extra JSON fields."""

    model_config = ConfigDict(extra="allow", frozen=True)
    __pydantic_extra__: dict[str, JsonValue] = {}


class BaseModelResponse(BaseModel):
    """Pydantic Basemodel response."""

    model_config = ConfigDict(extra="forbid")
