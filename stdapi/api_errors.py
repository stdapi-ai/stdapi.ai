"""Unified API errors — format-agnostic exceptions for all API routes."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


class ApiError(Exception):
    """Base API error raised by low-level code.

    Exception handlers in ``main.py`` convert this to the correct JSON envelope
    (OpenAI or Anthropic) depending on the route that was called.
    """

    status: int = 400
    code: str | None = None
    param: str | None = None

    def __init__(self, message: str, *, status: int | None = None) -> None:
        """Create an API error.

        When *status* is provided and the subclass has not overridden
        ``error_type``, the type is automatically derived from the status code
        (e.g. 401 → ``"authentication_error"``).

        Args:
            message: Human-readable error message.
            status: Optional HTTP status code override (defaults to class-level ``status``).
        """
        if status is not None:
            self.status = status
        super().__init__(message)


class UnsupportedModelError(ApiError):
    """Requested model does not exist or is not accessible."""

    status = 404
    code = "model_not_found"

    def __init__(
        self,
        model: str,
        available_models: Iterable[str] | None = None,
        *,
        status: int | None = None,
    ) -> None:
        """Create an unsupported model error with optional alternatives.

        Args:
            model: The requested model identifier that is unsupported or not accessible.
            available_models: Optional iterable of available model identifiers to include
                in the error message to guide clients toward valid choices.
            status: Optional HTTP status code override.  When ``None`` the
                class-level default (404) is used.
        """
        models = (
            f" Available models: {', '.join(available_models)}"
            if available_models
            else ""
        )
        super().__init__(
            f"The model `{model}` does not exist or you do not have access to it.{models}",
            status=status,
        )


class UnsupportedParameterError(ApiError):
    """A request parameter is not supported for the current model."""

    code = "unsupported_parameter"

    def __init__(self, param: str) -> None:
        """Create an unsupported parameter error.

        Args:
            param: The name of the request parameter that is not supported with this model.
        """
        self.param = param
        super().__init__(
            f"Unsupported parameter: '{param}' is not supported with this model."
        )


class InvalidLanguageFormatError(ApiError):
    """Language format is invalid."""

    code = "invalid_language_format"
