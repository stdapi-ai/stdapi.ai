"""OpenAI Exceptions."""

from collections.abc import Iterable


class OpenaiError(Exception):
    """OpenAI API error."""

    status: int = 400
    type: str = "invalid_request_error"
    code: str | None = None
    param: str | None = None


class OpenaiUnsupportedModelError(OpenaiError):
    """Unsupported model error."""

    status = 404
    code = "model_not_found"

    def __init__(
        self, model: str, available_models: Iterable[str] | None = None
    ) -> None:
        """Create an unsupported model error with optional alternatives.

        Args:
            model: The requested model identifier that is unsupported or not accessible.
            available_models: Optional iterable of available model identifiers to include
                in the error message to guide clients toward valid choices.
        """
        models = (
            f" Available models: {', '.join(available_models)}"
            if available_models
            else ""
        )
        super().__init__(
            f"The model `{model}` does not exist or you do not have access to it.{models}"
        )


class OpenaiUnsupportedParameterError(OpenaiError):
    """Unsupported parameter error."""

    code = "unsupported_parameter"

    def __init__(self, param: str) -> None:
        """Create an unsupported parameter error.

        Args:
            param: The name of the request parameter that is not supported for the
                current model or backend.
        """
        self.param = param
        super().__init__(
            f"Unsupported parameter: '{param}' is not supported with this model."
        )
