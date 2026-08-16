"""Unified API errors — format-agnostic exceptions for all API routes."""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    EndpointResolutionError,
)

if TYPE_CHECKING:
    from collections.abc import Generator

#: AWS error codes meaning the identity a call was signed with was denied it.
ACCESS_DENIED_CODES: Final = frozenset({"AccessDeniedException", "AccessDenied"})

#: Botocore failures meaning the service endpoint is unreachable or timed out.
UNREACHABLE_ENDPOINT_ERRORS: Final = (
    EndpointConnectionError,
    EndpointResolutionError,
    ConnectTimeoutError,
)


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
        self, model: str, *, detail: str | None = None, status: int | None = None
    ) -> None:
        """Refuse a model this server does not serve, and say where to find one it does.

        Args:
            model: The requested model identifier that is unsupported or not accessible.
            detail: Optional extra context (e.g. deprecation info) appended after the
                standard "does not exist" sentence.
            status: Optional HTTP status code override.  When ``None`` the
                class-level default (404) is used.
        """
        extra = f" {detail}" if detail else ""
        super().__init__(
            f"The model `{model}` does not exist or you do not have access to it."
            f"{extra} Call the models endpoint to list the models this server provides.",
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


class FileNotExistError(ApiError):
    """Requested file does not exist or has expired."""

    status = 404
    code = "not_found"


class InputAccessDeniedError(ApiError):
    """An object the request named could not be read.

    Deliberately not a :class:`FeatureUnavailableError`: the refused object is
    the one the caller pointed at, in storage the deployment does not own, so
    the caller is the only one who can grant access to it or name another.
    """

    code = "input_access_denied"

    def __init__(self, uri: str) -> None:
        """Refuse an input the server is not allowed to read.

        Args:
            uri: The input reference, as the caller wrote it.  Only ever an
                object the caller named, never one of the deployment's own.
        """
        super().__init__(
            f"Access denied reading the input '{uri}'. Grant this server read "
            "access to that object, or provide an input it can read."
        )


class FeatureUnavailableError(ApiError):
    """A feature this deployment cannot run: missing permission, setting or resource.

    Both audiences are served in one place: the caller always reads the same
    generic message, whatever is missing, while the operator reads *detail* in
    the server log at ``warning`` level.
    """

    status = 503
    code = "feature_unavailable"

    def __init__(self, feature: str, detail: str) -> None:
        """Refuse a feature the deployment cannot serve, and report why in the log.

        Args:
            feature: The feature as the caller knows it, e.g. "The Batch API".
            detail: What is missing, named for the operator: the permission,
                the setting or the resource.
        """
        # Imported here: stdapi.monitoring imports this module (import cycle).
        from stdapi.monitoring import log_error_details  # noqa: PLC0415

        log_error_details(detail, level="warning")
        super().__init__(
            f"{feature} is not available on the current server. "
            "Please contact the administrator to enable it."
        )


# Names the denied IAM action, which the handler-level net cannot.
@contextmanager
def feature_unavailable_guard(
    feature: str, *, missing: str, unreachable: str | None = None
) -> Generator[None]:
    """Answer a denied (or unreachable) AWS call as a feature this deployment lacks.

    Args:
        feature: The feature as the caller knows it, e.g. "The Batch API".
        missing: The IAM permissions the denied call needs, named for the
            operator.
        unreachable: What the operator must configure when the service endpoint
            cannot be reached at all; without it, that failure propagates.

    Yields:
        None

    Raises:
        FeatureUnavailableError: The call was denied, or its endpoint is
            unreachable and *unreachable* says what that means.
    """
    try:
        yield
    except ClientError as error:
        if error.response["Error"]["Code"] not in ACCESS_DENIED_CODES:
            raise
        detail = f"Access denied: the server role is missing {missing}."
        raise FeatureUnavailableError(feature, detail) from error
    except UNREACHABLE_ENDPOINT_ERRORS as error:
        if unreachable is None:
            raise
        raise FeatureUnavailableError(feature, unreachable) from error


#: Feature name a denial no call site named a feature for is refused under.
_DENIED_FEATURE: Final = "The requested feature"


def denied_feature_unavailable(exc: ClientError) -> FeatureUnavailableError | None:
    """Answer a denial of the server's own role as a feature the deployment lacks.

    Last resort for the denials no call site wrapped in
    :func:`feature_unavailable_guard`, so that one under-permissioned
    deployment answers the same way on every route.

    A model invocation signed as its end user (``aws_bedrock_user_role_arn``)
    is the exception: AWS evaluated a policy written about *that* caller, an
    authorization decision the deployment makes on purpose, so it stays the
    caller's own permission error instead of reading as an outage.

    The other request-attributable denial never reaches here: an object the
    caller named in an ``s3://`` input is refused as
    :class:`InputAccessDeniedError` at the read itself, which is the only place
    the refused bucket is known (``aws_s3.caller_input_denial_guard``).

    Args:
        exc: The AWS client error to classify.

    Returns:
        The error to answer the caller with, or ``None`` when *exc* is not a
        denial of the server's own role.
    """
    if exc.response["Error"]["Code"] not in ACCESS_DENIED_CODES:
        return None
    # Imported here: both modules import this one (import cycle).
    from stdapi.aws import signed_as_end_user  # noqa: PLC0415
    from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415

    operation = exc.operation_name
    if signed_as_end_user(operation):
        return None
    log = REQUEST_LOG.get(None)
    model_id = log.get("model_id") if log is not None else None
    model = f" for model {model_id}" if model_id else ""
    return FeatureUnavailableError(
        _DENIED_FEATURE,
        f"Access denied calling {operation}{model}: grant the server role the "
        f"permission AWS names. AWS reported: {exc.response['Error']['Message']}",
    )
