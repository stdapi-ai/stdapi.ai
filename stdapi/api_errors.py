"""Unified API errors — format-agnostic exceptions for all API routes."""

import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Final

from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    EndpointResolutionError,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from botocore.model import OperationModel

#: AWS error codes meaning the identity a call was signed with was denied it.
ACCESS_DENIED_CODES: Final = frozenset({"AccessDeniedException", "AccessDenied"})

#: Botocore failures meaning the service endpoint is unreachable or timed out.
UNREACHABLE_ENDPOINT_ERRORS: Final = (
    EndpointConnectionError,
    EndpointResolutionError,
    ConnectTimeoutError,
)

#: Response key the AWS ``after-call`` hook records a denied call's IAM action under.
DENIED_CALL_KEY: Final = "stdapiDeniedCall"

#: How an AWS denial names the action it refused, and the resource it was attempted on.
_DENIAL_RE: Final = re.compile(
    r"to perform:?\s+(?P<action>[a-z0-9][a-z0-9-]*:[A-Za-z0-9_*]+)"
    r"(?:\s+on\s+(?:resource:?\s+)?(?P<resource>arn:[^\s,\"']+))?"
)

#: The S3 operations authorized by an IAM action that is not their own name.
_S3_IAM_ACTIONS: Final[dict[str, str]] = {
    "CompleteMultipartUpload": "s3:PutObject",
    "CopyObject": "s3:PutObject",
    "CreateMultipartUpload": "s3:PutObject",
    "DeleteObjects": "s3:DeleteObject",
    "HeadBucket": "s3:ListBucket",
    "HeadObject": "s3:GetObject",
    "ListMultipartUploads": "s3:ListBucketMultipartUploads",
    "ListObjects": "s3:ListBucket",
    "ListObjectsV2": "s3:ListBucket",
    "ListParts": "s3:ListMultipartUploadParts",
    "UploadPart": "s3:PutObject",
    "UploadPartCopy": "s3:PutObject",
}


def iam_action(model: OperationModel) -> str:
    """Return the IAM action that authorizes an AWS API operation.

    The action is ``<service prefix>:<OperationName>``, and botocore already
    carries both halves: ``signingName`` is the service's IAM prefix wherever
    the two differ from the endpoint prefix (``pricing`` for ``api.pricing``,
    ``bedrock`` for ``bedrock-runtime``, ``aws-marketplace`` for
    ``metering.marketplace``), and the endpoint prefix is the prefix for the
    services that declare no signing name. Amazon S3 is the one service this
    server calls whose operations are authorized by a differently named action,
    so those are mapped explicitly.

    Args:
        model: Operation model of the call, as botocore passes it to its
            ``after-call`` hook.

    Returns:
        The IAM action, e.g. ``bedrock:ListProvisionedModelThroughputs``.
    """
    metadata = model.service_model.metadata
    prefix: str = metadata.get("signingName") or metadata["endpointPrefix"]
    if prefix == "s3":
        return _S3_IAM_ACTIONS.get(model.name, f"s3:{model.name}")
    return f"{prefix}:{model.name}"


def _denied_call(exc: ClientError) -> tuple[str, str | None, bool] | None:
    """Return the IAM action and resource a denial is about.

    The action AWS itself named in its message wins, because it is the
    authoritative answer for an operation authorized by several actions; the
    name derived from the call at :func:`iam_action` covers the services that
    answer a bare "Access Denied", Amazon S3 above all.

    Args:
        exc: The AWS client error to read.

    Returns:
        ``(action, resource ARN or None, whether AWS named the action itself)``,
        or ``None`` when *exc* is not a denial or names no action at all.
    """
    response: Any = exc.response
    error = response.get("Error") or {}
    if error.get("Code") not in ACCESS_DENIED_CODES:
        return None
    if match := _DENIAL_RE.search(error.get("Message") or ""):
        return match["action"], match["resource"], True
    action = (response.get(DENIED_CALL_KEY) or {}).get("action")
    return (action, None, False) if action else None


def iam_denial_detail(error: BaseException) -> str | None:
    """Return what an operator has to grant for a denied call, named in full.

    A denial that IAM itself evaluated always names the principal and the
    action ("... is not authorized to perform: <action>"), a grammar AWS keeps
    across every service, and only that one is certainly a policy gap. A
    service refusing the account outright answers the same code with prose of
    its own, so a denial AWS did not word that way is reported without claiming
    which of the two it is — the action is named either way, which is what an
    operator needs first.

    Args:
        error: The failure to describe; anything that is not an AWS denial
            answers ``None``, so a caller sorting mixed failures needs no
            type check of its own.

    Returns:
        The operator-facing sentence, or ``None`` when *error* is not a denial
        whose IAM action could be named.
    """
    if not isinstance(error, ClientError) or (denial := _denied_call(error)) is None:
        return None
    action, resource, named_by_aws = denial
    response: Any = error.response
    recorded = response.get(DENIED_CALL_KEY) or {}
    where = f" in {region}" if (region := recorded.get("region")) else ""
    on = f" on {resource}" if resource else ""
    if named_by_aws:
        return f"the server role is missing the IAM permission {action}{where}{on}"
    return (
        f"AWS denied {action}{where}{on}: grant that permission to the server "
        "role, unless the service does not offer the operation there"
    )


class ApiError(Exception):
    """Base API error raised by low-level code.

    Exception handlers in ``main.py`` convert this to the correct JSON envelope
    (OpenAI or Anthropic) depending on the route that was called.
    """

    status: int = 400
    code: str | None = None
    param: str | None = None
    #: Whether the message is sent as written instead of being flattened.
    disclosed: bool = False

    def __init__(self, message: str, *, status: int | None = None) -> None:
        """Create an API error.

        Args:
            message: Human-readable error message.
            status: Optional HTTP status code override (defaults to class-level ``status``).
        """
        if status is not None:
            self.status = status
        super().__init__(message)


class TenantCredentialError(ApiError):
    """A refusal of the AWS credential the request's tenant key registered.

    A ``403`` reaches a client as the bare word "Forbidden", which is right for
    a refusal of this deployment's own identity and wrong here: the message is
    fixed, written for the tenant, and names only the tenant's own resources —
    never an account, a role or an AWS error code of this deployment's. It is
    therefore sent as written, so the tenant can tell a broken registration
    apart from a model its account may not use.
    """

    status = 403
    disclosed = True


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


class AmbiguousModelError(ApiError):
    """A model pattern names several models the server cannot choose between."""

    code = "ambiguous_model"
    param = "model"

    def __init__(self, pattern: str, candidates: Sequence[str]) -> None:
        """Refuse a pattern that does not name one model, and list the ones it names.

        Args:
            pattern: The model pattern as the caller wrote it.
            candidates: The models the pattern matches, all released on the same
                date.  Already public on the models endpoint.
        """
        super().__init__(
            f"The model `{pattern}` matches several models released on the same "
            f"date: {', '.join(candidates)}. Name the one you want, or use a "
            "pattern that matches only it."
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

#: AWS error codes meaning a tenant role session died under the call signed with it.
_TENANT_SESSION_ERROR_CODES: Final = frozenset(
    {
        "ExpiredTokenException",
        "UnrecognizedClientException",
        "InvalidSignatureException",
    }
)


def _tenant_signed_denial(exc: ClientError, operation: str) -> ApiError | None:
    """Answer a refusal of a tenant-signed call as the tenant's own 403.

    AWS evaluated the tenant's own principal, so the refusal is neither an
    outage of this deployment (503) nor the caller's API key being wrong
    (401). The fixed messages carry nothing of the AWS failure: its raw text
    names principals and resources, and the generic handlers would otherwise
    map a dead session to an authentication error against the API key.

    Args:
        exc: The AWS client error to classify.
        operation: AWS API operation the failing call invoked.

    Returns:
        The 403 to answer the tenant with, or ``None`` when *exc* is not a
        tenant-signed denial.
    """
    # Imported here: both modules import this one (import cycle).
    from stdapi.aws import (  # noqa: PLC0415
        TENANT_ACCESS_DENIED_MESSAGE,
        TENANT_CREDENTIAL_FAILURE_MESSAGE,
        drop_tenant_sessions,
        signed_as_tenant,
    )

    code = exc.response["Error"]["Code"]
    if not signed_as_tenant(operation):
        return None
    if code in ACCESS_DENIED_CODES:
        return TenantCredentialError(TENANT_ACCESS_DENIED_MESSAGE)
    if code in _TENANT_SESSION_ERROR_CODES:
        # Imported here: stdapi.monitoring imports this module (import cycle).
        from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415

        # Dropped so the next request opens a fresh session instead of
        # re-signing with the dead one for up to an hour.
        log = REQUEST_LOG.get(None)
        if log is not None and (key_id := log.get("aws_tenant_key_id")):
            drop_tenant_sessions(key_id)
        return TenantCredentialError(TENANT_CREDENTIAL_FAILURE_MESSAGE)
    return None


def denied_feature_unavailable(exc: ClientError) -> ApiError | None:
    """Answer a denial of the server's own role as a feature the deployment lacks.

    Last resort for the denials no call site wrapped in
    :func:`feature_unavailable_guard`, so that one under-permissioned
    deployment answers the same way on every route.

    Two request-attributable identities are the exception, because AWS
    evaluated a policy written about *that* caller rather than about this
    deployment: a model invocation signed with the tenant's registered AWS
    credential answers the tenant's own fixed 403, and one signed as its end
    user (``aws_bedrock_user_role_arn``) stays the caller's own permission
    error instead of reading as an outage.

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
    operation = exc.operation_name
    if (tenant_denial := _tenant_signed_denial(exc, operation)) is not None:
        return tenant_denial
    if exc.response["Error"]["Code"] not in ACCESS_DENIED_CODES:
        return None
    # Imported here: both modules import this one (import cycle).
    from stdapi.aws import signed_as_end_user  # noqa: PLC0415
    from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415

    if signed_as_end_user(operation):
        return None
    log = REQUEST_LOG.get(None)
    model_id = log.get("model_id") if log is not None else None
    model = f" for model {model_id}" if model_id else ""
    missing = iam_denial_detail(exc) or (
        f"the server role is missing the permission that authorizes {operation}"
    )
    return FeatureUnavailableError(
        _DENIED_FEATURE,
        f"Access denied calling {operation}{model}: {missing}. "
        f"AWS reported: {exc.response['Error']['Message']}",
    )
