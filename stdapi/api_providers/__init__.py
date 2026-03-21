"""Handle API polymorphism based on the upstream API provider."""

from typing import TYPE_CHECKING, TypeVar, overload

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import Request
    from fastapi.responses import Response
    from pydantic import JsonValue

    from stdapi.monitoring import EventLog
    from stdapi.types import JsonMapping

T = TypeVar("T")

#: Route tag to request-id header name mapping (Else default to "x-request-id").
REQUEST_ID_HEADER_BY_TAG: dict[str, str] = {}

#: Default request-id header
_DEFAULT_REQUEST_ID_HEADER: str = "x-request-id"

#: Route tag to formatter.
FORMATTER_BY_TAG: dict[
    str, Callable[[int, str, str | None, str | None], tuple[dict[str, JsonValue], int]]
] = {}

#: Route tag to response-headers setter.
SET_RESPONSE_HEADERS_BY_TAG: dict[str, Callable[[Request, Response, int], None]] = {}

#: Route tag to log-fields setter.
SET_LOG_FIELDS_BY_TAG: dict[str, Callable[[Request, EventLog], None]] = {}


@overload
def _lookup_by_route_tag(
    request: Request,
    mapping: dict[
        str, Callable[[int, str, str | None, str | None], tuple[JsonMapping, int]]
    ],
    default: Callable[[int, str, str | None, str | None], tuple[JsonMapping, int]],
) -> Callable[[int, str, str | None, str | None], tuple[JsonMapping, int]]: ...


@overload
def _lookup_by_route_tag(
    request: Request, mapping: dict[str, str], default: str
) -> str: ...


def _lookup_by_route_tag[T](request: Request, mapping: dict[str, T], default: T) -> T:
    """Generic lookup function for route tag-based dispatch.

    Args:
        request: The incoming HTTP request object.
        mapping: Dictionary mapping route tags to values.
        default: Default value if no tag matches.

    Returns:
        Value from mapping if a matching tag is found, otherwise default.
    """
    if (route := request.scope.get("route")) is not None:
        for tag in getattr(route, "tags", None) or ():
            if (value := mapping.get(tag)) is not None:
                return value
    return default


def _default_formatter(
    status: int,
    message: str,
    param: str | None = None,  # noqa: ARG001
    code: str | None = None,  # noqa: ARG001
) -> tuple[JsonMapping, int]:
    """Format a minimal JSON error envelope.

    Args:
        status: HTTP status code.
        message: Error message.
        param: Input parameter related to the error, if any.
        code: Error code for further categorization, if any.

    Returns:
        A tuple of the JSON error body and the HTTP status code.
    """
    return {"error": message}, status


def format_http_error(
    request: Request,
    status: int,
    message: str,
    param: str | None = None,
    code: str | None = None,
) -> tuple[JsonMapping, int]:
    """Pick the correct error envelope based on the route.

    Args:
        request: The incoming HTTP request object.
        status: The HTTP status code for the error response.
        message: The error message to include in the response.
        param: An optional parameter indicating the specific input parameter related to the error. Defaults to None.
        code: An optional code for further categorization of the error. Defaults to None.

    Returns:
        A tuple containing the error response in JSON format and the corresponding HTTP status code.
    """
    return _lookup_by_route_tag(request, FORMATTER_BY_TAG, _default_formatter)(
        status, message, param, code
    )


def get_request_id_header(request: Request) -> str:
    """Return the appropriate request-id header name for the matched route.

    Inspects the resolved route's tags to determine which API provider handled
    the request, then returns the corresponding header name.
    Falls back to ``x-request-id`` (OpenAI convention) when no tag matches.

    Args:
        request: Incoming HTTP request.

    Returns:
        Header name.
    """
    return _lookup_by_route_tag(
        request, REQUEST_ID_HEADER_BY_TAG, _DEFAULT_REQUEST_ID_HEADER
    )


def set_response_headers(
    request: Request, response: Response, processing_ms: int
) -> None:
    """Attach provider-specific headers to the response.

    Dispatches to the handler registered for the matched route tag, if any.

    Args:
        request: Incoming HTTP request.
        response: Outgoing response object.
        processing_ms: Processing time in milliseconds.
    """
    if (route := request.scope.get("route")) is not None:
        for tag in getattr(route, "tags", None) or ():
            if (handler := SET_RESPONSE_HEADERS_BY_TAG.get(tag)) is not None:
                handler(request, response, processing_ms)
                return


def set_log_fields(request: Request, log: EventLog) -> None:
    """Attach provider-specific fields to the request log entry.

    Dispatches to the handler registered for the matched route tag, if any.

    Args:
        request: Incoming HTTP request.
        log: The event log dictionary.
    """
    if (route := request.scope.get("route")) is not None:
        for tag in getattr(route, "tags", None) or ():
            if (handler := SET_LOG_FIELDS_BY_TAG.get(tag)) is not None:
                handler(request, log)
                return
