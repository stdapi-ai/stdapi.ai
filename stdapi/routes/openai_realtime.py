"""OpenAI-compatible Realtime API implementation over a WebSocket."""

from typing import Annotated, NoReturn

from fastapi import APIRouter, Depends, Path, Query, Request, Response, WebSocket
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import TypeAdapter, ValidationError
from starlette.datastructures import Headers

from stdapi.api_errors import ApiError
from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate, enforce_tenant_endpoint_scope
from stdapi.aws_bedrock import (
    GUARDRAIL_CONFIG_VAR,
    set_guardrail_configuration,
    set_performance_configuration,
)
from stdapi.aws_bedrock_mantle import MANTLE_PROJECT_VAR, set_mantle_project
from stdapi.config import SETTINGS
from stdapi.models import is_model_wildcard
from stdapi.models.capabilities import Capability, register_route_capability
from stdapi.monitoring import (
    PRINCIPAL,
    TENANT,
    log_error_details,
    log_request_params,
    log_response_params,
)
from stdapi.realtime import (
    mint_client_secret,
    read_client_secret,
    serve_realtime_session,
)
from stdapi.tenant_keys import resume_tenant
from stdapi.types.openai_realtime import (
    DEFAULT_CLIENT_SECRET_TTL,
    ClientSecretCreateParams,
    ClientSecretCreateResponse,
    RealtimeSessionConfig,
    SessionConfig,
)

register_route_capability(
    "openai_realtime",
    f"{SETTINGS.openai_routes_prefix}/v1/realtime",
    "SPEECH",
    "SPEECH",
    Capability.REALTIME,
    mcp_tool=False,
)

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1/realtime", tags=["Realtime", TAG_OPENAI]
)


@router.post(
    "/client_secrets",
    summary="Mint an ephemeral client secret for a realtime session (OpenAI format)",
    operation_id="openai_realtime_client_secret",
    description=(
        "Creates a short-lived client secret carrying a session configuration "
        "(OpenAI Realtime Client Secrets API).\n\n"
        "Hand the returned `value` to an untrusted client -- a browser or a "
        "mobile application -- so it can open a realtime WebSocket without ever "
        "holding this deployment's API key. The secret carries the session "
        "configuration given here and applies it to every session opened with "
        "it, until it expires.\n\n"
        "Connect with it to "
        f"`wss://<host>{SETTINGS.openai_routes_prefix}/v1/realtime?model=<model>`, "
        "sending it as the `Authorization: Bearer` header.\n\n"
        "**Find compatible models:** Call `search_models` with "
        "`route=openai_realtime` to discover model IDs that support live "
        "speech-to-speech sessions."
    ),
    response_description="Returns the client secret, its expiry and its session",
    responses={
        200: {"description": "Client secret created."},
        400: {"description": "Invalid request or unsupported parameters."},
    },
)
async def create_realtime_client_secret(
    request: ClientSecretCreateParams | None = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> ClientSecretCreateResponse:
    """Mint an ephemeral client secret for a realtime session.

    Args:
        request: Optional lifetime and session configuration for the secret.

    Returns:
        The client secret, the moment it expires, and the session it opens.

    Raises:
        ApiError: When the session names a model pattern.
    """
    log_request_params(request)
    params = request or ClientSecretCreateParams()
    session: SessionConfig = params.session or RealtimeSessionConfig()
    if session.model is not None and is_model_wildcard(session.model):
        # The secret outlives the request that minted it, so its model is fixed
        # here: a pattern would be read again, later, and could mean another model.
        msg = (
            "A model pattern is not available on this endpoint. Name the model "
            "the secret opens a session with."
        )
        raise ApiError(msg)
    ttl = (
        params.expires_after.seconds
        if params.expires_after is not None
        else DEFAULT_CLIENT_SECRET_TTL
    )
    value, expires_at = mint_client_secret(session, ttl)
    return log_response_params(
        ClientSecretCreateResponse(value=value, expires_at=expires_at, session=session),
        exclude={"value"},
    )


@router.websocket("")
async def openai_realtime(
    websocket: WebSocket,
    model: Annotated[
        str | None, Query(description="Model serving the session.")
    ] = None,
    call_id: Annotated[
        str | None,
        Query(
            description=(
                "Identifier of a WebRTC call to attach to as its sideband, "
                "from the Location header of the call's creation. Requires "
                "API credentials -- never an ephemeral client secret -- and "
                "ignores 'model'."
            )
        ),
    ] = None,
) -> None:
    """Open a live speech-to-speech session (OpenAI Realtime API).

    Audio flows in both directions over the connection: send
    ``input_audio_buffer.append`` events carrying the caller's speech, and
    receive ``response.output_audio.delta`` events carrying the model's. With
    ``call_id``, the connection instead observes and controls an open WebRTC
    call.

    Args:
        websocket: The connection being opened.
        model: Model serving the session.
        call_id: WebRTC call to attach to as its sideband, if any.
    """
    await serve_realtime_session(websocket, model, call_id)


#: Validator of the ``session`` field of a multipart call creation.
_SESSION_ADAPTER: TypeAdapter[SessionConfig] = TypeAdapter(SessionConfig)

#: Largest SDP offer accepted, in bytes; a real offer is a few kilobytes.
_MAX_OFFER_BYTES = 64 * 1024

#: Empty headers, resetting per-request configuration an untrusted caller set.
_NO_HEADERS = Headers()

#: What the SIP-only call-control verbs answer, per verb.
_SIP_ONLY_VERBS: dict[str, str] = {
    "accept": "answers an incoming SIP call",
    "reject": "declines an incoming SIP call",
    "refer": "transfers a SIP call to another destination",
}


class _UnsupportedContentTypeError(ApiError):
    """The request body encoding is not one this endpoint reads."""

    code = "unsupported_content_type"


def _require_webrtc() -> None:
    """Refuse the request when WebRTC calls are not enabled.

    Raises:
        ApiError: 404 when the transport is not served, matching the response
            the endpoint gave before the feature existed.
    """
    if not SETTINGS.realtime_webrtc_enabled:
        log_error_details(
            "WebRTC calls are disabled: set realtime_webrtc_enabled -- with "
            "the 'webrtc' optional dependencies installed and a UDP media "
            "path to this server -- to serve them.",
            level="warning",
        )
        message = (
            "WebRTC calls are not available on this deployment. Connect over "
            "the WebSocket transport instead, or contact the administrator."
        )
        raise ApiError(message, status=404)


async def _authenticate_call(request: Request) -> tuple[SessionConfig | None, bool]:
    """Authenticate a call creation, honouring an ephemeral client secret.

    A browser holding an ephemeral secret posts its offer directly, as
    upstream's WebRTC flow does; any other credential goes through the
    standard verification. A secret-authenticated request must not keep the
    per-request header configuration the middleware read before this ran:
    those headers need the deployment's own key.

    Args:
        request: The creating request.

    Returns:
        The session configuration a secret carried, or None, and whether that
        configuration is locked against overrides.

    Raises:
        ApiError: 401 when the credential is missing or refused.
    """
    x_api_key = request.headers.get("x-api-key")
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    bearer = token if scheme.lower() == "bearer" and token else None
    credential = x_api_key or bearer
    secret = read_client_secret(credential) if credential else None
    if secret is None:
        # The standard dependency, called directly: a tenant key in x-api-key
        # then verifies alongside the Bearer identity, as on every HTTP route.
        await authenticate(
            HTTPAuthorizationCredentials(scheme="Bearer", credentials=bearer)
            if bearer
            else None,
            x_api_key,
            request,
        )
        return None, False
    PRINCIPAL.set(None)
    TENANT.set(None)
    # The setters keep a caller-set header value when no deployment default is
    # configured: reset the context vars first, so the untrusted secret holder
    # cannot smuggle the deployment-key-only headers past them. None reads as
    # unset everywhere the guardrail var is consumed.
    GUARDRAIL_CONFIG_VAR.set(None)  # type: ignore[arg-type]
    MANTLE_PROJECT_VAR.set("")
    set_guardrail_configuration(_NO_HEADERS)
    set_performance_configuration(_NO_HEADERS)
    set_mantle_project(_NO_HEADERS)
    if secret.tenant_key_id is not None:
        # The mint was tenant-authorized; the call keeps the tenant's scopes.
        TENANT.set(await resume_tenant(secret.tenant_key_id))
        enforce_tenant_endpoint_scope(request.scope)
    return secret.session, not SETTINGS.realtime_allow_session_override


def _check_offer_size(offer: str) -> None:
    """Refuse an SDP offer larger than any real one, before it is parsed.

    Args:
        offer: The offer, as the caller sent it.

    Raises:
        ApiError: 413 when the offer exceeds the size bound.
    """
    if len(offer) > _MAX_OFFER_BYTES:
        message = "The SDP offer is too large."
        raise ApiError(message, status=413)


async def _call_request_body(
    request: Request, fallback: SessionConfig | None, *, locked: bool
) -> tuple[str, SessionConfig]:
    """Read the SDP offer and session configuration off a call creation.

    Args:
        request: The creating request.
        fallback: Session configuration the credential carried, if any.
        locked: Whether that configuration may not be overridden.

    Returns:
        The offer and the session configuration the call opens with.

    Raises:
        ApiError: The encoding is not one this endpoint reads, the offer is
            missing, or the session configuration is not valid or is locked.
    """
    media_type = (
        request.headers.get("content-type", "").partition(";")[0].strip().lower()
    )
    if media_type == "application/sdp":
        offer = (await request.body()).decode(errors="replace")
        _check_offer_size(offer)
        return offer, fallback or RealtimeSessionConfig()
    if media_type != "multipart/form-data":
        message = (
            "Unsupported content type. This API method only accepts "
            "'application/sdp' or 'multipart/form-data' requests, but you "
            f"specified the header 'Content-Type: {media_type}'. Please try "
            "again with a supported content type."
        )
        raise _UnsupportedContentTypeError(message)
    async with request.form() as form:
        offer = value if isinstance(value := form.get("sdp"), str) else ""
        session_field = value if isinstance(value := form.get("session"), str) else None
    if not offer:
        message = "The 'sdp' form field must carry the SDP offer."
        raise ApiError(message)
    _check_offer_size(offer)
    if session_field is None:
        return offer, fallback or RealtimeSessionConfig()
    if locked:
        message = (
            "The session configuration this credential was issued for cannot "
            "be replaced. Post the offer without a 'session' field."
        )
        raise ApiError(message)
    try:
        return offer, _SESSION_ADAPTER.validate_json(session_field)
    except ValidationError:
        message = "The 'session' form field is not a valid session configuration."
        raise ApiError(message) from None


@router.post(
    "/calls",
    summary="Open a WebRTC realtime call with an SDP offer (OpenAI format)",
    operation_id="openai_realtime_call_create",
    description=(
        "Trades a WebRTC SDP offer for an SDP answer, opening a live "
        "speech-to-speech call terminated by this server (OpenAI Realtime "
        "Calls API).\n\n"
        "Send the offer as a raw `application/sdp` body -- with an ephemeral "
        "client secret or the deployment's credentials -- or as "
        "`multipart/form-data` with an `sdp` field and an optional `session` "
        "JSON field. Audio then flows as Opus media tracks and session events "
        "ride an `oai-events` data channel, in the same vocabulary as the "
        "WebSocket transport. The call's identifier is returned in the "
        "`Location` header; end the call with "
        f"`POST {SETTINGS.openai_routes_prefix}/v1/realtime/calls/"
        "{call_id}/hangup`.\n\n"
        "**Find compatible models:** Call `search_models` with "
        "`route=openai_realtime`."
    ),
    response_description="The SDP answer, with the call ID in Location",
    responses={
        201: {
            "description": "Call created; the body is the SDP answer.",
            "content": {"text/plain": {"schema": {"type": "string"}}},
        },
        400: {"description": "Invalid offer, encoding or configuration."},
        404: {"description": "WebRTC calls are not enabled."},
    },
    status_code=201,
)
async def create_realtime_call(
    request: Request,
    model: Annotated[
        str | None,
        Query(
            description=(
                "Model answering the call; the session configuration may "
                "name it instead."
            )
        ),
    ] = None,
) -> Response:
    """Open a WebRTC call from an SDP offer.

    Args:
        request: The creating request.
        model: Model named on the query string, if any.

    Returns:
        The SDP answer, with the call identifier in the Location header.

    Raises:
        ApiError: The feature is disabled, the credential was refused, or the
            offer could not be answered.
    """
    session, locked = await _authenticate_call(request)
    _require_webrtc()
    offer, config = await _call_request_body(request, session, locked=locked)
    log_request_params({"model": model or config.model})
    # Imported lazily: the webrtc extra is only present when the setting proved it.
    from stdapi.realtime_webrtc import open_call  # noqa: PLC0415

    call_id, answer = await open_call(request, model, config, offer, locked=locked)
    return Response(
        content=answer,
        status_code=201,
        media_type="text/plain",
        headers={
            "Location": f"{SETTINGS.openai_routes_prefix}/v1/realtime/calls/{call_id}"
        },
    )


@router.post(
    "/calls/{call_id}/hangup",
    summary="End a WebRTC realtime call (OpenAI format)",
    operation_id="openai_realtime_call_hangup",
    description=(
        "Ends an active WebRTC call opened by "
        f"`POST {SETTINGS.openai_routes_prefix}/v1/realtime/calls` "
        "(OpenAI Realtime Calls API). The call's media path and model "
        "session are torn down."
    ),
    response_description="Empty response once the hangup is under way",
    responses={
        200: {"description": "Call ended."},
        404: {"description": "No such call."},
    },
)
async def hangup_realtime_call(
    call_id: Annotated[
        str, Path(description="Call identifier, from the creation's Location header.")
    ],
    _: Annotated[None, Depends(authenticate)] = None,
) -> Response:
    """End one active call.

    Args:
        call_id: Identifier of the call to end.

    Returns:
        An empty 200 response.

    Raises:
        ApiError: The feature is disabled, or no call under that identifier
            is held by this instance.
    """
    _require_webrtc()
    # Imported lazily: the webrtc extra is only present when the setting proved it.
    from stdapi.realtime_webrtc import hangup_call  # noqa: PLC0415

    hangup_call(call_id)
    return Response(status_code=200)


def _refuse_sip_verb(verb: str) -> NoReturn:
    """Refuse a SIP-only call-control verb, naming what would serve it.

    Args:
        verb: The verb the caller used.

    Raises:
        ApiError: Always: SIP is not terminated by this server.
    """
    message = (
        f"SIP calls are not available on this server: '{verb}' "
        f"{_SIP_ONLY_VERBS[verb]}, and inbound SIP is not terminated here. "
        "For telephony, put a SIP-capable media framework (such as LiveKit "
        "Agents or Pipecat) in front of this deployment's WebSocket "
        "transport."
    )
    raise ApiError(message)


@router.post(
    "/calls/{call_id}/accept",
    summary="Accept an incoming SIP call (not available)",
    operation_id="openai_realtime_call_accept",
    description=(
        "Always refused: accepting an incoming SIP call requires a SIP trunk "
        "terminated by the server, which this deployment does not do. Use a "
        "SIP-capable media framework in front of the WebSocket transport."
    ),
    responses={400: {"description": "SIP calls are not available."}},
)
async def accept_realtime_call(
    call_id: Annotated[str, Path(description="Call identifier.")],  # noqa: ARG001 - upstream path shape
    _: Annotated[None, Depends(authenticate)] = None,
) -> Response:
    """Refuse the SIP-only accept verb.

    Args:
        call_id: Identifier of the call, never resolved.

    Returns:
        Never; the refusal always raises.

    Raises:
        ApiError: Always: SIP is not terminated by this server.
    """
    return _refuse_sip_verb("accept")


@router.post(
    "/calls/{call_id}/reject",
    summary="Reject an incoming SIP call (not available)",
    operation_id="openai_realtime_call_reject",
    description=(
        "Always refused: rejecting an incoming SIP call requires a SIP trunk "
        "terminated by the server, which this deployment does not do. Use a "
        "SIP-capable media framework in front of the WebSocket transport."
    ),
    responses={400: {"description": "SIP calls are not available."}},
)
async def reject_realtime_call(
    call_id: Annotated[str, Path(description="Call identifier.")],  # noqa: ARG001 - upstream path shape
    _: Annotated[None, Depends(authenticate)] = None,
) -> Response:
    """Refuse the SIP-only reject verb.

    Args:
        call_id: Identifier of the call, never resolved.

    Returns:
        Never; the refusal always raises.

    Raises:
        ApiError: Always: SIP is not terminated by this server.
    """
    return _refuse_sip_verb("reject")


@router.post(
    "/calls/{call_id}/refer",
    summary="Transfer an incoming SIP call (not available)",
    operation_id="openai_realtime_call_refer",
    description=(
        "Always refused: transferring a SIP call requires a SIP trunk "
        "terminated by the server, which this deployment does not do. Use a "
        "SIP-capable media framework in front of the WebSocket transport."
    ),
    responses={400: {"description": "SIP calls are not available."}},
)
async def refer_realtime_call(
    call_id: Annotated[str, Path(description="Call identifier.")],  # noqa: ARG001 - upstream path shape
    _: Annotated[None, Depends(authenticate)] = None,
) -> Response:
    """Refuse the SIP-only refer verb.

    Args:
        call_id: Identifier of the call, never resolved.

    Returns:
        Never; the refusal always raises.

    Raises:
        ApiError: Always: SIP is not terminated by this server.
    """
    return _refuse_sip_verb("refer")
