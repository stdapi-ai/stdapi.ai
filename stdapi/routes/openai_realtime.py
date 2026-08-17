"""OpenAI-compatible Realtime API implementation over a WebSocket."""

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Query, WebSocket

from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.models.capabilities import Capability, register_route_capability
from stdapi.monitoring import log_request_params, log_response_params
from stdapi.realtime import mint_client_secret, serve_realtime_session
from stdapi.types.openai_realtime import (
    DEFAULT_CLIENT_SECRET_TTL,
    ClientSecretCreateParams,
    ClientSecretCreateResponse,
    RealtimeSessionConfig,
)

if TYPE_CHECKING:
    from stdapi.types.openai_realtime import SessionConfig

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
    """
    log_request_params(request)
    params = request or ClientSecretCreateParams()
    session: SessionConfig = params.session or RealtimeSessionConfig()
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
) -> None:
    """Open a live speech-to-speech session (OpenAI Realtime API).

    Audio flows in both directions over the connection: send
    ``input_audio_buffer.append`` events carrying the caller's speech, and
    receive ``response.output_audio.delta`` events carrying the model's.

    Args:
        websocket: The connection being opened.
        model: Model serving the session.
    """
    await serve_realtime_session(websocket, model)
