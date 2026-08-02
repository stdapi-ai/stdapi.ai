"""OpenAI-compatible Text-to-Speech API implementation using AWS Polly."""

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse
from sse_starlette import EventSourceResponse, JSONServerSentEvent

from stdapi.api_errors import ApiError
from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.aws_bedrock import apply_guardrail_to_text, get_extra_model_parameters
from stdapi.config import SETTINGS
from stdapi.mcp import is_mcp
from stdapi.models import validate_model
from stdapi.models.audio import get_audio_model
from stdapi.models.capabilities import Capability, register_route_capability
from stdapi.monitoring import (
    log_request_params,
    log_request_stream_event,
    log_response_params,
)
from stdapi.types.openai_audio import (
    SpeechAudioDeltaEvent,
    SpeechAudioDoneEvent,
    SpeechCreateParams,
    SpeechUsage,
)
from stdapi.utils import b64encode

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

register_route_capability(
    "openai_audio_speech",
    f"{SETTINGS.openai_routes_prefix}/v1/audio/speech",
    "TEXT",
    "SPEECH",
    Capability.TTS,
)

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1/audio", tags=["Audio", TAG_OPENAI]
)

#: Content-type format name if different from the response format name
_FORMAT_CONTENT_TYPE = {"mp3": "mpeg"}


async def _speech_audio_bytestream(
    stream: AsyncGenerator[bytes],
) -> AsyncGenerator[bytes]:
    """Generate real-time audio streaming, with logging.

    Args:
        stream: Audio stream yielding audio bytes chunks

    Yields:
        Audio stream yielding audio bytes chunks
    """
    try:
        async for chunk in stream:
            yield chunk
    finally:
        await stream.aclose()


async def _speech_audio_sse(
    stream: AsyncGenerator[bytes], input_tokens: int, output_tokens: int
) -> AsyncGenerator[JSONServerSentEvent]:
    """Generate SSE events for real-time audio streaming.

    Args:
        stream: Audio stream yielding bytes chunks.
        input_tokens: Input token count for usage tracking.
        output_tokens: Output token count for usage tracking.

    Yields:
        JSONServerSentEvent with speech.audio.delta and speech.audio.done events.
    """
    try:
        async for chunk in stream:
            yield JSONServerSentEvent(
                data=SpeechAudioDeltaEvent(audio=await b64encode(chunk)).model_dump(
                    mode="json", exclude_none=True
                )
            )
    finally:
        await stream.aclose()
        yield JSONServerSentEvent(
            data=log_response_params(
                SpeechAudioDoneEvent(
                    usage=SpeechUsage(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    )
                ).model_dump(mode="json", exclude_none=True)
            )
        )


@router.post(
    "/speech",
    summary="Convert text to speech audio (OpenAI format)",
    operation_id="openai_audio_speech",
    description=(
        "Generates audio from the input text (OpenAI Audio Speech API).\n\n"
        "Returns the audio file as a streaming download in the requested format, "
        "or a stream of SSE audio events when `stream_format=sse`.\n\n"
        "Provider-specific parameters may return a non-audio payload instead: "
        "Amazon Polly `SpeechMarkTypes` returns timing marks as an "
        "`application/x-json-stream` JSON lines stream, which ignores "
        "`response_format` and does not support `stream_format=sse`.\n\n"
        "**Find compatible models:** Call `search_models` with `route=openai_audio_speech` "
        "to discover model IDs that support text-to-speech."
    ),
    response_description="Returns audio file in the specified format",
    responses={
        200: {"description": "Audio generated (or streaming)."},
        400: {"description": "Invalid request or unsupported parameters."},
        404: {"description": "Model not found."},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "examples": {
                        "file": {
                            "summary": "Generate MP3 file",
                            "value": {
                                "model": "amazon.polly-standard",
                                "voice": "Amy",
                                "input": "Hello, I'am Amy",
                                "response_format": "mp3",
                            },
                        },
                        "sse": {
                            "summary": "Stream using SSE",
                            "value": {
                                "model": "amazon.polly-standard",
                                "voice": "Amy",
                                "input": "Hello, I'am Amy",
                                "response_format": "mp3",
                                "stream_format": "sse",
                            },
                        },
                    }
                }
            }
        }
    },
)
async def create_speech(
    request: SpeechCreateParams, _: Annotated[None, Depends(authenticate)] = None
) -> Response:
    """Generates audio from the input text.

    Converts input text to audio using advanced text-to-speech models. Supports
    multiple voices, output formats, and playback speeds. Provides both standard
    audio file responses and real-time streaming capabilities.

    When called via MCP, defaults to SSE streaming format for better compatibility
    with MCP clients.

    Args:
        request: The text-to-speech request containing model, voice, and format parameters.

    Returns:
        Response: Audio file in the specified format or streaming response.

    Raises:
        ApiError: When audio generation fails, validation errors occur, or
            unsupported voice/model combinations are provided.
    """
    log_request_params(request)
    model_id = (
        await validate_model(
            request.model,
            input_modality="TEXT",
            output_modality="SPEECH",
            bedrock_only=False,
        )
    ).id

    tts_response = await get_audio_model(model_id).tts(
        text=await apply_guardrail_to_text(request.input, source="INPUT"),
        voice=request.voice,
        resp_format=request.response_format,
        speed=request.speed,
        extra_params=get_extra_model_parameters(model_id, request),
    )

    if content_type := tts_response.get("content_type"):
        # Non-audio payloads (Polly speech marks) cannot be framed as OpenAI
        # audio SSE deltas: stream them as-is with the backend content type.
        if request.stream_format == "sse":
            await tts_response["audio_stream"].aclose()
            msg = "'stream_format' 'sse' is not supported with a non-audio output such as speech marks."
            raise ApiError(msg)
        return StreamingResponse(
            content=_speech_audio_bytestream(
                await log_request_stream_event(tts_response["audio_stream"])
            ),
            media_type=content_type,
        )

    audio_stream = await log_request_stream_event(tts_response["audio_stream"])
    if request.stream_format == "sse" or (
        is_mcp() and "stream_format" not in request.model_fields_set
    ):
        return EventSourceResponse(
            _speech_audio_sse(
                audio_stream,
                tts_response["input_tokens"],
                tts_response["output_tokens"],
            )
        )

    return StreamingResponse(
        content=_speech_audio_bytestream(audio_stream),
        media_type=f"audio/{_FORMAT_CONTENT_TYPE.get(fmt := request.response_format, fmt)}",
        headers={"Content-Disposition": f"attachment; filename=speech.{fmt}"},
    )
