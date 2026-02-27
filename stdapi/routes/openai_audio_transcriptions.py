"""OpenAI-compatible Audio Transcription API implementation."""

from typing import TYPE_CHECKING, Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    Response,
    UploadFile,
)
from sse_starlette import EventSourceResponse, JSONServerSentEvent

from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.models import validate_model
from stdapi.models.audio import get_audio_model
from stdapi.models.audio.amazon_transcribe import AWS_TRANSCRIBE_MODEL_ID
from stdapi.monitoring import log_request_params, log_request_stream_event
from stdapi.types.openai_audio import (
    AudioResponseFormat,
    ChunkingStrategy,
    TranscriptionCreateParams,
    TranscriptionCreateResponse,
    TranscriptionDiarized,
    TranscriptionInclude,
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
)
from stdapi.utils import validation_error_handler

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator


router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1/audio", tags=["Audio", TAG_OPENAI]
)


async def _transcript_audio_sse(
    event_stream: AsyncGenerator[
        TranscriptionTextDeltaEvent | TranscriptionTextDoneEvent
    ],
) -> AsyncGenerator[JSONServerSentEvent]:
    """Generate Server-Sent Events for real-time audio transcription streaming.

    Args:
        event_stream: Generator yielding TranscriptionTextDeltaEvent or TranscriptionTextDoneEvent objects

    Yields:
        JSONServerSentEvent: SSE events with transcript.text.delta and transcript.text.done events
    """
    async for event in event_stream:
        yield JSONServerSentEvent(data=event.model_dump(mode="json", exclude_none=True))


@router.post(
    "/transcriptions",
    response_model=None,
    summary="OpenAI - /v1/audio/transcriptions",
    description=(
        "Transcribes audio into the input language.\n\n"
        "Returns a transcription object in json, diarized_json, or verbose_json format, or a stream of transcript events."
    ),
    response_description="Returns transcription in the specified format",
    responses={
        200: {"description": "Transcription completed (or streaming)."},
        400: {"description": "Invalid request or unsupported parameters."},
        404: {"description": "Model not found."},
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "examples": {
                        "json": {
                            "summary": "JSON response",
                            "value": {
                                "model": "amazon.transcribe",
                                "response_format": "json",
                            },
                        },
                        "vtt": {
                            "summary": "Subtitle (VTT)",
                            "value": {"response_format": "vtt"},
                        },
                        "stream": {
                            "summary": "Streaming SSE",
                            "value": {"stream": True},
                        },
                    }
                }
            }
        }
    },
    response_model_exclude_none=True,
)
async def create_transcription(
    file: Annotated[
        UploadFile,
        File(
            ...,
            description="The audio file object (not file name) to transcribe, in one of these formats: flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm.",
        ),
    ],
    *,
    model: Annotated[
        str,
        Form(
            description=(
                "The transcription model to use.\nAvailable models: amazon.transcribe"
            )
        ),
    ] = AWS_TRANSCRIBE_MODEL_ID,
    language: Annotated[
        str | None,
        Form(
            description=(
                "The language of the input audio.\n"
                "Supplying the input language in ISO-639-1 (e.g. `en`) format will improve accuracy and latency."
            )
        ),
    ] = None,
    prompt: Annotated[
        str | None,
        Form(
            description=(
                "An optional text to guide the model's style or continue a previous audio segment.\n"
                "The prompt should match the audio language."
            )
        ),
    ] = None,
    chunking_strategy: Annotated[
        ChunkingStrategy,
        Form(
            description=(
                "Controls how the audio is cut into chunks.\n"
                "When set to `auto`, the server first normalizes loudness and then uses voice activity detection (VAD) to choose boundaries. "
                "`server_vad` object can be provided to tweak VAD detection parameters manually. "
                "If unset, the audio is transcribed as a single block.\nUNSUPPORTED on this implementation."
            )
        ),
    ] = "auto",
    response_format: Annotated[
        AudioResponseFormat,
        Form(
            description=(
                "The format of the transcript output.\n"
                "Supported formats: `json`, `text`, `srt`, `verbose_json`, `vtt`"
            )
        ),
    ] = "json",
    timestamp_granularities: Annotated[
        str,
        Form(
            description=(
                "Comma-separated list of timestamp granularities to populate for this transcription (e.g. `word,segment`).\n"
                "`response_format` must be set to `verbose_json` to use timestamp granularities.\n"
                "Either or both of these options are supported: `word`, or `segment`."
            )
        ),
    ] = "",
    include: Annotated[
        TranscriptionInclude | None,
        Form(
            description=(
                "Additional information to include in the transcription response.\n"
                "`logprobs` will return the log probabilities of the tokens in the response to understand the model's confidence in the transcription. "
                "`logprobs` only works with response_format set to `json`."
            )
        ),
    ] = None,
    temperature: Annotated[
        float | None,
        Form(
            description=(
                "The sampling temperature, between `0` and `1`.\n"
                "Higher values like `0.8` will make the output more random, while lower values like `0.2` will make it more focused and deterministic."
            )
        ),
    ] = None,
    stream: Annotated[
        bool | None,
        Form(
            description=(
                "If set to true, the model response data will be streamed to the client as it is generated using "
                "server-sent events."
            )
        ),
    ] = False,
    known_speaker_names: Annotated[
        list[str] | None,
        Form(
            description=(
                "Optional list of speaker names that correspond to the audio samples provided in "
                "`known_speaker_references[]`. Each entry should be a short identifier (for "
                "example `customer` or `agent`).\n"
                "UNSUPPORTED on this implementation."
            )
        ),
    ] = None,
    known_speaker_references: Annotated[
        list[str] | None,
        Form(
            description=(
                "Optional list of audio samples (as "
                "[data URLs](https://developer.mozilla.org/en-US/docs/Web/HTTP/Basics_of_HTTP/Data_URLs)) "
                "that contain known speaker references matching `known_speaker_names[]`. Each "
                "sample must be between 2 and 10 seconds, and can use any of the same input audio "
                "formats supported by `file`.\n"
                "UNSUPPORTED on this implementation."
            )
        ),
    ] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    _: Annotated[None, Depends(authenticate)] = None,
) -> (
    str
    | TranscriptionCreateResponse
    | TranscriptionDiarized
    | EventSourceResponse
    | Response
):
    """Transcribes audio into the input language.

    Converts audio to text in the same language as the source audio. The model will
    use the specified language or automatically detect the language if not provided.
    Supports multiple output formats including plain text, JSON with metadata, and
    subtitle formats for video content.

    Args:
        file: The audio file to transcribe.
        model: The transcription model to use. Available models: amazon.transcribe.
        language: The language of the input audio (ISO-639-1 code, e.g. `en`). Improves accuracy and latency when provided.
        prompt: Optional style guidance for the model. UNSUPPORTED on this implementation.
        chunking_strategy: Controls how the audio is cut into chunks. `auto` only is supported on this implementation.
        response_format: Output format: `json`, `text`, `srt`, `verbose_json`, `vtt`, or `diarized_json`.
        timestamp_granularities: For `verbose_json` only; comma-separated values among `word` and `segment` (e.g. `word,segment`).
        include: Additional information to include in the transcription response. `logprobs` only works with response_format set to `json`.
        temperature: Sampling temperature. UNSUPPORTED on this implementation (must be 0.0).
        stream: Whether to stream partial results via Server-Sent Events.
        known_speaker_names: Optional list of known speaker names. UNSUPPORTED on this implementation.
        known_speaker_references: Optional list of audio references for known speakers. UNSUPPORTED on this implementation.
        background_tasks: FastAPI background tasks for cleanup.

    Returns:
        The transcribed text in the requested format.

    Raises:
        ApiError: When transcription fails or invalid parameters are provided.
    """
    with validation_error_handler():
        request = TranscriptionCreateParams(
            model=model,
            language=language,
            prompt=prompt,
            chunking_strategy=chunking_strategy,
            response_format=response_format,
            timestamp_granularities=(
                timestamp_granularities.split(",") if timestamp_granularities else []  # type: ignore[arg-type]
            ),
            include=include,
            temperature=temperature,
            stream=stream,
            known_speaker_names=known_speaker_names,
            known_speaker_references=known_speaker_references,
        )
    log_request_params(request)

    model = (
        await validate_model(
            request.model,
            input_modality="SPEECH",
            output_modality="TEXT",
            bedrock_only=False,
        )
    ).id

    if request.stream:
        return EventSourceResponse(
            await log_request_stream_event(
                _transcript_audio_sse(
                    get_audio_model(model).stt_stream(
                        audio_content=file,
                        background_tasks=background_tasks,
                        response_format=request.response_format,
                        language=request.language,
                        temperature=request.temperature,
                        prompt=request.prompt,
                        logprobs=request.include == "logprobs",
                    )
                )
            )
        )

    return await get_audio_model(model).stt(
        audio_content=file,
        background_tasks=background_tasks,
        response_format=request.response_format,
        language=request.language,
        timestamp_granularities=request.timestamp_granularities,
        temperature=request.temperature,
        prompt=request.prompt,
        logprobs=request.include == "logprobs",
    )
