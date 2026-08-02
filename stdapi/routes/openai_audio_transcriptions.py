"""OpenAI-compatible Audio Transcription API implementation."""

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile
from sse_starlette import EventSourceResponse, JSONServerSentEvent

from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.aws_bedrock import get_extra_model_parameters
from stdapi.config import SETTINGS
from stdapi.input_file import InputFile
from stdapi.models import validate_model
from stdapi.models.audio import get_audio_model
from stdapi.models.audio.amazon_transcribe import AWS_TRANSCRIBE_MODEL_ID
from stdapi.models.capabilities import Capability, register_route_capability
from stdapi.monitoring import log_request_params, log_request_stream_event
from stdapi.types.openai_audio import (
    AudioResponseFormat,
    AudioTranscriptionJsonBody,
    ChunkingStrategy,
    TranscriptionCreateParams,
    TranscriptionCreateResponse,
    TranscriptionDiarized,
    TranscriptionInclude,
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
)
from stdapi.utils import missing_file_error, validation_error_handler

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

register_route_capability(
    "openai_audio_transcription",
    f"{SETTINGS.openai_routes_prefix}/v1/audio/transcriptions",
    "SPEECH",
    "TEXT",
    Capability.STT,
)

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1/audio", tags=["Audio", TAG_OPENAI]
)


def _merge_form_list[T: str](
    bare: list[T] | None, bracketed: list[T] | None
) -> list[T] | None:
    """Merge a bare-named multipart form list with its ``[]``-suffixed counterpart.

    The official OpenAI SDKs send list-valued form fields under a ``name[]``
    key; this gateway also accepts the bare ``name`` key for non-SDK clients.

    Args:
        bare: Values collected under the plain field name.
        bracketed: Values collected under the ``name[]`` alias.

    Returns:
        The combined list in bare-then-bracketed order, or None if both are empty.
    """
    merged = [*(bare or []), *(bracketed or [])]
    return merged or None


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
    summary="Transcribe audio to text (OpenAI format)",
    operation_id="openai_audio_transcription",
    description=(
        "Transcribes audio into the input language (OpenAI Audio Transcriptions API).\n\n"
        "**Providing the audio file:** Two request formats are accepted:\n"
        "- `multipart/form-data`: standard binary file upload via the `file` field.\n"
        "- `application/json`: pass `file` as a base64 string, data URI "
        "(`data:audio/<fmt>;base64,<data>`), HTTPS URL, or S3 URI — preferred for **MCP tools** and "
        "AI agents that cannot construct multipart requests.\n\n"
        "**MCP / AI agent usage:** Call this tool with a JSON body containing the audio as a "
        "data URI or URL, along with `model` and any other parameters. Example: "
        '`{"file": "data:audio/mp3;base64,<data>", "model": "amazon.transcribe", "response_format": "json"}`.\n\n'
        "Returns the transcription as plain text, JSON with metadata, subtitle formats (SRT/VTT), "
        "or a stream of SSE events.\n\n"
        "**Extended output format (beyond original OpenAI API):**\n"
        "- **`diarized_json`**: Speaker diarization — returns labelled segments identifying "
        "which speaker said what, with timestamps.\n\n"
        "**Find compatible models:** Call `search_models` with `mcp_tool=openai_audio_transcription` "
        "to discover model IDs that support speech-to-text."
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
                },
                "application/json": {
                    "examples": {
                        "data_uri": {
                            "summary": "Audio via data URI (MCP / AI agent)",
                            "value": {
                                "file": "data:audio/mp3;base64,<base64-encoded-audio>",
                                "model": "amazon.transcribe",
                                "response_format": "json",
                            },
                        },
                        "url": {
                            "summary": "Audio from URL",
                            "value": {
                                "file": "https://example.com/audio.mp3",
                                "model": "amazon.transcribe",
                            },
                        },
                    }
                },
            }
        }
    },
    response_model_exclude_none=True,
)
async def create_transcription(
    http_request: Request,
    file: Annotated[
        UploadFile | None,
        File(
            description=(
                "The audio file to transcribe, in one of these formats: "
                "flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, or webm. "
                "Use an ``application/json`` body to pass a base64 string, data URI, or URL instead."
            )
        ),
    ] = None,
    *,
    model: Annotated[
        str,
        Form(
            description=(
                "The transcription model to use.\n"
                "`amazon.transcribe` or a speech-to-text model (e.g. Mistral Voxtral)."
            )
        ),
    ] = AWS_TRANSCRIBE_MODEL_ID,
    language: Annotated[
        str | None,
        Form(
            description=(
                "The language of the input audio.\n"
                "Supplying it in ISO-639-1 format (e.g. `en`) improves accuracy and latency."
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
                "`auto` normalizes loudness then uses voice activity detection (VAD) to choose "
                "boundaries; a `server_vad` object tunes VAD parameters manually.\n"
                "server_vad is UNSUPPORTED on this implementation."
            )
        ),
    ] = "auto",
    response_format: Annotated[
        AudioResponseFormat, Form(description="Transcript output format.")
    ] = "json",
    timestamp_granularities: Annotated[
        str,
        Form(
            description=(
                "Comma-separated timestamp granularities to populate (e.g. "
                "`word,segment`): `word` and/or `segment`. Requires "
                "`response_format=verbose_json`.\n"
                "Repeated `timestamp_granularities[]` form fields (the official SDKs' "
                "wire format) are also accepted and merged with this field."
            )
        ),
    ] = "",
    timestamp_granularities_bracket: Annotated[
        list[str] | None,
        Form(
            alias="timestamp_granularities[]",
            description="Same as `timestamp_granularities`, as repeated form fields "
            "(official SDKs' wire format).",
        ),
    ] = None,
    include: Annotated[
        list[TranscriptionInclude] | None,
        Form(
            description=(
                "Additional information to include in the transcription response.\n"
                "`logprobs` returns token log probabilities (confidence) and only works "
                "with `response_format=json`.\n"
                "Repeated `include[]` form fields (the official SDKs' wire format) are "
                "also accepted and merged with this field."
            )
        ),
    ] = None,
    include_bracket: Annotated[
        list[TranscriptionInclude] | None,
        Form(
            alias="include[]",
            description="Same as `include`, as repeated form fields (official SDKs' "
            "wire format).",
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
                "Speaker names corresponding to the samples in `known_speaker_references[]` "
                "(e.g. `customer`, `agent`).\n"
                "Accepted but ignored: diarization degrades to generic speaker labels."
            )
        ),
    ] = None,
    known_speaker_names_bracket: Annotated[
        list[str] | None,
        Form(
            alias="known_speaker_names[]",
            description="Same as `known_speaker_names`, as repeated form fields "
            "(official SDKs' wire format). Accepted but ignored: diarization "
            "degrades to generic speaker labels.",
        ),
    ] = None,
    known_speaker_references: Annotated[
        list[str] | None,
        Form(
            description=(
                "Audio samples (as data URLs, 2-10 seconds each, same formats as `file`) "
                "for known-speaker diarization, matching `known_speaker_names[]`.\n"
                "Accepted but ignored: diarization degrades to generic speaker labels."
            )
        ),
    ] = None,
    known_speaker_references_bracket: Annotated[
        list[str] | None,
        Form(
            alias="known_speaker_references[]",
            description="Same as `known_speaker_references`, as repeated form fields "
            "(official SDKs' wire format). Accepted but ignored: diarization "
            "degrades to generic speaker labels.",
        ),
    ] = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> (
    str
    | TranscriptionCreateResponse
    | TranscriptionDiarized
    | EventSourceResponse
    | Response
):
    """Transcribes audio into the input language.

    Accepts ``multipart/form-data`` (binary upload) or ``application/json``
    (base64, data URI, HTTPS URL, or S3 URI in the ``file`` field).

    Args:
        http_request: FastAPI request object used to detect content-type.
        file: The audio file to transcribe (multipart only).
        model: The transcription model to use: ``amazon.transcribe`` or a
            speech-to-text model (e.g. Mistral Voxtral).
        language: The language of the input audio (ISO-639-1 code, e.g. `en`). Improves accuracy and latency when provided.
        prompt: Optional style guidance for the model. Supported by Bedrock
            models (e.g. Mistral Voxtral); rejected by ``amazon.transcribe``.
        chunking_strategy: Controls how the audio is cut into chunks. `auto` only is supported on this implementation.
        response_format: Output format: `json`, `text`, `srt`, `verbose_json`, `vtt`, or `diarized_json`.
        timestamp_granularities: For `verbose_json` only; comma-separated values among `word` and `segment` (e.g. `word,segment`).
        timestamp_granularities_bracket: Same values as `timestamp_granularities`, as repeated `timestamp_granularities[]` form fields.
        include: Additional information to include in the transcription response. `logprobs` only works with response_format set to `json`.
        include_bracket: Same values as `include`, as repeated `include[]` form fields.
        temperature: Sampling temperature. Supported by Bedrock models
            (e.g. Mistral Voxtral); rejected by ``amazon.transcribe``.
        stream: Whether to stream partial results via Server-Sent Events.
        known_speaker_names: Optional list of known speaker names. Accepted but ignored: diarization degrades to generic speaker labels.
        known_speaker_names_bracket: Same values as `known_speaker_names`, as repeated `known_speaker_names[]` form fields.
        known_speaker_references: Optional list of audio references for known speakers. Accepted but ignored: diarization degrades to generic speaker labels.
        known_speaker_references_bracket: Same values as `known_speaker_references`, as repeated `known_speaker_references[]` form fields.

    Returns:
        The transcribed text in the requested format.

    Raises:
        ApiError: When transcription fails or invalid parameters are provided.
    """
    request: TranscriptionCreateParams
    if "application/json" in http_request.headers.get("content-type", ""):
        with validation_error_handler():
            request = AudioTranscriptionJsonBody.model_validate(
                await http_request.json()
            )
        audio_content = request.file
    elif file is None:
        missing_file_error()
    else:
        audio_content = InputFile(file)
        granularities: list[str] = [
            *(timestamp_granularities.split(",") if timestamp_granularities else []),
            *(timestamp_granularities_bracket or []),
        ]
        with validation_error_handler():
            request = TranscriptionCreateParams(
                model=model,
                language=language,
                prompt=prompt,
                chunking_strategy=chunking_strategy,
                response_format=response_format,
                timestamp_granularities=granularities,  # type: ignore[arg-type]
                include=_merge_form_list(include, include_bracket),
                temperature=temperature,
                stream=stream,
                known_speaker_names=_merge_form_list(
                    known_speaker_names, known_speaker_names_bracket
                ),
                known_speaker_references=_merge_form_list(
                    known_speaker_references, known_speaker_references_bracket
                ),
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

    extra_params = get_extra_model_parameters(model, request)

    if request.stream:
        return EventSourceResponse(
            await log_request_stream_event(
                _transcript_audio_sse(
                    get_audio_model(model).stt_stream(
                        audio_content=audio_content,
                        response_format=request.response_format,
                        language=request.language,
                        temperature=request.temperature,
                        prompt=request.prompt,
                        extra_params=extra_params,
                        logprobs="logprobs" in (request.include or []),
                    )
                )
            )
        )

    return await get_audio_model(model).stt(
        audio_content=audio_content,
        response_format=request.response_format,
        language=request.language,
        timestamp_granularities=request.timestamp_granularities,
        temperature=request.temperature,
        prompt=request.prompt,
        extra_params=extra_params,
        logprobs="logprobs" in (request.include or []),
    )
