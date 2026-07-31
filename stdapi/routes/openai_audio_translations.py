"""OpenAI-compatible Audio Translation API implementation."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Request, Response, UploadFile

from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.aws_bedrock import get_extra_model_parameters
from stdapi.config import SETTINGS
from stdapi.input_file import InputFile
from stdapi.models import validate_model
from stdapi.models.audio import get_audio_model
from stdapi.models.audio.amazon_transcribe import AWS_TRANSCRIBE_MODEL_ID
from stdapi.models.capabilities import Capability, register_route_capability
from stdapi.monitoring import log_request_params
from stdapi.types.openai_audio import (
    AudioTranslationJsonBody,
    TranslateAudioResponseFormat,
    TranslationCreateParams,
    TranslationCreateResponse,
)
from stdapi.utils import missing_file_error, validation_error_handler

register_route_capability(
    "openai_audio_translation",
    f"{SETTINGS.openai_routes_prefix}/v1/audio/translations",
    "SPEECH",
    "TEXT",
    Capability.STT_TRANSLATE,
)

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1/audio", tags=["Audio", TAG_OPENAI]
)


@router.post(
    "/translations",
    response_model=None,
    summary="Transcribe and translate audio to English text (OpenAI format)",
    operation_id="openai_audio_translation",
    description=(
        "Transcribes audio from any supported language and translates the result into English "
        "(OpenAI Audio Translations API).\n\n"
        "**Output is always English** — regardless of the source language. "
        "If you want the transcription in the original language, use `openai_audio_transcription` instead.\n\n"
        "**Providing the audio file:** Two request formats are accepted:\n"
        "- `multipart/form-data`: standard binary file upload via the `file` field.\n"
        "- `application/json`: pass `file` as a base64 string, data URI "
        "(`data:audio/<fmt>;base64,<data>`), HTTPS URL, or S3 URI — preferred for **MCP tools** and "
        "AI agents that cannot construct multipart requests.\n\n"
        "**MCP / AI agent usage:** Call this tool with a JSON body containing the audio as a "
        "data URI or URL, along with `model`. Example: "
        '`{"file": "data:audio/mp3;base64,<data>", "model": "amazon.transcribe"}`.\n\n'
        "**Find compatible models:** Call `search_models` with `mcp_tool=openai_audio_translation` "
        "to discover model IDs that support speech-to-text translation."
    ),
    response_description="Returns translation in the specified format",
    responses={
        200: {"description": "Translation completed."},
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
                        "srt": {
                            "summary": "Subtitle (SRT)",
                            "value": {"response_format": "srt"},
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
async def create_translation(
    http_request: Request,
    file: Annotated[
        UploadFile | None,
        File(
            description=(
                "The audio file to translate, in one of these formats: "
                "flac, m4a, mp3, mp4, mpeg, mpga, oga, ogg, wav, webm. "
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
    prompt: Annotated[
        str | None,
        Form(
            description=(
                "An optional text to guide the model's style or continue a previous audio segment.\n"
                "The prompt should be in English.\n"
                "Supported by speech-to-text models such as Mistral Voxtral; rejected by `amazon.transcribe`."
            )
        ),
    ] = None,
    response_format: Annotated[
        TranslateAudioResponseFormat, Form(description="Transcript output format.")
    ] = "json",
    temperature: Annotated[
        float | None,
        Form(
            description=(
                "The sampling temperature, between `0` and `1`.\n"
                "Higher values like `0.8` will make the output more random, while lower values like `0.2` will make it more focused and deterministic."
            )
        ),
    ] = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> str | TranslationCreateResponse | Response:
    """Translates audio into English.

    Accepts ``multipart/form-data`` (binary upload) or ``application/json``
    (base64, data URI, HTTPS URL, or S3 URI in the ``file`` field).

    Args:
        http_request: FastAPI request object used to detect content-type.
        file: The audio file to translate (multipart only).
        model: The transcription model to use: ``amazon.transcribe`` or a
            speech-to-text model (e.g. Mistral Voxtral).
        prompt: Optional style guidance for the model. Supported by
            speech-to-text models such as Mistral Voxtral; rejected by ``amazon.transcribe``.
        response_format: Output format: `json`, `text`, `srt`, `verbose_json`, or `vtt`.
        temperature: Sampling temperature. Supported by speech-to-text models
            such as Mistral Voxtral; rejected by ``amazon.transcribe``.

    Returns:
        The translated text in English in the requested format.

    Raises:
        ApiError: When translation fails or invalid parameters are provided.
    """
    request: TranslationCreateParams
    if "application/json" in http_request.headers.get("content-type", ""):
        with validation_error_handler():
            request = AudioTranslationJsonBody.model_validate(await http_request.json())
        audio_content = request.file
    elif file is None:
        missing_file_error()
    else:
        audio_content = InputFile(file)
        with validation_error_handler():
            request = TranslationCreateParams(
                model=model,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
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

    return await get_audio_model(model).stt_translate(
        audio_content=audio_content,
        response_format=request.response_format,
        temperature=request.temperature,
        prompt=request.prompt,
        extra_params=get_extra_model_parameters(model, request),
    )
