"""OpenAI-compatible Audio Translation API implementation."""

from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile

from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.config import SETTINGS
from stdapi.input_file import InputFile
from stdapi.models import validate_model
from stdapi.models.audio import get_audio_model
from stdapi.models.audio.amazon_transcribe import AWS_TRANSCRIBE_MODEL_ID
from stdapi.models.capabilities import Capability, register_route_capability
from stdapi.monitoring import log_request_params
from stdapi.types.openai_audio import (
    TranslateAudioResponseFormat,
    TranslationCreateParams,
    TranslationCreateResponse,
)
from stdapi.utils import validation_error_handler

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
    summary="OpenAI - /v1/audio/translations",
    operation_id="openai_audio_translation",
    description="Translates audio into English.",
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
                }
            }
        }
    },
    response_model_exclude_none=True,
)
async def create_translation(
    file: Annotated[
        UploadFile,
        File(
            ...,
            description=(
                "The audio file object (not file name) to translate, in one of these formats: "
                "flac, m4a, mp3, mp4, mpeg, mpga, oga, ogg, wav, webm"
            ),
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
    prompt: Annotated[
        str | None,
        Form(
            description=(
                "An optional text to guide the model's style or continue a previous audio segment.\n"
                "The prompt should be in English.\nUNSUPPORTED on this implementation."
            )
        ),
    ] = None,
    response_format: Annotated[
        TranslateAudioResponseFormat,
        Form(
            description=(
                "The format of the transcript output.\n"
                "Supported formats: `json`, `text`, `srt`, `verbose_json`, `vtt`"
            )
        ),
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

    Translates audio from any supported language into English text. The model will
    automatically detect the source language and convert the audio to English text.
    Supports multiple output formats including plain text, JSON with metadata, and
    subtitle formats for video content.

    Args:
        file: The audio file to translate.
        model: The transcription model to use. Available models: amazon.transcribe.
        prompt: Optional style guidance for the model. UNSUPPORTED on this implementation.
        response_format: Output format: `json`, `text`, `srt`, `verbose_json`, or `vtt`.
        temperature: Sampling temperature. UNSUPPORTED on this implementation (must be 0.0).

    Returns:
        The translated text in English in the requested format.

    Raises:
        ApiError: When translation fails or invalid parameters are provided.
    """
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
        audio_content=InputFile(file),
        response_format=request.response_format,
        temperature=request.temperature,
        prompt=request.prompt,
    )
