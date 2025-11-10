"""OpenAI-compatible Audio Translation API implementation using AWS Transcribe and Translate."""

from html import escape, unescape
from io import StringIO
from re import DOTALL, IGNORECASE, search
from typing import TYPE_CHECKING, Annotated

from botocore.exceptions import ClientError
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
)

from stdapi.auth import authenticate
from stdapi.aws import get_client
from stdapi.config import SETTINGS
from stdapi.monitoring import REQUEST_LOG, log_request_params
from stdapi.openai_exceptions import OpenaiUnsupportedModelError
from stdapi.routes.openai_audio_transcriptions import (
    TRANSCRIBE_MODEL_ID,
    format_subtitle_response,
    get_audio_duration,
    get_transcript_text,
    perform_transcription_task,
)
from stdapi.types.openai_audio import (
    AudioResponseFormat,
    TranscriptionSegment,
    Translation,
    TranslationCreateParams,
    TranslationCreateResponse,
    TranslationVerbose,
)
from stdapi.utils import language_code_to_name, validation_error_handler

if TYPE_CHECKING:
    from types_aiobotocore_translate.client import TranslateClient

    from stdapi.routes.openai_audio_transcriptions import TranscriptionJobData

router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1/audio", tags=["audio", "openai"]
)


class TranslationError(Exception):
    """Exception raised when translation fails."""


async def translate_text_to_english(text: str, source_language_code: str) -> str:
    """Translate text from source language to English using AWS Translate.

    Args:
        text: Text to translate
        source_language_code: Source language code (e.g., 'es-US', 'fr-FR')

    Returns:
        Translated text in English

    Raises:
        HTTPException: When translation fails
    """
    source_language_code = source_language_code.split("-", 1)[0]
    if not text.strip() or source_language_code == "en":
        return text

    try:
        translate_client: TranslateClient = get_client("translate")
        return (
            await translate_client.translate_text(
                Text=text,
                SourceLanguageCode=source_language_code,
                TargetLanguageCode="en",
            )
        )["TranslatedText"]

    except ClientError as error:
        if error.response["Error"]["Code"] == "UnsupportedLanguagePairException":
            raise HTTPException(
                status_code=400,
                detail=f"Translation from {language_code_to_name(source_language_code).capitalize()} "
                f"is not supported: {error.response['Error']['Message']}",
            ) from None
        raise


def _is_subtitle_text_line(stripped: str) -> bool:
    """Check if a line contains subtitle text content.

    Args:
        stripped: Current line being processed stripped of whitespace

    Returns:
        True if the line contains text content for subtitles
    """
    return bool(stripped and not stripped.isdigit() and "-->" not in stripped)


def _process_subtitle_segment(segments: list[str], current_segment: list[str]) -> None:
    """Process a completed subtitle segment and add to segments list.

    Args:
        segments: List to append completed segment to
        current_segment: Current segment being built
    """
    if current_segment:
        segments.append("\n".join(current_segment))
        current_segment.clear()


def _should_skip_webvtt_header(stripped: str) -> bool:
    """Determine if line should be skipped for WebVTT header processing.

    Args:
        stripped: Current line being processed stripped of whitespace

    Returns:
        Tuple of (should_skip, updated_webvtt_header_done)
    """
    return stripped.isdigit()  # Found first subtitle number, header done


def extract_subtitle_text_segments(subtitle_content: str) -> list[str]:
    """Extract text segments from subtitle content while preserving structure.

    Works with both SRT and VTT formats from AWS Transcribe.

    Args:
        subtitle_content: Raw subtitle content (SRT or VTT format)

    Returns:
        List of text segments to be translated
    """
    segments: list[str] = []
    lines = subtitle_content.strip().split("\n")
    webvtt_header_done = False
    segment: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Handle WebVTT header processing
        if not webvtt_header_done:
            webvtt_header_done = _should_skip_webvtt_header(stripped)
            continue

        # Process different line types
        if _is_subtitle_text_line(stripped):
            segment.append(line)
        elif not line.strip():  # Empty line indicates segment boundary
            _process_subtitle_segment(segments, segment)

    # Handle final segment if file doesn't end with empty line
    _process_subtitle_segment(segments, segment)
    return segments


def reconstruct_subtitle_with_translation(
    original_content: str, text_segments: list[str], translated_segments: list[str]
) -> str:
    """Reconstruct subtitle content with translated text segments.

    Args:
        original_content: Original subtitle content
        text_segments: Original text segments
        translated_segments: Translated text segments

    Returns:
        Reconstructed subtitle content with translated text
    """
    result = StringIO()
    current_pos = 0
    for text, translated in zip(text_segments, translated_segments, strict=False):
        segment_start = original_content.find(text, current_pos)
        result.write(original_content[current_pos:segment_start])
        result.write(translated)
        current_pos = segment_start + len(text)
    result.write(original_content[current_pos:])
    return result.getvalue()


def create_html_for_translation(text_segments: list[str]) -> str:
    """Create HTML document with text segments wrapped in span tags for AWS Translate.

    AWS Translate can process HTML documents and preserve the structure while translating
    the text content. Each subtitle segment is wrapped in a span tag with a unique ID
    to maintain the mapping between original and translated segments.

    Args:
        text_segments: List of text segments to be translated

    Returns:
        HTML document with segments wrapped in span tags
    """
    html_parts = ["<!DOCTYPE html><html><body>"]
    for i, segment in enumerate(text_segments):
        html_parts.append(f'<span id="seg{i}">{escape(segment)}</span>')
    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def parse_translated_html(translated_html: str, segment_count: int) -> list[str]:
    """Parse translated HTML response to extract translated text segments.

    Extracts the translated text from each span tag in the HTML response,
    maintaining the original order based on the span IDs.

    Args:
        translated_html: HTML response from AWS Translate
        segment_count: Expected number of segments

    Returns:
        List of translated text segments in original order
    """
    translated_segments = []
    for i in range(segment_count):
        match = search(
            rf'<span[^>]*id="seg{i}"[^>]*>(.*?)</span>',
            translated_html,
            IGNORECASE | DOTALL,
        )
        if match:
            translated_segments.append(unescape(match.group(1)))
            continue
        msg = "Unable to parse translated HTML"
        raise TranslationError(msg)  # pragma: no cover
    return translated_segments


async def translate_subtitle_content(
    subtitle_content: str, source_language_code: str
) -> str:
    """Translate subtitle content while preserving timing and structure.

    Uses AWS Translate with HTML span tags to efficiently translate all subtitle
    segments in a single API call, then reconstructs the subtitle format with
    translated text while preserving timing and structure.

    This function extracts text segments from subtitle files (SRT or VTT format),
    wraps them in HTML span tags for batch translation, sends them to AWS Translate,
    and then reconstructs the original subtitle format with the translated text
    while preserving all timing information and structural elements.

    Args:
        subtitle_content: Original subtitle content in SRT or VTT format
        source_language_code: ISO language code of the source language (e.g., 'es-US', 'fr-FR')

    Returns:
        Translated subtitle content in the same format as input, with text translated
        to English while preserving timing, sequence numbers, and formatting

    Raises:
        HTTPException: When AWS Translate service fails or returns an error
        Exception: For any other translation processing errors (returns original content as fallback)

    Example:
        Input: SRT content with Spanish text
        Output: SRT content with English translation, preserving timing
    """
    text_segments = extract_subtitle_text_segments(subtitle_content)
    if not text_segments:
        return subtitle_content

    translated_html = await translate_text_to_english(
        create_html_for_translation(text_segments), source_language_code
    )
    return reconstruct_subtitle_with_translation(
        subtitle_content,
        text_segments,
        parse_translated_html(translated_html, len(text_segments)),
    )


def format_translation_response(
    transcript_data: "TranscriptionJobData",
    text: str,
    response_format: AudioResponseFormat,
) -> str | TranslationCreateResponse:
    """Format translation response based on requested output format.

    Converts transcript data into the appropriate translation response format following
    OpenAI API specification. Supports plain text, JSON, and verbose JSON formats.
    Translation responses always convert audio to English text.

    Args:
        transcript_data: Parsed transcription results from AWS Transcribe
        text: Processed and translated transcript text content (in English)
        response_format: Desired output format (text, json, verbose_json)

    Returns:
        Formatted response as string for text format or OpenAI translation types for JSON formats
    """
    if response_format == "text":
        return text

    if response_format == "verbose_json":
        return TranslationVerbose(
            duration=get_audio_duration(transcript_data),
            language="english",  # Translation output is always English
            text=text,
            segments=[
                TranscriptionSegment(
                    id=segment["id"],
                    end=float(segment["end_time"]),
                    start=float(segment["start_time"]),
                    text=segment["transcript"],
                    # Not supported
                    no_speech_prob=0.0 if len(segment["transcript"]) else 1.0,
                    avg_logprob=0.0,
                    compression_ratio=0.0,
                    seek=0,
                    temperature=0.0,
                    tokens=[],
                )
                for segment in transcript_data["audio_segments"]
            ],
        )
    return Translation(text=text)


@router.post(
    "/translations",
    response_model=None,
    summary="OpenAI - /v1/audio/translations",
    description=(
        "Translates audio from any supported language into English text.\n"
        "The model will automatically detect the source language and convert the audio to English text."
        "Supports multiple output formats including plain text, JSON, verbose JSON, and subtitle formats (SRT/VTT)."
    ),
    response_description="Returns translation in the specified format (always English text)",
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
    ] = TRANSCRIBE_MODEL_ID,
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
        AudioResponseFormat,
        Form(
            description=(
                "The format of the transcript output.\n"
                "Supported formats: `json`, `text`, `srt`, `verbose_json`, `vtt`"
            )
        ),
    ] = "json",
    temperature: Annotated[
        float,
        Form(
            description=(
                "The sampling temperature, between `0` and `1`.\n"
                "Higher values like `0.8` will make the output more random, while lower values like `0.2` will make it more focused and deterministic. "
                "If set to `0`, the model will use log probability to automatically increase the temperature until certain thresholds are hit.\n"
                "UNSUPPORTED on this implementation."
            )
        ),
    ] = 0.0,
    background_tasks: BackgroundTasks = BackgroundTasks(),
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
        background_tasks: FastAPI background tasks for cleanup.

    Returns:
        The translated text in English in the requested format.

    Raises:
        HTTPException: When translation fails or invalid parameters are provided.
    """
    with validation_error_handler():
        request = TranslationCreateParams(
            model=model,
            prompt=prompt,
            response_format=response_format,
            temperature=temperature,
        )
    log_request_params(request)

    if model != TRANSCRIBE_MODEL_ID:
        raise OpenaiUnsupportedModelError(model)
    log = REQUEST_LOG.get()
    log["model_id"] = model

    transcript_data = await perform_transcription_task(
        audio_content=await file.read(),
        background_tasks=background_tasks,
        language=None,  # Auto-detect source language
        response_format=request.response_format,
    )
    language = transcript_data["language_code"]

    try:
        subtitle_content = transcript_data["subtitle_content"]
    except KeyError:
        return format_translation_response(
            transcript_data,
            await translate_text_to_english(
                get_transcript_text(transcript_data), language
            ),
            request.response_format,
        )
    else:
        return format_subtitle_response(
            request.response_format,
            (await translate_subtitle_content(subtitle_content, language)),
            file,
        )
