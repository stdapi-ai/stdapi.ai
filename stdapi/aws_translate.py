"""AWS Translate utilities."""

from html import escape, unescape
from io import StringIO
from re import DOTALL, IGNORECASE, search
from typing import TYPE_CHECKING

from botocore.exceptions import ClientError

from stdapi.api_errors import ApiError
from stdapi.aws import get_client
from stdapi.utils import language_code_to_name

if TYPE_CHECKING:
    from types_aiobotocore_translate import TranslateClient


class TranslationError(Exception):
    """Exception raised when translation fails."""


async def translate(
    text: str, source_language_code: str, target_language_code: str = "en"
) -> str:
    """Translate text from source language to English using AWS Translate.

    Args:
        text: Text to translate
        source_language_code: Source language code (e.g., 'es-US', 'fr-FR')
        target_language_code: Target language code (default: 'en')

    Returns:
        Translated text in English

    Raises:
        ApiError: When translation fails
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
                TargetLanguageCode=target_language_code,
            )
        )["TranslatedText"]

    except ClientError as error:
        if error.response["Error"]["Code"] == "UnsupportedLanguagePairException":
            msg = (
                f"Translation from {language_code_to_name(source_language_code).capitalize()} "
                f"is not supported: {error.response['Error']['Message']}"
            )
            raise ApiError(msg) from None
        raise


async def translate_subtitle(
    subtitle_content: str, source_language_code: str, target_language_code: str = "en"
) -> str:
    """Translate subtitle content while preserving timing and structure.

    Uses AWS Translate with HTML span tags to efficiently translate all subtitle
    segments in a single API call, then reconstructs the subtitle format with
    translated text while preserving timing and structure.

    Args:
        subtitle_content: Original subtitle content in SRT or VTT format
        source_language_code: ISO language code of the source language (e.g., 'es-US', 'fr-FR')
        target_language_code: ISO language code of the target language (default: 'en')

    Returns:
        Translated subtitle content in the same format as input

    Raises:
        ApiError: When AWS Translate service fails or returns an error
    """
    text_segments = _subtitle_extract_text_segments(subtitle_content)
    if not text_segments:
        return subtitle_content

    translated_html = await translate(
        _subtitle_create_html_for_translation(text_segments),
        source_language_code,
        target_language_code,
    )
    return _subtitle_reconstruct_with_translation(
        subtitle_content,
        text_segments,
        _subtitle_parse_translated_html(translated_html, len(text_segments)),
    )


def _subtitle_is_text_line(stripped: str) -> bool:
    """Check if a line contains subtitle text content.

    Args:
        stripped: Current line being processed stripped of whitespace

    Returns:
        True if the line contains text content for subtitles
    """
    return bool(stripped and not stripped.isdigit() and "-->" not in stripped)


def _subtitle_process_segment(segments: list[str], current_segment: list[str]) -> None:
    """Process a completed subtitle segment and add to segments list.

    Args:
        segments: List to append completed segment to
        current_segment: Current segment being built
    """
    if current_segment:
        segments.append("\n".join(current_segment))
        current_segment.clear()


def _subtitle_should_skip_webvtt_header(stripped: str) -> bool:
    """Determine if line should be skipped for WebVTT header processing.

    Args:
        stripped: Current line being processed stripped of whitespace

    Returns:
        True if the line is the first subtitle number (header done)
    """
    return stripped.isdigit()  # Found first subtitle number, header done


def _subtitle_extract_text_segments(subtitle_content: str) -> list[str]:
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
            webvtt_header_done = _subtitle_should_skip_webvtt_header(stripped)
            continue

        # Process different line types
        if _subtitle_is_text_line(stripped):
            segment.append(line)
        elif not line.strip():  # Empty line indicates segment boundary
            _subtitle_process_segment(segments, segment)

    # Handle final segment if file doesn't end with empty line
    _subtitle_process_segment(segments, segment)
    return segments


def _subtitle_reconstruct_with_translation(
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


def _subtitle_create_html_for_translation(text_segments: list[str]) -> str:
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


def _subtitle_parse_translated_html(
    translated_html: str, segment_count: int
) -> list[str]:
    """Parse translated HTML response to extract translated text segments.

    Extracts the translated text from each span tag in the HTML response,
    maintaining the original order based on the span IDs.

    Args:
        translated_html: HTML response from AWS Translate
        segment_count: Expected number of segments

    Returns:
        List of translated text segments in original order

    Raises:
        TranslationError: If a translated span tag cannot be parsed from the HTML response.
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
