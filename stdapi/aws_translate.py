"""AWS Translate utilities."""

from html import escape, unescape
from io import StringIO
from re import DOTALL, IGNORECASE
from re import compile as compile_regex
from typing import TYPE_CHECKING

from botocore.exceptions import ClientError, ParamValidationError

from stdapi.api_errors import ApiError
from stdapi.aws import call_with_region_failover, service_regions
from stdapi.config import SETTINGS
from stdapi.monitoring import log_error_details
from stdapi.usage import record_translate_usage
from stdapi.utils import language_code_to_name

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_translate import TranslateClient
    from types_aiobotocore_translate.type_defs import (
        TranslateTextRequestTypeDef,
        TranslateTextResponseTypeDef,
    )

    from stdapi.types import JsonMapping


class TranslationError(Exception):
    """Exception raised when translation fails."""


#: Region-qualified codes AWS Translate treats as distinct from their base language
_TRANSLATE_DISTINCT_LANGUAGE_CODES: frozenset[str] = frozenset(
    ("es-MX", "fa-AF", "fr-CA", "pt-PT", "zh-TW")
)

#: Matches a subtitle span tag, capturing its segment number and inner text.
_SUBTITLE_SPAN_RE = compile_regex(
    r'<span[^>]*id="seg(\d+)"[^>]*>(.*?)</span>', IGNORECASE | DOTALL
)


async def translate(
    text: str,
    source_language_code: str,
    target_language_code: str = "en",
    settings: JsonMapping | None = None,
    terminology_names: list[str] | None = None,
) -> str:
    """Translate text from source language to English using AWS Translate.

    Args:
        text: Text to translate
        source_language_code: Source language code (e.g., 'es-US', 'fr-FR')
        target_language_code: Target language code (default: 'en')
        settings: Optional TranslateText ``Settings`` (``Formality``, ``Profanity``,
            ``Brevity``)
        terminology_names: Optional pre-existing custom terminology names

    Returns:
        Translated text in English

    Raises:
        ApiError: When translation fails, or ``settings``/``terminology_names``
            hold a value AWS Translate rejects
    """
    if source_language_code not in _TRANSLATE_DISTINCT_LANGUAGE_CODES:
        source_language_code = source_language_code.split("-", 1)[0]
    if not text.strip() or source_language_code == "en":
        return text

    def _translate(
        client: TranslateClient, _region: RegionName
    ) -> Awaitable[TranslateTextResponseTypeDef]:
        """Start the translation call on one region's client."""
        request: TranslateTextRequestTypeDef = {
            "Text": text,
            "SourceLanguageCode": source_language_code,
            "TargetLanguageCode": target_language_code,
        }
        if settings:
            request["Settings"] = settings  # type: ignore[typeddict-item]
        if terminology_names:
            request["TerminologyNames"] = terminology_names
        return client.translate_text(**request)

    try:
        result, used_region = await call_with_region_failover(
            "translate", service_regions(SETTINGS.aws_translate_region), _translate
        )
        record_translate_usage(len(text), region=used_region)
        return result["TranslatedText"]

    except ClientError as error:
        if error.response["Error"]["Code"] == "UnsupportedLanguagePairException":
            msg = (
                f"Translation from {language_code_to_name(source_language_code).capitalize()} "
                f"to {language_code_to_name(target_language_code).capitalize()} is not "
                "supported. Choose a supported language pair."
            )
            raise ApiError(msg) from None
        raise
    except ParamValidationError as error:
        # botocore validates Settings/TerminologyNames client-side; surface it
        # as a caller 400 instead of an unhandled 500.
        log_error_details(str(error))
        msg = "Invalid translation settings or terminology names."
        raise ApiError(msg) from error


async def translate_subtitle(
    subtitle_content: str,
    source_language_code: str,
    target_language_code: str = "en",
    settings: JsonMapping | None = None,
    terminology_names: list[str] | None = None,
) -> str:
    """Translate subtitle content while preserving timing and structure.

    Segments are wrapped in HTML span tags so a single AWS Translate call
    covers them all, then reassembled into the original subtitle format.

    Args:
        subtitle_content: Original subtitle content in SRT or VTT format
        source_language_code: ISO language code of the source language (e.g., 'es-US', 'fr-FR')
        target_language_code: ISO language code of the target language (default: 'en')
        settings: Optional TranslateText ``Settings`` (``Formality``, ``Profanity``,
            ``Brevity``)
        terminology_names: Optional pre-existing custom terminology names

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
        settings,
        terminology_names,
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
    return stripped.isdigit()


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
        if not webvtt_header_done:
            webvtt_header_done = _subtitle_should_skip_webvtt_header(stripped)
            continue

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

    AWS Translate preserves HTML structure while translating text content, so the
    unique span ID maps each translated segment back to its original.

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
    """Extract the translated text of every span tag, ordered by span ID.

    Args:
        translated_html: HTML response from AWS Translate
        segment_count: Expected number of segments

    Returns:
        List of translated text segments in original order

    Raises:
        TranslationError: If a translated span tag cannot be parsed from the HTML response.
    """
    segments_by_index: dict[int, str] = {}
    for match in _SUBTITLE_SPAN_RE.finditer(translated_html):
        # First occurrence wins on a duplicated segment ID.
        segments_by_index.setdefault(int(match.group(1)), match.group(2))

    translated_segments = []
    for i in range(segment_count):
        if i not in segments_by_index:
            msg = "Unable to parse translated HTML"
            raise TranslationError(msg)  # pragma: no cover
        translated_segments.append(unescape(segments_by_index[i]))
    return translated_segments
