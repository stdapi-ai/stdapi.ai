"""Unit tests for stdapi.types.openai_audio local OpenAI-compatible audio types.

Ref: https://developers.openai.com/api/docs/guides/text-to-speech#voice-options
     https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create
     stdapi/types/openai_audio.py
"""

import pytest
from pydantic import ValidationError

from stdapi.types.openai_audio import (
    OPENAI_VOICES_FEMALE,
    Transcription,
    TranscriptionCreateParams,
    TranscriptionLanguage,
)

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Gender the gateway assigns to the voices OpenAI added after its original set.
_EXPECTED_FEMALE = {"marin": True, "cedar": False}


class TestOpenaiVoicesFemale:
    """OPENAI_VOICES_FEMALE maps every OpenAI built-in voice to a Polly gender.

    Amazon Polly has no OpenAI voice names, so ``_select_voice`` picks a Polly
    voice by gender and detected language.  A voice missing from this table is
    forwarded to Polly verbatim and rejected as an unknown ``VoiceId``.

    Ref: https://docs.aws.amazon.com/polly/latest/dg/available-voices.html
         https://stdapi.ai/api_openai_audio_speech/
         stdapi/models/audio/amazon_polly.py:_select_voice
    """

    @pytest.mark.parametrize("voice", ["marin", "cedar"])
    def test_newer_voices_are_mapped(self, voice: str) -> None:
        """The newer marin/cedar voices resolve to a gender instead of KeyError.

        Ref: stdapi/types/openai_audio.py:OPENAI_VOICES_FEMALE
        """
        assert voice in OPENAI_VOICES_FEMALE
        assert OPENAI_VOICES_FEMALE[voice] is _EXPECTED_FEMALE[voice], (
            f"{voice} must resolve to a fixed gender so Polly voice selection is stable"
        )


class TestTranscriptionCreateParamsKnownSpeaker:
    """TranscriptionCreateParams: known-speaker diarization is accepted and ignored.

    Amazon Transcribe cannot be seeded with reference audio, so the gateway
    keeps these upstream fields valid and degrades to generic ``A``/``B``
    speaker labels rather than failing the request.

    Ref: https://docs.aws.amazon.com/transcribe/latest/dg/diarization.html
         https://stdapi.ai/api_openai_audio_transcriptions/
         stdapi/types/openai_audio.py:TranscriptionCreateParams._unsupported
    """

    def test_known_speaker_names_is_accepted(self) -> None:
        """known_speaker_names validates instead of raising, keeping upstream clients working."""
        params = TranscriptionCreateParams(
            model="amazon.transcribe", known_speaker_names=["agent"]
        )
        assert params.known_speaker_names == ["agent"]

    def test_known_speaker_references_is_accepted(self) -> None:
        """known_speaker_references validates instead of raising, keeping upstream clients working."""
        params = TranscriptionCreateParams(
            model="amazon.transcribe",
            known_speaker_references=["data:audio/wav;base64,AAA"],
        )
        assert params.known_speaker_references == ["data:audio/wav;base64,AAA"]

    def test_request_without_known_speaker_fields_is_unaffected(self) -> None:
        """A request that omits both fields still validates normally."""
        params = TranscriptionCreateParams(model="amazon.transcribe")
        assert params.known_speaker_names is None
        assert params.known_speaker_references is None


class TestTranscriptionCreateParamsStreamingSubtitles:
    """Subtitle formats are refused with ``stream=true``; ``diarized_json`` is not.

    A streamed transcription carries text events, so ``srt`` and ``vtt`` ask for
    cues it has no event to hold: accepting the pair would answer 200 with plain
    text, a payload the caller cannot parse as subtitles. ``diarized_json`` has
    an event of its own -- ``transcript.text.segment`` -- so it is accepted here
    and refused by the models that cannot label speakers.

    Ref: https://docs.aws.amazon.com/transcribe/latest/dg/subtitles.html
         https://developers.openai.com/api/docs/guides/speech-to-text
         stdapi/types/openai_audio.py:TranscriptionCreateParams
    """

    @pytest.mark.parametrize("response_format", ["srt", "vtt"])
    def test_unstreamable_format_with_streaming_is_rejected(
        self, response_format: str
    ) -> None:
        """Each format the stream cannot express is refused when streaming."""
        with pytest.raises(ValidationError, match="stream=true"):
            TranscriptionCreateParams(
                model="amazon.transcribe",
                response_format=response_format,  # type: ignore[arg-type]
                stream=True,
            )

    def test_diarized_json_with_streaming_is_accepted(self) -> None:
        """``diarized_json`` streams as ``transcript.text.segment`` events.

        Ref: openai.types.audio.transcription_text_segment_event.TranscriptionTextSegmentEvent
        """
        params = TranscriptionCreateParams(
            model="amazon.transcribe", response_format="diarized_json", stream=True
        )
        assert params.response_format == "diarized_json"
        assert params.stream is True

    @pytest.mark.parametrize("response_format", ["srt", "vtt", "diarized_json"])
    def test_unstreamable_format_without_streaming_is_accepted(
        self, response_format: str
    ) -> None:
        """The same formats stay valid for a non-streaming request."""
        params = TranscriptionCreateParams(
            model="amazon.transcribe",
            response_format=response_format,  # type: ignore[arg-type]
        )
        assert params.response_format == response_format

    @pytest.mark.parametrize("response_format", ["json", "text"])
    def test_streamable_formats_are_unaffected(self, response_format: str) -> None:
        """Formats the streaming path can emit keep working with ``stream=true``."""
        params = TranscriptionCreateParams(
            model="amazon.transcribe",
            response_format=response_format,  # type: ignore[arg-type]
            stream=True,
        )
        assert params.stream is True


class TestTranscriptionCreateParamsLanguages:
    """TranscriptionCreateParams: the plural ``languages`` expected-language hints.

    The gpt-transcribe family replaces the singular ``language`` field with a
    ``languages`` list; the upstream docs are explicit that both must not be
    sent in the same request.

    Ref: https://developers.openai.com/api/docs/guides/transcription
         https://platform.openai.com/docs/api-reference/audio/createTranscription
         stdapi/types/openai_audio.py:TranscriptionCreateParams._unsupported
    """

    def test_languages_list_is_accepted(self) -> None:
        """A multi-entry languages list validates and is preserved as sent.

        Ref: stdapi/types/openai_audio.py:TranscriptionCreateParams
        """
        params = TranscriptionCreateParams(model="m", languages=["en", "fr"])
        assert params.languages == ["en", "fr"]
        assert params.language is None

    def test_single_entry_languages_is_accepted(self) -> None:
        """A one-entry list stays valid; backends treat it like ``language``.

        Ref: stdapi/models/audio/amazon_transcribe.py:_apply_language_params
        """
        params = TranscriptionCreateParams(model="m", languages=["en"])
        assert params.languages == ["en"]

    def test_language_with_languages_is_rejected(self) -> None:
        """Sending both fields is refused instead of one silently winning.

        Ref: https://developers.openai.com/api/docs/guides/transcription
             stdapi/types/openai_audio.py:TranscriptionCreateParams._unsupported
        """
        with pytest.raises(ValidationError, match="cannot be combined"):
            TranscriptionCreateParams(model="m", language="en", languages=["en", "fr"])


class TestTranscriptionCreateParamsKeywords:
    """TranscriptionCreateParams: ``keywords`` must be single-line literals.

    The upstream cookbook requires each keyword to be a literal term without
    markup or line breaks, so malformed entries are refused up front instead
    of degrading the transcription context.

    Ref: https://developers.openai.com/cookbook/examples/migrating_from_whisper_to_gpt_transcribe
         stdapi/types/openai_audio.py:TranscriptionCreateParams._unsupported
    """

    def test_keywords_are_accepted(self) -> None:
        """Plain literal terms validate and are preserved as sent.

        Ref: stdapi/types/openai_audio.py:TranscriptionCreateParams
        """
        params = TranscriptionCreateParams(
            model="m", keywords=["Amoxicillin", "EBITDA"]
        )
        assert params.keywords == ["Amoxicillin", "EBITDA"]

    @pytest.mark.parametrize("keyword", ["<tag>", "a>b", "line\nbreak", "cr\rhere", ""])
    def test_invalid_keyword_is_rejected(self, keyword: str) -> None:
        """Markup characters, line breaks and empty entries are refused.

        Ref: stdapi/types/openai_audio.py:TranscriptionCreateParams._unsupported
        """
        with pytest.raises(ValidationError, match="keyword"):
            TranscriptionCreateParams(model="m", keywords=[keyword])


class TestTranscriptionResponseLanguages:
    """Transcription response: the detected-``languages`` array.

    Ref: https://platform.openai.com/docs/api-reference/audio/createTranscription
         stdapi/types/openai_audio.py:Transcription
    """

    def test_languages_entries_carry_a_code(self) -> None:
        """Each entry is a ``{code}`` object mirroring upstream's shape.

        Ref: stdapi/types/openai_audio.py:TranscriptionLanguage
        """
        response = Transcription(
            text="hello", languages=[TranscriptionLanguage(code="en")]
        )
        assert [entry.code for entry in response.languages or []] == ["en"]

    def test_languages_defaults_to_none(self) -> None:
        """Models without language identification keep the field unset.

        Ref: stdapi/models/audio/_default.py:AudioModel.stt
        """
        assert Transcription(text="hello").languages is None
