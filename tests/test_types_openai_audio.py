"""Unit tests for stdapi.types.openai_audio local OpenAI-compatible audio types.

Ref: https://developers.openai.com/api/docs/guides/text-to-speech#voice-options
     https://developers.openai.com/api/reference/resources/audio/subresources/transcriptions/methods/create
     stdapi/types/openai_audio.py
"""

import pytest

from stdapi.types.openai_audio import OPENAI_VOICES_FEMALE, TranscriptionCreateParams

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
