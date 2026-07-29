"""Unit tests for stdapi.types.openai_audio local OpenAI-compatible audio types."""

import pytest

from stdapi.types.openai_audio import OPENAI_VOICES_FEMALE, TranscriptionCreateParams

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


class TestOpenaiVoicesFemale:
    """OPENAI_VOICES_FEMALE: covers every OpenAI-documented built-in voice."""

    @pytest.mark.parametrize("voice", ["marin", "cedar"])
    def test_newer_voices_are_mapped(self, voice: str) -> None:
        """The newer marin/cedar voices resolve to a gender instead of KeyError."""
        assert voice in OPENAI_VOICES_FEMALE


class TestTranscriptionCreateParamsKnownSpeaker:
    """TranscriptionCreateParams: known-speaker diarization is accepted and ignored."""

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
