"""Unit tests for stdapi.types.openai_audio local OpenAI-compatible audio types."""

import pytest

from stdapi.types.openai_audio import OPENAI_VOICES_FEMALE

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


class TestOpenaiVoicesFemale:
    """OPENAI_VOICES_FEMALE: covers every OpenAI-documented built-in voice."""

    @pytest.mark.parametrize("voice", ["marin", "cedar"])
    def test_newer_voices_are_mapped(self, voice: str) -> None:
        """The newer marin/cedar voices resolve to a gender instead of KeyError."""
        assert voice in OPENAI_VOICES_FEMALE
