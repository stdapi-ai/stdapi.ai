"""Mistral Voxtral model implementation.

Voxtral transcribes through the generic Converse speech-to-text default; this
subclass only pins the model family in the audio registry.
"""

from stdapi.models.audio._default import AudioModel as ConverseAudioModel


class AudioModel(ConverseAudioModel):
    """Mistral Voxtral audio model implementation."""

    MATCHER = "mistral.voxtral-"
