"""Medical transcription through ``/v1/audio/transcriptions``, against AWS.

Two proofs on a few seconds of synthesized clinical speech: a transcription job,
and a live session with a specialty only a live session takes. No upstream API
offers a medical transcription model, so neither runs on the vendor lane.

Ref: https://docs.aws.amazon.com/transcribe/latest/APIReference/API_StartMedicalTranscriptionJob.html
     https://docs.aws.amazon.com/transcribe/latest/APIReference/API_streaming_StartMedicalStreamTranscription.html
     https://stdapi.ai/api_openai_audio_transcriptions/#medical-transcription
"""

from base64 import b64encode
from json import loads
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from openai import OpenAI

#: No upstream API offers a medical transcription model.
pytestmark = pytest.mark.gateway("no upstream medical transcription model")

#: Model under test.
_MODEL = "amazon.transcribe-medical"

#: Clinical dictation, spoken by the speech endpoint.
_DICTATION = "Patient reports chest pain. Prescribed metoprolol twenty five milligrams."

#: Words of the dictation a correct transcript contains; one is enough.
_DICTATION_WORDS = ("chest", "pain", "metoprolol", "prescribed")


@pytest.fixture(scope="module")
def medical_audio(openai_client: OpenAI, speech_standard_model: str) -> str:
    """Synthesize the dictation once per module, as a WAV data URI.

    Returns:
        The audio, as the ``file`` of a JSON request body.
    """
    audio = openai_client.audio.speech.create(
        model=speech_standard_model,
        voice="alloy",
        input=_DICTATION,
        response_format="wav",
    ).content
    return f"data:audio/wav;base64,{b64encode(audio).decode()}"


def _post(openai_client: OpenAI, body: dict[str, Any]) -> Any:  # noqa: ANN401
    """Post a JSON transcription request, the only body carrying extra parameters.

    Args:
        openai_client: Client whose base URL and key target the gateway.
        body: The request body.

    Returns:
        The HTTP response.
    """
    http_client = openai_client._client  # noqa: SLF001
    return http_client.post(
        f"{openai_client.base_url}audio/transcriptions",
        json=body,
        headers={"Authorization": f"Bearer {openai_client.api_key}"},
        timeout=120,
    )


@pytest.mark.slow
def test_a_dictation_is_transcribed_by_a_medical_job(
    openai_client: OpenAI, medical_audio: str
) -> None:
    """A non-streamed request runs a medical job and answers in US English.

    The job waits for AWS to finish (about ten seconds), hence ``slow``.

    Ref: https://docs.aws.amazon.com/transcribe/latest/APIReference/API_StartMedicalTranscriptionJob.html
         stdapi/models/audio/amazon_transcribe_medical.py:AudioModel._transcribe
    """
    response = _post(
        openai_client,
        {"file": medical_audio, "model": _MODEL, "language": "en", "Type": "DICTATION"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert any(word in body["text"].lower() for word in _DICTATION_WORDS), body
    assert body["usage"]["type"] == "duration"
    assert body["usage"]["seconds"] > 0
    assert body["languages"] == [{"code": "en"}]


def test_a_specialty_is_transcribed_by_a_live_medical_session(
    openai_client: OpenAI, medical_audio: str
) -> None:
    """A streamed request takes a specialty a job refuses, and streams the transcript.

    Ref: https://docs.aws.amazon.com/transcribe/latest/APIReference/API_streaming_StartMedicalStreamTranscription.html
         stdapi/models/audio/amazon_transcribe_medical.py:AudioModel._live_request
    """
    response = _post(
        openai_client,
        {
            "file": medical_audio,
            "model": _MODEL,
            "stream": True,
            "Specialty": "CARDIOLOGY",
            "Type": "DICTATION",
        },
    )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [
        loads(line.removeprefix("data:").strip())
        for line in response.text.splitlines()
        if line.startswith("data:") and "[DONE]" not in line
    ]
    types = [event["type"] for event in events]
    assert "transcript.text.delta" in types
    assert types[-1] == "transcript.text.done"
    done = events[-1]["text"]
    assert done == "".join(
        event["delta"] for event in events if event["type"] == "transcript.text.delta"
    )
    assert any(word in done.lower() for word in _DICTATION_WORDS), done
