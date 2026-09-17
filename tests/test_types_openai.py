"""Unit tests for the OpenAI-compatible types shared by several surfaces.

Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     https://github.com/openai/openai-python
     stdapi/types/openai.py
"""

from typing import TYPE_CHECKING, Any

import pytest
from openai.types.audio.speech_create_params import Voice as SpeechVoice
from openai.types.chat.chat_completion_audio_param import Voice as ChatAudioVoice
from openai.types.realtime.realtime_audio_config_output_param import (
    Voice as SessionAudioVoice,
)
from openai.types.realtime.realtime_response_create_audio_output_param import (
    OutputVoice as ResponseAudioVoice,
)
from pydantic import BaseModel, TypeAdapter, ValidationError

from stdapi.types.openai_audio import SpeechCreateParams
from stdapi.types.openai_chat_completions import ChatCompletionAudioParam
from stdapi.types.openai_realtime import RealtimeSessionConfig

if TYPE_CHECKING:
    from collections.abc import Callable

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Every OpenAI SDK alias typing a `voice` as a name or a custom-voice object.
SDK_VOICE_ALIASES: dict[str, Any] = {
    "audio.speech": SpeechVoice,
    "chat.audio": ChatAudioVoice,
    "realtime.session.audio.output": SessionAudioVoice,
    "realtime.response.audio.output": ResponseAudioVoice,
}

#: Mappings carrying no usable custom voice identifier.
UNUSABLE_VOICE_OBJECTS: tuple[dict[str, Any], ...] = (
    {},
    {"id": None},
    {"id": 1234},
    {"name": "Amy"},
)


def _speech(voice: object) -> SpeechCreateParams:
    """Validate a speech request body naming *voice*.

    Args:
        voice: The value sent as `voice`.

    Returns:
        The validated request body.
    """
    return SpeechCreateParams.model_validate(
        {"model": "amazon.polly-neural", "input": "Test.", "voice": voice}
    )


def _chat_audio(voice: object) -> ChatCompletionAudioParam:
    """Validate a chat-completion audio output configuration naming *voice*.

    Args:
        voice: The value sent as `voice`.

    Returns:
        The validated configuration.
    """
    return ChatCompletionAudioParam.model_validate({"format": "mp3", "voice": voice})


def _session(voice: object) -> RealtimeSessionConfig:
    """Validate a Realtime session configuration naming *voice*.

    Args:
        voice: The value sent as `audio.output.voice`.

    Returns:
        The validated session configuration.
    """
    return RealtimeSessionConfig.model_validate(
        {"type": "realtime", "audio": {"output": {"voice": voice}}}
    )


class TestCustomVoiceObject:
    """A voice may be named as a string or as a `{"id": ...}` custom voice object.

    The OpenAI specification types `voice` as a union of a built-in name, any
    string, and an object carrying the identifier of a custom voice, on the
    speech, chat-completion audio and Realtime audio surfaces alike. The
    gateway serves the object form by resolving it to the identifier it names.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/types/openai.py:VoiceName
    """

    @pytest.mark.parametrize("surface", SDK_VOICE_ALIASES)
    def test_the_object_form_is_part_of_the_upstream_contract(
        self, surface: str
    ) -> None:
        """Every upstream `voice` accepts the object form beside the plain name.

        The assertion is made against the installed SDK rather than against a
        remembered specification, so an upstream that stopped typing the object
        form would fail here instead of leaving a divergence behind.

        Ref: https://github.com/openai/openai-python
             https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
        """
        upstream = TypeAdapter(SDK_VOICE_ALIASES[surface])

        assert upstream.validate_python({"id": "voice_1234"}) == {"id": "voice_1234"}
        assert upstream.validate_python("alloy") == "alloy"

    def test_speech_resolves_the_object_form_to_the_named_voice(self) -> None:
        """`/v1/audio/speech` reads the same voice from either form.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai_audio.py:SpeechCreateParams
        """
        assert _speech({"id": "Amy"}).voice == "Amy"
        assert _speech({"id": "Amy"}) == _speech("Amy")

    def test_chat_completion_audio_resolves_the_object_form(self) -> None:
        """A chat completion asking for audio reads the same voice from either form.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai_chat_completions.py:ChatCompletionAudioParam
        """
        assert _chat_audio({"id": "Amy"}).voice == "Amy"
        assert _chat_audio({"id": "Amy"}) == _chat_audio("Amy")

    def test_a_realtime_session_resolves_the_object_form(self) -> None:
        """A Realtime session configuration reads the same voice from either form.

        The configuration reaches this shape from `session.update` and from the
        body minting a client secret, so both are covered by validating it.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai_realtime.py:AudioOutputConfig
        """
        assert _session({"id": "Amy"}).audio.output.voice == "Amy"
        assert _session({"id": "Amy"}) == _session("Amy")

    @pytest.mark.parametrize("voice", UNUSABLE_VOICE_OBJECTS)
    @pytest.mark.parametrize("build", [_speech, _chat_audio, _session])
    def test_an_object_naming_no_voice_is_refused(
        self, build: Callable[[object], BaseModel], voice: dict[str, Any]
    ) -> None:
        """An object carrying no usable `id` is refused, never quietly dropped.

        `id` is required upstream, so a body without one is invalid there too;
        on the Realtime surface, where the field is optional, accepting it would
        silently open the session with the deployment's default voice instead.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai.py:VoiceName
        """
        with pytest.raises(ValidationError) as exc_info:
            build(voice)

        error = exc_info.value.errors()[0]
        assert error["type"] == "string_type"
        assert error["loc"][-1] == "voice"

    @pytest.mark.parametrize(
        ("model", "path"),
        [
            (SpeechCreateParams, ("voice",)),
            (ChatCompletionAudioParam, ("voice",)),
            (RealtimeSessionConfig, ("audio", "output", "voice")),
        ],
    )
    def test_the_published_schema_keeps_the_voice_a_plain_string(
        self, model: type[BaseModel], path: tuple[str, ...]
    ) -> None:
        """The advertised schema stays a string, so no agent reads an object branch.

        These fields become MCP tool schemas: an agent reads them on every
        session, and a widened union would put a second branch in front of it
        for a form it has no identifier to fill.

        Ref: stdapi/mcp.py:mount_mcp
             stdapi/types/openai.py:VoiceName
        """
        schema = model.model_json_schema()
        defs = schema.get("$defs", {})
        for name in path[:-1]:
            ref = schema["properties"][name].get("$ref", "")
            schema = defs[ref.rsplit("/", 1)[-1]]
        field = schema["properties"][path[-1]]

        types = (
            {branch["type"] for branch in field["anyOf"]}
            if "anyOf" in field
            else {field["type"]}
        )
        assert types <= {"string", "null"}, f"{path[-1]} is no longer a plain string"
        assert "string" in types
