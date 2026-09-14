"""Tests for scoping guardrail evaluation to the most recent conversation turns.

``AWS_BEDROCK_GUARDRAIL_SCOPE_TURNS`` limits what an Amazon Bedrock guardrail
evaluates on the chat routes: the text of the N most recent user turns is
tagged, and Amazon Bedrock then charges and evaluates those turns only. The
setting is off by default, because scoping is a real detection loss -- content
that appears only in the conversation history, including an attack built across
several turns, stops being seen.

Tagging rules the backend imposes, each pinned by a test below: a blank tagged
text is rejected outright, tool results and documents are never evaluated
anyway, and an image can only be tagged as inline bytes -- so only non-empty
text blocks of user messages are tagged, and every other block is left as the
very object a later stage resolves in place.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-how.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-enforcements.html
     stdapi/models/chat/_default.py:_scope_guardrail_content
     stdapi/config.py:_Settings
"""

from typing import TYPE_CHECKING, Any, ClassVar

import pytest
from sse_starlette import EventSourceResponse
from starlette.datastructures import Headers

from stdapi.aws_bedrock import GUARDRAIL_CONFIG_VAR, set_guardrail_configuration
from stdapi.config import SETTINGS
from stdapi.models.chat import twelvelabs_pegasus as pegasus
from stdapi.models.chat._default import ChatModel, _scope_guardrail_content
from stdapi.types.anthropic_messages import MessageCreateParams
from stdapi.types.openai_chat_completions import CompletionCreateParams
from stdapi.types.openai_completions import (
    CompletionCreateParams as TextCompletionCreateParams,
)
from stdapi.types.openai_responses import ResponseCreateParams

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator

    from openai import OpenAI
    from types_aiobotocore_bedrock_runtime.type_defs import MessageTypeDef

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Chat model whose class applies the shared Converse request builder.
_MODEL = "amazon.nova-2-lite-v1:0"

#: Two-turn history plus a new question, in OpenAI chat message form.
_HISTORY: list[dict[str, Any]] = [
    {"role": "user", "content": "first question"},
    {"role": "assistant", "content": "first answer"},
    {"role": "user", "content": "second question"},
]

#: The same three turns, already translated to Converse blocks and untagged.
_UNTAGGED_HISTORY: list[dict[str, Any]] = [
    {"role": "user", "content": [{"text": "first question"}]},
    {"role": "assistant", "content": [{"text": "first answer"}]},
    {"role": "user", "content": [{"text": "second question"}]},
]


def _tagged(text: str) -> dict[str, Any]:
    """Return the Converse block a scoped text block is expected to become.

    Args:
        text: Text of the original block.

    Returns:
        The ``guardContent`` block wrapping *text*.
    """
    return {"guardContent": {"text": {"text": text}}}


async def _empty_stream() -> AsyncGenerator[dict[str, Any]]:
    """Yield no Converse stream event."""
    return
    yield {}


def _install_capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace both Converse calls with recorders of the request body.

    Args:
        monkeypatch: Patcher scoped to the test.

    Returns:
        Mapping updated in place with the Converse request body.
    """
    captured: dict[str, Any] = {}

    async def fake_converse(
        _self: ChatModel, request: ConverseRequestBaseTypeDef
    ) -> dict[str, Any]:
        captured.update(request)
        return {
            "output": {"message": {"role": "assistant", "content": []}},
            "stopReason": "end_turn",
            "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
        }

    async def fake_converse_stream(
        _self: ChatModel, request: ConverseRequestBaseTypeDef
    ) -> dict[str, Any]:
        captured.update(request)
        return {"stream": _empty_stream()}

    monkeypatch.setattr(ChatModel, "converse", fake_converse)
    monkeypatch.setattr(ChatModel, "converse_stream", fake_converse_stream)
    return captured


async def _chat_request(
    monkeypatch: pytest.MonkeyPatch,
    messages: list[dict[str, Any]],
    *,
    stream: bool = False,
) -> dict[str, Any]:
    """Build a Chat Completions request and return the Converse body it sends.

    Args:
        monkeypatch: Patcher scoped to the test.
        messages: OpenAI chat messages to send.
        stream: Whether to take the streaming path.

    Returns:
        The captured Converse request body.
    """
    captured = _install_capture(monkeypatch)
    request = CompletionCreateParams.model_validate(
        {"model": "test-model", "messages": messages, "stream": stream}
    )
    result = await ChatModel(_MODEL).create_completion(request, "chatcmpl-1", 0)
    assert isinstance(result, EventSourceResponse) is stream
    return captured


async def _responses_request(
    monkeypatch: pytest.MonkeyPatch, input_items: list[dict[str, Any]] | str
) -> dict[str, Any]:
    """Build a Responses request and return the Converse body it sends.

    Args:
        monkeypatch: Patcher scoped to the test.
        input_items: Responses API ``input`` value.

    Returns:
        The captured Converse request body.
    """
    captured = _install_capture(monkeypatch)
    request = ResponseCreateParams.model_validate(
        {"model": "test-model", "input": input_items}
    )
    await ChatModel(_MODEL).create_response(request, "resp-1", 0.0)
    return captured


async def _anthropic_request(
    monkeypatch: pytest.MonkeyPatch, messages: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build an Anthropic Messages request and return the Converse body it sends.

    Args:
        monkeypatch: Patcher scoped to the test.
        messages: Anthropic messages to send.

    Returns:
        The captured Converse request body.
    """
    captured = _install_capture(monkeypatch)
    request = MessageCreateParams.model_validate(
        {"model": "test-model", "max_tokens": 16, "messages": messages}
    )
    await ChatModel(_MODEL).create_message(request, "msg-1")
    return captured


async def _text_completion_request(
    monkeypatch: pytest.MonkeyPatch, prompt: str
) -> dict[str, Any]:
    """Build a legacy Completions request and return the Converse body it sends.

    Args:
        monkeypatch: Patcher scoped to the test.
        prompt: Prompt to complete.

    Returns:
        The captured Converse request body.
    """
    captured = _install_capture(monkeypatch)
    request = TextCompletionCreateParams.model_validate(
        {"model": "test-model", "prompt": prompt}
    )
    await ChatModel(_MODEL).create_text_completion(request, "cmpl-1", 0)
    return captured


@pytest.fixture
def deployment_guardrail(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Configure a deployment guardrail and resolve it onto the test's context."""
    monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", "gr123")
    monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", "1")
    token = GUARDRAIL_CONFIG_VAR.set(None)  # type: ignore[arg-type]
    set_guardrail_configuration(Headers({}))
    try:
        yield
    finally:
        GUARDRAIL_CONFIG_VAR.reset(token)


@pytest.fixture
def no_guardrail(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Leave the request without any guardrail, as an unconfigured deployment does."""
    monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", None)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", None)
    token = GUARDRAIL_CONFIG_VAR.set(None)  # type: ignore[arg-type]
    set_guardrail_configuration(Headers({}))
    try:
        yield
    finally:
        GUARDRAIL_CONFIG_VAR.reset(token)


@pytest.fixture
def scope_one_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """Scope guardrail evaluation to the most recent user turn."""
    monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_scope_turns", 1)


@pytest.mark.usefixtures("deployment_guardrail")
class TestScopeIsOffByDefault:
    """Unset, the setting leaves every turn of every route evaluated.

    Scoping weakens detection, so a deployment that did not ask for it keeps
    sending the whole conversation for evaluation, exactly as before.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html
         stdapi/models/chat/_default.py:ChatModel._prepare_converse_request
    """

    def test_the_setting_defaults_to_unset(self) -> None:
        """Nothing is scoped until an operator sets a number of turns."""
        assert (
            type(SETTINGS).model_fields["aws_bedrock_guardrail_scope_turns"].default
            is None
        )

    async def test_chat_completions_sends_the_whole_conversation(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """Chat Completions tags nothing while the setting is unset."""
        del request_log
        captured = await _chat_request(monkeypatch, _HISTORY)
        assert captured["messages"] == _UNTAGGED_HISTORY
        assert captured["guardrailConfig"]["guardrailIdentifier"] == "gr123"

    async def test_responses_sends_the_whole_conversation(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """The Responses API tags nothing while the setting is unset."""
        del request_log
        captured = await _responses_request(monkeypatch, _HISTORY)
        assert captured["messages"] == _UNTAGGED_HISTORY

    async def test_anthropic_messages_sends_the_whole_conversation(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """Anthropic Messages tags nothing while the setting is unset."""
        del request_log
        captured = await _anthropic_request(monkeypatch, _HISTORY)
        assert captured["messages"] == _UNTAGGED_HISTORY

    async def test_completions_sends_the_prompt_untagged(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """The legacy Completions route tags nothing while the setting is unset."""
        del request_log
        captured = await _text_completion_request(monkeypatch, "a question")
        assert captured["messages"] == [
            {"role": "user", "content": [{"text": "a question"}]}
        ]


@pytest.mark.usefixtures("deployment_guardrail", "scope_one_turn")
class TestScopedToTheLatestTurn:
    """Set to one turn, only the latest user turn is submitted for evaluation.

    The history the client replays on every request -- its own earlier turns and
    the answers it got back -- stops being evaluated and stops being charged,
    on every dialect the chat routes speak.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html
         stdapi/models/chat/_default.py:_scope_guardrail_content
    """

    async def test_chat_completions_tags_the_latest_user_turn(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """Only the last user message is tagged; the guardrail still applies."""
        del request_log
        captured = await _chat_request(monkeypatch, _HISTORY)
        assert captured["messages"] == [
            *_UNTAGGED_HISTORY[:2],
            {"role": "user", "content": [_tagged("second question")]},
        ]
        assert captured["guardrailConfig"]["guardrailIdentifier"] == "gr123"

    async def test_responses_tags_the_latest_user_turn(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """The Responses API scopes the conversation it replays."""
        del request_log
        captured = await _responses_request(monkeypatch, _HISTORY)
        assert captured["messages"] == [
            *_UNTAGGED_HISTORY[:2],
            {"role": "user", "content": [_tagged("second question")]},
        ]

    async def test_anthropic_messages_tags_the_latest_user_turn(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """Anthropic Messages scopes the conversation it replays."""
        del request_log
        captured = await _anthropic_request(monkeypatch, _HISTORY)
        assert captured["messages"] == [
            *_UNTAGGED_HISTORY[:2],
            {"role": "user", "content": [_tagged("second question")]},
        ]

    async def test_completions_tags_the_prompt(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A single-turn Completions prompt is the turn that gets evaluated."""
        del request_log
        captured = await _text_completion_request(monkeypatch, "a question")
        assert captured["messages"] == [
            {"role": "user", "content": [_tagged("a question")]}
        ]

    async def test_the_system_prompt_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """System instructions are sent as they were, and stay outside the scope."""
        del request_log
        captured = await _chat_request(
            monkeypatch, [{"role": "system", "content": "be brief"}, *_HISTORY]
        )
        assert captured["system"] == [{"text": "be brief"}]

    async def test_streaming_sends_the_same_scoped_conversation(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A streamed request is scoped exactly like a buffered one."""
        del request_log
        captured = await _chat_request(monkeypatch, _HISTORY, stream=True)
        assert captured["messages"] == [
            *_UNTAGGED_HISTORY[:2],
            {"role": "user", "content": [_tagged("second question")]},
        ]

    async def test_an_assistant_prefill_is_left_alone(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A trailing assistant turn is not a user turn, so the question is tagged."""
        del request_log
        captured = await _anthropic_request(
            monkeypatch,
            [
                {"role": "user", "content": "what is the capital of France?"},
                {"role": "assistant", "content": "The capital is"},
            ],
        )
        assert captured["messages"] == [
            {"role": "user", "content": [_tagged("what is the capital of France?")]},
            {"role": "assistant", "content": [{"text": "The capital is"}]},
        ]

    async def test_a_tool_result_turn_falls_back_to_the_question(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """In a tool loop the human's instruction is what stays evaluated.

        The message carrying the tool output has no text of its own, and a
        blank tag is rejected by the backend, so the scope reaches back to the
        last message that does carry text. Tool blocks travel untouched.
        """
        del request_log
        captured = await _responses_request(
            monkeypatch,
            [
                {"role": "user", "content": "what is the weather?"},
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "weather",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "sunny",
                },
            ],
        )
        messages = captured["messages"]
        assert messages[0] == {
            "role": "user",
            "content": [_tagged("what is the weather?")],
        }
        assert messages[1]["content"][0]["toolUse"]["name"] == "weather"
        assert messages[-1]["content"][0]["toolResult"]["toolUseId"] == "call_1"
        assert "guardContent" not in messages[-1]["content"][0]

    async def test_a_blank_text_block_is_not_tagged(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """An empty text block travels as it is: the backend rejects a blank tag."""
        del request_log
        captured = await _chat_request(
            monkeypatch,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ""},
                        {"type": "text", "text": "the question"},
                    ],
                }
            ],
        )
        assert captured["messages"] == [
            {"role": "user", "content": [{"text": ""}, _tagged("the question")]}
        ]

    async def test_an_image_block_is_left_in_place(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """Non-text blocks keep their position and their content, unwrapped."""
        del request_log
        image = "data:image/png;base64,iVBORw0KGgo="
        captured = await _chat_request(
            monkeypatch,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image_url", "image_url": {"url": image}},
                        {"type": "text", "text": "at this"},
                    ],
                }
            ],
        )
        (message,) = captured["messages"]
        assert message["content"][0] == _tagged("look")
        assert message["content"][1]["image"]["format"] == "png"
        assert message["content"][2] == _tagged("at this")


@pytest.mark.usefixtures("deployment_guardrail")
class TestScopeWidth:
    """The number of turns set is the number of user turns evaluated.

    Ref: stdapi/models/chat/_default.py:_scope_guardrail_content
    """

    async def test_two_turns_reach_back_past_the_assistant_answer(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """Two turns tags both user messages and neither assistant one."""
        del request_log
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_scope_turns", 2)
        captured = await _chat_request(
            monkeypatch,
            [
                {"role": "user", "content": "oldest question"},
                {"role": "assistant", "content": "oldest answer"},
                *_HISTORY,
            ],
        )
        assert captured["messages"] == [
            {"role": "user", "content": [{"text": "oldest question"}]},
            {"role": "assistant", "content": [{"text": "oldest answer"}]},
            {"role": "user", "content": [_tagged("first question")]},
            {"role": "assistant", "content": [{"text": "first answer"}]},
            {"role": "user", "content": [_tagged("second question")]},
        ]

    async def test_a_scope_wider_than_the_conversation_tags_all_of_it(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """Asking for more turns than were sent is not an error."""
        del request_log
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_scope_turns", 5)
        captured = await _chat_request(monkeypatch, _HISTORY)
        assert captured["messages"] == [
            {"role": "user", "content": [_tagged("first question")]},
            {"role": "assistant", "content": [{"text": "first answer"}]},
            {"role": "user", "content": [_tagged("second question")]},
        ]

    def test_zero_turns_is_refused_by_the_settings_model(self) -> None:
        """A scope of zero would evaluate nothing at all, so it is not accepted."""
        field = type(SETTINGS).model_fields["aws_bedrock_guardrail_scope_turns"]
        assert any(getattr(item, "ge", None) == 1 for item in field.metadata)


@pytest.mark.usefixtures("deployment_guardrail", "scope_one_turn")
class TestModelsThatReadTheirOwnPrompt:
    """A model that rebuilds its request from the messages is never scoped.

    Its guardrail is applied by a different backend mechanism, which does not
    read the tag, and its prompt is recovered from the text blocks -- so tagging
    them would send an empty prompt for no gain in return.

    Ref: stdapi/models/chat/twelvelabs_pegasus.py:ChatModel
         stdapi/models/chat/_default.py:ChatModel.GUARDRAIL_SCOPE_SUPPORTED
    """

    async def test_the_video_model_keeps_its_prompt(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """The prompt stays a plain text block the model can still read back."""
        del request_log
        captured = _install_capture(monkeypatch)
        request = CompletionCreateParams.model_validate(
            {"model": "test-model", "messages": [_HISTORY[-1]]}
        )
        await pegasus.ChatModel("twelvelabs.pegasus-1-2-v1:0").create_completion(
            request, "chatcmpl-1", 0
        )
        assert captured["messages"] == [_UNTAGGED_HISTORY[-1]]
        assert captured["guardrailConfig"]["guardrailIdentifier"] == "gr123"


class TestScopeAppliesToEveryGuardrailSource:
    """Whatever put a guardrail on the request, the operator's scope applies.

    A client cannot narrow the evaluation by selecting its own guardrail, and a
    request with no guardrail at all is never rewritten.

    Ref: stdapi/aws_bedrock.py:set_guardrail_configuration
         stdapi/models/chat/_default.py:ChatModel._prepare_converse_request
    """

    @pytest.mark.usefixtures("no_guardrail", "scope_one_turn")
    async def test_nothing_is_tagged_without_a_guardrail(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """With no guardrail there is nothing to scope, so the request is untouched."""
        del request_log
        captured = await _chat_request(monkeypatch, _HISTORY)
        assert captured["messages"] == _UNTAGGED_HISTORY
        assert not captured.get("guardrailConfig")

    @pytest.mark.usefixtures("scope_one_turn")
    async def test_a_request_selected_guardrail_is_scoped_too(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A guardrail chosen by request headers is evaluated on the scope as well."""
        del request_log
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", None)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", None)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", True)
        token = GUARDRAIL_CONFIG_VAR.set(None)  # type: ignore[arg-type]
        try:
            set_guardrail_configuration(
                Headers(
                    {
                        "X-Amzn-Bedrock-GuardrailIdentifier": "gr-request",
                        "X-Amzn-Bedrock-GuardrailVersion": "DRAFT",
                    }
                )
            )
            captured = await _chat_request(monkeypatch, _HISTORY)
        finally:
            GUARDRAIL_CONFIG_VAR.reset(token)
        assert captured["guardrailConfig"]["guardrailIdentifier"] == "gr-request"
        assert captured["messages"][-1] == {
            "role": "user",
            "content": [_tagged("second question")],
        }


class TestScopedBlocksKeepTheirIdentity:
    """A block left untagged is the very object later stages still write into.

    Media blocks reach the request builder with an empty ``source`` and are
    filled in once the region is known, so replacing them with copies would
    send an empty image.

    Ref: stdapi/models/__init__.py:ModelBase._prepare_converse_request_for_region
         stdapi/models/chat/_default.py:_scope_guardrail_content
    """

    def test_untagged_blocks_are_not_copied(self) -> None:
        """The image block in the tagged message is the same object as before."""
        image_block: dict[str, Any] = {"image": {"format": "png", "source": {}}}
        messages: list[MessageTypeDef] = [
            {
                "role": "user",
                "content": [{"text": "look"}, image_block],  # type: ignore[list-item]
            }
        ]

        _scope_guardrail_content(messages, 1)

        content = messages[0]["content"]
        assert content[0] == _tagged("look")
        assert content[1] is image_block

    def test_a_message_without_text_is_skipped_entirely(self) -> None:
        """A user message carrying no text does not consume the scope."""
        messages: list[MessageTypeDef] = [
            {"role": "user", "content": [{"text": "the question"}]},
            {
                "role": "user",
                "content": [
                    {"toolResult": {"toolUseId": "call_1", "content": [{"text": "42"}]}}
                ],
            },
        ]

        _scope_guardrail_content(messages, 1)

        assert messages[0]["content"] == [_tagged("the question")]
        assert messages[1]["content"][0]["toolResult"]["toolUseId"] == "call_1"


@pytest.mark.slow
@pytest.mark.xdist_group("moderations_guardrail")
class TestScopeAgainstALiveGuardrail:
    """Against a real guardrail, the scope decides what is still detected.

    This is the trade-off the setting exists to expose: a forbidden word left
    in the replayed history blocks every later request today, and stops doing so
    once evaluation is scoped to the latest turn. The same request is sent both
    times, so nothing but the scope explains the difference.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html
         stdapi/models/chat/_default.py:_scope_guardrail_content
    """

    #: Conversation whose forbidden word sits in the history, not in the new turn.
    _MESSAGES: ClassVar[list[dict[str, Any]]] = [
        {"role": "user", "content": "Remember the code word BLOCKWORDXYZ."},
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    @staticmethod
    def _headers(guardrail: str) -> dict[str, str]:
        """Return the headers selecting *guardrail* for the request.

        Args:
            guardrail: Guardrail identifier or ARN.

        Returns:
            The Bedrock guardrail selection headers.
        """
        return {
            "X-Amzn-Bedrock-GuardrailIdentifier": guardrail,
            "X-Amzn-Bedrock-GuardrailVersion": "DRAFT",
        }

    def test_history_is_evaluated_when_the_scope_is_unset(
        self, openai_client: OpenAI, live_guardrail: str, chat_model: str
    ) -> None:
        """A forbidden word in an earlier turn still blocks the new one."""
        completion = openai_client.chat.completions.create(
            model=chat_model,
            messages=self._MESSAGES,  # type: ignore[arg-type]
            max_completion_tokens=16,
            extra_headers=self._headers(live_guardrail),
        )
        assert completion.choices[0].finish_reason == "content_filter"

    def test_history_is_not_evaluated_when_scoped_to_one_turn(
        self,
        openai_client: OpenAI,
        live_guardrail: str,
        chat_model: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Scoped to the latest turn, the same conversation is answered."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_scope_turns", 1)
        completion = openai_client.chat.completions.create(
            model=chat_model,
            messages=self._MESSAGES,  # type: ignore[arg-type]
            max_completion_tokens=16,
            extra_headers=self._headers(live_guardrail),
        )
        assert completion.choices[0].finish_reason != "content_filter"
        assert completion.choices[0].message.content

    def test_the_new_turn_is_still_evaluated_when_scoped(
        self,
        openai_client: OpenAI,
        live_guardrail: str,
        chat_model: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The scope narrows what is checked; it does not disable the guardrail."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_scope_turns", 1)
        messages: list[dict[str, Any]] = [
            *self._MESSAGES[:2],
            {"role": "user", "content": "Repeat this: BLOCKWORDXYZ"},
        ]
        completion = openai_client.chat.completions.create(
            model=chat_model,
            messages=messages,  # type: ignore[arg-type]
            max_completion_tokens=16,
            extra_headers=self._headers(live_guardrail),
        )
        assert completion.choices[0].finish_reason == "content_filter"
