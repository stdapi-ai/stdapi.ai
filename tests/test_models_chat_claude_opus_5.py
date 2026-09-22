"""Claude Opus 5.x on the chat routes: always-on reasoning, computer use, forced tool choice.

Opus 5.5 always reasons (adaptive thinking cannot be disabled) and refuses a forced
``tool_choice``, on Bedrock and on the official Anthropic API alike. Bedrock serves
the ``computer_20251124`` tool on Opus 5 and 5.5; the official API serves it on
Opus 5 only.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-5-5.html
     https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
     stdapi/models/chat/anthropic_claude_opus_5.py:ChatModel
"""

import re
from typing import TYPE_CHECKING, Any

import pytest
from anthropic import Anthropic
from anthropic import BadRequestError as AnthropicBadRequestError
from openai import BadRequestError, OpenAI

if TYPE_CHECKING:
    from anthropic.types import ToolChoiceParam, ToolParam
    from openai.types.chat import (
        ChatCompletionFunctionToolParam,
        ChatCompletionToolChoiceOptionParam,
    )

#: Opus 5.5 model ID on the gateway.
_OPUS_5_5 = "anthropic.claude-opus-5-5"

#: Opus 5 model ID on the gateway.
_OPUS_5 = "anthropic.claude-opus-5"

#: Computer use tool as the Anthropic API declares it for Claude 4.6 and later.
_COMPUTER_TOOL: dict[str, object] = {
    "type": "computer_20251124",
    "name": "computer",
    "display_width_px": 1024,
    "display_height_px": 768,
}

#: Input schema of the client function the forced ``tool_choice`` tests name.
_WEATHER_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"city": {"type": "string"}},
}

#: That client function, declared on the Messages route.
_WEATHER_TOOL: ToolParam = {"name": "get_weather", "input_schema": _WEATHER_SCHEMA}

#: That client function, declared on the Chat Completions route.
_CHAT_WEATHER_TOOL: ChatCompletionFunctionToolParam = {
    "type": "function",
    "function": {"name": "get_weather", "parameters": _WEATHER_SCHEMA},
}


def _anthropic_model_id(
    model_id: str, *, use_official_api: bool, is_bedrock_direct: bool
) -> str:
    """Return a gateway model ID in the format the Anthropic target expects.

    Args:
        model_id: Model ID on the gateway.
        use_official_api: Whether the target is the official API.
        is_bedrock_direct: Whether that official target is Bedrock itself.

    Returns:
        The model ID to send.
    """
    if not use_official_api:
        return model_id
    if is_bedrock_direct:
        return f"global.{model_id}"
    return re.sub(r"-v\d+(?::\d+)?$", "", model_id.removeprefix("anthropic."))


@pytest.fixture
def opus_5_5_anthropic_model(use_official_api: bool, is_bedrock_direct: bool) -> str:
    """Return the Opus 5.5 model ID in the format the Anthropic target expects."""
    return _anthropic_model_id(
        _OPUS_5_5,
        use_official_api=use_official_api,
        is_bedrock_direct=is_bedrock_direct,
    )


@pytest.mark.expensive
def test_disabled_thinking_on_opus_5_5(
    anthropic_client: Anthropic, opus_5_5_anthropic_model: str, use_official_api: bool
) -> None:
    """``thinking: {"type": "disabled"}`` on Opus 5.5: a 400 upstream, served by the gateway.

    The official API answers ``invalid_request_error``. The gateway drops the
    disabled configuration with a warning, as for the other always-reasoning
    Claude families, and answers with the adaptive default; ``max_tokens`` leaves
    room for that thinking. The Chat Completions side (``reasoning_effort="none"``)
    is covered by ``test_reasoning_effort_none_explicit_disable_all_models``.

    Ref: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
         stdapi/models/chat/anthropic_claude_opus_5.py:ChatModel.REASONING_DISABLE_SUPPORTED
    """
    if use_official_api:
        with pytest.raises(AnthropicBadRequestError) as excinfo:
            anthropic_client.messages.create(
                model=opus_5_5_anthropic_model,
                max_tokens=4096,
                messages=[{"role": "user", "content": "Reply with OK."}],
                thinking={"type": "disabled"},
            )
        assert excinfo.value.status_code == 400
        body = excinfo.value.body
        assert isinstance(body, dict)
        assert body["error"]["type"] == "invalid_request_error"
        return

    response = anthropic_client.messages.create(
        model=opus_5_5_anthropic_model,
        max_tokens=4096,
        messages=[{"role": "user", "content": "Reply with OK."}],
        thinking={"type": "disabled"},
    )
    assert response.type == "message"
    assert any(block.type == "text" for block in response.content)
    assert response.usage.output_tokens > 0


class TestForcedToolChoiceRejected:
    """A forced ``tool_choice`` on Opus 5.5 is refused with a clean 400.

    Forcing a tool is the request itself, so it is not silently downgraded to
    ``auto``: the client gets an ``invalid_request_error`` naming ``tool_choice``,
    as the official API returns, and on the OpenAI routes a message free of
    Anthropic terms naming ``auto`` as the way forward. The request is refused before
    inference and bills nothing, hence no ``expensive`` marker.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use#forcing-tool-use
         stdapi/models/chat/anthropic_claude_opus_5.py:ChatModel._req_configure_tools
    """

    @pytest.mark.parametrize(
        "tool_choice",
        [{"type": "any"}, {"type": "tool", "name": "get_weather"}],
        ids=["any", "tool"],
    )
    def test_messages_route(
        self,
        anthropic_client: Anthropic,
        opus_5_5_anthropic_model: str,
        tool_choice: ToolChoiceParam,
    ) -> None:
        """``tool_choice`` ``any`` and ``tool`` are rejected on the Messages route.

        Ref: https://platform.claude.com/docs/en/api/messages
        """
        with pytest.raises(AnthropicBadRequestError) as excinfo:
            anthropic_client.messages.create(
                model=opus_5_5_anthropic_model,
                max_tokens=64,
                messages=[{"role": "user", "content": "Weather in Lisbon?"}],
                tools=[_WEATHER_TOOL],
                tool_choice=tool_choice,
            )

        assert excinfo.value.status_code == 400
        body = excinfo.value.body
        assert isinstance(body, dict)
        assert body["error"]["type"] == "invalid_request_error"
        assert "tool_choice" in body["error"]["message"]

    @pytest.mark.gateway("Anthropic Claude is not served by the official OpenAI API")
    @pytest.mark.parametrize(
        "tool_choice",
        ["required", {"type": "function", "function": {"name": "get_weather"}}],
        ids=["required", "function"],
    )
    def test_chat_completions_route(
        self, openai_client: OpenAI, tool_choice: ChatCompletionToolChoiceOptionParam
    ) -> None:
        """``tool_choice`` ``required`` and a named function are rejected on Chat Completions.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
        """
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.chat.completions.create(
                model=_OPUS_5_5,
                messages=[{"role": "user", "content": "Weather in Lisbon?"}],
                max_completion_tokens=64,
                tools=[_CHAT_WEATHER_TOOL],
                tool_choice=tool_choice,
            )

        assert excinfo.value.status_code == 400
        assert excinfo.value.type == "invalid_request_error"
        assert "'auto'" in excinfo.value.message
        assert '"any"' not in excinfo.value.message, "no Anthropic wording"

    @pytest.mark.gateway("Anthropic Claude is not served by the official OpenAI API")
    def test_responses_route(self, openai_client: OpenAI) -> None:
        """``tool_choice: "required"`` is rejected on the Responses route.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.responses.create(
                model=_OPUS_5_5,
                input="Weather in Lisbon?",
                max_output_tokens=64,
                tools=[
                    {
                        "type": "function",
                        "name": "get_weather",
                        "parameters": _WEATHER_SCHEMA,
                        "strict": False,
                    }
                ],
                tool_choice="required",
            )

        assert excinfo.value.status_code == 400
        assert "'auto'" in excinfo.value.message


@pytest.mark.expensive
@pytest.mark.parametrize("model", [_OPUS_5, _OPUS_5_5], ids=["opus-5", "opus-5-5"])
def test_computer_use_tool_on_the_messages_route(
    anthropic_client: Anthropic,
    use_official_api: bool,
    is_bedrock_direct: bool,
    model: str,
) -> None:
    """``computer_20251124`` with its beta flag, on Opus 5 and Opus 5.5.

    Opus 5 serves it on every target. On Opus 5.5 the Anthropic API refuses the
    tool type (400 ``invalid_request_error`` naming it), while Bedrock runs it, so
    the gateway and a Bedrock-direct client are served.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
         stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
    """
    kwargs: dict[str, Any] = {
        "model": _anthropic_model_id(
            model,
            use_official_api=use_official_api,
            is_bedrock_direct=is_bedrock_direct,
        ),
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "tools": [_COMPUTER_TOOL],
        "extra_headers": {"anthropic-beta": "computer-use-2025-11-24"},
    }
    if model == _OPUS_5_5 and use_official_api and not is_bedrock_direct:
        with pytest.raises(AnthropicBadRequestError) as excinfo:
            anthropic_client.messages.create(**kwargs)
        body = excinfo.value.body
        assert isinstance(body, dict)
        assert body["error"]["type"] == "invalid_request_error"
        assert "computer_20251124" in body["error"]["message"]
        return

    response = anthropic_client.messages.create(**kwargs)

    assert response.type == "message"
    assert response.stop_reason in {"end_turn", "max_tokens", "tool_use"}
    assert response.usage.output_tokens > 0


@pytest.mark.gateway("Anthropic Claude is not served by the official OpenAI API")
@pytest.mark.expensive
def test_bare_computer_tool_is_served_on_opus_5(openai_client: OpenAI) -> None:
    """A bare ``computer`` function reaches Opus 5 as its computer use tool and is served.

    The gateway promotes it to ``computer_20251124`` with its beta flag, which is
    the only version Opus 5 accepts; ``test_computer_tool_type_matches_the_model_generation``
    pins the promotion itself.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
         stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
    """
    resp = openai_client.chat.completions.create(
        model="anthropic.claude-opus-5",
        messages=[{"role": "user", "content": "Take a screenshot."}],
        max_completion_tokens=1024,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "computer",
                    "parameters": {
                        "type": "object",
                        "display_width_px": 1024,
                        "display_height_px": 768,
                    },
                },
            }
        ],
    )

    assert resp.object == "chat.completion"
    assert resp.choices[0].finish_reason in {"stop", "length", "tool_calls"}
    assert resp.usage is not None
    assert resp.usage.completion_tokens > 0
