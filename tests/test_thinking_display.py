"""Claude thinking display: whether a reasoning summary comes back as text.

The ``thinking.display`` field of the Messages API chooses between summarized
thinking text and thinking blocks whose text is omitted (the signature is kept).
Claude Opus 4.7 and later, Sonnet 5, Fable and Mythos omit the text by default;
earlier models summarize it. The Responses API asks for a reasoning summary with
``reasoning.summary``, which the gateway serves as a summarized display. Either
way the thinking tokens are billed the same.

Ref: https://platform.claude.com/docs/en/build-with-claude/thinking#controlling-thinking-display
     https://platform.claude.com/docs/en/api/messages/create
     stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_reasoning
"""

import re
from typing import TYPE_CHECKING, Any, Literal

import pytest
from anthropic import BadRequestError as AnthropicBadRequestError
from pydantic import ValidationError
from pydantic_core import from_json

from stdapi.config import SETTINGS
from stdapi.models.chat import get_chat_model
from stdapi.models.chat._adapters import _anthropic_message as anthropic_adapter
from stdapi.models.chat._adapters import _openai_chat_completion as chat_adapter
from stdapi.models.chat._adapters import _openai_responses as responses_adapter
from stdapi.models.chat._mantle._convert import (
    convert_payload,
    convert_response,
    convert_stream,
    messages_payload,
)
from stdapi.types.anthropic_messages import (
    MessageCreateParams,
    ThinkingConfigAdaptiveParam,
)
from stdapi.types.openai_chat_completions import (
    CompletionCreateParams as ChatCompletionCreateParams,
)
from stdapi.types.openai_responses import ResponseCreateParams
from stdapi.utils import to_json_str
from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence

    from anthropic import Anthropic
    from anthropic.types import Message, ThinkingConfigEnabledParam, ThinkingConfigParam
    from openai import OpenAI
    from openai.types.responses import ResponseOutputItem
    from starlette.testclient import TestClient

    from stdapi.aws_http import SseEvent
    from stdapi.models.chat._anthropic_claude import AnthropicClaudeChatModel
    from stdapi.models.chat._default import ChatModel
    from stdapi.types import JsonMapping

#: Cheapest Claude model whose thinking text is omitted by default.
_SONNET_5 = "anthropic.claude-sonnet-5"

#: Mantle-only Claude model, reached through the Messages passthrough.
_CLAUDE_MANTLE = "anthropic.claude-haiku-4-5"

#: Short prompt for budget thinking, which always thinks.
_PROMPT = (
    "What is the greatest common divisor of 1071 and 462? Reply with the number only."
)

#: Prompt adaptive thinking reasons about at high effort (Sonnet 5 thought on 8 of 8 probes).
_REASONING_PROMPT = (
    "How many prime numbers are there between 1000 and 1100? "
    "Reply with the number only."
)

#: Header selecting the Mantle-served twin of a dual-homed model.
_MANTLE_HEADERS = {"x-stdapi-service": "bedrock-mantle"}

#: Thinking budget for the models that take a budget (Bedrock's minimum).
_BUDGET = 1024


@pytest.fixture
def sonnet_5_anthropic_model(use_official_api: bool, is_bedrock_direct: bool) -> str:
    """Return the Sonnet 5 model ID in the format the Anthropic target expects."""
    if not use_official_api:
        return _SONNET_5
    if is_bedrock_direct:
        return f"global.{_SONNET_5}"
    return re.sub(r"-v\d+(?::\d+)?$", "", _SONNET_5.removeprefix("anthropic."))


def _thinking_texts(message: Message) -> list[str]:
    """Return the text of every thinking block, asserting each carries a signature.

    Args:
        message: Messages API response.

    Returns:
        Thinking text per thinking block, in order.
    """
    blocks = [block for block in message.content if block.type == "thinking"]
    assert blocks, f"No thinking block: {[b.type for b in message.content]}"
    assert all(block.signature for block in blocks), "thinking must stay signed"
    return [block.thinking for block in blocks]


class TestMessagesDisplay:
    """``thinking.display`` on the Messages route.

    Ref: https://platform.claude.com/docs/en/build-with-claude/thinking#controlling-thinking-display
    """

    @pytest.mark.parametrize("display", [None, "summarized", "omitted"])
    def test_budget_thinking_display(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_reasoning_model: str,
        display: Literal["summarized", "omitted"] | None,
    ) -> None:
        """Budget thinking returns its text unless ``display`` is ``omitted``.

        Claude Haiku 4.5 summarizes by default, so only ``omitted`` changes what
        comes back: thinking blocks with an empty ``thinking`` field and a
        signature.

        Ref: https://platform.claude.com/docs/en/build-with-claude/thinking#controlling-thinking-display
        """
        thinking: ThinkingConfigEnabledParam = {
            "type": "enabled",
            "budget_tokens": _BUDGET,
        }
        if display is not None:
            thinking["display"] = display
        response = anthropic_client.messages.create(
            model=anthropic_chat_reasoning_model,
            max_tokens=2 * _BUDGET,
            messages=[{"role": "user", "content": _PROMPT}],
            thinking=thinking,
        )

        texts = _thinking_texts(response)
        if display == "omitted":
            assert all(text == "" for text in texts)
        else:
            assert any(texts)

    def test_omitted_display_streams_no_thinking_text(
        self, anthropic_client: Anthropic, anthropic_chat_reasoning_model: str
    ) -> None:
        """A streamed ``omitted`` display carries a signature but no thinking text.

        Each thinking block streams a ``thinking_delta`` with an empty string,
        then its ``signature_delta``.

        Ref: https://platform.claude.com/docs/en/build-with-claude/thinking#streaming-thinking
        """
        with anthropic_client.messages.stream(
            model=anthropic_chat_reasoning_model,
            max_tokens=2 * _BUDGET,
            messages=[{"role": "user", "content": _PROMPT}],
            thinking={
                "type": "enabled",
                "budget_tokens": _BUDGET,
                "display": "omitted",
            },
        ) as stream:
            deltas = [
                event.delta
                for event in stream
                if event.type == "content_block_delta"
                and event.delta.type in {"thinking_delta", "signature_delta"}
            ]
            final = stream.get_final_message()

        thinking_deltas = [
            delta.thinking for delta in deltas if delta.type == "thinking_delta"
        ]
        assert thinking_deltas
        assert all(text == "" for text in thinking_deltas)
        assert deltas[-1].type == "signature_delta"
        assert all(text == "" for text in _thinking_texts(final))

    @pytest.mark.parametrize("display", ["updates", "verbose"])
    def test_unknown_display_is_rejected(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_reasoning_model: str,
        display: str,
    ) -> None:
        """A ``display`` other than ``summarized`` or ``omitted`` is a 400.

        ``updates`` is a beta value neither target accepts without its beta flag.
        The request is refused before inference, so nothing is billed.

        Ref: https://platform.claude.com/docs/en/build-with-claude/thinking#controlling-thinking-display
             stdapi/types/anthropic_messages.py:ThinkingDisplay
        """
        thinking: Any = {
            "type": "enabled",
            "budget_tokens": _BUDGET,
            "display": display,
        }
        with pytest.raises(AnthropicBadRequestError) as excinfo:
            anthropic_client.messages.create(
                model=anthropic_chat_reasoning_model,
                max_tokens=2 * _BUDGET,
                messages=[{"role": "user", "content": _PROMPT}],
                thinking=thinking,
            )

        body = excinfo.value.body
        assert excinfo.value.status_code == 400
        assert isinstance(body, dict)
        assert body["error"]["type"] == "invalid_request_error"
        assert "display" in body["error"]["message"]
        assert "'summarized'" in body["error"]["message"]

    @pytest.mark.expensive
    @pytest.mark.retry("adaptive thinking decides whether a prompt needs reasoning")
    @pytest.mark.parametrize("display", [None, "summarized"])
    def test_adaptive_thinking_display_on_a_model_omitting_by_default(
        self,
        anthropic_client: Anthropic,
        sonnet_5_anthropic_model: str,
        display: Literal["summarized"] | None,
    ) -> None:
        """Sonnet 5 omits its thinking text unless ``display`` is ``summarized``.

        Adaptive thinking may skip reasoning on an easy problem at low effort, so
        the request asks for high effort on a problem that needs working out.

        Ref: https://platform.claude.com/docs/en/build-with-claude/thinking#controlling-thinking-display
             https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
        """
        thinking: ThinkingConfigParam = (
            {"type": "adaptive"}
            if display is None
            else {"type": "adaptive", "display": display}
        )
        response = anthropic_client.messages.create(
            model=sonnet_5_anthropic_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": _REASONING_PROMPT}],
            thinking=thinking,
            output_config={"effort": "high"},
        )

        texts = _thinking_texts(response)
        if display is None:
            assert all(text == "" for text in texts)
        else:
            assert all(texts)

    @pytest.mark.gateway("Bedrock Mantle is not an official API")
    @pytest.mark.xdist_group("mantle_live")
    def test_display_reaches_a_mantle_served_model(
        self, anthropic_client: Anthropic
    ) -> None:
        """``display`` is forwarded to a Claude model the gateway serves through Mantle.

        Ref: stdapi/models/chat/_mantle/_convert.py:messages_payload
        """
        response = anthropic_client.messages.create(
            model=_CLAUDE_MANTLE,
            max_tokens=2 * _BUDGET,
            messages=[{"role": "user", "content": _PROMPT}],
            thinking={
                "type": "enabled",
                "budget_tokens": _BUDGET,
                "display": "omitted",
            },
        )

        assert all(text == "" for text in _thinking_texts(response))

    def test_count_tokens_accepts_display(
        self,
        anthropic_client: Anthropic,
        anthropic_count_tokens_model: str,
        is_bedrock_direct: bool,
    ) -> None:
        """Counting a request that sets ``display`` succeeds.

        Ref: https://platform.claude.com/docs/en/api/messages/count_tokens
        """
        if is_bedrock_direct:
            pytest.skip("The AnthropicBedrock client does not implement count_tokens")
        count = anthropic_client.messages.count_tokens(
            model=anthropic_count_tokens_model,
            messages=[{"role": "user", "content": _PROMPT}],
            thinking={
                "type": "enabled",
                "budget_tokens": _BUDGET,
                "display": "summarized",
            },
        )

        assert count.input_tokens > 0


@pytest.mark.gateway("Anthropic Claude is not served by the official OpenAI API")
class TestResponsesReasoningSummary:
    """``reasoning.summary`` on the Responses route returns Claude's reasoning text.

    Ref: https://developers.openai.com/api/docs/guides/reasoning#reasoning-summaries
         stdapi/models/chat/_adapters/_openai_responses.py:extract_reasoning
         stdapi/models/chat/_mantle/_convert.py:convert_payload
    """

    @pytest.mark.expensive
    @pytest.mark.retry("adaptive thinking decides whether a prompt needs reasoning")
    @pytest.mark.parametrize("summary", [None, "auto"])
    def test_summary_request(
        self, openai_client: OpenAI, summary: Literal["auto"] | None
    ) -> None:
        """A summary request returns reasoning text; without one, none comes back.

        Adaptive thinking may skip reasoning on an easy problem at low effort, so
        the request asks for high effort on a problem that needs working out.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking
        """
        response = openai_client.responses.create(
            model=_SONNET_5,
            input=_REASONING_PROMPT,
            reasoning={"effort": "high", "summary": summary},
            max_output_tokens=4096,
        )

        assert response.status == "completed"
        if summary is None:
            assert not any(_reasoning_texts(response.output))
        else:
            _assert_reasoning_in_summary(response.output)

    @pytest.mark.expensive
    @pytest.mark.retry("adaptive thinking decides whether a prompt needs reasoning")
    @pytest.mark.xdist_group("mantle_live")
    @pytest.mark.usefixtures("local_test_client")
    def test_summary_request_on_a_mantle_served_model(
        self,
        openai_client: OpenAI,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A summary request reaches Sonnet 5 served through Mantle and returns text.

        The per-request service header routes the dual-homed model to Mantle,
        where the Responses request is converted to the Messages API.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/models/chat/_mantle/_convert.py:convert_payload
        """
        from stdapi.models import MANTLE_MODELS  # noqa: PLC0415

        if _SONNET_5 not in MANTLE_MODELS:
            pytest.skip("Requires Claude Sonnet 5 listed by a configured Mantle region")
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_service_header", True)
        capfd.readouterr()
        response = openai_client.responses.create(
            model=_SONNET_5,
            input=_REASONING_PROMPT,
            reasoning={"effort": "high", "summary": "auto"},
            max_output_tokens=4096,
            extra_headers=_MANTLE_HEADERS,
        )

        assert response.status == "completed"
        _assert_reasoning_in_summary(response.output)
        assert logged_usage_entries(
            capfd.readouterr().out, service="bedrock-mantle", model=_SONNET_5
        ), "the request must be served through Mantle"

    @pytest.mark.xdist_group("mantle_live")
    def test_mantle_served_claude_streams_its_reasoning(
        self, openai_client: OpenAI
    ) -> None:
        """A Claude model served through Mantle streams its reasoning as a summary.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
             stdapi/models/chat/_mantle/_convert.py:convert_stream
        """
        with openai_client.responses.stream(
            model=_CLAUDE_MANTLE,
            input=_PROMPT,
            reasoning={"effort": "high", "summary": "auto"},
            max_output_tokens=4096,
        ) as stream:
            deltas = [
                event.delta
                for event in stream
                if event.type == "response.reasoning_summary_text.delta"
            ]
            final = stream.get_final_response()

        assert "".join(deltas)
        assert any(
            part.text
            for item in final.output
            if item.type == "reasoning"
            for part in item.summary
        )


def _reasoning_texts(output: Sequence[ResponseOutputItem]) -> list[str]:
    """Return the text of every reasoning content and summary part.

    Args:
        output: Responses API output items.

    Returns:
        Reasoning texts, in order.
    """
    return [
        part.text
        for item in output
        if item.type == "reasoning"
        for part in (*(item.content or ()), *item.summary)
    ]


def _assert_reasoning_in_summary(output: Sequence[ResponseOutputItem]) -> None:
    """Assert every reasoning item carries its text in ``summary``, as upstream does.

    Args:
        output: Responses API output items.
    """
    reasoning = [item for item in output if item.type == "reasoning"]
    assert reasoning, [item.type for item in output]
    for item in reasoning:
        assert item.summary, item
        assert all(part.type == "summary_text" and part.text for part in item.summary)
        assert not item.content, item


def _claude(model_id: str) -> AnthropicClaudeChatModel:
    """Return the Claude chat model implementation selected for *model_id*."""
    model: AnthropicClaudeChatModel = get_chat_model(model_id)  # type: ignore[assignment]
    return model


@pytest.mark.local
class TestDisplayPlumbing:
    """Which ``display`` each route hands the model, and what the model sends.

    Ref: stdapi/models/chat/_adapters/_anthropic_message.py:extract_reasoning
         stdapi/models/chat/_adapters/_openai_responses.py:extract_reasoning
         stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_reasoning
    """

    @pytest.mark.parametrize(
        "thinking",
        [
            {"type": "adaptive", "display": "summarized"},
            {"type": "enabled", "budget_tokens": _BUDGET, "display": "omitted"},
            {"type": "adaptive"},
        ],
    )
    def test_messages_route_forwards_the_client_display(
        self, thinking: dict[str, Any]
    ) -> None:
        """The Messages route hands over ``display`` exactly as the client sent it."""
        request = MessageCreateParams.model_validate(
            {
                "model": "m",
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": "hi"}],
                "thinking": thinking,
            }
        )

        params = anthropic_adapter.extract_reasoning(request)

        assert params is not None
        assert params["display"] == thinking.get("display")

    def test_messages_route_rejects_an_unknown_display(self) -> None:
        """A ``display`` value outside ``summarized``/``omitted`` fails validation."""
        with pytest.raises(ValidationError):
            ThinkingConfigAdaptiveParam.model_validate(
                {"type": "adaptive", "display": "updates"}
            )

    @pytest.mark.parametrize(
        "headers",
        [{}, {"anthropic-beta": "thinking-display-updates-2026-08-18"}],
        ids=["no-beta", "beta"],
    )
    def test_updates_display_is_refused(
        self, anthropic_app_client: TestClient, headers: dict[str, str]
    ) -> None:
        """The beta ``updates`` display is refused with a 400, beta header or not.

        Ref: https://platform.claude.com/docs/en/build-with-claude/thinking#controlling-thinking-display
             stdapi/types/anthropic_messages.py:ThinkingDisplay
        """
        response = anthropic_app_client.post(
            "/anthropic/v1/messages",
            headers=headers,
            json={
                "model": _SONNET_5,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "hi"}],
                "thinking": {"type": "adaptive", "display": "updates"},
            },
        )

        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"

    @pytest.mark.parametrize(
        ("reasoning", "display"),
        [
            ({"summary": "auto"}, "summarized"),
            ({"effort": "low", "summary": "concise"}, "summarized"),
            ({"summary": "detailed"}, "summarized"),
            ({"generate_summary": "auto"}, "summarized"),
            ({"effort": "low"}, None),
        ],
    )
    def test_responses_summary_asks_for_summarized_thinking(
        self, reasoning: dict[str, str], display: str | None
    ) -> None:
        """A Responses reasoning summary request becomes a summarized display."""
        request = ResponseCreateParams.model_validate(
            {"model": "m", "input": "hi", "reasoning": reasoning}
        )

        params = responses_adapter.extract_reasoning(request)

        assert params is not None
        assert params["display"] == display

    def test_chat_completions_sends_no_display(self) -> None:
        """Chat Completions has no summary request, so the model default applies."""
        request = ChatCompletionCreateParams.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "high",
            }
        )

        params = chat_adapter.extract_reasoning(request)

        assert params is not None
        assert params["display"] is None

    @pytest.mark.parametrize(
        ("model_id", "budget_tokens", "expected"),
        [
            (_SONNET_5, None, {"type": "adaptive", "display": "summarized"}),
            (
                _SONNET_5,
                _BUDGET,
                {"type": "enabled", "budget_tokens": _BUDGET, "display": "summarized"},
            ),
            (
                "anthropic.claude-haiku-4-5-20251001-v1:0",
                _BUDGET,
                {"type": "enabled", "budget_tokens": _BUDGET, "display": "summarized"},
            ),
        ],
        ids=["adaptive", "budget", "claude-4.5-budget"],
    )
    def test_claude_sends_the_display(
        self, model_id: str, budget_tokens: int | None, expected: JsonMapping
    ) -> None:
        """Claude receives ``display`` inside its thinking configuration.

        Ref: stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_reasoning
             stdapi/models/chat/anthropic_claude_37_to_45.py:ChatModel._req_configure_reasoning
        """
        fields: JsonMapping = {}
        _claude(model_id)._req_configure_reasoning(  # noqa: SLF001
            fields,
            enabled=True,
            budget_tokens=budget_tokens,
            max_tokens=2048,
            display="summarized",
        )

        assert fields["reasoning_config"] == expected

    @pytest.mark.parametrize(
        "model_id", [_SONNET_5, "anthropic.claude-haiku-4-5-20251001-v1:0"]
    )
    def test_claude_without_display_keeps_the_model_default(
        self, model_id: str
    ) -> None:
        """Without ``display`` the thinking configuration names none, twice in a row.

        The first call asks for a display, so a display leaking into a shared
        default would show up in the second.
        """
        _claude(model_id)._req_configure_reasoning(  # noqa: SLF001
            {}, enabled=True, max_tokens=2048, display="summarized"
        )
        fields: JsonMapping = {}
        _claude(model_id)._req_configure_reasoning(  # noqa: SLF001
            fields, enabled=True, max_tokens=2048
        )

        reasoning_config = fields["reasoning_config"]
        assert isinstance(reasoning_config, dict)
        assert "display" not in reasoning_config

    def test_disabled_thinking_carries_no_display(self) -> None:
        """A disabled thinking configuration never names a display."""
        fields: JsonMapping = {}
        _claude(_SONNET_5)._req_configure_reasoning(  # noqa: SLF001
            fields, enabled=False, display="summarized"
        )

        assert fields["reasoning_config"] == {"type": "disabled"}

    @pytest.mark.parametrize("display", [None, "summarized"])
    def test_always_reasoning_claude_honours_the_display_when_disabled(
        self, display: Literal["summarized"] | None
    ) -> None:
        """A model that cannot stop reasoning still returns the summary it was asked for.

        Disabling is dropped with a warning; the requested display is kept, and
        without one the model default is left alone.

        Ref: stdapi/models/chat/anthropic_claude_opus_5.py:ChatModel.REASONING_DISABLE_SUPPORTED
        """
        fields: JsonMapping = {}
        _claude("anthropic.claude-opus-5-5")._req_configure_reasoning(  # noqa: SLF001
            fields, enabled=False, display=display
        )

        expected = {"type": "adaptive", "display": display} if display else None
        assert fields.get("reasoning_config") == expected

    @pytest.mark.parametrize("model_id", ["amazon.nova-2-lite-v1:0", "deepseek.v3.2"])
    def test_other_models_ignore_the_display(self, model_id: str) -> None:
        """A model without Claude thinking accepts ``display`` and sends nothing for it."""
        fields: JsonMapping = {}
        model: ChatModel = get_chat_model(model_id)  # type: ignore[assignment]
        model._req_configure_reasoning(  # noqa: SLF001
            fields, enabled=True, reasoning_effort="low", display="summarized"
        )

        assert "display" not in str(fields)

    async def test_mantle_passthrough_keeps_the_display(self) -> None:
        """The Mantle Messages payload carries ``display`` as the client sent it.

        Ref: stdapi/models/chat/_mantle/_convert.py:messages_payload
        """
        request = MessageCreateParams.model_validate(
            {
                "model": _CLAUDE_MANTLE,
                "max_tokens": 2048,
                "messages": [{"role": "user", "content": "hi"}],
                "thinking": {"type": "adaptive", "display": "summarized"},
            }
        )

        payload = await messages_payload(request, _CLAUDE_MANTLE)

        assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}


@pytest.mark.local
class TestMantleResponsesConversion:
    """A Responses request converted for a Claude model served through Mantle.

    Ref: stdapi/models/chat/_mantle/_convert.py:convert_payload
         stdapi/models/chat/_mantle/_convert.py:convert_response
         stdapi/models/chat/_mantle/_convert.py:convert_stream
    """

    @pytest.mark.parametrize(
        ("model", "reasoning", "thinking"),
        [
            (
                _SONNET_5,
                {"summary": "auto"},
                {"type": "adaptive", "display": "summarized"},
            ),
            (
                _SONNET_5,
                {"effort": "low", "summary": "concise"},
                {"type": "adaptive", "display": "summarized"},
            ),
            (
                "anthropic.claude-haiku-4-5",
                {"effort": "high", "generate_summary": "auto"},
                {"type": "enabled", "budget_tokens": 3071, "display": "summarized"},
            ),
            (
                "anthropic.claude-haiku-4-5",
                {"summary": "auto"},
                {"type": "enabled", "budget_tokens": 2047, "display": "summarized"},
            ),
            (_SONNET_5, {"effort": "none", "summary": "auto"}, {"type": "disabled"}),
            (
                "anthropic.claude-opus-5-5",
                {"effort": "none", "summary": "auto"},
                {"type": "adaptive", "display": "summarized"},
            ),
            (_SONNET_5, {"effort": "low"}, None),
        ],
        ids=[
            "summary-only",
            "with-effort",
            "budget-model",
            "budget-model-no-effort",
            "effort-none",
            "effort-none-always-reasoning",
            "no-summary",
        ],
    )
    def test_summary_asks_for_summarized_thinking(
        self, model: str, reasoning: dict[str, str], thinking: dict[str, object] | None
    ) -> None:
        """A reasoning summary becomes a summarized display on the Messages payload.

        Without ``effort`` Claude reasons at ``medium``, as on the runtime path,
        so a budget model gets the budget that level derives.

        Ref: stdapi/models/chat/_mantle/_convert.py:_request_thinking_summary
             stdapi/models/chat/_adapters/_openai_responses.py:extract_reasoning
        """
        out = convert_payload(
            "responses",
            "messages",
            {
                "model": model,
                "input": "hi",
                "reasoning": reasoning,
                "max_output_tokens": 4096,
            },
        )

        assert out.get("thinking") == thinking

    def test_reasoning_without_effort_reasons_at_medium(self) -> None:
        """A ``reasoning`` object without ``effort`` asks for ``medium``, as on runtime.

        Ref: stdapi/models/chat/_mantle/_convert.py:convert_payload
             stdapi/models/chat/_adapters/_openai_responses.py:extract_reasoning
        """
        out = convert_payload(
            "responses",
            "messages",
            {"model": _SONNET_5, "input": "hi", "reasoning": {"summary": "auto"}},
        )

        assert out["output_config"] == {"effort": "medium"}

    def test_thinking_comes_back_as_reasoning(self) -> None:
        """Claude's thinking text becomes a ``reasoning`` output item.

        Ref: stdapi/models/chat/_mantle/_convert.py:_messages_to_chat_response
             stdapi/models/chat/_mantle/_convert.py:_chat_to_responses_response
        """
        out = convert_response(
            "messages",
            "responses",
            {
                "id": "msg_1",
                "model": _SONNET_5,
                "content": [
                    {"type": "thinking", "thinking": "Count primes.", "signature": "s"},
                    {"type": "redacted_thinking", "data": "opaque"},
                    {"type": "text", "text": "16"},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

        reasoning = [item for item in out["output"] if item["type"] == "reasoning"]
        assert reasoning[0]["content"] == [
            {"type": "reasoning_text", "text": "Count primes."}
        ]
        assert out["output"][-1]["content"][0]["text"] == "16"

    def test_omitted_thinking_adds_no_reasoning_item(self) -> None:
        """A thinking block with its text omitted adds no empty reasoning item.

        Ref: stdapi/models/chat/_mantle/_convert.py:_messages_to_chat_response
        """
        out = convert_response(
            "messages",
            "chat_completions",
            {
                "id": "msg_1",
                "model": _SONNET_5,
                "content": [
                    {"type": "thinking", "thinking": "", "signature": "s"},
                    {"type": "text", "text": "16"},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

        assert "reasoning_content" not in out["choices"][0]["message"]

    async def test_thinking_deltas_stream_as_reasoning(self) -> None:
        """Streamed thinking deltas become ``reasoning_text`` deltas; signatures do not.

        Ref: stdapi/models/chat/_mantle/_convert.py:_chat_delta_from_messages
        """
        upstream: list[SseEvent] = [
            (
                "message_start",
                to_json_str(
                    {
                        "type": "message_start",
                        "message": {"id": "msg_1", "model": _SONNET_5, "usage": {}},
                    }
                ),
            ),
            (
                "content_block_start",
                to_json_str(
                    {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "thinking", "thinking": ""},
                    }
                ),
            ),
            *(
                (
                    "content_block_delta",
                    to_json_str(
                        {"type": "content_block_delta", "index": 0, "delta": delta}
                    ),
                )
                for delta in (
                    {"type": "thinking_delta", "thinking": "Count "},
                    {"type": "thinking_delta", "thinking": "primes."},
                    {"type": "signature_delta", "signature": "sig"},
                )
            ),
            (
                "message_delta",
                to_json_str(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 3},
                    }
                ),
            ),
        ]

        async def _upstream() -> AsyncGenerator[SseEvent]:
            for event in upstream:
                yield event

        events = [
            event
            async for event in convert_stream("messages", "responses", _upstream())
        ]

        deltas = [
            from_json(data)["delta"]
            for name, data in events
            if name == "response.reasoning_text.delta"
        ]
        assert "".join(deltas) == "Count primes."

    def test_summary_request_returns_thinking_as_summary(self) -> None:
        """With a summary request, Claude's thinking fills ``summary``, not ``content``.

        Ref: https://developers.openai.com/api/docs/guides/reasoning#reasoning-summaries
             stdapi/models/chat/_mantle/_convert.py:_chat_to_responses_response
        """
        out = convert_response(
            "messages",
            "responses",
            {
                "id": "msg_1",
                "model": _SONNET_5,
                "content": [
                    {"type": "thinking", "thinking": "Count primes.", "signature": "s"},
                    {"type": "text", "text": "16"},
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            reasoning_summary=True,
        )

        reasoning = [item for item in out["output"] if item["type"] == "reasoning"]
        assert reasoning[0]["summary"] == [
            {"type": "summary_text", "text": "Count primes."}
        ]
        assert reasoning[0]["content"] == []

    async def test_summary_request_streams_summary_events(self) -> None:
        """With a summary request, thinking deltas stream as summary part 0.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
             stdapi/models/chat/_mantle/_convert.py:_responses_reasoning_delta
             stdapi/models/chat/_mantle/_convert.py:_close_responses_reasoning
        """
        upstream: list[SseEvent] = [
            (
                "message_start",
                to_json_str(
                    {
                        "type": "message_start",
                        "message": {"id": "msg_1", "model": _SONNET_5, "usage": {}},
                    }
                ),
            ),
            *(
                (
                    "content_block_delta",
                    to_json_str(
                        {"type": "content_block_delta", "index": 0, "delta": delta}
                    ),
                )
                for delta in (
                    {"type": "thinking_delta", "thinking": "Count "},
                    {"type": "thinking_delta", "thinking": "primes."},
                )
            ),
            (
                "message_delta",
                to_json_str(
                    {
                        "type": "message_delta",
                        "delta": {"stop_reason": "end_turn"},
                        "usage": {"output_tokens": 3},
                    }
                ),
            ),
        ]

        async def _upstream() -> AsyncGenerator[SseEvent]:
            for event in upstream:
                yield event

        events = [
            (name, from_json(data))
            async for name, data in convert_stream(
                "messages", "responses", _upstream(), reasoning_summary=True
            )
        ]

        reasoning = [
            (name, payload)
            for name, payload in events
            if "reasoning" in (name or "")
            or payload.get("item", {}).get("type") == "reasoning"
        ]
        assert [name for name, _ in reasoning] == [
            "response.output_item.added",
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done",
            "response.reasoning_summary_part.done",
            "response.output_item.done",
        ]
        for _, payload in reasoning[1:-1]:
            assert payload["summary_index"] == 0
            assert "content_index" not in payload
        assert reasoning[4][1]["text"] == "Count primes."
        assert reasoning[-1][1]["item"]["summary"] == [
            {"type": "summary_text", "text": "Count primes."}
        ]
        assert reasoning[-1][1]["item"]["content"] == []
