"""A thinking token budget asked of a model that only takes an effort level.

Anthropic's ``thinking`` object makes ``budget_tokens`` a required field of
``{"type": "enabled"}``, so on the Anthropic Messages route it is the only way a
standard client can ask for reasoning at all.  Models whose reasoning knob is a
categorical effort (DeepSeek V3, Amazon Nova 2) therefore have to serve the
request rather than refuse it, or reasoning is unreachable on that route while
working on the other two.

Ref: https://docs.claude.com/en/api/messages
     https://developers.openai.com/api/docs/guides/reasoning
     stdapi/models/chat/deepseek_v3.py:ChatModel._req_configure_reasoning
     stdapi/models/chat/amazon_nova_2.py:ChatModel._req_configure_reasoning
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from stdapi.models.chat.amazon_nova_2 import ChatModel as NovaChatModel
from stdapi.models.chat.deepseek_v3 import ChatModel as DeepseekChatModel
from stdapi.types.anthropic_messages import MessageCreateParams
from stdapi.types.openai_chat_completions import CompletionCreateParams

if TYPE_CHECKING:
    from stdapi.models.chat._default import ChatModel as ConverseChatModel

pytestmark = pytest.mark.local

#: A budget comfortably above Anthropic's documented 1,024 minimum and below ``max_tokens``.
_BUDGET = 2048

#: The reasoning configuration each family writes when reasoning is on at its own default.
#:
#: DeepSeek takes a bare effort string, Nova 2 an object; both are what the model
#: emits for an enabled request that named no effort level.
_MODELS = [
    pytest.param(
        DeepseekChatModel("deepseek.v3.2"),
        "deepseek.v3.2",
        {"reasoning_config": "high"},
        id="deepseek-v3.2",
    ),
    pytest.param(
        NovaChatModel("amazon.nova-2-lite-v1:0"),
        "amazon.nova-2-lite-v1:0",
        {"reasoningConfig": {"type": "enabled", "maxReasoningEffort": "medium"}},
        id="nova-2-lite",
    ),
]


def _anthropic_request(model_id: str, **overrides: object) -> MessageCreateParams:
    """Build a validated Anthropic message request for *model_id*."""
    body: dict[str, Any] = {
        "model": model_id,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        **overrides,
    }
    return MessageCreateParams.model_validate(body)


def _completion_request(model_id: str, **overrides: object) -> CompletionCreateParams:
    """Build a validated chat completion request for *model_id*."""
    body: dict[str, Any] = {
        "model": model_id,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        **overrides,
    }
    return CompletionCreateParams.model_validate(body)


@pytest.mark.parametrize(("model", "model_id", "expected"), _MODELS)
async def test_anthropic_thinking_budget_enables_reasoning(
    model: ConverseChatModel, model_id: str, expected: dict[str, Any]
) -> None:
    """``thinking.budget_tokens`` reaches the backend as this model's own effort level.

    ``budget_tokens`` is not optional in Anthropic's enabled-thinking object, so
    refusing it would leave the Anthropic Messages route with no way to ask these
    models to reason -- while ``reasoning_effort`` and ``reasoning.effort`` both
    work on the other two routes.

    Ref: stdapi/models/chat/_adapters/_anthropic_message.py:extract_reasoning
    """
    request = _anthropic_request(
        model_id, thinking={"type": "enabled", "budget_tokens": _BUDGET}
    )
    payload, _ = await model.build_message_request(request)
    assert payload["additionalModelRequestFields"] == expected


@pytest.mark.parametrize(("model", "model_id", "expected"), _MODELS)
async def test_chat_completions_thinking_budget_enables_reasoning(
    model: ConverseChatModel, model_id: str, expected: dict[str, Any]
) -> None:
    """The Qwen-compatible ``thinking_budget`` extra is served the same way.

    The documented contract is that ``thinking_budget`` is accepted for every
    reasoning model; a budget these models cannot size still says "think", so the
    depth falls back to their own default rather than the request failing.

    Ref: stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_reasoning
    """
    request = _completion_request(
        model_id, enable_thinking=True, thinking_budget=_BUDGET
    )
    payload, _, _ = await model.build_completion_request(request)
    assert payload["additionalModelRequestFields"] == expected


@pytest.mark.parametrize(("model", "model_id", "expected"), _MODELS)
async def test_effort_still_wins_over_a_budget_on_the_anthropic_route(
    model: ConverseChatModel, model_id: str, expected: dict[str, Any]
) -> None:
    """An explicit effort level is what sizes the reasoning, not the budget.

    Both fields can be sent together on this route, and only one of them is
    something these models can act on.

    Ref: stdapi/types/anthropic_messages.py:OutputConfigParam
    """
    request = _anthropic_request(
        model_id,
        thinking={"type": "enabled", "budget_tokens": _BUDGET},
        output_config={"effort": "low"},
    )
    payload, _ = await model.build_message_request(request)
    assert payload["additionalModelRequestFields"] != expected
    assert "low" in str(payload["additionalModelRequestFields"])


def test_a_budget_contradicting_an_effort_is_still_refused() -> None:
    """``reasoning_effort`` with ``thinking_budget`` remains a naming 400.

    Accepting an unsizable budget must not loosen the rule that two different
    ways of sizing the same budget in one request is a contradiction, which is
    refused before any model is resolved.

    Ref: stdapi/types/openai_chat_completions.py:CompletionCreateParams
    """
    with pytest.raises(ValueError, match="thinking_budget") as excinfo:
        _completion_request(
            "deepseek.v3.2",
            reasoning_effort="minimal",
            enable_thinking=True,
            thinking_budget=_BUDGET,
        )
    assert "reasoning_effort" in str(excinfo.value)
