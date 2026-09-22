"""Moonshot Kimi K3: reasoning effort, automatic prompt caching and reasoning replay.

Probed on Converse (``tests/probes/results/moonshotai.kimi-k3.json``): the model
reasons by default and only ``additionalModelRequestFields.reasoning.effort``
changes that (``none`` returns no reasoning); the Kimi K2 ``thinking`` toggle is
accepted and ignored. Every ``cachePoint`` is rejected, while a repeated prefix
is cached automatically and reported as cache read and write tokens.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-moonshot-ai-kimi-k3.html
     tests/probes/results/moonshotai.kimi-k3.json
     stdapi/models/chat/kimi_k3.py:ChatModel
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

from stdapi.models import _find_model_class
from stdapi.models.chat._reasoning_effort import ReasoningEffortChatModel
from stdapi.models.chat.kimi_k3 import ChatModel as KimiK3ChatModel
from stdapi.models.chat.kimi_k25 import ChatModel as KimiK2ChatModel
from stdapi.types.anthropic_messages import MessageCreateParams
from stdapi.types.openai_chat_completions import CompletionCreateParams
from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from anthropic import Anthropic
    from openai import OpenAI

#: Kimi K3, served through its cross-region inference profiles only.
KIMI_K3 = "moonshotai.kimi-k3"

#: Seconds between two calls sharing a prefix, so the first call's cache write lands.
_CACHE_SETTLE_SECONDS = 3

#: Prompt that needs several steps, so the model has a reason to reason.
_REASONING_PROMPT = (
    "How many positive integers below 200 are divisible by 3 or 5 but not by 7? "
    "Answer with the number only."
)


def _cacheable_prefix() -> str:
    """Return a prefix above the 1,024-token caching minimum, unique to this call.

    The run-specific tag keeps a previous run's cache entry from answering the
    first call.
    """
    filler = "The gateway translates requests between API dialects. " * 220
    return f"[{uuid4()}] {filler.strip()}"


@pytest.mark.local
class TestMatcher:
    """Kimi K3 and later reach the K3 class, the K2 generation stays on its own.

    Ref: stdapi/models/__init__.py:_find_model_class
    """

    @pytest.mark.parametrize(
        "model_id",
        [
            KIMI_K3,
            "moonshot.kimi-k3-thinking",
            "moonshotai.kimi-k4",
            "moonshotai.kimi-k10",
            "moonshotai.kimi-k20",
            "moonshotai.kimi-k25",
            "moonshotai.kimi-k100",
        ],
    )
    def test_k3_and_later_take_the_effort_object(self, model_id: str) -> None:
        """Both provider prefixes, and every later generation, resolve to the K3 class.

        K20 to K29 start like K2 and must still not fall back to the K2 class.
        """
        model_class = _find_model_class(model_id)

        assert model_class is KimiK3ChatModel
        assert issubclass(model_class, ReasoningEffortChatModel)

    @pytest.mark.parametrize(
        "model_id",
        [
            "moonshotai.kimi-k2.5",
            "moonshot.kimi-k2-thinking",
            "moonshotai.kimi-k2",
            "moonshotai.kimi-k2.6",
        ],
    )
    def test_k2_generation_is_unchanged(self, model_id: str) -> None:
        """The K2 matcher and the K3 one never claim the same model."""
        assert _find_model_class(model_id) is KimiK2ChatModel


@pytest.mark.local
class TestRequestShape:
    """What reaches the model: the effort object, and never a cache point.

    Ref: stdapi/models/chat/kimi_k3.py:ChatModel
    """

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [("none", "none"), ("minimal", "low"), ("xhigh", "xhigh"), ("max", "max")],
    )
    async def test_reasoning_effort_is_the_effort_object(
        self, requested: str, expected: str
    ) -> None:
        """``reasoning_effort`` is sent as ``reasoning.effort``, never as the K2 fields.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
        """
        request = CompletionCreateParams.model_validate(
            {
                "model": KIMI_K3,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "reasoning_effort": requested,
            }
        )

        payload, _, _ = await KimiK3ChatModel(KIMI_K3).build_completion_request(request)

        assert payload["additionalModelRequestFields"] == {
            "reasoning": {"effort": expected}
        }

    async def test_a_chat_cache_request_sends_no_cache_point(self) -> None:
        """``prompt_cache_key`` is accepted, and no cache point reaches the model.

        Every cache point is rejected by Kimi K3, and its caching is automatic.

        Ref: https://developers.openai.com/api/docs/guides/prompt-caching
        """
        request = CompletionCreateParams.model_validate(
            {
                "model": KIMI_K3,
                "messages": [
                    {"role": "system", "content": _cacheable_prefix()},
                    {"role": "user", "content": "Reply with OK."},
                ],
                "prompt_cache_key": "kimi-k3",
            }
        )

        payload, _, _ = await KimiK3ChatModel(KIMI_K3).build_completion_request(request)

        assert "cachePoint" not in str(payload)

    async def test_an_anthropic_cache_control_sends_no_cache_point(self) -> None:
        """``cache_control`` breakpoints are dropped rather than sent and rejected.

        Ref: https://docs.claude.com/en/docs/build-with-claude/prompt-caching
        """
        request = MessageCreateParams.model_validate(
            {
                "model": KIMI_K3,
                "max_tokens": 1024,
                "system": [
                    {
                        "type": "text",
                        "text": _cacheable_prefix(),
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "messages": [{"role": "user", "content": "Reply with OK."}],
            }
        )

        payload, _ = await KimiK3ChatModel(KIMI_K3).build_message_request(request)

        assert "cachePoint" not in str(payload)


@pytest.mark.gateway("Kimi K3 is not served by the official APIs")
class TestReasoningEffort:
    """The requested effort reaches Kimi K3.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-moonshot-ai-kimi-k3.html
    """

    def test_reasoning_effort_none_returns_no_reasoning(
        self, openai_client: OpenAI
    ) -> None:
        """``none`` answers without reasoning, where ``low`` still reasons.

        Kimi K3 reasons by default and ignores the K2 ``thinking`` toggle, so a
        dropped effort would leave both answers reasoning.

        Ref: https://developers.openai.com/api/docs/guides/reasoning
             stdapi/models/chat/kimi_k3.py:ChatModel
        """
        answers = {
            effort: openai_client.chat.completions.create(
                model=KIMI_K3,
                messages=[{"role": "user", "content": _REASONING_PROMPT}],
                reasoning_effort=effort,
                max_completion_tokens=1024,
            )
            .choices[0]
            .message
            for effort in ("none", "low")
        }

        assert answers["none"].content
        assert not getattr(answers["none"], "reasoning_content", None)
        assert getattr(answers["low"], "reasoning_content", None)


@pytest.mark.gateway("Kimi K3 is not served by the official APIs")
@pytest.mark.usefixtures("local_test_client")
class TestAutomaticPromptCaching:
    """A repeated prefix is reported as cached on every dialect, and billed as such.

    The request carries no cache control at all: Kimi K3 caches on its own.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
         stdapi/models/__init__.py:ModelBase._record_converse_usage
    """

    @staticmethod
    def _assert_billed(stdout: str) -> None:
        """Assert the two calls logged a cache write, then a cache read."""
        entries = logged_usage_entries(stdout, service="bedrock-runtime")
        assert len(entries) == 2, entries
        assert entries[0].get("cache_write_tokens", 0) > 0
        assert entries[1].get("cached_tokens", 0) > 0

    def test_chat_completions_reports_cached_tokens(
        self, openai_client: OpenAI, capfd: pytest.CaptureFixture[str]
    ) -> None:
        """The second call reports ``prompt_tokens_details.cached_tokens``.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
        """
        prefix = _cacheable_prefix()
        capfd.readouterr()
        usages = []
        for _ in range(2):
            response = openai_client.chat.completions.create(
                model=KIMI_K3,
                messages=[
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": "Reply with OK."},
                ],
                reasoning_effort="none",
                max_completion_tokens=64,
            )
            usages.append(response.usage)
            time.sleep(_CACHE_SETTLE_SECONDS)

        cached = usages[1].prompt_tokens_details.cached_tokens  # type: ignore[union-attr]
        assert cached is not None
        assert cached > 1024
        assert usages[1].prompt_tokens >= cached  # type: ignore[union-attr]
        self._assert_billed(capfd.readouterr().out)

    def test_responses_reports_cached_tokens(
        self, openai_client: OpenAI, capfd: pytest.CaptureFixture[str]
    ) -> None:
        """The second call reports ``input_tokens_details.cached_tokens``.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        prefix = _cacheable_prefix()
        capfd.readouterr()
        usages = []
        for _ in range(2):
            response = openai_client.responses.create(
                model=KIMI_K3,
                instructions=prefix,
                input="Reply with OK.",
                reasoning={"effort": "none"},
                max_output_tokens=64,
                store=False,
            )
            usages.append(response.usage)
            time.sleep(_CACHE_SETTLE_SECONDS)

        cached = usages[1].input_tokens_details.cached_tokens  # type: ignore[union-attr]
        assert cached > 1024
        assert usages[1].input_tokens >= cached  # type: ignore[union-attr]
        self._assert_billed(capfd.readouterr().out)

    def test_anthropic_messages_reports_cache_read_and_creation(
        self, anthropic_client: Anthropic, capfd: pytest.CaptureFixture[str]
    ) -> None:
        """The first call reports a cache creation, the second a cache read.

        Ref: https://docs.claude.com/en/api/messages
        """
        prefix = _cacheable_prefix()
        capfd.readouterr()
        usages = []
        for _ in range(2):
            response = anthropic_client.messages.create(
                model=KIMI_K3,
                system=prefix,
                messages=[{"role": "user", "content": "Reply with OK."}],
                thinking={"type": "disabled"},
                max_tokens=64,
            )
            usages.append(response.usage)
            time.sleep(_CACHE_SETTLE_SECONDS)

        assert (usages[0].cache_creation_input_tokens or 0) > 1024
        assert (usages[1].cache_read_input_tokens or 0) > 1024
        self._assert_billed(capfd.readouterr().out)


@pytest.mark.expensive
@pytest.mark.gateway("Kimi K3 is not served by the official APIs")
class TestReasoningReplay:
    """A follow-up turn replaying the model's own reasoning is served.

    The model card warns that reasoning from earlier turns can fail on this
    model; replaying it is what every agent client does.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-moonshot-ai-kimi-k3.html
         stdapi/models/chat/_adapters/_openai_chat_completion.py:_map_assistant_reasoning_content
    """

    def test_chat_completions_replays_reasoning_content(
        self, openai_client: OpenAI
    ) -> None:
        """An assistant turn carrying ``reasoning_content`` is accepted on the next call.

        Ref: https://api-docs.deepseek.com/guides/reasoning_model
        """
        messages: list[Any] = [{"role": "user", "content": _REASONING_PROMPT}]
        first = (
            openai_client.chat.completions.create(
                model=KIMI_K3,
                messages=messages,
                reasoning_effort="low",
                max_completion_tokens=4096,
            )
            .choices[0]
            .message
        )
        reasoning = getattr(first, "reasoning_content", None)
        assert reasoning

        messages += [
            {
                "role": "assistant",
                "content": first.content,
                "reasoning_content": reasoning,
            },
            {"role": "user", "content": "Double it. Answer with the number only."},
        ]
        second = openai_client.chat.completions.create(
            model=KIMI_K3,
            messages=messages,
            reasoning_effort="low",
            max_completion_tokens=4096,
        )

        assert second.choices[0].message.content

    def test_anthropic_messages_replays_thinking(
        self, anthropic_client: Anthropic
    ) -> None:
        """An assistant turn carrying the model's ``thinking`` block is accepted.

        Ref: https://docs.claude.com/en/docs/build-with-claude/extended-thinking
        """
        messages: list[Any] = [{"role": "user", "content": _REASONING_PROMPT}]
        first = anthropic_client.messages.create(
            model=KIMI_K3,
            messages=messages,
            output_config={"effort": "low"},
            max_tokens=4096,
        )
        assert any(block.type == "thinking" for block in first.content)

        messages += [
            {"role": "assistant", "content": [b.model_dump() for b in first.content]},
            {"role": "user", "content": "Double it. Answer with the number only."},
        ]
        second = anthropic_client.messages.create(
            model=KIMI_K3,
            messages=messages,
            output_config={"effort": "low"},
            max_tokens=4096,
        )

        assert any(block.type == "text" and block.text for block in second.content)

    def test_responses_replays_reasoning_items(self, openai_client: OpenAI) -> None:
        """The previous output, reasoning item included, is accepted as input.

        Ref: https://developers.openai.com/api/docs/guides/reasoning
        """
        first = openai_client.responses.create(
            model=KIMI_K3,
            input=_REASONING_PROMPT,
            reasoning={"effort": "low"},
            max_output_tokens=4096,
            store=False,
        )
        assert any(item.type == "reasoning" for item in first.output)
        replay: list[Any] = [
            {"role": "user", "content": _REASONING_PROMPT},
            *[item.model_dump(exclude_none=True) for item in first.output],
            {"role": "user", "content": "Double it. Answer with the number only."},
        ]

        second = openai_client.responses.create(
            model=KIMI_K3,
            input=replay,
            reasoning={"effort": "low"},
            max_output_tokens=4096,
            store=False,
        )

        assert second.output_text
