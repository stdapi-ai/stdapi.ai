"""Qwen Code driven end to end against the gateway's Chat Completions route.

Qwen Code adds no route the lane does not already reach; it is here for one
behaviour no other CLI has. It reads the non-standard ``reasoning_content`` field
off an assistant message, keeps it as a thought part in its own history, and
writes it back onto that assistant message when it replays the conversation on the
next turn. Claude Code, Codex and pi all drop the reasoning text once they have
displayed it, so this is the only *agent binary* that sends the gateway's
``ChatCompletionAssistantMessageParam.reasoning_content`` back for it to map into
a Bedrock ``reasoningContent`` block -- and it sends text the gateway itself
produced, which no unit test can.

What this exercises that the other tools do not:

- a full round trip of the reasoning text. The gateway emits it, the client parses
  it, and the client sends it back for the gateway to translate into a Bedrock
  reasoning block, all inside one agent loop.
- the ``reasoning`` request object. Qwen Code sends the effort level as
  ``{"reasoning": {"effort": ...}}`` rather than as the flat ``reasoning_effort``
  every other client uses, so the object form is exercised by a real client.

Requires ``--agentic``, podman, and Bedrock credentials.

Ref: https://qwenlm.github.io/qwen-code-docs/en/users/features/headless/
     https://qwenlm.github.io/qwen-code-docs/en/users/configuration/auth/
     tests/agentic/_tools.py:QWEN_CODE
     stdapi/types/openai_chat_completions.py:ChatCompletionAssistantMessageParam
     stdapi/models/chat/_adapters/_openai_chat_completion.py:_map_assistant_reasoning_content
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ._runner import (
    DEEPSEEK_DSML_LEAK,
    ModelConfig,
    assert_result,
    log_metrics,
    run_agent,
)
from ._tools import QWEN_CODE, SRC_MOUNT

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from ._server import AgenticServer

pytestmark = pytest.mark.agentic

#: The CLI this module drives; read by the autouse model-identity fixture.
TOOL = QWEN_CODE

#: Qwen Code plans several read-only tool calls per turn, like pi. Still bounded
#: well under the lane's 15-minute budget: the slowest measured run here is ~205 s.
_TIMEOUT = 900

#: Models exercised on the standard agent task.
#:
#: One Anthropic model and one Amazon model cover the two native Converse
#: dialects, DeepSeek covers the open-weight path whose reasoning text is what
#: the second class here round-trips, and Qwen3-Coder is the family this client
#: is built for -- the pairing most likely to exercise its own defaults.
_MODEL_CONFIGS = [
    pytest.param(
        ModelConfig(model="anthropic.claude-haiku-4-5-20251001-v1:0", timeout=_TIMEOUT),
        id="claude-haiku-4-5",
    ),
    pytest.param(
        ModelConfig(model="amazon.nova-2-lite-v1:0", timeout=_TIMEOUT), id="nova-2-lite"
    ),
    pytest.param(
        ModelConfig(
            model="deepseek.v3.2",
            timeout=_TIMEOUT,
            supports_effort=True,
            flaky=DEEPSEEK_DSML_LEAK,
        ),
        id="deepseek-v3.2",
    ),
    pytest.param(
        # Deliberately not flaky, unlike the same model in test_claude_code.py:
        # this is the client built for it, and its only failures here have passed
        # on the immediate re-run. See the note in test_codex.py.
        ModelConfig(model="qwen.qwen3-coder-30b-a3b-v1:0", timeout=_TIMEOUT),
        id="qwen3-coder-30b",
    ),
]

#: Models whose reasoning text the client can read back and replay.
#:
#: DeepSeek is the family whose reasoning arrives as ``reasoning_content`` on the
#: Converse path today, which is the field Qwen Code mirrors. MiniMax M2.5 is
#: served by Bedrock Mantle, which names the same field ``reasoning``, so the two
#: cover both spellings the gateway has to normalise before the client sees them.
#:
#: The Mantle entry was ``qwen.qwen3-next-80b-a3b-instruct``, which cannot carry
#: this test: called directly, it answers with ``content``, ``refusal`` and
#: ``role`` and nothing else -- no reasoning field at any effort, streamed or not
#: -- so there was never any thinking text for the client to replay. It also
#: never answers at all when asked through the flat ``reasoning_effort: "high"``
#: the gateway sends (measured: no event in 300 s, 4/4 runs, while the equivalent
#: ``reasoning: {"effort": "high"}`` answers the same request in under a second).
#: Both are upstream conditions. M2.5 emits ``reasoning`` through either spelling,
#: streamed and not, and already completes agent loops elsewhere in this lane.
_REASONING_MODEL_CONFIGS = [
    pytest.param(
        ModelConfig(
            model="deepseek.v3.2",
            timeout=_TIMEOUT,
            supports_effort=True,
            flaky=DEEPSEEK_DSML_LEAK,
        ),
        id="deepseek-v3.2",
    ),
    pytest.param(
        ModelConfig(
            model="minimax.minimax-m2.5", timeout=_TIMEOUT, supports_effort=True
        ),
        id="minimax-m2.5",
    ),
]

#: Effort level requested of the reasoning models.
#:
#: The replay only happens when the model returns reasoning text at all, so the
#: highest tier is asked for rather than left to the model's own default.
_EFFORT = "high"

_PROMPT_ADAPTER_LAYOUT = f"""\
You are working in the stdapi.ai source tree at {SRC_MOUNT}.

Identify every chat API adapter the gateway implements and what each one
translates between.

Use shell commands to read the actual source. Do not guess:
  1. List every file in {SRC_MOUNT}/stdapi/models/chat/_adapters/
  2. Read the module docstring of each adapter and quote it
  3. For each adapter, name the client API it accepts and quote the exact
     signature of the function that converts a request into Bedrock's shape

Report one section per adapter, with real code quotes.
"""

_PROMPT_REASONING_FIELD = f"""\
You are working in the stdapi.ai source tree at {SRC_MOUNT}.

Explain how the gateway carries a reasoning model's thinking text on the
Chat Completions route, in both directions.

Use shell commands to read source files. Quote actual code, never guess:
  1. Find the field an assistant message uses to carry reasoning text back to
     the gateway, and quote its declaration
  2. Find where that field is turned into a Bedrock reasoning block, and quote
     the function's signature
  3. Explain what the operator setting that renames the emitted field does

Read at least three distinct files and quote code from each.
"""

#: Vocabulary that appears only in the adapter files the first prompt forces open.
_ADAPTER_KEYWORDS = (
    "_openai_chat_completion",
    "_openai_responses",
    "_anthropic_message",
    "translate_request",
    "adapter",
)

#: Vocabulary any correct account of the reasoning plumbing uses.
_REASONING_KEYWORDS = (
    "reasoning_content",
    "reasoningcontent",
    "chat_completions_reasoning_field",
    "thinking",
)


@pytest.mark.parametrize("model_config", _MODEL_CONFIGS)
class TestQwenCodeAgentLoop:
    """The lane's standard agent task, driven by Qwen Code.

    Qwen Code speaks Chat Completions through its own OpenAI-compatible provider
    rather than through the OpenAI SDK's defaults, so it is a second, independent
    implementation of the wire format pi already drives -- and the only one that
    keeps reasoning text in its history.

    Ref: https://qwenlm.github.io/qwen-code-docs/en/users/features/headless/
         stdapi/routes/openai_chat_completions.py
    """

    def test_enumerate_adapters(
        self,
        request: pytest.FixtureRequest,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """Qwen Code enumerates the adapters after reading them.

        The answer names files the prompt does not, so it can only come from tool
        output carried back through the gateway's own streaming translation.

        Ref: stdapi/models/chat/_adapters/__init__.py
        """
        result = run_agent(
            tool=TOOL,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_PROMPT_ADAPTER_LAYOUT,
            workdir=agentic_workdir,
            test_name=request.node.originalname or request.node.name,
        )
        log_metrics(TOOL, result, model_config, "test_enumerate_adapters")
        assert_result(
            result,
            config=model_config,
            contains="adapter",
            any_of=_ADAPTER_KEYWORDS,
            min_steps=2,
        )


@pytest.mark.parametrize("model_config", _REASONING_MODEL_CONFIGS)
class TestQwenCodeReasoningReplay:
    """The reasoning text the gateway emitted, sent back to it by the client.

    The assertion is on the gateway's own request log rather than on the answer:
    a client that read the reasoning text and dropped it would still produce a
    correct answer, and only the logged request body shows the text arriving back
    on an assistant message the gateway then has to translate.

    Ref: https://api-docs.deepseek.com/guides/reasoning_model
         stdapi/models/chat/deepseek_v3.py:_req_configure_reasoning
         stdapi/models/chat/_adapters/_openai_chat_completion.py:_map_assistant_reasoning_content
    """

    def test_reasoning_content_is_replayed_to_the_gateway(
        self,
        request: pytest.FixtureRequest,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
    ) -> None:
        """A later request carries the reasoning text of an earlier answer.

        The task needs several tool calls, so the conversation is replayed to the
        gateway repeatedly; each replay carries the assistant turns that came
        before it, and on a reasoning model those turns carry the thinking text
        the gateway itself emitted.

        Ref: stdapi/types/openai_chat_completions.py:ChatCompletionAssistantMessageParam
        """
        log_start = len(agentic_server.logs)
        result = run_agent(
            tool=TOOL,
            server=agentic_server,
            image=agentic_image,
            config=model_config,
            prompt=_PROMPT_REASONING_FIELD,
            workdir=agentic_workdir,
            test_name=request.node.originalname or request.node.name,
            effort=_EFFORT,
        )
        log_metrics(
            TOOL,
            result,
            model_config,
            "test_reasoning_content_is_replayed_to_the_gateway",
        )

        requests = [
            params
            for entry in agentic_server.log_entries(log_start)
            if entry.get("path") == "/v1/chat/completions"
            and (params := _request_params(entry))
        ]
        efforts = [
            reasoning.get("effort")
            for params in requests
            if isinstance(reasoning := params.get("reasoning"), dict)
        ]
        assert _EFFORT in efforts, (
            f"No logged request carried reasoning.effort={_EFFORT!r}: the effort "
            f"level never reached the gateway, so the answer's reasoning text is "
            f"whatever the model produces by default. Saw: {efforts}"
        )

        replayed = [
            message
            for params in requests
            for message in _request_messages(params)
            if message.get("role") == "assistant" and message.get("reasoning_content")
        ]
        assert replayed, (
            "No logged request carried a prior assistant message with reasoning "
            "content: the client never replayed the gateway's own reasoning text."
        )

        # Last, because it is the only assertion here a model's own answer can
        # fail, and it must not cost the replay evidence above.
        assert_result(
            result,
            config=model_config,
            contains="reasoning",
            any_of=_REASONING_KEYWORDS,
            min_steps=2,
        )


def _request_params(entry: Mapping[str, object]) -> dict[str, object]:
    """Return the request body one log event recorded.

    Args:
        entry: One JSON event from the gateway's request log.

    Returns:
        The logged body, or an empty mapping for an event that carries none.
    """
    params = entry.get("request_params")
    return params if isinstance(params, dict) else {}


def _request_messages(params: Mapping[str, object]) -> list[dict[str, object]]:
    """Return the ``messages`` array of one logged request body.

    Args:
        params: Body from :func:`_request_params`.

    Returns:
        One dict per message the request carried; empty when it carried none.
    """
    if not isinstance(messages := params.get("messages"), list):
        return []
    return [message for message in messages if isinstance(message, dict)]
