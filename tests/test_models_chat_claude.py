"""Per-generation Claude capability tables: server tools, reasoning, system messages.

Anthropic-provided tools are version-keyed (``computer_20251124``,
``bash_20250124``, ...) and Bedrock accepts only the versions a given Claude
generation was released with, so the gateway keeps a per-model table instead of a
single global mapping.

Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
     stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel
"""

from typing import TYPE_CHECKING, cast

import pytest

from stdapi.models.chat import get_chat_model
from stdapi.monitoring import REQUEST_LOG

if TYPE_CHECKING:
    from collections.abc import Iterator

    from stdapi.models.chat._anthropic_claude import AnthropicClaudeChatModel
    from stdapi.monitoring import EventLog
    from stdapi.types import JsonMapping

pytestmark = pytest.mark.local


@pytest.fixture
def request_log() -> Iterator[EventLog]:
    """Provide a request log context capturing warnings."""
    log: EventLog = cast("EventLog", {"level": "info"})
    token = REQUEST_LOG.set(log)
    yield log
    REQUEST_LOG.reset(token)


def _claude_model(model_id: str) -> AnthropicClaudeChatModel:
    """Return the Claude chat model implementation selected for *model_id*."""
    return cast("AnthropicClaudeChatModel", get_chat_model(model_id))


#: Computer use tool type each Claude generation accepts, as reported by Bedrock.
_COMPUTER_TOOL_TYPES = {
    "anthropic.claude-3-7-sonnet-20250219-v1:0": "computer_20250124",
    "anthropic.claude-haiku-4-5-20251001-v1:0": "computer_20250124",
    "anthropic.claude-sonnet-4-5-20250929-v1:0": "computer_20250124",
    "anthropic.claude-opus-4-5-20251101-v1:0": "computer_20250124",
    "anthropic.claude-opus-4-6-v1": "computer_20251124",
    "anthropic.claude-opus-4-7": "computer_20251124",
    "anthropic.claude-opus-4-8": "computer_20251124",
    "anthropic.claude-sonnet-4-6": "computer_20251124",
    "anthropic.claude-sonnet-5": "computer_20251124",
    "anthropic.claude-fable-5": "computer_20251124",
    # Opus 5 accepts no computer use tool version at all.
    "anthropic.claude-opus-5": None,
}


#: Model IDs of unreleased versions, mapped to the behavior they must inherit.
_FUTURE_MODELS = {
    "anthropic.claude-opus-5-1": None,
    "anthropic.claude-opus-6": None,
    "anthropic.claude-opus-10": None,
    "anthropic.claude-sonnet-5-1": "computer_20251124",
    "anthropic.claude-sonnet-6": "computer_20251124",
    "anthropic.claude-haiku-6": "computer_20251124",
    "anthropic.claude-fable-5-1": "computer_20251124",
    "anthropic.claude-fable-6": "computer_20251124",
    "anthropic.claude-mythos-6": "computer_20251124",
}


@pytest.mark.parametrize(
    ("model_id", "tool_type"), [*_COMPUTER_TOOL_TYPES.items(), *_FUTURE_MODELS.items()]
)
def test_computer_tool_type_matches_the_model_generation(
    model_id: str, tool_type: str | None
) -> None:
    """Each Claude model promotes ``computer`` to the tool type Bedrock accepts.

    Unreleased model IDs resolve through the same class hierarchy, so a future
    version inherits its family's version instead of falling back to the oldest one.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
         stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_extract_server_tools
    """
    model = _claude_model(model_id)

    assert model.SERVER_TOOL_NAME_TO_TYPE.get("computer") == tool_type


@pytest.mark.parametrize("model_id", _COMPUTER_TOOL_TYPES)
def test_every_claude_model_promotes_the_universally_supported_tools(
    model_id: str,
) -> None:
    """Bash, text editor and memory are server tools on every Claude generation.

    ``text_editor_20250728`` is the Claude 4+ version, whose tool name is
    ``str_replace_based_edit_tool`` rather than the older ``str_replace_editor``.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
         https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool
    """
    tools = _claude_model(model_id).SERVER_TOOL_NAME_TO_TYPE

    assert tools["bash"] == "bash_20250124"
    assert tools["str_replace_based_edit_tool"] == "text_editor_20250728"
    assert tools["memory"] == "memory_20250818"


def test_opus_5_requires_no_computer_use_beta_flag() -> None:
    """Opus 5 advertises no computer use tool, so it needs no computer use beta.

    The ``anthropic_beta`` flag is version-keyed: with no computer tool version in the
    table there is nothing to gate, and sending the flag would be rejected.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
         stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
    """
    model = _claude_model("anthropic.claude-opus-5")

    assert "computer" not in model.TOOL_BETA_FLAGS
    assert "computer" not in model.SERVER_TOOL_NAME_TO_TYPE
    assert model.SERVER_TOOL_NAME_TO_TYPE["bash"] == "bash_20250124", (
        "only the computer tool is missing, not the whole server tool table"
    )


class TestReasoningDisabled:
    """Disabling reasoning is skipped on the models Bedrock rejects it for.

    Claude models default to adaptive thinking; an explicit
    ``reasoning_config: {"type": "disabled"}`` is a Bedrock validation error on the
    generations that always reason, so the gateway drops it and warns instead.

    Ref: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
         stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_reasoning
    """

    @pytest.mark.parametrize(
        "model_id",
        [
            "anthropic.claude-opus-5",
            "anthropic.claude-opus-6",
            "anthropic.claude-sonnet-5",
            "anthropic.claude-sonnet-6",
        ],
    )
    def test_disabled_reasoning_is_forwarded_when_supported(
        self, model_id: str
    ) -> None:
        """Models accepting a disabled configuration receive it."""
        fields: JsonMapping = {}

        _claude_model(model_id)._req_configure_reasoning(fields, enabled=False)  # noqa: SLF001

        assert fields["reasoning_config"] == {"type": "disabled"}

    @pytest.mark.parametrize(
        "model_id",
        [
            "anthropic.claude-fable-5",
            "anthropic.claude-fable-5-1",
            "anthropic.claude-fable-6",
            "anthropic.claude-mythos-5",
            "anthropic.claude-mythos-preview",
            "anthropic.claude-mythos-6",
        ],
    )
    def test_disabled_reasoning_is_dropped_when_the_model_always_reasons(
        self, model_id: str, request_log: EventLog
    ) -> None:
        """Fable and Mythos always reason, so the rejected configuration is dropped with a warning."""
        fields: JsonMapping = {}

        _claude_model(model_id)._req_configure_reasoning(fields, enabled=False)  # noqa: SLF001

        assert not fields
        assert request_log["level"] == "warning"
        assert any(
            "Reasoning cannot be disabled on this model" in str(detail)
            for detail in request_log["error_detail"]
        ), "the dropped configuration must be reported in the request log"


class TestSystemMessageAsMessages:
    """Native mid-conversation system messages are enabled per model family.

    The flag decides whether a ``system``-role entry inside ``messages`` is forwarded
    to Bedrock or folded into the ``system`` field.

    Ref: stdapi/models/chat/_adapters/_anthropic_message.py:_prepare_messages_and_system
    """

    @pytest.mark.parametrize(
        "model_id",
        [
            "anthropic.claude-opus-4-8",
            "anthropic.claude-opus-4-10",
            "anthropic.claude-opus-5",
            "anthropic.claude-sonnet-5",
            "anthropic.claude-haiku-5",
            "anthropic.claude-opus-6",
            "anthropic.claude-fable-5",
            "anthropic.claude-mythos-5",
        ],
    )
    def test_opus_48_and_later_forward_system_messages(self, model_id: str) -> None:
        """Opus 4.8+, Fable and Mythos accept system-role messages natively."""
        assert _claude_model(model_id).SYSTEM_MESSAGE_AS_MESSAGES_SUPPORTED is True

    @pytest.mark.parametrize(
        "model_id",
        [
            # Generations before 4.8 fold system messages, whatever the family.
            "anthropic.claude-opus-4-7",
            "anthropic.claude-opus-4-6",
            "anthropic.claude-haiku-4-5-20251001-v1:0",
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "anthropic.claude-3-opus-20240229-v1:0",
            "anthropic.claude-v2:1",
        ],
    )
    def test_unsupported_models_fold_system_messages(self, model_id: str) -> None:
        """Models rejecting system-role messages keep them folded into `system`."""
        assert _claude_model(model_id).SYSTEM_MESSAGE_AS_MESSAGES_SUPPORTED is False
