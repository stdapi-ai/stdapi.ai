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
from stdapi.monitoring import REQUEST

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastapi import Request

    from stdapi.models.chat._anthropic_claude import AnthropicClaudeChatModel
    from stdapi.monitoring import EventLog
    from stdapi.types import JsonMapping

pytestmark = pytest.mark.local


def _claude_model(model_id: str) -> AnthropicClaudeChatModel:
    """Return the Claude chat model implementation selected for *model_id*."""
    return cast("AnthropicClaudeChatModel", get_chat_model(model_id))


class _StubRequest:
    """Minimal stand-in for ``fastapi.Request`` exposing only ``.headers``."""

    def __init__(self, headers: Mapping[str, str]) -> None:
        self.headers = headers


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


class TestServerToolRePromotion:
    """Server tools keep their native Anthropic definition on every turn (issue #97).

    ``_req_configure_tools`` used to move a server tool's real definition into
    ``additionalModelRequestFields["tools"]`` only on the first turn, leaving a
    schema-less ``toolSpec`` stub in ``toolConfig`` from the second turn on — so
    the model never saw the documented argument names again.  It must now be
    re-promoted regardless of prior ``toolResult`` blocks in history, without
    ever also leaving a same-named stub behind in ``toolConfig`` (Anthropic
    rejects duplicate tool names).

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
         stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
    """

    def test_native_definition_is_repromoted_on_a_toolresult_turn(self) -> None:
        """A server tool is still moved to native format when history has a ``toolResult``."""
        model = _claude_model("anthropic.claude-haiku-4-5-20251001-v1:0")
        tool_config: JsonMapping = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "bash",
                        "inputSchema": {"json": {"type": "object"}},
                    }
                }
            ]
        }
        additional_request_fields: JsonMapping = {}
        server_tools: list[JsonMapping] = [{"name": "bash", "type": "bash_20250124"}]
        # A minimal Bedrock message history containing a toolResult block, the
        # shape that previously kept the tool in schema-less stub mode.
        bedrock_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tooluse_1",
                            "content": [{"text": "ok"}],
                        }
                    }
                ],
            }
        ]

        model._req_configure_tools(  # noqa: SLF001
            tool_config=tool_config,  # type: ignore[arg-type]
            additional_request_fields=additional_request_fields,
            server_tools=server_tools,
            bedrock_messages=bedrock_messages,  # type: ignore[arg-type]
        )

        assert additional_request_fields["tools"] == [
            {"name": "bash", "type": "bash_20250124"}
        ], "the real tool definition must reach additionalModelRequestFields"
        assert not tool_config, (
            "the schema-less toolSpec stub must not remain once the tool is "
            "natively promoted"
        )

    def test_no_duplicate_toolspec_when_history_has_a_tooluse_block(self) -> None:
        """``_req_configure_tools`` itself never leaves a stub for an already-native tool name.

        Realistic turn-2+ history carries a ``toolUse`` block for the tool the
        assistant just invoked, not only the ``toolResult``. History here also
        references a *second*, non-native tool name (``get_time``) so the
        history-based resynthesis branch actually fires: with only the
        native tool's own name in history, ``other_names`` comes out empty and
        the resynthesis branch this test targets never even runs, which is
        exactly what made an earlier version of this test unable to catch a
        reintroduced duplicate. This only checks ``_req_configure_tools``'s
        own output in isolation; see
        ``test_assembled_converse_request_has_no_duplicate_tool_name`` for the
        full-pipeline check and its known limitation on a server-tool-only turn.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview#tool-use-examples
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
             stdapi/models/chat/_default.py:_synthesize_tool_config_from_history
        """
        model = _claude_model("anthropic.claude-haiku-4-5-20251001-v1:0")
        tool_config: JsonMapping = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "bash",
                        "inputSchema": {"json": {"type": "object"}},
                    }
                }
            ]
        }
        additional_request_fields: JsonMapping = {}
        server_tools: list[JsonMapping] = [{"name": "bash", "type": "bash_20250124"}]
        bedrock_messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_1",
                            "name": "bash",
                            "input": {"command": "ls"},
                        }
                    },
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_2",
                            "name": "get_time",
                            "input": {},
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tooluse_1",
                            "content": [{"text": "ok"}],
                        }
                    },
                    {
                        "toolResult": {
                            "toolUseId": "tooluse_2",
                            "content": [{"text": "12:00"}],
                        }
                    },
                ],
            },
        ]

        model._req_configure_tools(  # noqa: SLF001
            tool_config=tool_config,  # type: ignore[arg-type]
            additional_request_fields=additional_request_fields,
            server_tools=server_tools,
            bedrock_messages=bedrock_messages,  # type: ignore[arg-type]
        )

        assert additional_request_fields["tools"] == [
            {"name": "bash", "type": "bash_20250124"}
        ]
        assert tool_config == {
            "tools": [
                {
                    "toolSpec": {
                        "name": "get_time",
                        "inputSchema": {"json": {"type": "object"}},
                    }
                }
            ]
        }, (
            "no toolConfig stub must remain for a tool name already promoted "
            "natively, even though history references that same name -- only "
            "the other, non-native tool name from history gets a stub"
        )

    def test_other_tool_name_in_history_still_gets_a_stub(self) -> None:
        """A non-native tool name still found in history keeps a ``toolConfig`` stub.

        Only the natively-promoted name must be excluded from the synthesized
        fallback; a custom tool the client also used earlier in the conversation
        still needs *some* ``toolConfig`` entry, since its real definition never
        moves to ``additionalModelRequestFields``.

        Ref: stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
             stdapi/models/chat/_default.py:_synthesize_tool_config_from_history
        """
        model = _claude_model("anthropic.claude-haiku-4-5-20251001-v1:0")
        tool_config: JsonMapping = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "bash",
                        "inputSchema": {"json": {"type": "object"}},
                    }
                }
            ]
        }
        additional_request_fields: JsonMapping = {}
        server_tools: list[JsonMapping] = [{"name": "bash", "type": "bash_20250124"}]
        bedrock_messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_1",
                            "name": "get_time",
                            "input": {},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tooluse_1",
                            "content": [{"text": "12:00"}],
                        }
                    }
                ],
            },
        ]

        model._req_configure_tools(  # noqa: SLF001
            tool_config=tool_config,  # type: ignore[arg-type]
            additional_request_fields=additional_request_fields,
            server_tools=server_tools,
            bedrock_messages=bedrock_messages,  # type: ignore[arg-type]
        )

        assert additional_request_fields["tools"] == [
            {"name": "bash", "type": "bash_20250124"}
        ]
        assert tool_config == {
            "tools": [
                {
                    "toolSpec": {
                        "name": "get_time",
                        "inputSchema": {"json": {"type": "object"}},
                    }
                }
            ]
        }, "the non-native tool name from history must still get a permissive stub"

    def test_tool_choice_forcing_the_promoted_tool_is_forwarded_not_dropped(
        self,
    ) -> None:
        """A ``toolChoice`` forcing the native tool moves to ``additionalModelRequestFields``.

        Once ``bash``'s ``toolSpec`` stub is removed for native promotion, no
        entry left in ``toolConfig`` can back a ``toolChoice`` naming it --
        Bedrock rejects a ``toolChoice.tool.name`` with no matching
        ``toolSpec``. The choice must reach the model through
        ``additionalModelRequestFields["tool_choice"]`` instead, and the
        surviving ``toolConfig`` (holding the still-needed ``get_time`` stub)
        must not carry a now-dangling ``toolChoice``.

        Ref: stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
             stdapi/models/chat/_anthropic_claude.py:_forward_tool_choice_to_additional_request_fields
        """
        model = _claude_model("anthropic.claude-haiku-4-5-20251001-v1:0")
        tool_config: JsonMapping = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "bash",
                        "inputSchema": {"json": {"type": "object"}},
                    }
                }
            ],
            "toolChoice": {"tool": {"name": "bash"}},
        }
        additional_request_fields: JsonMapping = {}
        server_tools: list[JsonMapping] = [{"name": "bash", "type": "bash_20250124"}]
        bedrock_messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_1",
                            "name": "get_time",
                            "input": {},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tooluse_1",
                            "content": [{"text": "12:00"}],
                        }
                    }
                ],
            },
        ]

        model._req_configure_tools(  # noqa: SLF001
            tool_config=tool_config,  # type: ignore[arg-type]
            additional_request_fields=additional_request_fields,
            server_tools=server_tools,
            bedrock_messages=bedrock_messages,  # type: ignore[arg-type]
        )

        assert additional_request_fields["tool_choice"] == {
            "type": "tool",
            "name": "bash",
        }
        assert tool_config == {
            "tools": [
                {
                    "toolSpec": {
                        "name": "get_time",
                        "inputSchema": {"json": {"type": "object"}},
                    }
                }
            ]
        }, "no dangling toolChoice for a removed toolSpec may remain in toolConfig"

    def test_tool_choice_auto_survives_on_the_synthesized_tool_config(self) -> None:
        """A non-tool-specific ``toolChoice`` (``auto``/``any``) needs no forwarding.

        ``auto`` stays valid against whatever tools end up in ``toolConfig``,
        including a synthesized stub for a non-native tool name from history,
        so it is left in place rather than also being duplicated into
        ``additionalModelRequestFields``.

        Ref: stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
        """
        model = _claude_model("anthropic.claude-haiku-4-5-20251001-v1:0")
        tool_config: JsonMapping = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "bash",
                        "inputSchema": {"json": {"type": "object"}},
                    }
                }
            ],
            "toolChoice": {"auto": {}},
        }
        additional_request_fields: JsonMapping = {}
        server_tools: list[JsonMapping] = [{"name": "bash", "type": "bash_20250124"}]
        bedrock_messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_1",
                            "name": "get_time",
                            "input": {},
                        }
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tooluse_1",
                            "content": [{"text": "12:00"}],
                        }
                    }
                ],
            },
        ]

        model._req_configure_tools(  # noqa: SLF001
            tool_config=tool_config,  # type: ignore[arg-type]
            additional_request_fields=additional_request_fields,
            server_tools=server_tools,
            bedrock_messages=bedrock_messages,  # type: ignore[arg-type]
        )

        assert "tool_choice" not in additional_request_fields
        assert tool_config == {
            "tools": [
                {
                    "toolSpec": {
                        "name": "get_time",
                        "inputSchema": {"json": {"type": "object"}},
                    }
                }
            ],
            "toolChoice": {"auto": {}},
        }

    async def test_assembled_converse_request_has_no_duplicate_tool_name(self) -> None:
        """The final Converse payload never names a natively-promoted tool twice.

        Drives the full ``_req_configure_tools`` + ``_prepare_converse_request``
        pipeline for a turn-2 conversation mixing a server tool with a custom
        tool: this is the level at which a duplicate would actually reach
        Bedrock, since ``_prepare_converse_request`` is where the base class's
        history-based ``toolConfig`` fallback is applied. The custom tool's name
        must still reach ``toolConfig`` (its definition never moves to
        ``additionalModelRequestFields``), while ``bash`` must appear there
        only, not also as a ``toolConfig`` stub.

        History references ``bash`` (the natively-promoted tool) *and*
        ``get_time`` (a custom tool). Without the exclusion this test guards,
        ``_req_configure_tools`` would leave ``toolConfig`` fully empty, and
        the base class's own ``_synthesize_tool_config_from_history`` fallback
        in ``_prepare_converse_request`` would then resynthesize a stub for
        *every* name found in history -- including ``bash`` -- reproducing the
        duplicate. A history that only mentions ``get_time`` cannot catch this:
        the fallback would only ever regenerate ``get_time``, which never
        collides with ``bash`` regardless of whether the exclusion runs.

        A server-tool-*only* turn-2 conversation (no custom tool anywhere in
        history) is not covered here: closing that gap requires
        ``_synthesize_tool_config_from_history`` in ``_default.py`` — outside
        this module — to skip names already present in
        ``additionalModelRequestFields["tools"]``, since it unconditionally
        resynthesizes a stub for every ``toolUse`` name it finds once
        ``tool_config`` is left empty.

        Ref: stdapi/models/chat/_default.py:ChatModel._prepare_converse_request
             stdapi/models/chat/_default.py:_synthesize_tool_config_from_history
        """
        model = _claude_model("anthropic.claude-haiku-4-5-20251001-v1:0")
        tool_config: JsonMapping = {
            "tools": [
                {
                    "toolSpec": {
                        "name": "bash",
                        "inputSchema": {"json": {"type": "object"}},
                    }
                }
            ]
        }
        additional_request_fields: JsonMapping = {}
        server_tools: list[JsonMapping] = [{"name": "bash", "type": "bash_20250124"}]
        bedrock_messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_1",
                            "name": "bash",
                            "input": {"command": "ls"},
                        }
                    },
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_2",
                            "name": "get_time",
                            "input": {},
                        }
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "tooluse_1",
                            "content": [{"text": "ok"}],
                        }
                    },
                    {
                        "toolResult": {
                            "toolUseId": "tooluse_2",
                            "content": [{"text": "12:00"}],
                        }
                    },
                ],
            },
        ]

        model._req_configure_tools(  # noqa: SLF001
            tool_config=tool_config,  # type: ignore[arg-type]
            additional_request_fields=additional_request_fields,
            server_tools=server_tools,
            bedrock_messages=bedrock_messages,  # type: ignore[arg-type]
        )
        token = REQUEST.set(cast("Request", _StubRequest({})))
        try:
            request = await model._prepare_converse_request(  # noqa: SLF001
                bedrock_messages=bedrock_messages,  # type: ignore[arg-type]
                inference_cfg={},
                system_blocks=None,
                tool_config=tool_config,  # type: ignore[arg-type]
                additional_request_fields=additional_request_fields,
                service_tier=None,
            )
        finally:
            REQUEST.reset(token)

        native_names = {
            tool["name"]
            for tool in request.get("additionalModelRequestFields", {}).get("tools", [])
        }
        config_names = {
            entry["toolSpec"]["name"]
            for entry in request.get("toolConfig", {}).get("tools", [])
        }
        assert native_names == {"bash"}
        assert config_names == {"get_time"}
        assert not (native_names & config_names), (
            "the same tool name must not appear in both toolConfig and "
            "additionalModelRequestFields"
        )


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
