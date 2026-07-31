"""Per-generation Claude capability tables: server tools, reasoning, system messages.

Anthropic-provided tools are version-keyed (``computer_20251124``,
``bash_20250124``, ...) and Bedrock accepts only the versions a given Claude
generation was released with, so the gateway keeps a per-model table instead of a
single global mapping.

Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
     stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel
"""

from typing import TYPE_CHECKING, Any, cast

import pytest

from stdapi.models.chat import get_chat_model
from stdapi.models.chat._anthropic_claude import _STUB_INPUT_SCHEMAS
from stdapi.models.chat._default import ChatModel
from stdapi.monitoring import REQUEST
from stdapi.types.anthropic_messages import (
    CacheControlEphemeralParam,
    MessageCreateParams,
    MessageParam,
    TextBlockParam,
    ToolResultBlockParam,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from fastapi import Request
    from types_aiobotocore_bedrock_runtime.type_defs import ToolConfigurationTypeDef

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef
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


class TestClientToolNameCollision:
    """A client function is never mistaken for the Anthropic server tool of that name.

    Server tools are declared by name with no parameters, so the gateway detects
    them by name. A third-party client is free to declare its own function called
    ``bash``, ``memory`` or ``str_replace_editor`` with a real schema -- the pi
    coding agent ships exactly such a ``bash`` tool -- and promoting it would swap
    the client's schema for a typed server tool and forward the schema keys as
    tool configuration. Anthropic rejects that outright:
    ``tools.4.bash_20250124.properties: Extra inputs are not permitted`` (measured
    against a live gateway on 2026-07-31), so a whole class of agentic clients
    could not use the gateway at all.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool
         stdapi/models/chat/_anthropic_claude.py:_req_extract_server_tools
    """

    @staticmethod
    def _tool_config(json_schema: JsonMapping) -> ToolConfigurationTypeDef:
        """Return a tool config declaring a single ``bash`` tool with *json_schema*."""
        return {
            "tools": [
                {"toolSpec": {"name": "bash", "inputSchema": {"json": json_schema}}}
            ]
        }

    def test_schema_bearing_tool_is_left_alone(self) -> None:
        """A ``bash`` tool carrying properties stays a plain client function."""
        model = _claude_model("anthropic.claude-haiku-4-5-20251001-v1:0")
        config = self._tool_config(
            {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            }
        )

        assert model._req_extract_server_tools(config) == [], (  # noqa: SLF001
            "a function declaring its own schema is the client's, not Anthropic's"
        )
        assert config["tools"][0]["toolSpec"]["inputSchema"]["json"]["properties"], (
            "the client's schema must survive untouched"
        )

    def test_schemaless_tool_is_still_promoted(self) -> None:
        """A parameterless ``bash`` entry is still recognised as the server tool."""
        model = _claude_model("anthropic.claude-haiku-4-5-20251001-v1:0")

        promoted = model._req_extract_server_tools(self._tool_config({}))  # noqa: SLF001

        assert promoted == [{"name": "bash", "type": "bash_20250124"}]

    def test_configuration_only_tool_is_still_promoted(self) -> None:
        """Server-tool configuration keys are not a function schema."""
        model = _claude_model("anthropic.claude-haiku-4-5-20251001-v1:0")

        promoted = model._req_extract_server_tools(  # noqa: SLF001
            self._tool_config({"type": "object", "display_width_px": 1024})
        )

        assert promoted == [
            {"name": "bash", "type": "bash_20250124", "display_width_px": 1024}
        ]


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


class TestMultiTurnStubSchema:
    """A retained multi-turn stub carries the documented server tool input schema.

    When history references only server tool names and their stubs are all that
    keeps ``toolConfig`` populated, the native definition cannot be promoted for
    the turn: Bedrock requires a ``toolConfig`` once history carries
    ``toolUse``/``toolResult`` blocks, and Anthropic rejects the same tool name
    in both ``toolConfig`` and the native ``tools`` list ("Tool names must be
    unique"). Issue #97: with a bare ``{"type": "object"}`` stub the model
    invents argument names (measured live: ``old_text``/``new_text`` instead of
    the documented ``old_str``/``new_str``), so the retained stub must carry the
    documented input schema instead.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
         https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool
         https://platform.claude.com/docs/en/agents-and-tools/tool-use/memory-tool
         stdapi/models/chat/_anthropic_claude.py:_apply_stub_input_schemas
    """

    @staticmethod
    def _turn_two_history(tool_name: str) -> list[JsonMapping]:
        """Return a minimal turn-2 Bedrock history referencing only *tool_name*."""
        return [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_1",
                            "name": tool_name,
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
                            "content": [{"text": "ok"}],
                        }
                    }
                ],
            },
        ]

    def _configure(
        self, tool_name: str, tool_type: str
    ) -> tuple[JsonMapping, JsonMapping]:
        """Run ``_req_configure_tools`` for a server-tool-only turn-2 conversation."""
        model = _claude_model("anthropic.claude-haiku-4-5-20251001-v1:0")
        tool_config: JsonMapping = {
            "tools": [
                {
                    "toolSpec": {
                        "name": tool_name,
                        "inputSchema": {"json": {"type": "object"}},
                    }
                }
            ]
        }
        additional_request_fields: JsonMapping = {}
        model._req_configure_tools(  # noqa: SLF001
            tool_config=tool_config,  # type: ignore[arg-type]
            additional_request_fields=additional_request_fields,
            server_tools=[{"name": tool_name, "type": tool_type}],
            bedrock_messages=self._turn_two_history(tool_name),  # type: ignore[arg-type]
        )
        return tool_config, additional_request_fields

    @staticmethod
    def _stub_schema(tool_config: JsonMapping) -> dict[str, Any]:
        """Return the retained stub's ``inputSchema.json`` payload."""
        tools = cast("list[dict[str, Any]]", tool_config["tools"])
        return cast("dict[str, Any]", tools[0]["toolSpec"]["inputSchema"]["json"])

    def test_text_editor_stub_gets_the_documented_schema(self) -> None:
        """The retained editor stub pins the documented commands and argument names."""
        tool_config, additional_request_fields = self._configure(
            "str_replace_based_edit_tool", "text_editor_20250728"
        )

        assert "tools" not in additional_request_fields, (
            "the native definition must not be sent while its stub keeps the "
            "same name in toolConfig"
        )
        schema = self._stub_schema(tool_config)
        assert schema["required"] == ["command", "path"]
        properties = schema["properties"]
        assert properties["command"]["enum"] == [
            "view",
            "create",
            "str_replace",
            "insert",
        ], "undo_edit was removed in text_editor_20250429"
        assert {"old_str", "new_str", "file_text", "insert_line", "insert_text"} <= (
            properties.keys()
        )

    def test_bash_stub_gets_the_documented_schema(self) -> None:
        """The retained bash stub pins the documented ``command``/``restart`` arguments."""
        tool_config, _ = self._configure("bash", "bash_20250124")

        assert set(self._stub_schema(tool_config)["properties"]) == {
            "command",
            "restart",
        }

    def test_memory_stub_gets_the_documented_schema(self) -> None:
        """The retained memory stub pins the documented commands, including ``rename``."""
        tool_config, _ = self._configure("memory", "memory_20250818")

        properties = self._stub_schema(tool_config)["properties"]
        assert "rename" in properties["command"]["enum"]
        assert {"old_path", "new_path"} <= properties.keys()

    def test_unknown_tool_version_keeps_the_permissive_stub(self) -> None:
        """A tool version without a documented schema keeps the permissive stub."""
        tool_config, _ = self._configure(
            "str_replace_based_edit_tool", "text_editor_20991231"
        )

        assert self._stub_schema(tool_config) == {"type": "object"}

    def test_stub_schema_is_a_copy_per_request(self) -> None:
        """Each request gets its own schema copy, so mutations cannot leak across requests."""
        first, _ = self._configure("bash", "bash_20250124")
        second, _ = self._configure("bash", "bash_20250124")

        first_schema = self._stub_schema(first)
        second_schema = self._stub_schema(second)
        assert first_schema == second_schema
        assert first_schema is not second_schema

    @pytest.mark.parametrize("model_id", [*_COMPUTER_TOOL_TYPES, *_FUTURE_MODELS])
    def test_every_promotable_tool_version_has_a_documented_schema(
        self, model_id: str
    ) -> None:
        """No Claude generation may promote a tool version with no documented schema.

        A version missing from the table degrades to the permissive
        ``{"type": "object"}`` stub, which is the exact condition that made the
        model invent argument names. Nothing else would report it: the request
        still succeeds and the answer still looks plausible. Adding a tool
        version to a generation's table therefore fails here until its schema is
        added too.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
             stdapi/models/chat/_anthropic_claude.py:_STUB_INPUT_SCHEMAS
        """
        promotable = set(_claude_model(model_id).SERVER_TOOL_NAME_TO_TYPE.values())

        assert promotable <= _STUB_INPUT_SCHEMAS.keys(), (
            f"{model_id} promotes {sorted(promotable - _STUB_INPUT_SCHEMAS.keys())} "
            "with no documented input schema, so its multi-turn stub falls back "
            "to a schema-less one and the model invents its argument names"
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


class _ToolCacheUnsupportedModel(ChatModel):
    """Chat model with prompt caching but no tool-turn caching (``PROMPT_CACHING_TOOL_SUPPORTED`` stays False)."""

    PROMPT_CACHING_SUPPORTED = True


class TestCreateMessageCachePointLimiting:
    """``create_message`` must enforce cache-point limits like its sibling routes.

    ``create_completion`` and ``create_response`` both call
    ``_req_limit_cache_points`` after placing cache points, which strips any
    ``cachePoint`` from a whole message turn that also carries a ``toolUse``/
    ``toolResult`` block on models without tool caching. The Anthropic adapter's
    own per-block check only inspects the block actually carrying
    ``cache_control``, so a breakpoint on a text block sharing a message with a
    ``tool_result`` block survives unless ``create_message`` makes the same call.

    Ref: stdapi/models/chat/_default.py:ChatModel.create_message
         stdapi/models/chat/_default.py:ChatModel._req_limit_cache_points
    """

    async def test_tool_turn_cache_point_is_stripped_like_the_other_routes(
        self, request_log: EventLog
    ) -> None:
        """A text-block breakpoint sharing a turn with a ``tool_result`` is dropped."""
        del request_log
        model = _ToolCacheUnsupportedModel("model")
        request = MessageCreateParams.model_validate(
            {
                "model": "model",
                "max_tokens": 16,
                "messages": [
                    MessageParam(
                        role="user",
                        content=[
                            ToolResultBlockParam(
                                type="tool_result",
                                tool_use_id="tooluse_1",
                                content="ok",
                            ),
                            TextBlockParam(
                                type="text",
                                text="continue",
                                cache_control=CacheControlEphemeralParam(),
                            ),
                        ],
                    )
                ],
            }
        )
        captured: dict[str, Any] = {}

        async def fake_converse(
            _self: ChatModel, bedrock_request: ConverseRequestBaseTypeDef
        ) -> dict[str, Any]:
            captured.update(bedrock_request)
            return {
                "output": {"message": {"role": "assistant", "content": []}},
                "stopReason": "end_turn",
                "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
            }

        model.converse = fake_converse.__get__(model, _ToolCacheUnsupportedModel)  # type: ignore[method-assign]

        await model.create_message(request, "msg_1")

        content = captured["messages"][0]["content"]
        assert not any("cachePoint" in block for block in content), (
            "a cache point sharing a turn with a tool_result must be dropped on "
            "a model without tool caching, exactly as create_completion and "
            "create_response already do"
        )
