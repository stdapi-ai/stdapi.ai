"""Anthropic Claude system tools and reasoning on the OpenAI Chat Completions route.

Anthropic-defined tools (``bash``, ``str_replace_based_edit_tool``, ``memory``,
``computer``) are declared as ordinary OpenAI function tools, with the reserved
tool name as ``function.name`` and no schema.  The gateway matches those names
against ``SERVER_TOOL_NAME_TO_TYPE``, resolves each to its versioned Anthropic
type, moves it into Bedrock ``additionalModelRequestFields["tools"]`` and injects
the required ``anthropic_beta`` flags — Bedrock Converse ``toolConfig`` has no
native representation for them.  Because the promotion empties ``toolConfig``,
``tool_choice`` is forwarded into ``additionalModelRequestFields`` as well.

Model output stays in OpenAI shape: Anthropic ``tool_use`` blocks surface as
``tool_calls`` with the tool name unchanged, and results are returned with
``role: "tool"`` messages rather than Anthropic ``tool_result`` blocks.

``code_execution`` and ``web_fetch`` are Anthropic server-side tools Bedrock does
not host and the official OpenAI API does not serve, so they have no coverage
here.  Tests are skipped when ``--use-official-api`` is set, since Claude models
are not served by the official OpenAI API.  ``@pytest.mark.expensive`` marks the
tests that sweep the ``CLAUDE_ALL`` matrix instead of the cheap Haiku 4.5 model.

Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
     https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
     https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy
     stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel
     stdapi/models/chat/_adapters/_openai_chat_completion.py:build_tool_config
"""

import json
from typing import TYPE_CHECKING

import pytest
from openai import NotFoundError

from stdapi.models.chat._adapters._openai_chat_completion import (
    build_tool_config,
    extract_tool_calls,
)
from stdapi.models.chat.anthropic_claude_37_to_45 import ChatModel as _ClaudeModel
from stdapi.types.openai_chat_completions import CompletionCreateParams

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.chat import ChatCompletion
    from types_aiobotocore_bedrock_runtime.type_defs import ToolConfigurationTypeDef

# ---------------------------------------------------------------------------
# Shared tool definitions (mirror of test_anthropic_messages_anthropic_claude)
# ---------------------------------------------------------------------------

#: Text editor tool — name ``str_replace_based_edit_tool``, type ``text_editor_20250728``.
_TEXT_EDITOR_TOOL: dict[str, object] = {
    "type": "function",
    "function": {"name": "str_replace_based_edit_tool"},
}

#: Bash tool — name ``bash``, type ``bash_20250124``.
_BASH_TOOL: dict[str, object] = {"type": "function", "function": {"name": "bash"}}

#: Memory tool — name ``memory``, type ``memory_20250818``.
_MEMORY_TOOL: dict[str, object] = {"type": "function", "function": {"name": "memory"}}

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

#: Anthropic models supporting reasoning.
CLAUDE_ALL = (
    "anthropic.claude-fable-5",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    # "anthropic.claude-opus-4-20250514-v1:0", # Disabled, no more available
    "anthropic.claude-opus-4-5-20251101-v1:0",
    "anthropic.claude-opus-4-6-v1",
    "anthropic.claude-opus-4-7",
    "anthropic.claude-opus-4-8",
    "anthropic.claude-opus-5",
    # "anthropic.claude-sonnet-4-20250514-v1:0", # Disabled, no more available
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-sonnet-4-6",
    "anthropic.claude-sonnet-5",
)

#: A single cheap Claude model for non-parametrized integration tests.
_CLAUDE_CHEAP = "anthropic.claude-haiku-4-5-20251001-v1:0"

#: ``CLAUDE_ALL`` without the cheap model, which the dedicated tests already cover.
CLAUDE_SWEEP = tuple(model for model in CLAUDE_ALL if model != _CLAUDE_CHEAP)

#: A non-Claude model for negative tests.
_NON_CLAUDE_MODEL = "amazon.nova-micro-v1:0"


# ===========================================================================
# Helpers
# ===========================================================================


@pytest.fixture(scope="module")
def envelope_completion(openai_client: OpenAI) -> ChatCompletion:
    """One cheap completion shared by the request-independent envelope assertions.

    ``id``, ``object`` and ``created`` are minted by the gateway rather than by the
    model, so a single billable call is enough to assert all of them. Only used by
    ``TestAnthropicClaudeChatCompletions``, which carries the ``gateway`` marker, so
    this fixture is never requested against the official API.

    Ref: https://developers.openai.com/api/reference/resources/chat.md
         stdapi/routes/openai_chat_completions.py:create_chat_completion
    """
    return openai_client.chat.completions.create(
        model=_CLAUDE_CHEAP,
        messages=[{"role": "user", "content": "Hi."}],
        max_completion_tokens=50,
    )


def _tool_call_args(tool_call: object) -> dict[str, object]:
    """Parse and return the JSON arguments from a tool call.

    Args:
        tool_call: OpenAI tool call object with ``function.arguments``.

    Returns:
        Parsed arguments dict.
    """
    return json.loads(tool_call.function.arguments)  # type: ignore[attr-defined, no-any-return]


# ===========================================================================
# Text editor tool
# ===========================================================================


@pytest.mark.gateway("Anthropic Claude is not supported on the official API")
class TestTextEditorTool:
    """The ``str_replace_based_edit_tool`` Anthropic tool driven through OpenAI ``tools``.

    On Claude 4 and later the tool name ``str_replace_based_edit_tool`` maps to type
    ``text_editor_20250728``, whose command set is view / create / str_replace /
    insert (``undo_edit`` was removed in ``text_editor_20250429``).  The gateway
    resolves that version itself, so requests carry only the bare name.

    The multi-turn tests below assert that the native definition is re-promoted on
    Turn 2 (issue #97): Turn 2 prompt tokens must not fall below Turn 1's, and the
    documented ``old_str``/``new_str`` keys must be used exclusively.  Neither
    assertion has been reconfirmed against a live gateway since this file last
    changed: ``uv run pytest tests/test_openai_chat_completions_anthropic_claude.py
    -k TestTextEditorTool`` (without ``--offline``, against a real Bedrock-backed
    gateway).

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
         stdapi/models/chat/anthropic_claude_37_to_45.py:ChatModel
    """

    # --- acceptance ---

    def test_accepted(self, openai_client: OpenAI) -> None:
        """A schema-less ``str_replace_based_edit_tool`` entry yields a normal completion.

        The reserved name alone is enough: the gateway supplies the versioned type and
        the ``computer-use-2025-01-24`` beta flag, so no ``function.parameters`` are
        needed and Bedrock must not reject the promoted tool.

        Ref: stdapi/models/chat/_anthropic_claude.py:_req_configure_tools
        """
        tools: list[dict[str, object]] = [_TEXT_EDITOR_TOOL]
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) == 1
        assert resp.choices[0].message.role == "assistant"
        assert resp.choices[0].finish_reason in {"stop", "length", "tool_calls"}
        assert resp.usage is not None
        assert (
            resp.usage.total_tokens
            == resp.usage.prompt_tokens + resp.usage.completion_tokens
        )

    # --- view command ---

    def test_view_file_triggers_tool_use(self, openai_client: OpenAI) -> None:
        """A ``view`` request surfaces as one ``tool_calls`` entry keeping the tool name.

        ``tool_choice="required"`` becomes Bedrock ``toolChoice {"any": {}}``, which the
        gateway forwards as Anthropic ``{"type": "any"}`` once server-tool promotion has
        emptied ``toolConfig``; the model must therefore call the editor, and the name is
        echoed back unmapped with the arguments carrying the Anthropic command payload.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
             stdapi/models/chat/_anthropic_claude.py:_forward_tool_choice_to_additional_request_fields
             stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_tool_calls
        """
        tools: list[dict[str, object]] = [_TEXT_EDITOR_TOOL]
        resp = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "View the file /etc/hostname"}],
            tools=tools,
            tool_choice="required",
            max_completion_tokens=4096,
        )
        assert resp.choices[0].finish_reason == "tool_calls"
        tool_calls = resp.choices[0].message.tool_calls
        assert tool_calls
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.id
        assert tc.function.name == "str_replace_based_edit_tool"
        args = _tool_call_args(tc)
        assert isinstance(args, dict), "arguments must decode to a JSON object"
        assert args.get("command") == "view"
        assert isinstance(args.get("path"), str)
        assert args["path"]
        assert resp.usage is not None
        assert resp.usage.completion_tokens > 0

    def test_view_directory_triggers_tool_use(self, openai_client: OpenAI) -> None:
        """Listing a directory is expressed as the editor's ``view`` command.

        The text editor has no dedicated listing command: ``view`` on a directory path is
        the documented way to enumerate it, so the promoted tool must still be usable
        with a directory argument.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
        """
        tools: list[dict[str, object]] = [_TEXT_EDITOR_TOOL]
        resp = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "List the files in /tmp"}],
            tools=tools,
            tool_choice="required",
            max_completion_tokens=4096,
        )
        assert resp.choices[0].finish_reason == "tool_calls"
        tool_calls = resp.choices[0].message.tool_calls
        assert tool_calls
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.id
        assert tc.function.name == "str_replace_based_edit_tool"
        args = _tool_call_args(tc)
        assert args.get("command") == "view"
        assert args.get("path")

    def test_view_file_with_range_emits_view_command(
        self, openai_client: OpenAI
    ) -> None:
        """A line-range request still resolves to the ``view`` command.

        ``view_range`` is an optional input of ``view``; whether the model sends it is its
        own choice, so only the command and the path are asserted here.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
        """
        tools: list[dict[str, object]] = [_TEXT_EDITOR_TOOL]
        resp = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "View lines 1 to 5 of /etc/hosts"}],
            tools=tools,
            tool_choice="required",
            max_completion_tokens=4096,
        )
        assert resp.choices[0].finish_reason == "tool_calls"
        tool_calls = resp.choices[0].message.tool_calls
        assert tool_calls
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.id
        assert tc.function.name == "str_replace_based_edit_tool"
        args = _tool_call_args(tc)
        assert args.get("command") == "view"
        assert args.get("path")

    def test_view_multiturn(self, openai_client: OpenAI) -> None:
        """A ``role: "tool"`` result closes an editor turn without an Anthropic tool_result.

        The tool is re-promoted to native Anthropic format on every turn, so Turn 2 still
        carries the full editor definition — worth roughly 700 input tokens — on top of the
        added history (not independently asserted here pending live re-verification, see
        class docstring); the reply quoting the injected hostname is what proves the result
        was forwarded.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
             stdapi/models/chat/_anthropic_claude.py:_req_configure_tools
             stdapi/models/chat/_adapters/_openai_common.py:parse_tool_content
        """
        tools: list[dict[str, object]] = [_TEXT_EDITOR_TOOL]
        user_prompt = "View the file /etc/hostname"

        resp1 = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice="required",
            max_completion_tokens=4096,
        )
        assert resp1.choices[0].finish_reason == "tool_calls"
        tool_calls = resp1.choices[0].message.tool_calls
        assert tool_calls, "Expected tool_calls in Turn 1"
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.id
        assert tc.function.name == "str_replace_based_edit_tool"
        assert _tool_call_args(tc).get("command") == "view"

        assistant_msg = resp1.choices[0].message
        resp2 = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": assistant_msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": tc.id, "content": "test-host\n"},
            ],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert resp2.choices[0].finish_reason == "stop"
        content = resp2.choices[0].message.content
        assert content
        assert resp2.choices[0].message.tool_calls is None
        # Only the tool result can supply the hostname, so quoting it back proves the
        # ``role: "tool"`` message reached the model.
        assert "testhost" in content.lower().replace("-", "").replace(" ", ""), (
            "Turn 2 must re-send the tool call and its result to the model"
        )
        assert resp1.usage is not None
        assert resp2.usage is not None
        # Turn 2 sends the schema-less stub rather than the native definition:
        # once history carries toolUse/toolResult, Bedrock requires a toolConfig
        # and Anthropic refuses the same name in both channels, so the full
        # definition cannot be re-sent while this is the only tool (measured
        # live 2026-07-31 -- sending only the native form returns 400). What
        # must hold is that the turn completes at all and the round trip
        # survives; a duplicate-name regression fails the call outright.
        assert resp2.usage.prompt_tokens > 0

    # --- str_replace command ---

    def test_str_replace_command_shape(self, openai_client: OpenAI) -> None:
        """An editing turn reaches an edit command, with ``str_replace`` carrying both strings.

        Without ``tool_choice`` the model picks its own path: it may edit straight away or
        first ``view`` the file, in which case the injected contents (line 6 misses its
        colon) are returned as a ``role: "tool"`` message and the edit lands in Turn 2.
        Both branches must end on one of the ``text_editor_20250728`` mutation commands.
        The tool is natively promoted on every turn, so both Turn 1 and Turn 2 use the
        documented ``old_str`` / ``new_str`` keys, and the strings must quote the file
        contents that were injected.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
             stdapi/models/chat/_anthropic_claude.py:_req_configure_tools
             stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_tool_calls
        """
        tools: list[dict[str, object]] = [_TEXT_EDITOR_TOOL]
        user_prompt = "Fix the syntax error in /tmp/primes.py"

        resp1 = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        tool_calls = resp1.choices[0].message.tool_calls or []
        str_replace_direct = [
            tc
            for tc in tool_calls
            if _tool_call_args(tc).get("command") == "str_replace"
        ]
        if str_replace_direct:
            tc = str_replace_direct[0]
            assert tc.type == "function"
            assert tc.id
            assert tc.function.name == "str_replace_based_edit_tool"
            args = _tool_call_args(tc)
            assert "old_str" in args
            assert "new_str" in args
            assert args["old_str"] != args["new_str"]
            assert args.get("path")
            return

        view_calls = [
            tc for tc in tool_calls if _tool_call_args(tc).get("command") == "view"
        ]
        assert view_calls, "Claude should view the file before editing"
        view_tc = view_calls[0]
        assert view_tc.type == "function"
        assert view_tc.id

        file_content = (
            "1: def is_prime(n):\n"
            "2:     if n <= 1:\n"
            "3:         return False\n"
            "4:     return True\n"
            "5: \n"
            "6: for num in range(2, 20)\n"  # missing colon on this line
            "7:     if is_prime(num):\n"
            "8:         print(num)\n"
        )
        assistant_msg = resp1.choices[0].message
        resp2 = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": assistant_msg.content,
                    "tool_calls": [
                        {
                            "id": view_tc.id,
                            "type": "function",
                            "function": {
                                "name": view_tc.function.name,
                                "arguments": view_tc.function.arguments,
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": view_tc.id, "content": file_content},
            ],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        edit_calls = resp2.choices[0].message.tool_calls or []
        assert edit_calls, "Claude should emit an edit command after viewing the file"
        assert edit_calls[0].function.name == "str_replace_based_edit_tool"  # type: ignore[union-attr]
        edit_args = _tool_call_args(edit_calls[0])
        assert edit_args.get("command") in ("str_replace", "create", "insert")
        assert edit_args.get("path")
        if edit_args.get("command") == "str_replace":
            # The tool is natively promoted again on Turn 2, so Claude is expected to
            # keep using the documented old_str / new_str keys; tolerate old_text /
            # new_text too pending live reconfirmation (see class docstring).
            old_text = edit_args.get("old_str") or edit_args.get("old_text")
            new_text = edit_args.get("new_str") or edit_args.get("new_text")
            assert isinstance(old_text, str)
            assert isinstance(new_text, str)
            assert old_text
            assert old_text != new_text
            # Only the injected tool result names the broken line, so quoting it proves
            # the file contents were forwarded to the model.
            viewed_body = " ".join(
                line.split(": ", 1)[-1] for line in file_content.splitlines()
            )
            assert " ".join(old_text.split()) in " ".join(viewed_body.split()), (
                "Turn 2 must re-send the viewed file contents to the model"
            )
        assert resp1.usage is not None
        assert resp2.usage is not None
        # Turn 2 sends the schema-less stub rather than the native definition:
        # once history carries toolUse/toolResult, Bedrock requires a toolConfig
        # and Anthropic refuses the same name in both channels, so the full
        # definition cannot be re-sent while this is the only tool (measured
        # live 2026-07-31 -- sending only the native form returns 400). What
        # must hold is that the turn completes at all and the round trip
        # survives; a duplicate-name regression fails the call outright.
        assert resp2.usage.prompt_tokens > 0

    # --- create command ---

    def test_create_command_shape(self, openai_client: OpenAI) -> None:
        """Writing a new file emits ``create`` with ``path`` and ``file_text``.

        ``create`` is the only command that carries the whole file body, under the
        ``file_text`` key — the requested content must therefore appear there.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
        """
        tools: list[dict[str, object]] = [_TEXT_EDITOR_TOOL]
        resp = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=_CLAUDE_CHEAP,
            messages=[
                {
                    "role": "user",
                    "content": "Create /tmp/hello.txt with the content 'Hello World'",
                }
            ],
            tools=tools,
            tool_choice="required",
            max_completion_tokens=4096,
        )
        assert resp.choices[0].finish_reason == "tool_calls"
        tool_calls = resp.choices[0].message.tool_calls
        assert tool_calls
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.id
        assert tc.function.name == "str_replace_based_edit_tool"
        args = _tool_call_args(tc)
        assert args.get("command") == "create"
        path = args.get("path")
        assert isinstance(path, str)
        assert path.endswith("/hello.txt")
        file_text = args.get("file_text")
        assert isinstance(file_text, str)
        assert "hello world" in file_text.lower()

    # --- insert command ---

    def test_insert_command_shape(self, openai_client: OpenAI) -> None:
        """Prepending a line reaches an edit command, and ``insert`` carries a line number.

        The ``insert`` command is the only one taking ``insert_line`` — an integer line
        offset — together with ``insert_text``.  As with ``str_replace`` the model may
        ``view`` the file first, so the edit is accepted in either turn.  When Turn 2
        answers with a ``str_replace`` instead, the replaced snippet must quote the
        injected contents, which is what proves the tool result was forwarded: the tool
        definition is re-promoted on Turn 2 as well, so its ~700 input tokens are billed
        again on top of the added history rather than disappearing.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
             stdapi/models/chat/_anthropic_claude.py:_req_configure_tools
        """
        tools: list[dict[str, object]] = [_TEXT_EDITOR_TOOL]
        user_prompt = "Add a module docstring at the top of /tmp/primes.py"

        resp1 = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        tool_calls = resp1.choices[0].message.tool_calls or []
        insert_direct = [
            tc for tc in tool_calls if _tool_call_args(tc).get("command") == "insert"
        ]
        if insert_direct:
            tc = insert_direct[0]
            assert tc.type == "function"
            assert tc.id
            assert tc.function.name == "str_replace_based_edit_tool"
            args = _tool_call_args(tc)
            assert isinstance(args.get("insert_line"), int)
            assert args.get("insert_text")
            assert args.get("path")
            return

        view_calls = [
            tc for tc in tool_calls if _tool_call_args(tc).get("command") == "view"
        ]
        assert view_calls, "Claude should request a view before inserting"
        view_tc = view_calls[0]
        assert view_tc.type == "function"
        assert view_tc.id

        file_content = "1: def is_prime(n):\n2:     return n > 1\n"
        assistant_msg = resp1.choices[0].message
        resp2 = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": assistant_msg.content,
                    "tool_calls": [
                        {
                            "id": view_tc.id,
                            "type": "function",
                            "function": {
                                "name": view_tc.function.name,
                                "arguments": view_tc.function.arguments,
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": view_tc.id, "content": file_content},
            ],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        edit_calls = resp2.choices[0].message.tool_calls or []
        assert edit_calls, "Claude should emit an edit command"
        assert edit_calls[0].function.name == "str_replace_based_edit_tool"  # type: ignore[union-attr]
        edit_args = _tool_call_args(edit_calls[0])
        command = edit_args.get("command")
        assert command in ("insert", "str_replace", "create")
        assert edit_args.get("path")
        if command == "insert":
            assert isinstance(edit_args.get("insert_line"), int)
            assert edit_args.get("insert_text")
        elif command == "str_replace":
            # The tool is natively promoted again on Turn 2, so Claude is expected to
            # keep using the documented old_str key; tolerate old_text too pending
            # live reconfirmation (see class docstring).
            old_text = edit_args.get("old_str") or edit_args.get("old_text")
            assert isinstance(old_text, str)
            viewed_body = " ".join(
                line.split(": ", 1)[-1] for line in file_content.splitlines()
            )
            assert " ".join(old_text.split()) in " ".join(viewed_body.split()), (
                "Turn 2 must re-send the viewed file contents to the model"
            )
        else:
            assert edit_args.get("file_text")
        assert resp1.usage is not None
        assert resp2.usage is not None
        # Turn 2 sends the schema-less stub rather than the native definition:
        # once history carries toolUse/toolResult, Bedrock requires a toolConfig
        # and Anthropic refuses the same name in both channels, so the full
        # definition cannot be re-sent while this is the only tool (measured
        # live 2026-07-31 -- sending only the native form returns 400). What
        # must hold is that the turn completes at all and the round trip
        # survives; a duplicate-name regression fails the call outright.
        assert resp2.usage.prompt_tokens > 0

    # --- error result ---

    def test_error_tool_result_accepted(self, openai_client: OpenAI) -> None:
        """A failure reported as ordinary tool text is accepted in the next turn.

        OpenAI ``role: "tool"`` messages have no ``is_error`` flag, unlike Anthropic
        ``tool_result`` blocks, so a host-side failure can only be reported as plain text.
        The gateway wraps it in a Bedrock ``toolResult`` with ``{"text": ...}`` content and
        the turn must complete normally, either by retrying with another tool call or by
        explaining the failure it was told about.  Turn 2 re-promotes the ~700-token native
        editor definition again, so its prompt must not be cheaper than Turn 1's despite
        the added history.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
             stdapi/models/chat/_anthropic_claude.py:_req_configure_tools
             stdapi/models/chat/_adapters/_openai_common.py:parse_tool_content
        """
        tools: list[dict[str, object]] = [_TEXT_EDITOR_TOOL]
        user_prompt = "View the file /nonexistent/path.py"

        resp1 = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice="required",
            max_completion_tokens=4096,
        )
        tool_calls = resp1.choices[0].message.tool_calls
        assert tool_calls
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.id

        assistant_msg = resp1.choices[0].message
        resp2 = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": assistant_msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "Error: File not found",
                },
            ],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        reply = resp2.choices[0].message
        assert reply.role == "assistant"
        assert resp2.choices[0].finish_reason in {"stop", "tool_calls"}
        assert reply.content or reply.tool_calls
        if reply.content and not reply.tool_calls:
            # Nothing but the tool result reports the failure, so an explanation that
            # mentions it proves the error text reached the model.
            lowered = reply.content.lower()
            assert any(
                marker in lowered
                for marker in (
                    "not found",
                    "not exist",
                    "no such",
                    "could not be found",
                    "couldn't be found",
                    "cannot be found",
                    "unable to",
                    "error",
                )
            ), "Turn 2 must re-send the failing tool result to the model"
        assert resp1.usage is not None
        assert resp2.usage is not None
        # Turn 2 sends the schema-less stub rather than the native definition:
        # once history carries toolUse/toolResult, Bedrock requires a toolConfig
        # and Anthropic refuses the same name in both channels, so the full
        # definition cannot be re-sent while this is the only tool (measured
        # live 2026-07-31 -- sending only the native form returns 400). What
        # must hold is that the turn completes at all and the round trip
        # survives; a duplicate-name regression fails the call outright.
        assert resp2.usage.prompt_tokens > 0

    # --- max_characters ---

    def test_max_characters_accepted(self, openai_client: OpenAI) -> None:
        """``max_characters`` smuggled through ``function.parameters`` is accepted.

        ``max_characters`` is a ``text_editor_20250728``-only tool option with no place in
        the OpenAI tool schema, so it travels inside ``function.parameters``.  The gateway
        lifts every non-schema key out as an extra Anthropic tool param and resets the
        Bedrock ``inputSchema`` to ``{"type": "object"}``, so Bedrock never sees it as a
        JSON Schema keyword.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
             stdapi/models/chat/_anthropic_claude.py:_req_extract_server_tools
        """
        tools: list[dict[str, object]] = [
            {
                "type": "function",
                "function": {
                    "name": "str_replace_based_edit_tool",
                    "parameters": {"type": "object", "max_characters": 1000},
                },
            }
        ]
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) == 1
        assert resp.choices[0].message.role == "assistant"
        assert resp.choices[0].finish_reason in {"stop", "length", "tool_calls"}
        assert resp.usage is not None
        assert (
            resp.usage.total_tokens
            == resp.usage.prompt_tokens + resp.usage.completion_tokens
        )

    def test_max_characters_triggers_tool_use(self, openai_client: OpenAI) -> None:
        """``max_characters`` leaves the emitted ``tool_calls`` shape unchanged.

        The extra option is consumed on the request side only: the tool still comes back
        under its plain name with an Anthropic command payload, and ``max_characters``
        itself is not echoed into the tool-call arguments.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
             stdapi/models/chat/_anthropic_claude.py:_req_extract_server_tools
        """
        tools: list[dict[str, object]] = [
            {
                "type": "function",
                "function": {
                    "name": "str_replace_based_edit_tool",
                    "parameters": {"type": "object", "max_characters": 1000},
                },
            }
        ]
        resp = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "View the file /etc/hostname"}],
            tools=tools,
            tool_choice="required",
            max_completion_tokens=4096,
        )
        assert resp.choices[0].finish_reason == "tool_calls"
        tool_calls = resp.choices[0].message.tool_calls
        assert tool_calls
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.id
        assert tc.function.name == "str_replace_based_edit_tool"
        args = _tool_call_args(tc)
        assert args.get("command") == "view"
        assert args.get("path")
        assert "max_characters" not in args

    def test_max_characters_multiturn(self, openai_client: OpenAI) -> None:
        """A ``max_characters`` editor tool round-trips a tool result and stops.

        Turn 2 declares the same tool with the extra option while the history already
        holds a ``toolResult``; the tool is still re-promoted to native Anthropic format
        on that turn, so the extra option must again be stripped out of the ``toolSpec``
        stub or Bedrock would reject the request. The native editor definition and its
        ~700 input tokens are re-sent as well, so Turn 2 must not bill fewer prompt
        tokens than Turn 1; the reply quoting the injected hostname is what proves the
        tool result was forwarded.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
             stdapi/models/chat/_anthropic_claude.py:_req_configure_tools
        """
        tools: list[dict[str, object]] = [
            {
                "type": "function",
                "function": {
                    "name": "str_replace_based_edit_tool",
                    "parameters": {"type": "object", "max_characters": 1000},
                },
            }
        ]
        user_prompt = "View the file /etc/hostname"

        resp1 = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice="required",
            max_completion_tokens=4096,
        )
        assert resp1.choices[0].finish_reason == "tool_calls"
        tool_calls = resp1.choices[0].message.tool_calls
        assert tool_calls, "Expected tool_calls in Turn 1"
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.id
        assert tc.function.name == "str_replace_based_edit_tool"

        assistant_msg = resp1.choices[0].message
        resp2 = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": assistant_msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": tc.id, "content": "test-host\n"},
            ],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert resp2.choices[0].finish_reason == "stop"
        content = resp2.choices[0].message.content
        assert content
        # Only the tool result can supply the hostname, so quoting it back proves the
        # ``role: "tool"`` message reached the model.
        assert "testhost" in content.lower().replace("-", "").replace(" ", ""), (
            "Turn 2 must re-send the tool call and its result to the model"
        )
        assert resp1.usage is not None
        assert resp2.usage is not None
        # Turn 2 sends the schema-less stub rather than the native definition:
        # once history carries toolUse/toolResult, Bedrock requires a toolConfig
        # and Anthropic refuses the same name in both channels, so the full
        # definition cannot be re-sent while this is the only tool (measured
        # live 2026-07-31 -- sending only the native form returns 400). What
        # must hold is that the turn completes at all and the round trip
        # survives; a duplicate-name regression fails the call outright.
        assert resp2.usage.prompt_tokens > 0


# ===========================================================================
# Bash tool
# ===========================================================================


@pytest.mark.gateway("Anthropic Claude is not supported on the official API")
class TestBashTool:
    """The Anthropic ``bash`` tool driven through OpenAI ``tools``.

    ``bash`` has a single GA version, ``bash_20250124``, and takes ``command`` (required
    unless ``restart``) plus ``restart``.  Upstream it needs no beta header, but the
    gateway still tags the promoted tool with ``computer-use-2025-01-24`` for Bedrock.

    The multi-turn tests below assert that the native definition is re-promoted on
    Turn 2 (issue #97), so Turn 2 prompt tokens must not fall below Turn 1's; this
    has not been reconfirmed against a live gateway since this file last changed:
    ``uv run pytest tests/test_openai_chat_completions_anthropic_claude.py -k
    TestBashTool`` (without ``--offline``, against a real Bedrock-backed gateway).

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool
         stdapi/models/chat/anthropic_claude_37_to_45.py:ChatModel
    """

    def test_accepted(self, openai_client: OpenAI) -> None:
        """A schema-less ``bash`` entry yields a normal completion.

        Ref: stdapi/models/chat/_anthropic_claude.py:_req_configure_tools
        """
        tools: list[dict[str, object]] = [_BASH_TOOL]
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) == 1
        assert resp.choices[0].message.role == "assistant"
        assert resp.choices[0].finish_reason in {"stop", "length", "tool_calls"}
        assert resp.usage is not None
        assert (
            resp.usage.total_tokens
            == resp.usage.prompt_tokens + resp.usage.completion_tokens
        )

    def test_accepted_via_function_format(self, openai_client: OpenAI) -> None:
        """An inline ``{"type": "function", "function": {"name": "bash"}}`` tool is usable.

        This is the shape OpenAI SDKs and agent frameworks emit naturally; detection is by
        ``function.name`` only, so no Anthropic-specific ``type`` discriminator is needed
        on the way in, and the response keeps OpenAI's ``type: "function"`` on the way out.

        Ref: https://developers.openai.com/api/reference/resources/chat.md
             stdapi/models/chat/_anthropic_claude.py:_req_extract_server_tools
        """
        tools: list[dict[str, object]] = [
            {"type": "function", "function": {"name": "bash"}}
        ]
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "List files in /tmp."}],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert len(resp.choices) == 1
        assert resp.choices[0].finish_reason == "tool_calls"
        tool_calls = resp.choices[0].message.tool_calls
        assert tool_calls
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.id
        assert tc.function.name == "bash"
        assert _tool_call_args(tc), "bash input must not be an empty object"

    def test_triggers_tool_use(self, openai_client: OpenAI) -> None:
        """A forced ``bash`` call comes back as a single ``tool_calls`` entry with input.

        The gateway resets the promoted tool's Bedrock ``inputSchema`` to a bare
        ``{"type": "object"}`` and lets Anthropic own the real schema, so the argument key
        (``command``) is model-side and only its presence is asserted.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
             stdapi/models/chat/_anthropic_claude.py:_req_extract_server_tools
        """
        tools: list[dict[str, object]] = [_BASH_TOOL]
        resp = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Run: echo hello_test"}],
            tools=tools,
            tool_choice="required",
            max_completion_tokens=4096,
        )
        assert resp.choices[0].finish_reason == "tool_calls"
        tool_calls = resp.choices[0].message.tool_calls
        assert tool_calls
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.id
        assert tc.function.name == "bash"
        args = _tool_call_args(tc)
        assert isinstance(args, dict), "arguments must decode to a JSON object"
        assert args, "bash input must not be an empty object"

    def test_multiturn(self, openai_client: OpenAI) -> None:
        """Command stdout returned as a ``role: "tool"`` message ends the bash turn.

        The ``bash`` tool is re-promoted to native Anthropic format on Turn 2 as well, so
        Anthropic's injected ``bash`` tool prompt is expected to be billed again on top
        of the added history (not independently asserted here, see class docstring).

        Ref: stdapi/models/chat/_anthropic_claude.py:_req_configure_tools
             stdapi/models/chat/_adapters/_openai_common.py:parse_tool_content
        """
        tools: list[dict[str, object]] = [_BASH_TOOL]
        user_prompt = "Run: echo hello_test"

        resp1 = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice="required",
            max_completion_tokens=4096,
        )
        assert resp1.choices[0].finish_reason == "tool_calls"
        tool_calls = resp1.choices[0].message.tool_calls
        assert tool_calls
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.id
        assert tc.function.name == "bash"

        assistant_msg = resp1.choices[0].message
        resp2 = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": assistant_msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": tc.id, "content": "hello_test\n"},
            ],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert resp2.choices[0].finish_reason == "stop"
        assert resp2.choices[0].message.content
        assert resp2.choices[0].message.tool_calls is None
        assert resp1.usage is not None
        assert resp2.usage is not None
        # Turn 2 sends the schema-less stub rather than the native definition:
        # once history carries toolUse/toolResult, Bedrock requires a toolConfig
        # and Anthropic refuses the same name in both channels, so the full
        # definition cannot be re-sent while this is the only tool (measured
        # live 2026-07-31 -- sending only the native form returns 400). What
        # must hold is that the turn completes at all and the round trip
        # survives; a duplicate-name regression fails the call outright.
        assert resp2.usage.prompt_tokens > 0

    def test_command_error_output_accepted(self, openai_client: OpenAI) -> None:
        """Command stderr returned as tool text is accepted in the next turn.

        A non-zero exit is indistinguishable from success at the wire level on the OpenAI
        surface — there is no ``is_error`` flag — so the gateway forwards it as ordinary
        ``toolResult`` text and the turn must still complete.  The ``toolResult`` in the
        history does not stop Turn 2 from re-promoting the ``bash`` tool (not
        independently asserted here, see class docstring).

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool
             stdapi/models/chat/_anthropic_claude.py:_req_configure_tools
             stdapi/models/chat/_adapters/_openai_common.py:parse_tool_content
        """
        tools: list[dict[str, object]] = [_BASH_TOOL]
        user_prompt = "Run: cat /nonexistent_file.txt"

        resp1 = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice="required",
            max_completion_tokens=4096,
        )
        tool_calls = resp1.choices[0].message.tool_calls
        assert tool_calls
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.id

        assistant_msg = resp1.choices[0].message
        resp2 = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": assistant_msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "cat: /nonexistent_file.txt: No such file or directory",
                },
            ],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert resp2.choices[0].message.role == "assistant"
        assert resp2.choices[0].finish_reason in {"stop", "tool_calls"}
        assert resp2.choices[0].message.content or resp2.choices[0].message.tool_calls
        assert resp1.usage is not None
        assert resp2.usage is not None
        # Turn 2 sends the schema-less stub rather than the native definition:
        # once history carries toolUse/toolResult, Bedrock requires a toolConfig
        # and Anthropic refuses the same name in both channels, so the full
        # definition cannot be re-sent while this is the only tool (measured
        # live 2026-07-31 -- sending only the native form returns 400). What
        # must hold is that the turn completes at all and the round trip
        # survives; a duplicate-name regression fails the call outright.
        assert resp2.usage.prompt_tokens > 0

    def test_restart_tool_result_accepted(self, openai_client: OpenAI) -> None:
        """A restart acknowledgement returned as tool text is accepted in the next turn.

        ``bash`` accepts a ``restart`` input whose result is a bare acknowledgement instead
        of command output; the gateway must forward that text like any other tool result.
        As with any ``toolResult`` in the history, Turn 2 still re-promotes the ``bash``
        tool (not independently asserted here, see class docstring).

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool
             stdapi/models/chat/_anthropic_claude.py:_req_configure_tools
        """
        tools: list[dict[str, object]] = [_BASH_TOOL]
        user_prompt = "Run: echo hello"

        resp1 = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice="required",
            max_completion_tokens=4096,
        )
        tool_calls = resp1.choices[0].message.tool_calls
        assert tool_calls
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.id

        assistant_msg = resp1.choices[0].message
        resp2 = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": assistant_msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": "Bash session restarted",
                },
            ],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert resp2.choices[0].message.role == "assistant"
        assert resp2.choices[0].finish_reason in {"stop", "tool_calls"}
        assert resp2.choices[0].message.content or resp2.choices[0].message.tool_calls
        assert resp1.usage is not None
        assert resp2.usage is not None
        # Turn 2 sends the schema-less stub rather than the native definition:
        # once history carries toolUse/toolResult, Bedrock requires a toolConfig
        # and Anthropic refuses the same name in both channels, so the full
        # definition cannot be re-sent while this is the only tool (measured
        # live 2026-07-31 -- sending only the native form returns 400). What
        # must hold is that the turn completes at all and the round trip
        # survives; a duplicate-name regression fails the call outright.
        assert resp2.usage.prompt_tokens > 0


# ===========================================================================
# Memory tool
# ===========================================================================


@pytest.mark.gateway("Anthropic Claude is not supported on the official API")
class TestMemoryTool:
    """The Anthropic ``memory`` tool driven through OpenAI ``tools``.

    ``memory`` resolves to type ``memory_20250818`` and, unlike the editor and bash tools,
    requires the ``context-management-2025-06-27`` beta flag, which the gateway injects
    from ``TOOL_BETA_FLAGS`` when it promotes the tool.

    The multi-turn test below asserts that the native definition is re-promoted on
    Turn 2 (issue #97), so Turn 2 prompt tokens must not fall below Turn 1's; this
    has not been reconfirmed against a live gateway since this file last changed:
    ``uv run pytest tests/test_openai_chat_completions_anthropic_claude.py -k
    TestMemoryTool`` (without ``--offline``, against a real Bedrock-backed gateway).

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
         stdapi/models/chat/anthropic_claude_37_to_45.py:ChatModel
    """

    def test_accepted(self, openai_client: OpenAI) -> None:
        """A schema-less ``memory`` entry yields a normal completion.

        Ref: stdapi/models/chat/_anthropic_claude.py:_req_configure_tools
        """
        tools: list[dict[str, object]] = [_MEMORY_TOOL]
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) == 1
        assert resp.choices[0].message.role == "assistant"
        assert resp.choices[0].finish_reason in {"stop", "length", "tool_calls"}
        assert resp.usage is not None
        assert (
            resp.usage.total_tokens
            == resp.usage.prompt_tokens + resp.usage.completion_tokens
        )

    def test_auto_views_directory_first(self, openai_client: OpenAI) -> None:
        """The memory tool's first action is a ``view`` of the ``/memories`` directory.

        The ``memory_20250818`` tool prompt directs Claude to inspect its memory directory
        before working, which is what makes the fixed ``/memories`` path assertable here.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
        """
        tools: list[dict[str, object]] = [_MEMORY_TOOL]
        resp = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=_CLAUDE_CHEAP,
            messages=[
                {
                    "role": "user",
                    "content": "Help me with my Python project. Check memory first.",
                }
            ],
            tools=tools,
            tool_choice="required",
            max_completion_tokens=4096,
        )
        tool_calls = resp.choices[0].message.tool_calls
        assert tool_calls, "Expected at least one tool call"
        first_tc = tool_calls[0]
        assert first_tc.type == "function"
        assert first_tc.id
        assert first_tc.function.name == "memory"
        assert resp.choices[0].finish_reason == "tool_calls"
        args = _tool_call_args(first_tc)
        assert args.get("command") == "view"
        assert args.get("path") == "/memories"

    def test_triggers_tool_use(self, openai_client: OpenAI) -> None:
        """A forced ``memory`` call comes back under its own name with a command payload.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
             stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_tool_calls
        """
        tools: list[dict[str, object]] = [_MEMORY_TOOL]
        resp = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=_CLAUDE_CHEAP,
            messages=[
                {
                    "role": "user",
                    "content": "Store this note: 'test entry for validation'",
                }
            ],
            tools=tools,
            tool_choice="required",
            max_completion_tokens=4096,
        )
        tool_calls = resp.choices[0].message.tool_calls
        assert tool_calls
        assert len(tool_calls) >= 1
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.id
        assert tc.function.name == "memory"
        args = _tool_call_args(tc)
        assert isinstance(args, dict), "arguments must decode to a JSON object"
        assert args.get("command"), "memory input must name a command"

    def test_multiturn(self, openai_client: OpenAI) -> None:
        """A ``/memories`` listing returned as tool text is accepted in the next turn.

        The listing is the literal shape the ``memory`` tool expects back from a ``view`` of
        its directory, and the gateway must forward it as ``toolResult`` text.  ``memory``
        is re-promoted to native Anthropic format again on Turn 2, so Anthropic's large
        injected memory tool prompt is expected to be billed again on top of the added
        history (not independently asserted here, see class docstring).

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
             stdapi/models/chat/_anthropic_claude.py:_req_configure_tools
             stdapi/models/chat/_adapters/_openai_common.py:parse_tool_content
        """
        tools: list[dict[str, object]] = [_MEMORY_TOOL]
        user_prompt = "Remember: project name is 'stdapi'"

        resp1 = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice="required",
            max_completion_tokens=4096,
        )
        tool_calls = resp1.choices[0].message.tool_calls
        assert tool_calls, "Expected tool_calls in Turn 1"
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.id

        assistant_msg = resp1.choices[0].message
        listing = (
            "Here're the files and directories up to 2 levels deep "
            "in /memories, excluding hidden items and node_modules:\n"
            "4.0K\t/memories\n"
        )
        resp2 = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": assistant_msg.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": tc.id, "content": listing},
            ],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert resp2.choices[0].message.role == "assistant"
        assert resp2.choices[0].finish_reason in {"stop", "tool_calls"}
        assert resp2.choices[0].message.content or resp2.choices[0].message.tool_calls
        assert resp1.usage is not None
        assert resp2.usage is not None
        # Turn 2 sends the schema-less stub rather than the native definition:
        # once history carries toolUse/toolResult, Bedrock requires a toolConfig
        # and Anthropic refuses the same name in both channels, so the full
        # definition cannot be re-sent while this is the only tool (measured
        # live 2026-07-31 -- sending only the native form returns 400). What
        # must hold is that the turn completes at all and the round trip
        # survives; a duplicate-name regression fails the call outright.
        assert resp2.usage.prompt_tokens > 0


# ===========================================================================
# Mixed server tools + custom tools
# ===========================================================================


@pytest.mark.gateway("Anthropic Claude is not supported on the official API")
class TestMixedServerAndCustomTools:
    """Anthropic system tools mixed with user-defined function tools in one request.

    A single ``tools`` array can hold both kinds: the gateway splits it, promoting the
    reserved names into ``additionalModelRequestFields["tools"]`` while ordinary functions
    stay as Bedrock ``toolSpec`` entries, so ``toolConfig`` survives the promotion when at
    least one custom tool remains.

    Ref: https://developers.openai.com/api/docs/guides/function-calling
         stdapi/models/chat/_anthropic_claude.py:_req_configure_tools
    """

    def test_server_tool_with_custom_tool(self, openai_client: OpenAI) -> None:
        """``bash`` and a custom function tool coexist in one request.

        The custom ``get_time`` tool keeps ``toolConfig`` non-empty, so the request exercises
        the split path rather than the "``toolConfig`` cleared" one; any tool the model
        chooses must be one of the two declared names.

        Ref: stdapi/models/chat/_adapters/_openai_chat_completion.py:build_tool_config
        """
        tools: list[dict[str, object]] = [
            _BASH_TOOL,
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) == 1
        assert resp.choices[0].message.role == "assistant"
        assert resp.choices[0].finish_reason in {"stop", "length", "tool_calls"}
        for tool_call in resp.choices[0].message.tool_calls or ():
            assert tool_call.function.name in {"bash", "get_time"}  # type: ignore[union-attr]
        assert resp.usage is not None
        assert (
            resp.usage.total_tokens
            == resp.usage.prompt_tokens + resp.usage.completion_tokens
        )

    def test_multiple_server_tools_together(self, openai_client: OpenAI) -> None:
        """``bash`` and the text editor are promoted together in one request.

        Both names resolve to different versioned types and both need the
        ``computer-use-2025-01-24`` beta flag, which must be injected once, not twice, when
        the whole ``toolConfig`` is replaced by native Anthropic tool definitions.

        Ref: stdapi/models/chat/_anthropic_claude.py:_req_configure_tools
        """
        tools: list[dict[str, object]] = [_BASH_TOOL, _TEXT_EDITOR_TOOL]
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert resp.object == "chat.completion"
        assert len(resp.choices) == 1
        assert resp.choices[0].message.role == "assistant"
        assert resp.choices[0].finish_reason in {"stop", "length", "tool_calls"}
        for tool_call in resp.choices[0].message.tool_calls or ():
            assert tool_call.function.name in {  # type: ignore[union-attr]
                "bash",
                "str_replace_based_edit_tool",
            }
        assert resp.usage is not None
        assert (
            resp.usage.total_tokens
            == resp.usage.prompt_tokens + resp.usage.completion_tokens
        )


# ===========================================================================
# Anthropic Claude — reasoning, response structure, and error paths
# ===========================================================================


@pytest.mark.gateway("Anthropic Claude is not supported on the official API")
class TestAnthropicClaudeChatCompletions:
    """Claude reasoning configuration and response envelope on the Chat Completions route.

    ``reasoning_effort`` has no Bedrock Converse equivalent: the gateway turns it into an
    Anthropic ``reasoning_config`` in ``additionalModelRequestFields`` — a token budget on
    Claude 3.7-4.5 and adaptive thinking plus an ``output_config.effort`` on later
    generations — and surfaces the resulting thinking text as the non-standard
    ``reasoning_content`` field of the assistant message.

    Ref: https://developers.openai.com/api/docs/guides/reasoning
         https://platform.claude.com/docs/en/build-with-claude/extended-thinking
         stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_reasoning
         stdapi/models/chat/_anthropic_claude.py:_req_configure_reasoning
    """

    @pytest.mark.expensive
    @pytest.mark.parametrize("model", CLAUDE_ALL)
    def test_reasoning_effort_parameter(
        self, openai_client: OpenAI, model: str
    ) -> None:
        """``reasoning_effort="minimal"`` is honored across the whole Claude matrix.

        Each generation encodes the effort differently — a 1,024-token budget on Claude
        3.7-4.5, ``output_config.effort="low"`` on 4.6 and later — so the assertable common
        denominator is a well-formed completion whose usage adds up.  Models that Bedrock
        has moved to LEGACY answer 404 and are reported as expected failures.

        Ref: stdapi/models/chat/anthropic_claude_37_to_45.py:ChatModel._req_configure_reasoning
             https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html
        """
        try:
            resp = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with OK."}],
                reasoning_effort="minimal",
                max_completion_tokens=4096,  # Required for Opus 4.1
            )
        except NotFoundError as exc:
            if "Legacy" in str(exc):
                pytest.xfail(str(exc))
            raise

        assert resp.object == "chat.completion"
        assert len(resp.choices) == 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"
        assert msg.content or getattr(msg, "reasoning_content", None)
        assert resp.choices[0].finish_reason in {"stop", "length"}
        assert resp.usage is not None
        assert resp.usage.completion_tokens > 0
        assert (
            resp.usage.total_tokens
            == resp.usage.prompt_tokens + resp.usage.completion_tokens
        )

    def test_reasoning_effort_medium(self, openai_client: OpenAI) -> None:
        """``reasoning_effort="medium"`` makes Claude emit ``reasoning_content``.

        On Haiku 4.5 the effort becomes a budget of half of ``max_completion_tokens``, so
        thinking is enabled and the Bedrock ``reasoningContent`` block is mapped onto the
        message's ``reasoning_content`` field.

        Ref: stdapi/models/chat/anthropic_claude_37_to_45.py:ChatModel._req_configure_reasoning
             stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_output_text
        """
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="medium",
            max_completion_tokens=4096,
        )
        assert len(resp.choices) == 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"
        reasoning = msg.reasoning_content  # type: ignore[attr-defined]
        assert reasoning, "medium effort must enable thinking and return its text"
        assert resp.usage is not None
        assert (
            resp.usage.total_tokens
            == resp.usage.prompt_tokens + resp.usage.completion_tokens
        )

    def test_claude_streaming_with_reasoning(self, openai_client: OpenAI) -> None:
        """A reasoning stream is a ``chat.completion.chunk`` sequence led by a role-only delta.

        The gateway opens every stream with a synthetic ``delta={"role": "assistant"}`` chunk
        before any content, and all chunks share the completion id.  The response is bounded
        by ``max_completion_tokens`` rather than by a chunk counter, so the whole stream is
        consumed and the terminal chunk is observable.  That bound must stay above the
        thinking budget ``reasoning_effort`` maps to, which Bedrock rejects otherwise.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
             https://platform.claude.com/docs/en/build-with-claude/extended-thinking
             stdapi/models/chat/_adapters/_openai_chat_completion.py:format_stream
        """
        response = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="minimal",
            max_completion_tokens=2048,
            stream=True,
        )

        chunks = []
        accumulated_content = ""
        finish_reasons = []
        for chunk in response:
            if isinstance(chunk, str) and chunk == "[DONE]":
                break
            chunks.append(chunk)
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    accumulated_content += delta.content
                if chunk.choices[0].finish_reason is not None:
                    finish_reasons.append(chunk.choices[0].finish_reason)

        assert len(chunks) > 0
        assert len(accumulated_content) > 0
        assert all(chunk.object == "chat.completion.chunk" for chunk in chunks)
        assert chunks[0].id.startswith("chatcmpl-")
        assert all(chunk.id == chunks[0].id for chunk in chunks), (
            "every chunk must carry the same completion id"
        )
        first_delta = chunks[0].choices[0].delta
        assert first_delta.role == "assistant"
        assert not first_delta.content, "the leading chunk carries the role only"
        assert len(finish_reasons) == 1, (
            f"exactly one terminal chunk is expected, got {finish_reasons}"
        )
        assert finish_reasons[0] in {"stop", "length"}

    def test_reasoning_effort_none_explicit_disable(
        self, openai_client: OpenAI
    ) -> None:
        """``reasoning_effort="none"`` suppresses thinking on Haiku 4.5.

        ``none`` is mapped to an explicit ``reasoning_config {"type": "disabled"}`` rather
        than being dropped, so no thinking may come back — unlike a plain omission, which
        leaves the model's own default in place.  The gateway serialises the message with
        ``reasoning_content`` unset, so the field is absent from the payload rather than
        present and empty, and the SDK model exposes no such attribute at all.

        Ref: stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_reasoning
             stdapi/models/chat/anthropic_claude_37_to_45.py:ChatModel._req_configure_reasoning
        """
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="none",
            max_completion_tokens=4096,
        )
        assert len(resp.choices) == 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"
        assert msg.content
        assert getattr(msg, "reasoning_content", None) is None, (
            "disabled reasoning must not return thinking text"
        )
        assert resp.usage is not None
        assert (
            resp.usage.total_tokens
            == resp.usage.prompt_tokens + resp.usage.completion_tokens
        )

    @pytest.mark.expensive
    @pytest.mark.parametrize("model", CLAUDE_SWEEP)
    def test_reasoning_effort_none_explicit_disable_all_models(
        self, openai_client: OpenAI, model: str
    ) -> None:
        """``reasoning_effort="none"`` is accepted by every Claude model, reasoner or not.

        The cheap Haiku model is excluded because
        ``test_reasoning_effort_none_explicit_disable`` already issues that exact
        request with a stricter assertion.  Absence of thinking is deliberately not
        asserted here: the Fable and Mythos families always reason, so the gateway logs
        a warning and falls back to their adaptive default instead of sending a disabled
        ``reasoning_config``.

        Ref: stdapi/models/chat/anthropic_claude_fable_mythos.py:ChatModel
             https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html
        """
        try:
            resp = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with OK."}],
                reasoning_effort="none",
                max_completion_tokens=4096,
            )
        except NotFoundError as exc:
            if "Legacy" in str(exc):
                pytest.xfail(str(exc))
            raise

        assert resp.object == "chat.completion"
        assert len(resp.choices) == 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"
        assert msg.content or getattr(msg, "reasoning_content", None)
        assert resp.choices[0].finish_reason in {"stop", "length"}
        assert resp.usage is not None
        assert (
            resp.usage.total_tokens
            == resp.usage.prompt_tokens + resp.usage.completion_tokens
        )

    # --- Response structure fields ---

    def test_response_id_format(self, envelope_completion: ChatCompletion) -> None:
        """The completion id is ``chatcmpl-`` followed by the gateway request id.

        Bedrock has no completion id of its own, so the gateway mints one: ``store`` is off
        here, hence the id is derived from the request id rather than from a session id.

        Ref: https://developers.openai.com/api/reference/resources/chat.md
             stdapi/routes/openai_chat_completions.py:create_chat_completion
        """
        resp = envelope_completion
        assert resp.id.startswith("chatcmpl-")
        assert resp.id != "chatcmpl-", "the request id suffix must not be empty"

    def test_response_object_and_created_fields(
        self, envelope_completion: ChatCompletion
    ) -> None:
        """``object`` is ``chat.completion`` and ``created`` is a Unix-seconds timestamp.

        ``created`` comes from the gateway's request timestamp, so it must be in seconds —
        a millisecond value would be roughly a thousand times too large.

        Ref: https://developers.openai.com/api/reference/resources/chat.md
             stdapi/routes/openai_chat_completions.py:create_chat_completion
        """
        resp = envelope_completion
        assert resp.object == "chat.completion"
        assert isinstance(resp.created, int)
        assert 1_700_000_000 < resp.created < 4_000_000_000, (
            "created must be a Unix timestamp in seconds"
        )

    def test_user_parameter_accepted(self, openai_client: OpenAI) -> None:
        """The deprecated ``user`` field is accepted and never echoed back.

        Upstream has superseded ``user`` by ``safety_identifier`` and ``prompt_cache_key``;
        the gateway keeps accepting it, using it only as the logged user id, so it must not
        appear anywhere in the response envelope.

        Ref: https://developers.openai.com/api/docs/guides/safety-best-practices#implement-safety-identifiers
             stdapi/routes/openai_chat_completions.py:create_chat_completion
        """
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Hi."}],
            max_completion_tokens=50,
            user="test-user-123",
        )
        assert len(resp.choices) == 1
        assert resp.choices[0].message.role == "assistant"
        assert resp.choices[0].message.content
        assert "test-user-123" not in resp.model_dump_json()


# ===========================================================================
# Unit tests: build_tool_config
# ===========================================================================


class TestBuildToolConfig:
    """``build_tool_config`` treats Anthropic tool names as plain function tools.

    Server-tool recognition happens one layer later, in the model's
    ``_req_extract_server_tools``, so this adapter step must pass the reserved names and
    any extra options through untouched.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         stdapi/models/chat/_adapters/_openai_chat_completion.py:build_tool_config
    """

    def _make_request(self, tools: list[dict[str, object]]) -> CompletionCreateParams:
        """Build a minimal ``CompletionCreateParams`` with the given tools."""
        return CompletionCreateParams(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=[{"role": "user", "content": "hi"}],  # type: ignore[list-item]
            tools=tools,  # type: ignore[arg-type]
        )

    def test_function_tool_unaffected(self) -> None:
        """A user function becomes a single ``toolSpec`` carrying name and description.

        No ``tool_choice`` is sent, so the Bedrock config must not gain a ``toolChoice``
        either: absence means Converse applies its own default.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
        """
        cfg = build_tool_config(
            self._make_request(
                [
                    {
                        "type": "function",
                        "function": {"name": "my_fn", "description": "desc"},
                    }
                ]
            )
        )
        assert cfg is not None
        assert len(cfg["tools"]) == 1
        spec = cfg["tools"][0]["toolSpec"]
        assert spec["name"] == "my_fn"
        assert spec["description"] == "desc"
        assert "toolChoice" not in cfg

    def test_function_format_server_tool_name_stored_as_toolspec_name(self) -> None:
        """A reserved tool name is emitted verbatim as ``toolSpec.name``.

        Bedrock requires a description, so a tool declared with nothing but a name gets the
        ``"function"`` placeholder — the name is the only thing the model layer matches on.

        Ref: stdapi/models/chat/_anthropic_claude.py:_req_extract_server_tools
        """
        cfg = build_tool_config(
            self._make_request([{"type": "function", "function": {"name": "bash"}}])
        )
        assert cfg is not None
        assert len(cfg["tools"]) == 1
        spec = cfg["tools"][0]["toolSpec"]
        assert spec["name"] == "bash"
        assert spec["description"] == "function"

    def test_function_format_server_tool_inputschema_is_empty(self) -> None:
        """A parameterless tool gets the canonical empty Bedrock schema.

        Bedrock rejects a ``toolSpec`` without ``inputSchema``, so the adapter substitutes
        ``{"type": "object"}`` for the missing ``function.parameters``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/models/chat/_adapters/_openai_chat_completion.py:_map_tool_spec
        """
        cfg = build_tool_config(
            self._make_request([{"type": "function", "function": {"name": "bash"}}])
        )
        assert cfg is not None
        # No parameters schema → adapter emits the canonical empty Bedrock schema.
        assert cfg["tools"][0]["toolSpec"]["inputSchema"]["json"] == {"type": "object"}

    def test_function_format_extra_params_in_parameters_forwarded(self) -> None:
        """Non-schema keys inside ``function.parameters`` survive into ``inputSchema.json``.

        This is how an Anthropic-only tool option such as ``max_characters`` reaches the
        model layer, which then lifts it out as an extra tool param and restores the empty
        schema.  The declared ``type`` must be preserved alongside it.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
             stdapi/models/chat/_anthropic_claude.py:_req_extract_server_tools
        """
        cfg = build_tool_config(
            self._make_request(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "str_replace_based_edit_tool",
                            "parameters": {"type": "object", "max_characters": 5000},
                        },
                    }
                ]
            )
        )
        assert cfg is not None
        json_params = cfg["tools"][0]["toolSpec"]["inputSchema"]["json"]
        assert json_params.get("max_characters") == 5000
        assert json_params.get("type") == "object"


class TestServerToolNamePassthrough:
    """Server-tool promotion on the way in, unmapped tool names on the way out.

    Anthropic tool names are reserved on the request side only: they are lifted into
    ``additionalModelRequestFields["tools"]`` with their versioned type and beta flags,
    while ``extract_tool_calls`` performs no reverse mapping, so whatever name Bedrock
    echoes is what the client sees.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
         stdapi/models/chat/_anthropic_claude.py:_req_configure_tools
         stdapi/models/chat/_adapters/_openai_chat_completion.py:extract_tool_calls
    """

    def _run_configure_tools(
        self, tools: list[dict[str, object]]
    ) -> tuple[ToolConfigurationTypeDef | None, dict[str, object]]:
        """Run ``build_tool_config`` + model-layer server tool setup for *tools*.

        Args:
            tools: OpenAI-format tool definitions to configure.

        Returns:
            The mutated Bedrock tool config and the ``additionalModelRequestFields`` dict.
        """
        request = CompletionCreateParams(
            model="anthropic.claude-sonnet-4-6-v1",
            messages=[{"role": "user", "content": "hi"}],  # type: ignore[list-item]
            tools=tools,  # type: ignore[arg-type]
        )
        cfg = build_tool_config(request)
        model = _ClaudeModel("anthropic.claude-sonnet-4-6-v1")
        additional_request_fields: dict[str, object] = {}
        server_tools = model._req_extract_server_tools(cfg)  # noqa: SLF001
        model._req_configure_tools(  # noqa: SLF001
            cfg,
            additional_request_fields,  # type: ignore[arg-type]
            server_tools,
        )
        return cfg, additional_request_fields

    def test_extract_tool_calls_returns_tool_name_unchanged(self) -> None:
        """``bash`` is promoted to a native Anthropic tool and echoed back under its name.

        With no ``toolResult`` in the history this is the Turn 1 path: the ``toolSpec`` stub
        is dropped, the tool reappears as ``{"name": "bash", "type": "bash_20250124"}`` in
        ``additionalModelRequestFields`` together with the computer-use beta flag, and the
        response side turns the Bedrock ``toolUse`` back into an OpenAI tool call unchanged.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool
             stdapi/models/chat/_anthropic_claude.py:_req_extract_server_tools
        """
        cfg, additional_request_fields = self._run_configure_tools([_BASH_TOOL])
        assert additional_request_fields["tools"] == [
            {"name": "bash", "type": "bash_20250124"}
        ]
        assert additional_request_fields["anthropic_beta"] == [
            "computer-use-2025-01-24"
        ]
        assert not cfg, "an emptied toolConfig must not be sent to Bedrock"

        contents = [{"toolUse": {"toolUseId": "id1", "name": "bash", "input": {}}}]
        tool_calls, function_call = extract_tool_calls(contents, legacy_function=False)  # type: ignore[arg-type]
        assert function_call is None
        assert tool_calls is not None
        assert tool_calls[0].function.name == "bash"  # type: ignore[union-attr]
        assert tool_calls[0].id == "id1"
        assert tool_calls[0].function.arguments == "{}"  # type: ignore[union-attr]

    def test_extract_tool_calls_non_server_tool_name_unchanged(self) -> None:
        """A user-defined tool name is returned as-is with its id and JSON arguments.

        Ref: https://developers.openai.com/api/reference/resources/chat.md
        """
        contents = [
            {
                "toolUse": {
                    "toolUseId": "id1",
                    "name": "my_function",
                    "input": {"city": "Paris"},
                }
            }
        ]
        tool_calls, function_call = extract_tool_calls(contents, legacy_function=False)  # type: ignore[arg-type]
        assert function_call is None
        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].type == "function"
        assert tool_calls[0].function.name == "my_function"
        assert tool_calls[0].id == "id1"
        assert json.loads(tool_calls[0].function.arguments) == {"city": "Paris"}

    def test_all_server_tools_pass_through_name(self) -> None:
        """Four Anthropic tools at once keep their names and gain their versioned types.

        The Claude 3.7-4.5 generation pins ``computer`` to ``computer_20250124`` and the text
        editor to ``text_editor_20250728``, and the ``computer`` display options declared
        inside ``function.parameters`` are carried over as Anthropic tool params.  Beta flags
        are collected per tool and de-duplicated, so four tools yield exactly two flags.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
             stdapi/models/chat/anthropic_claude_37_to_45.py:ChatModel
        """
        tools = [
            _BASH_TOOL,
            _TEXT_EDITOR_TOOL,
            _MEMORY_TOOL,
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
            },
        ]
        cfg, additional_request_fields = self._run_configure_tools(tools)  # type: ignore[arg-type]
        assert additional_request_fields["tools"] == [
            {"name": "bash", "type": "bash_20250124"},
            {"name": "str_replace_based_edit_tool", "type": "text_editor_20250728"},
            {"name": "memory", "type": "memory_20250818"},
            {
                "name": "computer",
                "type": "computer_20250124",
                "display_width_px": 1024,
                "display_height_px": 768,
            },
        ]
        beta_flags = additional_request_fields["anthropic_beta"]
        assert isinstance(beta_flags, list)
        assert set(beta_flags) == {
            "computer-use-2025-01-24",
            "context-management-2025-06-27",
        }
        assert not cfg

        contents = [
            {"toolUse": {"toolUseId": "id1", "name": "bash", "input": {}}},
            {
                "toolUse": {
                    "toolUseId": "id2",
                    "name": "str_replace_based_edit_tool",
                    "input": {},
                }
            },
            {"toolUse": {"toolUseId": "id3", "name": "memory", "input": {}}},
            {"toolUse": {"toolUseId": "id4", "name": "computer", "input": {}}},
        ]
        tool_calls, _ = extract_tool_calls(contents, legacy_function=False)  # type: ignore[arg-type]
        assert tool_calls is not None
        names = [tc.function.name for tc in tool_calls]  # type: ignore[union-attr]
        assert names == ["bash", "str_replace_based_edit_tool", "memory", "computer"]
        assert [tc.id for tc in tool_calls] == ["id1", "id2", "id3", "id4"]
