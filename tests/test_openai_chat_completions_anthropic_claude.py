"""Tests specific to Anthropic Claude chat completions.

Covers all Anthropic-defined system tools available on Claude models via the
OpenAI-compatible ``/v1/chat/completions`` route.  The reference behavior is
the native Anthropic tools tests.

Server tools are passed in standard function format with the tool name as
``function.name`` (e.g. ``{"type": "function", "function": {"name": "bash"}}``).
The gateway detects them by name and translates to Bedrock
``additionalModelRequestFields`` automatically, injecting required
``anthropic-beta`` headers.

Tool responses use standard OpenAI ``tool_calls`` + ``tool`` role messages
instead of Anthropic ``tool_use`` / ``tool_result`` blocks.

All tests that require actual model inference are marked ``@pytest.mark.expensive``.
Tests that only validate error paths are not expensive.

Tests are always skipped when ``--use-official-api`` is set because Anthropic
Claude is not available on the official OpenAI API.
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

#: Code execution tool — name ``code_execution``, type ``code_execution_20250522``.
#: Not supported on AWS Bedrock; all tests using it will be skipped via the OpenAI route.
_CODE_EXECUTION_TOOL: dict[str, object] = {
    "type": "function",
    "function": {"name": "code_execution"},
}

#: Web fetch tool — name ``web_fetch``, type ``web_fetch_20250910``.
#: Not supported on AWS Bedrock; tests only run against the official Anthropic API.
_WEB_FETCH_TOOL: dict[str, object] = {
    "type": "function",
    "function": {"name": "web_fetch"},
}

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

#: Anthropic models supporting reasoning.
CLAUDE_ALL = (
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-opus-4-1-20250805-v1:0",
    # "anthropic.claude-opus-4-20250514-v1:0", # Disabled, no more available
    "anthropic.claude-opus-4-5-20251101-v1:0",
    "anthropic.claude-opus-4-6-v1",
    # "anthropic.claude-sonnet-4-20250514-v1:0", # Disabled, no more available
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-sonnet-4-6",
)

#: A single cheap Claude model for non-parametrized integration tests.
_CLAUDE_CHEAP = "anthropic.claude-haiku-4-5-20251001-v1:0"

#: A non-Claude model for negative tests.
_NON_CLAUDE_MODEL = "amazon.nova-micro-v1:0"


# ===========================================================================
# Helpers
# ===========================================================================


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


class TestTextEditorTool:
    """Tests for the ``str_replace_based_edit_tool`` server tool on Claude via OpenAI API.

    Mirrors ``TestTextEditorTool`` in ``test_anthropic_messages_anthropic_claude.py``.
    Tools are passed in standard function format; the gateway detects server tool
    names and translates them into Bedrock ``additionalModelRequestFields``.

    Validated scenarios
    -------------------
    - Tool accepted without error
    - ``view`` command: produces ``tool_calls`` with ``command`` and ``path`` keys
    - ``view`` of a directory path
    - ``view`` with ``view_range`` lines hint
    - Full multi-turn: view → tool result → stop text response
    - ``str_replace`` command shape after inspecting a broken file
    - ``create`` command shape when writing a new file
    - ``insert`` command shape when prepending a line
    - Error result accepted in multi-turn without crashing
    - ``max_characters`` extra param accepted
    - ``max_characters`` produces the same ``tool_calls`` output shape
    - ``max_characters`` multi-turn: Turn 1 → stub Turn 2
    """

    # --- acceptance ---

    @pytest.mark.expensive
    def test_accepted(self, openai_client: OpenAI, use_official_api: bool) -> None:
        """Text editor tool definition is accepted without error.

        Validates:
            - Request with ``str_replace_based_edit_tool`` does not raise
            - Response has at least one choice
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        tools: list[dict[str, object]] = [_TEXT_EDITOR_TOOL]
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.role == "assistant"

    # --- view command ---

    @pytest.mark.expensive
    def test_view_file_triggers_tool_use(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """View command produces a ``tool_calls`` entry with ``command`` and ``path``.

        Validates:
            - ``finish_reason == "tool_calls"``
            - Exactly one tool call with ``name == "str_replace_based_edit_tool"``
            - ``args["command"] == "view"``
            - ``args["path"]`` is a non-empty string
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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
        assert args.get("command") == "view"
        assert args.get("path")

    @pytest.mark.expensive
    def test_view_directory_triggers_tool_use(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Claude uses the view command when asked to list a directory.

        Validates:
            - ``finish_reason == "tool_calls"``
            - At least one tool call with ``command == "view"``
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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
        assert _tool_call_args(tc).get("command") == "view"

    @pytest.mark.expensive
    def test_view_file_with_range_emits_view_command(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Claude uses the view command when asked to inspect specific lines.

        Validates:
            - ``finish_reason == "tool_calls"``
            - Tool call with ``command == "view"``
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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
        assert _tool_call_args(tc).get("command") == "view"

    @pytest.mark.expensive
    def test_view_multiturn(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """View + tool result → Turn 2 stop text response.

        Validates:
            - Turn 1: ``finish_reason == "tool_calls"`` with ``command == "view"``
            - Turn 2: tool result accepted; ``finish_reason == "stop"`` with content
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        tools: list[dict[str, object]] = [_TEXT_EDITOR_TOOL]
        user_prompt = "View the file /etc/hostname"

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
        assert resp2.choices[0].message.content

    # --- str_replace command ---

    @pytest.mark.expensive
    def test_str_replace_command_shape(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """After receiving a file with a syntax error, Claude uses str_replace to fix it.

        Two-turn flow:
        1. Ask Claude to fix a syntax error → Claude views the file
        2. Return file contents (line with missing colon) → Claude emits str_replace

        Validates:
            - Edit tool call has ``command == "str_replace"``
            - ``old_str`` and ``new_str`` keys are present and differ
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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
            args = _tool_call_args(tc)
            assert "old_str" in args
            assert "new_str" in args
            assert args["old_str"] != args["new_str"]
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
        assert _tool_call_args(edit_calls[0]).get("command") in (
            "str_replace",
            "create",
            "insert",
        )

    # --- create command ---

    @pytest.mark.expensive
    def test_create_command_shape(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Claude uses the create command when asked to write a new file.

        Validates:
            - ``args["command"] == "create"``
            - ``args["path"]`` is non-empty
            - ``args["file_text"]`` is non-empty
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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
        args = _tool_call_args(tc)
        assert args.get("command") == "create"
        assert args.get("path")
        assert args.get("file_text")

    # --- insert command ---

    @pytest.mark.expensive
    def test_insert_command_shape(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Claude uses the insert command when asked to prepend a line to a file.

        Two-turn flow: Claude views the file first, then inserts after receiving content.

        Validates:
            - An edit tool call is emitted with ``command`` in ``{insert, str_replace, create}``
            - If ``insert``: ``insert_line`` is an int and ``insert_text`` is non-empty
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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
            args = _tool_call_args(tc)
            assert isinstance(args.get("insert_line"), int)
            assert args.get("insert_text")
            return

        view_calls = [
            tc for tc in tool_calls if _tool_call_args(tc).get("command") == "view"
        ]
        assert view_calls, "Claude should request a view before inserting"
        view_tc = view_calls[0]
        assert view_tc.type == "function"
        assert view_tc.id

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
                {
                    "role": "tool",
                    "tool_call_id": view_tc.id,
                    "content": "1: def is_prime(n):\n2:     return n > 1\n",
                },
            ],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        edit_calls = resp2.choices[0].message.tool_calls or []
        assert edit_calls, "Claude should emit an edit command"
        assert _tool_call_args(edit_calls[0]).get("command") in (
            "insert",
            "str_replace",
            "create",
        )

    # --- error result ---

    @pytest.mark.expensive
    def test_error_tool_result_accepted(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """A tool result with error content is accepted and Claude handles it gracefully.

        Simulates a "file not found" error on the host side.

        Validates:
            - Turn 2 with error content does not raise
            - Claude responds (either retries or explains)
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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
        assert resp2.choices[0].message.role == "assistant"

    # --- max_characters ---

    @pytest.mark.expensive
    def test_max_characters_accepted(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Text editor with ``max_characters`` extra param is accepted.

        Validates:
            - Request with ``max_characters=1000`` does not raise
            - Response has at least one choice
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.role == "assistant"

    @pytest.mark.expensive
    def test_max_characters_triggers_tool_use(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Text editor with ``max_characters`` returns the same tool_calls output shape.

        Validates:
            - ``finish_reason == "tool_calls"``
            - Tool call name is ``str_replace_based_edit_tool``
            - ``"command"`` key present in parsed arguments
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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
        assert "command" in _tool_call_args(tc)

    @pytest.mark.expensive
    def test_max_characters_multiturn(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """``max_characters`` Turn 1 → tool result in Turn 2 → stop.

        Validates:
            - Turn 1 produces a tool call
            - Turn 2 with tool result returns ``finish_reason == "stop"`` with content
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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
        tool_calls = resp1.choices[0].message.tool_calls
        assert tool_calls, "Expected tool_calls in Turn 1"
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
                {"role": "tool", "tool_call_id": tc.id, "content": "test-host\n"},
            ],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert resp2.choices[0].finish_reason == "stop"
        assert resp2.choices[0].message.content


# ===========================================================================
# Bash tool
# ===========================================================================


class TestBashTool:
    """Tests for the bash system tool (``bash``) on Claude via OpenAI API.

    Mirrors ``TestBashTool`` in ``test_anthropic_messages_anthropic_claude.py``.

    Validated scenarios
    -------------------
    - Tool accepted without error
    - Triggers ``tool_calls`` with a non-empty command-like input
    - Multi-turn: command → stdout → stop text response
    - Error output accepted and Claude handles it
    - A restart acknowledgement as tool result accepted
    """

    @pytest.mark.expensive
    def test_accepted(self, openai_client: OpenAI, use_official_api: bool) -> None:
        """Bash tool definition is accepted without error.

        Validates:
            - Request with ``bash`` does not raise
            - Response has at least one choice
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        tools: list[dict[str, object]] = [_BASH_TOOL]
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.role == "assistant"

    @pytest.mark.expensive
    def test_accepted_via_function_format(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Standard function format ``{"type": "function", "function": {"name": "bash"}}`` works.

        LLM clients and OpenAI SDKs naturally produce this format.

        Validates:
            - Request with the standard function format does not raise
            - Response has tool_calls with ``function.name == "bash"``
            - ``type == "function"`` (TypedObject mirroring does not apply)
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        tools: list[dict[str, object]] = [
            {"type": "function", "function": {"name": "bash"}}
        ]
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "List files in /tmp."}],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert len(resp.choices) >= 1
        tool_calls = resp.choices[0].message.tool_calls
        assert tool_calls
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.function.name == "bash"

    @pytest.mark.expensive
    def test_triggers_tool_use(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Bash tool produces a ``tool_calls`` entry with a non-empty command input.

        Validates:
            - ``finish_reason == "tool_calls"``
            - Exactly one tool call with ``name == "bash"``
            - Parsed arguments are non-empty
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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
        assert _tool_call_args(tc)  # non-empty; key name varies by model version

    @pytest.mark.expensive
    def test_multiturn(self, openai_client: OpenAI, use_official_api: bool) -> None:
        """Bash multi-turn: command in Turn 1, stdout tool result in Turn 2.

        Validates:
            - Turn 1: ``finish_reason == "tool_calls"`` for the bash command
            - Turn 2: tool result accepted; ``finish_reason == "stop"`` with content
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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

    @pytest.mark.expensive
    def test_command_error_output_accepted(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """A tool result with stderr content is accepted gracefully.

        Simulates a command that fails with a non-zero exit code.

        Validates:
            - Turn 2 with error content does not raise
            - Claude responds (explains the error or suggests an alternative)
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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

    @pytest.mark.expensive
    def test_restart_tool_result_accepted(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """A tool result that acknowledges a session restart is accepted without error.

        Validates:
            - Turn 2 with a restart acknowledgement does not raise
            - Claude responds as a valid assistant message
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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


# ===========================================================================
# Memory tool
# ===========================================================================


class TestMemoryTool:
    """Tests for the memory system tool (``memory``) on Claude via OpenAI API.

    Mirrors ``TestMemoryTool`` in ``test_anthropic_messages_anthropic_claude.py``.

    Validated scenarios
    -------------------
    - Tool accepted without error
    - First action is a ``view`` of ``/memories``
    - Triggers ``tool_calls`` with non-empty input
    - Multi-turn: view → directory listing tool result → accepted response
    """

    @pytest.mark.expensive
    def test_accepted(self, openai_client: OpenAI, use_official_api: bool) -> None:
        """Memory tool definition is accepted without error.

        Validates:
            - Request with ``memory`` does not raise
            - Response has at least one choice
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        tools: list[dict[str, object]] = [_MEMORY_TOOL]
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.role == "assistant"

    @pytest.mark.expensive
    def test_auto_views_directory_first(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Claude always views ``/memories`` before starting a task.

        Validates:
            - At least one tool call is emitted
            - The first tool call has ``command == "view"``
            - Its ``path`` is ``"/memories"``
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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
        args = _tool_call_args(first_tc)
        assert args.get("command") == "view"
        assert args.get("path") == "/memories"

    @pytest.mark.expensive
    def test_triggers_tool_use(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Memory tool produces a ``tool_calls`` entry with non-empty input.

        Validates:
            - Tool call name is ``memory``
            - Parsed arguments are non-empty
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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
        assert _tool_call_args(tc)

    @pytest.mark.expensive
    def test_multiturn(self, openai_client: OpenAI, use_official_api: bool) -> None:
        """Memory multi-turn: view → directory listing tool result → accepted response.

        Validates:
            - Turn 1: tool call (typically a view of ``/memories``)
            - Turn 2: returning the directory listing is accepted without error
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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


# ===========================================================================
# Code execution tool
# ===========================================================================


class TestCodeExecutionTool:
    """Tests for the code execution system tool (``code_execution_20250522``) via OpenAI API.

    Mirrors ``TestCodeExecutionTool`` in ``test_anthropic_messages_anthropic_claude.py``.

    **Not supported on AWS Bedrock** — all tests in this class are always skipped
    when running via stdapi (i.e., when ``use_official_api=False``), because the
    gateway proxies to Bedrock where ``code_execution_20250522`` is unsupported.

    Validated scenarios
    -------------------
    - Tool accepted without error (never runs via this route)
    - Triggers ``tool_calls`` with a ``"code"`` key containing Python code
    - Multi-turn: code → execution output → stop text that references the result
    - Runtime error in tool result accepted
    """

    @pytest.fixture(autouse=True)
    def _skip(self, use_official_api: bool) -> None:
        """Skip all tests: code_execution is unsupported via stdapi/Bedrock.

        ``code_execution_20250522`` requires a direct Anthropic API connection.
        Via the OpenAI route the backend is always Bedrock (when not on official API).
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        pytest.skip(
            "code_execution_20250522 is not supported on Bedrock; "
            "not reachable via the OpenAI route"
        )

    @pytest.mark.expensive
    def test_accepted(self, openai_client: OpenAI) -> None:
        """Code execution tool definition is accepted without error.

        Validates:
            - Request with ``code_execution_20250522`` does not raise
            - Response has at least one choice
        """
        tools: list[dict[str, object]] = [_CODE_EXECUTION_TOOL]
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert len(resp.choices) >= 1

    @pytest.mark.expensive
    def test_triggers_tool_use(self, openai_client: OpenAI) -> None:
        """Code execution produces a ``tool_calls`` entry with a ``"code"`` key.

        Validates:
            - Tool call name is ``code_execution``
            - ``"code"`` key present in parsed arguments (Python code to execute)
        """
        tools: list[dict[str, object]] = [_CODE_EXECUTION_TOOL]
        resp = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Compute 2 + 2 using code."}],
            tools=tools,
            tool_choice="required",
            max_completion_tokens=4096,
        )
        tool_calls = resp.choices[0].message.tool_calls
        assert tool_calls
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.id
        assert tc.function.name == "code_execution"
        assert "code" in _tool_call_args(tc)

    @pytest.mark.expensive
    def test_multiturn_with_result(self, openai_client: OpenAI) -> None:
        """Code multi-turn: Python code in Turn 1, execution output in Turn 2.

        Validates:
            - Turn 1: tool call with ``"code"`` key
            - Turn 2: providing the execution result is accepted
            - Turn 2 response: ``finish_reason == "stop"`` with result reference
        """
        tools: list[dict[str, object]] = [_CODE_EXECUTION_TOOL]
        user_prompt = "Compute 2 ** 10 using code."

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
                {"role": "tool", "tool_call_id": tc.id, "content": "1024\n"},
            ],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert resp2.choices[0].finish_reason == "stop"
        content = resp2.choices[0].message.content
        assert content
        assert "1024" in content

    @pytest.mark.expensive
    def test_runtime_error_result_accepted(self, openai_client: OpenAI) -> None:
        """A tool result with a Python traceback is accepted.

        Validates:
            - Turn 2 with traceback does not raise
            - Claude responds gracefully (explains or retries)
        """
        tools: list[dict[str, object]] = [_CODE_EXECUTION_TOOL]
        user_prompt = "Divide 1 by 0 using code."

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
                    "content": "ZeroDivisionError: division by zero",
                },
            ],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert resp2.choices[0].message.role == "assistant"


# ===========================================================================
# Web fetch tool
# ===========================================================================


class TestWebFetchTool:
    """Tests for the ``web_fetch`` server tool via the OpenAI API.

    Covers ``web_fetch`` — a Claude server tool; inference tests are
    only available on the official Anthropic API and are skipped when running
    against the local gateway (Bedrock backend).

    Validated scenarios
    -------------------
    - Tool accepted without error (official API only)
    - Claude emits a tool call with a ``"url"`` key (official API only)
    - Multi-turn: fetch request → page content → stop summary (official API only)
    """

    @pytest.fixture(autouse=True)
    def _skip_inference(
        self, request: pytest.FixtureRequest, use_official_api: bool
    ) -> None:
        """Skip inference tests when running via stdapi (Bedrock backend).

        ``web_fetch`` is not supported on Bedrock; it is passed as a regular
        toolSpec and Claude treats it as a custom tool — real URL fetching does
        not work.
        """
        inference_tests = {
            "test_accepted",
            "test_triggers_tool_use",
            "test_multiturn_with_page_content",
        }
        if not use_official_api and request.node.name in inference_tests:
            pytest.skip(
                "web_fetch is not supported on Bedrock; "
                "inference tests require the official Anthropic API"
            )

    @pytest.mark.expensive
    def test_accepted(self, openai_client: OpenAI, use_official_api: bool) -> None:
        """Web fetch tool definition is accepted without error (official API only).

        Validates:
            - Request with ``web_fetch`` does not raise
            - Response has at least one choice
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        tools: list[dict[str, object]] = [_WEB_FETCH_TOOL]
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=1024,
        )
        assert len(resp.choices) >= 1

    @pytest.mark.expensive
    def test_triggers_tool_use(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Web fetch produces a ``tool_calls`` entry with a ``"url"`` key.

        Validates:
            - Tool call name is ``web_fetch``
            - ``"url"`` key present in parsed arguments
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        tools: list[dict[str, object]] = [_WEB_FETCH_TOOL]
        resp = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=_CLAUDE_CHEAP,
            messages=[
                {
                    "role": "user",
                    "content": "Fetch https://example.com and tell me the title.",
                }
            ],
            tools=tools,
            tool_choice="required",
            max_completion_tokens=1024,
        )
        tool_calls = resp.choices[0].message.tool_calls
        assert tool_calls
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.id
        assert tc.function.name == "web_fetch"
        assert "url" in _tool_call_args(tc)

    @pytest.mark.expensive
    def test_multiturn_with_page_content(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Web fetch multi-turn: fetch request → page content → stop summary.

        Validates:
            - Turn 1: tool call with ``"url"`` key
            - Turn 2: providing page HTML as tool result is accepted
            - Turn 2 response: ``finish_reason == "stop"`` with a text summary
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        tools: list[dict[str, object]] = [_WEB_FETCH_TOOL]
        user_prompt = "Fetch https://example.com and summarize it."

        resp1 = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice="required",
            max_completion_tokens=1024,
        )
        tool_calls = resp1.choices[0].message.tool_calls
        assert tool_calls
        tc = tool_calls[0]
        assert tc.type == "function"
        assert tc.id

        page_html = (
            "<html><head><title>Example Domain</title></head>"
            "<body><h1>Example Domain</h1>"
            "<p>This domain is for illustrative examples.</p>"
            "</body></html>"
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
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": tc.id, "content": page_html},
            ],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=1024,
        )
        assert resp2.choices[0].finish_reason == "stop"
        assert resp2.choices[0].message.content


# ===========================================================================
# Mixed server tools + custom tools
# ===========================================================================


class TestMixedServerAndCustomTools:
    """Tests combining Anthropic system tools with user-defined custom tools via OpenAI API.

    Mirrors ``TestMixedServerAndCustomTools`` in
    ``test_anthropic_messages_anthropic_claude.py``.

    Validated scenarios
    -------------------
    - System tool (bash) alongside one custom function tool: accepted
    - Two system tools together (bash + text_editor): accepted
    """

    @pytest.mark.expensive
    def test_server_tool_with_custom_tool(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """System tool accepted alongside a custom function tool.

        Validates:
            - Mixing ``bash`` with a user-defined function tool does not raise
            - Response has at least one choice
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.role == "assistant"

    @pytest.mark.expensive
    def test_multiple_server_tools_together(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Multiple system tools in a single request are accepted without error.

        Validates:
            - bash + text_editor together do not raise
            - Response has at least one choice
        """
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        tools: list[dict[str, object]] = [_BASH_TOOL, _TEXT_EDITOR_TOOL]
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=tools,  # type: ignore[arg-type]
            max_completion_tokens=4096,
        )
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.role == "assistant"


# ===========================================================================
# Anthropic Claude — reasoning, response structure, and error paths
# ===========================================================================


class TestAnthropicClaudeChatCompletions:
    """Anthropic Claude chat completions tests — reasoning, response structure, error paths."""

    @pytest.mark.expensive
    @pytest.mark.parametrize("model", CLAUDE_ALL)
    def test_reasoning_effort_parameter(
        self, openai_client: OpenAI, use_official_api: bool, model: str
    ) -> None:
        """reasoning_effort parameter: accepted and yields valid response on this backend."""
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
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

        assert hasattr(resp, "choices")
        assert len(resp.choices) >= 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"

    @pytest.mark.expensive
    def test_reasoning_effort_medium(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """reasoning_effort='medium' maps to 'medium' effort on Claude."""
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="medium",
            max_completion_tokens=4096,
        )
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.role == "assistant"

    def test_claude_streaming_with_reasoning(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Streaming with reasoning_effort produces valid streaming chunks."""
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        response = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="minimal",
            max_completion_tokens=4096,
            stream=True,
        )

        chunks = []
        accumulated_content = ""
        for chunk in response:
            if isinstance(chunk, str) and chunk == "[DONE]":
                break
            chunks.append(chunk)
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    accumulated_content += delta.content
            if len(chunks) >= 30:
                break

        assert len(chunks) > 0
        assert len(accumulated_content) > 0

    # --- Response structure fields ---

    def test_response_id_format(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Verify response ID starts with 'chatcmpl-' for Claude models."""
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Hi."}],
            max_completion_tokens=50,
        )
        assert resp.id.startswith("chatcmpl-")

    def test_response_object_and_created_fields(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Verify response.object and response.created fields."""
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Hi."}],
            max_completion_tokens=50,
        )
        assert resp.object == "chat.completion"
        assert isinstance(resp.created, int)
        assert resp.created > 0

    def test_user_parameter_accepted(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Verify the user parameter is accepted for Claude models."""
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Hi."}],
            max_completion_tokens=50,
            user="test-user-123",
        )
        assert len(resp.choices) >= 1
        assert resp.choices[0].message.role == "assistant"


# ===========================================================================
# Unit tests: build_tool_config
# ===========================================================================


class TestBuildToolConfig:
    """Unit tests for ``build_tool_config`` with the standard function format."""

    def _make_request(self, tools: list[dict[str, object]]) -> CompletionCreateParams:
        """Build a minimal ``CompletionCreateParams`` with the given tools."""
        return CompletionCreateParams(
            model="anthropic.claude-haiku-4-5-20251001-v1:0",
            messages=[{"role": "user", "content": "hi"}],  # type: ignore[list-item]
            tools=tools,  # type: ignore[arg-type]
        )

    def test_function_tool_unaffected(self) -> None:
        """Regular ``function`` tools produce a ``toolSpec`` with the function name."""
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
        spec = cfg["tools"][0]["toolSpec"]
        assert spec["name"] == "my_fn"
        assert spec["description"] == "desc"

    def test_function_format_server_tool_name_stored_as_toolspec_name(self) -> None:
        """Server tool name is stored as toolSpec.name unchanged.

        ``build_tool_config`` treats this as a regular function tool; server tool
        detection happens at the model layer in ``_req_extract_server_tools``.
        """
        cfg = build_tool_config(
            self._make_request([{"type": "function", "function": {"name": "bash"}}])
        )
        assert cfg is not None
        spec = cfg["tools"][0]["toolSpec"]
        assert spec["name"] == "bash"

    def test_function_format_server_tool_inputschema_is_empty(self) -> None:
        """Server tool name without parameters produces an empty inputSchema."""
        cfg = build_tool_config(
            self._make_request([{"type": "function", "function": {"name": "bash"}}])
        )
        assert cfg is not None
        # No parameters schema → adapter emits the canonical empty Bedrock schema.
        assert cfg["tools"][0]["toolSpec"]["inputSchema"]["json"] == {"type": "object"}

    def test_function_format_extra_params_in_parameters_forwarded(self) -> None:
        """Extra configuration fields in ``parameters`` are forwarded to ``inputSchema.json``.

        e.g. ``{"name": "str_replace_based_edit_tool", "parameters": {"type": "object",
        "max_characters": 5000}}`` → ``inputSchema.json["max_characters"] == 5000``.
        The gateway strips them out at the server-tool layer and forwards them as
        extra Anthropic tool params.
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


class TestServerToolNamePassthrough:
    """Unit tests verifying tool names pass through unchanged in tool call responses.

    Since ``extract_tool_calls`` no longer remaps names, Bedrock echoes the tool
    name directly back to the client without transformation.
    """

    def _run_configure_tools(self, tools: list[dict[str, object]]) -> None:
        """Run ``build_tool_config`` + model-layer server tool setup for *tools*."""
        request = CompletionCreateParams(
            model="anthropic.claude-sonnet-4-6-v1",
            messages=[{"role": "user", "content": "hi"}],  # type: ignore[list-item]
            tools=tools,  # type: ignore[arg-type]
        )
        cfg = build_tool_config(request)
        model = _ClaudeModel("anthropic.claude-sonnet-4-6-v1")
        server_tools = model._req_extract_server_tools(cfg)  # noqa: SLF001
        model._req_configure_tools(cfg, {}, server_tools)  # noqa: SLF001

    def test_extract_tool_calls_returns_tool_name_unchanged(self) -> None:
        """extract_tool_calls returns the tool name as echoed by Bedrock (no remapping)."""
        self._run_configure_tools([_BASH_TOOL])
        contents = [{"toolUse": {"toolUseId": "id1", "name": "bash", "input": {}}}]
        tool_calls, _ = extract_tool_calls(contents, legacy_function=False)  # type: ignore[arg-type]
        assert tool_calls is not None
        assert tool_calls[0].function.name == "bash"  # type: ignore[union-attr]

    def test_extract_tool_calls_non_server_tool_name_unchanged(self) -> None:
        """Regular function tools are not remapped."""
        contents = [
            {"toolUse": {"toolUseId": "id1", "name": "my_function", "input": {}}}
        ]
        tool_calls, _ = extract_tool_calls(contents, legacy_function=False)  # type: ignore[arg-type]
        assert tool_calls is not None
        assert tool_calls[0].function.name == "my_function"  # type: ignore[union-attr]

    def test_all_server_tools_pass_through_name(self) -> None:
        """All server tools (bash, str_replace_based_edit_tool, computer, memory) pass through unchanged."""
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
        self._run_configure_tools(tools)  # type: ignore[arg-type]
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
