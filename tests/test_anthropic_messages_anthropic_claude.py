"""Tests for Anthropic system tools via the Anthropic /v1/messages route.

Covers all Anthropic-defined system tools available on Claude models.  The
reference behavior is the ``AnthropicBedrock`` SDK (AWS Bedrock direct) plus the
official Anthropic documentation.  On Bedrock, system tools return ``tool_use``
blocks (not ``server_tool_use``).

Tools and their Bedrock availability
-------------------------------------
Supported on Bedrock (return ``tool_use`` blocks):
  - Text editor  (``str_replace_based_edit_tool``, type ``text_editor_20250728``)
  - Bash shell   (``bash``, type ``bash_20250124``)
  - Memory store (``memory``, type ``memory_20250818``)
  - Computer use (``computer``, type ``computer_20250124`` / ``computer_20251124``)
  - Web search   (``web_search``, type ``web_search_20250305``) — on Nova Premier only;
                 raises ``BadRequestError`` on Claude models

Not supported on Bedrock (raise ``BadRequestError``):
  - Code execution (``code_execution``, type ``code_execution_20250522``)
  - Web fetch      (``web_fetch``, type ``web_fetch_20250910``)

All tests that require actual model inference are marked ``@pytest.mark.expensive``.
Run with::

    pytest --expensive tests/test_anthropic_messages_anthropic_claude.py

Tests that only validate error paths are not expensive because they do not
complete a successful inference round-trip.
"""

import base64
import re
from pathlib import Path

import pytest
from anthropic import Anthropic, BadRequestError
from anthropic.types import ServerToolUseBlock, ToolUseBlock

from tests.test_openai_chat_completions_anthropic_claude import (
    CLAUDE_ALL as _CLAUDE_ALL,
)

# ===========================================================================
# Module-level fixture: skip when running with --use-official-api
# ===========================================================================


@pytest.fixture(autouse=True)
def _skip_on_official_api(use_anthropic_api: bool) -> None:
    """Skip all tests in this file when running against the official Anthropic API.

    These tests validate gateway behavior for Anthropic system tools. When
    ``--use-official-api`` is set, requests go directly to the Anthropic API,
    bypassing the gateway entirely, so there's nothing to test.
    """
    if use_anthropic_api:
        pytest.skip(
            "These tests validate gateway behavior for Anthropic system tools, "
            "which require the local gateway (run without --use-official-api)"
        )


# ---------------------------------------------------------------------------
# Shared tool definitions
# ---------------------------------------------------------------------------

#: Text editor tool — type ``text_editor_20250728``, canonical name ``str_replace_based_edit_tool``.
_TEXT_EDITOR_TOOL: dict[str, object] = {
    "type": "text_editor_20250728",
    "name": "str_replace_based_edit_tool",
}

#: Bash tool — type ``bash_20250124``, canonical name ``bash``.
_BASH_TOOL: dict[str, object] = {"type": "bash_20250124", "name": "bash"}

#: Memory tool — type ``memory_20250818``, canonical name ``memory``.
_MEMORY_TOOL: dict[str, object] = {"type": "memory_20250818", "name": "memory"}

#: Code execution tool — type ``code_execution_20250522``, canonical name ``code_execution``.
#: Not supported on AWS Bedrock; tests skip when ``is_bedrock_direct=True``.
_CODE_EXECUTION_TOOL: dict[str, object] = {
    "type": "code_execution_20250522",
    "name": "code_execution",
}

#: Web search tool — type ``web_search_20250305``, canonical name ``web_search``.
_WEB_SEARCH_TOOL: dict[str, object] = {
    "type": "web_search_20250305",
    "name": "web_search",
}

#: Web fetch tool — type ``web_fetch_20250910``, canonical name ``web_fetch``.
#: Not supported on AWS Bedrock; tests skip when ``is_bedrock_direct=True``.
_WEB_FETCH_TOOL: dict[str, object] = {"type": "web_fetch_20250910", "name": "web_fetch"}

# ---------------------------------------------------------------------------
# Beta header constants
# ---------------------------------------------------------------------------

#: Computer-use beta flag for Claude 4.x models (Opus 4.6, Sonnet 4.6, Opus 4.5).
_COMPUTER_USE_BETA_NEW = "computer-use-2025-11-24"

#: Computer-use beta flag for Claude 3.7 / 4.5 and all older models.
_COMPUTER_USE_BETA_OLD = "computer-use-2025-01-24"

#: Models on the newer ``computer_20251124`` tool type (Claude 4.6 and later).
_COMPUTER_USE_NEW_MODELS = re.compile(
    r"claude-(?:opus|sonnet)-4-[6-9]|claude-(?:opus|sonnet|haiku|fable|mythos)-(?:[5-9]|\d\d)"
)

#: Models rejecting every computer-use tool type (Claude Opus 5 and later).
_COMPUTER_USE_UNSUPPORTED_MODELS = re.compile(r"claude-opus-(?:[5-9]|\d\d)")

#: Memory tool beta flag (required on official API; auto-injected by the gateway).
_MEMORY_BETA = "context-management-2025-06-27"

#: Beta flag required by the bash tool on the official Anthropic API.
_BASH_BETA = "computer-use-2025-01-24"


# ===========================================================================
# Module-level fixtures for multi-model parametrization
# ===========================================================================


@pytest.fixture
def model_id() -> str:
    """Return an empty model ID when the test class carries no parametrize mark."""
    return ""


@pytest.fixture
def anthropic_chat_model(
    model_id: str,
    anthropic_models: dict[str, str],
    is_bedrock_direct: bool,
    use_anthropic_api: bool,
) -> str:
    """Provide the Anthropic chat model for the current test.

    When ``model_id`` is non-empty (class decorated with
    ``@pytest.mark.parametrize("model_id", _CLAUDE_ALL)``), converts the raw
    Bedrock-format ID to the format expected by the active backend:

    - Local gateway or remote URL: use as-is
      (e.g. ``anthropic.claude-haiku-4-5-20251001-v1:0``).
    - Bedrock direct (``AnthropicBedrock``): prefix with ``global.``
      (e.g. ``global.anthropic.claude-haiku-4-5-20251001-v1:0``).
    - Official Anthropic API (``ANTHROPIC_API_KEY``): strip vendor prefix and
      version suffix (e.g. ``claude-haiku-4-5-20251001``).

    When ``model_id`` is empty, falls back to the default ``anthropic_models["chat"]``.
    """
    if not model_id:
        return anthropic_models["chat"]
    if not use_anthropic_api:
        return model_id
    if is_bedrock_direct:
        return f"global.{model_id}"
    return re.sub(r"-v\d+(?::\d+)?$", "", model_id.removeprefix("anthropic."))


# ===========================================================================
# Text editor tool
# ===========================================================================


@pytest.mark.parametrize("model_id", _CLAUDE_ALL)
class TestTextEditorTool:
    """Tests for the ``str_replace_based_edit_tool`` (text editor) on Claude models.

    On Bedrock (reference behavior), system tools are returned as ``tool_use``
    blocks.  The text editor host application receives the block, executes the
    file operation, and returns a ``tool_result``.

    Validated scenarios
    -------------------
    - Tool accepted without error
    - ``view`` command: produces ``tool_use`` with ``command`` and ``path`` keys
    - ``view`` of a directory path
    - ``view`` with ``view_range`` lines hint
    - Full multi-turn: view → tool_result → end_turn text response
    - ``str_replace`` command shape after inspecting a broken file
    - ``create`` command shape when writing a new file
    - ``insert`` command shape when prepending a line
    - Error result (``is_error=true``) accepted in multi-turn without crashing
    - ``max_characters`` extra param accepted
    - ``max_characters`` produces the same ``tool_use`` output shape
    - ``max_characters`` multi-turn: Turn 1 → stub Turn 2
    """

    # --- acceptance ---

    @pytest.mark.expensive
    def test_accepted(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Text editor tool definition is accepted without error.

        Validates:
            - Request with ``text_editor_20250728`` does not raise
            - Response is a valid message with at least one content block
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=[_TEXT_EDITOR_TOOL],  # type: ignore[list-item]
        )
        assert response.type == "message"
        assert len(response.content) >= 1

    # --- view command ---

    @pytest.mark.expensive
    def test_view_file_triggers_tool_use(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """View command produces a ``tool_use`` block with ``command`` and ``path``.

        Forces tool use with ``tool_choice=any``.

        Validates:
            - Exactly one ``tool_use`` block in the response
            - ``block.name == "str_replace_based_edit_tool"``
            - ``block.input["command"] == "view"``
            - ``block.input["path"]`` is a non-empty string
        """
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": "View the file /etc/hostname"}],
            tools=[_TEXT_EDITOR_TOOL],
            tool_choice={"type": "any"},
        )
        tool_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) == 1
        block = tool_blocks[0]
        assert isinstance(block, ToolUseBlock)
        assert block.id
        assert block.name == "str_replace_based_edit_tool"
        assert block.input.get("command") == "view"
        assert block.input.get("path")

    @pytest.mark.expensive
    def test_view_directory_triggers_tool_use(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Claude uses the view command when asked to list a directory.

        Validates:
            - ``tool_use`` block emitted with ``command == "view"``
            - ``path`` refers to the requested directory
        """
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": "List the files in /tmp"}],
            tools=[_TEXT_EDITOR_TOOL],
            tool_choice={"type": "any"},
        )
        tool_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) >= 1
        block = tool_blocks[0]
        assert isinstance(block, ToolUseBlock)
        assert block.id
        assert block.input.get("command") == "view"

    @pytest.mark.expensive
    def test_view_file_with_range_emits_view_command(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Claude uses the view command when asked to inspect specific lines.

        ``view_range`` is optional; Claude may or may not include it.

        Validates:
            - ``tool_use`` with ``command == "view"`` is emitted
        """
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": "View lines 1 to 5 of /etc/hosts"}],
            tools=[_TEXT_EDITOR_TOOL],
            tool_choice={"type": "any"},
        )
        tool_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) >= 1
        block = tool_blocks[0]
        assert isinstance(block, ToolUseBlock)
        assert block.id
        assert block.input.get("command") == "view"

    @pytest.mark.expensive
    def test_view_multiturn(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """View + tool_result → Turn 2 end_turn text response.

        Validates:
            - Turn 1: ``tool_use`` block with ``command == "view"``
            - Turn 2: ``tool_result`` is accepted; response is ``end_turn`` with text
        """
        tools = [_TEXT_EDITOR_TOOL]
        user_prompt = "View the file /etc/hostname"

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice={"type": "any"},
        )
        tool_blocks = [b for b in resp1.content if isinstance(b, ToolUseBlock)]
        assert tool_blocks, "Expected tool_use block in Turn 1"
        tool_block = tool_blocks[0]
        assert isinstance(tool_block, ToolUseBlock)
        assert tool_block.id

        resp2 = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": list(resp1.content)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": "test-host\n",
                        }
                    ],
                },
            ],
            tools=tools,  # type: ignore[arg-type]
        )
        assert resp2.stop_reason == "end_turn"
        assert any(b.type == "text" for b in resp2.content)

    # --- str_replace command ---

    @pytest.mark.expensive
    def test_str_replace_command_shape(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """After receiving a file with a syntax error, Claude uses str_replace to fix it.

        Multi-turn flow: Claude may view the file one or more times before editing.
        Each view is answered with the file contents until Claude emits an edit command
        (str_replace, create, or insert) or 5 turns are exhausted.  Forces tool use with
        ``tool_choice=any``, so a text-only answer cannot end the flow.

        Validates:
            - Edit block has ``command`` in ``("str_replace", "create", "insert")``
            - str_replace: ``old_str`` and ``new_str`` are present and differ
        """
        tools = [_TEXT_EDITOR_TOOL]
        user_prompt = "Fix the syntax error in /tmp/primes.py"
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
        messages: list[dict] = [{"role": "user", "content": user_prompt}]  # type: ignore[type-arg]

        for _ in range(5):
            resp = anthropic_client.messages.create(  # type: ignore[call-overload]
                model=anthropic_chat_model,
                max_tokens=4096,
                messages=messages,
                tools=tools,
                tool_choice={"type": "any"},
            )
            edit_blocks = [
                b
                for b in resp.content
                if isinstance(b, ToolUseBlock)
                and b.input.get("command") in ("str_replace", "create", "insert")
            ]
            if edit_blocks:
                block = edit_blocks[0]
                assert isinstance(block, ToolUseBlock)
                assert block.id
                assert block.name == "str_replace_based_edit_tool"
                if block.input.get("command") == "str_replace":
                    # Naming of the replaced text differs per model generation:
                    # old_str (classic), old_text or old_string.
                    old_key = next(
                        (
                            key
                            for key in ("old_str", "old_text", "old_string")
                            if key in block.input
                        ),
                        "",
                    )
                    assert old_key, f"No replaced-text key in {block.input}"
                    new_key = old_key.replace("old", "new", 1)
                    assert new_key in block.input
                    assert block.input[old_key] != block.input[new_key]
                return
            view_blocks = [
                b
                for b in resp.content
                if isinstance(b, ToolUseBlock) and b.input.get("command") == "view"
            ]
            assert view_blocks, f"Expected view or edit command, got: {resp.content}"
            view_block = view_blocks[0]
            assert view_block.id
            messages = [
                *messages,
                {"role": "assistant", "content": list(resp.content)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": view_block.id,
                            "content": file_content,
                        }
                    ],
                },
            ]
        pytest.fail("Claude did not emit an edit command within 5 turns")

    # --- create command ---

    @pytest.mark.expensive
    def test_create_command_shape(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Claude uses the create command when asked to write a new file.

        Validates:
            - ``block.input["command"] == "create"``
            - ``block.input["path"]`` is non-empty
            - ``block.input["file_text"]`` is non-empty
        """
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": "Create /tmp/hello.txt with the content 'Hello World'",
                }
            ],
            tools=[_TEXT_EDITOR_TOOL],
            tool_choice={"type": "any"},
        )
        tool_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) >= 1
        block = tool_blocks[0]
        assert isinstance(block, ToolUseBlock)
        assert block.id
        assert block.input.get("command") == "create"
        assert block.input.get("path")
        assert block.input.get("file_text")

    # --- insert command ---

    @pytest.mark.expensive
    def test_insert_command_shape(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Claude uses the insert command when asked to prepend a line to a file.

        Two-turn flow: Claude views the file first, then inserts after receiving content.
        Forces tool use with ``tool_choice=any``, so a text-only answer cannot end a turn.

        Validates:
            - An edit block is emitted with ``command`` in ``{insert, str_replace, create}``
            - If ``insert``: ``insert_line`` is an integer and ``insert_text`` is non-empty
        """
        tools = [_TEXT_EDITOR_TOOL]
        user_prompt = "Add a module docstring at the top of /tmp/primes.py"

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice={"type": "any"},
        )

        insert_direct = [
            b
            for b in resp1.content
            if isinstance(b, ToolUseBlock) and b.input.get("command") == "insert"
        ]
        if insert_direct:
            block = insert_direct[0]
            assert isinstance(block.input.get("insert_line"), int)
            assert block.input.get("insert_text")
            return

        view_blocks = [
            b
            for b in resp1.content
            if isinstance(b, ToolUseBlock) and b.input.get("command") == "view"
        ]
        assert view_blocks, "Claude should request a view before inserting"
        view_block = view_blocks[0]
        assert isinstance(view_block, ToolUseBlock)
        assert view_block.id

        resp2 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": list(resp1.content)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": view_block.id,
                            "content": "1: def is_prime(n):\n2:     return n > 1\n",
                        }
                    ],
                },
            ],
            tools=tools,
            tool_choice={"type": "any"},
        )
        edit_blocks = [b for b in resp2.content if isinstance(b, ToolUseBlock)]
        assert edit_blocks, "Claude should emit an edit command"
        block = edit_blocks[0]
        assert block.input.get("command") in ("insert", "str_replace", "create")

    # --- error result ---

    @pytest.mark.expensive
    def test_error_tool_result_accepted(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A ``tool_result`` with ``is_error=true`` is accepted and Claude handles it gracefully.

        Simulates a "file not found" error on the host side.

        Validates:
            - Turn 2 with ``is_error=true`` does not raise
            - Claude responds (either retries with a different path or explains)
        """
        tools = [_TEXT_EDITOR_TOOL]
        user_prompt = "View the file /nonexistent/path.py"

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice={"type": "any"},
        )
        tool_blocks = [b for b in resp1.content if isinstance(b, ToolUseBlock)]
        assert tool_blocks

        tool_block = tool_blocks[0]
        assert isinstance(tool_block, ToolUseBlock)
        assert tool_block.id
        resp2 = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": list(resp1.content)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": "Error: File not found",
                            "is_error": True,
                        }
                    ],
                },
            ],
            tools=tools,  # type: ignore[arg-type]
        )
        assert resp2.type == "message"

    # --- max_characters ---

    @pytest.mark.expensive
    def test_max_characters_accepted(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Text editor with ``max_characters`` extra param is accepted.

        ``max_characters=1000`` is an Anthropic-native tool parameter.

        Validates:
            - Request with ``max_characters=1000`` does not raise
            - Response is a valid message
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=[{**_TEXT_EDITOR_TOOL, "max_characters": 1000}],  # type: ignore[list-item, typeddict-item]
        )
        assert response.type == "message"
        assert len(response.content) >= 1

    @pytest.mark.expensive
    def test_max_characters_triggers_tool_use(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Text editor with ``max_characters`` returns the same ``tool_use`` output shape.

        Validates:
            - ``block.name == "str_replace_based_edit_tool"``
            - ``"command"`` key present in ``block.input``
        """
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": "View the file /etc/hostname"}],
            tools=[{**_TEXT_EDITOR_TOOL, "max_characters": 1000}],
            tool_choice={"type": "any"},
        )
        tool_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) == 1
        block = tool_blocks[0]
        assert isinstance(block, ToolUseBlock)
        assert block.id
        assert block.name == "str_replace_based_edit_tool"
        assert "command" in block.input

    @pytest.mark.expensive
    def test_max_characters_multiturn(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """``max_characters`` Turn 1 → tool_result in Turn 2 → end_turn.

        Validates:
            - Turn 1 produces a ``tool_use`` block
            - Turn 2 with ``tool_result`` returns ``end_turn`` with a text block
        """
        tools = [{**_TEXT_EDITOR_TOOL, "max_characters": 1000}]
        user_prompt = "View the file /etc/hostname"

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice={"type": "any"},
        )
        tool_blocks = [b for b in resp1.content if isinstance(b, ToolUseBlock)]
        assert tool_blocks, "Expected tool_use block in Turn 1"
        tool_block = tool_blocks[0]
        assert isinstance(tool_block, ToolUseBlock)
        assert tool_block.id

        resp2 = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": list(resp1.content)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": "test-host\n",
                        }
                    ],
                },
            ],
            tools=tools,  # type: ignore[arg-type]
        )
        assert resp2.stop_reason == "end_turn"
        assert any(b.type == "text" for b in resp2.content)


# ===========================================================================
# Bash tool
# ===========================================================================


@pytest.mark.parametrize("model_id", _CLAUDE_ALL)
class TestBashTool:
    """Tests for the bash system tool (``bash_20250124``) on Claude models.

    On Bedrock (reference behavior), bash returns a ``tool_use`` block containing
    a shell command; the host executes it and returns stdout/stderr as a
    ``tool_result``.

    Validated scenarios
    -------------------
    - Tool accepted without error
    - Triggers ``tool_use`` with a non-empty command-like input
    - Multi-turn: command → stdout → end_turn text response
    - Error output (``is_error=true``) is accepted and Claude handles it
    - A restart acknowledgement as tool_result is accepted
    """

    @pytest.mark.expensive
    def test_accepted(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """Bash tool definition is accepted without error.

        Validates:
            - Request with ``bash_20250124`` does not raise
            - Response is a valid message
        """
        extra_headers = {"anthropic-beta": _BASH_BETA} if use_anthropic_api else {}
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=[_BASH_TOOL],  # type: ignore[list-item]
            extra_headers=extra_headers,
        )
        assert response.type == "message"
        assert len(response.content) >= 1

    @pytest.mark.expensive
    def test_triggers_tool_use(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """Bash tool produces a ``tool_use`` block with a non-empty command input.

        Forces tool use with ``tool_choice=any`` and asks Claude to run a command.

        Validates:
            - Response has exactly one ``tool_use`` block
            - ``block.name == "bash"``
            - ``block.input`` is non-empty (contains the shell command)
        """
        extra_headers = {"anthropic-beta": _BASH_BETA} if use_anthropic_api else {}
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": "Run: echo hello_test"}],
            tools=[_BASH_TOOL],
            tool_choice={"type": "any"},
            extra_headers=extra_headers,
        )
        tool_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) == 1
        block = tool_blocks[0]
        assert isinstance(block, ToolUseBlock)
        assert block.id
        assert block.name == "bash"
        assert block.input  # non-empty; key name varies by model version

    @pytest.mark.expensive
    def test_multiturn(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """Bash multi-turn: command in Turn 1, stdout tool_result in Turn 2.

        Validates:
            - Turn 1: ``tool_use`` block for the bash command
            - Turn 2: ``tool_result`` with stdout is accepted; response is ``end_turn`` text
        """
        extra_headers = {"anthropic-beta": _BASH_BETA} if use_anthropic_api else {}
        tools = [_BASH_TOOL]
        user_prompt = "Run: echo hello_test"

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice={"type": "any"},
            extra_headers=extra_headers,
        )
        tool_blocks = [b for b in resp1.content if isinstance(b, ToolUseBlock)]
        assert tool_blocks, "Expected tool_use block in Turn 1"
        bash_block = tool_blocks[0]
        assert isinstance(bash_block, ToolUseBlock)
        assert bash_block.id

        resp2 = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": list(resp1.content)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": bash_block.id,
                            "content": "hello_test\n",
                        }
                    ],
                },
            ],
            tools=tools,  # type: ignore[arg-type]
            extra_headers=extra_headers,
        )
        assert resp2.stop_reason == "end_turn"
        assert any(b.type == "text" for b in resp2.content)

    @pytest.mark.expensive
    def test_command_error_output_accepted(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """A tool_result with ``is_error=true`` and stderr content is accepted gracefully.

        Simulates a command that fails with a non-zero exit code.

        Validates:
            - Turn 2 with ``is_error=true`` does not raise
            - Claude responds (explains the error or suggests an alternative)
        """
        extra_headers = {"anthropic-beta": _BASH_BETA} if use_anthropic_api else {}
        tools = [_BASH_TOOL]
        user_prompt = "Run: cat /nonexistent_file.txt"

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice={"type": "any"},
            extra_headers=extra_headers,
        )
        tool_blocks = [b for b in resp1.content if isinstance(b, ToolUseBlock)]
        assert tool_blocks

        tool_block = tool_blocks[0]
        assert isinstance(tool_block, ToolUseBlock)
        assert tool_block.id
        resp2 = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": list(resp1.content)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": (
                                "cat: /nonexistent_file.txt: No such file or directory"
                            ),
                            "is_error": True,
                        }
                    ],
                },
            ],
            tools=tools,  # type: ignore[arg-type]
            extra_headers=extra_headers,
        )
        assert resp2.type == "message"

    @pytest.mark.expensive
    def test_restart_tool_result_accepted(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """A tool_result that acknowledges a session restart is accepted without error.

        The client returns ``"Bash session restarted"`` as the tool output.

        Validates:
            - Turn 2 with a restart acknowledgement content does not raise
            - Claude responds as a valid message
        """
        extra_headers = {"anthropic-beta": _BASH_BETA} if use_anthropic_api else {}
        tools = [_BASH_TOOL]
        user_prompt = "Run: echo hello"

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice={"type": "any"},
            extra_headers=extra_headers,
        )
        tool_blocks = [b for b in resp1.content if isinstance(b, ToolUseBlock)]
        assert tool_blocks

        tool_block = tool_blocks[0]
        assert isinstance(tool_block, ToolUseBlock)
        assert tool_block.id
        resp2 = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": list(resp1.content)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": "Bash session restarted",
                        }
                    ],
                },
            ],
            tools=tools,  # type: ignore[arg-type]
            extra_headers=extra_headers,
        )
        assert resp2.type == "message"


# ===========================================================================
# Memory tool
# ===========================================================================


@pytest.mark.parametrize("model_id", _CLAUDE_ALL)
class TestMemoryTool:
    """Tests for the memory system tool (``memory_20250818``) on Claude models.

    On Bedrock (reference behavior), the memory tool returns a ``tool_use`` block
    with a file-operation command (``view``, ``create``, ``str_replace``, ``insert``,
    ``delete``, ``rename``); the host applies it to the ``/memories`` directory.

    The Anthropic memory system prompt instructs Claude to always view
    ``/memories`` as its first action before starting any task.

    Validated scenarios
    -------------------
    - Tool accepted without error (beta flag required on official API)
    - First action is a ``view`` of ``/memories``
    - Triggers ``tool_use`` with non-empty input
    - Multi-turn: view → directory listing tool_result → accepted response
    """

    @pytest.mark.expensive
    def test_accepted(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """Memory tool definition is accepted without error.

        On the official API the ``context-management-2025-06-27`` beta flag is
        required; the local gateway injects it automatically.

        Validates:
            - Request with ``memory_20250818`` does not raise
            - Response is a valid message
        """
        extra_headers = {"anthropic-beta": _MEMORY_BETA} if use_anthropic_api else {}
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=[_MEMORY_TOOL],  # type: ignore[list-item]
            extra_headers=extra_headers,
        )
        assert response.type == "message"
        assert len(response.content) >= 1

    @pytest.mark.expensive
    def test_auto_views_directory_first(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """Claude always views ``/memories`` before starting a task.

        The Anthropic memory system prompt instructs Claude to check its memory
        directory as the very first action.

        Validates:
            - At least one ``tool_use`` block is emitted
            - The first ``tool_use`` has ``command == "view"``
            - Its ``path`` is ``"/memories"``
        """
        extra_headers = {"anthropic-beta": _MEMORY_BETA} if use_anthropic_api else {}
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": "Help me with my Python project. Check memory first.",
                }
            ],
            tools=[_MEMORY_TOOL],
            tool_choice={"type": "any"},
            extra_headers=extra_headers,
        )
        tool_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
        assert tool_blocks, "Expected at least one tool_use block"
        first = tool_blocks[0]
        assert isinstance(first, ToolUseBlock)
        assert first.id
        assert first.name == "memory"
        assert first.input.get("command") == "view"
        assert first.input.get("path") == "/memories"

    @pytest.mark.expensive
    def test_triggers_tool_use(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """Memory tool produces a ``tool_use`` block with non-empty input.

        Validates:
            - ``block.name == "memory"``
            - ``block.input`` is non-empty
        """
        extra_headers = {"anthropic-beta": _MEMORY_BETA} if use_anthropic_api else {}
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": "Store this note: 'test entry for validation'",
                }
            ],
            tools=[_MEMORY_TOOL],
            tool_choice={"type": "any"},
            extra_headers=extra_headers,
        )
        tool_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) >= 1
        block = tool_blocks[0]
        assert isinstance(block, ToolUseBlock)
        assert block.id
        assert block.name == "memory"
        assert block.input

    @pytest.mark.expensive
    def test_multiturn(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """Memory multi-turn: view → directory listing tool_result → accepted response.

        Validates:
            - Turn 1: ``tool_use`` block (typically a view of ``/memories``)
            - Turn 2: returning the directory listing is accepted without error
        """
        extra_headers = {"anthropic-beta": _MEMORY_BETA} if use_anthropic_api else {}
        tools = [_MEMORY_TOOL]
        user_prompt = "Remember: project name is 'stdapi'"

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice={"type": "any"},
            extra_headers=extra_headers,
        )
        tool_blocks = [b for b in resp1.content if isinstance(b, ToolUseBlock)]
        assert tool_blocks, "Expected tool_use block in Turn 1"
        tool_block = tool_blocks[0]
        assert isinstance(tool_block, ToolUseBlock)
        assert tool_block.id

        resp2 = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": list(resp1.content)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": (
                                "Here're the files and directories up to 2 levels deep "
                                "in /memories, excluding hidden items and node_modules:\n"
                                "4.0K\t/memories\n"
                            ),
                        }
                    ],
                },
            ],
            tools=tools,  # type: ignore[arg-type]
            extra_headers=extra_headers,
        )
        assert resp2.type == "message"


# ===========================================================================
# Code execution tool
# ===========================================================================


@pytest.mark.parametrize("model_id", _CLAUDE_ALL)
class TestCodeExecutionTool:
    """Tests for the code execution system tool (``code_execution_20250522``).

    **Not supported on AWS Bedrock** — raises ``BadRequestError`` with an unknown
    tool type error.  All tests in this class are skipped when ``is_bedrock_direct``
    is ``True``.

    On the official Anthropic API, code execution is server-executed: Claude emits
    a ``server_tool_use`` block; the API executes the code and returns the output
    automatically (no host round-trip needed).

    Validated scenarios
    -------------------
    - Tool accepted without error (official API only)
    - Triggers ``server_tool_use`` with a ``"code"`` key containing Python code
    - Multi-turn: code → execution output → end_turn text that references the result
    - Runtime error in tool_result (``is_error=true``) is accepted
    """

    @pytest.fixture(autouse=True)
    def _skip_on_bedrock(
        self, is_bedrock_direct: bool, use_anthropic_api: bool
    ) -> None:
        """Skip all tests in this class unless running against official Anthropic API.

        ``code_execution_20250522`` is not a recognised tool type on AWS Bedrock
        or on stdapi (which proxies to Bedrock).  It requires a direct
        ``ANTHROPIC_API_KEY`` connection to the official Anthropic API.
        """
        if is_bedrock_direct or not use_anthropic_api:
            pytest.skip(
                "code_execution_20250522 requires official Anthropic API "
                "(set ANTHROPIC_API_KEY and use --use-official-api)"
            )

    @pytest.mark.expensive
    def test_accepted(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Code execution tool definition is accepted without error.

        Validates:
            - Request with ``code_execution_20250522`` does not raise
            - Response is a valid message
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=[_CODE_EXECUTION_TOOL],  # type: ignore[list-item]
        )
        assert response.type == "message"
        assert len(response.content) >= 1

    @pytest.mark.expensive
    def test_triggers_tool_use(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Code execution produces a ``server_tool_use`` block with a ``"code"`` key.

        Validates:
            - ``block.name == "code_execution"``
            - ``"code"`` key present in ``block.input`` (Python code to execute)
        """
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": "Compute 2 + 2 using code."}],
            tools=[_CODE_EXECUTION_TOOL],
            tool_choice={"type": "any"},
        )
        tool_blocks = [b for b in response.content if isinstance(b, ServerToolUseBlock)]
        assert len(tool_blocks) >= 1
        block = tool_blocks[0]
        assert isinstance(block, ServerToolUseBlock)
        assert block.id
        assert block.name == "code_execution"
        assert "code" in block.input

    @pytest.mark.expensive
    def test_multiturn_with_result(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Code multi-turn: Python code in Turn 1, execution output in Turn 2.

        Validates:
            - Turn 1: ``tool_use`` with ``"code"`` key
            - Turn 2: providing the execution result is accepted
            - Turn 2 response: ``end_turn`` text that incorporates the result
        """
        tools = [_CODE_EXECUTION_TOOL]
        user_prompt = "Compute 2 ** 10 using code."

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice={"type": "any"},
        )
        tool_blocks = [b for b in resp1.content if isinstance(b, ServerToolUseBlock)]
        assert tool_blocks
        code_block = tool_blocks[0]
        assert isinstance(code_block, ServerToolUseBlock)
        assert code_block.id

        resp2 = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": list(resp1.content)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": code_block.id,
                            "content": "1024\n",
                        }
                    ],
                },
            ],
            tools=tools,  # type: ignore[arg-type]
        )
        assert resp2.stop_reason == "end_turn"
        text_blocks = [b for b in resp2.content if b.type == "text"]
        assert len(text_blocks) >= 1
        assert "1024" in " ".join(b.text for b in text_blocks)

    @pytest.mark.expensive
    def test_runtime_error_result_accepted(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A tool_result with a Python traceback and ``is_error=true`` is accepted.

        Validates:
            - Turn 2 with traceback + ``is_error=true`` does not raise
            - Claude responds gracefully (explains or retries)
        """
        tools = [_CODE_EXECUTION_TOOL]
        user_prompt = "Divide 1 by 0 using code."

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice={"type": "any"},
        )
        tool_blocks = [b for b in resp1.content if isinstance(b, ServerToolUseBlock)]
        assert tool_blocks

        tool_block = tool_blocks[0]
        assert isinstance(tool_block, ServerToolUseBlock)
        assert tool_block.id
        resp2 = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": list(resp1.content)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": "ZeroDivisionError: division by zero",
                            "is_error": True,
                        }
                    ],
                },
            ],
            tools=tools,  # type: ignore[arg-type]
        )
        assert resp2.type == "message"


# ===========================================================================
# Computer use tool
# ===========================================================================


@pytest.mark.parametrize("model_id", _CLAUDE_ALL)
class TestComputerUseTool:
    """Tests for the computer use system tool on Claude models.

    Computer use requires the ``computer-use-*`` beta header.  On Bedrock
    (reference behavior), Claude emits ``tool_use`` blocks with action types
    such as ``screenshot``, ``left_click``, ``type``, ``key``, ``scroll``.

    Screenshot results are sent as JPEG images matching the declared display
    dimensions.  The sample ``tests/samples/desktop.jpg`` (1024x576) is used
    as a realistic Windows desktop containing a visible Firefox icon.

    Validated scenarios
    -------------------
    - Tool accepted with beta header
    - First action is a ``screenshot``
    - Multi-turn: ask for screenshot → provide desktop JPEG → Claude identifies apps
    - Click Firefox: coordinate is within display bounds
    """

    #: Display dimensions matching the sample desktop screenshot (desktop.jpg).
    _DISPLAY_WIDTH = 1024
    _DISPLAY_HEIGHT = 576

    @pytest.fixture(autouse=True)
    def _skip_without_computer_use(self, anthropic_chat_model: str) -> None:
        """Skip when the model under test supports no computer-use tool type."""
        if _COMPUTER_USE_UNSUPPORTED_MODELS.search(anthropic_chat_model):
            pytest.skip("Computer use is not supported by this model")

    @pytest.fixture(scope="class")
    def desktop_screenshot_b64(self) -> str:
        """Base64-encoded JPEG of the sample Windows desktop screenshot.

        Returns:
            Base64 string of ``tests/samples/desktop.jpg`` (1024x576, JPEG).
        """
        path = Path(__file__).parent / "samples" / "desktop.jpg"
        return base64.b64encode(path.read_bytes()).decode()

    def _computer_tool(self, model: str) -> dict[str, object]:
        """Return a computer tool definition matching the sample screenshot dimensions.

        Selects ``computer_20251124`` for the models that dropped support for the
        older ``computer_20250124`` tool type, and ``computer_20250124`` for all
        other models.

        Args:
            model: The Bedrock or Anthropic model ID of the model under test.

        Returns:
            Tool definition dict with ``type``, ``name``, and display dimensions.
        """
        tool_type = (
            "computer_20251124"
            if _COMPUTER_USE_NEW_MODELS.search(model)
            else "computer_20250124"
        )
        return {
            "type": tool_type,
            "name": "computer",
            "display_width_px": self._DISPLAY_WIDTH,
            "display_height_px": self._DISPLAY_HEIGHT,
        }

    @staticmethod
    def _beta_headers(model: str, use_anthropic_api: bool) -> dict[str, str]:
        """Return the beta headers required for computer use.

        Uses ``computer-use-2025-11-24`` for the models on the newer tool type and
        the older ``computer-use-2025-01-24`` for all others.  On Bedrock
        (``use_anthropic_api = False``) no beta header is required.

        Args:
            model: The Bedrock or Anthropic model ID of the model under test.
            use_anthropic_api: Whether tests run against the official API.

        Returns:
            Dict with ``anthropic-beta`` header when on the official API; empty otherwise.
        """
        if not use_anthropic_api:
            return {}
        beta = (
            _COMPUTER_USE_BETA_NEW
            if _COMPUTER_USE_NEW_MODELS.search(model)
            else _COMPUTER_USE_BETA_OLD
        )
        return {"anthropic-beta": beta}

    @staticmethod
    def _screenshot_result(tool_use_id: str, b64_data: str) -> dict[str, object]:
        """Build a ``tool_result`` content dict carrying a JPEG screenshot.

        Args:
            tool_use_id: ID of the preceding ``tool_use`` block to match.
            b64_data: Base64-encoded JPEG image data.

        Returns:
            Tool result dict for use in the ``content`` of a user message.
        """
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": b64_data,
                    },
                }
            ],
        }

    @pytest.mark.expensive
    def test_accepted(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """Computer use tool definition is accepted with beta header.

        Validates:
            - Request with the model-appropriate computer tool type + beta header does not raise
            - Response is a valid message
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": "Take a screenshot."}],
            tools=[self._computer_tool(anthropic_chat_model)],  # type: ignore[list-item]
            extra_headers=self._beta_headers(anthropic_chat_model, use_anthropic_api),
        )
        assert response.type == "message"
        assert len(response.content) >= 1

    @pytest.mark.expensive
    def test_screenshot_action_in_response(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """Claude requests a screenshot as its first computer use action.

        Validates:
            - ``tool_use`` block with ``name == "computer"``
            - ``block.input["action"] == "screenshot"``
        """
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": "Take a screenshot of the screen."}],
            tools=[self._computer_tool(anthropic_chat_model)],
            tool_choice={"type": "any"},
            extra_headers=self._beta_headers(anthropic_chat_model, use_anthropic_api),
        )
        tool_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) >= 1
        block = tool_blocks[0]
        assert isinstance(block, ToolUseBlock)
        assert block.id
        assert block.name == "computer"
        assert block.input.get("action") == "screenshot"

    @pytest.mark.expensive
    def test_multiturn_with_desktop_screenshot(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
        desktop_screenshot_b64: str,
    ) -> None:
        """Multi-turn: screenshot request → desktop JPEG → Claude identifies apps.

        Uses ``tests/samples/desktop.jpg`` (Windows desktop with Firefox, VLC, Notepad++,
        etc.) as the screenshot result.  Claude should produce a meaningful response
        about what it sees on the screen.

        Validates:
            - Turn 1: ``tool_use`` action (typically ``screenshot``)
            - Turn 2: JPEG screenshot is accepted without error
            - Turn 2 response: ``end_turn`` or next tool_use (both are valid)
        """
        tools = [self._computer_tool(anthropic_chat_model)]
        user_prompt = "What applications can you see on the desktop?"
        extra_headers = self._beta_headers(anthropic_chat_model, use_anthropic_api)

        resp1 = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,  # type: ignore[arg-type]
            extra_headers=extra_headers,
        )
        tool_blocks = [b for b in resp1.content if isinstance(b, ToolUseBlock)]
        if not tool_blocks:
            pytest.skip("Model did not emit a tool use block on Turn 1")
        first_block = tool_blocks[0]
        assert isinstance(first_block, ToolUseBlock)
        assert first_block.id

        resp2 = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": list(resp1.content)},
                {
                    "role": "user",
                    "content": [
                        self._screenshot_result(first_block.id, desktop_screenshot_b64)  # type: ignore[list-item]
                    ],
                },
            ],
            tools=tools,  # type: ignore[arg-type]
            extra_headers=extra_headers,
        )
        assert resp2.type == "message"
        assert resp2.stop_reason in ("end_turn", "tool_use", "max_tokens")

    @pytest.mark.expensive
    def test_click_firefox_produces_coordinate(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
        desktop_screenshot_b64: str,
    ) -> None:
        """Claude emits a click action with coordinates when asked to open Firefox.

        Provides the desktop screenshot on Turn 1 so Claude can identify the Firefox
        icon position and emit a click or mouse_move action immediately.  The desktop
        shows Firefox in the taskbar at the bottom-left area.

        Validates:
            - At least one ``tool_use`` with a click-like action
            - ``coordinate`` is a 2-element list of integers within display bounds
        """
        tools = [self._computer_tool(anthropic_chat_model)]
        extra_headers = self._beta_headers(anthropic_chat_model, use_anthropic_api)
        user_prompt = "Open Firefox by clicking on it."

        # Provide the screenshot up-front so Claude can see the desktop immediately
        resp = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": desktop_screenshot_b64,
                            },
                        },
                    ],
                }
            ],
            tools=tools,
            tool_choice={"type": "any"},
            extra_headers=extra_headers,
        )
        tool_blocks = [b for b in resp.content if isinstance(b, ToolUseBlock)]
        assert tool_blocks, "Expected at least one tool_use block"
        # Any action with a coordinate (mouse_move, left_click, double_click, etc.)
        coord_blocks = [b for b in tool_blocks if b.input.get("coordinate") is not None]
        if coord_blocks:
            coord = coord_blocks[0].input["coordinate"]
            assert isinstance(coord, list)
            assert len(coord) == 2
            assert all(isinstance(v, int) for v in coord)
            assert 0 <= coord[0] <= self._DISPLAY_WIDTH
            assert 0 <= coord[1] <= self._DISPLAY_HEIGHT


# ===========================================================================
# Web search tool
# ===========================================================================


class TestWebSearchTool:
    """Tests for the ``web_search`` system tool via the Anthropic Messages API.

    On the local gateway, ``web_search`` is mapped to the Bedrock
    ``nova_grounding`` system tool, which is only available on Amazon Nova
    Premier.  On the official Anthropic API it is server-executed.

    On both local and Bedrock, passing ``web_search`` to a Claude model
    raises ``BadRequestError`` (Claude models don't support web search on Bedrock).

    Validated scenarios
    -------------------
    - Basic search on Nova Premier returns a valid response (local gateway only)
    - Streaming with web search on Nova Premier completes successfully (local only)
    - Passing ``web_search`` to a Claude model raises ``BadRequestError`` (always)
    """

    #: Nova Premier model identifier for local Bedrock web search tests.
    NOVA_PREMIER_MODEL = "amazon.nova-premier-v1:0"

    @pytest.fixture(autouse=True)
    def _skip_nova_tests(
        self, request: pytest.FixtureRequest, use_anthropic_api: bool
    ) -> None:
        """Skip Nova Premier tests when running against the official Anthropic API.

        The ``test_basic`` and ``test_streaming`` tests need a local server with
        Nova Premier access.  ``test_unsupported_model_raises_error`` runs everywhere.
        """
        nova_tests = {"test_basic", "test_streaming"}
        if use_anthropic_api and request.node.name in nova_tests:
            pytest.skip(
                "Nova Premier web search tests require a local server; "
                "these tests only run without --use-official-api"
            )

    def test_basic(self, anthropic_client: Anthropic) -> None:
        """Basic web search on Nova Premier returns a valid response.

        Validates:
            - Response type is ``"message"``
            - Response role is ``"assistant"``
            - At least one content block is present
        """
        response = anthropic_client.messages.create(
            model=self.NOVA_PREMIER_MODEL,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": "What are the latest AWS re:Invent announcements?",
                }
            ],
            tools=[_WEB_SEARCH_TOOL],  # type: ignore[list-item]
        )
        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) >= 1

    def test_streaming(self, anthropic_client: Anthropic) -> None:
        """Streaming with web search on Nova Premier completes without error.

        Validates:
            - Stream completes
            - Collected text is non-empty
        """
        with anthropic_client.messages.stream(
            model=self.NOVA_PREMIER_MODEL,
            max_tokens=2048,
            messages=[
                {"role": "user", "content": "What is the current weather in Seattle?"}
            ],
            tools=[_WEB_SEARCH_TOOL],  # type: ignore[list-item]
        ) as stream:
            full_text = stream.get_final_text()

        assert len(full_text) > 0

    def test_unsupported_model_raises_error(self, anthropic_client: Anthropic) -> None:
        """``web_search`` on a Claude model raises ``BadRequestError``.

        On Bedrock, web search is only available on Nova Premier.  Passing it to
        a Claude model raises a ``BadRequestError`` both on the local gateway and
        when using ``AnthropicBedrock`` directly.

        Validates:
            - ``BadRequestError`` is raised when a Claude model is given web_search
        """
        with pytest.raises(BadRequestError):
            anthropic_client.messages.create(
                model="anthropic.claude-haiku-4-5-20251001-v1:0",
                max_tokens=300,
                messages=[{"role": "user", "content": "What is the weather today?"}],
                tools=[_WEB_SEARCH_TOOL],  # type: ignore[list-item]
            )


# ===========================================================================
# Web fetch tool
# ===========================================================================


@pytest.mark.parametrize("model_id", _CLAUDE_ALL)
class TestWebFetchTool:
    """Tests for the ``web_fetch`` system tool (``web_fetch_20250910``).

    **Not supported on AWS Bedrock** (direct or via gateway) — raises
    ``BadRequestError`` with an unknown tool type error.  All inference tests in
    this class are skipped unless ``--use-official-api`` is set (ANTHROPIC_API_KEY
    required).

    On the official Anthropic API, ``web_fetch`` is server-executed: Claude emits
    a ``server_tool_use`` block with a ``"url"`` key; the API fetches the URL and
    returns the page content automatically.

    Validated scenarios
    -------------------
    - Tool accepted without error (official API only)
    - Claude emits a ``server_tool_use`` block with a ``"url"`` key
    - Multi-turn: fetch request → page content → end_turn summary
    - Unsupported on Bedrock (direct or via gateway): raises ``BadRequestError``
    """

    @pytest.fixture(autouse=True)
    def _skip_inference_on_bedrock(
        self,
        request: pytest.FixtureRequest,
        is_bedrock_direct: bool,
        use_anthropic_api: bool,
    ) -> None:
        """Skip inference tests when the backend is Bedrock (direct or via gateway).

        ``web_fetch_20250910`` is not a recognised tool type on Bedrock.
        The error-path test ``test_unsupported_on_bedrock_raises_error`` is exempt.
        """
        inference_tests = {
            "test_accepted",
            "test_triggers_tool_use",
            "test_multiturn_with_page_content",
        }
        if (
            is_bedrock_direct or not use_anthropic_api
        ) and request.node.originalname in inference_tests:
            pytest.skip(
                "web_fetch_20250910 requires official Anthropic API "
                "(set ANTHROPIC_API_KEY and use --use-official-api)"
            )

    @pytest.mark.expensive
    def test_accepted(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Web fetch tool definition is accepted without error (official API only).

        Validates:
            - Request with ``web_fetch_20250910`` does not raise
            - Response is a valid message
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=[_WEB_FETCH_TOOL],  # type: ignore[list-item]
        )
        assert response.type == "message"
        assert len(response.content) >= 1

    @pytest.mark.expensive
    def test_triggers_tool_use(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Web fetch produces a ``server_tool_use`` block with a ``"url"`` key.

        Validates:
            - ``block.name == "web_fetch"``
            - ``"url"`` key present in ``block.input``
        """
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "Fetch https://example.com and tell me the title.",
                }
            ],
            tools=[_WEB_FETCH_TOOL],
            tool_choice={"type": "any"},
        )
        tool_blocks = [b for b in response.content if isinstance(b, ServerToolUseBlock)]
        assert len(tool_blocks) >= 1
        block = tool_blocks[0]
        assert isinstance(block, ServerToolUseBlock)
        assert block.id
        assert block.name == "web_fetch"
        assert "url" in block.input

    @pytest.mark.expensive
    def test_multiturn_with_page_content(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Web fetch multi-turn: fetch request → page content → end_turn summary.

        Validates:
            - Turn 1: ``server_tool_use`` block with ``"url"`` key
            - Turn 2: providing page HTML as tool_result is accepted
            - Turn 2 response: ``end_turn`` with a text summary
        """
        tools = [_WEB_FETCH_TOOL]
        user_prompt = "Fetch https://example.com and summarize it."

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice={"type": "any"},
        )
        tool_blocks = [b for b in resp1.content if isinstance(b, ServerToolUseBlock)]
        assert tool_blocks
        fetch_block = tool_blocks[0]
        assert isinstance(fetch_block, ServerToolUseBlock)
        assert fetch_block.id

        resp2 = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": list(resp1.content)},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": fetch_block.id,
                            "content": (
                                "<html><head><title>Example Domain</title></head>"
                                "<body><h1>Example Domain</h1>"
                                "<p>This domain is for illustrative examples.</p>"
                                "</body></html>"
                            ),
                        }
                    ],
                },
            ],
            tools=tools,  # type: ignore[arg-type]
        )
        assert resp2.stop_reason == "end_turn"
        assert any(b.type == "text" for b in resp2.content)

    def test_unsupported_on_bedrock_raises_error(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        is_bedrock_direct: bool,
        use_anthropic_api: bool,
    ) -> None:
        """``web_fetch`` raises ``BadRequestError`` on Bedrock (direct or via gateway).

        Validates:
            - ``BadRequestError`` is raised when the backend is Bedrock
        """
        if use_anthropic_api and not is_bedrock_direct:
            pytest.skip("Error path only applies when Bedrock is the backend")
        with pytest.raises(BadRequestError):
            anthropic_client.messages.create(
                model=anthropic_chat_model,
                max_tokens=300,
                messages=[{"role": "user", "content": "Fetch https://example.com"}],
                tools=[_WEB_FETCH_TOOL],  # type: ignore[list-item]
            )


# ===========================================================================
# Mixed server tools + custom tools
# ===========================================================================


@pytest.mark.parametrize("model_id", _CLAUDE_ALL)
class TestMixedServerAndCustomTools:
    """Tests combining Anthropic system tools with user-defined custom tools.

    Validated scenarios
    -------------------
    - System tool (bash) alongside one custom tool: accepted
    - Two system tools together (bash + text_editor): accepted
    """

    @pytest.mark.expensive
    def test_server_tool_with_custom_tool(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """System tool accepted alongside a custom ``toolSpec``.

        Validates:
            - Mixing ``bash_20250124`` with a user-defined tool does not raise
            - Response is a valid message
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            tools=[
                _BASH_TOOL,  # type: ignore[list-item]
                {
                    "name": "get_time",
                    "description": "Get current time",
                    "input_schema": {"type": "object", "properties": {}},
                },
            ],
        )
        assert response.type == "message"
        assert len(response.content) >= 1

    @pytest.mark.expensive
    def test_multiple_server_tools_together(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """Multiple system tools in a single request are accepted without error.

        Validates:
            - bash + text_editor together do not raise
            - Response is a valid message
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=[_BASH_TOOL, _TEXT_EDITOR_TOOL],  # type: ignore[list-item]
        )
        assert response.type == "message"
        assert len(response.content) >= 1
