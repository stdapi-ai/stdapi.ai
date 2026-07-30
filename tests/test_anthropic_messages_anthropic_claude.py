"""Anthropic-defined tool types on Claude models through the Anthropic Messages route.

Claude receives these tools in Anthropic native form: the gateway turns each one into
a bare Bedrock ``toolSpec`` stub, then moves it into
``additionalModelRequestFields["tools"]`` on the first turn and keeps the stub inside
``toolConfig`` once the history carries a ``toolResult``.  Bedrock therefore answers
with ordinary ``tool_use`` blocks — never ``server_tool_use`` — and the host
application is responsible for executing the tool and returning a ``tool_result``.

Each tool ``type`` version dictates the ``name`` the request must use and the beta
flag the gateway injects on its behalf:

- ``text_editor_20250728`` → ``str_replace_based_edit_tool``
  (``text_editor_20250124`` and earlier use ``str_replace_editor``)
- ``bash_20250124`` → ``bash``
- ``memory_20250818`` → ``memory`` (``context-management-2025-06-27``)
- ``computer_20250124`` → ``computer`` (``computer-use-2025-01-24``);
  ``computer_20251124`` → ``computer`` (``computer-use-2025-11-24``)
- ``web_search_20250305`` → ``web_search``: reachable only on Amazon Nova Premier,
  where the gateway maps it to the Bedrock ``nova_grounding`` system tool
- ``code_execution_20250522`` and ``web_fetch_20250910``: absent from Bedrock

Every test needs the local gateway, so the autouse ``_skip_on_official_api`` fixture
skips the module under ``--use-official-api``.  For the two tool types Bedrock does
not implement (``code_execution`` and ``web_fetch``) only the rejection path is
reachable here, so that is all these classes assert.

Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
     https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy
     stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
     stdapi/models/chat/_adapters/_anthropic_message.py:_handle_system_tool
"""

import base64
import re
from pathlib import Path

import pytest
from anthropic import Anthropic, BadRequestError
from anthropic.types import Message, ToolUseBlock

#: Claude models covering every system-tool code branch (old and new computer-use
#: tool types, and the unsupported-model skip), for tools billed on every call.
_CLAUDE_SYSTEM_TOOLS = (
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-sonnet-4-6",
    "anthropic.claude-sonnet-5",
    "anthropic.claude-opus-5",
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

#: Code execution tool — legacy Python-only type ``code_execution_20250522``, absent on Bedrock.
_CODE_EXECUTION_TOOL: dict[str, object] = {
    "type": "code_execution_20250522",
    "name": "code_execution",
}

#: Web search tool — type ``web_search_20250305``, canonical name ``web_search``.
_WEB_SEARCH_TOOL: dict[str, object] = {
    "type": "web_search_20250305",
    "name": "web_search",
}

#: Web fetch tool — type ``web_fetch_20250910``, canonical name ``web_fetch``, absent on Bedrock.
_WEB_FETCH_TOOL: dict[str, object] = {"type": "web_fetch_20250910", "name": "web_fetch"}

#: Anthropic ``stop_reason`` values a completed tool-enabled Claude turn can carry.
_TURN_STOP_REASONS = frozenset({"end_turn", "max_tokens", "stop_sequence", "tool_use"})

#: Commands of the ``text_editor_20250728`` tool (``undo_edit`` exists only up to ``20250124``).
_TEXT_EDITOR_COMMANDS = frozenset({"view", "create", "str_replace", "insert"})


def _tool_result(
    tool_use_id: str, content: str, *, is_error: bool = False
) -> dict[str, object]:
    """Build the ``tool_result`` block answering the tool use *tool_use_id*."""
    block: dict[str, object] = {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": content,
    }
    if is_error:
        block["is_error"] = True
    return block


def _reply_with_tool_result(
    anthropic_client: Anthropic,
    model: str,
    user_prompt: str,
    first_turn: Message,
    tools: list[dict[str, object]],
    result: dict[str, object],
    *,
    extra_headers: dict[str, str] | None = None,
) -> Message:
    """Answer a forced ``tool_use`` turn with *result* and return the model's reply.

    Turn 2 replays the original user prompt and the assistant's tool-use blocks
    verbatim: Bedrock rejects a conversation whose earlier turns changed, and the
    declared tools must still be present for the ``tool_result`` to be accepted.

    Args:
        anthropic_client: SDK client bound to the target under test.
        model: Model ID for both turns.
        user_prompt: The turn-1 user prompt, replayed unchanged.
        first_turn: The turn-1 assistant message.
        tools: Tool definitions, re-sent unchanged.
        result: ``tool_result`` block built by :func:`_tool_result`.
        extra_headers: Beta headers the tool type requires, if any.

    Returns:
        The assistant message of the second turn.
    """
    return anthropic_client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": list(first_turn.content)},
            {"role": "user", "content": [result]},  # type: ignore[list-item]
        ],
        tools=tools,  # type: ignore[arg-type]
        extra_headers=extra_headers,
    )


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
    ``@pytest.mark.parametrize("model_id", _CLAUDE_SYSTEM_TOOLS)``), converts the
    raw Bedrock-format ID to the format expected by the active backend:

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


@pytest.mark.parametrize("model_id", _CLAUDE_SYSTEM_TOOLS)
class TestTextEditorTool:
    """Text editor tool type ``text_editor_20250728`` on Claude models.

    The type version fixes the tool ``name`` to ``str_replace_based_edit_tool``, the
    command set to ``view`` / ``create`` / ``str_replace`` / ``insert`` and is the
    first version accepting ``max_characters``.  The gateway forwards the definition
    to Claude natively and injects ``computer-use-2025-01-24`` for it, so the model
    answers with a host-executed ``tool_use`` block.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
         stdapi/models/chat/anthropic_claude.py:ChatModel
    """

    # --- acceptance ---

    @pytest.mark.expensive
    def test_accepted(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A ``text_editor_20250728`` definition is accepted and inference still completes.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
             stdapi/models/chat/_adapters/_anthropic_message.py:_handle_system_tool
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=[_TEXT_EDITOR_TOOL],  # type: ignore[list-item]
        )
        assert response.type == "message"
        assert response.role == "assistant"
        assert response.id.startswith("msg_")
        assert len(response.content) >= 1
        assert response.stop_reason in _TURN_STOP_REASONS
        assert response.usage.output_tokens > 0

    # --- view command ---

    @pytest.mark.expensive
    def test_view_file_triggers_tool_use(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """The ``view`` command is emitted as a ``tool_use`` block naming the requested file.

        ``tool_choice=any`` forces a tool call, so the model cannot answer in prose;
        the block must carry the ``text_editor_20250728`` name and the ``command`` /
        ``path`` arguments of that version's schema.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/define-tools#forcing-tool-use
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_tool_choice
        """
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
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
        assert "/etc/hostname" in str(block.input["path"])
        assert response.stop_reason == "tool_use"

    @pytest.mark.expensive
    def test_view_directory_triggers_tool_use(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A directory listing request maps onto the ``view`` command with the directory path.

        The ``20250728`` schema has no dedicated list command: directories are read
        through ``view``.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
        """
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": "List the files in /tmp"}],
            tools=[_TEXT_EDITOR_TOOL],
            tool_choice={"type": "any"},
        )
        tool_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) >= 1
        block = tool_blocks[0]
        assert isinstance(block, ToolUseBlock)
        assert block.id
        assert block.name == "str_replace_based_edit_tool"
        assert block.input.get("command") == "view"
        assert "tmp" in str(block.input.get("path", "")), (
            f"view should target the requested directory, got {block.input}"
        )

    @pytest.mark.expensive
    def test_view_file_with_range_emits_view_command(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A line-range request stays a ``view`` command, optionally carrying ``view_range``.

        ``view_range`` is an optional argument of the ``view`` command, so the test
        validates its ``[start, end]`` shape only when the model includes it.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
        """
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": "View lines 1 to 5 of /etc/hosts"}],
            tools=[_TEXT_EDITOR_TOOL],
            tool_choice={"type": "any"},
        )
        tool_blocks = [b for b in response.content if isinstance(b, ToolUseBlock)]
        assert len(tool_blocks) >= 1
        block = tool_blocks[0]
        assert isinstance(block, ToolUseBlock)
        assert block.id
        assert block.name == "str_replace_based_edit_tool"
        assert block.input.get("command") == "view"
        assert "/etc/hosts" in str(block.input.get("path", ""))
        if (view_range := block.input.get("view_range")) is not None:
            assert isinstance(view_range, list)
            assert len(view_range) == 2
            assert all(isinstance(bound, int) for bound in view_range)

    @pytest.mark.expensive
    def test_view_multiturn(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A text-editor ``tool_result`` closes the turn with an ``end_turn`` text answer.

        Turn 2 exercises the gateway's multi-turn stub mode: once the history holds a
        ``toolResult``, the tool must stay a ``toolSpec`` entry in ``toolConfig``
        instead of moving to ``additionalModelRequestFields``, otherwise Bedrock
        rejects the request.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
        """
        tools = [_TEXT_EDITOR_TOOL]
        user_prompt = "View the file /etc/hostname"

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice={"type": "any"},
        )
        tool_blocks = [b for b in resp1.content if isinstance(b, ToolUseBlock)]
        assert tool_blocks, "Expected tool_use block in Turn 1"
        tool_block = tool_blocks[0]
        assert isinstance(tool_block, ToolUseBlock)
        assert tool_block.id
        assert tool_block.name == "str_replace_based_edit_tool"
        assert tool_block.input.get("command") == "view"

        resp2 = _reply_with_tool_result(
            anthropic_client,
            anthropic_chat_model,
            user_prompt,
            resp1,
            tools,
            _tool_result(tool_block.id, "test-host\n"),
        )
        assert resp2.stop_reason == "end_turn"
        assert any(b.type == "text" for b in resp2.content)
        assert resp2.usage.output_tokens > 0

    # --- str_replace command ---

    @pytest.mark.expensive
    def test_str_replace_command_shape(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """An edit command round-trips with a replaced-text pair that differs from the original.

        Each ``view`` is answered with the file contents until an edit command appears
        or 5 forced-tool turns are exhausted.  ``text_editor_20250728`` documents the
        replaced-text arguments as ``old_str`` / ``new_str``, but the emitted key name
        varies across Claude generations, so the assertion accepts the observed
        ``old_str`` / ``old_text`` / ``old_string`` spellings and pairs each with its
        ``new_*`` counterpart.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
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
                max_tokens=1024,
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
                assert block.input.get("command") in _TEXT_EDITOR_COMMANDS
                assert block.input.get("path"), (
                    f"every text editor command requires a path, got {block.input}"
                )
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
        """The ``create`` command carries the target ``path`` and the new ``file_text``.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
        """
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
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
        assert block.name == "str_replace_based_edit_tool"
        assert block.input.get("command") == "create"
        assert block.input.get("path")
        assert "hello" in str(block.input["path"]).lower()
        assert block.input.get("file_text")
        assert "hello world" in str(block.input["file_text"]).lower()

    # --- insert command ---

    @pytest.mark.expensive
    def test_insert_command_shape(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """The ``insert`` command carries an integer ``insert_line`` and the text to insert.

        The model usually reads the file before editing, so the flow answers a first
        ``view`` with file contents and then accepts any of the version's write
        commands; the ``insert``-specific argument shape is asserted whenever
        ``insert`` is the command actually chosen.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
        """
        tools = [_TEXT_EDITOR_TOOL]
        user_prompt = "Add a module docstring at the top of /tmp/primes.py"

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
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
            assert block.name == "str_replace_based_edit_tool"
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
            max_tokens=1024,
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
        assert block.name == "str_replace_based_edit_tool"
        command = block.input.get("command")
        assert command in ("insert", "str_replace", "create")
        assert block.input.get("path")
        if command == "insert":
            assert isinstance(block.input.get("insert_line"), int)
            assert block.input.get("insert_text")

    # --- error result ---

    @pytest.mark.expensive
    def test_error_tool_result_accepted(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A text-editor ``tool_result`` flagged ``is_error=true`` is accepted and answered.

        Anthropic signals a host-side tool failure with ``is_error`` on the
        ``tool_result`` block; the gateway must map it to the Bedrock ``toolResult``
        status instead of rejecting the message.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        """
        tools = [_TEXT_EDITOR_TOOL]
        user_prompt = "View the file /nonexistent/path.py"

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice={"type": "any"},
        )
        tool_blocks = [b for b in resp1.content if isinstance(b, ToolUseBlock)]
        assert tool_blocks

        tool_block = tool_blocks[0]
        assert isinstance(tool_block, ToolUseBlock)
        assert tool_block.id
        assert tool_block.name == "str_replace_based_edit_tool"
        resp2 = _reply_with_tool_result(
            anthropic_client,
            anthropic_chat_model,
            user_prompt,
            resp1,
            tools,
            _tool_result(tool_block.id, "Error: File not found", is_error=True),
        )
        assert resp2.type == "message"
        assert resp2.role == "assistant"
        assert resp2.stop_reason in _TURN_STOP_REASONS
        assert resp2.usage.output_tokens > 0

    # --- max_characters ---

    @pytest.mark.expensive
    def test_max_characters_accepted(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """``max_characters`` is forwarded as an extra tool argument without being rejected.

        ``max_characters`` exists only from ``text_editor_20250728`` onwards; the
        gateway keeps unknown-to-Bedrock tool arguments by serialising the whole tool
        into ``additionalModelRequestFields["tools"]``.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=[{**_TEXT_EDITOR_TOOL, "max_characters": 1000}],  # type: ignore[list-item]
        )
        assert response.type == "message"
        assert response.role == "assistant"
        assert response.id.startswith("msg_")
        assert len(response.content) >= 1
        assert response.stop_reason in _TURN_STOP_REASONS
        assert response.usage.output_tokens > 0

    @pytest.mark.expensive
    def test_max_characters_triggers_tool_use(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """``max_characters`` leaves the emitted ``tool_use`` block shape unchanged.

        The extra argument constrains what the host returns, not what the model
        requests, so the block still carries the version's name and one of its four
        commands.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
        """
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
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
        assert block.input["command"] in _TEXT_EDITOR_COMMANDS

    @pytest.mark.expensive
    def test_max_characters_multiturn(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A ``max_characters`` text editor survives the multi-turn ``tool_result`` round trip.

        Turn 2 keeps the same augmented tool definition, so the gateway has to accept
        the extra argument in stub mode as well as in native mode.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/text-editor-tool
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
        """
        tools = [{**_TEXT_EDITOR_TOOL, "max_characters": 1000}]
        user_prompt = "View the file /etc/hostname"

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice={"type": "any"},
        )
        tool_blocks = [b for b in resp1.content if isinstance(b, ToolUseBlock)]
        assert tool_blocks, "Expected tool_use block in Turn 1"
        tool_block = tool_blocks[0]
        assert isinstance(tool_block, ToolUseBlock)
        assert tool_block.id
        assert tool_block.name == "str_replace_based_edit_tool"

        resp2 = _reply_with_tool_result(
            anthropic_client,
            anthropic_chat_model,
            user_prompt,
            resp1,
            tools,
            _tool_result(tool_block.id, "test-host\n"),
        )
        assert resp2.stop_reason == "end_turn"
        assert any(b.type == "text" for b in resp2.content)
        assert resp2.usage.output_tokens > 0


# ===========================================================================
# Bash tool
# ===========================================================================


@pytest.mark.parametrize("model_id", _CLAUDE_SYSTEM_TOOLS)
class TestBashTool:
    """Bash tool type ``bash_20250124`` (name ``bash``) on Claude models.

    ``bash_20250124`` is the single GA version and needs no beta header upstream;
    the gateway nevertheless injects ``computer-use-2025-01-24`` for it because
    Bedrock keys the flag on the tool name.  Its inputs are ``command`` (required
    unless ``restart``) and ``restart``, and the host runs the command and returns
    stdout/stderr as a ``tool_result``.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool
         stdapi/models/chat/anthropic_claude.py:ChatModel
    """

    @pytest.mark.expensive
    def test_accepted(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """A ``bash_20250124`` definition is accepted and inference still completes.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool
             stdapi/models/chat/_adapters/_anthropic_message.py:_handle_system_tool
        """
        extra_headers = {"anthropic-beta": _BASH_BETA} if use_anthropic_api else {}
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=[_BASH_TOOL],  # type: ignore[list-item]
            extra_headers=extra_headers,
        )
        assert response.type == "message"
        assert response.role == "assistant"
        assert response.id.startswith("msg_")
        assert len(response.content) >= 1
        assert response.stop_reason in _TURN_STOP_REASONS
        assert response.usage.output_tokens > 0

    @pytest.mark.expensive
    def test_triggers_tool_use(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """The bash tool emits one ``tool_use`` block whose input carries the requested command.

        ``tool_choice=any`` forces the call.  The prompt pins the command text, so the
        assertion looks for it across the input values rather than under a fixed key:
        the argument name of the shell command varies across Claude generations.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool
             stdapi/models/chat/_adapters/_anthropic_message.py:_map_tool_choice
        """
        extra_headers = {"anthropic-beta": _BASH_BETA} if use_anthropic_api else {}
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
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
        assert "hello_test" in " ".join(str(value) for value in block.input.values()), (
            f"bash input should carry the requested command, got {block.input}"
        )
        assert response.stop_reason == "tool_use"

    @pytest.mark.expensive
    def test_multiturn(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """A bash ``tool_result`` carrying stdout closes the turn with ``end_turn`` text.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
        """
        extra_headers = {"anthropic-beta": _BASH_BETA} if use_anthropic_api else {}
        tools = [_BASH_TOOL]
        user_prompt = "Run: echo hello_test"

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
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
        assert bash_block.name == "bash"
        assert bash_block.name == "bash"

        resp2 = _reply_with_tool_result(
            anthropic_client,
            anthropic_chat_model,
            user_prompt,
            resp1,
            tools,
            _tool_result(bash_block.id, "hello_test\n"),
            extra_headers=extra_headers,
        )
        assert resp2.stop_reason == "end_turn"
        assert any(b.type == "text" for b in resp2.content)
        assert resp2.usage.output_tokens > 0

    @pytest.mark.expensive
    def test_command_error_output_accepted(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """A bash ``tool_result`` with ``is_error=true`` and stderr text is accepted and answered.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        """
        extra_headers = {"anthropic-beta": _BASH_BETA} if use_anthropic_api else {}
        tools = [_BASH_TOOL]
        user_prompt = "Run: cat /nonexistent_file.txt"

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
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
        assert tool_block.name == "bash"
        resp2 = _reply_with_tool_result(
            anthropic_client,
            anthropic_chat_model,
            user_prompt,
            resp1,
            tools,
            _tool_result(
                tool_block.id,
                "cat: /nonexistent_file.txt: No such file or directory",
                is_error=True,
            ),
            extra_headers=extra_headers,
        )
        assert resp2.type == "message"
        assert resp2.role == "assistant"
        assert resp2.stop_reason in _TURN_STOP_REASONS
        assert resp2.usage.output_tokens > 0

    @pytest.mark.expensive
    def test_restart_tool_result_accepted(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """A ``tool_result`` acknowledging a bash session restart is accepted and answered.

        ``restart`` is a documented bash input, and its acknowledgement comes back as
        an ordinary successful ``tool_result`` payload rather than an error.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/bash-tool
        """
        extra_headers = {"anthropic-beta": _BASH_BETA} if use_anthropic_api else {}
        tools = [_BASH_TOOL]
        user_prompt = "Run: echo hello"

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
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
        assert tool_block.name == "bash"
        resp2 = _reply_with_tool_result(
            anthropic_client,
            anthropic_chat_model,
            user_prompt,
            resp1,
            tools,
            _tool_result(tool_block.id, "Bash session restarted"),
            extra_headers=extra_headers,
        )
        assert resp2.type == "message"
        assert resp2.role == "assistant"
        assert resp2.stop_reason in _TURN_STOP_REASONS
        assert resp2.usage.output_tokens > 0


# ===========================================================================
# Memory tool
# ===========================================================================


@pytest.mark.parametrize("model_id", _CLAUDE_SYSTEM_TOOLS)
class TestMemoryTool:
    """Memory tool type ``memory_20250818`` (name ``memory``) on Claude models.

    The tool exposes file operations (``view``, ``create``, ``str_replace``,
    ``insert``, ``delete``, ``rename``) scoped to the ``/memories`` directory, and its
    hidden system prompt makes the model inspect that directory first.  The gateway
    injects the required ``context-management-2025-06-27`` flag from
    ``TOOL_BETA_FLAGS`` so callers need not send the beta header.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
         stdapi/models/chat/anthropic_claude.py:ChatModel
    """

    #: File operations the ``memory_20250818`` tool exposes over ``/memories``.
    _MEMORY_COMMANDS = frozenset(
        {"view", "create", "str_replace", "insert", "delete", "rename"}
    )

    @pytest.mark.expensive
    def test_accepted(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """A ``memory_20250818`` definition is accepted without the caller sending the beta flag.

        The gateway adds ``context-management-2025-06-27`` to ``anthropic_beta``
        itself, so a request that omits the header must still be accepted by Bedrock.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
        """
        extra_headers = {"anthropic-beta": _MEMORY_BETA} if use_anthropic_api else {}
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=[_MEMORY_TOOL],  # type: ignore[list-item]
            extra_headers=extra_headers,
        )
        assert response.type == "message"
        assert response.role == "assistant"
        assert response.id.startswith("msg_")
        assert len(response.content) >= 1
        assert response.stop_reason in _TURN_STOP_REASONS
        assert response.usage.output_tokens > 0

    @pytest.mark.expensive
    def test_auto_views_directory_first(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """The first memory action is a ``view`` of ``/memories``.

        The tool's built-in system prompt makes directory inspection the opening
        action, which also proves the gateway forwarded the tool in native form
        (a bare Bedrock ``toolSpec`` stub carries no such instruction).

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
        """
        extra_headers = {"anthropic-beta": _MEMORY_BETA} if use_anthropic_api else {}
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
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
        """A memory write request produces a ``memory`` ``tool_use`` block with a file command.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
        """
        extra_headers = {"anthropic-beta": _MEMORY_BETA} if use_anthropic_api else {}
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
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
        assert block.input.get("command") in self._MEMORY_COMMANDS, (
            f"unexpected memory command in {block.input}"
        )
        assert response.stop_reason == "tool_use"

    @pytest.mark.expensive
    def test_multiturn(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """A ``/memories`` directory listing returned as ``tool_result`` is accepted and answered.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
        """
        extra_headers = {"anthropic-beta": _MEMORY_BETA} if use_anthropic_api else {}
        tools = [_MEMORY_TOOL]
        user_prompt = "Remember: project name is 'stdapi'"

        resp1 = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
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
        assert tool_block.name == "memory"

        resp2 = _reply_with_tool_result(
            anthropic_client,
            anthropic_chat_model,
            user_prompt,
            resp1,
            tools,
            _tool_result(
                tool_block.id,
                "Here're the files and directories up to 2 levels deep "
                "in /memories, excluding hidden items and node_modules:\n"
                "4.0K\t/memories\n",
            ),
            extra_headers=extra_headers,
        )
        assert resp2.type == "message"
        assert resp2.role == "assistant"
        assert resp2.stop_reason in _TURN_STOP_REASONS
        assert resp2.usage.output_tokens > 0


# ===========================================================================
# Code execution tool
# ===========================================================================


class TestCodeExecutionTool:
    """Code execution tool type ``code_execution_20250522`` — Anthropic-only, never on Bedrock.

    Upstream this is a server tool: the API runs the code itself and answers with
    ``server_tool_use`` plus ``code_execution_tool_result`` blocks.  Bedrock exposes
    no equivalent for Claude, so the only behavior reachable through this gateway is
    the rejection.  (Amazon Nova serves the same tool type through its own
    ``nova_code_interpreter`` system tool; see
    ``tests/test_anthropic_messages_amazon_nova_2.py``.)

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
         https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy
    """

    def test_unsupported_on_bedrock_raises_error(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        is_bedrock_direct: bool,
        use_anthropic_api: bool,
    ) -> None:
        """``code_execution`` on a Bedrock-backed Claude model is refused as a 400.

        No Bedrock system tool matches ``code_execution_20250522`` for Claude, so the
        native definition reaches Bedrock and is rejected there; the Anthropic
        envelope reports the resulting 400 as ``invalid_request_error``.

        Ref: https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy
             https://platform.claude.com/docs/en/api/errors
             stdapi/api_providers/anthropic.py:_format_error
        """
        if use_anthropic_api and not is_bedrock_direct:
            pytest.skip("Error path only applies when Bedrock is the backend")
        with pytest.raises(BadRequestError) as excinfo:
            anthropic_client.messages.create(
                model=anthropic_chat_model,
                max_tokens=300,
                messages=[{"role": "user", "content": "Compute 2 ** 10 using code."}],
                tools=[_CODE_EXECUTION_TOOL],  # type: ignore[list-item]
            )
        error = excinfo.value
        assert error.status_code == 400
        assert error.type == "invalid_request_error"
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["message"]
        assert "request_id" in body


# ===========================================================================
# Computer use tool
# ===========================================================================


@pytest.mark.parametrize("model_id", _CLAUDE_SYSTEM_TOOLS)
class TestComputerUseTool:
    """Computer use tool (name ``computer``) on the Claude models that still accept it.

    Computer use stays in beta and its beta flag is keyed on the tool ``type``, not on
    the model: ``computer_20251124`` requires ``computer-use-2025-11-24`` while
    ``computer_20250124`` requires ``computer-use-2025-01-24``.  Both carry
    ``display_width_px`` / ``display_height_px``, and the model answers with
    ``tool_use`` blocks whose ``action`` drives the host desktop.  Screenshots are
    returned as a JPEG image ``tool_result``; ``tests/samples/desktop.jpg``
    (1024x576) supplies a Windows desktop with a visible Firefox icon.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
         stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
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
            Base64 string of ``tests/samples/desktop.jpg``, sized to the declared display.
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
        """The model-appropriate ``computer`` tool type is accepted with its display params.

        Against the gateway no beta header is sent: the version-keyed flag is added
        server-side from the tool ``type``.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Take a screenshot."}],
            tools=[self._computer_tool(anthropic_chat_model)],  # type: ignore[list-item]
            extra_headers=self._beta_headers(anthropic_chat_model, use_anthropic_api),
        )
        assert response.type == "message"
        assert response.role == "assistant"
        assert response.id.startswith("msg_")
        assert len(response.content) >= 1
        assert response.stop_reason in _TURN_STOP_REASONS
        assert response.usage.output_tokens > 0

    @pytest.mark.expensive
    def test_screenshot_action_in_response(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
    ) -> None:
        """The opening computer action is ``screenshot``.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
        """
        response = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
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
        assert response.stop_reason == "tool_use"

    @pytest.mark.expensive
    def test_multiturn_with_desktop_screenshot(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
        desktop_screenshot_b64: str,
    ) -> None:
        """A JPEG screenshot returned as a ``tool_result`` image is accepted on the next turn.

        The image travels as a base64 ``image/jpeg`` block inside the ``tool_result``
        content list, which the gateway has to translate into a Bedrock ``toolResult``
        image block; the follow-up turn may either answer or request another action.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
        """
        tools = [self._computer_tool(anthropic_chat_model)]
        user_prompt = "What applications can you see on the desktop?"
        extra_headers = self._beta_headers(anthropic_chat_model, use_anthropic_api)

        resp1 = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=1024,
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
        assert first_block.name == "computer"

        resp2 = _reply_with_tool_result(
            anthropic_client,
            anthropic_chat_model,
            user_prompt,
            resp1,
            tools,
            self._screenshot_result(first_block.id, desktop_screenshot_b64),
            extra_headers=extra_headers,
        )
        assert resp2.type == "message"
        assert resp2.role == "assistant"
        assert resp2.stop_reason in ("end_turn", "tool_use", "max_tokens")
        assert resp2.usage.output_tokens > 0

    @pytest.mark.expensive
    def test_click_firefox_produces_coordinate(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_anthropic_api: bool,
        desktop_screenshot_b64: str,
    ) -> None:
        """A coordinate-bearing computer action stays inside the declared display bounds.

        The desktop screenshot is supplied up front so the model can locate the Firefox
        icon without a screenshot round trip.  Which action it picks is not
        deterministic — it may still open with ``screenshot`` — so the bounds check
        applies to whichever emitted action carries a ``coordinate``.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool
        """
        tools = [self._computer_tool(anthropic_chat_model)]
        extra_headers = self._beta_headers(anthropic_chat_model, use_anthropic_api)
        user_prompt = "Open Firefox by clicking on it."

        # Provide the screenshot up-front so Claude can see the desktop immediately
        resp = anthropic_client.messages.create(  # type: ignore[call-overload]
            model=anthropic_chat_model,
            max_tokens=1024,
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
        assert all(b.name == "computer" for b in tool_blocks)
        actions = [b.input.get("action") for b in tool_blocks]
        assert all(isinstance(action, str) and action for action in actions), (
            f"every computer tool_use must name an action, got {actions}"
        )
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
    """Web search tool type ``web_search_20250305`` on Nova Premier and on Claude.

    Web search does not exist on Amazon Bedrock as an Anthropic tool, so the gateway
    can only honor it where a Bedrock system tool matches: on Amazon Nova Premier it
    translates ``web_search`` into ``nova_grounding``, while on Claude the definition
    is forwarded natively and rejected.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
         https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
         stdapi/models/chat/amazon_nova_premier.py:ChatModel
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

    @pytest.mark.expensive
    def test_basic(self, anthropic_client: Anthropic) -> None:
        """Web search on Nova Premier resolves through the Bedrock ``nova_grounding`` tool.

        Nova Premier is the only catalogue entry whose
        ``CANONICAL_TO_BEDROCK_TOOL_MAP`` translates ``web_search``, and Bedrock runs
        the grounding search itself.

        Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
             stdapi/models/chat/_adapters/_anthropic_message.py:_handle_system_tool
        """
        response = anthropic_client.messages.create(
            model=self.NOVA_PREMIER_MODEL,
            max_tokens=256,
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
        assert response.stop_reason in _TURN_STOP_REASONS
        assert response.usage.output_tokens > 0

    @pytest.mark.expensive
    def test_streaming(self, anthropic_client: Anthropic) -> None:
        """A web-search turn on Nova Premier streams to a complete final message.

        Bedrock performs the grounding search server-side, so the SSE stream still
        terminates in an assistant message with text, a mapped ``stop_reason`` and
        token usage.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
        """
        with anthropic_client.messages.stream(
            model=self.NOVA_PREMIER_MODEL,
            max_tokens=256,
            messages=[
                {"role": "user", "content": "What is the current weather in Seattle?"}
            ],
            tools=[_WEB_SEARCH_TOOL],  # type: ignore[list-item]
        ) as stream:
            final_message = stream.get_final_message()
            full_text = stream.get_final_text()

        assert len(full_text) > 0
        assert final_message.type == "message"
        assert final_message.role == "assistant"
        assert final_message.stop_reason in _TURN_STOP_REASONS
        assert final_message.usage.output_tokens > 0

    def test_unsupported_model_raises_error(self, anthropic_client: Anthropic) -> None:
        """``web_search`` on a Claude model is refused as a 400 ``invalid_request_error``.

        Claude has no Bedrock system tool for web search, so the gateway forwards
        ``web_search_20250305`` natively and Bedrock rejects the unknown tool type;
        the Anthropic envelope labels any status outside its own table
        ``invalid_request_error``.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
             https://platform.claude.com/docs/en/api/errors
             stdapi/api_providers/anthropic.py:_format_error
        """
        with pytest.raises(BadRequestError) as excinfo:
            anthropic_client.messages.create(
                model="anthropic.claude-haiku-4-5-20251001-v1:0",
                max_tokens=300,
                messages=[{"role": "user", "content": "What is the weather today?"}],
                tools=[_WEB_SEARCH_TOOL],  # type: ignore[list-item]
            )
        error = excinfo.value
        assert error.status_code == 400
        assert error.type == "invalid_request_error"
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["message"]
        assert "request_id" in body


# ===========================================================================
# Web fetch tool
# ===========================================================================


class TestWebFetchTool:
    """Web fetch tool type ``web_fetch_20250910`` — Anthropic-only, never on Bedrock.

    Upstream, web fetch is server-executed and answers with a ``server_tool_use``
    block carrying a ``url``.  Bedrock offers no equivalent, so the only behavior
    reachable through this gateway is the rejection.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
         https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy
    """

    def test_unsupported_on_bedrock_raises_error(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        is_bedrock_direct: bool,
        use_anthropic_api: bool,
    ) -> None:
        """``web_fetch`` on a Bedrock-backed model is refused as a 400 ``invalid_request_error``.

        The gateway has no Bedrock system tool to map ``web_fetch`` onto, so the
        native tool definition reaches Bedrock and is rejected there; the Anthropic
        envelope reports the resulting 400 as ``invalid_request_error``.

        Ref: https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock-legacy
             https://platform.claude.com/docs/en/api/errors
             stdapi/api_providers/anthropic.py:_format_error
        """
        if use_anthropic_api and not is_bedrock_direct:
            pytest.skip("Error path only applies when Bedrock is the backend")
        with pytest.raises(BadRequestError) as excinfo:
            anthropic_client.messages.create(
                model=anthropic_chat_model,
                max_tokens=300,
                messages=[{"role": "user", "content": "Fetch https://example.com"}],
                tools=[_WEB_FETCH_TOOL],  # type: ignore[list-item]
            )
        error = excinfo.value
        assert error.status_code == 400
        assert error.type == "invalid_request_error"
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["message"]
        assert "request_id" in body


# ===========================================================================
# Mixed server tools + custom tools
# ===========================================================================


@pytest.mark.parametrize("model_id", _CLAUDE_SYSTEM_TOOLS)
class TestMixedServerAndCustomTools:
    """Anthropic-defined tools combined with each other and with user-defined tools.

    The two families travel by different routes: a custom tool stays a ``toolSpec``
    entry in ``toolConfig`` while an Anthropic-defined tool moves into
    ``additionalModelRequestFields["tools"]`` on the first turn, so a mixed request
    exercises both branches of the same translation at once.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
         stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
    """

    @pytest.mark.expensive
    def test_server_tool_with_custom_tool(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """A bash tool and a custom ``get_time`` tool are usable in the same request.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             stdapi/models/chat/_adapters/_anthropic_message.py:_build_tool_config
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=1024,
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
        assert response.role == "assistant"
        assert len(response.content) >= 1
        assert response.stop_reason in _TURN_STOP_REASONS
        assert response.usage.output_tokens > 0
        tool_names = {b.name for b in response.content if isinstance(b, ToolUseBlock)}
        assert tool_names <= {"bash", "get_time"}, (
            f"only the offered tools may be called, got {tool_names}"
        )

    @pytest.mark.expensive
    def test_multiple_server_tools_together(
        self, anthropic_client: Anthropic, anthropic_chat_model: str
    ) -> None:
        """``bash_20250124`` and ``text_editor_20250728`` are accepted in one request.

        Both tools map to the same ``anthropic_beta`` flag, which the gateway must
        deduplicate rather than send twice.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-reference
             stdapi/models/chat/_anthropic_claude.py:AnthropicClaudeChatModel._req_configure_tools
        """
        response = anthropic_client.messages.create(
            model=anthropic_chat_model,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Say hello."}],
            tools=[_BASH_TOOL, _TEXT_EDITOR_TOOL],  # type: ignore[list-item]
        )
        assert response.type == "message"
        assert response.role == "assistant"
        assert len(response.content) >= 1
        assert response.stop_reason in _TURN_STOP_REASONS
        assert response.usage.output_tokens > 0
        tool_names = {b.name for b in response.content if isinstance(b, ToolUseBlock)}
        assert tool_names <= {"bash", "str_replace_based_edit_tool"}, (
            f"only the offered tools may be called, got {tool_names}"
        )
