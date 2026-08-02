"""Amazon Nova 2 system tools exposed through the Anthropic /v1/messages route.

Anthropic's server tools do not exist on Bedrock, so the gateway maps the canonical tool
types onto Nova's Bedrock ``systemTool`` entries: ``code_execution`` →
``nova_code_interpreter`` and ``web_search`` → ``nova_grounding``.  Bedrock runs both
tools inside the invocation and answers with a ``toolUse`` (plus, for the code
interpreter, a ``toolResult``) in the same turn, which the gateway republishes as
``server_tool_use`` / ``code_execution_tool_result`` blocks carrying ``srvtoolu_`` ids.

Inference tests run with ``pytest --expensive``.

Ref: https://platform.claude.com/docs/en/api/messages
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_SystemTool.html
     https://docs.aws.amazon.com/nova/latest/nova2-userguide/using-tools.html
     stdapi/routes/anthropic_messages.py:create_message
     stdapi/models/chat/amazon_nova_2.py:ChatModel
"""

from typing import TYPE_CHECKING

import pytest
from anthropic import BadRequestError
from anthropic.types import (
    CodeExecutionResultBlock,
    CodeExecutionToolResultBlock,
    ServerToolUseBlock,
)

if TYPE_CHECKING:
    from anthropic import Anthropic

#: Nova 2 Lite — smallest Nova 2 model, sufficient for code execution tests.
_NOVA_2_LITE = "amazon.nova-2-lite-v1:0"

pytestmark = pytest.mark.gateway(
    "Nova 2 system tools are only available on AWS Bedrock"
)

#: code_execution tool definition (Anthropic canonical format).
_CODE_EXECUTION_TOOL: dict[str, object] = {
    "type": "code_execution_20250522",
    "name": "code_execution",
}

#: web_search tool definition (Anthropic canonical format).
_WEB_SEARCH_TOOL: dict[str, object] = {
    "type": "web_search_20250305",
    "name": "web_search",
}


def _result_stdout(block: CodeExecutionToolResultBlock) -> str:
    """Return stdout from a ``code_execution_tool_result`` block."""
    content = block.content
    return content.stdout if isinstance(content, CodeExecutionResultBlock) else ""


# ===========================================================================
# code_execution / nova_code_interpreter
# ===========================================================================


class TestCodeExecutionTool:
    """The ``code_execution`` server tool served by ``nova_code_interpreter``.

    Bedrock returns the interpreter output as a ``toolResult`` whose first content item is
    a JSON payload of ``stdOut`` / ``stdErr`` / ``exitCode`` / ``isError``; the gateway
    converts it to a ``code_execution_tool_result`` block whose ``return_code`` is
    ``exitCode or 1`` when ``isError`` is set and ``0`` otherwise.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
         stdapi/models/chat/amazon_nova_2.py:ChatModel._build_code_execution_result
    """

    @pytest.mark.expensive
    def test_code_execution_surfaces_code_execution_tool_result(
        self, anthropic_client: Anthropic
    ) -> None:
        """Successful code execution produces a ``CodeExecutionToolResultBlock``.

        The invocation and its result are correlated by id: the Bedrock ``toolUseId`` is
        re-prefixed ``srvtoolu_`` on both blocks, so the result must point back at the
        emitted ``server_tool_use`` rather than at the raw Bedrock id.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
             stdapi/models/chat/amazon_nova_2.py:ChatModel._resp_map_tool_result
        """
        response = anthropic_client.messages.create(
            model=_NOVA_2_LITE,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "Compute 2 ** 10 using code. Show me the result.",
                }
            ],
            tools=[_CODE_EXECUTION_TOOL],  # type: ignore[list-item]
        )

        block_types = {b.type for b in response.content}
        assert "server_tool_use" in block_types, (
            f"Expected server_tool_use block, got: {block_types}"
        )
        assert "code_execution_tool_result" in block_types, (
            f"Expected code_execution_tool_result block, got: {block_types}"
        )
        assert "tool_use" not in block_types, (
            f"nova_code_interpreter must not leak as a client tool_use: {block_types}"
        )
        assert response.stop_reason != "tool_use", (
            "Bedrock executes the code itself, so the client is never asked to run a tool"
        )

        invocation = next(b for b in response.content if b.type == "server_tool_use")
        assert invocation.name == "code_execution"
        assert invocation.id.startswith("srvtoolu_"), (
            f"Expected srvtoolu_ prefix on server_tool_use id, got: {invocation.id!r}"
        )

        result_block = next(
            b for b in response.content if b.type == "code_execution_tool_result"
        )
        assert isinstance(result_block, CodeExecutionToolResultBlock)
        assert result_block.tool_use_id == invocation.id
        assert isinstance(result_block.content, CodeExecutionResultBlock)
        assert result_block.content.type == "code_execution_result"
        assert "1024" in result_block.content.stdout
        assert result_block.content.return_code == 0, (
            f"Successful execution must map to return_code 0, got: "
            f"{result_block.content.return_code}"
        )
        assert response.model == _NOVA_2_LITE
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    @pytest.mark.expensive
    def test_code_execution_multi_turn(self, anthropic_client: Anthropic) -> None:
        """A ``code_execution_tool_result`` can be replayed as assistant history.

        Turn 2 echoes the turn-1 blocks back, which forces the reverse mapping:
        ``srvtoolu_`` ids become Bedrock ``tooluse_`` ids again and the result block is
        rebuilt as a Bedrock ``toolResult`` with a ``status`` derived from
        ``return_code``.  A broken round trip makes Converse reject the conversation.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/models/chat/amazon_nova_2.py:ChatModel._req_map_content_block
        """
        # ── Turn 1 ──────────────────────────────────────────────────────────
        resp1 = anthropic_client.messages.create(
            model=_NOVA_2_LITE,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "Compute 6 ** 6 using code. Show me the result.",
                }
            ],
            tools=[_CODE_EXECUTION_TOOL],  # type: ignore[list-item]
        )

        block_types1 = {b.type for b in resp1.content}
        assert "server_tool_use" in block_types1, (
            f"Turn 1: expected server_tool_use, got: {block_types1}"
        )
        assert "code_execution_tool_result" in block_types1, (
            f"Turn 1: expected code_execution_tool_result, got: {block_types1}"
        )
        inv1 = next(b for b in resp1.content if b.type == "server_tool_use")
        assert inv1.id.startswith("srvtoolu_"), (
            f"Turn 1: expected srvtoolu_ prefix, got: {inv1.id!r}"
        )
        res1 = next(b for b in resp1.content if b.type == "code_execution_tool_result")
        assert isinstance(res1, CodeExecutionToolResultBlock)
        assert res1.tool_use_id == inv1.id
        assert isinstance(res1.content, CodeExecutionResultBlock)
        assert "46656" in res1.content.stdout

        # ── Turn 2 ──────────────────────────────────────────────────────────
        resp2 = anthropic_client.messages.create(
            model=_NOVA_2_LITE,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "Compute 6 ** 6 using code. Show me the result.",
                },
                {"role": "assistant", "content": resp1.content},
                {"role": "user", "content": "Now compute 7 ** 7 using code."},
            ],
            tools=[_CODE_EXECUTION_TOOL],  # type: ignore[list-item]
        )

        block_types2 = {b.type for b in resp2.content}
        assert "server_tool_use" in block_types2, (
            f"Turn 2: expected server_tool_use, got: {block_types2}"
        )
        assert "code_execution_tool_result" in block_types2, (
            f"Turn 2: expected code_execution_tool_result, got: {block_types2}"
        )
        inv2 = next(b for b in resp2.content if b.type == "server_tool_use")
        assert inv2.id.startswith("srvtoolu_"), (
            f"Turn 2: expected srvtoolu_ prefix, got: {inv2.id!r}"
        )
        res2 = next(b for b in resp2.content if b.type == "code_execution_tool_result")
        assert isinstance(res2, CodeExecutionToolResultBlock)
        assert res2.tool_use_id == inv2.id
        assert isinstance(res2.content, CodeExecutionResultBlock)
        assert "823543" in res2.content.stdout
        assert resp2.usage.input_tokens > 0, (
            "Turn 2 must bill the replayed code-execution history"
        )

    @pytest.mark.expensive
    def test_code_execution_stderr_and_nonzero_exit_preserved(
        self, anthropic_client: Anthropic
    ) -> None:
        """A failing snippet still returns a result block reporting the failure.

        Bedrock decides where the traceback lands: it may set ``isError`` (mapped to a
        non-zero ``return_code``), fill ``stdErr``, or print the exception on ``stdOut``.
        The block must therefore carry the failure through at least one of the three
        channels rather than being dropped or replaced by an error envelope.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
             stdapi/models/chat/amazon_nova_2.py:ChatModel._build_code_execution_result
        """
        response = anthropic_client.messages.create(
            model=_NOVA_2_LITE,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "Run this exact Python code: raise ZeroDivisionError('test')",
                }
            ],
            tools=[_CODE_EXECUTION_TOOL],  # type: ignore[list-item]
        )

        block_types = {b.type for b in response.content}
        assert "code_execution_tool_result" in block_types, (
            f"Expected code_execution_tool_result block, got: {block_types}"
        )

        result_block = next(
            b for b in response.content if b.type == "code_execution_tool_result"
        )
        assert isinstance(result_block, CodeExecutionToolResultBlock)
        assert result_block.tool_use_id.startswith("srvtoolu_"), (
            f"Expected srvtoolu_ prefix, got: {result_block.tool_use_id!r}"
        )
        assert isinstance(result_block.content, CodeExecutionResultBlock)
        assert (
            result_block.content.return_code != 0
            or result_block.content.stderr
            or "ZeroDivisionError" in result_block.content.stdout
        )


class TestCodeExecutionToolStreaming:
    """The ``code_execution`` server tool over the Anthropic SSE stream.

    Bedrock streams the interpreter result as a ``toolResult`` block split over
    ``contentBlockDelta`` events; the gateway buffers them and emits one complete
    ``code_execution_tool_result`` in a single ``content_block_start``, because Anthropic
    has no delta shape for that block type.

    Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
         stdapi/models/chat/amazon_nova_2.py:ChatModel._resp_stream_map_tool_result
    """

    @pytest.mark.expensive
    def test_streaming_surfaces_server_tool_use_and_result(
        self, anthropic_client: Anthropic
    ) -> None:
        """Streaming response contains ``server_tool_use`` + ``code_execution_tool_result``.

        The result block arrives as exactly one ``content_block_start`` event with its
        payload already complete, and its ``tool_use_id`` matches the ``srvtoolu_`` id of
        the invocation block accumulated in the final message.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_adapters/_anthropic_message.py:_process_content_block_stop
        """
        with anthropic_client.messages.stream(
            model=_NOVA_2_LITE,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": "Compute 3 ** 3 using code. Show me the result.",
                }
            ],
            tools=[_CODE_EXECUTION_TOOL],  # type: ignore[list-item]
        ) as stream:
            code_result_blocks = [
                event.content_block
                for event in stream
                if event.type == "content_block_start"
                and event.content_block.type == "code_execution_tool_result"
            ]
            msg = stream.get_final_message()

        block_types = {b.type for b in msg.content}
        assert "server_tool_use" in block_types, (
            f"Expected server_tool_use block, got: {block_types}"
        )
        assert "code_execution_tool_result" in block_types, (
            f"Expected code_execution_tool_result block, got: {block_types}"
        )
        assert "tool_use" not in block_types, (
            f"nova_code_interpreter must not leak as a client tool_use: {block_types}"
        )

        invocation = next(b for b in msg.content if b.type == "server_tool_use")
        assert invocation.name == "code_execution"
        assert invocation.id.startswith("srvtoolu_"), (
            f"Expected srvtoolu_ prefix, got: {invocation.id!r}"
        )

        assert len(code_result_blocks) == 1
        result_block = code_result_blocks[0]
        assert isinstance(result_block, CodeExecutionToolResultBlock)
        assert result_block.tool_use_id == invocation.id
        assert "27" in _result_stdout(result_block)
        assert msg.model == _NOVA_2_LITE
        assert msg.usage.output_tokens > 0

    @pytest.mark.expensive
    def test_streaming_no_empty_text_blocks(self, anthropic_client: Anthropic) -> None:
        """Streaming response does not surface Nova's empty preamble text block.

        Nova opens a system-tool turn with a text block whose only delta is ``{"text":
        ""}``.  Suppression is deferred until ``contentBlockStop``, so the block is
        discarded while a block that later receives real text is still emitted.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_adapters/_anthropic_message.py:_process_content_block_delta
        """
        with anthropic_client.messages.stream(
            model=_NOVA_2_LITE,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Compute 4 ** 4 using code."}],
            tools=[_CODE_EXECUTION_TOOL],  # type: ignore[list-item]
        ) as stream:
            msg = stream.get_final_message()

        for block in msg.content:
            if block.type == "text":
                assert block.text, (
                    f"Empty text block found in streaming response: {block!r}"
                )
        assert msg.content, "Expected at least one content block in the final message"

    @pytest.mark.expensive
    def test_streaming_multi_turn(self, anthropic_client: Anthropic) -> None:
        """Streamed code-execution blocks are accepted back as assistant history.

        The blocks replayed in turn 2 are the ones rebuilt from buffered stream deltas, so
        this covers the streaming half of the ``srvtoolu_`` ↔ ``tooluse_`` round trip.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/amazon_nova_2.py:ChatModel._req_map_content_block
        """
        # ── Turn 1 ──────────────────────────────────────────────────────────
        with anthropic_client.messages.stream(
            model=_NOVA_2_LITE,
            max_tokens=1024,
            messages=[{"role": "user", "content": "Compute 5 ** 5 using code."}],
            tools=[_CODE_EXECUTION_TOOL],  # type: ignore[list-item]
        ) as stream:
            code_result_blocks1 = [
                event.content_block
                for event in stream
                if event.type == "content_block_start"
                and event.content_block.type == "code_execution_tool_result"
            ]
            resp1 = stream.get_final_message()

        block_types1 = {b.type for b in resp1.content}
        assert "server_tool_use" in block_types1, (
            f"Turn 1: expected server_tool_use, got: {block_types1}"
        )
        assert "code_execution_tool_result" in block_types1, (
            f"Turn 1: expected code_execution_tool_result, got: {block_types1}"
        )
        inv1 = next(b for b in resp1.content if b.type == "server_tool_use")
        assert len(code_result_blocks1) == 1
        res1 = code_result_blocks1[0]
        assert isinstance(res1, CodeExecutionToolResultBlock)
        assert res1.tool_use_id == inv1.id
        assert "3125" in _result_stdout(res1)

        # ── Turn 2 ──────────────────────────────────────────────────────────
        with anthropic_client.messages.stream(
            model=_NOVA_2_LITE,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "Compute 5 ** 5 using code."},
                {"role": "assistant", "content": resp1.content},
                {"role": "user", "content": "Now compute 6 ** 6 using code."},
            ],
            tools=[_CODE_EXECUTION_TOOL],  # type: ignore[list-item]
        ) as stream:
            code_result_blocks2 = [
                event.content_block
                for event in stream
                if event.type == "content_block_start"
                and event.content_block.type == "code_execution_tool_result"
            ]
            resp2 = stream.get_final_message()

        block_types2 = {b.type for b in resp2.content}
        assert "server_tool_use" in block_types2, (
            f"Turn 2: expected server_tool_use, got: {block_types2}"
        )
        assert "code_execution_tool_result" in block_types2, (
            f"Turn 2: expected code_execution_tool_result, got: {block_types2}"
        )
        inv2 = next(b for b in resp2.content if b.type == "server_tool_use")
        assert len(code_result_blocks2) == 1
        res2 = code_result_blocks2[0]
        assert isinstance(res2, CodeExecutionToolResultBlock)
        assert res2.tool_use_id == inv2.id
        assert "46656" in _result_stdout(res2)
        assert resp2.usage.input_tokens > 0, (
            "Turn 2 must bill the replayed code-execution history"
        )


# ===========================================================================
# web_search / nova_grounding
# ===========================================================================


class TestWebSearchTool:
    """The ``web_search`` server tool served by ``nova_grounding``.

    Bedrock only accepts ``nova_grounding`` on a geo-scoped (``us.``) inference profile
    and answers a ``global.`` profile with a 400, so ``conftest`` pins Nova 2 Lite to
    us-east-1 through ``aws_bedrock_model_region_restrict``.  Bedrock searches inside the
    invocation: the gateway republishes its ``toolUse`` as a ``server_tool_use`` block and
    any ``searchResult`` block as a ``web_search_tool_result`` keyed by that block's id,
    so no client-executable ``tool_use`` is ever emitted.

    Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
         stdapi/models/chat/_default.py:ChatModel._resp_map_tool_use
    """

    @pytest.mark.expensive
    def test_web_search_surfaces_server_tool_use_block(
        self, anthropic_client: Anthropic
    ) -> None:
        """Successful web search produces a ``server_tool_use`` block.

        The block is renamed to Anthropic's canonical ``web_search`` by the inverse
        lookup over ``CANONICAL_TO_BEDROCK_TOOL_MAP``, and the turn ends normally because
        Bedrock has already consumed the search itself.

        Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
             stdapi/models/chat/_default.py:ChatModel._canonical_name_for
        """
        response = anthropic_client.messages.create(
            model=_NOVA_2_LITE,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "What is the current version of Python?"}
            ],
            tools=[_WEB_SEARCH_TOOL],  # type: ignore[list-item]
        )

        block_types = {b.type for b in response.content}
        assert "server_tool_use" in block_types, (
            f"Expected server_tool_use block, got: {block_types}"
        )
        assert "tool_use" not in block_types, (
            f"nova_grounding must not leak as tool_use, got: {block_types}"
        )

        invocation = next(b for b in response.content if b.type == "server_tool_use")
        assert isinstance(invocation, ServerToolUseBlock)
        assert invocation.name == "web_search", (
            f"Expected name='web_search', got: {invocation.name!r}"
        )
        assert invocation.id.startswith("srvtoolu_"), (
            f"Expected srvtoolu_ prefix, got: {invocation.id!r}"
        )
        # Bedrock searchResult blocks are wrapped in web_search_tool_result and
        # must reference the emitted server_tool_use, never the raw Bedrock ID.
        for block in response.content:
            if block.type == "web_search_tool_result":
                assert block.tool_use_id == invocation.id, (
                    f"Expected tool_use_id={invocation.id!r}, got: {block.tool_use_id!r}"
                )
                assert isinstance(block.content, list), (
                    "A Bedrock searchResult must be wrapped in a content list"
                )
                for result in block.content:
                    assert result.type == "web_search_result"
        assert response.stop_reason == "end_turn"
        assert response.model == _NOVA_2_LITE
        assert response.usage.output_tokens > 0

    @pytest.mark.expensive
    def test_web_search_streaming_surfaces_server_tool_use_block(
        self, anthropic_client: Anthropic
    ) -> None:
        """Streaming web search produces a ``server_tool_use`` block.

        Nova may ground an answer with several searches, so the number of start events is
        not fixed; each one must still carry the canonical name and a ``srvtoolu_`` id.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_default.py:ChatModel._resp_stream_map_tool_use
        """
        with anthropic_client.messages.stream(
            model=_NOVA_2_LITE,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "What is the current version of Python?"}
            ],
            tools=[_WEB_SEARCH_TOOL],  # type: ignore[list-item]
        ) as stream:
            server_tool_starts = [
                event.content_block
                for event in stream
                if event.type == "content_block_start"
                and event.content_block.type == "server_tool_use"
            ]
            msg = stream.get_final_message()

        block_types = {b.type for b in msg.content}
        assert "server_tool_use" in block_types, (
            f"Expected server_tool_use in final message, got: {block_types}"
        )
        assert "tool_use" not in block_types, (
            f"nova_grounding must not leak as tool_use, got: {block_types}"
        )

        assert len(server_tool_starts) >= 1, (
            f"Expected at least one server_tool_use start, got {len(server_tool_starts)}"
        )
        for start in server_tool_starts:
            assert isinstance(start, ServerToolUseBlock)
            assert start.name == "web_search", (
                f"Expected name='web_search', got: {start.name!r}"
            )
            assert start.id.startswith("srvtoolu_"), (
                f"Expected srvtoolu_ prefix, got: {start.id!r}"
            )

    @pytest.mark.expensive
    def test_web_search_multi_turn(self, anthropic_client: Anthropic) -> None:
        """A ``server_tool_use`` block can be replayed as assistant history.

        ``_req_map_content_block`` turns the echoed block back into a Bedrock ``toolUse``
        named ``nova_grounding`` with the ``srvtoolu_`` prefix swapped for ``tooluse_``;
        without that rewrite Converse rejects the assistant turn as an unknown tool.

        Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
             stdapi/models/chat/_default.py:ChatModel._req_map_content_block
        """
        # ── Turn 1 ──────────────────────────────────────────────────────────
        resp1 = anthropic_client.messages.create(
            model=_NOVA_2_LITE,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "What is the current Python version?"}
            ],
            tools=[_WEB_SEARCH_TOOL],  # type: ignore[list-item]
        )

        block_types1 = {b.type for b in resp1.content}
        assert "server_tool_use" in block_types1, (
            f"Turn 1: expected server_tool_use, got: {block_types1}"
        )
        inv1 = next(b for b in resp1.content if b.type == "server_tool_use")
        assert inv1.id.startswith("srvtoolu_"), (
            f"Turn 1: expected srvtoolu_ prefix, got: {inv1.id!r}"
        )

        # ── Turn 2 ──────────────────────────────────────────────────────────
        # resp1.content contains ServerToolUseBlock objects; the gateway must
        # translate them back to nova_grounding toolUse blocks for Bedrock.
        resp2 = anthropic_client.messages.create(
            model=_NOVA_2_LITE,
            max_tokens=1024,
            messages=[
                {"role": "user", "content": "What is the current Python version?"},
                {"role": "assistant", "content": resp1.content},
                {"role": "user", "content": "And what about Node.js?"},
            ],
            tools=[_WEB_SEARCH_TOOL],  # type: ignore[list-item]
        )

        block_types2 = {b.type for b in resp2.content}
        assert "server_tool_use" in block_types2, (
            f"Turn 2: expected server_tool_use, got: {block_types2}"
        )
        assert "tool_use" not in block_types2, (
            f"Turn 2: nova_grounding must not leak as tool_use, got: {block_types2}"
        )
        inv2 = next(b for b in resp2.content if b.type == "server_tool_use")
        assert inv2.id.startswith("srvtoolu_"), (
            f"Turn 2: expected srvtoolu_ prefix, got: {inv2.id!r}"
        )
        assert resp2.usage.input_tokens > 0, (
            "Turn 2 must bill the replayed grounding history"
        )
        assert resp2.usage.output_tokens > 0

    def test_web_search_filters_are_rejected(self, anthropic_client: Anthropic) -> None:
        """Search filters nova_grounding cannot honor are rejected, not dropped.

        Anthropic's ``web_search`` accepts ``allowed_domains`` / ``blocked_domains`` /
        ``max_uses`` / ``user_location``, but a Bedrock ``systemTool`` takes no input, so
        silently dropping them would return unfiltered results.  ``_handle_system_tool``
        fails the request instead, and the gateway renders the 400 as an
        ``invalid_request_error`` naming the offending fields.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
             stdapi/models/chat/_adapters/_anthropic_message.py:_handle_system_tool
             stdapi/api_providers/anthropic.py:_format_error
        """
        with pytest.raises(BadRequestError) as exc_info:
            anthropic_client.messages.create(
                model=_NOVA_2_LITE,
                max_tokens=16,
                messages=[{"role": "user", "content": "What is the Python version?"}],
                tools=[{**_WEB_SEARCH_TOOL, "allowed_domains": ["python.org"]}],  # type: ignore[list-item]
            )

        assert exc_info.value.status_code == 400
        assert exc_info.value.type == "invalid_request_error", (
            f"Expected an invalid_request_error envelope, got: {exc_info.value.type!r}"
        )
        assert "allowed_domains" in str(exc_info.value)
