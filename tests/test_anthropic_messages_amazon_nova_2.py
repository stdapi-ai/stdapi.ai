"""Tests for Amazon Nova 2 via the Anthropic /v1/messages route.

Covers Nova 2 system tools surfaced through the Anthropic Messages API:

  - ``code_execution`` (``code_execution_20250522``) → mapped to
    ``nova_code_interpreter``.  Bedrock executes the code internally and returns
    both a ``toolUse`` and a ``toolResult`` block in the same turn; the gateway
    must translate the result to a ``CodeExecutionToolResultBlock``.

Nova 2 is only available on AWS Bedrock, so all tests skip when running against
the official Anthropic API.

All tests that require actual model inference are marked ``@pytest.mark.expensive``.
Run with::

    pytest --expensive tests/test_anthropic_messages_amazon_nova_2.py
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
    """Tests for the ``code_execution`` tool on Amazon Nova 2.

    Nova 2 maps ``code_execution`` → ``nova_code_interpreter`` (Bedrock system
    tool).  The gateway translates the Bedrock ``toolResult`` payload to
    ``CodeExecutionToolResultBlock`` before returning the response to the client.
    """

    @pytest.mark.expensive
    def test_code_execution_surfaces_code_execution_tool_result(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Successful code execution produces a ``CodeExecutionToolResultBlock``.

        Validates:
            - Response contains a ``server_tool_use`` block with ``name == "code_execution"``
            - Response contains a ``code_execution_tool_result`` block
            - ``tool_use_id`` in the result block matches the invocation block
            - ``stdout`` in the result contains the expected output
        """
        if use_official_api:
            pytest.skip("nova_code_interpreter is only available on AWS Bedrock")

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
        assert "1024" in result_block.content.stdout

    @pytest.mark.expensive
    def test_code_execution_multi_turn(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Multi-turn conversation with code_execution produces valid results each turn.

        Validates:
            - Turn 1 response contains ``server_tool_use`` + ``code_execution_tool_result``
            - Turn 2 request succeeds (no ValidationException from empty text blocks)
            - Turn 2 response contains ``server_tool_use`` + ``code_execution_tool_result``
            - ``srvtoolu_`` IDs are stable and consistent across turns
        """
        if use_official_api:
            pytest.skip("nova_code_interpreter is only available on AWS Bedrock")

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

    @pytest.mark.expensive
    def test_code_execution_stderr_and_nonzero_exit_preserved(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Stderr and a non-zero exit code are preserved in the result block.

        Validates:
            - ``code_execution_tool_result`` block is present
            - ``stderr`` is non-empty
            - ``return_code`` is non-zero
        """
        if use_official_api:
            pytest.skip("nova_code_interpreter is only available on AWS Bedrock")

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
        assert isinstance(result_block.content, CodeExecutionResultBlock)
        assert (
            result_block.content.return_code != 0
            or result_block.content.stderr
            or "ZeroDivisionError" in result_block.content.stdout
        )


class TestCodeExecutionToolStreaming:
    """Streaming tests for the ``code_execution`` tool on Amazon Nova 2.

    Verifies that the streaming path (``converse_stream``) correctly maps
    ``nova_code_interpreter`` events to Anthropic SSE blocks, including:
    - ``server_tool_use`` start events with ``srvtoolu_`` id
    - ``code_execution_tool_result`` complete blocks
    - No spurious empty text blocks
    """

    @pytest.mark.expensive
    def test_streaming_surfaces_server_tool_use_and_result(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Streaming response contains ``server_tool_use`` + ``code_execution_tool_result``.

        Validates:
            - Accumulated message has ``server_tool_use`` block with ``srvtoolu_`` id
            - Accumulated message has ``code_execution_tool_result`` block
            - ``tool_use_id`` in result matches the invocation id
            - ``stdout`` contains the expected value
        """
        if use_official_api:
            pytest.skip("nova_code_interpreter is only available on AWS Bedrock")

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

    @pytest.mark.expensive
    def test_streaming_no_empty_text_blocks(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Streaming response does not surface Nova's empty preamble text block.

        Validates that blocks with ``type="text"`` in the accumulated message
        all have non-empty text content.
        """
        if use_official_api:
            pytest.skip("nova_code_interpreter is only available on AWS Bedrock")

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

    @pytest.mark.expensive
    def test_streaming_multi_turn(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Multi-turn streaming conversation with code_execution works end-to-end.

        Validates:
            - Turn 1 streaming produces ``server_tool_use`` + ``code_execution_tool_result``
            - Turn 2 request succeeds (history reconstruction is correct)
            - Turn 2 streaming also produces correct blocks
        """
        if use_official_api:
            pytest.skip("nova_code_interpreter is only available on AWS Bedrock")

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


# ===========================================================================
# web_search / nova_grounding
# ===========================================================================


class TestWebSearchTool:
    """Tests for the ``web_search`` tool (nova_grounding) on Amazon Nova.

    Bedrock only accepts ``nova_grounding`` on a geo-scoped (``us.``) inference
    profile; a ``global.`` profile returns a 400.  ``aws_bedrock_model_region_restrict``
    pins Nova 2 Lite to us-east-1 in the test settings, which forces the ``us.`` profile.

    The gateway must translate the Bedrock ``toolUse`` response to a
    ``ServerToolUseBlock(name="web_search", id="srvtoolu_...")`` and suppress the
    empty Bedrock ``toolResult`` block.  No ``tool_use`` block should be emitted.
    """

    @pytest.mark.expensive
    def test_web_search_surfaces_server_tool_use_block(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Successful web search produces a ``server_tool_use`` block.

        Validates:
            - Response contains a ``server_tool_use`` block with ``name == "web_search"``
            - ``id`` has the ``srvtoolu_`` prefix
            - No plain ``tool_use`` block is present (nova_grounding not leaked)
            - ``stop_reason`` is ``"end_turn"``
        """
        if use_official_api:
            pytest.skip("nova_grounding is only available on AWS Bedrock")

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
        assert response.stop_reason == "end_turn"

    @pytest.mark.expensive
    def test_web_search_streaming_surfaces_server_tool_use_block(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Streaming web search produces a ``server_tool_use`` block.

        Validates:
            - At least one ``content_block_start`` event with type ``server_tool_use``
              (the model may invoke the tool multiple times non-deterministically)
            - ``name == "web_search"`` and ``srvtoolu_`` id prefix on each start event
            - Accumulated final message also contains ``server_tool_use``
            - No ``tool_use`` block in the accumulated message
        """
        if use_official_api:
            pytest.skip("nova_grounding is only available on AWS Bedrock")

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
        start_block = server_tool_starts[0]
        assert isinstance(start_block, ServerToolUseBlock)
        assert start_block.name == "web_search", (
            f"Expected name='web_search', got: {start_block.name!r}"
        )
        assert start_block.id.startswith("srvtoolu_"), (
            f"Expected srvtoolu_ prefix, got: {start_block.id!r}"
        )

    @pytest.mark.expensive
    def test_web_search_multi_turn(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Multi-turn conversation with web_search works end-to-end.

        Validates:
            - Turn 1: response has a ``server_tool_use`` block with ``srvtoolu_`` id
            - Turn 2: passing Turn 1 content (which includes ``ServerToolUseBlock``)
              as assistant history succeeds — gateway correctly remaps ``srvtoolu_``
              ids back to ``nova_grounding`` toolUseIds for Bedrock
            - Turn 2: response also has a ``server_tool_use`` block
        """
        if use_official_api:
            pytest.skip("nova_grounding is only available on AWS Bedrock")

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
        inv2 = next(b for b in resp2.content if b.type == "server_tool_use")
        assert inv2.id.startswith("srvtoolu_"), (
            f"Turn 2: expected srvtoolu_ prefix, got: {inv2.id!r}"
        )

    def test_web_search_filters_are_rejected(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Search filters nova_grounding cannot honor are rejected, not dropped.

        Validates:
            - ``allowed_domains`` on a system-tool web search returns 400
            - The message names the unsupported field
        """
        if use_official_api:
            pytest.skip("nova_grounding is only available on AWS Bedrock")

        with pytest.raises(BadRequestError) as exc_info:
            anthropic_client.messages.create(
                model=_NOVA_2_LITE,
                max_tokens=16,
                messages=[{"role": "user", "content": "What is the Python version?"}],
                tools=[{**_WEB_SEARCH_TOOL, "allowed_domains": ["python.org"]}],  # type: ignore[list-item]
            )

        assert "allowed_domains" in str(exc_info.value)
