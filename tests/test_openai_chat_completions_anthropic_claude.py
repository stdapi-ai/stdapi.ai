"""Tests specific to Anthropic Claude chat completions."""

from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError

if TYPE_CHECKING:
    from openai import OpenAI

#: Anthropic models supporting reasoning
CLAUDE_ALL = (
    "anthropic.claude-3-7-sonnet-20250219-v1:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-opus-4-1-20250805-v1:0",
    "anthropic.claude-opus-4-20250514-v1:0",
    "anthropic.claude-opus-4-5-20251101-v1:0",
    "anthropic.claude-opus-4-6-v1",
    "anthropic.claude-sonnet-4-20250514-v1:0",
    "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-sonnet-4-6",
)

#: A single cheap Claude model for non-parametrized integration tests.
_CLAUDE_CHEAP = "anthropic.claude-haiku-4-5-20251001-v1:0"

#: A non-Claude model for negative tests.
_NON_CLAUDE_MODEL = "amazon.nova-micro-v1:0"


class TestAnthropicClaudeChatCompletions:
    """Anthropic Claude chat completions tests."""

    @pytest.mark.expensive
    @pytest.mark.parametrize("model", CLAUDE_ALL)
    def test_reasoning_effort_parameter(
        self, openai_client: OpenAI, use_official_api: bool, model: str
    ) -> None:
        """reasoning_effort parameter: accepted and yields valid response on this backend."""
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="minimal",
            max_completion_tokens=4096,  # Required for Opus 4.1
        )
        assert hasattr(resp, "choices")
        assert len(resp.choices) >= 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"

    # --- Claude server tools via systemTool_ prefix ---

    @pytest.mark.expensive
    def test_claude_server_tool_bash_accepted(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Short form systemTool_bash is accepted on a Claude model."""
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            tools=[
                {"type": "function", "function": {"name": "systemTool_bash_20250124"}}
            ],
            max_completion_tokens=4096,
        )
        assert hasattr(resp, "choices")
        assert len(resp.choices) >= 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"

    @pytest.mark.expensive
    def test_claude_server_tool_with_custom_tool(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Mixing a Claude server tool with a regular function tool is accepted."""
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")
        resp = openai_client.chat.completions.create(
            model=_CLAUDE_CHEAP,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            tools=[
                {"type": "function", "function": {"name": "systemTool_bash_20250124"}},
                {
                    "type": "function",
                    "function": {
                        "name": "get_time",
                        "description": "Get current time",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                            "required": [],
                        },
                    },
                },
            ],
            max_completion_tokens=4096,
        )
        assert hasattr(resp, "choices")
        assert len(resp.choices) >= 1
        msg = resp.choices[0].message
        assert msg.role == "assistant"

    @pytest.mark.expensive
    def test_claude_server_tool_unsupported_on_non_claude_model(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Claude server tools on a non-Claude model raise a BadRequestError."""
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=_NON_CLAUDE_MODEL,
                messages=[{"role": "user", "content": "Say hello."}],
                tools=[{"type": "function", "function": {"name": "systemTool_bash"}}],
            )
        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"

    # --- Reasoning effort mappings ---

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

    @pytest.mark.expensive
    def test_reasoning_effort_on_non_claude_model_error(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """reasoning_effort on a non-Claude model raises a BadRequestError."""
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=_NON_CLAUDE_MODEL,
                messages=[{"role": "user", "content": "Reply with OK."}],
                reasoning_effort="medium",
            )
        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"

    # --- Unsupported server tool name ---

    @pytest.mark.expensive
    def test_claude_unsupported_server_tool_name(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Unsupported systemTool_ name raises a BadRequestError on Claude."""
        if use_official_api:
            pytest.skip("Anthropic Claude is not supported on the official API")

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(
                model=_CLAUDE_CHEAP,
                messages=[{"role": "user", "content": "Say hello."}],
                tools=[
                    {"type": "function", "function": {"name": "systemTool_nonexistent"}}
                ],
                max_completion_tokens=4096,
            )
        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"

    # --- Streaming with reasoning ---

    @pytest.mark.expensive
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
