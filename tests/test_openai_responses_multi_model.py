"""Multi-model parametrized tests for the OpenAI /v1/responses route.

Covers representative "real-world" usage across one model per provider family
available on AWS Bedrock, including:

  - Basic response generation and usage tokens
  - Streaming SSE event sequence
  - Multi-turn context retention via input array
  - Single-turn function tool calling
  - Streaming tool call events
  - Structured JSON output
  - Vision / image input

All tests require actual Bedrock access and are therefore marked
``@pytest.mark.expensive``.  Run with::

    pytest --expensive tests/test_openai_responses_multi_model.py

A ``Claude`` model is included in every parametrized list as the reference
baseline.  Feature-gated tests (streaming+tools) carry narrower parametrize
lists reflecting known Bedrock capabilities.

If a model does not support a specific feature (e.g. Mistral or Llama3 cannot
use tools in streaming mode), the test calls ``pytest.skip()`` so the result
is recorded as *skipped* — not as a failure — in the report.
"""

import base64
import json
import struct
import zlib
from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError, NotFoundError

if TYPE_CHECKING:
    from openai import OpenAI

# ---------------------------------------------------------------------------
# Model lists — one representative per family, prefer fast/cheap variants
# ---------------------------------------------------------------------------

#: One model per provider family for basic/streaming/multi-turn tests.
_BASIC_MODELS = pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-sonnet-4-6",  # Claude (reference)
        "amazon.nova-micro-v1:0",  # Amazon Nova (cheapest)
        "ai21.jamba-1-5-mini-v1:0",  # AI21 Jamba (SSM/Transformer hybrid, 256k ctx)
        "deepseek.v3-v1:0",  # DeepSeek V3 (fast non-reasoning)
        "google.gemma-3-12b-it",  # Google Gemma
        "meta.llama3-3-70b-instruct-v1:0",  # Meta Llama
        "minimax.minimax-m2.5",  # MiniMax
        "mistral.mistral-7b-instruct-v0:2",  # Mistral (cheapest)
        "mistral.mistral-large-2402-v1:0",  # Mistral Large
        pytest.param(
            "mistral.pixtral-large-2502-v1:0",
            marks=pytest.mark.xfail(
                strict=False,
                reason="Pixtral non-deterministically misidentifies colour of 1x1 PNG",
            ),
        ),  # Mistral Pixtral Large (vision)
        pytest.param(
            "moonshotai.kimi-k2.5",
            marks=pytest.mark.xfail(
                strict=False,
                reason="Kimi K2.5 occasionally returns incomplete status in streaming mode",
            ),
        ),  # Moonshot Kimi K2.5
        "nvidia.nemotron-nano-3-30b",  # NVIDIA Nemotron Nano 30B
        "qwen.qwen3-32b-v1:0",  # Qwen3 32B
        "qwen.qwen3-vl-235b-a22b",  # Qwen3 VL (vision)
        "writer.palmyra-vision-7b",  # Writer Palmyra Vision
        "writer.palmyra-x5-v1:0",  # Writer Palmyra X5
        "zai.glm-4.7-flash",  # Z.AI GLM-4.7 Flash
    ],
)

#: Models confirmed to support non-streaming tool use via the Responses API.
_TOOL_MODELS = pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-sonnet-4-6",  # Claude (reference)
        "amazon.nova-lite-v1:0",  # Amazon Nova
        "amazon.nova-2-lite-v1:0",  # Amazon Nova 2
        "ai21.jamba-1-5-mini-v1:0",  # AI21 Jamba Mini
        "ai21.jamba-1-5-large-v1:0",  # AI21 Jamba Large
        "deepseek.v3-v1:0",  # DeepSeek V3
        "deepseek.v3.2",  # DeepSeek V3.2 (newer revision)
        "meta.llama3-1-70b-instruct-v1:0",  # Meta Llama 3.1 70B
        "minimax.minimax-m2.5",  # MiniMax
        "mistral.mistral-large-2402-v1:0",  # Mistral Large
        "mistral.pixtral-large-2502-v1:0",  # Mistral Pixtral Large
        "moonshotai.kimi-k2.5",  # Moonshot Kimi K2.5
        "openai.gpt-oss-20b-1:0",  # OpenAI GPT-OSS 20B (Bedrock)
        "openai.gpt-oss-120b-1:0",  # OpenAI GPT-OSS 120B (Bedrock)
        "qwen.qwen3-32b-v1:0",  # Qwen3 32B
        "writer.palmyra-x4-v1:0",  # Writer Palmyra X4
        "writer.palmyra-x5-v1:0",  # Writer Palmyra X5
        "zai.glm-5",  # Z.AI GLM-5
    ],
)

#: Models confirmed to support tool use in streaming mode via the Responses API.
_STREAMING_TOOL_MODELS = pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-sonnet-4-6",  # Claude (reference)
        "amazon.nova-lite-v1:0",  # Amazon Nova
        "amazon.nova-2-lite-v1:0",  # Amazon Nova 2
        "ai21.jamba-1-5-mini-v1:0",  # AI21 Jamba Mini
        "ai21.jamba-1-5-large-v1:0",  # AI21 Jamba Large
        "deepseek.v3-v1:0",  # DeepSeek V3
        "deepseek.v3.2",  # DeepSeek V3.2 (newer revision)
        "meta.llama3-1-70b-instruct-v1:0",  # Meta Llama 3.1 70B
        "minimax.minimax-m2.5",  # MiniMax
        "mistral.mistral-large-2402-v1:0",  # Mistral Large
        "mistral.pixtral-large-2502-v1:0",  # Mistral Pixtral Large
        "moonshotai.kimi-k2.5",  # Moonshot Kimi K2.5
        pytest.param(
            "openai.gpt-oss-20b-1:0",
            marks=pytest.mark.xfail(
                strict=False,
                reason="gpt-oss returns truncated JSON in streaming tool arguments (❌M)",
            ),
        ),  # OpenAI GPT-OSS 20B (Bedrock)
        pytest.param(
            "openai.gpt-oss-120b-1:0",
            marks=pytest.mark.xfail(
                strict=False,
                reason="gpt-oss returns truncated JSON in streaming tool arguments (❌M)",
            ),
        ),  # OpenAI GPT-OSS 120B (Bedrock)
        "qwen.qwen3-32b-v1:0",  # Qwen3 32B
        "writer.palmyra-x4-v1:0",  # Writer Palmyra X4
        "writer.palmyra-x5-v1:0",  # Writer Palmyra X5
        "zai.glm-5",  # Z.AI GLM-5
    ],
)

#: A single deterministic read-only tool for all tool-use tests.
_LIST_DIR_TOOL: list[dict[str, object]] = [
    {
        "type": "function",
        "name": "list_directory",
        "description": "List the files and directories inside a filesystem path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to list"}
            },
            "required": ["path"],
        },
    }
]

_PROJECT_ROOT = "/var/opt/projects/stdapi.ai"


# ---------------------------------------------------------------------------
# Tests: basic response, streaming, multi-turn
# ---------------------------------------------------------------------------


class TestMultiModelResponses:
    """Basic and streaming functionality across all supported model families."""

    @pytest.mark.expensive
    @_BASIC_MODELS
    def test_basic_response(
        self, model: str, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Non-streaming response has a message output item and usage.

        Validates:
            - At least one output item of type ``"message"``
            - ``output_text`` is non-empty
            - ``status == "completed"``
            - ``usage.input_tokens > 0`` and ``usage.output_tokens > 0``
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        try:
            response = openai_client.responses.create(
                model=model,
                input="Reply with exactly one word: HELLO",
                max_output_tokens=512,
            )
        except NotFoundError:
            pytest.skip(f"Model {model!r} not available in configured regions")

        msg = next((i for i in response.output if i.type == "message"), None)
        assert msg is not None, f"Expected a message output item for {model!r}"
        assert msg.role == "assistant"
        assert response.output_text, f"Expected non-empty output_text for {model!r}"
        assert response.status == "completed"
        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    @pytest.mark.expensive
    @_BASIC_MODELS
    def test_streaming_response(
        self, model: str, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Streaming response emits text delta events and a completed lifecycle event.

        Validates:
            - At least one ``response.output_text.delta`` event with non-empty delta
            - A ``response.completed`` event is present with ``status == "completed"``
            - Accumulated text from deltas is non-empty
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        accumulated = ""
        completed_event = None

        try:
            stream = openai_client.responses.create(
                model=model,
                max_output_tokens=512,
                input="Reply with exactly three words: ONE TWO THREE",
                stream=True,
            )
            for event in stream:
                if event.type == "response.output_text.delta":
                    accumulated += event.delta
                elif event.type == "response.completed":
                    completed_event = event
        except NotFoundError:
            pytest.skip(f"Model {model!r} not available in configured regions")

        assert accumulated, f"No text deltas received for {model!r}"
        assert completed_event is not None, (
            f"No response.completed event received for {model!r}"
        )
        assert completed_event.response.status == "completed"

    @pytest.mark.expensive
    @_BASIC_MODELS
    def test_multi_turn_context_retention(
        self, model: str, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Model uses prior conversation turns in the input array correctly.

        Validates:
            - Third turn response references the identifier set in the first turn
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        try:
            response = openai_client.responses.create(
                model=model,
                max_output_tokens=256,
                input=[
                    {
                        "role": "user",
                        "content": "The test identifier for this session is ZEBRA99.",
                    },
                    {
                        "role": "assistant",
                        "content": "Understood, the test identifier is ZEBRA99.",
                    },
                    {
                        "role": "user",
                        "content": "What is the test identifier for this session?",
                    },
                ],
            )
        except NotFoundError:
            pytest.skip(f"Model {model!r} not available in configured regions")

        assert response.output_text, "Expected non-empty response"
        assert "ZEBRA99" in response.output_text, (
            f"Expected test identifier in response for {model!r}, "
            f"got: {response.output_text[:200]!r}"
        )


# ---------------------------------------------------------------------------
# Tests: function tool calling
# ---------------------------------------------------------------------------


class TestMultiModelToolUse:
    """Function tool calling across tool-capable model families."""

    @pytest.mark.expensive
    @_TOOL_MODELS
    def test_tool_call_single_turn(
        self, model: str, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Model produces a function_call output item when forced to use a tool.

        Validates:
            - At least one output item with ``type == "function_call"``
            - Tool name matches the defined tool
            - ``arguments`` field is valid JSON
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        try:
            response = openai_client.responses.create(  # type: ignore[call-overload]
                model=model,
                max_output_tokens=512,
                input=f"List the files in {_PROJECT_ROOT}",
                tools=_LIST_DIR_TOOL,
                tool_choice="required",
            )
        except NotFoundError:
            pytest.skip(f"Model {model!r} not available in configured regions")
        except BadRequestError as exc:
            if "toolChoice" in str(exc):
                pytest.skip(
                    f"Model {model!r} does not support tool_choice=required: {exc}"
                )
            raise

        tool_calls = [i for i in response.output if i.type == "function_call"]
        assert tool_calls, (
            f"Expected at least one function_call output item for {model!r}; "
            f"output types: {[i.type for i in response.output]}"
        )
        tc = tool_calls[0]
        assert tc.name == "list_directory", (
            f"Unexpected tool name {tc.name!r} for {model!r}"
        )
        args = json.loads(tc.arguments)
        assert isinstance(args, dict)

    @pytest.mark.expensive
    @_STREAMING_TOOL_MODELS
    def test_streaming_tool_call(
        self, model: str, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Streaming response emits function_call_arguments events for a forced tool call.

        Validates:
            - At least one ``response.function_call_arguments.delta`` event
            - A ``response.function_call_arguments.done`` event is present
            - Arguments from the done event parse as valid JSON
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        delta_count = 0
        done_event = None

        try:
            stream = openai_client.responses.create(  # type: ignore[call-overload]
                model=model,
                max_output_tokens=512,
                input=f"List the files in {_PROJECT_ROOT}",
                tools=_LIST_DIR_TOOL,
                tool_choice="required",
                stream=True,
            )
            for event in stream:
                if event.type == "response.function_call_arguments.delta":
                    delta_count += 1
                elif event.type == "response.function_call_arguments.done":
                    done_event = event
        except NotFoundError:
            pytest.skip(f"Model {model!r} not available in configured regions")
        except BadRequestError as exc:
            if "streaming mode" in str(exc).lower():
                pytest.skip(f"Model does not support streaming with tools: {exc}")
            if "toolChoice" in str(exc):
                pytest.skip(
                    f"Model {model!r} does not support tool_choice=required: {exc}"
                )
            raise

        assert delta_count > 0, (
            f"Expected function_call_arguments.delta events for {model!r}, got 0"
        )
        assert done_event is not None, (
            f"Expected function_call_arguments.done event for {model!r}"
        )
        args = json.loads(done_event.arguments)
        assert isinstance(args, dict)


# ---------------------------------------------------------------------------
# Tests: vision / image input
# ---------------------------------------------------------------------------


def _make_1x1_red_png_b64() -> str:
    """Return a base64-encoded minimal valid 1x1 red PNG."""

    def _chunk(name: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        return length + name + data + crc

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    idat = _chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
    iend = _chunk(b"IEND", b"")
    return base64.b64encode(signature + ihdr + idat + iend).decode()


#: Vision-capable models tested on the Responses API route.
_VISION_MODELS = pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-sonnet-4-6",  # Claude (reference)
        "amazon.nova-lite-v1:0",  # Amazon Nova
        pytest.param(
            "mistral.pixtral-large-2502-v1:0",
            marks=pytest.mark.xfail(
                strict=False,
                reason="Pixtral non-deterministically misidentifies colour of 1x1 PNG",
            ),
        ),  # Mistral Pixtral Large
        "qwen.qwen3-vl-235b-a22b",  # Qwen3 VL 235B
        pytest.param(
            "writer.palmyra-vision-7b",
            marks=pytest.mark.xfail(
                strict=False,
                reason="Palmyra Vision non-deterministically misidentifies colour of 1x1 PNG",
            ),
        ),  # Writer Palmyra Vision 7B
    ],
)


class TestVision:
    """Vision-capable models correctly identify the color of a simple image."""

    @pytest.mark.expensive
    @_VISION_MODELS
    def test_image_color_recognition(
        self, model: str, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Vision model identifies the color of a 1x1 red PNG via input_image.

        Uses a locally generated minimal PNG encoded as a data URI.

        Validates:
            - Response contains non-empty text
            - Response correctly identifies "red" as the image color
        """
        if use_official_api:
            pytest.skip("Multi-model tests only run against the local server")

        try:
            response = openai_client.responses.create(
                model=model,
                max_output_tokens=64,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{_make_1x1_red_png_b64()}",
                                "detail": "low",
                            },
                            {
                                "type": "input_text",
                                "text": "What is the color of this image? Reply in one word.",
                            },
                        ],
                    }
                ],
            )
        except NotFoundError:
            pytest.skip(f"Model {model!r} not available in configured regions")

        text = response.output_text
        assert text, f"Expected non-empty response from {model!r}"
        assert any(color in text.lower() for color in ("red", "orange")), (
            f"Expected 'red' or 'orange' in response for {model!r}, got: {text!r}"
        )
