"""Unit tests for OpenAI Responses API input-side parsing and Bedrock mapping."""

from base64 import b64encode
from typing import Any, cast

import pytest
from pydantic import TypeAdapter

from stdapi.models.chat._adapters import _openai_responses as adapter
from stdapi.models.chat._adapters._openai_responses import (
    _map_tool_choice,
    encode_reasoning_content,
    extract_reasoning,
    map_input,
)
from stdapi.types.openai_responses import (
    CodeInterpreterCallInput,
    CompactionTrigger,
    ComputerCallInput,
    CustomToolCallInput,
    CustomToolCallOutput,
    FileSearchCallInput,
    FunctionCallInput,
    FunctionCallOutput,
    ImageGenerationCallInput,
    InputTokenCountParams,
    Reasoning,
    ReasoningItemSummary,
    ResponseCreateParams,
    ResponseInputFile,
    ResponseInputImage,
    ResponseInputItem,
    ResponseInputText,
    ResponseOutputMessage,
    ResponseOutputMessageInput,
    ResponseOutputRefusal,
    ResponseReasoningItem,
    ResponseReasoningItemInput,
    ToolChoiceAllowed,
    ToolChoiceTypes,
    WebSearchCallInput,
)

#: Mark the whole module as local (in-process, no AWS calls).
pytestmark = pytest.mark.local

#: Pydantic adapter validating raw dicts against the full input item union.
_ITEM_ADAPTER: TypeAdapter[ResponseInputItem] = TypeAdapter[ResponseInputItem](
    ResponseInputItem
)

#: Minimal PNG payload (magic bytes plus filler).
_PNG_BYTES = b"\x89PNG\r\n\x1a\n-fake-png-body"

#: Minimal JPEG payload (magic bytes plus filler).
_JPEG_BYTES = b"\xff\xd8\xff\xe0-fake-jpeg-body"


def _parse(payload: dict[str, object]) -> object:
    """Validate a raw item payload against the ResponseInputItem union."""
    return _ITEM_ADAPTER.validate_python(payload)


class TestHostedToolItemsParseAndDrop:
    """Hosted-tool call items parse without error and are dropped on mapping."""

    @pytest.mark.parametrize(
        ("payload", "expected_cls"),
        [
            (
                {
                    "id": "ws_1",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {"type": "search", "query": "cats"},
                },
                WebSearchCallInput,
            ),
            (
                {
                    "id": "fs_1",
                    "type": "file_search_call",
                    "status": "completed",
                    "queries": ["cats"],
                    "results": [{"file_id": "f_1", "text": "meow", "score": 0.9}],
                },
                FileSearchCallInput,
            ),
            (
                {
                    "id": "ci_1",
                    "type": "code_interpreter_call",
                    "container_id": "cont_1",
                    "status": "completed",
                    "code": "print(1)",
                    "outputs": [{"type": "logs", "logs": "1"}],
                },
                CodeInterpreterCallInput,
            ),
            (
                {
                    "id": "cc_1",
                    "type": "computer_call",
                    "call_id": "call_1",
                    "status": "completed",
                    "pending_safety_checks": [],
                    "action": {"type": "screenshot"},
                },
                ComputerCallInput,
            ),
            ({"type": "compaction_trigger"}, CompactionTrigger),
        ],
    )
    async def test_parses_and_is_dropped(
        self, payload: dict[str, object], expected_cls: type
    ) -> None:
        """Each hosted item validates to its model and maps to no message."""
        item = _parse(payload)
        assert isinstance(item, expected_cls)
        messages, system = await map_input(
            cast("list[ResponseInputItem]", [item]), None
        )
        assert messages == []
        assert system == []


class TestCustomToolCallItems:
    """custom_tool_call and custom_tool_call_output map to toolUse/toolResult."""

    async def test_call_with_json_object_input(self) -> None:
        """JSON object input is parsed into the Bedrock toolUse input."""
        item = _parse(
            {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "sql",
                "input": '{"query": "SELECT 1"}',
            }
        )
        assert isinstance(item, CustomToolCallInput)
        messages, _ = await map_input([item], None)
        assert messages == [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "call_1",
                            "name": "sql",
                            "input": {"query": "SELECT 1"},
                        }
                    }
                ],
            }
        ]

    async def test_call_with_freeform_input_is_wrapped(self) -> None:
        """Non-JSON freeform input is wrapped as {"input": <raw>}."""
        item = CustomToolCallInput(
            type="custom_tool_call",
            call_id="call_1",
            name="sql",
            input="SELECT * FROM cats",
        )
        messages, _ = await map_input([item], None)
        tool_use = messages[0]["content"][0]["toolUse"]
        assert tool_use["input"] == {"input": "SELECT * FROM cats"}

    async def test_output_string_form(self) -> None:
        """String output becomes a user toolResult with a text block."""
        item = _parse(
            {"type": "custom_tool_call_output", "call_id": "call_1", "output": "done"}
        )
        assert isinstance(item, CustomToolCallOutput)
        messages, _ = await map_input([item], None)
        assert messages == [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "call_1",
                            "content": [{"text": "done"}],
                        }
                    }
                ],
            }
        ]

    async def test_output_list_form_keeps_text_parts(self) -> None:
        """Content-part list output keeps text parts as text blocks."""
        item = CustomToolCallOutput(
            type="custom_tool_call_output",
            call_id="call_1",
            output=[
                ResponseInputText(type="input_text", text="a"),
                ResponseInputText(type="input_text", text="b"),
            ],
        )
        messages, _ = await map_input([item], None)
        tool_result = messages[0]["content"][0]["toolResult"]
        assert tool_result["content"] == [{"text": "a"}, {"text": "b"}]


class TestEchoedItemsTolerateUnknownFields:
    """Echoed output items parse even when carrying unknown upstream fields."""

    async def test_output_message_with_unknown_field(self) -> None:
        """An echoed assistant message with a new field parses and maps."""
        item = _parse(
            {
                "type": "message",
                "role": "assistant",
                "id": "msg_1",
                "status": "completed",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}],
                "brand_new_upstream_field": "x",
            }
        )
        assert isinstance(item, ResponseOutputMessageInput)
        messages, _ = await map_input(cast("list[ResponseInputItem]", [item]), None)
        assert messages == [{"role": "assistant", "content": [{"text": "hi"}]}]

    async def test_reasoning_item_with_unknown_field(self) -> None:
        """An echoed reasoning item with a new field parses without error."""
        item = _parse(
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "brand_new_upstream_field": 1,
            }
        )
        assert isinstance(item, ResponseReasoningItemInput)
        messages, _ = await map_input(cast("list[ResponseInputItem]", [item]), None)
        assert messages == []

    @pytest.mark.parametrize(
        "payload",
        [
            # Codex re-serializes reasoning content parts as `text`.
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "content": [{"type": "text", "text": "thinking"}],
            },
            # Codex may omit `summary` and null the item id.
            {
                "type": "reasoning",
                "id": None,
                "content": [{"type": "reasoning_text", "text": "thinking"}],
            },
        ],
    )
    async def test_codex_style_reasoning_item_maps(
        self, payload: dict[str, object]
    ) -> None:
        """Codex-serialized reasoning items parse and map to Bedrock blocks."""
        item = _parse(payload)
        assert isinstance(item, ResponseReasoningItemInput)
        messages, _ = await map_input(cast("list[ResponseInputItem]", [item]), None)
        assert messages == [
            {
                "role": "assistant",
                "content": [
                    {"reasoningContent": {"reasoningText": {"text": "thinking"}}}
                ],
            }
        ]


class TestFunctionCallOutputContentParts:
    """function_call_output part lists keep image and file parts."""

    async def test_image_part_maps_to_tool_result_image_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An all-image output produces a non-empty toolResult image block."""
        image_block: dict[str, object] = {
            "image": {"format": "png", "source": {"bytes": b"img"}}
        }

        class _StubInputFile:
            def __init__(self, source: str) -> None:
                self.source = source

            async def to_bedrock_content_block(self) -> dict[str, object]:
                return image_block

        monkeypatch.setattr(adapter, "InputFile", _StubInputFile)
        item = FunctionCallOutput(
            type="function_call_output",
            call_id="call_1",
            output=[
                ResponseInputImage(type="input_image", image_url="https://x/i.png")
            ],
        )
        messages, _ = await map_input([item], None)
        tool_result = messages[0]["content"][0]["toolResult"]
        assert tool_result == {"toolUseId": "call_1", "content": [image_block]}

    async def test_file_and_text_parts_map_to_document_and_text_blocks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """File parts become document blocks alongside text parts."""
        document_block: dict[str, object] = {
            "document": {"format": "pdf", "name": "doc", "source": {"bytes": b"pdf"}}
        }

        class _StubInputFile:
            def __init__(self, source: str) -> None:
                self.source = source

            async def to_bedrock_content_block(self) -> dict[str, object]:
                return document_block

        monkeypatch.setattr(adapter, "InputFile", _StubInputFile)
        item = FunctionCallOutput(
            type="function_call_output",
            call_id="call_1",
            output=[
                ResponseInputText(type="input_text", text="see attachment"),
                ResponseInputFile(type="input_file", file_data="JVBER", filename="d"),
            ],
        )
        messages, _ = await map_input([item], None)
        tool_result = messages[0]["content"][0]["toolResult"]
        assert tool_result["content"] == [{"text": "see attachment"}, document_block]


class TestFunctionCallArguments:
    """function_call arguments always produce a JSON object toolUse input."""

    async def test_invalid_json_arguments_are_wrapped(self) -> None:
        """Non-JSON arguments fall back to {"input": <raw>}."""
        item = FunctionCallInput(
            type="function_call",
            call_id="call_1",
            name="fn",
            arguments="run all the tests",
        )
        messages, _ = await map_input([item], None)
        tool_use = messages[0]["content"][0]["toolUse"]
        assert tool_use["input"] == {"input": "run all the tests"}

    async def test_json_object_arguments_pass_through(self) -> None:
        """Valid JSON object arguments are used as-is."""
        item = FunctionCallInput(
            type="function_call", call_id="call_1", name="fn", arguments='{"a": 1}'
        )
        messages, _ = await map_input([item], None)
        assert messages[0]["content"][0]["toolUse"]["input"] == {"a": 1}


class TestImageGenerationCallInput:
    """Echoed image_generation_call items replay the tool call and its image."""

    async def test_result_maps_to_tool_use_and_image_tool_result(self) -> None:
        """A completed call becomes toolUse plus a toolResult image block."""
        item = ImageGenerationCallInput(
            id="ig_1",
            status="completed",
            type="image_generation_call",
            result=b64encode(_PNG_BYTES).decode(),
        )
        messages, _ = await map_input([item], None)
        assert messages == [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "ig_1",
                            "name": "image_generation",
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
                            "toolUseId": "ig_1",
                            "content": [
                                {
                                    "image": {
                                        "format": "png",
                                        "source": {"bytes": _PNG_BYTES},
                                    }
                                }
                            ],
                        }
                    }
                ],
            },
        ]

    async def test_jpeg_magic_bytes_are_detected(self) -> None:
        """Non-PNG results are labeled with their sniffed format."""
        item = ImageGenerationCallInput(
            id="ig_1",
            status="completed",
            type="image_generation_call",
            result=b64encode(_JPEG_BYTES).decode(),
        )
        messages, _ = await map_input([item], None)
        tool_result = messages[1]["content"][0]["toolResult"]
        assert tool_result["content"][0]["image"]["format"] == "jpeg"

    async def test_empty_result_is_dropped(self) -> None:
        """Items without a result produce no Bedrock message."""
        item = ImageGenerationCallInput(
            id="ig_1", status="failed", type="image_generation_call", result=None
        )
        messages, _ = await map_input([item], None)
        assert messages == []


class TestSystemMessageContentParts:
    """System/developer list content keeps input_text and output_text parts."""

    @pytest.mark.parametrize("role", ["system", "developer"])
    async def test_output_text_parts_are_included(self, role: str) -> None:
        """output_text parts join input_text parts in the system blocks."""
        item = _parse(
            {
                "type": "message",
                "role": role,
                "content": [
                    {"type": "input_text", "text": "be brief"},
                    {"type": "output_text", "text": "and kind", "annotations": []},
                ],
            }
        )
        messages, system = await map_input(
            cast("list[ResponseInputItem]", [item]), None
        )
        assert messages == []
        assert system == [{"text": "be brief and kind"}]


class TestCountInputTokensToolConfig:
    """count_input_tokens_via_bedrock mirrors the real converse toolConfig."""

    @staticmethod
    def _stub_client(captured: dict[str, Any]) -> object:
        """Build a fake Bedrock runtime client recording count_tokens kwargs."""

        class _StubClient:
            async def count_tokens(self, **kwargs: object) -> dict[str, int]:
                captured.update(kwargs)
                return {"inputTokens": 7}

        return _StubClient()

    async def test_synthetic_image_generation_tool_is_counted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The toolConfig includes function and synthetic image_generation tools."""
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            adapter, "get_client", lambda *_args, **_kwargs: self._stub_client(captured)
        )
        request = InputTokenCountParams.model_validate(
            {
                "model": "m",
                "input": "hi",
                "tools": [
                    {"type": "function", "name": "fn", "parameters": None},
                    {"type": "image_generation"},
                ],
            }
        )
        count = await adapter.count_input_tokens_via_bedrock(
            request, "model-id", "us-east-1"
        )
        assert count == 7
        tool_config = captured["input"]["converse"]["toolConfig"]
        names = [tool["toolSpec"]["name"] for tool in tool_config["tools"]]
        assert names == ["fn", "image_generation"]

    async def test_tool_choice_none_omits_tool_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """tool_choice="none" keeps the toolConfig out of the count request."""
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            adapter, "get_client", lambda *_args, **_kwargs: self._stub_client(captured)
        )
        request = InputTokenCountParams.model_validate(
            {
                "model": "m",
                "input": "hi",
                "tool_choice": "none",
                "tools": [{"type": "function", "name": "fn", "parameters": None}],
            }
        )
        assert (
            await adapter.count_input_tokens_via_bedrock(
                request, "model-id", "us-east-1"
            )
            == 7
        )
        assert "toolConfig" not in captured["input"]["converse"]


class TestRefusalParts:
    """Assistant refusal parts are preserved as assistant text."""

    async def test_refusal_maps_to_assistant_text(self) -> None:
        """A refusal-only echoed message keeps the refusal text."""
        item = ResponseOutputMessage(
            id="msg_1",
            content=[
                ResponseOutputRefusal(
                    refusal="I cannot help with that.", type="refusal"
                )
            ],
            role="assistant",
            status="completed",
            type="message",
        )
        messages, _ = await map_input(cast("list[ResponseInputItem]", [item]), None)
        assert messages == [
            {"role": "assistant", "content": [{"text": "I cannot help with that."}]}
        ]


class TestExtractReasoning:
    """Reasoning extraction honors the upstream default effort."""

    def test_reasoning_without_effort_defaults_to_medium(self) -> None:
        """A reasoning object without effort enables medium-effort reasoning."""
        request = ResponseCreateParams(model="m", input="hi", reasoning=Reasoning())
        assert extract_reasoning(request) == {
            "enabled": True,
            "reasoning_effort": "medium",
            "budget_tokens": None,
            "max_tokens": None,
        }

    def test_effort_none_disables_reasoning(self) -> None:
        """effort="none" still disables reasoning."""
        request = ResponseCreateParams(
            model="m", input="hi", reasoning=Reasoning(effort="none")
        )
        params = extract_reasoning(request)
        assert params is not None
        assert params["enabled"] is False

    def test_no_reasoning_returns_none(self) -> None:
        """A request without a reasoning object configures nothing."""
        assert extract_reasoning(ResponseCreateParams(model="m", input="hi")) is None


class TestToolChoiceAllowedTools:
    """allowed_tools and type-variant tool choices map to Bedrock equivalents."""

    def test_required_with_single_function_forces_that_tool(self) -> None:
        """Required + one allowed function tool maps to a named tool choice."""
        choice = ToolChoiceAllowed(
            type="allowed_tools",
            mode="required",
            tools=[{"type": "function", "name": "get_weather"}],
        )
        assert _map_tool_choice(choice) == {"tool": {"name": "get_weather"}}

    def test_required_with_several_functions_forces_any_tool(self) -> None:
        """Required + several allowed function tools maps to any."""
        choice = ToolChoiceAllowed(
            type="allowed_tools",
            mode="required",
            tools=[
                {"type": "function", "name": "get_weather"},
                {"type": "function", "name": "get_time"},
            ],
        )
        assert _map_tool_choice(choice) == {"any": {}}

    def test_required_without_function_entries_forces_any_tool(self) -> None:
        """Required with only non-function entries still maps to any."""
        choice = ToolChoiceAllowed(
            type="allowed_tools",
            mode="required",
            tools=[{"type": "mcp", "server_label": "srv"}],
        )
        assert _map_tool_choice(choice) == {"any": {}}

    def test_auto_mode_maps_to_auto(self) -> None:
        """allowed_tools with mode auto maps to Bedrock auto."""
        choice = ToolChoiceAllowed(
            type="allowed_tools",
            mode="auto",
            tools=[{"type": "function", "name": "get_weather"}],
        )
        assert _map_tool_choice(choice) == {"auto": {}}

    def test_builtin_type_variant_maps_to_no_constraint(self) -> None:
        """Built-in tool type variants remain unconstrained (auto behavior)."""
        assert _map_tool_choice(ToolChoiceTypes(type="file_search")) is None


class TestReasoningSummarySignatures:
    """Envelope signatures never attach to summary fallback texts."""

    async def test_signature_not_attached_to_summary_text(self) -> None:
        """Summary fallback texts map without the envelope signature."""
        item = ResponseReasoningItem(
            id="rs_1",
            summary=[ReasoningItemSummary(text="sum", type="summary_text")],
            type="reasoning",
            encrypted_content=encode_reasoning_content(["sig-1"], []),
        )
        messages, _ = await map_input(cast("list[ResponseInputItem]", [item]), None)
        assert messages[0]["content"] == [
            {"reasoningContent": {"reasoningText": {"text": "sum"}}}
        ]

    async def test_redacted_blocks_survive_summary_fallback(self) -> None:
        """Redacted payloads are kept even when signatures are discarded."""
        item = ResponseReasoningItem(
            id="rs_1",
            summary=[ReasoningItemSummary(text="sum", type="summary_text")],
            type="reasoning",
            encrypted_content=encode_reasoning_content(["sig-1"], [b"\x02"]),
        )
        messages, _ = await map_input(cast("list[ResponseInputItem]", [item]), None)
        assert messages[0]["content"] == [
            {"reasoningContent": {"reasoningText": {"text": "sum"}}},
            {"reasoningContent": {"redactedContent": b"\x02"}},
        ]
