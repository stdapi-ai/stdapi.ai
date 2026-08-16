"""Responses API ``input`` items and their mapping onto Bedrock Converse messages.

Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
     stdapi/models/chat/_adapters/_openai_responses.py:map_input
"""

from base64 import b64encode
from typing import Any, cast

import pytest
from pydantic import TypeAdapter

from stdapi.api_errors import ApiError
from stdapi.models.chat._adapters import _openai_responses as adapter
from stdapi.models.chat._adapters._openai_responses import (
    _map_tool_choice,
    encode_reasoning_content,
    extract_reasoning,
    map_input,
)
from stdapi.models.chat._default import ChatModel
from stdapi.types.openai_responses import (
    ApplyPatchCall,
    ApplyPatchCallOutput,
    CodeInterpreterCallInput,
    CompactionTrigger,
    ComputerCallInput,
    ComputerCallOutput,
    CustomToolCallInput,
    CustomToolCallOutput,
    FileSearchCallInput,
    FunctionCallInput,
    FunctionCallOutput,
    ImageGenerationCallInput,
    InputTokenCountParams,
    ItemReference,
    LocalShellCallInput,
    LocalShellCallOutputInput,
    McpApprovalRequestInput,
    McpApprovalResponse,
    McpCallInput,
    McpListToolsInput,
    Reasoning,
    ReasoningItemContent,
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
    ShellCall,
    ShellCallOutput,
    ToolChoiceAllowed,
    ToolChoiceTypes,
    ToolSearchCallInput,
    ToolSearchOutputInput,
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


def _stub_input_file(
    monkeypatch: pytest.MonkeyPatch,
    block: dict[str, object],
    attribute: str = "InputFile",
) -> list[str]:
    """Replace an adapter file loader with a stub returning a fixed Bedrock block.

    Args:
        monkeypatch: Patcher applied to the adapter module.
        block: Bedrock content block the stub resolves to.
        attribute: Loader to replace, ``InputFile`` or ``FileIdInputFile``.

    Returns:
        The mutable list recording every source handed to that loader.
    """
    sources: list[str] = []

    class _StubInputFile:
        def __init__(self, source: str) -> None:
            self.source = source
            sources.append(source)

        async def to_bedrock_content_block(self) -> dict[str, object]:
            return block

    monkeypatch.setattr(adapter, attribute, _StubInputFile)
    return sources


@pytest.fixture
def captured_count_tokens(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch the Bedrock runtime client used for CountTokens with a recorder.

    Returns:
        The mutable dict receiving the ``count_tokens`` keyword arguments; the
        stub always reports seven input tokens.
    """
    captured: dict[str, Any] = {}

    class _StubClient:
        async def count_tokens(self, **kwargs: object) -> dict[str, int]:
            captured.update(kwargs)
            return {"inputTokens": 7}

    monkeypatch.setattr(adapter, "get_client", lambda *_args, **_kwargs: _StubClient())
    return captured


class TestHostedToolItemsParseAndDrop:
    """Items with no Bedrock equivalent validate and contribute no request content.

    Upstream accepts a broad item union in ``input`` (hosted-tool calls, shell,
    MCP, apply_patch).  Bedrock Converse can represent none of them, so
    ``_map_input_item`` has no branch for these types and drops them silently
    rather than rejecting a client that replays a whole previous response.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         https://developers.openai.com/api/docs/guides/tools
         stdapi/models/chat/_adapters/_openai_responses.py:_map_input_item
    """

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
            (
                {
                    "type": "tool_search_call",
                    "id": "tsc_1",
                    "arguments": {"query": "cats"},
                },
                ToolSearchCallInput,
            ),
            (
                {
                    "type": "tool_search_output",
                    "id": "tso_1",
                    "call_id": "call_1",
                    "execution": "server",
                    "status": "completed",
                    "tools": [{"type": "function", "name": "search"}],
                },
                ToolSearchOutputInput,
            ),
            (
                {
                    "type": "local_shell_call",
                    "id": "lsc_1",
                    "call_id": "call_1",
                    "status": "completed",
                    "action": {"type": "exec", "command": ["ls"], "env": {}},
                },
                LocalShellCallInput,
            ),
            (
                {"type": "local_shell_call_output", "id": "lsc_1", "output": "{}"},
                LocalShellCallOutputInput,
            ),
            (
                {
                    "type": "shell_call",
                    "call_id": "call_1",
                    "action": {"commands": ["ls"]},
                },
                ShellCall,
            ),
            (
                {
                    "type": "shell_call_output",
                    "call_id": "call_1",
                    "output": [
                        {
                            "outcome": {"type": "exit", "exit_code": 0},
                            "stderr": "",
                            "stdout": "ok",
                        }
                    ],
                },
                ShellCallOutput,
            ),
            (
                {
                    "type": "apply_patch_call",
                    "call_id": "call_1",
                    "status": "completed",
                    "operation": {
                        "type": "create_file",
                        "path": "a.txt",
                        "diff": "diff",
                    },
                },
                ApplyPatchCall,
            ),
            (
                {
                    "type": "apply_patch_call_output",
                    "call_id": "call_1",
                    "status": "completed",
                },
                ApplyPatchCallOutput,
            ),
            (
                {
                    "type": "mcp_call",
                    "id": "mcp_1",
                    "arguments": "{}",
                    "name": "tool",
                    "server_label": "srv",
                },
                McpCallInput,
            ),
            (
                {
                    "type": "mcp_list_tools",
                    "id": "mlt_1",
                    "server_label": "srv",
                    "tools": [{"name": "tool", "input_schema": {}}],
                },
                McpListToolsInput,
            ),
            (
                {
                    "type": "mcp_approval_request",
                    "id": "mar_1",
                    "arguments": "{}",
                    "name": "tool",
                    "server_label": "srv",
                },
                McpApprovalRequestInput,
            ),
            (
                {
                    "type": "computer_call_output",
                    "call_id": "call_1",
                    "output": {
                        "type": "computer_screenshot",
                        "image_url": "https://example.com/a.png",
                    },
                },
                ComputerCallOutput,
            ),
            (
                {
                    "type": "mcp_approval_response",
                    "approval_request_id": "mar_1",
                    "approve": True,
                },
                McpApprovalResponse,
            ),
            ({"type": "item_reference", "id": "ref_1"}, ItemReference),
        ],
    )
    async def test_parses_and_is_dropped(
        self, payload: dict[str, object], expected_cls: type
    ) -> None:
        """Each item resolves to its own union member and yields no Bedrock content."""
        item = _parse(payload)
        assert isinstance(item, expected_cls)
        assert getattr(item, "type", None) == payload["type"], (
            "the discriminated union must resolve on the item's own type literal"
        )
        messages, system = await map_input(
            cast("list[ResponseInputItem]", [item]), None
        )
        assert messages == []
        assert system == []


class TestFileSearchCallItems:
    """A replayed ``file_search_call`` carries its passages back to the model.

    The gateway runs the search itself, so the item is the only record of what
    was retrieved: it is replayed as the ``toolUse`` the model made and the
    ``toolResult`` that answered it. An item with no results carries nothing
    the model could read, and a ``toolUse`` without its result is not a valid
    conversation, so both are dropped together.

    Ref: https://developers.openai.com/api/docs/guides/tools-file-search
         stdapi/models/chat/_adapters/_openai_responses.py:_map_file_search_call
    """

    async def test_an_answered_call_is_replayed_with_its_passages(self) -> None:
        """The call becomes a toolUse and its passages the matching toolResult."""
        item = _parse(
            {
                "id": "fs_1",
                "type": "file_search_call",
                "status": "completed",
                "queries": ["cats"],
                "results": [{"text": "meow"}],
            }
        )
        assert isinstance(item, FileSearchCallInput)

        messages, _system = await map_input(
            cast("list[ResponseInputItem]", [item]), None
        )

        assert [message["role"] for message in messages] == ["assistant", "user"]
        tool_use = messages[0]["content"][0]["toolUse"]
        assert tool_use == {
            "toolUseId": "fs_1",
            "name": "file_search",
            "input": {"query": "cats"},
        }
        tool_result = messages[1]["content"][0]["toolResult"]
        assert tool_result["toolUseId"] == "fs_1"
        assert tool_result["content"] == [{"text": "meow"}]

    async def test_a_call_without_results_is_dropped(self) -> None:
        """An item whose results were never included leaves no dangling toolUse."""
        item = _parse(
            {
                "id": "fs_1",
                "type": "file_search_call",
                "status": "completed",
                "queries": ["cats"],
            }
        )

        messages, _system = await map_input(
            cast("list[ResponseInputItem]", [item]), None
        )

        assert messages == []


class TestCustomToolCallItems:
    """``custom_tool_call``/``custom_tool_call_output`` map to toolUse/toolResult.

    Bedrock has no free-form tool payload, so a custom tool call is replayed as
    a regular ``toolUse`` whose ``input`` is always a JSON object, and its
    output as a ``toolResult`` carried by a user turn.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/models/chat/_adapters/_openai_responses.py:_map_custom_tool_call
         stdapi/models/chat/_adapters/_openai_responses.py:_map_custom_tool_call_output
    """

    async def test_call_with_json_object_input(self) -> None:
        """A JSON object ``input`` becomes the Bedrock toolUse input verbatim."""
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
        """Non-JSON freeform ``input`` is wrapped as ``{"input": <raw>}``.

        Bedrock's ``toolUse.input`` must be a JSON object, so a free-form custom
        tool payload cannot be forwarded as a bare string.
        """
        item = CustomToolCallInput(
            type="custom_tool_call",
            call_id="call_1",
            name="sql",
            input="SELECT * FROM cats",
        )
        messages, _ = await map_input([item], None)
        assert messages[0]["role"] == "assistant"
        tool_use = messages[0]["content"][0]["toolUse"]
        assert tool_use["input"] == {"input": "SELECT * FROM cats"}
        assert tool_use["toolUseId"] == "call_1"
        assert tool_use["name"] == "sql"

    async def test_output_string_form(self) -> None:
        """A string ``output`` becomes a user toolResult with a single text block."""
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
        """A content-part list ``output`` keeps one text block per text part."""
        item = CustomToolCallOutput(
            type="custom_tool_call_output",
            call_id="call_1",
            output=[
                ResponseInputText(type="input_text", text="a"),
                ResponseInputText(type="input_text", text="b"),
            ],
        )
        messages, _ = await map_input([item], None)
        assert messages[0]["role"] == "user"
        tool_result = messages[0]["content"][0]["toolResult"]
        assert tool_result["content"] == [{"text": "a"}, {"text": "b"}]
        assert tool_result["toolUseId"] == "call_1"


class TestEchoedItemsTolerateUnknownFields:
    """Echoed output items parse even when carrying unknown upstream fields.

    OpenAI reserves the right to add response fields, and clients replay whole
    output items back into ``input``.  The ``*Input`` item models therefore set
    ``extra="ignore"`` so an unknown field is dropped instead of failing
    validation, even though the suite runs with ``strict_input_validation``.

    Ref: https://developers.openai.com/api/reference/overview
         stdapi/types/openai_responses.py:ResponseOutputMessageInput
         stdapi/types/openai_responses.py:ResponseReasoningItemInput
    """

    async def test_output_message_with_unknown_field(self) -> None:
        """An echoed assistant message with an unknown field parses and still maps."""
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
        assert "brand_new_upstream_field" not in item.model_dump(), (
            "unknown fields must be dropped, not echoed back into the model"
        )
        messages, _ = await map_input(cast("list[ResponseInputItem]", [item]), None)
        assert messages == [{"role": "assistant", "content": [{"text": "hi"}]}]

    async def test_reasoning_item_with_unknown_field(self) -> None:
        """An unknown field on a reasoning item parses and leaves it empty.

        The item carries neither ``content`` nor ``summary``, so it maps to no
        Bedrock block at all.
        """
        item = _parse(
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [],
                "brand_new_upstream_field": 1,
            }
        )
        assert isinstance(item, ResponseReasoningItemInput)
        assert "brand_new_upstream_field" not in item.model_dump()
        messages, _ = await map_input(cast("list[ResponseInputItem]", [item]), None)
        assert messages == []

    @pytest.mark.parametrize(
        "payload",
        [
            # Shorthand form: content as a bare string.
            {"type": "message", "role": "user", "id": "msg_1", "content": "hi"},
            # Full form, with the status the API reports alongside the id.
            {
                "type": "message",
                "role": "user",
                "id": "msg_1",
                "status": "completed",
                "content": [{"type": "input_text", "text": "hi"}],
            },
        ],
    )
    async def test_message_item_keeps_its_echoed_id(
        self, payload: dict[str, object]
    ) -> None:
        """A message item replayed with the ``id`` the API gave it still parses.

        Every message the API hands back carries an ``id``; a client replaying a
        listed item sends it straight back. Codex does so from its very first
        request, so rejecting the field fails the whole turn.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/list_input_items
             stdapi/types/openai_responses.py:EasyInputMessage
        """
        item = _parse(payload)
        messages, _ = await map_input(cast("list[ResponseInputItem]", [item]), None)
        assert messages == [{"role": "user", "content": [{"text": "hi"}]}]

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
        """Codex-serialized reasoning items map to a Bedrock reasoningText block.

        Codex CLI re-serializes reasoning content parts with ``type: "text"``
        instead of ``reasoning_text``, may omit ``summary`` and may null the item
        ``id``; all three variants must still reach the model as reasoning.

        Ref: stdapi/types/openai_responses.py:ReasoningItemContentInput
             stdapi/models/chat/_adapters/_openai_responses.py:_map_reasoning_item
        """
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
    """A ``function_call_output`` part list keeps its image and file parts.

    Upstream allows a function result to be a content-part list, and Bedrock
    ``toolResult`` content blocks share their shape with message content blocks,
    so image and document parts are resolved through the same input-file
    machinery as user content instead of being flattened to text.

    Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
         https://developers.openai.com/api/docs/guides/file-inputs
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
         stdapi/models/chat/_adapters/_openai_responses.py:_map_function_call_output
    """

    async def test_image_part_maps_to_tool_result_image_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ``input_image`` output part resolves to a toolResult image block.

        ``InputFile`` is stubbed so the test pins the routing of the part's
        ``image_url`` into the file loader, not the loader's own fetching.
        """
        image_block: dict[str, object] = {
            "image": {"format": "png", "source": {"bytes": b"img"}}
        }
        sources = _stub_input_file(monkeypatch, image_block)
        item = FunctionCallOutput(
            type="function_call_output",
            call_id="call_1",
            output=[
                ResponseInputImage(type="input_image", image_url="https://x/i.png")
            ],
        )
        messages, _ = await map_input([item], None)
        assert messages[0]["role"] == "user", "toolResult must ride a user turn"
        tool_result = messages[0]["content"][0]["toolResult"]
        assert tool_result == {"toolUseId": "call_1", "content": [image_block]}
        assert sources == ["https://x/i.png"]

    async def test_file_and_text_parts_map_to_document_and_text_blocks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ``input_file`` part becomes a document block, keeping part order."""
        document_block: dict[str, object] = {
            "document": {"format": "pdf", "name": "doc", "source": {"bytes": b"pdf"}}
        }
        sources = _stub_input_file(monkeypatch, document_block)
        item = FunctionCallOutput(
            type="function_call_output",
            call_id="call_1",
            output=[
                ResponseInputText(type="input_text", text="see attachment"),
                ResponseInputFile(type="input_file", file_data="JVBER", filename="d"),
            ],
        )
        messages, _ = await map_input([item], None)
        assert messages[0]["role"] == "user"
        tool_result = messages[0]["content"][0]["toolResult"]
        assert tool_result["content"] == [{"text": "see attachment"}, document_block]
        assert tool_result["toolUseId"] == "call_1"
        assert sources == ["JVBER"], "file_data is handed to the input-file loader"


class TestFunctionCallArguments:
    """A replayed ``function_call`` always produces a JSON object toolUse input.

    Bedrock rejects a non-object ``toolUse.input``, while upstream types
    ``arguments`` as an opaque JSON string that a client may echo back verbatim.

    Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
         stdapi/models/chat/_adapters/_openai_responses.py:_tool_use_input
    """

    async def test_invalid_json_arguments_are_wrapped(self) -> None:
        """Non-JSON ``arguments`` fall back to ``{"input": <raw>}``."""
        item = FunctionCallInput(
            type="function_call",
            call_id="call_1",
            name="fn",
            arguments="run all the tests",
        )
        messages, _ = await map_input([item], None)
        assert messages[0]["role"] == "assistant"
        tool_use = messages[0]["content"][0]["toolUse"]
        assert tool_use["input"] == {"input": "run all the tests"}
        assert tool_use["toolUseId"] == "call_1"
        assert tool_use["name"] == "fn"

    async def test_json_object_arguments_pass_through(self) -> None:
        """A JSON object in ``arguments`` is used as the toolUse input unchanged."""
        item = FunctionCallInput(
            type="function_call", call_id="call_1", name="fn", arguments='{"a": 1}'
        )
        messages, _ = await map_input([item], None)
        tool_use = messages[0]["content"][0]["toolUse"]
        assert tool_use["input"] == {"a": 1}
        assert tool_use["name"] == "fn"


class TestImageGenerationCallInput:
    """An echoed ``image_generation_call`` replays the tool call and its image.

    ``image_generation`` is not a Bedrock hosted tool: the gateway exposes it to
    the model as a synthetic function tool and generates the image itself, so a
    replayed call has to be rebuilt as an assistant ``toolUse`` plus the user
    ``toolResult`` that carries the decoded image.

    Ref: https://developers.openai.com/api/docs/guides/tools-image-generation
         stdapi/models/chat/_adapters/_openai_responses.py:_map_image_generation_call
    """

    async def test_result_maps_to_tool_use_and_image_tool_result(self) -> None:
        """A completed call becomes a toolUse plus a toolResult image block."""
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
        """A JPEG result is labeled ``jpeg`` rather than the ``png`` fallback.

        The item carries no MIME type, so the Bedrock image format is sniffed
        from the decoded payload's magic bytes.
        """
        item = ImageGenerationCallInput(
            id="ig_1",
            status="completed",
            type="image_generation_call",
            result=b64encode(_JPEG_BYTES).decode(),
        )
        messages, _ = await map_input([item], None)
        tool_result = messages[1]["content"][0]["toolResult"]
        assert tool_result["content"][0]["image"] == {
            "format": "jpeg",
            "source": {"bytes": _JPEG_BYTES},
        }

    async def test_empty_result_is_dropped(self) -> None:
        """A call without a ``result`` produces no Bedrock message at all.

        A failed generation must not leave a dangling ``toolUse`` with no
        matching ``toolResult``, which Bedrock rejects.
        """
        item = ImageGenerationCallInput(
            id="ig_1", status="failed", type="image_generation_call", result=None
        )
        messages, _ = await map_input([item], None)
        assert messages == []


class TestSystemMessageContentParts:
    """``system``/``developer`` list content becomes Bedrock system blocks.

    Upstream states that instructions given with the ``developer`` or ``system``
    role take precedence over the ``user`` role, so both roles are lifted out of
    the message list into Converse ``system`` blocks; their ``input_text`` and
    ``output_text`` parts each become one system block, matching the Chat
    Completions adapter.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
         stdapi/models/chat/_adapters/_openai_responses.py:_map_message_item
    """

    @pytest.mark.parametrize("role", ["system", "developer"])
    async def test_output_text_parts_are_included(self, role: str) -> None:
        """``output_text`` parts join ``input_text`` parts, one system block each."""
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
        assert system == [{"text": "be brief"}, {"text": "and kind"}]

    @pytest.mark.parametrize("role", ["system", "developer"])
    async def test_plain_string_content_becomes_a_system_block(self, role: str) -> None:
        """A string ``content`` is lifted just like the list form, not left as a turn.

        The string shorthand is the common spelling for instructions; leaving it
        in the message list would make it a user-precedence turn.
        """
        item = _parse({"type": "message", "role": role, "content": "be brief"})
        messages, system = await map_input(
            cast("list[ResponseInputItem]", [item]), None
        )
        assert messages == []
        assert system == [{"text": "be brief"}]


class TestSystemBreakpointCachePoints:
    """System-part cache breakpoints never lead with nor repeat a cache point.

    Mirrors the Chat Completions guard: an empty part yields no text block, so
    its breakpoint must not emit a leading cache point, and a breakpoint right
    after another must not emit consecutive cache points, which Bedrock
    rejects.

    Ref: https://developers.openai.com/api/docs/guides/prompt-caching
         stdapi/models/chat/_adapters/_openai_chat_completion.py:_extract_system_content_blocks
         stdapi/models/chat/_adapters/_openai_responses.py:_map_message_item
    """

    async def test_leading_and_repeated_cache_points_are_suppressed(self) -> None:
        """Only the text-backed breakpoint emits a cache point."""
        breakpoint_ = {"mode": "explicit"}
        item = _parse(
            {
                "type": "message",
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "",
                        "prompt_cache_breakpoint": breakpoint_,
                    },
                    {
                        "type": "input_text",
                        "text": "cached prefix",
                        "prompt_cache_breakpoint": breakpoint_,
                    },
                    {
                        "type": "input_text",
                        "text": "",
                        "prompt_cache_breakpoint": breakpoint_,
                    },
                ],
            }
        )
        messages, system = await map_input(
            cast("list[ResponseInputItem]", [item]), None, allow_explicit_caching=True
        )
        assert messages == []
        assert system == [
            {"text": "cached prefix"},
            {"cachePoint": {"type": "default"}},
        ]


class TestCountInputTokensToolConfig:
    """POST /responses/input_tokens counts the same toolConfig Converse would get.

    Tool definitions are billable input, and Bedrock ``CountTokens`` returns the
    count that the equivalent ``Converse`` call would be charged, so the route
    reuses ``_build_tool_config`` verbatim instead of estimating.

    Ref: https://developers.openai.com/api/reference/resources/responses/subresources/input_tokens
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html
         stdapi/models/chat/_adapters/_openai_responses.py:count_input_tokens_via_bedrock
    """

    async def test_synthetic_image_generation_tool_is_counted(
        self, captured_count_tokens: dict[str, Any]
    ) -> None:
        """The counted toolConfig holds the function and synthetic image tools.

        ``image_generation`` is executed gateway-side but is still presented to
        the model as a function tool, so its schema is part of the billed input.
        """
        captured = captured_count_tokens
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
            request, "model-id", "us-east-1", ChatModel("model-id")
        )
        assert count == 7, "the route returns Bedrock's inputTokens unmodified"
        assert captured["modelId"] == "model-id"
        converse = captured["input"]["converse"]
        assert converse["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]
        tool_config = converse["toolConfig"]
        names = [tool["toolSpec"]["name"] for tool in tool_config["tools"]]
        assert names == ["fn", "image_generation"]
        synthetic = tool_config["tools"][1]["toolSpec"]["inputSchema"]["json"]
        assert synthetic["required"] == ["prompt"], (
            "the synthetic tool must be counted with its real schema"
        )

    @pytest.mark.usefixtures("captured_count_tokens")
    async def test_reasoning_config_reaches_the_model_hook(self) -> None:
        """A ``reasoning`` request calls the model's reasoning hook with its effort.

        Each model builds its own reasoning fields in that hook, so the counted
        request only matches the one the model receives if the adapter hands the
        requested effort over; the resulting fields are the model's business.

        Ref: stdapi/models/chat/_adapters/_openai_responses.py:count_input_tokens_via_bedrock
             stdapi/models/chat/_default.py:ChatModel._req_configure_reasoning
        """
        received: dict[str, Any] = {}

        class _ReasoningChatModel(ChatModel):
            """Model whose reasoning hook records the arguments it was handed."""

            def _req_configure_reasoning(
                self,
                additional_request_fields: dict[str, Any],
                *,
                enabled: bool,
                reasoning_effort: str | None = None,
                budget_tokens: int | None = None,
                max_tokens: int | None = None,
            ) -> None:
                """Record the reasoning arguments without writing any request field."""
                received.update(
                    enabled=enabled,
                    reasoning_effort=reasoning_effort,
                    budget_tokens=budget_tokens,
                    max_tokens=max_tokens,
                )

        request = InputTokenCountParams.model_validate(
            {"model": "m", "input": "hi", "reasoning": {"effort": "high"}}
        )
        assert (
            await adapter.count_input_tokens_via_bedrock(
                request, "model-id", "us-east-1", _ReasoningChatModel("model-id")
            )
            == 7
        )
        assert received == {
            "enabled": True,
            "reasoning_effort": "high",
            "budget_tokens": None,
            "max_tokens": None,
        }

    async def test_no_reasoning_leaves_the_request_bare(
        self, captured_count_tokens: dict[str, Any]
    ) -> None:
        """Without ``reasoning`` no additional field is sent, not an empty object.

        Bedrock validates ``additionalModelRequestFields`` against the model, so
        an always-present empty object would be a gratuitous divergence from the
        Converse request being counted.
        """
        captured = captured_count_tokens
        request = InputTokenCountParams.model_validate({"model": "m", "input": "hi"})
        assert (
            await adapter.count_input_tokens_via_bedrock(
                request, "model-id", "us-east-1", ChatModel("model-id")
            )
            == 7
        )
        assert "additionalModelRequestFields" not in captured["input"]["converse"]

    async def test_tool_choice_none_omits_tool_config(
        self, captured_count_tokens: dict[str, Any]
    ) -> None:
        """``tool_choice="none"`` keeps the whole toolConfig out of the count.

        The gateway implements ``none`` by omitting the Bedrock tool
        configuration, so the declared tools are not billed either.
        """
        captured = captured_count_tokens
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
                request, "model-id", "us-east-1", ChatModel("model-id")
            )
            == 7
        )
        assert "toolConfig" not in captured["input"]["converse"]
        assert captured["input"]["converse"]["messages"] == [
            {"role": "user", "content": [{"text": "hi"}]}
        ]

    async def test_tool_use_history_without_tools_synthesizes_a_tool_config(
        self, captured_count_tokens: dict[str, Any]
    ) -> None:
        """An echoed function_call with no ``tools`` still counts a toolConfig.

        Bedrock CountTokens rejects ``toolUse`` blocks without a ``toolConfig``
        just like Converse does, so the count path synthesizes the same
        permissive stub the create path falls back to instead of failing.
        """
        request = InputTokenCountParams.model_validate(
            {
                "model": "m",
                "input": [
                    {"role": "user", "content": "hi"},
                    {
                        "type": "function_call",
                        "call_id": "c1",
                        "name": "fn",
                        "arguments": "{}",
                    },
                    {"type": "function_call_output", "call_id": "c1", "output": "ok"},
                ],
            }
        )
        assert (
            await adapter.count_input_tokens_via_bedrock(
                request, "model-id", "us-east-1", ChatModel("model-id")
            )
            == 7
        )
        assert captured_count_tokens["input"]["converse"]["toolConfig"] == {
            "tools": [
                {
                    "toolSpec": {
                        "name": "fn",
                        "inputSchema": {"json": {"type": "object"}},
                    }
                }
            ]
        }

    async def test_image_and_file_parts_are_counted(
        self, captured_count_tokens: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``input_image`` and ``input_file`` parts reach the counted Converse request.

        Media dominates the token cost this route exists to predict, so the
        resolved image and document blocks must be part of what CountTokens
        sees; dropping them would under-report against the create call's bill.

        Ref: https://developers.openai.com/api/docs/guides/token-counting
             stdapi/models/chat/_adapters/_openai_responses.py:count_input_tokens_via_bedrock
        """
        media_block: dict[str, object] = {
            "image": {"format": "png", "source": {"bytes": _PNG_BYTES}}
        }
        sources = _stub_input_file(monkeypatch, media_block)
        request = InputTokenCountParams.model_validate(
            {
                "model": "m",
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "hi"},
                            {
                                "type": "input_image",
                                "image_url": "https://example.com/a.png",
                            },
                            {"type": "input_file", "file_url": "https://x/d.pdf"},
                        ],
                    }
                ],
            }
        )
        assert (
            await adapter.count_input_tokens_via_bedrock(
                request, "model-id", "us-east-1", ChatModel("model-id")
            )
            == 7
        )
        assert captured_count_tokens["input"]["converse"]["messages"] == [
            {"role": "user", "content": [{"text": "hi"}, media_block, media_block]}
        ]
        assert sources == ["https://example.com/a.png", "https://x/d.pdf"]


class TestRequestMetadata:
    """``metadata`` reaches Bedrock as the Converse ``requestMetadata`` mapping.

    The pairs are cost-attribution tags, so they must arrive verbatim: the
    response object echoing them proves nothing about what Converse received.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
         stdapi/models/chat/_adapters/_openai_responses.py:translate_request
    """

    def test_metadata_pairs_are_forwarded_verbatim(self) -> None:
        """Every key/value pair is handed over unchanged as ``request_metadata``."""
        metadata = {"session_id": "abc", "test_type": "automated"}
        request = ResponseCreateParams.model_validate(
            {"model": "m", "input": "hi", "metadata": metadata}
        )
        assert adapter.translate_request(request, "model-id")[-1] == metadata

    def test_absent_metadata_forwards_none(self) -> None:
        """No ``metadata`` yields ``None`` so the Converse key is omitted entirely.

        An empty mapping would still be sent and count against the Bedrock
        16-pair budget the gateway's own tracing keys share.
        """
        request = ResponseCreateParams.model_validate({"model": "m", "input": "hi"})
        assert adapter.translate_request(request, "model-id")[-1] is None


class TestInputFileAndImageSources:
    """Every documented ``input_file``/``input_image`` source reaches a file loader.

    Upstream accepts a Files API ``file_id``, base64 ``file_data`` or a
    ``file_url`` on ``input_file``, and a ``file_id`` or ``image_url`` on
    ``input_image``. Bedrock accepts inline bytes only, so each source is
    resolved before the Converse block is built: ``file_id`` through the
    S3-backed ``FileIdInputFile``, every other source through ``InputFile``.

    Ref: https://developers.openai.com/api/docs/guides/file-inputs
         stdapi/models/chat/_adapters/_openai_responses.py:_convert_input_content
    """

    @pytest.mark.parametrize(
        ("part", "loader"),
        [
            ({"type": "input_file", "file_id": "file-abc"}, "FileIdInputFile"),
            ({"type": "input_file", "file_url": "https://x/d.pdf"}, "InputFile"),
            ({"type": "input_file", "file_data": "JVBER"}, "InputFile"),
            ({"type": "input_image", "file_id": "file-abc"}, "FileIdInputFile"),
            ({"type": "input_image", "image_url": "https://x/i.png"}, "InputFile"),
        ],
        ids=[
            "file-file_id",
            "file-file_url",
            "file-file_data",
            "image-file_id",
            "image-image_url",
        ],
    )
    async def test_source_routes_to_its_loader(
        self, monkeypatch: pytest.MonkeyPatch, part: dict[str, str], loader: str
    ) -> None:
        """The part's source is handed to its loader and becomes the message block.

        Both loaders are stubbed, so the assertion pins the routing decision
        rather than S3 access or URL fetching.
        """
        block: dict[str, object] = {
            "document": {"format": "pdf", "name": "d", "source": {"bytes": b"pdf"}}
        }
        other = "InputFile" if loader == "FileIdInputFile" else "FileIdInputFile"
        used = _stub_input_file(monkeypatch, block, loader)
        unused = _stub_input_file(monkeypatch, block, other)
        item = _parse({"type": "message", "role": "user", "content": [part]})
        messages, system = await map_input(
            cast("list[ResponseInputItem]", [item]), None
        )
        assert system == []
        assert messages == [{"role": "user", "content": [block]}]
        assert used == [next(value for key, value in part.items() if key != "type")]
        assert unused == [], "the other loader must not be involved"

    async def test_file_id_takes_precedence_over_a_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A part carrying both ``file_id`` and ``file_url`` resolves the ``file_id``.

        The match arms are ordered, so the stored file wins over the URL rather
        than the block being built twice or the URL being fetched needlessly.
        """
        block: dict[str, object] = {
            "document": {"format": "pdf", "name": "d", "source": {"bytes": b"pdf"}}
        }
        by_id = _stub_input_file(monkeypatch, block, "FileIdInputFile")
        by_url = _stub_input_file(monkeypatch, block, "InputFile")
        item = _parse(
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_id": "file-abc",
                        "file_url": "https://x/d.pdf",
                    }
                ],
            }
        )
        messages, _ = await map_input(cast("list[ResponseInputItem]", [item]), None)
        assert messages == [{"role": "user", "content": [block]}]
        assert by_id == ["file-abc"]
        assert by_url == []


class TestRefusalParts:
    """An echoed ``refusal`` part is replayed as assistant text.

    Bedrock has no refusal content block, so dropping the part would erase a
    whole assistant turn from the replayed history and leave the conversation
    with two consecutive user turns.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/models/chat/_adapters/_openai_responses.py:_map_output_message
    """

    async def test_refusal_maps_to_assistant_text(self) -> None:
        """A refusal-only echoed message keeps its refusal text as assistant text."""
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


class TestEchoedAssistantContentShapes:
    """Every content shape an echoed assistant message may carry maps to text.

    ``ResponseOutputMessageInput`` widens ``content`` with the input part types,
    because clients relabel the parts they replay. The mapper must therefore
    accept each of them: a part shape it does not know is an unhandled attribute
    access, which surfaces to the client as an opaque 500 mid-conversation.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/models/chat/_adapters/_openai_responses.py:_output_message_text
         stdapi/types/openai_responses.py:ResponseOutputMessageInput
    """

    @staticmethod
    def _echo(part: dict[str, object]) -> dict[str, object]:
        """Return an echoed assistant message carrying a single content *part*."""
        return {
            "type": "message",
            "role": "assistant",
            "id": "msg_1",
            "status": "completed",
            "content": [part],
        }

    @pytest.mark.parametrize(
        "part",
        [
            pytest.param(
                {"type": "output_text", "text": "hi", "annotations": []},
                id="output-text-with-annotations",
            ),
            pytest.param({"type": "output_text", "text": "hi"}, id="output-text-bare"),
            pytest.param({"type": "input_text", "text": "hi"}, id="input-text"),
        ],
    )
    async def test_every_text_part_shape_maps_to_assistant_text(
        self, part: dict[str, object]
    ) -> None:
        """A text part maps to assistant text whichever shape the client replayed.

        ``output_text`` without ``annotations`` is the shape Hermes replays; it
        validates as ``ResponseOutputTextContent``, not ``ResponseOutputText``,
        so the mapper must not reach for a ``refusal`` attribute it lacks.
        """
        item = _parse(self._echo(part))
        assert isinstance(item, ResponseOutputMessageInput)
        messages, _ = await map_input(cast("list[ResponseInputItem]", [item]), None)
        assert messages == [{"role": "assistant", "content": [{"text": "hi"}]}]

    async def test_a_non_text_part_is_refused_rather_than_crashing(self) -> None:
        """An image part in an echoed assistant message raises a client error.

        Bedrock's assistant role carries no image, so the turn cannot be
        replayed; what matters is that it is refused as a request error instead
        of escaping as an unhandled exception.
        """
        item = _parse(
            self._echo(
                {"type": "input_image", "image_url": "https://example.com/a.png"}
            )
        )
        assert isinstance(item, ResponseOutputMessageInput)
        with pytest.raises(ApiError, match="input_image"):
            await map_input(cast("list[ResponseInputItem]", [item]), None)


class TestExtractReasoning:
    """The presence of ``reasoning`` decides whether reasoning is configured.

    Upstream documents the ``reasoning.effort`` default as model-dependent; this
    gateway pins its own default so that a bare ``reasoning: {}`` is actionable,
    and treats only ``effort="none"`` as a request to disable reasoning.

    Ref: https://developers.openai.com/api/docs/guides/reasoning#preserve-reasoning-without-stored-responses
         stdapi/models/chat/_adapters/_openai_responses.py:extract_reasoning
    """

    def test_reasoning_without_effort_defaults_to_medium(self) -> None:
        """A ``reasoning`` object without ``effort`` enables medium-effort reasoning."""
        request = ResponseCreateParams(model="m", input="hi", reasoning=Reasoning())
        assert extract_reasoning(request) == {
            "enabled": True,
            "reasoning_effort": "medium",
            "budget_tokens": None,
            "max_tokens": None,
        }

    def test_effort_none_disables_reasoning(self) -> None:
        """``effort="none"`` disables reasoning while keeping the effort value."""
        request = ResponseCreateParams(
            model="m", input="hi", reasoning=Reasoning(effort="none")
        )
        params = extract_reasoning(request)
        assert params is not None
        assert params["enabled"] is False
        assert params["reasoning_effort"] == "none"

    def test_no_reasoning_returns_none(self) -> None:
        """A request without a ``reasoning`` object configures nothing at all.

        ``None`` leaves the model's own default in place instead of forcing
        reasoning off, which matters for models that always reason.
        """
        assert extract_reasoning(ResponseCreateParams(model="m", input="hi")) is None


class TestToolChoiceAllowedTools:
    """``allowed_tools`` and type-variant tool choices map onto Bedrock toolChoice.

    Bedrock offers only ``auto``, ``any`` and a single named ``tool``, so the
    richer upstream union is approximated; choices with no equivalent map to no
    ``toolChoice`` key, leaving the model unconstrained.

    Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
         stdapi/models/chat/_adapters/_openai_responses.py:_map_tool_choice
    """

    def test_required_with_single_function_forces_that_tool(self) -> None:
        """``required`` plus one allowed function tool forces that named tool."""
        choice = ToolChoiceAllowed(
            type="allowed_tools",
            mode="required",
            tools=[{"type": "function", "name": "get_weather"}],
        )
        assert _map_tool_choice(choice) == {"tool": {"name": "get_weather"}}

    def test_required_with_several_functions_forces_any_tool(self) -> None:
        """``required`` plus several allowed function tools maps to ``any``.

        Bedrock cannot restrict the choice to a subset of the declared tools, so
        the allow-list is widened to "any tool" rather than silently narrowed.
        """
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
        """``required`` with only non-function entries still maps to ``any``."""
        choice = ToolChoiceAllowed(
            type="allowed_tools",
            mode="required",
            tools=[{"type": "mcp", "server_label": "srv"}],
        )
        assert _map_tool_choice(choice) == {"any": {}}

    def test_auto_mode_maps_to_auto(self) -> None:
        """``allowed_tools`` with ``mode: "auto"`` maps to Bedrock ``auto``."""
        choice = ToolChoiceAllowed(
            type="allowed_tools",
            mode="auto",
            tools=[{"type": "function", "name": "get_weather"}],
        )
        assert _map_tool_choice(choice) == {"auto": {}}

    def test_builtin_type_variant_maps_to_no_constraint(self) -> None:
        """A built-in tool type choice yields no Bedrock constraint.

        ``None`` makes ``_build_tool_config`` omit ``toolChoice`` entirely, so
        the model keeps its default (auto) behavior instead of being forced into
        a tool Bedrock cannot name.
        """
        assert _map_tool_choice(ToolChoiceTypes(type="file_search")) is None


class TestReasoningSummarySignatures:
    """Envelope signatures are never attached to summary fallback texts.

    Bedrock computes a ``reasoningText.signature`` over the raw reasoning text;
    replaying it against a summary would be a signature/content mismatch, which
    Bedrock rejects.  Redacted payloads carry no such binding and survive.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlock.html
         stdapi/models/chat/_adapters/_openai_responses.py:_map_reasoning_item
    """

    async def test_signature_not_attached_to_summary_text(self) -> None:
        """A summary fallback text maps to a reasoningText block with no signature."""
        item = ResponseReasoningItem(
            id="rs_1",
            summary=[ReasoningItemSummary(text="sum", type="summary_text")],
            type="reasoning",
            encrypted_content=encode_reasoning_content(["sig-1"], []),
        )
        messages, _ = await map_input(cast("list[ResponseInputItem]", [item]), None)
        assert messages[0]["content"] == [
            {"reasoningContent": {"reasoningText": {"text": "sum"}}}
        ], "a signature must not be bound to text it was not computed over"

    async def test_redacted_blocks_survive_summary_fallback(self) -> None:
        """Redacted payloads are still replayed when signatures are discarded."""
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


class TestReplayedReasoningSignatureRequirement:
    """Unsigned replayed reasoning is dropped for models that mandate signatures.

    Some models reject a ``reasoningText`` block they cannot verify, which would
    fail the whole turn.  Replaying only the texts that still carry their
    signature keeps the turn alive at the cost of some replayed context; models
    that do not require signatures keep everything.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlock.html
         stdapi/models/chat/_adapters/_openai_responses.py:_map_reasoning_item
    """

    @staticmethod
    def _item(encrypted_content: str | None) -> ResponseReasoningItem:
        """Build a two-part reasoning item with the given envelope.

        Args:
            encrypted_content: Envelope to attach, or ``None`` for a bare item.

        Returns:
            The reasoning item to replay as an input.
        """
        return ResponseReasoningItem(
            id="rs_1",
            summary=[],
            type="reasoning",
            content=[
                ReasoningItemContent(text="first", type="reasoning_text"),
                ReasoningItemContent(text="second", type="reasoning_text"),
            ],
            encrypted_content=encrypted_content,
        )

    async def test_unsigned_texts_beyond_the_signatures_are_dropped(self) -> None:
        """Only the texts covered by a signature survive, in order."""
        messages, _ = await map_input(
            cast(
                "list[ResponseInputItem]",
                [self._item(encode_reasoning_content(["sig-1"], []))],
            ),
            None,
            reasoning_signature_required=True,
        )
        assert messages[0]["content"] == [
            {
                "reasoningContent": {
                    "reasoningText": {"text": "first", "signature": "sig-1"}
                }
            }
        ], "an unsigned text would be rejected by the model, failing the whole turn"

    async def test_a_fully_unsigned_item_contributes_no_reasoning(self) -> None:
        """An item whose envelope was stripped replays no reasoning at all.

        Clients that drop ``encrypted_content`` (it is only returned on request)
        must not turn the next turn into a model-side rejection.
        """
        messages, _ = await map_input(
            cast("list[ResponseInputItem]", [self._item(None)]),
            None,
            reasoning_signature_required=True,
        )
        assert messages == [], "an assistant turn with no block at all is not emitted"

    async def test_redacted_blocks_are_unaffected_by_the_requirement(self) -> None:
        """``redactedContent`` carries no text binding, so it is always replayed."""
        messages, _ = await map_input(
            cast(
                "list[ResponseInputItem]",
                [self._item(encode_reasoning_content([], [b"\x02"]))],
            ),
            None,
            reasoning_signature_required=True,
        )
        assert messages[0]["content"] == [
            {"reasoningContent": {"redactedContent": b"\x02"}}
        ]

    async def test_models_without_the_requirement_keep_every_text(self) -> None:
        """Without the requirement, the unsigned text is replayed as-is."""
        messages, _ = await map_input(
            cast(
                "list[ResponseInputItem]",
                [self._item(encode_reasoning_content(["sig-1"], []))],
            ),
            None,
        )
        assert messages[0]["content"] == [
            {
                "reasoningContent": {
                    "reasoningText": {"text": "first", "signature": "sig-1"}
                }
            },
            {"reasoningContent": {"reasoningText": {"text": "second"}}},
        ]
