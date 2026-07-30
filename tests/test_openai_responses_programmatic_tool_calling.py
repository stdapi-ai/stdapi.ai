"""Unit tests for the Responses API programmatic-tool-calling surface (no AWS calls).

Bedrock Converse has no equivalent of OpenAI's ``programmatic_tool_calling``
hosted tool (the model calls tools directly), so the gateway accepts the tool,
its ``tool_choice`` form and the ``program``/``program_output`` items it
produces, then drops them instead of failing the request.

Ref: https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
     stdapi/types/openai_responses.py:ProgrammaticToolCalling
     stdapi/models/chat/_adapters/_openai_responses.py:_build_tool_config
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from stdapi.models.chat._adapters._openai_responses import _build_tool_config, map_input
from stdapi.types.openai_responses import (
    FunctionTool,
    InputTokenCountParams,
    Program,
    ProgrammaticToolCalling,
    ProgramOutput,
    ResponseCreateParams,
    ResponseInputItem,
    ResponseItem,
    ResponseOutputItem,
    ToolChoiceProgrammaticToolCalling,
)

pytestmark = pytest.mark.local

#: Canned upstream `program` output item payload.
_PROGRAM_ITEM: dict[str, str] = {
    "id": "prog_1",
    "call_id": "call_1",
    "code": "print(get_weather('Paris'))",
    "fingerprint": "fp_1",
    "type": "program",
}

#: Canned upstream `program_output` output item payload.
_PROGRAM_OUTPUT_ITEM: dict[str, str] = {
    "id": "progout_1",
    "call_id": "call_1",
    "result": "sunny",
    "status": "completed",
    "type": "program_output",
}


class TestTypeSurface:
    """The upstream programmatic-tool-calling types parse instead of 422-ing."""

    def test_tool_parses(self) -> None:
        """A programmatic_tool_calling tool validates as its own Tool union member."""
        request = ResponseCreateParams.model_validate(
            {
                "model": "m",
                "input": "x",
                "tools": [{"type": "programmatic_tool_calling"}],
            }
        )
        assert request.tools is not None
        (tool,) = request.tools
        assert isinstance(tool, ProgrammaticToolCalling)
        assert tool.type == "programmatic_tool_calling"

    def test_tool_choice_parses(self) -> None:
        """The matching tool_choice object validates through the ToolChoice union."""
        request = ResponseCreateParams.model_validate(
            {
                "model": "m",
                "input": "x",
                "tool_choice": {"type": "programmatic_tool_calling"},
            }
        )
        assert isinstance(request.tool_choice, ToolChoiceProgrammaticToolCalling)
        assert request.tool_choice.type == "programmatic_tool_calling"

    @pytest.mark.parametrize("payload", [_PROGRAM_ITEM, _PROGRAM_OUTPUT_ITEM])
    def test_input_items_parse(self, payload: dict[str, str]) -> None:
        """program/program_output items validate as request input items.

        The input-item union is a plain (non-discriminated) union, so the
        round-trip dump is asserted to prove the payload is not absorbed by a
        laxer member that would silently drop ``code``/``result``.
        """
        item = TypeAdapter[ResponseInputItem](ResponseInputItem).validate_python(
            payload
        )
        assert item.type == payload["type"]
        assert item.model_dump(exclude_none=True) == payload

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [(_PROGRAM_ITEM, Program), (_PROGRAM_OUTPUT_ITEM, ProgramOutput)],
    )
    def test_output_items_parse(self, payload: dict[str, str], expected: type) -> None:
        """program/program_output items validate as response output items."""
        item = TypeAdapter[ResponseOutputItem](ResponseOutputItem).validate_python(
            payload
        )
        assert isinstance(item, expected)
        assert item.model_dump(exclude_none=True) == payload

    @pytest.mark.parametrize("payload", [_PROGRAM_ITEM, _PROGRAM_OUTPUT_ITEM])
    def test_stored_items_parse(self, payload: dict[str, str]) -> None:
        """program/program_output items validate as listed stored items."""
        item = TypeAdapter[ResponseItem](ResponseItem).validate_python(payload)
        assert item.type == payload["type"]
        assert item.model_dump(exclude_none=True) == payload


class TestConverseDrop:
    """Converse-served models accept the tool and drop it instead of 400-ing.

    Ref: stdapi/models/chat/_adapters/_openai_responses.py:_build_tool_config
    """

    def test_tool_is_dropped(self) -> None:
        """The tool is dropped while the other tools stay callable directly."""
        request = ResponseCreateParams.model_validate(
            {
                "model": "m",
                "input": "x",
                "tools": [
                    {"type": "programmatic_tool_calling"},
                    {"type": "function", "name": "get_weather"},
                ],
            }
        )
        tool_config = _build_tool_config(request)
        assert tool_config is not None
        assert [tool["toolSpec"]["name"] for tool in tool_config["tools"]] == [
            "get_weather"
        ]

    def test_tool_alone_yields_no_tool_config(self) -> None:
        """A request whose only tool is dropped produces no Bedrock tool config."""
        request = ResponseCreateParams.model_validate(
            {
                "model": "m",
                "input": "x",
                "tools": [{"type": "programmatic_tool_calling"}],
            }
        )
        assert _build_tool_config(request) is None

    def test_tool_choice_degrades_to_default(self) -> None:
        """The forced tool_choice is ignored, leaving the model's default choice.

        Bedrock's ToolChoice union has no programmatic variant, so omitting
        ``toolChoice`` is the only mapping that keeps the request valid.
        """
        request = ResponseCreateParams.model_validate(
            {
                "model": "m",
                "input": "x",
                "tools": [{"type": "function", "name": "get_weather"}],
                "tool_choice": {"type": "programmatic_tool_calling"},
            }
        )
        tool_config = _build_tool_config(request)
        assert tool_config is not None
        assert "toolChoice" not in tool_config
        assert [tool["toolSpec"]["name"] for tool in tool_config["tools"]] == [
            "get_weather"
        ]

    def test_tool_choice_without_tools_yields_no_tool_config(self) -> None:
        """The tool_choice alone is accepted and yields no Bedrock tool config."""
        request = ResponseCreateParams.model_validate(
            {
                "model": "m",
                "input": "x",
                "tool_choice": {"type": "programmatic_tool_calling"},
            }
        )
        assert _build_tool_config(request) is None

    def test_input_token_count_drops_the_tool(self) -> None:
        """The token-count path accepts and drops the tool the same way."""
        request = InputTokenCountParams.model_validate(
            {
                "model": "m",
                "input": "x",
                "tools": [{"type": "programmatic_tool_calling"}],
            }
        )
        assert _build_tool_config(request) is None

    def test_programmatic_only_callers_stay_direct_tools(self) -> None:
        """A tool opted into programmatic callers only is still exposed directly.

        Without a program to run the tool from, honoring ``allowed_callers``
        would make the tool unreachable; it is parsed and then left out of the
        Bedrock toolSpec.
        """
        request = ResponseCreateParams.model_validate(
            {
                "model": "m",
                "input": "x",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "allowed_callers": ["programmatic"],
                    }
                ],
            }
        )
        assert request.tools is not None
        (function_tool,) = request.tools
        assert isinstance(function_tool, FunctionTool)
        assert function_tool.allowed_callers == ["programmatic"]
        tool_config = _build_tool_config(request)
        assert tool_config is not None
        (tool,) = tool_config["tools"]
        assert tool["toolSpec"]["name"] == "get_weather"
        assert "allowed_callers" not in tool["toolSpec"]

    def test_other_tools_are_unaffected(self) -> None:
        """Requests without programmatic tool calling still build a tool config."""
        request = ResponseCreateParams.model_validate(
            {
                "model": "m",
                "input": "x",
                "tools": [{"type": "function", "name": "get_weather"}],
            }
        )
        tool_config = _build_tool_config(request)
        assert tool_config is not None
        assert tool_config["tools"][0]["toolSpec"]["name"] == "get_weather"


class TestInputReplay:
    """Echoed program history items are accepted and dropped on replay.

    Ref: stdapi/models/chat/_adapters/_openai_responses.py:map_input
    """

    async def test_program_items_are_dropped(self) -> None:
        """program/program_output items produce no Bedrock message."""
        items = [
            TypeAdapter[ResponseInputItem](ResponseInputItem).validate_python(payload)
            for payload in (_PROGRAM_ITEM, _PROGRAM_OUTPUT_ITEM)
        ]
        messages, system = await map_input(items, None)
        assert messages == []
        assert system == []
