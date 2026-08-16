"""Unit tests for the Responses tool config: ``text.format`` and server tools.

Structured Outputs are served by Bedrock's native ``outputConfig.textFormat``
JSON schema definition rather than by prompt-level instructions, so the schema
name and description supplied by the client must reach Bedrock verbatim.

Ref: https://developers.openai.com/api/docs/guides/structured-outputs
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
     stdapi/models/chat/_adapters/_openai_responses.py:_build_output_config
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from stdapi.api_errors import ApiError
from stdapi.models.chat._adapters._anthropic_message import _handle_system_tool
from stdapi.models.chat._adapters._openai_responses import (
    _build_output_config,
    _build_tool_config,
    _resolve_integrated_tool_name,
)
from stdapi.types.anthropic_messages import WebSearchToolParam
from stdapi.types.openai_responses import (
    ResponseCreateParams,
    ResponseTextConfig,
    WebSearchTool,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from stdapi.types.anthropic_messages import ServerTools

pytestmark = pytest.mark.local

#: Server tool map of a model serving none, as every Converse model but Nova has.
_NO_SERVER_TOOLS: Mapping[ServerTools, str] = {}

#: Server tool map of a model serving a different tool than the one requested.
_OTHER_SERVER_TOOL: Mapping[ServerTools, str] = {"code_execution": "code_interpreter"}

#: Server tool map of a model serving web search as its own grounding tool.
_NOVA_SERVER_TOOL: Mapping[ServerTools, str] = {"web_search": "nova_grounding"}


def test_build_output_config_returns_none_for_json_object() -> None:
    """``text.format={"type": "json_object"}`` builds no Bedrock outputConfig.

    Bedrock's strict structured output has no schema for "any JSON object": an
    empty schema is rejected and the only closed alternative,
    ``{"type": "object", "additionalProperties": false}``, admits only ``{}``,
    so the adapter must skip outputConfig entirely rather than constrain the
    model to an empty response (issue #96).
    """
    text = ResponseTextConfig.model_validate({"format": {"type": "json_object"}})
    assert _build_output_config(text) is None


def test_build_output_config_forwards_json_schema_name_and_description() -> None:
    """The client-supplied schema, name and description reach Bedrock's jsonSchema."""
    text = ResponseTextConfig.model_validate(
        {
            "format": {
                "type": "json_schema",
                "name": "weather_report",
                "description": "A weather report.",
                "schema": {"type": "object"},
            }
        }
    )
    output_config = _build_output_config(text)
    assert output_config is not None
    assert output_config["name"] == "weather_report"
    assert output_config["description"] == "A weather report."
    assert output_config["schema"] == '{"type":"object"}', (
        "the schema is forwarded as a serialised JSON string"
    )


def test_build_output_config_omits_description_when_unset() -> None:
    """An unset schema description is not forwarded to Bedrock's jsonSchema."""
    text = ResponseTextConfig.model_validate(
        {
            "format": {
                "type": "json_schema",
                "name": "weather_report",
                "schema": {"type": "object"},
            }
        }
    )
    output_config = _build_output_config(text)
    assert output_config is not None
    assert "description" not in output_config
    assert output_config["name"] == "weather_report"


class TestServerToolParity:
    """Both API surfaces answer a server tool the model cannot serve alike.

    Upstream OpenAI hosts its integrated tools, so it rejects one the model
    cannot run with a 400. Bedrock hosts them only for the models that declare
    a mapping, so the gateway forwards the tool as a stub for every other
    model: Claude requires that stub in ``toolConfig`` for Bedrock to accept
    the matching ``toolResult`` blocks in a multi-turn conversation. What must
    not diverge is the two surfaces answering the same request differently.

    Ref: https://developers.openai.com/api/docs/guides/tools-web-search
         stdapi/models/chat/_adapters/_openai_responses.py:_resolve_integrated_tool_name
         stdapi/models/chat/_adapters/_anthropic_message.py:_handle_system_tool
    """

    def test_a_model_declaring_no_server_tool_gets_the_stub(self) -> None:
        """Neither surface rejects: the Responses call site passes ``None``."""
        tool_list: list[Any] = []
        _handle_system_tool(
            WebSearchToolParam(name="web_search", type="web_search_20250305"),
            tool_list,
            tool_name_map=_NO_SERVER_TOOLS,
        )

        assert tool_list[0]["toolSpec"]["name"] == "web_search"
        assert (
            _resolve_integrated_tool_name(
                WebSearchTool(type="web_search"), _NO_SERVER_TOOLS or None
            )
            == "web_search"
        )

    def test_a_model_declaring_other_server_tools_rejects(self) -> None:
        """Both surfaces raise when the model maps tools but not this one."""
        with pytest.raises(ApiError, match="not supported by this model"):
            _handle_system_tool(
                WebSearchToolParam(name="web_search", type="web_search_20250305"),
                [],
                tool_name_map=_OTHER_SERVER_TOOL,
            )

        with pytest.raises(ApiError, match="not supported by this model"):
            _resolve_integrated_tool_name(
                WebSearchTool(type="web_search"), _OTHER_SERVER_TOOL or None
            )


def _request(*tools: dict[str, Any]) -> ResponseCreateParams:
    """Build a Responses request declaring *tools*."""
    return ResponseCreateParams.model_validate(
        {"model": "ignored", "input": "Who won?", "tools": list(tools)}
    )


class TestWebSearchRestrictions:
    """Search restrictions the backend cannot apply are refused, not dropped.

    A backend-served web search takes no parameters, so a dropped restriction
    runs a wider search than the caller asked for -- with ``allowed_domains``,
    one reaching the very domains the caller excluded.  The same request is
    refused on the Anthropic Messages surface, on the same backend.

    Ref: https://developers.openai.com/api/docs/guides/tools-web-search
         https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
         stdapi/models/chat/_adapters/_openai_responses.py:_reject_web_search_restrictions
    """

    @pytest.mark.parametrize(
        "tool_name_map",
        [_NO_SERVER_TOOLS or None, _NOVA_SERVER_TOOL],
        ids=["none", "nova"],
    )
    def test_allowed_domains_is_refused(
        self, tool_name_map: Mapping[ServerTools, str] | None
    ) -> None:
        """A domain allowlist is a security filter: silently dropping it is not an option."""
        tool = {"type": "web_search", "filters": {"allowed_domains": ["example.com"]}}
        with pytest.raises(ApiError) as excinfo:
            _build_tool_config(_request(tool), tool_name_map)
        assert excinfo.value.status == 400
        message = str(excinfo.value)
        assert "filters.allowed_domains" in message
        assert "not supported by this model" in message

    @pytest.mark.parametrize(
        "tool_type", ["web_search", "web_search_preview"], ids=["search", "preview"]
    )
    def test_user_location_is_refused_on_both_tool_spellings(
        self, tool_type: str
    ) -> None:
        """Both spellings of the tool carry ``user_location``, and both are refused."""
        tool = {
            "type": tool_type,
            "user_location": {"type": "approximate", "city": "Paris"},
        }
        with pytest.raises(ApiError, match="user_location") as excinfo:
            _build_tool_config(_request(tool), _NOVA_SERVER_TOOL)
        assert excinfo.value.status == 400

    def test_search_context_size_is_accepted_and_ignored(self) -> None:
        """A context-size hint still yields the searched answer, so it is kept.

        Ref: stdapi/models/chat/_adapters/_openai_responses.py:_build_tool_config
        """
        tool = {"type": "web_search", "search_context_size": "high"}
        tool_config = _build_tool_config(_request(tool), _NOVA_SERVER_TOOL)
        assert tool_config is not None
        (spec,) = tool_config["tools"]
        assert spec["toolSpec"]["name"] == "nova_grounding"
        assert spec["toolSpec"]["inputSchema"] == {"json": {"type": "object"}}

    def test_a_plain_web_search_tool_is_untouched(self) -> None:
        """The control case: a tool restricting nothing reaches the backend."""
        tool_config = _build_tool_config(
            _request({"type": "web_search"}), _NOVA_SERVER_TOOL
        )
        assert tool_config is not None
        assert tool_config["tools"][0]["toolSpec"]["name"] == "nova_grounding"


class TestFileSearchTool:
    """``file_search`` is declared to the model as a tool the gateway answers.

    The retrieval runs here, not in the model, so the model is given a plain
    function tool it can ask a search with; the gateway runs it against the
    attached vector stores and continues the turn. It is withdrawn once the
    turn has already searched twice, so the last invocation has to answer.

    Ref: https://developers.openai.com/api/docs/guides/tools-file-search
         stdapi/models/chat/_adapters/_openai_responses.py:_build_tool_config
    """

    #: One answered search, as it comes back in the continued turn's input.
    _ANSWERED: ClassVar[dict[str, object]] = {
        "id": "fs_1",
        "type": "file_search_call",
        "status": "completed",
        "queries": ["cats"],
        "results": [{"text": "meow"}],
    }

    def test_file_search_is_declared_as_a_function_tool(self) -> None:
        """The model can call it, on a model serving no system tool of its own."""
        tool = {"type": "file_search", "vector_store_ids": ["vs_123"]}
        tool_config = _build_tool_config(_request(tool), _NO_SERVER_TOOLS or None)
        assert tool_config is not None
        (spec,) = tool_config["tools"]
        assert spec["toolSpec"]["name"] == "file_search"
        assert spec["toolSpec"]["inputSchema"]["json"]["required"] == ["query"]

    def test_it_is_declared_beside_a_function_tool(self) -> None:
        """A caller's own tools stay declared alongside it."""
        tool_config = _build_tool_config(
            _request(
                {"type": "function", "name": "get_weather"},
                {"type": "file_search", "vector_store_ids": ["vs_123"]},
            ),
            None,
        )
        assert tool_config is not None
        assert {spec["toolSpec"]["name"] for spec in tool_config["tools"]} == {
            "file_search",
            "get_weather",
        }

    def test_it_is_withdrawn_once_the_turn_reached_its_search_limit(self) -> None:
        """Two answered searches in the input leave the model no tool to loop with."""
        request = ResponseCreateParams.model_validate(
            {
                "model": "test-model",
                "input": [
                    {"role": "user", "content": "hi"},
                    self._ANSWERED,
                    self._ANSWERED | {"id": "fs_2"},
                ],
                "tools": [{"type": "file_search", "vector_store_ids": ["vs_123"]}],
            }
        )

        assert _build_tool_config(request, None) is None

    def test_a_replayed_search_from_a_previous_turn_does_not_withdraw_it(self) -> None:
        """Searches followed by a message belong to a turn that already ended."""
        request = ResponseCreateParams.model_validate(
            {
                "model": "test-model",
                "input": [
                    self._ANSWERED,
                    self._ANSWERED | {"id": "fs_2"},
                    {"role": "user", "content": "and now?"},
                ],
                "tools": [{"type": "file_search", "vector_store_ids": ["vs_123"]}],
            }
        )

        tool_config = _build_tool_config(request, None)
        assert tool_config is not None
        assert tool_config["tools"][0]["toolSpec"]["name"] == "file_search"
