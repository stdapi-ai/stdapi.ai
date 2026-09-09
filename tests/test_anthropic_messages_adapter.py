"""Server-tool results replayed as history through the Messages adapter (no AWS calls).

An agent loop appends the assistant turn it received to the next request, so a
``web_search_tool_result`` or ``code_execution_tool_result`` this gateway
produced itself comes straight back on the following turn and has to map to
something Bedrock accepts.  Converse refuses a ``searchResult`` block inside an
assistant turn — including nested in a ``toolResult`` — and accepts a
``toolResult`` carrying text, which is the shape asserted here (checked against
``amazon.nova-2-lite-v1:0`` on 2026-09-10).

Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-search-tool
     https://docs.aws.amazon.com/nova/latest/nova2-userguide/web-grounding.html
     stdapi/models/chat/_adapters/_anthropic_message.py:_map_content_block_to_bedrock
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from stdapi.api_errors import ApiError
from stdapi.models.chat._adapters._anthropic_message import (
    _map_messages,
    format_response,
)
from stdapi.models.chat.amazon_nova_2 import ChatModel
from stdapi.types.anthropic_messages import (
    ContainerUploadBlockParam,
    MessageParam,
    ServerToolUseBlockParam,
    WebFetchToolResultBlockParam,
    WebFetchToolResultErrorBlockParam,
    WebSearchToolRequestErrorParam,
    WebSearchToolResultBlockParam,
)

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ContentBlockOutputTypeDef,
        MessageTypeDef,
        TokenUsageTypeDef,
    )

pytestmark = pytest.mark.local

#: Model whose system tools produce the blocks replayed here.
_MODEL = ChatModel("amazon.nova-2-lite-v1:0")

#: Complete Bedrock ``TokenUsage``; these tests assert on content, not usage.
_NO_USAGE: TokenUsageTypeDef = {"inputTokens": 0, "outputTokens": 0, "totalTokens": 0}

#: What Bedrock answers a grounded question with: one tool use, then its results.
_GROUNDED_ANSWER = cast(
    "list[ContentBlockOutputTypeDef]",
    [
        {
            "toolUse": {
                "toolUseId": "tooluse_g1",
                "name": "nova_grounding",
                "input": {"query": "python version"},
            }
        },
        {
            "searchResult": {
                "source": "https://example.com/alpha",
                "title": "Alpha",
                "content": [{"text": "Alpha page"}],
            }
        },
        {
            "searchResult": {
                "source": "https://example.com/beta",
                "title": "Beta",
                "content": [{"text": "Beta page"}],
            }
        },
        {"text": "Python 3.14."},
    ],
)

#: What Bedrock answers with once it has run code for itself.
_EXECUTED_ANSWER = cast(
    "list[ContentBlockOutputTypeDef]",
    [
        {
            "toolUse": {
                "toolUseId": "tooluse_c1",
                "name": "nova_code_interpreter",
                "input": {"code": "print(27)"},
            }
        },
        {
            "toolResult": {
                "toolUseId": "tooluse_c1",
                "content": [
                    {
                        "json": {
                            "stdOut": "27\n",
                            "stdErr": "",
                            "exitCode": 0,
                            "isError": False,
                        }
                    }
                ],
            }
        },
        {"text": "27."},
    ],
)


async def _map_turn(content: list[ContentBlockOutputTypeDef]) -> MessageTypeDef:
    """Format *content* as an assistant turn, replay it, and return it as Bedrock.

    Args:
        content: Bedrock output content blocks of the answered turn.

    Returns:
        The Bedrock message the replayed assistant turn maps back to.
    """
    answer = await format_response(
        contents=content,
        stop_reason="end_turn",
        usage=_NO_USAGE,
        message_id="msg_1",
        model_id="model-x",
        forced_tool=None,
        resp_map_tool_result=_MODEL._resp_map_tool_result,  # noqa: SLF001
        resp_map_tool_use=_MODEL._resp_map_tool_use,  # noqa: SLF001
    )
    recorded = MessageParam.model_validate(
        {
            "role": "assistant",
            "content": answer.model_dump(exclude_none=True)["content"],
        }
    )
    assistant, _next_turn = await _map_messages(
        [recorded, MessageParam(role="user", content="And Node.js?")],
        req_map_content_block=_MODEL._req_map_content_block,  # noqa: SLF001
    )
    return assistant


async def _map_recorded(*blocks: object) -> MessageTypeDef:
    """Replay an assistant turn made of *blocks* and return it as Bedrock.

    Args:
        blocks: Content blocks a client recorded from an earlier turn.

    Returns:
        The Bedrock message the replayed assistant turn maps back to.
    """
    assistant, _next_turn = await _map_messages(
        [
            MessageParam.model_validate({"role": "assistant", "content": list(blocks)}),
            MessageParam(role="user", content="And Node.js?"),
        ],
        req_map_content_block=_MODEL._req_map_content_block,  # noqa: SLF001
    )
    return assistant


class TestReplayedWebSearchResults:
    """A recorded ``web_search_tool_result`` is history, not a rejected request.

    The gateway emits the block itself, and every Anthropic agent loop sends the
    assistant turn back on the next request, so refusing it makes the second turn
    of every grounded conversation fail — a divergence from the Messages API,
    which accepts the replay.
    """

    async def test_the_recorded_result_pairs_with_the_tool_call_it_answers(
        self,
    ) -> None:
        """The result maps to the ``toolUseId`` its tool call was mapped to.

        A result whose identifier matches no tool call is rejected by Bedrock, so
        both halves have to survive the same identifier rewrite.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultBlock.html
        """
        assistant = await _map_turn(_GROUNDED_ANSWER)

        tool_use = assistant["content"][0]["toolUse"]
        tool_result = assistant["content"][1]["toolResult"]
        assert tool_use["name"] == "nova_grounding"
        assert tool_use["toolUseId"] == "tooluse_g1"
        assert tool_result["toolUseId"] == "tooluse_g1"
        assert assistant["content"][2] == {"text": "Python 3.14."}

    async def test_every_result_keeps_its_title_and_url(self) -> None:
        """Both results reach the model, so the replayed turn still cites its sources.

        Converse rejects a ``searchResult`` block in an assistant turn, so the
        results are rendered into the tool result's text instead of being
        forwarded as the blocks they arrived as.
        """
        assistant = await _map_turn(_GROUNDED_ANSWER)

        text = assistant["content"][1]["toolResult"]["content"][0]["text"]
        for expected in (
            "Alpha",
            "https://example.com/alpha",
            "Beta",
            "https://example.com/beta",
        ):
            assert expected in text, f"{expected!r} missing from {text!r}"
        assert "status" not in assistant["content"][1]["toolResult"]

    async def test_a_failed_search_is_replayed_as_a_failure(self) -> None:
        """An error result keeps its error status and its code.

        Replaying a failed search as a successful one leaves the model reading an
        empty result set as an answer, and the failure it should react to is gone.
        """
        assistant = await _map_recorded(
            ServerToolUseBlockParam(
                type="server_tool_use",
                id="srvtoolu_g1",
                name="web_search",
                input={"query": "python version"},
            ),
            WebSearchToolResultBlockParam(
                type="web_search_tool_result",
                tool_use_id="srvtoolu_g1",
                content=WebSearchToolRequestErrorParam(
                    type="web_search_tool_result_error", error_code="max_uses_exceeded"
                ),
            ),
        )

        tool_result = assistant["content"][1]["toolResult"]
        assert tool_result["toolUseId"] == "tooluse_g1"
        assert tool_result["status"] == "error"
        assert "max_uses_exceeded" in tool_result["content"][0]["text"]


class TestReplayedToolResultsOfOtherServerTools:
    """The remaining ``*_tool_result`` blocks of the Messages API replay too.

    A transcript recorded elsewhere carries results of server tools this gateway
    never runs; they are history all the same, and a request carrying one must
    not be refused for it.
    """

    async def test_a_code_execution_result_keeps_the_model_s_own_mapping(self) -> None:
        """Nova's own translation still wins over the generic one.

        Its code interpreter reads the result back as the JSON payload it
        produced, which a text rendering would replace with prose.

        Ref: stdapi/models/chat/amazon_nova_2.py:ChatModel._req_map_content_block
        """
        assistant = await _map_turn(_EXECUTED_ANSWER)

        assert assistant["content"][1] == {
            "toolResult": {
                "toolUseId": "tooluse_c1",
                "content": [
                    {
                        "json": {
                            "stdOut": "27\n",
                            "stdErr": "",
                            "exitCode": 0,
                            "isError": False,
                        }
                    }
                ],
                "status": "success",
            }
        }

    async def test_a_result_of_a_tool_the_gateway_never_runs_is_still_replayed(
        self,
    ) -> None:
        """A ``web_fetch_tool_result`` from a foreign transcript maps, not raises.

        Nothing here produces one, so it can only arrive as history; refusing the
        whole request over a block the client is merely echoing back is stricter
        than the Messages API.
        """
        assistant = await _map_recorded(
            ServerToolUseBlockParam(
                type="server_tool_use",
                id="srvtoolu_f1",
                name="web_fetch",
                input={"url": "https://example.com/alpha"},
            ),
            WebFetchToolResultBlockParam(
                type="web_fetch_tool_result",
                tool_use_id="srvtoolu_f1",
                content=WebFetchToolResultErrorBlockParam(
                    type="web_fetch_tool_result_error", error_code="url_not_accessible"
                ),
            ),
        )

        tool_result = assistant["content"][1]["toolResult"]
        assert tool_result["toolUseId"] == "tooluse_f1"
        assert tool_result["status"] == "error"
        assert "url_not_accessible" in tool_result["content"][0]["text"]

    async def test_a_block_with_no_backend_equivalent_is_still_refused(self) -> None:
        """``container_upload`` has nothing to be replayed as, and still fails.

        Accepting the results of server tools must not turn the mapping into a
        catch-all: a block that cannot reach the model at all is a request the
        client has to be told about.
        """
        with pytest.raises(ApiError, match="container_upload"):
            await _map_recorded(
                ContainerUploadBlockParam(type="container_upload", file_id="file_1")
            )
