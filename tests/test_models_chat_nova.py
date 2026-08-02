"""Converse request building and code-interpreter mapping for Amazon Nova 2 (no AWS calls).

Nova 2 rejects a Converse request that combines ``inferenceConfig.maxTokens``
with ``reasoningConfig.maxReasoningEffort: high``, so the model class drops the
token limit and records a warning in the request log instead of letting Bedrock
fail the call.  It also serves Anthropic's ``code_execution`` server tool through
its own ``nova_code_interpreter`` system tool, which needs a translation in both
directions.

Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
     https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
     stdapi/models/chat/amazon_nova_2.py:ChatModel._prepare_converse_request
"""

from typing import TYPE_CHECKING

import pytest

from stdapi.models.chat.amazon_nova_2 import ChatModel
from stdapi.types.anthropic_messages import (
    CodeExecutionResultBlock,
    CodeExecutionResultBlockParam,
    CodeExecutionToolResultBlock,
    CodeExecutionToolResultBlockParam,
    CodeExecutionToolResultErrorParam,
    TextBlockParam,
)

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime.type_defs import (
        InferenceConfigurationTypeDef,
    )

    from stdapi.monitoring import EventLog

pytestmark = [pytest.mark.local, pytest.mark.usefixtures("request_log")]

#: Model instance used across tests; construction is side-effect free.
_MODEL = ChatModel("amazon.nova-2-lite-v1:0")


class TestPrepareConverseRequestReasoningMaxTokens:
    """``_prepare_converse_request`` drops ``maxTokens`` only for high-effort reasoning.

    Ref: stdapi/models/chat/amazon_nova_2.py:ChatModel._prepare_converse_request
    """

    async def test_high_effort_reasoning_drops_max_tokens(
        self, request_log: EventLog
    ) -> None:
        """High reasoning effort strips 'maxTokens' from the inference config and warns.

        Only the token limit is removed: the ``reasoningConfig`` that triggered
        the conflict is still forwarded in ``additionalModelRequestFields``.
        """
        inference_cfg: InferenceConfigurationTypeDef = {"maxTokens": 1000}
        request = await _MODEL._prepare_converse_request(  # noqa: SLF001
            bedrock_messages=[],
            inference_cfg=inference_cfg,
            system_blocks=None,
            tool_config=None,
            additional_request_fields={
                "reasoningConfig": {"type": "enabled", "maxReasoningEffort": "high"}
            },
            service_tier=None,
        )
        assert "maxTokens" not in request["inferenceConfig"]
        assert request["additionalModelRequestFields"]["reasoningConfig"] == {
            "type": "enabled",
            "maxReasoningEffort": "high",
        }
        assert request_log["level"] == "warning"
        assert any(
            "max_tokens" in str(detail)
            for detail in request_log.get("error_detail", [])
        ), "the dropped token limit must be reported in the request log"

    async def test_medium_effort_reasoning_preserves_max_tokens(
        self, request_log: EventLog
    ) -> None:
        """Medium reasoning effort leaves 'maxTokens' untouched and does not warn."""
        inference_cfg: InferenceConfigurationTypeDef = {"maxTokens": 1000}
        request = await _MODEL._prepare_converse_request(  # noqa: SLF001
            bedrock_messages=[],
            inference_cfg=inference_cfg,
            system_blocks=None,
            tool_config=None,
            additional_request_fields={
                "reasoningConfig": {"type": "enabled", "maxReasoningEffort": "medium"}
            },
            service_tier=None,
        )
        assert request["inferenceConfig"]["maxTokens"] == 1000
        assert request_log["level"] == "info"
        assert "error_detail" not in request_log

    async def test_no_reasoning_preserves_max_tokens(
        self, request_log: EventLog
    ) -> None:
        """No reasoning config leaves 'maxTokens' untouched and does not warn."""
        inference_cfg: InferenceConfigurationTypeDef = {"maxTokens": 1000}
        request = await _MODEL._prepare_converse_request(  # noqa: SLF001
            bedrock_messages=[],
            inference_cfg=inference_cfg,
            system_blocks=None,
            tool_config=None,
            additional_request_fields={},
            service_tier=None,
        )
        assert request["inferenceConfig"]["maxTokens"] == 1000
        assert request_log["level"] == "info"
        assert "error_detail" not in request_log

    async def test_high_effort_reasoning_without_max_tokens_does_not_warn(
        self, request_log: EventLog
    ) -> None:
        """High reasoning effort with no 'maxTokens' set logs no warning."""
        inference_cfg: InferenceConfigurationTypeDef = {}
        request = await _MODEL._prepare_converse_request(  # noqa: SLF001
            bedrock_messages=[],
            inference_cfg=inference_cfg,
            system_blocks=None,
            tool_config=None,
            additional_request_fields={
                "reasoningConfig": {"type": "enabled", "maxReasoningEffort": "high"}
            },
            service_tier=None,
        )
        assert "maxTokens" not in request["inferenceConfig"]
        assert request_log["level"] == "info"
        assert "error_detail" not in request_log


class TestCodeInterpreterRoundTrip:
    """Anthropic ``code_execution`` blocks translate to Nova's ``nova_code_interpreter``.

    Nova exposes code execution as a Bedrock ``systemTool`` returning a
    ``toolResult`` with a JSON payload, while Anthropic models it as a
    ``code_execution_tool_result`` block wrapping stdout/stderr/return code.  The
    two shapes must survive a full turn: the response mapping is what the client
    reads, and the request mapping is what lets it replay the block on the next
    turn.  The ``srvtoolu_``/``tooluse_`` prefix swap keeps the two sides
    correlated.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultBlock.html
         https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
         stdapi/models/chat/amazon_nova_2.py:ChatModel._resp_map_tool_result
    """

    def test_response_tool_result_becomes_a_code_execution_block(self) -> None:
        """A successful ``nova_code_interpreter`` result maps to return code zero.

        Bedrock reports success through ``isError``, not through ``exitCode``, so a
        payload flagged as successful must yield ``return_code == 0`` whatever the
        reported exit code is.
        """
        blocks = _MODEL._resp_map_tool_result(  # noqa: SLF001
            "tooluse_abc",
            "nova_code_interpreter",
            [
                {
                    "json": {
                        "stdOut": "42\n",
                        "stdErr": "",
                        "exitCode": 0,
                        "isError": False,
                    }
                }
            ],
        )
        assert blocks is not None
        (block,) = blocks
        assert isinstance(block, CodeExecutionToolResultBlock)
        assert block.tool_use_id == "srvtoolu_abc"
        result = block.content
        assert isinstance(result, CodeExecutionResultBlock)
        assert result.stdout == "42\n"
        assert result.return_code == 0

    def test_failed_code_execution_keeps_its_exit_code(self) -> None:
        """A failing result carries stderr and the reported exit code.

        ``exitCode`` is only trusted when ``isError`` is set; a missing or zero
        exit code on a failed run degrades to ``1`` so the client never reads a
        failure as a success.
        """
        blocks = _MODEL._resp_map_tool_result(  # noqa: SLF001
            "tooluse_abc",
            "nova_code_interpreter",
            [{"json": {"stdOut": "", "stdErr": "boom", "isError": True}}],
        )
        assert blocks is not None
        (block,) = blocks
        assert isinstance(block, CodeExecutionToolResultBlock)
        result = block.content
        assert isinstance(result, CodeExecutionResultBlock)
        assert result.stderr == "boom"
        assert result.return_code == 1

    def test_unknown_tool_name_falls_back_to_the_base_mapping(self) -> None:
        """A ``toolResult`` from any other tool is not claimed by the Nova mapping.

        Returning ``None`` is what hands the block back to the generic adapter, so
        a future Nova system tool is not silently rendered as a code-execution
        result.
        """
        assert (
            _MODEL._resp_map_tool_result(  # noqa: SLF001
                "tooluse_abc", "nova_grounding", [{"json": {}}]
            )
            is None
        )

    def test_stream_tool_result_is_mapped_by_result_type(self) -> None:
        """The streamed ``nova_code_interpreter_result`` type yields the same block.

        The streaming path identifies the result by the ``type`` on the
        ``contentBlockStart`` rather than by a tool name, and must agree with the
        non-streaming mapping.
        """
        block = _MODEL._resp_stream_map_tool_result(  # noqa: SLF001
            "tooluse_abc",
            "nova_code_interpreter_result",
            [{"json": {"stdOut": "hi", "stdErr": "", "exitCode": 0, "isError": False}}],
        )
        assert block is not None
        assert isinstance(block, CodeExecutionToolResultBlock)
        assert block.tool_use_id == "srvtoolu_abc"
        result = block.content
        assert isinstance(result, CodeExecutionResultBlock)
        assert result.stdout == "hi"
        assert (
            _MODEL._resp_stream_map_tool_result(  # noqa: SLF001
                "tooluse_abc", "nova_grounding_result", []
            )
            is None
        )

    def test_replayed_result_block_becomes_a_bedrock_tool_result(self) -> None:
        """A replayed ``code_execution_tool_result`` maps back to a ``toolResult``.

        The Anthropic ``srvtoolu_`` id is restored to the Bedrock ``tooluse_``
        form it came from, and a non-zero return code is reported through both
        ``isError`` and the block ``status`` Bedrock uses.
        """
        block = _MODEL._req_map_content_block(  # noqa: SLF001
            CodeExecutionToolResultBlockParam(
                type="code_execution_tool_result",
                tool_use_id="srvtoolu_abc",
                content=CodeExecutionResultBlockParam(
                    type="code_execution_result",
                    content=[],
                    return_code=2,
                    stderr="boom",
                    stdout="",
                ),
            )
        )
        assert block == {
            "toolResult": {
                "toolUseId": "tooluse_abc",
                "content": [
                    {
                        "json": {
                            "stdOut": "",
                            "stdErr": "boom",
                            "exitCode": 2,
                            "isError": True,
                        }
                    }
                ],
                "status": "error",
            }
        }

    def test_replayed_error_result_becomes_an_empty_error_tool_result(self) -> None:
        """An error-variant result replays as an empty payload flagged as an error.

        ``code_execution_tool_result_error`` has no stdout/stderr to forward, so
        the Bedrock block carries an empty JSON object; dropping the ``status``
        would replay the failure to Nova as a success.
        """
        block = _MODEL._req_map_content_block(  # noqa: SLF001
            CodeExecutionToolResultBlockParam(
                type="code_execution_tool_result",
                tool_use_id="srvtoolu_abc",
                content=CodeExecutionToolResultErrorParam(
                    type="code_execution_tool_result_error",
                    error_code="execution_time_exceeded",
                ),
            )
        )
        assert block == {
            "toolResult": {
                "toolUseId": "tooluse_abc",
                "content": [{"json": {}}],
                "status": "error",
            }
        }

    def test_other_block_types_fall_through_to_the_base_mapping(self) -> None:
        """A block Nova does not special-case is left to the generic adapter.

        ``None`` means "not mine", which is what keeps text, image and tool_use
        blocks on the shared mapping path.
        """
        assert (
            _MODEL._req_map_content_block(  # noqa: SLF001
                TextBlockParam(type="text", text="hi")
            )
            is None
        )
