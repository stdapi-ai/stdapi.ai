"""Amazon Nova 2 chat model implementation."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from stdapi.models.chat._default import ChatModel as _BaseChatModel
from stdapi.monitoring import log_error_details
from stdapi.types.anthropic_messages import (
    CodeExecutionResultBlock,
    CodeExecutionResultBlockParam,
    CodeExecutionToolResultBlock,
    CodeExecutionToolResultBlockParam,
)

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime.literals import ServiceTierTypeType
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ContentBlockTypeDef,
        InferenceConfigurationTypeDef,
        JsonSchemaDefinitionTypeDef,
        MessageTypeDef,
        SystemContentBlockTypeDef,
        ToolConfigurationTypeDef,
        ToolResultContentBlockOutputTypeDef,
    )

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef
    from stdapi.models.chat import Effort
    from stdapi.types import JsonMapping
    from stdapi.types.anthropic_messages import ContentBlock, ContentBlockParam

    class _NovaCodeInterpreterResult(TypedDict, total=False):
        """JSON payload returned by Bedrock for a ``nova_code_interpreter`` toolResult."""

        stdOut: str
        stdErr: str
        exitCode: int
        isError: bool


# Nova reasoning effort values
NovaReasoning = Literal["low", "medium", "high"]


#: OpenAI to Deepseek override
_REASONING_OVERRIDE: dict[Effort | None, NovaReasoning] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


class ChatModel(_BaseChatModel):
    """Amazon Nova-specific chat model implementation."""

    MATCHER = "amazon.nova-2-"
    PROMPT_CACHING_SUPPORTED = True
    SUPPORTED_SYSTEM_TOOLS = frozenset({"nova_grounding", "nova_code_interpreter"})
    CANONICAL_TO_BEDROCK_TOOL_MAP = MappingProxyType(
        {"web_search": "nova_grounding", "code_execution": "nova_code_interpreter"}
    )

    def _req_configure_reasoning(
        self,
        additional_request_fields: JsonMapping,
        *,
        enabled: bool,
        reasoning_effort: Effort | None = None,
        budget_tokens: int | None = None,
        max_tokens: int | None = None,  # noqa: ARG002
    ) -> None:
        """Configure Nova reasoning parameters.

        When ``enabled`` is ``False``, reasoning is explicitly disabled.

        Args:
            additional_request_fields: Mutated with the ``reasoningConfig`` entry.
            enabled: Whether reasoning is explicitly enabled.
            reasoning_effort: Effort level; defaults to ``"medium"`` when ``None``.
            budget_tokens: Not supported; raises if set.
            max_tokens: Unused.

        Raises:
            ApiError: If *budget_tokens* is not ``None``.
        """
        self._validate_no_budget_tokens(budget_tokens)
        if not enabled:
            additional_request_fields["reasoningConfig"] = {"type": "disabled"}
            return
        additional_request_fields["reasoningConfig"] = {
            "type": "enabled",
            "maxReasoningEffort": _REASONING_OVERRIDE.get(reasoning_effort, "medium"),
        }

    async def _prepare_converse_request(
        self,
        bedrock_messages: list[MessageTypeDef],
        inference_cfg: InferenceConfigurationTypeDef,
        system_blocks: list[SystemContentBlockTypeDef] | None,
        tool_config: ToolConfigurationTypeDef | None,
        additional_request_fields: dict[str, Any],
        service_tier: ServiceTierTypeType | None,
        output_config: JsonSchemaDefinitionTypeDef | None = None,
        request_metadata: dict[str, str] | None = None,
    ) -> ConverseRequestBaseTypeDef:
        """Build the Converse request, honoring the Nova high-effort constraint.

        Nova rejects requests that set ``maxTokens`` while ``reasoningConfig``
        is enabled with ``maxReasoningEffort: high`` — the token limit is
        dropped with a logged warning in that case.

        Args:
            bedrock_messages: Converted Bedrock message list.
            inference_cfg: Bedrock inference configuration.
            system_blocks: Optional top-level system instruction blocks.
            tool_config: Optional Bedrock tool configuration.
            additional_request_fields: Additional request fields.
            service_tier: Service tier configuration.
            output_config: Optional Bedrock output JSON Schema configuration.
            request_metadata: Optional key-value metadata.

        Returns:
            Bedrock Converse request payload.
        """
        reasoning = additional_request_fields.get("reasoningConfig") or {}
        if (
            reasoning.get("type") == "enabled"
            and reasoning.get("maxReasoningEffort") == "high"
            and inference_cfg.pop("maxTokens", None) is not None
        ):
            log_error_details(
                "'max_tokens' is not supported with high reasoning effort on "
                "Amazon Nova 2 models: ignored.",
                level="warning",
            )
        return await super()._prepare_converse_request(
            bedrock_messages=bedrock_messages,
            inference_cfg=inference_cfg,
            system_blocks=system_blocks,
            tool_config=tool_config,
            additional_request_fields=additional_request_fields,
            service_tier=service_tier,
            output_config=output_config,
            request_metadata=request_metadata,
        )

    @staticmethod
    def _build_code_execution_result(
        tool_use_id: str, content_items: list[Any]
    ) -> CodeExecutionToolResultBlock:
        """Build a ``CodeExecutionToolResultBlock`` from a ``nova_code_interpreter`` payload.

        Args:
            tool_use_id: Raw ``toolUseId`` from Bedrock (``tooluse_`` prefix).
            content_items: Content list from the Bedrock ``toolResult`` block or
                accumulated stream deltas.  The first item's ``"json"`` key should
                contain ``stdOut``, ``stdErr``, ``exitCode``, and ``isError``.

        Returns:
            A ``CodeExecutionToolResultBlock`` with ``srvtoolu_`` id prefix.
        """
        raw: _NovaCodeInterpreterResult = (
            content_items[0].get("json", {}) if content_items else {}
        )
        return CodeExecutionToolResultBlock(
            type="code_execution_tool_result",
            tool_use_id=f"srvtoolu_{tool_use_id.removeprefix('tooluse_')}",
            content=CodeExecutionResultBlock(
                type="code_execution_result",
                stdout=raw.get("stdOut", ""),
                stderr=raw.get("stdErr", ""),
                return_code=(raw.get("exitCode") or 1) if raw.get("isError") else 0,
                content=[],
            ),
        )

    def _resp_map_tool_result(
        self,
        tool_use_id: str,
        bedrock_tool_name: str,
        content_items: list[ToolResultContentBlockOutputTypeDef],
    ) -> list[ContentBlock] | None:
        """Map a Bedrock Nova toolResult block to Anthropic content blocks.

        Args:
            tool_use_id: The raw ``toolUseId`` from the Bedrock ``toolResult`` block.
            bedrock_tool_name: The Bedrock-side tool name.
            content_items: The full content list from the Bedrock ``toolResult`` block.

        Returns:
            A list containing a ``CodeExecutionToolResultBlock`` for
            ``nova_code_interpreter``, or the base-class result for unknown tools.
        """
        if bedrock_tool_name == "nova_code_interpreter":
            return [self._build_code_execution_result(tool_use_id, content_items)]
        return None

    def _req_map_content_block(
        self, block: ContentBlockParam
    ) -> ContentBlockTypeDef | None:
        """Map Nova-specific Anthropic request blocks to Bedrock content blocks.

        Handles ``CodeExecutionToolResultBlockParam`` blocks; delegates
        ``ServerToolUseBlockParam`` mapping to the base class via ``super()``.

        Args:
            block: An Anthropic content block param from the request messages.

        Returns:
            A Bedrock content block dict for known Nova block types, or ``None``
            to fall back to the generic adapter mapping.
        """
        if isinstance(block, CodeExecutionToolResultBlockParam):
            bedrock_id = f"tooluse_{block.tool_use_id.removeprefix('srvtoolu_')}"
            if isinstance(block.content, CodeExecutionResultBlockParam):
                json_content = {
                    "stdOut": block.content.stdout,
                    "stdErr": block.content.stderr,
                    "exitCode": block.content.return_code,
                    "isError": block.content.return_code != 0,
                }
                status: Literal["error", "success"] = (
                    "error" if block.content.return_code != 0 else "success"
                )
            else:
                json_content = {}
                status = "error"
            return {
                "toolResult": {
                    "toolUseId": bedrock_id,
                    "content": [{"json": json_content}],
                    "status": status,
                }
            }
        return super()._req_map_content_block(block)

    def _resp_stream_map_tool_result(
        self, tool_use_id: str, result_type: str, content_items: list[Any]
    ) -> ContentBlock | None:
        """Map a buffered Bedrock streaming ``toolResult`` to an Anthropic content block.

        Args:
            tool_use_id: The raw ``toolUseId`` from the Bedrock ``contentBlockStart``.
            result_type: The Bedrock result type string from the stream start event.
            content_items: The accumulated content item dicts from all
                ``contentBlockDelta`` events for this block.

        Returns:
            A ``CodeExecutionToolResultBlock`` for ``nova_code_interpreter_result``,
            or ``None`` for unknown result types.
        """
        if result_type == "nova_code_interpreter_result":
            return self._build_code_execution_result(tool_use_id, content_items)
        return None
