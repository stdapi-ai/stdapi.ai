"""Default chat model implementation using AWS Bedrock Converse API.

This module provides the default chat completion implementation that works with
all AWS Bedrock models supporting the Converse and ConverseStream APIs.
"""

from contextlib import suppress
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from sse_starlette import EventSourceResponse

from stdapi.api_errors import ApiError
from stdapi.aws import get_client
from stdapi.aws_bedrock import (
    GUARDTRAIL_CONFIG_VAR,
    PERFORMANCE_CONFIG_VAR,
    PROMPT_CACHING_DEFAULT,
    PromptCaching,
    handle_bedrock_client_error,
)
from stdapi.config import SETTINGS
from stdapi.models.chat import ChatModelBase
from stdapi.models.chat._adapters import _anthropic_message as anthropic_adapter
from stdapi.models.chat._adapters import _openai_chat_completion as openai_adapter
from stdapi.monitoring import (
    REQUEST_HEADERS,
    log_request_stream_event,
    log_response_params,
)
from stdapi.types.anthropic_messages import ToolChoiceToolParam

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime import BedrockRuntimeClient
    from types_aiobotocore_bedrock_runtime.literals import (
        CacheTTLType,
        ServiceTierTypeType,
    )
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ContentBlockTypeDef,
        InferenceConfigurationTypeDef,
        JsonSchemaDefinitionTypeDef,
        MessageTypeDef,
        SystemContentBlockTypeDef,
        ToolConfigurationTypeDef,
    )

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef
    from stdapi.models import ModelDetails
    from stdapi.types import JsonMapping
    from stdapi.types.anthropic_messages import (
        Message,
        MessageCreateParams,
        ServerTools,
        ToolUnionParam,
    )
    from stdapi.types.openai_chat_completions import (
        ChatCompletion,
        CompletionCreateParams,
        ReasoningEffort,
    )


class ChatModel(ChatModelBase[Any, Any]):
    """Default chat model using AWS Bedrock Converse API."""

    #: Whether this model supports prompt caching
    PROMPT_CACHING_SUPPORTED: ClassVar[bool] = False

    #: Whether this model supports tools prompt caching
    PROMPT_CACHING_TOOL_SUPPORTED: ClassVar[bool] = False

    #: Whether this model supports system prompt
    SYSTEM_PROMPT_SUPPORTED: ClassVar[bool] = True

    #: Maximum number of cache control blocks allowed (Bedrock limit for most models)
    MAX_CACHE_BLOCKS: ClassVar[int] = 4

    #: Whether to use simplified cache management (automatic prefix checking)
    SIMPLIFIED_CACHE_MANAGEMENT: ClassVar[bool] = False

    #: Mapping of system tool names to Bedrock system tool configurations.
    SUPPORTED_SYSTEM_TOOLS: ClassVar[MappingProxyType[ServerTools, str]] = (
        MappingProxyType({})
    )

    async def create_completion(
        self,
        model: ModelDetails,
        request: CompletionCreateParams,
        completion_id: str,
        created: int,
    ) -> ChatCompletion | EventSourceResponse:
        """Creates a completion response for the given input parameters.

        Either as a streaming response or a non-streaming one, based on the request
        configuration. Delegates OpenAI-specific translation and formatting to the
        OpenAI chat completion adapter.

        Args:
            model: The model details to use for generating the completion.
            request: The parameters defining the completion request.
            completion_id: A unique identifier for the completion request.
            created: The timestamp indicating when the request was created.

        Returns:
            Completed response or streaming event source response based on the configuration.
        """
        bedrock_messages, system_blocks = await openai_adapter.map_messages(
            request.messages
        )

        (
            inference_cfg,
            additional_request_fields,
            tool_config,
            system_tools,
            bedrock_service_tier,
            openai_service_tier,
            choices_count,
        ) = openai_adapter.translate_request(request, self._model_id)

        if system_tools:
            tool_config = self._req_configure_system_tools(
                tool_config=tool_config,
                system_tools=system_tools,
                additional_request_fields=additional_request_fields,
            )

        if request.reasoning_effort not in (None, "none") or request.enable_thinking:
            self._req_configure_reasoning(
                additional_request_fields=additional_request_fields,
                reasoning_effort=request.reasoning_effort,
                budget_tokens=request.thinking_budget,
                max_tokens=request.max_completion_tokens or request.max_tokens,
            )

        self._req_enable_prompt_caching(
            system_blocks=system_blocks,
            tool_config=tool_config,
            bedrock_messages=bedrock_messages,
            prompt_caching=openai_adapter.parse_prompt_cache_key(
                request.prompt_cache_key
            ),
            prompt_caching_ttl=openai_adapter.CACHE_TTL.get(
                request.prompt_cache_retention
            ),
        )

        bedrock_runtime, bedrock_request = await self._prepare_converse_request(
            model=model,
            bedrock_messages=bedrock_messages,
            inference_cfg=inference_cfg,
            system_blocks=system_blocks,
            tool_config=tool_config,
            additional_request_fields=additional_request_fields,
            service_tier=bedrock_service_tier,
        )
        if request.stream:
            return EventSourceResponse(
                await log_request_stream_event(
                    openai_adapter.format_stream(
                        completion_id,
                        created,
                        self._model_id,
                        bedrock_runtime,
                        bedrock_request,
                        openai_service_tier,
                        include_usage=(
                            request.stream_options is not None
                            and request.stream_options.include_usage is True
                        ),
                    )
                )
            )
        return await openai_adapter.format_response(
            completion_id,
            created,
            self._model_id,
            bedrock_runtime,
            bedrock_request,
            openai_service_tier,
            choices_count,
            request.audio,
            request.modalities or openai_adapter.DEFAULT_OUTPUT_MODALITIES,  # type: ignore[arg-type]
        )

    async def create_message(
        self, model: ModelDetails, request: MessageCreateParams, message_id: str
    ) -> Message | EventSourceResponse:
        """Create a message using Anthropic Messages API format.

        Translates the Anthropic request to Bedrock Converse, executes it,
        and formats the response back to Anthropic format.

        Args:
            model: Model details for the chat model.
            request: Message creation request following Anthropic spec.
            message_id: Stable identifier for the message.

        Returns:
            Message when stream is False, EventSourceResponse when stream is True.
        """
        (
            bedrock_messages,
            system_blocks,
            inference_cfg,
            additional_request_fields,
            tool_config,
            service_tier,
            prompt_caching,
            prompt_caching_ttl,
            output_config,
            system_tools,
        ) = await anthropic_adapter.translate_request(
            request,
            self._model_id,
            prompt_caching_supported=self.PROMPT_CACHING_SUPPORTED,
            prompt_caching_tool_supported=self.PROMPT_CACHING_TOOL_SUPPORTED,
        )

        if system_tools:
            tool_config = self._req_configure_system_tools(
                tool_config=tool_config,
                system_tools=system_tools,
                additional_request_fields=additional_request_fields,
            )

        if request.thinking is not None and request.thinking.type != "disabled":
            self._req_configure_reasoning(
                additional_request_fields=additional_request_fields,
                budget_tokens=request.thinking.budget_tokens
                if request.thinking.type == "enabled"
                else None,
                max_tokens=request.max_tokens,
            )

        if prompt_caching is not None:
            self._req_enable_prompt_caching(
                system_blocks=system_blocks,
                tool_config=tool_config,
                bedrock_messages=bedrock_messages,
                prompt_caching=prompt_caching,
                prompt_caching_ttl=prompt_caching_ttl,
            )

        forced_tool = (
            request.tool_choice.name
            if isinstance(request.tool_choice, ToolChoiceToolParam)
            else None
        )

        bedrock_runtime, bedrock_request = await self._prepare_converse_request(
            model=model,
            bedrock_messages=bedrock_messages,
            inference_cfg=inference_cfg,
            system_blocks=system_blocks,
            tool_config=tool_config,
            additional_request_fields=additional_request_fields,
            service_tier=service_tier,
            output_config=output_config,
        )

        if request.stream:
            with handle_bedrock_client_error():
                bedrock_stream = (
                    await bedrock_runtime.converse_stream(**bedrock_request)
                )["stream"]
            return EventSourceResponse(
                await log_request_stream_event(
                    anthropic_adapter.format_stream(
                        message_id, request.model, bedrock_stream, forced_tool
                    )
                )
            )

        with handle_bedrock_client_error():
            response = await bedrock_runtime.converse(**bedrock_request)

        return log_response_params(
            await anthropic_adapter.format_response(
                response["output"]["message"]["content"],
                response["stopReason"],
                response["usage"],  # type: ignore[arg-type]
                message_id,
                request.model,
                forced_tool,
            )
        )

    async def _prepare_converse_request(
        self,
        model: ModelDetails,
        bedrock_messages: list[MessageTypeDef],
        inference_cfg: InferenceConfigurationTypeDef,
        system_blocks: list[SystemContentBlockTypeDef],
        tool_config: ToolConfigurationTypeDef | None,
        additional_request_fields: dict[str, Any],
        service_tier: ServiceTierTypeType | None,
        output_config: JsonSchemaDefinitionTypeDef | None = None,
    ) -> tuple[BedrockRuntimeClient, ConverseRequestBaseTypeDef]:
        """Prepare a Bedrock Converse request payload and client.

        Args:
            model: Model details.
            bedrock_messages: Converted Bedrock message list.
            inference_cfg: Bedrock inference configuration.
            system_blocks: Optional top-level system instruction blocks.
            tool_config: Optional Bedrock tool configuration.
            additional_request_fields: Additional request fields.
            service_tier: Service tier configuration.
            output_config: Optional Bedrock output JSON Schema configuration.

        Returns:
            Tuple of (BedrockRuntimeClient, request payload dict).
        """
        latency, default_service_tier = PERFORMANCE_CONFIG_VAR.get()
        request: ConverseRequestBaseTypeDef = {
            "modelId": model.get_id(inference_profile=True),
            "messages": bedrock_messages,
            "inferenceConfig": inference_cfg,
        }
        if system_blocks and (
            self.SYSTEM_PROMPT_SUPPORTED or not SETTINGS.drop_unsupported_system_prompt
        ):
            request["system"] = system_blocks
        if tool_config:
            request["toolConfig"] = tool_config
        if additional_request_fields := self._prepare_additional_request_fields(
            additional_request_fields
        ):
            request["additionalModelRequestFields"] = additional_request_fields
        if service_tier := (service_tier or default_service_tier):
            request["serviceTier"] = {"type": service_tier}
        if latency:
            request["performanceConfig"] = {"latency": latency}
        if output_config:
            request["outputConfig"] = {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {"jsonSchema": output_config},
                }
            }

        with suppress(LookupError):
            request["guardrailConfig"] = GUARDTRAIL_CONFIG_VAR.get()
        return get_client("bedrock-runtime", model.region), request

    def _get_passthrough_header_fields(self) -> dict[str, Any]:
        """Extract additional request fields from passthrough HTTP headers.

        Reads the current request headers and transforms matching entries
        according to :attr:`PASSTHROUGH_HEADERS`.

        Returns:
            A dict of field names to transformed header values.
        """
        if not self.PASSTHROUGH_HEADERS:
            return {}
        headers = REQUEST_HEADERS.get()
        return {
            field_name: transform(headers[header_name])
            for header_name, (field_name, transform) in self.PASSTHROUGH_HEADERS.items()
            if header_name in headers
        }

    def _prepare_additional_request_fields(
        self, additional_request_fields: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge passthrough headers into additional request fields.

        Combines header-derived fields with the existing
        ``additional_request_fields``.  Header values have lower priority:
        if a key already exists in *additional_request_fields* (from request
        body or default model params), it is kept.

        Subclasses may override this to apply model-specific transformations
        (e.g. filtering unsupported ``anthropic_beta`` flags).

        Args:
            additional_request_fields: Fields already collected from the
                request body and default model parameters.

        Returns:
            The merged additional request fields dict.
        """
        if header_fields := self._get_passthrough_header_fields():
            return header_fields | additional_request_fields
        return additional_request_fields

    def _req_configure_reasoning(
        self,
        additional_request_fields: dict[str, Any],  # noqa: ARG002
        reasoning_effort: ReasoningEffort | None = None,
        budget_tokens: int | None = None,
        max_tokens: int | None = None,  # noqa: ARG002
    ) -> None:
        """Configure reasoning parameters for the model.

        Default implementation raises an error. Models that support reasoning
        must override this method.

        Args:
            additional_request_fields: Request fields to modify with reasoning config.
            reasoning_effort: The reasoning effort level.
            budget_tokens: Optional token budget for reasoning.
            max_tokens: Maximum number of tokens allowed for the model.

        Raises:
            ApiError: If reasoning is not supported by this model.
        """
        if reasoning_effort is not None or budget_tokens is not None:
            msg = "Reasoning configuration is not supported for this model"
            raise ApiError(msg)

    def _req_enable_prompt_caching(
        self,
        system_blocks: list[SystemContentBlockTypeDef],
        tool_config: ToolConfigurationTypeDef | None,
        bedrock_messages: list[MessageTypeDef],
        prompt_caching: set[PromptCaching] | frozenset[PromptCaching],
        prompt_caching_ttl: CacheTTLType | None,
    ) -> None:
        """Enables explicit prompt caching for specified components including system blocks, tools, and messages.

        Note: This is for explicit cache breakpoints only. For automatic caching (top-level cache_control),
        use _req_apply_automatic_caching instead.

        Supports two modes:
        1. Simplified Cache Management (when SIMPLIFIED_CACHE_MANAGEMENT=True):
           - Places a single cache checkpoint at the end of static content
           - Bedrock automatically checks for cache hits at previous content block boundaries
           - Looks back up to ~20 blocks from the checkpoint
           - Ideal for most use cases with dynamic content

        2. Multiple Cache Checkpoints (when SIMPLIFIED_CACHE_MANAGEMENT=False):
           - Places cache checkpoints at each specified component (system, tools, messages)
           - Respects MAX_CACHE_BLOCKS limit (default 4 for Claude models)
           - Provides granular control over cache boundaries
           - Best for content that changes at different frequencies

        Args:
            system_blocks: System content blocks to append cache point to.
            tool_config: Tool configuration to append cache point to.
            bedrock_messages: Message list to append cache points to.
            prompt_caching: Set of components where caching should be enabled.
            prompt_caching_ttl: Prompt caching TTL configuration.
        """
        if self.PROMPT_CACHING_SUPPORTED:
            cache_point: ContentBlockTypeDef = (
                {"cachePoint": {"type": "default", "ttl": prompt_caching_ttl}}
                if prompt_caching_ttl
                else PROMPT_CACHING_DEFAULT
            )
            if self.SIMPLIFIED_CACHE_MANAGEMENT:
                self._apply_simplified_caching(
                    system_blocks,
                    tool_config,
                    bedrock_messages,
                    prompt_caching,
                    cache_point,
                )
            else:
                self._apply_multiple_cache_checkpoints(
                    system_blocks,
                    tool_config,
                    bedrock_messages,
                    prompt_caching,
                    cache_point,
                )

    def _apply_simplified_caching(
        self,
        system_blocks: list[SystemContentBlockTypeDef],
        tool_config: ToolConfigurationTypeDef | None,
        bedrock_messages: list[MessageTypeDef],
        prompt_caching: set[PromptCaching] | frozenset[PromptCaching],
        cache_point: ContentBlockTypeDef,
    ) -> None:
        """Apply simplified cache management with a single checkpoint at the end of static content.

        Places one cache checkpoint at the highest priority available location (messages > tools > system).
        Bedrock automatically checks for cache hits at previous content block boundaries.

        Args:
            system_blocks: System content blocks to append cache point to.
            tool_config: Tool configuration to append cache point to.
            bedrock_messages: Message list to append cache point to.
            prompt_caching: Set of components where caching should be enabled.
            cache_point: Cache point block to append.
        """
        if "messages" in prompt_caching and bedrock_messages:
            bedrock_messages[-1]["content"].append(cache_point)  # type: ignore[attr-defined]
        elif (
            "tools" in prompt_caching
            and tool_config
            and self.PROMPT_CACHING_TOOL_SUPPORTED
        ):
            tool_config["tools"].append(cache_point)  # type: ignore[attr-defined]
        elif "system" in prompt_caching and system_blocks:
            system_blocks.append(cache_point)

    def _apply_multiple_cache_checkpoints(
        self,
        system_blocks: list[SystemContentBlockTypeDef],
        tool_config: ToolConfigurationTypeDef | None,
        bedrock_messages: list[MessageTypeDef],
        prompt_caching: set[PromptCaching] | frozenset[PromptCaching],
        cache_point: ContentBlockTypeDef,
    ) -> None:
        """Apply multiple cache checkpoints with granular control respecting MAX_CACHE_BLOCKS limit.

        Places cache checkpoints at each specified component (system, tools, messages) up to the
        MAX_CACHE_BLOCKS limit for fine-grained cache boundary control.

        Args:
            system_blocks: System content blocks to append cache point to.
            tool_config: Tool configuration to append cache point to.
            bedrock_messages: Message list to append cache points to.
            prompt_caching: Set of components where caching should be enabled.
            cache_point: Cache point block to append.
        """
        cache_blocks_used = 0

        if "system" in prompt_caching and system_blocks:
            system_blocks.append(cache_point)
            cache_blocks_used += 1

        if (
            "tools" in prompt_caching
            and tool_config
            and self.PROMPT_CACHING_TOOL_SUPPORTED
            and cache_blocks_used < self.MAX_CACHE_BLOCKS
        ):
            tool_config["tools"].append(cache_point)  # type: ignore[attr-defined]
            cache_blocks_used += 1

        if "messages" in prompt_caching and bedrock_messages:
            for message in bedrock_messages[
                : self.MAX_CACHE_BLOCKS - cache_blocks_used
            ]:
                message["content"].append(cache_point)  # type: ignore[attr-defined]

    def _req_configure_system_tools(
        self,
        tool_config: ToolConfigurationTypeDef | None,
        system_tools: list[ToolUnionParam],
        additional_request_fields: JsonMapping,  # noqa: ARG002
    ) -> ToolConfigurationTypeDef | None:
        """Configure system tools for the Bedrock request.

        Validates that each system tool is supported by this model and adds
        them to the Bedrock tool configuration as ``systemTool`` entries
        inside ``toolConfig.tools[]``.

        Subclasses may override this to use a different mechanism (e.g.,
        Anthropic Claude models inject tools via ``additionalModelRequestFields``).

        Args:
            tool_config: Existing Bedrock tool configuration, or ``None``.
            system_tools: List of Anthropic system tool params.
            additional_request_fields: Mutable dict of additional request fields
                for the Bedrock Converse API. Subclasses may mutate this.

        Returns:
            Updated Bedrock tool configuration with system tools added.

        Raises:
            ApiError: If a system tool is not supported by this model.
        """
        if tool_config is None:
            tool_config = {"tools": []}
        for tool in system_tools:
            tool_name = getattr(tool, "name", None)
            if tool_name is None or tool_name not in self.SUPPORTED_SYSTEM_TOOLS:
                tool_type = getattr(tool, "type", type(tool).__name__)
                msg = f"System tool '{tool_type}' is not supported by this model."
                raise ApiError(msg)
            tool_config["tools"].append(  # type: ignore[attr-defined]
                {"systemTool": {"name": self.SUPPORTED_SYSTEM_TOOLS[tool_name]}}
            )
        return tool_config

    @staticmethod
    def _validate_no_budget_tokens(budget_tokens: int | None) -> None:
        """Validates that budget tokens are not provided.

        Args:
            budget_tokens: Budget tokens value to validate.

        Raises:
            ApiError: If budget tokens are provided.
        """
        if budget_tokens is not None:
            msg = "This model do not support 'thinking_budget'. Use 'reasoning_effort' instead."
            raise ApiError(msg)
