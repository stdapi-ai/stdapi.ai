"""Default chat model implementation using AWS Bedrock Converse API.

This module provides the default chat completion implementation that works with
all AWS Bedrock models supporting the Converse and ConverseStream APIs.
"""

from asyncio import gather
from contextlib import suppress
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from sse_starlette import EventSourceResponse, JSONServerSentEvent

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock import (
    GUARDRAIL_CONFIG_VAR,
    PROMPT_CACHING_DEFAULT,
    PromptCaching,
)
from stdapi.config import SETTINGS
from stdapi.input_file import prefetch_all_content_types
from stdapi.models.capabilities import Capability
from stdapi.models.chat import ChatModelBase
from stdapi.models.chat._adapters import _anthropic_message as anthropic_adapter
from stdapi.models.chat._adapters import _openai_chat_completion as openai_adapter
from stdapi.models.chat._adapters import _openai_common
from stdapi.models.chat._adapters import _openai_completion as text_completion_adapter
from stdapi.models.chat._adapters import _openai_responses as responses_adapter
from stdapi.monitoring import REQUEST, log_request_sse_stream_event, log_response_params
from stdapi.types.anthropic_messages import (
    ServerToolUseBlock,
    ServerToolUseBlockParam,
    ToolChoiceToolParam,
    ToolParam,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable

    from types_aiobotocore_bedrock_runtime.literals import (
        CacheTTLType,
        ServiceTierTypeType,
    )
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ContentBlockTypeDef,
        ConverseResponseTypeDef,
        InferenceConfigurationTypeDef,
        JsonSchemaDefinitionTypeDef,
        MessageTypeDef,
        SystemContentBlockTypeDef,
        ToolConfigurationTypeDef,
        ToolResultContentBlockOutputTypeDef,
        ToolTypeDef,
    )

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef
    from stdapi.models.chat import Effort
    from stdapi.types import JsonMapping
    from stdapi.types.anthropic_messages import (
        ContentBlock,
        ContentBlockParam,
        Message,
        MessageCreateParams,
        ServerTools,
    )
    from stdapi.types.openai import ResponseModeration
    from stdapi.types.openai_chat_completions import ChatCompletion
    from stdapi.types.openai_chat_completions import (
        CompletionCreateParams as ChatCompletionCreateParams,
    )
    from stdapi.types.openai_completions import Completion, CompletionCreateParams
    from stdapi.types.openai_responses import Response, ResponseCreateParams


def _synthesize_tool_config_from_history(
    messages: list[MessageTypeDef],
) -> ToolConfigurationTypeDef | None:
    """Synthesize a permissive tool config from ``toolUse`` blocks in history.

    Bedrock Converse rejects a request that carries ``toolUse``/``toolResult``
    content blocks without a ``toolConfig``.  OpenAI clients routinely omit
    ``tools`` on the final round-trip turn (and ``tool_choice='none'`` drops the
    config entirely), so when no config is otherwise present a minimal one is
    built: one ``toolSpec`` per distinct tool name found in history, each with a
    permissive ``{"type": "object"}`` input schema and no ``toolChoice``.

    Args:
        messages: Converted Bedrock message history.

    Returns:
        A synthesized tool configuration, or ``None`` if history contains no
        ``toolUse`` blocks.
    """
    names = sorted(
        {
            block["toolUse"]["name"]
            for message in messages
            for block in message.get("content", ())
            if "toolUse" in block
        }
    )
    if not names:
        return None
    return {
        "tools": [
            {"toolSpec": {"name": name, "inputSchema": {"json": {"type": "object"}}}}
            for name in names
        ]
    }


class ChatModel(ChatModelBase[Any, Any]):
    """Default chat model using AWS Bedrock Converse API."""

    #: Prompt caching supported.
    PROMPT_CACHING_SUPPORTED: ClassVar[bool] = False

    #: Tool-list prompt caching supported.
    PROMPT_CACHING_TOOL_SUPPORTED: ClassVar[bool] = False

    #: System prompt supported.
    SYSTEM_PROMPT_SUPPORTED: ClassVar[bool] = True

    #: System-role messages in the messages list are forwarded to Bedrock as-is
    #: (mid-conversation system instructions, Claude Opus 4.8+).
    #: When False (default), they are extracted and merged into the system prompt field.
    SYSTEM_MESSAGE_AS_MESSAGES_SUPPORTED: ClassVar[bool] = False

    #: Maximum cache control blocks (Bedrock limit).
    MAX_CACHE_BLOCKS: ClassVar[int] = 4

    #: Use simplified cache management (single checkpoint, Bedrock auto-lookback).
    SIMPLIFIED_CACHE_MANAGEMENT: ClassVar[bool] = False

    #: Bedrock system tool names eligible for auto-promotion from ``toolSpec``.
    SUPPORTED_SYSTEM_TOOLS: ClassVar[frozenset[str]] = frozenset()

    #: Canonical (Anthropic-style) server tool name → Bedrock system tool name; consulted on all routes.
    CANONICAL_TO_BEDROCK_TOOL_MAP: ClassVar[MappingProxyType[ServerTools, str]] = (
        MappingProxyType({})
    )

    @classmethod
    def get_supported_operations(cls) -> Capability:
        """Return capability flags for route-based model matching.

        Returns:
            Capability flags. Converse models support token counting via the
            Bedrock CountTokens API (unavailable on Bedrock Mantle models).
        """
        return Capability.COUNT_TOKENS

    async def create_completion(
        self, request: ChatCompletionCreateParams, completion_id: str, created: int
    ) -> ChatCompletion | EventSourceResponse:
        """Handle a chat completion request via the OpenAI route.

        Args:
            request: OpenAI-format completion request.
            completion_id: Unique identifier for the completion.
            created: Unix timestamp of request creation.

        Returns:
            Completed response or streaming ``EventSourceResponse``.
        """
        await prefetch_all_content_types()
        bedrock_messages, system_blocks = await openai_adapter.map_messages(
            request.messages
        )

        (
            inference_cfg,
            additional_request_fields,
            tool_config,
            bedrock_service_tier,
            openai_service_tier,
            choices_count,
            output_config,
            request_metadata,
        ) = openai_adapter.translate_request(request, self._model_id)

        server_tools = self._req_extract_server_tools(tool_config)
        tool_config = self._req_promote_system_tools(tool_config)
        self._req_configure_tools(
            tool_config=tool_config,
            additional_request_fields=additional_request_fields,
            server_tools=server_tools,
            bedrock_messages=bedrock_messages,
        )

        if reasoning := openai_adapter.extract_reasoning(request):
            self._req_configure_reasoning(
                additional_request_fields=additional_request_fields, **reasoning
            )

        self._req_enable_prompt_caching(
            system_blocks=system_blocks,
            tool_config=tool_config,
            bedrock_messages=bedrock_messages,
            prompt_caching=_openai_common.parse_prompt_cache_key(
                request.prompt_cache_key
            ),
            prompt_caching_ttl=_openai_common.CACHE_TTL.get(
                request.prompt_cache_retention
            ),
        )

        bedrock_request = await self._prepare_converse_request(
            bedrock_messages=bedrock_messages,
            inference_cfg=inference_cfg,
            system_blocks=system_blocks,
            tool_config=tool_config,
            additional_request_fields=additional_request_fields,
            service_tier=bedrock_service_tier,
            output_config=output_config,
            request_metadata=request_metadata,
        )
        if request.stream:
            return EventSourceResponse(
                log_request_sse_stream_event(
                    openai_adapter.format_stream(
                        completion_id,
                        created,
                        self._model_id,
                        (await self.converse_stream(bedrock_request))["stream"],
                        openai_service_tier,
                        include_usage=(
                            request.stream_options is not None
                            and request.stream_options.include_usage is True
                        ),
                        suppress_tool_names=self.SUPPORTED_SYSTEM_TOOLS or None,
                    )
                )
            )
        return await openai_adapter.format_response(
            completion_id,
            created,
            self._model_id,
            await gather(
                *(self.converse(bedrock_request) for _ in range(choices_count))
            ),
            openai_service_tier,
            request.audio,
            request.modalities or openai_adapter.DEFAULT_OUTPUT_MODALITIES,  # type: ignore[arg-type]
            self.SUPPORTED_SYSTEM_TOOLS or None,
        )

    async def create_text_completion(
        self, request: CompletionCreateParams, completion_id: str, created: int
    ) -> Completion | EventSourceResponse:
        """Handle a completion request (OpenAI ``POST /v1/completions``).

        Args:
            request: Completion creation request following the OpenAI spec.
            completion_id: Stable identifier for the completion.
            created: Unix timestamp (seconds) of the request.

        Returns:
            ``Completion`` when ``stream`` is ``False``, otherwise an
            ``EventSourceResponse`` streaming completion chunks.
        """
        await prefetch_all_content_types()
        user_messages = await text_completion_adapter.build_user_messages(
            request.prompt
        )

        (
            inference_cfg,
            additional_request_fields,
            bedrock_service_tier,
            openai_service_tier,
            n,
            request_metadata,
        ) = text_completion_adapter.translate_request(request, self._model_id)

        prompt_caching = _openai_common.parse_prompt_cache_key(request.prompt_cache_key)
        prompt_caching_ttl = _openai_common.CACHE_TTL.get(
            request.prompt_cache_retention
        )

        bedrock_requests: list[ConverseRequestBaseTypeDef] = []
        for user_message in user_messages:
            messages = [user_message]
            if prompt_caching:
                self._req_enable_prompt_caching(
                    system_blocks=None,
                    tool_config=None,
                    bedrock_messages=messages,
                    prompt_caching=prompt_caching,
                    prompt_caching_ttl=prompt_caching_ttl,
                )
            bedrock_requests.append(
                await self._prepare_converse_request(
                    bedrock_messages=messages,
                    inference_cfg=inference_cfg,
                    system_blocks=None,
                    tool_config=None,
                    additional_request_fields=additional_request_fields,
                    service_tier=bedrock_service_tier,
                    request_metadata=request_metadata,
                )
            )

        if request.stream:
            stream_responses = await gather(
                *(
                    self.converse_stream(req)
                    for req in bedrock_requests
                    for _ in range(n)
                )
            )
            return EventSourceResponse(
                log_request_sse_stream_event(
                    text_completion_adapter.format_stream(
                        completion_id,
                        created,
                        self._model_id,
                        [r["stream"] for r in stream_responses],
                        openai_service_tier,
                        include_usage=(
                            request.stream_options is not None
                            and request.stream_options.include_usage is True
                        ),
                    )
                )
            )

        responses: list[ConverseResponseTypeDef] = await gather(
            *(self.converse(req) for req in bedrock_requests for _ in range(n))
        )
        return text_completion_adapter.format_response(
            completion_id, created, self._model_id, responses, openai_service_tier
        )

    async def create_message(
        self, request: MessageCreateParams, message_id: str
    ) -> Message | EventSourceResponse:
        """Handle a message request via the Anthropic Messages route.

        Args:
            request: Anthropic-format message creation request.
            message_id: Stable identifier for the message.

        Returns:
            ``Message`` or streaming ``EventSourceResponse``.
        """
        await prefetch_all_content_types()
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
        ) = await anthropic_adapter.translate_request(
            request,
            self._model_id,
            prompt_caching_supported=self.PROMPT_CACHING_SUPPORTED,
            prompt_caching_tool_supported=self.PROMPT_CACHING_TOOL_SUPPORTED,
            tool_name_map=self.CANONICAL_TO_BEDROCK_TOOL_MAP,
            req_map_content_block=self._req_map_content_block,
            system_message_as_messages=self.SYSTEM_MESSAGE_AS_MESSAGES_SUPPORTED,
        )

        tool_config = self._req_promote_system_tools(tool_config)
        self._req_configure_tools(
            tool_config=tool_config,
            additional_request_fields=additional_request_fields,
            server_tools=[
                t.model_dump(exclude_none=True)
                for t in (request.tools or ())
                if not isinstance(t, ToolParam)
            ],
            bedrock_messages=bedrock_messages,
        )

        if reasoning := anthropic_adapter.extract_reasoning(request):
            self._req_configure_reasoning(
                additional_request_fields=additional_request_fields, **reasoning
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

        bedrock_request = await self._prepare_converse_request(
            bedrock_messages=bedrock_messages,
            inference_cfg=inference_cfg,
            system_blocks=system_blocks,
            tool_config=tool_config,
            additional_request_fields=additional_request_fields,
            service_tier=service_tier,
            output_config=output_config,
        )

        if request.stream:
            return EventSourceResponse(
                log_request_sse_stream_event(
                    anthropic_adapter.format_stream(
                        message_id,
                        request.model,
                        (await self.converse_stream(bedrock_request))["stream"],
                        forced_tool,
                        self._resp_stream_map_tool_use,
                        self._resp_stream_map_tool_result,
                    )
                )
            )

        response = await self.converse(bedrock_request)
        return log_response_params(
            await anthropic_adapter.format_response(
                response["output"]["message"]["content"],
                response["stopReason"],
                response["usage"],  # type: ignore[arg-type]
                message_id,
                request.model,
                forced_tool,
                self._resp_map_tool_result,
                self._resp_map_tool_use,
            )
        )

    async def create_response(
        self,
        request: ResponseCreateParams,
        response_id: str,
        created_at: float,
        moderation_builder: Callable[[], ResponseModeration | None] | None = None,
    ) -> Response | EventSourceResponse:
        """Handle a response request via the OpenAI Responses route.

        Args:
            request: Responses API creation request.
            response_id: Unique identifier for the response.
            created_at: Unix timestamp of request creation.
            moderation_builder: Optional callable building the response
                ``moderation`` field, invoked once the full guardrail trace
                is available (at stream end when streaming).

        Returns:
            Completed response or streaming ``EventSourceResponse``.
        """
        await prefetch_all_content_types()

        bedrock_messages, system_blocks = await responses_adapter.map_input(
            request.input, request.instructions
        )

        (
            inference_cfg,
            additional_request_fields,
            tool_config,
            output_config,
            service_tier,
            prompt_caching,
            prompt_caching_ttl,
            request_metadata,
        ) = responses_adapter.translate_request(
            request,
            self._model_id,
            tool_name_map=self.CANONICAL_TO_BEDROCK_TOOL_MAP or None,
        )

        server_tools = self._req_extract_server_tools(tool_config)
        tool_config = self._req_promote_system_tools(tool_config)
        self._req_configure_tools(
            tool_config=tool_config,
            additional_request_fields=additional_request_fields,
            server_tools=server_tools,
            bedrock_messages=bedrock_messages,
        )

        if reasoning := responses_adapter.extract_reasoning(request):
            self._req_configure_reasoning(
                additional_request_fields=additional_request_fields, **reasoning
            )

        if prompt_caching:
            self._req_enable_prompt_caching(
                system_blocks=system_blocks,
                tool_config=tool_config,
                bedrock_messages=bedrock_messages,
                prompt_caching=prompt_caching,
                prompt_caching_ttl=prompt_caching_ttl,
            )

        bedrock_request = await self._prepare_converse_request(
            bedrock_messages=bedrock_messages,
            inference_cfg=inference_cfg,
            system_blocks=system_blocks,
            tool_config=tool_config,
            additional_request_fields=additional_request_fields,
            service_tier=service_tier,
            output_config=output_config,
            request_metadata=request_metadata,
        )

        web_search_names: frozenset[str] | None = (
            frozenset({ws_name})
            if (ws_name := self.CANONICAL_TO_BEDROCK_TOOL_MAP.get("web_search"))
            else None
        )
        suppress_names = (
            self.SUPPORTED_SYSTEM_TOOLS - (web_search_names or frozenset())
        ) or None
        image_gen_tool = responses_adapter.get_image_generation_tool(request)

        if request.stream:
            suppress_with_img: frozenset[str] | None = suppress_names
            post_handler: (
                Callable[
                    [responses_adapter._StreamState],
                    AsyncGenerator[JSONServerSentEvent],
                ]
                | None
            ) = None
            if image_gen_tool:
                suppress_with_img = (suppress_names or frozenset()) | {
                    "image_generation"
                }
                post_handler = partial(
                    responses_adapter.image_generation_stream_handler,
                    image_gen_tool=image_gen_tool,
                    response_id=response_id,
                    fallback_model=SETTINGS.image_generation_model,
                )

            return EventSourceResponse(
                log_request_sse_stream_event(
                    responses_adapter.format_stream(
                        response_id,
                        created_at,
                        self._model_id,
                        (await self.converse_stream(bedrock_request))["stream"],
                        request,
                        suppress_with_img,
                        post_handler,
                        web_search_names,
                        moderation_builder,
                    )
                )
            )

        response = await responses_adapter.format_response(
            response_id,
            created_at,
            self._model_id,
            await self.converse(bedrock_request),
            request,
            suppress_names,
            web_search_names,
        )
        if image_gen_tool:
            response.output = await responses_adapter.execute_image_generation_calls(
                response.output,
                image_gen_tool,
                response_id,
                SETTINGS.image_generation_model,
            )
        if moderation_builder is not None:
            response.moderation = moderation_builder()
        return response

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
        """Build a Bedrock Converse request payload.

        Assembles all request fields from the translated inputs.  Region
        selection and the actual AWS call are delegated to
        :meth:`~stdapi.models.ModelBase.converse` /
        :meth:`~stdapi.models.ModelBase.converse_stream`, which overwrite the
        ``modelId`` placeholder with the final region-specific value.

        Args:
            bedrock_messages: Converted Bedrock message list.
            inference_cfg: Bedrock inference configuration.
            system_blocks: Optional top-level system instruction blocks.
            tool_config: Optional Bedrock tool configuration.
            additional_request_fields: Additional request fields.
            service_tier: Service tier configuration.
            output_config: Optional Bedrock output JSON Schema configuration.
            request_metadata: Optional key-value metadata forwarded as Bedrock ``requestMetadata``.

        Returns:
            Bedrock Converse request payload with ``modelId`` set to an empty
            string placeholder; the routing layer fills in the real value.
        """
        request: ConverseRequestBaseTypeDef = {
            "modelId": "",  # placeholder — overwritten by converse()/converse_stream()
            "messages": bedrock_messages,
            "inferenceConfig": inference_cfg,
        }
        if system_blocks and (
            self.SYSTEM_PROMPT_SUPPORTED or not SETTINGS.drop_unsupported_system_prompt
        ):
            request["system"] = system_blocks
        if tool_config:
            request["toolConfig"] = tool_config
        elif synthesized_tool_config := _synthesize_tool_config_from_history(
            bedrock_messages
        ):
            request["toolConfig"] = synthesized_tool_config
        if additional_request_fields := self._prepare_additional_request_fields(
            additional_request_fields
        ):
            request["additionalModelRequestFields"] = additional_request_fields
        if service_tier:
            # Fallback chain and performanceConfig are applied later, in ModelBase._prepare_converse_request_for_region.
            request["serviceTier"] = {"type": service_tier}
        if output_config:
            request["outputConfig"] = {
                "textFormat": {
                    "type": "json_schema",
                    "structure": {"jsonSchema": output_config},
                }
            }
        if request_metadata:
            request["requestMetadata"] = request_metadata
        with suppress(LookupError):
            request["guardrailConfig"] = GUARDRAIL_CONFIG_VAR.get()
        return request

    def _get_passthrough_header_fields(self) -> dict[str, Any]:
        """Extract additional request fields from passthrough HTTP headers.

        Reads the current request headers and transforms matching entries
        according to :attr:`PASSTHROUGH_HEADERS`.

        Returns:
            A dict of field names to transformed header values.
        """
        if not self.PASSTHROUGH_HEADERS:
            return {}
        headers = REQUEST.get().headers
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
        additional_request_fields: dict[str, Any],
        *,
        enabled: bool,
        reasoning_effort: Effort | None = None,
        budget_tokens: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        """Override this method in model subclasses that support reasoning.

        Args:
            additional_request_fields: Mutable in overrides.
            enabled: Whether reasoning is explicitly enabled.
            reasoning_effort: Reasoning effort level.
            budget_tokens: Explicit token budget for reasoning.
            max_tokens: Used by overrides to derive budget.

        Raises:
            ApiError: If *enabled* is ``True`` and *reasoning_effort* or *budget_tokens*
                is not ``None``.
        """

    def _resp_map_tool_result(
        self,
        tool_use_id: str,
        bedrock_tool_name: str,
        content_items: list[ToolResultContentBlockOutputTypeDef],
    ) -> list[ContentBlock] | None:
        """Map a Bedrock toolResult block to zero or more Anthropic content blocks.

        Args:
            tool_use_id: The raw ``toolUseId`` from the Bedrock ``toolResult`` block.
            bedrock_tool_name: The Bedrock-side tool name (e.g. ``"nova_code_interpreter"``).
            content_items: The full content list from the Bedrock ``toolResult`` block.

        Returns:
            A list of Anthropic content blocks  or ``None`` to drop the result.
        """

    def _canonical_name_for(self, bedrock_tool_name: str) -> ServerTools | None:
        """Return the canonical (Anthropic-style) server-tool name for *bedrock_tool_name*.

        Inverse lookup over ``CANONICAL_TO_BEDROCK_TOOL_MAP``.

        Args:
            bedrock_tool_name: Bedrock-side tool name.

        Returns:
            Matching canonical server-tool key, or ``None`` if unmapped.
        """
        return next(
            (
                k
                for k, v in self.CANONICAL_TO_BEDROCK_TOOL_MAP.items()
                if v == bedrock_tool_name
            ),
            None,
        )

    def _resp_map_tool_use(
        self, tool_use_id: str, bedrock_tool_name: str, tool_input: JsonMapping
    ) -> ContentBlock | None:
        """Map a Bedrock ``toolUse`` block to an Anthropic content block.

        Args:
            tool_use_id: Raw ``toolUseId`` from the Bedrock ``toolUse`` block.
            bedrock_tool_name: Bedrock-side tool name.
            tool_input: Tool input dict from the Bedrock ``toolUse`` block.

        Returns:
            Anthropic content block, or ``None`` for unmapped tools.
        """
        if canonical_name := self._canonical_name_for(bedrock_tool_name):
            return ServerToolUseBlock(
                type="server_tool_use",
                id=f"srvtoolu_{tool_use_id.removeprefix('tooluse_')}",
                name=canonical_name,
                input=tool_input,
            )
        return None

    def _req_map_content_block(
        self, block: ContentBlockParam
    ) -> ContentBlockTypeDef | None:
        """Map an Anthropic request content block to a Bedrock content block.

        Maps ``ServerToolUseBlockParam`` blocks whose ``name`` appears in
        ``CANONICAL_TO_BEDROCK_TOOL_MAP`` to Bedrock ``toolUse`` blocks.  Returns ``None``
        for all other block types (use the default adapter mapping).

        Args:
            block: An Anthropic content block param from the request messages.

        Returns:
            A Bedrock content block dict, or ``None`` to use the default mapping.
        """
        if isinstance(block, ServerToolUseBlockParam) and (
            bedrock_name := self.CANONICAL_TO_BEDROCK_TOOL_MAP.get(block.name)  # type: ignore[call-overload]
        ):
            return {
                "toolUse": {
                    "toolUseId": f"tooluse_{block.id.removeprefix('srvtoolu_')}",
                    "name": bedrock_name,
                    "input": block.input,
                }
            }
        return None

    def _resp_stream_map_tool_use(
        self, tool_use_id: str, bedrock_tool_name: str
    ) -> ContentBlock | None:
        """Map a Bedrock streaming ``toolUse`` start to an Anthropic content block.

        Args:
            tool_use_id: Raw ``toolUseId`` from the Bedrock ``contentBlockStart``.
            bedrock_tool_name: Bedrock-side tool name.

        Returns:
            Anthropic content block for the stream start, or ``None`` for unmapped tools.
        """
        if canonical_name := self._canonical_name_for(bedrock_tool_name):
            return ServerToolUseBlock(
                type="server_tool_use",
                id=f"srvtoolu_{tool_use_id.removeprefix('tooluse_')}",
                name=canonical_name,
                input={},
            )
        return None

    def _resp_stream_map_tool_result(
        self, tool_use_id: str, result_type: str, content_items: list[Any]
    ) -> ContentBlock | None:
        """Map a buffered Bedrock streaming ``toolResult`` to an Anthropic content block.

        Args:
            tool_use_id: The raw ``toolUseId`` from the Bedrock ``contentBlockStart``.
            result_type: The Bedrock result type string (e.g.
                ``"nova_code_interpreter_result"``).
            content_items: The accumulated content item dicts from all
                ``contentBlockDelta`` events for this block.

        Returns:
            An Anthropic content block to emit, or ``None`` to discard the block.
        """

    def _req_enable_prompt_caching(
        self,
        system_blocks: list[SystemContentBlockTypeDef] | None,
        tool_config: ToolConfigurationTypeDef | None,
        bedrock_messages: list[MessageTypeDef],
        prompt_caching: set[PromptCaching] | frozenset[PromptCaching],
        prompt_caching_ttl: CacheTTLType | None,
    ) -> None:
        """Append cache-point blocks to the requested components.

        Two modes depending on ``SIMPLIFIED_CACHE_MANAGEMENT``:

        - ``True``: single checkpoint at the highest-priority component
          (Bedrock auto-looks back ~20 blocks).
        - ``False``: one checkpoint per component up to ``MAX_CACHE_BLOCKS``.

        Args:
            system_blocks: System content blocks list.
            tool_config: Bedrock tool configuration.
            bedrock_messages: Bedrock message list.
            prompt_caching: Components to cache (``"system"``, ``"tools"``, ``"messages"``).
            prompt_caching_ttl: Cache TTL, or ``None`` for the default.
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
        system_blocks: list[SystemContentBlockTypeDef] | None,
        tool_config: ToolConfigurationTypeDef | None,
        bedrock_messages: list[MessageTypeDef],
        prompt_caching: set[PromptCaching] | frozenset[PromptCaching],
        cache_point: ContentBlockTypeDef,
    ) -> None:
        """Place a single cache checkpoint at the highest-priority available location.

        Priority order: messages > tools > system.

        Args:
            system_blocks: System content blocks list.
            tool_config: Bedrock tool configuration.
            bedrock_messages: Bedrock message list.
            prompt_caching: Requested cache components.
            cache_point: Cache-point block to append.
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
        system_blocks: list[SystemContentBlockTypeDef] | None,
        tool_config: ToolConfigurationTypeDef | None,
        bedrock_messages: list[MessageTypeDef],
        prompt_caching: set[PromptCaching] | frozenset[PromptCaching],
        cache_point: ContentBlockTypeDef,
    ) -> None:
        """Place one checkpoint per requested component up to ``MAX_CACHE_BLOCKS``.

        When ``PROMPT_CACHING_TOOL_SUPPORTED`` is ``False``, messages containing
        ``toolUse`` or ``toolResult`` blocks are skipped — those models reject
        cachePoint in tool-call-related turns.

        Args:
            system_blocks: System content blocks list.
            tool_config: Bedrock tool configuration.
            bedrock_messages: Bedrock message list.
            prompt_caching: Requested cache components.
            cache_point: Cache-point block to append.
        """
        remaining = self.MAX_CACHE_BLOCKS

        if "system" in prompt_caching and system_blocks:
            system_blocks.append(cache_point)
            remaining -= 1

        if (
            remaining
            and "tools" in prompt_caching
            and tool_config
            and self.PROMPT_CACHING_TOOL_SUPPORTED
        ):
            tool_config["tools"].append(cache_point)  # type: ignore[attr-defined]
            remaining -= 1

        if remaining and "messages" in prompt_caching:
            for message in bedrock_messages:
                if not remaining:
                    break
                content: list[ContentBlockTypeDef] = message["content"]  # type: ignore[assignment]
                if not self.PROMPT_CACHING_TOOL_SUPPORTED and any(
                    "toolUse" in block or "toolResult" in block for block in content
                ):
                    continue
                content.append(cache_point)
                remaining -= 1

    def _req_configure_tools(
        self,
        tool_config: ToolConfigurationTypeDef | None,
        additional_request_fields: JsonMapping,
        server_tools: list[JsonMapping],
        bedrock_messages: list[MessageTypeDef] | None = None,
    ) -> None:
        """Apply model-specific tool configuration.  No-op by default.

        Called after ``_req_promote_system_tools`` and ``_req_extract_server_tools``.
        Override to inject required beta flags or apply other model-specific
        transformations (e.g. Anthropic Claude injects ``anthropic_beta`` flags and
        optionally moves server tools with extra params to ``additionalModelRequestFields``).

        Args:
            tool_config: Bedrock tool configuration after system tool promotion.
                Mutable — subclasses may remove ``toolSpec`` stubs in-place.
            additional_request_fields: Mutable additional request fields dict.
            server_tools: Per-tool dicts.  On the OpenAI route these contain only
                ``{"name": canonical_name}``; on the Anthropic route they are full
                ``model_dump(exclude_none=True)`` dicts that may include extra params
                (e.g. ``max_characters``, ``max_uses``) used for native-format routing.
            bedrock_messages: Translated Bedrock message list, used to detect
                ``toolResult`` blocks that require multi-turn stub mode on both
                the OpenAI and Anthropic routes.  ``None`` disables native-format routing.
        """

    def _req_extract_server_tools(
        self,
        tool_config: ToolConfigurationTypeDef | None,  # noqa: ARG002
    ) -> list[JsonMapping]:
        """Detect model-specific server tools in *tool_config*.

        Scans ``toolSpec`` entries for versioned server tool type names and
        returns ``{"name": canonical_name}`` dicts for beta flag injection.
        Does **not** mutate *tool_config* — server tools remain in place so
        Bedrock always has a ``toolConfig`` for multi-turn conversations.

        Args:
            tool_config: Bedrock tool configuration before system tool promotion.

        Returns:
            List of ``{"name": canonical_name}`` dicts for detected server
            tools.  Empty list by default; overridden on Claude models.
        """
        return []

    def _req_promote_system_tools(
        self, tool_config: ToolConfigurationTypeDef | None
    ) -> ToolConfigurationTypeDef | None:
        """Promote eligible ``toolSpec`` entries to ``systemTool`` entries.

        A ``toolSpec`` entry is promoted when its name appears in
        ``SUPPORTED_SYSTEM_TOOLS``.  Both the Anthropic Messages adapter
        (pre-translated via ``tool_name_map``) and the Responses adapter
        (pre-translated via ``tool_name_map``) emit Bedrock names directly, so
        a simple membership check suffices.  Promoted entries move to the end of
        the list; ``toolChoice`` is dropped when no regular entries remain.

        This is the only place that emits ``{"systemTool": {"name": ...}}``.

        Args:
            tool_config: Existing Bedrock tool configuration, or ``None``.

        Returns:
            Updated tool configuration, or ``None`` if input was ``None``.
        """
        if tool_config is None:
            return None
        remaining: list[ToolTypeDef] = []
        promoted: list[ToolTypeDef] = []
        for entry in tool_config["tools"]:
            spec = entry.get("toolSpec") if isinstance(entry, dict) else None
            if not spec:
                remaining.append(entry)
                continue
            if (name := spec.get("name", "")) in self.SUPPORTED_SYSTEM_TOOLS:
                promoted.append({"systemTool": {"name": name}})
            else:
                remaining.append(entry)
        if not promoted:
            return tool_config
        tool_config["tools"] = remaining + promoted
        if not remaining:
            tool_config.pop("toolChoice", None)
        return tool_config

    @staticmethod
    def _validate_no_budget_tokens(budget_tokens: int | None) -> None:
        """Raise if *budget_tokens* is set (unsupported on this model).

        Raises:
            ApiError: If *budget_tokens* is not ``None``.
        """
        if budget_tokens is not None:
            msg = "This model does not support 'thinking_budget'. Use 'reasoning_effort' instead."
            raise ApiError(msg)
