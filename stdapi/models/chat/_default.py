"""Default chat model implementation using AWS Bedrock Converse API.

This module provides the default chat completion implementation that works with
all AWS Bedrock models supporting the Converse and ConverseStream APIs.
"""

from asyncio import create_task, gather
from contextlib import suppress
from contextvars import ContextVar
from os.path import splitext
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from fastapi import HTTPException
from pydantic_core import from_json, to_json
from sse_starlette import EventSourceResponse, JSONServerSentEvent

from stdapi.aws import get_client
from stdapi.aws_bedrock import (
    GUARDTRAIL_CONFIG_VAR,
    MIME_TYPES_TO_AUDIO_TYPE,
    MIME_TYPES_TO_DOCUMENT_TYPE,
    MIME_TYPES_TO_VIDEO_TYPE,
    PERFORMANCE_CONFIG_VAR,
    PROMPT_CACHING,
    PROMPT_CACHING_DEFAULT,
    PromptCaching,
    handle_bedrock_client_error,
    image_block_from_bytes,
    image_block_from_data_url,
    image_block_from_http_url,
    image_block_from_s3_url,
    set_inference_configuration,
)
from stdapi.config import SETTINGS
from stdapi.models.audio import synthesize_speech
from stdapi.models.chat import ChatModelBase
from stdapi.monitoring import log_request_stream_event, log_response_params
from stdapi.openai_exceptions import OpenaiError
from stdapi.tokenizer import estimate_token_count
from stdapi.types.openai import FunctionDefinition
from stdapi.types.openai_chat_completions import (
    Annotation,
    AnnotationURLCitation,
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionAudio,
    ChatCompletionChunk,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartRefusalParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolUnionParam,
    Choice,
    ChoiceDelta,
    ChoiceDeltaFunctionCall,
    ChoiceDeltaToolCall,
    ChunkChoice,
    CompletionTokensDetails,
    CompletionUsage,
    File,
    FunctionCall,
    FunctionCallParam,
    PromptTokensDetails,
    ReasoningEffort,
)
from stdapi.utils import b64decode_data_or_uri_with_mime, b64encode, try_parse_json

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterable

    from pydantic import JsonValue
    from types_aiobotocore_bedrock_runtime import BedrockRuntimeClient
    from types_aiobotocore_bedrock_runtime.literals import (
        AudioFormatType,
        CacheTTLType,
        ConversationRoleType,
        ServiceTierTypeType,
        StopReasonType,
        VideoFormatType,
    )
    from types_aiobotocore_bedrock_runtime.type_defs import (
        AudioBlockTypeDef,
        CachePointBlockTypeDef,
        ContentBlockDeltaEventTypeDef,
        ContentBlockOutputTypeDef,
        ContentBlockStartEventTypeDef,
        ContentBlockTypeDef,
        ConverseStreamMetadataEventTypeDef,
        ConverseStreamOutputTypeDef,
        DocumentBlockTypeDef,
        InferenceConfigurationTypeDef,
        MessageStopEventTypeDef,
        MessageTypeDef,
        ReasoningContentBlockUnionTypeDef,
        SystemContentBlockTypeDef,
        SystemToolTypeDef,
        ToolChoiceTypeDef,
        ToolConfigurationTypeDef,
        ToolResultContentBlockUnionTypeDef,
        ToolSpecificationTypeDef,
        ToolTypeDef,
        ToolUseBlockTypeDef,
        VideoBlockTypeDef,
    )

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef
    from stdapi.models import ModelDetails
    from stdapi.types.openai_chat_completions import (
        ChatCompletionAudioParam,
        ChatCompletionNamedToolChoiceCustomParam,
        CompletionCreateParams,
        FinishReason,
        OutputModalities,
        ServiceTiers,
    )


class ChatModel(ChatModelBase[Any, Any]):
    """Default chat model using AWS Bedrock Converse API."""

    #: OpenAI to Bedrock prompt cache retention mapping
    _CACHE_TTL: ClassVar[dict[Any, Any]] = {"in-memory": None, "24h": "1h"}

    #: System message role names recognized by Bedrock
    _SYSTEM_ROLES: ClassVar[set[str]] = {"system", "developer"}

    #: Context variable tracking legacy function call format usage
    _LEGACY_FUNCTION: ClassVar[ContextVar[bool]] = ContextVar("legacy_function")

    #: Default output modalities when none specified
    _DEFAULT_OUTPUT_MODALITIES: ClassVar[list[str]] = ["text"]

    #: Empty tool schema for Bedrock tool configuration
    _EMPTY_TOOL: ClassVar[dict[str, str]] = {"type": "object"}

    #: Prefix for system-provided tool identifiers
    _SYSTEM_TOOL_PREFIX: ClassVar[str] = "systemTool_"

    #: Supported service tier values for Bedrock requests
    _SERVICES_TIERS: ClassVar[frozenset[str]] = frozenset(
        {"priority", "flex", "reserved"}
    )

    #: Image processing functions for different URL schemes
    _IMAGES_FUNCTIONS = (
        image_block_from_data_url,
        image_block_from_s3_url,
        image_block_from_http_url,
    )

    #: Bedrock stop reasons to OpenAI finish reasons mapping
    _FINISH_REASONS: ClassVar[dict[StopReasonType | None, FinishReason]] = {
        "max_tokens": "length",
        "model_context_window_exceeded": "length",
        "content_filtered": "content_filter",
        "guardrail_intervened": "content_filter",
        "malformed_model_output": "content_filter",
        "malformed_tool_use": "content_filter",
        "tool_use": "tool_calls",
    }

    #: Whether this model supports prompt caching
    PROMPT_CACHING_SUPPORTED: ClassVar[bool] = False

    #: Whether this model supports tools prompt caching
    PROMPT_CACHING_TOOL_SUPPORTED: ClassVar[bool] = False

    #: Whether this model supports system prompt
    SYSTEM_PROMPT_SUPPORTED: ClassVar[bool] = True

    async def create_completion(
        self,
        model: ModelDetails,
        request: CompletionCreateParams,
        completion_id: str,
        created: int,
    ) -> ChatCompletion | EventSourceResponse:
        """Creates a completion response for the given input parameters.

        Eeither as a streaming response or a non-streaming one, based on the request configuration.
        This method utilizes specific model configurations, applies optional reasoning settings, and
        handles prompt caching before preparing and submitting the request.

        Args:
            model (ModelDetails): The model details to use for generating the completion.
            request (CompletionCreateParams): The parameters defining the completion request,
                including messages, tokens, and other optional settings.
            completion_id (str): A unique identifier for the completion request.
            created (int): The timestamp indicating when the request was created.

        Returns:
            ChatCompletion | EventSourceResponse: Returns a completed response or a streaming
            event source response based on the configuration of the request.

        Raises:
            None explicitly, as error handling is assumed to occur internally or upstream.
        """
        bedrock_messages, system_blocks = await self._req_map_messages(request.messages)
        max_tokens = request.max_completion_tokens or request.max_tokens
        choices_count = request.n or 1
        additional_request_fields: dict[str, Any] = {}
        inference_cfg = set_inference_configuration(
            self._model_id,
            additional_request_fields,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=max_tokens,
            stop_sequences=request.stop,
            frequency_penalty=request.frequency_penalty,
            presence_penalty=request.presence_penalty,
            logit_bias=request.logit_bias,  # type: ignore[arg-type]
            seed=request.seed,
            top_logprobs=request.top_logprobs,
            top_k=request.top_k,
            **request.model_extra,
        )
        self._LEGACY_FUNCTION.set(request.functions is not None)

        if request.reasoning_effort not in (None, "none") or request.enable_thinking:
            self._req_configure_reasoning(
                reasoning_effort=request.reasoning_effort,
                budget_tokens=request.thinking_budget,
                max_tokens=max_tokens,
                additional_request_fields=additional_request_fields,
            )

        bedrock_service_tier, openai_service_tier = self._req_map_service_tier(
            request.service_tier
        )
        tool_config = self._req_build_tool_config(request)
        prompt_caching = self._req_parse_prompt_cache_key(request.prompt_cache_key)
        prompt_caching_ttl = self._CACHE_TTL.get(
            request.prompt_cache_retention, request.prompt_cache_retention
        )

        # Apply prompt caching before preparing the request
        self._req_enable_prompt_caching(
            system_blocks=system_blocks,
            tool_config=tool_config,
            bedrock_messages=bedrock_messages,
            prompt_caching=prompt_caching,
            prompt_caching_ttl=prompt_caching_ttl,
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

        legacy_function = self._LEGACY_FUNCTION.get()
        if request.stream:
            return EventSourceResponse(
                await log_request_stream_event(
                    self._streaming_completion(
                        completion_id,
                        created,
                        bedrock_runtime,
                        bedrock_request,
                        openai_service_tier,
                        legacy_function=legacy_function,
                        include_usage=(
                            request.stream_options is not None
                            and request.stream_options.include_usage is True
                        ),
                    )
                )
            )
        return await self._non_streaming_completion(
            completion_id,
            created,
            bedrock_runtime,
            bedrock_request,
            openai_service_tier,
            choices_count,
            request.audio,
            request.modalities or self._DEFAULT_OUTPUT_MODALITIES,  # type: ignore[arg-type]
            legacy_function=legacy_function,
        )

    async def _non_streaming_completion(
        self,
        completion_id: str,
        created: int,
        bedrock_runtime: BedrockRuntimeClient,
        request: ConverseRequestBaseTypeDef,
        service_tier: ServiceTiers | None,
        choices_count: int,
        audio_params: ChatCompletionAudioParam | None,
        modalities: list[OutputModalities],
        *,
        legacy_function: bool,
    ) -> ChatCompletion:
        """Handles non-streaming completion requests.

        Processes responses, extracts and formats relevant data, and generates chat completion results.

        Args:
            completion_id (str): Unique identifier for the completion request.
            created (int): Timestamp indicating when the request was created.
            bedrock_runtime (BedrockRuntimeClient): Client used to execute the Converse API.
            request (ConverseRequestBaseTypeDef): Payload for the Converse API request.
            service_tier (ServiceTiers | None): Optional tier of service for the request.
            choices_count (int): Number of response choices to generate.
            audio_params (ChatCompletionAudioParam | None): Optional parameters for audio generation.
            modalities (list[OutputModalities]): List of output modalities such as text or audio.
            legacy_function (bool): Indicates whether to use legacy logic for tool calls.

        Returns:
            ChatCompletion: A structured completion response containing generated choices,
            usage statistics, and additional metadata.
        """
        with handle_bedrock_client_error():
            responses = await gather(
                *(bedrock_runtime.converse(**request) for _ in range(choices_count))
            )

        choices: list[Choice] = []
        usage = CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        reasoning_contents: list[str] = []
        tts_tasks: dict[int, Any] = {}
        cached_tokens = 0
        for index, response in enumerate(responses):
            usage.prompt_tokens += response["usage"]["inputTokens"]
            usage.completion_tokens += response["usage"]["outputTokens"]
            usage.total_tokens += response["usage"]["totalTokens"]
            cached_tokens += response["usage"].get("cacheReadInputTokens", 0)
            message = response["output"]["message"]["content"]
            tool_calls, function_call = self._resp_extract_tool_calls_from_converse(
                message, legacy_function=legacy_function
            )
            content, reasoning_content = self._resp_extract_output_text_from_converse(
                message
            )
            annotations = self._resp_extract_citations_from_output_blocks(message)
            if reasoning_content:
                reasoning_contents.append(reasoning_content)
            if audio_params and content:
                tts_tasks[index] = create_task(
                    self._resp_get_or_generate_audio(
                        audio_params, message, completion_id, content, created, index
                    )
                )
            choices.append(
                Choice(
                    finish_reason=self._resp_map_bedrock_stop_reason(
                        response["stopReason"], legacy_function=legacy_function
                    ),
                    index=index,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=content if "text" in modalities else None,
                        reasoning_content=reasoning_content,
                        tool_calls=tool_calls,
                        function_call=function_call,
                        annotations=annotations,
                    ),
                )
            )
        if cached_tokens:
            usage.prompt_tokens_details = PromptTokensDetails(
                cached_tokens=cached_tokens
            )
        if reasoning_contents:
            # Estimate reasoning tokens, not included in Bedrock result
            reasoning_tokens = await estimate_token_count(*reasoning_contents)
            if reasoning_tokens:
                usage.completion_tokens_details = CompletionTokensDetails(
                    reasoning_tokens=reasoning_tokens
                )
                usage.total_tokens += reasoning_tokens
                usage.completion_tokens += reasoning_tokens

        for index, tts_task in tts_tasks.items():
            choices[index].message.audio = await tts_task

        return log_response_params(
            ChatCompletion(
                id=completion_id,
                choices=choices,
                created=created,
                model=self._model_id,
                object="chat.completion",
                usage=usage,
                service_tier=service_tier,
            )
        )

    async def _streaming_completion(
        self,
        completion_id: str,
        created: int,
        bedrock_runtime: BedrockRuntimeClient,
        request: ConverseRequestBaseTypeDef,
        service_tier: ServiceTiers | None,
        *,
        legacy_function: bool,
        include_usage: bool = False,
    ) -> AsyncGenerator[JSONServerSentEvent]:
        """Streams a completion response asynchronously, handling chunked output and streaming specific events.

        Processes the response stream returned by the bedrock runtime
        client and formats JSON Server-Sent Events (SSE) for output.

        Args:
            completion_id: Unique identifier for the completion process.
            created: Timestamp indicating when the completion request was initiated.
            bedrock_runtime: Client instance for accessing the Bedrock runtime for
                streaming operations.
            request: Request definition containing data required to initiate the
                converse stream.
            service_tier: Enum representing the service tier being used for the
                operation. Can be None.
            legacy_function: A flag indicating whether to use legacy logic during stream
                processing.
            include_usage: A flag that specifies whether to include resource usage
                information in the output. Defaults to False.

        Yields:
            JSONServerSentEvent: Server-Sent Event object containing the formatted
            response payload data.

        """
        with handle_bedrock_client_error():
            stream: AsyncIterator[ConverseStreamOutputTypeDef] = (
                await bedrock_runtime.converse_stream(**request)
            )["stream"]

        yield JSONServerSentEvent(
            data=log_response_params(
                self._resp_stream_initial_chunk(
                    completion_id, created, service_tier
                ).model_dump(mode="json", exclude_none=True)
            )
        )

        end_state = False
        chunk: ChatCompletionChunk | None = None
        async for event in stream:
            chunk, end = self._resp_stream_delta_chunk(
                completion_id,
                created,
                event,
                service_tier,
                legacy_function=legacy_function,
                chunk=chunk,
            )
            end_state |= end
            if end_state:
                if include_usage and chunk:
                    usage = self._resp_stream_extract_usage_from_metadata(event)
                    if usage:
                        chunk.usage = usage
            elif chunk:
                yield JSONServerSentEvent(
                    data=chunk.model_dump(mode="json", exclude_none=True)
                )
                chunk = None
        if chunk:
            yield JSONServerSentEvent(
                data=chunk.model_dump(mode="json", exclude_none=True)
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

        Returns:
            A tuple of (BedrockRuntimeClient, request payload dict).
        """
        latency, default_service_tier = PERFORMANCE_CONFIG_VAR.get()
        service_tier = service_tier or default_service_tier
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
        if additional_request_fields:
            request["additionalModelRequestFields"] = additional_request_fields
        if service_tier:
            request["serviceTier"] = {"type": service_tier}
        if latency:
            request["performanceConfig"] = {"latency": latency}

        with suppress(LookupError):
            request["guardrailConfig"] = GUARDTRAIL_CONFIG_VAR.get()

        return get_client("bedrock-runtime", model.region), request

    def _resp_map_bedrock_stop_reason(
        self, stop_reason: StopReasonType | None, *, legacy_function: bool
    ) -> FinishReason:
        """Translate Bedrock stop reasons to OpenAI finish reasons.

        Args:
            stop_reason: Bedrock stop reason value (or None).
            legacy_function: Whether to use legacy function_call format.

        Returns:
            OpenAI stop reason value.
        """
        reason = self._FINISH_REASONS.get(stop_reason, "stop")
        if legacy_function and reason == "tool_calls":
            return "function_call"
        return reason

    @staticmethod
    def _resp_stream_get_content_block_delta(
        choice_delta: ChoiceDelta,
        delta_block: ContentBlockDeltaEventTypeDef,
        *,
        legacy_function: bool,
    ) -> None:
        """Processes a content block delta and updates the corresponding choice delta object with relevant information.

        This method is primarily used to handle content block delta updates, extracting
        specific fields such as text, reasoning content, and tool use, then updating
        the choice delta object accordingly. It also manages integration differences
        based on the legacy or non-legacy function handling approach.

        Args:
            choice_delta: The choice delta object to be updated with new fields extracted
                from the content block delta.
            delta_block: The dictionary representing a delta event containing the changes
                to be processed.
            legacy_function: Determines whether to process tool calls in the context of
                legacy function handling or modernized tool call support.

        """
        delta = delta_block["delta"]
        with suppress(KeyError):
            choice_delta.content = delta["text"]
        with suppress(KeyError):
            choice_delta.reasoning_content = delta["reasoningContent"]["text"]
        try:
            delta_tool_use = delta["toolUse"]
        except KeyError:
            return
        function = ChoiceDeltaFunctionCall(arguments=delta_tool_use["input"])
        if legacy_function:
            choice_delta.function_call = function
        else:
            choice_delta.tool_calls = [
                ChoiceDeltaToolCall(
                    index=delta_block["contentBlockIndex"],
                    type="function",
                    function=function,
                )
            ]

    def _resp_stream_initial_chunk(
        self, completion_id: str, created: int, service_tier: ServiceTiers | None
    ) -> ChatCompletionChunk:
        """Generates the initial response chunk for a chat completion request.

        This function creates and returns the first chunk of the response in
        a streaming chat completion. It includes metadata and an initial choice
        to represent the assistant's role.

        Args:
            completion_id: A unique identifier for the chat completion request.
            created: A Unix timestamp indicating when the completion was created.
            service_tier: The service tier for the chat completion request,
                which could be a specific plan or tier level. Can be None.

        Returns:
            ChatCompletionChunk: The initial chunk of the chat completion response.
        """
        return ChatCompletionChunk(
            id=completion_id,
            choices=[ChunkChoice(index=0, delta=ChoiceDelta(role="assistant"))],
            created=created,
            model=self._model_id,
            object="chat.completion.chunk",
            service_tier=service_tier,
        )

    def _resp_stream_delta_chunk(
        self,
        completion_id: str,
        created: int,
        event: ConverseStreamOutputTypeDef,
        service_tier: ServiceTiers | None,
        *,
        legacy_function: bool,
        chunk: ChatCompletionChunk | None = None,
    ) -> tuple[ChatCompletionChunk | None, bool]:
        """Processes and updates a streaming chat response chunk with delta events.

        This function handles various event types such as `contentBlockStart`,
        `contentBlockDelta`, and `messageStop` to dynamically update the chat
        completion chunk. It utilizes tools, functions, and finish reasons to
        construct and return the updated chunk and a flag indicating the end of
        the response stream.

        Args:
            completion_id (str): The unique identifier for the chat completion.
            created (int): The timestamp when the chat completion chunk was created.
            event (ConverseStreamOutputTypeDef): The event data containing updates for
                the current chunk.
            service_tier (ServiceTiers | None): The service tier information for the
                chat completion, if any.
            legacy_function (bool): A flag indicating whether legacy handling for
                functions should be used.
            chunk (ChatCompletionChunk | None, optional): The current chat completion
                chunk to be updated. Defaults to None.

        Returns:
            tuple[ChatCompletionChunk | None, bool]: A tuple containing the updated
            chat completion chunk and a flag indicating if the response stream has
            ended.
        """
        if chunk:
            choice = chunk.choices[0]
            choice_delta = choice.delta
        else:
            choice_delta = ChoiceDelta()
            choice = ChunkChoice(index=0, delta=choice_delta)
            chunk = ChatCompletionChunk(
                id=completion_id,
                choices=[choice],
                created=created,
                model=self._model_id,
                object="chat.completion.chunk",
                service_tier=service_tier,
            )
        end = False

        if "contentBlockStart" in event:
            start_block: ContentBlockStartEventTypeDef = event["contentBlockStart"]
            start = start_block["start"]
            try:
                start_tool_use = start["toolUse"]
            except KeyError:
                return None, end
            tool_id = start_tool_use["toolUseId"]
            function = ChoiceDeltaFunctionCall(name=start_tool_use["name"])
            if legacy_function:
                choice_delta.function_call = function
            else:
                choice_delta.tool_calls = [
                    ChoiceDeltaToolCall(
                        index=start_block["contentBlockIndex"],
                        id=tool_id,
                        type="function",
                        function=function,
                    )
                ]

        elif "contentBlockDelta" in event:
            self._resp_stream_get_content_block_delta(
                choice_delta,
                event["contentBlockDelta"],
                legacy_function=legacy_function,
            )

        elif "messageStop" in event:
            stop_block: MessageStopEventTypeDef = event["messageStop"]
            choice.finish_reason = self._resp_map_bedrock_stop_reason(
                stop_block["stopReason"], legacy_function=legacy_function
            )
            end = True

        return chunk, end

    @staticmethod
    def _resp_stream_extract_usage_from_metadata(
        stream_event: ConverseStreamOutputTypeDef,
    ) -> CompletionUsage | None:
        """Extracts usage data from metadata of a stream event.

        This method processes the input stream event to extract detailed information
        regarding the completion usage. The usage data may include details about
        completion tokens, prompt tokens, total tokens, and optionally cached tokens
        used during the prompt.

        Args:
            stream_event (ConverseStreamOutputTypeDef): The stream event containing
                metadata with usage details.

        Returns:
            CompletionUsage | None: An instance of CompletionUsage containing the
                extracted usage data if successful, otherwise None.
        """
        try:
            metadata_event: ConverseStreamMetadataEventTypeDef = stream_event[
                "metadata"
            ]
        except KeyError:
            return None
        usage = metadata_event["usage"]
        completion_usage = CompletionUsage(
            completion_tokens=usage["outputTokens"],
            prompt_tokens=usage["inputTokens"],
            total_tokens=usage["totalTokens"],
        )
        with suppress(KeyError):
            completion_usage.prompt_tokens_details = PromptTokensDetails(
                cached_tokens=usage["cacheReadInputTokens"]
            )
        return completion_usage

    @staticmethod
    def _resp_extract_output_text_from_converse(
        contents: list[ContentBlockOutputTypeDef],
    ) -> tuple[str | None, str | None]:
        """Extracts and formats output text and reasoning text from a list of content blocks.

        This method processes a list of content blocks, attempting to extract text associated with
        content and reasoning attributes. Text is aggregated and returned as concatenated strings,
        or `None` if no text is found for a specific attribute.

        Args:
            contents (list[ContentBlockOutputTypeDef]): A list of content blocks where each block
                represents a data structure that may contain content text and reasoning text.

        Returns:
            tuple[str | None, str | None]: A tuple containing two elements:
                - The first element is a concatenated string of content text if text exists,
                  otherwise `None`.
                - The second element is a concatenated string of reasoning text if text exists,
                  otherwise `None`.
        """
        content_text: list[str] = []
        reasoning_text: list[str] = []
        for block in contents:
            with suppress(KeyError):
                content_text.append(block["text"])
            with suppress(KeyError):
                reasoning_text.append(
                    block["reasoningContent"]["reasoningText"]["text"]
                )
        return "".join(content_text) if content_text else None, "".join(
            reasoning_text
        ) if reasoning_text else None

    @staticmethod
    def _resp_extract_citations_from_output_blocks(
        contents: list[ContentBlockOutputTypeDef],
    ) -> list[Annotation] | None:
        """Extracts citation annotations from a list of content blocks.

        This method processes a list of content blocks to extract citation-related information
        and creates a list of annotation objects for citations of type "url_citation."
        If no valid citation is found, it returns None.

        Args:
            contents (list[ContentBlockOutputTypeDef]): A list of content block dictionaries
                that contain citation data.

        Returns:
            list[Annotation] | None: A list of annotations if citations are found, or
            None if no valid citations are present.
        """
        annotations: list[Annotation] = []
        for block in contents:
            try:
                citations = block["citationsContent"]["citations"]
            except KeyError:
                continue

            for citation in citations:
                try:
                    web_location = citation["location"]["web"]
                    url = web_location["url"]
                except KeyError:
                    continue
                annotations.append(
                    Annotation(
                        type="url_citation",
                        url_citation=AnnotationURLCitation(
                            url=url,
                            title=citation.get("title")
                            or web_location.get("domain", ""),
                            start_index=0,
                            end_index=0,
                        ),
                    )
                )

        return annotations or None

    @staticmethod
    def _resp_extract_tool_calls_from_converse(
        contents: list[ContentBlockOutputTypeDef], *, legacy_function: bool
    ) -> tuple[list[ChatCompletionMessageToolCallUnion] | None, FunctionCall | None]:
        """Extracts tool calls and function calls from conversation response content.

        This method processes a list of conversation content blocks to identify tool
        calls or function calls based on the provided data structure. If the
        `legacy_function` flag is set, it directly returns the first observed
        function call and bypasses tool call extraction. Otherwise, it collects
        tool calls as `ChatCompletionMessageFunctionToolCall` objects.

        Args:
            contents (list[ContentBlockOutputTypeDef]): A list of content blocks containing
                response data that may include tool use information.
            legacy_function (bool): A flag indicating whether to prioritize returning
                a single legacy-style function call instead of processing and returning
                tool calls.

        Returns:
            tuple[list[ChatCompletionMessageToolCallUnion] | None, FunctionCall | None]:
                A tuple containing two values:
                - A list of extracted tool calls or None if no tool use is detected.
                - A single function call if the `legacy_function` flag is set or None
                  if no function call is detected.
        """
        tool_calls: list[ChatCompletionMessageToolCallUnion] = []
        for content in contents:
            try:
                tool_use = content["toolUse"]
            except KeyError:
                continue
            function = FunctionCall(
                name=tool_use["name"], arguments=to_json(tool_use["input"]).decode()
            )
            if legacy_function:
                return None, function
            tool_calls.append(
                ChatCompletionMessageFunctionToolCall(
                    type="function", id=tool_use["toolUseId"], function=function
                )
            )

        return tool_calls or None, None

    @staticmethod
    async def _resp_get_or_generate_audio(
        audio_params: ChatCompletionAudioParam,
        contents: list[ContentBlockOutputTypeDef],
        completion_id: str,
        content: str,
        created: int,
        index: int,
    ) -> ChatCompletionAudio:
        """Handles the generation or retrieval of audio content for a specific content block.

        If audio data is already present in the content block, it is directly used.
        Otherwise, the method synthesizes audio data based on the provided text content.

        Args:
            audio_params: Configuration parameters specifying the voice and format to
                be used for audio synthesis.
            contents: A list of content blocks. Each block may optionally include an
                audio source in its nested structure.
            completion_id: A unique identifier for the completion task associated with
                the audio.
            content: The text content used for synthesizing audio if existing audio
                content is not available in the content block.
            created: A timestamp indicating the creation time of the request.
            index: The position of this specific content block in the overall content
                list.

        Returns:
            ChatCompletionAudio: An object representing the generated or retrieved audio
            associated with this content block. This includes metadata like its unique
            identifier, Base64-encoded audio data, expiration timestamp, and a transcript
            of the synthesized or associated text.
        """
        for block in contents:
            with suppress(KeyError):
                audio_content = block["audio"]["source"]["bytes"]
                # Get only the first audio block,
                # OpenAI API doesn't support multiple audio blocks
                break
        else:
            # Fall back to synthesizing audio from text
            audio_content = b"".join(
                [
                    chunk
                    async for chunk in await synthesize_speech(
                        text=content,
                        voice=audio_params.voice,
                        resp_format="pcm"
                        if audio_params.format == "pcm16"
                        else audio_params.format,
                    )
                ]
            )
        return ChatCompletionAudio(
            id=f"audio-{completion_id}-{index}",
            data=await b64encode(audio_content),
            expires_at=created,  # Not stored on the server, so expire immediately
            transcript=content,
        )

    @staticmethod
    def _req_extract_system_content_blocks(
        content: str | Iterable[ChatCompletionContentPartTextParam],
    ) -> list[SystemContentBlockTypeDef]:
        """Extract Bedrock system content blocks from an OpenAI content field.

        Args:
            content: Message content which may be a plain string or a list of
                ChatCompletionContentPartParam entries.

        Returns:
            A list of Bedrock SystemContentBlockTypeDef items (text blocks) in order.
        """
        results: list[SystemContentBlockTypeDef] = []
        if isinstance(content, str):
            if content:
                results.append({"text": content})
        else:
            results.extend({"text": part.text} for part in content if part.text)
        return results

    async def _req_extract_image_content_block(
        self, image_part: ChatCompletionContentPartImageParam
    ) -> ContentBlockTypeDef:
        """Convert an OpenAI image_url section to a Bedrock content block.

        Supports data URLs, s3:// URIs, and http(s) URLs (downloaded via aiohttp).

        Args:
            image_part: Image content part as provided by OpenAI Chat API.

        Returns:
            A Bedrock ContentBlockTypeDef for the referenced image.

        Raises:
            HTTPException: If the URL is invalid or unsupported by this implementation.
        """
        url = image_part.image_url.url
        for func in self._IMAGES_FUNCTIONS:
            content_block = await func(url)
            if content_block:
                return content_block
        raise HTTPException(status_code=400, detail=f"Invalid image URL: {url}")

    @staticmethod
    async def _req_extract_audio_content_block(
        audio_part: ChatCompletionContentPartInputAudioParam,
    ) -> ContentBlockTypeDef:
        """Convert an OpenAI input_audio section to a Bedrock content block.

        Args:
            audio_part: Audio content part as provided by OpenAI Chat API.

        Returns:
            A Bedrock ContentBlockTypeDef for the referenced audio.

        Raises:
            HTTPException: If the URL is invalid or unsupported by this implementation.
        """
        try:
            data = (await b64decode_data_or_uri_with_mime(audio_part.input_audio.data))[
                0
            ]
        except ValueError as error:
            raise HTTPException(status_code=400, detail=error.args[0]) from None
        audio_block_bytes: AudioBlockTypeDef = {
            "source": {"bytes": data},
            "format": audio_part.input_audio.format,
        }
        return {"audio": audio_block_bytes}

    @staticmethod
    async def _req_extract_file_content_block(file_part: File) -> ContentBlockTypeDef:
        """Convert an OpenAI file section to a Bedrock content block.

        The OpenAI File part contains base64-encoded bytes (file_data). This helper
        detects the file's MIME type using python-magic and maps it to the proper
        Bedrock content block:
        - image/* ➜ image block with inferred format and bytes
        - video/* ➜ video block with inferred/normalized format and bytes
        - audio/* ➜ audio block with inferred/normalized format and bytes
        - text/* or application/* ➜ document block with inferred/normalized format and bytes

        Args:
            file_part: OpenAI chat content part with type "file".

        Returns:
            A Bedrock ContentBlockTypeDef containing an image, video, or document block
            depending on the detected MIME type.

        Raises:
            HTTPException: When file_data is missing/invalid/empty or the detected
                MIME type is not supported by this implementation.
        """
        file_section = file_part.file
        b64_data = file_section.file_data
        try:
            data, mime = await b64decode_data_or_uri_with_mime(b64_data, validate=True)
        except ValueError as error:
            raise HTTPException(
                status_code=400, detail=f"Invalid {file_part}: {error.args[0]}"
            ) from None

        if mime.startswith("image/"):
            return image_block_from_bytes(data, mime)

        file_format = mime.split("/", 1)[1]
        if mime.startswith("video/"):
            video_format: VideoFormatType = MIME_TYPES_TO_VIDEO_TYPE.get(
                file_format,
                file_format,  # type: ignore[arg-type]
            )
            video_block_bytes: VideoBlockTypeDef = {
                "source": {"bytes": data},
                "format": video_format,
            }
            return {"video": video_block_bytes}

        if mime.startswith("audio/"):
            audio_format: AudioFormatType = MIME_TYPES_TO_AUDIO_TYPE.get(
                file_format,
                file_format,  # type: ignore[arg-type]
            )
            audio_block_bytes: AudioBlockTypeDef = {
                "source": {"bytes": data},
                "format": audio_format,
            }
            return {"audio": audio_block_bytes}

        if mime.startswith(("text/", "application/")):
            # Default to 'txt' when the MIME subtype is unknown
            document_format = MIME_TYPES_TO_DOCUMENT_TYPE.get(file_format, "txt")
            name_value = (
                # Remove file extension, "." is not supported
                splitext(file_section.filename)[0]  # noqa: PTH122
                if file_section.filename is not None
                else f"file-{document_format}"
            )
            document_block_bytes: DocumentBlockTypeDef = {
                "name": name_value,
                "source": {"bytes": data},
                "format": document_format,
            }
            return {"document": document_block_bytes}

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file MIME type for 'file' ({mime}): {file_part}",
        )

    async def _req_extract_content_blocks(
        self,
        content: (
            str
            | Iterable[
                ChatCompletionContentPartParam | ChatCompletionContentPartRefusalParam
            ]
            | None
        ),
    ) -> list[ContentBlockTypeDef]:
        """Extract Bedrock content blocks from OpenAI message content.

        Supports:
        - text parts
        - image_url parts with data URLs (base64), s3:// URIs, and http(s) downloads via aiohttp
        - file parts (image/video/document) with base64 body and MIME sniffing
        """
        blocks: list[ContentBlockTypeDef] = []
        if isinstance(content, str):
            blocks.append({"text": content})
            return blocks

        for part in content or ():
            if isinstance(part, ChatCompletionContentPartTextParam):
                blocks.append({"text": part.text})
            elif isinstance(part, ChatCompletionContentPartImageParam):
                blocks.append(await self._req_extract_image_content_block(part))
            elif isinstance(part, ChatCompletionContentPartInputAudioParam):
                blocks.append(await self._req_extract_audio_content_block(part))
            elif isinstance(part, File):
                blocks.append(await self._req_extract_file_content_block(part))
            else:  # pragma: no cover
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported content part type: {getattr(part, 'type', type(part))}",
                )

        return blocks

    @staticmethod
    def _req_build_tool_use_block(
        name: str, arguments: str | JsonValue, call_id: str
    ) -> ContentBlockTypeDef:
        """Build a Bedrock toolUse content block from OpenAI function call data.

        Args:
            name: Function/tool name.
            arguments: Either a JSON string (function tools) or a JSON value (custom tools).
            call_id: Optional stable tool call id.

        Returns:
            A ContentBlockTypeDef representing a toolUse block.
        """
        tool_input = (
            try_parse_json(arguments) if isinstance(arguments, str) else arguments
        ) or {}
        if isinstance(tool_input, dict):
            tool_use: ToolUseBlockTypeDef = {
                "toolUseId": call_id,
                "name": name,
                "input": tool_input,
            }
            return {"toolUse": tool_use}
        msg = (
            f'Invalid arguments for tool call "{name}" with ID "{call_id}": {arguments}'
        )
        raise OpenaiError(msg)

    @staticmethod
    def _req_append_text_content_block(
        content: str, content_blocks: list[ContentBlockTypeDef]
    ) -> None:
        """Adds a new content block to the list if content is provided.

        Args:
            content: The content string to be evaluated and added as a
                content block if not empty.
            content_blocks: The list of content blocks to which the new content block will be appended.
        """
        if content:
            content_blocks.append({"text": content})

    def _req_map_assistant_content(
        self,
        content_blocks: list[ContentBlockTypeDef],
        message_param: ChatCompletionAssistantMessageParam,
    ) -> None:
        """Maps the assistant message content into content block structures.

        Args:
            content_blocks: The list of content blocks to append the message content to.
            message_param: The assistant message.

        Raises:
            HTTPException: If the content part in `message_param` contains an unsupported
                type.
        """
        content = message_param.content
        if content is not None:
            if isinstance(content, str):
                self._req_append_text_content_block(content, content_blocks)
            else:
                for part in content:
                    if isinstance(part, ChatCompletionContentPartTextParam):
                        self._req_append_text_content_block(part.text, content_blocks)
                    elif isinstance(part, ChatCompletionContentPartRefusalParam):
                        self._req_append_text_content_block(
                            part.refusal, content_blocks
                        )
                    else:  # pragma: no cover
                        raise HTTPException(
                            status_code=400, detail=f"Unsupported message type: {part}"
                        )

    @staticmethod
    def _req_map_assistant_reasoning_content(
        content_blocks: list[ContentBlockTypeDef],
        message_param: ChatCompletionAssistantMessageParam,
    ) -> None:
        """Maps the reasoning message content into content block structures.

        Args:
            content_blocks: The list of content blocks to append the message content to.
            message_param: The assistant message.

        Raises:
            HTTPException: If the content part in `message_param` contains an unsupported
                type.
        """
        reasoning_content = message_param.reasoning_content
        if reasoning_content is not None:
            reasoning_block: ReasoningContentBlockUnionTypeDef = {}
            if isinstance(reasoning_content, str):
                reasoning_block["reasoningText"] = {"text": reasoning_content}
            else:
                text: list[str] = []
                for part in reasoning_content:
                    if isinstance(part, ChatCompletionContentPartTextParam):
                        text.append(part.text)
                    else:  # pragma: no cover
                        raise HTTPException(
                            status_code=400, detail=f"Unsupported message type: {part}"
                        )
                reasoning_block["reasoningText"] = {"text": "".join(text)}
            content_blocks.append({"reasoningContent": reasoning_block})

    def _req_extract_assistant_blocks(
        self, message_param: ChatCompletionAssistantMessageParam
    ) -> list[ContentBlockTypeDef]:
        """Append assistant tool use and content blocks.

        Appends Bedrock toolUse blocks derived from OpenAI assistant message
        `tool_calls` or legacy `function_call`, followed by any textual content
        (including refusal text when present).

        Args:
            message_param: The assistant message to convert (may include tool calls).

        Returns:
            Content blocks.

        Raises:
            HTTPException: If an unsupported tool call or content part type is encountered.
        """
        content_blocks: list[ContentBlockTypeDef] = []

        self._req_map_assistant_content(content_blocks, message_param)
        self._req_map_assistant_reasoning_content(content_blocks, message_param)

        # Tools and function calls must be at the end
        tool_calls: list[ChatCompletionMessageToolCallUnion] = (
            message_param.tool_calls if message_param.tool_calls is not None else []
        )
        for tool_call in tool_calls:
            call_id = tool_call.id
            if tool_call.type == "function":
                function_tool = tool_call.function
                name = function_tool.name
                arguments = function_tool.arguments
            elif tool_call.type == "custom":
                custom_tool = tool_call.custom
                name = custom_tool.name
                arguments = custom_tool.input
            else:  # pragma: no cover
                raise HTTPException(
                    status_code=400, detail=f"Unsupported tool call type: {tool_call}"
                )
            content_blocks.append(
                self._req_build_tool_use_block(
                    name=name, arguments=arguments, call_id=call_id
                )
            )

        function_call = message_param.function_call
        if function_call is not None:
            content_blocks.append(
                self._req_build_tool_use_block(
                    name=function_call.name,
                    arguments=function_call.arguments,
                    call_id=function_call.name,
                )
            )
            self._LEGACY_FUNCTION.set(True)

        return content_blocks

    @staticmethod
    def _req_parse_tool_content(
        text_content: str,
    ) -> ToolResultContentBlockUnionTypeDef:
        """Parses the content of a tool's textual output to determine its structure.

        The function attempts to parse the provided textual content as JSON. If it
        succeeds, the JSON structure is returned. If the parsing fails, the content
        is assumed to be plain text and is returned encapsulated in a dictionary.

        Args:
            text_content (str): The textual content to be parsed.

        Returns:
            ToolResultContentBlockUnionTypeDef: A dictionary containing either the
            parsed JSON mapping or the original text content. The result will be
            in the form {"json": JSON} if the content is valid JSON, or {"text":
            text_content} otherwise.
        """
        try:
            json_content = from_json(text_content)
        except ValueError:
            return {"text": text_content}
        else:
            return (
                {"json": json_content}
                if isinstance(json_content, dict)
                else {"text": text_content}
            )

    def _req_extract_tool_blocks(
        self, message_param: ChatCompletionToolMessageParam
    ) -> list[ContentBlockTypeDef]:
        """Extracts tool blocks from the given message parameter.

        This function processes the content of a `ChatCompletionToolMessageParam` to break it
        into structured content blocks. It supports both string-based and object-based message
        content. Text content is parsed and converted into a standard block format for further
        processing.

        Args:
            message_param: The message parameter containing tool invocation data, including
                content and tool call ID.

        Returns:
            A list of structured content blocks extracted and formatted from the given
            message parameter.
        """
        content_parts: list[ChatCompletionContentPartTextParam] = (
            [
                ChatCompletionContentPartTextParam(
                    text=message_param.content, type="text"
                )
            ]
            if isinstance(message_param.content, str)
            else message_param.content
        )

        content: list[ToolResultContentBlockUnionTypeDef] = []
        for part in content_parts:
            if part.type == "text":
                text_content = part.text
                content.append(self._req_parse_tool_content(text_content))

        return [
            {
                "toolResult": {
                    "toolUseId": message_param.tool_call_id,
                    "content": content,
                }
            }
        ]

    def _req_extract_function_blocks(
        self, message_param: ChatCompletionFunctionMessageParam
    ) -> list[ContentBlockTypeDef]:
        """Extracts function blocks from the given message parameter.

        This function processes a message parameter with possible tool content,
        parsing it into structured content blocks. It also sets an indicator
        for legacy function usage and organizes the results into the required
        format.

        Args:
            message_param: A structured input parameter containing details about
                the chat completion function message.

        Returns:
            A list of structured content blocks with parsed tool results.
        """
        self._LEGACY_FUNCTION.set(True)
        content: list[ToolResultContentBlockUnionTypeDef] = []
        text_content = message_param.content
        if text_content is not None:
            content.append(self._req_parse_tool_content(text_content))
        return [{"toolResult": {"toolUseId": message_param.name, "content": content}}]

    async def _req_map_messages(
        self, messages: list[ChatCompletionMessageParam]
    ) -> tuple[list[MessageTypeDef], list[SystemContentBlockTypeDef]]:
        """Asynchronously processes and maps a list of message parameters into two structured lists.

        One for chat messages and one for system content blocks. This is done by
        categorizing and processing the input based on the role of each message and its content.

        Args:
            messages (list[ChatCompletionMessageParam]): A list of message parameters to be
                processed. Each message contains details such as the role, content, and
                additional metadata required for categorization and processing.

        Returns:
            tuple[list[MessageTypeDef], list[SystemContentBlockTypeDef]]: A tuple where the
            first element is a list of structured chat messages and the second element is a
            list of extracted system content blocks.
        """
        bedrock_messages: list[MessageTypeDef] = []
        system_blocks: list[SystemContentBlockTypeDef] = []

        previous_role_name = ""
        for message_param in messages:
            role_name = message_param.role
            role: ConversationRoleType = (
                "assistant" if role_name == "assistant" else "user"
            )

            if role_name in self._SYSTEM_ROLES:
                system_input: str | list[ChatCompletionContentPartTextParam]
                content_value = message_param.content
                if isinstance(content_value, str):
                    system_input = content_value
                else:
                    # Only text parts are allowed/used for system messages
                    system_input = [
                        p
                        for p in (content_value or [])
                        if isinstance(p, ChatCompletionContentPartTextParam)
                    ]
                system_blocks += self._req_extract_system_content_blocks(system_input)
                continue

            if role_name == "tool":
                tool_msg: ChatCompletionToolMessageParam = message_param  # type: ignore[assignment]
                content_blocks = self._req_extract_tool_blocks(tool_msg)
                if previous_role_name == "tool":
                    # All consecutive tool blocks must be merged
                    bedrock_messages[-1]["content"] += content_blocks  # type: ignore[operator]
                    continue
            elif role_name == "function":
                function_msg: ChatCompletionFunctionMessageParam = message_param  # type: ignore[assignment]
                content_blocks = self._req_extract_function_blocks(function_msg)
            elif role_name == "assistant":
                assistant_msg: ChatCompletionAssistantMessageParam = message_param  # type: ignore[assignment]
                content_blocks = self._req_extract_assistant_blocks(assistant_msg)
            else:
                content_blocks = await self._req_extract_content_blocks(
                    message_param.content
                )

            bedrock_messages.append({"role": role, "content": content_blocks})
            previous_role_name = role_name

        return bedrock_messages, system_blocks

    @staticmethod
    def _req_map_tools(
        request: CompletionCreateParams,
    ) -> list[ChatCompletionToolUnionParam]:
        """Maps the tools and functions from the given request into a unified list of tools.

        If tools are provided within the request, they are directly added to the result.
        If no tools are provided but functions are available, each function is wrapped
        into a `ChatCompletionFunctionToolParam` and added to the resulting list.

        Args:
            request: CompletionCreateParams
                The request object containing the tools and/or functions to be
                processed.

        Returns:
            list[ChatCompletionToolUnionParam]: A list of tools derived from the
            request, including converted function definitions if applicable.
        """
        tools: list[ChatCompletionToolUnionParam] = (
            list(request.tools) if request.tools is not None else []
        )
        if not tools and request.functions is not None:
            tools.extend(
                ChatCompletionFunctionToolParam(
                    type="function",
                    function=FunctionDefinition(
                        name=function_spec.name,
                        description=function_spec.description,
                        parameters=function_spec.parameters,
                    ),
                )
                for function_spec in request.functions
            )
        return tools

    @staticmethod
    def _req_map_tool_choice_literal(value: str) -> ToolChoiceTypeDef:
        """Map OpenAI tool_choice literal to Bedrock ToolChoiceTypeDef.

        Args:
            value: One of 'auto', 'required', 'none'.

        Returns:
            Bedrock toolChoice equivalent or None when no explicit choice.
        """
        if value == "auto":
            return {"auto": {}}
        if value == "required":
            return {"any": {}}
        raise HTTPException(  # pragma: no cover
            status_code=400, detail=f"Unsupported tool choice literal: {value}"
        )

    def _req_map_tool_choice(
        self, tool_choice: ChatCompletionToolChoiceOptionParam | None
    ) -> ToolChoiceTypeDef | None:
        """Convert OpenAI tool_choice union to a Bedrock ToolChoiceTypeDef.

        Args:
            tool_choice: None, a literal ('auto'|'required'|'none'), or a named tool choice.

        Returns:
            The Bedrock-specific toolChoice representation, or None.

        Raises:
            HTTPException: If the tool choice type is unsupported.
        """
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            return self._req_map_tool_choice_literal(tool_choice)
        tool_type = tool_choice.type
        if tool_type == "function":
            function_choice: ChatCompletionNamedToolChoiceParam = tool_choice  # type: ignore[assignment]
            return {"tool": {"name": function_choice.function.name}}
        if tool_type == "custom":
            custom_choice: ChatCompletionNamedToolChoiceCustomParam = tool_choice  # type: ignore[assignment]
            return {"tool": {"name": custom_choice.custom.name}}
        raise HTTPException(  # pragma: no cover
            status_code=400,
            detail=f"Unsupported tool choice type '{tool_type}': {to_json(tool_choice).decode()}",
        )

    def _req_map_function_call(
        self, function_call: FunctionCallParam | None
    ) -> ToolChoiceTypeDef | None:
        """Map legacy function_call to Bedrock ToolChoiceTypeDef.

        Args:
            function_call: Legacy function_call value (literal or dict with name).

        Returns:
            The corresponding Bedrock toolChoice representation, or None.
        """
        if function_call is None:
            return None
        if isinstance(function_call, str):
            return self._req_map_tool_choice_literal(function_call)
        return {"tool": {"name": function_call.name}}

    def _req_map_tool_or_function(
        self, request: CompletionCreateParams
    ) -> ToolChoiceTypeDef | None:
        """Map OpenAI tool_choice/function_call to Bedrock ToolChoiceTypeDef."""
        return self._req_map_tool_choice(
            request.tool_choice
        ) or self._req_map_function_call(request.function_call)

    def _req_map_tool_spec(
        self, tool: ChatCompletionToolUnionParam, tools: list[ToolTypeDef]
    ) -> None:
        """Maps a tool's specification to the provided tools list based on its type.

        Args:
            tool: The tool to be processed and mapped.
            tools: The list where processed tool specifications will be appended.
        """
        tool_type = tool.type
        if tool_type == "function":
            function_tool: ChatCompletionFunctionToolParam = tool  # type: ignore[assignment]
            function_spec = function_tool.function
            name = function_spec.name
            if (
                name.startswith(self._SYSTEM_TOOL_PREFIX)
                and not function_spec.parameters
            ):
                system_tool: SystemToolTypeDef = {
                    "name": name.removeprefix(self._SYSTEM_TOOL_PREFIX)
                }
                tools.append({"systemTool": system_tool})
            else:
                tool_spec: ToolSpecificationTypeDef = {
                    "name": function_spec.name,
                    "description": function_spec.description or tool_type,
                    "inputSchema": {
                        "json": function_spec.parameters or self._EMPTY_TOOL
                    },
                }
                tools.append({"toolSpec": tool_spec})
        else:  # pragma: no cover
            msg = f"Unsupported tool type '{tool_type}': {to_json(tool).decode()}"
            raise OpenaiError(msg)

    def _req_build_tool_config(
        self, request: CompletionCreateParams
    ) -> ToolConfigurationTypeDef | None:
        """Builds a configuration for tools based on the provided request.

        Args:
            request: The request object containing the data
                to map and configure tools.

        Returns:
            The mapped tool configuration object if tools are present, otherwise None.
        """
        tools: list[ToolTypeDef] = []
        for tool in self._req_map_tools(request):
            self._req_map_tool_spec(tool, tools)
        if not tools:
            return None

        tool_config: ToolConfigurationTypeDef = {"tools": tools}
        tool_choice_bedrock = self._req_map_tool_or_function(request)
        if tool_choice_bedrock:
            tool_config["toolChoice"] = tool_choice_bedrock
        return tool_config

    def _req_map_service_tier(
        self, value: ServiceTiers | None
    ) -> tuple[ServiceTierTypeType | None, ServiceTiers | None]:
        """Map OpenAI service tier to Bedrock service tier.

        Args:
            value: OpenAI service tier.

        Returns:
            Bedrock service tier, Effective OpenAI service tier
        """
        if value is None:
            return None, None
        if value in self._SERVICES_TIERS:
            return value, value  # type: ignore[return-value]
        return None, "default"

    @staticmethod
    def _req_parse_prompt_cache_key(prompt_cache_key: str | None) -> set[PromptCaching]:
        """Parses and validates the given prompt cache key.

        Args:
            prompt_cache_key: The cache key string to be parsed.

        Returns:
            A set containing valid keys derived from the input `prompt_cache_key`.
        """
        if prompt_cache_key:
            return (
                set(prompt_cache_key.split(".")) & PROMPT_CACHING  # type: ignore[return-value]
            ) or PROMPT_CACHING
        return set()

    def _req_configure_reasoning(
        self,
        *,
        reasoning_effort: ReasoningEffort | None,
        budget_tokens: int | None,
        max_tokens: int | None,  # noqa: ARG002
        additional_request_fields: dict[str, Any],  # noqa: ARG002
    ) -> None:
        """Configure reasoning parameters for the model.

        Default implementation raises an error. Models that support reasoning
        must override this method.

        Args:
            reasoning_effort: The reasoning effort level.
            budget_tokens: Optional token budget for reasoning.
            max_tokens: Maximum number of tokens allowed for the model.
            additional_request_fields: Request fields to modify with reasoning config.

        Raises:
            OpenaiError: If reasoning is not supported by this model.
        """
        if reasoning_effort is not None or budget_tokens is not None:
            msg = "Reasoning configuration is not supported for this model"
            raise OpenaiError(msg)

    def _req_enable_prompt_caching(
        self,
        system_blocks: list[SystemContentBlockTypeDef],
        tool_config: ToolConfigurationTypeDef | None,
        bedrock_messages: list[MessageTypeDef],
        prompt_caching: set[PromptCaching],
        prompt_caching_ttl: CacheTTLType | None,
    ) -> None:
        """Enables prompt caching for specified components including system blocks, tools, and messages.

        Args:
            system_blocks: A list of system content blocks of type SystemContentBlockTypeDef to
                which the cache point will be appended if "system" is in the prompt_caching set.
            tool_config: An optional tool configuration of type ToolConfigurationTypeDef.
                If "tools" is in prompt_caching and the configuration is provided,
                the cache point is appended to its tools attribute.
            bedrock_messages: A list of message objects of type MessageTypeDef,
                on which caching will be applied if "messages" is in the prompt_caching set.
            prompt_caching: A set of PromptCaching values that specifies the components
                (e.g., "system", "tools", "messages") for which caching should be enabled.
            prompt_caching_ttl: Prompt caching TTL configuration.
        """
        cache_point: dict[Literal["cachePoint"], CachePointBlockTypeDef] = (
            {"cachePoint": {"type": "default", "ttl": prompt_caching_ttl}}
            if prompt_caching_ttl
            else PROMPT_CACHING_DEFAULT
        )

        if self.PROMPT_CACHING_SUPPORTED:
            if "system" in prompt_caching and system_blocks:
                system_blocks.append(cache_point)  # type: ignore[arg-type]
            if "messages" in prompt_caching and bedrock_messages:
                for message in bedrock_messages:
                    message["content"].append(cache_point)  # type: ignore[attr-defined]
            if (
                "tools" in prompt_caching
                and tool_config
                and self.PROMPT_CACHING_TOOL_SUPPORTED
            ):
                tool_config["tools"].append(cache_point)  # type: ignore[attr-defined]

    @staticmethod
    def _validate_no_budget_tokens(budget_tokens: int | None) -> None:
        """Validates that no budget tokens are provided, as they are not supported for this model.

        Args:
            budget_tokens: The number of budget tokens to validate. If not None,
                an error will be raised indicating that this model does not support
                'thinking_budget' and suggests using 'reasoning_effort' instead.
        """
        if budget_tokens is not None:
            msg = "This model do not support 'thinking_budget'. Use 'reasoning_effort' instead."
            raise OpenaiError(msg)
