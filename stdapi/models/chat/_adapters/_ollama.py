"""Ollama dialect, translated onto the OpenAI Chat Completions path.

The Ollama surface adds no backend call of its own: a request is rewritten as
OpenAI chat completion parameters, and the answer is rewritten back. Both
Amazon Bedrock backends are therefore served by the same code.
"""

from re import compile as compile_regex
from time import perf_counter_ns
from typing import TYPE_CHECKING, Any

from pydantic_core import from_json

from stdapi.config import SETTINGS
from stdapi.types.ollama import (
    ChatRequest,
    ChatResponse,
    GenerateRequest,
    GenerateResponse,
    Metrics,
    ResponseMessage,
    ToolCall,
    ToolCallFunction,
    created_at,
    streamed_at,
    total_duration,
)
from stdapi.types.openai import (
    FunctionDefinition,
    JSONSchema,
    ResponseFormatJSONObject,
    ResponseFormatJSONSchema,
)
from stdapi.types.openai_chat_completions import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionStreamOptionsParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
    CompletionCreateParams,
    FunctionCall,
    ImageURL,
)
from stdapi.utils import to_json_str

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterable

    from pydantic import JsonValue
    from sse_starlette import ServerSentEvent

    from stdapi.types import JsonMapping
    from stdapi.types.ollama import (
        ChatMessage,
        KeepAlive,
        ModelOptions,
        ResponseFormat,
        ThinkLevel,
    )
    from stdapi.types.openai_chat_completions import (
        ChatCompletion,
        ChatCompletionContentPartParam,
        ChatCompletionMessage,
        ChatCompletionMessageParam,
        ChatCompletionMessageToolCallUnion,
    )

#: Ollama's two stop reasons, keyed by the OpenAI finish reason.
_DONE_REASON_BY_FINISH_REASON: dict[str, str] = {
    "stop": "stop",
    "length": "length",
    # ``llm/server.go`` emits nothing else, so the three reasons with no Ollama
    # spelling report the closer of the two.
    "tool_calls": "stop",
    "function_call": "stop",
    "content_filter": "stop",
}

#: Name given to the schema wrapper Ollama's bare ``format`` object lacks.
_SCHEMA_NAME: str = "response"

#: Fallback for an upstream error event whose payload is not an object.
_STREAM_FAILED: str = "The request could not be completed. Retry the request."

#: A ``keep_alive`` of zero, in the duration spellings Ollama parses.
_ZERO_KEEP_ALIVE = compile_regex(r"-?0+(\.0*)?(ns|us|\u00b5s|ms|s|m|h)?")


def load_done_reason(keep_alive: KeepAlive | None) -> str:
    """Return why a request naming no input is done, as Ollama reports it.

    A request carrying neither prompt nor message only makes the named model
    resident, and a ``keep_alive`` of zero evicts it instead.

    Args:
        keep_alive: How long the client asked for the model to stay resident.

    Returns:
        ``"unload"`` when residency was asked to end, ``"load"`` otherwise.
    """
    if keep_alive is None:
        return "load"
    return "unload" if _ZERO_KEEP_ALIVE.fullmatch(str(keep_alive).strip()) else "load"


def load_generate_response(request: GenerateRequest) -> GenerateResponse:
    """Answer a generate request that carries no prompt.

    Nothing is generated and no backend is called: Ollama answers an empty
    prompt by making the model resident, which a hosted model already is.

    Args:
        request: The generate request, as the client sent it.

    Returns:
        The single done object the client expects.
    """
    return GenerateResponse(
        model=request.model,
        created_at=created_at(),
        done=True,
        done_reason=load_done_reason(request.keep_alive),
    )


def load_chat_response(request: ChatRequest) -> ChatResponse:
    """Answer a chat request that carries no message.

    Args:
        request: The chat request, as the client sent it.

    Returns:
        The single done object the client expects, its message empty.
    """
    return ChatResponse(
        model=request.model,
        created_at=created_at(),
        message=ResponseMessage(role="assistant"),
        done=True,
        done_reason=load_done_reason(request.keep_alive),
    )


def _reasoning_of(message: ChatCompletionMessage | JsonMapping) -> str | None:
    """Read the thinking text off a response message or delta.

    Args:
        message: A ``ChatCompletionMessage``/``ChoiceDelta``, or the mapping one
            was serialized to.

    Returns:
        The thinking text, or None when there is none or the operator disabled
        emitting it.
    """
    field = SETTINGS.chat_completions_reasoning_field
    if field == "none":
        return None
    if isinstance(message, dict):
        value = message.get(field)
        return value if isinstance(value, str) else None
    return message.reasoning_content


def _content_parts(
    text: str, images: list[str] | None
) -> str | list[ChatCompletionContentPartParam]:
    """Build the OpenAI user content for one Ollama message.

    Args:
        text: Message text.
        images: Base64 images or URLs carried by the message, if any.

    Returns:
        The text alone when there is no image, else a content-parts list.
    """
    if not images:
        return text
    # An image with no caption is an ordinary Ollama message; the empty text
    # block it would otherwise carry is what the backend refuses.
    parts: list[ChatCompletionContentPartParam] = (
        [ChatCompletionContentPartTextParam(type="text", text=text)] if text else []
    )
    parts.extend(
        ChatCompletionContentPartImageParam(
            # Base64, data URI, URL or S3 URI: input_file.py owns every form.
            type="image_url",
            image_url=ImageURL(url=image),  # type: ignore[arg-type]
        )
        for image in images
    )
    return parts


def _map_messages(messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
    """Translate an Ollama conversation into OpenAI chat messages.

    Ollama tool calls carry no identifier, so one is synthesized per call and
    the tool results that follow are correlated to it by ``tool_call_id`` when
    the client sent one, then by tool name, then in call order.

    Args:
        messages: The conversation, oldest message first.

    Returns:
        The equivalent OpenAI messages.
    """
    mapped: list[ChatCompletionMessageParam] = []
    pending: list[tuple[str, str]] = []
    call_count = 0
    for message in messages:
        match message.role:
            case "system":
                mapped.append(
                    ChatCompletionSystemMessageParam(
                        role="system", content=message.content
                    )
                )
            case "user":
                mapped.append(
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=_content_parts(message.content, message.images),
                    )
                )
            case "assistant":
                tool_calls: list[ChatCompletionMessageToolCallUnion] = []
                for call in message.tool_calls or ():
                    call_id = f"call_{call_count}"
                    call_count += 1
                    pending.append((call_id, call.function.name))
                    tool_calls.append(
                        ChatCompletionMessageFunctionToolCall(
                            type="function",
                            id=call_id,
                            function=FunctionCall(
                                name=call.function.name,
                                arguments=to_json_str(call.function.arguments),
                            ),
                        )
                    )
                mapped.append(
                    ChatCompletionAssistantMessageParam(
                        role="assistant",
                        content=message.content or None,
                        reasoning_content=message.thinking,
                        tool_calls=tool_calls or None,
                    )
                )
            case "tool":
                mapped.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        content=message.content,
                        tool_call_id=_take_tool_call_id(pending, message),
                    )
                )
    return mapped


def _take_tool_call_id(pending: list[tuple[str, str]], message: ChatMessage) -> str:
    """Resolve the tool call a tool result answers, consuming the match.

    A client-sent ``tool_call_id`` selects a pending call and is never returned
    as is: this dialect emits no identifier for a client to echo, so a foreign
    one names a tool call the backend never saw and would be refused.

    Args:
        pending: Synthesized ``(id, name)`` pairs not yet answered, in call order.
        message: The tool result message.

    Returns:
        The identifier of the tool call this result answers.
    """
    index = next(
        (
            i
            for i, (call_id, _) in enumerate(pending)
            if call_id == message.tool_call_id
        ),
        None,
    )
    if index is None:
        index = next(
            (i for i, (_, name) in enumerate(pending) if name == message.tool_name), 0
        )
    if pending:
        return pending.pop(index)[0]
    return "call_0"


def _closed_objects(node: JsonValue) -> JsonValue:
    """Return *node* with every object schema closed to extra properties.

    Ollama constrains decoding to the schema, so an answer never carries a
    property the schema does not name, while the backend refuses a schema that
    does not say so. Adding it is what makes a schema written for Ollama work
    unchanged. Rebuilt rather than modified: the request model is frozen.

    Args:
        node: Any node of the JSON schema the request carried.

    Returns:
        The node, with ``additionalProperties: false`` on each object that left
        it unset.
    """
    if isinstance(node, dict):
        closed: dict[str, JsonValue] = {
            key: _closed_objects(value) for key, value in node.items()
        }
        if closed.get("type") == "object" and "additionalProperties" not in closed:
            closed["additionalProperties"] = False
        return closed
    if isinstance(node, list):
        return [_closed_objects(item) for item in node]
    return node


def _apply_options(
    params: dict[str, Any],
    options: ModelOptions | None,
    request_format: ResponseFormat | None,
) -> None:
    """Fold the Ollama option block and output format into OpenAI parameters.

    Options that tune a local runner have no hosted equivalent and are ignored.

    Args:
        params: OpenAI parameters under construction, modified in place.
        options: The request's ``options`` block, if any.
        request_format: The request's ``format`` field, if any.
    """
    if request_format == "json":
        params["response_format"] = ResponseFormatJSONObject()
    elif isinstance(request_format, dict):
        # Ollama sends a bare schema; the name and wrapper are ours to supply.
        params["response_format"] = ResponseFormatJSONSchema(
            json_schema=JSONSchema(
                name=_SCHEMA_NAME,
                schema=_closed_objects(request_format),  # type: ignore[arg-type]
            )
        )
    if options is None:
        return
    for source, target in (
        ("temperature", "temperature"),
        ("top_p", "top_p"),
        ("top_k", "top_k"),
        ("seed", "seed"),
        ("stop", "stop"),
        ("presence_penalty", "presence_penalty"),
        ("frequency_penalty", "frequency_penalty"),
    ):
        if (value := getattr(options, source)) is not None:
            params[target] = value
    # Ollama's negative values mean "unbounded", which is the default here.
    if options.num_predict is not None and options.num_predict > 0:
        params["max_completion_tokens"] = options.num_predict


def _apply_think(params: dict[str, Any], *, think: bool | ThinkLevel | None) -> None:
    """Fold the Ollama ``think`` field into OpenAI reasoning parameters.

    Args:
        params: OpenAI parameters under construction, modified in place.
        think: The request's ``think`` field.
    """
    if think is None:
        return
    if isinstance(think, bool):
        params["enable_thinking"] = think
    else:
        params["enable_thinking"] = True
        params["reasoning_effort"] = think


def _tools(request: ChatRequest) -> list[ChatCompletionFunctionToolParam] | None:
    """Translate the declared tools into OpenAI function tools.

    Args:
        request: The chat request.

    Returns:
        The function tools, or None when none were declared.
    """
    if not request.tools:
        return None
    return [
        ChatCompletionFunctionToolParam(
            type="function",
            function=FunctionDefinition(
                name=tool.function.name,
                description=tool.function.description,
                parameters=tool.function.parameters or None,
            ),
        )
        for tool in request.tools
    ]


def to_chat_completion_params(
    request: ChatRequest | GenerateRequest, model_id: str
) -> CompletionCreateParams:
    """Translate an Ollama chat or generate request into OpenAI parameters.

    Args:
        request: The Ollama request.
        model_id: Resolved model identifier the completion is created for.

    Returns:
        The equivalent OpenAI chat completion parameters.
    """
    if isinstance(request, ChatRequest):
        messages = _map_messages(request.messages)
        tools = _tools(request)
    else:
        messages = []
        if request.system:
            messages.append(
                ChatCompletionSystemMessageParam(role="system", content=request.system)
            )
        messages.append(
            ChatCompletionUserMessageParam(
                role="user", content=_content_parts(request.prompt, request.images)
            )
        )
        tools = None
    params: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
        "stream": request.stream,
    }
    if tools:
        params["tools"] = tools
    if request.stream:
        # The trailing usage chunk is where the token counts come from.
        params["stream_options"] = ChatCompletionStreamOptionsParam(include_usage=True)
    _apply_options(params, request.options, request.format)
    _apply_think(params, think=request.think)
    return CompletionCreateParams(**params)


def _tool_calls_of(message: ChatCompletionMessage) -> list[ToolCall] | None:
    """Translate assistant tool calls into the Ollama shape.

    Args:
        message: The OpenAI assistant message.

    Returns:
        The tool calls, or None when the message requested none.
    """
    calls = [
        ToolCall(
            function=ToolCallFunction(
                name=call.function.name,
                arguments=_parse_arguments(call.function.arguments),
                index=index,
            )
        )
        for index, call in enumerate(message.tool_calls or ())
        if call.type == "function"
    ]
    return calls or None


def _parse_arguments(arguments: str | None) -> dict[str, Any]:
    """Parse tool call arguments into the JSON object Ollama reports.

    Args:
        arguments: Serialized arguments, which a model may leave malformed.

    Returns:
        The parsed object, empty when it is absent or not an object.
    """
    if not arguments:
        return {}
    try:
        parsed = from_json(arguments)
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _completion_metrics(completion: ChatCompletion) -> Metrics:
    """Build the metric block of a non-streamed answer.

    Only the counts and the gateway's own wall clock are measured on this path:
    a buffered answer carries no prompt/generation split, and there is no model
    load phase to time, so those fields are omitted rather than invented.

    Args:
        completion: The OpenAI chat completion.

    Returns:
        The metrics Ollama reports.
    """
    usage = completion.usage
    return Metrics(
        total_duration=total_duration(),
        prompt_eval_count=usage.prompt_tokens if usage else None,
        eval_count=usage.completion_tokens if usage else None,
    )


def from_chat_completion(completion: ChatCompletion, model: str) -> ChatResponse:
    """Translate an OpenAI chat completion into an Ollama chat response.

    Args:
        completion: The OpenAI chat completion.
        model: Model name as the client spelled it, echoed back.

    Returns:
        The Ollama chat response.
    """
    choice = completion.choices[0]
    return ChatResponse(
        model=model,
        created_at=created_at(),
        message=ResponseMessage(
            role="assistant",
            content=choice.message.content or "",
            thinking=_reasoning_of(choice.message),
            tool_calls=_tool_calls_of(choice.message),
        ),
        done=True,
        done_reason=_DONE_REASON_BY_FINISH_REASON.get(
            choice.finish_reason or "stop", "stop"
        ),
        **_completion_metrics(completion).model_dump(),
    )


def from_chat_completion_as_generate(
    completion: ChatCompletion, model: str
) -> GenerateResponse:
    """Translate an OpenAI chat completion into an Ollama generate response.

    Args:
        completion: The OpenAI chat completion.
        model: Model name as the client spelled it, echoed back.

    Returns:
        The Ollama generate response.
    """
    choice = completion.choices[0]
    return GenerateResponse(
        model=model,
        created_at=created_at(),
        response=choice.message.content or "",
        thinking=_reasoning_of(choice.message),
        done=True,
        done_reason=_DONE_REASON_BY_FINISH_REASON.get(
            choice.finish_reason or "stop", "stop"
        ),
        **_completion_metrics(completion).model_dump(),
    )


class _StreamState:
    """Running totals and timings of one translated stream."""

    __slots__ = (
        "eval_count",
        "first_token_ns",
        "prompt_eval_count",
        "start_ns",
        "tool_calls",
    )

    def __init__(self) -> None:
        """Start measuring at the first upstream event."""
        self.start_ns = perf_counter_ns()
        self.first_token_ns: int | None = None
        self.prompt_eval_count: int | None = None
        self.eval_count: int | None = None
        self.tool_calls: dict[int, tuple[str, list[str]]] = {}

    def mark_first_token(self) -> None:
        """Record the arrival of the first generated token."""
        if self.first_token_ns is None:
            self.first_token_ns = perf_counter_ns()

    def collect(self, chunk: JsonMapping) -> None:
        """Accumulate usage and tool call fragments from one upstream chunk.

        Args:
            chunk: A serialized ``ChatCompletionChunk``.
        """
        if isinstance(usage := chunk.get("usage"), dict):
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            if isinstance(prompt_tokens, int):
                self.prompt_eval_count = prompt_tokens
            if isinstance(completion_tokens, int):
                self.eval_count = completion_tokens

    def add_tool_call_delta(self, delta: JsonMapping) -> None:
        """Accumulate the tool call fragments of one delta.

        Ollama never streams partial arguments, so the calls are assembled here
        and emitted whole once the model has finished requesting them.

        Args:
            delta: The chunk's ``delta`` object.
        """
        calls = delta.get("tool_calls")
        if not isinstance(calls, list):
            return
        for call in calls:
            if not isinstance(call, dict):
                continue
            index = call.get("index")
            if not isinstance(index, int):
                continue
            function = call.get("function")
            if not isinstance(function, dict):
                continue
            name, fragments = self.tool_calls.setdefault(index, ("", []))
            if isinstance(new_name := function.get("name"), str) and new_name:
                name = new_name
            if isinstance(arguments := function.get("arguments"), str):
                fragments.append(arguments)
            self.tool_calls[index] = (name, fragments)

    def assembled_tool_calls(self) -> list[ToolCall] | None:
        """Return the tool calls accumulated over the stream.

        Returns:
            The assembled tool calls, or None when the model requested none.
        """
        if not self.tool_calls:
            return None
        return [
            ToolCall(
                function=ToolCallFunction(
                    name=name,
                    arguments=_parse_arguments("".join(fragments)),
                    index=index,
                )
            )
            for index, (name, fragments) in sorted(self.tool_calls.items())
        ]

    def metrics(self) -> Metrics:
        """Build the metric block of a streamed answer.

        A stream gives a real time to first token, which is the prompt
        evaluation phase as the gateway can measure it; the remainder is the
        generation phase. Both are omitted when nothing was ever generated.

        Returns:
            The metrics Ollama reports.
        """
        prompt_eval_duration = eval_duration = None
        if self.first_token_ns is not None:
            prompt_eval_duration = self.first_token_ns - self.start_ns
            eval_duration = perf_counter_ns() - self.first_token_ns
        return Metrics(
            total_duration=total_duration(),
            prompt_eval_count=self.prompt_eval_count,
            prompt_eval_duration=prompt_eval_duration,
            eval_count=self.eval_count,
            eval_duration=eval_duration,
        )


def _delta_of(chunk: JsonMapping) -> JsonMapping | None:
    """Return the first choice's delta of a serialized chunk.

    Args:
        chunk: A serialized ``ChatCompletionChunk``.

    Returns:
        The delta object, or None for a chunk carrying no choice.
    """
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    delta = choice.get("delta")
    return delta if isinstance(delta, dict) else None


def _finish_reason_of(chunk: JsonMapping) -> str | None:
    """Return the first choice's finish reason of a serialized chunk.

    Args:
        chunk: A serialized ``ChatCompletionChunk``.

    Returns:
        The finish reason, or None when the chunk carries none.
    """
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0]
    if not isinstance(choice, dict):
        return None
    reason = choice.get("finish_reason")
    return reason if isinstance(reason, str) else None


async def _upstream_chunks(
    stream: AsyncIterable[ServerSentEvent], state: _StreamState
) -> AsyncGenerator[JsonMapping]:
    """Yield the parsed chunks of the upstream SSE stream.

    The stream is always read to exhaustion, never abandoned at its last event:
    the wrapper underneath writes its log entry and drains the usage AWS billed
    in a ``finally``, which only runs once the consumer asks for one item past
    the last one. Returning early leaves that to generator finalization.

    Args:
        stream: The OpenAI chat completion SSE stream.
        state: Running state of the translation, updated as chunks arrive.

    Yields:
        Each parsed chunk, and the terminal error envelope the wrapper
        formatted when the generation failed.
    """
    async for event in stream:
        data = event.data
        if not isinstance(data, str) or data == "[DONE]":
            continue
        if event.event == "error":
            # Already the provider-formatted envelope; hand it back as it is.
            parsed = from_json(data)
            yield parsed if isinstance(parsed, dict) else {"error": _STREAM_FAILED}
            continue
        chunk = from_json(data)
        if not isinstance(chunk, dict):
            continue
        state.collect(chunk)
        yield chunk


async def chat_stream(
    stream: AsyncIterable[ServerSentEvent], model: str
) -> AsyncGenerator[JsonMapping]:
    """Translate an OpenAI chat completion stream into Ollama chat events.

    Args:
        stream: The OpenAI chat completion SSE stream.
        model: Model name as the client spelled it, echoed back.

    Yields:
        One Ollama stream event per upstream delta, then the terminal event
        carrying ``done: true`` and the metrics.
    """
    state = _StreamState()
    finish_reason: str | None = None
    error: JsonMapping | None = None
    async for chunk in _upstream_chunks(stream, state):
        if "error" in chunk:
            error = chunk
            continue
        finish_reason = _finish_reason_of(chunk) or finish_reason
        if (delta := _delta_of(chunk)) is None:
            continue
        state.add_tool_call_delta(delta)
        content = delta.get("content")
        thinking = _reasoning_of(delta)
        if not content and not thinking:
            continue
        state.mark_first_token()
        message: JsonMapping = {"role": "assistant", "content": content or ""}
        if thinking:
            message["thinking"] = thinking
        yield {
            "model": model,
            "created_at": streamed_at(),
            "message": message,
            "done": False,
        }
    if error is not None:
        # The generation failed: the client is left with an error line rather
        # than a terminal object claiming the answer is complete.
        yield error
        return
    if (tool_calls := state.assembled_tool_calls()) is not None:
        state.mark_first_token()
        yield {
            "model": model,
            "created_at": streamed_at(),
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    call.model_dump(exclude_none=True) for call in tool_calls
                ],
            },
            "done": False,
        }
    yield {
        "model": model,
        "created_at": streamed_at(),
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": _DONE_REASON_BY_FINISH_REASON.get(
            finish_reason or "stop", "stop"
        ),
        **state.metrics().model_dump(exclude_none=True),
    }


async def generate_stream(
    stream: AsyncIterable[ServerSentEvent], model: str
) -> AsyncGenerator[JsonMapping]:
    """Translate an OpenAI chat completion stream into Ollama generate events.

    Args:
        stream: The OpenAI chat completion SSE stream.
        model: Model name as the client spelled it, echoed back.

    Yields:
        One Ollama stream event per upstream delta, then the terminal event
        carrying ``done: true`` and the metrics.
    """
    state = _StreamState()
    finish_reason: str | None = None
    error: JsonMapping | None = None
    async for chunk in _upstream_chunks(stream, state):
        if "error" in chunk:
            error = chunk
            continue
        finish_reason = _finish_reason_of(chunk) or finish_reason
        if (delta := _delta_of(chunk)) is None:
            continue
        content = delta.get("content")
        thinking = _reasoning_of(delta)
        if not content and not thinking:
            continue
        state.mark_first_token()
        event: JsonMapping = {
            "model": model,
            "created_at": streamed_at(),
            "response": content or "",
            "done": False,
        }
        if thinking:
            event["thinking"] = thinking
        yield event
    if error is not None:
        # The generation failed: the client is left with an error line rather
        # than a terminal object claiming the answer is complete.
        yield error
        return
    yield {
        "model": model,
        "created_at": streamed_at(),
        "response": "",
        "done": True,
        "done_reason": _DONE_REASON_BY_FINISH_REASON.get(
            finish_reason or "stop", "stop"
        ),
        **state.metrics().model_dump(exclude_none=True),
    }
