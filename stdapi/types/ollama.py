"""Ollama API request and response models.

Ref: https://docs.ollama.com/openapi.yaml
"""

from typing import Literal, Self

from pydantic import ConfigDict, Field, JsonValue, model_validator

from stdapi.config import SETTINGS
from stdapi.monitoring import REQUEST_TIME
from stdapi.types import BaseModelRequest, BaseModelRequestWithExtra, BaseModelResponse

#: Nanoseconds in one second, the unit every Ollama duration is reported in.
_NS_PER_SECOND: int = 1_000_000_000

#: Thinking levels accepted by ``think`` in addition to a boolean.
ThinkLevel = Literal["low", "medium", "high", "max"]

#: Structured-output request: the string ``json``, or a JSON schema object.
ResponseFormat = Literal["json"] | dict[str, JsonValue]

#: Model residency hint: a duration string (``5m``) or a number of seconds.
KeepAlive = str | float

#: Capabilities advertised for a chat model, in the order Ollama publishes them.
CHAT_CAPABILITIES: tuple[str, ...] = ("completion", "tools")

#: Capability advertised for an embedding model.
EMBEDDING_CAPABILITY: str = "embedding"


def created_at() -> str:
    """Return the request's start time in the ISO 8601 form Ollama reports.

    Returns:
        The timestamp every object of this response carries.
    """
    return REQUEST_TIME.get().isoformat()


def streamed_at() -> str:
    """Return the current time in the ISO 8601 form Ollama stamps an event with.

    Every object of a stream carries the moment it was emitted, so a client
    reading consecutive timestamps measures the generation as it happens.

    Returns:
        The timestamp of the event being emitted.
    """
    return SETTINGS.now().isoformat()


def total_duration() -> int:
    """Return the time spent on the request so far, in nanoseconds.

    Returns:
        Wall-clock nanoseconds since the request started.
    """
    return int((SETTINGS.now() - REQUEST_TIME.get()).total_seconds() * _NS_PER_SECOND)


class ModelOptions(BaseModelRequestWithExtra):
    """Runtime options controlling text generation.

    Options tuning a local runner (``num_ctx``, ``num_gpu``, ``num_thread``, …)
    have no equivalent on a hosted backend and are accepted and ignored.
    """

    seed: int | None = Field(
        default=None, description="Random seed for reproducible outputs."
    )
    temperature: float | None = Field(
        default=None, description="Sampling randomness; higher is more random."
    )
    top_k: int | None = Field(
        default=None, description="Limits next-token selection to the K most likely."
    )
    top_p: float | None = Field(
        default=None, description="Nucleus sampling cumulative probability threshold."
    )
    min_p: float | None = Field(
        default=None, description="Minimum probability threshold for token selection."
    )
    stop: str | list[str] | None = Field(
        default=None, description="Stop sequences that halt generation."
    )
    num_predict: int | None = Field(
        default=None, description="Maximum number of tokens to generate."
    )
    presence_penalty: float | None = Field(
        default=None, description="Penalty applied to tokens already present."
    )
    frequency_penalty: float | None = Field(
        default=None, description="Penalty applied in proportion to token frequency."
    )


class ToolCallFunction(BaseModelResponse):
    """Function a tool call invokes."""

    name: str = Field(description="Name of the function to call.")
    arguments: dict[str, JsonValue] = Field(
        default_factory=dict, description="JSON object of arguments to pass."
    )
    index: int | None = Field(
        default=None, description="Position of the call in the tool call list."
    )


class ToolCall(BaseModelResponse):
    """Tool call requested by the model."""

    function: ToolCallFunction = Field(description="Function call details.")


class RequestToolCallFunction(BaseModelRequest):
    """Function of a tool call replayed in the conversation."""

    name: str = Field(description="Name of the function that was called.")
    arguments: dict[str, JsonValue] = Field(
        default_factory=dict, description="JSON object of arguments passed."
    )
    index: int | None = Field(
        default=None, description="Position of the call in the tool call list."
    )


class RequestToolCall(BaseModelRequest):
    """Tool call replayed in an assistant message."""

    function: RequestToolCallFunction = Field(description="Function call details.")


class ToolFunction(BaseModelRequest):
    """Function a tool exposes to the model."""

    name: str = Field(description="Function name exposed to the model.")
    description: str | None = Field(default=None, description="What the function does.")
    parameters: dict[str, JsonValue] = Field(
        default_factory=dict, description="JSON Schema for the function parameters."
    )


class ToolDefinition(BaseModelRequest):
    """Function tool the model may call."""

    type: Literal["function"] = Field(description="Tool type. Always `function`.")
    function: ToolFunction = Field(description="Function exposed to the model.")


class ChatMessage(BaseModelRequest):
    """One message of the conversation."""

    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="Author of the message."
    )
    content: str = Field(default="", description="Message text content.")
    thinking: str | None = Field(
        default=None, description="Thinking text this assistant message came with."
    )
    images: list[str] | None = Field(
        default=None,
        description="Images for multimodal models, base64-encoded or as a URL.",
    )
    tool_calls: list[RequestToolCall] | None = Field(
        default=None, description="Tool calls this assistant message requested."
    )
    tool_name: str | None = Field(
        default=None, description="Name of the tool a `tool` message answers."
    )
    tool_call_id: str | None = Field(
        default=None,
        description="Identifier of the tool call a `tool` message answers.",
    )


class _InferenceRequest(BaseModelRequest):
    """Fields shared by the chat and generate requests."""

    model: str = Field(description="Model name.")
    format: ResponseFormat | None = Field(
        default=None,
        description="Structured output: `json`, or a JSON schema the answer must match.",
    )
    options: ModelOptions | None = Field(
        default=None, description="Runtime generation options."
    )
    stream: bool = Field(
        default=True, description="Return a stream of partial responses."
    )
    think: bool | ThinkLevel | None = Field(
        default=None,
        description="Return the model's thinking separately, optionally at a chosen level.",
    )
    keep_alive: KeepAlive | None = Field(
        default=None,
        description="UNSUPPORTED, accepted and ignored: models are never resident.",
    )
    logprobs: bool | None = Field(
        default=None, description="UNSUPPORTED: log probabilities are not available."
    )
    top_logprobs: int | None = Field(
        default=None, description="UNSUPPORTED: log probabilities are not available."
    )

    @model_validator(mode="after")
    def _reject_logprobs(self) -> Self:
        """Refuse a request whose answer would silently lack log probabilities.

        Returns:
            The validated request.

        Raises:
            ValueError: If log probabilities were requested.
        """
        if self.logprobs or self.top_logprobs is not None:
            msg = (
                "'logprobs' and 'top_logprobs' are not available on this server. "
                "Send the request without them."
            )
            raise ValueError(msg)
        return self


class ChatRequest(_InferenceRequest):
    """Create a chat response request."""

    messages: list[ChatMessage] = Field(
        default_factory=list,
        description=(
            "Conversation history, oldest message first. Empty makes the model "
            "resident without generating anything."
        ),
    )
    tools: list[ToolDefinition] | None = Field(
        default=None, description="Function tools the model may call."
    )


class GenerateRequest(_InferenceRequest):
    """Create a single-prompt response request."""

    prompt: str = Field(default="", description="Text for the model to answer.")
    system: str | None = Field(default=None, description="System prompt.")
    images: list[str] | None = Field(
        default=None,
        description="Images for multimodal models, base64-encoded or as a URL.",
    )
    suffix: str | None = Field(
        default=None, description="UNSUPPORTED: fill-in-the-middle is not available."
    )
    template: str | None = Field(
        default=None, description="UNSUPPORTED: prompt templates are not available."
    )
    context: list[int] | None = Field(
        default=None, description="UNSUPPORTED: token contexts are not available."
    )
    raw: bool | None = Field(
        default=None, description="UNSUPPORTED: raw prompting is not available."
    )

    @model_validator(mode="after")
    def _reject_prompt_level_fields(self) -> Self:
        """Refuse the fields that need the model's own tokenizer and prompt template.

        Returns:
            The validated request.

        Raises:
            ValueError: If one of the unsupported prompt-level fields was set.
        """
        unsupported = [
            name
            for name, value in (
                ("suffix", self.suffix),
                ("template", self.template),
                ("context", self.context),
                ("raw", self.raw),
            )
            if value
        ]
        if unsupported:
            names = ", ".join(f"'{name}'" for name in unsupported)
            msg = (
                f"{names} not available on this server: they need the model's own "
                "prompt template, which is not exposed. Send 'prompt' and 'system' "
                "instead, or use /api/chat."
            )
            raise ValueError(msg)
        return self


class EmbedRequest(BaseModelRequestWithExtra):
    """Create embeddings request."""

    model: str = Field(description="Model name.")
    input: str | list[str] = Field(description="Text, or texts, to embed.")
    dimensions: int | None = Field(
        default=None, description="Number of dimensions of the returned vectors."
    )
    truncate: bool | None = Field(
        default=None,
        description=(
            "UNSUPPORTED, accepted and ignored: inputs longer than the context "
            "window are handled by the backend."
        ),
    )
    keep_alive: KeepAlive | None = Field(
        default=None,
        description="UNSUPPORTED, accepted and ignored: models are never resident.",
    )
    options: ModelOptions | None = Field(
        default=None, description="Runtime options; ignored for embeddings."
    )


class EmbeddingsRequest(BaseModelRequestWithExtra):
    """Create a single embedding request (deprecated, use `/api/embed`)."""

    model: str = Field(description="Model name.")
    prompt: str = Field(default="", description="Text to embed.")
    keep_alive: KeepAlive | None = Field(
        default=None,
        description="UNSUPPORTED, accepted and ignored: models are never resident.",
    )
    options: ModelOptions | None = Field(
        default=None, description="Runtime options; ignored for embeddings."
    )


class ShowRequest(BaseModelRequest):
    """Show model information request."""

    model: str = Field(default="", description="Model name to show.")
    name: str | None = Field(
        default=None, description="Deprecated alias of `model`.", deprecated=True
    )
    verbose: bool | None = Field(
        default=None, description="Include the large verbose fields in the response."
    )

    @model_validator(mode="after")
    def _require_a_model_name(self) -> Self:
        """Check that a model was named, under either accepted field.

        Returns:
            The validated request.

        Raises:
            ValueError: If neither ``model`` nor ``name`` carries a value.
        """
        if not (self.model or self.name):
            msg = "'model' is required."
            raise ValueError(msg)
        return self

    def requested_model(self) -> str:
        """Return the requested model name.

        Returns:
            The value of ``model``, falling back to the legacy ``name`` field.
        """
        return self.model or self.name or ""


class PullRequest(BaseModelRequest):
    """Make a model available request."""

    model: str = Field(description="Name of the model to make available.")
    insecure: bool | None = Field(
        default=None, description="UNSUPPORTED, accepted and ignored."
    )
    stream: bool = Field(default=True, description="Stream progress updates.")


class PushRequest(BaseModelRequest):
    """Publish a model request."""

    model: str = Field(description="Name of the model to publish.")
    insecure: bool | None = Field(default=None, description="UNSUPPORTED.")
    stream: bool = Field(default=True, description="Stream progress updates.")


class CreateRequest(BaseModelRequestWithExtra):
    """Create a model request."""

    model: str = Field(description="Name for the model to create.")


class CopyRequest(BaseModelRequest):
    """Copy a model request."""

    source: str = Field(description="Existing model name to copy from.")
    destination: str = Field(description="New model name to create.")


class DeleteRequest(BaseModelRequest):
    """Delete a model request."""

    model: str = Field(description="Model name to delete.")


class Metrics(BaseModelResponse):
    """Token counts and durations, in nanoseconds.

    Every field is optional upstream and omitted here when the gateway has no
    measurement behind it, rather than reported as zero.
    """

    total_duration: int | None = Field(
        default=None, description="Total time spent answering, in nanoseconds."
    )
    prompt_eval_count: int | None = Field(
        default=None, description="Number of input tokens."
    )
    prompt_eval_duration: int | None = Field(
        default=None, description="Time to the first generated token, in nanoseconds."
    )
    eval_count: int | None = Field(
        default=None, description="Number of generated tokens."
    )
    eval_duration: int | None = Field(
        default=None,
        description="Time spent generating tokens after the first, in nanoseconds.",
    )


class ResponseMessage(BaseModelResponse):
    """Assistant message returned by the chat endpoint."""

    role: Literal["assistant"] = Field(description="Always `assistant`.")
    content: str = Field(default="", description="Assistant message text.")
    thinking: str | None = Field(
        default=None, description="Thinking text, when `think` is enabled."
    )
    tool_calls: list[ToolCall] | None = Field(
        default=None, description="Tool calls the assistant requested."
    )


class ChatResponse(Metrics):
    """Chat response, and the terminal event of a chat stream."""

    model: str = Field(description="Model name, as the request spelled it.")
    created_at: str = Field(description="Creation timestamp, ISO 8601.")
    message: ResponseMessage = Field(description="Assistant message.")
    done: bool = Field(description="Whether the response is complete.")
    done_reason: str | None = Field(
        default=None, description="Why generation stopped: `stop` or `length`."
    )


class GenerateResponse(Metrics):
    """Generate response, and the terminal event of a generate stream."""

    model: str = Field(description="Model name, as the request spelled it.")
    created_at: str = Field(description="Creation timestamp, ISO 8601.")
    response: str = Field(default="", description="Generated text.")
    thinking: str | None = Field(
        default=None, description="Thinking text, when `think` is enabled."
    )
    done: bool = Field(description="Whether the response is complete.")
    done_reason: str | None = Field(
        default=None, description="Why generation stopped: `stop` or `length`."
    )


class EmbedResponse(BaseModelResponse):
    """Embeddings response."""

    model: str = Field(description="Model name, as the request spelled it.")
    embeddings: list[list[float]] = Field(
        description="One embedding vector per input, in request order."
    )
    total_duration: int | None = Field(
        default=None, description="Total time spent answering, in nanoseconds."
    )
    prompt_eval_count: int | None = Field(
        default=None, description="Number of input tokens."
    )


class EmbeddingsResponse(BaseModelResponse):
    """Single embedding response (deprecated, use `/api/embed`)."""

    embedding: list[float] = Field(description="Embedding vector for the prompt.")


class ModelDetailsSummary(BaseModelResponse):
    """Origin of a model's weights.

    Describes a local model file; a hosted model has no such file, so every
    field but the family is reported empty rather than invented.
    """

    parent_model: str = Field(default="", description="Model this one derives from.")
    format: str = Field(default="", description="Model file format.")
    family: str = Field(default="", description="Primary model family.")
    families: list[str] = Field(
        default_factory=list, description="Every family the model belongs to."
    )
    parameter_size: str = Field(default="", description="Approximate parameter count.")
    quantization_level: str = Field(default="", description="Quantization level.")


class ModelSummary(BaseModelResponse):
    """One entry of the model list."""

    name: str = Field(description="Model name.")
    model: str = Field(description="Model name.")
    modified_at: str = Field(description="Last modified timestamp, ISO 8601.")
    size: int = Field(
        default=0, description="Size on disk in bytes; always 0 for a hosted model."
    )
    digest: str = Field(description="Stable identifier derived from the model name.")
    details: ModelDetailsSummary = Field(description="Origin of the model's weights.")


class ListResponse(BaseModelResponse):
    """Model list response."""

    models: list[ModelSummary] = Field(description="Available models.")


class PsResponse(BaseModelResponse):
    """Running model list response."""

    models: list[JsonValue] = Field(
        default_factory=list, description="Models currently loaded in memory."
    )


class ShowResponse(BaseModelResponse):
    """Model information response.

    The five fields describing a local model file are omitted rather than
    reported empty, which is what Ollama Cloud does for a cloud-hosted model.
    """

    # ``model_info`` is upstream's field name and collides with pydantic's
    # protected ``model_`` namespace, which the empty tuple releases.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    license: str | None = Field(default=None, description="License of the model.")
    modelfile: str | None = Field(
        default=None, description="Modelfile the model was built from."
    )
    parameters: str | None = Field(
        default=None, description="Model parameter settings."
    )
    template: str | None = Field(
        default=None, description="Prompt template of the model."
    )
    system: str | None = Field(default=None, description="System prompt of the model.")
    details: ModelDetailsSummary = Field(description="Origin of the model's weights.")
    model_info: dict[str, JsonValue] = Field(
        default_factory=dict, description="Additional model metadata."
    )
    capabilities: list[str] = Field(
        default_factory=list, description="Features the model supports."
    )
    modified_at: str | None = Field(
        default=None, description="Last modified timestamp, ISO 8601."
    )


class VersionResponse(BaseModelResponse):
    """Version response."""

    version: str = Field(
        description="Ollama API version this server is compatible with."
    )


class StatusResponse(BaseModelResponse):
    """Single-status response of a model management operation."""

    status: str = Field(description="Status message.")
