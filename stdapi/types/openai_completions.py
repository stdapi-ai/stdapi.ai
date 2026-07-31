"""Local OpenAI-compatible completions (``/v1/completions``) types."""

from typing import Literal, Self

from pydantic import Field, model_validator

from stdapi.input_file import InputFileUrl
from stdapi.types import BaseModelRequestWithExtra, BaseModelResponse
from stdapi.types.openai_chat_completions import (
    ChatCompletionStreamOptionsParam as StreamOptions,
)
from stdapi.types.openai_chat_completions import (
    CompletionUsage,
    PromptCacheRetention,
    ServiceTiers,
)

#: Literal type for the completion response object.
CompletionObjectLiteral = Literal["text_completion"]

#: Literal type for completion finish reasons.
CompletionFinishReasonLiteral = Literal["stop", "length", "content_filter"]


# Ref: openai.types.completion_choice.Logprobs
class CompletionLogprobs(BaseModelResponse):
    """Log probability information for a completion choice.

    Always returned as ``None`` — the backend does not surface logprob data.
    """

    tokens: list[str] | None = Field(
        default=None, description="The tokens chosen by the model."
    )
    token_logprobs: list[float] | None = Field(
        default=None, description="Log probabilities of each token."
    )
    top_logprobs: list[dict[str, float]] | None = Field(
        default=None, description="Top log probabilities for each token."
    )
    text_offset: list[int] | None = Field(
        default=None, description="Character offsets into the prompt text."
    )


# Ref: openai.types.completion_choice.CompletionChoice
class CompletionChoice(BaseModelResponse):
    """A single completion choice."""

    text: str = Field(description="The generated text for this choice.")
    index: int = Field(
        default=0, ge=0, description="The index of this choice in the list of choices."
    )
    finish_reason: CompletionFinishReasonLiteral | None = Field(
        default=None, description="The reason the model stopped generating text."
    )
    logprobs: CompletionLogprobs | None = Field(
        default=None, description="Log probability information for this choice."
    )


# Ref: openai.types.completion_create_params.CompletionCreateParams
class CompletionCreateParams(BaseModelRequestWithExtra):
    """Request body for the OpenAI completions API (``POST /v1/completions``)."""

    model: str = Field(
        ..., min_length=1, max_length=255, description="ID of the model to use."
    )
    prompt: InputFileUrl | str | list[InputFileUrl | str] = Field(
        ...,
        description="The prompt(s) to generate completions for: a single string or array "
        "of strings. Non-inline prompts can be a URL, S3 URI, base64 data URI, or Files API reference. "
        "Each file is forwarded using its detected modality; the model errors if unsupported.\n"
        "An array with exactly one text string and file prompts is sent as a single multimodal request. "
        "Other array shapes return one choice per element. "
        "Token arrays are UNSUPPORTED on this implementation.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="The maximum number of tokens to generate. "
        "Prompt tokens plus max_tokens cannot exceed the model's context length.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Higher values (e.g. 0.8) make output more random; "
        "lower values (e.g. 0.2) make it more focused. "
        "We generally recommend altering this or ``top_p`` but not both.",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling: the model considers only tokens within the top ``top_p`` "
        "probability mass. We generally recommend altering this or ``temperature`` but not both.",
    )
    stop: str | list[str] | None = Field(
        default=None,
        description="Up to 4 sequences where the API will stop generating. "
        "The returned text will not contain the stop sequence.",
    )
    stream: bool | None = Field(
        default=None,
        description="If true, partial completion deltas are sent as server-sent events "
        "as they become available, terminated by a ``data: [DONE]`` message.",
    )
    stream_options: StreamOptions | None = Field(
        default=None, description="Options that apply only when ``stream`` is ``True``."
    )
    n: int | None = Field(
        default=None,
        ge=1,
        le=128,
        description="How many completions to generate for each prompt.",
    )
    user: str | None = Field(
        default=None,
        description="Deprecated by OpenAI in favor of ``safety_identifier``. "
        "A unique identifier representing your end-user.",
    )
    service_tier: ServiceTiers | None = Field(
        default=None,
        description="Processing tier used for serving the request (`auto`, `priority`, `flex`).",
    )
    safety_identifier: str | None = Field(
        default=None,
        description="A stable identifier for detecting users who may be violating usage policies. "
        "Prefer a hash of username or email over the raw value.",
    )
    prompt_cache_key: str | None = Field(
        default=None,
        description="Controls prompt caching for similar requests to reduce costs and improve "
        "response times. Any non-empty value enables caching; a dot-separated list of "
        "'system', 'messages', and/or 'tools' scopes it to specific prompt sections. "
        "Custom hash keys are UNSUPPORTED in this implementation.",
    )
    prompt_cache_retention: PromptCacheRetention | None = Field(
        default=None,
        description="The retention policy for the prompt cache. "
        "`in_memory` is applied as 5m, `24h` as 1h.",
    )
    best_of: int | None = Field(
        default=None,
        description="Generates ``best_of`` completions server-side and returns the one with "
        "the highest log probability per token. Results cannot be streamed, and ``best_of`` "
        "must be greater than ``n``.\n"
        "UNSUPPORTED in this implementation.",
    )
    echo: bool | None = Field(
        default=None,
        description="Echo back the prompt in addition to the completion.\n"
        "UNSUPPORTED in this implementation.",
    )
    frequency_penalty: float | None = Field(
        default=None,
        description="Number between -2.0 and 2.0. Positive values penalize tokens by their "
        "existing frequency so far, reducing verbatim repetition.\n"
        "UNSUPPORTED in this implementation.",
    )
    logit_bias: dict[int, float] | None = Field(
        default=None,
        description="Maps token IDs (GPT tokenizer) to a bias value from -100 to 100 to modify "
        "their likelihood of appearing in the completion.\n"
        "UNSUPPORTED in this implementation.",
    )
    logprobs: int | None = Field(
        default=None,
        description="Include the log probabilities on the ``logprobs`` most likely output "
        "tokens (max 5), as well as the chosen tokens.\n"
        "UNSUPPORTED in this implementation.",
    )
    presence_penalty: float | None = Field(
        default=None,
        description="Number between -2.0 and 2.0. Positive values penalize tokens that already "
        "appear in the text so far, encouraging new topics.\n"
        "UNSUPPORTED in this implementation.",
    )
    seed: int | None = Field(
        default=None,
        description="If specified, the system makes a best effort to sample deterministically "
        "for repeated requests with the same ``seed`` and parameters. Determinism is not "
        "guaranteed.\n"
        "UNSUPPORTED in this implementation.",
    )
    suffix: str | None = Field(
        default=None,
        description="The suffix that comes after a completion of inserted text "
        "(OpenAI: ``gpt-3.5-turbo-instruct`` only).\n"
        "UNSUPPORTED in this implementation.",
    )

    @model_validator(mode="after")
    def _validate_prompt_and_streaming(self) -> Self:
        """Reject token-array prompts.

        Raises:
            ValueError: When ``prompt`` is a list of token arrays (unsupported
                here — use strings or file references instead).
        """
        if (
            isinstance(self.prompt, list)
            and self.prompt
            and not isinstance(self.prompt[0], (str, InputFileUrl))
        ):
            msg = (
                "Token array prompts are not supported on this backend. "
                "Provide strings instead."
            )
            raise ValueError(msg)
        self._validate_stop_sequences()
        return self

    def _validate_stop_sequences(self) -> None:
        """Validate stop sequences are not whitespace-only."""
        sequences = [self.stop] if isinstance(self.stop, str) else self.stop or []
        if any(not sequence.strip() for sequence in sequences):
            msg = "Stop sequences must contain at least one non-whitespace character."
            raise ValueError(msg)


# Ref: openai.types.completion.Completion
class Completion(BaseModelResponse):
    """Response for the OpenAI completions API (``POST /v1/completions``)."""

    id: str = Field(description="Unique identifier for the completion.")
    object: CompletionObjectLiteral = Field(
        default="text_completion",
        description="The object type, always ``text_completion``.",
    )
    created: int = Field(
        description="Unix timestamp (in seconds) when the completion was created."
    )
    model: str = Field(description="The model used to generate the completion.")
    choices: list[CompletionChoice] = Field(
        default_factory=list, description="The list of generated completion choices."
    )
    usage: CompletionUsage | None = Field(
        default=None, description="Usage statistics for the completion request."
    )
    system_fingerprint: str | None = Field(
        default=None,
        description="Backend configuration fingerprint that the model ran with.",
    )
    service_tier: ServiceTiers | None = Field(
        default=None, description="Processing tier used to serve the request."
    )
