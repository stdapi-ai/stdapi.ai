"""Local OpenAI-compatible completions (``/v1/completions``) types."""

from typing import Literal, Self

from pydantic import Field, model_validator

from stdapi.input_file import InputFileUrl
from stdapi.types import BaseModelRequestWithExtra, BaseModelResponse
from stdapi.types.openai_chat_completions import (  # noqa: TC001
    ChatCompletionStreamOptionsParam as StreamOptions,
)
from stdapi.types.openai_chat_completions import (  # noqa: TC001
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
        description="The prompt(s) to generate completions for.\n"
        "Pass a single string or an array of strings.\n"
        "Non-inline prompts can be passed as a URL (`https://...`), "
        "S3 URI (`s3://bucket/key`), base64 data URI "
        "(`data:[<mediatype>][;base64],<data>`), or a Files API reference "
        "(`file-id:<file-id>`). Each file is forwarded to the model using "
        "its detected modality (`image`, `video`, `audio`, `document`); "
        "the model errors if it does not support that modality.\n"
        "Special case: an array containing exactly one text string and one or "
        "more file prompts is sent as a single multimodal request (one choice), "
        "with the text and files packed in input order — the natural 'ask once "
        "using these files as context' pattern. Other array shapes return one "
        "choice per element.\n"
        "Token arrays (`list[int]` / `list[list[int]]`) are UNSUPPORTED on this "
        "implementation.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="The maximum number of tokens that can be generated in the completion.\n"
        "The token count of your prompt plus ``max_tokens`` cannot exceed the model's context length.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="What sampling temperature to use, between 0 and 2.\n"
        "Higher values like 0.8 make the output more random; lower values like 0.2 make "
        "it more focused and deterministic.\n"
        "We generally recommend altering this or ``top_p`` but not both.",
    )
    top_p: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="An alternative to sampling with temperature, called nucleus sampling, "
        "where the model considers the results of the tokens with ``top_p`` probability mass.\n"
        "So 0.1 means only the tokens comprising the top 10% probability mass are considered.\n"
        "We generally recommend altering this or ``temperature`` but not both.",
    )
    stop: str | list[str] | None = Field(
        default=None,
        description="Up to 4 sequences where the API will stop generating further tokens.\n"
        "The returned text will not contain the stop sequence.",
    )
    stream: bool | None = Field(
        default=None,
        description="If set, partial completion deltas are sent as server-sent events as they "
        "become available, terminated by a ``data: [DONE]`` message.",
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
        description="Processing tier used for serving the request "
        "(`auto`, `priority`, `flex`, `default`/`scale`).",
    )
    safety_identifier: str | None = Field(
        default=None,
        description="A stable identifier used to help detect users of your application that may be "
        "violating usage policies.\nThe IDs should be a string that uniquely identifies each user. "
        "We recommend hashing their username or email address, in order to avoid sending any "
        "identifying information.",
    )
    prompt_cache_key: str | None = Field(
        default=None,
        description="Used to cache responses for similar requests.\n"
        "Controls prompt caching for similar requests to reduce costs and improve response times.\n"
        "Set to any non-empty value to enable prompt caching globally on supported models.\n"
        "Set to a dot-separated list of 'system', 'messages', and/or 'tools' to enable caching "
        "only for specific prompt sections.\n"
        "Note: Custom hash keys are UNSUPPORTED in this implementation.",
    )
    prompt_cache_retention: PromptCacheRetention | None = Field(
        default=None,
        description="The retention policy for the prompt cache.\n"
        "OpenAI values are currently mapped to AWS Bedrock possible values for increased "
        "compatibility: in-memory -> 5m, 24h -> 1h.",
    )
    best_of: int | None = Field(
        default=None,
        description="Generates ``best_of`` completions server-side and returns the 'best' "
        "(the one with the highest log probability per token).\n"
        "Results cannot be streamed. When used with ``n``, ``best_of`` controls the number "
        "of candidate completions and ``n`` specifies how many to return — ``best_of`` must "
        "be greater than ``n``.\n"
        "UNSUPPORTED in this implementation.",
    )
    echo: bool | None = Field(
        default=None,
        description="Echo back the prompt in addition to the completion.\n"
        "UNSUPPORTED in this implementation.",
    )
    frequency_penalty: float | None = Field(
        default=None,
        description="Number between -2.0 and 2.0. Positive values penalize new tokens based on "
        "their existing frequency in the text so far, decreasing the model's likelihood to "
        "repeat the same line verbatim.\n"
        "UNSUPPORTED in this implementation.",
    )
    logit_bias: dict[int, float] | None = Field(
        default=None,
        description="Modify the likelihood of specified tokens appearing in the completion.\n"
        "Accepts a JSON object that maps tokens (specified by their token ID in the GPT "
        "tokenizer) to an associated bias value from -100 to 100.\n"
        "UNSUPPORTED in this implementation.",
    )
    logprobs: int | None = Field(
        default=None,
        description="Include the log probabilities on the ``logprobs`` most likely output tokens, "
        "as well as the chosen tokens.\n"
        "The maximum value for ``logprobs`` is 5.\n"
        "UNSUPPORTED in this implementation.",
    )
    presence_penalty: float | None = Field(
        default=None,
        description="Number between -2.0 and 2.0. Positive values penalize new tokens based on "
        "whether they appear in the text so far, increasing the model's likelihood to talk "
        "about new topics.\n"
        "UNSUPPORTED in this implementation.",
    )
    seed: int | None = Field(
        default=None,
        description="If specified, the system makes a best effort to sample deterministically, "
        "such that repeated requests with the same ``seed`` and parameters should return the "
        "same result.\n"
        "Determinism is not guaranteed.\n"
        "UNSUPPORTED in this implementation.",
    )
    suffix: str | None = Field(
        default=None,
        description="The suffix that comes after a completion of inserted text.\n"
        "On OpenAI this is only supported for ``gpt-3.5-turbo-instruct``.\n"
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
        return self


# Ref: openai.types.completion.Completion
class Completion(BaseModelResponse):
    """Response for the OpenAI completions API (``POST /v1/completions``)."""

    id: str = Field(description="Unique identifier for the completion.")
    object: CompletionObjectLiteral = Field(
        default="text_completion",
        description="The object type, always ``text_completion``.",
    )
    created: int = Field(
        description="Unix timestamp (in seconds) of when the completion was created."
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
