"""OpenAI-compatible organization usage and costs types.

One envelope serves every endpoint of the surface: a page of time buckets, each
carrying the result objects of the endpoint that was called. Every grouping key
is nullable and is only filled when that key was named in ``group_by``.
"""

from typing import Annotated, Literal

from pydantic import Field

from stdapi.types import BaseModelResponse

#: Bucket widths a usage query may ask for, and their width in seconds.
BUCKET_SECONDS: dict[str, int] = {"1m": 60, "1h": 3600, "1d": 86400}


class UsageAmount(BaseModelResponse):
    """Monetary amount of a costs result."""

    value: float | None = Field(default=None, description="The amount.")
    currency: str | None = Field(
        default=None, description="Lowercase ISO-4217 currency code of the amount."
    )


class _Result(BaseModelResponse):
    """Fields every usage result object carries."""

    project_id: str | None = Field(
        default=None, description="Set when grouping by project_id."
    )


class _CallerResult(_Result):
    """Usage result identified by the caller as well as the project."""

    user_id: str | None = Field(
        default=None, description="Set when grouping by user_id."
    )
    api_key_id: str | None = Field(
        default=None, description="Set when grouping by api_key_id."
    )


class _ModelResult(_CallerResult):
    """Usage result of a model-backed endpoint."""

    model: str | None = Field(default=None, description="Set when grouping by model.")
    num_model_requests: int = Field(
        default=0, description="Number of requests made to the model."
    )


class CompletionsResult(_ModelResult):
    """Aggregated completions usage of one time bucket."""

    object: Literal["organization.usage.completions.result"] = (
        "organization.usage.completions.result"
    )
    input_tokens: int = Field(
        default=0, description="Input tokens, cached and cache-write tokens included."
    )
    output_tokens: int = Field(default=0, description="Output tokens.")
    input_cached_tokens: int | None = Field(
        default=None, description="Input tokens read from the prompt cache."
    )
    input_cache_write_tokens: int | None = Field(
        default=None, description="Input tokens written to the prompt cache."
    )
    input_uncached_tokens: int | None = Field(
        default=None, description="Input tokens that were not read from the cache."
    )
    batch: bool | None = Field(default=None, description="Set when grouping by batch.")
    service_tier: str | None = Field(
        default=None, description="Set when grouping by service_tier."
    )


class EmbeddingsResult(_ModelResult):
    """Aggregated embeddings usage of one time bucket."""

    object: Literal["organization.usage.embeddings.result"] = (
        "organization.usage.embeddings.result"
    )
    input_tokens: int = Field(default=0, description="Input tokens.")


class ModerationsResult(_ModelResult):
    """Aggregated moderations usage of one time bucket."""

    object: Literal["organization.usage.moderations.result"] = (
        "organization.usage.moderations.result"
    )
    input_tokens: int = Field(default=0, description="Input tokens.")


class ImagesResult(_ModelResult):
    """Aggregated image generation usage of one time bucket."""

    object: Literal["organization.usage.images.result"] = (
        "organization.usage.images.result"
    )
    images: int = Field(default=0, description="Number of images generated.")
    source: str | None = Field(default=None, description="Set when grouping by source.")
    size: str | None = Field(default=None, description="Set when grouping by size.")


class AudioSpeechesResult(_ModelResult):
    """Aggregated speech synthesis usage of one time bucket."""

    object: Literal["organization.usage.audio_speeches.result"] = (
        "organization.usage.audio_speeches.result"
    )
    characters: int = Field(default=0, description="Number of characters synthesized.")


class AudioTranscriptionsResult(_ModelResult):
    """Aggregated transcription usage of one time bucket."""

    object: Literal["organization.usage.audio_transcriptions.result"] = (
        "organization.usage.audio_transcriptions.result"
    )
    seconds: int = Field(default=0, description="Number of seconds transcribed.")


class VectorStoresResult(_Result):
    """Aggregated vector store storage of one time bucket."""

    object: Literal["organization.usage.vector_stores.result"] = (
        "organization.usage.vector_stores.result"
    )
    usage_bytes: int = Field(default=0, description="Bytes stored in vector stores.")


class CodeInterpreterSessionsResult(_Result):
    """Aggregated code interpreter sessions of one time bucket."""

    object: Literal["organization.usage.code_interpreter_sessions.result"] = (
        "organization.usage.code_interpreter_sessions.result"
    )
    num_sessions: int = Field(default=0, description="Number of sessions.")


class FileSearchesResult(_CallerResult):
    """Aggregated file search usage of one time bucket."""

    object: Literal["organization.usage.file_searches.result"] = (
        "organization.usage.file_searches.result"
    )
    num_requests: int = Field(default=0, description="Number of file search requests.")
    vector_store_id: str | None = Field(
        default=None, description="Set when grouping by vector_store_id."
    )


class WebSearchesResult(_ModelResult):
    """Aggregated web search usage of one time bucket."""

    object: Literal["organization.usage.web_searches.result"] = (
        "organization.usage.web_searches.result"
    )
    num_requests: int = Field(default=0, description="Number of web search requests.")
    context_level: str | None = Field(
        default=None, description="Set when grouping by context_level."
    )


class CostsResult(_Result):
    """Aggregated cost of one time bucket."""

    object: Literal["organization.costs.result"] = "organization.costs.result"
    amount: UsageAmount | None = Field(default=None, description="The billed amount.")
    line_item: str | None = Field(
        default=None, description="Set when grouping by line_item."
    )
    api_key_id: str | None = Field(
        default=None, description="Set when grouping by api_key_id."
    )
    quantity: float | None = Field(
        default=None, description="Quantity the amount was billed for."
    )


#: Any result object a time bucket of this surface may carry.
UsageResult = Annotated[
    CompletionsResult
    | EmbeddingsResult
    | ModerationsResult
    | ImagesResult
    | AudioSpeechesResult
    | AudioTranscriptionsResult
    | VectorStoresResult
    | CodeInterpreterSessionsResult
    | FileSearchesResult
    | WebSearchesResult
    | CostsResult,
    Field(discriminator="object"),
]


class UsageTimeBucket(BaseModelResponse):
    """One time bucket of a usage or costs page."""

    object: Literal["bucket"] = "bucket"
    start_time: int = Field(description="Start of the bucket, in Unix seconds.")
    end_time: int = Field(description="End of the bucket, in Unix seconds.")
    results: list[UsageResult] = Field(
        default_factory=list, description="Aggregated results of this bucket."
    )


class UsagePage(BaseModelResponse):
    """A page of time buckets."""

    object: Literal["page"] = "page"
    data: list[UsageTimeBucket] = Field(
        default_factory=list, description="The time buckets of this page."
    )
    has_more: bool = Field(default=False, description="Whether a next page exists.")
    next_page: str | None = Field(
        default=None, description="Cursor to pass as `page` to read the next page."
    )
