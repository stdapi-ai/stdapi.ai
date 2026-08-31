"""OpenAI-compatible ``/v1/organization/usage`` and ``/v1/organization/costs`` endpoints.

Reports what this deployment consumed and what it was billed, in time buckets,
from the usage metrics the server publishes. The whole surface is off unless
``usage_api`` is enabled, and it reports the deployment as a whole rather than
one caller, so it is restricted to administrator credentials.
"""

from base64 import urlsafe_b64decode, urlsafe_b64encode
from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Annotated, Final, Literal

from fastapi import APIRouter, Depends, Query

from stdapi.api_errors import ApiError, FeatureUnavailableError
from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.auth import authenticate
from stdapi.aws_cloudwatch import (
    USAGE_FEATURE,
    Series,
    list_series,
    read_series,
    retention_limit,
    validate_values,
)
from stdapi.config import SETTINGS
from stdapi.monitoring import (
    PRINCIPAL,
    TENANT,
    log_error_details,
    log_request_params,
    log_response_params,
)
from stdapi.types.openai_organization_usage import (
    BUCKET_SECONDS,
    AudioSpeechesResult,
    AudioTranscriptionsResult,
    CompletionsResult,
    CostsResult,
    EmbeddingsResult,
    FileSearchesResult,
    ImagesResult,
    ModerationsResult,
    UsageAmount,
    UsagePage,
    UsageTimeBucket,
    WebSearchesResult,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from stdapi.types import BaseModelResponse

_router = APIRouter(
    prefix=f"{SETTINGS.openai_routes_prefix}/v1",
    tags=["Organization Usage", TAG_OPENAI],
)

#: Default and maximum number of buckets a page holds, per bucket width.
_PAGE_LIMITS: Final[dict[str, tuple[int, int]]] = {
    "1m": (60, 1440),
    "1h": (24, 168),
    "1d": (7, 31),
}

#: Same, for the costs endpoint, whose only bucket width is a day.
_COSTS_LIMITS: Final[tuple[int, int]] = (7, 180)

#: Grouping keys the published metrics carry a dimension for, by request key.
_GROUP_DIMENSION: Final[dict[str, str]] = {
    "model": "Model",
    "api_key_id": "ApiKey",
    "user_id": "User",
    "source": "Operation",
}

#: Why a grouping key or filter upstream declares cannot be answered here.
_REFUSALS: Final[dict[str, str]] = {
    "project_id": "this server has no projects, so usage is never attributed to one",
    "project_ids": "this server has no projects, so usage is never attributed to one",
    "batch": "Batch API usage is not reported by the usage endpoints at all; "
    "its spend appears in '/v1/organization/costs'",
    "service_tier": "the service tier a request ran under is not reported apart",
    "size": "the size of a generated image is not reported",
    "sizes": "the size of a generated image is not reported",
    "context_level": "the context size of a web search is not reported",
    "context_levels": "the context size of a web search is not reported",
    "vector_store_id": "file searches are not reported per vector store",
    "vector_store_ids": "file searches are not reported per vector store",
    "line_item": "costs are not reported per product line item",
    "api_key_id": "this server issues no per-caller API keys ('tenant_api_keys')",
    "api_key_ids": "this server issues no per-caller API keys ('tenant_api_keys')",
    "user_id": "usage is not reported per user on this server "
    "('cloudwatch_metrics_user_dimension')",
    "user_ids": "usage is not reported per user on this server "
    "('cloudwatch_metrics_user_dimension')",
}

#: Image sources, and the operation each one is recorded under.
_IMAGE_SOURCES: Final[dict[str, str]] = {
    "image.generation": "images.generations",
    "image.edit": "images.edits",
    "image.variation": "images.variations",
}

#: Operation of an image source, reversed.
_IMAGE_OPERATIONS: Final[dict[str, str]] = {
    operation: source for source, operation in _IMAGE_SOURCES.items()
}

#: Metric counting the backend invocations behind a usage figure.
_REQUESTS: Final = "Requests"

#: Grouping keys every model-backed endpoint shares, each behind its own setting.
_CALLER_KEYS: Final = frozenset({"api_key_id", "user_id"})


@dataclass(frozen=True, slots=True)
class _Endpoint:
    """One upstream usage endpoint and the metrics that answer it."""

    result: type[BaseModelResponse]
    #: Operation dimension values the endpoint aggregates; empty matches any.
    operations: tuple[str, ...]
    #: Result field each metric name contributes to; several may share a field.
    quantities: dict[str, str]
    #: Grouping keys this endpoint can serve, beyond the caller-identifying ones.
    group_keys: frozenset[str]
    #: Count invocations only on the series that carried one of the quantities
    #: above, so a record made beside the endpoint's own request -- a guardrail
    #: policy, the built-in web search, a knowledge base retrieval, the
    #: translation behind an audio translation -- is not read as a request to
    #: it. Off where a backend of the endpoint publishes no quantity of its own.
    requests_follow_quantities: bool = True


_COMPLETION_OPERATIONS: Final = (
    "chat.completions",
    "completions",
    "responses",
    "messages",
    "realtime",
)

_ENDPOINTS: Final[dict[str, _Endpoint]] = {
    "completions": _Endpoint(
        result=CompletionsResult,
        operations=_COMPLETION_OPERATIONS,
        quantities={
            "InputTokens": "input_tokens",
            "OutputTokens": "output_tokens",
            "CachedTokens": "input_cached_tokens",
            "CacheWriteTokens": "input_cache_write_tokens",
            _REQUESTS: "num_model_requests",
        },
        group_keys=frozenset({"model"}),
    ),
    "embeddings": _Endpoint(
        result=EmbeddingsResult,
        operations=("embeddings",),
        quantities={"InputTokens": "input_tokens", _REQUESTS: "num_model_requests"},
        group_keys=frozenset({"model"}),
    ),
    "moderations": _Endpoint(
        result=ModerationsResult,
        operations=("moderations",),
        quantities={"InputTokens": "input_tokens", _REQUESTS: "num_model_requests"},
        group_keys=frozenset({"model"}),
        # A guardrail and Amazon Comprehend are moderating backends here rather
        # than a filter around another call, and neither publishes input
        # tokens, so every record under the operation is a moderation.
        requests_follow_quantities=False,
    ),
    "images": _Endpoint(
        result=ImagesResult,
        operations=tuple(_IMAGE_SOURCES.values()),
        quantities={"OutputImages": "images", _REQUESTS: "num_model_requests"},
        group_keys=frozenset({"model", "source"}),
    ),
    "audio_speeches": _Endpoint(
        result=AudioSpeechesResult,
        operations=("audio.speech",),
        quantities={"InputCharacters": "characters", _REQUESTS: "num_model_requests"},
        group_keys=frozenset({"model"}),
    ),
    "audio_transcriptions": _Endpoint(
        result=AudioTranscriptionsResult,
        operations=("audio.transcriptions", "audio.translations"),
        quantities={"InputSeconds": "seconds", _REQUESTS: "num_model_requests"},
        group_keys=frozenset({"model"}),
    ),
    "file_search_calls": _Endpoint(
        result=FileSearchesResult,
        operations=("vector_stores.search",),
        quantities={_REQUESTS: "num_requests"},
        group_keys=frozenset(),
        # The search itself is the only record the operation carries, and the
        # invocation count is the only quantity it publishes.
        requests_follow_quantities=False,
    ),
    "web_search_calls": _Endpoint(
        result=WebSearchesResult,
        # Web search is recorded as grounding requests, and only ever during a
        # model invocation: 'SearchUnits' is the rerank and knowledge base
        # retrieval unit, which this endpoint does not report.
        operations=_COMPLETION_OPERATIONS,
        quantities={
            "GroundingRequests": "num_requests",
            _REQUESTS: "num_model_requests",
        },
        group_keys=frozenset({"model"}),
    ),
}

#: Answered pages, keyed by the normalised query, with their expiry.
_CACHE: OrderedDict[str, tuple[float, UsagePage]] = OrderedDict()

#: Most cached pages kept before the oldest is dropped.
_CACHE_SIZE: Final = 128

_StartTime = Annotated[
    int,
    Query(ge=0, description="Start of the reported range, in Unix seconds, inclusive."),
]
_EndTime = Annotated[
    int | None,
    Query(
        ge=0,
        description="End of the reported range, in Unix seconds, exclusive. "
        "Defaults to now.",
    ),
]
_BucketWidth = Annotated[
    Literal["1m", "1h", "1d"],
    Query(
        description="Width of each time bucket. Narrow widths are only "
        "available for recent ranges: 1m for the last 15 days, 1h and 1d for "
        "the last 455 days."
    ),
]
_Models = Annotated[
    list[str] | None, Query(description="Report only usage of these models.")
]
_UserIds = Annotated[
    list[str] | None, Query(description="Report only usage of these users.")
]
_ApiKeyIds = Annotated[
    list[str] | None, Query(description="Report only usage of these API keys.")
]
_ProjectIds = Annotated[
    list[str] | None, Query(description="UNSUPPORTED: this server has no projects.")
]
_Limit = Annotated[
    int | None,
    Query(
        ge=1,
        le=1440,
        description="Number of buckets to return. Defaults to 7 for 1d, 24 for "
        "1h and 60 for 1m; the maximum is 31, 168 and 1440 respectively.",
    ),
]
_Page = Annotated[
    str | None,
    Query(description="Cursor from the `next_page` field of a previous response."),
]
_GroupBy = Annotated[
    list[str] | None,
    Query(
        description="Keys to group the reported usage by: 'model' everywhere, "
        "'source' on images, plus 'api_key_id' and 'user_id' where the server "
        "records them. Any other key upstream declares is refused with the "
        "reason it cannot be answered."
    ),
]


def _require_admin() -> None:
    """Refuse a caller that is not an administrator of this deployment.

    Raises:
        ApiError: 403 when the request is authenticated as a tenant or as an
            end user without every scope ``usage_api_admin_scopes`` names.
    """
    message = (
        "This endpoint reports usage for the whole organization and is "
        "restricted to administrator credentials."
    )
    if TENANT.get() is not None:
        raise ApiError(message, status=403)
    if (principal := PRINCIPAL.get()) is None:
        return
    required = frozenset(SETTINGS.usage_api_admin_scopes)
    if not required:
        log_error_details(
            "An authenticated user was refused the usage API because "
            "'usage_api_admin_scopes' is not set: name the scopes an "
            "administrator's token carries to let one read it.",
            level="warning",
        )
        raise ApiError(message, status=403)
    if not required <= principal.scopes:
        raise ApiError(message, status=403)


def _require_metrics(*, costs: bool = False) -> None:
    """Refuse the surface when the settings it reads from are off.

    Args:
        costs: Whether the costs endpoint, which also needs cost tracking, is
            the one being served.

    Raises:
        FeatureUnavailableError: 503 when a required setting is disabled.
    """
    if not SETTINGS.usage_api:
        raise FeatureUnavailableError(
            USAGE_FEATURE,
            "The usage API is disabled: enable 'usage_api', and read what a "
            "query costs before you do.",
        )
    if not SETTINGS.cloudwatch_metrics:
        raise FeatureUnavailableError(
            USAGE_FEATURE,
            "The usage API reports the metrics 'cloudwatch_metrics' publishes, "
            "which is disabled: enable it, and note that usage is only "
            "reported from the moment it is on.",
        )
    if costs and not SETTINGS.cost_tracking:
        raise FeatureUnavailableError(
            USAGE_FEATURE,
            "The costs endpoint reports the cost 'cost_tracking' computes, "
            "which is disabled: enable it alongside 'cloudwatch_metrics'.",
        )


def _refuse(param: str, subject: str) -> None:
    """Refuse a parameter this server measures nothing for, saying why.

    Args:
        param: The request parameter, as the caller wrote it.
        subject: How the refusal names it, e.g. "'models'" or "group_by='size'".

    Raises:
        ApiError: 400, always.
    """
    reason = _REFUSALS.get(param.removeprefix("group_by:"), "it is not reported here")
    error = ApiError(f"{subject} is not available on this server: {reason}.")
    error.param = "group_by" if param.startswith("group_by:") else param
    raise error


def _refuse_unsupported(name: str, value: object) -> None:
    """Refuse a filter this server measures nothing for, when it was sent.

    Args:
        name: The request parameter, as the caller wrote it.
        value: Its value; nothing is refused when it was not sent.

    Raises:
        ApiError: 400 when the filter was sent.
    """
    if value is None or value == []:
        return
    _refuse(name, f"'{name}'")


def _refuse_caller_pair(param: str, subject: str) -> None:
    """Refuse a request naming an API key and a user at once.

    Each identity is recorded on its own, so that no deployment pays to store a
    series for every key-and-user pair it ever serves. Asking for both together
    names a series that was never published, which would otherwise read as a
    report of no usage.

    Args:
        param: The request parameter to blame.
        subject: How the refusal names the two, e.g. "'api_key_ids' and
            'user_ids'".

    Raises:
        ApiError: 400, always.
    """
    error = ApiError(
        f"{subject} cannot be combined on this server: usage is reported per "
        "key or per user, never per pair of the two."
    )
    error.param = param
    raise error


def _check_group_by(group_by: Sequence[str], allowed: frozenset[str]) -> list[str]:
    """Validate the requested grouping keys against what is recorded.

    Args:
        group_by: The keys the caller asked to group by.
        allowed: The keys this endpoint could answer, before the settings that
            decide whether the caller-identifying ones are recorded at all.

    Returns:
        The keys, de-duplicated, in the order they were given.

    Raises:
        ApiError: 400 naming the key that cannot be answered and why.
    """
    keys = list(dict.fromkeys(group_by))
    for key in keys:
        if (
            key not in allowed
            or (key == "api_key_id" and not SETTINGS.tenant_api_keys)
            or (key == "user_id" and not SETTINGS.cloudwatch_metrics_user_dimension)
        ):
            _refuse(f"group_by:{key}", f"group_by='{key}'")
    if set(keys) >= _CALLER_KEYS:
        _refuse_caller_pair("group_by", "group_by='api_key_id' and group_by='user_id'")
    return keys


def _resolve_range(
    start_time: int, end_time: int | None, bucket_width: str
) -> tuple[int, int]:
    """Align a requested range to the bucket grid and check it may be served.

    Args:
        start_time: Range start, in Unix seconds.
        end_time: Range end, in Unix seconds, or None for now.
        bucket_width: The requested bucket width.

    Returns:
        The aligned ``(start, end)`` pair.

    Raises:
        ApiError: 400 when the range is empty, longer than
            ``usage_api_max_range_days``, or older than the width is kept for.
    """
    period = BUCKET_SECONDS[bucket_width]
    now = int(SETTINGS.now().timestamp())
    end = end_time if end_time is not None else now
    if end <= start_time:
        error = ApiError("'end_time' must be later than 'start_time'.")
        error.param = "end_time"
        raise error
    span_days = (end - start_time) // 86400
    if span_days > SETTINGS.usage_api_max_range_days:
        error = ApiError(
            f"The reported range may not span more than "
            f"{SETTINGS.usage_api_max_range_days} days."
        )
        error.param = "start_time"
        raise error
    oldest = now - retention_limit(period)
    if start_time < oldest:
        error = ApiError(
            f"bucket_width='{bucket_width}' is only reported for the last "
            f"{retention_limit(period) // 86400} days. Ask for a later "
            "'start_time', or a wider 'bucket_width'."
        )
        error.param = "start_time"
        raise error
    return start_time - start_time % period, end + (-end % period)


def _page_window(
    page: str | None, limit: int, buckets: int
) -> tuple[int, bool, str | None]:
    """Resolve the cursor into the slice of buckets this page carries.

    Args:
        page: The cursor from a previous response, if any.
        limit: Number of buckets in a page.
        buckets: Total number of buckets in the requested range.

    Returns:
        The ``(offset, has_more, next_page)`` triple.

    Raises:
        ApiError: 400 when the cursor is not one this server issued.
    """
    offset = 0
    if page:
        try:
            offset = int(urlsafe_b64decode(page.encode()).decode())
        except (ValueError, UnicodeDecodeError) as error:
            api_error = ApiError("Invalid 'page' cursor.")
            api_error.param = "page"
            raise api_error from error
        if offset < 0:
            api_error = ApiError("Invalid 'page' cursor.")
            api_error.param = "page"
            raise api_error
    has_more = offset + limit < buckets
    cursor = (
        urlsafe_b64encode(str(offset + limit).encode()).decode() if has_more else None
    )
    return offset, has_more, cursor


def _resolve_limit(limit: int | None, bucket_width: str, *, costs: bool) -> int:
    """Resolve the page size, refusing one wider than upstream allows.

    Args:
        limit: The requested page size, or None for the default.
        bucket_width: The requested bucket width.
        costs: Whether the costs endpoint is being served.

    Returns:
        The number of buckets a page carries.

    Raises:
        ApiError: 400 when the requested size is above the maximum.
    """
    default, maximum = _COSTS_LIMITS if costs else _PAGE_LIMITS[bucket_width]
    if limit is None:
        return default
    if limit > maximum:
        error = ApiError(
            f"'limit' may not exceed {maximum} with bucket_width='{bucket_width}'."
        )
        error.param = "limit"
        raise error
    return limit


def _cached(key: str) -> UsagePage | None:
    """Return a still-valid cached page, dropping it once it has expired.

    Args:
        key: The normalised query the page answered.

    Returns:
        The cached page, or None.
    """
    if not SETTINGS.usage_api_cache_ttl:
        return None
    entry = _CACHE.get(key)
    if entry is None:
        return None
    if entry[0] <= monotonic():
        _CACHE.pop(key, None)
        return None
    _CACHE.move_to_end(key)
    return entry[1]


def _store(key: str, page: UsagePage) -> UsagePage:
    """Cache an answered page for ``usage_api_cache_ttl`` seconds.

    Args:
        key: The normalised query the page answers.
        page: The page to cache.

    Returns:
        The page, unchanged.
    """
    if ttl := SETTINGS.usage_api_cache_ttl:
        _CACHE[key] = (monotonic() + ttl, page)
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_SIZE:
            _CACHE.popitem(last=False)
    return page


def _dimension_names(
    group_by: Sequence[str], filters: dict[str, tuple[str, ...]]
) -> tuple[str, ...]:
    """Return the dimension set the query must read, given what it asks for.

    Args:
        group_by: The requested grouping keys.
        filters: The dimension filters already resolved.

    Returns:
        The dimension names, in the order the metrics are published under.
    """
    names = ["Model", "Operation"]
    for key, dimension in (("api_key_id", "ApiKey"), ("user_id", "User")):
        if key in group_by or filters.get(dimension):
            names.append(dimension)
    return tuple(names)


def _group_key(
    series: Series, group_by: Sequence[str]
) -> tuple[tuple[str, str | None], ...]:
    """Reduce a series to the identity of the result row it belongs to.

    Args:
        series: The series read back from the metrics.
        group_by: The requested grouping keys.

    Returns:
        One ``(result field, value)`` pair per grouping key.
    """
    pairs: list[tuple[str, str | None]] = []
    for key in group_by:
        value = series.dimensions.get(_GROUP_DIMENSION[key])
        if key == "source":
            value = _IMAGE_OPERATIONS.get(value or "")
        pairs.append((key, value))
    return tuple(pairs)


def _build_page(
    endpoint: _Endpoint,
    series_list: list[Series],
    group_by: Sequence[str],
    buckets: Sequence[int],
    period: int,
    *,
    has_more: bool,
    next_page: str | None,
) -> UsagePage:
    """Aggregate the read series into the page of buckets they answer.

    Args:
        endpoint: The endpoint's definition.
        series_list: Every series read back.
        group_by: The requested grouping keys.
        buckets: Start of each bucket this page carries, in Unix seconds.
        period: The bucket width in seconds.
        has_more: Whether a next page exists.
        next_page: The cursor of the next page, if any.

    Returns:
        The page to answer with.
    """
    # Identity of every series that carried a quantity, so the invocation count
    # follows the records that produced the endpoint's own measurement rather
    # than every record sharing their operation.
    quantity_series = {
        tuple(sorted(series.dimensions.items()))
        for series in series_list
        if series.metric != _REQUESTS
    }
    totals: dict[int, dict[tuple[tuple[str, str | None], ...], dict[str, int]]] = {}
    for series in series_list:
        field = endpoint.quantities.get(series.metric)
        if field is None:
            continue
        if (
            series.metric == _REQUESTS
            and endpoint.requests_follow_quantities
            and tuple(sorted(series.dimensions.items())) not in quantity_series
        ):
            continue
        key = _group_key(series, group_by)
        for bucket, value in series.points.items():
            row = totals.setdefault(bucket, {}).setdefault(key, {})
            row[field] = row.get(field, 0) + int(value)
    data = []
    for start in buckets:
        results: list[BaseModelResponse] = []
        for key, row in sorted(totals.get(start, {}).items(), key=_row_order):
            fields: dict[str, object] = dict(key)
            fields.update(row)
            if endpoint.result is CompletionsResult:
                # The metric counts the three input buckets apart, as the backend
                # reports them; input_tokens is declared to cover all three, and
                # only the cache read was read from the cache.
                cached = row.get("input_cached_tokens", 0)
                uncached = row.get("input_tokens", 0) + row.get(
                    "input_cache_write_tokens", 0
                )
                fields["input_uncached_tokens"] = uncached
                fields["input_tokens"] = uncached + cached
            results.append(endpoint.result(**fields))
        data.append(
            UsageTimeBucket(
                start_time=start,
                end_time=start + period,
                results=results,  # type: ignore[arg-type]
            )
        )
    return UsagePage(data=data, has_more=has_more, next_page=next_page)


def _row_order(
    item: tuple[tuple[tuple[str, str | None], ...], object],
) -> tuple[str, ...]:
    """Order result rows deterministically, an unset grouping key sorting first.

    Args:
        item: One ``(group key, accumulated values)`` pair.

    Returns:
        The sort key.
    """
    return tuple(value or "" for _, value in item[0])


async def _serve(
    name: str,
    *,
    start_time: int,
    end_time: int | None,
    bucket_width: str,
    group_by: Sequence[str] | None,
    models: Sequence[str] | None,
    user_ids: Sequence[str] | None,
    api_key_ids: Sequence[str] | None,
    limit: int | None,
    page: str | None,
    operations: Sequence[str] | None = None,
) -> UsagePage:
    """Answer one usage endpoint from the published metrics.

    Args:
        name: The endpoint's own name, e.g. "completions".
        start_time: Range start, in Unix seconds.
        end_time: Range end, in Unix seconds, or None for now.
        bucket_width: The requested bucket width.
        group_by: The requested grouping keys.
        models: Models to restrict the report to.
        user_ids: Users to restrict the report to.
        api_key_ids: API keys to restrict the report to.
        limit: Number of buckets in a page.
        page: Cursor from a previous response.
        operations: Operation values overriding the endpoint's own, used by the
            image sources filter.

    Returns:
        The page of time buckets.
    """
    _require_admin()
    _require_metrics()
    endpoint = _ENDPOINTS[name]
    keys = _check_group_by(group_by or (), endpoint.group_keys | _CALLER_KEYS)
    period = BUCKET_SECONDS[bucket_width]
    size = _resolve_limit(limit, bucket_width, costs=False)
    start, end = _resolve_range(start_time, end_time, bucket_width)
    if not SETTINGS.cloudwatch_metrics_user_dimension:
        _refuse_unsupported("user_ids", user_ids)
    if not SETTINGS.tenant_api_keys:
        _refuse_unsupported("api_key_ids", api_key_ids)
    named = set(keys)
    if api_key_ids:
        named.add("api_key_id")
    if user_ids:
        named.add("user_id")
    if named >= _CALLER_KEYS:
        _refuse_caller_pair("api_key_ids", "'api_key_ids' and 'user_ids'")
    filters: dict[str, tuple[str, ...]] = {
        "Model": validate_values(models or (), "models"),
        "Operation": tuple(
            operations if operations is not None else endpoint.operations
        ),
        "ApiKey": validate_values(api_key_ids or (), "api_key_ids"),
        "User": validate_values(user_ids or (), "user_ids"),
    }
    names = _dimension_names(keys, filters)
    cache_key = repr((name, names, filters, start, end, period, size, page, keys))
    if (cached := _cached(cache_key)) is not None:
        return cached
    total_buckets = (end - start) // period
    offset, has_more, cursor = _page_window(page, size, total_buckets)
    buckets = [
        start + index * period
        for index in range(offset, min(offset + size, total_buckets))
    ]
    metrics = tuple(endpoint.quantities)
    series_list: list[Series] = []
    if buckets:
        await _check_budget(metrics, names, filters)
        series_list = await read_series(
            metrics=metrics,
            dimension_names=names,
            filters=filters,
            start=buckets[0],
            end=buckets[-1] + period,
            period=period,
        )
    return _store(
        cache_key,
        _build_page(
            endpoint,
            series_list,
            keys,
            buckets,
            period,
            has_more=has_more,
            next_page=cursor,
        ),
    )


async def _check_budget(
    metrics: Sequence[str], names: Sequence[str], filters: Mapping[str, Sequence[str]]
) -> None:
    """Refuse a query reading more metric series than the operator allows.

    The listing is free where the read that follows is billed per series, so it
    is what bounds the cost. It is taken under the query's own filters, so a
    narrower query is counted narrower rather than charged for the whole
    deployment. Both the listing and the read find series through the same
    index, which only carries what reported data in the last two weeks, so what
    is counted here is what the read will match -- and a series idle longer
    than that is reported by neither.

    Args:
        metrics: The metric names the query reads.
        names: The dimension set the query reads them under.
        filters: Values the query restricts a dimension to, by dimension name.

    Raises:
        ApiError: 400 when the query matches more series than allowed.
    """
    listed = await list_series(metrics, names, filters)
    if len(listed) > SETTINGS.usage_api_max_metrics:
        message = (
            f"This query reports on more than {SETTINGS.usage_api_max_metrics} "
            "series. Narrow it with 'models', or ask for fewer grouping keys."
        )
        raise ApiError(message)


def _empty_page(
    start_time: int,
    end_time: int | None,
    bucket_width: str,
    limit: int | None,
    page: str | None,
) -> UsagePage:
    """Build the well-formed empty page of an endpoint nothing is measured for.

    An empty page rather than a 404: the request is well-formed and asks for a
    measurement this deployment never takes, and a 404 would tell a client the
    endpoint does not exist.

    Args:
        start_time: Range start, in Unix seconds.
        end_time: Range end, in Unix seconds, or None for now.
        bucket_width: The requested bucket width.
        limit: Number of buckets in a page.
        page: Cursor from a previous response.

    Returns:
        A page whose buckets carry no results.
    """
    _require_admin()
    _require_metrics()
    period = BUCKET_SECONDS[bucket_width]
    size = _resolve_limit(limit, bucket_width, costs=False)
    start, end = _resolve_range(start_time, end_time, bucket_width)
    total = (end - start) // period
    offset, has_more, cursor = _page_window(page, size, total)
    return UsagePage(
        data=[
            UsageTimeBucket(
                start_time=start + index * period, end_time=start + (index + 1) * period
            )
            for index in range(offset, min(offset + size, total))
        ],
        has_more=has_more,
        next_page=cursor,
    )


@_router.get(
    "/organization/usage/completions",
    summary="Report chat, responses and completions usage (OpenAI format)",
    operation_id="openai_organization_usage_completions",
    description=(
        "Reports the tokens consumed by chat, responses, text completion and "
        "realtime requests, in time buckets (OpenAI Usage API).\n\n"
        "Administrator endpoint: it reports the whole deployment, not the "
        "calling client."
    ),
    response_description="A page of time buckets.",
    response_model_exclude_none=True,
)
async def usage_completions(
    start_time: _StartTime,
    end_time: _EndTime = None,
    bucket_width: _BucketWidth = "1d",
    group_by: _GroupBy = None,
    models: _Models = None,
    user_ids: _UserIds = None,
    api_key_ids: _ApiKeyIds = None,
    project_ids: _ProjectIds = None,
    batch: Annotated[
        bool | None,
        Query(
            description="UNSUPPORTED: Batch API usage is not reported by this "
            "endpoint; its spend appears in `/v1/organization/costs`."
        ),
    ] = None,
    limit: _Limit = None,
    page: _Page = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> UsagePage:
    """Report token usage of the model-invoking endpoints.

    Args:
        start_time: Range start, in Unix seconds.
        end_time: Range end, in Unix seconds.
        bucket_width: Width of each time bucket.
        group_by: Keys to group the reported usage by.
        models: Models to restrict the report to.
        user_ids: Users to restrict the report to.
        api_key_ids: API keys to restrict the report to.
        project_ids: Unsupported.
        batch: Unsupported.
        limit: Number of buckets to return.
        page: Cursor from a previous response.

    Returns:
        A page of time buckets.

    Raises:
        ApiError: 400 for a filter or grouping key nothing is recorded for;
            403 when the caller is not an administrator.
    """
    log_request_params({"start_time": start_time, "bucket_width": bucket_width})
    _require_admin()
    _refuse_unsupported("project_ids", project_ids)
    _refuse_unsupported("batch", batch)
    return log_response_params(
        await _serve(
            "completions",
            start_time=start_time,
            end_time=end_time,
            bucket_width=bucket_width,
            group_by=group_by,
            models=models,
            user_ids=user_ids,
            api_key_ids=api_key_ids,
            limit=limit,
            page=page,
        )
    )


@_router.get(
    "/organization/usage/embeddings",
    summary="Report embeddings usage (OpenAI format)",
    operation_id="openai_organization_usage_embeddings",
    description=(
        "Reports the tokens consumed by embeddings requests, in time buckets "
        "(OpenAI Usage API).\n\nAdministrator endpoint: it reports the whole "
        "deployment, not the calling client."
    ),
    response_description="A page of time buckets.",
    response_model_exclude_none=True,
)
async def usage_embeddings(
    start_time: _StartTime,
    end_time: _EndTime = None,
    bucket_width: _BucketWidth = "1d",
    group_by: _GroupBy = None,
    models: _Models = None,
    user_ids: _UserIds = None,
    api_key_ids: _ApiKeyIds = None,
    project_ids: _ProjectIds = None,
    limit: _Limit = None,
    page: _Page = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> UsagePage:
    """Report token usage of the embeddings endpoints.

    Args:
        start_time: Range start, in Unix seconds.
        end_time: Range end, in Unix seconds.
        bucket_width: Width of each time bucket.
        group_by: Keys to group the reported usage by.
        models: Models to restrict the report to.
        user_ids: Users to restrict the report to.
        api_key_ids: API keys to restrict the report to.
        project_ids: Unsupported.
        limit: Number of buckets to return.
        page: Cursor from a previous response.

    Returns:
        A page of time buckets.

    Raises:
        ApiError: 400 for a filter or grouping key nothing is recorded for;
            403 when the caller is not an administrator.
    """
    log_request_params({"start_time": start_time, "bucket_width": bucket_width})
    _require_admin()
    _refuse_unsupported("project_ids", project_ids)
    return log_response_params(
        await _serve(
            "embeddings",
            start_time=start_time,
            end_time=end_time,
            bucket_width=bucket_width,
            group_by=group_by,
            models=models,
            user_ids=user_ids,
            api_key_ids=api_key_ids,
            limit=limit,
            page=page,
        )
    )


@_router.get(
    "/organization/usage/moderations",
    summary="Report moderations usage (OpenAI format)",
    operation_id="openai_organization_usage_moderations",
    description=(
        "Reports the tokens consumed by moderation requests, in time buckets "
        "(OpenAI Usage API). A moderation served without a language model "
        "counts as a request and reports no tokens.\n\nAdministrator endpoint: "
        "it reports the whole deployment, not the calling client."
    ),
    response_description="A page of time buckets.",
    response_model_exclude_none=True,
)
async def usage_moderations(
    start_time: _StartTime,
    end_time: _EndTime = None,
    bucket_width: _BucketWidth = "1d",
    group_by: _GroupBy = None,
    models: _Models = None,
    user_ids: _UserIds = None,
    api_key_ids: _ApiKeyIds = None,
    project_ids: _ProjectIds = None,
    limit: _Limit = None,
    page: _Page = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> UsagePage:
    """Report token usage of the moderations endpoint.

    Args:
        start_time: Range start, in Unix seconds.
        end_time: Range end, in Unix seconds.
        bucket_width: Width of each time bucket.
        group_by: Keys to group the reported usage by.
        models: Models to restrict the report to.
        user_ids: Users to restrict the report to.
        api_key_ids: API keys to restrict the report to.
        project_ids: Unsupported.
        limit: Number of buckets to return.
        page: Cursor from a previous response.

    Returns:
        A page of time buckets.

    Raises:
        ApiError: 400 for a filter or grouping key nothing is recorded for;
            403 when the caller is not an administrator.
    """
    log_request_params({"start_time": start_time, "bucket_width": bucket_width})
    _require_admin()
    _refuse_unsupported("project_ids", project_ids)
    return log_response_params(
        await _serve(
            "moderations",
            start_time=start_time,
            end_time=end_time,
            bucket_width=bucket_width,
            group_by=group_by,
            models=models,
            user_ids=user_ids,
            api_key_ids=api_key_ids,
            limit=limit,
            page=page,
        )
    )


@_router.get(
    "/organization/usage/images",
    summary="Report image generation usage (OpenAI format)",
    operation_id="openai_organization_usage_images",
    description=(
        "Reports the images produced by generation, edit and variation "
        "requests, in time buckets (OpenAI Usage API).\n\nAdministrator "
        "endpoint: it reports the whole deployment, not the calling client."
    ),
    response_description="A page of time buckets.",
    response_model_exclude_none=True,
)
async def usage_images(
    start_time: _StartTime,
    end_time: _EndTime = None,
    bucket_width: _BucketWidth = "1d",
    group_by: _GroupBy = None,
    models: _Models = None,
    user_ids: _UserIds = None,
    api_key_ids: _ApiKeyIds = None,
    project_ids: _ProjectIds = None,
    sources: Annotated[
        list[Literal["image.generation", "image.edit", "image.variation"]] | None,
        Query(description="Report only images produced by these request kinds."),
    ] = None,
    sizes: Annotated[
        list[str] | None, Query(description="UNSUPPORTED: image size is not reported.")
    ] = None,
    limit: _Limit = None,
    page: _Page = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> UsagePage:
    """Report the images produced by the image endpoints.

    Args:
        start_time: Range start, in Unix seconds.
        end_time: Range end, in Unix seconds.
        bucket_width: Width of each time bucket.
        group_by: Keys to group the reported usage by.
        models: Models to restrict the report to.
        user_ids: Users to restrict the report to.
        api_key_ids: API keys to restrict the report to.
        project_ids: Unsupported.
        sources: Request kinds to restrict the report to.
        sizes: Unsupported.
        limit: Number of buckets to return.
        page: Cursor from a previous response.

    Returns:
        A page of time buckets.

    Raises:
        ApiError: 400 for a filter or grouping key nothing is recorded for;
            403 when the caller is not an administrator.
    """
    log_request_params({"start_time": start_time, "bucket_width": bucket_width})
    _require_admin()
    _refuse_unsupported("project_ids", project_ids)
    _refuse_unsupported("sizes", sizes)
    return log_response_params(
        await _serve(
            "images",
            start_time=start_time,
            end_time=end_time,
            bucket_width=bucket_width,
            group_by=group_by,
            models=models,
            user_ids=user_ids,
            api_key_ids=api_key_ids,
            limit=limit,
            page=page,
            operations=tuple(_IMAGE_SOURCES[source] for source in sources)
            if sources
            else None,
        )
    )


@_router.get(
    "/organization/usage/audio_speeches",
    summary="Report speech synthesis usage (OpenAI format)",
    operation_id="openai_organization_usage_audio_speeches",
    description=(
        "Reports the characters synthesized by speech requests, in time "
        "buckets (OpenAI Usage API).\n\nAdministrator endpoint: it reports the "
        "whole deployment, not the calling client."
    ),
    response_description="A page of time buckets.",
    response_model_exclude_none=True,
)
async def usage_audio_speeches(
    start_time: _StartTime,
    end_time: _EndTime = None,
    bucket_width: _BucketWidth = "1d",
    group_by: _GroupBy = None,
    models: _Models = None,
    user_ids: _UserIds = None,
    api_key_ids: _ApiKeyIds = None,
    project_ids: _ProjectIds = None,
    limit: _Limit = None,
    page: _Page = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> UsagePage:
    """Report the characters synthesized by the speech endpoint.

    Args:
        start_time: Range start, in Unix seconds.
        end_time: Range end, in Unix seconds.
        bucket_width: Width of each time bucket.
        group_by: Keys to group the reported usage by.
        models: Models to restrict the report to.
        user_ids: Users to restrict the report to.
        api_key_ids: API keys to restrict the report to.
        project_ids: Unsupported.
        limit: Number of buckets to return.
        page: Cursor from a previous response.

    Returns:
        A page of time buckets.

    Raises:
        ApiError: 400 for a filter or grouping key nothing is recorded for;
            403 when the caller is not an administrator.
    """
    log_request_params({"start_time": start_time, "bucket_width": bucket_width})
    _require_admin()
    _refuse_unsupported("project_ids", project_ids)
    return log_response_params(
        await _serve(
            "audio_speeches",
            start_time=start_time,
            end_time=end_time,
            bucket_width=bucket_width,
            group_by=group_by,
            models=models,
            user_ids=user_ids,
            api_key_ids=api_key_ids,
            limit=limit,
            page=page,
        )
    )


@_router.get(
    "/organization/usage/audio_transcriptions",
    summary="Report transcription usage (OpenAI format)",
    operation_id="openai_organization_usage_audio_transcriptions",
    description=(
        "Reports the seconds of audio transcribed or translated, in time "
        "buckets (OpenAI Usage API).\n\nAdministrator endpoint: it reports the "
        "whole deployment, not the calling client."
    ),
    response_description="A page of time buckets.",
    response_model_exclude_none=True,
)
async def usage_audio_transcriptions(
    start_time: _StartTime,
    end_time: _EndTime = None,
    bucket_width: _BucketWidth = "1d",
    group_by: _GroupBy = None,
    models: _Models = None,
    user_ids: _UserIds = None,
    api_key_ids: _ApiKeyIds = None,
    project_ids: _ProjectIds = None,
    limit: _Limit = None,
    page: _Page = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> UsagePage:
    """Report the audio seconds consumed by the transcription endpoints.

    Args:
        start_time: Range start, in Unix seconds.
        end_time: Range end, in Unix seconds.
        bucket_width: Width of each time bucket.
        group_by: Keys to group the reported usage by.
        models: Models to restrict the report to.
        user_ids: Users to restrict the report to.
        api_key_ids: API keys to restrict the report to.
        project_ids: Unsupported.
        limit: Number of buckets to return.
        page: Cursor from a previous response.

    Returns:
        A page of time buckets.

    Raises:
        ApiError: 400 for a filter or grouping key nothing is recorded for;
            403 when the caller is not an administrator.
    """
    log_request_params({"start_time": start_time, "bucket_width": bucket_width})
    _require_admin()
    _refuse_unsupported("project_ids", project_ids)
    return log_response_params(
        await _serve(
            "audio_transcriptions",
            start_time=start_time,
            end_time=end_time,
            bucket_width=bucket_width,
            group_by=group_by,
            models=models,
            user_ids=user_ids,
            api_key_ids=api_key_ids,
            limit=limit,
            page=page,
        )
    )


@_router.get(
    "/organization/usage/web_search_calls",
    summary="Report web search usage (OpenAI format)",
    operation_id="openai_organization_usage_web_search_calls",
    description=(
        "Reports the web searches run by the built-in search tool, in time "
        "buckets (OpenAI Usage API).\n\nAdministrator endpoint: it reports the "
        "whole deployment, not the calling client."
    ),
    response_description="A page of time buckets.",
    response_model_exclude_none=True,
)
async def usage_web_search_calls(
    start_time: _StartTime,
    end_time: _EndTime = None,
    bucket_width: _BucketWidth = "1d",
    group_by: _GroupBy = None,
    models: _Models = None,
    user_ids: _UserIds = None,
    api_key_ids: _ApiKeyIds = None,
    project_ids: _ProjectIds = None,
    context_levels: Annotated[
        list[str] | None,
        Query(description="UNSUPPORTED: search context size is not reported."),
    ] = None,
    limit: _Limit = None,
    page: _Page = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> UsagePage:
    """Report the web searches run by the built-in search tool.

    Args:
        start_time: Range start, in Unix seconds.
        end_time: Range end, in Unix seconds.
        bucket_width: Width of each time bucket.
        group_by: Keys to group the reported usage by.
        models: Models to restrict the report to.
        user_ids: Users to restrict the report to.
        api_key_ids: API keys to restrict the report to.
        project_ids: Unsupported.
        context_levels: Unsupported.
        limit: Number of buckets to return.
        page: Cursor from a previous response.

    Returns:
        A page of time buckets.

    Raises:
        ApiError: 400 for a filter or grouping key nothing is recorded for;
            403 when the caller is not an administrator.
    """
    log_request_params({"start_time": start_time, "bucket_width": bucket_width})
    _require_admin()
    _refuse_unsupported("project_ids", project_ids)
    _refuse_unsupported("context_levels", context_levels)
    return log_response_params(
        await _serve(
            "web_search_calls",
            start_time=start_time,
            end_time=end_time,
            bucket_width=bucket_width,
            group_by=group_by,
            models=models,
            user_ids=user_ids,
            api_key_ids=api_key_ids,
            limit=limit,
            page=page,
        )
    )


@_router.get(
    "/organization/usage/file_search_calls",
    summary="Report file search usage (OpenAI format)",
    operation_id="openai_organization_usage_file_search_calls",
    description=(
        "Reports the vector store searches run, in time buckets (OpenAI Usage "
        "API). Searches are counted for the deployment as a whole, never per "
        "vector store.\n\nAdministrator endpoint: it reports the whole "
        "deployment, not the calling client."
    ),
    response_description="A page of time buckets.",
    response_model_exclude_none=True,
)
async def usage_file_search_calls(
    start_time: _StartTime,
    end_time: _EndTime = None,
    bucket_width: _BucketWidth = "1d",
    group_by: _GroupBy = None,
    user_ids: _UserIds = None,
    api_key_ids: _ApiKeyIds = None,
    project_ids: _ProjectIds = None,
    vector_store_ids: Annotated[
        list[str] | None,
        Query(description="UNSUPPORTED: searches are not reported per vector store."),
    ] = None,
    limit: _Limit = None,
    page: _Page = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> UsagePage:
    """Report the vector store searches run.

    Args:
        start_time: Range start, in Unix seconds.
        end_time: Range end, in Unix seconds.
        bucket_width: Width of each time bucket.
        group_by: Keys to group the reported usage by.
        user_ids: Users to restrict the report to.
        api_key_ids: API keys to restrict the report to.
        project_ids: Unsupported.
        vector_store_ids: Unsupported.
        limit: Number of buckets to return.
        page: Cursor from a previous response.

    Returns:
        A page of time buckets.

    Raises:
        ApiError: 400 for a filter or grouping key nothing is recorded for;
            403 when the caller is not an administrator.
    """
    log_request_params({"start_time": start_time, "bucket_width": bucket_width})
    _require_admin()
    _refuse_unsupported("project_ids", project_ids)
    _refuse_unsupported("vector_store_ids", vector_store_ids)
    return log_response_params(
        await _serve(
            "file_search_calls",
            start_time=start_time,
            end_time=end_time,
            bucket_width=bucket_width,
            group_by=group_by,
            models=None,
            user_ids=user_ids,
            api_key_ids=api_key_ids,
            limit=limit,
            page=page,
        )
    )


@_router.get(
    "/organization/usage/vector_stores",
    summary="Report vector store storage (OpenAI format)",
    operation_id="openai_organization_usage_vector_stores",
    description=(
        "Reports vector store storage in time buckets (OpenAI Usage API). This "
        "server takes no storage measurement, so the buckets carry no "
        "results.\n\nAdministrator endpoint: it reports the whole deployment, "
        "not the calling client."
    ),
    response_description="A page of empty time buckets.",
    response_model_exclude_none=True,
)
async def usage_vector_stores(
    start_time: _StartTime,
    end_time: _EndTime = None,
    bucket_width: _BucketWidth = "1d",
    group_by: _GroupBy = None,
    project_ids: _ProjectIds = None,
    limit: _Limit = None,
    page: _Page = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> UsagePage:
    """Report vector store storage, which is not measured here.

    Args:
        start_time: Range start, in Unix seconds.
        end_time: Range end, in Unix seconds.
        bucket_width: Width of each time bucket.
        group_by: Keys to group the reported usage by.
        project_ids: Unsupported.
        limit: Number of buckets to return.
        page: Cursor from a previous response.

    Returns:
        A page of time buckets carrying no results.

    Raises:
        ApiError: 400 for a filter or grouping key nothing is recorded for;
            403 when the caller is not an administrator.
    """
    log_request_params({"start_time": start_time, "bucket_width": bucket_width})
    _require_admin()
    _refuse_unsupported("project_ids", project_ids)
    _check_group_by(group_by or (), frozenset())
    return log_response_params(
        _empty_page(start_time, end_time, bucket_width, limit, page)
    )


@_router.get(
    "/organization/usage/code_interpreter_sessions",
    summary="Report code interpreter sessions (OpenAI format)",
    operation_id="openai_organization_usage_code_interpreter_sessions",
    description=(
        "Reports code interpreter sessions in time buckets (OpenAI Usage API). "
        "This server runs no code interpreter, so the buckets carry no "
        "results.\n\nAdministrator endpoint: it reports the whole deployment, "
        "not the calling client."
    ),
    response_description="A page of empty time buckets.",
    response_model_exclude_none=True,
)
async def usage_code_interpreter_sessions(
    start_time: _StartTime,
    end_time: _EndTime = None,
    bucket_width: _BucketWidth = "1d",
    group_by: _GroupBy = None,
    project_ids: _ProjectIds = None,
    limit: _Limit = None,
    page: _Page = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> UsagePage:
    """Report code interpreter sessions, which this server does not run.

    Args:
        start_time: Range start, in Unix seconds.
        end_time: Range end, in Unix seconds.
        bucket_width: Width of each time bucket.
        group_by: Keys to group the reported usage by.
        project_ids: Unsupported.
        limit: Number of buckets to return.
        page: Cursor from a previous response.

    Returns:
        A page of time buckets carrying no results.

    Raises:
        ApiError: 400 for a filter or grouping key nothing is recorded for;
            403 when the caller is not an administrator.
    """
    log_request_params({"start_time": start_time, "bucket_width": bucket_width})
    _require_admin()
    _refuse_unsupported("project_ids", project_ids)
    _check_group_by(group_by or (), frozenset())
    return log_response_params(
        _empty_page(start_time, end_time, bucket_width, limit, page)
    )


@_router.get(
    "/organization/costs",
    summary="Report costs (OpenAI format)",
    operation_id="openai_organization_costs",
    description=(
        "Reports what serving the requests cost, in daily buckets (OpenAI "
        "Costs API). The amount is what the underlying cloud provider bills "
        "this deployment for the work, not what any reseller charges its own "
        "clients.\n\nAdministrator endpoint: it reports the whole deployment, "
        "not the calling client."
    ),
    response_description="A page of daily time buckets.",
    response_model_exclude_none=True,
)
async def organization_costs(
    start_time: _StartTime,
    end_time: _EndTime = None,
    bucket_width: Annotated[
        Literal["1d"], Query(description="Width of each time bucket.")
    ] = "1d",
    group_by: _GroupBy = None,
    api_key_ids: _ApiKeyIds = None,
    project_ids: _ProjectIds = None,
    limit: Annotated[
        int | None,
        Query(ge=1, le=180, description="Number of buckets to return. Defaults to 7."),
    ] = None,
    page: _Page = None,
    _: Annotated[None, Depends(authenticate)] = None,
) -> UsagePage:
    """Report the cost of the work this deployment performed.

    Args:
        start_time: Range start, in Unix seconds.
        end_time: Range end, in Unix seconds.
        bucket_width: Width of each time bucket.
        group_by: Keys to group the reported cost by.
        api_key_ids: API keys to restrict the report to.
        project_ids: Unsupported.
        limit: Number of buckets to return.
        page: Cursor from a previous response.

    Returns:
        A page of time buckets.

    Raises:
        ApiError: 400 for a filter or grouping key nothing is recorded for;
            403 when the caller is not an administrator.
    """
    log_request_params({"start_time": start_time, "bucket_width": bucket_width})
    _require_admin()
    _require_metrics(costs=True)
    _refuse_unsupported("project_ids", project_ids)
    keys = _check_group_by(group_by or (), frozenset({"api_key_id"}))
    if not SETTINGS.tenant_api_keys:
        _refuse_unsupported("api_key_ids", api_key_ids)
    period = BUCKET_SECONDS["1d"]
    size = _resolve_limit(limit, bucket_width, costs=True)
    start, end = _resolve_range(start_time, end_time, bucket_width)
    filters: dict[str, tuple[str, ...]] = {
        "ApiKey": validate_values(api_key_ids or (), "api_key_ids")
    }
    names = ("Model", "Currency") + (
        ("ApiKey",) if "api_key_id" in keys or filters["ApiKey"] else ()
    )
    cache_key = repr(("costs", names, filters, start, end, size, page, keys))
    if (cached := _cached(cache_key)) is not None:
        return log_response_params(cached)
    total_buckets = (end - start) // period
    offset, has_more, cursor = _page_window(page, size, total_buckets)
    buckets = [
        start + index * period
        for index in range(offset, min(offset + size, total_buckets))
    ]
    series_list: list[Series] = []
    if buckets:
        await _check_budget(("Cost",), names, filters)
        series_list = await read_series(
            metrics=("Cost",),
            dimension_names=names,
            filters=filters,
            start=buckets[0],
            end=buckets[-1] + period,
            period=period,
        )
    return log_response_params(
        _store(
            cache_key,
            _build_costs_page(
                series_list, keys, buckets, period, has_more=has_more, next_page=cursor
            ),
        )
    )


def _build_costs_page(
    series_list: list[Series],
    group_by: Sequence[str],
    buckets: Sequence[int],
    period: int,
    *,
    has_more: bool,
    next_page: str | None,
) -> UsagePage:
    """Aggregate cost series into the page of buckets they answer.

    Currencies are never summed together: one result is reported per currency,
    exactly as the cost was recorded.

    Args:
        series_list: Every cost series read back.
        group_by: The requested grouping keys.
        buckets: Start of each bucket this page carries, in Unix seconds.
        period: The bucket width in seconds.
        has_more: Whether a next page exists.
        next_page: The cursor of the next page, if any.

    Returns:
        The page to answer with.
    """
    totals: dict[int, dict[tuple[str, str | None], float]] = {}
    for series in series_list:
        currency = series.dimensions.get("Currency", "").lower()
        key_id = series.dimensions.get("ApiKey") if "api_key_id" in group_by else None
        for bucket, value in series.points.items():
            rows = totals.setdefault(bucket, {})
            rows[currency, key_id] = rows.get((currency, key_id), 0.0) + value
    data = [
        UsageTimeBucket(
            start_time=start,
            end_time=start + period,
            results=[
                CostsResult(
                    amount=UsageAmount(value=value, currency=currency),
                    api_key_id=key_id,
                )
                for (currency, key_id), value in sorted(totals.get(start, {}).items())
            ],
        )
        for start in buckets
    ]
    return UsagePage(data=data, has_more=has_more, next_page=next_page)


router = _router
