"""Reading published usage metrics back out of Amazon CloudWatch.

Kept out of the route module so the usage endpoints stay thin adapters: this
owns the search expressions, the retention rules that decide whether a bucket
width can be served at all, and the pagination of the underlying reads.
"""

from dataclasses import dataclass, field
from re import compile as regex_compile
from typing import TYPE_CHECKING, Final

from stdapi.api_errors import ApiError, feature_unavailable_guard
from stdapi.aws import get_client
from stdapi.config import AWS_REGION, SETTINGS

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.monitoring import EventLog

#: The feature name a CloudWatch failure is refused to the caller under.
USAGE_FEATURE: Final = "The usage API"

#: IAM actions the usage API needs, named for the operator on a denial.
_USAGE_PERMISSIONS: Final = (
    "cloudwatch:GetMetricData and cloudwatch:ListMetrics on Resource '*' "
    "(CloudWatch metric actions have no resource-level permissions)"
)

#: Longest age, in seconds, CloudWatch still serves a period at full resolution.
#: Shorter periods are aggregated up rather than refused, so a query past the
#: boundary would silently answer at a coarser resolution than it asked for.
RETENTION_SECONDS: Final[dict[int, int]] = {
    60: 15 * 86400,
    3600: 455 * 86400,
    86400: 455 * 86400,
}

#: Characters a dimension value may carry into a search expression. Anything
#: else is refused rather than quoted: the expression is a query language, and
#: caller-supplied values reach it verbatim.
_SAFE_VALUE: Final = regex_compile(r"[A-Za-z0-9._:/@+-]{1,255}").fullmatch

#: Most result pages one query reads; still paginating past it is refused.
_MAX_PAGES: Final = 20


@dataclass(slots=True)
class Series:
    """One metric time series, keyed by the dimension values it was stored under."""

    metric: str
    dimensions: dict[str, str]
    #: Value per bucket, keyed by the bucket's start in Unix seconds.
    points: dict[int, float] = field(default_factory=dict)


def add_usage_api_warnings(start_event: EventLog) -> None:
    """Report at startup what the usage API will and will not answer.

    Args:
        start_event: The startup log event to add the warnings to.
    """
    # Imported here: stdapi.monitoring pulls in this module's neighbours.
    from stdapi.monitoring import add_server_warning  # noqa: PLC0415

    if not SETTINGS.cloudwatch_metrics:
        add_server_warning(
            start_event,
            "Usage API enabled ('usage_api') without 'cloudwatch_metrics': the "
            "organization usage endpoints have nothing to report",
        )
    elif not SETTINGS.cost_tracking:
        add_server_warning(
            start_event,
            "Usage API enabled ('usage_api') without 'cost_tracking': the "
            "organization costs endpoint has nothing to report",
        )
    if SETTINGS.aws_cognito_user_pool_id and not SETTINGS.usage_api_admin_scopes:
        add_server_warning(
            start_event,
            "Usage API administrator scopes not configured "
            "('usage_api_admin_scopes' not set): no user pool token is accepted "
            "on the organization usage endpoints",
        )


def metrics_region() -> RegionName:
    """Return the region the usage metrics are read from.

    Derived rather than asked for: the metrics are extracted from the server's
    own log stream, so they exist in the region the server runs in, whatever
    regions its models are served from.

    Returns:
        The region to read from.
    """
    region: RegionName = SETTINGS.cloudwatch_metrics_region or AWS_REGION  # type: ignore[assignment]
    return region


def retention_limit(period: int) -> int:
    """Return the oldest timestamp *period* is still served at full resolution.

    Args:
        period: The bucket width in seconds.

    Returns:
        The number of seconds before now past which the data is aggregated up.
    """
    return RETENTION_SECONDS[period]


def validate_values(values: Iterable[str], param: str) -> tuple[str, ...]:
    """Check caller-supplied dimension values before they reach a query.

    Args:
        values: The values as the caller wrote them.
        param: The request parameter they came from, for the error message.

    Returns:
        The values, unchanged.

    Raises:
        ApiError: 400 when a value carries a character a dimension never does.
    """
    checked = tuple(values)
    for value in checked:
        if not _SAFE_VALUE(value):
            error = ApiError(f"Invalid value for '{param}': {value!r}.")
            error.param = param
            raise error
    return checked


def _search_expression(
    dimension_names: Sequence[str],
    metric: str,
    filters: Mapping[str, Sequence[str]],
    period: int,
) -> str:
    """Build the search expression matching one metric across a dimension set.

    Args:
        dimension_names: The dimension-set schema the metric is published under.
        metric: The metric name to match.
        filters: Values to restrict a dimension to, by dimension name.
        period: The bucket width in seconds.

    Returns:
        A CloudWatch metric-math ``SEARCH`` expression summing per period.
    """
    schema = ",".join(
        ('"' + SETTINGS.cloudwatch_metrics_namespace + '"', *dimension_names)
    )
    terms = [f"{{{schema}}}", f'MetricName="{metric}"']
    for name in dimension_names:
        values = filters.get(name) or ()
        if not values:
            continue
        rendered = " OR ".join(f'"{value}"' for value in values)
        terms.append(
            f"{name}=({rendered})" if len(values) > 1 else f"{name}={rendered}"
        )
    return f"SEARCH('{' '.join(terms)}', 'Sum', {period})"


def _parse_label(label: str, dimension_names: Sequence[str]) -> dict[str, str] | None:
    """Read the dimension values back out of a result's dynamic label.

    A search expression answers with one result per matched series, and the
    dimension values reach us only through the label CloudWatch resolves for
    each of them, in the order the query declared.

    Args:
        label: The label CloudWatch resolved for the series.
        dimension_names: The dimension-set schema, in label order.

    Returns:
        The dimension values, or None when the label does not carry them, in
        which case the series is dropped rather than mis-attributed.
    """
    parts = label.split("|")
    if len(parts) != len(dimension_names) or not all(parts):
        return None
    return dict(zip(dimension_names, parts, strict=True))


async def list_series(
    metrics: Sequence[str],
    dimension_names: Sequence[str],
    filters: Mapping[str, Sequence[str]],
) -> list[tuple[str, dict[str, str]]]:
    """List the recently published series a query would match.

    Only series published in the last two weeks are listed, which is what
    CloudWatch's own listing returns -- the same index the read discovers
    series through. The dimension values are matched here rather than in the
    request, so one pagination answers a filter of any width; it is used to
    bound a query's cost before paying for it, never as the usage figures
    themselves.

    Args:
        metrics: The metric names of interest.
        dimension_names: The dimension-set schema to match.
        filters: Values to restrict a dimension to, by dimension name; a
            dimension with no value listed is not restricted.

    Returns:
        One ``(metric name, dimension values)`` pair per listed series.
    """
    client = get_client("cloudwatch", metrics_region())
    wanted = frozenset(metrics)
    restricted = {name: frozenset(values) for name, values in filters.items() if values}
    listed: list[tuple[str, dict[str, str]]] = []
    with feature_unavailable_guard(USAGE_FEATURE, missing=_USAGE_PERMISSIONS):
        paginator = client.get_paginator("list_metrics")
        async for page in paginator.paginate(
            Namespace=SETTINGS.cloudwatch_metrics_namespace,
            Dimensions=[{"Name": name} for name in dimension_names],
        ):
            for entry in page.get("Metrics") or ():
                name = entry.get("MetricName", "")
                if name not in wanted:
                    continue
                dimensions = {
                    d["Name"]: d["Value"] for d in entry.get("Dimensions", ())
                }
                if len(dimensions) != len(dimension_names):
                    continue
                if all(
                    dimensions.get(dimension) in values
                    for dimension, values in restricted.items()
                ):
                    listed.append((name, dimensions))
    return listed


async def read_series(
    *,
    metrics: Sequence[str],
    dimension_names: Sequence[str],
    filters: Mapping[str, Sequence[str]],
    start: int,
    end: int,
    period: int,
) -> list[Series]:
    """Read every matching series over a time range, one point per bucket.

    Args:
        metrics: The metric names to read.
        dimension_names: The dimension-set schema they are published under.
        filters: Values to restrict a dimension to, by dimension name.
        start: Range start, in Unix seconds, aligned to *period*.
        end: Range end, in Unix seconds, aligned to *period*.
        period: The bucket width in seconds.

    Returns:
        One entry per metric and dimension combination that carried data.

    Raises:
        ApiError: 400 when the range still paginates past ``_MAX_PAGES``, rather
            than reporting the pages already read as if they were the total.
    """
    label = "|".join(f"${{PROP('Dim.{name}')}}" for name in dimension_names)
    queries = [
        {
            "Id": f"q{index}",
            "Expression": _search_expression(dimension_names, metric, filters, period),
            "Label": label,
            "Period": period,
            "ReturnData": True,
        }
        for index, metric in enumerate(metrics)
    ]
    by_id = {f"q{index}": metric for index, metric in enumerate(metrics)}
    client = get_client("cloudwatch", metrics_region())
    collected: dict[tuple[str, str], Series] = {}
    token: str | None = None
    with feature_unavailable_guard(USAGE_FEATURE, missing=_USAGE_PERMISSIONS):
        for _ in range(_MAX_PAGES):
            response = await client.get_metric_data(
                MetricDataQueries=queries,
                StartTime=start,
                EndTime=end,
                ScanBy="TimestampAscending",
                **({"NextToken": token} if token else {}),
            )
            for result in response.get("MetricDataResults") or ():
                metric = by_id.get(result.get("Id", ""))
                dimensions = _parse_label(result.get("Label", ""), dimension_names)
                if metric is None or dimensions is None:
                    continue
                key = (metric, result["Label"])
                series = collected.get(key)
                if series is None:
                    series = collected[key] = Series(metric, dimensions)
                for timestamp, value in zip(
                    result.get("Timestamps") or (),
                    result.get("Values") or (),
                    strict=False,
                ):
                    bucket = int(timestamp.timestamp())
                    series.points[bucket] = series.points.get(bucket, 0.0) + value
            token = response.get("NextToken")
            if not token:
                break
    if token:
        message = (
            "This query reads more data than one report can carry. Narrow it "
            "with 'models', ask for fewer grouping keys, or ask for a shorter "
            "range."
        )
        raise ApiError(message)
    return list(collected.values())
