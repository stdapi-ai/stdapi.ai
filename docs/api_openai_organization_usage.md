---
title: Organization Usage & Costs API - Token, Character and Spend Reporting
description: Report token, character, audio and image consumption and the AWS cost of serving it, in time buckets, through an OpenAI-compatible Administration Usage and Costs API backed by Amazon CloudWatch metrics.
keywords: OpenAI usage API, organization usage endpoint, organization costs endpoint, num_model_requests, bucket_width, group_by model, token usage reporting, AWS cost reporting, admin API usage
---

# Organization Usage & Costs API

Report what this deployment consumed and what it cost, in time buckets, through
the OpenAI Administration [Usage](https://platform.openai.com/docs/api-reference/usage)
and [Costs](https://platform.openai.com/docs/api-reference/usage/costs) API.

```python
from openai import OpenAI

client = OpenAI(base_url="https://your-gateway/v1", api_key="...")

page = client.admin.organization.usage.completions.list(
    start_time=1756000000, bucket_width="1d", limit=7, group_by=["model"]
)
for bucket in page.data:
    for result in bucket.results:
        print(bucket.start_time, result.model, result.input_tokens)
```

!!! warning "Administrator endpoints, disabled by default"
    These endpoints report the whole deployment, not the calling client, and
    every query is billed by Amazon CloudWatch. They are served only when
    [`usage_api`](operations_configuration.md#usage-api) is enabled, and only to
    administrator credentials — see [Who may call it](#authorization).

## :material-toggle-switch: Enabling the Surface

| Setting | Needed for |
|---|---|
| [`cloudwatch_metrics`](operations_configuration.md#cloudwatch-metrics) | Publishing the usage the endpoints read back. Nothing is reported for the period before it was enabled. |
| [`usage_api`](operations_configuration.md#usage-api) | Serving the endpoints at all, and publishing usage per endpoint. |
| [`cost_tracking`](operations_configuration.md#cost-tracking) | `/v1/organization/costs` only. |

With `usage_api` disabled every endpoint answers `503`, as it does when
`cloudwatch_metrics` is off; the server log names whichever setting is missing.

The server role needs `cloudwatch:GetMetricData` and `cloudwatch:ListMetrics` —
see [Usage API IAM permissions](operations_iam_permissions.md#usage-api-iam) —
and the queries are billed per metric read, which
[Usage API query cost](operations_cost_management.md#usage-api-cost) puts a
number on.

## :material-api: Endpoints

| Endpoint | Reports |
|---|---|
| `GET /v1/organization/usage/completions` | Input, output and cached tokens of chat, responses, text completion, messages and realtime requests |
| `GET /v1/organization/usage/embeddings` | Input tokens of embeddings requests |
| `GET /v1/organization/usage/moderations` | Input tokens of moderation requests |
| `GET /v1/organization/usage/images` | Images produced by generation, edit and variation requests |
| `GET /v1/organization/usage/audio_speeches` | Characters synthesized |
| `GET /v1/organization/usage/audio_transcriptions` | Seconds of audio transcribed or translated |
| `GET /v1/organization/usage/web_search_calls` | Searches run by the built-in web search tool |
| `GET /v1/organization/usage/file_search_calls` | Vector store searches run |
| `GET /v1/organization/usage/vector_stores` | Nothing: vector store storage is not measured |
| `GET /v1/organization/usage/code_interpreter_sessions` | Nothing: no code interpreter is served |
| `GET /v1/organization/costs` | What AWS bills this deployment for serving the requests |

The two endpoints that report nothing answer a well-formed page of empty
buckets rather than a `404`: the endpoint exists, and no measurement was taken.

!!! note "`GET /v1/usage` is not served"
    The retired top-level usage endpoint is absent from OpenAI's current API
    surface and from the `openai` package, so there is no shape to mirror. Use
    the `/v1/organization/usage/*` family above.

## :material-format-list-bulleted: Query Parameters

Every endpoint takes:

| Parameter | Default | Notes |
|---|---|---|
| `start_time` | — | **Required.** Unix seconds, inclusive. |
| `end_time` | now | Unix seconds, exclusive. |
| `bucket_width` | `1d` | `1m`, `1h` or `1d`. `/v1/organization/costs` accepts `1d` only. |
| `limit` | 7 (`1d`), 24 (`1h`), 60 (`1m`) | Buckets per page. Maximum 31, 168 and 1440 respectively; 180 on `/v1/organization/costs`. |
| `page` | — | Cursor from the `next_page` of a previous response. |
| `group_by` | — | See [Grouping](#grouping). |
| `models` | — | Report only these models. Not on `/v1/organization/costs` or `file_search_calls`. |
| `user_ids`, `api_key_ids` | — | Report only these callers. Accepted only where that identity is recorded — see [Grouping](#grouping). |

Buckets are aligned to the UTC grid of the requested width, so `start_time` is
rounded down to a bucket boundary and the reported `start_time` of the first
bucket may be earlier than the one asked for.

`/v1/organization/usage/images` additionally takes `sources`
(`image.generation`, `image.edit`, `image.variation`).

## :material-group: Grouping { #grouping }

| `group_by` | Served |
|---|---|
| `model` | :material-check: On every model-backed endpoint. |
| `source` | :material-check: On `images`. |
| `api_key_id` | :material-check: When [tenant API keys](operations_authentication_security.md) are issued. Otherwise refused: the deployment has one key, which identifies itself rather than a caller. |
| `user_id` | :material-check: When [`cloudwatch_metrics_user_dimension`](operations_configuration.md#cloudwatch-metrics-user-dimension) is enabled. Otherwise refused: consumption is not recorded per user by default, because that is one stored metric series per user. |
| `project_id` | :material-close: There are no projects here, so usage is never attributed to one. |
| `batch` | :material-close: Batch API usage is not reported by these endpoints at all — see [Differences from Upstream](#differences-from-upstream). |
| `service_tier` | :material-close: The service tier a request ran under is not reported apart. |
| `size` | :material-close: The size of a generated image is not reported. |
| `context_level` | :material-close: The context size of a web search is not reported. |
| `vector_store_id` | :material-close: File searches are not reported per vector store. |
| `line_item` | :material-close: Costs are not reported per product line item. |

A refused key answers `400` naming the key and the reason. A refused key is
never silently ignored: the grouping *is* the request.

`api_key_id` and `user_id` may each be combined with `model`, but not with each
other: usage is reported per key or per user, never per pair of the two. The
same holds for the `api_key_ids` and `user_ids` filters, and for one of them
combined with a grouping by the other.

A key that was not grouped by is **omitted from the JSON object** rather than
sent as an explicit `null`. The `openai` SDK reads it back as `None` either
way, so only a client parsing the raw JSON sees the difference — read such a
key with `result.get("model")`, never `result["model"]`.

## :material-clock-alert: Bucket Width and How Far Back { #retention }

Narrow buckets are only kept for so long:

| `bucket_width` | Reported for |
|---|---|
| `1m` | The last 15 days |
| `1h` | The last 455 days |
| `1d` | The last 455 days |

A query past the boundary is **refused with `400` naming the limit**, rather
than answered at a coarser resolution than it asked for — a usage report that
silently changes resolution mid-range is a number that ends up in a
spreadsheet. Ask for a later `start_time`, or a wider `bucket_width`.

The span between `start_time` and `end_time` is additionally bounded by
[`usage_api_max_range_days`](operations_configuration.md#usage-api-max-range-days),
92 days by default.

!!! warning "A model idle for two weeks stops being reported"
    The endpoints find the series to read through CloudWatch's metric index,
    which only carries what **reported data in the last 14 days**. A model,
    endpoint, tenant key or user that has served nothing for a fortnight is
    therefore missing from *every* bucket of the answer, however recent the
    range asked for — and so is its spend on `/v1/organization/costs`. Retire a
    model and its history stops being reported two weeks later; the same holds
    for a user or a tenant that went quiet.

    This bounds the surface to reporting on what a deployment is currently
    serving. Where a permanent record is needed, keep the
    [request logs](operations_logging_monitoring.md), which are written per
    request and retained for as long as their log group is.

## :material-cash: Costs { #costs }

`/v1/organization/costs` reports the cost of the work in daily buckets, one
result per currency:

```json
{
  "object": "bucket",
  "start_time": 1756000000,
  "end_time": 1756086400,
  "results": [
    {
      "object": "organization.costs.result",
      "amount": {"value": 1.42, "currency": "usd"}
    }
  ]
}
```

!!! warning "This is the AWS bill, not an invoice to your clients"
    `amount` is what AWS charges this deployment's account for serving the
    requests. It is not what a reseller charges its own customers, and it
    carries no margin.

Amounts are never summed across currencies: a deployment spanning partitions
reports one result per currency.

## :material-shield-account: Who May Call It { #authorization }

| Credential | Accepted |
|---|---|
| The deployment's own API key | :material-check: It is the operator's own credential. |
| An Amazon Cognito token | :material-check: Only when it carries **every** scope in [`usage_api_admin_scopes`](operations_configuration.md#usage-api-admin-scopes). With that list empty, no token is accepted. |
| A tenant API key | :material-close: A per-customer credential is never an administrator credential. |

Anything else answers `403`. Upstream gates the same endpoints on a separate
administrator credential; this is the closest equivalent a deployment can
express.

## :material-alert-circle: Differences from Upstream

- Only the grouping keys and filters in [Grouping](#grouping) are served; the
  rest are refused with an explanation rather than ignored.
- `completions` reports `input_tokens`, `output_tokens`, `input_cached_tokens`,
  `input_cache_write_tokens`, `input_uncached_tokens` and
  `num_model_requests`. `input_tokens` covers the whole prompt, cached and
  cache-write tokens included, so it reconciles with the `prompt_tokens` the
  same requests returned. The text/audio/image token breakdown upstream also
  publishes is not reported.
- `num_model_requests` counts only the records carrying the endpoint's own
  measurement. A guardrail applied around the call, the built-in web search, a
  managed knowledge base retrieval and the translation behind an audio
  translation are each billed apart and recorded under the operation of the
  request they served, so they reach neither the count nor the `group_by=model`
  rows of the endpoint that request went to.
- `moderations` reports `num_model_requests` for every moderation, and
  `input_tokens` only for the ones a language model answered: a guardrail and
  Amazon Comprehend are moderating backends here rather than a filter around
  another call, so their records are counted — and a moderation answered by an
  Amazon Bedrock guardrail counts once per policy that guardrail applies.
- `web_search_calls` reports `num_requests` and `num_model_requests`;
  `context_level` is never reported. `num_model_requests` counts every request
  the model answered in the bucket, not only the ones that called the search
  tool, whenever the two shared the same model, operation and grouping in that
  bucket: the search count is recorded on the model's own usage record, so a
  bucket mixing searching and non-searching traffic to the same model cannot
  be told apart from the metrics alone.
- `file_search_calls` counts searches for the deployment as a whole;
  `vector_store_id` is never reported.
- **Batch API usage is absent from the usage endpoints.** A batch's tokens are
  recorded against the route that collected its results, which carries no
  reported endpoint, so they reach neither `completions` nor `embeddings` — yet
  their spend *is* included in `/v1/organization/costs`. On a deployment
  running significant batch traffic, usage and costs therefore do not
  reconcile, by the whole batch workload.
- Usage is reported from a single region — see
  [`cloudwatch_metrics_region`](operations_configuration.md#cloudwatch-metrics-region).
  A deployment publishing from more than one region reports the one it reads.
- `next_page` is this server's own cursor and is only meaningful to it.

## :material-arrow-right: Next Steps

- [Configuration](operations_configuration.md#usage-api) — the settings behind this surface
- [Cost Management](operations_cost_management.md#usage-api-cost) — what a query costs
- [Logging & Monitoring](operations_logging_monitoring.md) — the metrics it reads
- [IAM Permissions](operations_iam_permissions.md#usage-api-iam) — the two actions it needs
