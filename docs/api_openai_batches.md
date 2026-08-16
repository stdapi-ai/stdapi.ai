---
title: Batch API - Asynchronous Bulk Inference (OpenAI Compatible)
description: Run thousands of chat completion requests asynchronously at a discounted price with an OpenAI-compatible Batch API backed by Amazon Bedrock batch inference.
keywords: OpenAI batch API, bulk inference AWS, batch processing LLM, discounted inference, JSONL batch requests, Amazon Bedrock batch inference, asynchronous completions
---

# Batch API

Run a large set of chat completion requests asynchronously, at a lower price than the synchronous API, through the OpenAI Batch API shape.

A batch is created from a file of requests, runs without a connection held open, and is read back from the result files it produces — exactly the OpenAI workflow, so the official OpenAI SDKs work by changing the base URL.

## Why Choose the Batch API?

<div class="grid cards" markdown>

- :material-tag-arrow-down: __Lower Price per Token__
  <br>Batched requests are billed at the published batch rate, well below the on-demand rate for the same model.

- :material-swap-horizontal: __Drop-in OpenAI Compatibility__
  <br>`client.batches.create(...)` and `client.batches.retrieve(...)` work unchanged, as does the JSONL input format.

- :material-server-off: __No Connection to Hold__
  <br>Submit and walk away. Results stay readable through the [Files API](api_openai_files.md) once the batch ends.

- :material-cloud-lock: __Private AWS Backend__
  <br>Requests and results are stored in your own S3 buckets — no traffic to third-party endpoints.

</div>

## Prerequisites

The Batch API is disabled until the deployment declares an AWS IAM service role that Amazon Bedrock assumes to read the requests and write the results:

- [`AWS_BEDROCK_BATCH_ROLE_ARN`](operations_configuration.md#aws-bedrock-batch-role-arn) — the service role.
- [`AWS_S3_BUCKET`](operations_configuration.md#aws-s3-bucket) — the bucket holding the batch data.
- [`AWS_S3_BATCHES_PREFIX`](operations_configuration.md#aws-s3-batches-prefix) — the prefix it is stored under.

The permissions the role and the server need are listed in [IAM Permissions](operations_iam_permissions.md#batch-inference).
While the role is unset, every batch endpoint answers `503`.

## Workflow

### 1. Upload the requests

One JSON object per line, uploaded with `purpose="batch"`. Every line names the same model, carries a unique `custom_id`, and holds the request itself in `body`.

```json
{"custom_id": "req-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "amazon.nova-micro-v1:0", "messages": [{"role": "user", "content": "Summarize: ..."}]}}
{"custom_id": "req-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "amazon.nova-micro-v1:0", "messages": [{"role": "user", "content": "Summarize: ..."}]}}
```

The sample above is abridged: a real file needs at least 100 requests, the minimum a batch carries for each model it names.

```python
from openai import OpenAI

client = OpenAI(base_url="https://your-host/v1", api_key="...")

requests_file = client.files.create(file=open("requests.jsonl", "rb"), purpose="batch")
```

### 2. Create the batch

```python
batch = client.batches.create(
    input_file_id=requests_file.id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
)
```

**Example request (curl):**

```bash
curl -X POST "https://your-host/v1/batches" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input_file_id": "file-06fvfg3lbdqarbad8kbo55g0sg5h3s4a",
    "endpoint": "/v1/chat/completions",
    "completion_window": "24h"
  }'
```

**Example response:**

```json
{
  "id": "batch_06fvfg3lbdqarbad8kbo55g0sg5h3s4a",
  "object": "batch",
  "endpoint": "/v1/chat/completions",
  "input_file_id": "file-06fvfg3lbdqarbad8kbo55g0sg5h3s4a",
  "completion_window": "24h",
  "status": "validating",
  "created_at": 1786568013,
  "expires_at": 1786654413,
  "request_counts": {"total": 100, "completed": 0, "failed": 0},
  "model": "amazon.nova-micro-v1:0"
}
```

### 3. Poll until it ends

```python
batch = client.batches.retrieve(batch.id)
print(batch.status, batch.request_counts)
```

### 4. Read the results

```python
results = client.files.content(batch.output_file_id).text
```

Each line pairs a `custom_id` with the completion it produced:

```json
{"id": "batch_req_9f2c...", "custom_id": "req-1", "response": {"status_code": 200, "request_id": "batch_req_9f2c...", "body": {"id": "chatcmpl-req-1", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 22, "completion_tokens": 9, "total_tokens": 31}}}, "error": null}
```

Requests that failed are collected in a separate file, named by `error_file_id`.

!!! tip "Expiring the result files"
    Result files are kept until deleted, and are billed as stored objects
    meanwhile. Pass `output_expires_after` when creating the batch to have both
    files expire on their own — between 1 hour and 30 days, counted from the
    moment they are written:

    ```python
    batch = client.batches.create(
        input_file_id=requests_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        output_expires_after={"anchor": "created_at", "seconds": 7 * 24 * 3600},
    )
    ```

!!! warning "Results Are Not in Request Order"
    Output lines may come back in any order, as upstream also warns. Match a result to its request with `custom_id`, never with the line number.

## Limits

| Limit                          | Value                        |
|--------------------------------|------------------------------|
| Minimum requests per model     | 100 (default quota)          |
| Maximum requests per batch     | 50,000                       |
| Maximum input file size        | 200 MB                       |
| `custom_id` length             | 64 characters                |
| Distinct models per input file | 1 (upstream rule)            |
| Processing window              | 24 hours from creation       |

A batch below the minimum, or past any of these caps, is refused when it is created and the message names the shortfall, rather than accepted and failed later.

!!! note "The 100-request minimum is a quota default"
    100 is the default of the Amazon Bedrock quota *Minimum number of records per batch inference job*, which is set **per model** and adjustable for some of them — see [Amazon Bedrock quotas](https://docs.aws.amazon.com/general/latest/gr/bedrock.html). The check applied here is the default, whatever your account's own value is: an account that raised the quota has its smaller batches accepted here and refused on creation, and one that lowered it still cannot submit fewer than 100 requests for a model.

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                          |                  Status                  | Notes                                                                       |
|----------------------------------|:----------------------------------------:|-----------------------------------------------------------------------------|
| **Creation**                     |                                          |                                                                             |
| `input_file_id`                  |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Must be uploaded with `purpose="batch"`                                     |
| `endpoint`                       |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | `/v1/chat/completions` only                                                 |
| `completion_window`              |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | `24h`, as upstream                                                          |
| `metadata`                       |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Up to 16 key-value pairs, returned on every read                            |
| `output_expires_after`           |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | `anchor: "created_at"` and 1 hour to 30 days, counted from the moment the result files are written; omit it to keep them until deleted |
| **Per-request body**             |                                          |                                                                             |
| `messages`, `max_tokens`, sampling |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Same parameters as [Chat Completions](api_openai_chat_completions.md)     |
| `response_format` `json_object`  |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Full support                                                                |
| `tools` / `tool_choice` / `functions` | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Refused when the batch is created — tool use is not available in a batch |
| `response_format` `json_schema`  | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Refused when the batch is created                                           |
| `stream`                         | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | A batch has nothing to stream to                                            |
| `n` above 1                      | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Send one request per completion instead                                     |
| `prompt_cache_key` / `prompt_cache_breakpoint` |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Accepted and ignored — a batch reads and writes no prompt cache, and the request is answered without one |
| `input`, `dimensions`            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Same parameters as [Embeddings](api_openai_embeddings.md), one `input` per request |
| `encoding_format` `base64`       | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Refused when the batch is created — batched vectors come back as numbers  |
| **Lifecycle**                    |                                          |                                                                             |
| Retrieve / poll                  |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | `validating` → `in_progress` → `finalizing` → `completed`                    |
| Cancel                           |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | `cancelling` then `cancelled`; requests already answered stay in `output_file_id`, and a batch that has ended is unchanged |
| List batches                     |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Newest first, with an `after` cursor                                        |
| `output_file_id` / `error_file_id` |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Readable through the [Files API](api_openai_files.md)                     |
| `usage`                          |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Token totals, reported once the batch ends                                  |
| `finalizing` status              |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Reported with `finalizing_at` while the results of a batch whose requests have run are being assembled; `completed` follows once they are readable |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with OpenAI API
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation

</div>

!!! note "Content Guardrails and Batches"
    A request that a [guardrail](operations_configuration.md#aws-bedrock-guardrail-identifier) would apply to is refused rather than run unguarded. Send those requests without batching.

!!! note "Prompt Caching and Batches"
    Batched requests neither read nor write a prompt cache, on any model. A request carrying a cache hint is still accepted and answered — the hint is dropped rather than the request — so a batch reports no cached tokens in `usage.input_tokens_details`. Nothing is lost by leaving the hint in: batched requests are already billed at the batch rate, and the cache discount was never available at that rate.

## Model Support

Any chat model available for batch inference in your configured Amazon Bedrock regions can be used — the same identifiers as [Chat Completions](api_openai_chat_completions.md). To shortlist them, call [`search_models`](api_search_models.md) with `route=openai_chat_completion&batch=true`; each entry also carries a `batch` field.

!!! warning "The shortlist is a hint, not a rule"
    `batch` is advertised on a best-effort basis and never used to reject a request. A model it does not advertise — or says nothing about — may still run a batch, so submit the batch rather than ruling the model out; the answer you get back is the authoritative one.

A model that cannot serve batched requests is refused when the batch is created, naming the model. A model this deployment normally serves through another Amazon Bedrock endpoint is batched under the identifier the batch endpoint knows it by, so it needs nothing from you.

## Pricing

Batched requests are billed at the published batch rate for the model, roughly half the on-demand rate. Usage is recorded once, when the batch ends. See [Cost Management](operations_cost_management.md#batch-inference).

## Related

- [Files API](api_openai_files.md) — upload the requests, download the results
- [Chat Completions API](api_openai_chat_completions.md) — the per-request body
- [Message Batches API](api_anthropic_batches.md) — the Anthropic-shaped equivalent
- [Configuration](operations_configuration.md#aws-bedrock-batch-role-arn) — enabling batches
