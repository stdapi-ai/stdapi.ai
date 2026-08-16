---
title: Message Batches API - Asynchronous Bulk Messages (Anthropic Compatible)
description: Run thousands of Messages API requests asynchronously at a discounted price with an Anthropic-compatible Message Batches API backed by Amazon Bedrock batch inference.
keywords: Anthropic message batches, batch processing Claude, bulk inference AWS, discounted inference, msgbatch, Amazon Bedrock batch inference, asynchronous messages
---

# Message Batches API

Run a large set of Messages API requests asynchronously, at a lower price than the synchronous API, through the Anthropic Message Batches API shape.

The requests are sent inline, the batch runs without a connection held open, and its results are streamed back as JSONL — exactly the Anthropic workflow, so the official Anthropic SDKs work by changing the base URL.

## Why Choose the Message Batches API?

<div class="grid cards" markdown>

- :material-tag-arrow-down: __Lower Price per Token__
  <br>Batched requests are billed at the published batch rate, well below the on-demand rate for the same model.

- :material-swap-horizontal: __Drop-in Anthropic Compatibility__
  <br>`client.messages.batches.create(...)` and `.results(...)` work unchanged.

- :material-set-split: __Several Models, One Batch__
  <br>Each request names its own model, as upstream allows; the batch reports a single aggregate state.

- :material-cloud-lock: __Private AWS Backend__
  <br>Requests and results are stored in your own S3 buckets — no traffic to third-party endpoints.

</div>

## Prerequisites

The Message Batches API is disabled until the deployment declares an AWS IAM service role that Amazon Bedrock assumes to read the requests and write the results:

- [`AWS_BEDROCK_BATCH_ROLE_ARN`](operations_configuration.md#aws-bedrock-batch-role-arn) — the service role.
- [`AWS_S3_BUCKET`](operations_configuration.md#aws-s3-bucket) — the bucket holding the batch data.
- [`AWS_S3_BATCHES_PREFIX`](operations_configuration.md#aws-s3-batches-prefix) — the prefix it is stored under.

The permissions the role and the server need are listed in [IAM Permissions](operations_iam_permissions.md#batch-inference).
While the role is unset, every batch endpoint answers `529`.

## Workflow

### 1. Create the batch

```python
from anthropic import Anthropic

client = Anthropic(base_url="https://your-host/anthropic", api_key="...")

batch = client.messages.batches.create(
    requests=[
        {
            "custom_id": f"req-{index}",
            "params": {
                "model": "amazon.nova-micro-v1:0",
                "max_tokens": 256,
                "messages": [{"role": "user", "content": text}],
            },
        }
        for index, text in enumerate(documents)
    ]
)
```

**Example request (curl):**

```bash
curl -X POST "https://your-host/anthropic/v1/messages/batches" \
  -H "x-api-key: $API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "requests": [
      {
        "custom_id": "req-1",
        "params": {
          "model": "amazon.nova-micro-v1:0",
          "max_tokens": 256,
          "messages": [{"role": "user", "content": "Summarize: ..."}]
        }
      }
    ]
  }'
```

The samples above are abridged: a real batch needs at least 100 requests for each model it names.

**Example response:**

```json
{
  "id": "msgbatch_06fvfg3lbdqarbad8kbo55g0sg5h3s4a",
  "type": "message_batch",
  "processing_status": "in_progress",
  "request_counts": {"processing": 100, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0},
  "created_at": "2026-08-12T20:53:33Z",
  "expires_at": "2026-08-13T20:53:33Z"
}
```

### 2. Poll until processing ends

```python
batch = client.messages.batches.retrieve(batch.id)
print(batch.processing_status, batch.request_counts)
```

`succeeded` and `errored` move as the batch runs, so they can drive a progress bar. `canceled` and `expired` are known only once `processing_status` is `ended`.

### 3. Read the results

```python
for entry in client.messages.batches.results(batch.id):
    if entry.result.type == "succeeded":
        print(entry.custom_id, entry.result.message.content[0].text)
```

Each line pairs a `custom_id` with its outcome:

```json
{"custom_id": "req-1", "result": {"type": "succeeded", "message": {"id": "msg_req-1", "type": "message", "role": "assistant", "content": [{"type": "text", "text": "..."}], "stop_reason": "end_turn", "usage": {"input_tokens": 22, "output_tokens": 9}}}}
{"custom_id": "req-2", "result": {"type": "errored", "error": {"type": "error", "error": {"type": "invalid_request_error", "message": "..."}}}}
```

!!! warning "Results Are Not in Request Order"
    Result lines may come back in any order, as upstream also warns. Match a result to its request with `custom_id`, never with the line number.

## Limits

| Limit                        | Value                    |
|------------------------------|--------------------------|
| Minimum requests per model   | 100 (default quota)      |
| Maximum requests per batch   | 100,000                  |
| Distinct models per batch    | 8                        |
| Processing window            | 24 hours from creation   |

A batch below the minimum, or over the model cap, is refused when it is created and the message names the shortfall — a batch naming several models must reach the minimum **for each of them**.

!!! note "The 100-request minimum is a quota default"
    100 is the default of the Amazon Bedrock quota *Minimum number of records per batch inference job*, which is set **per model** and adjustable for some of them — see [Amazon Bedrock quotas](https://docs.aws.amazon.com/general/latest/gr/bedrock.html). The check applied here is the default, whatever your account's own value is: an account that raised the quota has its smaller batches accepted here and refused on creation, and one that lowered it still cannot submit fewer than 100 requests for a model.

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                          |                  Status                  | Notes                                                                     |
|----------------------------------|:----------------------------------------:|---------------------------------------------------------------------------|
| **Creation**                     |                                          |                                                                           |
| `requests[].custom_id`           |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Up to 64 characters, unique within the batch                              |
| `requests[].params`              |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Same parameters as [Messages](api_anthropic_messages.md)                |
| Several models in one batch      |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Up to 8, each needing the 100-request minimum                             |
| `tools` / `tool_choice`          | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Refused when the batch is created — tool use is not available in a batch |
| Structured output schema         | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Refused when the batch is created                                         |
| `stream`                         | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | A batch has nothing to stream to                                          |
| `cache_control`                  |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Accepted and ignored — a batch reads and writes no prompt cache, and the request is answered without one |
| **Lifecycle**                    |                                          |                                                                           |
| Retrieve / poll                  |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | `in_progress` → `canceling` → `ended`                                     |
| Results (JSONL)                  |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Streamed; available once `processing_status` is `ended`                   |
| Cancel                           |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Requests that already produced a Message keep it, the ones that never ran are reported `canceled`; cancelling twice, or cancelling a batch that has ended, changes nothing |
| Delete                           |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Only once the batch has ended — cancel it first, as upstream requires     |
| List batches                     |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Newest first, with `before_id` / `after_id` cursors                       |
| `archived_at`                    | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Results stay readable until the batch is deleted                          |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with Anthropic API
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation

</div>

!!! note "`results_url` and Reverse Proxies"
    `results_url` is an absolute URL on the address the request came in on, so `client.messages.batches.results(...)` works with no extra configuration and a client fetching it outside the SDK gets a URL it can dial as-is. Behind a reverse proxy it names the proxy's own origin, taken from the `Host` and `X-Forwarded-Proto` headers — enable [`ENABLE_PROXY_HEADERS`](operations_configuration.md#enable-proxy-headers) so the forwarded scheme is trusted, or a TLS-terminating proxy yields an `http://` URL.

!!! note "Content Guardrails and Batches"
    A request that a [guardrail](operations_configuration.md#aws-bedrock-guardrail-identifier) would apply to is refused rather than run unguarded. Send those requests without batching.

!!! note "Prompt Caching and Batches"
    Batched requests neither read nor write a prompt cache, on any model. A request carrying `cache_control` is still accepted and answered — the hint is dropped rather than the request — so a result reports no `cache_read_input_tokens` and no `cache_creation_input_tokens`. Nothing is lost by leaving it in: batched requests are already billed at the batch rate, and the cache discount was never available at that rate.

## Model Support

Any chat model available for batch inference in your configured Amazon Bedrock regions can be used — the same identifiers as [Messages](api_anthropic_messages.md). To shortlist them, call [`search_models`](api_search_models.md) with `route=anthropic_message&batch=true`; each entry also carries a `batch` field.

!!! warning "The shortlist is a hint, not a rule"
    `batch` is advertised on a best-effort basis and never used to reject a request. A model it does not advertise — or says nothing about — may still run a batch, so submit the batch rather than ruling the model out; the answer you get back is the authoritative one.

A model that cannot serve batched requests is refused when the batch is created, naming the model; no sibling job is left running. A model this deployment normally serves through another Amazon Bedrock endpoint is batched under the identifier the batch endpoint knows it by, so it needs nothing from you.

## Pricing

Batched requests are billed at the published batch rate for the model, roughly half the on-demand rate. Usage is recorded once, when the batch ends. See [Cost Management](operations_cost_management.md#batch-inference).

## Related

- [Messages API](api_anthropic_messages.md) — the per-request parameters
- [Batch API](api_openai_batches.md) — the OpenAI-shaped equivalent
- [Configuration](operations_configuration.md#aws-bedrock-batch-role-arn) — enabling batches
