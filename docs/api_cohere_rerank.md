---
title: Rerank API - Amazon Bedrock Document Reranking (Cohere Compatible)
description: Rerank documents by semantic relevance with Amazon Bedrock rerank models using a Cohere-compatible API. Improve RAG and search quality with Amazon Rerank and Cohere Rerank 3.5.
keywords: rerank API, document reranking AWS, Cohere rerank, Amazon rerank, semantic reranking, RAG reranking, search relevance, AWS Bedrock rerank, Cohere compatible API
---

# Rerank API (Cohere Compatible)

Rank documents by semantic relevance to a query with Amazon Bedrock rerank models through a Cohere-compatible interface.

!!! warning "Route Prefix & Base URL"
    By default, all Cohere-compatible routes are prefixed with `/cohere`. This means the Rerank API is available at `/cohere/v2/rerank` instead of `/v2/rerank`. You can customize this prefix using the `COHERE_ROUTES_PREFIX` configuration variable documented in [Operations Configuration](operations_configuration.md#cohere-routes-prefix).

    The `curl` examples below use a `$BASE` variable that **must include this prefix** — set it to your scheme and host followed by `COHERE_ROUTES_PREFIX`:

    ```bash
    export BASE="https://your-host/cohere"  # <scheme>://<host> + COHERE_ROUTES_PREFIX
    ```

## Why Choose the Rerank API?

<div class="grid cards" markdown>

- :material-sort: __Better Search Relevance__
  <br>Re-order candidate documents by true semantic relevance to the query. A precise second stage after vector or keyword search.

- :material-book-open-page-variant: __Higher RAG Quality__
  <br>Feed your LLM only the most relevant passages. Reranking reduces context noise and improves answer accuracy.

- :material-swap-horizontal: __Drop-in Cohere Compatibility__
  <br>Follows the Cohere v2 Rerank API shape. Existing Cohere rerank integrations work by changing the base URL.

- :material-cloud-lock: __Private AWS Backend__
  <br>Served entirely by Bedrock rerank models in your own AWS account — no traffic to third-party endpoints.

</div>

## Available Endpoints

| Endpoint     | Method | What It Does                                        | Powered By            | MCP Tool           |
|--------------|--------|-----------------------------------------------------|-----------------------|--------------------|
| `/v2/rerank` | `POST` | Rank documents by semantic relevance to a query     | Bedrock rerank models | `cohere_rerank`    |
| `/v1/rerank` | `POST` | Legacy v1 rerank for older SDKs and integrations    | Bedrock rerank models | `cohere_rerank_v1` |

**Example request:**

```bash
curl -X POST "$BASE/v2/rerank" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cohere.rerank-v3-5:0",
    "query": "What is the capital of the United States?",
    "documents": [
      "Carson City is the capital city of Nevada.",
      "Washington, D.C. is the capital of the United States.",
      "Capital punishment has existed in the United States since colonial times."
    ],
    "top_n": 2
  }'
```

**Example response:**

```json
{
  "id": "0f1b3c6e8d9a4b5c8e7f6a5b4c3d2e1f",
  "results": [
    {"index": 1, "relevance_score": 0.9871},
    {"index": 2, "relevance_score": 0.3251}
  ],
  "meta": {
    "api_version": {"version": "2"},
    "billed_units": {"search_units": 1}
  }
}
```

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                     |                  Status                  | Notes                                                             |
|-----------------------------|:----------------------------------------:|-------------------------------------------------------------------|
| **Input**                   |                                          |                                                                   |
| `query` + `documents` (strings) | :material-check-circle:{ .success role="img" aria-label="Supported" } | Full support                                                      |
| `top_n`                     |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Limits the number of returned results                             |
| `max_tokens_per_doc`        |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Forwarded to the model; support depends on the model              |
| `priority`                  | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted but ignored — request scheduling priority is not applicable on Bedrock |
| `return_documents`          | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted but ignored — v2 results reference input documents by `index` |
| Extra model-specific params | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Extra fields are forwarded as additional model request parameters |
| **Output**                  |                                          |                                                                   |
| `results` (index + score)   |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Ordered by decreasing relevance                                   |
| `meta.billed_units`         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | One search unit per started batch of 100 documents                |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with the Cohere API
* :material-cog:{ .model-dep role="img" aria-label="Model-dependent" } **Available on Select Models** — Check your model's capabilities
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } **Extra Feature** — Enhanced capability beyond the Cohere API

</div>

## Model Support

Any rerank model available in your configured Bedrock regions can be used, for example:

### ![Cohere](styles/logo_cohere.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Cohere Models

| Model             | Model ID               | Cohere Name   | Notes                                    |
|-------------------|------------------------|---------------|-------------------------------------------|
| Cohere Rerank 3.5 | `cohere.rerank-v3-5:0` | `rerank-v3.5` | Multilingual, state-of-the-art relevance |

!!! tip "Cohere's Own Model Names Resolve As They Stand"
    Each Cohere model is published under the name [Cohere's API](https://docs.cohere.com/docs/models) uses as well as its Bedrock ID, derived from the ID rather than curated by hand, so an application already calling Cohere changes only its base URL. Both forms reach the same model. The Amazon rerank model has no such name, having no upstream Cohere API to stay compatible with.

### ![Amazon](styles/logo_amazon.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Models

| Model             | Model ID             | Notes                                                        |
|-------------------|----------------------|----------------------------------------------------------------|
| Amazon Rerank 1.0 | `amazon.rerank-v1:0` | Not available in every region (e.g. absent from `us-east-1`) |

**Find compatible models:** Call [`/search_models`](api_search_models.md) with `route=cohere_rerank` to discover model IDs that support reranking in your deployment.

## Cohere v1 Rerank API (Legacy)

The legacy `/v1/rerank` endpoint is also available for older Cohere SDKs (`cohere.Client`) and third-party integrations that predate the v2 API. It shares the same Bedrock backend and model support as `/v2/rerank`; new clients should prefer the v2 endpoint.

**Differences from the v2 endpoint:**

<div class="feature-table" markdown>

| Feature                     |                  Status                  | Notes                                                             |
|-----------------------------|:----------------------------------------:|-------------------------------------------------------------------|
| `documents` as objects      |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Each document is a string, or a field->value object                |
| `return_documents`          |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | When `true`, each result echoes back the document text            |
| `rank_fields`               |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Ranks object documents on the selected fields only                |
| `max_chunks_per_doc`        | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Rejected with 400 — no Bedrock equivalent                         |
| `meta.api_version.version`  |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Reported as `"1"`                                                 |

</div>

Object documents with a single `text` field use the same plain-text encoding as a string document. Multi-field objects (or object documents combined with a non-default `rank_fields`) are sent to Bedrock as structured JSON documents, natively reranked on their fields. When `rank_fields` is set, object documents are first reduced to the listed fields (missing fields are skipped) before being sent; `return_documents` always echoes back the *original*, unreduced document.

!!! note "Echoed text for multi-field documents"
    The Cohere v1 response type only carries a single `text` string per echoed document. For multi-field object documents, this implementation joins the original fields as `key: value` lines (one per field) — an approximation, since Cohere's own algorithm for this case is not publicly documented.

**Example request:**

```bash
curl -X POST "$BASE/v1/rerank" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cohere.rerank-v3-5:0",
    "query": "What is the capital of the United States?",
    "documents": [
      {"title": "Nevada", "text": "Carson City is the capital city of Nevada."},
      {"title": "United States", "text": "Washington, D.C. is the capital of the United States."}
    ],
    "rank_fields": ["title", "text"],
    "top_n": 1,
    "return_documents": true
  }'
```

**Example response:**

```json
{
  "id": "0f1b3c6e8d9a4b5c8e7f6a5b4c3d2e1f",
  "results": [
    {
      "document": {"text": "title: United States\ntext: Washington, D.C. is the capital of the United States."},
      "index": 1,
      "relevance_score": 0.9871
    }
  ],
  "meta": {
    "api_version": {"version": "1"},
    "billed_units": {"search_units": 1}
  }
}
```

## How It Works

Requests are served by the [Amazon Bedrock Rerank API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_agent-runtime_Rerank.html), with automatic multi-region routing and failover across the regions where the selected model is available.

!!! note "Required IAM Permission"
    The Rerank API requires the `bedrock:Rerank` IAM action in addition to `bedrock:InvokeModel`. See [IAM Permissions](operations_configuration.md#iam-permissions).

## Billing

AWS bills reranking per **search unit**: one search unit covers a single query with up to 100 documents. A request with more than 100 documents is billed one additional search unit per started batch of 100. Search units appear in [usage logs and cost tracking](operations_logging_monitoring.md) as `search_units`.
