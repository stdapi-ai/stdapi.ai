---
title: Embed API - Amazon Bedrock Vector Embeddings (Cohere Compatible)
description: Generate vector embeddings with Amazon Bedrock embedding models using a Cohere-compatible API. Semantic search, RAG, and multimodal embeddings through the Cohere v2 Embed interface.
keywords: Cohere embed API, vector embeddings AWS, semantic search API, RAG embeddings, Cohere compatible API, AWS Bedrock embeddings, input_type embeddings
---

# Embed API (Cohere Compatible)

Generate vector embeddings for semantic search and RAG applications with Amazon Bedrock embedding models through a Cohere-compatible interface.

This is an alternate route to the [OpenAI-compatible Embeddings API](api_openai_embeddings.md): both are served by the same embedding backends and models, so anything supported there is supported here.

!!! warning "Route Prefix & Base URL"
    By default, all Cohere-compatible routes are prefixed with `/cohere`. This means the Embed API is available at `/cohere/v2/embed` instead of `/v2/embed`. You can customize this prefix using the `COHERE_ROUTES_PREFIX` configuration variable documented in [Operations Configuration](operations_configuration.md#cohere-routes-prefix).

    The `curl` examples below use a `$BASE` variable that **must include this prefix** — set it to your scheme and host followed by `COHERE_ROUTES_PREFIX`:

    ```bash
    export BASE="https://your-host/cohere"  # <scheme>://<host> + COHERE_ROUTES_PREFIX
    ```

## Why Choose Embed?

<div class="grid cards" markdown>

- :material-magnify: __Semantic Search__
  <br>Turn texts and images into dense vectors for similarity search that understands meaning, not just keywords.

- :material-book-open-page-variant: __Higher RAG Quality__
  <br>Build retrieval pipelines on high-quality embeddings, with `input_type` tuning for queries versus documents.

- :material-swap-horizontal: __Drop-in Cohere Compatibility__
  <br>Follows the Cohere v2 Embed API shape. Existing Cohere embed integrations work by changing the base URL.

- :material-cloud-lock: __Private AWS Backend__
  <br>Served entirely by Bedrock embedding models in your own AWS account — no traffic to third-party endpoints.

</div>

## Quick Start: Available Endpoints

| Endpoint    | Method | What It Does                                           | Powered By               | MCP Tool          |
|-------------|--------|--------------------------------------------------------|--------------------------|-------------------|
| `/v2/embed` | `POST` | Transform texts and images into semantic float vectors | Bedrock embedding models | `cohere_embed`    |
| `/v1/embed` | `POST` | Legacy v1 embed for older SDKs and integrations        | Bedrock embedding models | `cohere_embed_v1` |

**Example request:**

```bash
curl -X POST "$BASE/v2/embed" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cohere.embed-multilingual-v3",
    "input_type": "search_document",
    "texts": ["Hello world", "Bonjour le monde"]
  }'
```

**Example response:**

```json
{
  "response_type": "embeddings_by_type",
  "id": "0f1b3c6e8d9a4b5c8e7f6a5b4c3d2e1f",
  "embeddings": {"float": [[0.012, -0.034, ...], [0.041, 0.007, ...]]},
  "texts": ["Hello world", "Bonjour le monde"],
  "meta": {
    "api_version": {"version": "2"},
    "billed_units": {"input_tokens": 8}
  }
}
```

**Find compatible models:** Call [`/search_models`](api_search_models.md) with `mcp_tool=cohere_embed` to discover model IDs that support embeddings — every Bedrock embedding model works, not just Cohere ones.

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                       |                  Status                  | Notes                                                              |
|-------------------------------|:----------------------------------------:|--------------------------------------------------------------------|
| **Input**                     |                                          |                                                                    |
| `texts`                       |   :material-check-circle:{ .success }    | Full support                                                       |
| `images`                      |       :material-cog:{ .model-dep }       | Multimodal models only; data URIs, plus URLs and S3 URIs           |
| `inputs` (fused text + image) | :material-close-circle:{ .unsupported }  | Rejected with 400 — use `texts` or `images` instead                |
| **Model Parameters**          |                                          |                                                                    |
| `input_type`                  |       :material-cog:{ .model-dep }       | Applied to Cohere models; no equivalent on other providers         |
| `output_dimension`            |       :material-cog:{ .model-dep }       | Some models support dimension reduction                            |
| `truncate`, `max_tokens`      |       :material-cog:{ .model-dep }       | Cohere models only                                                 |
| `embedding_types`             |   :material-minus-circle:{ .partial }    | Only `["float"]` is accepted; other types return 400               |
| `priority`                    | :material-close-circle:{ .unsupported }  | Accepted but ignored — request scheduling priority is not applicable on Bedrock |
| Extra model-specific params   | :material-plus-circle:{ .extra-feature } | Extra fields are forwarded as additional model request parameters  |
| **Output**                    |                                          |                                                                    |
| `images` metadata array       | :material-close-circle:{ .unsupported }  | Image dimensions are not echoed in the response                    |
| **Usage tracking**            |                                          |                                                                    |
| `billed_units.input_tokens`   |       :material-cog:{ .model-dep }       | Estimated on some models                                           |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success } **Supported** — Fully compatible with the Cohere API
* :material-cog:{ .model-dep } **Available on Select Models** — Check your model's capabilities
* :material-minus-circle:{ .partial } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature } **Extra Feature** — Enhanced capability beyond the Cohere API

</div>

## Cohere v1 Embed API (Legacy)

The legacy `/v1/embed` endpoint is also available for older Cohere SDKs (`cohere.Client`) and third-party integrations that predate the v2 API. It shares the same Bedrock backend and model support as `/v2/embed`; new clients should prefer the v2 endpoint.

**Differences from the v2 endpoint:**

<div class="feature-table" markdown>

| Feature                    |                 Status                  | Notes                                                                          |
|----------------------------|:---------------------------------------:|---------------------------------------------------------------------------------|
| Default response shape     |   :material-check-circle:{ .success }   | Legacy `embeddings_floats`: a plain list of float vectors                      |
| `embedding_types`          |   :material-check-circle:{ .success }   | `["float"]` switches to the `embeddings_by_type` shape; other types return 400 |
| `input_type`               |   :material-check-circle:{ .success }   | Optional — forwarded to Cohere models when provided; the backend defaults to `search_document` otherwise |
| `meta.api_version.version` |   :material-check-circle:{ .success }   | Reported as `"1"`                                                              |

</div>

**Example request:**

```bash
curl -X POST "$BASE/v1/embed" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cohere.embed-multilingual-v3",
    "input_type": "search_document",
    "texts": ["Hello world", "Bonjour le monde"]
  }'
```

**Example response:**

```json
{
  "response_type": "embeddings_floats",
  "id": "0f1b3c6e8d9a4b5c8e7f6a5b4c3d2e1f",
  "embeddings": [[0.012, -0.034, ...], [0.041, 0.007, ...]],
  "texts": ["Hello world", "Bonjour le monde"],
  "meta": {
    "api_version": {"version": "1"},
    "billed_units": {"input_tokens": 8}
  }
}
```

## How It Works

Requests are served by the same Bedrock embedding backends as the [OpenAI-compatible Embeddings API](api_openai_embeddings.md), with automatic multi-region routing and failover across the regions where the selected model is available.

- When both `texts` and `images` are provided, embeddings are returned in request order: all texts first, then all images.
- Guardrail and performance headers available on the [OpenAI-compatible Embeddings API](api_openai_embeddings.md#available-request-headers) work on this route too.

## Billing

Requests are billed through Bedrock (per token), not in Cohere search units; `billed_units.input_tokens` reports the Bedrock-metered input tokens. Usage appears in [usage logs and cost tracking](operations_logging_monitoring.md) as `input_tokens`.
