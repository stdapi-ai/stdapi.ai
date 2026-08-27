---
title: Ollama Embed API - Amazon Bedrock Vector Embeddings via the Ollama Interface
description: Generate vector embeddings with Amazon Bedrock embedding models using the Ollama-compatible /api/embed and /api/embeddings endpoints.
keywords: Ollama embed API, Ollama compatible API, Amazon Bedrock embeddings, Ollama /api/embed, Ollama /api/embeddings, vector embeddings AWS
---

# Embed API (Ollama Compatible)

Generate vector embeddings for semantic search and RAG applications with Amazon Bedrock embedding models through the Ollama `/api/embed` interface.

This is an alternate route to the [OpenAI-compatible Embeddings API](api_openai_embeddings.md): both are served by the same embedding backends and models.

!!! warning "Route Prefix & Base URL"
    Ollama routes carry **no prefix by default**, so the Embed API sits exactly where an Ollama client expects it from a bare base URL: `/api/embed`. You can add a prefix with the `OLLAMA_ROUTES_PREFIX` configuration variable documented in [Operations Configuration](operations_configuration.md#ollama-routes-prefix).

    The `curl` examples below use a `$BASE` variable — set it to your scheme and host:

    ```bash
    export BASE="https://your-host"  # <scheme>://<host> + OLLAMA_ROUTES_PREFIX, if configured
    ```

## Why Choose the Ollama Embed API?

<div class="grid cards" markdown>

- :material-swap-horizontal: __Drop-in Ollama Compatibility__
  <br>Follows the Ollama `/api/embed` request and response shape, so an existing Ollama embedding client works by changing the base URL.

- :material-magnify: __Semantic Search__
  <br>Turn one or several texts into dense vectors for similarity search that understands meaning, not just keywords.

- :material-book-open-page-variant: __Higher RAG Quality__
  <br>Build retrieval pipelines on the same high-quality embedding models served by the [OpenAI-compatible Embeddings API](api_openai_embeddings.md).

- :material-cloud-lock: __Private AWS Backend__
  <br>Served entirely by Amazon Bedrock embedding models in your own AWS account — no traffic to third-party endpoints.

</div>

## Available Endpoints

| Endpoint          | Method | What It Does                                        | Powered By                       | MCP Tool     |
|--------------------|--------|-------------------------------------------------------|------------------------------------|--------------|
| `/api/embed`       | `POST` | Embed one or several inputs, in request order         | Amazon Bedrock embedding models    | Not exposed  |
| `/api/embeddings`  | `POST` | Legacy single-prompt embed, deprecated upstream        | Amazon Bedrock embedding models    | Not exposed  |

!!! note "Not Exposed as an MCP Tool"
    `ollama_embed` and `ollama_embeddings` duplicate a tool an agent already has on the OpenAI surface, and a redundant tool degrades an agent's tool choice, so both are excluded from the MCP tool set by default. An operator who wants one back names it in [`MCP_INCLUDE_TOOLS`](operations_configuration.md#mcp-include-tools).

**Example request:**

```bash
curl -X POST "$BASE/api/embed" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.titan-embed-text-v2:0",
    "input": ["first", "second"]
  }'
```

**Example response:**

```json
{
  "model": "amazon.titan-embed-text-v2:0",
  "embeddings": [[0.012, -0.034, ...], [0.041, 0.007, ...]],
  "total_duration": 214567890,
  "prompt_eval_count": 4
}
```

## Model Names

Send the model names [`GET /api/tags`](api_ollama_models.md#get-apitags) publishes — those are the canonical identifiers this server resolves directly. A trailing `:latest` is accepted and stripped as a fallback when the exact name is not found. Short aliases accepted on this server's other APIs also work here even though `/api/tags` does not list them. A name learned from ollama.com is not available through this server and answers `404`. `/api/embed` echoes the model name exactly as the request spelled it, in `model`; the legacy `/api/embeddings` returns the vector alone, as it does upstream.

**Find compatible models:** Call [`/search_models`](api_search_models.md) with `route=ollama_embed` to discover model IDs that support embeddings.

## Feature Compatibility

<div class="feature-table" markdown>

| Feature               |                  Status                  | Notes                                                              |
|------------------------|:----------------------------------------:|--------------------------------------------------------------------|
| **Input**              |                                          |                                                                    |
| `input` (string)       |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Full support                                                       |
| `input` (array)        |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | One vector per entry, returned in request order                    |
| `dimensions`           |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Sets the vector width on models that support dimension reduction   |
| `truncate`             | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted and ignored                                                |
| `keep_alive`           | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted and ignored — models are never resident                   |
| `options`              | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted and ignored — no runner options apply to embeddings        |
| **Output**             |                                          |                                                                    |
| `embeddings`           |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | One vector per input, in request order                             |
| `total_duration`       |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Real wall-clock time                                                |
| `prompt_eval_count`    |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Real input token count                                              |
| `load_duration`        | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Never reported — there is no model-loading phase to measure         |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with the Ollama API
* :material-cog:{ .model-dep role="img" aria-label="Model-dependent" } **Available on Select Models** — Check your model's capabilities
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation

</div>

## Ollama Legacy Embeddings API (`/api/embeddings`)

`/api/embeddings` is the singular form Ollama itself has deprecated in favor of `/api/embed`. It takes a single `prompt` instead of `input`, embeds it against the same models, and returns only the vector — no token count, no duration:

```bash
curl -X POST "$BASE/api/embeddings" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.titan-embed-text-v2:0",
    "prompt": "first"
  }'
```

```json
{"embedding": [0.012, -0.034, ...]}
```

!!! warning "Deprecated"
    Prefer [`/api/embed`](#available-endpoints): it embeds several inputs in one call and reports token usage, which `/api/embeddings` cannot.

## How It Works

Requests are served by the same embedding backends and models as the [OpenAI-compatible Embeddings API](api_openai_embeddings.md) — anything supported there through `model` is reachable here through the same identifier.

## Limitations

- `truncate`, `keep_alive` and `options` are accepted and ignored.
- `load_duration` is never reported: there is no model-loading phase to measure, and a number there would be invented.
- `/api/embeddings` never reports `prompt_eval_count` or `total_duration` — it only ever returns the vector, matching Ollama's own legacy response shape.
