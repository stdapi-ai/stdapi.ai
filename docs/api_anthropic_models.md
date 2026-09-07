---
title: Models API - List Amazon Bedrock Models (Anthropic Compatible)
description: Discover and list available Amazon Bedrock text models through an Anthropic-compatible API.
keywords: AWS Bedrock models, Claude models, list models API, Anthropic models API, AI model catalog, text models, model discovery, Nova, Llama
---

# Models API (Anthropic Compatible)

Discover and list available Amazon Bedrock text models through an Anthropic-compatible interface. Served under `/anthropic` by default; the examples below use `$BASE`, which includes that prefix.

## At a glance

- :material-view-grid: **Text model catalog** — Browse the available Bedrock text models across regions. See the [Models](models.md) page for a browsable table.
- :material-sync: **Always up to date** — Dynamic model discovery automatically shows new models as they become available in Bedrock.
- :material-map-marker-multiple: **Multi-region aggregation** — Combines models from all configured AWS regions in one list. See which models are available in each region.
- :material-aws: **Foundation models** — Includes Claude, Nova, Llama, and other Bedrock foundation text models.
- :material-swap-horizontal: **Differs from the Anthropic API:** model IDs are Bedrock identifiers, the catalogue is not limited to Claude, and `limit` defaults to `1000` instead of `20` — see [Limits and behaviour to know](#limits-and-behaviour-to-know).
- :material-filter-variant: **Only text-in, text-out models are listed.** For embedding, image, audio or video models, and for filtering by route, region or MCP tool, use [`GET /search_models`](api_search_models.md).

!!! info "Base URL and route prefix"
    By default, all Anthropic-compatible routes are prefixed with `/anthropic`. This means the Models API is available at `/anthropic/v1/models` instead of `/v1/models`. You can customize this prefix using the `ANTHROPIC_ROUTES_PREFIX` configuration variable documented in [HTTP Server and MCP](operations_configuration_server.md#anthropic-routes-prefix).

    The `curl` examples on this page use a `$BASE` variable that **must include this prefix** — set it to your scheme and host followed by `ANTHROPIC_ROUTES_PREFIX`:

    ```bash
    export BASE="https://your-host/anthropic"  # <scheme>://<host> + ANTHROPIC_ROUTES_PREFIX
    ```

    **Note:** `ANTHROPIC_ROUTES_PREFIX` must always be a non-empty path and is validated at startup to differ from `OPENAI_ROUTES_PREFIX` (the server refuses to start otherwise). This Anthropic-compatible Models API is therefore always served on its own path, distinct from the [OpenAI-compatible Models API](api_openai_models.md).

```bash
curl "$BASE/v1/models" -H "x-api-key: $ANTHROPIC_API_KEY"
```

## Endpoints { #available-endpoints }

| Endpoint                | Method | What It Does                     | Powered By     | MCP Tool               |
|-------------------------|--------|----------------------------------|----------------|------------------------|
| `/v1/models`            | `GET`  | List all available text models   | Amazon Bedrock | `anthropic_model_list` |
| `/v1/models/{model_id}` | `GET`  | Get details for a specific model | Amazon Bedrock | `anthropic_model_get`  |

## Working with the catalogue { #anthropic-compatible-with-amazon-bedrock-power }

**Features:**

- **Multi-region aggregation**: Combines models from all configured Bedrock regions
- **Cursor-based pagination**: Use `limit`, `after_id`, and `before_id` query parameters
- **Text models only**: Returns only models with text input and text output modalities (Claude, Nova, Llama, etc.)

### Response fields

`GET /v1/models` returns a paginated envelope; `GET /v1/models/{model_id}` returns one `model` object with the same four fields.

| Field          | Type             | Description                                                                    |
|----------------|------------------|--------------------------------------------------------------------------------|
| `data[]`       | array            | The page of models, sorted by model ID                                          |
| `id`           | string           | Bedrock model identifier, the value to send as `model`                          |
| `type`         | string           | Always `model`                                                                  |
| `display_name` | string           | Human-readable model name                                                       |
| `created_at`   | string           | RFC 3339 release date — see the note below                                      |
| `has_more`     | boolean          | Whether another page follows the cursor                                         |
| `first_id`     | string \| null   | ID of the first model on this page, to use as `before_id`                       |
| `last_id`      | string \| null   | ID of the last model on this page, to use as `after_id`                         |

```json
{
  "data": [
    {
      "id": "amazon.nova-micro-v1:0",
      "type": "model",
      "display_name": "Amazon Nova Micro",
      "created_at": "2025-01-01T00:00:00Z"
    }
  ],
  "has_more": false,
  "first_id": "amazon.nova-micro-v1:0",
  "last_id": "amazon.nova-micro-v1:0"
}
```

!!! info "Created Date (`created_at`)"
    The `created_at` field is an RFC 3339 datetime string representing the time at which the model was released. This value is sourced from the Bedrock model lifecycle metadata (`startOfLifeTime`). If the release date is not available from Bedrock, it defaults to the Unix epoch (`"1970-01-01T00:00:00Z"`).

## Limits and behaviour to know

### What's Different from Anthropic?

- **Model IDs**: Uses Bedrock model identifiers (e.g., `anthropic.claude-haiku-4-5-20251001-v1:0`) instead of Anthropic model names
- **Extended catalog**: Includes all Bedrock text models (Claude, Nova, Llama, etc.), not just Anthropic models
- **Default page size**: `limit` defaults to `1000` (the Anthropic API defaults to `20`)

**Only text-in, text-out models appear.** A model is listed when its Bedrock metadata declares both `TEXT` input and `TEXT` output, so embedding, image, speech and video models are absent from this catalogue even though the deployment serves them. [`GET /search_models`](api_search_models.md) lists every model with its modalities and routes.

**The listing carries no capability metadata.** `id`, `type`, `display_name` and `created_at` are the whole object: there is no context window, no pricing, no region and no tool support. Those come from [`GET /search_models`](api_search_models.md) and [`GET /model_pricing`](api_model_pricing.md).

## Request headers

| Header              | Purpose                   | Notes                                          |
|---------------------|---------------------------|------------------------------------------------|
| `x-api-key`         | Gateway API key           | Required, like every other route               |
| `anthropic-version` | Anthropic API version pin | Accepted and ignored; sent by the official SDKs |

## Try it { #try-it-now }

**List all available models:**

```bash
curl -X GET "$BASE/v1/models" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01"
```

**List models with pagination:**

```bash
curl -X GET "$BASE/v1/models?limit=10" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01"
```

The `limit` query parameter accepts values from `1` to `1000` and defaults to `1000`.

**Get details for a specific model:**

```bash
curl -X GET "$BASE/v1/models/amazon.nova-micro-v1:0" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01"
```

## Next steps

Next: [Messages API](api_anthropic_messages.md) · [Search Models API](api_search_models.md) · [Model Pricing API](api_model_pricing.md) · [Models catalogue](models.md)
