---
title: Models API - List AWS Bedrock Models (Anthropic Compatible)
description: Discover and list available AWS Bedrock models through Anthropic-compatible API.
keywords: AWS Bedrock models, Claude models, list models API, Anthropic models API, AI model catalog, text models, model discovery, Nova, Llama
---

# Models API

Discover and list available AWS Bedrock chat models through an Anthropic-compatible interface.

!!! warning "Route Prefix & Base URL"
    By default, all Anthropic-compatible routes are prefixed with `/anthropic`. This means the Models API is available at `/anthropic/v1/models` instead of `/v1/models`. You can customize this prefix using the `ANTHROPIC_ROUTES_PREFIX` configuration variable documented in [Operations Configuration](operations_configuration.md#anthropic-routes-prefix).

    The `curl` examples below use a `$BASE` variable that **must include this prefix** — set it to your scheme and host followed by `ANTHROPIC_ROUTES_PREFIX`:

    ```bash
    export BASE="https://your-host/anthropic"  # <scheme>://<host> + ANTHROPIC_ROUTES_PREFIX
    ```

    **Note:** If `ANTHROPIC_ROUTES_PREFIX` is set to the same value as `OPENAI_ROUTES_PREFIX`, this Anthropic-compatible Models API route will be disabled to avoid conflicts with the OpenAI-compatible Models API.


## Why Use the Models API?

<div class="grid cards" markdown>

- :material-view-grid: __Complete Catalog__
  <br>Browse all available AWS Bedrock models across regions.

- :material-sync: __Always Up-to-Date__
  <br>Dynamic model discovery automatically shows new models as they become available in AWS Bedrock.

- :material-map-marker-multiple: __Multi-Region Aggregation__
  <br>Combines models from all configured AWS regions in one list. See which models are available in each region.

- :material-aws: __Foundation Models__
  <br>Includes Claude, Nova, Llama, and other AWS Bedrock foundation models.

</div>

## Quick Start: Available Endpoints

| Endpoint                | Method | What It Does                     | Powered By  | MCP Tool            |
|-------------------------|--------|----------------------------------|-------------|---------------------|
| `/v1/models`            | GET    | List all available models        | AWS Bedrock | `anthropic_model_list` |
| `/v1/models/{model_id}` | GET    | Get details for a specific model | AWS Bedrock | `anthropic_model_get`  |

## Anthropic-Compatible with AWS Bedrock Power

**Features:**

- **Multi-region aggregation**: Combines models from all configured AWS Bedrock regions
- **Cursor-based pagination**: Use `limit`, `after_id`, and `before_id` query parameters
- **Text models only**: Returns only models with text input and text output modalities (Claude, Nova, Llama, etc.)

### What's Different from Anthropic?

- **Model IDs**: Uses AWS Bedrock model identifiers (e.g., `anthropic.claude-haiku-4-5-20251001-v1:0`) instead of Anthropic model names
- **Extended catalog**: Includes all AWS Bedrock models (Claude, Nova, Llama, etc.), not just Anthropic models

!!! info "Created Date (`created_at`)"
    The `created_at` field is an RFC 3339 datetime string representing the time at which the model was released. This value is sourced from the AWS Bedrock model lifecycle metadata (`startOfLifeTime`). If the release date is not available from AWS Bedrock, it defaults to the Unix epoch (`"1970-01-01T00:00:00Z"`).

## Try It Now

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

---

Browse AWS Bedrock foundation models for chat and completion tasks.
