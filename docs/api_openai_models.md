---
title: Models API - List Amazon Bedrock Available Models
description: Discover and list available Amazon Bedrock models through OpenAI-compatible API. Browse 100+ models including Claude, Nova, Llama, and more across all configured regions.
keywords: Amazon Bedrock models, available AI models, list models API, Claude models, OpenAI models API, AI model catalog, foundation models AWS, model discovery
---

# Models API

Discover and list available Amazon Bedrock models across all configured regions through an OpenAI-compatible interface.

## Why Choose the Models API?

<div class="grid cards" markdown>

- :material-view-grid: __Complete Catalog__
  <br>Browse all available models across Amazon Bedrock regions. Chat, embeddings, images, and specialized AI services. See the [Models](models.md) page for a browsable table.

- :material-sync: __Always Up-to-Date__
  <br>Dynamic model discovery automatically shows new models as they become available in Amazon Bedrock.

- :material-map-marker-multiple: __Multi-Region Aggregation__
  <br>Combines models from all configured AWS regions in one deduplicated list; use the [Search Models API](api_search_models.md) to see per-region availability.

- :material-aws: __Comprehensive Coverage__
  <br>Includes Bedrock foundation models plus AWS AI services (Polly, Transcribe) in one unified API.

</div>

## Available Endpoints

| Endpoint | Method | What It Does | Powered By | MCP Tool |
|----------|--------|--------------|------------|----------|
| `/v1/models` | `GET` | List all available models | Amazon Bedrock + AWS AI Services | `openai_model_list` |
| `/v1/models/{model_id}` | `GET` | Get details for a specific model | Amazon Bedrock + AWS AI Services | `openai_model_get` |

## OpenAI-Compatible with Amazon Bedrock Power

**Features:**

- **Multi-region aggregation**: Combines models from all configured Amazon Bedrock regions
- **Comprehensive catalog**: Includes Bedrock foundation models plus specialized models (Transcribe, Polly, etc.)

### What's Different from OpenAI?

- **Provider ownership**: `owned_by` field shows the model provider (e.g., `Amazon`, `Anthropic`, `Mistral AI`)
- **Model-specific capabilities**: Modalities and context windows vary by model—consult AWS documentation for specifics

!!! info "Created Date (`created`)"
    The `created` field is a Unix timestamp (integer) representing the time at which the model was released. This value is sourced from the Amazon Bedrock model lifecycle metadata (`startOfLifeTime`). If the release date is not available from Amazon Bedrock, it defaults to `0` (Unix epoch, January 1, 1970).

## Try It Now

**List all available models:**

```bash
curl -X GET "$BASE/v1/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**Get details for a specific model:**

```bash
curl -X GET "$BASE/v1/models/amazon.nova-micro-v1:0" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**Response:**

```json
{
  "id": "amazon.nova-micro-v1:0",
  "object": "model",
  "created": 1733212800,
  "owned_by": "Amazon"
}
```

---

Browse foundation models for chat, embeddings, images, audio, and more.
