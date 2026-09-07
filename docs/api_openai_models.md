---
title: Models API - List Amazon Bedrock Available Models
description: Discover and list available Amazon Bedrock models through OpenAI-compatible API. Browse 100+ models including Claude, Nova, Llama, and more across all configured regions.
keywords: Amazon Bedrock models, available AI models, list models API, Claude models, OpenAI models API, AI model catalog, foundation models AWS, model discovery
---

# Models API

Discover and list available Amazon Bedrock models across all configured regions through an OpenAI-compatible interface.

## At a glance

- :material-view-grid: **Two endpoints**, `GET /v1/models` and `GET /v1/models/{model_id}`, exposed to AI agents as the `openai_model_list` and `openai_model_get` MCP tools.
- :material-map-marker-multiple: **One deduplicated list across every configured AWS region.** Per-region availability is on the [Search Models API](api_search_models.md); the browsable catalogue is on the [Models](models.md) page.
- :material-aws: **Bedrock foundation models plus the AWS AI services** the gateway fronts, such as Amazon Polly and Amazon Transcribe, in the same list.
- :material-sync: **Discovery is dynamic.** A model that becomes available in your Amazon Bedrock account appears in the listing without a configuration change or a redeploy.
- :material-swap-horizontal: **Differs from OpenAI:** `owned_by` names the model provider (`Amazon`, `Anthropic`, `Mistral AI`), and `created` is the model's release date as published by Amazon Bedrock.

```bash
curl -X GET "$BASE/v1/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

## Endpoints { #available-endpoints }

| Endpoint | Method | What It Does | Powered By | MCP Tool |
|----------|--------|--------------|------------|----------|
| `/v1/models` | `GET` | List all available models | Amazon Bedrock + AWS AI Services | `openai_model_list` |
| `/v1/models/{model_id}` | `GET` | Get details for a specific model | Amazon Bedrock + AWS AI Services | `openai_model_get` |

## Limits and behaviour to know

`created` is a Unix timestamp taken from the Amazon Bedrock model lifecycle metadata (`startOfLifeTime`), the date the model was released. Amazon Bedrock does not publish that date for every model; where it is missing the field is `0` (the Unix epoch, 1 January 1970) rather than absent, so a client that sorts on `created` groups those models together at the start.

Modalities and context windows vary by model and this listing does not carry them: it returns the OpenAI `Model` shape (`id`, `object`, `created`, `owned_by`) and nothing more. Call [`search_models`](api_search_models.md) to filter by modality, route, MCP tool, region or legacy status, or consult the AWS documentation for a specific model.

## Try it { #try-it-now }

**List the model identifiers, one per line:**

```bash
curl -s -X GET "$BASE/v1/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY" | jq -r '.data[].id'
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

## Next steps

Next: [Browse the catalogue](models.md) · [Search Models API](api_search_models.md) · [Chat Completions API](api_openai_chat_completions.md) · [Responses API](api_openai_responses.md)
