---
title: Search Models API - Discover AWS Bedrock Models by Capability
description: Search and filter available AWS Bedrock models by modality, route, MCP tool, region, streaming support, and legacy status. Designed for AI agents that need to discover the right model before calling other endpoints.
keywords: AWS Bedrock model search, model discovery API, filter models by modality, MCP model discovery, available models API, AI agent model selection
---

# Search Models API

Discover available models by capability — filter by modality, route, region, streaming support, or legacy status. This endpoint is purpose-built for AI agents that need to identify the right model ID before invoking another endpoint.

## Quick Start

| Endpoint | Method | MCP Tool |
|----------|--------|----------|
| `/search_models` | GET | `search_models` |

## How It Works

All query parameters are optional. Parameters combine with **AND** logic — only models matching every supplied filter are returned. Results are sorted by model ID. With no filters, all available models are returned.

**Agent workflow:** call `search_models` first to obtain the correct model ID, then pass it to the target endpoint. To compare costs before picking, pass the shortlisted IDs to the [Model Pricing API](api_model_pricing.md).

## Query Parameters

| Parameter           | Type      | Description                                                                                                                                                              |
|---------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `input_modalities`  | `string`  | Filter by input modality: `TEXT`, `IMAGE`, `VIDEO`, `AUDIO`, `SPEECH`                                                                                                    |
| `output_modalities` | `string`  | Filter by output modality: `TEXT`, `IMAGE`, `VIDEO`, `SPEECH`, `EMBEDDING`, `RERANKING`, `MODERATION`                                                                    |
| `route`             | `string`  | Filter to models supporting a route path (e.g. `/v1/chat/completions`) **or** an MCP tool name (e.g. `openai_chat_completion`) — both formats are accepted transparently |
| `region`            | `string`  | Filter to models available in a specific AWS region (e.g. `us-east-1`)                                                                                                   |
| `streaming`         | `boolean` | `true` = streaming-capable models only, `false` = non-streaming only                                                                                                     |
| `legacy`            | `boolean` | `true` = deprecated models only, `false` = active models only                                                                                                            |

!!! note "Modality values are case-insensitive"
    `TEXT`, `text`, and `Text` are all accepted.

## Response Fields

Each item in the returned list is a `ModelDetails` object:

| Field | Description |
|-------|-------------|
| `id` | AWS Bedrock model ID — pass this to other endpoints |
| `name` | Human-readable model name |
| `provider` | Model provider (e.g. `Anthropic`, `Amazon`, `Meta`) |
| `input_modalities` | List of accepted input types |
| `output_modalities` | List of produced output types |
| `supported_routes` | API routes this model can be used with |
| `supported_mcp_tools` | MCP tool names this model supports |
| `regions` | AWS regions where this model is available |
| `response_streaming` | Whether streaming responses are supported |
| `legacy` | `true` = deprecated model; `false` or absent = active |
| `inference_profiles` | Per-region inference profile IDs as a `region → profile ID` mapping (if any) |

## Examples

**All models accepting TEXT input:**

```bash
curl -G "$BASE/search_models" \
  --data-urlencode "input_modalities=TEXT" \
  -H "Authorization: Bearer $API_KEY"
```

**Image-generation capable models — route path or MCP tool name, both work:**

```bash
# Using the API route path
curl -G "$BASE/search_models" \
  --data-urlencode "route=/v1/images/generations" \
  -H "Authorization: Bearer $API_KEY"

# Using the MCP tool name — same result
curl -G "$BASE/search_models" \
  --data-urlencode "route=openai_image_generation" \
  -H "Authorization: Bearer $API_KEY"
```

**Active non-deprecated chat models:**

```bash
curl -G "$BASE/search_models" \
  --data-urlencode "route=openai_chat_completion" \
  --data-urlencode "legacy=false" \
  -H "Authorization: Bearer $API_KEY"
```

**Active streaming models in `us-east-1`:**

```bash
curl -G "$BASE/search_models" \
  --data-urlencode "region=us-east-1" \
  --data-urlencode "streaming=true" \
  --data-urlencode "legacy=false" \
  -H "Authorization: Bearer $API_KEY"
```

## Error Responses

| Status | Cause |
|--------|-------|
| `400` | Unknown modality name, route path, or MCP tool name — no model matches the filter |
| `200` | Empty list — valid filters, but no model satisfies all of them simultaneously |

!!! tip "Empty list vs. 400"
    A `400` is returned only when a filter value is completely unrecognised (e.g. a typo in a modality name). A valid combination of filters that happens to match zero models still returns `200` with an empty list.

## Using `search_models` as an MCP Tool

When MCP is enabled, `search_models` is exposed as an MCP tool under the same name. AI agents should call it **before** any other tool to identify which model ID to use.

The `route` parameter accepts either format — agents can pass the MCP tool name they intend to call next without needing to know the corresponding HTTP path:

```json
{
  "tool": "search_models",
  "arguments": {
    "route": "openai_chat_completion",
    "legacy": false
  }
}
```

This is the preferred tool for model discovery — it returns richer metadata than `openai_model_list` or `anthropic_model_list` and supports capability-based filtering so agents can select the most appropriate model for their task.

!!! tip "Always include `search_models` in your MCP tool set"
    When configuring `MCP_INCLUDE_TOOLS`, always add `search_models` so agents can discover the right model ID dynamically rather than relying on hardcoded values. See [Operations Configuration → MCP](operations_configuration.md#mcp-model-context-protocol) for details.
