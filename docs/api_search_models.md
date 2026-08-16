---
title: Search Models API - Discover Amazon Bedrock Models by Capability
description: Search and filter available Amazon Bedrock models by modality, route, MCP tool, region, streaming support, Batch API support, and legacy status. Designed for AI agents that need to discover the right model before calling other endpoints.
keywords: AWS Bedrock model search, model discovery API, filter models by modality, MCP model discovery, available models API, AI agent model selection, batch capable models
---

# Search Models API

Discover available models by capability — filter by modality, route, region, streaming support, Batch API support, or legacy status. This endpoint is purpose-built for AI agents that need to identify the right model ID before invoking another endpoint.

## Quick Start

| Endpoint | Method | MCP Tool |
|----------|--------|----------|
| `/search_models` | `GET` | `search_models` |

## How It Works

All query parameters are optional. Parameters combine with **AND** logic — only models matching every supplied filter are returned. Results are sorted by model ID. With no filters, every active (non-legacy) model is returned (see the `legacy` note below for deprecated-model lookups).

**Agent workflow:** call `search_models` first to obtain the correct model ID, then pass it to the target endpoint. To compare costs before picking, pass the shortlisted IDs to the [Model Pricing API](api_model_pricing.md).

## Query Parameters

| Parameter           | Type      | Description                                                                                                                                                              |
|---------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `input_modalities`  | `string`  | Repeatable. Filter by input modality: `TEXT`, `IMAGE`, `VIDEO`, `AUDIO`, `SPEECH`                                                                                        |
| `output_modalities` | `string`  | Repeatable. Filter by output modality: `TEXT`, `IMAGE`, `VIDEO`, `SPEECH`, `EMBEDDING`, `RERANKING`, `MODERATION`                                                        |
| `route`             | `string`  | Filter to models supporting a route path (e.g. `/v1/chat/completions`) **or** an MCP tool name (e.g. `openai_chat_completion`) — both formats are accepted transparently |
| `region`            | `string`  | Filter to models available in a specific AWS region (e.g. `us-east-1`)                                                                                                   |
| `streaming`         | `boolean` | `true` = streaming-capable models only, `false` = non-streaming only                                                                                                     |
| `batch`             | `boolean` | `true` = models advertised for the [Batch API](api_openai_batches.md) only, `false` = the rest. Best effort — see the note below.                                        |
| `legacy`            | `boolean` | `true` = deprecated models only, `false` = active models only. Deprecated models are excluded when omitted.                                                             |

!!! note "Modality values are case-insensitive"
    `TEXT`, `text`, and `Text` are all accepted.

!!! warning "Batch support is advertised on a best-effort basis"
    `batch` is a discovery hint, not a guarantee, and it is never used to reject a request: a batch naming a model that is not advertised is still submitted, and only the backend decides. So a model with `batch: false` — or with no `batch` field at all — may well run a batch successfully, and the authoritative answer is what you get back when you submit one. Treat it as a shortlist to start from, not as a list of the only models that work.

!!! note "Legacy models are excluded by default"
    Deprecated models are left out of the results unless you pass `legacy=true`. Pass it if you specifically need to look up a deprecated model, for example to check its replacement — it returns deprecated models only, not the active ones plus the deprecated ones. Combine both calls (with and without `legacy=true`) if you need the full catalogue.

## Response Fields

Each item in the returned list is a `ModelDetails` object:

| Field | Description |
|-------|-------------|
| `id` | Amazon Bedrock model ID — pass this to other endpoints |
| `name` | Human-readable model name |
| `provider` | Model provider (e.g. `Anthropic`, `Amazon`, `Meta`) |
| `service` | AWS service serving the model: `AWS Bedrock Runtime`, `AWS Bedrock Mantle`, `AWS Comprehend`, `AWS Polly`, or `AWS Transcribe` |
| `input_modalities` | List of accepted input types |
| `output_modalities` | List of produced output types |
| `aliases` | Alternate model names accepted by the `model` parameter of the other endpoints (if any) |
| `supported_routes` | API routes this model can be used with |
| `supported_mcp_tools` | MCP tool names this model supports |
| `regions` | AWS regions where this model is available |
| `response_streaming` | Whether streaming responses are supported |
| `batch` | `true` = advertised for the [Batch API](api_openai_batches.md); `false` = not advertised; absent = unknown. Best effort — see the note above |
| `legacy` | `true` = deprecated model; `false` or absent = active |
| `start_of_life_time` | GA date, if known |
| `end_of_life_time` | Deprecation date, if known |
| `legacy_time` | Date the model was marked legacy, if known |
| `public_extended_access_time` | Extended public-access end date, if known |
| `inference_profiles` | Per-region inference profile IDs as a `region → profile ID` mapping (if any) |

## Examples

The `curl` examples below use a `$BASE` variable set to your scheme and host — native routes such as `/search_models` are not prefixed:

```bash
export BASE="https://your-host"
```

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

**Active chat models (legacy models are excluded by default):**

```bash
curl -G "$BASE/search_models" \
  --data-urlencode "route=openai_chat_completion" \
  -H "Authorization: Bearer $API_KEY"
```

**Active streaming models in `us-east-1`:**

```bash
curl -G "$BASE/search_models" \
  --data-urlencode "region=us-east-1" \
  --data-urlencode "streaming=true" \
  -H "Authorization: Bearer $API_KEY"
```

**Chat models advertised for the Batch API:**

```bash
curl -G "$BASE/search_models" \
  --data-urlencode "route=openai_chat_completion" \
  --data-urlencode "batch=true" \
  -H "Authorization: Bearer $API_KEY"
```

**Look up a deprecated model (see the `legacy=true` note above):**

```bash
curl -G "$BASE/search_models" \
  --data-urlencode "route=openai_chat_completion" \
  --data-urlencode "legacy=true" \
  -H "Authorization: Bearer $API_KEY"
```

## Status Codes

| Status | Cause |
|--------|-------|
| `200` | Success — valid filters that match zero models still return `200` with an empty list |
| `400` | Unrecognized filter value: unknown modality name, route path, MCP tool name, or a region where no model is available |

!!! tip "Empty list vs. 400"
    A `400` is returned when a filter value is completely unrecognized (e.g. a typo in a modality name) or when the requested `region` serves no model at all. Otherwise, a combination of filters that happens to match zero models still returns `200` with an empty list.

## Using `search_models` as an MCP Tool

When MCP is enabled, `search_models` is exposed as an MCP tool under the same name. AI agents should call it **before** any other tool to identify which model ID to use.

The `route` parameter accepts either format — agents can pass the MCP tool name they intend to call next without needing to know the corresponding HTTP path:

```json
{
  "tool": "search_models",
  "arguments": {
    "route": "openai_chat_completion"
  }
}
```

This is the preferred tool for model discovery — it returns richer metadata than `openai_model_list` or `anthropic_model_list` and supports capability-based filtering so agents can select the most appropriate model for their task.

!!! tip "Always include `search_models` in your MCP tool set"
    When configuring `MCP_INCLUDE_TOOLS`, always add `search_models` so agents can discover the right model ID dynamically rather than relying on hardcoded values. See [Operations Configuration → MCP](operations_configuration.md#mcp-model-context-protocol) for details.
