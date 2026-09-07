---
title: Search Models API - Discover Amazon Bedrock Models by Capability
description: Search and filter available Amazon Bedrock models by modality, route, MCP tool, region, streaming support, Batch API support, and legacy status. Designed for AI agents that need to discover the right model before calling other endpoints.
keywords: AWS Bedrock model search, model discovery API, filter models by modality, MCP model discovery, available models API, AI agent model selection, batch capable models
---

# Search Models API

Discover the models this deployment serves and what each one can do, from the catalogue the gateway builds out of Amazon Bedrock, Comprehend, Polly and Transcribe. Native route, served without a dialect prefix.

## At a glance

- :material-filter-variant: **Eight filters, combined with AND** — modality in and out, route or MCP tool name, region, streaming, Batch API, wildcard pattern, legacy status.
- :material-identifier: **The ID to send next** — every result carries `id`, its `aliases`, `supported_routes`, `supported_mcp_tools`, `regions` and per-region `inference_profiles`.
- :material-robot: **Built for agents** — published as the `search_models` MCP tool, and `route` accepts the MCP tool name an agent means to call next, so it never has to know the HTTP path.
- :material-target: **Richer than the dialect model lists** — [`openai_model_list`](api_openai_models.md) and [`anthropic_model_list`](api_anthropic_models.md) return names and dates; this returns capabilities.
- :material-swap-horizontal: **Differs from those lists:** deprecated models are excluded unless `legacy=true`, and `batch` is a best-effort hint that never rejects a request — see [Limits and behaviour to know](#limits-and-behaviour-to-know).

```bash
export BASE="https://your-host"  # native routes carry no dialect prefix

curl -G "$BASE/search_models" \
  --data-urlencode "route=openai_chat_completion" \
  -H "Authorization: Bearer $API_KEY"
```

!!! tip "Browsing rather than calling?"
    The [Models](models.md) page publishes the same catalogue as an interactive table — with AWS prices, regional availability and public leaderboard scores — before you have a deployment.

## Endpoints { #quick-start }

| Endpoint | Method | MCP Tool |
|----------|--------|----------|
| `/search_models` | `GET` | `search_models` |

## How It Works

All query parameters are optional. Parameters combine with **AND** logic — only models matching every supplied filter are returned. Results are sorted by model ID, except that a `model` filter sorts its matches **newest first** instead — see the note below. With no filters, every active (non-legacy) model is returned (see the `legacy` note below for deprecated-model lookups).

**Agent workflow:** call `search_models` first to obtain the correct model ID, then pass it to the target endpoint. To compare costs before picking, pass the shortlisted IDs to the [Model Pricing API](api_model_pricing.md).

## Query Parameters

| Parameter           | Type      | Description                                                                                                                                                              |
|---------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `model`             | `string`  | A [wildcard pattern](operations_configuration_models.md#model-wildcard-patterns) (e.g. `claude-sonnet-*`) or an exact model name. Returns the whole match set, newest first — everything a pattern *could* select, not just the one it would |
| `input_modalities`  | `string`  | Repeatable. Filter by input modality: `TEXT`, `IMAGE`, `VIDEO`, `AUDIO`, `SPEECH`                                                                                        |
| `output_modalities` | `string`  | Repeatable. Filter by output modality: `TEXT`, `IMAGE`, `VIDEO`, `SPEECH`, `EMBEDDING`, `RERANKING`, `MODERATION`                                                        |
| `route`             | `string`  | Filter to models supporting a route path (e.g. `/v1/chat/completions`) **or** an MCP tool name (e.g. `openai_chat_completion`) — both formats are accepted transparently |
| `region`            | `string`  | Filter to models available in a specific AWS region (e.g. `us-east-1`)                                                                                                   |
| `streaming`         | `boolean` | `true` = streaming-capable models only, `false` = non-streaming only                                                                                                     |
| `batch`             | `boolean` | `true` = models advertised for the [Batch API](api_openai_batches.md) only, `false` = the rest. Best effort — see [Limits](#limits-and-behaviour-to-know).               |
| `legacy`            | `boolean` | `true` = deprecated models only, `false` = active models only. Deprecated models are excluded when omitted — see [Limits](#limits-and-behaviour-to-know).                |

!!! note "Modality values are case-insensitive"
    `TEXT`, `text`, and `Text` are all accepted.

!!! tip "`model` is how you check a pattern before using it"
    A `model` filter returns every match, sorted newest first by release date, with models of unknown release date last — including the ones a pattern cannot select. This is the recommended way to see what a pattern would resolve to, and whether it would be refused as ambiguous, before sending it on a request. Add `route` to narrow it to one endpoint, exactly as a pattern on a request is scoped. Unlike a pattern on a request, this filter accepts any pattern, `*` included, and — like every other search — it lists non-legacy models unless you pass `legacy=true`.

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

The response is a bare JSON array of these objects — no envelope, no pagination:

```json
[
  {
    "id": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "name": "Claude Sonnet 4.5",
    "provider": "Anthropic",
    "service": "AWS Bedrock Runtime",
    "input_modalities": ["TEXT", "IMAGE"],
    "output_modalities": ["TEXT"],
    "aliases": ["claude-sonnet-4-5", "claude-sonnet-4-5-20250929"],
    "supported_routes": ["/v1/chat/completions", "/v1/responses", "/v1/messages"],
    "supported_mcp_tools": ["openai_chat_completion", "openai_response", "anthropic_message"],
    "regions": ["us-east-1", "us-west-2"],
    "inference_profiles": {"us-east-1": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"},
    "response_streaming": true,
    "batch": true
  }
]
```

Fields with nothing to report are omitted, so a model with no alias carries no `aliases` key.

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
    When configuring `MCP_INCLUDE_TOOLS`, always add `search_models` so agents can discover the right model ID dynamically rather than relying on hardcoded values. See [HTTP Server and MCP → MCP](operations_configuration_server.md#mcp-model-context-protocol) for details.

## Status Codes

| Status | Cause |
|--------|-------|
| `200` | Success — valid filters that match zero models still return `200` with an empty list |
| `400` | Unrecognized filter value: unknown modality name, route path, MCP tool name, or a region where no model is available |

## Limits and behaviour to know

**Deprecated models are excluded unless you ask for them.** `legacy=true` returns deprecated models *only*, not the active ones plus the deprecated ones; run both calls to see the whole catalogue.

**`batch` is advertised on a best-effort basis and never rejects a request.** A model with `batch: false`, or with no `batch` field, may still run a batch: the authoritative answer is what the backend returns when you submit one. Treat it as a shortlist to start from.

**An unrecognised filter value is a `400`, an unmatched combination is a `200`.** A typo in a modality name, an unknown route or MCP tool name, or a `region` where no model is available, are refused; a valid combination that matches nothing returns an empty array.

**`service` is not comparable with the `model_pricing` field of the same name.** This one carries display names (`AWS Bedrock Runtime`); [`model_pricing`](api_model_pricing.md) carries AWS API endpoint identifiers (`bedrock-runtime`).

**A `model` filter changes the sort order.** Results are sorted by model ID, except with a `model` filter, which sorts its matches newest first by release date, models of unknown release date last.

## Request headers

| Header          | Purpose         | Notes                                           |
|-----------------|-----------------|-------------------------------------------------|
| `Authorization` | Gateway API key | `Bearer <key>`, required like every other route |

## Try it { #examples }

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

**Everything a wildcard pattern matches, newest first:**

```bash
curl -G "$BASE/search_models" \
  --data-urlencode "model=claude-sonnet-*" \
  -H "Authorization: Bearer $API_KEY"
```

**Look up a deprecated model (see the `legacy=true` note above):**

```bash
curl -G "$BASE/search_models" \
  --data-urlencode "route=openai_chat_completion" \
  --data-urlencode "legacy=true" \
  -H "Authorization: Bearer $API_KEY"
```

## Next steps

Next: [Model Pricing API](api_model_pricing.md) · [Models catalogue](models.md) · [MCP configuration](operations_configuration_server.md#mcp-model-context-protocol) · [Model aliases and wildcards](operations_configuration_models.md#model-wildcard-patterns)
