---
title: Model Pricing API - Exact AWS Unit Prices per Model
description: Query the exact AWS unit prices for any model — token rates, service tiers, prompt-cache TTLs, cross-region routing, image specs, and long-context premiums — straight from the AWS Price List catalog used for cost tracking.
keywords: AWS Bedrock pricing API, model price comparison, token price API, AI agent cost-aware model selection, AWS Price List, cost tracking
---

# Model Pricing API

Query the exact AWS unit prices for one or more models, straight from the same AWS Price List catalog the server uses for [request cost tracking](operations_cost_management.md#cost-tracking-real-time-aws-pricing). Purpose-built for cost-aware model selection: shortlist models with [`search_models`](api_search_models.md), then compare their price cards in one call.

## Quick Start

| Endpoint | Method | MCP Tool |
|----------|--------|----------|
| `/model_pricing` | `GET` | `model_pricing` |

## How It Works

Each returned row is one **published AWS price**: a billed dimension (same vocabulary as the request-log usage entries) plus its published variants — service tier, prompt-cache write TTL, serving profile, media spec, and long-context bucket. `unit_price` is an exact decimal string per **one** billed unit (token, image, second, character, request, search unit).

By default the card reflects **your server's configuration**: only the configured Bedrock regions, each model's default service tier, and its effective serving profile (routing) are returned, with the same fallbacks billing applies (a tier or routing without a distinct published rate falls back to the standard/plain row). Pass `all_prices=true` for the full published price table, and explicit `region`/`tier`/`routing` filters to override any axis in either mode.

Prices are indexed **eagerly**: models not currently accessible to your account can still be priced. An empty `prices` list means AWS publishes no rows for that model (or none match the filters). A missing variant row means AWS publishes no distinct rate for it — billing then falls back the same way cost tracking does.

This endpoint responds without contacting AWS, so it is fast enough to call inline. It requires cost tracking to be enabled (`COST_TRACKING=true`), and returns a retry-later `503` while the price catalog is still loading after startup.

## Query Parameters

All parameters are optional and combine with **AND** logic.

| Parameter   | Type      | Description                                                                                             |
|-------------|-----------|---------------------------------------------------------------------------------------------------------|
| `model`     | `string`  | Repeatable. Model IDs or aliases to price; omit for the full list of available models                    |
| `region`    | `string`  | Only prices for this AWS region (e.g. `us-east-1`)                                                       |
| `tier`      | `string`  | Only this service tier: `standard`, `flex`, `priority`, `batch`                                          |
| `dimension` | `string`  | Repeatable. Only these billed dimensions (e.g. `input_tokens`, `output_tokens`, `output_images`)         |
| `variants`  | `boolean` | `false` = base price card only: standard tier without cache-TTL, routing, or long-context rows (media spec rows are kept) |
| `currency`  | `string`  | Only prices in this ISO currency code, case-insensitive; must be a currency present in the catalog (e.g. `USD`, `EUR` on the AWS European Sovereign Cloud (EUSC)) |
| `routing`   | `string`  | Only this published serving-profile price variant: `global`, `latency`. Row `routing` values enriched for display (geography prefixes, AWS regions) cannot be filtered on — use `region` for those |
| `context`   | `string`  | Only this context-length bucket: `long` (prompts beyond 200K tokens)                                     |
| `all_prices` | `boolean` | `true` = the full published price table; `false` (default) = only the prices matching the server's configuration |

## Response Fields

One `ModelPricing` object per requested model, in request order (duplicates removed); with no `model` filter, one per available model, sorted by ID:

| Field | Description |
|-------|-------------|
| `id` | The model ID as requested |
| `service` | AWS service/API the prices apply to (e.g. `bedrock-runtime`). Note: this uses AWS API endpoint identifiers, a different vocabulary from the display names in the [`search_models`](api_search_models.md) `service` field — both match the code, but the values are not comparable |
| `default_tier` | Service tier this server applies to the model by default ([`DEFAULT_MODEL_SERVICE_TIERS`](operations_configuration.md#default-model-service-tiers)) |
| `default_routings` | Serving profiles this server can use for the model across its configured regions, in configured-region order: `global`, geography prefixes (`eu`, `us`, …), or AWS regions |
| `prices` | Price rows, sorted by region then remaining axes; empty when AWS publishes none |

Each row in `prices` (axes are omitted when not applicable):

| Field | Description |
|-------|-------------|
| `region` | AWS region the price applies to |
| `dimension` | Billed dimension — same names as the [usage log entries](operations_logging_monitoring.md#usage-metrics-fields) |
| `tier` | Service tier: `standard`, `flex`, `priority`, `batch` |
| `cache_ttl` | Prompt-cache write TTL bucket (`5m`, `1h`), when distinctly priced |
| `routing` | Serving profile: `global` (global cross-region inference), a geography prefix like `eu`/`us` (regional cross-region inference), an AWS region (single-region inference), or `latency` for the latency-optimized variant |
| `spec` | Media bucket, e.g. image `resolution:quality` (`1024:standard`) or a modality qualifier |
| `context` | `long` for the beyond-200K-tokens prompt rate |
| `unit_price` | Exact plain-decimal price per **one** billed unit (no exponent, no trailing zeros) |
| `currency` | ISO currency code (`USD` commercially, `EUR` on EUSC) |

## Examples

The `curl` examples below use a `$BASE` variable set to your scheme and host — native routes such as `/model_pricing` are not prefixed:

```bash
export BASE="https://your-host"
```

**Base price card of a shortlist:**

```bash
curl -G "$BASE/model_pricing" \
  --data-urlencode "model=anthropic.claude-sonnet-4-5-20250929-v1:0" \
  --data-urlencode "model=amazon.nova-pro-v1:0" \
  --data-urlencode "variants=false" \
  -H "Authorization: Bearer $API_KEY"
```

**Base price card of every available model:**

```bash
curl -G "$BASE/model_pricing" \
  --data-urlencode "variants=false" \
  -H "Authorization: Bearer $API_KEY"
```

**Token rates in one region:**

```bash
curl -G "$BASE/model_pricing" \
  --data-urlencode "model=amazon.nova-pro-v1:0" \
  --data-urlencode "region=us-east-1" \
  --data-urlencode "dimension=input_tokens" \
  --data-urlencode "dimension=output_tokens" \
  -H "Authorization: Bearer $API_KEY"
```

**Response (trimmed):**

```json
[
  {
    "id": "amazon.nova-pro-v1:0",
    "service": "bedrock-runtime",
    "default_tier": "standard",
    "default_routings": ["us"],
    "prices": [
      {"region": "us-east-1", "dimension": "input_tokens", "tier": "standard", "routing": "us", "unit_price": "0.0000008", "currency": "USD"},
      {"region": "us-east-1", "dimension": "output_tokens", "tier": "standard", "routing": "us", "unit_price": "0.0000032", "currency": "USD"}
    ]
  }
]
```

## Status Codes

| Status | Cause |
|--------|-------|
| `200` | Success — a model with no published price, or no row matching the filters, still returns `200` with an empty `prices` list |
| `400` | Unknown `tier`, `dimension`, `routing`, `context`, or `currency` filter value |
| `503` | Model pricing is not available on this server, or the price catalog is not loaded yet (retry later) |

## Using `model_pricing` as an MCP Tool

When MCP is enabled, `model_pricing` is exposed as an MCP tool under the same name. The intended agent workflow:

1. Call `search_models` to shortlist model IDs for the task.
2. Call `model_pricing` with the shortlist and compare.
3. Keep responses small: use `variants=false` and `dimension` filters.

```json
{
  "tool": "model_pricing",
  "arguments": {
    "model": ["anthropic.claude-sonnet-4-5-20250929-v1:0", "amazon.nova-pro-v1:0"],
    "variants": false,
    "dimension": ["input_tokens", "output_tokens"]
  }
}
```

!!! tip "Prices are strings on purpose"
    `unit_price` values are exact decimal strings (`"0.000003"`), never floats — JSON floats cannot represent small per-token rates without exponent notation or rounding noise. Parse them with a decimal type for arithmetic.
