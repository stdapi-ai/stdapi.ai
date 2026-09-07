---
title: Ollama Models API - Discover Amazon Bedrock Models via the Ollama Interface
description: List, describe and manage Amazon Bedrock models through the Ollama-compatible /api/tags, /api/show, /api/ps, /api/version and model management endpoints.
keywords: Ollama models API, Ollama compatible API, Amazon Bedrock models, Ollama /api/tags, Ollama /api/show, Ollama /api/ps, Ollama /api/version, Ollama /api/pull
---

# Models API (Ollama Compatible)

Discover, describe and manage Amazon Bedrock models through the Ollama model endpoints. Served under `/ollama` by default; the examples below use `$BASE`, which includes that prefix.

## At a glance

- :material-swap-horizontal: **Drop-in Ollama compatibility** — An Ollama client's usual discovery flow — list, show, check what's resident, check the version — works by changing the base URL.
- :material-format-list-bulleted: **Canonical model names** — `/api/tags` publishes the exact names to send back as `model` on every other Ollama endpoint.
- :material-tag-check: **Honest capability hints** — `capabilities` reports what the catalogue actually knows about a model, never a guess dressed up as a fact.
- :material-cloud-lock: **Private AWS backend** — Backed entirely by Amazon Bedrock models in your own AWS account — no traffic to third-party endpoints.
- :material-swap-horizontal: **Differs from the Ollama API:** the four store-writing verbs answer `403`, `/api/ps` is always empty, `model_info` is always `{}`, and `GET /` does not answer `Ollama is running` — see [Limits and behaviour to know](#limitations).

!!! info "Base URL and route prefix"
    By default, all Ollama-compatible routes are prefixed with `/ollama`. This means these endpoints are available at `/ollama/api/tags`, `/ollama/api/show`, `/ollama/api/ps`, `/ollama/api/version`, and so on, instead of their bare paths. You can customize this prefix using the `OLLAMA_ROUTES_PREFIX` configuration variable documented in [HTTP Server and MCP](operations_configuration_server.md#ollama-routes-prefix).

    The `curl` examples on this page use a `$BASE` variable that **must include this prefix** — set it to your scheme and host followed by `OLLAMA_ROUTES_PREFIX`:

    ```bash
    export BASE="https://your-host/ollama"  # <scheme>://<host> + OLLAMA_ROUTES_PREFIX
    ```

```bash
curl "$BASE/api/tags" -H "Authorization: Bearer $API_KEY"
```

## Endpoints { #available-endpoints }

| Endpoint          | Method   | What It Does                                             | MCP Tool        |
|--------------------|----------|-------------------------------------------------------------|-----------------|
| `/api/tags`        | `GET`    | List the models this server can serve                       | `ollama_tags`   |
| `/api/show`        | `POST`   | Describe one model's details and capabilities                | `ollama_show`   |
| `/api/ps`          | `GET`    | List the models currently resident — always empty            | `ollama_ps`     |
| `/api/version`     | `GET`    | Report the Ollama API version this server is compatible with | `ollama_version`|
| `/api/pull`        | `POST`   | Confirm a model is available for use                         | `ollama_pull`   |
| `/api/create`      | `POST`   | Refused — no model store to write to                          | Not exposed     |
| `/api/copy`        | `POST`   | Refused — no model store to write to                          | Not exposed     |
| `/api/push`        | `POST`   | Refused — no model store to write to                          | Not exposed     |
| `/api/delete`      | `DELETE` | Refused — no model store to write to                          | Not exposed     |

!!! note "The Four Refused Verbs Are Not Exposed as MCP Tools"
    `ollama_create`, `ollama_copy`, `ollama_push` and `ollama_delete` always refuse, since this deployment stores no models — a tool schema for a call that can never succeed would only mislead an agent — so none of the four is in the MCP tool set by default. Every other operation on this page is published; model discovery is also covered by [`/search_models`](api_search_models.md). An operator who wants a refused verb back anyway names it (for example `ollama_create`) in [`MCP_INCLUDE_TOOLS`](operations_configuration_server.md#mcp-include-tools).

## `GET /api/tags`

Lists the models this server can serve **through the Ollama endpoints** — a model without a chat or embedding route reachable from this dialect is not listed.

```bash
curl "$BASE/api/tags" -H "Authorization: Bearer $API_KEY"
```

```json
{
  "models": [
    {
      "name": "amazon.nova-micro-v1:0",
      "model": "amazon.nova-micro-v1:0",
      "modified_at": "2024-12-03T00:00:00+00:00",
      "size": 0,
      "digest": "3f1a9c...e2b7",
      "details": {
        "parent_model": "",
        "format": "",
        "family": "Amazon",
        "families": ["Amazon"],
        "parameter_size": "",
        "quantization_level": ""
      }
    }
  ]
}
```

Each entry names the model by its **canonical identifier** — the name to send back as `model` on every other Ollama endpoint. `size` is always `0`, and `digest` is a stable, synthetic identifier derived from the model name, 64 characters like an Ollama server's own: usable as a cache key, but explicitly not a hash of any model content, since no model file exists on this server to hash. `details.format`, `parameter_size`, `quantization_level` and `parent_model` are empty because they describe a model file this server does not have; `details.family` and `details.families` carry the model's **provider** instead. `modified_at` is the model's publication date, falling back to the Unix epoch when that date is unknown.

`HEAD /api/tags` answers `200` as a liveness probe, matching what an Ollama server itself answers. Like every other route it requires the API key, so it answers `401` without one; a load-balancer health check has no key to send and should target the gateway's own [`/health`](operations_getting_started.md) endpoint, which needs no authentication.

## `POST /api/show`

Describes one model. The model is named in the request **body**, not the URL — `POST` with `{"model": "..."}`; `name` is accepted as a legacy alias of `model`.

```bash
curl -X POST "$BASE/api/show" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "amazon.nova-micro-v1:0"}'
```

```json
{
  "details": {
    "parent_model": "",
    "format": "",
    "family": "Amazon",
    "families": ["Amazon"],
    "parameter_size": "",
    "quantization_level": ""
  },
  "model_info": {},
  "capabilities": ["completion", "tools"],
  "modified_at": "2024-12-03T00:00:00+00:00"
}
```

`license`, `modelfile`, `template`, `parameters` and `system` are **omitted entirely**, and `model_info` is always an empty object: each describes a local model file, or GGUF metadata read out of one, and Amazon Bedrock publishes no such file or header. Rather than invent an architecture, a parameter count and a context length, this server answers the empty object Ollama uses when it knows none of them.

### Capabilities

`capabilities` can report `completion`, `tools`, `embedding`, `vision` and `audio`, derived from the routes and modalities the catalogue already publishes for the model. It is a **best-effort hint, not a contract** — the backend remains the authority, and a model missing a capability here is still worth trying.

Two capabilities are never advertised:

- **`thinking`** — there is no per-model source recording whether a model reasons, so it is never claimed. [`think`](api_ollama_chat.md#thinking) can still be sent to any model regardless: one that does not reason simply returns no `thinking` text.
- **`insert`** — fill-in-the-middle completion is not available on this server, on any model.

## `GET /api/ps`

Lists the models currently resident in memory.

```bash
curl "$BASE/api/ps" -H "Authorization: Bearer $API_KEY"
```

```json
{"models": []}
```

Always empty. This is the truth, not a stub: models are served on demand, so nothing is ever loaded before a request or left resident after one.

## `GET /api/version`

Reports the Ollama API version this server is compatible with.

```bash
curl "$BASE/api/version" -H "Authorization: Bearer $API_KEY"
```

```json
{"version": "0.33.1"}
```

This is a **compatibility declaration**, not this server's own version: an Ollama client uses it to decide which features of the Ollama API it may send. `HEAD /api/version` also answers `200`, for clients that probe it as a liveness check — and, like every other route, it requires the API key.

!!! warning "GET / Does Not Answer \"Ollama is running\""
    A real Ollama server answers `GET /` with the plain-text body `Ollama is running`, and some clients probe that path to detect one. On this server, `GET /` is the server's own root document, unrelated to Ollama compatibility. Clients that need to detect this server as an Ollama-compatible endpoint should probe `GET /api/version` instead.

## Model Management

Every model this server offers is already available and none of them is stored locally, which leaves one verb whose post-condition can be met, and four that would have to change a model store this server does not have.

### `POST /api/pull`

Answers success immediately for any model `/api/tags` lists — nothing is transferred, because the model is already usable — and `404` for a model this server does not offer. `insecure` is accepted and ignored.

```bash
curl -X POST "$BASE/api/pull" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "amazon.nova-micro-v1:0", "stream": false}'
```

```json
{"status": "success"}
```

By default (`stream` unset, or `true`) the same status is streamed as a single newline-delimited JSON object: `{"status":"success"}`. Set `"stream": false` to receive it as one JSON object instead, as in the example above.

### `POST /api/create`, `POST /api/copy`, `POST /api/push`, `DELETE /api/delete`

All four always answer `403`: the request is well-formed, the server simply will not perform it.

```bash
curl -X POST "$BASE/api/create" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "my-custom-model"}'
```

```json
{"error": "This server does not store models: the models it offers are hosted and already available, so they cannot be created, copied, published or deleted. Call the model list endpoint to see what is available."}
```

## Limits and behaviour to know { #limitations }

- `POST /api/create`, `POST /api/copy`, `POST /api/push` and `DELETE /api/delete` always answer `403`: this server does not store models, so these operations have no state to change, and answering `200` would tell the caller something changed when nothing did.
- `GET /` does not answer `Ollama is running` — see the note under [`GET /api/version`](#get-apiversion) above.
- `capabilities` is a best-effort hint derived from the catalogue, never a contract; `thinking` and `insert` are never advertised, for the reasons given under [Capabilities](#capabilities).
- `model_info` is always an empty object, and `details.parameter_size` and `details.quantization_level` always empty strings: Amazon Bedrock publishes no GGUF header for the models it serves, and a plausible-looking invention is worse than the empty value Ollama itself uses for an unknown one.
- `digest` is the full 64-character identifier an Ollama server returns, derived from the model name rather than from any content, so it is a stable cache key and never a content hash.
- `HEAD /api/tags` and `HEAD /api/version` require the API key like every other route, so they answer `401` to an unauthenticated probe. Point a load balancer at `/health` instead.

## Request headers

| Header          | Purpose         | Notes                                           |
|-----------------|-----------------|-------------------------------------------------|
| `Authorization` | Gateway API key | `Bearer <key>`, required like every other route |

A local Ollama server needs no key; this one does, on every route including the `HEAD` probes.

## Try it

```bash
# List what this server serves
curl "$BASE/api/tags" -H "Authorization: Bearer $API_KEY"

# Describe one model
curl -X POST "$BASE/api/show" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "amazon.nova-micro-v1:0"}'

# Check the compatibility version
curl "$BASE/api/version" -H "Authorization: Bearer $API_KEY"
```

## Compared with Ollama Cloud

Both servers answer the Ollama model endpoints for models they host rather than store, so most fields agree: `format`, `parameter_size`, `quantization_level` and `parent_model` are empty strings on both, `size` is `0`, and `license`, `modelfile`, `template`, `parameters` and `system` are omitted from `/api/show`. Where the two differ:

| Field or endpoint     | This server                                              | Ollama Cloud                                                |
|-----------------------|----------------------------------------------------------|-------------------------------------------------------------|
| `details.family` / `families` | The model's provider                              | Empty and null — it reads them from a model file it has none of |
| `digest`              | Full 64 characters, as an Ollama server returns          | Abbreviated to 12 characters                                 |
| `model_info`          | `{}` — Amazon Bedrock publishes no GGUF header           | Populated from the GGUF header of the weights it hosts       |
| `capabilities`        | Never claims `thinking`                                  | Advertises `thinking` on every model it hosts                |
| `GET /api/ps`         | Answers `{"models": []}`                                 | Answers `401` to a cloud API key                             |
| `GET /api/version`    | The Ollama API version this server is compatible with    | `0.0.0`                                                      |
| The four store verbs  | `403`                                                    | `401`                                                        |

## Next steps

Next: [Chat API](api_ollama_chat.md) · [Generate API](api_ollama_generate.md) · [Embed API](api_ollama_embed.md) · [Search Models API](api_search_models.md)
