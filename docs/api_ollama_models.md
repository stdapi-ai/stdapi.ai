---
title: Ollama Models API - Discover Amazon Bedrock Models via the Ollama Interface
description: List, describe and manage Amazon Bedrock models through the Ollama-compatible /api/tags, /api/show, /api/ps, /api/version and model management endpoints.
keywords: Ollama models API, Ollama compatible API, Amazon Bedrock models, Ollama /api/tags, Ollama /api/show, Ollama /api/ps, Ollama /api/version, Ollama /api/pull
---

# Models API (Ollama Compatible)

Discover, describe and manage Amazon Bedrock models through the Ollama model endpoints.

!!! warning "Route Prefix & Base URL"
    Ollama routes carry **no prefix by default**, so these endpoints sit exactly where an Ollama client expects them from a bare base URL: `/api/tags`, `/api/show`, `/api/ps`, `/api/version`, and so on. You can add a prefix with the `OLLAMA_ROUTES_PREFIX` configuration variable documented in [Operations Configuration](operations_configuration.md#ollama-routes-prefix).

    The `curl` examples below use a `$BASE` variable — set it to your scheme and host:

    ```bash
    export BASE="https://your-host"  # <scheme>://<host> + OLLAMA_ROUTES_PREFIX, if configured
    ```

## Why Choose the Ollama Models API?

<div class="grid cards" markdown>

- :material-swap-horizontal: __Drop-in Ollama Compatibility__
  <br>An Ollama client's usual discovery flow — list, show, check what's resident, check the version — works by changing the base URL.

- :material-format-list-bulleted: __Canonical Model Names__
  <br>`/api/tags` publishes the exact names to send back as `model` on every other Ollama endpoint.

- :material-tag-check: __Honest Capability Hints__
  <br>`capabilities` reports what the catalogue actually knows about a model, never a guess dressed up as a fact.

- :material-cloud-lock: __Private AWS Backend__
  <br>Backed entirely by Amazon Bedrock models in your own AWS account — no traffic to third-party endpoints.

</div>

## Available Endpoints

| Endpoint          | Method   | What It Does                                             | MCP Tool     |
|--------------------|----------|-------------------------------------------------------------|--------------|
| `/api/tags`        | `GET`    | List the models this server can serve                       | Not exposed  |
| `/api/show`        | `POST`   | Describe one model's details and capabilities                | Not exposed  |
| `/api/ps`          | `GET`    | List the models currently resident — always empty            | Not exposed  |
| `/api/version`     | `GET`    | Report the Ollama API version this server is compatible with | Not exposed  |
| `/api/pull`        | `POST`   | Confirm a model is available for use                         | Not exposed  |
| `/api/create`      | `POST`   | Refused — no model store to write to                          | Not exposed  |
| `/api/copy`        | `POST`   | Refused — no model store to write to                          | Not exposed  |
| `/api/push`        | `POST`   | Refused — no model store to write to                          | Not exposed  |
| `/api/delete`      | `DELETE` | Refused — no model store to write to                          | Not exposed  |

!!! note "Not Exposed as an MCP Tool"
    Every operation on this page duplicates a tool an agent already has on the OpenAI or Anthropic surface — model discovery in particular is already covered by [`/search_models`](api_search_models.md) — and a redundant tool degrades an agent's tool choice, so none are in the MCP tool set by default. An operator who wants one back names it (for example `ollama_tags`) in [`MCP_INCLUDE_TOOLS`](operations_configuration.md#mcp-include-tools).

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

Each entry names the model by its **canonical identifier** — the name to send back as `model` on every other Ollama endpoint. `size` is always `0`, and `digest` is a stable, synthetic identifier derived from the model name: usable as a cache key, but explicitly not a hash of any model content, since no model file exists on this server to hash. `details.format`, `parameter_size`, `quantization_level` and `parent_model` are empty because they describe a model file this server does not have; `details.family` carries the model's provider instead. `modified_at` is the model's publication date, falling back to the Unix epoch when that date is unknown.

!!! info "Compared With Ollama Cloud"
    Ollama Cloud returns the same fields for its own cloud models, and leaves the same ones empty: `format`, `parameter_size`, `quantization_level` and `parent_model` are empty strings there too, and `size` is `0` for a model whose weight size it does not publish. Two values differ deliberately: Ollama Cloud leaves `family` empty and `families` null, having no model file to read them from, while this server publishes the model's **provider** in both; and its `digest` is a 12-character abbreviation where this server's is the full 64-character one an Ollama server itself returns.

`HEAD /api/tags` answers `200` as a liveness probe, matching what an Ollama server itself answers.

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

`license`, `modelfile`, `template`, `parameters` and `system` are **omitted entirely**, and `model_info` is always an empty object: each describes a local model file, or GGUF metadata read out of one, and this server has no such file to read. Ollama Cloud omits the same five keys for a cloud model; it does publish `model_info`, because it keeps the GGUF header of the weights it hosts, which is metadata Amazon Bedrock does not expose. Rather than invent an architecture, a parameter count and a context length, this server answers the empty object Ollama uses when it knows none of them.

### Capabilities

`capabilities` can report `completion`, `tools`, `embedding`, `vision` and `audio`, derived from the routes and modalities the catalogue already publishes for the model. It is a **best-effort hint, not a contract** — the backend remains the authority, and a model missing a capability here is still worth trying.

Two capabilities are never advertised:

- **`thinking`** — there is no per-model source recording whether a model reasons, so it is never claimed, even though Ollama Cloud advertises it on every model it hosts. [`think`](api_ollama_chat.md#thinking) can still be sent to any model regardless: one that does not reason simply returns no `thinking` text.
- **`insert`** — fill-in-the-middle completion is not available on this server, on any model.

## `GET /api/ps`

Lists the models currently resident in memory.

```bash
curl "$BASE/api/ps" -H "Authorization: Bearer $API_KEY"
```

```json
{"models": []}
```

Always empty. This is the truth, not a stub: models are served on demand, so nothing is ever loaded before a request or left resident after one. Ollama Cloud does not serve this endpoint to a cloud API key at all and answers `401`, so a client that lists what is resident works here and not there.

## `GET /api/version`

Reports the Ollama API version this server is compatible with.

```bash
curl "$BASE/api/version" -H "Authorization: Bearer $API_KEY"
```

```json
{"version": "0.33.1"}
```

This is a **compatibility declaration**, not this server's own version: an Ollama client uses it to decide which features of the Ollama API it may send. Ollama Cloud answers `0.0.0` here, declaring no version at all, which leaves a version-gating client to guess. `HEAD /api/version` also answers `200`, for clients that probe it as a liveness check.

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

All four always answer `400`.

```bash
curl -X POST "$BASE/api/create" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "my-custom-model"}'
```

```json
{"error": "This server does not store models: the models it offers are hosted and already available, so they cannot be created, copied, published or deleted. Call the model list endpoint to see what is available."}
```

## Limitations

- `POST /api/create`, `POST /api/copy`, `POST /api/push` and `DELETE /api/delete` always answer `400`: this server does not store models, so these operations have no state to change, and answering `200` would tell the caller something changed when nothing did. Ollama Cloud refuses the same four with `401`.
- `GET /` does not answer `Ollama is running` — see the note under [`GET /api/version`](#get-apiversion) above.
- `capabilities` is a best-effort hint derived from the catalogue, never a contract; `thinking` and `insert` are never advertised, for the reasons given under [Capabilities](#capabilities).
- `model_info` is always an empty object, and `details.parameter_size` and `details.quantization_level` always empty strings: Amazon Bedrock publishes no GGUF header for the models it serves, and a plausible-looking invention is worse than the empty value Ollama itself uses for an unknown one.
- `digest` is the full 64-character identifier an Ollama server returns, derived from the model name rather than from any content. Ollama Cloud abbreviates its own to 12 characters; a client comparing digest lengths across the two will see them differ.
