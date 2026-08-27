---
title: Ollama Generate API - Amazon Bedrock Models via the Ollama Interface
description: Generate text for a single prompt with Amazon Bedrock models using the Ollama-compatible /api/generate endpoint. Streaming, images, structured output, and thinking.
keywords: Ollama generate API, Ollama compatible API, Amazon Bedrock generate, Ollama /api/generate, streaming NDJSON, structured output, Ollama thinking
---

# Generate API (Ollama Compatible)

Generate text for a single prompt with Amazon Bedrock models through the Ollama `/api/generate` interface.

!!! warning "Route Prefix & Base URL"
    Ollama routes carry **no prefix by default**, so the Generate API sits exactly where an Ollama client expects it from a bare base URL: `/api/generate`. You can add a prefix with the `OLLAMA_ROUTES_PREFIX` configuration variable documented in [Operations Configuration](operations_configuration.md#ollama-routes-prefix).

    The `curl` examples below use a `$BASE` variable — set it to your scheme and host:

    ```bash
    export BASE="https://your-host"  # <scheme>://<host> + OLLAMA_ROUTES_PREFIX, if configured
    ```

!!! tip "Prefer /api/chat"
    [`/api/chat`](api_ollama_chat.md) is the endpoint to prefer for new integrations: it carries conversation history, tool calling and multi-turn thinking. `/api/generate` exists for clients that only speak the single-prompt Ollama Generate API.

## Why Choose the Ollama Generate API?

<div class="grid cards" markdown>

- :material-swap-horizontal: __Drop-in Ollama Compatibility__
  <br>Follows the Ollama `/api/generate` request and response shape, including its newline-delimited JSON streaming transport, so an existing Ollama client works by changing the base URL.

- :material-text-box-outline: __Single-Prompt Simplicity__
  <br>A minimal shape for clients that send one prompt at a time rather than a full conversation.

- :material-code-json: __Structured Output__
  <br>`format` accepts `"json"` or a full JSON Schema, and the answer is constrained to it.

- :material-cloud-lock: __Private AWS Backend__
  <br>Served entirely by Amazon Bedrock models in your own AWS account — no traffic to third-party endpoints.

</div>

## Available Endpoints

| Endpoint        | Method | What It Does                                        | Powered By                | MCP Tool     |
|------------------|--------|-------------------------------------------------------|----------------------------|--------------|
| `/api/generate`  | `POST` | Text generation for a single prompt, Ollama Generate API | Amazon Bedrock chat models | Not exposed  |

!!! note "Not Exposed as an MCP Tool"
    `ollama_generate` duplicates a tool an agent already has on the OpenAI or Anthropic surface, and a redundant tool degrades an agent's tool choice, so it is excluded from the MCP tool set by default. An operator who wants it back names `ollama_generate` in [`MCP_INCLUDE_TOOLS`](operations_configuration.md#mcp-include-tools).

**Example request:**

```bash
curl -X POST "$BASE/api/generate" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "prompt": "Why is the sky blue?",
    "stream": false
  }'
```

**Example response:**

```json
{
  "model": "amazon.nova-micro-v1:0",
  "created_at": "2026-08-27T12:00:00.000000+00:00",
  "response": "The sky is blue because of Rayleigh scattering...",
  "done": true,
  "done_reason": "stop",
  "total_duration": 734567890,
  "prompt_eval_count": 7,
  "eval_count": 42
}
```

## Model Names

Send the model names [`GET /api/tags`](api_ollama_models.md#get-apitags) publishes — those are the canonical identifiers this server resolves directly. A trailing `:latest` is accepted and stripped as a fallback when the exact name is not found. Short aliases accepted on this server's other APIs also work here even though `/api/tags` does not list them. A name learned from ollama.com (for example `llama3.2:3b`) is not available through this server and answers `404`. Every response echoes the model name exactly as the request spelled it, in `model`.

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                          |                  Status                  | Notes                                                              |
|-----------------------------------|:----------------------------------------:|--------------------------------------------------------------------|
| **Input**                         |                                          |                                                                    |
| `prompt`                          |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Full support                                                       |
| `system`                          |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Sent as the conversation's system instruction                      |
| `images`                          |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Multimodal models only; base64, a URL, a data URI or an `s3://` URI |
| `format`                          |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | `"json"` or a JSON Schema object; same behavior as [`/api/chat`](api_ollama_chat.md#structured-output) |
| `options`                         |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | `temperature`, `top_p`, `top_k`, `seed`, `stop`, `num_predict` (max output tokens), `presence_penalty` and `frequency_penalty` are forwarded; runner options (`num_ctx`, `num_gpu`, `num_thread`, `num_batch`, `main_gpu`, `use_mmap`, `min_p`, and any other unknown key) are accepted and ignored |
| `stream`                          |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Newline-delimited JSON; defaults to `true` — see [Streaming](api_ollama_chat.md#streaming) |
| `think`                           |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Boolean, or `low`/`medium`/`high`/`max`; comes back in `thinking` |
| `keep_alive`                      | :material-minus-circle:{ .partial role="img" aria-label="Partial" }  | Ignored for residency — models are never resident — but still tells a request with no prompt apart as a load or an unload, [below](#loading-and-unloading) |
| `raw`, `suffix`, `template`, `context` | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Rejected with `400` — see [Fields Not Available](#fields-not-available) |
| `logprobs` / `top_logprobs`       | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Rejected with `400`                                                |
| **Output**                        |                                          |                                                                    |
| `response`                        |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Full support                                                       |
| `thinking`                        |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Follows [`CHAT_COMPLETIONS_REASONING_FIELD`](operations_configuration.md#chat-completions-reasoning-field); omitted when the operator sets that to `none` |
| `done_reason`                     |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | `stop` or `length`, the only two values Ollama itself emits         |
| `context`                         | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Never returned — see [Fields Not Available](#fields-not-available) |
| **Usage tracking**                |                                          |                                                                    |
| `prompt_eval_count`, `eval_count` |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Real token counts                                                   |
| `total_duration`                  |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Real wall-clock time                                                |
| `load_duration`                   | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Never reported — there is no model-loading phase to measure         |
| `prompt_eval_duration`, `eval_duration` |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Reported only when streaming, measured from the stream itself; omitted on a non-streaming response |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with the Ollama API
* :material-cog:{ .model-dep role="img" aria-label="Model-dependent" } **Available on Select Models** — Check your model's capabilities
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation

</div>

## Streaming

By default (`stream` unset, or `true`), the response body is **newline-delimited JSON** (`application/x-ndjson`): one bare JSON object per line, with no `data:` prefix and no `[DONE]` sentinel. The stream is terminated by an object carrying `"done": true` and the response metrics. See [Streaming](api_ollama_chat.md#streaming) on the Chat API for the transport details and how a mid-stream failure is reported.

```bash
curl -N -X POST "$BASE/api/generate" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "prompt": "Write a haiku about the sea."
  }'
```

```json
{"model":"amazon.nova-micro-v1:0","created_at":"2026-08-27T12:00:00.000000+00:00","response":"Waves","done":false}
{"model":"amazon.nova-micro-v1:0","created_at":"2026-08-27T12:00:00.100000+00:00","response":" crash on silent shores","done":false}
{"model":"amazon.nova-micro-v1:0","created_at":"2026-08-27T12:00:00.200000+00:00","response":"","done":true,"done_reason":"stop","total_duration":812345678,"prompt_eval_count":9,"prompt_eval_duration":123456789,"eval_count":11,"eval_duration":688888889}
```

## Fields Not Available

`raw`, `suffix`, `template` and `context` are each rejected with `400` when set. Every one of them needs the model's own prompt template or tokenizer to honor — the exact text the model was trained to expect around a prompt, or the token IDs of a prior turn — and neither is exposed here. The error names the offending field and points at [`/api/chat`](api_ollama_chat.md) as the alternative. Because there is no token context to hand back, `response.context` is never returned by this endpoint.

Send `prompt` together with `system` for a system-prompted single turn, or use [`/api/chat`](api_ollama_chat.md) for a multi-turn conversation.

## Images

`images` accepts base64-encoded image data, as Ollama does. This server additionally accepts a URL, a data URI or an `s3://` URI in the same field, on models that support image input.

## Loading and Unloading

A request with **no `prompt`** is upstream's way of making a model resident — it is what `ollama run <model>` sends before it opens its REPL — and the same request with `keep_alive` set to `0` is what `ollama stop <model>` sends to evict it. A hosted model is always resident, so both are answered without invoking anything:

```json
{
  "model": "amazon.nova-micro-v1:0",
  "created_at": "2026-01-01T00:00:00Z",
  "response": "",
  "done": true,
  "done_reason": "load"
}
```

`done_reason` is `unload` when `keep_alive` is `0`, `load` otherwise. The answer is a single JSON object whatever `stream` says, as upstream's is.

## Limitations

- `raw`, `suffix`, `template` and `context` are rejected with `400` — see [Fields Not Available](#fields-not-available).
- `logprobs` and `top_logprobs` are rejected with `400` — log probabilities are not available.
- `keep_alive` does not keep anything loaded: models are never resident. It is read only to tell a prompt-less request's `done_reason` apart, `load` from `unload`.
- Runner options inside `options` (`num_ctx`, `num_gpu`, `num_thread`, `num_batch`, `main_gpu`, `use_mmap`, `min_p`, and any other key a local runner would use) are accepted and ignored.
- `load_duration` is never reported: there is no model-loading phase to measure, and a number there would be invented.
- `prompt_eval_duration` and `eval_duration` are reported only when streaming. All duration and count fields are optional in the Ollama API, so a client computing tokens-per-second from a non-streamed response has no duration to divide by.
