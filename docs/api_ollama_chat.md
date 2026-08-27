---
title: Ollama Chat API - Amazon Bedrock Models via the Ollama Interface
description: Generate conversational AI responses with Amazon Bedrock models using the Ollama-compatible /api/chat endpoint. Streaming, tool calling, images, structured output, and thinking.
keywords: Ollama chat API, Ollama compatible API, Amazon Bedrock chat, Ollama /api/chat, streaming NDJSON, tool calling, structured output, Ollama thinking
---

# Chat API (Ollama Compatible)

Generate conversational AI responses with Amazon Bedrock models through the Ollama `/api/chat` interface.

!!! warning "Route Prefix & Base URL"
    Ollama routes carry **no prefix by default**, so the Chat API sits exactly where an Ollama client expects it from a bare base URL: `/api/chat`. You can add a prefix with the `OLLAMA_ROUTES_PREFIX` configuration variable documented in [Operations Configuration](operations_configuration.md#ollama-routes-prefix).

    The `curl` examples below use a `$BASE` variable — set it to your scheme and host:

    ```bash
    export BASE="https://your-host"  # <scheme>://<host> + OLLAMA_ROUTES_PREFIX, if configured
    ```

## Why Choose the Ollama Chat API?

<div class="grid cards" markdown>

- :material-swap-horizontal: __Drop-in Ollama Compatibility__
  <br>Follows the Ollama `/api/chat` request and response shape, including its newline-delimited JSON streaming transport, so an existing Ollama client works by changing the base URL.

- :material-brain: __Tool Calling and Thinking__
  <br>Function tools and thinking (`think`) work the same way they do against a local Ollama server.

- :material-code-json: __Structured Output__
  <br>`format` accepts `"json"` or a full JSON Schema, and the answer is constrained to it.

- :material-cloud-lock: __Private AWS Backend__
  <br>Served entirely by Amazon Bedrock models in your own AWS account — no traffic to third-party endpoints.

</div>

## Available Endpoints

| Endpoint    | Method | What It Does                                    | Powered By                | MCP Tool     |
|-------------|--------|--------------------------------------------------|----------------------------|--------------|
| `/api/chat` | `POST` | Conversational AI, following the Ollama Chat API | Amazon Bedrock chat models | Not exposed  |

!!! note "Not Exposed as an MCP Tool"
    `ollama_chat` duplicates a tool an agent already has on the OpenAI or Anthropic surface, and a redundant tool degrades an agent's tool choice, so it is excluded from the MCP tool set by default. An operator who wants it back names `ollama_chat` in [`MCP_INCLUDE_TOOLS`](operations_configuration.md#mcp-include-tools).

**Example request:**

```bash
curl -X POST "$BASE/api/chat" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "messages": [{"role": "user", "content": "Say hello world"}],
    "stream": false
  }'
```

**Example response:**

```json
{
  "model": "amazon.nova-micro-v1:0",
  "created_at": "2026-08-27T12:00:00.000000+00:00",
  "message": {"role": "assistant", "content": "Hello, world!"},
  "done": true,
  "done_reason": "stop",
  "total_duration": 812345678,
  "prompt_eval_count": 6,
  "eval_count": 5
}
```

## Model Names

Send the model names [`GET /api/tags`](api_ollama_models.md#get-apitags) publishes — those are the canonical identifiers this server resolves directly. A trailing `:latest` is accepted and stripped as a fallback when the exact name is not found. Short aliases accepted on this server's other APIs also work here even though `/api/tags` does not list them. A name learned from ollama.com (for example `llama3.2:3b`) is not available through this server and answers `404`. Every response echoes the model name exactly as the request spelled it, in `model`.

**Find compatible models:** Call [`/search_models`](api_search_models.md) with `route=ollama_chat` to discover model IDs that support this route, or call [`GET /api/tags`](api_ollama_models.md#get-apitags) for the Ollama-shaped listing.

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                         |                  Status                  | Notes                                                              |
|----------------------------------|:----------------------------------------:|--------------------------------------------------------------------|
| **Input**                       |                                          |                                                                    |
| `messages` (text)                |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Full support                                                       |
| `messages[].images`              |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Multimodal models only; base64, a URL, a data URI or an `s3://` URI |
| `messages[].thinking`            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Replayed as the assistant turn's reasoning text                    |
| `messages[].tool_calls`          |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Replayed tool calls; correlated to results as described [below](#tool-calling) |
| `tools`                          |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Function tools; support depends on the model                       |
| `format`                         |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | `"json"` or a JSON Schema object; see [Structured Output](#structured-output) |
| `options`                        |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | `temperature`, `top_p`, `top_k`, `seed`, `stop`, `num_predict` (max output tokens), `presence_penalty` and `frequency_penalty` are forwarded; runner options (`num_ctx`, `num_gpu`, `num_thread`, `num_batch`, `main_gpu`, `use_mmap`, `min_p`, and any other unknown key) are accepted and ignored |
| `stream`                         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Newline-delimited JSON; defaults to `true` — see [Streaming](#streaming) |
| `think`                          |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Boolean, or `low`/`medium`/`high`/`max`; see [Thinking](#thinking) |
| `keep_alive`                     | :material-minus-circle:{ .partial role="img" aria-label="Partial" }  | Ignored for residency — models are never resident — but still tells a request with no message apart as a load or an unload, [below](#loading-and-unloading) |
| `logprobs` / `top_logprobs`      | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Rejected with `400`                                                |
| **Output**                       |                                          |                                                                    |
| `message.content`                |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Full support                                                       |
| `message.thinking`               |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Follows [`CHAT_COMPLETIONS_REASONING_FIELD`](operations_configuration.md#chat-completions-reasoning-field); omitted when the operator sets that to `none` |
| `message.tool_calls`             |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Streamed whole in one event, never as partial argument fragments — see [Tool Calling](#tool-calling) |
| `done_reason`                    |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | `stop` or `length`, the only two values Ollama itself emits         |
| **Usage tracking**               |                                          |                                                                    |
| `prompt_eval_count`, `eval_count`|   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Real token counts                                                   |
| `total_duration`                 |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Real wall-clock time                                                |
| `load_duration`                  | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Never reported — there is no model-loading phase to measure         |
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

By default (`stream` unset, or `true`), the response body is **newline-delimited JSON** (`application/x-ndjson`): one bare JSON object per line, with no `data:` prefix and no `[DONE]` sentinel. The stream is terminated by an object carrying `"done": true` and the response metrics.

```bash
curl -N -X POST "$BASE/api/chat" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "messages": [{"role": "user", "content": "Write a haiku about the sea."}]
  }'
```

```json
{"model":"amazon.nova-micro-v1:0","created_at":"2026-08-27T12:00:00.000000+00:00","message":{"role":"assistant","content":"Waves"},"done":false}
{"model":"amazon.nova-micro-v1:0","created_at":"2026-08-27T12:00:00.100000+00:00","message":{"role":"assistant","content":" crash on silent shores"},"done":false}
{"model":"amazon.nova-micro-v1:0","created_at":"2026-08-27T12:00:00.200000+00:00","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","total_duration":812345678,"prompt_eval_count":9,"prompt_eval_duration":123456789,"eval_count":11,"eval_duration":688888889}
```

!!! warning "A Streamed Failure Is Not an HTTP Status"
    Once the stream has begun, a failure can no longer be reported as an HTTP status: the response headers were already sent with a `200`. Instead, the stream ends with a final `{"error": "<message>"}` line in place of the terminal `done: true` object. Check every line for an `error` key rather than relying on the status code alone.

## Tool Calling

Ollama tool calls carry no identifier of their own. When replaying a conversation, `messages[].tool_calls` entries are matched to the following `tool` message that answers them by `tool_call_id` when the client sent one, then by `tool_name`, then in call order.

Streamed tool calls are emitted **whole**, in a single event once the model has finished requesting them, rather than as partial argument fragments — matching what Ollama itself does.

```json
{"model":"amazon.nova-micro-v1:0","created_at":"2026-08-27T12:00:00.000000+00:00","message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"get_weather","arguments":{"city":"Paris"},"index":0}}]},"done":false}
```

## Structured Output

Set `format` to `"json"` for unstructured JSON mode, or to a JSON Schema object for validated structured output:

```bash
curl -X POST "$BASE/api/chat" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-5",
    "messages": [{"role": "user", "content": "Extract the city and country."}],
    "format": {
      "type": "object",
      "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
      "required": ["city", "country"]
    },
    "stream": false
  }'
```

An object schema that does not set `additionalProperties` is closed automatically (`additionalProperties: false`), so a schema written for a local Ollama server works unchanged here.

A schema constrains the answer only on models whose backend accepts one; a model that does not answers `400` naming the parameter. `"json"` mode is more widely available. Filter with [`/search_models`](api_search_models.md) or try the request — the model is the authority.

## Thinking

Set `think` to `true`, or to `low`, `medium`, `high` or `max`, to request the model's reasoning. The reasoning text comes back in `message.thinking`, both in the final response and on the streaming deltas that carry it.

```bash
curl -X POST "$BASE/api/chat" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-5",
    "think": "high",
    "messages": [{"role": "user", "content": "Solve 12*13"}],
    "stream": false
  }'
```

```json
{
  "model": "anthropic.claude-sonnet-5",
  "created_at": "2026-08-27T12:00:00.000000+00:00",
  "message": {
    "role": "assistant",
    "thinking": "12 x 10 = 120, plus 12 x 3 = 36 -> 156",
    "content": "156"
  },
  "done": true,
  "done_reason": "stop",
  "total_duration": 950123456,
  "prompt_eval_count": 14,
  "eval_count": 22
}
```

`message.thinking` follows the [`CHAT_COMPLETIONS_REASONING_FIELD`](operations_configuration.md#chat-completions-reasoning-field) server setting: when an operator sets it to `none`, no thinking text is emitted on this API either, whatever `think` was sent.

## Images

`messages[].images` accepts base64-encoded image data, as Ollama does. This server additionally accepts a URL, a data URI or an `s3://` URI in the same field, on models that support image input.

## Loading and Unloading

A request with an **empty `messages` array** is upstream's way of making a model resident, and the same request with `keep_alive` set to `0` is how it is evicted — what a client's "load model" and "unload model" controls send. A hosted model is always resident, so both are answered without invoking anything:

```json
{
  "model": "amazon.nova-micro-v1:0",
  "created_at": "2026-01-01T00:00:00Z",
  "message": { "role": "assistant", "content": "" },
  "done": true,
  "done_reason": "load"
}
```

`done_reason` is `unload` when `keep_alive` is `0`, `load` otherwise. The answer is a single JSON object whatever `stream` says, as upstream's is.

## Limitations

- `logprobs` and `top_logprobs` are rejected with `400` — log probabilities are not available.
- `keep_alive` does not keep anything loaded: models are never resident. It is read only to tell a message-less request's `done_reason` apart, `load` from `unload`.
- Runner options inside `options` (`num_ctx`, `num_gpu`, `num_thread`, `num_batch`, `main_gpu`, `use_mmap`, `min_p`, and any other key a local runner would use) are accepted and ignored.
- `load_duration` is never reported: there is no model-loading phase to measure, and a number there would be invented.
- `prompt_eval_duration` and `eval_duration` are reported only when streaming. All duration and count fields are optional in the Ollama API, so a client computing tokens-per-second from a non-streamed response has no duration to divide by.
