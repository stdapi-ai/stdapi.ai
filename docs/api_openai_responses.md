---
title: Responses API - Amazon Bedrock with OpenAI Compatibility
description: OpenAI-compatible Responses API for Amazon Bedrock models. Supports streaming, tool calling, structured outputs, and multimodal inputs.
keywords: responses API, OpenAI responses API, Amazon Bedrock responses, streaming responses, tool calling API, structured output
---

# Responses API

Generate model responses with Amazon Bedrock foundation models through an OpenAI Responses API-compatible interface. Supports text, images, tool calling, and streaming.

## Why Choose the Responses API?

<div class="grid cards" markdown>

- :material-tools: __Tool Calling__
  <br>Define function tools and get structured tool calls back. Full round-trip support with `function_call_output`.

- :material-code-json: __Structured Output__
  <br>Request JSON object or JSON schema output via `text.format` to get machine-readable responses.

- :material-lightning-bolt: __Streaming__
  <br>Real-time token streaming with granular events for text deltas, tool calls, and lifecycle milestones.

- :material-brain: __Extended Reasoning__
  <br>Enable chain-of-thought reasoning on supported models via `reasoning.effort`.

</div>

## Quick Start: Available Endpoints

| Endpoint                                  | Method   | What It Does                                                             | Powered By                                | MCP Tool                       |
|-------------------------------------------|----------|--------------------------------------------------------------------------|-------------------------------------------|--------------------------------|
| `/v1/responses`                           | `POST`   | Create a model response                                                  | Amazon Bedrock Converse API · Amazon Bedrock Mantle | `openai_response`              |
| `/v1/responses/input_tokens`              | `POST`   | Count input tokens without generating a response                         | Amazon Bedrock CountTokens API               | `openai_response_input_tokens` |
| `/v1/responses/compact`                   | `POST`   | Compact a conversation into a reusable summary                           | Amazon Bedrock Converse API                  | `openai_response_compact`      |
| `/v1/responses/{response_id}`             | `GET`    | Retrieve a stored response                                               | Amazon Bedrock Sessions · Bedrock Mantle     | `openai_response_get`          |
| `/v1/responses/{response_id}`             | `DELETE` | Delete a stored response                                                 | Amazon Bedrock Sessions · Bedrock Mantle     | `openai_response_delete`       |
| `/v1/responses/{response_id}/cancel`      | `POST`   | Cancel a background response — see [Stored Responses](#stored-responses) | Amazon Bedrock Sessions · Bedrock Mantle     | `openai_response_cancel`       |
| `/v1/responses/{response_id}/input_items` | `GET`    | List the input items of a stored response                                | Amazon Bedrock Sessions                      | `openai_response_input_items`  |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                                                               |                 Status                  | Notes                                                                        |
|-----------------------------------------------------------------------|:---------------------------------------:|------------------------------------------------------------------------------|
| **Input**                                                             |                                         |                                                                              |
| Plain text (`input` as string)                                        |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Simple string shorthand for a single user message                            |
| Structured message array                                              |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Array of `EasyInputMessage` / `InputMessage` items                           |
| `instructions` (system prompt)                                        |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Injected as a Bedrock system block                                           |
| `system` / `developer` role                                           |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Treated as a system instruction                                              |
| Image input (`input_image`)                                           |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | HTTP URLs and base64 data URIs supported                                     |
| File input (`input_file`)                                             |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | File URLs and base64 data supported                                          |
| `function_call_output`                                                |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Submit tool results as input; supports text, image, and file parts           |
| Echoed output items (message, reasoning, refusal)                     |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Replayed to the model; refusal parts preserved; unknown upstream fields tolerated |
| Echoed `custom_tool_call` / `image_generation_call`                   |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Replayed as tool calls (freeform input wrapped as `{"input": ...}`; image results attached) |
| Echoed `file_search_call`                                             |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Replayed as a search the model already ran, with the passages it returned; an item carrying no `results` is dropped with its call |
| Hosted-tool call items (`web_search_call`, `code_interpreter_call`, `computer_call`, `tool_search_call`, shell/apply-patch/MCP items, `compaction_trigger`) | :material-check-circle:{ .success role="img" aria-label="Supported" } | Input-history tolerance only: echoed items are accepted and dropped on replay (no Bedrock equivalent). Whether each *tool* can actually be used is listed under Tool Calling below |
| Echoed `program` / `program_output` items                             |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Input-history tolerance only: accepted and dropped on replay (no Bedrock equivalent) |
| `item_reference`                                                      |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Accepted and dropped on replay                                               |
| **Tool Calling**                                                      |                                         |                                                                              |
| Function tools (`type: "function"`)                                   |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Full schema mapping to Bedrock toolSpec                                      |
| `tool_choice: "auto"`                                                 |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Model selects among available tools                                          |
| `tool_choice: "required"`                                             |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Model must call at least one tool                                            |
| `tool_choice: "none"`                                                 |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Prevents tool calls                                                          |
| Named `tool_choice` (force)                                           |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Force a specific function to be called                                       |
| `tool_choice: allowed_tools`                                          |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Approximated: `required` + 1 function → forced tool; `required` + many → any tool; `auto` → auto; type-variants add no constraint |
| `parallel_tool_calls`                                                 |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Accepted for every model and honored by models able to constrain tool use; echoed in the response, which reports the tool calls actually made |
| Built-in tools (`code_interpreter`, `web_search`, `image_generation`) |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | See [OpenAI Integrated Tools](#openai-integrated-tools)                      |
| `file_search` tool                                                    |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Served from the vector stores named in `vector_store_ids` (see [File Search](#file-search)); forwarded upstream on Bedrock Mantle native models |
| `web_search` `filters.allowed_domains` / `user_location`              |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Rejected with a `400` where the search cannot be restricted; honored on Bedrock Mantle native models |
| `web_search` `search_context_size`                                    |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Accepted and ignored — the answer is still searched and cited; honored on Bedrock Mantle native models |
| `computer` / `computer_use_preview` tools                             |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | No Converse equivalent — accepted and dropped (see [Computer Use Not Supported](#computer-use-not-supported)); forwarded upstream on Bedrock Mantle native models |
| `mcp` tool                                                            |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | No Converse equivalent — accepted and dropped; forwarded upstream on Bedrock Mantle native models |
| `local_shell` / `shell` tools                                         | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Accepted and dropped; no Bedrock equivalent                                  |
| `custom` / `namespace` / `tool_search` / `apply_patch` tools          | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Accepted and dropped; no Bedrock equivalent                                  |
| `programmatic_tool_calling` tool / `tool_choice`                      |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | No Converse equivalent — accepted and dropped, the model calls the declared tools directly (the `tool_choice` degrades to the model's default choice); forwarded upstream on Bedrock Mantle native models |
| **Generation Control**                                                |                                         |                                                                              |
| `max_output_tokens`                                                   |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Maps to Bedrock `maxTokens`                                                  |
| `temperature`                                                         |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | 0–2 range; mapped to Bedrock inference config                                |
| `top_p`                                                               |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | 0–1 range; nucleus sampling                                                  |
| `top_logprobs`                                                        |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | 0–20 range accepted and echoed; log probabilities are never returned on Converse-served models; forwarded upstream on Bedrock Mantle native models |
| `reasoning` (effort)                                                  |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Configures reasoning; without `effort` defaults to `medium`; `effort: "none"` disables; chain of thought returned as `reasoning` output items |
| `reasoning.summary` / `generate_summary`                              |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Accepted but ignored on Converse-served models — no summary is generated; forwarded upstream on Bedrock Mantle native models |
| `reasoning.context`                                                   |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Accepted but ignored on Converse-served models — context scoping is not applied; forwarded upstream on Bedrock Mantle native models |
| `reasoning.mode`                                                      |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Accepted but ignored on Converse-served models — pro-mode reasoning selection is not applied; forwarded upstream on Bedrock Mantle native models |
| `text.verbosity`                                                      |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Accepted but ignored on Converse-served models; forwarded upstream on Bedrock Mantle native models |
| `include`                                                             |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }   | `reasoning.encrypted_content` is honored; other values are accepted and ignored (forwarded upstream on Bedrock Mantle native models) |
| `metadata`                                                            |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Forwarded to Bedrock `requestMetadata`                                       |
| `prompt_cache_key`                                                    |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Cache prompts to reduce costs and latency                                    |
| `prompt_cache_options`                                                |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | `mode: "explicit"` caches only the parts marked with `prompt_cache_breakpoint`; `ttl: "30m"` mapped to a 1 hour Amazon Bedrock retention on Anthropic models (other models use the default 5 minute TTL) when `prompt_cache_retention` is unset; echoed on the response; forwarded upstream on Bedrock Mantle native models |
| `prompt_cache_breakpoint` (input content part)                        |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Explicit cache boundary mapped to an Amazon Bedrock `cachePoint` (max. 4 per request) |
| `prompt_cache_retention`                                              |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Cache TTL: `in_memory`, `24h`, `1h`, or `5m`                                 |
| `service_tier`                                                        |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Maps to Bedrock service tier header                                          |
| Extra model-specific params                                           | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Extra model-specific parameters not supported by the OpenAI API              |
| `truncation`                                                          |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }   | `disabled` (the OpenAI default) is the behavior served, and is accepted; `auto` returns `400` |
| `max_tool_calls`                                                      | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Returns `400`; not supported                                                 |
| `context_management`                                                  | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Returns `400`; not supported                                                 |
| `background`                                                          |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Accepted but ignored on Converse-served models — execution is synchronous; forwarded upstream on Bedrock Mantle native models, where background responses can be cancelled — see [Stored Responses](#stored-responses) |
| `store`                                                               |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Persists the response — Amazon Bedrock session storage (non-streaming) or Mantle native storage for Mantle models (streaming supported) |
| `stream_options`                                                      |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Accepted but ignored on Converse-served models; forwarded upstream on Bedrock Mantle native models |
| `conversation`                                                        |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Prepends the conversation's items to `input` and appends the turn to it unless `store` is false; rejected (`400`) with `previous_response_id` — see [Conversations](api_openai_conversations.md) |
| `prompt` (template reference)                                         |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }   | Amazon Bedrock Prompt Management prompt ARN only, when enabled server-side — see [Managed Prompt Templates](#managed-prompt-templates) |
| `safety_identifier` / `user`                                          |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }   | Does not affect generation; identifies the end user in the request log and in [per-user cost attribution](operations_cost_management.md#per-user-attribution) |
| `client_metadata`                                                     |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Accepted but ignored on Converse-served models (sent by newer OpenAI clients such as Codex); forwarded upstream on Bedrock Mantle native models |
| `moderation`                                                          |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Applies an Amazon Bedrock guardrail; results in the response `moderation` field (on the terminal event when streaming) — rejected (`400`) on Mantle-served models |
| **Output Format**                                                     |                                         |                                                                              |
| `text.format: "text"`                                                 |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Plain text output                                                            |
| `text.format: "json_object"`                                          |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Accepted for all models; syntactically valid JSON is not guaranteed for every model |
| `text.format: "json_schema"`                                          |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Structured JSON output with schema validation                                |
| **Multi-Turn**                                                        |                                         |                                                                              |
| `previous_response_id`                                                |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Continues a response stored with `store=true`                                |
| Compaction (`POST /v1/responses/compact`)                             |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Stateless summary item; send it back in `input` to continue                  |
| **Streaming**                                                         |                                         |                                                                              |
| `stream: true`                                                        |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | SSE stream with full lifecycle events                                        |
| `response.created`                                                    |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Emitted at stream start                                                      |
| `response.in_progress`                                                |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Emitted after created                                                        |
| `response.output_text.delta`                                          |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Text token deltas                                                            |
| `response.output_text.done`                                           |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Final text for each content part                                             |
| `response.function_call_arguments.delta`                              |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Tool call argument deltas                                                    |
| `response.function_call_arguments.done`                               |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Finalized tool call arguments                                                |
| `response.reasoning_text.delta` / `.done`                             |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Reasoning text deltas on reasoning models                                    |
| `response.output_text.annotation.added`                               |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | `url_citation` annotations as web-search citations arrive                    |
| `response.web_search_call.in_progress` / `.searching` / `.completed`  |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Bracket each `web_search_call` item, in that order                           |
| `response.completed`                                                  |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Final event when generation finishes normally                                |
| `response.incomplete`                                                 |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Final event when output is truncated or filtered (no `response.completed`)   |
| `response.failed`                                                     |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Final event when generation fails; the response carries `error`              |
| `error`                                                               |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Spec error event on mid-stream failures, followed by `response.failed`       |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep role="img" aria-label="Model-dependent" } **Model-Dependent** — Behavior depends on the model or backend; check the Notes column
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation

</div>

!!! note "Bedrock Mantle passthrough"
    On [Mantle](features.md#bedrock-mantle-models) models served natively by the upstream Responses API, the parameters that the Converse path accepts but ignores — `background`, `include` (values other than `reasoning.encrypted_content`), `stream_options`, `reasoning.summary`, `text.verbosity`, `client_metadata`, `top_logprobs` — the hosted tools (`file_search`, `code_interpreter`, `computer`, `mcp`, `image_generation`) and the `web_search` search options (`filters`, `search_context_size`, `user_location`) are forwarded verbatim upstream: the upstream API decides whether they take effect or return a clean error. Web access is the exception: on every Mantle model the server decides whether a search may reach the external web, and unless the deployment allows the override, a request asking for a different value is rejected with a `400` — see [OpenAI GPT web search](#openai-gpt-web-search), where the tool is served natively.

## Model Support

All models supported by the Amazon Bedrock Converse and Converse Stream API are supported, plus every model served by [Bedrock Mantle](features.md#bedrock-mantle-models) when enabled — including OpenAI GPT-5.x, xAI Grok, and Google Gemma 4. Requests to Mantle models are passed through natively or converted automatically depending on the model's upstream API support.

!!! note "Project attribution (`OpenAI-Project`)"
    Mantle requests can be attributed to a Bedrock Project for cost tracking and observability with the `OpenAI-Project: <project-id>` header (a bare project ID such as `proj_abc123`, not an ARN). It is honored per-request only when [`AWS_BEDROCK_ALLOW_MANTLE_PROJECT_OVERRIDE`](operations_configuration.md#bedrock-allow-mantle-project-override) is `true`; otherwise the server default ([`AWS_BEDROCK_MANTLE_PROJECT`](operations_configuration.md#bedrock-mantle-project)) applies. This applies **only** to models served by the Bedrock Mantle endpoint — classic `bedrock-runtime` models ignore the header.

## Advanced Features

### System Prompt (`instructions`)

Use `instructions` to define the assistant's behavior — it is injected as a Bedrock system block.

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "instructions": "You are a helpful assistant that answers in French.",
    "input": "Say hello."
  }'
```

### Function Tool Calling

Define function tools and submit results in a round-trip conversation.

!!! note "Multi-Turn Conversations"
    For multi-turn conversations, pass the full message history in the `input` array, or store a response with `store=true` and continue it via `previous_response_id` (see [Stored Responses](#stored-responses)).

!!! warning "Unsupported Built-In Tools"
    `computer`, `computer_use_preview`, `mcp`, `local_shell`, `shell`, `custom`,
    `namespace`, `tool_search`, and `apply_patch` tools have no backend
    equivalent: they are **accepted for compatibility and dropped** from the tool
    configuration, so the model cannot call them.

    `file_search` is the exception: it is **served** from the vector stores the
    request names — see [File Search](#file-search).

!!! warning "Programmatic Tool Calling"
    The `programmatic_tool_calling` tool — and `tool_choice: {"type": "programmatic_tool_calling"}` —
    has no Bedrock Converse equivalent: the tool is **accepted and dropped** from
    the tool configuration, and the `tool_choice` degrades to the model's default
    choice. The request still succeeds and the model calls the declared tools
    **directly**, one round trip at a time, instead of orchestrating them from
    generated code, so no `program` or `program_output` items are returned.
    Tools restricted to `allowed_callers: ["programmatic"]` remain exposed as
    regular directly-callable tools. Bedrock Mantle native models receive the
    parameters unchanged and serve programmatic tool calling themselves when the
    model supports it.

    Echoed `program` and `program_output` history items are also accepted (and
    dropped on replay) so conversations recorded against the real API can be
    replayed; the paired `function_call` items carry the actual tool traffic.

**Step 1 — Request a tool call:**

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "input": "What'\''s the weather in Paris?",
    "tool_choice": "required",
    "tools": [
      {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    ]
  }'
```

**Step 2 — Submit the tool result:**

Without `previous_response_id`, each request is stateless: replay the full
history, including the `function_call` item from step 1's output alongside
its matching `function_call_output`.

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "input": [
      {
        "role": "user",
        "content": "What'\''s the weather in Paris?"
      },
      {
        "type": "function_call",
        "call_id": "<call_id from step 1>",
        "name": "get_weather",
        "arguments": "{\"city\": \"Paris\"}"
      },
      {
        "type": "function_call_output",
        "call_id": "<call_id from step 1>",
        "output": "{\"temperature\": \"18°C\", \"condition\": \"cloudy\"}"
      }
    ],
    "tools": [
      {
        "type": "function",
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    ]
  }'
```

### Streaming

Real-time token streaming with granular SSE lifecycle events.

```bash
curl -N -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "input": "Tell me a short story.",
    "stream": true
  }'
```

The stream emits events in order: `response.created` → `response.in_progress` → `response.output_item.added` → `response.content_part.added` → `response.output_text.delta` (repeated) → `response.output_text.done` → `response.content_part.done` → `response.output_item.done` → a terminal event.

The terminal event matches the outcome, exactly like the OpenAI API:

- `response.completed` — generation finished normally.
- `response.incomplete` — output was cut short (e.g. `max_output_tokens` reached or content filtered); `response.completed` is **not** emitted, and the SDK's `get_final_response()` raises accordingly.
- `response.failed` — generation failed; the embedded response carries an `error` object.

If an error occurs mid-stream, a spec `error` event (with `code`, `message`, `param`, and `sequence_number`) is emitted, followed by a terminal `response.failed` snapshot.

!!! note "Non-streaming failures"
    Without `stream: true` there is no terminal event to carry the failure, so a
    generation that ends in `status: "failed"` is reported as an HTTP `502` whose
    `error.message` is the failure reason — never a `200` with an empty `output`.
    Only `background: true` requests keep the `200` with the `failed` response
    body, so the client can poll the stored response.

### Structured JSON Output

Request machine-readable output using `text.format`.

**JSON object:**

`json_object` is not schema-validated: the gateway asks the model to reply with a
JSON object on your behalf, but nothing constrains its decoding, so this is a
best-effort guarantee rather than a hard one. Including the word "JSON" in the
input, as below, still helps the model comply. Use `json_schema` when the
response must conform to a specific shape.

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "input": "Return the current date and day of week as JSON.",
    "text": {"format": {"type": "json_object"}}
  }'
```

**JSON schema:**

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "input": "What is 2 + 2? Reply with answer and confidence.",
    "text": {
      "format": {
        "type": "json_schema",
        "name": "MathResult",
        "schema": {
          "type": "object",
          "properties": {
            "answer": {"type": "number"},
            "confidence": {"type": "number"}
          },
          "required": ["answer", "confidence"]
        }
      }
    }
  }'
```

### Extended Reasoning

Enable chain-of-thought reasoning on supported models (e.g. Amazon Nova 2, Anthropic Claude 3.7+) via `reasoning.effort`.

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai.gpt-5.6-sol",
    "input": "Solve: if a train travels 120 km in 90 minutes, what is its speed?",
    "reasoning": {"effort": "low"},
    "max_output_tokens": 4096
  }'
```

The chain of thought is returned as a `reasoning` output item preceding the assistant message, with the text in `content` parts of type `reasoning_text`. Streaming shape and token accounting depend on the serving path:

- **Converse-served models**: the item streams through `response.output_item.added`, `response.reasoning_text.delta` / `.done`, and `response.output_item.done` events before the message events. Bedrock does not split reasoning tokens out of `outputTokens`, so `usage.output_tokens_details.reasoning_tokens` is always `0`; reasoning tokens are still billed inside `output_tokens`.
- **Mantle-native models**: the reasoning is returned as an encrypted-content item with no plaintext `reasoning_text.delta` events — only `response.output_item.added` / `.done` bracket it. The reasoning-token split reported in `usage` is whatever the upstream API returns.
- **Mantle models converted to another API**: the chain of thought comes back like the Converse path — `content` parts of type `reasoning_text`, opened with `response.content_part.added`, streamed through `response.reasoning_text.delta` / `.done`, and closed with `response.content_part.done` — not as `summary` events.

Add `"include": ["reasoning.encrypted_content"]` to attach an `encrypted_content` envelope to each reasoning item. Echo the item back in the `input` of the next request to carry the model's reasoning state (including signatures and redacted content) across turns with no server-side storage — reasoning items from the official OpenAI API are accepted too, with their encrypted content safely ignored. Echoing a reasoning item back **without** its `encrypted_content` replays it unsigned: Anthropic Claude models only continue from a thinking passage they can recognise as their own, so that reasoning is left out of the turn — the request still succeeds, only the earlier chain of thought is no longer visible to the model. Every other model family receives the unsigned reasoning as-is.

### Prompt Caching

!!! warning "Cache Creation Costs"
    Cache creation incurs a higher cost than regular token processing. Only use prompt caching when you expect a high cache hit ratio across multiple requests with similar prompts.

Prompt caching reduces latency and costs by caching repetitive prompt components. Set the `prompt_cache_key` parameter to enable:

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai.gpt-5.6-sol",
    "instructions": "You are a helpful assistant.",
    "input": "What is Python?",
    "prompt_cache_key": "default"
  }'
```

**Granular Cache Control:**

Use dot-separated values to cache specific components:

- `"system"` — Cache system messages only
- `"messages"` — Cache conversation history
- `"tools"` — Cache tool/function definitions (Anthropic Claude only)
- `"system.messages"` — Cache both system and messages
- `"system.tools"` — Cache system and tools
- `"messages.tools"` — Cache messages and tools
- `"system.messages.tools"` — Cache all components
- Any other non-empty value — Cache all components

!!! note "Custom Cache Keys Not Supported"
    Custom cache hash keys are not supported. The parameter is used only to control which sections are cached, not as a cache identifier.

**Example — Cache system and tools:**

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-5",
    "instructions": "You are a data analysis assistant.",
    "input": "Analyze this dataset: ...",
    "tools": [{"type": "function", "name": "run_sql", ...}],
    "prompt_cache_key": "system.tools"
  }'
```

**Benefits:**

- **Cost Reduction**: Cached tokens are billed at a lower rate than regular input tokens
- **Lower Latency**: Cached prompts eliminate reprocessing time
- **Automatic Management**: The API handles cache invalidation and updates

**Cache Retention (TTL):**

Control how long cached prompts persist using the `prompt_cache_retention` parameter:

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai.gpt-5.6-sol",
    "input": "Hello",
    "prompt_cache_key": "default",
    "prompt_cache_retention": "24h"
  }'
```

Valid values: `in_memory` (default), `24h`, `1h`, or `5m`. The `1h` and `5m` values are Amazon Bedrock-specific. On Amazon Bedrock, `in_memory` maps to 5 minutes and `24h` maps to 1 hour. The `prompt_cache_options.ttl` value `"30m"` is mapped to a 1 hour retention on Anthropic models (other models use the default 5 minute TTL) when `prompt_cache_retention` is unset.

**Explicit Cache Breakpoints:**

Instead of relying on the `prompt_cache_key` section heuristics, mark the exact cache boundaries with `prompt_cache_breakpoint` on any input content part (`input_text`, `input_image`, `input_file`). Each marked part is followed by an Amazon Bedrock `cachePoint`, so the prompt prefix ending with that part is cached:

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-5",
    "prompt_cache_options": {"mode": "explicit"},
    "input": [
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": "Long reusable context...",
            "prompt_cache_breakpoint": {"mode": "explicit"}
          },
          {"type": "input_text", "text": "Summarize it."}
        ]
      }
    ]
  }'
```

- `"mode": "explicit"` caches **only** the marked parts: the `prompt_cache_key` heuristics are disabled for that request.
- `"mode": "implicit"` (default) keeps the `prompt_cache_key` heuristics **and** honors the marked parts.
- At most 4 cache points are sent per request (Amazon Bedrock limit); the oldest ones are dropped when more are requested.
- Breakpoints on models without prompt caching support are accepted and ignored, as are breakpoints on tool output items — those never become a cache point, whatever the model.
- `prompt_cache_options` is echoed back on the response object.

!!! note "Model Support"
    Cache retention configuration is only available on select models. See [Amazon Bedrock Prompt Caching - Supported Models](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html#prompt-caching-models) for details on which models support configurable TTL.

Cached token usage is reported in the response:

```json
{
  "usage": {
    "input_tokens": 1500,
    "input_tokens_details": {
      "cached_tokens": 1200
    },
    "output_tokens": 300,
    "total_tokens": 1800
  }
}
```

Following OpenAI semantics, `input_tokens` covers the **full** prompt: tokens read from and written to the cache are included, and `cached_tokens` (the tokens read from cache) is a subset of `input_tokens`. In this example, 1,200 of the 1,500 input tokens were retrieved from cache.

### Provider-Specific Parameters

A top-level field this API does not declare is forwarded to the model as a
provider-specific inference parameter, as on
[Chat Completions](api_openai_chat_completions.md#provider-specific-parameters)
and [Messages](api_anthropic_messages.md#provider-specific-parameters). A
capability Amazon Bedrock exposes and the OpenAI API has no field for is
therefore reachable without leaving this endpoint — the OpenAI SDK sends these
through `extra_body`:

```json
{
  "model": "anthropic.claude-sonnet-5",
  "input": "Write a poem about the sea",
  "top_k": 50
}
```

Server-wide defaults per model come from `DEFAULT_MODEL_PARAMS`, and a
per-request value wins over them.

**Behavior:**

- :material-check-circle:{ .success role="img" aria-label="Supported" } **Compatible parameters**: forwarded to the model and applied
- :material-alert-circle:{ .warning } **Unsupported parameters**: the backend refuses the request, returned as a `400`
- :material-alert-circle:{ .warning } **Reserved names**: `additional_request_fields`, `max_tokens`, `model_id`, `stop_sequences`, `temperature`, `top_logprobs` and `top_p` are the argument names the gateway binds when it builds the Bedrock call, so sending one as an extra is rejected with a `400` naming it instead of binding twice — use the declared `max_output_tokens`, `temperature`, `top_p` and `top_logprobs` fields
- :material-alert-circle:{ .warning } **Client-side control fields**: names no provider treats as inference parameters (LiteLLM's `drop_params` among them) are dropped before the call. [`EXTRA_MODEL_PARAMS_DENYLIST`](operations_configuration.md#extra-model-params-denylist) extends that list, and [`EXTRA_MODEL_PARAMS_DROP_ALL`](operations_configuration.md#extra-model-params-drop-all) disables the passthrough entirely

### Managed Prompt Templates { #managed-prompt-templates }

The `prompt` parameter references a prompt template stored in [Amazon Bedrock Prompt Management](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-management.html). Amazon Bedrock renders the template server-side, so the request body carries only the variable values.

!!! warning "Disabled by Default"
    `prompt` returns `400` unless the server operator sets [`AWS_BEDROCK_ALLOW_PROMPT_ARN`](operations_configuration.md#bedrock-allow-prompt-arn) to `true`.

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-5",
    "prompt": {
      "id": "arn:aws:bedrock:us-east-1:123456789012:prompt/ABCDE12345",
      "version": "1",
      "variables": {"genre": "pop", "number": "3"}
    }
  }'
```

:octicons-key-24: **Requirements**

- `prompt.id` must be an Amazon Bedrock prompt ARN. OpenAI-hosted prompt template IDs (`pmpt_…`) do not exist on this gateway and return `400`.
- `prompt.version` is an Amazon Bedrock version number. It is appended to the ARN, and must not disagree with a version already present in `prompt.id`. Omit it to run the working draft.
- The prompt must be a **TEXT** prompt bound to a model that this server can serve, and `model` must be that exact model: it is the model used for response formatting and cost attribution. The error message names the model the prompt uses.
- `prompt.variables` values must be plain strings — Amazon Bedrock prompt variables only carry text, so image, file and structured content parts return `400`.
- The prompt's region is derived from its ARN and must be a configured Amazon Bedrock region; cross-region failover is disabled for the request.
- The model must be served by the Converse API: [Mantle](features.md#bedrock-mantle-models) native models return `400`, as they have no Prompt Management equivalent.

:octicons-x-circle-24: **Rejected Alongside `prompt`**

The stored prompt version already provides the conversation, the system prompt, the tools and the inference parameters, so combining `prompt` with `input`, `instructions`, `tools`, `tool_choice`, `text`, `temperature`, `top_p`, `max_output_tokens`, `reasoning` or `previous_response_id` returns `400` instead of silently dropping them.

Streaming, `store`, `moderation` and guardrail headers remain available. The remaining request-level parameters (`metadata`, `service_tier`, `prompt_cache_*`, …) are accepted and echoed on the response, but not applied: the Bedrock call carries only the prompt resource and its variables.

### File Search { #file-search }

`file_search` lets any chat model answer from the files you indexed in a
[vector store](api_openai_vector_stores.md), with no change to your client code:

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-2-lite-v1:0",
    "input": "How many vacation days do I get?",
    "tools": [{
      "type": "file_search",
      "vector_store_ids": ["vs_abc123"]
    }]
  }'
```

The model decides when to search and with which query. Each search it runs is
reported as a `file_search_call` output item — carrying the `queries` used —
ahead of the `message` it grounded. When streaming, that item is framed by
`response.file_search_call.in_progress`, `.searching` and `.completed`, and the
answer streams after it.

The retrieved passages are **not** returned unless you ask for them:

```json
{
  "model": "amazon.nova-2-lite-v1:0",
  "input": "How many vacation days do I get?",
  "include": ["file_search_call.results"],
  "tools": [{"type": "file_search", "vector_store_ids": ["vs_abc123"]}]
}
```

Each result then carries its `file_id`, `filename`, `text`, `score` and the
`attributes` stored with the file.

!!! note "Citations"
    The grounded answer carries a `file_citation` annotation on its
    `output_text` content for every file the passages were read from, each
    naming the `file_id` and `filename`, so a client can attribute the answer
    without asking for the passages themselves.

    A model reports the passages it was given, never which sentence came from
    which one, so there is one citation per file rather than one per passage,
    and `index` is the end of the answer text rather than the position of a
    cited span. When streaming, they arrive on the message item itself
    (`response.output_item.done` and the terminal event) rather than as
    separate `response.output_text.annotation.added` events.

**Narrowing the search:**

| Field | Behavior |
|---|---|
| `vector_store_ids` | Every store listed is searched and the best passages across all of them are kept. At least one is required, and a store this deployment does not serve answers `404` before the model is called |
| `max_num_results` | Passages kept per search, `1`–`50`; defaults to `20` |
| `filters` | Restricts the search to files carrying given `attributes`. A comparison operator the store cannot apply is refused with a `400` naming the ones it accepts |
| `ranking_options.score_threshold` | Drops passages below the score. Refused with a `400` on a store whose relevance scores are not comparable between searches |
| `ranking_options.ranker`, `ranking_options.hybrid_search` | Accepted and ignored — the passages are still ranked by relevance |

!!! note "Rounds per response"
    The model may refine its query and search again; after two searches the
    tool is withdrawn and the model answers with what it has, so one response
    never loops indefinitely. Each round is a further model invocation and is
    billed as such.

### OpenAI Integrated Tools

The Responses API supports OpenAI's built-in tool types, automatically mapped to the target model's native tools.

#### ![Amazon Nova](styles/logo_amazon_nova.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Nova Tools

Nova models support web search and code execution as integrated tools.

**Web Search** (`web_search`, `web_search_preview`):

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-2-lite-v1:0",
    "input": "What is the current version of Python?",
    "tools": [{"type": "web_search"}]
  }'
```

**Code Interpreter** (`code_interpreter`):

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-2-lite-v1:0",
    "input": "Calculate the first 10 Fibonacci numbers",
    "tools": [{"type": "code_interpreter"}]
  }'
```

!!! note "Citations: sources and annotations"
    Web-search citations surface in two places:

    - `action.sources` on the `web_search_call` item (in streaming mode it is
      `null` on intermediate events and populated on the terminal event, since
      citations arrive after the tool call closes).
    - `url_citation` annotations on the assistant message's `output_text`
      content. When streaming, each citation also emits a
      `response.output_text.annotation.added` event as it arrives.

    On Nova models, Amazon Bedrock does not report character positions, so
    `start_index` and `end_index` are approximated to the length of the
    generated text at the time the citation arrived. The OpenAI GPT-5.x family
    reports its own spans — see
    [OpenAI GPT Web Search](#openai-gpt-web-search).

!!! warning "Search options"
    `filters.allowed_domains` and `user_location` restrict which sources a
    search may use, and Nova's grounding cannot apply either: a request
    carrying one is **rejected with a `400`** rather than searched
    unrestricted. `search_context_size` is accepted and ignored — the answer is
    still searched and cited. All three are honored on the models that serve
    web search natively, listed below.

    The `external_web_access` extra model parameter is refused the same way: a
    Nova search runs with the web access the server is configured for, so a
    request asking for a different one is **rejected with a `400`** instead of
    being searched under the server's value. Sending the configured value, or
    omitting the parameter, always works.

!!! warning "Region Compatibility"
    `web_search` is available on Amazon Nova 2 and Nova Premier models, in US regions only. Not available on EU inference profiles.

#### ![OpenAI](styles/logo_openai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } OpenAI GPT Web Search

The OpenAI GPT-5.x family answers `web_search` with the search tool built into
Amazon Bedrock. The model decides when a question needs current information,
runs one or more queries, and grounds its answer in what it finds.

!!! warning "Amazon Bedrock Mantle only"
    Amazon Bedrock serves this tool on the Mantle endpoint alone; it is refused
    on the `bedrock-runtime` endpoint. Models offered on both — the GPT-5.6
    family among them — resolve to their runtime twin by default, which cannot
    answer `web_search`: the request is **rejected with a `400`** naming both
    ways to reach Mantle, rather than answered without a search. Send it to
    Mantle explicitly, with the `x-stdapi-service` header below or by naming the
    model in
    [`AWS_BEDROCK_MANTLE_PREFERRED_MODELS`](operations_configuration.md#bedrock-mantle-preferred-models).
    Available in `us-east-1`, `us-east-2` and `us-west-2`, and billed per query.

    `code_interpreter` is refused the same way and for the same reason: no
    OpenAI GPT server tool is served on `bedrock-runtime`.

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -H "x-stdapi-service: bedrock-mantle" \
  -d '{
    "model": "openai.gpt-5.6-luna",
    "input": "What are the most significant AWS launches announced this month?",
    "tools": [{"type": "web_search"}]
  }'
```

Each search appears as a `web_search_call` output item, and every grounded
statement carries a `url_citation` annotation on the assistant message's
`output_text` content, whose `start_index` and `end_index` delimit the answer
text it supports. When streaming, the lifecycle arrives as
`response.web_search_call.in_progress` / `.searching` / `.completed`, and each
citation as a `response.output_text.annotation.added` event.

!!! info "External web access"
    Searches are answered from the Amazon Bedrock web index and cache, and
    results are current and cited either way.
    [`AWS_BEDROCK_EXTERNAL_WEB_ACCESS`](operations_configuration.md#bedrock-external-web-access)
    controls whether a search *may* reach the external web, and it takes the
    `bedrock-websearch:ExternalWebAccess` permission as well — see
    [Web Search IAM](operations_iam_permissions.md#web-search-iam).

    AWS
    [documents](https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html)
    that retrieval is served entirely from that index and cache today, so no
    request data leaves the AWS boundary even when the permission is granted, and
    that a future release may allow live external retrieval — at which point
    request data may leave it. Enabling this is therefore an advance decision
    about behaviour that can change: leave it off unless you intend that.

    A request may choose its own web access by sending `external_web_access` as
    an extra model parameter (a top-level field, or `extra_body` in the OpenAI
    SDK), and only when
    [`AWS_BEDROCK_ALLOW_EXTERNAL_WEB_ACCESS_OVERRIDE`](operations_configuration.md#bedrock-allow-external-web-access-override)
    is enabled; otherwise a value differing from the configured one is rejected
    with a `400`. These are the models whose search takes the choice per
    request: everywhere else the parameter must match the server's value.

    ```json
    {
      "model": "openai.gpt-5.6-luna",
      "input": "What shipped this week?",
      "tools": [{"type": "web_search"}],
      "external_web_access": true
    }
    ```

!!! warning "Availability and cost"
    Web search is available on the OpenAI GPT-5.x models in `us-east-1`,
    `us-east-2` and `us-west-2`. Each search runs in the Region that served the
    model call and is never routed to another Region, so keep one of the three
    in [`AWS_BEDROCK_REGIONS`](operations_configuration.md#aws-bedrock-regions):
    a model served from anywhere else cannot search.

    It is billed per query on top of the model's
    tokens — see [Built-in tool pricing](operations_cost_management.md#built-in-tool-pricing).
    It is offered on this endpoint only: requesting the equivalent server tool
    on `/v1/messages` returns a `400`.

#### :material-image: Image Generation

The `image_generation` integrated tool works with **all text models** — Claude, Nova, and any future model. The gateway intercepts the tool, lets the LLM compose the image prompt and parameters via a synthetic function call, then generates the image against a configured Bedrock image model and returns an `image_generation_call` output item to the client. Intermediate `function_call` items are suppressed.

!!! warning "Configuration Required"
    Set the [`IMAGE_GENERATION_MODEL`](operations_configuration.md#image-generation-model) environment variable to a Bedrock image model ID (e.g. `amazon.nova-canvas-v1:0`). The tool definition may also specify a `model` field to override the default per request.

**Example — Generate an image:**

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "input": "Generate a photorealistic image of a red panda sitting on a tree branch.",
    "tools": [{"type": "image_generation"}],
    "tool_choice": "required"
  }'
```

The response contains an `image_generation_call` output item:

```json
{
  "output": [
    {
      "type": "image_generation_call",
      "id": "img_abc123",
      "status": "completed",
      "result": "<base64-encoded PNG>"
    }
  ]
}
```

You can also specify image parameters in the tool definition:

```json
{
  "type": "image_generation",
  "size": "1024x1024",
  "quality": "high",
  "output_format": "png"
}
```

!!! info "`quality` is set by you, not by the model"
    The model chooses the prompt, size and output format for each call, but `quality` is read from the tool definition only. Most image models have no quality control and reject the parameter, so a value the model volunteered would fail the generation.

!!! info "`partial_images` is accepted and ignored"
    `partial_images` (0-3) is accepted for OpenAI API compatibility but never acts: no available model streams partial images, so no `response.image_generation_call.partial_image` event is emitted and the finished image always arrives in a single `response.image_generation_call.completed` event. The same applies to [image generations](api_openai_images_generations.md) and [image edits](api_openai_images_edits.md).

#### Computer Use Not Supported

!!! warning "Computer Use Not Supported"
    The `computer` and `computer_use_preview` integrated tools are **not supported**: requests succeed, but the tools are accepted and dropped from the tool configuration, so the model can never call them.

## Available Request Headers

This endpoint supports standard Bedrock headers for enhanced control over your requests. All headers are optional and can be combined as needed.

### Content Safety (Guardrails)

| Header                               | Purpose                            | Valid Values                          |
|--------------------------------------|------------------------------------|---------------------------------------|
| `X-Amzn-Bedrock-GuardrailIdentifier` | Guardrail ID for content filtering | Your guardrail identifier             |
| `X-Amzn-Bedrock-GuardrailVersion`    | Guardrail version                  | Version number (e.g., `1`)            |
| `X-Amzn-Bedrock-Trace`               | Guardrail trace level              | `disabled`, `enabled`, `enabled_full` |

### Performance Optimization

| Header                                     | Purpose                | Valid Values                  |
|--------------------------------------------|------------------------|-------------------------------|
| `X-Amzn-Bedrock-Service-Tier`              | Service tier selection | `default`, `flex`, `priority`, `reserved` |
| `X-Amzn-Bedrock-PerformanceConfig-Latency` | Latency optimization   | `standard`, `optimized`       |

**Example with all headers:**

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Amzn-Bedrock-GuardrailIdentifier: your-guardrail-id" \
  -H "X-Amzn-Bedrock-GuardrailVersion: 1" \
  -H "X-Amzn-Bedrock-Trace: enabled" \
  -H "X-Amzn-Bedrock-Service-Tier: priority" \
  -H "X-Amzn-Bedrock-PerformanceConfig-Latency: optimized" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "input": "Hello!"
  }'
```

!!! info "Detailed Documentation"
    For complete information about these headers, configuration options, and use cases, see:

    - [Bedrock Guardrails Configuration](operations_configuration.md#bedrock-guardrails)
    - [Service Tier and Performance Configuration](operations_configuration.md#bedrock-service-tier-and-performance-configuration)

## Model-Specific Features

### ![TwelveLabs](styles/logo_twelvelabs.svg){ style="height: 1.2em; vertical-align: text-bottom;" } TwelveLabs Pegasus

`twelvelabs.pegasus-1-2-v1:0` is a video-understanding model. Because Pegasus accepts exactly one video and one text prompt per call, this API adapts the conversation automatically:

- The **latest video** found anywhere in the conversation (any role, any position) is forwarded as the video input.
- The **latest contiguous run of user text** (back to the previous assistant or tool turn) is concatenated and forwarded as the text prompt.
- `temperature` and `max_output_tokens` are forwarded.
- `text.format: json_schema` is forwarded as Pegasus's structured output.

**Silently ignored** (no error): system prompt, tools, `top_p`, stop sequences, and prompt caching.

**Upstream format limitation:** The OpenAI Responses API has no `input_video` content type in its stable spec. To stay fully compatible with standard OpenAI clients, pass the video as an **`input_image`** content item — the server detects the video MIME type automatically and routes it to Pegasus correctly.

**Video input formats**: `data:video/mp4;base64,…`, `https://…`, `s3://bucket/key`, or `file-id:…`. Videos above 18.75 MB are automatically uploaded to S3.

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "twelvelabs.pegasus-1-2-v1:0",
    "input": [
      {
        "type": "message",
        "role": "user",
        "content": [
          {"type": "input_image", "image_url": "s3://my-bucket/video.mp4"},
          {"type": "input_text", "text": "Describe what happens in this video."}
        ]
      }
    ]
  }'
```

## Try It Now

**Basic response:**

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "input": "Say hello world"
  }'
```

**Streaming response:**

```bash
curl -N -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "input": "Write a haiku about the sea.",
    "stream": true
  }'
```

**Multi-modal with image:**

```json
{
  "model": "amazon.nova-micro-v1:0",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Describe this image"},
        {"type": "input_image", "image_url": "https://example.com/photo.jpg"}
      ]
    }
  ]
}
```

**With reasoning:**

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai.gpt-5.6-sol",
    "input": "Solve 12 × 13",
    "reasoning": {"effort": "low"},
    "max_output_tokens": 4096
  }'
```

## Input Token Counting

Count input tokens without generating a response. Useful for estimating costs or checking context-window fit before making a full response call.

**Basic usage:**

```bash
curl -X POST "$BASE/v1/responses/input_tokens" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "input": "Hello, how are you?"
  }'
```

**Response:**

```json
{
  "object": "response.input_tokens",
  "input_tokens": 142
}
```

**With instructions and tools:**

```bash
curl -X POST "$BASE/v1/responses/input_tokens" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "input": "What is the weather?",
    "instructions": "You are a helpful assistant.",
    "tools": [{"type": "function", "name": "get_weather", "description": "Get weather for a location", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}]
  }'
```

!!! note "Limitations"
    The `previous_response_id` and `conversation` parameters are not supported for token counting (they would change the count); `personality` (a token-counting-only schema field) and `reasoning.context` are accepted and ignored. Token counting is not available for models served by [Amazon Bedrock Mantle](features.md#bedrock-mantle-models) (the request is rejected with a `400` error).

## Stored Responses

Set `store: true` to persist a response in [Amazon Bedrock session storage](https://docs.aws.amazon.com/bedrock/latest/userguide/sessions.html): one AWS-managed session per stored response, encrypted at rest (optionally with [your own KMS key](operations_configuration.md#aws-bedrock-session-encryption-key-arn)), with no state on the server itself.

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "amazon.nova-micro-v1:0", "input": "Hello!", "store": true}'
```

The returned `id` then works with:

- `GET /v1/responses/{response_id}` — retrieve the stored response.
- `GET /v1/responses/{response_id}/input_items` — list the input items that produced it. Bedrock Mantle native storage does not serve input item listings: for Mantle-stored responses this returns `404` with an explanatory message.
- `DELETE /v1/responses/{response_id}` — delete it (and its Amazon Bedrock session).
- `POST /v1/responses/{response_id}/cancel` — for Mantle region-tagged IDs, proxied to Bedrock Mantle (background responses are cancellable upstream); for Bedrock-session-stored responses it fails with the OpenAI synchronous-response error since execution is synchronous.
- `previous_response_id` on a new request — continue the conversation: the stored input and output are automatically prepended to the new input (instructions are not carried over, per the OpenAI API).

!!! warning "Response IDs are stdapi.ai-specific"
    Response IDs embed the serving AWS region, so conversation turns chained with `previous_response_id` stay region-local. These IDs **cannot** be used directly against the Bedrock Mantle API, and raw [Mantle](features.md#bedrock-mantle-models) response IDs are not accepted by stdapi.ai.

!!! note "Behavior notes"
    - `store` defaults to **false** on this implementation (the OpenAI API defaults to true).
    - `POST /v1/responses/input_tokens` and `POST /v1/responses/compact` cannot reference a Mantle-stored response via `previous_response_id` — like input-item listings, Mantle native storage does not serve the stored items back.
    - On Amazon Bedrock session storage, `store=true` is ignored with `stream=true` (a warning is recorded in the request log). [Mantle](features.md#bedrock-mantle-models) models persist responses in Mantle native storage instead, where `store` works with streaming too.
    - Mantle models without native Responses storage (Messages- or Chat-Completions-bound) use Amazon Bedrock session storage like classic models. Only a `store=true` request answered through a mid-request API fallback (away from the upstream Responses API) is served without storage, with a warning recorded in the request log; its ID cannot be retrieved later. `previous_response_id` on such a fallback returns `400` instead — conversation history is never silently dropped.
    - Sessions are created in the primary Bedrock region and persist until deleted through the API — see [operator guidance on cleaning up stale sessions](operations_configuration.md#bedrock-session-storage-optional).
    - `GET /v1/responses/{response_id}` rejects `stream=true` with `400`; `include` and `starting_after` are accepted and ignored.
    - Amazon Bedrock session storage is offered in fewer regions than model inference. Where the primary Bedrock region does not provide it, `store=true` is ignored and a warning naming the region as the cause is recorded in the request log — the response itself is still returned. Configure a primary region that provides session storage to avoid this entirely.
    - Requires the Amazon Bedrock session management IAM permissions (`bedrock:CreateSession`, `bedrock:CreateInvocation`, `bedrock:PutInvocationStep`, `bedrock:GetInvocationStep`, `bedrock:ListInvocationSteps`, `bedrock:ListInvocations`, `bedrock:ListSessions`, `bedrock:ListTagsForResource`, `bedrock:EndSession`, `bedrock:DeleteSession`, `bedrock:TagResource`). Without them, `store=true` is ignored (with a request-log warning) and the response is not persisted.

## Conversation Compaction

Compact a long conversation into a single `compaction` item to keep multi-turn sessions within the context window. The model summarizes the provided `input`; the summary comes back as an opaque item that you include in the `input` of later requests instead of the full history.

```bash
curl -X POST "$BASE/v1/responses/compact" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-pro-v1:0",
    "input": [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."}
    ]
  }'
```

**Response** (trimmed to the compaction item; `output` also echoes the conversation's message items before it):

```json
{
  "id": "resp-...",
  "object": "response.compaction",
  "created_at": 1752000000,
  "output": [
    {"id": "ci-...", "type": "compaction", "encrypted_content": "..."}
  ],
  "usage": {"input_tokens": 1500, "output_tokens": 220, "total_tokens": 1720}
}
```

Continue the conversation by sending the compaction item back, followed by new messages:

```json
{
  "model": "amazon.nova-pro-v1:0",
  "input": [
    {"id": "ci-...", "type": "compaction", "encrypted_content": "..."},
    {"role": "user", "content": "Next question..."}
  ]
}
```

!!! note "Stateless compaction"
    The compaction content is fully self-contained (marker-prefixed and encoded, not encrypted): no conversation state is needed, and any server instance can expand it. Only compaction items produced by this server can be expanded — items encrypted by the upstream OpenAI API are rejected with `400`, and locally-produced items cannot be continued on a [Mantle](features.md#bedrock-mantle-models)-served model. `previous_response_id` may reference a [stored response](#stored-responses) to include its conversation in the compaction.

---

**Ready to build with AI?** Check out the [Models API](api_openai_models.md) to see all available foundation models!
