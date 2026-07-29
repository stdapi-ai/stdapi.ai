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
| Hosted-tool call items (`web_search_call`, `file_search_call`, `code_interpreter_call`, `computer_call`, `tool_search_call`, shell/apply-patch/MCP items, `compaction_trigger`) | :material-check-circle:{ .success role="img" aria-label="Supported" } | Input-history tolerance only: echoed items are accepted and dropped on replay (no Bedrock equivalent). Whether each *tool* can actually be used is listed under Tool Calling below |
| `item_reference`                                                      |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Accepted and dropped on replay                                               |
| **Tool Calling**                                                      |                                         |                                                                              |
| Function tools (`type: "function"`)                                   |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Full schema mapping to Bedrock toolSpec                                      |
| `tool_choice: "auto"`                                                 |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Model selects among available tools                                          |
| `tool_choice: "required"`                                             |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Model must call at least one tool                                            |
| `tool_choice: "none"`                                                 |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Prevents tool calls                                                          |
| Named `tool_choice` (force)                                           |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Force a specific function to be called                                       |
| `tool_choice: allowed_tools`                                          |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Approximated: `required` + 1 function → forced tool; `required` + many → any tool; `auto` → auto; type-variants add no constraint |
| `parallel_tool_calls`                                                 |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Echoed in response; not transmitted to Bedrock                               |
| Built-in tools (`code_interpreter`, `web_search`, `image_generation`) |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | See [OpenAI Integrated Tools](#openai-integrated-tools)                      |
| `file_search` tool                                                    |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | No Converse equivalent — accepted and dropped; forwarded upstream on Bedrock Mantle native models |
| `computer` / `computer_use_preview` tools                             |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | No Converse equivalent — accepted and dropped (see [Computer Use Not Supported](#computer-use-not-supported)); forwarded upstream on Bedrock Mantle native models |
| `mcp` tool                                                            |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | No Converse equivalent — accepted and dropped; forwarded upstream on Bedrock Mantle native models |
| `local_shell` / `shell` tools                                         | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Accepted and dropped; no Bedrock equivalent                                  |
| `custom` / `namespace` / `tool_search` / `apply_patch` tools          | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Accepted and dropped; no Bedrock equivalent                                  |
| **Generation Control**                                                |                                         |                                                                              |
| `max_output_tokens`                                                   |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Maps to Bedrock `maxTokens`                                                  |
| `temperature`                                                         |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | 0–2 range; mapped to Bedrock inference config                                |
| `top_p`                                                               |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | 0–1 range; nucleus sampling                                                  |
| `top_logprobs`                                                        |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | 0–20 range accepted and echoed; log probabilities are never returned on Converse-served models; forwarded upstream on Bedrock Mantle native models |
| `reasoning` (effort)                                                  |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Configures reasoning; without `effort` defaults to `medium`; `effort: "none"` disables; chain of thought returned as `reasoning` output items |
| `reasoning.summary` / `generate_summary`                              |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Accepted but ignored on Converse-served models — no summary is generated; forwarded upstream on Bedrock Mantle native models |
| `reasoning.context`                                                   | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Accepted but ignored — context scoping is not applied                        |
| `reasoning.mode`                                                      | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Accepted but ignored — pro-mode reasoning selection is not applied           |
| `text.verbosity`                                                      |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Accepted but ignored on Converse-served models; forwarded upstream on Bedrock Mantle native models |
| `include`                                                             |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }   | `reasoning.encrypted_content` is honored; other values are accepted and ignored (forwarded upstream on Bedrock Mantle native models) |
| `metadata`                                                            |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Forwarded to Bedrock `requestMetadata`                                       |
| `prompt_cache_key`                                                    |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Cache prompts to reduce costs and latency                                    |
| `prompt_cache_options`                                                | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Accepted but ignored — caching is driven by `prompt_cache_key`/`prompt_cache_retention` instead |
| `prompt_cache_retention`                                              |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Cache TTL: `in_memory`, `24h`, `1h`, or `5m`                                 |
| `service_tier`                                                        |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Maps to Bedrock service tier header                                          |
| `truncation`                                                          | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Returns `400`; Bedrock manages context automatically                         |
| `max_tool_calls`                                                      | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Returns `400`; not supported                                                 |
| `context_management`                                                  | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Returns `400`; not supported                                                 |
| `background`                                                          |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Accepted but ignored on Converse-served models — execution is synchronous; forwarded upstream on Bedrock Mantle native models, where background responses can be cancelled — see [Stored Responses](#stored-responses) |
| `store`                                                               |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Persists the response — Amazon Bedrock session storage (non-streaming) or Mantle native storage for Mantle models (streaming supported) |
| `stream_options`                                                      |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Accepted but ignored on Converse-served models; forwarded upstream on Bedrock Mantle native models |
| `conversation`                                                        | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Returns `400`; use `previous_response_id` or `input`                         |
| `prompt` (template reference)                                         | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Returns `400`; not supported                                                 |
| `safety_identifier`                                                   | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Accepted but ignored by generation; recorded in request logs                 |
| `client_metadata`                                                     |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Accepted but ignored on Converse-served models (sent by newer OpenAI clients such as Codex); forwarded upstream on Bedrock Mantle native models |
| `moderation`                                                          |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Applies an Amazon Bedrock guardrail; results in the response `moderation` field (on the terminal event when streaming) — rejected (`400`) on Mantle-served models |
| **Output Format**                                                     |                                         |                                                                              |
| `text.format: "text"`                                                 |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Plain text output                                                            |
| `text.format: "json_object"`                                          |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | JSON object output via Bedrock outputConfig                                  |
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
    On [Mantle](features.md#bedrock-mantle-models) models served natively by the upstream Responses API, the parameters that the Converse path accepts but ignores — `background`, `include` (values other than `reasoning.encrypted_content`), `stream_options`, `reasoning.summary`, `text.verbosity`, `client_metadata`, `top_logprobs` — and the hosted tools (`file_search`, `code_interpreter`, `computer`, `mcp`, `image_generation`) are forwarded verbatim upstream: the upstream API decides whether they take effect or return a clean error. The `web_search` tool runs in cache-only mode (`external_web_access` is forced off).

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
    `file_search`, `computer`, `computer_use_preview`, `mcp`, `local_shell`, `shell`,
    `custom`, `namespace`, `tool_search`, and `apply_patch` tools have no backend
    equivalent: they are **accepted for compatibility and dropped** from the tool
    configuration, so the model cannot call them.

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

The chain of thought is returned as a `reasoning` output item preceding the assistant message, with the text in `content` parts of type `reasoning_text`. When streaming on Converse-served models, the item is delivered through `response.output_item.added`, `response.reasoning_text.delta` / `.done`, and `response.output_item.done` events before the message events. Mantle-native models instead return the reasoning as an encrypted-content item with no plaintext `reasoning_text.delta` events — only `response.output_item.added` / `.done` bracket it. Bedrock does not split reasoning tokens out of `outputTokens`, so `usage.output_tokens_details.reasoning_tokens` is always `0`; reasoning tokens are still billed inside `output_tokens`.

Add `"include": ["reasoning.encrypted_content"]` to attach an `encrypted_content` envelope to each reasoning item. Echo the item back in the `input` of the next request to carry the model's reasoning state (including signatures and redacted content) across turns with no server-side storage — reasoning items from the official OpenAI API are accepted too, with their encrypted content safely ignored. Echoing a reasoning item back **without** its `encrypted_content` replays it unsigned; Anthropic models may reject it in tool-use continuation turns.

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

Valid values: `in_memory` (default), `24h`, `1h`, or `5m`. The `1h` and `5m` values are Amazon Bedrock-specific. On Amazon Bedrock, `in_memory` maps to 5 minutes and `24h` maps to 1 hour.

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

    Amazon Bedrock does not report character positions, so `start_index` and
    `end_index` are approximated to the length of the generated text at the
    time the citation arrived.

!!! warning "Region Compatibility"
    `web_search` is available on Amazon Nova 2 and Nova Premier models, in US regions only. Not available on EU inference profiles.

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
| `X-Amzn-Bedrock-Service-Tier`              | Service tier selection | `priority`, `default`, `flex` |
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
