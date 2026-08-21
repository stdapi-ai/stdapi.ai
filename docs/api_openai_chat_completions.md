---
title: Chat Completions API - Amazon Bedrock with OpenAI Compatibility
description: OpenAI-compatible chat completions API for Amazon Bedrock models including Claude, Nova, Llama. Supports streaming, reasoning modes, prompt caching, and multi-modal inputs.
keywords: chat completions API, OpenAI chat API, Amazon Bedrock chat, streaming chat API, AI chatbot API, Claude API, function calling API, multi-modal chat
---

# Chat Completions API

Generate conversational AI responses with Amazon Bedrock foundation models—including Claude, Nova, Llama, and more—through an OpenAI-compatible interface.

## Why Choose the Chat Completions API?

<div class="grid cards" markdown>

- :material-brain: __Multiple Models__
  <br>Access models from Anthropic, Amazon, Meta, and more through one API. Choose the best model for your task without vendor lock-in.

- :material-image-multiple: __Multi-Modal__
  <br>Process text, images, videos, and documents together. Support for URLs, data URIs, and direct S3 references.

- :material-shield-check: __Built-In Safety__
  <br>Amazon Bedrock Guardrails provide content filtering and safety policies.

- :material-aws: __AWS Scale & Reliability__
  <br>Run on AWS infrastructure with service tiers for optimized latency. Multi-region model access for availability and performance.

</div>

## Available Endpoints

| Endpoint               | Method | What It Does                               | Powered By               | MCP Tool                  |
|------------------------|--------|--------------------------------------------|--------------------------|---------------------------|
| `/v1/chat/completions` | `POST`   | Conversational AI with multi-modal support | Amazon Bedrock Converse API · Amazon Bedrock Mantle | `openai_chat_completion` |
| `/v1/chat/completions` | `GET` | List stored chat completions | Amazon Bedrock Sessions     | `openai_chat_completion_list` |
| `/v1/chat/completions/{completion_id}` | `GET` | Retrieve a stored chat completion     | Amazon Bedrock Sessions     | `openai_chat_completion_get` |
| `/v1/chat/completions/{completion_id}` | `POST` | Update a stored chat completion's metadata | Amazon Bedrock Sessions | `openai_chat_completion_update` |
| `/v1/chat/completions/{completion_id}` | `DELETE` | Delete a stored chat completion    | Amazon Bedrock Sessions     | `openai_chat_completion_delete` |
| `/v1/chat/completions/{completion_id}/messages` | `GET` | List the input messages of a stored chat completion | Amazon Bedrock Sessions | `openai_chat_completion_messages` |

## Feature Compatibility

Two outcomes are possible for a parameter no model behind this API can honor, and the Notes column below says which one applies. A parameter that only tunes an answer you can still use — a quality or latency hint, a telemetry opt-in — is **accepted and ignored**, so a client setting it on every request keeps working. A parameter that *is* the request — an output format, a modality, or a safety restriction that would silently disappear — is **rejected with a `400`** naming it, because the alternative is returning something you did not ask for.

<div class="feature-table" markdown>

| Feature                                  |                  Status                  | Notes                                                           |
|------------------------------------------|:----------------------------------------:|-----------------------------------------------------------------|
| **Messages & Roles**                     |                                          |                                                                 |
| Text messages                            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Full support for all text content                               |
| Image input (`image_url`)                |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | HTTP, data URIs                                                 |
| Image input from S3                      | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | S3 URLs                                                         |
| Video input                              |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Supported by select models                                      |
| Audio input                              |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Supported by select models                                      |
| Document input (`file`)                  |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | PDF and document support varies by model                        |
| Assistant `audio` reference              |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | An assistant turn carrying only `audio: {"id": …}` is dropped (past audio is not replayable); resend the `transcript` as text to keep it in context |
| Files API (`file_id`)                    |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Reference uploaded files via `type: "file"` — see [Files API](api_openai_files.md) |
| System messages                          |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Includes `developer` role                                       |
| **Tool Calling**                         |                                          |                                                                 |
| Function calling (`tools`)               |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Full OpenAI-compatible schema                                   |
| Legacy `function_call`                   |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Backward compatibility maintained                               |
| Parallel tool calls                      |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Multiple tools in one turn                                      |
| Disable parallel tool calls              |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | `parallel_tool_calls: false` is accepted for every model and honored by models able to constrain tool use; the response reports the tool calls actually made |
| Server tools                             | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Provider system tools and Claude server tools                   |
| `tool_choice`                             |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | `auto`, `none`, `required`, and named-function choice are supported; `tool_choice: {"type": "allowed_tools"}` is rejected with `400` — supported on the [Responses API](api_openai_responses.md) |
| **Generation Control**                   |                                          |                                                                 |
| `max_tokens` / `max_completion_tokens`   |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Output length limits                                            |
| `temperature`                            |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Mapped to Bedrock inference params                              |
| `top_p`                                  |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Nucleus sampling control                                        |
| `stop` sequences                         |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Custom stop strings. Whitespace-only sequences are rejected with `400` (Amazon Bedrock limitation) |
| `frequency_penalty` / `presence_penalty` |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Repetition control                                              |
| `seed`                                   |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Deterministic generation                                        |
| `logit_bias`                             |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Not all models support biasing                                  |
| `top_logprobs`                           |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Forwarded to the model as a provider-specific field; honored only by models that support it. Usable even though `logprobs` is rejected |
| `top_k` (From Qwen API)                  |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Candidate token set size for sampling                           |
| `reasoning_effort` (OpenAI API-compatible) |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Reasoning control: none/minimal/low/medium/high/xhigh/max (accepted for all models) |
| `enable_thinking` (Qwen API-compatible)  |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Enable/disable thinking mode (accepted for all reasoning models) |
| `thinking_budget` (Qwen API-compatible)  |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Thinking token budget (accepted for all reasoning models)      |
| `thinking` (Moonshot API-compatible)     |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Thinking config: {"type": "enabled"/"disabled"} (accepted for all models) |
| `reasoning` (OpenRouter API-compatible)  |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Reasoning object: `effort`, `max_tokens`, `enabled`, `exclude`. Equivalent to `reasoning_effort`, `thinking_budget` and `enable_thinking`; conflicting values are rejected with `400` |
| `include_reasoning` (OpenRouter API-compatible) |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | `false` omits the reasoning text from the response, like `reasoning: {"exclude": true}`; the reasoning tokens are still generated and billed |
| `n` (multiple choices)                   |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Generate multiple responses, not supported with streaming       |
| `logprobs`                               | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Rejected with `400` when enabled (`false`/`null` accepted, as they request the default behavior); `top_logprobs` (above) remains usable |
| `prediction`                             | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Static predicted output content. Accepted and ignored — a latency hint the Amazon Bedrock Converse API has no equivalent for, and the answer is unchanged without it; forwarded upstream on [Mantle](#bedrock-mantle) passthrough models |
| `response_format: "json_object"`         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Accepted for all models; syntactically valid JSON is not guaranteed for every model |
| `response_format: "json_schema"`         |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Structured JSON output validated against the supplied schema     |
| `verbosity`                              | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Model verbosity. Accepted and ignored — steer the answer length from the prompt; forwarded upstream on [Mantle](#bedrock-mantle) passthrough models |
| `web_search_options`                     | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Web search tool. Rejected with `400` when set                    |
| `translation_options` (Qwen API-compatible) | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Translation tuning options. Rejected with `400` when set          |
| `prompt_cache_key`                       |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Cache prompts to reduce costs and latency                       |
| `prompt_cache_options`                   |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | `mode: "explicit"` caches only the parts marked with `prompt_cache_breakpoint`; `ttl: "30m"` mapped to a 1 hour Amazon Bedrock retention on Anthropic models (other models use the default 5 minute TTL) when `prompt_cache_retention` is unset |
| `prompt_cache_breakpoint` (content part) |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Explicit cache boundary mapped to an Amazon Bedrock `cachePoint` (max. 4 per request) |
| Extra model-specific params              | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Extra model-specific parameters not supported by the OpenAI API |
| **Streaming & Output**                   |                                          |                                                                 |
| Text                                     |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Text messages                                                   |
| Streaming (`stream: true`)               |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Server-Sent Events (SSE)                                        |
| Streaming obfuscation                    | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Unsupported                                                     |
| Audio                                    |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Model output or synthesis from text output (synthesis is Converse-only — not performed for Mantle-served requests); non-streaming only — `stream: true` with audio output is rejected with `400` |
| `response_format` (JSON mode)            |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | `json_object` accepted for all models, without a syntax guarantee on every model; `json_schema` structured output is model-specific |
| `reasoning_content` (From Deepseek API)  |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Text reasoning messages. The single response field; on an assistant message replayed in `messages`, `reasoning` is accepted as an alias for it, and it is [dropped on models that only accept their own thinking](#replaying-reasoning-in-a-multi-turn-conversation) |
| `annotations` (URL citations)            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | URL citations from system tools (non-streaming only)            |
| **Usage tracking**                       |                                          |                                                                 |
| Input text tokens                        |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Billing unit                                                    |
| Output tokens                            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Billing unit                                                    |
| Reasoning tokens                         |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Converse-served models do not split them out: `completion_tokens_details` is not populated and reasoning tokens are billed inside `completion_tokens`. Mantle-native models report whatever split their upstream API returns |
| **Other**                                |                                          |                                                                 |
| Service tiers                            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Mapped to Bedrock service tiers and latency options             |
| `metadata`                               |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Echoed in the response, updatable on stored completions, and usable to filter the Bedrock invocation log. Also forwarded to Bedrock `requestMetadata`, whose limits apply: max 16 pairs, values ≤256 characters, restricted character set |
| `store`                                  |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Persists the completion in Amazon Bedrock session storage (non-streaming) |
| List / update stored completions         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | List with `model`/`metadata` filters; metadata update            |
| `safety_identifier` / `user`             |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Does not affect generation; identifies the end user in the request log and in [per-user cost attribution](operations_cost_management.md#per-user-attribution) |
| Bedrock Guardrails                       | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Content safety policies — not applied to Mantle-served requests |
| `moderation`                             |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Applies an Amazon Bedrock guardrail; results in the response (non-streaming) — rejected (`400`) on Mantle-served models |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep role="img" aria-label="Model-dependent" } **Model-Dependent** — Behavior depends on the model or backend; check the Notes column
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } **Extra Feature** — Enhanced capability beyond OpenAI API

</div>

## Model Support

All models supported by the Amazon Bedrock Converse and Converse Stream API are supported, plus every model served by [Bedrock Mantle](features.md#bedrock-mantle-models) when enabled — including OpenAI GPT-5.x, xAI Grok, and Google Gemma 4. Requests to Mantle models are passed through natively or converted automatically depending on the model's upstream API support — see [Bedrock Mantle](#bedrock-mantle) below.

### Bedrock Mantle

Mantle-served requests follow one of three paths, each with its own parameter fidelity:

| Serving path                | Models                                                                    | Parameter behavior |
|-----------------------------|---------------------------------------------------------------------------|--------------------|
| **Passthrough**             | Chat-native models (xAI Grok, OpenAI gpt-oss, Google Gemma 4, other open-weight models) | All schema-accepted parameters are forwarded; the upstream API may reject unsupported ones per model with a clean `400` (the upstream error code and parameter are propagated) |
| **Converted to Responses**  | OpenAI GPT frontier models; unknown models                                | Dropped silently: `stop`, `seed`, `frequency_penalty`, `presence_penalty`, `logit_bias`, `top_logprobs`, `audio`, `modalities`, `input_audio` content parts, legacy `functions`/`function_call`. Preserved: `metadata`, `safety_identifier`. `n > 1` rejected with `400`. `store` is handled by stdapi.ai only, never forwarded upstream |
| **Converted to Messages**   | Mantle-only Anthropic Claude models                                       | Same drops and `n > 1` rejection as the Responses conversion, except `stop` which is forwarded as `stop_sequences`; plus: `temperature` clamped to ≤ 1.0; `max_tokens` defaults to `4096` when unset; `reasoning_effort` mapped to Anthropic effort levels; `response_format` `json_object`/`json_schema` not available; `metadata`, `prompt_cache_key`, and `prompt_cache_retention` dropped; `service_tier` forwarded only when `auto` |

!!! note "Project attribution (`OpenAI-Project`)"
    Mantle requests can be attributed to a Bedrock Project for cost tracking and observability with the `OpenAI-Project: <project-id>` header (a bare project ID such as `proj_abc123`, not an ARN). It is honored per-request only when [`AWS_BEDROCK_ALLOW_MANTLE_PROJECT_OVERRIDE`](operations_configuration.md#bedrock-allow-mantle-project-override) is `true`; otherwise the server default ([`AWS_BEDROCK_MANTLE_PROJECT`](operations_configuration.md#bedrock-mantle-project)) applies. This applies **only** to models served by the Bedrock Mantle endpoint — classic `bedrock-runtime` models ignore the header.

### Model Name Aliases

This API supports dynamic model name aliases matching official provider APIs. Models like OpenAI and Anthropic provide dynamic aliases in their official APIs—this gateway supports the same model names, automatically resolving them to Amazon Bedrock model identifiers.

**Examples (OpenAI GPT OSS models supported by Bedrock):**

- `gpt-oss-20b` → `openai.gpt-oss-20b-1:0`

## Advanced Features

### Prompt Caching

Reduce costs and improve response times by caching frequently-used prompt components across multiple requests. This feature is particularly effective for applications with consistent system prompts, tool definitions, or conversation contexts.

**Supported Models:**

- **Anthropic Claude**: Full support for system, messages, and tools caching
- **Amazon Nova**: Support for system and messages caching

!!! info "Documentation"
    See [Amazon Bedrock Prompt Caching - Supported Models](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html#prompt-caching-models) for the complete list of models supporting prompt caching.

!!! warning "Cache Creation Costs"
    Cache creation incurs a higher cost than regular token processing. Only use prompt caching when you expect a high cache hit ratio across multiple requests with similar prompts.

**How to Use:**

Set the `prompt_cache_key` parameter to enable caching:

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-fable-5",
    "prompt_cache_key": "default",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant with extensive knowledge..."
      },
      {"role": "user", "content": "What is 2 + 2?"}
    ]
  }'
```

**Granular Cache Control:**

Enable caching for specific prompt sections using dot-separated values:

- `"system"` - Cache system messages only
- `"messages"` - Cache conversation history
- `"tools"` - Cache tool/function definitions (Anthropic Claude only)
- `"system.messages"` - Cache both system and messages
- `"system.tools"` - Cache system and tools
- `"messages.tools"` - Cache messages and tools
- `"system.messages.tools"` - Cache all components
- Any other non-empty value - Cache all components

!!! note "Custom Cache Keys Not Supported"
    Custom cache hash keys are not supported. The parameter is used only to control which sections are cached, not as a cache identifier.

```json
{
  "model": "anthropic.claude-fable-5",
  "prompt_cache_key": "system.tools",
  "messages": [...],
  "tools": [...]
}
```

**Benefits:**

- **Cost Reduction**: Cached tokens are billed at a lower rate than regular input tokens
- **Lower Latency**: Cached prompts eliminate reprocessing time
- **Automatic Management**: The API handles cache invalidation and updates

**Cache Retention (TTL):**

Control how long cached prompts persist using the `prompt_cache_retention` parameter:

!!! info "Model Support"
    Cache retention configuration is only available on select models. See [Amazon Bedrock Prompt Caching - Supported Models](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html#prompt-caching-models) for details on which models support configurable TTL.

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-fable-5",
    "prompt_cache_key": "default",
    "prompt_cache_retention": "24h",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant..."
      },
      {"role": "user", "content": "What is 2 + 2?"}
    ]
  }'
```

**Available Retention Values:**

- `"in_memory"` - Short-term caching (mapped to 5 minutes on Amazon Bedrock)
- `"24h"` - Long-term caching (mapped to 1 hour on Amazon Bedrock)
- Additional Amazon Bedrock values: `"1h"`, `"5m"` (provider-specific)

!!! note "OpenAI to Amazon Bedrock Mapping"
    OpenAI retention values are mapped to Amazon Bedrock equivalents for compatibility:

    - `"in_memory"` → 5 minutes
    - `"24h"` → 1 hour

The OpenAI `prompt_cache_options` object is also accepted: its `ttl` (`"30m"`) is mapped to a 1 hour Amazon Bedrock retention on Anthropic models (other models use the default 5 minute TTL) when `prompt_cache_retention` is not set.

**Explicit Cache Breakpoints:**

Instead of relying on the `prompt_cache_key` section heuristics, mark the exact cache boundaries with `prompt_cache_breakpoint` on any content part (`text`, `image_url`, `input_audio`, `file`, `refusal`). Each marked part is followed by an Amazon Bedrock `cachePoint`, so the prompt prefix ending with that part is cached:

```json
{
  "model": "anthropic.claude-fable-5",
  "prompt_cache_options": {"mode": "explicit"},
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "Long reusable instructions...",
          "prompt_cache_breakpoint": {"mode": "explicit"}
        }
      ]
    },
    {"role": "user", "content": "What is 2 + 2?"}
  ]
}
```

- `"mode": "explicit"` caches **only** the marked parts: the `prompt_cache_key` heuristics are disabled for that request.
- `"mode": "implicit"` (default) keeps the `prompt_cache_key` heuristics **and** honors the marked parts.
- At most 4 cache points are sent per request (Amazon Bedrock limit); the oldest ones are dropped when more are requested.
- Breakpoints on models without prompt caching support are accepted and ignored, as are breakpoints on tool result messages — those never become a cache point, whatever the model.

**Usage Tracking:**

Cached token usage is reported in the response:

```json
{
  "usage": {
    "prompt_tokens": 1500,
    "completion_tokens": 100,
    "total_tokens": 1600,
    "prompt_tokens_details": {
      "cached_tokens": 1200,
      "cache_write_tokens": 300
    }
  }
}
```

In this example, 1,200 tokens were retrieved from cache and the remaining 300 tokens were processed and written to the cache. `cache_write_tokens` (an extra field beyond the OpenAI API) reports the tokens written to the cache when the model reports them, on both non-streaming responses and the trailing `stream_options.include_usage` chunk; it is omitted when no cache write occurred.

### System Prompt

System prompts define the AI assistant's behavior, personality, and instructions (e.g., "You are a helpful assistant"). Most models support system prompts.

!!! warning "Unsupported Models"
    Some models don't support system prompts (`mistral.mistral-7b-instruct-v0:2`, `mistral.mixtral-8x7b-instruct-v0:1`). By default, **stdapi.ai silently drops system messages** for these models, allowing cross-model compatibility. To receive errors instead, configure [`DROP_UNSUPPORTED_SYSTEM_PROMPT=false`](operations_configuration.md#drop-unsupported-system-prompt).

### ![Amazon S3](styles/logo_amazon_s3.svg){ style="height: 1.2em; vertical-align: text-bottom;" } S3 Image Support

Access images directly from your S3 buckets without generating pre-signed URLs or downloading files locally.

**Supported Formats:**

- **Images**: JPEG, PNG, GIF, WebP

**How to Use:**

Simply reference your S3 images using the `s3://` URI scheme in `image_url` fields:

```json
{
  "model": "anthropic.claude-fable-5",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {
          "type": "image_url",
          "image_url": {"url": "s3://my-bucket/images/photo.jpg"}
        }
      ]
    }
  ]
}
```

!!! warning "IAM Permissions Required"
    Your API service must have IAM permissions to read from the specified S3 buckets. S3 objects must be in the same AWS region as the executed model or accessible via your IAM role. Standard S3 data transfer and request costs apply.

**Benefits:**

- No pre-signed URLs - Direct S3 access without generating temporary URLs
- Security - Images stay in your AWS account with IAM-controlled access
- Performance - Optimized data transfer within AWS infrastructure
- Large images - No size limitations of data URIs or base64 encoding

### :material-file-link: Files API References (`file-id:`)

The string-overloaded `image_url.url`, `file.file_data`, and `input_audio.data` fields also accept the project-local `file-id:` URI scheme to reference files previously uploaded via the [Files API](api_openai_files.md):

```json
{
  "model": "anthropic.claude-fable-5",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image"},
      {"type": "image_url", "image_url": {"url": "file-id:file-0190c51c7de7455d9b8c2efe27dfbf67"}}
    ]
  }]
}
```

!!! info "Two equivalent ways to reference an uploaded file"
    The OpenAI-native typed path `{"type": "file", "file": {"file_id": "file-…"}}` is unchanged and still preferred when a typed object is acceptable. The `file-id:` URI is the equivalent for the *string-overloaded* `image_url.url` / `file.file_data` / `input_audio.data` fields, where today you would otherwise pass an `s3://`, `https://`, or `data:` URI. See [Referencing Uploaded Files via the `file-id:` URI Scheme](api_openai_files.md#referencing-uploaded-files-via-the-file-id-uri-scheme).

### Server Tools

Amazon Bedrock system tools are built-in capabilities that foundation models can use directly, without requiring you to implement backend integrations.

**How to Use:**

Add system tools to your `tools` array as normal. System tools don't require parameter definitions—just specify the tool name and the model will handle the rest.

#### ![Amazon Nova](styles/logo_amazon_nova.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Nova Tools

| Tool | `function.name` | Amazon Nova 2 | Amazon Nova Premier (legacy) | API Support |
|------|-----------------|:-------------:|:-------------------:|:-----------:|
| Web Grounding | `nova_grounding` | :material-check-circle:{ .success role="img" aria-label="Supported" } | :material-check-circle:{ .success role="img" aria-label="Supported" } | :material-check-circle:{ .success role="img" aria-label="Supported" } |
| Code Interpreter | `nova_code_interpreter` | :material-check-circle:{ .success role="img" aria-label="Supported" } | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } |

!!! danger "Code Interpreter Not Compatible"
    **`nova_code_interpreter` cannot be used via this API.** The code execution result cannot be surfaced in the OpenAI Chat Completions response format.

##### Web Grounding

Amazon Nova Web Grounding enables models to search the web for current information, helping answer questions requiring real-time data like news, weather, product availability, or recent events. The model automatically determines when to use web grounding based on the user's query.

!!! info "Learn More"
    - [Amazon Nova Web Grounding - User Guide](https://docs.aws.amazon.com/nova/latest/userguide/grounding.html)
    - [Build More Accurate AI Applications with Amazon Nova Web Grounding - Blog Post](https://aws.amazon.com/blogs/aws/build-more-accurate-ai-applications-with-amazon-nova-web-grounding/)

**Usage:**

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-2-lite-v1:0",
    "messages": [
      {
        "role": "user",
        "content": "What are the current AWS Regions and their locations?"
      }
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "nova_grounding"
        }
      }
    ]
  }'
```

**Response Format:**

When using web grounding, the API response includes `annotations` with URL citations in non-streaming mode:

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The AWS Regions include...",
      "annotations": [
        {
          "type": "url_citation",
          "url_citation": {
            "url": "https://aws.amazon.com/about-aws/global-infrastructure/",
            "title": "AWS Global Infrastructure"
          }
        }
      ]
    }
  }]
}
```

!!! note "Streaming Mode"
    URL citation `annotations` are only available in non-streaming responses.

**Limitations:**

- **No streaming citations**: URL citation `annotations` are not emitted in streaming responses.

!!! warning "Region Compatibility"
    Web Grounding is only available in US Amazon Bedrock regions. To ensure all requests are routed to a US region, restrict the model using [`AWS_BEDROCK_MODEL_REGION_RESTRICT`](operations_configuration.md#bedrock-model-region-restrict):

    ```bash
    export AWS_BEDROCK_MODEL_REGION_RESTRICT='{"amazon.nova-": ["us-east-1"]}'
    ```

#### ![Claude](styles/logo_anthropic_claude.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Anthropic Claude Server Tools

Anthropic Claude models support server-side tools (bash, text editor, memory) that are executed by the model provider. Declare them using the standard OpenAI function tool format: set `type` to `"function"` and `function.name` to the tool name.

**Supported Tools by Model:**

| Tool | `function.name` | Claude 3.5 Sonnet v2 | Claude 3.7+ |
|------|-----------------|:--------------------:|:-----------:|
| Bash | `bash` | :material-check-circle:{ .success role="img" aria-label="Supported" } | :material-check-circle:{ .success role="img" aria-label="Supported" } |
| Text Editor | `str_replace_based_edit_tool` | :material-check-circle:{ .success role="img" aria-label="Supported" } | :material-check-circle:{ .success role="img" aria-label="Supported" } |
| Computer | `computer` | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } |
| Memory | `memory` | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | :material-check-circle:{ .success role="img" aria-label="Supported" } |

!!! warning "Computer Use Not Supported"
    The computer use workflow requires screenshots to be returned as images inside tool results. The OpenAI Chat Completions API does not support image content in `role: "tool"` messages, so the complete agent loop cannot be implemented. **`computer` is not usable via this route.**

**Usage:**

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-5",
    "messages": [
      {"role": "user", "content": "Run a Python script that prints hello world."}
    ],
    "tools": [
      {"type": "function", "function": {"name": "bash"}},
      {"type": "function", "function": {"name": "str_replace_based_edit_tool"}}
    ]
  }'
```

**Tool Parameters:**

Some Claude server tools accept additional configuration. Pass tool-specific parameters inside `function.parameters`:

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-5",
    "messages": [
      {"role": "user", "content": "Edit the file hello.py to print hello world."}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "str_replace_based_edit_tool",
          "parameters": {"type": "object", "max_characters": 5000}
        }
      }
    ]
  }'
```

!!! tip "Beta Headers"
    Claude server tools require specific `anthropic-beta` flags, which are **automatically injected** — no manual header needed:

    - `bash`, `str_replace_based_edit_tool` → `computer-use-2024-10-22` (Claude 3.5) or `computer-use-2025-01-24` (Claude 3.7+)
    - `memory` → `context-management-2025-06-27` (Claude 3.7+)

### Reasoning Control

This API supports several approaches to control [Amazon Bedrock reasoning](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-reasoning.html) behavior. Reasoning enables foundation models to break down complex tasks into smaller steps ("chain of thought"), improving accuracy for multi-step analysis, math problems, and complex reasoning tasks. All approaches work with all Amazon Bedrock models that support reasoning capabilities.

!!! info "Model Support for Configurable Reasoning"
    Not all reasoning-capable models support configurable reasoning control. Support varies by model:

    - **Anthropic Claude 3.7 - 4.5**: Both `reasoning_effort` and `thinking_budget` parameters supported (token budget-based reasoning)
    - **Anthropic Claude Sonnet 4.6 / Opus 4.6 and later** (including Fable and Mythos): `reasoning_effort` parameter only (adaptive reasoning)
    - **Amazon Nova 2 models**: `reasoning_effort` parameter only
    - **DeepSeek V3 models**: `reasoning_effort` parameter only

    Models listed as effort-only still accept a token budget: it turns reasoning
    on, and the depth comes from their own effort scale.

#### ![OpenAI](styles/logo_openai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } OpenAI and DeepSeek API-Compatible Reasoning Parameters

Use the `reasoning_effort` parameter with predefined effort levels. This format is shared by the OpenAI and [DeepSeek](https://api-docs.deepseek.com/api/create-chat-completion) Chat Completions APIs and works with all Amazon Bedrock models supporting reasoning.

**Available Levels:**

- `none` - Disable reasoning
- `minimal` - Quick responses with minimal reasoning
- `low` - Light reasoning for straightforward tasks
- `medium` - Balanced reasoning for most use cases
- `high` - Deep reasoning for complex problems
- `xhigh` - Maximum reasoning for complex problems
- `max` - Its own (higher) effort tier on the adaptive Claude models served by the Converse API (Sonnet/Opus 4.6 and later, plus Fable), which forward it unchanged; collapsed onto the model's top reasoning tier on the fixed-scale models (Claude 3.7 - 4.5, Amazon Nova 2, DeepSeek, Kimi). On Amazon Bedrock Mantle, Claude models — Mythos among them — are reached over the Anthropic Messages API, and that conversion maps `max` to `high`; every other Mantle-served model receives the level as sent. Claude 4.6 also maps `xhigh` down to `high`

**What You Get:**

- **`reasoning_content` field** (DeepSeek API-compatible): models include their thinking process in the response. The field name is an operator setting ([`CHAT_COMPLETIONS_REASONING_FIELD`](operations_configuration.md#chat-completions-reasoning-field)): `reasoning_content` by default, `reasoning` for clients written against OpenRouter or vLLM, or `none` to keep responses strictly OpenAI-shaped.
- **Streaming support**: `choices[].delta.reasoning_content` chunks in real time

**Example:**

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-fable-5",
    "reasoning_effort": "high",
    "messages": [{"role": "user", "content": "Solve this complex problem..."}]
  }'
```

!!! note "Compatibility"
    This format is accepted for all reasoning-capable models. Models that don't support this parameter will ignore it.

#### ![Qwen](styles/logo_qwen.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Qwen API-Compatible Reasoning Parameters

Use explicit `enable_thinking` & `thinking_budget` parameters for fine-grained control over thinking mode. This is a Qwen API-compatible format that works with all Amazon Bedrock models supporting reasoning.

**Parameters:**

- `enable_thinking` (boolean): Enable or disable thinking mode
    - Default: Model-specific (usually `false`)
- `thinking_budget` (integer): Maximum thinking process length in tokens
    - Only effective when `enable_thinking` is `true`
    - Passed to the model as `budget_tokens`
    - Default: Model's maximum chain-of-thought length

**Example:**

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-5",
    "enable_thinking": true,
    "thinking_budget": 2000,
    "messages": [{"role": "user", "content": "Solve this complex problem..."}]
  }'
```

!!! note "Compatibility"
    This format is accepted for all reasoning-capable models. Models that don't support these parameters will ignore them.

#### ![Moonshot](styles/logo_moonshot.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Moonshot API-Compatible Thinking Control

The `thinking` parameter provides a Moonshot API-compatible format for controlling thinking/reasoning on models that support it.

!!! info "Documentation"
    See [Moonshot Kimi API](https://platform.kimi.ai/docs/api/chat) for more information.

**Parameters:**

- `thinking={"type": "enabled"}` — Enable thinking mode
- `thinking={"type": "disabled"}` — Disable thinking mode

This format is accepted for all reasoning-capable models. Models that don't support this parameter will ignore it.

#### :material-router-network: OpenRouter API-Compatible Reasoning Object

The `reasoning` object groups the same controls into a single field, and `include_reasoning` toggles whether the reasoning text is returned.

!!! info "Documentation"
    See [OpenRouter Reasoning Tokens](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens) for more information.

**Parameters:**

- `reasoning.effort` (string): Effort level, exactly as `reasoning_effort`
- `reasoning.max_tokens` (integer): Reasoning token budget, exactly as `thinking_budget` (implies `enabled`)
- `reasoning.enabled` (boolean): Enable reasoning, exactly as `enable_thinking`
- `reasoning.exclude` (boolean): Omit the reasoning text from the response
- `include_reasoning` (boolean): `false` is equivalent to `reasoning: {"exclude": true}`

**Example:**

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-5",
    "reasoning": {"effort": "high", "exclude": true},
    "messages": [{"role": "user", "content": "Solve this complex problem..."}]
  }'
```

!!! note "Conflicting Values"
    `reasoning.effort` and `reasoning.max_tokens` are mutually exclusive, and a sub-field disagreeing with its flat equivalent (for example `reasoning: {"effort": "high"}` with `reasoning_effort: "low"`) is rejected with `400`.

!!! note "Excluded Reasoning"
    Excluding the reasoning text does not disable reasoning: the model still thinks, and the reasoning tokens are still counted in `usage` and billed. Use `reasoning_effort: "none"` or `reasoning: {"enabled": false}` to turn reasoning off.

#### :material-history: Replaying Reasoning in a Multi-Turn Conversation

Appending the assistant message you just received to `messages` — the standard
multi-turn idiom, and what the [DeepSeek API](https://api-docs.deepseek.com/api/create-chat-completion)
asks for after a tool call — is always accepted, `reasoning_content` included.

Anthropic Claude models only continue from a thinking passage they can recognise
as their own, which a replayed text field is not, so their reasoning is left out
of that turn instead. The message content, tool calls and refusal are sent
unchanged and the conversation continues normally; only the earlier chain of
thought is no longer visible to the model. Every other model family receives the
replayed reasoning as-is.

What this does **not** affect, measured on Claude Haiku 4.5:

- **Reasoning on the new turn.** The model still thinks, and still returns
  `reasoning_content` — the reasoning setting for the turn being generated is
  independent of the history. It re-derives rather than continuing the earlier
  chain.
- **Tool-call continuations.** A turn that carried a tool call is answered
  correctly with the earlier reasoning left out.
- **[Prompt caching](#prompt-caching).** Cache hits are unaffected, including
  when the cache point sits inside the conversation immediately after the turn
  whose reasoning is left out — measured across three turns, each one reading
  the previous turn's cache in full and extending it. The omission is the same
  on every turn, so the cached prefix stays identical and keeps growing.

!!! tip "Keeping the reasoning in context on Claude"
    Use the [Responses API](api_openai_responses.md), which carries the thinking
    passage in a form Claude accepts. Request
    `include: ["reasoning.encrypted_content"]` and echo the reasoning items back,
    and the model continues from its own earlier reasoning.

### Provider-Specific Parameters

Unlock advanced model capabilities by passing provider-specific parameters directly in your requests. These parameters are forwarded to Amazon Bedrock and allow you to access features unique to each foundation model provider.

!!! info "Documentation"
    See [Bedrock Model Parameters](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html) for the complete list of available parameters per model.

**How It Works:**

Add provider-specific fields at the top level of your request body alongside standard OpenAI parameters. The API automatically forwards these to the appropriate model provider via Amazon Bedrock.

**Examples:**

**Top K Sampling:**
```json
{
  "model": "anthropic.claude-fable-5",
  "messages": [{"role": "user", "content": "Write a poem"}],
  "top_k": 50,
  "temperature": 0.7
}
```

**Configuration Options:**

**Option 1: Per-Request**

Add provider-specific parameters directly in your request body (as shown in examples above).

**Option 2: Server-Wide Defaults**

Configure default parameters for specific models via the `DEFAULT_MODEL_PARAMS` environment variable:

```bash
export DEFAULT_MODEL_PARAMS='{
  "anthropic.claude-sonnet-5": {
    "anthropic_beta": ["context-management-2025-06-27"]
  }
}'
```

!!! tip "Parameter Priority"
    Per-request parameters override server-wide defaults.

**Behavior:**

- :material-check-circle:{ .success role="img" aria-label="Supported" } **Compatible parameters**: Forwarded to the model and applied
- :material-alert-circle:{ .warning } **Unsupported parameters**: Return HTTP 400 with an error message
- :material-alert-circle:{ .warning } **Reserved names**: `model_id` and `additional_request_fields` collide with the gateway's own request-building parameters and are rejected with a `400 invalid_request_error` naming the key, instead of being forwarded

#### ![Claude](styles/logo_anthropic_claude.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Anthropic Claude Features

Enable cutting-edge Claude capabilities through Anthropic beta feature flags.

##### Beta Feature Flags

Enable experimental Claude features like interleaved thinking by adding the `anthropic_beta` array to your request (extended thinking itself is controlled through the [reasoning parameters](#reasoning-control), not a beta flag):

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-5",
    "messages": [{"role":"user","content":"Summarize the news headline."}],
    "anthropic_beta": ["Interleaved-thinking-2025-05-14"]
  }'
```

!!! tip "Server-Wide Configuration"
    You can also configure beta flags server-wide using the `DEFAULT_MODEL_PARAMS` environment variable (see [Provider-Specific Parameters](#provider-specific-parameters)).

!!! warning "Unsupported Beta Flags"
    Unsupported flags that would change output return HTTP 400 errors.

!!! info "Documentation"
    See [Using Claude on Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html) for more details on Claude-specific parameters.


## Stored Chat Completions

Set `store: true` to persist a chat completion in [Amazon Bedrock session storage](https://docs.aws.amazon.com/bedrock/latest/userguide/sessions.html) — same mechanism, region, and [KMS setting](operations_configuration.md#aws-bedrock-session-encryption-key-arn) as [stored responses](api_openai_responses.md#stored-responses). The returned `id` then works with the full stored-completion surface:

- `GET /v1/chat/completions` — list stored completions, sorted by creation time (`order`, `after`, `limit`), filterable by `model` and by metadata pairs. Metadata filters accept either one `metadata[key]=value` parameter per key (what the OpenAI SDKs send) or a single `metadata={"key": "value"}` JSON object of string values, for clients that can only send a whole object in one query parameter. A bare `metadata` in any other shape is rejected with `400` naming both accepted forms.
- `GET /v1/chat/completions/{completion_id}` — retrieve the stored completion.
- `POST /v1/chat/completions/{completion_id}` — replace its `metadata` (`null` clears it).
- `GET /v1/chat/completions/{completion_id}/messages` — list its input messages.
- `DELETE /v1/chat/completions/{completion_id}` — delete it and its backing session.

`store` defaults to **false** on this implementation and is ignored with `stream=true` or when the server lacks the [session storage IAM permissions](operations_configuration.md#bedrock-session-storage-optional) (a warning is recorded in the request log). Listings scan a capped number of sessions (1,000) in the primary Bedrock region; accounts beyond the cap may see incomplete listings.

## Available Request Headers

This endpoint supports standard Bedrock headers for enhanced control over your requests. All headers are optional and can be combined as needed.

### Content Safety (Guardrails)

| Header                               | Purpose                            | Valid Values                          |
|--------------------------------------|------------------------------------|---------------------------------------|
| `X-Amzn-Bedrock-GuardrailIdentifier` | Guardrail ID for content filtering | Your guardrail identifier             |
| `X-Amzn-Bedrock-GuardrailVersion`    | Guardrail version                  | Version number (e.g., `1`)            |
| `X-Amzn-Bedrock-Trace`               | Guardrail trace level              | `disabled`, `enabled`, `enabled_full` |

### Performance Optimization

| Header                                     | Purpose                | Valid Values                              |
|--------------------------------------------|------------------------|-------------------------------------------|
| `X-Amzn-Bedrock-Service-Tier`              | Service tier selection | `default`, `flex`, `priority`, `reserved` |
| `X-Amzn-Bedrock-PerformanceConfig-Latency` | Latency optimization   | `standard`, `optimized`                   |

**Example with all headers:**

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Amzn-Bedrock-GuardrailIdentifier: your-guardrail-id" \
  -H "X-Amzn-Bedrock-GuardrailVersion: 1" \
  -H "X-Amzn-Bedrock-Trace: enabled" \
  -H "X-Amzn-Bedrock-Service-Tier: priority" \
  -H "X-Amzn-Bedrock-PerformanceConfig-Latency: optimized" \
  -d '{
    "model": "anthropic.claude-fable-5",
    "messages": [{"role": "user", "content": "Hello!"}]
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
- `temperature` and `max_tokens` are forwarded.
- `response_format: json_schema` is forwarded as Pegasus's structured output.

**Silently ignored** (no error): system prompts, tools, `top_p`, stop sequences, and prompt caching.

**Upstream format limitation:** The OpenAI Chat Completions API has no `video_url` content part type. To stay fully compatible with standard OpenAI clients, pass the video as an **`image_url`** content part — the server detects the video MIME type automatically and routes it to Pegasus correctly.

**Video input formats**: `data:video/mp4;base64,…`, `https://…`, `s3://bucket/key`, or `file-id:…`. Videos above 18.75 MB are automatically uploaded to S3.

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "twelvelabs.pegasus-1-2-v1:0",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image_url",
            "image_url": {"url": "s3://my-bucket/video.mp4"}
          },
          {"type": "text", "text": "Describe what happens in this video."}
        ]
      }
    ]
  }'
```

## Try It Now

**Basic chat completion:**

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "messages": [{"role": "user", "content": "Say hello world"}]
  }'
```

**Streaming response:**

```bash
curl -N -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "stream": true,
    "messages": [{"role": "user", "content": "Write a haiku about the sea."}]
  }'
```

**Multi-modal with image:**

```json
{
  "model": "amazon.nova-micro-v1:0",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
      ]
    }
  ]
}
```

**With reasoning:**

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-5",
    "reasoning_effort": "low",
    "messages": [{"role": "user", "content": "Solve 12*13"}]
  }'
```

**Response with reasoning:**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "reasoning_content": "12 × 10 = 120, plus 12 × 3 = 36 → 156",
      "content": "156"
    }
  }]
}
```

---

**Ready to build with AI?** Check out the [Models API](api_openai_models.md) to see all available foundation models!
