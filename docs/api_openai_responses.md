---
title: Responses API - AWS Bedrock with OpenAI Compatibility
description: OpenAI-compatible Responses API for AWS Bedrock models. Supports streaming, tool calling, structured outputs, and multimodal inputs.
keywords: responses API, OpenAI responses API, AWS Bedrock responses, streaming responses, tool calling API, structured output
---

# Responses API

Generate model responses with AWS Bedrock foundation models through an OpenAI Responses API-compatible interface. Supports text, images, tool calling, and streaming.

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

## Quick Start: Available Endpoint

| Endpoint                     | Method | What It Does                                     | Powered By                  | MCP Tool                       |
|------------------------------|--------|--------------------------------------------------|-----------------------------|--------------------------------|
| `/v1/responses`              | `POST` | Create a model response                          | AWS Bedrock Converse API    | `openai_response`              |
| `/v1/responses/input_tokens` | `POST` | Count input tokens without generating a response | AWS Bedrock CountTokens API | `openai_response_input_tokens` |
| `/v1/responses/compact`      | `POST` | Compact a conversation into a reusable summary   | AWS Bedrock Converse API    | `openai_response_compact`      |
| `/v1/responses/{response_id}` | `GET`  | Retrieve a stored response                       | AWS Bedrock Sessions        | `openai_response_get`          |
| `/v1/responses/{response_id}` | `DELETE` | Delete a stored response                       | AWS Bedrock Sessions        | `openai_response_delete`       |
| `/v1/responses/{response_id}/cancel` | `POST` | Cancel a background response (always fails: no background support) | AWS Bedrock Sessions | `openai_response_cancel` |
| `/v1/responses/{response_id}/input_items` | `GET` | List the input items of a stored response | AWS Bedrock Sessions   | `openai_response_input_items`  |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                                                               |                 Status                  | Notes                                                                        |
|-----------------------------------------------------------------------|:---------------------------------------:|------------------------------------------------------------------------------|
| **Input**                                                             |                                         |                                                                              |
| Plain text (`input` as string)                                        |   :material-check-circle:{ .success }   | Simple string shorthand for a single user message                            |
| Structured message array                                              |   :material-check-circle:{ .success }   | Array of `EasyInputMessage` / `InputMessage` items                           |
| `instructions` (system prompt)                                        |   :material-check-circle:{ .success }   | Injected as a Bedrock system block                                           |
| `system` / `developer` role                                           |   :material-check-circle:{ .success }   | Treated as a system instruction                                              |
| Image input (`input_image`)                                           |      :material-cog:{ .model-dep }       | HTTP URLs and base64 data URIs supported                                     |
| File input (`input_file`)                                             |      :material-cog:{ .model-dep }       | File URLs and base64 data supported                                          |
| `function_call_output`                                                |   :material-check-circle:{ .success }   | Submit tool results as input for round-trip tool calling                     |
| **Tool Calling**                                                      |                                         |                                                                              |
| Function tools (`type: "function"`)                                   |   :material-check-circle:{ .success }   | Full schema mapping to Bedrock toolSpec                                      |
| `tool_choice: "auto"`                                                 |   :material-check-circle:{ .success }   | Model selects among available tools                                          |
| `tool_choice: "required"`                                             |   :material-check-circle:{ .success }   | Model must call at least one tool                                            |
| `tool_choice: "none"`                                                 |   :material-check-circle:{ .success }   | Prevents tool calls                                                          |
| Named `tool_choice` (force)                                           |   :material-check-circle:{ .success }   | Force a specific function to be called                                       |
| `parallel_tool_calls`                                                 |   :material-check-circle:{ .success }   | Echoed in response; not transmitted to Bedrock                               |
| Built-in tools (`code_interpreter`, `web_search`, `image_generation`) |      :material-cog:{ .model-dep }       | See [OpenAI Integrated Tools](#openai-integrated-tools)                      |
| `file_search` tool                                                    | :material-close-circle:{ .unsupported } | Returns `400`; no Bedrock equivalent                                         |
| `computer` / `computer_use_preview` tools                             | :material-close-circle:{ .unsupported } | Returns `400`; see [Computer Use Not Supported](#computer-use-not-supported) |
| `mcp` tool                                                            | :material-close-circle:{ .unsupported } | Returns `400`; MCP not supported                                             |
| `local_shell` / `shell` tools                                         | :material-close-circle:{ .unsupported } | Returns `400`; local shell not supported                                     |
| `custom` / `namespace` / `tool_search` / `apply_patch` tools          | :material-close-circle:{ .unsupported } | Returns `400`; not supported                                                 |
| **Generation Control**                                                |                                         |                                                                              |
| `max_output_tokens`                                                   |   :material-check-circle:{ .success }   | Maps to Bedrock `maxTokens`                                                  |
| `temperature`                                                         |      :material-cog:{ .model-dep }       | 0–2 range; mapped to Bedrock inference config                                |
| `top_p`                                                               |      :material-cog:{ .model-dep }       | 0–1 range; nucleus sampling                                                  |
| `top_logprobs`                                                        |      :material-cog:{ .model-dep }       | 0–20 range; token log-probability output                                     |
| `reasoning` (effort)                                                  |      :material-cog:{ .model-dep }       | Configures reasoning on models that support it                               |
| `reasoning.context`                                                   | :material-close-circle:{ .unsupported } | Returns `400`; not supported                                                 |
| `metadata`                                                            |   :material-check-circle:{ .success }   | Forwarded to Bedrock `requestMetadata`                                       |
| `prompt_cache_key`                                                    |      :material-cog:{ .model-dep }       | Cache prompts to reduce costs and latency                                    |
| `prompt_cache_retention`                                              |      :material-cog:{ .model-dep }       | Cache TTL: `in-memory`, `24h`, `1h`, or `5m`                                 |
| `service_tier`                                                        |   :material-check-circle:{ .success }   | Maps to Bedrock service tier header                                          |
| `truncation`                                                          | :material-close-circle:{ .unsupported } | Returns `400`; Bedrock manages context automatically                         |
| `max_tool_calls`                                                      | :material-close-circle:{ .unsupported } | Returns `400`; not supported                                                 |
| `background`                                                          | :material-close-circle:{ .unsupported } | Returns `400`; async background mode not supported                           |
| `store`                                                               |   :material-check-circle:{ .success }   | Persists the response in AWS Bedrock session storage (non-streaming)         |
| `stream_options`                                                      | :material-close-circle:{ .unsupported } | Returns `400`; not supported                                                 |
| `conversation`                                                        | :material-close-circle:{ .unsupported } | Returns `400`; use `previous_response_id` or `input`                         |
| `prompt` (template reference)                                         | :material-close-circle:{ .unsupported } | Returns `400`; not supported                                                 |
| `safety_identifier`                                                   | :material-close-circle:{ .unsupported } | Returns `400`; not supported                                                 |
| `moderation`                                                          |   :material-check-circle:{ .success }   | Applies an AWS Bedrock guardrail; results in the response (non-streaming)   |
| **Output Format**                                                     |                                         |                                                                              |
| `text.format: "text"`                                                 |   :material-check-circle:{ .success }   | Plain text output                                                            |
| `text.format: "json_object"`                                          |      :material-cog:{ .model-dep }       | JSON object output via Bedrock outputConfig                                  |
| `text.format: "json_schema"`                                          |      :material-cog:{ .model-dep }       | Structured JSON output with schema validation                                |
| **Multi-Turn**                                                        |                                         |                                                                              |
| `previous_response_id`                                                |   :material-check-circle:{ .success }   | Continues a response stored with `store=true`                                |
| Compaction (`POST /v1/responses/compact`)                             |   :material-check-circle:{ .success }   | Stateless summary item; send it back in `input` to continue                  |
| **Streaming**                                                         |                                         |                                                                              |
| `stream: true`                                                        |   :material-check-circle:{ .success }   | SSE stream with full lifecycle events                                        |
| `response.created`                                                    |   :material-check-circle:{ .success }   | Emitted at stream start                                                      |
| `response.in_progress`                                                |   :material-check-circle:{ .success }   | Emitted after created                                                        |
| `response.output_text.delta`                                          |   :material-check-circle:{ .success }   | Text token deltas                                                            |
| `response.output_text.done`                                           |   :material-check-circle:{ .success }   | Final text for each content part                                             |
| `response.function_call_arguments.delta`                              |   :material-check-circle:{ .success }   | Tool call argument deltas                                                    |
| `response.function_call_arguments.done`                               |   :material-check-circle:{ .success }   | Finalized tool call arguments                                                |
| `response.completed`                                                  |   :material-check-circle:{ .success }   | Final complete response at stream end                                        |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep } **Available on Select Models** — Check your model's capabilities
* :material-close-circle:{ .unsupported } **Unsupported** — Not available in this implementation

</div>

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
    All responses are stateless. Response IDs are generated for compatibility but `previous_response_id` is not supported. For multi-turn conversations, pass the full message history in the `input` array.

!!! warning "Unsupported Built-In Tools"
    `file_search`, `computer`, `computer_use_preview`, `mcp`, `local_shell`, `shell`,
    `custom`, `namespace`, `tool_search`, and `apply_patch` tools are **not supported**.
    Requests that include any of these tools will receive a `400` error.

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

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "input": [
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

The stream emits events in order: `response.created` → `response.in_progress` → `response.output_text.delta` (repeated) → `response.output_text.done` → `response.completed`.

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
    "model": "anthropic.claude-sonnet-5",
    "input": "Solve: if a train travels 120 km in 90 minutes, what is its speed?",
    "reasoning": {"effort": "low"},
    "max_output_tokens": 4096
  }'
```

### Prompt Caching

!!! warning "Cache Creation Costs"
    Cache creation incurs a higher cost than regular token processing. Only use prompt caching when you expect a high cache hit ratio across multiple requests with similar prompts.

Prompt caching reduces latency and costs by caching repetitive prompt components. Set the `prompt_cache_key` parameter to enable:

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-5",
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
    "model": "anthropic.claude-sonnet-5",
    "input": "Hello",
    "prompt_cache_key": "default",
    "prompt_cache_retention": "24h"
  }'
```

Valid values: `in-memory` (default), `24h`, `1h`, or `5m`. The `1h` and `5m` values are AWS Bedrock-specific. On AWS Bedrock, `in-memory` maps to 5 minutes and `24h` maps to 1 hour.

!!! note "Model Support"
    Cache retention configuration is only available on select models. See [AWS Bedrock Prompt Caching - Supported Models](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html#prompt-caching-models) for details on which models support configurable TTL.

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

In this example, 1,200 tokens were retrieved from cache, with only 300 tokens requiring processing.

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
    "model": "amazon.nova-premier-v1:0",
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

!!! warning "Streaming sources"
    `action.sources` (citation URLs) is only populated in non-streaming responses.
    In streaming mode the field is `null`, though all lifecycle events
    (`web_search_call.in_progress`, `web_search_call.completed`) are still emitted.

!!! warning "Region Compatibility"
    `web_search` is only available on Nova Premier in US regions. Not available on EU inference profiles.

#### :material-image: Image Generation

The `image_generation` integrated tool works with **all text models** — Claude, Nova, and any future model. The gateway intercepts the tool, lets the LLM compose the image prompt and parameters via a synthetic function call, then generates the image against a configured Bedrock image model and returns an `image_generation_call` output item to the client. Intermediate `function_call` items are suppressed.

!!! info "Configuration Required"
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

#### Computer Use Not Supported

!!! failure "Computer Use Not Supported"
    The `computer` and `computer_use_preview` integrated tools are **not supported**. Requests that include these tools will receive a `400` error.

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

## Model-specific features

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
curl https://api.example.com/v1/responses \
  -H "Authorization: Bearer $API_KEY" \
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
    "model": "anthropic.claude-sonnet-5",
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
    The `previous_response_id`, `conversation`, and `personality` parameters are not supported for token counting.

## Stored Responses

Set `store: true` to persist a response in [AWS Bedrock session storage](https://docs.aws.amazon.com/bedrock/latest/userguide/sessions.html): one AWS-managed session per stored response, encrypted at rest (optionally with [your own KMS key](operations_configuration.md#aws-bedrock-session-encryption-key-arn)), with no state on the server itself.

```bash
curl -X POST "$BASE/v1/responses" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "amazon.nova-micro-v1:0", "input": "Hello!", "store": true}'
```

The returned `id` then works with:

- `GET /v1/responses/{response_id}` — retrieve the stored response.
- `GET /v1/responses/{response_id}/input_items` — list the input items that produced it.
- `DELETE /v1/responses/{response_id}` — delete it (and its AWS Bedrock session).
- `POST /v1/responses/{response_id}/cancel` — present for API parity; always fails with the OpenAI synchronous-response error since `background=true` is not supported.
- `previous_response_id` on a new request — continue the conversation: the stored input and output are automatically prepended to the new input (instructions are not carried over, per the OpenAI API).

!!! note "Behavior notes"
    - `store` defaults to **false** on this implementation (the OpenAI API defaults to true).
    - `store=true` is ignored with `stream=true` (a warning is recorded in the request log).
    - Sessions are created in the primary Bedrock region and persist until deleted through the API.
    - Requires the AWS Bedrock session management IAM permissions (`bedrock:CreateSession`, `bedrock:CreateInvocation`, `bedrock:PutInvocationStep`, `bedrock:GetInvocationStep`, `bedrock:ListInvocationSteps`, `bedrock:ListInvocations`, `bedrock:ListSessions`, `bedrock:ListTagsForResource`, `bedrock:EndSession`, `bedrock:DeleteSession`, `bedrock:TagResource`). Without them, `store=true` is ignored (with a request-log warning) and the response is not persisted.

## Conversation Compaction

Compact a long conversation into a single `compaction` item to keep multi-turn sessions within the context window. The model summarises the provided `input`; the summary comes back as an opaque item that you include in the `input` of later requests instead of the full history.

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

**Response:**

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
    The compaction content is fully self-contained (encoded, not encrypted): no conversation state is needed, and any server instance can expand it. `previous_response_id` may reference a [stored response](#stored-responses) to include its conversation in the compaction.

---

**Ready to build with AI?** Check out the [Models API](api_openai_models.md) to see all available foundation models!
