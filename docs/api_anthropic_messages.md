---
title: Messages API - AWS Bedrock with Anthropic Compatibility
description: Anthropic-compatible Messages API for AWS Bedrock models including Claude, Nova, Llama. Supports streaming, extended thinking, tool calling, prompt caching, and multi-modal inputs.
keywords: anthropic messages API, claude messages API, AWS Bedrock chat, streaming messages API, AI assistant API, anthropic API, tool calling API, multi-modal messages
---

# Messages API

Generate conversational AI responses with AWS Bedrock foundation models—including Claude, Nova, Llama, and more—through an Anthropic-compatible Messages API interface.

!!! warning "Route Prefix"
    By default, all Anthropic-compatible routes are prefixed with `/anthropic`. This means the Messages API is available at `/anthropic/v1/messages` instead of `/v1/messages`. You can customize this prefix using the `ANTHROPIC_ROUTES_PREFIX` configuration variable documented in [Operations Configuration](operations_configuration.md#anthropic-routes-prefix).

## Why Choose Messages API?

<div class="grid cards" markdown>

- :material-brain: __Multiple Models__
  <br>Access models from Anthropic, Amazon, Meta, and more through one API. Choose the best model for your task without vendor lock-in.

- :material-image-multiple: __Multi-Modal__
  <br>Process text, images, videos, and documents together. Support for URLs, data URIs, and direct S3 references.

- :material-shield-check: __Built-In Safety__
  <br>AWS Bedrock Guardrails provide content filtering and safety policies.

- :material-aws: __AWS Scale & Reliability__
  <br>Run on AWS infrastructure with service tiers for optimized latency. Multi-region model access for availability and performance.

</div>

## Quick Start: Available Endpoints

| Endpoint                    | Method | What It Does                               | Powered By                  |
|-----------------------------|--------|--------------------------------------------|-----------------------------|
| `/v1/messages`              | POST   | Conversational AI with multi-modal support | AWS Bedrock Converse API    |
| `/v1/messages/count_tokens` | POST   | Count tokens in a message without sending  | AWS Bedrock CountTokens API |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                               |                  Status                  | Notes                                                                      |
|---------------------------------------|:----------------------------------------:|----------------------------------------------------------------------------|
| **Messages & Roles**                  |                                          |                                                                            |
| Text messages                         |   :material-check-circle:{ .success }    | Full support for all text content                                          |
| Image input (`image`)                 |       :material-cog:{ .model-dep }       | HTTP URLs, data URIs, base64                                               |
| Document input (`document`)           |       :material-cog:{ .model-dep }       | PDF (base64/URL), plain text, content blocks                               |
| Document citations                    |       :material-cog:{ .model-dep }       | Citation locations in responses (PDF only on some models)                  |
| Search result input (`search_result`) |   :material-check-circle:{ .success }    | Pass search results as context                                             |
| System messages                       |       :material-cog:{ .model-dep }       | System prompts                                                             |
| Image & Document input from S3        | :material-plus-circle:{ .extra-feature } | S3 URLs                                                                    |
| **Tool Calling**                      |                                          |                                                                            |
| Tool use (`tools`)                    |       :material-cog:{ .model-dep }       | Full Anthropic-compatible schema                                           |
| Tool choice (`auto`, `any`, `tool`)   |       :material-cog:{ .model-dep }       | Control tool selection behavior                                            |
| Tool choice `none`                    |   :material-minus-circle:{ .partial }    | Remove tools from request instead                                          |
| Parallel tool calls                   |       :material-cog:{ .model-dep }       | Multiple tools in one turn                                                 |
| Web search tool (`web_search`)        |       :material-cog:{ .model-dep }       | Available on models with system tool support (e.g., Amazon Nova 2)         |
| Claude server tools                   |       :material-cog:{ .model-dep }       | Bash, text editor, computer use (Claude 3.5+), memory (Claude 3.7-4.5) |
| **Generation Control**                |                                          |                                                                            |
| `max_tokens`                          |   :material-check-circle:{ .success }    | Output length limits (required)                                            |
| `temperature`                         |       :material-cog:{ .model-dep }       | Mapped to Bedrock inference params                                         |
| `top_p`                               |       :material-cog:{ .model-dep }       | Nucleus sampling control                                                   |
| `top_k`                               |       :material-cog:{ .model-dep }       | Top-k sampling control                                                     |
| `stop_sequences`                      |       :material-cog:{ .model-dep }       | Custom stop strings                                                        |
| Thinking                              |       :material-cog:{ .model-dep }       |                                                                            |
| Prompt caching                        |       :material-cog:{ .model-dep }       | Cache prompts to reduce costs and latency                                  |
| Extra model-specific params           | :material-plus-circle:{ .extra-feature } | Extra model-specific parameters not supported by the Anthropic API         |
| **Streaming & Output**                |                                          |                                                                            |
| Text                                  |   :material-check-circle:{ .success }    | Text messages                                                              |
| Streaming (`stream: true`)            |   :material-check-circle:{ .success }    | Server-Sent Events (SSE)                                                   |
| Thinking content                      |       :material-cog:{ .model-dep }       | Extended thinking output in content blocks                                 |
| **Usage tracking**                    |                                          |                                                                            |
| Input text tokens                     |   :material-check-circle:{ .success }    | Billing unit                                                               |
| Output tokens                         |   :material-check-circle:{ .success }    | Billing unit                                                               |
| Cache creation tokens                 |   :material-check-circle:{ .success }    | Prompt caching metrics (streaming and non-streaming)                       |
| Cache read tokens                     |   :material-check-circle:{ .success }    | Prompt caching metrics                                                     |
| **Other**                             |                                          |                                                                            |
| Metadata                              |   :material-minus-circle:{ .partial }    | Logged                                                                     |
| Bedrock Guardrails                    | :material-plus-circle:{ .extra-feature } | Content safety policies                                                    |
| Service tiers                         |   :material-check-circle:{ .success }    | Mapped to Bedrock service tiers and latency options                        |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success } **Supported** — Fully compatible with Anthropic API
* :material-cog:{ .model-dep } **Available on Select Models** — Check your model's capabilities
* :material-minus-circle:{ .partial } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature } **Extra Feature** — Enhanced capability beyond Anthropic API

</div>

## Model Support

All models supported by AWS Bedrock Converse and Converse Stream API are supported.

### ![Claude](styles/logo_anthropic_claude.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Claude Models Name Aliases

This API supports dynamic model name aliases matching the official Anthropic API. You can use Claude model names exactly as they appear in [Anthropic's documentation](https://docs.anthropic.com/en/docs/about-claude/models), and they will be automatically resolved to the corresponding AWS Bedrock model identifiers.

**Examples:**

- `claude-opus-4-6` → `anthropic.claude-opus-4-6-v1`
- `claude-sonnet-4-6` → `anthropic.claude-sonnet-4-6`
- `claude-haiku-4-5-20251001` → `anthropic.claude-haiku-4-5-20251001-v1:0`

Aliases for non-Anthropic models are also supported as normal.

## Advanced Features

### Prompt Caching

Reduce costs and improve response times by caching frequently-used prompt components across multiple requests. This feature is particularly effective for applications with consistent system prompts, tool definitions, or conversation contexts.

**Supported Models:**

- **Anthropic Claude**: Full support for system, messages, and tools caching
- **Amazon Nova**: Support for system and messages caching

!!! info "Documentation"
    See [AWS Bedrock Prompt Caching - Supported Models](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html#prompt-caching-models) for the complete list of models supporting prompt caching.

!!! warning "Cache Creation Costs"
    Cache creation incurs a higher cost than regular token processing. Only use prompt caching when you expect a high cache hit ratio across multiple requests with similar prompts.

**How to Use:**

Add `cache_control` blocks to the content you want to cache:

```bash
curl -X POST "$BASE/v1/messages" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-opus-4-6-v1",
    "max_tokens": 1024,
    "system": [
      {
        "type": "text",
        "text": "You are a helpful assistant with extensive knowledge...",
        "cache_control": {"type": "ephemeral"}
      }
    ],
    "messages": [
      {"role": "user", "content": "What is 2 + 2?"}
    ]
  }'
```

**Granular Cache Control:**

Enable caching for specific sections by adding `cache_control` blocks:

- **System messages**: Add to system text blocks
- **Messages**: Add to the last message content block you want cached
- **Tools**: Add to the last tool definition you want cached (Anthropic Claude only)

```json
{
  "model": "anthropic.claude-opus-4-6-v1",
  "max_tokens": 1024,
  "system": [
    {
      "type": "text",
      "text": "System instructions...",
      "cache_control": {"type": "ephemeral"}
    }
  ],
  "tools": [
    {
      "name": "get_weather",
      "description": "Get weather data",
      "input_schema": {...},
      "cache_control": {"type": "ephemeral"}
    }
  ],
  "messages": [...]
}
```

**Benefits:**

- **Cost Reduction**: Cached tokens are billed at a lower rate than regular input tokens
- **Lower Latency**: Cached prompts eliminate reprocessing time
- **Automatic Management**: The API handles cache invalidation and updates

**Usage Tracking:**

Cached token usage is reported in the response:

```json
{
  "usage": {
    "input_tokens": 300,
    "cache_creation_input_tokens": 1200,
    "cache_read_input_tokens": 0,
    "output_tokens": 100
  }
}
```

In subsequent requests with cache hits:

```json
{
  "usage": {
    "input_tokens": 300,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 1200,
    "output_tokens": 100
  }
}
```

### System Prompt

System prompts define the AI assistant's behavior, personality, and instructions (e.g., "You are a helpful assistant"). Most models support system prompts.

!!! warning "Unsupported Models"
    Some models don't support system prompts (`mistral.mistral-7b-instruct-v0:2`, `mistral.mistral-8x7b-instruct-v0:1`). By default, **stdapi.ai silently drops system messages** for these models, allowing cross-model compatibility. To receive errors instead, configure [`DROP_UNSUPPORTED_SYSTEM_PROMPT=false`](operations_configuration.md#drop-unsupported-system-prompt).

### ![AWS S3](styles/logo_amazon_s3.svg){ style="height: 1.2em; vertical-align: text-bottom;" } S3 Image Support

Access images directly from your S3 buckets without generating pre-signed URLs or downloading files locally.

**Supported Formats:**

- **Images**: JPEG, PNG, GIF, WebP

**How to Use:**

Simply reference your S3 images using the `s3://` URI scheme in image source fields:

```json
{
  "model": "anthropic.claude-opus-4-6-v1",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {
          "type": "image",
          "source": {
            "type": "url",
            "url": "s3://my-bucket/images/photo.jpg"
          }
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

### Document Input

Send documents as context for the model to analyze and reference. Supports multiple source types:

- **Base64 PDF**: Inline PDF documents encoded in base64
- **URL PDF**: PDF documents fetched from HTTP(S) URLs (downloaded server-side)
- **Plain text**: Raw text content as documents
- **Content blocks**: Structured content with text and images

Enable `citations` on document blocks to get precise source references in responses:

```json
{
  "type": "document",
  "source": {
    "type": "text",
    "media_type": "text/plain",
    "data": "The capital of France is Paris."
  },
  "title": "Geography",
  "citations": {"enabled": true}
}
```

!!! note "Citation Support"
    Citation support varies by model and document format. PDF documents generally have the best citation support across models.

### Server Tools

Server tools are built-in capabilities that foundation models can use directly without requiring you to implement backend integrations. Different model providers support different server tools through their native tool formats.

#### Web Search Tool

The Anthropic `web_search` tool is supported on models that declare web search as a system tool. When you include a `web_search` tool in your request, it is automatically mapped to the model's native system tool (e.g., `nova_grounding` for Amazon Nova 2 models).

**Supported Models:**

- ![Amazon Nova](styles/logo_amazon_nova.svg){ style="height: 1.2em; vertical-align: text-bottom;" } **Amazon Nova 2** (e.g., `amazon.nova-2-lite-v1:0`) and **Amazon Nova Premier** (`amazon.nova-premier-v1:0`): Mapped to `nova_grounding`

**Usage:**

```bash
curl -X POST "$BASE/v1/messages" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-2-lite-v1:0",
    "messages": [
      {"role": "user", "content": "What are the latest news today?"}
    ],
    "tools": [
      {"type": "web_search_20250305", "name": "web_search"}
    ]
  }'
```

!!! note "Model Compatibility"
    Requesting `web_search` on a model that does not support it will return a `400 Bad Request` error.

#### ![Claude](styles/logo_anthropic_claude.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Claude Server Tools

Anthropic Claude models support server-side tools that are executed by the model provider. These tools are passed through to Bedrock via `additionalModelRequestFields` in their native Anthropic JSON format.

**Supported Tools by Model:**

| Tool | Claude 3.5 Sonnet v2 | Claude 3.7+ |
|------|:---------------------:|:-----------:|
| `bash` | :material-check-circle:{ .success } | :material-check-circle:{ .success } |
| `text_editor` (`str_replace_editor`) | :material-check-circle:{ .success } | :material-check-circle:{ .success } |
| `computer` | :material-check-circle:{ .success } | :material-check-circle:{ .success } |
| `memory` | :material-close-circle:{ .unsupported } | :material-check-circle:{ .success } |

**Usage:**

```bash
curl -X POST "$BASE/v1/messages" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "anthropic-beta: computer-use-2025-01-24" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "max_tokens": 4096,
    "messages": [
      {"role": "user", "content": "Run a Python script that prints hello world."}
    ],
    "tools": [
      {"type": "bash_20250124", "name": "bash"},
      {"type": "text_editor_20250124", "name": "str_replace_editor"}
    ]
  }'
```

!!! tip "Beta Headers"
    Claude server tools require specific `anthropic-beta` flags on Bedrock. These flags are **automatically injected** when the corresponding server tools are included in the request — no manual header required:

    - `bash`, `text_editor`, `computer` → `computer-use-2024-10-22` (Claude 3.5) or `computer-use-2025-01-24` (Claude 3.7+)
    - `memory` → `context-management-2025-06-27` (Claude 3.7-4.5)

    You can still pass additional `anthropic-beta` flags via the HTTP header or request body for non-tool beta features (e.g., `output-128k-2025-02-19`).

!!! note "Model Compatibility"
    Requesting a server tool on a model that does not support it will return a `400 Bad Request` error. Non-Claude models do not support these tools.

##### Unsupported Anthropic Server Tools

The following Anthropic server tools are **not supported** via Bedrock:

- `code_execution` — Code execution sandbox
- `web_search` — Web search (only available on Nova Premier via `nova_grounding`)
- `web_fetch` — Web page fetching
- `tool_search` — Tool search
- `container_upload` — Container file upload

Requests using these tools on any Claude model will return a `400 Bad Request` error.

### Provider-Specific Parameters

Unlock advanced model capabilities by passing provider-specific parameters directly in your requests. These parameters are forwarded to AWS Bedrock and allow you to access features unique to each foundation model provider.

!!! info "Documentation"
    See [Bedrock Model Parameters](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html) for the complete list of available parameters per model.

**How It Works:**

Add provider-specific fields at the top level of your request body alongside standard Anthropic parameters. The API automatically forwards these to the appropriate model provider via AWS Bedrock.

**Configuration Options:**

**Option 1: Per-Request**

Add provider-specific parameters directly in your request body.

**Option 2: Server-Wide Defaults**

Configure default parameters for specific models via the `DEFAULT_MODEL_PARAMS` environment variable:

```bash
export DEFAULT_MODEL_PARAMS='{
  "anthropic.claude-sonnet-4-5-20250929-v1:0": {
    "anthropic_beta": ["extended-thinking-2024-12-12"]
  }
}'
```

!!! tip "Parameter Priority"
    Per-request parameters override server-wide defaults.

**Behavior:**

- ✅ **Compatible parameters**: Forwarded to the model and applied
- ⚠️ **Unsupported parameters**: Return HTTP 400 with an error message

#### ![Claude](styles/logo_anthropic_claude.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Anthropic Claude Features

Enable cutting-edge Claude capabilities including extended thinking and reasoning.

##### Extended Thinking

Enable extended thinking by passing the `anthropic-beta` header, just like the official Anthropic API:

```bash
curl -X POST "$BASE/v1/messages" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "anthropic-beta: extended-thinking-2024-12-12" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "max_tokens": 1024,
    "messages": [{"role":"user","content":"Solve a complex problem"}]
  }'
```

**Response with Thinking:**

When extended thinking is enabled, the response includes thinking content blocks:

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "thinking",
      "thinking": "Let me think about this step by step..."
    },
    {
      "type": "text",
      "text": "Here's the solution..."
    }
  ],
  "usage": {...}
}
```

!!! tip "Server-Wide Configuration"
    You can also configure beta flags server-wide using the `DEFAULT_MODEL_PARAMS` environment variable (see [Provider-Specific Parameters](#provider-specific-parameters)).

!!! warning "Unsupported Beta Flags"
    Unsupported flags that would change output return HTTP 400 errors.

!!! info "Documentation"
    See [Using Claude on AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html) for more details on Claude-specific parameters.

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

### Model-Specific Headers

| Header           | Purpose                        | Valid Values                                                                                       | Models           |
|------------------|--------------------------------|----------------------------------------------------------------------------------------------------|------------------|
| `anthropic-beta` | Enable Anthropic beta features | Comma-separated feature names (e.g., `extended-thinking-2024-12-12,context-management-2025-06-27`) | Anthropic Claude |

**Example with all headers:**

```bash
curl -X POST "$BASE/v1/messages" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -H "X-Amzn-Bedrock-GuardrailIdentifier: your-guardrail-id" \
  -H "X-Amzn-Bedrock-GuardrailVersion: 1" \
  -H "X-Amzn-Bedrock-Trace: enabled" \
  -H "X-Amzn-Bedrock-Service-Tier: priority" \
  -H "X-Amzn-Bedrock-PerformanceConfig-Latency: optimized" \
  -d '{
    "model": "anthropic.claude-opus-4-6-v1",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

!!! info "Detailed Documentation"
    For complete information about these headers, configuration options, and use cases, see:

    - [Bedrock Guardrails Configuration](operations_configuration.md#bedrock-guardrails)
    - [Service Tier and Performance Configuration](operations_configuration.md#bedrock-service-tier-and-performance-configuration)

## Try It Now

**Basic message:**

```bash
curl -X POST "$BASE/v1/messages" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Say hello world"}]
  }'
```

**Streaming response:**

```bash
curl -N -X POST "$BASE/v1/messages" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "max_tokens": 1024,
    "stream": true,
    "messages": [{"role": "user", "content": "Write a haiku about the sea."}]
  }'
```

**Multi-modal with image:**

```json
{
  "model": "amazon.nova-micro-v1:0",
  "max_tokens": 1024,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {
          "type": "image",
          "source": {
            "type": "url",
            "url": "https://example.com/photo.jpg"
          }
        }
      ]
    }
  ]
}
```

**With tool calling:**

```bash
curl -X POST "$BASE/v1/messages" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-opus-4-6-v1",
    "max_tokens": 1024,
    "tools": [
      {
        "name": "get_weather",
        "description": "Get weather information",
        "input_schema": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    ],
    "messages": [
      {"role": "user", "content": "What is the weather in Paris?"}
    ]
  }'
```

---

**Ready to build with AI?** Check out the [Models API](api_openai_models.md) to see all available foundation models!
