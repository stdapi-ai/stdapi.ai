---
title: Messages API - Amazon Bedrock with Anthropic Compatibility
description: Anthropic-compatible Messages API for Amazon Bedrock models including Claude, Nova, Llama. Supports streaming, extended thinking, tool calling, prompt caching, and multi-modal inputs.
keywords: anthropic messages API, claude messages API, AWS Bedrock chat, streaming messages API, AI assistant API, anthropic API, tool calling API, multi-modal messages
---

# Messages API (Anthropic Compatible)

Generate conversational AI responses with Amazon Bedrock foundation models—including Claude, Nova, Llama, and more—through an Anthropic-compatible Messages API interface.

!!! warning "Route Prefix & Base URL"
    By default, all Anthropic-compatible routes are prefixed with `/anthropic`. This means the Messages API is available at `/anthropic/v1/messages` instead of `/v1/messages`. You can customize this prefix using the `ANTHROPIC_ROUTES_PREFIX` configuration variable documented in [Operations Configuration](operations_configuration.md#anthropic-routes-prefix).

    The `curl` examples below use a `$BASE` variable that **must include this prefix** — set it to your scheme and host followed by `ANTHROPIC_ROUTES_PREFIX`:

    ```bash
    export BASE="https://your-host/anthropic"  # <scheme>://<host> + ANTHROPIC_ROUTES_PREFIX
    ```

## Why Choose Messages API?

<div class="grid cards" markdown>

- :material-brain: __Multiple Models__
  <br>Access models from Anthropic, Amazon, Meta, and more through one API. Choose the best model for your task without vendor lock-in.

- :material-image-multiple: __Multi-Modal__
  <br>Process text, images, videos, and documents together. Support for URLs, data URIs, and direct S3 references.

- :material-shield-check: __Built-In Safety__
  <br>Bedrock Guardrails provide content filtering and safety policies.

- :material-aws: __AWS Scale & Reliability__
  <br>Run on AWS infrastructure with service tiers for optimized latency. Multi-region model access for availability and performance.

</div>

## Quick Start: Available Endpoints

| Endpoint                    | Method | What It Does                               | Powered By                                                                    | MCP Tool                         |
|-----------------------------|--------|--------------------------------------------|-------------------------------------------------------------------------------|----------------------------------|
| `/v1/messages`              | `POST` | Conversational AI with multi-modal support | Bedrock Converse API · [Amazon Bedrock Mantle](features.md#bedrock-mantle-models) | `anthropic_message`              |
| `/v1/messages/count_tokens` | `POST` | Count tokens in a message without sending  | Bedrock CountTokens API · Bedrock Mantle                                      | `anthropic_message_count_tokens` |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                               |                  Status                  | Notes                                                                                        |
|---------------------------------------|:----------------------------------------:|----------------------------------------------------------------------------------------------|
| **Messages & Roles**                  |                                          |                                                                                              |
| Text messages                         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Full support for all text content                                                            |
| Image input (`image`)                 |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | HTTP URLs, data URIs, base64                                                                 |
| Document input (`document`)           |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | PDF (base64/URL), plain text, content blocks                                                 |
| Document citations                    |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Citation locations in responses (PDF only on some models)                                    |
| Search result input (`search_result`) |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Pass search results as context                                                               |
| System messages                       |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | System prompts                                                                               |
| Image & Document input from S3        | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | S3 URLs                                                                                      |
| Files API (`file_id`)                 |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Reference uploaded files in document/image sources — see [Files API](api_anthropic_files.md) |
| **Tool Calling**                      |                                          |                                                                                              |
| Tool use (`tools`)                    |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Full Anthropic-compatible schema                                                             |
| Tool choice (`auto`, `any`, `tool`)   |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Control tool selection behavior                                                              |
| Tool choice `none`                    |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Remove tools from request instead                                                            |
| Parallel tool calls                   |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Multiple tools in one turn                                                                   |
| Web search tool (`web_search`)        |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Available on models with system tool support (e.g., Amazon Nova 2)                           |
| Claude server tools                   |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Bash, text editor, computer use (Claude 3.5+), memory (Claude 3.7+)                          |
| **Generation Control**                |                                          |                                                                                              |
| `max_tokens`                          |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Output length limits. Optional on this gateway (divergence from the Anthropic API, which requires it): the model's default output limit applies when omitted |
| `temperature`                         |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Mapped to Bedrock inference params                                                           |
| `top_p`                               |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Nucleus sampling control                                                                     |
| `top_k`                               |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Top-k sampling control                                                                       |
| `stop_sequences`                      |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Custom stop strings                                                                          |
| Thinking                              |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       |                                                                                              |
| Prompt caching                        |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Cache prompts to reduce costs and latency                                                    |
| Extra model-specific params           | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Extra model-specific parameters not supported by the Anthropic API                           |
| **Streaming & Output**                |                                          |                                                                                              |
| Text                                  |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Text messages                                                                                |
| Streaming (`stream: true`)            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Server-Sent Events (SSE). Bedrock only reports usage in the trailing event, so `message_start.message.usage` is always `0`/`0`; read final counts from `message_delta.usage` instead |
| Thinking content                      |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Extended thinking output in content blocks                                                   |
| **Usage tracking**                    |                                          |                                                                                              |
| Input text tokens                     |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Billing unit                                                                                 |
| Output tokens                         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Billing unit                                                                                 |
| Cache creation tokens                 |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Prompt caching metrics (streaming and non-streaming)                                         |
| Cache read tokens                     |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Prompt caching metrics                                                                       |
| **Other**                             |                                          |                                                                                              |
| Metadata                              |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Converse path: logged only. Mantle path: `metadata.user_id` is forwarded upstream            |
| Bedrock Guardrails                    | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Content safety policies                                                                      |
| Service tiers                         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Mapped to Bedrock service tiers and latency options                                          |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with Anthropic API
* :material-cog:{ .model-dep role="img" aria-label="Model-dependent" } **Available on Select Models** — Check your model's capabilities
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } **Extra Feature** — Enhanced capability beyond Anthropic API

</div>

## Model Support

All models supported by the Bedrock Converse and ConverseStream APIs are supported, plus every model served by [Bedrock Mantle](features.md#bedrock-mantle-models) when enabled — including OpenAI GPT-5.x, xAI Grok, and Google Gemma 4. Requests to Mantle models are passed through natively or converted automatically depending on the model's upstream API support — see [Bedrock Mantle](#bedrock-mantle) below.

### Bedrock Mantle

Mantle-only Claude models are passed through to the upstream Anthropic Messages API; other Mantle models are converted to an OpenAI shape (Responses or Chat Completions). Parameter fidelity differs per path:

| Parameter | Claude passthrough | Converted to an OpenAI shape |
|-----------|--------------------|------------------------------|
| Server tools (`web_search`, `code_execution`, `bash`, `text_editor`, `computer`, …) | Forwarded verbatim (`anthropic-beta` flags are **not** auto-injected on the Mantle path — pass them yourself) | Rejected with `400` |
| `thinking` | Forwarded | Dropped on conversion (use `output_config.effort` for portable reasoning control) |
| `output_config.effort` | Forwarded | Mapped to reasoning effort |
| `output_config.format` | Fails upstream — not supported by the Mantle Messages API | `json_schema` mapped to OpenAI structured output |
| `top_k` | Forwarded | Dropped |
| `cache_control` markers | Forwarded (prompt caching preserved) | Dropped |
| `stop_sequences` | Forwarded | Dropped when served via the Responses API |
| `metadata.user_id` | Forwarded | Forwarded, SHA-256-hashed when over 64 characters |
| `service_tier` | Forwarded | Only `auto` is forwarded |

!!! note "Workspace attribution (`anthropic-workspace`)"
    Mantle requests can be attributed to a Bedrock Workspace for cost tracking and observability with the `anthropic-workspace: <project-id>` header (a bare project ID such as `proj_abc123`, not an ARN). It is honored per-request only when [`AWS_BEDROCK_ALLOW_MANTLE_PROJECT_OVERRIDE`](operations_configuration.md#bedrock-allow-mantle-project-override) is `true`; otherwise the server default ([`AWS_BEDROCK_MANTLE_PROJECT`](operations_configuration.md#bedrock-mantle-project)) applies. This applies **only** to models served by the Bedrock Mantle endpoint — classic `bedrock-runtime` models ignore the header.

### ![Claude](styles/logo_anthropic_claude.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Claude Models Name Aliases

This API supports dynamic model name aliases matching the official Anthropic API. You can use Claude model names exactly as they appear in [Anthropic's documentation](https://docs.anthropic.com/en/docs/about-claude/models), and they will be automatically resolved to the corresponding Bedrock model identifiers.

**Examples:**

- `claude-opus-5` → `anthropic.claude-opus-5`
- `claude-sonnet-5` → `anthropic.claude-sonnet-5`
- `claude-haiku-4-5-20251001` → `anthropic.claude-haiku-4-5-20251001-v1:0`

For Claude 4 and later, a date-stripped shortcut (e.g. `claude-haiku-4-5`) is also accepted and resolves to the latest dated variant.

Aliases for non-Anthropic models are also supported as normal.

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

Add `cache_control` blocks to the content you want to cache:

```bash
curl -X POST "$BASE/v1/messages" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-fable-5",
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
  "model": "anthropic.claude-fable-5",
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
    Some models don't support system prompts (`mistral.mistral-7b-instruct-v0:2`, `mistral.mixtral-8x7b-instruct-v0:1`). By default, **stdapi.ai silently drops system messages** for these models, allowing cross-model compatibility. To receive errors instead, configure [`DROP_UNSUPPORTED_SYSTEM_PROMPT=false`](operations_configuration.md#drop-unsupported-system-prompt).

### ![Amazon S3](styles/logo_amazon_s3.svg){ style="height: 1.2em; vertical-align: text-bottom;" } S3 Image Support

Access images directly from your S3 buckets without generating pre-signed URLs or downloading files locally.

**Supported Formats:**

- **Images**: JPEG, PNG, GIF, WebP

**How to Use:**

Simply reference your S3 images using the `s3://` URI scheme in image source fields:

```json
{
  "model": "anthropic.claude-fable-5",
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

### :material-file-link: Files API References (`file-id:`)

Image and document content blocks also accept the project-local `file-id:` URI scheme in their string-overloaded `source.url` and `source.data` fields, to reference a file previously uploaded via the [Anthropic Files API](api_anthropic_files.md):

```json
{
  "type": "image",
  "source": {
    "type": "url",
    "url": "file-id:file_0190c51c7de7455d9b8c2efe27dfbf67"
  }
}
```

!!! info "When to use which path"
    The Anthropic-native `{"type": "file", "file_id": "file_…"}` source (typed JSON) is unchanged and preferred for new code. The `file-id:` URI is the equivalent for the *string-overloaded* `source.url` / `source.data` fields, used alongside `s3://`, `https://`, and `data:` URIs. See [Files API → Referencing Uploaded Files](api_anthropic_files.md#referencing-uploaded-files-via-the-file-id-uri-scheme).

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

#### ![Amazon Nova](styles/logo_amazon_nova.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Nova Tools

| Tool | Anthropic Tool Name | Amazon Nova 2 | Amazon Nova Premier (legacy) |
|------|---------------------|:-------------:|:-------------------:|
| Web Grounding | `web_search` | :material-check-circle:{ .success role="img" aria-label="Supported" } | :material-check-circle:{ .success role="img" aria-label="Supported" } |
| Code Interpreter | `code_execution` | :material-check-circle:{ .success role="img" aria-label="Supported" } | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } |

##### Web Grounding

The Anthropic `web_search` tool is supported on models that declare web search as a system tool. When you include a `web_search` tool in your request, it is automatically mapped to the model's native system tool (e.g., `nova_grounding` for Amazon Nova 2 models).

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

!!! warning "Region Compatibility"
    Web grounding is only available in US Bedrock regions. To ensure all requests are routed to a US region, restrict the model using [`AWS_BEDROCK_MODEL_REGION_RESTRICT`](operations_configuration.md#bedrock-model-region-restrict):

    ```bash
    export AWS_BEDROCK_MODEL_REGION_RESTRICT='{"amazon.nova-": ["us-east-1"]}'
    ```

**Limitations:**

- **No citation text in response blocks**: Unlike native Anthropic `web_search`, the `web_search_tool_result` content block carries only the `url` and `title` of each result — never `cited_text` or `encrypted_index`. The cited content itself is reflected only through the text content of the response.
- **No streaming citation data**: Citation information is not emitted in streaming events. The `server_tool_use` block is streamed as a start event with empty input — no citation delta is produced.
- **No search filtering on non-Claude models**: Amazon's `systemTool` grounding has no equivalent for `allowed_domains`, `blocked_domains`, `max_uses`, or `user_location`. Requests to a system-tool web search model (e.g. Amazon Nova 2) that set any of these fields are rejected with a `400 Bad Request` rather than silently running an unfiltered search. Anthropic Claude models forward these fields natively and are unaffected.

!!! note "Model Compatibility"
    Requesting `web_search` on a model that does not support it will return a `400 Bad Request` error.

##### Code Interpreter

Amazon Nova Code Interpreter enables models to securely execute Python code in isolated sandbox environments. Enable it by passing a `code_execution` tool, which is automatically mapped to the model's native `nova_code_interpreter` system tool.

!!! info "Learn More"
    [Amazon Nova Built-in Tools - User Guide](https://docs.aws.amazon.com/nova/latest/nova2-userguide/using-tools.html#builtin-tools)

**Usage:**

```bash
curl -X POST "$BASE/v1/messages" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-2-lite-v1:0",
    "messages": [
      {"role": "user", "content": "Calculate the first 10 Fibonacci numbers."}
    ],
    "tools": [
      {"type": "code_execution_20250522", "name": "code_execution"}
    ]
  }'
```

!!! note "Model Compatibility"
    Requesting `code_execution` on a model that does not support it will return a `400 Bad Request` error.

#### ![Claude](styles/logo_anthropic_claude.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Claude Server Tools

Anthropic Claude models support server-side tools that are executed by the model provider. These tools are passed through to Bedrock via `additionalModelRequestFields` in their native Anthropic JSON format.

**Supported Tools by Model:**

| Tool | Claude 3.5 Sonnet v2 | Claude 3.7 – 4.5 | Claude 4.6+ |
|------|:---------------------:|:----------------:|:-----------:|
| `bash` | :material-check-circle:{ .success role="img" aria-label="Supported" } | :material-check-circle:{ .success role="img" aria-label="Supported" } | :material-check-circle:{ .success role="img" aria-label="Supported" } |
| `text_editor` (`str_replace_based_edit_tool` or `str_replace_editor`) | :material-check-circle:{ .success role="img" aria-label="Supported" } | :material-check-circle:{ .success role="img" aria-label="Supported" } | :material-check-circle:{ .success role="img" aria-label="Supported" } |
| `computer` | :material-check-circle:{ .success role="img" aria-label="Supported" } | :material-check-circle:{ .success role="img" aria-label="Supported" } | :material-check-circle:{ .success role="img" aria-label="Supported" } |
| `memory` | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | :material-check-circle:{ .success role="img" aria-label="Supported" } | :material-check-circle:{ .success role="img" aria-label="Supported" } |

On Claude 4.6 and later, a bare `computer` tool is promoted to the newer `computer_20251124` tool type — except on Claude Opus 5 and later, which support no computer-use tool version: there, `computer` is passed through as a regular custom tool instead of a server tool.

**Usage:**

```bash
curl -X POST "$BASE/v1/messages" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-5",
    "max_tokens": 4096,
    "messages": [
      {"role": "user", "content": "Run a Python script that prints hello world."}
    ],
    "tools": [
      {"type": "bash_20250124", "name": "bash"},
      {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"}
    ]
  }'
```

!!! tip "Beta Headers"
    Claude server tools require specific `anthropic-beta` flags on Bedrock. On the classic Bedrock (Converse) path these flags are **automatically injected** when the corresponding server tools are included in the request — no manual header required (on the [Mantle](#bedrock-mantle) path they are not auto-injected; pass them yourself):

    - `bash`, `text_editor` → `computer-use-2024-10-22` (Claude 3.5) or `computer-use-2025-01-24` (Claude 3.7+)
    - `computer` → `computer-use-2024-10-22` (Claude 3.5), `computer-use-2025-01-24` (Claude 3.7 – 4.5), or `computer-use-2025-11-24` (Claude 4.6+, tool type `computer_20251124`)
    - `memory` → `context-management-2025-06-27` (Claude 3.7+)

    You can still pass additional `anthropic-beta` flags via the HTTP header or request body for non-tool beta features (e.g., `output-128k-2025-02-19`).

!!! note "Model Compatibility"
    Requesting a server tool on a model that does not support it will return a `400 Bad Request` error. Non-Claude models do not support these tools.

##### Unsupported Anthropic Server Tools

The following Anthropic server tools are **not supported** via the classic Bedrock (Converse) path:

- `code_execution` — Code execution sandbox
- `web_search` — Web search (only available on Amazon Nova models via `nova_grounding`)
- `web_fetch` — Web page fetching
- `tool_search` — Tool search
- `container_upload` — Container file upload

Requests using these tools on Converse-served Claude models will return a `400 Bad Request` error. On [Mantle](#bedrock-mantle)-served Claude models (passthrough), server tools are instead forwarded verbatim to the upstream Messages API, which decides support; when a Mantle request must be converted to an OpenAI shape, server tools are rejected with `400`.

### Provider-Specific Parameters

Unlock advanced model capabilities by passing provider-specific parameters directly in your requests. These parameters are forwarded to Bedrock and allow you to access features unique to each foundation model provider.

!!! info "Documentation"
    See [Bedrock Model Parameters](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html) for the complete list of available parameters per model.

**How It Works:**

Add provider-specific fields at the top level of your request body alongside standard Anthropic parameters. The API automatically forwards these to the appropriate model provider via Bedrock.

**Configuration Options:**

**Option 1: Per-Request**

Add provider-specific parameters directly in your request body.

**Option 2: Server-Wide Defaults**

Configure default parameters for specific models via the `DEFAULT_MODEL_PARAMS` environment variable:

```bash
export DEFAULT_MODEL_PARAMS='{
  "anthropic.claude-sonnet-4-5-20250929-v1:0": {
    "anthropic_beta": ["output-128k-2025-02-19"]
  }
}'
```

!!! tip "Parameter Priority"
    Per-request parameters override server-wide defaults.

**Behavior:**

- :material-check-circle:{ .success role="img" aria-label="Supported" } **Compatible parameters**: Forwarded to the model and applied
- :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported parameters**: Return HTTP 400 with an error message

#### ![Claude](styles/logo_anthropic_claude.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Anthropic Claude Features

Enable cutting-edge Claude capabilities including extended thinking and reasoning.

##### Extended Thinking

Enable extended thinking with the first-class `thinking` request parameter, just like the official Anthropic API — no beta header is required:

```bash
curl -X POST "$BASE/v1/messages" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-5",
    "max_tokens": 2048,
    "thinking": {"type": "enabled", "budget_tokens": 1024},
    "messages": [{"role":"user","content":"Solve a complex problem"}]
  }'
```

`thinking` accepts `{"type": "enabled", "budget_tokens": <n>}` (the budget must be less than `max_tokens`), `{"type": "disabled"}`, or `{"type": "adaptive"}`. Alternatively, control reasoning depth with `output_config.effort` (`low`, `medium`, `high`, `xhigh`, `max`).

!!! note "`display` Not Honored"
    The `display` field (`summarized`/`omitted`) is accepted but has no effect: Bedrock's reasoning configuration has no equivalent, so full thinking text is always returned.

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
    You can also configure default model parameters server-wide using the `DEFAULT_MODEL_PARAMS` environment variable (see [Provider-Specific Parameters](#provider-specific-parameters)).

!!! warning "Unsupported Beta Flags"
    Unsupported flags that would change output return HTTP 400 errors.

!!! info "Documentation"
    See [Using Claude on Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html) for more details on Claude-specific parameters.

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

| Header           | Purpose                        | Valid Values                                                                                     | Models           |
|------------------|--------------------------------|---------------------------------------------------------------------------------------------------|------------------|
| `anthropic-beta` | Enable Anthropic beta features | Comma-separated feature names (e.g., `computer-use-2025-01-24,context-management-2025-06-27`)     | Anthropic Claude |

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
    "model": "anthropic.claude-fable-5",
    "max_tokens": 1024,
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

**Silently ignored** (no error): system prompts, tools, `top_p`, stop sequences, and prompt caching.

**Upstream format limitation:** The Anthropic Messages API does not define a `video` content block in its stable spec. To stay fully compatible with standard Anthropic clients, pass the video as an **`image`** content block with `media_type` set to the video MIME type (e.g. `video/mp4`) — the server detects the video MIME type automatically and routes it to Pegasus correctly.

**Video input formats**: `data:video/mp4;base64,…`, `https://…`, `s3://bucket/key`, or `file-id:…`. Videos above 18.75 MB are automatically uploaded to S3.

```bash
curl -X POST "$BASE/v1/messages" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "twelvelabs.pegasus-1-2-v1:0",
    "max_tokens": 1024,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image",
            "source": {"type": "url", "url": "s3://my-bucket/video.mp4"}
          },
          {"type": "text", "text": "Describe what happens in this video."}
        ]
      }
    ]
  }'
```

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
    "model": "anthropic.claude-fable-5",
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

**Count tokens (without sending a message):**

```bash
curl -X POST "$BASE/v1/messages/count_tokens" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-fable-5",
    "messages": [{"role": "user", "content": "Hello, how are you?"}]
  }'
```

**Response:**

```json
{"input_tokens": 13}
```

!!! info "Counted Request"
    The count is computed on the exact request `anthropic_message` would send for the same body: `thinking`/`output_config.effort`, server tools in their model-native form, `cache_control` breakpoints, and mid-conversation system message placement are all taken into account.

---

**Ready to build with AI?** Check out the [Models API](api_openai_models.md) to see all available foundation models!
