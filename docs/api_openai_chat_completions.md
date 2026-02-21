---
title: Chat Completions API - AWS Bedrock with OpenAI Compatibility
description: OpenAI-compatible chat completions API for AWS Bedrock models including Claude, Nova, Llama. Supports streaming, reasoning modes, prompt caching, and multi-modal inputs.
keywords: chat completions API, OpenAI chat API, AWS Bedrock chat, streaming chat API, AI chatbot API, Claude API, function calling API, multi-modal chat
---

# Chat Completions API

Generate conversational AI responses with AWS Bedrock foundation models—including Claude, Nova, Llama, and more—through an OpenAI-compatible interface.

## Why Choose Chat Completions?

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

## Quick Start: Available Endpoint

| Endpoint               | Method | What It Does                               | Powered By               |
|------------------------|--------|--------------------------------------------|--------------------------|
| `/v1/chat/completions` | POST   | Conversational AI with multi-modal support | AWS Bedrock Converse API |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                                  |                  Status                  | Notes                                                           |
|------------------------------------------|:----------------------------------------:|-----------------------------------------------------------------|
| **Messages & Roles**                     |                                          |                                                                 |
| Text messages                            |   :material-check-circle:{ .success }    | Full support for all text content                               |
| Image input (`image_url`)                |       :material-cog:{ .model-dep }       | HTTP, data URIs                                                 |
| Image input from S3                      | :material-plus-circle:{ .extra-feature } | S3 URLs                                                         |
| Video input                              |       :material-cog:{ .model-dep }       | Supported by select models                                      |
| Audio input                              |       :material-cog:{ .model-dep }       | Supported by select models                                      |
| Document input (`file`)                  |       :material-cog:{ .model-dep }       | PDF and document support varies by model                        |
| System messages                          |       :material-cog:{ .model-dep }       | Includes `developer` role                                       |
| **Tool Calling**                         |                                          |                                                                 |
| Function calling (`tools`)               |       :material-cog:{ .model-dep }       | Full OpenAI-compatible schema                                   |
| Legacy `function_call`                   |       :material-cog:{ .model-dep }       | Backward compatibility maintained                               |
| Parallel tool calls                      |       :material-cog:{ .model-dep }       | Multiple tools in one turn                                      |
| Disable Parallel tool calls              | :material-close-circle:{ .unsupported }  | Parallel tool calls are always on                               |
| Non-function tool types                  | :material-close-circle:{ .unsupported }  | Only function tools supported                                   |
| System tools (`systemTool_*`)            | :material-plus-circle:{ .extra-feature } | AWS Bedrock system tools (e.g., web grounding with citations)   |
| **Generation Control**                   |                                          |                                                                 |
| `max_tokens` / `max_completion_tokens`   |   :material-check-circle:{ .success }    | Output length limits                                            |
| `temperature`                            |       :material-cog:{ .model-dep }       | Mapped to Bedrock inference params                              |
| `top_p`                                  |       :material-cog:{ .model-dep }       | Nucleus sampling control                                        |
| `stop` sequences                         |       :material-cog:{ .model-dep }       | Custom stop strings                                             |
| `frequency_penalty` / `presence_penalty` |       :material-cog:{ .model-dep }       | Repetition control                                              |
| `seed`                                   |       :material-cog:{ .model-dep }       | Deterministic generation                                        |
| `logit_bias`                             |       :material-cog:{ .model-dep }       | Not all models support biasing                                  |
| `top_logprobs`                           |       :material-cog:{ .model-dep }       | Token probability output                                        |
| `top_k` (From Qwen API)                  |       :material-cog:{ .model-dep }       | Candidate token set size for sampling                           |
| `reasoning_effort`                       |       :material-cog:{ .model-dep }       | Reasoning control (minimal/low/medium/high)                     |
| `enable_thinking` (From Qwen API)        |       :material-cog:{ .model-dep }       | Enable thinking mode                                            |
| `thinking_budget` (From Qwen API)        |       :material-cog:{ .model-dep }       | Thinking token budget                                           |
| `n` (multiple choices)                   |   :material-minus-circle:{ .partial }    | Generate multiple responses, not supported with streaming       |
| `logprobs`                               | :material-close-circle:{ .unsupported }  | Log probabilities                                               |
| `prediction`                             | :material-close-circle:{ .unsupported }  | Static predicted output content                                 |
| `response_format`                        | :material-close-circle:{ .unsupported }  | Response format specification                                   |
| `verbosity`                              | :material-close-circle:{ .unsupported }  | Model verbosity                                                 |
| `web_search_options`                     | :material-close-circle:{ .unsupported }  | Web search tool                                                 |
| `prompt_cache_key`                       |       :material-cog:{ .model-dep }       | Cache prompts to reduce costs and latency                       |
| Extra model-specific params              | :material-plus-circle:{ .extra-feature } | Extra model-specific parameters not supported by the OpenAI API |
| **Streaming & Output**                   |                                          |                                                                 |
| Text                                     |   :material-check-circle:{ .success }    | Text messages                                                   |
| Streaming (`stream: true`)               |   :material-check-circle:{ .success }    | Server-Sent Events (SSE)                                        |
| Streaming obfuscation                    | :material-close-circle:{ .unsupported }  | Unsupported                                                     |
| Audio                                    |   :material-check-circle:{ .success }    | Model output or synthesis from text output                      |
| `response_format` (JSON mode)            |       :material-cog:{ .model-dep }       | Model-specific JSON support                                     |
| `reasoning_content` (From Deepseek API)  |       :material-cog:{ .model-dep }       | Text reasoning messages                                         |
| `annotations` (URL citations)            |   :material-check-circle:{ .success }    | URL citations from system tools (non-streaming only)            |
| **Usage tracking**                       |                                          |                                                                 |
| Input text tokens                        |   :material-check-circle:{ .success }    | Billing unit                                                    |
| Output tokens                            |   :material-check-circle:{ .success }    | Billing unit                                                    |
| Reasoning tokens                         |   :material-minus-circle:{ .partial }    | Estimated                                                       |
| **Other**                                |                                          |                                                                 |
| Service tiers                            |   :material-check-circle:{ .success }    | Mapped to Bedrock service tiers and latency options             |
| `store` / `metadata`                     | :material-close-circle:{ .unsupported }  | OpenAI-specific features                                        |
| `safety_identifier` / `user`             |   :material-minus-circle:{ .partial }    | Logged                                                          |
| Bedrock Guardrails                       | :material-plus-circle:{ .extra-feature } | Content safety policies                                         |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep } **Available on Select Models** — Check your model's capabilities
* :material-minus-circle:{ .partial } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature } **Extra Feature** — Enhanced capability beyond OpenAI API

</div>

## Model Support

All models supported by AWS Bedrock Converse and Converse Stream API are supported.

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

Set the `prompt_cache_key` parameter to enable caching:

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-opus-4-6-v1",
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
  "model": "anthropic.claude-opus-4-6-v1",
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
    Cache retention configuration is only available on select models. See [AWS Bedrock Prompt Caching - Supported Models](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html#prompt-caching-models) for details on which models support configurable TTL.

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-opus-4-6-v1",
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

- `"in-memory"` - Short-term caching (mapped to 5 minutes on AWS Bedrock)
- `"24h"` - Long-term caching (mapped to 1 hour on AWS Bedrock)
- Additional AWS Bedrock values: `"1h"`, `"5m"` (provider-specific)

!!! note "OpenAI to AWS Bedrock Mapping"
    OpenAI retention values are mapped to AWS Bedrock equivalents for compatibility:

    - `"in-memory"` → 5 minutes
    - `"24h"` → 1 hour

**Usage Tracking:**

Cached token usage is reported in the response:

```json
{
  "usage": {
    "prompt_tokens": 1500,
    "completion_tokens": 100,
    "total_tokens": 1600,
    "prompt_tokens_details": {
      "cached_tokens": 1200
    }
  }
}
```

In this example, 1,200 tokens were retrieved from cache, with only 300 tokens requiring processing.

### System Prompt

System prompts define the AI assistant's behavior, personality, and instructions (e.g., "You are a helpful assistant"). Most models support system prompts.

!!! warning "Unsupported Models"
    Some models don't support system prompts (`mistral.mistral-7b-instruct-v0:2`, `mistral.mistral-8x7b-instruct-v0:1`). By default, **stdapi.ai silently drops system messages** for these models, allowing cross-model compatibility. To receive errors instead, configure [`DROP_UNSUPPORTED_SYSTEM_PROMPT=false`](operations_configuration.md#drop-unsupported-system-prompt).

### ![AWS S3](styles/logo_amazon_s3.svg){ style="height: 1.2em; vertical-align: text-bottom;" } S3 Image Support

Access images directly from your S3 buckets without generating pre-signed URLs or downloading files locally.

**Supported Formats:**

- **Images**: JPEG, PNG, GIF, WebP

**How to Use:**

Simply reference your S3 images using the `s3://` URI scheme in `image_url` fields:

```json
{
  "model": "anthropic.claude-opus-4-6-v1",
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

### AWS Bedrock System Tools

AWS Bedrock system tools are built-in capabilities that foundation models can use directly without requiring you to implement backend integrations. Access any AWS Bedrock system tool by adding the `systemTool_` prefix to its name—this works for current tools and any future system tools AWS releases.

**How to Use:**

Add system tools to your `tools` array using the `systemTool_` prefix followed by the tool name. System tools don't require parameter definitions—just specify the tool name and the model will handle the rest.

As AWS releases new system tools, simply use the same `systemTool_` prefix pattern to access them.

#### ![Amazon Nova](styles/logo_amazon_nova.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Nova Web Grounding

Amazon Nova Web Grounding enables models to search the web for current information, helping answer questions requiring real-time data like news, weather, product availability, or recent events. The model automatically determines when to use web grounding based on the user's query.

!!! info "Learn More"
    - [Amazon Nova Web Grounding - User Guide](https://docs.aws.amazon.com/nova/latest/userguide/grounding.html)
    - [Build More Accurate AI Applications with Amazon Nova Web Grounding - Blog Post](https://aws.amazon.com/fr/blogs/aws/build-more-accurate-ai-applications-with-amazon-nova-web-grounding/)

**Usage:**

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-premier-v1:0",
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
          "name": "systemTool_nova_grounding"
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
    Citations are only available in non-streaming responses. The OpenAI API does not support annotations in streaming mode.

**Use Cases:**

- **Current Events**: Get up-to-date information about news, weather, stock prices, or sports scores
- **Dynamic Data**: Query information that changes frequently like AWS service availability or product prices
- **Verification**: Cross-reference facts with current web sources for improved accuracy
- **Knowledge Extension**: Supplement model training data with real-time information

**Benefits:**

- **Zero Integration**: No need to implement or maintain web search APIs
- **Automatic Invocation**: Models intelligently decide when to use web grounding
- **Enhanced Accuracy**: Reduce hallucinations with real-time information retrieval
- **OpenAI-Compatible**: Works seamlessly with standard tool calling patterns

!!! warning "Model and Region Compatibility"
    **Model**: Only Amazon Nova Premier (`amazon.nova-premier-v1:0`) supports the `systemTool_nova_grounding` tool.

    **Region**: Web Grounding is only available in US regions.

### Provider-Specific Parameters

Unlock advanced model capabilities by passing provider-specific parameters directly in your requests. These parameters are forwarded to AWS Bedrock and allow you to access features unique to each foundation model provider.

!!! info "Documentation"
    See [Bedrock Model Parameters](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html) for the complete list of available parameters per model.

**How It Works:**

Add provider-specific fields at the top level of your request body alongside standard OpenAI parameters. The API automatically forwards these to the appropriate model provider via AWS Bedrock.

**Examples:**

**Top K Sampling:**
```json
{
  "model": "anthropic.claude-opus-4-6-v1",
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

##### Beta Feature Flags

Enable experimental Claude features like extended thinking by adding the `anthropic_beta` array to your request:

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "messages": [{"role":"user","content":"Summarize the news headline."}],
    "anthropic_beta": ["Interleaved-thinking-2025-05-14"]
  }'
```

!!! tip "Server-Wide Configuration"
    You can also configure beta flags server-wide using the `DEFAULT_MODEL_PARAMS` environment variable (see [Provider-Specific Parameters](#provider-specific-parameters)).

!!! warning "Unsupported Beta Flags"
    Unsupported flags that would change output return HTTP 400 errors.

!!! info "Documentation"
    See [Using Claude on AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html) for more details on Claude-specific parameters.


### Reasoning Control

This API supports two different approaches to control [AWS Bedrock reasoning](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-reasoning.html) behavior. Reasoning enables foundation models to break down complex tasks into smaller steps ("chain of thought"), improving accuracy for multi-step analysis, math problems, and complex reasoning tasks. Both approaches work with all AWS Bedrock models that support reasoning capabilities.

!!! info "Model Support for Configurable Reasoning"
    Not all reasoning-capable models support configurable reasoning control. Support varies by model:

    - **Anthropic Claude 3.7 - 4.5**: Both `reasoning_effort` and `thinking_budget` parameters supported (token budget-based reasoning)
    - **Anthropic Claude Opus 4.6+**: `reasoning_effort` parameter only (adaptive reasoning)
    - **Amazon Nova 2 models**: `reasoning_effort` parameter only
    - **DeepSeek V3 models**: `reasoning_effort` parameter only

#### ![OpenAI](styles/logo_openai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } OpenAI-Style reasoning parameters

Use the `reasoning_effort` parameter with predefined effort levels. This approach works with all AWS Bedrock models that support reasoning, providing a simple way to control reasoning depth.

**Available Levels:**

- `minimal` - Quick responses with minimal reasoning
- `low` - Light reasoning for straightforward tasks
- `medium` - Balanced reasoning for most use cases
- `high` - Deep reasoning for complex problems
- `xhigh` - Maximum reasoning for complex problems

**Example:**

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-opus-4-6-v1",
    "reasoning_effort": "high",
    "messages": [{"role": "user", "content": "Solve this complex problem..."}]
  }'
```

#### ![Qwen](styles/logo_qwen.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Qwen-Style reasoning parameters

Use explicit `enable_thinking` & `thinking_budget` parameters for fine-grained control over thinking mode. This approach works with all AWS Bedrock models that support reasoning, offering precise control over reasoning behavior and token budgets.

**Parameters:**

- `enable_thinking` (boolean): Enable or disable thinking mode
    - Default: Model-specific (usually `false`)
    - Some models have reasoning always enabled
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
    "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "enable_thinking": true,
    "thinking_budget": 2000,
    "messages": [{"role": "user", "content": "Solve this complex problem..."}]
  }'
```

**Using Python SDK:**

```python
from openai import OpenAI

client = OpenAI(
    api_key="your-api-key",
    base_url="https://your-endpoint/v1"
)

# OpenAI-style reasoning (predefined effort levels)
response = client.chat.completions.create(
    model="anthropic.claude-opus-4-6-v1",
    reasoning_effort="high",
    messages=[{"role": "user", "content": "Complex problem..."}]
)

# Qwen-style reasoning (fine-grained control)
response = client.chat.completions.create(
    model="anthropic.claude-sonnet-4-5-20250929-v1:0",
    messages=[{"role": "user", "content": "Complex problem..."}],
    extra_body={
        "enable_thinking": True,
        "thinking_budget": 2000
    }
)
```

!!! note "Reasoning Output"
    Models that support reasoning will include their thinking process in `reasoning_content` fields in the response.

#### ![DeepSeek](styles/logo_deepSeek.svg){ style="height: 1.2em; vertical-align: text-bottom;" } DeepSeek reasoning responses

DeepSeek models with reasoning capabilities are automatically handled—their chain-of-thought reasoning appears in `reasoning_content` fields without any special configuration, just like DeepSeek's native chat completions endpoint.

!!! info "Documentation"
    See [DeepSeek API - Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion) for more information about DeepSeek's reasoning capabilities.

**What You Get:**

- **Automatic reasoning**: DeepSeek reasoning models automatically include their thinking process
- **`reasoning_content` field**: Receive visible reasoning text in assistant messages
- **Streaming support**: Get `choices[].delta.reasoning_content` chunks in real-time as the model thinks
- **Compatible format**: Uses the same DeepSeek-compatible response format

**How It Works:**

- When using DeepSeek reasoning models, the API automatically surfaces their chain-of-thought
- Non-reasoning models simply omit the `reasoning_content` field
- No special parameters needed—just use the model and reasoning appears automatically

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
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Amzn-Bedrock-GuardrailIdentifier: your-guardrail-id" \
  -H "X-Amzn-Bedrock-GuardrailVersion: 1" \
  -H "X-Amzn-Bedrock-Trace: enabled" \
  -H "X-Amzn-Bedrock-Service-Tier: priority" \
  -H "X-Amzn-Bedrock-PerformanceConfig-Latency: optimized" \
  -d '{
    "model": "anthropic.claude-opus-4-6-v1",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

!!! info "Detailed Documentation"
    For complete information about these headers, configuration options, and use cases, see:

    - [Bedrock Guardrails Configuration](operations_configuration.md#bedrock-guardrails)
    - [Service Tier and Performance Configuration](operations_configuration.md#bedrock-service-tier-and-performance-configuration)

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
    "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
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
