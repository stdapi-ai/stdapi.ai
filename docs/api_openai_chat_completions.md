# Chat Completions API

This OpenAI-compatible endpoint provides access to AWS Bedrock foundation models—including Claude, Nova, and more—through a familiar interface.

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
| Audio input                              | :material-close-circle:{ .unsupported }  | Unsupported                                                     |
| Document input (`file`)                  |       :material-cog:{ .model-dep }       | PDF and document support varies by model                        |
| System messages                          |       :material-cog:{ .model-dep }       | Includes `developer` role                                       |
| **Tool Calling**                         |                                          |                                                                 |
| Function calling (`tools`)               |       :material-cog:{ .model-dep }       | Full OpenAI-compatible schema                                   |
| Legacy `function_call`                   |       :material-cog:{ .model-dep }       | Backward compatibility maintained                               |
| Parallel tool calls                      |       :material-cog:{ .model-dep }       | Multiple tools in one turn                                      |
| Disable Parallel tool calls              | :material-close-circle:{ .unsupported }  | Parallel tool calls are always on                               |
| Non-function tool types                  | :material-close-circle:{ .unsupported }  | Only function tools supported                                   |
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
| prompt cache                             | :material-close-circle:{ .unsupported }  | Prompt cache for similar request                                |
| Extra model-specific params              | :material-plus-circle:{ .extra-feature } | Extra model-specific parameters not supported by the OpenAI API |
| **Streaming & Output**                   |                                          |                                                                 |
| Text                                     |   :material-check-circle:{ .success }    | Text messages                                                   |
| Streaming (`stream: true`)               |   :material-check-circle:{ .success }    | Server-Sent Events (SSE)                                        |
| Streaming obfuscation                    | :material-close-circle:{ .unsupported }  | Unsupported                                                     |
| Audio                                    |   :material-check-circle:{ .success }    | Synthesis from text output                                      |
| `response_format` (JSON mode)            |       :material-cog:{ .model-dep }       | Model-specific JSON support                                     |
| `reasoning_content` (From Deepseek API)  |       :material-cog:{ .model-dep }       | Text reasoning messages                                         |
| **Usage tracking**                       |                                          |                                                                 |
| Input text tokens                        |   :material-check-circle:{ .success }    | Billing unit                                                    |
| Output tokens                            |   :material-check-circle:{ .success }    | Billing unit                                                    |
| Reasoning tokens                         |   :material-minus-circle:{ .partial }    | Estimated                                                       |
| **Other**                                |                                          |                                                                 |
| Service tiers                            |   :material-check-circle:{ .success }    | Mapped to Bedrock latency options                               |
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

## Advanced Features

### ![AWS S3](styles/logo_amazon_s3.svg){ style="height: 1.2em; vertical-align: text-bottom;" } S3 Image Support

Access images directly from your S3 buckets without generating pre-signed URLs or downloading files locally.

**Supported Formats:**

- **Images**: JPEG, PNG, GIF, WebP

**How to Use:**

Simply reference your S3 images using the `s3://` URI scheme in `image_url` fields:

```json
{
  "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
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

**Requirements:**

- Your API service must have IAM permissions to read from the specified S3 buckets
- S3 objects must be in the same AWS region as the executed model or accessible via your IAM role
- Standard S3 data transfer and request costs apply

**Benefits:**

- No pre-signed URLs - Direct S3 access without generating temporary URLs
- Security - Images stay in your AWS account with IAM-controlled access
- Performance - Optimized data transfer within AWS infrastructure
- Large images - No size limitations of data URIs or base64 encoding

### AWS Bedrock Guardrails

Protect your applications with content filtering and safety policies using AWS Bedrock Guardrails. This implementation supports the same guardrails integration as AWS Bedrock's native OpenAI-compatible endpoint.

**Documentation:** [AWS Bedrock OpenAI Chat Completions API - Include a guardrail in a chat completion](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html#inference-chat-completions-guardrails)

**How to Use:**

Add guardrail headers to your chat completion requests to apply your configured safety policies:

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Amzn-Bedrock-GuardrailIdentifier: your-guardrail-id" \
  -H "X-Amzn-Bedrock-GuardrailVersion: 1" \
  -H "X-Amzn-Bedrock-Trace: ENABLED" \
  -d '{
    "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Headers:**

- **`X-Amzn-Bedrock-GuardrailIdentifier`** (required): The ID of your configured guardrail
- **`X-Amzn-Bedrock-GuardrailVersion`** (required): The version number of your guardrail
- **`X-Amzn-Bedrock-Trace`** (optional): Set to `ENABLED` to enable trace logging for debugging

**What Happens:**

- Requests are validated against your guardrail policies before reaching the model
- Responses are filtered according to your content safety rules
- Violations are blocked and return appropriate error responses

**Note:** The `tagSuffix` parameter is not supported in this implementation.

### Provider-Specific Parameters

Unlock advanced model capabilities by passing provider-specific parameters directly in your requests. These parameters are forwarded to AWS Bedrock and allow you to access features unique to each foundation model provider.

**Documentation:** [Bedrock Model Parameters](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html)

**How It Works:**

Add provider-specific fields at the top level of your request body alongside standard OpenAI parameters. The API automatically forwards these to the appropriate model provider via AWS Bedrock.

**Examples:**

**Top K Sampling:**
```json
{
  "model": "anthropic.claude-sonnet-4-5-20250929-v1:0"",
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

**Note:** Per-request parameters override server-wide defaults.

**Behavior:**

- ✅ **Compatible parameters**: Forwarded to the model and applied
- ⚠️ **Unsupported parameters**: Return HTTP 400 with an error message

### ![Claude](styles/logo_anthropic_claude.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Anthropic Claude Features

Enable cutting-edge Claude capabilities including extended thinking and reasoning.

#### Beta Feature Flags

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

**Note:** You can also configure beta flags server-wide using the `DEFAULT_MODEL_PARAMS` environment variable (see [Provider-Specific Parameters](#provider-specific-parameters)). Unsupported flags that would change output return HTTP 400 errors.

**Documentation:**

- [Using Claude on AWS Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html)


#### Reasoning Control

This API supports two different approaches to control [AWS Bedrock reasoning](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-reasoning.html) behavior. Reasoning enables foundation models to break down complex tasks into smaller steps ("chain of thought"), improving accuracy for multi-step analysis, math problems, and complex reasoning tasks. Both approaches work with all AWS Bedrock models that support reasoning capabilities.

**Option 1: ![OpenAI](styles/logo_openai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } OpenAI-Style Reasoning (`reasoning_effort`)**

Use the `reasoning_effort` parameter with predefined effort levels. This approach works with all AWS Bedrock models that support reasoning, providing a simple way to control reasoning depth.

**Available Levels:**

- `minimal` - Quick responses with minimal reasoning (25% of max tokens)
- `low` - Light reasoning for straightforward tasks (50% of max tokens)
- `medium` - Balanced reasoning for most use cases (75% of max tokens)
- `high` - Deep reasoning for complex problems (100% of max tokens)

**Example:**

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "reasoning_effort": "high",
    "messages": [{"role": "user", "content": "Solve this complex problem..."}]
  }'
```

**Option 2: ![Qwen](styles/logo_qwen.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Qwen-Style Reasoning (`enable_thinking` + `thinking_budget`)**

Use explicit parameters for fine-grained control over thinking mode. This approach works with all AWS Bedrock models that support reasoning, offering precise control over reasoning behavior and token budgets.

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
    model="anthropic.claude-sonnet-4-5-20250929-v1:0",
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

**Note:** Models that support reasoning will include their thinking process in `reasoning_content` fields in the response.

### ![DeepSeek](styles/logo_deepSeek.svg){ style="height: 1.2em; vertical-align: text-bottom;" } DeepSeek Reasoning Support

DeepSeek models with reasoning capabilities are automatically handled—their chain-of-thought reasoning appears in `reasoning_content` fields without any special configuration, just like DeepSeek's native chat completions endpoint.

**Documentation:** [DeepSeek API - Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion)

**What You Get:**

- **Automatic reasoning**: DeepSeek reasoning models automatically include their thinking process
- **`reasoning_content` field**: Receive visible reasoning text in assistant messages
- **Streaming support**: Get `choices[].delta.reasoning_content` chunks in real-time as the model thinks
- **Compatible format**: Uses the same DeepSeek-compatible response format

**How It Works:**

- When using DeepSeek reasoning models, the API automatically surfaces their chain-of-thought
- Non-reasoning models simply omit the `reasoning_content` field
- No special parameters needed—just use the model and reasoning appears automatically

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
