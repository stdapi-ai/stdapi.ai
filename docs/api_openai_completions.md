---
title: Completions API - OpenAI-Compatible Text Completion
description: OpenAI-compatible completions API for Amazon Bedrock models including Claude, Nova, Llama. Simple prompt-to-text generation with the smallest token footprint.
keywords: completions API, OpenAI completions API, text completion API, Amazon Bedrock text completion, MCP text completion
---

# Completions API

Generate text completions with Amazon Bedrock foundation models—including Claude, Nova, Llama, and more—through an OpenAI-compatible interface using the simple completions format.

!!! info "Legacy upstream, first-class here"

    OpenAI labels `/v1/completions` as **legacy** in their platform documentation and recommends new OpenAI projects migrate to `/v1/chat/completions` or `/v1/responses` for vendor compatibility. On stdapi.ai, this endpoint is a **first-class route** with the same quality guarantees as the others — its compact schema and small token footprint make it an excellent pick for MCP-based text agents and simple prompt-to-text workloads.

## Why Choose the Completions API?

<div class="grid cards" markdown>

- :material-feather: __Smallest Token Footprint__
  <br>The most compact request/response schema of the text APIs — ideal for MCP-based text agents and high-volume prompt-to-text workloads.

- :material-format-list-group: __Batch Prompts__
  <br>Send multiple independent prompts in one request and get one choice back per prompt, with streaming support.

- :material-file-link: __Multimodal Prompt Inputs__
  <br>Reference prompts and files via `https://`, `s3://`, `data:`, or `file-id:` URIs — including images, documents, audio, and video.

- :material-aws: __AWS Scale & Reliability__
  <br>Run on AWS infrastructure with service tiers and multi-region model access for availability and performance.

</div>

## Quick Start: Available Endpoint

| Endpoint          | Method | What It Does                     | Powered By                                | MCP Tool            |
|-------------------|--------|----------------------------------|-------------------------------------------|---------------------|
| `/v1/completions` | `POST`   | Simple prompt-to-text completion | Amazon Bedrock Converse API · Amazon Bedrock Mantle | `openai_completion` |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                            |                  Status                  | Notes                                                            |
|------------------------------------|:----------------------------------------:|------------------------------------------------------------------|
| **Prompt Input**                   |                                          |                                                                  |
| Single text prompt                 |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Full support for string prompts                                  |
| Multiple prompts (batch)           |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Returns one choice per prompt; rejected with `400` on Mantle-served models |
| Text + files collapse (multimodal) | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | `[text, file, …]` sent as one multimodal request with one choice |
| Prompt from URL (`https://`)       |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | HTTP URL reference                                               |
| Prompt from S3 (`s3://`)           | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | S3 URI reference                                                 |
| Prompt from data URI (`data:`)     |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Base64-encoded data URI                                          |
| Prompt from Files API (`file-id:`) | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Reference uploaded files                                         |
| Token array prompts                | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Not supported — use string prompts; rejected with `400` on Mantle-served models |
| **Generation Control**             |                                          |                                                                  |
| `max_tokens`                       |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Output length limits                                             |
| `temperature`                      |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Mapped to Bedrock inference params                               |
| `top_p`                            |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Nucleus sampling control                                         |
| `stop` sequences                   |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Custom stop strings; dropped when a Mantle request is converted to the Responses API |
| `n` (multiple choices)             |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Supported with and without streaming; `n > 1` rejected with `400` when a Mantle request is converted to the Responses or Messages API |
| `best_of`                          | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted but ignored                                             |
| `echo`                             | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted but ignored; rejected with `400` on Mantle-served models |
| `frequency_penalty`                | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted but ignored                                             |
| `presence_penalty`                 | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted but ignored                                             |
| `logit_bias`                       | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted but ignored                                             |
| `logprobs`                         | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted but ignored; rejected with `400` on Mantle-served models |
| `seed`                             | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted but ignored                                             |
| `suffix`                           | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted but ignored; rejected with `400` on Mantle-served models |
| **Streaming**                      |                                          |                                                                  |
| Streaming (`stream: true`)         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Server-Sent Events (SSE)                                         |
| `stream_options.include_usage`     |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Usage in final chunk                                             |
| Streaming with multiple prompts    |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Deltas interleave; `choices[0].index` identifies the prompt      |
| Streaming with n>1                 |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Deltas interleave; `choices[0].index` identifies each choice     |
| **Other**                          |                                          |                                                                  |
| Service tiers                      |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Mapped to Bedrock service tiers; `service_tier` and `prompt_cache_*` are not forwarded when a Mantle request is converted |
| `user` / `safety_identifier`       |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Forwarded to Amazon Bedrock as `requestMetadata`; on Mantle, `user` is forwarded as the OpenAI `user` field (as `metadata.user_id` when served via the Anthropic API) and `safety_identifier` is not forwarded |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep role="img" aria-label="Model-dependent" } **Model-Dependent** — Behavior depends on the model or backend; check the Notes column
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } **Extra Feature** — Enhanced capability beyond OpenAI API

</div>

## Prompt Input Types

_stdapi.ai extends the standard completions interface with multiple input modes:_

### String Prompt (Standard)

The simplest input — a single text prompt:

```bash
curl -X POST "$BASE/v1/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "prompt": "The capital of France is",
    "max_tokens": 20
  }'
```

### Batch Prompts

Send multiple independent prompts in a single request — the server returns one `choices[]` entry per prompt:

```bash
curl -X POST "$BASE/v1/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "prompt": ["One plus one is", "Two plus two is", "Three plus three is"],
    "max_tokens": 10
  }'
```

### Text + Files: Single-Request Multimodal Collapse

When the prompt list contains **exactly one text string and one or more file references**, stdapi.ai packs them in input order into a **single multimodal request** and returns **one choice**. The text becomes a `text` block and each file becomes an `image` / `document` / `audio` / `video` Bedrock block (content type auto-detected) — the natural "ask once using these files as context" pattern.

- **Trigger**: list with exactly one `str` element and ≥1 URL elements (`https://`, `s3://`, `data:`, `file-id:`).
- **Effect**: elements are resolved concurrently, packed as Bedrock content blocks preserving input order, and sent as a single request. You get one `Completion` choice back.
- **Requires**: a model that supports the target modalities (e.g. Claude, Nova for image / document input).
- **Unchanged otherwise**: any other shape returns one `Completion` choice per list element. Each `str` becomes a `text` block; each `InputFileUrl` becomes the content block matching its detected MIME type (image, video, audio, document). See [Files-only prompts](#files-only-prompts) below.

Image analysis with a base64 data URI (Claude handles image input):

```bash
IMAGE_B64=$(base64 < chart.png | tr -d '\n')

curl -X POST "$BASE/v1/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d @- <<JSON
{
  "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
  "prompt": [
    "Describe what is shown in this chart in one short sentence:",
    "data:image/png;base64,${IMAGE_B64}"
  ],
  "max_tokens": 80
}
JSON
```

Or, referencing previously uploaded files via `file-id:`:

```bash
curl -X POST "$BASE/v1/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
    "prompt": [
      "Summarize the main findings and cross-reference with the appendix:",
      "file-id:file-abc123",
      "file-id:file-def456"
    ],
    "max_tokens": 400
  }'
```

Input order is preserved — place the instruction before, after, or between the files as best fits your use case.

### Files-Only Prompts

When the prompt is a lone file — or a list of files without any text string — stdapi.ai forwards each file to the model using its detected modality (`image`, `video`, `audio`, `document`). No instruction is injected:

- `prompt: "<file-id or URL>"` → one request, one choice, with a single media block.
- `prompt: [<file1>, <file2>, …]` → one request per file (batch), one choice per file, each carrying its own media block.

The request reaches the model as-is; the model returns output or an error depending on whether it supports that modality (for example, Claude handles images and documents, Nova handles images, video, and documents). Use this shape for quick "what is this?" queries where the model's default behavior is enough; for tighter control over the response, prefer the collapse pattern above with an explicit instruction.

```bash
IMAGE_B64=$(base64 < screenshot.png | tr -d '\n')

curl -X POST "$BASE/v1/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d @- <<JSON
{
  "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
  "prompt": "data:image/png;base64,${IMAGE_B64}",
  "max_tokens": 120
}
JSON
```

### URL, S3, and Files API Inputs (Extension)

stdapi.ai supports additional input schemes beyond the OpenAI standard:

- **`https://`** — Load prompt from an HTTP URL
- **`s3://bucket/key/file.txt`** — Load directly from S3
- **`data:text/plain;base64,...`** — Embedded base64 data
- **`file-id:file-abc123`** — Reference files uploaded via the Files API

#### Using `file-id:` for Prompt Input

Reference files uploaded via the [Files API](api_openai_files.md) using the `file-id:` URI scheme. Useful for long prompts or when reusing the same prompt across multiple requests:

```bash
# Upload a text file
FILE_ID=$(curl -s -X POST "$BASE/v1/files" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "purpose=assistants" \
  -F "file=@prompt.txt" | jq -r .id)

# Reference it in a completion request
curl -X POST "$BASE/v1/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"amazon.nova-micro-v1:0\",
    \"prompt\": \"file-id:${FILE_ID}\",
    \"max_tokens\": 100
  }"
```

#### Using S3 for Prompt Input

Reference files directly in S3 without uploading to stdapi.ai:

```bash
curl -X POST "$BASE/v1/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "prompt": "s3://my-bucket/prompts/essay-topic.txt",
    "max_tokens": 500
  }'
```

The gateway reads the file from S3 using your configured IAM role — no pre-signed URLs required.

## Streaming

Set `"stream": true` to receive incremental text deltas as Server-Sent Events terminated by `data: [DONE]`:

```bash
curl -N -X POST "$BASE/v1/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "prompt": "Write a short story",
    "max_tokens": 100,
    "stream": true
  }'
```

### Usage in Streaming Responses

Request usage statistics on the final chunk by setting `stream_options.include_usage`:

```bash
curl -N -X POST "$BASE/v1/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "prompt": "Hello",
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

## Model-Specific Features

### ![TwelveLabs](styles/logo_twelvelabs.svg){ style="height: 1.2em; vertical-align: text-bottom;" } TwelveLabs Pegasus

`twelvelabs.pegasus-1-2-v1:0` is a video-understanding model. The Completions endpoint supports multimodal input: pass an array with exactly one text instruction and one video URL as `prompt` — the server combines them into a single Pegasus request.

- `temperature` and `max_tokens` are forwarded.
- A text-only `prompt` without a video returns HTTP 400 (Pegasus requires exactly one video).

**Video input formats**: `s3://bucket/key`, `https://…`, `data:video/mp4;base64,…`, or `file-id:…`. Videos above 18.75 MB are automatically uploaded to S3.

```bash
curl -X POST "$BASE/v1/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "twelvelabs.pegasus-1-2-v1:0",
    "prompt": ["Describe what happens in this video.", "s3://my-bucket/video.mp4"]
  }'
```

## Available Request Headers

This endpoint supports standard Bedrock headers for enhanced control over your requests — they are applied by the shared request middleware, exactly as on the [Chat Completions API](api_openai_chat_completions.md#available-request-headers). All headers are optional and can be combined as needed.

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

!!! info "Detailed Documentation"
    For complete information about these headers, configuration options, and use cases, see:

    - [Bedrock Guardrails Configuration](operations_configuration.md#bedrock-guardrails)
    - [Service Tier and Performance Configuration](operations_configuration.md#bedrock-service-tier-and-performance-configuration)

## Try It Now

Send your first completion in one line — then explore the richer input modes shown in [Prompt Input Types](#prompt-input-types) above:

```bash
curl -X POST "$BASE/v1/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "prompt": "Say hello world",
    "max_tokens": 20
  }'
```

---

**Ready to build with AI?** Check out the [Models API](api_openai_models.md) to see all available foundation models!
