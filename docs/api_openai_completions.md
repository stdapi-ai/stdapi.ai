---
title: Completions API - OpenAI-Compatible Text Completion
description: OpenAI-compatible completions API for AWS Bedrock models including Claude, Nova, Llama. Simple prompt-to-text generation with the smallest token footprint.
keywords: completions API, OpenAI completions API, text completion API, AWS Bedrock text completion, MCP text completion
---

# Completions API

Generate text completions with AWS Bedrock foundation models—including Claude, Nova, Llama, and more—through an OpenAI-compatible interface using the simple completions format.

!!! info "Legacy upstream, first-class here"

    OpenAI labels `/v1/completions` as **legacy** in their platform documentation and recommends new OpenAI projects migrate to `/v1/chat/completions` or `/v1/responses` for vendor compatibility. On stdapi, this endpoint is a **first-class route** with the same quality guarantees as the others — its compact schema and small token footprint make it an excellent pick for MCP-based text agents and simple prompt-to-text workloads.

## Quick Start: Available Endpoint

| Endpoint          | Method | What It Does                     | Powered By               | MCP Tool            |
|-------------------|--------|----------------------------------|--------------------------|---------------------|
| `/v1/completions` | POST   | Simple prompt-to-text completion | AWS Bedrock Converse API | `openai_completion` |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                            |                  Status                  | Notes                                                            |
|------------------------------------|:----------------------------------------:|------------------------------------------------------------------|
| **Prompt Input**                   |                                          |                                                                  |
| Single text prompt                 |   :material-check-circle:{ .success }    | Full support for string prompts                                  |
| Multiple prompts (batch)           |   :material-check-circle:{ .success }    | Returns one choice per prompt                                    |
| Text + files collapse (multimodal) | :material-plus-circle:{ .extra-feature } | `[text, file, …]` sent as one multimodal request with one choice |
| Prompt from URL (`https://`)       |   :material-check-circle:{ .success }    | HTTP URL reference                                               |
| Prompt from S3 (`s3://`)           | :material-plus-circle:{ .extra-feature } | S3 URI reference                                                 |
| Prompt from data URI (`data:`)     |   :material-check-circle:{ .success }    | Base64-encoded data URI                                          |
| Prompt from Files API (`file-id:`) | :material-plus-circle:{ .extra-feature } | Reference uploaded files                                         |
| Token array prompts                | :material-close-circle:{ .unsupported }  | Not supported — use string prompts                               |
| **Generation Control**             |                                          |                                                                  |
| `max_tokens`                       |   :material-check-circle:{ .success }    | Output length limits                                             |
| `temperature`                      |       :material-cog:{ .model-dep }       | Mapped to Bedrock inference params                               |
| `top_p`                            |       :material-cog:{ .model-dep }       | Nucleus sampling control                                         |
| `stop` sequences                   |       :material-cog:{ .model-dep }       | Custom stop strings                                              |
| `n` (multiple choices)             |   :material-check-circle:{ .success }    | Supported with and without streaming                             |
| `best_of`                          | :material-close-circle:{ .unsupported }  | Accepted but ignored                                             |
| `echo`                             | :material-close-circle:{ .unsupported }  | Accepted but ignored                                             |
| `frequency_penalty`                | :material-close-circle:{ .unsupported }  | Accepted but ignored                                             |
| `presence_penalty`                 | :material-close-circle:{ .unsupported }  | Accepted but ignored                                             |
| `logit_bias`                       | :material-close-circle:{ .unsupported }  | Accepted but ignored                                             |
| `logprobs`                         | :material-close-circle:{ .unsupported }  | Accepted but ignored                                             |
| `seed`                             | :material-close-circle:{ .unsupported }  | Accepted but ignored                                             |
| `suffix`                           | :material-close-circle:{ .unsupported }  | Accepted but ignored                                             |
| **Streaming**                      |                                          |                                                                  |
| Streaming (`stream: true`)         |   :material-check-circle:{ .success }    | Server-Sent Events (SSE)                                         |
| `stream_options.include_usage`     |   :material-check-circle:{ .success }    | Usage in final chunk                                             |
| Streaming with multiple prompts    |   :material-check-circle:{ .success }    | Deltas interleave; `choices[0].index` identifies the prompt      |
| Streaming with n>1                 |   :material-check-circle:{ .success }    | Deltas interleave; `choices[0].index` identifies each choice     |
| **Other**                          |                                          |                                                                  |
| Service tiers                      |   :material-check-circle:{ .success }    | Mapped to Bedrock service tiers                                  |
| `user` / `safety_identifier`       |   :material-minus-circle:{ .partial }    | Logged                                                           |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep } **Available on Select Models** — Check your model's capabilities
* :material-minus-circle:{ .partial } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature } **Extra Feature** — Enhanced capability beyond OpenAI API

</div>

## Prompt Input Types

_stdapi extends the standard completions interface with multiple input modes:_

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

### Text + Files: single-request multimodal collapse

When the prompt list contains **exactly one text string and one or more file references**, stdapi packs them in input order into a **single multimodal request** and returns **one choice**. The text becomes a `text` block and each file becomes an `image` / `document` / `audio` / `video` Bedrock block (content type auto-detected) — the natural "ask once using these files as context" pattern.

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

### Files-only prompts

When the prompt is a lone file — or a list of files without any text string — stdapi forwards each file to the model using its detected modality (`image`, `video`, `audio`, `document`). No instruction is injected:

- `prompt: "<file-id or URL>"` → one request, one choice, with a single media block.
- `prompt: [<file1>, <file2>, …]` → one request per file (batch), one choice per file, each carrying its own media block.

The request reaches the model as-is; the model returns output or an error depending on whether it supports that modality (for example, Claude handles images and documents, Nova handles images, video, and documents). Use this shape for quick "what is this?" queries where the model's default behaviour is enough; for tighter control over the response, prefer the collapse pattern above with an explicit instruction.

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

stdapi supports additional input schemes beyond the OpenAI standard:

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

Reference files directly in S3 without uploading to stdapi:

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

## Model-specific features

### ![TwelveLabs](styles/logo_twelvelabs.svg){ style="height: 1.2em; vertical-align: text-bottom;" } TwelveLabs Pegasus

`twelvelabs.pegasus-1-2-v1:0` is a video-understanding model. The Completions endpoint supports multimodal input: pass an array with exactly one text instruction and one video URL as `prompt` — the server combines them into a single Pegasus request.

- `temperature` and `max_tokens` are forwarded.
- A text-only `prompt` without a video returns HTTP 400 (Pegasus requires exactly one video).

**Video input formats**: `s3://bucket/key`, `https://…`, `data:video/mp4;base64,…`, or `file-id:…`. Videos above 18.75 MB are automatically uploaded to S3.

```bash
curl https://api.example.com/v1/completions \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "twelvelabs.pegasus-1-2-v1:0",
    "prompt": ["Describe what happens in this video.", "s3://my-bucket/video.mp4"]
  }'
```

## Examples

### Single Prompt Completion

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

### Batch Prompt Completion

```bash
curl -X POST "$BASE/v1/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "prompt": [
      "One plus one is",
      "Two plus two is",
      "Three plus three is"
    ],
    "max_tokens": 10
  }'
```

### Streaming Completion

```bash
curl -N -X POST "$BASE/v1/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "prompt": "Tell me a joke",
    "max_tokens": 50,
    "stream": true
  }'
```
