---
title: Embeddings API - Amazon Bedrock Vector Embeddings
description: Generate vector embeddings with Amazon Bedrock using OpenAI-compatible API. Support for semantic search, RAG, multimodal embeddings with text, images, audio, and documents.
keywords: embeddings API, vector embeddings AWS, semantic search API, RAG embeddings, text embeddings, multimodal embeddings, Amazon Bedrock embeddings, OpenAI embeddings
---

# Embeddings API

Generate vector embeddings for semantic search and RAG applications with Amazon Bedrock embedding models through an OpenAI-compatible interface.

## At a glance

- :material-magnify: **Vectors from the Amazon Bedrock embedding models** — Amazon Titan Embed, Amazon Nova, Cohere Embed and TwelveLabs Marengo — for semantic search, RAG and recommendation.
- :material-image-multiple: **Multimodal on the same route.** Images, video, audio and PDF pages go in as base64 data URIs, `s3://` URIs or `file-id:` references, for cross-modal search against the same vector space.
- :material-puzzle: **`dimensions` and `encoding_format`.** Vectors come back as floats or base64-encoded float32; models that support dimension reduction honor `dimensions`, trading accuracy for storage and compute.
- :material-format-list-group: **Batches, on and off the request path.** One request embeds an array of inputs; a whole corpus goes through the [Batch API](api_openai_batches.md) at the published batch rate.
- :material-swap-horizontal: **Differs from OpenAI:** an array of token integers is rejected — send strings — and unrecognized top-level fields are forwarded to the model as provider-specific parameters instead of being ignored.

```bash
curl -X POST "$BASE/v1/embeddings" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-2-multimodal-embeddings-v1:0",
    "input": "Semantic search transforms how we find information"
  }'
```

## Endpoints { #available-endpoints }

| Endpoint         | Method | What It Does                                                | Powered By                   | MCP Tool           |
|------------------|--------|-------------------------------------------------------------|------------------------------|--------------------|
| `/v1/embeddings` | `POST`   | Transform text and multimodal content into semantic vectors | Amazon Bedrock Embedding Models | `openai_embedding` |

## Feature compatibility

<div class="feature-table" markdown>

| Feature                      |                  Status                  | Notes                                                           |
|------------------------------|:----------------------------------------:|-----------------------------------------------------------------|
| **Input Types**              |                                          |                                                                 |
| Text input (single string)   |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Full support for text embeddings                                |
| Multimodal input             | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Image, audio, video, document (image + text)                    |
| Multiple input (batch array) |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Process multiple inputs efficiently                             |
| Token array input            | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Array of token integers not supported                           |
| **Output Formats**           |                                          |                                                                 |
| Float vectors                |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Standard floating-point arrays                                  |
| Base64 encoding              |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Base64-encoded float32 arrays                                   |
| **Model Parameters**         |                                          |                                                                 |
| `dimensions` override        |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Some models support dimension reduction                         |
| `encoding_format`            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Choose `float` or `base64`                                      |
| Extra model-specific params  | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Extra model-specific parameters not supported by the OpenAI API |
| **Usage tracking**           |                                          |                                                                 |
| Input text tokens            |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Reported as billed by Amazon Bedrock; zero on models and inputs for which Bedrock returns no token count |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep role="img" aria-label="Model-dependent" } **Model-Dependent** — Behavior depends on the model or backend; check the Notes column
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } **Extra Feature** — Enhanced capability beyond OpenAI API

</div>

## Models { #model-support }

Any Amazon Bedrock model that produces embeddings answers on this route — the Amazon Titan Embed and Amazon Nova embedding families, Cohere Embed, and TwelveLabs Marengo. A model that produces something else is refused with `400`, naming it.

To list the models this deployment serves on this route, call [`search_models`](api_search_models.md) with `route=openai_embedding`. Which of them accept images, video or audio rather than text alone is covered under [Multimodal Embeddings](#multimodal-embeddings).

!!! note "No Bedrock Mantle models here"
    [Bedrock Mantle](features.md#bedrock-mantle-models) serves chat models only. Embeddings always come from the classic `bedrock-runtime` catalogue, whether or not Mantle is enabled.

### Model Name Aliases

Cohere models carry the dotted version their own API publishes, so the name you already send resolves without change:

- `embed-v4.0` → `cohere.embed-v4:0`
- `embed-english-v3.0` → `cohere.embed-english-v3`

## Working with embeddings { #advanced-features }

### Embedding a Corpus in Bulk

A large corpus does not have to be embedded on the request path. Upload the
requests as a file and run them through the [Batch API](api_openai_batches.md)
against `/v1/embeddings`: they run without a connection held open, at the
published batch rate, and the vectors are read back from the result file. One
batched request embeds one `input`, and its vectors come back as numbers.

### Provider-Specific Parameters

Access advanced embedding capabilities by passing provider-specific parameters directly in your requests. These parameters are forwarded to Amazon Bedrock and allow you to access features unique to each embedding model provider.

**Documentation:** [Bedrock Embedding Model Parameters](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html)

**How It Works:**

Add provider-specific fields at the top level of your request body alongside standard OpenAI parameters. The API automatically forwards these to the appropriate model provider via Amazon Bedrock.

**Examples:**

**Cohere Embed v4 - Input Type:**
```json
{
  "model": "cohere.embed-v4:0",
  "input": "Semantic search transforms how we find information",
  "input_type": "search_query"
}
```

**Amazon Titan Embed v2 - Normalization:**
```json
{
  "model": "amazon.titan-embed-text-v2:0",
  "input": "Product description for similarity matching",
  "normalize": true
}
```

**Configuration Options:**

**Option 1: Per-Request**

Add provider-specific parameters directly in your request body (as shown in examples above).

**Option 2: Server-Wide Defaults**

Configure default parameters for specific models via the `DEFAULT_MODEL_PARAMS` environment variable:

```bash
export DEFAULT_MODEL_PARAMS='{
  "cohere.embed-v4:0": {
    "input_type": "search_document",
    "truncate": "END"
  }
}'
```

**Note:** Per-request parameters override server-wide defaults.

**Behavior:**

- :material-check-circle:{ .success role="img" aria-label="Supported" } **Compatible parameters**: Forwarded to the model and applied
- :material-alert-circle:{ .warning } **Unsupported parameters**: Return HTTP 400 with an error message

### Multimodal Embeddings

Supported models embed images, videos and audio alongside text, for cross-modal search and similarity in one vector space.

#### Input Format

Multimodal content is passed as base64-encoded data URIs:

```text
data:<mime-type>;base64,<base64-encoded-content>
```

#### Example: Image Embedding

```bash
# First, encode your image to base64
IMAGE_B64=$(base64 -w 0 image.jpg)

# Send the embedding request
curl -X POST "$BASE/v1/embeddings" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"amazon.nova-2-multimodal-embeddings-v1:0\",
    \"input\": \"data:image/jpeg;base64,$IMAGE_B64\"
  }"
```

#### Example: Video Embedding

**Option 1: Base64-encoded video (for small files)**

```bash
# First, encode your video to base64
VIDEO_B64=$(base64 -w 0 video.mp4)

# Send the embedding request
curl -X POST "$BASE/v1/embeddings" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"amazon.nova-2-multimodal-embeddings-v1:0\",
    \"input\": \"data:video/mp4;base64,$VIDEO_B64\"
  }"
```

!!! info "Automatic S3 Upload"
    When you provide Base64-encoded data that exceeds the model's size limit (or Bedrock's 25 MB quota), the server automatically stages it in S3 so the request still succeeds.

    To allow this behavior, configure regional S3 buckets via `AWS_S3_REGIONAL_BUCKETS` in the same region as your Bedrock model. See [configuration guide](operations_configuration_storage.md#aws-s3-regional-buckets).

**Option 2: S3 URL (for large files)**

```bash
# Send the embedding request with S3 URL
curl -X POST "$BASE/v1/embeddings" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-2-multimodal-embeddings-v1:0",
    "input": "s3://my-bucket/path/to/video.mp4"
  }'
```

**Option 3: Files API reference (`file-id:`)**

Reference a file previously uploaded via the [Files API](api_openai_files.md#referencing-uploaded-files-via-the-file-id-uri-scheme) using the project-local `file-id:` URI scheme:

```bash
curl -X POST "$BASE/v1/embeddings" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-2-multimodal-embeddings-v1:0",
    "input": "file-id:file-0190c51c7de7455d9b8c2efe27dfbf67"
  }'
```

#### Example: PDF Document Embedding

For PDFs, convert each page to an image and send via inputs along with page metadata (e.g., file_name, entities) in adjacent text parts. **For RAG applications, smaller chunks often improve retrieval accuracy and reduce costs.**

**![Cohere](styles/logo_cohere.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Cohere Embed v4** (supports multiple text+image pairs in one request):

```bash
# Convert PDF pages to images (using ImageMagick or similar tool)
convert -density 150 document.pdf page-%d.jpg

# Encode each page image to base64
PAGE_1=$(base64 -w 0 page-0.jpg)
PAGE_2=$(base64 -w 0 page-1.jpg)

# Generate document embedding with metadata
curl -X POST "$BASE/v1/embeddings" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"cohere.embed-v4:0\",
    \"input\": [
      \"file_name: report.pdf, page: 1\",
      \"data:image/jpeg;base64,$PAGE_1\",
      \"file_name: report.pdf, page: 2\",
      \"data:image/jpeg;base64,$PAGE_2\"
    ]
  }"
```

**![TwelveLabs](styles/logo_twelvelabs.svg){ style="height: 1.2em; vertical-align: text-bottom;" } TwelveLabs Marengo v3** (requires exactly one text + one image per request):

!!! info "Text+Image Pairing for Marengo v3"
    When using `twelvelabs.marengo-embed-3-0-v1:0`, if you provide exactly **2 inputs** where one is text and one is image, they are automatically combined into a single `text_image` embedding. This creates a unified multimodal representation of the text-image pair.

```bash
# Encode image to base64
IMAGE_B64=$(base64 -w 0 page-0.jpg)

# Generate text+image embedding (automatically uses text_image mode)
curl -X POST "$BASE/v1/embeddings" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"twelvelabs.marengo-embed-3-0-v1:0\",
    \"input\": [
      \"A diagram showing the quarterly sales report\",
      \"data:image/jpeg;base64,$IMAGE_B64\"
    ]
  }"
```

#### Mixed-Content Batching

Combine text and multimodal inputs in a single request:

```bash
curl -X POST "$BASE/v1/embeddings" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"cohere.embed-v4:0\",
    \"input\": [
      \"A beautiful sunset over mountains\",
      \"data:image/jpeg;base64,/9j/4AAQSkZJRg...\",
      \"Nature photography collection\"
    ]
  }"
```

#### Use Cases

- **Visual Search**: Find images similar to a query image or text description
- **Video Analysis**: Search and retrieve video content based on visual similarity or text descriptions
- **Audio Similarity**: Find similar audio clips or match audio to text descriptions
- **Document Retrieval**: Find relevant PDFs based on visual and textual content
- **Cross-Modal Recommendations**: Recommend images, videos, or audio based on text queries and vice versa
- **Content Moderation**: Analyze and classify multimodal content at scale

## Limits and behaviour to know

**An `s3://` input has three requirements.** The bucket is in the same AWS region as the Bedrock model, the stdapi.ai server has read access to the object, and — for TwelveLabs Marengo models — the bucket is in the same AWS account as the server.

**Token usage is reported only where Amazon Bedrock returns it.** Amazon Nova models report no token count for inputs above their inline size limit, and TwelveLabs Marengo models report none at all: the `usage` field then reports zero tokens.

**Base64 input is decoded in memory.** A large inline file, a video especially, can consume significant memory while it is processed, so size the server's memory limit for the largest input you send — or pass an `s3://` URI instead of inlining the bytes.

## Request headers { #available-request-headers }

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

**Example with headers:**

```bash
curl -X POST "$BASE/v1/embeddings" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Amzn-Bedrock-Service-Tier: flex" \
  -H "X-Amzn-Bedrock-PerformanceConfig-Latency: standard" \
  -d '{
    "model": "amazon.nova-2-multimodal-embeddings-v1:0",
    "input": ["Batch text 1", "Batch text 2", "Batch text 3"]
  }'
```

!!! info "Detailed Documentation"
    For complete information about these headers, configuration options, and use cases, see:

    - [Bedrock Guardrails Configuration](operations_configuration_bedrock.md#bedrock-guardrails)
    - [Service Tier and Performance Configuration](operations_configuration_bedrock.md#bedrock-service-tier-and-performance-configuration)

## Try it { #try-it-now }

**Batch inputs, base64-encoded vectors:**

```bash
curl -X POST "$BASE/v1/embeddings" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-2-multimodal-embeddings-v1:0",
    "input": ["Product description", "User query", "Related content"],
    "encoding_format": "base64"
  }'
```

## Next steps

Next: [Batch API](api_openai_batches.md) · [Files API](api_openai_files.md) · [Cohere Embed API](api_cohere_embed.md) · [Models API](api_openai_models.md)
