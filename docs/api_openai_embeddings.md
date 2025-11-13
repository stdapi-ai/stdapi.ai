# Embeddings API

Transform text into semantic vectors. Power your search, recommendations, and similarity features with AWS Bedrock embedding models through an OpenAI-compatible interface.

## Why Choose Embeddings?

<div class="grid cards" markdown>

- :material-magnify: __Semantic Search__
  <br>Find content based on meaning and context, not just exact words. For knowledge bases and document retrieval.

- :material-lightning-bolt: __High Performance__
  <br>AWS Bedrock embedding models deliver fast vectors optimized for production workloads. Batch processing for large-scale operations.

- :material-puzzle: __Flexible Dimensions__
  <br>Choose vector dimensions that match your needs. Balance accuracy and storage/compute costs with model-specific dimension control.

- :material-image-multiple: __Multimodal Embeddings__
  <br>Process images, videos, audio, and PDF documents alongside text. Unified embeddings for cross-modal search using base64 data URI input.

</div>

## Quick Start: Available Endpoint

| Endpoint         | Method | What It Does                         | Powered By                   |
|------------------|--------|--------------------------------------|------------------------------|
| `/v1/embeddings` | POST   | Transform text into semantic vectors | AWS Bedrock Embedding Models |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                      |                  Status                  | Notes                                                           |
|------------------------------|:----------------------------------------:|-----------------------------------------------------------------|
| **Input Types**              |                                          |                                                                 |
| Text input (single string)   |   :material-check-circle:{ .success }    | Full support for text embeddings                                |
| Multimodal input             | :material-plus-circle:{ .extra-feature } | Image, audio, video                                             |
| Multiple input (batch array) |   :material-check-circle:{ .success }    | Process multiple inputs efficiently                             |
| Token array input            | :material-close-circle:{ .unsupported }  | Array of token integers not supported                           |
| **Output Formats**           |                                          |                                                                 |
| Float vectors                |   :material-check-circle:{ .success }    | Standard floating-point arrays                                  |
| Base64 encoding              |   :material-check-circle:{ .success }    | Base64-encoded float32 arrays                                   |
| **Model Parameters**         |                                          |                                                                 |
| `dimensions` override        |       :material-cog:{ .model-dep }       | Some models support dimension reduction                         |
| `encoding_format`            |   :material-check-circle:{ .success }    | Choose `float` or `base64`                                      |
| Extra model-specific params  | :material-plus-circle:{ .extra-feature } | Extra model-specific parameters not supported by the OpenAI API |
| **Usage tracking**           |                                          |                                                                 |
| Input text tokens            |       :material-cog:{ .model-dep }       | Estimated on some models                                        |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep } **Available on Select Models** — Check your model's capabilities
* :material-close-circle:{ .unsupported } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature } **Extra Feature** — Enhanced capability beyond OpenAI API

</div>

## Advanced Features

### Provider-Specific Parameters

Access advanced embedding capabilities by passing provider-specific parameters directly in your requests. These parameters are forwarded to AWS Bedrock and allow you to access features unique to each embedding model provider.

**Documentation:** [Bedrock Embedding Model Parameters](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html)

**How It Works:**

Add provider-specific fields at the top level of your request body alongside standard OpenAI parameters. The API automatically forwards these to the appropriate model provider via AWS Bedrock.

**Examples:**

**Cohere Embed v4 - Input Type:**
```json
{
  "model": "cohere.embed-v4",
  "input": "Semantic search transforms how we find information",
  "input_type": "search_query"
}
```

**Amazon Titan Embed v2 - Normalization:**
```json
{
  "model": "amazon.nova-2-multimodal-embeddings-v1:0",
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
  "cohere.embed-v4": {
    "input_type": "search_document",
    "truncate": "END"
  }
}'
```

**Note:** Per-request parameters override server-wide defaults.

**Behavior:**

- ✅ **Compatible parameters**: Forwarded to the model and applied
- ⚠️ **Unsupported parameters**: Return HTTP 400 with an error message

## Try It Now

**Single text embedding:**

```bash
curl -X POST "$BASE/v1/embeddings" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-2-multimodal-embeddings-v1:0",
    "input": "Semantic search transforms how we find information"
  }'
```

**Batch processing with base64 encoding:**

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

## Multimodal Embeddings

Go beyond text! Supported models can process images, videos, and audio through base64 data URI input. This enables powerful cross-modal search and similarity features.

### Input Format

Multimodal content is passed as base64-encoded data URIs:

```
data:<mime-type>;base64,<base64-encoded-content>
```

### Example: Image Embedding

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

### Example: Video Embedding

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

!!! info "Automatic S3 Upload and Asynchronous Invocation"
    When you provide Base64-encoded data that exceeds the model's size limit (or Bedrock's 25 MB quota), the server automatically uploads it to S3 and selects the appropriate invocation method (synchronous or asynchronous).

    To allow this behavior, configure regional S3 buckets via `AWS_S3_REGIONAL_BUCKETS` in the same region as your Bedrock model. See [configuration guide](operations_configuration.md#aws-s3-regional-buckets).

!!! warning "Large Base64 Files and Memory Configuration"
    While passing large files as Base64 is supported, ensure your server has sufficient memory configured. Large Base64-encoded files (especially videos) can consume significant memory during processing. Consider using S3 URLs directly for very large files, or adjust your server's memory limits accordingly.

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

!!! warning "S3 URL Requirements"
    When using S3 URLs directly:

    - S3 bucket **must be in the same AWS region** as the Bedrock model
    - The stdapi.ai server **must have read access** to the S3 object
    - For TwelveLabs Marengo models: S3 bucket **must be in the same AWS account** as the STDAPI server

### Example: PDF Document Embedding

For PDFs, convert each page to an image and send via inputs along with page metadata (e.g., file_name, entities) in adjacent text parts. **For RAG applications, smaller chunks often improve retrieval accuracy and reduce costs.**

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
    \"model\": \"cohere.embed-v4\",
    \"input\": [
      \"file_name: report.pdf, page: 1\",
      \"data:image/jpeg;base64,$PAGE_1\",
      \"file_name: report.pdf, page: 2\",
      \"data:image/jpeg;base64,$PAGE_2\"
    ]
  }"
```

### Mixed-Content Batching

Combine text and multimodal inputs in a single request:

```bash
curl -X POST "$BASE/v1/embeddings" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"cohere.embed-v4\",
    \"input\": [
      \"A beautiful sunset over mountains\",
      \"data:image/jpeg;base64,/9j/4AAQSkZJRg...\",
      \"Nature photography collection\"
    ]
  }"
```

### Use Cases

- **Visual Search**: Find images similar to a query image or text description
- **Video Analysis**: Search and retrieve video content based on visual similarity or text descriptions
- **Audio Similarity**: Find similar audio clips or match audio to text descriptions
- **Document Retrieval**: Find relevant PDFs based on visual and textual content
- **Cross-Modal Recommendations**: Recommend images, videos, or audio based on text queries and vice versa
- **Content Moderation**: Analyze and classify multimodal content at scale

---

**Build smarter search and recommendations!** Explore available embedding models in the [Models API](api_openai_models.md).
