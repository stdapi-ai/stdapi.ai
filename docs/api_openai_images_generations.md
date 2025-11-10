# Images API

Generate images with AWS Bedrock image models like Stability AI and Amazon Nova Canvas through an OpenAI-compatible interface.

## Why Choose Image Generation?

<div class="grid cards" markdown>

- :material-palette: __Quality Output__
  <br>Generate photorealistic images, digital art, and illustrations.

- :material-fast-forward: __Real-Time Streaming__
  <br>Progressive generation shows partial previews as the model works for interactive applications.

- :material-ruler: __Flexible Control__
  <br>Choose dimensions, quality levels, and styles. From quick drafts to high-resolution finals.

- :material-aws: __Scalable Infrastructure__
  <br>Generate images at scale with AWS Bedrock infrastructure. No GPU management required.

</div>

## Quick Start: Available Endpoint

| Endpoint                 | Method | What It Does                      | Powered By               |
|--------------------------|--------|-----------------------------------|--------------------------|
| `/v1/images/generations` | POST   | Generate images from text prompts | AWS Bedrock Image Models |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                        |                  Status                  | Notes                                                                    |
|--------------------------------|:----------------------------------------:|--------------------------------------------------------------------------|
| **Generation**                 |                                          |                                                                          |
| Text-to-image (`/generations`) |   :material-check-circle:{ .success }    | Generate images from prompts                                             |
| **Parameters**                 |                                          |                                                                          |
| `prompt`                       |   :material-check-circle:{ .success }    | Text description for generation                                          |
| `n` (number of images)         |   :material-check-circle:{ .success }    | Generate multiple images per request                                     |
| `size` (WIDTHxHEIGHT)          |       :material-cog:{ .model-dep }       | Supported dimensions vary by model; some may approximate requested sizes |
| `quality`                      |       :material-cog:{ .model-dep }       | `low`, `medium`, `high` + model specific quality levels                  |
| `style`                        |       :material-cog:{ .model-dep }       | Some models support style parameters                                     |
| `output_compression`           |   :material-check-circle:{ .success }    | Supported on all input formats                                           |
| `background`                   |   :material-minus-circle:{ .partial }    | Only opaque is supported                                                 |
| `moderation`                   |   :material-minus-circle:{ .partial }    | Only `auto` is supported                                                 |
| Extra model-specific params    | :material-plus-circle:{ .extra-feature } | Extra model-specific parameters not supported by the OpenAI API          |
| **Output**                     |                                          |                                                                          |
| URL response format            |   :material-check-circle:{ .success }    | Temporary URLs to generated images                                       |
| Base64 JSON format             |   :material-check-circle:{ .success }    | Inline base64-encoded images                                             |
| PNG format                     |   :material-check-circle:{ .success }    | Lossless image output                                                    |
| JPEG format                    |   :material-check-circle:{ .success }    | Compressed image output                                                  |
| WebP format                    |   :material-check-circle:{ .success }    | Modern compressed format                                                 |
| **Streaming**                  |                                          |                                                                          |
| SSE streaming                  |   :material-check-circle:{ .success }    | Real-time generation updates                                             |
| Partial images                 |       :material-cog:{ .model-dep }       | Progressive previews when available                                      |
| **Usage tracking**             |                                          |                                                                          |
| Input text tokens              |   :material-minus-circle:{ .partial }    | Estimated for reference                                                  |
| Output image tokens            |   :material-check-circle:{ .success }    | Image count (billing unit)                                               |
| **Other**                      |                                          |                                                                          |
| `user`                         |   :material-minus-circle:{ .partial }    | Logged                                                                   |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep } **Available on Select Models** — Check your model's capabilities
* :material-minus-circle:{ .partial } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature } **Extra Feature** — Enhanced capability beyond OpenAI API

</div>

!!! warning "Configuration Required"
    You must configure the `AWS_S3_BUCKET` environment variable with a bucket to use the URL response format.

!!! tip "Performance Optimization"
    For faster image downloads, especially for high-resolution images or globally distributed users, enable S3 Transfer Acceleration by setting `AWS_S3_ACCELERATE=true`. This uses CloudFront edge locations to accelerate file downloads, providing 50-500% faster speeds for users far from your S3 bucket region. See [S3 Transfer Acceleration configuration](operations_configuration.md#aws-s3-accelerate) for setup details.

## Advanced Features

### Provider-Specific Parameters

Unlock advanced image generation capabilities by passing provider-specific parameters directly in your requests. These parameters are forwarded to AWS Bedrock and allow you to access features unique to each image model provider.

**Documentation:** [Bedrock Image Model Parameters](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html)

**How It Works:**

Add provider-specific fields at the top level of your request body alongside standard OpenAI parameters. The API automatically forwards these to the appropriate model provider via AWS Bedrock.

**Examples:**

**Stability AI - Negative Prompts:**
```json
{
  "model": "stability.stable-image-core-v1:0",
  "prompt": "A serene mountain landscape at sunset",
  "negative_prompt": "blurry, distorted, low quality, watermark"
}
```

**Amazon Nova Canvas - Negative Prompts:**
```json
{
  "model": "amazon.nova-canvas-v1:0",
  "prompt": "An abstract watercolor painting",
  "textToImageParams": {"negativeText": "blurry, distorted, low quality, watermark"},
}
```

**Configuration Options:**

**Option 1: Per-Request**

Add provider-specific parameters directly in your request body (as shown in examples above).

**Option 2: Server-Wide Defaults**

Configure default parameters for specific models via the `DEFAULT_MODEL_PARAMS` environment variable:

```bash
export DEFAULT_MODEL_PARAMS='{
  "stability.stable-image-core-v1:0": {
    "negative_prompt": "blurry, low quality, watermark"
  }
}'
```

**Note:** Per-request parameters override server-wide defaults.

**Behavior:**

- ✅ **Compatible parameters**: Forwarded to the model and applied
- ⚠️ **Unsupported parameters**: Return HTTP 400 with an error message

## Try It Now

**Generate image (URL response):**

```bash
curl -X POST "$BASE/v1/images/generations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain landscape at sunset, photorealistic",
    "model": "amazon.nova-canvas-v1:0",
    "size": "1024x1024",
    "quality": "high",
    "response_format": "url"
  }'
```

**Generate with base64 encoding:**

```bash
curl -X POST "$BASE/v1/images/generations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A futuristic cityscape with flying cars, digital art style",
    "model": "amazon.nova-canvas-v1:0",
    "response_format": "b64_json"
  }'
```

**Stream generation with partial previews:**

```bash
curl -N -X POST "$BASE/v1/images/generations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "An abstract watercolor painting of emotions",
    "model": "amazon.nova-canvas-v1:0",
    "stream": true,
    "partial_images": 3
  }'
```

---

**Unleash your creativity!** Explore available image models in the [Models API](api_openai_models.md).
