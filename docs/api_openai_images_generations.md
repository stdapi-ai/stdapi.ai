---
title: Images Generation API - AWS Bedrock Text-to-Image
description: Generate images with AWS Bedrock using Stability AI and Amazon Nova Canvas. OpenAI-compatible text-to-image API with streaming support and multiple output formats.
keywords: text to image API, image generation API, AWS Bedrock image, Stable Diffusion API, AI image generation, OpenAI images API, DALL-E alternative, Nova Canvas
---

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

| Feature                        |                  Status                  | Notes                                                               |
|--------------------------------|:----------------------------------------:|---------------------------------------------------------------------|
| **Generation**                 |                                          |                                                                     |
| Text-to-image (`/generations`) |   :material-check-circle:{ .success }    | Generate images from prompts                                        |
| **Parameters**                 |                                          |                                                                     |
| `prompt`                       |   :material-check-circle:{ .success }    | Text description for generation (required, min 1 char)              |
| `model`                        |   :material-check-circle:{ .success }    | Required parameter                                                  |
| `n` (number of images)         |   :material-check-circle:{ .success }    | Generate multiple images per request (1-10, default: 1)             |
| `size` (WIDTHxHEIGHT)          |   :material-check-circle:{ .success }    | Output dimensions (default: 1024x1024, format validated)            |
| `response_format`              |   :material-check-circle:{ .success }    | `url` or `b64_json` (default: `url`)                                |
| `quality`                      |       :material-cog:{ .model-dep }       | Quality setting (default: `auto`, supports OpenAI & model-specific) |
| `style`                        |       :material-cog:{ .model-dep }       | Model-specific style parameters                                     |
| `output_format`                |   :material-check-circle:{ .success }    | `png`, `jpeg`, or `webp` (model-specific)                           |
| `output_compression`           |   :material-check-circle:{ .success }    | Compression level 1-100% (default: 100)                             |
| `stream`                       |   :material-check-circle:{ .success }    | Generate images in streaming mode with partial results              |
| `partial_images`               |       :material-cog:{ .model-dep }       | Number of partial images in stream (0-3, model-specific)            |
| `background`                   | :material-close-circle:{ .unsupported }  | Always `auto`, transparent backgrounds unsupported                  |
| `moderation`                   | :material-close-circle:{ .unsupported }  | Always `auto`, other values unsupported                             |
| Extra model-specific params    | :material-plus-circle:{ .extra-feature } | Extra model-specific parameters via JSON body                       |
| **Output**                     |                                          |                                                                     |
| URL response format            |   :material-check-circle:{ .success }    | Temporary URLs to generated images (requires AWS_S3_BUCKET)         |
| Base64 JSON format             |   :material-check-circle:{ .success }    | Inline base64-encoded images                                        |
| PNG format                     |   :material-check-circle:{ .success }    | Lossless image output                                               |
| JPEG format                    |       :material-cog:{ .model-dep }       | Lossy compression (model-specific)                                  |
| WebP format                    |       :material-cog:{ .model-dep }       | Modern format with compression (model-specific)                     |
| **Streaming**                  |                                          |                                                                     |
| SSE streaming                  |   :material-check-circle:{ .success }    | Server-sent events with partial and final images                    |
| Partial images                 |       :material-cog:{ .model-dep }       | Progressive previews when available                                 |
| **Usage tracking**             |                                          |                                                                     |
| Input text tokens              |   :material-check-circle:{ .success }    | Estimated from prompt using tiktoken                                |
| Output image tokens            |   :material-check-circle:{ .success }    | Number of images generated (`n` parameter)                          |
| **Other**                      |                                          |                                                                     |
| `user`                         |   :material-minus-circle:{ .partial }    | Logged but not used for abuse monitoring                            |

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

### ![Amazon](styles/logo_amazon.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Models

| Model                             | Supported Task Types                    | Notes                                                                                       |
|-----------------------------------|-----------------------------------------|---------------------------------------------------------------------------------------------|
| amazon.nova-canvas-v1:0           | `TEXT_IMAGE`, `COLOR_GUIDED_GENERATION` | Supports standard text-to-image generation and color-guided generation with 8 style presets |
| amazon.titan-image-generator-v1   | `TEXT_IMAGE`                            | Basic text-to-image generation                                                              |
| amazon.titan-image-generator-v2:0 | `TEXT_IMAGE`, `COLOR_GUIDED_GENERATION` | Enhanced text-to-image generation with color-guided generation support                      |

### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Stability AI Models

| Model                             | Supported Task Types | Notes                                             |
|-----------------------------------|----------------------|---------------------------------------------------|
| stability.sd3-5-large-v1:0        | `TEXT_IMAGE`         | Stable Diffusion 3.5 Large - high quality output  |
| stability.stable-image-core-v1:0  | `TEXT_IMAGE`         | Stable Image Core - balanced quality and speed    |
| stability.stable-image-ultra-v1:0 | `TEXT_IMAGE`         | Stable Image Ultra - premium quality and detail   |

!!! warning "Configuration Required"
    You must configure the `AWS_S3_BUCKET` environment variable with a bucket to use the URL response format.

!!! tip "Performance Optimization"
    For faster image downloads, especially for high-resolution images or globally distributed users, enable S3 Transfer Acceleration by setting `AWS_S3_ACCELERATE=true`. This uses CloudFront edge locations to accelerate file downloads, providing 50-500% faster speeds for users far from your S3 bucket region. See [S3 Transfer Acceleration configuration](operations_configuration.md#aws-s3-accelerate) for setup details.

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

**Example with headers:**

```bash
curl -X POST "$BASE/v1/images/generations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Amzn-Bedrock-Service-Tier: priority" \
  -H "X-Amzn-Bedrock-PerformanceConfig-Latency: optimized" \
  -d '{
    "model": "amazon.nova-canvas-v1:0",
    "prompt": "A serene mountain landscape at sunset"
  }'
```

!!! info "Detailed Documentation"
    For complete information about these headers, configuration options, and use cases, see:

    - [Bedrock Guardrails Configuration](operations_configuration.md#bedrock-guardrails)
    - [Service Tier and Performance Configuration](operations_configuration.md#bedrock-service-tier-and-performance-configuration)

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
  "textToImageParams": {"negativeText": "blurry, distorted, low quality, watermark"}
}
```

### ![Amazon Nova](styles/logo_amazon_nova.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Nova Canvas Extra Features

**Basic Usage (Standard OpenAI Parameters):**

```bash
curl -X POST "$BASE/v1/images/generations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-canvas-v1:0",
    "prompt": "A futuristic cityscape at night"
  }'
```

**Parameter Mapping:**

| OpenAI Parameter | Maps to                                | Notes                               |
|------------------|----------------------------------------|-------------------------------------|
| `prompt`         | Depends on `taskType`                  | See taskType-specific mapping below |
| `size`           | `imageGenerationConfig.width/height`   | Flexible (320-4096)                 |
| `quality`        | `imageGenerationConfig.quality`        | "high" → "premium"                  |
| `style`          | `textToImageParams.style`              | 8 preset styles                     |
| `n`              | `imageGenerationConfig.numberOfImages` | 1-5 images                          |

**TaskType-Specific Parameter Mapping:**

| taskType                  | `prompt` maps to                   |
|---------------------------|------------------------------------|
| `TEXT_IMAGE` (default)    | `textToImageParams.text`           |
| `COLOR_GUIDED_GENERATION` | `colorGuidedGenerationParams.text` |

**Advanced Generation Modes:**

Default `taskType` is `"TEXT_IMAGE"`.

Available task types:

- `"TEXT_IMAGE"` - Standard text-to-image generation
- `"COLOR_GUIDED_GENERATION"` - Generate images based on color palette

```bash
# Color-Guided Generation
curl -X POST "$BASE/v1/images/generations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-canvas-v1:0",
    "prompt": "A sunset landscape",
    "taskType": "COLOR_GUIDED_GENERATION",
    "colorGuidedGenerationParams": {
      "colors": ["#FF6B6B", "#FFD93D", "#6BCB77"]
    }
  }'
```

!!! info "Full Parameter Reference"
    For all parameters, styles, and task types, see [Amazon Nova Canvas documentation](https://docs.aws.amazon.com/nova/latest/userguide/image-generation.html)

### ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Titan Image Generator Extra Features

**Basic Usage (Standard OpenAI Parameters):**

```bash
curl -X POST "$BASE/v1/images/generations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.titan-image-generator-v2:0",
    "prompt": "A beautiful landscape with mountains"
  }'
```

**Parameter Mapping:**

| OpenAI Parameter | Maps to                                | Notes                               |
|------------------|----------------------------------------|-------------------------------------|
| `prompt`         | Depends on `taskType`                  | See taskType-specific mapping below |
| `size`           | `imageGenerationConfig.width/height`   | Fixed sizes (512-2048)              |
| `quality`        | `imageGenerationConfig.quality`        | "high" → "premium"                  |
| `n`              | `imageGenerationConfig.numberOfImages` | 1-5 images                          |

**TaskType-Specific Parameter Mapping:**

| taskType                  | `prompt` maps to                   |
|---------------------------|------------------------------------|
| `TEXT_IMAGE` (default)    | `textToImageParams.text`           |
| `COLOR_GUIDED_GENERATION` | `colorGuidedGenerationParams.text` |

**Advanced Generation Modes:**

Default `taskType` is `"TEXT_IMAGE"`.

Available task types:

- `"TEXT_IMAGE"` - Standard text-to-image generation
- `"COLOR_GUIDED_GENERATION"` - Generate images based on color palette

```bash
# Color-Guided Generation
curl -X POST "$BASE/v1/images/generations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.titan-image-generator-v2:0",
    "prompt": "Nature scene with colors",
    "taskType": "COLOR_GUIDED_GENERATION",
    "colorGuidedGenerationParams": {
      "colors": ["#2ECC71", "#3498DB", "#F39C12"]
    }
  }'
```

!!! info "Full Parameter Reference"
    For all parameters and task types, see [Amazon Titan Image Generator documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html)

### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Stability AI Models

**Basic Usage (Standard OpenAI Parameters):**

```bash
curl -X POST "$BASE/v1/images/generations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stability.stable-image-ultra-v1:0",
    "prompt": "A photorealistic mountain landscape at sunset"
  }'
```

**Parameter Mapping:**

| OpenAI Parameter | Maps to        | Notes                                        |
|------------------|----------------|----------------------------------------------|
| `prompt`         | `prompt`       | Text description for generation              |
| `size`           | `aspect_ratio` | Inferred from size (e.g., 1024x1024 → "1:1") |
| `n`              | Multiple calls | Each image is a separate request             |

**Model Comparison:**

| Model                             | Output Formats  | Best For                              |
|-----------------------------------|-----------------|---------------------------------------|
| stability.sd3-5-large-v1:0        | png, jpeg, webp | High quality, versatile compositions  |
| stability.stable-image-core-v1:0  | png, jpeg       | Balanced quality and speed            |
| stability.stable-image-ultra-v1:0 | png, jpeg       | Premium quality and detail            |

!!! info "Full Parameter Reference"
    For all Stability AI parameters, see [Stability AI documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-stability-diffusion.html)

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
