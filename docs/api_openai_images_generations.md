---
title: Images Generation API - Amazon Bedrock Text-to-Image
description: Generate images with Amazon Bedrock using Stability AI and Amazon Nova Canvas. OpenAI-compatible text-to-image API with streaming support and multiple output formats.
keywords: text to image API, image generation API, AWS Bedrock image, Stable Diffusion API, AI image generation, OpenAI images API, DALL-E alternative, Nova Canvas
---

# Images API - Image Generation

Generate images with Amazon Bedrock image models like Stability AI and Amazon Nova Canvas through an OpenAI-compatible interface.

## At a glance

- :material-palette: **Six Amazon Bedrock image models** — Amazon Nova Canvas, two Amazon Titan
  Image Generator versions and three Stability AI models, behind one endpoint, see
  [Models](#model-support).
- :material-ruler: **`n` accepts 1 to 10 images per request** — alongside `size`, `quality`,
  `style`, `output_format` and `output_compression`, see
  [Feature compatibility](#feature-compatibility).
- :material-fast-forward: **`stream: true` emits `image_generation.completed`** — each finished
  image arrives as its own server-sent event instead of waiting for the whole batch, see
  [Feature compatibility](#feature-compatibility).
- :material-aws: **Served by Amazon Bedrock in your own AWS account** — no GPU capacity to
  provision, and `url` responses are download links to your own `AWS_S3_BUCKET` valid for 60
  minutes, see [Models](#model-support).
- :material-swap-horizontal: **`gpt-image-1` and the other OpenAI image model names are not
  aliased** — map them with `MODEL_ALIASES`, since an unmapped name returns a model-not-found
  error, see [Limits and behaviour to know](#limits-and-behaviour-to-know).
- :material-swap-horizontal: **`partial_images`, `moderation` and transparent backgrounds are
  not served** — the parameters are accepted or refused but never change the image, see
  [Limits and behaviour to know](#limits-and-behaviour-to-know).

```bash
curl -X POST "$BASE/v1/images/generations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stability.stable-image-core-v1:1",
    "prompt": "A serene mountain landscape at sunset, photorealistic",
    "size": "1024x1024"
  }'
```

## Endpoints { #quick-start-available-endpoint }

| Endpoint                 | Method | What It Does                      | Powered By                  | MCP Tool                  |
|--------------------------|--------|-----------------------------------|-----------------------------|---------------------------|
| `/v1/images/generations` | `POST` | Generate images from text prompts | Amazon Bedrock Image Models | `openai_image_generation` |

## Feature compatibility

<div class="feature-table" markdown>

| Feature                        |                  Status                  | Notes                                                               |
|--------------------------------|:----------------------------------------:|---------------------------------------------------------------------|
| **Generation**                 |                                          |                                                                     |
| Text-to-image (`/generations`) |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Generate images from prompts                                        |
| **Parameters**                 |                                          |                                                                     |
| `prompt`                       |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Text description for generation (required, min 1 char)              |
| `model`                        |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Required parameter                                                  |
| `n` (number of images)         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Multiple images per request; accepted range is 1-10 (default: 1), but the effective maximum is model-dependent (e.g. Amazon Titan and Nova Canvas cap at 5) |
| `size` (WIDTHxHEIGHT)          |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Output dimensions (default: 1024x1024, format validated; `auto` resolves to the default) |
| `response_format`              |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | `url` or `b64_json` (default: `url`)                                |
| `quality`                      |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Quality setting (default: `auto`, supports OpenAI & model-specific); accepted and ignored by models with no quality control |
| `style`                        |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Model-specific style parameters; accepted and ignored by models with no style control |
| `output_format`                |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | `png`, `jpeg`, or `webp` on every model; the gateway re-encodes when the model cannot produce the format natively |
| `output_compression`           |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Compression level 1-100% (default: 100)                             |
| `stream`                       |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Generate images in streaming mode, sending each finished image as an `image_generation.completed` event |
| `partial_images`               | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted (0-3) but ignored — no available model currently streams partial images; the final image is always sent as a single event |
| `background`                   |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Accepts `auto` (default) and `opaque`; `transparent` is unsupported — responses report `opaque` |
| `moderation`                   | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Only the default `auto` is accepted; other values are rejected with an error |
| Extra model-specific params    | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Extra model-specific parameters via JSON body                       |
| **Output**                     |                                          |                                                                     |
| URL response format            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Temporary download URLs, valid for 60 minutes (requires AWS_S3_BUCKET) |
| Base64 JSON format             |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Inline base64-encoded images                                        |
| PNG format                     |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Lossless image output                                               |
| JPEG format                    |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Lossy compression, re-encoded server-side when the model has no native JPEG output |
| WebP format                    |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Modern format with compression, re-encoded server-side when the model has no native WebP output |
| Streaming response             |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Server-sent events with final images (no partial previews)          |
| **Usage tracking**             |                                          |                                                                     |
| Input text tokens              |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Sourced from AWS billing when available                             |
| Output image tokens            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Sourced from AWS billing data when available; falls back to the image count (`n`) |
| **Other**                      |                                          |                                                                     |
| `user`                         |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Logged but not used for abuse monitoring                            |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep role="img" aria-label="Model-dependent" } **Available on Select Models** — Check your model's capabilities
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } **Extra Feature** — Enhanced capability beyond OpenAI API

</div>

## Models { #model-support }

### ![Amazon](styles/logo_amazon.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Models

| Model                             | Supported Task Types                    | Notes                                                                                       |
|-----------------------------------|-----------------------------------------|---------------------------------------------------------------------------------------------|
| amazon.nova-canvas-v1:0 (legacy)  | `TEXT_IMAGE`, `COLOR_GUIDED_GENERATION` | Supports standard text-to-image generation and color-guided generation with 8 style presets |
| amazon.titan-image-generator-v1 (legacy)  | `TEXT_IMAGE`                            | Basic text-to-image generation                                                              |
| amazon.titan-image-generator-v2:0 (legacy) | `TEXT_IMAGE`, `COLOR_GUIDED_GENERATION` | Enhanced text-to-image generation with color-guided generation support                      |

!!! note "Legacy Amazon Image Models"
    AWS has scheduled `amazon.nova-canvas-v1:0` and the Titan image models to reach end of life on September 30, 2026. Deployments with existing access can keep using them until then (legacy models are hidden unless [`AWS_BEDROCK_LEGACY=true`](operations_configuration_models.md#bedrock-legacy)); the Stability AI Stable Image family is the long-term successor.

### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Stability AI Models

| Model                             | Supported Task Types | Notes                                             |
|-----------------------------------|----------------------|---------------------------------------------------|
| stability.sd3-5-large-v1:0        | `TEXT_IMAGE`         | Stable Diffusion 3.5 Large - high quality output  |
| stability.stable-image-core-v1:1  | `TEXT_IMAGE`         | Stable Image Core - balanced quality and speed    |
| stability.stable-image-ultra-v1:1 | `TEXT_IMAGE`         | Stable Image Ultra - premium quality and detail   |

!!! note "Output Formats"
    All models support the standard OpenAI output formats (`png`, `jpeg`, `webp`) via the `output_format` parameter. When a model cannot produce the requested format natively, the gateway re-encodes the result server-side, so the response always carries the format you asked for.

!!! info "No Built-In Aliases for OpenAI Image Model Names"
    `gpt-image-1` and `gpt-image-1-mini` have **no built-in alias**, so requests naming them fail with a model-not-found error — the most common first-call issue. Pass one of the model IDs above, or map those names to your preferred models with [`MODEL_ALIASES`](operations_configuration_models.md#model-aliases). The retired `dall-e-2` and `dall-e-3` names are legacy strings older clients may still send; map them the same way.

!!! warning "Configuration Required"
    You must configure the `AWS_S3_BUCKET` environment variable with a bucket to use the URL response format.

!!! tip "Performance Optimization"
    For faster image downloads, especially for high-resolution images or globally distributed users, enable S3 Transfer Acceleration by setting `AWS_S3_ACCELERATE=true`. This uses CloudFront edge locations to accelerate file downloads, providing 50-500% faster speeds for users far from your S3 bucket region. See [S3 Transfer Acceleration configuration](operations_configuration_storage.md#aws-s3-accelerate) for setup details.

## Working with image models { #advanced-features }

### Provider-Specific Parameters

Unlock advanced image generation capabilities by passing provider-specific parameters directly in your requests. These parameters are forwarded to Amazon Bedrock and allow you to access features unique to each image model provider.

**Documentation:** [Bedrock Image Model Parameters](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters.html)

**How It Works:**

Add provider-specific fields at the top level of your request body alongside standard OpenAI parameters. The API automatically forwards these to the appropriate model provider via Amazon Bedrock.

**Configuration Options:**

**Option 1: Per-Request**

Add provider-specific parameters directly in your request body (as shown in the examples below).

**Option 2: Server-Wide Defaults**

Configure default parameters for specific models via the `DEFAULT_MODEL_PARAMS` environment variable:

```bash
export DEFAULT_MODEL_PARAMS='{
  "stability.stable-image-core-v1:1": {
    "negative_prompt": "blurry, low quality, watermark"
  }
}'
```

**Note:** Per-request parameters override server-wide defaults.

**Behavior:**

**Compatible parameters** are forwarded to the model and applied; **unsupported parameters** return HTTP 400 with an error message.

**Examples:**

**Stability AI - Negative Prompts:**
```json
{
  "model": "stability.stable-image-core-v1:1",
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

#### ![Amazon Nova](styles/logo_amazon_nova.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Nova Canvas { #amazon-nova-canvas-extra-features }

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

#### ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Titan Image Generator { #amazon-titan-image-generator-extra-features }

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

| OpenAI Parameter | Maps to                                | Notes                                                                       |
|------------------|----------------------------------------|-----------------------------------------------------------------------------|
| `prompt`         | Depends on `taskType`                  | See taskType-specific mapping below                                         |
| `size`           | `imageGenerationConfig.width/height`   | Discrete sizes: 512, 768, 1024, 1152, 1216, 1344, 1536, 2048                |
| `quality`        | `imageGenerationConfig.quality`        | "high" → "premium"                                                          |
| `n`              | `imageGenerationConfig.numberOfImages` | 1-5 images                                                                  |

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

#### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Stability AI Models { #stability-ai-extra-features }

**Basic Usage (Standard OpenAI Parameters):**

```bash
curl -X POST "$BASE/v1/images/generations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "stability.stable-image-ultra-v1:1",
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

| Model                             | Native Formats  | Best For                              |
|-----------------------------------|-----------------|---------------------------------------|
| stability.sd3-5-large-v1:0        | png, jpeg, webp | High quality, versatile compositions  |
| stability.stable-image-core-v1:1  | png, jpeg       | Balanced quality and speed            |
| stability.stable-image-ultra-v1:1 | png, jpeg       | Premium quality and detail            |

A format outside the model's native set is produced by re-encoding the returned image, so any model answers `output_format: "webp"`.

!!! info "Full Parameter Reference"
    For all Stability AI parameters, see [Stability AI documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-stability-diffusion.html)

## Limits and behaviour to know

- **`n` is capped by the model, not by the endpoint.** The endpoint accepts 1-10; the effective
  maximum is model-dependent, and Amazon Titan and Nova Canvas stop at 5. Stability AI models
  have no batch call, so each of the `n` images is a separate Bedrock request.
- **`partial_images` never produces a preview.** No available model streams partial images, so
  the value (0-3) is accepted and ignored and each finished image is sent as a single
  `image_generation.completed` event.
- **`moderation` accepts only its default `auto`.** Any other value is rejected with an error;
  content filtering is configured with the guardrail [request headers](#available-request-headers)
  instead.
- **`background` has no transparent mode.** `auto` and `opaque` are accepted, `transparent` is
  not, and every response reports `opaque`.
- **`quality` and `style` reach only the models that have the control.** A model with no
  equivalent setting accepts the field and ignores it.
- **An unsupported provider-specific parameter returns HTTP 400.** Compatible ones are forwarded
  to Bedrock — see [Provider-Specific Parameters](#provider-specific-parameters).
- **OpenAI image model names resolve only once you map them**, as described in the
  [Models](#model-support) section.

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
    "model": "stability.stable-image-core-v1:1",
    "prompt": "A serene mountain landscape at sunset"
  }'
```

!!! info "Detailed Documentation"
    For complete information about these headers, configuration options, and use cases, see:

    - [Bedrock Guardrails Configuration](operations_configuration_bedrock.md#bedrock-guardrails)
    - [Service Tier and Performance Configuration](operations_configuration_bedrock.md#bedrock-service-tier-and-performance-configuration)

## Try it { #try-it-now }

**Generate image (URL response):**

```bash
curl -X POST "$BASE/v1/images/generations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain landscape at sunset, photorealistic",
    "model": "stability.stable-image-core-v1:1",
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
    "model": "stability.stable-image-core-v1:1",
    "response_format": "b64_json"
  }'
```

**Stream generation:**

Each image is sent as a single completed event; `partial_images` is accepted but never produces a preview.

```bash
curl -N -X POST "$BASE/v1/images/generations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "An abstract watercolor painting of emotions",
    "model": "stability.stable-image-core-v1:1",
    "stream": true,
    "n": 2
  }'
```

## Next steps

Next: [Models API](api_openai_models.md) · [Images Edits API](api_openai_images_edits.md) · [Images Variations API](api_openai_images_variations.md) · [IAM permissions](operations_iam_permissions.md)
