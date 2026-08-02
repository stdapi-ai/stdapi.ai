---
title: Images Variations API - Amazon Bedrock Image Variations
description: Generate variations of existing images with Amazon Bedrock. OpenAI-compatible API for creating alternative versions while maintaining style and composition.
keywords: image variations API, AI image variations, image style transfer, similar images API, AWS image processing, OpenAI variations, image alternatives
---

# Images API - Image Variations

Create variations of existing images using Amazon Bedrock image models through an OpenAI-compatible interface.

## Why Choose the Image Variations API?

<div class="grid cards" markdown>

- :material-image-multiple: __Multiple Variations__
  <br>Generate diverse versions of an existing image while maintaining composition.

- :material-palette-swatch: __Artistic Exploration__
  <br>Explore different artistic interpretations and styles.

- :material-auto-fix: __Quick Iterations__
  <br>Rapidly create variations without manual editing.

- :material-aws: __Flexible Models__
  <br>Amazon Titan, Amazon Nova Canvas, and Stability AI SD3.5 — each with unique variation modes (standard variation, text-guided conditioning, color-guided generation).

</div>

## Quick Start: Available Endpoint

| Endpoint                | Method | What It Does                           | Powered By                  | MCP Tool               |
|-------------------------|--------|----------------------------------------|-----------------------------|------------------------|
| `/v1/images/variations` | `POST` | Create variations of an existing image | Amazon Bedrock Image Models | `openai_image_variation` |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                        |                  Status                  | Notes                                                                                                                                                           |
|--------------------------------|:----------------------------------------:|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Variations**                 |                                          |                                                                                                                                                                 |
| Image-to-image (`/variations`) |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Create variations of existing images                                                                                                                            |
| **Parameters**                 |                                          |                                                                                                                                                                 |
| `image`                        |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Source image file (required)                                                                                                                                    |
| `model`                        |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Required parameter                                                                                                                                              |
| `n` (number of images)         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Multiple variations per request; accepted range is 1-10 (default: 1), but the effective maximum is model-dependent (e.g. Amazon Titan and Nova Canvas cap at 5) |
| `size` (WIDTHxHEIGHT)          |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Output dimensions (default: 1024x1024, format validated)                                                                                                        |
| `response_format`              |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | `url` or `b64_json` (default: `url`)                                                                                                                            |
| **Output**                     |                                          |                                                                                                                                                                 |
| URL response format            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Temporary download URLs, valid for 60 minutes (requires AWS_S3_BUCKET)                                                                                         |
| Base64 JSON format             |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Inline base64-encoded images                                                                                                                                    |
| PNG format                     |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Default output format                                                                                                                                           |
| JPEG format                    |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Via the provider-specific `output_format` extra parameter on supporting models (`/variations` has no `output_format` parameter)                                 |
| WebP format                    |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Via the provider-specific `output_format` extra parameter on supporting models (`/variations` has no `output_format` parameter)                                 |
| **Usage tracking**             |                                          |                                                                                                                                                                 |
| Input image tokens             |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Count of input images (always 1 for variations)                                                                                                                 |
| Output image tokens            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Sourced from AWS billing data when available; falls back to the image count (`n`)                                                                               |
| **Other**                      |                                          |                                                                                                                                                                 |
| `user`                         |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Logged but not used for abuse monitoring                                                                                                                        |
| Extra parameters via form data | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Provider-specific parameters passed through                                                                                                                     |
| JSON body request format       | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Reference images via Files API ID or URL instead of file upload                                                                                                 |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep role="img" aria-label="Model-dependent" } **Available on Select Models** — Check your model's capabilities
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Supported with limitations
* :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } **Extra Feature** — Enhanced capability beyond OpenAI API

</div>

## Model Support

### ![Amazon](styles/logo_amazon.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Models

| Model                             | Supported Task Types                                       | Notes                                                                                                       |
|-----------------------------------|------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| amazon.nova-canvas-v1:0 (legacy)  | `IMAGE_VARIATION`, `TEXT_IMAGE`, `COLOR_GUIDED_GENERATION` | Supports standard variations plus text-guided conditioning and color-guided generation with 8 style presets |
| amazon.titan-image-generator-v1 (legacy)  | `IMAGE_VARIATION`                                          | Basic variation support with similarity control                                                             |
| amazon.titan-image-generator-v2:0 (legacy) | `IMAGE_VARIATION`, `TEXT_IMAGE`, `COLOR_GUIDED_GENERATION` | Enhanced with text-guided conditioning (CANNY_EDGE, SEGMENTATION) and color-guided generation               |

!!! note "Legacy Amazon Image Models"
    AWS has scheduled `amazon.nova-canvas-v1:0` and the Titan image models to reach end of life on September 30, 2026. Deployments with existing access can keep using them until then (legacy models are hidden unless [`AWS_BEDROCK_LEGACY=true`](operations_configuration.md#bedrock-legacy)); the Stability AI Stable Image family is the long-term successor.

### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Stability AI Models

| Model                      | Notes                         |
|----------------------------|-------------------------------|
| stability.sd3-5-large-v1:0 | Image-to-image transformation |

!!! info "No Built-In Aliases for OpenAI Image Model Names"
    OpenAI's default image model names (`dall-e-2`, `dall-e-3`, `gpt-image-1`) have **no built-in alias**, so requests using them fail with a model-not-found error — the most common first-call issue. Pass one of the model IDs above, or map the OpenAI names to your preferred models with [`MODEL_ALIASES`](operations_configuration.md#model-aliases).

!!! warning "Configuration Required"
    You must configure the `AWS_S3_BUCKET` environment variable with a bucket to use the URL response format.

## Advanced Features

### Request Formats

The `/v1/images/variations` endpoint accepts two request formats:

#### Multipart Form-Data (Binary Uploads)

The classic format — upload an image file directly via the `image` field.

```bash
curl -X POST "$BASE/v1/images/variations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F image=@input.png \
  -F model="amazon.nova-canvas-v1:0"
```

#### JSON Body (Files API or URL References) :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" }

Reference an image already stored in the Files API or accessible via URL. Send `Content-Type: application/json` with an `image` object containing either `file_id` or `image_url`:

```bash
# Variation from a Files API file ID
curl -X POST "$BASE/v1/images/variations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-canvas-v1:0",
    "image": {"file_id": "file-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"},
    "n": 2,
    "size": "1024x1024"
  }'

# Variation from an HTTP URL
curl -X POST "$BASE/v1/images/variations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-canvas-v1:0",
    "image": {"image_url": "https://example.com/photo.png"},
    "response_format": "b64_json"
  }'
```

**`image` object fields:**

| Field       | Type   | Description                                              |
|-------------|--------|----------------------------------------------------------|
| `file_id`   | string | Files API file identifier (`file-*` or `file_*` prefix)  |
| `image_url` | string | HTTP/HTTPS URL, data URI (`data:image/png;base64,...`), S3 URI (`s3://bucket/key`), or Files API reference (`file-id:file-<id>` — see [Files API](api_openai_files.md#referencing-uploaded-files-via-the-file-id-uri-scheme)) |

Exactly one of `file_id` or `image_url` must be provided. `image` may also be
a plain reference string (equivalent to `image_url`) — the shape MCP clients
derive from the tool schema:

```json
{"model": "amazon.nova-canvas-v1:0", "image": "data:image/png;base64,..."}
```

!!! tip "Workflow Integration"
    The JSON body format works seamlessly with the [Files API](api_openai_files.md): upload images once, reuse them across multiple variation requests by file ID without re-uploading.

### Provider-Specific Parameters

#### ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Titan Image Generator

**Basic Usage (Standard OpenAI Parameters):**

```bash
curl -X POST "$BASE/v1/images/variations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@input.png \
  -F model="amazon.titan-image-generator-v2:0"
```

**Parameter Mapping:**

| OpenAI Parameter | Maps to                                | Notes                               |
|------------------|----------------------------------------|-------------------------------------|
| `image`          | Depends on `taskType`                  | See taskType-specific mapping below |
| `size`           | `imageGenerationConfig.width/height`   | Fixed sizes (512-2048)              |
| `n`              | `imageGenerationConfig.numberOfImages` | 1-5 variations                      |

**TaskType-Specific Parameter Mapping:**

| taskType                    | `image` maps to                              |
|-----------------------------|----------------------------------------------|
| `IMAGE_VARIATION` (default) | `imageVariationParams.images`                |
| `TEXT_IMAGE`                | `textToImageParams.conditionImage`           |
| `COLOR_GUIDED_GENERATION`   | `colorGuidedGenerationParams.referenceImage` |

**Advanced Variation Modes (with form fields):**

Default `taskType` is `"IMAGE_VARIATION"`.

Available task types:

- `"IMAGE_VARIATION"` - Standard image variations
- `"TEXT_IMAGE"` - Text-guided generation with condition image
- `"COLOR_GUIDED_GENERATION"` - Color palette-based variations

```bash
# Text-Guided Condition Image
curl -X POST "$BASE/v1/images/variations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@input.png \
  -F model="amazon.titan-image-generator-v2:0" \
  -F taskType="TEXT_IMAGE" \
  -F "textToImageParams[text]=Photorealistic version"

# Color-Guided Variations
curl -X POST "$BASE/v1/images/variations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@input.png \
  -F model="amazon.titan-image-generator-v2:0" \
  -F taskType="COLOR_GUIDED_GENERATION" \
  -F "colorGuidedGenerationParams[colors][]=#FF6B6B" \
  -F "colorGuidedGenerationParams[colors][]=#4ECDC4" \
  -F "colorGuidedGenerationParams[colors][]=#45B7D1"
```

!!! info "Full Parameter Reference"
    For all parameters and task types, see [Amazon Titan Image Generator documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html)

#### ![Amazon Nova](styles/logo_amazon_nova.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Nova Canvas

**Basic Usage (Standard OpenAI Parameters):**

```bash
curl -X POST "$BASE/v1/images/variations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@input.png \
  -F model="amazon.nova-canvas-v1:0"
```

**Parameter Mapping:**

| OpenAI Parameter | Maps to                                | Notes                               |
|------------------|----------------------------------------|-------------------------------------|
| `image`          | Depends on `taskType`                  | See taskType-specific mapping below |
| `size`           | `imageGenerationConfig.width/height`   | Flexible (320-4096)                 |
| `style`          | `textToImageParams.style`              | Not a standard variations field — pass via extra param `textToImageParams[style]` (TEXT_IMAGE task) |
| `n`              | `imageGenerationConfig.numberOfImages` | 1-5 variations                      |

**TaskType-Specific Parameter Mapping:**

| taskType                    | `image` maps to                              |
|-----------------------------|----------------------------------------------|
| `IMAGE_VARIATION` (default) | `imageVariationParams.images`                |
| `TEXT_IMAGE`                | `textToImageParams.conditionImage`           |
| `COLOR_GUIDED_GENERATION`   | `colorGuidedGenerationParams.referenceImage` |

**Advanced Variation Modes (with form fields):**

Default `taskType` is `"IMAGE_VARIATION"`.

Available task types:

- `"IMAGE_VARIATION"` - Standard image variations
- `"TEXT_IMAGE"` - Text-guided generation with condition image
- `"COLOR_GUIDED_GENERATION"` - Color palette-based variations

```bash
# Text-to-Image with Condition
curl -X POST "$BASE/v1/images/variations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@input.png \
  -F model="amazon.nova-canvas-v1:0" \
  -F taskType="TEXT_IMAGE" \
  -F "textToImageParams[text]=Photorealistic version"

# Color-Guided Variations
curl -X POST "$BASE/v1/images/variations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@input.png \
  -F model="amazon.nova-canvas-v1:0" \
  -F taskType="COLOR_GUIDED_GENERATION" \
  -F "colorGuidedGenerationParams[colors][]=#FF6B35" \
  -F "colorGuidedGenerationParams[colors][]=#F7931E" \
  -F "colorGuidedGenerationParams[colors][]=#FDC830"
```

!!! info "Full Parameter Reference"
    For all parameters, styles, and task types, see [Amazon Nova Canvas documentation](https://docs.aws.amazon.com/nova/latest/userguide/image-generation.html)

#### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Stability AI Models

**Basic Usage (Standard OpenAI Parameters):**

```bash
curl -X POST "$BASE/v1/images/variations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@input.png \
  -F model="stability.sd3-5-large-v1:0"
```

**Parameter Mapping:**

| OpenAI Parameter | Maps to          | Notes                                         |
|------------------|------------------|-----------------------------------------------|
| `image`          | `image` (base64) | Source image for variation                    |
| `size`           | `aspect_ratio`   | Inferred from size (e.g., 1024x1024 → "1:1")  |
| `n`              | Multiple calls   | Each variation is a separate request          |

**Model-Specific Features:**

| Model                      | Output Formats  | Notes                         |
|----------------------------|-----------------|-------------------------------|
| stability.sd3-5-large-v1:0 | png, jpeg, webp | Image-to-image transformation |

!!! info "Full Parameter Reference"
    For all Stability AI parameters, see [Stability AI documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-stability-diffusion.html)

## Available Request Headers

This endpoint supports the same standard Bedrock headers as the other images endpoints: guardrail headers (`X-Amzn-Bedrock-GuardrailIdentifier`, `X-Amzn-Bedrock-GuardrailVersion`, `X-Amzn-Bedrock-Trace`) and performance headers (`X-Amzn-Bedrock-Service-Tier`, `X-Amzn-Bedrock-PerformanceConfig-Latency`). All headers are optional and can be combined as needed.

See the [Images Generation API headers reference](api_openai_images_generations.md#available-request-headers) for the header tables, valid values, and configuration links.

**Example with headers:**

```bash
curl -X POST "$BASE/v1/images/variations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "X-Amzn-Bedrock-Service-Tier: priority" \
  -F image=@input.png \
  -F model="amazon.nova-canvas-v1:0"
```

## Try It Now

**Create a simple variation:**

```bash
curl -X POST "$BASE/v1/images/variations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@input.png \
  -F model="amazon.titan-image-generator-v2:0"
```

**Create multiple variations:**

```bash
curl -X POST "$BASE/v1/images/variations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@input.png \
  -F model="amazon.titan-image-generator-v2:0" \
  -F n=3
```

**Base64 response format:**

```bash
curl -X POST "$BASE/v1/images/variations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@input.png \
  -F model="amazon.titan-image-generator-v2:0" \
  -F response_format="b64_json"
```

---

**Ready to explore new versions of your images?** Discover available image models in the [Models API](api_openai_models.md).
