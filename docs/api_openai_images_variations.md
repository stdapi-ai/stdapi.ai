# Images API - Image Variations

Create variations of existing images using AWS Bedrock image models through an OpenAI-compatible interface.

## Why Choose Image Variations?

<div class="grid cards" markdown>

- :material-image-multiple: __Multiple Variations__
  <br>Generate diverse versions of an existing image while maintaining composition.

- :material-palette-swatch: __Artistic Exploration__
  <br>Explore different artistic interpretations and styles.

- :material-auto-fix: __Quick Iterations__
  <br>Rapidly create variations without manual editing.

- :material-aws: __Scalable Infrastructure__
  <br>Generate variations at scale with AWS Bedrock infrastructure.

</div>

## Quick Start: Available Endpoint

| Endpoint                | Method | What It Does                           | Powered By               |
|-------------------------|--------|----------------------------------------|--------------------------|
| `/v1/images/variations` | POST   | Create variations of an existing image | AWS Bedrock Image Models |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                        |                  Status                  | Notes                                                                    |
|--------------------------------|:----------------------------------------:|--------------------------------------------------------------------------|
| **Variations**                 |                                          |                                                                          |
| Image-to-image (`/variations`) |   :material-check-circle:{ .success }    | Create variations of existing images                                     |
| **Parameters**                 |                                          |                                                                          |
| `image`                        |   :material-check-circle:{ .success }    | Source image file (required)                                             |
| `model`                        |   :material-check-circle:{ .success }    | Required parameter                                                       |
| `n` (number of images)         |   :material-check-circle:{ .success }    | Generate multiple variations per request (1-10, default: 1)              |
| `size` (WIDTHxHEIGHT)          |   :material-check-circle:{ .success }    | Output dimensions (default: 1024x1024, format validated)                 |
| `response_format`              |   :material-check-circle:{ .success }    | `url` or `b64_json` (default: `url`)                                     |
| **Output**                     |                                          |                                                                          |
| URL response format            |   :material-check-circle:{ .success }    | Temporary URLs to variation images (requires AWS_S3_BUCKET)              |
| Base64 JSON format             |   :material-check-circle:{ .success }    | Inline base64-encoded images                                             |
| PNG format                     |   :material-check-circle:{ .success }    | Lossless image output                                                    |
| JPEG format                    |       :material-cog:{ .model-dep }       | Lossy compression (model-specific)                                       |
| WEBP format                    |       :material-cog:{ .model-dep }       | Modern format with compression (model-specific)                          |
| **Usage tracking**             |                                          |                                                                          |
| Input image tokens             |   :material-check-circle:{ .success }    | Count of input images (always 1 for variations)                          |
| Output image tokens            |   :material-check-circle:{ .success }    | Number of variations generated (`n` parameter)                           |
| **Other**                      |                                          |                                                                          |
| `user`                         |   :material-minus-circle:{ .partial }    | Logged but not used for abuse monitoring                                 |
| Extra parameters via form data |   :material-check-circle:{ .success }    | Provider-specific parameters passed through                              |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep } **Available on Select Models** — Check your model's capabilities
* :material-minus-circle:{ .partial } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported } **Unsupported** — Not available in this implementation

</div>

!!! warning "Configuration Required"
    You must configure the `AWS_S3_BUCKET` environment variable with a bucket to use the URL response format.

## Supported Models

### ![Amazon](styles/logo_amazon.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Models

| Model                             | Supported Task Types                                       | Notes                                                                                                       |
|-----------------------------------|------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| amazon.nova-canvas-v1:0           | `IMAGE_VARIATION`, `TEXT_IMAGE`, `COLOR_GUIDED_GENERATION` | Supports standard variations plus text-guided conditioning and color-guided generation with 8 style presets |
| amazon.titan-image-generator-v1   | `IMAGE_VARIATION`                                          | Basic variation support with similarity control                                                             |
| amazon.titan-image-generator-v2:0 | `IMAGE_VARIATION`, `TEXT_IMAGE`, `COLOR_GUIDED_GENERATION` | Enhanced with text-guided conditioning (CANNY_EDGE, SEGMENTATION) and color-guided generation               |

### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Stability AI Models

| Model                      | Notes                         |
|----------------------------|-------------------------------|
| stability.sd3-5-large-v1:0 | Image-to-image transformation |

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

## Advanced Features

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
  -F "colorGuidedGenerationParams[colors][]=\#FF6B6B" \
  -F "colorGuidedGenerationParams[colors][]=\#4ECDC4" \
  -F "colorGuidedGenerationParams[colors][]=\#45B7D1"
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
| `style`          | `textToImageParams.style`              | For TEXT_IMAGE task                 |
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
  -F "colorGuidedGenerationParams[colors][]=\#FF6B35" \
  -F "colorGuidedGenerationParams[colors][]=\#F7931E" \
  -F "colorGuidedGenerationParams[colors][]=\#FDC830"
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
