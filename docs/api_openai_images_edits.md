---
title: Images Edits API - Amazon Bedrock Image Editing
description: Edit and transform images with Amazon Bedrock image models. OpenAI-compatible API for image modifications, inpainting, and transformations.
keywords: image editing API, AI image editor, inpainting API, image modification API, AWS image editing, OpenAI image edit, image transformation API
---

# Images API - Image Editing

Edit images using inpainting with Amazon Bedrock image models through an OpenAI-compatible interface.

## Why Choose the Image Editing API?

<div class="grid cards" markdown>

- :material-image-edit: __Precise Control__
  <br>Edit specific regions of images while preserving the rest.

- :material-palette-advanced: __Creative Freedom__
  <br>Add, remove, or modify elements in existing images with AI assistance.

- :material-layers-triple: __Flexible Masking__
  <br>Define edit regions with an explicit mask image, using either alpha transparency or black/white pixels.

- :material-aws: __Scalable Infrastructure__
  <br>Edit images at scale with Amazon Bedrock infrastructure.

</div>

## Quick Start: Available Endpoint

| Endpoint           | Method | What It Does                            | Powered By                  | MCP Tool           |
|--------------------|--------|-----------------------------------------|-----------------------------|--------------------|
| `/v1/images/edits` | `POST` | Edit images using prompts and masks     | Amazon Bedrock Image Models | `openai_image_edit` |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                        |                  Status                  | Notes                                                                                                                                          |
|--------------------------------|:----------------------------------------:|------------------------------------------------------------------------------------------------------------------------------------------------|
| **Editing**                    |                                          |                                                                                                                                                |
| Image-to-image (`/edits`)      |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Edit images with prompts and masks                                                                                                             |
| **Request Formats**            |                                          |                                                                                                                                                |
| Multipart form-data            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Binary file uploads via `image` / `image[]` / `mask` fields                                                                                    |
| JSON body                      | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Structured `images` array with Files API IDs or URLs (the OpenAI edits API is multipart-only)                                                  |
| **Parameters**                 |                                          |                                                                                                                                                |
| `image` / `image[]`            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | PNG image(s) to edit; most models accept exactly one source image and reject requests providing more with an error                             |
| `images` (JSON)                |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Array of `{file_id}` or `{image_url}` references (JSON body)                                                                                   |
| `prompt`                       |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Text description of desired changes                                                                                                            |
| `mask`                         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Optional mask defining edit regions; models that do not use a mask reject requests that include one                                            |
| `n` (number of images)         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Multiple images per request; accepted range is 1-10, but the effective maximum is model-dependent (e.g. Amazon Titan and Nova Canvas cap at 5) |
| `size` (WIDTHxHEIGHT)          |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Output dimensions (default: 1024x1024, format validated; `auto` resolves to the default) |
| `model`                        |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Required parameter                                                                                                                             |
| `response_format`              |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | `url` or `b64_json` (default: `url`)                                                                                                           |
| `output_format`                |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | `png`, `jpeg`, or `webp` (model-specific)                                                                                                      |
| `output_compression`           |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Compression level 1-100% (default: 100)                                                                                                        |
| `quality`                      |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Quality setting (default: `auto`, supports OpenAI & model-specific); accepted and ignored by models with no quality control                    |
| `stream`                       |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Generate images in streaming mode, emitting the endpoint's `image_edit.partial_image` and `image_edit.completed` events                        |
| `partial_images`               | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted (0-3) but ignored — no available model currently streams partial images; the final image is always sent as a single event             |
| `background`                   |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Accepts `auto` (default) and `opaque`; `transparent` is unsupported — responses report `opaque`                                                |
| `input_fidelity`               | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted for OpenAI API compatibility and ignored (always behaves as `low`)                                                                    |
| **Output**                     |                                          |                                                                                                                                                |
| URL response format            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Temporary download URLs, valid for 60 minutes (requires AWS_S3_BUCKET)                                                                        |
| Base64 JSON format             |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Inline base64-encoded images                                                                                                                   |
| PNG format                     |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Lossless image output                                                                                                                          |
| JPEG format                    |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Lossy compression (model-specific)                                                                                                             |
| WebP format                    |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Modern format with compression (model-specific)                                                                                                |
| Streaming response             |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Server-sent events with final images (no partial previews)                                                                                     |
| **Usage tracking**             |                                          |                                                                                                                                                |
| Input text tokens              |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Sourced from AWS billing data when available; remainder after subtracting image tokens                                                         |
| Input image tokens             |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Count of input images (image files + mask file), capped at the billed input tokens                                                             |
| Output image tokens            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Sourced from AWS billing data when available; falls back to the image count (`n`)                                                              |
| **Other**                      |                                          |                                                                                                                                                |
| `user`                         |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Logged but not used for abuse monitoring                                                                                                       |
| Extra parameters via form data | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Provider-specific parameters passed through                                                                                                    |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep role="img" aria-label="Model-dependent" } **Available on Select Models** — Check your model's capabilities
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } **Extra Feature** — Enhanced capability beyond OpenAI API

</div>

## Model Support

!!! info "Model Support"
    **Inpainting** (mask-based editing) is supported by **Amazon Nova Canvas**, **Amazon Titan Image Generator**, and **Stability AI** inpaint models.

    **Image-to-image** (transformation without masks) is supported by **Stability AI** text-to-image models (SD3.5, Stable Image Core, Stable Image Ultra).

    **Upscale** (resolution enhancement) is supported by **Stability AI** upscale models (creative, conservative, fast).

    **Style Transfer** (applying reference image style) is supported by **Stability AI** style transfer models.

    **Search-based editing** (find & replace/recolor objects) is supported by **Stability AI** search models.

    **Background removal** is supported by **Amazon Titan Image Generator v2**, **Amazon Nova Canvas**, and **Stability AI** remove background model.

### ![Amazon](styles/logo_amazon.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Models

| Model                             | Supported Task Types                                                              | Mask Support                                                                    | Notes                                                                               |
|-----------------------------------|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| amazon.nova-canvas-v1:0 (legacy)  | `TEXT_IMAGE`, `INPAINTING`, `OUTPAINTING`, `BACKGROUND_REMOVAL`, `VIRTUAL_TRY_ON` | ✅ Required for inpainting/outpainting<br>✅ Used as reference for virtual try-on | Supports multiple editing modes including advanced virtual try-on with 3 mask types |
| amazon.titan-image-generator-v1 (legacy)  | `INPAINTING`, `OUTPAINTING`                                                       | ✅ Required for inpainting/outpainting                                           | Supports text-based mask prompts as alternative to mask images                      |
| amazon.titan-image-generator-v2:0 (legacy) | `INPAINTING`, `OUTPAINTING`, `BACKGROUND_REMOVAL`                                 | ✅ Required for inpainting/outpainting<br>❌ Rejected for background removal      | Enhanced features including background removal without mask                         |

!!! note "Legacy Amazon Image Models"
    AWS has scheduled `amazon.nova-canvas-v1:0` and the Titan image models to reach end of life on September 30, 2026. Deployments with existing access can keep using them until then (legacy models are hidden unless [`AWS_BEDROCK_LEGACY=true`](operations_configuration.md#bedrock-legacy)); the Stability AI Stable Image family is the long-term successor.

!!! info "Amazon Nova Canvas Default Behavior"
    **`amazon.nova-canvas-v1:0`** automatically selects the task type based on the presence of a mask when no `taskType` is explicitly provided:

    - **No mask provided** → Uses `TEXT_IMAGE` by default
    - **Mask provided** → Uses `INPAINTING` by default

### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Stability AI Models

#### Image-to-Image Models

| Model                             | Prompt Usage           | Mask Usage           | Extra Parameters Required | Notes                             |
|-----------------------------------|------------------------|----------------------|---------------------------|-----------------------------------|
| stability.sd3-5-large-v1:0        | Guides transformation  | Rejected if provided | None                      | Transform images with prompt      |

#### Upscale Models

| Model                                      | Prompt Usage           | Mask Usage           | Extra Parameters Required | Notes                                   |
|--------------------------------------------|------------------------|----------------------|---------------------------|-----------------------------------------|
| stability.stable-creative-upscale-v1:0     | Guides upscaling       | Rejected if provided | None                      | Prompt-guided upscaling with creativity |
| stability.stable-conservative-upscale-v1:0 | Guides upscaling       | Rejected if provided | None                      | Detail-preserving upscaling             |
| stability.stable-fast-upscale-v1:0         | **Not used**           | Rejected if provided | None                      | Fast 4x upscaling without prompt        |

#### Edit Models

| Model                                         | Prompt Usage              | Mask Usage                   | Extra Parameters Required | Notes                            |
|-----------------------------------------------|---------------------------|------------------------------|---------------------------|----------------------------------|
| stability.stable-image-inpaint-v1:0           | Guides inpainting         | Optional (marks edit region) | None                      | Fill masked regions              |
| stability.stable-outpaint-v1:0                | Guides outpainting        | Rejected if provided         | None                      | Extend image beyond borders      |
| stability.stable-image-search-recolor-v1:0    | Describes new color       | Rejected if provided         | `select_prompt`           | Recolor objects by search prompt |
| stability.stable-image-search-replace-v1:0    | Describes replacement     | Rejected if provided         | `search_prompt`           | Replace objects by search prompt |
| stability.stable-image-erase-object-v1:0      | **Not used**              | Required (marks object)      | None                      | Remove objects with mask         |
| stability.stable-image-remove-background-v1:0 | **Not used**              | Rejected if provided         | None                      | Automatic background removal     |

#### Control Models

| Model                                         | Prompt Usage         | Mask Usage           | Extra Parameters Required | Notes                           |
|-----------------------------------------------|----------------------|----------------------|---------------------------|---------------------------------|
| stability.stable-image-control-sketch-v1:0    | Guides generation    | Rejected if provided | None                      | Generate from sketch            |
| stability.stable-image-control-structure-v1:0 | Guides generation    | Rejected if provided | None                      | Structure-preserving generation |

#### Style Models

| Model                                   | Prompt Usage          | Mask Usage                             | Extra Parameters Required | Notes                         |
|-----------------------------------------|-----------------------|----------------------------------------|---------------------------|-------------------------------|
| stability.stable-image-style-guide-v1:0 | Guides style          | Rejected if provided                   | None                      | Extract and apply style       |
| stability.stable-style-transfer-v1:0    | Guides style transfer | Required (repurposed as `style_image`) | None                      | Transfer style between images |

!!! note "Output Formats"
    All models support standard OpenAI output formats (`png`, `jpeg`, `webp`) via the `output_format` parameter. Format availability may vary by model - unsupported formats will automatically fall back to PNG.

!!! warning "Extra Parameters Required"
    Some models require parameters **beyond the standard OpenAI API**:

    - **`stability.stable-image-search-recolor-v1:0`**: Requires `select_prompt` form field
    - **`stability.stable-image-search-replace-v1:0`**: Requires `search_prompt` form field

    **Models that don't use prompt**: `stability.stable-fast-upscale-v1:0`, `stability.stable-image-erase-object-v1:0`, `stability.stable-image-remove-background-v1:0` - provide empty string or omit the `prompt` parameter.

    All other Stability models use only standard OpenAI parameters (`image`, `prompt`, and optionally `mask`).

!!! info "No Built-In Aliases for OpenAI Image Model Names"
    OpenAI's default image model names (`dall-e-2`, `dall-e-3`, `gpt-image-1`) have **no built-in alias**, so requests using them fail with a model-not-found error — the most common first-call issue. Pass one of the model IDs above, or map the OpenAI names to your preferred models with [`MODEL_ALIASES`](operations_configuration.md#model-aliases).

!!! warning "Configuration Required"
    You must configure the `AWS_S3_BUCKET` environment variable with a bucket to use the URL response format.

## Advanced Features

### Request Formats

The `/v1/images/edits` endpoint accepts two request formats:

#### Multipart Form-Data (Binary Uploads)

The classic format — upload image files directly. Use `image` (single) or `image[]` (multiple) for source images and `mask` for the optional edit mask.

```bash
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F image=@source.png \
  -F mask=@mask.png \
  -F prompt="A red apple on a wooden table" \
  -F model="amazon.nova-canvas-v1:0"
```

#### JSON Body (Files API or URL References) :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" }

The modern format — reference images already stored in the Files API or accessible via URL. Send `Content-Type: application/json` with an `images` array, where each element has either `file_id` or `image_url`:

```bash
# Edit using a Files API file ID
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-canvas-v1:0",
    "prompt": "A red apple on a wooden table",
    "images": [{"file_id": "file-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}],
    "response_format": "b64_json",
    "size": "1024x1024"
  }'

# Edit using an HTTP URL
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-canvas-v1:0",
    "prompt": "Add a dramatic sky",
    "images": [{"image_url": "https://example.com/photo.png"}],
    "mask": {"file_id": "file-mxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"},
    "size": "1024x1024"
  }'
```

**`ImageRef` object** (used in `images` array and `mask` field):

| Field       | Type   | Description                                              |
|-------------|--------|----------------------------------------------------------|
| `file_id`   | string | Files API file identifier (`file-*` or `file_*` prefix)  |
| `image_url` | string | HTTP/HTTPS URL, data URI (`data:image/png;base64,...`), S3 URI (`s3://bucket/key`), or Files API reference (`file-id:file-<id>` — see [Files API](api_openai_files.md#referencing-uploaded-files-via-the-file-id-uri-scheme)) |

Exactly one of `file_id` or `image_url` must be provided per `ImageRef`.

!!! tip "Workflow Integration"
    The JSON body format works seamlessly with the [Files API](api_openai_files.md): upload images once, reuse them across multiple edit requests by file ID without re-uploading.

### How Image Editing Works

#### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Image-to-Image (Stability AI Models)

Stability AI models support image-to-image transformation without masks. The source image is transformed according to the prompt:

```bash
# Transform a photo into an oil painting style
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@photo.png \
  -F prompt="Transform into an oil painting style" \
  -F model="stability.sd3-5-large-v1:0"
```

!!! warning "Mask Not Supported"
    Stability AI image-to-image models do not support mask-based editing. Providing a `mask` parameter will result in an error.

#### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Upscale (Stability AI)

Upscale models increase image resolution while preserving quality:

```bash
# Fast upscaling (4x) - no prompt parameter needed
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@low_res.png \
  -F model="stability.stable-fast-upscale-v1:0"
```

!!! note "Upscale Characteristics"
    - **Fast Upscale**: Conservative 4x upscaling that preserves original details
    - No prompt parameter needed or used
    - Best for enlarging photos and preserving original content

#### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Style Transfer (Stability AI)

Apply visual characteristics from one image to another. The `mask` parameter is used to pass the style reference image:

```bash
# Apply style from reference image to target image
# image: content image, mask: style reference image
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@content.png \
  -F mask=@style_reference.png \
  -F prompt="Apply artistic style while preserving content" \
  -F model="stability.stable-style-transfer-v1:0"
```

!!! note "Style Transfer Parameter Mapping"
    - **`image`** (required): Target image to apply style to
    - **`mask`** (required): Maps to `style_image` - the reference style image
    - **`prompt`**: Guides the style application process

#### Inpainting with Masks (Amazon Models and Stability AI)

An image submitted without a `mask` is **not** auto-masked from its own transparency:
it is sent as a conditioning image for text-to-image generation instead of an
inpainting edit. To edit specific regions, always provide an explicit `mask`.

**With Explicit Mask:**

Provide an explicit mask image where transparent areas indicate regions to edit:

```bash
# Edit with explicit mask
# image: source image, mask: PNG where transparent areas mark edit regions
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@source.png \
  -F mask=@edit_mask.png \
  -F prompt="A beautiful flower" \
  -F model="amazon.nova-canvas-v1:0"
```

**Mask format**: PNG with alpha channel where transparent pixels indicate regions to
edit, opaque pixels are preserved (standard OpenAI edits-API mask). A mask with an
alpha channel is automatically converted to the black/white RGB format each backend
requires (Nova Canvas, Titan, and the Stability AI inpaint/erase-object models); a
mask that is already black/white RGB (no alpha channel) is passed through unchanged.

### Provider-Specific Parameters

#### ![Amazon Nova](styles/logo_amazon_nova.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Nova Canvas

**Basic Usage (Standard OpenAI Parameters):**

```bash
# Inpainting with mask
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@source.png \
  -F mask=@mask.png \
  -F prompt="A sunset over mountains" \
  -F model="amazon.nova-canvas-v1:0"
```

**Parameter Mapping:**

| OpenAI Parameter    | Maps to                                | Notes                                                       |
|---------------------|----------------------------------------|-------------------------------------------------------------|
| `prompt`            | Depends on `taskType`                  | See taskType-specific mapping below                         |
| `image` / `image[]` | Depends on `taskType`                  | See taskType-specific mapping below (single image required) |
| `mask`              | Depends on `taskType`                  | See taskType-specific mapping below                         |
| `size`              | `imageGenerationConfig.width/height`   | Output dimensions (320-4096)                                |
| `quality`           | `imageGenerationConfig.quality`        | "high" → "premium"                                          |
| `n`                 | `imageGenerationConfig.numberOfImages` | 1-5 images                                                  |

**TaskType-Specific Parameter Mapping:**

| taskType                          | `prompt` maps to                                              | `image` maps to                    | `mask` maps to                      |
|-----------------------------------|---------------------------------------------------------------|------------------------------------|-------------------------------------|
| `TEXT_IMAGE` (default, no mask)   | `textToImageParams.text`                                      | `textToImageParams.conditionImage` | Not used                            |
| `INPAINTING` (default with mask)  | `inPaintingParams.text`                                       | `inPaintingParams.image`           | `inPaintingParams.maskImage`        |
| `OUTPAINTING`                     | `outPaintingParams.text`                                      | `outPaintingParams.image`          | `outPaintingParams.maskImage`       |
| `BACKGROUND_REMOVAL`              | Not used                                                      | `backgroundRemovalParams.image`    | Rejected if provided                |
| `VIRTUAL_TRY_ON` (PROMPT)         | `promptBasedMask.maskPrompt`                                  | `virtualTryOnParams.sourceImage`   | `virtualTryOnParams.referenceImage` |
| `VIRTUAL_TRY_ON` (GARMENT)        | `garmentBasedMask.garmentClass`                               | `virtualTryOnParams.sourceImage`   | `virtualTryOnParams.referenceImage` |
| `VIRTUAL_TRY_ON` (IMAGE)          | `imageBasedMask.maskImage` (Base64 encoded image or data URI) | `virtualTryOnParams.sourceImage`   | `virtualTryOnParams.referenceImage` |

**Advanced Task Types (with form fields):**

Default `taskType` is `"INPAINTING"` when a mask is provided, `"TEXT_IMAGE"` otherwise.

Available task types:

- `"TEXT_IMAGE"` - Prompt-driven transformation using the source image as condition
- `"INPAINTING"` - Fill masked regions
- `"OUTPAINTING"` - Extend image beyond borders
- `"BACKGROUND_REMOVAL"` - Remove background
- `"VIRTUAL_TRY_ON"` - Virtual fashion try-on

```bash
# Outpainting
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@photo.png \
  -F prompt="Extend with a garden" \
  -F model="amazon.nova-canvas-v1:0" \
  -F taskType="OUTPAINTING"

# Background Removal (no prompt needed)
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@photo.png \
  -F model="amazon.nova-canvas-v1:0" \
  -F taskType="BACKGROUND_REMOVAL"

# Virtual Try-On - Prompt-Based (default)
# image: person photo, mask: garment image, prompt: area description
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@person.png \
  -F mask=@garment.png \
  -F prompt="upper body area" \
  -F model="amazon.nova-canvas-v1:0" \
  -F taskType="VIRTUAL_TRY_ON"

# Virtual Try-On - Garment-Based
# image: person photo, mask: garment image, prompt: garment class
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@person.png \
  -F mask=@garment.png \
  -F prompt="UPPER_BODY" \
  -F model="amazon.nova-canvas-v1:0" \
  -F taskType="VIRTUAL_TRY_ON" \
  -F "virtualTryOnParams[maskType]=GARMENT"

# Virtual Try-On - Image-Based Mask
# image: person photo, mask: garment image, prompt: base64 mask image
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@person.png \
  -F mask=@garment.png \
  -F prompt="BASE64_MASK_IMAGE" \
  -F model="amazon.nova-canvas-v1:0" \
  -F taskType="VIRTUAL_TRY_ON" \
  -F "virtualTryOnParams[maskType]=IMAGE"
```

!!! info "Full Parameter Reference"
    For all available parameters and task types, see [Amazon Nova Canvas documentation](https://docs.aws.amazon.com/nova/latest/userguide/image-generation.html)

#### ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Titan Image Generator

**Basic Usage (Standard OpenAI Parameters):**

```bash
# Inpainting with mask
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@source.png \
  -F mask=@mask.png \
  -F prompt="A beautiful garden with flowers" \
  -F model="amazon.nova-canvas-v1:0"
```

**Parameter Mapping:**

| OpenAI Parameter    | Maps to                                | Notes                                                       |
|---------------------|----------------------------------------|-------------------------------------------------------------|
| `prompt`            | Depends on `taskType`                  | See taskType-specific mapping below                         |
| `image` / `image[]` | Depends on `taskType`                  | See taskType-specific mapping below (single image required) |
| `mask`              | Depends on `taskType`                  | See taskType-specific mapping below                         |
| `size`              | `imageGenerationConfig.width/height`   | Fixed sizes (512-2048)                                      |
| `quality`           | `imageGenerationConfig.quality`        | "high" → "premium"                                          |
| `n`                 | `imageGenerationConfig.numberOfImages` | 1-5 images                                                  |

**TaskType-Specific Parameter Mapping:**

| taskType               | `prompt` maps to         | `image` maps to                 | `mask` maps to                |
|------------------------|--------------------------|---------------------------------|-------------------------------|
| `INPAINTING` (default) | `inPaintingParams.text`  | `inPaintingParams.image`        | `inPaintingParams.maskImage`  |
| `OUTPAINTING`          | `outPaintingParams.text` | `outPaintingParams.image`       | `outPaintingParams.maskImage` |
| `BACKGROUND_REMOVAL`   | Not used                 | `backgroundRemovalParams.image` | Rejected if provided          |

**Advanced Task Types (with form fields):**

Default `taskType` is `"INPAINTING"`.

Available task types:

- `"INPAINTING"` - Fill masked regions
- `"OUTPAINTING"` - Extend image beyond borders
- `"BACKGROUND_REMOVAL"` (v2 only) - Remove background

```bash
# Outpainting
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@photo.png \
  -F prompt="Extend with a forest" \
  -F model="amazon.nova-canvas-v1:0" \
  -F taskType="OUTPAINTING"

# Background Removal (v2 only, no prompt needed)
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@photo.png \
  -F model="amazon.nova-canvas-v1:0" \
  -F taskType="BACKGROUND_REMOVAL"
```

!!! info "Full Parameter Reference"
    For all available parameters and task types, see [Amazon Titan Image Generator documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-image.html)

#### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Stability AI Models

**Basic Usage (Standard OpenAI Parameters):**

Most Stability AI models work with standard OpenAI parameters:

```bash
# Image-to-image transformation
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@photo.png \
  -F prompt="A dramatic cinematic scene" \
  -F model="stability.sd3-5-large-v1:0"
```

**Parameter Mapping:**

All Stability AI models use standard OpenAI parameters directly:

| OpenAI Parameter    | Stability Parameter | Notes                                              |
|---------------------|---------------------|----------------------------------------------------|
| `image` / `image[]` | `image`             | Base64-encoded input image (single image required) |
| `prompt`            | `prompt`            | Text description (may be unused for some models)   |
| `mask`              | `mask`              | Base64-encoded mask (model-specific)               |
| `n`                 | Multiple requests   | Generates N images via multiple API calls          |
| `size`              | Model-specific      | Some models support width/height                   |

**Model-Specific Parameters:**

| Model(s)                              | Required Form Fields     | OpenAI `mask` Maps To | Notes                                    |
|---------------------------------------|--------------------------|-----------------------|------------------------------------------|
| `stable-image-search-recolor-v1:0`    | `select_prompt` (string) | Not used              | Identifies object to recolor             |
| `stable-image-search-replace-v1:0`    | `search_prompt` (string) | Not used              | Identifies object to find and replace    |
| `stable-style-transfer-v1:0`          | None (uses `mask` param) | `style_image`         | Mask parameter repurposed as style image |
| `stable-image-erase-object-v1:0`      | None                     | `mask` (required)     | Prompt not used                          |
| `stable-image-remove-background-v1:0` | None                     | Not used              | Prompt not used                          |
| `stable-fast-upscale-v1:0`            | None                     | Not used              | Prompt not used                          |

**Examples:**

```bash
# Search & Replace - requires search_prompt form field
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@input.png \
  -F prompt="a red car" \
  -F model="stability.stable-image-search-replace-v1:0" \
  -F search_prompt="blue car"

# Search & Recolor - requires select_prompt form field
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@input.png \
  -F prompt="bright red color" \
  -F model="stability.stable-image-search-recolor-v1:0" \
  -F select_prompt="car"

# Style Transfer - mask parameter is style image
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@content.png \
  -F mask=@style.png \
  -F prompt="Apply artistic style" \
  -F model="stability.stable-style-transfer-v1:0"

# Erase Object - no prompt needed, mask required
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@input.png \
  -F mask=@object_mask.png \
  -F model="stability.stable-image-erase-object-v1:0"

# Remove Background - no prompt needed
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@input.png \
  -F model="stability.stable-image-remove-background-v1:0"

# Fast Upscale - no prompt needed
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@low_res.png \
  -F model="stability.stable-fast-upscale-v1:0"
```

!!! info "Full Parameter Reference"
    For all Stability AI parameters, see [Stability AI documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-stability-diffusion.html)

## Available Request Headers

This endpoint supports the same standard Bedrock headers as the other images endpoints: guardrail headers (`X-Amzn-Bedrock-GuardrailIdentifier`, `X-Amzn-Bedrock-GuardrailVersion`, `X-Amzn-Bedrock-Trace`) and performance headers (`X-Amzn-Bedrock-Service-Tier`, `X-Amzn-Bedrock-PerformanceConfig-Latency`). All headers are optional and can be combined as needed.

See the [Images Generation API headers reference](api_openai_images_generations.md#available-request-headers) for the header tables, valid values, and configuration links.

**Example with headers:**

```bash
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "X-Amzn-Bedrock-Service-Tier: priority" \
  -F image=@source.png \
  -F prompt="A red apple on a wooden table" \
  -F model="amazon.nova-canvas-v1:0"
```

## Try It Now

### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Image-to-Image with Stability AI

```bash
# Transform image with default strength (0.35)
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@photo.png \
  -F prompt="Transform into a watercolor painting" \
  -F model="stability.sd3-5-large-v1:0"
```

### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Upscale with Stability AI

```bash
# Fast upscaling (4x resolution increase) - no prompt needed
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@low_res.png \
  -F model="stability.stable-fast-upscale-v1:0"
```

### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Style Transfer with Stability AI

```bash
# Apply style from reference image (mask parameter is style image)
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@content.png \
  -F mask=@style_reference.png \
  -F prompt="Apply artistic style" \
  -F model="stability.stable-style-transfer-v1:0"
```

### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Search & Replace with Stability AI

```bash
# Replace objects by search prompt (requires search_prompt form field)
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@input.png \
  -F prompt="a red car" \
  -F model="stability.stable-image-search-replace-v1:0" \
  -F search_prompt="blue car"
```

### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Search & Recolor with Stability AI

```bash
# Recolor objects by search prompt (requires select_prompt form field)
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@input.png \
  -F prompt="bright red color" \
  -F model="stability.stable-image-search-recolor-v1:0" \
  -F select_prompt="car"
```

### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Erase Object with Stability AI

```bash
# Erase object with mask - no prompt needed
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@input.png \
  -F mask=@object_mask.png \
  -F model="stability.stable-image-erase-object-v1:0"
```

### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Remove Background with Stability AI

```bash
# Remove background automatically - no prompt needed
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@input.png \
  -F model="stability.stable-image-remove-background-v1:0"
```

### Inpainting with Amazon Models

```bash
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F image=@image.png \
  -F prompt="A blue ocean with sailboats" \
  -F model="amazon.nova-canvas-v1:0"
```

### Edit with Explicit Mask

```bash
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F image=@source.png \
  -F mask=@mask.png \
  -F prompt="A red sports car" \
  -F model="amazon.nova-canvas-v1:0"
```

### Base64 Response Format

```bash
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F image=@image.png \
  -F prompt="A sunny day with blue sky" \
  -F model="amazon.nova-canvas-v1:0" \
  -F response_format="b64_json"
```

### Multiple Edited Images

```bash
# Generate three edited images from the same source (n parameter)
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F image=@image.png \
  -F prompt="A fantasy castle" \
  -F n=3 \
  -F model="amazon.nova-canvas-v1:0"
```

### Multiple Input Images (Composition)

!!! info "OpenAI Compatible Syntax"
    Use the `image[]` array parameter to provide multiple input images. The API will compose them according to your prompt.

```bash
# Compose multiple images into a single output (like creating a gift basket)
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "image[]=@body-lotion.png" \
  -F "image[]=@bath-bomb.png" \
  -F "image[]=@incense-kit.png" \
  -F "image[]=@soap.png" \
  -F prompt="Create a lovely gift basket with these four items in it" \
  -F model="amazon.nova-canvas-v1:0"
```

!!! warning "Model Support for Multiple Images"
    Not all models support multiple input images. Models that accept only a single source image reject requests providing more than one with an error. Check model documentation for multi-image composition support.

### Generate from Sketch or Structure (Control Models)

```bash
# Control Sketch - generate from sketch
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@sketch.png \
  -F prompt="A realistic portrait" \
  -F model="stability.stable-image-control-sketch-v1:0"

# Control Structure - preserve structure
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@structure.png \
  -F prompt="A modern building" \
  -F model="stability.stable-image-control-structure-v1:0"
```

### Inpainting & Outpainting with Stability AI

```bash
# Stability AI Inpainting - mask marks edit region
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@photo.png \
  -F mask=@edit_mask.png \
  -F prompt="A beautiful sunset" \
  -F model="stability.stable-image-inpaint-v1:0"

# Outpainting - extend image beyond borders
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@photo.png \
  -F prompt="Extend with a forest landscape" \
  -F model="stability.stable-outpaint-v1:0"
```

### Style Guide

```bash
# Extract and apply style from reference
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@content.png \
  -F prompt="Apply impressionist style" \
  -F model="stability.stable-image-style-guide-v1:0"
```

---

**Ready to transform your images?** Explore available image models in the [Models API](api_openai_models.md).
