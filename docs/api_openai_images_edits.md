---
title: Images Edits API - AWS Bedrock Image Editing
description: Edit and transform images with AWS Bedrock image models. OpenAI-compatible API for image modifications, inpainting, and transformations.
keywords: image editing API, AI image editor, inpainting API, image modification API, AWS image editing, OpenAI image edit, image transformation API
---

# Images API - Image Editing

Edit images using inpainting with AWS Bedrock image models through an OpenAI-compatible interface.

## Why Choose Image Editing?

<div class="grid cards" markdown>

- :material-image-edit: __Precise Control__
  <br>Edit specific regions of images while preserving the rest.

- :material-palette-advanced: __Creative Freedom__
  <br>Add, remove, or modify elements in existing images with AI assistance.

- :material-layers-triple: __Flexible Masking__
  <br>Use transparency or explicit masks to define edit regions.

- :material-aws: __Scalable Infrastructure__
  <br>Edit images at scale with AWS Bedrock infrastructure.

</div>

## Quick Start: Available Endpoint

| Endpoint           | Method | What It Does                            | Powered By               | MCP Tool           |
|--------------------|--------|-----------------------------------------|--------------------------|--------------------|
| `/v1/images/edits` | POST   | Edit images using prompts and masks     | AWS Bedrock Image Models | `openai_image_edit` |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                        |                  Status                  | Notes                                                        |
|--------------------------------|:----------------------------------------:|--------------------------------------------------------------|
| **Editing**                    |                                          |                                                              |
| Image-to-image (`/edits`)      |   :material-check-circle:{ .success }    | Edit images with prompts and masks                           |
| **Request Formats**            |                                          |                                                              |
| Multipart form-data            |   :material-check-circle:{ .success }    | Binary file uploads via `image` / `image[]` / `mask` fields  |
| JSON body                      |   :material-check-circle:{ .success }    | Structured `images` array with Files API IDs or URLs         |
| **Parameters**                 |                                          |                                                              |
| `image` / `image[]`            |   :material-check-circle:{ .success }    | PNG image(s) to edit (multipart: binary upload, 1+ images)   |
| `images` (JSON)                |   :material-check-circle:{ .success }    | Array of `{file_id}` or `{image_url}` references (JSON body) |
| `prompt`                       |   :material-check-circle:{ .success }    | Text description of desired changes                          |
| `mask`                         |   :material-check-circle:{ .success }    | Optional mask defining edit regions                          |
| `n` (number of images)         |   :material-check-circle:{ .success }    | Generate multiple variations per request (1-10)              |
| `size` (WIDTHxHEIGHT)          |   :material-check-circle:{ .success }    | Output dimensions (default: 1024x1024, format validated)     |
| `model`                        |   :material-check-circle:{ .success }    | Required parameter                                           |
| `response_format`              |   :material-check-circle:{ .success }    | `url` or `b64_json` (default: `url`)                         |
| `output_format`                |   :material-check-circle:{ .success }    | `png`, `jpeg`, or `webp` (model-specific)                    |
| `output_compression`           |   :material-check-circle:{ .success }    | Compression level 1-100% (default: 100)                      |
| `quality`                      |   :material-check-circle:{ .success }    | Quality setting (default: `auto`, model-specific values)     |
| `stream`                       |   :material-check-circle:{ .success }    | Generate images in streaming mode with partial results       |
| `partial_images`               |       :material-cog:{ .model-dep }       | Number of partial images in stream (0-3, model-specific)     |
| `background`                   | :material-close-circle:{ .unsupported }  | Always `auto`, transparent backgrounds unsupported           |
| `input_fidelity`               | :material-close-circle:{ .unsupported }  | Ignored, always `low`                                        |
| **Output**                     |                                          |                                                              |
| URL response format            |   :material-check-circle:{ .success }    | Temporary URLs to edited images (requires AWS_S3_BUCKET)     |
| Base64 JSON format             |   :material-check-circle:{ .success }    | Inline base64-encoded images                                 |
| PNG format                     |   :material-check-circle:{ .success }    | Lossless image output                                        |
| JPEG format                    |       :material-cog:{ .model-dep }       | Lossy compression (model-specific)                           |
| WEBP format                    |       :material-cog:{ .model-dep }       | Modern format with compression (model-specific)              |
| Streaming response             |   :material-check-circle:{ .success }    | Server-sent events with partial and final images             |
| **Usage tracking**             |                                          |                                                              |
| Input text tokens              |   :material-check-circle:{ .success }    | Estimated from prompt                                        |
| Input image tokens             |   :material-check-circle:{ .success }    | Count of input images (image + mask)                         |
| Output image tokens            |   :material-check-circle:{ .success }    | Image count (billing unit)                                   |
| **Other**                      |                                          |                                                              |
| `user`                         |   :material-minus-circle:{ .partial }    | Logged but not used for abuse monitoring                     |
| Extra parameters via form data | :material-plus-circle:{ .extra-feature } | Provider-specific parameters passed through                  |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep } **Available on Select Models** — Check your model's capabilities
* :material-minus-circle:{ .partial } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported } **Unsupported** — Not available in this implementation

</div>

!!! info "Model Support"
    **Inpainting** (mask-based editing) is supported by **Amazon Nova Canvas**, **Amazon Titan Image Generator**, and **Stability AI** inpaint models.

    **Image-to-image** (transformation without masks) is supported by **Stability AI** text-to-image models (SD3.5, Stable Image Core, Stable Image Ultra).

    **Upscale** (resolution enhancement) is supported by **Stability AI** upscale models (creative, conservative, fast).

    **Style Transfer** (applying reference image style) is supported by **Stability AI** style transfer models.

    **Search-based editing** (find & replace/recolor objects) is supported by **Stability AI** search models.

    **Background removal** is supported by **Amazon Titan Image Generator v2**, **Amazon Nova Canvas**, and **Stability AI** remove background model.

!!! warning "Configuration Required"
    You must configure the `AWS_S3_BUCKET` environment variable with a bucket to use the URL response format.

## Supported Models

### ![Amazon](styles/logo_amazon.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Models

| Model                             | Supported Task Types                                                              | Mask Support                                                                    | Notes                                                                               |
|-----------------------------------|-----------------------------------------------------------------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| amazon.nova-canvas-v1:0           | `TEXT_IMAGE`, `INPAINTING`, `OUTPAINTING`, `BACKGROUND_REMOVAL`, `VIRTUAL_TRY_ON` | ✅ Required for inpainting/outpainting<br>✅ Used as reference for virtual try-on | Supports multiple editing modes including advanced virtual try-on with 3 mask types |
| amazon.titan-image-generator-v1   | `INPAINTING`, `OUTPAINTING`                                                       | ✅ Required for inpainting/outpainting                                           | Supports text-based mask prompts as alternative to mask images                      |
| amazon.titan-image-generator-v2:0 | `INPAINTING`, `OUTPAINTING`, `BACKGROUND_REMOVAL`                                 | ✅ Required for inpainting/outpainting<br>❌ Not used for background removal      | Enhanced features including background removal without mask                         |

!!! info "Amazon Nova Canvas Default Behavior"
    **`amazon.nova-canvas-v1:0`** automatically selects the task type based on the presence of a mask when no `taskType` is explicitly provided:

    - **No mask provided** → Uses `TEXT_IMAGE` by default
    - **Mask provided** → Uses `INPAINTING` by default

### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Stability AI Models

#### Image-to-Image Models

| Model                             | Prompt Usage           | Mask Usage | Extra Parameters Required | Notes                             |
|-----------------------------------|------------------------|------------|---------------------------|-----------------------------------|
| stability.sd3-5-large-v1:0        | Guides transformation  | Not used   | None                      | Transform images with prompt      |

#### Upscale Models

| Model                                      | Prompt Usage           | Mask Usage | Extra Parameters Required | Notes                                   |
|--------------------------------------------|------------------------|------------|---------------------------|-----------------------------------------|
| stability.stable-creative-upscale-v1:0     | Guides upscaling       | Not used   | None                      | Prompt-guided upscaling with creativity |
| stability.stable-conservative-upscale-v1:0 | Guides upscaling       | Not used   | None                      | Detail-preserving upscaling             |
| stability.stable-fast-upscale-v1:0         | **Not used**           | Not used   | None                      | Fast 4x upscaling without prompt        |

#### Edit Models

| Model                                         | Prompt Usage              | Mask Usage                   | Extra Parameters Required | Notes                            |
|-----------------------------------------------|---------------------------|------------------------------|---------------------------|----------------------------------|
| stability.stable-image-inpaint-v1:0           | Guides inpainting         | Optional (marks edit region) | None                      | Fill masked regions              |
| stability.stable-outpaint-v1:0                | Guides outpainting        | Not used                     | None                      | Extend image beyond borders      |
| stability.stable-image-search-recolor-v1:0    | Describes new color       | Not used                     | `select_prompt`           | Recolor objects by search prompt |
| stability.stable-image-search-replace-v1:0    | Describes replacement     | Not used                     | `search_prompt`           | Replace objects by search prompt |
| stability.stable-image-erase-object-v1:0      | **Not used**              | Required (marks object)      | None                      | Remove objects with mask         |
| stability.stable-image-remove-background-v1:0 | **Not used**              | Not used                     | None                      | Automatic background removal     |

#### Control Models

| Model                                         | Prompt Usage         | Mask Usage | Extra Parameters Required | Notes                           |
|-----------------------------------------------|----------------------|------------|---------------------------|---------------------------------|
| stability.stable-image-control-sketch-v1:0    | Guides generation    | Not used   | None                      | Generate from sketch            |
| stability.stable-image-control-structure-v1:0 | Guides generation    | Not used   | None                      | Structure-preserving generation |

#### Style Models

| Model                                   | Prompt Usage          | Mask Usage                             | Extra Parameters Required | Notes                         |
|-----------------------------------------|-----------------------|----------------------------------------|---------------------------|-------------------------------|
| stability.stable-image-style-guide-v1:0 | Guides style          | Not used                               | None                      | Extract and apply style       |
| stability.stable-style-transfer-v1:0    | Guides style transfer | Required (repurposed as `style_image`) | None                      | Transfer style between images |

!!! note "Output Formats"
    All models support standard OpenAI output formats (`png`, `jpeg`, `webp`) via the `output_format` parameter. Format availability may vary by model - unsupported formats will automatically fall back to PNG.

!!! warning "Extra Parameters Required"
    Some models require parameters **beyond the standard OpenAI API**:

    - **`stability.stable-image-search-recolor-v1:0`**: Requires `select_prompt` form field
    - **`stability.stable-image-search-replace-v1:0`**: Requires `search_prompt` form field

    **Models that don't use prompt**: `stability.stable-fast-upscale-v1:0`, `stability.stable-image-erase-object-v1:0`, `stability.stable-image-remove-background-v1:0` - provide empty string or omit the `prompt` parameter.

    All other Stability models use only standard OpenAI parameters (`image`, `prompt`, and optionally `mask`).

## Request Formats

The `/v1/images/edits` endpoint accepts two request formats:

### Multipart Form-Data (Binary Uploads)

The classic format — upload image files directly. Use `image` (single) or `image[]` (multiple) for source images and `mask` for the optional edit mask.

```bash
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F image=@source.png \
  -F mask=@mask.png \
  -F prompt="A red apple on a wooden table" \
  -F model="amazon.nova-canvas-v1:0"
```

### JSON Body (Files API or URL References)

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

## How Image Editing Works

### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Image-to-Image (Stability AI Models)

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

### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Upscale (Stability AI)

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

### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Style Transfer (Stability AI)

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

### Inpainting with Masks (Amazon Models)

#### With Transparent Areas (Automatic Mask)

If your source image has transparent regions (alpha channel), those areas will automatically be used as the edit mask:

```bash
# Edit using a PNG image with transparent areas
# The transparent regions will be automatically detected as the mask
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@image_with_transparency.png \
  -F prompt="A blue circle in the center" \
  -F model="amazon.nova-canvas-v1:0"
```

**Image format**: PNG with alpha channel where transparent pixels indicate regions to edit.

#### With Explicit Mask

For more control, provide an explicit mask image where transparent areas indicate regions to edit:

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

**Mask format**: PNG with alpha channel where transparent pixels indicate regions to edit, opaque pixels are preserved.

## Try It Now

### ![Stability AI](styles/logo_stabilityai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Image-to-Image with Stability AI

```bash
# Transform image with default strength (0.5)
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
  -F model="amazon.nova-canvas-v1:0"
```

### Multiple Variations

```bash
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
    Not all models support multiple input images. Currently, models that do not support multiple images will only use the first image and raise an error if more than one is provided. Check model documentation for multi-image composition support.

## Additional Examples

### Inpainting with Control Models

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

### Inpainting with Mask

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

## Advanced Features

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

| OpenAI Parameter    | Maps to                                | Notes                                                  |
|---------------------|----------------------------------------|--------------------------------------------------------|
| `prompt`            | Depends on `taskType`                  | See taskType-specific mapping below                    |
| `image` / `image[]` | Depends on `taskType`                  | See taskType-specific mapping below (first image used) |
| `mask`              | Depends on `taskType`                  | See taskType-specific mapping below                    |
| `size`              | `imageGenerationConfig.width/height`   | Output dimensions (320-4096)                           |
| `quality`           | `imageGenerationConfig.quality`        | "high" → "premium"                                     |
| `n`                 | `imageGenerationConfig.numberOfImages` | 1-5 images                                             |

**TaskType-Specific Parameter Mapping:**

| taskType                   | `prompt` maps to                                              | `image` maps to                  | `mask` maps to                      |
|----------------------------|---------------------------------------------------------------|----------------------------------|-------------------------------------|
| `INPAINTING` (default)     | `inPaintingParams.text`                                       | `inPaintingParams.image`         | `inPaintingParams.maskImage`        |
| `OUTPAINTING`              | `outPaintingParams.text`                                      | `outPaintingParams.image`        | `outPaintingParams.maskImage`       |
| `BACKGROUND_REMOVAL`       | Not used                                                      | `backgroundRemovalParams.image`  | Not used                            |
| `VIRTUAL_TRY_ON` (PROMPT)  | `promptBasedMask.maskPrompt`                                  | `virtualTryOnParams.sourceImage` | `virtualTryOnParams.referenceImage` |
| `VIRTUAL_TRY_ON` (GARMENT) | `garmentBasedMask.garmentClass`                               | `virtualTryOnParams.sourceImage` | `virtualTryOnParams.referenceImage` |
| `VIRTUAL_TRY_ON` (IMAGE)   | `imageBasedMask.maskImage` (Base64 encoded image or data URI) | `virtualTryOnParams.sourceImage` | `virtualTryOnParams.referenceImage` |

**Advanced Task Types (with form fields):**

Default `taskType` is `"INPAINTING"`.

Available task types:

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
  -F model="amazon.titan-image-generator-v2:0"
```

**Parameter Mapping:**

| OpenAI Parameter    | Maps to                                | Notes                                                  |
|---------------------|----------------------------------------|--------------------------------------------------------|
| `prompt`            | Depends on `taskType`                  | See taskType-specific mapping below                    |
| `image` / `image[]` | Depends on `taskType`                  | See taskType-specific mapping below (first image used) |
| `mask`              | Depends on `taskType`                  | See taskType-specific mapping below                    |
| `size`              | `imageGenerationConfig.width/height`   | Fixed sizes (512-2048)                                 |
| `quality`           | `imageGenerationConfig.quality`        | "high" → "premium"                                     |
| `n`                 | `imageGenerationConfig.numberOfImages` | 1-5 images                                             |

**TaskType-Specific Parameter Mapping:**

| taskType               | `prompt` maps to         | `image` maps to                 | `mask` maps to                |
|------------------------|--------------------------|---------------------------------|-------------------------------|
| `INPAINTING` (default) | `inPaintingParams.text`  | `inPaintingParams.image`        | `inPaintingParams.maskImage`  |
| `OUTPAINTING`          | `outPaintingParams.text` | `outPaintingParams.image`       | `outPaintingParams.maskImage` |
| `BACKGROUND_REMOVAL`   | Not used                 | `backgroundRemovalParams.image` | Not used                      |

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
  -F model="amazon.titan-image-generator-v2:0" \
  -F taskType="OUTPAINTING"

# Background Removal (v2 only, no prompt needed)
curl -X POST "$BASE/v1/images/edits" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F image=@photo.png \
  -F model="amazon.titan-image-generator-v2:0" \
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

| OpenAI Parameter    | Stability Parameter | Notes                                                     |
|---------------------|---------------------|-----------------------------------------------------------|
| `image` / `image[]` | `image`             | Base64-encoded input image (first image used if multiple) |
| `prompt`            | `prompt`            | Text description (may be unused for some models)          |
| `mask`              | `mask`              | Base64-encoded mask (model-specific)                      |
| `n`                 | Multiple requests   | Generates N images via multiple API calls                 |
| `size`              | Model-specific      | Some models support width/height                          |

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

### Usage Tracking

The API tracks token usage for billing and monitoring:

- **Text tokens**: Estimated from your prompt using tiktoken
- **Image tokens**: Count of input images (image files + mask file if provided)
- **Output tokens**: Number of images generated (`n` parameter)
