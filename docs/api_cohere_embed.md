---
title: Embed API - Amazon Bedrock Vector Embeddings (Cohere Compatible)
description: Generate vector embeddings with Amazon Bedrock embedding models using a Cohere-compatible API. Semantic search, RAG, and multimodal embeddings through the Cohere v2 Embed interface.
keywords: Cohere embed API, vector embeddings AWS, semantic search API, RAG embeddings, Cohere compatible API, AWS Bedrock embeddings, input_type embeddings
---

# Embed API (Cohere Compatible)

Generate vector embeddings for semantic search and RAG applications with Amazon Bedrock embedding models through a Cohere-compatible interface.

This is an alternate route to the [OpenAI-compatible Embeddings API](api_openai_embeddings.md): both are served by the same embedding backends and models, so anything supported there is supported here.

!!! warning "Route Prefix & Base URL"
    By default, all Cohere-compatible routes are prefixed with `/cohere`. This means the Embed API is available at `/cohere/v2/embed` instead of `/v2/embed`. You can customize this prefix using the `COHERE_ROUTES_PREFIX` configuration variable documented in [Operations Configuration](operations_configuration.md#cohere-routes-prefix).

    The `curl` examples below use a `$BASE` variable that **must include this prefix** — set it to your scheme and host followed by `COHERE_ROUTES_PREFIX`:

    ```bash
    export BASE="https://your-host/cohere"  # <scheme>://<host> + COHERE_ROUTES_PREFIX
    ```

## Why Choose the Embed API?

<div class="grid cards" markdown>

- :material-magnify: __Semantic Search__
  <br>Turn texts and images into dense vectors for similarity search that understands meaning, not just keywords.

- :material-book-open-page-variant: __Higher RAG Quality__
  <br>Build retrieval pipelines on high-quality embeddings, with `input_type` tuning for queries versus documents.

- :material-swap-horizontal: __Drop-in Cohere Compatibility__
  <br>Follows the Cohere v2 Embed API shape. Existing Cohere embed integrations work by changing the base URL.

- :material-cloud-lock: __Private AWS Backend__
  <br>Served entirely by Bedrock embedding models in your own AWS account — no traffic to third-party endpoints.

</div>

## Available Endpoints

| Endpoint    | Method | What It Does                                           | Powered By               | MCP Tool          |
|-------------|--------|--------------------------------------------------------|--------------------------|-------------------|
| `/v2/embed` | `POST` | Transform texts and images into semantic float vectors | Bedrock embedding models | `cohere_embed`    |
| `/v1/embed` | `POST` | Legacy v1 embed for older SDKs and integrations        | Bedrock embedding models | `cohere_embed_v1` |

**Example request:**

```bash
curl -X POST "$BASE/v2/embed" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cohere.embed-multilingual-v3",
    "input_type": "search_document",
    "texts": ["Hello world", "Bonjour le monde"]
  }'
```

**Example response:**

```json
{
  "response_type": "embeddings_by_type",
  "id": "0f1b3c6e8d9a4b5c8e7f6a5b4c3d2e1f",
  "embeddings": {"float": [[0.012, -0.034, ...], [0.041, 0.007, ...]]},
  "texts": ["Hello world", "Bonjour le monde"],
  "meta": {
    "api_version": {"version": "2"},
    "billed_units": {"input_tokens": 8}
  }
}
```

**Find compatible models:** Call [`/search_models`](api_search_models.md) with `route=cohere_embed` to discover model IDs that support embeddings — every Bedrock embedding model works, not just Cohere ones.

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                       |                  Status                  | Notes                                                              |
|-------------------------------|:----------------------------------------:|--------------------------------------------------------------------|
| **Input**                     |                                          |                                                                    |
| `texts`                       |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Full support                                                       |
| `images`                      |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Multimodal models only; data URIs, plus URLs and S3 URIs           |
| `inputs` (fused text + image) |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Several content parts in one input require Cohere Embed v4; a single-part input works on every model. v2 endpoint only |
| **Model Parameters**          |                                          |                                                                    |
| `input_type`                  |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Applied to Cohere models; no equivalent on other providers         |
| `output_dimension`            |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Some models support dimension reduction                            |
| `truncate`, `max_tokens`      |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Cohere models only                                                 |
| `embedding_types`             |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | `int8`/`uint8`/`binary`/`ubinary` on Cohere models, `binary` also on Titan Embed v2; `base64` always computed client-side; other combinations return 400 |
| `priority`                    | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted but ignored — request scheduling priority is not applicable on Bedrock |
| Extra model-specific params   | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Extra fields are forwarded as additional model request parameters  |
| **Output**                    |                                          |                                                                    |
| `images` metadata array       |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Echoed by models that report image dimensions (e.g. Cohere Embed), and only when every input of the request is an image |
| **Usage tracking**            |                                          |                                                                    |
| `billed_units.input_tokens`   |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Estimated on some models                                           |
| `billed_units.images`         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Counts every submitted image, from `images` and from `inputs`      |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with the Cohere API
* :material-cog:{ .model-dep role="img" aria-label="Model-dependent" } **Available on Select Models** — Check your model's capabilities
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } **Extra Feature** — Enhanced capability beyond the Cohere API

</div>

## Model Support

These are the model families served by this route, with the constraint each one places on a request:

### ![Cohere](styles/logo_cohere.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Cohere Models

| Model                        | Model ID                       | Cohere Name                | Notes                                                              |
|------------------------------|--------------------------------|----------------------------|--------------------------------------------------------------------|
| Cohere Embed v4              | `cohere.embed-v4:0`            | `embed-v4.0`               | Several `images` per request; the only Cohere model accepting `texts` and `images` in the same request, and the only model accepting fused `inputs` entries |
| Cohere Embed Multilingual v3 | `cohere.embed-multilingual-v3` | `embed-multilingual-v3.0`  | `texts` or `images` in a request, not both; one image per request  |
| Cohere Embed English v3      | `cohere.embed-english-v3`      | `embed-english-v3.0`       | `texts` or `images` in a request, not both; one image per request  |

!!! note "Every Cohere Embed Model Is Discoverable As An Image Model"
    Amazon Bedrock publishes the two Embed v3 models as text-only, yet both embed an image and are [billed](#billing) for it. The [Models](models.md) page and [`/search_models`](api_search_models.md) correct that, so all three Cohere Embed models are returned by an `input_modalities=IMAGE` filter. The modality list stays a best-effort hint rather than a gate: what a model accepts is decided by the request, not by the filter.

!!! tip "Cohere's Own Model Names Resolve As They Stand"
    Each Cohere model is published under the name [Cohere's API](https://docs.cohere.com/docs/models) uses as well as its Bedrock ID, derived from the ID rather than curated by hand, so an application already calling Cohere changes only its base URL. Both forms reach the same model; a Cohere model Bedrock does not serve (e.g. `embed-english-light-v3.0`) returns `404` until you map it with [`MODEL_ALIASES`](operations_configuration.md#model-aliases).

### ![Amazon](styles/logo_amazon.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Models

| Model                               | Model ID                                  | Notes                                                          |
|-------------------------------------|-------------------------------------------|------------------------------------------------------------------|
| Amazon Nova 2 Multimodal Embeddings | `amazon.nova-2-multimodal-embeddings-v1:0` | `output_dimension` limited to `256`, `384`, `1024` or `3072`   |
| Amazon Titan Embed Text v2          | `amazon.titan-embed-text-v2:0`            | Text only                                                       |
| Amazon Titan Embed Image v1         | `amazon.titan-embed-image-v1`             | The only Titan model accepting `images`                         |
| Amazon Titan Embed Text v1          | `amazon.titan-embed-text-v1`              | Deprecated — served by `amazon.titan-embed-text-v2:0` instead   |

### ![TwelveLabs](styles/logo_twelvelabs.svg){ style="height: 1.2em; vertical-align: text-bottom;" } TwelveLabs Models

| Model                        | Model ID                          | Notes                                    |
|------------------------------|-----------------------------------|--------------------------------------------|
| TwelveLabs Marengo Embed 3.0 | `twelvelabs.marengo-embed-3-0-v1:0` | `output_dimension` is rejected with a 400; one text plus one image in the same request embeds into a single fused vector |
| TwelveLabs Marengo Embed 2.7 | `twelvelabs.marengo-embed-2-7-v1:0` | `output_dimension` is rejected with a 400 |

## Multimodal Inputs

Send `inputs` to the `/v2/embed` endpoint to embed a caption and a picture **into a single vector** — the representation a product catalogue or a document page needs, where the text and the image describe the same thing. Each entry of `inputs` carries a `content` array of `text` and `image_url` parts and produces exactly one vector.

```bash
curl -X POST "$BASE/v2/embed" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cohere.embed-v4:0",
    "input_type": "search_document",
    "inputs": [
      {
        "content": [
          {"type": "text", "text": "A red bicycle leaning on a wall"},
          {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0..."}}
        ]
      }
    ],
    "embedding_types": ["float"]
  }'
```

```json
{
  "response_type": "embeddings_by_type",
  "id": "0f1b3c6e8d9a4b5c8e7f6a5b4c3d2e1f",
  "embeddings": {"float": [[0.012, -0.034, ...]]},
  "meta": {
    "api_version": {"version": "2"},
    "billed_units": {"input_tokens": 330, "images": 1}
  }
}
```

Good to know:

* **Two or more content parts require Cohere Embed v4.** Any other model returns 400 rather than splitting the entry, which would answer with more vectors than inputs were sent. An entry with a **single** content part is accepted by every embedding model and is embedded exactly as the same `texts` or `images` entry would be. The response `texts` array echoes the `texts` field only, so a vector produced from an `inputs` entry has no echoed text.
* **`texts`, `images` and `inputs` can be combined**, and the vectors come back in that order: texts first, then images, then `inputs` entries. Carrying texts and images in the *same* request needs a model that takes both — Cohere Embed v4, Amazon Nova 2 Multimodal Embeddings, Amazon Titan Embed Image or TwelveLabs Marengo Embed. Cohere Embed v3 takes one or the other and returns 400 for both at once, naming those alternatives.
* **`images` metadata is only returned when the request embeds images alone** — every input an image, sent through `images`, through single-part `inputs` entries, or both. As soon as a text travels in the same request, or an image sits in a multi-part entry, no `images` metadata comes back; `meta.billed_units.images` counts every submitted image either way.
* **Cohere models accept at most 96 `texts` and at most 96 `inputs` entries per request.** The endpoint does not enforce that limit — the model rejects what exceeds it — and other embedding models have limits of their own. How many `images` a request may carry is a separate, per-model limit, given in the [model table](#cohere-models) above.
* Image parts take the same sources as `images`: a data URI, an `https://` URL or an `s3://` URI.

## Quantized and Base64 Embedding Types

Set `embedding_types` to request quantized vectors alongside, or instead of, the default `float` vectors. Bedrock Cohere Embed models natively support `int8`, `uint8`, `binary`, and `ubinary`; Titan Embed v2 natively supports `binary`. `base64` is always available: it is computed client-side (little-endian float32 bytes, base64-encoded) from the `float` embedding, matching the Cohere API encoding. Requesting a type not supported by the resolved model returns 400. Only the requested types are populated in the response.

```bash
curl -X POST "$BASE/v2/embed" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cohere.embed-v4:0",
    "input_type": "search_document",
    "texts": ["Hello world"],
    "embedding_types": ["int8", "base64"]
  }'
```

```json
{
  "response_type": "embeddings_by_type",
  "id": "0f1b3c6e8d9a4b5c8e7f6a5b4c3d2e1f",
  "embeddings": {
    "int8": [[12, -34, ...]],
    "base64": ["rBgKPw..."]
  },
  "texts": ["Hello world"],
  "meta": {
    "api_version": {"version": "2"},
    "billed_units": {"input_tokens": 4}
  }
}
```

## Cohere v1 Embed API (Legacy)

The legacy `/v1/embed` endpoint is also available for older Cohere SDKs (`cohere.Client`) and third-party integrations that predate the v2 API. It shares the same Bedrock backend and model support as `/v2/embed`; new clients should prefer the v2 endpoint.

**Differences from the v2 endpoint:**

<div class="feature-table" markdown>

| Feature                    |                 Status                  | Notes                                                                          |
|----------------------------|:---------------------------------------:|---------------------------------------------------------------------------------|
| Default response shape     |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Legacy `embeddings_floats`: a plain list of float vectors                      |
| `embedding_types`          |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Any value switches to the `embeddings_by_type` shape; same type support as the v2 endpoint |
| `input_type`               |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Optional — forwarded to Cohere models when provided; the backend defaults to `search_document` otherwise. On Embed v3, an all-image request always overrides it to `image` |
| `meta.api_version.version` |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Reported as `"1"`                                                              |

</div>

**Example request:**

```bash
curl -X POST "$BASE/v1/embed" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cohere.embed-multilingual-v3",
    "input_type": "search_document",
    "texts": ["Hello world", "Bonjour le monde"]
  }'
```

**Example response:**

```json
{
  "response_type": "embeddings_floats",
  "id": "0f1b3c6e8d9a4b5c8e7f6a5b4c3d2e1f",
  "embeddings": [[0.012, -0.034, ...], [0.041, 0.007, ...]],
  "texts": ["Hello world", "Bonjour le monde"],
  "meta": {
    "api_version": {"version": "1"},
    "billed_units": {"input_tokens": 8}
  }
}
```

## How It Works

Requests are served by the same Bedrock embedding backends as the [OpenAI-compatible Embeddings API](api_openai_embeddings.md), with automatic multi-region routing and failover across the regions where the selected model is available.

- When several input fields are provided, embeddings are returned in request order: all texts first, then all images, then the `inputs` entries.
- Guardrail and performance headers available on the [OpenAI-compatible Embeddings API](api_openai_embeddings.md#available-request-headers) work on this route too.

## Billing

Requests are billed through Bedrock, not in Cohere search units. `billed_units.input_tokens` reports the Bedrock-metered input tokens, which on Cohere Embed v4 already include the tokens of every embedded image. Cohere Embed v3 models meter an embedded image as a separate billed unit instead. Usage appears in [usage logs and cost tracking](operations_logging_monitoring.md) as `input_tokens`, plus the per-media-unit dimensions of the models that publish one — see [Media Input Pricing](operations_cost_management.md#media-input-pricing).
