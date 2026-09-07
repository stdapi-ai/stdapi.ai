---
title: Videos API - Amazon Bedrock Video Generation (OpenAI Compatible)
description: Generate videos from text prompts and reference images with Amazon Bedrock video models using an OpenAI-compatible Videos API. Amazon Nova Reel and Luma Ray 2 through the Sora API shape.
keywords: video generation API, text to video AWS, OpenAI videos API, Sora compatible, Amazon Nova Reel, Luma Ray, image to video, AWS Bedrock video
---

# Videos API

Generate videos from text prompts and reference images with Amazon Bedrock video models through the OpenAI Videos API shape.

Video generation is **asynchronous**: creating a video returns a job object immediately. Poll the job until its status is `completed`, then download the MP4 content — exactly like the OpenAI (Sora) workflow, so the official OpenAI SDKs work unchanged.

## At a glance

- :material-movie-open: **Amazon Nova Reel and Luma Ray 2** — clips from 5 to 120
  seconds depending on the model, from a prompt and an optional first-frame
  image, see [Models](#model-support).
- :material-api: **Five endpoints** — create, list, retrieve, download and
  delete a job, in the OpenAI Videos API shape, see [Endpoints](#quick-start-available-endpoints).
- :material-server-off: **Video IDs stay valid across restarts** — a job
  identifier carries the invocation it names, so retrieve, download and delete
  work from any instance behind a load balancer, see
  [How video generation works](#how-it-works).
- :material-cloud-lock: **Rendered into your own S3 buckets** — Amazon Bedrock
  writes the MP4 under the `videos/` prefix of the regional bucket you
  configure, with no traffic to third-party endpoints, see
  [How video generation works](#how-it-works).
- :material-close-circle: **No remix, edits, extensions or characters** — Amazon
  Bedrock generates the video asset alone, so `variant=thumbnail` and
  `variant=spritesheet` are refused too, see
  [Feature compatibility](#feature-compatibility).
- :material-alert-circle-outline: **`model` is required, and a running job
  cannot be deleted** — this gateway has no default video model, and `DELETE`
  answers `400` while a job is `in_progress`, see
  [Limits and behaviour to know](#limits-and-behaviour-to-know).

```bash
curl -X POST "$BASE/v1/videos" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "luma.ray-v2:0",
    "prompt": "Closeup of a seashell on a sandy beach, gentle waves"
  }'
```

## Endpoints { #quick-start-available-endpoints }

| Endpoint                       | Method   | What It Does                                     | Powered By                  | MCP Tool                 |
|--------------------------------|----------|--------------------------------------------------|-----------------------------|--------------------------|
| `/v1/videos`                   | `POST`   | Start an asynchronous video generation job       | Amazon Bedrock Video Models | `openai_video_generation` |
| `/v1/videos`                   | `GET`    | List video generation jobs across regions        | Amazon Bedrock              | `openai_video_list`      |
| `/v1/videos/{video_id}`        | `GET`    | Retrieve the current state of a job              | Amazon Bedrock              | `openai_video_get`       |
| `/v1/videos/{video_id}/content`| `GET`    | Download the generated MP4 once completed        | Amazon S3                   | `openai_video_content`   |
| `/v1/videos/{video_id}`        | `DELETE` | Delete the stored video output                   | Amazon S3                   | `openai_video_delete`    |

## Feature compatibility

<div class="feature-table" markdown>

| Feature                     |                  Status                  | Notes                                                                  |
|-----------------------------|:----------------------------------------:|------------------------------------------------------------------------|
| **Creation**                |                                          |                                                                        |
| `prompt`                    |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Full support                                                           |
| `model`                     |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Required — this gateway has no implicit default video model            |
| `seconds` / `size`          |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Supported values depend on the model (see table below)                 |
| `input_reference`           |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | First-frame image; also accepts URLs/S3/Files API IDs in JSON requests |
| Extra model-specific params | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Extra fields are forwarded to the model (e.g. `seed`, `loop`); a string value that parses as a JSON number, boolean, or null is forwarded as that type |
| **Lifecycle**               |                                          |                                                                        |
| Retrieve / poll job         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | `progress` is `0` while running and `100` when completed               |
| Download content (`video`)  |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Streamed MP4                                                           |
| Delete video                |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Removes the stored output from S3; a job still `in_progress` answers `400`, see [Deleting a job that is still running](#deleting-a-job-that-is-still-running) |
| `variant=thumbnail/spritesheet` | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Amazon Bedrock generates only the video asset                      |
| List videos (`GET /v1/videos`) | :material-check-circle:{ .success role="img" aria-label="Supported" }   | Merged across regions; listed while AWS retains the job record        |
| Remix video                 | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Not available on Amazon Bedrock                                        |
| Edits / extensions / characters | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | `/v1/videos/edits`, `/videos/extensions`, and `/videos/characters` are not available on Amazon Bedrock |
| `expires_at`                |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Reported only when a [retention period](operations_configuration_storage.md#aws-s3-videos-expires-after) is configured |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep role="img" aria-label="Model-dependent" } **Available on Select Models** — Check your model's capabilities
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } **Extra Feature** — Enhanced capability beyond OpenAI API

</div>

!!! note "Listing and Retrieval Behaviour"
    The `prompt` is echoed only in the creation response, not when retrieving or listing. A deleted video stays visible in listings until AWS expires its job record. Listing merges jobs across every configured region; a region that is temporarily unavailable is omitted rather than failing the request.

## Models { #model-support }

Any video generation model available in your configured Amazon Bedrock regions can be used, for example:

### ![Amazon Nova](styles/logo_amazon_nova.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Models

| Model                | Model ID                | Durations (`seconds`)              | Sizes (`size`) |
|----------------------|-------------------------|------------------------------------|----------------|
| Amazon Nova Reel 1.1 | `amazon.nova-reel-v1:1` | `6`, or multiples of 6 up to `120` | `1280x720`     |
| Amazon Nova Reel 1.0 | `amazon.nova-reel-v1:0` | `6`                                | `1280x720`     |

!!! note "Amazon Nova Reel Is Marked Legacy by AWS"
    AWS flags the Nova Reel models as legacy, so they are hidden by default. Set [`AWS_BEDROCK_LEGACY=true`](operations_configuration_models.md#bedrock-legacy) to expose and use them.

### ![Luma AI](styles/logo_luma.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Luma AI Models

| Model      | Model ID        | Durations (`seconds`) | Sizes (`size`)                                                                                     |
|------------|-----------------|-----------------------|-----------------------------------------------------------------------------------------------------|
| Luma Ray 2 | `luma.ray-v2:0` | `5` or `9`            | 540p/720p in ratios 1:1, 16:9, 9:16, 4:3, 3:4, 21:9, 9:21 (e.g. `1280x720`, `720x1280`, `960x540`) |

When `seconds` or `size` is omitted, the model's shortest duration and default resolution are used. For Luma Ray, the requested size selects the model's resolution (smaller dimension: 540 or 720) and aspect ratio (reduced width:height).

**Find compatible models:** Call [`/search_models`](api_search_models.md) with `route=openai_video_generation` to discover model IDs that support video generation in your deployment.

## Reference images (image-to-video)

The optional `input_reference` image is used as the video's first frame:

- **`multipart/form-data`** — upload the image file directly (this is what the OpenAI SDKs send). The SDK object form (`input_reference={"image_url": ...}` or `{"file_id": ...}`) is also accepted.
- **`application/json`** — pass a base64 string, data URI, HTTPS URL, S3 URI, or [Files API](api_openai_files.md) ID.

Amazon Nova Reel requires a PNG or JPEG matching the video resolution (1280x720) and only supports reference images for 6-second videos.

## How video generation works { #how-it-works }

1. `POST /v1/videos` starts an [Amazon Bedrock asynchronous invocation](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_StartAsyncInvoke.html) in a region where the model is available and a regional S3 bucket is configured ([`AWS_S3_REGIONAL_BUCKETS`](operations_configuration_storage.md#aws-s3-regional-buckets)).
2. Amazon Bedrock renders the video and writes the MP4 to that bucket under [`AWS_S3_VIDEOS_PREFIX`](operations_configuration_storage.md#aws-s3-videos-prefix) (`videos/` by default).
3. `GET /v1/videos/{video_id}` reads the job state directly from Amazon Bedrock; `.../content` streams the MP4 from S3; `DELETE` removes the stored objects.

!!! warning "S3 Bucket Requirement"
    Amazon Bedrock requires the output bucket to be **in the same region as the invocation**. Configure a bucket for each region offering video models via `AWS_S3_REGIONAL_BUCKETS`, otherwise video generation requests fail with a configuration error.

## Billing

AWS bills video generation per **second of generated video**. Billed seconds appear in [usage logs and cost tracking](operations_logging_monitoring.md) as `output_seconds`, recorded when the job is started. Standard S3 storage costs apply to the stored videos until they are deleted.

## Limits and behaviour to know

### Deleting a job that is still running

`DELETE /v1/videos/{video_id}` answers `400 Video is still being processed`
while the job reports `in_progress`: an Amazon Bedrock asynchronous invocation
cannot be cancelled. Poll the job until it reports `completed` or `failed`,
then delete it.

Downloading the content of an unfinished job returns `404` instead, matching
upstream.

### Retention

By default videos persist until deleted through the API. Set [`AWS_S3_VIDEOS_EXPIRES_AFTER`](operations_configuration_storage.md#aws-s3-videos-expires-after) to enforce a retention period: the `Video` object then reports `expires_at` (completion time plus the retention period) and downloading expired content returns a 404, matching the OpenAI API's automatic video expiry.

## Request headers

The Amazon Bedrock guardrail headers apply to this endpoint. All headers are optional.

### Content Safety (Guardrails)

| Header                               | Purpose                            | Valid Values               |
|--------------------------------------|------------------------------------|----------------------------|
| `X-Amzn-Bedrock-GuardrailIdentifier` | Guardrail ID for content filtering | Your guardrail identifier  |
| `X-Amzn-Bedrock-GuardrailVersion`    | Guardrail version                  | Version number (e.g., `1`) |

The guardrail evaluates the `prompt` before the generation job is started; the generated video is not itself evaluated. Both headers are honoured only when [`AWS_BEDROCK_ALLOW_GUARDRAIL_OVERRIDE`](operations_configuration_bedrock.md#bedrock-guardrails) is enabled — otherwise the deployment's configured guardrail applies. `X-Amzn-Bedrock-Trace` is accepted but has no effect on this route — no guardrail trace is returned.

**Example with headers:**

```bash
curl -X POST "$BASE/v1/videos" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -H "X-Amzn-Bedrock-GuardrailIdentifier: your-guardrail-id" \
  -H "X-Amzn-Bedrock-GuardrailVersion: 1" \
  -d '{
    "model": "luma.ray-v2:0",
    "prompt": "Closeup of a seashell on a sandy beach, gentle waves"
  }'
```

!!! note "No performance headers on this route"
    `X-Amzn-Bedrock-Service-Tier` and `X-Amzn-Bedrock-PerformanceConfig-Latency` have no effect here: a video is generated by an asynchronous Bedrock invocation, which carries neither a service tier nor a performance configuration.

!!! info "Detailed Documentation"
    For complete information about these headers, configuration options, and use cases, see:

    - [Bedrock Guardrails Configuration](operations_configuration_bedrock.md#bedrock-guardrails)

## Try it

**End to end with the OpenAI Python SDK:**

```python
import time

from openai import OpenAI

client = OpenAI(base_url="https://your-host/v1", api_key="your-api-key")

video = client.videos.create(
    model="luma.ray-v2:0", prompt="Closeup of a seashell on a sandy beach, gentle waves"
)
while video.status in ("queued", "in_progress"):
    time.sleep(10)
    video = client.videos.retrieve(video.id)

with open("video.mp4", "wb") as file:
    file.write(client.videos.download_content(video.id).read())
client.videos.delete(video.id)
```

**Create a job (curl):**

```bash
curl -X POST "$BASE/v1/videos" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "luma.ray-v2:0",
    "prompt": "Closeup of a seashell on a sandy beach, gentle waves",
    "seconds": "5",
    "size": "1280x720"
  }'
```

**Example response:**

```json
{
  "id": "video_eyJhcm4iOiJhcm46YXdzOmJlZHJvY2s6...",
  "object": "video",
  "model": "luma.ray-v2:0",
  "status": "queued",
  "progress": 0,
  "created_at": 1783805314,
  "seconds": "5",
  "size": "1280x720",
  "prompt": "Closeup of a seashell on a sandy beach, gentle waves"
}
```

**Poll the job, then download the MP4:**

```bash
VIDEO_ID="video_eyJhcm4iOiJhcm46YXdzOmJlZHJvY2s6..."

curl "$BASE/v1/videos/$VIDEO_ID" \
  -H "Authorization: Bearer $OPENAI_API_KEY"

curl "$BASE/v1/videos/$VIDEO_ID/content" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -o video.mp4
```

## Next steps

Next: [Models API](api_openai_models.md) · [Files API](api_openai_files.md) · [Images API](api_openai_images_generations.md) · [Storage configuration](operations_configuration_storage.md#aws-s3-regional-buckets)
