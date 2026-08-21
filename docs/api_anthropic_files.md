---
title: Files API - Anthropic-Compatible File Storage
description: Upload, manage, and reference files in Anthropic Messages requests using the Anthropic-compatible Files API backed by Amazon S3. Supports documents, images, and bidirectional cursor pagination.
keywords: Files API, Anthropic files, file upload, S3 file storage, Anthropic messages file, document source, cursor pagination, AWS Bedrock files
---

# Files API (Anthropic Compatible)

!!! warning "Route Prefix & Base URL"
    By default, all Anthropic-compatible routes are prefixed with `/anthropic`. This means the Files API is available at `/anthropic/v1/files` instead of `/v1/files`. You can customize this prefix using the `ANTHROPIC_ROUTES_PREFIX` configuration variable documented in [Operations Configuration](operations_configuration.md#anthropic-routes-prefix).

    The `curl` examples below use a `$BASE` variable that **must include this prefix** — set it to your scheme and host followed by `ANTHROPIC_ROUTES_PREFIX`:

    ```bash
    export BASE="https://your-host/anthropic"  # <scheme>://<host> + ANTHROPIC_ROUTES_PREFIX
    ```

    `ANTHROPIC_ROUTES_PREFIX` must always be a non-empty path and is validated at startup to differ from `OPENAI_ROUTES_PREFIX` (the server refuses to start otherwise). This Anthropic-compatible Files API is therefore always served on its own path, distinct from the [OpenAI-compatible Files API](api_openai_files.md).

Upload and manage files via an Anthropic-compatible interface. Files are stored in Amazon S3 and can be referenced directly in Messages requests as document or image sources.

## Why Choose the Files API?

<div class="grid cards" markdown>

- :material-upload: __Simple Upload__
  <br>Upload any file with a single `multipart/form-data` request. Files are immediately available for use in inference.

- :material-swap-vertical: __Bidirectional Pagination__
  <br>Traverse your file list in both directions using `after_id` and `before_id` cursors, matching the official Files API pagination.

- :material-file-document-multiple: __Messages Integration__
  <br>Reference uploaded files directly in Messages requests as document or image source blocks using `"type": "file"`.

- :material-download: __Content Download__
  <br>Download raw file bytes at any time via the `/content` endpoint.

</div>

## Available Endpoints

| Endpoint                      | Method   | What It Does               | Powered By | MCP Tool                 |
|-------------------------------|----------|----------------------------|------------|--------------------------|
| `/v1/files`                   | `POST`   | Upload a file              | Amazon S3  | `anthropic_file`         |
| `/v1/files`                   | `GET`    | List files with pagination | Amazon S3  | `anthropic_file_list`    |
| `/v1/files/{file_id}`         | `GET`    | Retrieve file metadata     | Amazon S3  | `anthropic_files_get`    |
| `/v1/files/{file_id}`         | `DELETE` | Delete a file              | Amazon S3  | `anthropic_files_delete` |
| `/v1/files/{file_id}/content` | `GET`    | Download raw file bytes    | Amazon S3  | `anthropic_file_content` |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                  |                  Status                  | Notes                                                            |
|--------------------------|:----------------------------------------:|------------------------------------------------------------------|
| **Upload**               |                                          |                                                                  |
| `file` (multipart)       |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Required binary form field                                       |
| `file` (JSON body)       | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Base64, data URI, HTTPS URL, or S3 URI — for MCP / AI agents    |
| **Listing**              |                                          |                                                                  |
| Listing order            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Most recently created first, by `created_at`                     |
| `after_id` cursor        |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Forward cursor: returns files older than the given ID            |
| `before_id` cursor       |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Backward cursor: returns files newer than the given ID           |
| `limit`                  |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | 1 – 1 000; default 20                                            |
| **File size cap**        | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | No artificial limit; S3 object limit (~5 TB)                     |
| **Messages integration** |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | `"source": {"type": "file", "file_id": "..."}` in document/image |
| `downloadable` field     |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Always `true`; spec default is `false` for user-uploaded files   |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with Anthropic API
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Implemented with minor deviations from spec
* :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } **Extra Feature** — Enhanced capability beyond Anthropic API

</div>

## Quick Start

!!! info "Optional Anthropic headers"
    The `anthropic-version` and `anthropic-beta: files-api-2025-04-14` headers sent by the official Anthropic SDKs are accepted and ignored by this gateway — the Files API works without them. The examples below include them only for parity with SDK-generated requests.

### Upload a File

```bash
curl -X POST "$BASE/v1/files" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14" \
  -F "file=@document.pdf;type=application/pdf"
```

**Response:**

```json
{
  "id": "file_0190c51c7de7455d9b8c2efe27dfbf67",
  "type": "file",
  "filename": "document.pdf",
  "mime_type": "application/pdf",
  "size_bytes": 102400,
  "created_at": "2025-04-15T12:00:00Z",
  "downloadable": true
}
```

### Upload via JSON Body (MCP and AI Agents)

When using MCP tools or HTTP clients that cannot construct `multipart/form-data` requests, pass the file as a base64 string, data URI, HTTPS URL, or S3 URI in a JSON body instead.

**Data URI (inline content):**

```bash
curl -X POST "$BASE/v1/files" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14" \
  -H "Content-Type: application/json" \
  -d '{"file": "data:text/plain;base64,SGVsbG8gV29ybGQ="}'
```

**HTTPS URL (server fetches the file):**

```bash
curl -X POST "$BASE/v1/files" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14" \
  -H "Content-Type: application/json" \
  -d '{"file": "https://example.com/document.pdf"}'
```

All variants return the same `FileMetadata` response as a multipart upload.

### Retrieve Metadata

```bash
curl "$BASE/v1/files/file_0190c51c7de7455d9b8c2efe27dfbf67" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14"
```

### List Files

```bash
# Default (newest first, up to 20 files)
curl "$BASE/v1/files" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14"

# Forward pagination: the page following a given ID (older files)
curl "$BASE/v1/files?after_id=file_0190c51c7de7455d9b8c2efe27dfbf67&limit=20" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14"

# Backward pagination: the page preceding a given ID (newer files)
curl "$BASE/v1/files?before_id=file_0190c51c7de7455d9b8c2efe27dfbf67" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14"
```

### Download Content

```bash
curl "$BASE/v1/files/file_0190c51c7de7455d9b8c2efe27dfbf67/content" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14" \
  -o downloaded.pdf
```

### Delete a File

```bash
curl -X DELETE "$BASE/v1/files/file_0190c51c7de7455d9b8c2efe27dfbf67" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14"
```

**Response:**

```json
{
  "id": "file_0190c51c7de7455d9b8c2efe27dfbf67",
  "type": "file_deleted"
}
```

## Messages Integration

Reference an uploaded file inside a `POST /v1/messages` request as a document or image source:

**Document (PDF or other supported format):**

```bash
curl -X POST "$BASE/v1/messages" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
    "max_tokens": 512,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "document",
            "source": {
              "type": "file",
              "file_id": "file_0190c51c7de7455d9b8c2efe27dfbf67"
            }
          },
          {
            "type": "text",
            "text": "Summarize this document."
          }
        ]
      }
    ]
  }'
```

**Image:**

```bash
curl -X POST "$BASE/v1/messages" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
    "max_tokens": 256,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "image",
            "source": {
              "type": "file",
              "file_id": "file_0190c51c7de7455d9b8c2efe27dfbf68"
            }
          },
          {
            "type": "text",
            "text": "Describe this image."
          }
        ]
      }
    ]
  }'
```

## End-to-End Example

```bash
# 1. Upload a file
FILE_ID=$(curl -s -X POST "$BASE/v1/files" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14" \
  -F "file=@document.pdf;type=application/pdf" | jq -r .id)
echo "Uploaded: $FILE_ID"

# 2. Reference in a message
curl -X POST "$BASE/v1/messages" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"anthropic.claude-haiku-4-5-20251001-v1:0\",
    \"max_tokens\": 512,
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"document\", \"source\": {\"type\": \"file\", \"file_id\": \"$FILE_ID\"}},
        {\"type\": \"text\", \"text\": \"What is the key finding in this document?\"}
      ]
    }]
  }"

# 3. Cleanup
curl -X DELETE "$BASE/v1/files/$FILE_ID" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14"
```

## Referencing Uploaded Files via the `file-id:` URI Scheme

The native `{"type": "file", "file_id": "..."}` source shown above is the Anthropic-compatible way to reference an uploaded file in Messages content blocks. For **string-overloaded** file fields that already accept URI schemes like `s3://`, `https://`, or `data:` — for example image and document content blocks with `source.type` `url` or `base64` — this implementation defines an additional project-local URI scheme:

```text
file-id:<file-id>
```

!!! tip "Project-local URI scheme — `file-id:`"
    `file-id:` is an **extension beyond the original Anthropic API**, parallel to the existing `s3://`, `https://`, and `data:` schemes already accepted on the same fields. It lets a client upload a file once and reuse it across Messages content blocks (image / document `source.url` and `source.data`), as well as the OpenAI-compatible routes — without re-uploading.

    * **Where accepted:** any string-overloaded file field. For Anthropic Messages: image and document content blocks where `source.type` is `url` (with `source.url: "file-id:<id>"`) or `base64` (with `source.data: "file-id:<id>"`).
    * **Where unchanged:** the typed `{"type": "file", "file_id": "..."}` source already accepted by the Anthropic API stays exactly as-is — do not wrap those bare IDs in `file-id:`.
    * **Where rejected:** the Files API ingest endpoint (`POST /v1/files`) returns **400** for `file-id:` inputs, because resolving it there would silently clone an existing file.
    * **Detection:** match is **case-sensitive** (`file-id:`, lowercase) with no whitespace stripping; the payload after the prefix must be a valid Files API ID, otherwise the request fails with `400 invalid_request_error`. A missing or expired file returns `404 not_found`.

### Worked Example — Send an Uploaded Image in Messages

```bash
# 1. Upload the file once.
FILE_ID=$(curl -s -X POST "$BASE/v1/files" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14" \
  -F "file=@chart.png;type=image/png" | jq -r .id)

# 2. Reference it via file-id: in a Messages request.
curl -X POST "$BASE/v1/messages" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"anthropic.claude-haiku-4-5-20251001-v1:0\",
    \"max_tokens\": 256,
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"Describe this chart.\"},
        {\"type\": \"image\", \"source\": {\"type\": \"url\", \"url\": \"file-id:${FILE_ID}\"}}
      ]
    }]
  }"
```

See the [OpenAI Files API documentation](api_openai_files.md#referencing-uploaded-files-via-the-file-id-uri-scheme) for the full list of supported routes — the same scheme works identically across both API surfaces.

## Errors

| HTTP | Cause                                                                               |
|------|-------------------------------------------------------------------------------------|
| 400  | Invalid filename characters                                                         |
| 400  | `file-id:` URI passed to the upload endpoint (`POST /v1/files`)                     |
| 400  | Malformed ID after the `file-id:` prefix in a Messages content block                |
| 404  | File not found or already deleted                                                   |
| 503  | `AWS_S3_BUCKET` is not configured                                                   |

## Configuration

Files are stored in S3 under the prefix configured by [`AWS_S3_FILES_PREFIX`](operations_configuration.md#aws-s3-files-prefix) (default: `files/`). All file IDs are shared across the OpenAI and Anthropic endpoints — a file uploaded via one API can be downloaded or deleted via the other.

---

**Use files across multiple requests without re-uploading.** See [OpenAI Files API](api_openai_files.md) for the OpenAI-compatible equivalent, including `expires_after` support.
