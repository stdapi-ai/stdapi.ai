---
title: Files API - Anthropic-Compatible File Storage
description: Upload, manage, and reference files in Anthropic Messages requests using the Anthropic-compatible Files API backed by Amazon S3. Supports documents, images, and bidirectional cursor pagination.
keywords: Files API, Anthropic files, file upload, S3 file storage, Anthropic messages file, document source, cursor pagination, AWS Bedrock files
---

# Files API (Anthropic Compatible)

!!! note "Route Prefix"
    By default, all Anthropic-compatible routes are prefixed with `/anthropic`. This means the Files API is available at `/anthropic/v1/files` instead of `/v1/files`. You can customize this prefix using the `ANTHROPIC_ROUTES_PREFIX` configuration variable documented in [Operations Configuration](operations_configuration.md#anthropic-routes-prefix).

Upload and manage files via an Anthropic-compatible interface. Files are stored in Amazon S3 and can be referenced directly in Messages requests as document or image sources.

<div class="grid cards" markdown>

- :material-upload: __Simple Upload__
  <br>Upload any file with a single `multipart/form-data` request. Files are immediately available for use in inference.

- :material-swap-vertical: __Bidirectional Pagination__
  <br>Traverse your file list in both directions using `after_id` and `before_id` cursors, just like the Anthropic SDK's `Page<T>` interface.

- :material-file-document-multiple: __Messages Integration__
  <br>Reference uploaded files directly in Messages requests as document or image source blocks using `"type": "file"`.

- :material-download: __Content Download__
  <br>Download raw file bytes at any time via the `/content` endpoint.

</div>

## Available Endpoints

| Endpoint                      | Method   | Description                    | MCP Tool               |
|-------------------------------|----------|--------------------------------|------------------------|
| `/v1/files`                   | `POST`   | Upload a file                  | `anthropic_file`       |
| `/v1/files`                   | `GET`    | List files with pagination     | `anthropic_file_list`  |
| `/v1/files/{file_id}`         | `GET`    | Retrieve file metadata         | `anthropic_files_get`  |
| `/v1/files/{file_id}`         | `DELETE` | Delete a file                  | `anthropic_files_delete` |
| `/v1/files/{file_id}/content` | `GET`    | Download raw file bytes        | `anthropic_file_content` |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                  |                  Status                  | Notes                                                            |
|--------------------------|:----------------------------------------:|------------------------------------------------------------------|
| **Upload**               |                                          |                                                                  |
| `file` (multipart)       |   :material-check-circle:{ .success }    | Required binary form field                                       |
| **Listing**              |                                          |                                                                  |
| `after_id` cursor        |   :material-check-circle:{ .success }    | Forward cursor: returns files newer than the given ID            |
| `before_id` cursor       |   :material-check-circle:{ .success }    | Backward cursor: returns files older than the given ID           |
| `limit`                  |   :material-check-circle:{ .success }    | 1 – 1 000; default 20                                            |
| **File size cap**        | :material-plus-circle:{ .extra-feature } | No artificial limit; S3 object limit (~5 TB)                     |
| **Messages integration** |   :material-check-circle:{ .success }    | `"source": {"type": "file", "file_id": "..."}` in document/image |
| `downloadable` field     |   :material-minus-circle:{ .partial }    | Always `true`; spec default is `false` for user-uploaded files   |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success } **Supported** — Fully compatible with Anthropic API
* :material-minus-circle:{ .partial } **Partial** — Implemented with minor deviations from spec
* :material-plus-circle:{ .extra-feature } **Extra Feature** — Enhanced capability beyond Anthropic API

</div>

## Quick Start

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

### Retrieve Metadata

```bash
curl "$BASE/v1/files/file_0190c51c7de7455d9b8c2efe27dfbf67" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14"
```

### List Files

```bash
# Default (oldest first, up to 20 files)
curl "$BASE/v1/files" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14"

# Forward pagination: files after a given ID
curl "$BASE/v1/files?after_id=file_0190c51c7de7455d9b8c2efe27dfbf67&limit=20" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14"

# Backward pagination: files before a given ID
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
            "text": "Summarise this document."
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

## Using the Anthropic Python SDK

```python
import io
from anthropic import Anthropic

client = Anthropic(base_url="http://localhost:8000", api_key="...")

# Upload
file = client.beta.files.upload(
    file=("document.pdf", open("document.pdf", "rb"), "application/pdf")
)
print(f"Uploaded: {file.id}")

# Reference in a message
response = client.beta.messages.create(
    model="anthropic.claude-haiku-4-5-20251001-v1:0",
    max_tokens=512,
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {"type": "file", "file_id": file.id},
                },
                {
                    "type": "text",
                    "text": "What is the key finding in this document?",
                },
            ],
        }
    ],
)
print(response.content[0].text)

# Cleanup
client.beta.files.delete(file.id)
```

### Paginating with the SDK

The Anthropic SDK's `Page` object supports forward and backward traversal:

```python
# List all files (auto-paginated)
page = client.beta.files.list(limit=50)
for file in page.data:
    print(file.id, file.filename)

# Forward cursor
next_page = client.beta.files.list(after_id=page.last_id)

# Backward cursor
prev_page = client.beta.files.list(before_id=page.first_id)
```

## Error Reference

| HTTP | Cause                             |
|------|-----------------------------------|
| 400  | Invalid filename characters       |
| 404  | File not found or already deleted |
| 503  | `AWS_S3_BUCKET` is not configured |

## Configuration

Files are stored in S3 under the prefix configured by [`AWS_S3_FILES_PREFIX`](operations_configuration.md#aws-s3-files-prefix) (default: `files/`). All file IDs are shared across the OpenAI and Anthropic endpoints — a file uploaded via one API can be downloaded or deleted via the other.

---

**Use files across multiple requests without re-uploading.** See [OpenAI Files API](api_openai_files.md) for the OpenAI-compatible equivalent, including `expires_after` support.
