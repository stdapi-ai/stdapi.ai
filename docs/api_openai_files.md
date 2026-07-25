---
title: Files API - OpenAI-Compatible File Storage
description: Upload, manage, and reference files in chat completions using the OpenAI-compatible Files API backed by Amazon S3. Supports documents, images, expiry, cursor pagination, and multipart uploads.
keywords: Files API, OpenAI files, file upload, S3 file storage, chat completion file, expires_after, cursor pagination, AWS Bedrock files, multipart upload
---

# Files API (OpenAI Compatible)

Upload and manage files via an OpenAI-compatible interface. Files are stored in Amazon S3 and can be referenced directly in Chat Completions requests. Large files can be uploaded in parts using the Uploads API.

<div class="grid cards" markdown>

- :material-upload: __Simple Upload__
  <br>Upload any file with a single `multipart/form-data` request. Files are immediately available for use in inference.

- :material-upload-multiple: __Multipart Upload__
  <br>Stream large files in parts via the Uploads API. Parts are assembled by S3 native multipart upload.

- :material-clock-outline: __Optional Expiry__
  <br>Set `expires_after` to automatically expire files after a configurable number of seconds (1 hour to 30 days).

- :material-format-list-bulleted: __Paginated Listing__
  <br>List files with ascending or descending order and cursor-based pagination using the `after` parameter.

- :material-file-document-multiple: __Chat Integration__
  <br>Reference uploaded files directly in Chat Completions messages using `"type": "file"` content parts.

</div>

## Available Endpoints

| Endpoint                           | Method   | Description                            | MCP Tool                 |
|------------------------------------|----------|----------------------------------------|--------------------------|
| `/v1/files`                        | `POST`   | Upload a file                          | `openai_file`            |
| `/v1/files`                        | `GET`    | List files with pagination             | `openai_file_list`       |
| `/v1/files/{file_id}`              | `GET`    | Retrieve file metadata                 | `openai_files_get`       |
| `/v1/files/{file_id}`              | `DELETE` | Delete a file                          | `openai_files_delete`    |
| `/v1/files/{file_id}/content`      | `GET`    | Download raw file bytes                | `openai_file_content`    |
| `/v1/uploads`                      | `POST`   | Create a multipart upload session      | `openai_upload`          |
| `/v1/uploads/{upload_id}/parts`    | `POST`   | Add a part to an upload session        | `openai_upload_part`     |
| `/v1/uploads/{upload_id}/complete` | `POST`   | Complete the upload and produce a file | `openai_upload_complete` |
| `/v1/uploads/{upload_id}/cancel`   | `POST`   | Cancel a pending upload session        | `openai_upload_cancel`   |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                    |                  Status                  | Notes                                                                                         |
|----------------------------|:----------------------------------------:|-----------------------------------------------------------------------------------------------|
| **Upload**                 |                                          |                                                                                               |
| `file` (multipart)         |   :material-check-circle:{ .success }    | Required binary form field                                                                    |
| `file` (JSON body)         | :material-plus-circle:{ .extra-feature } | Base64, data URI, HTTPS URL, or S3 URI — for MCP / AI agents                                  |
| `purpose`                  |   :material-minus-circle:{ .partial }    | Strictly validated against `assistants`, `batch`, `fine-tune`, `vision`, `user_data`, `evals` (others rejected); has no behavioral effect on storage or processing |
| `expires_after[anchor]`    |   :material-check-circle:{ .success }    | Only `"created_at"` is accepted; expiry is computed from `expires_after[seconds]`             |
| `expires_after[seconds]`   |   :material-check-circle:{ .success }    | Range: 3 600 – 2 592 000 (1 hour – 30 days)                                                   |
| **Listing**                |                                          |                                                                                               |
| `order=asc` / `order=desc` |   :material-check-circle:{ .success }    | Ascending and descending supported; default `desc`                                            |
| `after` cursor             |   :material-check-circle:{ .success }    | Forward cursor pagination                                                                     |
| `limit`                    |   :material-check-circle:{ .success }    | 1 – 10 000; default 10 000                                                                    |
| `purpose` filter           |   :material-check-circle:{ .success }    | Filter results by uploaded purpose                                                            |
| **File size cap**          | :material-plus-circle:{ .extra-feature } | No artificial limit; S3 object limit (~5 TB)                                                  |
| **Expiry enforcement**     |   :material-check-circle:{ .success }    | Expired files return 404 at read time; S3 Lifecycle as backstop                               |
| **Chat integration**       |   :material-check-circle:{ .success }    | Use `file_id` in `type: "file"` content parts                                                 |
| `status` field             |   :material-check-circle:{ .success }    | Always `"processed"` — no async processing pipeline                                           |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success } **Supported** — Fully compatible with OpenAI API
* :material-minus-circle:{ .partial } **Partial** — Supported with limitations or differences
* :material-close-circle:{ .unsupported } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature } **Extra Feature** — Enhanced capability beyond OpenAI API

</div>

## Quick Start

### Upload a File

```bash
curl -X POST "$BASE/v1/files" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "file=@document.pdf;type=application/pdf" \
  -F "purpose=assistants"
```

**Response:**

```json
{
  "id": "file-0190c51c7de7455d9b8c2efe27dfbf67",
  "object": "file",
  "bytes": 102400,
  "created_at": 1745000000,
  "filename": "document.pdf",
  "purpose": "assistants",
  "status": "processed"
}
```

### Upload via JSON Body (MCP and AI Agents)

When using MCP tools or HTTP clients that cannot construct `multipart/form-data` requests, pass the file as a base64 string, data URI, HTTPS URL, or S3 URI in a JSON body instead.

**Data URI (inline content):**

```bash
curl -X POST "$BASE/v1/files" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "file": "data:text/plain;base64,SGVsbG8gV29ybGQ=",
    "purpose": "user_data"
  }'
```

**HTTPS URL (server fetches the file):**

```bash
curl -X POST "$BASE/v1/files" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "file": "https://example.com/document.pdf",
    "purpose": "assistants"
  }'
```

**Raw base64:**

```bash
curl -X POST "$BASE/v1/files" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "file": "SGVsbG8gV29ybGQ=",
    "purpose": "user_data"
  }'
```

All three variants return the same `FileObject` response as a multipart upload.

### Upload with Expiry

```bash
curl -X POST "$BASE/v1/files" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "file=@temp.txt;type=text/plain" \
  -F "purpose=assistants" \
  -F "expires_after[anchor]=created_at" \
  -F "expires_after[seconds]=3600"
```

!!! info "Expiry Semantics"
    Expiry is enforced lazily at read time: calls to retrieve metadata, download content, or reference the file in inference return HTTP 404 once the expiry time has passed. S3 Lifecycle rules clean up expired objects as a background backstop.

### Retrieve Metadata

```bash
curl "$BASE/v1/files/file-0190c51c7de7455d9b8c2efe27dfbf67" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### List Files

```bash
# Default (newest first, up to 10 000 files)
curl "$BASE/v1/files" \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Ascending, limit 20
curl "$BASE/v1/files?order=asc&limit=20" \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Next page using after cursor
curl "$BASE/v1/files?order=asc&limit=20&after=file-0190c51c7de7455d9b8c2efe27dfbf67" \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Filter by purpose
curl "$BASE/v1/files?purpose=fine-tune" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Download Content

```bash
curl "$BASE/v1/files/file-0190c51c7de7455d9b8c2efe27dfbf67/content" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -o downloaded.pdf
```

### Delete a File

```bash
curl -X DELETE "$BASE/v1/files/file-0190c51c7de7455d9b8c2efe27dfbf67" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**Response:**

```json
{
  "id": "file-0190c51c7de7455d9b8c2efe27dfbf67",
  "object": "file",
  "deleted": true
}
```

## Uploads API

The Uploads API lets you stream large files to S3 in parts without buffering the entire file in memory. Each upload session is backed by an S3 native multipart upload.

### Upload ID format

Upload IDs use the same base32-encoded payload as file IDs. The prefix is swapped (`upload_` → `file-`) when the upload is completed, so the resulting file ID is known upfront.

### Create an Upload Session

Set the optional `expires_after` object to give the resulting file a TTL (same behavior as `expires_after` on `/v1/files`, 1 hour to 30 days). The pending upload's own `expires_at` always reflects the upload session's expiry (1 day); the requested file TTL appears on the resulting file object once the upload is completed.

```bash
curl -X POST "$BASE/v1/uploads" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "bytes": 6291456,
    "filename": "large_dataset.bin",
    "mime_type": "application/octet-stream",
    "purpose": "assistants",
    "expires_after": {"anchor": "created_at", "seconds": 86400}
  }'
```

**Response:**

```json
{
  "id": "upload_0190c51c7de7455d9b8c2efe27dfbf67",
  "object": "upload",
  "status": "pending",
  "bytes": 6291456,
  "filename": "large_dataset.bin",
  "purpose": "assistants",
  "created_at": 1745000000,
  "expires_at": 1745086400
}
```

### Add Parts

Each part except the last must be at least 5 MiB (S3 minimum part size). The last part may be any size.

**Binary upload (multipart/form-data):**

```bash
curl -X POST "$BASE/v1/uploads/upload_0190c51c7de7455d9b8c2efe27dfbf67/parts" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "data=@part1.bin"
```

**JSON body (MCP and AI agents):**

When using MCP tools or HTTP clients that cannot construct `multipart/form-data`, pass the chunk as a base64 string, data URI, HTTPS URL, or S3 URI:

```bash
curl -X POST "$BASE/v1/uploads/upload_0190c51c7de7455d9b8c2efe27dfbf67/parts" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "data": "data:application/octet-stream;base64,AAEC..."
  }'
```

**Response:**

```json
{
  "id": "part_0190c51c7de700010001abcdef012345",
  "object": "upload.part",
  "upload_id": "upload_0190c51c7de7455d9b8c2efe27dfbf67",
  "created_at": 1745000001
}
```

### Complete the Upload

```bash
curl -X POST "$BASE/v1/uploads/upload_0190c51c7de7455d9b8c2efe27dfbf67/complete" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "part_ids": [
      "part_0190c51c7de700010001abcdef012345",
      "part_0190c51c7de700020002fedcba987654"
    ]
  }'
```

`part_ids` must be listed in ascending upload order (part 1, part 2, ...); S3 cannot reassemble multipart uploads out of order, so a reordered list is rejected with a 400 error rather than silently reordered.

**Response:** A completed `Upload` object with the `file` field populated.

```json
{
  "id": "upload_0190c51c7de7455d9b8c2efe27dfbf67",
  "object": "upload",
  "status": "completed",
  "bytes": 6291456,
  "filename": "large_dataset.bin",
  "purpose": "assistants",
  "created_at": 1745000000,
  "expires_at": 1745086400,
  "file": {
    "id": "file-0190c51c7de7455d9b8c2efe27dfbf67",
    "object": "file",
    "bytes": 6291456,
    "created_at": 1745000005,
    "filename": "large_dataset.bin",
    "purpose": "assistants",
    "status": "processed"
  }
}
```

### Cancel an Upload

```bash
curl -X POST "$BASE/v1/uploads/upload_0190c51c7de7455d9b8c2efe27dfbf67/cancel" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

!!! info "Upload sessions expire after 1 day"
    If an upload is not completed within 1 day of creation it is automatically aborted. Parts uploaded to an expired session are discarded by S3.

### Uploads Feature Compatibility

<div class="feature-table" markdown>

| Feature                  |                  Status                  | Notes                                                        |
|--------------------------|:----------------------------------------:|--------------------------------------------------------------|
| `bytes` (declared size)  |   :material-check-circle:{ .success }    | Validated at completion against actual assembled size        |
| `filename`               |   :material-check-circle:{ .success }    | Carried through to the final file object                     |
| `mime_type`              |   :material-check-circle:{ .success }    | Set as the S3 `ContentType` for the assembled object         |
| `purpose`                |   :material-check-circle:{ .success }    | Echoed to the final file object                              |
| Part data (binary)       |   :material-check-circle:{ .success }    | Standard `multipart/form-data` binary upload via the `data` field |
| Part data (JSON body)    | :material-plus-circle:{ .extra-feature } | Base64, data URI, HTTPS URL, or S3 URI — for MCP / AI agents |
| Part ordering            |   :material-minus-circle:{ .partial }    | `part_ids` must be listed in ascending upload order; S3 cannot reassemble out of order |
| `md5` checksum           | :material-close-circle:{ .unsupported }  | Accepted but not validated                                   |
| Session TTL              |   :material-check-circle:{ .success }    | 1 day from creation                                          |

</div>

### End-to-End Example (Uploads)

```bash
# 1. Create an upload session
UPLOAD_ID=$(curl -s -X POST "$BASE/v1/uploads" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "bytes": 6291456,
    "filename": "large_file.bin",
    "mime_type": "application/octet-stream",
    "purpose": "assistants"
  }' | jq -r .id)

# 2. Upload parts (first part >= 5 MiB, last part any size)
PART_A_ID=$(curl -s -X POST "$BASE/v1/uploads/$UPLOAD_ID/parts" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "data=@part1.bin" | jq -r .id)

PART_B_ID=$(curl -s -X POST "$BASE/v1/uploads/$UPLOAD_ID/parts" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "data=@part2.bin" | jq -r .id)

# 3. Complete — file is immediately available
FILE_ID=$(curl -s -X POST "$BASE/v1/uploads/$UPLOAD_ID/complete" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"part_ids\": [\"$PART_A_ID\", \"$PART_B_ID\"]}" | jq -r .file.id)
echo "File ready: $FILE_ID"

# Cleanup
curl -X DELETE "$BASE/v1/files/$FILE_ID" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

## Chat Completions Integration

Reference an uploaded file inside a `POST /v1/chat/completions` message using `"type": "file"`:

```bash
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
    "max_tokens": 512,
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Summarise this document."
          },
          {
            "type": "file",
            "file": {
              "file_id": "file-0190c51c7de7455d9b8c2efe27dfbf67"
            }
          }
        ]
      }
    ]
  }'
```

!!! info "Model Support"
    Document and image file types are supported by models that accept those input modalities. Use a vision-capable model (e.g. Claude Haiku or Sonnet) when passing PDFs or images via `file_id`. Amazon Nova models do not currently support document inputs.

## End-to-End Example

```bash
# 1. Upload the file
FILE_ID=$(curl -s -X POST "$BASE/v1/files" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "file=@document.pdf;type=application/pdf" \
  -F "purpose=assistants" | jq -r .id)

# 2. Reference in a chat completion
curl -X POST "$BASE/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"anthropic.claude-haiku-4-5-20251001-v1:0\",
    \"max_tokens\": 512,
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"What is the key finding in this document?\"},
        {\"type\": \"file\", \"file\": {\"file_id\": \"$FILE_ID\"}}
      ]
    }]
  }"

# 3. Cleanup
curl -X DELETE "$BASE/v1/files/$FILE_ID" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

## Referencing Uploaded Files via the `file-id:` URI Scheme

The native `file_id` JSON field shown above is the OpenAI-compatible way to reference an uploaded file in routes that have a typed `file_id` slot (chat completions content parts, responses content parts, image edits, …). For **string-overloaded** file fields that already accept URI schemes like `s3://`, `https://`, or `data:` — and therefore do **not** have a typed `file_id` slot — this implementation defines an additional project-local URI scheme:

```
file-id:<file-id>
```

!!! tip "Project-local URI scheme — `file-id:`"
    `file-id:` is an **extension beyond the original OpenAI API**, parallel to the existing `s3://`, `https://`, and `data:` schemes already accepted on the same fields. It lets a client upload a file once via `/v1/files` and reuse it across embeddings, transcriptions, image edits, chat completions and other routes — without re-uploading or exposing an S3 URL.

    * **Where accepted:** any string-overloaded file field (e.g. `image_url.url`, `input_audio.data`, `file.file_data` in chat completions; `input` on `/v1/embeddings`; `file` on audio transcription/translation; `image_url`/`mask` on image edits/variations; Anthropic image/document `source.url` and `source.data`).
    * **Where rejected:** the ingest endpoints (`POST /v1/files`, `POST /v1/uploads/{id}/parts`) return **400** for `file-id:` inputs, because resolving them there would silently clone an existing file.
    * **Where unchanged:** JSON fields that already accept a typed `file_id` (e.g. chat content parts `{"type":"file","file":{"file_id":"file-…"}}`) keep working exactly as in the OpenAI API; do not wrap those bare IDs in the `file-id:` prefix.
    * **Detection:** match is **case-sensitive** (`file-id:`, lowercase) with no whitespace stripping; the payload after the prefix must be a valid Files API ID, otherwise the request fails with `400 invalid_request_error`.
    * **Resolution:** the file is fetched from its underlying S3 object using the same code path as `s3://` URIs. A missing or expired file returns `404 not_found`. Content-type validation is delegated to each route, so an audio file passed to an image-only field is rejected the same way an `https://…/foo.mp3` URL would be.

### Worked example — embed an uploaded file

```bash
# 1. Upload the file once.
FILE_ID=$(curl -s -X POST "$BASE/v1/files" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F "file=@research_paper.pdf;type=application/pdf" \
  -F "purpose=assistants" | jq -r .id)

# 2. Reference it via file-id: in another route.
curl -X POST "$BASE/v1/embeddings" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"amazon.nova-2-multimodal-embeddings-v1:0\",
    \"input\": \"file-id:${FILE_ID}\"
  }"

# 3. (Optional) delete the file when no longer needed.
curl -X DELETE "$BASE/v1/files/${FILE_ID}" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

## Error Reference

| HTTP | Cause                                                              |
|------|--------------------------------------------------------------------|
| 400  | Invalid `expires_after` range, bad filename, size mismatch, or unknown part ID |
| 404  | File or upload not found, already deleted, expired, or not pending |
| 503  | `AWS_S3_BUCKET` is not configured                                  |

## Configuration

Files are stored in S3 under the prefix configured by [`AWS_S3_FILES_PREFIX`](operations_configuration.md#aws-s3-files-prefix) (default: `files/`). Configure S3 Lifecycle rules on this prefix to automatically delete expired objects and apply Intelligent-Tiering.

---

**Store files once, use them across requests.** See [Anthropic Files API](api_anthropic_files.md) for the Anthropic-compatible equivalent.
