---
title: Files API - Anthropic-Compatible File Storage
description: Upload, manage, and reference files in Anthropic Messages requests using the Anthropic-compatible Files API backed by Amazon S3. Supports documents, images, selection by ID, page cursors, and bidirectional cursor pagination.
keywords: Files API, Anthropic files, file upload, S3 file storage, Anthropic messages file, document source, cursor pagination, page cursor, next_page, SDK auto-pagination, list files by id, ids filter, scope_id, AWS Bedrock files
---

# Files API (Anthropic Compatible)

Upload and manage files via an Anthropic-compatible interface. Files are stored in Amazon S3 and can be referenced directly in Messages requests as document or image sources. Served under `/anthropic` by default; the examples below use `$BASE`, which includes that prefix.

## At a glance

- :material-upload: **Simple upload** — Upload any file with a single `multipart/form-data` request. Files are immediately available for use in inference.
- :material-page-next: **Page cursors** — Every listing carries a `next_page` cursor; send it back as `page` to get the following page, and the official SDK's own iteration walks the whole list for you.
- :material-swap-vertical: **Bidirectional pagination** — Traverse your file list in both directions using the `after_id` and `before_id` ID cursors, an addition to the `page` cursor rather than a replacement for it.
- :material-format-list-checks: **Select by ID** — Pass `ids` to fetch a known set of files in one call, without paging through the list. Up to 100 IDs; the ones that name no readable file are simply left out.
- :material-file-document-multiple: **Messages integration** — Reference uploaded files directly in Messages requests as document or image source blocks using `"type": "file"`.
- :material-download: **Content download** — Download raw file bytes at any time via the `/content` endpoint.
- :material-database: **One file store for both dialects** — a file uploaded here is readable and deletable through the [OpenAI Files API](api_openai_files.md), and vice versa; both are backed by the same S3 bucket.
- :material-swap-horizontal: **Differs from the Anthropic API:** no file size cap beyond the ~78 GiB a direct upload reaches, `downloadable` is always `true`, and `scope_id` filtering is refused — see [Limits and behaviour to know](#limits-and-behaviour-to-know).

!!! info "Base URL and route prefix"
    By default, all Anthropic-compatible routes are prefixed with `/anthropic`. This means the Files API is available at `/anthropic/v1/files` instead of `/v1/files`. You can customize this prefix using the `ANTHROPIC_ROUTES_PREFIX` configuration variable documented in [HTTP Server and MCP](operations_configuration_server.md#anthropic-routes-prefix).

    The `curl` examples on this page use a `$BASE` variable that **must include this prefix** — set it to your scheme and host followed by `ANTHROPIC_ROUTES_PREFIX`:

    ```bash
    export BASE="https://your-host/anthropic"  # <scheme>://<host> + ANTHROPIC_ROUTES_PREFIX
    ```

    `ANTHROPIC_ROUTES_PREFIX` must always be a non-empty path and is validated at startup to differ from `OPENAI_ROUTES_PREFIX` (the server refuses to start otherwise). This Anthropic-compatible Files API is therefore always served on its own path, distinct from the [OpenAI-compatible Files API](api_openai_files.md).

```bash
curl -X POST "$BASE/v1/files" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -F "file=@document.pdf;type=application/pdf"
```

## Endpoints { #available-endpoints }

| Endpoint                      | Method   | What It Does               | Powered By | MCP Tool                 |
|-------------------------------|----------|----------------------------|------------|--------------------------|
| `/v1/files`                   | `POST`   | Upload a file              | Amazon S3  | `anthropic_file`         |
| `/v1/files`                   | `GET`    | List files with pagination | Amazon S3  | `anthropic_file_list`    |
| `/v1/files/{file_id}`         | `GET`    | Retrieve file metadata     | Amazon S3  | `anthropic_files_get`    |
| `/v1/files/{file_id}`         | `DELETE` | Delete a file              | Amazon S3  | `anthropic_files_delete` |
| `/v1/files/{file_id}/content` | `GET`    | Download raw file bytes    | Amazon S3  | `anthropic_file_content` |

## Feature compatibility { #feature-compatibility }

<div class="feature-table" markdown>

| Feature                  |                  Status                  | Notes                                                            |
|--------------------------|:----------------------------------------:|------------------------------------------------------------------|
| **Upload**               |                                          |                                                                  |
| `file` (multipart)       |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Required binary form field                                       |
| `file` (JSON body)       | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Base64, data URI, HTTPS URL, or S3 URI — for MCP / AI agents    |
| `expires_in_seconds`     |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | 3 600 – 7 776 000 (1 hour – 90 days); omit for no expiry          |
| `expires_at`             |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | RFC 3339 string on every response; `null` when the file has no TTL |
| **Listing**              |                                          |                                                                  |
| Listing order            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Most recently created first, by `created_at`                     |
| `ids` selection          |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Up to 100 IDs after de-duplication, served as a single page      |
| `page` / `next_page`     |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Forward page cursor; `null` on the last page                     |
| `after_id` cursor        | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Forward cursor: returns files older than the given ID            |
| `before_id` cursor       | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Backward cursor: returns files newer than the given ID           |
| `first_id` / `last_id` / `has_more` | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Page edges and continuation flag, served alongside `next_page`   |
| `limit`                  |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | 1 – 1 000; default 20 — ignored alongside `ids`                  |
| `scope_id` filter        | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Refused with a `400`: files here are not associated with a scope |
| **File size cap**        | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | No limit imposed by stdapi.ai; an upload streams in fixed 8 MiB parts, so S3's 10,000-part ceiling caps it at ~78 GiB |
| **Messages integration** |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | `"source": {"type": "file", "file_id": "..."}` in document/image |
| `downloadable` field     |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Always `true`; spec default is `false` for user-uploaded files   |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with Anthropic API
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Implemented with minor deviations from spec
* :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } **Extra Feature** — Enhanced capability beyond Anthropic API
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation

</div>

## Working with files { #quick-start }

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
  "expires_at": null,
  "downloadable": true
}
```

### Upload a File with an Expiry

Add `expires_in_seconds` (3 600 – 7 776 000, one hour to ninety days) to have the file expire automatically instead of persisting until deleted:

```bash
curl -X POST "$BASE/v1/files" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14" \
  -F "file=@document.pdf;type=application/pdf" \
  -F "expires_in_seconds=3600"
```

The response's `expires_at` carries the resulting instant as an RFC 3339 string. A file uploaded this way expires at the same time, and is refused the same way, whichever of the [OpenAI](api_openai_files.md) or Anthropic routes reads it afterwards.

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

A JSON body is held whole to decode it, so this form alone is bounded: a body carrying more than the base64 form of a 64 MiB file — or of [`MAX_INPUT_FILE_SIZE`](operations_configuration_server.md#max-input-file-size) when one is configured — is refused with a `413` naming the maximum. The URL and S3 URI variants carry no content, and a `multipart/form-data` upload is streamed, so neither is affected.

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

# Next page: send back the `next_page` value the previous response returned
curl "$BASE/v1/files?limit=20&page=page_MDE5MGM1MWM3ZGU3NDU1ZDliOGMyZWZlMjdkZmJmNjc" \
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

# Selection: only the files named, newest first, in one page
curl "$BASE/v1/files?ids=file_0190c51c7de7455d9b8c2efe27dfbf67&ids=file_0190c51c7de7455d9b8c2efe27dfbf68" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-beta: files-api-2025-04-14"
```

Each response carries `next_page`: a `page_`-prefixed cursor to the following page, or `null` on the last one. Send it back unchanged as `page` to continue, and the official SDK's own iteration (`client.beta.files.list()` used as an iterator) walks the whole list on its own. A `page` cursor is a complete instruction on its own, so combining it with `ids`, `after_id` or `before_id` is a `400`, as is a cursor this API never issued.

`ids` selects instead of paging: repeat the parameter once per file, up to 100 distinct IDs. The whole selection comes back in one response, so `after_id`, `before_id` and `limit` have no effect alongside it, and an ID that names a deleted, expired or unknown file is left out of `data` rather than failing the request. More than 100 distinct IDs is a `400`.

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

## Configuration

Files are stored in S3 under the prefix configured by [`AWS_S3_FILES_PREFIX`](operations_configuration_storage.md#aws-s3-files-prefix) (default: `files/`). All file IDs are shared across the OpenAI and Anthropic endpoints — a file uploaded via one API can be downloaded or deleted via the other.

## Limits and behaviour to know

**The API needs an S3 bucket.** Every route stores and reads objects in [`AWS_S3_BUCKET`](operations_configuration_storage.md#aws-s3-bucket); without it configured, all five endpoints answer `529 overloaded_error` — the Anthropic envelope's form of an unavailable feature.

**Two ways to page, and they do not mix.** The listing serves the `page`/`next_page` cursor the official Files API uses, *and* keeps the ID-cursor envelope (`first_id`, `last_id`, `has_more`, and the `after_id`/`before_id` query parameters) that clients here were written against, and which the official API refuses outright. Pick one scheme per request: `page` together with `ids`, `after_id` or `before_id` is a `400`, because each of them names a different set of files and honouring only one would answer with a page you did not ask for. A cursor carries the whole position, so it can be held and replayed against any instance of your deployment; one this API never issued is refused with a `400` rather than served as an arbitrary page. A `before_id` page carries no `next_page`: its remaining results lie the other way.

**Scope filtering is refused.** A file here is never associated with a scope — no `scope` appears on its metadata — so `GET /v1/files?scope_id=...` answers `400` instead of returning a set that would ignore the filter. Select the files you want with `ids`, or list them without a filter.

**`downloadable` is always `true`.** Every file here is stored in S3 and readable through `/content`, so the field never carries the spec's `false` default for user-uploaded files.

**Expiry is set with `expires_in_seconds`, enforced on both surfaces.** Pass `expires_in_seconds` (3 600 – 7 776 000, one hour to ninety days) on upload to have the file expire automatically; omit it to keep the file until it is deleted. Expiry is enforced in code on every read — through this route, through the [OpenAI Files API](api_openai_files.md), and as a `file-id:` reference in a Messages request — regardless of which surface set it, since both read the same S3 object. A TTL beyond 30 days is enforced this way alone: it outlives the [S3 bucket lifecycle rule](operations_configuration_storage.md#s3-lifecycle) that otherwise backs up the code-level check, so its bytes can remain in storage past expiry until something reads the file again.

**`file-id:` is refused on upload.** `POST /v1/files` answers `400` for a `file-id:` input, because resolving it there would silently clone an existing file. It is accepted only on string-overloaded file fields — see [the URI scheme](#referencing-uploaded-files-via-the-file-id-uri-scheme).

**Filenames are kept, not policed.** Only the last component of the part's `filename` is kept, with anything up to the final `/` or `\` dropped: `reports/q3.pdf` is stored and listed as `q3.pdf`. Every other character is kept exactly as sent, up to 500 of them — punctuation a filesystem dislikes included, since the name is never used as a path. An absent or empty `filename` becomes `unnamed` plus the extension of the file's `mime_type`, or `unnamed` when the type implies none. Only a double quote (`"`) and control characters are refused, with a `400`.

### Errors

| HTTP | Cause                                                                               |
|------|-------------------------------------------------------------------------------------|
| 400  | A filename over 500 characters once its path is dropped, or one carrying a double quote or a control character |
| 400  | `expires_in_seconds` outside 3 600 – 7 776 000 (1 hour – 90 days)                   |
| 400  | `file-id:` URI passed to the upload endpoint (`POST /v1/files`)                     |
| 400  | Malformed ID after the `file-id:` prefix in a Messages content block                |
| 400  | More than 100 distinct `ids` on a listing request, or an `ids` entry that is not a file ID |
| 400  | `scope_id` on a listing request (`GET /v1/files`)                                   |
| 400  | A `page` cursor this API did not issue, or one sent alongside `ids`, `after_id` or `before_id` |
| 404  | File not found or already deleted                                                   |
| 529  | `AWS_S3_BUCKET` is not configured                                                   |

## Request headers

| Header                                 | Purpose                                     | Notes                                          |
|----------------------------------------|---------------------------------------------|------------------------------------------------|
| `x-api-key`                            | Gateway API key                             | Required, like every other route               |
| `anthropic-version`                    | Anthropic API version pin                   | Accepted and ignored                           |
| `anthropic-beta: files-api-2025-04-14` | Files API beta opt-in                       | Accepted and ignored                           |

The two Anthropic headers are sent by the official SDKs and cost nothing to send; the Files API works without them. The examples on this page include them only for parity with SDK-generated requests.

## Try it { #end-to-end-example }

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

## Next steps

Next: [Messages API](api_anthropic_messages.md) · [OpenAI Files API](api_openai_files.md) · [Storage configuration](operations_configuration_storage.md)
