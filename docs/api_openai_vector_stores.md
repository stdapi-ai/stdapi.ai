---
title: Vector Stores API - OpenAI-Compatible Managed Semantic Search
description: Index files into a managed vector store and search them by meaning with the OpenAI-compatible Vector Stores API. Asynchronous indexing, file batches, attribute filters, expiration policies and per-chunk scoring, backed by your own AWS account.
keywords: Vector Stores API, OpenAI vector store, semantic search, managed retrieval, file indexing, chunking strategy, attribute filters, vector search AWS, RAG vector store
---

# Vector Stores API

Index your files once and search them by meaning. A vector store holds the
passages of the files attached to it; a search returns the passages closest to a
query, with the file they came from and a similarity score. There is no
embedding pipeline, chunker or vector database to run — attach a file and
search it.

## Why Choose the Vector Stores API?

<div class="grid cards" markdown>

- :material-magnify: __Search by Meaning__
  <br>A query finds the passages that answer it, not the ones sharing its words.

- :material-file-upload: __Attach and Forget__
  <br>Upload a file with the [Files API](api_openai_files.md), attach it, and it becomes searchable — indexing runs in the background.

- :material-filter-variant: __Attribute Filters__
  <br>Tag files with up to 16 attributes and restrict a search to the ones that match.

- :material-package-variant-closed: __File Batches__
  <br>Attach many files in one request and follow their progress with a single identifier.

- :material-timer-sand: __Expiration Policies__
  <br>Expire a store after a number of days without a search, so scratch stores do not accumulate.

- :material-shield-lock: __Your Own Account__
  <br>Documents, passages and vectors are stored in your AWS account, and never leave it.

</div>

## Available Endpoints

| Endpoint                                                             | Method   | What It Does                        | MCP Tool                                     |
|----------------------------------------------------------------------|----------|-------------------------------------|----------------------------------------------|
| `/v1/vector_stores`                                                  | `POST`   | Create a vector store               | `openai_vector_store_create`                 |
| `/v1/vector_stores`                                                  | `GET`    | List vector stores                  | `openai_vector_store_list`                   |
| `/v1/vector_stores/{vector_store_id}`                                | `GET`    | Retrieve a vector store             | `openai_vector_store_get`                    |
| `/v1/vector_stores/{vector_store_id}`                                | `POST`   | Update name, metadata or expiration | `openai_vector_store_update`                 |
| `/v1/vector_stores/{vector_store_id}`                                | `DELETE` | Delete a vector store               | `openai_vector_store_delete`                 |
| `/v1/vector_stores/{vector_store_id}/search`                         | `POST`   | Search the indexed passages         | `openai_vector_store_search`                 |
| `/v1/vector_stores/{vector_store_id}/files`                          | `POST`   | Attach a file                       | `openai_vector_store_file_create`            |
| `/v1/vector_stores/{vector_store_id}/files`                          | `GET`    | List the attached files             | `openai_vector_store_file_list`              |
| `/v1/vector_stores/{vector_store_id}/files/{file_id}`                | `GET`    | Retrieve an attached file           | `openai_vector_store_file_get`               |
| `/v1/vector_stores/{vector_store_id}/files/{file_id}`                | `POST`   | Replace a file's attributes         | `openai_vector_store_file_update`            |
| `/v1/vector_stores/{vector_store_id}/files/{file_id}`                | `DELETE` | Detach a file                       | `openai_vector_store_file_delete`            |
| `/v1/vector_stores/{vector_store_id}/files/{file_id}/content`        | `GET`    | Read a file's indexed passages      | `openai_vector_store_file_content`           |
| `/v1/vector_stores/{vector_store_id}/file_batches`                   | `POST`   | Attach several files at once        | `openai_vector_store_file_batch_create`      |
| `/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}`        | `GET`    | Retrieve a file batch               | `openai_vector_store_file_batch_get`         |
| `/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel` | `POST`   | Cancel a file batch                 | `openai_vector_store_file_batch_cancel`      |
| `/v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files`  | `GET`    | List a file batch's files           | `openai_vector_store_file_batch_file_list`   |

## Quick Start

```python
import time

from openai import OpenAI

client = OpenAI(base_url="https://your-gateway/v1", api_key="YOUR_API_KEY")

uploaded = client.files.create(file=open("handbook.txt", "rb"), purpose="assistants")
store = client.vector_stores.create(name="handbook", file_ids=[uploaded.id])

# Indexing is asynchronous: wait until the store reports it finished.
deadline = time.monotonic() + 300
while client.vector_stores.retrieve(store.id).status == "in_progress":
    assert time.monotonic() < deadline, "indexing did not finish"
    time.sleep(2)

for result in client.vector_stores.search(
    store.id, query="How much parental leave do I get?"
):
    print(result.score, result.filename, result.content[0].text)
```

## Indexing Is Asynchronous

Attaching a file returns immediately with `status="in_progress"`. Poll the file,
or the store, until it settles:

| Object | Field                    | Settled when                                                       |
|--------|--------------------------|--------------------------------------------------------------------|
| Store  | `status`                 | `completed` — no attached file is still being indexed.              |
| Store  | `file_counts`            | `in_progress` reaches `0`; the other counters sum to `total`.       |
| File   | `status`                 | `completed`, `failed` or `cancelled`.                               |
| Batch  | `status` / `file_counts` | Same, for the files of that batch only.                             |

The counters summarise the files, so they can trail a file that has just
settled — a store still reporting `in_progress` for a file already `completed`
converges within a moment. Poll the file itself when you need the earliest
possible answer.

## Supported Files

Files must be **text**: plain text, Markdown, source code, CSV, JSON, XML,
YAML and anything else whose bytes decode as UTF-8 and whose content type is
not a known binary one — a text file uploaded as `application/octet-stream` or
`application/pdf` is refused on its content type, before its bytes are read.

A file that is not text settles as `status="failed"` with
`last_error.code="unsupported_file"`. Convert documents to text before
uploading them — [RAG Pipelines](use_cases_rag.md#document-parsing) shows a
document-conversion stage that produces Markdown from PDF and office formats.

| `last_error.code` | Meaning                                                        |
|-------------------|----------------------------------------------------------------|
| `unsupported_file`| The file's bytes are not text.                                  |
| `invalid_file`    | The file is text but holds nothing to index, or is too large.   |
| `server_error`    | Indexing failed; detach the file and attach it again.           |

## Chunking

A file is split into overlapping passages before it is indexed. Send a
`chunking_strategy` to choose the split:

```json
{
  "type": "static",
  "static": {"max_chunk_size_tokens": 800, "chunk_overlap_tokens": 400}
}
```

| Field                   | Range          | Default |
|-------------------------|----------------|---------|
| `max_chunk_size_tokens` | 100 to 4096    | 800     |
| `chunk_overlap_tokens`  | 0 to half the chunk size | 400 |

`"type": "auto"` inherits the store's own strategy — the one given when it was
created — which also applies to every file attached without a strategy at all.
A store created without one falls back to the server default, which comes from
[`VECTOR_STORE_CHUNK_SIZE_TOKENS`](operations_configuration.md#vector-store-chunk-size-tokens).

!!! note "Chunk sizes are approximate"
    The chunk size is applied as a text-length budget, and a cut is moved back
    to the nearest line or word boundary, so a passage never ends mid-word. The
    resulting passages therefore do not match another provider's split
    character for character. A passage is additionally capped by what the
    configured embedding model accepts in one input, so a very large
    `max_chunk_size_tokens` may produce shorter passages than asked for.

`GET /v1/vector_stores/{id}/files/{file_id}/content` returns the passages a file
was indexed as, in document order — the fastest way to see what a chunking
strategy actually produced.

## Searching

```python
page = client.vector_stores.search(
    store.id,
    query="parental leave",
    max_num_results=5,
    filters={"key": "department", "type": "eq", "value": "hr"},
    ranking_options={"score_threshold": 0.4},
)
```

| Parameter                        | Default | Notes                                                          |
|----------------------------------|---------|-----------------------------------------------------------------|
| `query`                          | —       | A string, or an array of strings searched together.             |
| `max_num_results`                | `10`    | 1 to 50.                                                        |
| `filters`                        | —       | A comparison or compound filter over the files' `attributes`.   |
| `ranking_options.score_threshold`| —       | Drops results scoring below it.                                 |
| `ranking_options.ranker`         | —       | Accepted and ignored: results are always ranked by similarity.  |
| `rewrite_query`                  | `false` | Accepted and ignored: the query is searched as written.         |

Each result carries `file_id`, `filename`, the file's `attributes`, the matching
`content`, and a `score` between `0` and `1` where `1` is an exact match. Results
are ordered best first, and the page is complete — search is never paginated.

### Filters

| `type`                            | Matches                                    |
|-----------------------------------|--------------------------------------------|
| `eq`, `ne`                        | Equal / not equal — string, number or boolean |
| `gt`, `gte`, `lt`, `lte`          | Numeric comparison — `value` must be a number |
| `in`, `nin`                       | Value in / not in an array                 |
| `and`, `or`                       | Combine filters, nestable                  |

Filters apply to the `attributes` of the attached file, never to its content. A
filter that matches nothing returns an empty page rather than an error, while an
ordering operator given a non-numeric `value` is rejected with `400`: store a
date as a number (for example `20260115`) to compare ranges of it.

## Attributes

Up to 16 key-value pairs per attached file, with string, number or boolean
values. They are returned on the file and on every search result, and are what
`filters` matches against.

| Limit                              | Value        |
|------------------------------------|--------------|
| Key-value pairs                    | 16           |
| Key length                         | 64           |
| String value length                | 512          |
| Total size of all attributes       | 2048 bytes   |

Attributes larger than the total budget are rejected with `400`, naming the
limit. `POST /v1/vector_stores/{id}/files/{file_id}` **replaces** the whole set;
the new values apply to later searches once the file is `completed`. Replacing
the attributes of a file that is still `in_progress` is accepted, but the
indexing in flight writes the attributes it started with — wait for the file to
settle, then replace them.

## Expiration

```json
{"expires_after": {"anchor": "last_active_at", "days": 7}}
```

`expires_at` is `last_active_at` plus `days`. **Only a search refreshes
`last_active_at`** — attaching, reading or updating a store does not — so a
store that is written to but never searched still expires. Once past its
expiration a store reads back with `status="expired"` and returns no search
result; its indexed content is released and a search never brings it back. Send
`"expires_after": null` on an update to remove the policy.

## Limits

| Limit                             | Value                                      |
|-----------------------------------|--------------------------------------------|
| Files per file batch              | 2000                                       |
| Queries per search                | 16                                         |
| Results per search                | 50                                         |
| Store `metadata`                  | 16 pairs, 64-character keys, 512-character values |
| File size                         | 100 MiB, or [`MAX_INPUT_FILE_SIZE`](operations_configuration.md#max-input-file-size) when it is lower |

A file above the size limit is not rejected at request time: it settles as
`status="failed"` with `last_error.code="invalid_file"`, like any other file
that cannot be indexed.

## Errors

| Status | When                                                                             |
|--------|-----------------------------------------------------------------------------------|
| `400`  | Attributes above the total budget, a chunking strategy outside its bounds, or a `gt`/`gte`/`lt`/`lte` filter given a non-numeric value. |
| `404`  | An identifier that names no store, attached file or batch.                         |
| `409`  | The store is being updated concurrently by another request; retry it.              |
| `503`  | The deployment has no vector storage configured.                                   |

## Prerequisites

Vector stores are stored in your own AWS account, in an
[Amazon S3 vector bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html)
you create yourself. Set
[`AWS_S3_VECTORS_BUCKET`](operations_configuration.md#aws-s3-vectors-bucket) to
its name and [`AWS_S3_VECTORS_REGION`](operations_configuration.md#aws-s3-vectors-region)
to its Region; the endpoints answer `503` until both a vector bucket and
[`AWS_S3_BUCKET`](operations_configuration.md#aws-s3-bucket) are configured. The
gateway's IAM role needs the
[Vector Stores permissions](operations_iam_permissions.md#vector-stores-optional).

The model that turns text into vectors is
[`VECTOR_STORE_EMBEDDING_MODEL`](operations_configuration.md#vector-store-embedding-model).
It is recorded on each store when the store is created, so changing the setting
only affects stores created afterwards — existing stores keep answering with the
model they were built with.

## See Also

- [Files API](api_openai_files.md) — uploading the files you attach.
- [RAG Pipelines](use_cases_rag.md) — using a vector store as the retrieval stage.
- [Embeddings API](api_openai_embeddings.md) — embedding text yourself instead.
- [Configuration](operations_configuration.md#vector-stores-optional) — the settings above.
- [Cost Management](operations_cost_management.md) — what indexing and searching cost.
