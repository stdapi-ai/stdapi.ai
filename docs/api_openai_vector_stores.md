---
title: Vector Stores API - OpenAI-Compatible Managed Semantic Search
description: Index files into a managed vector store and search them by meaning with the OpenAI-compatible Vector Stores API. Asynchronous indexing, file batches, attribute filters, expiration policies and per-chunk scoring, backed by your own AWS account — or by an Amazon Bedrock knowledge base you already run.
keywords: Vector Stores API, OpenAI vector store, semantic search, managed retrieval, file indexing, chunking strategy, attribute filters, vector search AWS, RAG vector store, Amazon Bedrock knowledge base, knowledge base vector store, vs_kb
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

- :material-database-arrow-right: __Bring Your Own Knowledge Base__
  <br>Address an [Amazon Bedrock knowledge base](#knowledge-base-stores) you already run as a vector store, through the same endpoints.

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

## Listing Order

Every listing — stores, a store's files, a batch's files — is ordered by the
`created_at` it reports for each object, **most recent first**; `order=asc`
reverses it. Objects sharing a second are ordered by identifier, so the sequence
is stable from one page to the next.

| Object | `created_at` is                                                                 |
|--------|-----------------------------------------------------------------------------------|
| Store  | When the store was created.                                                       |
| File   | When the file was **attached to that store** — not when it was uploaded. Attaching the same file again moves it to the newest end. |

The `after` and `before` cursors are positions in that order: `after` returns the
objects that follow the named one, `before` the page that ends just before it,
both in the direction `order` asks for. Neither restarts the listing from the
top when the object it names has since been deleted.

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

Indexing is bounded server-wide: attaching many files at once never indexes
more than a couple at a time, so a large attach queues rather than being
refused. A poll always terminates: a file whose indexing was interrupted and
cannot be resumed settles as `failed` with `last_error.code="server_error"`,
and attaching it again indexes it.

### Durable Indexing { #durable-indexing }

Whether an interruption costs you anything depends on one deployment setting,
[`AWS_SQS_VECTOR_STORE_QUEUE_URL`](operations_configuration.md#aws-sqs-vector-store-queue-url):

| Setting | What happens when the server indexing a file is replaced |
|---|---|
| Unset (default) | The file settles as `failed`. Attach it again. |
| Set | Another server picks the work up and finishes it. The file stays `in_progress` a little longer, then settles as `completed`. |

Nothing about the API changes: the same fields, the same statuses, the same
polling. A file only takes longer to settle. Indexing stays at-least-once and
never bills twice — work that already completed is not redone.

Ask your administrator which of the two your deployment runs before designing a
client around it.

## Supported Files

Files must be **text**: plain text, Markdown, source code, CSV, JSON, XML,
YAML and anything else whose bytes decode as UTF-8 and whose content type is
not a known binary one — a text file uploaded as `application/octet-stream` or
`application/pdf` is refused on its content type, before its bytes are read.

A file that is not text settles as `status="failed"` with
`last_error.code="unsupported_file"`. The message names what **that** store
indexes, and — when another kind of store would take the file as it stands —
where to send it instead:

> This file type cannot be indexed by this vector store. It indexes text only.
> Provide the content as a text file. A knowledge base store indexes this file
> type as it stands.

Convert documents to text otherwise — [RAG
Pipelines](use_cases_rag.md#document-parsing) shows a document-conversion stage
that produces Markdown from PDF and office formats.

| `last_error.code` | Meaning                                                        |
|-------------------|----------------------------------------------------------------|
| `unsupported_file`| The file is not one this store indexes.                         |
| `invalid_file`    | The file is text but holds nothing to index, or is too large.   |
| `server_error`    | Indexing failed, or was interrupted; attach the file again.     |

A [knowledge base store](#knowledge-base-stores) indexes more than text — PDF and
office documents as they stand, and media on a fully managed one. Its own
refusals list the formats that store accepts.

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

!!! note "Scores on a knowledge base store"
    On a [`vs_kb_...` store](#knowledge-base-stores) the `score` is the relevance
    value the knowledge base measured, reported unchanged. It orders the results
    within one response and is **not** the `0`-to-`1` similarity above, so never
    compare it across stores or against a fixed threshold —
    `ranking_options.score_threshold` is refused on those stores for that reason.

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

## Knowledge Base Stores { #knowledge-base-stores }

A vector store can also be served by an
[Amazon Bedrock knowledge base](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html)
you already created. Allowlist it in
[`AWS_BEDROCK_KNOWLEDGE_BASE_IDS`](operations_configuration.md#aws-bedrock-knowledge-base-ids)
and it is addressed as the vector store `vs_kb_<knowledgeBaseId>` — the knowledge
base identifier is ten alphanumeric characters — on every `/v1/vector_stores`
endpoint, and returned by `GET /v1/vector_stores` next to the stores the server
owns.

**The knowledge base stays yours.** It is addressed, never created and never
deleted; the server searches it and manages the documents of its data source.
Its name, description, creation time and status are read from the knowledge base
itself.

!!! warning "Attaching a file needs a custom data source"
    A file attached through this API becomes an in-line document, which only a
    **custom** data source takes — on either generation. Point the allowlist
    entry at one, as `<knowledgeBaseId>/<dataSourceId>`; pointed at a data
    source that syncs its corpus from a bucket or another service, the store
    answers `400` to an attach and keeps serving search, listing and reading.
    A knowledge base can hold both kinds.

```python
store = client.vector_stores.retrieve("vs_kb_ABCDE12345")

uploaded = client.files.create(file=open("handbook.pdf", "rb"), purpose="assistants")
client.vector_stores.files.create(
    vector_store_id=store.id, file_id=uploaded.id, attributes={"department": "hr"}
)

for result in client.vector_stores.search(
    store.id, query="How much parental leave do I get?", max_num_results=5
):
    print(result.score, result.filename, result.content[0].text)
```

### What Works

| Request                                              | On a `vs_kb_...` store                                                                                     |
|------------------------------------------------------|--------------------------------------------------------------------------------------------------------------|
| `GET /v1/vector_stores`                              | Lists it alongside the stores the server owns.                                                               |
| `GET /v1/vector_stores/{id}`                         | Name, description, creation time and status, as the knowledge base reports them.                             |
| `POST /v1/vector_stores/{id}/files`                  | Attaches an uploaded file: it becomes a document of the knowledge base's custom data source, with its `attributes` kept searchable. |
| `GET /v1/vector_stores/{id}/files`                   | Lists the documents of the store's data source, including the ones put there outside this API.               |
| `GET /v1/vector_stores/{id}/files/{file_id}`         | Retrieves one document, including one a search returned from elsewhere in the knowledge base.                 |
| `DELETE /v1/vector_stores/{id}/files/{file_id}`      | Removes a document this API attached.                                                                        |
| `POST /v1/vector_stores/{id}/search`                 | Searches it, with `filters` and `max_num_results`.                                                           |

`filters` works in full: all eight comparison operators (`eq`, `ne`, `gt`,
`gte`, `lt`, `lte`, `in`, `nin`) and both combinators (`and`, `or`), over any
metadata key, with no schema to declare beforehand.

### What Is Refused

| Request                                                              | Answer                                                                                                              |
|------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|
| `POST /v1/vector_stores` creating one                                | Not creatable: the store is addressed, never created. No request field names a knowledge base.                        |
| `DELETE /v1/vector_stores/{id}`                                      | `400` — the store is managed outside the server.                                                                      |
| `POST /v1/vector_stores/{id}` (`name`, `metadata`, `expires_after`)  | `400` — they are read from the knowledge base.                                                                        |
| `chunking_strategy` on an attach                                     | `400` — the store chooses its own passage boundaries.                                                                 |
| `POST /v1/vector_stores/{id}/files/{file_id}` (attribute rewrite)    | `400` — attach the file again with the attributes it should carry.                                                    |
| `GET /v1/vector_stores/{id}/files/{file_id}/content`                 | `400` — the passages a file was indexed as cannot be listed. Download the file itself with the [Files API](api_openai_files.md). |
| `POST /v1/vector_stores/{id}/files` on a store whose data source syncs its corpus | `400` — that corpus is maintained where it comes from. Use a store this server owns, or ask for one that accepts uploads. |
| `DELETE /v1/vector_stores/{id}/files/{file_id}` on a document of the corpus | `400` — it was not attached here, and is removed where the corpus comes from.                                          |
| `POST /v1/vector_stores/{id}/file_batches` and the other batch routes | `400` — attach and follow files one at a time.                                                                        |
| `ranking_options.score_threshold`                                    | `400` — the store's relevance scores are not comparable between searches; use `max_num_results` instead.               |
| A `query` over the length limit                                      | `400` naming the limit. The query is never truncated.                                                                 |

`usage_bytes`, `file_counts`, and a file's `chunking_strategy` and `attributes`
are reported as unknown — zero, or absent — rather than invented: the corpus is
yours, and the server does not claim to know what it holds.

A document's `created_at` is the only time the knowledge base reports for it: the
instant it last ingested that document, which for a document ingested once is
when it was created. The document listing is ordered on that same value, so a
page never reports a time that contradicts its own order.

### Identifiers

A file you attach keeps its own `file-...` identifier. A document that was
already in the knowledge base — one of the corpus behind it — is reported under
an opaque `kbdoc_...` identifier, and the per-file routes accept it.

A search covers the whole knowledge base, not only the documents attached here,
so a result may name a document of a corpus this API never wrote to. Its
`kbdoc_...` identifier still reads back: `GET /v1/vector_stores/{id}/files/{file_id}`
answers for it, wherever in the knowledge base it lives. Removing it is the one
thing refused — that corpus is maintained where it comes from.

### Document Formats

Files are indexed as they stand, with no conversion step:

| Formats                                                                 | Indexed by                                     |
|---------------------------------------------------------------------------|--------------------------------------------------|
| `.txt`, `.md`, `.html`, `.csv`, `.doc`, `.docx`, `.xls`, `.xlsx`, `.pdf` | Every knowledge base                             |
| `.ppt`, `.pptx`, images, audio and video                                | A fully managed knowledge base, additionally     |
| Anything else whose bytes decode as text                                | Every knowledge base, indexed as text            |

One file is at most **5 MiB**.

A format outside that table is refused when the file is attached — `400`, naming
the formats **this** store takes — rather than accepted and then reported as
`failed`. A file that is attached and then fails is one of a format the store
does take, so it settles with `last_error.code="server_error"`.

### Two Generations, Two Differences

Both generations are supported: a knowledge base you provisioned the storage for,
and a fully managed one. Only two things differ from the client's point of view:

| Difference               | You provisioned the storage | Fully managed                              |
|--------------------------|-----------------------------|--------------------------------------------|
| Document formats indexed | The table above             | Also `.ppt`, `.pptx`, images, audio, video |
| `query` length           | 1,000 characters            | 10,000 characters                          |

### Cost

A knowledge base search costs more than a search on a store the server owns, and
a knowledge base backed by an always-on vector database bills whether it is
queried or not. This is information for choosing between the two, not a
recommendation.

What the [usage log](operations_cost_management.md#vector-stores) reports differs
per generation, because the retrieval runs inside the knowledge base rather than
through an embedding model of the server's:

| Generation                  | Search                                                         | Attaching a file                                     |
|-----------------------------|----------------------------------------------------------------|------------------------------------------------------|
| Fully managed               | One `search_units` unit per query, at the published flat rate  | Not reported: index storage is billed monthly per GB |
| You provisioned the storage | Not reported: no per-retrieval rate is published for it         | Not reported: billed by its own embedding model      |

Everything left unreported is on your AWS bill and readable from AWS Cost
Explorer. Full detail in [Cost Management](operations_cost_management.md#vector-stores).

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
| `400`  | Attributes above the total budget, a chunking strategy outside its bounds, a `gt`/`gte`/`lt`/`lte` filter given a non-numeric value, or a request a [knowledge base store](#knowledge-base-stores) does not accept. |
| `404`  | An identifier that names no store, attached file or batch. A `vs_kb_...` identifier that is not allowlisted answers exactly like any unknown store. |
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

Indexing survives a server being replaced only when
[`AWS_SQS_VECTOR_STORE_QUEUE_URL`](operations_configuration.md#aws-sqs-vector-store-queue-url)
names an Amazon SQS queue you created, with the
[durable indexing permissions](operations_iam_permissions.md#durable-vector-store-indexing)
on it — see [Durable indexing](#durable-indexing).

[Knowledge base stores](#knowledge-base-stores) need none of the above: they need
[`AWS_BEDROCK_KNOWLEDGE_BASE_IDS`](operations_configuration.md#aws-bedrock-knowledge-base-ids),
the [knowledge base permissions](operations_iam_permissions.md#knowledge-base-vector-stores),
and a knowledge base in the first
[`AWS_BEDROCK_REGIONS`](operations_configuration.md#aws-bedrock-regions) entry.
They bring their own storage and their own embedding model, and they are listed
and served even when no vector bucket is configured.

## See Also

- [Files API](api_openai_files.md) — uploading the files you attach.
- [RAG Pipelines](use_cases_rag.md) — using a vector store as the retrieval stage.
- [Embeddings API](api_openai_embeddings.md) — embedding text yourself instead.
- [Configuration](operations_configuration.md#vector-stores-optional) — the settings above.
- [Cost Management](operations_cost_management.md) — what indexing and searching cost.
