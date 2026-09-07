---
title: "Configuration - Storage"
description: "Configure the S3 buckets, prefixes and lifecycle rules stdapi.ai uses for files, images, videos, batches and vector stores, plus the DynamoDB table and the transcription buckets."
keywords: "S3 bucket configuration, S3 prefixes, S3 lifecycle, vector stores, S3 vector bucket, DynamoDB table, transcription bucket, regional buckets"
---

# :material-database: Storage

Where stdapi.ai puts files, images, videos, batch data and vector stores, and which buckets it is allowed to read. Part of the [Configuration Guide](operations_configuration.md).

## :material-format-list-bulleted: Settings Summary

### :material-database: AWS Storage { #summary-aws-storage }

| Variable                                                | Default         | Description                                                                                          |
|---------------------------------------------------------|-----------------|------------------------------------------------------------------------------------------------------|
| [`AWS_S3_ACCELERATE`](#aws-s3-accelerate)               | `false`         | Enable S3 Transfer Acceleration for faster global downloads via CloudFront edge locations            |
| [`AWS_S3_REGIONAL_BUCKETS`](#aws-s3-regional-buckets)   | `{}`            | Region-specific S3 buckets for Bedrock async/batch inference operations                              |
| [`AWS_S3_ACCEPTED_BUCKETS`](#aws-s3-accepted-buckets)   | `{}`            | External S3 buckets with read access, mapped to their region for S3 URI conversion and routing       |
| [`AWS_S3_TMP_PREFIX`](#aws-s3-tmp-prefix)               | `tmp/`          | S3 prefix for temporary files used for jobs; configure lifecycle policies on this prefix             |
| [`AWS_S3_FILES_PREFIX`](#aws-s3-files-prefix)           | `files/`        | S3 prefix for Files API objects; configure S3 lifecycle policies on this prefix                     |
| [`AWS_S3_VIDEOS_PREFIX`](#aws-s3-videos-prefix)         | `videos/`       | S3 prefix for generated videos (Videos API); persists until deleted through the API                  |
| [`AWS_S3_VIDEOS_EXPIRES_AFTER`](#aws-s3-videos-expires-after) | None      | Retention period in seconds for generated videos; sets `Video.expires_at` and blocks expired downloads |
| [`AWS_S3_BATCHES_PREFIX`](#aws-s3-batches-prefix)       | `batches/`      | S3 prefix for Batch API data (requests, results, batch records); configure lifecycle policies on it   |
| [`AWS_S3_VECTORS_BUCKET`](#aws-s3-vectors-bucket)       | None            | Amazon S3 vector bucket backing the Vector Stores API; unset disables it                             |
| [`AWS_S3_VECTORS_REGION`](#aws-s3-vectors-region)       | First Bedrock region | Region holding `AWS_S3_VECTORS_BUCKET`; the vector bucket has no failover                       |
| [`AWS_S3_VECTOR_STORES_PREFIX`](#aws-s3-vector-stores-prefix) | `vector_stores/` | S3 prefix for the Vector Stores API records (stores, attached files, batches)                  |
| [`VECTOR_STORE_EMBEDDING_MODEL`](#vector-store-embedding-model) | `amazon.titan-embed-text-v2:0` | Model embedding the indexed files and the search queries                    |
| [`VECTOR_STORE_CHUNK_SIZE_TOKENS`](#vector-store-chunk-size-tokens) | `800`   | Default chunk size for files indexed without an explicit `chunking_strategy`                         |
| [`VECTOR_STORE_CHUNK_OVERLAP_TOKENS`](#vector-store-chunk-overlap-tokens) | `400` | Default chunk overlap; must not exceed half the chunk size                                 |
| [`AWS_SQS_VECTOR_STORE_QUEUE_URL`](#aws-sqs-vector-store-queue-url) | None    | Amazon SQS queue making vector store indexing survive the server running it; unset keeps it in-process |
| [`AWS_BEDROCK_KNOWLEDGE_BASE_IDS`](#aws-bedrock-knowledge-base-ids) | `[]`    | Allowlist of Amazon Bedrock knowledge bases addressed as `vs_kb_...` vector stores; empty disables it |
| [`AWS_DYNAMODB_TABLE`](#aws-dynamodb-table)             | None            | Amazon DynamoDB table holding the records a deployment's instances share; unset disables every feature needing it |
| [`AWS_DYNAMODB_REGION`](#aws-dynamodb-region)           | First Bedrock region | Region holding `AWS_DYNAMODB_TABLE`; the table has no failover                                  |
| [`AWS_TRANSCRIBE_S3_BUCKET`](#aws-transcribe-s3-bucket) | `AWS_S3_BUCKET` | S3 bucket for temporary audio transcription files; must be in same region as `AWS_TRANSCRIBE_REGION` |
| [`AWS_TRANSCRIBE_OUTPUT_ENCRYPTION_KEY_ARN`](#aws-transcribe-output-encryption-key-arn) | None | AWS KMS key encrypting the transcription output objects; unset keeps the bucket's own encryption |
| [`AWS_TRANSCRIBE_STREAM_LANGUAGES`](#aws-transcribe-stream-languages) | `[]`  | Languages a streamed transcription picks between when the request names none |

## :material-bucket-outline: Storage Configuration { #storage-configuration }

The general purpose S3 bucket the gateway reads and writes, how it is reached, and the prefix each kind of object is stored under.

#### `AWS_S3_BUCKET` { #aws-s3-bucket }

:octicons-package-24: **Purpose**
:   Primary S3 bucket for storing generated files (images, audio, documents) and temporary data during processing

:octicons-gear-24: **Default**
:   None (must be configured for file operations)

:octicons-check-circle-24: **Best Practice**
:   The bucket must be in the first region specified in `AWS_BEDROCK_REGIONS` (your primary region where the server should be hosted) to avoid cross-region data transfer costs and reduce latency

```bash
export AWS_S3_BUCKET=my-llm-storage-us-east-1
```

!!! tip "Presigned URLs"
    Files are served via presigned URLs for secure, time-limited access. Presigned URLs expire after 1 hour.

!!! info "Terraform Module"
    When using the Terraform module, the main S3 bucket is created automatically — no manual configuration required.

!!! warning "Startup Warning"
    If not set, a warning is logged at startup and features that require file storage (image generation, audio output, document processing) will be unavailable.

#### `AWS_S3_ACCELERATE` { #aws-s3-accelerate }

:octicons-package-24: **Purpose**
:   Enable S3 Transfer Acceleration for presigned URLs to improve download performance for large files

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false`

:octicons-check-circle-24: **Best Practice**
:   Enable when serving large files (high-resolution images, audio) to geographically distributed users

```bash
export AWS_S3_ACCELERATE=true
```

!!! info "What is S3 Transfer Acceleration?"
    S3 Transfer Acceleration uses Amazon CloudFront's globally distributed edge locations to accelerate uploads and downloads to S3 buckets. When enabled, data is routed to the nearest edge location and then transferred to S3 over Amazon's optimized network paths.

    **Performance Benefits:**

    - :material-speedometer: **Faster downloads** for users far from your bucket's region
    - :material-earth: **Global reach** via CloudFront edge locations
    - :material-upload-network: **Optimized routing** over Amazon's private backbone network
    - :material-chart-line: **Consistent performance** regardless of user location

    Typical speed improvements: 50-500% faster for users located far from the bucket region.

!!! warning "Requirements"
    1. **Enable Transfer Acceleration** on your S3 bucket before setting this option:
       ```bash
       aws s3api put-bucket-accelerate-configuration \
         --bucket my-stdapi-bucket \
         --accelerate-configuration Status=Enabled
       ```
    2. **Additional costs**: Transfer Acceleration incurs extra data transfer fees. See [Amazon S3 Transfer Acceleration pricing](https://aws.amazon.com/s3/pricing/)

!!! tip "When to Enable"
    Consider enabling S3 Transfer Acceleration when:

    - :material-image: Serving generated images via [Images API](api_openai_images_generations.md)
    - :material-earth-arrow-right: Users are geographically distributed across multiple continents
    - :material-file-image: Generating high-resolution images that are large in file size
    - :material-speedometer: Download performance is critical to user experience

    For small images or users close to your bucket region, the performance benefit may not justify the additional cost.

!!! info "Current Usage"
    Presigned URLs with Transfer Acceleration are currently only used for the [Images API](api_openai_images_generations.md) when returning generated images as URLs.

#### `AWS_S3_TMP_PREFIX` { #aws-s3-tmp-prefix }

:octicons-package-24: **Purpose**
:   S3 prefix (folder path) for temporary files used during job processing

:octicons-gear-24: **Default**
:   `tmp/`

:octicons-check-circle-24: **Best Practice**
:   Configure S3 lifecycle policies to automatically delete objects under this prefix after 1 day

```bash
export AWS_S3_TMP_PREFIX=tmp/
```

!!! info "What is an S3 Prefix?"
    An S3 prefix is essentially a folder path within your S3 bucket. When you set `AWS_S3_TMP_PREFIX=tmp/`, all temporary files are stored under the `tmp/` folder structure in your bucket.

    **Example file paths:**

    - With prefix `tmp/`: `s3://my-bucket/tmp/request-id-123/output.json`
    - With prefix `temporary/`: `s3://my-bucket/temporary/request-id-123/output.json`
    - With empty prefix ``: `s3://my-bucket/request-id-123/output.json` (not recommended)

!!! tip "Why Use a Prefix?"
    Using a dedicated prefix for temporary files provides several benefits:

    - :material-auto-fix: **Easy Lifecycle Management** - Apply S3 lifecycle policies to automatically delete only temporary files
    - :material-file-tree: **Better Organization** - Keep temporary files separate from permanent storage
    - :material-shield-check: **Security** - Apply different IAM policies or bucket policies to the prefix
    - :material-cash: **Cost Control** - Easily identify and monitor temporary storage costs

!!! warning "Trailing Slash"
    Always include a trailing slash (`/`) in your prefix to create a proper folder structure. Without it, files will be stored with the prefix as part of the filename rather than in a folder.

    - ✅ Correct: `tmp/` → Files stored as `tmp/file.json`
    - ❌ Incorrect: `tmp` → Files stored as `tmpfile.json`

**Custom prefix examples:**

```bash
# Production environment
export AWS_S3_TMP_PREFIX=prod/tmp/

# Staging environment
export AWS_S3_TMP_PREFIX=staging/tmp/

# Organize by date (requires manual updates)
export AWS_S3_TMP_PREFIX=tmp/2025/01/

# No prefix (store at bucket root - not recommended)
export AWS_S3_TMP_PREFIX=
```

#### `AWS_S3_FILES_PREFIX` { #aws-s3-files-prefix }

:octicons-package-24: **Purpose**
:   S3 prefix (folder path) for Files API objects (OpenAI and Anthropic `/v1/files` endpoints)

:octicons-gear-24: **Default**
:   `files/`

:octicons-check-circle-24: **Best Practice**
:   Configure an `AbortIncompleteMultipartUpload` S3 lifecycle rule on this prefix to clean up abandoned upload parts, and apply Intelligent-Tiering for cost optimisation

```bash
export AWS_S3_FILES_PREFIX=files/
```

!!! info "S3 Prefix Format"
    Prefix semantics (folder-style paths, trailing-slash requirement) are explained under [`AWS_S3_TMP_PREFIX`](#aws-s3-tmp-prefix) and apply here identically.

**Custom prefix examples:**

```bash
# Production environment
export AWS_S3_FILES_PREFIX=prod/files/

# Staging environment
export AWS_S3_FILES_PREFIX=staging/files/

# No prefix (store at bucket root - not recommended)
export AWS_S3_FILES_PREFIX=
```

#### `AWS_S3_VIDEOS_PREFIX` { #aws-s3-videos-prefix }

:octicons-package-24: **Purpose**
:   S3 prefix (folder path) for videos generated through the [Videos API](api_openai_videos.md)

:octicons-gear-24: **Default**
:   `videos/`

:octicons-alert-24: **Requirement**
:   Must be non-empty, use only S3-safe characters (alphanumerics plus `! _ . * ' ( ) -` per path segment), and end with a trailing `/` — an empty value would widen the ownership check that scopes listing/retrieval to the whole bucket

:octicons-check-circle-24: **Best Practice**
:   Generated videos persist until deleted through the API — configure an S3 lifecycle rule on this prefix to cap storage costs

```bash
export AWS_S3_VIDEOS_PREFIX=videos/
```

Amazon Bedrock writes each video generation job's output (MP4 and manifest) under this prefix, in a folder named after the job. Because Amazon Bedrock requires the output bucket to be in the same region as the invocation, videos are stored in the [`AWS_S3_REGIONAL_BUCKETS`](#aws-s3-regional-buckets) bucket of the region that served the job.

#### `AWS_S3_VIDEOS_EXPIRES_AFTER` { #aws-s3-videos-expires-after }

:octicons-package-24: **Purpose**
:   Retention period in seconds (minimum `3600`) for videos generated through the [Videos API](api_openai_videos.md)

:octicons-gear-24: **Default**
:   Unset — videos never expire and persist until deleted through the API

:octicons-check-circle-24: **Best Practice**
:   Pair with an S3 Lifecycle expiration rule on [`AWS_S3_VIDEOS_PREFIX`](#aws-s3-videos-prefix) covering the same duration (rounded up to whole days) so the objects are actually deleted

```bash
# Expire generated videos after 24 hours
export AWS_S3_VIDEOS_EXPIRES_AFTER=86400
```

When set, the `Video` object reports `expires_at` (job completion time plus this value) and downloading expired video content returns a 404. The server enforces expiry at the API level only; the paired S3 Lifecycle rule performs the physical cleanup.

#### `AWS_S3_BATCHES_PREFIX` { #aws-s3-batches-prefix }

:octicons-package-24: **Purpose**
:   S3 prefix (folder path) for the data of the [Batch API](api_openai_batches.md) and the [Message Batches API](api_anthropic_batches.md) — the submitted requests, the results, and the batch records themselves

:octicons-gear-24: **Default**
:   `batches/`

:octicons-alert-24: **Requirement**
:   Must be non-empty, use only S3-safe characters (alphanumerics plus `! _ . * ' ( ) -` per path segment), and end with a trailing `/` — batches are addressed by prefix, so an empty value would make every stray object at the bucket root a candidate

:octicons-check-circle-24: **Best Practice**
:   Batch data persists until the batch is deleted — configure an S3 lifecycle rule on this prefix to cap storage costs, and grant [`AWS_BEDROCK_BATCH_ROLE_ARN`](operations_configuration_bedrock.md#aws-bedrock-batch-role-arn) read and write access under it

```bash
export AWS_S3_BATCHES_PREFIX=batches/
```

Each batch stores its data under a folder of its own below this prefix. Because the output bucket must be in the same region as the model that serves the batch, that data is stored in the [`AWS_S3_REGIONAL_BUCKETS`](#aws-s3-regional-buckets) bucket of the region that served it.

## :material-database-search: Vector Stores { #vector-stores-optional }

The [Vector Stores API](api_openai_vector_stores.md) needs an [Amazon S3 vector bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html) — a resource type of its own, created separately from a general purpose bucket — plus the general purpose bucket in [`AWS_S3_BUCKET`](#aws-s3-bucket), which holds the stores' records. The vector store endpoints answer `503` until both are configured, or until [`AWS_BEDROCK_KNOWLEDGE_BASE_IDS`](#aws-bedrock-knowledge-base-ids) names a knowledge base to serve instead.

#### `AWS_S3_VECTORS_BUCKET` { #aws-s3-vectors-bucket }

:octicons-package-24: **Purpose**
:   Name of the Amazon S3 vector bucket that holds the indexed content of every vector store

:octicons-gear-24: **Default**
:   None — the [Vector Stores API](api_openai_vector_stores.md) is disabled

:octicons-alert-24: **Requirement**
:   Requires [`AWS_S3_BUCKET`](#aws-s3-bucket), which holds the vector store records; startup fails if only the vector bucket is set. The gateway's role needs the [Vector Stores permissions](operations_iam_permissions.md#vector-stores-optional) on it

```bash
export AWS_S3_VECTORS_BUCKET=my-llm-vectors-us-east-1
```

Create the bucket yourself, then let the gateway create and delete the indexes inside it — one per vector store, removed when the store is deleted or expires.

#### `AWS_S3_VECTORS_REGION` { #aws-s3-vectors-region }

:octicons-package-24: **Purpose**
:   AWS region holding [`AWS_S3_VECTORS_BUCKET`](#aws-s3-vectors-bucket)

:octicons-gear-24: **Default**
:   The first [`AWS_BEDROCK_REGIONS`](operations_configuration_aws.md#aws-bedrock-regions) entry

:octicons-alert-24: **Requirement**
:   Must be the bucket's own region. A vector bucket is a regional resource whose content is only reachable there, so this setting has no failover: if the region is unreachable, so is the [Vector Stores API](api_openai_vector_stores.md)

```bash
export AWS_S3_VECTORS_REGION=us-east-1
```

#### `AWS_S3_VECTOR_STORES_PREFIX` { #aws-s3-vector-stores-prefix }

:octicons-package-24: **Purpose**
:   S3 prefix (folder path) in [`AWS_S3_BUCKET`](#aws-s3-bucket) for the [Vector Stores API](api_openai_vector_stores.md) records — the stores, their attached files and their file batches

:octicons-gear-24: **Default**
:   `vector_stores/`

:octicons-check-circle-24: **Best Practice**
:   Keep it distinct from the other prefixes so a lifecycle rule written for one never reaches the records of another; these records are the stores themselves, so no expiration rule belongs on this prefix

```bash
export AWS_S3_VECTOR_STORES_PREFIX=vector_stores/
```

#### `VECTOR_STORE_EMBEDDING_MODEL` { #vector-store-embedding-model }

:octicons-package-24: **Purpose**
:   Model that turns the indexed files and the search queries into vectors

:octicons-gear-24: **Default**
:   `amazon.titan-embed-text-v2:0`

:octicons-alert-24: **Requirement**
:   Must be an embedding model available in your configured regions — see [Embeddings API](api_openai_embeddings.md)

```bash
export VECTOR_STORE_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
```

Each vector store records the model it was created with and keeps using it, so changing this setting only affects stores created afterwards. Existing stores keep answering exactly as before.

#### `VECTOR_STORE_CHUNK_SIZE_TOKENS` { #vector-store-chunk-size-tokens }

:octicons-package-24: **Purpose**
:   Default chunk size, in tokens, for files indexed without an explicit `chunking_strategy`

:octicons-gear-24: **Default**
:   `800` — the same default the upstream API applies

:octicons-alert-24: **Requirement**
:   Between `100` and `4096`

```bash
export VECTOR_STORE_CHUNK_SIZE_TOKENS=800
```

A request's own `chunking_strategy` always wins, and a store created with one applies it to every file later attached without one. Chunk sizes are approximate — see [Chunking](api_openai_vector_stores.md#chunking).

#### `VECTOR_STORE_CHUNK_OVERLAP_TOKENS` { #vector-store-chunk-overlap-tokens }

:octicons-package-24: **Purpose**
:   Default number of tokens shared between consecutive chunks, for files indexed without an explicit `chunking_strategy`

:octicons-gear-24: **Default**
:   `400` — the same default the upstream API applies

:octicons-alert-24: **Requirement**
:   Must not exceed half of [`VECTOR_STORE_CHUNK_SIZE_TOKENS`](#vector-store-chunk-size-tokens); startup fails otherwise

```bash
export VECTOR_STORE_CHUNK_OVERLAP_TOKENS=400
```

More overlap keeps a sentence split across two chunks findable from either one, at the cost of more chunks to embed and store.

#### `AWS_SQS_VECTOR_STORE_QUEUE_URL` { #aws-sqs-vector-store-queue-url }

:octicons-package-24: **Purpose**
:   URL of the [Amazon SQS](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/welcome.html) standard queue that carries vector store indexing work, so a file keeps being indexed when the server that accepted it stops

:octicons-gear-24: **Default**
:   None — indexing runs in the server that accepted the request, and a file being indexed when that server stops is reported as `failed`

:octicons-alert-24: **Requirement**
:   Requires [`AWS_S3_VECTORS_BUCKET`](#aws-s3-vectors-bucket). Must be a standard queue (not FIFO), with a [dead-letter queue](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/sqs-dead-letter-queues.html), and the gateway's role needs the [Durable vector store indexing permissions](operations_iam_permissions.md#durable-vector-store-indexing) on it

```bash
export AWS_SQS_VECTOR_STORE_QUEUE_URL=https://sqs.us-east-1.amazonaws.com/123456789012/stdapi-ai-indexing
```

Attaching a file records the work on the queue before answering, and every server reads from that queue: whichever one is still running finishes it. The region comes from the URL, so there is nothing else to configure. Create the queue yourself, in the same account.

The queue only ever carries identifiers — which store, which files, which batch — never file content, so no indexed data is stored a second time.

See [Durable indexing](api_openai_vector_stores.md#durable-indexing) for what a client observes, and [Resilience](operations_resilience.md#vector-store-indexing) for how it behaves during a deployment.

!!! warning "Give the queue a dead-letter queue"
    A file the gateway cannot index is retried a few times and then reported as `failed`. Without a dead-letter queue its message is dropped at that point; with one it is kept, so you can see what was refused. The gateway reads the queue's own redrive policy at startup and reports a queue that has none as a startup warning.

#### `AWS_BEDROCK_KNOWLEDGE_BASE_IDS` { #aws-bedrock-knowledge-base-ids }

:octicons-package-24: **Purpose**
:   Comma-separated allowlist of [Amazon Bedrock knowledge bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html) served through the [Vector Stores API](api_openai_vector_stores.md). Each allowlisted knowledge base is addressed as the vector store `vs_kb_<knowledgeBaseId>` on every `/v1/vector_stores` endpoint, and is returned by `GET /v1/vector_stores` next to the stores the server owns

:octicons-gear-24: **Default**
:   Empty — no knowledge base is addressable, and a `vs_kb_...` identifier is answered exactly as an unknown vector store is, so the allowlist cannot be probed

:octicons-alert-24: **Requirement**
:   Each knowledge base must already exist, in the first [`AWS_BEDROCK_REGIONS`](operations_configuration_aws.md#aws-bedrock-regions) entry, be a [Bedrock managed](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-build-managed.html) (`MANAGED`) or [customer-managed](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-build.html) (`VECTOR`) knowledge base, and the gateway's role needs the [Knowledge Base Vector Stores permissions](operations_iam_permissions.md#knowledge-base-vector-stores) on it

```bash
export AWS_BEDROCK_KNOWLEDGE_BASE_IDS=ABCDE12345,FGHIJ67890/KLMNO13579
```

Write each entry as `<knowledgeBaseId>`, or as `<knowledgeBaseId>/<dataSourceId>` when the knowledge base has more than one data source; with a single data source the server resolves it itself.

Both kinds of document knowledge base are served, and a store behaves the same on either. A knowledge base [connected to a structured data store](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base-build-structured.html) (`SQL`) or backed by an Amazon Kendra GenAI index (`KENDRA`) is not a vector store and must not be allowlisted.

The knowledge base itself always stays yours: the server never creates one and never deletes one. It searches it, and manages the documents of its data source. Name, description, creation time and status are read from the knowledge base, and the requests that would change them are refused — see [Knowledge Base Stores](api_openai_vector_stores.md#knowledge-base-stores).

This setting is independent of [`AWS_S3_VECTORS_BUCKET`](#aws-s3-vectors-bucket): a deployment that sets only this one serves its allowlisted knowledge bases and creates no store of its own.

!!! info "What a knowledge base costs"
    A knowledge base search costs more than a search on a store the server owns, and a knowledge base backed by an always-on vector database bills whether it is queried or not. Both backends are offered so the choice is yours — see [Cost Management](operations_cost_management.md).

## :material-table: Shared DynamoDB Table

The table the instances of one deployment share their records through, and the region it is read from.

#### `AWS_DYNAMODB_TABLE` { #aws-dynamodb-table }

:octicons-package-24: **Purpose**
:   Name of the [Amazon DynamoDB](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/Introduction.html) table holding the records the instances of a deployment share

:octicons-gear-24: **Default**
:   None — every feature that needs the table is disabled, no table is opened, and nothing is billed

:octicons-alert-24: **Requirement**
:   The table must already exist, with `pk` (String) as its **partition key** and `sk` (String) as its **sort key**, on-demand capacity, and [time to live](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/TTL.html) enabled on the `expires_at` attribute. The gateway's role needs the [Shared Table permissions](operations_iam_permissions.md#shared-table) on it

```bash
export AWS_DYNAMODB_TABLE=stdapi-ai
```

The table is yours to create: the gateway reads and writes items, and never creates, deletes or reconfigures a table. The [official Terraform module](https://github.com/stdapi-ai/terraform-aws-stdapi-ai) creates one for you.

!!! warning "The sort key cannot be added later"
    Amazon DynamoDB fixes a table's key schema when the table is created, so a table made with `pk` alone has to be recreated rather than altered. The gateway checks the key schema at startup and reports a mismatch as a startup warning.

!!! info "What the table costs"
    On-demand capacity has no idle charge, the records are kilobytes, and expiring items are deleted without consuming write throughput — so a table nothing writes to bills essentially nothing. See [Amazon DynamoDB pricing](https://aws.amazon.com/dynamodb/pricing/on-demand/).

#### `AWS_DYNAMODB_REGION` { #aws-dynamodb-region }

:octicons-package-24: **Purpose**
:   AWS region holding [`AWS_DYNAMODB_TABLE`](#aws-dynamodb-table)

:octicons-gear-24: **Default**
:   The first [`AWS_BEDROCK_REGIONS`](operations_configuration_aws.md#aws-bedrock-regions) entry

:octicons-alert-24: **Requirement**
:   Must be the table's own region, and requires [`AWS_DYNAMODB_TABLE`](#aws-dynamodb-table); startup fails when it is set alone. A table is a regional resource, so this setting has no failover: if the region is unreachable, so is every feature built on the table

```bash
export AWS_DYNAMODB_REGION=us-east-1
```

## :material-microphone: Transcription

Where Amazon Transcribe stages its audio and its output, how that output is encrypted, and the languages a streamed transcription picks between.

#### `AWS_TRANSCRIBE_S3_BUCKET` { #aws-transcribe-s3-bucket }

:octicons-package-24: **Purpose**
:   Temporary S3 bucket for transcription workflows

:octicons-gear-24: **Default**
:   Falls back to `AWS_S3_BUCKET` if not specified

:octicons-alert-24: **Requirement**
:   Must be in the same region as `AWS_TRANSCRIBE_REGION` when that is set; with the default multi-region behavior it serves the primary Bedrock region, and [`AWS_S3_REGIONAL_BUCKETS`](#aws-s3-regional-buckets) entries serve the other candidate regions

```bash
# If AWS_TRANSCRIBE_REGION is us-east-1
export AWS_TRANSCRIBE_S3_BUCKET=my-transcribe-temp-us-east-1

# If AWS_TRANSCRIBE_REGION is eu-west-1
export AWS_TRANSCRIBE_S3_BUCKET=my-transcribe-temp-eu-west-1
```

#### `AWS_TRANSCRIBE_STREAM_LANGUAGES` { #aws-transcribe-stream-languages }

:octicons-package-24: **Purpose**
:   Languages a streamed transcription ([`stream=true`](api_openai_audio_transcriptions.md#streaming)) picks between when the request names none

:octicons-gear-24: **Default**
:   Empty — a request naming no language is transcribed once the whole recording has been read, and its language detected

:octicons-code-24: **Format**
:   JSON array of two or more language codes; a single entry has no effect

A streamed transcription returns text before the recording has been fully read, which requires knowing which languages to expect. A request that names its `language` — or two or more expected `languages` — always gets one. Set this to extend the same behavior to requests that name neither, listing the languages your callers actually send. Listing more than five is not recommended, and two variants of the same language (`en-US` and `en-GB`) cannot both appear.

```bash
export AWS_TRANSCRIBE_STREAM_LANGUAGES='["en-US", "es-US", "fr-FR"]'
```

#### `AWS_TRANSCRIBE_OUTPUT_ENCRYPTION_KEY_ARN` { #aws-transcribe-output-encryption-key-arn }

:octicons-package-24: **Purpose**
:   AWS KMS key encrypting the transcription output written to [`AWS_TRANSCRIBE_S3_BUCKET`](#aws-transcribe-s3-bucket)

:octicons-gear-24: **Default**
:   None — output objects keep the bucket's own default encryption (SSE-S3)

:octicons-code-24: **Format**
:   KMS key ARN: `arn:<partition>:kms:<region>:<account-id>:key/<key-id>`; startup fails on any other value

:octicons-alert-24: **Requirement**
:   The server's role needs `kms:GenerateDataKey` and `kms:Decrypt` on the key ([IAM Permissions](operations_iam_permissions.md#speech-to-text-optional)). With the default multi-region behavior the key must be usable from every candidate region — a [multi-Region key](https://docs.aws.amazon.com/kms/latest/developerguide/multi-region-keys-overview.html) or a single [`AWS_TRANSCRIBE_REGION`](operations_configuration_aws.md#aws-transcribe-region)

```bash
export AWS_TRANSCRIBE_OUTPUT_ENCRYPTION_KEY_ARN=arn:aws:kms:us-east-1:123456789012:key/12345678-1234-1234-1234-123456789012
```

Each job also sends its request identifiers (`stdapi-ai.request_id`, `stdapi-ai.server_id`, and `stdapi-ai.user_id` when the user is known) as the [KMS encryption context](https://docs.aws.amazon.com/kms/latest/developerguide/concepts.html#encrypt_context), so a key policy can be conditioned on them.

## :material-earth: Regional and External Buckets

Buckets in other regions for async and batch inference, and the buckets outside the deployment that a request is allowed to name.

#### `AWS_S3_REGIONAL_BUCKETS` { #aws-s3-regional-buckets }

:octicons-package-24: **Purpose**
:   Region-specific S3 buckets for Bedrock async and batch inference operations, and for staging attachments too large to travel inside a request

:octicons-gear-24: **Default**
:   Empty (no regional buckets configured)

:octicons-code-24: **Format**
:   JSON object with region names as keys and bucket names as values

:octicons-alert-24: **Requirement**
:   Some Bedrock models require S3 buckets in the same region for async and batch inference operations

```bash
export AWS_S3_REGIONAL_BUCKETS='{"us-east-1": "my-bedrock-temp-us-east-1", "eu-west-1": "my-bedrock-temp-eu-west-1"}'
```

!!! info "When to Use"
    Configure this setting when:

    - Using Bedrock async inference API
    - Using Bedrock batch inference API
    - Working with models that require regional S3 storage
    - Accepting chat, messages or responses requests with [large attachments](features.md#attachment-size), or embedding requests with large inputs — those are staged in the bucket of the region serving the request, which then serves that request alone without failing over

    If not specified for a region where async/batch operations are attempted, those operations may fail. Requests carrying an attachment larger than the model reads inline are refused with `413` when no region able to serve the model has a bucket.

!!! success "Automatic Fallback"
    For the first region in `AWS_BEDROCK_REGIONS` (your primary region), if no regional bucket is specified, the service automatically falls back to `AWS_S3_BUCKET`. You only need to configure regional buckets for additional regions beyond your primary one.

!!! info "Terraform Module"
    When using the Terraform module, regional S3 buckets are created automatically for each region in `aws_bedrock_regions`. The bucket names are exposed via the `aws_s3_regional_buckets` output and passed to the container as `AWS_S3_REGIONAL_BUCKETS`. No manual configuration required.

!!! tip "Best Practice"
    Apply the same [S3 Bucket Lifecycle Configuration](#s3-lifecycle) to these regional buckets as you would for the primary bucket to automatically clean up temporary files.

#### `AWS_S3_ACCEPTED_BUCKETS` { #aws-s3-accepted-buckets }

:octicons-package-24: **Purpose**
:   Declare external S3 buckets that the application has read access to, mapped to their AWS region

:octicons-database-24: **Type**
:   JSON object (keys: bucket names, values: AWS region identifiers)

:octicons-gear-24: **Default**
:   `{}` (empty — only the application's own buckets are recognized)

:octicons-workflow-24: **Behavior**
:   Declaring a bucket here enables input access to objects the application does not own:

    - **S3 URI and S3 HTTP URL access** — `s3://` URIs and S3 HTTP URLs (including presigned URLs) pointing at these buckets are accepted as input sources; an HTTP URL is automatically converted to an `s3://` URI so Bedrock can access the object directly.
    - **Declared region** — The region mapped to each bucket is used to reach that bucket in its own region when reading the input object. It does not influence model or inference region selection.

    Without this setting, only the application's own buckets (`AWS_S3_BUCKET` and `AWS_S3_REGIONAL_BUCKETS`) are recognized.

```bash
export AWS_S3_ACCEPTED_BUCKETS='{"my-data-bucket": "us-east-1", "my-eu-bucket": "eu-west-1"}'
```

!!! warning "Required IAM Permissions"
    The application's IAM role must have `s3:GetObject` permission on each declared bucket. Granting access at the bucket level is recommended:

    ```json
    {
      "Effect": "Allow",
      "Action": "s3:GetObject",
      "Resource": [
        "arn:aws:s3:::my-data-bucket/*",
        "arn:aws:s3:::my-eu-bucket/*"
      ]
    }
    ```

!!! tip "When to Use"
    Configure this when your users provide S3 URLs from buckets outside the application's own buckets. This enables automatic HTTP-to-S3 URI conversion and optimal region routing for those objects.

## :material-broom: S3 Bucket Lifecycle Configuration { #s3-lifecycle }

:octicons-package-24: **Purpose**
:   Configure automatic deletion of temporary files and abandoned multipart upload parts to minimize storage costs

:octicons-clock-24: **Recommendation**
:   Configure S3 lifecycle policies to automatically delete objects under the `AWS_S3_TMP_PREFIX` after 1 day, and abort incomplete multipart uploads under the `AWS_S3_FILES_PREFIX` after 1 day

stdapi.ai stores temporary files under the prefix configured by `AWS_S3_TMP_PREFIX` (default: `tmp/`). These include generated images, audio files, and transcription workflow files. Configure S3 lifecycle policies to automatically delete objects under this prefix after 1 day.

Additionally, multipart file uploads (OpenAI Uploads API) store parts under `AWS_S3_FILES_PREFIX` (default: `files/`). If a session is never completed or cancelled — for example when a client disconnects — the uploaded parts remain in S3 and accumulate costs. Add an `AbortIncompleteMultipartUpload` rule on the files prefix to clean these up automatically.

!!! info "Application Cleanup Behavior"
    **Short-lived temporary files:** The application attempts to clean up short-lived temporary files (such as intermediate transcription files) after processing completes.

    **Results shared with clients:** Files shared with clients using presigned URLs (such as generated images and audio) are never cleaned up automatically by the application. These files remain in S3 until removed by lifecycle policies or manual deletion.

    **Why lifecycle policies are essential:** Since the application cannot determine when a client has finished using a presigned URL, S3 lifecycle policies are the recommended mechanism to clean up these files and prevent unbounded storage growth.

```json
{
  "Rules": [
    {
      "Id": "DeleteTemporaryFiles",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "tmp/"
      },
      "Expiration": {
        "Days": 1
      },
      "AbortIncompleteMultipartUpload": {
        "DaysAfterInitiation": 1
      }
    },
    {
      "Id": "AbortIncompleteMultipartUploads",
      "Status": "Enabled",
      "Filter": {
        "Prefix": "files/"
      },
      "AbortIncompleteMultipartUpload": {
        "DaysAfterInitiation": 1
      }
    }
  ]
}
```

!!! warning "Important: Update the Prefixes"
    The `"Prefix"` values in the lifecycle policy must match your `AWS_S3_TMP_PREFIX` and `AWS_S3_FILES_PREFIX` settings. If you use custom prefixes, update the policy accordingly.

    **Examples:**

    - If `AWS_S3_TMP_PREFIX=temporary/`, use `"Prefix": "temporary/"` in the first rule
    - If `AWS_S3_FILES_PREFIX=prod/files/`, use `"Prefix": "prod/files/"` in the second rule

**Apply via AWS CLI:**

```bash
# For primary S3 bucket (AWS_S3_BUCKET)
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-stdapi-bucket \
  --lifecycle-configuration file://lifecycle-policy.json

# For transcribe S3 bucket (AWS_TRANSCRIBE_S3_BUCKET, if different from AWS_S3_BUCKET)
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-transcribe-temp-bucket \
  --lifecycle-configuration file://lifecycle-policy.json

# For regional buckets (AWS_S3_REGIONAL_BUCKETS)
aws s3api put-bucket-lifecycle-configuration \
  --bucket my-stdapi-us-west-2-bucket \
  --lifecycle-configuration file://lifecycle-policy.json
```

!!! tip "Apply to All S3 Buckets"
    Apply this lifecycle policy to:

    - **`AWS_S3_BUCKET`** - Primary bucket for generated files
    - **`AWS_TRANSCRIBE_S3_BUCKET`** - Transcription temporary files (if different from AWS_S3_BUCKET)
    - **`AWS_S3_REGIONAL_BUCKETS`** - All regional buckets for async/batch operations

    All these buckets use the same `AWS_S3_TMP_PREFIX` for temporary file storage, and the same `AWS_S3_FILES_PREFIX` for multipart upload parts.
