---
title: IAM Permissions - Amazon Bedrock API Gateway Access Policies
description: Complete AWS IAM permission reference for stdapi.ai, including required Amazon Bedrock policies, feature-specific statements, and full policy examples.
keywords: IAM permissions Bedrock, AWS IAM policy, Bedrock IAM actions, least privilege AWS, S3 IAM policy, Polly IAM, Transcribe IAM, API gateway permissions
---

# :material-shield-key: IAM Permissions

stdapi.ai requires specific AWS IAM permissions to access Amazon Bedrock models and other AWS services. The exact permissions needed depend on which features you enable.

!!! tip "Building Your Policy"
    Combine the permission statements below based on the features you need. At minimum, you need the **Bedrock** permissions. Add statements for S3, TTS, STT, and other features as required by your deployment. Only include the statements you need — start with the Bedrock permissions and add others as required (least privilege).

!!! info "Terraform Module"
    The official [stdapi-ai Terraform module](https://github.com/stdapi-ai/terraform-aws-stdapi-ai) provisions the ECS task role with the required permissions automatically. This reference is for custom deployments and policy auditing.

!!! warning "Multi-Region Failover and Region-Scoped Policies"
    By default, Amazon Bedrock **and** the other AWS AI services (Polly, Transcribe, Comprehend, Translate) are called in every region listed in [`AWS_BEDROCK_REGIONS`](operations_configuration.md#aws-bedrock-regions), failing over from one to the next on throttling, quota, or availability errors. See [Other AWS Services Failover](operations_resilience.md#other-aws-services-failover).

    The statements on this page use region-agnostic resources, so they work as-is. However, any `aws:RequestedRegion` condition — in the policy itself, a permissions boundary, or a service control policy — must allow **all** configured regions, otherwise failover silently fails and requests error out once the first region is unavailable.

---

## :material-aws: Bedrock (Required) { #bedrock-iam }

**Environment Variables**: Always required

These permissions are mandatory for stdapi.ai to discover and invoke Amazon Bedrock models:

??? example "Bedrock IAM Policy Statements"
    ```json
    {
      "Sid": "BedrockModelInvoke",
      "Effect": "Allow",
      "Action": [
        "bedrock:CountTokens",
        "bedrock:GetAsyncInvoke",
        "bedrock:InvokeGuardrailChecks",
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithBidirectionalStream",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:InvokeTool",
        "bedrock:Rerank"
      ],
      "Resource": "*"
    },
    {
      "Sid": "BedrockAsyncInvokeTagging",
      "Effect": "Allow",
      "Action": [
        "bedrock:TagResource"
      ],
      "Resource": "arn:aws:bedrock:*:*:async-invoke/*"
    },
    {
      "Sid": "BedrockModelDiscovery",
      "Effect": "Allow",
      "Action": [
        "bedrock:ListFoundationModels",
        "bedrock:GetFoundationModelAvailability",
        "bedrock:ListProvisionedModelThroughputs",
        "bedrock:ListInferenceProfiles"
      ],
      "Resource": "*"
    }
    ```

!!! note "Asynchronous Invocations"
    `bedrock:GetAsyncInvoke` and `bedrock:TagResource` (on `arn:aws:bedrock:*:*:async-invoke/*`) serve Bedrock asynchronous invocations, used by video generation models and asynchronous embedding models such as TwelveLabs Marengo (`twelvelabs.marengo-embed-*`). They can be dropped if your deployment uses none of these models. `bedrock:ListAsyncInvokes` and `bedrock:ListTagsForResource` are **not** part of this core set — they are only needed for video job listing (see [Video Generation](#video-generation-optional)).

!!! note "Bidirectional Streaming"
    `bedrock:InvokeModelWithBidirectionalStream` serves any model invoked over a persistent, two-way connection: live audio transcription with Amazon Nova Sonic (see [Speech-to-Text](#speech-to-text-optional)) and the [Realtime API](api_openai_realtime.md) (`POST /v1/realtime/client_secrets`, `WS /v1/realtime`) alike. No route-specific action exists for either — it is already part of the core Bedrock policy above.

---

## :material-storefront: Bedrock Marketplace Auto-Subscribe (Optional) { #bedrock-marketplace-auto-subscribe-iam }

**Environment Variables**: [`AWS_BEDROCK_MARKETPLACE_AUTO_SUBSCRIBE`](operations_configuration.md#bedrock-marketplace-auto-subscribe)

Required only if you want models sold as third-party AWS Marketplace listings to be usable without subscribing to each one by hand (`AWS_BEDROCK_MARKETPLACE_AUTO_SUBSCRIBE=true`, which is the default). The server never calls `Subscribe` itself: it keeps a listing with no agreement in the catalogue, and AWS creates the subscription under this role on the first invocation. It applies to whichever models AWS sells that way — see [Which Models Are Which](operations_cost_management.md#which-models-are-which); models billed as ordinary Amazon Bedrock usage need none of these permissions.

??? example "Bedrock Marketplace Auto-Subscribe IAM Policy Statement"
    ```json
    {
      "Sid": "BedrockMarketplaceAutoSubscribe",
      "Effect": "Allow",
      "Action": [
        "aws-marketplace:Subscribe",
        "aws-marketplace:ViewSubscriptions"
      ],
      "Resource": "*"
    }
    ```

    !!! warning "Cost Consideration"
        Automatic marketplace subscriptions may incur costs. Review AWS Marketplace pricing for individual models before enabling this feature, or set `AWS_BEDROCK_MARKETPLACE_AUTO_SUBSCRIBE=false` to require manual marketplace subscription.

---

## :material-storefront: AWS Marketplace Metering (AWS Marketplace Image Only) { #aws-marketplace-metering }

**Environment Variables**: none (always active on the AWS Marketplace image)

Required only for the **AWS Marketplace image** — not the community image. At startup it registers hourly usage with AWS Marketplace Metering on ECS, EKS, and Fargate; an `AccessDenied` error aborts startup.

??? example "AWS Marketplace Metering IAM Policy Statement"
    ```json
    {
      "Sid": "MarketplaceRegisterUsage",
      "Effect": "Allow",
      "Action": [
        "aws-marketplace:RegisterUsage"
      ],
      "Resource": "*"
    }
    ```

---

## :material-directions-fork: Bedrock Inference Profiles, Prompt Routers and Prompt Management (Optional) { #bedrock-inference-profiles-and-prompt-routers-optional }

**Environment Variables**: [`AWS_BEDROCK_ALLOW_CROSS_REGION_INFERENCE_PROFILE_ARN`](operations_configuration.md#bedrock-allow-cross-region-profile-arn), [`AWS_BEDROCK_ALLOW_APPLICATION_INFERENCE_PROFILE_ARN`](operations_configuration.md#bedrock-allow-application-profile-arn), [`AWS_BEDROCK_ALLOW_PROMPT_ROUTER_ARN`](operations_configuration.md#bedrock-allow-prompt-router-arn), [`AWS_BEDROCK_ALLOW_PROMPT_ARN`](operations_configuration.md#bedrock-allow-prompt-arn), [`AWS_BEDROCK_MODEL_ARN_MAPPING`](operations_configuration.md#bedrock-model-arn-mapping)

Required only if you enable ARN-based routing features that allow users to pass inference profile, prompt router or Prompt Management prompt ARNs directly, or if you configure server-side ARN mappings.

??? example "Bedrock Inference Profiles, Prompt Routers and Prompt Management IAM Policy Statements"
    ```json
    {
      "Sid": "BedrockInferenceProfilesAndPromptRouters",
      "Effect": "Allow",
      "Action": [
        "bedrock:GetInferenceProfile",
        "bedrock:GetPromptRouter"
      ],
      "Resource": "*"
    },
    {
      "Sid": "BedrockPromptManagement",
      "Effect": "Allow",
      "Action": [
        "bedrock:GetPrompt",
        "bedrock:RenderPrompt"
      ],
      "Resource": "arn:aws:bedrock:*:*:prompt/*"
    }
    ```

    `bedrock:GetPrompt` resolves the model bound to the prompt variant; `bedrock:RenderPrompt` is required because the prompt ARN is then sent to Bedrock as the invocation `modelId`.

    !!! note "When to Include"
        Add these permissions when:

        - `AWS_BEDROCK_ALLOW_CROSS_REGION_INFERENCE_PROFILE_ARN=true`
        - `AWS_BEDROCK_ALLOW_APPLICATION_INFERENCE_PROFILE_ARN=true`
        - `AWS_BEDROCK_ALLOW_PROMPT_ROUTER_ARN=true`
        - `AWS_BEDROCK_ALLOW_PROMPT_ARN=true` (`bedrock:GetPrompt` and `bedrock:RenderPrompt` only)
        - `AWS_BEDROCK_MODEL_ARN_MAPPING` is configured with any mappings

---

## :material-shield-check: Bedrock Guardrails (Optional)

**Environment Variables**: [`AWS_BEDROCK_GUARDRAIL_IDENTIFIER`](operations_configuration.md#aws-bedrock-guardrail-identifier), [`AWS_BEDROCK_GUARDRAIL_VERSION`](operations_configuration.md#aws-bedrock-guardrail-version)

Required if you configure Bedrock Guardrails for content filtering, use the `moderation` request parameter, or select a guardrail on the [Moderations API](api_openai_moderations.md) (without a guardrail, that API falls back to [Comprehend toxicity moderation](#comprehend-moderation)). See the [Bedrock Guardrails](operations_configuration.md#bedrock-guardrails) configuration section.

??? example "Bedrock Guardrails IAM Policy Statement"
    ```json
    {
      "Sid": "BedrockGuardrails",
      "Effect": "Allow",
      "Action": [
        "bedrock:ApplyGuardrail"
      ],
      "Resource": "arn:aws:bedrock:*:*:guardrail/*"
    }
    ```

---

## :material-database-lock: Bedrock Session Storage (Optional)

**Environment Variables**: none (enabled by the `store=true` request parameter; see [Bedrock Session Storage](operations_configuration.md#bedrock-session-storage-optional) configuration)

Required if clients use `store=true` on the [Responses](api_openai_responses.md#stored-responses) or [Chat Completions](api_openai_chat_completions.md#stored-chat-completions) APIs, or the [Conversations](api_openai_conversations.md) API, all of which persist state in Amazon Bedrock sessions.

??? example "Bedrock Session Storage IAM Policy Statement"
    ```json
    {
      "Sid": "BedrockSessionStorage",
      "Effect": "Allow",
      "Action": [
        "bedrock:CreateSession",
        "bedrock:GetSession",
        "bedrock:UpdateSession",
        "bedrock:CreateInvocation",
        "bedrock:PutInvocationStep",
        "bedrock:ListInvocations",
        "bedrock:ListInvocationSteps",
        "bedrock:GetInvocationStep",
        "bedrock:EndSession",
        "bedrock:DeleteSession",
        "bedrock:TagResource",
        "bedrock:ListTagsForResource"
      ],
      "Resource": "arn:aws:bedrock:*:*:session/*"
    },
    {
      "Sid": "BedrockSessionListing",
      "Effect": "Allow",
      "Action": "bedrock:ListSessions",
      "Resource": "*"
    }
    ```

    `bedrock:ListSessions` serves the stored chat completions listing endpoint (`GET /v1/chat/completions`); the account-level `ListSessions` action does not support resource scoping. `bedrock:GetSession` is used on deletion and `bedrock:ListTagsForResource` on both deletion and listing, to check that a stored object belongs to the API it is requested from. `bedrock:UpdateSession` serves the conversation metadata update (`POST /v1/conversations/{conversation_id}`) only.

    Add `kms:Decrypt` and `kms:GenerateDataKey` on the key when [`AWS_BEDROCK_SESSION_ENCRYPTION_KEY_ARN`](operations_configuration.md#aws-bedrock-session-encryption-key-arn) is configured.

---

## :material-layers-triple: Bedrock Mantle (Optional) { #bedrock-mantle-iam }

**Environment Variables**: [`AWS_BEDROCK_MANTLE_ENABLED`](operations_configuration.md#bedrock-mantle-enabled)

Required for [`AWS_BEDROCK_MANTLE_ENABLED`](operations_configuration.md#bedrock-mantle-enabled) (enabled by default), which exposes models served by the Amazon Bedrock Mantle endpoint (OpenAI GPT, xAI Grok, Google Gemma, and more). Without these permissions the server still starts normally: Mantle models are not listed and a warning is logged.

??? example "Bedrock Mantle IAM Policy Statements"
    ```json
    {
      "Sid": "BedrockMantleInference",
      "Effect": "Allow",
      "Action": [
        "bedrock-mantle:CreateInference",
        "bedrock-mantle:GetInference",
        "bedrock-mantle:DeleteInference",
        "bedrock-mantle:ListModels",
        "bedrock-mantle:GetModel",
        "bedrock-mantle:CancelInference"
      ],
      "Resource": "arn:aws:bedrock-mantle:*:*:project/*"
    },
    {
      "Sid": "BedrockMantleBearerToken",
      "Effect": "Allow",
      "Action": "bedrock-mantle:CallWithBearerToken",
      "Resource": "*"
    }
    ```

    `bedrock-mantle:CallWithBearerToken` authorizes the short-term bearer tokens the server derives from its AWS credential chain; it does not support resource scoping. Token counting for Mantle-only models is served by the same endpoint and needs no additional action.

---

## :material-web: Web Search (Optional) { #web-search-iam }

**Environment Variables**: none (enabled by a `web_search` tool in a request)

Required for the built-in [web search tool](api_openai_responses.md#openai-gpt-web-search) on the OpenAI GPT-5.x family, whichever settings are in use. Each action is authorized only when the model actually attempts that call, and a denied call does not fail the request: AWS documents the model continuing with the information it already has and telling you it could not retrieve enough current information ([Identity and access management for Web Search](https://docs.aws.amazon.com/bedrock/latest/userguide/security-web-search.html)).

!!! warning "A missing web search permission produces no error and no server log entry"
    The denial is handled inside the model call, so the request succeeds with a normal answer: the server sees nothing to report, and the response is indistinguishable from the model deciding it did not need to search. When answers never cite a source, check these permissions (and the Region the call was served in) before suspecting the model. AWS CloudTrail records the denied `bedrock-websearch` calls.

Add `bedrock-websearch:ExternalWebAccess` on top when a request can reach external web access — that is, when [`AWS_BEDROCK_EXTERNAL_WEB_ACCESS`](operations_configuration.md#bedrock-external-web-access) is enabled, or when [`AWS_BEDROCK_ALLOW_EXTERNAL_WEB_ACCESS_OVERRIDE`](operations_configuration.md#bedrock-allow-external-web-access-override) lets a client ask for it per request. Leaving it out is what keeps every search inside the AWS boundary.

??? example "Web Search IAM Policy Statement"
    ```json
    {
      "Sid": "BedrockWebSearch",
      "Effect": "Allow",
      "Action": [
        "bedrock-websearch:InvokeSearch",
        "bedrock-websearch:InvokeFetch"
      ],
      "Resource": "*"
    }
    ```

    `InvokeSearch` and `InvokeFetch` are authorized on AWS-owned tool resources (`arn:aws:bedrock-websearch:<region>:aws:tool/<name>`), which `"*"` covers; the AWS managed policies instead scope them to `arn:aws:bedrock-websearch:*:*:*`, and either form grants the same searches. `ExternalWebAccess` is a permission-only action with no resource of its own, so grant it with `"*"`. Web search runs in the Region that served the model call, in the [Regions where the tool is offered](api_openai_responses.md#openai-gpt-web-search); scope the statement with an `aws:RequestedRegion` condition to pin which Regions may run searches. See [Identity and access management for Web Search](https://docs.aws.amazon.com/bedrock/latest/userguide/security-web-search.html) and [Actions, resources, and condition keys](https://docs.aws.amazon.com/service-authorization/latest/reference/list_bedrock-websearch.html).

---

## :material-database: S3 File Storage (Optional)

**Environment Variables**: [`AWS_S3_BUCKET`](operations_configuration.md#aws-s3-bucket), [`AWS_S3_REGIONAL_BUCKETS`](operations_configuration.md#aws-s3-regional-buckets)

Required for storing generated images, audio files, documents, and videos. See [Storage Configuration](operations_configuration.md#storage-configuration) for bucket setup details.

??? example "S3 File Storage IAM Policy Statements"
    ```json
    {
      "Sid": "S3FileStorage",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:PutObjectTagging",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:AbortMultipartUpload",
        "s3:ListMultipartUploadParts"
      ],
      "Resource": "arn:aws:s3:::AWS_S3_BUCKET_VALUE/*"
    },
    {
      "Sid": "S3FileStorageList",
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket",
        "s3:ListBucketMultipartUploads"
      ],
      "Resource": "arn:aws:s3:::AWS_S3_BUCKET_VALUE"
    }
    ```

    !!! info "Replace Bucket Name"
        Replace `AWS_S3_BUCKET_VALUE` with the value of your [`AWS_S3_BUCKET`](operations_configuration.md#aws-s3-bucket) environment variable. Repeat both statements for each [`AWS_S3_REGIONAL_BUCKETS`](operations_configuration.md#aws-s3-regional-buckets) bucket — they serve video generation, asynchronous embeddings (TwelveLabs Marengo, Amazon Nova), [large attachments](features.md#attachment-size) on any multimodal route, and [Speech-to-Text](#speech-to-text-optional) failover, and the Files API looks up objects across every configured bucket.

    !!! note "Multipart Uploads"
        Large files are uploaded and copied with the multipart API. Its `CreateMultipartUpload`, `UploadPart`, `UploadPartCopy`, and `CompleteMultipartUpload` operations have no dedicated IAM actions — they are authorized by `s3:PutObject` — which is why only the abort and listing actions appear above.

    !!! note "Cross-Region Access"
        Bucket ARNs are region-agnostic, so one statement per bucket covers every region. When a request fails over to another region, the server copies the object server-side through the destination region's S3 endpoint, reading the source bucket from there. Grant these actions on the source and destination buckets alike, and on the KMS key of each encrypted bucket.

    **If your S3 bucket uses KMS encryption**, also add:

    ```json
    {
      "Sid": "KMSEncryptedBucket",
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt",
        "kms:GenerateDataKey"
      ],
      "Resource": "arn:aws:kms:REGION:ACCOUNT_ID:key/YOUR_KMS_KEY_ID",
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "s3.REGION.amazonaws.com"
        }
      }
    }
    ```

    !!! tip "KMS Security"
        The `kms:ViaService` condition restricts KMS key usage to S3 service calls only, following AWS security best practices. Because the condition pins a single region, add one statement per region — with that region's key ARN and `s3.REGION.amazonaws.com` — when you use regional buckets with per-region keys.

---

## :material-magnify: Vector Stores (Optional) { #vector-stores-optional }

**Environment Variables**: [`AWS_S3_VECTORS_BUCKET`](operations_configuration.md#aws-s3-vectors-bucket), [`AWS_S3_VECTORS_REGION`](operations_configuration.md#aws-s3-vectors-region), [`AWS_S3_BUCKET`](operations_configuration.md#aws-s3-bucket)

Required by the [Vector Stores API](api_openai_vector_stores.md). The indexed content lives in an [Amazon S3 vector bucket](https://docs.aws.amazon.com/AmazonS3/latest/userguide/s3-vectors.html) you create; the stores' own records live in the general purpose bucket under [`AWS_S3_VECTOR_STORES_PREFIX`](operations_configuration.md#aws-s3-vector-stores-prefix) and are covered by the [S3 File Storage](#s3-file-storage-optional) statements above.

??? example "Vector Stores IAM Policy Statements"
    ```json
    {
      "Sid": "VectorStoreIndexes",
      "Effect": "Allow",
      "Action": [
        "s3vectors:CreateIndex",
        "s3vectors:DeleteIndex",
        "s3vectors:PutVectors",
        "s3vectors:GetVectors",
        "s3vectors:QueryVectors",
        "s3vectors:DeleteVectors"
      ],
      "Resource": [
        "arn:aws:s3vectors:REGION:ACCOUNT_ID:bucket/AWS_S3_VECTORS_BUCKET_VALUE",
        "arn:aws:s3vectors:REGION:ACCOUNT_ID:bucket/AWS_S3_VECTORS_BUCKET_VALUE/index/*"
      ]
    }
    ```

    !!! info "Replace the Placeholders"
        Replace `AWS_S3_VECTORS_BUCKET_VALUE` with your [`AWS_S3_VECTORS_BUCKET`](operations_configuration.md#aws-s3-vectors-bucket) value, `REGION` with [`AWS_S3_VECTORS_REGION`](operations_configuration.md#aws-s3-vectors-region), and `ACCOUNT_ID` with your account. The bucket ARN itself is needed for the index actions; the index ARN pattern covers the per-store indexes the gateway creates and deletes.

    !!! note "The bucket is yours to create"
        The gateway never creates or deletes the vector bucket, only the indexes inside it, so no bucket-level create or delete action is granted.

    !!! note "Records live in the general purpose bucket"
        Grant the [S3 File Storage](#s3-file-storage-optional) statements on [`AWS_S3_BUCKET`](operations_configuration.md#aws-s3-bucket) as well: the stores, their attached files and their batches are JSON objects there.

---

## :material-tray-arrow-down: Durable Vector Store Indexing (Optional) { #durable-vector-store-indexing }

**Environment Variables**: [`AWS_SQS_VECTOR_STORE_QUEUE_URL`](operations_configuration.md#aws-sqs-vector-store-queue-url)

Required to keep indexing a vector store file when the server that accepted it stops. The gateway both writes the work to the [Amazon SQS](https://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSDeveloperGuide/welcome.html) queue you create and reads it back, so it needs the producer and the consumer actions on that one queue.

??? example "Durable Vector Store Indexing IAM Policy Statements"
    ```json
    {
      "Sid": "VectorStoreIndexingQueue",
      "Effect": "Allow",
      "Action": [
        "sqs:SendMessage",
        "sqs:ReceiveMessage",
        "sqs:DeleteMessage",
        "sqs:ChangeMessageVisibility",
        "sqs:GetQueueAttributes"
      ],
      "Resource": "arn:aws:sqs:REGION:ACCOUNT_ID:QUEUE_NAME"
    }
    ```

    !!! info "Replace the Placeholders"
        `REGION`, `ACCOUNT_ID` and `QUEUE_NAME` are the three parts of your [`AWS_SQS_VECTOR_STORE_QUEUE_URL`](operations_configuration.md#aws-sqs-vector-store-queue-url). Grant this on the queue ARN itself — never on `*`.

    !!! note "The queues are yours to create"
        The gateway never creates, deletes or reconfigures a queue, so no `sqs:CreateQueue`, `sqs:DeleteQueue` or `sqs:SetQueueAttributes` is granted. `sqs:GetQueueAttributes` is read-only and is what lets the gateway honour your dead-letter queue's redrive policy.

    !!! note "The dead-letter queue needs nothing"
        Amazon SQS moves an exhausted message itself; the gateway never reads the dead-letter queue, so grant it nothing.

---

## :material-book-search: Knowledge Base Vector Stores (Optional) { #knowledge-base-vector-stores }

**Environment Variables**: [`AWS_BEDROCK_KNOWLEDGE_BASE_IDS`](operations_configuration.md#aws-bedrock-knowledge-base-ids)

Required to serve an [Amazon Bedrock knowledge base](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html) you created as a [vector store](api_openai_vector_stores.md#knowledge-base-stores). Grant one statement per allowlisted knowledge base, on its own ARN.

??? example "Knowledge Base Vector Stores IAM Policy Statements"
    ```json
    {
      "Sid": "BedrockKnowledgeBaseVectorStores",
      "Effect": "Allow",
      "Action": [
        "bedrock:GetKnowledgeBase",
        "bedrock:Retrieve",
        "bedrock:ListDataSources",
        "bedrock:IngestKnowledgeBaseDocuments",
        "bedrock:ListKnowledgeBaseDocuments",
        "bedrock:GetKnowledgeBaseDocuments",
        "bedrock:DeleteKnowledgeBaseDocuments"
      ],
      "Resource": "arn:aws:bedrock:REGION:ACCOUNT_ID:knowledge-base/KNOWLEDGE_BASE_ID"
    }
    ```

    !!! info "Replace the Placeholders"
        Replace `REGION` with the first [`AWS_BEDROCK_REGIONS`](operations_configuration.md#aws-bedrock-regions) entry, `ACCOUNT_ID` with your account, and `KNOWLEDGE_BASE_ID` with the knowledge base identifier — one ARN per entry of [`AWS_BEDROCK_KNOWLEDGE_BASE_IDS`](operations_configuration.md#aws-bedrock-knowledge-base-ids).

    !!! note "No discovery action, deliberately"
        `bedrock:ListKnowledgeBases` is **not** granted, and is not needed: the server never discovers knowledge bases it was not given. Only the identifiers in the allowlist are ever addressed, and any other one is answered as an unknown vector store.

    !!! note "The knowledge base is yours to create"
        The gateway never creates or deletes a knowledge base, so no create, update or delete action on the knowledge base itself is granted — only the documents of its data source.

    !!! tip "Checked once at startup"
        `bedrock:GetKnowledgeBase` — already needed to serve a store — is also called on every allowlisted entry at startup, to confirm each one is a kind the server can serve. No extra action is required for that check, and a role missing the action never stops the server: it starts with one warning per entry instead.

---

## :material-video: Video Generation (Optional) { #video-generation-optional }

**Environment Variables**: [`AWS_S3_REGIONAL_BUCKETS`](operations_configuration.md#aws-s3-regional-buckets)

Video generation itself runs on the core Bedrock asynchronous invocation permissions (`bedrock:InvokeModel`, `bedrock:GetAsyncInvoke`, `bedrock:TagResource`) plus [S3 File Storage](#s3-file-storage-optional) permissions on each regional bucket. The video job listing endpoint (`GET /v1/videos`) additionally requires:

??? example "Video Job Listing IAM Policy Statements"
    ```json
    {
      "Sid": "BedrockVideoJobListing",
      "Effect": "Allow",
      "Action": [
        "bedrock:ListAsyncInvokes"
      ],
      "Resource": "*"
    },
    {
      "Sid": "BedrockVideoJobTags",
      "Effect": "Allow",
      "Action": [
        "bedrock:ListTagsForResource"
      ],
      "Resource": "arn:aws:bedrock:*:*:async-invoke/*"
    }
    ```

    The account-level `ListAsyncInvokes` action does not support resource scoping; `ListTagsForResource` reads the job metadata tags used to attribute listed jobs.

---

## :material-package-variant-closed: Batch Inference (Optional) { #batch-inference }

**Environment Variables**: [`AWS_BEDROCK_BATCH_ROLE_ARN`](operations_configuration.md#aws-bedrock-batch-role-arn), [`AWS_S3_BATCHES_PREFIX`](operations_configuration.md#aws-s3-batches-prefix), [`AWS_S3_BUCKET`](operations_configuration.md#aws-s3-bucket), [`AWS_S3_REGIONAL_BUCKETS`](operations_configuration.md#aws-s3-regional-buckets)

The [Batch API](api_openai_batches.md) and the [Message Batches API](api_anthropic_batches.md) run on [Amazon Bedrock batch inference](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html), which needs **two** policies: the server's own, and a service role Amazon Bedrock assumes to read the requests and write the results.

### The Server's Policy

??? example "Batch Inference IAM Policy Statements"
    ```json
    {
      "Sid": "BedrockBatchJobs",
      "Effect": "Allow",
      "Action": [
        "bedrock:CreateModelInvocationJob",
        "bedrock:GetModelInvocationJob",
        "bedrock:StopModelInvocationJob"
      ],
      "Resource": "arn:aws:bedrock:*:<account-id>:model-invocation-job/*"
    },
    {
      "Sid": "BedrockBatchPassRole",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::<account-id>:role/stdapi-ai-batch",
      "Condition": {
        "StringEquals": {"iam:PassedToService": "bedrock.amazonaws.com"}
      }
    }
    ```

    Substitute `<account-id>` with your AWS account ID, and the role ARN with the value of `AWS_BEDROCK_BATCH_ROLE_ARN`. The server also needs the [S3 File Storage](#s3-file-storage-optional) permissions on every bucket a batch may use.

### The Service Role

Create a role named by `AWS_BEDROCK_BATCH_ROLE_ARN` whose trust policy lets Amazon Bedrock assume it, scoped to your account and to batch jobs so it cannot be used by another account's jobs:

??? example "Batch Service Role Trust Policy"
    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Principal": {"Service": "bedrock.amazonaws.com"},
          "Action": "sts:AssumeRole",
          "Condition": {
            "StringEquals": {"aws:SourceAccount": "<account-id>"},
            "ArnEquals": {
              "aws:SourceArn": "arn:aws:bedrock:*:<account-id>:model-invocation-job/*"
            }
          }
        }
      ]
    }
    ```

??? example "Batch Service Role Permissions Policy"
    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Sid": "BatchDataAccess",
          "Effect": "Allow",
          "Action": ["s3:GetObject", "s3:PutObject"],
          "Resource": "arn:aws:s3:::<bucket-name>/batches/*"
        },
        {
          "Sid": "BatchDataListing",
          "Effect": "Allow",
          "Action": "s3:ListBucket",
          "Resource": "arn:aws:s3:::<bucket-name>",
          "Condition": {"StringLike": {"s3:prefix": "batches/*"}}
        },
        {
          "Sid": "BatchModelInvocation",
          "Effect": "Allow",
          "Action": "bedrock:InvokeModel",
          "Resource": [
            "arn:aws:bedrock:*::foundation-model/*",
            "arn:aws:bedrock:*:<account-id>:inference-profile/*"
          ]
        }
      ]
    }
    ```

    Repeat the S3 statements for each `AWS_S3_REGIONAL_BUCKETS` bucket, and substitute `batches/` with your `AWS_S3_BATCHES_PREFIX`. A cross-region inference profile needs `bedrock:InvokeModel` on **both** the profile and the foundation models behind it — see [Inference Profiles](#bedrock-inference-profiles-and-prompt-routers-optional).

---

## :material-account-voice: Text-to-Speech (Optional) { #text-to-speech-optional }

**Environment Variables**: [`AWS_POLLY_REGION`](operations_configuration.md#aws-polly-region), [`DEFAULT_TTS_MODEL`](operations_configuration.md#default-tts-model), [`DEFAULT_TTS_LANGUAGE`](operations_configuration.md#default-tts-language), [`AWS_S3_BUCKET`](operations_configuration.md#aws-s3-bucket), [`AWS_S3_REGIONAL_BUCKETS`](operations_configuration.md#aws-s3-regional-buckets)

Required for generating speech from text using Amazon Polly. See the [Audio and Text-to-Speech](operations_configuration.md#audio-and-text-to-speech) configuration section.

!!! tip "Optimize Performance"
    Set [`DEFAULT_TTS_LANGUAGE`](operations_configuration.md#default-tts-language) to skip language detection and avoid Amazon Comprehend API calls, improving response times and reducing costs.

??? example "Polly Text-to-Speech IAM Policy Statements"
    ```json
    {
      "Sid": "PollyTextToSpeech",
      "Effect": "Allow",
      "Action": [
        "polly:SynthesizeSpeech",
        "polly:StartSpeechSynthesisStream",
        "polly:DescribeVoices",
        "polly:StartSpeechSynthesisTask",
        "polly:GetSpeechSynthesisTask"
      ],
      "Resource": "*"
    },
    {
      "Sid": "PollyS3Storage",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::AWS_S3_BUCKET_VALUE/*"
    }
    ```

    !!! info "Only for Long Input"
        The two task actions and the S3 statement serve [input above 3,000 characters](api_openai_audio_speech.md#long-input), which Amazon Polly synthesizes into a bucket co-located with the serving region.

        Replace `AWS_S3_BUCKET_VALUE` with the value of your [`AWS_S3_BUCKET`](operations_configuration.md#aws-s3-bucket) environment variable, and repeat the statement for each [`AWS_S3_REGIONAL_BUCKETS`](operations_configuration.md#aws-s3-regional-buckets) bucket serving a Polly region. Amazon Polly writes the audio object with the identity that started the synthesis, which is why `s3:PutObject` belongs to this policy; the audio is then read back and deleted once the response has been sent.

    !!! warning "Granting the permissions is not what enables long input"
        The 3,000-character limit is decided by the configuration, not by this policy: text-to-speech stays capped at 3,000 characters per request — and requests above it are rejected with that limit — as long as no bucket is configured for the Polly regions ([`AWS_S3_BUCKET`](operations_configuration.md#aws-s3-bucket), [`AWS_S3_REGIONAL_BUCKETS`](operations_configuration.md#aws-s3-regional-buckets)).

        With a bucket configured but these actions missing, long requests are accepted and then fail on a permission error instead. Grant the whole set, or leave the bucket unconfigured.

    **If your S3 buckets use KMS encryption**, also add the KMS permissions for each bucket's key, with that region's `kms:ViaService` value.

---

## :material-microphone: Speech-to-Text (Optional) { #speech-to-text-optional }

**Environment Variables**: [`AWS_TRANSCRIBE_REGION`](operations_configuration.md#aws-transcribe-region), [`AWS_TRANSCRIBE_S3_BUCKET`](operations_configuration.md#aws-transcribe-s3-bucket), [`AWS_S3_REGIONAL_BUCKETS`](operations_configuration.md#aws-s3-regional-buckets), [`AWS_TRANSCRIBE_OUTPUT_ENCRYPTION_KEY_ARN`](operations_configuration.md#aws-transcribe-output-encryption-key-arn)

Required for transcribing audio files using Amazon Transcribe. Each transcription job stages its audio in a bucket co-located with the Transcribe endpoint, so the S3 statement must cover every bucket that serves a candidate region.

??? example "Transcribe Speech-to-Text IAM Policy Statements"
    ```json
    {
      "Sid": "TranscribeSpeechToText",
      "Effect": "Allow",
      "Action": [
        "transcribe:StartTranscriptionJob",
        "transcribe:GetTranscriptionJob",
        "transcribe:DeleteTranscriptionJob",
        "transcribe:StartStreamTranscription"
      ],
      "Resource": "*"
    },
    {
      "Sid": "TranscribeTagging",
      "Effect": "Allow",
      "Action": [
        "transcribe:TagResource"
      ],
      "Resource": "arn:aws:transcribe:*:*:transcription-job/*"
    },
    {
      "Sid": "TranscribeS3Storage",
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:PutObjectTagging",
        "s3:GetObject",
        "s3:DeleteObject",
        "s3:AbortMultipartUpload",
        "s3:ListMultipartUploadParts"
      ],
      "Resource": "arn:aws:s3:::AWS_TRANSCRIBE_S3_BUCKET_VALUE/*"
    }
    ```

    !!! info "Replace Bucket Name"
        Replace `AWS_TRANSCRIBE_S3_BUCKET_VALUE` with the value of your [`AWS_TRANSCRIBE_S3_BUCKET`](operations_configuration.md#aws-transcribe-s3-bucket) environment variable (or [`AWS_S3_BUCKET`](operations_configuration.md#aws-s3-bucket) if using the same bucket).

    !!! note "One Bucket per Candidate Region"
        With the default multi-region behavior ([`AWS_TRANSCRIBE_REGION`](operations_configuration.md#aws-transcribe-region) unset), Transcribe fails over across the [`AWS_BEDROCK_REGIONS`](operations_configuration.md#aws-bedrock-regions) that have a co-located bucket: the primary region uses the bucket above, the others their [`AWS_S3_REGIONAL_BUCKETS`](operations_configuration.md#aws-s3-regional-buckets) entry. Repeat the `TranscribeS3Storage` statement for each of those buckets.

        On failover the audio is server-side copied from the previous candidate's bucket to the next one, which is why the copy and multipart actions are required. Set `AWS_TRANSCRIBE_REGION` to pin a single region and keep a single bucket.

    !!! info "`StartStreamTranscription` needs no bucket"
        `transcribe:StartStreamTranscription` serves [`stream=true`](api_openai_audio_transcriptions.md#streaming) requests, which send their audio to Transcribe directly instead of staging it. A deployment with no bucket at all still serves those, and only those — the `TranscribeS3Storage` statement above is what the other requests need.

    **If your transcribe S3 buckets use KMS encryption**, also add the KMS permissions for each bucket's key, with that region's `kms:ViaService` value.

**Encrypting the transcription output with your own key** ([`AWS_TRANSCRIBE_OUTPUT_ENCRYPTION_KEY_ARN`](operations_configuration.md#aws-transcribe-output-encryption-key-arn)) additionally requires:

??? example "Transcribe Output Encryption IAM Policy Statement"
    ```json
    {
      "Sid": "TranscribeOutputEncryption",
      "Effect": "Allow",
      "Action": [
        "kms:GenerateDataKey",
        "kms:Decrypt"
      ],
      "Resource": "AWS_TRANSCRIBE_OUTPUT_ENCRYPTION_KEY_ARN_VALUE"
    }
    ```

    !!! info "Replace Key ARN"
        Replace `AWS_TRANSCRIBE_OUTPUT_ENCRYPTION_KEY_ARN_VALUE` with the value of your [`AWS_TRANSCRIBE_OUTPUT_ENCRYPTION_KEY_ARN`](operations_configuration.md#aws-transcribe-output-encryption-key-arn) environment variable. The key policy must allow the same actions for this role; `kms:Decrypt` is what lets the finished transcript be read back.

---

## :material-earth: Language Detection (Optional)

**Environment Variables**: [`AWS_COMPREHEND_REGION`](operations_configuration.md#aws-comprehend-region)

Required for automatic language detection (used by TTS for voice selection).

??? example "Comprehend Language Detection IAM Policy Statement"
    ```json
    {
      "Sid": "ComprehendLanguageDetection",
      "Effect": "Allow",
      "Action": [
        "comprehend:DetectDominantLanguage"
      ],
      "Resource": "*"
    }
    ```

---

## :material-shield-alert: Comprehend Moderation (Optional) { #comprehend-moderation }

**Environment Variables**: [`AWS_COMPREHEND_REGION`](operations_configuration.md#aws-comprehend-region)

Required for the [Moderations API](api_openai_moderations.md) toxicity backend — the default backend when no Bedrock guardrail is configured, and always available as the `amazon.comprehend-toxicity` model.

??? example "Comprehend Moderation IAM Policy Statement"
    ```json
    {
      "Sid": "ComprehendModeration",
      "Effect": "Allow",
      "Action": [
        "comprehend:DetectToxicContent"
      ],
      "Resource": "*"
    }
    ```

---

## :material-translate: Text Translation (Optional)

**Environment Variables**: [`AWS_TRANSLATE_REGION`](operations_configuration.md#aws-translate-region)

Required for text translation features.

??? example "Translate Text Translation IAM Policy Statement"
    ```json
    {
      "Sid": "TranslateTextTranslation",
      "Effect": "Allow",
      "Action": [
        "translate:TranslateText",
        "translate:ListLanguages"
      ],
      "Resource": "*"
    }
    ```

    !!! note "`translate:ListLanguages`"
        Read once at startup, to refuse a language Amazon Translate does not support before the audio is transcribed. Without it translation still works: an unsupported language is then reported once the translation call itself fails.

---

## :material-cash-multiple: Cost Tracking (Optional) { #cost-tracking-iam }

**Environment Variables**: [`COST_TRACKING`](operations_configuration.md#cost-tracking)

Required for [`COST_TRACKING`](operations_configuration.md#cost-tracking) (disabled by default), which prices requests from the AWS Price List API. Without this permission the catalog stays empty and request logs carry no cost data.

??? example "Cost Tracking IAM Policy Statement"
    ```json
    {
      "Sid": "PricingCatalog",
      "Effect": "Allow",
      "Action": [
        "pricing:GetProducts"
      ],
      "Resource": "*"
    }
    ```

---

## :material-account-cash: Per-User Cost Attribution (Optional) { #per-user-cost-attribution }

**Environment Variables**: [`AWS_BEDROCK_USER_ROLE_ARN`](operations_configuration.md#aws-bedrock-user-role-arn)

Required to run each end user's model calls under a role session of their own, so AWS reports [their spend separately](operations_cost_management.md#per-user-attribution). Three policies are involved: the server's own role must be allowed to open the sessions, the end user role must trust it to do so, and the end user role must be allowed to invoke models.

**1. On the server's role** — allow it to open sessions of the end user role, and of that role only:

??? example "Server Role IAM Policy Statement"
    ```json
    {
      "Sid": "EndUserRoleSessions",
      "Effect": "Allow",
      "Action": [
        "sts:AssumeRole",
        "sts:TagSession"
      ],
      "Resource": "arn:aws:iam::ACCOUNT_ID:role/stdapi-ai-end-user"
    }
    ```

**2. Trust policy of the end user role** — allow the server's role, and nothing else, to assume it *and* to tag the session. `sts:TagSession` is a separate action: without it, every session that carries the end user tag is denied.

??? example "End User Role Trust Policy"
    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Effect": "Allow",
          "Principal": {
            "AWS": "arn:aws:iam::ACCOUNT_ID:role/stdapi-ai-task-role"
          },
          "Action": [
            "sts:AssumeRole",
            "sts:TagSession"
          ]
        }
      ]
    }
    ```

**3. Permission policy of the end user role** — everything AWS authorizes against the caller of a model invocation: the invocation actions on the models the deployment serves, and the guardrail the invocation carries:

??? example "End User Role IAM Policy Statements"
    ```json
    {
      "Sid": "EndUserModelInvoke",
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": [
        "arn:aws:bedrock:*:ACCOUNT_ID:inference-profile/*",
        "arn:aws:bedrock:*:ACCOUNT_ID:application-inference-profile/*",
        "arn:aws:bedrock:*:ACCOUNT_ID:default-prompt-router/*",
        "arn:aws:bedrock:*::foundation-model/*"
      ]
    },
    {
      "Sid": "EndUserApplyGuardrail",
      "Effect": "Allow",
      "Action": [
        "bedrock:ApplyGuardrail"
      ],
      "Resource": "arn:aws:bedrock:*:ACCOUNT_ID:guardrail/*"
    }
    ```

Replace `ACCOUNT_ID` with your AWS account ID, and `stdapi-ai-task-role` with the role the server runs as.

!!! warning "An inference profile needs the foundation models behind it"
    A cross-region inference profile routes to a foundation model in each of its Regions, and AWS authorizes **both** the profile ARN and every foundation model ARN it reaches. A policy naming only `inference-profile/...` fails with an access-denied error naming `foundation-model/...` in a Region you never configured. Keep `arn:aws:bedrock:*::foundation-model/...` alongside the profile, or the call is denied.

!!! warning "A configured guardrail is authorized against the end user"
    A guardrail applied **during** an invocation — [`AWS_BEDROCK_GUARDRAIL_IDENTIFIER`](operations_configuration.md#aws-bedrock-guardrail-identifier), a model alias carrying one, or a request-level `moderation` parameter — is evaluated as part of that invocation, so AWS requires `bedrock:ApplyGuardrail` from the identity making the call. Without the `EndUserApplyGuardrail` statement, every model request fails with an access-denied error as soon as per-user attribution is enabled. See [Set up permissions to use Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-permissions.html). Narrow the resource to your guardrail ARN if you prefer.

!!! note "Name every model ARN form the deployment allows"
    The `Resource` list must cover every ARN a request can resolve to. Add `arn:aws:bedrock:*:ACCOUNT_ID:prompt/*` when [`AWS_BEDROCK_ALLOW_PROMPT_ARN`](operations_configuration.md#bedrock-allow-prompt-arn) is enabled, and keep the application inference profile and prompt router entries above whenever [`AWS_BEDROCK_ALLOW_APPLICATION_INFERENCE_PROFILE_ARN`](operations_configuration.md#bedrock-allow-application-profile-arn), [`AWS_BEDROCK_ALLOW_PROMPT_ROUTER_ARN`](operations_configuration.md#bedrock-allow-prompt-router-arn) or [`AWS_BEDROCK_MODEL_ARN_MAPPING`](operations_configuration.md#bedrock-model-arn-mapping) can put one in front of a model. An ARN form the end user role does not name is denied under it while it still works on the server's role.

    Add the [Web Search](#web-search-iam) actions to this role as well if you serve `web_search` requests: AWS evaluates them when the model actually runs a search, which happens inside the invocation the end user signed. A denied search does not fail the request, it degrades the answer.

!!! danger "A session tag is an access boundary only when the identity is verified"
    The end user identity is taken from the authenticated caller only under [Amazon Cognito authentication](operations_authentication_security.md). With an API key, or with no authentication, it is whatever the client declared in the request body (`safety_identifier`, `user`, `metadata.user_id`) — so any caller holding the key can send another user's identifier and obtain that user's session tag.

    Write policies conditioned on `aws:PrincipalTag/<key>` only when [`AUTHENTICATION_MODE`](operations_configuration.md#authentication-mode) is `cognito`, which is the configuration where every request carries an identity the gateway verified. Anywhere else, treat the tag as cost metadata, never as an authorization input.

!!! tip "Restricting a role per end user"
    Where the identity is verified, the session tag makes it testable in a policy: compare it to something on the resource side, so each session reaches only its own data — `"StringEquals": {"aws:ResourceTag/user": "${aws:PrincipalTag/user}"}`, an `s3:prefix` condition, or a `Resource` ARN embedding `${aws:PrincipalTag/user}`. A condition comparing the tag to itself always matches and restricts nothing. A `Deny` on any tag value the deployment does not expect is the other half of the same pattern. Set [`AWS_BEDROCK_USER_ROLE_TAG_KEY`](operations_configuration.md#aws-bedrock-user-role-tag-key) to the key the policy tests.

!!! note "Scope"
    Only Bedrock model invocations run under the end user role, together with the guardrail applied during them. Standalone guardrail evaluations (the [Moderations API](api_openai_moderations.md)), reranking, video generation and its output files, speech, transcription and translation keep the server's own role, so the end user role needs none of their permissions — and the server's role still needs all of them.

---

## :material-key: API Key Authentication (Optional)

Required if you configure API authentication. See the [Authentication](operations_configuration.md#authentication) configuration section.

### SSM Parameter Store

**Environment Variables**: [`API_KEY_SSM_PARAMETER`](operations_configuration.md#api-key-ssm)

??? example "SSM Parameter Store IAM Policy Statements"
    ```json
    {
      "Sid": "SSMParameterAccess",
      "Effect": "Allow",
      "Action": [
        "ssm:GetParameter"
      ],
      "Resource": "arn:aws:ssm:REGION:ACCOUNT_ID:parameter/API_KEY_SSM_PARAMETER_VALUE"
    }
    ```

    !!! info "Replace Parameter Path"
        Replace `API_KEY_SSM_PARAMETER_VALUE` with the value of your [`API_KEY_SSM_PARAMETER`](operations_configuration.md#api-key-ssm) environment variable (e.g., `/stdapi/prod/api-key`), and `REGION` with the server's own region (`AWS_REGION`) — parameters are read there, not in the Bedrock regions.

    **If using encrypted SSM parameters**, also add:

    ```json
    {
      "Sid": "KMSDecryptionForSSM",
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt"
      ],
      "Resource": "arn:aws:kms:REGION:ACCOUNT_ID:key/YOUR_KMS_KEY_ID",
      "Condition": {
        "StringEquals": {
          "kms:ViaService": "ssm.REGION.amazonaws.com"
        }
      }
    }
    ```

    !!! tip "KMS Security"
        The `kms:ViaService` condition restricts KMS key usage to SSM service calls only.

### Secrets Manager

**Environment Variables**: [`API_KEY_SECRETSMANAGER_SECRET`](operations_configuration.md#api-key-secretsmanager-secret)

??? example "Secrets Manager IAM Policy Statement"
    ```json
    {
      "Sid": "SecretsManagerAccess",
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue"
      ],
      "Resource": "arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:API_KEY_SECRETSMANAGER_SECRET_VALUE"
    }
    ```

    !!! info "Replace Secret Name"
        Replace `API_KEY_SECRETSMANAGER_SECRET_VALUE` with the value of your [`API_KEY_SECRETSMANAGER_SECRET`](operations_configuration.md#api-key-secretsmanager-secret) environment variable (e.g., `stdapi-api-key`), and `REGION` with the server's own region (`AWS_REGION`) — secrets are read there, not in the Bedrock regions.

---

## :material-file-document: Complete Policy Examples

??? example "Minimal Policy (Bedrock Only)"
    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Sid": "BedrockModelInvoke",
          "Effect": "Allow",
          "Action": [
            "bedrock:CountTokens",
            "bedrock:GetAsyncInvoke",
            "bedrock:InvokeGuardrailChecks",
            "bedrock:InvokeModel",
            "bedrock:InvokeModelWithBidirectionalStream",
            "bedrock:InvokeModelWithResponseStream",
            "bedrock:InvokeTool",
            "bedrock:Rerank"
          ],
          "Resource": "*"
        },
        {
          "Sid": "BedrockAsyncInvokeTagging",
          "Effect": "Allow",
          "Action": [
            "bedrock:TagResource"
          ],
          "Resource": "arn:aws:bedrock:*:*:async-invoke/*"
        },
        {
          "Sid": "BedrockModelDiscovery",
          "Effect": "Allow",
          "Action": [
            "bedrock:ListFoundationModels",
            "bedrock:GetFoundationModelAvailability",
            "bedrock:ListProvisionedModelThroughputs",
            "bedrock:ListInferenceProfiles"
          ],
          "Resource": "*"
        },
        {
          "Sid": "BedrockMarketplaceAutoSubscribe",
          "Effect": "Allow",
          "Action": [
            "aws-marketplace:Subscribe",
            "aws-marketplace:ViewSubscriptions"
          ],
          "Resource": "*"
        },
        {
          "Sid": "MarketplaceRegisterUsage",
          "Effect": "Allow",
          "Action": [
            "aws-marketplace:RegisterUsage"
          ],
          "Resource": "*"
        }
      ]
    }
    ```

    !!! note "Marketplace Auto-Subscribe (Default Enabled)"
        The marketplace permissions are included because `AWS_BEDROCK_MARKETPLACE_AUTO_SUBSCRIBE` defaults to `true`. If you set it to `false`, you can remove the `BedrockMarketplaceAutoSubscribe` statement.

    !!! note "Marketplace RegisterUsage (AWS Marketplace Image Only)"
        `MarketplaceRegisterUsage` is only needed on the **AWS Marketplace image**; remove it when deploying the community image.

??? example "Production Policy (Bedrock + S3 + Authentication)"
    ```json
    {
      "Version": "2012-10-17",
      "Statement": [
        {
          "Sid": "BedrockModelInvoke",
          "Effect": "Allow",
          "Action": [
            "bedrock:CountTokens",
            "bedrock:GetAsyncInvoke",
            "bedrock:InvokeGuardrailChecks",
            "bedrock:InvokeModel",
            "bedrock:InvokeModelWithBidirectionalStream",
            "bedrock:InvokeModelWithResponseStream",
            "bedrock:InvokeTool",
            "bedrock:Rerank"
          ],
          "Resource": "*"
        },
        {
          "Sid": "BedrockAsyncInvokeTagging",
          "Effect": "Allow",
          "Action": [
            "bedrock:TagResource"
          ],
          "Resource": "arn:aws:bedrock:*:*:async-invoke/*"
        },
        {
          "Sid": "BedrockModelDiscovery",
          "Effect": "Allow",
          "Action": [
            "bedrock:ListFoundationModels",
            "bedrock:GetFoundationModelAvailability",
            "bedrock:ListProvisionedModelThroughputs",
            "bedrock:ListInferenceProfiles"
          ],
          "Resource": "*"
        },
        {
          "Sid": "BedrockMarketplaceAutoSubscribe",
          "Effect": "Allow",
          "Action": [
            "aws-marketplace:Subscribe",
            "aws-marketplace:ViewSubscriptions"
          ],
          "Resource": "*"
        },
        {
          "Sid": "MarketplaceRegisterUsage",
          "Effect": "Allow",
          "Action": [
            "aws-marketplace:RegisterUsage"
          ],
          "Resource": "*"
        },
        {
          "Sid": "S3FileStorage",
          "Effect": "Allow",
          "Action": [
            "s3:PutObject",
            "s3:PutObjectTagging",
            "s3:GetObject",
            "s3:DeleteObject",
            "s3:AbortMultipartUpload",
            "s3:ListMultipartUploadParts"
          ],
          "Resource": "arn:aws:s3:::my-stdapi-bucket/*"
        },
        {
          "Sid": "S3FileStorageList",
          "Effect": "Allow",
          "Action": [
            "s3:ListBucket",
            "s3:ListBucketMultipartUploads"
          ],
          "Resource": "arn:aws:s3:::my-stdapi-bucket"
        },
        {
          "Sid": "SSMParameterAccess",
          "Effect": "Allow",
          "Action": [
            "ssm:GetParameter"
          ],
          "Resource": "arn:aws:ssm:us-east-1:123456789012:parameter/stdapi/prod/api-key"
        },
        {
          "Sid": "PricingCatalog",
          "Effect": "Allow",
          "Action": [
            "pricing:GetProducts"
          ],
          "Resource": "*"
        }
      ]
    }
    ```

    !!! note "Marketplace Auto-Subscribe (Default Enabled)"
        The marketplace permissions are included because `AWS_BEDROCK_MARKETPLACE_AUTO_SUBSCRIBE` defaults to `true`. If you set it to `false`, you can remove the `BedrockMarketplaceAutoSubscribe` statement to follow the principle of least privilege.

    !!! note "Marketplace RegisterUsage (AWS Marketplace Image Only)"
        `MarketplaceRegisterUsage` is only needed on the **AWS Marketplace image**; remove it when deploying the community image.

    !!! note "Cost Tracking (Opt-In)"
        `PricingCatalog` is only needed when [`COST_TRACKING`](operations_configuration.md#cost-tracking) is set to `true`; remove it otherwise.

---

## :material-table: Feature-Specific Permission Requirements

| Feature                                         | Required Permissions                                                                                                                                       | Configuration                                                                |
|-------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| **Bedrock Models (Invoke)**                     | `bedrock:CountTokens`<br>`bedrock:InvokeGuardrailChecks`<br>`bedrock:InvokeModel`<br>`bedrock:InvokeModelWithBidirectionalStream`<br>`bedrock:InvokeModelWithResponseStream`<br>`bedrock:InvokeTool` (the Amazon Nova grounding server tool)<br>`bedrock:Rerank`<br>`bedrock:GetAsyncInvoke` and `bedrock:TagResource` (on `arn:aws:bedrock:*:*:async-invoke/*`) for async-invoke models (video, TwelveLabs Marengo embeddings) | Always required                                                              |
| **Realtime API**                                | `bedrock:InvokeModelWithBidirectionalStream` (already part of the core Bedrock policy above); no additional action | `POST /v1/realtime/client_secrets`, `WS /v1/realtime`                        |
| **Bedrock Models (Discovery)**                  | `bedrock:ListFoundationModels`<br>`bedrock:GetFoundationModelAvailability`<br>`bedrock:ListProvisionedModelThroughputs`<br>`bedrock:ListInferenceProfiles` | Always required                                                              |
| **Bedrock Marketplace Auto-Subscribe**          | `aws-marketplace:Subscribe`<br>`aws-marketplace:ViewSubscriptions`                                                                                         | `AWS_BEDROCK_MARKETPLACE_AUTO_SUBSCRIBE=true` (default)                      |
| **AWS Marketplace Metering**                    | `aws-marketplace:RegisterUsage`                                                                                                                             | AWS Marketplace image only (always active); not required for the community image |
| **Bedrock Inference Profiles & Prompt Routers** | `bedrock:GetInferenceProfile`<br>`bedrock:GetPromptRouter`<br>`bedrock:GetPrompt` and `bedrock:RenderPrompt` (on `arn:aws:bedrock:*:*:prompt/*`) for Prompt Management prompts | `AWS_BEDROCK_ALLOW_*_ARN=true` or `AWS_BEDROCK_MODEL_ARN_MAPPING` configured |
| **Bedrock Guardrails & Moderations**            | `bedrock:ApplyGuardrail`                                                                                                                                   | `AWS_BEDROCK_GUARDRAIL_IDENTIFIER`                                           |
| **Stored Responses & Chat Completions**         | Bedrock session permissions (`bedrock:CreateSession`, `bedrock:GetSession`, `bedrock:*Invocation*`, `bedrock:ListSessions`, `bedrock:EndSession`, `bedrock:DeleteSession`, `bedrock:TagResource`, `bedrock:ListTagsForResource` on sessions) | `store=true` requests and stored-completion listings                         |
| **Bedrock Mantle**                              | `bedrock-mantle:CreateInference`<br>`bedrock-mantle:GetInference`<br>`bedrock-mantle:DeleteInference`<br>`bedrock-mantle:ListModels`<br>`bedrock-mantle:GetModel`<br>`bedrock-mantle:CancelInference` (on `arn:aws:bedrock-mantle:*:*:project/*`)<br>`bedrock-mantle:CallWithBearerToken` | `AWS_BEDROCK_MANTLE_ENABLED=true`                                            |
| **Web Search**                                  | `bedrock-websearch:InvokeSearch`<br>`bedrock-websearch:InvokeFetch`<br>`bedrock-websearch:ExternalWebAccess` only when external web access is enabled | `web_search` requests on the OpenAI GPT-5.x family                           |
| **File Storage**                                | `s3:PutObject`<br>`s3:PutObjectTagging`<br>`s3:GetObject`<br>`s3:DeleteObject`<br>`s3:AbortMultipartUpload`<br>`s3:ListMultipartUploadParts`<br>`s3:ListBucket`<br>`s3:ListBucketMultipartUploads`<br>on every bucket, including each `AWS_S3_REGIONAL_BUCKETS` entry | `AWS_S3_BUCKET`<br>`AWS_S3_REGIONAL_BUCKETS`                                 |
| **Video Generation**                            | Core Bedrock invoke permissions (incl. `bedrock:GetAsyncInvoke`, `bedrock:TagResource`)<br>`bedrock:ListAsyncInvokes` and `bedrock:ListTagsForResource` (on `arn:aws:bedrock:*:*:async-invoke/*`) for job listing<br>File Storage S3 permissions on each regional bucket | `AWS_S3_REGIONAL_BUCKETS`                                                    |
| **Batch Inference**                             | `bedrock:CreateModelInvocationJob`<br>`bedrock:GetModelInvocationJob`<br>`bedrock:StopModelInvocationJob` (on `arn:aws:bedrock:*:*:model-invocation-job/*`)<br>`iam:PassRole` on the batch service role, scoped with `iam:PassedToService: bedrock.amazonaws.com`<br>File Storage S3 permissions on each bucket a batch uses, plus the service role's own policy (see [Batch Inference](#batch-inference)) | `AWS_BEDROCK_BATCH_ROLE_ARN`                                                 |
| **Vector Stores**                               | `s3vectors:CreateIndex`<br>`s3vectors:DeleteIndex`<br>`s3vectors:PutVectors`<br>`s3vectors:GetVectors`<br>`s3vectors:QueryVectors`<br>`s3vectors:DeleteVectors` (on the vector bucket and its indexes)<br>File Storage S3 permissions on `AWS_S3_BUCKET` for the stores' records | `AWS_S3_VECTORS_BUCKET`<br>`AWS_S3_VECTORS_REGION`                           |
| **Durable Vector Store Indexing**               | `sqs:SendMessage`<br>`sqs:ReceiveMessage`<br>`sqs:DeleteMessage`<br>`sqs:ChangeMessageVisibility`<br>`sqs:GetQueueAttributes` (on the queue ARN only)                                                       | `AWS_SQS_VECTOR_STORE_QUEUE_URL`                                             |
| **Knowledge Base Vector Stores**                | `bedrock:GetKnowledgeBase`<br>`bedrock:Retrieve`<br>`bedrock:ListDataSources`<br>`bedrock:IngestKnowledgeBaseDocuments`<br>`bedrock:ListKnowledgeBaseDocuments`<br>`bedrock:GetKnowledgeBaseDocuments`<br>`bedrock:DeleteKnowledgeBaseDocuments` (on each allowlisted knowledge base ARN; no `bedrock:ListKnowledgeBases`) | `AWS_BEDROCK_KNOWLEDGE_BASE_IDS`                                             |
| **KMS Encrypted S3 Buckets**                    | `kms:Decrypt`<br>`kms:GenerateDataKey`<br>with `kms:ViaService` condition                                                                                  | If S3 buckets use KMS encryption                                             |
| **Text-to-Speech**                              | `polly:SynthesizeSpeech`<br>`polly:DescribeVoices`<br>`polly:StartSpeechSynthesisStream` for generative voices above 3,000 characters<br>`polly:StartSpeechSynthesisTask`, `polly:GetSpeechSynthesisTask` and S3 `PutObject`/`GetObject`/`DeleteObject` on each bucket serving a Polly region, for the other voices above 3,000 characters | `AWS_POLLY_REGION`<br>`AWS_S3_BUCKET`<br>`AWS_S3_REGIONAL_BUCKETS`           |
| **Speech-to-Text**                              | `transcribe:StartTranscriptionJob`<br>`transcribe:GetTranscriptionJob`<br>`transcribe:DeleteTranscriptionJob`<br>`transcribe:StartStreamTranscription`<br>`transcribe:TagResource` (on `arn:aws:transcribe:*:*:transcription-job/*`)<br>File Storage S3 permissions on every bucket serving a candidate region<br>`kms:GenerateDataKey`, `kms:Decrypt` on the output encryption key, when one is configured | `AWS_TRANSCRIBE_REGION`<br>`AWS_TRANSCRIBE_S3_BUCKET`<br>`AWS_S3_REGIONAL_BUCKETS`<br>`AWS_TRANSCRIBE_STREAM_LANGUAGES`<br>`AWS_TRANSCRIBE_OUTPUT_ENCRYPTION_KEY_ARN` |
| **Language Detection**                          | `comprehend:DetectDominantLanguage`                                                                                                                        | `AWS_COMPREHEND_REGION`                                                      |
| **Comprehend Moderations**                      | `comprehend:DetectToxicContent`                                                                                                                            | Moderations API without a configured guardrail                              |
| **Translation**                                 | `translate:TranslateText`<br>`translate:ListLanguages` (optional; validates the language pair before transcribing)                                          | `AWS_TRANSLATE_REGION`                                                       |
| **Cost Tracking**                               | `pricing:GetProducts`                                                                                                                                      | `COST_TRACKING=true` (opt-in; `false` by default)                            |
| **Per-User Cost Attribution**                   | `sts:AssumeRole` and `sts:TagSession` on the end user role, matched by that role's trust policy; on the end user role itself, `bedrock:InvokeModel`, `bedrock:InvokeModelWithResponseStream` on every model ARN form the deployment allows, plus `bedrock:ApplyGuardrail` when a guardrail is configured (see [Per-User Cost Attribution](#per-user-cost-attribution)) | `AWS_BEDROCK_USER_ROLE_ARN`                                                  |
| **SSM Parameter Store**                         | `ssm:GetParameter`<br>`kms:Decrypt` (if encrypted)                                                                                                         | `API_KEY_SSM_PARAMETER`                                                      |
| **Secrets Manager**                             | `secretsmanager:GetSecretValue`                                                                                                                            | `API_KEY_SECRETSMANAGER_SECRET`                                              |

---

## :material-account: IAM Role vs. IAM User

stdapi.ai supports both IAM roles and IAM users:

- **:material-aws: IAM Role (Recommended)**: Use when running on EC2, ECS, Lambda, or other AWS compute services. Attach the policy to the instance/task role.
- **:material-account: IAM User**: Use when running outside AWS or for development. Create an IAM user with the required permissions and configure AWS credentials via environment variables or AWS CLI configuration.

!!! success "Best Practice: Use IAM Roles"
    When deploying on AWS infrastructure, always prefer IAM roles over IAM users with access keys. IAM roles provide automatic credential rotation and better security.

---

## :material-tag: AWS Tag Policies

If your AWS organization enforces a [tag policy](https://docs.aws.amazon.com/organizations/latest/userguide/orgs_manage_policies_tag-policies.html), the following tag keys must be allowed on the relevant resource types.

| Tag key | Value | Applied to |
|---------|-------|------------|
| `stdapi-ai.expires` | `"true"` | S3 objects (Files API expiry) |
| `stdapi-ai.request_id` | request UUID | Bedrock async jobs, Transcribe jobs |
| `stdapi-ai.server_id` | server instance name | Bedrock async jobs, Transcribe jobs |
| `stdapi-ai.user_id` | user identifier | Bedrock async jobs, Transcribe jobs (when user identity is known) |
| `aws-apn-id` | `pc:<product-code>` | All AWS resources created at runtime and by the Terraform module. This is a [standard AWS Marketplace attribution tag](https://docs.aws.amazon.com/PRM/latest/aws-prm-onboarding-guide/what-is-service.html) required by any AWS Marketplace product — allowing it benefits all such products deployed in your organization, not only stdapi.ai. |

---

## :material-arrow-right: Next Steps

<div class="grid cards" markdown>

- :material-cog: [**Configuration Reference**](operations_configuration.md) — Environment variables for every feature
- :material-lock: [**Authentication & Security**](operations_authentication_security.md) — Secure your deployment
- :material-server-network: [**Advanced Deployment**](operations_deploy_advanced.md) — Terraform infrastructure examples
- :material-help-circle: [**Troubleshooting**](operations_troubleshooting.md) — Diagnose permission errors

</div>
