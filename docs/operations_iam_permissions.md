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

---

## :material-storefront: Bedrock Marketplace Auto-Subscribe (Optional) { #bedrock-marketplace-auto-subscribe-iam }

**Environment Variables**: [`AWS_BEDROCK_MARKETPLACE_AUTO_SUBSCRIBE`](operations_configuration.md#bedrock-marketplace-auto-subscribe)

Required only if you want to enable automatic subscription to new models in the AWS Marketplace (`AWS_BEDROCK_MARKETPLACE_AUTO_SUBSCRIBE=true`, which is the default). When enabled, the server can automatically subscribe to marketplace offerings for newly discovered models.

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

Required only if clients use `store=true` on the [Responses](api_openai_responses.md#stored-responses) or [Chat Completions](api_openai_chat_completions.md#stored-chat-completions) APIs, which persist generations in Amazon Bedrock sessions.

??? example "Bedrock Session Storage IAM Policy Statement"
    ```json
    {
      "Sid": "BedrockSessionStorage",
      "Effect": "Allow",
      "Action": [
        "bedrock:CreateSession",
        "bedrock:GetSession",
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

    `bedrock:ListSessions` serves the stored chat completions listing endpoint (`GET /v1/chat/completions`); the account-level `ListSessions` action does not support resource scoping. `bedrock:GetSession` is used on deletion and `bedrock:ListTagsForResource` on both deletion and listing, to check that a stored object belongs to the API it is requested from.

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
        Replace `AWS_S3_BUCKET_VALUE` with the value of your [`AWS_S3_BUCKET`](operations_configuration.md#aws-s3-bucket) environment variable. Repeat both statements for each [`AWS_S3_REGIONAL_BUCKETS`](operations_configuration.md#aws-s3-regional-buckets) bucket — they serve video generation, asynchronous embeddings (TwelveLabs Marengo, Amazon Nova), and [Speech-to-Text](#speech-to-text-optional) failover, and the Files API looks up objects across every configured bucket.

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

## :material-account-voice: Text-to-Speech (Optional)

**Environment Variables**: [`AWS_POLLY_REGION`](operations_configuration.md#aws-polly-region), [`DEFAULT_TTS_MODEL`](operations_configuration.md#default-tts-model), [`DEFAULT_TTS_LANGUAGE`](operations_configuration.md#default-tts-language)

Required for generating speech from text using Amazon Polly. See the [Audio and Text-to-Speech](operations_configuration.md#audio-and-text-to-speech) configuration section.

!!! tip "Optimize Performance"
    Set [`DEFAULT_TTS_LANGUAGE`](operations_configuration.md#default-tts-language) to skip language detection and avoid Amazon Comprehend API calls, improving response times and reducing costs.

??? example "Polly Text-to-Speech IAM Policy Statement"
    ```json
    {
      "Sid": "PollyTextToSpeech",
      "Effect": "Allow",
      "Action": [
        "polly:SynthesizeSpeech",
        "polly:DescribeVoices"
      ],
      "Resource": "*"
    }
    ```

---

## :material-microphone: Speech-to-Text (Optional) { #speech-to-text-optional }

**Environment Variables**: [`AWS_TRANSCRIBE_REGION`](operations_configuration.md#aws-transcribe-region), [`AWS_TRANSCRIBE_S3_BUCKET`](operations_configuration.md#aws-transcribe-s3-bucket), [`AWS_S3_REGIONAL_BUCKETS`](operations_configuration.md#aws-s3-regional-buckets)

Required for transcribing audio files using Amazon Transcribe. Each transcription job stages its audio in a bucket co-located with the Transcribe endpoint, so the S3 statement must cover every bucket that serves a candidate region.

??? example "Transcribe Speech-to-Text IAM Policy Statements"
    ```json
    {
      "Sid": "TranscribeSpeechToText",
      "Effect": "Allow",
      "Action": [
        "transcribe:StartTranscriptionJob",
        "transcribe:GetTranscriptionJob",
        "transcribe:DeleteTranscriptionJob"
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

    **If your transcribe S3 buckets use KMS encryption**, also add the KMS permissions for each bucket's key, with that region's `kms:ViaService` value.

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
        "translate:TranslateText"
      ],
      "Resource": "*"
    }
    ```

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
| **Bedrock Models (Invoke)**                     | `bedrock:CountTokens`<br>`bedrock:InvokeGuardrailChecks`<br>`bedrock:InvokeModel`<br>`bedrock:InvokeModelWithResponseStream`<br>`bedrock:InvokeTool`<br>`bedrock:Rerank`<br>`bedrock:GetAsyncInvoke` and `bedrock:TagResource` (on `arn:aws:bedrock:*:*:async-invoke/*`) for async-invoke models (video, TwelveLabs Marengo embeddings) | Always required                                                              |
| **Bedrock Models (Discovery)**                  | `bedrock:ListFoundationModels`<br>`bedrock:GetFoundationModelAvailability`<br>`bedrock:ListProvisionedModelThroughputs`<br>`bedrock:ListInferenceProfiles` | Always required                                                              |
| **Bedrock Marketplace Auto-Subscribe**          | `aws-marketplace:Subscribe`<br>`aws-marketplace:ViewSubscriptions`                                                                                         | `AWS_BEDROCK_MARKETPLACE_AUTO_SUBSCRIBE=true` (default)                      |
| **AWS Marketplace Metering**                    | `aws-marketplace:RegisterUsage`                                                                                                                             | AWS Marketplace image only (always active); not required for the community image |
| **Bedrock Inference Profiles & Prompt Routers** | `bedrock:GetInferenceProfile`<br>`bedrock:GetPromptRouter`<br>`bedrock:GetPrompt` and `bedrock:RenderPrompt` (on `arn:aws:bedrock:*:*:prompt/*`) for Prompt Management prompts | `AWS_BEDROCK_ALLOW_*_ARN=true` or `AWS_BEDROCK_MODEL_ARN_MAPPING` configured |
| **Bedrock Guardrails & Moderations**            | `bedrock:ApplyGuardrail`                                                                                                                                   | `AWS_BEDROCK_GUARDRAIL_IDENTIFIER`                                           |
| **Stored Responses & Chat Completions**         | Bedrock session permissions (`bedrock:CreateSession`, `bedrock:GetSession`, `bedrock:*Invocation*`, `bedrock:ListSessions`, `bedrock:EndSession`, `bedrock:DeleteSession`, `bedrock:TagResource`, `bedrock:ListTagsForResource` on sessions) | `store=true` requests and stored-completion listings                         |
| **Bedrock Mantle**                              | `bedrock-mantle:CreateInference`<br>`bedrock-mantle:GetInference`<br>`bedrock-mantle:DeleteInference`<br>`bedrock-mantle:ListModels`<br>`bedrock-mantle:GetModel`<br>`bedrock-mantle:CancelInference` (on `arn:aws:bedrock-mantle:*:*:project/*`)<br>`bedrock-mantle:CallWithBearerToken` | `AWS_BEDROCK_MANTLE_ENABLED=true`                                            |
| **File Storage**                                | `s3:PutObject`<br>`s3:PutObjectTagging`<br>`s3:GetObject`<br>`s3:DeleteObject`<br>`s3:AbortMultipartUpload`<br>`s3:ListMultipartUploadParts`<br>`s3:ListBucket`<br>`s3:ListBucketMultipartUploads`<br>on every bucket, including each `AWS_S3_REGIONAL_BUCKETS` entry | `AWS_S3_BUCKET`<br>`AWS_S3_REGIONAL_BUCKETS`                                 |
| **Video Generation**                            | Core Bedrock invoke permissions (incl. `bedrock:GetAsyncInvoke`, `bedrock:TagResource`)<br>`bedrock:ListAsyncInvokes` and `bedrock:ListTagsForResource` (on `arn:aws:bedrock:*:*:async-invoke/*`) for job listing<br>File Storage S3 permissions on each regional bucket | `AWS_S3_REGIONAL_BUCKETS`                                                    |
| **KMS Encrypted S3 Buckets**                    | `kms:Decrypt`<br>`kms:GenerateDataKey`<br>with `kms:ViaService` condition                                                                                  | If S3 buckets use KMS encryption                                             |
| **Text-to-Speech**                              | `polly:SynthesizeSpeech`<br>`polly:DescribeVoices`                                                                                                         | `AWS_POLLY_REGION`                                                           |
| **Speech-to-Text**                              | `transcribe:StartTranscriptionJob`<br>`transcribe:GetTranscriptionJob`<br>`transcribe:DeleteTranscriptionJob`<br>`transcribe:TagResource` (on `arn:aws:transcribe:*:*:transcription-job/*`)<br>File Storage S3 permissions on every bucket serving a candidate region | `AWS_TRANSCRIBE_REGION`<br>`AWS_TRANSCRIBE_S3_BUCKET`<br>`AWS_S3_REGIONAL_BUCKETS` |
| **Language Detection**                          | `comprehend:DetectDominantLanguage`                                                                                                                        | `AWS_COMPREHEND_REGION`                                                      |
| **Comprehend Moderations**                      | `comprehend:DetectToxicContent`                                                                                                                            | Moderations API without a configured guardrail                              |
| **Translation**                                 | `translate:TranslateText`                                                                                                                                  | `AWS_TRANSLATE_REGION`                                                       |
| **Cost Tracking**                               | `pricing:GetProducts`                                                                                                                                      | `COST_TRACKING=true` (opt-in; `false` by default)                            |
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
