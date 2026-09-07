---
title: "Configuration - Bedrock Features"
description: "Configure the Amazon Bedrock features stdapi.ai exposes: guardrails, session storage, service tiers and latency optimization, text-to-speech, the Realtime API and inference profile, prompt router and prompt ARNs."
keywords: "Bedrock guardrails, Bedrock session storage, service tier, latency optimized, text to speech, Realtime API, inference profile ARN, prompt router ARN, application inference profile"
---

# :material-shield-check: Bedrock Features

The Amazon Bedrock capabilities the gateway turns on for you, and the ARNs it is allowed to invoke. Part of the [Configuration Guide](operations_configuration.md).

## :material-format-list-bulleted: Settings Summary

### :material-shield-check: Bedrock Advanced { #summary-bedrock-advanced }

| Variable                                                                                          | Default | Description                                                                                         |
|---------------------------------------------------------------------------------------------------|---------|-----------------------------------------------------------------------------------------------------|
| [`AWS_BEDROCK_CROSS_REGION_INFERENCE`](operations_configuration_aws.md#cross-region-inference)                                   | `true`  | Allow automatic model routing to other configured regions                                           |
| [`AWS_BEDROCK_CROSS_REGION_INFERENCE_GLOBAL`](operations_configuration_aws.md#cross-region-global)                               | `true`  | Allow global cross-region inference routing to any region worldwide (disable for GDPR compliance)   |
| [`AWS_BEDROCK_MODEL_REGION_RESTRICT`](operations_configuration_aws.md#bedrock-model-region-restrict)                             | `{}`    | Restrict a model to specific region(s) only (e.g. for region-specific features like Nova grounding) |
| [`AWS_BEDROCK_LEGACY`](operations_configuration_models.md#bedrock-legacy)                                                           | `false` | Allow usage of deprecated/legacy Bedrock models                                                     |
| [`AWS_BEDROCK_DEPRECATED_MODEL_FALLBACK`](operations_configuration_models.md#bedrock-deprecated-model-fallback)                     | `true`  | Transparently reroute requests using a deprecated model ID to its recommended replacement           |
| [`AWS_BEDROCK_DEPRECATED_MODELS`](operations_configuration_models.md#bedrock-deprecated-models)                                     | `{}`    | Additional deprecated model mappings merged with the built-in registry at startup                   |
| [`AWS_BEDROCK_MARKETPLACE_AUTO_SUBSCRIBE`](operations_configuration_models.md#bedrock-marketplace-auto-subscribe)                   | `true`  | Allow automatic subscription to new models in AWS Marketplace                                       |
| [`AWS_BEDROCK_MARKETPLACE_ENDPOINTS_ENABLED`](operations_configuration_models.md#bedrock-marketplace-endpoints-enabled)             | `false` | Publish Amazon Bedrock Marketplace model endpoints deployed in this account as chat models           |
| [`AWS_BEDROCK_MARKETPLACE_ENDPOINT_REGIONS`](operations_configuration_models.md#bedrock-marketplace-endpoint-regions)                | All `AWS_BEDROCK_REGIONS` | Regions searched for Marketplace model endpoints; each must also be in `AWS_BEDROCK_REGIONS`  |
| [`AWS_BEDROCK_ALLOW_MARKETPLACE_ENDPOINT_ARN`](operations_configuration_models.md#bedrock-allow-marketplace-endpoint-arn)           | `false` | Allow users to pass a Marketplace model endpoint ARN directly as a model ID                         |
| [`AWS_SAGEMAKER_ENDPOINTS`](operations_configuration_models.md#aws-sagemaker-endpoints)                                             | `{}`    | Amazon SageMaker AI endpoints published as chat models, keyed by model ID                           |
| [`AWS_SAGEMAKER_WARMUP_TIMEOUT`](operations_configuration_models.md#aws-sagemaker-warmup-timeout)                                   | `600`   | Seconds a request waits for a SageMaker AI endpoint scaled to zero to come back up (`0` disables)    |
| [`AWS_SAGEMAKER_ENDPOINT_URL`](operations_configuration_models.md#aws-sagemaker-endpoint-url)                                       | Resolved | Override for the SageMaker AI runtime endpoint URL (VPC endpoint or proxy)                          |
| [`AWS_BEDROCK_ALLOW_CROSS_REGION_INFERENCE_PROFILE_ARN`](#bedrock-allow-cross-region-profile-arn) | `false` | Allow users to pass cross-region inference profile ARNs directly as model IDs                       |
| [`AWS_BEDROCK_ALLOW_APPLICATION_INFERENCE_PROFILE_ARN`](#bedrock-allow-application-profile-arn)   | `false` | Allow users to pass application inference profile ARNs directly as model IDs                        |
| [`AWS_BEDROCK_ALLOW_PROMPT_ROUTER_ARN`](#bedrock-allow-prompt-router-arn)                         | `false` | Allow users to pass prompt router ARNs directly as model IDs                                        |
| [`AWS_BEDROCK_ALLOW_PROMPT_ARN`](#bedrock-allow-prompt-arn)                                       | `false` | Allow users to reference Prompt Management prompt ARNs in the Responses API `prompt` parameter      |
| [`AWS_BEDROCK_MODEL_ARN_MAPPING`](#bedrock-model-arn-mapping)                                     | `{}`    | Map model IDs to custom inference profile or prompt router ARNs (server-controlled routing)         |
| [`AWS_BEDROCK_GUARDRAIL_IDENTIFIER`](#aws-bedrock-guardrail-identifier)                           | None    | Bedrock Guardrails ID for content filtering and safety controls                                     |
| [`AWS_BEDROCK_GUARDRAIL_VERSION`](#aws-bedrock-guardrail-version)                                 | None    | Bedrock Guardrails version number (required with identifier)                                        |
| [`AWS_BEDROCK_GUARDRAIL_TRACE`](#aws-bedrock-guardrail-trace)                                     | None    | Guardrails trace level: `disabled`, `enabled`, or `enabled_full`                                    |
| [`AWS_BEDROCK_ALLOW_GUARDRAIL_OVERRIDE`](#aws-bedrock-allow-guardrail-override)                   | `false` | Allow users to override global guardrail configuration via request headers (security: default off)  |
| [`AWS_BEDROCK_SESSION_ENCRYPTION_KEY_ARN`](#aws-bedrock-session-encryption-key-arn)               | None    | KMS key ARN encrypting Amazon Bedrock session storage (Responses API `store=true`)                     |
| [`AWS_BEDROCK_BATCH_ROLE_ARN`](#aws-bedrock-batch-role-arn)                                       | None    | Service role Amazon Bedrock assumes to run batch inference jobs; unset disables the Batch APIs        |
| [`AWS_BEDROCK_USER_ROLE_ARN`](#aws-bedrock-user-role-arn)                                         | None    | Run each end user's model calls under a role session of their own, so AWS reports their spend separately |
| [`AWS_BEDROCK_USER_ROLE_SESSION_DURATION`](#aws-bedrock-user-role-session-duration)               | `3600`  | Lifetime in seconds of a per-end-user role session (900–3600)                                       |
| [`AWS_BEDROCK_USER_ROLE_TAG_KEY`](#aws-bedrock-user-role-tag-key)                                 | `user`  | Session tag key carrying the end user identity, for cost allocation and access policies             |
| [`AWS_BEDROCK_USER_ROLE_REQUIRE_IDENTITY`](#aws-bedrock-user-role-require-identity)               | `false` | Reject a model request that identifies no end user instead of billing it to the server              |

---

## :material-shield-check: Bedrock Guardrails

Amazon Bedrock Guardrails add content filtering and safety controls to model inputs and outputs. The configured guardrail also powers the [OpenAI-compatible Moderations API](api_openai_moderations.md) (`POST /v1/moderations`); without one, that API falls back to [inline guardrail checks](api_openai_moderations.md#model-support) in supported regions, then Amazon Comprehend.

!!! info "Configuration Options"
    Guardrails can be configured in three ways:

    1. :material-cog: **Global** - Via environment variables
    2. :material-web: **Per-request** - Via HTTP headers
    3. :material-code-json: **Request body** - Via `amazon-bedrock-guardrailConfig` object

### Route Coverage

The configured guardrail applies to every route that serves a request directly. The [Batch API](api_openai_batches.md) is the exception: Amazon Bedrock batch inference cannot apply a guardrail, so a batch a configured guardrail would cover is refused rather than run unchecked. Chat routes use the native Bedrock integration; routes whose AWS backend has no guardrail mechanism enforce it through the [ApplyGuardrail API](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html): client-supplied text is checked as `INPUT` before the backend call and generated text as `OUTPUT` after it. On the Realtime API, where content streams both ways for as long as the session is open, the check runs per turn and an intervention ends the session — see [Guardrail coverage](api_openai_realtime.md#guardrail-coverage) for what that does and does not catch.

| Routes                                                     | Mechanism                            | Checked content                                        |
|------------------------------------------------------------|--------------------------------------|--------------------------------------------------------|
| Chat Completions, Responses, Completions, Anthropic Messages | :material-link: Native (Converse `guardrailConfig` / InvokeModel) | Model input and output                 |
| Moderations                                                | :material-shield-search: ApplyGuardrail (classification) | Submitted text and images                 |
| Embeddings (OpenAI and Cohere v1/v2)                       | :material-shield-check: ApplyGuardrail | `INPUT` — each text input                             |
| Rerank (Cohere v1/v2)                                      | :material-shield-check: ApplyGuardrail | `INPUT` — query and each document text                |
| Images Generations / Edits                                 | :material-shield-check: ApplyGuardrail | `INPUT` — prompt (Variations has no text to check)    |
| Videos                                                     | :material-shield-check: ApplyGuardrail | `INPUT` — prompt                                      |
| Audio Speech                                               | :material-shield-check: ApplyGuardrail | `INPUT` — text to synthesize                          |
| Audio Transcriptions (including streaming)                 | :material-shield-check: ApplyGuardrail | `OUTPUT` — transcript                                 |
| Audio Translations                                         | :material-shield-check: ApplyGuardrail | `OUTPUT` — translated text                            |
| [Realtime](api_openai_realtime.md#guardrail-coverage)      | :material-shield-check: ApplyGuardrail | `INPUT` — each written item before it reaches the model, and each transcribed caller turn; `OUTPUT` — each completed answer |

!!! warning "Cost Tracking"
    AWS bills the guardrail on **every** route it applies to, but only the ApplyGuardrail-enforced ones report the units consumed. The mechanism a route uses therefore decides whether its guardrail cost is visible.

    | Mechanism | Guardrail cost in [usage logs](operations_logging_monitoring.md) |
    |-----------|-----------------------------------------------------------------|
    | :material-shield-check: ApplyGuardrail | :material-check-circle:{ .success role="img" aria-label="Tracked" } **Tracked** — the response returns the units each policy consumed |
    | :material-link: Native (Converse / InvokeModel) | :material-close-circle:{ .unsupported role="img" aria-label="Not tracked" } **Not tracked** — the response reports no guardrail units |

    On ApplyGuardrail routes, the units AWS reports appear as `text_units` and `input_images` under one `amazon.bedrock-runtime-guardrail-*` model per applied policy, each priced at that policy's own rate; see [Moderations billing](api_openai_moderations.md#billing). A route that checks both `INPUT` and `OUTPUT` calls the API twice, so it records two sets of units for one request.

    On native routes the guardrail still runs and AWS still charges for it, but the Converse and InvokeModel responses carry no unit counts for the gateway to record. **Reported costs on these routes are lower than the AWS bill by the guardrail's share.** Deriving the units from text length instead would be a guess, not a measurement, so none is made.

!!! info "Intervention Behavior"
    On ApplyGuardrail-enforced routes, a blocking intervention fails the request with HTTP 400 and error code `content_filter` (the same code chat routes report as their finish reason), carrying the guardrail's configured blocked messaging. A masking-only intervention (sensitive-information anonymization) substitutes the masked text — input masking reaches the backend model, and a masked transcript or translation is returned on the plain `json`/`text` formats. Response formats that cannot carry masked text (`srt`, `vtt`, `verbose_json`, `diarized_json`) fail with the same `content_filter` error instead of leaking the unmasked content.

### Global Configuration

#### `AWS_BEDROCK_GUARDRAIL_IDENTIFIER` { #aws-bedrock-guardrail-identifier }

:octicons-package-24: **Purpose**
:   ID of the Bedrock Guardrail to apply

:octicons-alert-24: **Required**
:   Yes (together with `AWS_BEDROCK_GUARDRAIL_VERSION`)

```bash
export AWS_BEDROCK_GUARDRAIL_IDENTIFIER=abc123def456
```

#### `AWS_BEDROCK_GUARDRAIL_VERSION` { #aws-bedrock-guardrail-version }

:octicons-package-24: **Purpose**
:   Version of the Bedrock Guardrail

:octicons-alert-24: **Required**
:   Yes (together with `AWS_BEDROCK_GUARDRAIL_IDENTIFIER`)

```bash
export AWS_BEDROCK_GUARDRAIL_VERSION=1
```

#### `AWS_BEDROCK_GUARDRAIL_TRACE` { #aws-bedrock-guardrail-trace }

:octicons-package-24: **Purpose**
:   Trace level for guardrail evaluation

:octicons-gear-24: **Options**
:   `disabled`, `enabled`, `enabled_full`

:octicons-gear-24: **Default**
:   None (optional)

```bash
export AWS_BEDROCK_GUARDRAIL_TRACE=enabled
```

#### `AWS_BEDROCK_ALLOW_GUARDRAIL_OVERRIDE` { #aws-bedrock-allow-guardrail-override }

:octicons-package-24: **Purpose**
:   Control whether users can override the global guardrail configuration at request level via HTTP headers

:octicons-gear-24: **Default**
:   `false` (disabled for security)

:octicons-shield-24: **Security Consideration**
:   When set to `false` (default) and a global guardrail is configured, only the global configuration is enforced, preventing users from bypassing or modifying safety controls. Set to `true` if you need to allow per-request guardrail customization to override the global configuration.

:octicons-info-24: **Auto-Enable Behavior**
:   If no guardrail is configured at all — both `AWS_BEDROCK_GUARDRAIL_IDENTIFIER` and `AWS_BEDROCK_GUARDRAIL_VERSION` unset, and no [model alias](operations_configuration_models.md#model-aliases-configuration) carrying one — this setting is automatically set to `true` at startup, allowing per-request guardrails when no policy is enforced.

```bash
export AWS_BEDROCK_ALLOW_GUARDRAIL_OVERRIDE=true
```

!!! tip "Per-Alias Guardrails"
    A [model alias](operations_configuration_models.md#model-aliases-configuration) can carry its own guardrail, applied to the requests naming it and overriding the global one. That is how a single deployment publishes the same model under a strictly guarded name and an unguarded one.

!!! example "Complete Guardrail Configuration"
    ```bash
    export AWS_BEDROCK_GUARDRAIL_IDENTIFIER=abc123def456
    export AWS_BEDROCK_GUARDRAIL_VERSION=1
    export AWS_BEDROCK_GUARDRAIL_TRACE=enabled
    export AWS_BEDROCK_ALLOW_GUARDRAIL_OVERRIDE=false  # Default: prevent overrides
    ```

### Per-Request Guardrail Configuration

!!! info "Header Usage Behavior"
    Request headers can be used when [`AWS_BEDROCK_ALLOW_GUARDRAIL_OVERRIDE`](#aws-bedrock-allow-guardrail-override) is `true`:

    - **No global guardrail configured**: Setting is automatically `true` at startup, enabling per-request guardrails
    - **Global guardrail configured**: Setting defaults to `false` for security; set to `true` to allow overrides

    This prevents users from bypassing configured safety controls while still allowing flexibility when no global policy exists.

!!! warning "Refused on a Mantle-served model"
    Amazon Bedrock Mantle has no guardrail parameter, so a guardrail cannot be applied to a model served through it. Whether a request carries these headers is not known until it arrives, so this cannot be caught at startup the way a *global* or *alias* guardrail combined with [`AWS_BEDROCK_MANTLE_PREFERRED_MODELS`](operations_configuration_models.md#bedrock-mantle-preferred-models) is. Such a request is **refused with `400`** rather than served unguarded: a caller who asked to be guarded is never answered as though the guardrail had run. The same refusal covers a deployment-wide guardrail meeting a Mantle-only model. Clear `AWS_BEDROCK_MANTLE_PREFERRED_MODELS` for a deployment that needs per-request guardrails to apply to these models.

Use HTTP headers to specify guardrail settings per request:

| Header                                         | Purpose                                                              | Valid Values                          |
|-------------------------------------------------|-----------------------------------------------------------------------|----------------------------------------|
| `X-Amzn-Bedrock-GuardrailIdentifier`           | Guardrail ID                                                          | Your guardrail identifier             |
| `X-Amzn-Bedrock-GuardrailVersion`              | Guardrail version                                                     | Version number (e.g., `1`)            |
| `X-Amzn-Bedrock-Trace`                         | Trace level                                                           | `disabled`, `enabled`, `enabled_full` |
| `X-Amzn-Bedrock-GuardrailStreamProcessingMode` | Guardrail assessment timing for streaming requests (stripped from non-streaming requests) | `sync`, `async`                       |

```bash title="Example cURL Request"
curl -X POST https://api.example.com/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -H "X-Amzn-Bedrock-GuardrailIdentifier: abc123def456" \
  -H "X-Amzn-Bedrock-GuardrailVersion: 1" \
  -H "X-Amzn-Bedrock-Trace: enabled" \
  -d '{"model": "anthropic.claude-sonnet-5", "messages": [...]}'
```

### Request Body Configuration

The `amazon-bedrock-guardrailConfig` object in the request body is supported for OpenAI Chat Completions compatibility.

!!! warning "Compatibility Note"
    Only fields compatible with Bedrock Converse API are honored. The `tagSuffix` field is documented in AWS but **not supported** in this implementation.

---

## :material-database-lock: Bedrock Session Storage { #bedrock-session-storage-optional }

Requests with `store=true` on the [Responses](api_openai_responses.md#stored-responses) and [Chat Completions](api_openai_chat_completions.md#stored-chat-completions) APIs persist generations in Amazon Bedrock sessions. No environment variable is needed to enable this — it requires the [Bedrock Session Storage IAM permissions](operations_iam_permissions.md#bedrock-session-storage-optional).

!!! warning "Not available in every region"
    Amazon Bedrock session storage covers fewer regions than model inference. When the primary Bedrock region — the first entry of [`AWS_BEDROCK_REGIONS`](operations_configuration_aws.md#aws-bedrock-regions), which is where all sessions are created — does not provide it, `store=true` is **ignored**: the generation is still returned, and a warning is recorded in the request log stating that the session storage endpoint was unreachable or timed out and that session storage is offered in fewer regions than model inference. Retrieving a stored object then returns `404`. A missing `bedrock:CreateSession` permission produces a distinct `AccessDenied` warning pointing at the IAM permissions instead.

    Nothing fails and no request is lost, but stored responses and stored chat completions are simply unavailable. To rely on them, make the primary Bedrock region one that offers session storage — check the [Amazon Bedrock session management endpoints](https://docs.aws.amazon.com/general/latest/gr/bedrock.html) for current coverage.

#### `AWS_BEDROCK_SESSION_ENCRYPTION_KEY_ARN` { #aws-bedrock-session-encryption-key-arn }

:octicons-package-24: **Purpose**
:   KMS key ARN encrypting the Amazon Bedrock sessions that back [stored responses](api_openai_responses.md#stored-responses) and [stored chat completions](api_openai_chat_completions.md#stored-chat-completions) (`store=true`)

:octicons-gear-24: **Default**
:   None — sessions are encrypted with the AWS-managed key

:octicons-check-circle-24: **Validation**
:   Checked at startup: must be a KMS key ARN (`arn:<partition>:kms:<region>:<account-id>:key/<key-id>`).

```bash
export AWS_BEDROCK_SESSION_ENCRYPTION_KEY_ARN=arn:aws:kms:us-east-1:123456789012:key/abcd-...
```

!!! warning "Shared Visibility Across Deployments"
    Stored responses and chat completions are namespaced by AWS account and region, not by stdapi.ai deployment. Multiple deployments sharing the same account and region can list, retrieve, and delete each other's stored objects. Use a dedicated AWS account per deployment when isolation matters, or accept this shared visibility as a deliberate trade-off.

!!! info "Orphaned Session Cleanup"
    A session is created independently of the generation it will hold — before it for the Responses API, concurrently with it for Chat Completions — so a crash before the generation is written leaves an empty, orphaned session. Bedrock sessions have no TTL and persist until deleted, so periodically clean up stale sessions (`aws bedrock-agent-runtime list-sessions` plus `delete-session`, or an operator-managed lifecycle policy).

#### `AWS_BEDROCK_BATCH_ROLE_ARN` { #aws-bedrock-batch-role-arn }

:octicons-package-24: **Purpose**
:   AWS IAM service role that [Amazon Bedrock batch inference](https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html) assumes to read a batch's requests and write its results — required to enable the [Batch API](api_openai_batches.md) and the [Message Batches API](api_anthropic_batches.md)

:octicons-database-24: **Type**
:   String — an IAM role ARN

:octicons-gear-24: **Default**
:   None — the batch endpoints answer `503` (`529` on the Anthropic-compatible routes) and the server reports the disabled feature in its startup log

:octicons-workflow-24: **Behavior**
:   The server passes this role when it submits a batch; Amazon Bedrock then reads the requests and writes the results with it. The role must be able to read and write every bucket configured with [`AWS_S3_BUCKET`](operations_configuration_storage.md#aws-s3-bucket) and [`AWS_S3_REGIONAL_BUCKETS`](operations_configuration_storage.md#aws-s3-regional-buckets), under [`AWS_S3_BATCHES_PREFIX`](operations_configuration_storage.md#aws-s3-batches-prefix).

:octicons-check-circle-24: **Validation**
:   Checked at startup: must be an IAM role ARN (`arn:<partition>:iam::<account-id>:role/<name>`). Leaving it unset is reported as a startup warning, never a failure.

```bash
export AWS_BEDROCK_BATCH_ROLE_ARN=arn:aws:iam::123456789012:role/stdapi-ai-batch
```

!!! warning "The role and two IAM policies come first"
    The role's trust policy must allow `bedrock.amazonaws.com` to assume it, and the server's own role needs `iam:PassRole` on this ARN. See [IAM Permissions](operations_iam_permissions.md#batch-inference) for copyable policies.

#### `AWS_BEDROCK_USER_ROLE_ARN` { #aws-bedrock-user-role-arn }

:octicons-package-24: **Purpose**
:   Run each end user's model calls under an AWS IAM role session of their own, so AWS reports Amazon Bedrock model usage [per end user](operations_cost_management.md#per-user-attribution) in Cost Explorer and in the Cost and Usage Report

:octicons-database-24: **Type**
:   String — an IAM role ARN

:octicons-gear-24: **Default**
:   None — every request runs under the server's own identity, and AWS reports all model usage under it

:octicons-workflow-24: **Behavior**
:   The server opens one short-lived session of this role per end user, caches it, and signs that user's model invocations with it. The identity is the authenticated caller when [authentication](operations_authentication_security.md) is enabled, otherwise the identifier the request declares (`safety_identifier` or `user` on the OpenAI-compatible APIs, `metadata.user_id` on the Anthropic Messages API). Only model invocations are covered — guardrail evaluations, video generation, speech, transcription and translation keep the server's identity. A session that cannot be opened fails the request rather than falling back to the server's identity.

:octicons-check-circle-24: **Validation**
:   Checked at startup: must be an IAM role ARN (`arn:<partition>:iam::<account-id>:role/<name>`). The server also tries to assume it at startup and reports a warning — not a failure — when it cannot.

```bash
export AWS_BEDROCK_USER_ROLE_ARN=arn:aws:iam::123456789012:role/stdapi-ai-end-user
```

!!! warning "The role and two IAM policies come first"
    The role's trust policy must allow this server's own role to call both `sts:AssumeRole` **and** `sts:TagSession` on it, and the server's role needs the same two actions on this role ARN. See [IAM Permissions](operations_iam_permissions.md#per-user-cost-attribution) for copyable policies, including the model ARNs a cross-region inference profile requires.

#### `AWS_BEDROCK_USER_ROLE_SESSION_DURATION` { #aws-bedrock-user-role-session-duration }

:octicons-package-24: **Purpose**
:   Lifetime of a per-end-user role session, in seconds

:octicons-database-24: **Type**
:   Integer — 900 to 3600

:octicons-gear-24: **Default**
:   `3600`

:octicons-workflow-24: **Behavior**
:   Sessions are cached per end user and reopened shortly before they expire, so a longer lifetime means fewer AWS STS calls. The upper bound is imposed by AWS: the server itself runs under an assumed role, and a role session obtained from another role session cannot last longer than one hour, whatever the role's maximum session duration.

```bash
export AWS_BEDROCK_USER_ROLE_SESSION_DURATION=1800
```

#### `AWS_BEDROCK_USER_ROLE_TAG_KEY` { #aws-bedrock-user-role-tag-key }

:octicons-package-24: **Purpose**
:   Session tag key carrying the end user identity on each per-end-user role session

:octicons-database-24: **Type**
:   String, or null to send no session tag

:octicons-gear-24: **Default**
:   `user`

:octicons-workflow-24: **Behavior**
:   Activate this key as a cost allocation tag — in the AWS Billing console, under **Cost allocation tags** filtered by type **IAM principal** — to group Bedrock costs by end user in Cost Explorer. The same tag is testable in IAM policies as `aws:PrincipalTag/<key>`, so the role can be restricted per user. With no tag, end users are still distinguished by their role session name in the Cost and Usage Report.

:octicons-check-circle-24: **Validation**
:   Checked at startup: 1 to 128 characters over letters, digits, spaces and `_ . : / = + - @`; keys beginning with `aws:` are reserved by AWS and rejected.

```bash
export AWS_BEDROCK_USER_ROLE_TAG_KEY=end-user
```

#### `AWS_BEDROCK_USER_ROLE_REQUIRE_IDENTITY` { #aws-bedrock-user-role-require-identity }

:octicons-package-24: **Purpose**
:   Reject a model request that identifies no end user, instead of running it under the server's own identity

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false` — such requests run under the server's identity, and their usage is reported under it

:octicons-workflow-24: **Behavior**
:   Enable it so that no model usage escapes per-user attribution: a request carrying neither an authenticated caller nor an end user identifier is answered `400`. Clients that never send one stop working, so enable it only once every client identifies its user — and note that some APIs, audio transcription among them, have no end user field at all, so on those it takes an authenticated caller. A real-time speech-to-speech session keeps for its whole life the identity it opened with, so while this is enabled it is refused rather than attributed to the server. Requires [`AWS_BEDROCK_USER_ROLE_ARN`](#aws-bedrock-user-role-arn).

```bash
export AWS_BEDROCK_USER_ROLE_REQUIRE_IDENTITY=true
```

!!! warning "Two Backends Cannot Assume the Per-User Role"
    A [Bedrock Mantle](operations_configuration_aws.md#bedrock-mantle-enabled) request and an [Amazon SageMaker AI endpoint](operations_configuration_models.md#aws-sagemaker-endpoints) invocation are signed with the server's own credentials, so a model served there never runs under the per-end-user role and the `aws:PrincipalTag` conditions written on that role are never evaluated. The `400` still holds on both — a request identifying no end user is refused the same way, whichever backend serves the model — but the access policy does not.

    For Mantle that is decidable ahead of time, so routing a dual-homed model there with a non-empty [`AWS_BEDROCK_MANTLE_PREFERRED_MODELS`](operations_configuration_models.md#bedrock-mantle-preferred-models), which is the default, **stops the server at startup**; clear that setting to keep both. Mantle-only models and SageMaker AI endpoints have no classic endpoint to fall back to — nothing is refused at startup for them, and an identified request is served under the server's own role.

---

## :material-speedometer: Bedrock Service Tier and Performance Configuration

Amazon Bedrock service tiers and performance configurations allow you to optimize AI workload performance and cost trade-offs. Configure latency optimization and throughput priority for your inference requests.

!!! info "AWS Documentation"
    For detailed information about service tiers, see:

    - [Amazon Bedrock Service Tiers](https://aws.amazon.com/blogs/aws/new-amazon-bedrock-service-tiers-help-you-match-ai-workload-performance-with-cost/)

### Service Tiers

Service tiers help you match AI workload performance with cost by selecting the appropriate throughput and latency characteristics:

- **`priority`** - Highest priority processing with guaranteed capacity and fastest response times. Best for latency-sensitive applications.
- **`default`** - Standard processing with balanced performance and cost. Suitable for most production workloads.
- **`flex`** - Cost-optimized processing with flexible scheduling. Best for batch jobs and non-time-sensitive workloads.
- **`reserved`** - Dedicated reserved capacity with a throughput commitment (requires an AWS contract). Best for predictable workloads needing an uptime guarantee.

### Performance Configuration

Performance configuration allows you to optimize for latency:

- **`standard`** - Standard latency profile with balanced performance
- **`optimized`** - Optimized for lowest possible latency

### Per-Request Service Tier Configuration { #service-tier-per-request }

Configure service tier and performance settings per request using HTTP headers. These headers are available on all Bedrock-based routes (Chat Completions, Embeddings, Images). Server-side per-model defaults can be set with [`DEFAULT_MODEL_SERVICE_TIERS`](operations_configuration_models.md#default-model-service-tiers).

| Header                                     | Purpose                | Valid Values                  |
|--------------------------------------------|------------------------|-------------------------------|
| `X-Amzn-Bedrock-Service-Tier`              | Service tier selection | `priority`, `default`, `flex`, `reserved` |
| `X-Amzn-Bedrock-PerformanceConfig-Latency` | Latency optimization   | `standard`, `optimized`       |

!!! warning "The tier header is subject to the override gate"
    When [`AWS_BEDROCK_ALLOW_SERVICE_TIER_OVERRIDE`](operations_configuration_models.md#aws-bedrock-allow-service-tier-override) is `false`, `X-Amzn-Bedrock-Service-Tier` — like the `service_tier` request parameter — is ignored for any model that has a tier configured, by `DEFAULT_MODEL_SERVICE_TIERS` or by the [alias](operations_configuration_models.md#model-aliases-configuration) the request names. A model with no configured tier honors the header in either case. The response's `service_tier` field keeps echoing the request's own value; [usage and cost reporting](operations_cost_management.md) record the tier that actually served the call.

    Configured tiers, the header and this gate all apply to models served through the Bedrock Converse and InvokeModel APIs. On a [Bedrock Mantle](operations_configuration_aws.md#summary-bedrock-mantle)-served model, the request's own `service_tier` parameter is what applies — the header is not read, no configured tier is added, and the response reports the tier that model returns.

```bash title="Example: Chat Completions with Priority Tier and Optimized Latency"
curl -X POST https://api.example.com/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -H "X-Amzn-Bedrock-Service-Tier: priority" \
  -H "X-Amzn-Bedrock-PerformanceConfig-Latency: optimized" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "anthropic.claude-sonnet-5",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

```bash title="Example: Embeddings with Flex Tier for Batch Processing"
curl -X POST https://api.example.com/v1/embeddings \
  -H "Authorization: Bearer sk-..." \
  -H "X-Amzn-Bedrock-Service-Tier: flex" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-2-multimodal-embeddings-v1:0",
    "input": ["text 1", "text 2", "text 3"]
  }'
```

```bash title="Example: Image Generation with Default Tier"
curl -X POST https://api.example.com/v1/images/generations \
  -H "Authorization: Bearer sk-..." \
  -H "X-Amzn-Bedrock-Service-Tier: default" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-canvas-v1:0",
    "prompt": "A serene mountain landscape"
  }'
```

!!! tip "When to Use Each Tier"
    **Priority Tier:**

    - Real-time customer-facing applications
    - Interactive chatbots and assistants
    - Applications requiring guaranteed low latency
    - Production workloads with strict SLAs

    **Default Tier:**

    - Standard production workloads
    - General-purpose API usage
    - Applications with moderate latency requirements

    **Flex Tier:**

    - Batch processing and bulk operations
    - Offline content generation
    - Data processing pipelines
    - Non-time-sensitive workloads
    - Cost-optimized inference at scale

---

## :material-account-voice: Audio and Text-to-Speech

#### `DEFAULT_TTS_MODEL` { #default-tts-model }

:octicons-package-24: **Purpose**
:   Default text-to-speech model when not specified in requests

:octicons-gear-24: **Default**
:   `amazon.polly-standard`

| Model | Description | Quality |
|-------|-------------|---------|
| `amazon.polly-standard` | Standard Polly voices | :material-star: Classic quality |
| `amazon.polly-neural` | Neural Polly voices | :material-star-circle: Higher quality, more natural |
| `amazon.polly-long-form` | Long-form content | :material-text-long: Optimized for long content |
| `amazon.polly-generative` | Generative AI voices | :material-sparkles: Latest technology |

```bash
export DEFAULT_TTS_MODEL=amazon.polly-neural
```

#### `DEFAULT_TTS_LANGUAGE` { #default-tts-language }

:octicons-package-24: **Purpose**
:   Default language code for text-to-speech synthesis when using OpenAI voice names

:octicons-gear-24: **Default**
:   None (automatic language detection via Amazon Comprehend)

:octicons-check-circle-24: **Behavior**
:   When specified, this language is used instead of automatic detection. When not set, Amazon Comprehend detects the language automatically from the input text.

**Valid Language Codes**: Any Amazon Polly language code (e.g., `en-US`, `fr-FR`, `es-ES`, `de-DE`, `ja-JP`)

```bash
# Use English (US) for all TTS requests
export DEFAULT_TTS_LANGUAGE=en-US

# Use French for all TTS requests
export DEFAULT_TTS_LANGUAGE=fr-FR
```

!!! tip "Performance Benefits"
    Setting a default language improves performance by:

    - **Faster responses**: Skips language detection API call to Amazon Comprehend
    - **Reduced costs**: No Amazon Comprehend charges for language detection
    - **Predictable voice selection**: Always uses voices from the specified language

!!! info "When to Use"
    Consider setting a default language when:

    - Your application primarily serves content in a single language
    - You want to optimize response times and reduce AWS service calls
    - You prefer predictable voice selection over automatic language matching

!!! note "Interaction with Voice Selection"
    This setting only affects automatic language detection when using OpenAI voice names (like `alloy`, `echo`, `nova`). If you specify a Polly voice ID directly (like `Joanna`, `Matthew`), language detection is already skipped.

---

## :material-headset: Realtime API

#### `REALTIME_CLIENT_SECRET_KEY` { #realtime-client-secret-key }

:octicons-package-24: **Purpose**
:   Secret the [Realtime API](api_openai_realtime.md#ephemeral-client-secrets)'s ephemeral client secrets (`POST /v1/realtime/client_secrets`) are signed with

:octicons-database-24: **Type**
:   String (any value)

:octicons-gear-24: **Default**
:   None — a signing key is derived from the configured API key instead

:octicons-workflow-24: **Behavior**
:   Ephemeral client secrets are stateless: nothing is stored server-side, so any instance behind a load balancer verifies a secret minted by any other, as long as they all sign with the same key. By default that key is derived from the deployment's own API key, which is already shared across every instance — so this setting has nothing to add on a deployment that already configures one.

    Set it explicitly on a deployment that runs with **no API key configured at all**: without either one, each instance falls back to a random key generated **per process**, and a client secret minted by one instance then fails to verify on any other — the symptom is an ephemeral secret rejected intermittently on a multi-instance deployment. Any value works, as long as every instance shares it.

```bash
export REALTIME_CLIENT_SECRET_KEY=a-value-shared-by-every-instance
```

!!! warning "Changing it invalidates outstanding secrets"
    A secret minted with one key does not verify against another. Rotating this setting (or the API key it would otherwise derive from) invalidates every client secret minted before the change — they simply stop working once their bearer tries to open a session, the same as if they had expired.

#### `REALTIME_ALLOW_SESSION_OVERRIDE` { #realtime-allow-session-override }

:octicons-package-24: **Purpose**
:   Whether a client connecting with an [ephemeral client secret](api_openai_realtime.md#ephemeral-client-secrets) may override the session configuration that secret carries

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `true` — the upstream behavior: the carried configuration is a default the client may change

:octicons-workflow-24: **Behavior**
:   A minted secret carries a session configuration, and by default a client opening a session with it may name another model on the `?model=` query string and replace any of that configuration with its own `session.update` — exactly as it can against the upstream API.

    Set to `false` on a multi-tenant deployment, where the secret is the only thing constraining an untrusted browser or mobile client. The `model`, the `instructions` and `max_output_tokens` the secret was minted with are then final: connecting with a `?model=` naming a different model is refused before the session opens, and a `session.update` changing any of the three answers an `error` event. Everything else — voice, audio formats, turn detection, transcription — stays under the client's control.

```bash
export REALTIME_ALLOW_SESSION_OVERRIDE=false
```

---

#### `REALTIME_WEBRTC_ENABLED` { #realtime-webrtc-enabled }

:octicons-package-24: **Purpose**
:   Serve WebRTC calls on [`POST /v1/realtime/calls`](api_openai_realtime.md#webrtc-calls), terminating the media path — ICE, DTLS-SRTP, Opus — in the server process

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false` — the endpoint answers `404` and only the WebSocket transport is served

:octicons-workflow-24: **Behavior**
:   Enabling it requires the `webrtc` optional dependencies (`stdapi[webrtc]`, shipped in the container images; glibc only — aiortc publishes no musl wheels) and the server refuses to start without them. It also requires a network path the default deployment does not have: WebRTC media is UDP on ephemeral ports, negotiated directly to the instance that answered the SDP offer, which no HTTP(S) load balancer can carry. See [the deployment section](operations_deploy_advanced.md#webrtc-and-sip-need-their-own-ingress) for what the Terraform module's media mode provisions, and [the transport documentation](api_openai_realtime.md#webrtc-calls) for the call-control and duration limits that come with it.

```bash
export REALTIME_WEBRTC_ENABLED=true
```

---

#### `REALTIME_WEBRTC_STUN_SERVER` { #realtime-webrtc-stun-server }

:octicons-package-24: **Purpose**
:   STUN server the gateway queries to discover its public address, which is then advertised to WebRTC callers as an ICE candidate

:octicons-database-24: **Type**
:   String — a STUN URI, e.g. `stun:stun.l.google.com:19302`

:octicons-gear-24: **Default**
:   None — only the host's own addresses are advertised

:octicons-workflow-24: **Behavior**
:   Required whenever the server sits behind 1:1 NAT, which is exactly the shape of an ECS task with a public IP: the task sees only its private address, so without STUN the SDP answer advertises addresses no caller can reach — the exchange succeeds and the call carries no audio. Any public STUN server works; the queried server learns nothing but the deployment's public address.

```bash
export REALTIME_WEBRTC_STUN_SERVER="stun:stun.l.google.com:19302"
```

---

#### `REALTIME_WEBRTC_TURN_SERVER` { #realtime-webrtc-turn-server }

:octicons-package-24: **Purpose**
:   TURN relay advertised to WebRTC callers, carrying the media of callers whose networks block UDP to arbitrary ports

:octicons-database-24: **Type**
:   String — a TURN URI, e.g. `turn:turn.example.com:3478?transport=udp`, with `REALTIME_WEBRTC_TURN_USERNAME` and `REALTIME_WEBRTC_TURN_PASSWORD` carrying its long-term credentials (all three together, or none)

:octicons-gear-24: **Default**
:   None — callers on UDP-blocking networks cannot establish media

:octicons-workflow-24: **Behavior**
:   The relay is yours to run — [coturn](https://github.com/coturn/coturn) is the usual choice — because AWS offers no managed TURN service. It needs its own public address and its own always-on capacity, which is part of why a [media framework in front](api_openai_realtime.md#transports) remains the recommended shape for demanding audiences.

```bash
export REALTIME_WEBRTC_TURN_SERVER="turn:turn.example.com:3478?transport=udp"
export REALTIME_WEBRTC_TURN_USERNAME="stdapi"
export REALTIME_WEBRTC_TURN_PASSWORD="..."
```

---

#### `REALTIME_WEBRTC_ALLOW_PRIVATE_CANDIDATES` { #realtime-webrtc-allow-private-candidates }

:octicons-package-24: **Purpose**
:   Accept the ICE candidates a WebRTC caller offers on addresses that are not globally routable — private (RFC 1918), shared (RFC 6598), loopback and link-local ones

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false` — those candidates are dropped from the offer, and an offer left with no candidate at all is refused with `invalid_offer`

:octicons-workflow-24: **Behavior**
:   An SDP offer names the addresses the server sends its ICE connectivity checks to, and the caller posting it holds nothing more than an [ephemeral client secret](api_openai_realtime.md#ephemeral-client-secrets). Screening the offer is what keeps an untrusted caller from aiming those UDP probes at addresses inside the deployment's own VPC. Enable this only where callers legitimately share that network — a same-VPC service, an on-premises deployment, a LAN — and the exchange otherwise succeeds with no media path. Hostname and mDNS (`.local`) candidates are dropped either way, whatever this is set to: resolving one is itself a lookup on the deployment's network, and a browser that offers only mDNS candidates cannot establish media with this gateway regardless.

```bash
export REALTIME_WEBRTC_ALLOW_PRIVATE_CANDIDATES=true
```

---

## :material-identifier: ARN Access Controls

Which Amazon Bedrock ARNs a request may name as its model, and how an ARN is mapped onto a model identifier.

#### `AWS_BEDROCK_ALLOW_CROSS_REGION_INFERENCE_PROFILE_ARN` { #bedrock-allow-cross-region-profile-arn }

:octicons-package-24: **Purpose**
:   Allow users to pass cross-region inference profile ARNs directly as model IDs in API requests

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false`

:octicons-workflow-24: **Behavior**
:   When enabled, users can use cross-region inference profile ARNs instead of model IDs in the `model` parameter. Cross-region inference profiles enable routing to multiple regions for better availability

:octicons-lock-24: **IAM Permissions Required**
:   `bedrock:GetInferenceProfile` (see [IAM Permissions](operations_iam_permissions.md#bedrock-inference-profiles-and-prompt-routers-optional))

```bash
# Disabled (default) - users can only use standard model IDs
# No environment variable needed

# Enable cross-region inference profile ARN support
export AWS_BEDROCK_ALLOW_CROSS_REGION_INFERENCE_PROFILE_ARN=true
```

!!! warning "Additional IAM Permissions Required"
    Enabling this setting requires adding the `bedrock:GetInferenceProfile` IAM permission to your role/user. Without this permission, API requests using inference profile ARNs will fail with authorization errors.

    See the [Bedrock Inference Profiles and Prompt Routers IAM section](operations_iam_permissions.md#bedrock-inference-profiles-and-prompt-routers-optional) for the complete policy configuration.

!!! example "Example ARN"
    ```text
    arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-sonnet-5
    ```

!!! success "Automatic Cross-Region Routing (Default Behavior)"
    **By default, stdapi.ai automatically determines and uses the best cross-region inference profile for each model**, based on [`AWS_BEDROCK_REGIONS`](operations_configuration_aws.md#aws-bedrock-regions), [`AWS_BEDROCK_CROSS_REGION_INFERENCE`](operations_configuration_aws.md#cross-region-inference), and [`AWS_BEDROCK_CROSS_REGION_INFERENCE_GLOBAL`](operations_configuration_aws.md#cross-region-global). Manually passing cross-region inference profile ARNs is only needed in rare cases to override that selection — for most deployments, leave this disabled. See [Using Inference Profile and Prompt Router ARNs](#using-inference-profile-and-prompt-router-arns) for details.

#### `AWS_BEDROCK_ALLOW_APPLICATION_INFERENCE_PROFILE_ARN` { #bedrock-allow-application-profile-arn }

:octicons-package-24: **Purpose**
:   Allow users to pass application inference profile ARNs directly as model IDs in API requests

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false`

:octicons-workflow-24: **Behavior**
:   When enabled, users can use application inference profile ARNs instead of model IDs in the `model` parameter. Application inference profiles are custom routing configurations for specific use cases

:octicons-lock-24: **IAM Permissions Required**
:   `bedrock:GetInferenceProfile` (see [IAM Permissions](operations_iam_permissions.md#bedrock-inference-profiles-and-prompt-routers-optional))

```bash
# Disabled (default) - users can only use standard model IDs
# No environment variable needed

# Enable application inference profile ARN support
export AWS_BEDROCK_ALLOW_APPLICATION_INFERENCE_PROFILE_ARN=true
```

!!! warning "Additional IAM Permissions Required"
    Enabling this setting requires adding the `bedrock:GetInferenceProfile` IAM permission to your role/user. Without this permission, API requests using application inference profile ARNs will fail with authorization errors.

    See the [Bedrock Inference Profiles and Prompt Routers IAM section](operations_iam_permissions.md#bedrock-inference-profiles-and-prompt-routers-optional) for the complete policy configuration.

!!! example "Example ARN"
    ```text
    arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123xyz
    ```

!!! info "What are Application Inference Profiles?"
    Application inference profiles are custom routing configurations that you create in your AWS account. They allow you to define specific routing behavior, region preferences, and failover strategies tailored to your application's needs.

!!! tip "When to Enable"
    Enable this setting when:

    - :material-application: You have custom application inference profiles configured in your AWS account
    - :material-cog: You need application-specific routing configurations
    - :material-account-multiple: You want to give users access to custom profiles you've created

#### `AWS_BEDROCK_ALLOW_PROMPT_ROUTER_ARN` { #bedrock-allow-prompt-router-arn }

:octicons-package-24: **Purpose**
:   Allow users to pass prompt router ARNs directly as model IDs in API requests

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false`

:octicons-workflow-24: **Behavior**
:   When enabled, users can use prompt router ARNs instead of model IDs in the `model` parameter. Prompt routers enable dynamic model selection based on prompt characteristics

:octicons-lock-24: **IAM Permissions Required**
:   `bedrock:GetPromptRouter` (see [IAM Permissions](operations_iam_permissions.md#bedrock-inference-profiles-and-prompt-routers-optional))

```bash
# Disabled (default) - users can only use standard model IDs
# No environment variable needed

# Enable prompt router ARN support
export AWS_BEDROCK_ALLOW_PROMPT_ROUTER_ARN=true
```

!!! warning "Additional IAM Permissions Required"
    Enabling this setting requires adding the `bedrock:GetPromptRouter` IAM permission to your role/user. Without this permission, API requests using prompt router ARNs will fail with authorization errors.

    See the [Bedrock Inference Profiles and Prompt Routers IAM section](operations_iam_permissions.md#bedrock-inference-profiles-and-prompt-routers-optional) for the complete policy configuration.

!!! example "Example ARN"
    ```text
    arn:aws:bedrock:us-east-1:123456789012:default-prompt-router/my-router
    ```

!!! info "What are Prompt Routers?"
    Prompt routers are intelligent routing systems that analyze prompt characteristics (length, complexity, language) and dynamically select the most appropriate model. This enables cost optimization and performance tuning based on request patterns.

!!! tip "When to Enable"
    Enable this setting when:

    - :material-robot: You have prompt routers configured in your AWS account
    - :material-cash: You want intelligent cost optimization through dynamic model selection
    - :material-speedometer: You need automatic model selection based on prompt complexity

#### `AWS_BEDROCK_ALLOW_PROMPT_ARN` { #bedrock-allow-prompt-arn }

:octicons-package-24: **Purpose**
:   Allow users to reference an Amazon Bedrock Prompt Management prompt ARN in the OpenAI Responses API `prompt` parameter

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false`

:octicons-workflow-24: **Behavior**
:   When enabled, `prompt.id` accepts a prompt ARN (with an optional `prompt.version`) and `prompt.variables` fill in the template. Amazon Bedrock renders the stored prompt, and the model it is bound to serves the request. When disabled, any `prompt` parameter is rejected with a `400` error

:octicons-lock-24: **IAM Permissions Required**
:   `bedrock:GetPrompt` (resolve the prompt's model) and `bedrock:RenderPrompt` (invoke it)

```bash
# Disabled (default) - the Responses API `prompt` parameter returns 400
# No environment variable needed

# Enable Prompt Management prompt ARN support
export AWS_BEDROCK_ALLOW_PROMPT_ARN=true
```

!!! warning "Additional IAM Permissions Required"
    Enabling this setting requires adding the `bedrock:GetPrompt` and `bedrock:RenderPrompt` IAM permissions, scoped to the prompt resources you want to expose. Without them, requests using a prompt ARN fail with authorization errors.

!!! example "Example ARN"
    ```text
    arn:aws:bedrock:us-east-1:123456789012:prompt/ABCDE12345:1
    ```

!!! info "Scope and Limitations"
    - Only **TEXT** prompts are supported, and the request's `model` must be the model the prompt is bound to.
    - Prompt variable values must be plain strings.
    - `input`, `instructions`, `tools`, `text`, `previous_response_id` and inference parameters cannot be combined with `prompt`.

    See [Managed Prompt Templates](api_openai_responses.md#managed-prompt-templates) for the full request contract.

#### `AWS_BEDROCK_MODEL_ARN_MAPPING` { #bedrock-model-arn-mapping }

:octicons-package-24: **Purpose**
:   Map standard model IDs to custom inference profile or prompt router ARNs for server-controlled routing

:octicons-code-24: **Format**
:   JSON object with model IDs as keys and ARNs as values

:octicons-gear-24: **Default**
:   `{}` (empty, no mappings)

:octicons-workflow-24: **Behavior**
:   When configured, the mapped ARN is used instead of the default cross-region inference profile when clients request the model by its standard ID. This provides centralized control over model routing without requiring client changes

```bash
export AWS_BEDROCK_MODEL_ARN_MAPPING='{
  "anthropic.claude-sonnet-5": "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-custom-profile",
  "anthropic.claude-haiku-4-5-20251001-v1:0": "arn:aws:bedrock:us-east-1:123456789012:default-prompt-router/my-router"
}'
```

!!! info "What is Model ARN Mapping?"
    Model ARN mapping allows server administrators to override the default routing behavior for specific models. When a client requests a model using its standard ID (e.g., `anthropic.claude-sonnet-5`), the server automatically uses the mapped ARN for routing instead.

    **Supported ARN Types:**

    - :material-earth: **Cross-region inference profiles** - AWS-managed multi-region routing
    - :material-application: **Application inference profiles** - Custom routing configurations
    - :material-robot: **Prompt routers** - Intelligent dynamic model selection

!!! success "Key Benefits"
    - :material-server: **Centralized Control** - Change routing behavior without modifying client code
    - :material-account-group: **Transparent to Clients** - Clients use standard model IDs, server handles routing
    - :material-swap-horizontal: **Easy Migration** - Switch between routing strategies by updating server config
    - :material-cog: **Environment-Specific** - Different mappings for dev/staging/production environments

!!! example "Use Cases"

    **Cost Optimization with Prompt Router:**
    ```bash
    export AWS_BEDROCK_MODEL_ARN_MAPPING='{
      "anthropic.claude-sonnet-5": "arn:aws:bedrock:us-east-1:123456789012:default-prompt-router/cost-optimizer"
    }'
    ```
    Automatically route simple prompts to cheaper models, complex prompts to premium models.

    **Custom Application Profile:**
    ```bash
    export AWS_BEDROCK_MODEL_ARN_MAPPING='{
      "anthropic.claude-sonnet-5": "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/production-profile"
    }'
    ```
    Use your custom inference profile with specific region preferences and failover behavior.

    **Environment-Specific Routing:**
    ```bash
    # Production: Use cost-optimized prompt router
    export AWS_BEDROCK_MODEL_ARN_MAPPING='{"anthropic.claude-sonnet-5": "arn:aws:bedrock:us-east-1:123456789012:default-prompt-router/prod-router"}'

    # Development: Use standard cross-region profile
    export AWS_BEDROCK_MODEL_ARN_MAPPING='{}'
    ```

!!! tip "Best Practices"
    - :material-test-tube: Test mappings in development before deploying to production
    - :material-file-document: Document your ARN mappings and their purposes
    - :material-update: Keep ARN mappings in version control alongside other configuration
    - :material-monitor: Monitor routing behavior after updating mappings

!!! warning "Startup Warning"
    If any model IDs in `AWS_BEDROCK_MODEL_ARN_MAPPING` are not found among available Bedrock models, a warning listing the affected entries is logged at startup. This typically means the model is not enabled in your configured regions or the model ID contains a typo.

## :material-directions-fork: Using Inference Profile and Prompt Router ARNs { #using-inference-profile-and-prompt-router-arns }

stdapi.ai supports passing ARNs directly as model IDs in API requests, enabling advanced routing capabilities beyond standard model selection.

!!! tip "Simplify ARNs with Model Aliases"
    Instead of using long ARNs directly in API requests, you can create [Model Aliases](operations_configuration_models.md#model-aliases) that map friendly names to ARNs. This provides shorter, easier-to-use naming for your API users.

### Overview

Instead of using standard model IDs like `anthropic.claude-sonnet-5`, you can pass ARNs that reference:

- **Cross-Region Inference Profiles** - AWS-managed multi-region routing
- **Application Inference Profiles** - Your custom routing configurations
- **Prompt Routers** - Intelligent dynamic model selection

!!! info "Automatic Cross-Region Routing"
    **stdapi.ai automatically handles cross-region routing by default.** When you use standard model IDs, the application automatically selects and uses the optimal AWS-managed cross-region inference profile based on your configured `AWS_BEDROCK_REGIONS`.

    You typically **do not need to manually pass cross-region inference profile ARNs**. The automatic selection handles routing across your configured regions for best availability and latency.

    Manual ARN passing is primarily useful for:

    - :material-application: **Application inference profiles** - Your custom routing configurations
    - :material-robot: **Prompt routers** - Intelligent cost optimization and dynamic model selection
    - :material-cog: **Rare cases** - When you need to override automatic cross-region profile selection

### Enabling ARN Support

By default, users can only pass standard model IDs. To allow ARN usage, enable the appropriate settings:

```bash
# Allow cross-region inference profile ARNs
export AWS_BEDROCK_ALLOW_CROSS_REGION_INFERENCE_PROFILE_ARN=true

# Allow application inference profile ARNs
export AWS_BEDROCK_ALLOW_APPLICATION_INFERENCE_PROFILE_ARN=true

# Allow prompt router ARNs
export AWS_BEDROCK_ALLOW_PROMPT_ROUTER_ARN=true
```

!!! warning "Security Consideration"
    These settings are disabled by default. Only enable them when you want to give users explicit control over ARN-based routing. For centralized server-controlled routing, use [`AWS_BEDROCK_MODEL_ARN_MAPPING`](#bedrock-model-arn-mapping) instead.

### Using ARNs in API Requests

Once enabled, users can pass ARNs directly in the `model` parameter:

**Cross-Region Inference Profile Example:**

```bash
curl -X POST https://api.example.com/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-sonnet-5",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

**Application Inference Profile Example:**

```bash
curl -X POST https://api.example.com/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-custom-profile",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

**Prompt Router Example:**

```bash
curl -X POST https://api.example.com/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "arn:aws:bedrock:us-east-1:123456789012:default-prompt-router/my-router",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### Use Case Comparison

| Approach                    | Best For                                    | Configuration                                                 |
|-----------------------------|---------------------------------------------|---------------------------------------------------------------|
| **Standard Model IDs**      | Most common use case, simple routing        | No special configuration needed                               |
| **Server-Side ARN Mapping** | Centralized control, transparent to clients | [`AWS_BEDROCK_MODEL_ARN_MAPPING`](#bedrock-model-arn-mapping) |
| **Client-Side ARN Passing** | User-controlled routing, advanced use cases | Enable `AWS_BEDROCK_ALLOW_*_ARN` settings                     |

### Best Practices

!!! success "Recommended Approach"
    **For most deployments, use server-side ARN mapping** ([`AWS_BEDROCK_MODEL_ARN_MAPPING`](#bedrock-model-arn-mapping)):

    - :material-server: Centralized control over routing behavior
    - :material-account-group: Transparent to API clients
    - :material-cog: Easy to change routing without modifying client code
    - :material-shield-check: Better security (server controls which ARNs are used)

!!! info "When to Allow Client-Side ARNs"
    Enable `AWS_BEDROCK_ALLOW_*_ARN` settings when:

    - :material-api: Clients need fine-grained control over routing
    - :material-cog: Different clients require different routing strategies
    - :material-dev-to: Advanced users managing their own inference profiles
    - :material-test-tube: Testing and comparing different routing configurations

!!! warning "Security and Governance"
    When enabling client-side ARN passing:

    - :material-shield-alert: Clients can bypass server-configured routing
    - :material-cash: Monitor usage to prevent unexpected costs
    - :material-account-check: Ensure appropriate IAM permissions are in place
    - :material-chart-line: Track ARN usage through logs and monitoring

### Required IAM Permissions

When using ARN-based routing, ensure your IAM role/user has the appropriate permissions:

```json
{
  "Sid": "BedrockARNRouting",
  "Effect": "Allow",
  "Action": [
    "bedrock:GetInferenceProfile",
    "bedrock:GetPromptRouter"
  ],
  "Resource": "*"
}
```

See the [IAM Permissions](operations_iam_permissions.md) page for complete policy examples.

---
