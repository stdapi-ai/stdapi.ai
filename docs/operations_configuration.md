---
title: Configuration Guide - Amazon Bedrock API Gateway Setup
description: Configuration hub for stdapi.ai - the quick start recipes, the settings that matter ranked by tier, an A-Z index of every environment variable, and the seven configuration reference pages.
keywords: AWS API gateway configuration, environment variables AWS, IAM permissions Bedrock, AWS regions setup, API authentication, compliance configuration, AWS credentials setup, S3 integration
---

# :material-cog: Configuration Guide

stdapi.ai is configured entirely through environment variables, which are read once at startup and cannot be changed without restarting the service. This guide explains each setting category with practical examples to help you configure the service correctly.

**What you can configure:**

- **AWS regions** - Access models across multiple regions for availability and model selection
- **Data sovereignty** - Control which AWS regions are used for compliance (GDPR, HIPAA, etc.)
- **Storage** - S3 buckets for file operations, regional buckets for multi-region deployments
- **Authentication** - API keys via SSM or Secrets Manager for secure access control
- **Observability** - Logging levels, OpenTelemetry, request/response debugging
- **Security** - CORS, proxy headers, trusted hosts for production deployments
- **Performance** - Caching, model overrides, S3 acceleration
- **TLS / SSL** - End-to-end encryption using Granian environment variables

!!! tip "Secure Defaults on Startup"
    Every setting below has a default, so stdapi.ai starts with none of them set: it uses the AWS credentials it runs with, detects your current AWS region, and discovers the available Bedrock models.

!!! info "Prerequisites"
    Before configuring stdapi.ai, ensure you have:

    - **AWS Account** with access to Amazon Bedrock
    - **AWS Credentials** configured via environment variables, AWS CLI, or IAM role (for EC2/ECS/Lambda deployments)
    - **IAM Permissions** to access required AWS services (see the [IAM Permissions](operations_iam_permissions.md) guide)
    - **S3 Bucket** (optional, but recommended for production use with file operations)

<span id="granian-host"></span>

!!! info "Container Runtime"
    Both the AWS Marketplace and community Docker images run using [Granian](https://github.com/emmett-framework/granian), a high-performance Python ASGI server. In addition to the stdapi.ai-specific configuration variables documented below, you can also use Granian environment variables to configure the server runtime (e.g., `GRANIAN_PORT`, `GRANIAN_WORKERS`, `GRANIAN_THREADS`, etc.).

    The images listen on IPv4 only (`GRANIAN_HOST=0.0.0.0`). Set `GRANIAN_HOST=::` to bind a dual-stack socket answering both IPv4 and IPv6 clients. This is needed wherever a client may resolve the server to an IPv6 address — in particular with ECS service discovery, which publishes an `AAAA` record for every task in an IPv6-enabled subnet, and some clients (Node.js among them) try that address first and fail with `ECONNREFUSED` against an IPv4-only listener. The [official Terraform module](https://github.com/stdapi-ai/terraform-aws-stdapi-ai) sets it for you when the VPC has IPv6 enabled.

## :material-rocket-launch: Quick Start

For production deployments, configure these essential settings:

### Minimal Production Setup

Single-region deployment with file storage only.

```bash
# S3 bucket for file storage (must be in same region as your server)
export AWS_S3_BUCKET=my-stdapi-bucket

# AWS_BEDROCK_REGIONS is optional - will auto-detect your current AWS region if not specified
```

### Production with Authentication

Adds secure API key authentication via AWS Systems Manager.

```bash
# S3 bucket for file storage (must be in same region as your server)
export AWS_S3_BUCKET=my-stdapi-bucket

# Secure API authentication (recommended: SSM Parameter Store)
export API_KEY_SSM_PARAMETER=/stdapi/prod/api-key

# AWS_BEDROCK_REGIONS is optional - will auto-detect your current AWS region if not specified
```

### Full Production Setup (All Features Enabled)

Multi-region deployment with all AWS AI services, observability, and security features.

```bash
# Core AWS configuration - host server in first region
export AWS_BEDROCK_REGIONS=us-east-1,us-west-2,eu-west-1

# S3 bucket for file storage (must be in us-east-1, your first/primary region)
export AWS_S3_BUCKET=my-stdapi-us-east-1-bucket

# Optional: Transcribe S3 bucket (defaults to AWS_S3_BUCKET if not specified)
# Only set this if you need a separate bucket or if transcribe is in a different region
# export AWS_TRANSCRIBE_S3_BUCKET=my-stdapi-transcribe-us-east-1

# Optional: Regional buckets for async/batch inference in other regions
export AWS_S3_REGIONAL_BUCKETS='{"us-west-2": "my-stdapi-us-west-2-bucket", "eu-west-1": "my-stdapi-eu-west-1-bucket"}'

# AWS AI services regions (optional - when unset, every AWS_BEDROCK_REGIONS entry is a
# candidate with automatic failover; set one to pin the service to a single region)
export AWS_POLLY_REGION=us-east-1           # Text-to-speech
export AWS_TRANSCRIBE_REGION=us-east-1      # Speech-to-text (audio transcription)
export AWS_COMPREHEND_REGION=us-east-1      # Language detection & moderation
export AWS_TRANSLATE_REGION=us-east-1       # Text translation

# Authentication
export API_KEY_SSM_PARAMETER=/stdapi/prod/api-key

# Logging
export LOG_LEVEL=warning
export LOG_CLIENT_IP=true

# Optional: OpenTelemetry observability (AWS X-Ray integration)
# export OTEL_ENABLED=true
# export OTEL_SERVICE_NAME=stdapi-production
# export OTEL_SAMPLE_RATE=0.1

# Production security settings (when behind AWS ALB/CloudFront)
export ENABLE_PROXY_HEADERS=true

# Note: TRUSTED_HOSTS not recommended with AWS ALB - use ALB host-based routing instead
# Only use TRUSTED_HOSTS if you cannot configure host validation at the load balancer level

# Optional: CORS for browser-based web applications
# export CORS_ALLOW_ORIGINS='["https://app.example.com"]'
```

### Development Setup

Local development configuration with API documentation and debug logging enabled.

```bash
# Minimal configuration for local development
export AWS_S3_BUCKET=my-stdapi-dev-bucket

# Enable API documentation
export ENABLE_DOCS=true
export ENABLE_REDOC=true

# Full request/response logging for debugging
export LOG_LEVEL=info
export LOG_REQUEST_PARAMS=true

# AWS_BEDROCK_REGIONS is optional - will auto-detect your current AWS region if not specified
```

!!! warning "S3 Bucket Required for Certain Features"
    Without an S3 bucket configured, some features will be disabled (such as image output as URL, audio transcription). See the relevant API documentation for feature requirements.

!!! info "All Other Settings Are Optional"
    The configurations above are sufficient for most production deployments. All other settings can be configured as needed for your specific use case.

## :material-format-list-bulleted: What to Set { #environment-variable-summary }

Most deployments set two variables. The [official Terraform module](https://github.com/stdapi-ai/terraform-aws-stdapi-ai) sets both for you. Everything else has a working default; each tier below tells you when to leave that default behind.

Two flags mark the rows to read in full before setting them: :material-shield-lock: **security** changes who can reach the API or what a caller may do, :material-currency-usd: **cost** adds or changes an AWS bill line.

### :material-star: Tier 1 — Set these (2) { #summary-essential }

| Variable | What it does | Set it when | Flags |
|---|---|---|---|
| [`AWS_S3_BUCKET`](operations_configuration_storage.md#aws-s3-bucket) | Primary S3 bucket for file storage; must be in the first region of `AWS_BEDROCK_REGIONS` | Always, unless you never call a file, image, video, audio or batch endpoint | :material-currency-usd: cost |
| [`AWS_BEDROCK_REGIONS`](operations_configuration_aws.md#aws-bedrock-regions) | Which regions serve models; the first is where the server should be hosted | You want models from more than one region, or a region other than the one the server runs in | |

### :material-tune: Tier 2 — Common (about 20) { #tier-2-common }

| Variable | What it does | Set it when | Flags |
|---|---|---|---|
| [`API_KEY_SSM_PARAMETER`](operations_configuration_authentication.md#api-key-ssm) | Client API key read from SSM Parameter Store | Any deployment reachable by more than you | :material-shield-lock: security |
| [`AUTHENTICATION_MODE`](operations_configuration_authentication.md#authentication-mode) | Which credentials the gateway accepts (API key, Cognito token, or both) | You put an identity provider in front of the gateway | :material-shield-lock: security |
| [`AWS_BEDROCK_USER_ROLE_ARN`](operations_configuration_bedrock.md#aws-bedrock-user-role-arn) | Per-end-user IAM role the gateway assumes for Bedrock calls | You need per-user attribution or per-user isolation in CloudTrail and billing | :material-shield-lock: security |
| [`AWS_S3_VECTORS_BUCKET`](operations_configuration_storage.md#aws-s3-vectors-bucket) | S3 vector bucket backing the Vector Stores API; unset disables it | You use file search or RAG | :material-currency-usd: cost |
| [`AWS_SQS_VECTOR_STORE_QUEUE_URL`](operations_configuration_storage.md#aws-sqs-vector-store-queue-url) | SQS queue that makes vector store indexing outlive the server running it | Vector store indexing must survive a restart or a scale-in | :material-currency-usd: cost |
| [`AWS_DYNAMODB_TABLE`](operations_configuration_storage.md#aws-dynamodb-table) | DynamoDB table holding the records a deployment's instances share | You run more than one instance and want conversations, sessions and caches shared | :material-currency-usd: cost |
| [`AWS_S3_REGIONAL_BUCKETS`](operations_configuration_storage.md#aws-s3-regional-buckets) | Region-specific buckets for Bedrock async and batch inference | You run batch or async inference in more than one region | :material-currency-usd: cost |
| [`AWS_TRANSCRIBE_S3_BUCKET`](operations_configuration_storage.md#aws-transcribe-s3-bucket) | Bucket for temporary transcription files | Transcribe runs in a region other than the one holding `AWS_S3_BUCKET` | :material-currency-usd: cost |
| [`AWS_BEDROCK_MANTLE_ENABLED`](operations_configuration_aws.md#bedrock-mantle-enabled) | Serves models through Bedrock Mantle instead of the Bedrock runtime | You want the Mantle model catalogue and its pricing | :material-currency-usd: cost |
| [`AWS_BEDROCK_MARKETPLACE_ENDPOINTS_ENABLED`](operations_configuration_models.md#bedrock-marketplace-endpoints-enabled) | Exposes AWS Marketplace model endpoints | You serve a Marketplace model | :material-currency-usd: cost |
| [`AWS_BEDROCK_GUARDRAIL_IDENTIFIER`](operations_configuration_bedrock.md#aws-bedrock-guardrail-identifier) | Applies a Bedrock guardrail to every eligible route | You must filter prompts or responses centrally | :material-shield-lock: security :material-currency-usd: cost |
| [`DEFAULT_MODEL_SERVICE_TIERS`](operations_configuration_models.md#default-model-service-tiers) | Per-model default service tier (`default`, `flex`, `priority`, `reserved`) | You want a cheaper or a faster tier without changing every client | :material-currency-usd: cost |
| [`MODEL_ALIASES`](operations_configuration_models.md#model-aliases) | Maps a name your clients already send onto a model you serve | Clients ask for a model name the gateway does not have | |
| [`COST_TRACKING`](operations_configuration_observability.md#cost-tracking) | Per-request cost estimates in the logs and the Usage API | You want to see spend per request or per user | |
| [`CLOUDWATCH_METRICS`](operations_configuration_observability.md#cloudwatch-metrics) | Publishes request, token and cost metrics to CloudWatch | You want dashboards and alarms on gateway traffic | :material-currency-usd: cost |
| [`USAGE_API`](operations_configuration_observability.md#usage-api) | Serves the OpenAI-compatible organization usage and cost endpoints | Clients or a billing job read usage back from the gateway | :material-shield-lock: security :material-currency-usd: cost |
| [`LOG_LEVEL`](operations_configuration_observability.md#logging-level) | How much the server logs | The default `warning` hides something you need, or logs cost too much | :material-currency-usd: cost |
| [`OTEL_ENABLED`](operations_configuration_observability.md#otel-enabled) | Emits OpenTelemetry traces (AWS X-Ray and any OTLP backend) | You already run distributed tracing | :material-currency-usd: cost |
| [`CORS_ALLOW_ORIGINS`](operations_configuration_server.md#cors-allow-origins) | Browser origins allowed to call the API | A browser application calls the gateway directly | :material-shield-lock: security |
| [`ENABLE_PROXY_HEADERS`](operations_configuration_server.md#enable-proxy-headers) | Trusts `X-Forwarded-*` for the client address and scheme | The gateway sits behind an ALB, CloudFront or another proxy | :material-shield-lock: security |
| [`TRUSTED_HOSTS`](operations_configuration_server.md#trusted-hosts) | Host header allowlist | You cannot validate the host at the load balancer | :material-shield-lock: security |
| [`ENABLE_DOCS`](operations_configuration_server.md#enable-docs) | Serves the interactive API documentation | You want Swagger UI on a non-public deployment | :material-shield-lock: security |

### :material-format-list-bulleted-square: Tier 3 — Everything else { #tier-3-everything-else }

Grouped by page. The defaults work; open a page when a Tier 1 or Tier 2 row above sends you there, or when the reason in the last column is yours.

| Page | Variables | Typical reason to open it |
|---|---|---|
| [Regions & AWS clients](operations_configuration_aws.md) | 24 | Region pinning, retry and connection tuning, failover backoff, Mantle, data residency |
| [Storage](operations_configuration_storage.md) | 22 | A separate bucket or prefix, lifecycle rules, vector stores, the shared DynamoDB table |
| [Models & routing](operations_configuration_models.md) | 26 | A model name your clients already use, a Marketplace or SageMaker model, per-model defaults |
| [Authentication & tenants](operations_configuration_authentication.md) | 18 | Cognito instead of an API key, per-tenant keys, the discovery documents agents read |
| [HTTP server & MCP](operations_configuration_server.md) | 28 | Mounting the routes elsewhere, a browser client, a proxy in front, TLS, MCP tool selection |
| [Bedrock features](operations_configuration_bedrock.md) | 25 | Guardrails, stored sessions, a cheaper or faster service tier, the Realtime API, ARN access |
| [Observability & usage](operations_configuration_observability.md) | 21 | Sending traces or metrics somewhere, per-request cost, the Usage API |

??? note "All variables A-Z"

    Every documented environment variable, with the page that documents it.

    | Variable | Documented in |
    |---|---|
    | [`AI_RESPONSE_TIMEOUT`](operations_configuration_server.md#ai-response-timeout) | HTTP server & MCP |
    | [`ANTHROPIC_BETA_ALLOWLIST`](operations_configuration_models.md#anthropic-beta-allowlist) | Models & routing |
    | [`ANTHROPIC_BETA_FILTER`](operations_configuration_models.md#anthropic-beta-filter) | Models & routing |
    | [`ANTHROPIC_ROUTES_PREFIX`](operations_configuration_server.md#anthropic-routes-prefix) | HTTP server & MCP |
    | [`API_KEY`](operations_configuration_authentication.md#api-key) | Authentication & tenants |
    | [`API_KEY_SECRETSMANAGER_KEY`](operations_configuration_authentication.md#api-key-secretsmanager-key) | Authentication & tenants |
    | [`API_KEY_SECRETSMANAGER_SECRET`](operations_configuration_authentication.md#api-key-secretsmanager-secret) | Authentication & tenants |
    | [`API_KEY_SSM_PARAMETER`](operations_configuration_authentication.md#api-key-ssm) | Authentication & tenants |
    | [`AUTHENTICATION_MODE`](operations_configuration_authentication.md#authentication-mode) | Authentication & tenants |
    | [`AWS_ADAPTIVE_RETRY`](operations_configuration_aws.md#aws-adaptive-retry) | Regions & AWS clients |
    | [`AWS_BEDROCK_ALLOW_APPLICATION_INFERENCE_PROFILE_ARN`](operations_configuration_bedrock.md#bedrock-allow-application-profile-arn) | Bedrock features |
    | [`AWS_BEDROCK_ALLOW_CROSS_REGION_INFERENCE_PROFILE_ARN`](operations_configuration_bedrock.md#bedrock-allow-cross-region-profile-arn) | Bedrock features |
    | [`AWS_BEDROCK_ALLOW_EXTERNAL_WEB_ACCESS_OVERRIDE`](operations_configuration_models.md#bedrock-allow-external-web-access-override) | Models & routing |
    | [`AWS_BEDROCK_ALLOW_GUARDRAIL_OVERRIDE`](operations_configuration_bedrock.md#aws-bedrock-allow-guardrail-override) | Bedrock features |
    | [`AWS_BEDROCK_ALLOW_MANTLE_PROJECT_OVERRIDE`](operations_configuration_aws.md#bedrock-allow-mantle-project-override) | Regions & AWS clients |
    | [`AWS_BEDROCK_ALLOW_MARKETPLACE_ENDPOINT_ARN`](operations_configuration_models.md#bedrock-allow-marketplace-endpoint-arn) | Models & routing |
    | [`AWS_BEDROCK_ALLOW_PROMPT_ARN`](operations_configuration_bedrock.md#bedrock-allow-prompt-arn) | Bedrock features |
    | [`AWS_BEDROCK_ALLOW_PROMPT_ROUTER_ARN`](operations_configuration_bedrock.md#bedrock-allow-prompt-router-arn) | Bedrock features |
    | [`AWS_BEDROCK_ALLOW_SERVICE_TIER_OVERRIDE`](operations_configuration_models.md#aws-bedrock-allow-service-tier-override) | Models & routing |
    | [`AWS_BEDROCK_BATCH_ROLE_ARN`](operations_configuration_bedrock.md#aws-bedrock-batch-role-arn) | Bedrock features |
    | [`AWS_BEDROCK_CROSS_REGION_INFERENCE`](operations_configuration_aws.md#cross-region-inference) | Regions & AWS clients |
    | [`AWS_BEDROCK_CROSS_REGION_INFERENCE_GLOBAL`](operations_configuration_aws.md#cross-region-global) | Regions & AWS clients |
    | [`AWS_BEDROCK_DEPRECATED_MODELS`](operations_configuration_models.md#bedrock-deprecated-models) | Models & routing |
    | [`AWS_BEDROCK_DEPRECATED_MODEL_FALLBACK`](operations_configuration_models.md#bedrock-deprecated-model-fallback) | Models & routing |
    | [`AWS_BEDROCK_EXTERNAL_WEB_ACCESS`](operations_configuration_models.md#bedrock-external-web-access) | Models & routing |
    | [`AWS_BEDROCK_GUARDRAIL_CHECKS_PII_ENTITIES`](operations_configuration_bedrock.md#aws-bedrock-guardrail-checks-pii-entities) | Bedrock features |
    | [`AWS_BEDROCK_GUARDRAIL_CHECKS_PROMPT_ATTACK`](operations_configuration_bedrock.md#aws-bedrock-guardrail-checks-prompt-attack) | Bedrock features |
    | [`AWS_BEDROCK_GUARDRAIL_IDENTIFIER`](operations_configuration_bedrock.md#aws-bedrock-guardrail-identifier) | Bedrock features |
    | [`AWS_BEDROCK_GUARDRAIL_TRACE`](operations_configuration_bedrock.md#aws-bedrock-guardrail-trace) | Bedrock features |
    | [`AWS_BEDROCK_GUARDRAIL_VERSION`](operations_configuration_bedrock.md#aws-bedrock-guardrail-version) | Bedrock features |
    | [`AWS_BEDROCK_KNOWLEDGE_BASE_IDS`](operations_configuration_storage.md#aws-bedrock-knowledge-base-ids) | Storage |
    | [`AWS_BEDROCK_LEGACY`](operations_configuration_models.md#bedrock-legacy) | Models & routing |
    | [`AWS_BEDROCK_MANTLE_ENABLED`](operations_configuration_aws.md#bedrock-mantle-enabled) | Regions & AWS clients |
    | [`AWS_BEDROCK_MANTLE_ENDPOINT_URL`](operations_configuration_aws.md#bedrock-mantle-endpoint-url) | Regions & AWS clients |
    | [`AWS_BEDROCK_MANTLE_PREFERRED_MODELS`](operations_configuration_models.md#bedrock-mantle-preferred-models) | Models & routing |
    | [`AWS_BEDROCK_MANTLE_PROJECT`](operations_configuration_aws.md#bedrock-mantle-project) | Regions & AWS clients |
    | [`AWS_BEDROCK_MANTLE_REGIONS`](operations_configuration_aws.md#bedrock-mantle-regions) | Regions & AWS clients |
    | [`AWS_BEDROCK_MANTLE_SERVICE_HEADER`](operations_configuration_aws.md#bedrock-mantle-service-header) | Regions & AWS clients |
    | [`AWS_BEDROCK_MARKETPLACE_AUTO_SUBSCRIBE`](operations_configuration_models.md#bedrock-marketplace-auto-subscribe) | Models & routing |
    | [`AWS_BEDROCK_MARKETPLACE_ENDPOINTS_ENABLED`](operations_configuration_models.md#bedrock-marketplace-endpoints-enabled) | Models & routing |
    | [`AWS_BEDROCK_MARKETPLACE_ENDPOINT_REGIONS`](operations_configuration_models.md#bedrock-marketplace-endpoint-regions) | Models & routing |
    | [`AWS_BEDROCK_MAX_RETRIES`](operations_configuration_aws.md#bedrock-max-retries) | Regions & AWS clients |
    | [`AWS_BEDROCK_MODEL_ARN_MAPPING`](operations_configuration_bedrock.md#bedrock-model-arn-mapping) | Bedrock features |
    | [`AWS_BEDROCK_MODEL_REGION_RESTRICT`](operations_configuration_aws.md#bedrock-model-region-restrict) | Regions & AWS clients |
    | [`AWS_BEDROCK_REGIONS`](operations_configuration_aws.md#aws-bedrock-regions) | Regions & AWS clients |
    | [`AWS_BEDROCK_REGION_ROUTING`](operations_configuration_aws.md#bedrock-region-routing) | Regions & AWS clients |
    | [`AWS_BEDROCK_REGION_ROUTING_MAX_QUOTA_BACKOFF_SECONDS`](operations_configuration_aws.md#bedrock-region-routing-max-quota-backoff) | Regions & AWS clients |
    | [`AWS_BEDROCK_REGION_ROUTING_QUOTA_BACKOFF_SECONDS`](operations_configuration_aws.md#bedrock-region-routing-quota-backoff) | Regions & AWS clients |
    | [`AWS_BEDROCK_REGION_ROUTING_QUOTA_STALE_FACTOR`](operations_configuration_aws.md#bedrock-region-routing-quota-stale-factor) | Regions & AWS clients |
    | [`AWS_BEDROCK_REGION_ROUTING_UNAVAILABLE_BACKOFF_SECONDS`](operations_configuration_aws.md#bedrock-region-routing-unavailable-backoff) | Regions & AWS clients |
    | [`AWS_BEDROCK_SESSION_ENCRYPTION_KEY_ARN`](operations_configuration_bedrock.md#aws-bedrock-session-encryption-key-arn) | Bedrock features |
    | [`AWS_BEDROCK_USER_ROLE_ARN`](operations_configuration_bedrock.md#aws-bedrock-user-role-arn) | Bedrock features |
    | [`AWS_BEDROCK_USER_ROLE_REQUIRE_IDENTITY`](operations_configuration_bedrock.md#aws-bedrock-user-role-require-identity) | Bedrock features |
    | [`AWS_BEDROCK_USER_ROLE_SESSION_DURATION`](operations_configuration_bedrock.md#aws-bedrock-user-role-session-duration) | Bedrock features |
    | [`AWS_BEDROCK_USER_ROLE_TAG_KEY`](operations_configuration_bedrock.md#aws-bedrock-user-role-tag-key) | Bedrock features |
    | [`AWS_COGNITO_ACCEPT_ID_TOKEN`](operations_configuration_authentication.md#aws-cognito-accept-id-token) | Authentication & tenants |
    | [`AWS_COGNITO_CLIENT_IDS`](operations_configuration_authentication.md#aws-cognito-client-ids) | Authentication & tenants |
    | [`AWS_COGNITO_ISSUER_TYPE`](operations_configuration_authentication.md#aws-cognito-issuer-type) | Authentication & tenants |
    | [`AWS_COGNITO_REQUIRED_SCOPES`](operations_configuration_authentication.md#aws-cognito-required-scopes) | Authentication & tenants |
    | [`AWS_COGNITO_USER_POOL_ID`](operations_configuration_authentication.md#aws-cognito-user-pool-id) | Authentication & tenants |
    | [`AWS_COMPREHEND_REGION`](operations_configuration_aws.md#aws-comprehend-region) | Regions & AWS clients |
    | [`AWS_CONNECT_TIMEOUT`](operations_configuration_aws.md#aws-connect-timeout) | Regions & AWS clients |
    | [`AWS_DYNAMODB_REGION`](operations_configuration_storage.md#aws-dynamodb-region) | Storage |
    | [`AWS_DYNAMODB_TABLE`](operations_configuration_storage.md#aws-dynamodb-table) | Storage |
    | [`AWS_FAILOVER_MAX_RETRIES`](operations_configuration_aws.md#failover-max-retries) | Regions & AWS clients |
    | [`AWS_MAX_POOL_CONNECTIONS`](operations_configuration_aws.md#aws-max-pool-connections) | Regions & AWS clients |
    | [`AWS_POLLY_REGION`](operations_configuration_aws.md#aws-polly-region) | Regions & AWS clients |
    | [`AWS_S3_ACCELERATE`](operations_configuration_storage.md#aws-s3-accelerate) | Storage |
    | [`AWS_S3_ACCEPTED_BUCKETS`](operations_configuration_storage.md#aws-s3-accepted-buckets) | Storage |
    | [`AWS_S3_BATCHES_PREFIX`](operations_configuration_storage.md#aws-s3-batches-prefix) | Storage |
    | [`AWS_S3_BUCKET`](operations_configuration_storage.md#aws-s3-bucket) | Storage |
    | [`AWS_S3_FILES_PREFIX`](operations_configuration_storage.md#aws-s3-files-prefix) | Storage |
    | [`AWS_S3_REGIONAL_BUCKETS`](operations_configuration_storage.md#aws-s3-regional-buckets) | Storage |
    | [`AWS_S3_TMP_PREFIX`](operations_configuration_storage.md#aws-s3-tmp-prefix) | Storage |
    | [`AWS_S3_VECTORS_BUCKET`](operations_configuration_storage.md#aws-s3-vectors-bucket) | Storage |
    | [`AWS_S3_VECTORS_REGION`](operations_configuration_storage.md#aws-s3-vectors-region) | Storage |
    | [`AWS_S3_VECTOR_STORES_PREFIX`](operations_configuration_storage.md#aws-s3-vector-stores-prefix) | Storage |
    | [`AWS_S3_VIDEOS_EXPIRES_AFTER`](operations_configuration_storage.md#aws-s3-videos-expires-after) | Storage |
    | [`AWS_S3_VIDEOS_PREFIX`](operations_configuration_storage.md#aws-s3-videos-prefix) | Storage |
    | [`AWS_SAGEMAKER_ENDPOINTS`](operations_configuration_models.md#aws-sagemaker-endpoints) | Models & routing |
    | [`AWS_SAGEMAKER_ENDPOINT_URL`](operations_configuration_models.md#aws-sagemaker-endpoint-url) | Models & routing |
    | [`AWS_SAGEMAKER_WARMUP_TIMEOUT`](operations_configuration_models.md#aws-sagemaker-warmup-timeout) | Models & routing |
    | [`AWS_SQS_VECTOR_STORE_QUEUE_URL`](operations_configuration_storage.md#aws-sqs-vector-store-queue-url) | Storage |
    | [`AWS_TRANSCRIBE_OUTPUT_ENCRYPTION_KEY_ARN`](operations_configuration_storage.md#aws-transcribe-output-encryption-key-arn) | Storage |
    | [`AWS_TRANSCRIBE_REGION`](operations_configuration_aws.md#aws-transcribe-region) | Regions & AWS clients |
    | [`AWS_TRANSCRIBE_S3_BUCKET`](operations_configuration_storage.md#aws-transcribe-s3-bucket) | Storage |
    | [`AWS_TRANSCRIBE_STREAM_LANGUAGES`](operations_configuration_storage.md#aws-transcribe-stream-languages) | Storage |
    | [`AWS_TRANSLATE_REGION`](operations_configuration_aws.md#aws-translate-region) | Regions & AWS clients |
    | [`CHAT_COMPLETIONS_REASONING_FIELD`](operations_configuration_observability.md#chat-completions-reasoning-field) | Observability & usage |
    | [`CLOUDWATCH_METRICS`](operations_configuration_observability.md#cloudwatch-metrics) | Observability & usage |
    | [`CLOUDWATCH_METRICS_NAMESPACE`](operations_configuration_observability.md#cloudwatch-metrics-namespace) | Observability & usage |
    | [`CLOUDWATCH_METRICS_REGION`](operations_configuration_observability.md#cloudwatch-metrics-region) | Observability & usage |
    | [`CLOUDWATCH_METRICS_USER_DIMENSION`](operations_configuration_observability.md#cloudwatch-metrics-user-dimension) | Observability & usage |
    | [`COHERE_ROUTES_PREFIX`](operations_configuration_server.md#cohere-routes-prefix) | HTTP server & MCP |
    | [`CORS_ALLOW_ORIGINS`](operations_configuration_server.md#cors-allow-origins) | HTTP server & MCP |
    | [`COST_PRICE_OVERRIDES`](operations_configuration_observability.md#cost-price-overrides) | Observability & usage |
    | [`COST_TRACKING`](operations_configuration_observability.md#cost-tracking) | Observability & usage |
    | [`DEFAULT_MODEL_PARAMS`](operations_configuration_models.md#default-model-params) | Models & routing |
    | [`DEFAULT_MODEL_SERVICE_TIERS`](operations_configuration_models.md#default-model-service-tiers) | Models & routing |
    | [`DEFAULT_TTS_LANGUAGE`](operations_configuration_bedrock.md#default-tts-language) | Bedrock features |
    | [`DEFAULT_TTS_MODEL`](operations_configuration_bedrock.md#default-tts-model) | Bedrock features |
    | [`DROP_UNSUPPORTED_SYSTEM_PROMPT`](operations_configuration_models.md#drop-unsupported-system-prompt) | Models & routing |
    | [`ENABLE_DOCS`](operations_configuration_server.md#enable-docs) | HTTP server & MCP |
    | [`ENABLE_GZIP`](operations_configuration_server.md#enable-gzip) | HTTP server & MCP |
    | [`ENABLE_MCP_SSE`](operations_configuration_server.md#enable-mcp-sse) | HTTP server & MCP |
    | [`ENABLE_MCP_STREAMABLE_HTTP`](operations_configuration_server.md#enable-mcp-streamable-http) | HTTP server & MCP |
    | [`ENABLE_OPENAPI_JSON`](operations_configuration_server.md#enable-openapi-json) | HTTP server & MCP |
    | [`ENABLE_PROXY_HEADERS`](operations_configuration_server.md#enable-proxy-headers) | HTTP server & MCP |
    | [`ENABLE_REDOC`](operations_configuration_server.md#enable-redoc) | HTTP server & MCP |
    | [`EXTRA_MODEL_PARAMS_DENYLIST`](operations_configuration_models.md#extra-model-params-denylist) | Models & routing |
    | [`EXTRA_MODEL_PARAMS_DROP_ALL`](operations_configuration_models.md#extra-model-params-drop-all) | Models & routing |
    | [`GRANIAN_HOST`](operations_configuration.md#granian-host) | This page |
    | [`GRANIAN_SSL_CA`](operations_configuration_server.md#graniansslca) | HTTP server & MCP |
    | [`GRANIAN_SSL_CERTIFICATE`](operations_configuration_server.md#graniansslcertificate) | HTTP server & MCP |
    | [`GRANIAN_SSL_CLIENT_VERIFY`](operations_configuration_server.md#graniansslclientverify) | HTTP server & MCP |
    | [`GRANIAN_SSL_KEYFILE`](operations_configuration_server.md#graniansslkeyfile) | HTTP server & MCP |
    | [`GRANIAN_SSL_KEYFILE_PASSWORD`](operations_configuration_server.md#graniansslkeyfilepassword) | HTTP server & MCP |
    | [`GRANIAN_SSL_PROTOCOL_MIN`](operations_configuration_server.md#graniansslprotocolmin) | HTTP server & MCP |
    | [`IMAGE_GENERATION_MODEL`](operations_configuration_models.md#image-generation-model) | Models & routing |
    | [`LOG_CLIENT_IP`](operations_configuration_observability.md#client-ip-logging) | Observability & usage |
    | [`LOG_LEVEL`](operations_configuration_observability.md#logging-level) | Observability & usage |
    | [`LOG_REQUEST_PARAMS`](operations_configuration_observability.md#log-request-params) | Observability & usage |
    | [`MAX_CONCURRENT_INPUT_DOWNLOADS`](operations_configuration_server.md#max-concurrent-input-downloads) | HTTP server & MCP |
    | [`MAX_INPUT_FILE_SIZE`](operations_configuration_server.md#max-input-file-size) | HTTP server & MCP |
    | [`MCP_EXCLUDE_TOOLS`](operations_configuration_server.md#mcp-exclude-tools) | HTTP server & MCP |
    | [`MCP_INCLUDE_TOOLS`](operations_configuration_server.md#mcp-include-tools) | HTTP server & MCP |
    | [`MCP_STATELESS_HTTP`](operations_configuration_server.md#mcp-stateless-http) | HTTP server & MCP |
    | [`MODEL_ALIASES`](operations_configuration_models.md#model-aliases) | Models & routing |
    | [`MODEL_CACHE_MAX_STALE_SECONDS`](operations_configuration_models.md#model-cache-max-stale-seconds) | Models & routing |
    | [`MODEL_CACHE_SECONDS`](operations_configuration_models.md#model-cache-seconds) | Models & routing |
    | [`MODEL_CACHE_SHARED`](operations_configuration_models.md#model-cache-shared) | Models & routing |
    | [`OAUTH_AUTHORIZATION_SERVERS`](operations_configuration_authentication.md#oauth-authorization-servers) | Authentication & tenants |
    | [`OAUTH_RESOURCE_IDENTIFIER`](operations_configuration_authentication.md#oauth-resource-identifier) | Authentication & tenants |
    | [`OAUTH_SCOPES_SUPPORTED`](operations_configuration_authentication.md#oauth-scopes-supported) | Authentication & tenants |
    | [`OLLAMA_ROUTES_PREFIX`](operations_configuration_server.md#ollama-routes-prefix) | HTTP server & MCP |
    | [`OPENAI_ROUTES_PREFIX`](operations_configuration_server.md#openai-routes-prefix) | HTTP server & MCP |
    | [`OTEL_ENABLED`](operations_configuration_observability.md#otel-enabled) | Observability & usage |
    | [`OTEL_EXPORTER_ENDPOINT`](operations_configuration_observability.md#otel-exporter-endpoint) | Observability & usage |
    | [`OTEL_SAMPLE_RATE`](operations_configuration_observability.md#otel-sample-rate) | Observability & usage |
    | [`OTEL_SERVICE_NAME`](operations_configuration_observability.md#otel-service-name) | Observability & usage |
    | [`PROXY_TRUSTED_HOSTS`](operations_configuration_server.md#proxy-trusted-hosts) | HTTP server & MCP |
    | [`REALTIME_ALLOW_SESSION_OVERRIDE`](operations_configuration_bedrock.md#realtime-allow-session-override) | Bedrock features |
    | [`REALTIME_CLIENT_SECRET_KEY`](operations_configuration_bedrock.md#realtime-client-secret-key) | Bedrock features |
    | [`REALTIME_WEBRTC_ALLOW_PRIVATE_CANDIDATES`](operations_configuration_bedrock.md#realtime-webrtc-allow-private-candidates) | Bedrock features |
    | [`REALTIME_WEBRTC_ENABLED`](operations_configuration_bedrock.md#realtime-webrtc-enabled) | Bedrock features |
    | [`REALTIME_WEBRTC_STUN_SERVER`](operations_configuration_bedrock.md#realtime-webrtc-stun-server) | Bedrock features |
    | [`REALTIME_WEBRTC_TURN_PASSWORD`](operations_configuration_bedrock.md#realtime-webrtc-turn-server) | Bedrock features |
    | [`REALTIME_WEBRTC_TURN_SERVER`](operations_configuration_bedrock.md#realtime-webrtc-turn-server) | Bedrock features |
    | [`REALTIME_WEBRTC_TURN_USERNAME`](operations_configuration_bedrock.md#realtime-webrtc-turn-server) | Bedrock features |
    | [`SHUTDOWN_DRAIN_TIMEOUT`](operations_configuration_server.md#shutdown-drain-timeout) | HTTP server & MCP |
    | [`SSRF_PROTECTION_BLOCK_PRIVATE_NETWORKS`](operations_configuration_server.md#ssrf-protection-block-private-networks) | HTTP server & MCP |
    | [`STRICT_INPUT_VALIDATION`](operations_configuration_observability.md#strict-input-validation) | Observability & usage |
    | [`TENANT_API_KEYS`](operations_configuration_authentication.md#tenant-api-keys) | Authentication & tenants |
    | [`TENANT_AWS_CREDENTIALS`](operations_configuration_authentication.md#tenant-aws-credentials) | Authentication & tenants |
    | [`TENANT_KEY_CACHE_SECONDS`](operations_configuration_authentication.md#tenant-key-cache-seconds) | Authentication & tenants |
    | [`TENANT_KEY_SSM_KMS_KEY_ID`](operations_configuration_authentication.md#tenant-key-ssm-kms-key-id) | Authentication & tenants |
    | [`TENANT_KEY_SSM_PARAMETER_PREFIX`](operations_configuration_authentication.md#tenant-key-ssm-parameter-prefix) | Authentication & tenants |
    | [`TIMEZONE`](operations_configuration_observability.md#timezone) | Observability & usage |
    | [`TOKENS_ESTIMATION`](operations_configuration.md#tokens-estimation) | This page |
    | [`TOKENS_ESTIMATION_DEFAULT_ENCODING`](operations_configuration.md#tokens-encoding) | This page |
    | [`TRUSTED_HOSTS`](operations_configuration_server.md#trusted-hosts) | HTTP server & MCP |
    | [`USAGE_API`](operations_configuration_observability.md#usage-api) | Observability & usage |
    | [`USAGE_API_ADMIN_SCOPES`](operations_configuration_observability.md#usage-api-admin-scopes) | Observability & usage |
    | [`USAGE_API_CACHE_TTL`](operations_configuration_observability.md#usage-api-cache-ttl) | Observability & usage |
    | [`USAGE_API_MAX_METRICS`](operations_configuration_observability.md#usage-api-max-metrics) | Observability & usage |
    | [`USAGE_API_MAX_RANGE_DAYS`](operations_configuration_observability.md#usage-api-max-range-days) | Observability & usage |
    | [`VECTOR_STORE_CHUNK_OVERLAP_TOKENS`](operations_configuration_storage.md#vector-store-chunk-overlap-tokens) | Storage |
    | [`VECTOR_STORE_CHUNK_SIZE_TOKENS`](operations_configuration_storage.md#vector-store-chunk-size-tokens) | Storage |
    | [`VECTOR_STORE_EMBEDDING_MODEL`](operations_configuration_storage.md#vector-store-embedding-model) | Storage |

---

## :material-sort-numeric-ascending: Configuration Order

When deploying stdapi.ai, configure settings in this recommended order:

1. **[IAM Permissions](operations_iam_permissions.md)** - Set up AWS access first
2. **[AWS Services and Regions](operations_configuration_aws.md#aws-services-and-regions)** - Choose the Bedrock regions that serve your models, and how requests fail over between them
3. **[Storage](operations_configuration_storage.md#storage)** - Point file, image, video, batch and vector store features at their S3 buckets
4. **[Authentication](operations_configuration_authentication.md#authentication)** - Secure your API with authentication
5. **Optional features** - Add [observability](operations_configuration_observability.md#observability-opentelemetry), the [Bedrock features](operations_configuration_bedrock.md) such as guardrails, and [model routing](operations_configuration_models.md) as needed

---

## :material-shield-key: IAM Permissions { #iam-permissions }

<span id="bedrock-iam"></span>
<span id="bedrock-mantle-iam"></span>
<span id="speech-to-text-optional"></span>

Every AWS permission the gateway needs is on the [IAM Permissions](operations_iam_permissions.md) page: the Amazon Bedrock statements every deployment requires, then one section per optional feature — S3 file storage, vector stores, the shared table, speech, translation, cost tracking, the Usage API, tenant keys — each listing the exact actions and resources it adds. It ends with copy-ready complete policy examples and the AWS tag policy requirements. Go there to write the role your deployment runs under, or when a feature returns an access-denied error.

---

## :material-archive: Deprecated Settings

<span id="tokens-estimation"></span>
<span id="tokens-encoding"></span>

!!! warning "Deprecated and Ignored"
    `TOKENS_ESTIMATION` (default: `false`) and `TOKENS_ESTIMATION_DEFAULT_ENCODING` (default: `None`) are deprecated and ignored: tiktoken-based token estimation has been removed from the project. Token counts are now sourced directly from AWS billing data when available. Remove these variables from existing configurations.

---

## :material-arrow-right: Next Steps

<div class="grid cards" markdown>

- :material-shield-key: [**IAM Permissions**](operations_iam_permissions.md) — Complete IAM policy reference
- :material-lock: [**Authentication & Security**](operations_authentication_security.md) — Secure your deployment
- :material-shield-check: [**Resilience & Failover**](operations_resilience.md) — Region routing and failover behavior
- :material-chart-line: [**Logging & Monitoring**](operations_logging_monitoring.md) — Observability and metrics

</div>
