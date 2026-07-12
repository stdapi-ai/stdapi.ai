---
title: Releases & Roadmap - Active Development
description: stdapi.ai release history and upcoming features. Track regular updates, new AWS Bedrock capabilities, and active development progress.
keywords: stdapi.ai releases, AI gateway updates, AWS Bedrock features, API gateway roadmap, software changelog, active development, new AI features, product updates
---

# :material-timeline: Releases & Roadmap

**stdapi.ai is under active development** with regular feature releases.

## :material-tag-multiple: Recent Releases

See [Release History below](#release-history) for the full changelog of all releases.

**Latest: v1.14.0** – Amazon Bedrock Mantle support (enabled by default: OpenAI GPT-5.x, xAI Grok 4.3, Google Gemma 4, and more through every text API), video generation (OpenAI Videos API on Amazon Nova Reel and Luma Ray 2), Cohere-compatible Rerank and Embed APIs, content moderation via Bedrock Guardrails or Amazon Comprehend, extended reasoning as native Responses API reasoning items, stored responses and chat completions with a full list/retrieve/update/delete lifecycle and multi-turn continuation, conversation compaction, a model pricing API, multi-region failover for all AWS AI services, a security hardening pass, and broad Responses API compatibility improvements validated against agent SDKs (OpenAI Codex CLI)

---

## :material-rocket-launch: Roadmap

Pending features and current deployment state are tracked on the [GitHub Project](https://github.com/orgs/stdapi-ai/projects/1).

---

## :material-history: Release History

### v1.14.0 – Bedrock Mantle, Video Generation, Cohere APIs, Moderation & Stored Conversations

This release adds enabled-by-default [**Amazon Bedrock Mantle** support](features.md#bedrock-mantle-models) — models served by the Bedrock Mantle endpoint (OpenAI GPT-5.4/5.5/5.6, xAI Grok 4.3, Google Gemma 4, Qwen3, GLM, DeepSeek, MiniMax, Kimi, Nemotron, and more) become available through all four text APIs, with transparent API conversion, native stored conversations, and independent throughput quotas. It also turns stdapi.ai into a three-dialect gateway with the new **Cohere-compatible API** ([Rerank](api_cohere_rerank.md) and [Embed](api_cohere_embed.md)), adds the OpenAI-compatible [**Videos API**](api_openai_videos.md) for asynchronous video generation, [**content moderation**](api_openai_moderations.md) backed by AWS Bedrock Guardrails or Amazon Comprehend toxicity detection, **stored responses and chat completions** with `store=true`, `previous_response_id` multi-turn continuation, and a full list/retrieve/update/delete lifecycle on AWS Bedrock session storage, and [**conversation compaction**](api_openai_responses.md#conversation-compaction). The Responses API gains [**extended reasoning**](api_openai_responses.md#extended-reasoning): Bedrock `reasoningContent` now surfaces as native reasoning output items, both non-streaming and streamed, with signatures and redacted payloads round-tripping through an `encrypted_content` envelope. A broader compatibility pass brings request/response parity closer to the OpenAI SDK — hosted and agent tool types (web search, computer use, custom tools) are now accepted and ignored instead of rejected, streams correctly terminate with `response.incomplete`/`response.failed`, cached tokens are counted in `input_tokens`, and citation annotations are emitted with their streaming events — validated end-to-end against the OpenAI Codex CLI as an agent client. Operations gain a [model pricing API](api_model_pricing.md), multi-region failover for every AWS AI service, fault-tolerant startup, real AWS-billed usage and costs in request logs (optionally exported as CloudWatch metrics), and a [security hardening pass](#security-hardening) covering SSRF protection, input validation, and log/error redaction.

!!! warning "New Required IAM Permissions"
    v1.14.0 requires two new IAM permissions:

    - **`bedrock:Rerank`** — needed for the [Cohere-compatible Rerank API](api_cohere_rerank.md) (`/cohere/v2/rerank`). See [IAM Permissions](operations_configuration.md#bedrock-iam).
    - **`bedrock:ListAsyncInvokes`**, plus **`bedrock:ListTagsForResource`** on `arn:aws:bedrock:*:*:async-invoke/*` — needed for `GET /v1/videos` (listing video generation jobs across regions). See [IAM Permissions](operations_configuration.md#bedrock-iam).

    Ensure your IAM role or user policy includes both statements before upgrading to v1.14.0.

    !!! note "Session storage and Comprehend permissions already covered"
        The IAM permissions for [stored responses/chat completions](operations_configuration.md#bedrock-session-storage-optional) (`bedrock:CreateSession` and related session actions) and [Comprehend-based moderation](operations_configuration.md#iam-permissions) (`comprehend:DetectToxicContent`) were already added to the official [stdapi-ai Terraform module](https://github.com/stdapi-ai/terraform-aws-stdapi-ai) ahead of this release. Deployments using a hand-written policy still need to add those statements if they haven't already. Without the session permissions, `store=true` (previously accepted and ignored) is still ignored — a warning is recorded in the request log instead of failing the request.

#### :material-layers-triple: Amazon Bedrock Mantle

Enabled-by-default support ([`AWS_BEDROCK_MANTLE_ENABLED`](operations_configuration.md#bedrock-mantle-enabled)) for models served by the **Amazon Bedrock Mantle** endpoint — OpenAI GPT-5.4/5.5/5.6 (Sol, Terra, Luna), xAI Grok 4.3, Google Gemma 4, Qwen3, GLM 4.x/5, DeepSeek V3.x, MiniMax M2.x, Kimi K2.5, Nemotron, and more — alongside the classic Bedrock Converse catalog:

- All four text APIs (chat completions, responses, messages, legacy completions) are served for every Mantle model — native passthrough where the model supports the API upstream, transparent conversion otherwise
- Models available on both bedrock-runtime and Mantle are served by bedrock-runtime by default; [`AWS_BEDROCK_MANTLE_PREFERRED_MODELS`](operations_configuration.md#bedrock-mantle-preferred-models) or the opt-in `x-stdapi-service` request header ([`AWS_BEDROCK_MANTLE_SERVICE_HEADER`](operations_configuration.md#bedrock-mantle-service-header)) route them through Mantle instead — e.g. to tap Mantle's independent throughput quotas
- Native Mantle stored conversations on `/v1/responses` (`store`, `previous_response_id`, retrieval and deletion) — 30-day retention, region-local, project-scoped
- Multi-region failover and quota backoff across [`AWS_BEDROCK_MANTLE_REGIONS`](operations_configuration.md#bedrock-mantle-regions), matching classic Bedrock region routing
- Authentication via short-term bearer tokens derived from the server's AWS credential chain — no static secrets
- Usage recorded and priced at bedrock-mantle rates, including cached tokens and service tiers

[:octicons-arrow-right-24: Bedrock Mantle Models](features.md#bedrock-mantle-models)

!!! warning "Additional IAM Permissions (opt-in feature)"
    Enabling `AWS_BEDROCK_MANTLE_ENABLED` requires the `bedrock-mantle:CreateInference`, `bedrock-mantle:GetInference`, `bedrock-mantle:DeleteInference`, `bedrock-mantle:ListModels`, and `bedrock-mantle:GetModel` permissions on `arn:aws:bedrock-mantle:*:*:project/*`, plus `bedrock-mantle:CallWithBearerToken` on `*`. See [IAM Permissions](operations_configuration.md#bedrock-mantle-iam).

#### :material-api: New APIs

| Provider                                                                        | Endpoint/Feature                                                                                          | AWS Backend                                                                                                                       |
|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/videos`](api_openai_videos.md) – create, poll, list, download, and delete video generation jobs      | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Amazon Nova Reel, Luma Ray 2 |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/moderations`](api_openai_moderations.md) – text and image content classification                     | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Guardrails, Amazon Comprehend |
| ![Cohere](styles/logo_cohere.svg){: style="height:20px;width:20px"} **Cohere**  | [`/cohere/v2/rerank`](api_cohere_rerank.md) – document reranking (Amazon Rerank 1.0, Cohere Rerank 3.5)    | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Rerank API                   |
| ![Cohere](styles/logo_cohere.svg){: style="height:20px;width:20px"} **Cohere**  | [`/cohere/v2/embed`](api_cohere_embed.md) – embeddings over all Bedrock embedding models                   | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - embedding models             |
| **stdapi.ai**                                                                    | [`/model_pricing`](api_model_pricing.md) – exact AWS unit prices per model                                 | AWS Price List API                                                                                                                  |

#### :material-brain: Extended Reasoning

| Provider                                                                        | Endpoint/Feature                                                                                                       | AWS Backend                                                                                                             |
|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/responses`](api_openai_responses.md#extended-reasoning) – Bedrock `reasoningContent` returned as reasoning output items | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | Streaming `response.output_item.added` / `response.reasoning_text.delta` / `.done` events for reasoning content       | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API       |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | `include=["reasoning.encrypted_content"]` – signature/redacted round-trip for multi-turn reasoning continuation       | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API       |

#### :material-chat: Conversations

| Provider                                                                        | Endpoint/Feature                                                                                                     | AWS Backend                                                                                                             |
|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | `store=true` + `GET/DELETE /v1/responses/{id}`, input items listing, and `previous_response_id` continuation          | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - session management |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | `POST /v1/responses/{id}/cancel` – endpoint parity for the cancel lifecycle (always fails for session-stored responses, which never run in background mode; Mantle-stored responses are cancelled upstream) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - session management |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | `store=true` + `GET/DELETE /v1/chat/completions/{id}`, `GET /v1/chat/completions` listing, `POST /v1/chat/completions/{id}` metadata updates, and input messages listing | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - session management |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/responses/compact`](api_openai_responses.md#conversation-compaction) – stateless conversation compaction        | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API       |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | `moderation` request parameter on chat completions and responses, with results reported in the response               | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Guardrails         |

#### Platform Features

| Feature                                       | Description                                                                                                                                                                        |
|-----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Multi-region AWS AI services                  | Automatic multi-region failover for Amazon Polly, Transcribe, Translate, and Comprehend (per-engine voice discovery, co-located Transcribe buckets, latency-ordered region pools) |
| Fault-tolerant, faster startup                | Unreachable Bedrock regions or Polly engines no longer abort startup; they are skipped with a warning and retried on the next refresh — and startup is faster overall             |
| Usage & cost tracking                         | Request logs report the usage actually billed by AWS with its cost computed from live AWS pricing, optionally exported as CloudWatch metrics ([`CLOUDWATCH_METRICS`](operations_logging_monitoring.md#cloudwatch-metrics-emf)); the previous token estimation is removed and its `TOKENS_ESTIMATION*` settings are deprecated and ignored |
| Video retention (`AWS_S3_VIDEOS_EXPIRES_AFTER`) | Optional retention period for generated videos, reported as `expires_at` and enforced on download                                                                               |
| Upload expiry (`expires_after`)               | Multipart upload sessions honor the OpenAI `expires_after` policy on the resulting file                                                                                           |
| Session storage encryption                    | Optional KMS key for AWS Bedrock session storage (`AWS_BEDROCK_SESSION_ENCRYPTION_KEY_ARN`)                                                                                       |
| Proxy trust (`PROXY_TRUSTED_HOSTS`)           | `X-Forwarded-*` headers are only honored when sent by a trusted reverse-proxy address                                                                                             |
| Input file size limit (`MAX_INPUT_FILE_SIZE`) | Optional cap on the size of downloaded/decoded input files, with bounded download concurrency (`MAX_CONCURRENT_INPUT_DOWNLOADS`)                                                  |
| Legacy model opt-in fix                       | `AWS_BEDROCK_LEGACY` now also exposes models whose AWS legacy date has already passed (e.g. Amazon Nova Reel)                                                                     |

#### :material-shield-lock: Security Hardening

- **MCP transports and `/search_models` now require authentication** when an API key is configured — clients that relied on these endpoints being open must now send the API key
- SSRF protection hardened against IP-literal encoding and DNS-rebinding bypasses on URL file inputs
- `s3://` file inputs are restricted to the server's allowed buckets, and multipart upload filenames are validated
- Decoded image size is capped against decompression-bomb payloads
- ARNs and AWS account IDs are redacted from client-facing error messages, and presigned URL signatures are stripped from logs and traces
- An empty resolved API key (e.g. a blank secret value) now disables authentication cleanly instead of matching an empty bearer token, and CORS no longer allows credentialed cross-origin requests

#### :material-robot-outline: Agent SDK Compatibility

The Responses API request/response surface was audited and hardened against the OpenAI SDK and real agent clients, end-to-end tested against the **OpenAI Codex CLI**:

- Hosted and agent tool types (`web_search`, `computer_use`, `file_search`, `custom`/`namespace` tools, and other items without a Bedrock equivalent) are now accepted and dropped instead of rejected with `400`, preserving compatibility with existing agent tooling
- Streaming responses now correctly terminate with `response.incomplete` or `response.failed` (matching upstream behavior) instead of always reporting `response.completed`
- Mid-stream errors emit the spec-compliant `error` SSE event
- `input_tokens` usage now includes cache read/write tokens, matching OpenAI's accounting
- `url_citation` annotations are emitted alongside their streaming events
- Echoed reasoning items tolerate the field variations produced by different SDKs and agent clients

#### :material-bug: Fixes

- Rerank models are no longer incorrectly advertised on Converse-based chat routes and MCP tools
- Fixed per-request model parameter overrides (`default_model_params`) occasionally leaking into subsequent requests for the same model
- 5xx provider errors now report server-side error types (`server_error`/`api_error`) in OpenAI and Anthropic error envelopes instead of `invalid_request_error`
- Newer Anthropic client request fields (free-form JSON Schema keywords in tool `input_schema`, adaptive thinking `display`) are accepted instead of rejected in strict validation mode
- Amazon Nova 2 no longer fails on `max_tokens` combined with high reasoning effort (the cap is dropped with a logged warning)
- Explicit cache points are kept off tool-related content blocks for models without tool caching support
- The Files API unavailable error no longer exposes the S3 bucket configuration detail
- Fixed input files from one request occasionally leaking into later requests served by the same connection, which could fail those requests with internal errors
- Anthropic Messages streams now emit an empty tool-input delta for tool calls without arguments, so SDK stream accumulators no longer fail on argument-less tool calls
- JSON-body image edit and variation requests now accept the `model` field instead of rejecting the request
- Model listings now report `service: "AWS Bedrock Runtime"` for classic Bedrock models (previously `"AWS Bedrock"`), distinguishing them from `"AWS Bedrock Mantle"`
- High reasoning effort now maps to the intended thinking-token budget on Anthropic Claude models (the budget factor was previously miscomputed)
- Setting `log_level` to `disabled` now suppresses all log output as documented, instead of publishing every event

---

### v1.13.0 – Terraform Module Compliance & Security Hardening

This release focuses on the [stdapi-ai Terraform module](https://github.com/stdapi-ai/terraform-aws-stdapi-ai) and its child modules — [VPC](https://github.com/JGoutin/terraform-aws-vpc), [KMS](https://github.com/JGoutin/terraform-aws-kms-key), and [ECS Fargate](https://github.com/JGoutin/terraform-aws-ecs-fargate) — adding detailed AWS Security Hub control documentation and closing several compliance gaps: default security group lockdown, ALB access logging, EFS POSIX user enforcement with native backups, and optional compliance/GuardDuty/DNS Firewall VPC integrations. All four modules now also accept a `tags` variable for custom resource tagging.

!!! info "Documentation-first release"
    Every module README now includes a full Security Hub Foundational Security Best Practices (FSBP) control mapping. See [Authentication & Security](operations_authentication_security.md#aws-security-hub-guardduty-dns-firewall-integration) for a summary and links to each module.

#### :material-bug: Fixes

- Added the missing `1h` and `5m` values to `PromptCacheRetention` for Bedrock-specific prompt cache TTLs in the OpenAI Responses API

#### :material-shield-star: Security Hub & Compliance Hardening

| Feature                                 | Module                           | Description                                                                                                                                                                                                      |
|-----------------------------------------|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Security Hub FSBP control documentation | VPC, KMS, ECS Fargate, stdapi-ai | Per-control (pass/fail/conditional/N-A) tables added to each module README                                                                                                                                       |
| Default security group lockdown         | VPC                              | New `aws_default_security_group` resource revokes all default ingress/egress rules (EC2.2 / CIS 5.4)                                                                                                             |
| VPC Flow Logs retention                 | VPC                              | Default retention increased from 7 to 365 days (EC2.6)                                                                                                                                                           |
| Compliance VPC endpoints                | VPC                              | New `compliance_vpc_endpoints_enabled` variable adds ECR, SSM, SSM Contacts, and SSM Incidents interface endpoints                                                                                               |
| GuardDuty VPC endpoint                  | VPC                              | New `guardduty_vpc_endpoint_enabled` variable adds the `guardduty-data` interface endpoint                                                                                                                       |
| Route 53 Resolver DNS Firewall          | VPC                              | New `dns_firewall_enabled` variable blocks/alerts on DNS queries to known-malicious domains (AWS Managed Domain Lists, plus DGA/DNS-tunneling detection via `dns_firewall_advanced_enabled`); dedicated VPC only |
| ALB access logging                      | stdapi-ai                        | New `alb_access_logging_enabled` variable (default `true`) logs ALB access to a dedicated, encrypted S3 bucket                                                                                                   |
| EFS POSIX user enforcement              | ECS Fargate                      | `mount_points` now accepts an `efs_posix_user` object to enforce a POSIX identity on EFS access points (EFS.4)                                                                                                   |
| EFS native backups                      | ECS Fargate                      | New `mount_points_efs_backup_enable` variable enables native EFS automatic backups, independent of the existing AWS Backup plan (EFS.7)                                                                          |
| Resource tagging                        | VPC, KMS, ECS Fargate, stdapi-ai | New `tags` variable propagates custom tags to nearly all created resources (IAM.24 / EC2.48)                                                                                                                     |

#### :material-cog-outline: Other Infrastructure Changes

| Feature                       | Description                                                                                                                                                                                                         |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AWS provider version bump     | Requirement raised to `>= 6.27.0` across all four modules                                                                                                                                                           |
| S3 object tag rename          | Files API objects and the corresponding Terraform lifecycle rule now use the `stdapi-ai.expires` tag key instead of `expires`; a temporary backward-compatible rule still expires legacy-tagged objects             |
| `aws-apn-id` resource tagging | AWS resources created at runtime (Bedrock async jobs, Transcribe jobs, S3 objects) are tagged with `aws-apn-id`, the standard AWS Marketplace attribution tag — an internal, vendor-side tag, not user-configurable |

#### :material-robot-outline: MCP Token Optimization

- Significantly reduced the size of MCP tool descriptions across the API, lowering the token cost of every AI agent session connected to this server
- No change in functionality: all parameter constraints and usage guidance remain intact

---

### v1.12.0 – Completions API, Video Understanding & File References

This release adds the OpenAI-compatible [`/v1/completions`](api_openai_completions.md) endpoint for text-first coding agents and legacy completion clients, **TwelveLabs Pegasus** video understanding for analyzing `video/*` inputs in chat completions, and an input token counting endpoint for the Responses API. Files uploaded through the Files API can now be referenced anywhere a URL is accepted using the new `file-id:` URI scheme. The Anthropic Messages API now accepts `system`-role messages (merged into the system prompt for compatibility), reasoning can be explicitly enabled or disabled, and a new `DEFAULT_MODEL_SERVICE_TIERS` setting applies per-model service tiers automatically.

#### :material-chat: Chat Completions

| Provider                                                                                     | Endpoint/Feature                                                                | AWS Backend                                                                                                             |
|----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**               | `/v1/completions` – text completion endpoint for text-first coding agents       | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models  |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**               | `/v1/responses/input_tokens` – input token counting                             | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - CountTokens API    |
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic**      | `/v1/messages` – accepts `system`-role messages (merged into the system prompt) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Claude models      |
| ![Twelve Labs](styles/logo_twelvelabs.svg){: style="height:20px;width:20px"} **Twelve Labs** | Pegasus video understanding (`video/*` inputs)                                  | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - TwelveLabs Pegasus |

#### :material-microphone: Speech & Audio

| Provider                                                                       | Endpoint/Feature                                                  | AWS Backend                                                                                  |
|--------------------------------------------------------------------------------|-------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/audio/speech` – case-insensitive voice names & default model | ![Amazon Polly](styles/logo_amazon_polly.svg){: style="height:20px;width:20px"} Amazon Polly |

#### Platform Features

| Feature                                                     | Description                                                                                                                                                                                                                               |
|-------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `file-id:` URI scheme                                       | Reference Files API uploads via `file-id:<file-id>` anywhere a URL is accepted — embeddings, audio transcription/translation, chat, images, and messages                                                                                  |
| Default model service tiers (`DEFAULT_MODEL_SERVICE_TIERS`) | Automatically apply a per-model service tier (`default`, `flex`, `priority`, `reserved`) when none is provided in the request                                                                                                             |
| Explicit reasoning enable/disable                           | Reasoning/thinking can now be explicitly enabled or disabled via request parameters                                                                                                                                                       |
| Service tier & guardrail support for Pegasus                | TwelveLabs Pegasus requests honor `service_tier` and Bedrock Guardrail configuration                                                                                                                                                      |
| MCP speech streaming defaults to SSE                        | `/v1/audio/speech` defaults `stream_format` to `sse` when invoked as an MCP tool for broader client compatibility                                                                                                                         |
| Full regional S3 bucket handling                            | The Terraform module resolves regional S3 buckets via resource-level region (requires AWS provider >= 6.0.0)                                                                                                                              |
| Reliable cross-region model identifiers                     | Region routing no longer fails intermittently with "The provided model identifier is invalid": a region whose inference profile is missing or not yet propagated is skipped, and a geo-scoped profile is never sent to a different region |

---

### v1.11.0 – MCP Server, Agent Discovery & Model Search (with v1.11.1-v1.11.4 maintenance updates)

This release introduces a **Model Context Protocol (MCP) server**, making all stdapi.ai API endpoints directly accessible as MCP tools for AI agents and agentic workflows. A new `/search_models` endpoint enables precise discovery of models by route, MCP tool, region, streaming support, and legacy status. Agent-friendly discovery metadata is now exposed via RFC 8288 Link headers and an RFC 9727 machine-readable API catalog at `/.well-known/api-catalog`. Endpoints that previously required binary `multipart/form-data` uploads now also accept an `application/json` body for MCP and HTTP client compatibility. The Anthropic Messages API now accepts `xhigh` as a `reasoning_effort` value.

#### :material-robot-outline: MCP Server

| Feature                            | Description                                                                                                                                        |
|------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| MCP server (Streamable HTTP & SSE) | All API endpoints exposed as MCP tools; Streamable HTTP and SSE transports can be independently enabled or disabled via configuration              |
| Configurable MCP tool exposure     | Individual MCP tools can be selectively enabled or restricted via configuration                                                                    |
| JSON body for binary endpoints     | Audio transcription, audio translation, and image edit endpoints now accept `application/json` with files as base64, data URI, HTTP URL, or S3 URI |

#### :material-magnify: Model Search

| Feature          | Description                                                                                                                                                                                                                                                                                      |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `/search_models` | New official endpoint to filter models by route, MCP tool name, input/output modalities, region, streaming, and legacy status; returns richer metadata than `/v1/models` or Anthropic `/v1/models`, designed for LLM-driven model selection (replaces BETA and undocumented `/available_models`) |

#### :material-access-point: Agent Discovery

| Feature                                               | Description                                                                                   |
|-------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| RFC 8288 Link headers                                 | Root (`/`) endpoint returns Link headers for resource discovery                               |
| RFC 9727 API catalog (`/.well-known/api-catalog`)     | Machine-readable API catalog for automated agent and tool discovery                           |
| MCP Server Card (`/.well-known/mcp/server-card.json`) | Advertises available MCP transports and capabilities to AI agents (SEP-1649)                  |
| `robots.txt` AI signals                               | Updated `robots.txt` with `Content-Signal` directives and explicit `/.well-known/` allow rule |

#### :material-chat: Chat Completions & Messages

| Provider                                                                                | Endpoint/Feature                                           | AWS Backend                                                                                                            |
|-----------------------------------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic** | `/v1/messages` `reasoning_effort=xhigh` support            | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Claude models     |

#### :material-label-off: Deprecation Mappings

- Added automatic fallback for `amazon.nova-reel-v1:0` and `anthropic.claude-3-haiku-20240307-v1:0` to their respective replacements

#### Fixes

- Fix reasoning token double-counting in usage calculation in OpenAI Responses API adapter
- Fix missing `file_id` inputs for image and file processing in OpenAI Responses API adapter
- Remove `store` parameter from unsupported validations in chat completions to ensure client compatibility

#### Fixes & Maintenance (v1.11.1-v1.11.4)

**v1.11.1**

- Make `max_tokens` optional in Anthropic `/v1/messages` to align with the Anthropic API specification
- Remove unsupported reasoning configuration checks for broader client compatibility
- Rename `/v1/responses` route tag from "Responses" to "Chat" in OpenAPI documentation for consistency

**v1.11.2-v1.11.3**

- Add missing MCP dependencies to container image.

**v1.11.4**

- Upgrade Starlette dependency to fix CVE-2026-48710.

---

### v1.10.0 – OpenAI Responses API

This release adds support for the OpenAI [`/v1/responses`](api_openai_responses.md) endpoint—OpenAI's next-generation API designed for building agents and multi-step AI workflows. Drop-in compatible with the OpenAI SDK, it works with all AWS Bedrock Converse-compatible models and supports streaming, function tools, built-in tools (web search, code interpreter, image generation), extended reasoning, and structured output.

#### :material-chat: Responses (OpenAI-Compatible)

| Provider                                                                       | Endpoint/Feature                                                    | AWS Backend                                                                                                            |
|--------------------------------------------------------------------------------|---------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/responses`                                                     | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/responses` – `web_search` / `web_search_preview` built-in tool | ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} Amazon Nova models                       |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/responses` – `code_interpreter` built-in tool                  | ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} Amazon Nova models                       |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/responses` – `image_generation` built-in tool                  | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models      |

#### Fixes

- Fix prompt caching error when messages contain tool-related content on models that do not support tool caching
- Make `signature` field optional in Anthropic message types
- Fix model legacy detection when the end-of-life date falls before the next cache refresh

---

### v1.9.0 – Files API & Images API JSON Body

This release introduces a Files API backed by Amazon S3, available through both the OpenAI-compatible and Anthropic-compatible interfaces. Files uploaded via either API share the same S3 storage and can be referenced across both interfaces. Large files can be uploaded incrementally using the OpenAI multipart uploads API. Stored files can be referenced by ID directly in image edit and variation requests (JSON body), as well as in chat completion messages as document or image inputs. The image edits endpoint now also accepts an `application/json` body as an alternative to multipart form-data, making it easier to chain pipeline steps without re-uploading files.

!!! warning "New Required Configuration"
    Files API requires `AWS_S3_BUCKET` to be configured (shared with the image URL response feature). The S3 prefix for stored files defaults to `files/` and is configurable via `AWS_S3_FILES_PREFIX`. Ensure your IAM role includes read, write, delete, and list permissions on the files prefix in addition to the existing S3 permissions for presigned URLs.

#### :material-folder: Files & Storage

| Provider                                                                                | Endpoint/Feature                      | AWS Backend                                                                         |
|-----------------------------------------------------------------------------------------|---------------------------------------|-------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**          | `/v1/files` – CRUD operations         | ![Amazon S3](styles/logo_amazon_s3.svg){: style="height:20px;width:20px"} Amazon S3 |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**          | `/v1/uploads` – multipart uploads     | ![Amazon S3](styles/logo_amazon_s3.svg){: style="height:20px;width:20px"} Amazon S3 |
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic** | `/v1/files` – CRUD operations         | ![Amazon S3](styles/logo_amazon_s3.svg){: style="height:20px;width:20px"} Amazon S3 |

#### :material-image: Image Generation

| Provider                                                                       | Endpoint/Feature                                                                      | AWS Backend                                                                                                       |
|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/images/edits` – JSON body with `images`/`mask` referencing Files API IDs or URLs | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/images/variations` – JSON body with `image` referencing a Files API ID or URL    | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |

#### :material-chat: Chat Completions & Messages

| Provider                                                                                | Endpoint/Feature                                                               | AWS Backend                                                                                                            |
|-----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**          | Files API file IDs usable as document/image inputs in chat completions         | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic** | Files API file IDs usable as document/image inputs in messages                 | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |

#### Fixes

- Document inputs via S3 URLs are not supported as Bedrock Converse API inputs for some models (e.g., Claude) — now properly detected and handled

---

### v1.8.0 – Broader Model Compatibility & Structured Output

This release focuses on improving reliability and compatibility across a wide variety of models. Structured response formats (JSON object and JSON schema) are now supported on OpenAI chat completions, and request metadata can be forwarded to Bedrock. Tool handling has been significantly improved—both for model-specific system tools and for Amazon Nova's grounding tool, including multi-turn support. Region routing is now more robust, correctly enforcing non-global inference profiles for region-restricted models and handling edge cases gracefully.

!!! warning "New Required IAM Permissions"
    v1.8.0 requires two new IAM permissions to attach request metadata tags to jobs:

    - **`bedrock:TagResource`** on `arn:aws:bedrock:*:*:async-invoke/*` — needed for Bedrock asynchronous invocation jobs (see [IAM Permissions](operations_configuration.md#bedrock-iam)). The `twelvelabs.marengo-embed-3-0-v1:0` and `twelvelabs.marengo-embed-2-7-v1:0` models rely on asynchronous invocation and will fail with an access denied error if this permission is missing.
    - **`transcribe:TagResource`** on `arn:aws:transcribe:*:*:transcription-job/*` — needed for Amazon Transcribe transcription jobs (see [IAM Permissions](operations_configuration.md#speech-to-text-optional)). The `amazon.transcribe` model will fail with an access denied error if this permission is missing.

    Ensure your IAM role or user policy includes both statements before upgrading to v1.8.0.

#### :material-chat: Chat Completions

| Provider                                                                                      | Endpoint/Feature                                                  | AWS Backend                                                                                                            |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**                | `response_format` – JSON object and JSON schema structured output | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**                | `metadata` – request metadata forwarding to Bedrock               | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} **Amazon Nova** | Nova Code Interpreter global profile support                      | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Nova models       |

#### :material-message: Messages (Anthropic-Compatible)

| Provider                                                                                      | Endpoint/Feature                                                              | AWS Backend                                                                                                         |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} **Amazon Nova** | `nova_grounding` responses mapped to `web_search` content blocks              | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Nova models    |
| ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} **Amazon Nova** | Multi-turn conversation support with `nova_grounding`                         | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Nova models    |

#### Platform Features

| Feature                                          | Description                                                                                                                                                                                              |
|--------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Non-global profiles for region-restricted models | Region-restricted models are now always assigned non-global inference profiles, preventing requests from bypassing configured region restrictions                                                        |
| Region routing edge case handling                | Region routing gracefully handles cases where no usable regions are available                                                                                                                            |
| ECS-based server ID                              | When running on ECS, `server_id` in logs is set to `task_id.container_name` for precise instance identification across tasks and containers                                                              |
| Request metadata tagging                         | stdapi.ai request context (`request_id`, `server_id`, `user_id`) is automatically attached as tags to every Bedrock and Amazon Transcribe job, making it easy to trace API calls across AWS service logs |

#### Fixes

- Fix `systemTool_` prefix handling: removed broken auto-promotion logic; system tools require specific tool output handling not compatible with generic tool forwarding
- `AWS_BEDROCK_LEGACY` default changed from `true` to `false` to prevent access denied errors on legacy models that have not been actively used recently
- Bedrock read timeouts are now handled as standard model errors (503) instead of unhandled exceptions, and are properly retried across regions when multi-region routing is enabled

---

### v1.7.0 – Automatic Region Routing, Deprecated Model Fallback & Resilience Improvements

The headline feature of v1.7 is **automatic multi-region routing**: stdapi.ai now intelligently distributes requests across your configured AWS regions, failing over automatically on quota limits or unavailability—and because each region carries its own independent quota, adding regions directly multiplies your effective tokens-per-minute and daily limits. Alongside this, deprecated model IDs are transparently redirected to their replacements so clients survive AWS model retirements without any code changes. This release also adds S3 URL support for file inputs across all relevant endpoints, a configurable AI response timeout, and memory efficiency improvements.

#### Platform Features

| Feature                                               | Description                                                                                                                                                                                                            |
|-------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Automatic region routing with configurable strategies | Intelligently distributes Bedrock requests across configured AWS regions with automatic failover on quota limits or unavailability; supports `ordered`, `lowest_latency`, and `round_robin` strategies                 |
| Deprecated model fallback                             | Transparently reroute deprecated model IDs to their replacements; extend or override the built-in mapping; warns on legacy model usage                                                                                 |
| AI response timeout                                   | Configurable timeout for AI model responses to prevent indefinitely hanging requests                                                                                                                                   |
| Expanded file input support                           | File inputs (images, documents, audio) now support S3 URLs in addition to HTTP URLs, data URIs, and plain base64 across all relevant endpoints; improves memory efficiency by releasing file data as early as possible |
| Model lifecycle timestamps                            | Model created/updated timestamps now derived from lifecycle data (`startOfLifeTime`, `endOfLifeTime`)                                                                                                                  |

#### Fixes

- Fix SSE stream error handling in monitoring to handle specific API and AWS client errors gracefully
- Fix audio MIME type detection failure when `libmagic`'s in-memory buffer path silently returns `application/octet-stream`; fall back to file-based detection to ensure correct format is sent to Bedrock

---

### v1.6.0 – Anthropic API Compatibility & Advanced Claude Capabilities

Introduces a full Anthropic-compatible API layer, enabling direct use of the Anthropic SDK and Claude-native tools with AWS Bedrock. Adds Claude server tools support via OpenAI chat completions, token count estimation, automatic Anthropic beta flag filtering, and configurable route prefixes.

#### :material-chat: Chat Completions

| Provider                                                                       | Endpoint/Feature                                                                                                                     | AWS Backend                                                                                                   |
|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` Claude server tools (`bash`, `str_replace_based_edit_tool`, `computer`, `memory`) | ![Claude](styles/logo_anthropic_claude.svg){: style="height:20px;width:20px"} Claude models on Amazon Bedrock |

#### :material-message: Messages (Anthropic-Compatible)

| Provider                                                                                      | Endpoint/Feature                                          | AWS Backend                                                                                                          |
|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic**       | `/v1/messages` – Full Anthropic Messages API              | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API    |
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic**       | `/v1/messages/count_tokens` – Token counting              | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - CountTokens API |
| ![Claude](styles/logo_anthropic_claude.svg){: style="height:20px;width:20px"} **Claude**      | Claude server tools (bash, text editor, computer, memory) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Claude models   |
| ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} **Amazon Nova** | Web search tool (`web_search` → `nova_grounding`)         | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Nova models     |

#### :material-format-list-bulleted: Model Discovery (Anthropic-Compatible)

| Provider                                                                                | Endpoint/Feature                              | AWS Backend                                                                                                        |
|-----------------------------------------------------------------------------------------|-----------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic** | `/v1/models` – List models (Anthropic format) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - model catalog |
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic** | `/v1/models/{model_id}` – Get model details   | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - model catalog |

#### Platform Features

| Feature                                                 | Description                                                                                                                                        |
|---------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `ANTHROPIC_ROUTES_PREFIX` configuration                 | Configurable base path prefix for Anthropic-compatible routes (default: `/anthropic`)                                                              |
| `OPENAI_ROUTES_PREFIX` configuration                    | Configurable base path prefix for OpenAI-compatible routes                                                                                         |
| Real usage tracking (`usage` in logs)                   | Token counts sourced directly from AWS billing data (replaces tiktoken-based estimation)                                                           |
| Anthropic beta flag filtering (`ANTHROPIC_BETA_FILTER`) | Automatically filter unsupported `anthropic-beta` flags to prevent Bedrock `ValidationException` errors; extensible via `ANTHROPIC_BETA_ALLOWLIST` |
| Claude model name aliases                               | Use official Anthropic model names (e.g., `claude-opus-4-8`) auto-resolved to AWS Bedrock identifiers                                              |

---

### v1.5.0 – Advanced Reasoning & Model Compatibility (with v1.5.1–v1.5.2 maintenance updates)

Introduces advanced reasoning capabilities with Amazon Nova 2 and Anthropic Claude 4.6+ adaptive reasoning, enhanced system prompt handling for broader model compatibility.

#### :material-chat: Chat Completions

| Provider                                                                                      | Endpoint/Feature                              | AWS Backend                                                                                                            |
|-----------------------------------------------------------------------------------------------|-----------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**                | System prompt handling for unsupported models | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} **Amazon Nova** | Nova 2 chat model reasoning implementation    | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Claude](styles/logo_anthropic_claude.svg){: style="height:20px;width:20px"} **Claude**      | Claude 4.6+ adaptive reasoning configuration  | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Claude models     |

#### Fixes & Maintenance (v1.5.1–v1.5.2)

**v1.5.2**

- Add "/" route to avoid 404 errors on root endpoint
- Fix empty system content block handling (improves AWS Bedrock Converse API compatibility)

**v1.5.1**

- Fix Amazon Nova Canvas image editing to fall back to TEXT_IMAGE task type when no mask is provided

---

### v1.4.0 – Audio Enhancements & Model Compatibility

Expands audio capabilities with Mistral Voxtral support, speaker diarization, audio formats for chat completions, and introduces prompt caching TTL and model aliasing for better OpenAI compatibility.

#### :material-chat: Chat Completions

| Provider                                                                       | Endpoint/Feature                                                     | AWS Backend                                                                                                            |
|--------------------------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` audio format support                          | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` extended Bedrock finish reasons mapping       | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock                     |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | Prompt caching TTL support                                           | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - prompt caching    |

#### :material-microphone: Speech & Audio

| Provider                                                                            | Endpoint/Feature                                  | AWS Backend                                                                                                            |
|-------------------------------------------------------------------------------------|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**      | `/v1/audio/transcriptions` `diarized_json` format | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe            |
| ![Mistral](styles/logo_mistralai.svg){: style="height:20px;width:20px"} **Mistral** | Voxtral audio model                               | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |

#### Platform Features

| Feature                          | Description                                                  |
|----------------------------------|--------------------------------------------------------------|
| Model alias support              | Seamless OpenAI compatibility via model name aliasing        |

#### Fixes

- Fix chat completion file input handling and refactor base64 decoding and MIME handling for file processing.
- Re-raise startup exceptions and disable botocore logging to improve error visibility

---

### v1.3.0 – Image Editing & Variation Support (with v1.3.1–v1.3.5 maintenance updates)

Adds support for OpenAI's image editing and variation endpoints, enabling image manipulation capabilities backed by Amazon Bedrock. Includes maintenance updates for content block handling, tool call validation, streaming fixes, and TTS optimization.

#### :material-image: Image Generation

| Provider                                                                       | Endpoint/Feature        | AWS Backend                                                                                                       |
|--------------------------------------------------------------------------------|-------------------------|-------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/images/edits`      | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/images/variations` | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |

#### :material-microphone: Speech & Audio (v1.3.2)

| Feature                        | Description                                                   |
|--------------------------------|---------------------------------------------------------------|
| `DEFAULT_TTS_LANGUAGE` setting | Configurable default language for TTS to optimize performance |

#### Fixes & Maintenance (v1.3.1–v1.3.5)

**v1.3.5**

- Refactor content block handling to skip empty entries in assistant responses

**v1.3.4**

- Handle invalid tool call arguments with robust JSON content validation
- Add deprecation mapping for `amazon.titan-image-generator-v2:0` → `amazon.nova-canvas-v1:0`

**v1.3.3**

- Remove premature stop condition for `contentBlockStop` in streaming chat completions

**v1.3.2**

- Support `image[]` array-style notation for OpenAI image edits
- Handle empty audio segments in transcription duration calculation

**v1.3.1**

- Improve JSON parsing for tool arguments and results
- Correct `example` → `examples` in OpenAPI model path parameter

---

### v1.2.0 – Service Tiers, System Tools & Performance Enhancements

Introduces service tiers and latency headers for all Bedrock routes, Bedrock-specific system tools (Nova grounding), GPT5.2 API compatibility, configurable guardrail overrides, and Python 3.14 optimization.

#### :material-chat: Chat Completions

| Provider                                                                       | Endpoint/Feature                                                      | AWS Backend                                                                                                        |
|--------------------------------------------------------------------------------|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` `service_tier` parameter                       | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - service tiers |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` Bedrock-specific system tools (Nova grounding) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - system tools  |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` GPT5.2 API update (`reasoning_effort=xhigh`)   |                                                                                                                    |

#### :material-shield-check: Content Safety & Moderation

| Feature                                         | AWS Backend                                                                                                   |
|-------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Configuration flag for guardrail override allow | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock Guardrails |

#### Platform Features

| Feature                                                | AWS Backend / Description                                                                                          |
|--------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| Service tiers and latency headers (all Bedrock routes) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - service tiers |
| Python 3.14 support                                    | Upgraded to Python 3.14 with performance optimization                                                              |
| Dependency update                                      | Direct aiobotocore usage (replaced aioboto3)                                                                       |

#### Fixes

- Fix warnings for duplicated FastAPI routes (`/docs` and `/openapi.json`).

---

### v1.1.0 – Embeddings Enhancement, Prompt Caching & Advanced Routing

Expands multimodal embedding capabilities, adds prompt caching support, and introduces advanced routing with application inference profiles and prompt routers.

#### :material-chat: Chat Completions

| Provider                                                                       | Endpoint/Feature                                                    | AWS Backend                                                                                                         |
|--------------------------------------------------------------------------------|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | Prompt caching `/v1/chat/completions` `prompt_cache_key`            | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - prompt caching |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` GPT5.1 API update  (`reasoning_effort=none`) |                                                                                                                     |

#### :material-vector-polyline: Embeddings

| Provider                                                                                      | Endpoint/Feature                          | AWS Backend                                                                                        |
|-----------------------------------------------------------------------------------------------|-------------------------------------------|----------------------------------------------------------------------------------------------------|
|                                                                                               | Intelligent S3 multimodal upload          | ![Amazon S3](styles/logo_amazon_s3.svg){: style="height:20px;width:20px"} Amazon S3                |
|                                                                                               | Intelligent Sync/async Bedrock invocation | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock |
| ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} **Amazon Nova** | Multimodal embeddings models              |                                                                                                    |
| ![Twelve Labs](styles/logo_twelvelabs.svg){: style="height:20px;width:20px"} **Twelve Labs**  | Marengo V3 models                         |                                                                                                    |

#### :material-directions-fork: Advanced Routing

| Feature                            | AWS Backend                                                                                                                         |
|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Application inference profiles     | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - application inference profiles |
| Prompt routers                     | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - prompt routers                 |
| Server-side ARN mapping            | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock                                  |
| Client-side ARN passing (optional) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock                                  |

#### Fixes

- `/v1/chat/completions`: Fix default value passed to the converse API for tools without parameters.
- [stdapi-ai Terraform module](https://github.com/stdapi-ai/terraform-aws-stdapi-ai): Fix error if alarms_enabled = true but sns_topic_arn undefined.

---

### v1.0.0 – Foundation Release

The initial release establishes core OpenAI API compatibility with AWS Bedrock backing.

#### :material-chat: Chat Completions

| Provider                                                                             | Endpoint/Feature                                   | AWS Backend                                                                                                            |
|--------------------------------------------------------------------------------------|----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**       | `/v1/chat/completions`                             | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
|                                                                                      | All models supporting Converse/ConverseStream APIs | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API      |
| ![Deepseek](styles/logo_deepSeek.svg){: style="height:20px;width:20px"} **Deepseek** | `/v1/chat/completions` `reasoning_content`         | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Qwen](styles/logo_qwen.svg){: style="height:20px;width:20px"} **Qwen**             | `enable_thinking` + `thinking_budget` parameter    | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Qwen](styles/logo_qwen.svg){: style="height:20px;width:20px"} **Qwen**             | `top_k` parameter                                  | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |

#### :material-vector-polyline: Embeddings

| Provider                                                                                     | Endpoint/Feature      | AWS Backend                                                                                                           |
|----------------------------------------------------------------------------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**               | `/v1/embeddings`      | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - embedding models |
| ![Cohere](styles/logo_cohere.svg){: style="height:20px;width:20px"} **Cohere**               | Embed V3 & V4  models |                                                                                                                       |
| ![Twelve Labs](styles/logo_twelvelabs.svg){: style="height:20px;width:20px"} **Twelve Labs** | Marengo V2  models    |                                                                                                                       |
| ![Amazon](styles/logo_amazon.svg){: style="height:20px;width:20px"} **Amazon Titan**         | Embed V1 & V2  models |                                                                                                                       |

#### :material-microphone: Speech & Audio

| Provider                                                                       | Endpoint/Feature           | AWS Backend                                                                                                                    |
|--------------------------------------------------------------------------------|----------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/audio/speech`         | ![Amazon Polly](styles/logo_amazon_polly.svg){: style="height:20px;width:20px"} Amazon Polly + Amazon Comprehend               |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/audio/transcriptions` | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe                    |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/audio/translations`   | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe + Amazon Translate |

#### :material-image: Image Generation

| Provider                                                                                        | Endpoint/Feature                        | AWS Backend                                                                                                       |
|-------------------------------------------------------------------------------------------------|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**                  | `/v1/images/generations`                | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} **Amazon Nova**   | Canvas V1 models                        |                                                                                                                   |
| ![Amazon](styles/logo_amazon.svg){: style="height:20px;width:20px"} **Amazon Titan**            | Image Generator V1 & V2  models         |                                                                                                                   |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | Image Core, Ultra et SD3.5 Large models |                                                                                                                   |

#### :material-format-list-bulleted: Model Discovery

| Provider                                                                       | Endpoint/Feature | AWS Backend                                                                                                        |
|--------------------------------------------------------------------------------|------------------|--------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/models`     | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - model catalog |

#### Platform Features

| Feature                                     | AWS Backend                                                                                                                                                                                                                                 |
|---------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Bedrock Features**                        |                                                                                                                                                                                                                                             |
| Content filtering and safety                | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock Guardrails                                                                                                                               |
| Cross-region inference                      | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - global/regional                                                                                                                        |
| Application inference profiles              | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - inference profiles                                                                                                                     |
| Model parameters (temperature, top_p, etc.) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - native parameters                                                                                                                      |
| Multi-region failover                       | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - multi-region                                                                                                                           |
| Bedrock guardrails                          | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock Guardrails                                                                                                                               |
| **AWS Services**                            |                                                                                                                                                                                                                                             |
| File storage                                | ![Amazon S3](styles/logo_amazon_s3.svg){: style="height:20px;width:20px"} Amazon S3 - presigned URLs, Transfer Acceleration                                                                                                                 |
| **Authentication**                          |                                                                                                                                                                                                                                             |
| Static token authentication                 | ![AWS Systems Manager](styles/logo_amazon_systems_manager.svg){: style="height:20px;width:20px"} AWS SSM Parameter Store / ![AWS Secrets Manager](styles/logo_amazon_secrets_manager.svg){: style="height:20px;width:20px"} Secrets Manager |
| Development mode (no auth)                  |                                                                                                                                                                                                                                             |
| **Observability**                           |                                                                                                                                                                                                                                             |
| Distributed tracing                         | ![AWS X-Ray](styles/logo_amazon_xray.svg){: style="height:20px;width:20px"} AWS X-Ray + OpenTelemetry                                                                                                                                       |
| Structured logging                          | ![Amazon CloudWatch](styles/logo_amazon_cloudwatch.svg){: style="height:20px;width:20px"} Amazon CloudWatch (When running on ECS/EKS)                                                                                                       |
| Health check endpoint                       |                                                                                                                                                                                                                                             |
| **HTTP/Security**                           |                                                                                                                                                                                                                                             |
| CORS support                                |                                                                                                                                                                                                                                             |
| Trusted host validation                     |                                                                                                                                                                                                                                             |
| Proxy headers (X-Forwarded-*)               |                                                                                                                                                                                                                                             |
| GZip compression                            |                                                                                                                                                                                                                                             |
| **📚 Documentation**                        |                                                                                                                                                                                                                                             |
| Interactive API docs & OpenAPI schema       |                                                                                                                                                                                                                                             |
| **🔌 Compatibility**                        |                                                                                                                                                                                                                                             |
| Provider-specific parameters                |                                                                                                                                                                                                                                             |
