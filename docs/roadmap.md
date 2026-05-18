---
title: Releases & Roadmap - Active Development
description: stdapi.ai release history and upcoming features. Track regular updates, new AWS Bedrock capabilities, and active development progress.
keywords: stdapi.ai releases, AI gateway updates, AWS Bedrock features, API gateway roadmap, software changelog, active development, new AI features, product updates
---

# :material-timeline: Releases & Roadmap

**stdapi.ai is under active development** with regular feature releases.

## :material-tag-multiple: Recent Releases

See [Release History below](#release-history) for the full changelog of all releases.

**Latest: v1.11.1** – MCP server, agent discovery, `/search_models` endpoint, `xhigh` reasoning effort support, and optional `max_tokens` in Anthropic Messages API

---

## :material-rocket-launch: Planned Features

The following features may be implemented in future releases based on community demand and feedback. Implementation priority is determined by user requests and use case requirements.

**Want a feature?** Submit feedback on [GitHub Issues](https://github.com/stdapi-ai/stdapi.ai/issues).

### :material-chat: Chat Completions

| Provider                                                                       | Endpoint/Feature                                | AWS Backend                                                                                                            |
|--------------------------------------------------------------------------------|-------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/completions`                               | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/responses` – stateful conversations        | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` – stateful conversations | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Ollama](styles/logo_ollama.svg){: style="height:20px;width:20px"} **Ollama** | `/api/generate`                                 | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Ollama](styles/logo_ollama.svg){: style="height:20px;width:20px"} **Ollama** | `/api/chat`                                     | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Cohere](styles/logo_cohere.svg){: style="height:20px;width:20px"} **Cohere** | `/v1/chat`                                      | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Qwen](styles/logo_qwen.svg){: style="height:20px;width:20px"} **Qwen**       | `/v1/chat/completions` `translation_options`    | ![Amazon Translate](styles/logo_amazon_translate.svg){: style="height:20px;width:20px"} Amazon Translate               |

### :material-vector-polyline: Embeddings

| Provider                                                                       | Endpoint/Feature  | AWS Backend                                                                                                           |
|--------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------------------------------------------------------------------|
| ![Ollama](styles/logo_ollama.svg){: style="height:20px;width:20px"} **Ollama** | `/api/embeddings` | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - embedding models |
| ![Cohere](styles/logo_cohere.svg){: style="height:20px;width:20px"} **Cohere** | `/v1/embed`       | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - embedding models |

### :material-magnify: Semantic Search & Ranking

| Provider                                                                       | Endpoint/Feature | AWS Backend                                                                                                        |
|--------------------------------------------------------------------------------|------------------|--------------------------------------------------------------------------------------------------------------------|
| ![Cohere](styles/logo_cohere.svg){: style="height:20px;width:20px"} **Cohere** | `/v1/rerank`     | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - rerank models |

### :material-format-list-bulleted: Model Discovery

| Provider                                                                                        | Endpoint/Feature                                                   | AWS Backend                                                                                                        |
|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| ![Ollama](styles/logo_ollama.svg){: style="height:20px;width:20px"} **Ollama**                  | `/api/tags`                                                        | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - model listing |
| ![Ollama](styles/logo_ollama.svg){: style="height:20px;width:20px"} **Ollama**                  | `/api/show`                                                        | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - model details |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v1/engines/list`                                                 | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - model catalog |
|                                                                                                 | Model selection wildcards (To automatically latest model versions) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - model catalog |

### :material-microphone: Speech & Audio

| Provider                                                                       | Endpoint/Feature                      | AWS Backend                                                                                                                |
|--------------------------------------------------------------------------------|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/realtime/sessions`               | ![Amazon Nova Sonic](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} Amazon Nova Sonic                      |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/realtime/transcription_sessions` | ![Amazon Nova Sonic](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} Amazon Nova Sonic                      |
|                                                                                | Transcriptions with Nova Sonic        | ![Amazon Nova Sonic](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} Amazon Nova Sonic                      |
|                                                                                | Translations with Nova Sonic          | ![Amazon Nova Sonic](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} Amazon Nova Sonic                      |
|                                                                                | Long-form speech (async)              | ![Amazon Polly](styles/logo_amazon_polly.svg){: style="height:20px;width:20px"} Amazon Polly - async tasks                 |
|                                                                                | Streaming transcription               | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe - streaming    |
|                                                                                | Custom vocabularies                   | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe - custom vocab |

### :material-image: Image Generation

| Provider                                                                                        | Endpoint/Feature                        | AWS Backend                                                                                                       |
|-------------------------------------------------------------------------------------------------|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v1/generation/text-to-image`          | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v1/generation/image-to-image`         | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v1/generation/image-to-image/masking` | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v2/stable-image/generate`             | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v2/stable-image/upscale`              | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v2/stable-image/edit`                 | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |

### :material-tune: Model-Specific Features

| Provider | Endpoint/Feature               | AWS Backend                                                                                                             |
|----------|--------------------------------|-------------------------------------------------------------------------------------------------------------------------|
|          | Running Provisioned throughput | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - provisioned models |

### :material-robot: AWS Bedrock Advanced Features

| Provider                                                                       | Endpoint/Feature       | AWS Backend                                                                                                           |
|--------------------------------------------------------------------------------|------------------------|-----------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/agents`           | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Agents           |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/vector_stores`    | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Knowledge Bases  |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/evals`            | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Model Evaluation |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/fine_tuning/jobs` | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - custom models    |

### :material-package-variant: Batch & Async Processing

| Provider                                                                                | Endpoint/Feature                              | AWS Backend                                                                                                          |
|-----------------------------------------------------------------------------------------|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**          | `/v1/batches`                                 | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - batch inference |
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic** | `/v1/messages/batches`                        | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - batch inference |

### :material-shield-check: Content Safety & Moderation

| Provider                                                                       | Endpoint/Feature  | AWS Backend                                                                                                            |
|--------------------------------------------------------------------------------|-------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/moderations` | ![Amazon Comprehend](styles/logo_amazon_comprehend.svg){: style="height:20px;width:20px"} Amazon Comprehend - toxicity |

### :material-chart-bar: Usage & Analytics

| Provider                                                                       | Endpoint/Feature         | AWS Backend                                                                                                 |
|--------------------------------------------------------------------------------|--------------------------|-------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/usage`              | ![Amazon CloudWatch](styles/logo_amazon_cloudwatch.svg){: style="height:20px;width:20px"} Amazon CloudWatch |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/organization/usage` | ![Amazon CloudWatch](styles/logo_amazon_cloudwatch.svg){: style="height:20px;width:20px"} Amazon CloudWatch |

### :material-key: Authentication & Access Control

| Provider | Endpoint/Feature      | AWS Backend                                                                                                          |
|----------|-----------------------|----------------------------------------------------------------------------------------------------------------------|
|          | User authentication   | ![Amazon Cognito](styles/logo_amazon_cognito.svg){: style="height:20px;width:20px"} Amazon Cognito                   |
|          | Multi-tenant API keys | ![Amazon DynamoDB](styles/logo_amazon_dynamodb.svg){: style="height:20px;width:20px"} Amazon DynamoDB                |
|          | API key rotation      | ![AWS Secrets Manager](styles/logo_amazon_secrets_manager.svg){: style="height:20px;width:20px"} AWS Secrets Manager |
|          | Rate limiting         | ![Amazon DynamoDB](styles/logo_amazon_dynamodb.svg){: style="height:20px;width:20px"} Amazon DynamoDB                |
|          | AWS Bedrock API keys  | ![AWS Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} AWS Bedrock                         |

---

## :material-history: Release History

### v1.11.0 – MCP Server, Agent Discovery & Model Search (with v1.11.1-v1.11.2 maintenance updates)

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

#### Fixes & Maintenance (v1.11.1)

- Make `max_tokens` optional in Anthropic `/v1/messages` to align with the Anthropic API specification
- Remove unsupported reasoning configuration checks for broader client compatibility
- Rename `/v1/responses` route tag from "Responses" to "Chat" in OpenAPI documentation for consistency

#### Fixes & Maintenance (v1.11.2)

- Add missing MCP dependencies to container image.

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
| Token count estimation (`TOKENS_ESTIMATION`)            | Estimate token counts via tiktoken when models don't provide them; configurable encoding via `TOKENS_ESTIMATION_DEFAULT_ENCODING`                  |
| Anthropic beta flag filtering (`ANTHROPIC_BETA_FILTER`) | Automatically filter unsupported `anthropic-beta` flags to prevent Bedrock `ValidationException` errors; extensible via `ANTHROPIC_BETA_ALLOWLIST` |
| Claude model name aliases                               | Use official Anthropic model names (e.g., `claude-opus-4-6`) auto-resolved to AWS Bedrock identifiers                                              |

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
