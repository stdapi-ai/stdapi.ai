---
title: API Overview - OpenAI, Anthropic & Cohere Compatible Amazon Bedrock API
description: Complete API documentation for the stdapi.ai OpenAI-, Anthropic-, and Cohere-compatible gateway. Access Amazon Bedrock models, chat completions, messages, embeddings, image generation, and audio APIs with SDK compatibility.
keywords: OpenAI API documentation, Anthropic API documentation, AWS Bedrock API reference, OpenAI SDK compatibility, Anthropic SDK compatibility, chat completions API, responses API, messages API, embeddings API, image generation API, audio API AWS, OpenAI compatible endpoints, Anthropic compatible endpoints
---

# :material-api: API Overview

stdapi.ai provides OpenAI-, Anthropic-, and Cohere-compatible APIs backed by Amazon Bedrock and AWS AI services. Any application that works with OpenAI, Anthropic, or Cohere works with stdapi.ai by simply changing the API endpoint.

!!! tip "One catalog, discovered automatically"
    Amazon Bedrock, Bedrock Mantle, Amazon Polly, Amazon Transcribe and Amazon Comprehend all surface as **models in a single catalog**. stdapi.ai discovers them from your AWS account at startup — there is no model list to declare or maintain, and a model AWS adds appears without a configuration change. They are interchangeable by name on a shared endpoint: [`GET /v1/models`](api_openai_models.md) lists them together, [`GET /search_models`](api_search_models.md) filters them by capability, and the endpoint routes to whichever AWS service backs the model you named — `POST /v1/audio/transcriptions` reaches Amazon Transcribe or a Bedrock audio model, and `POST /v1/moderations` reaches Bedrock Guardrails or Amazon Comprehend, from the same request, and the [Models](models.md) page shows the whole catalogue with prices and scores.

!!! tip "Attachments, however large"
    Every multimodal route takes its images, documents, audio and video as base64, a data URI, an HTTPS URL, an `s3://` URI or a Files API ID. On chat completions, messages and responses served by Amazon Bedrock — Bedrock Mantle models excepted — an attachment past what the chosen model reads inside a request is delivered by reference instead, with no change to the request, wherever that model reads that kind of attachment from storage; the models that read it inline only refuse it with `413`, stating the size they accept. See [Attachment Size](features.md#attachment-size).

## :material-book-open-variant: Documentation & Tooling

stdapi.ai provides multiple resources for exploring and testing the API—choose the one that fits your workflow:

### :material-book-open-variant: Documentation Resources

* **Per-endpoint guides** – The pages in this section (linked from the [endpoint tables below](#supported-endpoints)) with parameter details, feature tables, and examples
* **[API Reference](api_reference.md)** – Browsable rendering of the full OpenAPI specification (request/response schemas for every endpoint)
* **[OpenAPI Specification](openapi.yml)** – Full machine-readable schema for integration and tooling

### :material-play-circle: Live API Playground

**When running the server**, access these interactive interfaces (can be enabled via [configuration options](operations_configuration.md)):

| Interface          | URL                             | Best For                                                                       |
|--------------------|---------------------------------|--------------------------------------------------------------------------------|
| **Swagger UI**     | `http://localhost/docs`         | Testing endpoints directly in your browser with live request/response examples |
| **ReDoc**          | `http://localhost/redoc`        | Reading and searching through clean, organized documentation                   |
| **OpenAPI Schema** | `http://localhost/openapi.json` | Generating client code or importing into API tools like Postman                |

## :material-api: Supported Endpoints

### ![OpenAI](styles/logo_openai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } OpenAI-Compatible API

| Category          | Endpoint                          | Capability                                                                  | Documentation                                          |
|-------------------|-----------------------------------|-----------------------------------------------------------------------------|--------------------------------------------------------|
| **💬 Chat**       | `POST /v1/chat/completions`       | Multi-modal conversations with text, images, video, documents               | [Chat Completions →](api_openai_chat_completions.md)   |
|                   | `GET /v1/chat/completions`        | List stored chat completions                                                | [Chat Completions →](api_openai_chat_completions.md)   |
|                   | `GET/POST/DELETE /v1/chat/completions/{id}` | Retrieve, update metadata, or delete a stored chat completion         | [Chat Completions →](api_openai_chat_completions.md)   |
|                   | `GET /v1/chat/completions/{id}/messages` | List the messages of a stored chat completion                        | [Chat Completions →](api_openai_chat_completions.md)   |
|                   | `POST /v1/completions`            | Simple prompt-to-text completion — recommended for MCP and text-only agents | [Completions →](api_openai_completions.md)             |
|                   | `POST /v1/responses`              | Conversational AI with tool calling, streaming, and server-side storage     | [Responses →](api_openai_responses.md)                 |
|                   | `POST /v1/responses/input_tokens` | Count input tokens without generating a response                            | [Responses →](api_openai_responses.md)                 |
|                   | `POST /v1/responses/compact`      | Compact a conversation into a reusable summary item                         | [Responses →](api_openai_responses.md)                 |
|                   | `GET/DELETE /v1/responses/{id}`   | Retrieve or delete stored responses                                         | [Responses →](api_openai_responses.md)                 |
|                   | `POST /v1/responses/{id}/cancel`  | Cancel a background response                                                 | [Responses →](api_openai_responses.md)                 |
|                   | `GET /v1/responses/{id}/input_items` | List the input items of a stored response                                | [Responses →](api_openai_responses.md)                 |
| **💬 Conversations** | `POST /v1/conversations`       | Create a conversation holding multi-turn state                              | [Conversations →](api_openai_conversations.md)         |
|                   | `GET/POST/DELETE /v1/conversations/{id}` | Retrieve, update the metadata of, or delete a conversation        | [Conversations →](api_openai_conversations.md)         |
|                   | `GET/POST /v1/conversations/{id}/items` | List or add conversation items                                     | [Conversations →](api_openai_conversations.md)         |
|                   | `GET/DELETE /v1/conversations/{id}/items/{item_id}` | Retrieve or delete one conversation item               | [Conversations →](api_openai_conversations.md)         |
| **🎨 Images**     | `POST /v1/images/generations`     | Text-to-image generation                                                    | [Generations →](api_openai_images_generations.md)      |
|                   | `POST /v1/images/edits`           | Image editing and transformations                                           | [Edits →](api_openai_images_edits.md)                  |
|                   | `POST /v1/images/variations`      | Generate image variations                                                   | [Variations →](api_openai_images_variations.md)        |
| **🎬 Videos**     | `POST/GET/DELETE /v1/videos`      | Asynchronous text/image-to-video generation jobs                            | [Videos →](api_openai_videos.md)                       |
|                   | `GET /v1/videos/{id}/content`     | Download generated video content                                            | [Videos →](api_openai_videos.md)                       |
| **🔊 Audio**      | `POST /v1/audio/speech`           | Text-to-speech synthesis                                                    | [Text to Speech →](api_openai_audio_speech.md)         |
|                   | `POST /v1/audio/transcriptions`   | Speech-to-text transcription                                                | [Transcriptions →](api_openai_audio_transcriptions.md) |
|                   | `POST /v1/audio/translations`     | Speech-to-English translation                                               | [Translations →](api_openai_audio_translations.md)     |
| **🎙️ Realtime**   | `POST /v1/realtime/client_secrets` | Mint a short-lived client secret carrying a session configuration          | [Realtime →](api_openai_realtime.md)                   |
|                   | `WS /v1/realtime`                 | Live, bidirectional speech-to-speech session                                | [Realtime →](api_openai_realtime.md)                   |
| **🧠 Embeddings** | `POST /v1/embeddings`             | Vector embeddings for semantic search                                       | [Embeddings →](api_openai_embeddings.md)               |
| **🛡️ Moderations** | `POST /v1/moderations`           | Content safety classification via Bedrock Guardrails or Amazon Comprehend   | [Moderations →](api_openai_moderations.md)             |
| **📋 Models**     | `GET /v1/models`                  | List available models                                                       | [Models →](api_openai_models.md)                       |
|                   | `GET /v1/models/{model}`          | Retrieve details for one model                                              | [Models →](api_openai_models.md)                       |
| **📁 Files**      | `POST/GET/DELETE /v1/files`       | Upload, list, retrieve, download, delete files                              | [Files →](api_openai_files.md)                         |
|                   | `POST /v1/uploads`                | Multipart upload sessions for large files                                   | [Files →](api_openai_files.md)                         |
|                   | `POST /v1/uploads/{id}/parts`, `…/complete`, `…/cancel` | Add parts to, complete, or cancel an upload session   | [Files →](api_openai_files.md)                         |
| **🔎 Vector Stores** | `POST/GET/DELETE /v1/vector_stores` | Create, list, retrieve, update, delete a searchable file collection       | [Vector Stores →](api_openai_vector_stores.md)         |
|                   | `POST /v1/vector_stores/{id}/search` | Search the indexed files by meaning                                        | [Vector Stores →](api_openai_vector_stores.md)         |
|                   | `POST/GET/DELETE /v1/vector_stores/{id}/files`, `…/file_batches` | Attach, list, read and detach the indexed files          | [Vector Stores →](api_openai_vector_stores.md)         |
| **📦 Batches**    | `POST/GET /v1/batches`            | Run a file of requests asynchronously at the batch price                    | [Batches →](api_openai_batches.md)                     |
|                   | `POST /v1/batches/{id}/cancel`    | Cancel a running batch                                                      | [Batches →](api_openai_batches.md)                     |
| **📊 Usage**      | `GET /v1/organization/usage/completions`, `…/embeddings`, `…/moderations`, `…/images`, `…/audio_speeches`, `…/audio_transcriptions`, `…/web_search_calls`, `…/file_search_calls`, `…/vector_stores`, `…/code_interpreter_sessions` | Consumption in time buckets, grouped by model, endpoint, key or user | [Organization Usage →](api_openai_organization_usage.md) |
|                   | `GET /v1/organization/costs`      | Spend in time buckets, in your AWS partition's currency                     | [Organization Usage →](api_openai_organization_usage.md) |

!!! info "The usage endpoints are an administrator surface"
    `/v1/organization/...` reports the whole deployment's consumption and spend, so it is **disabled by default** — enable it with [`USAGE_API`](operations_configuration.md#usage-api) and read it with the deployment's own API key, or a token carrying every scope in [`USAGE_API_ADMIN_SCOPES`](operations_configuration.md#usage-api-admin-scopes). The retired `GET /v1/usage` endpoint is not served: it is absent from OpenAI's current API surface and from the `openai` SDK.

### :material-magnify: stdapi.ai Native Extensions

| Category | Endpoint | Capability | Documentation |
|----------|----------|------------|---------------|
| **🔍 Models** | `GET /search_models` | Search models by capability: modality, route, MCP tool, region, streaming, batch, legacy status | [Search Models →](api_search_models.md) |
| **💰 Pricing** | `GET /model_pricing` | Exact AWS unit prices per model: tokens, tiers, cache TTLs, routing, media specs | [Model Pricing →](api_model_pricing.md) |

### ![Anthropic](styles/logo_anthropic_claude.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Anthropic-Compatible API

| Category        | Endpoint                                   | Capability                                                    | Documentation                           |
|-----------------|--------------------------------------------|---------------------------------------------------------------|-----------------------------------------|
| **💬 Messages** | `POST /anthropic/v1/messages`              | Multi-modal conversations with text, images, video, documents | [Messages →](api_anthropic_messages.md) |
|                 | `POST /anthropic/v1/messages/count_tokens` | Count tokens without sending a message                        | [Messages →](api_anthropic_messages.md) |
| **📋 Models**   | `GET /anthropic/v1/models`                 | List available models                                         | [Models →](api_anthropic_models.md)     |
|                 | `GET /anthropic/v1/models/{model_id}`      | Retrieve model details                                        | [Models →](api_anthropic_models.md)     |
| **📁 Files**    | `POST/GET/DELETE /anthropic/v1/files`      | Upload, list, retrieve, download, delete files                | [Files →](api_anthropic_files.md)       |
| **📦 Batches**  | `POST/GET/DELETE /anthropic/v1/messages/batches` | Run many message requests asynchronously at the batch price | [Message Batches →](api_anthropic_batches.md) |
|                 | `GET /anthropic/v1/messages/batches/{id}/results` | Stream a finished batch's results as JSONL             | [Message Batches →](api_anthropic_batches.md) |
|                 | `POST /anthropic/v1/messages/batches/{id}/cancel` | Cancel a processing batch                              | [Message Batches →](api_anthropic_batches.md) |

### ![Cohere](styles/logo_cohere.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Cohere-Compatible API

| Category          | Endpoint                 | Capability                                      | Documentation                     |
|-------------------|--------------------------|-------------------------------------------------|-----------------------------------|
| **🔀 Rerank**     | `POST /cohere/v2/rerank` | Rank documents by semantic relevance to a query | [Rerank →](api_cohere_rerank.md)  |
|                   | `POST /cohere/v1/rerank` | Legacy v1 rerank for older SDKs and tools       | [Rerank →](api_cohere_rerank.md#cohere-v1-rerank-api-legacy) |
| **🧠 Embeddings** | `POST /cohere/v2/embed`  | Vector embeddings for semantic search           | [Embed →](api_cohere_embed.md)    |
|                   | `POST /cohere/v1/embed`  | Legacy v1 embed for older SDKs and tools        | [Embed →](api_cohere_embed.md#cohere-v1-embed-api-legacy) |

## :material-tools: MCP (Model Context Protocol)

When `ENABLE_MCP_STREAMABLE_HTTP=true` or `ENABLE_MCP_SSE=true` is configured, stdapi.ai exposes all its endpoints as MCP tools. OpenAI-, Anthropic-, and Cohere-compatible tool names follow the pattern `provider_action`; the native extension tools use their bare names (`search_models`, `model_pricing`).

!!! tip "JSON body support for file and audio tools"
    MCP tools send JSON bodies — they cannot construct `multipart/form-data`. All file upload, audio, and upload-part tools therefore accept the file or audio content as a base64 string, data URI (`data:<mime>;base64,<data>`), HTTPS URL, or S3 URI in the `file` / `data` field instead of a binary attachment — as do the video generation tool's `input_reference` image, the moderation tool's `image_url` input, and the `openai_image_edit`/`openai_image_variation` tools' image inputs (also accepting a bare string in any of these forms, plus a Files API file ID). The full multipart upload workflow (`openai_upload` → `openai_upload_part` → `openai_upload_complete`) is fully MCP-compatible this way.

!!! tip "What the file, video, and audio tools return"
    An MCP tool result carries text, an image, or audio — never an arbitrary binary stream. Endpoints that answer with bytes therefore adapt to what the protocol can hold: text content comes back as text, an image as an image, and generated speech as audio when `stream_format` is set to `audio`. Anything else — a video, a PDF or archive read back through `openai_file_content` or `anthropic_file_content`, and any payload above 3 MB — comes back as a small JSON object holding the media type and the `url` to download it from over HTTP, so an agent is told where the result is rather than handed bytes it cannot use.

| MCP Tool                         | Endpoint                                    |
|----------------------------------|---------------------------------------------|
| **OpenAI Tools**                 |                                             |
| `openai_chat_completion`         | `POST /v1/chat/completions`                 |
| `openai_chat_completion_list`    | `GET /v1/chat/completions`                  |
| `openai_chat_completion_get`     | `GET /v1/chat/completions/{completion_id}`  |
| `openai_chat_completion_update`  | `POST /v1/chat/completions/{completion_id}` |
| `openai_chat_completion_delete`  | `DELETE /v1/chat/completions/{completion_id}` |
| `openai_chat_completion_messages` | `GET /v1/chat/completions/{completion_id}/messages` |
| `openai_completion`              | `POST /v1/completions`                      |
| `openai_response`                | `POST /v1/responses`                        |
| `openai_response_input_tokens`   | `POST /v1/responses/input_tokens`           |
| `openai_response_compact`        | `POST /v1/responses/compact`                |
| `openai_response_get`            | `GET /v1/responses/{response_id}`           |
| `openai_response_delete`         | `DELETE /v1/responses/{response_id}`        |
| `openai_response_cancel`         | `POST /v1/responses/{response_id}/cancel`   |
| `openai_response_input_items`    | `GET /v1/responses/{response_id}/input_items` |
| `openai_conversation`            | `POST /v1/conversations`                    |
| `openai_conversation_get`        | `GET /v1/conversations/{conversation_id}`   |
| `openai_conversation_update`     | `POST /v1/conversations/{conversation_id}`  |
| `openai_conversation_delete`     | `DELETE /v1/conversations/{conversation_id}` |
| `openai_conversation_items`      | `POST /v1/conversations/{conversation_id}/items` |
| `openai_conversation_items_list` | `GET /v1/conversations/{conversation_id}/items` |
| `openai_conversation_item_get`   | `GET /v1/conversations/{conversation_id}/items/{item_id}` |
| `openai_conversation_item_delete` | `DELETE /v1/conversations/{conversation_id}/items/{item_id}` |
| `openai_image_generation`        | `POST /v1/images/generations`               |
| `openai_image_edit`              | `POST /v1/images/edits`                     |
| `openai_image_variation`         | `POST /v1/images/variations`                |
| `openai_video_generation`        | `POST /v1/videos`                           |
| `openai_video_list`              | `GET /v1/videos`                            |
| `openai_video_get`               | `GET /v1/videos/{video_id}`                 |
| `openai_video_content`           | `GET /v1/videos/{video_id}/content`         |
| `openai_video_delete`            | `DELETE /v1/videos/{video_id}`              |
| `openai_audio_speech`            | `POST /v1/audio/speech`                     |
| `openai_audio_transcription`     | `POST /v1/audio/transcriptions`             |
| `openai_audio_translation`       | `POST /v1/audio/translations`               |
| `openai_realtime_client_secret`  | `POST /v1/realtime/client_secrets`          |
| `openai_embedding`               | `POST /v1/embeddings`                       |
| `openai_moderation`              | `POST /v1/moderations`                      |
| `openai_model_list`              | `GET /v1/models`                            |
| `openai_model_get`               | `GET /v1/models/{model}`                    |
| `openai_file`                    | `POST /v1/files`                            |
| `openai_file_list`               | `GET /v1/files`                             |
| `openai_files_get`               | `GET /v1/files/{file_id}`                   |
| `openai_files_delete`            | `DELETE /v1/files/{file_id}`                |
| `openai_file_content`            | `GET /v1/files/{file_id}/content`           |
| `openai_vector_store_create`     | `POST /v1/vector_stores`                    |
| `openai_vector_store_list`       | `GET /v1/vector_stores`                     |
| `openai_vector_store_get`        | `GET /v1/vector_stores/{vector_store_id}`   |
| `openai_vector_store_update`     | `POST /v1/vector_stores/{vector_store_id}`  |
| `openai_vector_store_delete`     | `DELETE /v1/vector_stores/{vector_store_id}` |
| `openai_vector_store_search`     | `POST /v1/vector_stores/{vector_store_id}/search` |
| `openai_vector_store_file_create` | `POST /v1/vector_stores/{vector_store_id}/files` |
| `openai_vector_store_file_list`  | `GET /v1/vector_stores/{vector_store_id}/files` |
| `openai_vector_store_file_get`   | `GET /v1/vector_stores/{vector_store_id}/files/{file_id}` |
| `openai_vector_store_file_update` | `POST /v1/vector_stores/{vector_store_id}/files/{file_id}` |
| `openai_vector_store_file_delete` | `DELETE /v1/vector_stores/{vector_store_id}/files/{file_id}` |
| `openai_vector_store_file_content` | `GET /v1/vector_stores/{vector_store_id}/files/{file_id}/content` |
| `openai_vector_store_file_batch_create` | `POST /v1/vector_stores/{vector_store_id}/file_batches` |
| `openai_vector_store_file_batch_get` | `GET /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}` |
| `openai_vector_store_file_batch_cancel` | `POST /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/cancel` |
| `openai_vector_store_file_batch_file_list` | `GET /v1/vector_stores/{vector_store_id}/file_batches/{batch_id}/files` |
| `openai_batch`                   | `POST /v1/batches`                          |
| `openai_batch_list`              | `GET /v1/batches`                           |
| `openai_batch_get`               | `GET /v1/batches/{batch_id}`                |
| `openai_batch_cancel`            | `POST /v1/batches/{batch_id}/cancel`        |
| `openai_upload`                  | `POST /v1/uploads`                          |
| `openai_upload_part`             | `POST /v1/uploads/{upload_id}/parts`        |
| `openai_upload_complete`         | `POST /v1/uploads/{upload_id}/complete`     |
| `openai_upload_cancel`           | `POST /v1/uploads/{upload_id}/cancel`       |
| **Anthropic Tools**              |                                             |
| `anthropic_message`              | `POST /anthropic/v1/messages`               |
| `anthropic_message_count_tokens` | `POST /anthropic/v1/messages/count_tokens`  |
| `anthropic_model_list`           | `GET /anthropic/v1/models`                  |
| `anthropic_model_get`            | `GET /anthropic/v1/models/{model_id}`       |
| `anthropic_file`                 | `POST /anthropic/v1/files`                  |
| `anthropic_file_list`            | `GET /anthropic/v1/files`                   |
| `anthropic_files_get`            | `GET /anthropic/v1/files/{file_id}`         |
| `anthropic_files_delete`         | `DELETE /anthropic/v1/files/{file_id}`      |
| `anthropic_file_content`         | `GET /anthropic/v1/files/{file_id}/content` |
| `anthropic_message_batch`        | `POST /anthropic/v1/messages/batches`       |
| `anthropic_message_batch_list`   | `GET /anthropic/v1/messages/batches`        |
| `anthropic_message_batch_get`    | `GET /anthropic/v1/messages/batches/{message_batch_id}` |
| `anthropic_message_batch_results` | `GET /anthropic/v1/messages/batches/{message_batch_id}/results` |
| `anthropic_message_batch_cancel` | `POST /anthropic/v1/messages/batches/{message_batch_id}/cancel` |
| `anthropic_message_batch_delete` | `DELETE /anthropic/v1/messages/batches/{message_batch_id}` |
| **Cohere Tools**                 |                                             |
| `cohere_rerank`                  | `POST /cohere/v2/rerank`                    |
| `cohere_rerank_v1`               | `POST /cohere/v1/rerank`                    |
| `cohere_embed`                   | `POST /cohere/v2/embed`                     |
| `cohere_embed_v1`                | `POST /cohere/v1/embed`                     |
| **Native Extension Tools**       |                                             |
| `search_models`                  | `GET /search_models`                        |
| `model_pricing`                  | `GET /model_pricing`                        |

!!! tip "Filtering MCP Tools"
    Use `MCP_INCLUDE_TOOLS` or `MCP_EXCLUDE_TOOLS` environment variables to control which tools are exposed. Always include `search_models` so agents can discover the right model ID dynamically. See [Operations Configuration →](operations_configuration.md#mcp-model-context-protocol) for details.

!!! warning "Token Usage for Complex API Tools"
    `anthropic_message`, `openai_chat_completion`, and `openai_response` map to large, complex APIs that may use many tokens (prompt, completion, and tool definitions). Select these tools only if your workflow requires the full API capabilities.

## :material-connection: Using stdapi.ai

stdapi.ai speaks the OpenAI, Anthropic, and Cohere APIs unchanged. Any application built on one of them—chatbots, coding assistants, automation tools, custom scripts—runs against stdapi.ai once you point it at your deployment's base URL and give it that deployment's API key. The model name usually stays as it is, and changes only where it differs.

That is because the Anthropic, OpenAI and Cohere models Bedrock serves are also published under the names their providers use, derived mechanically from the Bedrock identifier rather than curated by hand: `anthropic.claude-opus-5` answers to `claude-opus-5`, `openai.gpt-5.6-sol` to `gpt-5.6-sol`, `openai.gpt-oss-120b-1:0` to `gpt-oss-120b`, `cohere.embed-english-v3` to `embed-english-v3.0`, `cohere.rerank-v3-5:0` to `rerank-v3.5`. A client already asking for one of those names needs no model change at all. Where a name *does* differ — a model from another provider, or one named for a provider this deployment does not serve — [`MODEL_ALIASES`](operations_configuration.md#model-aliases) publishes a served model under the name your application already sends.

What the base URL buys is the catalogue behind it. A model name is resolved against the catalogue your deployment actually serves — Amazon Bedrock, Bedrock Mantle, Polly, Transcribe and Comprehend, across every region you enable — so the choice spans providers instead of one vendor's list. A name the catalogue does not contain is answered with `404`: it is never mapped onto another vendor's model of roughly similar class, because that would serve you a different model than the one you asked for. Use [`GET /search_models`](api_search_models.md) to find one.

Anywhere a request accepts a model name, it also accepts a glob pattern — `claude-sonnet-*`, say — and the server serves the most recently released model that matches. The response always names the concrete model that served the request, never the pattern. See [Model Wildcard Patterns](operations_configuration.md#model-wildcard-patterns) for the syntax and its rules, and [`GET /search_models`](api_search_models.md#query-parameters) to see everything a pattern matches before relying on it.

### ![OpenAI](styles/logo_openai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Using the OpenAI-Compatible API

**To connect your OpenAI application:**

1. **Replace the OpenAI API URL** with your stdapi.ai deployment URL
2. **Use the same authentication mechanism** (Bearer token in the `Authorization` header)
3. **Check the model name against what this deployment serves** — OpenAI's own names for the models Bedrock offers (e.g., `gpt-5.6-sol`, `gpt-oss-120b`) resolve as they stand, as do Bedrock model IDs (e.g., `amazon.nova-micro-v1:0`) and any configured alias. A name Bedrock does not serve, such as `gpt-4o` or `dall-e-3`, returns `404` until you [alias](operations_configuration.md#model-aliases) it onto one it does

That's it: the rest of the OpenAI SDK call is unchanged.

### ![Anthropic](styles/logo_anthropic_claude.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Using the Anthropic-Compatible API

**To connect your Anthropic application:**

1. **Replace the Anthropic API URL** (`https://api.anthropic.com`) with your stdapi.ai deployment URL + `/anthropic` (e.g., `https://your-endpoint.com/anthropic`)
2. **Use the same authentication mechanism** (`x-api-key` header and `anthropic-version` header)
3. **Check the model name against what this deployment serves** — official Anthropic names (e.g., `claude-opus-5`) resolve to their Bedrock IDs automatically, or use Bedrock model IDs directly

Anthropic names resolving on their own makes the base URL the only change for most applications — the same mechanism that resolves OpenAI's names on the surface above. A Claude version Bedrock no longer serves returns `404` rather than a substitute, so name a current one.

### ![Cohere](styles/logo_cohere.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Using the Cohere-Compatible API

**To connect your Cohere application:**

1. **Replace the Cohere API URL** (`https://api.cohere.com`) with your stdapi.ai deployment URL + `/cohere` (e.g., `https://your-endpoint.com/cohere`)
2. **Use the same authentication mechanism** (Bearer token in the `Authorization` header)
3. **Check the model name against what this deployment serves** — Cohere's own names for the models Bedrock offers (e.g., `embed-english-v3.0`, `embed-v4.0`, `rerank-v3.5`) resolve as they stand, as do Bedrock model IDs (e.g., `cohere.rerank-v3-5:0`, `cohere.embed-v4:0`) and any configured alias. A Cohere model Bedrock does not serve, such as `embed-english-light-v3.0`, returns `404` until you [alias](operations_configuration.md#model-aliases) it onto one it does

That's it: your Cohere rerank and embed integrations are otherwise unchanged.

## :material-arrow-right: Next Steps

<div class="grid cards" markdown>

- :material-chat: [**Chat Completions**](api_openai_chat_completions.md) — Conversational AI with multi-modal support
- :material-image: [**Images**](api_openai_images_generations.md) — Generation, edits, and variations
- :material-movie-open: [**Videos**](api_openai_videos.md) — Asynchronous text/image-to-video generation
- :material-music: [**Audio**](api_openai_audio_speech.md) — Text-to-speech, transcription, and translation
- :material-vector-polyline: [**Embeddings**](api_openai_embeddings.md) — Vector embeddings for search and RAG
- :material-sort: [**Rerank**](api_cohere_rerank.md) — Cohere-compatible document reranking for search and RAG
- :material-format-list-bulleted: [**Models**](api_openai_models.md) — List and discover available models
- :material-view-list: [**Models**](models.md) — Every model served, with modalities, regions, AWS prices and leaderboard scores
- :material-magnify: [**Search Models**](api_search_models.md) — Filter models by capability, modality, route, or MCP tool
- :material-currency-usd: [**Model Pricing**](api_model_pricing.md) — Exact AWS unit prices for cost-aware model selection
- :material-message: [**Messages**](api_anthropic_messages.md) — Anthropic-compatible conversational AI with tool calling
- :material-check-all: [**Features**](features.md) — Full capabilities and AWS integrations
- :material-rocket-launch: [**Getting Started**](operations_getting_started.md) — Deploy to AWS with Terraform
- :material-puzzle: [**Use Cases**](use_cases.md) — Integration examples with popular tools
- :material-cash-multiple: [**Cost Management**](operations_cost_management.md) — Model, infrastructure, and license costs, and per-request cost estimation
- :material-email-outline: [**Contact**](contact.md) — Technical questions, sales, and private offers

</div>
