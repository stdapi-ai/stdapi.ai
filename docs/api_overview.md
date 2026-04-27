---
title: API Overview - OpenAI & Anthropic Compatible AWS Bedrock API
description: Complete API documentation for stdapi.ai OpenAI and Anthropic compatible gateway. Access AWS Bedrock models, chat completions, messages, embeddings, image generation, and audio APIs with SDK compatibility.
keywords: OpenAI API documentation, Anthropic API documentation, AWS Bedrock API reference, OpenAI SDK compatibility, Anthropic SDK compatibility, chat completions API, responses API, messages API, embeddings API, image generation API, audio API AWS, OpenAI compatible endpoints, Anthropic compatible endpoints
---

# :material-api: API Overview

stdapi.ai provides OpenAI and Anthropic compatible APIs backed by AWS Bedrock and AWS AI services. Any application that works with OpenAI or Anthropic works with stdapi.ai by simply changing the API endpoint.

## :material-book-open-variant: Interactive Documentation

stdapi.ai provides multiple interfaces for exploring and testing the API—choose the one that fits your workflow:

### :material-book-open-variant: Documentation Resources

* **[Complete API Reference](api_reference.md)** – In-depth guides for every endpoint with parameter details
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

| Category          | Endpoint                        | Capability                                                    | Documentation                                          |
|-------------------|---------------------------------|---------------------------------------------------------------|--------------------------------------------------------|
| **💬 Chat**       | `POST /v1/chat/completions`     | Multi-modal conversations with text, images, video, documents | [Chat Completions →](api_openai_chat_completions.md)   |
|                   | `POST /v1/responses`            | Stateless conversational AI with tool calling and streaming   | [Responses →](api_openai_responses.md)                 |
| **🎨 Images**     | `POST /v1/images/generations`   | Text-to-image generation                                      | [Generations →](api_openai_images_generations.md)      |
|                   | `POST /v1/images/edits`         | Image editing and transformations                             | [Edits →](api_openai_images_edits.md)                  |
|                   | `POST /v1/images/variations`    | Generate image variations                                     | [Variations →](api_openai_images_variations.md)        |
| **🔊 Audio**      | `POST /v1/audio/speech`         | Text-to-speech synthesis                                      | [Text to Speech →](api_openai_audio_speech.md)         |
|                   | `POST /v1/audio/transcriptions` | Speech-to-text transcription                                  | [Transcriptions →](api_openai_audio_transcriptions.md) |
|                   | `POST /v1/audio/translations`   | Speech-to-English translation                                 | [Translations →](api_openai_audio_translations.md)     |
| **🧠 Embeddings** | `POST /v1/embeddings`           | Vector embeddings for semantic search                         | [Embeddings →](api_openai_embeddings.md)               |
| **📋 Models**     | `GET /v1/models`                | List available models                                         | [Models →](api_openai_models.md)                       |
| **📁 Files**      | `POST/GET/DELETE /v1/files`     | Upload, list, retrieve, download, delete files                | [Files →](api_openai_files.md)                         |
|                   | `POST /v1/uploads`              | Multipart upload sessions for large files                     | [Files →](api_openai_files.md)                         |

### :material-magnify: stdapi.ai Native Extensions

| Category | Endpoint | Capability | Documentation |
|----------|----------|------------|---------------|
| **🔍 Models** | `GET /search_models` | Search models by capability: modality, route, MCP tool, region, streaming, legacy status | [Search Models →](api_search_models.md) |

### ![Anthropic](styles/logo_anthropic_claude.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Anthropic-Compatible API

| Category        | Endpoint                                   | Capability                                                    | Documentation                           |
|-----------------|--------------------------------------------|---------------------------------------------------------------|-----------------------------------------|
| **💬 Messages** | `POST /anthropic/v1/messages`              | Multi-modal conversations with text, images, video, documents | [Messages →](api_anthropic_messages.md) |
|                 | `POST /anthropic/v1/messages/count_tokens` | Count tokens without sending a message                        | [Messages →](api_anthropic_messages.md) |
| **📋 Models**   | `GET /anthropic/v1/models`                 | List available models                                         | [Models →](api_anthropic_models.md)     |
|                 | `GET /anthropic/v1/models/{model_id}`      | Retrieve model details                                        | [Models →](api_anthropic_models.md)     |
| **📁 Files**    | `POST/GET/DELETE /anthropic/v1/files`      | Upload, list, retrieve, download, delete files                | [Files →](api_anthropic_files.md)       |

## :material-tools: MCP (Model Context Protocol)

When `ENABLE_MCP_STREAMABLE_HTTP=true` or `ENABLE_MCP_SSE=true` is configured, stdapi.ai exposes all its endpoints as MCP tools. The tool names follow the pattern `provider_action`.

!!! tip "JSON body support for file and audio tools"
    MCP tools send JSON bodies — they cannot construct `multipart/form-data`. All file upload, audio, and upload-part tools therefore accept the file or audio content as a base64 string, data URI (`data:<mime>;base64,<data>`), HTTPS URL, or S3 URI in the `file` / `data` field instead of a binary attachment. The full multipart upload workflow (`openai_upload` → `openai_upload_part` → `openai_upload_complete`) is fully MCP-compatible this way.

| MCP Tool                          | Endpoint                                |
|-----------------------------------|-----------------------------------------|
| **OpenAI Tools**                  |                                         |
| `openai_chat_completion`          | `POST /v1/chat/completions`             |
| `openai_response`                 | `POST /v1/responses`                    |
| `openai_image_generation`         | `POST /v1/images/generations`           |
| `openai_image_edit`               | `POST /v1/images/edits`                 |
| `openai_image_variation`          | `POST /v1/images/variations`            |
| `openai_audio_speech`             | `POST /v1/audio/speech`                 |
| `openai_audio_transcription`      | `POST /v1/audio/transcriptions`         |
| `openai_audio_translation`        | `POST /v1/audio/translations`           |
| `openai_embedding`                | `POST /v1/embeddings`                   |
| `openai_model_list`               | `GET /v1/models`                        |
| `openai_model_get`                | `GET /v1/models/{model}`                |
| `openai_file`                     | `POST /v1/files`                        |
| `openai_file_list`                | `GET /v1/files`                         |
| `openai_files_get`                | `GET /v1/files/{file_id}`               |
| `openai_files_delete`             | `DELETE /v1/files/{file_id}`            |
| `openai_file_content`             | `GET /v1/files/{file_id}/content`       |
| `openai_upload`                   | `POST /v1/uploads`                      |
| `openai_upload_part`              | `POST /v1/uploads/{upload_id}/parts`    |
| `openai_upload_complete`          | `POST /v1/uploads/{upload_id}/complete` |
| `openai_upload_cancel`            | `POST /v1/uploads/{upload_id}/cancel`   |
| **Anthropic Tools**               |                                         |
| `anthropic_message`               | `POST /anthropic/v1/messages`           |
| `anthropic_message_count_tokens`  | `POST /anthropic/v1/messages/count_tokens` |
| `anthropic_model_list`            | `GET /anthropic/v1/models`              |
| `anthropic_model_get`             | `GET /anthropic/v1/models/{model_id}`   |
| `anthropic_file`                  | `POST /anthropic/v1/files`              |
| `anthropic_file_list`             | `GET /anthropic/v1/files`               |
| `anthropic_files_get`             | `GET /anthropic/v1/files/{file_id}`     |
| `anthropic_files_delete`          | `DELETE /anthropic/v1/files/{file_id}`  |
| `anthropic_file_content`          | `GET /anthropic/v1/files/{file_id}/content` |
| **Native Extension Tools**        |                                         |
| `search_models`                   | `GET /search_models`                    |

!!! tip "Filtering MCP Tools"
    Use `MCP_INCLUDE_TOOLS` or `MCP_EXCLUDE_TOOLS` environment variables to control which tools are exposed. Always include `search_models` so agents can discover the right model ID dynamically. See [Operations Configuration →](operations_configuration.md#mcp-model-context-protocol) for details.

!!! warning "Token Usage for Complex API Tools"
    `anthropic_message`, `openai_chat_completion`, and `openai_response` map to large, complex APIs that may use many tokens (prompt, completion, and tool definitions). Select these tools only if your workflow requires the full API capabilities.

## :material-connection: Using stdapi.ai

stdapi.ai is a **drop-in replacement** for both OpenAI and Anthropic APIs. Any application that works with either provider—chatbots, coding assistants, automation tools, custom scripts—works with stdapi.ai by simply changing the API base URL.

### ![OpenAI](styles/logo_openai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Using the OpenAI-Compatible API

**To connect your OpenAI application:**

1. **Replace the OpenAI API URL** with your stdapi.ai deployment URL
2. **Use the same authentication mechanism** (Bearer token in the `Authorization` header)
3. **Use AWS Bedrock model IDs** (e.g., `amazon.nova-micro-v1:0`) or any configured model alias

That's it. Your application continues to work without any code changes—just point it to stdapi.ai instead of OpenAI.

### ![Anthropic](styles/logo_anthropic_claude.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Using the Anthropic-Compatible API

**To connect your Anthropic application:**

1. **Replace the Anthropic API URL** (`https://api.anthropic.com`) with your stdapi.ai deployment URL + `/anthropic` (e.g., `https://your-endpoint.com/anthropic`)
2. **Use the same authentication mechanism** (`x-api-key` header and `anthropic-version` header)
3. **Use your preferred model name** — official Anthropic names (e.g., `claude-opus-4-6`) are automatically resolved to Bedrock IDs, or use Bedrock model IDs directly

Your Anthropic SDK applications continue to work without any code changes—just point them to stdapi.ai instead of Anthropic.

## :material-arrow-right: Next Steps

<div class="grid cards" markdown>

- :material-chat: [**Chat Completions**](api_openai_chat_completions.md) — Conversational AI with multi-modal support
- :material-image: [**Images**](api_openai_images_generations.md) — Generation, edits, and variations
- :material-music: [**Audio**](api_openai_audio_speech.md) — Text-to-speech, transcription, and translation
- :material-vector-polyline: [**Embeddings**](api_openai_embeddings.md) — Vector embeddings for search and RAG
- :material-format-list-bulleted: [**Models**](api_openai_models.md) — List and discover available models
- :material-magnify: [**Search Models**](api_search_models.md) — Filter models by capability, modality, route, or MCP tool
- :material-message: [**Messages**](api_anthropic_messages.md) — Anthropic-compatible conversational AI with tool calling
- :material-check-all: [**Features**](features.md) — Full capabilities and AWS integrations
- :material-rocket-launch: [**Getting Started**](operations_getting_started.md) — Deploy to AWS with Terraform
- :material-puzzle: [**Use Cases**](use_cases.md) — Integration examples with popular tools

</div>
