---
title: API Overview - OpenAI & Anthropic Compatible AWS Bedrock API
description: Complete API documentation for stdapi.ai OpenAI and Anthropic compatible gateway. Access AWS Bedrock models, chat completions, messages, embeddings, image generation, and audio APIs with SDK compatibility.
keywords: OpenAI API documentation, Anthropic API documentation, AWS Bedrock API reference, OpenAI SDK compatibility, Anthropic SDK compatibility, chat completions API, messages API, embeddings API, image generation API, audio API AWS, OpenAI compatible endpoints, Anthropic compatible endpoints
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
| **🎨 Images**     | `POST /v1/images/generations`   | Text-to-image generation                                      | [Generations →](api_openai_images_generations.md)      |
|                   | `POST /v1/images/edits`         | Image editing and transformations                             | [Edits →](api_openai_images_edits.md)                  |
|                   | `POST /v1/images/variations`    | Generate image variations                                     | [Variations →](api_openai_images_variations.md)        |
| **🔊 Audio**      | `POST /v1/audio/speech`         | Text-to-speech synthesis                                      | [Text to Speech →](api_openai_audio_speech.md)         |
|                   | `POST /v1/audio/transcriptions` | Speech-to-text transcription                                  | [Transcriptions →](api_openai_audio_transcriptions.md) |
|                   | `POST /v1/audio/translations`   | Speech-to-English translation                                 | [Translations →](api_openai_audio_translations.md)     |
| **🧠 Embeddings** | `POST /v1/embeddings`           | Vector embeddings for semantic search                         | [Embeddings →](api_openai_embeddings.md)               |
| **📋 Models**     | `GET /v1/models`                | List available models                                         | [Models →](api_openai_models.md)                       |

### ![Anthropic](styles/logo_anthropic_claude.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Anthropic-Compatible API

| Category      | Endpoint                     | Capability                                                    | Documentation                                |
|---------------|------------------------------|---------------------------------------------------------------|----------------------------------------------|
| **💬 Messages** | `POST /anthropic/v1/messages` | Multi-modal conversations with text, images, video, documents | [Messages →](api_anthropic_messages.md)      |

## :material-connection: Using stdapi.ai

stdapi.ai is a **drop-in replacement** for both OpenAI and Anthropic APIs. Any application that works with either provider—chatbots, coding assistants, automation tools, custom scripts—works with stdapi.ai by simply changing the API base URL.

### ![OpenAI](styles/logo_openai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Using the OpenAI-Compatible API

**To connect your OpenAI application:**

1. **Replace the OpenAI API URL** with your stdapi.ai deployment URL
2. **Use the same authentication mechanism** (Bearer token in the `Authorization` header)
3. **Use AWS Bedrock model IDs** instead of OpenAI model names (e.g., `amazon.nova-micro-v1:0`)

That's it. Your application continues to work without any code changes—just point it to stdapi.ai instead of OpenAI.

### ![Anthropic](styles/logo_anthropic_claude.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Using the Anthropic-Compatible API

**To connect your Anthropic application:**

1. **Replace the Anthropic API URL** (`https://api.anthropic.com`) with your stdapi.ai deployment URL + `/anthropic` (e.g., `https://your-endpoint.com/anthropic`)
2. **Use the same authentication mechanism** (`x-api-key` header and `anthropic-version` header)
3. **Use AWS Bedrock model IDs** instead of Anthropic model names (e.g., `anthropic.claude-opus-4-6-v1` instead of `claude-opus-4.6-20250514`)

Your Anthropic SDK applications continue to work without any code changes—just point them to stdapi.ai instead of Anthropic.

## :material-arrow-right: Next Steps

<div class="grid cards" markdown>

- :material-chat: [**Chat Completions**](api_openai_chat_completions.md) — Conversational AI with multi-modal support
- :material-image: [**Images**](api_openai_images_generations.md) — Generation, edits, and variations
- :material-music: [**Audio**](api_openai_audio_speech.md) — Text-to-speech, transcription, and translation
- :material-vector-polyline: [**Embeddings**](api_openai_embeddings.md) — Vector embeddings for search and RAG
- :material-format-list-bulleted: [**Models**](api_openai_models.md) — List and discover available models
- :material-message: [**Messages**](api_anthropic_messages.md) — Anthropic-compatible conversational AI with tool calling
- :material-check-all: [**Features**](features.md) — Full capabilities and AWS integrations
- :material-rocket-launch: [**Getting Started**](operations_getting_started.md) — Deploy to AWS with Terraform
- :material-puzzle: [**Use Cases**](use_cases.md) — Integration examples with popular tools

</div>
