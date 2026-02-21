---
title: API Overview - OpenAI-Compatible AWS Bedrock API
description: Complete API documentation for stdapi.ai OpenAI-compatible gateway. Access AWS Bedrock models, chat completions, embeddings, image generation, and audio APIs with OpenAI SDK compatibility.
keywords: OpenAI API documentation, AWS Bedrock API reference, OpenAI SDK compatibility, chat completions API, embeddings API, image generation API, audio API AWS, OpenAI compatible endpoints
---

# API Overview

stdapi.ai provides an OpenAI-compatible API backed by AWS Bedrock and AWS AI services. Any application that works with OpenAI works with stdapi.ai by simply changing the API endpoint.

## Interactive Documentation

stdapi.ai provides multiple interfaces for exploring and testing the API—choose the one that fits your workflow:

### 📚 Documentation Resources

* **[Complete API Reference](api_reference.md)** – In-depth guides for every endpoint with parameter details
* **[OpenAPI Specification](openapi.yml)** – Full machine-readable schema for integration and tooling

### 🎮 Live API Playground

**When running the server**, access these interactive interfaces (can be enabled via [configuration options](operations_configuration.md)):

| Interface          | URL                             | Best For                                                                       |
|--------------------|---------------------------------|--------------------------------------------------------------------------------|
| **Swagger UI**     | `http://localhost/docs`         | Testing endpoints directly in your browser with live request/response examples |
| **ReDoc**          | `http://localhost/redoc`        | Reading and searching through clean, organized documentation                   |
| **OpenAPI Schema** | `http://localhost/openapi.json` | Generating client code or importing into API tools like Postman                |

## Supported Endpoints

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

## Using stdapi.ai

stdapi.ai is a **drop-in replacement** for the OpenAI API. Any application that works with OpenAI—chatbots, coding assistants, automation tools, custom scripts—works with stdapi.ai by simply changing the API base URL.

**To connect your application:**

1. **Replace the OpenAI API URL** with your stdapi.ai deployment URL
2. **Use the same authentication mechanism** (Bearer token in the `Authorization` header)
3. **Use AWS Bedrock model IDs** instead of OpenAI model names (e.g., `amazon.nova-micro-v1:0`)

That's it. Your application continues to work without any code changes—just point it to stdapi.ai instead of OpenAI.

## Next Steps

**Explore the API:**

- [Chat Completions](api_openai_chat_completions.md) - Conversational AI with multi-modal support
- [Images](api_openai_images_generations.md) - Generation, edits, and variations
- [Audio](api_openai_audio_speech.md) - Text-to-speech, transcription, and translation
- [Embeddings](api_openai_embeddings.md) - Vector embeddings for search and RAG
- [Models](api_openai_models.md) - List and discover available models

**Learn more:**

- [Features](features.md) - Full capabilities and AWS integrations
- [Getting Started](operations_getting_started.md) - Deploy to AWS with Terraform
- [Use Cases](use_cases.md) - Integration examples with popular tools
