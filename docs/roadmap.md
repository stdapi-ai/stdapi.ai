# Roadmap & Changelog

## What's Next

The following features may be implemented in future releases based on community demand and feedback. Implementation priority is determined by user requests, use case requirements, and alignment with the project's goals. All features can be implemented using AWS services as backends. If you need a specific feature, submit feedback or contribute to discussions.

### 💬 Chat Completions

| Provider                                                                                 | Endpoint/Feature                             | AWS Backend                                                                                                            |
|------------------------------------------------------------------------------------------|----------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**           | `/v1/completions`                            | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**           | `/v1/responses`                              | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Claude](styles/logo_anthropic_claude.svg){: style="height:20px;width:20px"} **Claude** | `/v1/messages`                               | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Claude](styles/logo_anthropic_claude.svg){: style="height:20px;width:20px"} **Claude** | Extended thinking mode                       | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Claude thinking   |
| ![Ollama](styles/logo_ollama.svg){: style="height:20px;width:20px"} **Ollama**           | `/api/generate`                              | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Ollama](styles/logo_ollama.svg){: style="height:20px;width:20px"} **Ollama**           | `/api/chat`                                  | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Cohere](styles/logo_cohere.svg){: style="height:20px;width:20px"} **Cohere**           | `/v1/chat`                                   | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Qwen](styles/logo_qwen.svg){: style="height:20px;width:20px"} **Qwen**                 | `/v1/chat/completions` `translation_options` | ![Amazon Translate](styles/logo_amazon_translate.svg){: style="height:20px;width:20px"} Amazon Translate               |

### 🌐 Translation

| Provider  | Endpoint/Feature | AWS Backend                                                                                              |
|-----------|------------------|----------------------------------------------------------------------------------------------------------|
| **DeepL** | `/v2/translate`  | ![Amazon Translate](styles/logo_amazon_translate.svg){: style="height:20px;width:20px"} Amazon Translate |

### 🧠 Embeddings

| Provider                                                                       | Endpoint/Feature  | AWS Backend                                                                                                           |
|--------------------------------------------------------------------------------|-------------------|-----------------------------------------------------------------------------------------------------------------------|
| ![Ollama](styles/logo_ollama.svg){: style="height:20px;width:20px"} **Ollama** | `/api/embeddings` | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - embedding models |
| ![Cohere](styles/logo_cohere.svg){: style="height:20px;width:20px"} **Cohere** | `/v1/embed`       | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - embedding models |

### 🔍 Semantic Search & Ranking

| Provider                                                                       | Endpoint/Feature | AWS Backend                                                                                                        |
|--------------------------------------------------------------------------------|------------------|--------------------------------------------------------------------------------------------------------------------|
| ![Cohere](styles/logo_cohere.svg){: style="height:20px;width:20px"} **Cohere** | `/v1/rerank`     | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - rerank models |

### 📋 Model Discovery

| Provider                                                                                        | Endpoint/Feature                                                   | AWS Backend                                                                                                        |
|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| ![Ollama](styles/logo_ollama.svg){: style="height:20px;width:20px"} **Ollama**                  | `/api/tags`                                                        | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - model listing |
| ![Ollama](styles/logo_ollama.svg){: style="height:20px;width:20px"} **Ollama**                  | `/api/show`                                                        | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - model details |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v1/engines/list`                                                 | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - model catalog |
|                                                                                                 | Model selection wildcards (To automatically latest model versions) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - model catalog |

### 🎙️ Speech & Audio

| Provider                                                                       | Endpoint/Feature                      | AWS Backend                                                                                                                |
|--------------------------------------------------------------------------------|---------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/realtime/sessions`               | ![Amazon Nova Sonic](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} Amazon Nova Sonic                      |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/realtime/transcription_sessions` | ![Amazon Nova Sonic](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} Amazon Nova Sonic                      |
|                                                                                | Transcriptions with Nova Sonic        | ![Amazon Nova Sonic](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} Amazon Nova Sonic                      |
|                                                                                | Translations with Nova Sonic          | ![Amazon Nova Sonic](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} Amazon Nova Sonic                      |
|                                                                                | Long-form speech (async)              | ![Amazon Polly](styles/logo_amazon_polly.svg){: style="height:20px;width:20px"} Amazon Polly - async tasks                 |
|                                                                                | Streaming transcription               | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe - streaming    |
|                                                                                | Custom vocabularies                   | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe - custom vocab |

### 🎨 Image Generation

| Provider                                                                                        | Endpoint/Feature                        | AWS Backend                                                                                                       |
|-------------------------------------------------------------------------------------------------|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v1/generation/text-to-image`          | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v1/generation/image-to-image`         | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v1/generation/image-to-image/masking` | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v2/stable-image/generate`             | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v2/stable-image/upscale`              | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v2/stable-image/edit`                 | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |

### 🎯 Model-Specific Features

| Provider | Endpoint/Feature               | AWS Backend                                                                                                             |
|----------|--------------------------------|-------------------------------------------------------------------------------------------------------------------------|
|          | Running Provisioned throughput | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - provisioned models |

### 🤖 AWS Bedrock Advanced Features

| Provider                                                                       | Endpoint/Feature       | AWS Backend                                                                                                           |
|--------------------------------------------------------------------------------|------------------------|-----------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/agents`           | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Agents           |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/vector_stores`    | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Knowledge Bases  |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/evals`            | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Model Evaluation |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/fine_tuning/jobs` | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - custom models    |

### 📦 Batch & Async Processing

| Provider                                                                                 | Endpoint/Feature       | AWS Backend                                                                                                          |
|------------------------------------------------------------------------------------------|------------------------|----------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**           | `/v1/batches`          | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - batch inference |
| ![Claude](styles/logo_anthropic_claude.svg){: style="height:20px;width:20px"} **Claude** | `/v1/messages/batches` | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - batch inference |

### 🛡️ Content Safety & Moderation

| Provider                                                                       | Endpoint/Feature  | AWS Backend                                                                                                            |
|--------------------------------------------------------------------------------|-------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/moderations` | ![Amazon Comprehend](styles/logo_amazon_comprehend.svg){: style="height:20px;width:20px"} Amazon Comprehend - toxicity |

### 📁 Files & Storage

| Provider                                                                       | Endpoint/Feature | AWS Backend                                                                         |
|--------------------------------------------------------------------------------|------------------|-------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/files`      | ![Amazon S3](styles/logo_amazon_s3.svg){: style="height:20px;width:20px"} Amazon S3 |

### 📊 Usage & Analytics

| Provider                                                                       | Endpoint/Feature         | AWS Backend                                                                                                 |
|--------------------------------------------------------------------------------|--------------------------|-------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/usage`              | ![Amazon CloudWatch](styles/logo_amazon_cloudwatch.svg){: style="height:20px;width:20px"} Amazon CloudWatch |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/organization/usage` | ![Amazon CloudWatch](styles/logo_amazon_cloudwatch.svg){: style="height:20px;width:20px"} Amazon CloudWatch |

### 🔐 Authentication & Access Control

| Provider | Endpoint/Feature      | AWS Backend                                                                                                          |
|----------|-----------------------|----------------------------------------------------------------------------------------------------------------------|
|          | User authentication   | ![Amazon Cognito](styles/logo_amazon_cognito.svg){: style="height:20px;width:20px"} Amazon Cognito                   |
|          | Multi-tenant API keys | ![Amazon DynamoDB](styles/logo_amazon_dynamodb.svg){: style="height:20px;width:20px"} Amazon DynamoDB                |
|          | API key rotation      | ![AWS Secrets Manager](styles/logo_amazon_secrets_manager.svg){: style="height:20px;width:20px"} AWS Secrets Manager |
|          | Rate limiting         | ![Amazon DynamoDB](styles/logo_amazon_dynamodb.svg){: style="height:20px;width:20px"} Amazon DynamoDB                |
|          | AWS Bedrock API keys  | ![AWS Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} AWS Bedrock                         |

---

## ✨ Release History

### v1.4.0 – Audio Enhancements & Model Compatibility

Expands audio capabilities with Mistral Voxtral support, speaker diarization, audio formats for chat completions, and introduces prompt caching TTL and model aliasing for better OpenAI compatibility.

#### 💬 Chat Completions

| Provider                                                                       | Endpoint/Feature                                                     | AWS Backend                                                                                                            |
|--------------------------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` audio format support                          | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` extended Bedrock finish reasons mapping       | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock                     |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | Prompt caching TTL support                                           | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - prompt caching    |

#### 🎙️ Speech & Audio

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

---

### v1.3.0 – Image Editing & Variation Support (with v1.3.1–v1.3.5 maintenance updates)

Adds support for OpenAI's image editing and variation endpoints, enabling image manipulation capabilities backed by Amazon Bedrock. Includes maintenance updates for content block handling, tool call validation, streaming fixes, and TTS optimization.

#### 🎨 Image Generation

| Provider                                                                       | Endpoint/Feature        | AWS Backend                                                                                                       |
|--------------------------------------------------------------------------------|-------------------------|-------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/images/edits`      | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/images/variations` | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |

#### 🎙️ Speech & Audio (v1.3.2)

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

#### 💬 Chat Completions

| Provider                                                                       | Endpoint/Feature                                                      | AWS Backend                                                                                                        |
|--------------------------------------------------------------------------------|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` `service_tier` parameter                       | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - service tiers |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` Bedrock-specific system tools (Nova grounding) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - system tools  |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` GPT5.2 API update (`reasoning_effort=xhigh`)   |                                                                                                                    |

#### 🛡️ Content Safety & Moderation

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

#### 💬 Chat Completions

| Provider                                                                       | Endpoint/Feature                                                    | AWS Backend                                                                                                         |
|--------------------------------------------------------------------------------|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | Prompt caching `/v1/chat/completions` `prompt_cache_key`            | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - prompt caching |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` GPT5.1 API update  (`reasoning_effort=none`) |                                                                                                                     |

#### 🧠 Embeddings

| Provider                                                                                      | Endpoint/Feature                          | AWS Backend                                                                                        |
|-----------------------------------------------------------------------------------------------|-------------------------------------------|----------------------------------------------------------------------------------------------------|
|                                                                                               | Intelligent S3 multimodal upload          | ![Amazon S3](styles/logo_amazon_s3.svg){: style="height:20px;width:20px"} Amazon S3                |
|                                                                                               | Intelligent Sync/async Bedrock invocation | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock |
| ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} **Amazon Nova** | Multimodal embeddings models              |                                                                                                    |
| ![Twelve Labs](styles/logo_twelvelabs.svg){: style="height:20px;width:20px"} **Twelve Labs**  | Marengo V3 models                         |                                                                                                    |

#### 🎯 Advanced Routing

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

#### 💬 Chat Completions

| Provider                                                                             | Endpoint/Feature                                   | AWS Backend                                                                                                            |
|--------------------------------------------------------------------------------------|----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**       | `/v1/chat/completions`                             | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
|                                                                                      | All models supporting Converse/ConverseStream APIs | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API      |
| ![Deepseek](styles/logo_deepSeek.svg){: style="height:20px;width:20px"} **Deepseek** | `/v1/chat/completions` `reasoning_content`         | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Qwen](styles/logo_qwen.svg){: style="height:20px;width:20px"} **Qwen**             | `enable_thinking` + `thinking_budget` parameter    | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Qwen](styles/logo_qwen.svg){: style="height:20px;width:20px"} **Qwen**             | `top_k` parameter                                  | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |

#### 🧠 Embeddings

| Provider                                                                                     | Endpoint/Feature      | AWS Backend                                                                                                           |
|----------------------------------------------------------------------------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**               | `/v1/embeddings`      | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - embedding models |
| ![Cohere](styles/logo_cohere.svg){: style="height:20px;width:20px"} **Cohere**               | Embed V3 & V4  models |                                                                                                                       |
| ![Twelve Labs](styles/logo_twelvelabs.svg){: style="height:20px;width:20px"} **Twelve Labs** | Marengo V2  models    |                                                                                                                       |
| ![Amazon](styles/logo_amazon.svg){: style="height:20px;width:20px"} **Amazon Titan**         | Embed V1 & V2  models |                                                                                                                       |

#### 🎙️ Speech & Audio

| Provider                                                                       | Endpoint/Feature           | AWS Backend                                                                                                                    |
|--------------------------------------------------------------------------------|----------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/audio/speech`         | ![Amazon Polly](styles/logo_amazon_polly.svg){: style="height:20px;width:20px"} Amazon Polly + Amazon Comprehend               |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/audio/transcriptions` | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe                    |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/audio/translations`   | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe + Amazon Translate |

#### 🎨 Image Generation

| Provider                                                                                        | Endpoint/Feature                        | AWS Backend                                                                                                       |
|-------------------------------------------------------------------------------------------------|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**                  | `/v1/images/generations`                | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} **Amazon Nova**   | Canvas V1 models                        |                                                                                                                   |
| ![Amazon](styles/logo_amazon.svg){: style="height:20px;width:20px"} **Amazon Titan**            | Image Generator V1 & V2  models         |                                                                                                                   |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | Image Core, Ultra et SD3.5 Large models |                                                                                                                   |

#### 📋 Model Discovery

| Provider                                                                       | Endpoint/Feature | AWS Backend                                                                                                        |
|--------------------------------------------------------------------------------|------------------|--------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/models`     | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - model catalog |

#### Platform Features

| Feature                                     | AWS Backend                                                                                                                                                                                                                                 |
|---------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **🤖 Bedrock Features**                     |                                                                                                                                                                                                                                             |
| Content filtering and safety                | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock Guardrails                                                                                                                               |
| Cross-region inference                      | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - global/regional                                                                                                                        |
| Application inference profiles              | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - inference profiles                                                                                                                     |
| Model parameters (temperature, top_p, etc.) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - native parameters                                                                                                                      |
| Multi-region failover                       | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - multi-region                                                                                                                           |
| Bedrock guardrails                          | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock Guardrails                                                                                                                               |
| **☁️ AWS Services**                         |                                                                                                                                                                                                                                             |
| File storage                                | ![Amazon S3](styles/logo_amazon_s3.svg){: style="height:20px;width:20px"} Amazon S3 - presigned URLs, Transfer Acceleration                                                                                                                 |
| **🔐 Authentication**                       |                                                                                                                                                                                                                                             |
| Static token authentication                 | ![AWS Systems Manager](styles/logo_amazon_systems_manager.svg){: style="height:20px;width:20px"} AWS SSM Parameter Store / ![AWS Secrets Manager](styles/logo_amazon_secrets_manager.svg){: style="height:20px;width:20px"} Secrets Manager |
| Development mode (no auth)                  |                                                                                                                                                                                                                                             |
| **📊 Observability**                        |                                                                                                                                                                                                                                             |
| Distributed tracing                         | ![AWS X-Ray](styles/logo_amazon_xray.svg){: style="height:20px;width:20px"} AWS X-Ray + OpenTelemetry                                                                                                                                       |
| Structured logging                          | ![Amazon CloudWatch](styles/logo_amazon_cloudwatch.svg){: style="height:20px;width:20px"} Amazon CloudWatch (When running on ECS/EKS)                                                                                                       |
| ❤Health check endpoint                      |                                                                                                                                                                                                                                             |
| **🔒 HTTP/Security**                        |                                                                                                                                                                                                                                             |
| CORS support                                |                                                                                                                                                                                                                                             |
| Trusted host validation                     |                                                                                                                                                                                                                                             |
| Proxy headers (X-Forwarded-*)               |                                                                                                                                                                                                                                             |
| GZip compression                            |                                                                                                                                                                                                                                             |
| **📚 Documentation**                        |                                                                                                                                                                                                                                             |
| Interactive API docs & OpenAPI schema       |                                                                                                                                                                                                                                             |
| **🔌 Compatibility**                        |                                                                                                                                                                                                                                             |
| Provider-specific parameters                |                                                                                                                                                                                                                                             |
