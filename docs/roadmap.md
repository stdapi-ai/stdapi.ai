# Roadmap & Changelog

## What's Next

The following features may be implemented in future releases based on community demand and feedback. Implementation priority is determined by user requests, use case requirements, and alignment with the project's goals. All features can be implemented using AWS services as backends. If you need a specific feature, submit feedback or contribute to discussions.

### 💬 Chat Completions

| Provider                                                                                 | Endpoint/Feature                             | AWS Backend                                                                                                            |
|------------------------------------------------------------------------------------------|----------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**           | `/v1/completions`                            | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**           | `/v1/responses`                              | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Claude](styles/logo_anthropic_claude.svg){: style="height:20px;width:20px"} **Claude** | `/v1/messages`                               | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
|                                                                                          | Prompt caching                               | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - prompt caching    |
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
|                                                                                | Speaker diarization                   | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe - diarization  |
|                                                                                | Custom vocabularies                   | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe - custom vocab |

### 🎨 Image Generation

| Provider                                                                                        | Endpoint/Feature                        | AWS Backend                                                                                                       |
|-------------------------------------------------------------------------------------------------|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**                  | `/v1/images/edits`                      | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**                  | `/v1/images/variations`                 | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v1/generation/text-to-image`          | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v1/generation/image-to-image`         | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v1/generation/image-to-image/masking` | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v2/stable-image/generate`             | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v2/stable-image/upscale`              | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | `/v2/stable-image/edit`                 | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |

### 🎯 Model-Specific Features

| Provider | Endpoint/Feature                       | AWS Backend                                                                                                                         |
|----------|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
|          | Running application inference profiles | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - application inference profiles |
|          | Running Provisioned throughput         | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - provisioned models             |

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

### v1.1.0 – Embeddings Enhancement

Expands multimodal embedding capabilities.

### 🧠 Embeddings

| Provider                                                                                      | Endpoint/Feature                          | AWS Backend                                                                                        |
|-----------------------------------------------------------------------------------------------|-------------------------------------------|----------------------------------------------------------------------------------------------------|
|                                                                                               | Intelligent S3 multimodal upload          | ![Amazon S3](styles/logo_amazon_s3.svg){: style="height:20px;width:20px"} Amazon S3                |
|                                                                                               | Intelligent Sync/async Bedrock invocation | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock |
| ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} **Amazon Nova** | Multimodal embeddings models              |                                                                                                    |

---

### v1.0.0 – Foundation Release

The initial release establishes core OpenAI API compatibility with AWS Bedrock backing.

### 💬 Chat Completions

| Provider                                                                             | Endpoint/Feature                                   | AWS Backend                                                                                                            |
|--------------------------------------------------------------------------------------|----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**       | `/v1/chat/completions`                             | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
|                                                                                      | All models supporting Converse/ConverseStream APIs | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API      |
| ![Deepseek](styles/logo_deepSeek.svg){: style="height:20px;width:20px"} **Deepseek** | `/v1/chat/completions` `reasoning_content`         | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Qwen](styles/logo_qwen.svg){: style="height:20px;width:20px"} **Qwen**             | `enable_thinking` + `thinking_budget` parameter    | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Qwen](styles/logo_qwen.svg){: style="height:20px;width:20px"} **Qwen**             | `top_k` parameter                                  | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |

### 🧠 Embeddings

| Provider                                                                                     | Endpoint/Feature      | AWS Backend                                                                                                           |
|----------------------------------------------------------------------------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**               | `/v1/embeddings`      | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - embedding models |
| ![Cohere](styles/logo_cohere.svg){: style="height:20px;width:20px"} **Cohere**               | Embed V3 & V4  models |                                                                                                                       |
| ![Twelve Labs](styles/logo_twelvelabs.svg){: style="height:20px;width:20px"} **Twelve Labs** | Marengo V2  models    |                                                                                                                       |
| ![Amazon](styles/logo_amazon.svg){: style="height:20px;width:20px"} **Amazon Titan**         | Embed V1 & V2  models |                                                                                                                       |

### 🎙️ Speech & Audio

| Provider                                                                       | Endpoint/Feature           | AWS Backend                                                                                                                    |
|--------------------------------------------------------------------------------|----------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/audio/speech`         | ![Amazon Polly](styles/logo_amazon_polly.svg){: style="height:20px;width:20px"} Amazon Polly + Amazon Comprehend               |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/audio/transcriptions` | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe                    |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/audio/translations`   | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe + Amazon Translate |

### 🎨 Image Generation

| Provider                                                                                        | Endpoint/Feature                        | AWS Backend                                                                                                       |
|-------------------------------------------------------------------------------------------------|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**                  | `/v1/images/generations`                | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} **Amazon Nova**   | Canvas V1 models                        |                                                                                                                   |
| ![Amazon](styles/logo_amazon.svg){: style="height:20px;width:20px"} **Amazon Titan**            | Image Generator V1 & V2  models         |                                                                                                                   |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | Image Core, Ultra et SD3.5 Large models |                                                                                                                   |

### 📋 Model Discovery

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
