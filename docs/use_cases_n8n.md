---
title: n8n Integration - AWS Bedrock Workflow Automation
description: Connect n8n workflows to AWS Bedrock using stdapi.ai. Automate business processes with AI using 400+ integrations, no-code interface, and OpenAI-compatible or Anthropic-compatible nodes.
keywords: n8n AI integration, workflow automation AI, no-code AI automation, AWS Bedrock workflows, business process automation, AI workflow tools, n8n AWS Bedrock, OpenAI n8n, Anthropic n8n
---

# :material-sitemap: n8n Integration

Connect n8n automation workflows to AWS Bedrock models through stdapi.ai's OpenAI-compatible or Anthropic-compatible interfaces. Existing OpenAI and Anthropic templates from the n8n marketplace work out of the box—simply point them to your stdapi.ai instance and access AWS Bedrock models.

## :material-information-outline: About n8n

**🔗 Links:** [Website](https://n8n.io/) | [GitHub](https://github.com/n8n-io/n8n) | [Documentation](https://docs.n8n.io/)

n8n is a powerful workflow automation platform that enables you to connect any app with an API to build intelligent automations. With its intuitive visual interface, you can create complex AI-powered workflows without writing code, connecting AWS Bedrock models to 400+ services including Slack, Google Sheets, Salesforce, and more.

**Key Features:**

- ⭐ **50,000+ GitHub stars** - Leading open-source workflow automation platform
- **400+ integrations** - Pre-built nodes for popular services and APIs
- **Visual no-code builder** - Drag-and-drop interface with JavaScript customization
- **Self-hosted or cloud** - Deploy on your infrastructure or use n8n Cloud
- **AI-native platform** - Built-in OpenAI nodes work instantly with AWS Bedrock via stdapi.ai
- **Template marketplace** - Thousands of pre-built workflows ready to use

## :material-help-circle-outline: Why n8n + stdapi.ai?

<div class="grid cards" markdown>

- :material-puzzle: __Use Existing OpenAI Templates__
  <br>stdapi.ai works with n8n's OpenAI nodes. Thousands of marketplace templates and workflows designed for OpenAI run on AWS Bedrock—zero modifications needed.

- :material-robot: __Use Existing Anthropic Templates__
  <br>stdapi.ai works with n8n's Anthropic nodes. Templates and workflows designed for Anthropic Claude run on AWS Bedrock—zero modifications needed.

- :material-aws: __Access AWS Bedrock Models__
  <br>Claude 4.6+, Nova 2, Llama 4, DeepSeek v3.2, Stable Diffusion, and 80+ models available in n8n workflows. Switch models without changing automation logic.

- :material-graph-outline: __Automate Business Processes__
  <br>Connect AWS Bedrock AI to Slack, Salesforce, Google Workspace, databases, and 400+ services. Build intelligent automation with no-code drag-and-drop.

- :material-lock: __Enterprise Data Control__
  <br>All AI processing stays in your AWS account. Self-host n8n and stdapi.ai for complete data sovereignty and compliance.

- :material-currency-usd-off: __Pay-Per-Use Pricing__
  <br>No OpenAI subscriptions or per-automation fees. Pay only AWS Bedrock rates for actual AI usage in your workflows.

</div>

```mermaid
%%{init: {'flowchart': {'htmlLabels': true}} }%%
flowchart LR
  n8n["<img src='../styles/logo_n8n.svg' style='height:64px;width:auto;vertical-align:middle;' /> n8n"] --> stdapi["<img src='../styles/logo.svg' style='height:64px;width:auto;vertical-align:middle;' /> stdapi.ai"]
  stdapi --> bedrock["<img src='../styles/logo_amazon_bedrock.svg' style='height:64px;width:auto;vertical-align:middle;' /> AWS Bedrock"]
  stdapi --> transcribe["<img src='../styles/logo_amazon_transcribe.svg' style='height:64px;width:auto;vertical-align:middle;' /> AWS Transcribe"]
  stdapi --> polly["<img src='../styles/logo_amazon_polly.svg' style='height:64px;width:auto;vertical-align:middle;' /> AWS Polly"]
```

## :material-check-circle: Prerequisites

!!! info "What You'll Need"
    - ✓ **stdapi.ai deployed** - [See deployment guide](operations_getting_started.md)
    - ✓ **Your stdapi.ai URL** - e.g., `https://api.example.com`
    - ✓ **Your API key** - From Terraform output or configuration
    - ✓ **n8n instance** - Self-hosted or [n8n Cloud](https://n8n.io/cloud/)

---

## :material-cog: Configuration

### ![OpenAI](styles/logo_openai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } OpenAI Nodes

#### :material-key: Set Up Your Credentials

The foundation of any n8n integration is configuring your API credentials. This one-time setup unlocks all AI capabilities.

!!! example "Creating Your stdapi.ai Credential"
    **In your n8n interface:**

    1. Navigate to **Credentials** menu
    2. Click **Create Credential**
    3. Search and select **"OpenAI"** in the credential list
    4. Configure the following fields:
        ```
        API Key:  YOUR_STDAPI_KEY
        Base URL: https://YOUR_STDAPI_URL/v1
        ```

!!! tip "What This Does"
    By setting a custom Base URL, you redirect all OpenAI API calls to your stdapi.ai instance. n8n will use this credential to authenticate and route requests to Amazon Bedrock models instead of OpenAI's servers.

#### :material-cog-outline: Configure Nodes

For each node, first select the credentials you previously created in the node parameters. Then, select the model you want to use. If you want to use a model that is not listed, you can enter its ID as an expression in the `Model` parameter.

#### :material-chat-outline: Chat Completions

Enables: Text generation and conversational AI in workflows.

!!! example "Supported Node"
    **`OpenAI Chat Model`**

    - Model can be selected directly in the `Model` parameter
    - ⚠️ **Important:** `Use Responses API` parameter must be **unchecked** (Responses API is not supported yet by stdapi.ai)

    n8n calls `POST /v1/chat/completions` (see [Chat Completions API](api_openai_chat_completions.md)), so the model must be a text/chat-capable model from the correct family.

#### :material-database: Embeddings

Enables: Vector embeddings for semantic search and RAG workflows.

!!! example "Supported Node"
    **`Embeddings OpenAI`**

    - Model can be selected directly in the `Model` parameter

    n8n calls `POST /v1/embeddings` (see [Embeddings API](api_openai_embeddings.md)), so the model must be an embeddings-capable model from the correct family.

#### :material-image: Image Generation

Enables: Text-to-image creation in workflows.

!!! example "Supported Node"
    **`OpenAI/Generate an image`**

    - Model ID can be entered as expression in the `Model` parameter

    n8n calls `POST /v1/images/generations` (see [Images Generations API](api_openai_images_generations.md)), so the model must be an image-generation model from the correct family.

#### :material-image-edit: Image Editing

Enables: Image transformation and editing in workflows.

!!! example "Supported Node"
    **`OpenAI/Edit image`**

    - Model ID can be entered as expression in the `Model` parameter

    n8n calls `POST /v1/images/edits` (see [Images Edits API](api_openai_images_edits.md)), so the model must be an image-editing model from the correct family.

#### :material-volume-high: Audio Generation (TTS)

Enables: Text-to-speech audio generation in workflows.

!!! example "Supported Node"
    **`OpenAI/Generate audio`**

    - Model ID can be entered as expression in the `Model` parameter
    - **Or use OpenAI model names directly:** `tts-1` and `tts-1-hd` work by default thanks to built-in model aliases

    n8n calls `POST /v1/audio/speech` (see [Audio Speech API](api_openai_audio_speech.md)), so the model must match the text-to-speech modality and family.

#### :material-microphone: Audio Transcription (STT)

Enables: Speech-to-text transcription in workflows.

!!! example "Supported Node"
    **`OpenAI/Transcribe a recording`**

    - Works out of the box with OpenAI's `whisper-1` model name
    - The model alias automatically maps to `amazon.transcribe`

    n8n calls `POST /v1/audio/transcriptions` (see [Audio Transcriptions API](api_openai_audio_transcriptions.md)), so the model must match the speech-to-text modality.

#### :material-alert-outline: Unsupported Nodes

The following nodes are not yet supported:

!!! warning "Known Limitations"
    - **`OpenAI/Message a model`** — Requires Responses API (not supported yet by stdapi.ai). Use `OpenAI Chat Model` instead.
    - **`OpenAI/Analyze image`** — Requires Responses API (not supported yet by stdapi.ai). Use `OpenAI Chat Model` instead.

---

### ![Anthropic](styles/logo_anthropic.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Anthropic Nodes

#### :material-key: Set Up Your Credentials

!!! example "Creating Your stdapi.ai Anthropic Credential"
    **In your n8n interface:**

    1. Navigate to **Credentials** menu
    2. Click **Create Credential**
    3. Search and select **"Anthropic"** in the credential list
    4. Configure the following fields:
        ```
        API Key:  YOUR_STDAPI_KEY
        Base URL: https://YOUR_STDAPI_URL/anthropic
        ```

!!! tip "Anthropic Base URL"
    By default, all Anthropic-compatible routes are prefixed with `/anthropic`, so the Base URL must end with `/anthropic`. You can customize this prefix using the `ANTHROPIC_ROUTES_PREFIX` configuration variable documented in [Operations Configuration](operations_configuration.md#anthropic-routes-prefix).

#### :material-cog-outline: Configure Nodes

For each node, first select the credentials you previously created in the node parameters. Then, select the model you want to use. The model can be selected directly in the `Model` parameter for all supported nodes.

#### :material-chat-outline: Chat Completions

Enables: Text generation and conversational AI in workflows.

!!! example "Supported Nodes"
    **`Anthropic Chat Model`**

    - Model can be selected directly in the `Model` parameter

    ---

    **`Anthropic/Message a model`**

    - Model can be selected directly in the `Model` parameter

    ---

    **`Message a model in Anthropic`**

    - Model can be selected directly in the `Model` parameter

    n8n calls `POST /anthropic/v1/messages` (see [Anthropic Messages API](api_anthropic_messages.md)), so the model must be a text/chat-capable model from the correct family.

#### :material-image-search: Image Analysis

Enables: Image understanding and analysis in workflows.

!!! example "Supported Node"
    **`Anthropic/Analyze image`**

    - Model can be selected directly in the `Model` parameter

    n8n calls `POST /anthropic/v1/messages` with image content (see [Anthropic Messages API](api_anthropic_messages.md)), so the model must support vision capabilities.

#### :material-file-document: Document Analysis

Enables: Document understanding and extraction in workflows.

!!! example "Supported Node"
    **`Anthropic/Analyze document`**

    - Model can be selected directly in the `Model` parameter

    n8n calls `POST /anthropic/v1/messages` with document content (see [Anthropic Messages API](api_anthropic_messages.md)), so the model must support document processing capabilities.

## :material-arrow-right: Next Steps

<div class="grid cards" markdown>

- :material-rocket-launch: [**Getting Started**](operations_getting_started.md) — Deploy stdapi.ai to AWS with Terraform
- :material-docker: [**Local Development**](operations_getting_started_local.md) — Run stdapi.ai locally with Docker
- :material-puzzle: [**More Use Cases**](use_cases.md) — Explore other integrations and tools
- :material-api: [**API Overview**](api_overview.md) — Explore supported endpoints

</div>
