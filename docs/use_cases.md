---
title: Use Cases - Amazon Bedrock Integration Examples
description: Integrate Amazon Bedrock with Claude Code, Open WebUI, n8n, voice agents, RAG pipelines, and other OpenAI, Anthropic, and Cohere-compatible tools. Step-by-step guides for coding assistants, chat interfaces, workflow automation, and more.
keywords: AWS Bedrock integration, Open WebUI AWS, ChatGPT alternative, Claude alternative, AI coding assistant AWS, n8n AI workflow, private ChatGPT, private Claude, AI automation tools, voice agent AWS, RAG AWS Bedrock, OpenAI integration examples, Anthropic integration examples
---

# :material-puzzle: Use Cases

Discover how to integrate stdapi.ai with popular AI applications and tools. stdapi.ai's OpenAI, Anthropic, and Cohere-compatible APIs make it a drop-in replacement in hundreds of applications and tools, giving you access to Amazon Bedrock models with zero code changes.

**Why use stdapi.ai for integrations?**

- **No code changes required** - Just update the API endpoint in your application settings
- **Access 100+ models** - Claude, OpenAI GPT, xAI Grok, Kimi, DeepSeek, Qwen, GLM, Nova, Llama, Stability AI, and more
- **Enterprise data control** - All processing stays in your AWS account
- **Pay-per-use pricing** - No subscriptions, pay only Amazon Bedrock rates for actual usage
- **AWS-native features** - Leverage prompt caching, reasoning modes, and guardrails through standard OpenAI, Anthropic, and Cohere APIs
- **Three-dialect API compatibility** - Use the OpenAI, Anthropic, or Cohere SDK with the same deployment

!!! tip "Try it before committing to anything"
    Run stdapi.ai on your laptop in minutes with the [free community Docker image](operations_getting_started_local.md) — or deploy to AWS with a [14-day free trial](operations_getting_started.md). Every integration below works identically against either.

## :material-lightning-bolt: How Integration Works

Every integration on this page follows the same three steps — there is no step four:

1. **Deploy** stdapi.ai — [on AWS](operations_getting_started.md) or [locally with Docker](operations_getting_started_local.md)
2. **Copy** your endpoint URL and API key
3. **Paste** them into your tool's AI-provider settings (or your SDK's `base_url`)

```python
from openai import OpenAI

client = OpenAI(base_url="https://your-endpoint/v1", api_key="YOUR_KEY")
# Everything else in your application stays exactly the same
```

## :material-view-grid: Choose Your Integration

Pick the category that matches your goal — categories marked :material-book-open-variant: have a dedicated step-by-step guide.

<div class="grid cards" markdown>

- :material-code-braces: **[AI Coding Assistants](#coding-assistants)** :material-book-open-variant:
  <br>Frontier coding models in your IDE and terminal
- :material-brain: **[Autonomous Agents](#autonomous-agents)** :material-book-open-variant:
  <br>Self-directed agents on infrastructure you control
- :material-chat: **[Chat Interfaces](#chat-interfaces)** :material-book-open-variant:
  <br>Private ChatGPT alternative for your organization
- :material-graph-outline: **[Workflow Automation](#workflow-automation)** :material-book-open-variant:
  <br>AI steps in business processes, no code required
- :material-microphone-message: **[Voice & Audio](#voice-audio)** :material-book-open-variant:
  <br>Voice agents, transcription, and subtitles
- :material-magnify: **[RAG & Semantic Search](#rag-semantic-search)** :material-book-open-variant:
  <br>Ground AI answers in your own documents
- :material-image-multiple: **[Content & Media Generation](#media-generation)**
  <br>Images and video for creative pipelines
- :material-note-text: **[Knowledge Management](#knowledge-management)**
  <br>Private AI inside your notes and research
- :material-robot: **[Team Chatbots](#team-chatbots)**
  <br>Assistants in Slack, Discord, and Teams

</div>

### :material-code-braces: Developer Tools — AI Coding Assistants { #coding-assistants }

*For teams that want frontier coding models in their IDEs and terminals — without sending code to third-party AI clouds.*

Enhance your development workflow with AI-powered coding assistants. stdapi.ai integrates seamlessly with popular IDEs and AI development frameworks, allowing you to leverage Amazon Bedrock models (Claude, Kimi thinking, Qwen3 Coder Next) for code completion, generation, and intelligent assistance.

**What you can do:**

- **Code completion** - Real-time suggestions as you type in VS Code, JetBrains IDEs
- **Code generation** - Natural language to code with Claude and specialized coding models
- **Codebase understanding** - Chat with your codebase, explain functions, refactor code

**Popular tools:** Claude Code, Cline, OpenCode, Pi Agent, OpenClaw, Zed, JetBrains AI Assistant

**[AI Coding Assistants Guide](use_cases_coding_assistants.md)** — Universal setup for IDEs and development frameworks

---

### :material-brain: Autonomous Agents — Research & Task Automation { #autonomous-agents }

*For builders of agents that must run on infrastructure you control.*

Build self-directed AI agents that can plan, execute, and refine complex tasks autonomously. Integrate stdapi.ai with agent frameworks to create intelligent systems powered by Amazon Bedrock that can conduct research, automate workflows, and solve multi-step problems.

**What you can build:**

- **Personal AI assistants** - Autonomous agents connected to messaging, email, and smart home
- **Research agents** - Autonomous web research, data gathering, and analysis
- **Multi-agent systems** - Collaborative agents for complex problem-solving
- **Task automation** - Self-improving workflows that adapt to results
- **Code agents** - Autonomous development and testing systems

**Compatible frameworks:** OpenClaw, Hermes Agent, LangChain, LangGraph, LlamaIndex, CrewAI, OpenAI Agents SDK, Pydantic AI, Strands Agents

All agent frameworks that support OpenAI or Anthropic SDKs work immediately — point the SDK's base URL to stdapi.ai. See the [API overview](api_overview.md) for connection details.

**[Python Client Libraries Guide](use_cases_python_libraries.md)** — Configuring LangChain and pydantic-ai directly against stdapi.ai

**[Autonomous Agent CLIs Guide](use_cases_autonomous_agents.md)** — Configuring Hermes and OpenClaw directly against stdapi.ai

!!! tip "Give your agents AI capabilities via MCP"
    stdapi.ai is also a native [MCP server](api_overview.md#mcp-model-context-protocol): agents can call image generation, speech synthesis, transcription, file management, and model discovery as MCP tools — no custom integration code needed.

---

### :material-chat: Chat Interfaces — Private ChatGPT Alternative { #chat-interfaces }

*For organizations replacing per-seat AI subscriptions with a private, pay-per-use assistant.*

Build ChatGPT-like experiences with Amazon Bedrock models and complete privacy control. Deploy feature-rich web interfaces that provide familiar chat experiences while keeping all data within your AWS environment.

**What you can build:**

- **Private team chat** - ChatGPT-style interface for your organization
- **Customer support assistant** - AI-powered help desk with your data
- **Internal knowledge base** - RAG-enabled chat with document search
- **Multi-modal applications** - Process text, voice, images, and documents
- **Voice chat & image generation** - Speech input/output and in-chat image creation through the same endpoint

**Popular tools:** Open WebUI, LobeHub, AnythingLLM, LibreChat

**[Open WebUI Integration Guide](use_cases_openwebui.md)** — Complete setup with Terraform deployment examples

---

### :material-graph-outline: Workflow Automation — AI-Powered Business Processes { #workflow-automation }

*For teams adding AI steps to business processes without writing code.*

Integrate Amazon Bedrock AI into your business processes and automation workflows. Connect models to hundreds of services and APIs through visual workflow builders, enabling sophisticated AI-powered automation without writing code.

**What you can automate:**

- **Customer support** - Auto-classify tickets, generate responses, route intelligently
- **Content creation** - Automated blog posts, social media, email campaigns
- **Data processing** - Extract, transform, and analyze data with AI
- **Document workflows** - Automated summarization, translation, and classification
- **Content safety** - Screen user-generated content with the [Moderations API](api_openai_moderations.md)

**Popular tools:** n8n, Langflow, Dify, Flowise

!!! note "Make & Zapier"
    Make and Zapier can call stdapi.ai through their generic HTTP/webhook modules, but their native OpenAI modules do not support custom endpoints.

**[n8n Integration Guide](use_cases_n8n.md)** — Complete setup for AI workflow automation

---

### :material-microphone-message: Voice & Audio — Speech Applications & Voice Agents { #voice-audio }

*For products that need speech in and speech out — without adding a second AI vendor.*

Build voice-first applications on the same OpenAI-compatible endpoint: text-to-speech with Amazon Polly voices, speech-to-text with Amazon Transcribe and Bedrock audio models (including streaming and speaker diarization), and speech translation with subtitle output.

**What you can build:**

- **Voice agents** - Real-time conversational agents for phone, web, and support lines
- **Meeting intelligence** - Transcription with speaker diarization and AI summaries
- **Subtitles & dubbing** - Transcribe and translate audio with SRT/VTT subtitle output
- **Voice interfaces** - Add speech input/output to chat interfaces and internal tools

**Popular frameworks:** Pipecat, LiveKit Agents, TEN Framework — all accept a custom OpenAI-compatible base URL for LLM, speech-to-text, and text-to-speech services

**Popular tools:** Home Assistant Assist (via the wyoming-openai proxy)

**[Home Assistant Voice Guide](use_cases_home_assistant.md)** — Complete setup for local voice assistants backed by Amazon Transcribe and Amazon Polly

!!! tip "Getting started"
    Point the framework's OpenAI plugin at your stdapi.ai `/v1` URL. See the [Audio Speech](api_openai_audio_speech.md), [Audio Transcriptions](api_openai_audio_transcriptions.md), and [Audio Translations](api_openai_audio_translations.md) APIs for supported models and formats.

---

### :material-magnify: RAG & Semantic Search — Embeddings and Reranking { #rag-semantic-search }

*For teams grounding AI answers in their own documents and data.*

Build retrieval-augmented generation and semantic search pipelines with Bedrock embedding models and Cohere-compatible reranking — two-stage retrieval (embed, then rerank) through one deployment.

**What you can build:**

- **RAG pipelines** - Ground model answers in your documents with [embeddings](api_openai_embeddings.md)
- **Two-stage retrieval** - Improve relevance with the [Rerank API](api_cohere_rerank.md) on top of vector search
- **Semantic search** - Search by meaning across documents, tickets, and knowledge bases
- **Multimodal search** - Embed text and images with models like Cohere Embed v4

**Popular tools:** LlamaIndex, Haystack, RAGFlow, LightRAG — works with any vector database (pgvector, Qdrant, and others store the vectors; stdapi.ai serves the embeddings)

**[RAG Pipelines Guide](use_cases_rag.md)** — Configuring embeddings, Cohere-compatible reranking, and generation together

---

### :material-image-multiple: Content & Media Generation — Images and Video { #media-generation }

*For creative and marketing pipelines that generate visuals at scale.*

Generate and edit visual content with Amazon Bedrock media models through the standard OpenAI Images and Videos APIs — from marketing assets to fully automated content pipelines.

**What you can build:**

- **Image generation** - Text-to-image with Amazon Nova Canvas and Stability AI models via [Images Generations](api_openai_images_generations.md)
- **Image editing** - Inpainting, outpainting, and style transfer via [Images Edits](api_openai_images_edits.md)
- **Video generation** - Asynchronous text/image-to-video with Amazon Nova Reel and Luma Ray via the [Videos API](api_openai_videos.md)
- **Safe publishing pipelines** - Combine generation with the [Moderations API](api_openai_moderations.md) for automated content review

**Popular tools:** Open WebUI (built-in image generation), n8n media workflows, or the APIs directly

!!! tip "Getting started"
    Any tool that supports the OpenAI Images API works by pointing it at your stdapi.ai `/v1` URL. Video generation requires S3 storage — see the [Videos API](api_openai_videos.md) for setup.

---

### :material-note-text: Knowledge Management — AI-Enhanced Notes & Research { #knowledge-management }

*For individuals and teams adding private AI to their notes and research.*

Transform your knowledge base with AI-powered insights and generation. Integrate stdapi.ai with note-taking applications to add semantic search, writing assistance, and intelligent content organization.

**What you can do:**

- **AI writing assistance** - Generate, edit, and improve your writing
- **Semantic search** - Find notes by meaning, not just keywords
- **Auto-summarization** - Extract key points from long documents
- **Smart organization** - Automatic tagging, linking, and categorization

**Compatible tools:** Obsidian (Copilot plugin), Khoj (self-hosted), SiYuan

!!! tip "Getting started"
    These tools accept a custom OpenAI-compatible endpoint for both chat and embedding models. Point them to your stdapi.ai `/v1` URL. No dedicated guide yet — [see the API overview](api_overview.md) for connection details.

---

### :material-robot: Team Chatbots & Assistants — Slack, Discord, Teams Integration { #team-chatbots }

*For support and ops teams meeting users where they already chat.*

Deploy intelligent AI assistants to your team's communication platforms powered by Amazon Bedrock models.

**What you can build:**

- **Team Q&A bot** - Answer common questions instantly in Slack or Teams
- **Documentation assistant** - Search and cite internal docs in real-time
- **Task automation** - Create tickets, schedule meetings, update databases via chat
- **Moderated channels** - Screen messages with the [Moderations API](api_openai_moderations.md)

**Compatible platforms:** Dify, Chatwoot (Captain, self-hosted), Typebot — or build directly for Slack, Discord, and Microsoft Teams with the OpenAI or Anthropic SDK

!!! tip "Getting started"
    Build bots using the OpenAI or Anthropic SDK, pointing to your stdapi.ai endpoint. No dedicated guide yet — [see the API overview](api_overview.md) for connection details.

---

## :material-help-circle: Common Questions

- **Where does my data go?** Nowhere outside your AWS account — all inference runs in your account and is never shared with model providers or used for training. [Data sovereignty & compliance →](operations_compliance.md)
- **What does it cost?** $0.10/container-hour for the gateway plus Amazon Bedrock rates — no markup, no subscriptions, no per-seat fees. [Cost management →](operations_cost_management.md)
- **Am I locked in?** No — stdapi.ai speaks the standard OpenAI, Anthropic, and Cohere APIs. Leaving is the same one-line base-URL change that got you in.

## :material-arrow-right: Ready to Get Started?

<div class="grid cards" markdown>

- :material-rocket-launch: [**Deploy to AWS**](operations_getting_started.md) — Production-ready in 5 minutes with Terraform (14-day free trial)
- :material-docker: [**Try Locally with Docker**](operations_getting_started_local.md) — Free community image for development and testing
- :material-book-open-variant: [**API Overview**](api_overview.md) — Endpoints, parameters, and usage examples

</div>
