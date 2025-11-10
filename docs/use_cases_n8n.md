# N8N Integration

Connect N8N automation workflows to Amazon Bedrock models through stdapi.ai's OpenAI-compatible interface. Use any existing OpenAI template from the N8N marketplace without modification—simply point it to your stdapi.ai instance.

## About N8N

**🔗 Links:** [Website](https://n8n.io/) | [GitHub](https://github.com/n8n-io/n8n) | [Documentation](https://docs.n8n.io/) | [Community](https://community.n8n.io/)

N8N is a workflow automation platform:

- 45,000+ GitHub stars - Open-source workflow automation tool
- 400+ integrations - Connect with services and APIs
- Visual workflow builder - No-code interface with customization
- Self-hosted or cloud - Deploy anywhere
- AI-native - Built-in nodes for OpenAI and other AI providers

## Why N8N + stdapi.ai?

!!! success "Full Compatibility"
    stdapi.ai is fully compatible with N8N's OpenAI nodes. Any workflow, template, or automation designed for OpenAI works with Amazon Bedrock models—no code changes required.

**Key Benefits:**

- Use familiar OpenAI nodes and templates
- Access Amazon Bedrock AI models
- AWS pricing and infrastructure
- Keep data in your AWS environment
- Mix different models in the same workflow

---

!!! warning "Work in Progress"
    This integration guide is actively being developed and refined. While the configuration examples are based on documented APIs and best practices, they are pending practical validation. Complete end-to-end deployment examples will be added once testing is finalized.

## Prerequisites

!!! info "What You'll Need"
    Before you begin, make sure you have:

    - ✓ A running N8N instance (cloud or self-hosted)
    - ✓ Your stdapi.ai server URL (e.g., `https://api.example.com`)
    - ✓ An API key (if authentication is enabled)
    - ✓ AWS access configured with the models you want to use

---

## Quick Start Guide

### Step 1: Set Up Your Credentials

The foundation of any N8N integration is configuring your API credentials. This one-time setup unlocks all AI capabilities.

!!! example "Creating Your stdapi.ai Credential"
    **In your N8N interface:**

    1. Navigate to **Credentials** in the left sidebar
    2. Click **New Credential** (or **Add Credential**)
    3. Search for **"OpenAI API"** in the credential list
    4. Configure the following fields:

    ```yaml
    Credential Name: stdapi.ai (or any name you prefer)
    API Key:         YOUR_STDAPI_KEY
    Base URL:        https://YOUR_SERVER_URL/v1
    ```

    5. Click **Save** to store your credential

!!! tip "What This Does"
    By setting a custom Base URL, you redirect all OpenAI API calls to your stdapi.ai instance. N8N will use this credential to authenticate and route requests to Amazon Bedrock models instead of OpenAI's servers.

---

## AI Capabilities: Step-by-Step

Now that your credentials are configured, let's explore each AI capability you can use in your N8N workflows.

### 💬 Chat Completions — Conversational AI

Build intelligent conversational workflows with state-of-the-art language models.

!!! example "Node Configuration"
    **Node:** OpenAI Chat Model

    **Settings:**

    | Parameter | Value | Description |
    |-----------|-------|-------------|
    | **Credential** | `stdapi.ai` | The credential you created in Step 1 |
    | **Model** | Any Bedrock model | Enter the model ID for your chosen model |
    | **Prompt** | `{{ $json.input }}` | Dynamic content from previous nodes |

**Available Models:**

All Amazon Bedrock chat models are accessible through N8N once you configure the stdapi.ai credential. This includes:

- **Anthropic Claude** — All Claude model variants (Sonnet, Haiku, Opus)
- **Amazon Nova** — All Nova family models (Pro, Lite, Micro)
- **Meta Llama** — Llama 3 models (if enabled in your AWS region)
- **Mistral AI** — Mistral and Mixtral models (if enabled in your AWS region)
- **And more** — Any other Bedrock models available in your configured AWS region

**Example Model IDs:**

- `anthropic.claude-sonnet-4-5-20250929-v1:0` — Superior reasoning and code
- `amazon.nova-pro-v1:0` — Advanced reasoning
- `amazon.nova-lite-v1:0` — Balanced performance
- `amazon.nova-micro-v1:0` — Fast, cost-effective responses

**💡 Real-World Examples:**

- **Customer Support:** Automate responses to common customer inquiries with context-aware AI that understands your knowledge base
- **Content Generation:** Generate blog posts, product descriptions, email campaigns, and social media content at scale
- **Data Analysis:** Extract insights from text data, classify documents, perform sentiment analysis on feedback
- **Code Assistant:** Generate, review, and debug code as part of your development workflows

---

### 🔊 Text to Speech — Give Your Workflows a Voice

Convert text into natural-sounding audio with Amazon Polly's lifelike voices.

!!! example "Node Configuration"
    **Node:** OpenAI (Audio → Create Speech)

    **Settings:**

    | Parameter | Value | Description |
    |-----------|-------|-------------|
    | **Resource** | `Audio` | Select the audio resource |
    | **Operation** | `Create Speech` | Generate audio from text |
    | **Credential** | `stdapi.ai` | Your configured credential |
    | **Model** | `amazon.polly-standard` | High-quality standard voices |
    | | `amazon.polly-neural` | Ultra-realistic neural voices |
    | | `amazon.polly-generative` | Most natural AI-generated voices |
    | | `amazon.polly-long-form` | Optimized for long content |
    | **Voice** | `Joanna`, `Matthew`, `Salli` | Choose from 60+ voices in 30+ languages |
    | **Input Text** | `{{ $json.message }}` | The text to synthesize |

**💡 Real-World Examples:**

- **Notifications:** Send audio alerts and notifications via phone, smart speakers, or mobile apps
- **Content Creation:** Convert blog posts to podcasts, generate audio versions of articles automatically
- **Accessibility:** Make your content accessible by providing audio alternatives to text
- **IVR Systems:** Generate dynamic voice responses for interactive voice response systems

!!! tip "Voice Configuration"
    **For Multi-Language Support:**

    Use any OpenAI-compatible voice name (e.g., `alloy`, `echo`, `nova`) in the Voice parameter. stdapi.ai will automatically detect the language of your text and select an appropriate Amazon Polly voice.

    **For Single-Language Use:**

    If your content is always in the same language, specify a particular Amazon Polly voice for consistent results:

    - **English (US):** Joanna, Matthew, Salli, Joey, Kendra, Kimberly
    - **English (UK):** Emma, Brian, Amy
    - **Spanish:** Lupe, Conchita, Miguel
    - **French:** Celine, Mathieu
    - **German:** Marlene, Hans
    - **And 50+ more voices** in 30+ languages

---

### 🎤 Speech to Text — Transcribe Audio Effortlessly

Transform audio recordings into accurate text transcriptions powered by Amazon Transcribe.

!!! example "Node Configuration"
    **Node:** OpenAI (Audio → Transcribe)

    **Settings:**

    | Parameter | Value | Description |
    |-----------|-------|-------------|
    | **Resource** | `Audio` | Select the audio resource |
    | **Operation** | `Transcribe` | Convert speech to text |
    | **Credential** | `stdapi.ai` | Your configured credential |
    | **Model** | `amazon.transcribe` | Amazon's transcription service |
    | **Audio File** | Binary data or file path | Supports WAV, MP3, FLAC, OGG |
    | **Language** | `en-US`, `es-ES`, etc. | Optional: improves accuracy |

**💡 Real-World Examples:**

- **Meeting Notes:** Automatically transcribe meetings, calls, and interviews for easy review and sharing
- **Customer Service:** Convert voicemails and support calls to text for analysis and ticketing systems
- **Media Production:** Generate subtitles and captions for videos, podcasts, and webinars
- **Voice Commands:** Build voice-controlled automation workflows and smart assistants

!!! tip "Language Support"
    Amazon Transcribe supports 35+ languages. Specifying the language code improves accuracy, especially for non-English audio.

---

### 🔍 Embeddings — Semantic Search & Intelligence

Generate vector embeddings to power semantic search, recommendations, and RAG (Retrieval-Augmented Generation) systems.

!!! example "Node Configuration"
    **Node:** Embeddings OpenAI

    **Settings:**

    | Parameter | Value | Description |
    |-----------|-------|-------------|
    | **Credential** | `stdapi.ai` | Your configured credential |
    | **Model** | Any embedding model | Enter your chosen embedding model ID |
    | **Input Text** | `{{ $json.document }}` | Text to convert to vectors |

**Available Embedding Models:**

The example uses `amazon.titan-embed-text-v2:0`, but you can choose from multiple embedding models available in Amazon Bedrock:

- **Amazon Titan Embed Text v2** — `amazon.titan-embed-text-v2:0` (recommended, optimized for RAG, 8192 dimensions)
- **Amazon Titan Embed Text v1** — `amazon.titan-embed-text-v1` (legacy, 1536 dimensions)
- **Cohere Embed** — Cohere embedding models (if enabled in your AWS region)
- **Other providers** — Any Bedrock embedding model

Choose your embedding model based on:

- **Dimension size** — Higher dimensions (like Titan v2's 8192) capture more nuance but use more storage
- **Performance** — Some models are faster than others
- **Language support** — Ensure your chosen model supports your documents' languages

!!! warning "Consistency Important"
    Once you start building workflows with embeddings, stick with the same model. Changing models requires regenerating all embeddings as vectors from different models are not compatible.

**💡 Real-World Examples:**

- **Semantic Search:** Find documents based on meaning rather than exact keyword matches—understand intent, not just words
- **Document Similarity:** Compare and cluster documents by semantic similarity for content organization
- **Knowledge Bases:** Build RAG systems that retrieve relevant context for AI-powered Q&A
- **Recommendations:** Create recommendation engines based on content similarity and user preferences

!!! info "What Are Embeddings?"
    Embeddings convert text into numerical vectors that capture semantic meaning. Similar concepts have similar vectors, enabling AI to understand relationships between documents, even when they use different words.

---

### 🎨 Image Generation — Visual Content Creation

Create stunning images from text descriptions using Amazon Bedrock's image generation models.

!!! example "Node Configuration"
    **Node:** OpenAI (Image → Generate)

    **Settings:**

    | Parameter | Value | Description |
    |-----------|-------|-------------|
    | **Resource** | `Image` | Select the image resource |
    | **Operation** | `Generate` | Create image from prompt |
    | **Credential** | `stdapi.ai` | Your configured credential |
    | **Model** | Any image model | Enter your chosen image model ID |
    | **Prompt** | Descriptive text | Describe the image you want |
    | **Size** | Model-dependent | Specify dimensions if supported |

**Available Image Models:**

While the example below uses `amazon.nova-canvas-v1:0`, you can use any image generation model available in Amazon Bedrock:

- **Amazon Nova Canvas** — `amazon.nova-canvas-v1:0` (recommended, fast and high-quality)
- **Stability AI** — Stable Diffusion models (if enabled in your AWS region)
- **Other providers** — Any Bedrock image generation model

Simply specify your preferred model ID in the Model parameter.

**💡 Real-World Examples:**

- **Marketing Assets:** Generate hero images, banners, and visual content for campaigns automatically
- **Product Mockups:** Create product visualizations and concept art for rapid prototyping
- **Social Media:** Produce eye-catching graphics for posts, stories, and advertisements at scale
- **Blog Illustrations:** Generate custom illustrations that match your article content perfectly

!!! tip "Prompt Engineering"
    Write detailed prompts for best results. Include style (e.g., "photorealistic," "watercolor"), subject, lighting, and composition details.

---

## 🔄 Using Existing OpenAI Templates

The best part? Any N8N workflow or template from the marketplace designed for OpenAI works with stdapi.ai—often with zero modifications needed.

### Migration in 4 Simple Steps

!!! success "Quick Migration Guide"
    **Step 1:** Import the OpenAI template into your N8N instance

    **Step 2:** Update all OpenAI nodes to use your stdapi.ai credential

    **Step 3:** Replace model names with Amazon Bedrock equivalents (see table below)

    **Step 4:** Test and deploy—you're done!

### 📋 Model Conversion Reference

Use this quick reference to map OpenAI models to Amazon Bedrock equivalents. **These are example suggestions**—many more models are available depending on your AWS region.

| OpenAI Model | Example stdapi.ai Replacement | Use Case |
|--------------|-------------------------------|----------|
| `gpt-4` | `anthropic.claude-sonnet-4-5-20250929-v1:0` | Complex reasoning, code |
| | `amazon.nova-pro-v1:0` | Advanced analysis |
| | *Any Claude or advanced model* | Based on your needs |
| `gpt-3.5-turbo` | `amazon.nova-micro-v1:0` | Fast, simple tasks |
| | `amazon.nova-lite-v1:0` | Balanced performance |
| | *Any efficient model* | Based on your needs |
| `whisper-1` | `amazon.transcribe` | Speech to text |
| `tts-1` | `amazon.polly-standard` | Text to speech |
| `tts-1-hd` | `amazon.polly-neural` or `amazon.polly-generative` | High-quality TTS |
| `text-embedding-ada-002` | `amazon.titan-embed-text-v2:0` | Embeddings |
| | *Any Bedrock embedding model* | Based on dimensions/language needs |
| `dall-e-3` | `amazon.nova-canvas-v1:0` | Image generation |
| | *Any Bedrock image model* | Based on style/quality needs |

!!! tip "Model Selection Tips"
    These are popular starting points, but experiment to find what works best:

    - **Speed + Cost** → Amazon Nova Micro, or similar fast models
    - **Balanced** → Amazon Nova Lite, Llama 3, or similar mid-tier models
    - **Quality + Reasoning** → Claude Sonnet, Nova Pro, or other premium models
    - **Specialized Tasks** → Explore model-specific strengths (e.g., coding, multilingual, long context)

### 🎯 Popular Template Categories

stdapi.ai is compatible with N8N templates across these categories:

!!! note "Customer Support & Service"
    - AI-powered chatbots and helpdesk automation
    - Ticket classification and routing
    - Sentiment analysis on customer feedback
    - Automated response generation

!!! note "Content & Marketing"
    - Blog post and article generation
    - Social media content automation
    - Email campaign personalization
    - SEO content optimization
    - Ad copy generation and A/B testing

!!! note "Data & Analytics"
    - Text classification and categorization
    - Entity extraction and data enrichment
    - Report generation and summarization
    - Trend analysis from text data

!!! note "Voice & Audio"
    - Meeting transcription pipelines
    - Voice notification systems
    - Podcast production workflows
    - IVR and telephony integration

!!! note "Knowledge & Intelligence"
    - Document Q&A systems
    - Knowledge base search and retrieval
    - Intelligent document processing
    - RAG (Retrieval-Augmented Generation)

!!! note "Creative Automation"
    - Image generation for marketing
    - Visual content for social media
    - Product mockup automation
    - Brand asset creation

---

## 🚀 Advanced Workflows

### Multi-Model Orchestration

One of the most powerful features is combining multiple models in a single workflow. Each node can use a different model optimized for its specific task.

!!! example "Smart Content Pipeline"
    ```yaml
    1. Webhook Trigger → Receives article topic
    2. Chat Node (amazon.nova-micro-v1:0) → Generate outline quickly
    3. Chat Node (anthropic.claude-sonnet-4-5-20250929-v1:0) → Write detailed article
    4. Embeddings Node (amazon.titan-embed-text-v2:0) → Create searchable vectors
    5. Image Node (amazon.nova-canvas-v1:0) → Generate hero image
    6. TTS Node (amazon.polly-neural) → Create audio version
    ```

    **Why this works:** Each model excels at its specific task—Nova Micro for speed, Claude Sonnet for quality writing, Titan for embeddings, Nova Canvas for images, and Polly for voice.

### Self-Hosted N8N Configuration

For self-hosted installations, streamline your setup with environment variables:

!!! example "Docker Compose Configuration"
    ```yaml
    version: '3.8'
    services:
      n8n:
        image: n8nio/n8n
        environment:
          # stdapi.ai defaults
          - N8N_OPENAI_BASE_URL=https://YOUR_SERVER_URL/v1
          - N8N_OPENAI_API_KEY=${STDAPI_KEY}
          # Other N8N settings
          - N8N_BASIC_AUTH_ACTIVE=true
          - N8N_BASIC_AUTH_USER=admin
          - N8N_BASIC_AUTH_PASSWORD=${ADMIN_PASSWORD}
        ports:
          - "5678:5678"
        volumes:
          - n8n_data:/home/node/.n8n
    ```

!!! tip "Environment Variables"
    Setting defaults via environment variables means new workflows automatically use stdapi.ai without manual credential configuration.

### Robust Error Handling

Build production-ready workflows with proper error handling and resilience.

!!! example "Error Handling Strategy"
    **1. Error Workflows**

    Create a dedicated error workflow that logs failures, sends notifications, and attempts recovery.

    **2. Retry Logic**

    Use N8N's built-in retry settings for transient errors:

    - **Max Retries:** 3
    - **Wait Between Retries:** 1000ms (exponential backoff)
    - **Continue on Fail:** Enable for non-critical nodes

    **3. Fallback Models**

    Implement model fallbacks for critical paths:
    ```
    Primary: anthropic.claude-sonnet-4-5-20250929-v1:0
    Fallback: amazon.nova-pro-v1:0
    Final Fallback: amazon.nova-lite-v1:0
    ```

    **4. Timeout Configuration**

    Set appropriate timeouts based on operation type:

    - Chat: 30-60 seconds
    - Image generation: 60-120 seconds
    - Transcription: 60-180 seconds (depends on audio length)
    - Embeddings: 10-30 seconds

!!! warning "Rate Limits & Quotas"
    Amazon Bedrock has model-specific rate limits and quotas. Monitor your usage and implement backoff strategies for high-volume workflows. Consider requesting quota increases for production deployments.

---

## 📊 Monitoring & Optimization

### Performance Best Practices

!!! success "Optimization Checklist"
    ✅ **Start Small:** Test workflows with limited data before scaling to production

    ✅ **Right-Size Models:** Use smaller models (Nova Micro/Lite) for simple tasks—save premium models for complex reasoning

    ✅ **Cache Intelligently:** Store embeddings and frequently used results to avoid redundant API calls

    ✅ **Batch Operations:** Process multiple items in parallel when possible to reduce total execution time

    ✅ **Monitor Token Usage:** Track token consumption in high-volume workflows to optimize costs

    ✅ **Set Execution Limits:** Configure max execution time and memory limits to prevent runaway workflows

### Cost Management

!!! tip "Reduce Your AI Costs"
    **Model Selection:** Amazon Nova Micro costs significantly less than Claude Sonnet. Use it for simple tasks.

    **Prompt Optimization:** Shorter, well-crafted prompts use fewer tokens while maintaining quality.

    **Caching Strategy:** Cache embeddings, common responses, and generated images to avoid regeneration.

    **Workflow Efficiency:** Eliminate unnecessary AI calls—use conditional logic to skip processing when results are cached.

---

## 🎓 Next Steps & Resources

### What You Can Build Now

With your stdapi.ai + N8N integration complete, you're ready to:

- ✨ **Import Templates:** Browse the N8N marketplace for OpenAI templates and adapt them instantly
- 🔨 **Build Custom Workflows:** Combine multiple AI capabilities for unique automation solutions
- 🔗 **Chain Complex Logic:** Create sophisticated multi-step AI pipelines with conditional branches
- 📈 **Scale Production:** Deploy high-volume workflows with confidence using enterprise-grade infrastructure
- 🌍 **Go Global:** Leverage multiple AWS regions for optimal performance worldwide

### Helpful Resources

!!! info "Learn More"
    - **[API Overview](api_overview.md)** — Complete list of available models and capabilities
    - **[Chat Completions API](api_openai_chat_completions.md)** — Detailed chat API documentation
    - **[Audio APIs](api_openai_audio_speech.md)** — TTS and STT implementation details
    - **[Configuration Guide](operations_configuration.md)** — Advanced stdapi.ai configuration options

### Get Support

!!! question "Need Help?"
    - 💬 Check the N8N community forums for workflow examples
    - 📖 Review Amazon Bedrock documentation for model-specific details
    - 🐛 Report issues on the [GitHub repository](https://github.com/stdapi-ai/stdapi.ai)
    - 🔧 Consult AWS Support for infrastructure and model access questions

---

## ⚠️ Important Considerations

!!! warning "Model Availability"
    **Regional Differences:** Not all Amazon Bedrock models are available in every AWS region. Before building workflows that depend on specific models, verify availability in your configured region.

    **Check your models:** See the [API Overview](api_overview.md) for a complete list of supported models by region.

!!! info "API Compatibility"
    **OpenAI Compatibility:** stdapi.ai implements the OpenAI API specification. Most OpenAI features work seamlessly, but some advanced features (like function calling) may have implementation differences.

    **Test thoroughly:** Always test workflows in a development environment before production deployment.

!!! tip "Security Best Practices"
    - 🔐 **Secure Your API Keys:** Use N8N's credential manager—never hardcode keys in workflows
    - 🛡️ **Enable Authentication:** Always require authentication on your stdapi.ai instance
    - 🔒 **HTTPS Only:** Use HTTPS for all API communications to protect data in transit
    - 📝 **Audit Logs:** Monitor API usage through AWS CloudTrail and N8N execution logs
    - 👥 **Least Privilege:** Grant minimal necessary permissions to API keys and service accounts
