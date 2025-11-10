# OpenWebUI Integration

Connect OpenWebUI to stdapi.ai as an OpenAI-compatible backend. Access Amazon Bedrock models through OpenWebUI's chat interface with no code changes required.

## About OpenWebUI

**🔗 Links:** [Website](https://openwebui.com/) | [GitHub](https://github.com/open-webui/open-webui) | [Documentation](https://docs.openwebui.com/) | [Discord](https://discord.gg/5rJgQTnV4s)

OpenWebUI is a feature-rich, self-hosted web UI for AI models:

- 40,000+ GitHub stars - Popular open-source AI web interface
- Feature-complete - Matches ChatGPT's interface capabilities
- Multi-modal - Chat, voice input/output, image generation, and document RAG
- Extensible - Plugin system, custom functions, and community tools
- Privacy-focused - Self-hosted with no external dependencies

## Why OpenWebUI + stdapi.ai?

!!! success "Seamless Integration"
    stdapi.ai acts as a drop-in replacement for OpenAI's API. Simply configure your OpenWebUI environment variables, and you're ready to use Amazon Bedrock models through the familiar ChatGPT-style interface.

**Key Benefits:**

- Familiar interface - Continue using OpenWebUI as before
- Enterprise models - Access Anthropic Claude, Amazon Nova, and more
- Full feature set - Chat, voice, image generation, and RAG all work
- Privacy - Your data stays in your AWS environment
- Cost efficient - AWS pricing without OpenAI lock-in
- Multi-modal - Combine text, voice, images, and documents

---

!!! warning "Work in Progress"
    This integration guide is actively being developed and refined. While the configuration examples are based on documented APIs and best practices, they are pending practical validation. Complete end-to-end deployment examples will be added once testing is finalized.

## Prerequisites

!!! info "What You'll Need"
    Before you begin, make sure you have:

    - ✓ A running OpenWebUI instance (Docker, Kubernetes, or bare metal)
    - ✓ Your stdapi.ai server URL (e.g., `https://api.example.com`)
    - ✓ An API key (if authentication is enabled on your stdapi.ai deployment)
    - ✓ AWS access configured with the Bedrock models you want to use

---

## 🚀 Quick Start Configuration

OpenWebUI is configured entirely through environment variables. Below is a step-by-step guide to enable each AI capability.

### Step 1: Core Connection — Chat Completions

Start by establishing the base connection to stdapi.ai. This configuration is **required** and enables conversational AI features.

!!! example "Environment Variables"
    ```bash
    # Core OpenAI API connection
    OPENAI_API_BASE_URL=https://YOUR_SERVER_URL/v1
    OPENAI_API_KEY=YOUR_STDAPI_KEY

    # Default model for background tasks
    TASK_MODEL_EXTERNAL=amazon.nova-micro-v1:0
    ```

!!! info "What This Enables"
    **💬 Chat Interface:** Access to all Amazon Bedrock chat models through OpenWebUI's interface

    **🎯 Model Selection:** Choose from any available Bedrock model in the UI

    **🔧 Task Processing:** Background operations like title generation and task summarization

**Available Models:**

All Amazon Bedrock chat models are automatically available in OpenWebUI once you configure the connection. This includes:

- **Anthropic Claude** — All Claude model variants (Sonnet, Haiku, Opus)
- **Amazon Nova** — All Nova family models (Pro, Lite, Micro)
- **Meta Llama** — Llama 3 models (if enabled in your AWS region)
- **Mistral AI** — Mistral and Mixtral models (if enabled in your AWS region)
- **And more** — Any other Bedrock models available in your configured AWS region

!!! tip "Model Discovery"
    OpenWebUI will automatically discover and list all available models from your stdapi.ai instance. Just configure the base URL and API key—no need to manually add models.

---

### Step 2: Voice Input — Speech to Text

Enable hands-free interaction by allowing users to speak their queries instead of typing.

!!! example "Environment Variables"
    ```bash
    # Enable speech-to-text
    AUDIO_STT_ENGINE=openai
    AUDIO_STT_OPENAI_API_BASE_URL=https://YOUR_SERVER_URL/v1
    AUDIO_STT_MODEL=amazon.transcribe
    ```

!!! info "What This Enables"
    **🎤 Voice Input:** Click the microphone button to speak your messages

    **🌍 Multi-Language:** Supports 35+ languages with automatic detection

    **📝 Accurate Transcription:** Powered by Amazon Transcribe for high accuracy

**Use Cases:**

- **Accessibility:** Make your AI interface accessible to users with typing difficulties
- **Mobile Use:** Perfect for on-the-go interactions on smartphones and tablets
- **Multitasking:** Interact with AI while performing other tasks
- **Natural Interaction:** Speak naturally instead of formulating written queries

---

### Step 3: Voice Output — Text to Speech

Transform text responses into natural-sounding audio for a fully voice-enabled experience.

!!! example "Environment Variables"
    ```bash
    # Enable text-to-speech
    AUDIO_TTS_ENGINE=openai
    AUDIO_TTS_OPENAI_API_BASE_URL=https://YOUR_SERVER_URL/v1
    AUDIO_TTS_MODEL=amazon.polly-standard
    ```

!!! info "What This Enables"
    **🔊 Audio Responses:** Listen to AI responses instead of reading them

    **🎭 Multiple Voices:** Choose from 60+ voices in 30+ languages

    **⚙️ Voice Customization:** Select voices in OpenWebUI settings

**Voice Options:**

| Voice Type | Model ID | Quality | Use Case |
|------------|----------|---------|----------|
| **Standard** | `amazon.polly-standard` | High-quality | General use, cost-effective |
| **Neural** | `amazon.polly-neural` | Ultra-realistic | Premium experiences, podcasts |
| **Generative** | `amazon.polly-generative` | Most natural | Latest AI-generated voices |
| **Long-form** | `amazon.polly-long-form` | Optimized for articles | Long content, audiobooks |

!!! tip "Voice Configuration"
    **For Multi-Language Support:**

    Set the voice to any OpenAI-compatible voice name (e.g., `alloy`, `echo`, `nova`) in OpenWebUI's settings. stdapi.ai will automatically detect the language of the text and select an appropriate Amazon Polly voice.

    **For Single-Language Use:**

    If your content is always in the same language, you can specify a particular Amazon Polly voice for consistent results:

    - **English (US):** Joanna, Matthew, Salli, Joey, Kendra, Kimberly
    - **English (UK):** Emma, Brian, Amy
    - **Spanish:** Lupe, Conchita, Miguel
    - **French:** Celine, Mathieu
    - **German:** Marlene, Hans
    - **And 50+ more voices** in 30+ languages

    Configure specific voices in OpenWebUI's **Settings → Audio → Text-to-Speech → Voice**

**Use Cases:**

- **Accessibility:** Support visually impaired users with audio output
- **Learning:** Improve comprehension through audio and text together
- **Productivity:** Listen to responses while working on other tasks
- **Content Creation:** Generate audio content directly from AI responses

---

### Step 4: Visual Creativity — Image Generation

Bring creative ideas to life by generating images directly from text descriptions within your chat interface.

!!! example "Environment Variables"
    ```bash
    # Enable image generation
    ENABLE_IMAGE_GENERATION=True
    IMAGE_GENERATION_ENGINE=openai
    IMAGES_OPENAI_API_BASE_URL=https://YOUR_SERVER_URL/v1
    IMAGE_GENERATION_MODEL=amazon.nova-canvas-v1:0
    ```

!!! info "What This Enables"
    **🎨 In-Chat Image Creation:** Generate images directly in conversations

    **🖼️ Multiple Formats:** Support for various image sizes and styles

    **⚡ Fast Generation:** Quick turnaround for creative iterations

**Available Image Models:**

While the example shows `amazon.nova-canvas-v1:0`, you can use any image generation model available in Amazon Bedrock:

- **Amazon Nova Canvas** — `amazon.nova-canvas-v1:0` (recommended, fast and high-quality)
- **Stability AI** — Stable Diffusion models (if enabled in your AWS region)
- **Other providers** — Any Bedrock image generation model

Simply change the `IMAGE_GENERATION_MODEL` environment variable to your preferred model ID.

**How to Use:**

1. Type an image description in the chat
2. Ask the AI to generate an image (e.g., "Create an image of...")
3. The image appears directly in the conversation
4. Download or refine with follow-up prompts

**Example Prompts:**

- "Generate a photorealistic image of a modern office with plants and natural lighting"
- "Create a minimalist logo for a tech startup focused on sustainability"
- "Design a watercolor illustration of a sunset over mountains"
- "Make a futuristic concept art of a smart city with flying vehicles"

**Use Cases:**

- **Marketing Materials:** Generate visuals for campaigns and presentations
- **Prototyping:** Quick mockups for design concepts
- **Content Creation:** Illustrations for blog posts and articles
- **Education:** Visual aids for teaching and learning
- **Brainstorming:** Rapid visualization of ideas

---

### Step 5: Intelligent Documents — RAG Embeddings

Unlock the full power of Retrieval-Augmented Generation (RAG) to chat with your documents using semantic search.

!!! example "Environment Variables"
    ```bash
    # Enable RAG with embeddings
    RAG_EMBEDDING_ENGINE=openai
    RAG_OPENAI_API_BASE_URL=https://YOUR_SERVER_URL/v1
    RAG_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
    ```

!!! info "What This Enables"
    **📚 Document Chat:** Upload PDFs, docs, and text files to chat with them

    **🔍 Semantic Search:** Find information by meaning, not just keywords

    **🧠 Context-Aware Responses:** AI answers questions using your document content

    **📊 Knowledge Base:** Build a searchable knowledge base from your files

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
    Once you start building your knowledge base with a specific embedding model, stick with it. Changing models requires re-indexing all documents as embeddings from different models are not compatible.

**How It Works:**

1. **Upload Documents:** Add PDFs, Word docs, text files, or web content
2. **Automatic Indexing:** OpenWebUI creates vector embeddings using Amazon Titan
3. **Semantic Search:** Ask questions in natural language
4. **Contextual Answers:** AI retrieves relevant passages and generates accurate responses

**What You Can Do:**

- **Research:** Analyze multiple research papers and extract insights
- **Documentation:** Search technical documentation using natural language
- **Legal:** Review contracts and legal documents with AI assistance
- **Business Intelligence:** Query reports, presentations, and business documents
- **Learning:** Study textbooks and educational materials interactively

**Supported File Types:**

- PDF documents
- Microsoft Word (.docx)
- Plain text (.txt)
- Markdown (.md)
- Web pages (via URL)

!!! tip "RAG Best Practices"
    - **Chunk Size:** OpenWebUI automatically splits documents into optimal chunks
    - **Context Window:** Larger models (Nova Pro, Claude Sonnet) handle more context
    - **Multiple Documents:** Combine multiple sources for comprehensive answers
    - **Citation Tracking:** OpenWebUI shows which documents were used for answers

---

## 🔧 Deployment Methods

Choose the deployment method that best fits your infrastructure.

### Docker Compose (Recommended)

The easiest way to deploy OpenWebUI with stdapi.ai using Docker.

!!! example "Complete docker-compose.yml"
    ```yaml
    version: '3.8'

    services:
      openwebui:
        image: ghcr.io/open-webui/open-webui:main
        container_name: openwebui
        ports:
          - "3000:8080"
        environment:
          # Core stdapi.ai connection
          - OPENAI_API_BASE_URL=https://YOUR_SERVER_URL/v1
          - OPENAI_API_KEY=${STDAPI_KEY}
          - TASK_MODEL_EXTERNAL=amazon.nova-micro-v1:0

          # Speech to Text
          - AUDIO_STT_ENGINE=openai
          - AUDIO_STT_OPENAI_API_BASE_URL=https://YOUR_SERVER_URL/v1
          - AUDIO_STT_MODEL=amazon.transcribe

          # Text to Speech
          - AUDIO_TTS_ENGINE=openai
          - AUDIO_TTS_OPENAI_API_BASE_URL=https://YOUR_SERVER_URL/v1
          - AUDIO_TTS_MODEL=amazon.polly-standard

          # Image Generation
          - ENABLE_IMAGE_GENERATION=True
          - IMAGE_GENERATION_ENGINE=openai
          - IMAGES_OPENAI_API_BASE_URL=https://YOUR_SERVER_URL/v1
          - IMAGE_GENERATION_MODEL=amazon.nova-canvas-v1:0

          # RAG Embeddings
          - RAG_EMBEDDING_ENGINE=openai
          - RAG_OPENAI_API_BASE_URL=https://YOUR_SERVER_URL/v1
          - RAG_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0

        volumes:
          - openwebui_data:/app/backend/data
        restart: unless-stopped

    volumes:
      openwebui_data:
    ```

**Deploy:**
```bash
# Create .env file with your API key
echo "STDAPI_KEY=your_api_key_here" > .env

# Start OpenWebUI
docker-compose up -d

# View logs
docker-compose logs -f
```

### Kubernetes / Helm

Deploy OpenWebUI in Kubernetes with a ConfigMap for environment variables.

!!! example "Kubernetes ConfigMap"
    ```yaml
    apiVersion: v1
    kind: ConfigMap
    metadata:
      name: openwebui-config
    data:
      OPENAI_API_BASE_URL: "https://YOUR_SERVER_URL/v1"
      TASK_MODEL_EXTERNAL: "amazon.nova-micro-v1:0"
      AUDIO_STT_ENGINE: "openai"
      AUDIO_STT_OPENAI_API_BASE_URL: "https://YOUR_SERVER_URL/v1"
      AUDIO_STT_MODEL: "amazon.transcribe"
      AUDIO_TTS_ENGINE: "openai"
      AUDIO_TTS_OPENAI_API_BASE_URL: "https://YOUR_SERVER_URL/v1"
      AUDIO_TTS_MODEL: "amazon.polly-standard"
      ENABLE_IMAGE_GENERATION: "True"
      IMAGE_GENERATION_ENGINE: "openai"
      IMAGES_OPENAI_API_BASE_URL: "https://YOUR_SERVER_URL/v1"
      IMAGE_GENERATION_MODEL: "amazon.nova-canvas-v1:0"
      RAG_EMBEDDING_ENGINE: "openai"
      RAG_OPENAI_API_BASE_URL: "https://YOUR_SERVER_URL/v1"
      RAG_EMBEDDING_MODEL: "amazon.titan-embed-text-v2:0"
    ---
    apiVersion: v1
    kind: Secret
    metadata:
      name: openwebui-secret
    type: Opaque
    stringData:
      OPENAI_API_KEY: "your_stdapi_key_here"
    ```

### Environment File (.env)

For bare metal or systemd deployments, use an environment file.

!!! example ".env Configuration"
    ```bash
    # Core connection
    OPENAI_API_BASE_URL=https://YOUR_SERVER_URL/v1
    OPENAI_API_KEY=your_stdapi_key_here
    TASK_MODEL_EXTERNAL=amazon.nova-micro-v1:0

    # Speech to Text
    AUDIO_STT_ENGINE=openai
    AUDIO_STT_OPENAI_API_BASE_URL=https://YOUR_SERVER_URL/v1
    AUDIO_STT_MODEL=amazon.transcribe

    # Text to Speech
    AUDIO_TTS_ENGINE=openai
    AUDIO_TTS_OPENAI_API_BASE_URL=https://YOUR_SERVER_URL/v1
    AUDIO_TTS_MODEL=amazon.polly-standard

    # Image Generation
    ENABLE_IMAGE_GENERATION=True
    IMAGE_GENERATION_ENGINE=openai
    IMAGES_OPENAI_API_BASE_URL=https://YOUR_SERVER_URL/v1
    IMAGE_GENERATION_MODEL=amazon.nova-canvas-v1:0

    # RAG Embeddings
    RAG_EMBEDDING_ENGINE=openai
    RAG_OPENAI_API_BASE_URL=https://YOUR_SERVER_URL/v1
    RAG_EMBEDDING_MODEL=amazon.titan-embed-text-v2:0
    ```

---

## 🎯 What You Can Do Now

Once configured and running, your OpenWebUI + stdapi.ai integration unlocks powerful AI capabilities:

### 💬 Intelligent Conversations

- **Multi-Turn Dialogues:** Natural conversations with context awareness across messages
- **Model Switching:** Change models mid-conversation based on task complexity
- **System Prompts:** Customize AI behavior with custom system instructions
- **Chat History:** Full conversation history with search and export

### 🎤 Voice Interactions

- **Voice-to-Voice:** Speak your question and listen to the response—true hands-free AI
- **Language Flexibility:** Switch between languages seamlessly
- **Mobile Friendly:** Perfect experience on smartphones and tablets
- **Meeting Mode:** Use voice for brainstorming sessions and meetings

### 🎨 Creative Content

- **Image Generation:** Create visuals directly in your chat with natural language
- **Iterative Design:** Refine images with follow-up prompts
- **Style Control:** Specify artistic styles, lighting, composition, and mood
- **Download & Share:** Export generated images for use in your projects

### 📚 Document Intelligence

- **Knowledge Retrieval:** Upload documents and get instant answers from their content
- **Multi-Document Analysis:** Compare and synthesize information across multiple files
- **Contextual Citations:** See which documents informed each answer
- **Continuous Learning:** Build a growing knowledge base over time

### 🔧 Advanced Features

- **Function Calling:** Extend AI capabilities with custom functions and tools
- **Code Execution:** Run code directly in the chat interface
- **Web Search:** Enable real-time web search for current information
- **Plugins:** Extend functionality with OpenWebUI's plugin ecosystem

---

## 📊 Model Selection Guide

Choose the right model for your use case to optimize performance and cost. These are **suggestions only**—many more models are available through Amazon Bedrock depending on your AWS region configuration.

!!! info "Available Models"
    The models shown below are popular examples. **All Amazon Bedrock models** accessible through your stdapi.ai instance will automatically appear in OpenWebUI's model selector, including:

    - Anthropic Claude family (all versions)
    - Amazon Nova family (all tiers)
    - Meta Llama models
    - Mistral AI models
    - Cohere models
    - AI21 Labs models
    - And any other Bedrock models enabled in your region

**Example Recommendations by Use Case:**

| Scenario | Example Model | Why |
|----------|--------------|-----|
| **Quick Questions** | `amazon.nova-micro-v1:0` | Fast, cost-effective, great for simple queries |
| **General Chat** | `amazon.nova-lite-v1:0` | Balanced performance for everyday use |
| **Complex Analysis** | `amazon.nova-pro-v1:0` | Long context, advanced reasoning |
| **Code & Technical** | `anthropic.claude-sonnet-4-5-20250929-v1:0` | Superior coding, debugging, technical writing |
| **Creative Writing** | `anthropic.claude-sonnet-4-5-20250929-v1:0` | Nuanced language, storytelling, content creation |
| **Cost-Sensitive** | `amazon.nova-micro-v1:0` | Lowest cost per token |

!!! tip "Experiment & Compare"
    OpenWebUI allows you to switch models mid-conversation. Start with a faster, cheaper model for brainstorming, then switch to a more powerful model for detailed implementation. Try different models to find what works best for your specific needs.

---

## 🔒 Security Best Practices

!!! warning "Production Deployment Checklist"
    ✅ **Use HTTPS:** Always use HTTPS for both OpenWebUI and stdapi.ai endpoints

    ✅ **Secure API Keys:** Store keys in environment variables or secrets management systems, never in code

    ✅ **Enable Authentication:** Configure OpenWebUI user authentication to control access

    ✅ **Network Isolation:** Deploy in a private network with proper firewall rules

    ✅ **Regular Updates:** Keep OpenWebUI and stdapi.ai updated with security patches

    ✅ **Audit Logs:** Enable logging for compliance and security monitoring

    ✅ **Rate Limiting:** Configure rate limits to prevent abuse

    ✅ **Backup Data:** Regular backups of conversation history and uploaded documents

---

## 🚀 Next Steps & Resources

### Getting Started

1. **Deploy:** Use the Docker Compose example above to get running in minutes
2. **Configure Models:** Add your preferred models in OpenWebUI's settings
3. **Test Features:** Try chat, voice, images, and document upload
4. **Customize:** Adjust system prompts and model parameters to your needs
5. **Scale:** Monitor usage and scale your infrastructure as needed

### Learn More

!!! info "Additional Resources"
    - **[API Overview](api_overview.md)** — Complete list of available models and capabilities
    - **[Chat Completions API](api_openai_chat_completions.md)** — Detailed chat API documentation
    - **[Audio APIs](api_openai_audio_speech.md)** — TTS and STT implementation details
    - **[Configuration Guide](operations_configuration.md)** — Advanced stdapi.ai configuration options
    - **[OpenWebUI Documentation](https://docs.openwebui.com)** — Official OpenWebUI docs

### Community & Support

!!! question "Need Help?"
    - 💬 Join the OpenWebUI Discord community for tips and troubleshooting
    - 📖 Review Amazon Bedrock documentation for model-specific details
    - 🐛 Report issues on the [GitHub repository](https://github.com/stdapi-ai/stdapi.ai)
    - 🔧 Consult AWS Support for infrastructure and model access questions

---

## ⚠️ Important Considerations

!!! warning "Model Availability"
    **Regional Differences:** Not all Amazon Bedrock models are available in every AWS region. Before configuring models, verify availability in your AWS region.

    **Check availability:** See the [API Overview](api_overview.md) for a complete list of supported models by region.

!!! info "Performance Tips"
    **Model Response Times:** Larger models (Claude Sonnet, Nova Pro) take longer to respond than smaller models (Nova Micro, Nova Lite). Choose appropriately for your use case.

    **Streaming Responses:** OpenWebUI supports streaming for a better user experience—responses appear word-by-word instead of all at once.

    **Concurrent Users:** Plan infrastructure capacity based on expected concurrent users and their model preferences.

!!! tip "Cost Optimization"
    - **Right-Size Models:** Use Nova Micro for simple tasks to reduce costs
    - **Monitor Usage:** Track token consumption through AWS CloudWatch
    - **Set Quotas:** Configure user quotas in OpenWebUI to prevent runaway costs
    - **Cache Responses:** Consider implementing caching for frequently asked questions
    - **Optimize Prompts:** Shorter prompts consume fewer tokens while maintaining quality
