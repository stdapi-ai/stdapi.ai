# LibreChat Integration

Connect LibreChat to Amazon Bedrock models through stdapi.ai. Deploy a self-hosted ChatGPT alternative with multi-user support, conversation management, and access to Claude, Amazon Nova, and other Bedrock models.

## About LibreChat

**🔗 Links:** [Website](https://librechat.ai) | [GitHub](https://github.com/danny-avila/LibreChat) | [Documentation](https://docs.librechat.ai) | [Discord](https://discord.librechat.ai)

LibreChat is a feature-rich, open-source AI chat platform with:

- 30,000+ GitHub stars - Popular open-source ChatGPT alternative
- Production-ready - Used by organizations worldwide
- Multi-modal - Chat, voice, images, and document analysis
- Extensible - Plugin system, custom endpoints, and integrations

## Why LibreChat + stdapi.ai?

!!! success "Enterprise ChatGPT Alternative"
    LibreChat is designed for teams and organizations needing a private, customizable AI assistant. With stdapi.ai, you get the familiar ChatGPT experience backed by Amazon Bedrock's enterprise-grade models—all within your own infrastructure.

**Key Benefits:**

- Multi-user platform - Team collaboration with individual accounts and conversations
- Privacy - Your conversations and data stay in your infrastructure
- Enterprise models - Access Claude, Amazon Nova, and all Bedrock models
- Rich features - Conversation history, presets, plugins, file uploads
- Cost control - AWS pricing with usage tracking and quotas
- Customizable - White-label, branding, and custom configurations
- Active development - Regular updates with new features

---

!!! warning "Work in Progress"
    This integration guide is actively being developed and refined. While the configuration examples are based on documented APIs and best practices, they are pending practical validation. Complete end-to-end deployment examples will be added once testing is finalized.

## Prerequisites

!!! info "What You'll Need"
    Before you begin, make sure you have:

    - ✓ Docker and Docker Compose installed (or Kubernetes for production)
    - ✓ Your stdapi.ai server URL (e.g., `https://api.example.com`)
    - ✓ An API key (if authentication is enabled)
    - ✓ AWS Bedrock access configured with desired models
    - ✓ (Optional) MongoDB for conversation storage
    - ✓ (Optional) Domain name and SSL certificate for production

---

## 🚀 Quick Start with Docker

### Step 1: Clone LibreChat Repository

Get the latest version of LibreChat from GitHub.

!!! example "Clone and Setup"
    ```bash
    # Clone the repository
    git clone https://github.com/danny-avila/LibreChat.git
    cd LibreChat

    # Copy the example environment file
    cp .env.example .env
    ```

---

### Step 2: Configure stdapi.ai Connection

Edit the `.env` file to configure LibreChat to use stdapi.ai as the OpenAI provider.

!!! example ".env Configuration"
    ```bash
    # Core LibreChat settings
    HOST=0.0.0.0
    PORT=3080

    # MongoDB (required for conversation storage)
    MONGO_URI=mongodb://mongodb:27017/LibreChat

    # Session secret (generate a random string)
    SESSION_SECRET=your_random_session_secret_here

    # stdapi.ai as OpenAI provider
    OPENAI_API_KEY=your_stdapi_key_here
    OPENAI_REVERSE_PROXY=https://YOUR_SERVER_URL/v1

    # Optional: Enable all features
    DEBUG_OPENAI=true
    ```

!!! tip "Generate Session Secret"
    Generate a secure session secret:
    ```bash
    openssl rand -base64 32
    ```

---

### Step 3: Configure Available Models

Create a `librechat.yaml` configuration file to define which Bedrock models are available to users.

!!! example "librechat.yaml - Model Configuration"
    ```yaml
    version: 1.0.5
    cache: true

    endpoints:
      custom:
        - name: "Amazon Bedrock"
          apiKey: "${OPENAI_API_KEY}"
          baseURL: "https://YOUR_SERVER_URL/v1"
          models:
            default:
              - "anthropic.claude-sonnet-4-5-20250929-v1:0"
              - "anthropic.claude-3-5-haiku-20241022-v1:0"
              - "amazon.nova-pro-v1:0"
              - "amazon.nova-lite-v1:0"
              - "amazon.nova-micro-v1:0"
            fetch: false
          titleConvo: true
          titleModel: "amazon.nova-micro-v1:0"
          summarize: false
          summaryModel: "amazon.nova-lite-v1:0"
          forcePrompt: false
          modelDisplayLabel: "Amazon Bedrock"
    ```

**Available Models:**

All Amazon Bedrock chat models are compatible. Add any model IDs available in your AWS region:

- **Anthropic Claude** — All Claude model variants (Sonnet, Haiku, Opus)
- **Amazon Nova** — All Nova family models (Pro, Lite, Micro)
- **Meta Llama** — Llama models (if enabled)
- **Mistral AI** — Mistral and Mixtral models (if enabled)
- **And more** — Any Bedrock chat model

!!! tip "Model Organization"
    List models in order of capability (best to fastest) for a better user experience. Users will see this order in the model selector dropdown.

---

### Step 4: Enable Additional Features

Enhance your LibreChat instance with file uploads, speech-to-text, and image generation.

!!! example "Enhanced .env Configuration"
    ```bash
    # Core settings (from Step 2)
    OPENAI_API_KEY=your_stdapi_key_here
    OPENAI_REVERSE_PROXY=https://YOUR_SERVER_URL/v1

    # File upload and parsing
    ENABLE_FILE_UPLOADS=true
    FILE_UPLOAD_SIZE_LIMIT=20  # MB

    # Speech to Text (Whisper API compatible)
    STT_API_KEY=your_stdapi_key_here
    STT_API_URL=https://YOUR_SERVER_URL/v1

    # Text to Speech
    TTS_API_KEY=your_stdapi_key_here
    TTS_API_URL=https://YOUR_SERVER_URL/v1

    # Image generation
    DALLE_API_KEY=your_stdapi_key_here
    DALLE_REVERSE_PROXY=https://YOUR_SERVER_URL/v1

    # Search (optional, for web search integration)
    # SEARCH_API_KEY=your_search_api_key
    ```

---

### Step 5: Deploy with Docker Compose

Start LibreChat with all dependencies using Docker Compose.

!!! example "docker-compose.override.yml"
    Create this file to customize the deployment:

    ```yaml
    version: '3.8'

    services:
      api:
        volumes:
          - ./librechat.yaml:/app/librechat.yaml:ro
        environment:
          - OPENAI_API_KEY=${OPENAI_API_KEY}
          - OPENAI_REVERSE_PROXY=${OPENAI_REVERSE_PROXY}
          - STT_API_KEY=${STT_API_KEY}
          - STT_API_URL=${STT_API_URL}
          - TTS_API_KEY=${TTS_API_KEY}
          - TTS_API_URL=${TTS_API_URL}
          - DALLE_API_KEY=${DALLE_API_KEY}
          - DALLE_REVERSE_PROXY=${DALLE_REVERSE_PROXY}
    ```

**Start the application:**

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Access LibreChat at http://localhost:3080
```

---

## 🎯 What You Can Do Now

Once deployed, LibreChat with stdapi.ai provides a full-featured AI assistant platform:

### 💬 Conversational AI

- **Multi-Turn Conversations:** Contextual discussions with full conversation history
- **Model Switching:** Change models mid-conversation based on task complexity
- **Conversation Management:** Save, search, share, and organize conversations
- **Presets:** Create reusable conversation templates with custom instructions

### 🎨 Multi-Modal Capabilities

- **Image Generation:** Create images using Amazon Nova Canvas directly in chat
- **Voice Input:** Speak your messages using speech-to-text (Amazon Transcribe)
- **Voice Output:** Listen to responses with text-to-speech (Amazon Polly)
- **File Analysis:** Upload documents for AI analysis and Q&A

### 👥 Team Collaboration

- **Multi-User Support:** Individual accounts with separate conversation histories
- **User Management:** Admin controls for user registration and permissions
- **Shared Conversations:** Export and share conversations with team members
- **Usage Tracking:** Monitor token usage and costs per user

### ⚙️ Customization

- **Custom Prompts:** Define system prompts and conversation starters
- **Branding:** Customize logo, colors, and application name
- **Model Presets:** Pre-configure optimal settings for specific use cases
- **Plugins:** Extend functionality with community plugins

---

## 🔧 Advanced Configuration

### User Registration and Authentication

Control who can access your LibreChat instance.

!!! example "Authentication Settings (.env)"
    ```bash
    # Allow registration (set to false for invite-only)
    ALLOW_REGISTRATION=true

    # Email verification (requires SMTP)
    ALLOW_EMAIL_LOGIN=true
    EMAIL_SERVICE=gmail
    EMAIL_USERNAME=your-email@gmail.com
    EMAIL_PASSWORD=your-app-password
    EMAIL_FROM=noreply@yourdomain.com

    # Social login (optional)
    GOOGLE_CLIENT_ID=your_google_client_id
    GOOGLE_CLIENT_SECRET=your_google_client_secret
    ```

---

### RAG and Document Intelligence

Enable document uploads and semantic search for knowledge base functionality.

!!! example "RAG Configuration (librechat.yaml)"
    ```yaml
    endpoints:
      custom:
        - name: "Amazon Bedrock"
          apiKey: "${OPENAI_API_KEY}"
          baseURL: "https://YOUR_SERVER_URL/v1"
          models:
            default:
              - "anthropic.claude-sonnet-4-5-20250929-v1:0"
              - "amazon.nova-pro-v1:0"
              - "amazon.nova-lite-v1:0"
          titleConvo: true
          titleModel: "amazon.nova-micro-v1:0"
          # Enable file uploads for RAG
          fileConfig:
            endpoints:
              - "custom"
            fileLimit: 10
            fileSizeLimit: 20  # MB
            totalSizeLimit: 100  # MB
            supportedMimeTypes:
              - "application/pdf"
              - "text/plain"
              - "text/markdown"
              - "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ```

**Embedding Configuration (.env):**

```bash
# Use stdapi.ai for embeddings
EMBEDDINGS_PROVIDER=openai
OPENAI_API_KEY=your_stdapi_key_here
EMBEDDINGS_MODEL=amazon.titan-embed-text-v2:0
```

---

### Production Deployment

Deploy LibreChat in production with SSL, persistent storage, and backups.

!!! example "Production docker-compose.yml"
    ```yaml
    version: '3.8'

    services:
      nginx:
        image: nginx:alpine
        ports:
          - "80:80"
          - "443:443"
        volumes:
          - ./nginx.conf:/etc/nginx/nginx.conf:ro
          - ./ssl:/etc/nginx/ssl:ro
        depends_on:
          - api

      api:
        image: ghcr.io/danny-avila/librechat:latest
        env_file:
          - .env
        volumes:
          - ./librechat.yaml:/app/librechat.yaml:ro
          - librechat-images:/app/client/public/images
          - librechat-logs:/app/api/logs
        depends_on:
          - mongodb
        restart: unless-stopped

      mongodb:
        image: mongo:6.0
        volumes:
          - mongodb-data:/data/db
        restart: unless-stopped
        command: mongod --quiet --logpath /dev/null

    volumes:
      mongodb-data:
      librechat-images:
      librechat-logs:
    ```

!!! warning "Production Checklist"
    ✅ **Use HTTPS:** Configure SSL certificates (Let's Encrypt recommended)

    ✅ **Secure MongoDB:** Use authentication and restrict network access

    ✅ **Regular Backups:** Backup MongoDB data and configuration files

    ✅ **Rate Limiting:** Configure rate limits in nginx to prevent abuse

    ✅ **Monitoring:** Set up logging and monitoring for uptime and performance

    ✅ **Updates:** Regularly update LibreChat and dependencies

---

## 📊 Model Recommendations

Choose the right models for different use cases to optimize performance and cost. **These are examples**—all Bedrock models are available.

| Use Case | Example Model | Why |
|----------|--------------|-----|
| **Complex Tasks** | `anthropic.claude-sonnet-4-5-20250929-v1:0` | Superior reasoning, coding, analysis |
| **Daily Chat** | `amazon.nova-lite-v1:0` | Balanced performance for general use |
| **Quick Questions** | `amazon.nova-micro-v1:0` | Fast, cost-effective responses |
| **Long Context** | `amazon.nova-pro-v1:0` | Large context window for documents |
| **Title Generation** | `amazon.nova-micro-v1:0` | Fast, efficient for background tasks |
| **Summarization** | `amazon.nova-lite-v1:0` | Good quality at reasonable cost |

!!! tip "User Choice"
    Let users select their preferred model for each conversation. Configure multiple models in `librechat.yaml` to give users flexibility based on their needs.

---

## 💡 Pro Tips & Best Practices

!!! tip "Performance Optimization"
    **Model Selection:** Use Nova Micro for title generation to reduce costs—it runs frequently

    **Caching:** Enable conversation caching to reduce redundant API calls

    **Connection Pooling:** Configure MongoDB connection pooling for better performance

    **CDN:** Serve static assets via CDN for faster page loads

!!! tip "Cost Management"
    **Set Quotas:** Configure per-user token limits to control costs

    **Monitor Usage:** Use LibreChat's built-in analytics to track token consumption

    **Right-Size Models:** Educate users on when to use efficient vs. premium models

    **Summarization:** Enable conversation summarization to reduce context token usage

!!! tip "Security Best Practices"
    **Environment Variables:** Never commit `.env` files—use secrets management

    **API Key Rotation:** Regularly rotate stdapi.ai API keys

    **User Validation:** Enable email verification to prevent spam registrations

    **Content Filtering:** Consider implementing content moderation policies

!!! tip "User Experience"
    **Default Model:** Set a balanced model (Nova Lite) as default for new users

    **Presets:** Create conversation presets for common use cases (coding, writing, analysis)

    **Instructions:** Add a welcome message explaining available models and features

    **Feedback:** Collect user feedback to optimize model selection and settings

---

## 🚀 Next Steps & Resources

### Getting Started

1. **Access LibreChat:** Open http://localhost:3080 and create an account
2. **Test Models:** Try different models to understand their strengths
3. **Configure Presets:** Set up conversation templates for your workflows
4. **Invite Team:** Share access with your team members
5. **Monitor Usage:** Track token consumption and adjust as needed

### Learn More

!!! info "Additional Resources"
    - **[LibreChat Documentation](https://docs.librechat.ai/)** — Official LibreChat guides and configuration
    - **[API Overview](api_overview.md)** — Complete list of available Bedrock models
    - **[Chat Completions API](api_openai_chat_completions.md)** — Detailed chat API documentation
    - **[Configuration Guide](operations_configuration.md)** — Advanced stdapi.ai configuration options

### Community & Support

!!! question "Need Help?"
    - 💬 Join the [LibreChat Discord](https://discord.librechat.ai) for tips and troubleshooting
    - 📖 Review Amazon Bedrock documentation for model-specific details
    - 🐛 Report LibreChat issues on [GitHub](https://github.com/danny-avila/LibreChat)
    - 🔧 Report stdapi.ai issues on the [GitHub repository](https://github.com/stdapi-ai/stdapi.ai)

---

## ⚠️ Important Considerations

!!! warning "Model Availability"
    **Regional Differences:** Not all Amazon Bedrock models are available in every AWS region. Verify model availability in your configured region before adding them to `librechat.yaml`.

    **Check availability:** See the [API Overview](api_overview.md) for supported models by region.

!!! info "Scaling Considerations"
    **User Capacity:** Plan infrastructure based on expected concurrent users

    **Database Performance:** MongoDB performance is critical—consider replica sets for production

    **API Rate Limits:** Be aware of Bedrock rate limits and request quota increases if needed

    **Storage Growth:** Conversation history grows over time—plan for storage expansion

!!! tip "Migration from OpenAI"
    If you're already using LibreChat with OpenAI:

    1. Add stdapi.ai as a custom endpoint in `librechat.yaml`
    2. Keep OpenAI endpoint alongside for comparison
    3. Test thoroughly with your team before switching completely
    4. Update documentation for your users about new models
    5. Monitor cost differences and adjust based on results
