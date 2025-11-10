# IDE Integration — Continue.dev & Others

Get AI-powered assistance in your IDE with Amazon Bedrock models through stdapi.ai. Code completion, generation, and chat directly in VS Code, JetBrains IDEs, and more.

## Why IDE Integration + stdapi.ai?

!!! success "For Developers"
    Many IDE extensions and AI-powered IDEs are designed for the OpenAI API, making stdapi.ai a drop-in replacement to access coding models like Claude Sonnet and Amazon Nova. This guide uses **Continue.dev** as the primary example, but the same configuration approach works for many other tools.

!!! info "Compatible IDE Tools & Extensions"
    **Popular IDE Extensions:**

    - **[Continue.dev](https://continue.dev)** ([GitHub](https://github.com/continuedev/continue)) — Open-source, 15,000+ stars, VS Code & JetBrains
    - **[Cline](https://github.com/cline/cline)** — Autonomous coding agent for VS Code
    - **[Twinny](https://github.com/twinnydotdev/twinny)** — Privacy-focused AI assistant for VS Code
    - **[JetBrains AI Assistant](https://www.jetbrains.com/ai/)** — Official JetBrains AI plugin
    - **[CodeGPT](https://codegpt.co/)** ([Docs](https://docs.codegpt.co/)) — Multi-provider AI assistant for various IDEs

    **AI-First IDEs:**

    - **[Cursor](https://cursor.sh/)** — AI-first fork of VS Code with OpenAI integration
    - **[Windsurf](https://codeium.com/windsurf)** — AI-native IDE by Codeium
    - **[Zed](https://zed.dev/)** — High-performance editor with AI features

    All these tools support custom OpenAI-compatible API endpoints, making them compatible with stdapi.ai using the same configuration pattern demonstrated in this guide.

---

## About Continue.dev

**🔗 Links:** [Website](https://continue.dev) | [GitHub](https://github.com/continuedev/continue) | [Documentation](https://docs.continue.dev) | [Discord](https://discord.gg/vapESyrFmJ)

Continue.dev is an open-source AI code assistant chosen for this guide because of its:

- Wide adoption - 15,000+ GitHub stars, active community
- Multi-IDE support - Works in VS Code and all JetBrains IDEs
- Open source - Transparent, extensible, and privacy-focused
- Simple configuration - JSON-based config for stdapi.ai integration

**Key Benefits:**

- Code models - Claude Sonnet and other Bedrock models for coding, debugging, and technical tasks
- Privacy & control - Your code stays in your AWS environment
- Cost efficient - AWS pricing instead of OpenAI rates
- IDE integration - Works in VS Code and JetBrains IDEs
- Simple setup - Configuration change, no extension modifications needed
- Enterprise ready - For teams with security and compliance requirements

---

!!! warning "Work in Progress"
    This integration guide is actively being developed and refined. While the configuration examples are based on documented APIs and best practices, they are pending practical validation. Complete end-to-end deployment examples will be added once testing is finalized.

## Prerequisites

!!! info "What You'll Need"
    Before you begin, make sure you have:

    - ✓ VS Code or a JetBrains IDE (IntelliJ, PyCharm, WebStorm, etc.)
    - ✓ Continue.dev extension installed ([VS Code](https://marketplace.visualstudio.com/items?itemName=Continue.continue) / [JetBrains](https://plugins.jetbrains.com/plugin/22707-continue))
    - ✓ Your stdapi.ai server URL (e.g., `https://api.example.com`)
    - ✓ An API key (if authentication is enabled)
    - ✓ AWS Bedrock access configured with coding-capable models

---

## 🚀 Quick Start Configuration

### Step 1: Open Continue Configuration

Continue.dev stores its configuration in a JSON file that you can edit directly from your IDE.

!!! example "Opening the Config File"
    **In VS Code:**

    1. Open Command Palette: `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
    2. Type: `Continue: Open config.json`
    3. Press Enter

    **In JetBrains IDEs:**

    1. Open Settings/Preferences: `Ctrl+Alt+S` (Windows/Linux) or `Cmd+,` (Mac)
    2. Navigate to: **Tools → Continue**
    3. Click **Edit config.json**

    Alternatively, find the config file at:
    - **VS Code:** `~/.continue/config.json`
    - **JetBrains:** `~/.continue/config.json`

---

### Step 2: Configure Chat Model

Update the chat model configuration to use stdapi.ai with your preferred Bedrock model.

!!! example "config.json - Chat Configuration"
    ```json
    {
      "models": [
        {
          "title": "Claude 4.5 Sonnet",
          "provider": "openai",
          "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
          "apiKey": "YOUR_STDAPI_KEY",
          "apiBase": "https://YOUR_SERVER_URL/v1"
        }
      ]
    }
    ```

**Available Models:**

All Amazon Bedrock chat models work with Continue.dev. Popular choices for coding include Claude Sonnet (best for complex coding tasks), Claude Haiku (fast, efficient for quick queries), Amazon Nova Pro (strong reasoning, long context), and Amazon Nova Lite (balanced performance and cost).

!!! tip "Model Selection for Coding"
    **Claude Sonnet** is highly recommended for coding tasks. It excels at:
    - Understanding complex codebases
    - Generating production-quality code
    - Debugging and refactoring
    - Explaining technical concepts
    - Multi-file code changes

---

### Step 3: Configure Autocomplete (Optional but Recommended)

For real-time code completions as you type, configure a fast model optimized for autocomplete.

!!! example "config.json - Autocomplete Configuration"
    ```json
    {
      "models": [
        {
          "title": "Claude 4.5 Sonnet",
          "provider": "openai",
          "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
          "apiKey": "YOUR_STDAPI_KEY",
          "apiBase": "https://YOUR_SERVER_URL/v1"
        }
      ],
      "tabAutocompleteModel": {
        "title": "Nova Lite Autocomplete",
        "provider": "openai",
        "model": "amazon.nova-lite-v1:0",
        "apiKey": "YOUR_STDAPI_KEY",
        "apiBase": "https://YOUR_SERVER_URL/v1"
      }
    }
    ```

!!! tip "Autocomplete Model Choice"
    Use faster, smaller models for autocomplete to get instant suggestions:

    - **Amazon Nova Lite** — `amazon.nova-lite-v1:0` (balanced, good quality)
    - **Amazon Nova Micro** — `amazon.nova-micro-v1:0` (fastest, most cost-effective)
    - **Claude Haiku** — Fast Claude model for quick completions

---

### Step 4: Add Multiple Models (Optional)

You can configure multiple models and switch between them based on your task.

!!! example "config.json - Multiple Models"
    ```json
    {
      "models": [
        {
          "title": "Claude 4.5 Sonnet (Complex Tasks)",
          "provider": "openai",
          "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
          "apiKey": "YOUR_STDAPI_KEY",
          "apiBase": "https://YOUR_SERVER_URL/v1"
        },
        {
          "title": "Claude 3.5 Haiku (Quick Questions)",
          "provider": "openai",
          "model": "anthropic.claude-3-5-haiku-20241022-v1:0",
          "apiKey": "YOUR_STDAPI_KEY",
          "apiBase": "https://YOUR_SERVER_URL/v1"
        },
        {
          "title": "Nova Pro (Long Context)",
          "provider": "openai",
          "model": "amazon.nova-pro-v1:0",
          "apiKey": "YOUR_STDAPI_KEY",
          "apiBase": "https://YOUR_SERVER_URL/v1"
        }
      ],
      "tabAutocompleteModel": {
        "title": "Nova Lite Autocomplete",
        "provider": "openai",
        "model": "amazon.nova-lite-v1:0",
        "apiKey": "YOUR_STDAPI_KEY",
        "apiBase": "https://YOUR_SERVER_URL/v1"
      }
    }
    ```

!!! tip "Model Switching"
    Continue.dev lets you switch models during a conversation using the dropdown in the chat interface. Set up multiple models with descriptive titles to quickly choose the right tool for each task.

---

### Step 5: Configure Embeddings for Codebase Context (Optional)

Enable Continue to use embeddings for better codebase understanding and retrieval.

!!! example "config.json - Embeddings Configuration"
    ```json
    {
      "models": [
        {
          "title": "Claude 4.5 Sonnet",
          "provider": "openai",
          "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
          "apiKey": "YOUR_STDAPI_KEY",
          "apiBase": "https://YOUR_SERVER_URL/v1"
        }
      ],
      "embeddingsProvider": {
        "provider": "openai",
        "model": "amazon.titan-embed-text-v2:0",
        "apiKey": "YOUR_STDAPI_KEY",
        "apiBase": "https://YOUR_SERVER_URL/v1"
      }
    }
    ```

**Available Embedding Models:**

- **Amazon Titan Embed Text v2** — `amazon.titan-embed-text-v2:0` (recommended, 8192 dimensions)
- **Amazon Titan Embed Text v1** — `amazon.titan-embed-text-v1` (legacy, 1536 dimensions)
- **Cohere Embed** — If enabled in your AWS region

!!! info "Why Use Embeddings?"
    Embeddings enable Continue to:
    - Search your entire codebase semantically
    - Find relevant code automatically based on your question
    - Provide better context-aware suggestions
    - Understand relationships between files and functions

---

## 🎯 What You Can Do Now

Once configured, Continue.dev with stdapi.ai unlocks powerful coding capabilities:

### 💬 AI Chat for Code

- **Ask Questions:** "How does the authentication system work?"
- **Generate Code:** "Write a function to validate email addresses"
- **Refactor:** "Refactor this function to use async/await"
- **Debug:** "Why is this throwing a NullPointerException?"
- **Explain:** "Explain what this regex pattern does"
- **Document:** "Add comprehensive JSDoc comments to this function"

### ✨ Tab Autocomplete

- **Real-time Suggestions:** Get inline code completions as you type
- **Context Aware:** Completions understand your codebase style and patterns
- **Multi-line:** Generate entire functions or code blocks
- **Smart:** Learns from your project structure and imports

### 📝 Code Actions

- **Highlight & Edit:** Select code and ask AI to modify it
- **Multi-file Changes:** Make coordinated changes across multiple files
- **Test Generation:** Generate unit tests for selected functions
- **Documentation:** Auto-generate docstrings and comments

### 🔍 Codebase Understanding

- **Semantic Search:** "@codebase how do we handle user authentication?"
- **Architecture Questions:** "What's the overall structure of the backend?"
- **Dependency Tracking:** "Where is this function used?"
- **API Discovery:** "Show me examples of using the database client"

---

## 📊 Model Recommendations by Task

Choose the right model for each coding task to optimize performance and cost. **These are suggestions**—experiment to find what works best for your workflow.

| Task | Recommended Model | Why |
|------|------------------|-----|
| **Complex Refactoring** | `anthropic.claude-sonnet-4-5-20250929-v1:0` | Superior code understanding and generation |
| **Quick Questions** | `anthropic.claude-3-5-haiku-20241022-v1:0` | Fast responses for simple queries |
| **Long Context** | `amazon.nova-pro-v1:0` | Large context window for big files |
| **Code Review** | `anthropic.claude-sonnet-4-5-20250929-v1:0` | Detailed analysis and suggestions |
| **Autocomplete** | `amazon.nova-lite-v1:0` | Fast, cost-effective inline suggestions |
| **Documentation** | `anthropic.claude-sonnet-4-5-20250929-v1:0` | Clear, comprehensive explanations |
| **Bug Hunting** | `anthropic.claude-sonnet-4-5-20250929-v1:0` | Deep debugging and root cause analysis |

---

## 💡 Pro Tips & Best Practices

!!! tip "Context is Key"
    **Tag relevant files:** Use `@filename` to include specific files in your context

    **Use @codebase:** Ask questions about your entire project with `@codebase`

    **Highlight code:** Select the exact code you're asking about for precise answers

!!! tip "Optimize for Speed"
    **Fast models for quick tasks:** Use Haiku or Nova Lite for simple questions

    **Premium models for complex work:** Use Claude Sonnet for refactoring and architecture

    **Autocomplete with lightweight models:** Keep completions fast with Nova Lite/Micro

!!! tip "Better Prompts = Better Code"
    **Be specific:** "Add error handling for network timeouts" vs "improve this"

    **Provide context:** Mention frameworks, languages, patterns you're using

    **Iterate:** Start broad, then refine with follow-up questions

!!! tip "Team Configuration"
    **Share config:** Commit a template `config.json` (without API keys) to your repo

    **Environment variables:** Use `${env:STDAPI_KEY}` to reference environment variables

    **Consistent models:** Standardize on models across your team for predictable results

---

## 🔧 Complete Configuration Example

!!! example "Full config.json Template"
    ```json
    {
      "models": [
        {
          "title": "Claude 4.5 Sonnet (Primary)",
          "provider": "openai",
          "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
          "apiKey": "YOUR_STDAPI_KEY",
          "apiBase": "https://YOUR_SERVER_URL/v1",
          "contextLength": 200000
        },
        {
          "title": "Claude 3.5 Haiku (Fast)",
          "provider": "openai",
          "model": "anthropic.claude-3-5-haiku-20241022-v1:0",
          "apiKey": "YOUR_STDAPI_KEY",
          "apiBase": "https://YOUR_SERVER_URL/v1",
          "contextLength": 200000
        },
        {
          "title": "Nova Pro (Long Context)",
          "provider": "openai",
          "model": "amazon.nova-pro-v1:0",
          "apiKey": "YOUR_STDAPI_KEY",
          "apiBase": "https://YOUR_SERVER_URL/v1",
          "contextLength": 300000
        }
      ],
      "tabAutocompleteModel": {
        "title": "Nova Lite Autocomplete",
        "provider": "openai",
        "model": "amazon.nova-lite-v1:0",
        "apiKey": "YOUR_STDAPI_KEY",
        "apiBase": "https://YOUR_SERVER_URL/v1"
      },
      "embeddingsProvider": {
        "provider": "openai",
        "model": "amazon.titan-embed-text-v2:0",
        "apiKey": "YOUR_STDAPI_KEY",
        "apiBase": "https://YOUR_SERVER_URL/v1"
      },
      "allowAnonymousTelemetry": false
    }
    ```

---

## 🚀 Next Steps & Resources

### Getting Started

1. **Test Your Setup:** Ask Continue a simple question like "What does this file do?"
2. **Try Autocomplete:** Start typing a function and wait for suggestions
3. **Explore Features:** Use `@codebase`, `@file`, and `/edit` commands
4. **Customize:** Adjust models based on your workflow and budget
5. **Share:** Help your team set up Continue with stdapi.ai

### Learn More

!!! info "Additional Resources"
    - **[Continue.dev Documentation](https://continue.dev/docs)** — Official Continue.dev guides
    - **[API Overview](api_overview.md)** — Complete list of available Bedrock models
    - **[Chat Completions API](api_openai_chat_completions.md)** — Detailed chat API documentation
    - **[Configuration Guide](operations_configuration.md)** — Advanced stdapi.ai configuration options

### Community & Support

!!! question "Need Help?"
    - 💬 Join the [Continue.dev Discord](https://discord.gg/vapESyrFmJ) for tips and troubleshooting
    - 📖 Review Amazon Bedrock documentation for model-specific details
    - 🐛 Report issues on the [GitHub repository](https://github.com/stdapi-ai/stdapi.ai)
    - 🔧 Consult AWS Support for infrastructure and model access questions

---

## ⚠️ Important Considerations

!!! warning "Model Availability"
    **Regional Differences:** Not all Amazon Bedrock models are available in every AWS region. Verify model availability in your configured region before setting up Continue.dev.

    **Check availability:** See the [API Overview](api_overview.md) for supported models by region.

!!! info "Performance Tips"
    **Context Length:** Larger context windows allow more code to be analyzed but increase latency and cost. Adjust based on your needs.

    **Autocomplete Frequency:** Faster models provide better autocomplete experience. Nova Lite/Micro are recommended over premium models.

    **Caching:** Continue.dev caches embeddings locally. First-time codebase indexing may take a few minutes.

!!! tip "Cost Optimization"
    - **Right-size models:** Use Haiku or Nova Lite for simple questions to reduce costs
    - **Monitor token usage:** Large file selections consume more tokens—be selective
    - **Autocomplete wisely:** Autocomplete can generate many requests; use efficient models
    - **Share knowledge:** Document good prompts and patterns for your team to reduce trial-and-error
