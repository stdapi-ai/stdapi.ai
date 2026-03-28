---
title: AI Coding Assistants - AWS Bedrock for IDEs
description: Connect Continue.dev, Cursor, Cline, Claude Code, and other AI coding assistants to AWS Bedrock via stdapi.ai. Use Claude, Kimi K2 thinking, and Qwen Coder in VS Code and JetBrains IDEs.
keywords: AI coding assistant AWS, Continue.dev AWS Bedrock, Cursor AWS integration, VS Code AI AWS, AI pair programming, coding copilot AWS, IDE AI integration, private Copilot, Claude Code AWS Bedrock
---

# AI Coding Assistants Integration

Connect your favorite AI coding assistants to AWS Bedrock models through stdapi.ai. Get intelligent code completions, chat assistance, and codebase understanding with powerful AWS models like Claude, Kimi K2 thinking, and Qwen Coder Next—no vendor lock-in required.

## About AI Coding Assistants

AI coding assistants are IDE extensions and terminal tools that leverage large language models to enhance developer productivity. These tools provide real-time code completions, intelligent suggestions, natural language code generation, and interactive chat capabilities directly within your coding environment—acting as AI pair programmers that understand your codebase context.

**What AI coding assistants can do:**

- **Real-time completions** - Autocomplete code as you type with context awareness
- **Interactive chat** - Ask questions about your codebase, get explanations
- **Code generation** - Natural language to code conversion
- **Refactoring** - Intelligent code improvements and optimization suggestions
- **Documentation** - Auto-generate comments, docstrings, and READMEs
- **Testing** - Create unit tests, debug issues, suggest fixes
- **Git integration** - Generate commit messages, review diffs
- **Multi-language** - Support for Python, JavaScript, TypeScript, Go, Rust, Java, and more

## Why AI Coding Assistants + stdapi.ai?

<div class="grid cards" markdown>

- :material-puzzle: __Works with Your IDE__
  <br>Almost any coding assistant that supports OpenAI or Anthropic compatible APIs works with stdapi.ai. Continue.dev, Cursor, Cline, Claude Code, Windsurf, Aider—all compatible with AWS Bedrock models.

- :material-brain: __Best-in-Class Coding Models__
  <br>Claude 4.6+ for reasoning and architecture, Kimi K2 thinking for complex problem-solving, Qwen Coder Next for specialized coding tasks. Choose the right model for each task.

- :material-lock: __Code Privacy Guaranteed__
  <br>Your code never leaves your AWS account. Perfect for proprietary codebases, enterprise security requirements, or compliance-sensitive projects.

- :material-server-network: __Flexible Deployment Options__
  <br>Run stdapi.ai in AWS for production or locally with Docker for development. Test locally, deploy to cloud—same API, same experience.

- :material-currency-usd-off: __Pay-Per-Use, No Subscriptions__
  <br>No per-developer licenses or monthly subscriptions. Pay only AWS Bedrock rates for actual usage. Use powerful models without per-seat costs.

</div>

```mermaid
%%{init: {'flowchart': {'htmlLabels': true}} }%%
flowchart LR
  ide["<img src='../styles/logo_vscode.svg' style='height:64px;width:auto;vertical-align:middle;' /> IDE + AI Assistant"] --> stdapi["<img src='../styles/logo.svg' style='height:64px;width:auto;vertical-align:middle;' /> stdapi.ai"]
  stdapi --> bedrock["<img src='../styles/logo_amazon_bedrock.svg' style='height:64px;width:auto;vertical-align:middle;' /> AWS Bedrock"]
```

## ✅ Prerequisites

!!! info "What You'll Need"
    - ✓ **stdapi.ai deployed** - [See deployment guide](operations_getting_started.md) or [run locally with Docker](operations_getting_started_local.md)
    - ✓ **Your stdapi.ai URL** - e.g., `https://api.example.com` or `http://localhost:8000` for local
    - ✓ **Your API key** - From Terraform output or configuration (optional for local development)
    - ✓ **IDE with AI assistant** - VS Code, JetBrains, Cursor, or your preferred editor with an AI coding extension

---

## ![OpenAI](styles/logo_openai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } OpenAI-Compatible Coding Assistants

**Popular Tools:** [Cline](https://github.com/cline/cline) | [JetBrains AI Assistant](https://www.jetbrains.com/ai/) | [Continue.dev](https://continue.dev/) | [Cursor](https://cursor.com/) | [Windsurf](https://codeium.com/windsurf)

Most IDE coding assistants use the OpenAI-compatible API. Configure them by pointing to stdapi.ai's `/v1` endpoint.

### ⚙️ Configuration

Most AI coding assistants follow a similar configuration pattern. The exact menu location and field names may vary, but the core settings remain consistent.

!!! example "Generic Configuration Steps"
    **In your coding assistant settings:**

    1. Navigate to **Settings** or **Preferences**
    2. Find the **AI Provider** or **Model Provider** section
    3. Select **"OpenAI Compatible"** or **"Custom OpenAI"** as the provider type
    4. Configure the connection:
        ```
        API Base URL: https://YOUR_STDAPI_URL/v1
        (or sometimes just: https://YOUR_STDAPI_URL)

        API Key: YOUR_STDAPI_KEY

        Model: anthropic.claude-opus-4-6-v1
        (or select from detected models if available)
        ```

!!! tip "Model Selection for Coding"
    **Recommended models for different tasks:**

    - **Advanced reasoning & architecture**: `anthropic.claude-opus-4-6-v1`
    - **Complex problem-solving**: Kimi K2 thinking models
    - **Specialized coding tasks**: `qwen2-coder-next-1-5-instruct-v1:0` (Qwen Coder Next)
    - **Fast completions**: Amazon Nova Micro or Nova Lite

    **Configuration tips:**

    - **Auto-detect**: Some assistants query `/v1/models` and show a dropdown
    - **Manual entry**: Use full Bedrock model ID (e.g., `anthropic.claude-opus-4-6-v1`)
    - **Multi-model setup**: Use fast, cheap models for secondary tasks (autocomplete, summaries) and powerful models for complex generation

### 💬 Chat Completions

All coding assistants use chat completions for interactive conversations, code generation, and explanations.

!!! example "How It Works"
    Your coding assistant calls `POST /v1/chat/completions` (see [Chat Completions API](api_openai_chat_completions.md)) to:

    - Answer questions about your code
    - Generate new code from natural language
    - Explain complex functions or algorithms
    - Suggest refactoring and improvements
    - Debug issues and propose fixes

    The model must be a text/chat-capable model from the correct family for your Bedrock region.

### 🛠️ Tool Calling Support

stdapi.ai fully supports tool calling (function calling) through the chat completions API, which is essential for autonomous and efficient coding agents.

!!! success "Advanced Agent Capabilities"
    **Tool calling enables your coding assistant to:**

    - Execute terminal commands and see results
    - Read and write files in your codebase
    - Search through code and documentation
    - Run tests and analyze output
    - Interact with external APIs and services

    Most modern autonomous agents like Cline or Junie rely heavily on tool calling to perform complex, multi-step coding tasks. stdapi.ai's tool calling support (see [Chat Completions API - Tool Calling](api_openai_chat_completions.md#feature-compatibility)) ensures these agents can work at their full potential with Amazon Bedrock models.

### ⚡ Code Completions

Some coding assistants support dedicated code completion endpoints for real-time suggestions as you type.

!!! example "Completion Support"
    Advanced assistants may call `POST /v1/completions` for:

    - Inline code suggestions
    - Auto-completion while typing
    - Context-aware code snippets

    Not all models or assistants support this mode. Chat-based assistants handle completions through the chat API instead.

---

## ![Anthropic](styles/logo_anthropic_claude.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Anthropic-Compatible Coding Assistants

**Popular Tools:** [Claude Code](https://docs.anthropic.com/en/docs/claude-code/overview) | [Aider](https://aider.chat/) | [JetBrains AI Assistant (With Claude code ACP)](https://www.jetbrains.com/ai/)

Tools that use the Anthropic messages API natively can be connected to stdapi.ai's `/anthropic` endpoint, enabling them to use Claude models via AWS Bedrock.

### Claude Code

Claude Code is Anthropic's agentic coding tool that runs in the terminal.

#### ⚙️ Configuration

Create or edit `~/.claude/claude.json`:

```json
{
  "env": {
    "ANTHROPIC_AUTH_TOKEN": "YOUR_API_KEY",
    "ANTHROPIC_BASE_URL": "https://YOUR_STDAPI_URL/anthropic",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-4-6",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-4-6",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-4-5"
  }
}
```

- Replace `YOUR_STDAPI_URL` with your stdapi.ai deployment URL (e.g., `https://api.example.com` or `http://localhost:8000` for local)
- Replace `YOUR_API_KEY` with your stdapi.ai API key
- The `/anthropic` path prefix is configured via the [`ANTHROPIC_ROUTES_PREFIX`](operations_configuration.md#anthropic-routes-prefix) setting (default: `/anthropic`)
- The `ANTHROPIC_DEFAULT_*_MODEL` variables are optional—they let you map Claude model tiers to specific Bedrock model IDs

!!! tip "Beta Flag Compatibility"
    stdapi.ai automatically filters unsupported `anthropic_beta` flags, so Claude Code works without needing `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`. Bedrock-supported flags (like `Interleaved-thinking-2025-05-14` and `token-efficient-tools-2025-02-19`) are preserved while unsupported ones are silently removed. See [`ANTHROPIC_BETA_FILTER`](operations_configuration.md#anthropic-beta-filter) and [`ANTHROPIC_BETA_ALLOWLIST`](operations_configuration.md#anthropic-beta-allowlist) for details.

!!! tip "Using Non-Claude Models"
    Claude Code is optimized for Claude models and may be incompatible with some non-Claude models. For the best experience, use Claude Code with stdapi.ai and Claude models. If you need to use non-Claude models (e.g., Kimi K2, Qwen Coder), some Claude-specific features may not be supported. Two common issues to address:

    - **Prompt caching** — Claude Code sends `cache_control` headers that can cause errors on models that support prompt caching but handle it differently (e.g., AWS Nova, Qwen). Set `DISABLE_PROMPT_CACHING=1` to suppress these headers.
    - **Extended thinking** — If the model does not support thinking/reasoning configuration, set `MAX_THINKING_TOKENS=0` to disable it.

    ```json
    {
      "env": {
        "ANTHROPIC_AUTH_TOKEN": "YOUR_API_KEY",
        "ANTHROPIC_BASE_URL": "https://YOUR_STDAPI_URL/anthropic",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "amazon.nova-2-lite-v1:0",
        "MAX_THINKING_TOKENS": "0",
        "DISABLE_PROMPT_CACHING": "1"
      }
    }
    ```

### Other Anthropic-Compatible Tools

Any tool using the Anthropic SDK or messages API can be configured the same way—set the `ANTHROPIC_BASE_URL` to `https://YOUR_STDAPI_URL/anthropic` and `ANTHROPIC_API_KEY` (or equivalent) to your stdapi.ai API key.

---

## 🐳 Running stdapi.ai Locally

stdapi.ai works well when running locally with Docker, making it ideal for your development environment.

!!! tip "Running Locally"
    For complete local deployment instructions, see the [Local Development Guide](operations_getting_started_local.md).

    **OpenAI-compatible tools:**
    ```
    API Base URL: http://localhost:8000/v1
    API Key: your_stdapi_key
    ```

    **Anthropic-compatible tools:**
    ```
    ANTHROPIC_BASE_URL: http://localhost:8000/anthropic
    ANTHROPIC_AUTH_TOKEN: your_stdapi_key
    ```

---
