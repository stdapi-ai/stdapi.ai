<div align="center">
<img src="docs/styles/logo.svg" alt="stdapi.ai logo" width="40%" />

# stdapi.ai

**OpenAI & Anthropic Compatible API Gateway for AWS Bedrock and AI Services**

Run your favorite OpenAI and Anthropic-compatible applications on AWS Bedrock. Access 80+ models including Claude, Kimi K2, MiniMax, Qwen with enterprise privacy, compliance controls, and pay-per-use AWS pricing.

---

[![AWS Marketplace](https://img.shields.io/badge/AWS%20Marketplace-Enterprise%20Edition-FF9900?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGZpbGw9IiNmZmZmZmYiIGZpbGwtcnVsZT0iZXZlbm9kZCIgc3R5bGU9ImZsZXg6bm9uZTtsaW5lLWhlaWdodDoxIiB2aWV3Qm94PSIwIDAgMjQgMjQiPjxwYXRoIGQ9Ik02Ljc2IDExLjIxcTAgLjQ1LjEuNzEuMDcuMjcuMjUuNTguMDYuMS4wNS4xOCAwIC4xMi0uMTUuMjRsLS41LjM0YS40LjQgMCAwIDEtLjIxLjA3cS0uMTIgMC0uMjQtLjExYTMgMyAwIDAgMS0uMjktLjM4bC0uMjQtLjQ3cS0uOTQgMS4xLTIuMzUgMS4xLTEgMC0xLjYtLjU3UTEgMTIuMzIgMSAxMS4zNnEwLTEuMDIuNzMtMS42NGEzIDMgMCAwIDEgMS45NS0uNjJxLjQyIDAgLjg1LjA2LjQ0LjA2LjkyLjE4di0uNTlxMC0uOS0uMzgtMS4yNy0uMzctLjM3LTEuMy0uMzctLjQyIDAtLjg2LjFhNiA2IDAgMCAwLTEuMTQuMzhsLS4xMy4wMnEtLjE3IDAtLjE3LS4yNHYtLjRxMC0uMTguMDYtLjI4YTEgMSAwIDAgMSAuMjItLjE2IDUgNSAwIDAgMSAxLS4zNkE1IDUgMCAwIDEgNCA2LjAycTEuNDMgMCAyLjEuNjQuNjYuNjUuNjYgMS45N3YyLjU4em0tMy4yNCAxLjIycS40IDAgLjgyLS4xNWEyIDIgMCAwIDAgLjc2LS41IDEgMSAwIDAgMCAuMjgtLjUycS4wNy0uMjkuMDgtLjd2LS4zM2E3IDcgMCAwIDAtLjc0LS4xNCA2IDYgMCAwIDAtLjc1LS4wNHEtLjggMC0xLjE5LjMyLS4zOS4zMS0uMzkuOTEgMCAuNTcuMy44NS4yOC4zLjgzLjNtNi40MS44NnEtLjIxIDAtLjMtLjA4LS4xLS4wOC0uMTctLjMxTDcuNiA2LjczIDcuNSA2LjRxMC0uMi4yLS4yaC43OHEuMjMgMCAuMy4wOC4xLjA4LjE3LjNsMS4zNCA1LjMgMS4yNS01LjNxLjA1LS4yMy4xNS0uM2EuNi42IDAgMCAxIC4zMi0uMDhoLjYzcS4yMyAwIC4zMi4wOC4xLjA4LjE2LjNsMS4yNiA1LjM2IDEuMzgtNS4zNXEuMDctLjI1LjE2LS4zMWEuNS41IDAgMCAxIC4zLS4wOGguNzVxLjIgMCAuMi4yIDAgLjA2LS4wMi4xMmwtLjA1LjItMS45MiA2LjE3cS0uMDcuMjUtLjE3LjMxYS41LjUgMCAwIDEtLjMuMDhoLS43cS0uMjEgMC0uMzEtLjA3LS4xLS4wOS0uMTUtLjMzbC0xLjI0LTUuMTQtMS4yMyA1LjE0cS0uMDYuMjQtLjE1LjMyLS4xLjA4LS4zMi4wOHptMTAuMjYuMjFhNSA1IDAgMCAxLTIuMTUtLjQ2cS0uMi0uMTEtLjI0LS4yMmExIDEgMCAwIDEtLjA1LS4yM3YtLjRxMC0uMjUuMTgtLjI1LjA4IDAgLjE0LjAybC4yLjA4YTQgNCAwIDAgMCAxLjgzLjM4cS43NiAwIDEuMTctLjI3YS45LjkgMCAwIDAgLjQxLS43NS44LjggMCAwIDAtLjIxLS41NiAyIDIgMCAwIDAtLjgxLS40MmwtMS4xNi0uMzZhMi40IDIuNCAwIDAgMS0xLjI3LS44MSAyIDIgMCAwIDEtLjQtMS4xNnEwLS41LjIxLS44OS4yMi0uMzguNTgtLjY1LjM1LS4yNy44My0uNDFhMy41IDMuNSAwIDAgMSAxLjU0LS4xbC41Mi4wOC40NS4xM3EuMjEuMDcuMzQuMTRhMSAxIDAgMCAxIC4yNC4yLjQuNCAwIDAgMSAuMDcuMjZ2LjM4cTAgLjI1LS4xOS4yNWExIDEgMCAwIDEtLjMtLjEgNCA0IDAgMCAwLTEuNTMtLjNxLS43IDAtMS4wNi4yMi0uMzguMjItLjM4LjcxIDAgLjM0LjI0LjU3LjI1LjIzLjg4LjQ0bDEuMTMuMzZxLjg4LjI4IDEuMjQuNzYuMzcuNS4zNyAxLjEyIDAgLjUxLS4yMS45M2EyIDIgMCAwIDEtLjU4LjdxLS4zOC4zLS44OS40NS0uNTMuMTYtMS4xNC4xNk0uMzggMTUuNDhhMjQgMjQgMCAwIDAgMTEuODggMy4xNWMyLjkgMCA2LjEtLjYgOS4wNi0xLjg1LjQ0LS4yLjguMjguMzguNi0yLjYzIDEuOTQtNi40NCAyLjk3LTkuNzIgMi45Ny00LjYgMC04Ljc0LTEuNy0xMS44Ny00LjUyLS4yNS0uMjMtLjAzLS41My4yNy0uMzVtMjMuNTMtLjJjLjI5LjM2LS4wOCAyLjgyLTEuNDkgNC0uMjEuMTktLjQyLjA5LS4zMi0uMTVsLjE3LS40NGMuMzQtLjg4LjgtMi4yLjUyLTIuNTUtLjMzLS40My0yLjIyLS4yMS0zLjA3LS4xLS4yNi4wMy0uMy0uMi0uMDctLjM3IDEuNS0xLjA1IDMuOTctLjc1IDQuMjYtLjQiLz48L3N2Zz4=)](https://aws.amazon.com/marketplace/pp/prodview-su2dajk5zawpo)
[![Terraform Module](https://img.shields.io/badge/Terraform-Enterprise%20Edition%20module-844FBA?logo=terraform&logoColor=ffffff)](https://registry.terraform.io/modules/stdapi-ai/stdapi-ai/aws/latest)
[![OpenTofu Module](https://img.shields.io/badge/OpenTofu-Enterprise%20Edition%20module-FFDA18?logo=opentofu&logoColor=ffffff)](https://search.opentofu.org/module/stdapi-ai/stdapi-ai/aws/latest)

[**14-day free trial on AWS Marketplace** · **$0.10/container-hour**](https://stdapi.ai/operations_getting_started/)

---
[![Community Edition Docker image](https://img.shields.io/badge/Docker-Community%20Edition-2496ED?logo=docker&logoColor=ffffff)](https://github.com/stdapi-ai/stdapi.ai/pkgs/container/stdapi.ai-community)
[![Community Edition TrueNAS App](https://img.shields.io/badge/TrueNAS-Community%20Edition-0095D5?logo=truenas&logoColor=ffffff)](https://apps.truenas.com/catalog/stdapi-ai/)
[![Community Edition License](https://img.shields.io/badge/Community%20Edition%20License-AGPL%203.0-BD0000.svg?logo=gplv3&logoColor=ffffff)](LICENSE-AGPL)

[**Free for local development** · **Try Locally with Docker**](#-try-it-locally-with-docker)

---
[![Documentation](https://img.shields.io/badge/Documentation-stdapi.ai-526CFE?logo=materialformkdocs&logoColor=ffffff)](https://stdapi.ai)
[![Samples](https://img.shields.io/badge/Examples-stdapi.ai--samples-24292E?logo=github&logoColor=ffffff)](https://github.com/stdapi-ai/samples)
[![Lint](https://github.com/stdapi-ai/stdapi.ai/actions/workflows/lint.yml/badge.svg)](https://github.com/stdapi-ai/stdapi.ai/actions/workflows/lint.yml)
[![Code quality](https://app.codacy.com/project/badge/Grade/a6036988660e47e7bf59821f55105464)](https://app.codacy.com/gh/stdapi-ai/stdapi.ai/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)

</div>

---

## ⚡ Try It Locally with Docker

Run stdapi.ai locally with the free community image. Requires [AWS credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html) configured (run `aws sso login` first).

```bash
docker run --rm -p 8000:8000 \
  -v ~/.aws:/home/nonroot/.aws:ro \
  -e AWS_BEDROCK_REGIONS=us-east-1,us-west-2 \
  -e ENABLE_DOCS=true \
  ghcr.io/stdapi-ai/stdapi.ai-community:latest
```

> **Podman on Fedora/RHEL (SELinux):** Add `--userns=keep-id` and use `:ro,z` instead of `:ro`

Open **[http://localhost:8000/docs](http://localhost:8000/docs)** in your browser — Swagger UI lets you explore all endpoints and send live requests without writing any code.

Or test from the terminal with `curl`:

**List available models:**

```bash
# All discovered models with capabilities, modalities, and supported routes
curl http://localhost:8000/search_models

# Filter — e.g. vision-capable chat models only
curl "http://localhost:8000/search_models?route=/v1/chat/completions&input_modalities=IMAGE"

# OpenAI-compatible model list
curl http://localhost:8000/v1/models
```

**OpenAI-compatible chat:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "messages": [{"role": "user", "content": "Hello from AWS Bedrock!"}]
  }'
```

**Anthropic-compatible chat:**

```bash
curl http://localhost:8000/anthropic/v1/messages \
  -H "anthropic-version: 2023-06-01" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "max_tokens": 1000,
    "messages": [{"role": "user", "content": "Hello from AWS Bedrock!"}]
  }'
```

**Using the official SDKs?** Point `base_url` at `http://localhost:8000/v1` (OpenAI SDK) or `http://localhost:8000/anthropic` (Anthropic SDK) — no other code changes.

**[Local development guide →](https://stdapi.ai/operations_getting_started_local/)**

---

## 🚀 Production Deployment

Deploy to AWS in minutes with Terraform. The [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-su2dajk5zawpo) subscription includes a **14-day free trial**.

```hcl
module "stdapi_ai" {
  source  = "stdapi-ai/stdapi-ai/aws"
  version = "~> 1.0"
}
```

That's it. You get a production-grade ECS Fargate deployment with HTTPS, WAF, auto-scaling, and monitoring.

**[Full deployment guide →](https://stdapi.ai/operations_getting_started/)** · **[Advanced deployment →](https://stdapi.ai/operations_deploy_advanced/)**

Prefer a hands-off setup? A [managed deployment service](https://aws.amazon.com/marketplace/pp/prodview-xknxzjgl7zi5s) is available.

---

## 🎯 Why stdapi.ai?

- **🔌 Drop-in replacement** — Change only the base URL. Works with LangChain, Continue.dev, Open WebUI, n8n, Claude Code, Aider, and 1000+ tools.
- **🔒 Data stays in your AWS account** — All inference runs in your account. Data never shared with model providers or used for training. Configure allowed regions for GDPR, HIPAA, FedRAMP.
- **🌍 Multiply quota across regions** — Each AWS region has independent quota. 3 regions = 3× tokens per minute. Automatic routing and failover—no client changes.
- **💰 Pay only what you use** — AWS Bedrock rates, no markup, no subscriptions or monthly minimums.
- **⚡ Advanced Bedrock features** — Reasoning modes (Claude, Nova), prompt caching, guardrails, service tiers, inference profiles, prompt routers—all through standard OpenAI and Anthropic API parameters.
- **🧠 80+ models** — Claude, Kimi K2, MiniMax, Qwen, GLM, Nova, Llama, Stability AI, and more. Switch instantly—no vendor lock-in.
- **🎨 Complete multi-modal API** — Chat, embeddings, image generation/editing/variations, audio speech/transcription/translation. Amazon Polly, Transcribe, Translate unified under OpenAI-compatible endpoints.
- **📊 Full observability** — OpenTelemetry integration, request/response logging, Swagger and ReDoc API documentation.
- **🤖 Integrated MCP server** — Every API endpoint exposed as a Model Context Protocol tool. AI agents connect directly—no HTTP client code required. Streamable HTTP and SSE transports with configurable tool selection.
- **🔄 Automatic deprecated model fallback** — When AWS retires a model, requests are transparently redirected to its replacement. Applications survive model deprecations without code changes.

**[See all features →](https://stdapi.ai/features/)**

---

## 💼 Use Cases

| Category | What You Can Build | Tools | Guide |
|---|---|---|---|
| **💬 Chat Interfaces** | Private ChatGPT alternative, team chat, knowledge base with RAG | Open WebUI, LibreChat | [Guide →](https://stdapi.ai/use_cases_openwebui/) |
| **💻 Coding Assistants** | AI pair programming, code completion, codebase chat | Continue.dev, Cline, Cursor, Claude Code, Aider | [Guide →](https://stdapi.ai/use_cases_coding_assistants/) |
| **🔄 Workflow Automation** | AI-powered ticket routing, content creation, data processing | n8n, Make, Zapier | [Guide →](https://stdapi.ai/use_cases_n8n/) |
| **🤖 Chatbots** | Slack/Discord/Teams bots, documentation assistants | Slack Bot, Botpress | |
| **🧠 Autonomous Agents** | Personal AI assistants, research agents, multi-agent systems, code agents | OpenClaw, LangGraph, CrewAI, AutoGPT | |

**[All use cases and integration guides →](https://stdapi.ai/use_cases/)**

---

## 🛒 AWS Marketplace — Commercial Edition

The commercial license via [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-su2dajk5zawpo) is for production, internal tools, and proprietary applications:

- ✅ **14-day free trial** — test in your environment risk-free
- ✅ **Commercial license** — no AGPL obligations, keep your code private
- ✅ **Hardened containers** — security-optimized with regular updates
- ✅ **Terraform module** — production-ready infrastructure in minutes
- ✅ **Streamlined AWS billing** — consolidated with your existing AWS costs

**[Start 14-Day Free Trial →](https://aws.amazon.com/marketplace/pp/prodview-su2dajk5zawpo)**

---

## 🛠️ Development from Source

For contributors working on stdapi.ai itself:

### Prerequisites

- Python 3.14+
- [uv](https://github.com/astral-sh/uv) package manager
- AWS credentials configured

### Setup

```bash
git clone https://github.com/stdapi-ai/stdapi.ai.git
cd stdapi.ai
uv sync --frozen --extra uvicorn

aws sso login --profile your-profile-name

export AWS_BEDROCK_REGIONS=us-east-1
export ENABLE_DOCS=true

uv run uvicorn stdapi.main:app --host 0.0.0.0 --port 8000
```

### Development Guidelines

- Follow existing code style and conventions
- Add tests for new features
- Update documentation for user-facing changes
- Ensure all tests pass before submitting PR

---

## 📜 License

Dual-licensed:

- **[AGPL-3.0-or-later](LICENSE-AGPL)** — Free for open-source projects that share alike
- **[Commercial License](https://aws.amazon.com/marketplace/pp/prodview-su2dajk5zawpo)** — AWS Marketplace, for proprietary applications (14-day free trial)

**[Learn more about licensing →](https://stdapi.ai/operations_licensing/)**

---

## 🤝 Contributing

We welcome contributions! Whether it's bug reports, new features, documentation improvements, or ideas — please feel free to open issues or submit pull requests.

---

## 💬 Support

- **Issues:** [GitHub Issue Tracker](https://github.com/stdapi-ai/stdapi.ai/issues)
- **Documentation:** [stdapi.ai](https://stdapi.ai)
- **Sponsor:** [GitHub Sponsors](https://github.com/sponsors/JGoutin) — Priority support, feature prioritization, and influence on the roadmap. [View tiers →](https://github.com/sponsors/JGoutin)

---

<div align="center">

**Ready to run 80+ AI models securely on AWS?**

[Start 14-Day Free Trial on AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-su2dajk5zawpo) · [Deployment guide](https://stdapi.ai/operations_getting_started/) · [Terraform examples](https://github.com/stdapi-ai/samples) · [Try Locally with Docker](#-try-it-locally-with-docker)

</div>
