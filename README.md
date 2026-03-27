<div align="center">
<img src="docs/styles/logo.svg" alt="stdapi.ai logo" width="40%" />

# stdapi.ai

**OpenAI & Anthropic Compatible API Gateway for AWS Bedrock and AI Services**

Run your favorite OpenAI and Anthropic-compatible applications on AWS Bedrock. Access 80+ models including Claude, Kimi K2, MiniMax M2.5, Qwen3 with enterprise privacy, compliance controls, and pay-per-use AWS pricing.

[![AWS Marketplace](https://img.shields.io/badge/AWS-Marketplace-FF9900?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/marketplace/pp/prodview-su2dajk5zawpo)
[![Documentation](https://img.shields.io/badge/docs-stdapi.ai-blue)](https://stdapi.ai)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE-AGPL)

**14-day free trial on AWS Marketplace** · **Free for local development**

[Start 14-Day Free Trial](https://stdapi.ai/operations_getting_started/) · [Try Locally with Docker](#-try-it-locally-with-docker)

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

Then make your first API call:

**OpenAI SDK:**

```python
from openai import OpenAI

client = OpenAI(
    api_key="not-needed-locally",
    base_url="http://localhost:8000/v1"
)

response = client.chat.completions.create(
    model="anthropic.claude-opus-4-6-v1",
    messages=[{"role": "user", "content": "Hello from AWS Bedrock!"}]
)

print(response.choices[0].message.content)
```

**Anthropic SDK:**

```python
from anthropic import Anthropic

client = Anthropic(
    api_key="not-needed-locally",
    base_url="http://localhost:8000/anthropic"
)

message = client.messages.create(
    model="anthropic.claude-opus-4-6-v1",
    messages=[{"role": "user", "content": "Hello from AWS Bedrock!"}]
)

print(message.content[0].text)
```

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
- **⚡ Advanced Bedrock features** — Reasoning modes (Claude 4.6+, Nova 2), prompt caching, guardrails, service tiers, inference profiles, prompt routers—all through standard OpenAI and Anthropic API parameters.
- **🧠 80+ models** — Claude, Kimi K2, MiniMax M2.5, Qwen3, GLM 5, Nova 2, Llama 4, Stability AI, and more. Switch instantly—no vendor lock-in.
- **🎨 Complete multi-modal API** — Chat, embeddings, image generation/editing/variations, audio speech/transcription/translation. Amazon Polly, Transcribe, Translate unified under OpenAI-compatible endpoints.
- **📊 Full observability** — OpenTelemetry integration, request/response logging, Swagger and ReDoc API documentation.

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

[Start 14-Day Free Trial](https://stdapi.ai/operations_getting_started/) · [Try Locally with Docker](#-try-it-locally-with-docker)

</div>
