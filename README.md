<div align="center">
<img src="docs/styles/logo.svg" alt="stdapi.ai logo" width="40%" />

# stdapi.ai

**The OpenAI-compatible API for AWS AI**

Access 80+ Bedrock models (Claude, Llama, Nova) and AWS AI services (Polly, Transcribe, Translate) with your existing OpenAI clients—no code changes required.

[![AWS Marketplace](https://img.shields.io/badge/AWS-Marketplace-FF9900?logo=amazon-aws&logoColor=white)](https://aws.amazon.com/marketplace/pp/prodview-su2dajk5zawpo)
[![Documentation](https://img.shields.io/badge/docs-stdapi.ai-blue)](https://stdapi.ai)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE-AGPL)
</div>

---

## 🚀 What is stdapi.ai?

**stdapi.ai** provides AWS AI infrastructure through an OpenAI-compatible interface. Connect your existing applications to over 40 Amazon Bedrock models and AWS AI services without changing code.

### Why Choose stdapi.ai?

- **🔌 OpenAI Compatible** – Applications designed for OpenAI's API work with stdapi.ai. Just change the endpoint.
- **🔒 Your Data, Your AWS Account** – All conversations and data remain within your AWS environment. For HIPAA, GDPR, and regulatory compliance.
- **🌍 Multi-Region Architecture** – Access models across multiple AWS Bedrock regions through one unified endpoint.
- **💰 Cost Optimized** – Pay standard AWS rates with zero markup.
- **⚡ Enterprise Ready** – CloudWatch integration, native guardrails, hardened container images, and production-ready infrastructure.

---

## 🎯 Key Features

### 80+ AI Models
Access models from:
- **Anthropic** (Claude for reasoning and coding)
- **Amazon** (Nova family for cost-effective performance)
- **OpenAI**, **Meta** (Llama), **Mistral AI**, **Google**, **Cohere**, **Stability AI**, **DeepSeek**, **Qwen**, and more

### Comprehensive AWS AI Services
Beyond Bedrock language models:
- **Amazon Polly** – Natural text-to-speech synthesis
- **Amazon Transcribe** – Accurate speech recognition
- **Amazon Translate** – Multi-language support

### Multi-Modal Capabilities
- Text conversations
- Image generation
- Audio processing
- Embeddings for semantic search
- Complete AI workflows

---

## 💼 Popular Use Cases

| Use Case                 | Description                                                                       |
|--------------------------|-----------------------------------------------------------------------------------|
| **Chat Interfaces**      | Connect OpenWebUI or LibreChat to Bedrock models for private ChatGPT alternatives |
| **Workflow Automation**  | Connect AI to 400+ services using N8N's visual automation platform                |
| **Developer Tools**      | AI assistance in VS Code, JetBrains IDEs, Cursor with LangChain and LlamaIndex    |
| **Knowledge Management** | Add AI-powered writing and semantic search to Obsidian and Notion                 |
| **Communication Bots**   | Deploy bots to Slack, Discord, Teams, and Telegram                                |

---

## 🛒 AWS Marketplace

### Enterprise Deployment Made Easy

**stdapi.ai is available on AWS Marketplace** with commercial licensing and production-ready deployment options.

#### Advantages of AWS Marketplace Version:
- ✅ **Commercial License** – Use in proprietary applications without AGPL obligations
- ✅ **Hardened Container Images** – Security-optimized for production workloads
- ✅ **Regular Security Updates** – Timely patches and vulnerability fixes
- ✅ **Streamlined Deployment** – Quick deployment with Terraform/OpenTofu modules
- ✅ **AWS Best Practices** – Following AWS Well-Architected Framework guidelines
- ✅ **Enterprise Support** – Professional support for deployment and configuration

**[Deploy from AWS Marketplace →](https://aws.amazon.com/marketplace/pp/prodview-su2dajk5zawpo)**

---

## 📖 Quick Start

### For Production Use

Deploy stdapi.ai to your AWS account in minutes using our Terraform module:

```hcl
module "stdapi_ai" {
  source  = "stdapi-ai/stdapi-ai/aws"
  version = "~> 1.0"
}
```

Then make your first API call:

```python
from openai import OpenAI

client = OpenAI(
    api_key="YOUR_API_KEY",
    base_url="https://YOUR_DEPLOYMENT_URL/v1"
)

response = client.chat.completions.create(
    model="anthropic.claude-sonnet-4-5-20250929-v1:0",
    messages=[{"role": "user", "content": "Hello from AWS!"}]
)

print(response.choices[0].message.content)
```

**📚 [Full Deployment Guide →](https://stdapi.ai/operations_getting_started/)**

---

## 🛠️ Local Development Setup

### Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- AWS credentials configured

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/stdapi-ai/stdapi.ai.git
   cd stdapi.ai
   ```

2. **Install dependencies**
   ```bash
   uv sync
   ```

3. **Login to AWS**
   ```bash
   # Login using AWS SSO
   aws sso login --profile your-profile-name

   # Or configure your default profile
   aws configure sso
   ```

4. **Configure the application**
   ```bash
   # Core AWS Configuration (auto-detects current region if not set)
   export AWS_BEDROCK_REGIONS=us-east-1  # Optional: defaults to current AWS region

   # S3 Storage (required for certain features like image generation, audio)
   export AWS_S3_BUCKET=my-dev-bucket  # Create bucket in same region as AWS_BEDROCK_REGIONS

   # API Authentication (optional for local dev)
   export API_KEY=sk-dev-your-secret-key-here

   # Enable API documentation (helpful for development)
   export ENABLE_DOCS=true
   export ENABLE_REDOC=true

   # Logging Configuration
   export LOG_LEVEL=info  # Options: info, warning, error, critical, disabled
   export LOG_REQUEST_PARAMS=true  # Enable detailed request/response logging for debugging
   ```

5. **Run locally**
   ```bash
   uv run stdapi
   ```

6. **Test the API**
   ```bash
   curl http://localhost:8000/v1/models
   ```

### Development Guidelines

- Follow existing code style and conventions
- Add tests for new features
- Update documentation for user-facing changes
- Ensure all tests pass before submitting PR

---

## 📚 Documentation

- **[Official Documentation](https://stdapi.ai)** – Complete guides and API reference
- **[Getting Started](https://stdapi.ai/operations_getting_started/)** – Deployment and configuration
- **[API Reference](https://stdapi.ai/api_overview/)** – Detailed API documentation
- **[Licensing](https://stdapi.ai/operations_licensing/)** – AGPL vs Commercial licensing

---

## 📜 License

This project is dual-licensed:

- **[AGPL-3.0-or-later](LICENSE-AGPL)** – Free for open-source projects that share alike
- **Commercial License** – Available via [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-su2dajk5zawpo) for proprietary applications

The AWS Marketplace version provides full commercial rights, no source disclosure requirements, and production-ready infrastructure.

[Learn more about licensing →](https://stdapi.ai/operations_licensing/)

---

## 🤝 Contributing

We welcome contributions! Whether it's:

- 🐛 Bug reports and fixes
- ✨ New features and enhancements
- 📖 Documentation improvements
- 💡 Ideas and suggestions

Please feel free to open issues or submit pull requests.

---

## 💬 Support

- **🐛 Issues:** [GitHub Issue Tracker](https://github.com/stdapi-ai/stdapi.ai/issues)
- **📖 Documentation:** [stdapi.ai](https://stdapi.ai)
- **💖 Sponsor:** [GitHub Sponsors](https://github.com/sponsors/JGoutin) – Support the project's development

Sponsorship benefits include priority support, feature prioritization, dedicated development time, SLA for critical issues, and influence on the project roadmap. [View sponsorship tiers →](https://github.com/sponsors/JGoutin)

---

## 🌟 Why stdapi.ai?

| For Developers                                                                                                                   | For Enterprises                                                                                                           |
|----------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| • Zero code changes required<br>• Works with existing OpenAI SDKs<br>• Simple endpoint swap<br>• Comprehensive API compatibility | • Data sovereignty & compliance<br>• Cost transparency & savings<br>• No vendor lock-in<br>• Enterprise-grade reliability |

---

<div align="center">

**Get Started with AWS AI**

[Documentation](https://stdapi.ai) • [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-su2dajk5zawpo) • [GitHub Issues](https://github.com/stdapi-ai/stdapi.ai/issues)

Made with ❤️ for the AWS and AI community

</div>