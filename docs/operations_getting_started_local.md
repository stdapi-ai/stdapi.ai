---
title: Local Development - Run stdapi.ai with Docker
description: Run stdapi.ai locally with Docker or Podman for development, testing, and evaluation. Free community image with full API compatibility for Amazon Bedrock models.
keywords: Docker OpenAI gateway, local AI development, Podman AI gateway, free OpenAI alternative, local AWS Bedrock, Docker AI API, community AI gateway, AGPL AI gateway
---

# :material-docker: Local Development with Docker/Podman

Run stdapi.ai locally for development, testing, and evaluation using the free community container image (AGPL-3.0). Full API compatibility — the same endpoints and features as the production deployment.

!!! tip "New to Amazon Bedrock?"
    To run stdapi.ai locally you need:

    1. An **AWS account** — [create one free](https://aws.amazon.com/free/)
    2. **AWS credentials** configured locally — via `aws configure` or `aws sso login` ([AWS CLI setup guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html))

---

## :material-docker: Run It

**With AWS credentials (after `aws sso login`):**

```bash
docker run --rm -p 8000:8000 \
  -v ~/.aws:/home/nonroot/.aws:ro \
  -e AWS_BEDROCK_REGIONS=us-east-1,us-west-2 \
  -e ENABLE_DOCS=true \
  ghcr.io/stdapi-ai/stdapi.ai-community:latest
```

**With environment variables instead:**

```bash
docker run --rm -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID=your-access-key-id \
  -e AWS_SECRET_ACCESS_KEY=your-secret-access-key \
  -e AWS_SESSION_TOKEN=your-session-token \
  -e AWS_BEDROCK_REGIONS=us-east-1,us-west-2 \
  -e ENABLE_DOCS=true \
  ghcr.io/stdapi-ai/stdapi.ai-community:latest
```

??? info "Podman on Fedora/RHEL with SELinux"

    Add the `:z` SELinux label and `--userns=keep-id`:

    ```bash
    podman run --rm -p 8000:8000 \
      --userns=keep-id \
      -v ~/.aws:/home/nonroot/.aws:ro,z \
      -e AWS_BEDROCK_REGIONS=us-east-1,us-west-2 \
      -e ENABLE_DOCS=true \
      ghcr.io/stdapi-ai/stdapi.ai-community:latest
    ```

    The `:z` flag relabels files for container access. Use `:Z` if multiple containers share the volume. `--userns=keep-id` maps your host user ID to the container user.

    See also [Troubleshooting → Podman volume mount fails on Fedora/RHEL with SELinux](operations_troubleshooting.md#terraform-deployment) if you hit this after the fact.

```mermaid
%%{init: {'flowchart': {'htmlLabels': true}} }%%
flowchart LR
  openai["<img src='../styles/logo_openai.svg' style='height:64px;width:auto;vertical-align:middle;' /> OpenAI SDK"] --> local["<img src='../styles/logo.svg' style='height:64px;width:auto;vertical-align:middle;' /> stdapi.ai (community)<br/>Docker/Podman"]
  anthropic["<img src='../styles/logo_anthropic.svg' style='height:64px;width:auto;vertical-align:middle;' /> Anthropic SDK"] --> local
  local --> bedrock["<img src='../styles/logo_amazon_bedrock.svg' style='height:64px;width:auto;vertical-align:middle;' /> Amazon Bedrock"]
  local --> polly["<img src='../styles/logo_amazon_polly.svg' style='height:64px;width:auto;vertical-align:middle;' /> Amazon Polly"]
  local --> transcribe["<img src='../styles/logo_amazon_transcribe.svg' style='height:64px;width:auto;vertical-align:middle;' /> Amazon Transcribe"]
  local --> s3["<img src='../styles/logo_amazon_s3.svg' style='height:64px;width:auto;vertical-align:middle;' /> Amazon S3"]
```

---

## :material-check-circle: Test It

```bash
# Check health
curl http://localhost:8000/health

# List all available models
curl http://localhost:8000/search_models

# Search models by capability (e.g. streaming-capable chat models only)
curl "http://localhost:8000/search_models?route=/v1/chat/completions&streaming=true"

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.nova-micro-v1:0",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**Interactive API docs:** Open [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI with all available endpoints.

!!! tip "Try other models"
    `amazon.nova-micro-v1:0` is a fast, low-cost model — great for confirming the pipeline works. Once you see a response, switch the `model` field to `anthropic.claude-fable-5`, `anthropic.claude-sonnet-5`, or any other Bedrock model available in your configured regions.

    Use `GET /search_models` (shown above) to discover what's available and filter by capability, or `GET /v1/models` for strict OpenAI SDK compatibility — see the [Search Models API](api_search_models.md) reference.

---

## :material-information: Technical Notes

**Building from source:** See the [Dockerfile](https://github.com/stdapi-ai/stdapi.ai/blob/main/Dockerfile) if you prefer to build the image yourself.

**Container runtime:** Both community and Marketplace images use [Granian](https://github.com/emmett-framework/granian), a high-performance Python ASGI server. Granian environment variables (e.g., `GRANIAN_PORT`, `GRANIAN_WORKERS`) are supported.

**Configuration:** See [Configuration Reference](operations_configuration.md) for all environment variables.

!!! tip "Ready for Production?"
    When you're ready to deploy to AWS with HTTPS, auto-scaling, and enterprise features, the [production deployment guide](operations_getting_started.md) gets you running in 5 minutes with Terraform. The [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-su2dajk5zawpo) subscription includes a **14-day free trial**.

---

## :material-arrow-right: Next Steps

stdapi.ai works with any OpenAI or Anthropic-compatible tool. Here are popular integrations to try locally:

<div class="grid cards" markdown>

- :material-chat: [**Open WebUI**](use_cases_openwebui.md) — Private ChatGPT-like interface with RAG, multi-modal support, and document upload
- :material-robot: [**n8n Workflows**](use_cases_n8n.md) — AI-powered automation with 400+ integrations
- :material-code-braces: [**AI Coding Assistants**](use_cases_coding_assistants.md) — Claude Code, Cline, OpenCode, Zed with Amazon Bedrock models
- :material-book-open-variant: [**API Overview**](api_overview.md) — All endpoints, parameters, and usage examples
- :material-rocket-launch: [**Getting Started**](operations_getting_started.md) — Deploy the production stack on AWS with Terraform
- :material-wrench: [**Troubleshooting**](operations_troubleshooting.md) — Podman/SELinux, auth, model-not-found, and other common errors

</div>
