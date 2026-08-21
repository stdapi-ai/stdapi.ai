---
title: Open WebUI Integration - Amazon Bedrock ChatGPT Alternative
description: Deploy Open WebUI with Amazon Bedrock using stdapi.ai. Complete setup guide for private ChatGPT alternative with RAG, voice, images, and multi-modal AI capabilities.
keywords: Open WebUI AWS, private ChatGPT, ChatGPT alternative, self-hosted ChatGPT, AWS Bedrock ChatGPT, enterprise chat interface, RAG chatbot AWS, multi-modal chat
---

# :material-chat: Open WebUI Integration

Connect Open WebUI to stdapi.ai as an OpenAI-compatible backend. Access Amazon Bedrock models through Open WebUI's chat interface—it works out of the box as a private ChatGPT alternative running on your AWS infrastructure.

## :material-information-outline: About Open WebUI

**🔗 Links:** [Website](https://openwebui.com/) | [GitHub](https://github.com/open-webui/open-webui) | [Documentation](https://docs.openwebui.com/)

Open WebUI is the leading open-source ChatGPT alternative. It provides a feature-rich, self-hosted web interface that operates entirely under your control, offering a ChatGPT-like experience while maintaining complete data privacy.

**Key Features:**

- ⭐ **140,000+ GitHub stars** - Most popular open-source AI chat interface
- **ChatGPT-like UI** - Familiar interface your team already knows
- **Multi-modal capabilities** - Text, voice, images, and document processing
- **RAG & embeddings** - Upload documents, search with semantic understanding
- **Extensible platform** - Plugins, custom functions, and community tools
- **Complete privacy** - Self-hosted, all data stays in your infrastructure

## :material-help-circle-outline: Why Open WebUI + stdapi.ai?

<div class="grid cards" markdown>

- :material-swap-horizontal: __Two Client-Side Changes__
  <br>stdapi.ai provides an OpenAI-compatible API. Update the endpoint URL, then pick a model—Open WebUI's model list is whatever your deployment serves, across every provider in the catalogue.

- :material-aws: __Access Amazon Bedrock Models__
  <br>Claude with reasoning, Nova, Llama, DeepSeek, Stable Diffusion, and 100+ models through Open WebUI's familiar chat interface.

- :material-application-cog: __Full Multi-Modal Support__
  <br>Text chat, voice input/output, image generation/editing, document RAG—all AWS AI services unified through one interface.

- :material-lock: __Enterprise Data Privacy__
  <br>All processing stays in your AWS account. Complete infrastructure control with AWS security, compliance, and data sovereignty.

- :material-currency-usd-off: __Pay-Per-Use Pricing__
  <br>No ChatGPT subscriptions. Pay only Amazon Bedrock rates for actual usage—no monthly minimums or per-user fees.

</div>

```mermaid
%%{init: {'flowchart': {'htmlLabels': true}} }%%
flowchart LR
  openwebui["<img src='../styles/logo_openwebui.svg' style='height:64px;width:auto;vertical-align:middle;' /> Open WebUI"] --> stdapi["<img src='../styles/logo.svg' style='height:64px;width:auto;vertical-align:middle;' /> stdapi.ai"]
  stdapi --> bedrock["<img src='../styles/logo_amazon_bedrock.svg' style='height:64px;width:auto;vertical-align:middle;' /> Amazon Bedrock"]
  stdapi --> transcribe["<img src='../styles/logo_amazon_transcribe.svg' style='height:64px;width:auto;vertical-align:middle;' /> Amazon Transcribe"]
  stdapi --> polly["<img src='../styles/logo_amazon_polly.svg' style='height:64px;width:auto;vertical-align:middle;' /> Amazon Polly"]
```

## :material-sitemap: Architecture

The diagram below is the topology the [Terraform sample](#terraform-deployment) builds: a browser-facing chat application and its AI gateway, both on ECS Fargate, in one VPC you own.

```mermaid
%%{init: {'flowchart': {'htmlLabels': true}} }%%
flowchart LR
  user["👤 Your users<br/>(browser)"]

  subgraph public["Your VPC · public subnets"]
    alb["<img src='../styles/logo_amazon_load_balancing.svg' style='height:40px;width:auto;vertical-align:middle;' /> Application Load Balancer<br/>HTTPS · ACM certificate"]
  end

  subgraph private["Your VPC · private app subnets — no inbound route from the internet"]
    openwebui["<img src='../styles/logo_openwebui.svg' style='height:40px;width:auto;vertical-align:middle;' /> Open WebUI<br/>ECS Fargate"]
    stdapi["<img src='../styles/logo.svg' style='height:40px;width:auto;vertical-align:middle;' /> stdapi.ai<br/>ECS Fargate"]
    tools["SearXNG · Playwright<br/>ECS Fargate"]
    aurora["Aurora PostgreSQL<br/>Serverless v2 + pgvector"]
    valkey["ElastiCache Valkey<br/>TLS + auth token"]
    egress["NAT gateways<br/>or interface VPC endpoints"]
  end

  subgraph regional["AWS service endpoints · your account, the regions you configure"]
    bedrock["<img src='../styles/logo_amazon_bedrock.svg' style='height:40px;width:auto;vertical-align:middle;' /> Amazon Bedrock"]
    polly["<img src='../styles/logo_amazon_polly.svg' style='height:40px;width:auto;vertical-align:middle;' /> Amazon Polly"]
    transcribe["<img src='../styles/logo_amazon_transcribe.svg' style='height:40px;width:auto;vertical-align:middle;' /> Amazon Transcribe"]
    s3["<img src='../styles/logo_amazon_s3.svg' style='height:40px;width:auto;vertical-align:middle;' /> Amazon S3<br/>SSE-KMS"]
    cw["<img src='../styles/logo_amazon_cloudwatch.svg' style='height:40px;width:auto;vertical-align:middle;' /> Amazon CloudWatch<br/>logs · metrics · alarms"]
  end

  user -->|"HTTPS · TLS 1.2+"| alb
  alb -->|"HTTP · private subnet"| openwebui
  openwebui -->|"OpenAI + Cohere API · API key<br/>private DNS, no public endpoint"| stdapi
  openwebui --> aurora
  openwebui --> valkey
  openwebui --> tools
  openwebui --> egress
  stdapi --> egress
  egress -->|"HTTPS · SigV4"| bedrock
  egress -->|"HTTPS · SigV4"| polly
  egress -->|"HTTPS · SigV4"| transcribe
  egress -->|"S3 gateway endpoint"| s3
  egress --> cw
```

Two properties of this topology are worth reading off the picture. The gateway has **no listener of its own**: Open WebUI resolves it through AWS Cloud Map private DNS inside the VPC, so the load balancer is the only thing with a public address, and it only ever forwards to Open WebUI. And every store that holds your users' content — the Aurora database, the Valkey cache, the S3 bucket — sits inside the account boundary; nothing in the picture is operated by a third party.

### What Each AWS Service Does Here

| AWS service | Role in this integration | Where it is configured |
| --- | --- | --- |
| **Amazon ECS on AWS Fargate** | Runs Open WebUI, the stdapi.ai gateway, SearXNG and Playwright as separate services with independent auto-scaling | Terraform sample |
| **Elastic Load Balancing** | The single public entry point; terminates TLS with an ACM certificate and forwards only to Open WebUI | Terraform sample |
| **AWS Cloud Map** | Private DNS name that lets Open WebUI reach the gateway without exposing it | Terraform sample (`service_discovery_dns_name`) |
| **Amazon Bedrock** | Chat completions, embeddings, reranking, image generation and editing | [`AWS_BEDROCK_REGIONS`](operations_configuration.md#aws-bedrock-regions) |
| **Amazon Transcribe** | Voice input, behind `POST /v1/audio/transcriptions` | `AUDIO_STT_MODEL` (above) |
| **Amazon Polly** | Spoken replies, behind `POST /v1/audio/speech` | `AUDIO_TTS_MODEL` (above) |
| **Amazon Aurora PostgreSQL** | Open WebUI's own database, and the pgvector store its RAG pipeline queries; Serverless v2, storage encrypted | Terraform sample |
| **Amazon ElastiCache (Valkey)** | WebSocket session state and model-list cache, reached over `rediss://` with an auth token | Terraform sample |
| **Amazon S3** | Open WebUI file uploads under an `openwebui/` prefix, plus the gateway's temporary multimodal objects | [S3 storage](operations_compliance.md#s3-data-storage) |
| **AWS KMS** | Customer-managed keys encrypting the S3 bucket and the Aurora cluster storage | Terraform sample |
| **AWS Secrets Manager** | Holds the Aurora master password; the task reads it at start-up | Terraform sample |
| **Amazon CloudWatch** | Container logs, Container Insights, gateway request logs and EMF usage metrics | [Logging & monitoring](operations_logging_monitoring.md) |
| **AWS IAM** | Separate least-privilege task roles for each service; the gateway's role grants only the model and AI-service actions it invokes | [IAM permissions](operations_iam_permissions.md) |

### Security Measures in This Flow

- **Authentication** — Open WebUI signs in its own users; every call it then makes to the gateway carries a stdapi.ai [API key](operations_authentication_security.md#api-key-authentication) that Terraform generates and injects into the container environment.
- **Encryption in transit** — HTTPS from the browser to the ALB, whose listener supports TLS 1.2 and 1.3; private-VPC traffic from the ALB to the container; HTTPS with SigV4 from the gateway to each AWS service.
- **Encryption at rest** — SSE-KMS on the S3 bucket, encrypted Aurora storage, and TLS plus an auth token on the Valkey connection.
- **Least privilege** — each ECS task assumes its own role; the gateway's role carries no permission for the Aurora cluster, and Open WebUI's role carries none for Amazon Bedrock.
- **Content policy** — a [Bedrock guardrail](operations_configuration.md#bedrock-guardrails) configured on the gateway applies to each route Open WebUI uses, not only to chat, and stays in force unless the deployment explicitly allows a per-request override.
- **Data handling** — the gateway is stateless and holds request bodies in memory only; CloudWatch receives request metadata, not prompts, unless payload logging is explicitly turned on for debugging.

## :material-check-circle: Prerequisites

!!! info "What You'll Need"
    - ✓ **stdapi.ai deployed** - [See deployment guide](operations_getting_started.md)
    - ✓ **Your stdapi.ai URL** - e.g., `https://api.example.com`
    - ✓ **Your API key** - From Terraform output or configuration
    - ✓ **Open WebUI instance** - Running or ready to deploy (see Deployment section below)

---

## :material-cog: Configuration

Open WebUI is configured entirely through environment variables. The sections below focus on the stdapi.ai integration. Use the same stdapi.ai key for all `*_OPENAI_API_KEY` entries. For more details on Open WebUI settings, refer to the official [Open WebUI Environment Variable Configuration](https://docs.openwebui.com/getting-started/env-configuration/) documentation.

!!! warning "Each section needs its own connection settings"
    Open WebUI does not fall back from `RAG_OPENAI_*`, `IMAGES_OPENAI_*`, or `AUDIO_*_OPENAI_*` to the core `OPENAI_API_*` pair — a missing pair disables that feature instead of inheriting the Core Connection. Set the base URL, key, and model explicitly for every section you enable.

!!! warning "These settings are read once, on first boot"
    Open WebUI reads its connection settings from the environment only the first time it starts against a given data directory, then stores them in its own database. Changing an environment variable afterwards has no effect until you either update the setting from the admin UI or start from a fresh `DATA_DIR`.

!!! note "Model choice"
    In every section below, pick any Bedrock-available model that matches the operation's modality — a chat model for the Core Connection, an embedding model for RAG Embeddings, and so on.

### :material-connection: Core Connection

Enables: Chat completions and Open WebUI background tasks (titles, summarization).

!!! example "Environment Variables"
    ```bash
    OPENAI_API_BASE_URL=https://YOUR_STDAPI_URL/v1
    OPENAI_API_KEY=YOUR_STDAPI_KEY
    TASK_MODEL_EXTERNAL=amazon.nova-micro-v1:0
    ```

Use a fast, low-cost chat model for `TASK_MODEL_EXTERNAL`. Open WebUI calls `POST /v1/chat/completions` for chat and background tasks (see [Chat Completions API](api_openai_chat_completions.md)).

### :material-database: RAG Embeddings

Enables: Document ingestion and semantic search for RAG.

!!! example "Environment Variables"
    ```bash
    RAG_EMBEDDING_ENGINE=openai
    RAG_OPENAI_API_BASE_URL=https://YOUR_STDAPI_URL/v1
    RAG_OPENAI_API_KEY=YOUR_STDAPI_KEY
    RAG_EMBEDDING_MODEL=cohere.embed-v4:0
    ```

Open WebUI calls `POST /v1/embeddings` (see [Embeddings API](api_openai_embeddings.md)).

### :material-sort-variant: RAG Reranking

Enables: Hybrid search, with retrieved documents reordered by relevance before they reach the model.

!!! example "Environment Variables"
    ```bash
    ENABLE_RAG_HYBRID_SEARCH=true
    RAG_RERANKING_ENGINE=external
    RAG_EXTERNAL_RERANKER_URL=https://YOUR_STDAPI_URL/cohere/v2/rerank
    RAG_EXTERNAL_RERANKER_API_KEY=YOUR_STDAPI_KEY
    RAG_RERANKING_MODEL=cohere.rerank-v3-5:0
    ```

Open WebUI's external reranker speaks the Cohere dialect, so it targets the Cohere-compatible route instead of `/v1` (see [Cohere Rerank API](api_cohere_rerank.md)). Give the full endpoint path: Open WebUI sends the request to the URL as-is and appends nothing.

!!! tip "Regional availability"
    Amazon Bedrock serves reranking from a subset of regions only. Keep at least one of them in [`AWS_BEDROCK_REGIONS`](operations_configuration.md#aws-bedrock-regions); stdapi.ai fails over to it automatically.

Without an external reranker, Open WebUI falls back to a local Sentence-Transformers cross-encoder that it downloads from Hugging Face at startup—unavailable when `OFFLINE_MODE` is enabled.

!!! tip "Chunk size determines whether reranking has anything to do"
    Set `CHUNK_SIZE` small enough that each retrieval chunk covers one self-contained idea. A chunk size large enough to fold a whole document into a single chunk leaves the reranker nothing to reorder—there is only one candidate to rank.

### :material-image: Image Generation

Enables: Text-to-image creation inside chats.

!!! example "Environment Variables"
    ```bash
    ENABLE_IMAGE_GENERATION=true
    IMAGE_GENERATION_ENGINE=openai
    IMAGES_OPENAI_API_BASE_URL=https://YOUR_STDAPI_URL/v1
    IMAGES_OPENAI_API_KEY=YOUR_STDAPI_KEY
    IMAGE_GENERATION_MODEL=stability.stable-image-core-v1:1
    ```

Open WebUI calls `POST /v1/images/generations` (see [Images Generations API](api_openai_images_generations.md)).

### :material-image-edit: Image Editing

Use Open WebUI's image editor to upload an image and describe the change. Masking is not configured.

Enables: Image edits and transformations in the editor.

!!! example "Environment Variables"
    ```bash
    ENABLE_IMAGE_EDIT=true
    IMAGE_EDIT_ENGINE=openai
    IMAGES_EDIT_OPENAI_API_BASE_URL=https://YOUR_STDAPI_URL/v1
    IMAGES_EDIT_OPENAI_API_KEY=YOUR_STDAPI_KEY
    IMAGE_EDIT_MODEL=stability.stable-image-control-structure-v1:0
    ```

Pick any image-editing model that supports edits without a mask. Open WebUI calls `POST /v1/images/edits` (see [Images Edits API](api_openai_images_edits.md)).

### :material-microphone: Speech to Text (STT)

Enables: Voice input and audio transcription.

!!! example "Environment Variables"
    ```bash
    AUDIO_STT_ENGINE=openai
    AUDIO_STT_OPENAI_API_BASE_URL=https://YOUR_STDAPI_URL/v1
    AUDIO_STT_OPENAI_API_KEY=YOUR_STDAPI_KEY
    AUDIO_STT_MODEL=amazon.transcribe
    ```

Open WebUI calls `POST /v1/audio/transcriptions` (see [Audio Transcriptions API](api_openai_audio_transcriptions.md)).

!!! tip "A cheaper transcription model"
    Setting `AUDIO_STT_MODEL=amazon.nova-2-sonic-v1:0` transcribes through [Amazon Nova Sonic](api_openai_audio_transcriptions.md#amazon-nova-sonic), the lowest-cost option here, punctuated and in the language spoken. It returns plain text with no timestamps and takes recordings up to 10 minutes — ample for chat voice input, but keep `amazon.transcribe` if you also transcribe long meeting recordings from the same setting.

### :material-volume-high: Text to Speech (TTS)

Enables: Spoken responses from chat outputs.

!!! example "Environment Variables"
    ```bash
    AUDIO_TTS_ENGINE=openai
    AUDIO_TTS_OPENAI_API_BASE_URL=https://YOUR_STDAPI_URL/v1
    AUDIO_TTS_OPENAI_API_KEY=YOUR_STDAPI_KEY
    AUDIO_TTS_MODEL=amazon.polly-neural
    ```

Open WebUI calls `POST /v1/audio/speech` (see [Audio Speech API](api_openai_audio_speech.md)).

!!! warning "TTS language detection"
    Open WebUI generates audio in small chunks, which makes language auto-detection inconsistent. Disable auto-detection by setting the stdapi.ai environment variable `DEFAULT_TTS_LANGUAGE` to a fixed language (for example, `en-US`).

### :material-tools: MCP Tool Server

Enables: Chat models calling stdapi.ai endpoints as tools, including the ones Open WebUI has no native feature for—such as video generation.

Open WebUI supports MCP servers over Streamable HTTP (v0.6.31 and later), the transport stdapi.ai exposes at `/mcp`. Turn it on with the stdapi.ai environment variables:

!!! example "Environment Variables (stdapi.ai)"
    ```bash
    ENABLE_MCP_STREAMABLE_HTTP=true
    MCP_INCLUDE_TOOLS=openai_video_generation,openai_video_get,search_models
    ```

Every endpoint is exposed as a tool by default. Restrict the list with [`MCP_INCLUDE_TOOLS`](operations_configuration.md#mcp-include-tools) so models see only the tools they need, and keep the ones already wired natively—chat, images, speech, embeddings—out of it.

Then register the server in Open WebUI. MCP connections are admin-only and configured in the interface, not through environment variables:

1. Open **Admin Settings → External Tools**
2. Click **+ (Add Server)** and set **Type** to **MCP (Streamable HTTP)**
3. Set the server URL to `https://YOUR_STDAPI_URL/mcp`
4. Set **Auth** to **Bearer** and paste your stdapi.ai API key
5. Save, then enable the tool in a chat with **+ → Integrations → Tools**

See [MCP tools](api_overview.md#mcp-model-context-protocol) for the full tool list.

---

## :material-rocket-launch: Terraform Deployment

Deploy Open WebUI + stdapi.ai together with production infrastructure:

**📦 [stdapi-ai/samples/getting_started_openwebui](https://github.com/stdapi-ai/samples/tree/main/getting_started_openwebui)**

**What's included:**

- Open WebUI on ECS Fargate with auto-scaling
- stdapi.ai gateway connected to Amazon Bedrock
- ElastiCache Valkey for caching
- Aurora PostgreSQL with pgvector extension for RAG, with hybrid search and reranking
- SearXNG for web search integration
- Playwright for web scraping
- HTTPS with ALB on your own domain
- All environment variables pre-configured
- Official container images used as published — no local Docker build and no registry credential

**Deploy:**

```bash
git clone https://github.com/stdapi-ai/samples.git
cd samples/getting_started_openwebui/terraform
tofu init
tofu apply
```

!!! note "Requires a sibling checkout for now"
    This example pins the ECS module to a local path, because the S3 Files support it relies on is not yet in a published module release. Clone [JGoutin/terraform-aws-ecs](https://github.com/JGoutin/terraform-aws-ecs) next to your `samples` checkout until that release ships.

---

## :material-gauge: Operating This Integration

### What It Costs to Run

| Charge | Driver |
| --- | --- |
| stdapi.ai licence | $0.10 per gateway container-hour, metered through AWS Marketplace, with a 14-day free trial on the licence |
| ECS Fargate | Four services — Open WebUI, the gateway, SearXNG, Playwright — each sized and auto-scaled independently |
| Load balancing and networking | One ALB, plus the NAT gateways the private subnets egress through |
| Aurora Serverless v2 | Scales with query load; the sample sets a minimum capacity of zero ACUs |
| ElastiCache Valkey | A standing node cost |
| Model and AI-service usage | Amazon Bedrock, Polly and Transcribe at AWS rates, billed to your account with no markup |

Read a model's price before you send anything to it with [`GET /model_pricing`](api_model_pricing.md). Setting [`COST_TRACKING=true`](operations_cost_management.md#cost-tracking-real-time-aws-pricing) additionally puts a per-request cost on each usage entry — estimated from published AWS prices, not read back from your invoice.

### What to Watch

Both containers log to CloudWatch: Open WebUI writes its audit trail to stdout (`ENABLE_AUDIT_STDOUT=true`), and the gateway writes one structured `request` event per call carrying the request id, path, status code, `execution_time_ms`, the model that served it, and the token counts AWS billed. Turning on [`CLOUDWATCH_METRICS`](operations_logging_monitoring.md#cloudwatch-metrics-emf) republishes those counts as CloudWatch metrics in the `stdapi` namespace, dimensioned by `Model`, so a dashboard can plot chat, embedding, image and speech consumption side by side.

For a chat deployment the useful first question is which feature is consuming the models — the Core Connection, RAG, images or voice all arrive on different paths:

```sql
fields path, execution_time_ms
| filter type = "request"
| stats count(*) as calls, pct(execution_time_ms, 95) as p95_ms by path
| sort calls desc
```

Amazon Bedrock [model invocation logging](operations_compliance.md#amazon-bedrock-invocation-logging) is the AWS-side counterpart — off by default, and the record to enable when you need the prompts and completions themselves rather than metadata.

---

## :material-alert-outline: Known Issues

Open WebUI may list all available models in the chat model selector, including models that do not support chat completions (like image or embedding models). Disable incompatible models in the Open WebUI admin panel.

!!! note "Per-user cost attribution needs an identifier Open WebUI does not send"
    Open WebUI identifies the signed-in user to its backend with `X-OpenWebUI-User-*` headers (`ENABLE_FORWARD_USER_INFO_HEADERS`), not with the OpenAI `safety_identifier` field. [Per-user attribution](operations_cost_management.md#per-user-attribution) reads that field, or an authenticated caller — neither of which one shared connection provides — so every chat is billed to the deployment's own identity. Where the split matters, give each team its own [model alias](operations_configuration.md#model-aliases-configuration) as a separate Open WebUI connection, and read the totals from Amazon Bedrock model invocation logs.

## :material-arrow-right: Next Steps

<div class="grid cards" markdown>

- :material-rocket-launch: [**Getting Started**](operations_getting_started.md) — Deploy stdapi.ai to AWS with Terraform
- :material-docker: [**Local Development**](operations_getting_started_local.md) — Run stdapi.ai locally with Docker
- :material-puzzle: [**More Use Cases**](use_cases.md) — Explore other integrations and tools
- :material-api: [**API Overview**](api_overview.md) — Explore supported endpoints

</div>
