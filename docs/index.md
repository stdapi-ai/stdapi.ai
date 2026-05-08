---
title: stdapi.ai - OpenAI & Anthropic Compatible AI Gateway for AWS Bedrock
description: Run your favorite OpenAI and Anthropic-compatible apps on AWS Bedrock. Access 80+ models including Claude, Kimi, MiniMax with enterprise compliance and pay-per-use pricing. 14-day free trial on AWS Marketplace.
keywords: OpenAI API gateway, Anthropic API gateway, AWS Bedrock API, OpenAI compatible API, Anthropic compatible API, AWS AI gateway, OpenAI AWS integration, Anthropic AWS integration, enterprise AI API, AWS Bedrock integration, OpenAI alternative, Anthropic alternative, private AI deployment, HIPAA compliant AI
hide:
  - toc
  - navigation
---

</style>

<div class="hero hero--home" markdown>
# Run OpenAI & Anthropic Apps on AWS Bedrock

Drop-in API gateway for AWS Bedrock and AI services. Build private AI products on AWS — without exposing your data to third-party AI providers, without subscriptions, and without rewriting your applications. Your existing OpenAI and Anthropic applications work immediately — just change the base URL. Access 80+ models with enterprise privacy, compliance controls, and pay-per-use AWS pricing.

**Two ways to try it:** 14-day free trial on AWS Marketplace for production, or free for local development.

<div class="buttons" markdown>
[Start 14-Day Free Trial on AWS Marketplace](operations_getting_started.md){ .md-button .md-button--primary }
[Try Locally with Docker](operations_getting_started_local.md){ .md-button }
</div>
</div>

<div class="grid cards" markdown>

- :material-swap-horizontal: __Change one line, access 80+ models__
  <br>Drop-in replacement for OpenAI and Anthropic SDKs. Works with Open WebUI, n8n, OpenClaw, Claude Code, LangChain, Continue.dev, and 1000+ tools—no code changes beyond the base URL.

- :material-shield-lock: __Your data stays in your AWS account__
  <br>All inference runs in your account. Data is never shared with model providers or used for training. Configure allowed regions for GDPR, HIPAA, and FedRAMP compliance.

- :material-currency-usd-off: __Pay only for what you use__
  <br>No subscriptions or monthly minimums. Pay AWS Bedrock rates directly with no markup from stdapi.ai. Pay only for what you actually use.

- :material-aws: __Purpose-built for AWS Bedrock__
  <br>Deep integration with prompt caching, reasoning modes, guardrails, service tiers, inference profiles, and prompt routers. Not a generic proxy—built to leverage every Bedrock feature.

- :material-brain: __Claude, Kimi, MiniMax, and 80+ more__
  <br>Claude (reasoning), Kimi, MiniMax, Qwen, Llama, GLM, Nova, Stability AI, and more. Switch models instantly—no vendor lock-in.

- :material-rocket-launch: __Deploy in 5 minutes__
  <br>3 lines of Terraform for production on AWS. Or run Docker locally for development. Production-ready infrastructure with HTTPS, WAF, auto-scaling, and monitoring included.

</div>

## Models and AI services at your fingertips
<div class="logo-marquee" aria-label="Top models and AWS AI services">
  <div class="logo-track">
    <a class="logo-item" href="https://aws.amazon.com/ai/generative-ai/" target="_blank" rel="noopener" title="Amazon Generative AI" aria-label="Amazon Generative AI">
      <img src="styles/logo_amazon.svg" alt="Amazon Generative AI logo" />
      <span>Amazon AI</span>
    </a>
    <a class="logo-item" href="https://aws.amazon.com/bedrock/" target="_blank" rel="noopener" title="Amazon Bedrock" aria-label="Amazon Bedrock">
      <img src="styles/logo_amazon_bedrock.svg" alt="Amazon Bedrock logo" />
      <span>AWS Bedrock</span>
    </a>
    <a class="logo-item" href="https://claude.ai" target="_blank" rel="noopener" title="Anthropic Claude" aria-label="Anthropic Claude">
      <img src="styles/logo_anthropic_claude.svg" alt="Anthropic Claude logo" />
      <span>Claude</span>
    </a>
    <a class="logo-item" href="https://www.deepseek.com" target="_blank" rel="noopener" title="DeepSeek" aria-label="DeepSeek">
      <img src="styles/logo_deepSeek.svg" alt="DeepSeek logo" />
      <span>DeepSeek</span>
    </a>
    <a class="logo-item" href="https://aws.amazon.com/polly/" target="_blank" rel="noopener" title="Amazon Polly" aria-label="Amazon Polly">
      <img src="styles/logo_amazon_polly.svg" alt="Amazon Polly logo" />
      <span>AWS Polly</span>
    </a>
    <a class="logo-item" href="https://ai.meta.com/llama/" target="_blank" rel="noopener" title="Meta Llama" aria-label="Meta Llama">
      <img src="styles/logo_meta.svg" alt="Meta logo" />
      <span>Meta Llama</span>
    </a>
    <a class="logo-item" href="https://www.nvidia.com/en-us/ai/" target="_blank" rel="noopener" title="Nvidia" aria-label="Nvidia">
      <img src="styles/logo_nvidia.svg" alt="Nvidia logo" />
      <span>Nvidia</span>
    </a>
    <a class="logo-item" href="https://qwen.ai" target="_blank" rel="noopener" title="Qwen" aria-label="Qwen">
      <img src="styles/logo_qwen.svg" alt="Qwen logo" />
      <span>Qwen</span>
    </a>
    <a class="logo-item" href="https://openai.com" target="_blank" rel="noopener" title="OpenAI" aria-label="OpenAI">
      <img src="styles/logo_openai.svg" alt="OpenAI logo" />
      <span>OpenAI</span>
    </a>
    <a class="logo-item" href="https://www.moonshot.ai/" target="_blank" rel="noopener" title="Moonshot AI" aria-label="Moonshot AI">
      <img src="styles/logo_moonshot.svg" alt="Moonshot AI logo" />
      <span>Moonshot AI</span>
    </a>
    <a class="logo-item" href="https://aws.amazon.com/translate/" target="_blank" rel="noopener" title="Amazon Translate" aria-label="Amazon Translate">
      <img src="styles/logo_amazon_translate.svg" alt="Amazon Translate logo" />
      <span>AWS Translate</span>
    </a>
    <a class="logo-item" href="https://mistral.ai" target="_blank" rel="noopener" title="Mistral AI" aria-label="Mistral AI">
      <img src="styles/logo_mistralai.svg" alt="Mistral AI logo" />
      <span>Mistral AI</span>
    </a>
    <a class="logo-item" href="https://cohere.com" target="_blank" rel="noopener" title="Cohere" aria-label="Cohere">
      <img src="styles/logo_cohere.svg" alt="Cohere logo" />
      <span>Cohere</span>
    </a>
    <a class="logo-item" href="https://stability.ai" target="_blank" rel="noopener" title="Stability AI" aria-label="Stability AI">
      <img src="styles/logo_stabilityai.svg" alt="Stability AI logo" />
      <span>Stability AI</span>
    </a>
    <a class="logo-item" href="https://www.minimax.io/" target="_blank" rel="noopener" title="Minimax" aria-label="Minimax">
      <img src="styles/logo_minimax.svg" alt="Minimax logo" />
      <span>Minimax</span>
    </a>
    <a class="logo-item" href="https://aws.amazon.com/transcribe/" target="_blank" rel="noopener" title="Amazon Transcribe" aria-label="Amazon Transcribe">
      <img src="styles/logo_amazon_transcribe.svg" alt="Amazon Transcribe logo" />
      <span>AWS Transcribe</span>
    </a>
    <a class="logo-item" href="https://www.ai21.com" target="_blank" rel="noopener" title="AI21 Labs" aria-label="AI21 Labs">
      <img src="styles/logo_ai21.svg" alt="AI21 Labs logo" />
      <span>AI21 Labs</span>
    </a>
    <a class="logo-item" href="https://www.anthropic.com" target="_blank" rel="noopener" title="Anthropic" aria-label="Anthropic">
      <img src="styles/logo_anthropic.svg" alt="Anthropic logo" />
      <span>Anthropic</span>
    </a>
    <a class="logo-item" href="https://z.ai" target="_blank" rel="noopener" title="Z.ai" aria-label="Z.ai">
      <img src="styles/logo_zai.svg" alt="Z.ai logo" />
      <span>Z.ai</span>
    </a>
    <a class="logo-item" href="https://aws.amazon.com/bedrock/nova/" target="_blank" rel="noopener" title="Amazon Nova" aria-label="Amazon Nova">
      <img src="styles/logo_amazon_nova.svg" alt="Amazon Nova logo" />
      <span>AWS Nova</span>
    </a>
    <a class="logo-item" href="https://ai.google/" target="_blank" rel="noopener" title="Google" aria-label="Google">
      <img src="styles/logo_google.svg" alt="Google logo" />
      <span>Google AI</span>
    </a>
    <a class="logo-item" href="https://luma.ai" target="_blank" rel="noopener" title="Luma AI" aria-label="Luma AI">
      <img src="styles/logo_luma.svg" alt="Luma AI logo" />
      <span>Luma AI</span>
    </a>
    <a class="logo-item" href="https://www.twelvelabs.io" target="_blank" rel="noopener" title="Twelve Labs" aria-label="Twelve Labs">
      <img src="styles/logo_twelvelabs.svg" alt="Twelve Labs logo" />
      <span>Twelve Labs</span>
    </a>
    <a class="logo-item" href="https://writer.com" target="_blank" rel="noopener" title="Writer" aria-label="Writer">
      <img src="styles/logo_writer.svg" alt="Writer logo" />
      <span>Writer</span>
    </a>
  </div>
</div>

<script>
  // Duplicate logo track for seamless infinite scroll
  document.addEventListener('DOMContentLoaded', function() {
    const tracks = document.querySelectorAll('.logo-track');
    tracks.forEach(track => {
      const items = Array.from(track.children);
      items.forEach(item => {
        const clone = item.cloneNode(true);
        track.appendChild(clone);
      });
    });
  });
</script>

## How It Works

<div class="section--soft" markdown>
<div class="center" markdown>

```mermaid
%%{init: {'flowchart': {'htmlLabels': true}} }%%
flowchart LR
  openai["<img src='styles/logo_openai.svg' style='height:64px;width:auto;vertical-align:middle;' /> OpenAI SDK Apps"] --> stdapi["<img src='styles/logo.svg' style='height:64px;width:auto;vertical-align:middle;' /> stdapi.ai"]
  anthropic["<img src='styles/logo_anthropic.svg' style='height:64px;width:auto;vertical-align:middle;' /> Anthropic SDK Apps"] --> stdapi
  stdapi --> bedrock["<img src='styles/logo_amazon_bedrock.svg' style='height:64px;width:auto;vertical-align:middle;' /> AWS Bedrock<br/>80+ models"]
  stdapi --> services["<img src='styles/logo_amazon.svg' style='height:64px;width:auto;vertical-align:middle;' /> AWS AI Services<br/>Polly · Transcribe · Translate"]
```

</div>

### Two paths to your first response

=== ":material-docker: 30 seconds — local Docker"

    One `docker run` command, your existing AWS credentials, and you have a working OpenAI-compatible endpoint on `localhost:8000`. Free community image, AGPL-3.0.

    [:octicons-arrow-right-24: Run Locally with Docker](operations_getting_started_local.md)

=== ":material-aws: 5 minutes — production on AWS"

    Three Terraform commands deploy a production-ready stack: ECS Fargate, HTTPS, WAF, auto-scaling, CloudWatch alarms — all IP-restricted to your current address out of the box. Hardened container from AWS Marketplace with a 14-day free trial.

    [:octicons-arrow-right-24: Deploy on AWS](operations_getting_started.md)

**Then point your application to stdapi.ai** — just change the base URL in your existing OpenAI or Anthropic SDK code. Use Claude, Kimi, MiniMax, or any Bedrock model. Switch between models, regions, and providers without changing application code.

!!! tip "Prefer a hands-off AWS setup?"

    A [managed deployment service](https://aws.amazon.com/marketplace/pp/prodview-xknxzjgl7zi5s) can deploy stdapi.ai into your AWS account — no Terraform required.

**Zero lock-in:** Standard OpenAI and Anthropic APIs mean you can switch back or to another provider anytime.

</div>

## Built for AWS

<div class="grid cards" markdown>

- :material-earth: __Multiply your quota across regions__
  <br>Each AWS region has its own independent quota. Add a second region — double your tokens-per-minute. Add a third — triple it. stdapi.ai routes and fails over automatically; clients never see a throttle error.
  <br>[:octicons-arrow-right-24: Resilience & Failover](operations_resilience.md)

- :material-star-settings: __Advanced Bedrock capabilities__
  <br>Reasoning modes (Claude, Nova), prompt caching, guardrails, service tiers, application inference profiles, and prompt routers—all through standard OpenAI API parameters.

- :material-api: __Complete multi-modal API__
  <br>Chat completions, embeddings, image generation/editing/variations, audio speech/transcription/translation. Every route maps OpenAI parameters to Bedrock equivalents.

- :material-aws: __Native AWS AI services__
  <br>Amazon Polly (text-to-speech), Transcribe (speech-to-text with speaker diarization), Translate—all unified under OpenAI-compatible endpoints.

- :material-chart-line: __Full observability__
  <br>OpenTelemetry integration for traces and metrics. Detailed request/response logging. Swagger and ReDoc API documentation served by the application.

- :material-connection: __Agent-ready by design__
  <br>Expose every API endpoint as a Model Context Protocol tool. AI agents connect directly — no HTTP client code. Streamable HTTP and SSE transports, configurable tool selection.

- :material-swap-horizontal: __Automatic deprecated model fallback__
  <br>When AWS retires a model, requests are transparently redirected to its replacement. Your applications survive model deprecations without code changes.

</div>

[:octicons-arrow-right-24: See all features](features.md)

## Who Uses stdapi.ai

<div class="grid cards" markdown>

- :material-server-network: __DevOps & Platform Teams__
  <br>Deploy Open WebUI, LibreChat, or custom chat interfaces for your organization. Unified API gateway for all AI services—no per-application AWS integration needed.
  <br>[:octicons-arrow-right-24: Open WebUI guide](use_cases_openwebui.md) · [:octicons-arrow-right-24: All use cases](use_cases.md)

- :material-code-braces: __Developers & AI Engineers__
  <br>Use Claude, Kimi thinking, and Qwen Coder Next in VS Code (Continue.dev, Cline, Cursor), JetBrains IDEs, or any OpenAI-compatible tool. Test locally with Docker, deploy to production with Terraform.
  <br>[:octicons-arrow-right-24: Coding assistants guide](use_cases_coding_assistants.md)

- :material-robot: __Workflow Automation Teams__
  <br>Connect n8n, Make, Zapier, or custom automation to AWS Bedrock. Access 400+ integrations with enterprise-grade AI—all through one API endpoint.
  <br>[:octicons-arrow-right-24: n8n integration guide](use_cases_n8n.md)

- :material-domain: __Enterprises with Compliance Needs__
  <br>Meet data sovereignty requirements with region controls. GDPR, HIPAA, FedRAMP workloads supported through AWS Bedrock's compliance certifications. CLOUD Act and FISA 702 risk mitigated through customer-managed encryption.
  <br>[:octicons-arrow-right-24: Data Sovereignty & Compliance](operations_compliance.md)

- :material-cash-multiple: __Cost-conscious Organizations__
  <br>Switch from subscription-based AI services to pay-per-use AWS Bedrock pricing. Pay only for actual usage with no monthly commitments while accessing leading models (Claude, Kimi, MiniMax, Qwen).

- :material-application-brackets: __Teams Migrating from OpenAI or Anthropic__
  <br>LangChain, LlamaIndex, Haystack, Claude SDK, or custom apps work immediately. Gradual migration supported—run both APIs in parallel during transition.

- :material-briefcase: __Legal & Professional Services__
  <br>Attorneys, consultants, and accountants cannot send client materials to third-party AI services. stdapi.ai processes all inference inside your own AWS account — client data never leaves your infrastructure.

</div>

## Choose Your Edition

<div class="grid cards" markdown>

-   :material-scale-balance:{ .lg .middle } __Community Edition — Free & Open Source__

    ---

    **Best for:** Open-source projects, local development, testing, and evaluation.

    - Full API compatibility and all features
    - Community Docker image
    - AGPL-3.0 license (source disclosure required for network use)

    [:octicons-arrow-right-24: Run locally with Docker](operations_getting_started_local.md)

-   :material-briefcase:{ .lg .middle } __Commercial Edition — AWS Marketplace__

    ---

    **Best for:** Internal tools, SaaS products, proprietary applications, production.

    - **14-day free trial** — test in your environment risk-free
    - Hardened container, security updates, commercial support
    - **$0.10/container-hour** — no markup on model usage; pay AWS Bedrock rates directly
    - No AGPL restrictions — keep your code and modifications private
    - Terraform module for production-ready deployment in minutes
    - Streamlined AWS billing

    [:octicons-arrow-right-24: Start 14-Day Free Trial](operations_getting_started.md)

</div>

<div class="cta-banner" markdown>
<strong>Ready to run 80+ AI models securely on AWS?</strong>
<div class="buttons" markdown>
[Start 14-Day Free Trial on AWS Marketplace](operations_getting_started.md){ .md-button .md-button--primary }
[Try Locally with Docker](operations_getting_started_local.md){ .md-button }
</div>

**Production:** Terraform module with ECS + hardened container via AWS Marketplace (14-day free trial, $0.10/container-hour, no markup on model usage)<br>
**Community:** Free Docker image for local development and open-source projects (AGPL-3.0)

</div>
