---
title: stdapi.ai - OpenAI, Anthropic & Cohere Compatible AI Gateway for Amazon Bedrock
description: Run your favorite OpenAI, Anthropic, and Cohere-compatible apps on Amazon Bedrock. Access 100+ models including Claude, OpenAI GPT, xAI Grok, Amazon Nova for chat, video generation, and content moderation, with stored conversations, enterprise compliance, and pay-per-use pricing. 14-day free trial on AWS Marketplace.
keywords: OpenAI API gateway, Anthropic API gateway, Cohere API gateway, AWS Bedrock API, OpenAI compatible API, Anthropic compatible API, Cohere compatible API, AWS AI gateway, OpenAI AWS integration, Anthropic AWS integration, enterprise AI API, AWS Bedrock integration, OpenAI alternative, Anthropic alternative, video generation API, content moderation API, private AI deployment, HIPAA compliant AI
hide:
  - toc
  - navigation
---

<!-- ACT 1 — Hero: the drop-in promise -->
<div class="hero2" markdown>
<div class="hero2__copy" markdown>

# Your OpenAI &amp; Anthropic apps, running on AWS Bedrock.

Change one line — the base URL. 100+ models — Claude, OpenAI GPT, DeepSeek, Nova — inside your own AWS account, at AWS Bedrock rates with zero markup. Works with Claude Code, OpenClaw, Open WebUI, n8n, and hundreds of other tools via the OpenAI, Anthropic, or Cohere SDK.

<div class="buttons" markdown>
[Start 14-day free trial](operations_getting_started.md){ .md-button .md-button--primary }
[Try locally with Docker — free](operations_getting_started_local.md){ .md-button }
</div>

<div class="hero2__proof">
<span><strong>AWS Qualified</strong> Software</span>
<span><strong>$0.10</strong>/container-hour</span>
<span><strong>0%</strong> markup on model usage</span>
<span><strong>~1 ms</strong> added latency</span>
<span>Open-source Community Edition</span>
</div>

</div>
<div class="hero2__code">
<div class="code-card">
<div class="code-card__title">app.py — the entire migration</div>
<pre><code><span class="del">- client = OpenAI()</span>
<span class="add">+ client = OpenAI(base_url=<span class="str">"https://ai.yourco.com/v1"</span>)</span>

response = client.chat.completions.create(
    model=<span class="str">"claude-fable-5"</span>, messages=messages
)
<span class="cmt"># same for the Anthropic SDK — point it at /anthropic</span></code></pre>
</div>
</div>
</div>

<!-- ACT 2 — Differentiator carousel. Without JS every panel renders stacked. -->
<div class="carousel" data-carousel markdown>
<div class="carousel__panels" markdown>

<div class="carousel__panel" data-tab="Quota &amp; failover" markdown>
<div class="carousel__kicker">FOR PRODUCTION WORKLOADS</div>

## Bedrock throttling you? Multiply your quota across regions.

Every AWS region has its own independent Bedrock quota. stdapi.ai routes requests across the regions you enable and fails over automatically — on throttling, a regional outage, or a retired model. One throttled region never reaches your clients, and you never touch application code.

<div class="panel-split" markdown>
<div class="panel-visual">
<div class="quota__diagram">
<div class="quota__gw"><img src="styles/logo.svg" alt="" width="1787" height="1953" decoding="async" /><div><strong>stdapi.ai</strong><br/><small>routing + failover</small></div></div>
<div class="quota__arrows" aria-hidden="true"><span></span><span></span><span></span></div>
<div class="quota__regions">
<div class="quota__region"><span><strong>us-east-1</strong> · own quota</span><code>1× tokens/min</code></div>
<div class="quota__region"><span><strong>us-west-2</strong> · own quota</span><code>2× tokens/min</code></div>
<div class="quota__region"><span><strong>eu-west-1</strong> · own quota</span><code>3× tokens/min</code></div>
</div>
</div>
<div class="panel-visual__caption">throttle on one region → transparent retry on the next — the client just gets its answer</div>
</div>
<div class="panel-stats" markdown>

- <code>n×</code> tokens/minute — each enabled region adds its full quota
- <code>0</code> errors when a region throttles — requests reroute before clients notice
- <code>0</code> code changes — routing happens in the gateway, not your app
- <code>24/7</code> multi-AZ ECS Fargate deployment via the validated Terraform module

[:octicons-arrow-right-24: Resilience &amp; failover documentation](operations_resilience.md)

</div>
</div>
</div>

<div class="carousel__panel" data-tab="Beyond chat" markdown>
<div class="carousel__kicker">FOR EVERY MODALITY</div>

## One gateway, the entire API surface.

Most gateways stop at chat completions. stdapi.ai delivers the full OpenAI, Anthropic, and Cohere surface on AWS — chat with server-side stored conversations, embeddings, images, video, speech, transcription, moderation, reranking, and file storage.

<div class="panel-split" markdown>
<div class="panel-visual" markdown>
<div class="api-groups" markdown>

- :material-message-text: <span class="chip">/v1/chat/completions</span> <span class="chip">/v1/responses</span> <span class="chip">/anthropic/v1/messages</span>
- :material-vector-link: <span class="chip">/v1/embeddings</span> <span class="chip">/v2/rerank</span>
- :material-image-outline: <span class="chip">/v1/images/*</span> <span class="chip">/v1/videos</span>
- :material-waveform: <span class="chip">/v1/audio/speech</span> <span class="chip">/v1/audio/transcriptions</span>
- :material-shield-check-outline: <span class="chip">/v1/moderations</span> <span class="chip">/v1/files</span>

</div>
<div class="panel-visual__caption">every parameter your SDK sends that AWS can honour — not just the common subset</div>
</div>
<div class="panel-stats" markdown>

- <code>3</code> API protocols — OpenAI, Anthropic, and Cohere — from one deployment
- <code>50+</code> endpoints — text, images, video, audio, embeddings, moderation, files
- <code>0</code> plugins or client changes — standard SDKs and tools connect instantly

[:octicons-arrow-right-24: API overview](api_overview.md)

</div>
</div>
</div>

<div class="carousel__panel" data-tab="Agent ready" markdown>
<div class="carousel__kicker">FOR AI AGENTS</div>

## Every endpoint is an agent tool.

Agents need no HTTP glue code. stdapi.ai publishes its whole API surface over the Model Context Protocol — chat, images, audio, files, model search — so Claude Code, OpenCode, OpenClaw, LangGraph or any MCP client calls it directly. Underneath, tool calling supports the full OpenAI and Anthropic schemas and tool-choice modes.

<div class="panel-split" markdown>
<div class="panel-visual panel-visual--mono">
<div class="receipt">
<div class="receipt__row receipt__row--head"><span>MCP server</span><span class="ok">enabled</span></div>
<div class="receipt__row"><span>streamable HTTP</span><span>/mcp</span></div>
<div class="receipt__row"><span>SSE · older clients</span><span>/sse</span></div>
<div class="receipt__row"><span>server card</span><span>/.well-known/mcp/server-card.json</span></div>
</div>
<div class="chips">
<span class="chip">openai_chat_completion</span>
<span class="chip">openai_image_generation</span>
<span class="chip">openai_audio_transcription</span>
<span class="chip">anthropic_message</span>
<span class="chip">cohere_rerank</span>
<span class="chip">search_models</span>
</div>
</div>
<div class="panel-stats" markdown>

- <code>50</code> endpoints exposed as named MCP tools, each with generated documentation
- <code>2</code> transports — Streamable HTTP at /mcp, SSE for older clients
- <code>0</code> HTTP client code — agents call every endpoint directly
- <code>auto</code> discovery — agents find every tool through the server card and API catalog

[:octicons-arrow-right-24: MCP &amp; agent capabilities](features.md#mcp-model-context-protocol)

</div>
</div>
</div>

<div class="carousel__panel" data-tab="100+ models" markdown>
<div class="carousel__kicker">FOR MODEL CHOICE</div>

## 100+ models — including OpenAI GPT and Anthropic Claude.

The full Bedrock catalog plus Bedrock Mantle models, discovered automatically across your regions. Every text API works with every model, and retired models transparently redirect to their replacement — your apps never break.

<div class="panel-split" markdown>
<div class="panel-visual">
<div class="model-logos">
<img src="styles/logo_anthropic_claude.svg" alt="Claude" title="Claude" width="24" height="24" decoding="async" loading="lazy" />
<img src="styles/logo_openai.svg" alt="OpenAI GPT" title="OpenAI GPT" width="503" height="499" decoding="async" loading="lazy" />
<img src="styles/logo_xai.svg" alt="xAI Grok" title="xAI Grok" width="21" height="23" decoding="async" loading="lazy" />
<img src="styles/logo_deepSeek.svg" alt="DeepSeek" title="DeepSeek" width="25" height="19" decoding="async" loading="lazy" />
<img src="styles/logo_google.svg" alt="Google Gemma" title="Google Gemma" width="23" height="23" decoding="async" loading="lazy" />
<img src="styles/logo_meta.svg" alt="Meta Llama" title="Meta Llama" width="25" height="17" decoding="async" />
<img src="styles/logo_qwen.svg" alt="Qwen" title="Qwen" width="153" height="151" decoding="async" />
<img src="styles/logo_moonshot.svg" alt="Moonshot AI" title="Moonshot AI" width="25" height="25" decoding="async" />
<img src="styles/logo_mistralai.svg" alt="Mistral AI" title="Mistral AI" width="191" height="135" decoding="async" />
<img src="styles/logo_cohere.svg" alt="Cohere" title="Cohere" width="78" height="78" decoding="async" />
<img src="styles/logo_stabilityai.svg" alt="Stability AI" title="Stability AI" width="103" height="86" decoding="async" />
<img src="styles/logo_minimax.svg" alt="Minimax" title="Minimax" width="25" height="21" decoding="async" />
<img src="styles/logo_ai21.svg" alt="AI21 Labs" title="AI21 Labs" width="38" height="36" decoding="async" />
<img src="styles/logo_zai.svg" alt="Z.ai" title="Z.ai" width="30" height="30" decoding="async" />
<img src="styles/logo_amazon_nova.svg" alt="AWS Nova" title="AWS Nova" width="80" height="80" decoding="async" />
<img src="styles/logo_nvidia.svg" alt="Nvidia" title="Nvidia" width="23" height="16" decoding="async" />
<img src="styles/logo_writer.svg" alt="Writer" title="Writer" width="40" height="40" decoding="async" />
<img src="styles/logo_luma.svg" alt="Luma AI" title="Luma AI" width="24" height="24" decoding="async" />
<img src="styles/logo_twelvelabs.svg" alt="Twelve Labs" title="Twelve Labs" width="600" height="600" decoding="async" />
<span class="model-logos__more">+ many more</span>
</div>
<div class="panel-visual__caption">standard model names resolve automatically — no ARNs, no ID mapping</div>
</div>
<div class="panel-stats" markdown>

- <code>100+</code> models across 10+ providers, auto-discovered at startup
- <code>4</code> text APIs on every model — passthrough or converted automatically
- <code>0</code> breaking changes when a model retires — automatic fallback to its successor

</div>
</div>
</div>

<div class="carousel__panel" data-tab="AWS native" markdown>
<div class="carousel__kicker">FOR AWS-NATIVE TEAMS</div>

## Every AWS capability, zero custom code.

Built for AWS, not around it — every Bedrock-native feature is exposed through standard OpenAI and Anthropic parameters, with AWS AI services and S3 woven into the same API.

<div class="panel-split" markdown>
<div class="panel-visual">
<div class="chips">
<span class="chip">Full S3 integration</span>
<span class="chip">Prompt caching</span>
<span class="chip">Extended thinking</span>
<span class="chip">Guardrails</span>
<span class="chip">Service tiers</span>
<span class="chip">Inference profiles</span>
<span class="chip">Prompt routers</span>
<span class="chip">Cross-region inference</span>
<span class="chip">Nova web grounding</span>
<span class="chip">Code interpreter</span>
<span class="chip">SSML speech</span>
<span class="chip">Speaker diarization</span>
</div>
<div class="model-logos">
<img src="styles/logo_amazon_bedrock.svg" alt="Amazon Bedrock" title="Amazon Bedrock" width="80" height="80" decoding="async" loading="lazy" />
<img src="styles/logo_amazon_polly.svg" alt="Amazon Polly" title="Amazon Polly" width="80" height="80" decoding="async" loading="lazy" />
<img src="styles/logo_amazon_transcribe.svg" alt="Amazon Transcribe" title="Amazon Transcribe" width="80" height="80" decoding="async" loading="lazy" />
<img src="styles/logo_amazon_translate.svg" alt="Amazon Translate" title="Amazon Translate" width="80" height="80" decoding="async" loading="lazy" />
<img src="styles/logo_amazon_comprehend.svg" alt="Amazon Comprehend" title="Amazon Comprehend" width="80" height="80" decoding="async" loading="lazy" />
<img src="styles/logo_amazon_s3.svg" alt="Amazon S3" title="Amazon S3" width="80" height="80" decoding="async" loading="lazy" />
<img src="styles/logo_amazon_cloudwatch.svg" alt="Amazon CloudWatch" title="Amazon CloudWatch" width="80" height="80" decoding="async" loading="lazy" />
</div>
<div class="panel-visual__caption">all via standard OpenAI &amp; Anthropic parameters — no AWS SDK in your app</div>
</div>
<div class="panel-stats" markdown>

- <code>5</code> AWS AI services unified — Bedrock, Polly, Transcribe, Translate, Comprehend
- <code>s3://</code> direct S3 inputs in chat, images, and embeddings — generated media lands back in your bucket
- <code>IAM</code> least-privilege reference policies documented per feature

[:octicons-arrow-right-24: All features](features.md)

</div>
</div>
</div>

<div class="carousel__panel" data-tab="Cost control" markdown>
<div class="carousel__kicker">FOR BUDGET OWNERS</div>

## Pay AWS rates. See the cost of every request.

No subscriptions, no minimums, no markup on model usage. Built-in cost tracking prices every call from AWS's own Price List — serving region, service tier, cached tokens, and long-context rates included.

<div class="panel-split" markdown>
<div class="panel-visual panel-visual--mono">
<div class="receipt">
<div class="receipt__row receipt__row--head"><span>POST /v1/chat/completions</span><span class="ok">200 · 1.9s</span></div>
<div class="receipt__row"><span>model</span><span>claude-fable-5</span></div>
<div class="receipt__row"><span>region · tier</span><span>eu-west-1 · priority</span></div>
<div class="receipt__row"><span>tokens</span><span>in 12,410 (9,800 cached) · out 642</span></div>
<div class="receipt__row receipt__row--total"><span>cost</span><span class="amount">$0.048231 USD</span></div>
</div>
</div>
<div class="panel-stats" markdown>

- <code>0%</code> markup on model usage — Bedrock billed by AWS directly
- <code>live</code> rates from the AWS Price List catalog — fetched from AWS, not hand-maintained
- <code>1:1</code> per-request and per-user cost attribution across all endpoints

[:octicons-arrow-right-24: Cost management documentation](operations_cost_management.md)

</div>
</div>
</div>

<div class="carousel__panel" data-tab="Ready to deploy" markdown>
<div class="carousel__kicker">FOR DEVOPS TEAMS</div>

## Production on AWS in three Terraform commands.

The validated Terraform module ships the whole stack — ECS Fargate, HTTPS, WAF, auto-scaling, monitoring. It works as-is with secure defaults, and exposes advanced options for power users: VPC integration, multi-region, cost-optimized setups.

<div class="panel-split" markdown>
<div class="panel-visual panel-visual--mono">
<div class="terminal">
<div><span class="ok">$</span> <span class="cmd">terraform init</span></div>
<div><span class="ok">$</span> <span class="cmd">terraform plan</span></div>
<div><span class="ok">$</span> <span class="cmd">terraform apply</span></div>
<div class="terminal__out">Apply complete! endpoint_url = <span class="str">https://ai.yourco.com</span></div>
</div>
<div class="panel-visual__caption panel-visual__caption--ruled">ECS Fargate · ALB TLS 1.3 · WAF · auto-scaling · CloudWatch alarms</div>
</div>
<div class="panel-stats" markdown>

- <code>5min</code> from AWS Marketplace subscription to a production endpoint
- <code>0</code> required configuration — secure, IP-restricted, FSBP-aligned defaults
- <code>100+</code> optional variables for power users — bring your own VPC, go multi-region, or cost-optimize

[:octicons-arrow-right-24: Deploy on AWS guide](operations_getting_started.md)

Prefer hands-off? A [managed deployment service](https://aws.amazon.com/marketplace/pp/prodview-xknxzjgl7zi5s) sets it up in your account — no Terraform required.
{ .panel-stats__note }

</div>
</div>
</div>

</div>
</div>

<div class="logo-marquee logo-marquee--slim" role="group" aria-label="Top models and AWS AI services">
  <div class="logo-track">
    <a class="logo-item" href="https://aws.amazon.com/ai/generative-ai/" target="_blank" rel="noopener" title="Amazon Generative AI" aria-label="Amazon Generative AI">
      <img src="styles/logo_amazon.svg" alt="Amazon Generative AI logo" width="40" height="40" decoding="async" />
      <span>Amazon AI</span>
    </a>
    <a class="logo-item" href="https://aws.amazon.com/bedrock/" target="_blank" rel="noopener" title="Amazon Bedrock" aria-label="Amazon Bedrock">
      <img src="styles/logo_amazon_bedrock.svg" alt="Amazon Bedrock logo" width="80" height="80" decoding="async" />
      <span>Amazon Bedrock</span>
    </a>
    <a class="logo-item" href="https://claude.ai" target="_blank" rel="noopener" title="Anthropic Claude" aria-label="Anthropic Claude">
      <img src="styles/logo_anthropic_claude.svg" alt="Anthropic Claude logo" width="24" height="24" decoding="async" />
      <span>Claude</span>
    </a>
    <a class="logo-item" href="https://www.deepseek.com" target="_blank" rel="noopener" title="DeepSeek" aria-label="DeepSeek">
      <img src="styles/logo_deepSeek.svg" alt="DeepSeek logo" width="25" height="19" decoding="async" />
      <span>DeepSeek</span>
    </a>
    <a class="logo-item" href="https://aws.amazon.com/polly/" target="_blank" rel="noopener" title="Amazon Polly" aria-label="Amazon Polly">
      <img src="styles/logo_amazon_polly.svg" alt="Amazon Polly logo" width="80" height="80" decoding="async" />
      <span>Amazon Polly</span>
    </a>
    <a class="logo-item" href="https://ai.meta.com/llama/" target="_blank" rel="noopener" title="Meta Llama" aria-label="Meta Llama">
      <img src="styles/logo_meta.svg" alt="Meta logo" width="25" height="17" decoding="async" />
      <span>Meta Llama</span>
    </a>
    <a class="logo-item" href="https://www.nvidia.com/en-us/ai/" target="_blank" rel="noopener" title="Nvidia" aria-label="Nvidia">
      <img src="styles/logo_nvidia.svg" alt="Nvidia logo" width="23" height="16" decoding="async" />
      <span>Nvidia</span>
    </a>
    <a class="logo-item" href="https://qwen.ai" target="_blank" rel="noopener" title="Qwen" aria-label="Qwen">
      <img src="styles/logo_qwen.svg" alt="Qwen logo" width="153" height="151" decoding="async" />
      <span>Qwen</span>
    </a>
    <a class="logo-item" href="https://openai.com" target="_blank" rel="noopener" title="OpenAI GPT" aria-label="OpenAI GPT">
      <img src="styles/logo_openai.svg" alt="OpenAI logo" width="503" height="499" decoding="async" />
      <span>OpenAI GPT</span>
    </a>
    <a class="logo-item" href="https://x.ai" target="_blank" rel="noopener" title="xAI Grok" aria-label="xAI Grok">
      <img src="styles/logo_xai.svg" alt="xAI logo" width="21" height="23" decoding="async" />
      <span>xAI Grok</span>
    </a>
    <a class="logo-item" href="https://www.moonshot.ai/" target="_blank" rel="noopener" title="Moonshot AI" aria-label="Moonshot AI">
      <img src="styles/logo_moonshot.svg" alt="Moonshot AI logo" width="25" height="25" decoding="async" />
      <span>Moonshot AI</span>
    </a>
    <a class="logo-item" href="https://aws.amazon.com/translate/" target="_blank" rel="noopener" title="Amazon Translate" aria-label="Amazon Translate">
      <img src="styles/logo_amazon_translate.svg" alt="Amazon Translate logo" width="80" height="80" decoding="async" />
      <span>Amazon Translate</span>
    </a>
    <a class="logo-item" href="https://mistral.ai" target="_blank" rel="noopener" title="Mistral AI" aria-label="Mistral AI">
      <img src="styles/logo_mistralai.svg" alt="Mistral AI logo" width="191" height="135" decoding="async" />
      <span>Mistral AI</span>
    </a>
    <a class="logo-item" href="https://cohere.com" target="_blank" rel="noopener" title="Cohere" aria-label="Cohere">
      <img src="styles/logo_cohere.svg" alt="Cohere logo" width="78" height="78" decoding="async" />
      <span>Cohere</span>
    </a>
    <a class="logo-item" href="https://stability.ai" target="_blank" rel="noopener" title="Stability AI" aria-label="Stability AI">
      <img src="styles/logo_stabilityai.svg" alt="Stability AI logo" width="103" height="86" decoding="async" />
      <span>Stability AI</span>
    </a>
    <a class="logo-item" href="https://www.minimax.io/" target="_blank" rel="noopener" title="Minimax" aria-label="Minimax">
      <img src="styles/logo_minimax.svg" alt="Minimax logo" width="25" height="21" decoding="async" />
      <span>Minimax</span>
    </a>
    <a class="logo-item" href="https://aws.amazon.com/transcribe/" target="_blank" rel="noopener" title="Amazon Transcribe" aria-label="Amazon Transcribe">
      <img src="styles/logo_amazon_transcribe.svg" alt="Amazon Transcribe logo" width="80" height="80" decoding="async" />
      <span>Amazon Transcribe</span>
    </a>
    <a class="logo-item" href="https://www.ai21.com" target="_blank" rel="noopener" title="AI21 Labs" aria-label="AI21 Labs">
      <img src="styles/logo_ai21.svg" alt="AI21 Labs logo" width="38" height="36" decoding="async" />
      <span>AI21 Labs</span>
    </a>
    <a class="logo-item" href="https://www.anthropic.com" target="_blank" rel="noopener" title="Anthropic" aria-label="Anthropic">
      <img src="styles/logo_anthropic.svg" alt="Anthropic logo" width="24" height="24" decoding="async" />
      <span>Anthropic</span>
    </a>
    <a class="logo-item" href="https://z.ai" target="_blank" rel="noopener" title="Z.ai" aria-label="Z.ai">
      <img src="styles/logo_zai.svg" alt="Z.ai logo" width="30" height="30" decoding="async" />
      <span>Z.ai</span>
    </a>
    <a class="logo-item" href="https://aws.amazon.com/bedrock/nova/" target="_blank" rel="noopener" title="Amazon Nova" aria-label="Amazon Nova">
      <img src="styles/logo_amazon_nova.svg" alt="Amazon Nova logo" width="80" height="80" decoding="async" />
      <span>Amazon Nova</span>
    </a>
    <a class="logo-item" href="https://ai.google/" target="_blank" rel="noopener" title="Google Gemma" aria-label="Google Gemma">
      <img src="styles/logo_google.svg" alt="Google logo" width="23" height="23" decoding="async" />
      <span>Google Gemma</span>
    </a>
    <a class="logo-item" href="https://luma.ai" target="_blank" rel="noopener" title="Luma AI" aria-label="Luma AI">
      <img src="styles/logo_luma.svg" alt="Luma AI logo" width="24" height="24" decoding="async" />
      <span>Luma AI</span>
    </a>
    <a class="logo-item" href="https://www.twelvelabs.io" target="_blank" rel="noopener" title="Twelve Labs" aria-label="Twelve Labs">
      <img src="styles/logo_twelvelabs.svg" alt="Twelve Labs logo" width="600" height="600" decoding="async" />
      <span>Twelve Labs</span>
    </a>
    <a class="logo-item" href="https://aws.amazon.com/comprehend/" target="_blank" rel="noopener" title="Amazon Comprehend" aria-label="Amazon Comprehend">
      <img src="styles/logo_amazon_comprehend.svg" alt="Amazon Comprehend logo" width="80" height="80" decoding="async" />
      <span>Amazon Comprehend</span>
    </a>
    <a class="logo-item" href="https://writer.com" target="_blank" rel="noopener" title="Writer" aria-label="Writer">
      <img src="styles/logo_writer.svg" alt="Writer logo" width="40" height="40" decoding="async" />
      <span>Writer</span>
    </a>
  </div>
</div>

<!-- ACT 3 — Compliance / sovereignty -->
<div class="band band--plain" markdown>
<div class="carousel__kicker">FOR REGULATED WORKLOADS</div>

## Your data never leaves your AWS account.

Unlike SaaS gateways, stdapi.ai is infrastructure you run: no third party ever sits between your users and your models.

<div class="grid cards" markdown>

- :material-shield-lock: __Data sovereignty by design__
  <br>All inference inside your AWS account. Never shared with model providers, never used for training.

- :material-earth: __Region allow-lists__
  <br>Pin workloads to approved regions for GDPR, HIPAA, and FedRAMP requirements.

- :material-key: __Customer-managed encryption__
  <br>CMK encryption mitigates CLOUD Act and FISA 702 exposure for regulated data.

- :material-shield-star: __Security Hub aligned__
  <br>Terraform module built against AWS FSBP controls; GuardDuty and DNS Firewall opt-ins close the gaps.

</div>

Legal, healthcare (HIPAA), EU sovereignty (GDPR), FedRAMP — [:octicons-arrow-right-24: Data Sovereignty &amp; Compliance guide](operations_compliance.md)
{ .band__note }

<div class="qualified-card">
<img src="styles/aws_qualified_software_badge_dark.png" alt="AWS Qualified Software badge" width="120" height="120" decoding="async" loading="lazy" />
<div><strong>AWS Qualified Software</strong><br/>Verified by AWS against its technical and security requirements for AWS Marketplace.</div>
</div>

</div>

<!-- How it compares -->
## How it compares

All four expose an OpenAI-compatible API in front of Amazon Bedrock — the coverage differs.

<div class="compare" role="region" aria-label="Feature comparison" tabindex="0" markdown>

|                                                                                             | stdapi.ai                | LiteLLM                     | Access Gateway              | Bedrock Mantle              |
| ------------------------------------------------------------------------------------------- | ------------------------ | --------------------------- | --------------------------- | --------------------------- |
| Full multi-modal API — images, video, audio, files                                            | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> |
| OpenAI + Anthropic + Cohere protocols                                                         | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> |
| Multi-region quota multiplication                                                             | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> |
| Automatic failover — throttling, region outages, retired models                               | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> |
| AWS AI services &amp; advanced Bedrock features — Polly, Transcribe, guardrails, service tiers | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> |
| Integrated MCP server                                                                         | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> |
| Real-time cost tracking &amp; observability                                                   | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> |
| Production deployment — Terraform, WAF, auto-scaling                                          | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> |

</div>

<div class="compare__legend" markdown>
<span class="m-y" aria-hidden="true">✓</span> full &nbsp; <span class="m-p" aria-hidden="true">◐</span> partial / manual setup &nbsp; <span class="m-n" aria-hidden="true">—</span> not available &nbsp;·&nbsp; [Full comparison](features.md#how-stdapiai-compares)
</div>

<div class="buttons" markdown>
[See pricing](#transparent-pricing){ .md-button }
</div>

<!-- What teams run on it -->
## What teams run on it

Every integration is the same three steps: deploy, copy your endpoint URL, paste it into the tool's settings. There is no step four.

<div class="usecases" markdown>

<div class="usecase" markdown>
<div class="usecase__tag">PRIVATE CHATGPT</div>
<div class="usecase__title">Enterprise chat</div>
<div class="usecase__body">ChatGPT-style assistant for your organization — chat, voice, images, and RAG, with every conversation staying in your account.</div>
<div class="usecase__tools">Open WebUI · LobeHub · LibreChat</div>
[Open WebUI guide](use_cases_openwebui.md)
</div>

<div class="usecase" markdown>
<div class="usecase__tag">CODING AGENTS</div>
<div class="usecase__title">AI-assisted development</div>
<div class="usecase__body">Frontier coding models in your IDE and terminal — without sending your codebase to a third-party AI cloud.</div>
<div class="usecase__tools">Claude Code · Codex · OpenCode · Zed</div>
[Coding assistants guide](use_cases_coding_assistants.md)
</div>

<div class="usecase" markdown>
<div class="usecase__tag">NO-CODE AUTOMATION</div>
<div class="usecase__title">AI in business workflows</div>
<div class="usecase__body">Add AI steps to business processes with visual workflow builders — classification, summarization, content generation.</div>
<div class="usecase__tools">n8n · Dify · Langflow · Flowise</div>
[n8n guide](use_cases_n8n.md)
</div>

<div class="usecase" markdown>
<div class="usecase__tag">AUTONOMOUS AGENTS</div>
<div class="usecase__title">Agents you control</div>
<div class="usecase__body">Self-directed agents on infrastructure you own — with the built-in MCP server exposing every endpoint as an agent tool.</div>
<div class="usecase__tools">OpenClaw · Hermes · LangGraph · CrewAI</div>
[All use cases](use_cases.md)
</div>

</div>

[:octicons-arrow-right-24: Use cases &amp; integration guides](use_cases.md)
{ .usecases__more }

<!-- Pricing — the closing conversion moment -->
## Transparent pricing

Start local, graduate to AWS — same API, your application code never changes. And zero lock-in: leaving is the same one-line base-URL change that got you in.

<div class="pricing">
<div class="pricing__col">
<div class="pricing__tier">Community</div>
<div class="pricing__price">$0</div>
<p>AGPL-3.0 · Docker image · full API · local dev &amp; open-source projects</p>
<p><a class="md-button" href="operations_getting_started_local/">Run with Docker</a></p>
</div>
<div class="pricing__col pricing__col--featured">
<div class="pricing__flag">14-DAY FREE TRIAL</div>
<div class="pricing__tier">Commercial · AWS Marketplace</div>
<div class="pricing__price">$0.10 <small>/container-hour</small></div>
<p>No markup on model usage — pay Bedrock rates directly. Hardened container, Terraform module, commercial support (1 business day), no AGPL obligations, AWS billing.</p>
<p><a class="md-button md-button--primary" href="operations_getting_started/">Start 14-day free trial</a></p>
</div>
<div class="pricing__col pricing__col--offer">
<div class="pricing__tier pricing__tier--accent">Private offer · save 10%</div>
<div class="pricing__price">$0.09 <small>/container-hour</small></div>
<p>Same pay-per-use model, no minimums. Want to try first? Use the free trial, then accept your offer. Send your AWS account ID — we'll set it up directly.</p>
<p><a class="md-button" href="contact/#private-offer">Request a private offer</a></p>
</div>
</div>

<div class="footer-line" markdown>
Questions before you commit? [Talk to the founder](contact.md) · [GitHub](https://github.com/stdapi-ai/stdapi.ai) · [Documentation](getting_started.md)
</div>
