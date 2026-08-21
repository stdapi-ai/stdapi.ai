---
title: stdapi.ai - OpenAI, Anthropic & Cohere Compatible AI Gateway for Amazon Bedrock
description: Run your favorite OpenAI, Anthropic, and Cohere-compatible apps on Amazon Bedrock. Access 100+ models including Claude, OpenAI GPT, xAI Grok, Amazon Nova for chat, images, video, audio, embeddings, and content moderation — in your own AWS account, at AWS rates with no markup. 14-day free trial on AWS Marketplace.
keywords: OpenAI API gateway, Anthropic API gateway, Cohere API gateway, AWS Bedrock API, OpenAI compatible API, Anthropic compatible API, Cohere compatible API, AWS AI gateway, OpenAI AWS integration, Anthropic AWS integration, enterprise AI API, AWS Bedrock integration, OpenAI alternative, Anthropic alternative, multimodal AI API, multi-region AI gateway, video generation API, content moderation API, private AI deployment
hide:
  - toc
  - navigation
---

<!-- ACT 1 — Hero: the two-step promise -->
<div class="hero2" markdown>
<div class="hero2__copy" markdown>

# Your OpenAI &amp; Anthropic apps on AWS. Not just chat.

An AI gateway you run in your own AWS account. Point Claude Code, Open WebUI, n8n, OpenClaw — or your own code — at it, and they reach 100+ models including Claude, OpenAI GPT, DeepSeek and Nova, at AWS Bedrock rates with zero markup. Two client-side changes: the base URL, and the model name — now picked from all of them, not one vendor's list.

<div class="buttons" markdown>
[Start 14-day free trial](operations_getting_started.md){ .md-button .md-button--primary }
[Try locally with Docker — free](operations_getting_started_local.md){ .md-button }
</div>

<div class="hero2__proof">
<span><strong>AWS Qualified</strong> Software</span>
<span><strong>$0.10</strong>/container-hour</span>
<span><strong>0%</strong> markup on model usage</span>
<span><strong>&lt;1 ms</strong> gateway overhead</span>
<span>Open-source Community Edition</span>
</div>

</div>
<div class="hero2__code">
<div class="code-card">
<div class="code-card__title">app.py — the two client-side changes</div>
<pre><code><span class="del">- client = OpenAI()</span>
<span class="add">+ client = OpenAI(base_url=<span class="str">"https://ai.yourco.com/v1"</span>)</span>

response = client.chat.completions.create(
<span class="del">-     model=<span class="str">"gpt-4o"</span>,</span>
<span class="add">+     model=<span class="str">"claude-fable-5"</span>,  <span class="cmt"># or any model in the catalogue</span></span>
    messages=messages
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

## Bedrock throttling you? Add a region, add its quota.

Every AWS region has its own independent Bedrock quota. stdapi.ai routes requests across the regions you enable and retries eligible failures elsewhere — on throttling, a temporary regional outage, or a retired model. Routing happens in the gateway, so you never touch application code.

<div class="panel-split" markdown>
<div class="panel-visual">
<div class="quota__diagram">
<div class="quota__gw"><img src="styles/logo.svg" alt="" width="1787" height="1953" decoding="async" /><div><strong>stdapi.ai</strong><br/><small>routing + failover</small></div></div>
<div class="quota__arrows" aria-hidden="true"><span></span><span></span><span></span></div>
<div class="quota__regions">
<div class="quota__region"><span><strong>us-east-1</strong> · enabled</span><code>own quota</code></div>
<div class="quota__region"><span><strong>us-west-2</strong> · enabled</span><code>own quota</code></div>
<div class="quota__region"><span><strong>eu-west-1</strong> · enabled</span><code>own quota</code></div>
</div>
</div>
<div class="panel-visual__caption">throttle on one region → eligible requests retry on the next</div>
</div>
<div class="panel-stats" markdown>

- <code>+1</code> quota per region — every region you enable brings its own
- <code>auto</code> retry in another enabled region on eligible throttling or outage
- <code>0</code> code changes — routing happens in the gateway, not your app
- <code>24/7</code> multi-AZ ECS Fargate deployment via the validated Terraform module

Streaming responses can only retry before the stream opens, and asynchronous jobs stay in the region that accepted them.
{ .panel-stats__note }

[:octicons-arrow-right-24: Resilience &amp; failover documentation](operations_resilience.md)

</div>
</div>
</div>

<div class="carousel__panel" data-tab="Beyond chat" markdown>
<div class="carousel__kicker">FOR EVERY MODALITY</div>

## One gateway, every modality you're already calling.

Most gateways stop at chat completions. stdapi.ai covers text, retrieval, embeddings, images, video, speech, live voice, batch inference, moderation, reranking, and file storage across the OpenAI, Anthropic, and Cohere protocols — with conversations kept server-side and continued by id instead of resent.

<div class="panel-split" markdown>
<div class="panel-visual" markdown>
<div class="api-groups" markdown>

- :material-message-text: <span class="chip">/v1/chat/completions</span> <span class="chip">/v1/responses</span> <span class="chip">/anthropic/v1/messages</span> <span class="chip">/v1/conversations</span>
- :material-vector-link: <span class="chip">/v1/embeddings</span> <span class="chip">/v1/vector_stores</span> <span class="chip">/cohere/v2/rerank</span>
- :material-image-outline: <span class="chip">/v1/images/*</span> <span class="chip">/v1/videos</span>
- :material-waveform: <span class="chip">/v1/audio/speech</span> <span class="chip">/v1/audio/transcriptions</span> <span class="chip">WS /v1/realtime</span>
- :material-tray-full: <span class="chip">/v1/batches</span> <span class="chip">/anthropic/v1/messages/batches</span>
- :material-shield-check-outline: <span class="chip">/v1/moderations</span> <span class="chip">/v1/files</span>

</div>
<div class="panel-visual__caption">the parameters your SDK sends are honoured wherever AWS supports them — not just the common subset</div>
</div>
<div class="panel-stats" markdown>

- <code>3</code> API protocols — OpenAI, Anthropic, and Cohere — from one deployment
- <code>80+</code> endpoints — text, retrieval, images, video, audio, batch, moderation, files
- <code>WS</code> live speech-to-speech on the OpenAI Realtime API — speech in, speech and a transcript out

[:octicons-arrow-right-24: API overview](api_overview.md)

</div>
</div>
</div>

<div class="carousel__panel" data-tab="Retrieval" markdown>
<div class="carousel__kicker">FOR RAG &amp; KNOWLEDGE</div>

## Your documents answer, and the model cites them.

Attach a file and it is chunked, embedded and indexed in a vector bucket in your own account, then searched by meaning. Point the same endpoints at an Amazon Bedrock knowledge base you already run and it answers as a vector store too. Then hand the stores to any chat model on the Responses API: it runs the searches the turn needs and cites the files it drew on.

<div class="panel-split" markdown>
<div class="panel-visual panel-visual--mono">
<div class="receipt">
<div class="receipt__row receipt__row--head"><span>POST /v1/responses</span><span class="ok">200 · file_search</span></div>
<div class="receipt__row"><span>model</span><span>any chat model on /v1/responses</span></div>
<div class="receipt__row"><span>searches run by the model</span><span>"vacation days" · "PTO accrual"</span></div>
<div class="receipt__row"><span>passages kept</span><span>4 · scored, attribute-filtered</span></div>
<div class="receipt__row receipt__row--total"><span>grounded answer</span><span class="amount">2 file citations</span></div>
</div>
<div class="chips">
<span class="chip">managed store · Amazon S3 Vectors</span>
<span class="chip">your Amazon Bedrock knowledge base</span>
</div>
</div>
<div class="panel-stats" markdown>

- <code>0</code> extra infrastructure — no chunker, no embedding pipeline and no vector database to run beside the gateway
- <code>2</code> kinds of store behind one API — files you attach here, or a knowledge base you already run
- <code>1</code> citation per file the answer drew on, with the passages returned on request

A knowledge base is addressed under an allowlist and is never created or deleted through this API.
{ .panel-stats__note }

[:octicons-arrow-right-24: Vector stores &amp; file search](features.md#retrieval-vector-stores)

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

- <code>80+</code> endpoints exposed as named MCP tools, each with generated documentation
- <code>2</code> transports — Streamable HTTP at /mcp, SSE for older clients
- <code>0</code> HTTP client code — agents call every endpoint directly
- <code>auto</code> discovery — agents find every tool through the server card and API catalog

This exposes the gateway's own AI and media APIs over MCP; it is not an aggregator for third-party MCP servers.
{ .panel-stats__note }

[:octicons-arrow-right-24: MCP &amp; agent capabilities](features.md#mcp-model-context-protocol)

</div>
</div>
</div>

<div class="carousel__panel" data-tab="100+ models" markdown>
<div class="carousel__kicker">FOR MODEL CHOICE</div>

## 100+ models — including OpenAI GPT and Anthropic Claude.

Bedrock, Bedrock Mantle, Polly, Transcribe and Comprehend all surface as models in one catalog, detected automatically at startup. On a shared endpoint they interchange by name — swapping a Polly voice for a Bedrock speech model is a one-word change, and nothing needs writing or maintaining as AWS adds and retires models. All four text APIs work with every discovered model, and retired model IDs can redirect to their supported successor instead of failing.

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

- <code>0</code> configuration — one catalog spanning Bedrock, Mantle, Polly, Transcribe and Comprehend
- <code>100+</code> models across 10+ providers in a typical multi-region catalog
- <code>4</code> text APIs on every model — passthrough or converted automatically

</div>
</div>
</div>

<div class="carousel__panel" data-tab="AWS native" markdown>
<div class="carousel__kicker">FOR AWS-NATIVE TEAMS</div>

## Deep AWS features, zero custom code.

Built for AWS, not around it — Bedrock-native capabilities are exposed through standard OpenAI and Anthropic parameters, with AWS AI services and S3 woven into the same API.

<div class="panel-split" markdown>
<div class="panel-visual">
<div class="chips">
<span class="chip">Amazon S3 inputs &amp; outputs</span>
<span class="chip">Prompt caching</span>
<span class="chip">Reasoning</span>
<span class="chip">Bedrock Guardrails</span>
<span class="chip">Service tiers</span>
<span class="chip">Inference profiles</span>
<span class="chip">Prompt routers</span>
<span class="chip">Geographic routing</span>
<span class="chip">Web grounding</span>
<span class="chip">Code interpreter</span>
<span class="chip">SSML speech</span>
<span class="chip">Batch inference</span>
<span class="chip">Live transcription</span>
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

## Pay AWS rates. See which user spent them.

No subscriptions, no minimums, no markup on model usage. Optional cost tracking prices each call from AWS's own Price List — serving region, service tier, cached tokens, and long-context rates included. And each end user's model calls can run under their own short-lived role session, so AWS reports their spend separately in Cost Explorer and the Cost and Usage Report — from the invoice itself, not from an estimate.

<div class="panel-split" markdown>
<div class="panel-visual panel-visual--mono">
<div class="receipt">
<div class="receipt__row receipt__row--head"><span>POST /v1/chat/completions</span><span class="ok">200 · 1.9s</span></div>
<div class="receipt__row"><span>model</span><span>claude-fable-5</span></div>
<div class="receipt__row"><span>region · tier</span><span>eu-west-1 · priority</span></div>
<div class="receipt__row"><span>tokens</span><span>in 12,410 (9,800 cached) · out 642</span></div>
<div class="receipt__row"><span>end user</span><span>billed under their own role session</span></div>
<div class="receipt__row receipt__row--total"><span>estimated cost</span><span class="amount">$0.048231 USD</span></div>
</div>
</div>
<div class="panel-stats" markdown>

- <code>0%</code> markup on model usage — Bedrock billed by AWS directly
- <code>live</code> rates from the AWS Price List catalog — fetched from AWS, not hand-maintained
- <code>per user</code> spend on the AWS bill itself — grouped in Cost Explorer and the CUR, and testable in IAM policies
- <code>batch</code> price on asynchronous request sets — submit a corpus, pay Bedrock's discounted batch rate

Per-request cost figures are estimated from published AWS prices, not read back from your invoice; per-user attribution is off by default and needs a role you create.
{ .panel-stats__note }

[:octicons-arrow-right-24: Cost management documentation](operations_cost_management.md)

</div>
</div>
</div>

<div class="carousel__panel" data-tab="Ready to deploy" markdown>
<div class="carousel__kicker">FOR DEVOPS TEAMS</div>

## Production on AWS in two Terraform commands.

The validated Terraform module ships the whole stack — ECS Fargate, HTTPS, auto-scaling, and optional WAF and monitoring. It works as-is with secure defaults, and exposes advanced options for power users: VPC integration, multi-region, cost-optimized setups.

<div class="panel-split" markdown>
<div class="panel-visual panel-visual--mono">
<div class="terminal">
<div><span class="ok">$</span> <span class="cmd">terraform init</span></div>
<div><span class="ok">$</span> <span class="cmd">terraform apply</span></div>
<div class="terminal__out">Apply complete! endpoint_url = <span class="str">https://ai.yourco.com</span></div>
</div>
<div class="panel-visual__caption panel-visual__caption--ruled">ECS Fargate · ALB TLS 1.3 · auto-scaling · CloudWatch alarms · optional WAF</div>
</div>
<div class="panel-stats" markdown>

- <code>2</code> commands from AWS Marketplace subscription to a production endpoint
- <code>FSBP</code> aligned defaults out of the box — private subnets, least privilege, encryption at rest
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
    <div class="logo-item" title="Amazon Generative AI">
      <img src="styles/logo_amazon.svg" alt="Amazon Generative AI logo" width="40" height="40" decoding="async" />
      <span>Amazon AI</span>
    </div>
    <div class="logo-item" title="Amazon Bedrock">
      <img src="styles/logo_amazon_bedrock.svg" alt="Amazon Bedrock logo" width="80" height="80" decoding="async" />
      <span>Amazon Bedrock</span>
    </div>
    <div class="logo-item" title="Anthropic Claude">
      <img src="styles/logo_anthropic_claude.svg" alt="Anthropic Claude logo" width="24" height="24" decoding="async" />
      <span>Claude</span>
    </div>
    <div class="logo-item" title="DeepSeek">
      <img src="styles/logo_deepSeek.svg" alt="DeepSeek logo" width="25" height="19" decoding="async" />
      <span>DeepSeek</span>
    </div>
    <div class="logo-item" title="Amazon Polly">
      <img src="styles/logo_amazon_polly.svg" alt="Amazon Polly logo" width="80" height="80" decoding="async" />
      <span>Amazon Polly</span>
    </div>
    <div class="logo-item" title="Meta Llama">
      <img src="styles/logo_meta.svg" alt="Meta logo" width="25" height="17" decoding="async" />
      <span>Meta Llama</span>
    </div>
    <div class="logo-item" title="Nvidia">
      <img src="styles/logo_nvidia.svg" alt="Nvidia logo" width="23" height="16" decoding="async" />
      <span>Nvidia</span>
    </div>
    <div class="logo-item" title="Qwen">
      <img src="styles/logo_qwen.svg" alt="Qwen logo" width="153" height="151" decoding="async" />
      <span>Qwen</span>
    </div>
    <div class="logo-item" title="OpenAI GPT">
      <img src="styles/logo_openai.svg" alt="OpenAI logo" width="503" height="499" decoding="async" />
      <span>OpenAI GPT</span>
    </div>
    <div class="logo-item" title="xAI Grok">
      <img src="styles/logo_xai.svg" alt="xAI logo" width="21" height="23" decoding="async" />
      <span>xAI Grok</span>
    </div>
    <div class="logo-item" title="Moonshot AI">
      <img src="styles/logo_moonshot.svg" alt="Moonshot AI logo" width="25" height="25" decoding="async" />
      <span>Moonshot AI</span>
    </div>
    <div class="logo-item" title="Amazon Translate">
      <img src="styles/logo_amazon_translate.svg" alt="Amazon Translate logo" width="80" height="80" decoding="async" />
      <span>Amazon Translate</span>
    </div>
    <div class="logo-item" title="Mistral AI">
      <img src="styles/logo_mistralai.svg" alt="Mistral AI logo" width="191" height="135" decoding="async" />
      <span>Mistral AI</span>
    </div>
    <div class="logo-item" title="Cohere">
      <img src="styles/logo_cohere.svg" alt="Cohere logo" width="78" height="78" decoding="async" />
      <span>Cohere</span>
    </div>
    <div class="logo-item" title="Stability AI">
      <img src="styles/logo_stabilityai.svg" alt="Stability AI logo" width="103" height="86" decoding="async" />
      <span>Stability AI</span>
    </div>
    <div class="logo-item" title="Minimax">
      <img src="styles/logo_minimax.svg" alt="Minimax logo" width="25" height="21" decoding="async" />
      <span>Minimax</span>
    </div>
    <div class="logo-item" title="Amazon Transcribe">
      <img src="styles/logo_amazon_transcribe.svg" alt="Amazon Transcribe logo" width="80" height="80" decoding="async" />
      <span>Amazon Transcribe</span>
    </div>
    <div class="logo-item" title="AI21 Labs">
      <img src="styles/logo_ai21.svg" alt="AI21 Labs logo" width="38" height="36" decoding="async" />
      <span>AI21 Labs</span>
    </div>
    <div class="logo-item" title="Anthropic">
      <img src="styles/logo_anthropic.svg" alt="Anthropic logo" width="24" height="24" decoding="async" />
      <span>Anthropic</span>
    </div>
    <div class="logo-item" title="Z.ai">
      <img src="styles/logo_zai.svg" alt="Z.ai logo" width="30" height="30" decoding="async" />
      <span>Z.ai</span>
    </div>
    <div class="logo-item" title="Amazon Nova">
      <img src="styles/logo_amazon_nova.svg" alt="Amazon Nova logo" width="80" height="80" decoding="async" />
      <span>Amazon Nova</span>
    </div>
    <div class="logo-item" title="Google Gemma">
      <img src="styles/logo_google.svg" alt="Google logo" width="23" height="23" decoding="async" />
      <span>Google Gemma</span>
    </div>
    <div class="logo-item" title="Luma AI">
      <img src="styles/logo_luma.svg" alt="Luma AI logo" width="24" height="24" decoding="async" />
      <span>Luma AI</span>
    </div>
    <div class="logo-item" title="Twelve Labs">
      <img src="styles/logo_twelvelabs.svg" alt="Twelve Labs logo" width="600" height="600" decoding="async" />
      <span>Twelve Labs</span>
    </div>
    <div class="logo-item" title="Amazon Comprehend">
      <img src="styles/logo_amazon_comprehend.svg" alt="Amazon Comprehend logo" width="80" height="80" decoding="async" />
      <span>Amazon Comprehend</span>
    </div>
    <div class="logo-item" title="Writer">
      <img src="styles/logo_writer.svg" alt="Writer logo" width="40" height="40" decoding="async" />
      <span>Writer</span>
    </div>
  </div>
</div>

<!-- ACT 3 — Compliance / sovereignty -->
<div class="band band--plain" markdown>
<div class="carousel__kicker">FOR REGULATED WORKLOADS</div>

## No third party sits between your users and your models.

Unlike SaaS gateways, stdapi.ai is infrastructure you run. There is no vendor endpoint in the request path — your traffic goes from your application to your own deployment to AWS.

<div class="grid cards" markdown>

- :material-shield-lock: __Runs in your account__
  <br>Inference stays on the AWS services you enable. Bedrock does not share your prompts with model providers or use them for training.

- :material-earth: __Region allow-lists__
  <br>Pin workloads to approved regions, disable global routing, or use geography-pinned inference profiles where supported.

- :material-key: __Customer-managed encryption__
  <br>Bring your own KMS key for data at rest, with prompt and response bodies unlogged unless you enable it.

- :material-shield-star: __Security Hub aligned__
  <br>Terraform module built against AWS FSBP controls; GuardDuty and DNS Firewall opt-ins close the gaps.

</div>

AWS compliance certifications apply to the AWS services and regions you choose — they are not inherited by stdapi.ai or by your application. [:octicons-arrow-right-24: Data sovereignty &amp; compliance guide](operations_compliance.md)
{ .band__note }

<div class="qualified-card">
<img src="styles/aws_qualified_software_badge_dark.png" alt="AWS Qualified Software badge" width="120" height="120" decoding="async" loading="lazy" />
<div><strong>AWS Qualified Software</strong><br/>Verified by AWS against its technical and security requirements for AWS Marketplace.</div>
</div>

</div>

<!-- How it compares -->
## How it compares

All four expose an OpenAI-compatible API in front of Amazon Bedrock — the coverage differs. stdapi.ai is AWS-only, and therefore AWS-deep: if you need multi-cloud routing or per-key spend budgets, LiteLLM is the better fit. Competitor capabilities verified against official sources on 5 August 2026.

<div class="compare" role="region" aria-label="Feature comparison" tabindex="0" markdown>

|                                                                                             | stdapi.ai                | LiteLLM                     | Access Gateway              | Bedrock Mantle              |
| ------------------------------------------------------------------------------------------- | ------------------------ | --------------------------- | --------------------------- | --------------------------- |
| Full multi-modal API — images, video, audio, files                                            | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> |
| OpenAI + Anthropic + Cohere protocols                                                         | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> |
| Multi-region capacity — combine independent regional quotas                                   | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> |
| Regional retry — throttling, region outages, retired models                                   | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> |
| Zero-config model discovery — every region, Bedrock + Mantle                                  | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> |
| AWS AI services &amp; advanced Bedrock features — Polly, Transcribe, guardrails, service tiers | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> |
| Own AI &amp; media APIs exposed as MCP tools                                                  | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> |
| Multi-provider routing beyond AWS                                                             | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> |
| Spend limits enforced at request time                                                         | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> |
| Per-request cost tracking &amp; observability                                                 | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-n" aria-hidden="true">—</span><span class="sr-only">not available</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> |
| Production AWS deployment — Terraform, auto-scaling, optional WAF                             | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> | <span class="m-p" aria-hidden="true">◐</span><span class="sr-only">partial</span> | <span class="m-y" aria-hidden="true">✓</span><span class="sr-only">full</span> |

</div>

MCP is not a like-for-like row: stdapi.ai exposes its own AI and media endpoints as tools, while LiteLLM gateways external MCP servers — related capabilities that solve different problems.
{ .compare__legend }

<div class="compare__legend" markdown>
<span class="m-y" aria-hidden="true">✓</span> full &nbsp; <span class="m-p" aria-hidden="true">◐</span> partial / manual setup &nbsp; <span class="m-n" aria-hidden="true">—</span> not available &nbsp;·&nbsp; [Full comparison](features.md#how-stdapiai-compares)
</div>

<div class="buttons" markdown>
[See pricing](#transparent-pricing){ .md-button }
</div>

<!-- What teams run on it -->
## Verified against the tools teams already use

Every integration is the same four steps: deploy, copy your endpoint URL, paste it into the tool's settings, then name a model the deployment serves — picked from the whole catalogue, not one vendor's list. There is no step five. The tools in **bold** are driven end to end by an automated suite against a real deployment — not just documented.

<div class="usecases" markdown>

<div class="usecase" markdown>
<div class="usecase__tag">PRIVATE CHATGPT</div>
<div class="usecase__title">Enterprise chat</div>
<div class="usecase__body">ChatGPT-style assistant for your organization — chat, voice, images, and RAG, with every conversation staying in your account.</div>
<div class="usecase__tools"><strong>Open WebUI</strong> · <strong>wyoming-openai</strong> · LobeHub · LibreChat</div>
[Open WebUI guide](use_cases_openwebui.md)
</div>

<div class="usecase" markdown>
<div class="usecase__tag">CODING AGENTS</div>
<div class="usecase__title">AI-assisted development</div>
<div class="usecase__body">Frontier coding models in your IDE and terminal — without sending your codebase to a third-party AI cloud.</div>
<div class="usecase__tools"><strong>Claude Code</strong> · <strong>Codex</strong> · <strong>Qwen Code</strong> · <strong>pi</strong> · OpenCode · Zed</div>
[Coding assistants guide](use_cases_coding_assistants.md)
</div>

<div class="usecase" markdown>
<div class="usecase__tag">NO-CODE AUTOMATION</div>
<div class="usecase__title">AI in business workflows</div>
<div class="usecase__body">Add AI steps to business processes with visual workflow builders — classification, summarization, content generation.</div>
<div class="usecase__tools"><strong>n8n</strong> · <strong>Haystack</strong> · Dify · Langflow · Flowise</div>
[n8n guide](use_cases_n8n.md)
</div>

<div class="usecase" markdown>
<div class="usecase__tag">AUTONOMOUS AGENTS</div>
<div class="usecase__title">Agents you control</div>
<div class="usecase__body">Self-directed agents on infrastructure you own — with the built-in MCP server exposing every endpoint as an agent tool, and Cognito tokens giving each caller its own identity.</div>
<div class="usecase__tools"><strong>OpenClaw</strong> · <strong>Hermes</strong> · <strong>LangChain</strong> · <strong>Pydantic AI</strong> · <strong>OpenAI Agents SDK</strong> · <strong>Agno</strong> · LangGraph · CrewAI</div>
[Autonomous agents guide](use_cases_autonomous_agents.md)
</div>

<div class="usecase" markdown>
<div class="usecase__tag">VOICE &amp; AUDIO</div>
<div class="usecase__title">Speech in, speech out</div>
<div class="usecase__body">Speech-to-speech agents over a single WebSocket, transcription streamed phrase by phrase, and subtitles — on Amazon Bedrock, Polly and Transcribe, without a second AI vendor.</div>
<div class="usecase__tools"><strong>wyoming-openai</strong> · <strong>Pipecat</strong> · <strong>LiveKit Agents</strong> · Home Assistant</div>
[Home Assistant voice guide](use_cases_home_assistant.md)
</div>

<div class="usecase" markdown>
<div class="usecase__tag">RAG &amp; SEARCH</div>
<div class="usecase__title">Answers grounded in your data</div>
<div class="usecase__body">Retrieval through one deployment — built-in vector stores or the knowledge base you already run, embeddings, and Cohere-compatible reranking.</div>
<div class="usecase__tools"><strong>Haystack</strong> · <strong>Docling Serve</strong> · <strong>LlamaIndex</strong> · RAGFlow · LightRAG</div>
[RAG pipelines guide](use_cases_rag.md)
</div>

</div>

Media generation, knowledge management and team chatbots are covered too. [:octicons-arrow-right-24: All use cases &amp; integration guides](use_cases.md)
{ .usecases__more }

<!-- Compatibility evidence -->
<div class="band band--plain" markdown>
<div class="carousel__kicker">PUBLIC ENGINEERING EVIDENCE</div>

## Compatibility you can inspect

&ldquo;Compatible&rdquo; should mean more than one successful chat request. The test suite is public, and the same test bodies also run against the real OpenAI, Anthropic, and Cohere endpoints — so compatibility is measured against the originals, not asserted.

<div class="grid cards" markdown>

- :material-test-tube: __6,000+ test cases__
  <br>Run against real AWS services rather than mocks.

- :material-account-check: __20 client &amp; framework suites__
  <br>Real CLIs, apps, and libraries driven end to end against a live deployment.

- :material-brain: __100+ model-probe records__
  <br>Committed observations of what each model actually accepts and rejects.

- :material-robot: __80+ MCP API tools__
  <br>Every exposed tool called end to end through the official MCP client.

- :material-shield-check: __95%+ branch coverage__
  <br>Measured across the full suite, with every test tier enabled.

</div>

[:octicons-arrow-right-24: Inspect the public test suite](https://github.com/stdapi-ai/stdapi.ai/tree/main/tests) &nbsp;·&nbsp; [what each client suite exercises](https://github.com/stdapi-ai/stdapi.ai/blob/main/tests/agentic/README.md)
{ .band__note }

</div>

<!-- Pricing — the closing conversion moment -->
## Transparent pricing

Start local, graduate to AWS — same API, same SDKs. And zero lock-in: leaving is the same client-side change that got you in.

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
<p>Per running container — the Terraform module defaults to one per Availability Zone. No markup on model usage: pay Bedrock rates directly. Hardened container, Terraform module, commercial support (1 business day), no AGPL obligations. Billed through AWS Marketplace onto your existing AWS invoice — no new vendor onboarding.</p>
<p><a class="md-button md-button--primary" href="operations_getting_started/">Start 14-day free trial</a></p>
</div>
<div class="pricing__col pricing__col--offer">
<div class="pricing__tier pricing__tier--accent">Private offer · buy on your terms</div>
<div class="pricing__price">$0.09 <small>/container-hour</small></div>
<p>Custom terms and duration, committed usage, and a preferential rate — procured through your existing AWS relationship, so there's no new vendor to onboard. Want to try first? Use the free trial, then accept your offer.</p>
<p><a class="md-button" href="contact/#private-offer">Request a private offer</a></p>
</div>
</div>

<div class="footer-line" markdown>
Questions before you commit? [Talk to the founder](contact.md) · [GitHub](https://github.com/stdapi-ai/stdapi.ai) · [Documentation](getting_started.md)
</div>
