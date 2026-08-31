---
title: Cost Management - Amazon Bedrock Gateway Spend, Pricing & Attribution
description: Understand and control what stdapi.ai costs — AWS Marketplace vs AWS-billed models and credit eligibility, infrastructure and license cost, per-request cost estimation from published AWS prices, and per-user cost attribution.
keywords: AWS Bedrock cost, AI gateway pricing, AWS Marketplace billing, AWS credits, Bedrock cost tracking, cost attribution, FinOps AWS AI, LLM cost management, CloudWatch GetMetricData cost, usage API cost
---

# :material-cash-multiple: Cost Management

Running stdapi.ai bills into three independent buckets:

| Cost                          | Billed by                                                                     | Scales with           | Covered by AWS credits            |
|:------------------------------|:------------------------------------------------------------------------------|:----------------------|:-----------------------------------|
| **AI service usage**          | AWS (Amazon Bedrock, Polly, Transcribe, …) **or** AWS Marketplace, per model  | Requests and tokens   | Only the AWS-billed models         |
| **Gateway infrastructure**    | AWS (ECS Fargate, ALB, S3, CloudWatch, …)                                     | Running containers    | Yes                                |
| **Gateway license**           | AWS Marketplace (Commercial License only)                                      | Container-hours       | No — Marketplace is never credited |

**AI service usage dominates any normal deployment.** The gateway runs on a small ARM64 container, so its infrastructure and license form a low fixed floor that model spend passes quickly. Read the AI-cost sections first; [Gateway Cost](#gateway-cost) matters mainly at low traffic or under a hard cost constraint.

stdapi.ai **adds no markup**: model usage is billed to you by AWS at the same rate as calling Bedrock directly.

---

## :material-tag-search: Knowing the Price Before You Call

The [Model Pricing API](api_model_pricing.md) (`GET /model_pricing`) exposes the AWS unit prices stdapi.ai resolved for each model, from the same published-price catalog it uses to estimate costs. Use it for cost-aware model selection — comparing candidates before routing traffic, rather than discovering the rate afterwards in the logs. Without a deployment, the [Models](models.md) page publishes the same rates per region and tier.

---

## :material-store: Marketplace-Billed vs AWS-Billed Models

A separate **AWS Marketplace** line on your bill for Bedrock usage is expected, not an error. AWS states it plainly:

!!! quote "[Amazon Bedrock FAQ](https://aws.amazon.com/bedrock/faqs/) — *Why do I see a billing entry for AWS Marketplace for my usage of Amazon Bedrock?*"
    "Customers will see an AWS Marketplace bill for certain Bedrock serverless models and Bedrock Marketplace models. This is because these models are sold by third party providers as *Third-Party Content*, as described in the AWS service terms section 50.12."

### Why It Matters: AWS Credits

This is the practical consequence, and it is easy to get wrong when planning a budget:

!!! warning "Promotional credits do not apply to AWS Marketplace charges"
    The [AWS Promotional Credit Terms](https://aws.amazon.com/awscredits/) exclude AWS Marketplace from eligible services: *"Promotional Credit will not be applied to any fees or charges for Amazon Mechanical Turk, AWS Managed Services, Ineligible AWS Support, **AWS Marketplace**, … (collectively, 'Ineligible Services')."*

    A model billed through AWS Marketplace therefore **consumes real money even when your account holds AWS credits**, while the same workload on an AWS-billed model would draw them down. If you are running on credits, promotional funding, or an activate/startup programme, model choice changes your actual outlay — not just your reported spend.

The same applies to committed-spend agreements: check whether your EDP or Private Pricing Agreement counts Marketplace spend before assuming a discount applies.

### Which Models Are Which

Some models on Amazon Bedrock are sold as third-party AWS Marketplace listings; the rest bill as ordinary Amazon Bedrock usage. Open-weight models are generally in the second group. Which side a model sits on is the provider's and AWS's decision, it changes without notice, and a model can move after launch — so check your own account rather than trusting any list, this page included.

!!! tip "How to check, and what each check can prove"
    - **Your invoice is the authority.** Group Cost Explorer by service, or read the AWS Marketplace line on the bill. Nothing else settles it.
    - A model that needs `aws-marketplace:Subscribe` on first use **is** a Marketplace listing. AWS documents the [permission error](https://repost.aws/knowledge-center/bedrock-resolve-marketplace-permission) this produces.
    - In the AWS Price List, an *"(Amazon Bedrock Edition)"* product name marks a Marketplace listing — the signal stdapi.ai's price catalog parses. It confirms one when present, but **cannot rule one out**: models do bill through Marketplace while carrying no such product name, no `MP:` usage type, and no price-list row in commercial regions at all.

### What stdapi.ai Does About It

- **Both paths are priced.** [Cost tracking](#cost-tracking-real-time-aws-pricing) ingests Marketplace listings and native Bedrock rows alike, so a request log entry carries a cost regardless of how AWS bills it.
- **Subscriptions are handled automatically.** AWS creates the Marketplace subscription on first invocation; stdapi.ai keeps a not-yet-subscribed listing in the catalogue so that call can happen, when [`AWS_BEDROCK_MARKETPLACE_AUTO_SUBSCRIBE`](operations_configuration.md#bedrock-marketplace-auto-subscribe) is enabled (the default), and hides it otherwise. It requires `aws-marketplace:Subscribe` and `aws-marketplace:ViewSubscriptions` — see [IAM Permissions](operations_configuration.md#bedrock-iam). Without them, the first call to a third-party model fails with `AccessDeniedException`.
- **Cost tracking does not separate the two lines.** Request logs report what a call cost, not which AWS invoice section it lands on. Use Cost Explorer, grouped by service, to see the Marketplace split.

---

## :material-currency-usd: Cost Tracking (Estimated from Published AWS Prices) { #cost-tracking-real-time-aws-pricing }

Cost tracking is **opt-in and off by default**. When `COST_TRACKING=true` is enabled, stdapi.ai estimates the cost of every request from AWS's published prices — an estimate computed at request time, not your billed amount read back from AWS. Costs are attributed to the actual region where each request was served.

### How It Works

1. **Price Catalog**: At startup, stdapi.ai fetches the AWS Price List API for all configured regions and services in a background task, then caches in memory. Server readiness never waits on it: requests served before the load completes simply record usage without a cost. Failed loads are retried with exponential backoff (1–15 min), and each attempt's outcome is logged as a `background` event named `price_catalog_load`
2. **On-Demand Refresh**: The catalog is refreshed whenever a newly available Bedrock model is discovered with no catalog entry yet — not on a proactive schedule. If that refresh's AWS call fails (Price List throttling, missing permissions), the failure never propagates: the errors are captured per region and service rather than raised, the triggering request completes normally, and the new model stays unpriced until a later refresh succeeds. The refresh itself emits no per-request diagnostic — the miss surfaces on first use through the unpriced-model `error_detail` described in step 5
3. **Per-Request Computation**: For each request, costs are computed by multiplying billed quantities by the resolved unit price
4. **Built-in Defaults**: A few models are not in the Price List API at all — the Stability AI Image Services, priced on the [AWS pricing page](https://aws.amazon.com/bedrock/pricing/), and OpenAI's hosted GPT models, priced on their Bedrock model cards. Those published rates ship built in, used only when AWS publishes no row for the model; `COST_PRICE_OVERRIDES` still takes precedence
5. **Fallback on a Missing Price**: Once the catalog has been fetched, if a specific model/dimension has no resolvable price in it, the cost field is omitted for that entry rather than blocking the request, and the request log carries a `warning`-level `error_detail` naming the model and unpriced dimensions (a hint to supply the missing price via `COST_PRICE_OVERRIDES`)

!!! warning "Pricing is an estimate, not a bill"
    stdapi.ai resolves prices from AWS's own Price List API and does its best to match every request to the right unit price — including tier, cache TTL, cross-region routing, region fallback, and image resolution/quality where applicable. This is still a **best-effort approximation**, not a guarantee: AWS's Price List API doesn't reliably map a Bedrock model ID to its own pricing rows, some pricing dimensions aren't modeled at all (see [Known Limitations](#known-limitations)), and fallbacks (regional, tier) substitute a nearby price when the exact one isn't published. For billing-critical use, always reconcile against AWS Cost Explorer or your actual invoice.

### Configuration

| Setting | Default | Description |
|:--------|:--------|:------------|
| `COST_TRACKING` | `false` | Enable/disable per-request cost estimation (needs `pricing:GetProducts`) |
| `COST_PRICE_OVERRIDES` | `{}` | JSON map for operator-supplied prices for models not in AWS catalog |

### Request Log Format

Each usage entry includes cost and currency when resolved:

```json
{
  "service": "bedrock-runtime",
  "model": "anthropic.claude-sonnet-5",
  "operation": "/v1/chat/completions",
  "region": "us-east-1",
  "input_tokens": 1500,
  "output_tokens": 450,
  "cost": "0.004575",
  "currency": "USD"
}
```

Costs are plain-decimal strings rather than floats, so no precision is lost and no exponent or trailing zeros appear. The request-level total is also logged:

```json
{
  "cost": {
    "USD": "0.012345"
  }
}
```

### EMF Cost Metric

When CloudWatch metrics are enabled, a `Cost` metric (unit: None) is emitted under the `["Model", "Currency"]` dimension set — a **separate directive** from the quantity metrics, which stay under `["Model"]`. A single directive spanning both sets would also publish `Cost` bare-by-`Model`, silently summing across currencies. Query `Cost` with **both** `Model` and `Currency` dimensions:

```json
{
  "_aws": {
    "CloudWatchMetrics": [
      {
        "Namespace": "stdapi",
        "Dimensions": [["Model"]],
        "Metrics": [{"Name": "InputTokens", "Unit": "Count"}]
      },
      {
        "Namespace": "stdapi",
        "Dimensions": [["Model", "Currency"]],
        "Metrics": [{"Name": "Cost", "Unit": "None"}]
      }
    ]
  },
  "Model": "anthropic.claude-sonnet-5",
  "Currency": "USD",
  "Cost": 0.004575,
  "InputTokens": 1500
}
```

### Regional Price Fallback

Some models — mostly older/deprecated ones — aren't published in every region's Price List (e.g. priced in `us-east-1` but not any EU region). A region with no price for a given model/dimension/tier always borrows one from a nearby region instead of omitting the cost:

1. Prefers another region in the same geography (`eu-west-3` tries other `eu-*` regions first)
2. Falls back to `us-east-1`, `eu-west-1`, or `us-west-2` — the regions always fetched regardless of your configured Bedrock regions
3. If neither is available, the cost is omitted as usual

This is a substitute price, not the actual published price for that region.

### Multi-Currency Support

stdapi.ai detects currency from the AWS partition:
- Standard AWS: USD
- AWS European Sovereign Cloud (EUSC): EUR
- AWS US GovCloud: USD
- AWS China: CNY

Costs are **never summed across currencies** — this safety behavior is always on, regardless of settings. It matters when a single request's billed dimensions resolve to different currencies, which can happen with [regional fallback](#regional-price-fallback) crossing a partition boundary (e.g. a EUSC deployment falling back to a standard-AWS-priced region). A usage entry that spans more than one currency reports a `costs` map (instead of `cost`/`currency`) with every currency's own amount:

```json
{
  "service": "bedrock-runtime",
  "model": "anthropic.claude-sonnet-5",
  "input_tokens": 1500,
  "output_tokens": 450,
  "costs": {
    "EUR": "0.0021",
    "USD": "0.0028"
  }
}
```

When only one currency is involved, the entry's `cost`/`currency` reports that currency directly instead. The request-level `cost` total still aggregates cleanly per currency either way.

### Routing-Tier Pricing

AWS prices some models differently per serving profile: the cross-region "global" routing profile is *lower* than the plain/regional rate (confirmed live: Claude Sonnet 4.5 input tokens at $3.30/M regional vs $3.00/M global; GPT-5.6 Luna, Sol and Terra are each about 9% cheaper on it — the ratio is regional = global × 1.1, so global is ~9% cheaper and regional ~10% dearer, which are the same figure read in opposite directions), while latency-optimized serving (requested via the `X-Amzn-Bedrock-PerformanceConfig-Latency: optimized` header) is *higher*. stdapi.ai tracks the profile that served each request (`"routing": "global"` or `"latency"`) and prices it at the matching rate. A geographic (`us.`/`eu.`/`apac.`) profile is billed at the regional rate, which is what AWS charges for it.

Falling back is normal, not a gap: where AWS publishes no distinct global rate for a model, the request is priced at that model's regional rate rather than left uncosted. This covers models whose rates come from the AWS pricing page or a model card instead of the Price List API, too. Bedrock Mantle offers no cross-region inference at all — a Mantle request is always served in the Region it was sent to, and always billed at the In-Region rate.

### Service Tiers as a Cost Lever

AWS prices each [service tier](operations_configuration.md#default-model-service-tiers-section) differently: `flex` trades latency for a lower rate, `priority` does the opposite. Two ways to apply one without changing client code: `DEFAULT_MODEL_SERVICE_TIERS` pins a tier per model, and a [model alias](operations_configuration.md#model-aliases-configuration) pins one per alias — publishing, say, a `flex` name for batch workloads and a `priority` name for interactive ones over the same model. Set [`AWS_BEDROCK_ALLOW_SERVICE_TIER_OVERRIDE`](operations_configuration.md#aws-bedrock-allow-service-tier-override) to `false` to stop clients selecting another tier, and the cost profile you configured holds. Both settings and the override gate cover models served through the Bedrock Converse and InvokeModel APIs; a Bedrock Mantle-served model runs on the tier its own request names.

### Batch Inference { #batch-inference }

Requests sent through the [Batch API](api_openai_batches.md) or the [Message Batches API](api_anthropic_batches.md) are billed by AWS at the model's published **batch** rate, roughly half its on-demand rate. stdapi.ai prices them at that rate, reported with `"tier": "batch"` in the usage entry, and falls back to half the standard rate when AWS publishes no batch rate for a model.

Usage is recorded **once per batch**, when the batch reaches a terminal state and its totals are read — not per request. The entry therefore appears on whichever read observed the end of the batch, which is a poll rather than the call that created it. A batch that is cancelled before its requests run records nothing.

### Tenant AWS Credentials { #tenant-aws-credentials }

A tenant that [registered an AWS credential](operations_authentication_security.md#tenant-aws-credentials) has its model invocations billed by AWS to **its own account**, not to the deployment's. The usage log stays complete but honest about who paid:

- Each such usage entry carries `"billed_to": "tenant"`, with the token quantities recorded as usual.
- **No cost is computed for those entries** — pricing them would claim spend the deployment never incurred — and they are excluded from the request's `cost` totals and from the `Cost` CloudWatch metric.
- Everything else in the same request (Amazon Polly, Transcribe, S3, Knowledge Bases, moderation guardrail calls) remains the deployment's spend and is recorded and priced exactly as before.

To see what a credentialed tenant spends, read the *tenant's* AWS bill: Amazon Bedrock usage appears in the tenant account's Cost Explorer, attributable to the role sessions named `stdapi-ai-tenant-<key id>`.

### Vector Stores { #vector-stores }

Indexing a file into a [vector store](api_openai_vector_stores.md) costs one embedding call per passage, and a search costs one per query. Those calls are recorded and priced like any other embedding usage, against the model in [`VECTOR_STORE_EMBEDDING_MODEL`](operations_configuration.md#vector-store-embedding-model) — a large file is many passages, so the cost of an attach scales with the file, not with the request.

Indexing runs after the response is sent, so its usage is reported on a `background` log event of its own rather than on the request that started it. Correlate the two through the `id` field they share.

The **storage and request charges of the vector storage itself are not recorded**: like the bytes the [Files API](api_openai_files.md) stores, they appear on your AWS bill and not in the usage log. Read them from AWS Cost Explorer, filtered on Amazon S3 Vectors — the service holding [`AWS_S3_VECTORS_BUCKET`](operations_configuration.md#aws-s3-vectors-bucket).

A deployment that sets [`AWS_SQS_VECTOR_STORE_QUEUE_URL`](operations_configuration.md#aws-sqs-vector-store-queue-url) adds Amazon SQS requests to that bill — a handful per attached file, plus one long-poll receive per idle server every 20 seconds. It adds **no embedding cost**: a job replayed after a server was replaced skips whatever already completed, so a file is never embedded twice.

A [knowledge base store](api_openai_vector_stores.md#knowledge-base-stores) is billed differently, because the retrieval happens inside Amazon Bedrock and no embedding call of this server's is involved:

- A **fully managed** knowledge base charges a flat rate per retrieval call, with the document parsing, the embedding and the reranking included in it. Every query of a search is one retrieval, recorded as one `search_units` unit under the model `amazon.bedrock-knowledge-base` and priced from the rate on the [Amazon Bedrock pricing page](https://aws.amazon.com/bedrock/pricing/) — the Price List API publishes no row for it, so that rate is built in and can be replaced through `COST_PRICE_OVERRIDES`.
- A knowledge base whose **vector storage you provisioned yourself** has no published per-retrieval rate: its search costs whatever its embedding model and its vector database charge, both billed outside this server's view. Nothing is recorded for it rather than a figure that would be invented.
- **Attaching a file records nothing, on either kind.** AWS publishes no per-document ingestion charge — a fully managed knowledge base includes parsing and embedding in its rates and bills its index storage monthly per GB of raw data, and the other kind bills the embedding model and the parser it is configured with.

Read all of these from AWS Cost Explorer, filtered on Amazon Bedrock.

### Bedrock Marketplace Model Endpoints { #bedrock-marketplace-model-endpoints }

A model served from a [Bedrock Marketplace model endpoint](https://docs.aws.amazon.com/bedrock/latest/userguide/what-is-bedrock-marketplace.html) you deployed to your own AWS account is billed on an entirely different model from everything else on this page: **by the instance-hour, not the token.**

!!! danger "There is no scale-to-zero on this path"
    The endpoint runs on dedicated instances and is billed for every hour it exists, whether or not a single request is ever sent to it — the instance count has a minimum of one. Deleting the endpoint is the only thing that stops the charge.

Real on-demand SageMaker hosting rates for one `ml.g6.xlarge` instance, read from the AWS Price List API on 2026-08-26:

| Region      | Hourly     | ≈ Daily | ≈ Monthly |
|:------------|:-----------|:--------|:----------|
| `eu-west-3` | $1.4302    | $34     | $1,044    |
| `eu-west-1` | $1.25776   | $30     | $918      |

Check current rates for your own instance type and region on the [SageMaker pricing page](https://aws.amazon.com/sagemaker/pricing/) before deploying — these move, and they are not what stdapi.ai's own [cost tracking](#cost-tracking-real-time-aws-pricing) reports for this path (see below).

!!! warning "A proprietary listing adds a software charge on top"
    Some Marketplace listings bill a **provider software charge** per instance-hour, in addition to the instance rate above. stdapi.ai can neither discover nor report that rate — it is published on the listing's own AWS Marketplace page, not in any API. Open-weights listings (Apache-2.0 and similar) carry no such charge.

**What the gateway reports.** Token counts for a Marketplace model endpoint are recorded in the request log and in [`/model_pricing`](api_model_pricing.md) under a service of their own — the same treatment as [a customer-provisioned knowledge base's search](#vector-stores), and for the same reason: **no unit price is resolved and none is invented**. AWS publishes no per-token rate for this path, because it does not bill per token. A request against a model endpoint therefore reports quantities with no cost, and raises no missing-price warning — that warning is reserved for a real gap in the price catalogue, not for a billing model this app has no rate to report. Read the actual charge from AWS Cost Explorer or the SageMaker hosting line of your bill.

#### Deploy, Test, Destroy

Because the meter runs from creation to deletion regardless of traffic, the only honest way to exercise this feature is **deploy → test → delete**, every time — create the endpoint, run whatever you need against it, then delete it immediately. Never leave one running.

!!! danger "Guarantee the teardown, don't remember it"
    Wrap the delete in a `trap`/`finally`, not a manual last step. A leaked endpoint bills silently, and because the per-instance-type endpoint quota can be as low as 1 or 2, it also blocks your next deployment of that instance type with a resource-limit error until the leaked one is found and removed.

A full cycle is roughly 10–15 minutes to deploy, a few minutes to use, a few minutes to delete — under $2 at the rates above:

```bash
aws bedrock create-marketplace-model-endpoint \
  --region <region> \
  --endpoint-name <name> \
  --model-source-identifier arn:aws:sagemaker:<region>:aws:hub-content/SageMakerPublicHub/Model/<listing>/<version> \
  --accept-eula \
  --endpoint-config "sageMaker={initialInstanceCount=1,instanceType=ml.g6.xlarge,executionRole=<role-arn>}"

# ... run your test, then always:
aws bedrock delete-marketplace-model-endpoint --region <region> --endpoint-arn <arn>
```

- `--accept-eula` accepts the listing's licence on your behalf — exactly why the gateway never calls either command itself: it only discovers an endpoint you already created, and never creates, updates or deletes one.
- The endpoint's execution role must be named to match `*Sagemaker*ForBedrock*` if you use AWS's documented `iam:PassRole` policy for this role.
- Deployment is gated by the **SageMaker endpoint-usage quota per instance type**, often 1 or 2 by default and adjustable through Service Quotas — request an increase before you need it, not after a deploy fails on it.

### SageMaker AI Endpoints { #sagemaker-endpoints-cost }

A model served from an [Amazon SageMaker AI endpoint](features.md#sagemaker-endpoints) you run is billed the same way as a Marketplace model endpoint: **by the instance-hour, not the token.** The same `ml.g6.xlarge` rates apply, and the same caveat about checking them on the [SageMaker pricing page](https://aws.amazon.com/sagemaker/pricing/) for your own instance type and Region.

**What the gateway reports.** Token counts are recorded in the request log and in [`/model_pricing`](api_model_pricing.md) under a service of their own, with **no unit price resolved and none invented** — AWS publishes no per-token rate for this path, because it does not bill per token. A request against a SageMaker AI endpoint therefore reports quantities with no cost, and raises no missing-price warning. Read the actual charge from AWS Cost Explorer, on the `*-Host:ml.*` usage type of the endpoint's Region.

**Unlike a Marketplace model endpoint, this path can cost nothing at rest.** An endpoint configured for [scale to zero](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-zero-instances.html) — inference components, a production variant with `MinInstanceCount: 0`, an Application Auto Scaling target with a minimum capacity of zero, and a step-scaling policy driven by a CloudWatch alarm on `NoCapacityInvocationFailures` — releases its instances when nothing is calling it, and provisions one again on the next request. Four things to budget for:

- **The wake is a few minutes**, absorbed by [`AWS_SAGEMAKER_WARMUP_TIMEOUT`](operations_configuration.md#aws-sagemaker-warmup-timeout) so the caller sees a slow request rather than an error.
- **The scale-in tail costs more than the wake.** AWS keeps the instance for a while after the last request, so one idle-to-idle cycle bills for noticeably longer than the traffic that caused it.
- **The alarm outlives the traffic.** `NoCapacityInvocationFailures` stays in ALARM for its evaluation window after the last rejected request, and the step-scaling policy keeps acting on it — so an endpoint scaled in by hand comes straight back up, and a burst of requests against a cold endpoint bills for 10 to 15 instance-minutes after the last caller has gone. Let it settle by itself rather than writing the copy count against the policy that owns it.
- **The CloudWatch alarm** that makes the wake happen at all lists at $0.10 per alarm-metric-month, inside CloudWatch's always-free allowance of ten alarms.

Everything else about an idle scale-to-zero endpoint is unbilled: no instances means no instance-hours and no EBS volume, the model, endpoint-configuration and inference-component objects carry no charge, Application Auto Scaling is free, and `AWS/SageMaker` metrics are not billed as custom metrics.

### Long-Context Pricing

Some 1M-context-capable models are billed by AWS at a higher rate — roughly double for input-side tokens — once a call's prompt (input + cache read/write tokens) passes the boundary that model publishes. **That boundary differs per model**: Claude Sonnet 4 (via the `context-1m` `anthropic-beta` flag) switches at 200K tokens, while the OpenAI GPT-5.6 models bill their short-context rate all the way to 272K. stdapi.ai applies each model's own boundary, prices the whole call at the published long-context rate, and reports it with `"context": "long"` in the usage entry. When AWS publishes no long-context rate for a model, the standard rate is used as the best available estimate.

### Built-in Tool Pricing

Amazon Nova's built-in grounding tool (`web_search` → `nova_grounding`, supported by Nova 2 and Nova Premier) is billed by AWS per grounding request, on top of token usage. stdapi.ai counts grounding invocations in the model response and reports them as `grounding_requests`, priced from the model's published per-request rate.

The Amazon Bedrock web search tool the OpenAI GPT-5.x family uses on [`/v1/responses`](api_openai_responses.md#openai-gpt-web-search) is billed per query at one flat rate — the same for every model — in every Region where that tool is available. Because AWS publishes it without a model, it is reported on its own usage entry under the model `amazon.bedrock-web-search`, also as `grounding_requests`, beside the invocation's own token entry. A single turn may run several queries, and one tool call may issue more than one, so the count comes from the queries the response reports rather than from the number of tool calls; page reads have no published per-query rate and are not counted.

### Image Pricing Granularity

Some image-generation models (currently: Amazon Titan Image Generator G1/V2 and Nova Canvas) are priced by AWS per resolution/quality combination, not a flat per-image rate. stdapi.ai automatically prices these per-image, based on the actual requested size and quality, falling back to a flat per-image rate for models where this isn't wired up yet (e.g. Stability). No configuration needed — this is purely additive precision with no accuracy trade-off.

### Media Input Pricing

Multimodal embedding inputs are billed by AWS per media unit on top of (or instead of) tokens: per input image (with a distinct "document image" rate where offered) and per second of audio/video. stdapi.ai records image counts directly and audio/video durations from the AWS-reported segment timings of the asynchronous (segmented) processing path, reported as `input_images`/`input_seconds` with their `*_by_spec` breakdowns. Rerank queries are recorded as `search_units` (one per query).

### Known Limitations

- **Speech-modality tokens** (speech-to-speech models, including [Realtime API](api_openai_realtime.md#cost) sessions): speech and text tokens are recorded in separate buckets and priced at their own published rates, so no modality is priced as the other. Where a model publishes no speech rate at all, its speech tokens fall back to the text rate, which underestimates speech-heavy calls. A Realtime session bills continuously in both directions for as long as it is open, not per request; usage is recorded per answer, matching each `response.done` event in the request log.
- **Asynchronous (segmented) embeddings**: AWS reports no token usage for this processing path (used automatically for large inputs), so segmented text embeddings report no token cost; audio/video durations are recovered from the AWS-reported segment timings and billed.
- **Synchronous audio/video embedding inputs**: media duration is only available from AWS on the segmented (asynchronous) processing path — small audio/video inputs processed synchronously report no duration and no per-second cost. No estimate is substituted (this app only reports AWS-confirmed real usage).
- **Client disconnect during streaming**: streamed chat responses still record their final usage after a disconnect. For streamed responses on other routes and for image generation jobs, AWS bills the input tokens and everything generated up to the cancellation, but no usage is recorded for that call. No estimate is substituted.
- **Rerank queries with more than 100 documents**: AWS bills one search unit per 100 documents; the document count isn't visible at recording time, so one unit per query is recorded.
- **Reserved capacity pricing**: if a request explicitly asks for AWS's Reserved Capacity service tier, its cost is computed at the standard on-demand rate instead — Reserved Capacity uses a separate monthly-commitment pricing model this app doesn't ingest. Avoid relying on this app's cost figures for Reserved Capacity workloads.
- Some very new or region-specific models may have no published price anywhere yet — AWS publishes pricing after model availability, sometimes with a delay.
- **AWS GovCloud**: the Price List API has no GovCloud endpoint, so catalog prices cannot be fetched there — usage is still recorded, a startup warning is emitted, and only `COST_PRICE_OVERRIDES` entries produce costs.

### Override Map for Missing Models

Some models — recently released ones, or region-specific listings — may not appear yet in the AWS Price List API. Use `COST_PRICE_OVERRIDES` to fill gaps (Bedrock models only — Polly/Transcribe/Translate/Comprehend prices always come from the catalog):

```bash
export COST_PRICE_OVERRIDES='{"anthropic.claude-sonnet-5":{"input_tokens":0.000003,"output_tokens":0.000015}}'
```

Prices are per **one unit** (token, character, second) in your partition's currency.

### IAM Permissions

Ensure your IAM role includes pricing read access (the Price List API serves identical data from its three commercial endpoints, and sovereign partitions such as EUSC host their own; the nearest one is selected automatically from your configured Bedrock regions):

```json
{
  "Effect": "Allow",
  "Action": ["pricing:GetProducts"],
  "Resource": "*"
}
```

---

## :material-tag-multiple: AWS Cost Attribution

Cost tracking prices **each request** as it happens. AWS-side attribution answers a different question — whose spend lands on the **AWS bill**, in [Cost Explorer](https://docs.aws.amazon.com/cost-management/latest/userguide/ce-what-is.html) and [CUR 2.0](https://docs.aws.amazon.com/cur/latest/userguide/what-is-cur.html). The two are complementary: request logs for granular analysis, AWS attribution for invoicing and chargeback.

| Dimension               | Mechanism                                                                                                                                            | Reported in                          | Setup                                                             |
|:------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------|:------------------------------------------------------------------|
| **Service / gateway**   | [IAM principal attribution](https://docs.aws.amazon.com/bedrock/latest/userguide/cost-mgmt-iam-principal-tracking.html) — AWS captures the caller identity | Cost Explorer, CUR 2.0               | Automatic; tag the execution role for finer breakdowns            |
| **Application / workload** | [Bedrock Project/Workspace](operations_configuration.md#bedrock-mantle-project) (Bedrock Mantle models)                                            | Cost Explorer, CUR 2.0               | Set `AWS_BEDROCK_MANTLE_PROJECT`                                  |
| **End user**            | `stdapi-ai.user_id` request metadata and job tags                                                                                                     | stdapi.ai logs, Bedrock invocation logs | Clients send `safety_identifier` — `user` is a deprecated alias — (OpenAI) or `metadata.user_id` (Anthropic) |
| **End user, on the AWS bill** | [Per-user role sessions](#per-user-attribution) — each user's model calls run under a session of their own                                       | Cost Explorer, CUR 2.0               | Set `AWS_BEDROCK_USER_ROLE_ARN` to a role you create                |
| **Team / tenant**       | Request metadata attached by the [model alias](operations_configuration.md#model-aliases-configuration) the client names                               | Bedrock model invocation logs only   | Give each team its own alias with a `metadata` entry, then [enable and deliver model invocation logging](https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html) |

!!! warning "Alias metadata is not a cost allocation tag"
    A model alias' `metadata` travels as Amazon Bedrock **request metadata**: it appears in [model invocation logs](https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html), which you enable and deliver to S3 or CloudWatch Logs yourself, and which it can be filtered on. It never appears in the stdapi.ai request logs, in Cost Explorer or in CUR 2.0 — splitting the bill by team still needs one of the mechanisms above.

### IAM Principal Attribution

Amazon Bedrock records the IAM identity behind every `bedrock-runtime` inference call and forwards it to Cost Explorer and CUR 2.0. Nothing to configure in stdapi.ai, and no change to client requests.

By default stdapi.ai calls AWS with a **single execution role** — the ECS task role — so all Bedrock spend is attributed to that one identity, the standard LLM-gateway pattern AWS documents. To break that total down, tag the execution role (for example `team` or `cost-center`), then activate those keys in the AWS Billing console under **Cost allocation tags**, filtering by type **IAM principal**. To split it per end user instead, see [Per-User Attribution](#per-user-attribution) below.

!!! note "Attribution grain"
    IAM principal attribution aggregates per usage type per day — it never yields a per-request cost. Use the [request logs](#cost-tracking-real-time-aws-pricing) for that.

### Per-User Attribution

Every end user is attributed at the request level out of the box: the client-supplied identifier is recorded as `stdapi-ai.user_id` in the request log — alongside that request's computed cost — and forwarded to Bedrock as request metadata, so per-user spend can be aggregated from the logs.

To split the **AWS bill itself** per end user, give each one an identity of their own. Set `AWS_BEDROCK_USER_ROLE_ARN` to a role you create, and stdapi.ai opens a short-lived session of that role per end user and runs their model calls under it. AWS then reports each user separately in Cost Explorer and CUR 2.0, from the invoice rather than from the logs.

=== ":material-cog: Configure"

    | Setting | Purpose |
    |:--------|:--------|
    | [`AWS_BEDROCK_USER_ROLE_ARN`](operations_configuration.md#aws-bedrock-user-role-arn) | The role each end user's calls run under. Enables the feature. |
    | [`AWS_BEDROCK_USER_ROLE_TAG_KEY`](operations_configuration.md#aws-bedrock-user-role-tag-key) | Session tag key carrying the user identity (`user` by default). |
    | [`AWS_BEDROCK_USER_ROLE_SESSION_DURATION`](operations_configuration.md#aws-bedrock-user-role-session-duration) | Session lifetime, 900–3600 seconds. Sessions are cached and reused. |
    | [`AWS_BEDROCK_USER_ROLE_REQUIRE_IDENTITY`](operations_configuration.md#aws-bedrock-user-role-require-identity) | Reject requests that identify no end user, instead of billing them to the gateway. |

    The role and the two IAM policies it needs are in [IAM Permissions](operations_iam_permissions.md#per-user-cost-attribution).

=== ":material-account-check: Identify the user"

    The identity is taken, in order:

    1. the **authenticated caller**, when [Amazon Cognito authentication](operations_authentication_security.md) is enabled — the identity the gateway itself verified;
    2. the identifier the request declares: `safety_identifier` (or the deprecated `user`) on the OpenAI-compatible APIs, `metadata.user_id` on the Anthropic Messages API;
    3. the key ID of the verified [tenant API key](operations_authentication_security.md#tenant-api-keys), when the request carried one — so a tenant that identifies no individual user is still attributed as itself.

    A request carrying neither runs under the gateway's own identity, unless `AWS_BEDROCK_USER_ROLE_REQUIRE_IDENTITY` is enabled, in which case it is rejected with a `400`. That rejection covers the requests that would run under the end user role, and only those — the services listed in the coverage warning below stay on the gateway's identity either way, with the one exception noted there. Some APIs — audio transcription among them — have no end user field, so there the identity can only come from an authenticated caller.

    !!! danger "A declared identifier is chosen by the caller"
        The identity is verified only when it comes from an authenticated caller. With an API key, or with no authentication, any client can send any other user's `safety_identifier`, and the resulting session — its name, its tag, its line in the bill — is that other user's. This is cost metadata, not an authorization boundary: see [Restricting a role per end user](operations_iam_permissions.md#per-user-cost-attribution) before writing an IAM policy on the session tag.

=== ":material-magnify: Read the bill"

    Each user appears as a distinct caller identity in [CUR 2.0](https://docs.aws.amazon.com/cur/latest/userguide/what-is-cur.html)'s `line_item_iam_principal` column, which holds the full ARN — `arn:aws:sts::<account-id>:assumed-role/<role>/<session>`. Two operator steps in the AWS console:

    - In **Data Exports**, enable *Include caller identity (IAM principal) allocation data* under **Additional export content** — on a new CUR 2.0 standard data export, or on an existing one through **Edit** ([editing export details](https://docs.aws.amazon.com/cur/latest/userguide/dataexports-edit-export-details.html): the report name and Billing view are fixed, the export content is not).
    - To group by user in Cost Explorer, activate the session tag key under **Billing → Cost allocation tags**, filtering by type **IAM principal**; it is then offered under **Group by → Tag** as `iamPrincipal/<key>`. The key appears there only after that identity has made at least one call, and AWS takes up to 24 hours to list it plus up to 24 hours to activate it.

    The request log's `aws_role_session_name` field records the session each request was billed under, which is what correlates a log line with a CUR row.

!!! warning "What is covered, and what is not"
    Per-user sessions apply to **model invocations**, and to the guardrail applied during them. Everything else the gateway calls on your behalf — standalone guardrail evaluations, reranking, video generation jobs and their output files, speech, transcription and translation — stays on the gateway's own identity. The one exception is a real-time speech-to-speech session, which is refused rather than billed to the gateway once [`AWS_BEDROCK_USER_ROLE_REQUIRE_IDENTITY`](operations_configuration.md#aws-bedrock-user-role-require-identity) is enabled. Two backends sign with the gateway's own credentials and carry no per-user session either, while still answering that same `400` to an unidentified request: [Bedrock Mantle](operations_configuration.md#bedrock-mantle-enabled), whose usage is attributed with [Projects](operations_configuration.md#bedrock-mantle-project) instead, and a [SageMaker AI endpoint](operations_configuration.md#aws-sagemaker-endpoints), billed by the instance-hour rather than per request.

!!! note "Cardinality"
    AWS multiplies CUR rows by the number of calling identities, and aggregates them per usage type per day. A deployment with a very large or unbounded user population gets a proportionally larger export, and still no per-request cost — the [request logs](#cost-tracking-real-time-aws-pricing) remain the per-request source.

---

## :material-chart-timeline-variant: Usage API Query Cost { #usage-api-cost }

The [Organization Usage and Costs API](api_openai_organization_usage.md) answers from the metrics this deployment publishes to Amazon CloudWatch, and **CloudWatch bills both halves of that**: the metric series it stores every month, and every metric a query reads. This is the one feature on this page whose own cost can rival a small model workload, so it is [off by default](operations_configuration.md#usage-api) and priced here rather than discovered on an invoice.

Rates below are `us-east-1`, read from the AWS Price List API (`AmazonCloudWatch`) on 2026-08-26 and cross-checked against the [CloudWatch pricing page](https://aws.amazon.com/cloudwatch/pricing/). They differ by region and they move — price your own before enabling.

| Charge                       | Rate                                                | Free tier                                             |
|:-----------------------------|:-----------------------------------------------------|:-------------------------------------------------------|
| `cloudwatch:GetMetricData`   | **$0.00001 per metric requested** ($0.01 per 1,000)  | **None — explicitly excluded**                          |
| `cloudwatch:ListMetrics`     | $0.01 per 1,000 requests                             | 1,000,000 requests per month                            |
| Stored custom metric series  | $0.30 per metric-month, first 10,000                 | —                                                       |

!!! danger "The free tier does not cover the query that matters"
    CloudWatch's 1,000,000 free API requests per month sound like they cover this. They do not: `GetMetricData` is excluded from that allowance, and it is billed **per metric series requested**, not per call — so a single call reading 108 series is charged for 108 metrics, from the first request onwards.

### What a Query Costs

On a deployment serving ~79 models, one `group_by=model` query against `/v1/organization/usage/completions` reads roughly 108 series:

| Usage pattern                               | ~79 models                | A few hundred models        |
|:--------------------------------------------|:--------------------------|:-----------------------------|
| One query                                   | ≈ **$0.00108**            | ≈ $0.003–$0.010              |
| One client polling once a minute, 30 days   | ≈ **$47/month**           | **$130–$430/month**          |

That figure is **per polling client**: two dashboards on the same schedule cost twice as much, and a shorter poll interval scales it linearly until the [response cache](operations_configuration.md#usage-api-cache-ttl) absorbs it.

### What Enabling It Costs at Rest

Turning [`USAGE_API`](operations_configuration.md#usage-api) on also publishes the usage metrics under an additional `Model`+`Operation` dimension set, so the endpoints can answer per-model *and* per-endpoint. Those extra series are stored metrics, billed monthly whether or not a single query is ever made — roughly **$120–$180/month** on a 79-model deployment, scaling with the number of models actually used.

!!! warning "`CLOUDWATCH_METRICS_USER_DIMENSION` has no upper bound"
    [`CLOUDWATCH_METRICS_USER_DIMENSION`](operations_configuration.md#cloudwatch-metrics-user-dimension) is what makes `group_by=user_id` answerable, and it adds one further series **per user × model × metric name**. Its cardinality is your caller population, so its cost is unbounded by anything the deployment controls. Enable it only where that population is small and known.

### The Four Protections

Each is a setting, and each default is chosen to keep the bill predictable:

| Protection                                                                          | Default | What it prevents                                                                                     |
|:------------------------------------------------------------------------------------|:--------|:------------------------------------------------------------------------------------------------------|
| [`USAGE_API`](operations_configuration.md#usage-api)                                | `false` | Everything above. While it is off the endpoints refuse and the extra series are not published.        |
| [`USAGE_API_CACHE_TTL`](operations_configuration.md#usage-api-cache-ttl)            | `60` s  | A polling client billing every poll. Within the TTL an identical query is served from cache and costs nothing — and a client polling faster than the bucket width learns nothing new anyway. |
| [`USAGE_API_MAX_METRICS`](operations_configuration.md#usage-api-max-metrics)        | `500`   | A wide `group_by` reading thousands of series. The query is refused **before** it is billed — see the caveat below. |
| [`USAGE_API_MAX_RANGE_DAYS`](operations_configuration.md#usage-api-max-range-days)  | `92`    | A single call asking for a year of daily buckets.                                                     |

Lower `USAGE_API_MAX_METRICS` and raise `USAGE_API_CACHE_TTL` on a large catalogue; the defaults are a ceiling, not a target.

!!! warning "`USAGE_API_MAX_METRICS` counts what a two-week index can see"
    The free listing the cap is checked against is CloudWatch's own, taken under the filters and the endpoint operations of the query being served — so narrowing a query with `models` genuinely narrows the count. That listing returns only the series **published in the last two weeks**, and the paid read finds its own series through the same index, so what is counted is what the read will match. The flip side is a limit of the surface rather than of this setting: a series idle for longer than a fortnight is reported by neither — see [a model idle for two weeks stops being reported](api_openai_organization_usage.md#retention).

!!! note "`/v1/organization/costs` reports your AWS bill, not your customers' invoices"
    It reports what **AWS bills this deployment** for serving the requests, on the same [estimated basis](#cost-tracking-real-time-aws-pricing) as the request logs. It is not a reseller's revenue figure: if you bill your own clients at a markup, or at a flat rate, that number is yours to compute and does not appear here.

---

## :material-server: Gateway Cost

Secondary for most deployments — read this section if you run under a hard cost constraint or at low traffic, where the gateway's fixed floor is a visible share of the bill.

### Infrastructure

The gateway is a single small container. The Terraform module defaults to **ARM64, 0.25 vCPU and 512 MiB** — the smallest Fargate size — which keeps this bucket cheap, though it is still a floor you pay whether or not any request arrives.

The [Terraform module](operations_getting_started.md) provisions:

| Component                         | Notes                                                                    |
|:----------------------------------|:-------------------------------------------------------------------------|
| **ECS Fargate** service           | The dominant infrastructure cost; 0.25 vCPU / 512 MiB ARM64 by default, times `autoscaling_min_capacity` |
| **Application Load Balancer**     | Hourly rate plus LCU; skipped when you attach your own                    |
| **S3 buckets** (regional)         | Input files, generated media; see `AWS_S3_VIDEOS_EXPIRES_AFTER`           |
| **S3 vector bucket** (optional)   | Created with `aws_s3_vectors_bucket_create`; billed on stored vectors and on the bytes each search reads — see [Vector Stores](#vector-stores) |
| **SQS queue** + dead-letter queue (optional) | Created with `aws_sqs_vector_store_queue_create`; a handful of requests per attached file, plus one long poll per task every 20 seconds |
| **DynamoDB table** (optional)     | Created with `aws_dynamodb_table_create`; on-demand, so nothing at rest — billed only by the features that use it, see [Model List Sharing](#model-list-sharing) |
| **CloudWatch** logs, metrics, alarms | Grows with log verbosity and EMF metric volume; the [Usage API](#usage-api-cost) adds stored metric series and a per-query charge on top |
| **KMS**, IAM, networking          | Keys and roles; NAT/VPC endpoints only when the module creates the network |
| **WAF** (optional)                | Per-rule and per-request charges when `alb_waf_enabled = true`            |

#### Cost Tiers at a Glance

Both infrastructure and license cost scale with **how many containers run and for how long** — there's no single "deployment cost," just a wide range between a minimal setup and a full production one. Three representative configurations:

| Tier | Configuration | You pay for | Main cost driver |
|:-----|:--------------|:-------------|:-------------------|
| **Minimal** | 1 ARM64 task, [Fargate Spot](operations_deploy_advanced.md#cost-optimized-deployment) (~70% cheaper than on-demand), [scheduled service hours](#keeping-it-low), no ALB — reached via [Service Discovery](operations_deploy_advanced.md#integration-with-existing-infrastructure) | Spot compute only for the scheduled hours; license for those same hours; no ALB/WAF charges at all | Hours the schedule keeps the service running — see [License](#license) for a worked example (~$17/month at ~165 h/month) |
| **Standard** | 1 on-demand task, single AZ (`subnet_ids` limited to one subnet or `autoscaling_min_capacity = 1`), ALB enabled, running 24/7 | On-demand compute for one container around the clock, plus the ALB's hourly rate and LCU usage | The ALB becomes a fixed cost on top of continuous compute; no Spot discount |
| **Full production** | One task per Availability Zone (module default), ALB + WAF, running 24/7 | Compute × number of AZs, ALB + WAF, and the per-container-hour license × total task-hours across all AZs | AZ count is the main multiplier — it scales infrastructure and license cost together |

None of these include AI service usage (Bedrock, Polly, Transcribe, …), which is billed separately by AWS at cost — see the buckets table at the top of this page.

#### Keeping It Low

- **Stop the service outside business hours.** `autoscaling_schedule_stop` and `autoscaling_schedule_start` take cron expressions, so the cluster can run weekdays only and cost nothing at night — the single largest saving for an internal workload.

    ```hcl
    autoscaling_schedule_stop  = "cron(0 19 ? * MON-FRI *)"
    autoscaling_schedule_start = "cron(0 8 ? * MON-FRI *)"
    ```

- **Reuse existing network infrastructure.** Passing `subnet_ids` and `security_group_id` deploys into your VPC and creates **no additional NAT gateways or load balancers** — see [Integration with Existing Infrastructure](operations_deploy_advanced.md). On a small deployment these are often larger than the gateway itself.
- **Scale to your real floor.** `autoscaling_min_capacity` sets how many containers run at idle. Because the Commercial License is billed per container-hour, this number drives both infrastructure *and* license cost.
- **Use Fargate Spot** and trim log retention, as the [Cost-Optimized Deployment](operations_deploy_advanced.md#cost-optimized-deployment) example does.
- **Trim observability volume.** Raising `LOG_LEVEL` and disabling request/response payload logging cut CloudWatch ingestion — see [Controlling Log Verbosity](operations_logging_monitoring.md#controlling-log-verbosity). Note that **EMF metric lines are not affected by `LOG_LEVEL`**: they are written to stdout on every request whenever `CLOUDWATCH_METRICS` is on, so disabling that setting is the only way to remove them.

!!! note "Estimating before you deploy"
    stdapi.ai publishes no infrastructure dollar estimate — the tiers above describe direction and drivers, not quotes: the total depends on your region, replica count, schedule and whether the module creates networking. Price the component list above with the [AWS Pricing Calculator](https://calculator.aws/) for your own configuration.

---

### Model List Sharing { #model-list-sharing }

[`MODEL_CACHE_SHARED`](operations_configuration.md#model-cache-shared) is the one optional feature whose own AWS cost is worth stating rather than rounding away. **Left off, or with no [`AWS_DYNAMODB_TABLE`](operations_configuration.md#aws-dynamodb-table) configured, it costs nothing**: no table is created, no request is made. The table alone, created but unused, is also $0 — DynamoDB on-demand has no idle charge, this table holds kilobytes, and its storage sits inside the 25 GB always-free allowance.

Enabled, it is not free. What it costs is the published list itself:

- **Writes dominate.** Each refresh writes the compressed model list, and DynamoDB bills a write in 1 KB units. At the default `MODEL_CACHE_SECONDS=900` that is 96 refreshes a day for the whole fleet, whoever performs them.
- **Reads round to nothing.** Each server reads the list once per interval — a few thousand read units a day — plus one small conditional write to claim the refresh.
- **Traffic does not change any of it.** The table is touched when the list expires, never per request, so this is a fixed monthly figure rather than one that scales with usage.

On [DynamoDB on-demand pricing](https://aws.amazon.com/dynamodb/pricing/on-demand/) in `us-east-1` ($0.625 per million write request units, $0.125 per million read request units, both higher in some regions), 96 refreshes a day of a list in the low hundreds of kilobytes comes to roughly **$0.60–$2 a month for the whole fleet**, plus a few cents per server for the reads.

!!! note "The size of your list is the variable"
    That range assumes the compressed list lands in the low hundreds of kilobytes, which is what a catalogue of a few hundred models across a handful of regions produces. A deployment with many regions and Marketplace endpoints will sit at the top of it or a little above; the figure scales linearly with the compressed size and inversely with `MODEL_CACHE_SECONDS`. Doubling the interval halves this cost.

    It buys back the discovery passes it replaces: *N* servers make one pass per interval between them instead of one each. Those Bedrock control-plane calls are not billed, so the saving is in start-up latency and API rate-limit headroom rather than in dollars — which is why this is worth enabling for a fleet and not worth enabling for a single container.

---

### License

stdapi.ai is [dual-licensed](operations_licensing.md): the free AGPL-3.0-or-later license costs nothing but requires sharing your modifications, while the Commercial License — billed per container-hour via AWS Marketplace, no per-request or per-token component — is what production deployments typically run. See [Licensing](operations_licensing.md) for the full pricing table and trial/private-offer terms.

!!! info "The license is itself a Marketplace charge"
    Being an AWS Marketplace subscription, the Commercial License appears on the Marketplace section of your bill and — like [Marketplace-billed models](#why-it-matters-aws-credits) — **cannot be paid with AWS promotional credits**.

---

## :material-arrow-right: Next Steps

<div class="grid cards" markdown>

- :material-chart-line: [**Logging & Monitoring**](operations_logging_monitoring.md) — Usage metrics fields, CloudWatch EMF, and log verbosity
- :material-tag-search: [**Model Pricing API**](api_model_pricing.md) — Query AWS unit prices per model for cost-aware selection
- :material-cog: [**Configuration Reference**](operations_configuration.md) — `COST_TRACKING`, `COST_PRICE_OVERRIDES` and IAM permissions
- :material-scale-balance: [**Licensing**](operations_licensing.md) — AGPL vs commercial license and AWS Marketplace subscription
- :material-email-outline: [**Contact**](contact.md) — Private offers, committed usage, and billing questions

</div>
