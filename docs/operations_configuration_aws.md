---
title: "Configuration - Regions and AWS Clients"
description: "Configure the AWS regions stdapi.ai calls, the Bedrock region router and failover, Bedrock Mantle, the shared AWS client settings and the regions of the other AWS AI services."
keywords: "AWS regions configuration, Bedrock regions, cross-region inference, region routing, failover, Bedrock Mantle, data residency, latency optimized inference"
---

# :material-aws: AWS Services and Regions

Which AWS regions serve your models, how a request fails over between them, and how the shared AWS clients behave. Part of the [Configuration Guide](operations_configuration.md).

## :material-format-list-bulleted: Settings Summary

### :material-aws: AWS Client { #summary-aws-client }

| Variable                                                | Default | Description                                                                                                 |
|---------------------------------------------------------|---------|-------------------------------------------------------------------------------------------------------------|
| [`AWS_ADAPTIVE_RETRY`](#aws-adaptive-retry)             | `false` | Enable adaptive retry mode that throttles back under congestion rather than using fixed exponential backoff |
| [`AWS_MAX_POOL_CONNECTIONS`](#aws-max-pool-connections) | `50`    | Maximum concurrent HTTP connections per AWS service client                                                  |
| [`AWS_CONNECT_TIMEOUT`](#aws-connect-timeout)           | `5`     | Timeout in seconds for establishing a connection to an AWS service endpoint, and for a real-time audio session to become ready in a region |

### :material-robot: AWS AI Services { #summary-aws-ai-services }

| Variable                                          | Default                     | Description                                                 |
|---------------------------------------------------|-----------------------------|-------------------------------------------------------------|
| [`AWS_POLLY_REGION`](#aws-polly-region)           | All `AWS_BEDROCK_REGIONS`   | Region for Amazon Polly; unset = per-engine regional discovery with automatic failover |
| [`AWS_COMPREHEND_REGION`](#aws-comprehend-region) | All `AWS_BEDROCK_REGIONS`   | Region for Amazon Comprehend (language detection, toxicity moderation); unset = automatic failover across all Bedrock regions |
| [`AWS_TRANSCRIBE_REGION`](#aws-transcribe-region) | All `AWS_BEDROCK_REGIONS`   | Region for Amazon Transcribe; unset = failover across Bedrock regions with a co-located bucket |
| [`AWS_TRANSLATE_REGION`](#aws-translate-region)   | All `AWS_BEDROCK_REGIONS`   | Region for Amazon Translate; unset = automatic failover across all Bedrock regions |

### :material-directions-fork: Resilience & Failover { #summary-resilience-failover }

| Variable                                                                                                | Default   | Description                                                                                                              |
|---------------------------------------------------------------------------------------------------------|-----------|--------------------------------------------------------------------------------------------------------------------------|
| [`AWS_BEDROCK_REGION_ROUTING`](#bedrock-region-routing)                                                 | `ordered` | Region routing strategy: `disabled`, `ordered`, `lowest_latency`, or `round_robin` ([details](operations_resilience.md)) |
| [`AWS_BEDROCK_REGION_ROUTING_QUOTA_BACKOFF_SECONDS`](#bedrock-region-routing-quota-backoff)             | `60`      | Base interval in seconds for exponential quota backoff per region                                                        |
| [`AWS_BEDROCK_REGION_ROUTING_MAX_QUOTA_BACKOFF_SECONDS`](#bedrock-region-routing-max-quota-backoff)     | `3600`    | Hard ceiling in seconds on the exponential quota backoff per region (default: 1 hour)                                    |
| [`AWS_BEDROCK_REGION_ROUTING_QUOTA_STALE_FACTOR`](#bedrock-region-routing-quota-stale-factor)           | `2`       | Multiplier on max quota backoff to determine when the consecutive-error counter resets                                   |
| [`AWS_BEDROCK_REGION_ROUTING_UNAVAILABLE_BACKOFF_SECONDS`](#bedrock-region-routing-unavailable-backoff) | `30`      | Seconds to avoid a region after unavailability errors                                                                    |
| [`AWS_BEDROCK_MAX_RETRIES`](#bedrock-max-retries)                                                       | `9`       | Cap on the retries per Bedrock invocation; with region routing, each candidate region is tried at most once              |
| [`AWS_FAILOVER_MAX_RETRIES`](#failover-max-retries)                                                     | `2`       | SDK retries per candidate region for the multi-region failover services (Polly, Transcribe, Translate, Comprehend)       |

### :material-layers-triple: Bedrock Mantle { #summary-bedrock-mantle }

| Variable                                                                    | Default               | Description                                                                                          |
|-----------------------------------------------------------------------------|-----------------------|------------------------------------------------------------------------------------------------------|
| [`AWS_BEDROCK_MANTLE_ENABLED`](#bedrock-mantle-enabled)                     | `true`                | Expose models served by the Amazon Bedrock Mantle endpoint alongside classic Bedrock Converse models |
| [`AWS_BEDROCK_MANTLE_REGIONS`](#bedrock-mantle-regions)                     | Mantle-capable subset of `AWS_BEDROCK_REGIONS` | AWS regions used for Bedrock Mantle, in failover priority order                     |
| [`AWS_BEDROCK_MANTLE_ENDPOINT_URL`](#bedrock-mantle-endpoint-url)           | None                  | Override the Bedrock Mantle endpoint URL template (`{region}` placeholder)                           |
| [`AWS_BEDROCK_MANTLE_PREFERRED_MODELS`](operations_configuration_models.md#bedrock-mantle-preferred-models)   | `openai.gpt-5.6`      | Model IDs served via Mantle even when also available on the classic bedrock-runtime endpoint. Incompatible with Guardrails; the default is a **price change** for the GPT-5.6 family |
| [`AWS_BEDROCK_MANTLE_SERVICE_HEADER`](#bedrock-mantle-service-header)       | `false`               | Honor the `x-stdapi-service: bedrock-mantle` request header to route dual-homed models through Mantle per request |
| [`AWS_BEDROCK_MANTLE_PROJECT`](#bedrock-mantle-project)                     | None                  | Default Bedrock Project/Workspace ID applied to Mantle requests for cost tracking and observability  |
| [`AWS_BEDROCK_ALLOW_MANTLE_PROJECT_OVERRIDE`](#bedrock-allow-mantle-project-override) | `false`     | Allow requests to override the configured Mantle project via the `OpenAI-Project` / `anthropic-workspace` header |
| [`AWS_BEDROCK_EXTERNAL_WEB_ACCESS`](operations_configuration_models.md#bedrock-external-web-access)           | `false`               | Let the built-in web search tool reach the public web instead of the Amazon Bedrock web index        |
| [`AWS_BEDROCK_ALLOW_EXTERNAL_WEB_ACCESS_OVERRIDE`](operations_configuration_models.md#bedrock-allow-external-web-access-override) | `false` | Allow requests to override external web access with the `external_web_access` extra model parameter |

## :material-cog: General Configuration { #general-configuration }

Settings shared by every AWS service client the gateway opens: retry pacing, connection pool size, and connection timeout.

#### `AWS_ADAPTIVE_RETRY` { #aws-adaptive-retry }

:octicons-package-24: **Purpose**
:   Enable adaptive retry mode that adjusts retry pacing based on observed error rates across all AWS service calls

:octicons-database-24: **Type**
:   Boolean (`true` / `false`)

:octicons-gear-24: **Default**
:   `false`

:octicons-workflow-24: **Behavior**
:   When enabled, the retry strategy dynamically responds to real-time congestion signals. If errors are occurring frequently, retries are spaced further apart to avoid amplifying load on an already-stressed endpoint. Once conditions improve, the pacing returns to normal. When disabled, retries follow a standard exponential backoff strategy with fixed intervals. Applies to all AWS services (Bedrock, S3, Polly, Transcribe, etc.).

!!! warning "Latency Impact"
    Adaptive retry paces retries based on real-time error signals, reducing the risk of retry storms when many clients share the same endpoint under sustained congestion — at the cost of increased per-request latency when throttling is detected, since the client intentionally delays retries to shed load. Prefer it under sustained high load; keep the default standard mode for latency-sensitive, low-traffic workloads.

```bash
# Default: standard exponential backoff
export AWS_ADAPTIVE_RETRY=false

# Enable adaptive retry (recommended under sustained high load)
export AWS_ADAPTIVE_RETRY=true
```

#### `AWS_MAX_POOL_CONNECTIONS` { #aws-max-pool-connections }

:octicons-package-24: **Purpose**
:   Maximum number of concurrent HTTP connections per AWS service client

:octicons-database-24: **Type**
:   Integer (must be > 0)

:octicons-gear-24: **Default**
:   `50`

:octicons-workflow-24: **Behavior**
:   Each AWS service client (one per service per region) maintains its own connection pool up to this limit. Under high concurrency, increasing this value prevents requests from queuing for an available connection. Setting it too high may exhaust system file descriptors.

```bash
# Default
export AWS_MAX_POOL_CONNECTIONS=50

# High-concurrency deployment
export AWS_MAX_POOL_CONNECTIONS=100
```

#### `AWS_CONNECT_TIMEOUT` { #aws-connect-timeout }

:octicons-package-24: **Purpose**
:   Timeout in seconds for establishing a connection to an AWS service endpoint

:octicons-database-24: **Type**
:   Integer (must be > 0)

:octicons-gear-24: **Default**
:   `5`

:octicons-workflow-24: **Behavior**
:   Limits how long the client waits when opening a new connection. It also bounds, per candidate region, the time a real-time audio session (generative-voice speech, speech-to-speech) may take to become ready: connection, initial handshake and the first response together. A short value allows fast failover to another region when an endpoint is unreachable. Increase it only if you see spurious connection timeouts on high-latency networks, or real-time audio requests failing with a `503` a few seconds after they start.

```bash
# Default: 5 seconds
export AWS_CONNECT_TIMEOUT=5

# High-latency network
export AWS_CONNECT_TIMEOUT=10
```

## :material-brain: Bedrock Configuration { #bedrock-configuration }

Which regions serve your Bedrock models, how a request is routed and retried across them, and the Bedrock Mantle endpoint.

#### `AWS_BEDROCK_REGIONS` { #aws-bedrock-regions }

:octicons-package-24: **Purpose**
:   List of AWS regions where Bedrock models are available

:octicons-list-ordered-24: **Format**
:   Comma-separated string

:octicons-gear-24: **Default**
:   Current AWS SDK region if not specified

:octicons-workflow-24: **Behavior**
:   Models are discovered in the same order as the listed regions. The first region is the primary region where your server should be hosted on AWS for optimal performance. Your S3 bucket (`AWS_S3_BUCKET`) must also be in this region. If a model is unavailable in the primary region, subsequent regions are checked in order

```bash
export AWS_BEDROCK_REGIONS=us-east-1,us-west-2,eu-west-1
```

!!! info "Region Selection Guide"
    | Region | Description |
    |--------|-------------|
    | `us-east-1` | :material-star: Widest model selection, usually gets latest releases first |
    | `us-west-2` | :material-rocket-launch: Good selection, often early access to new models |
    | `eu-west-1` | :material-shield-check: European compliance, subset of US models available |

!!! tip "Advanced Configuration"
    See [Compliance and Latency Optimization](#compliance-and-latency-optimization) for detailed configuration examples including GDPR compliance, regional optimization strategies, and best practices for multi-region deployments.

!!! warning "Startup Warning"
    If any models in the configured regions fail availability checks (not enabled, unauthorized, or missing entitlement/agreement in your AWS account), a warning listing the affected models and per-region issues is logged at startup. Enable the required models in the [Amazon Bedrock console](https://console.aws.amazon.com/bedrock/home#/modelaccess) for each configured region.

!!! info "Unreachable Region Tolerance"
    A configured region that cannot be reached (invalid region for the account, network issue, throttling) does not block startup: it is skipped with an `unreachable_bedrock_regions` warning and its models are served from the remaining regions. The skipped region is retried automatically on the next model list refresh (see [`MODEL_CACHE_SECONDS`](operations_configuration_models.md#model-cache-seconds)), so a recovered region rejoins without a restart. Startup only fails when **every** configured region fails, or when **every** per-model availability check errors (e.g. the `bedrock:GetFoundationModelAvailability` permission is denied) — which indicates broken credentials or configuration rather than a regional outage.

!!! info "Denied Region Reporting"
    A region AWS *refused* is never reported as unreachable. Denials are listed under their own `bedrock_regions_missing_iam_permission` warning, which names the IAM action, the region and — where AWS supplies it — the resource ARN. Nothing retries its way out of a denial: grant the named action from [IAM Permissions](operations_iam_permissions.md) and restart. The distinction matters when judging a model that vanished from the catalogue, because an unreachable region is an outage while a denied one is a policy gap.

    Two wordings are used, because AWS answers the same `AccessDeniedException` for two different situations and only one of them is yours to fix:

    - **`the server role is missing the IAM permission <action>`** — AWS named the action and the principal itself, which it only does when IAM policy evaluation produced the denial. This is a real policy gap.
    - **`AWS denied <action> … unless the service does not offer the operation there`** — AWS refused without naming an action, so the action shown is the one derived from the call. A service that is not offered in a region answers this way, and so does an account-level restriction; neither is fixed by editing a policy.

    Discovery degrades rather than failing where it can. `bedrock:ListProvisionedModelThroughputs` is refused outright in the regions that do not offer provisioned throughput, so a denial there costs only the models that are exclusively provisioned in that region — the region keeps serving its on-demand catalogue, and the warning reads `provisioned model discovery skipped: …`.

#### `AWS_BEDROCK_CROSS_REGION_INFERENCE` { #cross-region-inference }

:octicons-package-24: **Purpose**
:   Enable automatic cross-region routing when a model isn't available in the primary region

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `true`

```bash
export AWS_BEDROCK_CROSS_REGION_INFERENCE=true
```

#### `AWS_BEDROCK_CROSS_REGION_INFERENCE_GLOBAL` { #cross-region-global }

:octicons-package-24: **Purpose**
:   Allow global cross-region inference routing to any region worldwide

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `true`

!!! example "GDPR Compliance"
    Set to `false` to comply with data residency regulations (e.g., EU GDPR) by restricting to regional inference only
    ```bash
    export AWS_BEDROCK_CROSS_REGION_INFERENCE_GLOBAL=false
    ```

#### `AWS_BEDROCK_REGION_ROUTING` { #bedrock-region-routing }

:octicons-package-24: **Purpose**
:   Automatic region routing strategy for distributing Bedrock requests across configured regions

:octicons-database-24: **Type**
:   String

:octicons-gear-24: **Default**
:   `ordered`

:octicons-workflow-24: **Behavior**
:   When multiple regions are configured in `AWS_BEDROCK_REGIONS`, this setting controls how requests are distributed across them. The router automatically handles quota/throttling errors and regional unavailability by temporarily avoiding affected regions

:octicons-alert-24: **Requirement**
:   Requires at least 2 regions in `AWS_BEDROCK_REGIONS` to take effect

**Available strategies:**

| Strategy | Description |
|----------|-------------|
| `disabled` | No routing; uses the single region where the model was discovered |
| `ordered` | Try regions in configured order, demoting temporarily blocked ones to last resort (default). Best for prompt caching compatibility |
| `lowest_latency` | Prefer the region with lowest measured latency. Latencies are measured at startup |
| `round_robin` | Distribute requests evenly across regions. Incompatible with prompt caching |

```bash
# Use ordered routing (default)
export AWS_BEDROCK_REGION_ROUTING=ordered

# Use lowest latency routing
export AWS_BEDROCK_REGION_ROUTING=lowest_latency

# Disable routing
export AWS_BEDROCK_REGION_ROUTING=disabled
```

!!! tip "Strategy Selection"
    - **`ordered`** (default): Best general-purpose choice. Compatible with prompt caching since requests consistently go to the same region. Provides failover when a region hits quota limits
    - **`lowest_latency`**: Best when response time is critical. Measures region latencies at startup and prefers the fastest region. Falls back to others when the preferred region is blocked
    - **`round_robin`**: Best for maximizing aggregate throughput across regions. Not recommended with prompt caching as it distributes requests across all regions equally

!!! info "More Details"
    For comprehensive documentation on region routing including failover behavior, S3 bucket pinning, logging, and best practices, see the [Region Routing Guide](operations_resilience.md).

#### `AWS_BEDROCK_REGION_ROUTING_QUOTA_BACKOFF_SECONDS` { #bedrock-region-routing-quota-backoff }

:octicons-package-24: **Purpose**
:   Duration to temporarily avoid a region after receiving a quota or throttling error

:octicons-database-24: **Type**
:   Integer (seconds, must be > 0)

:octicons-gear-24: **Default**
:   `60`

:octicons-workflow-24: **Behavior**
:   This is the **base** backoff value. When a Bedrock API call fails due to quota limits (`ThrottlingException`, `TooManyRequestsException`, `ServiceQuotaExceededException`), the affected region is temporarily blocked. The actual delay doubles with each consecutive quota error on the same region (exponential backoff), up to a hard ceiling of 1 hour. The counter resets after a successful request. Subsequent requests are routed to other available regions during the backoff period.

```bash
# Default: 60 seconds (base value — actual delay doubles per consecutive error)
export AWS_BEDROCK_REGION_ROUTING_QUOTA_BACKOFF_SECONDS=60

# Shorter base backoff for aggressive retry
export AWS_BEDROCK_REGION_ROUTING_QUOTA_BACKOFF_SECONDS=30

# Longer base backoff for conservative approach
export AWS_BEDROCK_REGION_ROUTING_QUOTA_BACKOFF_SECONDS=120
```

!!! tip "Tuning"
    The base value controls how long the first quota error blocks a region. Subsequent consecutive errors on the same region double the delay (60 s → 120 s → 240 s → …, capped at 1 hour). Lower base values retry the region sooner but risk repeated throttling. Higher values provide more conservative avoidance at the cost of reduced region utilization.

    See [Region Routing — Overview](operations_resilience.md#overview) for full backoff behavior details.

#### `AWS_BEDROCK_REGION_ROUTING_UNAVAILABLE_BACKOFF_SECONDS` { #bedrock-region-routing-unavailable-backoff }

:octicons-package-24: **Purpose**
:   Duration to temporarily avoid a region after receiving an unavailability error

:octicons-database-24: **Type**
:   Integer (seconds, must be > 0)

:octicons-gear-24: **Default**
:   `30`

:octicons-workflow-24: **Behavior**
:   When a Bedrock API call fails due to service unavailability (`ServiceUnavailableException`, `ModelNotReadyException`), the affected region is temporarily blocked for this many seconds. These errors are typically shorter-lived than quota limits, so the default is shorter

```bash
# Default: 30 seconds
export AWS_BEDROCK_REGION_ROUTING_UNAVAILABLE_BACKOFF_SECONDS=30

# Longer backoff for stability
export AWS_BEDROCK_REGION_ROUTING_UNAVAILABLE_BACKOFF_SECONDS=60
```

!!! info "More Details"
    See [Region Routing — Overview](operations_resilience.md#overview) for full backoff behavior details.

#### `AWS_BEDROCK_REGION_ROUTING_MAX_QUOTA_BACKOFF_SECONDS` { #bedrock-region-routing-max-quota-backoff }

:octicons-package-24: **Purpose**
:   Hard ceiling in seconds on the exponential quota backoff for a single region

:octicons-database-24: **Type**
:   Integer (seconds, must be > 0)

:octicons-gear-24: **Default**
:   `3600` (1 hour)

:octicons-workflow-24: **Behavior**
:   Quota backoff grows exponentially with consecutive errors (base interval × 2^n). This setting caps how large that value can become, preventing a region from being blocked indefinitely. Reduce it to allow faster recovery; increase it to keep a misbehaving region sidelined for longer.

```bash
# Default: 1 hour ceiling
export AWS_BEDROCK_REGION_ROUTING_MAX_QUOTA_BACKOFF_SECONDS=3600

# More aggressive recovery
export AWS_BEDROCK_REGION_ROUTING_MAX_QUOTA_BACKOFF_SECONDS=600
```

#### `AWS_BEDROCK_REGION_ROUTING_QUOTA_STALE_FACTOR` { #bedrock-region-routing-quota-stale-factor }

:octicons-package-24: **Purpose**
:   Multiplier applied to the max quota backoff to compute the stale-error reset threshold

:octicons-database-24: **Type**
:   Integer (must be > 0)

:octicons-gear-24: **Default**
:   `2` (threshold = 2 × max quota backoff = 2 hours with defaults)

:octicons-workflow-24: **Behavior**
:   If the most recent quota error on a region occurred more than `max_quota_backoff × factor` seconds ago, the consecutive-error counter is reset and the next error is treated as a fresh start rather than an escalation. A higher value keeps memory of past errors for longer before resetting the counter.

```bash
# Default: reset counter after 2× the max backoff window
export AWS_BEDROCK_REGION_ROUTING_QUOTA_STALE_FACTOR=2

# Longer memory of past errors
export AWS_BEDROCK_REGION_ROUTING_QUOTA_STALE_FACTOR=4
```

#### `AWS_BEDROCK_MAX_RETRIES` { #bedrock-max-retries }

:octicons-package-24: **Purpose**
:   Maximum number of retries per Bedrock invocation, each retry escalating to the next available region

:octicons-database-24: **Type**
:   Integer (must be 0 or greater; `0` disables retries)

:octicons-gear-24: **Default**
:   `9`

:octicons-workflow-24: **Behavior**
:   Controls the retry budget for each Bedrock API call. When region routing is enabled, every retry escalates to the next region in priority order and each candidate region is tried at most once, so the attempts are bounded by the smaller of `AWS_BEDROCK_MAX_RETRIES` + 1 and the number of candidate regions for the model — with 3 regions and the default 9 retries, a request makes at most 3 attempts. A region that just failed is still blocked by its own backoff, and retrying it would only extend that backoff instead of recovering the request. When routing is disabled, or the region is pinned by S3 inputs, the full budget is spent as SDK retries against that single region.

```bash
# Default: 9 retries (10 total attempts)
export AWS_BEDROCK_MAX_RETRIES=9

# Fail faster (e.g. low-latency interactive use cases)
export AWS_BEDROCK_MAX_RETRIES=3

# Deeper in-region retrying for single-region or S3-pinned requests
export AWS_BEDROCK_MAX_RETRIES=18
```

!!! tip "Related setting"
    See [`AWS_BEDROCK_REGION_ROUTING`](#bedrock-region-routing) and [Region Routing](operations_resilience.md) for the full retry and failover behavior.

#### `AWS_FAILOVER_MAX_RETRIES` { #failover-max-retries }

:octicons-package-24: **Purpose**
:   Maximum SDK retry attempts per candidate region for the multi-region failover services (Polly, Transcribe, Translate, Comprehend)

:octicons-database-24: **Type**
:   Integer (must be 0 or greater)

:octicons-gear-24: **Default**
:   `2`

:octicons-workflow-24: **Behavior**
:   Only applied when a service has several candidate regions (no explicit region setting): each region attempt uses this reduced retry budget (`2` retries = 3 attempts per region) before failing over, so failover across regions replaces deep in-region retrying. When a service is pinned to a single region, the standard retry budget from [`AWS_BEDROCK_MAX_RETRIES`](#bedrock-max-retries) applies instead.

```bash
# Default: 2 retries (3 attempts) per candidate region
export AWS_FAILOVER_MAX_RETRIES=2

# Fail over after a single attempt per region
export AWS_FAILOVER_MAX_RETRIES=0
```

!!! tip "Related setting"
    See [Other AWS Services Failover](operations_resilience.md#other-aws-services-failover) for the full multi-region failover behavior.

#### `AWS_BEDROCK_MANTLE_ENABLED` { #bedrock-mantle-enabled }

:octicons-package-24: **Purpose**
:   Expose models served by the Amazon Bedrock Mantle endpoint (OpenAI/Anthropic-compatible APIs) in addition to the classic Bedrock Converse models

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `true`

:octicons-workflow-24: **Behavior**
:   Mantle-only models (e.g. OpenAI GPT, xAI Grok, Google Gemma 4) become available on the chat completions, responses, messages, and completions routes. Models available on both the classic bedrock-runtime endpoint and Mantle are served by bedrock-runtime unless listed in [`AWS_BEDROCK_MANTLE_PREFERRED_MODELS`](operations_configuration_models.md#bedrock-mantle-preferred-models), which defaults to the OpenAI GPT-5.6 family.

    Authentication requires no static secrets: short-term bearer tokens are derived automatically (SigV4-presigned) from the same AWS credential chain the server already uses, and refreshed transparently.

    When Bedrock Mantle is unreachable or the IAM role lacks `bedrock-mantle` permissions, Mantle models are simply not listed and a warning is logged at startup — no configuration change required.

```bash
export AWS_BEDROCK_MANTLE_ENABLED=false
```

!!! warning "Guardrails Not Supported"
    Amazon Bedrock Guardrails are not supported on Mantle-served requests. A guardrail configured alongside a non-empty [`AWS_BEDROCK_MANTLE_PREFERRED_MODELS`](operations_configuration_models.md#bedrock-mantle-preferred-models) — which is the default — **stops the server at startup**, because a model routed to Mantle would otherwise be served unfiltered; clear that setting to keep both. For the Mantle-only models, which have no classic endpoint to fall back to, a startup warning reports how many are affected; set `AWS_BEDROCK_MANTLE_ENABLED=false` to remove them from the catalogue.

!!! note "Cross-Region Inference Profiles Not Available"
    Bedrock cross-region inference profiles do not exist on the Mantle endpoint. Mantle relies on [multi-region failover](#bedrock-mantle-regions) and its own separate throughput quotas instead.

!!! warning "Required IAM Permissions"
    Enabling this setting requires the `bedrock-mantle` IAM permissions — see [Bedrock Mantle IAM Permissions](operations_iam_permissions.md#bedrock-mantle-iam).

[:octicons-arrow-right-24: Bedrock Mantle Models feature overview](features.md#bedrock-mantle-models)

#### `AWS_BEDROCK_MANTLE_REGIONS` { #bedrock-mantle-regions }

:octicons-package-24: **Purpose**
:   List of AWS regions used for Amazon Bedrock Mantle, in failover priority order

:octicons-database-24: **Type**
:   Comma-separated string of AWS region identifiers

:octicons-gear-24: **Default**
:   The regions of [`AWS_BEDROCK_REGIONS`](#aws-bedrock-regions) that offer Bedrock Mantle

:octicons-workflow-24: **Behavior**
:   Model availability differs per region; the served model catalog is the union of all listed regions. Region failover, quota backoff, and health tracking work exactly like classic Bedrock [region routing](operations_resilience.md).

```bash
export AWS_BEDROCK_MANTLE_REGIONS=us-east-1,eu-west-1
```

!!! note "Regions Without a Mantle Endpoint"
    Bedrock Mantle is offered in fewer regions than classic Bedrock — see [model availability by endpoint](https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html). Left unset, this setting keeps only the regions of `AWS_BEDROCK_REGIONS` known to offer it, so a deployment spanning other regions is not held up at startup by an address that does not exist.

    An explicit value is used exactly as given, which is how a region AWS adds later is used without waiting for a release. A region that turns out to have no Mantle endpoint is named in a startup warning rather than retried forever. If none of your regions offers it, set `AWS_BEDROCK_MANTLE_ENABLED=false`.

#### `AWS_BEDROCK_MANTLE_ENDPOINT_URL` { #bedrock-mantle-endpoint-url }

:octicons-package-24: **Purpose**
:   Override the Amazon Bedrock Mantle endpoint URL template

:octicons-database-24: **Type**
:   String — URL template with a `{region}` placeholder

:octicons-gear-24: **Default**
:   None (`https://bedrock-mantle.{region}.api.aws`)

:octicons-workflow-24: **Behavior**
:   The `{region}` placeholder is substituted with the target region.

```bash
export AWS_BEDROCK_MANTLE_ENDPOINT_URL='https://bedrock-mantle.{region}.api.aws'
```

#### `AWS_BEDROCK_MANTLE_SERVICE_HEADER` { #bedrock-mantle-service-header }

:octicons-package-24: **Purpose**
:   Honor the `x-stdapi-service: bedrock-mantle` request header to route a model available on both endpoints through Bedrock Mantle for that request instead of the default bedrock-runtime serving

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false`

```bash
export AWS_BEDROCK_MANTLE_SERVICE_HEADER=true
```

!!! warning "Incompatible with Bedrock Guardrails"
    Requires [`AWS_BEDROCK_MANTLE_ENABLED`](#bedrock-mantle-enabled) and cannot be enabled together with Amazon Bedrock Guardrails: guardrails do not apply to Mantle-served requests, so a per-request header would allow clients to bypass them.

#### `AWS_BEDROCK_MANTLE_PROJECT` { #bedrock-mantle-project }

:octicons-package-24: **Purpose**
:   Default Amazon Bedrock Project/Workspace ID attributed to Bedrock Mantle inference requests for cost tracking and observability

:octicons-database-24: **Type**
:   String — a bare project ID (e.g. `proj_abc123` or `default`), not an ARN

:octicons-gear-24: **Default**
:   None (requests fall to the account's `default` project)

:octicons-workflow-24: **Behavior**
:   Bedrock Projects (OpenAI-compatible APIs) and Workspaces (Anthropic Messages API) are the same underlying resource; the value is sent as the `OpenAI-Project` header on the Chat Completions and Responses APIs, and as the `anthropic-workspace` header on the Anthropic Messages API. When unset, requests fall to the account's `default` project — no failure.

```bash
export AWS_BEDROCK_MANTLE_PROJECT=proj_abc123
```

!!! note "Bedrock Mantle only"
    Project/Workspace attribution is honored **only** for models served by the Amazon Bedrock Mantle endpoint. Classic `bedrock-runtime` (non-Mantle) models ignore it and use application inference profiles instead.

#### `AWS_BEDROCK_ALLOW_MANTLE_PROJECT_OVERRIDE` { #bedrock-allow-mantle-project-override }

:octicons-package-24: **Purpose**
:   Allow a request to override the configured Mantle project via the `OpenAI-Project` / `anthropic-workspace` header

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false`

:octicons-workflow-24: **Behavior**
:   When `true`, a request may set its own project through the `OpenAI-Project` (Chat Completions, Responses) or `anthropic-workspace` (Anthropic Messages) header. When `false` **and** [`AWS_BEDROCK_MANTLE_PROJECT`](#bedrock-mantle-project) is configured, the request header is ignored and the server default applies. When **no** default project is configured, the request header is always honored regardless of this flag. A malformed request-supplied project ID returns `400`.

```bash
export AWS_BEDROCK_ALLOW_MANTLE_PROJECT_OVERRIDE=true
```

!!! note "Bedrock Mantle only"
    These headers apply **only** to models served by the Amazon Bedrock Mantle endpoint; classic `bedrock-runtime` models ignore them.

#### `AWS_BEDROCK_MODEL_REGION_RESTRICT` { #bedrock-model-region-restrict }

:octicons-package-24: **Purpose**
:   Restrict a model to specific region(s) only, useful when a model provides important features only in certain regions

:octicons-database-24: **Type**
:   JSON object (keys: Bedrock model IDs or prefixes, values: ordered lists of allowed regions)

:octicons-gear-24: **Default**
:   `{}` (empty — no model-specific region restriction)

:octicons-workflow-24: **Behavior**
:   When set, the model is made available **only** in the listed regions (intersected with the regions where it is actually available), and the list order defines the routing priority when the default `ordered` [routing strategy](#bedrock-region-routing) is used. No fallback to other regions occurs. Keys can be exact model IDs or prefixes that match the beginning of a model ID

```bash
# Restrict Nova Pro to us-east-1 for grounding support
export AWS_BEDROCK_MODEL_REGION_RESTRICT='{"amazon.nova-pro-v1:0": ["us-east-1"]}'
```

!!! tip "Use Case: Region-Specific Features"
    Some model features are only available in specific regions. For example, Nova grounding is only available in `us-east-1`. Restricting the model to that region ensures the feature is always available.

    See [Region Routing — Model Region Restrict](operations_resilience.md#model-region-restrict) for more details.

!!! warning "Startup Warning"
    If a key has no matching available model, a warning is logged at startup. This can happen for two reasons:

    - **Typo or unknown model** — the key (exact ID or prefix) does not match any model ID returned by Bedrock.
    - **No matching region** — the model exists but is not available in any of the regions listed in `AWS_BEDROCK_REGIONS` (e.g. the model is not enabled in those regions, or the restricted regions are not configured).

## :material-robot: Other AWS Services { #other-aws-services }

!!! note "Optional Configuration"
    Each service region is optional. Left unset, the service treats every `AWS_BEDROCK_REGIONS` entry as a candidate and fails over between them; setting one pins the service to that single region, with no failover.

#### `AWS_POLLY_REGION` { #aws-polly-region }

:octicons-package-24: **Purpose**
:   Region for Amazon Polly text-to-speech service

:octicons-gear-24: **Default**
:   All regions in `AWS_BEDROCK_REGIONS`, with per-engine regional discovery and automatic failover

:octicons-workflow-24: **Behavior**
:   When unset, voice availability is discovered per engine in every `AWS_BEDROCK_REGIONS` entry at startup: an engine (Standard, Neural, Long-form, Generative) is exposed as a model when at least one candidate region offers it, and each synthesis call routes to the regions offering the requested engine and voice, failing over on region-level errors. Setting an explicit region pins Polly to that single region — engines it does not offer are then disabled.

```bash
export AWS_POLLY_REGION=us-east-1
```

!!! warning "Amazon Polly Engine Availability"
    Not all Polly engines (Standard, Neural, Long-form, Generative) are available in all AWS regions. With the default multi-region behavior, an engine missing from one region is simply served from another candidate region that offers it. See [Amazon Polly feature and region compatibility](https://docs.aws.amazon.com/polly/latest/dg/limits.html#limits-regions) for detailed information.

#### `AWS_COMPREHEND_REGION` { #aws-comprehend-region }

:octicons-package-24: **Purpose**
:   Region for the Amazon Comprehend services (language detection and toxicity moderation)

:octicons-gear-24: **Default**
:   All regions in `AWS_BEDROCK_REGIONS`, tried in order with automatic failover

:octicons-workflow-24: **Behavior**
:   When unset, Comprehend calls try each `AWS_BEDROCK_REGIONS` entry in order and fail over to the next region on region-level errors (throttling, service unavailability, network issues, or a region that does not offer Comprehend or the requested operation). Setting an explicit region pins Comprehend to that single region with no failover.

```bash
export AWS_COMPREHEND_REGION=us-east-1
```

!!! warning "Amazon Comprehend Regional Availability"
    Amazon Comprehend is not available in all AWS regions. stdapi.ai uses the `detect_dominant_language` feature for language detection and `detect_toxic_content` for [Comprehend moderation](operations_iam_permissions.md#comprehend-moderation). Verify service and feature availability in your target region (with the default multi-region behavior, a region without Comprehend simply fails over to the next one). See [Amazon Comprehend supported regions](https://docs.aws.amazon.com/comprehend/latest/dg/guidelines-and-limits.html#limits-regions) for regional availability.

#### `AWS_TRANSCRIBE_REGION` { #aws-transcribe-region }

:octicons-package-24: **Purpose**
:   Region for Amazon Transcribe speech-to-text service

:octicons-gear-24: **Default**
:   All regions in `AWS_BEDROCK_REGIONS` that have a co-located S3 bucket, tried in order with automatic failover

:octicons-workflow-24: **Behavior**
:   Transcription jobs need an S3 bucket in the job's region. When unset, every `AWS_BEDROCK_REGIONS` entry with a usable bucket is a candidate — the primary region is served by [`AWS_TRANSCRIBE_S3_BUCKET`](operations_configuration_storage.md#aws-transcribe-s3-bucket) (or `AWS_S3_BUCKET`), the others by their [`AWS_S3_REGIONAL_BUCKETS`](operations_configuration_storage.md#aws-s3-regional-buckets) entry. On a region-level error while starting a job, the audio is server-side copied to the next candidate's bucket and the job restarts there. Setting an explicit region pins Transcribe to that single region with no failover.

```bash
export AWS_TRANSCRIBE_REGION=us-east-1
```

#### `AWS_TRANSLATE_REGION` { #aws-translate-region }

:octicons-package-24: **Purpose**
:   Region for Amazon Translate text translation service

:octicons-gear-24: **Default**
:   All regions in `AWS_BEDROCK_REGIONS`, tried in order with automatic failover

:octicons-workflow-24: **Behavior**
:   When unset, translation calls try each `AWS_BEDROCK_REGIONS` entry in order and fail over to the next region on region-level errors (throttling, service unavailability, network issues, or a region that does not offer Translate). Setting an explicit region pins Translate to that single region with no failover.

```bash
export AWS_TRANSLATE_REGION=us-east-1
```

---

## :material-scale-balance: Compliance and Latency Optimization { #compliance-and-latency-optimization }

Strategic region configuration is critical for both regulatory compliance and performance optimization. This section provides best practice configurations for common scenarios.

!!! info "AWS AI Services Data Privacy"
    **Amazon Bedrock**: Does not store or use user prompts and responses, and does not share them with third parties by default. Your content remains private and is not used to train models.

    **Other AI Services**: AWS collects telemetry data from other AI services (Polly, Comprehend, Transcribe, Translate) by default. For enhanced data privacy and compliance, you can opt out of AWS using your content to improve AI services. Configure [AI services opt-out policies](https://docs.aws.amazon.com/organizations/latest/userguide/orgs_manage_policies_ai-opt-out.html) at the AWS Organizations level to prevent your data from being used for service improvement.

#### GDPR and Data Residency Compliance

For applications serving European users, data residency regulations like GDPR may require that data processing occurs within specific geographic boundaries.

```bash title="EU-Only Configuration (Strict GDPR)"
# Use only European regions
export AWS_S3_BUCKET=my-stdapi-eu-bucket
export AWS_BEDROCK_REGIONS=eu-west-1,eu-west-3,eu-central-1

# Disable global cross-region inference to prevent data routing outside Europe
export AWS_BEDROCK_CROSS_REGION_INFERENCE_GLOBAL=false

# Keep cross-region inference enabled for failover within EU regions
export AWS_BEDROCK_CROSS_REGION_INFERENCE=true
```

!!! success "Key Compliance Settings"
    - **`AWS_BEDROCK_CROSS_REGION_INFERENCE_GLOBAL=false`**: Prevents requests from being routed to regions outside your specified list
    - **`AWS_BEDROCK_CROSS_REGION_INFERENCE=true`**: Enables cross-region inference within your specified EU regions
    - **All services in EU regions**: Ensures all data processing stays within European boundaries

!!! warning "Important Considerations"
    - Not all Bedrock models are available in all EU regions - verify model availability
    - Some newer models may be available in US regions first; this configuration prioritizes compliance over immediate access to latest models
    - S3 buckets must be created in EU regions and configured appropriately for data residency

#### Latency Optimization

For applications prioritizing low latency and high performance, configure regions closest to your users and application infrastructure.

**:flag_us: North America:**

```bash
# Primary region for lowest latency, with fallbacks
export AWS_S3_BUCKET=my-stdapi-us-east-1-bucket
export AWS_BEDROCK_REGIONS=us-east-1,us-west-2,us-east-2

# Enable all cross-region inference for maximum model availability
export AWS_BEDROCK_CROSS_REGION_INFERENCE=true
export AWS_BEDROCK_CROSS_REGION_INFERENCE_GLOBAL=true
```

**:flag_jp: Asia-Pacific:**

```bash
# Use Asia-Pacific regions for lowest latency to APAC users
export AWS_S3_BUCKET=my-stdapi-ap-southeast-1-bucket
export AWS_BEDROCK_REGIONS=ap-southeast-1,ap-northeast-1,us-west-2

# Enable global inference for fallback to US regions if needed
export AWS_BEDROCK_CROSS_REGION_INFERENCE=true
export AWS_BEDROCK_CROSS_REGION_INFERENCE_GLOBAL=true
```

**:earth_africa: Global Multi-Region:**

```bash
# Balanced configuration with worldwide coverage
export AWS_S3_BUCKET=my-stdapi-us-east-1-bucket
export AWS_BEDROCK_REGIONS=us-east-1,eu-west-1,ap-southeast-1,us-west-2

# Enable global inference for best availability
export AWS_BEDROCK_CROSS_REGION_INFERENCE=true
export AWS_BEDROCK_CROSS_REGION_INFERENCE_GLOBAL=true
```

!!! tip "Latency Optimization Tips"
    - :material-server: **Server and S3 co-location**: Deploy stdapi.ai and your `AWS_S3_BUCKET` in the first region specified in `AWS_BEDROCK_REGIONS` (your primary region)
    - :material-network: **Network proximity**: Choose the first region based on low latency to your application servers and end users
    - :material-cash: **Data transfer costs**: Cross-region data transfer incurs costs; co-locating server and S3 in the same region minimizes these
    - :material-check-circle: **Model availability**: While `us-east-1` often has the most models, check specific model availability in your target regions

#### Hybrid Approach: Compliance with Performance

Balance compliance requirements with performance needs:

```bash title="EU Primary with US Fallback"
# EU primary with US fallback (for model availability)
export AWS_S3_BUCKET=my-stdapi-eu-bucket
export AWS_BEDROCK_REGIONS=eu-west-1,eu-central-1,us-east-1

# Allow cross-region but restrict to specific regions only
export AWS_BEDROCK_CROSS_REGION_INFERENCE=true
export AWS_BEDROCK_CROSS_REGION_INFERENCE_GLOBAL=false
```

!!! warning "Legal Compliance Notice"
    Including `us-east-1` as a fallback region provides access to more models but may not comply with strict data residency requirements. **Consult your legal and compliance teams before using this configuration.**

---
