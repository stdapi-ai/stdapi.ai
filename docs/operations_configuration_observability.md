---
title: "Configuration - Observability and Usage"
description: "Configure what stdapi.ai reports about itself: log level and request logging, OpenTelemetry traces, CloudWatch metrics, cost tracking and the organization Usage API."
keywords: "log level, request logging, OpenTelemetry, OTLP, AWS X-Ray, CloudWatch metrics, cost tracking, usage API, input validation"
---

# :material-radar: Observability and Usage

What the gateway reports about itself: logs, traces, metrics, per-request cost and the usage endpoints. Part of the [Configuration Guide](operations_configuration.md).

## :material-format-list-bulleted: Settings Summary

### :material-chart-line: Logging { #summary-logging }

| Variable                                        | Default | Description                                                                           |
|-------------------------------------------------|---------|---------------------------------------------------------------------------------------|
| [`LOG_LEVEL`](#logging-level)                   | `info`  | Minimum log severity: `info`, `warning`, `error`, `critical`, or `disabled`           |
| [`LOG_REQUEST_PARAMS`](#log-request-params)     | `false` | Include request/response parameters in logs (not recommended for production)          |
| [`LOG_CLIENT_IP`](#client-ip-logging)           | `false` | Log client IP addresses (requires `ENABLE_PROXY_HEADERS` for real IPs behind proxies) |

### :material-chart-box-outline: CloudWatch Metrics { #summary-cloudwatch-metrics }

| Variable                                                                    | Default               | Description                                                            |
|-----------------------------------------------------------------------------|-----------------------|------------------------------------------------------------------|
| [`CLOUDWATCH_METRICS`](#cloudwatch-metrics)                                 | `false`               | Emit per-request AWS-billed usage as CloudWatch EMF log lines      |
| [`CLOUDWATCH_METRICS_NAMESPACE`](#cloudwatch-metrics-namespace)             | `stdapi`              | CloudWatch namespace for the emitted usage metrics                 |
| [`CLOUDWATCH_METRICS_USER_DIMENSION`](#cloudwatch-metrics-user-dimension)   | `false`               | Also publish the authenticated caller as a `User` metric dimension  |
| [`CLOUDWATCH_METRICS_REGION`](#cloudwatch-metrics-region)                   | Server's region       | Region the [Usage API](#usage-api-section) reads the published metrics from |

### :material-currency-usd: Cost Tracking { #summary-cost-tracking }

| Variable                                                    | Default        | Description                                                              |
|-----------------------------------------------------------------|----------------|-----------------------------------------------------------------------|
| [`COST_TRACKING`](#cost-tracking)                           | `false`        | Estimate each request's cost from the published AWS price list          |
| [`COST_PRICE_OVERRIDES`](#cost-price-overrides)             | `{}`           | JSON map of operator-supplied unit prices for models missing from the AWS catalog |

### :material-chart-timeline-variant: Usage API { #summary-usage-api }

| Variable                                                    | Default | Description                                                                       |
|-------------------------------------------------------------|---------|-----------------------------------------------------------------------------------|
| [`USAGE_API`](#usage-api)                                   | `false` | Serve the organization usage and costs endpoints (requires `CLOUDWATCH_METRICS`)  |
| [`USAGE_API_ADMIN_SCOPES`](#usage-api-admin-scopes)         | None    | OAuth 2.0 scopes an Amazon Cognito token must all carry to read these endpoints   |
| [`USAGE_API_MAX_METRICS`](#usage-api-max-metrics)           | `500`   | Refuse a query that would read more metric series than this                       |
| [`USAGE_API_MAX_RANGE_DAYS`](#usage-api-max-range-days)     | `92`    | Longest span between `start_time` and `end_time` on a query                       |
| [`USAGE_API_CACHE_TTL`](#usage-api-cache-ttl)               | `60`    | Seconds an answered query is reused for (`0` disables the cache)                  |

### :material-radar: Observability (OpenTelemetry) { #summary-observability }

| Variable                                            | Default                           | Description                                                                            |
|-----------------------------------------------------|-----------------------------------|----------------------------------------------------------------------------------------|
| [`OTEL_ENABLED`](#otel-enabled)                     | `false`                           | Enable distributed tracing via OpenTelemetry (integrates with AWS X-Ray, Jaeger, etc.) |
| [`OTEL_SERVICE_NAME`](#otel-service-name)           | `stdapi.ai`                       | Service name identifier in trace visualizations                                        |
| [`OTEL_EXPORTER_ENDPOINT`](#otel-exporter-endpoint) | `http://127.0.0.1:4318/v1/traces` | OTLP HTTP endpoint URL for trace export                                                |
| [`OTEL_SAMPLE_RATE`](#otel-sample-rate)             | `1.0`                             | Trace sampling rate from 0.0 (none) to 1.0 (all requests)                              |

---

## :material-radar: Observability (OpenTelemetry)

Configure distributed tracing for debugging and performance monitoring. stdapi.ai integrates with AWS X-Ray, Jaeger, DataDog, and other OTLP-compatible systems.

#### `OTEL_ENABLED` { #otel-enabled }

:octicons-package-24: **Purpose**
:   Enable or disable OpenTelemetry tracing

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false`

```bash
export OTEL_ENABLED=true
```

!!! note "Performance Consideration"
    Disable in performance-critical deployments where observability is not needed.

#### `OTEL_SERVICE_NAME` { #otel-service-name }

:octicons-package-24: **Purpose**
:   Service identifier in trace visualizations

:octicons-gear-24: **Default**
:   `stdapi.ai`

:octicons-check-circle-24: **Best Practice**
:   Use descriptive names with environment information

```bash
export OTEL_SERVICE_NAME=stdapi-production-us-east-1
```

#### `OTEL_EXPORTER_ENDPOINT` { #otel-exporter-endpoint }

:octicons-package-24: **Purpose**
:   OTLP HTTP endpoint URL for sending traces

:octicons-gear-24: **Default**
:   `http://127.0.0.1:4318/v1/traces`

:octicons-plug-24: **Protocol**
:   Must support OTLP HTTP format

**AWS X-Ray (via ADOT):**

```bash
export OTEL_EXPORTER_ENDPOINT=http://127.0.0.1:4318/v1/traces
```

**Jaeger:**

```bash
export OTEL_EXPORTER_ENDPOINT=http://jaeger:14268/api/traces
```

**Cloud Provider OTLP:**

```bash
# Use provider-specific OTLP endpoints
export OTEL_EXPORTER_ENDPOINT=https://your-provider-otlp-endpoint.com/v1/traces
```

#### `OTEL_SAMPLE_RATE` { #otel-sample-rate }

:octicons-package-24: **Purpose**
:   Percentage of requests to trace (controls cost vs. observability)

:octicons-database-24: **Type**
:   Float (0.0 to 1.0)

:octicons-gear-24: **Default**
:   `1.0` (100%)

**Development:**

```bash
# Trace everything for debugging
export OTEL_SAMPLE_RATE=1.0
```

**Production (Moderate Traffic):**

```bash
# Sample 10% of requests
export OTEL_SAMPLE_RATE=0.1
```

**Production (High Traffic):**

```bash
# Sample 1% of requests
export OTEL_SAMPLE_RATE=0.01
```

!!! tip "Sampling Recommendations"
    | Sample Rate | Use Case |
    |-------------|----------|
    | `1.0` (100%) | :material-bug: Development, debugging, low-traffic services |
    | `0.1` (10%) | :material-chart-line: Production with moderate traffic |
    | `0.01` (1%) | :material-rocket: High-traffic production services |
    | `0.0` (0%) | :material-close: Equivalent to disabling tracing |

---

## :material-chart-line: Validation and Logging

The settings below decide what the gateway writes; the [Logging and Monitoring](operations_logging_monitoring.md) guide describes what comes out of them — the event types and the fields common to all of them, the usage metrics carried on each entry, the CloudWatch EMF metric lines, and CloudWatch Logs Insights queries that follow a single request across its request, stream and background events.

#### `STRICT_INPUT_VALIDATION` { #strict-input-validation }

:octicons-package-24: **Purpose**
:   Reject API requests containing unknown/extra fields instead of ignoring them

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false`

```bash
# Returns HTTP 400 for requests with unexpected fields
export STRICT_INPUT_VALIDATION=true
```

#### `CHAT_COMPLETIONS_REASONING_FIELD` { #chat-completions-reasoning-field }

:octicons-package-24: **Purpose**
:   Choose which field carries a reasoning model's thinking text on `/v1/chat/completions`

:octicons-database-24: **Type**
:   String

:octicons-gear-24: **Default**
:   `reasoning_content`

:octicons-list-ordered-24: **Options**
:   `reasoning_content`, `reasoning`, `none`

:octicons-workflow-24: **Behavior**
:   The OpenAI Chat Completions API returns no thinking text of its own — it reports only a `reasoning_tokens` count — so the providers that do return it have settled on two different names. `reasoning_content` is the DeepSeek spelling, which most clients that read reasoning at all look for first. `reasoning` is the name used by OpenRouter and vLLM. `none` emits neither, keeping responses strictly OpenAI-shaped.
:   The setting applies to both the completed message and the streamed deltas, so a client never sees one name while streaming and another at the end. Callers can also suppress reasoning per request with `include_reasoning: false` or `reasoning: {"exclude": true}`, whatever this is set to.

```bash
# Default: the name most clients read
export CHAT_COMPLETIONS_REASONING_FIELD=reasoning_content

# For clients written against OpenRouter or vLLM
export CHAT_COMPLETIONS_REASONING_FIELD=reasoning

# Strict OpenAI shape: never return thinking text
export CHAT_COMPLETIONS_REASONING_FIELD=none
```

#### `LOG_LEVEL` { #logging-level }

:octicons-package-24: **Purpose**
:   Control the minimum severity of log events written to STDOUT

:octicons-gear-24: **Default**
:   `info`

:octicons-list-ordered-24: **Options**
:   `info`, `warning`, `error`, `critical`, `disabled`

:octicons-workflow-24: **Behavior**
:   Only log events at or above the configured level are output. Log levels are ordered by severity: **info < warning < error < critical**

```bash
# Default: Output all log events
export LOG_LEVEL=info

# Production: Suppress info logs, show only warnings and higher
export LOG_LEVEL=warning

# Critical only: Show only critical errors
export LOG_LEVEL=critical

# Disable logging: Suppress all log output (not recommended)
export LOG_LEVEL=disabled
```

!!! info "Log Level Examples"
    | Level | Outputs | Use Case |
    |-------|---------|----------|
    | `info` | info, warning, error, critical | :material-bug: Development, debugging, full visibility |
    | `warning` | warning, error, critical | :material-check: Production (recommended for most deployments) |
    | `error` | error, critical | :material-alert: High-traffic production, reduce log volume |
    | `critical` | critical only | :material-alert-octagon: Minimal logging, only show fatal errors |
    | `disabled` | none | :material-close: Not recommended - disables all logging |

!!! tip "Production Recommendation"
    For production deployments, `warning` is recommended to reduce log volume while maintaining visibility into issues. The `info` level can generate significant log volume in high-traffic environments.

    For detailed information about log events, structure, and monitoring strategies, see the [Logging and Monitoring](operations_logging_monitoring.md) guide.

#### `LOG_REQUEST_PARAMS` { #log-request-params }

:octicons-package-24: **Purpose**
:   Include request and response parameters (JSON body, form, query) in logs for integration debugging

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false`

```bash
# Enable for debugging (NOT recommended for production)
export LOG_REQUEST_PARAMS=true
```

!!! danger "Security and Cost Warning"
    Enabling `LOG_REQUEST_PARAMS` may expose sensitive data in logs. Use only in development/debugging environments.

    Logging full request/response payloads can also significantly increase log ingestion and storage costs, especially for large LLM prompts, tool calls, and generated outputs. If you must enable it, prefer short log retention, targeted sampling, and temporary use only.

#### `LOG_CLIENT_IP` { #client-ip-logging }

:octicons-package-24: **Purpose**
:   Enable logging of client IP addresses for each request and add IP to OpenTelemetry spans

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false` (disabled for privacy)

```bash
# Disabled (default) - no client IP logging
# No environment variable needed

# Enable client IP logging
export LOG_CLIENT_IP=true
```

!!! info "Client IP Behavior"
    When enabled, client IP addresses are:

    - Included in log output for each request
    - Added as the `client.address` attribute to OpenTelemetry spans (when `OTEL_ENABLED=true`)

    The IP address depends on your proxy configuration:

    **With `ENABLE_PROXY_HEADERS=true` (behind reverse proxy):**

    - Logs the real client IP address from the `X-Forwarded-For` header
    - Shows the actual end-user IP, not the proxy IP
    - Requires your reverse proxy (ALB, CloudFront, etc.) to set the header correctly

    **With `ENABLE_PROXY_HEADERS=false` (default):**

    - Logs the direct connection IP address
    - Typically shows your reverse proxy or load balancer IP, not the end-user IP
    - Limited usefulness unless application is directly exposed to clients

!!! tip "When to Enable"
    Enable `LOG_CLIENT_IP` when:

    - :material-shield-check: You need client IP addresses for security auditing or compliance
    - :material-chart-line: Analyzing traffic patterns and geographic distribution
    - :material-alert: Investigating abuse, fraud, or suspicious activity
    - :material-bug: Debugging client-specific issues

    **Important**: Also enable `ENABLE_PROXY_HEADERS=true` when behind AWS ALB, CloudFront, or other reverse proxies to log the real client IP instead of the proxy IP.

!!! warning "Privacy Consideration"
    Client IP addresses are considered personal data under privacy regulations like GDPR. When logging IP addresses:

    - :material-clock: Consider shorter log retention periods
    - :material-file-document: Document the purpose in your privacy policy
    - :material-shield-lock: Ensure logs are stored securely
    - :material-delete: Implement log deletion procedures aligned with your data retention policy

!!! example "Configuration for AWS Deployments"

    **Behind AWS ALB or CloudFront:**

    ```bash
    # Enable proxy headers to get real client IPs
    export ENABLE_PROXY_HEADERS=true
    # Enable client IP logging
    export LOG_CLIENT_IP=true
    ```

    **Direct exposure (not recommended for production):**

    ```bash
    # Only enable client IP logging
    export LOG_CLIENT_IP=true
    # ENABLE_PROXY_HEADERS remains false (default)
    ```

#### `TIMEZONE` { #timezone }

:octicons-package-24: **Purpose**
:   IANA timezone identifier used for request date and time

:octicons-database-24: **Type**
:   String (IANA timezone identifier)

:octicons-gear-24: **Default**
:   `UTC`

```bash
# UTC (default)
export TIMEZONE=UTC

# North America
export TIMEZONE=America/New_York

# Europe
export TIMEZONE=Europe/London
```

---

## :material-chart-box-outline: CloudWatch Metrics and Cost Tracking

The behavior of these settings — EMF line structure, cost log format, pricing accuracy, regional price fallback, known limitations, and the price override format with examples — is documented in [CloudWatch Metrics (EMF)](operations_logging_monitoring.md#cloudwatch-metrics-emf) and [Cost Tracking](operations_cost_management.md#cost-tracking-real-time-aws-pricing) in the Logging and Monitoring guide.

#### `CLOUDWATCH_METRICS` { #cloudwatch-metrics }

:octicons-package-24: **Purpose**
:   Emit per-request AWS-billed usage as CloudWatch Embedded Metric Format (EMF) log lines

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false`

```bash
export CLOUDWATCH_METRICS=true
```

#### `CLOUDWATCH_METRICS_NAMESPACE` { #cloudwatch-metrics-namespace }

:octicons-package-24: **Purpose**
:   CloudWatch namespace under which the emitted usage metrics are grouped

:octicons-database-24: **Type**
:   String

:octicons-gear-24: **Default**
:   `stdapi`

:octicons-alert-24: **Requirement**
:   1-255 characters, alphanumeric plus `. - _ / # :`, must not start with the reserved `AWS/` prefix

```bash
export CLOUDWATCH_METRICS_NAMESPACE=my-app-metrics
```

#### `CLOUDWATCH_METRICS_USER_DIMENSION` { #cloudwatch-metrics-user-dimension }

:octicons-package-24: **Purpose**
:   Also publish the authenticated caller as a `User` metric dimension. This is what makes `group_by=user_id` answerable on the [Usage API](api_openai_organization_usage.md); without it, those queries have no per-user series to read.

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false`

:octicons-alert-24: **Requirement**
:   [`CLOUDWATCH_METRICS`](#cloudwatch-metrics) and [`USAGE_API`](#usage-api) must both be enabled

!!! warning "The cardinality of this dimension is your caller population"
    Off by default because it adds one stored metric series per user × model × metric name, and Amazon CloudWatch bills every stored custom metric monthly. Enable it only where the number of distinct callers is bounded and known — see [Usage API Query Cost](operations_cost_management.md#usage-api-cost).

```bash
export CLOUDWATCH_METRICS_USER_DIMENSION=true
```

#### `CLOUDWATCH_METRICS_REGION` { #cloudwatch-metrics-region }

:octicons-package-24: **Purpose**
:   Region the [Usage API](api_openai_organization_usage.md) reads the published metrics from. Metrics are published in the region the server's logs are ingested in, which is the server's own region by default — set this only when the logs are shipped elsewhere, because a mismatch makes every usage query answer with empty buckets.

:octicons-database-24: **Type**
:   String (AWS region name)

:octicons-gear-24: **Default**
:   The region the server runs in — its configured AWS region, or the first [`AWS_BEDROCK_REGIONS`](operations_configuration_aws.md#aws-bedrock-regions) entry when none is set

```bash
export CLOUDWATCH_METRICS_REGION=eu-west-1
```

#### `COST_TRACKING` { #cost-tracking }

:octicons-package-24: **Purpose**
:   Estimate each request's cost from the published AWS price list — an estimate computed from billed quantities, not read back from your invoice ([details and accuracy caveats](operations_cost_management.md#cost-tracking-real-time-aws-pricing)). Disabled by default: it requires the extra `pricing:GetProducts` IAM permission — see [Cost Tracking IAM Permissions](operations_iam_permissions.md#cost-tracking-iam).

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false`

```bash
export COST_TRACKING=true
```

#### `COST_PRICE_OVERRIDES` { #cost-price-overrides }

:octicons-package-24: **Purpose**
:   Operator-supplied unit price overrides for models not covered by the AWS Price List API ([format and example](operations_cost_management.md#override-map-for-missing-models))

:octicons-database-24: **Type**
:   JSON object — keys are model IDs, values are dicts mapping dimension name to price per one unit

:octicons-gear-24: **Default**
:   `{}`

---

## :material-chart-timeline-variant: Usage API { #usage-api-section }

The [Organization Usage and Costs API](api_openai_organization_usage.md) serves the OpenAI Administration usage and costs endpoints from the metrics this deployment already publishes. It is **opt-in and off by default**; while it is off, the routes still exist and answer `503`, so a client can tell a disabled feature from a missing deployment.

The routes it adds, under the configured [`OPENAI_ROUTES_PREFIX`](operations_configuration_server.md#openai-routes-prefix):

```text
/v1/organization/usage/completions
/v1/organization/usage/embeddings
/v1/organization/usage/moderations
/v1/organization/usage/images
/v1/organization/usage/audio_speeches
/v1/organization/usage/audio_transcriptions
/v1/organization/usage/web_search_calls
/v1/organization/usage/file_search_calls
/v1/organization/usage/vector_stores
/v1/organization/usage/code_interpreter_sessions
/v1/organization/costs
```

!!! warning "Prerequisites"
    - [`CLOUDWATCH_METRICS`](#cloudwatch-metrics) must be enabled: the usage endpoints read the metrics it publishes. Enabling `USAGE_API` without it fails startup rather than serving endpoints that could never answer.
    - `/v1/organization/costs` additionally requires [`COST_TRACKING`](#cost-tracking), which is what puts a cost on the published metrics; missing it is only a startup warning, and the endpoint answers with no cost data.
    - The role needs `cloudwatch:GetMetricData` and `cloudwatch:ListMetrics` — see [Usage API IAM Permissions](operations_iam_permissions.md#usage-api-iam).

!!! danger "These queries are billed, and are not in the CloudWatch free tier"
    Amazon CloudWatch bills each query by the number of metric series it reads, and enabling the Usage API also publishes the usage metrics under an extra dimension set that is billed monthly. A single client polling the endpoints once a minute is a recurring three-figure monthly charge on a large catalogue. Read [Usage API Query Cost](operations_cost_management.md#usage-api-cost) before enabling it, and keep the limits below at their defaults unless you have priced the change.

#### `USAGE_API` { #usage-api }

:octicons-package-24: **Purpose**
:   Serve the [organization usage and costs endpoints](api_openai_organization_usage.md). When disabled, the routes exist and answer `503`. Enabling it also publishes the usage metrics under an additional dimension set, which carries its own [monthly cost](operations_cost_management.md#usage-api-cost).

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false`

:octicons-alert-24: **Requirement**
:   [`CLOUDWATCH_METRICS`](#cloudwatch-metrics) must be enabled, or startup fails; `/v1/organization/costs` additionally requires [`COST_TRACKING`](#cost-tracking), which is only a startup warning when missing

```bash
export USAGE_API=true
```

#### `USAGE_API_ADMIN_SCOPES` { #usage-api-admin-scopes }

:octicons-package-24: **Purpose**
:   OAuth 2.0 scopes an [Amazon Cognito](operations_configuration_authentication.md#cognito-authentication) token must **all** carry to read these endpoints. Left empty, no user pool token is accepted and only the deployment's own [`API_KEY`](operations_configuration_authentication.md#api-key) may read them. A [tenant API key](operations_configuration_authentication.md#tenant-api-keys) is never accepted, whatever this is set to — organization-wide usage is not a tenant's to read.

:octicons-database-24: **Type**
:   Comma-separated list of scope names

:octicons-gear-24: **Default**
:   Empty — admin API key only

```bash
export USAGE_API_ADMIN_SCOPES=stdapi/admin,stdapi/usage.read
```

#### `USAGE_API_MAX_METRICS` { #usage-api-max-metrics }

:octicons-package-24: **Purpose**
:   Refuse a query that would read more metric series than this, **before** it is billed. A query over the limit is rejected rather than served, so a client narrows its `group_by` or its time range instead of running up the bill.

:octicons-database-24: **Type**
:   Integer

:octicons-gear-24: **Default**
:   `500`

:octicons-alert-24: **Requirement**
:   1–500 — 500 is also the Amazon CloudWatch per-request maximum

```bash
# Tighter cap on a deployment with a large model catalogue
export USAGE_API_MAX_METRICS=100
```

#### `USAGE_API_MAX_RANGE_DAYS` { #usage-api-max-range-days }

:octicons-package-24: **Purpose**
:   Longest span allowed between `start_time` and `end_time` on a query. A longer range is rejected rather than served, so a client cannot ask for a year of daily buckets in one call.

:octicons-database-24: **Type**
:   Integer (days)

:octicons-gear-24: **Default**
:   `92`

:octicons-alert-24: **Requirement**
:   1–455

```bash
# One month per query
export USAGE_API_MAX_RANGE_DAYS=31
```

#### `USAGE_API_CACHE_TTL` { #usage-api-cache-ttl }

:octicons-package-24: **Purpose**
:   Seconds an answered query is reused for. This is what makes a polling client affordable: within the TTL, repeated identical queries are served from the cached answer and cost nothing. Set `0` to disable the cache — every query then reaches Amazon CloudWatch and is billed.

:octicons-database-24: **Type**
:   Integer (seconds)

:octicons-gear-24: **Default**
:   `60`

:octicons-alert-24: **Requirement**
:   0–3600

```bash
# Longer reuse window for dashboards that poll aggressively
export USAGE_API_CACHE_TTL=300
```

---
