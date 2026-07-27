---
title: Logging & Monitoring - Amazon Bedrock API Observability
description: Production-grade observability for stdapi.ai with CloudWatch, OpenTelemetry. Track API performance, monitor costs, debug issues, and ensure compliance.
keywords: AWS CloudWatch logs, API monitoring, OpenTelemetry, API observability, cost monitoring AWS, performance tracking, compliance logging
---

# :material-chart-line: Logging & Monitoring

stdapi.ai provides production-grade observability out of the box. Track request performance, debug issues, monitor costs, and ensure compliance with structured JSON logging and OpenTelemetry integration.

**Why this matters:**

- **Troubleshoot issues fast** - Structured logs with request IDs, timings, and error details
- **Monitor costs** - Track model usage, request volume, and performance by endpoint
- **Ensure compliance** - Full audit trail with client IPs, user IDs, and request/response logging
- **Performance optimization** - Identify slow requests, high-latency endpoints, bottlenecks
- **AWS integration** - Native CloudWatch Logs, X-Ray, and service-level metrics

stdapi.ai emits structured JSON logs for every request, stream, and background task, and integrates with OpenTelemetry (OTel) for traces and metrics. This guide shows how to enable observability, read the logs, and correlate signals across systems.

<div class="grid cards" markdown>

- :material-clipboard-text-outline: __Structured JSON Logging__
  <br>JSON logs to STDOUT—perfect for AWS CloudWatch Logs. One event per line with all request context.

- :material-identifier: __Request Correlation__
  <br>All events for a request share the same `id` (returned as `x-request-id`). Track full request lifecycle across logs and traces.

- :material-aws: __AWS-Native Integration__
  <br>Works seamlessly with CloudWatch Logs, X-Ray, and service metrics. ECS auto-forwards STDOUT to CloudWatch.

- :material-graphql: __OpenTelemetry Tracing__
  <br>Enable `OTEL_ENABLED=true` to export spans to AWS X-Ray, Jaeger, Tempo, or any OTLP-compatible backend.

- :material-chart-line: __Performance Insights__
  <br>Track execution times, model usage, endpoint latency. Includes ready-to-use CloudWatch Logs Insights queries.

- :material-alert-decagram-outline: __Debug Mode__
  <br>Enable `LOG_REQUEST_PARAMS=true` for full request/response logging when troubleshooting issues.

</div>

---

## :material-rocket-launch: Quick Start

Set these environment variables, then restart the service (see the [Configuration Guide](operations_configuration.md) for details):

```bash
# Set minimum log level (optional, defaults to "info")
# Options: info, warning, error, critical, disabled
export LOG_LEVEL=warning

# Enable OpenTelemetry tracing
export OTEL_ENABLED=true
export OTEL_SERVICE_NAME=stdapi
# 0.0–1.0 (10% example)
export OTEL_SAMPLE_RATE=0.1

# Include request/response payloads in logs (for debugging ONLY)
export LOG_REQUEST_PARAMS=true

# Log client IP addresses (requires ENABLE_PROXY_HEADERS for real client IPs)
export LOG_CLIENT_IP=true
export ENABLE_PROXY_HEADERS=true  # When behind ALB/CloudFront
```

!!! warning "Sensitive data and cost impact"
    Enabling `LOG_REQUEST_PARAMS` may expose sensitive content in logs. Use only in development or during targeted troubleshooting. Redact secrets before sharing logs externally.

    Additionally, logging full request/response payloads can dramatically increase log volume and costs, especially for large LLM prompts, tool calls, and generated outputs. In AWS CloudWatch Logs, ingestion and storage costs scale with log size. Prefer short retention, targeted sampling, and temporary enablement only when needed.

!!! info "Client IP Logging"
    When `LOG_CLIENT_IP=true`:

    - The `client_ip` field is added to request logs
    - The client IP is added as `client.address` attribute to OpenTelemetry spans (when `OTEL_ENABLED=true`)

    To log the real client IP address (instead of the proxy IP), also enable `ENABLE_PROXY_HEADERS=true` when running behind AWS ALB, CloudFront, or other reverse proxies. See the [Configuration Guide](operations_configuration.md#client-ip-logging) for details.

!!! tip "CloudWatch best practice"
    JSON to STDOUT is optimal for CloudWatch Logs Insights. In AWS ECS, the task's log driver forwards container STDOUT to CloudWatch Logs automatically.

---

## :material-format-list-bulleted: Event Types

stdapi.ai emits five kinds of JSON events (one per line):

| Event            | Description                                                                         |
|:-----------------|:------------------------------------------------------------------------------------|
| `start`          | Emitted once at server startup. Includes startup metadata and warnings.             |
| `stop`           | Emitted on graceful shutdown. Includes uptime.                                      |
| `request`        | One per HTTP request. Method, path, status, timings, and optional request/response. |
| `request_stream` | Streaming segments (SSE/audio). Indicates streaming activity and duration.          |
| `background`     | Background tasks correlated to the parent request.                                  |

---

## :material-table-column: Common Fields

Each event shares core fields and may add type‑specific ones.

|                                     Field | Applies to                          | Description                                                                                 |
|------------------------------------------:|:------------------------------------|:--------------------------------------------------------------------------------------------|
|                                    `type` | all                                 | One of `start`, `stop`, `request`, `request_stream`, `background`                           |
|                                   `level` | all                                 | `info`, `warning`, `error`, `critical` (controlled by `LOG_LEVEL`)                          |
|                                    `date` | all                                 | RFC3339, timezone‑aware timestamp                                                           |
|                               `server_id` | all                                 | Instance identifier — on ECS: `task_id-container_name-uuid`; elsewhere: `hostname-pid-uuid` |
|                          `server_version` | all                                 | Application version string                                                                  |
|                            `error_detail` | all                                 | Optional list of formatted exception strings                                                |
|                                      `id` | request, request_stream, background | Correlation ID (also returned as `x-request-id`)                                            |
|                       `execution_time_ms` | request, request_stream, background | Duration of the handled block                                                               |
|                                  `method` | request                             | HTTP method                                                                                 |
|                                    `path` | request                             | Request path                                                                                |
|                             `status_code` | request                             | Final HTTP status code                                                                      |
|                               `client_ip` | request                             | Client IP address (if `LOG_CLIENT_IP=true`)                                                 |
|                       `client_user_agent` | request                             | When provided by client                                                                     |
|                                `model_id` | request                             | Targeted model (if applicable)                                                              |
|                                `voice_id` | request                             | TTS voice (if applicable)                                                                   |
|                           `model_regions` | request                             | AWS region(s) that handled the request; may contain multiple values when failover occurred  |
|       `request_user_id`, `request_org_id` | request                             | Propagated identifiers (if applicable)                                                      |
|                          `request_params` | request                             | Sanitized request payload (if `LOG_REQUEST_PARAMS=true`)                                    |
|                        `request_response` | request                             | Sanitized response payload (if `LOG_REQUEST_PARAMS=true`)                                   |
|                                   `event` | background                          | Background operation name                                                                   |
| `server_start_time_ms`, `server_warnings` | start                               | Startup metrics and warnings                                                                |
|                        `server_uptime_ms` | stop                                | Uptime at shutdown                                                                          |

!!! note "Understanding warnings and errors"
    - For `request` events, default log levels are derived from the final HTTP status: 4xx → `warning`, 5xx → `error`. Unexpected server crashes (like HTTP 500) may appear as `critical`.
    - Authentication/authorization: For security, client responses for `401` and `403` include only generic messages. Full diagnostic details are captured in server logs under `error_detail` and can be correlated via `id` (see `x-request-id`).
    - `server_warnings` (on the `start` event) often highlights missing configuration and features that have been disabled as a result (for example, no S3 bucket configured disables certain image/audio features).
    - `error_detail` (on any event) contains formatted exception traces and diagnostic hints, which frequently point to missing configuration, unavailable dependencies, or disabled features.

---

## :material-cash-multiple: Usage Metrics Fields

Usage is reported as a nested `usage` list on `request` / `request_stream` events. Each entry represents billed quantities for a specific **(service, model, operation, region, tier, routing, context)** combination. Zero/empty fields are omitted.

### Entry Structure

| Field                       | Type | Description                                                                                            |
|:----------------------------|:-----|:-------------------------------------------------------------------------------------------------------|
| `service`                   | str | AWS service/API: `bedrock-runtime`, `polly`, `transcribe`, `translate`, `comprehend`                  |
| `model`                     | str | Model identifier (e.g., `amazon.titan-embed-v1`, `amazon.transcribe`)                                 |
| `operation`                 | str | Request route path (e.g., `/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/speech`)              |
| `region`                    | str | AWS region that served the request (part of the aggregation key)                                       |
| `tier`                      | str | Service tier that actually served the call (AWS-reported when available, else as requested): `standard`, `flex`, `priority`, `batch` |
| `routing`                   | str | Serving profile, present as `"global"` (cross-region global routing) or `"latency"` (latency-optimized) |
| `context`                   | str | Present as `"long"` only when the call's prompt (input + cache read/write tokens) exceeded 200K tokens — billed at the long-context rate where AWS publishes one |
| `cost`                      | str | Exact cost as plain-decimal text (no exponent, no trailing zeros, e.g. `"0.000015"`)                   |
| `currency`                  | str | ISO currency code for `cost`                                                                            |
| `costs`                     | dict| Per-currency exact cost text, replacing `cost`/`currency` when dimensions span multiple currencies      |
| `input_tokens`              | int | Real input token count (Converse, InvokeModel, embeddings)                                            |
| `output_tokens`             | int | Real output token count (Converse, InvokeModel)                                                        |
| `total_tokens`              | int | Real total tokens (Converse provides this directly)                                                    |
| `cached_tokens`             | int | Real cached read tokens from Converse prompt caching                                                   |
| `cache_write_tokens`        | int | Real cache write tokens from Converse prompt caching                                                   |
| `cache_write_tokens_by_ttl` | dict| Cache write tokens by TTL, from Converse `cacheDetails` (e.g., `{"1h": 700, "5m": 100}`)            |
| `output_images`             | int | Real output image count from image generation models                                                   |
| `output_images_by_spec`     | dict| Output image count keyed by `"<resolution>:<quality>"`, for models priced per resolution/quality       |
| `input_seconds`             | int | Real input media duration in seconds (Transcribe with a 15s minimum; audio/video embedding inputs)     |
| `input_seconds_by_spec`     | dict| Input seconds keyed by modality (e.g., `{"audio": 42}`), for models priced per media type              |
| `input_images`              | int | Input image count (multimodal embeddings, billed per image)                                            |
| `input_images_by_spec`      | dict| Input image count keyed by rate bucket (e.g., `{"document": 1}`)                                       |
| `input_characters`          | int | Real input character count (Polly `RequestCharacters`, Translate source text)                         |
| `comprehend_units`          | int | Real Comprehend units (100-char units, 3-unit minimum per call)                                       |
| `grounding_requests`        | int | Built-in grounding tool invocations (e.g., Amazon Nova Grounding `web_search`, billed per request)     |
| `search_units`              | int | Rerank search units (one per rerank query)                                                             |

!!! note "Usage field placement"
    - **Non-streaming requests**: Usage appears on the `request` event
    - **Streaming requests**: Usage appears on the `request_stream` event (not `request`) — this includes streaming speech-to-text
    - **Polly / Translate**: Always on `request` event
    - Cost dashboards must aggregate **both** entry types by request `id`

!!! note "Zero values are omitted"
    Fields are only populated when non-zero. Query for present fields rather than filtering by `> 0`. Missing fields = zero.

!!! note "Client disconnect behavior"
    - **Chat (Converse) streams**: if the client disconnects before the trailing `metadata.usage` event arrives, the wrapper keeps draining the already-open Bedrock stream (bounded) looking for it, so the usage already billed by AWS is still recorded.
    - **InvokeModel streams and image generation jobs**: if a stream/job is abandoned before its final `invocationMetrics` chunk (or job completion) arrives, no usage is produced and logged.

!!! note "No estimates"
    All values are real AWS-billed quantities. No estimation is performed — if AWS doesn't return a count, the field is absent (zero).

---

## :material-chart-line: CloudWatch Metrics (EMF)

When `CLOUDWATCH_METRICS=true` is enabled, usage is emitted as CloudWatch Embedded Metric Format (EMF) lines to stdout. These are automatically extracted as CloudWatch Metrics on ECS with no extra IAM/API calls.

### Configuration

| Setting                          | Default  | Description                               |
|:----------------------------------|:---------|:-------------------------------------------|
| `CLOUDWATCH_METRICS`             | `false`  | Enable/disable EMF emission               |
| `CLOUDWATCH_METRICS_NAMESPACE`   | `stdapi` | CloudWatch namespace for the metrics      |

### EMF Line Structure

Each billed usage entry generates one EMF JSON line:

```json
{
  "_aws": {
    "Timestamp": 1748716800000,
    "CloudWatchMetrics": [{
      "Namespace": "stdapi",
      "Dimensions": [["Model"]],
      "Metrics": [
        {"Name": "InputTokens", "Unit": "Count"},
        {"Name": "OutputTokens", "Unit": "Count"}
      ]
    }]
  },
  "Model": "anthropic.claude-sonnet-5",
  "operation": "/v1/chat/completions",
  "InputTokens": 1500,
  "OutputTokens": 450
}
```

### Metric Details

- **Single dimension**: `Model` — provides low cardinality for cost control. A second `[Model, Currency]` dimension set is added for the `Cost` metric — see [Cost Management](operations_cost_management.md#cost-tracking-real-time-aws-pricing) for why cost needs its own dimension set.
- **`operation`**: Included as a queryable field (not a dimension)
- **Metric units**: `Count` for all token/image/character counts, `Seconds` for audio duration

!!! tip "Querying EMF metrics"
    Use CloudWatch Logs Insights to search EMF lines:
    ```sql
    fields @timestamp, Model, InputTokens, OutputTokens
    | filter _aws.CloudWatchMetrics is not null
    | sort @timestamp desc
    | limit 20
    ```

!!! note "Performance"
    EMF lines bypass log-level filtering and write directly to stdout. This ensures metrics are always available on ECS where CloudWatch Agent scrapes stdout.

---

## :material-currency-usd: Cost Tracking and Attribution { #cost-tracking-real-time-aws-pricing }

Real-time cost computation (`COST_TRACKING`), the price catalog and its accuracy caveats, the `Cost` EMF metric, and AWS-side cost attribution now live on their own page.

[:octicons-arrow-right-24: Cost Management](operations_cost_management.md)

---

## :material-link-variant: Correlating Logs and Traces

- Group events by `id` to reconstruct a full request lifecycle (request → stream(s) → background).
- The `x-request-id` response header exposes the same value so external systems can propagate correlation.
- With OTel enabled, a root span named like `POST /v1/...` is created and carries attributes: `http.method`, `http.url`, `http.user_agent`, `request.id`, `server.id`, `http.status_code`, and `duration_ms`.

!!! tip "Do and Don't for correlation"
    - Do propagate `x-request-id` across client → service → downstreams when possible.
    - Do use `request_stream` durations to account for total user‑perceived latency.
    - Don't generate your own request IDs for the same hop; prefer the provided one.

---

## :material-magnify: Reading the Logs

- High latency: Inspect `execution_time_ms` on the `request` event. If the response was streamed, also sum `request_stream` durations. Combine with OTel spans to locate downstream delays (model provider, S3, etc.).
- Errors: Look for `level=critical` and `error_detail` (formatted exceptions). With OTel, the span is marked error with attributes `error=true` and `error.message`.
- Payload issues: Temporarily enable `LOG_REQUEST_PARAMS=true` to validate requests/responses, then disable.
- Client identification: `client_user_agent` and optional `request_user_id` / `request_org_id` help tie requests to users.
- Routing confirmation: `model_id` and `voice_id` confirm which provider/model/voice handled the request.

!!! warning "When to open a GitHub issue"
    If you encounter `level=critical` events, capture representative JSON log lines (redacting sensitive data) and open an issue on [GitHub](https://github.com/stdapi-ai/stdapi.ai/issues). Include information about the failing request to help reproduce the issue.

---

## :material-filter: Controlling Log Verbosity

The `LOG_LEVEL` environment variable controls which log events are written to STDOUT. Set it to filter out lower-severity events. For detailed configuration options, see the [Logging Level](operations_configuration.md#logging-level) section in the Configuration Guide.

- **`info`** (default): All events are logged (info, warning, error, critical)
- **`warning`**: Only warnings and higher severity (warning, error, critical) - recommended for production
- **`error`**: Only errors and critical events
- **`critical`**: Only critical events
- **`disabled`**: No log output (not recommended)

```bash
# Production example: reduce log volume while maintaining visibility
export LOG_LEVEL=warning
```

!!! tip "Reducing CloudWatch Costs"
    In high-traffic production environments, setting `LOG_LEVEL=warning` or `LOG_LEVEL=error` can significantly reduce CloudWatch Logs ingestion and storage costs by filtering out routine `info`-level events. This is especially effective when combined with appropriate retention policies.

    Additionally, infrastructure routes are automatically excluded from logging to reduce noise: `/docs`, `/favicon.ico`, `/health`, `/openapi.json`, `/redoc`.

---

## :material-graphql: OpenTelemetry Integration

When `OTEL_ENABLED=true`:

- A span is created per request and for streaming/background blocks.
- Spans carry `request.id` and `server.id` for correlation.
- 4xx/5xx `status_code` marks the span with an error status.
- Sampling is controlled via `OTEL_SAMPLE_RATE`.

For exporters and advanced setup, rely on standard OTel environment variables supported by your exporter/backend.

!!! note "No X-Ray IAM permissions needed"
    Export is push-based OTLP/HTTP to the endpoint you configure — the application makes no direct AWS X-Ray API calls and needs no extra IAM permissions for tracing. To forward traces to AWS X-Ray, point the OTLP endpoint at an [ADOT collector](https://aws-otel.github.io/docs/introduction), which handles the X-Ray upload using its own credentials.

---

## :material-code-json: Example Events

__Example — Request with payload logging enabled__

```json
{
  "type": "request",
  "level": "info",
  "date": "2025-01-01T12:00:00Z",
  "server_id": "stdapi-1",
  "id": "a1b2c3d4",
  "method": "POST",
  "path": "/v1/chat/completions",
  "status_code": 200,
  "model_id": "anthropic.claude-sonnet-5",
  "execution_time_ms": 842,
  "request_params": {"messages": [{"role": "user", "content": "..."}]},
  "request_response": {"id": "cmpl_...", "choices": [...], "usage": {...}}
}
```

__Example — Streaming segment (SSE/audio)__

```json
{
  "type": "request_stream",
  "level": "info",
  "date": "2025-01-01T12:00:01Z",
  "server_id": "stdapi-1",
  "id": "a1b2c3d4",
  "execution_time_ms": 1234
}
```

__Example — Background work correlated to a request__

```json
{
  "type": "background",
  "level": "info",
  "date": "2025-01-01T12:00:02Z",
  "server_id": "stdapi-1",
  "id": "a1b2c3d4",
  "event": "image-upload-s3",
  "execution_time_ms": 97
}
```

__Example — Error with captured details__

```json
{
  "type": "request",
  "level": "critical",
  "date": "2025-01-01T12:00:05Z",
  "server_id": "stdapi-1",
  "id": "e9f0a1b2",
  "method": "POST",
  "path": "/v1/images/edits",
  "status_code": 500,
  "error_detail": ["Traceback (most recent call last): ..."],
  "execution_time_ms": 12
}
```

---

## :material-text-search: CloudWatch Logs Insights Queries

These examples assume JSON logs in CloudWatch Logs (default with ECS awslogs/awsfirelens). Adjust the log group and time range.

### 1) Follow a specific request across request/stream/background

```sql
fields @timestamp, type, level, path, event, status_code, execution_time_ms
| filter id = "<paste-request-id>"
| sort @timestamp asc
```

!!! tip
    Copy the request ID from the `x-request-id` response header or any `request` log line. Expect one `request`, optional `request_stream` entries, and `background` entries.

### 2) Find recent errors with context

```sql
fields @timestamp, level, type, path, status_code, id, error_detail
| filter level in ["error", "critical"]
| sort @timestamp desc
| limit 100
```

### 3) High-latency endpoints (P95/P99)

```sql
fields path, execution_time_ms
| filter type = "request" and ispresent(execution_time_ms)
| stats pct(execution_time_ms, 95) as p95_ms, pct(execution_time_ms, 99) as p99_ms, avg(execution_time_ms) as avg_ms by path
| sort p95_ms desc
```

---

## :material-aws: AWS Service-Level Logs and Metrics

Beyond stdapi.ai logs and OTel traces,
use AWS-native signals from the underlying AI services to validate provider behavior,
monitor throttling/latency, and audit access.
Enable only what you need: some options can capture content and increase costs.
For details, refer to the official AWS documentation.

- CloudWatch Metrics: Throughput, latency, throttling, and error rates per service/region.
- CloudTrail: Control-plane auditing of API calls (who did what, when, from where).
- Content/Invocation logging: Optional features that may record inputs/outputs. Use with caution and encryption/retention controls.
- Correlation: Service logs won't include stdapi.ai's `x-request-id`. Correlate by time window, region, model/voice/job identifiers, and volume. Use stdapi.ai's `model_id`, `voice_id`, and `execution_time_ms` to narrow windows.
- Amazon Bedrock Invocation logging (optional): Export invocation metadata and, if enabled, content to CloudWatch Logs/S3/Firehose. Treat prompts/completions as sensitive; manage retention and KMS.

### AWS Service Correlation Metadata

stdapi.ai attaches correlation identifiers to outgoing service calls, allowing you to trace a stdapi.ai request back to the corresponding AWS service invocation.

| Key                    | Description                                                                                                                      |
|:-----------------------|:-----------------------------------------------------------------------------------------------------------------------------------|
| `stdapi-ai.request_id` | Matches the `id` field in stdapi.ai logs and the `x-request-id` response header                                                  |
| `stdapi-ai.server_id`  | Matches the `server_id` field in stdapi.ai logs                                                                                  |
| `stdapi-ai.user_id`    | Present only when the client supplies a user identifier — the `safety_identifier` field (or the deprecated `user` alias) in OpenAI requests, `metadata.user_id` in Anthropic requests |

Coverage and how to use it varies by service:

- **Bedrock — synchronous inference** (every chat, embedding, image and audio route served by `bedrock-runtime`): identifiers are embedded in the invocation request — both the `Converse` and the `InvokeModel` families, streaming included — and appear in Bedrock invocation log records when [model invocation logging](https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html) is enabled. Filter by `requestMetadata.stdapi-ai\.request_id` in CloudWatch Logs Insights to join a Bedrock record with its stdapi.ai event.
- **Bedrock — asynchronous batch jobs**: identifiers are attached as resource tags on the job, visible in the AWS Console and searchable via the CLI/API.
- **Bedrock Mantle**: the Mantle endpoint accepts no request metadata. Attribute those requests with a [Project/Workspace](operations_configuration.md#bedrock-mantle-project) instead.
- **Transcribe — audio transcription**: identifiers are attached as job tags. stdapi.ai deletes completed transcription jobs automatically, so tags are only available while the job is still running.

!!! note "Security"
    Any `stdapi-ai.*` key supplied by the client is silently dropped before the service call. Only values injected by stdapi.ai itself are forwarded under that prefix.

---

## :material-cloud-check: Infrastructure Observability

!!! info "Terraform Module"
    The Terraform module automatically configures production-ready observability for all infrastructure components — no manual setup required.

    **Application Logs (ECS)**

    Container STDOUT is forwarded to a dedicated CloudWatch Logs log group. Log retention defaults to **365 days** (`cloudwatch_logs_retention_in_days`). All log groups are **KMS-encrypted**.

    **Container Insights**

    ECS Container Insights is enabled by default (`container_insight = "enabled"`), providing CPU, memory, network, and storage metrics per task. Set `container_insight = "enhanced"` to enable enhanced observability with additional OS-level and application performance metrics.

    **CloudWatch Alarms**

    The module can trigger a CloudWatch alarm whenever an `error` or `critical` log event is detected. Enable with `alarms_enabled = true` and provide `sns_topic_arn` to receive notifications via SNS (email, Slack, PagerDuty, etc.). When enabled, a metric filter scans the ECS log group for any line containing `error` or `critical` and fires the alarm as soon as the count exceeds zero in a 5-minute window.

    **ALB and WAF Logs**

    ALB access logs and WAF logs are stored in dedicated S3 buckets, **KMS-encrypted**. These capture all HTTP requests at the infrastructure level — use them to audit traffic patterns and investigate security events before they reach the application.

---

## :material-wrench: Troubleshooting Checklist

- No logs visible: Ensure you are reading container STDOUT. On ECS/Kubernetes, verify the log driver and retention.
- Missing `request_params`: Confirm `LOG_REQUEST_PARAMS=true` and restart after changing environment variables.
- No traces: Verify `OTEL_ENABLED=true` and that exporters are configured and reachable.
- Correlation missed: Ensure clients read and propagate `x-request-id` for multi‑hop requests.

---

## :material-arrow-right: Next Steps

<div class="grid cards" markdown>

- :material-cash-multiple: [**Cost Management**](operations_cost_management.md) — Cost tracking, AWS Marketplace billing, and cost attribution
- :material-cog: [**Configuration Reference**](operations_configuration.md) — Complete list of environment variables including logging options
- :material-chart-bar: [**Data Sovereignty & Compliance**](operations_compliance.md) — Compliance-focused logging and audit requirements
- :material-server-network: [**Advanced Deployment**](operations_deploy_advanced.md) — Production deployment with observability stack

</div>
