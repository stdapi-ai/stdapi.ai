# Logging and Monitoring

stdapi.ai provides production observability. It emits structured JSON logs for every request, stream, and background task, and integrates with OpenTelemetry (OTel) for traces and metrics. This guide shows how to enable observability, read the logs, and correlate signals across systems.

<div class="grid cards" markdown>

- :material-clipboard-text-outline: __At a glance__
  <br>JSON logs to STDOUT (perfect for AWS CloudWatch Logs). One event per line.

- :material-identifier: __Correlation__
  <br>All events for a request share the same `id` and are returned as `x-request-id`.

- :material-aws: __ECS friendly__
  <br>ECS forwards container STDOUT to CloudWatch Logs automatically.

- :material-graphql: __Traces (optional)__
  <br>Enable `OTEL_ENABLED=true` to export spans to X‑Ray, Jaeger, Tempo, etc.

- :material-alert-decagram-outline: __Payload logging (optional)__
  <br>Enable `LOG_REQUEST_PARAMS=true` only for targeted debugging.

</div>

## Quick start (2 minutes)

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
    JSON to STDOUT is optimal for CloudWatch Logs Insights. In AWS ECS, the task’s log driver forwards container STDOUT to CloudWatch Logs automatically.

## Event types

stdapi.ai emits five kinds of JSON events (one per line):

| Event            | Description                                                                         |
|:-----------------|:------------------------------------------------------------------------------------|
| `start`          | Emitted once at server startup. Includes startup metadata and warnings.             |
| `stop`           | Emitted on graceful shutdown. Includes uptime.                                      |
| `request`        | One per HTTP request. Method, path, status, timings, and optional request/response. |
| `request_stream` | Streaming segments (SSE/audio). Indicates streaming activity and duration.          |
| `background`     | Background tasks correlated to the parent request.                                  |

## Common fields

Each event shares core fields and may add type‑specific ones.

|                                     Field | Applies to                          | Description                                                        |
|------------------------------------------:|:------------------------------------|:-------------------------------------------------------------------|
|                                    `type` | all                                 | One of `start`, `stop`, `request`, `request_stream`, `background`  |
|                                   `level` | all                                 | `info`, `warning`, `error`, `critical` (controlled by `LOG_LEVEL`) |
|                                    `date` | all                                 | RFC3339, timezone‑aware timestamp                                  |
|                               `server_id` | all                                 | Instance identifier                                                |
|                            `error_detail` | all                                 | Optional list of formatted exception strings                       |
|                                      `id` | request, request_stream, background | Correlation ID (also returned as `x-request-id`)                   |
|                       `execution_time_ms` | request, request_stream, background | Duration of the handled block                                      |
|                                  `method` | request                             | HTTP method                                                        |
|                                    `path` | request                             | Request path                                                       |
|                             `status_code` | request                             | Final HTTP status code                                             |
|                               `client_ip` | request                             | Client IP address (if `LOG_CLIENT_IP=true`)                        |
|                       `client_user_agent` | request                             | When provided by client                                            |
|                                `model_id` | request                             | Targeted model (if applicable)                                     |
|                                `voice_id` | request                             | TTS voice (if applicable)                                          |
|       `request_user_id`, `request_org_id` | request                             | Propagated identifiers (if applicable)                             |
|                          `request_params` | request                             | Sanitized request payload (if `LOG_REQUEST_PARAMS=true`)           |
|                        `request_response` | request                             | Sanitized response payload (if `LOG_REQUEST_PARAMS=true`)          |
|                                   `event` | background                          | Background operation name                                          |
| `server_start_time_ms`, `server_warnings` | start                               | Startup metrics and warnings                                       |
|                        `server_uptime_ms` | stop                                | Uptime at shutdown                                                 |

!!! note "Understanding warnings and errors"
    - For `request` events, default log levels are derived from the final HTTP status: 4xx → `warning`, 5xx → `error`. Unexpected server crashes (like HTTP 500) may appear as `critical`.
    - Authentication/authorization: For security, client responses for `401` and `403` include only generic messages. Full diagnostic details are captured in server logs under `error_detail` and can be correlated via `id` (see `x-request-id`).
    - `server_warnings` (on the `start` event) often highlights missing configuration and features that have been disabled as a result (for example, no S3 bucket configured disables certain image/audio features).
    - `error_detail` (on any event) contains formatted exception traces and diagnostic hints, which frequently point to missing configuration, unavailable dependencies, or disabled features.

## Correlating logs and traces

- Group events by `id` to reconstruct a full request lifecycle (request → stream(s) → background).
- The `x-request-id` response header exposes the same value so external systems can propagate correlation.
- With OTel enabled, a root span named like `POST /v1/...` is created and carries attributes: `http.method`, `http.url`, `http.user_agent`, `request.id`, `server.id`, `http.status_code`, and `duration_ms`.

!!! tip "Do and Don’t for correlation"
    - Do propagate `x-request-id` across client → service → downstreams when possible.
    - Do use `request_stream` durations to account for total user‑perceived latency.
    - Don’t generate your own request IDs for the same hop; prefer the provided one.

## Reading the logs (what to look for)

- High latency: Inspect `execution_time_ms` on the `request` event. If the response was streamed, also sum `request_stream` durations. Combine with OTel spans to locate downstream delays (model provider, S3, etc.).
- Errors: Look for `level=critical` and `error_detail` (formatted exceptions). With OTel, the span is marked error with attributes `error=true` and `error.message`.

!!! warning "When to open a GitHub issue"
    If you encounter `level=critical` events,
    capture representative JSON log lines (redacting sensitive data)
    and open an issue at https://github.com/stdapi-ai/stdapi.ai/issues. Include information about the failing request
    to help reproduce the issue.

- Payload issues: Temporarily enable `LOG_REQUEST_PARAMS=true` to validate requests/responses, then disable.
- Client identification: `client_user_agent` and optional `request_user_id` / `request_org_id` help tie requests to users.
- Routing confirmation: `model_id` and `voice_id` confirm which provider/model/voice handled the request.

## Controlling log verbosity

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

## OpenTelemetry integration

When `OTEL_ENABLED=true`:

- A span is created per request and for streaming/background blocks.
- Spans carry `request.id` and `server.id` for correlation.
- 4xx/5xx `status_code` marks the span with an error status.
- Sampling is controlled via `OTEL_SAMPLE_RATE`.

For exporters and advanced setup, rely on standard OTel environment variables supported by your exporter/backend.

## Example events

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
  "model_id": "anthropic.claude-sonnet-4-5-20250929-v1:0",
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

## CloudWatch Logs Insights: ready‑to‑use queries

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

## AWS service-level logs and metrics

Beyond stdapi.ai logs and OTel traces,
use AWS-native signals from the underlying AI services to validate provider behavior,
monitor throttling/latency, and audit access.
Enable only what you need: some options can capture content and increase costs.
For full, up-to-date details,
refer to the official AWS documentation for more information.

- CloudWatch Metrics: Throughput, latency, throttling, and error rates per service/region.
- CloudTrail: Control-plane auditing of API calls (who did what, when, from where).
- Content/Invocation logging: Optional features that may record inputs/outputs. Use with caution and encryption/retention controls.
- Correlation: Service logs won’t include StdAPI `x-request-id`. Correlate by time window, region, model/voice/job identifiers, and volume. Use StdAPI `model_id`, `voice_id`, and `execution_time_ms` to narrow windows.
- AWS Bedrock Invocation logging (optional): Export invocation metadata and, if enabled, content to CloudWatch Logs/S3/Firehose. Treat prompts/completions as sensitive; manage retention and KMS.

## Troubleshooting checklist

- No logs visible: Ensure you are reading container STDOUT. On ECS/Kubernetes, verify the log driver and retention.
- Missing `request_params`: Confirm `LOG_REQUEST_PARAMS=true` and restart after changing environment variables.
- No traces: Verify `OTEL_ENABLED=true` and that exporters are configured and reachable.
- Correlation missed: Ensure clients read and propagate `x-request-id` for multi‑hop requests.
