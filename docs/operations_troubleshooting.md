---
title: Troubleshooting - Common stdapi.ai deployment issues
description: Fixes for the most common errors encountered when deploying and running stdapi.ai - Terraform failures, 400/401/403/404/429/503 responses, Bedrock throttling, IAM permission errors, S3 bucket errors, VPC connectivity, and more.
keywords: stdapi.ai troubleshooting, AWS Bedrock errors, Terraform apply failed, 503 ECS service, 401 API key, 403 permission IAM, AccessDeniedException, 404 model not found, ThrottlingException Bedrock, S3 bucket region, ElastiCache capacity, VPC endpoint timeout, podman SELinux, ECONNREFUSED IPv6 service discovery, drop_params ValidationException
---

# :material-wrench: Troubleshooting

Common issues when deploying stdapi.ai for the first time. If your error isn't listed here, see the [Contact](contact.md) page, or open an issue on [GitHub](https://github.com/stdapi-ai/stdapi.ai/issues).

---

## :material-cloud-upload: Terraform / Deployment

??? failure "`terraform apply` fails with AccessDenied on IAM, KMS, or ECS actions"
    Your AWS profile does not have sufficient permissions. The stdapi.ai Terraform module provisions IAM roles, KMS keys, ECS, ALB, Route53 records, an optional WAF, and (for some samples) RDS and ElastiCache.

    - Use an **administrator-level** AWS profile for the evaluation deployment.
    - **Recommended**: deploy into a sandbox/non-production AWS account first, then replicate into your target account with scoped-down principals once validated.
    - Verify your active identity: `aws sts get-caller-identity`.

??? failure "`terraform apply` succeeds but nothing is reachable"
    Terraform completed but the ECS service is still coming up. The ALB returns `503 Service Unavailable` until tasks pass health checks.

    - Wait 2–3 minutes after `terraform apply` completes.
    - Check ECS service status: `aws ecs describe-services --cluster <cluster> --services <service>`.
    - Check task logs in CloudWatch: `/aws/ecs/<service-name>`.

??? failure "Wrong AWS region or profile used by Terraform"
    The AWS provider uses the region/profile from your environment, not a Terraform variable.

    - Confirm before applying:
      ```bash
      aws sts get-caller-identity
      aws configure get region
      ```
    - Set explicitly with `AWS_PROFILE=... AWS_REGION=... terraform apply` if needed.

??? failure "ElastiCache creation failed — insufficient capacity in AZ (Open WebUI sample)"
    The ElastiCache Valkey cache occasionally fails to create when the target availability zone is out of capacity.

    ```text
    Error: waiting for ElastiCache Replication Group ... create: unexpected state 'create-failed',
    wanted target 'available'
    ```

    - Remove the failed Valkey cache from the ElastiCache console (disable backups first, then wait for full deletion) and re-run `terraform apply`.
    - If the problem persists, change `node_type` in `valkey.tf` (e.g. `cache.t4g.micro` → `cache.t3.micro`) and retry.

??? failure "Podman volume mount fails on Fedora/RHEL with SELinux (local Docker)"
    SELinux blocks container access to `~/.aws` without a relabel.

    - Add `:z` (or `:Z` for exclusive use) to the volume and `--userns=keep-id`:
      ```bash
      podman run --rm -p 8000:8000 \
        --userns=keep-id \
        -v ~/.aws:/home/nonroot/.aws:ro,z \
        -e AWS_BEDROCK_REGIONS=us-east-1,us-west-2 \
        -e ENABLE_DOCS=true \
        ghcr.io/stdapi-ai/stdapi.ai-community:latest
      ```
    - See [Local Development](operations_getting_started_local.md#run-it) for the full run command.

---

## :material-api: Runtime / First API call

??? failure "`503 Service Unavailable` — on the /docs page or any endpoint"
    The ECS service is still starting up. Health checks take a few minutes.

    - Wait 2–3 minutes after deployment and refresh.
    - Check the ALB target group health in the AWS console.
    - If it persists longer than 5 minutes, inspect CloudWatch logs for the ECS task.

??? failure "`ECONNREFUSED` — connection refused, but only from some clients"
    The images bind IPv4 only (`GRANIAN_HOST=0.0.0.0`), so a client that resolves the server to an IPv6 address reaches a port nothing is listening on. Clients disagree about which address to try first, which is why the same deployment looks reachable from one language and dead from another: Node.js prefers the `AAAA` record and fails outright, while most Python clients fall back to the `A` record and hide the problem.

    - Typically hit with **ECS service discovery**, which publishes an `AAAA` record for every task in an IPv6-enabled subnet. Deployments fronted by an ALB are unaffected — the load balancer terminates the client connection itself and reaches the task over IPv4.
    - Set `GRANIAN_HOST=::` for a dual-stack socket answering both families; see the Container Runtime note in [Configuration](operations_configuration.md). The [Terraform module](https://github.com/stdapi-ai/terraform-aws-stdapi-ai) sets it when the VPC has IPv6 enabled.
    - After switching, extend [`PROXY_TRUSTED_HOSTS`](operations_configuration.md#proxy-trusted-hosts) with the IPv4-mapped form of each range (`::ffff:10.0.0.0/112`) — a dual-stack listener reports IPv4 peers in that form, and an untrusted proxy means `X-Forwarded-For` is ignored and the load balancer's own address is logged as the client IP.

??? failure "Targets never become healthy after setting `TRUSTED_HOSTS`"
    Host header validation applies to `/health` as well. A load balancer health check addresses the target directly, so its `Host` header carries the target's IP address — which a list of domain names does not match, and every probe is answered with `400`. The target group stays unhealthy and the ALB keeps returning `503`.

    - Prefer host validation at the load balancer: an ALB listener rule on the `Host` header, with `TRUSTED_HOSTS` left unset.
    - If the application-level allow-list is required, include the address the health check actually sends.
    - The container's own `HEALTHCHECK` is unaffected: it derives its `Host` header from `TRUSTED_HOSTS`. See [`TRUSTED_HOSTS`](operations_configuration.md#trusted-hosts).

??? failure "Browser TLS warning on the /docs page"
    The ALB uses the default `*.elb.amazonaws.com` domain, which has no trusted certificate. This is expected and safe to bypass for testing.

    - For a production-grade certificate, configure a custom domain — the [Terraform module](https://github.com/stdapi-ai/terraform-aws-stdapi-ai) supports ACM-managed certificates via `alb_domain_name`.

??? failure "`401 Unauthorized` — client API key missing or wrong"
    The API key is missing, wrong, or not configured.

    - Pass the key in the `Authorization: Bearer <key>` header (OpenAI-style) or `X-API-Key` header.
    - Retrieve the generated key with `terraform output -raw api_key`.
    - If `api_key_create = true` was not set, no API key is configured and requests pass through without authentication by default (useful for testing behind IP-restricted ALB, not for production).
    - See [Authentication & Security](operations_authentication_security.md) for all options.

??? failure "`403 Forbidden` — IAM permission denied on API calls"
    The gateway reached Amazon Bedrock, but the **ECS task role** (or your local AWS credentials) lacks permission for the requested action. AWS returns `AccessDeniedException`, which stdapi.ai maps to HTTP `403` with error type `permission_error`. This is an IAM misconfiguration, **not** a client API-key problem.

    - Confirm the task role grants `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` (and the `bedrock:Converse*` actions) for the target model ARNs.
    - For models invoked through an inference profile, allow both the profile ARN and the underlying foundation-model ARNs in the policy.
    - Some models require one-time activation in the **Bedrock console → Model access** page before they can be invoked.
    - Audio, embeddings, and file features need permissions for the relevant services (Polly, Transcribe, Translate, Comprehend, S3) — see [Configuration → IAM Permissions](operations_configuration.md#iam-permissions) for the full IAM reference.

??? failure "`401 Unauthorized` — AWS credentials invalid or expired (often local Docker)"
    stdapi.ai's **own** AWS credentials are missing, invalid, or expired — AWS returns `UnrecognizedClientException`, `InvalidSignatureException`, or `ExpiredTokenException`, which stdapi.ai maps to HTTP `401` with error type `authentication_error`. This is distinct from the client-facing API-key `401` above (which concerns your `Authorization` / `X-API-Key` header).

    - Locally: refresh with `aws sso login` (or update `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`) and restart the container.
    - On ECS: confirm the task is assuming its IAM role rather than relying on stale static keys.

??? failure "`404 Not Found` — model not available"
    The model ID isn't available in your configured region(s).

    - Start with a low-friction model to confirm the pipeline works: `amazon.nova-micro-v1:0` (available in all standard Bedrock regions).
    - List every discovered model with full details: `GET /search_models` (the default model-discovery endpoint). Filter by capability with query parameters — e.g. `GET /search_models?input_modalities=IMAGE&route=/v1/chat/completions` returns only vision-capable chat models. See the [Search Models API](api_search_models.md) reference.
    - `GET /v1/models` is also available for strict OpenAI SDK compatibility (lighter payload, no capability metadata).
    - Verify `AWS_BEDROCK_REGIONS` includes a region that offers the model — see the [Bedrock model availability table](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html).
    - For Anthropic SDK clients, use either the full Bedrock ID (`anthropic.claude-fable-5`) or the Anthropic alias (`claude-fable-5`) — both resolve automatically.

??? failure "`429 Too Many Requests` — Bedrock throttling / quota"
    AWS returned `ThrottlingException`, `TooManyRequestsException`, or `ServiceQuotaExceededException` — mapped to HTTP `429` with error type `rate_limit_error`. You've hit the per-region Bedrock quota.

    - Add more regions to `AWS_BEDROCK_REGIONS`. Each region has its own independent quota — three regions ≈ triple the throughput.
    - See [Resilience & Failover](operations_resilience.md) for multi-region routing configuration.
    - Check quotas in the AWS Service Quotas console for **Amazon Bedrock**.
    - When the router put a region on a quota backoff while serving the request, the response carries a `retry-after` header (in seconds) telling the client exactly how long to wait — OpenAI, Anthropic and Cohere SDKs honour it automatically instead of guessing an exponential backoff, up to the 60 s ceiling their retry loops apply to server-supplied delays.

??? failure "`400 Bad Request` — This model is not available under data retention mode 'default'."
    A specific model is unavailable or requests to it are rejected because your account's data retention mode is incompatible with what that model requires.

    Amazon Bedrock enforces retention compatibility at invocation time: each model declares the retention modes it accepts, and if your effective mode is not among them, the request is blocked.

    **Common scenarios:**

    - Your account is set to **zero data retention (`none`)** but the model requires `default` or `provider_data_share` for safety or abuse-prevention purposes. Bedrock blocks the request to honour your retention policy. To access the model, either switch to a compatible retention mode or contact your AWS account manager to request ZDR eligibility for that specific model.
    - Your account is set to **`default`** but the model exclusively requires `provider_data_share` (typically models with mandatory provider-side safety review). The model will appear as unavailable. Enabling `provider_data_share` grants access but means AWS will share your inference data with the model provider — see [Data Privacy](operations_compliance.md#data-privacy) before enabling it.

??? failure "`400 Bad Request` — invalid parameters from Bedrock"
    Bedrock rejected the request parameters (`ValidationException` / `BadRequestException`), mapped to HTTP `400` with error type `invalid_request_error` — for example an unsupported parameter for the chosen model, an out-of-range value, or content that exceeds the model's limits.

    - Read the message detail returned in the response (correlate with `x-request-id` in the server logs).
    - Confirm the parameter is supported by the model — see the per-API **Feature Compatibility** tables.

??? failure "`400 Bad Request` — a client's own control flag reaches Bedrock as a model parameter"
    Request fields stdapi.ai does not declare are forwarded to Amazon Bedrock as provider-specific inference parameters, so any parameter a model accepts can be passed through — including the highly model-specific ones no common API surface exposes. Some OpenAI-SDK-based clients also use that same channel for their *client-side* settings — LiteLLM-derived ones send `drop_params`, `api_key` or `custom_llm_provider` in `extra_body` — and Bedrock answers `ValidationException` for a field no model declares. The symptom is a route that fails for one client and works for every other.

    - The known LiteLLM control parameters are stripped by a built-in denylist, so this only appears for a name it does not yet cover. The rejected field is in the Bedrock message detail, correlated via `x-request-id` in the server logs.
    - Add that name to [`EXTRA_MODEL_PARAMS_DENYLIST`](operations_configuration.md#extra-model-params-denylist) — it is merged with the built-in list, and every other extra parameter keeps being forwarded.
    - If no client needs the passthrough, [`EXTRA_MODEL_PARAMS_DROP_ALL`](operations_configuration.md#extra-model-params-drop-all) disables it outright. Per-model defaults set through [`DEFAULT_MODEL_PARAMS`](operations_configuration.md#default-model-params) are unaffected — only request-supplied extras are dropped.

??? failure "S3 error on image generation or audio transcription"
    The S3 bucket is missing, unreachable, or in the wrong region.

    - The Terraform module creates the bucket automatically unless you pass your own via `aws_s3_bucket`.
    - If you're using your own bucket: `AWS_S3_BUCKET` must point to a bucket in the same region as the **first** entry in `AWS_BEDROCK_REGIONS`.
    - Verify the ECS task IAM role has `s3:PutObject` / `s3:GetObject` on the bucket.

??? failure "Connection timeout to AWS services from ECS"
    Outbound traffic to AWS endpoints is blocked.

    - Confirm the ECS task's security group allows outbound HTTPS (port 443).
    - If using **VPC endpoints** (the Terraform module default), verify the endpoint security groups and policies permit traffic from the ECS task subnet.
    - If ECS runs in a private subnet without VPC endpoints, confirm the NAT gateway / route table is configured.

??? failure "`413 Payload Too Large` — request or file rejected as oversized"
    Either the application-level file-size cap or an edge control rejected the request.

    - Check [`MAX_INPUT_FILE_SIZE`](operations_configuration.md#max-input-file-size) — it caps the bytes of any single file loaded into memory for model input; disabled by default, so if it's set and the error appears, raise it or reduce the input size.
    - If the deployment sits behind the Terraform module's WAF (`alb_waf_enabled=true`), check for a `SizeConstraintStatement` rule on the request body — see [Request Size & Resource Limits](operations_authentication_security.md#request-size-resource-limits).
    - If fronted by Amazon API Gateway instead of an ALB, remember its hard 10 MB payload limit.

??? failure "`503`/`504`/`408` — request times out mid-stream on long generations"
    A slow or hung generation exceeded a timeout somewhere between the model and the client. The status code tells you where: `503` comes from stdapi.ai's own gateway timeout; `504`/`408` come from an edge or proxy timeout in front of it.

    - **`503` from the gateway**: Check [`AI_RESPONSE_TIMEOUT`](operations_configuration.md#ai-response-timeout) — it closes stalled upstream model connections; raise it for workloads with long-running generations. The request is **not** retried in another region: the model already ran and AWS bills it either way, so a failover would pay twice for the same generation.
    - **`504`/`408` from the edge/proxy**: Check the Terraform module's `alb_idle_timeout` (default: 3600 s) — if you lowered it, or front the deployment with your own load balancer or reverse proxy at a shorter idle timeout, streaming responses can be cut off mid-flight before the gateway's own timeout fires. See [ALB Resilience](operations_resilience.md#alb-resilience).

### AWS error → HTTP status mapping

stdapi.ai translates upstream AWS error codes into standard HTTP responses with an OpenAI/Anthropic-style error type. Use this table to map a status code back to its likely AWS cause. HTTP status and error type are as returned on OpenAI-compatible routes (`/v1/...`); Anthropic-compatible routes (`/anthropic/...`) diverge on the two footnoted rows.

| HTTP  | Error type                  | AWS error codes                                                                                         | Typical cause                                 |
|-------|------------------------------|---------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| `400` | `invalid_request_error`     | `ValidationException`, `BadRequestException`                                                            | Unsupported/invalid request parameters        |
| `401` | `authentication_error`      | `UnrecognizedClientException`, `InvalidSignatureException`, `ExpiredTokenException`                     | stdapi.ai's AWS credentials missing/expired   |
| `403` | `permission_error`          | `AccessDeniedException`                                                                                 | IAM task role lacks permission / model access |
| `404` | `invalid_request_error`[^1] | `ResourceNotFoundException`                                                                             | Model or resource not available in the region |
| `429` | `rate_limit_error`          | `ThrottlingException`, `TooManyRequestsException`, `ServiceQuotaExceededException`                      | Bedrock quota / throttling                    |
| `503` | `server_error`[^2]          | `ServiceUnavailableException`, `InternalServerException`, `ServiceFailureException`, `ReadTimeoutError` | Transient AWS-side error — retry              |

[^1]: Anthropic-compatible routes return `not_found_error` instead.
[^2]: Anthropic-compatible routes return HTTP `529` with error type `overloaded_error` instead.

!!! note "Where to find the detail"
    For security, `401` and `403` responses returned to clients contain only a generic message. The full diagnostic detail is captured in the server logs under `error_detail` and can be correlated via the `x-request-id` response header (`request-id` on Anthropic-compatible `/anthropic/...` routes) — see [Logging & Monitoring](operations_logging_monitoring.md).

---

## :material-key-variant: Authentication & Identity

??? failure "Bearer token works, but Anthropic SDK requests fail"
    The Anthropic SDK uses a different auth header than OpenAI.

    - Use `x-api-key: <your-key>` (not `Authorization: Bearer`).
    - Set the base URL to `https://<your-endpoint>/anthropic` (not `/v1`).
    - See [API Overview → Anthropic-Compatible API](api_overview.md#using-the-anthropic-compatible-api).

??? failure "OIDC/Cognito redirect loop or 401 from the ALB"
    Authentication is enforced by the ALB listener, not stdapi.ai.

    - Verify the OIDC issuer URL, client ID, client secret, and redirect URI in the ALB listener rule.
    - For Cognito, confirm the app client is configured as a "confidential" client with a client secret.
    - See [Authentication & Security → via Application Load Balancer (ALB)](operations_authentication_security.md#via-application-load-balancer-alb).

---

## :material-arrow-right: Next Steps

<div class="grid cards" markdown>

- :material-cog: [**Configuration Reference**](operations_configuration.md) — Every environment variable
- :material-server-network: [**Advanced Deployment**](operations_deploy_advanced.md) — VPC integration, manual ECS, multi-region
- :material-lock: [**Authentication & Security**](operations_authentication_security.md) — API keys, OIDC/Cognito, IAM
- :material-terraform: [**Terraform Module Docs**](https://github.com/stdapi-ai/terraform-aws-stdapi-ai) — All module inputs and outputs
- :material-github: [**GitHub Issues**](https://github.com/stdapi-ai/stdapi.ai/issues) — Report a bug or ask a question
- :material-email-outline: [**Contact**](contact.md) — Reach the team directly

</div>
