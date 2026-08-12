---
title: Troubleshooting - Common stdapi.ai deployment issues
description: Fixes for the most common errors encountered when deploying and running stdapi.ai - Terraform failures, 400/401/403/404/429/503 responses, Bedrock throttling, IAM permission errors, S3 bucket errors, VPC connectivity, and more.
keywords: stdapi.ai troubleshooting, AWS Bedrock errors, Terraform apply failed, 503 ECS service, 401 API key, 403 permission IAM, AccessDeniedException, 404 model not found, ThrottlingException Bedrock, S3 bucket region, ElastiCache capacity, VPC endpoint timeout, podman SELinux, ECONNREFUSED IPv6 service discovery, drop_params ValidationException, ECS task unhealthy restart loop, container healthCheck command, text to speech input too long, speech input limited to 3000 characters, speech input limited to 20000 characters, generative voice long input, StartSpeechSynthesisStream AccessDenied, service_tier ignored, guardrail ignored, model parameters ignored, model alias configuration, MODEL_ALIASES validation error, extra inputs are not permitted alias, 413 attachment too large, large image rejected, large PDF rejected, attachments larger than not available, attached files are too large per request, attachment total too large, AWS_S3_REGIONAL_BUCKETS attachment staging, web search returns no results, external_web_access rejected, bedrock-websearch InvokeSearch AccessDenied, web search stale cached results, valid Cognito token rejected 401, Cognito JWT unauthorized, aws_cognito_client_ids required, server will not start after enabling Cognito, server will not start after enabling authentication, API key secret is empty, API_KEY_SSM_PARAMETER empty value, AUTHENTICATION_MODE method not enabled, issuer-cognito-idp issuer mismatch, aws.cognito.signin.user.admin scope missing, WWW-Authenticate Bearer challenge, MCP client cannot discover authentication, oauth-protected-resource 404, protected resource does not match expected, OAUTH_RESOURCE_IDENTIFIER invalid, openid-configuration not served, per-user cost attribution, sts:TagSession AccessDenied, AWS_BEDROCK_USER_ROLE_ARN, all Bedrock spend on one identity, line_item_iam_principal missing, cost allocation tag IAM principal, requests fail after enabling per-user roles, end user role ApplyGuardrail AccessDenied, 403 guardrail per-user cost attribution, end user role missing model ARN form, 503 speech to speech request fails after a few seconds, real-time audio session timeout, AWS_CONNECT_TIMEOUT generative voice
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

??? failure "The ECS task never reports healthy, or restarts in a loop"
    ECS ignores the image's own `HEALTHCHECK`, so a task definition that declares no `healthCheck` gets no container-level probe at all, and one that declares a probe carried over from an earlier version runs a command the current image no longer provides. Either way the container is reported unhealthy and the service replaces it.

    - Copy the `healthCheck` block from the [ECS task definition example](operations_deploy_advanced.md#ecs-task-definition-example), which declares the image's own probe, and re-copy it when upgrading.
    - The [Terraform module](https://github.com/stdapi-ai/terraform-aws-stdapi-ai) declares it for you.
    - Do not substitute a `curl` or `urllib` one-liner: it sends an untrusted `Host` and is answered with `400` as soon as [`TRUSTED_HOSTS`](operations_configuration.md#trusted-hosts) is set.

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

??? failure "A request's `service_tier`, guardrail or model parameters are ignored"
    Some configuration reaches the model that the client did not send, or the value the client sent is not the one applied. Two layers of server-side configuration sit behind every request, and both are deliberate.

    - The model name may be an alias carrying its own configuration — check the entry in [`MODEL_ALIASES`](operations_configuration.md#model-aliases-configuration) for that name. Requests naming it get its service tier, guardrail, metadata and model parameters; requests naming the target model directly do not.
    - The reverse also happens: alias and server-wide configuration apply to models served through Amazon Bedrock's Converse and InvokeModel operations. A model served through Amazon Bedrock Mantle applies the request's own values only, so a configured service tier, metadata or model parameters are ignored there by design — see [the scope note](operations_configuration.md#model-aliases-configuration).
    - A request value is discarded on purpose when its override setting is disabled: [`AWS_BEDROCK_ALLOW_SERVICE_TIER_OVERRIDE`](operations_configuration.md#aws-bedrock-allow-service-tier-override) for the tier, [`AWS_BEDROCK_ALLOW_GUARDRAIL_OVERRIDE`](operations_configuration.md#aws-bedrock-allow-guardrail-override) for the guardrail headers.
    - Otherwise the value comes from the server-wide setting for that field — [`DEFAULT_MODEL_SERVICE_TIERS`](operations_configuration.md#default-model-service-tiers), [`DEFAULT_MODEL_PARAMS`](operations_configuration.md#default-model-params) or [`AWS_BEDROCK_GUARDRAIL_IDENTIFIER`](operations_configuration.md#aws-bedrock-guardrail-identifier). The order is always the request, then the alias, then the setting.

??? failure "Startup fails with a validation error naming a model alias"
    An alias in [`MODEL_ALIASES`](operations_configuration.md#model-aliases-configuration) maps to an object that is not a valid alias configuration, so the server refuses to start rather than ignore it.

    - `Extra inputs are not permitted` names a field that does not exist — check its spelling against the [alias fields](operations_configuration.md#model-aliases-configuration).
    - `Field required` on `model` means the object gives configuration but no target model.
    - A guardrail needs both `guardrail_id` and `guardrail_version`.
    - An alias that only maps a name to a model stays a plain string: `{"my-model": "amazon.nova-lite-v1:0"}`.

??? failure "`400 Bad Request` — text-to-speech rejects a long `input`"
    The message states the length the server accepts (`'input' is limited to 3,000 characters…`, or 20,000 with a generative voice). A generative voice speaks up to 20,000 characters unaided; every other voice, and longer generative input, is synthesized into an S3 bucket co-located with the region serving the request, and none is configured there.

    - Set [`AWS_S3_BUCKET`](operations_configuration.md#aws-s3-bucket) for the first region of `AWS_BEDROCK_REGIONS`, and an [`AWS_S3_REGIONAL_BUCKETS`](operations_configuration.md#aws-s3-regional-buckets) entry for every other region that may serve speech — the server log names the one that was missing.
    - Grant the task role `polly:StartSpeechSynthesisStream` (generative voices beyond 3,000 characters), `polly:StartSpeechSynthesisTask`, `polly:GetSpeechSynthesisTask`, and S3 read/write/delete on those buckets — see [IAM Permissions](operations_iam_permissions.md#text-to-speech-optional).
    - A generative request that still behaves like the others — rejected without a bucket, or slower than expected with one — is missing `polly:StartSpeechSynthesisStream`; the server log names the failure.
    - Up to 3,000 characters (6,000 including SSML markup) never needs a bucket; the limits and the expected latency are in [Long Input](api_openai_audio_speech.md#long-input).

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
    Either an attachment exceeds what the chosen model reads, the application-level file-size cap, or an edge control rejected the request.

    - `Attachments larger than … are not available on the current server` means the attachment is too large to travel inside the request and there is nowhere to stage it: no region able to serve that model has an S3 bucket. Set [`AWS_S3_BUCKET`](operations_configuration.md#aws-s3-bucket) for the first region of `AWS_BEDROCK_REGIONS` and an [`AWS_S3_REGIONAL_BUCKETS`](operations_configuration.md#aws-s3-regional-buckets) entry for every other region that may serve the model — the server log names the one that was missing. The same request succeeds unchanged once a bucket exists.
    - `An attached file is too large: this model accepts at most … bytes per file` means that model only reads attachments sent inside the request. Send a smaller file, or choose a model that reads attachments from storage — see [Attachment Size](features.md#attachment-size).
    - `The attached files are too large: this model accepts at most … bytes of attachments per request` means the same, for the request as a whole: each file fits on its own, but their total does not. Split them across several requests, or choose a model that reads attachments from storage.
    - Check [`MAX_INPUT_FILE_SIZE`](operations_configuration.md#max-input-file-size) — it caps the bytes of any single file loaded into memory for model input; disabled by default, so if it's set and the error appears, raise it or reduce the input size.
    - If the deployment sits behind the Terraform module's WAF (`alb_waf_enabled=true`), check for a `SizeConstraintStatement` rule on the request body — see [Request Size & Resource Limits](operations_authentication_security.md#request-size-resource-limits).
    - If fronted by Amazon API Gateway instead of an ALB, remember its hard 10 MB payload limit.

??? failure "Web search returns nothing, stale results, or `external_web_access` is rejected"
    The built-in [web search tool](api_openai_responses.md#openai-gpt-web-search) is gated by both an IAM permission and a server setting, and each failure looks different.

    - **The model answers from its training data and says it could not search**: the task role is missing `bedrock-websearch:InvokeSearch` / `bedrock-websearch:InvokeFetch`. Add them — see [Web Search IAM](operations_iam_permissions.md#web-search-iam). The request itself still succeeds, so this shows up as a weak answer rather than an error.
    - **`400` on `external_web_access`**: the request asked for a value the server does not allow. By default searches stay inside the AWS boundary; set [`AWS_BEDROCK_EXTERNAL_WEB_ACCESS`](operations_configuration.md#bedrock-external-web-access) to change what the server does, or [`AWS_BEDROCK_ALLOW_EXTERNAL_WEB_ACCESS_OVERRIDE`](operations_configuration.md#bedrock-allow-external-web-access-override) to let requests choose.
    - **External web access was enabled but results still look cached**: `bedrock-websearch:ExternalWebAccess` is missing from the task role, so the search falls back to the Amazon Bedrock web index.
    - **The tool is rejected outright**: web search is served on `/v1/responses` for the OpenAI GPT-5.x family, in `us-east-1`, `us-east-2` and `us-west-2`. On `/v1/messages` and `/v1/chat/completions` it is not available for these models.

??? failure "`503`/`504`/`408` — request times out mid-stream on long generations"
    A slow or hung generation exceeded a timeout somewhere between the model and the client. The status code tells you where: `503` comes from stdapi.ai's own gateway timeout; `504`/`408` come from an edge or proxy timeout in front of it.

    - **`503` from the gateway**: Check [`AI_RESPONSE_TIMEOUT`](operations_configuration.md#ai-response-timeout) — it closes stalled upstream model connections; raise it for workloads with long-running generations. The request is **not** retried in another region: the model already ran and AWS bills it either way, so a failover would pay twice for the same generation.
    - **`504`/`408` from the edge/proxy**: Check the Terraform module's `alb_idle_timeout` (default: 3600 s) — if you lowered it, or front the deployment with your own load balancer or reverse proxy at a shorter idle timeout, streaming responses can be cut off mid-flight before the gateway's own timeout fires. See [ALB Resilience](operations_resilience.md#alb-resilience).

??? failure "Every request fails after enabling per-user cost attribution"
    Model calls run under a session of [`AWS_BEDROCK_USER_ROLE_ARN`](operations_configuration.md#aws-bedrock-user-role-arn), and a session that cannot be opened fails the request rather than silently falling back to the server's identity. The server also reports this at startup, in the `server_warnings` field of its `start` log event. Five causes, in order of likelihood:

    - **The trust policy allows only `sts:AssumeRole`.** Tagging the session is a separate action: add `sts:TagSession` to both the trust policy of the end user role and the server's own policy — see [Per-User Cost Attribution IAM](operations_iam_permissions.md#per-user-cost-attribution). Setting [`AWS_BEDROCK_USER_ROLE_TAG_KEY`](operations_configuration.md#aws-bedrock-user-role-tag-key) to null removes the need for it, at the cost of Cost Explorer grouping.
    - **The role was just created.** A new or newly-edited trust policy takes a few seconds to propagate; a task started immediately after logs the startup warning and recovers on its own.
    - **`403` on every request**: the end user role lacks `bedrock:InvokeModel` or `bedrock:InvokeModelWithResponseStream`, or its `Resource` list misses an ARN form requests actually reach. A cross-region inference profile also needs `arn:aws:bedrock:*::foundation-model/...` for every Region it routes to, and an application inference profile, a prompt router or a prompt ARN each has to be named in its own right — see [Per-User Cost Attribution IAM](operations_iam_permissions.md#per-user-cost-attribution).
    - **`403` once a guardrail is configured**: a guardrail applied during an invocation — [`AWS_BEDROCK_GUARDRAIL_IDENTIFIER`](operations_configuration.md#aws-bedrock-guardrail-identifier), a model alias carrying one, or a request-level `moderation` parameter — is evaluated as part of the call the end user signed, so the end user role needs `bedrock:ApplyGuardrail` on the guardrail ARN as well.
    - **`400` naming `safety_identifier`**: [`AWS_BEDROCK_USER_ROLE_REQUIRE_IDENTITY`](operations_configuration.md#aws-bedrock-user-role-require-identity) is enabled and the client sends no end user identifier. Either have the client send one, or disable that setting.

??? failure "All Bedrock spend still lands on one identity"
    [Per-user attribution](operations_cost_management.md#per-user-attribution) reaches the AWS bill through two AWS-side steps that are easy to miss, and neither is instant:

    - **The Cost and Usage Report export must include caller identity.** Create a Data Exports CUR 2.0 export with *Include caller identity (IAM principal) allocation data* enabled; an existing export cannot be changed and must be re-created. The identity then appears in `line_item_iam_principal` as `assumed-role/<role>/<session>`.
    - **The session tag must be activated as a cost allocation tag**, in the AWS Billing console under **Cost allocation tags**, filtered by type **IAM principal**. It is only listed there after that identity has made at least one call, and takes up to 24 hours to appear in Cost Explorer.
    - **Requests that identify no end user are billed to the server**, by design. The request log's `aws_role_session_name` field is absent on exactly those requests — use it to find the clients that send no identifier, then enable [`AWS_BEDROCK_USER_ROLE_REQUIRE_IDENTITY`](operations_configuration.md#aws-bedrock-user-role-require-identity).
    - **Only model invocations are attributed.** Video generation, guardrail evaluations, speech, transcription and translation stay on the server's own identity.

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

??? failure "A valid Amazon Cognito token is rejected with 401"
    The response body is always the same opaque `Unauthorized`; the check that failed is in the server's request log (`error_detail`). Work through the checks in order:

    - **Wrong app client**: the token's app client must be listed in [`AWS_COGNITO_CLIENT_IDS`](operations_configuration.md#aws-cognito-client-ids).
    - **Wrong token**: an identity token is rejected unless [`AWS_COGNITO_ACCEPT_ID_TOKEN`](operations_configuration.md#aws-cognito-accept-id-token) is enabled. Send the **access** token.
    - **Wrong issuer**: pools on the Essentials and Plus tiers can issue `https://issuer-cognito-idp.<region>.amazonaws.com/...`. Set [`AWS_COGNITO_ISSUER_TYPE`](operations_configuration.md#aws-cognito-issuer-type) to `updated` for those, and confirm the pool ID matches the pool that minted the token.
    - **Missing scope**: a token obtained by signing in with a username and password carries only `aws.cognito.signin.user.admin`. Clear [`AWS_COGNITO_REQUIRED_SCOPES`](operations_configuration.md#aws-cognito-required-scopes), or have clients obtain tokens from the pool's OAuth 2.0 token endpoint.
    - **Expired token**: tokens are accepted up to one minute past expiry only. Refresh the token, and check the container clock if expiry errors are constant.
    - See [Authentication & Security → Amazon Cognito User Pool Tokens](operations_authentication_security.md#amazon-cognito-user-pool-tokens).

??? failure "The server does not start after enabling authentication"
    A half-applied credential configuration is refused rather than accepted, so the deployment never runs unauthenticated by accident. The startup log's `error_detail` names the exact rule:

    - **Missing allowlist**: [`AWS_COGNITO_CLIENT_IDS`](operations_configuration.md#aws-cognito-client-ids) is required with a pool.
    - **Setting without a pool**: any other `AWS_COGNITO_*` variable requires [`AWS_COGNITO_USER_POOL_ID`](operations_configuration.md#aws-cognito-user-pool-id).
    - **Mode conflict**: [`AUTHENTICATION_MODE`](operations_configuration.md#authentication-mode) must not demand a method that is unconfigured, nor ignore one that is configured — use `any` to accept both.
    - **Empty API key**: the SSM parameter or Secrets Manager secret named by [`API_KEY_SSM_PARAMETER`](operations_configuration.md#api-key-ssm) or [`API_KEY_SECRETSMANAGER_SECRET`](operations_configuration.md#api-key-secretsmanager-secret) exists but holds an empty value — populate it, or unset the setting to run without an API key deliberately.
    - **Signing keys unreachable**: the pool's public keys are read at startup over HTTPS. Check the pool ID, and that the task can reach the internet or a suitable endpoint for outbound HTTPS.

??? failure "An MCP client or agent cannot discover how to authenticate"
    Discovery is off until it is configured, and a client that finds nothing falls back to asking the user for a key.

    - **Nothing published**: `GET /.well-known/oauth-protected-resource` answering `404` means [`OAUTH_RESOURCE_IDENTIFIER`](operations_configuration.md#oauth-resource-identifier) is unset. That is the default, not a fault — set it, together with [`OAUTH_AUTHORIZATION_SERVERS`](operations_configuration.md#oauth-authorization-servers), to turn discovery on.
    - **Nothing in the challenge**: a `401` carrying a bare `WWW-Authenticate: Bearer` means the same thing. Once configured it also carries `resource_metadata="…"` and, with [`OAUTH_SCOPES_SUPPORTED`](operations_configuration.md#oauth-scopes-supported), `scope="…"`.
    - **Browser-hosted client**: a client running in a page reads the document cross-origin and needs its origin in [`CORS_ALLOW_ORIGINS`](operations_configuration.md#cors-allow-origins).
    - **The client asks for `/.well-known/openid-configuration`**: stdapi.ai is a resource server and deliberately does not serve it; the client should follow `authorization_servers` to the issuer's own document.
    - See [Authentication & Security → Authentication Discovery for Agents](operations_authentication_security.md#authentication-discovery-for-agents).

??? failure "A client reports the protected resource does not match the expected URL"
    Clients compare the published `resource` against the URL they dialled character by character, so any difference aborts the flow.

    - **Scheme**: set [`OAUTH_RESOURCE_IDENTIFIER`](operations_configuration.md#oauth-resource-identifier) to `https://…` when clients reach the deployment over TLS, even though the container itself listens on plain HTTP behind the load balancer.
    - **Host**: it must be the public hostname clients use, not the internal service or task address.
    - **Port**: an explicit `:443` on the client's URL does not match an identifier without one, and vice versa. Drop the default port on both sides.
    - **Path**: the identifier is an origin. Do not append `/mcp`, `/v1`, or a trailing slash — one document at the root already covers every surface of the deployment.

??? failure "The server does not start after configuring authentication discovery"
    The three settings describe one document, so an incomplete set is refused rather than published half-formed. The startup log's `error_detail` names the rule:

    - **No authorization server**: [`OAUTH_AUTHORIZATION_SERVERS`](operations_configuration.md#oauth-authorization-servers) is required with [`OAUTH_RESOURCE_IDENTIFIER`](operations_configuration.md#oauth-resource-identifier); a document naming none leaves a client unable to obtain a token.
    - **Setting without an identifier**: the authorization servers and the scopes describe a document that is not published without [`OAUTH_RESOURCE_IDENTIFIER`](operations_configuration.md#oauth-resource-identifier).
    - **Malformed value**: the identifier is an origin with no path or query, each issuer is an `https` URL with no query or fragment, and a scope carries no space or quote.

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
