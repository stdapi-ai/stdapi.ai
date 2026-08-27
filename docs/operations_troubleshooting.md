---
title: Troubleshooting - Common stdapi.ai deployment issues
description: Fixes for the most common errors encountered when deploying and running stdapi.ai - Terraform failures, 400/401/403/404/429/503 responses, Bedrock throttling, IAM permission errors, S3 bucket errors, VPC connectivity, and more.
keywords: stdapi.ai troubleshooting, AWS Bedrock errors, Terraform apply failed, 503 ECS service, 401 API key, 403 permission IAM, AccessDeniedException, 404 model not found, model does not exist or you do not have access to it, model_not_found, gpt-4o 404, gpt-4o not found, gpt-3.5-turbo not found, text-embedding-3-small 404, dall-e-3 model not found, OpenAI model name rejected, which models are available, list available models, ThrottlingException Bedrock, S3 bucket region, ElastiCache capacity, VPC endpoint timeout, podman SELinux, ECONNREFUSED IPv6 service discovery, drop_params ValidationException, ECS task unhealthy restart loop, container healthCheck command, text to speech input too long, speech input limited to 3000 characters, speech input limited to 20000 characters, generative voice long input, StartSpeechSynthesisStream AccessDenied, service_tier ignored, guardrail ignored, model parameters ignored, model alias configuration, MODEL_ALIASES validation error, extra inputs are not permitted alias, 413 attachment too large, large image rejected, large PDF rejected, attachments larger than not available, attached files are too large per request, attachment total too large, AWS_S3_REGIONAL_BUCKETS attachment staging, web search returns no results, no error when web search permission missing, external_web_access rejected, external_web_access not available with this model, web search allowed_domains rejected, web search user_location rejected, vector store search responses, batch result files never expire, batch output_expires_after, batch results file 404 after expiry, bedrock-websearch InvokeSearch AccessDenied, web search stale cached results, mcp_servers ignored, mcp_toolset dropped, MCP tools never run, MCP connector not supported, valid Cognito token rejected 401, Cognito JWT unauthorized, aws_cognito_client_ids required, server will not start after enabling Cognito, server will not start after enabling authentication, API key secret is empty, API_KEY_SSM_PARAMETER empty value, AUTHENTICATION_MODE method not enabled, issuer-cognito-idp issuer mismatch, aws.cognito.signin.user.admin scope missing, WWW-Authenticate Bearer challenge, MCP client cannot discover authentication, oauth-protected-resource 404, protected resource does not match expected, OAUTH_RESOURCE_IDENTIFIER invalid, openid-configuration not served, OAUTH_AUTHORIZATION_SERVERS must name the issuer, published issuer does not match user pool, authorization server derived from Cognito, response format srt not available, verbose_json not available transcription, streamed transcription delivers nothing until the end, transcription stream hangs, stream=true events arrive all at once, transcript.text.delta arrives at the end, AWS_TRANSCRIBE_STREAM_LANGUAGES, MaxSpeakerLabels streamed transcription not phrase by phrase, timestamp_granularities rejected, 503 transcription not available, transcription bucket missing, 503 url response format not available, image url response bucket missing, audio too long 10 minutes, nova-2-sonic model not found, InvokeModelWithBidirectionalStream AccessDenied, 503 live conversation not available, tenant API key 503, sk-std key rejected, TENANT_API_KEYS, tenant key not in Parameter Store, revoked tenant key still accepted, TENANT_KEY_CACHE_SECONDS, tenant model 404, models_allow empty, tenant AWS credentials, TENANT_AWS_CREDENTIALS, aws_role_arn refused 503, tenant role 403, ExternalId mismatch, sts:AssumeRole tenant role AccessDenied, tenant credential could not be used, tenant account does not have access to this model, per-user cost attribution, sts:TagSession AccessDenied, AWS_BEDROCK_USER_ROLE_ARN, all Bedrock spend on one identity, line_item_iam_principal missing, cost allocation tag IAM principal, requests fail after enabling per-user roles, end user role ApplyGuardrail AccessDenied, 403 guardrail per-user cost attribution, end user role missing model ARN form, 503 speech to speech request fails after a few seconds, real-time audio session timeout, AWS_CONNECT_TIMEOUT generative voice, conversation items missing, conversation 404 after 30 days, cannot add items to conversation, bedrock UpdateSession AccessDenied, conversation metadata update 503, streamed response conversation items delayed, batch stays validating for minutes, batch rejected fewer than 100 requests, batch minimum 100 requests per model, minimum number of records per batch inference job quota, batch model marked legacy 30 days, 503 batch creation refused ValidationException, search_models batch false, model missing from batch=true, batch flag absent COST_TRACKING, batch role not configured, 503 batch API disabled, AWS_BEDROCK_BATCH_ROLE_ARN, iam:PassRole bedrock.amazonaws.com, CreateModelInvocationJob AccessDenied, 503 batch creation denied, batch line failed tool use not supported, batch results out of order, batch output_file_id missing, batch cached_tokens 0, prompt caching ignored in a batch, cache_control dropped batch, prompt_cache_key batch no effect, batch cache_read_input_tokens zero, vector store file stays in_progress, vector store unsupported_file, attaching a PDF fails unsupported_file, search returns nothing after attaching, 503 vector stores disabled, AWS_S3_VECTORS_BUCKET, vector bucket missing, vector bucket wrong region, s3vectors AccessDenied, 503 vector store operation denied, vector store 409 conflict, vs_kb vector store 404, knowledge base vector store not found, AWS_BEDROCK_KNOWLEDGE_BASE_IDS, knowledge base not allowlisted, bedrock GetKnowledgeBase AccessDenied, 503 attaching a file to a knowledge base store, knowledge base multiple data sources, dataSourceId ambiguous, 400 files cannot be attached to this vector store, knowledge base data source is not a custom one, attach refused knowledge base store, cannot delete a file a search returned, kbdoc file cannot be removed, 400 cannot delete knowledge base vector store, chunking_strategy rejected knowledge base, score_threshold rejected knowledge base, file_batches rejected knowledge base, search query too long knowledge base, realtime websocket upgrade 404, WS /v1/realtime not found, POST /v1/realtime/calls 404, realtime calls endpoint not found, realtime WebRTC not supported, realtime SIP not supported, WebRTC transport realtime API, SIP transport realtime API, SDP offer realtime 404, ALB does not carry UDP, WebRTC media ingress NLB, realtime session closes after 8 minutes, session_expired close code, realtime session closes mid-conversation, realtime session idle timeout, ALBRequestCountPerTarget realtime autoscaling, 403 NoUserAgent_HEADER websocket, WAF blocks websocket client no user agent, ephemeral client secret rejected, ek_ secret invalid on other instance, REALTIME_CLIENT_SECRET_KEY, realtime client secret multi-instance, 503 feature_unavailable AccessDeniedException, IAM permission denied 503 not available on the current server, model access not enabled 503, 403 permission_error end user role, access denied reading the input s3 URI, 400 input_access_denied, s3 input AccessDenied not 503, accepted bucket GetObject denied, Mantle models missing from catalog, bedrock-mantle unreachable, api.aws endpoint blocked, HTTPS_PROXY not honored, proxy environment variables ignored, NO_PROXY api.aws, com.amazonaws bedrock-mantle VPC endpoint, netrc credentials sent to AWS endpoints, bedrock_mantle_regions_without_endpoint, Mantle unreachable in every region, MantleError service is temporarily unavailable, bedrock-mantle Domain name not found, ClientConnectorDNSError bedrock-mantle, NXDOMAIN bedrock-mantle api.aws, region does not offer Bedrock Mantle, AWS_BEDROCK_MANTLE_REGIONS default, Mantle models missing in one region, server_start_time_ms slow startup, startup takes 30 seconds, ECS container metadata endpoint answered after attempts, ECS task metadata slow startup, unreachable region slows startup, AWS_CONNECT_TIMEOUT startup time, abandoned_background_tasks, background work lost on deploy, vector store file stuck in_progress after deploy, cleanup skipped on shutdown, SHUTDOWN_DRAIN_TIMEOUT, container stop timeout SIGKILL, graceful shutdown drain, work abandoned on scale-in, vector store file failed indexing was interrupted, attach the file again to index it, vector store indexing waits its turn, AWS_SQS_VECTOR_STORE_QUEUE_URL, vector store indexing queue, durable vector store indexing, sqs:SendMessage AccessDenied, must be an Amazon SQS queue URL, must name a standard queue, FIFO queue refused indexing, indexing queue has no dead-letter queue, vector store indexing job abandoned, indexing not picked up after deploy, another server finishes the indexing, aws_bedrock_mantle_preferred_models is incompatible with Amazon Bedrock Guardrails, server will not start guardrail Mantle, guardrail preferred models startup error, AWS_BEDROCK_MANTLE_PREFERRED_MODELS empty, GPT-5.6 more expensive, gpt-5.6 price increased 10%, GPT-5.6 billed under Bedrock Mantle, global cross-region discount lost, input_tokens 400 gpt-5.6, token counting rejected GPT-5.6, restore bedrock-runtime routing dual-homed, Marketplace model endpoint missing from model list, AWS_BEDROCK_MARKETPLACE_ENDPOINTS_ENABLED, AWS_BEDROCK_MARKETPLACE_ENDPOINT_REGIONS, marketplace model endpoint not in service, marketplace model endpoint deployment not registered, marketplace model endpoint requests refused, marketplace/model-endpoint ARN missing from policy, foundation-model wildcard denies marketplace endpoint, marketplace model endpoint reports usage but no cost, SageMaker hosting instance-hour billing, no scale to zero marketplace endpoint, two marketplace endpoints same listing one model missing, wildcard model pattern ambiguous, 400 model pattern matches several models released same date, wildcard model pattern skips a listed model, model release date unknown never selected by pattern, glob model name 400, model pattern fewer than three characters, bare asterisk model not accepted, organization usage endpoints 503, /v1/organization/usage 503, /v1/organization/costs 503, usage API not enabled, USAGE_API, GET /v1/usage not served, retired usage endpoint, usage endpoints empty buckets, no usage reported, organization costs empty, per-endpoint usage buckets missing, Operation dimension missing, CLOUDWATCH_METRICS_REGION, usage reports one region only, usage endpoints 403, tenant API key refused usage endpoints, USAGE_API_ADMIN_SCOPES, usage admin scope missing, usage query too large, too many metric series, USAGE_API_MAX_METRICS, USAGE_API_MAX_RANGE_DAYS, usage range too long, bucket_width 1m refused, one-minute buckets 15 days, num_model_requests, Ollama client shows no models, Ollama connection empty model list, ollama list size 0, ollama pull does nothing, Ollama 404 llama3.2, Ollama tokens per second 0, eval_duration missing, load_duration missing, Ollama is running not returned, /api/version probe, Ollama thinking empty, Ollama 401 API key, Unable to locate credentials docker run, ~/.aws mount permission denied container, container runs as uid 65532, nonroot 65532 cannot read credentials, docker run --user HOME /home/nonroot, userns keep-id uid 65532, marketplace image no shell, community image has a shell, docker run sh not found image, entrypoint python command override ignored
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

---

## :material-docker: Local Docker / Podman

??? failure "Podman volume mount fails on Fedora/RHEL with SELinux (local Docker)"
    SELinux blocks container access to `~/.aws` without a relabel.

    - Add `:z` (or `:Z` for exclusive use) to the volume and `--userns=keep-id:uid=65532,gid=65532`:
      ```bash
      podman run --rm -p 8000:8000 \
        --userns=keep-id:uid=65532,gid=65532 \
        -v ~/.aws:/home/nonroot/.aws:ro,z \
        -e AWS_BEDROCK_REGIONS=us-east-1,us-west-2 \
        -e ENABLE_DOCS=true \
        ghcr.io/stdapi-ai/stdapi.ai-community:latest
      ```
    - See [Local Development](operations_getting_started_local.md#run-it) for the full run command.

??? failure "`Unable to locate credentials` with `~/.aws` mounted into the container"
    The image runs as the unprivileged user `nonroot`, uid/gid **65532**, while
    the files under `~/.aws` belong to your own account and are typically
    readable by it alone. The mount succeeds, every file inside it is
    unreadable, and the server exits as if no credentials had been given.

    - Run the container as yourself: `--user "$(id -u):$(id -g)" -e HOME=/home/nonroot`. The `HOME` value matters — without it the AWS SDK looks for `.aws` in the wrong place. Run this from your own shell, never under `sudo` — under `sudo`, `$(id -u):$(id -g)` resolves to `0:0` and the container silently runs as root instead of `nonroot`.
    - On rootless Podman use `--userns=keep-id:uid=65532,gid=65532` instead, which maps your account onto the image's user.
    - Or skip the mount entirely and pass `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN` as environment variables — see [Local Development](operations_getting_started_local.md#run-it).

??? failure "A `docker run` command override stopped working after upgrading the image"
    Both images declare `python` as their entry point, so anything after the
    image name is passed to the interpreter rather than run as a program. The
    Marketplace image additionally ships **no shell and no package manager**, so
    `sh`, `bash` and any install command have nothing to run there; the
    community image is Debian-based and still has them.

    - Pass Python arguments directly: `docker run ... ghcr.io/stdapi-ai/stdapi.ai-community:latest -c "import stdapi; print(stdapi)"`.
    - To run another binary in the image, name it explicitly: `--entrypoint /usr/bin/ffmpeg`.
    - Server options are environment variables, not command arguments — see [Configuration](operations_configuration.md).

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

??? failure "`503` saying the feature is not available on the current server — IAM permission denied on an AWS call"
    The gateway reached the AWS service, but the **ECS task role** (or your local AWS credentials) lacks permission for the action it used. AWS returns `AccessDeniedException`, which stdapi.ai answers as a feature this deployment cannot run: HTTP `503`, error code `feature_unavailable`, and the same generic message whatever is missing. This is an IAM misconfiguration, **not** a client API-key problem, and the client is deliberately told nothing about it — **the server log names the operation, the model and the permission AWS refused**, under `error_detail`.

    - Confirm the task role grants `bedrock:InvokeModel` and `bedrock:InvokeModelWithResponseStream` (and the `bedrock:Converse*` actions) for the target model ARNs.
    - For models invoked through an inference profile, allow both the profile ARN and the underlying foundation-model ARNs in the policy.
    - **By a distance the most likely cause for a [Marketplace model endpoint](operations_cost_management.md#bedrock-marketplace-model-endpoints) that is listed but every call to it is refused**: a policy scoped to `foundation-model/*` and `inference-profile/*` — copied from an example that predates Marketplace model endpoints — denies every call to one. It needs `arn:aws:bedrock:*:ACCOUNT_ID:marketplace/model-endpoint/*` as well, on the task role and, when [per-user cost attribution](operations_cost_management.md#per-user-attribution) is enabled, on the end user role too — see [Bedrock Marketplace Model Endpoints IAM](operations_iam_permissions.md#bedrock-marketplace-endpoints-iam).
    - Some models require one-time activation in the **Bedrock console → Model access** page before they can be invoked.
    - Audio, embeddings, and file features need permissions for the relevant services (Polly, Transcribe, Translate, Comprehend, S3) — see [Configuration → IAM Permissions](operations_configuration.md#iam-permissions) for the full IAM reference.
    - A `403 permission_error` on a model call means the opposite: [per-user cost attribution](operations_cost_management.md#per-user-attribution) is enabled and the **end user's** role was denied, so the policy to fix is that role's, not the task role's.
    - A `400 invalid_request_error` naming an `s3://` input also means the opposite: the denial is on the object *the request pointed at*, in one of the external buckets declared in [`AWS_S3_ACCEPTED_BUCKETS`](operations_configuration.md#aws-s3-accepted-buckets). Only the caller can fix that one — a wrong key, or a bucket policy that does not grant this deployment's role `s3:GetObject` on the object. Objects in the deployment's own buckets keep the `503` above.

??? failure "`401 Unauthorized` — AWS credentials invalid or expired (often local Docker)"
    stdapi.ai's **own** AWS credentials are missing, invalid, or expired — AWS returns `UnrecognizedClientException`, `InvalidSignatureException`, or `ExpiredTokenException`, which stdapi.ai maps to HTTP `401` with error type `authentication_error`. This is distinct from the client-facing API-key `401` above (which concerns your `Authorization` / `X-API-Key` header).

    - Locally: refresh with `aws sso login` (or update `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`) and restart the container.
    - On ECS: confirm the task is assuming its IAM role rather than relying on stale static keys.

??? failure "`404 Not Found` — `The model ... does not exist or you do not have access to it`"
    The name in the request is not one this deployment serves. Most names need no change when you adopt the gateway: the Anthropic, OpenAI and Cohere models Bedrock serves are published under their providers' own names as well as their Bedrock IDs, derived mechanically from the ID, so `anthropic.claude-opus-5` also answers to `claude-opus-5`, `openai.gpt-oss-120b-1:0` to `gpt-oss-120b` and `cohere.rerank-v3-5:0` to `rerank-v3.5`. This `404` is what a name that *does* differ looks like: one your application hard-codes for a model this deployment does not serve (`gpt-4o`, `gpt-3.5-turbo`, `text-embedding-3-small`, `dall-e-3`) resolves only if you map it yourself with [`MODEL_ALIASES`](operations_configuration.md#model-aliases). Nothing is substituted on your behalf, because a lookalike would serve a different model than the one you asked for.

    - List what this deployment actually serves: `GET /search_models` (the default model-discovery endpoint). Filter by capability with query parameters — e.g. `GET /search_models?input_modalities=IMAGE&route=/v1/chat/completions` returns only vision-capable chat models. See the [Search Models API](api_search_models.md) reference.
    - `GET /v1/models` is also available for strict OpenAI SDK compatibility (lighter payload, no capability metadata).
    - Confirm the pipeline itself with a low-friction model: `amazon.nova-micro-v1:0` (available in all standard Bedrock regions).
    - **Keep the name your application already sends** by mapping it onto a served model with [`MODEL_ALIASES`](operations_configuration.md#model-aliases) — an alias only ever points at a model this deployment serves, so what it resolves to is your choice, not a guess.
    - Try the name with and without its `anthropic.` / `openai.` / `cohere.` prefix: both forms resolve for those three families, so `anthropic.claude-fable-5` and `claude-fable-5` reach the same model, as do `openai.gpt-oss-120b-1:0` and `gpt-oss-120b`. Cohere additionally spells its versions with a dot, so `cohere.embed-english-v3` answers to `embed-english-v3.0` and `cohere.rerank-v3-5:0` to `rerank-v3.5` — not to `embed-english-v3` or `rerank-v3-5`. Models from other providers are served under their Bedrock IDs only, so name the ID or alias it. A model version Bedrock has retired needs a current one, whichever form you use.
    - Only if the model *is* one Bedrock serves and it is still missing: verify `AWS_BEDROCK_REGIONS` includes a region that offers it — see the [Bedrock model availability table](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html). Adding regions never makes a name Bedrock does not serve resolve.

??? failure "`400 Bad Request` — a wildcard model pattern is refused as ambiguous"
    Two or more of its matches were released on the same date, so the server refuses to guess which one you meant, rather than spend your money on a model you never named — sibling models released together are often priced differently. The message names the tied models. `openai.gpt-5.6-*` (Sol, Terra, Luna) and `stability.*` on the image routes are real examples of ties.

    - Name the model you meant explicitly, or narrow the pattern so it matches only one of the tied models.
    - See everything a pattern matches, newest first, before relying on it: `GET /search_models?model=<pattern>` — see [Search Models](api_search_models.md#query-parameters).
    - See [Model Wildcard Patterns](operations_configuration.md#model-wildcard-patterns) for the full resolution rule.

??? failure "A wildcard model pattern never picks a model that is listed in `/v1/models` or `/search_models`"
    A pattern only selects from what it can order by release date, and four kinds of match are skipped even when they are listed and match the glob:

    - **A model with no known release date.** The server cannot order it, so it is never selected by a pattern, whatever else it matches — a property of that model, not a bug in the pattern or a gap in a particular backend's catalogue. It also never makes an otherwise-unique match ambiguous.
    - **A legacy model**, one [`AWS_BEDROCK_DEPRECATED_MODEL_FALLBACK`](operations_configuration.md#bedrock-deprecated-model-fallback) covers, or one whose first use would open a paid Marketplace subscription.

    Name any of these explicitly and it still resolves exactly as it does today — only pattern resolution skips them. `GET /search_models?model=<pattern>` returns the whole match set a pattern draws from, newest first with unknown-release-date models last, so you can see what it will do before it surprises you — see [Model Wildcard Patterns](operations_configuration.md#model-wildcard-patterns).

??? failure "A model that has been removed still appears in `/v1/models`, or a new one takes minutes to show up"
    Expected, within a bounded window. The model list is discovered from Amazon Bedrock and kept for [`MODEL_CACHE_SECONDS`](operations_configuration.md#model-cache-seconds) (15 minutes by default); once it expires the request that notices is answered from the list in hand and the refresh runs behind it, so no request pays for the discovery pass. A model AWS has withdrawn — or that this account has lost access to — can therefore stay listed until that refresh lands, and a request naming it is accepted and then fails at the backend with the same `404 model_not_found` as any unknown model. See [Model List Refresh](operations_resilience.md#model-list-refresh).

    - **A model that is new** is never served from an expired list: naming one the list does not know makes the server refresh before it answers, so it is usable as soon as it exists.
    - **Shorten the window** with a lower `MODEL_CACHE_SECONDS`, at the cost of more discovery calls.
    - **Remove it entirely** with `MODEL_CACHE_MAX_STALE_SECONDS=0`, which makes every expiry refresh synchronously — the freshest and the slowest setting.
    - Only if the list is *very* old, or a model that exists never appears, is this a fault rather than the window — see the next entry.

??? failure "The model list never changes, or keeps advertising models that are gone"
    A refresh that keeps failing leaves the list frozen. It is reported in the server log rather than in a response, because the request that triggered it was answered from the list already in memory.

    - Look for `Refreshing the model list from AWS Bedrock failed` in the server log. It is logged at `warning`, and at `error` once the list is more than two [`MODEL_CACHE_SECONDS`](operations_configuration.md#model-cache-seconds) old — the signal that the list is drifting toward the ceiling.
    - The usual causes are a revoked `bedrock:ListFoundationModels` or `bedrock:GetFoundationModelAvailability` permission and a prolonged outage in every configured region. Check [Bedrock IAM](operations_iam_permissions.md#bedrock-iam) first.
    - Beyond [`MODEL_CACHE_MAX_STALE_SECONDS`](operations_configuration.md#model-cache-max-stale-seconds) (24 hours by default) requests stop being answered from the frozen list and wait for a refresh instead, so a deployment in this state eventually surfaces the real error to clients as a `503` rather than serving a list it cannot confirm.

??? failure "Enabling `MODEL_CACHE_SHARED` did not make startup any faster"
    A published model list is only read by servers running the **same version, in the same AWS account, with the same `AWS_BEDROCK_*` and `AWS_SAGEMAKER_*` configuration** — every setting under those prefixes, not only the ones discovery reads. Anything else reads as an empty cache and the server discovers the catalogue itself, exactly as it would with the feature off, so changing one of those settings costs the fleet one discovery pass each rather than pointing at a table or permission fault.

    - **During a rolling deployment this is expected**: the new version recognises nothing the old one published, so the first server of each version performs a full discovery pass and publishes for the rest. Changing any `AWS_BEDROCK_*` setting has the same effect, once.
    - **If it never gets faster**, the table itself is the suspect. Every failure is reported in the server log at `WARNING`, naming the DynamoDB action, the table and the setting to fix — a missing `dynamodb:GetItem`/`PutItem`/`Query` on the table ARN is the common one, see [Shared Table IAM](operations_iam_permissions.md#shared-table).
    - **A single-container deployment gains nothing** by design: there is no second server to share with, and the first one still has to discover the catalogue.
    - Setting it without [`AWS_DYNAMODB_TABLE`](operations_configuration.md#aws-dynamodb-table) fails startup with a message naming both settings.

??? failure "A deployed Marketplace model endpoint does not appear in the model list"
    A [Bedrock Marketplace model endpoint](operations_cost_management.md#bedrock-marketplace-model-endpoints) is discovered, never created — the gateway only ever lists an endpoint that already exists in your account. Work through the checks in order:

    - **The feature is off.** [`AWS_BEDROCK_MARKETPLACE_ENDPOINTS_ENABLED`](operations_configuration.md#bedrock-marketplace-endpoints-enabled) defaults to `false`.
    - **The endpoint is in a region the server does not serve.** It must be in [`AWS_BEDROCK_REGIONS`](operations_configuration.md#aws-bedrock-regions), and in [`AWS_BEDROCK_MARKETPLACE_ENDPOINT_REGIONS`](operations_configuration.md#bedrock-marketplace-endpoint-regions) too when that is set. A Marketplace model endpoint has no cross-region form, so the server cannot reach one outside the regions it is configured to call.
    - **The endpoint is not yet in service.** Deployment takes roughly 10–15 minutes; it appears the next time the model cache refreshes, not the instant deployment finishes. An endpoint you *update* — a scale, an instance change — is a different case: SageMaker keeps serving it on the previous configuration throughout, so it stays listed and keeps answering.
    - **Amazon Bedrock could not register the endpoint.** Its own console reports this — check there before assuming a gateway-side gap.
    - **The server role is missing the discovery permissions.** The server reports this itself, in its own log at `WARNING` — see [Bedrock Marketplace Model Endpoints IAM](operations_iam_permissions.md#bedrock-marketplace-endpoints-iam).

    Two endpoints deployed from the same Marketplace listing in the same region publish only one model, because both carry the listing's name. Set [`AWS_BEDROCK_ALLOW_MARKETPLACE_ENDPOINT_ARN`](operations_configuration.md#bedrock-allow-marketplace-endpoint-arn) to let a client name the other by its ARN, or delete the duplicate.

??? failure "A Marketplace model endpoint reports usage but no cost"
    Not a fault. AWS bills a [Bedrock Marketplace model endpoint](operations_cost_management.md#bedrock-marketplace-model-endpoints) by the instance-hour, whether or not it ever serves a request, so there is no per-token rate to report against the requests that use it — nothing is invented in its place. Read the real charge from AWS Cost Explorer or the SageMaker hosting line of your bill.

??? failure "Every Bedrock Mantle model is missing, while the classic Bedrock models are all there"
    The catalog lists the classic Bedrock models and none of the ones served through Amazon Bedrock Mantle, in every configured region at once. A per-region or per-model gap looks different: this is the shape of a network policy that reaches one endpoint and not the other.

    Bedrock Mantle is served from `bedrock-mantle.<region>.api.aws` — a different domain from classic Bedrock's `bedrock-runtime.<region>.amazonaws.com`, so an allowlist, firewall rule, proxy exception or VPC endpoint written for `amazonaws.com` does not cover it.

    - Allow `bedrock-mantle.<region>.api.aws` outbound, for every region in `AWS_BEDROCK_REGIONS`.
    - On a private deployment, create the interface VPC endpoint `com.amazonaws.<region>.bedrock-mantle` with private DNS enabled.
    - Behind a proxy, make sure `HTTPS_PROXY` is set in the task environment and that `NO_PROXY` does not exclude `api.aws`.
    - Confirm the task role carries the Bedrock Mantle permissions — without them the models are simply not listed rather than refused. See [IAM Permissions](operations_iam_permissions.md).
    - Check what was discovered: `GET /search_models` returns every model the server found, with its regions.

    See [Outbound Network Requirements](operations_deploy_advanced.md#outbound-network-requirements) for the full destination list.

    The startup warning names the region, the endpoint address and the exception chain behind the failure, so a blocked route (`ConnectionTimeoutError`), a refused connection (`ConnectionRefusedError`), an intercepting proxy's certificate (`SSLCertVerificationError`) and an unresolvable address (`ClientConnectorDNSError`) are told apart without further instrumentation.

??? failure "Bedrock Mantle models are missing in one region only — `bedrock_mantle_regions_without_endpoint`"
    The startup log lists the region under `bedrock_mantle_regions_without_endpoint` instead of `unreachable_bedrock_regions`. Bedrock Mantle is offered in fewer regions than classic Bedrock, and where it is not offered `bedrock-mantle.<region>.api.aws` has no DNS record at all — nothing to retry, and no network policy to change. See [model availability by endpoint](https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html).

    - Remove the region from [`AWS_BEDROCK_MANTLE_REGIONS`](operations_configuration.md#bedrock-mantle-regions), or unset it to fall back to the regions of `AWS_BEDROCK_REGIONS` that offer Mantle. Classic Bedrock in that region is unaffected either way.
    - If no configured region offers Mantle, the log says so as well and no Mantle model is served — set [`AWS_BEDROCK_MANTLE_ENABLED`](operations_configuration.md#bedrock-mantle-enabled) to `false` to stop the warning.
    - If AWS has since added the region, list it explicitly in `AWS_BEDROCK_MANTLE_REGIONS`: an explicit list is used as given.
    - If the address should resolve because a VPC endpoint provides it, this is instead the private-DNS case covered by *Every Bedrock Mantle model is missing* above.

??? failure "Startup takes tens of seconds — `server_start_time_ms` far above the usual few seconds"
    The `start` log event reports `server_start_time_ms` in the tens of thousands where a healthy deployment reports a few thousand. Startup reads the model catalogs of every configured region, and the ECS task metadata endpoint before them; a destination that never answers is only given up on after a timeout.

    - Read the `server_warnings` of the same event first. `ECS container metadata endpoint answered after N attempts in X s` accounts for that many seconds on its own: the endpoint is served by the ECS agent over the task ENI and answers slowly when the task is CPU-starved at boot. Raise the task CPU, or the `cpu` of the Fargate task definition, so the agent is scheduled promptly.
    - `unreachable_bedrock_regions` and `bedrock_mantle_regions_without_endpoint` each name a region that spent its full timeout budget. Removing the region from `AWS_BEDROCK_REGIONS` or `AWS_BEDROCK_MANTLE_REGIONS` removes the delay.
    - [`AWS_CONNECT_TIMEOUT`](operations_configuration.md#aws-connect-timeout) bounds each connection attempt and the model-catalog fetch that follows it, so lowering it lowers what an unreachable region can cost. It also bounds failover between healthy regions, so keep it above your real inter-region latency.
    - Model discovery is per region and runs in parallel, so the count of regions costs far less than one unreachable region does.

??? failure "A deploy leaves work unfinished — `abandoned_background_tasks` in the `stop` log event"
    Some work is started outside the request that asked for it, so the caller is answered without waiting: temporary file cleanups, vector store file indexing, and the release of live audio sessions. When a task is stopped, the server waits [`SHUTDOWN_DRAIN_TIMEOUT`](operations_configuration.md#shutdown-drain-timeout) seconds for that work, then cancels the rest — and reports the counts as `abandoned_background_tasks` in a `stop` log event raised to `warning`.

    - A vector store file reported `failed` after a deployment, with `last_error` saying the indexing was interrupted, or a temporary object that outlived its request, is what those counts look like from the outside. Attach the file again, or let the object's lifecycle rule expire it. A file is never left `in_progress`: one whose indexing is gone is settled the next time it, its store, or the store's file list is read.
    - Raise [`SHUTDOWN_DRAIN_TIMEOUT`](operations_configuration.md#shutdown-drain-timeout) **and** the container stop timeout together — the container runtime sends `SIGKILL` a fixed delay after the stop signal (30 seconds by default on Amazon ECS), so raising the wait alone changes nothing past that delay.
    - Counts that persist after raising both mean the work itself is long, not that the wait is short: indexing a large file outlasts any stop timeout. Attach large files when a deployment is not in flight, and treat the operation as retryable.
    - The wait is best effort in every deployment: a Spot interruption or a hard kill ends the process regardless, so never rely on it for anything whose completion matters.

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

??? failure "The server refuses to start: `aws_bedrock_mantle_preferred_models is incompatible with Amazon Bedrock Guardrails`"
    A guardrail is configured — [`AWS_BEDROCK_GUARDRAIL_IDENTIFIER`](operations_configuration.md#aws-bedrock-guardrail-identifier) or a [`MODEL_ALIASES`](operations_configuration.md#model-aliases-configuration) entry carrying one — while [`AWS_BEDROCK_MANTLE_PREFERRED_MODELS`](operations_configuration.md#bedrock-mantle-preferred-models) routes models to Amazon Bedrock Mantle, which cannot apply guardrails. Those models would be served unfiltered, so startup stops instead. The setting defaults to `openai.gpt-5.6`, so a guardrailed deployment meets this without having set it.

    - Keep the guardrail: `export AWS_BEDROCK_MANTLE_PREFERRED_MODELS=` (empty). Every dual-homed model, the GPT-5.6 family included, is then served by the classic endpoint under the guardrail — and `web_search` and `code_interpreter` are refused with a `400` for that family.
    - Keep Mantle routing: remove the guardrail configuration, or point the offending alias at a model that stays on the classic endpoint.
    - The message names the routed entries, so a deployment that listed several sees which ones are at stake.

??? failure "The OpenAI GPT-5.6 models cost about 10% more than before, with no configuration change"
    They are served by Amazon Bedrock Mantle by default, so that their `web_search` and `code_interpreter` tools work. Mantle has no cross-region inference profiles, so those requests no longer ride the Global profile the classic endpoint uses by default, and pay the In-Region rate — exactly 10% above the Global one, on every token dimension.

    - Accept the routing and the rate: nothing to do. It is what makes [OpenAI GPT web search](api_openai_responses.md#openai-gpt-web-search) available, and the models' usage now appears under Bedrock Mantle rather than Bedrock in [cost reporting](operations_cost_management.md).
    - Prefer the Global rate: `export AWS_BEDROCK_MANTLE_PREFERRED_MODELS=` (empty). The family returns to the classic endpoint, its server tools are refused with a `400`, and token counting works again.
    - A deployment that had already disabled [`AWS_BEDROCK_CROSS_REGION_INFERENCE_GLOBAL`](operations_configuration.md#cross-region-global) paid the In-Region rate before and pays it now: nothing changed for it.

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

??? failure "`400 Bad Request` — a transcription is rejected only for one model"
    The message names the condition: `amazon.nova-2-sonic-v1:0` serves `json` and `text` only, and accepts at most 10 minutes of audio per request. It returns no timestamps and does not report a detected language, so subtitles, `verbose_json`, `diarized_json` and `timestamp_granularities` cannot be produced from it.

    - Request `json` or `text`, or send the same audio to `amazon.transcribe`, which produces timestamps, SRT/VTT subtitles, speaker diarization and longer recordings — see [Transcriptions](api_openai_audio_transcriptions.md).
    - The same model and the same limits apply on [Translations](api_openai_audio_translations.md).

??? failure "`400 Bad Request` — every transcription fails after setting an output encryption key"
    The message is Amazon Transcribe's own failure reason for the job, and it names KMS. Only requests that stage audio in a bucket are affected: [`AWS_TRANSCRIBE_OUTPUT_ENCRYPTION_KEY_ARN`](operations_configuration.md#aws-transcribe-output-encryption-key-arn) encrypts the job's output, so streamed transcriptions, which write nothing, keep working — which is what makes the failure look model-specific at first.

    - Grant the task role `kms:GenerateDataKey` and `kms:Decrypt` on that key, **and** allow the same role in the key's own policy — a grant on only one of the two denies the job. See [Speech-to-Text](operations_iam_permissions.md#speech-to-text-optional).
    - With [`AWS_TRANSCRIBE_REGION`](operations_configuration.md#aws-transcribe-region) unset, a job runs in whichever candidate Region has a co-located bucket, so a single-Region key fails as soon as failover moves the job. Use a [multi-Region key](https://docs.aws.amazon.com/kms/latest/developerguide/multi-region-keys-overview.html), or pin the Region.
    - A key policy conditioned on the [encryption context](operations_configuration.md#aws-transcribe-output-encryption-key-arn) must not require `stdapi-ai.user_id`: it is sent only when the request identifies an end user, so requiring it denies every anonymous call.

??? failure "A streamed transcription delivers nothing until the recording ends"
    `stream=true` always answers with server-sent events, but they arrive phrase by phrase only when the request can be served that way. Otherwise every event is delivered at once, after the whole recording has been read — which reads as a stream that hangs and then completes in a single burst. See [Streaming](api_openai_audio_transcriptions.md#streaming).

    - **Name the language expected**: send `language`, or two or more `languages`. A request naming neither has to read the recording before it can tell which language it is in.
    - **Set [`AWS_TRANSCRIBE_STREAM_LANGUAGES`](operations_configuration.md#aws-transcribe-stream-languages)** to the languages your callers actually send, which gives phrase-by-phrase delivery to requests that name none.
    - **Drop provider-specific parameters other than `VocabularyName`, `VocabularyFilterName` and `VocabularyFilterMethod`.** `MaxSpeakerLabels`, `ShowSpeakerLabels`, `ChannelIdentification`, `MaxAlternatives`, `ToxicityDetection`, `ContentRedaction`, `IdentifyMultipleLanguages` and the rest are honoured in full, but the request that uses one is answered in a single burst rather than phrase by phrase.
    - `response_format=diarized_json` on its own does not cost the phrase-by-phrase delivery: its `transcript.text.segment` events are interleaved with the deltas like any other.

??? failure "`amazon.nova-2-sonic-v1:0` is missing from `/v1/models` or returns `404`"
    The model is not offered in every AWS Region, and the catalog only lists what the configured Regions serve. [Check the model's regions](models.md).

    - Add a Region that offers it to [`AWS_BEDROCK_REGIONS`](operations_configuration.md#aws-bedrock-regions); `us-east-1`, `us-west-2`, `ap-northeast-1` and `eu-north-1` did at the time of writing, while `eu-west-3` and `eu-central-1` did not.
    - Request model access for it in the Amazon Bedrock console for that Region.
    - Grant the task role `bedrock:InvokeModelWithBidirectionalStream`, which this model needs on top of the usual invoke permissions — see [IAM Permissions](operations_iam_permissions.md). Without it the session opens and then ends with a `503` saying live conversation is not available on this server; the server log names the permission.
    - `/search_models?route=openai_audio_transcription` lists what this deployment can actually reach.

??? failure "`503 Service Unavailable` — a speech or speech-to-speech request fails a few seconds after it starts"
    Real-time audio requests must become ready within [`AWS_CONNECT_TIMEOUT`](operations_configuration.md#aws-connect-timeout) in each candidate region — that budget covers the connection, the initial handshake and the first response together, not the connection alone. On a high-latency or NAT-fronted network the default of 5 seconds can expire in every region, and the request then ends as a `503`.

    - Raise [`AWS_CONNECT_TIMEOUT`](operations_configuration.md#aws-connect-timeout) to 10 seconds or more; `AI_RESPONSE_TIMEOUT` governs the response itself and has no effect here.
    - The server log names each Region that was abandoned and why.

??? failure "Conversation items are missing, or cannot be added"
    A [conversation](api_openai_conversations.md) has a bounded lifetime and a bounded number of writes, and both are reached silently.

    - **After 30 days**, a conversation and its items are removed and every route on it returns `404`. Long-lived agents must create a new conversation rather than reusing one indefinitely.
    - **1,000 requests that add or delete items** is the per-conversation ceiling; a response bound to a conversation counts as one, whatever its number of output items. Past it, a listing stops early rather than adding failing: the gateway reads at most 1,000 invocation steps, and a single large item spans several. Start a new conversation, seeding it with the items you still need.
    - **`503` saying the API is not available on the current server** means the IAM role is missing the [Bedrock Session Storage permissions](operations_iam_permissions.md#bedrock-session-storage-optional), including `bedrock:UpdateSession`, which only the metadata update uses — a deployment created before conversations shipped fails on `POST /v1/conversations/{id}` alone. The client message is the same whichever one is absent; the server log names it.
    - **Items added by a streamed response appear when the stream ends**, not while it runs; a client that reads them from a callback fired on the terminal event must wait for the stream to close.

??? failure "S3 error on image generation or audio transcription"
    The S3 bucket is missing, unreachable, or in the wrong region.

    - The Terraform module creates the bucket automatically unless you pass your own via `aws_s3_bucket`.
    - If you're using your own bucket: `AWS_S3_BUCKET` must point to a bucket in the same region as the **first** entry in `AWS_BEDROCK_REGIONS`.
    - Verify the ECS task IAM role has `s3:PutObject` / `s3:GetObject` on the bucket.
    - **`503` saying transcription is not available on the current server** means no region that can run a transcription has a bucket at all: set [`AWS_S3_BUCKET`](operations_configuration.md#aws-s3-bucket), [`AWS_TRANSCRIBE_S3_BUCKET`](operations_configuration.md#aws-transcribe-s3-bucket) or an [`AWS_S3_REGIONAL_BUCKETS`](operations_configuration.md#aws-s3-regional-buckets) entry for it. The server log names the settings.
    - **`503` saying the `'url'` response format is not available on the current server** means there is no bucket to host the images a `url` response points at: set [`AWS_S3_BUCKET`](operations_configuration.md#aws-s3-bucket), or request `response_format="b64_json"`, which needs no storage. Image requests are refused before anything is generated, so a misconfigured deployment bills nothing for them.

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

??? failure "Web search returns nothing, stale results, or is rejected"
    The built-in [web search tool](api_openai_responses.md#openai-gpt-web-search) is gated by both an IAM permission and a server setting, and each failure looks different.

    - **The model answers from its training data and says it could not search**: the task role is missing `bedrock-websearch:InvokeSearch` / `bedrock-websearch:InvokeFetch`. Add them — see [Web Search IAM](operations_iam_permissions.md#web-search-iam). The request itself still succeeds, so this shows up as a weak answer rather than an error.
    - **`400` on `external_web_access`**: the request asked for a value the server does not allow. By default searches stay inside the AWS boundary; set [`AWS_BEDROCK_EXTERNAL_WEB_ACCESS`](operations_configuration.md#bedrock-external-web-access) to change what the server does, or [`AWS_BEDROCK_ALLOW_EXTERNAL_WEB_ACCESS_OVERRIDE`](operations_configuration.md#bedrock-allow-external-web-access-override) to let requests choose. It travels as an extra model parameter (a top-level request field), not as a field of the tool: a client that sets it on the tool changes nothing.
    - **`400` on `external_web_access` saying it is not available with this model**: the override is enabled, but only the models that serve web search natively — the OpenAI GPT-5.x family — take a web access choice per request. Everywhere else the parameter must match the configured value, and is refused rather than accepted and ignored. Send it only with those models, or drop it.
    - **External web access was enabled but results still look cached**: `bedrock-websearch:ExternalWebAccess` is missing from the task role, so the search falls back to the Amazon Bedrock web index.
    - **`400` naming `filters.allowed_domains` or `user_location`**: the model's own search cannot restrict which sources it uses, and running it unrestricted would answer from the very sources the request excluded. Drop the restriction, or send the request to a model that serves web search natively — the OpenAI GPT-5.x family.
    - **The tool is rejected outright**: web search is served on `/v1/responses` for the OpenAI GPT-5.x family, in `us-east-1`, `us-east-2` and `us-west-2`. On `/v1/messages` and `/v1/chat/completions` it is not available for these models.

??? failure "MCP tools declared with `mcp_servers` / `mcp_toolset` are never called"
    The request succeeds and the model answers plausibly, but no MCP server was ever contacted. The [MCP connector](api_anthropic_messages.md#mcp-connector) asks the **model** to act as an MCP client during the turn, and the models this API serves do not open those connections. The connector is therefore accepted and ignored rather than refused, so nothing in the response says it was dropped — no setting enables it.

    - The server log carries a `warning` naming exactly what was ignored (`mcp_servers`, `mcp_toolset`), correlated with `x-request-id` — the caller sees an ordinary `200`, so that line is the only place this is reported.
    - Run the MCP client yourself: connect to the server, declare its tools in `tools`, and return each result as a `tool_result` block. Every other tool in `tools` is kept and behaves normally.
    - A `tool_choice` is dropped with the toolsets when they were the only entries in `tools`, and a `cache_control` breakpoint carried by a dropped `mcp_toolset` goes with it — move the breakpoint to a tool that survives, or the cached prefix is shorter than the one the request paid to write.
    - `mcp_tool_use` and `mcp_tool_result` blocks replayed from a connector-enabled transcript are read as an ordinary `tool_use` and `tool_result`, so a repeated call comes back as a plain `tool_use` block for the client to run.
    - Unrelated to this deployment being an [MCP server](features.md#mcp-model-context-protocol) itself, which is the opposite direction: that lets an AI agent call these endpoints as tools, and is unaffected.

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

??? failure "A batch is refused, stuck, or its requests fail one by one"
    The [Batch API](api_openai_batches.md) and the [Message Batches API](api_anthropic_batches.md) refuse at submit what the backend would otherwise fail hours later, so most surprises land in the create call.

    - **`503` (or `529` on `/anthropic/...`) on every batch endpoint**: the deployment declares no batch service role. Set [`AWS_BEDROCK_BATCH_ROLE_ARN`](operations_configuration.md#aws-bedrock-batch-role-arn) and grant the policies in [Batch Inference IAM](operations_iam_permissions.md#batch-inference). The server also reports the disabled feature in the `server_warnings` field of its `start` log event.
    - **`400` naming a minimum of 100 requests**: batches run on a backend with a floor of 100 requests **per model**. A batch naming several models must reach it for each of them; combine the small ones or send them without batching. The 100 is the default of the per-model Amazon Bedrock quota *Minimum number of records per batch inference job* ([Amazon Bedrock quotas](https://docs.aws.amazon.com/general/latest/gr/bedrock.html)), and it is the value checked here whatever your own account's quota says.
    - **The batch stays `validating` (`in_progress`) for several minutes**: expected. Validation alone takes a few minutes before any request runs, and the whole batch has a 24-hour window. Poll rather than resubmit — a resubmission is a second, separately billed batch.
    - **`400` naming tool use or a structured output schema**: neither is available in a batch. Remove `tools`/`tool_choice` and `response_format` of type `json_schema`, or send those requests without batching.
    - **`400` saying the model is not available for batched requests**: not every model can run batched. Pick another one; when the batch named several models, no sibling job is left running.
    - **A model that batches fine is missing from `search_models?batch=true`, or reports `batch: false`**: that flag is a discovery hint published on a best-effort basis and is never used to reject anything — submit the batch and let the answer decide. It is reported for no model at all while [`COST_TRACKING`](operations_configuration.md#cost-tracking) is disabled, and for a few seconds after startup while the catalogue is still being built.
    - **`503` on creation, with nothing wrong with the request**: the backend refused the job for a reason that is not the model — the account's batch quota for that model, the service role, or a restriction such as a model the provider marked legacy and the account has not used in the last 30 days. The client message is deliberately generic; the server log carries the reason the backend gave, as a warning.
    - **`503` on creation, after the endpoints answered normally**: the task role is missing `bedrock:CreateModelInvocationJob` or the `iam:PassRole` statement on the batch service role; the server log names which. A batch that starts and then fails without results usually means the service role itself cannot read or write the bucket under [`AWS_S3_BATCHES_PREFIX`](operations_configuration.md#aws-s3-batches-prefix) — the reason the backend gives is logged as a warning when a job reports `Failed`.
    - **A batch reports no cached tokens, whatever its requests asked for**: prompt caching does not apply to batched requests, on any model. A cache hint — `cache_control` on `/anthropic/v1/messages/batches`, `prompt_cache_key` or `prompt_cache_breakpoint` on `/v1/batches` — is accepted and dropped rather than refused, so the request is answered normally and no cached tokens are reported for it. There is no discount to lose: batched requests are billed at the batch rate already.
    - **The first read after the batch ends is slow**: the results are translated and published on that read. Later reads are immediate.
    - **The result files are gone, or never go away**: a batch created with `output_expires_after` deletes both files that long after they are written, and one created without it keeps them until they are deleted with the [Files API](api_openai_files.md). The clock starts when the results are published, not when the batch was created.

??? failure "A vector store file stays in progress, fails, or returns nothing"
    Indexing runs after the response is sent, so a file is `in_progress` for a moment by design — see [Vector Stores](api_openai_vector_stores.md#indexing-is-asynchronous).

    - **`503` on every vector store endpoint**: the deployment declares no vector storage. Set [`AWS_S3_VECTORS_BUCKET`](operations_configuration.md#aws-s3-vectors-bucket) and [`AWS_S3_VECTORS_REGION`](operations_configuration.md#aws-s3-vectors-region), keep [`AWS_S3_BUCKET`](operations_configuration.md#aws-s3-bucket) set, and grant the [Vector Stores IAM permissions](operations_iam_permissions.md#vector-stores-optional).
    - **`503` on one operation only** (creating a store, searching, deleting): a single `s3vectors` action is missing from the task role. The client message is deliberately the same as above; the server log names the action and the bucket.
    - **Creating a store fails**: the vector bucket must already exist, in the Region named by [`AWS_S3_VECTORS_REGION`](operations_configuration.md#aws-s3-vectors-region). A vector bucket is a Region-local resource with no failover, so a bucket in another Region is not reachable at all.
    - **A file settles as `failed` with `unsupported_file`**: the store does not index that file type. Read `last_error.message` — it names what **this** store indexes, since that differs per store. A store the server owns indexes text only, so a PDF or an office document settles here; convert it first — [RAG Pipelines](use_cases_rag.md#document-parsing) shows a conversion stage — or attach it to a [knowledge base store](api_openai_vector_stores.md#knowledge-base-stores), which indexes those formats as they stand. When the message names formats and the file is already one of them, its bytes are not what the content type claims.
    - **A file settles as `failed` with `server_error`**: indexing hit a backend error, or was interrupted before it finished — `last_error.message` says which. Interrupted means the server was replaced, scaled in or killed while it was indexing, and nothing else is wrong. Check the `background` log event sharing the request's `id` for the backend case; attach the file again in both. To stop losing that work at every deployment, give the deployment an indexing queue — [`AWS_SQS_VECTOR_STORE_QUEUE_URL`](operations_configuration.md#aws-sqs-vector-store-queue-url) — and another server finishes the job instead.
    - **A file stays `in_progress` far longer than the others**: a large file is many passages, each embedded in turn, and indexing is bounded server-wide, so a file attached while others are being indexed waits its turn. A file no server is indexing any more settles as `failed` rather than waiting for good, so an unchanging `in_progress` is work that is still queued.
    - **A search returns nothing after attaching**: the store is still indexing (`status` is `in_progress`), the store has passed its expiration (`status` is `expired`), or the `filters` match no file. A filter applies to the file's `attributes`, never to its content.
    - **`409` on an update**: several requests are changing the same store at once. Retry the request.

??? failure "Vector store indexing is not picked up by the queue, or a queued file never settles"
    Only deployments that set [`AWS_SQS_VECTOR_STORE_QUEUE_URL`](operations_configuration.md#aws-sqs-vector-store-queue-url) hand indexing to a queue; without it, indexing runs in the server that accepted the request and everything below is expected behaviour rather than a fault.

    - **The server refuses to start, naming the setting**: the URL is not an Amazon SQS queue URL (`https://sqs.<region>.amazonaws.com/<account-id>/<queue-name>`), it names a FIFO queue, or [`AWS_S3_VECTORS_BUCKET`](operations_configuration.md#aws-s3-vectors-bucket) is unset. FIFO is refused because its deduplication would silently drop a legitimate re-attach of the same files.
    - **A `start` log warning says the queue could not be described**: the queue does not exist, or the task role lacks `sqs:GetQueueAttributes`. The deployment still runs and still queues, but it cannot read your redrive policy, so it falls back to its own retry count. Grant the [Durable Vector Store Indexing permissions](operations_iam_permissions.md#durable-vector-store-indexing).
    - **A `start` log warning says the queue has no dead-letter queue**: add a redrive policy. Without one, the message of a file that cannot be indexed is dropped once its retries run out instead of being kept for inspection.
    - **Every file still settles as `failed` after a deployment**: the send is failing, which the server log reports at `error` naming `sqs:SendMessage`. A deployment that cannot queue keeps indexing in-process, which is exactly the behaviour the setting was meant to replace, so the symptom looks like the setting doing nothing.
    - **Files sit `in_progress` for minutes under load**: a server only takes jobs off the queue while it is not busy answering requests, so indexing yields to clients by design. Scale out, or wait.
    - **A file settles as `failed` although the queue is configured**: the job ran out of deliveries. Its message is in your dead-letter queue; the server log says so at `error`. Attach the file again once the underlying cause is fixed.

??? failure "A `vs_kb_...` vector store answers `404`, or refuses a file attached to it"
    A [knowledge base store](api_openai_vector_stores.md#knowledge-base-stores) is addressed, never created, so most of these are configuration rather than a bad request.

    - **`404` on every route of a `vs_kb_...` identifier**: the knowledge base is not listed in [`AWS_BEDROCK_KNOWLEDGE_BASE_IDS`](operations_configuration.md#aws-bedrock-knowledge-base-ids), or it does not exist in the first [`AWS_BEDROCK_REGIONS`](operations_configuration.md#aws-bedrock-regions) entry, or the task role lacks the read permission on it. The three cases answer identically **by design**, so the allowlist cannot be probed by a client; the server log says which one it was. Check the setting first, then grant the [Knowledge Base Vector Stores permissions](operations_iam_permissions.md#knowledge-base-vector-stores) on the knowledge base ARN.
    - **`503` when attaching a file, while search and listing work**: the knowledge base has more than one data source, so which one a document belongs to is ambiguous. Name it in the setting as `<knowledgeBaseId>/<dataSourceId>`.
    - **`400` saying files cannot be attached to the store, while search and listing work**: the allowlisted data source keeps its corpus in sync from somewhere else — a bucket, or another connected service — and takes no file handed to it. Only a **custom** data source does. Point the entry at one, as `<knowledgeBaseId>/<dataSourceId>`; the server log names the data source that refused. A knowledge base can hold both kinds, and the store keeps serving search and listing meanwhile.
    - **`400` when deleting a file a search returned**: that document belongs to the corpus behind the store rather than to the files attached here, so it is readable and never removable. Remove it where the corpus comes from.
    - **`400` on an update, a delete, a chunking strategy, a file batch, a `score_threshold` or a file's `content`**: none of these apply to a store managed outside the server. The [refusal table](api_openai_vector_stores.md#knowledge-base-stores) lists each one and what to do instead.

??? failure "All Bedrock spend still lands on one identity"
    [Per-user attribution](operations_cost_management.md#per-user-attribution) reaches the AWS bill through two AWS-side steps that are easy to miss, and neither is instant:

    - **The Cost and Usage Report export must include caller identity.** Create a Data Exports CUR 2.0 export with *Include caller identity (IAM principal) allocation data* enabled; an existing export cannot be changed and must be re-created. The identity then appears in `line_item_iam_principal` as `assumed-role/<role>/<session>`.
    - **The session tag must be activated as a cost allocation tag**, in the AWS Billing console under **Cost allocation tags**, filtered by type **IAM principal**. It is only listed there after that identity has made at least one call, and takes up to 24 hours to appear in Cost Explorer.
    - **Requests that identify no end user are billed to the server**, by design. The request log's `aws_role_session_name` field is absent on exactly those requests — use it to find the clients that send no identifier, then enable [`AWS_BEDROCK_USER_ROLE_REQUIRE_IDENTITY`](operations_configuration.md#aws-bedrock-user-role-require-identity).
    - **Only model invocations are attributed.** Video generation, guardrail evaluations, speech, transcription and translation stay on the server's own identity.

??? failure "The organization usage endpoints answer `503 feature_unavailable`"
    [`USAGE_API`](operations_configuration.md#usage-api) is off, which is the default — the endpoints exist and refuse, so this is neither a wrong path nor a rejected credential.

    - Set `USAGE_API=true` and restart. It also needs [`CLOUDWATCH_METRICS`](operations_configuration.md#cloudwatch-metrics), which publishes the metrics these endpoints are answered from, and `/v1/organization/costs` additionally needs [`COST_TRACKING`](operations_configuration.md#cost-tracking). The server log names whichever one is missing.
    - Read [what a query costs](operations_cost_management.md#usage-api-cost) before turning it on: every query is billed per metric read, and enabling it also stores additional metric series.
    - A `404` instead means the prefix is wrong: the routes live under `${OPENAI_ROUTES_PREFIX}/v1/organization/...` — see [`OPENAI_ROUTES_PREFIX`](operations_configuration.md#openai-routes-prefix).
    - `GET /v1/usage` is not served at any setting: the retired endpoint is absent from OpenAI's current API surface and from the `openai` SDK. Use [`GET /v1/organization/usage/...`](api_openai_organization_usage.md) instead.

??? failure "The usage endpoints answer, but every bucket is empty"
    Nothing was refused — the query simply found no published metric in the range it covers. In order of likelihood:

    - **[`CLOUDWATCH_METRICS`](operations_configuration.md#cloudwatch-metrics) is disabled**: nothing is published, so there is nothing to report. These endpoints read those metrics and produce none of their own.
    - **`/v1/organization/costs` alone is empty**: [`COST_TRACKING`](operations_configuration.md#cost-tracking) is disabled, so no cost is computed and no `Cost` metric is published. The usage endpoints are unaffected.
    - **The range predates the feature**: usage exists only from the moment `CLOUDWATCH_METRICS` was enabled, and nothing is backfilled. Per-endpoint buckets start later still — only from the moment [`USAGE_API`](operations_configuration.md#usage-api) was enabled, since that is what publishes the `Operation` dimension, so a query grouped or filtered by endpoint is empty over traffic served before it.
    - **A multi-region deployment reports one region's traffic**: the endpoints read a single Amazon CloudWatch region, [`CLOUDWATCH_METRICS_REGION`](operations_configuration.md#cloudwatch-metrics-region), defaulting to the first entry of [`AWS_BEDROCK_REGIONS`](operations_configuration.md#aws-bedrock-regions). Point it at the region the metrics are actually ingested in.

??? failure "The organization usage endpoints answer `403`"
    They are an administrator surface, so the credential that calls the models is not the credential that reads them.

    - **A tenant API key is never accepted**, whatever its scope: these endpoints report the whole deployment's consumption and spend, not one tenant's. Read them with the deployment's own [API key](operations_configuration.md#api-key).
    - **A user pool token must carry every scope** named in [`USAGE_API_ADMIN_SCOPES`](operations_configuration.md#usage-api-admin-scopes) — all of them, not one of them.
    - **With that list empty** (the default) no token is accepted at all, and only the deployment's own API key may read them. Name a scope there to let an operator's token in.

??? failure "A usage query is refused as too large, or `bucket_width=1m` is refused over an older range"
    A query outside these bounds is refused rather than truncated or quietly answered at a resolution it did not ask for — each one is billed by Amazon CloudWatch per metric read, and a partial answer would be indistinguishable from a quiet period.

    - **Too many metric series**: the query matched more than [`USAGE_API_MAX_METRICS`](operations_configuration.md#usage-api-max-metrics) (500 by default, which is also CloudWatch's own per-request maximum). Narrow it with `models`, or ask for fewer `group_by` keys.
    - **The range is too long**: it spans more than [`USAGE_API_MAX_RANGE_DAYS`](operations_configuration.md#usage-api-max-range-days) (92 days by default). Split it across several queries, or raise the setting.
    - **`bucket_width=1m` over an older range**: one-minute buckets are reported for the last 15 days only, one-hour and one-day buckets for the last 455. Request `1h` or `1d`, or move the range inside 15 days.

??? failure "A WebSocket upgrade to `/v1/realtime` answers `404`"
    The upgrade request never reached the [Realtime API](api_openai_realtime.md) route — either the URL is wrong, or something in front of the deployment does not forward a WebSocket upgrade at all.

    - Check the path: the route lives at `${OPENAI_ROUTES_PREFIX}/v1/realtime` (`/v1/realtime` with the default empty prefix). A client dialling the wrong prefix reaches no route and gets `404` from the framework itself, before authentication is even checked.
    - An Application Load Balancer forwards a WebSocket upgrade by default, but anything placed **in front of** it — a CDN, an API Gateway REST API (which has no WebSocket support at all), a reverse proxy that does not pass through `Connection: Upgrade` / `Upgrade: websocket` — answers its own `404` or refuses the upgrade first. See [WebSocket-Capable Deployment](operations_deploy_advanced.md#websocket-capable-deployment-realtime-api).
    - Confirm the deployment actually ships the Realtime API: this is a versioned feature, not every earlier release includes it.

??? failure "`POST /v1/realtime/calls` answers `404`"
    That endpoint is the upstream API's WebRTC and SIP transport negotiation, and it is **not served here** — every realtime session runs over the WebSocket. Nothing in the deployment enables it: no setting, no IAM permission and no load balancer configuration changes the answer. See [Transports](api_openai_realtime.md#transports).

    - **A browser does not need it.** It connects to the same `WS /v1/realtime` with an [ephemeral client secret](api_openai_realtime.md#ephemeral-client-secrets) in the `Sec-WebSocket-Protocol` list, and captures and plays the audio itself.
    - **A client that must speak WebRTC or SIP** — a phone line, or a browser on a lossy network that needs jitter and packet-loss handling — belongs behind a voice-agent framework that terminates the media itself and reaches this API over the WebSocket. See [Put WebRTC or a phone line in front of the gateway](api_openai_realtime.md#put-webrtc-or-a-phone-line-in-front-of-the-gateway).
    - **Do not try to route UDP through the deployment's ALB**: its listeners carry HTTP and HTTPS only. A media terminator you run yourself needs its own ingress — see [WebRTC and SIP need their own ingress](operations_deploy_advanced.md#webrtc-and-sip-need-their-own-ingress).

??? failure "A realtime session closes exactly at 8 minutes, or unexpectedly earlier"
    Two different things end a [Realtime](api_openai_realtime.md) session, and the close code tells them apart — inspect it in the client's WebSocket close handler.

    - **Close code `1000`, reason `session_expired`**: expected. Every session is capped at 8 minutes; reconnect to continue the conversation — see [Session Lifecycle and Limits](api_openai_realtime.md#session-lifecycle-and-limits).
    - **Closes with no code from the gateway at all, well before 8 minutes**: the load balancer's idle timeout fired on a quiet stretch between spoken turns. Raise `alb_idle_timeout` to at least 8 minutes (480 seconds) — see [The idle timeout bounds a session](operations_deploy_advanced.md#the-idle-timeout-bounds-a-session).
    - **Closes during a deployment, scale-in, or Spot interruption**: the ECS task holding the session was replaced. There is no live handoff between tasks — see [A deploy truncates open sessions](operations_deploy_advanced.md#a-deploy-truncates-open-sessions).
    - **Close code `1001`, reason `server_shutdown`**: the deployment was shutting down when the session was still open; reconnect once it is back.
    - **Close code `3000`**: a fatal error, not a limit. The reason is `<error type>.<error code>`, and a terminal `error` event carrying the same detail was sent just before the close frame.

??? failure "A raw WebSocket client gets `403` that looks like an authentication failure"
    If the deployment's WAF is enabled with the AWS-managed Common Rule Set (`alb_waf_enabled = true`), its `NoUserAgent_HEADER` rule blocks any request — including a WebSocket upgrade — that carries no `User-Agent` header, with a plain `403` that is easy to mistake for a rejected credential.

    - Every mainstream WebSocket client library sets a `User-Agent` automatically; this only surfaces with a hand-rolled client (a bespoke SIP/telephony bridge, a minimal test script).
    - Check the WAF sampled requests in the console to confirm `NoUserAgent_HEADER` is the rule that matched, before assuming the API key or ephemeral secret is wrong.
    - Have the client send any non-empty `User-Agent`, or exclude the rule for the Realtime path. See [AWS WAF's `NoUserAgent_HEADER` rule reads like an auth failure](operations_deploy_advanced.md#aws-wafs-nouseragent_header-rule-reads-like-an-auth-failure).

??? failure "An ephemeral client secret works on one instance and is rejected on another"
    A [Realtime API ephemeral client secret](api_openai_realtime.md#ephemeral-client-secrets) is a signed token with nothing stored server-side, verified by re-checking its signature against a shared key — every instance must sign with the **same** key for that to work.

    - With no [`API_KEY`](operations_configuration.md#api-key)-family setting configured at all, each instance falls back to a **random signing key generated per process**, so a secret minted by one instance never verifies on another — the symptom is intermittent rejection that tracks which instance the client's connection happened to land on.
    - Set [`REALTIME_CLIENT_SECRET_KEY`](operations_configuration.md#realtime-client-secret-key) explicitly to a value shared by every instance; this also covers a deployment with no API key by design (e.g. behind an IP-restricted ALB).
    - A deployment that already configures an API key is unaffected: the signing key is derived from it automatically, and that same key is already shared across instances.
    - Rotating the API key or `REALTIME_CLIENT_SECRET_KEY` invalidates every client secret minted before the change, the same as an expired one.

??? failure "An Ollama client shows no models, or refuses to connect at all"
    The Ollama-compatible endpoints answer at `/api/*` on the deployment's base URL, and the models they list are this deployment's, not the ones a local Ollama had pulled.

    - Point the client at the deployment URL with **no path suffix** — `https://your-host`, not `https://your-host/v1` — unless you set [`OLLAMA_ROUTES_PREFIX`](operations_configuration.md#ollama-routes-prefix), in which case add it.
    - Send the deployment's API key as a Bearer token: a local Ollama needs no credentials, so a client configured against one usually has no field filled in, and every endpoint here answers `401` without it.
    - Choose a model from [`GET /api/tags`](api_ollama_models.md). A name learned from ollama.com such as `llama3.2:3b` is not served and answers `404`; a trailing `:latest` on a name that *is* served is accepted.
    - A client that probes `GET /` to detect an Ollama server will not recognise this one — that path serves the deployment's own root document. Probe [`GET /api/version`](api_ollama_models.md) instead.

??? failure "An Ollama client shows `0` or `NaN` tokens per second"
    Tokens per second is computed from `eval_count` divided by `eval_duration`, and `eval_duration` is only reported when the response streamed.

    - Request the response with `"stream": true` (the default on `/api/chat` and `/api/generate`) and the durations are measured and reported.
    - On a non-streamed response `prompt_eval_duration` and `eval_duration` are **omitted**, because a buffered answer carries no split between reading the prompt and generating the answer, and a number with nothing behind it would be an invention. `load_duration` is never reported for the same reason: nothing is loaded, since models are served on demand.
    - The token counts themselves (`prompt_eval_count`, `eval_count`) and `total_duration` are always reported.

??? failure "`ollama pull` appears to do nothing, and `ollama list` shows every model at size 0"
    Both are correct. Models here are served on demand and none is stored on the deployment.

    - [`POST /api/pull`](api_ollama_models.md) reports success immediately for any model `/api/tags` lists, because it is already usable — there is nothing to transfer. A model this deployment does not serve answers `404` instead.
    - `size` is `0` and the parameter-count, quantization and format details are empty because they describe a model file that does not exist here. `digest` is a stable identifier derived from the model name, usable as a cache key but not a hash of any content.
    - `create`, `copy`, `push` and `delete` answer `400`: there is no model store for them to change, and reporting success would tell the client that state changed when nothing did.

??? failure "An Ollama client shows no thinking text on a reasoning model"
    `message.thinking` follows the deployment-wide reasoning setting.

    - [`CHAT_COMPLETIONS_REASONING_FIELD`](operations_configuration.md#chat-completions-reasoning-field) set to `none` suppresses the reasoning text on every dialect, including this one. Set it back to `reasoning_content` or `reasoning` to have it emitted.
    - `think` must also be set on the request; without it a model returns its answer only.
    - The `capabilities` list from `/api/show` never advertises `thinking`, so a client gating its toggle on that list will not offer it. `think` can still be sent to any model — one that does not reason simply returns no thinking text.

??? failure "A batch is missing from the listing, but still answers when retrieved by ID"
    The listing answers from a window of the most recent batches, found by a seek bounded to a fixed number of storage requests. A burst of thousands of batches created inside the same minute can outrun that budget, and the ones beyond it fall outside the window.

    - The record is intact: [`GET /v1/batches/{batch_id}`](api_openai_batches.md) returns it, and so does cancelling or reading its output. Only the *listing* is bounded — see [Listing Order](api_openai_batches.md#listing-order).
    - A cursor that points outside the current window returns an empty page rather than an error, so a paginating client stops early instead of failing.
    - Record the `id` each `POST /v1/batches` returns and address batches by it, rather than rediscovering them through the listing. Amazon Bedrock's own batch quotas bound how fast batches can realistically be created, so this density is hard to reach by accident.

### AWS error → HTTP status mapping

stdapi.ai translates upstream AWS error codes into standard HTTP responses with an OpenAI/Anthropic-style error type. Use this table to map a status code back to its likely AWS cause. HTTP status and error type are as returned on OpenAI-compatible routes (`/v1/...`); Anthropic-compatible routes (`/anthropic/...`) diverge on the two footnoted rows.

| HTTP  | Error type                  | AWS error codes                                                                                         | Typical cause                                 |
|-------|------------------------------|---------------------------------------------------------------------------------------------------------|-----------------------------------------------|
| `400` | `invalid_request_error`     | `ValidationException`, `BadRequestException`                                                            | Unsupported/invalid request parameters        |
| `400` | `invalid_request_error`     | `AccessDenied` — on the object an `s3://` input named                                                   | The caller's own object cannot be read[^4]    |
| `401` | `authentication_error`      | `UnrecognizedClientException`, `InvalidSignatureException`, `ExpiredTokenException`                     | stdapi.ai's AWS credentials missing/expired   |
| `403` | `permission_error`          | `AccessDeniedException` — on a model call an end user's own role signed                                 | That end user is not allowed that model[^3]   |
| `404` | `invalid_request_error`[^1] | `ResourceNotFoundException`                                                                             | Model or resource not available in the region |
| `429` | `rate_limit_error`          | `ThrottlingException`, `TooManyRequestsException`, `ServiceQuotaExceededException`                      | Bedrock quota / throttling                    |
| `503` | `feature_unavailable`       | `AccessDeniedException`, `AccessDenied` — every other denial                                            | IAM task role lacks permission / model access |
| `503` | `server_error`[^2]          | `ServiceUnavailableException`, `InternalServerException`, `ServiceFailureException`, `ReadTimeoutError` | Transient AWS-side error — retry              |

[^1]: Anthropic-compatible routes return `not_found_error` instead.
[^2]: Anthropic-compatible routes return HTTP `529` with error type `overloaded_error` instead.
[^3]: Only when [per-user cost attribution](operations_cost_management.md#per-user-attribution) is enabled: the call then carries the end user's identity, and AWS evaluated a policy written about them.
[^4]: Only for a bucket declared in [`AWS_S3_ACCEPTED_BUCKETS`](operations_configuration.md#aws-s3-accepted-buckets), which the deployment reads but does not own — so the refused object is the one the request named. The message names that input, and nothing else. A denial on the deployment's own buckets stays `feature_unavailable`.

!!! note "Where to find the detail"
    For security, `401`, `403` and `feature_unavailable` responses returned to clients contain only a generic message — the same one whatever is missing, so that the difference between "no permission" and "not configured" is not disclosed. The full diagnostic detail is captured in the server logs under `error_detail` and can be correlated via the `x-request-id` response header (`request-id` on Anthropic-compatible `/anthropic/...` routes) — see [Logging & Monitoring](operations_logging_monitoring.md).

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

    - **Nothing published**: `GET /.well-known/oauth-protected-resource` answering `404` means [`OAUTH_RESOURCE_IDENTIFIER`](operations_configuration.md#oauth-resource-identifier) is unset. That is the default, not a fault — set it to turn discovery on. With an [`AWS_COGNITO_USER_POOL_ID`](operations_configuration.md#aws-cognito-user-pool-id) configured it is the only setting needed; otherwise set [`OAUTH_AUTHORIZATION_SERVERS`](operations_configuration.md#oauth-authorization-servers) too.
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

    - **No authorization server**: [`OAUTH_AUTHORIZATION_SERVERS`](operations_configuration.md#oauth-authorization-servers) is required with [`OAUTH_RESOURCE_IDENTIFIER`](operations_configuration.md#oauth-resource-identifier) unless [`AWS_COGNITO_USER_POOL_ID`](operations_configuration.md#aws-cognito-user-pool-id) is set, which supplies the issuer; a document naming none leaves a client unable to obtain a token.
    - **Issuer contradicting the pool**: with a user pool configured, the published issuers must include the pool's own — otherwise clients are sent to an authorization server whose tokens every request refuses. Leave [`OAUTH_AUTHORIZATION_SERVERS`](operations_configuration.md#oauth-authorization-servers) empty to publish exactly the pool issuer, or list it alongside the others.
    - **Setting without an identifier**: the authorization servers and the scopes describe a document that is not published without [`OAUTH_RESOURCE_IDENTIFIER`](operations_configuration.md#oauth-resource-identifier).
    - **Malformed value**: the identifier is an origin with no path or query, each issuer is an `https` URL with no query or fragment, and a scope carries no space or quote.

??? failure "A tenant API key is refused with 503"
    A `503` for an `sk-std-...` credential means the key could not be *checked*, not that it is wrong: the tenant records in the DynamoDB table are unreachable, and the gateway fails closed rather than guessing. The server log's `error_detail` names what to fix:

    - **Missing permission**: the task role needs the [shared table permissions](operations_iam_permissions.md#shared-table) (`dynamodb:GetItem` in particular) on the table ARN.
    - **Missing or wrong table**: [`AWS_DYNAMODB_TABLE`](operations_configuration.md#aws-dynamodb-table) and [`AWS_DYNAMODB_REGION`](operations_configuration.md#aws-dynamodb-region) must name the table holding the tenant records.
    - **Record from a newer build**: during a rolling deployment, an instance on the previous version refuses records written with a newer layout; the refusals stop when the rollout completes.
    - Other credential kinds are unaffected: the deployment API key and Cognito tokens keep working through the outage.

??? failure "A newly declared tenant has no key in Parameter Store"
    The server mints pending tenants at startup and then once a minute, so the parameter appears within about a minute of `terraform apply` — if it can:

    - **Missing permission**: minting needs `ssm:PutParameter` and `ssm:GetParameter` on the [delivery prefix](operations_iam_permissions.md#tenant-key-delivery), and the [shared table permissions](operations_iam_permissions.md#shared-table) to record the hash. The refusal is in the server log.
    - **A parameter already exists at that name** with something that is not this tenant's key: the server refuses to adopt it. Delete the parameter and let the next cycle mint a fresh key.
    - **The feature is off**: [`TENANT_API_KEYS`](operations_configuration.md#tenant-api-keys) must be `true` on the running service, not only in the table.

??? failure "A revoked or re-scoped tenant key still works"
    Each instance caches a validated key for [`TENANT_KEY_CACHE_SECONDS`](operations_configuration.md#tenant-key-cache-seconds) — 60 seconds by default — so a revocation, a `disabled = true` or a scope change takes up to that long to reach every instance. That window is the documented trade against a table read per request; lower the setting if a minute is too long, `0` disables the cache entirely.

??? failure "A tenant with a registered AWS role gets 403 on every model call"
    The fixed message *"The AWS credential registered for this API key could not be used"* means the gateway could not open (or keep) a session of the tenant's role; *"Your AWS account does not have access to this model"* means the session opened but the tenant's account refused the invocation. The full AWS detail is in the server log only. In order of likelihood:

    - **Trust policy**: the tenant role must trust the deployment's account with `sts:AssumeRole`, conditioned on the exact `ExternalId` the server minted — read it from the `external_id` attribute of the tenant's `secret#<key id>` record. A wrong or missing ExternalId is indistinguishable from a revoked trust, on purpose.
    - **Gateway-side permission**: the task role needs [`sts:AssumeRole` on the tenant role](operations_iam_permissions.md#tenant-aws-credentials).
    - **Model access in the tenant's account**: the tenant must have been granted access to the model in *its* account, in the serving Region — including [enabling opt-in Regions](https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html) a cross-Region profile routes to.
    - **The role's own policy**: it must allow the Bedrock invocation actions on the model (and its inference profiles).

    A `503` *"…could not be used right now. Retry the request"* is **not** one of these: it means AWS STS itself was throttled or unreachable, or the deployment's own session had expired. Nothing on the tenant's side is wrong, and the request is worth retrying — a client SDK retries it on its own.

??? failure "A tenant record declaring aws_role_arn is refused with 503"
    A declared role is never silently ignored — the gateway refuses the key rather than billing the deployment for a tenant that expects its own account. The server log names which of these it is:

    - **The feature is off**: [`TENANT_AWS_CREDENTIALS`](operations_configuration.md#tenant-aws-credentials) must be `true`, or the attribute removed.
    - **The ExternalId is not minted yet**: for a tenant created before this feature existed, the server mints one within a minute of the role being declared; the refusal covers that window.
    - **The ARN is malformed**: `aws_role_arn` must be an IAM role ARN, `arn:aws:iam::<account>:role/<name>`.
    - **A guardrail is configured**: the combination is refused at startup — see [the incompatibility](operations_authentication_security.md#tenant-aws-credentials).

??? failure "One tenant gets 404 for a model that works for everyone else"
    A model outside a tenant's scope answers the standard `model_not_found`, indistinguishable from a model that does not exist — by design, so the catalogue leaks nothing. Check the tenant's `models_allow` and `models_deny` patterns against the **resolved** model ID (after aliases — the ID the working requests are logged with), and remember an **empty** `models_allow` list allows nothing, while an absent one restricts nothing. `GET /v1/models` is not filtered per tenant, so a model appearing there can still be refused at invocation.

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
