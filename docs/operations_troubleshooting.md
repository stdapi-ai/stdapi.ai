---
title: Troubleshooting - Common stdapi.ai deployment issues
description: Fixes for the most common errors encountered when deploying and running stdapi.ai - Terraform failures, 503/403/404 responses, Bedrock throttling, S3 bucket errors, VPC connectivity, and more.
keywords: stdapi.ai troubleshooting, AWS Bedrock errors, Terraform apply failed, 503 ECS service, 403 API key, 404 model not found, ThrottlingException Bedrock, S3 bucket region, ElastiCache capacity, VPC endpoint timeout, podman SELinux
---

# :material-wrench: Troubleshooting

Common issues when deploying stdapi.ai for the first time. If your error isn't listed here, open an issue on [GitHub](https://github.com/stdapi-ai/stdapi.ai/issues) or reach out via the [AWS Marketplace contact form](https://aws.amazon.com/marketplace/pp/prodview-su2dajk5zawpo).

---

## :material-cloud-upload: Terraform / Deployment

??? failure "`terraform apply` fails with AccessDenied on IAM, KMS, or ECS actions"
    Your AWS profile does not have sufficient permissions. The stdapi.ai Terraform module provisions IAM roles, KMS keys, ECS, ALB, WAF, Route53 records, and (for some samples) RDS and ElastiCache.

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

    ```
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
        ghcr.io/stdapi-ai/stdapi.ai-community:latest
      ```

---

## :material-api: Runtime / First API call

??? failure "`503 Service Unavailable` on the /docs page or any endpoint"
    The ECS service is still starting up. Health checks take a few minutes.

    - Wait 2–3 minutes after deployment and refresh.
    - Check the ALB target group health in the AWS console.
    - If it persists longer than 5 minutes, inspect CloudWatch logs for the ECS task.

??? failure "Browser TLS warning on the /docs page"
    The ALB uses the default `*.elb.amazonaws.com` domain, which has no trusted certificate. This is expected and safe to bypass for testing.

    - For a production-grade certificate, configure a custom domain — the [Terraform module](https://github.com/stdapi-ai/terraform-aws-stdapi-ai) supports ACM-managed certificates via `alb_domain_name`.

??? failure "`403 Unauthorized` on all requests"
    The API key is missing, wrong, or not configured.

    - Pass the key in the `Authorization: Bearer <key>` header (OpenAI-style) or `X-API-Key` header.
    - Retrieve the generated key with `terraform output -raw api_key`.
    - If `api_key_create = true` was not set, no API key is configured and requests pass through without authentication by default (useful for testing behind IP-restricted ALB, not for production).
    - See [Authentication & Security](operations_authentication_security.md) for all options.

??? failure "`404 Not Found` / model not available"
    The model ID isn't available in your configured region(s).

    - Start with a low-friction model to confirm the pipeline works: `amazon.nova-micro-v1:0` (available in all standard Bedrock regions).
    - List every discovered model with full details: `GET /search_models` (the default model-discovery endpoint). Filter by capability with query parameters — e.g. `GET /search_models?input_modalities=IMAGE&route=/v1/chat/completions` returns only vision-capable chat models. See the [Search Models API](api_search_models.md) reference.
    - `GET /v1/models` is also available for strict OpenAI SDK compatibility (lighter payload, no capability metadata).
    - Verify `AWS_BEDROCK_REGIONS` includes a region that offers the model — see the [Bedrock model availability table](https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html).
    - For Anthropic SDK clients, use either the full Bedrock ID (`anthropic.claude-opus-4-8`) or the Anthropic alias (`claude-opus-4-8`) — both resolve automatically.

??? failure "`ThrottlingException` / too many requests immediately"
    You've hit the per-region Bedrock quota.

    - Add more regions to `AWS_BEDROCK_REGIONS`. Each region has its own independent quota — three regions ≈ triple the throughput.
    - See [Resilience & Failover](operations_resilience.md) for multi-region routing configuration.
    - Check quotas in the AWS Service Quotas console for **Amazon Bedrock**.

??? failure "S3 error on image generation or audio transcription"
    The S3 bucket is missing, unreachable, or in the wrong region.

    - The Terraform module creates this bucket automatically via `s3_bucket_create = true`.
    - If you're using your own bucket: `AWS_S3_BUCKET` must point to a bucket in the same region as the **first** entry in `AWS_BEDROCK_REGIONS`.
    - Verify the ECS task IAM role has `s3:PutObject` / `s3:GetObject` on the bucket.

??? failure "Connection timeout to AWS services from ECS"
    Outbound traffic to AWS endpoints is blocked.

    - Confirm the ECS task's security group allows outbound HTTPS (port 443).
    - If using **VPC endpoints** (the commercial Terraform default), verify the endpoint security groups and policies permit traffic from the ECS task subnet.
    - If ECS runs in a private subnet without VPC endpoints, confirm the NAT gateway / route table is configured.

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
    - See [Authentication & Security → ALB OIDC](operations_authentication_security.md).

---

## :material-arrow-right: Still stuck?

- [Configuration reference](operations_configuration.md) — Every environment variable.
- [Terraform module docs](https://github.com/stdapi-ai/terraform-aws-stdapi-ai) — All module inputs and outputs.
- [GitHub issues](https://github.com/stdapi-ai/stdapi.ai/issues) — Report a bug or ask a question.
- [Advanced Deployment](operations_deploy_advanced.md) — VPC integration, manual ECS, multi-region.
