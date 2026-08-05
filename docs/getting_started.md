---
title: Get Started - Run stdapi.ai locally or on AWS
description: Pick your path to try stdapi.ai - a one-line Docker run for local development, or a two-command Terraform deployment on AWS with a 14-day free trial of the license.
keywords: stdapi.ai getting started, try stdapi.ai, OpenAI gateway AWS, Anthropic gateway AWS, AWS Bedrock quickstart, Docker local AI gateway, Terraform AI gateway, AWS Marketplace AI
---

# :material-rocket-launch: Get Started

Pick the path that fits where you are right now. Both use the same OpenAI, Anthropic, and Cohere-compatible API — graduating from one to the other is a client-side change: the base URL, and usually the model name.

Looking for reference documentation rather than a quickstart? See [Features](features.md) for what the gateway does, and the [API Overview](api_overview.md) for endpoints, parameters, and SDK usage.

!!! tip trial "Start here: 14-day free trial on AWS"
    Deploy the production-ready stack today — the stdapi.ai license is free for 14 days. AWS charges for what it runs (ALB, Fargate, KMS, NAT) and for Bedrock usage from the first minute. After the trial, the license is $0.10/container-hour — cancel anytime.

## :material-clipboard-check-outline: Before You Start

Both paths need:

- **An AWS account** with access to [Amazon Bedrock](https://aws.amazon.com/bedrock/). [Create one free](https://aws.amazon.com/free/) if you don't have one.
- **AWS credentials configured locally** — `aws configure` or `aws sso login` ([AWS CLI setup guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html)).

The AWS Terraform path additionally needs:

- **Administrator-level AWS permissions** — the module provisions IAM roles and policies, KMS keys, ECS/Fargate, ALB, and networking. A restricted developer profile will fail.
- **A sandbox AWS account is strongly recommended** for evaluation. Replicate into your target account once you've validated the stack.
- [Terraform](https://www.terraform.io/downloads) or [OpenTofu](https://opentofu.org/docs/intro/install/) >= 1.5.
- An **[AWS Marketplace subscription](https://aws.amazon.com/marketplace/pp/prodview-su2dajk5zawpo)** (14-day free trial of the license).

The Docker path additionally needs [Docker](https://docs.docker.com/get-started/get-docker/) or [Podman](https://podman.io/docs/installation).

---

## :material-directions-fork: Pick Your Path

<div class="grid cards" markdown>

-   :material-aws:{ .lg .middle } __14-day free trial — Deploy on AWS with Terraform__

    ---

    **Start here — recommended for production.**

    - 2 Terraform commands → ECS Fargate with HTTPS, auto-scaling, optional WAF
    - Hardened container image from **AWS Marketplace** — license free for 14 days, then $0.10/container-hour
    - IP-restricted by default — safe to test right away
    - Multi-region variants available (EU / US) for data residency

    [:octicons-arrow-right-24: Deploy on AWS](operations_getting_started.md){ .md-button .md-button--primary }

-   :material-docker:{ .lg .middle } __One command — Try locally with Docker__

    ---

    **Lighter option, for:** local development, evaluation, open-source projects.

    - One `docker run` command — no AWS infrastructure to provision
    - Uses your local AWS credentials (mounted from `~/.aws`) to reach Bedrock
    - No authentication or IP restriction by default — intended for local use only
    - Free **community image** (AGPL-3.0)
    - Full API compatibility — same endpoints as production

    [:octicons-arrow-right-24: Run Locally with Docker](operations_getting_started_local.md){ .md-button }

</div>

---

## :material-help-circle-outline: Not Sure Which to Pick?

| You are... | Go with | Why |
|---|---|---|
| A developer evaluating for the first time | **Docker (local)** | Fastest feedback loop, no cloud resources to tear down. |
| A team lead validating for production use | **AWS (Terraform)** | Mirrors real deployment. 14-day free license trial during evaluation. |
| Already running a production workload elsewhere | **AWS (Terraform)** | Jump straight to the stack you'll operate long-term. |
| Contributing to an open-source project | **Docker (local)** | AGPL-3.0 community image is free to use and redistribute. |

You can start local and migrate to AWS later — same API, same SDKs. You point the base URL at your AWS endpoint, and usually update the model name.

---

## :material-arrow-right: After Your First Call

<div class="grid cards" markdown>

- :material-star-four-points-outline: [**Features**](features.md) — What the gateway supports, substantiated claim by claim
- :material-book-open-variant: [**API Overview**](api_overview.md) — Endpoints, parameters, and SDK usage
- :material-magnify: [**Search Models**](api_search_models.md) — Find the right model ID by modality, route, region, or streaming support
- :material-puzzle: [**Use Cases**](use_cases.md) — Open WebUI, n8n, coding assistants, and more
- :material-cog: [**Configuration**](operations_configuration.md) — Every environment variable and option
- :material-wrench: [**Troubleshooting**](operations_troubleshooting.md) — Common first-deployment errors and fixes
- :material-server-network: [**Advanced Deployment**](operations_deploy_advanced.md) — VPC integration, multi-region, cost optimization
- :material-scale-balance: [**Licensing**](operations_licensing.md) — Community (AGPL) vs Commercial

</div>
