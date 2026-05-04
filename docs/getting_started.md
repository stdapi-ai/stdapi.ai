---
title: Get Started - Run stdapi.ai in minutes
description: Pick your path to try stdapi.ai - a one-line Docker run for local development, or a 5-minute Terraform deployment on AWS with a 14-day free trial.
keywords: stdapi.ai getting started, try stdapi.ai, OpenAI gateway AWS, Anthropic gateway AWS, AWS Bedrock quickstart, Docker local AI gateway, Terraform AI gateway, AWS Marketplace AI
---

# :material-rocket-launch: Get Started

Pick the path that fits where you are right now. Both use the same OpenAI and Anthropic-compatible API — you can graduate from one to the other without changing your application code.

<div class="grid cards" markdown>

-   :material-docker:{ .lg .middle } __30 seconds — Try locally with Docker__

    ---

    **Best for:** First look, local development, evaluation, open-source projects.

    - One `docker run` command — no AWS infrastructure to provision
    - Uses your local AWS credentials to reach Bedrock
    - Free **community image** (AGPL-3.0)
    - Full API compatibility — same endpoints as production

    [:octicons-arrow-right-24: Run Locally with Docker](operations_getting_started_local.md){ .md-button .md-button--primary }

-   :material-aws:{ .lg .middle } __5 minutes — Deploy on AWS with Terraform__

    ---

    **Best for:** Production workloads, team access, internal tools, SaaS.

    - 3 Terraform commands → ECS Fargate with HTTPS, WAF, auto-scaling
    - Hardened container image from **AWS Marketplace** (14-day free trial)
    - IP-restricted by default — safe to test right away
    - Multi-region variants available (EU / US) for data residency

    [:octicons-arrow-right-24: Deploy on AWS](operations_getting_started.md){ .md-button .md-button--primary }

</div>

---

## :material-help-circle-outline: Not sure which to pick?

| You are... | Go with | Why |
|---|---|---|
| A developer evaluating for the first time | **Docker (local)** | Fastest feedback loop, no cloud resources to tear down. |
| A team lead validating for production use | **AWS (Terraform)** | Mirrors real deployment. 14-day free trial covers evaluation. |
| Already running a production workload elsewhere | **AWS (Terraform)** | Jump straight to the stack you'll operate long-term. |
| Contributing to an open-source project | **Docker (local)** | AGPL-3.0 community image is free to use and redistribute. |

You can start local and migrate to AWS later — your client code doesn't change.

---

## :material-clipboard-check-outline: Before you start

Both paths need:

- **An AWS account** with access to [Amazon Bedrock](https://aws.amazon.com/bedrock/). [Create one free](https://aws.amazon.com/free/) if you don't have one.
- **AWS credentials configured locally** — `aws configure` or `aws sso login` ([AWS CLI setup guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-quickstart.html)).

The AWS Terraform path additionally needs:

- **Administrator-level AWS permissions** — the module provisions IAM roles, KMS keys, ECS, ALB, WAF, and networking. A restricted developer profile will fail.
- **A sandbox AWS account is strongly recommended** for evaluation. Replicate into your target account once you've validated the stack.
- [Terraform](https://www.terraform.io/downloads) or [OpenTofu](https://opentofu.org/docs/intro/install/) >= 1.5.
- An **[AWS Marketplace subscription](https://aws.amazon.com/marketplace/pp/prodview-su2dajk5zawpo)** (14-day free trial).

---

## :material-arrow-right: After your first call

<div class="grid cards" markdown>

- :material-book-open-variant: [**API Overview**](api_overview.md) — Endpoints, parameters, and SDK usage
- :material-magnify: [**Search Models**](api_search_models.md) — Find the right model ID by modality, route, region, or streaming support
- :material-puzzle: [**Use Cases**](use_cases.md) — Open WebUI, n8n, coding assistants, and more
- :material-cog: [**Configuration**](operations_configuration.md) — Every environment variable and option
- :material-wrench: [**Troubleshooting**](operations_troubleshooting.md) — Common first-deployment errors and fixes
- :material-server-network: [**Advanced Deployment**](operations_deploy_advanced.md) — VPC integration, multi-region, cost optimization
- :material-scale-balance: [**Licensing**](operations_licensing.md) — Community (AGPL) vs Commercial

</div>
