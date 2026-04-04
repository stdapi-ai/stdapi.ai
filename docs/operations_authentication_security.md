---
title: Authentication & Security
description: Configuration guide for authenticating and securing requests to stdapi.ai — API keys via SSM/Secrets Manager, external authentication via AWS ALB/API Gateway, and encryption in transit.
keywords: API authentication, API key, AWS SSM, Secrets Manager, AWS ALB authentication, API Gateway authorizer, OIDC, Cognito, stdapi.ai security
---

# :material-lock: Authentication & Security

stdapi.ai provides flexible authentication options and built-in security mechanisms to protect your API. Choose from API key authentication for simple deployments to enterprise-grade identity management via AWS services — all backed by built-in SSRF protection, host header validation, and configurable encryption in transit.

<div class="grid cards" markdown>

- :material-key: __API Key Authentication__
  <br>Securely stored in AWS SSM Parameter Store or Secrets Manager. Mimics OpenAI and Anthropic auth.

- :material-account-check: __OIDC, Cognito & IAM Identity Center__
  <br>Offload user and workforce identity management to AWS ALB or API Gateway.

- :material-shield-key: __AWS IAM__
  <br>Use AWS native access control via API Gateway.

- :material-lock-open: __No Authentication__
  <br>Optional mode for private/internal VPC deployments where security is handled at the network level.

</div>

---

## :material-key-chain: Authentication Methods

| Method | Best for | AWS infrastructure | Enforced by |
|---|---|---|---|
| **API Key** | Server-to-server, simple deployments | Any | stdapi.ai |
| **OIDC / Cognito / IAM Identity Center** | User-facing apps, SSO, workforce identity | ALB or API Gateway | AWS (before request reaches stdapi.ai) |
| **AWS IAM** | Service-to-service within AWS | API Gateway | AWS (SigV4) |
| **No Authentication** | Local development, trusted VPC | Any | Network controls only |

### :material-key-star: API Key Authentication

The API key authentication method uses a single API key that mimics the behavior of upstream OpenAI and Anthropic authentication. It must be provided by clients in the `Authorization: Bearer <key>` or `X-API-Key` header.

Three mutually exclusive sources are supported — configure exactly one:

- **SSM Parameter Store** (`API_KEY_SSM_PARAMETER`) — the parameter must already exist before startup.
- **Secrets Manager** (`API_KEY_SECRETSMANAGER_SECRET`) — the secret must already exist; supports a configurable key within the JSON secret via `API_KEY_SECRETSMANAGER_KEY`.
- **Direct value** (`API_KEY`) — for local development and testing only; not recommended for production.

For the full list of environment variables and required IAM permissions for each method, see the [Configuration Guide](operations_configuration.md#authentication).

!!! abstract "In-Memory Key Protection"
    To ensure maximum security, stdapi.ai never stores your API key in plain text. At startup, the key is securely retrieved, salted, and hashed using a cryptographically strong algorithm. Only the resulting hash is used for verification, protecting your secret even in the event of a memory dump.

!!! info "Terraform Module"
    The Terraform module offers additional options for providing the API key:

    - **`api_key`** — provide the key directly; it is injected as an **ECS Secret** (sourced from an encrypted SSM parameter with KMS), never exposed in task definitions or logs.
    - **`api_key_create`** — auto-generate a secure 64-character random key; the generated value is returned as a sensitive Terraform output.
    - **`api_key_ssm_parameter`** / **`api_key_secretsmanager_secret`** — reference an existing SSM parameter or Secrets Manager secret; the module does not create these resources.

### :material-account-check: OIDC, Cognito & IAM Identity Center

For user-facing applications or enterprise SSO, you can offload authentication to an OpenID Connect (OIDC) provider, Amazon Cognito, or AWS IAM Identity Center (the AWS-native workforce SSO service for centralised employee and partner access).

When authentication is handled at this layer, requests are fully validated **before they reach stdapi.ai** — the application receives only authenticated, pre-authorised requests.

!!! warning "Terraform Module"
    The Terraform module does not configure OIDC, Cognito, or IAM Identity Center authentication. These integrations must be set up directly on the ALB or API Gateway.

#### via Application Load Balancer (ALB)
The AWS ALB can authenticate users before forwarding requests to stdapi.ai. This is the most common way to add OIDC or Cognito authentication.

-   **Capabilities**: Integrates with any OIDC-compliant identity provider — including **AWS IAM Identity Center** (workforce SSO), **Amazon Cognito** User Pools, or third-party providers such as Okta, Auth0, and Google.
-   **IAM Identity Center**: Expose the Identity Center OIDC application endpoints as the ALB OIDC configuration. This gives employees and partners seamless SSO using their AWS-managed identity, with support for MFA and permission sets defined in your AWS Organization.
-   **Documentation**: [Authenticate users using an Application Load Balancer](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/listener-authenticate-users.html)

#### via API Gateway
Both REST and HTTP APIs support OIDC and Cognito integration.

-   **HTTP API (JWT)**: Native support for JWT-based authorization — any OIDC provider (including IAM Identity Center configured as a trusted token issuer) can be used.
    -   [Controlling access to HTTP APIs with JWT](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-access-control.html)
-   **REST API**: Supports Cognito User Pool authorizers and Lambda Authorizers for custom authentication logic.
    -   [Control access to a REST API in API Gateway](https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-control-access-to-api.html)

### :material-shield-key: AWS IAM

For service-to-service communication within AWS or when using IAM roles for access control, you can use AWS IAM authentication via API Gateway. Clients sign their requests using the AWS Signature Version 4 (SigV4) process, enabling fine-grained access control using IAM policies attached to users or roles.

#### via API Gateway
Amazon API Gateway provides native IAM authentication for both REST and HTTP APIs.

-   **Documentation**:
    -   [IAM authorization for REST APIs](https://docs.aws.amazon.com/apigateway/latest/developerguide/apigateway-control-access-to-api.html)
    -   [IAM authorization for HTTP APIs](https://docs.aws.amazon.com/apigateway/latest/developerguide/http-api-access-control.html)

### :material-lock-open: No Authentication

In certain environments, you may choose to disable application-level authentication and rely entirely on network security.

#### When to Use
- Local development or testing using [Docker](operations_getting_started_local.md).
- Internal VPC deployments where traffic is trusted.
- When an upstream proxy handles all security and identity.

#### Configuration
To disable API key authentication, do **not** configure any API key environment variables (`API_KEY`, `API_KEY_SSM_PARAMETER`, or `API_KEY_SECRETSMANAGER_SECRET`).

When no API key authentication is configured, stdapi.ai will:

1.  Accept all incoming requests without validating an API key.
2.  Log a security warning at startup.

!!! danger "Network Security Requirement"
    When API key authentication is disabled, you **must** ensure strict network security. Use Security Groups to restrict access so that the stdapi.ai service only accepts traffic from your trusted Load Balancer or API Gateway.

---

## :material-shield-outline: Application Security

stdapi.ai includes built-in security mechanisms that are active regardless of the authentication method chosen.

!!! abstract "Hardened Container Image"
    The container images provided via the AWS Marketplace are security-hardened with a minimal base containing no unnecessary system tools or packages. The Terraform module further enforces a **read-only root filesystem** and **drops all Linux capabilities** from the ECS task definition, reducing the attack surface to the minimum required for operation.

    [:octicons-arrow-right-24: Subscribe on AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-su2dajk5zawpo)

### :material-shield-alert-outline: SSRF Protection

Users can pass URLs to the API as multimodal content references (images, documents, audio). The application fetches the content at those URLs before forwarding it to the model. Without protection, a malicious user could supply URLs pointing to internal services — such as the AWS EC2/ECS metadata endpoint or private network resources — and read sensitive data through the model response.

stdapi.ai validates every user-supplied URL via DNS resolution before fetching its content — each hostname is resolved and every resulting IP is checked against the blocklist. This protects against DNS rebinding attacks where a seemingly-safe domain resolves to a blocked address.

-   **Baseline Protection**: Always-active blocking of loopback (127.0.0.1, localhost), link-local (169.254.169.254), and reserved address ranges. This prevents access to the AWS EC2/ECS metadata service from within the container.
-   **Private Network Blocking**: By default, the service also blocks all RFC 1918 private networks (10.x.x.x, 172.16.x.x, 192.168.x.x). Disable only in controlled environments where accessing local networks is required (`SSRF_PROTECTION_BLOCK_PRIVATE_NETWORKS=false`).
-   **Defense in Depth**: Even with SSRF protection enabled, restrict outbound Security Group rules to only the necessary AWS service endpoints.

### :material-web-check: Host Header Validation

To protect against Host header injection and web cache poisoning, stdapi.ai can validate the `Host` header of incoming requests. When `TRUSTED_HOSTS` is not configured, no Host header validation is performed.

-   **Recommended Approach**: Use **AWS ALB host-based routing rules** to reject invalid Host headers before they reach the application. This is more performant and centrally managed.
-   **Application Validation**: Use `TRUSTED_HOSTS` to define a list of approved hostnames (supports wildcards like `*.example.com`). Requests with non-matching headers are rejected with an HTTP 400 error.

### :material-web: Browser Security (CORS)

If your API is accessed directly from web browsers, Cross-Origin Resource Sharing (CORS) must be configured to prevent unauthorized cross-site requests. When `CORS_ALLOW_ORIGINS` is not configured, the CORS middleware is not enabled and browser cross-origin requests are blocked by default.

-   **Principle of Least Privilege**: Only allow specific, trusted domains using `CORS_ALLOW_ORIGINS`.
-   **Offload to Infrastructure**: For production, consider managing CORS at the **AWS ALB** or **API Gateway** level for better performance and to keep your application logic clean.

### :material-swap-horizontal: Proxy Headers

When running behind a load balancer (ALB) or reverse proxy, stdapi.ai can be configured to trust `X-Forwarded-*` headers (`X-Forwarded-For`, `X-Forwarded-Proto`, `X-Forwarded-Port`) to correctly identify the original client IP and protocol. When enabled, the real client IP appears in request logs instead of the proxy's IP.

-   **Security Warning**: Only enable `ENABLE_PROXY_HEADERS` when the service is deployed behind a **trusted proxy**. Enabling this in a publicly exposed environment without a proxy allows attackers to spoof their IP address by sending custom headers.

---

## :material-shield-check: Security Best Practices

**Network Isolation:**

- :material-check: **Deploy in private subnets** — without direct internet access.
- :material-check: **Restrict security group inbound rules** — allow traffic to the stdapi.ai service only from the ALB or API Gateway security group on the application port.
- :material-check: **Apply strict egress rules** — limit outbound traffic to only the required AWS services (Bedrock, S3, etc.).
- :material-check: **Use VPC endpoints** — set `vpc_endpoints_allowed=true` (the default) to keep all traffic to Bedrock, S3, SSM, CloudWatch Logs, and AI services off the public internet.

**WAF & Rate Limiting:**

- :material-check: **Enable AWS WAF** — the Terraform module supports WAF via `alb_waf_enabled=true`. When enabled, it activates the AWS Managed Core Rule Set (SQLi, XSS, common exploits), Linux Rule Set, and IP Reputation List (known malicious IPs).
- :material-check: **Configure rate limiting** — set `alb_waf_rate_limit` to limit the number of requests per IP per 5-minute window, protecting against abuse and quota exhaustion.
- :material-check: **Block anonymous IPs** — set `alb_waf_block_anonymous_ips=true` to block known VPNs, open proxies, and Tor exit nodes.

**Secrets Management:**

- :material-check: **Use SSM Parameter Store or Secrets Manager** — always use encrypted secret storage for production API keys; never pass keys as plain environment variables.
- :material-check: **Rotate API keys** — Secrets Manager supports [automated secret rotation](https://docs.aws.amazon.com/secretsmanager/latest/userguide/rotating-secrets.html) via Lambda; note that stdapi.ai reads the key once at startup, so a container restart is required after rotation.

**IAM & Access Control:**

- :material-check: **Apply least-privilege IAM permissions** — ensure the ECS Task Role has only the [required permissions](operations_configuration.md#iam-permissions) needed for operation.

!!! info "Terraform Module"
    The Terraform module enforces security defaults out of the box: **least-privilege IAM** roles scoped to the minimum permissions required by each ECS task, **KMS-encrypted** SSM parameters, log groups, and S3 buckets, and **network isolation** with ECS tasks in private subnets and Security Group rules that allow only ALB-to-service traffic.

---

## :material-shield-sync: Encryption in Transit

To ensure data protection during transmission, stdapi.ai supports several encryption-in-transit configurations.

| Mode | TLS scope | Config required | Typical use case |
|---|---|---|---|
| **Standard** | Entry point only | None | Default for most AWS deployments |
| **End-to-End** | Entry point → container | `GRANIAN_SSL_CERTIFICATE` + `GRANIAN_SSL_KEYFILE` | Compliance-mandated full-path encryption |
| **mTLS at Container** | Entry point → container, mutual | Above + `GRANIAN_SSL_CA` + `GRANIAN_SSL_CLIENT_VERIFY` | Mutual authentication between ALB and container |
| **mTLS at Entry Point** | Client → entry point, mutual | ALB or API Gateway configuration | Client certificate enforcement before traffic reaches the application |

### :material-shield-half-full: Standard

This is the default configuration when no additional encryption settings are applied. TLS is terminated at the entry point (ALB or API Gateway). Traffic between the entry point and the stdapi.ai service travels over HTTP within the private VPC, secured by mandatory Security Group rules. This is the most common configuration for AWS deployments.

!!! info "Terraform Module"
    The Terraform module configures the ALB with the `ELBSecurityPolicy-TLS13-1-2-Res-PQ-2025-09` security policy by default (`alb_ssl_policy`), enforcing a minimum of TLS 1.2 and enabling TLS 1.3 with post-quantum hybrid key exchange. See [Data Sovereignty & Compliance](operations_compliance.md) for details on ALB TLS configuration and post-quantum key exchange support.

### :material-lock-check: End-to-End

If your compliance requirements mandate encryption all the way to the container, you can enable TLS between the load balancer and the container.

-   **Native Support**: The provided container images run [Granian](https://github.com/emmett-framework/granian) natively. TLS can be enabled directly by setting the following Granian environment variables:
    -   `GRANIAN_SSL_CERTIFICATE`: Path to the SSL certificate file (e.g., `/etc/ssl/certs/server.crt`).
    -   `GRANIAN_SSL_KEYFILE`: Path to the SSL private key file (**PKCS#8 format only**, e.g., `/etc/ssl/private/server.key`).
    -   `GRANIAN_SSL_KEYFILE_PASSWORD`: (Optional) Password for the private key file.
    -   `GRANIAN_SSL_PROTOCOL_MIN`: (Optional) Minimum supported TLS version — `tls1.2` or `tls1.3` (defaults to `tls1.3`).
-   **Mutual TLS — Container**: As an advanced form of end-to-end encryption, you can also require the upstream caller (e.g., the ALB) to present a certificate to the container.
    -   `GRANIAN_SSL_CA`: Path to the CA certificate bundle used to verify client certificates.
    -   `GRANIAN_SSL_CLIENT_VERIFY`: Set to `true` to enable client certificate verification.
-   **ECS Certificate Mounting**: When running on AWS ECS, certificate and key files must be mounted into the container. This is typically achieved using **Amazon EFS** volumes or by baking the certificates into a custom image (not recommended for secrets).
-   **Certificate Management**: Use ACM Private CA or a self-signed certificate for the internal service. Note that the ALB does not validate the backend certificate by default; it only ensures the connection is encrypted.
-   **Sidecar Proxy**: Alternatively, you can run a sidecar container (like Nginx or Envoy) alongside stdapi.ai to handle TLS termination locally.

!!! warning "Terraform Module Support"
    The provided Terraform module **does not currently support** automatic configuration of end-to-end encryption. You will need to manually configure the ALB target group for HTTPS and manage certificate provisioning.

### :material-certificate-outline: Mutual TLS — Client to Entry Point

For environments requiring mutual authentication between the final client and the entry point:

-   **Offloaded mTLS**: mTLS can be enabled at the **ALB** (using Application Load Balancer Mutual TLS) or **API Gateway** (using Mutual TLS authentication). In this setup, the entry point validates the client certificate before forwarding the request.
-   **Service Mesh / Sidecar**: You can implement mTLS between internal services using a Service Mesh (like AWS App Mesh) or by using a sidecar proxy (like Envoy) that manages certificate rotation and verification.

---

## :material-arrow-right: Next Steps

<div class="grid cards" markdown>

- :material-cog: [**Configuration Reference**](operations_configuration.md) — Detailed environment variable list
- :material-shield-lock: [**Data Sovereignty & Compliance**](operations_compliance.md) — How we protect your data
- :material-server-network: [**Advanced Deployment**](operations_deploy_advanced.md) — Terraform infrastructure examples

</div>
