---
title: "Configuration - HTTP Server and MCP"
description: "Configure the HTTP surface of stdapi.ai: route prefixes, CORS, trusted hosts, proxy headers, TLS, GZip, SSRF protection, request limits, the MCP server and the API documentation routes."
keywords: "route prefix, CORS configuration, trusted hosts, proxy headers, TLS configuration, GZip, SSRF protection, request limits, MCP server, OpenAPI documentation"
---

# :material-server: HTTP Server and MCP

The HTTP surface the gateway exposes: where the routes are mounted, who may call them from a browser or a proxy, and how long a request may take. Part of the [Configuration Guide](operations_configuration.md).

## :material-format-list-bulleted: Settings Summary

### :material-api: API Compatibility { #summary-api-compatibility }

| Variable                                              | Default      | Description                                          |
|-------------------------------------------------------|--------------|------------------------------------------------------|
| [`OPENAI_ROUTES_PREFIX`](#openai-routes-prefix)       | None (root)  | Base path prefix for OpenAI-compatible API routes    |
| [`ANTHROPIC_ROUTES_PREFIX`](#anthropic-routes-prefix) | `/anthropic` | Base path prefix for Anthropic-compatible API routes |
| [`COHERE_ROUTES_PREFIX`](#cohere-routes-prefix)       | `/cohere`    | Base path prefix for Cohere-compatible API routes    |
| [`OLLAMA_ROUTES_PREFIX`](#ollama-routes-prefix)       | `/ollama`    | Base path prefix for Ollama-compatible API routes    |

### :material-web: HTTP/Security { #summary-http-security }

| Variable                                                                            | Default  | Description                                                                           |
|-------------------------------------------------------------------------------------|----------|---------------------------------------------------------------------------------------|
| [`CORS_ALLOW_ORIGINS`](#cors-allow-origins)                                         | None     | JSON array of allowed origins for browser cross-origin requests                       |
| [`TRUSTED_HOSTS`](#trusted-hosts)                                                   | None     | JSON array of trusted Host header values (prefer ALB host-based routing; see details) |
| [`ENABLE_PROXY_HEADERS`](#enable-proxy-headers)                                     | `false`  | Trust X-Forwarded-* headers from reverse proxies (only enable behind trusted proxy)   |
| [`PROXY_TRUSTED_HOSTS`](#proxy-trusted-hosts)                                       | `*`      | Peer IPs/ranges whose X-Forwarded-* headers are trusted (restrict from `*` for safety) |
| [`GRANIAN_HOST`](operations_configuration.md#granian-host)                                                     | `0.0.0.0` | Listener bind address; `::` binds a dual-stack socket answering IPv4 and IPv6 clients |
| [`GRANIAN_SSL_CERTIFICATE`](#graniansslcertificate)                                 | None     | Path to SSL certificate file for end-to-end encryption                                |
| [`GRANIAN_SSL_KEYFILE`](#graniansslkeyfile)                                         | None     | Path to SSL private key file (PKCS#8) for end-to-end encryption                       |
| [`GRANIAN_SSL_KEYFILE_PASSWORD`](#graniansslkeyfilepassword)                        | None     | Password for the SSL private key file                                                 |
| [`GRANIAN_SSL_PROTOCOL_MIN`](#graniansslprotocolmin)                                | `tls1.3` | Minimum supported TLS version (`tls1.2` or `tls1.3`)                                  |
| [`GRANIAN_SSL_CA`](#graniansslca)                                                   | None     | Path to CA certificate bundle for client verification (mTLS)                          |
| [`GRANIAN_SSL_CLIENT_VERIFY`](#graniansslclientverify)                              | `false`  | Enable client certificate verification (mTLS)                                         |
| [`ENABLE_GZIP`](#enable-gzip)                                                       | `false`  | Enable GZip compression for responses >1KB (prefer AWS ALB/CloudFront compression)    |
| [`SSRF_PROTECTION_BLOCK_PRIVATE_NETWORKS`](#ssrf-protection-block-private-networks) | `true`   | Block requests to private/local networks for SSRF protection                          |
| [`MAX_INPUT_FILE_SIZE`](#max-input-file-size)                                       | `0`      | Maximum size in bytes of an inline input file loaded into memory (`0` disables)        |
| [`MAX_CONCURRENT_INPUT_DOWNLOADS`](#max-concurrent-input-downloads)                 | `8`      | Maximum input files fetched/resolved concurrently per request                          |

### :material-file-document: API Documentation { #summary-api-documentation }

| Variable                                      | Default | Description                                                                      |
|-----------------------------------------------|---------|----------------------------------------------------------------------------------|
| [`ENABLE_DOCS`](#enable-docs)                 | `false` | Enable interactive Swagger UI documentation at `/docs`                           |
| [`ENABLE_REDOC`](#enable-redoc)               | `false` | Enable ReDoc documentation UI at `/redoc`                                        |
| [`ENABLE_OPENAPI_JSON`](#enable-openapi-json) | `false` | Enable OpenAPI schema endpoint at `/openapi.json` (auto-enabled with docs/redoc) |

### :material-connection: MCP (Model Context Protocol) { #summary-mcp }

| Variable                                                            | Default | Description                                                                             |
|---------------------------------------------------------------------|---------|-----------------------------------------------------------------------------------------|
| [`ENABLE_MCP_STREAMABLE_HTTP`](#enable-mcp-streamable-http)         | `false` | Enable MCP server via Streamable HTTP at `/mcp` — recommended transport                 |
| [`MCP_STATELESS_HTTP`](#mcp-stateless-http)                         | `false` | Serve `/mcp` without server-side sessions — any replica may serve any request           |
| [`ENABLE_MCP_SSE`](#enable-mcp-sse)                                 | `false` | Enable MCP server via Server-Sent Events at `/sse` — legacy transport for older clients |
| [`MCP_INCLUDE_TOOLS`](#mcp-include-tools)                           | None    | Comma-separated tool names to expose exclusively; all others are hidden                 |
| [`MCP_EXCLUDE_TOOLS`](#mcp-exclude-tools)                           | None    | Comma-separated tool names to hide; all others remain exposed                           |

---

---

## :material-api: API Compatibility

Configure the base URL paths for OpenAI and Anthropic-compatible API routes.

#### `OPENAI_ROUTES_PREFIX` { #openai-routes-prefix }

:octicons-package-24: **Purpose**
:   Base path prefix for OpenAI-compatible API routes

:octicons-gear-24: **Default**
:   `` (empty, routes mounted at root)

:octicons-alert-24: **Requirement**
:   Empty, or a path starting with `/` with no trailing slash, using only alphanumeric characters and `. _ ~ -` per segment; must differ from every other routes prefix (`OPENAI_ROUTES_PREFIX`, `ANTHROPIC_ROUTES_PREFIX`, `COHERE_ROUTES_PREFIX`, `OLLAMA_ROUTES_PREFIX`)

:octicons-workflow-24: **Effect**
:   All OpenAI-compatible endpoints will be mounted under this prefix

```bash
export OPENAI_ROUTES_PREFIX=/api
```

!!! example "Example Endpoints"
    With the prefix `/api`, endpoints are available at:

    - `/api/v1/chat/completions`
    - `/api/v1/models`
    - `/api/v1/embeddings`

#### `ANTHROPIC_ROUTES_PREFIX` { #anthropic-routes-prefix }

:octicons-package-24: **Purpose**
:   Base path prefix for Anthropic-compatible API routes

:octicons-gear-24: **Default**
:   `/anthropic`

:octicons-alert-24: **Requirement**
:   A path starting with `/` with no trailing slash, using only alphanumeric characters and `. _ ~ -` per segment; must differ from every other routes prefix (`OPENAI_ROUTES_PREFIX`, `ANTHROPIC_ROUTES_PREFIX`, `COHERE_ROUTES_PREFIX`, `OLLAMA_ROUTES_PREFIX`)

:octicons-workflow-24: **Effect**
:   All Anthropic-compatible endpoints will be mounted under this prefix

```bash
export ANTHROPIC_ROUTES_PREFIX=/anthropic
```

!!! example "Example Endpoints"
    With the default prefix `/anthropic`, endpoints are available at:

    - `/anthropic/v1/messages`

!!! tip "Custom Prefix"
    You can change the prefix to match your organization's API structure:

    ```bash
    export ANTHROPIC_ROUTES_PREFIX=/api/anthropic
    ```

    This would mount the Messages API at `/api/anthropic/v1/messages`

#### `COHERE_ROUTES_PREFIX` { #cohere-routes-prefix }

:octicons-package-24: **Purpose**
:   Base path prefix for Cohere-compatible API routes

:octicons-gear-24: **Default**
:   `/cohere`

:octicons-alert-24: **Requirement**
:   A path starting with `/` with no trailing slash, using only alphanumeric characters and `. _ ~ -` per segment; must differ from every other routes prefix (`OPENAI_ROUTES_PREFIX`, `ANTHROPIC_ROUTES_PREFIX`, `COHERE_ROUTES_PREFIX`, `OLLAMA_ROUTES_PREFIX`)

:octicons-workflow-24: **Effect**
:   All Cohere-compatible endpoints will be mounted under this prefix

```bash
export COHERE_ROUTES_PREFIX=/cohere
```

!!! example "Example Endpoints"
    With the default prefix `/cohere`, endpoints are available at:

    - `/cohere/v2/rerank`

#### `OLLAMA_ROUTES_PREFIX` { #ollama-routes-prefix }

:octicons-package-24: **Purpose**
:   Base path prefix for Ollama-compatible API routes

:octicons-gear-24: **Default**
:   `/ollama`

:octicons-alert-24: **Requirement**
:   Empty, or a path starting with `/` with no trailing slash, using only alphanumeric characters and `. _ ~ -` per segment; must differ from every other routes prefix (`OPENAI_ROUTES_PREFIX`, `ANTHROPIC_ROUTES_PREFIX`, `COHERE_ROUTES_PREFIX`, `OLLAMA_ROUTES_PREFIX`)

:octicons-workflow-24: **Effect**
:   All Ollama-compatible endpoints will be mounted under this prefix

```bash
export OLLAMA_ROUTES_PREFIX=/ollama
```

!!! example "Example Endpoints"
    With the default prefix `/ollama`, endpoints are available at:

    - `/ollama/api/chat`
    - `/ollama/api/tags`

    Each API dialect this server speaks gets its own prefix by default — `/anthropic`, `/cohere`, and now `/ollama` — so the four cannot collide on the same path. This does not break Ollama clients: the official `ollama-python` and `ollama-js` clients both accept a path in their configured host, and tools built on them (Open WebUI, LlamaIndex) carry it through unchanged. Set this to an empty value to mount at the root instead, for a drop-in swap with a stock Ollama host.

---

## :material-web: CORS Configuration

Configure Cross-Origin Resource Sharing (CORS) to control which web origins can access your API from browsers.

#### `CORS_ALLOW_ORIGINS` { #cors-allow-origins }

:octicons-package-24: **Purpose**
:   List of origins allowed to make cross-origin requests

:octicons-list-ordered-24: **Format**
:   JSON array of origin URLs

:octicons-gear-24: **Default**
:   `None` (CORS not enabled)

:octicons-shield-check-24: **Best Practice**
:   Only enable if your API is accessed from web browsers; specify exact origins in production

```bash
# Not configured (default) - CORS middleware not enabled
# Browser cross-origin requests will be blocked
# No environment variable needed

# Development: Allow all origins
export CORS_ALLOW_ORIGINS='["*"]'

# Production: Specific origins only
export CORS_ALLOW_ORIGINS='["https://myapp.com", "https://app.example.com"]'

# Multiple environments
export CORS_ALLOW_ORIGINS='["https://app.example.com", "https://staging.example.com"]'
```

!!! info "What is CORS?"
    Cross-Origin Resource Sharing (CORS) is a browser security mechanism that restricts web pages from making requests to a different domain than the one serving the web page.

    **Without CORS enabled:**

    - Browser requests from web applications will fail due to missing CORS headers
    - Non-browser clients (curl, SDKs, mobile apps, server-to-server) work normally
    - Most secure default - no cross-origin access from browsers

    **With CORS enabled:**

    - Browsers can make requests from allowed origins
    - Preflight OPTIONS requests are handled automatically
    - Non-browser clients continue to work normally

!!! warning "Security Consideration"
    - **Default (not configured)**: CORS is disabled. Browser cross-origin requests will fail. This is the most secure default.
    - **`["*"]`**: Allows requests from any web origin. Convenient for development but not recommended for production.
    - **Specific origins**: Only allows requests from listed origins. Recommended for production.

!!! note "CORS Behavior"
    - When `CORS_ALLOW_ORIGINS` is not configured (default), CORS is **not enabled**
    - When configured with specific origins or `["*"]`, CORS is enabled with:
        - Authorization headers with credentials allowed
        - All HTTP methods allowed
        - All request headers allowed

!!! tip "When to Configure"
    Configure `CORS_ALLOW_ORIGINS` when:

    - :material-web: Your API is accessed from browser-based web applications (React, Vue, Angular, etc.)
    - :material-application-brackets: Building a web frontend that calls your API from a different domain
    - :material-dev-to: Developing locally with web apps (browser at `localhost:3000` calling API at `localhost:8000`)

!!! tip "When NOT to Configure"
    Do **not** configure CORS when:

    - :material-server: Your API is only accessed from server-to-server integrations
    - :material-cellphone: Your API is only accessed from mobile apps or desktop clients
    - :material-console: Your API is only accessed from CLI tools or SDKs
    - :material-api: Your API is only accessed from non-browser HTTP clients

    **Non-browser clients don't enforce CORS**, so enabling it is unnecessary overhead.

---

## :material-server-security: Trusted Host Configuration

Configure Host header validation to protect against Host header injection attacks.

#### `TRUSTED_HOSTS` { #trusted-hosts }

:octicons-package-24: **Purpose**
:   List of trusted Host header values for validation

:octicons-list-ordered-24: **Format**
:   JSON array of hostnames (supports wildcards)

:octicons-gear-24: **Default**
:   `None` (no Host header validation)

:octicons-shield-check-24: **Best Practice**
:   Use AWS ALB host-based routing rules instead when possible for better performance and management

```bash
# Production: Specific hosts only
export TRUSTED_HOSTS='["api.example.com", "www.example.com"]'
```

!!! info "What is Host Header Validation?"
    The Host header in HTTP requests specifies the domain name of the server. Validating it prevents **Host header injection attacks** (manipulated Host headers used to poison caches or exploit application logic) and **web cache poisoning**.

!!! warning "Security Consideration: prefer ALB host-based routing"
    Configure **AWS ALB listener rules** to validate the Host header and forward traffic only for approved hostnames — this rejects bad requests at the load balancer, before they reach the application, and is centrally managed. See the example below.

    Use `TRUSTED_HOSTS` only when you can't configure host-based routing at the load balancer level (no ALB, or you need application-level defense-in-depth).

!!! tip "Wildcard Support"
    - `*.example.com` matches any subdomain (`api.example.com`, `app.example.com`, ...)
    - `example.com` matches only the exact domain
    - `*` matches all hosts — not recommended, equivalent to no validation

!!! example "Common Configurations"

    **Multi-Domain with Subdomains:**

    ```bash
    export TRUSTED_HOSTS='["*.example.com", "*.myapp.com", "api.production.com"]'
    ```

    **Development and Production:**

    ```bash
    export TRUSTED_HOSTS='["api.example.com", "localhost", "127.0.0.1"]'
    ```

!!! note "Host Validation Behavior"
    - Not configured (default): Host header validation is **not enabled**
    - Configured: requests with a non-matching Host header are rejected with **HTTP 400 Bad Request**

!!! info "Container health probe"
    Validation applies to `/health` like any other path, so the container image's `HEALTHCHECK` derives its `Host` header from this setting: it requests `/health` on `127.0.0.1:$GRANIAN_PORT` announcing the **first** entry of `TRUSTED_HOSTS`. `*` or an unset value becomes `localhost`, and a leading `*.` becomes `healthcheck.` (so `*.example.com` is probed as `healthcheck.example.com`).

    A correct list therefore keeps the container healthy with no extra entry to add. Do not replace the probe with a hand-written `curl` call in a Compose `healthcheck:` block or an ECS task definition `healthCheck`: it would send an untrusted `Host` and get a `400`.

!!! warning "Load balancer health checks are rejected by default"
    An ALB or NLB target-group health check does **not** send your domain name: it addresses the target directly, so the `Host` header carries the target's IP address. With `TRUSTED_HOSTS` set to domain names, every one of those probes gets **HTTP 400**, the target never turns healthy, and the load balancer serves `503` — a failure that looks like a broken deployment rather than a configuration choice.

    Target-group health-check settings offer no `Host` header override, so either keep the Host allow-list at the load balancer (the recommended option above, leaving `TRUSTED_HOSTS` unset) or make sure the address the health check actually sends is in the list.

!!! success "AWS ALB Host-Based Routing Example"
    **Via AWS Console:** EC2 → Load Balancers → Your ALB → Listeners → add a rule on the HTTPS (443) listener with condition "Host header" is `api.example.com`, forwarding to the target group only on match.

    **Via AWS CLI:**

    ```bash
    aws elbv2 create-rule \
      --listener-arn arn:aws:elasticloadbalancing:... \
      --priority 1 \
      --conditions Field=host-header,Values=api.example.com \
      --actions Type=forward,TargetGroupArn=arn:aws:elasticloadbalancing:...
    ```

    Benefits: rejected at the load balancer (better performance, reduced load on application servers), centralized policy management, and ALB metrics/logging for rejected requests.

---

## :material-swap-horizontal: Proxy Headers Configuration

Configure X-Forwarded-* header processing when running behind reverse proxies or load balancers.

#### `ENABLE_PROXY_HEADERS` { #enable-proxy-headers }

:octicons-package-24: **Purpose**
:   Enable trusting X-Forwarded-* headers from reverse proxies

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false` (disabled)

:octicons-shield-check-24: **Best Practice**
:   Only enable when running behind a trusted reverse proxy

```bash
# Disabled (default) - do not trust X-Forwarded-* headers
# No environment variable needed

# Enable when behind reverse proxy
export ENABLE_PROXY_HEADERS=true
```

!!! info "What are X-Forwarded Headers?"
    When your application runs behind a reverse proxy (nginx, Apache, AWS ALB, CloudFront, etc.), the proxy sits between clients and your application. Without proxy header processing:

    - The application sees the proxy's IP address instead of the client's real IP
    - The application sees the proxy-to-app connection (e.g., HTTP) instead of the original client connection (e.g., HTTPS)
    - The application cannot distinguish between different clients behind the proxy

    Reverse proxies add `X-Forwarded-*` headers to preserve the original request information:

    - **X-Forwarded-For** - Client's real IP address (and chain of proxies)
    - **X-Forwarded-Proto** - Original protocol (http/https)
    - **X-Forwarded-Port** - Original port number

!!! warning "Security Warning"
    **CRITICAL**: Only enable `ENABLE_PROXY_HEADERS` when running behind a **trusted** reverse proxy that properly sets X-Forwarded-* headers.

    **If enabled without a trusted proxy:**

    - :material-alert: Clients can spoof their IP address by sending fake X-Forwarded-For headers
    - :material-shield-alert: Security controls based on client IP (rate limiting, allowlists) can be bypassed
    - :material-bug: Logging and monitoring will record incorrect client information
    - :material-lock-open: Authentication and authorization decisions may be affected

    **Never enable this setting if your application is directly exposed to the internet without a reverse proxy.**

!!! example "Common Deployment Scenarios"

    **Scenario 1: Direct to Internet (No Proxy)**

    ```bash
    # Do NOT enable proxy headers
    # ENABLE_PROXY_HEADERS should remain false (default)
    ```

    Your application receives requests directly from clients.

    **Scenario 2: Behind AWS ALB/CloudFront**

    ```bash
    export ENABLE_PROXY_HEADERS=true
    ```

    AWS load balancer or CDN forwards requests to your application.

    **Scenario 3: Multiple AWS Proxy Layers**

    ```bash
    export ENABLE_PROXY_HEADERS=true
    ```

    Example: CloudFront → ALB → Your Application

!!! note "Proxy Headers Behavior"
    - When `ENABLE_PROXY_HEADERS` is `false` (default), X-Forwarded-* headers are **not trusted**
    - When enabled, the server processes X-Forwarded-For, X-Forwarded-Proto, and X-Forwarded-Port headers to determine client information
    - Which peers' headers are trusted is controlled by [`PROXY_TRUSTED_HOSTS`](#proxy-trusted-hosts) — the default `*` trusts every peer, so restrict it to your reverse proxy's IP range

!!! tip "When to Enable"
    Enable `ENABLE_PROXY_HEADERS` when:

    - :material-aws: Deployed behind AWS ALB, NLB, API Gateway, or CloudFront
    - :material-network: Running behind any reverse proxy that sets X-Forwarded-* headers

!!! info "AWS Proxy Configuration"
    **AWS ALB, NLB, and CloudFront** automatically set X-Forwarded-* headers - no additional configuration needed.

    When you enable `ENABLE_PROXY_HEADERS=true`, your application will trust these headers to determine:

    - Client's real IP address (from X-Forwarded-For)
    - Original protocol (from X-Forwarded-Proto: http/https)
    - Original port (from X-Forwarded-Port)

#### `PROXY_TRUSTED_HOSTS` { #proxy-trusted-hosts }

:octicons-package-24: **Purpose**
:   Restrict which peer IPs may set trusted `X-Forwarded-*` headers when `ENABLE_PROXY_HEADERS` is enabled

:octicons-database-24: **Type**
:   JSON array of IPs/CIDRs, or `*`

:octicons-gear-24: **Default**
:   `*` (trust every peer — backward compatible)

:octicons-shield-check-24: **Best Practice**
:   Restrict to your reverse proxy's IP range so direct clients cannot spoof `X-Forwarded-For`

```bash
# Trust forwarded headers only from the VPC / proxy range
export ENABLE_PROXY_HEADERS=true
export PROXY_TRUSTED_HOSTS='["10.0.0.0/8"]'
```

!!! warning "Only effective with `ENABLE_PROXY_HEADERS=true`"
    This setting has no effect unless [`ENABLE_PROXY_HEADERS`](#enable-proxy-headers) is enabled. With the default `*`, any client that can reach the server directly can forge `X-Forwarded-For`, poisoning the client IP recorded in logs and OpenTelemetry spans. Restrict it to the address range of your load balancer or reverse proxy (AWS ALB/CloudFront, nginx, etc.).

!!! tip "Configured automatically by the official Terraform module"
    The [stdapi-ai Terraform module](https://github.com/stdapi-ai/terraform-aws-stdapi-ai) sets this for you when the ALB is enabled with client IP logging (`alb_enabled = true`, `log_client_ip = true`): it enables proxy headers and pins `PROXY_TRUSTED_HOSTS` to the ALB's subnet CIDRs, so only the load balancer is trusted and direct clients cannot forge `X-Forwarded-For`. Override it with the module's `proxy_trusted_hosts` variable when fronting the ALB with an additional proxy (for example CloudFront).

!!! warning "On a dual-stack listener, cover the IPv4-mapped form too"
    With `GRANIAN_HOST=::` the operating system reports an IPv4 peer as an IPv4-mapped IPv6 address such as `::ffff:10.0.1.5`, which belongs to no IPv4 network and therefore matches no IPv4 entry here. Add the mapped range alongside the plain one — an IPv4 `/16` becomes a `/112` once the 96-bit mapping prefix is counted:

    ```bash
    export PROXY_TRUSTED_HOSTS='["10.0.0.0/16", "::ffff:10.0.0.0/112"]'
    ```

    Miss it and the proxy stops being trusted: `X-Forwarded-For` is ignored and the load balancer's own address is recorded as the client IP. The Terraform module derives these entries for you, including for values passed to its `proxy_trusted_hosts` variable.

---

## :material-certificate: TLS / SSL Configuration

Configure end-to-end TLS encryption within the container. These are native [Granian](https://github.com/emmett-framework/granian) environment variables and are available with the provided container images.

#### `GRANIAN_SSL_CERTIFICATE` { #graniansslcertificate }
:octicons-package-24: **Purpose**
:   Path to the SSL certificate file

:octicons-database-24: **Type**
:   File path

#### `GRANIAN_SSL_KEYFILE` { #graniansslkeyfile }
:octicons-package-24: **Purpose**
:   Path to the SSL private key file (PKCS#8 format only)

:octicons-database-24: **Type**
:   File path

#### `GRANIAN_SSL_KEYFILE_PASSWORD` { #graniansslkeyfilepassword }
:octicons-package-24: **Purpose**
:   Password for the private key file

:octicons-database-24: **Type**
:   String

#### `GRANIAN_SSL_PROTOCOL_MIN` { #graniansslprotocolmin }
:octicons-package-24: **Purpose**
:   Minimum supported TLS version (`tls1.2` or `tls1.3`)

:octicons-database-24: **Type**
:   Enum

:octicons-gear-24: **Default**
:   `tls1.3`

#### `GRANIAN_SSL_CA` { #graniansslca }
:octicons-package-24: **Purpose**
:   Path to the CA certificate bundle used to verify client certificates (mTLS)

:octicons-database-24: **Type**
:   File path

#### `GRANIAN_SSL_CLIENT_VERIFY` { #graniansslclientverify }
:octicons-package-24: **Purpose**
:   Enable client certificate verification (mTLS)

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false` (disabled)

---

## :material-zip-box: GZip Compression

Configure automatic GZip compression for HTTP responses to reduce bandwidth usage and improve response times.

#### `ENABLE_GZIP` { #enable-gzip }

:octicons-package-24: **Purpose**
:   Enable GZip compression for HTTP responses

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false` (disabled)

:octicons-zap-24: **Best Practice**
:   Use AWS ALB or CloudFront compression instead when available for better performance

```bash
# Disabled (default) - no response compression
# No environment variable needed

# Enable GZip compression (responses larger than 1 KiB will be compressed)
export ENABLE_GZIP=true
```

!!! info "How GZip Compression Works"
    When enabled, the server automatically:

    1. :material-file-check: Checks if the response size exceeds 1 KiB (1024 bytes)
    2. :material-web: Verifies the client supports compression (via `Accept-Encoding: gzip` header)
    3. :material-zip-box: Compresses the response body using gzip
    4. :material-arrow-down: Adds `Content-Encoding: gzip` header to the response

    Typical compression ratios for JSON responses: **60-80% size reduction**

!!! success "Recommended: Use AWS Compression Services"
    Instead of enabling application-level compression, enable compression at the AWS layer — it offloads the CPU cost from your application servers, at the price of managing it in AWS instead of a single environment variable:

    - **AWS ALB** — enable the `compression.enabled` target group attribute ([documentation](https://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-target-groups.html#compression))
    - **Amazon CloudFront** — enable "Compress Objects Automatically" in the distribution behavior settings ([documentation](https://docs.aws.amazon.com/AmazonCloudFront/latest/DeveloperGuide/ServingCompressedFiles.html))

!!! tip "When to Enable Application-Level Compression"
    Enable `ENABLE_GZIP` only when:

    - :material-server-off: You're **not** using AWS ALB or CloudFront
    - :material-wan: Your API returns large JSON responses and you want to reduce bandwidth
    - :material-dev-to: Local development or non-AWS deployments

!!! warning "When NOT to Enable"
    Do **not** enable when:

    - :material-aws: You're behind AWS ALB with compression enabled
    - :material-cloud: You're using CloudFront with compression enabled
    - :material-speedometer-slow: CPU usage is a concern (compression adds CPU overhead)

    **Enabling compression at multiple layers is redundant and wastes CPU resources.**

!!! note "Compression Behavior"
    - When `ENABLE_GZIP` is `false` (default), compression is **not enabled**
    - When enabled, only responses meeting these criteria are compressed:
        - Response size ≥ 1 KiB (1024 bytes)
        - Client sends `Accept-Encoding: gzip` header
        - Response does not already have `Content-Encoding` header
    - Streaming responses are compressed on-the-fly

---

## :material-connection: MCP (Model Context Protocol)

When enabled, stdapi.ai exposes its API endpoints as MCP tools, allowing AI clients and agents to call them directly using the Model Context Protocol. The full list of available tool names is documented in [API Overview → MCP Tools](api_overview.md#mcp-model-context-protocol).

Both transport types can be enabled independently or simultaneously.

#### `ENABLE_MCP_STREAMABLE_HTTP` { #enable-mcp-streamable-http }

:octicons-package-24: **Purpose**
:   Enable the MCP server using Streamable HTTP transport — the recommended method

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false`

:octicons-workflow-24: **Behavior**
:   Exposes an MCP-compatible endpoint at `/mcp`. AI clients connect using standard HTTP requests following the MCP Streamable HTTP specification.

```bash
# Disabled (default)
# No environment variable needed

# Enable MCP Streamable HTTP transport
export ENABLE_MCP_STREAMABLE_HTTP=true
```

#### `MCP_STATELESS_HTTP` { #mcp-stateless-http }

:octicons-package-24: **Purpose**
:   Serve the Streamable HTTP transport without server-side sessions

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false`

:octicons-workflow-24: **Behavior**
:   Each request to `/mcp` is handled by a fresh transport that keeps no state. Clients may call `tools/list` and `tools/call` without an `initialize` handshake, an `Mcp-Session-Id` the server never issued is accepted rather than rejected, and any replica may serve any request.

:octicons-alert-24: **Requires**
:   `ENABLE_MCP_STREAMABLE_HTTP=true`. Ignored otherwise.

```bash
# Sessions enabled (default)
# No environment variable needed

# Stateless transport
export ENABLE_MCP_STREAMABLE_HTTP=true
export MCP_STATELESS_HTTP=true
```

#### `ENABLE_MCP_SSE` { #enable-mcp-sse }

:octicons-package-24: **Purpose**
:   Enable the MCP server using Server-Sent Events (SSE) transport

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false`

:octicons-workflow-24: **Behavior**
:   Exposes MCP endpoints at `/sse` for AI clients that require the SSE transport protocol.

```bash
# Disabled (default)
# No environment variable needed

# Enable MCP SSE transport
export ENABLE_MCP_SSE=true
```

!!! info "Transport Recommendation"
    **HTTP transport (`ENABLE_MCP_STREAMABLE_HTTP`) is the recommended method.** It implements the latest MCP Streamable HTTP specification and provides better session management and more robust connection handling.

    **SSE transport (`ENABLE_MCP_SSE`)** is maintained for backwards compatibility with older MCP client implementations. Prefer HTTP for new deployments.

    Both transports can be enabled simultaneously to support clients with different requirements:

    ```bash
    export ENABLE_MCP_STREAMABLE_HTTP=true
    export ENABLE_MCP_SSE=true
    ```

    The MCP server card (`/.well-known/mcp/server-card.json`) declares a single transport: Streamable HTTP (`/mcp`) whenever it is enabled, otherwise SSE (`/sse`). When both are enabled, `/sse` is therefore not listed in the card, but it remains fully functional for clients configured with it explicitly.

#### `MCP_INCLUDE_TOOLS` { #mcp-include-tools }

:octicons-package-24: **Purpose**
:   Expose only a specific subset of MCP tools; all others are hidden

:octicons-code-24: **Format**
:   Comma-separated list of tool names (duplicates are automatically removed)

:octicons-gear-24: **Default**
:   None (all tools exposed)

```bash
# All tools exposed by default
# No environment variable needed

# Expose only specific tools
export MCP_INCLUDE_TOOLS="openai_chat_completion,openai_embedding,search_models"

# When both MCP_INCLUDE_TOOLS and MCP_EXCLUDE_TOOLS are specified,
# tools in MCP_EXCLUDE_TOOLS are removed from MCP_INCLUDE_TOOLS:
export MCP_INCLUDE_TOOLS="openai_chat_completion,openai_embedding,search_models"
export MCP_EXCLUDE_TOOLS="openai_files_delete,anthropic_files_delete"
# Result: only openai_chat_completion, openai_embedding, search_models are exposed
```

See [API Overview → MCP Tools](api_overview.md#mcp-model-context-protocol) for the full list of available tool names.

!!! warning "Token Usage for Complex API Tools"
    `anthropic_message`, `openai_chat_completion`, and `openai_response` map to large, complex APIs that may use many tokens (prompt, completion, and tool definitions). Select these tools only if your workflow requires the full API capabilities.

#### `MCP_EXCLUDE_TOOLS` { #mcp-exclude-tools }

:octicons-package-24: **Purpose**
:   Hide specific MCP tools from clients; all others remain exposed

:octicons-code-24: **Format**
:   Comma-separated list of tool names (duplicates are automatically removed)

:octicons-gear-24: **Default**
:   None (no tools excluded)

!!! note "Behavior with `MCP_INCLUDE_TOOLS`"
    When both `MCP_INCLUDE_TOOLS` and `MCP_EXCLUDE_TOOLS` are specified, tools in `MCP_EXCLUDE_TOOLS` are removed from `MCP_INCLUDE_TOOLS`. The remaining tools in `MCP_INCLUDE_TOOLS` are what get exposed.

```bash
# No tools excluded by default
# No environment variable needed

# Exclude destructive tools
export MCP_EXCLUDE_TOOLS="openai_files_delete,anthropic_files_delete"
```

See [API Overview → MCP Tools](api_overview.md#mcp-model-context-protocol) for the full list of available tool names.

### Tool Selection Best Practices { #tool-selection-best-practices }

stdapi.ai exposes a fixed set of tools derived from its API surface — you can include or exclude them by name, but cannot modify or rename them. See [API Overview → MCP Tools](api_overview.md#mcp-model-context-protocol) for the full catalog.

**Start from the minimum, not the maximum**

By default all tools are exposed. It is safer and more effective to begin with a narrow `MCP_INCLUDE_TOOLS` list covering only what the workflow needs, then expand it deliberately. LLMs perform better with fewer choices, and many AI providers cap the number of active tools per session.

**Always include `search_models` for agent model discovery**

`search_models` is the recommended tool for agents to discover available model IDs — it supports capability-based filtering (by modality, route, region, streaming support) and returns richer metadata than `openai_model_list` or `anthropic_model_list`. Include it in every agent configuration so the agent can resolve the right model dynamically rather than relying on hardcoded IDs:

```bash
export MCP_INCLUDE_TOOLS="openai_chat_completion,search_models,openai_embedding"
```

**Always exclude file deletion tools unless required**

Uploaded files are the only durable, stateful data managed by stdapi.ai — deletion is permanent and cannot be undone. Unless your workflow explicitly needs to delete files, always suppress these tools:

```bash
export MCP_EXCLUDE_TOOLS="openai_files_delete,anthropic_files_delete"
```

**Exclude high-cost tools unless the workflow requires them**

Image generation (`openai_image_generation`, `openai_image_edit`, `openai_image_variation`) and speech synthesis (`openai_audio_speech`) incur a per-call cost that accumulates quickly if an agent invokes them speculatively. Only include them when the use case calls for it and the agent's decision to generate images or audio is intentional.

**Use `MCP_INCLUDE_TOOLS` for the tightest control**

For predictable, well-defined workflows, listing tools explicitly with `MCP_INCLUDE_TOOLS` is more reliable than maintaining an exclusion list. For example, a workflow limited to text generation and model discovery needs only:

```bash
export MCP_INCLUDE_TOOLS="openai_chat_completion,search_models"
```

!!! note
    Health and metadata endpoints are never exposed as MCP tools, so they do not need to be listed in `MCP_EXCLUDE_TOOLS`.

---

## :material-shield-alert: SSRF Protection

Configure Server-Side Request Forgery (SSRF) protection to prevent unauthorized access to internal networks.

#### `SSRF_PROTECTION_BLOCK_PRIVATE_NETWORKS` { #ssrf-protection-block-private-networks }

:octicons-package-24: **Purpose**
:   Enable SSRF protection by blocking requests to private/local networks

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `true` (enabled for security)

:octicons-shield-check-24: **Best Practice**
:   Keep enabled in production to protect against SSRF attacks

```bash
# Enabled (default) - block private networks
# No environment variable needed

# Disable only in controlled environments that need local network access
export SSRF_PROTECTION_BLOCK_PRIVATE_NETWORKS=false
```

!!! info "What is SSRF Protection?"
    Server-Side Request Forgery (SSRF) is an attack where an attacker can make the server send requests to unintended destinations, including internal network resources.

    **SSRF protection has two layers:**

    1. **Baseline Protection (Always Enabled)** - Cannot be disabled:
        - :material-restart: **Loopback Addresses** - 127.0.0.0/8, ::1
        - :material-network-off: **Unspecified Addresses** - 0.0.0.0, ::
        - :material-link: **Link-Local Addresses** - 169.254.0.0/16, fe80::/10
        - :material-network-off: **Reserved IP Ranges** - IETF reserved addresses
        - :material-network-off: **Multicast Addresses** - Multicast IP ranges

    2. **Private Network Protection (Controlled by this setting):**
        - :material-ip: **Every Non-Globally-Reachable Address** - anything outside the public Internet address space, in both families and in IPv4-mapped IPv6 form
        - :material-ip: **Examples** - RFC 1918 (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16), IPv6 unique local (fc00::/7), RFC 6598 shared address space (100.64.0.0/10), benchmarking (198.18.0.0/15) and documentation ranges

!!! warning "Security Warning"
    **CRITICAL**: Only disable `SSRF_PROTECTION_BLOCK_PRIVATE_NETWORKS` in controlled environments where accessing internal networks is explicitly required and safe.

    **If disabled, private network protection is removed:**

    - :material-alert: Attackers may be able to reach any non-globally-reachable address (private networks, shared address space, and the other special-purpose ranges) through your API
    - :material-shield-alert: Internal services on private networks (databases, admin panels, internal APIs) may be exposed
    - :material-lock-open: Internal APIs without authentication may be exploited

    **Important**: Even when disabled, baseline protection remains active and prevents access to:

    - :material-check: Loopback addresses (127.0.0.1, localhost) - **always blocked**
    - :material-check: Link-local addresses (169.254.x.x) including AWS EC2 metadata endpoint - **always blocked**
    - :material-check: Reserved and multicast addresses - **always blocked**

!!! tip "When to Disable"
    Disable `SSRF_PROTECTION_BLOCK_PRIVATE_NETWORKS` only when:

    - :material-lan: Your application legitimately needs to access internal network resources
    - :material-dev-to: Local development environment where accessing localhost services is required
    - :material-shield-check: You have other security controls in place (network segmentation, firewall rules)
    - :material-docker: Running in isolated Docker/container environments with restricted network access

!!! success "Defense in Depth"
    Even with SSRF protection enabled, implement additional security measures:

    - :material-network-strength-4: **Network Segmentation** - Isolate application servers from sensitive internal networks
    - :material-wall-fire: **Firewall Rules** - Restrict outbound connections from application servers
    - :material-security: **Security Groups** - Use AWS security groups to limit network access
    - :material-monitor: **Monitoring** - Log and monitor outbound requests for suspicious patterns

---

## :material-speedometer-slow: Request Limits

Bound per-request resource usage to protect the server when the API is exposed to untrusted clients.

#### `MAX_INPUT_FILE_SIZE` { #max-input-file-size }

:octicons-package-24: **Purpose**
:   Cap the size of an inline input file loaded into memory to protect against memory-exhaustion (DoS)

:octicons-database-24: **Type**
:   Integer (bytes)

:octicons-gear-24: **Default**
:   `0` (disabled — no limit)

:octicons-shield-check-24: **Best Practice**
:   Set a limit aligned with your largest expected inline input (e.g. `26214400` for 25 MiB) when the API is exposed to untrusted clients

```bash
# Disabled (default) - no size limit
# No environment variable needed

# Reject inline inputs larger than 25 MiB
export MAX_INPUT_FILE_SIZE=26214400
```

!!! info "What is limited"
    The limit applies to file content that is **loaded into memory** for model input:

    - :material-file-code: Base64 and `data:` URI inputs
    - :material-download: HTTP(S) and S3 sources downloaded and read for model input
    - :material-database-arrow-up: [Attachments too large to travel inside a request](features.md#attachment-size), on the size their source declares, before they are staged

    Requests exceeding the limit are rejected with **HTTP 413** before the content is fully decoded or downloaded. For downloads, the body is streamed and aborted as soon as the limit is exceeded, so a spoofed `Content-Length` cannot bypass it.

    **Streaming uploads are not affected**, so large file transfers remain possible:

    - :material-cloud-upload: Multipart form uploads
    - :material-file-move: Files API ingest from HTTP(S) URLs and S3-to-S3 copies

#### `MAX_CONCURRENT_INPUT_DOWNLOADS` { #max-concurrent-input-downloads }

:octicons-package-24: **Purpose**
:   Bound the number of input files fetched or resolved concurrently within a single request

:octicons-database-24: **Type**
:   Integer (> 0)

:octicons-gear-24: **Default**
:   `8`

:octicons-shield-check-24: **Best Practice**
:   Keep a modest value so a single request with many remote inputs cannot exhaust sockets/memory or amplify outbound requests against a target

```bash
# Allow up to 4 concurrent input downloads per request
export MAX_CONCURRENT_INPUT_DOWNLOADS=4
```

!!! info "Behaviour"
    Each remote input (image, document, or audio referenced by URL or S3 URI) is fetched in parallel, capped at this many at a time. Excess inputs queue and run as slots free up, so requests still complete — they are only paced. This prevents a request carrying thousands of URLs from opening thousands of simultaneous connections (socket/memory exhaustion and SSRF amplification).

---

## :material-file-document: API Documentation Routes

stdapi.ai provides automatic API documentation routes, which are **disabled by default** for security in production environments.

!!! warning "Security Consideration"
    Exposing API documentation routes in production can reveal internal API structure, available endpoints, and request/response schemas to potential attackers. Only enable these routes in development/testing environments or when absolutely necessary.

!!! info "Agent Discovery"
    The machine-readable API catalog at `/.well-known/api-catalog` (RFC 9727 Linkset) is always served, regardless of the settings below. Enabling a route adds its entry to the catalog:

    - `ENABLE_OPENAPI_JSON` — adds the `service-desc` link to `/openapi.json`
    - `ENABLE_DOCS` or `ENABLE_REDOC` — adds the `service-doc` link to `/docs` or `/redoc` (Swagger UI takes precedence when both are enabled)
    - [`ENABLE_MCP_STREAMABLE_HTTP`](#enable-mcp-streamable-http) or [`ENABLE_MCP_SSE`](#enable-mcp-sse) — adds the `mcp-server-card` link

    The same links are also advertised as RFC 8288 `Link` headers on the root endpoint (`/`). That header is only emitted when at least one of these routes is enabled; with all of them disabled, the catalog is still reachable but carries no links.

#### `ENABLE_DOCS` { #enable-docs }

:octicons-package-24: **Purpose**
:   Enable interactive Swagger UI documentation at `/docs`

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false` (disabled)

```bash
# Enable for development
export ENABLE_DOCS=true
```

!!! info "Interactive Documentation Features"
    The `/docs` endpoint provides an interactive interface to:

    - Browse all available API endpoints
    - Test API requests directly from the browser
    - View request/response schemas
    - Understand parameter requirements

#### `ENABLE_REDOC` { #enable-redoc }

:octicons-package-24: **Purpose**
:   Enable ReDoc documentation UI at `/redoc`

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false` (disabled)

```bash
# Enable for development
export ENABLE_REDOC=true
```

!!! info "ReDoc Features"
    The `/redoc` endpoint provides a clean, responsive documentation interface with:

    - Three-panel layout for easy navigation
    - Enhanced schema visualization
    - Better rendering for complex APIs
    - Export to OpenAPI specification

!!! success "Works with no outbound access"
    Both pages are served entirely by the gateway: the container image ships Swagger UI and ReDoc itself, pinned to an exact release and verified against a recorded SHA-256 during the build, alongside the icon and the schema. A browser that can reach the gateway renders them, with no request to any CDN, font host or other third party — so they work unchanged in an air-gapped VPC, behind an egress allow-list, or under a strict content security policy.

!!! tip "Static Documentation Available"
    ReDoc API documentation is also available as static documentation at [API Reference](api_reference.md) without requiring this endpoint to be enabled.

#### `ENABLE_OPENAPI_JSON` { #enable-openapi-json }

:octicons-package-24: **Purpose**
:   Enable OpenAPI schema JSON endpoint at `/openapi.json`

:octicons-database-24: **Type**
:   Boolean

:octicons-gear-24: **Default**
:   `false` (disabled)

```bash
# Enable for development
export ENABLE_OPENAPI_JSON=true
```

!!! info "OpenAPI Schema"
    The `/openapi.json` endpoint provides the raw OpenAPI 3.0 specification, useful for:

    - Generating API clients in various languages
    - Import into API testing tools (Postman, Insomnia)
    - API documentation generation
    - Contract testing and validation

!!! note "Automatic Enablement"
    If either `ENABLE_DOCS` or `ENABLE_REDOC` is set to `true`, the `/openapi.json` endpoint will be automatically enabled since both documentation UIs require the OpenAPI schema to function. You only need to explicitly set `ENABLE_OPENAPI_JSON=true` if you want to expose the schema endpoint without enabling the documentation UIs.

### Development Configuration

**Enable all documentation routes for local development:**

```bash
export ENABLE_DOCS=true
export ENABLE_REDOC=true
# ENABLE_OPENAPI_JSON is automatically enabled when ENABLE_DOCS or ENABLE_REDOC is true
```

**Or enable only Swagger UI:**

```bash
export ENABLE_DOCS=true
# ENABLE_OPENAPI_JSON is automatically enabled
```

**Or enable only ReDoc:**

```bash
export ENABLE_REDOC=true
# ENABLE_OPENAPI_JSON is automatically enabled
```

### Production Best Practice

```bash
# Keep all routes disabled in production (default)
# No environment variables needed - defaults to false
```

!!! danger "Production Warning"
    **Never enable these routes in production** unless you have specific security controls in place (e.g., IP allowlisting, VPN-only access, or additional authentication layer).

---

## :material-timer-sand: AI Response Timeout { #ai-response-timeout-section }

#### `AI_RESPONSE_TIMEOUT` { #ai-response-timeout }

:octicons-package-24: **Purpose**
:   Maximum time in seconds to wait without receiving any data from an AI model

:octicons-database-24: **Type**
:   Integer (seconds, must be greater than 0)

:octicons-gear-24: **Default**
:   `600` (10 minutes)

:octicons-workflow-24: **Behavior**
:   Inactivity (per-read) timeout on the upstream model connection, applied to both streaming and non-streaming requests. The timer resets every time data is received, so it fires only when the model stalls for longer than this value — it does **not** bound the total duration of a response: a stream that keeps producing chunks can run well past it. On a non-streaming request, where the whole response arrives at once, it effectively bounds the wait for that single response. When it fires, the connection is closed and the request fails with a timeout error

```bash
# Default (10 minutes) - suitable for extended thinking models
export AI_RESPONSE_TIMEOUT=600

# Shorter timeout for standard models (2 minutes)
export AI_RESPONSE_TIMEOUT=120

# Longer timeout for very long documents or high reasoning budgets (15 minutes)
export AI_RESPONSE_TIMEOUT=900
```

!!! tip "When to Adjust"
    - **Increase** if you see timeout errors with models that use extended thinking/reasoning, large document analysis, or high token budgets
    - **Decrease** to fail fast and free resources if your workload only uses standard models where long waits indicate a problem

!!! info "Extended Thinking Models"
    Models with extended reasoning capabilities (such as Claude with `thinking` enabled or high `reasoning_effort`) may spend significant time generating internal reasoning steps before producing output. The default of 600 seconds accommodates these use cases. Standard models without extended thinking typically respond within 60 seconds.

---

## :material-power-plug-off: Shutdown Drain { #shutdown-drain-section }

#### `SHUTDOWN_DRAIN_TIMEOUT` { #shutdown-drain-timeout }

:octicons-package-24: **Purpose**
:   Maximum time in seconds the server waits for background work to finish after it has been asked to stop

:octicons-database-24: **Type**
:   Number (seconds, `0` or greater)

:octicons-gear-24: **Default**
:   `10`

:octicons-workflow-24: **Behavior**
:   Some work is deliberately started outside the request that asked for it, so the caller is answered without waiting for it: temporary file cleanups, vector store file indexing, and the release of live audio sessions. On a stop signal the server waits up to this long for that work to finish, then cancels whatever is still running. The wait is a single deadline shared by all of it, not a budget per item, and a server with nothing outstanding stops immediately

```bash
# Default: comfortably inside a 30-second container stop timeout
export SHUTDOWN_DRAIN_TIMEOUT=10

# Longer wait, with the container stop timeout raised to match
export SHUTDOWN_DRAIN_TIMEOUT=20

# No wait: cancel background work immediately and stop as fast as possible
export SHUTDOWN_DRAIN_TIMEOUT=0
```

!!! warning "Best effort, not a delivery guarantee"
    A container runtime sends `SIGKILL` a fixed delay after the stop signal — 30 seconds by default on [Amazon ECS](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task_definition_parameters.html#container_definition_timeout) — so the server can be killed before the wait ends, and a deployment may run under an orchestrator that stops it sooner still. Keep this value comfortably below your container stop timeout, and raise it only together with that timeout. Never rely on this wait for anything whose completion matters: retry the operation instead.

!!! tip "When Work Is Lost"
    Anything cancelled at the deadline is counted in the server's `stop` log event, which is then emitted at `warning` level with one count per kind of work. Deployments that see those counts regularly are stopping the server faster than its work can finish: raise this value and the container stop timeout together, or reduce what each request defers.

---
