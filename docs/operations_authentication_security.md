---
title: Authentication & Security
description: Configuration guide for authenticating and securing requests to stdapi.ai — API keys via SSM/Secrets Manager, per-tenant API keys with model and endpoint scopes, Amazon Cognito user pool tokens, OAuth 2.0 discovery for AI agents, external authentication via AWS ALB/API Gateway, and encryption in transit.
keywords: API authentication, API key, AWS SSM, Secrets Manager, tenant API keys, multi-tenant API gateway, scoped API keys, model allow list, API key revocation, Amazon Cognito user pool, JWT, bearer token, AWS ALB authentication, API Gateway authorizer, OIDC, OAuth 2.0 discovery, protected resource metadata, WWW-Authenticate, MCP client authentication, stdapi.ai security
---

# :material-lock: Authentication & Security

stdapi.ai provides flexible authentication options and built-in security mechanisms to protect your API. Choose from API key authentication for simple deployments to enterprise-grade identity management via AWS services — all backed by built-in SSRF protection, host header validation, and configurable encryption in transit.

<div class="grid cards" markdown>

- :material-key: __API Key Authentication__
  <br>Securely stored in AWS SSM Parameter Store or Secrets Manager. Mimics OpenAI and Anthropic auth.

- :material-key-multiple: __Tenant API Keys__
  <br>One key per customer or team, scoped to models and endpoints, minted server-side and revocable within a minute.

- :material-account-key: __Amazon Cognito User Pool Tokens__
  <br>Per-user and per-application bearer tokens, validated on every request. No key to rotate.

- :material-compass-outline: __Authentication Discovery for Agents__
  <br>Publish where tokens come from, so an agent authenticates itself with no prior configuration.

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
| **[Tenant API keys](#tenant-api-keys)** | Serving several customers or teams from one deployment, each key scoped and revocable | DynamoDB table | stdapi.ai |
| **Amazon Cognito user pool token** | Per-user access, autonomous agents, revocable credentials | Cognito user pool | stdapi.ai |
| **OIDC / Cognito / IAM Identity Center** | User-facing apps, SSO, workforce identity | ALB or API Gateway | AWS (before request reaches stdapi.ai) |
| **AWS IAM** | Service-to-service within AWS | API Gateway | AWS (SigV4) |
| **No Authentication** | Local development, trusted VPC | Any | Network controls only |

The API key, tenant API keys and Cognito tokens can be used together: [`AUTHENTICATION_MODE`](operations_configuration.md#authentication-mode) selects which of them a deployment accepts, and defaults to accepting every method that is configured.

When stdapi.ai enforces the credential itself — the API key, tenant API keys and Cognito user pool tokens — a rejected request is answered with `401 Unauthorized`, a `WWW-Authenticate: Bearer` challenge, and a body that states nothing beyond `Unauthorized` — the reason is written to the server log only, so the response cannot be used to work out which half of a credential was wrong. That challenge is also where an AI agent starts: see [Authentication Discovery for Agents](#authentication-discovery-for-agents). The edge-enforced methods answer an unauthenticated request themselves, before it reaches stdapi.ai, with whatever their own configuration says — commonly a redirect to the identity provider.

### :material-key-star: API Key Authentication

The API key authentication method uses a single API key that mimics the behavior of upstream OpenAI and Anthropic authentication. It must be provided by clients in the `Authorization: Bearer <key>` or `X-API-Key` header.

Three sources are supported — configure exactly one. If more than one is set, the first match in this precedence order is used (the others are ignored): **Direct value → SSM Parameter Store → Secrets Manager**.

- **Direct value** (`API_KEY`) — for local development and testing only; not recommended for production.
- **SSM Parameter Store** (`API_KEY_SSM_PARAMETER`) — the parameter must already exist before startup.
- **Secrets Manager** (`API_KEY_SECRETSMANAGER_SECRET`) — the secret must already exist; supports a configurable key within the JSON secret via `API_KEY_SECRETSMANAGER_KEY`.

For the full list of environment variables and required IAM permissions for each method, see the [Configuration Guide](operations_configuration.md#authentication).

!!! abstract "In-Memory Key Protection"
    stdapi.ai never stores API keys in plain text. At startup, the key is retrieved, salted, and hashed using an industry-standard cryptographic function; only the hash is retained in memory — even a full memory dump cannot reconstruct the original key. Verification uses constant-time comparison to prevent timing-based side-channel attacks.

!!! info "Terraform Module"
    The Terraform module offers additional options for providing the API key:

    - **`api_key`** — provide the key directly; it is injected as an **ECS Secret** (sourced from an encrypted SSM parameter with KMS), never exposed in task definitions or logs.
    - **`api_key_create`** — auto-generate a secure 64-character random key; the generated value is returned as a sensitive Terraform output.
    - **`api_key_ssm_parameter`** / **`api_key_secretsmanager_secret`** — reference an existing SSM parameter or Secrets Manager secret; the module does not create these resources.

### :material-account-key: Amazon Cognito User Pool Tokens

stdapi.ai can accept the bearer tokens issued by an [Amazon Cognito user pool](https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools.html) as an alternative to the static API key, validating each one on every request. Each caller — a person or an application — gets its own short-lived credential, so there is no shared key to rotate, and the verified caller becomes the identity [per-user cost attribution](operations_cost_management.md#per-user-attribution) bills against.

Clients send the token exactly as they would send an API key, in either header:

```bash
curl https://your-gateway.example.com/v1/models \
  -H "Authorization: Bearer eyJraWQiOi..."

# Anthropic-compatible routes, whose SDK uses its own header
curl https://your-gateway.example.com/anthropic/v1/models \
  -H "x-api-key: eyJraWQiOi..." -H "anthropic-version: 2023-06-01"
```

#### What you configure in AWS

1. **A user pool**, in any Region. Its ID (for example `eu-west-3_a1b2c3d4e`) goes into [`AWS_COGNITO_USER_POOL_ID`](operations_configuration.md#aws-cognito-user-pool-id); the Region is read from the ID itself, and so is the issuer URL — nothing else names it, here or in [agent discovery](#authentication-discovery-for-agents).
2. **One or more app clients** in that pool. Their IDs go into [`AWS_COGNITO_CLIENT_IDS`](operations_configuration.md#aws-cognito-client-ids) — a token issued to any other app client is rejected, so this list is required.
3. **A [pool domain](https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-assign-domain.html)**, which is what gives the pool its OAuth 2.0 authorization and token endpoints. It is needed in two cases: to demand a scope such as `stdapi/invoke` through [`AWS_COGNITO_REQUIRED_SCOPES`](operations_configuration.md#aws-cognito-required-scopes), since custom scopes exist only on tokens those endpoints issue and on a resource server that defines them; and for the [agent discovery flow](#authentication-discovery-for-agents), scope or no scope, since an agent has nowhere to obtain a token otherwise. Machine-to-machine clients use the `client_credentials` grant there.

No IAM permission is involved: the pool's signing keys are public. The task must be able to **reach** them, though — it reads them over HTTPS from `cognito-idp.<region>.amazonaws.com` in the pool's Region at startup, and fails to start if it cannot. Give the task outbound HTTPS to that host: NAT or internet egress, or an [interface VPC endpoint](https://docs.aws.amazon.com/cognito/latest/developerguide/vpc-interface-endpoints.html) for `com.amazonaws.<region>.cognito-idp` in that Region — note that AWS declares a pool with a domain assigned incompatible with that endpoint, so a pool used for scopes or for agent discovery needs the egress path.

!!! danger "The pool decides who can call the gateway"
    Any identity that can obtain a token from an app client listed in [`AWS_COGNITO_CLIENT_IDS`](operations_configuration.md#aws-cognito-client-ids) can call the API, and its model usage is billed to your account. A Cognito user pool **allows self sign-up by default**, so pointing the gateway at a customer-facing pool without further restriction turns it into an open, self-service one.

    Before enabling it: disable self-registration on the pool (`AllowAdminCreateUserOnly`), or dedicate an app client to the gateway and require one of its resource-server scopes.

#### What stdapi.ai validates

Every request is checked against all of the following, and any failure returns the same `401 Unauthorized` with no detail (the reason is recorded in the server log only):

| Check | Requirement |
|---|---|
| **Signature** | RS256, against the pool's published keys. Unsigned (`alg=none`) and symmetric (`HS*`) tokens are always rejected. |
| **Issuer** | Exactly the configured pool's issuer — see [`AWS_COGNITO_ISSUER_TYPE`](operations_configuration.md#aws-cognito-issuer-type). A token from another pool is rejected. |
| **Token use** | An access token. Identity tokens are rejected unless [`AWS_COGNITO_ACCEPT_ID_TOKEN`](operations_configuration.md#aws-cognito-accept-id-token) is enabled. |
| **Application** | The token's app client is in [`AWS_COGNITO_CLIENT_IDS`](operations_configuration.md#aws-cognito-client-ids). |
| **Validity period** | Not expired, and not used before its start time, with a one-minute tolerance for clock drift. |
| **Scopes** | All of [`AWS_COGNITO_REQUIRED_SCOPES`](operations_configuration.md#aws-cognito-required-scopes) are present. |

!!! abstract "Signing keys are loaded once, at startup"
    The pool's public keys are read at startup and kept in memory, so validation adds no network call and no measurable latency to a request. When a pool rotates its keys, the first request carrying a token signed by the new key reloads them, at most once every five minutes — a forged key identifier cannot turn requests into outbound traffic. A deployment that cannot read the keys at startup **fails to start** rather than serving requests it could not authenticate.

!!! warning "Revocation takes effect when the token expires"
    A token is validated against the pool's published keys and its own claims, without calling the pool, so nothing is checked back with it once it has been issued. Disabling a user, deleting them or signing them out therefore stops the **next** token, not the one already in their hands: that one keeps working until its `exp`, up to the app client's access-token validity (one hour by default, 24 hours at most). Keep that validity short — it is the upper bound on how long a revoked credential stays usable.

!!! warning "Username and password sign-in yields no custom scope"
    Tokens obtained by signing in directly against the user pool API carry the single scope `aws.cognito.signin.user.admin`. Custom scopes only exist on tokens issued by the pool's OAuth 2.0 token endpoint. Requiring `stdapi/invoke` therefore rejects every client that signs in with a username and password — leave [`AWS_COGNITO_REQUIRED_SCOPES`](operations_configuration.md#aws-cognito-required-scopes) empty unless all your clients use the OAuth 2.0 endpoints.

!!! tip "Both methods, or one"
    With both a user pool and an API key configured, either credential is accepted: a bearer value shaped like a signed token is validated against the pool, anything else is compared to the API key. Set [`AUTHENTICATION_MODE`](operations_configuration.md#authentication-mode) to `cognito` or `api_key` to accept only one of them — the deployment then refuses to start if the other one is configured too, so a credential is never accepted by accident.

### :material-key-multiple: Tenant API Keys { #tenant-api-keys }

One deployment can serve several customers or teams, each holding an API key of its own — shaped `sk-std-<key id>-<secret>` — that is validated on every request and scoped to the models and endpoints its tenant is entitled to. Enable the method with [`TENANT_API_KEYS`](operations_configuration.md#tenant-api-keys); it needs the [shared DynamoDB table](operations_configuration.md#aws-dynamodb-table) and a [delivery prefix](operations_configuration.md#tenant-key-ssm-parameter-prefix). Clients send the key like any API key, in the `Authorization: Bearer <key>` or `X-API-Key` header. The deployment-wide API key and Cognito tokens keep working unchanged alongside it — enabling tenant keys changes nothing for existing credentials.

#### Declaring tenants and receiving their keys

A tenant is a record in the shared table: its name, its scopes, and a `disabled` flag. The [Terraform module](operations_getting_started.md#quick-start) declares one per entry of its `tenants` variable:

```hcl
tenants = {
  "acme" = {
    models_allow    = ["anthropic.*", "amazon.nova-lite-v1:0"]
    endpoints_allow = ["/v1/chat/completions", "/v1/models"]
  }
  "globex" = {
    models_deny = ["*opus*"]
  }
}
```

Scopes bound the models a tenant may invoke and the endpoints it may call; they do not partition stored objects, which stay deployment-wide ([the caveat below](#scopes)).

**The key secret never enters Terraform state.** Terraform owns the tenant record only; the server notices a declared tenant that has no credential yet — at startup and once a minute — mints a 256-bit secret for it, stores only a salted hash in the table, and delivers the full key exactly once as an SSM `SecureString` parameter named `<prefix>/<key id>` (the module's `tenant_keys` output gives the exact name per tenant). Retrieve it, hand it to the tenant, then delete the parameter:

```bash
aws ssm get-parameter --name /my-deployment/tenant-keys/AbC123... \
  --with-decryption --query Parameter.Value --output text
```

!!! abstract "Only a hash is ever stored"
    The table holds a salted BLAKE2b-256 digest of the secret, compared in constant time on every request — the same in-memory protection the deployment API key gets. Neither the table, nor Terraform state, nor a backup of either can reconstruct a tenant's key; the only copy is the delivered parameter, which is yours to delete after delivery.

!!! warning "The delivery prefix is a trust boundary in both directions"
    The parameter is created once and never overwritten, which is what makes minting idempotent across instances — so **whoever creates it defines the secret**. A principal able to call `ssm:PutParameter` under the prefix can therefore pre-create `<prefix>/<key id>` for a tenant that does not exist yet and hold a valid key from the moment it is declared, exactly as read access there exposes the keys already delivered. Grant both actions on `<prefix>/*` to the deployment's task role and to the operators who collect the keys, and to nothing else.

    Read access is wider than that grant while [`TENANT_KEY_SSM_KMS_KEY_ID`](operations_configuration.md#tenant-key-ssm-kms-key-id) is unset: the parameter is then encrypted with the AWS-managed `alias/aws/ssm` key, whose key policy lets **any principal of the account** decrypt through Parameter Store, so `ssm:GetParameter` on the path is the only permission an intruder needs. Point the setting at a key of your own and reading a delivered key also requires `kms:Decrypt` on it — the [Terraform module](operations_getting_started.md#quick-start) points it at the deployment's KMS key automatically.

!!! info "Without the Terraform module"
    Any tool that can write a DynamoDB item can declare a tenant: `pk` = `TENANT`, `sk` = `tenant#<key id>` (a key ID is 16 letters and digits of your choice, unique per tenant), attributes `name` (string), `schema` (number, `1`), optional `disabled` (boolean) and the four scope lists below as lists of strings. The server mints and delivers the key the same way.

#### Scopes

Each tenant record may carry four pattern lists, matched with `*` and `?` globs. A list that is absent restricts nothing; a list that is present and **empty allows nothing**; a deny match always wins over an allow match.

| List | Matched against | A refused request answers |
|---|---|---|
| `models_allow` / `models_deny` | The **resolved** model ID, after aliases, wildcards and deprecation fallbacks — an alias cannot launder a denied model | The standard `404` `model_not_found`, indistinguishable from a model that does not exist |
| `endpoints_allow` / `endpoints_deny` | The matched route's path template, e.g. `/v1/chat/completions` or `/v1/files/{file_id}` | The same detail-free `401 Unauthorized` as any refused credential |

!!! tip "Deny lists match names, so prefer an allow list"
    Model patterns are matched against the model ID the request resolves to. With [`AWS_BEDROCK_ALLOW_MARKETPLACE_ENDPOINT_ARN`](operations_configuration.md#bedrock-allow-marketplace-endpoint-arn) enabled, an endpoint addressed by its ARN keeps that ARN as its ID, which a name-based pattern such as `mistral.*` does not match. `models_allow` fails closed on it — no pattern matches, so the model is refused — while a deny-only tenant would reach it: add `arn:*` to `models_deny` where that opt-in is on.

Both are enforced at choke points every request passes through — the authentication dependency for endpoints, the single model-resolution step for models — not per route, so a new endpoint or model route cannot bypass them. `GET /v1/models` is deliberately **not** filtered per tenant: the catalogue advertises what the deployment serves, and the invocation-time check is the authority. A Realtime client secret minted with a tenant key stays bound to that tenant: the session it opens carries the same scopes and stops resuming once the key is revoked or disabled.

!!! warning "Scopes bound what a tenant may invoke, not what it may read"
    **Stored objects carry no tenant ownership.** Files, vector stores, batches and their result files, conversations, stored responses, and the Anthropic files and message batches are all deployment-wide: neither list filters them, so a tenant allowed on `/v1/files` lists, downloads and deletes what every other tenant uploaded. The model and endpoint scopes bound what a key may *invoke* and *call*; they are not a data boundary.

    Where tenants must not reach one another's data, deny the storing endpoints and keep the deployment stateless for them:

    ```hcl
    endpoints_deny = [
      "*/v1/files*",
      "*/v1/vector_stores*",
      "*/v1/batches*",
      "*/v1/messages/batches*",
      "*/v1/conversations*",
      "*/v1/responses/*",
    ]
    ```

    The leading `*` covers the routes prefix each dialect is mounted under, such as [`ANTHROPIC_ROUTES_PREFIX`](operations_configuration.md#anthropic-routes-prefix). `/v1/responses` itself stays allowed so the Responses API keeps serving requests; a response the tenant asks to `store` is still written, only no longer readable by it. Mutually untrusted tenants that need any of these features want one deployment each — a separate bucket and table is the only isolation there is.

#### Validation, caching and revocation

Validation is a direct read of the tenant's two records, cached in each server instance for [`TENANT_KEY_CACHE_SECONDS`](operations_configuration.md#tenant-key-cache-seconds) — 60 seconds by default. That cache is the revocation window: **a key that is revoked, disabled or re-scoped keeps its previous decision for up to 60 seconds per instance**, and no longer. Revoke a key by removing its tenant from `tenants` (destroying the record), or suspend it by setting `disabled = true`. Unknown key IDs are negative-cached, bounded in size and time, so a flood of fabricated keys neither amplifies table reads nor grows memory.

!!! warning "The table being unreachable fails closed"
    When tenant keys are enabled but the DynamoDB table cannot be read, a tenant-shaped credential is refused with `503` — never accepted, and never turned into a `401` that would mislabel a valid key as wrong. The reason (the IAM action, the table) is written to the server log. Other credential kinds are unaffected.

!!! tip "A tenant key and a user token together"
    A request may carry both `X-API-Key: sk-std-...` (the tenant key) and `Authorization: Bearer <token>` (a Cognito user token): **both are then verified** — the tenant key authorizes and scopes the request, the token identifies the user for [per-user cost attribution](operations_cost_management.md#per-user-attribution). This is the one place tenant keys extend the header rules: for every other combination, `X-API-Key` keeps winning exactly as before. Without a user, the tenant's key ID is the identity the request is attributed to.

### :material-compass-outline: Authentication Discovery for Agents

An AI agent that meets an API it has never been configured for has one thing to go on: the `401 Unauthorized` it just received. stdapi.ai answers that request with everything the agent needs to obtain a credential on its own — the standard discovery flow behind MCP client authentication.

**Name the public URL and the token issuer, and the discovery surface turns on:**

| Setting | Value |
|---|---|
| [`OAUTH_RESOURCE_IDENTIFIER`](operations_configuration.md#oauth-resource-identifier) | The public URL clients dial, exactly as they dial it — for example `https://api.example.com` |
| [`OAUTH_AUTHORIZATION_SERVERS`](operations_configuration.md#oauth-authorization-servers) | The issuer URL of whatever issues your tokens. Leave it unset with a user pool configured: its issuer is published for you |
| [`OAUTH_SCOPES_SUPPORTED`](operations_configuration.md#oauth-scopes-supported) | Leave it unset too — [`AWS_COGNITO_REQUIRED_SCOPES`](operations_configuration.md#aws-cognito-required-scopes) is published, since an agent that is not told which scope to ask for obtains a token without it and is refused on every retry |

!!! success "With a user pool, one variable turns discovery on"
    The pool named by [`AWS_COGNITO_USER_POOL_ID`](operations_configuration.md#aws-cognito-user-pool-id) issues the tokens this deployment accepts, so it *is* the authorization server clients must be sent to, and the scopes it requires are the scopes they must ask for. Both are published from it, and the issuer is exactly what the pool signs into `iss`:

    ```text
    https://cognito-idp.<region>.amazonaws.com/<pool-id>
    https://issuer-cognito-idp.<region>.amazonaws.com/<pool-id>   # AWS_COGNITO_ISSUER_TYPE=updated
    ```

    `<region>` and `<pool-id>` are read from the pool ID itself, and the host follows that Region's AWS partition — `amazonaws.com.cn` in China, `amazonaws.eu` in the European Sovereign Cloud. Set the two variables only for an issuer no pool can supply, such as an identity-aware proxy in front of the deployment; the pool's own issuer must stay in the list, or startup fails rather than sending clients to an authorization server whose tokens are refused.

**What an agent then sees:**

```console
$ curl -i https://api.example.com/v1/models
HTTP/1.1 401 Unauthorized
WWW-Authenticate: Bearer resource_metadata="https://api.example.com/.well-known/oauth-protected-resource", scope="stdapi/invoke"

$ curl https://api.example.com/.well-known/oauth-protected-resource
{
  "resource": "https://api.example.com",
  "authorization_servers": ["https://cognito-idp.eu-west-3.amazonaws.com/eu-west-3_a1b2c3d4e"],
  "scopes_supported": ["stdapi/invoke"],
  "bearer_methods_supported": ["header"],
  "resource_name": "stdapi.ai (Community Edition)",
  "resource_documentation": "https://stdapi.ai/api_reference/"
}
```

From there the agent reads the authorization server's own metadata, signs in, and retries the request with a bearer token. The document is public and unauthenticated by design — it is read by a client that has no credential yet — and contains nothing beyond the issuer URL you configured. It is also cacheable, and it is advertised from the [API catalog and the root `Link` header](features.md#agent-discovery), so an agent that starts from either entry point finds it without a 401.

!!! info "Both the API and the MCP server are covered"
    The identifier names the deployment's origin, which every surface shares: an MCP client connecting to `/mcp` or `/sse`, and an SDK calling `/v1/…`, all match the same published resource. Only one document is served, at the root.

!!! warning "`/.well-known/openid-configuration` is deliberately not served"
    stdapi.ai is a **resource server**, not an authorization server: it validates tokens, it does not issue them. Mirroring or redirecting the authorization server's own discovery document would create a second, staler copy of something the issuer already publishes authoritatively. Agents reach it through `authorization_servers` instead — which is exactly what the standard flow does.

!!! warning "Amazon Cognito needs a pre-registered app client"
    A Cognito user pool publishes neither a dynamic client registration endpoint nor client-id metadata document support, so an agent cannot register itself. Create the app client in the pool and give the agent its client ID (and secret, for confidential clients) ahead of time. Discovery still saves the agent everything else: the issuer, the endpoints, and the scopes.

    The pool also needs a [domain](https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-assign-domain.html): its authorization and token endpoints exist only once one is assigned, so without it the issuer's own discovery document names no endpoint the agent can obtain a token from, and the flow dead-ends after the 401.

!!! tip "Browser-hosted clients need CORS"
    A client running in a browser reads this document cross-origin. Add its origin to [`CORS_ALLOW_ORIGINS`](operations_configuration.md#cors-allow-origins), which it also needs for the API calls that follow.

    With a CORS origin configured, the gateway also marks `WWW-Authenticate` as readable cross-origin, so a page-hosted client reads the metadata URL and the advertised scope straight off the `401`. Without one, the browser hides that header and the client falls back to the default well-known path, losing the scope the challenge carries.

### :material-account-check: OIDC, Cognito & IAM Identity Center

For user-facing applications or enterprise SSO, you can offload authentication to an OpenID Connect (OIDC) provider, Amazon Cognito, or AWS IAM Identity Center (the AWS-native workforce SSO service for centralized employee and partner access).

When authentication is handled at this layer, requests are fully validated **before they reach stdapi.ai** — the application receives only authenticated, pre-authorized requests. This remains a valid alternative to [validating Cognito tokens in stdapi.ai](#amazon-cognito-user-pool-tokens): choose the edge when you need a hosted sign-in flow or session cookies, and choose in-application validation when clients already hold a bearer token and you want the verified caller to drive [per-user cost attribution](operations_cost_management.md#per-user-attribution).

!!! tip "Trusted by AWS teams — no custom implementation risk"
    Cognito and IAM Identity Center are the same identity systems AWS teams already rely on for console access and internal applications. By delegating authentication to these services, you get MFA, SSO, and fine-grained permission sets without building or maintaining custom user management logic — and without the security risk of a home-grown implementation. See [Eliminate key rotation with AWS native auth](#security-best-practices) for the operational payoff.

!!! info "Terraform Module"
    The Terraform module does not configure the ALB or API Gateway integrations described in this section, nor the identity provider itself — these are set up directly on the ALB or API Gateway. Validating Cognito tokens **in stdapi.ai** needs no infrastructure beyond the pool: see [Amazon Cognito User Pool Tokens](#amazon-cognito-user-pool-tokens).

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
To disable application-level authentication, do **not** configure any API key environment variable (`API_KEY`, `API_KEY_SSM_PARAMETER`, or `API_KEY_SECRETSMANAGER_SECRET`) and no user pool (`AWS_COGNITO_USER_POOL_ID`).

When no authentication method is configured, stdapi.ai will:

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

### :material-package-variant-closed: Supply Chain Security

Supply chain attacks — where malicious code is injected into a software package before it reaches your infrastructure — represent one of the most dangerous threat vectors for self-hosted AI gateways. A compromised package can silently exfiltrate API keys and credentials from every deployment that installs it, and attacks of this kind have targeted the AI tooling ecosystem.

stdapi.ai's distribution model eliminates this attack surface entirely:

- **Container-only distribution** — stdapi.ai is never distributed as a pip package or PyPI dependency. There is no `pip install stdapi.ai`, no transitive dependency chain to compromise, and no package registry to hijack. The only distribution channels are the AWS Marketplace ECR registry (commercial) and GHCR (community image).
- **AWS Marketplace security validation** — the commercial container image is scanned and validated by AWS before being made available in the Marketplace. The image you deploy is exactly the image AWS validated — nothing can be inserted between validation and deployment.
- **Minimal base image** — the container has no shell, no package manager, and no unnecessary system tools. There is no mechanism to install additional packages or execute arbitrary commands at runtime.
- **Immutable deployment** — the container image is pulled once at task start and run as-is. There are no runtime `pip install`, `apt install`, or dependency resolution steps that could fetch and execute untrusted code.
- **Read-only root filesystem** — when deployed via the Terraform module, the ECS task definition enforces a read-only root filesystem, preventing any modification to the container files at runtime.
- **Temporary credentials with least privilege** — CI/CD pipelines that build and publish the container image use short-lived, role-scoped credentials with only the permissions required for that specific job. No long-lived access keys are used in the build or release process. Jobs that do not require AWS access — such as linters and security scanners — run on isolated runners with no AWS credentials at all, limiting the blast radius of any compromised job.

### :material-shield-alert-outline: SSRF Protection

Users can pass URLs to the API as multimodal content references (images, documents, audio). The application fetches the content at those URLs before forwarding it to the model. Without protection, a malicious user could supply URLs pointing to internal services — such as the AWS EC2/ECS metadata endpoint or private network resources — and read sensitive data through the model response.

stdapi.ai validates every user-supplied URL via DNS resolution before fetching its content — each hostname is resolved and every resulting IP is checked against the blocklist. This protects against DNS rebinding attacks where a seemingly-safe domain resolves to a blocked address.

-   **Baseline Protection**: Always-active blocking of loopback (127.0.0.1, localhost), link-local (169.254.169.254), and reserved address ranges. This prevents access to the AWS EC2/ECS metadata service from within the container.
-   **Private Network Blocking**: By default, the service also blocks every address that is not globally reachable on the public Internet — RFC 1918 networks (10.x.x.x, 172.16-31.x.x, 192.168.x.x), IPv6 unique local addresses, RFC 6598 shared address space (100.64.0.0/10, used by EKS custom networking and Hybrid Nodes) and the remaining special-purpose ranges, in their IPv4-mapped IPv6 form as well. Disable only in controlled environments where accessing local networks is required (`SSRF_PROTECTION_BLOCK_PRIVATE_NETWORKS=false`).
-   **Defense in Depth**: Even with SSRF protection enabled, restrict outbound Security Group rules to only the necessary AWS service endpoints.

!!! tip "Network-layer complement: Route 53 Resolver DNS Firewall"
    SSRF protection blocks by IP range — it doesn't know whether a public IP belongs to a malicious domain (malware, phishing, botnet C2). The Terraform module can close that gap: set `dns_firewall_enabled = true` to block outbound DNS resolution of known-malicious domains before a fetch ever happens. See [AWS Security Hub, GuardDuty & DNS Firewall Integration](#aws-security-hub-guardduty-dns-firewall-integration).

### :material-web-check: Host Header Validation

To protect against Host header injection and web cache poisoning, stdapi.ai can validate the `Host` header of incoming requests. When `TRUSTED_HOSTS` is not configured, no Host header validation is performed.

-   **Recommended Approach**: Use **AWS ALB host-based routing rules** to reject invalid Host headers before they reach the application. This is more performant and centrally managed.
-   **Application Validation**: Use `TRUSTED_HOSTS` to define a list of approved hostnames (supports wildcards like `*.example.com`). Requests with non-matching headers are rejected with an HTTP 400 error.
-   **Probe Impact**: Validation covers `/health` too. The container image's health probe derives its `Host` header from `TRUSTED_HOSTS` and stays green on its own, but a load balancer health check sends the target's IP address as the `Host` and is rejected — see [`TRUSTED_HOSTS`](operations_configuration.md#trusted-hosts) before enabling it behind an ALB or NLB.

### :material-web: Browser Security (CORS)

If your API is accessed directly from web browsers, Cross-Origin Resource Sharing (CORS) must be configured to prevent unauthorized cross-site requests. When `CORS_ALLOW_ORIGINS` is not configured, the CORS middleware is not enabled and browser cross-origin requests are blocked by default.

-   **Principle of Least Privilege**: Only allow specific, trusted domains using `CORS_ALLOW_ORIGINS`.
-   **Offload to Infrastructure**: For production, consider managing CORS at the **AWS ALB** or **API Gateway** level for better performance and to keep your application logic clean.

### :material-swap-horizontal: Proxy Headers

When running behind a load balancer (ALB) or reverse proxy, stdapi.ai can be configured to trust `X-Forwarded-*` headers (`X-Forwarded-For`, `X-Forwarded-Proto`, `X-Forwarded-Port`) to correctly identify the original client IP and protocol. When enabled, the real client IP appears in request logs instead of the proxy's IP.

-   **Security Warning**: Only enable `ENABLE_PROXY_HEADERS` when the service is deployed behind a **trusted proxy**. Enabling this in a publicly exposed environment without a proxy allows attackers to spoof their IP address by sending custom headers.

### :material-resize: Request Size & Resource Limits

Large or numerous inputs are a denial-of-service vector: a single request can carry hundreds of megabytes of inline base64 content, reference many remote files, or stream for a long time. stdapi.ai provides application-level controls, but the most effective size limit is enforced at the edge, before the request reaches the application.

**Application-level controls:**

-   **Inline & downloaded input size**: [`MAX_INPUT_FILE_SIZE`](operations_configuration.md#max-input-file-size) caps the bytes of any single file loaded into memory for model input (base64, `data:` URIs, and HTTP(S)/S3 sources read for the model). Requests over the limit are rejected with HTTP 413 before the content is fully decoded or downloaded — a spoofed `Content-Length` cannot bypass it. Disabled by default; set a value aligned with your largest expected input when exposed to untrusted clients.
-   **Concurrent downloads**: [`MAX_CONCURRENT_INPUT_DOWNLOADS`](operations_configuration.md#max-concurrent-input-downloads) bounds how many remote inputs a single request fetches at once, preventing socket/memory exhaustion and SSRF amplification from a request carrying many URLs.

!!! warning "Application limits do not cap the total request body"
    `MAX_INPUT_FILE_SIZE` applies to individual **file** inputs. It does not bound the overall request body: a large prompt sent as plain text (for example a huge `messages` array rather than a file) is still received and parsed in full before any application limit applies. Cap the total body size at the edge (below).

**Edge body-size limit (recommended):**

The overall request body should be capped at the network edge, before it is buffered by the application — this is the primary protection against memory exhaustion from oversized bodies.

-   **AWS WAF**: the mechanism for an ALB-fronted deployment. Add a rule with a `SizeConstraintStatement` on the request `BODY` and set **oversize handling to block**, so bodies larger than the inspectable size are rejected outright. The Terraform module's WAF (`alb_waf_enabled=true`) is the place to add this rule. AWS WAF inspects only a portion of the body (8 KB by default, up to 64 KB on ALB), so rely on the oversize-handling block action rather than attempting to inspect the full payload.
-   **ALB**: does not enforce a request body size limit on its own — pair it with AWS WAF for that control.
-   **Amazon API Gateway**: if you front the service with API Gateway instead of an ALB, note its hard **10 MB** payload limit, which may be too small for large multimodal or long-context requests.

!!! note "Sizing the edge limit is inherently hard for AI workloads"
    There is no universally correct body-size cap: legitimate prompts are large and keep growing as model context windows expand, and multimodal inputs (images, audio, documents) push request bodies higher still. Set the edge limit **generously** — large enough for your largest legitimate request, small enough to stop obvious abuse — and rely on the finer-grained controls below for cost and abuse management rather than a single tight byte limit.

**Duration & throughput controls:**

-   **Output length**: bound generated output with the API's `max_tokens` / `max_output_tokens` parameters; large client-supplied defaults can be constrained at your application or gateway layer.
-   **Upstream timeout**: [`AI_RESPONSE_TIMEOUT`](operations_configuration.md#ai-response-timeout) closes stalled model connections so a slow or hung generation does not hold resources indefinitely.
-   **Rate limiting & concurrency**: use WAF rate limiting (`alb_waf_rate_limit`, see [Security Best Practices](#security-best-practices)) to bound requests per client, complementing the per-request download concurrency cap above.

---

## :material-shield-star: AWS Security Hub, GuardDuty & DNS Firewall Integration

!!! success "Recommended: deploy via the Terraform module"
    The [stdapi-ai Terraform module](https://github.com/stdapi-ai/terraform-aws-stdapi-ai) is built against the **AWS Security Hub Foundational Security Best Practices (FSBP)** standard and passes a large share of its applicable controls **by default** — no extra configuration required. A set of opt-in variables closes the remaining gaps for organizations with stricter compliance requirements, and the same dedicated VPC also enables native **GuardDuty Runtime Monitoring** and **Route 53 Resolver DNS Firewall** (see below). This makes the module the fastest path to a Security Hub-compliant, threat-monitored deployment; manual (non-Terraform) deployments must implement each control yourself.

The module composes three child modules — [VPC](https://github.com/JGoutin/terraform-aws-vpc), [KMS](https://github.com/JGoutin/terraform-aws-kms-key), and [ECS Fargate](https://github.com/JGoutin/terraform-aws-ecs-fargate) — plus its own ALB/WAF/S3/IAM resources. Each repository's README documents every relevant control — pass, fail, conditional (depends on your configuration), or not applicable — with remediation notes where relevant.

| Module                                                                                          | Key controls covered by default                                                                   | Extra options to improve compliance                                    |
|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [stdapi-ai (root)](https://github.com/stdapi-ai/terraform-aws-stdapi-ai#security-hub-controls)    | S3 encryption/versioning/public-access block, least-privilege IAM, ALB access logging             | `alb_waf_enabled`, `alb_certificate_arn` / `alb_domain_name` (HTTPS redirect), `deletion_protection` |
| [VPC](https://github.com/JGoutin/terraform-aws-vpc#security-hub-controls)                       | EC2.2 (default security group lockdown), EC2.6 (flow logs, 365-day retention), EC2.15/21/48, IAM.24 | `compliance_vpc_endpoints_enabled`, `guardduty_vpc_endpoint_enabled`, `dns_firewall_enabled` — **dedicated VPC only, see below** |
| [KMS](https://github.com/JGoutin/terraform-aws-kms-key#security-hub-controls)                   | KMS.3 (deletion window), KMS.4 (automatic rotation, hardcoded on), KMS.5 (key policy)             | None — rotation and deletion window are hardcoded, not configurable    |
| [ECS Fargate](https://github.com/JGoutin/terraform-aws-ecs-fargate#security-hub-controls)                | ECS.\*, EFS.1-8 (POSIX user enforcement, native backups), CloudWatch.15-17, Backup.1-5, IAM.1/24  | `mount_points_efs_backup_enable`, `mount_points[].efs_posix_user`, `alarms_enabled` |

All four modules also accept a `tags` variable to propagate custom resource tags — itself relevant to Security Hub controls IAM.24 and EC2.48 when a tagging policy is enforced on your account.

### :material-radar: GuardDuty Runtime Monitoring

stdapi.ai's ECS tasks support **[Amazon GuardDuty Runtime Monitoring](https://docs.aws.amazon.com/guardduty/latest/ug/runtime-monitoring.html)** natively. Set `guardduty_vpc_endpoint_enabled = true` and the Terraform module creates the dedicated `guardduty-data` interface VPC endpoint the GuardDuty agent needs to report findings — keeping that traffic off the internet and off any NAT gateway. This isn't a Security Hub control, so it stays disabled by default; enable it whenever GuardDuty Runtime Monitoring is turned on for your account, since provisioning the endpoint through the module guarantees correct subnet placement.

### :material-dns: Route 53 Resolver DNS Firewall

stdapi.ai fetches user-supplied URLs as multimodal content references (images, documents, audio) — see [SSRF Protection](#ssrf-protection) for the application-level defense (IP-based blocklisting of loopback/link-local/private ranges). **Route 53 Resolver DNS Firewall** adds a network-layer control on top of that: set `dns_firewall_enabled = true` and the Terraform module creates a DNS Firewall rule group on the dedicated VPC that blocks or alerts on outbound DNS queries to known-malicious domains, closing the gap SSRF protection doesn't cover — a public IP address that resolves from a domain associated with malware, phishing, or botnet command-and-control.

- `dns_firewall_managed_domain_list_ids` (default: `null`, which resolves automatically to the [AWS Managed](https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/resolver-dns-firewall-managed-domain-lists.html) Aggregate Threat List ID for your deployment region — no AWS CLI call or extra IAM permissions needed) selects which managed domain lists to enforce; `dns_firewall_action` controls whether matches are `BLOCK`ed, `ALERT`ed on, or explicitly `ALLOW`ed.
- `dns_firewall_advanced_enabled` turns on **DNS Firewall Advanced** (additional AWS cost), which detects domain generation algorithm (DGA) and DNS tunneling activity — patterns a static domain list can't catch. Tune sensitivity with `dns_firewall_advanced_confidence_threshold` (`LOW` / `MEDIUM` / `HIGH`).

!!! warning "`compliance_vpc_endpoints_enabled`, `guardduty_vpc_endpoint_enabled`, and `dns_firewall_enabled` require the module's dedicated VPC"
    All three variables attach resources (interface endpoints, or a DNS Firewall rule group) to the VPC the Terraform module creates for you. They have **no effect** — `dns_firewall_enabled` cannot even be set to `true` and fails validation — if you instead pass your own `subnet_ids` to integrate with existing network infrastructure (see [Integration with Existing Infrastructure](operations_deploy_advanced.md#integration-with-existing-infrastructure)). In that case the module never creates a VPC, and you are responsible for provisioning any interface endpoints or DNS Firewall rules your compliance or monitoring tooling needs (ECR, SSM, `guardduty-data`, Resolver rule groups, etc.) directly in your own VPC.

---

## :material-shield-bug: Prompt Injection

Prompt injection is an application-level security concern: just as AWS secures the database engine but customers are responsible for preventing SQL injection, AWS secures the Bedrock infrastructure but **your application is responsible for preventing prompt injection**. For full details, see the [Amazon Bedrock prompt injection security documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-injection.html).

Key practices recommended by AWS:

- **Input validation** — validate and sanitize all user input before passing it to the model; remove or escape special characters and enforce expected formats.
- **Secure coding** — avoid string concatenation for building prompts; apply the principle of least privilege when granting access to resources.
- **System prompt scoping** — when using system prompts, clearly define what the model can and cannot do. Newer models differentiate between system and user prompts — check the model provider's documentation for model-specific guidance.
- **Security testing** — regularly test for prompt injection using penetration testing, static analysis, and dynamic application security testing (DAST).
- **Keep dependencies updated** — monitor AWS security bulletins and keep the Bedrock SDK and libraries up to date.

!!! tip "Use Amazon Bedrock Guardrails"
    The most effective mitigation available within stdapi.ai is to configure an **Amazon Bedrock Guardrail**. Guardrails include a dedicated prompt attack detection layer and can be applied **per request** (via request headers) or **globally for the entire server** (via environment variables). See the [Bedrock Guardrails configuration section](operations_configuration.md#bedrock-guardrails) for setup instructions, and [Detect prompt attacks with Amazon Bedrock Guardrails](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-prompt-attack.html) for the upstream AWS documentation.

!!! warning "stdapi.ai does not include built-in prompt injection protection"
    stdapi.ai passes user prompts to the model as-is and does not apply any custom filtering or sanitization against prompt injection. Protecting against this risk is the responsibility of your application and infrastructure.

---

## :material-shield-check: Security Best Practices

**Network Isolation:**

- :material-check: **Deploy in private subnets** — without direct internet access.
- :material-check: **Restrict security group inbound rules** — allow traffic to the stdapi.ai service only from the ALB or API Gateway security group on the application port.
- :material-check: **Apply strict egress rules** — limit outbound traffic to only the required AWS services (Bedrock, S3, etc.).
- :material-check: **Use VPC endpoints** — the Terraform module creates VPC endpoints for all required AWS services (`vpc_endpoints_allowed=true` by default), keeping all traffic to Bedrock, S3, SSM, CloudWatch Logs, and AI services off the public internet with no NAT Gateway required.

**WAF & Rate Limiting:**

- :material-check: **Enable AWS WAF** — the Terraform module supports WAF via `alb_waf_enabled=true`. When enabled, it activates the AWS Managed Core Rule Set (SQLi, XSS, common exploits), Linux Rule Set, and IP Reputation List (known malicious IPs).
- :material-check: **Configure rate limiting** — set `alb_waf_rate_limit` to limit the number of requests per IP per 5-minute window, protecting against abuse and quota exhaustion.
- :material-check: **Block anonymous IPs** — set `alb_waf_block_anonymous_ips=true` to block known VPNs, open proxies, and Tor exit nodes.
- :material-check: **Cap request body size** — add an AWS WAF `SizeConstraintStatement` on the request body (oversize handling set to block) to reject oversized bodies at the edge; see [Request Size & Resource Limits](#request-size-resource-limits).

**Secrets Management:**

- :material-check: **Use SSM Parameter Store or Secrets Manager** — always use encrypted secret storage for production API keys; never pass keys as plain environment variables.
- :material-check: **Rotate API keys** — Secrets Manager supports [automated secret rotation](https://docs.aws.amazon.com/secretsmanager/latest/userguide/rotating-secrets.html) via Lambda; note that stdapi.ai reads the key once at startup, so a container restart is required after rotation.
- :material-check: **Share the Realtime signing key** — the [ephemeral client secrets](api_openai_realtime.md#ephemeral-client-secrets) minted for browser clients are signed, not stored, so every instance must sign with the same key. It is derived from the deployment's API key by default; a deployment running with no API key at all falls back to a per-process random key and must set [`REALTIME_CLIENT_SECRET_KEY`](operations_configuration.md#realtime-client-secret-key) instead. Treat it as a secret and store it the same way: anyone holding it can mint a client secret this deployment accepts.

!!! tip "Eliminate key rotation with AWS native auth"
    When using API key authentication, rotation requires updating SSM/Secrets Manager and performing a rolling ECS task replacement. This is predictable but adds a deployment step.

    To avoid key rotation entirely, use [**Amazon Cognito user pool tokens**](#amazon-cognito-user-pool-tokens), or **OIDC / Cognito / IAM Identity Center** via the ALB — there are no API keys to rotate, and access is granted and withdrawn in your identity provider without touching the deployment. With tokens validated by stdapi.ai, withdrawing access takes effect when the caller's current token expires: keep the app client's [access-token validity](#amazon-cognito-user-pool-tokens) short.

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

!!! info "Terraform Module Support"
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
