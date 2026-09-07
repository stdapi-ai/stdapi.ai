---
title: "Configuration - Authentication and Tenants"
description: "Configure how stdapi.ai authenticates clients: static API keys, SSM Parameter Store, Secrets Manager, Amazon Cognito user pool tokens, per-tenant API keys and the OAuth discovery documents agents read."
keywords: "API key authentication, SSM Parameter Store, Secrets Manager, Amazon Cognito, OAuth discovery, tenant API keys, multi-tenant, authentication mode"
---

# :material-lock: Authentication

How the gateway decides a caller is allowed in, and who that caller is. Part of the [Configuration Guide](operations_configuration.md).

stdapi.ai supports three sources for API key authentication, plus Amazon Cognito user pool tokens and per-tenant API keys.

!!! info "API Key Sources"
    **Configure exactly one source.** If several are set, the first match in this precedence order is used and the others are ignored:

    1. :material-key: **Direct API key** — `API_KEY` (highest precedence)
    2. :material-database-lock: **SSM Parameter Store** — `API_KEY_SSM_PARAMETER`
    3. :material-key-variant: **Secrets Manager** — `API_KEY_SECRETSMANAGER_SECRET` (lowest precedence)

    The methods below are listed in that precedence order. **SSM Parameter Store remains the recommended method for production.**

    :material-account-key: **Amazon Cognito user pool tokens** ([Method 4](#cognito-authentication)) are independent of the API key and can be accepted alongside it, or instead of it — [`AUTHENTICATION_MODE`](#authentication-mode) decides.

!!! warning "Conflicting Configuration"
    Only one combination is rejected at startup: `API_KEY` set together with a Secrets Manager source (`API_KEY_SECRETSMANAGER_SECRET`). Every other combination starts normally and is resolved silently by the precedence order above — the lower-precedence sources are never read.

!!! danger "No Authentication Warning"
    If neither an API key source nor a user pool is configured, the API accepts all requests without authentication and a security warning is logged at startup. This is suitable **only for internal/private deployments**.

## :material-format-list-bulleted: Settings Summary

### :material-lock: Authentication { #summary-authentication }

Configure **one** API key source. If several are set, precedence is `API_KEY` → SSM Parameter Store → Secrets Manager — see [Authentication](#authentication):

| Variable                                                          | Default   | Description                                                        |
|-------------------------------------------------------------------|-----------|--------------------------------------------------------------------|
| [`API_KEY_SSM_PARAMETER`](#api-key-ssm)                           | None      | AWS Systems Manager Parameter Store path for API key (recommended) |
| [`API_KEY_SECRETSMANAGER_SECRET`](#api-key-secretsmanager-secret) | None      | AWS Secrets Manager secret name containing API key                 |
| [`API_KEY_SECRETSMANAGER_KEY`](#api-key-secretsmanager-key)       | `api_key` | JSON key name within Secrets Manager secret                        |
| [`API_KEY`](#api-key)                                             | None      | Direct API key value (not recommended for production)              |
| [`TENANT_API_KEYS`](#tenant-api-keys)                             | `false`   | Accept per-tenant API keys scoped by the tenant records in the shared DynamoDB table |
| [`TENANT_KEY_CACHE_SECONDS`](#tenant-key-cache-seconds)           | `60`      | Per-instance validation cache, which is also the revocation window |
| [`TENANT_KEY_SSM_PARAMETER_PREFIX`](#tenant-key-ssm-parameter-prefix) | `/stdapi-ai/tenant-keys` | SSM prefix minted tenant keys are delivered under, once |
| [`TENANT_KEY_SSM_KMS_KEY_ID`](#tenant-key-ssm-kms-key-id)         | None      | KMS key encrypting the delivery parameters, instead of `alias/aws/ssm` |
| [`TENANT_AWS_CREDENTIALS`](#tenant-aws-credentials)               | `false`   | Let a tenant register a cross-account IAM role its model invocations run under |
| [`AUTHENTICATION_MODE`](#authentication-mode)                     | `any`     | Accepted methods: `any`, `api_key` or `cognito`                    |

Amazon Cognito user pool tokens are an alternative to the API key — see [Amazon Cognito Authentication](#cognito-authentication):

| Variable                                                            | Default    | Description                                                     |
|---------------------------------------------------------------------|------------|-----------------------------------------------------------------|
| [`AWS_COGNITO_USER_POOL_ID`](#aws-cognito-user-pool-id)             | None       | User pool whose tokens authenticate clients (enables the method) |
| [`AWS_COGNITO_CLIENT_IDS`](#aws-cognito-client-ids)                 | None       | App client IDs whose tokens are accepted (required with a pool)  |
| [`AWS_COGNITO_REQUIRED_SCOPES`](#aws-cognito-required-scopes)       | None       | Scopes a token must all carry                                    |
| [`AWS_COGNITO_ACCEPT_ID_TOKEN`](#aws-cognito-accept-id-token)       | `false`    | Also accept identity tokens, not only access tokens              |
| [`AWS_COGNITO_ISSUER_TYPE`](#aws-cognito-issuer-type)               | `original` | Pool issuer configuration: `original` or `updated`               |

Publishing where tokens come from lets an AI agent authenticate itself — see [Authentication Discovery](#oauth-discovery):

| Variable                                                              | Default              | Description                                                        |
|-----------------------------------------------------------------------|----------------------|--------------------------------------------------------------------|
| [`OAUTH_RESOURCE_IDENTIFIER`](#oauth-resource-identifier)             | None                 | Public URL clients dial (publishes the discovery document)         |
| [`OAUTH_AUTHORIZATION_SERVERS`](#oauth-authorization-servers)         | The user pool issuer | Issuer URLs of the authorization servers                           |
| [`OAUTH_SCOPES_SUPPORTED`](#oauth-scopes-supported)                   | Required scopes      | Scopes a token needs, advertised to clients                        |

## :material-key: Method 1: Direct API Key { #method-1-direct-api-key }

Provide the API key directly via environment variable. Intended for local development and testing; it takes precedence over both AWS-backed sources.

#### `API_KEY` { #api-key }

:octicons-package-24: **Purpose**
:   Static API key value

:octicons-alert-24: **Security Warning**
:   Avoid hardcoding in configuration files; use environment variables only

:octicons-person-24: **Client Usage**
:   Clients must include this key in the `Authorization: Bearer <key>` header or `X-API-Key` header

```bash
export API_KEY=sk-1234567890abcdef...
```

## :material-database-lock: Method 2: SSM Parameter Store (Recommended) { #method-2-ssm-parameter-store-recommended }

**Recommended** - Use AWS Systems Manager Parameter Store for secure key storage with encryption, access control, and auditing. This method should be used only with **already existing** parameters.

#### `API_KEY_SSM_PARAMETER` { #api-key-ssm }

:octicons-package-24: **Purpose**
:   Name of the SSM parameter containing the API key. The parameter is retrieved from the current region detected by the running container, or defaults to the first region in `AWS_BEDROCK_REGIONS`.

:octicons-shield-check-24: **Recommendation**
:   Use `SecureString` type for encryption at rest

:octicons-lock-24: **IAM Permissions Required**
:   `ssm:GetParameter`, `kms:Decrypt` (if encrypted)

```bash
export API_KEY_SSM_PARAMETER=/stdapi/prod/api-key
```

## :material-key-variant: Method 3: Secrets Manager { #method-3-secrets-manager }

Use AWS Secrets Manager for secure key storage with automatic rotation support. This method should be used only with **already existing** secrets.

#### `API_KEY_SECRETSMANAGER_SECRET` { #api-key-secretsmanager-secret }

:octicons-package-24: **Purpose**
:   Name of the Secrets Manager secret containing the API key. The secret is retrieved from the current region detected by the running container, or defaults to the first region in `AWS_BEDROCK_REGIONS`.

:octicons-code-24: **Format**
:   Can be a plain string or JSON object

:octicons-lock-24: **IAM Permissions Required**
:   `secretsmanager:GetSecretValue`

#### `API_KEY_SECRETSMANAGER_KEY` { #api-key-secretsmanager-key }

:octicons-package-24: **Purpose**
:   JSON key name within the secret (if the secret is a JSON object)

:octicons-gear-24: **Default**
:   `api_key`

**Plain String Secret:**

```bash
export API_KEY_SECRETSMANAGER_SECRET=stdapi-api-key
```

**JSON Secret:**

```bash
export API_KEY_SECRETSMANAGER_SECRET=stdapi-credentials
export API_KEY_SECRETSMANAGER_KEY=api_key
```

Example JSON secret structure:
```json
{
  "api_key": "sk-1234567890abcdef...",
  "other_config": "value"
}
```

## :material-account-check: Method 4: Amazon Cognito User Pool Tokens { #cognito-authentication }

Accept the bearer tokens issued by an [Amazon Cognito user pool](https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools.html) instead of, or alongside, the API key. Each caller gets its own short-lived credential, and the verified caller is the identity [per-user cost attribution](operations_cost_management.md#per-user-attribution) bills against; withdrawing a caller's access takes effect when their current token expires. Clients send the token in the `Authorization: Bearer <token>` or `X-API-Key` header, like an API key. What is validated on every request is described in [Authentication & Security](operations_authentication_security.md#amazon-cognito-user-pool-tokens).

```bash
export AWS_COGNITO_USER_POOL_ID=eu-west-3_a1b2c3d4e
export AWS_COGNITO_CLIENT_IDS=1example23456789abcdefghij
```

!!! warning "Incomplete configuration fails startup"
    A user pool without `AWS_COGNITO_CLIENT_IDS`, a Cognito setting without a pool, or an `AUTHENTICATION_MODE` that contradicts what is configured, all stop the server at startup with an explicit message — a partially configured pool never degrades into an unauthenticated deployment.

!!! tip "The pool also configures agent discovery"
    Add [`OAUTH_RESOURCE_IDENTIFIER`](#oauth-resource-identifier) and an AI agent can authenticate itself against the deployment. Nothing else is needed: the pool's issuer and required scopes are what get published — see [Authentication Discovery for Agents](#oauth-discovery).

#### `AUTHENTICATION_MODE` { #authentication-mode }

:octicons-package-24: **Purpose**
:   Which client authentication methods the deployment accepts

:octicons-gear-24: **Default**
:   `any` — every method that is configured

:octicons-list-unordered-24: **Values**
:   - `any`: the API key, tenant API keys and user pool tokens, whichever is configured
    - `api_key`: the API key and tenant API keys only; startup fails if a user pool is also configured
    - `cognito`: user pool tokens only; startup fails if an API key source or tenant API keys are also configured

```bash
export AUTHENTICATION_MODE=cognito
```

#### `AWS_COGNITO_USER_POOL_ID` { #aws-cognito-user-pool-id }

:octicons-package-24: **Purpose**
:   Identifier of the user pool whose tokens authenticate clients. Setting it enables the method; the pool's AWS Region is read from the identifier itself, and the public signing keys are loaded from that Region at startup.

:octicons-gear-24: **Default**
:   None — user pool tokens are not accepted

:octicons-alert-24: **Requirement**
:   `AWS_COGNITO_CLIENT_IDS` must be set too

:octicons-lock-24: **IAM Permissions Required**
:   None — the signing keys are public

```bash
export AWS_COGNITO_USER_POOL_ID=eu-west-3_a1b2c3d4e
```

#### `AWS_COGNITO_CLIENT_IDS` { #aws-cognito-client-ids }

:octicons-package-24: **Purpose**
:   Comma-separated app client IDs whose tokens are accepted. A token issued to any other app client of the pool is rejected.

:octicons-gear-24: **Default**
:   Empty — startup fails when a user pool is configured without it

:octicons-alert-24: **Requirement**
:   Required whenever `AWS_COGNITO_USER_POOL_ID` is set

```bash
export AWS_COGNITO_CLIENT_IDS=1example23456789abcdefghij,2example3456789abcdefghijk
```

#### `AWS_COGNITO_REQUIRED_SCOPES` { #aws-cognito-required-scopes }

:octicons-package-24: **Purpose**
:   Comma-separated OAuth 2.0 scopes a token must **all** carry to be accepted

:octicons-gear-24: **Default**
:   None — any scope set is accepted

:octicons-alert-24: **Requirement**
:   Custom scopes exist only on tokens issued by the pool's OAuth 2.0 token endpoint, which needs a [resource server](https://docs.aws.amazon.com/cognito/latest/developerguide/cognito-user-pools-define-resource-servers.html) and a pool domain. Tokens obtained by signing in with a username and password carry only `aws.cognito.signin.user.admin` and are rejected when a custom scope is required.

```bash
export AWS_COGNITO_REQUIRED_SCOPES=stdapi/invoke
```

#### `AWS_COGNITO_ACCEPT_ID_TOKEN` { #aws-cognito-accept-id-token }

:octicons-package-24: **Purpose**
:   Also accept identity tokens, not only access tokens

:octicons-gear-24: **Default**
:   `false`

:octicons-workflow-24: **Effect**
:   Identity tokens describe the signed-in user rather than granting API access, and carry no scopes. Enable only for clients that cannot obtain an access token.

```bash
export AWS_COGNITO_ACCEPT_ID_TOKEN=true
```

#### `AWS_COGNITO_ISSUER_TYPE` { #aws-cognito-issuer-type }

:octicons-package-24: **Purpose**
:   The pool's [issuer configuration](https://docs.aws.amazon.com/cognito-user-identity-pools/latest/APIReference/API_IssuerConfigurationType.html), which decides the issuer URL its tokens carry

:octicons-gear-24: **Default**
:   `original`

:octicons-list-unordered-24: **Values**
:   - `original`: `https://cognito-idp.<region>.amazonaws.com/<pool-id>`
    - `updated`: `https://issuer-cognito-idp.<region>.amazonaws.com/<pool-id>`, available on the Essentials and Plus pool tiers

:octicons-alert-24: **Requirement**
:   Must match the pool's own setting; tokens whose issuer differs are rejected

```bash
export AWS_COGNITO_ISSUER_TYPE=updated
```

## :material-account-key: Method 5: Tenant API Keys { #tenant-authentication }

Accept per-tenant API keys (`sk-std-...`), each backed by a record in the [shared DynamoDB table](operations_configuration_storage.md#aws-dynamodb-table) that scopes what the key may call — model allow/deny lists and endpoint restrictions. The records are declared by the operator (the [Terraform module](operations_getting_started.md#quick-start)'s `tenants` variable, or written directly); the secret is minted by the server and delivered once through SSM Parameter Store. How keys are issued, scoped, cached and revoked is described in [Authentication & Security](operations_authentication_security.md#tenant-api-keys). Clients send the key in the `Authorization: Bearer <key>` or `X-API-Key` header, like any API key.

```bash
export AWS_DYNAMODB_TABLE=stdapi-ai
export TENANT_API_KEYS=true
```

#### `TENANT_API_KEYS` { #tenant-api-keys }

:octicons-package-24: **Purpose**
:   Enable per-tenant API keys, validated against the tenant records in the shared DynamoDB table

:octicons-gear-24: **Default**
:   `false` — tenant-shaped credentials are only compared against the deployment API key, like any other value

:octicons-alert-24: **Requirement**
:   [`AWS_DYNAMODB_TABLE`](operations_configuration_storage.md#aws-dynamodb-table) must be set, or startup fails. [`TENANT_KEY_SSM_PARAMETER_PREFIX`](#tenant-key-ssm-parameter-prefix) has a default and needs no configuration of its own — override it if this deployment shares an AWS account with another

:octicons-lock-24: **IAM Permissions Required**
:   The [shared table permissions](operations_iam_permissions.md#shared-table), plus `ssm:PutParameter` and `ssm:GetParameter` on the delivery prefix — see [Tenant API Key Delivery](operations_iam_permissions.md#tenant-key-delivery)

#### `TENANT_KEY_CACHE_SECONDS` { #tenant-key-cache-seconds }

:octicons-package-24: **Purpose**
:   Seconds each server instance caches a validated tenant key before re-reading its records. This is the revocation window: a key revoked, disabled or re-scoped keeps its previous decision for up to this long per instance

:octicons-gear-24: **Default**
:   `60`

:octicons-list-unordered-24: **Values**
:   `0` disables the cache and reads the table on every request

#### `TENANT_KEY_SSM_PARAMETER_PREFIX` { #tenant-key-ssm-parameter-prefix }

:octicons-package-24: **Purpose**
:   SSM Parameter Store prefix minted tenant keys are delivered under, one `SecureString` parameter named `<prefix>/<key id>` per tenant

:octicons-gear-24: **Default**
:   `/stdapi-ai/tenant-keys`

:octicons-alert-24: **Security Warning**
:   Use a prefix private to this deployment: any principal allowed to read under it can read every tenant's key, so a deployment sharing an AWS account with another must not keep the default. Retrieve each key once, deliver it, then delete the parameter

```bash
export TENANT_KEY_SSM_PARAMETER_PREFIX=/stdapi-ai/prod/tenant-keys
```

#### `TENANT_KEY_SSM_KMS_KEY_ID` { #tenant-key-ssm-kms-key-id }

:octicons-package-24: **Purpose**
:   AWS KMS key encrypting the `SecureString` parameters the minted tenant keys are delivered through

:octicons-gear-24: **Default**
:   None — the parameters are encrypted with the AWS-managed `alias/aws/ssm` key, whose key policy lets **any principal of the account** holding `ssm:GetParameter` under the prefix decrypt them

:octicons-list-unordered-24: **Values**
:   A key ID, an alias (`alias/<name>`), or an ARN of either

:octicons-workflow-24: **Effect**
:   Reading a delivered key then also requires `kms:Decrypt` on that key, so the delivery is protected by a key policy of your own instead of the account-wide reach of the AWS-managed key

:octicons-lock-24: **IAM Permissions Required**
:   `kms:Encrypt` and `kms:Decrypt` on the key (add `kms:GenerateDataKey` if the account's default parameter tier creates advanced parameters) — see [Tenant API Key Delivery](operations_iam_permissions.md#tenant-key-delivery)

!!! tip "Set for you by the Terraform module"
    The [Terraform module](operations_getting_started.md#quick-start) passes the deployment's own KMS key here automatically; there is nothing to configure.

```bash
export TENANT_KEY_SSM_KMS_KEY_ID=alias/stdapi-ai
```

#### `TENANT_AWS_CREDENTIALS` { #tenant-aws-credentials }

:octicons-package-24: **Purpose**
:   Let a tenant register an IAM role of its own AWS account (`aws_role_arn` on its tenant record), so that tenant's model invocations run under the tenant's account — its Amazon Bedrock model access, quotas and bill. The server assumes the role with a server-minted `ExternalId` (the AWS confused-deputy pattern); no secret is stored anywhere. See [Tenant AWS credentials](operations_authentication_security.md#tenant-aws-credentials)

:octicons-gear-24: **Default**
:   `false` — every request runs under the server's own identity, and a tenant record declaring `aws_role_arn` is refused rather than silently billed to the deployment

:octicons-alert-24: **Requirement**
:   Requires [`TENANT_API_KEYS`](#tenant-api-keys). Incompatible with Amazon Bedrock Guardrails ([`AWS_BEDROCK_GUARDRAIL_IDENTIFIER`](operations_configuration_bedrock.md#aws-bedrock-guardrail-identifier) or a model-alias guardrail): a guardrail of this deployment's account cannot be evaluated by a tenant principal, so startup fails rather than serving tenant requests unguarded

:octicons-shield-check-24: **Required IAM Permissions**
:   `sts:AssumeRole` on the tenant roles — see [IAM permissions](operations_iam_permissions.md#tenant-aws-credentials)

```bash
export TENANT_AWS_CREDENTIALS=true
```

## :material-magnify: Authentication Discovery for Agents { #oauth-discovery }

Publishes, at `/.well-known/oauth-protected-resource`, where clients obtain a token, and points every `401 Unauthorized` at that document. An AI agent — or any MCP client — can then authenticate against this deployment without having been configured for it first. See [Authentication Discovery for Agents](operations_authentication_security.md#authentication-discovery-for-agents) for the full flow.

Nothing is published until `OAUTH_RESOURCE_IDENTIFIER` is set. With an [Amazon Cognito user pool](#cognito-authentication) configured, that variable is the only one to set: the pool already names the issuer and the scopes, and both are published from it. The document is public and unauthenticated, since a client reads it before it has any credential.

#### `OAUTH_RESOURCE_IDENTIFIER` { #oauth-resource-identifier }

:octicons-package-24: **Purpose**
:   Public URL clients use to reach this deployment, published as the identity of the protected resource

:octicons-gear-24: **Default**
:   None — no discovery document is published, and `401` responses only state that a bearer token is expected

:octicons-alert-24: **Requirement**
:   Must be the exact origin clients dial — scheme and host, an explicit port only when it is not the default one for the scheme, and no path, query or fragment. Clients compare it character by character against the URL they used, so `https://api.example.com` and `https://api.example.com:443` are not interchangeable. Requires `OAUTH_AUTHORIZATION_SERVERS`, unless a user pool supplies the issuer.

```bash
export OAUTH_RESOURCE_IDENTIFIER=https://api.example.com
```

#### `OAUTH_AUTHORIZATION_SERVERS` { #oauth-authorization-servers }

:octicons-package-24: **Purpose**
:   Issuer URLs of the OAuth 2.0 authorization servers that issue tokens for this deployment, comma-separated

:octicons-gear-24: **Default**
:   The issuer of the [`AWS_COGNITO_USER_POOL_ID`](#aws-cognito-user-pool-id) pool, when one is configured — otherwise none

:octicons-workflow-24: **Effect**
:   A client reads each issuer's own metadata to find where to sign in, so this deployment never describes the sign-in flow itself. A load balancer or API gateway authenticating in front of stdapi.ai publishes the issuer of whichever provider it uses.

    With a user pool configured, leave this unset: the pool issues the tokens the deployment accepts, so its own issuer is published — `https://cognito-idp.<region>.amazonaws.com/<pool-id>`, or `https://issuer-cognito-idp.<region>.amazonaws.com/<pool-id>` when [`AWS_COGNITO_ISSUER_TYPE`](#aws-cognito-issuer-type) is `updated`. `<region>` and `<pool-id>` come from the pool ID itself, and the host follows the pool Region's AWS partition (`amazonaws.com.cn` in China, `amazonaws.eu` in the European Sovereign Cloud). Set the variable only to publish further issuers.

:octicons-alert-24: **Requirement**
:   Each entry is an `https` URL with no query or fragment. Required when `OAUTH_RESOURCE_IDENTIFIER` is set and no user pool is configured. When one is, the list must include the pool's own issuer — a client sent anywhere else obtains a token every request refuses, so startup fails instead.

```bash
export OAUTH_AUTHORIZATION_SERVERS=https://cognito-idp.eu-west-3.amazonaws.com/eu-west-3_a1b2c3d4e
```

#### `OAUTH_SCOPES_SUPPORTED` { #oauth-scopes-supported }

:octicons-package-24: **Purpose**
:   Scopes a token needs to call this API, comma-separated

:octicons-gear-24: **Default**
:   [`AWS_COGNITO_REQUIRED_SCOPES`](#aws-cognito-required-scopes) — with neither set, no scope is advertised and a client asks for whatever its own configuration names

:octicons-workflow-24: **Effect**
:   Advertised both in the discovery document and in the `401` challenge, so a client asks its authorization server for the right scopes on its first attempt. The scopes a token must carry to be accepted are exactly the scopes to ask for, so they are published unless this variable names others.

```bash
export OAUTH_SCOPES_SUPPORTED=stdapi/invoke
```

---
