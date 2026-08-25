---
title: Releases & Roadmap - Active Development
description: stdapi.ai release history and upcoming features. Track regular updates, new Amazon Bedrock capabilities, and active development progress.
keywords: stdapi.ai releases, AI gateway updates, AWS Bedrock features, API gateway roadmap, software changelog, active development, new AI features, product updates
---

# :material-timeline: Releases & Roadmap

**stdapi.ai is under active development** with regular feature releases.

## :material-tag-multiple: Recent Releases

See [Release History below](#release-history) for the full changelog of all releases.

**Latest: v1.16.1** – A maintenance update to v1.16.0, which added four new API surfaces — conversations, batches, vector stores and realtime speech — plus retrieval a model calls for itself, a vector store served from your own Amazon Bedrock knowledge base, long and streamed text-to-speech, live transcription, per-caller authentication, and end-user cost attribution on the AWS bill. See the [full release notes](#v1160-conversations-batches-vector-stores-realtime-speech-per-user-identity-with-v1161-maintenance-update) below.

---

## :material-rocket-launch: Roadmap (Tracked on GitHub)

Pending features and current deployment state are tracked on the [GitHub Project](https://github.com/orgs/stdapi-ai/projects/1).

---

## :material-history: Release History

### v1.16.0 – Conversations, Batches, Vector Stores, Realtime Speech & Per-User Identity (with v1.16.1 maintenance update)

This release adds four API surfaces and finishes the speech story. **New APIs**: [**Conversations**](api_openai_conversations.md) keep a thread server-side, so a client continues it by id instead of resending the history; the OpenAI [**Batch API**](api_openai_batches.md) and Anthropic [**Message Batches API**](api_anthropic_batches.md) run large request sets asynchronously at the discounted batch price; [**Vector Stores**](api_openai_vector_stores.md) index and search files by meaning — or address [a knowledge base you already run](api_openai_vector_stores.md#knowledge-base-stores) — with a model reaching either kind for itself through [`file_search`](api_openai_responses.md#file-search); and the [**Realtime API**](api_openai_realtime.md) holds a spoken conversation over one WebSocket. **Speech**: 100,000-character [synthesis](api_openai_audio_speech.md#long-input) spoken as it is produced, [live transcription](api_openai_audio_transcriptions.md#streaming) needing no bucket, and [**Amazon Nova Sonic**](api_openai_audio_transcriptions.md#amazon-nova-sonic) as the lowest-cost speech-to-text backend here. **Identity per caller**: [Amazon Cognito tokens](operations_configuration.md#cognito-authentication) alongside or instead of the API key, [published discovery](operations_configuration.md#oauth-discovery) so an agent authenticates itself, and [per-user cost attribution](operations_cost_management.md#per-user-attribution) reporting each end user's spend from the AWS invoice rather than an estimate.

!!! warning "New Required IAM Permissions"
    v1.16.0 adds one action every deployment needs, a handful that belong to statements you may already grant, and one statement per optional feature. See [IAM Permissions](operations_iam_permissions.md) for the policies in full.

    Enough of them together that **they no longer fit one policy**: IAM caps a customer managed policy at 6,144 characters, and a deployment enabling most of these exceeds it. Attach several policies to the role rather than widening actions to save room — the Terraform module now ships two, one for Amazon Bedrock and one for the supporting services, and does that for you.

    **Required on upgrade, whatever the deployment does:**

    - **`bedrock:InvokeModelWithBidirectionalStream`** — serves every model invoked over a two-way connection — the [Realtime API](api_openai_realtime.md), and Amazon Nova Sonic transcription and translation. It belongs to the [core Bedrock policy](operations_iam_permissions.md#bedrock-iam); no route-specific action exists for any of them.

    **Add to a statement you already grant, if the deployment uses that feature:**

    - **`bedrock:UpdateSession`**, on the [session storage statement](operations_iam_permissions.md#bedrock-session-storage-optional) — the conversation metadata update (`POST /v1/conversations/{id}`) and nothing else. The rest of the Conversations API uses the actions stored responses already require.
    - **`bedrock-mantle:CountTokens`**, on the [Bedrock Mantle statement](operations_iam_permissions.md#bedrock-mantle-iam) — counts the tokens of a Mantle-served model on `/anthropic/v1/messages/count_tokens`, since Amazon Bedrock's own `CountTokens` takes Anthropic models only. Needed by any deployment serving Mantle models, which is the default.
    - **`transcribe:StartStreamTranscription`**, on the [speech-to-text statement](operations_iam_permissions.md#speech-to-text-optional) — serves [`stream=true`](api_openai_audio_transcriptions.md#streaming) on `/v1/audio/transcriptions`. Add it with the upgrade: without it a streamed request that names its language answers `503` `feature_unavailable`, and the server log names the permission. It stages nothing, so a deployment with no bucket at all grants this one alone.
    - **`polly:StartSpeechSynthesisStream`, `polly:StartSpeechSynthesisTask` and `polly:GetSpeechSynthesisTask`**, plus `s3:PutObject`, `s3:GetObject` and `s3:DeleteObject` on each bucket serving an Amazon Polly Region, on the [text-to-speech statement](operations_iam_permissions.md#text-to-speech-optional) — they serve [input above 3,000 characters](api_openai_audio_speech.md#long-input) and nothing else. With a bucket configured and these missing, long requests are accepted and then fail on the permission: grant the whole set, or leave the bucket unconfigured and keep the 3,000-character answer.
    - **`translate:ListLanguages`**, on the [translation statement](operations_iam_permissions.md#text-translation-optional) — read once at startup so an unsupported language pair is refused before the audio is transcribed. Genuinely optional: without it the check stays off and translation still works, reporting the unsupported pair once the translation call itself fails.

    **New statements, one per optional feature:**

    - [**Vector stores**](operations_iam_permissions.md#vector-stores-optional) — `s3vectors:CreateIndex`, `DeleteIndex`, `PutVectors`, `GetVectors`, `QueryVectors` and `DeleteVectors`, scoped to your vector bucket and its indexes. No bucket-level create or delete is granted: the gateway creates and deletes the indexes inside the bucket, never the bucket.
    - [**Knowledge base vector stores**](operations_iam_permissions.md#knowledge-base-vector-stores) — `bedrock:GetKnowledgeBase`, `Retrieve`, `ListDataSources`, `IngestKnowledgeBaseDocuments`, `ListKnowledgeBaseDocuments`, `GetKnowledgeBaseDocuments` and `DeleteKnowledgeBaseDocuments`, one statement per allowlisted knowledge base ARN. `bedrock:ListKnowledgeBases` is deliberately **not** granted and not needed — the server only ever addresses the identifiers it was given.
    - [**Batch inference**](operations_iam_permissions.md#batch-inference) — `bedrock:CreateModelInvocationJob`, `GetModelInvocationJob` and `StopModelInvocationJob` on the server's role, plus `iam:PassRole` conditioned on `bedrock.amazonaws.com`. The service role Amazon Bedrock assumes carries its own policy: `s3:GetObject`, `s3:PutObject` and `s3:ListBucket` on the batch prefix, and `bedrock:InvokeModel` on the models you batch.
    - [**Per-user cost attribution**](operations_iam_permissions.md#per-user-cost-attribution) — `sts:AssumeRole` and `sts:TagSession` on the server's role *and* in the end user role's trust policy (both actions: without `TagSession`, every tagged session is denied), and `bedrock:InvokeModel`, `bedrock:InvokeModelWithResponseStream` and `bedrock:ApplyGuardrail` on the end user role itself, since AWS authorizes those against the caller of the invocation.
    - [**Web search**](operations_iam_permissions.md#web-search-iam) — `bedrock-websearch:InvokeSearch` and `bedrock-websearch:InvokeFetch`, plus `bedrock-websearch:ExternalWebAccess` only where a request may reach the open internet. Leaving that last one out is what keeps every search inside the AWS boundary. A missing web-search permission produces no error and no server log entry: the model answers without having searched, so check these before suspecting the model.
    - [**Transcription output encryption**](operations_iam_permissions.md#speech-to-text-optional) — `kms:GenerateDataKey` and `kms:Decrypt` on the key named by [`AWS_TRANSCRIBE_OUTPUT_ENCRYPTION_KEY_ARN`](operations_configuration.md#aws-transcribe-output-encryption-key-arn), in the key policy as well as on the role.
    - [**Durable vector store indexing**](operations_iam_permissions.md#durable-vector-store-indexing) — `sqs:SendMessage`, `sqs:ReceiveMessage`, `sqs:DeleteMessage`, `sqs:ChangeMessageVisibility` and `sqs:GetQueueAttributes`, on the single queue named by [`AWS_SQS_VECTOR_STORE_QUEUE_URL`](operations_configuration.md#aws-sqs-vector-store-queue-url) and never on `*`. Needed only if you configure that queue; leave it unset and nothing here applies. No queue is ever created, deleted or reconfigured, so none of those actions is granted.

    !!! note "Five features stay inert until you create the resource they need"
        Nothing in this release is breaking — but these five answer `503`, or stay off, until the resource exists in your own account:

        - **Batches** — an IAM service role Amazon Bedrock assumes to read the requests and write the results ([`AWS_BEDROCK_BATCH_ROLE_ARN`](operations_configuration.md#aws-bedrock-batch-role-arn)), plus the bucket it reads and writes.
        - **Vector stores** — an [Amazon S3 vector bucket](operations_configuration.md#aws-s3-vectors-bucket) you create yourself, and the [Region](operations_configuration.md#aws-s3-vectors-region) it lives in.
        - **Knowledge base vector stores** — an allowlist of the knowledge bases this deployment may address ([`AWS_BEDROCK_KNOWLEDGE_BASE_IDS`](operations_configuration.md#aws-bedrock-knowledge-base-ids)), empty by default. One that is not on it answers exactly as a store that does not exist, so the setting cannot be probed for what a deployment holds.
        - **Cognito authentication** — a user pool and its app clients ([`AWS_COGNITO_USER_POOL_ID`](operations_configuration.md#aws-cognito-user-pool-id)); until then the API key remains the only method, exactly as before.
        - **Per-user cost attribution** — a role for the end user sessions ([`AWS_BEDROCK_USER_ROLE_ARN`](operations_configuration.md#aws-bedrock-user-role-arn)); off by default, and every call keeps being billed to the deployment's own identity until it is set.

        Conversations, the Realtime API and streamed transcription need no new resource. Long speech input needs a [bucket for the serving region](operations_configuration.md#aws-s3-regional-buckets), which is the same one the rest of the gateway already uses — except on generative voices, which speak up to 20,000 characters without one.

!!! warning "Behavior Changes"
    Review these before upgrading — they may change what existing clients or dashboards observe:

    - **A missing deployment permission is no longer reported as the caller's.** An `AccessDeniedException` on the gateway's own AWS calls reached clients as `403 permission_error` — which every OpenAI and Anthropic SDK reads as *their* key being refused. Every route now answers `503` `feature_unavailable`, with the server log naming the operation, model and permission. Clients matching `403` for a backend permission error should match `503`/`feature_unavailable` instead; a `403` now means only that [per-user attribution](operations_cost_management.md#per-user-attribution) is on and *that end user's* role was denied.
    - **Built-in web search now appears in usage and cost reporting.** Queries were recorded as nothing at all, so a measured turn under-reported its cost by 58%. Nothing AWS charges changed; what the gateway reports does. Web access is also an operator setting now ([`AWS_BEDROCK_EXTERNAL_WEB_ACCESS`](operations_configuration.md#bedrock-external-web-access)), defaulting to the previous behaviour.
    - **A request that would be answered without what it asked for is refused.** A `/v1/responses` `web_search` restricting its sources (`filters.allowed_domains`, `user_location`) was accepted and dropped, so answers came back sourced from domains the caller had excluded. Now a `400` on models that cannot serve it; [Bedrock Mantle](features.md#bedrock-mantle-models) models receive the options unchanged. The same rule governs [file search](api_openai_responses.md#file-search) filters and score thresholds.
    - **Two output-shaping hints that returned `400` now succeed.** `prediction` and `verbosity` on chat completions are accepted and dropped, as the Responses surface already did; `truncation="disabled"` is likewise accepted, while `truncation="auto"` is still refused.
    - **`/v1/responses` forwards undeclared request fields to the model**, as chat completions and messages already did, so the backend may refuse one it does not recognise. Conversely, client-side control fields no provider treats as parameters (LiteLLM's `drop_params` among them) are dropped rather than forwarded. Both are governed by [`EXTRA_MODEL_PARAMS_DENYLIST`](operations_configuration.md#extra-model-params-denylist) and [`EXTRA_MODEL_PARAMS_DROP_ALL`](operations_configuration.md#extra-model-params-drop-all).
    - **Attachments are measured against what the model actually accepts.** The old guard compared raw bytes where the backend enforces base64 length, so it was ~33% too permissive. Oversized attachments are now staged and referenced where the model reads from storage, or refused with `413` naming the size it accepts. Smaller attachments are unaffected — see [Attachment Size](features.md#attachment-size).
    - **The server's own connections follow the proxy environment.** `HTTPS_PROXY`, `HTTP_PROXY` and `NO_PROXY` were honoured by the AWS SDK and ignored by everything else, so a proxied deployment saw no [Bedrock Mantle](features.md#bedrock-mantle-models) models. Two connections deliberately still bypass it: container metadata, and the fetch of a caller-supplied URL, where a proxy would defeat address validation. See [proxied deployments](operations_deploy_advanced.md#proxied-deployments).
    - **A declared upload checksum is now verified.** The value was stored and never looked at, so a corrupted upload completed like a clean one. It covers the file's contents, **not** the storage layer's multipart identifier — declaring the latter is now refused.
    - **An unknown model name answers with a sentence, not the catalogue.** The `404` body carried every served identifier, roughly 2,500 characters. Clients that parsed it for a model list should call [`/v1/models`](api_openai_models.md).
    - **Bedrock Mantle is only probed in the Regions that serve it**, so a deployment listing others no longer warns at every start. An explicit [`AWS_BEDROCK_MANTLE_REGIONS`](operations_configuration.md#bedrock-mantle-regions) list is still used exactly as given.
    - **The container health probe's command changed.** Deployments that re-declare the probe instead of running the image's own — an [ECS task definition](operations_deploy_advanced.md#ecs-task-definition-example) among them — should take the command from the image.

#### :material-api: New APIs

| Provider                                                                        | Endpoint/Feature                                                                                                                      | AWS Backend                                                                                                                 |
|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/conversations`](api_openai_conversations.md) – create, retrieve, update and delete a conversation, list and manage its items, and continue it from the Responses API with the `conversation` parameter | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - session management |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/vector_stores`](api_openai_vector_stores.md) – attach files, follow the indexing as it progresses, then search by meaning with attribute filters and per-passage scores | ![Amazon S3](styles/logo_amazon_s3.svg){: style="height:20px;width:20px"} Amazon S3 Vectors, ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - embedding models |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/vector_stores`](api_openai_vector_stores.md#knowledge-base-stores) – address an Amazon Bedrock knowledge base you already run as a vector store, Bedrock managed or customer-managed: search it, attach, list, read and delete documents. Allowlisted per knowledge base, never created or deleted here | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Knowledge Bases |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`file_search` on `/v1/responses`](api_openai_responses.md#file-search) – a chat model answers from the stores you name, reporting the searches it ran and citing a `file_citation` per file it drew on | ![Amazon S3](styles/logo_amazon_s3.svg){: style="height:20px;width:20px"} Amazon S3 Vectors, ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Knowledge Bases |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/batches`](api_openai_batches.md) – run a JSONL file of chat completion or embedding requests asynchronously at the batch price: submit, poll, cancel, read the result files | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - batch inference |
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic** | [`/anthropic/v1/messages/batches`](api_anthropic_batches.md) – the same asynchronous, batch-priced run for the Messages API, results streamed back as JSONL; each request may name its own model, up to eight per batch | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - batch inference |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`WS /v1/realtime`](api_openai_realtime.md) – a live speech-to-speech session over one WebSocket, with a transcript of both sides, server-side turn detection or manual turns, barge-in, and G.711 for telephony | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Amazon Nova Sonic |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`POST /v1/realtime/client_secrets`](api_openai_realtime.md#ephemeral-client-secrets) – mint a short-lived, browser-safe credential carrying a session configuration; signed and stateless, so any instance verifies one minted by any other | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Amazon Nova Sonic |

!!! note "Limits worth knowing before building on these"
    **Realtime**: a session lasts at most 8 minutes and calls no tools, and a spoken answer is guardrail-checked once complete, so a blocked one may already have been partly heard ([coverage](api_openai_realtime.md#guardrail-coverage)). **WebRTC and SIP are not served** — put [LiveKit Agents or Pipecat](api_openai_realtime.md#transports) in front for a browser media path or a phone line. The [compatibility table](api_openai_realtime.md#feature-compatibility) lists every event the session does not emit.

    **Knowledge-base stores** address a knowledge base that already exists and refuse, naming why, anything that would reshape it — creating, deleting, renaming, expiry, chunking strategy, attribute rewrites and the file-batch routes. Attaching needs a **custom** data source. Retrieval scores are reported as the backend states them rather than rescaled into similarities, and unknown values are reported unknown rather than invented. See [Knowledge Base Stores](api_openai_vector_stores.md#knowledge-base-stores).

#### :material-microphone: Speech & Audio

| Provider                                                                        | Endpoint/Feature                                                                                                                      | AWS Backend                                                                                                                 |
|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/audio/speech`](api_openai_audio_speech.md#long-input) – up to 100,000 billed characters per request, 24× the upstream 4,096, with no API change and no new request field | ![Amazon Polly](styles/logo_amazon_polly.svg){: style="height:20px;width:20px"} Amazon Polly |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/audio/speech`](api_openai_audio_speech.md#long-input) – long input is spoken as it is synthesized instead of after a whole job finishes; generative voices reach 20,000 characters with no bucket at all, and each request takes whichever path can serve it, so long input is no longer tied to one voice | ![Amazon Polly](styles/logo_amazon_polly.svg){: style="height:20px;width:20px"} Amazon Polly |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/audio/transcriptions`](api_openai_audio_transcriptions.md#amazon-nova-sonic) – naming Amazon Nova Sonic transcribes at the lowest cost available here, streamed as it is recognized; `json` and `text` only, up to 10 minutes, no timestamps. No existing request is re-routed | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Amazon Nova Sonic |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/audio/translations`](api_openai_audio_translations.md) – Amazon Nova Sonic translates speech to English itself, in one request                 | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Amazon Nova Sonic |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/audio/transcriptions`](api_openai_audio_transcriptions.md#streaming) – `stream=true` returns each phrase as it is recognized, whenever the request names the language to expect; needs no bucket. `gpt-live-transcribe` is now an alias, and requests naming no language are unchanged unless [`AWS_TRANSCRIBE_STREAM_LANGUAGES`](operations_configuration.md#aws-transcribe-stream-languages) says which to expect | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/audio/transcriptions`](api_openai_audio_transcriptions.md) – per-language custom vocabularies and language models, so a request identifying between several languages can apply the right resources to each one instead of being refused. Accepted only where the backend would use them: alongside a single fixed language, where they would apply to nothing, they are still refused | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe |
| **stdapi.ai**                                                                    | [`AWS_TRANSCRIBE_OUTPUT_ENCRYPTION_KEY_ARN`](operations_configuration.md#aws-transcribe-output-encryption-key-arn) – encrypt a transcription's output with a key you name rather than the bucket's own. The job's request identifiers travel as the encryption context, so a key policy can be scoped to this workload instead of to the whole bucket | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe, AWS KMS |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/audio/translations`](api_openai_audio_translations.md) – the supported language pairs are read once at startup and checked before the call, so a pair that cannot be served is named as the request problem it is instead of surfacing as a failure after the audio was transcribed. The permission that reads them is optional: without it the check stays off and everything else works | ![AWS Translate](styles/logo_amazon_translate.svg){: style="height:20px;width:20px"} AWS Translate |

#### :material-account-key: Identity & Cost Attribution

| Provider                                                                        | Endpoint/Feature                                                                                                                      | AWS Backend                                                                                                                 |
|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| **stdapi.ai**                                                                    | [Amazon Cognito user pool tokens](operations_configuration.md#cognito-authentication) – accept access tokens instead of, or alongside, the API key, so each caller reaches the API with their own credential; validated in-process against the pool's published keys, with no AWS call on the request path | ![Amazon Cognito](styles/logo_amazon_cognito.svg){: style="height:20px;width:20px"} Amazon Cognito |
| **stdapi.ai**                                                                    | [`AUTHENTICATION_MODE`](operations_configuration.md#authentication-mode) – assert the posture rather than infer it: the server refuses to start when the selected method is not configured, or when a configured method would be silently ignored | ![Amazon Cognito](styles/logo_amazon_cognito.svg){: style="height:20px;width:20px"} Amazon Cognito |
| **stdapi.ai**                                                                    | [Authentication discovery for agents](operations_configuration.md#oauth-discovery) – an OAuth 2.0 protected resource metadata document, pointed at by every unauthorized response, so an MCP client finds the authorization server and the scope it needs without being configured for this deployment; published only once an authorization server is declared | ![Amazon Cognito](styles/logo_amazon_cognito.svg){: style="height:20px;width:20px"} Amazon Cognito, or any OAuth 2.0 authorization server |
| **stdapi.ai**                                                                    | [Per-user cost attribution](operations_cost_management.md#per-user-attribution) – model calls issued under a short-lived role session tagged with the caller, so AWS reports each end user's spend in Cost Explorer and the Cost and Usage Report, from the invoice rather than an estimate. Off by default; a deployment can also [require](operations_configuration.md#aws-bedrock-user-role-require-identity) every call to name its end user rather than bill it to the deployment | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock, AWS STS |
| **stdapi.ai**                                                                    | [Vector store cost reporting](operations_cost_management.md#vector-stores) – a search against a Bedrock-managed knowledge base is recorded and priced like every other billed unit; what cannot be accounted for is stated rather than approximated | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Knowledge Bases |

#### Platform Features

| Feature                                | Description                                                                                                                                                                             |
|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Aliases that carry configuration](operations_configuration.md#model-aliases-configuration) | A `MODEL_ALIASES` entry may map a public name to the target model *plus* the service tier, guardrail, metadata and extra parameters applied to requests naming it, so one model is published under several names with different policies. The plain-string form is unchanged, and a malformed alias stops startup naming itself rather than failing once per request |
| [Attachment size policy](features.md#attachment-size) | On the multimodal routes served by Amazon Bedrock, an attachment is measured before the request is built and travels inline or by reference according to the limits each model class declares. Staging is per model and per media kind: of the families measured, only the Amazon Nova families and TwelveLabs Pegasus accept a reference |
| Ephemeral secret signing key           | [`REALTIME_CLIENT_SECRET_KEY`](operations_configuration.md#realtime-client-secret-key) signs the Realtime API's client secrets. A deployment with an API key already shares one and needs nothing; one running with no API key at all should set it, or a secret minted by one instance fails to verify on another |
| Dual-stack container listener          | The image's bind address moved out of its command into `GRANIAN_HOST`, so a deployment that needs a dual-stack socket — an ECS service whose discovery record includes an AAAA record, for instance — sets one variable instead of replacing the whole command. The IPv4-only default is unchanged |
| Faster container health probe          | The probe ships as a module of the application itself, byte-compiled with the rest of the package and covered by the linters and the test suite; it speaks HTTP over a socket rather than pulling in 123 modules per probe, cutting roughly 250 ms of import work per run in the community image and halving its peak memory |
| OpenAI Daybreak models                 | Daybreak Red (GPT-5.6 Cyber) and Daybreak Blue (GPT-5.6 Sol) are served and priced with the rest of the GPT-5.6 family, image input included. Both answer on the Responses API through Bedrock's next-generation inference endpoint, in US East (Ohio) only, and both are gated on enrollment with OpenAI's Daybreak programme — an account without it does not see them in the catalogue at all |
| Capability discovery                   | The [model catalogue](api_search_models.md) advertises what this release added — speech to speech, its transcription and translation, the search surfaces, and whether a model can be used with the Batch API — filterable over HTTP and through the same tool an agent reads before it calls anything. Web search is credited to every model that provides it, not only to the family the last release added it for |
| [Durable vector store indexing](api_openai_vector_stores.md#durable-indexing) | [`AWS_SQS_VECTOR_STORE_QUEUE_URL`](operations_configuration.md#aws-sqs-vector-store-queue-url) hands indexing to an Amazon SQS queue you create, so a file keeps being indexed — and finishes — when the server that accepted it is replaced. Off by default; needs a standard queue with a dead-letter queue and the [durable indexing permissions](operations_iam_permissions.md#durable-vector-store-indexing) ([resilience](operations_resilience.md#vector-store-indexing)) |
| End-to-end client coverage             | The suite driving complete, unmodified third-party clients against a live gateway gains five: **LiteLLM**, **Docling Serve**'s vision pipeline, the **OpenAI Agents SDK** (realtime voice, conversations, web search, vector-store retrieval), and **LiveKit Agents** and **Pipecat** running the exact [WebRTC and telephony configurations](api_openai_realtime.md#transports) this documentation prints. Deployment guides follow for [LobeHub](use_cases_lobehub.md) and [RAGFlow](use_cases_ragflow.md) |

#### :material-bug: Fixes { #fixes-3 }

- **Vector store durability**: an attached file whose indexing was interrupted is now reported `failed` with a `last_error` instead of sitting `in_progress` for ever; a detached file leaves every listing and search immediately and is gone only once its passages are, so a server replaced mid-delete no longer leaves a deleted document answering searches; indexing is bounded for the whole server rather than per request, so memory and embedding quota no longer scale with the number of callers; and deployments that would rather the work finished than reported can [hand indexing to a queue](api_openai_vector_stores.md#durable-indexing)
- **Work a request left running is finished before the server stops**: a deployment, scale-in or Spot interruption used to drop temporary file cleanups, vector store indexing and live audio session releases with nothing in the logs. Shutdown now waits under [`SHUTDOWN_DRAIN_TIMEOUT`](operations_configuration.md#shutdown-drain-timeout) (10 seconds by default), settles whatever the deadline leaves, and counts it in the `stop` event. Raise it together with your container runtime's kill delay, never one alone
- **Streamed responses run the work they scheduled**: the drain was attached before the body produced a byte, so three leaks followed — a vector store searched only through streamed answers could expire mid-query, an expired store left a paid index behind, and a streamed transcription falling back to a job left its audio, transcript and job record behind on every request
- **Realtime speaks the released vocabulary, not the beta one**: item events were unparsable, the caller's transcript event was dropped for a missing field, and the item lifecycle clients wait on was never emitted. Barge-in did not work at all — the session refused truncation precisely while an answer was playing, the only moment it is ever sent. Truncate, retrieve and delete now work against the tracked conversation, a written turn is answered instead of timing out, and every answer reports the six response fields upstream always sends
- **Batch API**: listing a batch neither settled nor published it, so a client that only ever listed never had its usage recorded; every validation failure at submission was reported as an unsupported model, quota and role failures included; and the batch record was written only after the jobs started, leaving billable work running with nothing on disk to stop it
- **Knowledge-base stores answer in this API's own words**: attaching to a managed knowledge base always failed, listing its files always failed, deleted documents never left a listing, and internal bookkeeping reached the caller as attributes. A file refused because the corpus is maintained elsewhere is now told so in the store's terms, and a refused file is explained by the store that refused it rather than by one fixed sentence describing limits the caller never met
- **Cost reporting matches what AWS charges**: GPT-5.6 Luna was reported at five times its real cost and Terra a quarter over, since AWS repriced them on the model's own page rather than in the live price catalogue; a [Global cross-Region](operations_cost_management.md#routing-tier-pricing) call — the only way Luna, Sol and Terra are ever served — was costed at the In-Region rate, about 10% over; and moderation usage naming an alias matched no price at all. Every source page is now re-checked weekly
- **Reasoning and web search reach the models that serve them**: Amazon Nova 2 and DeepSeek V3 refused the token budget the Anthropic dialect requires, leaving [extended thinking](api_anthropic_messages.md#extended-thinking) unreachable on that route while the identical ask worked elsewhere; and a `web_search` sent to a GPT-5.6 model resolved to its non-Mantle twin travelled as an ordinary function tool, so no search ran and nothing said so — now a `400` naming both ways to route the model to the endpoint that serves it
- **The Messages surface reports what an answer cost and why it stopped**: refusals carry the policy category, the reasoning-token breakdown is reported, and service tier and per-TTL cache-creation counts are populated — most visibly on a batch, which claimed no tier while being billed as one
- **Audio, embeddings and attachments are bounded correctly**: inline audio was measured against raw bytes where the backend enforces the encoded length, so files between ~18.75 MB and 25 MB passed and were refused downstream; long text-to-speech now answers with the length a bucket-less deployment can honour; and models that embed one input per call no longer open a connection per chunk
- **The interactive documentation pages render with no outbound access**: [`/docs` and `/redoc`](operations_configuration.md#enable-docs) pulled the icon, Swagger UI, ReDoc and a web font from three third parties — blank pages in an air-gapped VPC, and elsewhere a report to those hosts of who was reading this API and when, running whatever a floating major version tag resolved to that day. Both are now served whole from the image, pinned to exact releases verified by SHA-256 at build time, with upstream licences beside them
- **MCP tools return what they produce**: every route answering with bytes was published as a tool an agent could call and then could not use. An image now arrives as an image and audio as audio, anything the protocol cannot carry arrives as a reference rather than failing, and the 4 MiB body cap that blocked image edits now follows [`MAX_INPUT_FILE_SIZE`](operations_configuration.md#max-input-file-size)
- **Addresses and listings are the ones this deployment serves**: a custom [route prefix](operations_configuration.md#openai-routes-prefix) still quoted default paths to [`search_models`](api_search_models.md) and to video job polling, neither recoverable client-side; and a file's `created_at` and its place in a listing came from two different clocks, so a [multipart upload](api_openai_files.md#uploads-api) sat among older files reporting a later time. The [Anthropic listing](api_anthropic_files.md#list-files) also answered oldest first, hiding every recent file, and now runs newest first as upstream does
- **Diagnostics name their cause**: an unreachable Region rendered six different conditions as one identical sentence, and the slow startup beside it was the container metadata lookup retrying, unreported; a request abandoned mid-flight left its OpenTelemetry trace current, so later work was recorded under a closed request's trace id; and behind a proxy, `client_ip` recorded the load balancer whatever [`PROXY_TRUSTED_HOSTS`](operations_configuration.md#proxy-trusted-hosts) allowed
- **The API describes itself, not the service behind it**: a synthesis limit credited to the service enforcing it, prices credited to their catalogue and a moderation route naming the engine underneath all shipped in the OpenAPI document and in the tool descriptions agents read before calling

#### Fixes & Maintenance (v1.16.1)

- Update `cryptography` to 50.0.1, rebuilt against OpenSSL 4.0.2

### v1.15.0 – Reliability, Performance & Feature Completeness

This release focuses on making the whole gateway better rather than just bigger. **Reliability and quality**: the largest correctness pass to date — three successive deep audits plus an independent full-branch review closed hundreds of fidelity gaps across all three API dialects, every fix pinned by tests and the whole surface validated by **real, unmodified client applications**. **Performance**: hot paths now run in compiled native code and independent work in parallel, [measurably cutting the gateway's processing overhead](features.md#performance). **Feature completeness**: existing capabilities are rounded out end to end — [**explicit prompt caching**](api_openai_chat_completions.md#prompt-caching), operator [**reasoning controls**](operations_configuration.md#chat-completions-reasoning-field), the Responses API [`prompt` parameter](operations_configuration.md#bedrock-allow-prompt-arn) from **Amazon Bedrock Prompt Management**, native mid-conversation system messages on Claude 4.8+, richer speech and transcription (Polly speech marks, Transcribe/Translate extras, generic Converse speech-to-text), Cohere `embedding_types` and Rerank v1 structured documents, guardrail enforcement on every route, and an inline guardrail-checks moderation backend.

!!! warning "Behavior Changes"
    Review these before upgrading — they may change what existing clients observe:

    - **Unsupported parameters are accepted and ignored, not rejected.** Parameters the AWS backends cannot honour (e.g. `known_speaker_*`, `partial_images`, unsupported image `quality`/`style`, programmatic tool calling) now behave like they do on OpenAI: the request succeeds, the parameter is dropped, and a warning is recorded in the request log. Requests that returned `400` on v1.14 may now succeed.
    - **Impossible combinations are now clean `400`s instead of silent degradation**: subtitle or diarized formats with `stream=true`, contradictory Amazon Transcribe settings, and web-search filters Nova grounding cannot apply are rejected with actionable messages.
    - **Speech output quality**: `wav`/`flac`/`aac` are now encoded from lossless PCM instead of Ogg Vorbis, and the default `pcm` output is resampled to 24 kHz for OpenAI parity (pass an explicit `SampleRate` to keep Polly's native rate). Same formats, different — better — bytes.
    - **Error responses no longer expose backend internals.** Server-side (`5xx`) error messages are generic with details kept in the server log, and Anthropic error types now match the official SDK exactly.
    - **Comprehend-backed moderation always analyses text as English** — the only language the AWS API accepts at runtime.
    - **A configured guardrail now applies to every route.** Embeddings, rerank, images, videos, and the audio routes enforce it through the ApplyGuardrail API ([route coverage](operations_configuration.md#route-coverage)) — requests that silently bypassed the guardrail on v1.14 may now return `400` (code `content_filter`) or masked text, and each check is billed as guardrail text units.
    - **SSRF protection covers every non-globally-reachable address.** With [`SSRF_PROTECTION_BLOCK_PRIVATE_NETWORKS`](operations_configuration.md#ssrf-protection-block-private-networks) enabled (the default), a user-supplied URL resolving to shared address space (100.64.0.0/10, used by EKS custom networking and Hybrid Nodes) or another special-purpose range is now rejected with `403`, alongside the RFC 1918 ranges.
    - **Usage reporting is additive but richer**: cached-token buckets are folded into `prompt_tokens` with `prompt_tokens_details` on every surface, and the Anthropic API now reports `cache_creation_input_tokens` (it was always `null` on v1.14).
    - **Two request-body keys are reserved.** `model_id` and `additional_request_fields` (plus `stop_sequences` on the legacy `/v1/completions`, where `stop` is the parameter to use) collide with the gateway's own request-building parameters: instead of being forwarded to Bedrock as [provider extras](api_openai_chat_completions.md#provider-specific-parameters), they return a `400 invalid_request_error` naming the key.
    - **The container health probe now respects `TRUSTED_HOSTS`.** The image's `HEALTHCHECK` requests `/health` with a `Host` header derived from [`TRUSTED_HOSTS`](operations_configuration.md#trusted-hosts) — a correct list keeps the container healthy with no extra entry. Deployments that re-declare the probe, such as an [ECS task definition](operations_deploy_advanced.md#ecs-task-definition-example), should run the image's own command. Note that a load balancer health check still sends the target's IP address as the `Host` and is rejected with `400` when the allow-list is enabled.

#### :material-cached: Explicit Prompt Caching

| Provider                                                                        | Endpoint/Feature                                                                                                                      | AWS Backend                                                                                                                 |
|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [Explicit cache breakpoints](api_openai_chat_completions.md#prompt-caching) on chat completions and responses, mapped to Bedrock `cachePoint` blocks with the per-request block budget enforced | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | `prompt_cache_options` – cache TTL control on chat completions and responses, honoured on models that support it                      | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API |

Prompt caching itself is not new — this release adds *explicit placement and lifetime control* on the OpenAI dialect, and fixes the caching plumbing that already existed (see [Fixes](#fixes-2)).

#### :material-brain: Reasoning Controls

| Provider                                                                        | Endpoint/Feature                                                                                                                      | AWS Backend                                                                                                                 |
|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| **stdapi.ai**                                                                    | [`CHAT_COMPLETIONS_REASONING_FIELD`](operations_configuration.md#chat-completions-reasoning-field) – return thinking text under `reasoning_content`, `reasoning`, or suppress it with `none`; applied identically to streamed deltas and final messages | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API & Mantle |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | OpenRouter-style `reasoning` request object accepted on chat completions (`effort`, `max_tokens`, `enabled`, `exclude`)               | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API & Mantle |

#### :material-api: New API Features

| Provider                                                                        | Endpoint/Feature                                                                                                                      | AWS Backend                                                                                                                 |
|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | Responses API `prompt` parameter – serve prompts stored in Bedrock Prompt Management, with versions and variables (opt-in, [`AWS_BEDROCK_ALLOW_PROMPT_ARN`](operations_configuration.md#bedrock-allow-prompt-arn)) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Prompt Management |
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic** / ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | Mid-conversation system messages forwarded natively on Claude 4.8+ and Claude 5 family models instead of being folded into the system prompt | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API & Mantle |
| **stdapi.ai**                                                                    | Guardrail asynchronous stream processing via the `X-Amzn-Bedrock-GuardrailStreamProcessingMode` request header                        | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Guardrails |
| **stdapi.ai**                                                                    | Configured guardrails enforced on every route: embeddings, rerank, images, videos, and audio now apply them via the ApplyGuardrail API ([route coverage](operations_configuration.md#route-coverage)) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Guardrails |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [Moderations](api_openai_moderations.md) `amazon.bedrock-runtime-guardrail-checks` model – inline guardrail content filter checks with no guardrail resource required, the new default fallback for `omni-moderation-*` in supported regions | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Guardrails |

#### :material-microphone: Speech & Audio

| Provider                                                                        | Endpoint/Feature                                                                                                                      | AWS Backend                                                                                                                 |
|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/audio/speech`](api_openai_audio_speech.md) – Polly `SpeechMarkTypes` for word/sentence/viseme/SSML timing marks                 | ![Amazon Polly](styles/logo_amazon_polly.svg){: style="height:20px;width:20px"} Amazon Polly |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/audio/transcriptions`](api_openai_audio_transcriptions.md) – Amazon Transcribe extra parameters: multi-language identification, custom vocabularies, PII redaction, and more | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/audio/transcriptions`](api_openai_audio_transcriptions.md) – `gpt-transcribe` context inputs: `keywords` and multi-language `languages`, with detected languages reported in the response | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/audio/transcriptions`](api_openai_audio_transcriptions.md) & translations – any Converse-capable speech-input Bedrock model transcribes through a generic default (Voxtral rebuilt on the Converse API); uploads outside the accepted formats are transcoded automatically | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/audio/translations`](api_openai_audio_translations.md) – AWS Translate `Formality`, `Profanity`, `Brevity`, and custom terminologies | ![Amazon](styles/logo_amazon.svg){: style="height:20px;width:20px"} AWS Translate |

#### :material-vector-combine: Cohere Embed & Rerank

| Provider                                                                        | Endpoint/Feature                                                                                                                      | AWS Backend                                                                                                                 |
|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| ![Cohere](styles/logo_cohere.svg){: style="height:20px;width:20px"} **Cohere**  | [`embedding_types`](api_cohere_embed.md) – quantized (`int8`/`uint8`/`binary`/`ubinary`) and `base64` embeddings on both embed routes; image embedding metadata reported | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - embedding models |
| ![Cohere](styles/logo_cohere.svg){: style="height:20px;width:20px"} **Cohere**  | [Rerank v1](api_cohere_rerank.md) – structured JSON documents with `rank_fields` selection                                            | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Rerank API |

#### Platform Features

| Feature                                | Description                                                                                                                                                                             |
|----------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Stateless MCP transport                | The [MCP server](operations_configuration.md#mcp-model-context-protocol) can serve `/mcp` without server-side sessions ([`MCP_STATELESS_HTTP`](operations_configuration.md#mcp-stateless-http)), so any replica may answer any request, alongside a `GET /ping` health probe kept out of request logs                                                                              |
| `Retry-After` on `429`                 | Throttled responses advertise the region router's computed backoff, so well-behaved clients retry exactly when capacity returns                                                          |
| AWS request-ID correlation             | Request logs record every AWS API call's request ID (and incoming ALB/CloudFront trace headers), so a gateway request ties directly to CloudTrail and AWS support cases                  |
| Programmatic tool calling types        | The OpenAI SDK's programmatic tool calling type surface parses on every request union, accepted and ignored on models without the capability                                              |
| [Performance](features.md#performance) | Hot paths run in compiled native code and independent work runs in parallel: a 1 MB request costs 30% less CPU, multi-image generations finish in the time of the slowest image, and every optimization is pinned by regression tests |
| MCP context efficiency                 | Tool schemas hide parameters MCP callers cannot use (streaming modes, token-level tuning, caller identifiers) and tool results return as compact JSON, cutting the tokens each call costs the calling agent; every exposed MCP tool is exercised end to end through a real MCP client in the test suite |
| Slimmer container image                | The community image shrinks from 230 MB to 156 MB: unused dependency payloads are removed (language-name data, `cryptography`, `rich`/`typer`, uvicorn extras) and AWS service models are pruned to the services actually used, guarded by a build-time smoke test; the package inventory stays complete for vulnerability scanners (self-built ffmpeg registered, Python package metadata retained) and every redistributed component keeps its [licence and notice files](operations_licensing.md#frequently-asked-questions) |
| Metadata filter for MCP clients        | Listing stored chat completions accepts the metadata filter as a single [`metadata={"key": "value"}` JSON object](api_openai_chat_completions.md#stored-chat-completions) as well as the OpenAI SDK's `metadata[key]=value` pairs, so clients that can only send one query parameter per field — MCP tool calls among them — can filter too |

#### :material-robot-happy: Verified with Real Clients

Compatibility claims in this release are backed by a new test tier that runs **complete, unmodified third-party client software against a live gateway** — not just HTTP assertions. Coding agents (Claude Code, Codex, pi, Qwen Code), the n8n workflow platform, Open WebUI, Home Assistant's voice bridge, a Haystack RAG pipeline, and the LangChain and pydantic-ai libraries drive real multi-turn tool-calling, retrieval, and speech sessions across dozens of models and all three API dialects, in isolated sandboxes. Alongside them, every served model is **empirically probed** for the parameters it genuinely honours, with the results recorded and pinned by tests. See [Quality Assurance](features.md#quality-assurance) for the full methodology.

#### :material-bug: Fixes { #fixes-2 }

Three audit passes and an independent full-branch review closed over a hundred fidelity gaps. The user-visible highlights:

- **MCP tool calls match their schemas**: union-typed parameters no longer advertise a contradictory single `type` (which made valid string arguments randomly fail schema validation), and the JSON image edit/variation bodies accept the plain string references the tool schemas advertise
- **Clean errors on log-exempt paths**: a 404 or 405 on paths kept out of request logs (e.g. `/favicon.ico`, auto-requested by every browser visiting `/`) returned a 500 with a traceback instead of the JSON error envelope
- **Prompt caching plumbing**: Anthropic `cache_control` breakpoints land on their marked block; cache reads and writes are counted, priced, and reported consistently in responses, request logs, and `count_tokens`
- **Reasoning**: `reasoning_effort="max"` accepted end to end; thinking text returned by Bedrock Mantle models is surfaced on every API instead of dropped; unsigned reasoning is no longer replayed to models that would reject it
- **Streaming parity**: streamed and non-streamed results are now identical (tool-call indices, same-role message merging, text concatenation); mid-stream errors emit proper error events on every API instead of ending streams silently; redacted thinking and web-search results round-trip exactly as native Anthropic emits them; failed generations return `502` instead of an empty `200`
- **Routing & billing**: a read timeout on an already-sent request is no longer re-invoked in another region (no double billing), on Converse and Mantle alike; region failover now tries each candidate region at most once per request instead of looping back over regions it just marked as throttled, so a single request can no longer escalate a region's quota backoff toward the one-hour ceiling; Bedrock prompt-router usage is billed against the actually-invoked model; the [price card](api_model_pricing.md) reprices a standard-tier row served as a tier fallback at the rate that tier actually bills; `store=true` degrades gracefully with a logged warning in regions without the session API
- **Responses API parity**: the type surface is synchronized with the current OpenAI SDK (tool fields, error codes, tool-call `caller` provenance); Anthropic `count_tokens` counts exactly what generation sends, and error bodies carry the `request_id`
- **Audio & images**: the transcoding pipeline is fully bounded — a stalled or failed encode returns a clean error instead of holding the connection open; multipart forms bind every list field the OpenAI SDK sends; `size="auto"` works on generation, edits, and variations; `zh-TW`/`pt-PT` stay distinct in translation; PII-redacted transcripts are read from the key Amazon Transcribe actually writes; Polly voice auto-selection is deterministic; batch-purpose files apply the documented 30-day default expiry, and an expired file now disappears from [file listings](api_openai_files.md#upload-with-expiry) instead of being listed with an entry that 404s on retrieve

### v1.14.0 – Bedrock Mantle, Video Generation, Cohere APIs, Moderation & Stored Conversations

This release adds enabled-by-default [**Amazon Bedrock Mantle** support](features.md#bedrock-mantle-models) — models served by the Bedrock Mantle endpoint (OpenAI GPT-5.4/5.5/5.6, xAI Grok 4.3, Google Gemma 4, Qwen3, GLM, DeepSeek, MiniMax, Kimi, Nemotron, and more) become available through all four text APIs, with transparent API conversion, native stored conversations, and independent throughput quotas. It also turns stdapi.ai into a three-dialect gateway with the new **Cohere-compatible API** ([Rerank](api_cohere_rerank.md) and [Embed](api_cohere_embed.md)), adds the OpenAI-compatible [**Videos API**](api_openai_videos.md) for asynchronous video generation, [**content moderation**](api_openai_moderations.md) backed by Amazon Bedrock Guardrails or Amazon Comprehend toxicity detection, **stored responses and chat completions** with `store=true`, `previous_response_id` multi-turn continuation, and a full list/retrieve/update/delete lifecycle on Amazon Bedrock session storage, and [**conversation compaction**](api_openai_responses.md#conversation-compaction). The Responses API gains [**extended reasoning**](api_openai_responses.md#extended-reasoning): Bedrock `reasoningContent` now surfaces as native reasoning output items, both non-streaming and streamed, with signatures and redacted payloads round-tripping through an `encrypted_content` envelope. A broader compatibility pass brings request/response parity closer to the OpenAI SDK — hosted and agent tool types (web search, computer use, custom tools) are now accepted and ignored instead of rejected, streams correctly terminate with `response.incomplete`/`response.failed`, cached tokens are counted in `input_tokens`, and citation annotations are emitted with their streaming events — validated end-to-end against the OpenAI Codex CLI as an agent client. Operations gain a [model pricing API](api_model_pricing.md), multi-region failover for every AWS AI service, fault-tolerant startup, real AWS-billed usage and costs in request logs (optionally exported as CloudWatch metrics), and a [security hardening pass](#security-hardening) covering SSRF protection, input validation, and log/error redaction.

!!! warning "New Required IAM Permissions"
    v1.14.0 requires two new IAM permissions:

    - **`bedrock:Rerank`** — needed for the [Cohere-compatible Rerank API](api_cohere_rerank.md) (`/cohere/v2/rerank`). See [IAM Permissions](operations_configuration.md#bedrock-iam).
    - **`bedrock:ListAsyncInvokes`**, plus **`bedrock:ListTagsForResource`** on `arn:aws:bedrock:*:*:async-invoke/*` — needed for `GET /v1/videos` (listing video generation jobs across regions). See [IAM Permissions](operations_configuration.md#bedrock-iam).

    Ensure your IAM role or user policy includes both statements before upgrading to v1.14.0.

    !!! note "Session storage and Comprehend permissions already covered"
        The IAM permissions for [stored responses/chat completions](operations_configuration.md#bedrock-session-storage-optional) (`bedrock:CreateSession` and related session actions) and [Comprehend-based moderation](operations_configuration.md#iam-permissions) (`comprehend:DetectToxicContent`) were already added to the official [stdapi-ai Terraform module](https://github.com/stdapi-ai/terraform-aws-stdapi-ai) ahead of this release. Deployments using a hand-written policy still need to add those statements if they haven't already. Without the session permissions, `store=true` (previously accepted and ignored) is still ignored — a warning is recorded in the request log instead of failing the request.

#### :material-layers-triple: Amazon Bedrock Mantle

Enabled-by-default support ([`AWS_BEDROCK_MANTLE_ENABLED`](operations_configuration.md#bedrock-mantle-enabled)) for models served by the **Amazon Bedrock Mantle** endpoint — OpenAI GPT-5.4/5.5/5.6 (Sol, Terra, Luna), xAI Grok 4.3, Google Gemma 4, Qwen3, GLM 4.x/5, DeepSeek V3.x, MiniMax M2.x, Kimi K2.5, Nemotron, and more — alongside the classic Bedrock Converse catalog:

- All four text APIs (chat completions, responses, messages, legacy completions) are served for every Mantle model — native passthrough where the model supports the API upstream, transparent conversion otherwise
- Models available on both bedrock-runtime and Mantle are served by bedrock-runtime by default; [`AWS_BEDROCK_MANTLE_PREFERRED_MODELS`](operations_configuration.md#bedrock-mantle-preferred-models) or the opt-in `x-stdapi-service` request header ([`AWS_BEDROCK_MANTLE_SERVICE_HEADER`](operations_configuration.md#bedrock-mantle-service-header)) route them through Mantle instead — e.g. to tap Mantle's independent throughput quotas
- Native Mantle stored conversations on `/v1/responses` (`store`, `previous_response_id`, retrieval and deletion) — 30-day retention, region-local, project-scoped
- Multi-region failover and quota backoff across [`AWS_BEDROCK_MANTLE_REGIONS`](operations_configuration.md#bedrock-mantle-regions), matching classic Bedrock region routing
- Authentication via short-term bearer tokens derived from the server's AWS credential chain — no static secrets
- Usage recorded and priced at bedrock-mantle rates, including cached tokens and service tiers
- Optional Bedrock Project/Workspace attribution for cost tracking via [`AWS_BEDROCK_MANTLE_PROJECT`](operations_configuration.md#bedrock-mantle-project), with per-request override ([`AWS_BEDROCK_ALLOW_MANTLE_PROJECT_OVERRIDE`](operations_configuration.md#bedrock-allow-mantle-project-override)) through the `OpenAI-Project` / `anthropic-workspace` header

[:octicons-arrow-right-24: Bedrock Mantle Models](features.md#bedrock-mantle-models)

!!! warning "Additional IAM Permissions (opt-in feature)"
    Enabling `AWS_BEDROCK_MANTLE_ENABLED` requires the `bedrock-mantle:CreateInference`, `bedrock-mantle:GetInference`, `bedrock-mantle:DeleteInference`, `bedrock-mantle:ListModels`, `bedrock-mantle:GetModel`, and `bedrock-mantle:CancelInference` permissions on `arn:aws:bedrock-mantle:*:*:project/*`, plus `bedrock-mantle:CallWithBearerToken` on `*`. See [IAM Permissions](operations_iam_permissions.md#bedrock-mantle-iam).

#### :material-api: New APIs

| Provider                                                                        | Endpoint/Feature                                                                                          | AWS Backend                                                                                                                       |
|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/videos`](api_openai_videos.md) – create, poll, list, download, and delete video generation jobs      | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Amazon Nova Reel, Luma Ray 2 |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/moderations`](api_openai_moderations.md) – text and image content classification                     | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Guardrails, Amazon Comprehend |
| ![Cohere](styles/logo_cohere.svg){: style="height:20px;width:20px"} **Cohere**  | [`/cohere/v2/rerank`](api_cohere_rerank.md) – document reranking (Amazon Rerank 1.0, Cohere Rerank 3.5)    | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Rerank API                   |
| ![Cohere](styles/logo_cohere.svg){: style="height:20px;width:20px"} **Cohere**  | [`/cohere/v2/embed`](api_cohere_embed.md) – embeddings over all Bedrock embedding models                   | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - embedding models             |
| **stdapi.ai**                                                                    | [`/model_pricing`](api_model_pricing.md) – exact AWS unit prices per model                                 | AWS Price List API                                                                                                                  |

#### :material-brain: Extended Reasoning

| Provider                                                                        | Endpoint/Feature                                                                                                       | AWS Backend                                                                                                             |
|----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/responses`](api_openai_responses.md#extended-reasoning) – Bedrock `reasoningContent` returned as reasoning output items | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | Streaming `response.output_item.added` / `response.reasoning_text.delta` / `.done` events for reasoning content       | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API       |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | `include=["reasoning.encrypted_content"]` – signature/redacted round-trip for multi-turn reasoning continuation       | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API       |

#### :material-chat: Conversations

| Provider                                                                        | Endpoint/Feature                                                                                                     | AWS Backend                                                                                                             |
|----------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | `store=true` + `GET/DELETE /v1/responses/{id}`, input items listing, and `previous_response_id` continuation          | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - session management |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | `POST /v1/responses/{id}/cancel` – endpoint parity for the cancel lifecycle (always fails for session-stored responses, which never run in background mode; Mantle-stored responses are cancelled upstream) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - session management |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | `store=true` + `GET/DELETE /v1/chat/completions/{id}`, `GET /v1/chat/completions` listing, `POST /v1/chat/completions/{id}` metadata updates, and input messages listing | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - session management |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | [`/v1/responses/compact`](api_openai_responses.md#conversation-compaction) – stateless conversation compaction        | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API       |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**  | `moderation` request parameter on chat completions and responses, with results reported in the response               | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Guardrails         |

#### Platform Features

| Feature                                       | Description                                                                                                                                                                        |
|-----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Claude 5 models                               | Explicit support for the Claude 5 generation — **Opus 5**, **Sonnet 5**, **Fable 5**, and **Mythos** — with each model's server tool set and reasoning configuration matched to what Bedrock actually accepts (Opus 5 exposes no computer use tool; Fable and Mythos always reason and reject a disabled configuration). Model matching covers unreleased versions of each family, so a new minor or major release inherits its family's behavior instead of a generic fallback. Validated end-to-end across the full Claude feature matrix, from Claude 4.5 through Claude 5 |
| Multi-region AWS AI services                  | Automatic multi-region failover for Amazon Polly, Transcribe, Translate, and Comprehend (per-engine voice discovery, co-located Transcribe buckets, latency-ordered region pools) |
| Fault-tolerant, faster startup                | Unreachable Bedrock regions or Polly engines no longer abort startup; they are skipped with a warning and retried on the next refresh — and startup is faster overall             |
| Usage & cost tracking                         | Request logs report the usage actually billed by AWS with its cost computed from live AWS pricing, optionally exported as CloudWatch metrics ([`CLOUDWATCH_METRICS`](operations_logging_monitoring.md#cloudwatch-metrics-emf)); the previous token estimation is removed and its `TOKENS_ESTIMATION*` settings are deprecated and ignored |
| [Cost attribution](operations_cost_management.md#aws-cost-attribution) | Request, server, and user correlation metadata is now attached to every synchronous Bedrock inference call — the `InvokeModel` family included, not only `Converse` — so Bedrock invocation logs can be filtered and costs attributed per request or per user |
| Smaller container images                      | The published images shrink by around 40% — 413 MB to 253 MB for the AWS Marketplace image, 377 MB to 230 MB for the community image — cutting pull time and storage. ffmpeg is now built with only the audio encoders the server uses, and the unused OpenTelemetry gRPC exporter is no longer installed |
| Video retention (`AWS_S3_VIDEOS_EXPIRES_AFTER`) | Optional retention period for generated videos, reported as `expires_at` and enforced on download                                                                               |
| Upload expiry (`expires_after`)               | Multipart upload sessions honor the OpenAI `expires_after` policy on the resulting file                                                                                           |
| Session storage encryption                    | Optional KMS key for Amazon Bedrock session storage (`AWS_BEDROCK_SESSION_ENCRYPTION_KEY_ARN`)                                                                                       |
| Proxy trust (`PROXY_TRUSTED_HOSTS`)           | `X-Forwarded-*` headers are only honored when sent by a trusted reverse-proxy address                                                                                             |
| Input file size limit (`MAX_INPUT_FILE_SIZE`) | Optional cap on the size of downloaded/decoded input files, with bounded download concurrency (`MAX_CONCURRENT_INPUT_DOWNLOADS`)                                                  |
| Legacy model opt-in fix                       | `AWS_BEDROCK_LEGACY` now also exposes models whose AWS legacy date has already passed (e.g. Amazon Nova Reel)                                                                     |

#### :material-shield-lock: Security Hardening

- **MCP transports and `/search_models` now require authentication** when an API key is configured — clients that relied on these endpoints being open must now send the API key
- SSRF protection hardened against IP-literal encoding and DNS-rebinding bypasses on URL file inputs
- `s3://` file inputs are restricted to the server's allowed buckets, and multipart upload filenames are validated
- Decoded image size is capped against decompression-bomb payloads
- ARNs and AWS account IDs are redacted from client-facing error messages, and presigned URL signatures are stripped from logs and traces
- An empty resolved API key (e.g. a blank secret value) now disables authentication cleanly instead of matching an empty bearer token, and CORS no longer allows credentialed cross-origin requests
- **Reduced container attack surface**: the images no longer carry the video codec, X11 and font libraries that a distribution ffmpeg package links — x264, x265, AOM, dav1d, SVT-AV1 and SDL2 among them — nor the gRPC stack. None were reachable from the audio transcoding ffmpeg is used for, and they accounted for the bulk of the images' third-party native code. ffmpeg is built from the same version the base distribution ships, with only the audio encoders in use, and enables no GPL-licensed component

#### :material-robot-outline: Agent SDK Compatibility

The Responses API request/response surface was audited and hardened against the OpenAI SDK and real agent clients, end-to-end tested against the **OpenAI Codex CLI**:

- Hosted and agent tool types (`web_search`, `computer_use`, `file_search`, `custom`/`namespace` tools, and other items without a Bedrock equivalent) are now accepted and dropped instead of rejected with `400`, preserving compatibility with existing agent tooling
- Streaming responses now correctly terminate with `response.incomplete` or `response.failed` (matching upstream behavior) instead of always reporting `response.completed`
- Mid-stream errors emit the spec-compliant `error` SSE event
- `input_tokens` usage now includes cache read/write tokens, matching OpenAI's accounting
- `url_citation` annotations are emitted alongside their streaming events
- Echoed reasoning items tolerate the field variations produced by different SDKs and agent clients

#### :material-bug: Fixes

- Rerank models are no longer incorrectly advertised on Converse-based chat routes and MCP tools
- Fixed per-request model parameter overrides (`default_model_params`) occasionally leaking into subsequent requests for the same model
- 5xx provider errors now report server-side error types (`server_error`/`api_error`) in OpenAI and Anthropic error envelopes instead of `invalid_request_error`
- Unknown paths (`404`) and wrong methods (`405`) now return the error envelope of the API family they were sent to, instead of the framework's default `detail` payload
- The Anthropic Messages API now returns `404` instead of `400` for an unknown model, matching the upstream API, and rejects a `top_p` above `1.0`
- Audio transcription returns plain text for `response_format=text` and now defaults `verbose_json` to segment timestamps
- Responses API usage reports `input_tokens_details.cache_write_tokens`, which recent OpenAI SDKs require to parse a response
- Files API listing and cursor pagination order by creation time again: file IDs now use an order-preserving alphabet, where the previous one could sort a newer file first. IDs issued before this release keep working, but sort among themselves as before until they expire
- Newer Anthropic client request fields (free-form JSON Schema keywords in tool `input_schema`, adaptive thinking `display`) are accepted instead of rejected in strict validation mode
- Amazon Nova 2 no longer fails on `max_tokens` combined with high reasoning effort (the cap is dropped with a logged warning)
- Explicit cache points are kept off tool-related content blocks for models without tool caching support
- The Files API unavailable error no longer exposes the S3 bucket configuration detail
- Fixed input files from one request occasionally leaking into later requests served by the same connection, which could fail those requests with internal errors
- Anthropic Messages streams now emit an empty tool-input delta for tool calls without arguments, so SDK stream accumulators no longer fail on argument-less tool calls
- JSON-body image edit and variation requests now accept the `model` field instead of rejecting the request
- Model listings now report `service: "AWS Bedrock Runtime"` for classic Bedrock models (previously `"Amazon Bedrock"`), distinguishing them from `"AWS Bedrock Mantle"`
- High reasoning effort now maps to the intended thinking-token budget on Anthropic Claude models (the budget factor was previously miscomputed)
- Setting `log_level` to `disabled` now suppresses all log output as documented, instead of publishing every event
- Server startup no longer fails when the ECS container metadata endpoint answers slowly, which could prevent small Fargate tasks from starting: the lookup is retried, then falls back to the STS caller identity with a startup warning
- Multipart upload parts are numbered from the parts already stored in S3 instead of a per-instance counter: with several server instances behind a load balancer, two parts of one upload could be given the same number, overwriting each other and failing the upload
- Multi-region failover now covers a region that does not offer the service at all: with no [`AWS_COMPREHEND_REGION`](operations_configuration.md#aws-comprehend-region) set, a Bedrock region without Amazon Comprehend moves on to the next one as documented, instead of failing language detection and Comprehend moderation

---

### v1.13.0 – Terraform Module Compliance & Security Hardening

This release focuses on the [stdapi-ai Terraform module](https://github.com/stdapi-ai/terraform-aws-stdapi-ai) and its child modules — [VPC](https://github.com/JGoutin/terraform-aws-vpc), [KMS](https://github.com/JGoutin/terraform-aws-kms-key), and [ECS Fargate](https://github.com/JGoutin/terraform-aws-ecs-fargate) — adding detailed AWS Security Hub control documentation and closing several compliance gaps: default security group lockdown, ALB access logging, EFS POSIX user enforcement with native backups, and optional compliance/GuardDuty/DNS Firewall VPC integrations. All four modules now also accept a `tags` variable for custom resource tagging.

!!! info "Documentation-first release"
    Every module README now includes a full Security Hub Foundational Security Best Practices (FSBP) control mapping. See [Authentication & Security](operations_authentication_security.md#aws-security-hub-guardduty-dns-firewall-integration) for a summary and links to each module.

#### :material-bug: Fixes

- Added the missing `1h` and `5m` values to `PromptCacheRetention` for Bedrock-specific prompt cache TTLs in the OpenAI Responses API

#### :material-shield-star: Security Hub & Compliance Hardening

| Feature                                 | Module                           | Description                                                                                                                                                                                                      |
|-----------------------------------------|----------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Security Hub FSBP control documentation | VPC, KMS, ECS Fargate, stdapi-ai | Per-control (pass/fail/conditional/N-A) tables added to each module README                                                                                                                                       |
| Default security group lockdown         | VPC                              | New `aws_default_security_group` resource revokes all default ingress/egress rules (EC2.2 / CIS 5.4)                                                                                                             |
| VPC Flow Logs retention                 | VPC                              | Default retention increased from 7 to 365 days (EC2.6)                                                                                                                                                           |
| Compliance VPC endpoints                | VPC                              | New `compliance_vpc_endpoints_enabled` variable adds ECR, SSM, SSM Contacts, and SSM Incidents interface endpoints                                                                                               |
| GuardDuty VPC endpoint                  | VPC                              | New `guardduty_vpc_endpoint_enabled` variable adds the `guardduty-data` interface endpoint                                                                                                                       |
| Route 53 Resolver DNS Firewall          | VPC                              | New `dns_firewall_enabled` variable blocks/alerts on DNS queries to known-malicious domains (AWS Managed Domain Lists, plus DGA/DNS-tunneling detection via `dns_firewall_advanced_enabled`); dedicated VPC only |
| ALB access logging                      | stdapi-ai                        | New `alb_access_logging_enabled` variable (default `true`) logs ALB access to a dedicated, encrypted S3 bucket                                                                                                   |
| EFS POSIX user enforcement              | ECS Fargate                      | `mount_points` now accepts an `efs_posix_user` object to enforce a POSIX identity on EFS access points (EFS.4)                                                                                                   |
| EFS native backups                      | ECS Fargate                      | New `mount_points_efs_backup_enable` variable enables native EFS automatic backups, independent of the existing AWS Backup plan (EFS.7)                                                                          |
| Resource tagging                        | VPC, KMS, ECS Fargate, stdapi-ai | New `tags` variable propagates custom tags to nearly all created resources (IAM.24 / EC2.48)                                                                                                                     |

#### :material-cog-outline: Other Infrastructure Changes

| Feature                       | Description                                                                                                                                                                                                         |
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| AWS provider version bump     | Requirement raised to `>= 6.27.0` across all four modules                                                                                                                                                           |
| S3 object tag rename          | Files API objects and the corresponding Terraform lifecycle rule now use the `stdapi-ai.expires` tag key instead of `expires`; a temporary backward-compatible rule still expires legacy-tagged objects             |
| `aws-apn-id` resource tagging | AWS resources created at runtime (Bedrock async jobs, Transcribe jobs, S3 objects) are tagged with `aws-apn-id`, the standard AWS Marketplace attribution tag — an internal, vendor-side tag, not user-configurable |

#### :material-robot-outline: MCP Token Optimization

- Significantly reduced the size of MCP tool descriptions across the API, lowering the token cost of every AI agent session connected to this server
- No change in functionality: all parameter constraints and usage guidance remain intact

---

### v1.12.0 – Completions API, Video Understanding & File References

This release adds the OpenAI-compatible [`/v1/completions`](api_openai_completions.md) endpoint for text-first coding agents and legacy completion clients, **TwelveLabs Pegasus** video understanding for analyzing `video/*` inputs in chat completions, and an input token counting endpoint for the Responses API. Files uploaded through the Files API can now be referenced anywhere a URL is accepted using the new `file-id:` URI scheme. The Anthropic Messages API now accepts `system`-role messages (merged into the system prompt for compatibility), reasoning can be explicitly enabled or disabled, and a new `DEFAULT_MODEL_SERVICE_TIERS` setting applies per-model service tiers automatically.

#### :material-chat: Chat Completions

| Provider                                                                                     | Endpoint/Feature                                                                | AWS Backend                                                                                                             |
|----------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**               | `/v1/completions` – text completion endpoint for text-first coding agents       | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models  |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**               | `/v1/responses/input_tokens` – input token counting                             | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - CountTokens API    |
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic**      | `/v1/messages` – accepts `system`-role messages (merged into the system prompt) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Claude models      |
| ![Twelve Labs](styles/logo_twelvelabs.svg){: style="height:20px;width:20px"} **Twelve Labs** | Pegasus video understanding (`video/*` inputs)                                  | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - TwelveLabs Pegasus |

#### :material-microphone: Speech & Audio

| Provider                                                                       | Endpoint/Feature                                                  | AWS Backend                                                                                  |
|--------------------------------------------------------------------------------|-------------------------------------------------------------------|----------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/audio/speech` – case-insensitive voice names & default model | ![Amazon Polly](styles/logo_amazon_polly.svg){: style="height:20px;width:20px"} Amazon Polly |

#### Platform Features

| Feature                                                     | Description                                                                                                                                                                                                                               |
|-------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `file-id:` URI scheme                                       | Reference Files API uploads via `file-id:<file-id>` anywhere a URL is accepted — embeddings, audio transcription/translation, chat, images, and messages                                                                                  |
| Default model service tiers (`DEFAULT_MODEL_SERVICE_TIERS`) | Automatically apply a per-model service tier (`default`, `flex`, `priority`, `reserved`) when none is provided in the request                                                                                                             |
| Explicit reasoning enable/disable                           | Reasoning/thinking can now be explicitly enabled or disabled via request parameters                                                                                                                                                       |
| Service tier & guardrail support for Pegasus                | TwelveLabs Pegasus requests honor `service_tier` and Bedrock Guardrail configuration                                                                                                                                                      |
| MCP speech streaming defaults to SSE                        | `/v1/audio/speech` defaults `stream_format` to `sse` when invoked as an MCP tool for broader client compatibility                                                                                                                         |
| Full regional S3 bucket handling                            | The Terraform module resolves regional S3 buckets via resource-level region (requires AWS provider >= 6.0.0)                                                                                                                              |
| Reliable cross-region model identifiers                     | Region routing no longer fails intermittently with "The provided model identifier is invalid": a region whose inference profile is missing or not yet propagated is skipped, and a geo-scoped profile is never sent to a different region |

---

### v1.11.0 – MCP Server, Agent Discovery & Model Search (with v1.11.1–v1.11.4 maintenance updates)

This release introduces a **Model Context Protocol (MCP) server**, making all stdapi.ai API endpoints directly accessible as MCP tools for AI agents and agentic workflows. A new `/search_models` endpoint enables precise discovery of models by route, MCP tool, region, streaming support, and legacy status. Agent-friendly discovery metadata is now exposed via RFC 8288 Link headers and an RFC 9727 machine-readable API catalog at `/.well-known/api-catalog`. Endpoints that previously required binary `multipart/form-data` uploads now also accept an `application/json` body for MCP and HTTP client compatibility. The Anthropic Messages API now accepts `xhigh` as a `reasoning_effort` value.

#### :material-robot-outline: MCP Server

| Feature                            | Description                                                                                                                                        |
|------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| MCP server (Streamable HTTP & SSE) | All API endpoints exposed as MCP tools; Streamable HTTP and SSE transports can be independently enabled or disabled via configuration              |
| Configurable MCP tool exposure     | Individual MCP tools can be selectively enabled or restricted via configuration                                                                    |
| JSON body for binary endpoints     | Audio transcription, audio translation, and image edit endpoints now accept `application/json` with files as base64, data URI, HTTP URL, or S3 URI |

#### :material-magnify: Model Search

| Feature          | Description                                                                                                                                                                                                                                                                                      |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `/search_models` | New official endpoint to filter models by route, MCP tool name, input/output modalities, region, streaming, and legacy status; returns richer metadata than `/v1/models` or Anthropic `/v1/models`, designed for LLM-driven model selection (replaces BETA and undocumented `/available_models`) |

#### :material-access-point: Agent Discovery

| Feature                                               | Description                                                                                   |
|-------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| RFC 8288 Link headers                                 | Root (`/`) endpoint returns Link headers for resource discovery                               |
| RFC 9727 API catalog (`/.well-known/api-catalog`)     | Machine-readable API catalog for automated agent and tool discovery                           |
| MCP Server Card (`/.well-known/mcp/server-card.json`) | Advertises available MCP transports and capabilities to AI agents (SEP-1649)                  |
| `robots.txt` AI signals                               | Updated `robots.txt` with `Content-Signal` directives and explicit `/.well-known/` allow rule |

#### :material-chat: Chat Completions & Messages

| Provider                                                                                | Endpoint/Feature                                           | AWS Backend                                                                                                            |
|-----------------------------------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic** | `/v1/messages` `reasoning_effort=xhigh` support            | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Claude models     |

#### :material-label-off: Deprecation Mappings

- Added automatic fallback for `amazon.nova-reel-v1:0` and `anthropic.claude-3-haiku-20240307-v1:0` to their respective replacements

#### Fixes

- Fix reasoning token double-counting in usage calculation in OpenAI Responses API adapter
- Fix missing `file_id` inputs for image and file processing in OpenAI Responses API adapter
- Remove `store` parameter from unsupported validations in chat completions to ensure client compatibility

#### Fixes & Maintenance (v1.11.1–v1.11.4)

**v1.11.1**

- Make `max_tokens` optional in Anthropic `/v1/messages` to align with the Anthropic API specification
- Remove unsupported reasoning configuration checks for broader client compatibility
- Rename `/v1/responses` route tag from "Responses" to "Chat" in OpenAPI documentation for consistency

**v1.11.2-v1.11.3**

- Add missing MCP dependencies to container image.

**v1.11.4**

- Upgrade Starlette dependency to fix CVE-2026-48710.

---

### v1.10.0 – OpenAI Responses API

This release adds support for the OpenAI [`/v1/responses`](api_openai_responses.md) endpoint—OpenAI's next-generation API designed for building agents and multi-step AI workflows. Drop-in compatible with the OpenAI SDK, it works with all Amazon Bedrock Converse-compatible models and supports streaming, function tools, built-in tools (web search, code interpreter, image generation), extended reasoning, and structured output.

#### :material-chat: Responses (OpenAI-Compatible)

| Provider                                                                       | Endpoint/Feature                                                    | AWS Backend                                                                                                            |
|--------------------------------------------------------------------------------|---------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/responses`                                                     | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/responses` – `web_search` / `web_search_preview` built-in tool | ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} Amazon Nova models                       |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/responses` – `code_interpreter` built-in tool                  | ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} Amazon Nova models                       |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/responses` – `image_generation` built-in tool                  | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models      |

#### Fixes

- Fix prompt caching error when messages contain tool-related content on models that do not support tool caching
- Make `signature` field optional in Anthropic message types
- Fix model legacy detection when the end-of-life date falls before the next cache refresh

---

### v1.9.0 – Files API & Images API JSON Body

This release introduces a Files API backed by Amazon S3, available through both the OpenAI-compatible and Anthropic-compatible interfaces. Files uploaded via either API share the same S3 storage and can be referenced across both interfaces. Large files can be uploaded incrementally using the OpenAI multipart uploads API. Stored files can be referenced by ID directly in image edit and variation requests (JSON body), as well as in chat completion messages as document or image inputs. The image edits endpoint now also accepts an `application/json` body as an alternative to multipart form-data, making it easier to chain pipeline steps without re-uploading files.

!!! warning "New Required Configuration"
    Files API requires `AWS_S3_BUCKET` to be configured (shared with the image URL response feature). The S3 prefix for stored files defaults to `files/` and is configurable via `AWS_S3_FILES_PREFIX`. Ensure your IAM role includes read, write, delete, and list permissions on the files prefix in addition to the existing S3 permissions for presigned URLs.

#### :material-folder: Files & Storage

| Provider                                                                                | Endpoint/Feature                      | AWS Backend                                                                         |
|-----------------------------------------------------------------------------------------|---------------------------------------|-------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**          | `/v1/files` – CRUD operations         | ![Amazon S3](styles/logo_amazon_s3.svg){: style="height:20px;width:20px"} Amazon S3 |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**          | `/v1/uploads` – multipart uploads     | ![Amazon S3](styles/logo_amazon_s3.svg){: style="height:20px;width:20px"} Amazon S3 |
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic** | `/v1/files` – CRUD operations         | ![Amazon S3](styles/logo_amazon_s3.svg){: style="height:20px;width:20px"} Amazon S3 |

#### :material-image: Image Generation

| Provider                                                                       | Endpoint/Feature                                                                      | AWS Backend                                                                                                       |
|--------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/images/edits` – JSON body with `images`/`mask` referencing Files API IDs or URLs | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/images/variations` – JSON body with `image` referencing a Files API ID or URL    | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |

#### :material-chat: Chat Completions & Messages

| Provider                                                                                | Endpoint/Feature                                                               | AWS Backend                                                                                                            |
|-----------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**          | Files API file IDs usable as document/image inputs in chat completions         | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic** | Files API file IDs usable as document/image inputs in messages                 | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |

#### Fixes

- Document inputs via S3 URLs are not supported as Bedrock Converse API inputs for some models (e.g., Claude) — now properly detected and handled

---

### v1.8.0 – Broader Model Compatibility & Structured Output

This release focuses on improving reliability and compatibility across a wide variety of models. Structured response formats (JSON object and JSON schema) are now supported on OpenAI chat completions, and request metadata can be forwarded to Bedrock. Tool handling has been significantly improved—both for model-specific system tools and for Amazon Nova's grounding tool, including multi-turn support. Region routing is now more robust, correctly enforcing non-global inference profiles for region-restricted models and handling edge cases gracefully.

!!! warning "New Required IAM Permissions"
    v1.8.0 requires two new IAM permissions to attach request metadata tags to jobs:

    - **`bedrock:TagResource`** on `arn:aws:bedrock:*:*:async-invoke/*` — needed for Bedrock asynchronous invocation jobs (see [IAM Permissions](operations_configuration.md#bedrock-iam)). The `twelvelabs.marengo-embed-3-0-v1:0` and `twelvelabs.marengo-embed-2-7-v1:0` models rely on asynchronous invocation and will fail with an access denied error if this permission is missing.
    - **`transcribe:TagResource`** on `arn:aws:transcribe:*:*:transcription-job/*` — needed for Amazon Transcribe transcription jobs (see [IAM Permissions](operations_configuration.md#speech-to-text-optional)). The `amazon.transcribe` model will fail with an access denied error if this permission is missing.

    Ensure your IAM role or user policy includes both statements before upgrading to v1.8.0.

#### :material-chat: Chat Completions

| Provider                                                                                      | Endpoint/Feature                                                  | AWS Backend                                                                                                            |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**                | `response_format` – JSON object and JSON schema structured output | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**                | `metadata` – request metadata forwarding to Bedrock               | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} **Amazon Nova** | Nova Code Interpreter global profile support                      | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Nova models       |

#### :material-message: Messages (Anthropic-Compatible)

| Provider                                                                                      | Endpoint/Feature                                                              | AWS Backend                                                                                                         |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} **Amazon Nova** | `nova_grounding` responses mapped to `web_search` content blocks              | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Nova models    |
| ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} **Amazon Nova** | Multi-turn conversation support with `nova_grounding`                         | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Nova models    |

#### Platform Features

| Feature                                          | Description                                                                                                                                                                                              |
|--------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Non-global profiles for region-restricted models | Region-restricted models are now always assigned non-global inference profiles, preventing requests from bypassing configured region restrictions                                                        |
| Region routing edge case handling                | Region routing gracefully handles cases where no usable regions are available                                                                                                                            |
| ECS-based server ID                              | When running on ECS, `server_id` in logs is set to `task_id.container_name` for precise instance identification across tasks and containers                                                              |
| Request metadata tagging                         | stdapi.ai request context (`request_id`, `server_id`, `user_id`) is automatically attached as tags to every Bedrock and Amazon Transcribe job, making it easy to trace API calls across AWS service logs |

#### Fixes

- Fix `systemTool_` prefix handling: removed broken auto-promotion logic; system tools require specific tool output handling not compatible with generic tool forwarding
- `AWS_BEDROCK_LEGACY` default changed from `true` to `false` to prevent access denied errors on legacy models that have not been actively used recently
- Bedrock read timeouts are now handled as standard model errors (503) instead of unhandled exceptions, and are properly retried across regions when multi-region routing is enabled

---

### v1.7.0 – Automatic Region Routing, Deprecated Model Fallback & Resilience Improvements

The headline feature of v1.7 is **automatic multi-region routing**: stdapi.ai now intelligently distributes requests across your configured AWS regions, failing over automatically on quota limits or unavailability—and because each region carries its own independent quota, adding regions directly multiplies your effective tokens-per-minute and daily limits. Alongside this, deprecated model IDs are transparently redirected to their replacements so clients survive AWS model retirements without any code changes. This release also adds S3 URL support for file inputs across all relevant endpoints, a configurable AI response timeout, and memory efficiency improvements.

#### Platform Features

| Feature                                               | Description                                                                                                                                                                                                            |
|-------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Automatic region routing with configurable strategies | Intelligently distributes Bedrock requests across configured AWS regions with automatic failover on quota limits or unavailability; supports `ordered`, `lowest_latency`, and `round_robin` strategies                 |
| Deprecated model fallback                             | Transparently reroute deprecated model IDs to their replacements; extend or override the built-in mapping; warns on legacy model usage                                                                                 |
| AI response timeout                                   | Configurable timeout for AI model responses to prevent indefinitely hanging requests                                                                                                                                   |
| Expanded file input support                           | File inputs (images, documents, audio) now support S3 URLs in addition to HTTP URLs, data URIs, and plain base64 across all relevant endpoints; improves memory efficiency by releasing file data as early as possible |
| Model lifecycle timestamps                            | Model created/updated timestamps now derived from lifecycle data (`startOfLifeTime`, `endOfLifeTime`)                                                                                                                  |

#### Fixes

- Fix SSE stream error handling in monitoring to handle specific API and AWS client errors gracefully
- Fix audio MIME type detection failure when `libmagic`'s in-memory buffer path silently returns `application/octet-stream`; fall back to file-based detection to ensure correct format is sent to Bedrock

---

### v1.6.0 – Anthropic API Compatibility & Advanced Claude Capabilities

Introduces a full Anthropic-compatible API layer, enabling direct use of the Anthropic SDK and Claude-native tools with Amazon Bedrock. Adds Claude server tools support via OpenAI chat completions, token count estimation, automatic Anthropic beta flag filtering, and configurable route prefixes.

#### :material-chat: Chat Completions

| Provider                                                                       | Endpoint/Feature                                                                                                                     | AWS Backend                                                                                                   |
|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` Claude server tools (`bash`, `str_replace_based_edit_tool`, `computer`, `memory`) | ![Claude](styles/logo_anthropic_claude.svg){: style="height:20px;width:20px"} Claude models on Amazon Bedrock |

#### :material-message: Messages (Anthropic-Compatible)

| Provider                                                                                      | Endpoint/Feature                                          | AWS Backend                                                                                                          |
|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic**       | `/v1/messages` – Full Anthropic Messages API              | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API    |
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic**       | `/v1/messages/count_tokens` – Token counting              | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - CountTokens API |
| ![Claude](styles/logo_anthropic_claude.svg){: style="height:20px;width:20px"} **Claude**      | Claude server tools (bash, text editor, computer, memory) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Claude models   |
| ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} **Amazon Nova** | Web search tool (`web_search` → `nova_grounding`)         | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Nova models     |

#### :material-format-list-bulleted: Model Discovery (Anthropic-Compatible)

| Provider                                                                                | Endpoint/Feature                              | AWS Backend                                                                                                        |
|-----------------------------------------------------------------------------------------|-----------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic** | `/v1/models` – List models (Anthropic format) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - model catalog |
| ![Anthropic](styles/logo_anthropic.svg){: style="height:20px;width:20px"} **Anthropic** | `/v1/models/{model_id}` – Get model details   | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - model catalog |

#### Platform Features

| Feature                                                 | Description                                                                                                                                        |
|---------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|
| `ANTHROPIC_ROUTES_PREFIX` configuration                 | Configurable base path prefix for Anthropic-compatible routes (default: `/anthropic`)                                                              |
| `OPENAI_ROUTES_PREFIX` configuration                    | Configurable base path prefix for OpenAI-compatible routes                                                                                         |
| Real usage tracking (`usage` in logs)                   | Token counts sourced directly from AWS billing data (replaces tiktoken-based estimation)                                                           |
| Anthropic beta flag filtering (`ANTHROPIC_BETA_FILTER`) | Automatically filter unsupported `anthropic-beta` flags to prevent Bedrock `ValidationException` errors; extensible via `ANTHROPIC_BETA_ALLOWLIST` |
| Claude model name aliases                               | Use official Anthropic model names (e.g., `claude-opus-4-8`) auto-resolved to Amazon Bedrock identifiers                                              |

---

### v1.5.0 – Advanced Reasoning & Model Compatibility (with v1.5.1–v1.5.2 maintenance updates)

Introduces advanced reasoning capabilities with Amazon Nova 2 and Anthropic Claude 4.6+ adaptive reasoning, enhanced system prompt handling for broader model compatibility.

#### :material-chat: Chat Completions

| Provider                                                                                      | Endpoint/Feature                              | AWS Backend                                                                                                            |
|-----------------------------------------------------------------------------------------------|-----------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**                | System prompt handling for unsupported models | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} **Amazon Nova** | Nova 2 chat model reasoning implementation    | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Claude](styles/logo_anthropic_claude.svg){: style="height:20px;width:20px"} **Claude**      | Claude 4.6+ adaptive reasoning configuration  | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Claude models     |

#### Fixes & Maintenance (v1.5.1–v1.5.2)

**v1.5.2**

- Add "/" route to avoid 404 errors on root endpoint
- Fix empty system content block handling (improves Amazon Bedrock Converse API compatibility)

**v1.5.1**

- Fix Amazon Nova Canvas image editing to fall back to TEXT_IMAGE task type when no mask is provided

---

### v1.4.0 – Audio Enhancements & Model Compatibility

Expands audio capabilities with Mistral Voxtral support, speaker diarization, audio formats for chat completions, and introduces prompt caching TTL and model aliasing for better OpenAI compatibility.

#### :material-chat: Chat Completions

| Provider                                                                       | Endpoint/Feature                                                     | AWS Backend                                                                                                            |
|--------------------------------------------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` audio format support                          | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` extended Bedrock finish reasons mapping       | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock                     |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | Prompt caching TTL support                                           | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - prompt caching    |

#### :material-microphone: Speech & Audio

| Provider                                                                            | Endpoint/Feature                                  | AWS Backend                                                                                                            |
|-------------------------------------------------------------------------------------|---------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**      | `/v1/audio/transcriptions` `diarized_json` format | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe            |
| ![Mistral](styles/logo_mistralai.svg){: style="height:20px;width:20px"} **Mistral** | Voxtral audio model                               | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |

#### Platform Features

| Feature                          | Description                                                  |
|----------------------------------|--------------------------------------------------------------|
| Model alias support              | Seamless OpenAI compatibility via model name aliasing        |

#### Fixes

- Fix chat completion file input handling and refactor base64 decoding and MIME handling for file processing.
- Re-raise startup exceptions and disable botocore logging to improve error visibility

---

### v1.3.0 – Image Editing & Variation Support (with v1.3.1–v1.3.5 maintenance updates)

Adds support for OpenAI's image editing and variation endpoints, enabling image manipulation capabilities backed by Amazon Bedrock. Includes maintenance updates for content block handling, tool call validation, streaming fixes, and TTS optimization.

#### :material-image: Image Generation

| Provider                                                                       | Endpoint/Feature        | AWS Backend                                                                                                       |
|--------------------------------------------------------------------------------|-------------------------|-------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/images/edits`      | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/images/variations` | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |

#### :material-microphone: Speech & Audio (v1.3.2)

| Feature                        | Description                                                   |
|--------------------------------|---------------------------------------------------------------|
| `DEFAULT_TTS_LANGUAGE` setting | Configurable default language for TTS to optimize performance |

#### Fixes & Maintenance (v1.3.1–v1.3.5)

**v1.3.5**

- Refactor content block handling to skip empty entries in assistant responses

**v1.3.4**

- Handle invalid tool call arguments with robust JSON content validation
- Add deprecation mapping for `amazon.titan-image-generator-v2:0` → `amazon.nova-canvas-v1:0`

**v1.3.3**

- Remove premature stop condition for `contentBlockStop` in streaming chat completions

**v1.3.2**

- Support `image[]` array-style notation for OpenAI image edits
- Handle empty audio segments in transcription duration calculation

**v1.3.1**

- Improve JSON parsing for tool arguments and results
- Correct `example` → `examples` in OpenAPI model path parameter

---

### v1.2.0 – Service Tiers, System Tools & Performance Enhancements

Introduces service tiers and latency headers for all Bedrock routes, Bedrock-specific system tools (Nova grounding), GPT5.2 API compatibility, configurable guardrail overrides, and Python 3.14 optimization.

#### :material-chat: Chat Completions

| Provider                                                                       | Endpoint/Feature                                                      | AWS Backend                                                                                                        |
|--------------------------------------------------------------------------------|-----------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` `service_tier` parameter                       | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - service tiers |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` Bedrock-specific system tools (Nova grounding) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - system tools  |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` GPT5.2 API update (`reasoning_effort=xhigh`)   |                                                                                                                    |

#### :material-shield-check: Content Safety & Moderation

| Feature                                         | AWS Backend                                                                                                   |
|-------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| Configuration flag for guardrail override allow | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock Guardrails |

#### Platform Features

| Feature                                                | AWS Backend / Description                                                                                          |
|--------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| Service tiers and latency headers (all Bedrock routes) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - service tiers |
| Python 3.14 support                                    | Upgraded to Python 3.14 with performance optimization                                                              |
| Dependency update                                      | Direct aiobotocore usage (replaced aioboto3)                                                                       |

#### Fixes

- Fix warnings for duplicated FastAPI routes (`/docs` and `/openapi.json`).

---

### v1.1.0 – Embeddings Enhancement, Prompt Caching & Advanced Routing

Expands multimodal embedding capabilities, adds prompt caching support, and introduces advanced routing with application inference profiles and prompt routers.

#### :material-chat: Chat Completions

| Provider                                                                       | Endpoint/Feature                                                    | AWS Backend                                                                                                         |
|--------------------------------------------------------------------------------|---------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | Prompt caching `/v1/chat/completions` `prompt_cache_key`            | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - prompt caching |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/chat/completions` GPT5.1 API update  (`reasoning_effort=none`) |                                                                                                                     |

#### :material-vector-polyline: Embeddings

| Provider                                                                                      | Endpoint/Feature                          | AWS Backend                                                                                        |
|-----------------------------------------------------------------------------------------------|-------------------------------------------|----------------------------------------------------------------------------------------------------|
|                                                                                               | Intelligent S3 multimodal upload          | ![Amazon S3](styles/logo_amazon_s3.svg){: style="height:20px;width:20px"} Amazon S3                |
|                                                                                               | Intelligent Sync/async Bedrock invocation | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock |
| ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} **Amazon Nova** | Multimodal embeddings models              |                                                                                                    |
| ![Twelve Labs](styles/logo_twelvelabs.svg){: style="height:20px;width:20px"} **Twelve Labs**  | Marengo V3 models                         |                                                                                                    |

#### :material-directions-fork: Advanced Routing

| Feature                            | AWS Backend                                                                                                                         |
|------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Application inference profiles     | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - application inference profiles |
| Prompt routers                     | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - prompt routers                 |
| Server-side ARN mapping            | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock                                  |
| Client-side ARN passing (optional) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock                                  |

#### Fixes

- `/v1/chat/completions`: Fix default value passed to the converse API for tools without parameters.
- [stdapi-ai Terraform module](https://github.com/stdapi-ai/terraform-aws-stdapi-ai): Fix error if alarms_enabled = true but sns_topic_arn undefined.

---

### v1.0.0 – Foundation Release

The initial release establishes core OpenAI API compatibility with Amazon Bedrock backing.

#### :material-chat: Chat Completions

| Provider                                                                             | Endpoint/Feature                                   | AWS Backend                                                                                                            |
|--------------------------------------------------------------------------------------|----------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**       | `/v1/chat/completions`                             | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
|                                                                                      | All models supporting Converse/ConverseStream APIs | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - Converse API      |
| ![Deepseek](styles/logo_deepSeek.svg){: style="height:20px;width:20px"} **Deepseek** | `/v1/chat/completions` `reasoning_content`         | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Qwen](styles/logo_qwen.svg){: style="height:20px;width:20px"} **Qwen**             | `enable_thinking` + `thinking_budget` parameter    | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |
| ![Qwen](styles/logo_qwen.svg){: style="height:20px;width:20px"} **Qwen**             | `top_k` parameter                                  | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - foundation models |

#### :material-vector-polyline: Embeddings

| Provider                                                                                     | Endpoint/Feature      | AWS Backend                                                                                                           |
|----------------------------------------------------------------------------------------------|-----------------------|-----------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**               | `/v1/embeddings`      | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - embedding models |
| ![Cohere](styles/logo_cohere.svg){: style="height:20px;width:20px"} **Cohere**               | Embed V3 & V4  models |                                                                                                                       |
| ![Twelve Labs](styles/logo_twelvelabs.svg){: style="height:20px;width:20px"} **Twelve Labs** | Marengo V2  models    |                                                                                                                       |
| ![Amazon](styles/logo_amazon.svg){: style="height:20px;width:20px"} **Amazon Titan**         | Embed V1 & V2  models |                                                                                                                       |

#### :material-microphone: Speech & Audio

| Provider                                                                       | Endpoint/Feature           | AWS Backend                                                                                                                    |
|--------------------------------------------------------------------------------|----------------------------|--------------------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/audio/speech`         | ![Amazon Polly](styles/logo_amazon_polly.svg){: style="height:20px;width:20px"} Amazon Polly + Amazon Comprehend               |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/audio/transcriptions` | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe                    |
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/audio/translations`   | ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){: style="height:20px;width:20px"} Amazon Transcribe + Amazon Translate |

#### :material-image: Image Generation

| Provider                                                                                        | Endpoint/Feature                        | AWS Backend                                                                                                       |
|-------------------------------------------------------------------------------------------------|-----------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI**                  | `/v1/images/generations`                | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - image models |
| ![Amazon Nova](styles/logo_amazon_nova.svg){: style="height:20px;width:20px"} **Amazon Nova**   | Canvas V1 models                        |                                                                                                                   |
| ![Amazon](styles/logo_amazon.svg){: style="height:20px;width:20px"} **Amazon Titan**            | Image Generator V1 & V2  models         |                                                                                                                   |
| ![Stability AI](styles/logo_stabilityai.svg){: style="height:20px;width:20px"} **Stability AI** | Image Core, Ultra et SD3.5 Large models |                                                                                                                   |

#### :material-format-list-bulleted: Model Discovery

| Provider                                                                       | Endpoint/Feature | AWS Backend                                                                                                        |
|--------------------------------------------------------------------------------|------------------|--------------------------------------------------------------------------------------------------------------------|
| ![OpenAI](styles/logo_openai.svg){: style="height:20px;width:20px"} **OpenAI** | `/v1/models`     | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - model catalog |

#### Platform Features

| Feature                                     | AWS Backend                                                                                                                                                                                                                                 |
|---------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Bedrock Features**                        |                                                                                                                                                                                                                                             |
| Content filtering and safety                | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock Guardrails                                                                                                                               |
| Cross-region inference                      | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - global/regional                                                                                                                        |
| Application inference profiles              | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - inference profiles                                                                                                                     |
| Model parameters (temperature, top_p, etc.) | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - native parameters                                                                                                                      |
| Multi-region failover                       | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock - multi-region                                                                                                                           |
| Bedrock guardrails                          | ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){: style="height:20px;width:20px"} Amazon Bedrock Guardrails                                                                                                                               |
| **AWS Services**                            |                                                                                                                                                                                                                                             |
| File storage                                | ![Amazon S3](styles/logo_amazon_s3.svg){: style="height:20px;width:20px"} Amazon S3 - presigned URLs, Transfer Acceleration                                                                                                                 |
| **Authentication**                          |                                                                                                                                                                                                                                             |
| Static token authentication                 | ![AWS Systems Manager](styles/logo_amazon_systems_manager.svg){: style="height:20px;width:20px"} AWS SSM Parameter Store / ![AWS Secrets Manager](styles/logo_amazon_secrets_manager.svg){: style="height:20px;width:20px"} Secrets Manager |
| Development mode (no auth)                  |                                                                                                                                                                                                                                             |
| **Observability**                           |                                                                                                                                                                                                                                             |
| Distributed tracing                         | ![AWS X-Ray](styles/logo_amazon_xray.svg){: style="height:20px;width:20px"} AWS X-Ray + OpenTelemetry                                                                                                                                       |
| Structured logging                          | ![Amazon CloudWatch](styles/logo_amazon_cloudwatch.svg){: style="height:20px;width:20px"} Amazon CloudWatch (When running on ECS/EKS)                                                                                                       |
| Health check endpoint                       |                                                                                                                                                                                                                                             |
| **HTTP/Security**                           |                                                                                                                                                                                                                                             |
| CORS support                                |                                                                                                                                                                                                                                             |
| Trusted host validation                     |                                                                                                                                                                                                                                             |
| Proxy headers (X-Forwarded-*)               |                                                                                                                                                                                                                                             |
| GZip compression                            |                                                                                                                                                                                                                                             |
| **📚 Documentation**                        |                                                                                                                                                                                                                                             |
| Interactive API docs & OpenAPI schema       |                                                                                                                                                                                                                                             |
| **🔌 Compatibility**                        |                                                                                                                                                                                                                                             |
| Provider-specific parameters                |                                                                                                                                                                                                                                             |
