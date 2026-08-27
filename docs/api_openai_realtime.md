---
title: Realtime API - Live Speech-to-Speech over WebSocket (OpenAI Compatible)
description: Build low-latency voice agents on Amazon Bedrock with an OpenAI-compatible Realtime API — bidirectional WebSocket audio, browser-safe ephemeral client secrets, and G.711 telephony formats.
keywords: realtime API, speech to speech API, voice agent API AWS, WebSocket audio API, OpenAI Realtime API compatible, low latency voice AI, ephemeral client secret, G.711 telephony API, AWS Bedrock voice API
---

# Realtime API

Hold a live, bidirectional speech-to-speech conversation over a single WebSocket, through the OpenAI Realtime API shape. Audio flows in both directions on the same connection: send the caller's speech as it is captured, and receive the model's spoken answer as it is generated — no request/response round trip per turn.

## Why Choose the Realtime API?

<div class="grid cards" markdown>

- :material-swap-horizontal: __Drop-in OpenAI Compatibility__
  <br>The same client event and server event vocabulary as the OpenAI Realtime API. `client.realtime.connect(model=...)` works by changing the base URL.

- :material-incognito: __Browser-Safe Ephemeral Secrets__
  <br>Mint a short-lived credential server-side and hand it to an untrusted browser or mobile client — your API key never leaves your backend.

- :material-server-off: __Stateless at Any Scale__
  <br>An ephemeral secret is a signed token, not a server-side record: any instance behind a load balancer verifies one minted by any other, with no shared session store.

- :material-phone-in-talk: __Telephony-Ready Audio__
  <br>24 kHz PCM by default, or G.711 (`audio/pcmu`, `audio/pcma`) at 8 kHz for direct interoperability with telephony and SIP media.

</div>

## Available Endpoints

| Endpoint                     | Method | What It Does                                                    | Powered By      | MCP Tool                     |
|-------------------------------|--------|-------------------------------------------------------------------|------------------|-------------------------------|
| `/v1/realtime/client_secrets` | `POST` | Mint a short-lived, signed client secret carrying a session configuration | Amazon Bedrock  | `openai_realtime_client_secret` |
| `/v1/realtime?model=<id>`     | `WS`   | Open a live, bidirectional speech-to-speech session                | Amazon Bedrock  | Not applicable — a persistent connection |
| `/v1/realtime/calls`          | `POST` | Trade a WebRTC SDP offer for an SDP answer, opening a call the gateway terminates — [opt-in](#webrtc-calls) | Amazon Bedrock  | Not applicable — negotiates a media connection |
| `/v1/realtime/calls/{call_id}/hangup` | `POST` | End an active WebRTC call                                  | Amazon Bedrock  | Not applicable — controls a media connection |

!!! info "WebSocket always — WebRTC opt-in, SIP never"
    Every deployment serves the WebSocket. `POST /v1/realtime/calls` answers WebRTC offers only when the operator enables [`REALTIME_WEBRTC_ENABLED`](operations_configuration.md#realtime-webrtc-enabled) — it needs a UDP media path the default deployment does not have — and answers `404` otherwise. Inbound SIP is never terminated here: the SIP-only verbs (`accept`, `reject`, `refer`) answer a clean `400` naming what serves telephony instead.

    [Transports](#transports) covers the whole picture: how a browser connects, what enabling WebRTC entails, and how to put a media framework or a phone line in front of this deployment.

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                                       |                  Status                  | Notes                                                              |
|------------------------------------------------|:-----------------------------------------:|---------------------------------------------------------------------|
| **Client Events**                             |                                          |                                                                     |
| `session.update`                              |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Voice, instructions and audio formats are fixed once the conversation opens — see [below](#voice-instructions-and-audio-formats-are-fixed-once-the-conversation-opens) |
| `input_audio_buffer.append`                   |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Base64-encoded audio in the session's configured input format; at most 4 MiB per event, so send it in chunks as it is captured |
| `input_audio_buffer.commit`                   |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Required to end a turn when `turn_detection` is `null`; at most 5.7 MB of audio may wait for one |
| `input_audio_buffer.clear`                    |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Discards buffered, not-yet-committed audio                          |
| `conversation.item.create`                    |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Text items only — an audio item is refused with a clear `error`; send speech through `input_audio_buffer.append` |
| `conversation.item.truncate`                   |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Answered with `conversation.item.truncated` — see [below](#truncating-an-answer-the-caller-spoke-over) |
| `conversation.item.retrieve`                   |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Answered with `conversation.item.retrieved`, carrying the item's role, status and transcript; audio is not retained, so the item carries no `audio` field |
| `conversation.item.delete`                     |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Answered with `conversation.item.deleted`; the item stops being addressable, and the model keeps its own memory of the conversation |
| `response.create`                             |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Ends any open turn and starts the model answering; its per-response `response` payload is ignored, and the session's own configuration serves every answer |
| `response.cancel`                              |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Ends the answer in progress with `status: "cancelled"`; what the model keeps speaking is dropped rather than reported |
| `output_audio_buffer.clear`                    |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Acknowledged with `output_audio_buffer.cleared`                     |
| **Server Events**                             |                                          |                                                                     |
| `session.created` / `session.updated`          |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Sent on connect and after every accepted `session.update`           |
| `input_audio_buffer.speech_started` / `.speech_stopped` |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Server-side voice activity detection only                           |
| `input_audio_buffer.committed` / `.cleared`    |   :material-check-circle:{ .success role="img" aria-label="Supported" }    |                                                                     |
| `conversation.item.added` / `.done`            |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Sent for every item: a written one, the caller's committed audio, and each answer — the answer's `.done` precedes its `response.done` |
| `conversation.item.created`                    |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Sent beside `conversation.item.added` for a written item, for clients predating the added/done pair |
| `conversation.item.truncated` / `.retrieved` / `.deleted` |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Answers to the matching client event                                 |
| `conversation.item.input_audio_transcription.delta` / `.completed` |       :material-cog:{ .model-dep role="img" aria-label="Conditional" }       | Only when `audio.input.transcription` is set on the session          |
| `conversation.item.input_audio_transcription.failed` |       :material-cog:{ .model-dep role="img" aria-label="Conditional" }       | Sent instead of a transcript when a caller turn could not be read, so it is not mistaken for a caller who said nothing; only when `audio.input.transcription` is set |
| `conversation.item.input_audio_transcription.segment` | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Not emitted — a transcript carries no per-speaker segments or timings |
| `response.created` / `response.done`           |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Both carry the whole response object — see [below](#what-a-response-object-reports); `response.done` adds the answer's token usage |
| `response.output_item.added` / `.done`         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    |                                                                     |
| `response.content_part.added` / `.done`        |   :material-check-circle:{ .success role="img" aria-label="Supported" }    |                                                                     |
| `response.output_audio.delta` / `.done`        |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Spoken answers only                                                  |
| `response.output_audio_transcript.delta` / `.done` |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Spoken answers only                                                  |
| `response.output_text.delta` / `.done`         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Text-only answers (`output_modalities: ["text"]`)                    |
| `output_audio_buffer.cleared`                  |   :material-check-circle:{ .success role="img" aria-label="Supported" }    |                                                                     |
| `error`                                        |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Non-fatal for a rejected event; terminal (closes the socket) for a fatal one |
| `rate_limits.updated`                          | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Not emitted                                                          |
| `input_audio_buffer.timeout_triggered`         | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Not emitted — it reports an `idle_timeout_ms` that is not available   |
| `output_audio_buffer.started` / `.stopped`     | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Not emitted, on the WebRTC [transport](#webrtc-calls) either |
| **Audio Formats**                             |                                          |                                                                     |
| `audio/pcm`                                    |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Default — 24 kHz, 16-bit, mono, little-endian                       |
| `audio/pcmu`, `audio/pcma`                     |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | G.711 at 8 kHz, for telephony interoperability                      |
| Independent input/output formats               |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Configured separately under `audio.input.format` / `audio.output.format` |
| **Turn Detection**                            |                                          |                                                                     |
| Server-side voice activity detection           |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Default; ends each turn automatically                                |
| Manual turns (`turn_detection: null`)          |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | End each turn yourself with `input_audio_buffer.commit`              |
| `threshold`, `prefix_padding_ms`, `silence_duration_ms`, `idle_timeout_ms`, `eagerness` | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Accepted and ignored — detection sensitivity is not tunable          |
| `create_response`, `interrupt_response`        | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Accepted and ignored — a detected turn always starts a response, and interruption is the model's own decision |
| Barge-in (caller speaks over the answer)       |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Handled by the model itself                                          |
| **Voices**                                    |                                          |                                                                     |
| OpenAI voice names                             |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | `alloy`, `ash`, `ballad`, `cedar`, `coral`, `echo`, `marin`, `sage`, `shimmer`, `verse` — each served by the model's own nearest voice, so the timbre is not the upstream one |
| Any other voice name                           | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Passed through to the model as given, so a model voice can be named directly |
| **Not Available**                             |                                          |                                                                     |
| `POST /v1/realtime/calls` (WebRTC)             | :material-minus-circle:{ .partial role="img" aria-label="Partial" } | [Opt-in](#webrtc-calls): served when `REALTIME_WEBRTC_ENABLED` is set and the deployment has a UDP media path; answers `404` otherwise |
| SIP (`accept`, `reject`, `refer` call verbs)   | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Inbound SIP is not terminated by the gateway, permanently — the verbs answer `400`; see [Transports](#transports) for the telephony route |
| `tools`, `tool_choice`, `parallel_tool_calls`  | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Accepted and ignored — the session calls no tools                    |
| `prompt` (prompt templates)                    | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Accepted and ignored                                                 |
| `reasoning`, `tracing`, `truncation`           | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Accepted and ignored                                                 |
| `include`                                      | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Accepted and ignored — no extra output fields are available          |
| `audio.input.noise_reduction`                  | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Accepted and ignored — incoming audio is not filtered                |
| `audio.input.transcription.model` / `.language` / `.prompt` | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Accepted and ignored — the transcript comes from the session's own model, which detects the language and takes no vocabulary hint. Setting the `transcription` object at all is what turns the events on |
| `audio.output.speed`                           | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } | Accepted and ignored — the spoken answer is not time-scaled          |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep role="img" aria-label="Conditional" } **Conditional** — Depends on session configuration
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } **Extra Feature** — Enhanced capability beyond OpenAI API

</div>

### Voice, instructions and audio formats are fixed once the conversation opens

The model's voice, its system `instructions`, and both audio formats are set when the conversation with the model opens, and cannot change for the rest of that session. The conversation opens on the **first thing sent into it** — the first `input_audio_buffer.append` under the default server voice activity detection, or the first `input_audio_buffer.commit`, `response.create` or `conversation.item.create` under manual turns — which is well before the model has answered anything.

Send `session.update` with these settings **before sending anything else**, or open a new session to change them. Afterwards, a `session.update` touching only other fields (`turn_detection`, `max_output_tokens`, transcription settings, and so on) is still accepted; one that would change voice, instructions or an audio format is refused with an `error` event.

### Answering a written turn

`conversation.item.create` carrying an `input_text` part adds the text to the conversation without starting an answer. Follow it with `response.create`, and the model answers it exactly as it answers a spoken one — the same `response.output_audio.delta` chunks and the same transcript — so a written nudge into a voice session ("the caller has been on hold", "wrap up now") needs no second channel.

### What a response object reports

`response.created` and `response.done` carry the same response object, and every field the upstream API sends is present on both — a voice framework validates each frame against its own models, and a missing field is the event never arriving rather than a cosmetic difference.

| Field | What it carries |
|---|---|
| `status_details` | `null` while the answer is in progress and once it has completed. An answer the caller spoke over reports `{"type": "incomplete", "reason": "turn_detected"}`, and one ended by `response.cancel` reports `{"type": "cancelled", "reason": "client_cancelled"}` |
| `conversation_id` | The conversation the answer was added to — one per session, so every answer of a session names the same one |
| `output_modalities` | `["audio"]`, or `["text"]` when the session asked for text-only answers |
| `max_output_tokens` | The session's `max_output_tokens`, or `"inf"` when it sets none |
| `audio` | The session's effective output `format` and `voice`; `voice` is `null` when the session named none and the model answered in its own |
| `metadata` | Always `null` — an answer carries no metadata, since `response.create` takes no per-response configuration |
| `output`, `usage` | The answer's conversation item, and the tokens it used (on `response.done`) |

### Truncating an answer the caller spoke over

The model generates speech faster than it is played, so a caller who interrupts has heard less of the answer than was sent. Send `conversation.item.truncate` with the item's `id`, `content_index: 0` and the `audio_end_ms` your player actually reached; the session cuts its record of that item to what was heard and answers `conversation.item.truncated`.

- The item's **transcript is removed whole**, not trimmed: nothing aligns a transcript to a position in the audio, and leaving text the caller never heard in the record is the failure this event exists to prevent.
- `audio_end_ms` past the end of the item's audio, an item that is not an assistant message, and an item this session never sent are each refused with an `error`.
- What the model itself remembers of the answer is the model's own; truncation aligns the record this session reports through `conversation.item.retrieved`.

### Guardrail coverage

When the deployment configures an [Amazon Bedrock guardrail](operations_configuration.md#bedrock-guardrails), a realtime session is checked **per turn**: what the caller said (as the model transcribes it) as `INPUT`, and each completed answer as `OUTPUT`. A blocked turn ends the session with a terminal `error` event and close code `3000`.

Unlike a request/response route, the check cannot come before the content reaches the client: the model's speech is streamed while it is being generated and its transcript is only complete once the answer is over, so a blocked answer may already have been partly heard when the session ends. Written items sent with `conversation.item.create` are checked as `INPUT` **before** they reach the model, as on every other route.

## Model Support

Every deployment's catalog differs, and a model's own name is never guaranteed stable across accounts. Find which models serve this route:

```bash
curl "$BASE/search_models?route=openai_realtime" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

Pass the returned model ID as `model` on the WebSocket URL, or in the `session.model` field of an ephemeral secret's configuration. See the [Search Models API](api_search_models.md) for the full filter syntax.

!!! note "`session.model` does not accept a wildcard pattern; the WebSocket's `model` does"
    `POST /v1/realtime/client_secrets` fixes the model into the signed token before a connection exists, so `session.model` must name an exact model — a [wildcard pattern](operations_configuration.md#model-wildcard-patterns) is rejected. The `model` query parameter of `WS /v1/realtime` itself has no such constraint and accepts a pattern.

## Authentication

Open the WebSocket with one of three credentials, carried in whichever channel the client can use:

| Client                                    | Credential carrier                                                   |
|--------------------------------------------|------------------------------------------------------------------------|
| Server-side SDKs                           | `Authorization: Bearer <api key or ephemeral secret>` header           |
| Other gateway clients                      | `x-api-key: <api key or ephemeral secret>` header                      |
| Browser (cannot set custom WebSocket headers) | `Sec-WebSocket-Protocol` list entry `openai-insecure-api-key.<credential>` |

Either the deployment's own API key or an [ephemeral client secret](#ephemeral-client-secrets) (`ek_...`) works as the credential. The `model` query parameter (`/v1/realtime?model=<model id>`) selects the model serving the session; it may be omitted when the credential is an ephemeral secret whose session configuration already names one. See [Authentication & Security](operations_authentication_security.md) for how the API key itself is configured.

!!! warning "A refused credential is not an HTTP status"
    The WebSocket upgrade always completes first, so a rejected or expired credential is **not** answered with `401`/`403`. The connection opens, the first and only event is a terminal `error` with `code: "invalid_api_key"`, and the socket is then closed with close code `3000` and reason `invalid_request_error.invalid_api_key` — the same shape the upstream API uses. Instrument the `error` event and the close code, not the handshake status.

## Ephemeral Client Secrets

`POST /v1/realtime/client_secrets` mints a short-lived credential — a value starting with `ek_` — that carries a session configuration. Hand it to a browser or mobile client so it can open a session directly, without ever holding the deployment's own API key.

```bash
curl -X POST "$BASE/v1/realtime/client_secrets" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "expires_after": {"anchor": "created_at", "seconds": 60},
    "session": {
      "type": "realtime",
      "model": "amazon.nova-2-sonic-v1:0",
      "instructions": "You are a helpful support agent."
    }
  }'
```

```json
{
  "value": "ek_...",
  "expires_at": 1731000060,
  "session": {
    "type": "realtime",
    "model": "amazon.nova-2-sonic-v1:0",
    "instructions": "You are a helpful support agent.",
    "audio": { "...": "..." }
  }
}
```

- `expires_after.seconds` accepts **10 to 7,200** seconds, defaulting to **600** (10 minutes) when omitted. This bounds how long the secret can be used to *open* a session — a session already opened with it keeps running for its own [session limit](#session-lifecycle-and-limits).
- `session` accepts the same configuration a client would otherwise send in a `session.update` event; it is applied to every session opened with the secret. It also accepts `{"type": "transcription"}`, which opens sessions that only transcribe the caller and never answer out loud — the only way to ask for one, since the socket itself takes no session type.
- By default the carried configuration is a **default, not a constraint**: the client may name another model on the `?model=` query string and change the configuration with its own `session.update`, as it can upstream. Set [`REALTIME_ALLOW_SESSION_OVERRIDE=false`](operations_configuration.md#realtime-allow-session-override) to make the model, the `instructions` and `max_output_tokens` the secret was minted with final — a mismatching `?model=` is then refused at connect, and a `session.update` changing one of them answers an `error`.

!!! warning "What a secret grants until it expires"
    A secret cannot be revoked: rotating [`REALTIME_CLIENT_SECRET_KEY`](operations_configuration.md#realtime-client-secret-key) invalidates every outstanding one at once, and nothing else does. Until then it may open **any number of concurrent sessions**, each billed to the deployment — so keep `expires_after.seconds` as short as the flow allows.

    Its payload is signed, not encrypted: whoever holds the secret can read the session configuration it carries. Nothing confidential belongs in `instructions`.

!!! info "Stateless, and signed"
    Nothing is stored server-side: the secret is the session configuration plus a signature, so **any instance behind a load balancer verifies a secret minted by any other** — no shared session store, no sticky routing required.

    The signing key is derived from the deployment's configured API key by default. When the deployment runs with **no API key configured at all**, the signing key falls back to a random value generated **per process**: minted secrets then only verify on the instance that minted them, and stop working once a request reaches a different one. Set [`realtime_client_secret_key`](operations_configuration.md#realtime-client-secret-key) explicitly to fix a key shared by every instance regardless of the API key configuration.

## Transports

Upstream offers a realtime session over three transports — WebSocket, WebRTC and SIP. **This API always serves the WebSocket; WebRTC is an operator opt-in; SIP is never terminated here.** `POST /v1/realtime/calls` trades an SDP offer for an answer once [`REALTIME_WEBRTC_ENABLED`](operations_configuration.md#realtime-webrtc-enabled) is set — see [WebRTC calls terminated by the gateway](#webrtc-calls) for what that transport delivers and what it demands of the deployment — and answers `404` otherwise.

### A browser connects to that same WebSocket

There is no separate browser transport to be missing. The page opens `wss://<host>/v1/realtime?model=<id>` itself, carrying an [ephemeral client secret](#ephemeral-client-secrets) in the way [Authentication](#authentication) describes, so the deployment's own API key never leaves your backend. It is two steps — mint the secret server-side, connect with it client-side — and the [browser example](#browser-ephemeral-client-secret) below is both of them.

What the page owns in exchange is the media. Capturing the microphone, resampling it to the session's input format and playing back the `response.output_audio.delta` chunks are its own work, because a WebSocket carries the audio bytes handed to it and nothing else: no jitter buffer, no packet-loss concealment, no echo cancellation. On a good network that is unremarkable; on a lossy one it is audible, and it is the reason to reach for a media stack rather than write one.

### WebRTC calls terminated by the gateway { #webrtc-calls }

With [`REALTIME_WEBRTC_ENABLED`](operations_configuration.md#realtime-webrtc-enabled) set — and the `webrtc` optional dependencies installed, which the container images ship — the gateway terminates the whole WebRTC media path itself: ICE, DTLS-SRTP and Opus. A browser (or any WebRTC client) posts its SDP offer and connects directly, with nothing in between:

- **`POST /v1/realtime/calls`** accepts the offer as a raw `application/sdp` body — with `?model=<id>` on the query string, exactly as upstream's browser flow — or as `multipart/form-data` with an `sdp` field and an optional `session` JSON field. Either encoding authenticates with an [ephemeral client secret](#ephemeral-client-secrets) or the deployment's credentials; a secret whose session is [locked](#ephemeral-client-secrets) refuses a `session` field. A JSON body is refused with `unsupported_content_type`, which is what the upstream endpoint answers too. The response is `201` with the SDP answer as its body and the call's identifier in the `Location` header (`/v1/realtime/calls/rtc_...`).
- **Audio rides the media tracks** as Opus, both directions — no base64, no `response.output_audio.delta` events. **Events ride a data channel** the client opens under the label `oai-events`, in exactly the WebSocket vocabulary: `session.created` arrives on it once the channel opens, `session.update`, `response.create` and the rest work unchanged. The session's audio formats are fixed by the media negotiation, so a `session.update` changing them is refused with an `error`.
- **Playback is the gateway's, so barge-in is too.** The model generates speech faster than it is played, and on a call the unplayed tail sits in the gateway rather than in your player. A caller who starts speaking over it — `input_audio_buffer.speech_started` — stops hearing it immediately, and the drop is reported as `output_audio_buffer.cleared`, as it is upstream. An answer ended by `response.cancel` or interrupted mid-generation stops the same way.
- **`POST /v1/realtime/calls/{call_id}/hangup`** ends the call; the SDP answer's `Location` header is where the identifier comes from.
- **A sideband WebSocket** — `WS /v1/realtime?call_id=<id>`, API credentials only, never an ephemeral secret — observes the call's server events and may send client events into its session, mirroring upstream's monitoring channel.
- **Call control follows the credential that opened the call.** A call opened under a [tenant API key](operations_authentication_security.md) can be ended or observed only with that tenant's key or the deployment's own credentials; any other caller is answered the same `404` as an unknown call.

!!! warning "What a WebRTC call demands of the deployment"
    - **A UDP path to the exact instance that answered.** ICE negotiates ephemeral UDP ports directly to the server process; an HTTP(S) load balancer cannot carry them. The [Terraform module's WebRTC media mode](operations_deploy_advanced.md#webrtc-and-sip-need-their-own-ingress) provisions the public task IP and UDP ingress this needs, off by default. Behind 1:1 NAT, set [`REALTIME_WEBRTC_STUN_SERVER`](operations_configuration.md#realtime-webrtc-stun-server) so the gateway advertises its public address; for callers on UDP-blocking networks, run a TURN relay (for example coturn) and set [`REALTIME_WEBRTC_TURN_SERVER`](operations_configuration.md#realtime-webrtc-turn-server) — AWS offers no managed TURN.
    - **The caller must offer a publicly routable candidate.** An SDP offer names the addresses the gateway sends its ICE checks to, so candidates on addresses that are not globally routable — private, loopback, link-local — are dropped, and an offer left with none is refused with `invalid_offer`. Hostname and mDNS (`.local`) candidates are always dropped. For callers that legitimately share the deployment's network, set [`REALTIME_WEBRTC_ALLOW_PRIVATE_CANDIDATES`](operations_configuration.md#realtime-webrtc-allow-private-candidates).
    - **One instance, or routed call control.** A call lives in the memory of the instance that answered its offer. `hangup` and the sideband WebSocket answer `404` on any other instance; run a single instance, or route call-control requests to the answering instance yourself.
    - **The 8-minute session cap applies to calls too.** Amazon Nova Sonic ends a session at 480 seconds, so a call hard-stops at 8 minutes with the connection torn down — a real limit for the phone-length conversations WebRTC invites.
    - **Scale-in, deployments and Spot interruption end live calls.** The media path cannot drain: an instance that stops mid-call drops it.

### Put WebRTC or a phone line in front of the gateway

The route to a browser peer connection or a phone call does not have to run through `POST /v1/realtime/calls`, and for SIP it never does. The two frameworks most voice agents are already built on — **LiveKit Agents** and **Pipecat** — terminate WebRTC themselves (and SIP, through their telephony transports), and reach the model over exactly the WebSocket this API serves. The caller speaks WebRTC or SIP to the framework; the framework speaks this API. Pointing one at this deployment asks no more of it than the rest of this gateway does: the base URL, the deployment's API key, and the model name only where the name differs.

**LiveKit Agents** takes an HTTP base URL and derives the WebSocket from it, exactly as the official SDK does:

```python
from livekit.agents import AgentSession
from livekit.plugins import openai

session = AgentSession(
    llm=openai.realtime.RealtimeModel(
        model="amazon.nova-2-sonic-v1:0",
        base_url="https://your-deployment.example.com/v1",
        api_key="YOUR_API_KEY",
    )
)
```

`base_url` also reads from the `OPENAI_BASE_URL` environment variable, and `api_key` from `OPENAI_API_KEY`. A base URL ending in `/v1` has `/realtime` appended for you; a deployment served under a non-default [`OPENAI_ROUTES_PREFIX`](operations_configuration.md#openai-routes-prefix) has to name the full path itself.

**Pipecat** takes the WebSocket URL whole, `/v1/realtime` included:

```python
import os

from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService

llm = OpenAIRealtimeLLMService(
    base_url="wss://your-deployment.example.com/v1/realtime",
    api_key=os.environ["OPENAI_API_KEY"],
    settings=OpenAIRealtimeLLMService.Settings(model="amazon.nova-2-sonic-v1:0"),
)
```

On a telephony leg, set the session's audio formats to `audio/pcmu` or `audio/pcma` — G.711 at 8 kHz is what a phone call already carries, so nothing resamples it twice.

!!! note "Check the constructor against the version you install"
    The parameters above are as shipped in `livekit-plugins-openai` 1.6.10 and `pipecat-ai` 1.7.0. Both projects have renamed this integration before — Pipecat's service moved out of `pipecat.services.openai_realtime_beta`, and LiveKit does not list `base_url` on its own parameters page — so read [LiveKit's OpenAI realtime plugin](https://docs.livekit.io/agents/models/realtime/plugins/openai/) and [Pipecat's OpenAI realtime service](https://docs.pipecat.ai/api-reference/server/services/s2s/openai) for the release you pin.

### Why WebRTC is a different kind of endpoint

An HTTP request that returns an SDP answer is the small, visible part of WebRTC. The rest is a media transport in its own right: a UDP path negotiated separately from the HTTPS connection, carrying encrypted RTP, with its own address discovery, NAT traversal and congestion control. Serving it means **terminating that media path**, not routing one more request — and the two are not the same thing to build, nor the same thing to put an ingress in front of. SIP has the same shape under different names: signalling on one connection, audio on another.

That is also why a framework remains the shorter path for anything the [gateway-terminated transport](#webrtc-calls) does not cover — telephony, more than one instance, calls past 8 minutes, hostile networks without your own TURN. LiveKit and Pipecat already own a media stack, already run wherever your users are, and already speak this API on the other side. [WebRTC and SIP need their own ingress](operations_deploy_advanced.md#webrtc-and-sip-need-their-own-ingress) covers what terminating media costs a deployment, whichever process does it.

## Session Lifecycle and Limits

- **Duration cap** — a session lasts at most **8 minutes**. When it is reached, the server closes the connection with WebSocket close code `1000` and reason `session_expired`; reconnect to continue the conversation. A [WebRTC call](#webrtc-calls) is under the same cap: at 8 minutes its session ends and the peer connection is torn down.
- **Conversation ended by the model side** — when the model ends the conversation itself, the connection closes with close code `1000` and reason `session_ended`. Normal, and reconnecting starts a new session.
- **Server shutdown** — a session still open when the deployment shuts down is closed with close code `1001` and reason `server_shutdown`.
- **Fatal errors** — a fatal error sends a terminal `error` event, then closes the connection with close code `3000`, whose reason is `<error type>.<error code>` (e.g. `invalid_request_error.model_not_found`).
- **Event size** — a single client event may carry at most **4 MiB**, base64 included; a larger one answers an `error` and is dropped. Append audio in the small chunks it is captured in rather than whole files.
- **Uncommitted audio** — under manual turns (`turn_detection: null`), at most **5.7 MB** of decoded audio may be buffered before an `input_audio_buffer.commit` (about 2 minutes of 24 kHz PCM, longer for G.711); past that the append answers an `error`. Commit each turn, or clear the buffer with `input_audio_buffer.clear`.
- **Addressable items** — the session keeps its **200** most recent conversation items addressable, dropping the oldest past that. `conversation.item.truncate`, `.retrieve` and `.delete` answer an `error` for an item that has fallen out; the model's own memory of the conversation is unaffected.

## Billing { #cost }

A realtime session bills audio and text tokens continuously, in both directions, for as long as the connection is open — not per request. Usage is reported per answer, in each `response.done` event, and recorded in the gateway's usage log the same way, so a session that drops mid-conversation still accounts for everything spoken before it. Speech tokens are priced well above text tokens by AWS, and the two are recorded and priced separately here. See [Cost Management](operations_cost_management.md) for how usage becomes cost.

## Try It Now

### Python (server-side, with the official SDK)

Requires the `openai` package with realtime support (`pip install "openai[realtime]"`). Point the client's `base_url` at this deployment; `client.realtime.connect(...)` derives the WebSocket URL from it automatically.

```python
import base64

from openai import AsyncOpenAI

client = AsyncOpenAI(
    api_key="YOUR_API_KEY", base_url="https://your-deployment.example.com/v1"
)


async def main() -> None:
    async with client.realtime.connect(model="amazon.nova-2-sonic-v1:0") as connection:
        await connection.session.update(
            session={
                "type": "realtime",
                "instructions": "You are a concise, friendly voice assistant.",
            }
        )

        # Stream 24 kHz, 16-bit, mono, little-endian PCM in the small chunks it
        # is captured in -- ~100 ms each here, well under the per-event limit.
        with open("question.pcm", "rb") as audio_file:
            while chunk := audio_file.read(4800):
                await connection.input_audio_buffer.append(
                    audio=base64.b64encode(chunk).decode()
                )
        await connection.input_audio_buffer.commit()
        await connection.response.create()

        async for event in connection:
            if event.type == "response.output_audio.delta":
                # event.delta is base64-encoded audio in the session's output format.
                ...
            elif event.type == "response.output_audio_transcript.delta":
                print(event.delta, end="", flush=True)
            elif event.type == "response.done":
                break
```

### Browser (ephemeral client secret)

Mint the secret from your backend — never expose the deployment's own API key to the browser — then connect directly from client-side JavaScript using the `openai-insecure-api-key.<secret>` subprotocol, since a browser cannot set custom WebSocket headers:

```javascript
// Fetched from your own backend, which called POST /v1/realtime/client_secrets
const { value: ephemeralSecret } = await fetch("/api/realtime-secret").then((r) => r.json());

const ws = new WebSocket(
  "wss://your-deployment.example.com/v1/realtime?model=amazon.nova-2-sonic-v1:0",
  ["realtime", `openai-insecure-api-key.${ephemeralSecret}`],
);

ws.addEventListener("open", async () => {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

  // Your own capture code: resample the microphone to the session's input
  // format (24 kHz, 16-bit, mono PCM by default) and hand over small chunks.
  captureAudioChunks(stream, (pcmChunk) => {
    ws.send(
      JSON.stringify({
        type: "input_audio_buffer.append",
        audio: btoa(String.fromCharCode(...new Uint8Array(pcmChunk))),
      }),
    );
  });
  // Nothing else is needed: server voice activity detection ends each turn and
  // starts the answer. Under `turn_detection: null`, send
  // `input_audio_buffer.commit` yourself instead.
});

ws.addEventListener("message", (event) => {
  const serverEvent = JSON.parse(event.data);
  if (serverEvent.type === "response.output_audio.delta") {
    // serverEvent.delta is base64-encoded audio in the session's output format.
  }
});
```

---

**Ready to add voice to your application?** Find compatible models with the [Search Models API](api_search_models.md), or explore the full [audio suite](api_openai_audio_speech.md) for turn-based speech and transcription.
