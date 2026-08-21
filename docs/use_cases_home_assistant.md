---
title: Home Assistant Voice Integration - Amazon Bedrock Speech
description: Connect Home Assistant Assist voice pipelines to Amazon Bedrock using stdapi.ai and the wyoming-openai proxy. Speech-to-text and streaming text-to-speech for local voice assistants, backed by Amazon Transcribe and Amazon Polly.
keywords: Home Assistant AI, Home Assistant voice assistant AWS, wyoming protocol, Wyoming OpenAI proxy, Amazon Polly Home Assistant, Amazon Transcribe Home Assistant, local voice assistant AWS, Assist pipeline AWS Bedrock
---

# :material-microphone-message: Home Assistant Voice Integration

Give Home Assistant's Assist voice pipeline speech-to-text and text-to-speech backed by Amazon Transcribe and Amazon Polly, through stdapi.ai's OpenAI-compatible audio routes.

## :material-information-outline: About Home Assistant Assist and Wyoming

**🔗 Links:** [Home Assistant Assist](https://www.home-assistant.io/voice_control/) | [wyoming-openai](https://github.com/roryeckel/wyoming_openai) | [Wyoming protocol](https://github.com/OHF-Voice/wyoming)

Home Assistant's Assist voice pipeline speaks [Wyoming](https://github.com/OHF-Voice/wyoming), a lightweight protocol for local voice satellites and speech services—not the OpenAI or Anthropic APIs directly. [wyoming-openai](https://github.com/roryeckel/wyoming_openai) is an open-source proxy that bridges Wyoming to any OpenAI-compatible speech-to-text and text-to-speech backend, which is what lets Assist reach stdapi.ai.

**What the proxy adds on top of the audio routes:**

- **Wyoming discovery** - Advertises configured speech-to-text models and text-to-speech voices to Assist
- **Streaming synthesis** - Speaks a response as it is generated, in overlapping chunks, rather than waiting for the whole sentence
- **Format translation** - Reassembles the response as raw PCM frames for Assist's audio pipeline

## :material-help-circle-outline: Why Home Assistant + stdapi.ai?

<div class="grid cards" markdown>

- :material-swap-horizontal: __No Cloud Voice Subscription__
  <br>Replace a cloud speech-to-text/text-to-speech subscription with Amazon Transcribe and Amazon Polly, billed at Bedrock/AWS rates.

- :material-lock: __No Third-Party Voice Vendor__
  <br>Spoken audio and transcripts are processed by Amazon Transcribe, Polly and Bedrock in the regions you enable, reached through your own deployment — no consumer voice-assistant vendor in the path.

- :material-home-automation: __Works with Your Existing Assist Setup__
  <br>Assist's speech-to-text and text-to-speech pipeline selection is unchanged—only the backend the proxy talks to is stdapi.ai.

- :material-currency-usd-off: __Pay-Per-Use Pricing__
  <br>No per-satellite or per-minute voice assistant fees. Pay only Amazon Transcribe and Amazon Polly rates for actual usage.

</div>

```mermaid
%%{init: {'flowchart': {'htmlLabels': true}} }%%
flowchart LR
  ha["Home Assistant\nAssist"] -->|Wyoming| proxy["wyoming-openai"]
  proxy -->|OpenAI API| stdapi["<img src='../styles/logo.svg' style='height:64px;width:auto;vertical-align:middle;' /> stdapi.ai"]
  stdapi --> transcribe["<img src='../styles/logo_amazon_transcribe.svg' style='height:64px;width:auto;vertical-align:middle;' /> Amazon Transcribe"]
  stdapi --> polly["<img src='../styles/logo_amazon_polly.svg' style='height:64px;width:auto;vertical-align:middle;' /> Amazon Polly"]
```

## :material-check-circle: Prerequisites

!!! info "What You'll Need"
    - ✓ **stdapi.ai deployed** - [See deployment guide](operations_getting_started.md) or [run locally with Docker](operations_getting_started_local.md)
    - ✓ **Your stdapi.ai URL** - reachable from wherever the proxy runs, e.g. `https://api.example.com`
    - ✓ **Your API key** - From Terraform output or configuration
    - ✓ **Home Assistant** - With the Assist voice pipeline set up
    - ✓ **A place to run wyoming-openai** - A container alongside Home Assistant, e.g. as a Home Assistant OS add-on or a standalone container

---

## :material-cog: Configuration

wyoming-openai is configured through environment variables, split into a speech-to-text half and a text-to-speech half. Point both at your stdapi.ai deployment.

!!! example "Environment Variables"
    ```bash
    # Speech to text
    STT_OPENAI_URL=https://YOUR_STDAPI_URL/v1
    STT_OPENAI_KEY=YOUR_STDAPI_KEY
    STT_MODELS=amazon.transcribe

    # Text to speech
    TTS_OPENAI_URL=https://YOUR_STDAPI_URL/v1
    TTS_OPENAI_KEY=YOUR_STDAPI_KEY
    TTS_MODELS=amazon.polly-neural
    TTS_VOICES=alloy

    # Backend selection
    STT_BACKEND=OPENAI
    TTS_BACKEND=OPENAI
    ```

The proxy calls `POST /v1/audio/transcriptions` (see [Audio Transcriptions API](api_openai_audio_transcriptions.md)) for speech to text and `POST /v1/audio/speech` (see [Audio Speech API](api_openai_audio_speech.md)) for text to speech, so `STT_MODELS` must be a speech-to-text-capable model and `TTS_MODELS` a text-to-speech-capable model from the correct family.

!!! tip "Pin the backend"
    Left unset, wyoming-openai probes a few well-known self-hosted backends before falling back to a generic OpenAI-compatible one. Setting `STT_BACKEND=OPENAI` and `TTS_BACKEND=OPENAI` skips that probing and connects directly.

!!! tip "A cheaper speech-to-text model"
    `STT_MODELS=amazon.nova-2-sonic-v1:0` transcribes through [Amazon Nova Sonic](api_openai_audio_transcriptions.md#amazon-nova-sonic) instead of Amazon Transcribe: the lowest-cost transcription available here, punctuated and in the language spoken. It answers `json` and `text` only — which is all the proxy asks for — and caps a recording at 10 minutes, well beyond any voice command. Keep `amazon.transcribe` if the same deployment also serves subtitles, timestamps or speaker labels.

### :material-waveform: Streaming Speech to Text

Enables: recognizing a spoken command phrase by phrase, instead of after the whole utterance has been recorded.

!!! example "Environment Variables"
    ```bash
    STT_STREAMING_MODELS=amazon.transcribe
    ```

Only the models listed there are called in streaming mode, which is what makes the proxy ask stdapi.ai for a [streamed transcription](api_openai_audio_transcriptions.md#streaming). The gateway returns each phrase as it is recognized whenever the request names the language to expect; if the proxy sends none, set [`AWS_TRANSCRIBE_STREAM_LANGUAGES`](operations_configuration.md#aws-transcribe-stream-languages) on stdapi.ai to the languages your satellites actually speak and those requests take the same fast path. Streamed transcription stages nothing, so it works on a deployment with no S3 bucket configured.

!!! note "This is the streaming option to use, not the realtime one"
    stdapi.ai's [Realtime API](api_openai_realtime.md) serves speech-to-speech sessions, and a transcription-only session is requested through an [ephemeral client secret](api_openai_realtime.md#ephemeral-client-secrets) rather than on the socket — so a client that expects OpenAI's realtime *transcription* socket gets no transcript from it. Assist's pipeline is turn-based anyway: speech to text, then a conversation agent, then text to speech.

### :material-volume-high: Streaming Text to Speech

Enables: speaking a response as it is generated, instead of waiting for the whole sentence to synthesize.

!!! example "Environment Variables"
    ```bash
    TTS_STREAMING_MODELS=amazon.polly-neural
    ```

Naming the same model in both `TTS_MODELS` and `TTS_STREAMING_MODELS` puts its voice in the proxy's streaming program, so Assist can use it for both a plain synthesis request and a streamed one. The proxy splits a streamed reply into sentences and synthesizes several `/v1/audio/speech` calls concurrently, then replays the audio in the original order.

### :material-tune-vertical: Voice Mapping

`TTS_VOICES` lists OpenAI-style voice names (`alloy`, `echo`, `fable`, and so on); stdapi.ai maps each one to an Amazon Polly voice of matching gender and language. List one entry per voice you want Assist to offer.

---

## :material-rocket-launch: Terraform Deployment

Deploy Home Assistant, wyoming-openai, and stdapi.ai together on ECS Fargate:

**📦 [stdapi-ai/samples/getting_started_home_assistant](https://github.com/stdapi-ai/samples/tree/main/getting_started_home_assistant)**

**What's included:**

- Home Assistant and wyoming-openai in the same ECS Fargate task, talking over `localhost`
- stdapi.ai gateway connected to Amazon Bedrock, Amazon Transcribe, and Amazon Polly
- `configuration.yaml` seeded on first boot with the reverse-proxy trust settings Home Assistant needs behind an ALB
- Both container images pulled directly and anonymously from ghcr.io — no local build, no registry credential
- HTTPS-capable ALB on your own domain (needed for microphone access in the browser)

!!! warning "Demonstration sample, not a production Home Assistant deployment"
    AWS Fargate has no access to a home network: Zigbee/Z-Wave USB dongles, mDNS device discovery, and other LAN-only integrations do not work here, and a Wyoming voice satellite on your home LAN cannot reach this deployment unless you put a VPN between them. Use it to try Assist voice through Amazon Transcribe/Polly, or as a starting point for a self-hosted, cloud-reachable instance you administer through the web UI.

!!! warning "Local ECS module source"
    `module "home_assistant"` currently points at a local relative path (`../../../terraform-aws-ecs`) instead of the published registry module, because it needs S3 Files mount-point support (used to seed `configuration.yaml`) that isn't in a tagged release yet. This means `terraform-aws-ecs` must be checked out as a sibling of `samples/` before `tofu init` succeeds — see the sample's README for the exact layout.

**Deploy:**

```bash
git clone https://github.com/stdapi-ai/samples.git
git clone https://github.com/JGoutin/terraform-aws-ecs.git
cd samples/getting_started_home_assistant/terraform
tofu init
tofu apply
```

Three steps stay manual after `tofu apply`, for reasons specific to Home Assistant: creating the owner account through the onboarding wizard, adding the Wyoming integration (**Settings → Devices & Services**), and pointing an Assist pipeline at it. See the sample's README for the exact steps.

---

## :material-alert-outline: Known Issues

The proxy speaks the Wyoming protocol over its own TCP port, not HTTP—there is no `/health` endpoint to check readiness with a plain web request. Wait for a successful Wyoming `describe` exchange (or check the container logs) rather than polling an HTTP path.

## :material-arrow-right: Next Steps

<div class="grid cards" markdown>

- :material-rocket-launch: [**Getting Started**](operations_getting_started.md) — Deploy stdapi.ai to AWS with Terraform
- :material-docker: [**Local Development**](operations_getting_started_local.md) — Run stdapi.ai locally with Docker
- :material-puzzle: [**More Use Cases**](use_cases.md) — Explore other integrations and tools
- :material-api: [**API Overview**](api_overview.md) — Explore supported endpoints

</div>
