---
title: Speech to English API - Amazon Audio Translation
description: Translate audio from any language to English text using Amazon Transcribe + Translate or Amazon Bedrock audio models. OpenAI-compatible API with automatic language detection.
keywords: audio translation API, speech translation, AWS Translate API, multilingual transcription, audio to English, OpenAI translation, language detection API
---

# Speech to English API

Translate audio from any language to English text with Amazon Transcribe + Translate or Amazon Bedrock audio-capable models through an OpenAI-compatible interface.

## Why Choose the Speech to English API?

<div class="grid cards" markdown>

- :material-earth-arrow-right: __Automatic Language Detection__
  <br>Upload audio in any language. AWS automatically detects the source language and translates to English text.

- :material-account-network: __Multiple Translation Options__
  <br>Choose Amazon Transcribe + Translate for a traditional pipeline, or use Bedrock audio models with built-in translation capabilities.

- :material-file-multiple: __Multiple Output Formats__
  <br>Choose from text, JSON, verbose JSON with timestamps, or translated subtitle files (SRT/VTT).

- :material-subtitles: __Subtitle Translation__
  <br>Generate translated SRT and VTT subtitle files directly with precise timing for international video content.

</div>

## Available Endpoints

| Endpoint                 | Method | What It Does                                     | Powered By                                                   | MCP Tool                    |
|--------------------------|--------|--------------------------------------------------|--------------------------------------------------------------|-----------------------------|
| `/v1/audio/translations` | `POST` | Transcribe any language and translate to English | Amazon Transcribe + Translate or Amazon Bedrock Audio Models | `openai_audio_translation` |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                 |                 Status                  | Notes                         |
|-------------------------|:---------------------------------------:|-------------------------------|
| **Input**               |                                         |                               |
| Audio file upload       |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Multipart file upload         |
| JSON body input         | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" }| Base64, data URI, HTTPS URL, S3 URI, or `file-id:` reference — for MCP / AI agents |
| **Output Formats**      |                                         |                               |
| `json`                  |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Structured translation        |
| `text`                  |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Plain English text            |
| `verbose_json`          |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | With timestamps (Amazon Transcribe; not Bedrock models) |
| `srt`                   |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | English subtitles with timing (Amazon Transcribe; not Bedrock models) |
| `vtt`                   |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | English WebVTT subtitles (Amazon Transcribe; not Bedrock models) |
| **Language**            |                                         |                               |
| Auto language detection |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Automatic source detection    |
| **Translation**         |                                         |                               |
| Translation to English  |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Amazon Translate (with `amazon.transcribe`) or native model translation (Bedrock models) |
| **Advanced**            |                                         |                               |
| `prompt`                |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Bedrock models only; rejected by Amazon Transcribe |
| `temperature`           |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Bedrock models only; rejected by Amazon Transcribe |
| Extra model-specific params | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" }| Amazon Transcribe + Translate optional settings via JSON body (see below) |
| **Usage tracking**      |                                         |                               |
| Input audio duration    |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Seconds (billing unit on Amazon Transcribe) |
| Output text tokens      |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | On models from Bedrock        |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep role="img" aria-label="Model-dependent" } **Available on Select Models** — Check your model's capabilities
* :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } **Extra Feature** — Enhanced capability beyond OpenAI API

</div>

## Model Support

### ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Models

| Model             | Supported Languages | Notes                                                                                                      |
|-------------------|---------------------|------------------------------------------------------------------------------------------------------------|
| amazon.transcribe | 100+                | Full-featured transcription with speaker diarization and subtitle generation at the cost of higher latency |

!!! warning "Configuration Required"
    You must configure a bucket to use this model, through `AWS_S3_BUCKET`, `AWS_TRANSCRIBE_S3_BUCKET`, or an `AWS_S3_REGIONAL_BUCKETS` entry for a region where Amazon Transcribe is a candidate. This bucket is used for temporary storage during transcription processing.

### ![Mistral](styles/logo_mistralai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Mistral Models

| Model                           | Supported Languages       | Notes                                              |
|---------------------------------|---------------------------|----------------------------------------------------|
| mistral.voxtral-mini-3b-2507    | Multilingual (auto-detected) | Compact model for fast transcription            |
| mistral.voxtral-small-24b-2507  | Multilingual (auto-detected) | Larger model for enhanced accuracy              |

!!! warning "Mistral Voxtral Limitations"
    Mistral Voxtral models have the following restrictions when running on Amazon Bedrock:

    - **File size limit**: ~2MB maximum input file size
    - **Audio channels**: Mono channel audio only (single channel)

### ![Amazon Nova](styles/logo_amazon_nova.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Nova Sonic

| Model                     | Supported Languages          | Notes                                        |
|---------------------------|------------------------------|----------------------------------------------|
| amazon.nova-2-sonic-v1:0  | Multilingual (auto-detected) | Speech translated to English in one request  |

Name this model to translate through Amazon Nova Sonic instead of Amazon Transcribe and Amazon Translate. The model listens to the audio and answers with the English text directly, in a single request.

!!! warning "What this model does not provide"
    - **Response formats**: `json` and `text` only. `srt`, `vtt` and `verbose_json` are rejected — this model returns no timestamps and does not report which language it detected. Use `amazon.transcribe` for subtitles or timestamps.
    - **Audio length**: up to 10 minutes per request. Longer recordings are rejected; use `amazon.transcribe`, which has no such limit.
    - **Cost**: translation is billed for the English answer the model produces as well as for the audio it hears, so it costs more per minute than transcription with the same model.

### Other Amazon Bedrock Models

Any Amazon Bedrock model that accepts the `SPEECH` input modality through the Converse API can also translate audio out of the box: the gateway sends the audio together with a translation prompt and returns the model's English text output.

!!! tip "Audio Input Formats on Bedrock Models"
    Uploads in the formats the Bedrock Converse audio block accepts — `aac`, `flac`, `m4a`, `mka`, `mkv`, `mp3`, `mp4`, `mpeg`, `mpga`, `ogg`, `opus`, `pcm`, `wav`, `webm`, and `x-aac` — are sent through as-is. Any other audio or video upload is automatically converted to FLAC before translation (requires FFmpeg on the server), including the audio track of a video container. An upload that is neither audio nor video is rejected with the list of accepted formats; an audio or video file whose track cannot be decoded is rejected as carrying no decodable audio.

## Advanced Features

### ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Transcribe Features

**Model & Features:**

- Use `amazon.transcribe` with the same interface as OpenAI's Whisper API
- **Or use OpenAI model names directly**: `whisper-1`, `gpt-transcribe`, `gpt-4o-transcribe`, and `gpt-4o-mini-transcribe` work out of the box (they map to `amazon.transcribe`)
- Automatic transcription + translation pipeline in one request
- Multiple output formats: `text`, `json`, `verbose_json`, `srt`, `vtt`
- Automatic source language detection (zero configuration)
- **Smart Subtitle Translation** :material-translate:{ .highlight }: Subtitle timing is preserved during translation

!!! tip "OpenAI Model Compatibility"
    stdapi.ai includes built-in model aliases that map the OpenAI model names to Amazon Transcribe:

    - `whisper-1` → `amazon.transcribe`
    - `gpt-transcribe` → `amazon.transcribe`
    - `gpt-4o-transcribe` → `amazon.transcribe`
    - `gpt-4o-mini-transcribe` → `amazon.transcribe`

    These aliases enable seamless compatibility with OpenAI-based tools and applications without any configuration changes (the realtime-oriented `gpt-live-transcribe` is not aliased: it belongs to a streaming API this route does not emulate). You can also [customize or override these aliases](operations_configuration.md#model-aliases) to suit your needs.

**Note:** With `amazon.transcribe`, the `prompt` and `temperature` parameters are rejected with an error to ensure consistent translation accuracy. Bedrock audio models accept both.

!!! warning "Source languages Amazon Translate does not cover"
    Amazon Transcribe recognises more languages than Amazon Translate can translate into English. When the detected source language is not one of [Amazon Translate's supported languages](https://docs.aws.amazon.com/translate/latest/dg/what-is-languages.html), the request returns HTTP 400 listing the supported language codes instead of a partial result. Transcribe the audio with [`/v1/audio/transcriptions`](api_openai_audio_transcriptions.md) to keep it in its original language.

### Provider-Specific Parameters

`amazon.transcribe` first transcribes the audio, then translates it, and each step has its own provider-specific parameters — both reachable via the same `application/json` request body.

**Transcription step (Amazon Transcribe):** the same [extra parameters documented for `/v1/audio/transcriptions`](api_openai_audio_transcriptions.md#provider-specific-parameters) (`ContentRedaction`, `VocabularyName`, `VocabularyFilterName`/`VocabularyFilterMethod`, `ShowAlternatives`/`MaxAlternatives`, `ChannelIdentification`, `ToxicityDetection`, `IdentifyMultipleLanguages`/`LanguageOptions`, `LanguageIdSettings`, `ModelSettings`) apply here too.

**Translation step (Amazon Translate):** `Settings` and `TerminologyNames` control the English output register and glossary:

```json
{
  "model": "amazon.transcribe",
  "file": "data:audio/mp3;base64,<base64-encoded-audio>",
  "Settings": {
    "Formality": "FORMAL",
    "Profanity": "MASK"
  },
  "TerminologyNames": ["MyProductGlossary"]
}
```

- `Settings.Formality` (`FORMAL` or `INFORMAL`): Register of the translated text, for languages that support formality
- `Settings.Profanity` (`MASK`): Mask profane words and phrases in the translation
- `Settings.Brevity` (`ON`): Per AWS Translate, reduces the length of the translation output for most translations; unsupported language pairs silently ignore it
- `TerminologyNames` (list): Apply one or more custom terminologies (domain-specific glossaries) to the translation

Both settings apply consistently to the primary translated text and, for `response_format=verbose_json`, to every per-segment translation.

!!! warning "`verbose_json` translates the transcript twice"
    For `response_format=verbose_json`, the full transcript is sent to AWS Translate once, and then every segment is sent again individually so segment-level translations are available. AWS Translate bills by character, so `verbose_json` costs roughly double the translation characters of `text` or `json` for the same audio.

!!! warning "Terminologies must already exist"
    `TerminologyNames` references AWS Translate custom terminology resources created ahead of time via the AWS Translate console, CLI, or SDK (`ImportTerminology`) — stdapi.ai does not create or manage them. An unknown name is rejected by AWS Translate with a client error.

Invalid `Settings` values (e.g. an unsupported `Formality`) are rejected with HTTP 400 before any partial translation occurs.

## Available Request Headers

This endpoint supports standard Bedrock headers for enhanced control over your requests. All headers are optional and can be combined as needed.

### Content Safety (Guardrails)

| Header                               | Purpose                            | Valid Values               |
|--------------------------------------|------------------------------------|----------------------------|
| `X-Amzn-Bedrock-GuardrailIdentifier` | Guardrail ID for content filtering | Your guardrail identifier  |
| `X-Amzn-Bedrock-GuardrailVersion`    | Guardrail version                  | Version number (e.g., `1`) |

The guardrail evaluates the English translation the model produced, not the audio sent. `X-Amzn-Bedrock-Trace` is accepted but has no effect on this route — no guardrail trace is returned.

### Performance Optimization

| Header                                     | Purpose                | Valid Values                              |
|--------------------------------------------|------------------------|-------------------------------------------|
| `X-Amzn-Bedrock-Service-Tier`              | Service tier selection | `default`, `flex`, `priority`, `reserved` |
| `X-Amzn-Bedrock-PerformanceConfig-Latency` | Latency optimization   | `standard`, `optimized`                   |

Both apply only to models translated through the Amazon Bedrock runtime. They have no effect on `amazon.transcribe`, which is served by Amazon Transcribe and AWS Translate, nor on Amazon Nova Sonic, which is served over its own bidirectional stream.

**Example with headers:**

```bash
curl -X POST "$BASE/v1/audio/translations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "X-Amzn-Bedrock-GuardrailIdentifier: your-guardrail-id" \
  -H "X-Amzn-Bedrock-GuardrailVersion: 1" \
  -F file=@spanish-interview.mp3 \
  -F model=amazon.transcribe \
  -F response_format=json
```

!!! info "Detailed Documentation"
    For complete information about these headers, configuration options, and use cases, see:

    - [Bedrock Guardrails Configuration](operations_configuration.md#bedrock-guardrails)
    - [Service Tier and Performance Configuration](operations_configuration.md#bedrock-service-tier-and-performance-configuration)

## Try It Now

**Translate foreign audio to English text:**

```bash
curl -X POST "$BASE/v1/audio/translations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F file=@spanish-interview.mp3 \
  -F model=amazon.transcribe \
  -F response_format=json
```

**Translate via JSON body (MCP and AI agents):**

When using MCP tools or HTTP clients that cannot construct multipart requests, pass the audio as a data URI or URL:

```bash
# Data URI (inline base64)
curl -X POST "$BASE/v1/audio/translations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "file": "data:audio/mp3;base64,<base64-encoded-audio>",
    "model": "amazon.transcribe"
  }'
```

```bash
# HTTPS URL (server fetches the audio)
curl -X POST "$BASE/v1/audio/translations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "file": "https://example.com/audio.mp3",
    "model": "amazon.transcribe"
  }'
```

```bash
# Files API reference (file-id: URI scheme)
curl -X POST "$BASE/v1/audio/translations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "file": "file-id:file-0190c51c7de7455d9b8c2efe27dfbf67",
    "model": "amazon.transcribe"
  }'
```

See [Files API → Referencing Uploaded Files](api_openai_files.md#referencing-uploaded-files-via-the-file-id-uri-scheme) for the full description of the `file-id:` URI scheme.

**Translate foreign audio to English subtitles:**

```bash
curl -OJ -X POST "$BASE/v1/audio/translations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F file=@spanish-interview.mp3 \
  -F model=amazon.transcribe \
  -F response_format=srt
```

---

**Ready to translate multilingual audio?** Explore available models in the [Models API](api_openai_models.md).
