---
title: Speech to Text API - Amazon Transcribe & Bedrock Audio Models
description: Transcribe audio to text with Amazon Transcribe or Amazon Bedrock audio-capable models. OpenAI-compatible STT API supporting 100+ languages, speaker diarization, and multiple output formats.
keywords: speech to text API, audio transcription API, AWS Transcribe API, STT API, OpenAI Whisper alternative, audio to text, transcription service, speaker diarization
---

# Speech to Text API

Transcribe audio to text with Amazon Transcribe or Amazon Bedrock audio-capable models through an OpenAI-compatible interface.

## Why Choose the Speech to Text API?

<div class="grid cards" markdown>

- :material-translate: __Multiple Transcription Options__
  <br>Choose Amazon Transcribe for 100+ languages with speaker diarization, or use Bedrock audio models for advanced capabilities.

- :material-clock-fast: __Real-Time or Batch__
  <br>Stream transcriptions in real-time via SSE or process files efficiently with either service.

- :material-subtitles: __Subtitle Generation__
  <br>Generate SRT and VTT subtitle files directly with precise timing for video content.

- :material-account-multiple: __Advanced Features__
  <br>Speaker diarization, word-level timestamps, and automatic language detection. Feature availability varies by model choice.

</div>

## Quick Start: Available Endpoint

| Endpoint                    | Method | What It Does                             | Powered By                                       | MCP Tool                  |
|-----------------------------|--------|------------------------------------------|--------------------------------------------------|---------------------------|
| `/v1/audio/transcriptions`  | `POST` | Convert spoken audio to written text     | Amazon Transcribe or Amazon Bedrock Audio Models | `openai_audio_transcription` |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                    |                  Status                  | Notes                                                            |
|----------------------------|:----------------------------------------:|------------------------------------------------------------------|
| **Input**                  |                                          |                                                                  |
| Audio file upload          |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Multipart file upload                                            |
| JSON body input            | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Base64, data URI, HTTPS URL, S3 URI, or `file-id:` reference — for MCP / AI agents |
| **Output Formats**         |                                          |                                                                  |
| `json`                     |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Structured transcription                                         |
| `text`                     |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Plain text output                                                |
| `verbose_json`             |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | With timestamps and details (Amazon Transcribe; not Bedrock models) |
| `diarized_json`            |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | With speaker identification (Amazon Transcribe; not Bedrock models) |
| `srt`                      |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Subtitle format with timing (Amazon Transcribe; not Bedrock models) |
| `vtt`                      |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | WebVTT subtitle format (Amazon Transcribe; not Bedrock models)   |
| **Language**               |                                          |                                                                  |
| Language specification     |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | ISO-639-1 language codes                                         |
| Auto language detection    |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Automatic identification                                         |
| **Streaming**              |                                          |                                                                  |
| `stream` (SSE streaming)   |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Set `stream: true` to receive incremental results as server-sent events |
| **Advanced**               |                                          |                                                                  |
| `timestamp_granularities`  |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Word or segment level; requires `response_format=verbose_json` (Amazon Transcribe only) |
| Speaker diarization        |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Automatic speaker separation; requires `response_format=diarized_json` (Amazon Transcribe only) |
| `known_speaker_names`      | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted but ignored — diarization falls back to generic speaker labels |
| `known_speaker_references` | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted but ignored — diarization falls back to generic speaker labels |
| `chunking_strategy`        |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }    | Only `auto` is accepted; other values are rejected               |
| `temperature`              |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Bedrock models only; rejected by Amazon Transcribe               |
| `prompt`                   |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Bedrock models only; rejected by Amazon Transcribe               |
| `include` (`logprobs`)     |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Bedrock models only; requires `response_format=json`; rejected by Amazon Transcribe |
| Extra model-specific params | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Amazon Transcribe optional settings via JSON body (see below)  |
| **Usage tracking**         |                                          |                                                                  |
| Input audio duration       |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Seconds (billing unit on Amazon Transcribe)                      |
| Output text tokens         |       :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | On models from Bedrock                                           |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep role="img" aria-label="Model-dependent" } **Available on Select Models** — Check your model's capabilities
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } **Extra Feature** — Enhanced capability beyond OpenAI API

</div>

## Model Support

### ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Models

| Model             | Supported Languages | Notes                                                                                                      |
|-------------------|---------------------|------------------------------------------------------------------------------------------------------------|
| amazon.transcribe | 100+                | Full-featured transcription with speaker diarization and subtitle generation at the cost of higher latency |

!!! warning "Configuration Required"
    You must configure the `AWS_S3_BUCKET` or `AWS_TRANSCRIBE_S3_BUCKET` environment variable with a bucket in the main AWS region to use this model. This bucket is used for temporary storage during transcription processing.

### ![Mistral](styles/logo_mistralai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Mistral Models

| Model                           | Supported Languages       | Notes                                              |
|---------------------------------|---------------------------|----------------------------------------------------|
| mistral.voxtral-mini-3b-2507    | Multilingual (auto-detected) | Compact model for fast transcription            |
| mistral.voxtral-small-24b-2507  | Multilingual (auto-detected) | Larger model for enhanced accuracy              |

!!! warning "Mistral Voxtral Limitations"
    Mistral Voxtral models have the following restrictions when running on Amazon Bedrock:

    - **File size limit**: ~2MB maximum input file size
    - **Audio channels**: Mono channel audio only (single channel)

## Advanced Features

### ![Amazon Transcribe](styles/logo_amazon_transcribe.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Transcribe Features

**Model & Features:**

- Use `amazon.transcribe` with the same interface as OpenAI's Whisper API
- **Or use OpenAI model name directly**: `whisper-1` works out of the box (maps to `amazon.transcribe`)
- Auto-detect language or specify it for faster processing
- Word-level or segment-level timestamps with `verbose_json`
- **Speaker Diarization** :material-account-multiple:{ .highlight }: Automatically identify and label different speakers with `diarized_json`
- **Native Subtitles** :material-file-video:{ .highlight }: SRT/VTT files generated directly by Amazon Transcribe with precise timing

!!! tip "OpenAI Model Compatibility"
    stdapi.ai includes a built-in model alias that maps the OpenAI model name to Amazon Transcribe:

    - `whisper-1` → `amazon.transcribe`

    This alias enables seamless compatibility with OpenAI-based tools and applications without any configuration changes. You can also [customize or override this alias](operations_configuration.md#model-aliases) to suit your needs.

**Note:** With `amazon.transcribe`, the `prompt`, `temperature`, and `include` parameters are rejected with an error to ensure consistent transcription accuracy. The `known_speaker_names` and `known_speaker_references` parameters are accepted but ignored for every model: Amazon Transcribe's automatic speaker diarization runs without known speaker references, falling back to generic speaker labels.

!!! tip "Performance Tips: Optimize Speed & Cost"
    - **Specify the language** if you know it—skips auto-detection for faster processing and lower AWS costs

### Provider-Specific Parameters

Unlock advanced Amazon Transcribe capabilities by passing provider-specific parameters directly in your request body. These parameters are forwarded to Transcribe's `StartTranscriptionJob` API.

!!! warning "JSON body required"
    Unlike the `multipart/form-data` upload, extra parameters are only reachable through the `application/json` request body (`file` as base64, data URI, HTTPS URL, or `file-id:` reference) — the multipart path only accepts the documented OpenAI fields.

**PII Redaction:**

Redact personally identifiable information from the transcript (only the single-output `redacted` mode is supported):

```json
{
  "model": "amazon.transcribe",
  "file": "data:audio/mp3;base64,<base64-encoded-audio>",
  "ContentRedaction": {
    "RedactionType": "PII",
    "PiiEntityTypes": ["NAME", "SSN", "CREDIT_DEBIT_NUMBER"]
  }
}
```

**Custom Vocabulary and Filtering:**

Improve recognition of domain-specific terms and mask or remove profanity/sensitive words:

```json
{
  "model": "amazon.transcribe",
  "file": "data:audio/mp3;base64,<base64-encoded-audio>",
  "VocabularyName": "MyCustomVocabulary",
  "VocabularyFilterName": "MyProfanityFilter",
  "VocabularyFilterMethod": "mask"
}
```

**Alternative Transcriptions and Channel Identification:**

Request multiple candidate transcriptions per segment, or transcribe each audio channel separately (e.g. two-party phone calls recorded in stereo):

```json
{
  "model": "amazon.transcribe",
  "file": "data:audio/mp3;base64,<base64-encoded-audio>",
  "ShowAlternatives": true,
  "MaxAlternatives": 3,
  "ChannelIdentification": true
}
```

!!! warning "Incompatible with diarized_json"
    `ChannelIdentification` cannot be combined with `response_format=diarized_json`, which already forces AWS speaker-label diarization. Requesting both returns HTTP 400.

**Toxicity Detection:**

Flag toxic content (profanity, hate speech, harassment) in the transcript:

```json
{
  "model": "amazon.transcribe",
  "file": "data:audio/mp3;base64,<base64-encoded-audio>",
  "ToxicityDetection": [{"ToxicityCategories": ["ALL"]}]
}
```

**Multi-Language Identification:**

Detect and transcribe multiple languages spoken in the same audio, optionally restricted to a candidate list:

```json
{
  "model": "amazon.transcribe",
  "file": "data:audio/mp3;base64,<base64-encoded-audio>",
  "IdentifyMultipleLanguages": true,
  "LanguageOptions": ["en-US", "es-US", "fr-FR"]
}
```

**Configuration Options:**

**Option 1: Per-Request**

Add provider-specific parameters directly in your JSON request body (as shown in examples above).

**Option 2: Server-Wide Defaults**

Configure default parameters for `amazon.transcribe` via the `DEFAULT_MODEL_PARAMS` environment variable:

```bash
export DEFAULT_MODEL_PARAMS='{
  "amazon.transcribe": {
    "VocabularyFilterName": "MyProfanityFilter",
    "VocabularyFilterMethod": "mask"
  }
}'
```

**Note:** Per-request parameters override server-wide defaults.

**Behavior:**

**Compatible parameters** are forwarded to Amazon Transcribe and applied; **unsupported parameters or values** return HTTP 400 with an error message.

**Available Parameters:**

The following parameters from Amazon Transcribe's [StartTranscriptionJob API](https://docs.aws.amazon.com/transcribe/latest/APIReference/API_StartTranscriptionJob.html) can be used:

- `ContentRedaction` (object): PII redaction — `RedactionType` (`PII`), `PiiEntityTypes` (list), `RedactionOutput` (`redacted` only; `redacted_and_unredacted` is rejected — the unredacted copy is not tracked for automatic cleanup)
- `VocabularyName` (string): Custom vocabulary to improve recognition accuracy
- `VocabularyFilterName` / `VocabularyFilterMethod` (string / `mask`, `remove`, `tag`): Profanity or sensitive-word filtering
- `ShowAlternatives` / `MaxAlternatives` (bool / integer `2`-`10`): Return multiple candidate transcriptions per segment
- `ChannelIdentification` (bool): Transcribe each audio channel separately (incompatible with `diarized_json`)
- `MaxSpeakerLabels` (integer `2`-`30`): Maximum speakers to identify with `response_format=diarized_json` (default `10`)
- `ShowSpeakerLabels` (bool): Always on with `response_format=diarized_json`; setting it directly with another format runs AWS speaker labeling without exposing speaker data in the response
- `ToxicityDetection` (list): Toxic-content flagging — `[{"ToxicityCategories": ["ALL"]}]`
- `IdentifyMultipleLanguages` / `LanguageOptions` (bool / list): Multi-language identification, optionally restricted to a candidate list (supersedes `language`)
- `ModelSettings` (object): `LanguageModelName` — custom language model selection

`VocabularyName`, `VocabularyFilterName`, and custom language models must already exist in your AWS account (created via the AWS Transcribe console, CLI, or SDK) before being referenced here.

## Try It Now

**Transcribe audio to JSON:**

```bash
curl -X POST "$BASE/v1/audio/transcriptions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F file=@meeting-recording.mp3 \
  -F model=amazon.transcribe \
  -F response_format=json
```

**Transcribe via JSON body (MCP and AI agents):**

When using MCP tools or HTTP clients that cannot construct multipart requests, pass the audio as a data URI or URL:

```bash
# Data URI (inline base64)
curl -X POST "$BASE/v1/audio/transcriptions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "file": "data:audio/mp3;base64,<base64-encoded-audio>",
    "model": "amazon.transcribe",
    "response_format": "json"
  }'
```

```bash
# HTTPS URL (server fetches the audio)
curl -X POST "$BASE/v1/audio/transcriptions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "file": "https://example.com/audio.mp3",
    "model": "amazon.transcribe"
  }'
```

```bash
# Files API reference (file-id: URI scheme)
curl -X POST "$BASE/v1/audio/transcriptions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "file": "file-id:file-0190c51c7de7455d9b8c2efe27dfbf67",
    "model": "amazon.transcribe"
  }'
```

See [Files API → Referencing Uploaded Files](api_openai_files.md#referencing-uploaded-files-via-the-file-id-uri-scheme) for the full description of the `file-id:` URI scheme.

**Generate subtitles:**

```bash
curl -X POST "$BASE/v1/audio/transcriptions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F file=@video-audio.mp3 \
  -F model=amazon.transcribe \
  -F response_format=srt \
  -F language=en
```

**Transcribe with speaker diarization:**

```bash
curl -X POST "$BASE/v1/audio/transcriptions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F file=@meeting-recording.mp3 \
  -F model=amazon.transcribe \
  -F response_format=diarized_json
```

**Stream a transcription as SSE events:**

Set `stream=true` to receive the transcript incrementally as server-sent events (`transcript.text.delta` events followed by a final `transcript.text.done` event):

```bash
curl -N -X POST "$BASE/v1/audio/transcriptions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F file=@meeting-recording.mp3 \
  -F model=amazon.transcribe \
  -F stream=true
```

---

**Ready to transcribe audio?** Explore available transcription models in the [Models API](api_openai_models.md).
