---
title: Speech to Text API - AWS Transcribe & Bedrock Audio Models
description: Transcribe audio to text with AWS Transcribe or AWS Bedrock audio-capable models. OpenAI-compatible STT API supporting 100+ languages, speaker diarization, and multiple output formats.
keywords: speech to text API, audio transcription API, AWS Transcribe API, STT API, OpenAI Whisper alternative, audio to text, transcription service, speaker diarization
---

# Speech to Text API

Transcribe audio to text with AWS Transcribe or AWS Bedrock audio-capable models through an OpenAI-compatible interface.

## Why Choose Speech to Text?

<div class="grid cards" markdown>

- :material-translate: __Multiple Transcription Options__
  <br>Choose AWS Transcribe for 100+ languages with speaker diarization, or use Bedrock audio models for advanced capabilities.

- :material-clock-fast: __Real-Time or Batch__
  <br>Stream transcriptions in real-time via SSE or process files efficiently with either service.

- :material-subtitles: __Subtitle Generation__
  <br>Generate SRT and VTT subtitle files directly with precise timing for video content.

- :material-account-multiple: __Advanced Features__
  <br>Speaker diarization, word-level timestamps, and automatic language detection. Feature availability varies by model choice.

</div>

## Quick Start: Available Endpoint

| Endpoint                    | Method | What It Does                             | Powered By                                | MCP Tool                  |
|-----------------------------|--------|------------------------------------------|-------------------------------------------|---------------------------|
| `/v1/audio/transcriptions`  | POST   | Convert spoken audio to written text     | AWS Transcribe or AWS Bedrock Audio Models | `openai_audio_transcription` |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                    |                 Status                  | Notes                                                |
|----------------------------|:---------------------------------------:|------------------------------------------------------|
| **Input**                  |                                         |                                                      |
| Audio file upload          |   :material-check-circle:{ .success }   | Multipart file upload                                |
| **Output Formats**         |                                         |                                                      |
| `json`                     |   :material-check-circle:{ .success }   | Structured transcription                             |
| `text`                     |   :material-check-circle:{ .success }   | Plain text output                                    |
| `verbose_json`             |      :material-cog:{ .model-dep }       | With timestamps and details                          |
| `diarized_json`            |      :material-cog:{ .model-dep }       | With speaker identification                          |
| `srt`                      |      :material-cog:{ .model-dep }       | Subtitle format with timing                          |
| `vtt`                      |      :material-cog:{ .model-dep }       | WebVTT subtitle format                               |
| **Language**               |                                         |                                                      |
| Language specification     |      :material-cog:{ .model-dep }       | ISO-639-1 language codes                             |
| Auto language detection    |   :material-check-circle:{ .success }   | Automatic identification                             |
| **Streaming**              |                                         |                                                      |
| SSE streaming              |   :material-check-circle:{ .success }   | Event-based streaming                                |
| **Advanced**               |                                         |                                                      |
| Timestamp granularity      |   :material-check-circle:{ .success }   | Word or segment level                                |
| Speaker diarization        |   :material-check-circle:{ .success }   | Automatic speaker separation                         |
| `known_speaker_names`      | :material-close-circle:{ .unsupported } | Not available                                        |
| `known_speaker_references` | :material-close-circle:{ .unsupported } | Not available                                        |
| `chunking_strategy`        |   :material-minus-circle:{ .partial }   | Only `auto` is supported                             |
| `temperature`              |      :material-cog:{ .model-dep }       | Model temperature                                    |
| `prompt`                   |      :material-cog:{ .model-dep }       | Extra transcription prompt                           |
| `logprobs`                 |      :material-cog:{ .model-dep }       | Log probabilities for token-level confidence scoring |
| **Usage tracking**         |                                         |                                                      |
| Input audio duration       |   :material-check-circle:{ .success }   | Seconds (billing unit on AWS Transcribe)             |
| Output text tokens         |      :material-cog:{ .model-dep }       | On models from Bedrock                               |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep } **Available on Select Models** — Check your model's capabilities
* :material-minus-circle:{ .partial } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported } **Unsupported** — Not available in this implementation

</div>

## Model Support

### ![AWS Transcribe](styles/logo_amazon_transcribe.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Models

| Model             | Supported Languages | Notes                                                                                                      |
|-------------------|---------------------|------------------------------------------------------------------------------------------------------------|
| amazon.transcribe | 100+                | Full-featured transcription with speaker diarization and subtitle generation at the cost of higher latency |

!!! warning "Configuration Required"
    You must configure the `AWS_S3_BUCKET` or `AWS_TRANSCRIBE_S3_BUCKET` environment variable with a bucket in the main AWS region to use this model. This bucket is used for temporary storage during transcription processing.

### ![Mistral](styles/logo_mistralai.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Mistral Models

| Model                           | Supported Languages | Notes                                              |
|---------------------------------|---------------------|----------------------------------------------------|
| mistral.voxtral-mini-3b-2507    | 100+                | Compact model for fast transcription               |
| mistral.voxtral-small-24b-2507  | 100+                | Larger model for enhanced accuracy                 |

!!! warning "Mistral Voxtral Limitations"
    Mistral Voxtral models have the following restrictions when running on AWS Bedrock:

    - **File size limit**: ~2MB maximum input file size
    - **Audio channels**: Mono channel audio only (single channel)

## Advanced Features

### ![AWS Transcribe](styles/logo_amazon_transcribe.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Transcribe Features

**Model & Features:**

- Use `amazon.transcribe` with the same interface as OpenAI's Whisper API
- **Or use OpenAI model name directly**: `whisper-1` works out of the box (maps to `amazon.transcribe`)
- Auto-detect language or specify it for faster processing
- Word-level or segment-level timestamps with `verbose_json`
- **Speaker Diarization** :material-account-multiple:{ .highlight }: Automatically identify and label different speakers with `diarized_json`
- **Native Subtitles** :material-file-video:{ .highlight }: SRT/VTT files generated directly by AWS Transcribe with precise timing

!!! tip "OpenAI Model Compatibility"
    stdapi.ai includes a built-in model alias that maps the OpenAI model name to AWS Transcribe:

    - `whisper-1` → `amazon.transcribe`

    This alias enables seamless compatibility with OpenAI-based tools and applications without any configuration changes. You can also [customize or override this alias](operations_configuration.md#model-aliases) to suit your needs.

**Note:** The `prompt`, `temperature`, `chunking_strategy`, `known_speaker_names`, and `known_speaker_references` parameters are not supported to ensure consistent transcription accuracy. AWS Transcribe provides automatic speaker diarization without requiring known speaker references.

!!! tip "Performance Tips: Optimize Speed & Cost"
    - **Specify the language** if you know it—skips auto-detection for faster processing and lower AWS costs

## Try It Now

**Transcribe audio to JSON:**

```bash
curl -X POST "$BASE/v1/audio/transcriptions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F file=@meeting-recording.mp3 \
  -F model=amazon.transcribe \
  -F response_format=json
```

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

---

**Ready to transcribe audio?** Explore available transcription models in the [Models API](api_openai_models.md).
