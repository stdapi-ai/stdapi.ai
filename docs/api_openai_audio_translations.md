# Speech to English API

Upload audio in any language and get English transcriptions for international content, customer support, or multilingual applications.

## Why Choose Speech to English?

<div class="grid cards" markdown>

- :material-earth-arrow-right: __Automatic Language Detection__
  <br>Upload audio in any language. AWS automatically detects and translates to English.

- :material-file-multiple: __Multiple Output Formats__
  <br>Choose from text, JSON, verbose JSON with timestamps, or translated subtitle files (SRT/VTT).

</div>

## Quick Start: Available Endpoint

| Endpoint                 | Method | What It Does                                     | Powered By                     |
|--------------------------|--------|--------------------------------------------------|--------------------------------|
| `/v1/audio/translations` | POST   | Transcribe any language and translate to English | AWS Transcribe + AWS Translate |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                 |               Status                | Notes                         |
|-------------------------|:-----------------------------------:|-------------------------------|
| **Input**               |                                     |                               |
| Audio file upload       | :material-check-circle:{ .success } | Multipart file upload         |
| Auto language detection | :material-check-circle:{ .success } | Automatic source detection    |
| **Output Formats**      |                                     |                               |
| `json`                  | :material-check-circle:{ .success } | Structured translation        |
| `text`                  | :material-check-circle:{ .success } | Plain English text            |
| `verbose_json`          |    :material-cog:{ .model-dep }     | With timestamps               |
| `srt`                   |    :material-cog:{ .model-dep }     | English subtitles with timing |
| `vtt`                   |    :material-cog:{ .model-dep }     | English WebVTT subtitles      |
| **Translation**         |                                     |                               |
| Translation to English  | :material-check-circle:{ .success } | Using AWS Translate           |
| **Advanced**            |                                     |                               |
| `prompt`                |    :material-cog:{ .model-dep }     | Not available                 |
| `temperature`           |    :material-cog:{ .model-dep }     | Not available                 |
| **Usage tracking**      |                                     |                               |
| Input audio duration    | :material-check-circle:{ .success } | Seconds (billing unit)        |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success } **Supported** — Fully compatible with OpenAI API
* :material-close-circle:{ .unsupported } **Unsupported** — Not available in this
  implementation

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
- Automatic transcription + translation pipeline in one request
- Multiple output formats: `text`, `json`, `verbose_json`, `srt`, `vtt`
- Automatic source language detection (zero configuration)
- **Smart Subtitle Translation** :material-translate:{ .highlight }: Preserves original
  timing using intelligent HTML span processing

!!! tip "OpenAI Model Compatibility"
    stdapi.ai includes a built-in model alias that maps the OpenAI model name to AWS Transcribe:

    - `whisper-1` → `amazon.transcribe`

    This alias enables seamless compatibility with OpenAI-based tools and applications without any configuration changes. You can also [customize or override this alias](operations_configuration.md#model-aliases) to suit your needs.

**Note:** The `prompt` and `temperature` parameters are not supported to ensure
consistent translation accuracy.

## Try It Now

**Translate foreign audio to English text:**

```bash
curl -X POST "$BASE/v1/audio/translations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F file=@spanish-interview.mp3 \
  -F model=amazon.transcribe \
  -F response_format=json
```

**Translate foreign audio to English subtitles:**

```bash
curl -OJ -X POST "$BASE/v1/audio/translations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F file=@spanish-interview.mp3 \
  -F model=amazon.transcribe \
  -F response_format=srt
```

---

**Ready to translate multilingual audio?** Explore available models in
the [Models API](api_openai_models.md).
