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

| Feature                 |                 Status                  | Notes                         |
|-------------------------|:---------------------------------------:|-------------------------------|
| **Input**               |                                         |                               |
| Audio file upload       |   :material-check-circle:{ .success }   | Multipart file upload         |
| Auto language detection |   :material-check-circle:{ .success }   | Automatic source detection    |
| **Output Formats**      |                                         |                               |
| `json`                  |   :material-check-circle:{ .success }   | Structured translation        |
| `text`                  |   :material-check-circle:{ .success }   | Plain English text            |
| `verbose_json`          |   :material-check-circle:{ .success }   | With timestamps               |
| `srt`                   |   :material-check-circle:{ .success }   | English subtitles with timing |
| `vtt`                   |   :material-check-circle:{ .success }   | English WebVTT subtitles      |
| **Translation**         |                                         |                               |
| Translation to English  |   :material-check-circle:{ .success }   | Using AWS Translate           |
| **Advanced**            |                                         |                               |
| `prompt`                | :material-close-circle:{ .unsupported } | Not available                 |
| `temperature`           | :material-close-circle:{ .unsupported } | Not available                 |
| **Usage tracking**      |                                         |                               |
| Input audio duration    |   :material-check-circle:{ .success }   | Seconds (billing unit)        |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success } **Supported** — Fully compatible with OpenAI API
* :material-close-circle:{ .unsupported } **Unsupported** — Not available in this
  implementation

</div>

## Advanced Features

### ![AWS Translate](styles/logo_amazon_translate.svg){ style="height: 1.2em; vertical-align: text-bottom;" } OpenAI-Compatible with AWS Power

**Model & Features:**

- Use `amazon.transcribe` (instead of `whisper-1`) with the same interface
- Automatic transcription + translation pipeline in one request
- Multiple output formats: `text`, `json`, `verbose_json`, `srt`, `vtt`
- Automatic source language detection (zero configuration)
- **Smart Subtitle Translation** :material-translate:{ .highlight }: Preserves original
  timing using intelligent HTML span processing

**Note:** The `prompt` and `temperature` parameters are not supported to ensure
consistent translation accuracy.

!!! warning "Configuration Required"
You must configure the `AWS_S3_BUCKET` or `AWS_TRANSCRIBE_S3_BUCKET` environment
variable with a bucket in the main AWS region to use this endpoint. This bucket is used
for temporary storage during transcription processing.

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
