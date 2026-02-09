# Speech to Text API

Convert spoken words into text. For voice assistants, meeting transcription, or accessibility features using AWS Transcribe.

## Why Choose Speech to Text?

<div class="grid cards" markdown>

- :material-translate: __100+ Languages__
  <br>Transcribe audio in any language with automatic detection or manual specification for global applications and multilingual content.

- :material-clock-fast: __Real-Time or Batch__
  <br>Stream transcriptions in real-time via SSE or process files efficiently.

- :material-subtitles: __Subtitle Generation__
  <br>Generate SRT and VTT subtitle files directly from AWS Transcribe with precise timing.

- :material-target: __Word-Level Timestamps__
  <br>Get word-level or segment-level timestamps with verbose_json for video editing, searchable transcripts, and accessibility features.

- :material-account-multiple: __Speaker Diarization__
  <br>Identify and separate different speakers in conversations with automatic speaker labeling using diarized_json format.

</div>

## Quick Start: Available Endpoint

| Endpoint                    | Method | What It Does                             | Powered By     |
|-----------------------------|--------|------------------------------------------|----------------|
| `/v1/audio/transcriptions`  | POST   | Convert spoken audio to written text     | AWS Transcribe |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                    |                 Status                  | Notes                               |
|----------------------------|:---------------------------------------:|-------------------------------------|
| **Input**                  |                                         |                                     |
| Audio file upload          |   :material-check-circle:{ .success }   | Multipart file upload               |
| **Output Formats**         |                                         |                                     |
| `json`                     |   :material-check-circle:{ .success }   | Structured transcription            |
| `text`                     |   :material-check-circle:{ .success }   | Plain text output                   |
| `verbose_json`             |   :material-check-circle:{ .success }   | With timestamps and details         |
| `diarized_json`            |   :material-check-circle:{ .success }   | With speaker identification         |
| `srt`                      |   :material-check-circle:{ .success }   | Subtitle format with timing         |
| `vtt`                      |   :material-check-circle:{ .success }   | WebVTT subtitle format              |
| **Language**               |                                         |                                     |
| Language specification     |   :material-check-circle:{ .success }   | ISO-639-1 language codes            |
| Auto language detection    |   :material-check-circle:{ .success }   | Automatic identification            |
| **Streaming**              |                                         |                                     |
| SSE streaming              |   :material-check-circle:{ .success }   | Event-based streaming               |
| **Advanced**               |                                         |                                     |
| Timestamp granularity      |   :material-check-circle:{ .success }   | Word or segment level               |
| Speaker diarization        |   :material-check-circle:{ .success }   | Automatic speaker separation        |
| `known_speaker_names`      | :material-close-circle:{ .unsupported } | Not available                       |
| `known_speaker_references` | :material-close-circle:{ .unsupported } | Not available                       |
| `chunking_strategy`        |   :material-minus-circle:{ .partial }   | Only `auto` is supported            |
| `temperature`              | :material-close-circle:{ .unsupported } | Not available                       |
| `prompt`                   | :material-close-circle:{ .unsupported } | Not available                       |
| `logprobs`                 | :material-close-circle:{ .unsupported } | Not available                       |
| **Usage tracking**         |                                         |                                     |
| Input audio duration       |   :material-check-circle:{ .success }   | Seconds (billing unit)              |
| Output text tokens         |   :material-minus-circle:{ .partial }   | Estimated token count for reference |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success } **Supported** — Fully compatible with OpenAI API
* :material-minus-circle:{ .partial } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported } **Unsupported** — Not available in this implementation

</div>

## Advanced Features

### ![AWS Transcribe](styles/logo_amazon_transcribe.svg){ style="height: 1.2em; vertical-align: text-bottom;" } OpenAI-Compatible with AWS Power

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

!!! warning "Configuration Required"
    You must configure the `AWS_S3_BUCKET` or `AWS_TRANSCRIBE_S3_BUCKET` environment variable with a bucket in the main AWS region to use this endpoint. This bucket is used for temporary storage during transcription processing.

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
