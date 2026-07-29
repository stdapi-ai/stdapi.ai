---
title: Text to Speech API - Amazon Polly with OpenAI Compatibility
description: Generate natural-sounding speech from text using Amazon Polly. OpenAI-compatible TTS API with 60+ voices, 30+ languages, and SSML support.
keywords: text to speech API, TTS API AWS, AWS Polly API, voice synthesis API, OpenAI TTS, neural voices, speech generation, audio synthesis
---

# Text to Speech API

Generate natural-sounding speech from text with Amazon Polly through an OpenAI-compatible interface.

## Why Choose the Text to Speech API?

<div class="grid cards" markdown>

- :material-earth: __Global Support__
  <br>30+ languages supported. Choose from Neural, Generative, and Long-Form engines.

- :material-account-voice: __60+ Voices__
  <br>Professional narration to conversational voices. Use OpenAI voice names with automatic language detection or specify any Polly voice ID directly.

- :material-auto-fix: __Automatic Language Detection__
  <br>Using OpenAI voice names? Amazon Comprehend automatically detects your content's language and selects an appropriate Polly voice—matching language, gender, and quality.

- :material-xml: __Advanced Control with SSML__
  <br>Fine-tune pronunciation, emphasis, pauses, and prosody with SSML markup for complex audio requirements.

</div>

## Quick Start: Available Endpoint

| Endpoint            | Method | What It Does                           | Powered By                       | MCP Tool           |
|---------------------|--------|----------------------------------------|----------------------------------|--------------------|
| `/v1/audio/speech`  | `POST` | Turn text into natural-sounding speech | Amazon Polly + Amazon Comprehend | `openai_audio_speech` |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                     |                  Status                  | Notes                                                           |
|-----------------------------|:----------------------------------------:|-----------------------------------------------------------------|
| **Voice Selection**         |                                          |                                                                 |
| OpenAI voice names          |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Mapped to Polly voices                                          |
| Polly voice IDs             | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | 60+ voices across 30+ languages                                 |
| Dynamic voice selection     | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Select best Polly voice based on the detected language          |
| **Input**                   |                                          |                                                                 |
| Plain text                  |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Standard text input                                             |
| SSML markup                 | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Fine-grained speech control                                     |
| **Output Formats**          |                                          |                                                                 |
| MP3                         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Native Polly format                                             |
| PCM                         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Native Polly format                                             |
| Opus                        |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Native Polly format                                             |
| AAC                         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Encoded from PCM                                                |
| FLAC                        |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Encoded from PCM                                                |
| WAV                         |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Encoded from PCM                                                |
| OGG (Vorbis)                | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Native Polly format                                             |
| **Control**                 |                                          |                                                                 |
| `speed` parameter           |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | 0.2x to 2.0x playback speed; rejected with SSML input (set the speed in SSML instead) |
| `instructions` parameter    | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Accepted for OpenAI API compatibility and ignored (no Amazon Polly equivalent) |
| Extra model-specific params | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Extra model-specific parameters via JSON body                   |
| **Streaming**               |                                          |                                                                 |
| Byte streaming              |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Default streaming mode                                          |
| SSE streaming               |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Event-based streaming                                           |
| **Usage tracking**          |                                          |                                                                 |
| Input text tokens           |   :material-check-circle:{ .success role="img" aria-label="Supported" }    | Characters count (billing unit)                                 |
| Output tokens               | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" }  | Not available                                                   |
</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with OpenAI API
* :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } **Extra Feature** — Enhanced capability beyond OpenAI API
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation

</div>

## Model Support

### ![Amazon Polly](styles/logo_amazon_polly.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Polly Models

| Model                     | Polly Engine | Notes                                              |
|---------------------------|--------------|----------------------------------------------------|
| `amazon.polly-standard`   | Standard     | Lowest cost, widest language coverage              |
| `amazon.polly-neural`     | Neural       | Higher-quality, natural-sounding voices            |
| `amazon.polly-long-form`  | Long-form    | Expressive voices for narration-length content     |
| `amazon.polly-generative` | Generative   | Most human-like, conversational voices             |

Each engine supports a different subset of voices and languages — see the [Polly voice list](https://docs.aws.amazon.com/polly/latest/dg/voicelist.html) for details. OpenAI voice names work with every model through automatic language detection and voice selection, or specify any Polly voice ID directly for 60+ voices across 30+ languages.

!!! tip "OpenAI Model Compatibility"
    stdapi.ai includes built-in model aliases that map OpenAI model names to Amazon Polly engines:

    - `tts-1` → `amazon.polly-standard`
    - `tts-1-hd` → `amazon.polly-neural`

    These aliases enable seamless compatibility with OpenAI-based tools and applications without any configuration changes. You can also [customize or override these aliases](operations_configuration.md#model-aliases) to suit your needs.

## Advanced Features

### ![Amazon Polly](styles/logo_amazon_polly.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Polly Features

- **SSML Support** :material-star-circle:{ .highlight }: Fine-grained control over pronunciation, emphasis, pauses, and prosody — [SSML docs](https://docs.aws.amazon.com/polly/latest/dg/ssml.html). With SSML input, the `speed` parameter is rejected: set the speaking rate with SSML `<prosody>` instead.
- **Flexible Formats**: mp3, ogg, wav, flac, aac, opus, pcm — non-native formats are transcoded server-side
- **Streaming Options**: Raw bytes (default) or SSE events with `stream_format: "sse"`
- **Speed Control**: Adjust playback from 0.2x to 2.0x
- **Speech Marks**: Word, sentence, viseme, and SSML timing metadata with `SpeechMarkTypes` (returned as JSON instead of audio)
- **Character-Based Billing**: Usage tracks character counts—the native billing unit for Amazon Polly and Amazon Comprehend—rather than OpenAI-style tokens

!!! tip "Performance Tips: Optimize Speed & Cost"
    - **Use native Polly formats** (mp3, ogg, PCM) to skip server-side conversion
    - **Specify a Polly voice ID** to bypass language detection—faster responses, no Amazon Comprehend charges
    - **Configure a default language** via `DEFAULT_TTS_LANGUAGE` environment variable to skip language detection for all requests using OpenAI voice names

!!! info "Language Detection Behavior"
    When using OpenAI voice names without specifying a default language, the system analyzes only the first 500 characters of your text to detect the language. This approach:

    - **Works best** with long, single-language texts where the first 500 characters are representative
    - **May be inconsistent** with very short texts (< 100 characters) where language detection has limited context
    - **Can produce mixed results** with multi-language content where different parts use different languages

    **For consistent behavior across requests**, consider:

    - Setting `DEFAULT_TTS_LANGUAGE` for applications serving primarily one language
    - Using Polly voice IDs directly when you know the target language
    - Structuring multi-language applications to make separate API calls per language

!!! tip "Default Streaming Mode: API vs MCP"
    - **API usage**: Default is **byte streaming** (raw audio data)
    - **MCP tool usage**: Default is **SSE streaming** (`stream_format: "sse"`)

    When used as an MCP tool, the response defaults to SSE events (`speech.audio.delta`, `speech.audio.done`) for better client compatibility. Override by explicitly setting `stream_format: "audio"` in your request.

### Provider-Specific Parameters

Unlock advanced Amazon Polly capabilities by passing provider-specific parameters directly in your requests. These parameters are forwarded to Polly's `SynthesizeSpeech` API and allow you to access features unique to Polly.

**How It Works:**

Add provider-specific fields at the top level of your request body alongside standard OpenAI parameters. The API automatically forwards these to Amazon Polly.

**Examples:**

**Lexicon Support:**

Apply custom pronunciation lexicons to your speech synthesis:

```json
{
  "model": "amazon.polly-neural",
  "voice": "Joanna",
  "input": "Amazon Polly uses lexicons for custom pronunciation.",
  "response_format": "mp3",
  "LexiconNames": ["MyCustomLexicon"]
}
```

**Sample Rate:**

Specify custom audio sample rate (8000, 16000, 22050, or 24000 Hz; PCM output supports 8000 and 16000 only):

```json
{
  "model": "amazon.polly-neural",
  "voice": "Matthew",
  "input": "High quality audio at 24kHz.",
  "response_format": "mp3",
  "SampleRate": "24000"
}
```

**Language Code:**

Specify the language for bilingual voices (only useful for voices that support multiple languages):

```json
{
  "model": "amazon.polly-neural",
  "voice": "Aditi",
  "input": "Hello, how are you?",
  "response_format": "mp3",
  "LanguageCode": "en-IN"
}
```

**Speech Marks:**

Request word, sentence, viseme, or SSML timing marks instead of audio (useful for lip-sync, karaoke-style highlighting, or subtitle alignment):

```json
{
  "model": "amazon.polly-neural",
  "voice": "Joanna",
  "input": "Hello, how are you?",
  "SpeechMarkTypes": ["word", "sentence"]
}
```

!!! warning "Speech marks return JSON, not audio"
    When `SpeechMarkTypes` is set, Polly returns timing metadata only. The response is a stream of JSON objects (one per line) with the `application/x-json-stream` content type:

    - `response_format` is ignored — no audio is generated or transcoded.
    - `stream_format: "sse"` is rejected with HTTP 400, since the payload is not audio events.
    - The `ssml` mark type requires SSML input (`<speak>…</speak>`); requesting it with plain text returns HTTP 400.

    ```json
    {"time":0,"type":"word","start":0,"end":5,"value":"Hello"}
    {"time":576,"type":"word","start":7,"end":10,"value":"how"}
    ```

**Configuration Options:**

**Option 1: Per-Request**

Add provider-specific parameters directly in your request body (as shown in examples above).

**Option 2: Server-Wide Defaults**

Configure default parameters for specific models via the `DEFAULT_MODEL_PARAMS` environment variable:

```bash
export DEFAULT_MODEL_PARAMS='{
  "amazon.polly-neural": {
    "SampleRate": "24000"
  }
}'
```

**Note:** Per-request parameters override server-wide defaults.

**Behavior:**

**Compatible parameters** are forwarded to Polly and applied; **unsupported parameters** return HTTP 400 with an error message.

**Available Parameters:**

The following parameters from the Amazon Polly [SynthesizeSpeech API](https://docs.aws.amazon.com/polly/latest/dg/API_SynthesizeSpeech.html) can be used:

- `LexiconNames` (list): Apply pronunciation lexicons
- `SampleRate` (string): Audio sample rate in Hz — `8000`, `16000`, `22050`, or `24000` (`pcm` output: `8000` or `16000`)
- `LanguageCode` (string): Language code for bilingual voices only (e.g., `en-IN`, `hi-IN`)
- `SpeechMarkTypes` (list): Timing marks to return instead of audio — `sentence`, `ssml`, `viseme`, `word`

## Try It Now

**Stream audio as bytes (default):**

```bash
curl -OJ -X POST "$BASE/v1/audio/speech" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.polly-neural",
    "voice": "Amy",
    "input": "Welcome to the future of voice technology!",
    "response_format": "mp3"
  }'
```

**Stream audio as SSE events:**

```bash
curl -N -X POST "$BASE/v1/audio/speech" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "amazon.polly-neural",
    "voice": "Amy",
    "input": "This audio streams as SSE events!",
    "response_format": "mp3",
    "stream_format": "sse"
  }'
```

---

**Ready to add voice to your application?** Explore available voices and models in the [Models API](api_openai_models.md).
