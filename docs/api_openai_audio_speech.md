# Text to Speech API

Generate audio from text for voiceovers, audiobooks, accessibility features, or interactive voice experiences.

## Why Choose Text to Speech?

<div class="grid cards" markdown>

- :material-earth: __Global Support__
  <br>30+ languages supported. Choose from Neural, Generative, and Long-Form engines.

- :material-account-voice: __60+ Voices__
  <br>Professional narration to conversational voices. Use OpenAI voice names with automatic language detection or specify any Polly voice ID directly.

- :material-auto-fix: __Automatic Language Detection__
  <br>Using OpenAI voice names? AWS Comprehend automatically detects your content's language and selects an appropriate Polly voice—matching language, gender, and quality.

- :material-xml: __Advanced Control with SSML__
  <br>Fine-tune pronunciation, emphasis, pauses, and prosody with SSML markup for complex audio requirements.

</div>

## Quick Start: Available Endpoint

| Endpoint            | Method | What It Does                           | Powered By                 |
|---------------------|--------|----------------------------------------|----------------------------|
| `/v1/audio/speech`  | POST   | Turn text into natural-sounding speech | AWS Polly + AWS Comprehend |

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                     |                  Status                  | Notes                                                           |
|-----------------------------|:----------------------------------------:|-----------------------------------------------------------------|
| **Voice Selection**         |                                          |                                                                 |
| OpenAI voice names          |   :material-check-circle:{ .success }    | Mapped to Polly voices                                          |
| Polly voice IDs             | :material-plus-circle:{ .extra-feature } | 60+ voices across 30+ languages                                 |
| Dynamic voice selection     | :material-plus-circle:{ .extra-feature } | Select best Polly voice based on the detected language          |
| **Input**                   |                                          |                                                                 |
| Plain text                  |   :material-check-circle:{ .success }    | Standard text input                                             |
| SSML markup                 | :material-plus-circle:{ .extra-feature } | Fine-grained speech control                                     |
| **Output Formats**          |                                          |                                                                 |
| MP3                         |   :material-check-circle:{ .success }    | Native Polly format                                             |
| PCM                         |   :material-check-circle:{ .success }    | Native Polly format                                             |
| Opus                        |   :material-check-circle:{ .success }    | Native Polly format                                             |
| AAC                         |   :material-check-circle:{ .success }    | Encoded from PCM                                                |
| FLAC                        |   :material-check-circle:{ .success }    | Encoded from PCM                                                |
| WAV                         |   :material-check-circle:{ .success }    | Encoded from PCM                                                |
| OGG (Vorbis)                | :material-plus-circle:{ .extra-feature } | Native Polly format                                             |
| **Control**                 |                                          |                                                                 |
| `speed` parameter           |   :material-check-circle:{ .success }    | 0.2x to 2.0x playback speed                                     |
| Extra model-specific params | :material-plus-circle:{ .extra-feature } | Extra model-specific parameters not supported by the OpenAI API |
| **Streaming**               |                                          |                                                                 |
| Byte streaming              |   :material-check-circle:{ .success }    | Default streaming mode                                          |
| SSE streaming               |   :material-check-circle:{ .success }    | Event-based streaming                                           |
| **Usage tracking**          |                                          |                                                                 |
| Input text tokens           |   :material-check-circle:{ .success }    | Characters count (billing unit)                                 |
| Output tokens               | :material-close-circle:{ .unsupported }  | Not available                                                   |
</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success } **Supported** — Fully compatible with OpenAI API
* :material-plus-circle:{ .extra-feature } **Extra Feature** — Enhanced capability beyond OpenAI API
* :material-close-circle:{ .unsupported } **Unsupported** — Not available in this implementation

</div>

## Advanced Features

### ![AWS Polly](styles/logo_amazon_polly.svg){ style="height: 1.2em; vertical-align: text-bottom;" } OpenAI-Compatible with AWS Power

**Models & Voices:**

- Use `amazon.polly-standard`, `amazon.polly-neural`, `amazon.polly-long-form`, or `amazon.polly-generative` (instead of `tts-1`/`tts-1-hd`)
- OpenAI voice names work with automatic language detection and intelligent voice selection
- Or specify any [Polly voice ID](https://docs.aws.amazon.com/polly/latest/dg/voicelist.html) directly for 60+ voices across 30+ languages

**Enhanced Features:**

- **SSML Support** :material-star-circle:{ .highlight }: Fine-grained control over pronunciation, emphasis, pauses, and prosody — [SSML docs](https://docs.aws.amazon.com/polly/latest/dg/ssml.html)
- **Flexible Formats**: mp3, ogg, wav, flac, aac, opus (some transcoded server-side via ffmpeg)
- **Streaming Options**: Raw bytes (default) or SSE events with `stream_format: "sse"`
- **Speed Control**: Adjust playback from 0.25x to 4.0x
- **Character-Based Billing**: Usage tracks character counts—the native billing unit for AWS Polly and AWS Comprehend—rather than OpenAI-style tokens

!!! tip "Performance Tips: Optimize Speed & Cost"
    - **Use native Polly formats** (mp3, ogg, PCM) to skip server-side conversion
    - **Specify a Polly voice ID** to bypass language detection—faster responses, no AWS Comprehend charges

### Provider-Specific Parameters

Unlock advanced AWS Polly capabilities by passing provider-specific parameters directly in your requests. These parameters are forwarded to AWS Polly's `synthesize_speech` API and allow you to access features unique to Polly.

**How It Works:**

Add provider-specific fields at the top level of your request body alongside standard OpenAI parameters. The API automatically forwards these to AWS Polly.

**Examples:**

**Lexicon Support:**

Apply custom pronunciation lexicons to your speech synthesis:

```json
{
  "model": "amazon.polly-neural",
  "voice": "Joanna",
  "input": "AWS Polly uses lexicons for custom pronunciation.",
  "response_format": "mp3",
  "LexiconNames": ["MyCustomLexicon"]
}
```

**Sample Rate:**

Specify custom audio sample rate (8000, 16000, 22050, 24000, 44100, or 48000 Hz):

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

- ✅ **Compatible parameters**: Forwarded to Polly and applied
- ⚠️ **Unsupported parameters**: Return HTTP 400 with an error message

**Available Parameters:**

The following parameters from the AWS Polly [SynthesizeSpeech API](https://docs.aws.amazon.com/polly/latest/dg/API_SynthesizeSpeech.html) can be used:

- `LexiconNames` (list): Apply pronunciation lexicons
- `SampleRate` (string): Audio sample rate in Hz
- `LanguageCode` (string): Language code for bilingual voices only (e.g., `en-IN`, `hi-IN`)

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
