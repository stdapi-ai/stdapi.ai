---
title: Moderations API - Amazon Bedrock Guardrails & Amazon Comprehend Content Safety
description: Classify text and images with Amazon Bedrock Guardrails or Amazon Comprehend toxicity detection through an OpenAI-compatible Moderations API. Content safety, policy enforcement, and harm detection for AI applications.
keywords: OpenAI moderations API, content moderation AWS, Bedrock Guardrails API, Comprehend toxicity detection, content safety, harm detection, moderation categories, guardrail content filters
---

# Moderations API

Classify content for harm with [Amazon Bedrock Guardrails](https://aws.amazon.com/bedrock/guardrails/) or [Amazon Comprehend toxicity detection](https://docs.aws.amazon.com/comprehend/latest/dg/trust-safety.html) through an OpenAI-compatible Moderations interface.

## Why Choose the Moderations API?

<div class="grid cards" markdown>

- :material-shield-check: __Configurable Content Safety__
  <br>Bring your own Bedrock guardrail: categories, thresholds, denied topics, word filters, and sensitive-information policies are fully configurable in AWS.

- :material-flash: __Works Out of the Box__
  <br>Amazon Comprehend toxicity detection requires no setup at all, so `/v1/moderations` works immediately on any deployment.

- :material-swap-horizontal: __Drop-in OpenAI Compatibility__
  <br>OpenAI moderation model names are accepted as aliases. Existing integrations work by changing the base URL.

- :material-cloud-lock: __Private AWS Backend__
  <br>Classifications run entirely in your own AWS account — no traffic to third-party endpoints.

</div>

## Quick Start: Available Endpoint

| Endpoint          | Method | What It Does                                                | Powered By                                | MCP Tool            |
|-------------------|--------|--------------------------------------------------------------|-------------------------------------------|---------------------|
| `/v1/moderations` | `POST`   | Classify inputs with a guardrail or toxicity detection      | Amazon Bedrock Guardrails / Amazon Comprehend | `openai_moderation` |

**Example request:**

```bash
curl -X POST "$BASE/v1/moderations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Some text to classify"
  }'
```

**Example response:**

```json
{
  "id": "modr-0f1b3c6e8d9a4b5c",
  "model": "abcd1234efgh:1",
  "results": [
    {
      "flagged": true,
      "categories": {"hate": true, "harassment": false, "...": false},
      "category_scores": {"hate": 0.75, "harassment": 0.25, "...": 0.0},
      "category_applied_input_types": {"hate": ["text"], "harassment": ["text"], "...": ["text"]}
    }
  ]
}
```

## Feature Compatibility

<div class="feature-table" markdown>

| Feature                          |                 Status                  | Notes                                                                    |
|----------------------------------|:---------------------------------------:|---------------------------------------------------------------------------|
| **Input**                        |                                         |                                                                           |
| `input` string / array of strings |  :material-check-circle:{ .success role="img" aria-label="Supported" }   | Each element yields one independent result                                |
| `input` text parts               |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | `{"type": "text", "text": ...}`                                           |
| `input` image parts              |      :material-cog:{ .model-dep role="img" aria-label="Model-dependent" }       | Guardrail models only; PNG and JPEG                                       |
| Empty string input               |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Returns an unflagged result without calling AWS (OpenAI parity); whitespace-only strings are **not** covered by this shortcut and are classified normally (billed Comprehend call) |
| `model`                          |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Guardrail, Comprehend, or an OpenAI moderation model alias (see below)    |
| **Output**                       |                                         |                                                                           |
| `flagged`                        |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Also raised by guardrail policies without a mapped category, and on Comprehend by the overall toxicity score or unmapped labels such as profanity |
| `categories` / `category_scores` |   :material-minus-circle:{ .partial role="img" aria-label="Partial" }   | Mapped categories only; OpenAI categories without a counterpart stay `false` / `0.0` |
| `category_applied_input_types`   |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Reflects each classified element's modality                               |
| **Usage tracking**               |                                         |                                                                           |
| Guardrail text units / images    |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Billing units (one text unit per 1,000 characters per input; one unit per image) |
| Comprehend units                 |   :material-check-circle:{ .success role="img" aria-label="Supported" }   | Billing unit (100 characters, minimum 3 per call)                         |
| **Other**                        |                                         |                                                                           |
| Model discovery                  | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Moderation models and their aliases appear in the [model listings](api_search_models.md) |
| Long text inputs                 | :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } | Comprehend inputs of any length are split into API-sized segments transparently |

</div>

<div class="feature-table" markdown>

**Legend:**

* :material-check-circle:{ .success role="img" aria-label="Supported" } **Supported** — Fully compatible with OpenAI API
* :material-cog:{ .model-dep role="img" aria-label="Model-dependent" } **Model-Dependent** — Behavior depends on the model or backend; check the Notes column
* :material-minus-circle:{ .partial role="img" aria-label="Partial" } **Partial** — Supported with limitations
* :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } **Unsupported** — Not available in this implementation
* :material-plus-circle:{ .extra-feature role="img" aria-label="Extra feature" } **Extra Feature** — Enhanced capability beyond OpenAI API

</div>

## Model Support

Both moderation models appear in the [`/v1/models`](api_openai_models.md) and [`/search_models`](api_search_models.md) listings (`mcp_tool=openai_moderation`) with their OpenAI aliases.

### ![Amazon Bedrock](styles/logo_amazon_bedrock.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Bedrock Guardrails

| Model                              | OpenAI aliases                                       | Notes                                                                 |
|------------------------------------|------------------------------------------------------|------------------------------------------------------------------------|
| `amazon.bedrock-runtime-guardrail` | `omni-moderation-latest`, `omni-moderation-2024-09-26` | The server's default guardrail. Text and image inputs. Listed only when a guardrail is configured |
| `<guardrail-id>`, `<guardrail-id>:<version>`, or guardrail ARN | —                        | Any explicit guardrail (requires guardrail override to be allowed)     |

### ![Amazon Comprehend](styles/logo_amazon_comprehend.svg){ style="height: 1.2em; vertical-align: text-bottom;" } Amazon Comprehend

| Model                        | OpenAI aliases                                      | Notes                                        |
|------------------------------|-----------------------------------------------------|------------------------------------------------|
| `amazon.comprehend-toxicity` | `text-moderation-latest`, `text-moderation-stable`  | Toxicity detection. English text only, no images |

### Comparison

| Capability                | Amazon Bedrock Guardrails                                                  | Amazon Comprehend                                     |
|---------------------------|-----------------------------------------------------------------------------|--------------------------------------------------------|
| Setup                     | Create and configure a guardrail in Amazon Bedrock                         | None — works out of the box                            |
| Text inputs               | :material-check-circle:{ .success role="img" aria-label="Supported" } Any language supported by the guardrail | :material-check-circle:{ .success role="img" aria-label="Supported" } English only       |
| Image inputs              | :material-check-circle:{ .success role="img" aria-label="Supported" } PNG and JPEG                           | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } Not supported  |
| Mapped categories         | `hate`, `harassment`, `sexual`, `violence`, `illicit`                       | `hate`, `harassment`, `sexual`, `violence`, `violence/graphic` |
| Category scores           | Quantized confidence levels (`0.0` / `0.25` / `0.5` / `0.75`)               | Continuous scores (`0.0` – `1.0`)                       |
| Custom policies           | Denied topics, word filters, PII, prompt attacks, contextual grounding      | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } Fixed toxicity labels |
| Tunable thresholds        | Per-filter strengths configured on the guardrail                            | Fixed flagging threshold (score ≥ 0.5)                 |
| `moderation` request parameter | :material-check-circle:{ .success role="img" aria-label="Supported" } Applied to generations natively    | :material-close-circle:{ .unsupported role="img" aria-label="Unsupported" } Moderations API only |
| Input length              | ApplyGuardrail text unit limits                                             | Unlimited (split into 1 KB segments transparently)     |

## Selecting the Model

The `model` parameter selects the moderation model:

| `model` value                                | Model used                                                                                      |
|----------------------------------------------|--------------------------------------------------------------------------------------------------|
| Omitted                                      | The server's default guardrail, or Comprehend toxicity detection when none is configured         |
| `amazon.bedrock-runtime-guardrail`           | The server's default guardrail (an error when none is configured)                                |
| `omni-moderation-*`                          | Same as `amazon.bedrock-runtime-guardrail`, falling back to Comprehend when no guardrail is configured |
| `amazon.comprehend-toxicity` / `text-moderation-*` | Comprehend toxicity detection, even when a guardrail is configured                          |
| `<guardrail-id>` or `<guardrail-id>:<version>` | That guardrail (requires guardrail override to be allowed)                                     |
| Guardrail ARN                                | That guardrail, applied in the region embedded in the ARN                                        |

The server guardrail comes from [`AWS_BEDROCK_GUARDRAIL_IDENTIFIER` / `AWS_BEDROCK_GUARDRAIL_VERSION`](operations_configuration.md#bedrock-guardrails), or from the `X-Amzn-Bedrock-GuardrailIdentifier` / `X-Amzn-Bedrock-GuardrailVersion` request headers when [`AWS_BEDROCK_ALLOW_GUARDRAIL_OVERRIDE`](operations_configuration.md#bedrock-guardrails) is enabled. Explicit guardrails in `model` also require that setting.

Guardrails are regional: a plain guardrail ID is applied in the primary Bedrock region, while an ARN selects its own region. Comprehend calls use [`AWS_COMPREHEND_REGION`](operations_configuration.md#aws-comprehend-region) (with multi-region failover otherwise).

!!! tip "Moderating generations directly"
    The guardrail selection and category mapping also power the `moderation` request parameter of the [Chat Completions](api_openai_chat_completions.md) and [Responses](api_openai_responses.md) APIs: the guardrail is applied to the generation itself, and the classification of the input and output is reported in the response's `moderation` field — for Chat Completions on non-streaming requests only, and for Responses also on the terminal event when streaming. The `moderation` parameter requires a guardrail — Comprehend is not available there — and is rejected (`400`) on Amazon Bedrock Mantle-served models.

## Category Mapping

**Amazon Bedrock Guardrails** — content policy filters map to the OpenAI moderation categories:

| Amazon Bedrock filter | OpenAI category |
|--------------------|-----------------|
| `HATE`             | `hate`          |
| `INSULTS`          | `harassment`    |
| `SEXUAL`           | `sexual`        |
| `VIOLENCE`         | `violence`      |
| `MISCONDUCT`       | `illicit`       |

Filter confidence levels become scores: `NONE` → `0.0`, `LOW` → `0.25`, `MEDIUM` → `0.5`, `HIGH` → `0.75`.

Every other guardrail policy — denied topics, word filters, sensitive information (PII), prompt attacks, contextual grounding — still contributes to the top-level `flagged` field whenever the guardrail intervenes, even though no individual category is set.

**Amazon Comprehend** — toxicity labels map to the OpenAI moderation categories, with their detection scores (`0.0`–`1.0`) reported directly:

| Comprehend label      | OpenAI category    |
|-----------------------|--------------------|
| `HATE_SPEECH`         | `hate`             |
| `HARASSMENT_OR_ABUSE` | `harassment`       |
| `INSULT`              | `harassment`       |
| `SEXUAL`              | `sexual`           |
| `VIOLENCE_OR_THREAT`  | `violence`         |
| `GRAPHIC`             | `violence/graphic` |
| `PROFANITY`           | *(`flagged` only)* |

An input is flagged when its overall toxicity or any label score reaches `0.5`. Long texts are split into 1 KB segments and the highest score per category is kept.

With either model, OpenAI sub-categories without a counterpart (e.g. `self-harm`, `sexual/minors`) are always `false`.

## Inputs

Each input element is classified independently and yields one entry in `results`:

- **`input` as a string** — one text classification.
- **`input` as an array of strings** — one classification per string.
- **`input` as an array of parts** — `{"type": "text", "text": ...}` and `{"type": "image_url", "image_url": {"url": ...}}` parts. Images must be PNG or JPEG, and require a guardrail model.

Each result's `category_applied_input_types` reflects the classified element's modality: `["text"]` for every category on text inputs; on image inputs, `["image"]` for the categories that support images and `[]` for the text-only ones.

An `input` array holds at most 2048 elements. Each element is classified — and billed — independently by a separate AWS call, so large arrays incur a proportional number of AWS calls.

**MCP / AI agent usage:** `image_url.url` accepts an HTTPS URL, data URI (`data:<mime>;base64,<data>`), base64 string, or S3 URI — no binary upload needed.

```bash
curl -X POST "$BASE/v1/moderations" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      {"type": "text", "text": "Describe this image"},
      {"type": "image_url", "image_url": {"url": "https://example.com/photo.png"}}
    ]
  }'
```

## Billing

- **Guardrails** — AWS bills per text unit and per image processed by the ApplyGuardrail API; see [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/). No Bedrock model invocation is involved.
- **Comprehend** — AWS bills toxicity detection per 100-character unit with a 3-unit minimum per call; see [Amazon Comprehend pricing](https://aws.amazon.com/comprehend/pricing/). Billed units appear in [usage logs and cost tracking](operations_logging_monitoring.md) as `comprehend_units` under the `amazon.comprehend-toxicity` model.
