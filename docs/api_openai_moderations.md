---
title: Moderations API - AWS Bedrock Guardrails Content Safety (OpenAI Compatible)
description: Classify text and images with AWS Bedrock Guardrails through an OpenAI-compatible Moderations API. Content safety, policy enforcement, and harm detection for AI applications.
keywords: OpenAI moderations API, content moderation AWS, Bedrock Guardrails API, content safety, harm detection, moderation categories, guardrail content filters
---

# Moderations API (OpenAI Compatible)

Classify text and images for harmful content with [AWS Bedrock Guardrails](https://aws.amazon.com/bedrock/guardrails/) through an OpenAI-compatible Moderations interface.

Instead of OpenAI's fixed moderation models, each deployment brings its own guardrail: the categories, thresholds, denied topics, word filters, and sensitive-information policies are fully configurable in AWS.

## Quick Start: Available Endpoint

| Endpoint          | Method | What It Does                                    | Powered By             | MCP Tool            |
|-------------------|--------|--------------------------------------------------|------------------------|---------------------|
| `/v1/moderations` | POST   | Classify text and image inputs with a guardrail | AWS Bedrock Guardrails | `openai_moderation` |

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
      "category_scores": {"hate": 0.75, "harassment": 0.25, "...": 0.0}
    }
  ]
}
```

## Selecting the Guardrail

The `model` parameter selects the AWS Bedrock guardrail to apply:

| `model` value                                | Guardrail used                                                                    |
|----------------------------------------------|-----------------------------------------------------------------------------------|
| Omitted                                      | The server's configured guardrail                                                 |
| `omni-moderation-*` / `text-moderation-*`    | The server's configured guardrail (drop-in compatibility with OpenAI model names) |
| `<guardrail-id>` or `<guardrail-id>:<version>` | That guardrail (requires guardrail override to be allowed)                       |
| Guardrail ARN                                | That guardrail, applied in the region embedded in the ARN                         |

The server guardrail comes from [`AWS_BEDROCK_GUARDRAIL_IDENTIFIER` / `AWS_BEDROCK_GUARDRAIL_VERSION`](operations_configuration.md#bedrock-guardrails), or from the `X-Amzn-Bedrock-GuardrailIdentifier` / `X-Amzn-Bedrock-GuardrailVersion` request headers when [`AWS_BEDROCK_ALLOW_GUARDRAIL_OVERRIDE`](operations_configuration.md#bedrock-guardrails) is enabled. Explicit guardrails in `model` also require that setting.

Guardrails are regional: a plain guardrail ID is applied in the primary Bedrock region, while an ARN selects its own region.

!!! tip "Moderating generations directly"
    The same guardrail selection and category mapping power the `moderation` request parameter of the [Chat Completions](api_openai_chat_completions.md) and [Responses](api_openai_responses.md) APIs: the guardrail is applied to the generation itself, and the classification of the input and output is reported in the response's `moderation` field (non-streaming requests).

## Category Mapping

Guardrail **content policy filters** map to the OpenAI moderation categories:

| AWS Bedrock filter | OpenAI category |
|--------------------|-----------------|
| `HATE`             | `hate`          |
| `INSULTS`          | `harassment`    |
| `SEXUAL`           | `sexual`        |
| `VIOLENCE`         | `violence`      |
| `MISCONDUCT`       | `illicit`       |

Filter confidence levels become scores: `NONE` → `0.0`, `LOW` → `0.25`, `MEDIUM` → `0.5`, `HIGH` → `0.75`. OpenAI sub-categories without a guardrail counterpart (e.g. `self-harm`, `sexual/minors`) are always `false`.

Every other guardrail policy — denied topics, word filters, sensitive information (PII), prompt attacks, contextual grounding — still contributes to the top-level `flagged` field whenever the guardrail intervenes, even though no individual category is set.

## Inputs

Each input element is classified independently and yields one entry in `results`:

- **`input` as a string** — one text classification.
- **`input` as an array of strings** — one classification per string.
- **`input` as an array of parts** — `{"type": "text", "text": ...}` and `{"type": "image_url", "image_url": {"url": ...}}` parts. Images must be PNG or JPEG.

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

AWS bills guardrail usage directly per text unit and per image processed by the ApplyGuardrail API; see [AWS Bedrock pricing](https://aws.amazon.com/bedrock/pricing/). No Bedrock model invocation is involved.
