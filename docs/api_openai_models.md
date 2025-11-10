# Models API

Explore available AI models across AWS Bedrock regions through an OpenAI-compatible interface.

## Why Use the Models API?

<div class="grid cards" markdown>

- :material-view-grid: __Complete Catalog__
  <br>Browse all available models across AWS Bedrock regions. Chat, embeddings, images, and specialized models.

- :material-sync: __Always Up-to-Date__
  <br>Dynamic model discovery provides access to new models as they become available in AWS Bedrock.

</div>

## Quick Start: Available Endpoints

| Endpoint | Method | What It Does | Powered By |
|----------|--------|--------------|------------|
| `/v1/models` | GET | List all available models | AWS Bedrock + AWS AI Services |
| `/v1/models/{model_id}` | GET | Get details for a specific model | AWS Bedrock + AWS AI Services |

## OpenAI-Compatible with AWS Bedrock Power

**Features:**

- **Multi-region aggregation**: Combines models from all configured AWS Bedrock regions
- **Comprehensive catalog**: Includes Bedrock foundation models plus specialized models (Transcribe, Polly, etc.)

### What's Different from OpenAI?

- **Detailed ownership**: `owned_by` field shows provider and region (e.g., `Amazon (AWS Bedrock us-east-1)`)
- **Model-specific capabilities**: Modalities and context windows vary by model—consult AWS documentation for specifics

## Try It Now

**List all available models:**

```bash
curl -X GET "$BASE/v1/models" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

**Get details for a specific model:**

```bash
curl -X GET "$BASE/v1/models/amazon.nova-micro-v1:0" \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

---

Browse foundation models for chat, embeddings, images, audio, and more.
