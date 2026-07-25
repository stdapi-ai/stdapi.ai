"""Models Deprecation.

Mapping of deprecated (or pending deprecation) model IDs, with recommended alternatives.
This allows improving error messages for invalid models.

Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html
"""

from stdapi.config import SETTINGS

DEPRECATED_MODELS: dict[str, str] = {
    # AI21
    "ai21.j2-mid-v1": "ai21.jamba-instruct-v1:0",
    "ai21.j2-ultra-v1": "ai21.jamba-instruct-v1:0",
    "ai21.j2-grande-instruct": "ai21.jamba-instruct-v1:0",
    "ai21.j2-jumbo-instruct": "ai21.jamba-instruct-v1:0",
    "ai21.jamba-instruct-v1:0": "ai21.jamba-1-5-large-v1:0",
    # Amazon Nova
    "amazon.nova-premier-v1:0": "amazon.nova-2-lite-v1:0",
    "amazon.nova-sonic-v1:0": "amazon.nova-2-sonic-v1:0",
    "amazon.nova-reel-v1:0": "amazon.nova-reel-v1:1",
    # amazon.nova-reel-v1:1: EOL Sept 30, 2026 - no replacement confirmed
    # amazon.titan-tg1-large: EOL Oct 27, 2025 - no replacement confirmed
    # amazon.nova-canvas-v1:0: EOL Sept 30, 2026 - no replacement confirmed
    # Amazon Titan Image
    "amazon.titan-image-generator-v1": "amazon.nova-canvas-v1:0",
    "amazon.titan-image-generator-v2:0": "amazon.nova-canvas-v1:0",
    # Amazon Titan Text
    "amazon.titan-text-express-v1": "amazon.nova-micro-v1:0",
    "amazon.titan-text-lite-v1": "amazon.nova-lite-v1:0",
    "amazon.titan-text-premier-v1:0": "amazon.nova-pro-v1:0",
    # Amazon Titan Embed
    "amazon.titan-embed-text-v1": "amazon.titan-embed-text-v2:0",
    # Anthropic Claude (EOL)
    "anthropic.claude-v2": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-v2:1": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-instant-v1": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    # Anthropic Claude (Legacy)
    "anthropic.claude-3-opus-20240229-v1:0": "anthropic.claude-opus-5",
    "anthropic.claude-3-haiku-20240307-v1:0": "anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-opus-4-20250514-v1:0": "anthropic.claude-opus-5",
    "anthropic.claude-sonnet-4-20250514-v1:0": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-3-5-haiku-20241022-v1:0": "anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-3-7-sonnet-20250219-v1:0": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    # Cohere
    "cohere.command-text-v14": "cohere.command-r-v1:0",
    "cohere.command-light-text-v14": "cohere.command-r-v1:0",
    # Mistral AI
    "mistral.mistral-large-2402-v1:0": "mistral.mistral-large-3-675b-instruct",
    # Meta Llama 2 (EOL)
    "meta.llama2-13b-chat-v1": "meta.llama3-1-8b-instruct-v1:0",
    "meta.llama2-70b-chat-v1": "meta.llama3-1-70b-instruct-v1:0",
    "meta.llama2-13b-v1": "meta.llama3-1-8b-instruct-v1:0",
    "meta.llama2-70b-v1": "meta.llama3-1-70b-instruct-v1:0",
    # Meta Llama 3.1 / 3.2 (Legacy)
    "meta.llama3-1-405b-instruct-v1:0": "meta.llama4-maverick-17b-instruct-v1:0",
    "meta.llama3-2-1b-instruct-v1:0": "meta.llama4-scout-17b-instruct-v1:0",
    "meta.llama3-2-3b-instruct-v1:0": "meta.llama4-scout-17b-instruct-v1:0",
    "meta.llama3-2-11b-instruct-v1:0": "meta.llama4-maverick-17b-instruct-v1:0",
    "meta.llama3-2-90b-instruct-v1:0": "meta.llama4-maverick-17b-instruct-v1:0",
    # Stability
    "stability.stable-diffusion-xl-v1": "stability.stable-image-core-v1:1",
    "stability.sd3-large-v1:0": "stability.sd3-5-large-v1:0",
    "stability.stable-image-core-v1:0": "stability.stable-image-core-v1:1",
    "stability.stable-image-ultra-v1:0": "stability.stable-image-ultra-v1:1",
    **SETTINGS.aws_bedrock_deprecated_models,
}
