"""Models Deprecation.

Mapping of deprecated (or pending deprecation) model IDs, with recommended alternatives.
This allows improving error messages for invalid models.

Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html
"""

DEPRECATED_MODELS: dict[str, str] = {
    "ai21.jamba-instruct-v1:0": "ai21.jamba-1-5-large-v1:0",
    "amazon.titan-image-generator-v1": "amazon.nova-canvas-v1:0",
    "amazon.titan-text-express-v1": "amazon.nova-lite-v1:0",
    "amazon.titan-text-lite-v1": "amazon.nova-micro-v1:0",
    "amazon.titan-text-premier-v1:0": "amazon.nova-pro-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0": "anthropic.claude-opus-4-1-20250805-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-3-5-sonnet-20241022-v2:0": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-opus-4-20250514-v1:0": "anthropic.claude-opus-4-1-20250805-v1:0",
    "anthropic.claude-v2": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-v2:1": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "anthropic.claude-instant-v1": "anthropic.claude-sonnet-4-5-20250929-v1:0",
    "cohere.command-text-v14": "cohere.command-r-v1:0",
    "cohere.command-light-text-v14": "cohere.command-r-v1:0",
    "stability.sd3-large-v1:0": "stability.sd3-5-large-v1:0",
    "stability.stable-image-core-v1:0": "stability.stable-image-core-v1:1",
    "stability.stable-image-ultra-v1:0": "stability.stable-image-ultra-v1:1",
}
