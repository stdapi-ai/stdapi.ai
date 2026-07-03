"""Bedrock related models."""

from pydantic import Field

from stdapi.types import BaseModelRequest


# Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html#inference-chat-completions-guardrails
class AmazonBedrockGuardrailConfigParams(BaseModelRequest):
    """Amazon Bedrock Guardrail configuration parameters."""

    tagSuffix: str | None = Field(  # noqa: N815
        default=None,
        description="Amazon Bedrock Guardrail input tagging. UNSUPPORTED on this implementation.",
    )
