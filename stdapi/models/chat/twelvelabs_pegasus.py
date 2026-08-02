"""TwelveLabs Pegasus chat model (twelvelabs.pegasus-1-2-v1:0)."""

from typing import TYPE_CHECKING, Any, ClassVar, Literal, NotRequired, TypedDict

from stdapi.api_errors import ApiError
from stdapi.aws import AWS_ENVIRONMENT
from stdapi.aws_bedrock import usage_from_amazon_bedrock_invocation_metrics
from stdapi.aws_s3 import put_s3_object
from stdapi.models.chat._default import ChatModel as _BaseChatModel
from stdapi.utils import b64encode

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_runtime.literals import (
        ServiceTierTypeType,
        StopReasonType,
    )
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ConverseOutputTypeDef,
        ConverseResponseTypeDef,
        ConverseStreamOutputTypeDef,
        ConverseStreamResponseTypeDef,
        GuardrailStreamConfigurationTypeDef,
        MessageTypeDef,
        TokenUsageTypeDef,
    )

    from stdapi.aws_bedrock import BedrockInvocationTypeDef, ConverseRequestBaseTypeDef

    #: Pegasus FinishReason type.
    _FinishReason = Literal["stop", "length"]

    class _PegasusChunk(BedrockInvocationTypeDef):
        """Pegasus streaming chunk type."""

        message: str
        stopReason: _FinishReason | Literal[""]  # noqa: N815


#: Maximum raw-byte size for inline base64 video (Pegasus limit: 25 MB base64 string ≈ 18.75 MB raw).
_PEGASUS_INLINE_BYTES: int = 25 * 1024 * 1024 * 3 // 4  # 18_874_368

#: Maps Pegasus finish/stop reason strings to Bedrock Converse stop reasons.
_STOP_MAP: dict[_FinishReason, StopReasonType] = {
    "stop": "end_turn",
    "length": "max_tokens",
}


class _S3Location(TypedDict):
    """S3 location of a video referenced by URI rather than inline bytes."""

    uri: str
    bucketOwner: NotRequired[str]


class _MediaSource(TypedDict):
    """Pegasus video input: either inline base64 or an S3 reference."""

    base64String: NotRequired[str]
    s3Location: NotRequired[_S3Location]


class _JsonSchema(TypedDict):
    """Named JSON Schema for structured Pegasus output."""

    name: str
    schema: dict  # type: ignore[type-arg]


class _ResponseFormat(TypedDict):
    """Pegasus structured-output request wrapper."""

    jsonSchema: _JsonSchema


class _PegasusRequest(TypedDict):
    """Pegasus InvokeModel request body."""

    inputPrompt: str
    mediaSource: _MediaSource
    temperature: NotRequired[float]
    maxOutputTokens: NotRequired[int]
    responseFormat: NotRequired[_ResponseFormat]


class _PegasusResponse(TypedDict):
    """Pegasus InvokeModel response body."""

    message: str
    finishReason: _FinishReason


def _extract_latest_video(messages: list[MessageTypeDef]) -> dict[str, Any] | None:
    """Walk messages in reverse; return the first video content block found."""
    for message in reversed(messages):
        for block in reversed(message.get("content", [])):
            if "video" in block:
                return block["video"]  # type: ignore[return-value]
    return None


def _extract_latest_user_text(messages: list[MessageTypeDef]) -> str:
    """Return text from the latest contiguous run of user messages (reversed, then rejoined)."""
    parts: list[str] = []
    for message in reversed(messages):
        if message["role"] != "user":
            break
        parts.extend(
            text for block in message.get("content", []) if (text := block.get("text"))
        )
    return "\n".join(reversed(parts))


async def _video_to_media_source(
    video_block: dict[str, Any], region: RegionName
) -> _MediaSource:
    """Translate a resolved Bedrock video source dict to Pegasus's _MediaSource.

    Args:
        video_block: Resolved Bedrock video source (either s3Location or bytes).
        region: AWS region for auto-S3 upload.

    Returns:
        Pegasus _MediaSource ready for the API request.
    """
    if "s3Location" in (source := video_block.get("source", video_block)):
        loc = source["s3Location"]
        return _MediaSource(
            s3Location=_S3Location(
                uri=loc["uri"],
                bucketOwner=loc.get("bucketOwner") or AWS_ENVIRONMENT["account_id"],
            )
        )
    if len(raw := source["bytes"]) <= _PEGASUS_INLINE_BYTES:
        return _MediaSource(base64String=await b64encode(raw))
    return _MediaSource(
        s3Location=_S3Location(
            uri=(
                await put_s3_object(raw, "video/mp4", region=region, temporary=True)
            ).uri,
            bucketOwner=AWS_ENVIRONMENT["account_id"],
        )
    )


class ChatModel(_BaseChatModel):
    """TwelveLabs Pegasus 1.2 chat model.

    Pegasus is a video-understanding model exposed via Bedrock InvokeModel.
    Each call accepts exactly one video and one text prompt. This subclass
    keeps the full Converse adapter pipeline and translates the final
    Bedrock Converse request into Pegasus's native body inside _converse
    / _converse_stream.

    Unsupported features (system prompts, tools, reasoning, prompt caching)
    are silently ignored so cross-model client code works without
    modification; see documentation for details.
    """

    __slots__ = ()

    MATCHER: ClassVar[str] = "twelvelabs.pegasus"

    #: System prompts are silently ignored; Pegasus has no system parameter.
    SYSTEM_PROMPT_SUPPORTED: ClassVar[bool] = False

    async def _build_pegasus_body(
        self, request: ConverseRequestBaseTypeDef, region: RegionName
    ) -> tuple[
        _PegasusRequest,
        ServiceTierTypeType | None,
        GuardrailStreamConfigurationTypeDef | None,
    ]:
        """Translate a fully-resolved Converse request into a Pegasus InvokeModel body.

        Args:
            request: Fully-resolved Converse request (after _prepare_converse_request_for_region).
            region: AWS region for any needed S3 upload.

        Returns:
            Tuple of (pegasus_body, service_tier, guardrail_config), the last two
            being ``None`` when the Converse request carries neither.

        Raises:
            ApiError: If no video is found in the conversation messages.
        """
        messages: list[MessageTypeDef] = request["messages"]  # type: ignore[assignment]
        if (video_block := _extract_latest_video(messages)) is None:
            error = ApiError(
                status=400,
                message="Pegasus requires exactly one video. No video found in the request messages.",
            )
            error.code = "invalid_request_error"
            raise error
        body: _PegasusRequest = {
            "inputPrompt": _extract_latest_user_text(messages),
            "mediaSource": await _video_to_media_source(video_block, region),
        }
        if inference := request.get("inferenceConfig", {}):
            if (max_tokens := inference.get("maxTokens")) is not None:
                body["maxOutputTokens"] = max_tokens
            if (temperature := inference.get("temperature")) is not None:
                body["temperature"] = temperature
        if output_config := request.get("outputConfig", {}):
            text_format = output_config.get("textFormat", {})
            if (
                text_format.get("type") == "json_schema"
                and (structure := text_format.get("structure"))
                and (json_schema := structure.get("jsonSchema"))
            ):
                body["responseFormat"] = _ResponseFormat(
                    jsonSchema=_JsonSchema(
                        name=json_schema.get("name", "output"),
                        schema=json_schema.get("schema", {}),  # type: ignore[typeddict-item]
                    )
                )
        return (
            body,
            (
                service_tier_dict.get("type")
                if (service_tier_dict := request.get("serviceTier"))
                else None
            ),
            request.get("guardrailConfig"),
        )

    async def _converse(
        self,
        request: ConverseRequestBaseTypeDef,
        region: RegionName,
        *,
        single_region: bool,  # noqa: ARG002
    ) -> ConverseResponseTypeDef:
        """Override Bedrock Converse with Pegasus InvokeModel.

        Args:
            request: Base Converse request (will be mutated in place by _prepare_converse_request_for_region).
            region: Initial target region (may change after region resolution).
            single_region: When True, prevents cross-region retries.

        Returns:
            ConverseResponseTypeDef shaped like a standard Bedrock Converse response.
        """
        await self._prepare_converse_request_for_region(request, region)
        body, service_tier, guardrail_config = await self._build_pegasus_body(
            request, region
        )
        result = await self.invoke(
            body, region=region, service_tier=service_tier, guardrail=guardrail_config
        )

        response: _PegasusResponse = result.response
        output: ConverseOutputTypeDef = {
            "message": {"role": "assistant", "content": [{"text": response["message"]}]}
        }
        stop_reason: StopReasonType = _STOP_MAP.get(
            response.get("finishReason", "stop"), "end_turn"
        )

        input_tokens = result.input_tokens or 0
        output_tokens = result.output_tokens or 0
        usage: TokenUsageTypeDef = {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
        }

        return {"output": output, "stopReason": stop_reason, "usage": usage}  # type: ignore[typeddict-item]

    async def _converse_stream(
        self,
        request: ConverseRequestBaseTypeDef,
        region: RegionName,
        *,
        single_region: bool,  # noqa: ARG002
    ) -> ConverseStreamResponseTypeDef:
        """Override Bedrock Converse streaming with Pegasus InvokeModelWithResponseStream.

        Args:
            request: Base Converse request (will be mutated in place).
            region: Target region.
            single_region: When True, prevents cross-region retries.

        Returns:
            ConverseStreamResponseTypeDef with an async generator in the "stream" key.
        """
        await self._prepare_converse_request_for_region(request, region)
        body, service_tier, guardrail_config = await self._build_pegasus_body(
            request, region
        )
        raw_stream = self.invoke_stream(
            body, region=region, service_tier=service_tier, guardrail=guardrail_config
        )
        return {"stream": self._format_converse_stream(raw_stream)}  # type: ignore[typeddict-item]

    async def _format_converse_stream(
        self, raw_stream: AsyncGenerator[_PegasusChunk]
    ) -> AsyncGenerator[ConverseStreamOutputTypeDef]:
        """Translate Pegasus delta-style stream to Bedrock ConverseStream event format.

        Args:
            raw_stream: Async generator of Pegasus streaming chunks.

        Yields:
            Bedrock ConverseStream events.
        """
        input_tokens = 0
        output_tokens = 0
        yield {"messageStart": {"role": "assistant"}}
        async for chunk in raw_stream:
            if delta := chunk["message"]:
                yield {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"text": delta},
                    }
                }
            if stop_reason := chunk.get("stopReason"):
                yield {"contentBlockStop": {"contentBlockIndex": 0}}
                yield {
                    "messageStop": {
                        "stopReason": _STOP_MAP.get(stop_reason, "end_turn")
                    }
                }

                # Usage is recorded by invoke_stream(); extract here only for the
                # metadata event. inputTokenCount is always 0 for Pegasus streams.
                bedrock_usage = usage_from_amazon_bedrock_invocation_metrics(chunk)
                input_tokens = bedrock_usage.input_tokens
                output_tokens = bedrock_usage.output_tokens

        usage: TokenUsageTypeDef = {
            "inputTokens": input_tokens,
            "outputTokens": output_tokens,
            "totalTokens": input_tokens + output_tokens,
        }
        yield {"metadata": {"usage": usage}}  # type: ignore[typeddict-item]
