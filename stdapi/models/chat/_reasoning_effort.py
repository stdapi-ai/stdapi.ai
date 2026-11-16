"""Shared reasoning control of the models taking an OpenAI-shaped effort object."""

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from stdapi.models.chat._default import ChatModel as _BaseChatModel
from stdapi.monitoring import log_error_details

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime.literals import ServiceTierTypeType
    from types_aiobotocore_bedrock_runtime.type_defs import (
        InferenceConfigurationTypeDef,
        JsonSchemaDefinitionTypeDef,
        MessageTypeDef,
        SystemContentBlockTypeDef,
        ToolConfigurationTypeDef,
    )

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef
    from stdapi.models.chat import Effort
    from stdapi.types import JsonMapping
    from stdapi.types.anthropic_messages import ThinkingDisplay

#: Values the ``reasoning.effort`` field accepts.
type ReasoningObjectEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]

#: Requested effort mapped onto the accepted values; ``minimal`` is rejected, so it maps to ``low``.
_REASONING_EFFORT: frozendict[Effort | None, ReasoningObjectEffort] = frozendict(
    {
        "none": "none",
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
        "max": "max",
    }
)

#: Sampling fields a model with fixed sampling refuses, with the client parameter each comes from.
_SAMPLING_FIELDS: tuple[tuple[Literal["temperature", "topP"], str], ...] = (
    ("temperature", "temperature"),
    ("topP", "top_p"),
)

#: Logged when a request disables reasoning on a model that cannot stop reasoning.
REASONING_DISABLE_IGNORED = (
    "Reasoning cannot be disabled on this model: "
    "its default reasoning level is used instead"
)


class ReasoningEffortChatModel(_BaseChatModel):
    """Chat model whose reasoning is set by ``additionalModelRequestFields.reasoning.effort``.

    These models reason by default, and ``effort: "none"`` is their only off
    switch: disabling reasoning sends it, and a request enabling reasoning
    without naming a level sends nothing, leaving the model's own default.

    A model with fixed sampling refuses ``temperature`` and ``top_p``; as they
    only tune sampling, they are dropped with a logged warning rather than
    refused, while ``stop`` is still sent.
    """

    __slots__ = ()

    #: Whether the model accepts ``effort: "none"``; one that always reasons rejects it.
    REASONING_DISABLE_SUPPORTED: ClassVar[bool] = True

    #: Whether the model accepts ``temperature`` and ``topP``; one with fixed sampling rejects them.
    SAMPLING_SUPPORTED: ClassVar[bool] = True

    async def _prepare_converse_request(
        self,
        bedrock_messages: list[MessageTypeDef],
        inference_cfg: InferenceConfigurationTypeDef,
        system_blocks: list[SystemContentBlockTypeDef] | None,
        tool_config: ToolConfigurationTypeDef | None,
        additional_request_fields: dict[str, Any],
        service_tier: ServiceTierTypeType | None,
        output_config: JsonSchemaDefinitionTypeDef | None = None,
        request_metadata: dict[str, str] | None = None,
    ) -> ConverseRequestBaseTypeDef:
        """Build the Converse request, without the sampling a fixed-sampling model refuses.

        Args:
            bedrock_messages: Converted Bedrock message list.
            inference_cfg: Bedrock inference configuration, modified in place.
            system_blocks: Optional top-level system instruction blocks.
            tool_config: Optional Bedrock tool configuration.
            additional_request_fields: Additional request fields.
            service_tier: Service tier configuration.
            output_config: Optional Bedrock output JSON Schema configuration.
            request_metadata: Optional Bedrock ``requestMetadata``.

        Returns:
            Bedrock Converse request payload.
        """
        if not self.SAMPLING_SUPPORTED and (
            dropped := [
                parameter
                for field, parameter in _SAMPLING_FIELDS
                if inference_cfg.pop(field, None) is not None
            ]
        ):
            log_error_details(
                f"{' and '.join(repr(p) for p in dropped)} not supported by this "
                "model: ignored.",
                level="warning",
            )
        return await super()._prepare_converse_request(
            bedrock_messages,
            inference_cfg,
            system_blocks,
            tool_config,
            additional_request_fields,
            service_tier,
            output_config,
            request_metadata,
        )

    def _req_configure_reasoning(
        self,
        additional_request_fields: JsonMapping,
        *,
        enabled: bool,
        reasoning_effort: Effort | None = None,
        budget_tokens: int | None = None,  # noqa: ARG002
        max_tokens: int | None = None,  # noqa: ARG002
        display: ThinkingDisplay | None = None,  # noqa: ARG002
    ) -> None:
        """Set the reasoning effort object from the requested level.

        A model that cannot stop reasoning is served at its default level
        instead of being sent the ``none`` it rejects.

        Args:
            additional_request_fields: Request fields to modify with the effort.
            enabled: Whether reasoning is enabled.
            reasoning_effort: Requested effort level, mapped onto the accepted values.
            budget_tokens: Not supported: reasoning is sized by effort only.
            max_tokens: Not used.
            display: Unused.
        """
        effort = _REASONING_EFFORT.get(reasoning_effort) if enabled else "none"
        if effort == "none" and not self.REASONING_DISABLE_SUPPORTED:
            log_error_details(REASONING_DISABLE_IGNORED, level="warning")
            return
        if effort is not None:
            additional_request_fields["reasoning"] = {"effort": effort}
