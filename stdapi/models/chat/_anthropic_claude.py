"""Common base for all Anthropic Claude chat model implementations."""

from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

from stdapi.config import SETTINGS
from stdapi.models.chat._default import ChatModel as _BaseChatModel
from stdapi.monitoring import log_error_details

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime.type_defs import ToolConfigurationTypeDef

    from stdapi.types import JsonList, JsonMapping
    from stdapi.types.anthropic_messages import ServerTools, ToolUnionParam


#: Fields excluded when serializing Anthropic tool params to the wire format
_SERIALIZE_EXCLUDE_FIELDS: set[str] = {
    "cache_control",
    "defer_loading",
    "input_examples",
    "strict",
    "allowed_callers",
}


class AnthropicClaudeChatModel(_BaseChatModel):
    """Shared functionality for all Anthropic Claude model generations."""

    PROMPT_CACHING_SUPPORTED = True
    PROMPT_CACHING_TOOL_SUPPORTED = True
    PASSTHROUGH_HEADERS = MappingProxyType(
        {"anthropic-beta": ("anthropic_beta", lambda v: v.split(","))}
    )
    SIMPLIFIED_CACHE_MANAGEMENT = True

    #: Required ``anthropic_beta`` flags per tool name.
    TOOL_BETA_FLAGS: ClassVar[MappingProxyType[ServerTools, str]]

    def _req_configure_system_tools(
        self,
        tool_config: ToolConfigurationTypeDef | None,
        system_tools: list[ToolUnionParam],
        additional_request_fields: JsonMapping,
    ) -> ToolConfigurationTypeDef | None:
        """Configure Anthropic system tools via ``additionalModelRequestFields``.

        Args:
            tool_config: Existing Bedrock tool configuration, or ``None``.
            system_tools: List of Anthropic system tool params.
            additional_request_fields: Mutable dict of additional request fields.

        Returns:
            Unchanged tool configuration (system tools go via additionalModelRequestFields).
        """
        serialized_tools: JsonList = additional_request_fields.get("tools", [])  # type: ignore[assignment]
        serialized_tools.extend(
            tool.model_dump(
                mode="json", exclude_none=True, exclude=_SERIALIZE_EXCLUDE_FIELDS
            )
            for tool in system_tools
        )
        additional_request_fields["tools"] = serialized_tools

        if required_flags := {
            flag
            for tool in system_tools
            if (flag := self.TOOL_BETA_FLAGS.get(tool.name))  # type: ignore[call-overload]
        }:
            existing_flags: list[str] = additional_request_fields.get(  # type: ignore[assignment]
                "anthropic_beta", []
            )
            if new_flags := required_flags - set(existing_flags):
                additional_request_fields["anthropic_beta"] = existing_flags + list(
                    new_flags
                )

        return tool_config

    def _prepare_additional_request_fields(
        self, additional_request_fields: JsonMapping
    ) -> JsonMapping:
        """Merge passthrough headers and filter unsupported ``anthropic_beta`` flags.

        Extends the base implementation to remove ``anthropic_beta`` flags
        not supported by AWS Bedrock, preventing ``ValidationException`` errors.

        Args:
            additional_request_fields: Fields from request body and defaults.

        Returns:
            The merged and filtered additional request fields.
        """
        additional_request_fields = super()._prepare_additional_request_fields(
            additional_request_fields
        )
        if (
            SETTINGS.anthropic_beta_filter
            and "anthropic_beta" in additional_request_fields
        ):
            original_flags: list[str] = additional_request_fields["anthropic_beta"]  # type: ignore[assignment]
            rejected_flags = set(original_flags) - SETTINGS.anthropic_beta_allowlist
            if rejected_flags:
                log_error_details(
                    f"Filtered unsupported anthropic_beta flags: {', '.join(rejected_flags)}",
                    level="warning",
                )
            if allowed_flags := [
                flag for flag in original_flags if flag not in rejected_flags
            ]:
                additional_request_fields["anthropic_beta"] = allowed_flags  # type: ignore[assignment]
            else:
                del additional_request_fields["anthropic_beta"]
        return additional_request_fields
