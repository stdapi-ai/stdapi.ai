"""Anthropic Claude Opus 5 and later chat model implementation."""

from re import compile as re_compile
from typing import TYPE_CHECKING, ClassVar

from stdapi.api_errors import ApiError
from stdapi.models.chat._anthropic_claude import AnthropicClaudeChatModel

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime.type_defs import (
        MessageTypeDef,
        ToolConfigurationTypeDef,
    )

    from stdapi.types import JsonMapping

#: Opus 5.5+ versions, which always reason and refuse a forced tool choice.
OPUS_5_5_MATCHER = re_compile(
    r"^anthropic\.claude-opus-(?:5-(?:[5-9]|\d{2})|[6-9]|\d{2})(?:\D|$)"
)

#: Refusal of a forced tool choice, worded for every API dialect and legacy field.
FORCED_TOOL_CHOICE_REFUSED = (
    "Forcing tool use (tool_choice or function_call) is not available with "
    "this model. Let the model choose ('auto') and name the tool in the prompt."
)


class ChatModel(AnthropicClaudeChatModel):
    """Anthropic Claude Opus 5 and later chat model implementation.

    From Opus 5.5 on, adaptive thinking is always on: Bedrock rejects an
    explicitly disabled reasoning configuration, so a request disabling
    reasoning is served with the adaptive default instead. Opus 5 still accepts
    it. The same versions refuse a tool choice forcing tool use, which is
    rejected before sending with an error naming the way forward. Later
    versions are assumed to keep both behaviors.
    """

    __slots__ = ()

    MATCHER = re_compile(r"^anthropic\.claude-opus-(?:[5-9]|\d\d)")
    SYSTEM_MESSAGE_AS_MESSAGES_SUPPORTED: ClassVar[bool] = True

    @property
    def REASONING_DISABLE_SUPPORTED(self) -> bool:  # type: ignore[override]  # noqa: N802
        """Whether Bedrock accepts an explicitly disabled reasoning configuration."""
        return OPUS_5_5_MATCHER.match(self._model_id) is None

    def _req_configure_tools(
        self,
        tool_config: ToolConfigurationTypeDef | None,
        additional_request_fields: JsonMapping,
        server_tools: list[JsonMapping],
        bedrock_messages: list[MessageTypeDef] | None = None,
    ) -> None:
        """Refuse a forced tool choice on Opus 5.5+, then configure Claude tools.

        Args:
            tool_config: Bedrock tool configuration after system tool promotion.
            additional_request_fields: Mutable additional request fields dict.
            server_tools: Per-tool dicts.
            bedrock_messages: Translated Bedrock message list.

        Raises:
            ApiError: When the request forces tool use on a model refusing it.
        """
        choice = tool_config.get("toolChoice") if tool_config else None
        if (
            choice
            and ("any" in choice or "tool" in choice)
            and OPUS_5_5_MATCHER.match(self._model_id)
        ):
            raise ApiError(FORCED_TOOL_CHOICE_REFUSED)
        super()._req_configure_tools(
            tool_config, additional_request_fields, server_tools, bedrock_messages
        )
