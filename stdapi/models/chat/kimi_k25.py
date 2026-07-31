"""Moonshot Kimi K2 chat model implementation."""

from re import compile as re_compile
from typing import TYPE_CHECKING

from stdapi.models.chat._default import ChatModel as _BaseChatModel

if TYPE_CHECKING:
    from stdapi.models.chat import Effort
    from stdapi.types import JsonMapping


class ChatModel(_BaseChatModel):
    """Moonshot Kimi K2-specific chat model implementation.

    Supports Kimi-specific thinking/reasoning configuration via
    ``additionalModelRequestFields.thinking``.
    """

    #: Matches both Bedrock provider prefixes for any Kimi K2.x model, open-ended on version.
    MATCHER = re_compile(r"^moonshot(?:ai)?\.kimi-k2")

    def _req_configure_reasoning(
        self,
        additional_request_fields: JsonMapping,
        *,
        enabled: bool,
        reasoning_effort: Effort | None = None,  # noqa: ARG002
        budget_tokens: int | None = None,  # noqa: ARG002
        max_tokens: int | None = None,  # noqa: ARG002
    ) -> None:
        """Configure thinking parameters for Kimi K2 models.

        Kimi uses the ``thinking`` field in additionalModelRequestFields.
        Supports only enabling or disabling thinking (budget_tokens is ignored).

        Args:
            additional_request_fields: Request fields to modify with thinking config.
            enabled: Whether thinking should be enabled.
            reasoning_effort: Not used for Kimi.
            budget_tokens: Not supported by Kimi.
            max_tokens: Not used by Kimi.
        """
        additional_request_fields["thinking"] = {
            "type": "enabled" if enabled else "disabled"
        }
