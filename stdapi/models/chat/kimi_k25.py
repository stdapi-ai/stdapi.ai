"""Moonshot Kimi K2 chat model implementation."""

from re import compile as re_compile
from typing import TYPE_CHECKING, Literal

from stdapi.models.chat._default import ChatModel as _BaseChatModel

if TYPE_CHECKING:
    from stdapi.models.chat import Effort
    from stdapi.types import JsonMapping
    from stdapi.types.anthropic_messages import ThinkingDisplay

type KimiReasoning = Literal["low", "medium", "high"]

#: OpenAI effort levels mapped onto the values Kimi accepts (``minimal`` is rejected).
_REASONING_OVERRIDE: dict[Effort | None, KimiReasoning] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


class ChatModel(_BaseChatModel):
    """Moonshot Kimi K2 generation chat model implementation (K2 Thinking, K2.5).

    Configures reasoning through the K2 generation's
    ``additionalModelRequestFields.thinking`` and ``.reasoning_effort``; Kimi K3
    ignores both and has a class of its own.
    """

    __slots__ = ()

    #: Kimi K2.x under either Bedrock provider prefix; never K20 and later, which K3's class takes.
    MATCHER = re_compile(r"^moonshot(?:ai)?\.kimi-k2(?!\d)")

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
        """Configure thinking parameters for Kimi K2 models.

        Kimi carries both a ``thinking`` toggle and a ``reasoning_effort`` level in
        additionalModelRequestFields, and the level is what decides: Kimi K2.5
        returns reasoning content only at ``high``, whatever ``thinking`` says, so
        turning thinking on without naming a level asks for ``high``. Kimi K2
        Thinking reasons at every level. Budget tokens are not supported.

        Args:
            additional_request_fields: Request fields to modify with thinking config.
            enabled: Whether thinking should be enabled.
            reasoning_effort: Requested effort level, mapped onto Kimi's own scale.
            budget_tokens: Not supported by Kimi.
            max_tokens: Not used by Kimi.
            display: Unused.
        """
        additional_request_fields["thinking"] = {
            "type": "enabled" if enabled else "disabled"
        }
        if enabled:
            additional_request_fields["reasoning_effort"] = _REASONING_OVERRIDE.get(
                reasoning_effort, "high"
            )
