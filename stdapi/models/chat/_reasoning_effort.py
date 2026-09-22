"""Shared reasoning control of the models taking an OpenAI-shaped effort object."""

from typing import TYPE_CHECKING, ClassVar, Literal

from stdapi.models.chat._default import ChatModel as _BaseChatModel
from stdapi.monitoring import log_error_details

if TYPE_CHECKING:
    from stdapi.models.chat import Effort
    from stdapi.types import JsonMapping
    from stdapi.types.anthropic_messages import ThinkingDisplay

#: Values the ``reasoning.effort`` field accepts.
type ReasoningObjectEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]

#: Requested effort mapped onto the accepted values; ``minimal`` is rejected, so it maps to ``low``.
_REASONING_EFFORT: dict[Effort | None, ReasoningObjectEffort] = {
    "none": "none",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "max",
}

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
    """

    __slots__ = ()

    #: Whether the model accepts ``effort: "none"``; one that always reasons rejects it.
    REASONING_DISABLE_SUPPORTED: ClassVar[bool] = True

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
