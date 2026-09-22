"""OpenAI gpt-oss chat model implementation."""

from typing import TYPE_CHECKING, Literal

from stdapi.models.chat._reasoning_effort import REASONING_DISABLE_IGNORED
from stdapi.models.chat.openai_gpt import ChatModel as _GptChatModel
from stdapi.monitoring import log_error_details

if TYPE_CHECKING:
    from stdapi.models.chat import Effort
    from stdapi.types import JsonMapping
    from stdapi.types.anthropic_messages import ThinkingDisplay

#: Values the flat ``reasoning_effort`` field accepts on gpt-oss.
type GptOssReasoning = Literal["low", "medium", "high"]

#: Requested effort mapped onto gpt-oss's scale; ``none`` and ``minimal`` are rejected.
_REASONING_OVERRIDE: dict[Effort | None, GptOssReasoning] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


class ChatModel(_GptChatModel):
    """OpenAI gpt-oss-specific chat model implementation.

    Probed on Converse, gpt-oss ignores the ``reasoning.effort`` object the GPT
    models take and honours the flat ``reasoning_effort`` instead, at ``low``,
    ``medium`` or ``high``. It always reasons: ``none`` is rejected, so a request
    disabling reasoning is served at the model's default.
    """

    __slots__ = ()

    MATCHER = "openai.gpt-oss-"

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
        """Set the flat reasoning effort from the requested level.

        Args:
            additional_request_fields: Request fields to modify with the effort.
            enabled: Whether reasoning is enabled; disabling sends nothing.
            reasoning_effort: Requested effort level, mapped onto gpt-oss's scale.
            budget_tokens: Not supported: reasoning is sized by effort only.
            max_tokens: Not used.
            display: Unused.
        """
        if not enabled:
            log_error_details(REASONING_DISABLE_IGNORED, level="warning")
        elif effort := _REASONING_OVERRIDE.get(reasoning_effort):
            additional_request_fields["reasoning_effort"] = effort
