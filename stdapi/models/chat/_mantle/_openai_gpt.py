"""Reasoning effort shared by the OpenAI GPT families on Amazon Bedrock Mantle."""

from typing import TYPE_CHECKING, Any

from stdapi.models.chat._mantle._default import ChatModel as MantleChatModel
from stdapi.models.chat._reasoning_effort import REASONING_DISABLE_IGNORED
from stdapi.models.chat.openai_gpt import ALWAYS_REASONING_MATCHER
from stdapi.monitoring import log_error_details

if TYPE_CHECKING:
    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.aws_bedrock_mantle import MantleApi


class OpenAIGptChatModel(MantleChatModel):
    """OpenAI GPT model on Mantle, with its reasoning effort fitted to what it accepts.

    Probed on Mantle (GPT-5.6 Luna, GPT-6 Luna and Astra): ``minimal`` is
    rejected on both APIs, so it is sent as ``low``; GPT-6 Astra also rejects
    ``none``, so a request disabling reasoning there is served at its default.
    """

    __slots__ = ()

    async def _serve(
        self,
        inbound: MantleApi,
        payload: dict[str, Any],
        *,
        stream: bool,
        region: RegionName | None = None,
    ) -> tuple[MantleApi, RegionName, Any]:
        """Fit the requested reasoning effort, then serve the request.

        Args:
            inbound: API matching the inbound route.
            payload: Inbound request body, updated in place.
            stream: Whether to open a streaming invocation.
            region: Pinned region, if any.

        Returns:
            Tuple of (upstream API used, serving region, raw result).
        """
        if inbound == "chat_completions":
            self._fit_effort(payload, "reasoning_effort")
        elif inbound == "responses" and isinstance(
            reasoning := payload.get("reasoning"), dict
        ):
            self._fit_effort(reasoning, "effort")
        return await super()._serve(inbound, payload, stream=stream, region=region)

    def _fit_effort(self, container: dict[str, Any], key: str) -> None:
        """Replace an effort the model rejects in *container*, in place.

        Args:
            container: Object holding the effort.
            key: Key of the effort in *container*.
        """
        effort = container.get(key)
        if effort == "minimal":
            container[key] = "low"
        elif effort == "none" and ALWAYS_REASONING_MATCHER.match(self._model_id):
            del container[key]
            log_error_details(REASONING_DISABLE_IGNORED, level="warning")
