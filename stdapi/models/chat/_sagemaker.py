"""Chat model served by an Amazon SageMaker AI endpoint.

The supported containers serve the OpenAI Chat Completions API and nothing
else, so this class is the Bedrock Mantle chat model with its transport
replaced: the dialect translation in ``_mantle/_convert.py`` already turns a
Responses or an Anthropic Messages request into Chat Completions and back, and
duplicating it for a second OpenAI-compatible backend would double every
dialect fix.

There is one class rather than a registry of families: what a family class
encodes is how one *serverless* model diverges, while an endpoint's divergences
are its container's, and matching a family by accident on an operator-chosen
model ID would apply the wrong ones.
"""

from typing import TYPE_CHECKING, Any, ClassVar

from stdapi.api_errors import UnsupportedModelError
from stdapi.aws_bedrock_mantle import (
    refuse_unappliable_guardrail,
    refuse_unattributable_invocation,
    usage_from_chat_completion,
)
from stdapi.aws_sagemaker import invoke, invoke_stream
from stdapi.models import SAGEMAKER_ENDPOINT_MODELS, set_effective_region
from stdapi.models.chat._mantle._default import ChatModel as MantleChatModel
from stdapi.pricing import Service
from stdapi.usage import record_bedrock_usage

if TYPE_CHECKING:
    from collections.abc import Mapping

    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.aws_bedrock_mantle import MantleApi
    from stdapi.config import SageMakerEndpointConfig
    from stdapi.models.chat import ChatModelBase

#: The only API the supported inference containers serve.
_CHAT_COMPLETIONS: MantleApi = "chat_completions"


class SageMakerChatModel(MantleChatModel):
    """Chat model invoked on an operator-declared SageMaker AI endpoint."""

    __slots__ = ()

    #: The endpoint names the model it serves, so the request body must not.
    _EMPTY_MODEL: ClassVar[str] = ""

    #: Served over an OpenAI-compatible HTTP transport rather than Converse.
    IS_MANTLE: ClassVar[bool] = True

    #: Chat Completions is the only API the supported containers serve.
    NATIVE_APIS: ClassVar[frozenset[MantleApi]] = frozenset({_CHAT_COMPLETIONS})

    def _endpoint(self) -> SageMakerEndpointConfig:
        """Return the endpoint declaration serving this model.

        Returns:
            The operator's declaration for this model.

        Raises:
            UnsupportedModelError: When the model left the catalogue between
                its resolution and its invocation (a configuration reload).
        """
        model = SAGEMAKER_ENDPOINT_MODELS.get(self._model_id)
        endpoint = model.sagemaker_endpoint if model is not None else None
        if endpoint is None:
            raise UnsupportedModelError(self._model_id)
        return endpoint

    def _select_api(self, inbound: MantleApi, tried: set[MantleApi]) -> MantleApi:
        """Select the upstream API serving an inbound request.

        Args:
            inbound: API matching the inbound route (unused: there is one).
            tried: APIs already attempted for this request.

        Returns:
            Always Chat Completions.

        Raises:
            UnsupportedModelError: When it has already been tried and failed.
        """
        del inbound
        if _CHAT_COMPLETIONS in tried:  # pragma: no cover - nothing demotes it
            raise UnsupportedModelError(self._model_id)
        return _CHAT_COMPLETIONS

    async def _invoke_api(
        self,
        api: MantleApi,
        payload: dict[str, Any],
        *,
        stream: bool,
        region: RegionName | None = None,
    ) -> tuple[RegionName, Any]:
        """Invoke the endpoint's Chat Completions route.

        An endpoint answers only in its own Region and has no cross-Region
        form, so there is no candidate list to route across and no failover:
        what a cold endpoint gets instead is the warm-up wait in the transport.

        Args:
            api: Target API (always Chat Completions).
            payload: JSON request body.
            stream: Whether to open a streaming invocation.
            region: Pinned region, ignored: the endpoint has exactly one.

        Returns:
            Tuple of (serving region, parsed JSON response or SSE generator).

        Raises:
            ApiError: When a guardrail applies and cannot be carried, or when
                the request identifies no end user and one is required.
        """
        del api, region
        # This override replaces the Mantle dispatch and its transport, and
        # with them both refusals that path makes: an endpoint's container has
        # no guardrailConfig either, and its invocation is signed from the
        # server's own credentials just the same, off the botocore hook that
        # enforces the end-user identity requirement.
        refuse_unappliable_guardrail()
        refuse_unattributable_invocation()
        endpoint = self._endpoint()
        set_effective_region(self._model_id, endpoint.region)
        # The endpoint and its inference component name the model, in the URL.
        payload["model"] = self._EMPTY_MODEL
        args = (
            endpoint.region,
            endpoint.endpoint,
            endpoint.inference_component,
            payload,
        )
        if stream:
            return endpoint.region, await invoke_stream(*args)
        return endpoint.region, await invoke(*args)

    def _record_usage(
        self,
        api: MantleApi,
        usage: Mapping[str, Any],
        region: RegionName,
        tier: str | None,
    ) -> None:
        """Record the tokens the endpoint reported, with no cost attached.

        AWS bills a real-time endpoint by the instance-hour of the instances it
        runs on, not by the tokens it serves, so the quantities are recorded
        under a service that resolves no price rather than an invented one.

        Args:
            api: Upstream API that produced the usage (always Chat Completions).
            usage: Raw usage object from the response.
            region: Region that served the call.
            tier: Service tier reported by the response, if any.
        """
        del api, tier
        record_bedrock_usage(
            self._model_id,
            service=Service.SAGEMAKER,
            region=region,
            # An endpoint serves in its own Region, at no published tier.
            tier=None,
            routing="",
            **usage_from_chat_completion(usage),
        )


#: SageMaker AI chat model instance cache.
_SAGEMAKER_CHAT_MODEL_CACHE: dict[str, ChatModelBase[Any, Any]] = {}


def get_sagemaker_chat_model(model_id: str) -> ChatModelBase[Any, Any]:
    """Resolve the chat model instance serving a SageMaker AI endpoint.

    Args:
        model_id: The declared model identifier.

    Returns:
        The chat model associated to the ``model_id``.
    """
    model = _SAGEMAKER_CHAT_MODEL_CACHE.get(model_id)
    if model is None:
        model = _SAGEMAKER_CHAT_MODEL_CACHE[model_id] = SageMakerChatModel(model_id)
    return model
