"""Converse request building for the Amazon Nova 2 chat model (no AWS calls).

Nova 2 rejects a Converse request that combines ``inferenceConfig.maxTokens``
with ``reasoningConfig.maxReasoningEffort: high``, so the model class drops the
token limit and records a warning in the request log instead of letting Bedrock
fail the call.

Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
     stdapi/models/chat/amazon_nova_2.py:ChatModel._prepare_converse_request
"""

from typing import TYPE_CHECKING

import pytest

from stdapi.models.chat.amazon_nova_2 import ChatModel

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime.type_defs import (
        InferenceConfigurationTypeDef,
    )

    from stdapi.monitoring import EventLog

pytestmark = [pytest.mark.local, pytest.mark.usefixtures("request_log")]

#: Model instance used across tests; construction is side-effect free.
_MODEL = ChatModel("amazon.nova-2-lite-v1:0")


class TestPrepareConverseRequestReasoningMaxTokens:
    """``_prepare_converse_request`` drops ``maxTokens`` only for high-effort reasoning.

    Ref: stdapi/models/chat/amazon_nova_2.py:ChatModel._prepare_converse_request
    """

    async def test_high_effort_reasoning_drops_max_tokens(
        self, request_log: EventLog
    ) -> None:
        """High reasoning effort strips 'maxTokens' from the inference config and warns.

        Only the token limit is removed: the ``reasoningConfig`` that triggered
        the conflict is still forwarded in ``additionalModelRequestFields``.
        """
        inference_cfg: InferenceConfigurationTypeDef = {"maxTokens": 1000}
        request = await _MODEL._prepare_converse_request(  # noqa: SLF001
            bedrock_messages=[],
            inference_cfg=inference_cfg,
            system_blocks=None,
            tool_config=None,
            additional_request_fields={
                "reasoningConfig": {"type": "enabled", "maxReasoningEffort": "high"}
            },
            service_tier=None,
        )
        assert "maxTokens" not in request["inferenceConfig"]
        assert request["additionalModelRequestFields"]["reasoningConfig"] == {
            "type": "enabled",
            "maxReasoningEffort": "high",
        }
        assert request_log["level"] == "warning"
        assert any(
            "max_tokens" in str(detail)
            for detail in request_log.get("error_detail", [])
        ), "the dropped token limit must be reported in the request log"

    async def test_medium_effort_reasoning_preserves_max_tokens(
        self, request_log: EventLog
    ) -> None:
        """Medium reasoning effort leaves 'maxTokens' untouched and does not warn."""
        inference_cfg: InferenceConfigurationTypeDef = {"maxTokens": 1000}
        request = await _MODEL._prepare_converse_request(  # noqa: SLF001
            bedrock_messages=[],
            inference_cfg=inference_cfg,
            system_blocks=None,
            tool_config=None,
            additional_request_fields={
                "reasoningConfig": {"type": "enabled", "maxReasoningEffort": "medium"}
            },
            service_tier=None,
        )
        assert request["inferenceConfig"]["maxTokens"] == 1000
        assert request_log["level"] == "info"
        assert "error_detail" not in request_log

    async def test_no_reasoning_preserves_max_tokens(
        self, request_log: EventLog
    ) -> None:
        """No reasoning config leaves 'maxTokens' untouched and does not warn."""
        inference_cfg: InferenceConfigurationTypeDef = {"maxTokens": 1000}
        request = await _MODEL._prepare_converse_request(  # noqa: SLF001
            bedrock_messages=[],
            inference_cfg=inference_cfg,
            system_blocks=None,
            tool_config=None,
            additional_request_fields={},
            service_tier=None,
        )
        assert request["inferenceConfig"]["maxTokens"] == 1000
        assert request_log["level"] == "info"
        assert "error_detail" not in request_log

    async def test_high_effort_reasoning_without_max_tokens_does_not_warn(
        self, request_log: EventLog
    ) -> None:
        """High reasoning effort with no 'maxTokens' set logs no warning."""
        inference_cfg: InferenceConfigurationTypeDef = {}
        request = await _MODEL._prepare_converse_request(  # noqa: SLF001
            bedrock_messages=[],
            inference_cfg=inference_cfg,
            system_blocks=None,
            tool_config=None,
            additional_request_fields={
                "reasoningConfig": {"type": "enabled", "maxReasoningEffort": "high"}
            },
            service_tier=None,
        )
        assert "maxTokens" not in request["inferenceConfig"]
        assert request_log["level"] == "info"
        assert "error_detail" not in request_log
