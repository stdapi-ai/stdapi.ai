"""Amazon Nova-specific behavior on the OpenAI /v1/responses route.

Web search (``nova_grounding``) and code_interpreter (``nova_code_interpreter``)
tests live in ``tests/test_openai_responses.py`` alongside the official API
equivalents so behavior parity can be validated in a single test run.

Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
     https://docs.aws.amazon.com/nova/latest/nova2-userguide/what-is-nova-2.html
     stdapi/models/chat/amazon_nova_2.py:ChatModel
"""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openai import OpenAI

#: Nova models supporting reasoning and nova_code_interpreter.
NOVA_ALL = ("amazon.nova-2-lite-v1:0",)


class TestNovaResponses:
    """Nova reasoning configuration driven by the Responses ``reasoning`` object.

    Ref: https://developers.openai.com/api/docs/guides/reasoning
         stdapi/models/chat/amazon_nova_2.py:ChatModel._req_configure_reasoning
    """

    @pytest.mark.parametrize("model", NOVA_ALL)
    def test_reasoning_effort(
        self, openai_client: OpenAI, use_official_api: bool, model: str
    ) -> None:
        """``reasoning.effort="low"`` yields a reasoning item before the message.

        The gateway maps the effort onto Nova's
        ``additionalModelRequestFields.reasoningConfig``
        (``{"type": "enabled", "maxReasoningEffort": "low"}``) and turns the
        ``reasoningContent`` blocks Bedrock returns into a ``reasoning`` output
        item carrying ``reasoning_text`` parts.
        ``usage.output_tokens_details.reasoning_tokens`` is not asserted: Bedrock
        Converse reports no separate reasoning-token count, so the gateway always
        emits 0 there.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlock.html
             stdapi/models/chat/_adapters/_openai_responses.py:_build_reasoning_item
        """
        if use_official_api:
            pytest.skip("Amazon Nova is not supported on the official API")
        resp = openai_client.responses.create(
            model=model, input="Reply with OK.", reasoning={"effort": "low"}
        )
        assert resp.status == "completed"
        assert resp.output_text
        assert resp.reasoning is not None
        assert resp.reasoning.effort == "low", "The request reasoning object is echoed"

        item_types = [item.type for item in resp.output]
        assert "message" in item_types, f"No message item in output: {item_types}"
        assert "reasoning" in item_types, (
            f"Expected a reasoning item with reasoning enabled, got: {item_types}"
        )
        assert item_types.index("reasoning") < item_types.index("message")
        reasoning_items = [item for item in resp.output if item.type == "reasoning"]
        assert reasoning_items[0].content, "Reasoning item carries no reasoning_text"
        assert reasoning_items[0].content[0].type == "reasoning_text"
        assert reasoning_items[0].content[0].text

    @pytest.mark.parametrize("model", NOVA_ALL)
    def test_reasoning_effort_none_explicit_disable(
        self, openai_client: OpenAI, use_official_api: bool, model: str
    ) -> None:
        """``reasoning.effort="none"`` disables reasoning, leaving no reasoning item.

        ``"none"`` is the only effort value the gateway turns into Nova's
        ``reasoningConfig: {"type": "disabled"}``; every other value (including a
        ``reasoning`` object with no ``effort``) enables reasoning, so the absence
        of a ``reasoning`` output item is what proves the disable was honored.

        Ref: https://developers.openai.com/api/docs/guides/reasoning
             stdapi/models/chat/_adapters/_openai_responses.py:extract_reasoning
        """
        if use_official_api:
            pytest.skip("Amazon Nova is not supported on the official API")
        resp = openai_client.responses.create(
            model=model, input="Reply with OK.", reasoning={"effort": "none"}
        )
        assert resp.status == "completed"
        assert resp.output_text
        assert resp.reasoning is not None
        assert resp.reasoning.effort == "none", "The request reasoning object is echoed"

        item_types = [item.type for item in resp.output]
        assert "message" in item_types, f"No message item in output: {item_types}"
        assert "reasoning" not in item_types, (
            f"Reasoning was disabled but a reasoning item was returned: {item_types}"
        )
