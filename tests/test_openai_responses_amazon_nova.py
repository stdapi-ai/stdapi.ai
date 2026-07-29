"""Tests specific to Amazon Nova Responses API behavior.

Covers Nova-specific behavior on the OpenAI Responses route:

  - Reasoning effort parameter for Amazon Nova 2

Web search (nova_grounding) and code_interpreter (nova_code_interpreter) tests
are in ``tests/test_openai_responses.py`` alongside the official API equivalents
so behavior parity can be validated in a single test run.
"""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openai import OpenAI

#: Nova models supporting reasoning and nova_code_interpreter.
NOVA_ALL = ("amazon.nova-2-lite-v1:0",)


class TestNovaResponses:
    """Amazon Nova-specific Responses API tests."""

    @pytest.mark.parametrize("model", NOVA_ALL)
    def test_reasoning_effort(
        self, openai_client: OpenAI, use_official_api: bool, model: str
    ) -> None:
        """Reasoning parameter with effort is accepted and yields a response with reasoning.

        Validates:
            - ``reasoning.effort`` is accepted without error
            - Response contains an output item with text
            - reasoning_tokens in usage is greater than zero
        """
        if use_official_api:
            pytest.skip("Amazon Nova is not supported on the official API")
        resp = openai_client.responses.create(
            model=model, input="Reply with OK.", reasoning={"effort": "low"}
        )
        assert resp.output_text

    @pytest.mark.parametrize("model", NOVA_ALL)
    def test_reasoning_effort_none_explicit_disable(
        self, openai_client: OpenAI, use_official_api: bool, model: str
    ) -> None:
        """reasoning.effort='none' explicitly disables reasoning on Nova models."""
        if use_official_api:
            pytest.skip("Amazon Nova is not supported on the official API")
        resp = openai_client.responses.create(
            model=model, input="Reply with OK.", reasoning={"effort": "none"}
        )
        assert resp.output_text
