"""TwelveLabs Pegasus chat model: finishReason to Bedrock stopReason mapping.

Pegasus answers on InvokeModel with ``{"message": …, "finishReason": …}``; the model
class re-shapes that into a Converse response so the shared adapters can format it.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
     stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._converse
"""

from typing import TYPE_CHECKING, Any

import pytest

from stdapi.models import InvokeResult
from stdapi.models.chat.twelvelabs_pegasus import _STOP_MAP, ChatModel

if TYPE_CHECKING:
    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_runtime.type_defs import ConverseResponseTypeDef

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Bedrock stopReason values reachable from a Pegasus finishReason.
_EXPECTED_STOP_REASONS = frozenset({"end_turn", "max_tokens"})


async def _converse(
    monkeypatch: pytest.MonkeyPatch, response: dict[str, Any]
) -> ConverseResponseTypeDef:
    """Run ChatModel._converse with AWS calls stubbed out; return the Converse response.

    Args:
        monkeypatch: Fixture used to stub region prep, body building and invocation.
        response: Stub Pegasus InvokeModel response body.

    Returns:
        The Converse-shaped response built from the stubbed Pegasus body.
    """
    model = ChatModel("twelvelabs.pegasus-1-2-v1:0")

    async def _noop_prepare(
        _request: ConverseRequestBaseTypeDef, _region: RegionName
    ) -> None:
        return None

    async def _stub_build_body(
        _request: ConverseRequestBaseTypeDef, _region: RegionName
    ) -> tuple[dict[str, Any], None, None]:
        return {}, None, None

    async def _stub_invoke(
        _body: dict[str, Any],
        **_kwargs: Any,  # noqa: ANN401
    ) -> InvokeResult[Any]:
        return InvokeResult(response=response)

    monkeypatch.setattr(
        type(model), "_prepare_converse_request_for_region", staticmethod(_noop_prepare)
    )
    monkeypatch.setattr(
        type(model), "_build_pegasus_body", staticmethod(_stub_build_body)
    )
    monkeypatch.setattr(type(model), "invoke", staticmethod(_stub_invoke))

    return await model._converse(  # noqa: SLF001
        {"modelId": "", "messages": []}, "us-east-1", single_region=False
    )


class TestPegasusStopReasonMapping:
    """ChatModel._converse maps Pegasus finishReason to a Bedrock stopReason."""

    async def test_unknown_finish_reason_falls_back_to_end_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unrecognized finishReason value maps to ``end_turn``.

        ``_STOP_MAP`` only covers Pegasus' two documented values, so any value the
        service adds later degrades to the neutral Converse stop reason instead of
        leaking a non-Bedrock literal into the response.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/models/chat/twelvelabs_pegasus.py:_STOP_MAP
        """
        result = await _converse(
            monkeypatch, {"message": "hi", "finishReason": "brand_new_reason"}
        )
        assert result["stopReason"] == "end_turn"

    async def test_missing_finish_reason_defaults_to_stop_mapping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A response with no finishReason key is treated as Pegasus' ``stop``.

        ``end_turn`` is also the unknown-value fallback, so this asserts the mapped
        value of ``stop`` rather than a literal.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._converse
        """
        result = await _converse(monkeypatch, {"message": "hi"})
        assert result["stopReason"] == _STOP_MAP["stop"]

    @pytest.mark.parametrize(("finish_reason", "expected"), sorted(_STOP_MAP.items()))
    async def test_known_finish_reasons_map_to_documented_stop_reason(
        self, monkeypatch: pytest.MonkeyPatch, finish_reason: str, expected: str
    ) -> None:
        """Each documented Pegasus finishReason becomes a Bedrock stopReason.

        The response body is also re-shaped: the flat Pegasus ``message`` string becomes
        a single assistant text content block, and usage is derived from the invocation
        metrics rather than from the model body.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
             stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._converse
        """
        result = await _converse(
            monkeypatch, {"message": "hi", "finishReason": finish_reason}
        )
        assert result["stopReason"] == expected
        assert result["stopReason"] in _EXPECTED_STOP_REASONS, (
            f"{expected!r} is not a Converse stopReason Pegasus can produce"
        )
        message = result["output"]["message"]
        assert message["role"] == "assistant"
        assert message["content"] == [{"text": "hi"}]
        assert result["usage"] == {
            "inputTokens": 0,
            "outputTokens": 0,
            "totalTokens": 0,
        }
