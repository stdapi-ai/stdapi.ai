"""TwelveLabs Pegasus chat model: finishReason to Bedrock stopReason mapping."""

from typing import TYPE_CHECKING, Any

import pytest

from stdapi.models import InvokeResult
from stdapi.models.chat.twelvelabs_pegasus import _STOP_MAP, ChatModel

if TYPE_CHECKING:
    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


async def _converse_stop_reason(
    monkeypatch: pytest.MonkeyPatch, response: dict[str, Any]
) -> str:
    """Run ChatModel._converse with AWS calls stubbed out; return the resulting stopReason.

    Args:
        monkeypatch: Fixture used to stub region prep, body building and invocation.
        response: Stub Pegasus InvokeModel response body.

    Returns:
        The "stopReason" value from the formatted Converse response.
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

    monkeypatch.setattr(model, "_prepare_converse_request_for_region", _noop_prepare)
    monkeypatch.setattr(model, "_build_pegasus_body", _stub_build_body)
    monkeypatch.setattr(model, "invoke", _stub_invoke)

    result = await model._converse(  # noqa: SLF001
        {"modelId": "", "messages": []}, "us-east-1", single_region=False
    )
    return result["stopReason"]


class TestPegasusStopReasonMapping:
    """ChatModel._converse maps Pegasus finishReason to a Bedrock stopReason."""

    async def test_unknown_finish_reason_falls_back_to_end_turn(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unrecognized finishReason value maps to 'end_turn'."""
        stop_reason = await _converse_stop_reason(
            monkeypatch, {"message": "hi", "finishReason": "brand_new_reason"}
        )
        assert stop_reason == "end_turn"

    async def test_missing_finish_reason_defaults_to_stop_mapping(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A response with no finishReason key defaults to the 'stop' mapping."""
        stop_reason = await _converse_stop_reason(monkeypatch, {"message": "hi"})
        assert stop_reason == "end_turn"

    @pytest.mark.parametrize(("finish_reason", "expected"), sorted(_STOP_MAP.items()))
    async def test_known_finish_reasons_map_to_documented_stop_reason(
        self, monkeypatch: pytest.MonkeyPatch, finish_reason: str, expected: str
    ) -> None:
        """Each _STOP_MAP key maps to its documented Bedrock stop reason."""
        stop_reason = await _converse_stop_reason(
            monkeypatch, {"message": "hi", "finishReason": finish_reason}
        )
        assert stop_reason == expected
