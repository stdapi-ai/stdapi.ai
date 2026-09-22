"""A Responses stream's ``error`` event carries the error both ways the API sends it.

The live API nests the error under ``error``, which makes the official SDK raise
it; the API reference documents flat ``code``/``message``/``param`` fields. The
gateway sends both, so a client written against either sees the failure.

Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
     stdapi/models/chat/_adapters/_openai_responses.py:_failure_events
"""

from contextlib import suppress
from json import loads
from typing import TYPE_CHECKING, Any

import pytest
from openai import APIError
from openai.types.responses import ResponseErrorEvent as SdkResponseErrorEvent

from stdapi.api_errors import ApiError
from stdapi.models.chat._adapters._openai_responses import format_failed_stream
from stdapi.models.chat._adapters._responses_context import ContextLengthExceededError
from stdapi.monitoring import SseHandledStreamError
from stdapi.types.openai_responses import ResponseCreateParams

if TYPE_CHECKING:
    from openai import OpenAI

pytestmark = pytest.mark.usefixtures("request_log")

#: Per lane (official API or not): the cheapest Responses model with the smallest window, and that window.
_OVERFLOW_MODELS: dict[bool, tuple[str, int]] = {
    # 128k output tokens are reserved from gpt-5-nano's 400k window.
    True: ("gpt-5-nano", 272_000),
    False: ("meta.llama3-8b-instruct-v1:0", 8_192),
}


async def _error_event(exc: Exception) -> dict[str, Any]:
    """Return the ``error`` event a stream failing on *exc* sends.

    Args:
        exc: The failure.

    Returns:
        The event payload.
    """
    request = ResponseCreateParams(model="m", input="hello", stream=True)
    payloads: list[dict[str, Any]] = []
    with suppress(SseHandledStreamError):
        async for event in format_failed_stream("resp-1", 0.0, "m", request, exc):
            payloads.append(loads(str(event.data)))  # noqa: PERF401 - kept up to a failure
    return next(payload for payload in payloads if payload["type"] == "error")


@pytest.mark.local
class TestErrorEventShape:
    """Both shapes carry the same error.

    Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
    """

    async def test_the_error_is_nested_and_flat(self) -> None:
        """The nested object and the flat fields agree.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
        """
        event = await _error_event(ContextLengthExceededError())
        assert event["code"] == event["error"]["code"] == "context_length_exceeded"
        assert event["param"] == event["error"]["param"] == "input"
        assert event["message"] == event["error"]["message"]
        assert event["error"]["type"] == "invalid_request_error"
        parsed = SdkResponseErrorEvent.model_validate(event)
        assert parsed.code == "context_length_exceeded"

    async def test_a_server_failure_is_a_server_error(self) -> None:
        """A failure on the gateway's side is typed as the API types it.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
        """
        event = await _error_event(ApiError("The model failed.", status=502))
        assert event["error"]["type"] == "server_error"
        assert event["error"].get("code") is None


class TestOfficialClient:
    """The official SDK raises a failed Responses stream, on either target.

    Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
    """

    @pytest.mark.slow
    def test_the_sdk_raises_a_streamed_failure(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """A stream refused for its size raises ``context_length_exceeded``.

        Rejected before generation, so it costs nothing; ``slow`` for the upload.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
        """
        model, window = _OVERFLOW_MODELS[use_official_api]
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
        prompt = " ".join(words[i % len(words)] for i in range(window * 5 // 4))
        with pytest.raises(APIError) as excinfo:  # noqa: PT012
            stream = openai_client.responses.create(
                model=model, input=prompt, max_output_tokens=16, stream=True
            )
            for _ in stream:
                pass
        assert excinfo.value.code == "context_length_exceeded"
        assert excinfo.value.param == "input"
