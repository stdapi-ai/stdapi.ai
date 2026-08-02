"""TwelveLabs Pegasus chat model: Converse ⇄ InvokeModel re-shaping (no AWS calls).

Pegasus answers on InvokeModel with ``{"message": …, "finishReason": …}`` and takes a
flat ``{inputPrompt, mediaSource}`` body; the model class re-shapes the Converse
request and response around it so the shared adapters can serve every route.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
     stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._converse
"""

from base64 import b64encode
from typing import TYPE_CHECKING, Any

import pytest

from stdapi.aws import AWS_ENVIRONMENT
from stdapi.models import InvokeResult
from stdapi.models.chat.twelvelabs_pegasus import (
    _PEGASUS_INLINE_BYTES,
    _STOP_MAP,
    ChatModel,
    _extract_latest_user_text,
    _extract_latest_video,
    _video_to_media_source,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ConverseResponseTypeDef,
        MessageTypeDef,
    )

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Bedrock stopReason values reachable from a Pegasus finishReason.
_EXPECTED_STOP_REASONS = frozenset({"end_turn", "max_tokens"})

#: Account ID substituted for the caller's when an S3 video omits its bucket owner.
_ACCOUNT_ID = "123456789012"


def _video_message(uri: str) -> MessageTypeDef:
    """Return a user message carrying one S3-located Bedrock video block."""
    return {
        "role": "user",
        "content": [
            {"video": {"format": "mp4", "source": {"s3Location": {"uri": uri}}}}
        ],
    }


def _text_message(role: str, text: str) -> MessageTypeDef:
    """Return a single-text-block Bedrock message."""
    return {"role": role, "content": [{"text": text}]}  # type: ignore[typeddict-item]


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


class TestPegasusConversationFlattening:
    """A Converse conversation collapses into Pegasus' one video and one prompt.

    Pegasus takes no message history: the model class picks the most recent video
    and the trailing run of user text.  Getting either selection wrong sends the
    caller a plausible answer about the wrong video or the wrong question.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
         stdapi/models/chat/twelvelabs_pegasus.py:_extract_latest_video
         stdapi/models/chat/twelvelabs_pegasus.py:_extract_latest_user_text
    """

    def test_latest_video_wins_over_earlier_ones(self) -> None:
        """The most recent video block in the conversation is the one analysed.

        The walk is in reverse over both the messages and their content blocks, so
        the last block of the last video-bearing message wins.
        """
        video = _extract_latest_video(
            [
                _video_message("s3://bucket/first.mp4"),
                _text_message("assistant", "Seen."),
                {
                    "role": "user",
                    "content": [
                        {
                            "video": {
                                "format": "mp4",
                                "source": {"s3Location": {"uri": "s3://bucket/b.mp4"}},
                            }
                        },
                        {
                            "video": {
                                "format": "mp4",
                                "source": {"s3Location": {"uri": "s3://bucket/c.mp4"}},
                            }
                        },
                    ],
                },
            ]
        )
        assert video is not None
        assert video["source"]["s3Location"]["uri"] == "s3://bucket/c.mp4"

    def test_no_video_returns_none(self) -> None:
        """A conversation with no video block yields ``None``.

        ``_build_pegasus_body`` turns that into a 400 rather than calling Pegasus
        with no media.
        """
        assert _extract_latest_video([_text_message("user", "Describe it.")]) is None

    def test_prompt_is_the_trailing_run_of_user_text_in_order(self) -> None:
        """Only the trailing user turns contribute, rejoined in reading order.

        The walk is reversed and the collected parts are reversed back, so the
        prompt reads in conversation order; an earlier user turn separated by an
        assistant reply belongs to a different question and is excluded.
        """
        assert (
            _extract_latest_user_text(
                [
                    _text_message("user", "Old question."),
                    _text_message("assistant", "Old answer."),
                    _text_message("user", "First."),
                    _text_message("user", "Second."),
                ]
            )
            == "First.\nSecond."
        )

    def test_prompt_is_empty_when_the_last_turn_is_not_a_user_one(self) -> None:
        """A conversation ending on an assistant turn yields no prompt text."""
        assert _extract_latest_user_text([_text_message("assistant", "Hi.")]) == ""


class TestPegasusMediaSource:
    """Resolved Bedrock video sources translate to Pegasus ``mediaSource`` values.

    Pegasus accepts inline base64 or an S3 reference, and an S3 reference must
    name its bucket owner; small videos stay inline so no bucket is touched.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
         stdapi/models/chat/twelvelabs_pegasus.py:_video_to_media_source
    """

    async def test_s3_source_keeps_its_uri_and_bucket_owner(self) -> None:
        """An ``s3Location`` is forwarded with the caller's declared bucket owner."""
        source = await _video_to_media_source(
            {
                "source": {
                    "s3Location": {
                        "uri": "s3://bucket/clip.mp4",
                        "bucketOwner": "210987654321",
                    }
                }
            },
            "us-east-1",
        )
        assert source == {
            "s3Location": {"uri": "s3://bucket/clip.mp4", "bucketOwner": "210987654321"}
        }

    async def test_s3_source_without_owner_defaults_to_the_gateway_account(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing ``bucketOwner`` falls back to the account the gateway runs in.

        Pegasus requires the field; omitting it would fail the call, and the
        gateway's own account is the only one it can vouch for.
        """
        monkeypatch.setitem(AWS_ENVIRONMENT, "account_id", _ACCOUNT_ID)
        source = await _video_to_media_source(
            {"source": {"s3Location": {"uri": "s3://bucket/clip.mp4"}}}, "us-east-1"
        )
        assert source["s3Location"]["bucketOwner"] == _ACCOUNT_ID

    async def test_small_inline_video_is_sent_as_base64(self) -> None:
        """Raw bytes under the inline limit are base64-encoded into the body.

        Staging them in S3 instead would cost a bucket write on every short clip.
        """
        source = await _video_to_media_source(
            {"source": {"bytes": b"MP4DATA"}}, "us-east-1"
        )
        assert source == {"base64String": b64encode(b"MP4DATA").decode()}

    async def test_oversized_video_is_staged_in_s3(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bytes over the inline limit are uploaded and referenced by URI.

        Pegasus caps the inline payload at a 25 MB base64 string, so anything
        larger must travel through a temporary S3 object instead of being
        truncated or rejected.
        """
        monkeypatch.setitem(AWS_ENVIRONMENT, "account_id", _ACCOUNT_ID)
        uploaded: list[tuple[int, str, bool]] = []

        class _Uploaded:
            """Stand-in for the uploaded-object handle."""

            uri = "s3://staging/clip.mp4"

        async def _fake_put(
            data: bytes,
            content_type: str,
            *,
            region: RegionName,  # noqa: ARG001
            temporary: bool,
        ) -> _Uploaded:
            uploaded.append((len(data), content_type, temporary))
            return _Uploaded()

        monkeypatch.setattr(
            "stdapi.models.chat.twelvelabs_pegasus.put_s3_object", _fake_put
        )
        raw = b"\0" * (_PEGASUS_INLINE_BYTES + 1)
        source = await _video_to_media_source({"source": {"bytes": raw}}, "us-east-1")
        assert source == {
            "s3Location": {"uri": "s3://staging/clip.mp4", "bucketOwner": _ACCOUNT_ID}
        }
        assert uploaded == [(len(raw), "video/mp4", True)], (
            "the staged object must be marked temporary so it is reaped"
        )


class TestPegasusStreamTranslation:
    """Pegasus' delta-style stream becomes a Bedrock ConverseStream event sequence.

    Every route's streaming adapter consumes ConverseStream events, so the
    synthesized sequence must be well-formed: one ``messageStart``, deltas on a
    single block, then ``contentBlockStop``/``messageStop`` and a final
    ``metadata`` usage event.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStream.html
         stdapi/models/chat/twelvelabs_pegasus.py:ChatModel._format_converse_stream
    """

    @staticmethod
    async def _chunks(chunks: list[dict[str, Any]]) -> AsyncGenerator[Any]:
        """Yield pre-built Pegasus streaming chunks as a fake raw stream."""
        for chunk in chunks:
            yield chunk

    async def test_stream_is_translated_into_a_single_text_block(self) -> None:
        """Text deltas share block index 0 and the stop reason is mapped.

        Pegasus emits no block framing of its own, so the wrapper opens the
        message, keeps every delta on one block and closes it before
        ``messageStop`` — an unclosed block would strand the downstream adapters.
        """
        model = ChatModel("twelvelabs.pegasus-1-2-v1:0")
        events = [
            event
            async for event in model._format_converse_stream(  # noqa: SLF001
                self._chunks(
                    [
                        {"message": "Hel", "stopReason": ""},
                        {"message": "lo", "stopReason": ""},
                        {
                            "message": "",
                            "stopReason": "length",
                            "amazon-bedrock-invocationMetrics": {
                                "inputTokenCount": 0,
                                "outputTokenCount": 12,
                            },
                        },
                    ]
                )
            )
        ]
        assert [next(iter(event)) for event in events] == [
            "messageStart",
            "contentBlockDelta",
            "contentBlockDelta",
            "contentBlockStop",
            "messageStop",
            "metadata",
        ]
        assert events[0]["messageStart"]["role"] == "assistant"
        assert [
            event["contentBlockDelta"]["delta"]["text"]
            for event in events
            if "contentBlockDelta" in event
        ] == ["Hel", "lo"]
        assert {
            event["contentBlockDelta"]["contentBlockIndex"]
            for event in events
            if "contentBlockDelta" in event
        } == {0}
        assert events[4]["messageStop"]["stopReason"] == "max_tokens"
        assert events[5]["metadata"]["usage"] == {
            "inputTokens": 0,
            "outputTokens": 12,
            "totalTokens": 12,
        }

    async def test_stream_without_metrics_reports_zero_usage(self) -> None:
        """A stream carrying no invocation metrics still ends with a usage event.

        The metadata event is what the routes read to build their usage block, so
        it must be emitted even when Pegasus reports nothing.
        """
        model = ChatModel("twelvelabs.pegasus-1-2-v1:0")
        events = [
            event
            async for event in model._format_converse_stream(  # noqa: SLF001
                self._chunks([{"message": "hi", "stopReason": "stop"}])
            )
        ]
        assert events[-1]["metadata"]["usage"] == {
            "inputTokens": 0,
            "outputTokens": 0,
            "totalTokens": 0,
        }
        assert events[-2]["messageStop"]["stopReason"] == "end_turn"
