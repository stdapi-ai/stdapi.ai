"""Anthropic messages → Bedrock Converse messages (no AWS calls).

Covers cache-point placement and the request-direction content-block mappings that
have no response-direction twin.

Ref: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
     https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CachePointBlock.html
     stdapi/models/chat/_adapters/_anthropic_message.py:_map_messages
"""

from __future__ import annotations

import re
from base64 import b64encode

import pytest
from pydantic import ValidationError

from stdapi.input_file import InputFile
from stdapi.models.chat._adapters._anthropic_message import _map_messages
from stdapi.types.anthropic_messages import (
    CacheControlEphemeralParam,
    DocumentBlockParam,
    FileSource,
    ImageBlockParam,
    MessageParam,
    PlainTextSourceParam,
    RedactedThinkingBlockParam,
    TextBlockParam,
    ThinkingBlockParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
    URLImageSource,
    URLPDFSource,
)

pytestmark = pytest.mark.local

#: Cache control applied to every content block under test.
_CACHE_CONTROL = CacheControlEphemeralParam()


def _tool_use_message() -> MessageParam:
    """Return an assistant message with a cache-controlled tool_use block."""
    return MessageParam(
        role="assistant",
        content=[
            ToolUseBlockParam(
                type="tool_use",
                id="toolu_1",
                name="lookup",
                input={},
                cache_control=_CACHE_CONTROL,
            )
        ],
    )


def _tool_result_message() -> MessageParam:
    """Return a user message with a cache-controlled tool_result block."""
    return MessageParam(
        role="user",
        content=[
            ToolResultBlockParam(
                type="tool_result",
                tool_use_id="toolu_1",
                content="result",
                cache_control=_CACHE_CONTROL,
            )
        ],
    )


def _text_message() -> MessageParam:
    """Return a user message with a cache-controlled text block."""
    return MessageParam(
        role="user",
        content=[TextBlockParam(type="text", text="hi", cache_control=_CACHE_CONTROL)],
    )


class TestMapMessagesAllowToolCaching:
    """``allow_tool_caching`` gates cache points on tool_use/tool_result blocks only.

    Some Bedrock models reject a ``cachePoint`` in a turn that also carries
    ``toolUse``/``toolResult`` blocks, so the gateway drops the breakpoint there
    instead of failing the request; ``cache_control`` is silently ignored, which
    upstream permits since caching never errors.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_messages
    """

    async def test_tool_caching_disallowed_skips_cache_point_on_tool_use(self) -> None:
        """No cachePoint follows a cache-controlled tool_use block when disallowed.

        The ``toolu_`` prefix is stripped because Bedrock ``toolUseId`` values carry
        no Anthropic prefix.
        """
        result = await _map_messages(
            [_tool_use_message()], allow_explicit_caching=True, allow_tool_caching=False
        )
        assert result[0]["content"] == [
            {"toolUse": {"toolUseId": "1", "name": "lookup", "input": {}}}
        ]

    async def test_tool_caching_disallowed_skips_cache_point_on_tool_result(
        self,
    ) -> None:
        """No cachePoint follows a cache-controlled tool_result block when disallowed."""
        result = await _map_messages(
            [_tool_result_message()],
            allow_explicit_caching=True,
            allow_tool_caching=False,
        )
        assert result[0]["content"] == [
            {"toolResult": {"toolUseId": "1", "content": [{"text": "result"}]}}
        ]

    async def test_tool_caching_disallowed_keeps_cache_point_on_text(self) -> None:
        """A cachePoint still follows a cache-controlled text block when tool caching is off.

        Only tool blocks are affected, so an ordinary text breakpoint survives and
        becomes a ``cachePoint`` element after the block it terminates.
        """
        result = await _map_messages(
            [_text_message()], allow_explicit_caching=True, allow_tool_caching=False
        )
        assert result[0]["content"] == [
            {"text": "hi"},
            {"cachePoint": {"type": "default"}},
        ]

    async def test_tool_caching_allowed_keeps_cache_point_on_tool_use(self) -> None:
        """A cachePoint follows a cache-controlled tool_use block when allowed (default)."""
        result = await _map_messages(
            [_tool_use_message()], allow_explicit_caching=True, allow_tool_caching=True
        )
        assert result[0]["content"] == [
            {"toolUse": {"toolUseId": "1", "name": "lookup", "input": {}}},
            {"cachePoint": {"type": "default"}},
        ]

    async def test_tool_caching_allowed_keeps_cache_point_on_tool_result(self) -> None:
        """A cachePoint follows a cache-controlled tool_result block when allowed (default)."""
        result = await _map_messages(
            [_tool_result_message()],
            allow_explicit_caching=True,
            allow_tool_caching=True,
        )
        assert result[0]["content"] == [
            {"toolResult": {"toolUseId": "1", "content": [{"text": "result"}]}},
            {"cachePoint": {"type": "default"}},
        ]

    async def test_s3_image_source_outside_the_allow_list_is_rejected(self) -> None:
        """An ``s3://`` image source is refused unless its bucket is configured.

        ``InputFile`` validates the bucket against the gateway's accepted-bucket
        set, so a caller cannot make the server read an arbitrary bucket its role
        happens to have access to.

        Ref: stdapi/input_file.py:InputFile
        """
        with pytest.raises(ValidationError) as excinfo:
            ImageBlockParam.model_validate(
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "s3://not-an-allowed-bucket/picture.png",
                    },
                }
            )
        assert "S3 bucket not allowed" in str(excinfo.value)
        assert "not-an-allowed-bucket" in str(excinfo.value)


class TestMapMessagesThinkingBlocks:
    """Assistant thinking history is replayed to Bedrock as ``reasoningContent``.

    Bedrock validates the reasoning it previously emitted when it is sent back, so
    a follow-up turn only works if the redacted payload is decoded to the exact
    bytes Bedrock produced and the signature is echoed unchanged.

    Ref: https://platform.claude.com/docs/en/build-with-claude/extended-thinking
         https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlock.html
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_messages
    """

    async def test_redacted_thinking_param_is_decoded_to_raw_bytes(self) -> None:
        """A ``redacted_thinking`` block becomes ``reasoningContent.redactedContent`` bytes.

        ``_map_content_block_to_bedrock`` returns ``None`` for this block type, so
        the decode happens in ``_map_messages`` itself: a regression there drops the
        block entirely and every turn after a redacted one is rejected by Bedrock.
        """
        payload = b"\x00\x01redacted-bytes\xff"
        result = await _map_messages(
            [
                MessageParam(
                    role="assistant",
                    content=[
                        RedactedThinkingBlockParam(
                            type="redacted_thinking",
                            data=b64encode(payload).decode("ascii"),
                        )
                    ],
                )
            ],
            allow_explicit_caching=False,
            allow_tool_caching=False,
        )
        assert result[0]["content"] == [
            {"reasoningContent": {"redactedContent": payload}}
        ]

    async def test_thinking_param_keeps_its_signature(self) -> None:
        """A ``thinking`` block becomes ``reasoningContent.reasoningText`` with its signature.

        The signature must survive byte-identical: Bedrock rejects a replayed
        reasoning block whose signature does not match the text.
        """
        result = await _map_messages(
            [
                MessageParam(
                    role="assistant",
                    content=[
                        ThinkingBlockParam(
                            type="thinking", thinking="step one", signature="sig-abc"
                        )
                    ],
                )
            ],
            allow_explicit_caching=False,
            allow_tool_caching=False,
        )
        assert result[0]["content"] == [
            {
                "reasoningContent": {
                    "reasoningText": {"text": "step one", "signature": "sig-abc"}
                }
            }
        ]

    async def test_thinking_param_without_signature_omits_the_key(self) -> None:
        """A signature-less ``thinking`` block emits no ``signature`` key.

        Bedrock rejects an empty signature string, so the field is left out rather
        than sent as ``""``.
        """
        result = await _map_messages(
            [
                MessageParam(
                    role="assistant",
                    content=[ThinkingBlockParam(type="thinking", thinking="step one")],
                )
            ],
            allow_explicit_caching=False,
            allow_tool_caching=False,
        )
        assert result[0]["content"] == [
            {"reasoningContent": {"reasoningText": {"text": "step one"}}}
        ]


class TestMapMessagesDocumentName:
    """The Anthropic document ``title`` is sanitized into a legal Bedrock document name.

    Bedrock's ``DocumentBlock.name`` accepts only alphanumerics, hyphens and
    underscores and is length-capped, while Anthropic's ``title`` is free text, so
    a real-world file name would otherwise fail the whole request.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html
         https://platform.claude.com/docs/en/api/messages
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_document_to_bedrock
    """

    @staticmethod
    async def _document_name(title: str | None) -> str:
        """Return the Bedrock document name produced for *title*."""
        result = await _map_messages(
            [
                MessageParam(
                    role="user",
                    content=[
                        DocumentBlockParam(
                            type="document",
                            title=title,
                            source=PlainTextSourceParam(
                                type="text", media_type="text/plain", data="body"
                            ),
                        )
                    ],
                )
            ],
            allow_explicit_caching=False,
            allow_tool_caching=False,
        )
        (block,) = result[0]["content"]
        return block["document"]["name"]

    async def test_punctuation_and_spaces_are_replaced(self) -> None:
        """Every character outside ``[a-zA-Z0-9_-]`` becomes an underscore."""
        name = await self._document_name("Q3 report (final).pdf")
        assert name == "Q3_report__final__pdf"
        assert re.fullmatch(r"[A-Za-z0-9_-]+", name)

    async def test_long_title_is_truncated_to_200_characters(self) -> None:
        """A title longer than Bedrock's name limit is truncated to 200 characters."""
        name = await self._document_name("Q3 report (final)!" * 30)
        assert len(name) == 200
        assert re.fullmatch(r"[A-Za-z0-9_-]{1,200}", name)

    async def test_missing_title_falls_back_to_document(self) -> None:
        """A document with no ``title`` is named ``document``.

        ``title`` is optional upstream but ``name`` is required by Bedrock.
        """
        assert await self._document_name(None) == "document"


class TestMapMessagesRemoteSources:
    """URL, S3 and Files API sources are routed to the deferred file resolver.

    Only base64 sources carry their bytes inline; every other source is an
    ``InputFile`` that builds its Bedrock block through
    ``to_bedrock_content_block``, whose payload is resolved later in the request.
    A regression in that dispatch drops the image or document from the prompt
    while the model still answers plausibly from the surrounding text.

    Ref: https://platform.claude.com/docs/en/api/messages
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_image_to_bedrock
         stdapi/models/chat/_adapters/_anthropic_message.py:_map_document_to_bedrock
    """

    @staticmethod
    @pytest.fixture
    def recorded_sources(monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Record every ``InputFile`` the mapper asks for a Bedrock content block."""
        recorded: list[str] = []

        async def _fake_block(
            self: InputFile, *_args: object, **_kwargs: object
        ) -> dict[str, object]:
            recorded.append(str(self))
            return {"image": {"format": "png", "source": {}}}

        monkeypatch.setattr(InputFile, "to_bedrock_content_block", _fake_block)
        return recorded

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/picture.png",
            "file-id:file_0123456789abcdef0123456789abcdef",
        ],
        ids=["https", "file-id"],
    )
    async def test_url_image_source_is_deferred_to_the_file_resolver(
        self, url: str, recorded_sources: list[str]
    ) -> None:
        """Every ``{"type": "url"}`` image source form reaches the file resolver.

        ``URLImageSource.url`` is an ``InputFile``, so HTTPS URLs and ``file-id:``
        references share the one branch.
        """
        block = ImageBlockParam.model_validate(
            {"type": "image", "source": {"type": "url", "url": url}}
        )
        assert isinstance(block.source, URLImageSource)
        result = await _map_messages(
            [MessageParam(role="user", content=[block])],
            allow_explicit_caching=False,
            allow_tool_caching=False,
        )
        assert "image" in result[0]["content"][0]
        assert len(recorded_sources) == 1

    async def test_file_image_source_is_deferred_to_the_file_resolver(
        self, recorded_sources: list[str]
    ) -> None:
        """A ``{"type": "file", "file_id": ...}`` image source reaches the file resolver.

        The Files API source is its own union member, distinct from the ``url``
        one, and resolves the stored object rather than a remote URL.
        """
        block = ImageBlockParam.model_validate(
            {
                "type": "image",
                "source": {
                    "type": "file",
                    "file_id": "file_0123456789abcdef0123456789abcdef",
                },
            }
        )
        assert isinstance(block.source, FileSource)
        result = await _map_messages(
            [MessageParam(role="user", content=[block])],
            allow_explicit_caching=False,
            allow_tool_caching=False,
        )
        assert "image" in result[0]["content"][0]
        assert len(recorded_sources) == 1

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/report.pdf",
            "file-id:file_0123456789abcdef0123456789abcdef",
        ],
        ids=["https", "file-id"],
    )
    async def test_url_pdf_document_source_is_deferred_to_the_file_resolver(
        self, url: str, recorded_sources: list[str]
    ) -> None:
        """Every ``{"type": "url"}`` PDF document source form reaches the file resolver.

        The document branch additionally passes the sanitized title as the Bedrock
        document name, so it cannot share the image call site.
        """
        block = DocumentBlockParam.model_validate(
            {
                "type": "document",
                "title": "Q3 report",
                "source": {"type": "url", "url": url},
            }
        )
        assert isinstance(block.source, URLPDFSource)
        await _map_messages(
            [MessageParam(role="user", content=[block])],
            allow_explicit_caching=False,
            allow_tool_caching=False,
        )
        assert len(recorded_sources) == 1

    async def test_s3_source_outside_the_allow_list_is_rejected(self) -> None:
        """An ``s3://`` source is refused unless its bucket is configured as accepted.

        ``InputFile`` validates the bucket against the gateway's accepted-bucket
        set, so a caller cannot make the server read an arbitrary bucket the task
        role happens to have access to.

        Ref: stdapi/input_file.py:InputFile
        """
        with pytest.raises(ValidationError) as excinfo:
            ImageBlockParam.model_validate(
                {
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": "s3://not-an-allowed-bucket/picture.png",
                    },
                }
            )
        assert "S3 bucket not allowed" in str(excinfo.value)
        assert "not-an-allowed-bucket" in str(excinfo.value)
