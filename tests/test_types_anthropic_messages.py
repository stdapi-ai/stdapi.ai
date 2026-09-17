"""Anthropic Messages type mirror against the Anthropic SDK types (no AWS calls).

The gateway never runs the ``web_fetch`` server tool itself, so a
``web_fetch_tool_result`` block can only arrive as conversation history replayed by a
client that talked to the Messages API. Every error code the SDK declares therefore
has to validate, otherwise the whole request is refused mid-conversation.

Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool
     https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/web_fetch_tool_result_error_code.py
     stdapi/types/anthropic_messages.py:WebFetchToolResultErrorCode
"""

from typing import get_args

import pytest
from anthropic.types import (
    WebFetchToolResultErrorCode as SdkWebFetchToolResultErrorCode,
)

from stdapi.types.anthropic_messages import (
    MessageCreateParams,
    WebFetchToolResultErrorBlock,
    WebFetchToolResultErrorCode,
)

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Every ``web_fetch`` tool-result error code the installed Anthropic SDK declares.
SDK_WEB_FETCH_ERROR_CODES = get_args(SdkWebFetchToolResultErrorCode)


class TestWebFetchToolResultErrorCode:
    """The ``web_fetch_tool_result_error`` code enumeration.

    Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool
         stdapi/types/anthropic_messages.py:WebFetchToolResultErrorCode
    """

    def test_codes_match_the_sdk_exactly(self) -> None:
        """The mirrored literal declares the same codes as the SDK.

        Ref: https://github.com/anthropics/anthropic-sdk-python/blob/main/src/anthropic/types/web_fetch_tool_result_error_code.py
             stdapi/types/anthropic_messages.py:WebFetchToolResultErrorCode
        """
        assert set(SDK_WEB_FETCH_ERROR_CODES)
        assert set(get_args(WebFetchToolResultErrorCode)) == set(
            SDK_WEB_FETCH_ERROR_CODES
        )

    @pytest.mark.parametrize("error_code", SDK_WEB_FETCH_ERROR_CODES)
    def test_a_replayed_error_block_is_accepted(self, error_code: str) -> None:
        """A request replaying a ``web_fetch`` error block validates for every code.

        The block is one arm of the ``MessageParam.content`` union, so an unknown
        code fails every arm and the entire request is rejected rather than the
        block alone.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool
             stdapi/types/anthropic_messages.py:WebFetchToolResultErrorBlockParam
        """
        params = MessageCreateParams.model_validate(
            {
                "model": "claude-x",
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "server_tool_use",
                                "id": "srvtoolu_1",
                                "name": "web_fetch",
                                "input": {"url": "https://example.com/alpha"},
                            },
                            {
                                "type": "web_fetch_tool_result",
                                "tool_use_id": "srvtoolu_1",
                                "content": {
                                    "type": "web_fetch_tool_result_error",
                                    "error_code": error_code,
                                },
                            },
                        ],
                    }
                ],
            }
        )

        content = params.messages[0].content
        assert not isinstance(content, str)
        assert content[1].content.error_code == error_code  # type: ignore[union-attr]

    @pytest.mark.parametrize("error_code", SDK_WEB_FETCH_ERROR_CODES)
    def test_a_returned_error_block_is_accepted(self, error_code: str) -> None:
        """The response-side block validates for every code the API can emit.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool
             stdapi/types/anthropic_messages.py:WebFetchToolResultErrorBlock
        """
        block = WebFetchToolResultErrorBlock.model_validate(
            {"type": "web_fetch_tool_result_error", "error_code": error_code}
        )

        assert block.error_code == error_code
