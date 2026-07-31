"""Bedrock helpers shared by every chat adapter, regardless of the client API shape."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime.literals import ConversationRoleType
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ContentBlockTypeDef,
        MessageTypeDef,
    )


def append_or_merge(
    bedrock_messages: list[MessageTypeDef],
    role: ConversationRoleType,
    blocks: list[ContentBlockTypeDef],
) -> None:
    """Append ``blocks`` to ``bedrock_messages`` under ``role``, merging with the last.

    Merges into the trailing message when its ``role`` matches; otherwise appends
    a new message.  Bedrock requires strictly alternating user/assistant turns and
    rejects empty content, so empty ``blocks`` lists are a no-op.

    Args:
        bedrock_messages: Mutable Bedrock messages list to append to.
        role: Bedrock role of the blocks to append.
        blocks: Content blocks to append.
    """
    if not blocks:
        return
    if bedrock_messages and bedrock_messages[-1]["role"] == role:
        bedrock_messages[-1]["content"] += blocks  # type: ignore[operator]
    else:
        bedrock_messages.append({"role": role, "content": blocks})
