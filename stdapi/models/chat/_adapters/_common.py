"""Bedrock helpers shared by every chat adapter, regardless of the client API shape."""

from typing import TYPE_CHECKING, Any

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock import filter_extra_model_parameters

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


def inference_extras(
    extras: dict[str, Any] | None, reserved: frozenset[str]
) -> dict[str, Any]:
    """Filter and validate request extras forwarded as provider-specific inference fields.

    Args:
        extras: Undeclared request keys, or ``None`` when the request has none.
        reserved: Argument names ``set_inference_configuration`` already binds for
            the calling adapter, which differ with the parameters its API exposes.

    Returns:
        The extras, with LiteLLM client-control parameters dropped
        (:func:`stdapi.aws_bedrock.filter_extra_model_parameters`).

    Raises:
        ApiError: If a remaining extra reuses a ``set_inference_configuration``
            argument name, which would bind twice and fail the call.
    """
    extras = filter_extra_model_parameters(extras)
    if conflicting := sorted(reserved.intersection(extras)):
        names = ", ".join(f"'{name}'" for name in conflicting)
        msg = f"Unsupported parameter: {names} cannot be sent as a model extra."
        raise ApiError(msg)
    return extras
