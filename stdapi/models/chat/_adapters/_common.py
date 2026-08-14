"""Bedrock helpers shared by every chat adapter, regardless of the client API shape."""

from typing import TYPE_CHECKING, Any

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock import filter_extra_model_parameters
from stdapi.config import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from types_aiobotocore_bedrock_runtime.literals import ConversationRoleType
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ContentBlockTypeDef,
        MessageTypeDef,
    )

#: Extra model parameter choosing whether a request's searches may reach the external web.
EXTERNAL_WEB_ACCESS_PARAM = "external_web_access"


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


def reject_unsupported_web_search_fields(fields: Mapping[str, Any]) -> None:
    """Refuse a web search tool restricting a search the model cannot restrict.

    Shared by every client API shape: a model whose web search is a bare
    system tool takes no parameters, so a restriction sent with it would be
    dropped and the search would run wider than the caller asked for.

    Args:
        fields: Restriction fields keyed by the name the request spells them
            with; an unset one holds ``None`` or an empty value.

    Raises:
        ApiError: If any of *fields* is set.
    """
    if not (named := [name for name, value in fields.items() if value]):
        return
    plural = len(named) > 1
    pronoun = "them" if plural else "it"
    msg = (
        f"web_search {', '.join(named)} {'are' if plural else 'is'} not supported "
        f"by this model. Remove {pronoun}, or use a model whose web search "
        f"supports {pronoun}."
    )
    raise ApiError(msg)


def resolve_external_web_access(requested: object, *, per_request: bool) -> bool:
    """Resolve the web access a request's searches run with.

    Web access decides whether a search leaves the AWS boundary, so it is the
    operator's to set: the request only wins where the backend serving it
    applies a per-request value *and* the operator allows the override.
    Anything else asking for a different value is refused rather than silently
    given the configured one.

    Args:
        requested: Value the request sent, or ``None`` when it sent none.
        per_request: Whether the backend serving this request takes a web
            access choice per request at all.

    Returns:
        Web access to apply to this request's searches.

    Raises:
        ApiError: When the request asks for a value it cannot be given, or one
            that is not a boolean.
    """
    configured = SETTINGS.aws_bedrock_external_web_access
    if requested is None:
        return configured
    if not isinstance(requested, bool):
        msg = f"'{EXTERNAL_WEB_ACCESS_PARAM}' must be a boolean."
        raise ApiError(msg, status=400)
    if requested == configured:
        return configured
    state = "enabled" if configured else "disabled"
    fallback = (
        f"Remove the parameter, or set it to {str(configured).lower()}, which "
        "is what this server searches with."
    )
    if not SETTINGS.aws_bedrock_allow_external_web_access_override:
        msg = (
            f"'{EXTERNAL_WEB_ACCESS_PARAM}' cannot be changed on this server: "
            f"external web access is {state}. {fallback}"
        )
        raise ApiError(msg, status=400)
    if not per_request:
        msg = (
            f"'{EXTERNAL_WEB_ACCESS_PARAM}' is not available with this model: "
            f"external web access is {state}. {fallback} Use a model whose web "
            "search takes its own web access to choose per request."
        )
        raise ApiError(msg, status=400)
    return requested


def inference_extras(
    extras: dict[str, Any] | None, reserved: frozenset[str]
) -> dict[str, Any]:
    """Filter and validate request extras forwarded as provider-specific inference fields.

    ``external_web_access`` never travels to the model: it is the operator's
    web access control rather than an inference parameter, so it is resolved
    here and removed.

    Args:
        extras: Undeclared request keys, or ``None`` when the request has none.
        reserved: Argument names ``set_inference_configuration`` already binds for
            the calling adapter, which differ with the parameters its API exposes.

    Returns:
        The extras, with LiteLLM client-control parameters dropped
        (:func:`stdapi.aws_bedrock.filter_extra_model_parameters`).

    Raises:
        ApiError: If a remaining extra reuses a ``set_inference_configuration``
            argument name, which would bind twice and fail the call, or if the
            request asks for web access it cannot be given.
    """
    extras = filter_extra_model_parameters(extras)
    if EXTERNAL_WEB_ACCESS_PARAM in extras:
        resolve_external_web_access(
            extras.pop(EXTERNAL_WEB_ACCESS_PARAM), per_request=False
        )
    if conflicting := sorted(reserved.intersection(extras)):
        names = ", ".join(f"'{name}'" for name in conflicting)
        msg = f"Unsupported parameter: {names} cannot be sent as a model extra."
        raise ApiError(msg)
    return extras
