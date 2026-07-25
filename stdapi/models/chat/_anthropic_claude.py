"""Common base for all Anthropic Claude chat model implementations."""

from re import compile as re_compile
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

from stdapi.config import SETTINGS
from stdapi.models import MANTLE_SERVICE
from stdapi.models.chat._default import ChatModel as _BaseChatModel
from stdapi.monitoring import log_error_details

if TYPE_CHECKING:
    from types_aiobotocore_bedrock_runtime.type_defs import (
        MessageTypeDef,
        ToolConfigurationTypeDef,
    )

    from stdapi.models import ModelDetails
    from stdapi.models.chat import Effort
    from stdapi.types import JsonMapping
    from stdapi.types.anthropic_messages import ServerTools, ThinkingEffort

#: ``anthropic_beta`` flag for computer-use tools (2025-01-24 version)
_BETA_COMPUTER_USE_2025 = "computer-use-2025-01-24"

#: ``anthropic_beta`` flag for computer-use tools (2024-10-22 version)
_BETA_COMPUTER_USE_2024 = "computer-use-2024-10-22"

#: ``anthropic_beta`` flag for computer-use tools (2025-11-24 version)
_BETA_COMPUTER_USE_2025_11 = "computer-use-2025-11-24"

#: ``anthropic_beta`` flag for context management tools
_BETA_CONTEXT_MANAGEMENT_2025 = "context-management-2025-06-27"

#: Beta flags keyed by versioned tool type, taking precedence over name-based ``TOOL_BETA_FLAGS``.
_VERSIONED_TYPE_BETA_FLAGS: dict[str, str] = {
    "computer_20251124": _BETA_COMPUTER_USE_2025_11
}

#: Fields excluded when serialising Anthropic server tools
_SERVER_TOOL_SERIALIZE_EXCLUDE: frozenset[str] = frozenset(
    {"cache_control", "defer_loading", "input_examples", "strict", "allowed_callers"}
)

#: Claude server tool keys
_SERVER_TOOL_KEYS = frozenset({"name", "type"})

#: Default reasoning config for Claude models
_REASONING_CONFIG: dict[str, str] = {"type": "adaptive"}

#: Regex to match a date suffix in model ID
_DATE_SUFFIX = re_compile(r"^(.+)-(\d{8})$")


def _has_tool_result(bedrock_messages: list[MessageTypeDef]) -> bool:
    """Return ``True`` if *bedrock_messages* contains any ``toolResult`` block.

    Args:
        bedrock_messages: Bedrock Converse message list.

    Returns:
        ``True`` when any message content block is a ``toolResult``.
    """
    return any(
        "toolResult" in block
        for msg in bedrock_messages
        for block in (msg.get("content") or [])
        if isinstance(block, dict)
    )


def _forward_tool_choice_to_additional_request_fields(
    tool_choice: dict[str, object], additional_request_fields: JsonMapping
) -> None:
    """Convert a Bedrock ``toolChoice`` into Anthropic ``additionalModelRequestFields`` ``tool_choice`` format.

    Called when all ``toolSpec`` stubs are removed from *tool_config* in
    native-format mode so that the choice directive is not silently lost.
    The Bedrock format uses a type-keyed wrapper dict (``{"any": {}}``);
    the Anthropic ``additionalModelRequestFields`` format uses an explicit
    ``"type"`` field (``{"type": "any"}``).

    Only writes to *additional_request_fields* when ``tool_choice`` is not
    already present (first-write wins).

    Args:
        tool_choice: Bedrock ``toolChoice`` dict extracted from ``toolConfig``.
        additional_request_fields: Mutable ``additionalModelRequestFields`` dict to update.
    """
    if "tool_choice" in additional_request_fields:
        return
    match tool_choice:
        case {"any": _}:
            additional_request_fields["tool_choice"] = {"type": "any"}
        case {"auto": _}:
            additional_request_fields["tool_choice"] = {"type": "auto"}
        case {"tool": {"name": str() as name}} if name:
            additional_request_fields["tool_choice"] = {"type": "tool", "name": name}


class AnthropicClaudeChatModel(_BaseChatModel):
    """Shared functionality for all Anthropic Claude model generations."""

    ALIAS_MATCHER = re_compile(r"^anthropic\.(.+?)(?:-v\d+(?::\d+)?)?$")
    PROMPT_CACHING_SUPPORTED = True
    PROMPT_CACHING_TOOL_SUPPORTED = True
    PASSTHROUGH_HEADERS = MappingProxyType(
        {"anthropic-beta": ("anthropic_beta", lambda v: v.split(","))}
    )
    SIMPLIFIED_CACHE_MANAGEMENT = True
    S3_LOCATION_DOCUMENT_SUPPORTED = False

    #: Required ``anthropic_beta`` flag per Anthropic server tool canonical name.
    TOOL_BETA_FLAGS: ClassVar[MappingProxyType[ServerTools, str]]

    #: Maps Claude server tool name to its versioned type (e.g. ``bash`` → ``bash_20250124``).
    SERVER_TOOL_NAME_TO_TYPE: ClassVar[MappingProxyType[str, str]]

    #: OpenAI to Anthropic reasoning effort override - subclass to customize
    REASONING_OVERRIDE: ClassVar[dict[Effort | None, ThinkingEffort]] = {
        "minimal": "low",
        "low": "low",
    }

    #: Whether the model accepts an explicitly disabled reasoning configuration.
    REASONING_DISABLE_SUPPORTED: ClassVar[bool] = True

    def _req_extract_server_tools(
        self, tool_config: ToolConfigurationTypeDef | None
    ) -> list[JsonMapping]:
        """Detect Claude server tools in *tool_config* by matching names against ``SERVER_TOOL_NAME_TO_TYPE``.

        A ``toolSpec`` entry is a server tool when its ``name`` matches a key in
        ``SERVER_TOOL_NAME_TO_TYPE``.  The versioned type is looked up from that
        map.  Extra fields in ``inputSchema.json`` (e.g. ``display_width_px``)
        become additional tool params, and the stub schema is reset to
        ``{"type": "object"}`` so Bedrock accepts it in multi-turn mode.

        Args:
            tool_config: Bedrock tool configuration before system tool promotion.

        Returns:
            List of ``{"name": <tool_name>, "type": <versioned_type>, **extra_params}``
            dicts.  Empty when *tool_config* is ``None`` or no entries match.
        """
        if not tool_config:
            return []
        server_tools: list[JsonMapping] = []
        for entry in tool_config["tools"]:
            if not (
                isinstance(entry, dict)
                and isinstance(spec := entry.get("toolSpec"), dict)
                and (tool_name := str(spec.get("name", "")))
                in self.SERVER_TOOL_NAME_TO_TYPE
            ):
                continue
            tool_type = self.SERVER_TOOL_NAME_TO_TYPE[tool_name]
            json_params: JsonMapping = (spec.get("inputSchema") or {}).get("json") or {}  # type: ignore[assignment]
            if (
                extra := {
                    k: v for k, v in json_params.items() if k not in _SERVER_TOOL_KEYS
                }
            ) and isinstance(input_schema := spec.get("inputSchema"), dict):
                input_schema["json"] = {"type": "object"}
            server_tools.append({"name": tool_name, "type": tool_type, **extra})
        return server_tools

    def _req_configure_tools(
        self,
        tool_config: ToolConfigurationTypeDef | None,
        additional_request_fields: JsonMapping,
        server_tools: list[JsonMapping],
        bedrock_messages: list[MessageTypeDef] | None = None,
    ) -> None:
        """Configure Claude server tools: native-format routing on Turn 1, stubs on Turn 2+, plus beta flags.

        **Turn 1 — native Anthropic format** (no ``toolResult`` in history):
        All server tools are moved to ``additionalModelRequestFields["tools"]``
        in Anthropic native format so Claude receives the full tool configuration.
        Their ``toolSpec`` stubs are removed from *tool_config*.

        **Multi-turn stub mode** (``toolResult`` present in history):
        Server tools remain as ``toolSpec`` stubs in *tool_config* (required by
        Bedrock when ``toolResult`` blocks appear in message history).

        ``anthropic_beta`` flags are injected in both cases.

        Args:
            tool_config: Bedrock tool configuration (mutable).  Stubs are removed
                in Turn 1; dict is cleared when it becomes empty.
            additional_request_fields: Mutable ``additionalModelRequestFields`` dict.
            server_tools: Per-tool dicts — on the Anthropic route these are full
                ``model_dump(exclude_none=True)`` dicts; on the OpenAI route they
                contain ``{"name": tool_name, "type": versioned_type}`` plus
                any extra params.
            bedrock_messages: Translated Bedrock messages, used to detect
                ``toolResult`` blocks.  ``None`` treated as empty (Turn 1).
        """
        if not server_tools:
            return

        if not _has_tool_result(bedrock_messages or []):
            # Turn 1: move all server tools to additionalModelRequestFields native format.
            native_tool_names = {tool["name"] for tool in server_tools}
            existing_tools: list[JsonMapping] = additional_request_fields.get(  # type: ignore[assignment]
                "tools", []
            )
            additional_request_fields["tools"] = existing_tools + [  # type: ignore[assignment]
                {
                    k: v
                    for k, v in tool.items()
                    if k not in _SERVER_TOOL_SERIALIZE_EXCLUDE
                }
                for tool in server_tools
            ]

            # Remove corresponding stubs from toolConfig.
            if tool_config:
                tool_config["tools"] = [
                    entry
                    for entry in tool_config["tools"]
                    if not (
                        isinstance(entry, dict)
                        and isinstance(spec := entry.get("toolSpec"), dict)
                        and spec.get("name") in native_tool_names
                    )
                ]
                if not tool_config["tools"]:
                    if tool_choice := tool_config.get("toolChoice"):
                        _forward_tool_choice_to_additional_request_fields(
                            tool_choice,  # type: ignore[arg-type]
                            additional_request_fields,
                        )
                    tool_config.clear()  # type: ignore[attr-defined]

        # Inject anthropic_beta flags for all server tools (both modes).
        if required_flags := {
            flag
            for tool in server_tools
            if (
                flag := _VERSIONED_TYPE_BETA_FLAGS.get(str(tool.get("type", "")))
                or self.TOOL_BETA_FLAGS.get(str(tool.get("name", "")))  # type: ignore[call-overload]
            )
        }:
            existing: list[str] = additional_request_fields.get("anthropic_beta", [])  # type: ignore[assignment]
            if new_flags := required_flags - set(existing):
                additional_request_fields["anthropic_beta"] = existing + list(new_flags)  # type: ignore[assignment]

    def _prepare_additional_request_fields(
        self, additional_request_fields: JsonMapping
    ) -> JsonMapping:
        """Filter unsupported ``anthropic_beta`` flags after merging passthrough headers.

        Args:
            additional_request_fields: Fields from request body and defaults.

        Returns:
            Merged and filtered additional request fields.
        """
        additional_request_fields = super()._prepare_additional_request_fields(
            additional_request_fields
        )
        if not (
            SETTINGS.anthropic_beta_filter
            and "anthropic_beta" in additional_request_fields
        ):
            return additional_request_fields
        flags: list[str] = additional_request_fields["anthropic_beta"]  # type: ignore[assignment]
        if rejected := set(flags) - SETTINGS.anthropic_beta_allowlist:
            log_error_details(
                f"Filtered unsupported anthropic_beta flags: {', '.join(rejected)}",
                level="warning",
            )
            if allowed := [f for f in flags if f not in rejected]:
                additional_request_fields["anthropic_beta"] = allowed  # type: ignore[assignment]
            else:
                del additional_request_fields["anthropic_beta"]
        return additional_request_fields

    def _req_configure_reasoning(
        self,
        additional_request_fields: JsonMapping,
        *,
        enabled: bool,
        reasoning_effort: Effort | None = None,
        budget_tokens: int | None = None,
        max_tokens: int | None = None,  # noqa: ARG002
    ) -> None:
        """Configure reasoning parameters for Claude models.

        When ``budget_tokens`` is explicitly provided (> 0), uses budget-based
        reasoning. Otherwise uses adaptive reasoning with an optional effort level.
        When ``enabled`` is ``False``, reasoning is explicitly disabled, unless the
        model rejects that configuration and always reasons in adaptive mode.

        Args:
            additional_request_fields: Additional request fields dict to update.
            enabled: Whether reasoning is explicitly enabled.
            reasoning_effort: The reasoning effort level.
            budget_tokens: Maximum token budget for reasoning.
            max_tokens: Unused.
        """
        if not enabled:
            if not self.REASONING_DISABLE_SUPPORTED:
                log_error_details(
                    "Reasoning cannot be disabled on this model: "
                    "its default adaptive mode is used instead",
                    level="warning",
                )
                return
            additional_request_fields["reasoning_config"] = {"type": "disabled"}
        elif budget_tokens:
            additional_request_fields["reasoning_config"] = {
                "type": "enabled",
                "budget_tokens": budget_tokens,
            }
        else:
            additional_request_fields["reasoning_config"] = _REASONING_CONFIG  # type: ignore[assignment]
            if reasoning_effort:
                additional_request_fields["output_config"] = {
                    "effort": self.REASONING_OVERRIDE.get(
                        reasoning_effort, reasoning_effort
                    )
                }

    @classmethod
    def get_aliases(cls, all_models: dict[str, ModelDetails]) -> dict[str, str]:
        """Return API model name aliases mapped to model IDs.

        Extends the base aliases with:
        - For Claude >= 4, A date-stripped alias for the most recent dated variant
          (e.g. ``claude-haiku-4-5-20251001`` -> ``claude-haiku-4-5``).
        - For Claude < 4, a ``-latest`` suffixed alias for the most recent dated variant
          (e.g. ``claude-3-7-sonnet-20250219`` -> ``claude-3-7-sonnet-latest``).

        Args:
            all_models: All available models keyed by Bedrock model ID.

        Returns:
            A dict mapping model alias to model ID.
        """
        aliases = super().get_aliases(all_models)
        newest: dict[str, tuple[str, str]] = {}
        for alias, model_id in aliases.items():
            if match := _DATE_SUFFIX.match(alias):
                base, date = match[1], match[2]
                if date > newest.get(base, ("",))[0]:
                    newest[base] = (date, model_id)

        for base, (_, model_id) in newest.items():
            alias = f"{base}-latest" if base.startswith("claude-3-") else base
            holder = aliases.get(alias)
            # The Mantle catalog lists undated Claude IDs whose direct alias
            # collides with the date-stripped one: bedrock-runtime dated
            # variants keep priority over them.
            if holder is None or all_models[holder].service == MANTLE_SERVICE:
                aliases[alias] = model_id

        return aliases
