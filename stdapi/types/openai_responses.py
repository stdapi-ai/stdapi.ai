"""Local OpenAI-compatible Responses API types."""

from typing import Annotated, ClassVar, Final, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from stdapi.api_errors import ApiError, UnsupportedParameterError
from stdapi.types import (
    BaseModelRequest,
    BaseModelRequestWithExtra,
    BaseModelResponse,
    JsonMapping,
)
from stdapi.types.openai import (
    Metadata,
    PaginatedListEnvelope,
    RequestModeration,
    ResponseFormatJSONObject,
    ResponseFormatText,
    ResponseModeration,
)

# Literals / type aliases

#: Response status values.
ResponseStatus = Literal[
    "completed", "failed", "in_progress", "cancelled", "queued", "incomplete"
]

#: Item-level status values.
ResponseItemStatus = Literal["in_progress", "completed", "incomplete"]

#: Reasoning effort levels for reasoning models.
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

#: Verbosity levels used across multiple parameters.
VerbosityLevel = Literal["low", "medium", "high"]

#: Service tier options.
ServiceTiers = Literal["auto", "default", "flex", "scale", "priority"]

#: Prompt cache retention options.
PromptCacheRetention = Literal[
    "in_memory",
    "24h",
    # Extra bedrock specific values
    "1h",
    "5m",
]

#: Prompt cache retention of the `prompt_cache_options.ttl` field.
PromptCacheOptionsTTL = Literal["30m"]

#: String-only tool-choice options.
ToolChoiceLiteral = Literal["none", "auto", "required"]

#: Valid `include` values for response output enrichment.
# Ref: openai.types.responses.response_includable.ResponseIncludable
ResponseIncludable = Literal[
    "file_search_call.results",
    "web_search_call.results",
    "web_search_call.action.sources",
    "message.input_image.image_url",
    "computer_call_output.output.image_url",
    "code_interpreter_call.outputs",
    "reasoning.encrypted_content",
    "message.output_text.logprobs",
]


# Filter types  (used in FileSearchTool)


# Ref: openai.types.shared.comparison_filter.ComparisonFilter
class ComparisonFilter(BaseModelRequest):
    """Compares a specified attribute key to a given value using a defined operator."""

    key: str = Field(description="The key to compare against.")
    type: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"] = Field(
        description="Comparison operator: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, or `nin`."
    )
    value: str | float | bool | list[str | float] = Field(
        description="The value to compare against the key."
    )


# Ref: openai.types.shared.compound_filter.CompoundFilter
class CompoundFilter(BaseModelRequest):
    """Combine multiple filters using `and` or `or`."""

    filters: list[ComparisonFilter | object] = Field(
        description="Array of filters to combine."
    )
    type: Literal["and", "or"] = Field(description="Combine operation: `and` or `or`.")


#: Union of filter types applicable to file search.
FileSearchFilters = ComparisonFilter | CompoundFilter | None


# Custom tool input format  (used in CustomTool)


# Ref: openai.types.shared.custom_tool_input_format.Text
class CustomToolInputFormatText(BaseModelRequest):
    """Unconstrained free-form text input format."""

    type: Literal["text"] = Field(description="Text format identifier.")


# Ref: openai.types.shared.custom_tool_input_format.Grammar
class CustomToolInputFormatGrammar(BaseModelRequest):
    """A grammar-constrained input format."""

    definition: str = Field(description="The grammar definition.")
    syntax: Literal["lark", "regex"] = Field(
        description="Grammar syntax type: `lark` or `regex`."
    )
    type: Literal["grammar"] = Field(description="Grammar format identifier.")


# Ref: openai.types.shared.custom_tool_input_format.CustomToolInputFormat
CustomToolInputFormat = Annotated[
    CustomToolInputFormatText | CustomToolInputFormatGrammar,
    Field(discriminator="type"),
]


# Container / environment types  (used in tool definitions)


# Ref: openai.types.responses.container_network_policy_disabled.ContainerNetworkPolicyDisabled
class ContainerNetworkPolicyDisabled(BaseModelRequest):
    """Disable outbound network access from the container."""

    type: Literal["disabled"] = Field(description="Network policy disabled.")


# Ref: openai.types.responses.container_network_policy_domain_secret.ContainerNetworkPolicyDomainSecret
class ContainerNetworkPolicyDomainSecret(BaseModelRequest):
    """A domain-scoped secret injected for an allowlisted domain."""

    domain: str = Field(description="The domain for this secret.")
    name: str = Field(description="The secret name.")
    value: str = Field(description="The secret value.")


# Ref: openai.types.responses.container_network_policy_allowlist.ContainerNetworkPolicyAllowlist
class ContainerNetworkPolicyAllowlist(BaseModelRequest):
    """Allow outbound network access only to specified domains."""

    allowed_domains: list[str] = Field(description="List of allowed domains.")
    type: Literal["allowlist"] = Field(description="Allowlist network policy.")
    domain_secrets: list[ContainerNetworkPolicyDomainSecret] | None = Field(
        default=None, description="Domain-scoped secrets for allowlisted domains."
    )


# Ref: openai.types.responses.container_auto.NetworkPolicy
ContainerNetworkPolicy = Annotated[
    ContainerNetworkPolicyDisabled | ContainerNetworkPolicyAllowlist,
    Field(discriminator="type"),
]


# Ref: openai.types.responses.local_skill.LocalSkill
class LocalSkill(BaseModelRequest):
    """A local skill for a shell environment."""

    description: str = Field(description="The description of the skill.")
    name: str = Field(description="The name of the skill.")
    path: str = Field(description="The path to the directory containing the skill.")


# Ref: openai.types.responses.skill_reference.SkillReference
class SkillReference(BaseModelRequest):
    """References a skill created with the /v1/skills endpoint."""

    skill_id: str = Field(description="The ID of the referenced skill.")
    type: Literal["skill_reference"] = Field(
        description="References a skill created with the /v1/skills endpoint."
    )
    version: str | None = Field(
        default=None,
        description="Optional skill version. Use a positive integer or 'latest'. Omit for default.",
    )


# Ref: openai.types.responses.inline_skill_source.InlineSkillSource
class InlineSkillSource(BaseModelRequest):
    """Inline skill payload."""

    data: str = Field(description="Base64-encoded skill zip bundle.")
    media_type: Literal["application/zip"] = Field(
        description="The media type of the inline skill payload. Must be `application/zip`."
    )
    type: Literal["base64"] = Field(
        description="The type of the inline skill source. Must be `base64`."
    )


# Ref: openai.types.responses.inline_skill.InlineSkill
class InlineSkill(BaseModelRequest):
    """An inline skill definition."""

    description: str = Field(description="The description of the skill.")
    name: str = Field(description="The name of the skill.")
    source: InlineSkillSource = Field(description="Inline skill payload.")
    type: Literal["inline"] = Field(
        description="Defines an inline skill for this request."
    )


# Ref: openai.types.responses.container_auto.Skill
ContainerSkill = Annotated[SkillReference | InlineSkill, Field(discriminator="type")]


# Ref: openai.types.responses.container_auto.ContainerAuto
class ContainerAuto(BaseModelRequest):
    """Automatically creates a container for this request."""

    type: Literal["container_auto"] = Field(description="Container auto.")
    file_ids: list[str] | None = Field(
        default=None, description="Uploaded files to make available to code."
    )
    memory_limit: Literal["1g", "4g", "16g", "64g"] | None = Field(
        default=None, description="Container memory limit."
    )
    network_policy: ContainerNetworkPolicy | None = Field(
        default=None, description="Network access policy."
    )
    skills: list[ContainerSkill] | None = Field(
        default=None, description="Skills to include in the container."
    )


# Ref: openai.types.responses.container_reference.ContainerReference
class ContainerReference(BaseModelRequest):
    """References a container created with the /v1/containers endpoint."""

    container_id: str = Field(description="Referenced container ID.")
    type: Literal["container_reference"] = Field(
        description="Container reference type."
    )


# Ref: openai.types.responses.local_environment.LocalEnvironment
class LocalEnvironment(BaseModelRequest):
    """Use a local computer environment."""

    type: Literal["local"] = Field(description="Local environment.")
    skills: list[LocalSkill] | None = Field(default=None, description="List of skills.")


# Tool definitions

#: Which caller types may invoke a tool: a direct model call or a program call.
ToolAllowedCallers = list[Literal["direct", "programmatic"]] | None


# Ref: openai.types.responses.function_tool.FunctionTool
class FunctionTool(BaseModelRequest):
    """Defines a function in your own code the model can choose to call."""

    name: str = Field(description="Function name.")
    type: Literal["function"] = Field(description="Function tool type.")
    allowed_callers: ToolAllowedCallers = Field(
        default=None, description="Caller types allowed to invoke this tool."
    )
    defer_loading: bool | None = Field(
        default=None,
        description="Whether this function is deferred and loaded via tool search.",
    )
    description: str | None = Field(
        default=None, description="Function description for the model."
    )
    output_schema: JsonMapping | None = Field(
        default=None, description="JSON schema describing the function output."
    )
    parameters: JsonMapping | None = Field(
        default=None, description="JSON schema for function parameters."
    )
    strict: bool | None = Field(
        default=None, description="Enforce strict parameter validation. Default: true."
    )


# Ref: openai.types.responses.file_search_tool.RankingOptionsHybridSearch
class FileSearchRankingOptionsHybridSearch(BaseModelRequest):
    """Hybrid search weighting for reciprocal rank fusion."""

    embedding_weight: float = Field(
        description="Embedding weight for reciprocal rank fusion."
    )
    text_weight: float = Field(description="Text weight for reciprocal rank fusion.")


# Ref: openai.types.responses.file_search_tool.RankingOptions
class FileSearchRankingOptions(BaseModelRequest):
    """Ranking options for file search."""

    hybrid_search: FileSearchRankingOptionsHybridSearch | None = Field(
        default=None, description="Weights for hybrid search reciprocal rank fusion."
    )
    ranker: Literal["auto", "default-2024-11-15"] | None = Field(
        default=None, description="File search ranker."
    )
    score_threshold: float | None = Field(
        default=None, description="Score threshold (0 to 1)."
    )


# Ref: openai.types.responses.file_search_tool.FileSearchTool
class FileSearchTool(BaseModelRequest):
    """A tool that searches for relevant content in the attached vector stores."""

    type: Literal["file_search"] = Field(description="File search tool type.")
    vector_store_ids: list[str] = Field(
        min_length=1, description="Vector store IDs to search; at least one."
    )
    filters: FileSearchFilters = Field(default=None, description="Filter to apply.")
    max_num_results: int | None = Field(
        default=None, ge=1, le=50, description="Maximum results to return (1-50)."
    )
    ranking_options: FileSearchRankingOptions | None = Field(
        default=None, description="Ranking options for search."
    )


# Ref: openai.types.responses.web_search_tool.Filters
class WebSearchFilters(BaseModelRequest):
    """Filters for web search."""

    allowed_domains: list[str] | None = Field(
        default=None, description="Allowed domains for the search."
    )


# Ref: openai.types.responses.web_search_tool.UserLocation
class WebSearchUserLocation(BaseModelRequest):
    """The approximate location of the user."""

    city: str | None = Field(default=None, description="User's city.")
    country: str | None = Field(default=None, description="User's ISO country code.")
    region: str | None = Field(default=None, description="User's region.")
    timezone: str | None = Field(default=None, description="User's IANA timezone.")
    type: Literal["approximate"] | None = Field(
        default=None, description="Approximate location type."
    )


# Ref: openai.types.responses.web_search_tool.WebSearchTool
class WebSearchTool(BaseModelRequest):
    """Search the web for sources related to the prompt.

    Whether a search may reach the external web is a server setting. Where the
    server allows choosing it, send the boolean `external_web_access` as a
    top-level extra parameter of the request; a value the server does not
    allow is rejected.
    """

    type: Literal["web_search", "web_search_2025_08_26"] = Field(
        description="Web search tool type."
    )
    filters: WebSearchFilters | None = Field(
        default=None, description="Search filters."
    )
    search_context_size: VerbosityLevel | None = Field(
        default=None,
        description="Context window size: `low`, `medium`, or `high`. Default: `medium`.",
    )
    user_location: WebSearchUserLocation | None = Field(
        default=None, description="User's approximate location."
    )


# Ref: openai.types.responses.web_search_preview_tool.UserLocation
class WebSearchPreviewUserLocation(BaseModelRequest):
    """The user's location for web search preview."""

    type: Literal["approximate"] = Field(description="Approximate location type.")
    city: str | None = Field(default=None, description="User's city.")
    country: str | None = Field(default=None, description="User's ISO country code.")
    region: str | None = Field(default=None, description="User's region.")
    timezone: str | None = Field(default=None, description="User's IANA timezone.")


# Ref: openai.types.responses.web_search_preview_tool.WebSearchPreviewTool
class WebSearchPreviewTool(BaseModelRequest):
    """This tool searches the web for relevant results to use in a response.

    Whether a search may reach the external web is a server setting. Where the
    server allows choosing it, send the boolean `external_web_access` as a
    top-level extra parameter of the request; a value the server does not
    allow is rejected.
    """

    type: Literal["web_search_preview", "web_search_preview_2025_03_11"] = Field(
        description="Web search preview tool type."
    )
    search_content_types: list[Literal["text", "image"]] | None = Field(
        default=None, description="Content types to include in search results."
    )
    search_context_size: VerbosityLevel | None = Field(
        default=None,
        description="Context window size: `low`, `medium`, or `high`. Default: `medium`.",
    )
    user_location: WebSearchPreviewUserLocation | None = Field(
        default=None, description="User's location."
    )


# Ref: openai.types.responses.computer_tool.ComputerTool
class ComputerTool(BaseModelRequest):
    """A tool that controls a virtual computer.

    UNSUPPORTED on this implementation.
    """

    type: Literal["computer"] = Field(description="Computer tool type.")


# Ref: openai.types.responses.computer_use_preview_tool.ComputerUsePreviewTool
class ComputerUsePreviewTool(BaseModelRequest):
    """A tool that controls a virtual computer (preview version).

    UNSUPPORTED on this implementation.
    """

    display_height: int = Field(description="Display height in pixels.")
    display_width: int = Field(description="Display width in pixels.")
    environment: Literal["windows", "mac", "linux", "ubuntu", "browser"] = Field(
        description="Computer environment to control."
    )
    type: Literal["computer_use_preview"] = Field(
        description="Computer use preview tool type."
    )


# Ref: openai.types.responses.tool.McpAllowedToolsMcpToolFilter
class McpAllowedToolsFilter(BaseModelRequest):
    """A filter object to specify which MCP tools are allowed."""

    read_only: bool | None = Field(
        default=None, description="Filter by read-only status."
    )
    tool_names: list[str] | None = Field(
        default=None, description="List of allowed tool names."
    )


#: MCP allowed tools specification: a list of names or a filter object.
McpAllowedTools = list[str] | McpAllowedToolsFilter | None


# Ref: openai.types.responses.tool.McpRequireApprovalMcpToolApprovalFilterAlways
# Ref: openai.types.responses.tool.McpRequireApprovalMcpToolApprovalFilterNever
class McpToolFilter(BaseModelRequest):
    """Filter specifying a set of MCP tools by name or read-only status."""

    read_only: bool | None = Field(
        default=None, description="Filter by read-only status."
    )
    tool_names: list[str] | None = Field(
        default=None, description="List of tool names."
    )


# Ref: openai.types.responses.tool.McpRequireApprovalMcpToolApprovalFilter
class McpRequireApprovalFilter(BaseModelRequest):
    """Specify which of the MCP server's tools require approval."""

    always: McpToolFilter | None = Field(
        default=None,
        description="A filter object to specify which tools always require approval.",
    )
    never: McpToolFilter | None = Field(
        default=None,
        description="A filter object to specify which tools never require approval.",
    )


#: MCP require approval specification.
McpRequireApproval = McpRequireApprovalFilter | Literal["always", "never"] | None


# Ref: openai.types.responses.tool.Mcp
class Mcp(BaseModelRequest):
    """Give the model access to additional tools via remote Model Context Protocol (MCP) servers.

    UNSUPPORTED on this implementation.
    """

    server_label: str = Field(description="Label for this MCP server.")
    type: Literal["mcp"] = Field(description="MCP tool type.")
    allowed_callers: ToolAllowedCallers = Field(
        default=None, description="Caller types allowed to invoke this tool."
    )
    allowed_tools: McpAllowedTools = Field(
        default=None, description="Allowed tool names or filter."
    )
    authorization: str | None = Field(
        default=None, description="OAuth access token for the MCP server."
    )
    connector_id: (
        Literal[
            "connector_dropbox",
            "connector_gmail",
            "connector_googlecalendar",
            "connector_googledrive",
            "connector_microsoftteams",
            "connector_outlookcalendar",
            "connector_outlookemail",
            "connector_sharepoint",
        ]
        | None
    ) = Field(
        default=None,
        description="Service connector. Requires `server_url` or `connector_id`.",
    )
    defer_loading: bool | None = Field(
        default=None, description="Deferred and discovered via tool search."
    )
    headers: dict[str, str] | None = Field(
        default=None, description="HTTP headers for the MCP server."
    )
    require_approval: McpRequireApproval = Field(
        default=None, description="Tools requiring approval."
    )
    server_description: str | None = Field(
        default=None, description="MCP server description."
    )
    server_url: str | None = Field(
        default=None,
        description="MCP server URL. Requires `server_url` or `connector_id`.",
    )
    tunnel_id: str | None = Field(
        default=None, description="Tunnel ID for connecting to a local MCP server."
    )


# Ref: openai.types.responses.tool.CodeInterpreterContainerCodeInterpreterToolAuto
class CodeInterpreterContainerAuto(BaseModelRequest):
    """Configuration for a code interpreter container."""

    type: Literal["auto"] = Field(description="Auto container type.")
    file_ids: list[str] | None = Field(
        default=None, description="Uploaded files for code interpreter."
    )
    memory_limit: Literal["1g", "4g", "16g", "64g"] | None = Field(
        default=None, description="Container memory limit."
    )
    network_policy: ContainerNetworkPolicy | None = Field(
        default=None, description="Network access policy."
    )


#: Code interpreter container specification.
CodeInterpreterContainer = str | CodeInterpreterContainerAuto


# Ref: openai.types.responses.tool.CodeInterpreter
class CodeInterpreter(BaseModelRequest):
    """A tool that runs Python code to help generate a response to a prompt."""

    container: CodeInterpreterContainer | None = Field(
        default=None, description="Code interpreter container (ID or config)."
    )
    type: Literal["code_interpreter"] = Field(description="Code interpreter tool type.")
    allowed_callers: ToolAllowedCallers = Field(
        default=None, description="Caller types allowed to invoke this tool."
    )


# Ref: openai.types.responses.tool.ImageGenerationInputImageMask
class ImageGenerationInputImageMask(BaseModelRequest):
    """Optional mask for inpainting."""

    file_id: str | None = Field(default=None, description="Mask image file ID.")
    image_url: str | None = Field(
        default=None, description="Base64-encoded mask image."
    )


# Ref: openai.types.responses.tool.ImageGeneration
class ImageGeneration(BaseModelRequest):
    """A tool that generates images."""

    type: Literal["image_generation"] = Field(description="Image generation tool type.")
    action: Literal["generate", "edit", "auto"] | None = Field(
        default=None, description="Generate new or edit existing image. Default: auto."
    )
    background: Literal["transparent", "opaque", "auto"] | None = Field(
        default=None,
        description="Background type: `transparent`, `opaque`, or `auto`. Default: auto.",
    )
    input_fidelity: Literal["high", "low"] | None = Field(
        default=None, description="Match style/features of input images. Default: low."
    )
    input_image_mask: ImageGenerationInputImageMask | None = Field(
        default=None, description="Mask for inpainting."
    )
    model: str | None = Field(
        default=None,
        description="Image generation model. Wildcard patterns are accepted "
        "and select the most recent matching model.",
    )
    moderation: Literal["auto", "low"] | None = Field(
        default=None, description="Moderation level. Default: auto."
    )
    output_compression: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Output compression (0-100). Default: 100.",
    )
    output_format: Literal["png", "webp", "jpeg"] | None = Field(
        default=None,
        description="Output format: `png`, `webp`, or `jpeg`. Default: png.",
    )
    partial_images: int | None = Field(
        default=None,
        ge=0,
        le=3,
        description="Partial images for streaming (0-3). Default: 0.",
    )
    quality: Literal["low", "medium", "high", "auto"] | None = Field(
        default=None,
        description="Image quality: `low`, `medium`, `high`, or `auto`. Default: auto.",
    )
    size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"] | None = Field(
        default=None,
        description="Image size: `1024x1024`, `1024x1536`, `1536x1024`, or `auto`. Default: auto.",
    )


# Ref: openai.types.responses.tool.LocalShell
class LocalShell(BaseModelRequest):
    """A tool that allows the model to execute shell commands in a local environment.

    UNSUPPORTED on this implementation.
    """

    type: Literal["local_shell"] = Field(description="Local shell tool type.")


#: Shell tool environment union.
FunctionShellEnvironment = (
    Annotated[
        ContainerAuto | LocalEnvironment | ContainerReference,
        Field(discriminator="type"),
    ]
    | None
)


# Ref: openai.types.responses.function_shell_tool.FunctionShellTool
class FunctionShellTool(BaseModelRequest):
    """A tool that allows the model to execute shell commands.

    UNSUPPORTED on this implementation.
    """

    type: Literal["shell"] = Field(description="Shell tool type.")
    allowed_callers: ToolAllowedCallers = Field(
        default=None, description="Caller types allowed to invoke this tool."
    )
    environment: FunctionShellEnvironment = Field(
        default=None, description="Environment for shell commands."
    )


# Ref: openai.types.responses.custom_tool.CustomTool
class CustomTool(BaseModelRequest):
    """A custom tool that processes input using a specified format.

    UNSUPPORTED on this implementation.
    """

    name: str = Field(description="Custom tool name.")
    type: Literal["custom"] = Field(description="Custom tool type.")
    allowed_callers: ToolAllowedCallers = Field(
        default=None, description="Caller types allowed to invoke this tool."
    )
    defer_loading: bool | None = Field(
        default=None, description="Deferred and discovered via tool search."
    )
    description: str | None = Field(default=None, description="Tool description.")
    format: CustomToolInputFormat | None = Field(
        default=None, description="Input format. Default: unconstrained text."
    )


# Ref: openai.types.responses.namespace_tool.ToolFunction
class NamespaceToolFunction(BaseModelRequest):
    """A function tool within a namespace."""

    name: str = Field(description="Function name.")
    type: Literal["function"] = Field(description="Function type.")
    defer_loading: bool | None = Field(
        default=None, description="Deferred and discovered via tool search."
    )
    description: str | None = Field(default=None, description="Function description.")
    parameters: object | None = Field(default=None, description="Parameter schema.")
    strict: bool | None = Field(default=None, description="Enforce strict validation.")


# Ref: openai.types.responses.namespace_tool.Tool
NamespaceToolTool = Annotated[
    NamespaceToolFunction | CustomTool, Field(discriminator="type")
]


# Ref: openai.types.responses.namespace_tool.NamespaceTool
class NamespaceTool(BaseModelRequest):
    """Groups function/custom tools under a shared namespace.

    UNSUPPORTED on this implementation.
    """

    description: str = Field(description="Description shown to the model.")
    name: str = Field(description="Namespace name.")
    tools: list[NamespaceToolTool] = Field(
        description="Tools available in this namespace."
    )
    type: Literal["namespace"] = Field(description="Namespace tool type.")


# Ref: openai.types.responses.tool_search_tool.ToolSearchTool
class ToolSearchTool(BaseModelRequest):
    """Hosted or BYOT tool search configuration for deferred tools.

    UNSUPPORTED on this implementation.
    """

    type: Literal["tool_search"] = Field(description="Tool search type.")
    description: str | None = Field(
        default=None, description="Description for client-executed tool search."
    )
    execution: Literal["server", "client"] | None = Field(
        default=None, description="Execute tool search on server or client."
    )
    parameters: object | None = Field(
        default=None, description="Parameter schema for client-executed tool search."
    )


# Ref: openai.types.responses.apply_patch_tool.ApplyPatchTool
class ApplyPatchTool(BaseModelRequest):
    """Allows the assistant to create, delete, or update files using unified diffs.

    UNSUPPORTED on this implementation.
    """

    type: Literal["apply_patch"] = Field(description="Apply patch tool type.")
    allowed_callers: ToolAllowedCallers = Field(
        default=None, description="Caller types allowed to invoke this tool."
    )


# Ref: openai.types.responses.tool.ProgrammaticToolCalling
class ProgrammaticToolCalling(BaseModelRequest):
    """Lets the model call other tools from generated code.

    Honored only by models that support programmatic tool calling;
    accepted and ignored on all others.
    """

    type: Literal["programmatic_tool_calling"] = Field(
        description="Programmatic tool calling type."
    )


# Ref: openai.types.responses.tool.Tool
Tool = Annotated[
    FunctionTool
    | FileSearchTool
    | ComputerTool
    | ComputerUsePreviewTool
    | WebSearchTool
    | Mcp
    | CodeInterpreter
    | ImageGeneration
    | LocalShell
    | FunctionShellTool
    | CustomTool
    | NamespaceTool
    | ToolSearchTool
    | WebSearchPreviewTool
    | ApplyPatchTool
    | ProgrammaticToolCalling,
    Field(discriminator="type"),
]


# Tool choice types


# Ref: openai.types.responses.tool_choice_types.ToolChoiceTypes
class ToolChoiceTypes(BaseModelRequest):
    """Indicates that the model should use a built-in tool to generate a response."""

    type: Literal[
        "file_search",
        "web_search_preview",
        "computer",
        "computer_use_preview",
        "computer_use",
        "web_search_preview_2025_03_11",
        "image_generation",
        "code_interpreter",
    ] = Field(description="Built-in tool type to use.")


# Ref: openai.types.responses.tool_choice_function.ToolChoiceFunction
class ToolChoiceFunction(BaseModelRequest):
    """Force the model to call a specific function."""

    name: str = Field(description="Function name to call.")
    type: Literal["function"] = Field(description="Function tool type.")


# Ref: openai.types.responses.tool_choice_allowed.ToolChoiceAllowed
class ToolChoiceAllowed(BaseModelRequest):
    """Constrains the tools available to the model to a pre-defined set."""

    mode: Literal["auto", "required"] = Field(
        description="`auto` lets model pick tools. `required` forces tool call."
    )
    tools: list[JsonMapping] = Field(description="Allowed tool definitions.")
    type: Literal["allowed_tools"] = Field(description="Allowed tools type.")


# Ref: openai.types.responses.tool_choice_mcp.ToolChoiceMcp
class ToolChoiceMcp(BaseModelRequest):
    """Force the model to call a specific tool on a remote MCP server."""

    server_label: str = Field(description="MCP server label.")
    type: Literal["mcp"] = Field(description="MCP tool type.")
    name: str | None = Field(default=None, description="Tool name on the server.")


# Ref: openai.types.responses.tool_choice_custom.ToolChoiceCustom
class ToolChoiceCustom(BaseModelRequest):
    """Force the model to call a specific custom tool."""

    name: str = Field(description="Custom tool name to call.")
    type: Literal["custom"] = Field(description="Custom tool type.")


# Ref: openai.types.responses.tool_choice_apply_patch.ToolChoiceApplyPatch
class ToolChoiceApplyPatch(BaseModelRequest):
    """Forces the model to call the apply_patch tool when executing a tool call."""

    type: Literal["apply_patch"] = Field(description="Apply patch tool.")


# Ref: openai.types.responses.tool_choice_shell.ToolChoiceShell
class ToolChoiceShell(BaseModelRequest):
    """Forces the model to call the shell tool when a tool call is required."""

    type: Literal["shell"] = Field(description="Shell tool.")


# Ref: openai.types.responses.response_create_params.ToolChoiceSpecificProgrammaticToolCallingParam
class ToolChoiceProgrammaticToolCalling(BaseModelRequest):
    """Forces the model to use programmatic tool calling.

    Honored only by models that support programmatic tool calling;
    accepted and ignored on all others (the default tool choice applies).
    """

    type: Literal["programmatic_tool_calling"] = Field(
        description="Programmatic tool calling."
    )


#: Full tool choice union: literal string or structured object.
ToolChoice = (
    ToolChoiceLiteral
    | ToolChoiceAllowed
    | ToolChoiceTypes
    | ToolChoiceFunction
    | ToolChoiceMcp
    | ToolChoiceCustom
    | ToolChoiceApplyPatch
    | ToolChoiceShell
    | ToolChoiceProgrammaticToolCalling
)


# Reasoning configuration


# Ref: openai.types.shared.reasoning.Reasoning
class Reasoning(BaseModelRequest):
    """Configuration options for reasoning models."""

    effort: ReasoningEffort | None = Field(
        default=None,
        description="Reasoning effort: `none`, `minimal`, `low`, `medium`, `high`, "
        "`xhigh`, or `max`.",
    )
    generate_summary: Literal["auto", "concise", "detailed"] | None = Field(
        default=None, description="Deprecated: use `summary` instead."
    )
    summary: Literal["auto", "concise", "detailed"] | None = Field(
        default=None, description="Reasoning summary: `auto`, `concise`, or `detailed`."
    )
    context: Literal["auto", "current_turn", "all_turns"] | None = Field(
        default=None,
        description="Reasoning context scope. Honored only by models that "
        "support it; ignored otherwise.",
    )
    mode: str | None = Field(
        default=None,
        description="Reasoning mode, such as `standard` or `pro`. Honored only "
        "by models that support it; ignored otherwise.",
    )


# Input content types


# Ref: openai.types.responses.response_input_text.PromptCacheBreakpoint
class PromptCacheBreakpoint(BaseModelRequest):
    """Explicit prompt-cache breakpoint set on an input content part."""

    mode: Literal["explicit"] = Field(
        default="explicit",
        description="Breakpoint mode. Always `explicit`: the prompt prefix ending "
        "with this content part is cached.",
    )


#: Description shared by every per-content-part `prompt_cache_breakpoint` field.
_CACHE_BREAKPOINT_DESCRIPTION = (
    "Cache the prompt prefix ending with this content part. Honored on models "
    "supporting prompt caching, accepted and ignored on the others."
)


# Ref: openai.types.responses.response_input_text.ResponseInputText
class ResponseInputText(BaseModelRequest):
    """A text input to the model."""

    text: str = Field(description="Text input.")
    type: Literal["input_text"] = Field(description="Input text type.")
    prompt_cache_breakpoint: PromptCacheBreakpoint | None = Field(
        default=None, description=_CACHE_BREAKPOINT_DESCRIPTION
    )


# Ref: openai.types.responses.response_input_image.ResponseInputImage
class ResponseInputImage(BaseModelRequest):
    """An image input to the model."""

    type: Literal["input_image"] = Field(description="Input image type.")
    detail: Literal["low", "high", "auto", "original"] | None = Field(
        default=None,
        description="Image detail level: `high`, `low`, `auto`, or `original`. Default: auto.",
    )
    file_id: str | None = Field(default=None, description="Image file ID.")
    image_url: str | None = Field(
        default=None, description="Image URL or base64 data URL."
    )
    prompt_cache_breakpoint: PromptCacheBreakpoint | None = Field(
        default=None, description=_CACHE_BREAKPOINT_DESCRIPTION
    )


# Ref: openai.types.responses.response_input_file.ResponseInputFile
class ResponseInputFile(BaseModelRequest):
    """A file input to the model."""

    type: Literal["input_file"] = Field(description="Input file type.")
    file_data: str | None = Field(default=None, description="File content.")
    file_id: str | None = Field(default=None, description="File ID.")
    file_url: str | None = Field(default=None, description="File URL.")
    filename: str | None = Field(default=None, description="Filename.")
    prompt_cache_breakpoint: PromptCacheBreakpoint | None = Field(
        default=None, description=_CACHE_BREAKPOINT_DESCRIPTION
    )


# Not in the OpenAI spec's InputContent: Codex echoes conversation history as an
# EasyInputMessage with role="assistant" and "output_text" content blocks, and
# EasyInputMessage matches the union before the spec-compliant
# ResponseOutputMessage (both use type="message"). Extra fields (annotations,
# logprobs) are accepted so the original output validates unchanged.
class ResponseOutputTextContent(BaseModelRequestWithExtra):
    """An output_text content block echoed back in the input array (previous assistant response)."""

    type: Literal["output_text"] = Field(description="Output text type.")
    text: str = Field(description="Text content from assistant response.")


# Ref: openai.types.responses.response_input_content.ResponseInputContent
ResponseInputContent = Annotated[
    ResponseInputText
    | ResponseInputImage
    | ResponseInputFile
    | ResponseOutputTextContent,
    Field(discriminator="type"),
]

#: List of input content items for a message.
ResponseInputMessageContentList = list[ResponseInputContent]


# Computer tool call output screenshot  (shared by input and output)


# Ref: openai.types.responses.response_computer_tool_call_output_screenshot.ResponseComputerToolCallOutputScreenshot
class ResponseComputerToolCallOutputScreenshot(BaseModelRequest):
    """A computer screenshot image used with the computer use tool."""

    type: Literal["computer_screenshot"] = Field(description="Computer screenshot.")
    file_id: str | None = Field(
        default=None, description="Uploaded file with screenshot."
    )
    image_url: str | None = Field(default=None, description="Screenshot URL.")


# Shell call output content  (used by ShellCall input item)


# Ref: openai.types.responses.response_function_shell_call_output_content.OutcomeTimeout
class ShellCallOutcomeTimeout(BaseModelRequest):
    """Indicates that the shell call exceeded its configured time limit."""

    type: Literal["timeout"] = Field(description="Timeout outcome.")


# Ref: openai.types.responses.response_function_shell_call_output_content.OutcomeExit
class ShellCallOutcomeExit(BaseModelRequest):
    """Indicates that the shell commands finished and returned an exit code."""

    exit_code: int = Field(description="Shell exit code.")
    type: Literal["exit"] = Field(description="Exit outcome.")


# Ref: openai.types.responses.response_function_shell_call_output_content.Outcome
ShellCallOutcome = Annotated[
    ShellCallOutcomeTimeout | ShellCallOutcomeExit, Field(discriminator="type")
]


# Ref: openai.types.responses.response_function_shell_call_output_content.ResponseFunctionShellCallOutputContent
class ShellCallOutputContent(BaseModelRequest):
    """Captured stdout and stderr for a portion of a shell tool call output."""

    outcome: ShellCallOutcome = Field(description="Exit or timeout outcome.")
    stderr: str = Field(description="Captured stderr.")
    stdout: str = Field(description="Captured stdout.")


# Apply patch operations  (used by ApplyPatchCall input item)


# Ref: openai.types.responses.response_input_item.ApplyPatchCallOperationCreateFile
class ApplyPatchOperationCreateFile(BaseModelRequest):
    """Instruction for creating a new file via the apply_patch tool."""

    diff: str = Field(description="Diff content for new file.")
    path: str = Field(description="Path relative to workspace root.")
    type: Literal["create_file"] = Field(description="Create file operation.")


# Ref: openai.types.responses.response_input_item.ApplyPatchCallOperationDeleteFile
class ApplyPatchOperationDeleteFile(BaseModelRequest):
    """Instruction for deleting an existing file via the apply_patch tool."""

    path: str = Field(description="Path to delete.")
    type: Literal["delete_file"] = Field(description="Delete file operation.")


# Ref: openai.types.responses.response_input_item.ApplyPatchCallOperationUpdateFile
class ApplyPatchOperationUpdateFile(BaseModelRequest):
    """Instruction for updating an existing file via the apply_patch tool."""

    diff: str = Field(description="Diff content to apply.")
    path: str = Field(description="Path to update.")
    type: Literal["update_file"] = Field(description="Update file operation.")


# Ref: openai.types.responses.response_input_item.ApplyPatchCallOperation
ApplyPatchOperation = Annotated[
    ApplyPatchOperationCreateFile
    | ApplyPatchOperationDeleteFile
    | ApplyPatchOperationUpdateFile,
    Field(discriminator="type"),
]


# Input items  (used in ResponseCreateParams.input)


# Ref: openai.types.responses.response_function_tool_call.CallerDirect
class CallerDirect(BaseModelRequest):
    """A tool call made directly by the model."""

    type: Literal["direct"] = Field(description="Direct caller type.")


# Ref: openai.types.responses.response_function_tool_call.CallerProgram
class CallerProgram(BaseModelRequest):
    """A tool call made by a program during programmatic tool calling."""

    caller_id: str = Field(description="ID of the program that made the call.")
    type: Literal["program"] = Field(description="Program caller type.")


# Ref: openai.types.responses.response_function_tool_call.ResponseFunctionToolCall.caller
Caller = Annotated[CallerDirect | CallerProgram, Field(discriminator="type")] | None


# Ref: openai.types.responses.easy_input_message.EasyInputMessage
class EasyInputMessage(BaseModelRequest):
    """A message input to the model with a role indicating instruction following hierarchy.

    ``id`` is absent from the SDK's request type but present on every message item
    the API hands back, so a client replaying a listed item -- Codex does, from the
    first turn on -- sends it. It is read-only here; only the role and content act.
    """

    content: str | ResponseInputMessageContentList = Field(
        description="Text, image, or audio input for the model."
    )
    role: Literal["user", "assistant", "system", "developer"] = Field(
        description="Message role: `user`, `assistant`, `system`, or `developer`."
    )
    id: str | None = Field(default=None, description="Message item ID.")
    phase: Literal["commentary", "final_answer"] | None = Field(
        default=None,
        description="Labels assistant message as commentary or final answer.",
    )
    type: Literal["message"] | None = Field(
        default=None, description="Message input type."
    )


# Ref: openai.types.responses.response_input_item.Message
class InputMessage(BaseModelRequest):
    """A message input with a restricted set of roles (no `assistant`).

    Accepts the echoed ``id`` for the same reason as :class:`EasyInputMessage`.
    """

    content: ResponseInputMessageContentList = Field(description="Input content items.")
    role: Literal["user", "system", "developer"] = Field(
        description="Message role: `user`, `system`, or `developer`."
    )
    id: str | None = Field(default=None, description="Message item ID.")
    status: ResponseItemStatus | None = Field(
        default=None,
        description="Item status: `in_progress`, `completed`, or `incomplete`.",
    )
    type: Literal["message"] | None = Field(
        default=None, description="Message input type."
    )


# Ref: openai.types.responses.response_input_item.ComputerCallOutputAcknowledgedSafetyCheck
class ComputerCallOutputAcknowledgedSafetyCheck(BaseModelRequest):
    """A pending safety check for the computer call."""

    id: str = Field(description="Safety check ID.")
    code: str | None = Field(default=None, description="Safety check type.")
    message: str | None = Field(default=None, description="Safety check details.")


# Ref: openai.types.responses.response_input_item.ComputerCallOutput
class ComputerCallOutput(BaseModelRequest):
    """The output of a computer tool call."""

    call_id: str = Field(description="Computer tool call ID.")
    output: ResponseComputerToolCallOutputScreenshot = Field(
        description="Computer screenshot."
    )
    type: Literal["computer_call_output"] = Field(
        description="Computer call output type."
    )
    id: str | None = Field(default=None, description="Output ID.")
    acknowledged_safety_checks: (
        list[ComputerCallOutputAcknowledgedSafetyCheck] | None
    ) = Field(default=None, description="Acknowledged safety checks.")
    status: ResponseItemStatus | None = Field(
        default=None, description="Status: `in_progress`, `completed`, or `incomplete`."
    )


# Ref: openai.types.responses.response_function_tool_call.ResponseFunctionToolCall
# Spec: components/schemas/FunctionToolCall (openapi.documented.yml)
class FunctionCallInput(BaseModelRequest):
    """A function tool call echoed back as an input item from a previous response."""

    arguments: str = Field(
        description="A JSON string of the arguments passed to the function."
    )
    call_id: str = Field(
        description="The unique ID of the function tool call generated by the model."
    )
    name: str = Field(description="The name of the function that was called.")
    type: Literal["function_call"] = Field(
        description="The type of the function tool call. Always `function_call`."
    )
    id: str | None = Field(
        default=None, description="The unique ID of the function tool call."
    )
    caller: Caller = Field(
        default=None,
        description="Provenance of this tool call: direct or programmatic.",
    )
    namespace: str | None = Field(
        default=None, description="The namespace of the function to run."
    )
    status: ResponseItemStatus | None = Field(
        default=None, description="The status of the item."
    )


# Ref: openai.types.responses.response_input_item.FunctionCallOutput
class FunctionCallOutput(BaseModelRequest):
    """The output of a function tool call."""

    call_id: str = Field(
        description="The unique ID of the function tool call generated by the model."
    )
    output: str | list[ResponseInputContent] = Field(
        description="Text, image, or file output of the function tool call."
    )
    type: Literal["function_call_output"] = Field(
        description="The type of the function tool call output. Always `function_call_output`."
    )
    id: str | None = Field(
        default=None,
        description="The unique ID of the function tool call output. Populated when this item is returned via API.",
    )
    caller: Caller = Field(
        default=None,
        description="Provenance of this tool call: direct or programmatic.",
    )
    status: ResponseItemStatus | None = Field(
        default=None,
        description="The status of the item. One of `in_progress`, `completed`, or `incomplete`. Populated when items are returned via API.",
    )


# Ref: openai.types.responses.response_input_item.ToolSearchCall
class ToolSearchCallInput(BaseModelRequest):
    """A tool search call input item."""

    arguments: object = Field(
        description="The arguments supplied to the tool search call."
    )
    type: Literal["tool_search_call"] = Field(
        description="The item type. Always `tool_search_call`."
    )
    id: str | None = Field(
        default=None, description="The unique ID of this tool search call."
    )
    call_id: str | None = Field(
        default=None,
        description="The unique ID of the tool search call generated by the model.",
    )
    execution: Literal["server", "client"] | None = Field(
        default=None,
        description="Whether tool search was executed by the server or by the client.",
    )
    status: ResponseItemStatus | None = Field(
        default=None, description="The status of the tool search call."
    )


# Ref: openai.types.responses.response_input_item.ResponseToolSearchOutputItemParam
# NOTE: Echoed back as input so clients can replay full previous responses without
# validation errors.  Has no Bedrock equivalent and is dropped during input mapping;
# `tools` is typed loosely (JsonMapping) to stay tolerant of upstream schema evolution.
class ToolSearchOutputInput(BaseModelRequest):
    """The loaded tool definitions returned by a tool search call (as input item)."""

    tools: list[JsonMapping] = Field(
        description="The loaded tool definitions returned by the tool search output."
    )
    type: Literal["tool_search_output"] = Field(
        description="The item type. Always `tool_search_output`."
    )
    id: str | None = Field(
        default=None, description="The unique ID of this tool search output."
    )
    call_id: str | None = Field(
        default=None,
        description="The unique ID of the tool search call generated by the model.",
    )
    execution: Literal["server", "client"] | None = Field(
        default=None,
        description="Whether tool search was executed by the server or by the client.",
    )
    status: Literal["in_progress", "completed", "incomplete"] | None = Field(
        default=None, description="The status of the tool search output."
    )


# Ref: openai.types.responses.response_input_item.ImageGenerationCall (input variant)
class ImageGenerationCallInput(BaseModelRequest):
    """An image generation call as an input item."""

    id: str = Field(description="The unique ID of the image generation call.")
    status: Literal["in_progress", "completed", "generating", "failed"] = Field(
        description="The status of the image generation call."
    )
    type: Literal["image_generation_call"] = Field(
        description="The type of the image generation call. Always `image_generation_call`."
    )
    result: str | None = Field(
        default=None, description="The generated image encoded in base64."
    )


# Ref: openai.types.responses.response_input_item.LocalShellCallAction
class LocalShellCallActionInput(BaseModelRequest):
    """Execute a shell command on the server."""

    command: list[str] = Field(description="Command to run.")
    env: dict[str, str] = Field(description="Environment variables.")
    type: Literal["exec"] = Field(description="Exec action type.")
    timeout_ms: int | None = Field(default=None, description="Timeout in milliseconds.")
    user: str | None = Field(default=None, description="User to run as.")
    working_directory: str | None = Field(
        default=None, description="Working directory."
    )


# Ref: openai.types.responses.response_input_item.LocalShellCall (input variant)
class LocalShellCallInput(BaseModelRequest):
    """A tool call to run a command on the local shell (as input item)."""

    id: str = Field(description="The unique ID of the local shell call.")
    action: LocalShellCallActionInput = Field(
        description="Execute a shell command on the server."
    )
    call_id: str = Field(
        description="The unique ID of the local shell tool call generated by the model."
    )
    status: Literal["in_progress", "completed", "incomplete"] = Field(
        description="The status of the local shell call."
    )
    type: Literal["local_shell_call"] = Field(
        description="The type of the local shell call. Always `local_shell_call`."
    )


# Ref: openai.types.responses.response_input_item.LocalShellCallOutput (input variant)
class LocalShellCallOutputInput(BaseModelRequest):
    """The output of a local shell tool call (as input item)."""

    id: str = Field(
        description="The unique ID of the local shell tool call generated by the model."
    )
    output: str = Field(
        description="A JSON string of the output of the local shell tool call."
    )
    type: Literal["local_shell_call_output"] = Field(
        description="The type of the local shell tool call output. Always `local_shell_call_output`."
    )
    status: ResponseItemStatus | None = Field(
        default=None,
        description="The status of the item. One of `in_progress`, `completed`, or `incomplete`.",
    )


# Ref: openai.types.responses.response_input_item.ShellCallAction
class ShellCallAction(BaseModelRequest):
    """The shell commands and limits that describe how to run the tool call."""

    commands: list[str] = Field(
        description="Ordered shell commands for the execution environment to run."
    )
    max_output_length: int | None = Field(
        default=None,
        description="Maximum number of UTF-8 characters to capture from combined stdout and stderr output.",
    )
    timeout_ms: int | None = Field(
        default=None,
        description="Maximum wall-clock time in milliseconds to allow the shell commands to run.",
    )


# Ref: openai.types.responses.response_input_item.ShellCallEnvironment
ShellCallEnvironment = (
    Annotated[LocalEnvironment | ContainerReference, Field(discriminator="type")] | None
)


# Ref: openai.types.responses.response_input_item.ShellCall
class ShellCall(BaseModelRequest):
    """A tool representing a request to execute one or more shell commands."""

    action: ShellCallAction = Field(
        description="The shell commands and limits that describe how to run the tool call."
    )
    call_id: str = Field(
        description="The unique ID of the shell tool call generated by the model."
    )
    type: Literal["shell_call"] = Field(
        description="The type of the item. Always `shell_call`."
    )
    id: str | None = Field(
        default=None,
        description="The unique ID of the shell tool call. Populated when this item is returned via API.",
    )
    caller: Caller = Field(
        default=None,
        description="Provenance of this tool call: direct or programmatic.",
    )
    environment: ShellCallEnvironment = Field(
        default=None, description="The environment to execute the shell commands in."
    )
    status: ResponseItemStatus | None = Field(
        default=None, description="The status of the shell call."
    )


# Ref: openai.types.responses.response_input_item.ShellCallOutput
class ShellCallOutput(BaseModelRequest):
    """The streamed output items emitted by a shell tool call."""

    call_id: str = Field(
        description="The unique ID of the shell tool call generated by the model."
    )
    output: list[ShellCallOutputContent] = Field(
        description="Captured chunks of stdout and stderr output, along with their associated outcomes."
    )
    type: Literal["shell_call_output"] = Field(
        description="The type of the item. Always `shell_call_output`."
    )
    id: str | None = Field(
        default=None,
        description="The unique ID of the shell tool call output. Populated when this item is returned via API.",
    )
    caller: Caller = Field(
        default=None,
        description="Provenance of this tool call: direct or programmatic.",
    )
    max_output_length: int | None = Field(
        default=None,
        description="The maximum number of UTF-8 characters captured for this shell call's combined output.",
    )
    status: ResponseItemStatus | None = Field(
        default=None, description="The status of the shell call output."
    )


# Ref: openai.types.responses.response_input_item.ApplyPatchCall
class ApplyPatchCall(BaseModelRequest):
    """A tool call representing a request to create, delete, or update files using diff patches."""

    call_id: str = Field(
        description="The unique ID of the apply patch tool call generated by the model."
    )
    operation: ApplyPatchOperation = Field(
        description="The specific create, delete, or update instruction for the apply_patch tool call."
    )
    status: Literal["in_progress", "completed"] = Field(
        description="The status of the apply patch tool call. One of `in_progress` or `completed`."
    )
    type: Literal["apply_patch_call"] = Field(
        description="The type of the item. Always `apply_patch_call`."
    )
    id: str | None = Field(
        default=None,
        description="The unique ID of the apply patch tool call. Populated when this item is returned via API.",
    )
    caller: Caller = Field(
        default=None,
        description="Provenance of this tool call: direct or programmatic.",
    )


# Ref: openai.types.responses.response_input_item.ApplyPatchCallOutput
class ApplyPatchCallOutput(BaseModelRequest):
    """The streamed output emitted by an apply patch tool call (as input item)."""

    call_id: str = Field(
        description="The unique ID of the apply patch tool call generated by the model."
    )
    status: Literal["completed", "failed"] = Field(
        description="The status of the apply patch tool call output."
    )
    type: Literal["apply_patch_call_output"] = Field(
        description="The type of the item. Always `apply_patch_call_output`."
    )
    id: str | None = Field(
        default=None,
        description="The unique ID of the apply patch tool call output. Populated when this item is returned via API.",
    )
    caller: Caller = Field(
        default=None,
        description="Provenance of this tool call: direct or programmatic.",
    )
    output: str | None = Field(
        default=None,
        description="Optional human-readable log text from the apply patch tool.",
    )


# Ref: openai.types.responses.response_input_item.McpListToolsTool
class McpListToolsToolItem(BaseModelRequest):
    """A tool available on an MCP server."""

    input_schema: object = Field(description="Tool input JSON schema.")
    name: str = Field(description="Tool name.")
    annotations: object | None = Field(default=None, description="Tool annotations.")
    description: str | None = Field(default=None, description="Tool description.")


# Ref: openai.types.responses.response_input_item.McpListTools (input variant)
class McpListToolsInput(BaseModelRequest):
    """A list of tools available on an MCP server (as input item)."""

    id: str = Field(description="The unique ID of the list.")
    server_label: str = Field(description="The label of the MCP server.")
    tools: list[McpListToolsToolItem] = Field(
        description="The tools available on the server."
    )
    type: Literal["mcp_list_tools"] = Field(
        description="The type of the item. Always `mcp_list_tools`."
    )
    error: str | None = Field(
        default=None, description="Error message if the server could not list tools."
    )


# Ref: openai.types.responses.response_input_item.McpApprovalRequest (input variant)
class McpApprovalRequestInput(BaseModelRequest):
    """A request for human approval of a tool invocation (as input item)."""

    id: str = Field(description="Approval request ID.")
    arguments: str = Field(description="Tool arguments JSON.")
    name: str = Field(description="Tool name.")
    server_label: str = Field(description="MCP server label.")
    type: Literal["mcp_approval_request"] = Field(
        description="MCP approval request type."
    )


# Ref: openai.types.responses.response_input_item.McpApprovalResponse
class McpApprovalResponse(BaseModelRequest):
    """A response to an MCP approval request."""

    approval_request_id: str = Field(
        description="The ID of the approval request being answered."
    )
    approve: bool = Field(description="Whether the request was approved.")
    type: Literal["mcp_approval_response"] = Field(
        description="The type of the item. Always `mcp_approval_response`."
    )
    id: str | None = Field(
        default=None, description="The unique ID of the approval response."
    )
    reason: str | None = Field(
        default=None, description="Optional reason for the decision."
    )


# Ref: openai.types.responses.response_input_item.McpCall (input variant)
class McpCallInput(BaseModelRequest):
    """An invocation of a tool on an MCP server (as input item)."""

    id: str = Field(description="Tool call ID.")
    arguments: str = Field(description="Tool arguments JSON.")
    name: str = Field(description="Tool name.")
    server_label: str = Field(description="MCP server label.")
    type: Literal["mcp_call"] = Field(description="MCP call type.")
    approval_request_id: str | None = Field(
        default=None, description="Approval request ID."
    )
    error: str | None = Field(default=None, description="Tool call error.")
    output: str | None = Field(default=None, description="Tool call output.")
    status: (
        Literal["in_progress", "completed", "incomplete", "calling", "failed"] | None
    ) = Field(default=None, description="Tool call status.")


# Ref: openai.types.responses.response_input_item.ItemReference
class ItemReference(BaseModelRequest):
    """An internal identifier for an item to reference."""

    id: str = Field(description="The ID of the item to reference.")
    type: Literal["item_reference"] | None = Field(
        default=None,
        description="The type of item to reference. Always `item_reference`.",
    )


# Ref: openai.types.responses.response_compaction_item_param.ResponseCompactionItemParam
class CompactionItemParam(BaseModelRequest):
    """A compaction item generated by the v1/responses/compact API."""

    encrypted_content: str = Field(
        description="The encrypted content of the compaction summary."
    )
    type: Literal["compaction"] = Field(
        description="The type of the item. Always `compaction`."
    )
    id: str | None = Field(default=None, description="The ID of the compaction item.")


# NOTE: The following hosted-tool call items are accepted as input so that clients
# can echo back full previous responses (including gateway-emitted items) without
# validation errors.  They have no Bedrock equivalent and are dropped during input
# mapping; nested payloads are therefore typed loosely (JsonMapping) to stay
# tolerant of upstream schema evolution.


# Ref: openai.types.responses.response_input_param.ResponseFunctionWebSearchParam (input variant)
class WebSearchCallInput(BaseModelRequest):
    """The results of a web search tool call (as input item)."""

    id: str = Field(description="The unique ID of the web search tool call.")
    action: JsonMapping = Field(
        description="The action taken in this web search call (search, open_page, or find_in_page)."
    )
    status: Literal["in_progress", "searching", "completed", "failed"] = Field(
        description="The status of the web search tool call."
    )
    type: Literal["web_search_call"] = Field(
        description="The type of the web search tool call. Always `web_search_call`."
    )


# Ref: openai.types.responses.response_input_param.ResponseFileSearchToolCallParam (input variant)
class FileSearchCallInput(BaseModelRequest):
    """The results of a file search tool call (as input item)."""

    id: str = Field(description="The unique ID of the file search tool call.")
    queries: list[str] = Field(description="The queries used to search for files.")
    status: Literal["in_progress", "searching", "completed", "incomplete", "failed"] = (
        Field(description="The status of the file search tool call.")
    )
    type: Literal["file_search_call"] = Field(
        description="The type of the file search tool call. Always `file_search_call`."
    )
    results: list[JsonMapping] | None = Field(
        default=None, description="The results of the file search tool call."
    )


# Ref: openai.types.responses.response_input_param.ResponseCodeInterpreterToolCallParam (input variant)
class CodeInterpreterCallInput(BaseModelRequest):
    """A tool call to run code (as input item)."""

    id: str = Field(description="The unique ID of the code interpreter tool call.")
    container_id: str = Field(
        description="The ID of the container used to run the code."
    )
    status: Literal[
        "in_progress", "completed", "incomplete", "interpreting", "failed"
    ] = Field(description="The status of the code interpreter tool call.")
    type: Literal["code_interpreter_call"] = Field(
        description="The type of the code interpreter tool call. Always `code_interpreter_call`."
    )
    code: str | None = Field(
        default=None, description="The code to run, or null if not available."
    )
    outputs: list[JsonMapping] | None = Field(
        default=None,
        description="The outputs generated by the code interpreter (logs or images).",
    )


# Ref: openai.types.responses.response_input_param.ResponseComputerToolCallParam (input variant)
class ComputerCallInput(BaseModelRequest):
    """A tool call to a computer use tool (as input item)."""

    id: str = Field(description="The unique ID of the computer call.")
    call_id: str = Field(
        description="An identifier used when responding to the tool call with output."
    )
    pending_safety_checks: list[JsonMapping] = Field(
        description="The pending safety checks for the computer call."
    )
    status: ResponseItemStatus = Field(description="The status of the item.")
    type: Literal["computer_call"] = Field(
        description="The type of the computer call. Always `computer_call`."
    )
    action: JsonMapping | None = Field(
        default=None, description="The action to perform."
    )
    actions: list[JsonMapping] | None = Field(
        default=None, description="Flattened batched actions for `computer_use`."
    )


# Ref: openai.types.responses.response_custom_tool_call_param.ResponseCustomToolCallParam (input variant)
class CustomToolCallInput(BaseModelRequest):
    """A call to a custom tool created by the model (as input item)."""

    call_id: str = Field(
        description="An identifier used to map this custom tool call to a tool call output."
    )
    input: str = Field(
        description="The input for the custom tool call generated by the model."
    )
    name: str = Field(description="The name of the custom tool being called.")
    type: Literal["custom_tool_call"] = Field(
        description="The type of the custom tool call. Always `custom_tool_call`."
    )
    id: str | None = Field(
        default=None, description="The unique ID of the custom tool call."
    )
    caller: Caller = Field(
        default=None,
        description="Provenance of this tool call: direct or programmatic.",
    )
    namespace: str | None = Field(
        default=None, description="The namespace of the custom tool being called."
    )


# Ref: openai.types.responses.response_custom_tool_call_output_param.ResponseCustomToolCallOutputParam (input variant)
class CustomToolCallOutput(BaseModelRequest):
    """The output of a custom tool call from your code, being sent back to the model."""

    call_id: str = Field(
        description="The call ID, used to map this custom tool call output to a custom tool call."
    )
    output: str | list[ResponseInputContent] = Field(
        description="The output from the custom tool call. Can be a string or a list of output content."
    )
    type: Literal["custom_tool_call_output"] = Field(
        description="The type of the custom tool call output. Always `custom_tool_call_output`."
    )
    id: str | None = Field(
        default=None, description="The unique ID of the custom tool call output."
    )
    caller: Caller = Field(
        default=None,
        description="Provenance of this tool call: direct or programmatic.",
    )


# Ref: openai.types.responses.response_output_item.AdditionalTools (input variant)
class AdditionalToolsInput(BaseModelRequest):
    """An echoed additional-tools advertisement item (as input item)."""

    role: Literal[
        "unknown",
        "user",
        "assistant",
        "system",
        "critic",
        "discriminator",
        "developer",
        "tool",
    ] = Field(description="The role that provided the additional tools.")
    tools: list[Tool] = Field(description="The additional tools.")
    type: Literal["additional_tools"] = Field(
        description="The type of the item. Always `additional_tools`."
    )
    id: str | None = Field(default=None, description="The unique ID of the item.")


# Ref: openai.types.responses.response_input_item.Program
class ProgramInput(BaseModelRequest):
    """A program emitted by programmatic tool calling (as input item)."""

    id: str = Field(description="The unique ID of the program.")
    call_id: str = Field(
        description="An identifier used to map this program to its output."
    )
    code: str = Field(description="The code executed by the program.")
    fingerprint: str = Field(description="The fingerprint of the program.")
    type: Literal["program"] = Field(
        description="The type of the item. Always `program`."
    )


# Ref: openai.types.responses.response_input_item.ProgramOutput
class ProgramOutputInput(BaseModelRequest):
    """The result of a program execution (as input item)."""

    id: str = Field(description="The unique ID of the program output.")
    call_id: str = Field(
        description="The call ID, used to map this output to its program."
    )
    result: str = Field(description="The result of the program execution.")
    status: Literal["completed", "incomplete"] = Field(
        description="The status of the program execution."
    )
    type: Literal["program_output"] = Field(
        description="The type of the item. Always `program_output`."
    )


# Ref: openai.types.responses.response_input_param.CompactionTrigger
class CompactionTrigger(BaseModelRequest):
    """Compacts the current context. Must be the final input item."""

    type: Literal["compaction_trigger"] = Field(
        description="The type of the item. Always `compaction_trigger`."
    )


# Ref: openai.types.responses.response_input_item.ResponseInputItem
# EasyInputMessage and InputMessage share type="message", so a discriminated
# union cannot be used here.
ResponseInputItem = (
    EasyInputMessage
    | InputMessage
    | ComputerCallOutput
    | FunctionCallInput
    | FunctionCallOutput
    | ToolSearchCallInput
    | ToolSearchOutputInput
    | ImageGenerationCallInput
    | LocalShellCallInput
    | LocalShellCallOutputInput
    | ShellCall
    | ShellCallOutput
    | ApplyPatchCall
    | ApplyPatchCallOutput
    | McpListToolsInput
    | McpApprovalRequestInput
    | McpApprovalResponse
    | McpCallInput
    | WebSearchCallInput
    | FileSearchCallInput
    | CodeInterpreterCallInput
    | ComputerCallInput
    | CustomToolCallInput
    | CustomToolCallOutput
    | AdditionalToolsInput
    | ProgramInput
    | ProgramOutputInput
    | CompactionTrigger
    | CompactionItemParam
    | ItemReference
)

#: The `input` parameter for a response creation request.
# ResponseInputItem is extended below with the echoed output-item types.
type ResponseInputParam = str | list[ResponseInputItem]


# Response output content  (model-generated text/refusal with annotations)


# Ref: openai.types.responses.response_output_text.AnnotationFileCitation
class AnnotationFileCitation(BaseModelResponse):
    """A citation to a file."""

    file_id: str = Field(description="File ID.")
    filename: str = Field(description="Filename.")
    index: int = Field(description="File index.")
    type: Literal["file_citation"] = Field(description="File citation type.")


# Ref: openai.types.responses.response_output_text.AnnotationURLCitation
class AnnotationURLCitation(BaseModelResponse):
    """A citation for a web resource used to generate a model response."""

    end_index: int = Field(description="Last character index of URL citation.")
    start_index: int = Field(description="First character index of URL citation.")
    title: str = Field(description="Web resource title.")
    type: Literal["url_citation"] = Field(description="URL citation type.")
    url: str = Field(description="Web resource URL.")


# Ref: openai.types.responses.response_output_text.AnnotationContainerFileCitation
class AnnotationContainerFileCitation(BaseModelResponse):
    """A citation for a container file used to generate a model response."""

    container_id: str = Field(description="Container file ID.")
    end_index: int = Field(
        description="Last character index of container file citation."
    )
    file_id: str = Field(description="File ID.")
    filename: str = Field(description="Container filename.")
    start_index: int = Field(
        description="First character index of container file citation."
    )
    type: Literal["container_file_citation"] = Field(
        description="Container file citation type."
    )


# Ref: openai.types.responses.response_output_text.AnnotationFilePath
class AnnotationFilePath(BaseModelResponse):
    """A path to a file."""

    file_id: str = Field(description="File ID.")
    index: int = Field(description="File index.")
    type: Literal["file_path"] = Field(description="File path type.")


# Ref: openai.types.responses.response_output_text.Annotation
Annotation = Annotated[
    AnnotationFileCitation
    | AnnotationURLCitation
    | AnnotationContainerFileCitation
    | AnnotationFilePath,
    Field(discriminator="type"),
]


# Ref: openai.types.responses.response_output_text.LogprobTopLogprob
class LogprobTopLogprob(BaseModelResponse):
    """A possible token with its log probability."""

    token: str | None = Field(default=None, description="Possible text token.")
    logprob: float | None = Field(default=None, description="Log probability.")


# Ref: openai.types.responses.response_output_text.Logprob
class Logprob(BaseModelResponse):
    """The log probability of a token."""

    token: str = Field(description="Text token.")
    bytes: list[int] = Field(description="Token bytes.")
    logprob: float = Field(description="Log probability.")
    top_logprobs: list[LogprobTopLogprob] = Field(description="Top log probabilities.")


# Ref: openai.types.responses.response_output_text.ResponseOutputText
class ResponseOutputText(BaseModelResponse):
    """A text output from the model."""

    annotations: list[Annotation] = Field(description="Text annotations.")
    text: str = Field(description="Model text output.")
    type: Literal["output_text"] = Field(description="Output text type.")
    logprobs: list[Logprob] | None = Field(
        default=None, description="Output token log probabilities."
    )


# Ref: openai.types.responses.response_output_refusal.ResponseOutputRefusal
class ResponseOutputRefusal(BaseModelResponse):
    """A refusal from the model."""

    refusal: str = Field(description="Refusal explanation.")
    type: Literal["refusal"] = Field(description="Refusal type.")


# Ref: openai.types.responses.response_output_message.Content
ResponseOutputMessageContent = Annotated[
    ResponseOutputText | ResponseOutputRefusal, Field(discriminator="type")
]


# Ref: openai.types.responses.response_output_message.ResponseOutputMessage
class ResponseOutputMessage(BaseModelResponse):
    """An output message from the model."""

    id: str = Field(description="Output message ID.")
    content: list[ResponseOutputMessageContent] = Field(description="Message content.")
    role: Literal["assistant"] = Field(description="Assistant role.")
    status: Literal["in_progress", "completed", "incomplete"] = Field(
        description="Message status."
    )
    type: Literal["message"] = Field(description="Message type.")
    phase: Literal["commentary", "final_answer"] | None = Field(
        default=None,
        description="Labels assistant message as commentary or final answer.",
    )


# Response function tool call


# Ref: openai.types.responses.response_function_tool_call.ResponseFunctionToolCall
class ResponseFunctionToolCall(BaseModelResponse):
    """A tool call to run a function."""

    arguments: str = Field(description="JSON string of function arguments.")
    call_id: str = Field(description="Function tool call ID.")
    name: str = Field(description="Function name.")
    type: Literal["function_call"] = Field(description="Function call type.")
    id: str | None = Field(default=None, description="Function call unique ID.")
    caller: Caller = Field(
        default=None,
        description="Provenance of this tool call: direct or programmatic.",
    )
    namespace: str | None = Field(default=None, description="Function namespace.")
    status: ResponseItemStatus | None = Field(
        default=None,
        description="Item status: `in_progress`, `completed`, or `incomplete`.",
    )


# Response file search tool call


# Ref: openai.types.responses.response_file_search_tool_call.Result
class FileSearchResult(BaseModelResponse):
    """A file search result."""

    attributes: dict[str, str | float | bool] | None = Field(
        default=None, description="Key-value pairs for the file."
    )
    file_id: str | None = Field(default=None, description="File ID.")
    filename: str | None = Field(default=None, description="Filename.")
    score: float | None = Field(default=None, description="Relevance score (0-1).")
    text: str | None = Field(default=None, description="Retrieved text from file.")


# Ref: openai.types.responses.response_file_search_tool_call.ResponseFileSearchToolCall
class ResponseFileSearchToolCall(BaseModelResponse):
    """The results of a file search tool call."""

    id: str = Field(description="File search tool call ID.")
    queries: list[str] = Field(description="Search queries.")
    status: Literal["in_progress", "searching", "completed", "incomplete", "failed"] = (
        Field(description="File search status.")
    )
    type: Literal["file_search_call"] = Field(description="File search call type.")
    results: list[FileSearchResult] | None = Field(
        default=None, description="File search results."
    )


# Response web search tool call


# Ref: openai.types.responses.response_function_web_search.ActionSearchSource
class WebSearchActionSource(BaseModelResponse):
    """A source used in the search."""

    type: Literal["url"] = Field(description="URL source type.")
    url: str = Field(description="Source URL.")


# Ref: openai.types.responses.response_function_web_search.ActionSearch
class WebSearchActionSearch(BaseModelResponse):
    """Web search action of type `search`."""

    query: str = Field(description="[DEPRECATED] Search query.")
    type: Literal["search"] = Field(description="Search action type.")
    queries: list[str] | None = Field(default=None, description="Search queries.")
    sources: list[WebSearchActionSource] | None = Field(
        default=None, description="Search sources."
    )


# Ref: openai.types.responses.response_function_web_search.ActionOpenPage
class WebSearchActionOpenPage(BaseModelResponse):
    """Web search action of type `open_page`."""

    type: Literal["open_page"] = Field(description="Open page action type.")
    url: str | None = Field(default=None, description="Opened URL.")


# Ref: openai.types.responses.response_function_web_search.ActionFind
class WebSearchActionFind(BaseModelResponse):
    """Web search action of type `find_in_page`."""

    pattern: str = Field(description="Pattern to find in page.")
    type: Literal["find_in_page"] = Field(description="Find in page action type.")
    url: str = Field(description="Page URL searched.")


# Ref: openai.types.responses.response_function_web_search.Action
WebSearchAction = Annotated[
    WebSearchActionSearch | WebSearchActionOpenPage | WebSearchActionFind,
    Field(discriminator="type"),
]


# Ref: openai.types.responses.response_function_web_search.ResponseFunctionWebSearch
class ResponseFunctionWebSearch(BaseModelResponse):
    """The results of a web search tool call."""

    id: str = Field(description="Web search tool call ID.")
    action: WebSearchAction = Field(description="Web search action taken.")
    status: Literal["in_progress", "searching", "completed", "failed"] = Field(
        description="Web search status."
    )
    type: Literal["web_search_call"] = Field(description="Web search call type.")


# Response computer tool call  (with all actions)


# Ref: openai.types.responses.response_computer_tool_call.PendingSafetyCheck
class PendingSafetyCheck(BaseModelResponse):
    """A pending safety check for the computer call."""

    id: str = Field(description="Safety check ID.")
    code: str | None = Field(default=None, description="Safety check type.")
    message: str | None = Field(default=None, description="Safety check details.")


# Ref: openai.types.responses.response_computer_tool_call.ActionClick
class ComputerActionClick(BaseModelResponse):
    """A click action."""

    button: Literal["left", "right", "wheel", "back", "forward"] = Field(
        description="Mouse button pressed."
    )
    type: Literal["click"] = Field(description="Click action type.")
    x: int = Field(description="Click x-coordinate.")
    y: int = Field(description="Click y-coordinate.")
    keys: list[str] | None = Field(
        default=None, description="Keys held while clicking."
    )


# Ref: openai.types.responses.response_computer_tool_call.ActionDoubleClick
class ComputerActionDoubleClick(BaseModelResponse):
    """A double click action."""

    type: Literal["double_click"] = Field(description="Double click action type.")
    x: int = Field(description="Double click x-coordinate.")
    y: int = Field(description="Double click y-coordinate.")
    keys: list[str] | None = Field(
        default=None, description="Keys held while double-clicking."
    )


# Ref: openai.types.responses.response_computer_tool_call.ActionDragPath
class ComputerActionDragPath(BaseModelResponse):
    """An x/y coordinate pair."""

    x: int = Field(description="X-coordinate.")
    y: int = Field(description="Y-coordinate.")


# Ref: openai.types.responses.response_computer_tool_call.ActionDrag
class ComputerActionDrag(BaseModelResponse):
    """A drag action."""

    path: list[ComputerActionDragPath] = Field(description="Drag path coordinates.")
    type: Literal["drag"] = Field(description="Drag action type.")
    keys: list[str] | None = Field(
        default=None, description="Keys held while dragging."
    )


# Ref: openai.types.responses.response_computer_tool_call.ActionKeypress
class ComputerActionKeypress(BaseModelResponse):
    """A collection of keypresses."""

    keys: list[str] = Field(description="Keys to press.")
    type: Literal["keypress"] = Field(description="Keypress action type.")


# Ref: openai.types.responses.response_computer_tool_call.ActionMove
class ComputerActionMove(BaseModelResponse):
    """A mouse move action."""

    type: Literal["move"] = Field(description="Move action type.")
    x: int = Field(description="Target x-coordinate.")
    y: int = Field(description="Target y-coordinate.")
    keys: list[str] | None = Field(default=None, description="Keys held while moving.")


# Ref: openai.types.responses.response_computer_tool_call.ActionScreenshot
class ComputerActionScreenshot(BaseModelResponse):
    """A screenshot action."""

    type: Literal["screenshot"] = Field(description="Screenshot action type.")


# Ref: openai.types.responses.response_computer_tool_call.ActionScroll
class ComputerActionScroll(BaseModelResponse):
    """A scroll action."""

    scroll_x: int = Field(description="Horizontal scroll distance.")
    scroll_y: int = Field(description="Vertical scroll distance.")
    type: Literal["scroll"] = Field(description="Scroll action type.")
    x: int = Field(description="Scroll x-coordinate.")
    y: int = Field(description="Scroll y-coordinate.")
    keys: list[str] | None = Field(
        default=None, description="Keys held while scrolling."
    )


# Ref: openai.types.responses.response_computer_tool_call.ActionType
class ComputerActionType(BaseModelResponse):
    """An action to type in text."""

    text: str = Field(description="Text to type.")
    type: Literal["type"] = Field(description="Type action type.")


# Ref: openai.types.responses.response_computer_tool_call.ActionWait
class ComputerActionWait(BaseModelResponse):
    """A wait action."""

    type: Literal["wait"] = Field(description="Wait action type.")


# Ref: openai.types.responses.computer_action.ComputerAction
ComputerAction = Annotated[
    ComputerActionClick
    | ComputerActionDoubleClick
    | ComputerActionDrag
    | ComputerActionKeypress
    | ComputerActionMove
    | ComputerActionScreenshot
    | ComputerActionScroll
    | ComputerActionType
    | ComputerActionWait,
    Field(discriminator="type"),
]

#: A list of computer actions.
ComputerActionList = list[ComputerAction]


# Ref: openai.types.responses.response_computer_tool_call.ResponseComputerToolCall
class ResponseComputerToolCall(BaseModelResponse):
    """A tool call to a computer use tool."""

    id: str = Field(description="Computer call ID.")
    call_id: str = Field(description="Response identifier.")
    pending_safety_checks: list[PendingSafetyCheck] = Field(
        description="Pending safety checks."
    )
    status: Literal["in_progress", "completed", "incomplete"] = Field(
        description="Call status."
    )
    type: Literal["computer_call"] = Field(description="Computer call type.")
    action: ComputerAction | None = Field(
        default=None, description="Action to perform."
    )
    actions: ComputerActionList | None = Field(
        default=None, description="Batched actions for computer_use."
    )


# Ref: openai.types.responses.response_computer_tool_call_output_item.AcknowledgedSafetyCheck
class AcknowledgedSafetyCheck(BaseModelResponse):
    """An acknowledged safety check for a computer call output."""

    id: str = Field(description="Safety check ID.")
    code: str | None = Field(default=None, description="Safety check type.")
    message: str | None = Field(default=None, description="Safety check details.")


# Ref: openai.types.responses.response_computer_tool_call_output_item.ResponseComputerToolCallOutputItem
class ResponseComputerToolCallOutputItem(BaseModelResponse):
    """The output of a computer tool call."""

    id: str = Field(description="Computer call output ID.")
    call_id: str = Field(description="Computer tool call ID.")
    output: ResponseComputerToolCallOutputScreenshot = Field(
        description="Computer screenshot."
    )
    status: Literal["completed", "incomplete", "failed", "in_progress"] = Field(
        description="Output status."
    )
    type: Literal["computer_call_output"] = Field(
        description="Computer call output type."
    )
    acknowledged_safety_checks: list[AcknowledgedSafetyCheck] | None = Field(
        default=None, description="Acknowledged safety checks."
    )
    created_by: str | None = Field(default=None, description="Item creator.")


# Response code interpreter tool call


# Ref: openai.types.responses.response_code_interpreter_tool_call.OutputLogs
class CodeInterpreterOutputLogs(BaseModelResponse):
    """The logs output from the code interpreter."""

    logs: str = Field(description="Code interpreter logs.")
    type: Literal["logs"] = Field(description="Logs output type.")


# Ref: openai.types.responses.response_code_interpreter_tool_call.OutputImage
class CodeInterpreterOutputImage(BaseModelResponse):
    """The image output from the code interpreter."""

    type: Literal["image"] = Field(description="Image output type.")
    url: str = Field(description="Image output URL.")


# Ref: openai.types.responses.response_code_interpreter_tool_call.Output
CodeInterpreterOutput = Annotated[
    CodeInterpreterOutputLogs | CodeInterpreterOutputImage, Field(discriminator="type")
]


# Ref: openai.types.responses.response_code_interpreter_tool_call.ResponseCodeInterpreterToolCall
class ResponseCodeInterpreterToolCall(BaseModelResponse):
    """A tool call to run code."""

    id: str = Field(description="Code interpreter tool call ID.")
    container_id: str = Field(description="Container ID.")
    status: Literal[
        "in_progress", "completed", "incomplete", "interpreting", "failed"
    ] = Field(description="Code interpreter status.")
    type: Literal["code_interpreter_call"] = Field(
        description="Code interpreter call type."
    )
    code: str | None = Field(default=None, description="Code to run.")
    outputs: list[CodeInterpreterOutput] | None = Field(
        default=None, description="Code interpreter outputs (logs or images)."
    )


# Response reasoning item


# Ref: openai.types.responses.response_reasoning_item.Summary
class ReasoningItemSummary(BaseModelResponse):
    """A summary text from the model."""

    text: str = Field(description="Reasoning summary.")
    type: Literal["summary_text"] = Field(description="Summary text type.")


# Ref: openai.types.responses.response_reasoning_item.Content
class ReasoningItemContent(BaseModelResponse):
    """Reasoning text from the model."""

    text: str = Field(description="Reasoning text.")
    type: Literal["reasoning_text"] = Field(description="Reasoning text type.")


# Ref: openai.types.responses.response_reasoning_item.ResponseReasoningItem
class ResponseReasoningItem(BaseModelResponse):
    """A description of the chain of thought used by a reasoning model while generating a response."""

    id: str = Field(description="Reasoning content ID.")
    summary: list[ReasoningItemSummary] = Field(description="Reasoning summary.")
    type: Literal["reasoning"] = Field(description="Reasoning type.")
    content: list[ReasoningItemContent] | None = Field(
        default=None, description="Reasoning text content."
    )
    encrypted_content: str | None = Field(
        default=None, description="Encrypted reasoning content when included."
    )
    status: ResponseItemStatus | None = Field(
        default=None, description="Status: `in_progress`, `completed`, or `incomplete`."
    )


# Input-tolerant variants of echoed output items.  Response models forbid extra
# fields, but clients replay previous API responses verbatim, so unknown upstream
# fields must never fail input validation (regardless of strict_input_validation).
class ResponseOutputMessageInput(ResponseOutputMessage):
    """An echoed output message accepted as an input item (ignores unknown fields).

    The content parts are widened to the input shapes as well: a client replaying
    an assistant turn may relabel its text parts as ``input_text`` rather than
    ``output_text`` -- Codex does -- and the item is only being read back, so the
    label carries nothing to act on. Refusing it would fail the turn over an echo
    of the gateway's own output.
    """

    model_config = ConfigDict(extra="ignore")

    content: list[ResponseOutputMessageContent | ResponseInputContent] = Field(  # type: ignore[assignment]
        description="Message content."
    )


class ReasoningItemContentInput(ReasoningItemContent):
    """Echoed reasoning text, tolerating any part type the client replays.

    A reasoning item arrives here only because the client is handing back an item
    it was given, so the part type carries no instruction to act on: only the text
    is read. Clients differ on what they put there -- ``reasoning_text``,
    ``summary_text``, and Codex's legacy ``text`` are all seen in the wild, and
    Codex replays a reasoning item as the *first* input item of the following
    turn. Rejecting an unrecognised value fails the whole turn over an echo, so
    the type is accepted as-is and ignored.
    """

    model_config = ConfigDict(extra="ignore")

    type: str = Field(  # type: ignore[assignment]
        description="Reasoning text type, such as `reasoning_text`."
    )


class ResponseReasoningItemInput(ResponseReasoningItem):
    """An echoed reasoning item accepted as an input item (ignores unknown fields).

    Tolerates client serializations seen in the wild (Codex): a missing
    ``summary``, a null ``id``, and ``text``-typed content parts.
    """

    model_config = ConfigDict(extra="ignore")

    id: str | None = Field(  # type: ignore[assignment]
        default=None, description="Reasoning content ID."
    )
    summary: list[ReasoningItemSummary] = Field(
        default_factory=list, description="Reasoning summary."
    )
    content: list[ReasoningItemContentInput] | None = Field(  # type: ignore[assignment]
        default=None, description="Reasoning text content."
    )


# Extend ResponseInputItem with the echoed output-item types defined above: both
# are valid input items per the SDK
# (openai.types.responses.response_input_item.ResponseInputItem).
#
# IMPORTANT: do NOT use the `type` statement here. `type X = X | ...` creates a
# lazy TypeAliasType whose __value__ is evaluated after the name is rebound, so
# `X` on the right side would refer to the NEW alias (itself) — a circular
# reference that breaks Pydantic schema building.  Capturing the original binding
# in `_ResponseInputItemBase` first makes the assignment evaluate immediately to a
# plain UnionType with no self-reference.
_ResponseInputItemBase = ResponseInputItem
ResponseInputItem = (  # type: ignore[misc]
    _ResponseInputItemBase | ResponseOutputMessageInput | ResponseReasoningItemInput
)
ResponseInputParam = str | list[ResponseInputItem]  # type: ignore[misc,assignment]


# Ref: openai.types.responses.response_apply_patch_tool_call.OperationCreateFile
class ResponseApplyPatchOperationCreateFile(BaseModelResponse):
    """Instruction describing how to create a file via the apply_patch tool."""

    diff: str = Field(description="Diff to apply.")
    path: str = Field(description="Path of the file to create.")
    type: Literal["create_file"] = Field(
        description="Create a new file with the provided diff."
    )


# Ref: openai.types.responses.response_apply_patch_tool_call.OperationDeleteFile
class ResponseApplyPatchOperationDeleteFile(BaseModelResponse):
    """Instruction describing how to delete a file via the apply_patch tool."""

    path: str = Field(description="Path of the file to delete.")
    type: Literal["delete_file"] = Field(description="Delete the specified file.")


# Ref: openai.types.responses.response_apply_patch_tool_call.OperationUpdateFile
class ResponseApplyPatchOperationUpdateFile(BaseModelResponse):
    """Instruction describing how to update a file via the apply_patch tool."""

    diff: str = Field(description="Diff to apply.")
    path: str = Field(description="Path of the file to update.")
    type: Literal["update_file"] = Field(
        description="Update an existing file with the provided diff."
    )


# Ref: openai.types.responses.response_apply_patch_tool_call.Operation
ResponseApplyPatchOperation = Annotated[
    ResponseApplyPatchOperationCreateFile
    | ResponseApplyPatchOperationDeleteFile
    | ResponseApplyPatchOperationUpdateFile,
    Field(discriminator="type"),
]


# Ref: openai.types.responses.response_apply_patch_tool_call.ResponseApplyPatchToolCall
class ResponseApplyPatchToolCall(BaseModelResponse):
    """A tool call that applies file diffs by creating, deleting, or updating files."""

    id: str = Field(description="Apply patch tool call ID.")
    call_id: str = Field(description="Model-generated apply patch call ID.")
    operation: ResponseApplyPatchOperation = Field(
        description="Patch operation to apply."
    )
    status: Literal["in_progress", "completed"] = Field(
        description="Apply patch status."
    )
    type: Literal["apply_patch_call"] = Field(description="Apply patch call type.")
    caller: Caller = Field(
        default=None,
        description="Provenance of this tool call: direct or programmatic.",
    )
    created_by: str | None = Field(default=None, description="Tool call creator.")


# Ref: openai.types.responses.response_apply_patch_tool_call_output.ResponseApplyPatchToolCallOutput
class ResponseApplyPatchToolCallOutput(BaseModelResponse):
    """The output emitted by an apply patch tool call."""

    id: str = Field(description="Apply patch output ID.")
    call_id: str = Field(description="Apply patch call ID.")
    status: Literal["completed", "failed"] = Field(description="Output status.")
    type: Literal["apply_patch_call_output"] = Field(
        description="Apply patch output type."
    )
    caller: Caller = Field(
        default=None,
        description="Provenance of this tool call: direct or programmatic.",
    )
    created_by: str | None = Field(default=None, description="Output creator.")
    output: str | None = Field(
        default=None, description="Textual output from apply patch."
    )


# Ref: openai.types.responses.response_compaction_item.ResponseCompactionItem
class ResponseCompactionItem(BaseModelResponse):
    """A compaction item generated by the v1/responses/compact API."""

    id: str = Field(description="Compaction item ID.")
    encrypted_content: str = Field(description="Encrypted compaction content.")
    type: Literal["compaction"] = Field(description="Compaction type.")
    created_by: str | None = Field(default=None, description="Item creator.")


# Ref: openai.types.responses.response_tool_search_call.ResponseToolSearchCall
class ResponseToolSearchCall(BaseModelResponse):
    """A tool search call item."""

    id: str = Field(description="Tool search call ID.")
    arguments: JsonMapping = Field(description="Tool search arguments.")
    execution: Literal["server", "client"] = Field(
        description="Server or client execution."
    )
    status: ResponseItemStatus = Field(description="Call status.")
    type: Literal["tool_search_call"] = Field(description="Tool search call type.")
    call_id: str | None = Field(default=None, description="Model-generated call ID.")
    created_by: str | None = Field(default=None, description="Item creator.")


# Ref: openai.types.responses.response_tool_search_output_item.ResponseToolSearchOutputItem
class ResponseToolSearchOutputItem(BaseModelResponse):
    """A tool search output item."""

    id: str = Field(description="The unique ID of the tool search output item.")
    execution: Literal["server", "client"] = Field(
        description="Whether tool search was executed by the server or by the client."
    )
    status: ResponseItemStatus = Field(
        description="The status of the tool search output item that was recorded."
    )
    tools: list[Tool] = Field(
        description="The loaded tool definitions returned by tool search."
    )
    type: Literal["tool_search_output"] = Field(
        description="The type of the item. Always `tool_search_output`."
    )
    call_id: str | None = Field(
        default=None,
        description="The unique ID of the tool search call generated by the model.",
    )
    created_by: str | None = Field(
        default=None, description="The identifier of the actor that created the item."
    )


# Ref: openai.types.responses.response_custom_tool_call.ResponseCustomToolCall
class ResponseCustomToolCall(BaseModelResponse):
    """A call to a custom tool created by the model."""

    call_id: str = Field(
        description="An identifier used to map this custom tool call to a tool call output."
    )
    input: str = Field(
        description="The input for the custom tool call generated by the model."
    )
    name: str = Field(description="The name of the custom tool being called.")
    type: Literal["custom_tool_call"] = Field(
        description="The type of the custom tool call. Always `custom_tool_call`."
    )
    id: str | None = Field(
        default=None, description="The unique ID of the custom tool call."
    )
    caller: Caller = Field(
        default=None,
        description="Provenance of this tool call: direct or programmatic.",
    )
    namespace: str | None = Field(
        default=None, description="The namespace of the custom tool being called."
    )


# Ref: openai.types.responses.response_custom_tool_call_item.ResponseCustomToolCallItem
class ResponseCustomToolCallItem(ResponseCustomToolCall):
    """A custom tool call item returned via API."""

    id: str = Field(description="The unique ID of the custom tool call item.")
    status: ResponseItemStatus = Field(description="The status of the item.")
    created_by: str | None = Field(
        default=None, description="The identifier of the actor that created the item."
    )


# Ref: openai.types.responses.response_function_tool_call_item.ResponseFunctionToolCallItem
class ResponseFunctionToolCallItem(ResponseFunctionToolCall):
    """A function tool call item returned via API."""

    id: str = Field(description="The unique ID of the function tool call.")
    status: ResponseItemStatus = Field(description="The status of the item.")
    created_by: str | None = Field(
        default=None, description="The identifier of the actor that created the item."
    )


# Ref: openai.types.responses.response_custom_tool_call_output.ResponseCustomToolCallOutput
class ResponseCustomToolCallOutput(BaseModelResponse):
    """The output of a custom tool call from your code, being sent back to the model."""

    call_id: str = Field(
        description="The call ID, used to map this custom tool call output to a custom tool call."
    )
    output: str | list[ResponseInputContent] = Field(
        description="The output from the custom tool call. Can be a string or a list of output content."
    )
    type: Literal["custom_tool_call_output"] = Field(
        description="The type of the custom tool call output. Always `custom_tool_call_output`."
    )
    id: str | None = Field(
        default=None, description="The unique ID of the custom tool call output."
    )
    caller: Caller = Field(
        default=None,
        description="Provenance of this tool call: direct or programmatic.",
    )


# Ref: openai.types.responses.response_custom_tool_call_output_item.ResponseCustomToolCallOutputItem
class ResponseCustomToolCallOutputItem(ResponseCustomToolCallOutput):
    """A custom tool call output item returned via API."""

    id: str = Field(description="The unique ID of the custom tool call output item.")
    status: ResponseItemStatus = Field(description="The status of the item.")
    created_by: str | None = Field(
        default=None, description="The identifier of the actor that created the item."
    )


# Ref: openai.types.responses.response_function_tool_call_output_item.ResponseFunctionToolCallOutputItem
class ResponseFunctionToolCallOutputItem(BaseModelResponse):
    """A function tool call output item returned via API."""

    id: str = Field(description="The unique ID of the function call tool output.")
    call_id: str = Field(
        description="The unique ID of the function tool call generated by the model."
    )
    output: str | list[ResponseInputContent] = Field(
        description="The output from the function call. Can be a string or a list of output content."
    )
    status: ResponseItemStatus = Field(description="The status of the item.")
    type: Literal["function_call_output"] = Field(
        description="The type of the function tool call output. Always `function_call_output`."
    )
    caller: Caller = Field(
        default=None,
        description="Provenance of this tool call: direct or programmatic.",
    )
    created_by: str | None = Field(
        default=None, description="The identifier of the actor that created the item."
    )


# Shell / local shell output items  (response-side)


# Ref: openai.types.responses.response_local_environment.ResponseLocalEnvironment
class ResponseLocalEnvironment(BaseModelResponse):
    """Represents the use of a local environment to perform shell actions."""

    type: Literal["local"] = Field(description="The environment type. Always `local`.")


# Ref: openai.types.responses.response_container_reference.ResponseContainerReference
class ResponseContainerReference(BaseModelResponse):
    """Represents a container created with /v1/containers."""

    container_id: str = Field(description="The container ID.")
    type: Literal["container_reference"] = Field(
        description="The environment type. Always `container_reference`."
    )


ResponseFunctionShellEnvironment = (
    Annotated[
        ResponseLocalEnvironment | ResponseContainerReference,
        Field(discriminator="type"),
    ]
    | None
)


# Ref: openai.types.responses.response_function_shell_tool_call.Action
class ResponseFunctionShellToolCallAction(BaseModelResponse):
    """The shell commands and limits that describe how to run the tool call."""

    commands: list[str] = Field(description="The shell commands to execute.")
    max_output_length: int | None = Field(
        default=None,
        description="Optional maximum number of characters to return from each command.",
    )
    timeout_ms: int | None = Field(
        default=None, description="Optional timeout in milliseconds for the commands."
    )


# Ref: openai.types.responses.response_function_shell_tool_call.ResponseFunctionShellToolCall
class ResponseFunctionShellToolCall(BaseModelResponse):
    """A tool call that executes one or more shell commands in a managed environment."""

    id: str = Field(description="The unique ID of the shell tool call.")
    action: ResponseFunctionShellToolCallAction = Field(
        description="The shell commands and limits that describe how to run the tool call."
    )
    call_id: str = Field(
        description="The unique ID of the shell tool call generated by the model."
    )
    status: Literal["in_progress", "completed", "incomplete"] = Field(
        description="The status of the shell call."
    )
    type: Literal["shell_call"] = Field(
        description="The type of the item. Always `shell_call`."
    )
    environment: ResponseFunctionShellEnvironment = Field(
        default=None,
        description="Represents the use of a local environment to perform shell actions.",
    )
    caller: Caller = Field(
        default=None,
        description="Provenance of this tool call: direct or programmatic.",
    )
    created_by: str | None = Field(
        default=None, description="The ID of the entity that created this tool call."
    )


# Ref: openai.types.responses.response_function_shell_tool_call_output.OutputOutcomeTimeout
class ShellToolCallOutputOutcomeTimeout(BaseModelResponse):
    """Indicates that the shell call exceeded its configured time limit."""

    type: Literal["timeout"] = Field(description="Timeout outcome.")


# Ref: openai.types.responses.response_function_shell_tool_call_output.OutputOutcomeExit
class ShellToolCallOutputOutcomeExit(BaseModelResponse):
    """Indicates that the shell commands finished and returned an exit code."""

    exit_code: int = Field(description="Exit code from the shell process.")
    type: Literal["exit"] = Field(description="The outcome type. Always `exit`.")


# Ref: openai.types.responses.response_function_shell_tool_call_output.OutputOutcome
ShellToolCallOutputOutcome = Annotated[
    ShellToolCallOutputOutcomeTimeout | ShellToolCallOutputOutcomeExit,
    Field(discriminator="type"),
]


# Ref: openai.types.responses.response_function_shell_tool_call_output.Output
class ShellToolCallOutputContent(BaseModelResponse):
    """The content of a shell tool call output that was emitted."""

    outcome: ShellToolCallOutputOutcome = Field(
        description="Represents either an exit outcome or a timeout outcome for a shell call output chunk."
    )
    stderr: str = Field(description="The standard error output that was captured.")
    stdout: str = Field(description="The standard output that was captured.")
    created_by: str | None = Field(
        default=None, description="The identifier of the actor that created the item."
    )


# Ref: openai.types.responses.response_function_shell_tool_call_output.ResponseFunctionShellToolCallOutput
class ResponseFunctionShellToolCallOutput(BaseModelResponse):
    """The output of a shell tool call that was emitted."""

    id: str = Field(description="The unique ID of the shell call output.")
    call_id: str = Field(
        description="The unique ID of the shell tool call generated by the model."
    )
    output: list[ShellToolCallOutputContent] = Field(
        description="An array of shell call output contents."
    )
    status: Literal["in_progress", "completed", "incomplete"] = Field(
        description="The status of the shell call output."
    )
    type: Literal["shell_call_output"] = Field(
        description="The type of the shell call output. Always `shell_call_output`."
    )
    max_output_length: int | None = Field(
        default=None, description="The maximum length of the shell command output."
    )
    caller: Caller = Field(
        default=None,
        description="Provenance of this tool call: direct or programmatic.",
    )
    created_by: str | None = Field(
        default=None, description="The identifier of the actor that created the item."
    )


# Image generation, local shell, MCP output items


# Ref: openai.types.responses.response_output_item.ImageGenerationCall
class ImageGenerationCall(BaseModelResponse):
    """An image generation request made by the model."""

    id: str = Field(description="The unique ID of the image generation call.")
    status: Literal["in_progress", "completed", "generating", "failed"] = Field(
        description="The status of the image generation call."
    )
    type: Literal["image_generation_call"] = Field(
        description="The type of the image generation call. Always `image_generation_call`."
    )
    result: str | None = Field(
        default=None, description="The generated image encoded in base64."
    )


# Ref: openai.types.responses.response_output_item.LocalShellCallAction
class LocalShellCallAction(BaseModelResponse):
    """Execute a shell command on the server."""

    command: list[str] = Field(description="Command to run.")
    env: dict[str, str] = Field(description="Environment variables.")
    type: Literal["exec"] = Field(description="Exec action type.")
    timeout_ms: int | None = Field(default=None, description="Timeout in milliseconds.")
    user: str | None = Field(default=None, description="User to run as.")
    working_directory: str | None = Field(
        default=None, description="Working directory."
    )


# Ref: openai.types.responses.response_output_item.LocalShellCall
class LocalShellCall(BaseModelResponse):
    """A tool call to run a command on the local shell."""

    id: str = Field(description="Local shell call ID.")
    action: LocalShellCallAction = Field(description="Shell command to execute.")
    call_id: str = Field(description="Model-generated call ID.")
    status: Literal["in_progress", "completed", "incomplete"] = Field(
        description="Call status."
    )
    type: Literal["local_shell_call"] = Field(description="Local shell call type.")


# Ref: openai.types.responses.response_output_item.LocalShellCallOutput
class LocalShellCallOutput(BaseModelResponse):
    """The output of a local shell tool call."""

    id: str = Field(description="Shell tool call ID.")
    output: str = Field(description="JSON output string.")
    type: Literal["local_shell_call_output"] = Field(
        description="Shell call output type."
    )
    status: ResponseItemStatus | None = Field(default=None, description="Item status.")


# Ref: openai.types.responses.response_output_item.McpListToolsTool
class McpListToolsToolOutput(BaseModelResponse):
    """A tool available on an MCP server."""

    input_schema: object = Field(description="Tool input JSON schema.")
    name: str = Field(description="Tool name.")
    annotations: object | None = Field(default=None, description="Tool annotations.")
    description: str | None = Field(default=None, description="Tool description.")


# Ref: openai.types.responses.response_output_item.McpListTools
class McpListTools(BaseModelResponse):
    """A list of tools available on an MCP server."""

    id: str = Field(description="List ID.")
    server_label: str = Field(description="MCP server label.")
    tools: list[McpListToolsToolOutput] = Field(description="Server tools.")
    type: Literal["mcp_list_tools"] = Field(description="MCP list tools type.")
    error: str | None = Field(
        default=None, description="Error if tools could not be listed."
    )


# Ref: openai.types.responses.response_output_item.McpApprovalRequest
class McpApprovalRequest(BaseModelResponse):
    """A request for human approval of a tool invocation."""

    id: str = Field(description="Approval request ID.")
    arguments: str = Field(description="Tool arguments JSON.")
    name: str = Field(description="Tool name.")
    server_label: str = Field(description="MCP server label.")
    type: Literal["mcp_approval_request"] = Field(
        description="MCP approval request type."
    )


# Ref: openai.types.responses.response_output_item.McpApprovalResponse
class McpApprovalResponseOutput(BaseModelResponse):
    """A response to an MCP approval request."""

    id: str = Field(description="Approval response ID.")
    approval_request_id: str = Field(description="Approval request ID.")
    approve: bool = Field(description="Whether approved.")
    type: Literal["mcp_approval_response"] = Field(
        description="MCP approval response type."
    )
    reason: str | None = Field(default=None, description="Decision reason.")


# Ref: openai.types.responses.response_output_item.McpCall
class McpCall(BaseModelResponse):
    """An invocation of a tool on an MCP server."""

    id: str = Field(description="Tool call ID.")
    arguments: str = Field(description="Tool arguments JSON.")
    name: str = Field(description="Tool name.")
    server_label: str = Field(description="MCP server label.")
    type: Literal["mcp_call"] = Field(description="MCP call type.")
    approval_request_id: str | None = Field(
        default=None, description="Approval request ID."
    )
    error: str | None = Field(default=None, description="Tool call error.")
    output: str | None = Field(default=None, description="Tool call output.")
    status: (
        Literal["in_progress", "completed", "incomplete", "calling", "failed"] | None
    ) = Field(default=None, description="Tool call status.")


# Ref: openai.types.responses.response_output_item.AdditionalTools
class AdditionalTools(BaseModelResponse):
    """An output item advertising additional tools (never emitted by this backend)."""

    id: str = Field(description="Item ID.")
    role: Literal[
        "unknown",
        "user",
        "assistant",
        "system",
        "critic",
        "discriminator",
        "developer",
        "tool",
    ] = Field(description="Role that provided the additional tools.")
    tools: list[Tool] = Field(description="The additional tools.")
    type: Literal["additional_tools"] = Field(
        description="Item type. Always `additional_tools`."
    )


# Ref: openai.types.responses.response_output_item.Program
class Program(BaseModelResponse):
    """A program emitted by programmatic tool calling."""

    id: str = Field(description="Program ID.")
    call_id: str = Field(description="Identifier mapping this program to its output.")
    code: str = Field(description="Code executed by the program.")
    fingerprint: str = Field(description="Program fingerprint.")
    type: Literal["program"] = Field(description="Item type. Always `program`.")


# Ref: openai.types.responses.response_output_item.ProgramOutput
class ProgramOutput(BaseModelResponse):
    """The result of a program execution."""

    id: str = Field(description="Program output ID.")
    call_id: str = Field(description="Identifier mapping this output to its program.")
    result: str = Field(description="Result of the program execution.")
    status: Literal["completed", "incomplete"] = Field(
        description="Status of the program execution."
    )
    type: Literal["program_output"] = Field(
        description="Item type. Always `program_output`."
    )


# Ref: openai.types.responses.response_output_item.ResponseOutputItem
ResponseOutputItem = Annotated[
    ResponseOutputMessage
    | ResponseFileSearchToolCall
    | ResponseFunctionToolCall
    | ResponseFunctionToolCallOutputItem
    | ResponseFunctionWebSearch
    | ResponseComputerToolCall
    | ResponseComputerToolCallOutputItem
    | ResponseReasoningItem
    | ResponseToolSearchCall
    | ResponseToolSearchOutputItem
    | ResponseCompactionItem
    | ImageGenerationCall
    | ResponseCodeInterpreterToolCall
    | LocalShellCall
    | LocalShellCallOutput
    | ResponseFunctionShellToolCall
    | ResponseFunctionShellToolCallOutput
    | ResponseApplyPatchToolCall
    | ResponseApplyPatchToolCallOutput
    | McpCall
    | McpListTools
    | McpApprovalRequest
    | McpApprovalResponseOutput
    | ResponseCustomToolCall
    | ResponseCustomToolCallOutputItem
    | AdditionalTools
    | Program
    | ProgramOutput,
    Field(discriminator="type"),
]


# Usage types


# Ref: openai.types.responses.response_usage.InputTokensDetails
class InputTokensDetails(BaseModelResponse):
    """A detailed breakdown of the input tokens."""

    cached_tokens: int = Field(description="Cached token count.")
    cache_write_tokens: int = Field(default=0, description="Cache write token count.")


# Ref: openai.types.responses.response_usage.OutputTokensDetails
class OutputTokensDetails(BaseModelResponse):
    """A detailed breakdown of the output tokens."""

    reasoning_tokens: int = Field(default=0, description="Reasoning token count.")


# Ref: openai.types.responses.response_usage.ResponseUsage
class ResponseUsage(BaseModelResponse):
    """Token usage details for a response, including input, output, and total counts."""

    input_tokens: int = Field(description="Input token count.")
    input_tokens_details: InputTokensDetails = Field(description="Input token details.")
    output_tokens: int = Field(description="Output token count.")
    output_tokens_details: OutputTokensDetails = Field(
        description="Output token details."
    )
    total_tokens: int = Field(description="Total token count.")


# Text format config types


# Ref: openai.types.responses.response_format_text_json_schema_config.ResponseFormatTextJSONSchemaConfig
class ResponseFormatTextJSONSchemaConfig(BaseModelRequest):
    """JSON Schema response format for generating structured JSON responses."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(
        description="Response format name (a-z, A-Z, 0-9, underscores, dashes; max 64 chars)."
    )
    schema_: JsonMapping = Field(
        alias="schema",
        serialization_alias="schema",
        description="JSON Schema for response format.",
    )
    type: Literal["json_schema"] = Field(description="JSON schema response format.")
    description: str | None = Field(
        default=None, description="Description of the response format for the model."
    )
    strict: bool | None = Field(
        default=None, description="Enable strict schema adherence."
    )


# Ref: openai.types.responses.response_format_text_config.ResponseFormatTextConfig
ResponseFormatTextConfig = Annotated[
    ResponseFormatText | ResponseFormatTextJSONSchemaConfig | ResponseFormatJSONObject,
    Field(discriminator="type"),
]


# Ref: openai.types.responses.response_text_config.ResponseTextConfig
class ResponseTextConfig(BaseModelRequest):
    """Configuration options for a text response from the model."""

    format: ResponseFormatTextConfig | None = Field(
        default=None,
        description=(
            "An object specifying the format that the model must output. "
            'Configuring `{ "type": "json_schema" }` enables Structured Outputs.'
        ),
    )
    verbosity: VerbosityLevel | None = Field(
        default=None,
        description="Constrains the verbosity of the model's response. Values: `low`, `medium`, or `high`.",
    )


# Prompt types

#: Union of prompt variable value types.
PromptVariables = str | ResponseInputText | ResponseInputImage | ResponseInputFile


# Ref: openai.types.responses.response_prompt.ResponsePrompt
class ResponsePrompt(BaseModelResponse):
    """Reference to a prompt template and its variables."""

    id: str = Field(description="The unique identifier of the prompt template to use.")
    variables: dict[str, PromptVariables] | None = Field(
        default=None,
        description="Optional map of values to substitute in for variables in the prompt template.",
    )
    version: str | None = Field(
        default=None, description="Optional version of the prompt template."
    )


# Error / incomplete details


#: Valid error code values for a response error.
ResponseErrorCode = Literal[
    "server_error",
    "rate_limit_exceeded",
    "invalid_prompt",
    "data_residency_mismatch",
    "bio_policy",
    "vector_store_timeout",
    "invalid_image",
    "invalid_image_format",
    "invalid_base64_image",
    "invalid_image_url",
    "image_too_large",
    "image_too_small",
    "image_parse_error",
    "image_content_policy_violation",
    "invalid_image_mode",
    "image_file_too_large",
    "unsupported_image_media_type",
    "empty_image_file",
    "failed_to_download_image",
    "image_file_not_found",
]


# Ref: openai.types.responses.response_error.ResponseError
class ResponseError(BaseModelResponse):
    """An error object returned when the model fails to generate a response."""

    code: ResponseErrorCode = Field(description="Error code.")
    message: str = Field(description="Error message.")


# Ref: openai.types.responses.response.IncompleteDetails
class IncompleteDetails(BaseModelResponse):
    """Details about why the response is incomplete."""

    reason: Literal["max_output_tokens", "content_filter"] | None = Field(
        default=None, description="Incomplete reason."
    )


# Conversation


# Ref: openai.types.responses.response.Conversation
class Conversation(BaseModelResponse):
    """The conversation that this response belonged to."""

    id: str = Field(description="Conversation ID.")


# Main Response object


# Ref: openai.types.responses.response.Response
class Response(BaseModelResponse):
    """A model response from the Responses API."""

    id: str = Field(description="Response ID.")
    created_at: int | float = Field(description="Unix timestamp of creation.")
    error: ResponseError | None = Field(
        default=None, description="Response error if failed."
    )
    incomplete_details: IncompleteDetails | None = Field(
        default=None, description="Incomplete details."
    )
    moderation: ResponseModeration | None = Field(
        default=None,
        description="Guardrail moderation results, when the request set "
        "`moderation` (in streaming mode, carried by the terminal event).",
    )
    instructions: str | list[ResponseInputItem] | None = Field(
        default=None, description="System or developer message."
    )
    metadata: Metadata | None = Field(
        default=None, description="Key-value pairs for the response."
    )
    model: str = Field(description="Model ID.")
    object: Literal["response"] = Field(description="Object type.")
    output: list[ResponseOutputItem] = Field(description="Generated content items.")
    parallel_tool_calls: bool = Field(description="Allow parallel tool calls.")
    temperature: float | None = Field(
        default=None, description="Sampling temperature (0-2)."
    )
    tool_choice: ToolChoice = Field(description="Tool selection method.")
    tools: list[Tool] = Field(description="Available tools.")
    top_p: float | None = Field(default=None, description="Nucleus sampling parameter.")
    background: bool | None = Field(default=None, description="Run in background.")
    completed_at: int | float | None = Field(
        default=None, description="Completion timestamp."
    )
    conversation: Conversation | None = Field(default=None, description="Conversation.")
    max_output_tokens: int | None = Field(
        default=None, description="Max output tokens."
    )
    max_tool_calls: int | None = Field(
        default=None, description="Max tool calls allowed."
    )
    previous_response_id: str | None = Field(
        default=None, description="Previous response ID for multi-turn."
    )
    prompt: ResponsePrompt | None = Field(
        default=None, description="Prompt template reference."
    )
    prompt_cache_key: str | None = Field(
        default=None, description="Cache key for similar requests."
    )
    prompt_cache_options: PromptCacheOptions | None = Field(
        default=None,
        description="Explicit prompt-caching configuration, echoed from the request.",
    )
    prompt_cache_retention: PromptCacheRetention | None = Field(
        default=None, description="Cache retention policy."
    )
    reasoning: Reasoning | None = Field(
        default=None, description="Reasoning configuration."
    )
    safety_identifier: str | None = Field(
        default=None, description="User policy violation identifier."
    )
    service_tier: ServiceTiers | None = Field(
        default=None, description="Service tier for request."
    )
    status: ResponseStatus | None = Field(default=None, description="Response status.")
    text: ResponseTextConfig | None = Field(
        default=None, description="Text response config."
    )
    top_logprobs: int | None = Field(
        default=None, description="Top logprobs count (0-20)."
    )
    truncation: Literal["auto", "disabled"] | None = Field(
        default=None, description="Truncation strategy."
    )
    usage: ResponseUsage | None = Field(
        default=None, description="Token usage details."
    )
    user: str | None = Field(
        default=None, description="User identifier (use safety_identifier instead)."
    )


# Stream event helper types


# Ref: openai.types.responses.response_content_part_added_event.PartReasoningText
class ContentPartReasoningText(BaseModelResponse):
    """Reasoning text part within a stream content-part event."""

    text: str = Field(description="Reasoning text.")
    type: Literal["reasoning_text"] = Field(description="Reasoning text type.")


# Ref: openai.types.responses.response_content_part_added_event.Part
ContentPart = Annotated[
    ResponseOutputText | ResponseOutputRefusal | ContentPartReasoningText,
    Field(discriminator="type"),
]


# Ref: openai.types.responses.response_reasoning_summary_part_added_event.Part
class ReasoningSummaryPart(BaseModelResponse):
    """A reasoning summary text part."""

    text: str = Field(description="The text of the summary part.")
    type: Literal["summary_text"] = Field(
        description="The type of the summary part. Always `summary_text`."
    )


class _StreamEventItemBase(BaseModelResponse):
    """Base for stream events that are associated with a specific output item."""

    item_id: str = Field(description="The unique identifier of the output item.")
    output_index: int = Field(
        description="The index of the output item in the response."
    )
    sequence_number: int = Field(description="The sequence number of this event.")


# Stream events — response lifecycle


# Ref: openai.types.responses.response_queued_event.ResponseQueuedEvent
class ResponseQueuedEvent(BaseModelResponse):
    """Emitted when a response is queued and waiting to be processed."""

    response: Response = Field(description="The full response object that is queued.")
    sequence_number: int = Field(description="The sequence number for this event.")
    type: Literal["response.queued"] = Field(
        description="The type of the event. Always `response.queued`."
    )


# Ref: openai.types.responses.response_created_event.ResponseCreatedEvent
class ResponseCreatedEvent(BaseModelResponse):
    """An event that is emitted when a response is created."""

    response: Response = Field(description="The response that was created.")
    sequence_number: int = Field(description="The sequence number for this event.")
    type: Literal["response.created"] = Field(
        description="The type of the event. Always `response.created`."
    )


# Ref: openai.types.responses.response_in_progress_event.ResponseInProgressEvent
class ResponseInProgressEvent(BaseModelResponse):
    """Emitted when the response is in progress."""

    response: Response = Field(description="The response that is in progress.")
    sequence_number: int = Field(description="The sequence number of this event.")
    type: Literal["response.in_progress"] = Field(
        description="The type of the event. Always `response.in_progress`."
    )


# Ref: openai.types.responses.response_completed_event.ResponseCompletedEvent
class ResponseCompletedEvent(BaseModelResponse):
    """Emitted when the model response is complete."""

    response: Response = Field(description="Properties of the completed response.")
    sequence_number: int = Field(description="The sequence number for this event.")
    type: Literal["response.completed"] = Field(
        description="The type of the event. Always `response.completed`."
    )


# Ref: openai.types.responses.response_failed_event.ResponseFailedEvent
class ResponseFailedEvent(BaseModelResponse):
    """An event that is emitted when a response fails."""

    response: Response = Field(description="The response that failed.")
    sequence_number: int = Field(description="The sequence number of this event.")
    type: Literal["response.failed"] = Field(
        description="The type of the event. Always `response.failed`."
    )


# Ref: openai.types.responses.response_incomplete_event.ResponseIncompleteEvent
class ResponseIncompleteEvent(BaseModelResponse):
    """An event that is emitted when a response finishes as incomplete."""

    response: Response = Field(description="The response that was incomplete.")
    sequence_number: int = Field(description="The sequence number of this event.")
    type: Literal["response.incomplete"] = Field(
        description="The type of the event. Always `response.incomplete`."
    )


# Ref: openai.types.responses.response_error_event.ResponseErrorEvent
class ResponseErrorEvent(BaseModelResponse):
    """Emitted when an error occurs."""

    message: str = Field(description="The error message.")
    sequence_number: int = Field(description="The sequence number of this event.")
    type: Literal["error"] = Field(description="The type of the event. Always `error`.")
    code: str | None = Field(default=None, description="The error code.")
    param: str | None = Field(default=None, description="The error parameter.")


# Stream events — output items


# Ref: openai.types.responses.response_output_item_added_event.ResponseOutputItemAddedEvent
class ResponseOutputItemAddedEvent(BaseModelResponse):
    """Emitted when a new output item is added."""

    item: ResponseOutputItem = Field(description="The output item that was added.")
    output_index: int = Field(
        description="The index of the output item that was added."
    )
    sequence_number: int = Field(description="The sequence number of this event.")
    type: Literal["response.output_item.added"] = Field(
        description="The type of the event. Always `response.output_item.added`."
    )


# Ref: openai.types.responses.response_output_item_done_event.ResponseOutputItemDoneEvent
class ResponseOutputItemDoneEvent(BaseModelResponse):
    """Emitted when an output item is marked done."""

    item: ResponseOutputItem = Field(
        description="The output item that was marked done."
    )
    output_index: int = Field(
        description="The index of the output item that was marked done."
    )
    sequence_number: int = Field(description="The sequence number of this event.")
    type: Literal["response.output_item.done"] = Field(
        description="The type of the event. Always `response.output_item.done`."
    )


# Stream events — content parts


# Ref: openai.types.responses.response_content_part_added_event.ResponseContentPartAddedEvent
class ResponseContentPartAddedEvent(_StreamEventItemBase):
    """Emitted when a new content part is added."""

    content_index: int = Field(
        description="The index of the content part that was added."
    )
    part: ContentPart = Field(description="The content part that was added.")
    type: Literal["response.content_part.added"] = Field(
        description="The type of the event. Always `response.content_part.added`."
    )


# Ref: openai.types.responses.response_content_part_done_event.ResponseContentPartDoneEvent
class ResponseContentPartDoneEvent(_StreamEventItemBase):
    """Emitted when a content part is done."""

    content_index: int = Field(
        description="The index of the content part that is done."
    )
    part: ContentPart = Field(description="The content part that is done.")
    type: Literal["response.content_part.done"] = Field(
        description="The type of the event. Always `response.content_part.done`."
    )


# Stream events — text deltas


# Ref: openai.types.responses.response_text_delta_event.ResponseTextDeltaEvent
class ResponseTextDeltaEvent(_StreamEventItemBase):
    """Emitted when there is an additional text delta."""

    content_index: int = Field(
        description="The index of the content part that the text delta was added to."
    )
    delta: str = Field(description="The text delta that was added.")
    logprobs: list[Logprob] = Field(
        description="The log probabilities of the tokens in the delta."
    )
    type: Literal["response.output_text.delta"] = Field(
        description="The type of the event. Always `response.output_text.delta`."
    )


# Ref: openai.types.responses.response_text_done_event.ResponseTextDoneEvent
class ResponseTextDoneEvent(_StreamEventItemBase):
    """Emitted when text content is finalized."""

    content_index: int = Field(
        description="The index of the content part that the text content is finalized."
    )
    logprobs: list[Logprob] = Field(
        description="The log probabilities of the tokens in the delta."
    )
    text: str = Field(description="The text content that is finalized.")
    type: Literal["response.output_text.done"] = Field(
        description="The type of the event. Always `response.output_text.done`."
    )


# Ref: openai.types.responses.response_output_text_annotation_added_event.ResponseOutputTextAnnotationAddedEvent
class ResponseOutputTextAnnotationAddedEvent(_StreamEventItemBase):
    """Emitted when an annotation is added to output text content."""

    annotation: Annotation = Field(description="The annotation object being added.")
    annotation_index: int = Field(
        description="The index of the annotation within the content part."
    )
    content_index: int = Field(
        description="The index of the content part within the output item."
    )
    type: Literal["response.output_text.annotation.added"] = Field(
        description="The type of the event. Always `response.output_text.annotation.added`."
    )


# Stream events — refusal


# Ref: openai.types.responses.response_refusal_delta_event.ResponseRefusalDeltaEvent
class ResponseRefusalDeltaEvent(_StreamEventItemBase):
    """Emitted when there is a partial refusal text."""

    content_index: int = Field(
        description="The index of the content part that the refusal text is added to."
    )
    delta: str = Field(description="The refusal text that is added.")
    type: Literal["response.refusal.delta"] = Field(
        description="The type of the event. Always `response.refusal.delta`."
    )


# Ref: openai.types.responses.response_refusal_done_event.ResponseRefusalDoneEvent
class ResponseRefusalDoneEvent(_StreamEventItemBase):
    """Emitted when refusal text is finalized."""

    content_index: int = Field(
        description="The index of the content part that the refusal text is finalized."
    )
    refusal: str = Field(description="The refusal text that is finalized.")
    type: Literal["response.refusal.done"] = Field(
        description="The type of the event. Always `response.refusal.done`."
    )


# Stream events — function call arguments


# Ref: openai.types.responses.response_function_call_arguments_delta_event.ResponseFunctionCallArgumentsDeltaEvent
class ResponseFunctionCallArgumentsDeltaEvent(_StreamEventItemBase):
    """Emitted when there is a partial function-call arguments delta."""

    delta: str = Field(description="The function-call arguments delta that is added.")
    type: Literal["response.function_call_arguments.delta"] = Field(
        description="The type of the event. Always `response.function_call_arguments.delta`."
    )


# Ref: openai.types.responses.response_function_call_arguments_done_event.ResponseFunctionCallArgumentsDoneEvent
class ResponseFunctionCallArgumentsDoneEvent(_StreamEventItemBase):
    """Emitted when function-call arguments are finalized."""

    arguments: str = Field(description="The function-call arguments.")
    name: str = Field(description="The name of the function that was called.")
    type: Literal["response.function_call_arguments.done"] = Field(
        description="The type of the event. Always `response.function_call_arguments.done`."
    )


# Stream events — audio


# Ref: openai.types.responses.response_audio_delta_event.ResponseAudioDeltaEvent
class ResponseAudioDeltaEvent(BaseModelResponse):
    """Emitted when there is a partial audio response."""

    delta: str = Field(description="A chunk of Base64 encoded response audio bytes.")
    sequence_number: int = Field(
        description="A sequence number for this chunk of the stream response."
    )
    type: Literal["response.audio.delta"] = Field(
        description="The type of the event. Always `response.audio.delta`."
    )


# Ref: openai.types.responses.response_audio_done_event.ResponseAudioDoneEvent
class ResponseAudioDoneEvent(BaseModelResponse):
    """Emitted when the audio response is complete."""

    sequence_number: int = Field(description="The sequence number of the delta.")
    type: Literal["response.audio.done"] = Field(
        description="The type of the event. Always `response.audio.done`."
    )


# Ref: openai.types.responses.response_audio_transcript_delta_event.ResponseAudioTranscriptDeltaEvent
class ResponseAudioTranscriptDeltaEvent(BaseModelResponse):
    """Emitted when there is a partial transcript of audio."""

    delta: str = Field(description="The partial transcript of the audio response.")
    sequence_number: int = Field(description="The sequence number of this event.")
    type: Literal["response.audio.transcript.delta"] = Field(
        description="The type of the event. Always `response.audio.transcript.delta`."
    )


# Ref: openai.types.responses.response_audio_transcript_done_event.ResponseAudioTranscriptDoneEvent
class ResponseAudioTranscriptDoneEvent(BaseModelResponse):
    """Emitted when the full audio transcript is completed."""

    sequence_number: int = Field(description="The sequence number of this event.")
    type: Literal["response.audio.transcript.done"] = Field(
        description="The type of the event. Always `response.audio.transcript.done`."
    )


# Stream events — web search


# Ref: openai.types.responses.response_web_search_call_in_progress_event.ResponseWebSearchCallInProgressEvent
class ResponseWebSearchCallInProgressEvent(_StreamEventItemBase):
    """Emitted when a web search call is initiated."""

    type: Literal["response.web_search_call.in_progress"] = Field(
        description="The type of the event. Always `response.web_search_call.in_progress`."
    )


# Ref: openai.types.responses.response_web_search_call_searching_event.ResponseWebSearchCallSearchingEvent
class ResponseWebSearchCallSearchingEvent(_StreamEventItemBase):
    """Emitted when a web search call is executing."""

    type: Literal["response.web_search_call.searching"] = Field(
        description="The type of the event. Always `response.web_search_call.searching`."
    )


# Ref: openai.types.responses.response_web_search_call_completed_event.ResponseWebSearchCallCompletedEvent
class ResponseWebSearchCallCompletedEvent(_StreamEventItemBase):
    """Emitted when a web search call is completed."""

    type: Literal["response.web_search_call.completed"] = Field(
        description="The type of the event. Always `response.web_search_call.completed`."
    )


# Stream events — file search


# Ref: openai.types.responses.response_file_search_call_in_progress_event.ResponseFileSearchCallInProgressEvent
class ResponseFileSearchCallInProgressEvent(_StreamEventItemBase):
    """Emitted when a file search call is initiated."""

    type: Literal["response.file_search_call.in_progress"] = Field(
        description="The type of the event. Always `response.file_search_call.in_progress`."
    )


# Ref: openai.types.responses.response_file_search_call_searching_event.ResponseFileSearchCallSearchingEvent
class ResponseFileSearchCallSearchingEvent(_StreamEventItemBase):
    """Emitted when a file search is currently searching."""

    type: Literal["response.file_search_call.searching"] = Field(
        description="The type of the event. Always `response.file_search_call.searching`."
    )


# Ref: openai.types.responses.response_file_search_call_completed_event.ResponseFileSearchCallCompletedEvent
class ResponseFileSearchCallCompletedEvent(_StreamEventItemBase):
    """Emitted when a file search call is completed."""

    type: Literal["response.file_search_call.completed"] = Field(
        description="The type of the event. Always `response.file_search_call.completed`."
    )


# Stream events — code interpreter


# Ref: openai.types.responses.response_code_interpreter_call_in_progress_event.ResponseCodeInterpreterCallInProgressEvent
class ResponseCodeInterpreterCallInProgressEvent(_StreamEventItemBase):
    """Emitted when a code interpreter call is in progress."""

    type: Literal["response.code_interpreter_call.in_progress"] = Field(
        description="The type of the event. Always `response.code_interpreter_call.in_progress`."
    )


# Ref: openai.types.responses.response_code_interpreter_call_interpreting_event.ResponseCodeInterpreterCallInterpretingEvent
class ResponseCodeInterpreterCallInterpretingEvent(_StreamEventItemBase):
    """Emitted when the code interpreter is actively interpreting the code snippet."""

    type: Literal["response.code_interpreter_call.interpreting"] = Field(
        description="The type of the event. Always `response.code_interpreter_call.interpreting`."
    )


# Ref: openai.types.responses.response_code_interpreter_call_completed_event.ResponseCodeInterpreterCallCompletedEvent
class ResponseCodeInterpreterCallCompletedEvent(_StreamEventItemBase):
    """Emitted when the code interpreter call is completed."""

    type: Literal["response.code_interpreter_call.completed"] = Field(
        description="The type of the event. Always `response.code_interpreter_call.completed`."
    )


# Ref: openai.types.responses.response_code_interpreter_call_code_delta_event.ResponseCodeInterpreterCallCodeDeltaEvent
class ResponseCodeInterpreterCallCodeDeltaEvent(_StreamEventItemBase):
    """Emitted when a partial code snippet is streamed by the code interpreter."""

    delta: str = Field(
        description="The partial code snippet being streamed by the code interpreter."
    )
    type: Literal["response.code_interpreter_call_code.delta"] = Field(
        description="The type of the event. Always `response.code_interpreter_call_code.delta`."
    )


# Ref: openai.types.responses.response_code_interpreter_call_code_done_event.ResponseCodeInterpreterCallCodeDoneEvent
class ResponseCodeInterpreterCallCodeDoneEvent(_StreamEventItemBase):
    """Emitted when the code snippet is finalized by the code interpreter."""

    code: str = Field(
        description="The final code snippet output by the code interpreter."
    )
    type: Literal["response.code_interpreter_call_code.done"] = Field(
        description="The type of the event. Always `response.code_interpreter_call_code.done`."
    )


# Stream events — reasoning


# Ref: openai.types.responses.response_reasoning_text_delta_event.ResponseReasoningTextDeltaEvent
class ResponseReasoningTextDeltaEvent(_StreamEventItemBase):
    """Emitted when a delta is added to a reasoning text."""

    content_index: int = Field(
        description="The index of the reasoning content part this delta is associated with."
    )
    delta: str = Field(
        description="The text delta that was added to the reasoning content."
    )
    type: Literal["response.reasoning_text.delta"] = Field(
        description="The type of the event. Always `response.reasoning_text.delta`."
    )


# Ref: openai.types.responses.response_reasoning_text_done_event.ResponseReasoningTextDoneEvent
class ResponseReasoningTextDoneEvent(_StreamEventItemBase):
    """Emitted when a reasoning text is completed."""

    content_index: int = Field(description="The index of the reasoning content part.")
    text: str = Field(description="The full text of the completed reasoning content.")
    type: Literal["response.reasoning_text.done"] = Field(
        description="The type of the event. Always `response.reasoning_text.done`."
    )


# Ref: openai.types.responses.response_reasoning_summary_part_added_event.ResponseReasoningSummaryPartAddedEvent
class ResponseReasoningSummaryPartAddedEvent(_StreamEventItemBase):
    """Emitted when a new reasoning summary part is added."""

    part: ReasoningSummaryPart = Field(description="The summary part that was added.")
    summary_index: int = Field(
        description="The index of the summary part within the reasoning summary."
    )
    type: Literal["response.reasoning_summary_part.added"] = Field(
        description="The type of the event. Always `response.reasoning_summary_part.added`."
    )


# Ref: openai.types.responses.response_reasoning_summary_part_done_event.ResponseReasoningSummaryPartDoneEvent
class ResponseReasoningSummaryPartDoneEvent(_StreamEventItemBase):
    """Emitted when a reasoning summary part is completed."""

    part: ReasoningSummaryPart = Field(description="The completed summary part.")
    summary_index: int = Field(
        description="The index of the summary part within the reasoning summary."
    )
    type: Literal["response.reasoning_summary_part.done"] = Field(
        description="The type of the event. Always `response.reasoning_summary_part.done`."
    )


# Ref: openai.types.responses.response_reasoning_summary_text_delta_event.ResponseReasoningSummaryTextDeltaEvent
class ResponseReasoningSummaryTextDeltaEvent(_StreamEventItemBase):
    """Emitted when a delta is added to a reasoning summary text."""

    delta: str = Field(description="The text delta that was added to the summary.")
    summary_index: int = Field(
        description="The index of the summary part within the reasoning summary."
    )
    type: Literal["response.reasoning_summary_text.delta"] = Field(
        description="The type of the event. Always `response.reasoning_summary_text.delta`."
    )


# Ref: openai.types.responses.response_reasoning_summary_text_done_event.ResponseReasoningSummaryTextDoneEvent
class ResponseReasoningSummaryTextDoneEvent(_StreamEventItemBase):
    """Emitted when a reasoning summary text is completed."""

    summary_index: int = Field(
        description="The index of the summary part within the reasoning summary."
    )
    text: str = Field(description="The full text of the completed reasoning summary.")
    type: Literal["response.reasoning_summary_text.done"] = Field(
        description="The type of the event. Always `response.reasoning_summary_text.done`."
    )


# Stream events — image generation


# Ref: openai.types.responses.response_image_gen_call_in_progress_event.ResponseImageGenCallInProgressEvent
class ResponseImageGenCallInProgressEvent(_StreamEventItemBase):
    """Emitted when an image generation tool call is in progress."""

    type: Literal["response.image_generation_call.in_progress"] = Field(
        description="The type of the event. Always `response.image_generation_call.in_progress`."
    )


# Ref: openai.types.responses.response_image_gen_call_generating_event.ResponseImageGenCallGeneratingEvent
class ResponseImageGenCallGeneratingEvent(_StreamEventItemBase):
    """Emitted when an image generation tool call is actively generating an image."""

    type: Literal["response.image_generation_call.generating"] = Field(
        description="The type of the event. Always `response.image_generation_call.generating`."
    )


# Ref: openai.types.responses.response_image_gen_call_partial_image_event.ResponseImageGenCallPartialImageEvent
class ResponseImageGenCallPartialImageEvent(_StreamEventItemBase):
    """Emitted when a partial image is available during image generation streaming."""

    partial_image_b64: str = Field(
        description="Base64-encoded partial image data, suitable for rendering as an image."
    )
    partial_image_index: int = Field(description="0-based index for the partial image.")
    type: Literal["response.image_generation_call.partial_image"] = Field(
        description="The type of the event. Always `response.image_generation_call.partial_image`."
    )


# Ref: openai.types.responses.response_image_gen_call_completed_event.ResponseImageGenCallCompletedEvent
class ResponseImageGenCallCompletedEvent(_StreamEventItemBase):
    """Emitted when an image generation tool call has completed and the final image is available."""

    type: Literal["response.image_generation_call.completed"] = Field(
        description="The type of the event. Always `response.image_generation_call.completed`."
    )


# Stream events — MCP


# Ref: openai.types.responses.response_mcp_call_in_progress_event.ResponseMcpCallInProgressEvent
class ResponseMcpCallInProgressEvent(_StreamEventItemBase):
    """Emitted when an MCP tool call is in progress."""

    type: Literal["response.mcp_call.in_progress"] = Field(
        description="The type of the event. Always `response.mcp_call.in_progress`."
    )


# Ref: openai.types.responses.response_mcp_call_arguments_delta_event.ResponseMcpCallArgumentsDeltaEvent
class ResponseMcpCallArgumentsDeltaEvent(_StreamEventItemBase):
    """Emitted when there is a delta to the arguments of an MCP tool call."""

    delta: str = Field(
        description="A JSON string containing the partial update to the arguments for the MCP tool call."
    )
    type: Literal["response.mcp_call_arguments.delta"] = Field(
        description="The type of the event. Always `response.mcp_call_arguments.delta`."
    )


# Ref: openai.types.responses.response_mcp_call_arguments_done_event.ResponseMcpCallArgumentsDoneEvent
class ResponseMcpCallArgumentsDoneEvent(_StreamEventItemBase):
    """Emitted when the arguments for an MCP tool call are finalized."""

    arguments: str = Field(
        description="A JSON string containing the finalized arguments for the MCP tool call."
    )
    type: Literal["response.mcp_call_arguments.done"] = Field(
        description="The type of the event. Always `response.mcp_call_arguments.done`."
    )


# Ref: openai.types.responses.response_mcp_call_completed_event.ResponseMcpCallCompletedEvent
class ResponseMcpCallCompletedEvent(_StreamEventItemBase):
    """Emitted when an MCP tool call has completed successfully."""

    type: Literal["response.mcp_call.completed"] = Field(
        description="The type of the event. Always `response.mcp_call.completed`."
    )


# Ref: openai.types.responses.response_mcp_call_failed_event.ResponseMcpCallFailedEvent
class ResponseMcpCallFailedEvent(_StreamEventItemBase):
    """Emitted when an MCP tool call has failed."""

    type: Literal["response.mcp_call.failed"] = Field(
        description="The type of the event. Always `response.mcp_call.failed`."
    )


# Ref: openai.types.responses.response_mcp_list_tools_in_progress_event.ResponseMcpListToolsInProgressEvent
class ResponseMcpListToolsInProgressEvent(_StreamEventItemBase):
    """Emitted when the system is retrieving the list of available MCP tools."""

    type: Literal["response.mcp_list_tools.in_progress"] = Field(
        description="The type of the event. Always `response.mcp_list_tools.in_progress`."
    )


# Ref: openai.types.responses.response_mcp_list_tools_completed_event.ResponseMcpListToolsCompletedEvent
class ResponseMcpListToolsCompletedEvent(_StreamEventItemBase):
    """Emitted when the list of available MCP tools has been successfully retrieved."""

    type: Literal["response.mcp_list_tools.completed"] = Field(
        description="The type of the event. Always `response.mcp_list_tools.completed`."
    )


# Ref: openai.types.responses.response_mcp_list_tools_failed_event.ResponseMcpListToolsFailedEvent
class ResponseMcpListToolsFailedEvent(_StreamEventItemBase):
    """Emitted when the attempt to list available MCP tools has failed."""

    type: Literal["response.mcp_list_tools.failed"] = Field(
        description="The type of the event. Always `response.mcp_list_tools.failed`."
    )


# Stream events — custom tool call


# Ref: openai.types.responses.response_custom_tool_call_input_delta_event.ResponseCustomToolCallInputDeltaEvent
class ResponseCustomToolCallInputDeltaEvent(_StreamEventItemBase):
    """Emitted when there is a delta to the input of a custom tool call."""

    delta: str = Field(
        description="The incremental input data (delta) for the custom tool call."
    )
    type: Literal["response.custom_tool_call_input.delta"] = Field(
        description="The type of the event. Always `response.custom_tool_call_input.delta`."
    )


# Ref: openai.types.responses.response_custom_tool_call_input_done_event.ResponseCustomToolCallInputDoneEvent
class ResponseCustomToolCallInputDoneEvent(_StreamEventItemBase):
    """Emitted when input for a custom tool call is complete."""

    input: str = Field(description="The complete input data for the custom tool call.")
    type: Literal["response.custom_tool_call_input.done"] = Field(
        description="The type of the event. Always `response.custom_tool_call_input.done`."
    )


# ResponseStreamEvent union

# Ref: openai.types.responses.response_stream_event.ResponseStreamEvent
ResponseStreamEvent = Annotated[
    ResponseAudioDeltaEvent
    | ResponseAudioDoneEvent
    | ResponseAudioTranscriptDeltaEvent
    | ResponseAudioTranscriptDoneEvent
    | ResponseCodeInterpreterCallCodeDeltaEvent
    | ResponseCodeInterpreterCallCodeDoneEvent
    | ResponseCodeInterpreterCallCompletedEvent
    | ResponseCodeInterpreterCallInProgressEvent
    | ResponseCodeInterpreterCallInterpretingEvent
    | ResponseCompletedEvent
    | ResponseContentPartAddedEvent
    | ResponseContentPartDoneEvent
    | ResponseCreatedEvent
    | ResponseErrorEvent
    | ResponseFileSearchCallCompletedEvent
    | ResponseFileSearchCallInProgressEvent
    | ResponseFileSearchCallSearchingEvent
    | ResponseFunctionCallArgumentsDeltaEvent
    | ResponseFunctionCallArgumentsDoneEvent
    | ResponseInProgressEvent
    | ResponseFailedEvent
    | ResponseIncompleteEvent
    | ResponseOutputItemAddedEvent
    | ResponseOutputItemDoneEvent
    | ResponseReasoningSummaryPartAddedEvent
    | ResponseReasoningSummaryPartDoneEvent
    | ResponseReasoningSummaryTextDeltaEvent
    | ResponseReasoningSummaryTextDoneEvent
    | ResponseReasoningTextDeltaEvent
    | ResponseReasoningTextDoneEvent
    | ResponseRefusalDeltaEvent
    | ResponseRefusalDoneEvent
    | ResponseTextDeltaEvent
    | ResponseTextDoneEvent
    | ResponseWebSearchCallCompletedEvent
    | ResponseWebSearchCallInProgressEvent
    | ResponseWebSearchCallSearchingEvent
    | ResponseImageGenCallCompletedEvent
    | ResponseImageGenCallGeneratingEvent
    | ResponseImageGenCallInProgressEvent
    | ResponseImageGenCallPartialImageEvent
    | ResponseMcpCallArgumentsDeltaEvent
    | ResponseMcpCallArgumentsDoneEvent
    | ResponseMcpCallCompletedEvent
    | ResponseMcpCallFailedEvent
    | ResponseMcpCallInProgressEvent
    | ResponseMcpListToolsCompletedEvent
    | ResponseMcpListToolsFailedEvent
    | ResponseMcpListToolsInProgressEvent
    | ResponseOutputTextAnnotationAddedEvent
    | ResponseQueuedEvent
    | ResponseCustomToolCallInputDeltaEvent
    | ResponseCustomToolCallInputDoneEvent,
    Field(discriminator="type"),
]


# ResponseInputMessageItem  (items API response)


# Ref: openai.types.responses.response_input_message_item.ResponseInputMessageItem
class ResponseInputMessageItem(BaseModelResponse):
    """A message input item returned from the items API."""

    id: str = Field(description="The unique ID of the message input.")
    content: ResponseInputMessageContentList = Field(
        description="A list of one or many input items to the model, containing different content types."
    )
    role: Literal["user", "system", "developer"] = Field(
        description="The role of the message input. One of `user`, `system`, or `developer`."
    )
    type: Literal["message"] = Field(
        description="The type of the message input. Always set to `message`."
    )
    status: Literal["in_progress", "completed", "incomplete"] | None = Field(
        default=None,
        description="The status of item. One of `in_progress`, `completed`, or `incomplete`.",
    )


# ResponseItem union  (items returned via /v1/responses/{id}/input_items)

# Ref: openai.types.responses.response_item.ResponseItem
# NOTE: ResponseInputMessageItem and ResponseOutputMessage both carry type="message",
#       so a discriminated union is not possible; use a plain Union instead.
ResponseItem = (
    ResponseInputMessageItem
    | ResponseOutputMessage
    | ResponseFileSearchToolCall
    | ResponseComputerToolCall
    | ResponseComputerToolCallOutputItem
    | ResponseFunctionWebSearch
    | ResponseFunctionToolCallItem
    | ResponseFunctionToolCallOutputItem
    | ResponseToolSearchCall
    | ResponseToolSearchOutputItem
    | ResponseReasoningItem
    | ResponseCompactionItem
    | ImageGenerationCall
    | ResponseCodeInterpreterToolCall
    | LocalShellCall
    | LocalShellCallOutput
    | ResponseFunctionShellToolCall
    | ResponseFunctionShellToolCallOutput
    | ResponseApplyPatchToolCall
    | ResponseApplyPatchToolCallOutput
    | McpListTools
    | McpApprovalRequest
    | McpApprovalResponseOutput
    | McpCall
    | ResponseCustomToolCallItem
    | ResponseCustomToolCallOutputItem
    | Program
    | ProgramOutput
)


# ResponseItemList  (paginated list from /v1/responses/{id}/input_items)


# Ref: openai.types.responses.response_item_list.ResponseItemList
class ResponseItemList(PaginatedListEnvelope):
    """A paginated list of Response items."""

    data: list[ResponseItem] = Field(
        description="A list of items used to generate this response."
    )
    object: Literal["list"] = Field(
        description="The type of object returned. Always `list`."
    )


# ResponseCreateParams helpers


# Ref: openai.types.responses.response_create_params.ContextManagement
class ContextManagement(BaseModelRequest):
    """A context management entry for the request."""

    type: str = Field(
        description="The context management entry type. Currently only `compaction` is supported."
    )
    compact_threshold: int | None = Field(
        default=None,
        description="Token threshold at which compaction should be triggered for this entry.",
    )


# Ref: openai.types.responses.response_create_params.StreamOptions
class StreamOptions(BaseModelRequest):
    """Options for streaming responses. Only set this when you set `stream: true`."""

    include_obfuscation: bool | None = Field(
        default=None,
        description=(
            "When true, stream obfuscation will be enabled. "
            "Adds random characters to normalize payload sizes as a mitigation to side-channel attacks."
        ),
    )


# ResponseCreateParams  (request body for POST /v1/responses)


# Ref: openai.types.responses.response_conversation_param_param.ResponseConversationParamParam
class ConversationObject(BaseModelRequest):
    """A conversation reference passed as an object with an ID."""

    id: str = Field(description="The unique ID of the conversation.")


#: Conversation parameter: either a conversation ID string or an object with an `id` field.
ConversationParam = str | ConversationObject | None


def conversation_id_of(conversation: ConversationParam) -> str | None:
    """Return the conversation ID a ``conversation`` parameter names.

    Args:
        conversation: The request's ``conversation`` value, in either form.

    Returns:
        The conversation ID, or None when the parameter was not set.
    """
    return (
        conversation.id
        if isinstance(conversation, ConversationObject)
        else conversation
    )


def reject_conversation_with_previous_response(
    conversation: ConversationParam, previous_response_id: str | None
) -> None:
    """Reject a request that chains on a response and a conversation at once.

    Args:
        conversation: The request's ``conversation`` value.
        previous_response_id: The request's ``previous_response_id`` value.

    Raises:
        ApiError: 400 when both are set.
    """
    if conversation is None or previous_response_id is None:
        return
    error = ApiError(
        "Mutually exclusive parameters: provide only one of "
        "'previous_response_id' or 'conversation'."
    )
    error.code = "mutually_exclusive_parameters"
    raise error


# Ref: openai.types.responses.response_create_params.PromptCacheOptions
class PromptCacheOptions(BaseModelRequest):
    """Explicit prompt-caching configuration for a request."""

    mode: Literal["implicit", "explicit"] | None = Field(
        default=None,
        description="Caching mode. `explicit`: only content parts marked with "
        "`prompt_cache_breakpoint` are cached. `implicit` (default): sections "
        "selected with `prompt_cache_key` are cached too.",
    )
    ttl: PromptCacheOptionsTTL | None = Field(
        default=None,
        description="Cache retention: `30m` is applied as 1h. "
        "Ignored when `prompt_cache_retention` is set.",
    )


#: Unsupported-parameter values naming the behavior this implementation already has.
_ACCEPTED_DEFAULTS: Final[dict[str, str]] = {"truncation": "disabled"}


# Ref: openai.types.responses.response_create_params.ResponseCreateParamsBase
class ResponseCreateParams(BaseModelRequestWithExtra):
    """Request body for POST /v1/responses.

    Undeclared fields are forwarded to the model as extra parameters.
    """

    model: str = Field(
        min_length=1,
        max_length=255,
        description="Model ID used to generate the response. Wildcard patterns "
        "are accepted and select the most recent matching model.",
    )
    input: str | ResponseInputParam | None = Field(
        default=None,
        description="Text, image, or file inputs to the model, used to generate a response.",
    )
    background: bool | None = Field(
        default=None,
        description="Whether to run the model response in the background. "
        "Accepted for compatibility and ignored.",
    )
    client_metadata: JsonMapping | None = Field(
        default=None,
        description="Client environment metadata (sent by newer OpenAI "
        "clients such as Codex). Accepted for compatibility and ignored.",
    )
    context_management: list[ContextManagement] | None = Field(
        default=None,
        description="Context management configuration for this request.\nUNSUPPORTED on this implementation.",
    )
    conversation: ConversationParam = Field(
        default=None,
        description="The conversation this response belongs to: its items are "
        "prepended to the input, and the request's input and the response's "
        "output items are appended to it unless `store` is false. Cannot be "
        "used with `previous_response_id`.",
    )
    include: list[ResponseIncludable] | None = Field(
        default=None,
        description="Specify additional output data to include in the model response. "
        "`reasoning.encrypted_content` is honored (reasoning items carry a "
        "self-contained round-trip envelope) and `file_search_call.results` "
        "attaches the retrieved passages; other values are accepted and ignored.",
    )
    instructions: str | None = Field(
        default=None,
        description="A system (or developer) message inserted into the model's context.",
    )
    max_output_tokens: int | None = Field(
        default=None,
        gt=0,
        description="An upper bound for the number of tokens that can be generated for a response.",
    )
    max_tool_calls: int | None = Field(
        default=None,
        description="The maximum number of total calls to built-in tools that can be processed in a response.\nUNSUPPORTED on this implementation.",
    )
    metadata: Metadata | None = Field(
        default=None,
        description="Set of 16 key-value pairs that can be attached to an object.",
    )
    moderation: RequestModeration | None = Field(
        default=None,
        description="Apply an AWS Bedrock guardrail to this request; results "
        "are reported in the response `moderation` field (on the terminal "
        "event when streaming).",
    )
    parallel_tool_calls: bool | None = Field(
        default=None,
        description="Whether to allow the model to run tool calls in parallel.",
    )
    previous_response_id: str | None = Field(
        default=None,
        description="The unique ID of the previous response. Use to create "
        "multi-turn conversations; the previous response must have been "
        "created with `store=true`.",
    )
    prompt: ResponsePrompt | None = Field(
        default=None,
        description="Reference to a prompt template and its variables. `id` must "
        "be an Amazon Bedrock Prompt Management prompt ARN, and the server must "
        "allow prompt ARNs. Variable values must be plain strings.",
    )
    prompt_cache_key: str | None = Field(
        default=None, description="Cache key for similar requests."
    )
    prompt_cache_options: PromptCacheOptions | None = Field(
        default=None, description="Explicit prompt-caching configuration."
    )
    prompt_cache_retention: PromptCacheRetention | None = Field(
        default=None, description="Cache retention policy."
    )
    reasoning: Reasoning | None = Field(
        default=None, description="Reasoning configuration."
    )
    safety_identifier: str | None = Field(
        default=None,
        description="User policy violation identifier. Accepted for "
        "compatibility and ignored by generation; recorded in request logs.",
    )
    service_tier: ServiceTiers | None = Field(
        default=None, description="Service tier for request."
    )
    store: bool | None = Field(
        default=None,
        description="Persist the response for later retrieval and multi-turn "
        "continuation. Defaults to false on this implementation. Ignored "
        "(with a request-log warning) when streaming or when storage is "
        "not enabled on the server.",
    )
    stream: bool | None = Field(
        default=None, description="Stream response as it is generated."
    )
    stream_options: StreamOptions | None = Field(
        default=None,
        description="Streaming options. Accepted for compatibility and ignored.",
    )
    temperature: float | None = Field(
        default=None, ge=0, le=2, description="Sampling temperature (0-2)."
    )
    text: ResponseTextConfig | None = Field(
        default=None, description="Text response config."
    )
    tool_choice: ToolChoice | None = Field(
        default=None, description="Tool selection method."
    )
    tools: list[Tool] | None = Field(default=None, description="Available tools.")
    top_logprobs: int | None = Field(
        default=None, ge=0, le=20, description="Top logprobs count (0-20)."
    )
    top_p: float | None = Field(
        default=None, ge=0, le=1, description="Nucleus sampling parameter."
    )
    truncation: Literal["auto", "disabled"] | None = Field(
        default=None,
        description="Truncation strategy. `disabled` (the default) is the "
        "behavior served; `auto` is UNSUPPORTED on this implementation.",
    )
    user: str | None = Field(
        default=None, description="User identifier (use safety_identifier instead)."
    )

    # Extra validations
    _UNSUPPORTED: ClassVar[frozenset[str]] = frozenset(
        {
            # Ignored silently: "background", "safety_identifier", "stream_options"
            "context_management",
            "max_tool_calls",
            "truncation",
        }
    )

    #: Parameters a managed prompt template carries itself, so a request cannot also set them
    _PROMPT_INCOMPATIBLE: ClassVar[frozenset[str]] = frozenset(
        {
            "input",
            "instructions",
            "max_output_tokens",
            "previous_response_id",
            "reasoning",
            "temperature",
            "text",
            "tool_choice",
            "tools",
            "top_p",
        }
    )

    @model_validator(mode="after")
    def _mutually_exclusive(self) -> Self:
        """Validate that the two conversation-chaining parameters are not combined.

        Raises:
            ApiError: If both ``conversation`` and ``previous_response_id`` are set.
        """
        reject_conversation_with_previous_response(
            self.conversation, self.previous_response_id
        )
        return self

    @model_validator(mode="after")
    def _unsupported(self) -> Self:
        """Validate that unsupported parameters are not used.

        Tool types without a backend equivalent (computer use, MCP, shells,
        custom/namespace tools, ...) are accepted for compatibility and dropped
        from the Bedrock tool configuration.

        A ``prompt`` request body carries only the prompt variables: Amazon
        Bedrock renders the messages, system prompt, tools and inference
        parameters from the stored prompt version, so any request-level
        equivalent is rejected instead of being silently dropped.

        Raises:
            UnsupportedParameterError: If a parameter marked as unsupported is used.
            ApiError: If a parameter is incompatible with ``prompt``.
        """
        for key in self._UNSUPPORTED & self.model_fields_set:
            # `null`/`false`, and a value naming the behavior already in force,
            # request the supported default behavior, like omission
            value = getattr(self, key)
            if (
                value is not None
                and value is not False
                and value != _ACCEPTED_DEFAULTS.get(key)
            ):
                raise UnsupportedParameterError(key)
        if self.prompt is not None and (
            incompatible := sorted(self._PROMPT_INCOMPATIBLE & self.model_fields_set)
        ):
            msg = (
                f"Parameter(s) {', '.join(f"'{key}'" for key in incompatible)} cannot "
                "be used with 'prompt': the prompt template provides them."
            )
            raise ApiError(msg)
        return self


# Ref: openai.types.responses.input_token_count_params.InputTokenCountParams
class InputTokenCountParams(BaseModelRequest):
    """Request body for POST /v1/responses/input_tokens.

    Counts input tokens without producing a response.
    """

    model: str = Field(
        description="Model ID. Wildcard patterns are accepted and select the "
        "most recent matching model."
    )
    input: str | ResponseInputParam | None = Field(
        default=None, description="Text, image, or file inputs."
    )
    instructions: str | None = Field(
        default=None,
        description="System message. Not carried over with previous_response_id.",
    )
    tools: list[Tool] | None = Field(default=None, description="Available tools.")
    tool_choice: ToolChoice | None = Field(default=None, description="Tool selection.")
    parallel_tool_calls: bool | None = Field(
        default=None, description="Allow parallel tool calls."
    )
    reasoning: Reasoning | None = Field(
        default=None, description="Reasoning configuration."
    )
    text: ResponseTextConfig | None = Field(
        default=None,
        description="Text response config.\nUNSUPPORTED on this implementation.",
    )
    truncation: Literal["auto", "disabled"] | None = Field(
        default=None,
        description="Truncation strategy. `disabled` (the default) is the "
        "behavior served; `auto` is UNSUPPORTED on this implementation.",
    )
    previous_response_id: str | None = Field(
        default=None,
        description="Previous response ID for multi-turn.\n"
        "UNSUPPORTED on this implementation.",
    )
    conversation: ConversationParam = Field(
        default=None,
        description="The conversation whose items are counted ahead of `input`. "
        "Cannot be used with `previous_response_id`.",
    )
    personality: str | None = Field(
        default=None,
        description="Personality preset applied to the model. Accepted for "
        "compatibility and ignored.",
    )

    # Extra validations
    _UNSUPPORTED: ClassVar[frozenset[str]] = frozenset(
        {"text", "truncation", "previous_response_id"}
    )

    @model_validator(mode="after")
    def _mutually_exclusive(self) -> Self:
        """Validate that the two conversation-chaining parameters are not combined.

        Raises:
            ApiError: If both ``conversation`` and ``previous_response_id`` are set.
        """
        reject_conversation_with_previous_response(
            self.conversation, self.previous_response_id
        )
        return self

    @model_validator(mode="after")
    def _unsupported(self) -> Self:
        """Validate that unsupported parameters are not used.

        Raises:
            UnsupportedParameterError: If a parameter marked as unsupported is used.
        """
        for key in self._UNSUPPORTED & self.model_fields_set:
            # `null`/`false`, and a value naming the behavior already in force,
            # request the supported default behavior, like omission
            value = getattr(self, key)
            if (
                value is not None
                and value is not False
                and value != _ACCEPTED_DEFAULTS.get(key)
            ):
                raise UnsupportedParameterError(key)
        return self


# Ref: openai.types.responses.input_token_count_response.InputTokenCountResponse
class InputTokenCountResponse(BaseModelResponse):
    """Response body for POST /v1/responses/input_tokens."""

    object: Literal["response.input_tokens"] = "response.input_tokens"
    input_tokens: int = Field(description="Total input token count.")


class ResponseDeleted(BaseModelResponse):
    """Stored response deletion confirmation."""

    id: str = Field(description="Identifier of the deleted response.")
    object: Literal["response.deleted"] = Field(
        default="response.deleted",
        description="The object type, which is always `response.deleted`.",
    )
    deleted: bool = Field(default=True, description="Whether the response was deleted.")


# Ref: openai.resources.responses.Responses.compact (openai-python SDK)
class CompactParams(BaseModelRequest):
    """Request body for POST /v1/responses/compact.

    Compacts a conversation into a single ``compaction`` item that later
    requests can send back as input.
    """

    model: str = Field(
        description="Model ID. Wildcard patterns are accepted and select the "
        "most recent matching model."
    )
    input: ResponseInputParam | None = Field(
        default=None, description="Conversation inputs to compact."
    )
    instructions: str | None = Field(
        default=None, description="System message inserted into the model's context."
    )
    previous_response_id: str | None = Field(
        default=None,
        description="ID of a stored response whose conversation is compacted "
        "along with the new input.",
    )
    prompt_cache_key: str | None = Field(
        default=None, description="Cache key for similar requests."
    )
    prompt_cache_options: PromptCacheOptions | None = Field(
        default=None, description="Explicit prompt-caching configuration."
    )
    prompt_cache_retention: PromptCacheRetention | None = Field(
        default=None, description="Cache retention policy."
    )
    service_tier: ServiceTiers | None = Field(
        default=None, description="Service tier for request."
    )


class CompactionUserMessage(BaseModelResponse):
    """A user message echoed in the compacted output."""

    id: str = Field(description="Unique identifier of the echoed message.")
    type: Literal["message"] = Field(
        default="message", description="The item type, which is always `message`."
    )
    status: Literal["completed"] = Field(
        default="completed", description="The item status, which is always `completed`."
    )
    role: Literal["user"] = Field(
        default="user", description="The message role, which is always `user`."
    )
    content: list[JsonMapping] = Field(
        description="The message content parts (e.g. `input_text`)."
    )


# Ref: openai.types.responses.compacted_response.CompactedResponse
class CompactedResponse(BaseModelResponse):
    """Response body for POST /v1/responses/compact."""

    id: str = Field(description="Compacted response ID.")
    created_at: int = Field(
        description="Unix timestamp (in seconds) when the compaction was created."
    )
    object: Literal["response.compaction"] = Field(
        default="response.compaction",
        description="The object type, which is always `response.compaction`.",
    )
    output: list[CompactionUserMessage | ResponseCompactionItem] = Field(
        description="The conversation's user messages followed by the compaction item."
    )
    usage: ResponseUsage = Field(description="Token usage of the compaction request.")
