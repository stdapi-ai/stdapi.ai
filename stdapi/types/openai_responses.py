"""Local OpenAI-compatible Responses API types."""

from typing import Annotated, ClassVar, Literal, Self

from pydantic import ConfigDict, Field, model_validator

from stdapi.api_errors import UnsupportedParameterError
from stdapi.types import (
    BaseModelRequest,
    BaseModelRequestWithExtra,
    BaseModelResponse,
    JsonMapping,
)
from stdapi.types.openai import Metadata, ResponseFormatJSONObject, ResponseFormatText

# ---------------------------------------------------------------------------
# Literals / type aliases
# ---------------------------------------------------------------------------

#: Response status values.
ResponseStatus = Literal[
    "completed", "failed", "in_progress", "cancelled", "queued", "incomplete"
]

#: Item-level status values.
ResponseItemStatus = Literal["in_progress", "completed", "incomplete"]

#: Reasoning effort levels for reasoning models.
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]

#: Verbosity levels used across multiple parameters.
VerbosityLevel = Literal["low", "medium", "high"]

#: Service tier options.
ServiceTiers = Literal["auto", "default", "flex", "scale", "priority"]

#: Prompt cache retention options.
PromptCacheRetention = Literal["in-memory", "24h"]

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


# ---------------------------------------------------------------------------
# Filter types  (used in FileSearchTool)
# ---------------------------------------------------------------------------


# Ref: openai.types.shared.comparison_filter.ComparisonFilter
class ComparisonFilter(BaseModelRequest):
    """Compares a specified attribute key to a given value using a defined operator."""

    key: str = Field(description="The key to compare against the value.")
    type: Literal["eq", "ne", "gt", "gte", "lt", "lte", "in", "nin"] = Field(
        description=(
            "Specifies the comparison operator: `eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `in`, `nin`."
        )
    )
    value: str | float | bool | list[str | float] = Field(
        description="The value to compare against the attribute key; supports string, number, or boolean types."
    )


# Ref: openai.types.shared.compound_filter.CompoundFilter
class CompoundFilter(BaseModelRequest):
    """Combine multiple filters using `and` or `or`."""

    filters: list[ComparisonFilter | object] = Field(
        description="Array of filters to combine. Items can be `ComparisonFilter` or `CompoundFilter`."
    )
    type: Literal["and", "or"] = Field(description="Type of operation: `and` or `or`.")


#: Union of filter types applicable to file search.
FileSearchFilters = ComparisonFilter | CompoundFilter | None


# ---------------------------------------------------------------------------
# Custom tool input format  (used in CustomTool)
# ---------------------------------------------------------------------------


# Ref: openai.types.shared.custom_tool_input_format.Text
class CustomToolInputFormatText(BaseModelRequest):
    """Unconstrained free-form text input format."""

    type: Literal["text"] = Field(
        description="Unconstrained text format. Always `text`."
    )


# Ref: openai.types.shared.custom_tool_input_format.Grammar
class CustomToolInputFormatGrammar(BaseModelRequest):
    """A grammar-constrained input format."""

    definition: str = Field(description="The grammar definition.")
    syntax: Literal["lark", "regex"] = Field(
        description="The syntax of the grammar definition. One of `lark` or `regex`."
    )
    type: Literal["grammar"] = Field(description="Grammar format. Always `grammar`.")


# Ref: openai.types.shared.custom_tool_input_format.CustomToolInputFormat
CustomToolInputFormat = Annotated[
    CustomToolInputFormatText | CustomToolInputFormatGrammar,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Container / environment types  (used in tool definitions)
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.container_network_policy_disabled.ContainerNetworkPolicyDisabled
class ContainerNetworkPolicyDisabled(BaseModelRequest):
    """Disable outbound network access from the container."""

    type: Literal["disabled"] = Field(description="Always `disabled`.")


# Ref: openai.types.responses.container_network_policy_domain_secret.ContainerNetworkPolicyDomainSecret
class ContainerNetworkPolicyDomainSecret(BaseModelRequest):
    """A domain-scoped secret injected for an allowlisted domain."""

    domain: str = Field(description="The domain associated with the secret.")
    name: str = Field(description="The name of the secret to inject for the domain.")
    value: str = Field(description="The secret value to inject for the domain.")


# Ref: openai.types.responses.container_network_policy_allowlist.ContainerNetworkPolicyAllowlist
class ContainerNetworkPolicyAllowlist(BaseModelRequest):
    """Allow outbound network access only to specified domains."""

    allowed_domains: list[str] = Field(
        description="A list of allowed domains when type is `allowlist`."
    )
    type: Literal["allowlist"] = Field(description="Always `allowlist`.")
    domain_secrets: list[ContainerNetworkPolicyDomainSecret] | None = Field(
        default=None,
        description="Optional domain-scoped secrets for allowlisted domains.",
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

    type: Literal["container_auto"] = Field(description="Always `container_auto`.")
    file_ids: list[str] | None = Field(
        default=None,
        description="An optional list of uploaded files to make available to your code.",
    )
    memory_limit: Literal["1g", "4g", "16g", "64g"] | None = Field(
        default=None, description="The memory limit for the container."
    )
    network_policy: ContainerNetworkPolicy | None = Field(
        default=None, description="Network access policy for the container."
    )
    skills: list[ContainerSkill] | None = Field(
        default=None,
        description="An optional list of skills referenced by id or inline data.",
    )


# Ref: openai.types.responses.container_reference.ContainerReference
class ContainerReference(BaseModelRequest):
    """References a container created with the /v1/containers endpoint."""

    container_id: str = Field(description="The ID of the referenced container.")
    type: Literal["container_reference"] = Field(
        description="References a container created with the /v1/containers endpoint."
    )


# Ref: openai.types.responses.local_environment.LocalEnvironment
class LocalEnvironment(BaseModelRequest):
    """Use a local computer environment."""

    type: Literal["local"] = Field(description="Always `local`.")
    skills: list[LocalSkill] | None = Field(
        default=None, description="An optional list of skills."
    )


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.function_tool.FunctionTool
class FunctionTool(BaseModelRequest):
    """Defines a function in your own code the model can choose to call."""

    name: str = Field(description="The name of the function to call.")
    type: Literal["function"] = Field(
        description="The type of the function tool. Always `function`."
    )
    defer_loading: bool | None = Field(
        default=None,
        description="Whether this function is deferred and loaded via tool search.",
    )
    description: str | None = Field(
        default=None,
        description="A description of the function. Used by the model to determine whether or not to call the function.",
    )
    parameters: JsonMapping | None = Field(
        default=None,
        description="A JSON schema object describing the parameters of the function.",
    )
    strict: bool | None = Field(
        default=None,
        description="Whether to enforce strict parameter validation. Default `true`.",
    )


# Ref: openai.types.responses.file_search_tool.RankingOptionsHybridSearch
class FileSearchRankingOptionsHybridSearch(BaseModelRequest):
    """Hybrid search weighting for reciprocal rank fusion."""

    embedding_weight: float = Field(
        description="The weight of the embedding in the reciprocal ranking fusion."
    )
    text_weight: float = Field(
        description="The weight of the text in the reciprocal ranking fusion."
    )


# Ref: openai.types.responses.file_search_tool.RankingOptions
class FileSearchRankingOptions(BaseModelRequest):
    """Ranking options for file search."""

    hybrid_search: FileSearchRankingOptionsHybridSearch | None = Field(
        default=None,
        description="Weights that control how reciprocal rank fusion balances semantic embedding matches versus sparse keyword matches when hybrid search is enabled.",
    )
    ranker: Literal["auto", "default-2024-11-15"] | None = Field(
        default=None, description="The ranker to use for the file search."
    )
    score_threshold: float | None = Field(
        default=None,
        description="The score threshold for the file search, a number between 0 and 1.",
    )


# Ref: openai.types.responses.file_search_tool.FileSearchTool
class FileSearchTool(BaseModelRequest):
    """A tool that searches for relevant content from uploaded files.

    UNSUPPORTED on this implementation.
    """

    type: Literal["file_search"] = Field(
        description="The type of the file search tool. Always `file_search`."
    )
    vector_store_ids: list[str] = Field(
        description="The IDs of the vector stores to search."
    )
    filters: FileSearchFilters = Field(default=None, description="A filter to apply.")
    max_num_results: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="The maximum number of results to return. This number should be between 1 and 50 inclusive.",
    )
    ranking_options: FileSearchRankingOptions | None = Field(
        default=None, description="Ranking options for search."
    )


# Ref: openai.types.responses.web_search_tool.Filters
class WebSearchFilters(BaseModelRequest):
    """Filters for web search."""

    allowed_domains: list[str] | None = Field(
        default=None,
        description="Allowed domains for the search. If not provided, all domains are allowed. Subdomains of the provided domains are allowed as well.",
    )


# Ref: openai.types.responses.web_search_tool.UserLocation
class WebSearchUserLocation(BaseModelRequest):
    """The approximate location of the user."""

    city: str | None = Field(
        default=None, description="Free text input for the city of the user."
    )
    country: str | None = Field(
        default=None, description="The two-letter ISO country code of the user."
    )
    region: str | None = Field(
        default=None, description="Free text input for the region of the user."
    )
    timezone: str | None = Field(
        default=None, description="The IANA timezone of the user."
    )
    type: Literal["approximate"] | None = Field(
        default=None,
        description="The type of location approximation. Always `approximate`.",
    )


# Ref: openai.types.responses.web_search_tool.WebSearchTool
class WebSearchTool(BaseModelRequest):
    """Search the web for sources related to the prompt."""

    type: Literal["web_search", "web_search_2025_08_26"] = Field(
        description="The type of the web search tool. One of `web_search` or `web_search_2025_08_26`."
    )
    filters: WebSearchFilters | None = Field(
        default=None, description="Filters for the search."
    )
    search_context_size: VerbosityLevel | None = Field(
        default=None,
        description="High level guidance for the amount of context window space to use for the search. One of `low`, `medium`, or `high`. `medium` is the default.",
    )
    user_location: WebSearchUserLocation | None = Field(
        default=None, description="The approximate location of the user."
    )
    external_web_access: bool | None = Field(
        default=None, description="Whether external web access is allowed."
    )


# Ref: openai.types.responses.web_search_preview_tool.UserLocation
class WebSearchPreviewUserLocation(BaseModelRequest):
    """The user's location for web search preview."""

    type: Literal["approximate"] = Field(
        description="The type of location approximation. Always `approximate`."
    )
    city: str | None = Field(
        default=None, description="Free text input for the city of the user."
    )
    country: str | None = Field(
        default=None, description="The two-letter ISO country code of the user."
    )
    region: str | None = Field(
        default=None, description="Free text input for the region of the user."
    )
    timezone: str | None = Field(
        default=None, description="The IANA timezone of the user."
    )


# Ref: openai.types.responses.web_search_preview_tool.WebSearchPreviewTool
class WebSearchPreviewTool(BaseModelRequest):
    """This tool searches the web for relevant results to use in a response."""

    type: Literal["web_search_preview", "web_search_preview_2025_03_11"] = Field(
        description="The type of the web search tool. One of `web_search_preview` or `web_search_preview_2025_03_11`."
    )
    search_content_types: list[Literal["text", "image"]] | None = Field(
        default=None, description="Content types to include in search results."
    )
    search_context_size: VerbosityLevel | None = Field(
        default=None,
        description="High level guidance for the amount of context window space to use for the search. One of `low`, `medium`, or `high`. `medium` is the default.",
    )
    user_location: WebSearchPreviewUserLocation | None = Field(
        default=None, description="The user's location."
    )
    external_web_access: bool | None = Field(
        default=None, description="Whether external web access is allowed. "
    )


# Ref: openai.types.responses.computer_tool.ComputerTool
class ComputerTool(BaseModelRequest):
    """A tool that controls a virtual computer.

    UNSUPPORTED on this implementation.
    """

    type: Literal["computer"] = Field(
        description="The type of the computer tool. Always `computer`."
    )


# Ref: openai.types.responses.computer_use_preview_tool.ComputerUsePreviewTool
class ComputerUsePreviewTool(BaseModelRequest):
    """A tool that controls a virtual computer (preview version).

    UNSUPPORTED on this implementation.
    """

    display_height: int = Field(description="The height of the computer display.")
    display_width: int = Field(description="The width of the computer display.")
    environment: Literal["windows", "mac", "linux", "ubuntu", "browser"] = Field(
        description="The type of computer environment to control."
    )
    type: Literal["computer_use_preview"] = Field(
        description="The type of the computer use tool. Always `computer_use_preview`."
    )


# Ref: openai.types.responses.tool.McpAllowedToolsMcpToolFilter
class McpAllowedToolsFilter(BaseModelRequest):
    """A filter object to specify which MCP tools are allowed."""

    read_only: bool | None = Field(
        default=None,
        description="Indicates whether or not a tool modifies data or is read-only.",
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

    server_label: str = Field(
        description="A label for this MCP server, used to identify it in tool calls."
    )
    type: Literal["mcp"] = Field(description="The type of the MCP tool. Always `mcp`.")
    allowed_tools: McpAllowedTools = Field(
        default=None, description="List of allowed tool names or a filter object."
    )
    authorization: str | None = Field(
        default=None, description="An OAuth access token for the remote MCP server."
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
        description=(
            "Identifier for service connectors. One of `server_url` or `connector_id` must be provided."
        ),
    )
    defer_loading: bool | None = Field(
        default=None,
        description="Whether this MCP tool is deferred and discovered via tool search.",
    )
    headers: dict[str, str] | None = Field(
        default=None, description="Optional HTTP headers to send to the MCP server."
    )
    require_approval: McpRequireApproval = Field(
        default=None,
        description="Specify which of the MCP server's tools require approval.",
    )
    server_description: str | None = Field(
        default=None, description="Optional description of the MCP server."
    )
    server_url: str | None = Field(
        default=None,
        description="The URL for the MCP server. One of `server_url` or `connector_id` must be provided.",
    )


# Ref: openai.types.responses.tool.CodeInterpreterContainerCodeInterpreterToolAuto
class CodeInterpreterContainerAuto(BaseModelRequest):
    """Configuration for a code interpreter container."""

    type: Literal["auto"] = Field(description="Always `auto`.")
    file_ids: list[str] | None = Field(
        default=None,
        description="An optional list of uploaded files to make available to your code.",
    )
    memory_limit: Literal["1g", "4g", "16g", "64g"] | None = Field(
        default=None, description="The memory limit for the code interpreter container."
    )
    network_policy: ContainerNetworkPolicy | None = Field(
        default=None, description="Network access policy for the container."
    )


#: Code interpreter container specification.
CodeInterpreterContainer = str | CodeInterpreterContainerAuto


# Ref: openai.types.responses.tool.CodeInterpreter
class CodeInterpreter(BaseModelRequest):
    """A tool that runs Python code to help generate a response to a prompt."""

    container: CodeInterpreterContainer | None = Field(
        default=None,
        description="The code interpreter container. Can be a container ID or an object that specifies uploaded file IDs to make available to your code.",
    )
    type: Literal["code_interpreter"] = Field(
        description="The type of the code interpreter tool. Always `code_interpreter`."
    )


# Ref: openai.types.responses.tool.ImageGenerationInputImageMask
class ImageGenerationInputImageMask(BaseModelRequest):
    """Optional mask for inpainting."""

    file_id: str | None = Field(default=None, description="File ID for the mask image.")
    image_url: str | None = Field(
        default=None, description="Base64-encoded mask image."
    )


# Ref: openai.types.responses.tool.ImageGeneration
class ImageGeneration(BaseModelRequest):
    """A tool that generates images."""

    type: Literal["image_generation"] = Field(
        description="The type of the image generation tool. Always `image_generation`."
    )
    action: Literal["generate", "edit", "auto"] | None = Field(
        default=None,
        description="Whether to generate a new image or edit an existing image. Default: `auto`.",
    )
    background: Literal["transparent", "opaque", "auto"] | None = Field(
        default=None,
        description="Background type for the generated image. One of `transparent`, `opaque`, or `auto`. Default: `auto`.",
    )
    input_fidelity: Literal["high", "low"] | None = Field(
        default=None,
        description="Control how much effort the model will exert to match the style and features of input images. Supports `high` and `low`. Defaults to `low`.",
    )
    input_image_mask: ImageGenerationInputImageMask | None = Field(
        default=None, description="Optional mask for inpainting."
    )
    model: str | None = Field(
        default=None, description="The image generation model to use."
    )
    moderation: Literal["auto", "low"] | None = Field(
        default=None,
        description="Moderation level for the generated image. Default: `auto`.",
    )
    output_compression: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Compression level for the output image. Default: 100.",
    )
    output_format: Literal["png", "webp", "jpeg"] | None = Field(
        default=None,
        description="The output format of the generated image. One of `png`, `webp`, or `jpeg`. Default: `png`.",
    )
    partial_images: int | None = Field(
        default=None,
        ge=0,
        le=3,
        description="Number of partial images to generate in streaming mode, from 0 (default) to 3.",
    )
    quality: Literal["low", "medium", "high", "auto"] | None = Field(
        default=None,
        description="The quality of the generated image. One of `low`, `medium`, `high`, or `auto`. Default: `auto`.",
    )
    size: Literal["1024x1024", "1024x1536", "1536x1024", "auto"] | None = Field(
        default=None,
        description="The size of the generated image. One of `1024x1024`, `1024x1536`, `1536x1024`, or `auto`. Default: `auto`.",
    )


# Ref: openai.types.responses.tool.LocalShell
class LocalShell(BaseModelRequest):
    """A tool that allows the model to execute shell commands in a local environment.

    UNSUPPORTED on this implementation.
    """

    type: Literal["local_shell"] = Field(
        description="The type of the local shell tool. Always `local_shell`."
    )


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

    type: Literal["shell"] = Field(
        description="The type of the shell tool. Always `shell`."
    )
    environment: FunctionShellEnvironment = Field(
        default=None, description="The environment in which to execute shell commands."
    )


# Ref: openai.types.responses.custom_tool.CustomTool
class CustomTool(BaseModelRequest):
    """A custom tool that processes input using a specified format.

    UNSUPPORTED on this implementation.
    """

    name: str = Field(
        description="The name of the custom tool, used to identify it in tool calls."
    )
    type: Literal["custom"] = Field(
        description="The type of the custom tool. Always `custom`."
    )
    defer_loading: bool | None = Field(
        default=None,
        description="Whether this tool should be deferred and discovered via tool search.",
    )
    description: str | None = Field(
        default=None, description="Optional description of the custom tool."
    )
    format: CustomToolInputFormat | None = Field(
        default=None,
        description="The input format for the custom tool. Default is unconstrained text.",
    )


# Ref: openai.types.responses.namespace_tool.ToolFunction
class NamespaceToolFunction(BaseModelRequest):
    """A function tool within a namespace."""

    name: str = Field(description="The name of the function.")
    type: Literal["function"] = Field(description="Always `function`.")
    defer_loading: bool | None = Field(
        default=None,
        description="Whether this function should be deferred and discovered via tool search.",
    )
    description: str | None = Field(
        default=None, description="Description of the function."
    )
    parameters: object | None = Field(
        default=None, description="The function's parameter schema."
    )
    strict: bool | None = Field(
        default=None, description="Whether to enforce strict parameter validation."
    )


# Ref: openai.types.responses.namespace_tool.Tool
NamespaceToolTool = Annotated[
    NamespaceToolFunction | CustomTool, Field(discriminator="type")
]


# Ref: openai.types.responses.namespace_tool.NamespaceTool
class NamespaceTool(BaseModelRequest):
    """Groups function/custom tools under a shared namespace.

    UNSUPPORTED on this implementation.
    """

    description: str = Field(
        description="A description of the namespace shown to the model."
    )
    name: str = Field(description="The namespace name used in tool calls.")
    tools: list[NamespaceToolTool] = Field(
        description="The function/custom tools available inside this namespace."
    )
    type: Literal["namespace"] = Field(
        description="The type of the tool. Always `namespace`."
    )


# Ref: openai.types.responses.tool_search_tool.ToolSearchTool
class ToolSearchTool(BaseModelRequest):
    """Hosted or BYOT tool search configuration for deferred tools.

    UNSUPPORTED on this implementation.
    """

    type: Literal["tool_search"] = Field(
        description="The type of the tool. Always `tool_search`."
    )
    description: str | None = Field(
        default=None,
        description="Description shown to the model for a client-executed tool search tool.",
    )
    execution: Literal["server", "client"] | None = Field(
        default=None,
        description="Whether tool search is executed by the server or by the client.",
    )
    parameters: object | None = Field(
        default=None,
        description="Parameter schema for a client-executed tool search tool.",
    )


# Ref: openai.types.responses.apply_patch_tool.ApplyPatchTool
class ApplyPatchTool(BaseModelRequest):
    """Allows the assistant to create, delete, or update files using unified diffs.

    UNSUPPORTED on this implementation.
    """

    type: Literal["apply_patch"] = Field(
        description="The type of the tool. Always `apply_patch`."
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
    | ApplyPatchTool,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Tool choice types
# ---------------------------------------------------------------------------


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
    ] = Field(description="The type of built-in tool the model should use.")


# Ref: openai.types.responses.tool_choice_function.ToolChoiceFunction
class ToolChoiceFunction(BaseModelRequest):
    """Force the model to call a specific function."""

    name: str = Field(description="The name of the function to call.")
    type: Literal["function"] = Field(
        description="For function calling, the type is always `function`."
    )


# Ref: openai.types.responses.tool_choice_allowed.ToolChoiceAllowed
class ToolChoiceAllowed(BaseModelRequest):
    """Constrains the tools available to the model to a pre-defined set."""

    mode: Literal["auto", "required"] = Field(
        description="`auto` allows the model to pick from among the allowed tools. `required` requires the model to call one or more of the allowed tools."
    )
    tools: list[JsonMapping] = Field(
        description="A list of tool definitions that the model should be allowed to call."
    )
    type: Literal["allowed_tools"] = Field(description="Always `allowed_tools`.")


# Ref: openai.types.responses.tool_choice_mcp.ToolChoiceMcp
class ToolChoiceMcp(BaseModelRequest):
    """Force the model to call a specific tool on a remote MCP server."""

    server_label: str = Field(description="The label of the MCP server to use.")
    type: Literal["mcp"] = Field(description="For MCP tools, the type is always `mcp`.")
    name: str | None = Field(
        default=None, description="The name of the tool to call on the server."
    )


# Ref: openai.types.responses.tool_choice_custom.ToolChoiceCustom
class ToolChoiceCustom(BaseModelRequest):
    """Force the model to call a specific custom tool."""

    name: str = Field(description="The name of the custom tool to call.")
    type: Literal["custom"] = Field(
        description="For custom tool calling, the type is always `custom`."
    )


# Ref: openai.types.responses.tool_choice_apply_patch.ToolChoiceApplyPatch
class ToolChoiceApplyPatch(BaseModelRequest):
    """Forces the model to call the apply_patch tool when executing a tool call."""

    type: Literal["apply_patch"] = Field(
        description="The tool to call. Always `apply_patch`."
    )


# Ref: openai.types.responses.tool_choice_shell.ToolChoiceShell
class ToolChoiceShell(BaseModelRequest):
    """Forces the model to call the shell tool when a tool call is required."""

    type: Literal["shell"] = Field(description="The tool to call. Always `shell`.")


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
)


# ---------------------------------------------------------------------------
# Reasoning configuration
# ---------------------------------------------------------------------------


# Ref: openai.types.shared.reasoning.Reasoning
class Reasoning(BaseModelRequest):
    """Configuration options for reasoning models."""

    effort: ReasoningEffort | None = Field(
        default=None,
        description=(
            "Constrains effort on reasoning for reasoning models. Supported values are "
            "`none`, `minimal`, `low`, `medium`, `high`, and `xhigh`."
        ),
    )
    generate_summary: Literal["auto", "concise", "detailed"] | None = Field(
        default=None, description="**Deprecated:** use `summary` instead."
    )
    summary: Literal["auto", "concise", "detailed"] | None = Field(
        default=None,
        description="A summary of the reasoning performed by the model. One of `auto`, `concise`, or `detailed`.",
    )


# ---------------------------------------------------------------------------
# Input content types
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.response_input_text.ResponseInputText
class ResponseInputText(BaseModelRequest):
    """A text input to the model."""

    text: str = Field(description="The text input to the model.")
    type: Literal["input_text"] = Field(
        description="The type of the input item. Always `input_text`."
    )


# Ref: openai.types.responses.response_input_image.ResponseInputImage
class ResponseInputImage(BaseModelRequest):
    """An image input to the model."""

    type: Literal["input_image"] = Field(
        description="The type of the input item. Always `input_image`."
    )
    detail: Literal["low", "high", "auto", "original"] | None = Field(
        default=None,
        description="The detail level of the image to be sent to the model. One of `high`, `low`, `auto`, or `original`. Defaults to `auto`.",
    )
    file_id: str | None = Field(
        default=None, description="The ID of the file to be sent to the model."
    )
    image_url: str | None = Field(
        default=None,
        description="The URL of the image to be sent to the model. A fully qualified URL or base64 encoded image in a data URL.",
    )


# Ref: openai.types.responses.response_input_file.ResponseInputFile
class ResponseInputFile(BaseModelRequest):
    """A file input to the model."""

    type: Literal["input_file"] = Field(
        description="The type of the input item. Always `input_file`."
    )
    file_data: str | None = Field(
        default=None, description="The content of the file to be sent to the model."
    )
    file_id: str | None = Field(
        default=None, description="The ID of the file to be sent to the model."
    )
    file_url: str | None = Field(
        default=None, description="The URL of the file to be sent to the model."
    )
    filename: str | None = Field(
        default=None, description="The name of the file to be sent to the model."
    )


# NOTE: Not in the OpenAI spec's InputContent — this type handles a real-world behaviour
# where Codex (codex-x86_64) sends EasyInputMessage with role="assistant" and content
# blocks of type "output_text" when echoing back conversation history.  The spec-compliant
# path is ResponseOutputMessage (added to ResponseInputItem), but EasyInputMessage matches
# first in the union (both use type="message"), so the content list also needs this type.
# BaseModelRequestWithExtra is used so that extra fields (annotations, logprobs) from the
# original output are accepted without error.
class ResponseOutputTextContent(BaseModelRequestWithExtra):
    """An output_text content block echoed back in the input array (previous assistant response)."""

    type: Literal["output_text"] = Field(
        description="The type of the content block. Always `output_text`."
    )
    text: str = Field(
        description="The text content from the previous assistant response."
    )


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


# ---------------------------------------------------------------------------
# Computer tool call output screenshot  (shared by input and output)
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.response_computer_tool_call_output_screenshot.ResponseComputerToolCallOutputScreenshot
class ResponseComputerToolCallOutputScreenshot(BaseModelRequest):
    """A computer screenshot image used with the computer use tool."""

    type: Literal["computer_screenshot"] = Field(
        description="Always `computer_screenshot`."
    )
    file_id: str | None = Field(
        default=None,
        description="The identifier of an uploaded file that contains the screenshot.",
    )
    image_url: str | None = Field(
        default=None, description="The URL of the screenshot image."
    )


# ---------------------------------------------------------------------------
# Shell call output content  (used by ShellCall input item)
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.response_function_shell_call_output_content.OutcomeTimeout
class ShellCallOutcomeTimeout(BaseModelRequest):
    """Indicates that the shell call exceeded its configured time limit."""

    type: Literal["timeout"] = Field(description="The outcome type. Always `timeout`.")


# Ref: openai.types.responses.response_function_shell_call_output_content.OutcomeExit
class ShellCallOutcomeExit(BaseModelRequest):
    """Indicates that the shell commands finished and returned an exit code."""

    exit_code: int = Field(description="The exit code returned by the shell process.")
    type: Literal["exit"] = Field(description="The outcome type. Always `exit`.")


# Ref: openai.types.responses.response_function_shell_call_output_content.Outcome
ShellCallOutcome = Annotated[
    ShellCallOutcomeTimeout | ShellCallOutcomeExit, Field(discriminator="type")
]


# Ref: openai.types.responses.response_function_shell_call_output_content.ResponseFunctionShellCallOutputContent
class ShellCallOutputContent(BaseModelRequest):
    """Captured stdout and stderr for a portion of a shell tool call output."""

    outcome: ShellCallOutcome = Field(
        description="The exit or timeout outcome associated with this shell call."
    )
    stderr: str = Field(description="Captured stderr output for the shell call.")
    stdout: str = Field(description="Captured stdout output for the shell call.")


# ---------------------------------------------------------------------------
# Apply patch operations  (used by ApplyPatchCall input item)
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.response_input_item.ApplyPatchCallOperationCreateFile
class ApplyPatchOperationCreateFile(BaseModelRequest):
    """Instruction for creating a new file via the apply_patch tool."""

    diff: str = Field(
        description="Unified diff content to apply when creating the file."
    )
    path: str = Field(
        description="Path of the file to create relative to the workspace root."
    )
    type: Literal["create_file"] = Field(
        description="The operation type. Always `create_file`."
    )


# Ref: openai.types.responses.response_input_item.ApplyPatchCallOperationDeleteFile
class ApplyPatchOperationDeleteFile(BaseModelRequest):
    """Instruction for deleting an existing file via the apply_patch tool."""

    path: str = Field(
        description="Path of the file to delete relative to the workspace root."
    )
    type: Literal["delete_file"] = Field(
        description="The operation type. Always `delete_file`."
    )


# Ref: openai.types.responses.response_input_item.ApplyPatchCallOperationUpdateFile
class ApplyPatchOperationUpdateFile(BaseModelRequest):
    """Instruction for updating an existing file via the apply_patch tool."""

    diff: str = Field(description="Unified diff content to apply to the existing file.")
    path: str = Field(
        description="Path of the file to update relative to the workspace root."
    )
    type: Literal["update_file"] = Field(
        description="The operation type. Always `update_file`."
    )


# Ref: openai.types.responses.response_input_item.ApplyPatchCallOperation
ApplyPatchOperation = Annotated[
    ApplyPatchOperationCreateFile
    | ApplyPatchOperationDeleteFile
    | ApplyPatchOperationUpdateFile,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Input items  (used in ResponseCreateParams.input)
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.easy_input_message.EasyInputMessage
class EasyInputMessage(BaseModelRequest):
    """A message input to the model with a role indicating instruction following hierarchy."""

    content: str | ResponseInputMessageContentList = Field(
        description="Text, image, or audio input to the model, used to generate a response. Can also contain previous assistant responses."
    )
    role: Literal["user", "assistant", "system", "developer"] = Field(
        description="The role of the message input. One of `user`, `assistant`, `system`, or `developer`."
    )
    phase: Literal["commentary", "final_answer"] | None = Field(
        default=None,
        description="Labels an `assistant` message as intermediate commentary or the final answer.",
    )
    type: Literal["message"] | None = Field(
        default=None, description="The type of the message input. Always `message`."
    )


# Ref: openai.types.responses.response_input_item.Message
class InputMessage(BaseModelRequest):
    """A message input with a restricted set of roles (no `assistant`)."""

    content: ResponseInputMessageContentList = Field(
        description="A list of one or many input items to the model, containing different content types."
    )
    role: Literal["user", "system", "developer"] = Field(
        description="The role of the message input. One of `user`, `system`, or `developer`."
    )
    status: ResponseItemStatus | None = Field(
        default=None,
        description="The status of item. One of `in_progress`, `completed`, or `incomplete`. Populated when items are returned via API.",
    )
    type: Literal["message"] | None = Field(
        default=None,
        description="The type of the message input. Always set to `message`.",
    )


# Ref: openai.types.responses.response_input_item.ComputerCallOutputAcknowledgedSafetyCheck
class ComputerCallOutputAcknowledgedSafetyCheck(BaseModelRequest):
    """A pending safety check for the computer call."""

    id: str = Field(description="The ID of the pending safety check.")
    code: str | None = Field(
        default=None, description="The type of the pending safety check."
    )
    message: str | None = Field(
        default=None, description="Details about the pending safety check."
    )


# Ref: openai.types.responses.response_input_item.ComputerCallOutput
class ComputerCallOutput(BaseModelRequest):
    """The output of a computer tool call."""

    call_id: str = Field(
        description="The ID of the computer tool call that produced the output."
    )
    output: ResponseComputerToolCallOutputScreenshot = Field(
        description="A computer screenshot image used with the computer use tool."
    )
    type: Literal["computer_call_output"] = Field(
        description="The type of the computer tool call output. Always `computer_call_output`."
    )
    id: str | None = Field(
        default=None, description="The ID of the computer tool call output."
    )
    acknowledged_safety_checks: (
        list[ComputerCallOutputAcknowledgedSafetyCheck] | None
    ) = Field(
        default=None,
        description="The safety checks reported by the API that have been acknowledged by the developer.",
    )
    status: ResponseItemStatus | None = Field(
        default=None,
        description="The status of the message input. One of `in_progress`, `completed`, or `incomplete`. Populated when input items are returned via API.",
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

    command: list[str] = Field(description="The command to run.")
    env: dict[str, str] = Field(
        description="Environment variables to set for the command."
    )
    type: Literal["exec"] = Field(
        description="The type of the local shell action. Always `exec`."
    )
    timeout_ms: int | None = Field(
        default=None, description="Optional timeout in milliseconds for the command."
    )
    user: str | None = Field(
        default=None, description="Optional user to run the command as."
    )
    working_directory: str | None = Field(
        default=None, description="Optional working directory to run the command in."
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
    output: str | None = Field(
        default=None,
        description="Optional human-readable log text from the apply patch tool.",
    )


# Ref: openai.types.responses.response_input_item.McpListToolsTool
class McpListToolsToolItem(BaseModelRequest):
    """A tool available on an MCP server."""

    input_schema: object = Field(
        description="The JSON schema describing the tool's input."
    )
    name: str = Field(description="The name of the tool.")
    annotations: object | None = Field(
        default=None, description="Additional annotations about the tool."
    )
    description: str | None = Field(
        default=None, description="The description of the tool."
    )


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

    id: str = Field(description="The unique ID of the approval request.")
    arguments: str = Field(description="A JSON string of arguments for the tool.")
    name: str = Field(description="The name of the tool to run.")
    server_label: str = Field(
        description="The label of the MCP server making the request."
    )
    type: Literal["mcp_approval_request"] = Field(
        description="The type of the item. Always `mcp_approval_request`."
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

    id: str = Field(description="The unique ID of the tool call.")
    arguments: str = Field(
        description="A JSON string of the arguments passed to the tool."
    )
    name: str = Field(description="The name of the tool that was run.")
    server_label: str = Field(
        description="The label of the MCP server running the tool."
    )
    type: Literal["mcp_call"] = Field(
        description="The type of the item. Always `mcp_call`."
    )
    approval_request_id: str | None = Field(
        default=None,
        description="Unique identifier for the MCP tool call approval request.",
    )
    error: str | None = Field(
        default=None, description="The error from the tool call, if any."
    )
    output: str | None = Field(
        default=None, description="The output from the tool call."
    )
    status: (
        Literal["in_progress", "completed", "incomplete", "calling", "failed"] | None
    ) = Field(default=None, description="The status of the tool call.")


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


# Ref: openai.types.responses.response_input_item.ResponseInputItem
#
# Note: EasyInputMessage and InputMessage share type="message" so a Pydantic
# discriminated union cannot be used here; plain Union is used instead.
ResponseInputItem = (
    EasyInputMessage
    | InputMessage
    | ComputerCallOutput
    | FunctionCallInput
    | FunctionCallOutput
    | ToolSearchCallInput
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
    | CompactionItemParam
    | ItemReference
)

#: The `input` parameter for a response creation request.
# Note: ResponseInputItem is extended further below (after ResponseOutputMessage and
# ResponseReasoningItem are defined) to include those types.
type ResponseInputParam = str | list[ResponseInputItem]


# ---------------------------------------------------------------------------
# Response output content  (model-generated text/refusal with annotations)
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.response_output_text.AnnotationFileCitation
class AnnotationFileCitation(BaseModelResponse):
    """A citation to a file."""

    file_id: str = Field(description="The ID of the file.")
    filename: str = Field(description="The filename of the file cited.")
    index: int = Field(description="The index of the file in the list of files.")
    type: Literal["file_citation"] = Field(
        description="The type of the file citation. Always `file_citation`."
    )


# Ref: openai.types.responses.response_output_text.AnnotationURLCitation
class AnnotationURLCitation(BaseModelResponse):
    """A citation for a web resource used to generate a model response."""

    end_index: int = Field(
        description="The index of the last character of the URL citation in the message."
    )
    start_index: int = Field(
        description="The index of the first character of the URL citation in the message."
    )
    title: str = Field(description="The title of the web resource.")
    type: Literal["url_citation"] = Field(
        description="The type of the URL citation. Always `url_citation`."
    )
    url: str = Field(description="The URL of the web resource.")


# Ref: openai.types.responses.response_output_text.AnnotationContainerFileCitation
class AnnotationContainerFileCitation(BaseModelResponse):
    """A citation for a container file used to generate a model response."""

    container_id: str = Field(description="The ID of the container file.")
    end_index: int = Field(
        description="The index of the last character of the container file citation in the message."
    )
    file_id: str = Field(description="The ID of the file.")
    filename: str = Field(description="The filename of the container file cited.")
    start_index: int = Field(
        description="The index of the first character of the container file citation in the message."
    )
    type: Literal["container_file_citation"] = Field(
        description="The type of the container file citation. Always `container_file_citation`."
    )


# Ref: openai.types.responses.response_output_text.AnnotationFilePath
class AnnotationFilePath(BaseModelResponse):
    """A path to a file."""

    file_id: str = Field(description="The ID of the file.")
    index: int = Field(description="The index of the file in the list of files.")
    type: Literal["file_path"] = Field(
        description="The type of the file path. Always `file_path`."
    )


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

    token: str | None = Field(default=None, description="A possible text token.")
    logprob: float | None = Field(
        default=None, description="The log probability of this token."
    )


# Ref: openai.types.responses.response_output_text.Logprob
class Logprob(BaseModelResponse):
    """The log probability of a token."""

    token: str = Field(description="A possible text token.")
    bytes: list[int] = Field(description="The bytes of this token.")
    logprob: float = Field(description="The log probability of this token.")
    top_logprobs: list[LogprobTopLogprob] = Field(
        description="The log probability of the top most likely tokens."
    )


# Ref: openai.types.responses.response_output_text.ResponseOutputText
class ResponseOutputText(BaseModelResponse):
    """A text output from the model."""

    annotations: list[Annotation] = Field(
        description="The annotations of the text output."
    )
    text: str = Field(description="The text output from the model.")
    type: Literal["output_text"] = Field(
        description="The type of the output text. Always `output_text`."
    )
    logprobs: list[Logprob] | None = Field(
        default=None, description="Log probabilities for the output tokens."
    )


# Ref: openai.types.responses.response_output_refusal.ResponseOutputRefusal
class ResponseOutputRefusal(BaseModelResponse):
    """A refusal from the model."""

    refusal: str = Field(description="The refusal explanation from the model.")
    type: Literal["refusal"] = Field(
        description="The type of the refusal. Always `refusal`."
    )


# Ref: openai.types.responses.response_output_message.Content
ResponseOutputMessageContent = Annotated[
    ResponseOutputText | ResponseOutputRefusal, Field(discriminator="type")
]


# Ref: openai.types.responses.response_output_message.ResponseOutputMessage
class ResponseOutputMessage(BaseModelResponse):
    """An output message from the model."""

    id: str = Field(description="The unique ID of the output message.")
    content: list[ResponseOutputMessageContent] = Field(
        description="The content of the output message."
    )
    role: Literal["assistant"] = Field(
        description="The role of the output message. Always `assistant`."
    )
    status: Literal["in_progress", "completed", "incomplete"] = Field(
        description="The status of the message input."
    )
    type: Literal["message"] = Field(
        description="The type of the output message. Always `message`."
    )
    phase: Literal["commentary", "final_answer"] | None = Field(
        default=None,
        description="Labels an `assistant` message as intermediate commentary or the final answer.",
    )


# ---------------------------------------------------------------------------
# Response function tool call
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.response_function_tool_call.ResponseFunctionToolCall
class ResponseFunctionToolCall(BaseModelResponse):
    """A tool call to run a function."""

    arguments: str = Field(
        description="A JSON string of the arguments to pass to the function."
    )
    call_id: str = Field(
        description="The unique ID of the function tool call generated by the model."
    )
    name: str = Field(description="The name of the function to run.")
    type: Literal["function_call"] = Field(
        description="The type of the function tool call. Always `function_call`."
    )
    id: str | None = Field(
        default=None, description="The unique ID of the function tool call."
    )
    namespace: str | None = Field(
        default=None, description="The namespace of the function to run."
    )
    status: ResponseItemStatus | None = Field(
        default=None,
        description="The status of the item. One of `in_progress`, `completed`, or `incomplete`. Populated when items are returned via API.",
    )


# ---------------------------------------------------------------------------
# Response file search tool call
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.response_file_search_tool_call.Result
class FileSearchResult(BaseModelResponse):
    """A file search result."""

    attributes: dict[str, str | float | bool] | None = Field(
        default=None,
        description="Set of 16 key-value pairs that can be attached to an object.",
    )
    file_id: str | None = Field(default=None, description="The unique ID of the file.")
    filename: str | None = Field(default=None, description="The name of the file.")
    score: float | None = Field(
        default=None, description="The relevance score of the file, between 0 and 1."
    )
    text: str | None = Field(
        default=None, description="The text that was retrieved from the file."
    )


# Ref: openai.types.responses.response_file_search_tool_call.ResponseFileSearchToolCall
class ResponseFileSearchToolCall(BaseModelResponse):
    """The results of a file search tool call."""

    id: str = Field(description="The unique ID of the file search tool call.")
    queries: list[str] = Field(description="The queries used to search for files.")
    status: Literal["in_progress", "searching", "completed", "incomplete", "failed"] = (
        Field(description="The status of the file search tool call.")
    )
    type: Literal["file_search_call"] = Field(
        description="The type of the file search tool call. Always `file_search_call`."
    )
    results: list[FileSearchResult] | None = Field(
        default=None, description="The results of the file search tool call."
    )


# ---------------------------------------------------------------------------
# Response web search tool call
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.response_function_web_search.ActionSearchSource
class WebSearchActionSource(BaseModelResponse):
    """A source used in the search."""

    type: Literal["url"] = Field(description="The type of source. Always `url`.")
    url: str = Field(description="The URL of the source.")


# Ref: openai.types.responses.response_function_web_search.ActionSearch
class WebSearchActionSearch(BaseModelResponse):
    """Web search action of type `search`."""

    query: str = Field(description="[DEPRECATED] The search query.")
    type: Literal["search"] = Field(description="The action type.")
    queries: list[str] | None = Field(default=None, description="The search queries.")
    sources: list[WebSearchActionSource] | None = Field(
        default=None, description="The sources used in the search."
    )


# Ref: openai.types.responses.response_function_web_search.ActionOpenPage
class WebSearchActionOpenPage(BaseModelResponse):
    """Web search action of type `open_page`."""

    type: Literal["open_page"] = Field(description="The action type.")
    url: str | None = Field(default=None, description="The URL opened by the model.")


# Ref: openai.types.responses.response_function_web_search.ActionFind
class WebSearchActionFind(BaseModelResponse):
    """Web search action of type `find_in_page`."""

    pattern: str = Field(
        description="The pattern or text to search for within the page."
    )
    type: Literal["find_in_page"] = Field(description="The action type.")
    url: str = Field(description="The URL of the page searched for the pattern.")


# Ref: openai.types.responses.response_function_web_search.Action
WebSearchAction = Annotated[
    WebSearchActionSearch | WebSearchActionOpenPage | WebSearchActionFind,
    Field(discriminator="type"),
]


# Ref: openai.types.responses.response_function_web_search.ResponseFunctionWebSearch
class ResponseFunctionWebSearch(BaseModelResponse):
    """The results of a web search tool call."""

    id: str = Field(description="The unique ID of the web search tool call.")
    action: WebSearchAction = Field(
        description="An object describing the specific action taken in this web search call."
    )
    status: Literal["in_progress", "searching", "completed", "failed"] = Field(
        description="The status of the web search tool call."
    )
    type: Literal["web_search_call"] = Field(
        description="The type of the web search tool call. Always `web_search_call`."
    )


# ---------------------------------------------------------------------------
# Response computer tool call  (with all actions)
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.response_computer_tool_call.PendingSafetyCheck
class PendingSafetyCheck(BaseModelResponse):
    """A pending safety check for the computer call."""

    id: str = Field(description="The ID of the pending safety check.")
    code: str | None = Field(
        default=None, description="The type of the pending safety check."
    )
    message: str | None = Field(
        default=None, description="Details about the pending safety check."
    )


# Ref: openai.types.responses.response_computer_tool_call.ActionClick
class ComputerActionClick(BaseModelResponse):
    """A click action."""

    button: Literal["left", "right", "wheel", "back", "forward"] = Field(
        description="Indicates which mouse button was pressed."
    )
    type: Literal["click"] = Field(
        description="Specifies the event type. Always `click`."
    )
    x: int = Field(description="The x-coordinate where the click occurred.")
    y: int = Field(description="The y-coordinate where the click occurred.")
    keys: list[str] | None = Field(
        default=None, description="The keys being held while clicking."
    )


# Ref: openai.types.responses.response_computer_tool_call.ActionDoubleClick
class ComputerActionDoubleClick(BaseModelResponse):
    """A double click action."""

    type: Literal["double_click"] = Field(
        description="Specifies the event type. Always `double_click`."
    )
    x: int = Field(description="The x-coordinate where the double click occurred.")
    y: int = Field(description="The y-coordinate where the double click occurred.")
    keys: list[str] | None = Field(
        default=None, description="The keys being held while double-clicking."
    )


# Ref: openai.types.responses.response_computer_tool_call.ActionDragPath
class ComputerActionDragPath(BaseModelResponse):
    """An x/y coordinate pair."""

    x: int = Field(description="The x-coordinate.")
    y: int = Field(description="The y-coordinate.")


# Ref: openai.types.responses.response_computer_tool_call.ActionDrag
class ComputerActionDrag(BaseModelResponse):
    """A drag action."""

    path: list[ComputerActionDragPath] = Field(
        description="An array of coordinates representing the path of the drag action."
    )
    type: Literal["drag"] = Field(
        description="Specifies the event type. Always `drag`."
    )
    keys: list[str] | None = Field(
        default=None, description="The keys being held while dragging."
    )


# Ref: openai.types.responses.response_computer_tool_call.ActionKeypress
class ComputerActionKeypress(BaseModelResponse):
    """A collection of keypresses."""

    keys: list[str] = Field(
        description="The combination of keys the model is requesting to be pressed."
    )
    type: Literal["keypress"] = Field(
        description="Specifies the event type. Always `keypress`."
    )


# Ref: openai.types.responses.response_computer_tool_call.ActionMove
class ComputerActionMove(BaseModelResponse):
    """A mouse move action."""

    type: Literal["move"] = Field(
        description="Specifies the event type. Always `move`."
    )
    x: int = Field(description="The x-coordinate to move to.")
    y: int = Field(description="The y-coordinate to move to.")
    keys: list[str] | None = Field(
        default=None, description="The keys being held while moving."
    )


# Ref: openai.types.responses.response_computer_tool_call.ActionScreenshot
class ComputerActionScreenshot(BaseModelResponse):
    """A screenshot action."""

    type: Literal["screenshot"] = Field(
        description="Specifies the event type. Always `screenshot`."
    )


# Ref: openai.types.responses.response_computer_tool_call.ActionScroll
class ComputerActionScroll(BaseModelResponse):
    """A scroll action."""

    scroll_x: int = Field(description="The horizontal scroll distance.")
    scroll_y: int = Field(description="The vertical scroll distance.")
    type: Literal["scroll"] = Field(
        description="Specifies the event type. Always `scroll`."
    )
    x: int = Field(description="The x-coordinate where the scroll occurred.")
    y: int = Field(description="The y-coordinate where the scroll occurred.")
    keys: list[str] | None = Field(
        default=None, description="The keys being held while scrolling."
    )


# Ref: openai.types.responses.response_computer_tool_call.ActionType
class ComputerActionType(BaseModelResponse):
    """An action to type in text."""

    text: str = Field(description="The text to type.")
    type: Literal["type"] = Field(
        description="Specifies the event type. Always `type`."
    )


# Ref: openai.types.responses.response_computer_tool_call.ActionWait
class ComputerActionWait(BaseModelResponse):
    """A wait action."""

    type: Literal["wait"] = Field(
        description="Specifies the event type. Always `wait`."
    )


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

    id: str = Field(description="The unique ID of the computer call.")
    call_id: str = Field(
        description="An identifier used when responding to the tool call with output."
    )
    pending_safety_checks: list[PendingSafetyCheck] = Field(
        description="The pending safety checks for the computer call."
    )
    status: Literal["in_progress", "completed", "incomplete"] = Field(
        description="The status of the item."
    )
    type: Literal["computer_call"] = Field(
        description="The type of the computer call. Always `computer_call`."
    )
    action: ComputerAction | None = Field(
        default=None, description="The action to perform."
    )
    actions: ComputerActionList | None = Field(
        default=None, description="Flattened batched actions for `computer_use`."
    )


# Ref: openai.types.responses.response_computer_tool_call_output_item.AcknowledgedSafetyCheck
class AcknowledgedSafetyCheck(BaseModelResponse):
    """An acknowledged safety check for a computer call output."""

    id: str = Field(description="The ID of the pending safety check.")
    code: str | None = Field(
        default=None, description="The type of the pending safety check."
    )
    message: str | None = Field(
        default=None, description="Details about the pending safety check."
    )


# Ref: openai.types.responses.response_computer_tool_call_output_item.ResponseComputerToolCallOutputItem
class ResponseComputerToolCallOutputItem(BaseModelResponse):
    """The output of a computer tool call."""

    id: str = Field(description="The unique ID of the computer call tool output.")
    call_id: str = Field(
        description="The ID of the computer tool call that produced the output."
    )
    output: ResponseComputerToolCallOutputScreenshot = Field(
        description="A computer screenshot image used with the computer use tool."
    )
    status: Literal["completed", "incomplete", "failed", "in_progress"] = Field(
        description="The status of the message input."
    )
    type: Literal["computer_call_output"] = Field(
        description="The type of the computer tool call output. Always `computer_call_output`."
    )
    acknowledged_safety_checks: list[AcknowledgedSafetyCheck] | None = Field(
        default=None,
        description="The safety checks reported by the API that have been acknowledged by the developer.",
    )
    created_by: str | None = Field(
        default=None, description="The identifier of the actor that created the item."
    )


# ---------------------------------------------------------------------------
# Response code interpreter tool call
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.response_code_interpreter_tool_call.OutputLogs
class CodeInterpreterOutputLogs(BaseModelResponse):
    """The logs output from the code interpreter."""

    logs: str = Field(description="The logs output from the code interpreter.")
    type: Literal["logs"] = Field(description="The type of the output. Always `logs`.")


# Ref: openai.types.responses.response_code_interpreter_tool_call.OutputImage
class CodeInterpreterOutputImage(BaseModelResponse):
    """The image output from the code interpreter."""

    type: Literal["image"] = Field(
        description="The type of the output. Always `image`."
    )
    url: str = Field(
        description="The URL of the image output from the code interpreter."
    )


# Ref: openai.types.responses.response_code_interpreter_tool_call.Output
CodeInterpreterOutput = Annotated[
    CodeInterpreterOutputLogs | CodeInterpreterOutputImage, Field(discriminator="type")
]


# Ref: openai.types.responses.response_code_interpreter_tool_call.ResponseCodeInterpreterToolCall
class ResponseCodeInterpreterToolCall(BaseModelResponse):
    """A tool call to run code."""

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
    outputs: list[CodeInterpreterOutput] | None = Field(
        default=None,
        description="The outputs generated by the code interpreter, such as logs or images.",
    )


# ---------------------------------------------------------------------------
# Response reasoning item
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.response_reasoning_item.Summary
class ReasoningItemSummary(BaseModelResponse):
    """A summary text from the model."""

    text: str = Field(
        description="A summary of the reasoning output from the model so far."
    )
    type: Literal["summary_text"] = Field(
        description="The type of the object. Always `summary_text`."
    )


# Ref: openai.types.responses.response_reasoning_item.Content
class ReasoningItemContent(BaseModelResponse):
    """Reasoning text from the model."""

    text: str = Field(description="The reasoning text from the model.")
    type: Literal["reasoning_text"] = Field(
        description="The type of the reasoning text. Always `reasoning_text`."
    )


# Ref: openai.types.responses.response_reasoning_item.ResponseReasoningItem
class ResponseReasoningItem(BaseModelResponse):
    """A description of the chain of thought used by a reasoning model while generating a response."""

    id: str = Field(description="The unique identifier of the reasoning content.")
    summary: list[ReasoningItemSummary] = Field(
        description="Reasoning summary content."
    )
    type: Literal["reasoning"] = Field(
        description="The type of the object. Always `reasoning`."
    )
    content: list[ReasoningItemContent] | None = Field(
        default=None, description="Reasoning text content."
    )
    encrypted_content: str | None = Field(
        default=None,
        description="The encrypted content of the reasoning item, populated when `reasoning.encrypted_content` is included.",
    )
    status: ResponseItemStatus | None = Field(
        default=None,
        description="The status of the item. One of `in_progress`, `completed`, or `incomplete`. Populated when items are returned via API.",
    )


# Extend ResponseInputItem with types defined after the initial alias.
# ResponseOutputMessage (role=assistant, content=[output_text/refusal]) and
# ResponseReasoningItem (type="reasoning") are valid input items per the SDK
# (openai.types.responses.response_input_item.ResponseInputItem) and may appear
# when a client echoes back a full previous response as conversation history.
#
# IMPORTANT: do NOT use the `type` statement here. `type X = X | ...` creates a
# lazy TypeAliasType whose __value__ is evaluated after the name is rebound, so
# `X` on the right side would refer to the NEW alias (itself) — a circular
# reference that breaks Pydantic schema building.  Capturing the original binding
# in `_ResponseInputItemBase` first makes the assignment evaluate immediately to a
# plain UnionType with no self-reference.
_ResponseInputItemBase = ResponseInputItem
ResponseInputItem = (  # type: ignore[misc]
    _ResponseInputItemBase | ResponseOutputMessage | ResponseReasoningItem
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

    id: str = Field(description="The unique ID of the apply patch tool call.")
    call_id: str = Field(
        description="The unique ID of the apply patch tool call generated by the model."
    )
    operation: ResponseApplyPatchOperation = Field(
        description="The patch operation to apply."
    )
    status: Literal["in_progress", "completed"] = Field(
        description="The status of the apply patch tool call."
    )
    type: Literal["apply_patch_call"] = Field(
        description="The type of the item. Always `apply_patch_call`."
    )
    created_by: str | None = Field(
        default=None, description="The ID of the entity that created this tool call."
    )


# Ref: openai.types.responses.response_apply_patch_tool_call_output.ResponseApplyPatchToolCallOutput
class ResponseApplyPatchToolCallOutput(BaseModelResponse):
    """The output emitted by an apply patch tool call."""

    id: str = Field(description="The unique ID of the apply patch tool call output.")
    call_id: str = Field(
        description="The unique ID of the apply patch tool call generated by the model."
    )
    status: Literal["completed", "failed"] = Field(
        description="The status of the apply patch tool call output."
    )
    type: Literal["apply_patch_call_output"] = Field(
        description="The type of the item. Always `apply_patch_call_output`."
    )
    created_by: str | None = Field(
        default=None,
        description="The ID of the entity that created this tool call output.",
    )
    output: str | None = Field(
        default=None,
        description="Optional textual output returned by the apply patch tool.",
    )


# Ref: openai.types.responses.response_compaction_item.ResponseCompactionItem
class ResponseCompactionItem(BaseModelResponse):
    """A compaction item generated by the v1/responses/compact API."""

    id: str = Field(description="The unique ID of the compaction item.")
    encrypted_content: str = Field(
        description="The encrypted content that was produced by compaction."
    )
    type: Literal["compaction"] = Field(
        description="The type of the item. Always `compaction`."
    )
    created_by: str | None = Field(
        default=None, description="The identifier of the actor that created the item."
    )


# Ref: openai.types.responses.response_tool_search_call.ResponseToolSearchCall
class ResponseToolSearchCall(BaseModelResponse):
    """A tool search call item."""

    id: str = Field(description="The unique ID of the tool search call item.")
    arguments: JsonMapping = Field(
        description="Arguments used for the tool search call."
    )
    execution: Literal["server", "client"] = Field(
        description="Whether tool search was executed by the server or by the client."
    )
    status: ResponseItemStatus = Field(
        description="The status of the tool search call item that was recorded."
    )
    type: Literal["tool_search_call"] = Field(
        description="The type of the item. Always `tool_search_call`."
    )
    call_id: str | None = Field(
        default=None,
        description="The unique ID of the tool search call generated by the model.",
    )
    created_by: str | None = Field(
        default=None, description="The identifier of the actor that created the item."
    )


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
    created_by: str | None = Field(
        default=None, description="The identifier of the actor that created the item."
    )


# ---------------------------------------------------------------------------
# Shell / local shell output items  (response-side)
# ---------------------------------------------------------------------------


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
    created_by: str | None = Field(
        default=None, description="The ID of the entity that created this tool call."
    )


# Ref: openai.types.responses.response_function_shell_tool_call_output.OutputOutcomeTimeout
class ShellToolCallOutputOutcomeTimeout(BaseModelResponse):
    """Indicates that the shell call exceeded its configured time limit."""

    type: Literal["timeout"] = Field(description="The outcome type. Always `timeout`.")


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
    created_by: str | None = Field(
        default=None, description="The identifier of the actor that created the item."
    )


# ---------------------------------------------------------------------------
# Image generation, local shell, MCP output items
# ---------------------------------------------------------------------------


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

    command: list[str] = Field(description="The command to run.")
    env: dict[str, str] = Field(
        description="Environment variables to set for the command."
    )
    type: Literal["exec"] = Field(
        description="The type of the local shell action. Always `exec`."
    )
    timeout_ms: int | None = Field(
        default=None, description="Optional timeout in milliseconds for the command."
    )
    user: str | None = Field(
        default=None, description="Optional user to run the command as."
    )
    working_directory: str | None = Field(
        default=None, description="Optional working directory to run the command in."
    )


# Ref: openai.types.responses.response_output_item.LocalShellCall
class LocalShellCall(BaseModelResponse):
    """A tool call to run a command on the local shell."""

    id: str = Field(description="The unique ID of the local shell call.")
    action: LocalShellCallAction = Field(
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


# Ref: openai.types.responses.response_output_item.LocalShellCallOutput
class LocalShellCallOutput(BaseModelResponse):
    """The output of a local shell tool call."""

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
        default=None, description="The status of the item."
    )


# Ref: openai.types.responses.response_output_item.McpListToolsTool
class McpListToolsToolOutput(BaseModelResponse):
    """A tool available on an MCP server."""

    input_schema: object = Field(
        description="The JSON schema describing the tool's input."
    )
    name: str = Field(description="The name of the tool.")
    annotations: object | None = Field(
        default=None, description="Additional annotations about the tool."
    )
    description: str | None = Field(
        default=None, description="The description of the tool."
    )


# Ref: openai.types.responses.response_output_item.McpListTools
class McpListTools(BaseModelResponse):
    """A list of tools available on an MCP server."""

    id: str = Field(description="The unique ID of the list.")
    server_label: str = Field(description="The label of the MCP server.")
    tools: list[McpListToolsToolOutput] = Field(
        description="The tools available on the server."
    )
    type: Literal["mcp_list_tools"] = Field(
        description="The type of the item. Always `mcp_list_tools`."
    )
    error: str | None = Field(
        default=None, description="Error message if the server could not list tools."
    )


# Ref: openai.types.responses.response_output_item.McpApprovalRequest
class McpApprovalRequest(BaseModelResponse):
    """A request for human approval of a tool invocation."""

    id: str = Field(description="The unique ID of the approval request.")
    arguments: str = Field(description="A JSON string of arguments for the tool.")
    name: str = Field(description="The name of the tool to run.")
    server_label: str = Field(
        description="The label of the MCP server making the request."
    )
    type: Literal["mcp_approval_request"] = Field(
        description="The type of the item. Always `mcp_approval_request`."
    )


# Ref: openai.types.responses.response_output_item.McpApprovalResponse
class McpApprovalResponseOutput(BaseModelResponse):
    """A response to an MCP approval request."""

    id: str = Field(description="The unique ID of the approval response.")
    approval_request_id: str = Field(
        description="The ID of the approval request being answered."
    )
    approve: bool = Field(description="Whether the request was approved.")
    type: Literal["mcp_approval_response"] = Field(
        description="The type of the item. Always `mcp_approval_response`."
    )
    reason: str | None = Field(
        default=None, description="Optional reason for the decision."
    )


# Ref: openai.types.responses.response_output_item.McpCall
class McpCall(BaseModelResponse):
    """An invocation of a tool on an MCP server."""

    id: str = Field(description="The unique ID of the tool call.")
    arguments: str = Field(
        description="A JSON string of the arguments passed to the tool."
    )
    name: str = Field(description="The name of the tool that was run.")
    server_label: str = Field(
        description="The label of the MCP server running the tool."
    )
    type: Literal["mcp_call"] = Field(
        description="The type of the item. Always `mcp_call`."
    )
    approval_request_id: str | None = Field(
        default=None,
        description="Unique identifier for the MCP tool call approval request.",
    )
    error: str | None = Field(
        default=None, description="The error from the tool call, if any."
    )
    output: str | None = Field(
        default=None, description="The output from the tool call."
    )
    status: (
        Literal["in_progress", "completed", "incomplete", "calling", "failed"] | None
    ) = Field(default=None, description="The status of the tool call.")


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
    | ResponseCustomToolCallOutputItem,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Usage types
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.response_usage.InputTokensDetails
class InputTokensDetails(BaseModelResponse):
    """A detailed breakdown of the input tokens."""

    cached_tokens: int = Field(
        description="The number of tokens that were retrieved from the cache."
    )


# Ref: openai.types.responses.response_usage.OutputTokensDetails
class OutputTokensDetails(BaseModelResponse):
    """A detailed breakdown of the output tokens."""

    reasoning_tokens: int = Field(description="The number of reasoning tokens.")


# Ref: openai.types.responses.response_usage.ResponseUsage
class ResponseUsage(BaseModelResponse):
    """Token usage details for a response, including input, output, and total counts."""

    input_tokens: int = Field(description="The number of input tokens.")
    input_tokens_details: InputTokensDetails = Field(
        description="A detailed breakdown of the input tokens."
    )
    output_tokens: int = Field(description="The number of output tokens.")
    output_tokens_details: OutputTokensDetails = Field(
        description="A detailed breakdown of the output tokens."
    )
    total_tokens: int = Field(description="The total number of tokens used.")


# ---------------------------------------------------------------------------
# Text format config types
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.response_format_text_json_schema_config.ResponseFormatTextJSONSchemaConfig
class ResponseFormatTextJSONSchemaConfig(BaseModelRequest):
    """JSON Schema response format for generating structured JSON responses."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(
        description="The name of the response format. Must be a-z, A-Z, 0-9, or contain underscores and dashes, with a maximum length of 64."
    )
    schema_: JsonMapping = Field(
        alias="schema",
        serialization_alias="schema",
        description="The schema for the response format, described as a JSON Schema object.",
    )
    type: Literal["json_schema"] = Field(
        description="The type of response format. Always `json_schema`."
    )
    description: str | None = Field(
        default=None,
        description="A description of what the response format is for, used by the model to determine how to respond.",
    )
    strict: bool | None = Field(
        default=None,
        description="Whether to enable strict schema adherence when generating the output.",
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


# ---------------------------------------------------------------------------
# Prompt types
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Error / incomplete details
# ---------------------------------------------------------------------------


#: Valid error code values for a response error.
ResponseErrorCode = Literal[
    "server_error",
    "rate_limit_exceeded",
    "invalid_prompt",
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

    code: ResponseErrorCode = Field(description="The error code for the response.")
    message: str = Field(description="A human-readable description of the error.")


# Ref: openai.types.responses.response.IncompleteDetails
class IncompleteDetails(BaseModelResponse):
    """Details about why the response is incomplete."""

    reason: Literal["max_output_tokens", "content_filter"] | None = Field(
        default=None, description="The reason why the response is incomplete."
    )


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.response.Conversation
class Conversation(BaseModelResponse):
    """The conversation that this response belonged to."""

    id: str = Field(
        description="The unique ID of the conversation that this response was associated with."
    )


# ---------------------------------------------------------------------------
# Main Response object
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.response.Response
class Response(BaseModelResponse):
    """A model response from the Responses API."""

    id: str = Field(description="Unique identifier for this Response.")
    created_at: float = Field(
        description="Unix timestamp (in seconds) of when this Response was created."
    )
    error: ResponseError | None = Field(
        default=None,
        description="An error object returned when the model fails to generate a Response.",
    )
    incomplete_details: IncompleteDetails | None = Field(
        default=None, description="Details about why the response is incomplete."
    )
    instructions: str | list[ResponseInputItem] | None = Field(
        default=None,
        description="A system (or developer) message inserted into the model's context.",
    )
    metadata: Metadata | None = Field(
        default=None,
        description="Set of 16 key-value pairs that can be attached to an object.",
    )
    model: str = Field(description="Model ID used to generate the response.")
    object: Literal["response"] = Field(
        description="The object type of this resource. Always `response`."
    )
    output: list[ResponseOutputItem] = Field(
        description="An array of content items generated by the model."
    )
    parallel_tool_calls: bool = Field(
        description="Whether to allow the model to run tool calls in parallel."
    )
    temperature: float | None = Field(
        default=None,
        description="Sampling temperature to use, between 0 and 2. Higher values make output more random.",
    )
    tool_choice: ToolChoice = Field(
        description="How the model should select which tool to use when generating a response."
    )
    tools: list[Tool] = Field(
        description="An array of tools the model may call while generating a response."
    )
    top_p: float | None = Field(
        default=None,
        description="Nucleus sampling parameter. Consider tokens with top_p probability mass.",
    )
    background: bool | None = Field(
        default=None, description="Whether to run the model response in the background."
    )
    completed_at: float | None = Field(
        default=None,
        description="Unix timestamp (in seconds) of when this Response was completed.",
    )
    conversation: Conversation | None = Field(
        default=None, description="The conversation that this response belonged to."
    )
    max_output_tokens: int | None = Field(
        default=None,
        description="An upper bound for the number of tokens that can be generated for a response.",
    )
    max_tool_calls: int | None = Field(
        default=None,
        description="The maximum number of total calls to built-in tools that can be processed in a response.",
    )
    previous_response_id: str | None = Field(
        default=None,
        description="The unique ID of the previous response to the model. Use to create multi-turn conversations.",
    )
    prompt: ResponsePrompt | None = Field(
        default=None, description="Reference to a prompt template and its variables."
    )
    prompt_cache_key: str | None = Field(
        default=None,
        description="Used to cache responses for similar requests to optimize cache hit rates.",
    )
    prompt_cache_retention: PromptCacheRetention | None = Field(
        default=None, description="The retention policy for the prompt cache."
    )
    reasoning: Reasoning | None = Field(
        default=None, description="Configuration options for reasoning models."
    )
    safety_identifier: str | None = Field(
        default=None,
        description="A stable identifier to help detect users violating usage policies.",
    )
    service_tier: ServiceTiers | None = Field(
        default=None,
        description="Specifies the processing type used for serving the request.",
    )
    status: ResponseStatus | None = Field(
        default=None, description="The status of the response generation."
    )
    text: ResponseTextConfig | None = Field(
        default=None,
        description="Configuration options for a text response from the model.",
    )
    top_logprobs: int | None = Field(
        default=None,
        description="An integer between 0 and 20 specifying the number of most likely tokens to return at each token position.",
    )
    truncation: Literal["auto", "disabled"] | None = Field(
        default=None,
        description="The truncation strategy to use for the model response.",
    )
    usage: ResponseUsage | None = Field(
        default=None,
        description="Token usage details including input tokens, output tokens, and total count.",
    )
    user: str | None = Field(
        default=None,
        description="Stable identifier for end-users. Replaced by `safety_identifier` and `prompt_cache_key`.",
    )


# ---------------------------------------------------------------------------
# Stream event helper types
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.response_content_part_added_event.PartReasoningText
class ContentPartReasoningText(BaseModelResponse):
    """Reasoning text part within a stream content-part event."""

    text: str = Field(description="The reasoning text from the model.")
    type: Literal["reasoning_text"] = Field(
        description="The type of the reasoning text. Always `reasoning_text`."
    )


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


# ---------------------------------------------------------------------------
# Stream events — response lifecycle
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stream events — output items
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stream events — content parts
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stream events — text deltas
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stream events — refusal
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stream events — function call arguments
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stream events — audio
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stream events — web search
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stream events — file search
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stream events — code interpreter
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stream events — reasoning
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stream events — image generation
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stream events — MCP
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Stream events — custom tool call
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ResponseStreamEvent union
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ResponseInputMessageItem  (items API response)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ResponseItem union  (items returned via /v1/responses/{id}/input_items)
# ---------------------------------------------------------------------------

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
)


# ---------------------------------------------------------------------------
# ResponseItemList  (paginated list from /v1/responses/{id}/input_items)
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.response_item_list.ResponseItemList
class ResponseItemList(BaseModelResponse):
    """A paginated list of Response items."""

    data: list[ResponseItem] = Field(
        description="A list of items used to generate this response."
    )
    first_id: str = Field(description="The ID of the first item in the list.")
    has_more: bool = Field(description="Whether there are more items available.")
    last_id: str = Field(description="The ID of the last item in the list.")
    object: Literal["list"] = Field(
        description="The type of object returned. Always `list`."
    )


# ---------------------------------------------------------------------------
# ResponseCreateParams helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# ResponseCreateParams  (request body for POST /v1/responses)
# ---------------------------------------------------------------------------


# Ref: openai.types.responses.response_conversation_param_param.ResponseConversationParamParam
class ConversationObject(BaseModelRequest):
    """A conversation reference passed as an object with an ID."""

    id: str = Field(description="The unique ID of the conversation.")


#: Conversation parameter: either a conversation ID string or an object with an `id` field.
ConversationParam = str | ConversationObject | None


# Ref: openai.types.responses.response_create_params.ResponseCreateParamsBase
class ResponseCreateParams(BaseModelRequest):
    """Request body for POST /v1/responses."""

    model: str = Field(description="Model ID used to generate the response.")
    input: str | ResponseInputParam | None = Field(
        default=None,
        description="Text, image, or file inputs to the model, used to generate a response.",
    )
    background: bool | None = Field(
        default=None,
        description="Whether to run the model response in the background.\nUNSUPPORTED on this implementation.",
    )
    context_management: list[ContextManagement] | None = Field(
        default=None,
        description="Context management configuration for this request.\nUNSUPPORTED on this implementation.",
    )
    conversation: ConversationParam = Field(
        default=None,
        description="The conversation that this response belongs to. Cannot be used with `previous_response_id`.\nUNSUPPORTED on this implementation.",
    )
    include: list[ResponseIncludable] | None = Field(
        default=None,
        description="Specify additional output data to include in the model response.",
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
    parallel_tool_calls: bool | None = Field(
        default=None,
        description="Whether to allow the model to run tool calls in parallel.",
    )
    previous_response_id: str | None = Field(
        default=None,
        description="The unique ID of the previous response. Use to create multi-turn conversations.",
    )
    prompt: ResponsePrompt | None = Field(
        default=None,
        description="Reference to a prompt template and its variables.\nUNSUPPORTED on this implementation.",
    )
    prompt_cache_key: str | None = Field(
        default=None,
        description="Used to cache responses for similar requests to optimize cache hit rates.",
    )
    prompt_cache_retention: PromptCacheRetention | None = Field(
        default=None, description="The retention policy for the prompt cache."
    )
    reasoning: Reasoning | None = Field(
        default=None, description="Configuration options for reasoning models."
    )
    safety_identifier: str | None = Field(
        default=None,
        description="A stable identifier to help detect users violating usage policies.\nUNSUPPORTED on this implementation.",
    )
    service_tier: ServiceTiers | None = Field(
        default=None,
        description="Specifies the processing type used for serving the request.",
    )
    store: bool | None = Field(
        default=None,
        description="Whether to store the generated model response for later retrieval via API.\nUNSUPPORTED on this implementation.",
    )
    stream: bool | None = Field(
        default=None,
        description="If true, the model response data will be streamed to the client as it is generated.",
    )
    stream_options: StreamOptions | None = Field(
        default=None,
        description="Options for streaming responses. Only set this when you set `stream: true`.\nUNSUPPORTED on this implementation.",
    )
    temperature: float | None = Field(
        default=None,
        ge=0,
        le=2,
        description="Sampling temperature to use, between 0 and 2. Higher values make output more random.",
    )
    text: ResponseTextConfig | None = Field(
        default=None,
        description="Configuration options for a text response from the model.",
    )
    tool_choice: ToolChoice | None = Field(
        default=None,
        description="How the model should select which tool to use when generating a response.",
    )
    tools: list[Tool] | None = Field(
        default=None,
        description="An array of tools the model may call while generating a response.",
    )
    top_logprobs: int | None = Field(
        default=None,
        ge=0,
        le=20,
        description="An integer between 0 and 20 specifying the number of most likely tokens to return at each token position.",
    )
    top_p: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Nucleus sampling parameter. Consider tokens with top_p probability mass.",
    )
    truncation: Literal["auto", "disabled"] | None = Field(
        default=None,
        description="The truncation strategy to use for the model response.\nUNSUPPORTED on this implementation.",
    )
    user: str | None = Field(
        default=None,
        description="Stable identifier for end-users. Replaced by `safety_identifier` and `prompt_cache_key`.",
    )

    # Extra validations
    _UNSUPPORTED: ClassVar[set[str]] = {
        # Ignored silently: "background", "store"
        "context_management",
        "conversation",
        "max_tool_calls",
        "prompt",
        "safety_identifier",
        "stream_options",
        "truncation",
    }

    @model_validator(mode="after")
    def _unsupported(self) -> Self:
        """Validate that unsupported parameters are not used.

        Raises:
            ValueError: If an unsupported tool type is present in ``tools``.
            UnsupportedParameterError: If a parameter marked as unsupported is used.
        """
        _unsupported_tools = (
            FileSearchTool,
            ComputerTool,
            ComputerUsePreviewTool,
            Mcp,
            LocalShell,
            FunctionShellTool,
            CustomTool,
            NamespaceTool,
            ToolSearchTool,
            ApplyPatchTool,
        )
        if any(isinstance(tool, _unsupported_tools) for tool in (self.tools or [])):
            tool_names = " ,".join(
                t.__class__.__name__
                for t in (self.tools or [])
                if isinstance(t, _unsupported_tools)
            )
            msg = f"Unsupported tool type(s): {tool_names}."
            raise ValueError(msg)
        for key in self._UNSUPPORTED & self.model_fields_set:
            raise UnsupportedParameterError(key)
        return self


# Ref: openai.types.responses.input_token_count_params.InputTokenCountParams
class InputTokenCountParams(BaseModelRequest):
    """Request body for POST /v1/responses/input_tokens.

    Counts input tokens without producing a response.
    """

    model: str = Field(description="Model ID used to generate the response.")
    input: str | ResponseInputParam | None = Field(
        default=None,
        description="Text, image, or file inputs to the model, used to generate a response",
    )
    instructions: str | None = Field(
        default=None,
        description="A system (or developer) message inserted into the model's context. "
        "When used along with `previous_response_id`, the instructions from a previous "
        "response will not be carried over to the next response.",
    )
    tools: list[Tool] | None = Field(
        default=None,
        description="An array of tools the model may call while generating a response. "
        "You can specify which tool to use by setting the `tool_choice` parameter.",
    )
    tool_choice: ToolChoice | None = Field(
        default=None, description="Controls which tool the model should use, if any."
    )
    parallel_tool_calls: bool | None = Field(
        default=None,
        description="Whether to allow the model to run tool calls in parallel.",
    )
    reasoning: Reasoning | None = Field(
        default=None, description="Configuration options for reasoning models."
    )
    text: ResponseTextConfig | None = Field(
        default=None,
        description="Configuration options for a text response from the model.\n"
        "UNSUPPORTED on this implementation.",
    )
    truncation: Literal["auto", "disabled"] | None = Field(
        default=None,
        description="The truncation strategy to use for the model response.\n"
        "UNSUPPORTED on this implementation.",
    )
    previous_response_id: str | None = Field(
        default=None,
        description="The unique ID of the previous response to the model. Use to create "
        "multi-turn conversations. Cannot be used with `conversation`.\n"
        "UNSUPPORTED on this implementation.",
    )
    conversation: ConversationParam = Field(
        default=None,
        description="The conversation that this response belongs to. Items from this conversation "
        "are prepended to `input_items` for this response request. Cannot be used with "
        "`previous_response_id`.\nUNSUPPORTED on this implementation.",
    )

    # Extra validations
    _UNSUPPORTED: ClassVar[set[str]] = {
        "text",
        "truncation",
        "previous_response_id",
        "conversation",
    }

    @model_validator(mode="after")
    def _unsupported(self) -> Self:
        """Validate that unsupported parameters are not used.

        Raises:
            UnsupportedParameterError: If a parameter marked as unsupported is used.
        """
        for key in self._UNSUPPORTED & self.model_fields_set:
            raise UnsupportedParameterError(key)
        return self


# Ref: openai.types.responses.input_token_count_response.InputTokenCountResponse
class InputTokenCountResponse(BaseModelResponse):
    """Response body for POST /v1/responses/input_tokens."""

    object: Literal["response.input_tokens"] = "response.input_tokens"
    input_tokens: int = Field(description="The total number of tokens in the input.")
