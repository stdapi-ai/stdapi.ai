"""Local Anthropic-compatible Messages API types."""

from typing import Annotated, Literal

from pydantic import AliasChoices, Field

from stdapi.input_file import FileIdInputFile, InputFile  # noqa: TC001
from stdapi.types import (
    BaseModelRequest,
    BaseModelRequestWithExtra,
    BaseModelResponse,
    JsonMapping,
)

#: Ref: anthropic.types.stop_reason.StopReason
StopReason = Literal[
    "end_turn", "max_tokens", "stop_sequence", "tool_use", "pause_turn", "refusal"
]

#: Tool choice literal values
ToolChoiceLiteral = Literal["auto", "any", "tool"]

#: Cache control types for prompt caching
CacheControlType = Literal["ephemeral"]

#: Allowed caller type with future-proof pattern matching
AllowedCaller = Annotated[
    str, Field(pattern=r"^(?:direct|code_execution(?:_[0-9]{8})?)$")
]

#: Service tiers
ServiceTiers = Literal[
    "auto",
    "standard_only",
    # Extra bedrock specific values
    "flex",
    "priority",
    "reserved",
]

# Server tools
ServerTools = Literal[
    "web_search",
    "web_fetch",
    "code_execution",
    "memory",
    "tool_search",
    "bash",
    "str_replace_editor",
    "str_replace_based_edit_tool",
    "computer",
]

SERVER_TOOL_NAMES: frozenset[ServerTools] = frozenset(
    {
        "web_search",
        "web_fetch",
        "code_execution",
        "memory",
        "tool_search",
        "bash",
        "str_replace_editor",
        "str_replace_based_edit_tool",
        "computer",
    }
)

#: Thinking effort level
ThinkingEffort = Literal["low", "medium", "high", "xhigh", "max"]


# Ref: anthropic.types.citation_char_location.CitationCharLocation
class CitationCharLocation(BaseModelResponse):
    """Character location citation."""

    type: Literal["char_location"] = Field(
        description="Citation type. Always `char_location`."
    )
    document_title: str | None = Field(
        default=None, description="Title of the cited document."
    )
    cited_text: str = Field(description="The text content of the citation.")
    document_index: int = Field(description="The index of the cited document.")
    end_char_index: int = Field(description="The ending character index.")
    file_id: str | None = Field(default=None, description="The file identifier.")
    start_char_index: int = Field(description="The starting character index.")


# Ref: anthropic.types.citation_page_location.CitationPageLocation
class CitationPageLocation(BaseModelResponse):
    """Page location citation."""

    type: Literal["page_location"] = Field(
        description="Citation type. Always `page_location`."
    )
    document_title: str | None = Field(
        default=None, description="Title of the cited document."
    )
    cited_text: str = Field(description="The text content of the citation.")
    document_index: int = Field(description="The index of the cited document.")
    end_page_number: int = Field(description="The ending page number.")
    file_id: str | None = Field(default=None, description="The file identifier.")
    start_page_number: int = Field(description="The starting page number.")


# Ref: anthropic.types.citation_content_block_location.CitationContentBlockLocation
class CitationContentBlockLocation(BaseModelResponse):
    """Content block location citation."""

    type: Literal["content_block_location"] = Field(
        description="Citation type. Always `content_block_location`."
    )
    document_title: str | None = Field(
        default=None, description="Title of the cited document."
    )
    cited_text: str = Field(description="The text content of the citation.")
    document_index: int = Field(description="The index of the cited document.")
    end_block_index: int = Field(description="The ending content block index.")
    file_id: str | None = Field(default=None, description="The file identifier.")
    start_block_index: int = Field(description="The starting content block index.")


# Ref: anthropic.types.citations_web_search_result_location.CitationsWebSearchResultLocation
class CitationsWebSearchResultLocation(BaseModelResponse):
    """Web search result citation location."""

    type: Literal["web_search_result_location"] = Field(
        description="Citation type. Always `web_search_result_location`."
    )
    url: str = Field(description="URL of the web search result.")
    title: str | None = Field(
        default=None, description="Title of the web search result."
    )
    cited_text: str = Field(description="The text content of the citation.")
    encrypted_index: str = Field(description="Encrypted index for the search result.")


# Ref: anthropic.types.citations_search_result_location.CitationsSearchResultLocation
class CitationsSearchResultLocation(BaseModelResponse):
    """Search result citation location."""

    type: Literal["search_result_location"] = Field(
        description="Citation type. Always `search_result_location`."
    )
    search_result_index: int = Field(description="Index of the search result.")
    cited_text: str = Field(description="The text content of the citation.")
    end_block_index: int = Field(description="The ending content block index.")
    source: str = Field(description="The source of the search result.")
    start_block_index: int = Field(description="The starting content block index.")
    title: str | None = Field(default=None, description="The title.")


# Ref: anthropic.types.text_citation.TextCitation
TextCitation = Annotated[
    CitationCharLocation
    | CitationPageLocation
    | CitationContentBlockLocation
    | CitationsWebSearchResultLocation
    | CitationsSearchResultLocation,
    Field(discriminator="type"),
]


# Ref: anthropic.types.cache_control_ephemeral_param.CacheControlEphemeralParam
class CacheControlEphemeralParam(BaseModelRequest):
    """Cache control configuration for prompt caching."""

    type: CacheControlType = Field(
        default="ephemeral", description="Cache control type. Always `ephemeral`."
    )
    ttl: Literal["5m", "1h"] | None = Field(
        default=None, description="The time-to-live for the cache control breakpoint."
    )


# Ref: anthropic.types.text_block.TextBlock
class TextBlock(BaseModelResponse):
    """Text content block."""

    type: Literal["text"] = Field(description="Content block type. Always `text`.")
    text: str = Field(description="Text content.")
    citations: list[TextCitation] | None = Field(
        default=None, description="Citations supporting the text block."
    )


# Ref: anthropic.types.citation_char_location_param.CitationCharLocationParam
class CitationCharLocationParam(BaseModelRequest):
    """Citation char location parameter."""

    cited_text: str = Field(description="The text content of the citation.")
    document_index: int = Field(description="The index of the cited document.")
    document_title: str | None = Field(
        default=None, description="The title of the cited document."
    )
    end_char_index: int = Field(description="The ending character index.")
    start_char_index: int = Field(description="The starting character index.")
    type: Literal["char_location"] = Field(description="The type discriminator.")


# Ref: anthropic.types.citation_page_location_param.CitationPageLocationParam
class CitationPageLocationParam(BaseModelRequest):
    """Citation page location parameter."""

    cited_text: str = Field(description="The text content of the citation.")
    document_index: int = Field(description="The index of the cited document.")
    document_title: str | None = Field(
        default=None, description="The title of the cited document."
    )
    end_page_number: int = Field(description="The ending page number.")
    start_page_number: int = Field(description="The starting page number.")
    type: Literal["page_location"] = Field(description="The type discriminator.")


# Ref: anthropic.types.citation_content_block_location_param.CitationContentBlockLocationParam
class CitationContentBlockLocationParam(BaseModelRequest):
    """Citation content block location parameter."""

    cited_text: str = Field(description="The text content of the citation.")
    document_index: int = Field(description="The index of the cited document.")
    document_title: str | None = Field(
        default=None, description="The title of the cited document."
    )
    end_block_index: int = Field(description="The ending content block index.")
    start_block_index: int = Field(description="The starting content block index.")
    type: Literal["content_block_location"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.citation_web_search_result_location_param.CitationWebSearchResultLocationParam
class CitationWebSearchResultLocationParam(BaseModelRequest):
    """Citation web search result location parameter."""

    cited_text: str = Field(description="The text content of the citation.")
    encrypted_index: str = Field(description="Encrypted index for the search result.")
    title: str | None = Field(default=None, description="The title.")
    type: Literal["web_search_result_location"] = Field(
        description="The type discriminator."
    )
    url: str = Field(description="The URL.")


# Ref : anthropic.types.citation_search_result_location_param.CitationSearchResultLocationParam
class CitationSearchResultLocationParam(BaseModelRequest):
    """Citation search result location parameter."""

    cited_text: str = Field(description="The text content of the citation.")
    end_block_index: int = Field(description="The ending content block index.")
    search_result_index: int = Field(description="The index of the search result.")
    source: str = Field(description="The source of the search result.")
    start_block_index: int = Field(description="The starting content block index.")
    title: str | None = Field(default=None, description="The title.")
    type: Literal["search_result_location"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.text_citation_param.TextCitationParam
TextCitationParam = (
    CitationCharLocationParam
    | CitationPageLocationParam
    | CitationContentBlockLocationParam
    | CitationWebSearchResultLocationParam
    | CitationSearchResultLocationParam
)


# Ref: anthropic.types.text_block_param.TextBlockParam
class TextBlockParam(BaseModelRequest):
    """Text content block parameter for system prompts and messages."""

    type: Literal["text"] = Field(description="Content block type. Always `text`.")
    text: str = Field(description="Text content.")
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None, description="Cache control for this content block."
    )
    citations: list[TextCitationParam] | None = Field(
        default=None,
        description="Citations supporting the text block. The type of citation returned will depend on the type of document being cited. Citing a PDF results in `page_location`, plain text results in `char_location`, and content document results in `content_block_location`.",
    )


# Ref: anthropic.types.base64_image_source_param.Base64ImageSourceParam
class Base64ImageSource(BaseModelRequest):
    """Image source for image content block."""

    type: Literal["base64"] = Field(description="Image source type. Always `base64`.")
    media_type: Literal["image/jpeg", "image/png", "image/gif", "image/webp"] = Field(
        description="Image media type."
    )
    data: InputFile = Field(
        description="Base64 encoded image data (or data URI, S3 URI or URL)."
    )


# Ref: anthropic.types.url_image_source_param.URLImageSourceParam
class URLImageSource(BaseModelRequest):
    """URL image source for image content block."""

    type: Literal["url"] = Field(description="Image source type. Always `url`.")
    url: InputFile = Field(
        description="URL of the image (or data URI, S3 URI or base64 encode string)."
    )


# Ref: anthropic.types.image_block_param.ImageBlockParam
class ImageBlockParam(BaseModelRequest):
    """Image content block parameter."""

    type: Literal["image"] = Field(description="Content block type. Always `image`.")
    source: Annotated[
        Base64ImageSource | URLImageSource | FileSource,
        Field(discriminator="type", description="Image source data."),
    ]
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None, description="Cache control for this content block."
    )


# Ref: anthropic.types.base64_pdf_source_param.Base64PDFSourceParam
class Base64PDFSource(BaseModelRequest):
    """Document source for document content block."""

    type: Literal["base64"] = Field(
        description="Document source type. Always `base64`."
    )
    media_type: Literal["application/pdf"] = Field(
        description="Document media type. Only `application/pdf` is supported."
    )
    data: InputFile = Field(
        description="Base64 encoded document data (or data URI, S3 URI or URL)."
    )


# Ref: anthropic.types.url_pdf_source_param.URLPDFSourceParam
class URLPDFSource(BaseModelRequest):
    """URL PDF source for document content block."""

    type: Literal["url"] = Field(description="Document source type. Always `url`.")
    url: InputFile = Field(
        description="URL of the PDF document (or data URI, S3 URI or base 64 encoded string)."
    )


# Ref: anthropic.types.plain_text_source_param.PlainTextSourceParam
class PlainTextSourceParam(BaseModelRequest):
    """Plain text source for document content block."""

    data: str = Field(description="The data content.")
    media_type: Literal["text/plain"] = Field(description="The media type.")
    type: Literal["text"] = Field(description="The type discriminator.")


# Ref: anthropic.types.content_block_source_param.ContentBlockSourceParam
class ContentBlockSourceParam(BaseModelRequest):
    """Content block source for document content block."""

    content: str | list[TextBlockParam | ImageBlockParam] = Field(
        description="The content of the source."
    )
    type: Literal["content"] = Field(description="The type discriminator.")


# Ref: anthropic.types.beta.BetaFileDocumentSourceParam / BetaFileImageSourceParam
class FileSource(BaseModelRequest):
    """File source for document or image content block (Files API)."""

    type: Literal["file"] = Field(description="Source type. Always `file`.")
    file_id: FileIdInputFile = Field(description="The Files API file identifier.")


# Ref: anthropic.types.citations_config_param.CitationsConfigParam
class CitationsConfigParam(BaseModelRequest):
    """Citations config parameter."""

    enabled: bool = Field(description="Whether the feature is enabled.")


# Ref: anthropic.types.document_block_param.DocumentBlockParam
class DocumentBlockParam(BaseModelRequest):
    """Document content block parameter."""

    type: Literal["document"] = Field(
        description="Content block type. Always `document`."
    )
    source: Annotated[
        Base64PDFSource
        | PlainTextSourceParam
        | ContentBlockSourceParam
        | URLPDFSource
        | FileSource,
        Field(discriminator="type", description="Document source data."),
    ]
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None, description="Cache control for this content block."
    )
    citations: CitationsConfigParam | None = Field(
        default=None, description="Citation configuration for the document"
    )
    context: str | None = Field(
        default=None, description="Additional context for the document."
    )
    title: str | None = Field(default=None, description="The title of the document")


# Ref: anthropic.types.direct_caller.DirectCaller
# Ref: anthropic.types.direct_caller_param.DirectCallerParam
class DirectCaller(BaseModelResponse):
    """Tool invocation directly from the model."""

    type: Literal["direct"] = Field(description="The type discriminator.")


# Ref: anthropic.types.server_tool_caller.ServerToolCaller
# Ref: anthropic.types.server_tool_caller_20260120.ServerToolCaller20260120
# Ref: anthropic.types.server_tool_caller_param.ServerToolCallerParam
# Ref: anthropic.types.server_tool_caller_20260120_param.ServerToolCaller20260120Param
class ServerToolCaller(BaseModelResponse):
    """Tool invocation generated by a server-side tool."""

    tool_id: str = Field(description="The tool identifier.")
    type: str = Field(
        pattern=r"^code_execution(?:_[0-9]{8})?$", description="The type discriminator."
    )


# Ref : anthropic.types.tool_use_block.Caller
Caller = DirectCaller | ServerToolCaller


# Ref: anthropic.types.tool_use_block.ToolUseBlock
class ToolUseBlock(BaseModelResponse):
    """Tool use content block."""

    type: Literal["tool_use"] = Field(
        description="Content block type. Always `tool_use`."
    )
    id: str = Field(description="Unique identifier for this tool use.")
    name: str = Field(description="Name of the tool being used.")
    input: JsonMapping = Field(description="Tool input parameters as a JSON object.")
    caller: Caller | None = Field(
        default=None, description="Tool invocation directly from the model."
    )


# Ref: anthropic.types.tool_use_block_param.ToolUseBlockParam
class ToolUseBlockParam(BaseModelRequest):
    """Tool use content block parameter."""

    type: Literal["tool_use"] = Field(
        description="Content block type. Always `tool_use`."
    )
    id: str = Field(description="Unique identifier for this tool use.")
    name: str = Field(description="Name of the tool being used.")
    input: JsonMapping = Field(description="Tool input parameters as a JSON object.")
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None, description="Cache control for this content block."
    )
    caller: Caller | None = Field(
        default=None, description="Tool invocation directly from the model."
    )


# Ref: anthropic.types.tool_result_block_param.ToolResultBlockParam
class ToolResultBlockParam(BaseModelRequest):
    """Tool result content block parameter."""

    type: Literal["tool_result"] = Field(
        description="Content block type. Always `tool_result`."
    )
    tool_use_id: str = Field(
        description="ID of the tool use this result corresponds to."
    )
    content: str | list[TextBlockParam | ImageBlockParam] = Field(
        description="Tool result content. Can be text or list of content blocks."
    )
    is_error: bool | None = Field(
        default=None, description="Whether this tool result represents an error."
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None, description="Cache control for this content block."
    )


# Ref: anthropic.types.thinking_block.ThinkingBlock
class ThinkingBlock(BaseModelResponse):
    """Thinking content block for extended thinking."""

    type: Literal["thinking"] = Field(
        description="Content block type. Always `thinking`."
    )
    thinking: str = Field(description="The thinking process content.")
    signature: str | None = Field(
        default=None, description="Signature for the thinking block."
    )


# Ref: anthropic.types.thinking_block_param.ThinkingBlockParam
class ThinkingBlockParam(BaseModelRequest):
    """Thinking content block parameter."""

    type: Literal["thinking"] = Field(
        description="Content block type. Always `thinking`."
    )
    thinking: str = Field(description="The thinking process content.")
    signature: str | None = Field(
        default=None,
        description="A token that verifies that the thinking text was generated by the model.",
    )


# Ref: anthropic.types.redacted_thinking_block.RedactedThinkingBlock
class RedactedThinkingBlock(BaseModelResponse):
    """Redacted thinking content block."""

    type: Literal["redacted_thinking"] = Field(
        description="Content block type. Always `redacted_thinking`."
    )
    data: str = Field(
        description="The redacted thinking content as a base64-encoded string."
    )


# Ref: anthropic.types.redacted_thinking_block_param.RedactedThinkingBlockParam
class RedactedThinkingBlockParam(BaseModelRequest):
    """Redacted thinking content block parameter."""

    type: Literal["redacted_thinking"] = Field(
        description="Content block type. Always `redacted_thinking`."
    )
    data: str = Field(
        description="The redacted thinking content as a base64-encoded string."
    )


# Ref: anthropic.types.server_tool_use_block.ServerToolUseBlock
class ServerToolUseBlock(BaseModelResponse):
    """Server-side tool use content block."""

    type: Literal["server_tool_use"] = Field(
        description="Content block type. Always `server_tool_use`."
    )
    id: str = Field(description="Unique identifier for this server tool use.")
    name: str = Field(description="Name of the server tool being used.")
    input: JsonMapping = Field(description="Tool input parameters as a JSON object.")
    caller: Caller | None = Field(
        default=None, description="Tool invocation directly from the model."
    )


# Ref: anthropic.types.server_tool_use_block_param.ServerToolUseBlockParam
class ServerToolUseBlockParam(BaseModelRequest):
    """Server-side tool use content block parameter."""

    type: Literal["server_tool_use"] = Field(
        description="Content block type. Always `server_tool_use`."
    )
    id: str = Field(description="Unique identifier for this server tool use.")
    name: str = Field(description="Name of the server tool being used.")
    input: JsonMapping = Field(description="Tool input parameters as a JSON object.")
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None, description="Cache control for this content block."
    )
    caller: Caller | None = Field(
        default=None, description="Tool invocation directly from the model."
    )


# Ref: anthropic.types.web_search_result_block.WebSearchResultBlock
class WebSearchResultBlock(BaseModelResponse):
    """Individual web search result."""

    type: Literal["web_search_result"] = Field(
        description="Result type. Always `web_search_result`."
    )
    encrypted_content: str | None = Field(
        default=None, description="Encrypted content of the web page."
    )
    title: str = Field(description="Title of the web page.")
    url: str = Field(description="URL of the web page.")
    page_age: str | None = Field(
        default=None, description="Age of the page (e.g., '2 days ago')."
    )


# Ref: anthropic.types.web_search_tool_result_error.WebSearchToolResultError
class WebSearchToolResultError(BaseModelResponse):
    """Web search tool result error."""

    error_code: str = Field(description="Machine-readable error code.")
    type: Literal["error"] = Field(description="Result type. Always `error`.")


# Ref: anthropic.types.web_search_tool_result_block_content.WebSearchToolResultBlockContent
WebSearchToolResultBlockContent = WebSearchToolResultError | list[WebSearchResultBlock]


# Ref: anthropic.types.web_search_tool_result_block.WebSearchToolResultBlock
class WebSearchToolResultBlock(BaseModelResponse):
    """Web search tool result content block."""

    type: Literal["web_search_tool_result"] = Field(
        description="Content block type. Always `web_search_tool_result`."
    )
    tool_use_id: str = Field(
        description="ID of the tool use this result corresponds to."
    )
    content: WebSearchToolResultBlockContent = Field(
        description="Search results or error."
    )
    caller: Caller | None = Field(
        default=None, description="Tool invocation directly from the model."
    )


# Ref: anthropic.types.search_result_block_param.SearchResultBlockParam
class SearchResultBlockParam(BaseModelRequest):
    """Search result content block parameter."""

    type: Literal["search_result"] = Field(
        description="Content block type. Always `search_result`."
    )
    content: list[TextBlockParam] = Field(
        description="The content of the search result."
    )
    source: str = Field(description="The source URL of the search result.")
    title: str = Field(description="Title of the search result.")
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None, description="Cache control for this content block."
    )
    citations: CitationsConfigParam | None = Field(
        default=None, description="Citations configuration for the search result."
    )


# Ref: anthropic.types.web_fetch_tool_result_error_block.WebFetchToolResultErrorBlock
class WebFetchToolResultErrorBlock(BaseModelResponse):
    """Web fetch tool result error block."""

    error_code: Literal[
        "invalid_tool_input",
        "url_too_long",
        "url_not_allowed",
        "url_not_accessible",
        "unsupported_content_type",
        "too_many_requests",
        "max_uses_exceeded",
        "unavailable",
    ] = Field(description="Machine-readable error code.")
    type: Literal["web_fetch_tool_result_error"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.citations_config.CitationsConfig
class CitationsConfig(BaseModelResponse):
    """Citations config."""

    enabled: bool = Field(description="Whether the feature is enabled.")


# Ref: anthropic.types.plain_text_source.PlainTextSource
class PlainTextSource(BaseModelResponse):
    """Plain text source."""

    data: str = Field(description="The data content.")
    media_type: Literal["text/plain"] = Field(description="The media type.")
    type: Literal["text"] = Field(description="The type discriminator.")


# Ref: anthropic.types.document_block.DocumentBlock
class DocumentBlock(BaseModelResponse):
    """Document block."""

    citations: CitationsConfig | None = Field(
        default=None, description="Citation configuration for the document"
    )
    source: Annotated[
        Base64PDFSource | PlainTextSource,
        Field(discriminator="type", description="The document source."),
    ]
    title: str | None = Field(default=None, description="The title of the document")
    type: Literal["document"] = Field(description="The type discriminator.")


# Ref: anthropic.types.web_fetch_block.WebFetchBlock
class WebFetchBlock(BaseModelResponse):
    """Web fetch block."""

    content: DocumentBlock = Field(description="The content of the block.")
    retrieved_at: str | None = Field(
        default=None, description="ISO 8601 timestamp when the content was retrieved"
    )
    type: Literal["web_fetch_result"] = Field(description="The type discriminator.")
    url: str = Field(description="Fetched content URL")


# Ref: anthropic.types.web_fetch_tool_result_block.WebFetchToolResultBlock
class WebFetchToolResultBlock(BaseModelResponse):
    """Web fetch tool result block."""

    caller: Caller | None = Field(
        default=None, description="Tool invocation directly from the model."
    )
    content: WebFetchToolResultErrorBlock | WebFetchBlock = Field(
        description="The content of the block."
    )
    tool_use_id: str = Field(
        description="The ID of the tool use that produced this result."
    )
    type: Literal["web_fetch_tool_result"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.code_execution_tool_result_error_code.CodeExecutionToolResultErrorCode
type CodeExecutionToolResultErrorCode = Literal[
    "invalid_tool_input", "unavailable", "too_many_requests", "execution_time_exceeded"
]


# Ref: anthropic.types.code_execution_tool_result_error.CodeExecutionToolResultError
class CodeExecutionToolResultError(BaseModelResponse):
    """Code execution tool result error."""

    error_code: CodeExecutionToolResultErrorCode = Field(
        description="Machine-readable error code."
    )
    type: Literal["code_execution_tool_result_error"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.code_execution_output_block.CodeExecutionOutputBlock
class CodeExecutionOutputBlock(BaseModelResponse):
    """Code execution output block."""

    file_id: str = Field(description="The file identifier.")
    type: Literal["code_execution_output"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.code_execution_result_block.CodeExecutionResultBlock
class CodeExecutionResultBlock(BaseModelResponse):
    """Code execution result block."""

    content: list[CodeExecutionOutputBlock] = Field(
        description="The content of the block."
    )
    return_code: int = Field(description="The return code of the execution.")
    stderr: str = Field(description="Standard error output.")
    stdout: str = Field(description="Standard output.")
    type: Literal["code_execution_result"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.encrypted_code_execution_result_block.EncryptedCodeExecutionResultBlock
class EncryptedCodeExecutionResultBlock(BaseModelResponse):
    """Code execution result with encrypted stdout for PFC + web_search results."""

    content: list[CodeExecutionOutputBlock] = Field(
        description="The content of the block."
    )
    encrypted_stdout: str = Field(description="Encrypted standard output.")
    return_code: int = Field(description="The return code of the execution.")
    stderr: str = Field(description="Standard error output.")
    type: Literal["encrypted_code_execution_result"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.code_execution_tool_result_block.CodeExecutionToolResultBlock
class CodeExecutionToolResultBlock(BaseModelResponse):
    """Code execution tool result block."""

    content: (
        CodeExecutionToolResultError
        | CodeExecutionResultBlock
        | EncryptedCodeExecutionResultBlock
    ) = Field(
        description="Code execution result with encrypted stdout for PFC + web_search results."
    )
    tool_use_id: str = Field(
        description="The ID of the tool use that produced this result."
    )
    type: Literal["code_execution_tool_result"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.bash_code_execution_tool_result_error_code.BashCodeExecutionToolResultErrorCode
BashCodeExecutionToolResultErrorCode = Literal[
    "invalid_tool_input",
    "unavailable",
    "too_many_requests",
    "execution_time_exceeded",
    "output_file_too_large",
]


# Ref: anthropic.types.bash_code_execution_tool_result_error.BashCodeExecutionToolResultError
class BashCodeExecutionToolResultError(BaseModelResponse):
    """Bash code execution tool result error."""

    error_code: BashCodeExecutionToolResultErrorCode = Field(
        description="Machine-readable error code."
    )
    type: Literal["bash_code_execution_tool_result_error"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.bash_code_execution_output_block.BashCodeExecutionOutputBlock
class BashCodeExecutionOutputBlock(BaseModelResponse):
    """Bash code execution output block."""

    file_id: str = Field(description="The file identifier.")
    type: Literal["bash_code_execution_output"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.bash_code_execution_result_block.BashCodeExecutionResultBlock
class BashCodeExecutionResultBlock(BaseModelResponse):
    """Bash code execution result block."""

    content: list[BashCodeExecutionOutputBlock] = Field(
        description="The content of the block."
    )
    return_code: int = Field(description="The return code of the execution.")
    stderr: str = Field(description="Standard error output.")
    stdout: str = Field(description="Standard output.")
    type: Literal["bash_code_execution_result"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.bash_code_execution_tool_result_block.BashCodeExecutionToolResultBlock
class BashCodeExecutionToolResultBlock(BaseModelResponse):
    """Bash code execution tool result block."""

    content: BashCodeExecutionToolResultError | BashCodeExecutionResultBlock = Field(
        description="The content of the block."
    )
    tool_use_id: str = Field(
        description="The ID of the tool use that produced this result."
    )
    type: Literal["bash_code_execution_tool_result"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.text_editor_code_execution_tool_result_error_code.TextEditorCodeExecutionToolResultErrorCode
TextEditorCodeExecutionToolResultErrorCode = Literal[
    "invalid_tool_input",
    "unavailable",
    "too_many_requests",
    "execution_time_exceeded",
    "file_not_found",
]


# Ref: anthropic.types.text_editor_code_execution_tool_result_error.TextEditorCodeExecutionToolResultError
class TextEditorCodeExecutionToolResultError(BaseModelResponse):
    """Text editor code execution tool result error."""

    error_code: TextEditorCodeExecutionToolResultErrorCode = Field(
        description="Machine-readable error code."
    )
    error_message: str | None = Field(
        default=None, description="Human-readable error message."
    )
    type: Literal["text_editor_code_execution_tool_result_error"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.text_editor_code_execution_view_result_block.TextEditorCodeExecutionViewResultBlock
class TextEditorCodeExecutionViewResultBlock(BaseModelResponse):
    """Text editor code execution view result block."""

    content: str = Field(description="The content of the block.")
    file_type: Literal["text", "image", "pdf"] = Field(
        description="The type of file output."
    )
    num_lines: int | None = Field(default=None, description="The number of lines.")
    start_line: int | None = Field(
        default=None, description="The starting line number."
    )
    total_lines: int | None = Field(
        default=None, description="Total number of lines in the file."
    )
    type: Literal["text_editor_code_execution_view_result"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.text_editor_code_execution_create_result_block.TextEditorCodeExecutionCreateResultBlock
class TextEditorCodeExecutionCreateResultBlock(BaseModelResponse):
    """Text editor code execution create result block."""

    is_file_update: bool = Field(description="Whether this result is a file update.")
    type: Literal["text_editor_code_execution_create_result"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.text_editor_code_execution_str_replace_result_block.TextEditorCodeExecutionStrReplaceResultBlock
class TextEditorCodeExecutionStrReplaceResultBlock(BaseModelResponse):
    """Text editor code execution str replace result block."""

    lines: list[str] | None = Field(default=None, description="The lines of content.")
    new_lines: int | None = Field(
        default=None, description="The number of lines in the new text."
    )
    new_start: int | None = Field(
        default=None, description="The starting line of the new text."
    )
    old_lines: int | None = Field(
        default=None, description="The number of lines in the original text."
    )
    old_start: int | None = Field(
        default=None, description="The starting line of the original text."
    )
    type: Literal["text_editor_code_execution_str_replace_result"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.text_editor_code_execution_tool_result_block.TextEditorCodeExecutionToolResultBlock
class TextEditorCodeExecutionToolResultBlock(BaseModelResponse):
    """Text editor code execution tool result block."""

    content: (
        TextEditorCodeExecutionToolResultError
        | TextEditorCodeExecutionViewResultBlock
        | TextEditorCodeExecutionCreateResultBlock
        | TextEditorCodeExecutionStrReplaceResultBlock
    ) = Field(description="The content of the block.")
    tool_use_id: str = Field(
        description="The ID of the tool use that produced this result."
    )
    type: Literal["text_editor_code_execution_tool_result"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.tool_search_tool_result_error_code.ToolSearchToolResultErrorCode
type ToolSearchToolResultErrorCode = Literal[
    "invalid_tool_input", "unavailable", "too_many_requests", "execution_time_exceeded"
]


# Ref: anthropic.types.tool_search_tool_result_error.ToolSearchToolResultError
class ToolSearchToolResultError(BaseModelResponse):
    """Tool search tool result error."""

    error_code: ToolSearchToolResultErrorCode = Field(
        description="Machine-readable error code."
    )
    error_message: str | None = Field(
        default=None, description="Human-readable error message."
    )
    type: Literal["tool_search_tool_result_error"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.tool_reference_block.ToolReferenceBlock
class ToolReferenceBlock(BaseModelResponse):
    """Tool reference block."""

    tool_name: str = Field(description="The name of the tool.")
    type: Literal["tool_reference"] = Field(description="The type discriminator.")


# Ref: anthropic.types.tool_search_tool_search_result_block.ToolSearchToolSearchResultBlock
class ToolSearchToolSearchResultBlock(BaseModelResponse):
    """Tool search tool search result block."""

    tool_references: list[ToolReferenceBlock] = Field(
        description="References to tools."
    )
    type: Literal["tool_search_tool_search_result"] = Field(
        description="The type discriminator."
    )


# Ref:anthropic.types.tool_search_tool_result_block.ToolSearchToolResultBlock
class ToolSearchToolResultBlock(BaseModelResponse):
    """Tool search tool result block."""

    content: ToolSearchToolResultError | ToolSearchToolSearchResultBlock = Field(
        description="The content of the block."
    )
    tool_use_id: str = Field(
        description="The ID of the tool use that produced this result."
    )
    type: Literal["tool_search_tool_result"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.container_upload_block.ContainerUploadBlock
class ContainerUploadBlock(BaseModelResponse):
    """Response model for a file uploaded to the container."""

    file_id: str = Field(description="The file identifier.")
    type: Literal["container_upload"] = Field(description="The type discriminator.")


# Content block unions
# Ref: anthropic.types.content_block.ContentBlock
ContentBlock = Annotated[
    TextBlock
    | ThinkingBlock
    | RedactedThinkingBlock
    | ToolUseBlock
    | ServerToolUseBlock
    | WebSearchToolResultBlock
    | WebFetchToolResultBlock
    | CodeExecutionToolResultBlock
    | BashCodeExecutionToolResultBlock
    | TextEditorCodeExecutionToolResultBlock
    | ToolSearchToolResultBlock
    | ContainerUploadBlock,
    Field(discriminator="type"),
]


# Ref: anthropic.types.web_search_result_block_param.WebSearchResultBlockParam
class WebSearchResultBlockParam(BaseModelRequest):
    """Web search result block parameter."""

    encrypted_content: str = Field(description="Encrypted content of the web page.")
    title: str = Field(description="The title.")
    type: Literal["web_search_result"] = Field(description="The type discriminator.")
    url: str = Field(description="The URL.")
    page_age: str | None = Field(
        default=None, description="How long ago the page was published or updated."
    )


# Ref: anthropic.types.web_search_tool_result_error_code.WebSearchToolResultErrorCode
WebSearchToolResultErrorCode = Literal[
    "invalid_tool_input",
    "unavailable",
    "max_uses_exceeded",
    "too_many_requests",
    "query_too_long",
    "request_too_large",
]


# Ref: anthropic.types.web_search_tool_request_error_param.WebSearchToolRequestErrorParam
class WebSearchToolRequestErrorParam(BaseModelRequest):
    """Web search tool request error parameter."""

    error_code: WebSearchToolResultErrorCode = Field(
        description="Machine-readable error code."
    )
    type: Literal["web_search_tool_result_error"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.web_search_tool_result_block_param_content_param.WebSearchToolResultBlockParamContentParam
WebSearchToolResultBlockParamContentParam = (
    list[WebSearchResultBlockParam] | WebSearchToolRequestErrorParam
)


# Ref: anthropic.types.web_search_tool_result_block_param.WebSearchToolResultBlockParam
class WebSearchToolResultBlockParam(BaseModelRequest):
    """Web search tool result block parameter."""

    content: WebSearchToolResultBlockParamContentParam = Field(
        description="The content of the block."
    )
    tool_use_id: str = Field(
        description="The ID of the tool use that produced this result."
    )
    type: Literal["web_search_tool_result"] = Field(
        description="The type discriminator."
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Create a cache control breakpoint at this content block.",
    )
    caller: Caller | None = Field(
        default=None, description="Tool invocation directly from the model."
    )


# Ref: anthropic.types.web_fetch_tool_result_error_code.WebFetchToolResultErrorCode
WebFetchToolResultErrorCode = Literal[
    "invalid_tool_input",
    "url_too_long",
    "url_not_allowed",
    "url_not_accessible",
    "unsupported_content_type",
    "too_many_requests",
    "max_uses_exceeded",
    "unavailable",
]


# Ref: anthropic.types.web_fetch_tool_result_error_block_param.WebFetchToolResultErrorBlockParam"
class WebFetchToolResultErrorBlockParam(BaseModelRequest):
    """Web fetch tool result error block parameter."""

    error_code: WebFetchToolResultErrorCode = Field(
        description="Machine-readable error code."
    )
    type: Literal["web_fetch_tool_result_error"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.web_fetch_block_param.WebFetchBlockParam
class WebFetchBlockParam(BaseModelRequest):
    """Web fetch block parameter."""

    content: DocumentBlockParam = Field(description="The content of the block.")
    type: Literal["web_fetch_result"] = Field(description="The type discriminator.")
    url: str = Field(description="Fetched content URL")
    retrieved_at: str | None = Field(
        default=None, description="ISO 8601 timestamp when the content was retrieved"
    )


# Ref: anthropic.types.web_fetch_tool_result_block_param.WebFetchToolResultBlockParam
class WebFetchToolResultBlockParam(BaseModelRequest):
    """Web fetch tool result block parameter."""

    content: WebFetchToolResultErrorBlockParam | WebFetchBlockParam = Field(
        description="The content of the block."
    )
    tool_use_id: str = Field(
        description="The ID of the tool use that produced this result."
    )
    type: Literal["web_fetch_tool_result"] = Field(
        description="The type discriminator."
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Create a cache control breakpoint at this content block.",
    )
    caller: Caller | None = Field(
        default=None, description="Tool invocation directly from the model."
    )


# Ref : anthropic.types.code_execution_output_block_param.CodeExecutionOutputBlockParam
class CodeExecutionOutputBlockParam(BaseModelRequest):
    """Code execution output block parameter."""

    file_id: str = Field(description="The file identifier.")
    type: Literal["code_execution_output"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.code_execution_tool_result_error_param.CodeExecutionToolResultErrorParam
class CodeExecutionToolResultErrorParam(BaseModelRequest):
    """Code execution tool result error parameter."""

    error_code: CodeExecutionToolResultErrorCode = Field(
        description="Machine-readable error code."
    )
    type: Literal["code_execution_tool_result_error"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.code_execution_result_block_param.CodeExecutionResultBlockParam
class CodeExecutionResultBlockParam(BaseModelRequest):
    """Code execution result block parameter."""

    content: list[CodeExecutionOutputBlockParam] = Field(
        description="The content of the block."
    )
    return_code: int = Field(description="The return code of the execution.")
    stderr: str = Field(description="Standard error output.")
    stdout: str = Field(description="Standard output.")
    type: Literal["code_execution_result"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.encrypted_code_execution_result_block_param.EncryptedCodeExecutionResultBlockParam
class EncryptedCodeExecutionResultBlockParam(BaseModelRequest):
    """Code execution result with encrypted stdout for PFC + web_search results."""

    content: list[CodeExecutionOutputBlockParam] = Field(
        description="The content of the block."
    )
    encrypted_stdout: str = Field(description="Encrypted standard output.")
    return_code: int = Field(description="The return code of the execution.")
    stderr: str = Field(description="Standard error output.")
    type: Literal["encrypted_code_execution_result"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.code_execution_tool_result_block_param_content_param.CodeExecutionToolResultBlockParamContentParam
type CodeExecutionToolResultBlockParamContentParam = (
    CodeExecutionToolResultErrorParam
    | CodeExecutionResultBlockParam
    | EncryptedCodeExecutionResultBlockParam
)


# Ref: anthropic.types.code_execution_tool_result_block_param.CodeExecutionToolResultBlockParam
class CodeExecutionToolResultBlockParam(BaseModelRequest):
    """Code execution tool result block parameter."""

    content: CodeExecutionToolResultBlockParamContentParam = Field(
        description="Code execution result with encrypted stdout for PFC + web_search results."
    )
    tool_use_id: str = Field(
        description="The ID of the tool use that produced this result."
    )
    type: Literal["code_execution_tool_result"] = Field(
        description="The type discriminator."
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Create a cache control breakpoint at this content block.",
    )


# Ref: anthropic.types.bash_code_execution_output_block_param.BashCodeExecutionOutputBlockParam
class BashCodeExecutionOutputBlockParam(BaseModelRequest):
    """Bash code execution output block parameter."""

    file_id: str = Field(description="The file identifier.")
    type: Literal["bash_code_execution_output"] = Field(
        description="The type discriminator."
    )


# Ref : anthropic.types.bash_code_execution_result_block_param.BashCodeExecutionResultBlockParam
class BashCodeExecutionResultBlockParam(BaseModelRequest):
    """Bash code execution result block parameter."""

    content: list[BashCodeExecutionOutputBlockParam] = Field(
        description="The content of the block."
    )
    return_code: int = Field(description="The return code of the execution.")
    stderr: str = Field(description="Standard error output.")
    stdout: str = Field(description="Standard output.")
    type: Literal["bash_code_execution_result"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.bash_code_execution_tool_result_error_param.BashCodeExecutionToolResultErrorParam
class BashCodeExecutionToolResultErrorParam(BaseModelRequest):
    """Bash code execution tool result error parameter."""

    error_code: BashCodeExecutionToolResultErrorCode = Field(
        description="Machine-readable error code."
    )
    type: Literal["bash_code_execution_tool_result_error"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.bash_code_execution_tool_result_block_param.BashCodeExecutionToolResultBlockParam
class BashCodeExecutionToolResultBlockParam(BaseModelRequest):
    """Bash code execution tool result block parameter."""

    content: (
        BashCodeExecutionToolResultErrorParam | BashCodeExecutionResultBlockParam
    ) = Field()
    tool_use_id: str = Field(
        description="The ID of the tool use that produced this result."
    )
    type: Literal["bash_code_execution_tool_result"] = Field(
        description="The type discriminator."
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Create a cache control breakpoint at this content block.",
    )


# Ref: anthropic.types.text_editor_code_execution_tool_result_error_param.TextEditorCodeExecutionToolResultErrorParam
class TextEditorCodeExecutionToolResultErrorParam(BaseModelRequest):
    """Text editor code execution tool result error parameter."""

    error_code: TextEditorCodeExecutionToolResultErrorCode = Field(
        description="Machine-readable error code."
    )
    type: Literal["text_editor_code_execution_tool_result_error"] = Field(
        description="The type discriminator."
    )
    error_message: str | None = Field(
        default=None, description="Human-readable error message."
    )


# Ref: anthropic.types.text_editor_code_execution_view_result_block_param.TextEditorCodeExecutionViewResultBlockParam
class TextEditorCodeExecutionViewResultBlockParam(BaseModelRequest):
    """Text editor code execution view result block parameter."""

    content: str = Field(description="The content of the block.")
    file_type: Literal["text", "image", "pdf"] = Field(
        description="The type of file output."
    )
    type: Literal["text_editor_code_execution_view_result"] = Field(
        description="The type discriminator."
    )
    num_lines: int | None = Field(default=None, description="The number of lines.")
    start_line: int | None = Field(
        default=None, description="The starting line number."
    )
    total_lines: int | None = Field(
        default=None, description="Total number of lines in the file."
    )


# Ref: anthropic.types.text_editor_code_execution_create_result_block_param.TextEditorCodeExecutionCreateResultBlockParam
class TextEditorCodeExecutionCreateResultBlockParam(BaseModelRequest):
    """Text editor code execution create result block parameter."""

    is_file_update: bool = Field(description="Whether this result is a file update.")
    type: Literal["text_editor_code_execution_create_result"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.text_editor_code_execution_str_replace_result_block_param.TextEditorCodeExecutionStrReplaceResultBlockParam
class TextEditorCodeExecutionStrReplaceResultBlockParam(BaseModelRequest):
    """Text editor code execution str replace result block parameter."""

    type: Literal["text_editor_code_execution_str_replace_result"] = Field(
        description="The type discriminator."
    )
    lines: list[str] | None = Field(default=None, description="The lines of content.")
    new_lines: int | None = Field(
        default=None, description="The number of lines in the new text."
    )
    new_start: int | None = Field(
        default=None, description="The starting line of the new text."
    )
    old_lines: int | None = Field(
        default=None, description="The number of lines in the original text."
    )
    old_start: int | None = Field(
        default=None, description="The starting line of the original text."
    )


# Ref: anthropic.types.text_editor_code_execution_tool_result_block_param.TextEditorCodeExecutionToolResultBlockParam
class TextEditorCodeExecutionToolResultBlockParam(BaseModelRequest):
    """Text editor code execution tool result block parameter."""

    content: (
        TextEditorCodeExecutionToolResultErrorParam
        | TextEditorCodeExecutionViewResultBlockParam
        | TextEditorCodeExecutionCreateResultBlockParam
        | TextEditorCodeExecutionStrReplaceResultBlockParam
    ) = Field()
    tool_use_id: str = Field(
        description="The ID of the tool use that produced this result."
    )
    type: Literal["text_editor_code_execution_tool_result"] = Field(
        description="The type discriminator."
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Create a cache control breakpoint at this content block.",
    )


# Ref: anthropic.types.tool_search_tool_result_error_param.ToolSearchToolResultErrorParam
class ToolSearchToolResultErrorParam(BaseModelRequest):
    """Tool search tool result error parameter."""

    error_code: ToolSearchToolResultErrorCode = Field(
        description="Machine-readable error code."
    )
    type: Literal["tool_search_tool_result_error"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.tool_reference_block_param.ToolReferenceBlockParam
class ToolReferenceBlockParam(BaseModelRequest):
    """Tool reference block that can be included in tool_result content."""

    tool_name: str = Field(description="The name of the tool.")
    type: Literal["tool_reference"] = Field(description="The type discriminator.")
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Create a cache control breakpoint at this content block.",
    )


# Ref: anthropic.types.tool_search_tool_search_result_block_param.ToolSearchToolSearchResultBlockParam
class ToolSearchToolSearchResultBlockParam(BaseModelRequest):
    """Tool search tool search result block parameter."""

    tool_references: list[ToolReferenceBlockParam] = Field(
        description="References to tools."
    )
    type: Literal["tool_search_tool_search_result"] = Field(
        description="The type discriminator."
    )


# Ref: anthropic.types.tool_search_tool_result_block_param.ToolSearchToolResultBlockParam
class ToolSearchToolResultBlockParam(BaseModelRequest):
    """Tool search tool result block parameter."""

    content: ToolSearchToolResultErrorParam | ToolSearchToolSearchResultBlockParam = (
        Field()
    )

    tool_use_id: str = Field(
        description="The ID of the tool use that produced this result."
    )
    type: Literal["tool_search_tool_result"] = Field(
        description="The type discriminator."
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Create a cache control breakpoint at this content block.",
    )


# Ref: anthropic.types.container_upload_block_param.ContainerUploadBlockParam
class ContainerUploadBlockParam(BaseModelRequest):
    """A content block for file upload to the container.

    Files uploaded via this block will be available in the container's input directory.
    UNSUPPORTED on this implementation.
    """

    file_id: str = Field(description="The file identifier.")
    type: Literal["container_upload"] = Field(description="The type discriminator.")
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Create a cache control breakpoint at this content block.",
    )


# Ref: anthropic.types.content_block_param.ContentBlockParam
# Note: ContentBlock (response types) is included because assistant messages may
# contain response blocks that are passed back in multi-turn conversations.
# The discriminator is omitted because param and response types share type values.
ContentBlockParam = (
    TextBlockParam
    | ImageBlockParam
    | DocumentBlockParam
    | SearchResultBlockParam
    | ThinkingBlockParam
    | RedactedThinkingBlockParam
    | ToolUseBlockParam
    | ToolResultBlockParam
    | ServerToolUseBlockParam
    | WebSearchToolResultBlockParam
    | WebFetchToolResultBlockParam
    | CodeExecutionToolResultBlockParam
    | BashCodeExecutionToolResultBlockParam
    | TextEditorCodeExecutionToolResultBlockParam
    | ToolSearchToolResultBlockParam
    | ContainerUploadBlockParam
    | ContentBlock
)


# Ref: anthropic.types.message_param.MessageParam
class MessageParam(BaseModelRequest):
    """Base message parameter."""

    role: Literal["user", "assistant"] = Field(
        description="The role of the message author."
    )
    content: str | list[ContentBlockParam] = Field(description="The message content.")


# Ref: anthropic.types.tool_param.ToolParam.InputSchema
# Ref: anthropic.types.tool_param.InputSchemaTyped
class ToolInputSchema(BaseModelRequest):
    """JSON schema for tool input parameters."""

    type: Literal["object"] = Field(
        description="Schema type. Must be `object`.", default="object"
    )
    properties: JsonMapping | None = Field(
        default=None, description="Properties of the tool input schema."
    )
    required: list[str] | None = Field(default=None, description="Required properties.")


# Ref: anthropic.types.tool_param.ToolParam
class ToolParam(BaseModelRequest):
    """Tool definition for function calling."""

    type: Literal["custom"] = Field(
        default="custom", description="Tool type. Always `custom` for custom tools."
    )
    name: str = Field(
        description="""Name of the tool.

    This is how the tool will be called by the model and in `tool_use` blocks.
    """
    )
    input_schema: ToolInputSchema = Field(
        description="""[JSON schema](https://json-schema.org/draft/2020-12) for this tool's input.

    This defines the shape of the `input` that your tool accepts and that the model
    will produce.
    """
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Create a cache control breakpoint at this content block.",
    )
    description: str | None = Field(
        default=None,
        description="""Description of what this tool does.

    Tool descriptions should be as detailed as possible. The more information that
    the model has about what the tool is and how to use it, the better it will
    perform. You can use natural language descriptions to reinforce important
    aspects of the tool input JSON schema.
    """,
    )
    eager_input_streaming: bool | None = Field(
        default=None,
        description="""Enable eager input streaming for this tool.

    When true, tool input parameters will be streamed incrementally as they are
    generated, and types will be inferred on-the-fly rather than buffering the full
    JSON output. When false, streaming is disabled for this tool even if the
    fine-grained-tool-streaming beta is active. When null (default), uses the
    default behavior based on beta headers.
    """,
    )
    strict: bool | None = Field(
        default=None,
        description="When true, guarantees schema validation on tool names and inputs.",
    )
    allowed_callers: list[AllowedCaller] | None = Field(
        default=None, description="List of allowed callers for this tool."
    )
    defer_loading: bool | None = Field(
        default=None,
        description="If true, tool will not be included in initial system prompt.  Only loaded when returned via tool_reference from tool search.",
    )
    input_examples: list[dict[str, object]] | None = Field(
        default=None, description="Example inputs for the tool."
    )


# Ref: anthropic.types.tool_bash_20250124_param.ToolBash20250124Param
class ToolBashParam(BaseModelRequest):
    """Bash tool definition for command execution."""

    type: str = Field(
        pattern=r"^bash(?:_[0-9]{8})?$", description="Tool type. Always `bash_*`."
    )
    name: Literal["bash"] = Field(
        description="""Name of the tool.

    This is how the tool will be called by the model and in `tool_use` blocks.
    """
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Create a cache control breakpoint at this content block.",
    )
    allowed_callers: list[AllowedCaller] | None = Field(
        default=None, description="List of allowed callers for this tool."
    )
    defer_loading: bool | None = Field(
        default=None,
        description="If true, tool will not be included in initial system prompt.  Only loaded when returned via tool_reference from tool search.",
    )
    input_examples: list[dict[str, object]] | None = Field(
        default=None, description="Example inputs for the tool."
    )
    strict: bool | None = Field(
        default=None,
        description="When true, guarantees schema validation on tool names and inputs",
    )


# Ref: anthropic.types.tool_text_editor_20250124_param.ToolTextEditor20250124Param
# Ref: anthropic.types.tool_text_editor_20250429_param.ToolTextEditor20250429Param
# Ref: anthropic.types.tool_text_editor_20250728_param.ToolTextEditor20250728Param
class ToolTextEditorParam(BaseModelRequest):
    """Text editor tool definition for file editing."""

    type: str = Field(
        pattern=r"^text_editor(?:_[0-9]{8})?$",
        description="Tool type. Always `text_editor_*`.",
    )
    name: Literal["str_replace_editor", "str_replace_based_edit_tool"] = Field(
        description="""Name of the tool.

    This is how the tool will be called by the model and in `tool_use` blocks.
    """
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Create a cache control breakpoint at this content block.",
    )
    allowed_callers: list[AllowedCaller] | None = Field(
        default=None, description="List of allowed callers for this tool."
    )
    defer_loading: bool | None = Field(
        default=None,
        description="If true, tool will not be included in initial system prompt.  Only loaded when returned via tool_reference from tool search.",
    )
    input_examples: list[dict[str, object]] | None = Field(
        default=None, description="Example inputs for the tool."
    )
    strict: bool | None = Field(
        default=None,
        description="When true, guarantees schema validation on tool names and inputs",
    )
    max_characters: int | None = Field(
        default=None,
        description="Maximum number of characters to display when viewing a file.  If not specified, defaults to displaying the full file.",
    )


# Ref: anthropic.types.user_location_param.UserLocationParam
class UserLocationParam(BaseModelRequest):
    """User location parameter."""

    type: Literal["approximate"] = Field(description="The type discriminator.")
    city: str | None = Field(default=None, description="The city of the user.")
    country: str | None = Field(
        default=None,
        description="The two letter [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) of the user.",
    )
    region: str | None = Field(default=None, description="The region of the user.")
    timezone: str | None = Field(
        default=None,
        description="The [IANA timezone](https://nodatime.org/TimeZones) of the user.",
    )


# Ref: anthropic.types.web_search_tool_20250305_param.WebSearchTool20250305Param
# Ref: anthropic.types.web_search_tool_20260209_param.WebSearchTool20260209Param
class WebSearchToolParam(BaseModelRequest):
    """Web search tool definition.

    Supported on models that declare web search as a system tool
    (e.g., Amazon Nova 2 via ``nova_grounding``).
    """

    type: str = Field(
        pattern=r"^web_search(?:_[0-9]{8})?$",
        description="Tool type. Always `web_search_*`.",
    )
    name: Literal["web_search"] = Field(
        description="""Name of the tool.

    This is how the tool will be called by the model and in `tool_use` blocks.
    """
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Create a cache control breakpoint at this content block.",
    )
    allowed_callers: list[AllowedCaller] | None = Field(
        default=None, description="List of allowed callers for this tool."
    )
    allowed_domains: list[str] | None = Field(
        default=None,
        description="If provided, only these domains will be included in results.  Cannot be used alongside `blocked_domains`.",
    )
    blocked_domains: list[str] | None = Field(
        default=None,
        description="If provided, these domains will never appear in results.  Cannot be used alongside `allowed_domains`.",
    )
    defer_loading: bool | None = Field(
        default=None,
        description="If true, tool will not be included in initial system prompt.  Only loaded when returned via tool_reference from tool search.",
    )
    max_uses: int | None = Field(
        default=None,
        description="Maximum number of times the tool can be used in the API request.",
    )
    strict: bool | None = Field(
        default=None,
        description="When true, guarantees schema validation on tool names and inputs",
    )
    user_location: UserLocationParam | None = Field(
        default=None,
        description="Parameters for the user's location.  Used to provide more relevant search results.",
    )


# Ref: anthropic.types.code_execution_tool_20250522_param.CodeExecutionTool20250522Param
# Ref: anthropic.types.code_execution_tool_20250825_param.CodeExecutionTool20250825Param
# Ref: anthropic.types.code_execution_tool_20260120_param.CodeExecutionTool20260120Param
class CodeExecutionToolParam(BaseModelRequest):
    """Code execution tool parameter."""

    name: Literal["code_execution"] = Field(
        description="Name of the tool.  This is how the tool will be called by the model and in `tool_use` blocks."
    )
    type: str = Field(
        pattern=r"^code_execution(?:_[0-9]{8})?$", description="The type discriminator."
    )
    allowed_callers: list[AllowedCaller] | None = Field(
        default=None, description="List of allowed callers for this tool."
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Create a cache control breakpoint at this content block.",
    )
    defer_loading: bool | None = Field(
        default=None,
        description="If true, tool will not be included in initial system prompt.  Only loaded when returned via tool_reference from tool search.",
    )
    strict: bool | None = Field(
        default=None,
        description="When true, guarantees schema validation on tool names and inputs",
    )


# Ref : anthropic.types.memory_tool_20250818_param.MemoryTool20250818Param
class MemoryToolParam(BaseModelRequest):
    """Memory tool parameter."""

    name: Literal["memory"] = Field(
        description="Name of the tool.  This is how the tool will be called by the model and in `tool_use` blocks."
    )
    type: str = Field(
        pattern=r"^memory(?:_[0-9]{8})?$", description="The type discriminator."
    )
    allowed_callers: list[AllowedCaller] | None = Field(
        default=None, description="List of allowed callers for this tool."
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Create a cache control breakpoint at this content block.",
    )
    defer_loading: bool | None = Field(
        default=None,
        description="If true, tool will not be included in initial system prompt.  Only loaded when returned via tool_reference from tool search.",
    )
    input_examples: list[dict[str, object]] | None = Field(
        default=None, description="Example inputs for the tool."
    )
    strict: bool | None = Field(
        default=None,
        description="When true, guarantees schema validation on tool names and inputs",
    )


# Ref : anthropic.types.web_fetch_tool_20250910_param.WebFetchTool20250910Param
# Ref : anthropic.types.web_fetch_tool_20260209_param.WebFetchTool20260209Param
class WebFetchToolParam(BaseModelRequest):
    """Web fetch tool parameter."""

    name: Literal["web_fetch"] = Field(
        description="Name of the tool.  This is how the tool will be called by the model and in `tool_use` blocks."
    )
    type: str = Field(
        pattern=r"^web_fetch(?:_[0-9]{8})?$", description="The type discriminator."
    )
    allowed_callers: list[AllowedCaller] | None = Field(
        default=None, description="List of allowed callers for this tool."
    )
    allowed_domains: list[str] | None = Field(
        default=None, description="List of domains to allow fetching from"
    )
    blocked_domains: list[str] | None = Field(
        default=None, description="List of domains to block fetching from"
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Create a cache control breakpoint at this content block.",
    )
    citations: CitationsConfigParam | None = Field(
        default=None,
        description="Citations configuration for fetched documents.  Citations are disabled by default.",
    )
    defer_loading: bool | None = Field(
        default=None,
        description="If true, tool will not be included in initial system prompt.  Only loaded when returned via tool_reference from tool search.",
    )
    max_content_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens used by including web page text content in the context.  The limit is approximate and does not apply to binary content such as PDFs.",
    )
    max_uses: int | None = Field(
        default=None,
        description="Maximum number of times the tool can be used in the API request.",
    )
    strict: bool | None = Field(
        default=None,
        description="When true, guarantees schema validation on tool names and inputs",
    )


# Ref: anthropic.types.beta.beta_tool_computer_use_20241022_param.BetaToolComputerUse20241022Param
# Ref: anthropic.types.beta.beta_tool_computer_use_20250124_param.BetaToolComputerUse20250124Param
# Ref: anthropic.types.beta.beta_tool_computer_use_20251124_param.BetaToolComputerUse20251124Param
class ToolComputerParam(BaseModelRequest):
    """Computer use tool definition for GUI automation."""

    type: str = Field(
        pattern=r"^computer(?:_[0-9]{8})?$",
        description="Tool type. Always ``computer_*``.",
    )
    name: Literal["computer"] = Field(
        description="Name of the tool.  This is how the tool will be called by the model and in ``tool_use`` blocks."
    )
    display_width_px: int = Field(description="The width of the display in pixels.")
    display_height_px: int = Field(description="The height of the display in pixels.")
    display_number: int | None = Field(
        default=None, description="The X11 display number (e.g. 0, 1) for the display."
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Create a cache control breakpoint at this content block.",
    )
    allowed_callers: list[AllowedCaller] | None = Field(
        default=None, description="List of allowed callers for this tool."
    )
    defer_loading: bool | None = Field(
        default=None,
        description="If true, tool will not be included in initial system prompt.  Only loaded when returned via tool_reference from tool search.",
    )
    enable_zoom: bool | None = Field(
        default=None,
        description="Whether to enable an action to take a zoomed-in screenshot of the screen.  Added in ``computer_20251124``.",
    )
    input_examples: list[dict[str, object]] | None = Field(
        default=None, description="Example inputs for the tool."
    )
    strict: bool | None = Field(
        default=None,
        description="When true, guarantees schema validation on tool names and inputs",
    )


# Ref: anthropic.types.tool_search_tool_bm25_20251119_param.ToolSearchToolBm25_20251119Param
class ToolSearchToolBm25Param(BaseModelRequest):
    """Tool search tool BM25 parameter."""

    name: Literal["tool_search_tool_bm25"] = Field(
        description="Name of the tool.  This is how the tool will be called by the model and in `tool_use` blocks."
    )
    type: str = Field(
        pattern=r"^tool_search_tool_bm25(?:_[0-9]{8})?$",
        description="The type discriminator.",
    )
    allowed_callers: list[AllowedCaller] | None = Field(
        default=None, description="List of allowed callers for this tool."
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Create a cache control breakpoint at this content block.",
    )
    defer_loading: bool | None = Field(
        default=None,
        description="If true, tool will not be included in initial system prompt.  Only loaded when returned via tool_reference from tool search.",
    )
    strict: bool | None = Field(
        default=None,
        description="When true, guarantees schema validation on tool names and inputs",
    )


# Ref: anthropic.types.tool_search_tool_regex_20251119_param.ToolSearchToolRegex20251119Param
class ToolSearchToolRegexParam(BaseModelRequest):
    """Tool search tool regex parameter."""

    name: Literal["tool_search_tool_regex"] = Field(
        description="Name of the tool.  This is how the tool will be called by the model and in `tool_use` blocks."
    )
    type: str = Field(
        pattern=r"^tool_search_tool_regex(?:_[0-9]{8})?$",
        description="The type discriminator.",
    )
    allowed_callers: list[AllowedCaller] | None = Field(
        default=None, description="List of allowed callers for this tool."
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Create a cache control breakpoint at this content block.",
    )
    defer_loading: bool | None = Field(
        default=None,
        description="If true, tool will not be included in initial system prompt.  Only loaded when returned via tool_reference from tool search.",
    )
    strict: bool | None = Field(
        default=None,
        description="When true, guarantees schema validation on tool names and inputs",
    )


# Ref: anthropic.types.tool_union_param.ToolUnionParam
# Ref: anthropic.types.message_count_tokens_tool_param.MessageCountTokensToolParam
ToolUnionParam = (
    ToolParam
    | ToolBashParam
    | ToolTextEditorParam
    | ToolComputerParam
    | WebSearchToolParam
    | CodeExecutionToolParam
    | MemoryToolParam
    | WebFetchToolParam
    | ToolSearchToolBm25Param
    | ToolSearchToolRegexParam
)


# Ref: anthropic.types.tool_choice_auto_param.ToolChoiceAutoParam
class ToolChoiceAutoParam(BaseModelRequest):
    """Auto tool choice - model decides whether to use tools."""

    type: Literal["auto"] = Field(description="Tool choice type. Always `auto`.")
    disable_parallel_tool_use: bool | None = Field(
        default=None, description="Whether to disable parallel tool use."
    )


# Ref: anthropic.types.tool_choice_any_param.ToolChoiceAnyParam
class ToolChoiceAnyParam(BaseModelRequest):
    """Any tool choice - model must use at least one tool."""

    type: Literal["any"] = Field(description="Tool choice type. Always `any`.")
    disable_parallel_tool_use: bool | None = Field(
        default=None, description="Whether to disable parallel tool use."
    )


# Ref: anthropic.types.tool_choice_tool_param.ToolChoiceToolParam
class ToolChoiceToolParam(BaseModelRequest):
    """Specific tool choice - model must use the specified tool."""

    type: Literal["tool"] = Field(description="Tool choice type. Always `tool`.")
    name: str = Field(description="Name of the tool to use.")
    disable_parallel_tool_use: bool | None = Field(
        default=None, description="Whether to disable parallel tool use."
    )


# Ref: anthropic.types.tool_choice_none_param.ToolChoiceNoneParam
class ToolChoiceNoneParam(BaseModelRequest):
    """The model will not be allowed to use tools."""

    type: Literal["none"] = Field(description="The type discriminator.")


# Ref: anthropic.types.tool_choice_param.ToolChoiceParam
ToolChoiceParam = Annotated[
    ToolChoiceAutoParam
    | ToolChoiceAnyParam
    | ToolChoiceToolParam
    | ToolChoiceNoneParam,
    Field(discriminator="type"),
]


# Ref: anthropic.types.metadata_param.MetadataParam
class MetadataParam(BaseModelRequest):
    """Request metadata for tracking and filtering."""

    user_id: str | None = Field(
        default=None, description="An external identifier for the end-user."
    )


# Ref: anthropic.types.cache_creation.CacheCreation
class CacheCreation(BaseModelResponse):
    """Cache creation token usage breakdown."""

    ephemeral_1h_input_tokens: int = Field(
        description="The number of input tokens used to create the 1 hour cache entry."
    )
    ephemeral_5m_input_tokens: int = Field(
        description="The number of input tokens used to create the 5 minute cache entry."
    )


# Ref: anthropic.types.server_tool_usage.ServerToolUsage
class ServerToolUsage(BaseModelResponse):
    """Server tool usage statistics."""

    web_search_requests: int = Field(
        description="The number of web search tool requests."
    )
    web_fetch_requests: int = Field(
        description="The number of web fetch tool requests."
    )


# Ref: anthropic.types.usage.Usage
class Usage(BaseModelResponse):
    """Token usage information."""

    input_tokens: int = Field(description="Number of input tokens.")
    output_tokens: int = Field(description="Number of output tokens.")
    cache_creation_input_tokens: int | None = Field(
        default=None,
        description="Number of input tokens used to create the cache entry.",
    )
    cache_read_input_tokens: int | None = Field(
        default=None, description="Number of input tokens read from the cache."
    )
    cache_creation: CacheCreation | None = Field(
        default=None, description="Detailed cache creation token usage breakdown."
    )
    inference_geo: str | None = Field(
        default=None, description="Inference geographic region."
    )
    server_tool_use: ServerToolUsage | None = Field(
        default=None, description="Server tool usage statistics."
    )
    service_tier: Literal["standard", "priority", "batch"] | None = Field(
        default=None, description="The service tier used for the request."
    )


# Ref: anthropic.types.container.Container
class Container(BaseModelResponse):
    """Information about the container used in the request (for the code execution tool)."""

    id: str = Field(description="Identifier for the container used in this request.")
    expires_at: str = Field(description="The time at which the container will expire.")


# Ref: anthropic.types.message.Message
class Message(BaseModelResponse):
    """Messages API response."""

    id: str = Field(
        description="""Unique object identifier.

    The format and length of IDs may change over time.
    """
    )
    type: Literal["message"] = Field(
        description="""Object type.

    For Messages, this is always `"message"`.
    """
    )
    role: Literal["assistant"] = Field(
        description="""Conversational role of the generated message.

    This will always be `"assistant"`.
    """
    )
    content: list[ContentBlock] = Field(
        description="""Content generated by the model.

    This is an array of content blocks, each of which has a `type` that determines
    its shape.

    Example:

    ```json
    [{ "type": "text", "text": "Hi" }]
    ```

    If the request input `messages` ended with an `assistant` turn, then the
    response `content` will continue directly from that last turn. You can use this
    to constrain the model's output.

    For example, if the input `messages` were:

    ```json
    [
      {
        "role": "user",
        "content": "What's the Greek name for Sun? (A) Sol (B) Helios (C) Sun"
      },
      { "role": "assistant", "content": "The best answer is (" }
    ]
    ```

    Then the response `content` might be:

    ```json
    [{ "type": "text", "text": "B)" }]
    ```
    """
    )
    model: str = Field(description="The model that will complete your prompt.")
    stop_reason: StopReason | None = Field(
        default=None,
        description="""The reason that we stopped.

    This may be one the following values:

    - `"end_turn"`: the model reached a natural stopping point
    - `"max_tokens"`: we exceeded the requested `max_tokens` or the model's maximum
    - `"stop_sequence"`: one of your provided custom `stop_sequences` was generated
    - `"tool_use"`: the model invoked one or more tools
    - `"pause_turn"`: we paused a long-running turn. You may provide the response
      back as-is in a subsequent request to let the model continue.
    - `"refusal"`: when streaming classifiers intervene to handle potential policy
      violations

    In non-streaming mode this value is always non-null. In streaming mode, it is
    null in the `message_start` event and non-null otherwise.
    """,
    )
    stop_sequence: str | None = Field(
        default=None,
        description="""Which custom stop sequence was generated, if any.

    This value will be a non-null string if one of your custom stop sequences was
    generated.
    """,
    )
    usage: Usage = Field(
        description="""Billing and rate-limit usage.

    The API bills and rate-limits by token counts, as tokens represent the
    underlying cost to our systems.

    Under the hood, the API transforms requests into a format suitable for the
    model. The model's output then goes through a parsing stage before becoming an
    API response. As a result, the token counts in `usage` will not match one-to-one
    with the exact visible content of an API request or response.

    For example, `output_tokens` will be non-zero, even for an empty string response.

    Total input tokens in a request is the summation of `input_tokens`,
    `cache_creation_input_tokens`, and `cache_read_input_tokens`.
    """
    )

    container: Container | None = Field(
        description="Information about the container used in the request (for the code execution tool).",
        default=None,
    )


# Ref: anthropic.types.text_delta.TextDelta
class TextDelta(BaseModelResponse):
    """Text delta for streaming updates."""

    type: Literal["text_delta"] = Field(description="Delta type. Always `text_delta`.")
    text: str = Field(description="Text content delta.")


# Ref: anthropic.types.input_json_delta.InputJSONDelta
class InputJSONDelta(BaseModelResponse):
    """Input JSON delta for streaming tool use."""

    type: Literal["input_json_delta"] = Field(
        description="Delta type. Always `input_json_delta`."
    )
    partial_json: str = Field(description="Partial JSON string.")


# Ref: anthropic.types.citations_delta.CitationsDelta
class CitationsDelta(BaseModelResponse):
    """Citations delta for streaming updates."""

    type: Literal["citations_delta"] = Field(
        description="Delta type. Always `citations_delta`."
    )
    citation: TextCitation = Field(description="The citation update.")


# Ref: anthropic.types.thinking_delta.ThinkingDelta
class ThinkingDelta(BaseModelResponse):
    """Thinking delta for streaming updates."""

    type: Literal["thinking_delta"] = Field(
        description="Delta type. Always `thinking_delta`."
    )
    thinking: str = Field(description="Thinking content delta.")


# Ref: anthropic.types.signature_delta.SignatureDelta
class SignatureDelta(BaseModelResponse):
    """Signature delta for streaming updates."""

    type: Literal["signature_delta"] = Field(
        description="Delta type. Always `signature_delta`."
    )
    signature: str | None = Field(default=None, description="Signature content delta.")


# Ref: anthropic.types.raw_content_block_delta.RawContentBlockDelta
RawContentBlockDelta = Annotated[
    TextDelta | InputJSONDelta | CitationsDelta | ThinkingDelta | SignatureDelta,
    Field(discriminator="type"),
]


# Ref: anthropic.types.message_delta_usage.MessageDeltaUsage
class MessageDeltaUsage(BaseModelResponse):
    """Usage information for message delta streaming updates."""

    output_tokens: int = Field(
        description="The cumulative number of output tokens which were used."
    )
    cache_creation_input_tokens: int | None = Field(
        default=None,
        description="The cumulative number of input tokens used to create the cache entry.",
    )
    cache_read_input_tokens: int | None = Field(
        default=None,
        description="The cumulative number of input tokens read from the cache.",
    )
    input_tokens: int | None = Field(
        default=None,
        description="The cumulative number of input tokens which were used.",
    )
    server_tool_use: ServerToolUsage | None = Field(
        default=None, description="The number of server tool requests."
    )


# Ref: anthropic.types.raw_message_delta_event.RawMessageDeltaEvent.Delta
class MessageDelta(BaseModelResponse):
    """Message delta for streaming updates."""

    container: Container | None = Field(
        default=None,
        description="Information about the container used in this request.",
    )
    stop_reason: StopReason | None = Field(
        default=None, description="Stop reason if available."
    )
    stop_sequence: str | None = Field(
        default=None, description="Stop sequence if available."
    )


# Ref: anthropic.types.raw_content_block_start_event.RawContentBlockStartEvent
class RawContentBlockStartEvent(BaseModelResponse):
    """Stream event when a content block starts."""

    type: Literal["content_block_start"] = Field(
        description="Event type. Always `content_block_start`."
    )
    index: int = Field(description="Index of the content block.")
    content_block: ContentBlock = Field(description="The content block that started.")


# Ref: anthropic.types.raw_content_block_delta_event.RawContentBlockDeltaEvent
class RawContentBlockDeltaEvent(BaseModelResponse):
    """Stream event for content block delta."""

    type: Literal["content_block_delta"] = Field(
        description="Event type. Always `content_block_delta`."
    )
    index: int = Field(description="Index of the content block.")
    delta: RawContentBlockDelta = Field(
        description="Delta update for the content block."
    )


# Ref: anthropic.types.raw_content_block_stop_event.RawContentBlockStopEvent
class RawContentBlockStopEvent(BaseModelResponse):
    """Stream event when a content block stops."""

    type: Literal["content_block_stop"] = Field(
        description="Event type. Always `content_block_stop`."
    )
    index: int = Field(description="Index of the content block.")


# Ref: anthropic.types.raw_message_start_event.RawMessageStartEvent
class RawMessageStartEvent(BaseModelResponse):
    """Stream event when message starts."""

    type: Literal["message_start"] = Field(
        description="Event type. Always `message_start`."
    )
    message: Message = Field(description="The message object.")


# Ref: anthropic.types.raw_message_delta_event.RawMessageDeltaEvent
class RawMessageDeltaEvent(BaseModelResponse):
    """Stream event for message delta."""

    type: Literal["message_delta"] = Field(
        description="Event type. Always `message_delta`."
    )
    delta: MessageDelta = Field(description="Delta update for the message.")
    usage: MessageDeltaUsage = Field(
        description="""Billing and rate-limit usage.

    The API bills and rate-limits by token counts, as tokens represent the
    underlying cost to our systems.

    Under the hood, the API transforms requests into a format suitable for the
    model. The model's output then goes through a parsing stage before becoming an
    API response. As a result, the token counts in `usage` will not match one-to-one
    with the exact visible content of an API request or response.

    For example, `output_tokens` will be non-zero, even for an empty string response.

    Total input tokens in a request is the summation of `input_tokens`,
    `cache_creation_input_tokens`, and `cache_read_input_tokens`.
    """
    )


# Ref: anthropic.types.raw_message_stop_event.RawMessageStopEvent
class RawMessageStopEvent(BaseModelResponse):
    """Stream event when message stops."""

    type: Literal["message_stop"] = Field(
        description="Event type. Always `message_stop`."
    )


# Ref: anthropic.types.message_stream_event.MessageStreamEvent
MessageStreamEvent = Annotated[
    RawMessageStartEvent
    | RawMessageDeltaEvent
    | RawMessageStopEvent
    | RawContentBlockStartEvent
    | RawContentBlockDeltaEvent
    | RawContentBlockStopEvent,
    Field(discriminator="type"),
]


# Ref: anthropic.types.message_create_params.MessageCreateParamsBase
class MessageCreateParams(BaseModelRequestWithExtra):
    """Create message request following the Messages API specification."""

    model: str = Field(description="The model that will complete your prompt.")
    messages: list[MessageParam] = Field(
        description="""Input messages.

    Models are trained to operate on alternating `user` and `assistant`
    conversational turns. When creating a new `Message`, you specify the prior
    conversational turns with the `messages` parameter, and the model then generates
    the next `Message` in the conversation. Consecutive `user` or `assistant` turns
    in your request will be combined into a single turn.

    Each input message must be an object with a `role` and `content`. You can
    specify a single `user`-role message, or you can include multiple `user` and
    `assistant` messages.

    If the final message uses the `assistant` role, the response content will
    continue immediately from the content in that message. This can be used to
    constrain part of the model's response.

    Example with a single `user` message:

    ```json
    [{ "role": "user", "content": "Hello" }]
    ```

    Example with multiple conversational turns:

    ```json
    [
      { "role": "user", "content": "Hello there." },
      { "role": "assistant", "content": "Hi. How can I help you?" },
      { "role": "user", "content": "Can you explain LLMs in plain English?" }
    ]
    ```

    Example with a partially-filled response:

    ```json
    [
      {
        "role": "user",
        "content": "What's the Greek name for Sun? (A) Sol (B) Helios (C) Sun"
      },
      { "role": "assistant", "content": "The best answer is (" }
    ]
    ```

    Each input message `content` may be either a single `string` or an array of
    content blocks, where each block has a specific `type`. Using a `string` for
    `content` is shorthand for an array of one content block of type `"text"`. The
    following input messages are equivalent:

    ```json
    { "role": "user", "content": "Hello" }
    ```

    ```json
    { "role": "user", "content": [{ "type": "text", "text": "Hello" }] }
    ```

    Note that if you want to include a system prompt, you can use the
    top-level `system` parameter — there is no `"system"` role for input messages in
    the Messages API.
    """
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="""
    Top-level cache control automatically applies a cache_control marker to the last
    cacheable block in the request.
    """,
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        validation_alias=AliasChoices("max_tokens", "maxTokens"),
        description="""The maximum number of tokens to generate before stopping.

    Note that our models may stop _before_ reaching this maximum. This parameter
    only specifies the absolute maximum number of tokens to generate.

    Different models have different maximum values for this parameter.
    """,
    )
    inference_geo: str | None = Field(
        default=None,
        description="Specifies the geographic region for inference processing.\n"
        "UNSUPPORTED on this implementation. Data residency configuration is managed at server configuration level.",
    )
    metadata: MetadataParam | None = Field(
        default=None, description="Object describing metadata about the request."
    )
    output_config: OutputConfigParam | None = Field(
        default=None,
        description="Configuration options for the model's output, such as the output format.",
    )
    service_tier: ServiceTiers | None = Field(
        default=None,
        description="Determines whether to use priority capacity (if available) or standard capacity. "
        "See service-tiers documentation for details.",
    )
    stop_sequences: list[str] | None = Field(
        default=None,
        validation_alias=AliasChoices("stop_sequences", "stopSequences"),
        description="""Custom text sequences that will cause the model to stop generating.

    Models will normally stop when they have naturally completed their turn,
    which will result in a response `stop_reason` of `"end_turn"`.

    If you want the model to stop generating when it encounters custom strings of
    text, you can use the `stop_sequences` parameter. If the model encounters one of
    the custom sequences, the response `stop_reason` value will be `"stop_sequence"`
    and the response `stop_sequence` value will contain the matched stop sequence.
    """,
    )
    stream: bool = Field(
        default=False,
        description="Whether to incrementally stream the response using server-sent events (SSE).",
    )
    system: str | list[TextBlockParam] | None = Field(
        default=None,
        description="""System prompt.

    A system prompt is a way of providing context and instructions, such
    as specifying a particular goal or role.
    """,
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="""Amount of randomness injected into the response.

    Defaults to `1.0`. Ranges from `0.0` to `1.0`. Use `temperature` closer to `0.0`
    for analytical / multiple choice, and closer to `1.0` for creative and
    generative tasks.

    Note that even with `temperature` of `0.0`, the results will not be fully
    deterministic.
    """,
    )
    thinking: ThinkingConfigParam | None = Field(
        default=None,
        description="""Configuration for enabling extended thinking.

    When enabled, responses include `thinking` content blocks showing model's
    thinking process before the final answer.""",
    )
    tool_choice: ToolChoiceParam | None = Field(
        default=None,
        description="""How the model should use the provided tools.

    The model can use a specific tool, any available tool, decide by itself, or not
    use tools at all.
    """,
    )
    tools: list[ToolUnionParam] | None = Field(
        default=None,
        description="""Definitions of tools that the model may use.

    If you include `tools` in your API request, the model may return `tool_use`
    content blocks that represent the model's use of those tools. You can then run
    those tools using the tool input generated by the model and then optionally
    return results back to the model using `tool_result` content blocks.

    There are two types of tools: **client tools** and **server tools**. The
    behavior described below applies to client tools. For
    server tools, see their individual documentation as each has its own behavior.

    Each tool definition includes:

    - `name`: Name of the tool.
    - `description`: Optional, but strongly-recommended description of the tool.
    - `input_schema`: [JSON schema](https://json-schema.org/draft/2020-12) for the
      tool `input` shape that the model will produce in `tool_use` output content
      blocks.

    For example, if you defined `tools` as:

    ```json
    [
      {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol.",
        "input_schema": {
          "type": "object",
          "properties": {
            "ticker": {
              "type": "string",
              "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."
            }
          },
          "required": ["ticker"]
        }
      }
    ]
    ```

    And then asked the model "What's the S&P 500 at today?", the model might produce
    `tool_use` content blocks in the response like this:

    ```json
    [
      {
        "type": "tool_use",
        "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
        "name": "get_stock_price",
        "input": { "ticker": "^GSPC" }
      }
    ]
    ```

    You might then run your `get_stock_price` tool with `{"ticker": "^GSPC"}` as an
    input, and return the following back to the model in a subsequent `user`
    message:

    ```json
    [
      {
        "type": "tool_result",
        "tool_use_id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
        "content": "259.75 USD"
      }
    ]
    ```

    Tools can be used for workflows that include running client-side tools and
    functions, or more generally whenever you want the model to produce a particular
    JSON structure of output.""",
    )
    top_k: int | None = Field(
        default=None,
        ge=0,
        description="""Only sample from the top K options for each subsequent token.

    Used to remove "long tail" low probability responses.

    Recommended for advanced use cases only. You usually only need to use
    `temperature`.

    Note: top_k is not supported by The AWs Bedrock Converse API and is passed as = Field()
    extra arguments, some models may required passing this argument with a different name.
    """,
    )
    top_p: float | None = Field(
        default=None,
        validation_alias=AliasChoices("top_p", "topP"),
        ge=0.0,
        description="""Use nucleus sampling.

    In nucleus sampling, we compute the cumulative distribution over all the options
    for each subsequent token in decreasing probability order and cut it off once it
    reaches a particular probability specified by `top_p`. You should either alter
    `temperature` or `top_p`, but not both.

    Recommended for advanced use cases only. You usually only need to use
    `temperature`.
    """,
    )

    container: str | None = Field(
        default=None, description="Container identifier for reuse across requests."
    )


# Ref: anthropic.types.json_output_format_param.JSONOutputFormatParam
class JSONOutputFormatParam(BaseModelRequest):
    """JSON output format configuration."""

    type: Literal["json_schema"] = Field(
        description="Output format type. Always `json_schema`."
    )
    schema_: JsonMapping = Field(
        alias="schema", description="The JSON schema of the format."
    )


# Ref: anthropic.types.output_config_param.OutputConfigParam
class OutputConfigParam(BaseModelRequest):
    """Configuration options for the model's output."""

    effort: ThinkingEffort | None = Field(
        default=None, description="Effort level for the model's output processing."
    )
    format: JSONOutputFormatParam | None = Field(
        default=None,
        description="A schema to specify output format in responses. "
        "See structured outputs documentation.",
    )


# Ref: anthropic.types.thinking_config_enabled_param.ThinkingConfigEnabledParam
class ThinkingConfigEnabledParam(BaseModelRequest):
    """Enabled thinking configuration."""

    type: Literal["enabled"] = Field(
        description="Thinking config type. Always `enabled`."
    )
    budget_tokens: int = Field(
        description="Determines how many tokens the model can use for its internal reasoning process. "
        "Larger budgets can enable more thorough analysis for complex problems, improving response quality. "
        "Must be less than `max_tokens`."
    )


# Ref: anthropic.types.thinking_config_disabled_param.ThinkingConfigDisabledParam
class ThinkingConfigDisabledParam(BaseModelRequest):
    """Disabled thinking configuration."""

    type: Literal["disabled"] = Field(
        description="Thinking config type. Always `disabled`."
    )


# Ref: anthropic.types.thinking_config_adaptive_param.ThinkingConfigAdaptiveParam
class ThinkingConfigAdaptiveParam(BaseModelRequest):
    """Adaptive thinking configuration."""

    type: Literal["adaptive"] = Field(
        description="Thinking config type. Always `adaptive`."
    )


# Ref: anthropic.types.thinking_config_param.ThinkingConfigParam
ThinkingConfigParam = Annotated[
    ThinkingConfigEnabledParam
    | ThinkingConfigDisabledParam
    | ThinkingConfigAdaptiveParam,
    Field(discriminator="type"),
]


# Ref: anthropic.types.message_count_tokens_params.MessageCountTokensParams
class MessageCountTokensParams(BaseModelRequestWithExtra):
    """Count tokens request for the Messages API."""

    model: str = Field(description="The model that will complete your prompt.")
    messages: list[MessageParam] = Field(
        description="""Input messages.

    Models are trained to operate on alternating `user` and `assistant`
    conversational turns. When creating a new `Message`, you specify the prior
    conversational turns with the `messages` parameter, and the model then generates
    the next `Message` in the conversation. Consecutive `user` or `assistant` turns
    in your request will be combined into a single turn.

    Each input message must be an object with a `role` and `content`. You can
    specify a single `user`-role message, or you can include multiple `user` and
    `assistant` messages.

    If the final message uses the `assistant` role, the response content will
    continue immediately from the content in that message. This can be used to
    constrain part of the model's response.

    Example with a single `user` message:

    ```json
    [{ "role": "user", "content": "Hello" }]
    ```

    Example with multiple conversational turns:

    ```json
    [
      { "role": "user", "content": "Hello there." },
      { "role": "assistant", "content": "Hi. How can I help you?" },
      { "role": "user", "content": "Can you explain LLMs in plain English?" }
    ]
    ```

    Example with a partially-filled response:

    ```json
    [
      {
        "role": "user",
        "content": "What's the Greek name for Sun? (A) Sol (B) Helios (C) Sun"
      },
      { "role": "assistant", "content": "The best answer is (" }
    ]
    ```

    Each input message `content` may be either a single `string` or an array of
    content blocks, where each block has a specific `type`. Using a `string` for
    `content` is shorthand for an array of one content block of type `"text"`. The
    following input messages are equivalent:

    ```json
    { "role": "user", "content": "Hello" }
    ```

    ```json
    { "role": "user", "content": [{ "type": "text", "text": "Hello" }] }
    ```

    Note that if you want to include a system prompt, you can use the
    top-level `system` parameter — there is no `"system"` role for input messages in
    the Messages API.
    """
    )
    system: str | list[TextBlockParam] | None = Field(
        default=None,
        description="""System prompt.

    A system prompt is a way of providing context and instructions, such
    as specifying a particular goal or role.
    """,
    )
    tools: list[ToolUnionParam] | None = Field(
        default=None,
        description="""Definitions of tools that the model may use.

    If you include `tools` in your API request, the model may return `tool_use`
    content blocks that represent the model's use of those tools. You can then run
    those tools using the tool input generated by the model and then optionally
    return results back to the model using `tool_result` content blocks.

    There are two types of tools: **client tools** and **server tools**. The
    behavior described below applies to client tools. For
    server tools; see their individual documentation as each has its own behavior.

    Each tool definition includes:

    - `name`: Name of the tool.
    - `description`: Optional, but strongly-recommended description of the tool.
    - `input_schema`: [JSON schema](https://json-schema.org/draft/2020-12) for the
      tool `input` shape that the model will produce in `tool_use` output content
      blocks.

    For example, if you defined `tools` as:

    ```json
    [
      {
        "name": "get_stock_price",
        "description": "Get the current stock price for a given ticker symbol.",
        "input_schema": {
          "type": "object",
          "properties": {
            "ticker": {
              "type": "string",
              "description": "The stock ticker symbol, e.g. AAPL for Apple Inc."
            }
          },
          "required": ["ticker"]
        }
      }
    ]
    ```

    And then asked the model "What's the S&P 500 at today?", the model might produce
    `tool_use` content blocks in the response like this:

    ```json
    [
      {
        "type": "tool_use",
        "id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
        "name": "get_stock_price",
        "input": { "ticker": "^GSPC" }
      }
    ]
    ```

    You might then run your `get_stock_price` tool with `{"ticker": "^GSPC"}` as an
    input, and return the following back to the model in a subsequent `user`
    message:

    ```json
    [
      {
        "type": "tool_result",
        "tool_use_id": "toolu_01D7FLrfh4GYq7yT1ULFeyMV",
        "content": "259.75 USD"
      }
    ]
    ```

    Tools can be used for workflows that include running client-side tools and
    functions, or more generally whenever you want the model to produce a particular
    JSON structure of output.
    """,
    )
    tool_choice: ToolChoiceParam | None = Field(
        default=None,
        description="""How the model should use the provided tools.

    The model can use a specific tool, any available tool, decide by itself, or not
    use tools at all.
    """,
    )
    output_config: OutputConfigParam | None = Field(
        default=None,
        description="Configuration options for the model's output, such as the output format.",
    )
    thinking: ThinkingConfigParam | None = Field(
        default=None,
        description="""Configuration for enabling extended thinking.

    When enabled, responses include `thinking` content blocks showing models's
    thinking process before the final answer.
    """,
    )
    cache_control: CacheControlEphemeralParam | None = Field(
        default=None,
        description="Top-level cache control automatically applies a cache_control marker to the last cacheable block in the request.",
    )


# Ref: anthropic.types.message_tokens_count.MessageTokensCount
class MessageTokensCount(BaseModelResponse):
    """Token count response from the Messages API."""

    input_tokens: int = Field(
        description="The total number of tokens across the provided list of messages, "
        "system prompt, and tools."
    )


# Ref: anthropic.types.model_info.ModelInfo
class ModelInfo(BaseModelResponse):
    """Model information."""

    id: str = Field(description="Unique model identifier.")
    created_at: str = Field(
        description="RFC 3339 datetime string representing the time at which the model "
        "was released. May be set to an epoch value if the release date is unknown."
    )
    display_name: str = Field(description="A human-readable name for the model.")
    type: Literal["model"] = Field(
        default="model",
        description='Object type. For Models, this is always `"model"`.',
    )


# Ref: anthropic.pagination.SyncPage
class ModelListResponse(BaseModelResponse):
    """Paginated list of models."""

    data: list[ModelInfo] = Field(description="List of model objects.")
    has_more: bool = Field(
        default=False, description="Whether there are more results available."
    )
    first_id: str | None = Field(
        default=None, description="The ID of the first model in the list."
    )
    last_id: str | None = Field(
        default=None, description="The ID of the last model in the list."
    )
