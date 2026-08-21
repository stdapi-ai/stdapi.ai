"""OpenAI GPT chat model implementation."""

from re import compile as re_compile

from stdapi.models.chat._adapters._common import NoServerTools
from stdapi.models.chat._default import ChatModel as _BaseChatModel


class ChatModel(_BaseChatModel):
    """OpenAI GPT-specific chat model implementation.

    Amazon Bedrock serves the OpenAI server tools (web search, code interpreter)
    on the Bedrock Mantle endpoint only: on ``bedrock-runtime`` the Responses API
    answers ``400 "web search is not supported for this request"`` and Converse
    answers ``400 "This model doesn't support the systemTool field"``.  The
    GPT-5.6 models are served by both endpoints and resolve to this runtime class
    by default, so a server tool has to be refused here rather than forwarded as
    a function tool no client can answer (issue #186).

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html
    """

    __slots__ = ()

    MATCHER = "openai.gpt-"
    ALIAS_MATCHER = re_compile(r"^openai\.(.+?)(?:-1:0)?$")

    #: No server tool is served for the OpenAI models on the bedrock-runtime endpoint.
    SERVER_TOOLS_UNSERVED = NoServerTools(
        "Amazon Bedrock serves the OpenAI server tools on the Bedrock Mantle "
        "endpoint only, and this model is served by bedrock-runtime. Route it "
        "to Mantle by naming it in AWS_BEDROCK_MANTLE_PREFERRED_MODELS, or by "
        "sending the 'x-stdapi-service: bedrock-mantle' header where "
        "AWS_BEDROCK_MANTLE_SERVICE_HEADER enables it."
    )
