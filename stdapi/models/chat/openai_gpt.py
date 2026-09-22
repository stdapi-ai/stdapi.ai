"""OpenAI GPT chat model implementation."""

from re import compile as re_compile

from stdapi.models.chat._adapters._common import NoServerTools
from stdapi.models.chat._reasoning_effort import ReasoningEffortChatModel

#: GPT Astra models, with or without a ``daybreak-<edition>-`` qualifier, which reject effort ``none``.
ALWAYS_REASONING_MATCHER = re_compile(r"^openai\.gpt-(?:daybreak-\w+-)?[\d.]+-astra")


class ChatModel(ReasoningEffortChatModel):
    """OpenAI GPT-specific chat model implementation.

    Reasoning is set by ``additionalModelRequestFields.reasoning.effort``: probed
    on GPT-5.6 and GPT-6, it takes ``none`` through ``max`` (``low`` through
    ``max`` on GPT-6 Astra) and rejects ``minimal``, and the flat
    ``reasoning_effort`` and ``thinking`` fields are rejected as unknown
    parameters.

    Amazon Bedrock serves the OpenAI server tools (web search, code interpreter)
    on the Bedrock Mantle endpoint only: on ``bedrock-runtime`` the Responses API
    answers ``400 "web search is not supported for this request"`` and Converse
    answers ``400 "This model doesn't support the systemTool field"``.  The
    GPT-5.6 models are served by both endpoints and reach this runtime class
    where the deployment has taken them out of
    ``aws_bedrock_mantle_preferred_models``, where Mantle is disabled or no
    configured region serves it, and for a request whose API key carries an AWS
    credential of its own, which cannot pay for a Mantle call and is pinned to
    the runtime twin. A server tool has to be refused here rather than forwarded
    as a function tool no client can answer (issue #186).

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/web-search.html
    """

    __slots__ = ()

    MATCHER = "openai.gpt-"
    ALIAS_MATCHER = re_compile(r"^openai\.(.+?)(?:-1:0)?$")

    #: No server tool is served for the OpenAI models on the bedrock-runtime endpoint.
    SERVER_TOOLS_UNSERVED = NoServerTools(
        "Amazon Bedrock serves the OpenAI server tools on the Bedrock Mantle "
        "endpoint only, and this model was served by bedrock-runtime. Route it "
        "to Mantle by naming it in AWS_BEDROCK_MANTLE_PREFERRED_MODELS, or by "
        "sending the 'x-stdapi-service: bedrock-mantle' header where "
        "AWS_BEDROCK_MANTLE_SERVICE_HEADER enables it. Neither applies while "
        "Amazon Bedrock Guardrails are configured, nor to an API key carrying "
        "an AWS credential of its own: both are always served by bedrock-runtime."
    )

    @property
    def REASONING_DISABLE_SUPPORTED(self) -> bool:  # type: ignore[override]  # noqa: N802
        """Whether the model accepts ``reasoning.effort: "none"``."""
        return ALWAYS_REASONING_MATCHER.match(self._model_id) is None
