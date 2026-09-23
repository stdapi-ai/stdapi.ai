"""A prompt over the model's context window is refused as each upstream API refuses it.

Chat Completions answers ``400 context_length_exceeded`` naming ``messages``;
Anthropic Messages answers ``invalid_request_error`` with ``prompt is too
long``. The backend's own wording never reaches the client. Streamed, the
same ``400`` is answered before the stream starts, even from a model refusing
only on its stream's first event. The live tests run unchanged against the
vendors and the gateway, on each lane's cheapest model with the smallest
context window, so the refused upload is small and costs nothing. An input
the window holds alone, but not beside its output, is served instead, and
billed.

Ref: https://developers.openai.com/api/docs/guides/error-codes
     https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
     https://platform.claude.com/docs/en/api/errors
     stdapi/monitoring.py:context_length_error
     stdapi/aws_bedrock.py:handle_bedrock_client_error
"""

from contextlib import contextmanager
from json import loads
from re import compile as re_compile
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from anthropic import BadRequestError as AnthropicBadRequestError
from openai import APIError, BadRequestError
from sse_starlette import EventSourceResponse, ServerSentEvent

from stdapi.api_errors import ApiError
from stdapi.api_providers.anthropic import TAG_ANTHROPIC
from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.aws_bedrock import handle_bedrock_client_error
from stdapi.aws_bedrock_mantle import MantleError, _map_error
from stdapi.models.chat._adapters import _openai_completion
from stdapi.models.chat._adapters._responses_context import (
    OUTPUT_BUDGET_TOO_LARGE,
    ContextLengthExceededError,
    ContextOverflow,
    collect_stream_open_errors,
    context_overflow,
    record_stream_open_error,
)
from stdapi.models.chat._adapters._stream_open import context_refusal, open_peeked
from stdapi.models.chat._default import ChatModel
from stdapi.models.chat._mantle import _default as mantle_default
from stdapi.monitoring import (
    REQUEST,
    context_length_error,
    log_request_sse_stream_event,
)
from stdapi.routes import (
    anthropic_messages,
    ollama_chat,
    ollama_generate,
    openai_chat_completions,
    openai_completions,
    openai_responses,
)
from stdapi.utils import to_json_str
from tests._helpers import make_client_error, make_model_details, ollama_route

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, AsyncIterator, Iterator

    from anthropic import Anthropic
    from openai import OpenAI
    from starlette.testclient import TestClient
    from types_aiobotocore_bedrock_runtime.type_defs import ConverseStreamOutputTypeDef

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef
    from stdapi.aws_http import SseEvent
    from stdapi.models import ModelDetails

#: Filler vocabulary: every tokenizer measured reads each word as about one token.
_WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]

#: Per lane (official API or not): the cheapest chat model with the smallest window, and that window.
_CHAT_MODELS: dict[bool, tuple[str, int]] = {
    True: ("gpt-5-nano", 272_000),
    False: ("meta.llama3-8b-instruct-v1:0", 8_192),
}

#: Per lane: the cheapest Anthropic Messages model with the smallest window, and that window.
_ANTHROPIC_MODELS: dict[bool, tuple[str, int]] = {
    True: ("claude-haiku-4-5-20251001", 200_000),
    False: ("meta.llama3-8b-instruct-v1:0", 8_192),
}

#: Per lane: the cheapest legacy Completions model with the smallest window, and that window.
_COMPLETION_MODELS: dict[bool, tuple[str, int]] = {
    True: ("gpt-3.5-turbo-instruct", 4_096),
    False: ("meta.llama3-8b-instruct-v1:0", 8_192),
}

#: Per lane: a cheap Chat Completions model whose refusal states the sizes, and its window.
_SIZED_MODELS: dict[bool, tuple[str, int]] = {
    True: ("gpt-5-nano", 272_000),
    False: ("anthropic.claude-haiku-4-5-20251001-v1:0", 200_000),
}

#: Per lane: the Anthropic Messages model whose refusal states the sizes, and its window.
_SIZED_ANTHROPIC_MODELS: dict[bool, tuple[str, int]] = {
    True: ("claude-haiku-4-5-20251001", 200_000),
    False: ("anthropic.claude-haiku-4-5-20251001-v1:0", 200_000),
}

#: Chat Completions' sized refusal, as the OpenAI API words it.
_MESSAGES_SIZED = re_compile(
    r"Input tokens exceed the configured limit of \d+ tokens\. Your messages "
    r"resulted in \d+ tokens\. Please reduce the length of the messages\."
)

#: Chat Completions' refusal of messages that fit, but not with the completion requested.
_COMPLETION_SIZED = re_compile(
    r"This model's maximum context length is \d+ tokens\. However, you requested "
    r"\d+ tokens \(\d+ in the messages, \d+ in the completion\)\. Please reduce "
    r"the length of the messages or completion\."
)

#: Per lane: a chat model refusing messages that fit only without the completion, the prompt words, and the completion tokens.
_COMBINED_MODELS: dict[bool, tuple[str, int, int]] = {
    True: ("gpt-4o-mini", 102_000, 16_000),
    False: ("writer.palmyra-vision-7b", 2_500, 2_000),
}

#: Per lane: a legacy Completions model refusing a prompt that fits only without the completion, the prompt words, and the completion tokens.
_COMBINED_COMPLETION_MODELS: dict[bool, tuple[str, int, int]] = {
    True: ("gpt-3.5-turbo-instruct", 3_000, 2_000),
    False: ("writer.palmyra-vision-7b", 2_500, 2_000),
}

#: Legacy Completions' refusal stating the prompt and completion tokens, as the OpenAI API words it.
_PROMPT_AND_COMPLETION_SIZED = re_compile(
    r"This model's maximum context length is \d+ tokens, however you requested "
    r"\d+ tokens \(\d+ in your prompt; \d+ for the completion\)\. Please reduce "
    r"your prompt; or completion length\."
)

#: Per lane: a Responses model, filler words leaving a few hundred tokens of its window, and the output tokens asked.
_CAPPED_RESPONSES_MODELS: dict[bool, tuple[str, int, int]] = {
    True: ("gpt-4o-mini", 113_500, 16_000),
    False: ("meta.llama3-8b-instruct-v1:0", 7_000, 2_048),
}

#: The Anthropic Messages model, filler words leaving about 60 tokens of its 200,000-token window, and the output tokens asked.
_CAPPED_ANTHROPIC_MODEL = ("claude-haiku-4-5-20251001", 177_680, 10_000)

#: Anthropic's sized refusal, as its API words it.
_PROMPT_SIZED = re_compile(r"prompt is too long: \d+ tokens > \d+ maximum")

#: A Bedrock refusal naming the sizes, as Claude words it.
_SIZED = (
    "The model returned the following errors: prompt is too long: 286028 tokens "
    "> 200000 maximum"
)

#: A Bedrock refusal wrapped in internal wording, as recorded for gpt-oss streams.
_LEAKY = (
    "Mantle streaming error for requestId 9293f38b-0000: ErrorEvent { error: "
    "APIError { message: \"Input length (195071) exceeds model's maximum context "
    'length (131072)." } }'
)

#: A refusal stating no sizes, as Llama words it.
_UNSIZED = (
    "The model returned the following errors: This model's maximum context "
    "length is 8192 tokens. Please reduce the length of the prompt"
)


#: A Mantle-served model's refusal, as Google Gemma 4 words it.
_MANTLE_GEMMA = (
    "prompt tokens (170029) exceed model maximum (131072) for google.gemma-4-e2b"
)

#: A Mantle-served model's refusal on the Responses API, as gpt-oss words it.
_MANTLE_ENGINE = (
    'ErrorEvent { error: APIError { type: "invalid_request_error", code: Some(400), '
    'message: "The engine prompt length 185692 exceeds the max_model_len 131072. '
    'Please reduce prompt.", param: Some("input") } }'
)

#: vLLM's refusal of a prompt that fits, but not with the output tokens requested.
_COMBINED = (
    "This model's maximum context length is 131072 tokens. However, you requested "
    "8192 output tokens and your prompt contains 125000 input tokens, for a total "
    "of 133192 tokens."
)

#: vLLM's refusal of a prompt over the input tokens the output requested leaves.
_COMBINED_UNSIZED = (
    "This model's maximum context length is 4096 tokens. However, you requested "
    "100 output tokens and your prompt contains at least 50000 characters (more "
    "than 15984 characters, which is the upper bound for 3996 input tokens)."
)

#: Palmyra's refusal of output tokens that alone fill the window.
_OUTPUT_ONLY = (
    "This model's maximum context length is 4096 tokens. However, you requested "
    "4096 output tokens and your prompt contains 196 characters. Please reduce "
    "the length of the input prompt or the number of requested output tokens."
)

#: vLLM's refusal of output tokens leaving no input tokens.
_OUTPUT_ONLY_BOUND = (
    "This model's maximum context length is 4096 tokens. However, you requested "
    "4096 output tokens and your prompt contains at least 1 characters (more than "
    "0 characters, which is the upper bound for 0 input tokens)."
)

#: Per refusal kind: the Converse message, and the Chat Completions error it becomes.
_KINDS: dict[str, tuple[str, str | None, str]] = {
    "prompt": (
        _SIZED,
        "context_length_exceeded",
        (
            "Input tokens exceed the configured limit of 200000 tokens. Your messages "
            "resulted in 286028 tokens. Please reduce the length of the messages."
        ),
    ),
    "combined": (
        _COMBINED,
        "context_length_exceeded",
        (
            "This model's maximum context length is 131072 tokens. However, you "
            "requested 133192 tokens (125000 in the messages, 8192 in the completion). "
            "Please reduce the length of the messages or completion."
        ),
    ),
    "output-only": (_OUTPUT_ONLY, None, OUTPUT_BUDGET_TOO_LARGE),
}


def _filler(words: int) -> str:
    """Return a prompt of *words* filler words.

    Args:
        words: Number of words.

    Returns:
        The prompt.
    """
    return " ".join(_WORDS[i % len(_WORDS)] for i in range(words))


def _prompt(window: int) -> str:
    """Return a prompt about 1.25 times larger than *window*.

    Args:
        window: Tokens the model's context window takes.

    Returns:
        The prompt.
    """
    return _filler(window * 5 // 4)


def _long_answer_prompt(words: int) -> str:
    """Return *words* filler words framed by a task needing more output than they leave.

    Args:
        words: Number of filler words.

    Returns:
        The prompt.
    """
    return (
        "Below is filler text for a load test; your task follows it.\n"
        + _filler(words)
        + "\nYour task: write a long, detailed story of at least 3000 words "
        "about a lighthouse keeper."
    )


@contextmanager
def _on_route(path: str, *tags: str) -> Iterator[None]:
    """Bind the current request to a route while the context is open.

    Args:
        path: The route path.
        *tags: The route's tags.

    Yields:
        None.
    """
    request: Any = SimpleNamespace(
        scope={"route": SimpleNamespace(path=path, tags=list(tags))}
    )
    token = REQUEST.set(request)
    try:
        yield
    finally:
        REQUEST.reset(token)


def _refusal(message: str = _SIZED) -> Exception:
    """Return a Converse context-window refusal.

    Args:
        message: The backend's message.

    Returns:
        The error.
    """
    return make_client_error("ValidationException", message=message)


class TestDialects:
    """Each API gets its own upstream wording, and the backend's is withheld.

    Ref: stdapi/monitoring.py:context_length_error
    """

    def test_chat_completions_names_messages_and_the_sizes(self) -> None:
        """Chat Completions' wording, with the sizes the backend stated.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
        """
        with _on_route("/v1/chat/completions", TAG_OPENAI):
            error = context_length_error(_refusal())
        assert isinstance(error, ContextLengthExceededError)
        assert (error.status, error.code, error.param) == (
            400,
            "context_length_exceeded",
            "messages",
        )
        assert str(error) == (
            "Input tokens exceed the configured limit of 200000 tokens. Your "
            "messages resulted in 286028 tokens. Please reduce the length of the "
            "messages."
        )

    def test_anthropic_messages_says_prompt_is_too_long(self) -> None:
        """Anthropic's wording, sized when the backend stated the sizes.

        Ref: https://platform.claude.com/docs/en/api/errors
        """
        with _on_route("/anthropic/v1/messages", TAG_ANTHROPIC):
            sized = context_length_error(_refusal())
            unsized = context_length_error(_refusal(_UNSIZED))
        assert str(sized) == "prompt is too long: 286028 tokens > 200000 maximum"
        assert str(unsized) == "prompt is too long"
        assert sized is not None
        assert sized.param is None

    @pytest.mark.parametrize(
        ("path", "param"), [("/v1/responses", "input"), ("/api/chat", None)]
    )
    def test_other_apis_keep_the_generic_wording(
        self, path: str, param: str | None
    ) -> None:
        """Other routes name the parameter holding the input, when they have one.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
        """
        with _on_route(path):
            error = context_length_error(_refusal(_LEAKY))
        assert error is not None
        assert error.param == param
        assert "Mantle" not in str(error)
        assert "exceeds the context window" in str(error)

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            (
                _COMBINED,
                (
                    "This model's maximum context length is 131072 tokens, however "
                    "you requested 133192 tokens (125000 in your prompt; 8192 for "
                    "the completion). Please reduce your prompt; or completion length."
                ),
            ),
            (
                _SIZED,
                (
                    "This model's maximum context length is 200000 tokens, however "
                    "your prompt is 286028 tokens. Please reduce your prompt; or "
                    "completion length."
                ),
            ),
            (
                _COMBINED_UNSIZED,
                (
                    "This model's maximum context length was exceeded by your prompt "
                    "and the requested completion. Please reduce your prompt; or "
                    "completion length."
                ),
            ),
            (
                _UNSIZED,
                (
                    "This model's maximum context length was exceeded by your prompt. "
                    "Please reduce your prompt; or completion length."
                ),
            ),
        ],
        ids=["all-sizes", "sized", "with-output", "unsized"],
    )
    def test_legacy_completions_word_it_as_that_api(
        self, message: str, expected: str
    ) -> None:
        """Legacy Completions states the tokens requested, and no code or parameter.

        Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
        """
        with _on_route("/v1/completions", TAG_OPENAI):
            error = context_length_error(_refusal(message))
        assert isinstance(error, ContextLengthExceededError)
        assert (error.status, error.code, error.param) == (400, None, None)
        assert str(error) == expected

    def test_the_sizes_survive_the_mapping(self) -> None:
        """A mapped error still carries the sizes a truncation retry plans with.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:context_overflow
        """
        with _on_route("/v1/responses"):
            error = context_length_error(_refusal())
        assert error is not None
        assert context_overflow(error) == ContextOverflow(286028, 200000)
        assert context_length_error(error) is None, "never mapped twice"

    @pytest.mark.parametrize(
        ("path", "message"),
        [
            ("/v1/embeddings", "Too many input tokens. Max input tokens: 8192."),
            ("/v1/rerank", "The query is too long."),
            ("/v1/chat/completions", "Malformed input request, please reformat."),
            ("/v1/chat/completions", "messages.0.content is too long for this field"),
        ],
        ids=["embeddings", "rerank", "malformed", "field-too-long"],
    )
    def test_near_misses_are_left_alone(self, path: str, message: str) -> None:
        """A validation error unrelated to the window keeps its own path.

        Ref: stdapi/monitoring.py:context_length_error
             stdapi/aws_bedrock.py:handle_bedrock_client_error
        """
        refusal = _refusal(message)
        with _on_route(path):
            assert context_length_error(refusal) is None
            with pytest.raises(type(refusal)) as excinfo, handle_bedrock_client_error():
                raise refusal
        assert excinfo.value is refusal

    @pytest.mark.parametrize(
        "message", [_OUTPUT_ONLY, _OUTPUT_ONLY_BOUND], ids=["palmyra", "vllm-bound-0"]
    )
    @pytest.mark.parametrize(
        ("path", "tags"),
        [
            ("/v1/chat/completions", (TAG_OPENAI,)),
            ("/v1/messages", (TAG_ANTHROPIC,)),
            ("/v1/responses", (TAG_OPENAI,)),
        ],
        ids=["chat", "anthropic", "responses"],
    )
    def test_output_tokens_filling_the_window_name_the_output_budget(
        self, message: str, path: str, tags: tuple[str, ...]
    ) -> None:
        """Output tokens filling the window alone leave no input to shorten.

        The client is told to lower its output limit, without the backend's text.

        Ref: stdapi/monitoring.py:context_length_error
        """
        with _on_route(path, *tags):
            error = context_length_error(_refusal(message))
        assert error is not None
        assert not isinstance(error, ContextLengthExceededError)
        assert (error.status, error.code, error.param) == (400, None, None)
        assert str(error) == OUTPUT_BUDGET_TOO_LARGE

    @pytest.mark.parametrize(
        ("path", "tags", "param", "message"),
        [
            ("/v1/chat/completions", (TAG_OPENAI,), "messages", _KINDS["combined"][2]),
            (
                "/v1/messages",
                (TAG_ANTHROPIC,),
                None,
                (
                    "input length and `max_tokens` exceed context limit: 125000 + 8192 > "
                    "131072, decrease input length or `max_tokens` and try again"
                ),
            ),
            (
                "/v1/responses",
                (TAG_OPENAI,),
                "input",
                (
                    "Your input exceeds the context window of this model. Please adjust "
                    "your input and try again."
                ),
            ),
        ],
        ids=["chat", "anthropic", "responses"],
    )
    def test_an_input_over_the_window_with_the_output_is_an_overflow(
        self, path: str, tags: tuple[str, ...], param: str | None, message: str
    ) -> None:
        """A prompt fitting alone, but not with its output, is ``context_length_exceeded``.

        Each API names the output budget as upstream does, and the sizes stay
        for a truncation retry.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
             https://platform.claude.com/docs/en/api/errors
        """
        with _on_route(path, *tags):
            error = context_length_error(_refusal(_COMBINED))
        assert isinstance(error, ContextLengthExceededError)
        assert (error.status, error.code, error.param) == (
            400,
            "context_length_exceeded",
            param,
        )
        assert str(error) == message
        assert error.overflow == ContextOverflow(125000, 131072, 8192)

    @pytest.mark.parametrize(
        ("path", "tags", "message"),
        [
            (
                "/v1/chat/completions",
                (TAG_OPENAI,),
                (
                    "The messages and the requested completion tokens exceed the context "
                    "window of this model. Please reduce the length of the messages or "
                    "completion."
                ),
            ),
            (
                "/v1/messages",
                (TAG_ANTHROPIC,),
                (
                    "input length and `max_tokens` exceed context limit, decrease input "
                    "length or `max_tokens` and try again"
                ),
            ),
        ],
        ids=["chat", "anthropic"],
    )
    def test_an_unsized_input_over_what_the_output_leaves_is_an_overflow(
        self, path: str, tags: tuple[str, ...], message: str
    ) -> None:
        """A prompt over the input tokens the output leaves names both.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
        """
        with _on_route(path, *tags):
            error = context_length_error(_refusal(_COMBINED_UNSIZED))
        assert isinstance(error, ContextLengthExceededError)
        assert error.code == "context_length_exceeded"
        assert str(error) == message


@pytest.mark.usefixtures("request_log")
class TestBackendMapping:
    """The shared Bedrock error mapping and stream guard apply the dialect wording.

    Ref: stdapi/aws_bedrock.py:handle_bedrock_client_error
         stdapi/monitoring.py:log_request_sse_stream_event
    """

    def test_a_refused_call_raises_the_dialect_error(self) -> None:
        """A Converse overflow becomes the caller's error, chained to the original.

        Ref: stdapi/aws_bedrock.py:handle_bedrock_client_error
        """
        refusal = _refusal(_LEAKY)
        with (
            _on_route("/v1/chat/completions", TAG_OPENAI),
            pytest.raises(ContextLengthExceededError) as excinfo,
            handle_bedrock_client_error(),
        ):
            raise refusal
        assert excinfo.value.param == "messages"
        assert excinfo.value.__cause__ is refusal
        assert "Mantle" not in str(excinfo.value)

    @pytest.mark.parametrize("kind", list(_KINDS))
    def test_each_kind_of_window_refusal_raises_its_error(self, kind: str) -> None:
        """A refused call raises the one classification, the backend text withheld.

        Ref: stdapi/aws_bedrock.py:handle_bedrock_client_error
        """
        message, code, client_message = _KINDS[kind]
        refusal = _refusal(message)
        with (
            _on_route("/v1/chat/completions", TAG_OPENAI),
            pytest.raises(ApiError) as excinfo,
            handle_bedrock_client_error(),
        ):
            raise refusal
        assert (excinfo.value.status, excinfo.value.code) == (400, code)
        assert str(excinfo.value) == client_message
        assert excinfo.value.__cause__ is refusal

    @pytest.mark.parametrize("kind", list(_KINDS))
    async def test_each_kind_of_window_refusal_ends_a_stream_with_its_error(
        self, kind: str
    ) -> None:
        """A stream refused on its first event reports the same classification.

        Ref: stdapi/monitoring.py:_stream_backend_error
        """
        message, code, client_message = _KINDS[kind]
        raised = make_client_error("validationException", message=message)

        async def _refused() -> AsyncGenerator[ServerSentEvent]:
            raise raised
            yield  # pragma: no cover - makes this an async generator

        with _on_route("/v1/chat/completions", TAG_OPENAI):
            events = [event async for event in log_request_sse_stream_event(_refused())]
        (event,) = events
        error = loads(str(event.data))["error"]
        assert error.get("code") == code
        assert error["message"] == client_message

    def test_an_unrelated_refusal_is_raised_unchanged(self) -> None:
        """Any other validation error keeps its own path.

        Ref: stdapi/aws_bedrock.py:handle_bedrock_client_error
        """
        refusal = _refusal("Malformed input.")
        with pytest.raises(type(refusal)) as excinfo, handle_bedrock_client_error():
            raise refusal
        assert excinfo.value is refusal

    @pytest.mark.parametrize(
        "raised",
        [make_client_error("validationException", message=_LEAKY), ApiError(_UNSIZED)],
        ids=["bedrock", "api-error"],
    )
    async def test_a_stream_refused_on_its_first_event_reports_the_dialect_error(
        self, raised: Exception
    ) -> None:
        """A stream that fails after its headers ends with the dialect's error event.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
             stdapi/monitoring.py:_stream_backend_error
        """

        async def _refused() -> AsyncGenerator[ServerSentEvent]:
            raise raised
            yield  # pragma: no cover - makes this an async generator

        with _on_route("/v1/chat/completions", TAG_OPENAI):
            events = [event async for event in log_request_sse_stream_event(_refused())]
        (event,) = events
        assert event.event == "error"
        error = loads(str(event.data))["error"]
        assert error["code"] == "context_length_exceeded"
        assert error["param"] == "messages"
        assert "Mantle" not in error["message"]


class _RefusingBackend:
    """Scripted Converse backend refusing output tokens the window cannot hold."""

    def __init__(self) -> None:
        #: Requests received, in order.
        self.requests: list[ConverseRequestBaseTypeDef] = []
        #: Wording of the refusal.
        self.message = _COMBINED
        #: Largest output budget accepted, or None to refuse every request.
        self.output_room: int | None = 131072 - 125000
        #: Why an answer stops.
        self.stop_reason = "end_turn"
        #: Whether a streamed refusal comes on the first event instead of at open.
        self.refuse_in_stream = False
        #: Error code of a refusal on the stream's first event.
        self.stream_error = "validationException"

    def _refuses(self, request: ConverseRequestBaseTypeDef) -> bool:
        output = request.get("inferenceConfig", {}).get("maxTokens")
        return self.output_room is None or output is None or output > self.output_room

    def _check(self, request: ConverseRequestBaseTypeDef) -> None:
        self.requests.append(request)
        if self._refuses(request):
            refusal = make_client_error("ValidationException", message=self.message)
            # Mapped as the real Converse call maps it.
            with handle_bedrock_client_error():
                raise refusal

    async def converse(self, request: ConverseRequestBaseTypeDef) -> dict[str, Any]:
        self._check(request)
        return {
            "output": {"message": {"role": "assistant", "content": [{"text": "OK"}]}},
            "stopReason": self.stop_reason,
            "usage": {
                "inputTokens": len(self.requests) * 100,
                "outputTokens": 1,
                "totalTokens": len(self.requests) * 100 + 1,
            },
        }

    async def converse_stream(
        self, request: ConverseRequestBaseTypeDef
    ) -> dict[str, Any]:
        refused = self.refuse_in_stream and self._refuses(request)
        if refused:
            self.requests.append(request)
        else:
            self._check(request)
        tokens = len(self.requests) * 100
        stop_reason = self.stop_reason
        refusal = make_client_error(self.stream_error, message=self.message)

        async def _events() -> AsyncGenerator[dict[str, Any]]:
            if refused:
                raise refusal
            for event in (
                {"messageStart": {"role": "assistant"}},
                {
                    "contentBlockDelta": {
                        "delta": {"text": "OK"},
                        "contentBlockIndex": 0,
                    }
                },
                {"contentBlockStop": {"contentBlockIndex": 0}},
                {"messageStop": {"stopReason": stop_reason}},
                {"metadata": {"usage": {"inputTokens": tokens, "outputTokens": 1}}},
            ):
                yield event

        return {"stream": _events()}


def _sse_events(text: str) -> list[dict[str, Any]]:
    """Decode a streamed response's event payloads.

    Args:
        text: The response body.

    Returns:
        The payloads, in order.
    """
    return [
        loads(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> _RefusingBackend:
    """Serve every chat route from the real Converse model over a scripted backend."""

    async def _validate_model(
        model_id: str, *_args: object, **_kwargs: object
    ) -> ModelDetails:
        return make_model_details(model_id)

    stub = _RefusingBackend()
    for route in (
        anthropic_messages,
        openai_chat_completions,
        openai_completions,
        ollama_chat,
        ollama_generate,
    ):
        monkeypatch.setattr(route, "validate_model", _validate_model)
        monkeypatch.setattr(route, "get_chat_model", ChatModel)
    monkeypatch.setattr(ChatModel, "converse", stub.converse)
    monkeypatch.setattr(ChatModel, "converse_stream", stub.converse_stream)
    return stub


@pytest.mark.local
class TestCappedOutputRetry:
    """An input the window holds alone, but not beside its output.

    Anthropic Messages serves it with the output capped, as upstream does;
    Chat Completions refuses it, as upstream does.

    Ref: https://platform.claude.com/docs/en/build-with-claude/context-windows
         https://developers.openai.com/api/docs/guides/error-codes
         stdapi/routes/anthropic_messages.py:create_message
    """

    @staticmethod
    def _message(client: TestClient, *, stream: bool) -> Any:  # noqa: ANN401
        return client.post(
            "/anthropic/v1/messages",
            json={
                "model": "amazon.nova-micro-v1:0",
                "max_tokens": 8192,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
            },
        )

    @pytest.mark.parametrize(
        ("backend_stop", "stop_reason"),
        [("end_turn", "end_turn"), ("max_tokens", "model_context_window_exceeded")],
        ids=["answered", "ran-out"],
    )
    @pytest.mark.parametrize("stream", [False, True])
    def test_anthropic_messages_answers_with_capped_output(
        self,
        app_client: TestClient,
        backend: _RefusingBackend,
        stream: bool,
        backend_stop: str,
        stop_reason: str,
    ) -> None:
        """One retry asks for what the window leaves; only the served call counts.

        An answer using all of that room has reached the context window, so it
        stops on ``model_context_window_exceeded``, the stop reason the
        Anthropic API defines for it; its own client limit was never reached.

        Ref: https://platform.claude.com/docs/en/build-with-claude/context-windows
             https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons
        """
        backend.stop_reason = backend_stop
        response = self._message(app_client, stream=stream)
        assert response.status_code == 200, response.text
        first, served = backend.requests
        assert first["inferenceConfig"]["maxTokens"] == 8192
        assert served["inferenceConfig"]["maxTokens"] == 131072 - 125000
        if stream:
            events = _sse_events(response.text)
            assert "error" not in [event["type"] for event in events]
            (delta,) = [event for event in events if event["type"] == "message_delta"]
            assert delta["delta"]["stop_reason"] == stop_reason
            assert "event: message_delta\r\ndata: " in response.text, (
                "SDKs dispatch on the event name"
            )
            assert delta["usage"]["input_tokens"] == 200, "the served call alone"
        else:
            body = response.json()
            assert body["content"][0]["text"] == "OK"
            assert body["stop_reason"] == stop_reason
            assert body["usage"]["input_tokens"] == 200, "the served call alone"

    def test_anthropic_messages_uncapped_answer_keeps_max_tokens(
        self, app_client: TestClient, backend: _RefusingBackend
    ) -> None:
        """An answer served on the first call keeps its own ``max_tokens`` stop.

        Ref: https://platform.claude.com/docs/en/build-with-claude/handling-stop-reasons
        """
        backend.output_room = 8192
        backend.stop_reason = "max_tokens"
        response = self._message(app_client, stream=False)
        assert response.status_code == 200, response.text
        assert response.json()["stop_reason"] == "max_tokens"
        assert len(backend.requests) == 1

    def test_anthropic_stream_refused_on_its_first_event_is_served_capped(
        self, app_client: TestClient, backend: _RefusingBackend
    ) -> None:
        """A refusal on the stream's first event is retried before the stream starts.

        Ref: https://platform.claude.com/docs/en/build-with-claude/context-windows
             stdapi/routes/anthropic_messages.py:create_message
        """
        backend.refuse_in_stream = True
        response = self._message(app_client, stream=True)
        assert response.status_code == 200, response.text
        events = _sse_events(response.text)
        assert "error" not in [event["type"] for event in events]
        assert events[-1]["type"] == "message_stop"
        first, served = backend.requests
        assert first["inferenceConfig"]["maxTokens"] == 8192
        assert served["inferenceConfig"]["maxTokens"] == 131072 - 125000

    @pytest.mark.parametrize(
        ("stream", "in_stream"),
        [(False, False), (True, False), (True, True)],
        ids=["unstreamed", "stream-open", "first-event"],
    )
    def test_anthropic_messages_refused_again_gets_the_refusal(
        self,
        app_client: TestClient,
        backend: _RefusingBackend,
        stream: bool,
        in_stream: bool,
    ) -> None:
        """The capped retry happens once; refused again, the usual error follows.

        Ref: https://platform.claude.com/docs/en/api/errors
        """
        backend.output_room = None
        backend.refuse_in_stream = in_stream
        response = self._message(app_client, stream=stream)
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert error["message"].startswith("input length and `max_tokens` exceed")
        assert len(backend.requests) == 2

    def test_output_tokens_filling_the_window_are_refused_once(
        self, app_client: TestClient, backend: _RefusingBackend
    ) -> None:
        """Output tokens filling the window alone get no retry and a plain 400.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:output_budget_exceeded
        """
        backend.message = _OUTPUT_ONLY
        backend.output_room = None
        response = self._message(app_client, stream=False)
        assert response.status_code == 400, response.text
        assert response.json()["error"]["message"] == OUTPUT_BUDGET_TOO_LARGE
        assert len(backend.requests) == 1

    @pytest.mark.parametrize(
        ("stream", "in_stream"),
        [(False, False), (True, False), (True, True)],
        ids=["unstreamed", "stream-open", "first-event"],
    )
    def test_chat_completions_still_refuses(
        self,
        app_client: TestClient,
        backend: _RefusingBackend,
        stream: bool,
        in_stream: bool,
    ) -> None:
        """Chat Completions answers ``context_length_exceeded`` naming the completion.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
        """
        backend.refuse_in_stream = in_stream
        response = app_client.post(
            "/v1/chat/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "max_completion_tokens": 8192,
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
            },
        )
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert (error["code"], error["param"]) == (
            "context_length_exceeded",
            "messages",
        )
        assert error["message"] == _KINDS["combined"][2]
        assert len(backend.requests) == 1


#: Per streamed route: its path, a streamed request, and the refusal the unstreamed request gets.
_STREAMED_ROUTES: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {
    "chat": (
        "/v1/chat/completions",
        {
            "max_completion_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
        },
        {
            "error": {
                "message": (
                    "Input tokens exceed the context window of this model. Please "
                    "reduce the length of the messages."
                ),
                "type": "invalid_request_error",
                "param": "messages",
                "code": "context_length_exceeded",
            }
        },
    ),
    "completions": (
        "/v1/completions",
        {"max_tokens": 16, "prompt": "hello"},
        {
            "error": {
                "message": (
                    "This model's maximum context length was exceeded by your "
                    "prompt. Please reduce your prompt; or completion length."
                ),
                "type": "invalid_request_error",
                "param": None,
                "code": None,
            }
        },
    ),
    "anthropic": (
        "/anthropic/v1/messages",
        {"max_tokens": 16, "messages": [{"role": "user", "content": "hello"}]},
        {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "prompt is too long"},
        },
    ),
    "ollama-chat": (
        ollama_route("/api/chat"),
        {
            "options": {"num_predict": 16},
            "messages": [{"role": "user", "content": "hello"}],
        },
        {
            "error": (
                "Your input exceeds the context window of this model. Please adjust "
                "your input and try again."
            )
        },
    ),
    "ollama-generate": (
        ollama_route("/api/generate"),
        {"options": {"num_predict": 16}, "prompt": "hello"},
        {
            "error": (
                "Your input exceeds the context window of this model. Please adjust "
                "your input and try again."
            )
        },
    ),
}


def _stream_on(client: TestClient, route: str) -> Any:  # noqa: ANN401
    """Send a streamed request on one of ``_STREAMED_ROUTES``.

    Args:
        client: The client.
        route: The route's key.

    Returns:
        The response.
    """
    path, body, _ = _STREAMED_ROUTES[route]
    return client.post(
        path, json={"model": "amazon.nova-micro-v1:0", "stream": True, **body}
    )


@pytest.mark.local
@pytest.mark.parametrize("route", list(_STREAMED_ROUTES))
class TestStreamRefusedOnItsFirstEvent:
    """A stream the backend refuses on its first event answers before any byte.

    Some models (Amazon Nova, Meta Llama) refuse an input over the window only
    on the stream's first event. Every streamed route reads it before sending
    its status, so the client gets the refusal the unstreamed request gets.

    Ref: https://developers.openai.com/api/docs/guides/error-codes
         https://platform.claude.com/docs/en/api/errors
         stdapi/models/chat/_adapters/_stream_open.py:open_peeked
    """

    def test_the_refusal_is_the_unstreamed_one(
        self, app_client: TestClient, backend: _RefusingBackend, route: str
    ) -> None:
        """A ``400`` in the route's own words, the backend's withheld, sent once.

        Ref: stdapi/models/chat/_adapters/_stream_open.py:context_refusal
        """
        backend.message = _UNSIZED
        backend.output_room = None
        backend.refuse_in_stream = True
        response = _stream_on(app_client, route)
        assert response.status_code == 400, response.text
        body = response.json()
        body.pop("request_id", None)
        assert body == _STREAMED_ROUTES[route][2]
        assert "8192" not in response.text
        assert len(backend.requests) == 1

    def test_output_tokens_filling_the_window_get_a_plain_400(
        self, app_client: TestClient, backend: _RefusingBackend, route: str
    ) -> None:
        """Output tokens filling the window alone are refused before the stream.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:output_budget_exceeded
        """
        backend.message = _OUTPUT_ONLY
        backend.output_room = None
        backend.refuse_in_stream = True
        response = _stream_on(app_client, route)
        assert response.status_code == 400, response.text
        assert OUTPUT_BUDGET_TOO_LARGE in response.text
        assert len(backend.requests) == 1

    def test_a_served_stream_keeps_its_first_event(
        self, app_client: TestClient, backend: _RefusingBackend, route: str
    ) -> None:
        """A stream the backend serves is relayed whole, its first event included.

        Ref: stdapi/models/chat/_adapters/_stream_open.py:replay_stream
        """
        backend.output_room = 1_000_000
        backend.refuse_in_stream = True
        response = _stream_on(app_client, route)
        assert response.status_code == 200, response.text
        assert "OK" in response.text
        assert "error" not in response.text
        assert len(backend.requests) == 1

    def test_an_error_other_than_the_window_is_streamed(
        self, app_client: TestClient, backend: _RefusingBackend, route: str
    ) -> None:
        """Any other error on the first event ends a ``200`` stream, never retried.

        Ref: stdapi/models/chat/_adapters/_stream_open.py:context_refusal
        """
        backend.message = (
            "Too many requests to arn:aws:bedrock:us-east-1:123456789012:model/x"
        )
        backend.output_room = None
        backend.refuse_in_stream = True
        backend.stream_error = "ThrottlingException"
        response = _stream_on(app_client, route)
        assert response.status_code == 200, response.text
        assert "error" in response.text
        assert "123456789012" not in response.text
        assert len(backend.requests) == 1


@pytest.mark.local
class TestBatchRefusedOnItsFirstEvent:
    """A streamed legacy Completions batch is refused when any prompt's stream is.

    Every prompt's stream is read in its own task before the first chunk.

    Ref: https://developers.openai.com/api/docs/guides/error-codes
         stdapi/models/chat/_adapters/_openai_completion.py:format_stream
    """

    @pytest.mark.parametrize("echo", [False, True])
    def test_one_refused_prompt_refuses_the_batch(
        self,
        app_client: TestClient,
        backend: _RefusingBackend,
        monkeypatch: pytest.MonkeyPatch,
        echo: bool,
    ) -> None:
        """The legacy ``400`` before any byte, and the served prompt's stream closed.

        Ref: stdapi/models/chat/_adapters/_stream_open.py:open_peeked
        """
        closed: list[bool] = []

        async def converse_stream(
            _model: ChatModel, request: ConverseRequestBaseTypeDef
        ) -> dict[str, Any]:
            refused = "oversized" in str(request)
            refusal = make_client_error("validationException", message=_UNSIZED)

            async def _events() -> AsyncGenerator[dict[str, Any]]:
                try:
                    if refused:
                        raise refusal
                    yield {"messageStart": {"role": "assistant"}}
                    yield {
                        "contentBlockDelta": {
                            "delta": {"text": "OK"},
                            "contentBlockIndex": 0,
                        }
                    }
                finally:
                    closed.append(refused)

            return {"stream": _events()}

        monkeypatch.setattr(ChatModel, "converse_stream", converse_stream)
        response = app_client.post(
            "/v1/completions",
            json={
                "model": "amazon.nova-micro-v1:0",
                "prompt": ["hello", "oversized"],
                "max_tokens": 16,
                "stream": True,
                "echo": echo,
            },
        )
        assert response.status_code == 400, response.text
        body = response.json()
        body.pop("request_id", None)
        assert body == _STREAMED_ROUTES["completions"][2]
        assert sorted(closed) == [False, True]


async def test_a_batch_closed_on_its_echo_closes_every_stream() -> None:
    """A batch closed before its streams are drained closes each one at once.

    Ref: stdapi/models/chat/_adapters/_openai_completion.py:format_stream
    """
    closed: list[int] = []

    async def events(index: int) -> AsyncGenerator[ConverseStreamOutputTypeDef]:
        """Yield content until closed, recording the close."""
        try:
            while True:
                yield {
                    "contentBlockDelta": {
                        "delta": {"text": "OK"},
                        "contentBlockIndex": 0,
                    }
                }
        finally:
            closed.append(index)

    # Held here, so only the batch can close them.
    streams: list[AsyncIterator[ConverseStreamOutputTypeDef]] = [events(0), events(1)]
    batch = _openai_completion.format_stream(
        "cmpl-1", 0, "model", streams, None, include_usage=False, echo_texts=["a", "b"]
    )
    await anext(batch)
    await batch.aclose()
    assert sorted(closed) == [0, 1]


class TestLiveRefusals:
    """A prompt over the window, sent to each vendor and to the gateway.

    Ref: https://developers.openai.com/api/docs/guides/error-codes
         https://platform.claude.com/docs/en/api/errors
    """

    @pytest.mark.slow
    def test_chat_completions(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Chat Completions answers ``400 context_length_exceeded`` on ``messages``.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
        """
        model, window = _CHAT_MODELS[use_official_api]
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _prompt(window)}],
                max_completion_tokens=16,
            )
        error = excinfo.value
        assert error.code == "context_length_exceeded"
        assert error.param == "messages"
        assert isinstance(error.body, dict)
        assert error.body["type"] == "invalid_request_error"
        assert "The model returned" not in error.message

    @pytest.mark.slow
    def test_chat_completions_states_the_sizes(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """A refusal stating its sizes is worded as the OpenAI API words it.

        Claude states the sizes in its refusal, which is not billed.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
        """
        model, window = _SIZED_MODELS[use_official_api]
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _prompt(window)}],
                max_completion_tokens=16,
            )
        assert excinfo.value.code == "context_length_exceeded"
        assert isinstance(excinfo.value.body, dict)
        assert _MESSAGES_SIZED.fullmatch(excinfo.value.body["message"])

    @pytest.mark.slow
    def test_chat_completions_names_the_completion(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Messages fitting alone, but not with the completion, are an overflow too.

        The message states both sizes; Palmyra bounds the messages at the
        input tokens the completion leaves, plus one. Refused before
        generation on either lane, so it costs nothing.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
        """
        model, words, completion = _COMBINED_MODELS[use_official_api]
        prompt = _filler(words)
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=completion,
            )
        error = excinfo.value
        assert error.code == "context_length_exceeded"
        assert error.param == "messages"
        assert isinstance(error.body, dict)
        assert _COMPLETION_SIZED.fullmatch(error.body["message"]), error.body

    @pytest.mark.slow
    def test_chat_completions_streamed(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Streamed, the refusal is the same ``400``, answered before the stream starts.

        The gateway's Llama refuses only on the stream's first event, which is
        read before the status is sent.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
             stdapi/models/chat/_adapters/_stream_open.py:open_peeked
        """
        model, window = _CHAT_MODELS[use_official_api]
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _prompt(window)}],
                max_completion_tokens=16,
                stream=True,
            )
        error = excinfo.value
        assert error.code == "context_length_exceeded"
        assert error.param == "messages"
        assert isinstance(error.body, dict)
        assert error.body["type"] == "invalid_request_error"
        assert "The model returned" not in error.message

    @pytest.mark.slow
    def test_completions_streamed(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """A streamed legacy completion over the window is a ``400`` before the stream.

        The gateway's Llama refuses only on the stream's first event, and states
        no sizes, so only the wording's frame is asserted.

        Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
             stdapi/models/chat/_adapters/_stream_open.py:open_peeked
        """
        model, window = _COMPLETION_MODELS[use_official_api]
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.completions.create(
                model=model, prompt=_prompt(window), max_tokens=16, stream=True
            )
        error = excinfo.value
        assert (error.code, error.param) == (None, None)
        assert isinstance(error.body, dict)
        assert error.body["type"] == "invalid_request_error"
        message = error.body["message"]
        assert message.startswith("This model's maximum context length ")
        assert message.endswith("Please reduce your prompt; or completion length.")

    @pytest.mark.slow
    def test_completions_states_the_tokens_requested(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Legacy Completions states the prompt and completion tokens, with no code.

        Refused before generation on either lane, so it costs nothing.

        Ref: https://developers.openai.com/api/reference/resources/completions/methods/create
        """
        model, words, completion = _COMBINED_COMPLETION_MODELS[use_official_api]
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.completions.create(
                model=model, prompt=_filler(words), max_tokens=completion
            )
        error = excinfo.value
        assert (error.code, error.param) == (None, None)
        assert isinstance(error.body, dict)
        assert error.body["type"] == "invalid_request_error"
        assert _PROMPT_AND_COMPLETION_SIZED.fullmatch(error.body["message"]), error.body

    @pytest.mark.slow
    def test_anthropic_messages(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Anthropic Messages answers ``invalid_request_error``: prompt is too long.

        Ref: https://platform.claude.com/docs/en/api/errors
        """
        model, window = _ANTHROPIC_MODELS[use_official_api]
        with pytest.raises(AnthropicBadRequestError) as excinfo:
            anthropic_client.messages.create(
                model=model,
                max_tokens=16,
                messages=[{"role": "user", "content": _prompt(window)}],
            )
        assert isinstance(excinfo.value.body, dict)
        error = excinfo.value.body["error"]
        assert error["type"] == "invalid_request_error"
        assert error["message"].startswith("prompt is too long")

    @pytest.mark.slow
    def test_anthropic_messages_states_the_sizes(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """A sized refusal reads ``prompt is too long: M tokens > N maximum``.

        Ref: https://platform.claude.com/docs/en/api/errors
        """
        model, window = _SIZED_ANTHROPIC_MODELS[use_official_api]
        with pytest.raises(AnthropicBadRequestError) as excinfo:
            anthropic_client.messages.create(
                model=model,
                max_tokens=16,
                messages=[{"role": "user", "content": _prompt(window)}],
            )
        assert isinstance(excinfo.value.body, dict)
        assert _PROMPT_SIZED.fullmatch(excinfo.value.body["error"]["message"])

    @pytest.mark.slow
    def test_anthropic_messages_streamed(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """Streamed, the refusal is the same ``400``, answered before the stream starts.

        The gateway's Llama refuses only on the stream's first event, which is
        read before the status is sent.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/models/chat/_adapters/_stream_open.py:open_peeked
        """
        model, window = _ANTHROPIC_MODELS[use_official_api]
        with (
            pytest.raises(AnthropicBadRequestError) as excinfo,
            anthropic_client.messages.stream(
                model=model,
                max_tokens=16,
                messages=[{"role": "user", "content": _prompt(window)}],
            ) as stream,
        ):
            for _ in stream:
                pass
        assert isinstance(excinfo.value.body, dict)
        error = excinfo.value.body["error"]
        assert error["type"] == "invalid_request_error"
        assert error["message"].startswith("prompt is too long")


class TestLiveCappedOutput:
    """An input the window holds alone, but not beside its output, is served.

    The prompt asks for more output than the room the window leaves.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         https://platform.claude.com/docs/en/build-with-claude/context-windows
    """

    @pytest.mark.slow
    def test_responses(self, openai_client: OpenAI, use_official_api: bool) -> None:
        """The answer runs out at the window: ``incomplete`` on ``max_output_tokens``.

        The response echoes the requested ``max_output_tokens``, not the room.
        This pins what the client sees, whether the gateway lane's model serves
        the pair itself or refuses it; ``TestCappedOutputRetry`` proves the
        retry.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        model, words, output = _CAPPED_RESPONSES_MODELS[use_official_api]
        response = openai_client.responses.create(
            model=model, input=_long_answer_prompt(words), max_output_tokens=output
        )
        assert response.max_output_tokens == output
        assert response.status == "incomplete"
        assert response.incomplete_details is not None
        assert response.incomplete_details.reason == "max_output_tokens"
        assert response.usage is not None
        assert 0 < response.usage.output_tokens < output

    @pytest.mark.slow
    @pytest.mark.expensive
    def test_anthropic_messages(
        self, anthropic_client: Anthropic, use_official_api: bool
    ) -> None:
        """The answer is served whole, past the 200,000-token window.

        Claude Haiku 4.5 answered 199,936 input tokens with 4,256 output
        tokens and ``end_turn`` (2026-09-23). Only the official API runs it:
        no gateway model refuses the pair cheaply, so ``TestCappedOutputRetry``
        proves the gateway's side, which caps the output at the window.

        Ref: https://platform.claude.com/docs/en/build-with-claude/context-windows
        """
        if not use_official_api:
            pytest.skip("No gateway model refuses the pair cheaply")
        model, words, output = _CAPPED_ANTHROPIC_MODEL
        message = anthropic_client.messages.create(
            model=model,
            max_tokens=output,
            messages=[{"role": "user", "content": _long_answer_prompt(words)}],
        )
        assert message.stop_reason == "end_turn", message.usage
        assert message.usage.input_tokens + message.usage.output_tokens > 200_000


async def test_window_stop_relay_closes_the_stream_it_relays() -> None:
    """Closing the capped answer's relay closes the backend stream at once.

    Ref: stdapi/routes/anthropic_messages.py:_window_stop_events
    """
    closed: list[bool] = []

    async def events() -> AsyncGenerator[ServerSentEvent]:
        """Yield events until closed, recording the close."""
        try:
            while True:
                yield ServerSentEvent("{}", event="ping")
        finally:
            closed.append(True)

    relay = anthropic_messages._window_stop_events(events())  # noqa: SLF001
    await anext(relay)
    await relay.aclose()
    assert closed == [True]


class TestOpenPeeked:
    """A route reads its stream's first event before committing its status.

    Ref: stdapi/models/chat/_adapters/_stream_open.py:open_peeked
    """

    @staticmethod
    async def _opened(events: AsyncGenerator[ServerSentEvent]) -> EventSourceResponse:
        """Answer a model call with a stream of *events*."""
        return EventSourceResponse(events)

    @staticmethod
    async def _payloads(result: EventSourceResponse) -> list[object]:
        """Read what a stream sends, each event as its payload."""
        return [
            event.data if isinstance(event, ServerSentEvent) else event
            async for event in result.body_iterator
        ]

    async def test_only_the_first_event_is_read_before_returning(self) -> None:
        """The status waits for one event, and the stream replays it first.

        Ref: stdapi/models/chat/_adapters/_stream_open.py:replay_stream
        """
        reads: list[int] = []

        async def events() -> AsyncGenerator[ServerSentEvent]:
            """Yield three events, recording each read."""
            for index in range(3):
                reads.append(index)
                yield ServerSentEvent(str(index))

        result = await open_peeked(self._opened(events()), context_refusal)
        assert reads == [0]
        assert await self._payloads(result) == ["0", "1", "2"]

    async def test_a_refused_stream_is_closed_and_its_refusal_raised(self) -> None:
        """The refusal replaces the stream, which is closed at once.

        Ref: stdapi/models/chat/_adapters/_stream_open.py:close_stream
        """
        closed: list[bool] = []
        refused = ValueError("refused on the first event")
        answered = ApiError("answered")

        async def events() -> AsyncGenerator[ServerSentEvent]:
            """Report a refusal on the first event, recording the close."""
            try:
                record_stream_open_error(refused)
                yield ServerSentEvent("error")
                yield ServerSentEvent("never read")  # pragma: no cover
            finally:
                closed.append(True)

        with pytest.raises(ApiError) as excinfo:
            await open_peeked(
                self._opened(events()),
                lambda error: answered if error is refused else None,
            )
        assert excinfo.value is answered
        assert closed == [True]

    async def test_an_error_nothing_refuses_is_streamed(self) -> None:
        """An error the refusal does not claim stays the stream's own.

        Ref: stdapi/models/chat/_adapters/_stream_open.py:open_peeked
        """

        async def events() -> AsyncGenerator[ServerSentEvent]:
            """Report an unrelated error on the first event."""
            record_stream_open_error(ValueError("throttled"))
            yield ServerSentEvent("error")

        result = await open_peeked(self._opened(events()), lambda _error: None)
        assert await self._payloads(result) == ["error"]

    async def test_a_response_is_returned_as_is(self) -> None:
        """An unstreamed answer has no event to read.

        Ref: stdapi/models/chat/_adapters/_stream_open.py:open_peeked
        """
        answer = object()

        async def call() -> object:
            """Answer without streaming."""
            return answer

        assert await open_peeked(call(), context_refusal) is answer


def test_context_refusal_words_the_calling_api_and_logs_the_backend(
    request_log: dict[str, Any],
) -> None:
    """A window refusal gets the route's words; the backend's go to the log only.

    Ref: stdapi/models/chat/_adapters/_stream_open.py:context_refusal
    """
    with _on_route("/v1/chat/completions", TAG_OPENAI):
        refused = context_refusal(
            make_client_error("validationException", message=_UNSIZED)
        )
    assert isinstance(refused, ContextLengthExceededError)
    assert refused.param == "messages"
    assert "8192" not in str(refused)
    assert "8192 tokens" in str(request_log)
    mapped = ContextLengthExceededError(ContextOverflow(286028, 200000))
    assert context_refusal(mapped) is mapped
    assert context_refusal(ValueError("unrelated")) is None


@pytest.mark.parametrize(
    ("message", "sizes"),
    [
        (_MANTLE_GEMMA, ContextOverflow(170029, 131072)),
        (_MANTLE_ENGINE, ContextOverflow(185692, 131072)),
    ],
    ids=["gemma", "engine"],
)
def test_mantle_refusals_are_recognized_with_their_sizes(
    message: str, sizes: ContextOverflow
) -> None:
    """The refusals Mantle-served models word their own way are overflows too.

    Ref: stdapi/models/chat/_adapters/_responses_context.py:context_overflow
    """
    assert context_overflow(MantleError(message, status=400)) == sizes


#: The Mantle-served model the scripted Mantle endpoint answers for.
_MANTLE_MODEL = "test.mantle-overflow"

#: The upstream APIs a Mantle-served model can serve a request on.
_MANTLE_APIS = ("chat_completions", "responses", "messages")

#: A Chat Completions chunk opening a stream with the assistant role only.
_ROLE_CHUNK = to_json_str(
    {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}
)


class _MantleEndpoint:
    """Scripted Mantle endpoint refusing every request as Gemma 4 does."""

    def __init__(self) -> None:
        #: Upstream API the model serves.
        self.api = "chat_completions"
        #: Whether a stream opens, then refuses in an error event after its opening events.
        self.refuse_in_stream = False
        #: Requests received.
        self.requests = 0
        #: Wording of the refusal.
        self.message = _MANTLE_GEMMA

    def _refusal(self) -> MantleError:
        return _map_error(
            400,
            to_json_str(
                {"error": {"code": "validation_error", "message": self.message}}
            ),
            "us-east-1",
        )

    async def invoke(self, *_args: object, **_kwargs: object) -> dict[str, Any]:
        self.requests += 1
        if self.api == "responses":
            # A Responses answer can fail instead of raising.
            return {
                "object": "response",
                "status": "failed",
                "error": {"code": "invalid_prompt", "message": self.message},
                "output": [],
            }
        raise self._refusal()

    async def invoke_stream(
        self, *_args: object, **_kwargs: object
    ) -> AsyncGenerator[SseEvent]:
        self.requests += 1
        if not self.refuse_in_stream:
            raise self._refusal()
        events = _refusing_stream(self.api, self.message)

        async def _events() -> AsyncGenerator[SseEvent]:
            for event in events:
                yield event

        return _events()


#: Per route: the refusal of the Gemma 4 wording, as the unstreamed request gets it.
_MANTLE_REFUSALS: dict[str, dict[str, Any]] = {
    "chat": {
        "error": {
            "message": (
                "Input tokens exceed the configured limit of 131072 tokens. Your "
                "messages resulted in 170029 tokens. Please reduce the length of "
                "the messages."
            ),
            "type": "invalid_request_error",
            "param": "messages",
            "code": "context_length_exceeded",
        }
    },
    "completions": {
        "error": {
            "message": (
                "This model's maximum context length is 131072 tokens, however your "
                "prompt is 170029 tokens. Please reduce your prompt; or completion "
                "length."
            ),
            "type": "invalid_request_error",
            "param": None,
            "code": None,
        }
    },
    "anthropic": {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "prompt is too long: 170029 tokens > 131072 maximum",
        },
    },
    "ollama-chat": {
        "error": (
            "Your input exceeds the context window of this model. Please adjust "
            "your input and try again."
        )
    },
    "ollama-generate": {
        "error": (
            "Your input exceeds the context window of this model. Please adjust "
            "your input and try again."
        )
    },
}


@pytest.mark.local
class TestMantleRefusals:
    """A Mantle-served model's refusal is each API's own, streamed or not.

    Mantle answers a refusal either on the call or, streamed, in an error
    event after the stream opened; the backend's wording, which names the
    model, never reaches the client.

    Ref: https://developers.openai.com/api/docs/guides/error-codes
         https://platform.claude.com/docs/en/api/errors
         stdapi/models/chat/_mantle/_default.py:ChatModel._serve
    """

    @pytest.fixture
    def endpoint(self, monkeypatch: pytest.MonkeyPatch) -> _MantleEndpoint:
        """Serve every chat route from a Mantle model over a scripted endpoint."""

        async def _validate_model(
            model_id: str, *_args: object, **_kwargs: object
        ) -> ModelDetails:
            return make_model_details(model_id)

        stub = _MantleEndpoint()
        for route in (
            anthropic_messages,
            openai_chat_completions,
            openai_completions,
            ollama_chat,
            ollama_generate,
        ):
            monkeypatch.setattr(route, "validate_model", _validate_model)
            monkeypatch.setattr(route, "get_chat_model", mantle_default.ChatModel)
        monkeypatch.setitem(
            mantle_default._LEARNED_APIS,  # noqa: SLF001
            _MANTLE_MODEL,
            frozenset({"chat_completions"}),
        )
        monkeypatch.setattr(mantle_default, "invoke", stub.invoke)
        monkeypatch.setattr(mantle_default, "invoke_stream", stub.invoke_stream)
        return stub

    @pytest.mark.parametrize(
        ("stream", "in_stream"),
        [(False, False), (True, False), (True, True)],
        ids=["unstreamed", "stream-open", "in-stream"],
    )
    @pytest.mark.parametrize("route", list(_MANTLE_REFUSALS))
    @pytest.mark.parametrize("api", _MANTLE_APIS)
    def test_the_refusal_is_the_api_own(
        self,
        app_client: TestClient,
        endpoint: _MantleEndpoint,
        monkeypatch: pytest.MonkeyPatch,
        api: str,
        route: str,
        stream: bool,
        in_stream: bool,
    ) -> None:
        """A ``400`` in the route's words, before any byte when streamed.

        Whatever upstream API serves it, a failed Responses answer included.

        Ref: stdapi/models/chat/_mantle/_default.py:_read_past_preamble
             stdapi/models/chat/_mantle/_default.py:_failed_response_error
        """
        monkeypatch.setitem(
            mantle_default._LEARNED_APIS,  # noqa: SLF001
            _MANTLE_MODEL,
            frozenset({api}),
        )
        endpoint.api = api
        endpoint.refuse_in_stream = in_stream
        path, body, _ = _STREAMED_ROUTES[route]
        response = app_client.post(
            path, json={"model": _MANTLE_MODEL, "stream": stream, **body}
        )
        assert response.status_code == 400, response.text
        payload = response.json()
        payload.pop("request_id", None)
        assert payload == _MANTLE_REFUSALS[route]
        assert "gemma" not in response.text
        assert endpoint.requests == 1

    @pytest.mark.parametrize("route", list(_MANTLE_REFUSALS))
    def test_an_error_other_than_the_window_is_streamed(
        self, app_client: TestClient, endpoint: _MantleEndpoint, route: str
    ) -> None:
        """Any other error event after the stream opened ends a ``200`` stream.

        Ref: stdapi/models/chat/_adapters/_stream_open.py:context_refusal
        """
        endpoint.refuse_in_stream = True
        endpoint.message = (
            "Engine failure on arn:aws:bedrock:us-east-1:123456789012:model/x"
        )
        path, body, _ = _STREAMED_ROUTES[route]
        response = app_client.post(
            path, json={"model": _MANTLE_MODEL, "stream": True, **body}
        )
        assert response.status_code == 200, response.text
        assert "error" in response.text
        assert "123456789012" not in response.text
        assert endpoint.requests == 1

    def test_a_responses_retry_refused_in_stream_fails_as_the_first_attempt(
        self,
        app_client: TestClient,
        endpoint: _MantleEndpoint,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The capped retry refused again in its stream fails as upstream streams it.

        The Responses request is converted to another upstream API, whose own
        stream failure would otherwise be a server error.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
             stdapi/routes/_responses_context.py:_open_once
        """

        async def _validate_model(
            model_id: str, *_args: object, **_kwargs: object
        ) -> ModelDetails:
            return make_model_details(model_id)

        monkeypatch.setattr(openai_responses, "validate_model", _validate_model)
        monkeypatch.setattr(
            openai_responses, "get_chat_model", mantle_default.ChatModel
        )
        endpoint.refuse_in_stream = True
        endpoint.message = _COMBINED
        response = app_client.post(
            "/v1/responses",
            json={
                "model": _MANTLE_MODEL,
                "input": "hello",
                "max_output_tokens": 8192,
                "stream": True,
            },
        )
        assert response.status_code == 200, response.text
        events = _sse_events(response.text)
        assert [event["type"] for event in events] == [
            "response.created",
            "response.in_progress",
            "error",
            "response.failed",
        ]
        assert events[2]["code"] == "context_length_exceeded"
        assert events[3]["response"]["error"]["code"] == "context_length_exceeded"
        assert endpoint.requests == 2, "the capped retry, once"


def _mantle_events(*events: tuple[str | None, dict[str, Any]]) -> list[SseEvent]:
    """Serialize scripted upstream events.

    Args:
        *events: Event names and payloads.

    Returns:
        The events, as the Mantle transport yields them.
    """
    return [(name, to_json_str(payload)) for name, payload in events]


def _refusing_stream(api: str, message: str) -> list[SseEvent]:
    """Script an upstream stream opening, then refusing in an error event.

    Args:
        api: Upstream Mantle API serving the stream.
        message: Wording of the refusal.

    Returns:
        The events, as the Mantle transport yields them.
    """
    if api == "responses":
        # Some streams name their events in the payload only.
        return _mantle_events(
            (None, {"type": "response.created", "response": {"error": None}}),
            (None, {"type": "response.in_progress", "response": {"error": None}}),
            (None, {"type": "error", "code": "invalid_prompt", "message": message}),
        )
    if api == "messages":
        return _mantle_events(
            ("message_start", {"type": "message_start", "message": {}}),
            ("ping", {"type": "ping"}),
            (
                "error",
                {"type": "error", "error": {"type": "api_error", "message": message}},
            ),
        )
    return [
        (None, _ROLE_CHUNK),
        *_mantle_events(
            (None, {"error": {"code": "validation_error", "message": message}})
        ),
    ]


#: Per upstream API: a stream opening, then refusing in an error event.
_REFUSING_STREAMS: dict[str, list[SseEvent]] = {
    api: _refusing_stream(api, _MANTLE_GEMMA) for api in _MANTLE_APIS
}

#: Per upstream API: a stream opening, its first content event, then one more event.
_ANSWERING_STREAMS: dict[str, list[SseEvent]] = {
    "chat_completions": [
        (None, _ROLE_CHUNK),
        *_mantle_events(
            (None, {"choices": [{"index": 0, "delta": {"content": "Hi"}}]}),
            (None, {"choices": [{"index": 0, "delta": {"content": "!"}}]}),
        ),
    ],
    "responses": _mantle_events(
        (None, {"type": "response.created", "response": {"error": None}}),
        (None, {"type": "response.in_progress", "response": {"error": None}}),
        (None, {"type": "response.output_item.added", "item": {"type": "message"}}),
        (None, {"type": "response.output_text.delta", "delta": "Hi"}),
    ),
    "messages": _mantle_events(
        ("message_start", {"type": "message_start", "message": {}}),
        ("ping", {"type": "ping"}),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {}},
        ),
        ("content_block_delta", {"type": "content_block_delta", "index": 0}),
    ),
}


class TestMantleReadAhead:
    """A Mantle stream is read past its opening events before its status is sent.

    Ref: stdapi/models/chat/_mantle/_default.py:_read_past_preamble
    """

    @staticmethod
    def _counted(events: list[SseEvent], reads: list[int]) -> AsyncGenerator[SseEvent]:
        """Yield *events*, counting each read into *reads*."""

        async def _stream() -> AsyncGenerator[SseEvent]:
            for index, event in enumerate(events):
                reads.append(index)
                yield event

        return _stream()

    @pytest.mark.parametrize("api", list(_REFUSING_STREAMS))
    async def test_a_refusal_after_the_opening_events_is_reported(
        self,
        api: Any,  # noqa: ANN401
    ) -> None:
        """The refusal reaches the opener, and the stream still replays whole.

        Ref: stdapi/models/chat/_adapters/_responses_context.py:collect_stream_open_errors
        """
        events = _REFUSING_STREAMS[api]
        with collect_stream_open_errors() as errors:
            replay = await mantle_default._read_past_preamble(  # noqa: SLF001
                api, self._counted(events, [])
            )
        (error,) = errors
        assert context_overflow(error) == ContextOverflow(170029, 131072)
        assert [event async for event in replay] == events

    @pytest.mark.parametrize("api", list(_ANSWERING_STREAMS))
    async def test_content_stops_the_read_ahead(
        self,
        api: Any,  # noqa: ANN401
    ) -> None:
        """A stream answering is read up to its first content event, and reports nothing.

        Ref: stdapi/models/chat/_mantle/_default.py:_opens_stream
        """
        events = _ANSWERING_STREAMS[api]
        reads: list[int] = []
        with collect_stream_open_errors() as errors:
            replay = await mantle_default._read_past_preamble(  # noqa: SLF001
                api, self._counted(events, reads)
            )
        assert (reads, errors) == (list(range(len(events) - 1)), [])
        assert [event async for event in replay] == events

    async def test_a_failing_stream_raises_on_replay(self) -> None:
        """An error reading ahead is reported, and raised after the events read.

        Ref: stdapi/models/chat/_adapters/_stream_open.py:replay_stream
        """
        failure = MantleError("interrupted", status=502)

        async def _failing() -> AsyncGenerator[SseEvent]:
            yield None, _ROLE_CHUNK
            raise failure

        with collect_stream_open_errors() as errors:
            replay = await mantle_default._read_past_preamble(  # noqa: SLF001
                "chat_completions", _failing()
            )
        assert errors == [failure]
        assert await anext(replay) == (None, _ROLE_CHUNK)
        with pytest.raises(MantleError):
            await anext(replay)


class TestMantleRelayedRefusal:
    """A refusal relayed as stream events reads as the calling API words it.

    A streamed Responses request keeps its events, as upstream streams them.

    Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
         stdapi/models/chat/_mantle/_default.py:_scrub_error_event
    """

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            (
                {
                    "error": {
                        "code": "invalid_prompt",
                        "message": _MANTLE_GEMMA,
                        "param": None,
                        "type": "invalid_request_error",
                    },
                    "type": "error",
                },
                {
                    "error": {
                        "code": "context_length_exceeded",
                        "message": (
                            "Your input exceeds the context window of this model. "
                            "Please adjust your input and try again."
                        ),
                        "param": "input",
                        "type": "invalid_request_error",
                    },
                    "type": "error",
                },
            ),
            (
                {"type": "error", "code": "invalid_prompt", "message": _MANTLE_GEMMA},
                {
                    "type": "error",
                    "code": "context_length_exceeded",
                    "message": (
                        "Your input exceeds the context window of this model. "
                        "Please adjust your input and try again."
                    ),
                },
            ),
            (
                {
                    "type": "response.failed",
                    "response": {
                        "status": "failed",
                        "error": {"code": "invalid_prompt", "message": _MANTLE_GEMMA},
                    },
                },
                {
                    "type": "response.failed",
                    "response": {
                        "status": "failed",
                        "error": {
                            "code": "context_length_exceeded",
                            "message": (
                                "Your input exceeds the context window of this "
                                "model. Please adjust your input and try again."
                            ),
                        },
                    },
                },
            ),
        ],
        ids=["error", "top-level-error", "response-failed"],
    )
    def test_responses_events_carry_upstream_refusal(
        self, payload: dict[str, Any], expected: dict[str, Any]
    ) -> None:
        """Each event keeps its shape, with upstream's code, parameter and words.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
        """
        with _on_route("/v1/responses", TAG_OPENAI):
            relayed = mantle_default._scrub_error_event(  # noqa: SLF001
                to_json_str(payload)
            )
        assert loads(relayed) == expected

    def test_a_failed_response_refusal_is_the_api_error(self) -> None:
        """An unstreamed failed answer refusing the input is the API's ``400``.

        Ref: stdapi/models/chat/_mantle/_default.py:_failed_response_error
        """
        raw = {"status": "failed", "error": {"message": _MANTLE_ENGINE}}
        with _on_route("/anthropic/v1/messages", TAG_ANTHROPIC):
            error = mantle_default._failed_response_error(raw)  # noqa: SLF001
        assert isinstance(error, ContextLengthExceededError)
        assert str(error) == "prompt is too long: 185692 tokens > 131072 maximum"
        other = mantle_default._failed_response_error(  # noqa: SLF001
            {"status": "failed", "error": {"message": "The model crashed."}}
        )
        assert other.status == 502


#: The cheapest Mantle-served chat model, and its context window.
_MANTLE_LIVE_MODEL = ("google.gemma-4-e2b", 131_072)


@pytest.mark.slow
@pytest.mark.gateway("Only a gateway serves Bedrock Mantle models")
class TestLiveMantleRefusals:
    """A prompt over the window, sent to a Mantle-served model.

    Gemma 4 refuses it in its own words, naming itself, on the call or in an
    error event after the stream opened; refused, it costs nothing.

    Ref: https://developers.openai.com/api/docs/guides/error-codes
         https://platform.claude.com/docs/en/api/errors
         stdapi/models/chat/_mantle/_default.py:ChatModel._serve
    """

    @pytest.mark.parametrize("stream", [False, True])
    def test_chat_completions(self, openai_client: OpenAI, stream: bool) -> None:
        """Chat Completions answers ``400 context_length_exceeded`` before any byte.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
        """
        model, window = _MANTLE_LIVE_MODEL
        with pytest.raises(BadRequestError) as excinfo:
            openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _prompt(window)}],
                max_completion_tokens=16,
                stream=stream,
            )
        error = excinfo.value
        assert (error.code, error.param) == ("context_length_exceeded", "messages")
        assert isinstance(error.body, dict)
        assert _MESSAGES_SIZED.fullmatch(error.body["message"]), error.body

    @pytest.mark.parametrize("stream", [False, True])
    def test_anthropic_messages(
        self, anthropic_client: Anthropic, stream: bool
    ) -> None:
        """Anthropic Messages answers ``prompt is too long`` before any byte.

        Ref: https://platform.claude.com/docs/en/api/errors
        """
        model, window = _MANTLE_LIVE_MODEL
        with pytest.raises(AnthropicBadRequestError) as excinfo:
            anthropic_client.messages.create(
                model=model,
                max_tokens=16,
                messages=[{"role": "user", "content": _prompt(window)}],
                stream=stream,
            )
        assert isinstance(excinfo.value.body, dict)
        assert _PROMPT_SIZED.fullmatch(excinfo.value.body["error"]["message"])

    def test_responses_streamed(self, openai_client: OpenAI) -> None:
        """A streamed Responses request gets upstream's ``error`` event; the SDK raises it.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
        """
        model, window = _MANTLE_LIVE_MODEL
        stream = openai_client.responses.create(
            model=model, input=_prompt(window), max_output_tokens=16, stream=True
        )
        with pytest.raises(APIError) as excinfo:
            for _ in stream:
                pass
        error = excinfo.value
        assert error.code == "context_length_exceeded"
        assert str(error) == (
            "Your input exceeds the context window of this model. Please adjust "
            "your input and try again."
        )
