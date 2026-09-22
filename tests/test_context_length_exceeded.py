"""A prompt over the model's context window is refused as each upstream API refuses it.

Chat Completions answers ``400 context_length_exceeded`` naming ``messages``;
Anthropic Messages answers ``invalid_request_error`` with ``prompt is too
long``. The backend's own wording never reaches the client. Streamed, the
same error reaches the client as the stream's error, which the official SDKs
raise. The live tests run unchanged against the vendors and the gateway, on
each lane's cheapest model with the smallest context window, so the refused
upload is small and costs nothing. An input the window holds alone, but not
beside its output, is served instead, and billed.

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
from anthropic import APIStatusError as AnthropicAPIStatusError
from anthropic import BadRequestError as AnthropicBadRequestError
from openai import APIError, BadRequestError
from sse_starlette import ServerSentEvent

from stdapi.api_errors import ApiError
from stdapi.api_providers.anthropic import TAG_ANTHROPIC
from stdapi.api_providers.openai import TAG_OPENAI
from stdapi.aws_bedrock import handle_bedrock_client_error
from stdapi.models.chat._adapters._responses_context import (
    OUTPUT_BUDGET_TOO_LARGE,
    ContextLengthExceededError,
    ContextOverflow,
    context_overflow,
)
from stdapi.models.chat._default import ChatModel
from stdapi.monitoring import (
    REQUEST,
    context_length_error,
    log_request_sse_stream_event,
)
from stdapi.routes import anthropic_messages, openai_chat_completions
from tests._helpers import make_client_error, make_model_details

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterator

    from anthropic import Anthropic
    from openai import OpenAI
    from starlette.testclient import TestClient

    from stdapi.aws_bedrock import ConverseRequestBaseTypeDef
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
        refusal = make_client_error("validationException", message=self.message)

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


@pytest.mark.local
class TestCappedOutputRetry:
    """An input the window holds alone, but not beside its output.

    Anthropic Messages serves it with the output capped, as upstream does;
    Chat Completions refuses it, as upstream does.

    Ref: https://platform.claude.com/docs/en/build-with-claude/context-windows
         https://developers.openai.com/api/docs/guides/error-codes
         stdapi/routes/anthropic_messages.py:create_message
    """

    @pytest.fixture
    def backend(self, monkeypatch: pytest.MonkeyPatch) -> _RefusingBackend:
        """Serve both routes from the real Converse model over a scripted backend."""

        async def _validate_model(
            model_id: str, *_args: object, **_kwargs: object
        ) -> ModelDetails:
            return make_model_details(model_id)

        stub = _RefusingBackend()
        for route in (anthropic_messages, openai_chat_completions):
            monkeypatch.setattr(route, "validate_model", _validate_model)
            monkeypatch.setattr(route, "get_chat_model", ChatModel)
        monkeypatch.setattr(ChatModel, "converse", stub.converse)
        monkeypatch.setattr(ChatModel, "converse_stream", stub.converse_stream)
        return stub

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

    def test_anthropic_stream_refused_after_its_headers_is_not_retried(
        self, app_client: TestClient, backend: _RefusingBackend
    ) -> None:
        """A refusal on the stream's first event ends the stream; nothing is resent.

        The headers are out, so the refusal is the stream's ``error`` event.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
             stdapi/routes/anthropic_messages.py:create_message
        """
        backend.refuse_in_stream = True
        response = self._message(app_client, stream=True)
        assert response.status_code == 200
        (error,) = [e for e in _sse_events(response.text) if e["type"] == "error"]
        assert error["error"]["type"] == "invalid_request_error"
        assert error["error"]["message"].startswith(
            "input length and `max_tokens` exceed"
        )
        assert len(backend.requests) == 1

    @pytest.mark.parametrize("stream", [False, True])
    def test_anthropic_messages_refused_again_gets_the_refusal(
        self, app_client: TestClient, backend: _RefusingBackend, stream: bool
    ) -> None:
        """The capped retry happens once; refused again, the usual error follows.

        Ref: https://platform.claude.com/docs/en/api/errors
        """
        backend.output_room = None
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

    @pytest.mark.parametrize("stream", [False, True])
    def test_chat_completions_still_refuses(
        self, app_client: TestClient, backend: _RefusingBackend, stream: bool
    ) -> None:
        """Chat Completions answers ``context_length_exceeded`` naming the completion.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
        """
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
        """Streamed, the SDK raises the refusal, as a 400 or as the stream's error.

        The official API refuses before the stream starts: ``BadRequestError``.
        The gateway's Llama refuses on the stream's first event, after the 200,
        so the SDK raises its base ``APIError`` carrying the same error.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events
        """
        model, window = _CHAT_MODELS[use_official_api]
        with pytest.raises(APIError) as excinfo:  # noqa: PT012
            stream = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": _prompt(window)}],
                max_completion_tokens=16,
                stream=True,
            )
            for _ in stream:
                pass
        error = excinfo.value
        assert type(error) is (BadRequestError if use_official_api else APIError)
        assert error.code == "context_length_exceeded"
        assert isinstance(error.body, dict)
        assert error.body["param"] == "messages"
        assert "The model returned" not in error.message

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
        """Streamed, the SDK raises the refusal, as a 400 or as the stream's error.

        The official API refuses before the stream starts: ``BadRequestError``.
        The gateway's Llama refuses after the 200, so the SDK raises an
        ``APIStatusError`` whose status is that 200.

        Ref: https://platform.claude.com/docs/en/build-with-claude/streaming
        """
        model, window = _ANTHROPIC_MODELS[use_official_api]
        with (
            pytest.raises(AnthropicAPIStatusError) as excinfo,
            anthropic_client.messages.stream(
                model=model,
                max_tokens=16,
                messages=[{"role": "user", "content": _prompt(window)}],
            ) as stream,
        ):
            for _ in stream:
                pass
        error = excinfo.value
        if use_official_api:
            assert isinstance(error, AnthropicBadRequestError)
        else:
            assert type(error) is AnthropicAPIStatusError
            assert error.status_code == 200
        assert "prompt is too long" in str(error)


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
