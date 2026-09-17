"""Ollama-compatible POST /api/generate, driven by the official ``ollama`` client.

The served subset is ``prompt`` with ``system`` and ``images``. ``raw``,
``suffix``, ``template`` and ``context`` all need the model's own tokenizer and
prompt template, which a hosted backend does not expose, so each is refused
rather than quietly ignored: dropping any of them answers something other than
what was asked for. Ollama Cloud serves ``suffix``, ``template`` and
``context``, so that refusal is asserted on both targets rather than only
where it holds; ``raw`` is a gateway-only assertion, since the cheapest cloud
chat model refuses it too, for an unrelated, model-specific reason.

Ref: https://docs.ollama.com/api/generate
     stdapi/routes/ollama_generate.py:generate
     stdapi/models/chat/_adapters/_ollama.py:generate_stream
"""

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import ollama
import pytest

from stdapi.models.chat._adapters import _ollama as ollama_adapter
from stdapi.routes import ollama_generate as generate_route
from stdapi.types.openai_chat_completions import (
    ChatCompletionUserMessageParam,
    CompletionCreateParams,
)
from tests._helpers import make_model_details, ollama_route

if TYPE_CHECKING:
    from starlette.testclient import TestClient

# Keeps the shared answer below on one worker, so it is requested once.
pytestmark = pytest.mark.xdist_group("ollama_generate")

#: Prompt short enough to keep a live answer cheap, long enough to stream.
_PROMPT = "Count from 1 to 5."

#: Option values Ollama reads as "off": an unseeded answer, no candidate limit.
_OPTION_SENTINELS: dict[str, Any] = {"seed": -1, "top_k": 0}

#: The same kind of value, on the three options Ollama Cloud alone refuses.
_CLOUD_REFUSED_SENTINELS: list[tuple[str, float, str]] = [
    ("num_predict", -1, "max_tokens must be positive"),
    ("top_p", 0, "Input should be greater than 0"),
    ("temperature", -0.5, "Input should be greater than or equal to 0"),
]

#: The four fields that need the model's own prompt template and tokenizer.
PROMPT_LEVEL_FIELDS: list[tuple[str, object]] = [
    ("suffix", "the end"),
    ("template", "{{ .Prompt }}"),
    ("context", [1, 2, 3]),
    ("raw", True),
]


@pytest.fixture(scope="module")
def buffered_generate(
    ollama_client: ollama.Client, ollama_chat_model: str
) -> ollama.GenerateResponse:
    """One buffered generate answer, shared by the tests that only read it.

    Returns:
        The complete generate response.
    """
    return ollama_client.generate(model=ollama_chat_model, prompt=_PROMPT, stream=False)


def test_generate_answers_a_prompt(
    buffered_generate: ollama.GenerateResponse, ollama_chat_model: str
) -> None:
    """A buffered generate answers one object carrying the generated text.

    Ref: https://docs.ollama.com/api/generate
    """
    assert buffered_generate.model == ollama_chat_model
    assert buffered_generate.response
    assert buffered_generate.done is True
    assert buffered_generate.done_reason == "stop"
    assert buffered_generate.created_at


def test_generate_reports_only_the_metrics_it_measured(
    buffered_generate: ollama.GenerateResponse,
) -> None:
    """Token counts are reported; the load phase that never happened is not.

    Ref: https://docs.ollama.com/api/usage
    """
    assert buffered_generate.prompt_eval_count
    assert buffered_generate.prompt_eval_count > 0
    assert buffered_generate.eval_count
    assert buffered_generate.eval_count > 0
    assert buffered_generate.load_duration is None


def test_generate_applies_the_system_prompt(
    ollama_client: ollama.Client, ollama_chat_model: str
) -> None:
    """`system` reaches the model as its system instruction.

    Ref: https://docs.ollama.com/openapi.yaml (GenerateRequest.system)
    """
    answer = ollama_client.generate(
        model=ollama_chat_model,
        system="You always answer with the single word BANANA.",
        prompt="What is the capital of France?",
        stream=False,
    )
    assert answer.response
    assert "banana" in answer.response.lower()


def test_generate_serves_an_empty_format(
    ollama_client: ollama.Client, ollama_chat_model: str
) -> None:
    """An empty `format` asks for no structured output and is answered normally.

    The official client types `format` as the empty string, `"json"` or a
    schema, and clients built on it send the empty string as their "no
    structured output" value on every call, so it has to be served rather than
    refused.

    Ref: https://docs.ollama.com/openapi.yaml (GenerateRequest.format)
         ollama/_types.py:154 (BaseGenerateRequest.format)
    """
    answer = ollama_client.generate(
        model=ollama_chat_model,
        prompt="The capital of France.",
        format="",
        stream=False,
    )
    assert answer.done is True
    assert answer.response


def test_generate_streams_to_a_terminal_done_event(
    ollama_client: ollama.Client, ollama_chat_model: str, use_official_api: bool
) -> None:
    """The client walks the stream to a terminal event carrying the metrics.

    The gateway measures the prompt/generation split off its own stream; Ollama
    Cloud reports neither duration even when streaming, so only the counts and
    the assembled text are asserted on both targets.

    Ref: https://docs.ollama.com/api/generate
    """
    parts = list(
        ollama_client.generate(model=ollama_chat_model, prompt=_PROMPT, stream=True)
    )
    assert len(parts) > 1
    assert "".join(part.response or "" for part in parts)
    terminal = parts[-1]
    assert terminal.done is True
    assert terminal.eval_count
    assert terminal.load_duration is None
    if not use_official_api:
        assert terminal.prompt_eval_duration


@pytest.mark.parametrize(("field", "value"), PROMPT_LEVEL_FIELDS)
def test_generate_refuses_the_prompt_level_fields(
    ollama_client: ollama.Client,
    ollama_chat_model: str,
    use_official_api: bool,
    field: str,
    value: object,
) -> None:
    """Each field needing the model's prompt template is refused with a reason.

    A deliberate divergence, asserted on both targets so it stays one: Ollama
    Cloud runs the model's own template and serves ``suffix``, ``template`` and
    ``context``, while this gateway reaches the backend through a chat API that
    exposes neither the template nor the tokenizer, and says so rather than
    silently answering something else. ``raw`` is skipped on the official
    target: ``gpt-oss:20b``, the cheapest Ollama Cloud chat model, is rendered
    through Ollama's Harmony pipeline, which raw mode bypasses entirely, so
    Ollama Cloud refuses ``raw`` for this model specifically ("does not
    currently support raw mode") -- a model-level restriction, not evidence
    that upstream dropped the parameter.

    Ref: stdapi/types/ollama.py:GenerateRequest._reject_prompt_level_fields
         https://docs.ollama.com/api/generate
    """
    request: dict[str, Any] = {
        "model": ollama_chat_model,
        "prompt": "hi",
        "stream": False,
        "options": {"num_predict": 1},
        field: value,
    }
    if use_official_api:
        if field == "raw":
            pytest.skip(
                'ollama.com refuses raw mode for "gpt-oss:20b" itself (its '
                "Harmony renderer requires templating), so this model cannot "
                "prove the divergence for the raw field"
            )
        assert ollama_client.generate(**request).done is True
        return
    with pytest.raises(ollama.ResponseError) as raised:
        ollama_client.generate(**request)
    assert raised.value.status_code == 400
    assert field in raised.value.error
    assert "/api/chat" in raised.value.error


def test_generate_answers_the_option_values_that_mean_off(
    ollama_client: ollama.Client, ollama_chat_model: str
) -> None:
    """The sentinel option values Ollama defines are answered, never refused.

    An Ollama server starts from ``seed: -1``, and reads a non-positive
    ``top_k`` as "off" rather than as an out-of-range value, so a client
    forwarding its own defaults expects an answer. Both targets answer them.

    Ref: https://github.com/ollama/ollama/blob/main/api/types.go (DefaultOptions)
         https://github.com/ollama/ollama/blob/main/mlxrunner/sample/sample.go
    """
    answer = ollama_client.generate(
        model=ollama_chat_model,
        prompt="Say hello.",
        stream=False,
        options=_OPTION_SENTINELS,
    )
    assert answer.response
    assert answer.done is True


@pytest.mark.parametrize(("option", "value", "refusal"), _CLOUD_REFUSED_SENTINELS)
def test_generate_answers_the_sentinels_ollama_cloud_refuses(
    ollama_client: ollama.Client,
    ollama_chat_model: str,
    use_official_api: bool,
    option: str,
    value: float,
    refusal: str,
) -> None:
    """Three more sentinels are answered here, and refused by Ollama Cloud.

    A deliberate divergence, asserted on both targets so it stays one. An
    Ollama server defaults ``num_predict`` to -1 and reads a non-positive
    ``top_p`` or ``temperature`` as "off"; Ollama Cloud answers 400 for each.
    Its wording, measured per option, names the bound rather than the option
    -- only ``num_predict`` is named, and then as the ``max_tokens`` it is
    forwarded to. A client sending its own defaults has to be answered, so
    this gateway follows the server.

    Ref: https://github.com/ollama/ollama/blob/main/api/types.go (DefaultOptions)
         https://github.com/ollama/ollama/blob/main/mlxrunner/sample/sample.go
    """

    def call() -> ollama.GenerateResponse:
        """Send the sentinel on its own, so only it can be refused."""
        return ollama_client.generate(
            model=ollama_chat_model,
            prompt="Say hello.",
            stream=False,
            options={option: value},
        )

    if use_official_api:
        with pytest.raises(ollama.ResponseError) as raised:
            call()
        assert raised.value.status_code == 400
        assert refusal in raised.value.error, (
            "the refusal must be the one this option really draws, so that a "
            "vendor that starts accepting it shows up here"
        )
        return
    answer = call()
    assert answer.response
    assert answer.done is True


@pytest.mark.local
def test_generate_reports_an_untranslatable_option_as_a_bad_request(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An option value the completion parameters refuse answers 400, not 500.

    The sentinels above are handled, so the failure is injected: any option
    bound the translation stops matching has to reach the caller as a bad
    request naming the field, in Ollama's own single-field envelope.

    Ref: https://docs.ollama.com/openapi.yaml (ErrorResponse)
         stdapi/routes/ollama_generate.py:generate
    """

    def out_of_range(*_args: object, **_kwargs: object) -> CompletionCreateParams:
        """Fail the way an unsanitised option value would."""
        return CompletionCreateParams(
            model="m",
            messages=[ChatCompletionUserMessageParam(role="user", content="hi")],
            top_p=-1.0,
        )

    monkeypatch.setattr(
        generate_route,
        "validate_model",
        AsyncMock(return_value=make_model_details("m")),
    )
    monkeypatch.setattr(ollama_adapter, "to_chat_completion_params", out_of_range)
    response = app_client.post(
        ollama_route("/api/generate"),
        json={"model": "m", "prompt": "hi", "stream": False},
    )
    assert response.status_code == 400
    envelope = response.json()
    assert list(envelope) == ["error"]
    assert "top_p" in envelope["error"]


@pytest.mark.parametrize(("keep_alive", "reason"), [(None, "load"), (0, "unload")])
def test_generate_without_a_prompt_is_the_load_no_op(
    ollama_client: ollama.Client,
    ollama_chat_model: str,
    use_official_api: bool,
    keep_alive: int | None,
    reason: str,
) -> None:
    """A prompt-less generate answers the load, and with ``keep_alive`` 0 the unload.

    ``ollama run <model>`` preloads with exactly this request before it opens
    its REPL, and ``ollama stop <model>`` sends it with ``keep_alive`` at zero;
    a client that met an error here would abort before its first prompt. A
    hosted model needs neither, so nothing is generated and no backend is
    called -- the answer is the single done object, as upstream's is.

    Ref: https://docs.ollama.com/api/generate
         stdapi/routes/ollama_generate.py:generate
    """
    request: dict[str, Any] = {"model": ollama_chat_model, "stream": False}
    if keep_alive is not None:
        request["keep_alive"] = keep_alive
    answer = ollama_client.generate(**request)
    assert answer.done is True
    assert not answer.response
    if not use_official_api:
        assert answer.model == ollama_chat_model
        assert answer.done_reason == reason
