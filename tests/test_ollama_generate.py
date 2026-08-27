"""Ollama-compatible POST /api/generate, driven by the official ``ollama`` client.

The served subset is ``prompt`` with ``system`` and ``images``. ``raw``,
``suffix``, ``template`` and ``context`` all need the model's own tokenizer and
prompt template, which a hosted backend does not expose, so each is refused
rather than quietly ignored: dropping any of them answers something other than
what was asked for. Ollama Cloud serves them, so that refusal is asserted on
both targets rather than only where it holds.

Ref: https://docs.ollama.com/api/generate
     stdapi/routes/ollama_generate.py:generate
     stdapi/models/chat/_adapters/_ollama.py:generate_stream
"""

from typing import Any

import ollama
import pytest

# Keeps the shared answer below on one worker, so it is requested once.
pytestmark = pytest.mark.xdist_group("ollama_generate")

#: Prompt short enough to keep a live answer cheap, long enough to stream.
_PROMPT = "Count from 1 to 5."

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
    Cloud runs the model's own template and serves all four, while this gateway
    reaches the backend through a chat API that exposes neither the template nor
    the tokenizer, and says so rather than silently answering something else.

    Ref: stdapi/types/ollama.py:GenerateRequest._reject_prompt_level_fields
    """
    request: dict[str, Any] = {
        "model": ollama_chat_model,
        "prompt": "hi",
        "stream": False,
        "options": {"num_predict": 1},
        field: value,
    }
    if use_official_api:
        assert ollama_client.generate(**request).done is True
        return
    with pytest.raises(ollama.ResponseError) as raised:
        ollama_client.generate(**request)
    assert raised.value.status_code == 400
    assert field in raised.value.error
    assert "/api/chat" in raised.value.error


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
