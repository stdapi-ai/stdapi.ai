"""Ollama-compatible POST /api/chat, driven by the official ``ollama`` client.

The client is what makes these assertions worth anything: it parses the answer
into its own ``ChatResponse``, so a field this gateway spells differently, or
types differently, fails here rather than in a user's application.

Streaming is newline-delimited JSON rather than server-sent events: one bare
JSON object per line, no ``data:`` prefix and no ``[DONE]`` sentinel, ended by
an object carrying ``done: true`` and the metrics. The client iterates those
lines, so a stream it can walk to the terminal event is the assertion; the media
type itself is checked once, over raw HTTP.

Ref: https://docs.ollama.com/api/chat
     https://docs.ollama.com/openapi.yaml
     stdapi/routes/ollama_chat.py:chat
     stdapi/models/chat/_adapters/_ollama.py:chat_stream
"""

from io import BytesIO
from typing import TYPE_CHECKING, Any

import ollama
import pytest
from PIL import Image as PILImage
from pydantic_core import from_json

from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    import httpx
    from starlette.testclient import TestClient


# Module scope on the shared answers below only saves a call when every test of
# the module runs on one worker.
pytestmark = pytest.mark.xdist_group("ollama_chat")

#: Media type every Ollama stream is served with by an Ollama server.
NDJSON = "application/x-ndjson"

#: An Ollama-shaped name neither target serves.
UNKNOWN_MODEL = "llama3.2:3b"

#: Prompt short enough to keep a live answer cheap, long enough to stream.
#: The same one on both shared answers, so their token counts are comparable.
_PROMPT = "Count from 1 to 5."

#: Tool the model is offered whenever a test needs a tool call.
WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _red_square_png() -> bytes:
    """Build a plain red PNG, small enough to send inline and easy to describe.

    Returns:
        The image bytes; the client base64-encodes them itself.
    """
    buffer = BytesIO()
    PILImage.new("RGB", (64, 64), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


#: The image the vision test sends.
RED_SQUARE_PNG = _red_square_png()


def ndjson_lines(response: httpx.Response) -> list[dict[str, Any]]:
    """Parse a newline-delimited JSON body into its objects.

    Args:
        response: The streamed response, already read.

    Returns:
        One parsed object per line, blank lines dropped.
    """
    return [from_json(line) for line in response.text.splitlines() if line.strip()]


@pytest.fixture(scope="module")
def buffered_chat(
    ollama_client: ollama.Client, ollama_chat_model: str
) -> ollama.ChatResponse:
    """One buffered chat answer, shared by the tests that only read it.

    A cloud answer is billed and counts against the account's rate limit, so the
    shape assertions and the metric assertions read the same one.

    Returns:
        The complete chat response.
    """
    return ollama_client.chat(
        model=ollama_chat_model,
        messages=[{"role": "user", "content": _PROMPT}],
        stream=False,
    )


@pytest.fixture(scope="module")
def streamed_chat(
    ollama_client: ollama.Client, ollama_chat_model: str
) -> list[ollama.ChatResponse]:
    """Every event of one streamed chat, shared by the streaming tests.

    Returns:
        The parts the client yielded, in order.
    """
    return list(
        ollama_client.chat(
            model=ollama_chat_model,
            messages=[{"role": "user", "content": _PROMPT}],
            stream=True,
        )
    )


def test_chat_returns_an_assistant_message(
    buffered_chat: ollama.ChatResponse, ollama_chat_model: str
) -> None:
    """A buffered chat answers one object carrying the assistant's message.

    Ref: https://docs.ollama.com/api/chat
    """
    assert buffered_chat.model == ollama_chat_model
    assert buffered_chat.message.role == "assistant"
    assert buffered_chat.message.content
    assert buffered_chat.done is True
    assert buffered_chat.done_reason == "stop"
    assert buffered_chat.created_at


def test_chat_reports_only_the_metrics_it_measured(
    buffered_chat: ollama.ChatResponse,
) -> None:
    """Token counts and wall clock are reported; a buffered answer omits the rest.

    ``load_duration`` has no source at all here, and a buffered answer carries
    no prompt/generation split, so both are omitted rather than reported as a
    number nothing measured. Every metric is ``omitempty`` upstream, and Ollama
    Cloud omits exactly the same three on a buffered cloud answer.

    Ref: https://docs.ollama.com/api/usage
    """
    assert buffered_chat.prompt_eval_count
    assert buffered_chat.prompt_eval_count > 0
    assert buffered_chat.eval_count
    assert buffered_chat.eval_count > 0
    assert buffered_chat.total_duration
    assert buffered_chat.total_duration > 0
    assert buffered_chat.load_duration is None
    assert buffered_chat.prompt_eval_duration is None
    assert buffered_chat.eval_duration is None


def test_chat_streams_to_a_terminal_done_event(
    streamed_chat: list[ollama.ChatResponse], use_official_api: bool
) -> None:
    """The client walks the stream to a terminal event carrying the metrics.

    The gateway times the prompt and the generation separately because it reads
    the split out of its own stream; Ollama Cloud reports neither duration even
    when streaming, so only the counts are asserted on both targets.

    Ref: https://docs.ollama.com/api/chat
    """
    assert len(streamed_chat) > 1
    assert all(part.done is False for part in streamed_chat[:-1])
    assert all(part.message.role == "assistant" for part in streamed_chat)
    assert "".join(part.message.content or "" for part in streamed_chat)
    terminal = streamed_chat[-1]
    assert terminal.done is True
    assert terminal.done_reason == "stop"
    assert terminal.prompt_eval_count
    assert terminal.eval_count
    assert terminal.load_duration is None
    if not use_official_api:
        assert terminal.prompt_eval_duration
        assert terminal.eval_duration


def test_chat_streamed_and_buffered_token_counts_agree(
    streamed_chat: list[ollama.ChatResponse], buffered_chat: ollama.ChatResponse
) -> None:
    """The same prompt reports the same input token count either way.

    Ref: stdapi/models/chat/_adapters/_ollama.py:_StreamState.collect
    """
    assert buffered_chat.prompt_eval_count
    assert streamed_chat[-1].prompt_eval_count == buffered_chat.prompt_eval_count


def test_chat_streams_newline_delimited_json(
    ollama_http: httpx.Client, ollama_chat_model: str, use_official_api: bool
) -> None:
    """The transport is newline-delimited JSON, not server-sent events.

    Raw HTTP: the official client reads lines and never looks at the media type,
    so nothing it exposes can tell an NDJSON stream from an SSE one. Both
    targets send one bare JSON object per line with no ``data:`` prefix; they
    label it differently, the gateway with the ``application/x-ndjson`` an
    Ollama server sends and Ollama Cloud with a plain ``application/json``.

    Ref: https://docs.ollama.com/api/chat
    """
    response = ollama_http.post(
        "/api/chat",
        json={
            "model": ollama_chat_model,
            "messages": [{"role": "user", "content": _PROMPT}],
            "stream": True,
            "options": {"num_predict": 8},
        },
    )
    assert response.status_code == 200
    media_type = "application/json" if use_official_api else NDJSON
    assert response.headers["content-type"].startswith(media_type)
    assert "data:" not in response.text
    events = ndjson_lines(response)
    assert len(events) > 1
    assert events[-1]["done"] is True


@pytest.mark.local
def test_a_streamed_chat_still_records_its_usage(
    ollama_client: ollama.Client,
    local_test_client: TestClient,  # noqa: ARG001  (binds the in-process gateway)
    ollama_chat_model: str,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """The tokens AWS billed are recorded, on the streamed path as on the buffered one.

    Usage arrives on the trailing event of the upstream stream, so a
    translation that stopped reading at the last event it cared about would
    bill nothing.

    Ref: stdapi/models/chat/_adapters/_ollama.py:_upstream_chunks
    """
    capfd.readouterr()
    parts = list(
        ollama_client.chat(
            model=ollama_chat_model,
            messages=[{"role": "user", "content": "Say hello."}],
            stream=True,
        )
    )
    assert parts[-1].done is True
    entries = logged_usage_entries(
        capfd.readouterr().out, operation="/api/chat", model=ollama_chat_model
    )
    assert entries, "Expected a usage entry for the streamed chat"
    assert entries[0]["input_tokens"] > 0
    assert entries[0]["output_tokens"] > 0


def test_chat_calls_a_tool(
    ollama_client: ollama.Client, ollama_chat_model: str
) -> None:
    """A declared tool comes back as a complete tool call with parsed arguments.

    The call is asked for in the prompt: this dialect has no ``tool_choice``, so
    an open question would leave the assertion resting on the model's discretion
    rather than on the translation under test.

    Ref: https://docs.ollama.com/api/chat#tools
    """
    answer = ollama_client.chat(
        model=ollama_chat_model,
        messages=[
            {"role": "user", "content": "Call the get_weather tool for Paris, France."}
        ],
        tools=[WEATHER_TOOL],
        stream=False,
    )
    calls = answer.message.tool_calls
    assert calls
    assert calls[0].function.name == "get_weather"
    assert isinstance(calls[0].function.arguments, dict)


def test_chat_replays_a_tool_result(
    ollama_client: ollama.Client, ollama_chat_model: str
) -> None:
    """A tool result with no call identifier is correlated by tool name.

    Ollama's own tool calls carry no ``id``, and the client's ``Message`` has no
    field for one, so a conversation replaying a call is the shape every Ollama
    client sends. The prompt asks for the result to be stated so that answering
    it is not a second tool call, which would leave ``content`` empty.

    Ref: stdapi/models/chat/_adapters/_ollama.py:_take_tool_call_id
    """
    answer = ollama_client.chat(
        model=ollama_chat_model,
        messages=[
            ollama.Message(
                role="user",
                content="Call the get_weather tool for Paris, then state its result.",
            ),
            ollama.Message(
                role="assistant",
                content="",
                tool_calls=[
                    ollama.Message.ToolCall(
                        function=ollama.Message.ToolCall.Function(
                            name="get_weather", arguments={"city": "Paris"}
                        )
                    )
                ],
            ),
            ollama.Message(
                role="tool", tool_name="get_weather", content="18 degrees and sunny"
            ),
        ],
        tools=[WEATHER_TOOL],
        stream=False,
    )
    assert answer.message.content


@pytest.mark.gateway(
    "Ollama Cloud ignores a JSON schema in 'format' and answers prose on every "
    "free cloud model -- gpt-oss:20b, nemotron-3-nano:30b and gemma4:31b were "
    "each measured doing so -- and only the weaker 'json' shape is honoured "
    "there, too loosely to assert. The gateway marking is a fact about the "
    "vendor, not an untested route"
)
def test_chat_returns_structured_output(
    ollama_client: ollama.Client, ollama_json_output_model: str
) -> None:
    """A bare JSON schema in `format` constrains the answer.

    The schema needs the name and wrapper Ollama does not send, which the
    dialect supplies; the model is the one whose backend accepts a schema at all.

    Ref: https://docs.ollama.com/api/chat#structured-outputs
    """
    answer = ollama_client.chat(
        model=ollama_json_output_model,
        messages=[{"role": "user", "content": "The capital of France."}],
        format={
            "type": "object",
            "properties": {"capital": {"type": "string"}},
            "required": ["capital"],
        },
        stream=False,
    )
    assert answer.message.content
    assert "capital" in from_json(answer.message.content)


def test_chat_returns_thinking(
    ollama_client: ollama.Client, ollama_reasoning_model: str
) -> None:
    """`think` puts the reasoning trace in its own field, beside the answer.

    Ref: https://docs.ollama.com/api/chat#thinking
    """
    answer = ollama_client.chat(
        model=ollama_reasoning_model,
        messages=[{"role": "user", "content": "What is 17 times 23?"}],
        think=True,
        stream=False,
    )
    assert answer.message.thinking
    assert answer.message.content


def test_chat_reads_an_image(
    ollama_client: ollama.Client, ollama_vision_model: str
) -> None:
    """An image on a message reaches a vision model.

    The client base64-encodes the bytes itself, which is the only encoding an
    Ollama message has for an image.

    Ref: https://docs.ollama.com/openapi.yaml (ChatMessage.images)
    """
    answer = ollama_client.chat(
        model=ollama_vision_model,
        messages=[
            ollama.Message(
                role="user",
                content="What colour is this image? Answer with one word.",
                images=[ollama.Image(value=RED_SQUARE_PNG)],
            )
        ],
        stream=False,
    )
    assert answer.message.content
    assert "red" in answer.message.content.lower()


@pytest.mark.gateway("A Bedrock model ID carries a colon, which Ollama Cloud's do not")
def test_chat_echoes_the_model_name_the_client_sent(
    ollama_client: ollama.Client, ollama_chat_model: str
) -> None:
    """A ``:latest`` tag resolves, and the answer names the model as it was asked for.

    Ollama clients match responses on the exact name they sent, and a Bedrock
    model ID ends in a ``:<version>`` suffix of its own, so the tag is stripped
    only after the exact name has missed.

    Ref: stdapi/models/__init__.py:_lookup_with_latest_fallback
    """
    tagged = f"{ollama_chat_model}:latest"
    answer = ollama_client.chat(
        model=tagged, messages=[{"role": "user", "content": "Say hello."}], stream=False
    )
    assert answer.model == tagged
    assert answer.message.content


def test_chat_refuses_an_unknown_model(
    ollama_client: ollama.Client, use_official_api: bool
) -> None:
    """An Ollama-shaped name this server does not offer answers 404, Ollama-shaped.

    The client unwraps the ``error`` field of the envelope into
    ``ResponseError.error``, so the envelope shape is what is asserted here.
    Both targets answer 404 and name the model; only the wording differs, the
    gateway's being the one it gives on every dialect, which points the caller
    at the model list instead of at a pull it cannot perform.

    Ref: https://docs.ollama.com/openapi.yaml (ErrorResponse)
    """
    with pytest.raises(ollama.ResponseError) as raised:
        ollama_client.chat(
            model=UNKNOWN_MODEL,
            messages=[{"role": "user", "content": "hi"}],
            stream=False,
        )
    assert raised.value.status_code == 404
    assert UNKNOWN_MODEL in raised.value.error
    wording = "not found" if use_official_api else "does not exist"
    assert wording in raised.value.error


def test_chat_refuses_logprobs(
    ollama_client: ollama.Client, ollama_chat_model: str, use_official_api: bool
) -> None:
    """Log probabilities are refused rather than silently dropped.

    A deliberate divergence, asserted on both targets so it stays one: Ollama
    Cloud accepts ``logprobs`` and answers 200 with no log probabilities in the
    body, while this gateway -- which has no source for them at all -- tells the
    caller so instead of answering something other than what was asked for.

    Ref: stdapi/types/ollama.py:_InferenceRequest._reject_logprobs
    """

    def call() -> ollama.ChatResponse:
        """Ask for log probabilities, capped to the shortest possible answer."""
        return ollama_client.chat(
            model=ollama_chat_model,
            messages=[{"role": "user", "content": "hi"}],
            logprobs=True,
            stream=False,
            options={"num_predict": 1},
        )

    if use_official_api:
        assert call().logprobs is None
        return
    with pytest.raises(ollama.ResponseError) as raised:
        call()
    assert raised.value.status_code == 400
    assert "logprobs" in raised.value.error


@pytest.mark.parametrize(("keep_alive", "reason"), [(None, "load"), (0, "unload")])
def test_chat_without_a_message_is_the_load_no_op(
    ollama_client: ollama.Client,
    ollama_chat_model: str,
    use_official_api: bool,
    keep_alive: int | None,
    reason: str,
) -> None:
    """An empty conversation answers the load, and with ``keep_alive`` 0 the unload.

    Upstream defines an empty ``messages`` array as the request that makes a
    model resident, and the same request with ``keep_alive`` at zero as the one
    that evicts it -- what a client's "load model" and "unload model" controls
    send. A hosted model needs neither, so nothing is generated and no backend
    is called: the answer is the single done object carrying an empty assistant
    message, as upstream's is.

    Ref: https://docs.ollama.com/api/chat
         stdapi/routes/ollama_chat.py:chat
    """
    request: dict[str, Any] = {
        "model": ollama_chat_model,
        "messages": [],
        "stream": False,
    }
    if keep_alive is not None:
        request["keep_alive"] = keep_alive
    answer = ollama_client.chat(**request)
    assert answer.done is True
    assert answer.message.role == "assistant"
    assert not answer.message.content
    if not use_official_api:
        assert answer.model == ollama_chat_model
        assert answer.done_reason == reason
