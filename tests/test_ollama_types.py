"""Ollama request models and the translation onto the chat completions path.

Everything here is the gateway's own translation layer, below the HTTP surface
the official client reaches -- except the last class, which captures the exact
body ``ollama.Client`` puts on the wire and feeds it to these models, so a field
the client sends and the gateway refuses fails at this level rather than in a
live call.

Ref: https://docs.ollama.com/openapi.yaml
     stdapi/types/ollama.py
     stdapi/models/chat/_adapters/_ollama.py
"""

from datetime import UTC, datetime
from json import dumps, loads
from typing import TYPE_CHECKING, Any

import httpx
import ollama
import pytest
from sse_starlette import ServerSentEvent

from stdapi.config import SETTINGS
from stdapi.models.chat._adapters import _ollama as adapter
from stdapi.monitoring import REQUEST_TIME
from stdapi.types.ollama import (
    ChatRequest,
    EmbedRequest,
    GenerateRequest,
    ShowRequest,
    created_at,
    total_duration,
)
from stdapi.types.openai import ResponseFormatJSONSchema
from stdapi.types.openai_chat_completions import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionFunctionToolParam,
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionToolMessageParam,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Generator

pytestmark = pytest.mark.gateway(
    "Exercises the gateway's own request models and adapter, below the HTTP "
    "surface any upstream endpoint exposes"
)

#: Host the mocked transport capturing the official client's request answers for.
_MOCK_HOST = "http://ollama.test"


@pytest.fixture
def request_time() -> Generator[datetime]:
    """Bind a fixed request start time, as the request middleware would.

    Yields:
        The bound timestamp.
    """
    moment = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
    token = REQUEST_TIME.set(moment)
    try:
        yield moment
    finally:
        REQUEST_TIME.reset(token)


def test_chat_request_defaults_to_streaming() -> None:
    """`stream` defaults to true, as the Ollama contract specifies.

    Ref: https://docs.ollama.com/openapi.yaml (ChatRequest.stream default)
    """
    assert ChatRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    ).stream


def test_chat_request_refuses_logprobs() -> None:
    """A request explicitly asking for log probabilities is refused, not silently served.

    Ref: stdapi/types/ollama.py:_InferenceRequest._reject_logprobs
    """
    with pytest.raises(ValueError, match="logprobs"):
        ChatRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "logprobs": True,
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("suffix", "tail"),
        ("template", "{{ .Prompt }}"),
        ("context", [1, 2, 3]),
        ("raw", True),
    ],
)
def test_generate_request_refuses_prompt_level_fields(
    field: str, value: object
) -> None:
    """The four fields needing the model's own prompt template are refused.

    Ref: stdapi/types/ollama.py:GenerateRequest._reject_prompt_level_fields
    """
    with pytest.raises(ValueError, match=field):
        GenerateRequest.model_validate(dict(model="m", prompt="hi", **{field: value}))


def test_generate_request_accepts_prompt_and_system() -> None:
    """The served subset of `/api/generate` parses.

    Ref: https://docs.ollama.com/openapi.yaml (GenerateRequest)
    """
    request = GenerateRequest.model_validate(
        {"model": "m", "prompt": "hi", "system": "be brief", "stream": False}
    )
    assert request.prompt == "hi"
    assert request.system == "be brief"


def test_show_request_accepts_the_legacy_name_field() -> None:
    """`name` is still folded into `model`, as the Ollama server does.

    Ref: https://docs.ollama.com/api/show
    """
    assert ShowRequest.model_validate({"name": "m"}).requested_model() == "m"
    assert (
        ShowRequest.model_validate({"model": "m", "name": "other"}).requested_model()
        == "m"
    )


def test_show_request_requires_a_model() -> None:
    """Naming no model at all is refused.

    Ref: stdapi/types/ollama.py:ShowRequest._require_a_model_name
    """
    with pytest.raises(ValueError, match="'model' is required"):
        ShowRequest.model_validate({})


def test_embed_request_accepts_one_string_or_a_list() -> None:
    """Both `input` shapes parse.

    Ref: https://docs.ollama.com/openapi.yaml (EmbedRequest.input)
    """
    assert EmbedRequest.model_validate({"model": "m", "input": "a"}).input == "a"
    assert EmbedRequest.model_validate({"model": "m", "input": ["a", "b"]}).input == [
        "a",
        "b",
    ]


def test_created_at_and_total_duration_use_the_request_clock(
    request_time: datetime,
) -> None:
    """Timestamps come from the request's own start time, in Ollama's units.

    Ref: stdapi/types/ollama.py:created_at
    """
    assert created_at() == request_time.isoformat()
    assert total_duration() > 0


def test_options_map_onto_the_chat_completion_parameters() -> None:
    """The option knobs with a hosted equivalent are forwarded, the rest ignored.

    Ref: stdapi/models/chat/_adapters/_ollama.py:_apply_options
    """
    params = adapter.to_chat_completion_params(
        ChatRequest.model_validate(
            {
                "model": "ollama-name",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "options": {
                    "temperature": 0.5,
                    "top_p": 0.9,
                    "top_k": 20,
                    "seed": 7,
                    "stop": ["END"],
                    "num_predict": 128,
                    "num_ctx": 4096,
                    "min_p": 0.1,
                },
            }
        ),
        "amazon.nova-micro-v1:0",
    )
    assert params.model == "amazon.nova-micro-v1:0"
    assert params.temperature == 0.5
    assert params.top_p == 0.9
    assert params.top_k == 20
    assert params.seed == 7
    assert params.stop == ["END"]
    assert params.max_completion_tokens == 128
    # Runner knobs are accepted and ignored rather than sent to the backend.
    assert "num_ctx" not in (params.model_extra or {})
    assert "min_p" not in (params.model_extra or {})


def test_negative_num_predict_leaves_the_limit_unset() -> None:
    """Ollama's -1/-2 mean "unbounded", which is the default here.

    Ref: stdapi/models/chat/_adapters/_ollama.py:_apply_options
    """
    params = adapter.to_chat_completion_params(
        ChatRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"num_predict": -1},
            }
        ),
        "amazon.nova-micro-v1:0",
    )
    assert params.max_completion_tokens is None


@pytest.mark.parametrize(
    ("think", "enable_thinking", "effort"),
    [
        (True, True, None),
        (False, False, None),
        ("low", True, "low"),
        ("max", True, "max"),
    ],
)
def test_think_maps_onto_the_reasoning_parameters(
    think: bool | str, enable_thinking: bool, effort: str | None
) -> None:
    """`think` selects reasoning, and a level selects the effort.

    Ref: https://docs.ollama.com/openapi.yaml (ChatRequest.think)
    """
    params = adapter.to_chat_completion_params(
        ChatRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "think": think,
            }
        ),
        "amazon.nova-micro-v1:0",
    )
    assert params.enable_thinking is enable_thinking
    assert params.reasoning_effort == effort


def test_format_json_and_schema_map_onto_response_format() -> None:
    """Both `format` shapes become a response format, the bare schema wrapped.

    Ref: https://docs.ollama.com/openapi.yaml (ChatRequest.format)
    """
    messages = [{"role": "user", "content": "hi"}]
    json_params = adapter.to_chat_completion_params(
        ChatRequest.model_validate(
            {"model": "m", "messages": messages, "format": "json"}
        ),
        "m",
    )
    assert json_params.response_format is not None
    assert json_params.response_format.type == "json_object"

    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    schema_params = adapter.to_chat_completion_params(
        ChatRequest.model_validate(
            {"model": "m", "messages": messages, "format": schema}
        ),
        "m",
    )
    schema_format = schema_params.response_format
    assert isinstance(schema_format, ResponseFormatJSONSchema)
    assert schema_format.json_schema.schema_ == {
        **schema,
        "additionalProperties": False,
    }


def test_a_bare_schema_is_closed_to_extra_properties() -> None:
    """Every object node gains `additionalProperties: false` unless it set one.

    Ollama constrains decoding to the schema and asks for nothing more, while
    the backend refuses a schema that leaves the question open; adding it is
    what lets a schema written for Ollama work unchanged.

    Ref: stdapi/models/chat/_adapters/_ollama.py:_closed_objects
    """
    params = adapter.to_chat_completion_params(
        ChatRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "format": {
                    "type": "object",
                    "properties": {
                        "inner": {"type": "object", "properties": {}},
                        "open": {"type": "object", "additionalProperties": True},
                        "list": {"type": "array", "items": {"type": "object"}},
                    },
                },
            }
        ),
        "m",
    )
    response_format = params.response_format
    assert isinstance(response_format, ResponseFormatJSONSchema)
    schema = response_format.json_schema.schema_
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert schema["additionalProperties"] is False
    for name, expected in (("inner", False), ("open", True)):
        node = properties[name]
        assert isinstance(node, dict)
        assert node["additionalProperties"] is expected
    items = properties["list"]
    assert isinstance(items, dict)
    nested = items["items"]
    assert isinstance(nested, dict)
    assert nested["additionalProperties"] is False


def test_streaming_requests_ask_for_the_usage_chunk() -> None:
    """Token counts on a stream come from the trailing usage chunk.

    Ref: stdapi/models/chat/_adapters/_ollama.py:to_chat_completion_params
    """
    params = adapter.to_chat_completion_params(
        ChatRequest.model_validate(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
        ),
        "m",
    )
    assert params.stream_options is not None
    assert params.stream_options.include_usage is True


def test_images_become_image_content_parts() -> None:
    """A message's images join its text as content parts.

    Ref: https://docs.ollama.com/openapi.yaml (ChatMessage.images)
    """
    png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
        "IQAAAABJRU5ErkJggg=="
    )
    params = adapter.to_chat_completion_params(
        ChatRequest.model_validate(
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "what is this", "images": [png]}
                ],
            }
        ),
        "m",
    )
    content = params.messages[0].content
    assert isinstance(content, list)
    assert [part.type for part in content] == ["text", "image_url"]


def test_an_uncaptioned_image_carries_no_empty_text_part() -> None:
    """An image sent with no text must not become an empty text block.

    ``images`` with no ``content`` is an ordinary Ollama message -- it is how
    the client's own examples send a picture on its own -- but the backend
    refuses a text block with nothing in it, so the whole request would fail.

    Ref: stdapi/models/chat/_adapters/_ollama.py:_content_parts
         https://docs.ollama.com/openapi.yaml (ChatMessage.images)
    """
    png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
        "IQAAAABJRU5ErkJggg=="
    )
    params = adapter.to_chat_completion_params(
        ChatRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "", "images": [png]}],
            }
        ),
        "m",
    )
    content = params.messages[0].content
    assert isinstance(content, list)
    assert [part.type for part in content] == ["image_url"]


def test_tool_calls_are_correlated_without_identifiers() -> None:
    """Ollama tool calls carry no id, so one is synthesized and reused by the result.

    Ref: stdapi/models/chat/_adapters/_ollama.py:_map_messages
    """
    params = adapter.to_chat_completion_params(
        ChatRequest.model_validate(
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "weather?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "get_weather",
                                    "arguments": {"c": "FR"},
                                }
                            }
                        ],
                    },
                    {"role": "tool", "content": "sunny", "tool_name": "get_weather"},
                ],
            }
        ),
        "m",
    )
    assistant, tool = params.messages[1], params.messages[2]
    assert isinstance(assistant, ChatCompletionAssistantMessageParam)
    assert isinstance(tool, ChatCompletionToolMessageParam)
    assert assistant.tool_calls is not None
    call = assistant.tool_calls[0]
    assert isinstance(call, ChatCompletionMessageFunctionToolCall)
    assert tool.tool_call_id == call.id
    assert call.function.arguments == '{"c":"FR"}'


def test_a_foreign_tool_call_id_never_reaches_the_backend() -> None:
    """An id this dialect never minted selects a pending call instead of being echoed.

    Only a raw-HTTP client can send one -- the official client's ``Message`` has
    no such field, and assistant tool calls carry no id to echo -- so it can only
    name a tool call the backend never declared, which Converse refuses. It is
    therefore used to consume a pending entry and dropped otherwise, leaving the
    correlation to the tool name and then to call order.

    Ref: stdapi/models/chat/_adapters/_ollama.py:_take_tool_call_id
    """
    params = adapter.to_chat_completion_params(
        ChatRequest.model_validate(
            {
                "model": "m",
                "messages": [
                    {"role": "user", "content": "weather and time?"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"function": {"name": "get_weather", "arguments": {}}},
                            {"function": {"name": "get_time", "arguments": {}}},
                        ],
                    },
                    {
                        "role": "tool",
                        "content": "noon",
                        "tool_name": "get_time",
                        "tool_call_id": "toolu_from_another_server",
                    },
                    {"role": "tool", "content": "sunny", "tool_name": "get_weather"},
                ],
            }
        ),
        "m",
    )
    assistant = params.messages[1]
    assert isinstance(assistant, ChatCompletionAssistantMessageParam)
    assert assistant.tool_calls is not None
    declared = [call.id for call in assistant.tool_calls]
    answered = [
        message.tool_call_id
        for message in params.messages[2:]
        if isinstance(message, ChatCompletionToolMessageParam)
    ]
    assert answered == [declared[1], declared[0]]


def test_tools_become_function_tools() -> None:
    """A declared tool reaches the model as an OpenAI function tool.

    Ref: https://docs.ollama.com/openapi.yaml (ToolDefinition)
    """
    params = adapter.to_chat_completion_params(
        ChatRequest.model_validate(
            {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Weather for a city",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            }
        ),
        "m",
    )
    assert params.tools is not None
    tool = params.tools[0]
    assert isinstance(tool, ChatCompletionFunctionToolParam)
    assert tool.function.name == "get_weather"


def test_generate_folds_the_system_prompt_into_a_message_list() -> None:
    """`prompt` plus `system` becomes a one-turn conversation.

    Ref: stdapi/models/chat/_adapters/_ollama.py:to_chat_completion_params
    """
    params = adapter.to_chat_completion_params(
        GenerateRequest.model_validate(
            {"model": "m", "prompt": "hi", "system": "be brief"}
        ),
        "m",
    )
    assert [message.role for message in params.messages] == ["system", "user"]


def _sent_body(
    call: Callable[[ollama.Client], object], answer: dict[str, Any]
) -> dict[str, Any]:
    """Run *call* against a mocked transport and return the body it sent.

    The client is the only authority on what an Ollama request looks like on the
    wire, so the body is captured from it rather than hand-written.

    Args:
        call: Client call to make.
        answer: JSON body the mocked endpoint answers with.

    Returns:
        The decoded request body the client sent.
    """
    sent: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request body and answer the stub."""
        sent.update(loads(request.content))
        return httpx.Response(200, json=answer)

    call(ollama.Client(host=_MOCK_HOST, transport=httpx.MockTransport(handler)))
    return sent


class TestTheOfficialClientRequestBodies:
    """Every field ``ollama.Client`` sends is a field these models accept.

    ``strict_input_validation`` makes the request models reject an unknown
    field, so a client option this dialect never modelled is a 400 for a user
    who sets it. Capturing the real body is the only way to see that coming.

    Ref: https://github.com/ollama/ollama-python
    """

    def test_a_chat_request_parses(self) -> None:
        """The client's ``chat`` body, with every option it exposes, validates."""
        body = _sent_body(
            lambda client: client.chat(
                model="m",
                messages=[
                    ollama.Message(role="system", content="be brief"),
                    ollama.Message(role="user", content="hi"),
                ],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Weather for a city",
                            "parameters": {
                                "type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"],
                            },
                        },
                    }
                ],
                think="low",
                format="json",
                keep_alive="5m",
                options=ollama.Options(temperature=0.5, num_ctx=4096, stop=["END"]),
                stream=False,
            ),
            {
                "model": "m",
                "created_at": "2026-01-02T03:04:05Z",
                "message": {"role": "assistant", "content": "ok"},
                "done": True,
            },
        )
        request = ChatRequest.model_validate(body)
        assert request.think == "low"
        assert request.tools is not None
        assert request.options is not None
        assert request.options.temperature == 0.5

    def test_a_generate_request_parses(self) -> None:
        """The client's ``generate`` body, minus the fields this server refuses."""
        body = _sent_body(
            lambda client: client.generate(
                model="m",
                prompt="hi",
                system="be brief",
                images=[ollama.Image(value=b"\x89PNG")],
                think=True,
                keep_alive=300,
                options={"seed": 7},
                stream=False,
            ),
            {
                "model": "m",
                "created_at": "2026-01-02T03:04:05Z",
                "response": "ok",
                "done": True,
            },
        )
        request = GenerateRequest.model_validate(body)
        assert request.prompt == "hi"
        assert request.system == "be brief"
        assert request.images

    def test_an_embed_request_parses(self) -> None:
        """The client's ``embed`` body validates, dimensions and all."""
        body = _sent_body(
            lambda client: client.embed(
                model="m",
                input=["a", "b"],
                dimensions=256,
                truncate=True,
                keep_alive="5m",
                options={"seed": 7},
            ),
            {"model": "m", "embeddings": [[0.1], [0.2]]},
        )
        assert EmbedRequest.model_validate(body).dimensions == 256

    def test_a_show_request_parses(self) -> None:
        """The client names the model under ``model``, which is what is read."""
        body = _sent_body(
            lambda client: client.show("m"),
            {"details": {}, "model_info": {}, "capabilities": []},
        )
        assert ShowRequest.model_validate(body).requested_model() == "m"


async def _chat_completion_events(
    deltas: list[dict[str, Any]],
) -> AsyncGenerator[ServerSentEvent]:
    """Serialize deltas as the upstream chat completion stream would.

    Args:
        deltas: One ``delta`` object per chunk.

    Yields:
        One event per delta, then the terminal ``[DONE]``.
    """
    for delta in deltas:
        yield ServerSentEvent(data=dumps({"choices": [{"index": 0, "delta": delta}]}))
    yield ServerSentEvent(data="[DONE]")


async def _translate_chat(deltas: list[dict[str, Any]]) -> list[Any]:
    """Run the deltas through the Ollama chat stream translation.

    Args:
        deltas: One ``delta`` object per chunk.

    Returns:
        Every event the translation yielded, in order.
    """
    return [
        event
        async for event in adapter.chat_stream(_chat_completion_events(deltas), "m")
    ]


@pytest.mark.usefixtures("request_time")
async def test_each_streamed_event_carries_the_moment_it_was_emitted(
    monkeypatch: pytest.MonkeyPatch, request_time: datetime
) -> None:
    """A stream's timestamps advance, as the documented examples show them doing.

    A client reading consecutive ``created_at`` values measures time to first
    token and inter-token latency from them; one stamp for the whole stream
    makes every delta zero.

    Ref: https://docs.ollama.com/api/chat (streaming response)
         stdapi/types/ollama.py:streamed_at
    """
    ticks = iter(
        datetime(2026, 1, 2, 3, 4, 5 + second, tzinfo=UTC) for second in range(1, 10)
    )
    monkeypatch.setattr(type(SETTINGS), "now", lambda _: next(ticks))

    stamps = [
        event["created_at"]
        for event in await _translate_chat([{"content": "a"}, {"content": "b"}])
    ]

    assert request_time.isoformat() not in stamps
    assert len(set(stamps)) == len(stamps)
    assert stamps == sorted(stamps)


@pytest.mark.usefixtures("request_time")
@pytest.mark.parametrize(
    ("setting", "delta_field", "expected"),
    [
        ("reasoning_content", "reasoning_content", "thought"),
        ("reasoning", "reasoning", "thought"),
        ("none", "reasoning_content", None),
    ],
)
async def test_the_reasoning_field_setting_is_read_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
    setting: str,
    delta_field: str,
    expected: str | None,
) -> None:
    """`message.thinking` follows the setting as it stands when the answer is built.

    Resolving it once at import would freeze whichever value the first import
    saw, which is what makes the documented ``none`` behaviour assertable here.

    Ref: docs/api_ollama_chat.md (message.thinking)
         stdapi/config.py:_Settings.chat_completions_reasoning_field
    """
    monkeypatch.setattr(SETTINGS, "chat_completions_reasoning_field", setting)

    events = await _translate_chat([{"content": "hi", delta_field: "thought"}])

    assert events[0]["message"].get("thinking") == expected


class TestStreamedToolCalls:
    """Tool calls arriving in fragments, reassembled before the terminal event.

    Ollama never streams a partial call, so the gateway holds the fragments and
    emits one whole call. Nothing above this layer can drive it: the split is
    the upstream model's choice, and no prompt makes it split on demand.

    Ref: stdapi/models/chat/_adapters/_ollama.py:_StreamState.add_tool_call_delta
         https://docs.ollama.com/api/chat
    """

    @pytest.fixture(autouse=True)
    def _clock(self, request_time: datetime) -> None:
        """Pin the request clock the translated metrics are measured from."""

    async def _translate(self, deltas: list[dict[str, Any]]) -> list[Any]:
        """Run the deltas through the Ollama stream translation.

        Args:
            deltas: One ``delta`` object per chunk.

        Returns:
            Every event the translation yielded, in order.
        """
        return await _translate_chat(deltas)

    @staticmethod
    def _fragment(index: int, name: str | None, arguments: str) -> dict[str, Any]:
        """Build one tool-call fragment of a delta.

        Args:
            index: The call's index within the message.
            name: The function name, on the fragment that carries it.
            arguments: This fragment's slice of the argument JSON.

        Returns:
            A ``delta`` object holding that one fragment.
        """
        function: dict[str, Any] = {"arguments": arguments}
        if name is not None:
            function["name"] = name
        return {"tool_calls": [{"index": index, "function": function}]}

    async def test_argument_fragments_are_joined_into_one_call(self) -> None:
        """Arguments split over deltas arrive as one parsed object.

        Concatenation is the whole point: a client that received
        ``{"city":`` and ``"Paris"}`` as two calls could not act on either.
        """
        events = await self._translate(
            [
                self._fragment(0, "get_weather", '{"city"'),
                self._fragment(0, None, ': "Paris"'),
                self._fragment(0, None, "}"),
            ]
        )

        calls = next(
            event["message"]["tool_calls"]
            for event in events
            if event.get("message", {}).get("tool_calls")
        )
        assert len(calls) == 1
        assert calls[0]["function"]["name"] == "get_weather"
        assert calls[0]["function"]["arguments"] == {"city": "Paris"}

    async def test_the_assembled_call_precedes_the_terminal_event(self) -> None:
        """A call emitted after ``done`` would reach a client that stopped reading."""
        events = await self._translate([self._fragment(0, "get_weather", "{}")])

        assert [bool(event.get("done")) for event in events][-1] is True
        assert events[-2]["message"]["tool_calls"]

    async def test_several_calls_keep_the_order_the_model_asked_for(self) -> None:
        """Indices order the calls, whatever order their fragments arrived in."""
        events = await self._translate(
            [self._fragment(1, "second", "{}"), self._fragment(0, "first", "{}")]
        )

        calls = next(
            event["message"]["tool_calls"]
            for event in events
            if event.get("message", {}).get("tool_calls")
        )
        assert [call["function"]["name"] for call in calls] == ["first", "second"]

    async def test_a_stream_requesting_no_tool_emits_no_tool_event(self) -> None:
        """Nothing is invented for a plain answer."""
        events = await self._translate([{"content": "hello"}])

        assert not any(event.get("message", {}).get("tool_calls") for event in events)
