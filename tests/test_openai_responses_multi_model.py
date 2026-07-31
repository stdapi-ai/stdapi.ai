"""Multi-model parametrized coverage of the OpenAI /v1/responses route.

One model per provider family available on AWS Bedrock, with a Claude model in
every parametrize list as the reference baseline.  Only assertions that hold on
both backing paths are made here: models served by bedrock-runtime go through
the Converse adapter, while Bedrock Mantle models answer on the Chat Completions
or Messages API and their responses are composed into the Responses shape, which
drops the echo of request parameters.

The 85 live calls here are billed and issued sequentially, so every test is
``expensive`` and the whole file is ``slow``.  The markers are conjunctive::

    pytest --expensive --slow tests/test_openai_responses_multi_model.py

Models lacking a feature (streaming tool use, ``tool_choice="required"``) or
missing from the configured Regions call ``pytest.skip()`` so the result is
recorded as *skipped* rather than as a failure.

Ref: https://developers.openai.com/api/reference/resources/responses
     stdapi/routes/openai_responses.py:create_response
     stdapi/models/chat/_adapters/_openai_responses.py:translate_request
     stdapi/models/chat/_mantle/_convert.py:_chat_to_responses_response
"""

import json
from contextlib import contextmanager
from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError, NotFoundError

from tests._helpers import red_png_b64
from tests._multi_model import VISION_MODELS_OPENAI, with_marks
from tests.conftest import REPO_ROOT

if TYPE_CHECKING:
    from collections.abc import Iterator

    from openai import OpenAI
    from openai.types.responses import ResponseFunctionToolCall, ResponseStreamEvent


#: 85 sequential live calls across many model families; requires --expensive --slow.
pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _skip_official_api(use_official_api: bool) -> None:
    """Skip the whole module against the official API.

    The point of these tests is the gateway's per-family mapping, and the model
    IDs are Bedrock-only.
    """
    if use_official_api:
        pytest.skip("Multi-model tests only run against the local server")


@contextmanager
def _skip_unavailable_model(
    model: str, *, unsupported: tuple[str, ...] = ()
) -> Iterator[None]:
    """Turn "model absent" and "feature absent" answers into skips.

    Args:
        model: Model under test, named in the skip reason.
        unsupported: Fragments of a ``BadRequestError`` message that mean the
            model lacks the feature rather than the request being wrong.

    Yields:
        None, around the calls that may hit an unavailable model.
    """
    try:
        yield
    except NotFoundError:
        pytest.skip(f"Model {model!r} not available in configured regions")
    except BadRequestError as exc:
        if any(fragment in str(exc).lower() for fragment in unsupported):
            pytest.skip(f"Model {model!r} does not support this request: {exc}")
        raise


# ---------------------------------------------------------------------------
# Model lists — one representative per family, prefer fast/cheap variants
# ---------------------------------------------------------------------------

#: One model per provider family for basic/streaming/multi-turn tests.
_BASIC_MODELS = pytest.mark.parametrize(
    "model",
    [
        "anthropic.claude-haiku-4-5-20251001-v1:0",  # Claude (reference)
        "amazon.nova-micro-v1:0",  # Amazon Nova (cheapest)
        # "ai21.jamba-1-5-mini-v1:0",  # AI21 Jamba (SSM/Transformer hybrid, 256k ctx)
        "deepseek.v3-v1:0",  # DeepSeek V3 (fast non-reasoning)
        "google.gemma-3-12b-it",  # Google Gemma
        "meta.llama3-3-70b-instruct-v1:0",  # Meta Llama
        "minimax.minimax-m2.5",  # MiniMax
        "mistral.mistral-7b-instruct-v0:2",  # Mistral (cheapest)
        "mistral.mistral-large-2402-v1:0",  # Mistral Large
        pytest.param(
            "mistral.pixtral-large-2502-v1:0",
            marks=pytest.mark.xfail(
                strict=False,
                reason="Pixtral non-deterministically misidentifies colour of 1x1 PNG",
            ),
        ),  # Mistral Pixtral Large (vision)
        pytest.param(
            "moonshotai.kimi-k2.5",
            marks=pytest.mark.xfail(
                strict=False,
                reason="Kimi K2.5 occasionally returns incomplete status in streaming mode",
            ),
        ),  # Moonshot Kimi K2.5
        "nvidia.nemotron-nano-3-30b",  # NVIDIA Nemotron Nano 30B
        "qwen.qwen3-32b-v1:0",  # Qwen3 32B
        "qwen.qwen3-vl-235b-a22b",  # Qwen3 VL (vision)
        "writer.palmyra-vision-7b",  # Writer Palmyra Vision
        "writer.palmyra-x5-v1:0",  # Writer Palmyra X5
        "zai.glm-4.7-flash",  # Z.AI GLM-4.7 Flash
    ],
)

#: Models confirmed to support tool use via the Responses API, streaming or not.
_TOOL_MODEL_IDS = (
    "anthropic.claude-haiku-4-5-20251001-v1:0",  # Claude (reference)
    "amazon.nova-lite-v1:0",  # Amazon Nova
    "amazon.nova-2-lite-v1:0",  # Amazon Nova 2
    # "ai21.jamba-1-5-mini-v1:0",  # AI21 Jamba Mini
    # "ai21.jamba-1-5-large-v1:0",  # AI21 Jamba Large
    "deepseek.v3-v1:0",  # DeepSeek V3
    "deepseek.v3.2",  # DeepSeek V3.2 (newer revision)
    "meta.llama3-1-70b-instruct-v1:0",  # Meta Llama 3.1 70B
    "minimax.minimax-m2.5",  # MiniMax
    "mistral.mistral-large-2402-v1:0",  # Mistral Large
    "mistral.pixtral-large-2502-v1:0",  # Mistral Pixtral Large
    "moonshotai.kimi-k2.5",  # Moonshot Kimi K2.5
    "openai.gpt-oss-20b-1:0",  # OpenAI GPT-OSS 20B (Bedrock)
    "openai.gpt-oss-120b-1:0",  # OpenAI GPT-OSS 120B (Bedrock)
    "qwen.qwen3-32b-v1:0",  # Qwen3 32B
    "writer.palmyra-x4-v1:0",  # Writer Palmyra X4
    "writer.palmyra-x5-v1:0",  # Writer Palmyra X5
    "zai.glm-5",  # Z.AI GLM-5
)

#: Models confirmed to support non-streaming tool use via the Responses API.
_TOOL_MODELS = pytest.mark.parametrize("model", _TOOL_MODEL_IDS)

#: gpt-oss returns truncated JSON in streaming tool arguments (❌M).
_GPT_OSS_STREAMING_XFAIL = pytest.mark.xfail(
    strict=False,
    reason="gpt-oss returns truncated JSON in streaming tool arguments (❌M)",
)

#: Models confirmed to support tool use in streaming mode via the Responses API.
_STREAMING_TOOL_MODELS = pytest.mark.parametrize(
    "model",
    with_marks(
        _TOOL_MODEL_IDS,
        {
            "openai.gpt-oss-20b-1:0": _GPT_OSS_STREAMING_XFAIL,
            "openai.gpt-oss-120b-1:0": _GPT_OSS_STREAMING_XFAIL,
        },
    ),
)

#: A single deterministic read-only tool for all tool-use tests.
_LIST_DIR_TOOL: list[dict[str, object]] = [
    {
        "type": "function",
        "name": "list_directory",
        "description": "List the files and directories inside a filesystem path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to list"}
            },
            "required": ["path"],
        },
    }
]

#: Directory the tool tests ask the models to list.
_PROJECT_ROOT = str(REPO_ROOT)


# ---------------------------------------------------------------------------
# Tests: basic response, streaming, multi-turn
# ---------------------------------------------------------------------------


class TestMultiModelResponses:
    """Non-streaming, streaming and multi-turn generation per model family.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/models/chat/_adapters/_openai_responses.py:format_response
    """

    @pytest.mark.expensive
    @_BASIC_MODELS
    def test_basic_response(self, model: str, openai_client: OpenAI) -> None:
        """A plain string input returns one completed assistant message with usage.

        ``output_text`` is an SDK-side aggregation of the ``output_text`` parts of
        the message items, so it must equal their concatenation.  ``total_tokens``
        is the sum of the input and output counts on both the Converse and the
        Mantle-composed path.

        Ref: https://developers.openai.com/api/docs/guides/migrate-to-responses
             stdapi/models/chat/_adapters/_openai_responses.py:_extract_output_items
        """
        with _skip_unavailable_model(model):
            response = openai_client.responses.create(
                model=model,
                input="Reply with exactly one word: HELLO",
                max_output_tokens=512,
            )

        assert response.object == "response"
        assert response.status == "completed"
        assert response.error is None
        assert response.incomplete_details is None

        msg = next((i for i in response.output if i.type == "message"), None)
        assert msg is not None, f"Expected a message output item for {model!r}"
        assert msg.role == "assistant"
        assert msg.status == "completed"
        assert response.output_text, f"Expected non-empty output_text for {model!r}"
        assert response.output_text == "".join(
            part.text
            for item in response.output
            if item.type == "message"
            for part in item.content
            if part.type == "output_text"
        ), "output_text must aggregate the output_text parts of the message items"

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0
        assert response.usage.total_tokens == (
            response.usage.input_tokens + response.usage.output_tokens
        )

    @pytest.mark.expensive
    @_BASIC_MODELS
    def test_streaming_response(self, model: str, openai_client: OpenAI) -> None:
        """A stream opens with ``response.created`` and ends with ``response.completed``.

        Both backing paths emit the same wire grammar: ``response.created`` first,
        a ``sequence_number`` starting at 0 and incremented once per event, and a
        terminal event whose response snapshot carries the full text — so the
        concatenated deltas must equal the snapshot's ``output_text``.

        Ref: https://developers.openai.com/api/docs/guides/streaming-responses
             stdapi/models/chat/_adapters/_openai_responses.py:format_stream
             stdapi/models/chat/_mantle/_convert.py:_chat_stream_to_responses
        """
        accumulated = ""
        completed_event = None
        events: list[ResponseStreamEvent] = []

        with _skip_unavailable_model(model):
            stream = openai_client.responses.create(
                model=model,
                max_output_tokens=512,
                input="Reply with exactly three words: ONE TWO THREE",
                stream=True,
            )
            for event in stream:
                events.append(event)
                if event.type == "response.output_text.delta":
                    accumulated += event.delta
                elif event.type == "response.completed":
                    completed_event = event

        assert accumulated, f"No text deltas received for {model!r}"
        assert completed_event is not None, (
            f"No response.completed event received for {model!r}"
        )
        assert completed_event.response.status == "completed"
        assert completed_event.response.output_text == accumulated, (
            f"Deltas do not rebuild the final text for {model!r}"
        )
        assert completed_event.response.usage is not None
        assert completed_event.response.usage.output_tokens > 0

        assert events[0].type == "response.created", (
            f"Stream must open with response.created for {model!r}, "
            f"got {events[0].type!r}"
        )
        assert events[-1] is completed_event, (
            "response.completed must be the terminal event, got "
            f"{events[-1].type!r} last for {model!r}"
        )
        assert [event.sequence_number for event in events] == list(
            range(len(events))
        ), f"sequence_number must increase by one per event for {model!r}"

    @pytest.mark.expensive
    @_BASIC_MODELS
    def test_multi_turn_context_retention(
        self, model: str, openai_client: OpenAI
    ) -> None:
        """A prior ``assistant`` turn replayed in ``input`` is visible to the model.

        The identifier only exists in the first two items of the input array, so
        quoting it back proves the gateway mapped the ``user``/``assistant``
        message items onto Bedrock messages in order instead of keeping the last
        turn only.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/models/chat/_adapters/_openai_responses.py:map_input
        """
        with _skip_unavailable_model(model):
            response = openai_client.responses.create(
                model=model,
                max_output_tokens=256,
                input=[
                    {
                        "role": "user",
                        "content": "The test identifier for this session is ZEBRA99.",
                    },
                    {
                        "role": "assistant",
                        "content": "Understood, the test identifier is ZEBRA99.",
                    },
                    {
                        "role": "user",
                        "content": "What is the test identifier for this session?",
                    },
                ],
            )

        assert response.output_text, "Expected non-empty response"
        assert "ZEBRA99" in response.output_text, (
            f"Expected test identifier in response for {model!r}, "
            f"got: {response.output_text[:200]!r}"
        )
        msg = next((i for i in response.output if i.type == "message"), None)
        assert msg is not None, f"Expected a message output item for {model!r}"
        assert msg.role == "assistant"
        assert response.usage is not None
        # Three replayed turns are billed as input, not just the trailing question.
        assert response.usage.input_tokens > 0


# ---------------------------------------------------------------------------
# Tests: function tool calling
# ---------------------------------------------------------------------------


class TestMultiModelToolUse:
    """Function tool calling across tool-capable model families.

    Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
         stdapi/models/chat/_adapters/_openai_responses.py:_build_tool_config
    """

    @pytest.mark.expensive
    @_TOOL_MODELS
    def test_tool_call_single_turn(self, model: str, openai_client: OpenAI) -> None:
        """``tool_choice="required"`` yields a function_call item for the declared tool.

        Bedrock's ``toolChoice: {any: {}}`` forces a tool call, so the output must
        name the only declared tool and carry a JSON-object ``arguments`` string
        plus the ``call_id`` a later ``function_call_output`` item is keyed by.
        Models whose Bedrock family rejects a forced tool choice are skipped on the
        ``toolChoice`` validation error rather than failing.

        Ref: https://developers.openai.com/api/docs/guides/function-calling#tool-choice
             stdapi/models/chat/_adapters/_openai_responses.py:_map_tool_choice
        """
        with _skip_unavailable_model(model, unsupported=("toolchoice",)):
            response = openai_client.responses.create(  # type: ignore[call-overload]
                model=model,
                max_output_tokens=512,
                input=f"List the files in {_PROJECT_ROOT}",
                tools=_LIST_DIR_TOOL,
                tool_choice="required",
            )

        tool_calls = [i for i in response.output if i.type == "function_call"]
        assert tool_calls, (
            f"Expected at least one function_call output item for {model!r}; "
            f"output types: {[i.type for i in response.output]}"
        )
        tc = tool_calls[0]
        assert tc.name == "list_directory", (
            f"Unexpected tool name {tc.name!r} for {model!r}"
        )
        assert tc.call_id, "function_call items must carry the call_id to answer with"
        args = json.loads(tc.arguments)
        assert isinstance(args, dict), f"arguments must be a JSON object, got {args!r}"
        assert all(call.name == "list_directory" for call in tool_calls), (
            f"Only the declared tool may be called: {[c.name for c in tool_calls]}"
        )

    @pytest.mark.expensive
    @_STREAMING_TOOL_MODELS
    def test_streaming_tool_call(self, model: str, openai_client: OpenAI) -> None:
        """Streamed tool-call argument deltas concatenate to the ``.done`` arguments.

        Both backing paths build the ``.done`` event's ``arguments`` by joining the
        fragments already streamed as ``response.function_call_arguments.delta``,
        and close the item with a ``response.output_item.done`` carrying the
        complete ``function_call``.  The ``.done`` event's ``name`` field is not
        checked because the Mantle-composed path omits it; the closing output item
        carries the tool name on both paths.

        Ref: https://developers.openai.com/api/reference/resources/responses/streaming-events
             stdapi/models/chat/_adapters/_openai_responses.py:_emit_tool_done
             stdapi/models/chat/_mantle/_convert.py:_close_responses_tool
        """
        deltas: dict[str, str] = {}
        done_event = None
        done_items: list[ResponseFunctionToolCall] = []

        with _skip_unavailable_model(
            model, unsupported=("streaming mode", "toolchoice")
        ):
            stream = openai_client.responses.create(  # type: ignore[call-overload]
                model=model,
                max_output_tokens=512,
                input=f"List the files in {_PROJECT_ROOT}",
                tools=_LIST_DIR_TOOL,
                tool_choice="required",
                stream=True,
            )
            for event in stream:
                if event.type == "response.function_call_arguments.delta":
                    deltas[event.item_id] = deltas.get(event.item_id, "") + event.delta
                elif event.type == "response.function_call_arguments.done":
                    done_event = event
                elif (
                    event.type == "response.output_item.done"
                    and event.item.type == "function_call"
                ):
                    done_items.append(event.item)

        assert deltas, (
            f"Expected function_call_arguments.delta events for {model!r}, got 0"
        )
        assert done_event is not None, (
            f"Expected function_call_arguments.done event for {model!r}"
        )
        args = json.loads(done_event.arguments)
        assert isinstance(args, dict), f"arguments must be a JSON object, got {args!r}"
        assert done_event.arguments == deltas.get(done_event.item_id), (
            f"Argument deltas do not rebuild the final arguments for {model!r}"
        )

        closed = [item for item in done_items if item.id == done_event.item_id]
        assert closed, (
            f"No response.output_item.done closing {done_event.item_id!r} for {model!r}"
        )
        assert closed[0].name == "list_directory"
        assert closed[0].arguments == done_event.arguments
        assert closed[0].status == "completed"
        assert closed[0].call_id


# ---------------------------------------------------------------------------
# Tests: vision / image input
# ---------------------------------------------------------------------------


#: Vision-capable models tested on the Responses API route.
_VISION_MODELS = pytest.mark.parametrize(
    "model",
    with_marks(
        VISION_MODELS_OPENAI,
        {
            "mistral.pixtral-large-2502-v1:0": pytest.mark.xfail(
                strict=False,
                reason="Pixtral non-deterministically misidentifies colour of 1x1 PNG",
            ),
            "writer.palmyra-vision-7b": pytest.mark.xfail(
                strict=False,
                reason="Palmyra Vision non-deterministically misidentifies colour of 1x1 PNG",
            ),
        },
    ),
)


class TestVision:
    """Image input through ``input_image`` parts on vision-capable families.

    Ref: https://developers.openai.com/api/docs/guides/file-inputs
         stdapi/models/chat/_adapters/_openai_responses.py:_convert_input_content
    """

    @pytest.mark.expensive
    @_VISION_MODELS
    def test_image_color_recognition(self, model: str, openai_client: OpenAI) -> None:
        """A base64 ``input_image`` data URL reaches the model, which reports its color.

        The PNG is generated in-process, so no network fetch or Files API entry is
        involved: the gateway must decode the data URL into a Bedrock image block.
        ``"orange"`` is accepted alongside ``"red"`` because a 1x1 pixel leaves the
        models some latitude in naming the hue.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/modalities-image.html
             stdapi/models/chat/_adapters/_openai_responses.py:map_input
        """
        with _skip_unavailable_model(model):
            response = openai_client.responses.create(
                model=model,
                max_output_tokens=64,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{red_png_b64()}",
                                "detail": "low",
                            },
                            {
                                "type": "input_text",
                                "text": "What is the color of this image? Reply in one word.",
                            },
                        ],
                    }
                ],
            )

        text = response.output_text
        assert text, f"Expected non-empty response from {model!r}"
        assert any(color in text.lower() for color in ("red", "orange")), (
            f"Expected 'red' or 'orange' in response for {model!r}, got: {text!r}"
        )
        msg = next((i for i in response.output if i.type == "message"), None)
        assert msg is not None, f"Expected a message output item for {model!r}"
        assert msg.role == "assistant"
        assert response.usage is not None
        # The image is billed as input tokens, so the prompt cannot be text-only.
        assert response.usage.input_tokens > 0
