"""Serving a really deployed Amazon Bedrock Marketplace model endpoint.

**This lane provisions nothing.** It runs against an endpoint the operator
deployed around the run, and it **detects** that endpoint rather than being told
about it: ``marketplace_endpoint_arn`` lists the Marketplace endpoints of every
configured Region and picks the one that is ``REGISTERED`` and ``InService``. No
endpoint, no credentials, or no permission to list them, and the whole module
skips with the command that deploys one -- so a normal run never touches paid
infrastructure. ``TEST_MARKETPLACE_ENDPOINT_ARN`` in ``tests/.env`` still pins
one endpoint when several exist.

The claim under test is that an operator gets a served model out of a deployed
endpoint with **no per-endpoint configuration and no client-side change**: the
endpoint is discovered, published and invoked through ``bedrock-runtime`` with
its ARN as ``modelId``, so it rides the same Converse path as every other chat
model on all three client protocols. Everything a client can observe is
therefore driven through the vendor SDKs; what only the server can observe --
usage records, the price catalogue, a refusal decided before any backend call --
is driven through the in-process client, which is the only place that state is
readable.

The endpoint bills by the **instance-hour**, not by the token, so an extra
request costs nothing once it stands and only latency is spent. Token budgets
here are deliberately generous: the listing is a reasoning model and emits its
whole thinking block before the answer, so a small budget returns a truncated
thought and no answer at all.

Two properties of the backend are pinned here rather than asserted away, both
measured against the deployed endpoint on 2026-08-27 and both differing from a
Bedrock foundation model. They are limitations of the endpoint's own container,
not of the gateway, and the tests that cover them fail if AWS ever changes them:

- **Truncation is not reported.** ``Converse`` answers ``stopReason='end_turn'``
  even when the output was cut at ``maxTokens`` -- verified at ``maxTokens=1``,
  which returns exactly one token, mid-word, and still says ``end_turn``. The
  cap itself is honoured, so that is what is asserted.
- **A stream carries no usage.** ``ConverseStream`` emits ``messageStart``,
  ``contentBlockDelta`` and ``contentBlockStop`` only -- no ``messageStop`` and
  no ``metadata`` event, and ``metadata`` is where token counts live. A
  foundation model in the same account emits both. The gateway therefore has
  nothing to meter a streamed request from, and inventing counts is not an
  option; the non-streamed path is unaffected and is asserted strictly.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-marketplace-call-the-endpoint.html
     stdapi/models/marketplace_endpoints.py
     stdapi/models/__init__.py:usage_service
"""

from __future__ import annotations

from json import loads
from typing import TYPE_CHECKING, NamedTuple

import pytest
from aiobotocore.session import get_session
from anthropic import APIStatusError as AnthropicAPIStatusError
from openai import APIStatusError, BadRequestError, NotFoundError

from stdapi.api_errors import ApiError
from stdapi.config import SETTINGS
from stdapi.models import (
    MARKETPLACE_ENDPOINT_MODELS,
    MARKETPLACE_SERVICE,
    ModelRegionUnavailableError,
    initialize_bedrock_models,
    usage_service,
    validate_model,
)
from stdapi.utils import match_marketplace_endpoint_arn, match_sagemaker_hub_content_arn
from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from typing import Any

    from anthropic import Anthropic
    from openai import OpenAI
    from openai.types.chat import ChatCompletion
    from starlette.testclient import TestClient as TestClientType
    from types_aiobotocore_bedrock.literals import RegionName

#: Real money on a dedicated instance, and a deployment that takes 10-15 minutes.
pytestmark = [
    pytest.mark.expensive,
    pytest.mark.slow,
    # The catalogue and the usage log are in-process state, so there is no
    # remote form of this module: every test reads what this server discovered.
    pytest.mark.local,
    # One endpoint, one worker: these tests share a single paid resource.
    pytest.mark.xdist_group("marketplace_endpoint"),
]

#: Room for a reasoning model's thinking block plus its answer. Tokens are free here.
_TOKEN_BUDGET = 1024

#: Enough for a whole exchange the model has to finish by itself, for a stop reason.
_TURN_BUDGET = 4096

#: Small enough that any answer is cut off, for the length/max_tokens stop reason.
_TRUNCATING_BUDGET = 16

#: Prompt no model finishes inside the truncating budget above.
_LONG_PROMPT = "Write a very long detailed essay about the universe."

#: One tool, in the OpenAI Chat Completions shape.
_OPENAI_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City and state"}
                },
                "required": ["location"],
            },
        },
    }
]

#: The same tool in the Anthropic shape, which is a different translation path.
_ANTHROPIC_TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_weather",
        "description": "Get current weather information",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City and state"}
            },
            "required": ["location"],
        },
    }
]

#: A Region no deployment serves, for the endpoint-ARN region guard.
_UNCONFIGURED_REGION: RegionName = "me-central-1"


class MarketplaceEndpointUnderTest(NamedTuple):
    """The deployed endpoint, and what the catalogue made of it.

    ``listing`` is read back from ``GetMarketplaceModelEndpoint`` rather than
    from the catalogue, so the discovery assertions compare the published entry
    against AWS's own answer instead of against themselves.
    """

    arn: str
    region: str
    listing: str
    model_id: str


@pytest.fixture(scope="module")
async def marketplace_endpoint(
    marketplace_endpoint_arn: str,
    local_test_client: TestClientType,  # noqa: ARG001 - builds the AWS client pool
) -> MarketplaceEndpointUnderTest:
    """Discover the deployed endpoint and the model ID it publishes.

    The lane runs against the server's real configuration rather than patching
    it: ``AWS_BEDROCK_MARKETPLACE_ENDPOINTS_ENABLED=true`` belongs in
    ``tests/.env``, because settings are immutable after startup and the
    catalogue is built once for the whole session.

    ``local_test_client`` is required for its side effect, not its value:
    ``initialize_bedrock_models`` below reaches the ``bedrock`` control-plane
    pool, and that pool exists only inside the app's lifespan. Without the
    dependency the fixture works or raises ``KeyError: 'bedrock'`` depending on
    whether some earlier test happened to start the app first, which is not a
    property a fixture may have.

    Args:
        marketplace_endpoint_arn: The detected endpoint, or the pinned override.
        local_test_client: The in-process app, entered so its lifespan has run.

    Returns:
        The endpoint and the catalogue entry it produced.

    Ref: stdapi/models/marketplace_endpoints.py:collect_marketplace_endpoint_models
         tests/conftest.py:test_client
    """
    if not SETTINGS.aws_bedrock_marketplace_endpoints_enabled:
        pytest.skip(
            "An endpoint is deployed but this server does not serve it: set "
            "AWS_BEDROCK_MARKETPLACE_ENDPOINTS_ENABLED=true in tests/.env"
        )
    arn_match = match_marketplace_endpoint_arn(marketplace_endpoint_arn)
    assert arn_match, (
        f"Not a model endpoint ARN: {marketplace_endpoint_arn}. "
        "TEST_MARKETPLACE_ENDPOINT_ARN overrides detection and must name one."
    )
    region = arn_match.group("region")
    if region not in SETTINGS.aws_bedrock_regions:
        pytest.skip(
            f"The endpoint is in {region}, which AWS_BEDROCK_REGIONS does not serve"
        )

    session = get_session()
    async with session.create_client("bedrock", region_name=region) as bedrock:
        detail = (
            await bedrock.get_marketplace_model_endpoint(
                endpointArn=marketplace_endpoint_arn
            )
        )["marketplaceModelEndpoint"]
    listing_match = match_sagemaker_hub_content_arn(detail["modelSourceIdentifier"])
    assert listing_match, (
        "The endpoint was not registered from SageMaker public-hub content, so "
        "it publishes under no listing name and this lane cannot address it"
    )

    await initialize_bedrock_models()
    served = [
        model_id
        for model_id, model in MARKETPLACE_ENDPOINT_MODELS.items()
        if marketplace_endpoint_arn in (model.marketplace_endpoints or {}).values()
    ]
    assert served, (
        "The deployed endpoint was not discovered: it must be REGISTERED and "
        "InService, and the server role needs "
        "bedrock:ListMarketplaceModelEndpoints and bedrock:GetMarketplaceModelEndpoint"
    )
    return MarketplaceEndpointUnderTest(
        arn=marketplace_endpoint_arn,
        region=region,
        listing=listing_match.group("name"),
        model_id=served[0],
    )


@pytest.fixture(scope="module")
def marketplace_model(marketplace_endpoint: MarketplaceEndpointUnderTest) -> str:
    """The model ID a client asks for, which is all most tests need."""
    return marketplace_endpoint.model_id


@pytest.fixture(scope="module")
def marketplace_tool_call(
    openai_client: OpenAI, marketplace_model: str
) -> ChatCompletion:
    """Force one tool call against the endpoint, or skip the tool tests.

    Tool support belongs to the listing's container, not to the gateway, so it
    is probed once rather than assumed: a listing that refuses a tool
    configuration, or accepts one and calls nothing, skips with the reason
    instead of failing a claim the feature never made.

    Returns:
        The completion carrying the tool call.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolChoice.html
    """
    try:
        response: ChatCompletion = openai_client.chat.completions.create(  # type: ignore[call-overload]
            model=marketplace_model,
            messages=[{"role": "user", "content": "What's the weather in New York?"}],
            tools=_OPENAI_TOOLS,
            tool_choice="required",
            max_completion_tokens=_TOKEN_BUDGET,
        )
    except APIStatusError as error:
        pytest.skip(f"The listing does not accept a tool configuration: {error}")
    if not response.choices[0].message.tool_calls:
        pytest.skip(
            "The listing accepts a tool configuration but the model produced no "
            "tool call, so there is no round trip to assert on"
        )
    return response


class TestDiscovery:
    """The endpoint becomes a served model with no per-endpoint configuration.

    Ref: stdapi/models/marketplace_endpoints.py:_model_from_endpoint
    """

    def test_the_published_id_is_the_listing_name_aws_reports(
        self, marketplace_endpoint: MarketplaceEndpointUnderTest
    ) -> None:
        """Discovery alone names the model, from the listing behind the endpoint.

        Nothing in ``tests/.env`` or in the settings names this model: the ID a
        client asks for is derived from the endpoint's own
        ``modelSourceIdentifier``, which is what "no per-endpoint configuration"
        means concretely.

        Ref: stdapi/models/marketplace_endpoints.py:_model_from_endpoint
        """
        assert marketplace_endpoint.model_id == marketplace_endpoint.listing

    def test_the_catalogue_entry_carries_the_endpoint_service_and_provider(
        self, marketplace_endpoint: MarketplaceEndpointUnderTest
    ) -> None:
        """The entry is attributed to the endpoint service, with a real provider.

        The provider is read off the listing's vendor prefix, so it must be a
        name rather than the generic fallback for an unrecognised one.

        Ref: stdapi/models/marketplace_endpoints.py:_listing_provider
        """
        model = MARKETPLACE_ENDPOINT_MODELS[marketplace_endpoint.model_id]

        assert model.service == MARKETPLACE_SERVICE
        assert model.provider
        assert model.name
        assert usage_service(marketplace_endpoint.model_id).value == (
            "bedrock-marketplace"
        )

    def test_the_endpoint_is_published_in_v1_models(
        self, openai_client: OpenAI, marketplace_endpoint: MarketplaceEndpointUnderTest
    ) -> None:
        """``/v1/models`` lists the endpoint like any other model, owned by its vendor.

        Ref: https://developers.openai.com/api/reference/resources/models/methods/list
             stdapi/routes/openai_models.py
        """
        listed = {model.id: model for model in openai_client.models.list()}

        assert marketplace_endpoint.model_id in listed, (
            f"{marketplace_endpoint.model_id} is missing from /v1/models"
        )
        entry = listed[marketplace_endpoint.model_id]
        assert entry.owned_by == (
            MARKETPLACE_ENDPOINT_MODELS[marketplace_endpoint.model_id].provider
        )
        assert entry.object == "model"

    def test_retrieving_the_model_answers_for_the_endpoint(
        self, openai_client: OpenAI, marketplace_model: str
    ) -> None:
        """``/v1/models/{id}`` resolves the discovered ID, not only the list.

        Ref: https://developers.openai.com/api/reference/resources/models/methods/retrieve
        """
        model = openai_client.models.retrieve(marketplace_model)

        assert model.id == marketplace_model

    def test_the_endpoint_is_published_in_search_models(
        self, local_test_client: TestClientType, marketplace_model: str, api_key: str
    ) -> None:
        """``/search_models`` publishes it as a text chat model, with no ARN in sight.

        The ARN carries the account ID, which is backend detail the catalogue
        never publishes. Neither counting route may be advertised either: Bedrock's
        token counter takes a foundation model identifier.

        Ref: stdapi/models/marketplace_endpoints.py:_model_from_endpoint
             stdapi/models/__init__.py:reject_unsupported_token_counting
        """
        response = local_test_client.get(
            "/search_models?route=/v1/chat/completions&output_modalities=TEXT",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        assert response.status_code == 200
        entries = [m for m in response.json() if m["id"] == marketplace_model]
        assert entries, f"{marketplace_model} is missing from the catalogue"
        entry = entries[0]
        assert entry["input_modalities"] == ["TEXT"]
        assert entry["output_modalities"] == ["TEXT"]
        assert entry["response_streaming"] is True
        assert "/v1/chat/completions" in entry["supported_routes"]
        assert "/v1/responses" in entry["supported_routes"]
        assert "/anthropic/v1/messages" in entry["supported_routes"]
        assert "/v1/responses/input_tokens" not in entry["supported_routes"]
        assert "/anthropic/v1/messages/count_tokens" not in entry["supported_routes"]
        assert "arn:aws:sagemaker" not in response.text

    def test_the_endpoint_is_pinned_to_its_own_region(
        self, marketplace_endpoint: MarketplaceEndpointUnderTest
    ) -> None:
        """The entry serves one Region and offers no substitute elsewhere.

        An endpoint has no cross-region form, so it must not participate in
        failover: a request routed to another Region would be a guaranteed
        ``ValidationException`` rather than a retry that helps.

        Ref: stdapi/models/__init__.py:ModelDetails._get_endpoint_id
        """
        model = MARKETPLACE_ENDPOINT_MODELS[marketplace_endpoint.model_id]
        region: RegionName = marketplace_endpoint.region  # type: ignore[assignment]

        assert model.regions == [region]
        assert model.get_id(region, inference_profile=True) == marketplace_endpoint.arn
        with pytest.raises(ModelRegionUnavailableError):
            model.get_id(_UNCONFIGURED_REGION, inference_profile=True)


class TestChatCompletions:
    """``/v1/chat/completions`` serves the endpoint through the OpenAI SDK.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
    """

    def test_a_completion_is_served(
        self, openai_client: OpenAI, marketplace_model: str
    ) -> None:
        """A non-streamed completion is served with the endpoint ARN as ``modelId``.

        This is the assertion the whole lane exists for: nothing short of a
        deployed endpoint can show that ``Converse`` accepts one, and the client
        that gets the answer is the unmodified vendor SDK.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-marketplace-call-the-endpoint.html
        """
        response = openai_client.chat.completions.create(
            model=marketplace_model,
            messages=[{"role": "user", "content": "Say hello."}],
            max_completion_tokens=_TOKEN_BUDGET,
        )

        assert response.object == "chat.completion"
        assert response.model == marketplace_model
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
        assert isinstance(response.choices[0].message.content, str)
        assert response.choices[0].finish_reason in {"stop", "length"}
        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0

    def test_a_streamed_completion_is_served(
        self, openai_client: OpenAI, marketplace_model: str
    ) -> None:
        """A streamed completion is served, since the catalogue advertises streaming.

        ``ConverseStream`` takes the same ``modelId`` shape as ``Converse``,
        which is why ``response_streaming`` is published as true; a failure here
        means that published flag is wrong rather than that the test is.

        Ref: stdapi/models/marketplace_endpoints.py:_model_from_endpoint
        """
        stream = openai_client.chat.completions.create(
            model=marketplace_model,
            messages=[{"role": "user", "content": "Count to five."}],
            max_completion_tokens=_TOKEN_BUDGET,
            stream=True,
        )
        chunks = []
        content = ""
        finish_reasons = []
        for chunk in stream:
            chunks.append(chunk)
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content:
                content += chunk.choices[0].delta.content
            if chunk.choices[0].finish_reason is not None:
                finish_reasons.append(chunk.choices[0].finish_reason)

        assert len(chunks) > 1, "A stream must deliver more than one event"
        assert chunks[0].choices[0].delta.role == "assistant"
        assert content
        assert len(finish_reasons) == 1
        assert finish_reasons[0] in {"stop", "length"}

    @pytest.mark.retry("an open-weights model does not always repeat a given word")
    def test_a_multi_turn_conversation_keeps_the_earlier_turns(
        self, openai_client: OpenAI, marketplace_model: str
    ) -> None:
        """An assistant turn sent back in the history reaches the endpoint.

        Converse carries the history as alternating messages, so a fact stated
        two turns earlier has to survive the translation to be answerable.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        """
        response = openai_client.chat.completions.create(
            model=marketplace_model,
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
                {"role": "user", "content": "What is my name?"},
            ],
            max_completion_tokens=_TURN_BUDGET,
        )

        content = response.choices[0].message.content
        assert content is not None
        assert "alice" in content.lower()

    @pytest.mark.retry("an open-weights model does not always obey a one-word rule")
    def test_a_system_prompt_steers_the_completion(
        self, openai_client: OpenAI, marketplace_model: str
    ) -> None:
        """A system message becomes Converse's own ``system`` block and is obeyed.

        A system prompt is not another user turn: it is a separate Converse
        field, and a translation that dropped or merged it would leave the
        instruction unheeded rather than raise.

        Ref: stdapi/models/chat/_adapters/_openai_chat_completion.py
        """
        response = openai_client.chat.completions.create(
            model=marketplace_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Whatever the user writes, your final answer is exactly "
                        "the word TEAL and nothing else."
                    ),
                },
                {"role": "user", "content": "What is the capital of France?"},
            ],
            max_completion_tokens=_TURN_BUDGET,
        )

        content = response.choices[0].message.content
        assert content is not None
        assert "teal" in content.lower()

    def test_max_completion_tokens_caps_the_output(
        self, openai_client: OpenAI, marketplace_model: str
    ) -> None:
        """The budget reaches the endpoint and stops it exactly at the cap.

        Spending the whole budget on a prompt no model finishes inside it is
        what proves ``max_completion_tokens`` became Bedrock's
        ``inferenceConfig.maxTokens``: a budget that never arrived would answer
        with more tokens than were allowed.

        The stop reason cannot say so. This listing reports ``end_turn``
        whatever happens -- measured at ``maxTokens=1``, which returns one
        token, mid-word, and still says ``end_turn`` -- so the gateway maps it
        to ``stop``, correctly, because that is what the backend said. Pinned
        rather than skipped: a container that starts reporting ``max_tokens``
        must show up here as a failure to be turned into ``length``.

        Ref: stdapi/models/chat/_adapters/_openai_chat_completion.py:_FINISH_REASONS
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        """
        response = openai_client.chat.completions.create(
            model=marketplace_model,
            messages=[{"role": "user", "content": _LONG_PROMPT}],
            max_completion_tokens=_TRUNCATING_BUDGET,
        )

        assert response.usage is not None
        assert response.usage.completion_tokens == _TRUNCATING_BUDGET, (
            "The cap did not reach the endpoint's inferenceConfig"
        )
        assert response.choices[0].finish_reason == "stop", (
            "The listing has started reporting truncation: map it to 'length'"
        )

    def test_temperature_is_accepted(
        self, openai_client: OpenAI, marketplace_model: str
    ) -> None:
        """``temperature`` reaches the endpoint's ``inferenceConfig`` unrejected.

        The sampling effect itself is not observable, so acceptance and a
        well-formed completion are what is asserted; a parameter the endpoint
        would not take fails the call outright.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InferenceConfiguration.html
        """
        response = openai_client.chat.completions.create(
            model=marketplace_model,
            messages=[{"role": "user", "content": "Say hello."}],
            max_completion_tokens=_TOKEN_BUDGET,
            temperature=0.0,
        )

        assert isinstance(response.choices[0].message.content, str)


class TestResponses:
    """``/v1/responses`` serves the same endpoint, unchanged on the client side.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
    """

    def test_a_response_is_served(
        self, openai_client: OpenAI, marketplace_model: str
    ) -> None:
        """A non-streamed Response completes and carries an assistant message.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
        """
        response = openai_client.responses.create(
            model=marketplace_model, input="Say hello.", max_output_tokens=_TOKEN_BUDGET
        )

        assert response.id
        assert response.model == marketplace_model
        assert response.status in {"completed", "incomplete"}
        message = next(
            (item for item in response.output if item.type == "message"), None
        )
        assert message is not None, "Expected a message item in the response output"
        assert message.role == "assistant"
        assert response.usage is not None
        assert response.usage.output_tokens > 0

    def test_a_streamed_response_is_served(
        self, openai_client: OpenAI, marketplace_model: str
    ) -> None:
        """The Responses event sequence opens and closes around the endpoint's output.

        Ref: https://developers.openai.com/api/docs/guides/streaming-responses
        """
        stream = openai_client.responses.create(
            model=marketplace_model,
            input="Count to five.",
            max_output_tokens=_TOKEN_BUDGET,
            stream=True,
        )
        events = list(stream)

        assert events[0].type == "response.created"
        assert events[-1].type in {"response.completed", "response.incomplete"}
        assert any(event.type == "response.output_text.delta" for event in events)

    @pytest.mark.retry("an open-weights model does not always obey a one-word rule")
    def test_instructions_steer_the_response(
        self, openai_client: OpenAI, marketplace_model: str
    ) -> None:
        """``instructions`` is the Responses system prompt and reaches the endpoint.

        Ref: https://developers.openai.com/api/docs/guides/text
        """
        response = openai_client.responses.create(
            model=marketplace_model,
            instructions=(
                "Whatever the user writes, your final answer is exactly the word "
                "TEAL and nothing else."
            ),
            input="What is the capital of France?",
            max_output_tokens=_TURN_BUDGET,
        )

        assert "teal" in response.output_text.lower()


class TestAnthropicMessages:
    """``/anthropic/v1/messages`` serves the same endpoint through the Anthropic SDK.

    The dialect is the point: a Marketplace listing is not a Claude model, and
    the gateway is what makes an Anthropic client able to talk to it at all.

    Ref: https://platform.claude.com/docs/en/api/messages
    """

    def test_a_message_is_served(
        self, anthropic_client: Anthropic, marketplace_model: str
    ) -> None:
        """A non-streamed message completes with an Anthropic-shaped envelope.

        Ref: https://platform.claude.com/docs/en/api/messages
        """
        response = anthropic_client.messages.create(
            model=marketplace_model,
            max_tokens=_TURN_BUDGET,
            messages=[{"role": "user", "content": "Say hello."}],
        )

        assert response.type == "message"
        assert response.role == "assistant"
        assert response.id.startswith("msg_")
        assert response.content[0].type == "text"
        assert response.stop_reason == "end_turn"
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    def test_a_streamed_message_is_served(
        self, anthropic_client: Anthropic, marketplace_model: str
    ) -> None:
        """The Anthropic event sequence opens and closes around the endpoint's output.

        The endpoint sends no ``messageStop``, so the closing ``message_delta``
        carries the stop reason the gateway defaults to rather than none at all:
        an SDK agent loop branches on ``stop_reason``, and the same request
        answers ``end_turn`` unstreamed.

        Ref: https://platform.claude.com/docs/en/api/messages-streaming
        """
        stream = anthropic_client.messages.create(
            model=marketplace_model,
            max_tokens=_TOKEN_BUDGET,
            messages=[{"role": "user", "content": "Count to five."}],
            stream=True,
        )
        event_types = []
        text = ""
        stop_reasons = []
        for event in stream:
            event_types.append(event.type)
            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                text += event.delta.text
            elif event.type == "message_delta":
                stop_reasons.append(event.delta.stop_reason)

        assert event_types[0] == "message_start"
        assert "content_block_delta" in event_types
        assert event_types[-1] == "message_stop"
        assert text
        assert stop_reasons == ["end_turn"]

    def test_max_tokens_caps_the_message(
        self, anthropic_client: Anthropic, marketplace_model: str
    ) -> None:
        """``max_tokens`` reaches the endpoint and stops it exactly at the cap.

        The Anthropic dialect would report a truncation as ``max_tokens``, and
        this listing never produces one to report: it answers ``end_turn`` even
        at ``maxTokens=1``. Pinned for the same reason as the OpenAI route --
        the day the container reports truncation, this is where it surfaces.

        Ref: https://platform.claude.com/docs/en/api/handling-stop-reasons
        """
        response = anthropic_client.messages.create(
            model=marketplace_model,
            max_tokens=_TRUNCATING_BUDGET,
            messages=[{"role": "user", "content": _LONG_PROMPT}],
        )

        assert response.usage.output_tokens == _TRUNCATING_BUDGET, (
            "The cap did not reach the endpoint's inferenceConfig"
        )
        assert response.stop_reason == "end_turn", (
            "The listing has started reporting truncation: map it to 'max_tokens'"
        )

    @pytest.mark.retry("an open-weights model does not always obey a one-word rule")
    def test_a_system_prompt_steers_the_message(
        self, anthropic_client: Anthropic, marketplace_model: str
    ) -> None:
        """``system`` is a separate Anthropic field and reaches the endpoint as one.

        Ref: stdapi/models/chat/_adapters/_anthropic_message.py:_map_system_blocks
        """
        response = anthropic_client.messages.create(
            model=marketplace_model,
            max_tokens=_TURN_BUDGET,
            system=(
                "Whatever the user writes, your final answer is exactly the word "
                "TEAL and nothing else."
            ),
            messages=[{"role": "user", "content": "What is the capital of France?"}],
        )

        text = "".join(block.text for block in response.content if block.type == "text")
        assert "teal" in text.lower()


class TestToolCalling:
    """Tool calling, when the deployed listing's container supports it.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolConfiguration.html
    """

    def test_a_tool_call_is_well_formed(
        self, marketplace_tool_call: ChatCompletion
    ) -> None:
        """The forced call arrives with an ID, the tool's name and parsable arguments.

        Which tool the model picks is its own business; that the gateway
        translated Converse's ``toolUse`` into a usable OpenAI ``tool_call`` is
        not.

        Ref: stdapi/models/chat/_adapters/_openai_common.py:parse_tool_content
        """
        choice = marketplace_tool_call.choices[0]
        assert choice.message.tool_calls is not None
        call = choice.message.tool_calls[0]

        assert call.id
        assert call.type == "function"
        assert call.function.name == "get_weather"
        assert isinstance(loads(call.function.arguments), dict)
        assert choice.finish_reason in {"tool_calls", "stop"}

    def test_a_tool_result_completes_the_round_trip(
        self,
        openai_client: OpenAI,
        marketplace_model: str,
        marketplace_tool_call: ChatCompletion,
    ) -> None:
        """A tool result sent back is accepted and answered in text.

        The second turn is what proves the translation is bidirectional: the
        gateway has to render the ``tool`` message as a Converse ``toolResult``
        block the endpoint accepts.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultBlock.html
        """
        tool_calls = marketplace_tool_call.choices[0].message.tool_calls
        assert tool_calls
        call = tool_calls[0]
        assert call.type == "function"

        response = openai_client.chat.completions.create(
            model=marketplace_model,
            messages=[
                {"role": "user", "content": "What's the weather in New York?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": '{"temperature": "22C", "condition": "sunny"}',
                },
            ],
            tools=_OPENAI_TOOLS,  # type: ignore[arg-type]
            max_completion_tokens=_TURN_BUDGET,
        )

        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].finish_reason in {"stop", "length", "tool_calls"}

    def test_the_anthropic_dialect_serves_a_tool_call(
        self,
        anthropic_client: Anthropic,
        marketplace_model: str,
        marketplace_tool_call: ChatCompletion,
    ) -> None:
        """An Anthropic-shaped tool definition reaches the same endpoint.

        The Anthropic dialect translates ``input_schema`` and ``tool_choice``
        differently from the OpenAI one, so a listing that supports tools has to
        be proved on both paths rather than one.

        Ref: stdapi/models/chat/_adapters/_anthropic_message.py
        """
        try:
            response = anthropic_client.messages.create(  # type: ignore[call-overload]
                model=marketplace_model,
                max_tokens=_TOKEN_BUDGET,
                messages=[
                    {"role": "user", "content": "What's the weather in New York?"}
                ],
                tools=_ANTHROPIC_TOOLS,
                tool_choice={"type": "any"},
            )
        except AnthropicAPIStatusError as error:
            pytest.skip(
                f"The Anthropic dialect refused the tool configuration: {error}"
            )

        blocks = [block for block in response.content if block.type == "tool_use"]
        if not blocks:
            pytest.skip("The model produced no tool call on the Anthropic dialect")
        assert response.stop_reason == "tool_use"
        assert blocks[0].name == "get_weather"
        assert blocks[0].id.startswith("toolu_")
        assert isinstance(blocks[0].input, dict)


class TestUsageAndPricing:
    """AWS bills the endpoint by the instance-hour, so no request carries a price.

    Ref: stdapi/usage.py:UNPRICED_SERVICES
         stdapi/models/__init__.py:usage_service
    """

    def test_usage_is_recorded_with_no_price_and_no_warning(
        self,
        local_test_client: TestClientType,
        marketplace_model: str,
        api_key: str,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """Token counts are recorded apart from bedrock-runtime, and priced at nothing.

        AWS publishes no per-token rate for an endpoint, so the record must carry
        the counts, resolve no cost, and raise no pricing-miss warning -- that
        signal is reserved for a real catalogue gap, and a false one there
        trains the operator to ignore it.

        Ref: stdapi/usage.py:_apply_record_cost
        """
        capfd.readouterr()
        response = local_test_client.post(
            "/v1/chat/completions",
            json={
                "model": marketplace_model,
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_completion_tokens": _TOKEN_BUDGET,
            },
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert response.status_code == 200, response.text
        captured = capfd.readouterr().out

        entries = logged_usage_entries(
            captured,
            service="bedrock-marketplace",
            operation="/v1/chat/completions",
            model=marketplace_model,
        )
        assert entries, "Expected a Marketplace model endpoint usage entry"
        assert len(entries) == 1, "One request must be billed exactly once"
        assert entries[0]["input_tokens"] > 0
        assert entries[0]["output_tokens"] > 0
        assert (
            entries[0]["output_tokens"]
            == (response.json()["usage"]["completion_tokens"])
        ), "The logged usage must match the usage returned to the client"
        assert "cost" not in entries[0], (
            "An instance-hour backend must resolve no per-token cost"
        )
        assert not logged_usage_entries(
            captured, service="bedrock-runtime", model=marketplace_model
        ), "The endpoint must not be billed as an ordinary bedrock-runtime model"
        assert "No price found" not in captured, (
            "A backend with no published per-token rate is not a pricing miss"
        )

    def test_a_streamed_request_carries_no_usage_the_backend_never_sent(
        self,
        local_test_client: TestClientType,
        marketplace_model: str,
        api_key: str,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A streamed request is served but records no usage, and says so on purpose.

        ``ConverseStream`` against this endpoint emits ``messageStart``,
        ``contentBlockDelta`` and ``contentBlockStop`` and nothing else -- no
        ``messageStop``, and no ``metadata``, which is the only event carrying
        token counts. A foundation model in the same account emits both, and its
        streamed request is metered normally, so this is the endpoint's
        container and not the gateway's stream handling.

        There is nothing to fix in the gateway short of inventing counts, which
        is worse than reporting none: an operator summing a fabricated zero
        would read it as "no output was produced". The bill is instance-hours
        either way, so nothing chargeable is lost -- what is lost is the record
        that the request happened, and this test is what will notice the day AWS
        starts sending the event.

        Ref: stdapi/models/chat/_adapters/_openai_common.py:_stream_usage
             https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ConverseStreamOutput.html
        """
        capfd.readouterr()
        with local_test_client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": marketplace_model,
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_completion_tokens": _TOKEN_BUDGET,
                "stream": True,
            },
            headers={"Authorization": f"Bearer {api_key}"},
        ) as response:
            assert response.status_code == 200
            lines = [line for line in response.iter_lines() if line.startswith("data:")]
        captured = capfd.readouterr().out

        assert lines[-1] == "data: [DONE]"
        assert len(lines) > 1, "The stream must still deliver content"
        assert not logged_usage_entries(
            captured, service="bedrock-marketplace", model=marketplace_model
        ), (
            "The endpoint has started sending a ConverseStream metadata event: "
            "assert the streamed usage entry strictly, as the non-streamed one is"
        )
        # Whatever else changes, a backend with no published per-token rate must
        # never read as a pricing miss.
        assert "No price found" not in captured

    def test_model_pricing_reports_the_endpoint_service_and_no_rate(
        self, local_test_client: TestClientType, marketplace_model: str, api_key: str
    ) -> None:
        """``/model_pricing`` names the serving endpoint and publishes no unit price.

        Reporting a per-token rate here would be inventing one: the operator's
        bill is instance-hours, which no request can attribute.

        Ref: stdapi/routes/core_models.py:_preferred_service
        """
        response = local_test_client.get(
            f"/model_pricing?model={marketplace_model}",
            headers={"Authorization": f"Bearer {api_key}"},
        )

        if response.status_code == 503:
            pytest.skip("Cost tracking is disabled on this server")
        assert response.status_code == 200
        card = response.json()[0]
        assert card["service"] == "bedrock-marketplace"
        assert card["prices"] == []


class TestEndpointArnAsModelId:
    """Naming the endpoint ARN directly, under the setting that allows it.

    The published listing name is one way in; the ARN is the other, and it is
    what an operator falls back to when two endpoints share a listing name. It
    resolves through a synthesized catalogue entry rather than a discovered one,
    which is a second, independent path into the same invocation.

    Ref: stdapi/config.py:_Settings.aws_bedrock_allow_marketplace_endpoint_arn
         stdapi/models/__init__.py:_marketplace_endpoint_from_arn
    """

    def test_a_completion_is_served_for_the_arn(
        self,
        openai_client: OpenAI,
        marketplace_endpoint: MarketplaceEndpointUnderTest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``/v1/chat/completions`` serves a request naming the endpoint ARN.

        The synthesized entry has to survive the whole invocation path, including
        the region resolution that re-reads the model ID out of the catalogue --
        a lookup the ARN is not in.

        Ref: stdapi/models/__init__.py:get_model_details
        """
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_allow_marketplace_endpoint_arn", True
        )

        response = openai_client.chat.completions.create(
            model=marketplace_endpoint.arn,
            messages=[{"role": "user", "content": "Say hello."}],
            max_completion_tokens=_TOKEN_BUDGET,
        )

        assert isinstance(response.choices[0].message.content, str)
        assert response.usage is not None
        assert response.usage.completion_tokens > 0

    def test_a_response_is_served_for_the_arn(
        self,
        openai_client: OpenAI,
        marketplace_endpoint: MarketplaceEndpointUnderTest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``/v1/responses`` serves a request naming the endpoint ARN.

        Ref: stdapi/models/__init__.py:_marketplace_endpoint_from_arn
        """
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_allow_marketplace_endpoint_arn", True
        )

        response = openai_client.responses.create(
            model=marketplace_endpoint.arn,
            input="Say hello.",
            max_output_tokens=_TOKEN_BUDGET,
        )

        assert response.status in {"completed", "incomplete"}
        assert response.output

    def test_a_message_is_served_for_the_arn(
        self,
        anthropic_client: Anthropic,
        marketplace_endpoint: MarketplaceEndpointUnderTest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``/anthropic/v1/messages`` serves a request naming the endpoint ARN.

        Ref: stdapi/models/__init__.py:_marketplace_endpoint_from_arn
        """
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_allow_marketplace_endpoint_arn", True
        )

        response = anthropic_client.messages.create(
            model=marketplace_endpoint.arn,
            max_tokens=_TOKEN_BUDGET,
            messages=[{"role": "user", "content": "Say hello."}],
        )

        assert response.content[0].type == "text"
        assert response.usage.output_tokens > 0

    def test_the_usage_of_an_arn_request_is_unpriced_too(
        self,
        local_test_client: TestClientType,
        marketplace_endpoint: MarketplaceEndpointUnderTest,
        api_key: str,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """The ARN path meters on the endpoint service, not on bedrock-runtime.

        The synthesized entry carries its own service, and getting that wrong
        would silently price the request against a foundation model's rate card.

        Ref: stdapi/models/__init__.py:usage_service
        """
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_allow_marketplace_endpoint_arn", True
        )
        capfd.readouterr()
        response = local_test_client.post(
            "/v1/chat/completions",
            json={
                "model": marketplace_endpoint.arn,
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_completion_tokens": _TOKEN_BUDGET,
            },
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert response.status_code == 200, response.text
        captured = capfd.readouterr().out

        entries = logged_usage_entries(captured, service="bedrock-marketplace")
        assert entries, "Expected a Marketplace model endpoint usage entry"
        assert "cost" not in entries[0]
        assert "No price found" not in captured


class TestRefusals:
    """Everything the gateway refuses before it can cost anything.

    Ref: stdapi/api_errors.py
    """

    def test_the_arn_is_refused_when_the_setting_is_off(
        self,
        openai_client: OpenAI,
        marketplace_endpoint: MarketplaceEndpointUnderTest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Naming an endpoint ARN is opt-in, because it directs paid traffic.

        Ref: stdapi/config.py:_Settings.aws_bedrock_allow_marketplace_endpoint_arn
        """
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_allow_marketplace_endpoint_arn", False
        )

        with pytest.raises(APIStatusError) as raised:
            openai_client.chat.completions.create(
                model=marketplace_endpoint.arn,
                messages=[{"role": "user", "content": "Say hello."}],
                max_completion_tokens=_TRUNCATING_BUDGET,
            )

        assert raised.value.status_code in {400, 404}
        assert "not allowed by server configuration" in str(raised.value)

    def test_an_arn_outside_the_configured_regions_is_refused(
        self,
        openai_client: OpenAI,
        marketplace_endpoint: MarketplaceEndpointUnderTest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An ARN naming an unserved Region is refused before any client is built.

        This is the guard that keeps a data-residency deployment residential: a
        client must not be able to route paid traffic into a Region the operator
        never approved by naming an ARN in it.

        Ref: stdapi/models/__init__.py:_marketplace_endpoint_from_arn
        """
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_allow_marketplace_endpoint_arn", True
        )
        foreign_arn = marketplace_endpoint.arn.replace(
            f":{marketplace_endpoint.region}:", f":{_UNCONFIGURED_REGION}:"
        )
        assert _UNCONFIGURED_REGION not in SETTINGS.aws_bedrock_regions

        with pytest.raises(BadRequestError) as raised:
            openai_client.chat.completions.create(
                model=foreign_arn,
                messages=[{"role": "user", "content": "Say hello."}],
                max_completion_tokens=_TRUNCATING_BUDGET,
            )

        assert "is not a configured Bedrock region" in str(raised.value)

    def test_an_undeployed_endpoint_arn_fails_as_a_client_error(
        self,
        openai_client: OpenAI,
        marketplace_endpoint: MarketplaceEndpointUnderTest,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A well-formed ARN for an endpoint that does not exist is a clean 4xx.

        Nothing validates the ARN against the account before the call, so the
        backend is what refuses it -- and that refusal must arrive as the
        upstream-shaped error, never as a 500 or a forwarded AWS exception.

        Ref: stdapi/api_errors.py
        """
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_allow_marketplace_endpoint_arn", True
        )
        missing_arn = (
            f"arn:aws:sagemaker:{marketplace_endpoint.region}:"
            f"{marketplace_endpoint.arn.split(':')[4]}:endpoint/stdapi-ai-absent"
        )

        with pytest.raises(APIStatusError) as raised:
            openai_client.chat.completions.create(
                model=missing_arn,
                messages=[{"role": "user", "content": "Say hello."}],
                max_completion_tokens=_TRUNCATING_BUDGET,
            )

        assert 400 <= raised.value.status_code < 500, (
            "An endpoint that does not exist is the caller's mistake, not a server fault"
        )
        message = str(raised.value)
        assert "ValidationException" not in message
        assert "botocore" not in message
        assert "Traceback" not in message

    def test_an_unknown_model_id_is_not_found(self, openai_client: OpenAI) -> None:
        """A listing name nothing publishes is a 404, as for any other model.

        The Marketplace catalogue merges into the ordinary one, so a miss must
        not become a Marketplace-shaped error of its own.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
             stdapi/api_errors.py:UnsupportedModelError
        """
        with pytest.raises(NotFoundError) as raised:
            openai_client.chat.completions.create(
                model="huggingface-nothing-deployed-here",
                messages=[{"role": "user", "content": "Say hello."}],
                max_completion_tokens=_TRUNCATING_BUDGET,
            )

        envelope = raised.value.body
        assert isinstance(envelope, dict)
        assert envelope["type"] == "invalid_request_error"
        assert envelope["code"] == "model_not_found"

    def test_an_embedding_request_is_refused(
        self, openai_client: OpenAI, marketplace_model: str
    ) -> None:
        """The endpoint publishes only TEXT output, so an embedding request is refused.

        The refusal is decided from the catalogue entry, before any paid call.

        Ref: stdapi/models/__init__.py:validate_model
        """
        with pytest.raises(APIStatusError) as raised:
            openai_client.embeddings.create(model=marketplace_model, input="Hello.")

        assert raised.value.status_code in {400, 404}

    @pytest.mark.parametrize(
        ("route", "payload"),
        [
            ("/v1/responses/input_tokens", {"input": "Hi"}),
            (
                "/anthropic/v1/messages/count_tokens",
                {"messages": [{"role": "user", "content": "Hi"}]},
            ),
        ],
    )
    def test_token_counting_is_refused(
        self,
        local_test_client: TestClientType,
        marketplace_model: str,
        api_key: str,
        route: str,
        payload: dict[str, object],
    ) -> None:
        """Both counters answer 400 from the gateway, not a backend validation error.

        Bedrock's token counter takes a foundation model identifier, so an
        endpoint has no counting form at all; answering with the backend's own
        rejection would leak the reason.

        Ref: stdapi/models/__init__.py:reject_unsupported_token_counting
        """
        response = local_test_client.post(
            route,
            json={"model": marketplace_model, **payload},
            headers={"Authorization": f"Bearer {api_key}"},
        )

        assert response.status_code == 400, response.text
        assert "Token counting is not supported" in response.text
        assert "arn:aws" not in response.text

