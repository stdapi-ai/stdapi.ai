"""Catalogue entry and configuration for an Amazon SageMaker AI endpoint.

An endpoint is declared rather than discovered: SageMaker AI publishes neither
which container an endpoint runs -- and only some serve the OpenAI Chat
Completions API -- nor what the model behind it should be called. The entry
still has to land where a chat route can resolve it, which is the Bedrock
catalogue rather than ``EXTRA_MODELS``.

Ref: https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-openai-compatible.html
     stdapi/models/sagemaker_endpoints.py
     stdapi/config.py:SageMakerEndpointConfig
"""

from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

import stdapi.routes.anthropic_messages
import stdapi.routes.openai_chat_completions
import stdapi.routes.openai_responses  # noqa: F401
from stdapi.api_errors import ApiError
from stdapi.config import SETTINGS, SageMakerEndpointConfig, _Settings
from stdapi.models import (
    SAGEMAKER_ENDPOINT_MODELS,
    SAGEMAKER_SERVICE,
    ModelDetails,
    _compute_model_capabilities,
    is_sagemaker_endpoint,
    reject_unsupported_token_counting,
    usage_service,
)
from stdapi.models.capabilities import ROUTE_CAPABILITIES
from stdapi.models.chat import get_chat_model
from stdapi.models.chat._sagemaker import SageMakerChatModel
from stdapi.models.sagemaker_endpoints import (
    _model_from_endpoint,
    merge_sagemaker_endpoint_models,
)
from stdapi.pricing import Service
from tests._helpers import make_model_details

if TYPE_CHECKING:
    from collections.abc import Iterator

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Model ID the declared endpoint is published under.
MODEL_ID = "my-qwen3"

#: The operator's declaration for it.
ENDPOINT = SageMakerEndpointConfig(
    endpoint="stdapi-test-endpoint",
    region="us-east-1",
    inference_component="stdapi-test-component",
    name="Qwen3 1.7B",
    provider="Qwen",
)


@pytest.fixture(autouse=True)
def _isolate_endpoint_index() -> Iterator[None]:
    """Restore the process-wide endpoint index whatever a test does to it.

    Every merge in this module rewrites the live ``SAGEMAKER_ENDPOINT_MODELS``,
    including the ones that clear it, so the restore cannot sit in a fixture
    only some of them ask for -- nor at the end of a test body, which a failing
    assertion skips.
    """
    saved = dict(SAGEMAKER_ENDPOINT_MODELS)
    yield
    SAGEMAKER_ENDPOINT_MODELS.clear()
    SAGEMAKER_ENDPOINT_MODELS.update(saved)


@pytest.fixture
def declared(monkeypatch: pytest.MonkeyPatch) -> dict[str, ModelDetails]:
    """Publish one declared endpoint into the catalogue."""
    monkeypatch.setattr(SETTINGS, "aws_sagemaker_endpoints", {MODEL_ID: ENDPOINT})
    catalogue: dict[str, ModelDetails] = {}
    merge_sagemaker_endpoint_models(catalogue, {})
    return catalogue


class TestCatalogueEntry:
    """A declared endpoint publishes as an ordinary chat model.

    Ref: stdapi/models/sagemaker_endpoints.py:_model_from_endpoint
    """

    def test_declaration_maps_to_a_catalogue_entry(self) -> None:
        """Every published field comes from the declaration or from the transport."""
        model = _model_from_endpoint(MODEL_ID, ENDPOINT)

        assert model.id == MODEL_ID
        assert model.name == "Qwen3 1.7B"
        assert model.provider == "Qwen"
        assert model.service == SAGEMAKER_SERVICE
        assert model.input_modalities == ["TEXT"]
        assert model.output_modalities == ["TEXT"]
        assert model.regions == ["us-east-1"]
        assert model.response_streaming is True
        assert model.batch is False

    def test_the_endpoint_never_reaches_the_public_response(self) -> None:
        """The declaration names the operator's own infrastructure.

        The catalogue is public output, and the endpoint and component names
        are backend identifiers a client can neither use nor act on.
        """
        model = _model_from_endpoint(MODEL_ID, ENDPOINT)

        dumped = model.model_dump()

        assert model.sagemaker_endpoint == ENDPOINT
        assert "sagemaker_endpoint" not in dumped
        assert ENDPOINT.endpoint not in str(dumped)

    def test_the_model_id_defaults_the_display_name(self) -> None:
        """An operator who names nothing still gets a usable listing."""
        model = _model_from_endpoint(
            MODEL_ID, SageMakerEndpointConfig(endpoint="e", region="us-east-1")
        )

        assert model.name == MODEL_ID
        assert model.provider == "Amazon SageMaker AI"
        assert model.sagemaker_endpoint is not None
        assert model.sagemaker_endpoint.inference_component == ""

    def test_declared_image_input_is_advertised(self) -> None:
        """A container serving vision content parts can say so."""
        model = _model_from_endpoint(
            MODEL_ID,
            SageMakerEndpointConfig(
                endpoint="e", region="us-east-1", input_modalities=["text", "image"]
            ),
        )

        assert model.input_modalities == ["TEXT", "IMAGE"]


class TestMerge:
    """Declared endpoints join the catalogue every text route resolves against.

    ``validate_model`` defaults to ``bedrock_only=True`` and reads ``_MODELS``
    alone, so an entry only in ``EXTRA_MODELS`` would be listed by
    ``/v1/models`` and then answer 404.

    Ref: stdapi/models/sagemaker_endpoints.py:merge_sagemaker_endpoint_models
    """

    def test_merge_publishes_and_indexes_the_endpoint(
        self, declared: dict[str, ModelDetails]
    ) -> None:
        """The entry lands in the resolved catalogue and in the endpoint index."""
        assert declared[MODEL_ID].service == SAGEMAKER_SERVICE
        assert SAGEMAKER_ENDPOINT_MODELS[MODEL_ID] is declared[MODEL_ID]
        assert is_sagemaker_endpoint(MODEL_ID)
        assert not is_sagemaker_endpoint("amazon.nova-lite-v1:0")

    def test_a_collision_keeps_the_existing_model_and_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Naming an endpoint after a catalogue model does not shadow it.

        Replacing a serverless model, available in every served Region and free
        at rest, with one endpoint in one Region would be a silent downgrade.
        """
        monkeypatch.setattr(SETTINGS, "aws_sagemaker_endpoints", {MODEL_ID: ENDPOINT})
        existing = make_model_details(MODEL_ID, name="serverless")
        catalogue = {MODEL_ID: existing}
        reported: dict[str, str] = {}

        merge_sagemaker_endpoint_models(catalogue, reported)

        assert catalogue[MODEL_ID] is existing
        assert not SAGEMAKER_ENDPOINT_MODELS
        assert MODEL_ID in next(iter(reported.values()))

    def test_every_collision_of_one_region_is_reported(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two ignored declarations produce two warnings, not one.

        Keyed by region, the second message would overwrite the first: the
        operator fixes the one they were told about and the other endpoint
        stays invisible with no further signal.
        """
        other_id = "my-llama"
        monkeypatch.setattr(
            SETTINGS,
            "aws_sagemaker_endpoints",
            {MODEL_ID: ENDPOINT, other_id: ENDPOINT},
        )
        catalogue = {
            MODEL_ID: make_model_details(MODEL_ID),
            other_id: make_model_details(other_id),
        }
        reported: dict[str, str] = {}

        merge_sagemaker_endpoint_models(catalogue, reported)

        messages = "".join(reported.values())
        assert len(reported) == 2
        assert MODEL_ID in messages
        assert other_id in messages


class TestCapabilities:
    """A model endpoint serves the chat routes and nothing capability-gated.

    Ref: stdapi/models/__init__.py:_compute_model_capabilities
    """

    def test_chat_routes_are_advertised(
        self, declared: dict[str, ModelDetails]
    ) -> None:
        """The three chat dialects are all served through the one transport."""
        routes, _ = _compute_model_capabilities(MODEL_ID, declared[MODEL_ID])

        for operation in (
            "openai_chat_completion",
            "openai_response",
            "anthropic_message",
        ):
            assert ROUTE_CAPABILITIES[operation].path in routes

    def test_token_counting_is_not_advertised(
        self, declared: dict[str, ModelDetails]
    ) -> None:
        """Bedrock's token counter takes a foundation model, which this is not."""
        routes, tools = _compute_model_capabilities(MODEL_ID, declared[MODEL_ID])

        assert ROUTE_CAPABILITIES["anthropic_message_count_tokens"].path not in routes
        assert "anthropic_message_count_tokens" not in tools
        assert "openai_response_input_tokens" not in tools

    def test_token_counting_is_refused_at_request_time(
        self, declared: dict[str, ModelDetails]
    ) -> None:
        """The gateway answers it itself rather than forwarding a backend error."""
        with pytest.raises(ApiError) as exc_info:
            reject_unsupported_token_counting(declared[MODEL_ID])

        assert exc_info.value.status == 400
        assert "Token counting is not supported" in str(exc_info.value)


class TestDispatch:
    """A declared endpoint resolves to its own chat model, never a family.

    Ref: stdapi/models/chat/__init__.py:get_chat_model
    """

    def test_the_endpoint_resolves_to_the_sagemaker_chat_model(
        self, declared: dict[str, ModelDetails]
    ) -> None:
        """The model ID is the operator's, so no family matcher may claim it."""
        del declared

        model = get_chat_model(MODEL_ID)

        assert isinstance(model, SageMakerChatModel)
        assert get_chat_model(MODEL_ID) is model

    def test_the_endpoint_declaration_reaches_the_model(
        self, declared: dict[str, ModelDetails]
    ) -> None:
        """Invocation reads the endpoint back out of the catalogue entry."""
        del declared

        model = get_chat_model(MODEL_ID)

        assert isinstance(model, SageMakerChatModel)
        assert model._endpoint() == ENDPOINT  # noqa: SLF001


class TestUsageService:
    """Tokens are recorded, and no cost is invented for them.

    AWS bills a real-time endpoint by the instance-hour of the instances it
    runs on, so no per-token rate exists to resolve.

    Ref: stdapi/models/__init__.py:usage_service
         stdapi/usage.py:UNPRICED_SERVICES
    """

    def test_usage_is_recorded_under_the_sagemaker_service(
        self, declared: dict[str, ModelDetails]
    ) -> None:
        """The service exists to key quantities, not to price them."""
        del declared
        # Imported here: only this test reads the unpriced-service set.
        from stdapi.usage import UNPRICED_SERVICES  # noqa: PLC0415

        assert usage_service(MODEL_ID) == Service.SAGEMAKER
        assert Service.SAGEMAKER in UNPRICED_SERVICES

    def test_an_ordinary_model_is_unaffected(self) -> None:
        """Every other model still bills under bedrock-runtime."""
        assert usage_service("amazon.nova-lite-v1:0") == Service.BEDROCK


class TestConfiguration:
    """The settings a misconfiguration must be refused at startup.

    Ref: stdapi/config.py:_Settings._validate_sagemaker_warmup_timeout
         stdapi/config.py:_Settings._validate_sagemaker_endpoint_url
    """

    def test_a_warmup_longer_than_the_response_budget_is_refused(self) -> None:
        """A wait cannot outlast the connection it is allowed to hold."""
        with pytest.raises(ValidationError) as exc_info:
            _Settings(aws_sagemaker_warmup_timeout=900, ai_response_timeout=600)

        assert "exceeds ai_response_timeout" in str(exc_info.value)

    def test_the_default_warmup_fits_the_default_response_budget(self) -> None:
        """The shipped defaults agree, so a deployment starts unconfigured.

        Read off the field declarations rather than the live ``SETTINGS``: the
        environment driving the live SageMaker lane configures both values, and
        that is not a defect in what ships.
        """
        warmup = _Settings.model_fields["aws_sagemaker_warmup_timeout"].default
        response = _Settings.model_fields["ai_response_timeout"].default

        assert warmup == 600
        assert warmup <= response
        # And the shipped pair passes the cross-field validator.
        assert _Settings(
            aws_sagemaker_warmup_timeout=warmup, ai_response_timeout=response
        )

    def test_disabling_the_warmup_is_accepted(self) -> None:
        """``0`` is a supported value, not an out-of-range one."""
        assert (
            _Settings(aws_sagemaker_warmup_timeout=0).aws_sagemaker_warmup_timeout == 0
        )

    def test_a_negative_warmup_is_refused(self) -> None:
        """A negative budget has no meaning and would silently fail fast."""
        with pytest.raises(ValidationError):
            _Settings(aws_sagemaker_warmup_timeout=-1)

    @pytest.mark.parametrize(
        "url", ["http://runtime.{region}.example", "https://runtime.{typo}.example"]
    )
    def test_a_bad_endpoint_url_override_is_refused(self, url: str) -> None:
        """Plain HTTP and a broken placeholder both fail at startup, not per request."""
        with pytest.raises(ValidationError):
            _Settings(aws_sagemaker_endpoint_url=url)

    def test_an_unknown_declaration_field_is_refused(self) -> None:
        """A typo in a declaration is refused rather than silently ignored."""
        with pytest.raises(ValidationError):
            SageMakerEndpointConfig(
                endpoint="e",
                region="us-east-1",
                inference_components="typo",  # type: ignore[call-arg]
            )
