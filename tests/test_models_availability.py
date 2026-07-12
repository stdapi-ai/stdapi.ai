"""Unit tests for stdapi.models._filter_model's availability-warning logic.

Regression: AWS's foundation-model-availability API can report
``regionAvailability != AVAILABLE`` for a model that ``list_foundation_models``
just advertised in that same region (seen live, e.g.
``amazon.titan-embed-g1-text-02``). That specific inconsistency isn't
operator-actionable, so it's skipped silently instead of surfacing a
false-positive warning; any other issue (alone or combined with it) is still
recorded.
"""

from typing import TYPE_CHECKING, Any

from stdapi.models import ModelDetails, _filter_model

if TYPE_CHECKING:
    from types_aiobotocore_bedrock.literals import RegionName


def _make_model(
    model_id: str = "vendor.some-model-v1", region: RegionName = "us-east-1"
) -> ModelDetails:
    """Build a minimal ModelDetails instance with a single region."""
    return ModelDetails(
        id=model_id,
        name=model_id,
        provider="Vendor",
        input_modalities=["TEXT"],
        output_modalities=["TEXT"],
        regions=[region],
    )


class _StubBedrockClient:
    """Stub Bedrock control-plane client returning a fixed availability payload."""

    def __init__(self, availability: dict[str, Any]) -> None:
        self._availability = availability

    async def get_foundation_model_availability(
        self,
        modelId: str,  # noqa: N803
    ) -> dict[str, Any]:
        """Return the fixed availability payload regardless of the requested model."""
        return self._availability


def _availability(
    *,
    authorization: str = "AUTHORIZED",
    entitlement: str = "AVAILABLE",
    region: str = "AVAILABLE",
    agreement: str = "AVAILABLE",
) -> dict[str, Any]:
    """Build an availability payload in the shape returned by get_foundation_model_availability."""
    return {
        "authorizationStatus": authorization,
        "entitlementAvailability": entitlement,
        "regionAvailability": region,
        "agreementAvailability": {"status": agreement},
    }


class TestFilterModelAvailabilityWarnings:
    """Whether an unavailable model surfaces an entry in unavailable_models."""

    async def test_region_unavailable_alone_is_skipped_silently(self) -> None:
        """Issues == ["unavailable"] alone: skipped, no models/unavailable_models entry."""
        model = _make_model()
        client = _StubBedrockClient(_availability(region="UNAVAILABLE"))
        models: dict[str, ModelDetails] = {}
        unavailable_models: dict[str, dict[str, list[str]]] = {}

        await _filter_model(client, model, models, unavailable_models)  # type: ignore[arg-type]

        assert models == {}
        assert unavailable_models == {}

    async def test_unauthorized_alone_is_recorded(self) -> None:
        """Issues == ["unauthorized"] alone: recorded in unavailable_models."""
        model = _make_model()
        client = _StubBedrockClient(_availability(authorization="DENIED"))
        models: dict[str, ModelDetails] = {}
        unavailable_models: dict[str, dict[str, list[str]]] = {}

        await _filter_model(client, model, models, unavailable_models)  # type: ignore[arg-type]

        assert models == {}
        assert unavailable_models == {model.id: {"us-east-1": ["unauthorized"]}}

    async def test_unauthorized_and_unavailable_together_are_recorded(self) -> None:
        """Issues == ["unauthorized", "unavailable"]: recorded (not silently skipped)."""
        model = _make_model()
        client = _StubBedrockClient(
            _availability(authorization="DENIED", region="UNAVAILABLE")
        )
        models: dict[str, ModelDetails] = {}
        unavailable_models: dict[str, dict[str, list[str]]] = {}

        await _filter_model(client, model, models, unavailable_models)  # type: ignore[arg-type]

        assert models == {}
        assert unavailable_models == {
            model.id: {"us-east-1": ["unauthorized", "unavailable"]}
        }

    async def test_fully_available_model_is_added_to_models(self) -> None:
        """A model with no issues at all is added to `models`, not `unavailable_models`."""
        model = _make_model()
        client = _StubBedrockClient(_availability())
        models: dict[str, ModelDetails] = {}
        unavailable_models: dict[str, dict[str, list[str]]] = {}

        await _filter_model(client, model, models, unavailable_models)  # type: ignore[arg-type]

        assert models == {model.id: model}
        assert unavailable_models == {}
