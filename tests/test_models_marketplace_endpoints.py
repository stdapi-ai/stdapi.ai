"""Discovery and catalogue mapping for Amazon Bedrock Marketplace model endpoints.

An endpoint the operator deployed is discovered through
``ListMarketplaceModelEndpoints`` plus a ``GetMarketplaceModelEndpoint`` per
entry -- the list summary carries neither the SageMaker status nor the endpoint
configuration -- and is published as an ordinary chat model whose ``modelId`` is
the endpoint ARN. Readiness needs both statuses: the Bedrock registration
(``REGISTERED`` / ``INCOMPATIBLE_ENDPOINT``, an enum) and the SageMaker endpoint
status (an unconstrained string).

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-marketplace-call-the-endpoint.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/setup-amazon-bedrock-marketplace.html
     stdapi/models/marketplace_endpoints.py
     stdapi/models/__init__.py:ModelDetails.get_id
"""

from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError

import stdapi.models
import stdapi.routes.anthropic_messages

# Imported for its side effect: registering the token counter under test.
import stdapi.routes.openai_responses
from stdapi import region_routing
from stdapi.api_errors import ApiError
from stdapi.aws import _CLIENTS
from stdapi.config import SETTINGS
from stdapi.models import (
    _MODELS,
    _TOKEN_COUNTING_OPERATIONS,
    MARKETPLACE_ENDPOINT_MODELS,
    MARKETPLACE_SERVICE,
    RUNTIME_SERVICE,
    ModelDetails,
    ModelRegionUnavailableError,
    _compute_model_capabilities,
    _marketplace_endpoint_from_arn,
    compute_candidate_regions,
    get_model_details,
    is_marketplace_endpoint,
    resolve_routed_model_id,
    usage_service,
)
from stdapi.models.capabilities import ROUTE_CAPABILITIES
from stdapi.models.chat import get_chat_model
from stdapi.models.chat._default import ChatModel
from stdapi.models.marketplace_endpoints import (
    _model_from_endpoint,
    collect_marketplace_endpoint_models,
    marketplace_endpoint_regions,
    merge_marketplace_endpoint_models,
)
from stdapi.pricing import Service
from tests._helpers import make_model_details

if TYPE_CHECKING:
    from collections.abc import Iterator

    from starlette.testclient import TestClient
    from types_aiobotocore_bedrock.type_defs import MarketplaceModelEndpointTypeDef

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: A well-formed endpoint ARN, in the shape bedrock-runtime takes as a model ID.
ENDPOINT_ARN = "arn:aws:sagemaker:eu-west-1:123456789012:endpoint/qwen3-4b"

#: The public-hub listing ARN an endpoint was deployed from.
SOURCE_ARN = (
    "arn:aws:sagemaker:eu-west-1:aws:hub-content/SageMakerPublicHub/Model/"
    "huggingface-reasoning-qwen3-4b/1.41.0"
)

#: The model ID the listing ARN yields, which is what clients ask for.
MODEL_ID = "huggingface-reasoning-qwen3-4b"


def _endpoint(**overrides: Any) -> MarketplaceModelEndpointTypeDef:  # noqa: ANN401
    """Build a GetMarketplaceModelEndpoint payload for a servable endpoint."""
    payload = {
        "endpointArn": ENDPOINT_ARN,
        "modelSourceIdentifier": SOURCE_ARN,
        "status": "REGISTERED",
        "endpointStatus": "InService",
    }
    payload.update(overrides)
    return payload  # type: ignore[return-value]


class _FakeBedrockClient:
    """A ``bedrock`` client answering the two discovery calls from a fixed list."""

    def __init__(
        self, endpoints: list[MarketplaceModelEndpointTypeDef], *, pages: int = 1
    ) -> None:
        self.endpoints = {e["endpointArn"]: e for e in endpoints}
        self.pages = pages
        self.list_calls = 0

    async def list_marketplace_model_endpoints(
        self,
        nextToken: str = "",  # noqa: N803
    ) -> dict[str, Any]:
        """Answer one page of endpoint summaries."""
        self.list_calls += 1
        arns = list(self.endpoints)
        if self.pages == 1:
            return {"marketplaceModelEndpoints": [{"endpointArn": a} for a in arns]}
        if not nextToken:
            return {
                "marketplaceModelEndpoints": [{"endpointArn": arns[0]}],
                "nextToken": "more",
            }
        return {"marketplaceModelEndpoints": [{"endpointArn": a} for a in arns[1:]]}

    async def get_marketplace_model_endpoint(
        self,
        endpointArn: str,  # noqa: N803
    ) -> dict[str, Any]:
        """Answer the full endpoint record for one ARN."""
        return {"marketplaceModelEndpoint": self.endpoints[endpointArn]}


def _patch_client(monkeypatch: pytest.MonkeyPatch, client: object) -> None:
    """Serve *client* for every ``get_client`` call the discovery makes."""
    monkeypatch.setattr(
        "stdapi.models.marketplace_endpoints.get_client",
        lambda service, region=None: client,  # noqa: ARG005
    )


def _published(
    endpoint: MarketplaceModelEndpointTypeDef, region: str = "eu-west-1"
) -> ModelDetails:
    """Answer the catalogue entry for an endpoint that must be publishable."""
    model, reason = _model_from_endpoint(endpoint, region)  # type: ignore[arg-type]
    assert model is not None, reason
    return model


@pytest.fixture(autouse=True)
def _isolate_endpoint_index() -> Iterator[None]:
    """Restore the process-wide endpoint index whatever a test does to it.

    ``merge_marketplace_endpoint_models`` rewrites the live
    ``MARKETPLACE_ENDPOINT_MODELS``, so the restore cannot sit at the end of a
    test body: a failing assertion above it would leave the global polluted and
    report one regression as two, the second in an unrelated module.
    """
    saved = dict(MARKETPLACE_ENDPOINT_MODELS)
    yield
    MARKETPLACE_ENDPOINT_MODELS.clear()
    MARKETPLACE_ENDPOINT_MODELS.update(saved)


def test_endpoint_maps_to_a_catalogue_entry() -> None:
    """A registered, in-service endpoint publishes under its listing name.

    The endpoint ARN never becomes the public model ID: it carries the account
    ID, so it is held on an excluded field and only surfaces as the ``modelId``
    sent to bedrock-runtime.

    Ref: stdapi/models/marketplace_endpoints.py:_model_from_endpoint
    """
    model = _published(_endpoint())

    assert model.id == MODEL_ID
    assert model.provider == "Hugging Face"
    assert model.service == MARKETPLACE_SERVICE
    assert model.input_modalities == ["TEXT"]
    assert model.output_modalities == ["TEXT"]
    assert model.regions == ["eu-west-1"]
    assert model.batch is False
    assert model.response_streaming is True
    assert model.get_id("eu-west-1", inference_profile=True) == ENDPOINT_ARN
    assert model.get_id() == ENDPOINT_ARN
    assert "marketplace_endpoints" not in model.model_dump()


@pytest.mark.parametrize(
    ("field", "value", "reported"),
    [
        ("status", "INCOMPATIBLE_ENDPOINT", True),
        ("endpointStatus", "Creating", False),
        ("endpointStatus", "Deleting", False),
        ("endpointStatus", "Failed", True),
        ("endpointStatus", "OutOfService", True),
    ],
)
def test_endpoint_that_cannot_serve_is_not_published(
    field: str, value: str, reported: bool
) -> None:
    """Both statuses gate publication, and neither one alone is enough.

    ``status`` is the Amazon Bedrock registration and ``endpointStatus`` the
    SageMaker one; an endpoint failing either cannot answer an invocation. A
    decline the operator can act on says so; one the endpoint resolves by
    itself, a few minutes later, would be noise.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_MarketplaceModelEndpoint.html
    """
    model, reason = _model_from_endpoint(_endpoint(**{field: value}), "eu-west-1")

    assert model is None
    assert bool(reason) is reported
    if reported:
        assert value in reason


@pytest.mark.parametrize("value", ["Updating", "SystemUpdating"])
def test_an_endpoint_being_updated_keeps_serving_and_stays_published(
    value: str,
) -> None:
    """A scale or a config change must not unpublish a model that answers.

    ``UpdateEndpoint`` shifts traffic onto the new instances and documents that
    there is no availability loss, so the endpoint answers for the whole
    ``Updating`` window -- more than ten minutes for an instance change, which
    a catalogue refresh is very likely to fall inside. Dropping it would answer
    ``404 model_not_found`` for a model SageMaker is serving normally, until a
    later refresh finds it ``InService`` again.

    Ref: stdapi/models/marketplace_endpoints.py:_model_from_endpoint
         https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_UpdateEndpoint.html
    """
    model, reason = _model_from_endpoint(_endpoint(endpointStatus=value), "eu-west-1")

    assert model is not None
    assert model.id == MODEL_ID
    assert not reason


def test_an_absent_registration_status_is_reported() -> None:
    """``status`` is optional in the response, and its absence drops the endpoint.

    Silently, it would drop every endpoint in the account with no diagnostic.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_MarketplaceModelEndpoint.html
    """
    payload = _endpoint()
    del payload["status"]

    model, reason = _model_from_endpoint(payload, "eu-west-1")

    assert model is None
    assert "not registered with Amazon Bedrock" in reason


def test_endpoint_status_is_matched_as_an_open_string() -> None:
    """``endpointStatus`` is not an enum, so it is compared case-insensitively.

    Amazon Bedrock republishes SageMaker's own status as an unconstrained
    string; treating it as a fixed set would drop a servable endpoint.

    Ref: stdapi/models/marketplace_endpoints.py:_model_from_endpoint
    """
    assert _published(_endpoint(endpointStatus="INSERVICE"))
    assert _published(_endpoint(endpointStatus="inservice"))


def test_endpoint_outside_its_own_region_is_not_published() -> None:
    """An endpoint ARN naming another region is dropped rather than routed to.

    There is no cross-region form for a model endpoint, so calling it anywhere
    but its own region is a guaranteed ``ValidationException``.

    Ref: stdapi/models/marketplace_endpoints.py:_model_from_endpoint
    """
    model, reason = _model_from_endpoint(_endpoint(), "us-east-1")

    assert model is None
    assert "us-east-1" in reason


def test_non_public_hub_source_is_not_published() -> None:
    """An endpoint whose source is not a public-hub listing yields no model name.

    ``CreateMarketplaceModelEndpoint`` constrains the source to public-hub
    content, but ``RegisterMarketplaceModelEndpoint`` can attach an arbitrary
    one; nothing can be published without a listing name -- and an operator who
    attached one gets told why their endpoint never appeared.

    Ref: stdapi/utils.py:match_sagemaker_hub_content_arn
    """
    source = "arn:aws:sagemaker:eu-west-1:aws:hub/X"

    model, reason = _model_from_endpoint(
        _endpoint(modelSourceIdentifier=source), "eu-west-1"
    )

    assert model is None
    assert source in reason
    assert "AWS_BEDROCK_ALLOW_MARKETPLACE_ENDPOINT_ARN=true" in reason


async def test_discovery_pages_and_merges_regions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery follows ``nextToken`` and merges one listing across two regions.

    Ref: stdapi/models/marketplace_endpoints.py:collect_marketplace_endpoint_models
    """
    other_arn = "arn:aws:sagemaker:us-east-1:123456789012:endpoint/qwen3-4b"
    other_source = SOURCE_ARN.replace("eu-west-1", "us-east-1")
    clients = {
        "eu-west-1": _FakeBedrockClient(
            [
                _endpoint(),
                _endpoint(
                    endpointArn=ENDPOINT_ARN.replace("qwen3-4b", "other"),
                    endpointStatus="Creating",
                ),
            ],
            pages=2,
        ),
        "us-east-1": _FakeBedrockClient(
            [_endpoint(endpointArn=other_arn, modelSourceIdentifier=other_source)]
        ),
    }
    monkeypatch.setattr(
        "stdapi.models.marketplace_endpoints.get_client",
        lambda service, region=None: clients[region],  # noqa: ARG005
    )
    monkeypatch.setattr(SETTINGS, "aws_bedrock_marketplace_endpoints_enabled", True)
    monkeypatch.setattr(
        SETTINGS, "aws_bedrock_marketplace_endpoint_regions", ["eu-west-1", "us-east-1"]
    )
    failed: dict[str, str] = {}

    models = await collect_marketplace_endpoint_models(failed)

    assert clients["eu-west-1"].list_calls == 2
    assert not failed
    assert set(models) == {MODEL_ID}
    assert models[MODEL_ID].regions == ["eu-west-1", "us-east-1"]
    assert models[MODEL_ID].get_id("us-east-1", inference_profile=True) == other_arn


async def test_two_endpoints_for_one_listing_in_one_region_report_the_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the first endpoint of a colliding pair is published, and it is reported.

    The report is keyed by the colliding listing: two listings each duplicated
    in one region would otherwise share a key and the operator would be told
    about the last one only.

    Ref: stdapi/models/marketplace_endpoints.py:collect_marketplace_endpoint_models
    """
    other_source = SOURCE_ARN.replace("huggingface-reasoning-qwen3-4b", "meta-llama-3")
    _patch_client(
        monkeypatch,
        _FakeBedrockClient(
            [
                _endpoint(),
                _endpoint(endpointArn=ENDPOINT_ARN.replace("qwen3-4b", "qwen3-4b-b")),
                _endpoint(
                    endpointArn=ENDPOINT_ARN.replace("qwen3-4b", "llama-a"),
                    modelSourceIdentifier=other_source,
                ),
                _endpoint(
                    endpointArn=ENDPOINT_ARN.replace("qwen3-4b", "llama-b"),
                    modelSourceIdentifier=other_source,
                ),
            ]
        ),
    )
    monkeypatch.setattr(SETTINGS, "aws_bedrock_marketplace_endpoints_enabled", True)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_marketplace_endpoint_regions", [])
    monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["eu-west-1"])
    failed: dict[str, str] = {}

    models = await collect_marketplace_endpoint_models(failed)

    assert set(models) == {MODEL_ID, "meta-llama-3"}
    assert models[MODEL_ID].get_id("eu-west-1", inference_profile=True) == ENDPOINT_ARN
    assert set(failed) == {
        f"{MODEL_ID} (Marketplace endpoints in eu-west-1)",
        "meta-llama-3 (Marketplace endpoints in eu-west-1)",
    }
    assert all("Several endpoints serve" in detail for detail in failed.values())


async def test_denied_discovery_names_the_permission_and_keeps_the_catalogue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused region degrades to a warning naming what the role is missing.

    Discovery runs inside the model refresh, so raising would empty the whole
    catalogue for a permission only this optional feature needs. A refusal AWS
    worded itself, without naming an action, cannot be told from a service
    refusing the account outright, so it stays with the unreachable regions and
    names both permissions in prose instead.

    Ref: stdapi/models/marketplace_endpoints.py:collect_marketplace_endpoint_models
    """

    class _Denied:
        async def list_marketplace_model_endpoints(self, **_: object) -> dict[str, Any]:
            raise ClientError(
                {"Error": {"Code": "AccessDeniedException", "Message": "no"}},
                "ListMarketplaceModelEndpoints",
            )

    _patch_client(monkeypatch, _Denied())
    monkeypatch.setattr(SETTINGS, "aws_bedrock_marketplace_endpoints_enabled", True)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_marketplace_endpoint_regions", [])
    monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["eu-west-1"])
    failed: dict[str, str] = {}
    denied: dict[str, str] = {}

    assert await collect_marketplace_endpoint_models(failed, denied) == {}
    assert not denied
    detail = failed["eu-west-1 (Marketplace endpoints)"]
    assert "bedrock:ListMarketplaceModelEndpoints" in detail
    assert "bedrock:GetMarketplaceModelEndpoint" in detail


async def test_an_iam_denial_is_never_reported_as_an_unreachable_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A region IAM refused is filed as a policy gap, like the core sweep's.

    The startup warning splits the two on purpose: an unreachable region reads
    as a regional outage worth retrying, a denied one as the one-line policy
    change it is. Filing this collector's denials with the failures also put
    the raw ``ClientError`` text -- which carries the principal ARN AWS named
    -- into the structured warning.

    Ref: stdapi/models/__init__.py:_collect_region_candidates
         stdapi/api_errors.py:iam_denial_detail
         docs/operations_troubleshooting.md
    """

    class _Denied:
        async def list_marketplace_model_endpoints(self, **_: object) -> dict[str, Any]:
            raise ClientError(
                {
                    "Error": {
                        "Code": "AccessDeniedException",
                        "Message": (
                            "User: arn:aws:sts::123456789012:assumed-role/"
                            "stdapi/task is not authorized to perform: "
                            "bedrock:ListMarketplaceModelEndpoints"
                        ),
                    }
                },
                "ListMarketplaceModelEndpoints",
            )

    _patch_client(monkeypatch, _Denied())
    monkeypatch.setattr(SETTINGS, "aws_bedrock_marketplace_endpoints_enabled", True)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_marketplace_endpoint_regions", [])
    monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["eu-west-1"])
    failed: dict[str, str] = {}
    denied: dict[str, str] = {}

    assert await collect_marketplace_endpoint_models(failed, denied) == {}
    assert not failed
    detail = denied["eu-west-1 (Marketplace endpoints)"]
    assert "bedrock:ListMarketplaceModelEndpoints" in detail
    assert "assumed-role" not in detail


async def test_a_declined_endpoint_reaches_the_operator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A discovery that publishes nothing says why, rather than looking empty.

    Every other way discovery can come up short records a reason; an endpoint
    silently skipped leaves the operator a clean startup log and no model.

    Ref: stdapi/models/marketplace_endpoints.py:_endpoints_in_region
    """
    _patch_client(
        monkeypatch, _FakeBedrockClient([_endpoint(endpointStatus="OutOfService")])
    )
    monkeypatch.setattr(SETTINGS, "aws_bedrock_marketplace_endpoints_enabled", True)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_marketplace_endpoint_regions", [])
    monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["eu-west-1"])
    failed: dict[str, str] = {}

    assert await collect_marketplace_endpoint_models(failed) == {}
    assert "OutOfService" in failed["eu-west-1 (Marketplace endpoints)"]


async def test_disabled_feature_searches_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The feature is opt-in: disabled, it makes no AWS call at all.

    The client raises on every call it is given, so a discovery that stopped
    consulting ``marketplace_endpoint_regions`` -- and iterated the configured
    Bedrock regions instead, which is the same list when none is set -- fails
    here instead of asking for a permission an opt-out feature must not need.

    Ref: stdapi/config.py:Settings.aws_bedrock_marketplace_endpoints_enabled
    """

    class _Forbidden:
        def __getattr__(self, name: str) -> object:
            msg = f"the disabled feature called {name}"
            raise AssertionError(msg)

    _patch_client(monkeypatch, _Forbidden())
    monkeypatch.setattr(SETTINGS, "aws_bedrock_marketplace_endpoints_enabled", False)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["eu-west-1"])
    failed: dict[str, str] = {}

    assert marketplace_endpoint_regions() == []
    assert await collect_marketplace_endpoint_models(failed) == {}
    assert not failed


async def test_an_unbuilt_client_pool_degrades_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``bedrock`` pool at all declines the region rather than raising.

    ``get_client`` indexes the pool directly, so a pool that was never built
    raises a bare ``KeyError`` -- which is neither a ``BotoCoreError`` nor a
    ``ClientError``, so the caller's fault-isolation handler re-raises it and an
    optional feature takes the whole catalogue refresh down. The real
    ``get_client`` is used here, unpatched, because the patched one is exactly
    what hid this.

    Ref: stdapi/aws.py:get_client
         stdapi/models/marketplace_endpoints.py:_no_client_reason
    """
    # Removed rather than asserted absent: a live test sharing the process
    # builds the pool, and the condition under test is a missing service
    # entry, which is what an unbuilt pool means to ``get_client``.
    monkeypatch.delitem(_CLIENTS, "bedrock", raising=False)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_marketplace_endpoints_enabled", True)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_marketplace_endpoint_regions", [])
    monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["eu-west-1"])
    failed: dict[str, str] = {}

    assert await collect_marketplace_endpoint_models(failed) == {}
    detail = failed["eu-west-1 (Marketplace endpoints)"]
    assert "control-plane clients are not available" in detail
    assert "AWS_BEDROCK_MARKETPLACE_ENDPOINTS_ENABLED=false" in detail


async def test_a_region_absent_from_a_built_pool_reads_differently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A populated pool missing one region names that region, not a start-up fault.

    The operator fixes the two differently -- one is a deployment that never
    started, the other a region to add or drop -- so reporting them with one
    message would send them looking in the wrong place.

    Ref: stdapi/models/marketplace_endpoints.py:_no_client_reason
    """
    # Two entries, so get_client's deliberate single-client fallback (which
    # answers from the only pooled region) cannot mask the missing one.
    monkeypatch.setitem(
        _CLIENTS, "bedrock", {"us-east-1": object(), "eu-west-1": object()}
    )
    monkeypatch.setattr(SETTINGS, "aws_bedrock_marketplace_endpoints_enabled", True)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_marketplace_endpoint_regions", [])
    monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["ap-south-1"])
    failed: dict[str, str] = {}

    assert await collect_marketplace_endpoint_models(failed) == {}
    detail = failed["ap-south-1 (Marketplace endpoints)"]
    assert "No Amazon Bedrock control-plane client is pooled for ap-south-1" in detail
    assert "eu-west-1, us-east-1" in detail
    assert "AWS_BEDROCK_MARKETPLACE_ENDPOINT_REGIONS" in detail


async def test_a_key_error_from_the_discovery_calls_still_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The client lookup is guarded; the calls it feeds are not.

    A ``KeyError`` from the discovery itself is a defect in this module, and
    swallowing it would turn the fix for the pool bug into a way to hide the
    next one.

    Ref: stdapi/models/marketplace_endpoints.py:_endpoints_in_region
    """

    class _Broken:
        async def list_marketplace_model_endpoints(self, **_: object) -> dict[str, Any]:
            missing = "marketplaceModelEndpoints"
            raise KeyError(missing)

    _patch_client(monkeypatch, _Broken())
    monkeypatch.setattr(SETTINGS, "aws_bedrock_marketplace_endpoints_enabled", True)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_marketplace_endpoint_regions", [])
    monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["eu-west-1"])

    with pytest.raises(KeyError, match="marketplaceModelEndpoints"):
        await collect_marketplace_endpoint_models({})


async def test_the_catalogue_still_builds_with_discovery_on_and_no_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The whole refresh survives Marketplace discovery finding no client.

    This is the condition nobody had run: the offline suite with the feature
    switched on. Discovery runs inside ``_collect_all_models``, so the invariant
    under test is the one the fault-isolation tests assert for every other
    backend -- a broken optional feature costs its own models and a warning, not
    the catalogue.

    Ref: stdapi/models/__init__.py:_collect_all_models
    """
    # Removed rather than asserted absent: a live test sharing the process
    # builds the pool, and the condition under test is a missing service
    # entry, which is what an unbuilt pool means to ``get_client``.
    monkeypatch.delitem(_CLIENTS, "bedrock", raising=False)
    monkeypatch.setattr(region_routing, "ORDERED_BEDROCK_REGIONS", ["eu-west-1"])

    async def _fetch(_region: str) -> list[ModelDetails]:
        return [make_model_details("vendor.some-model-v1", regions=["eu-west-1"])]

    async def _available(_model: ModelDetails) -> list[str]:
        return []

    monkeypatch.setattr(stdapi.models, "_get_bedrock_models_from_region", _fetch)
    monkeypatch.setattr(stdapi.models, "_check_model_availability", _available)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_marketplace_endpoints_enabled", True)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_marketplace_endpoint_regions", [])
    monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["eu-west-1"])
    failed: dict[str, str] = {}

    all_models, _ = await stdapi.models._collect_all_models(failed, {}, {})  # noqa: SLF001

    assert "vendor.some-model-v1" in all_models, (
        "Marketplace discovery must not cost the catalogue its other models"
    )
    assert (
        "control-plane clients are not available"
        in (failed["eu-west-1 (Marketplace endpoints)"])
    )


def test_serverless_model_of_the_same_name_keeps_priority() -> None:
    """A published foundation model is never shadowed by an endpoint of that name.

    The serverless model costs nothing at rest and serves every configured
    region, so replacing it with one endpoint would be a silent downgrade.

    Ref: stdapi/models/marketplace_endpoints.py:merge_marketplace_endpoint_models
    """
    serverless = ModelDetails(
        id=MODEL_ID,
        name="Serverless",
        provider="Vendor",
        input_modalities=["TEXT"],
        output_modalities=["TEXT"],
        regions=["eu-west-1"],
    )
    all_models = {MODEL_ID: serverless}
    endpoint = _published(_endpoint())

    merge_marketplace_endpoint_models(all_models, {MODEL_ID: endpoint})

    assert all_models[MODEL_ID] is serverless
    assert MODEL_ID not in MARKETPLACE_ENDPOINT_MODELS
    assert usage_service(MODEL_ID) is Service.BEDROCK


def test_published_endpoint_bills_under_its_own_service() -> None:
    """Marketplace usage is keyed apart from bedrock-runtime usage.

    AWS bills the endpoint by the instance-hour and publishes no per-token rate,
    so the quantities must not merge into a record the price catalog can price.

    Ref: stdapi/models/__init__.py:usage_service
    """
    endpoint = _published(_endpoint())
    all_models: dict[str, ModelDetails] = {}

    merge_marketplace_endpoint_models(all_models, {MODEL_ID: endpoint})

    assert usage_service(MODEL_ID) is Service.BEDROCK_MARKETPLACE
    assert usage_service(ENDPOINT_ARN) is Service.BEDROCK_MARKETPLACE
    assert usage_service("amazon.nova-micro-v1:0") is Service.BEDROCK
    assert is_marketplace_endpoint(MODEL_ID)


def test_an_endpoint_that_stopped_being_discovered_is_unpublished() -> None:
    """A refresh that finds nothing empties the index, rather than keeping it.

    A stale ID left behind keeps forcing the generic chat model and keeps
    billing under a service with no per-token rate, for a listing bedrock-runtime
    may well serve on its own.

    Ref: stdapi/models/marketplace_endpoints.py:merge_marketplace_endpoint_models
    """
    all_models: dict[str, ModelDetails] = {}
    merge_marketplace_endpoint_models(all_models, {MODEL_ID: _published(_endpoint())})
    assert MODEL_ID in MARKETPLACE_ENDPOINT_MODELS

    merge_marketplace_endpoint_models({}, {})

    assert not MARKETPLACE_ENDPOINT_MODELS
    assert not is_marketplace_endpoint(MODEL_ID)
    assert usage_service(MODEL_ID) is Service.BEDROCK


def test_endpoint_advertises_no_capability_gated_route() -> None:
    """An endpoint publishes the plain text routes and nothing a capability gates.

    ``CountTokens`` takes a foundation model identifier, and the class a listing
    name happens to match is not the class that serves it, so publishing either
    one's capabilities would advertise a route that fails at request time.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html
         stdapi/models/__init__.py:_compute_model_capabilities
    """
    # A listing named like a family whose matcher could one day widen to it.
    endpoint = _published(
        _endpoint(
            modelSourceIdentifier=SOURCE_ARN.replace(
                "huggingface-reasoning-qwen3-4b", "amazon.nova-marketplace"
            )
        )
    )
    serverless = endpoint.model_copy(
        update={"service": RUNTIME_SERVICE, "marketplace_endpoints": None}
    )
    gated = {cap.path for cap in ROUTE_CAPABILITIES.values() if cap.required_capability}
    assert gated, "no route is capability-gated -- nothing was checked"

    routes, _tools = _compute_model_capabilities(endpoint.id, endpoint)

    # The same ID as a serverless model does reach a capability-gated route.
    assert gated & set(_compute_model_capabilities(endpoint.id, serverless)[0])
    assert not gated & set(routes)
    assert routes, "a text model endpoint must still publish its ungated routes"


def test_endpoint_advertises_neither_token_counting_route() -> None:
    """Both token counters are withheld, including the one no capability gates.

    ``/anthropic/v1/messages/count_tokens`` is published for every text model
    because Bedrock Mantle serves it through a counter of its own, so its
    absence here cannot come from the capability flags -- and a route the
    catalogue advertises that the model cannot answer is the bug this guards.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html
         stdapi/models/__init__.py:_TOKEN_COUNTING_OPERATIONS
    """
    endpoint = _published(_endpoint())
    # A serverless Bedrock model of the same shape, which publishes both.
    serverless_id = "amazon.nova-micro-v1:0"
    serverless = endpoint.model_copy(
        update={"service": RUNTIME_SERVICE, "marketplace_endpoints": None}
    )
    counting = {
        ROUTE_CAPABILITIES[op].path
        for op in _TOKEN_COUNTING_OPERATIONS
        if op in ROUTE_CAPABILITIES
    }
    assert len(counting) == 2, f"both counters must be registered, got {counting}"

    routes, tools = _compute_model_capabilities(MODEL_ID, endpoint)

    assert counting <= set(_compute_model_capabilities(serverless_id, serverless)[0])
    assert not counting & set(routes)
    assert not _TOKEN_COUNTING_OPERATIONS & set(tools)


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
def test_token_counting_is_refused_by_the_gateway(
    app_client: TestClient,
    api_key: str,
    route: str,
    payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Counting tokens for an endpoint model answers 400 from the gateway itself.

    The backend would refuse it too, but with a validation error about an ARN
    the caller never wrote and cannot act on. The gateway knows the counter
    takes a foundation model identifier, so it answers first.

    Ref: stdapi/models/__init__.py:reject_unsupported_token_counting
    """
    endpoint = _published(_endpoint())
    monkeypatch.setitem(_MODELS, MODEL_ID, endpoint)

    response = app_client.post(
        route,
        json={"model": MODEL_ID, **payload},
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 400, response.text
    assert "Token counting is not supported" in response.text
    # Nothing of the backend reaches the caller (AGENTS.md, Never leak internals).
    assert "sagemaker" not in response.text.lower()
    assert "arn:aws" not in response.text


def test_endpoint_resolves_to_the_generic_converse_implementation() -> None:
    """A listing name must not be captured by a serverless model's family class.

    A family class encodes what one serverless model does differently; an
    endpoint's divergences are its container's, which Amazon Bedrock has already
    mapped onto Converse.

    Ref: stdapi/models/chat/__init__.py:get_chat_model
    """
    endpoint = _published(
        _endpoint(
            modelSourceIdentifier=SOURCE_ARN.replace(
                "huggingface-reasoning-qwen3-4b", "deepseek-llm-r1-distill-qwen-7b"
            )
        )
    )
    merge_marketplace_endpoint_models({}, {endpoint.id: endpoint})

    assert type(get_chat_model(endpoint.id)) is ChatModel


def test_endpoint_arn_as_a_model_id_is_refused_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Naming an endpoint ARN directly is opt-in, because it directs paid traffic.

    Ref: stdapi/config.py:Settings.aws_bedrock_allow_marketplace_endpoint_arn
    """
    monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_marketplace_endpoint_arn", False)

    with pytest.raises(ApiError, match="not allowed by server configuration"):
        _marketplace_endpoint_from_arn(ENDPOINT_ARN)


def test_endpoint_arn_as_a_model_id_when_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An allowed endpoint ARN resolves to details pinned to its own region.

    Ref: stdapi/models/__init__.py:_marketplace_endpoint_from_arn
    """
    monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_marketplace_endpoint_arn", True)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["eu-west-1"])

    model = _marketplace_endpoint_from_arn(ENDPOINT_ARN)

    assert model is not None
    assert model.regions == ["eu-west-1"]
    assert model.get_id("eu-west-1", inference_profile=True) == ENDPOINT_ARN
    with pytest.raises(ModelRegionUnavailableError):
        model.get_id("us-east-1", inference_profile=True)


def test_an_endpoint_arn_outside_the_configured_regions_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The region check is the only guard on an ARN nobody discovered.

    A deployment restricted to one region for data residency must not route a
    client-named ARN to another: without this the model resolves with that
    region and the invocation reaches a region the operator never approved --
    or a client pool that has none, which is a 500.

    Ref: stdapi/aws_bedrock.py:validate_bedrock_region
    """
    monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_marketplace_endpoint_arn", True)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["eu-west-1"])
    foreign = "arn:aws:sagemaker:us-east-1:123456789012:endpoint/qwen3-4b"

    with pytest.raises(ApiError, match="is not a configured Bedrock region"):
        _marketplace_endpoint_from_arn(foreign)


async def test_an_endpoint_arn_resolves_on_the_invocation_path(
    monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
) -> None:
    """The ID ``validate_model`` returns for an ARN is one the invocation resolves.

    ``_marketplace_endpoint_from_arn`` answers details whose ``id`` is the ARN,
    which is never a catalog key; the invocation path re-resolves that ID
    through ``get_model_details``, so without a fallback there every chat
    request naming an ARN raised ``KeyError`` and answered HTTP 500 -- a
    documented way of naming a model that 500s.

    Ref: stdapi/models/__init__.py:get_model_details
         stdapi/models/__init__.py:compute_candidate_regions
         stdapi/models/__init__.py:resolve_routed_model_id
    """
    del request_log
    monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_marketplace_endpoint_arn", True)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["eu-west-1"])

    details = await get_model_details(ENDPOINT_ARN)

    assert details.id == ENDPOINT_ARN
    assert details.service == MARKETPLACE_SERVICE
    # The two invocation-path lookups the generic ChatModel makes for it.
    assert await compute_candidate_regions(ENDPOINT_ARN) == ["eu-west-1"]
    assert (
        await resolve_routed_model_id(ENDPOINT_ARN, "eu-west-1", inference_profile=True)
        == ENDPOINT_ARN
    )
    assert usage_service(ENDPOINT_ARN) is Service.BEDROCK_MARKETPLACE
    # An unknown model is still a KeyError, which every caller relies on.
    with pytest.raises(KeyError):
        await get_model_details("no-such-model")


def test_a_non_endpoint_arn_is_left_to_the_other_resolvers() -> None:
    """Only the endpoint ARN shape is claimed here.

    Ref: stdapi/utils.py:match_marketplace_endpoint_arn
    """
    assert (
        _marketplace_endpoint_from_arn(
            "arn:aws:bedrock:eu-west-1:123456789012:inference-profile/x"
        )
        is None
    )
    assert _marketplace_endpoint_from_arn("amazon.nova-micro-v1:0") is None
