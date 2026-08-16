"""Route-capability gating in the model registry.

A route is advertised for a model only when the model's modalities and its
model class' capability flags both satisfy the route's ``RouteCapability``
descriptor. ``COUNT_TOKENS`` is the discriminating flag here: Bedrock
``CountTokens`` has no Mantle equivalent. A route that is not an MCP tool is
advertised as a path and stays filterable by its operation ID, but never
appears in ``supported_mcp_tools``.

Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CountTokens.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/models-endpoint-availability.html
     stdapi/models/__init__.py:_compute_model_capabilities
     stdapi/models/capabilities.py:RouteCapability
"""

from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

import stdapi.routes.openai_responses  # noqa: F401  (registers the input-tokens route capability)
from stdapi import models
from stdapi.config import SETTINGS
from stdapi.models import MANTLE_SERVICE, ModelDetails, _compute_model_capabilities
from stdapi.models.capabilities import ROUTE_CAPABILITIES, Capability
from stdapi.models.chat._default import ChatModel as ConverseChatModel
from stdapi.models.chat._mantle._default import ChatModel as MantleChatModel
from stdapi.pricing import Dimension
from tests._helpers import make_model_details
from tests.conftest import set_test_price

if TYPE_CHECKING:
    from collections.abc import Generator

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: MCP tool that must be gated on the COUNT_TOKENS capability.
_INPUT_TOKENS_TOOL = "openai_response_input_tokens"

#: Ungated sibling tool on the same modalities, used as the gating control.
_RESPONSE_TOOL = "openai_response"

#: Live speech-to-speech model, the discovery case for the realtime surface.
_SPEECH_MODEL = "amazon.nova-2-sonic-v1:0"

#: Operation ID of the realtime route, which has no MCP tool.
_REALTIME_OPERATION = "openai_realtime"

#: Text model AWS publishes a batch rate for.
_BATCH_MODEL = "vendor.batchmodel"

#: Price-catalog key ``_BATCH_MODEL`` normalizes to.
_BATCH_MODEL_PRICE_KEY = "vendorbatchmodel"

#: Text model with no published batch rate.
_PLAIN_MODEL = "vendor.plainmodel"

#: Mantle-served model, named as the runtime endpoint does not know it.
_MANTLE_MODEL = "vendor.twinmodel"

#: Price-catalog key ``_MANTLE_MODEL`` normalizes to.
_MANTLE_MODEL_PRICE_KEY = "vendortwinmodel"

#: The same model as ``_MANTLE_MODEL``, named as the runtime endpoint knows it.
_TWIN_MODEL = "vendor.twinmodel-1:0"

#: Price-catalog key ``_TWIN_MODEL`` normalizes to, distinct from the Mantle one.
_TWIN_MODEL_PRICE_KEY = "vendortwinmodel10"


def _text_model(service: str) -> ModelDetails:
    """Build a minimal TEXT/TEXT model detail for the given hosting service.

    Args:
        service: AWS service hosting the model.

    Returns:
        A ModelDetails with TEXT input and output modalities.
    """
    return ModelDetails(
        id="test.model-v1:0",
        name="Test Model",
        provider="Test",
        service=service,
        input_modalities=["TEXT"],
        output_modalities=["TEXT"],
        regions=["us-east-1"],
    )


class TestCountTokensCapabilityGating:
    """The input-tokens tool is advertised only by models that can count tokens.

    Ref: stdapi/models/__init__.py:_compute_model_capabilities
         stdapi/routes/openai_responses.py:count_input_tokens
    """

    def test_converse_declares_count_tokens(self) -> None:
        """Converse chat models declare the COUNT_TOKENS capability.

        Ref: stdapi/models/chat/_default.py:ChatModel
        """
        assert ConverseChatModel.get_supported_operations() & Capability.COUNT_TOKENS

    def test_mantle_does_not_declare_count_tokens(self) -> None:
        """Mantle chat models do not declare the COUNT_TOKENS capability.

        Bedrock ``CountTokens`` lives on bedrock-runtime only, so the Mantle
        chat model cannot serve it.

        Ref: stdapi/models/chat/_mantle/_default.py:ChatModel
        """
        assert not (
            MantleChatModel.get_supported_operations() & Capability.COUNT_TOKENS
        )

    def test_converse_model_advertises_input_tokens_tool(self) -> None:
        """A Converse-served TEXT model lists the input-tokens tool and route."""
        with patch.object(
            models,
            "_model_capability_flags",
            return_value=ConverseChatModel.get_supported_operations(),
        ):
            routes, tools = _compute_model_capabilities(
                "test.model-v1:0", _text_model("AWS Bedrock Runtime")
            )
        assert _INPUT_TOKENS_TOOL in tools
        assert ROUTE_CAPABILITIES[_INPUT_TOKENS_TOOL].path in routes

    def test_mantle_model_hides_input_tokens_tool(self) -> None:
        """A Mantle-served TEXT model omits the input-tokens tool it always rejects.

        Only the ``COUNT_TOKENS``-gated entry is dropped: the ungated Responses
        route on the same modalities stays advertised.
        """
        with patch.object(
            models,
            "_model_capability_flags",
            return_value=MantleChatModel.get_supported_operations(),
        ):
            routes, tools = _compute_model_capabilities(
                "test.model-v1:0", _text_model(MANTLE_SERVICE)
            )
        assert _INPUT_TOKENS_TOOL not in tools
        assert ROUTE_CAPABILITIES[_INPUT_TOKENS_TOOL].path not in routes
        assert _RESPONSE_TOOL in tools, (
            "gating must drop only the COUNT_TOKENS route, not every TEXT route"
        )


class TestOperatorModelAliases:
    """MODEL_ALIASES rebuild: operator aliases are applied last and therefore win.

    ``MODEL_ALIASES`` is rebuilt from the model classes on every catalog refresh, so
    an operator alias only survives because ``SETTINGS.model_aliases`` is merged
    after the class-provided ones.

    Ref: stdapi/models/__init__.py:_populate_model_aliases
         stdapi/models/__init__.py:resolve_model_alias
    """

    @staticmethod
    def _rebuild(
        monkeypatch: pytest.MonkeyPatch, operator_aliases: dict[str, str]
    ) -> dict[str, ModelDetails]:
        """Rebuild the alias table from one fake class plus *operator_aliases*.

        Returns:
            The catalog the aliases were rebuilt against.
        """

        class _AliasingModel:
            @staticmethod
            def get_aliases(_all_models: dict[str, ModelDetails]) -> dict[str, str]:
                return {"fast": "other.model-v1:0"}

        all_models = {
            "test.model-v1:0": _text_model("AWS Bedrock Runtime"),
            "other.model-v1:0": _text_model("AWS Bedrock Runtime"),
        }
        all_models["other.model-v1:0"].id = "other.model-v1:0"
        monkeypatch.setattr(models, "MODEL_ALIASES", {})
        monkeypatch.setattr(models, "_GLOBAL_MODEL_REGISTRY", [_AliasingModel])
        monkeypatch.setattr(SETTINGS, "model_aliases", operator_aliases)
        models._populate_model_aliases(all_models)  # noqa: SLF001
        return all_models

    def test_class_alias_applies_without_an_operator_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no operator alias configured, the model class' own alias is used."""
        self._rebuild(monkeypatch, {})
        assert models.resolve_model_alias("fast") == "other.model-v1:0"

    def test_operator_alias_overrides_the_class_alias(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator alias reusing a built-in name redirects it to the operator target."""
        self._rebuild(monkeypatch, {"fast": "test.model-v1:0"})
        assert models.resolve_model_alias("fast") == "test.model-v1:0"

    def test_operator_alias_is_advertised_on_the_target_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An operator alias pointing at a catalog model is listed in its ``aliases``.

        The reverse index is built from the merged table, so ``GET /v1/models`` shows
        the operator alias and no longer shows it on the built-in target.
        """
        all_models = self._rebuild(monkeypatch, {"fast": "test.model-v1:0"})
        assert all_models["test.model-v1:0"].aliases == ["fast"]
        assert not all_models["other.model-v1:0"].aliases

    def test_unknown_alias_is_returned_unchanged(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A name that is not an alias is passed through as a model ID."""
        self._rebuild(monkeypatch, {})
        assert models.resolve_model_alias("test.model-v1:0") == "test.model-v1:0"


@pytest.fixture(scope="module")
def mcp_tool_names() -> frozenset[str]:
    """Names of every MCP tool ``fastapi_mcp`` builds from the mounted application.

    Tools are derived from the OpenAPI operations, so a route without one -- a
    WebSocket route -- yields no tool at all.

    Returns:
        The tool names an MCP client can call on this server.
    """
    from fastapi_mcp import FastApiMCP  # type: ignore[import-untyped] # noqa: PLC0415

    from stdapi.main import app  # noqa: PLC0415

    return frozenset(
        tool.name for tool in FastApiMCP(app, name="test", description="test").tools
    )


@pytest.fixture
def isolated_catalog() -> Generator[None]:
    """Restore the extra-model registry and the rebuilt indexes after the test.

    Ref: stdapi/models/__init__.py:update_unified_models_collections
    """
    from stdapi.models import EXTRA_MODELS  # noqa: PLC0415

    saved = dict(EXTRA_MODELS)
    yield
    EXTRA_MODELS.clear()
    EXTRA_MODELS.update(saved)
    models.update_unified_models_collections()


def _speech_model() -> ModelDetails:
    """Build the details of a live speech model: SPEECH in, SPEECH and TEXT out.

    Returns:
        A ModelDetails matching the live speech model family.
    """
    return make_model_details(
        _SPEECH_MODEL, input_modalities=["SPEECH"], output_modalities=["SPEECH", "TEXT"]
    )


class TestRealtimeRouteAdvertised:
    """A live speech model advertises the audio and realtime surfaces it can serve.

    The capability flags are unioned across every family serving the model, so a
    model registered as both an audio and a realtime one advertises the routes of
    both. The realtime route is a WebSocket route with no MCP tool, so it is
    advertised as a path only.

    Ref: https://stdapi.ai/api_openai_realtime/
         stdapi/models/__init__.py:_model_capability_flags
         stdapi/routes/openai_realtime.py:register_route_capability
    """

    def test_speech_model_advertises_every_family_route(self) -> None:
        """Transcription, translation and realtime are all advertised, and nothing else.

        Taking only the most specific model class overall would drop two of the
        three, leaving the model undiscoverable for those surfaces.

        Ref: stdapi/routes/openai_audio.py:register_route_capability
        """
        import stdapi.main  # noqa: F401, PLC0415

        routes, _tools = _compute_model_capabilities(_SPEECH_MODEL, _speech_model())

        openai = SETTINGS.openai_routes_prefix
        assert routes == [
            f"{openai}/v1/audio/transcriptions",
            f"{openai}/v1/audio/translations",
            f"{openai}/v1/realtime",
        ]

    def test_realtime_is_not_advertised_as_an_mcp_tool(self) -> None:
        """The realtime operation ID stays out of ``supported_mcp_tools``.

        Its route is advertised all the same, so only the tool list is trimmed.
        """
        import stdapi.main  # noqa: F401, PLC0415

        routes, tools = _compute_model_capabilities(_SPEECH_MODEL, _speech_model())

        assert _REALTIME_OPERATION not in tools
        assert tools == ["openai_audio_transcription", "openai_audio_translation"]
        assert ROUTE_CAPABILITIES[_REALTIME_OPERATION].path in routes

    def test_advertised_tools_are_tools_the_server_exposes(
        self, mcp_tool_names: frozenset[str]
    ) -> None:
        """Every advertised MCP tool name is a tool an MCP client can actually call.

        An agent that follows ``supported_mcp_tools`` must never be sent to a
        tool the server does not expose.

        Ref: stdapi/mcp.py:mount_mcp
        """
        _routes, tools = _compute_model_capabilities(_SPEECH_MODEL, _speech_model())

        assert tools, "the live speech model advertises at least one tool"
        assert not set(tools) - mcp_tool_names

    def test_realtime_operation_id_still_filters_the_catalogue(
        self, isolated_catalog: None
    ) -> None:
        """``route=openai_realtime`` resolves to the models serving that route.

        ``search_models`` accepts a path or an operation ID; the operation ID of
        a route without an MCP tool is indexed even though no model lists it as
        a tool.

        Ref: stdapi/routes/core_models.py:search_models
        """
        import stdapi.main  # noqa: F401, PLC0415
        from stdapi.models import EXTRA_MODELS  # noqa: PLC0415

        EXTRA_MODELS[_SPEECH_MODEL] = _speech_model()
        models.update_unified_models_collections()

        index = models._ALL_MODELS_BY_ROUTE_OR_TOOL  # noqa: SLF001
        assert _SPEECH_MODEL in index[_REALTIME_OPERATION]
        assert _SPEECH_MODEL in index[ROUTE_CAPABILITIES[_REALTIME_OPERATION].path]


class TestBatchAdvertisement:
    """A model is advertised for the Batch API when a batch rate is published for it.

    The rate is the only signal available before a batch is submitted, and it
    is a hint rather than a rule: the flag never decides whether a request is
    accepted, which ``tests/test_openai_batches.py`` covers on the batch route.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-supported.html
         https://stdapi.ai/api_search_models/
         stdapi/models/__init__.py:sync_batch_support
         stdapi/pricing.py:batch_priced_models
    """

    @staticmethod
    def _catalog() -> dict[str, ModelDetails]:
        """Register two text models, one of them batch-priced, and rebuild the indexes.

        Returns:
            The two registered models, keyed by model ID.
        """
        from stdapi.models import EXTRA_MODELS  # noqa: PLC0415

        registered = {
            model_id: make_model_details(model_id)
            for model_id in (_BATCH_MODEL, _PLAIN_MODEL)
        }
        EXTRA_MODELS.update(registered)
        set_test_price(
            _BATCH_MODEL_PRICE_KEY,
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "0.000001",
            "USD",
            tier="batch",
        )
        models.update_unified_models_collections()
        return registered

    def test_a_batch_priced_model_is_advertised(self, isolated_catalog: None) -> None:
        """The model AWS publishes a batch rate for carries ``batch`` true."""
        assert self._catalog()[_BATCH_MODEL].batch is True
        assert _BATCH_MODEL in models._ALL_MODELS_BATCH  # noqa: SLF001

    def test_a_model_without_a_batch_rate_is_not_advertised(
        self, isolated_catalog: None
    ) -> None:
        """A model priced for no batch rate carries ``batch`` false, not true.

        Advertising every model would make the flag useless for discovery.
        """
        assert self._catalog()[_PLAIN_MODEL].batch is False
        assert _PLAIN_MODEL in models._ALL_MODELS_NON_BATCH  # noqa: SLF001
        assert _PLAIN_MODEL not in models._ALL_MODELS_BATCH  # noqa: SLF001

    def test_an_unloaded_price_catalog_advertises_nothing_either_way(
        self, isolated_catalog: None
    ) -> None:
        """With no price published at all, ``batch`` stays unknown rather than false.

        The catalog loads in the background and can be switched off entirely, so
        deriving "false" from its absence would advertise every model as
        unbatchable on a deployment that simply does not track cost.
        """
        from stdapi.models import EXTRA_MODELS  # noqa: PLC0415

        EXTRA_MODELS[_BATCH_MODEL] = make_model_details(_BATCH_MODEL)
        models.update_unified_models_collections()

        assert EXTRA_MODELS[_BATCH_MODEL].batch is None
        assert not models._ALL_MODELS_BATCH  # noqa: SLF001
        assert not models._ALL_MODELS_NON_BATCH  # noqa: SLF001

    def test_a_price_published_after_the_catalogue_was_built_is_picked_up(
        self, isolated_catalog: None
    ) -> None:
        """A batch rate arriving after startup still reaches the advertisement.

        The price catalog loads in a background task that finishes well after
        the model catalogue is first built, so a flag derived once at build
        time would stay unknown for the whole process lifetime.

        Ref: stdapi/main.py:lifespan
        """
        from stdapi.models import EXTRA_MODELS  # noqa: PLC0415

        EXTRA_MODELS[_BATCH_MODEL] = make_model_details(_BATCH_MODEL)
        models.update_unified_models_collections()
        assert EXTRA_MODELS[_BATCH_MODEL].batch is None

        set_test_price(
            _BATCH_MODEL_PRICE_KEY,
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "0.000001",
            "USD",
            tier="batch",
        )
        models.sync_batch_support()

        assert EXTRA_MODELS[_BATCH_MODEL].batch is True

    def test_a_rebuilt_catalogue_drops_the_flags_derived_before_it(
        self, isolated_catalog: None
    ) -> None:
        """A catalogue rebuilt with no price catalog loaded advertises nothing.

        An unloaded price catalog reads as unchanged, so a rebuild that only
        consults the catalog would keep the flags and the search indexes it
        derived for models the catalogue no longer holds.

        Ref: stdapi/models/__init__.py:sync_batch_support
        """
        from stdapi.pricing import _state  # noqa: PLC0415

        registered = self._catalog()
        assert registered[_BATCH_MODEL].batch is True

        _state.price_index = {}
        models.update_unified_models_collections()

        assert registered[_BATCH_MODEL].batch is None
        assert not models._ALL_MODELS_BATCH  # noqa: SLF001
        assert not models._ALL_MODELS_NON_BATCH  # noqa: SLF001

    def test_a_mantle_served_model_alone_is_not_advertised(
        self, isolated_catalog: None
    ) -> None:
        """A Mantle-served model with no runtime form is not advertised.

        AWS publishes its own batch-tier rate rows under a Mantle usagetype for
        models a batch is refused for outright, so that model's own rate says
        nothing about whether it can be batched.

        Ref: stdapi/batches.py:_batch_model_id
        """
        from stdapi.models import EXTRA_MODELS  # noqa: PLC0415

        EXTRA_MODELS[_MANTLE_MODEL] = make_model_details(
            _MANTLE_MODEL, service=MANTLE_SERVICE
        )
        set_test_price(
            _MANTLE_MODEL_PRICE_KEY,
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "0.000001",
            "USD",
            tier="batch",
        )
        models.update_unified_models_collections()

        assert EXTRA_MODELS[_MANTLE_MODEL].batch is False

    def test_a_mantle_served_model_is_advertised_from_its_runtime_twin(
        self, isolated_catalog: None
    ) -> None:
        """A Mantle-served model is advertised when its runtime form is batch-priced.

        Batches run on the runtime endpoint only, so the rate that follows what
        a batch accepts is the one published for the model under the identifier
        that endpoint knows it by -- never its own.

        Ref: stdapi/models/__init__.py:build_runtime_twins
        """
        from stdapi.models import EXTRA_MODELS  # noqa: PLC0415

        EXTRA_MODELS[_MANTLE_MODEL] = make_model_details(
            _MANTLE_MODEL, service=MANTLE_SERVICE
        )
        EXTRA_MODELS[_TWIN_MODEL] = make_model_details(_TWIN_MODEL)
        set_test_price(
            _TWIN_MODEL_PRICE_KEY,
            "us-east-1",
            Dimension.INPUT_TOKENS,
            "0.000001",
            "USD",
            tier="batch",
        )
        models.update_unified_models_collections()

        assert EXTRA_MODELS[_MANTLE_MODEL].batch is True
        assert _MANTLE_MODEL in models._ALL_MODELS_BATCH  # noqa: SLF001


class TestRuntimeTwins:
    """A Mantle-served model is paired with the runtime model naming it.

    Batch inference runs on the runtime endpoint only, so a Mantle-served model
    is batchable exactly when the same model exists there under another
    identifier. Every case below is one the pairing rule was derived from, and
    each fails without one part of the normalisation.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-supported.html
         stdapi/models/__init__.py:build_runtime_twins
    """

    @staticmethod
    def _pair(mantle_id: str, runtime_id: str, runtime_name: str) -> str | None:
        """Register one Mantle model and one runtime model, and pair them.

        Args:
            mantle_id: Identifier of the Mantle-served model.
            runtime_id: Identifier of the runtime model.
            runtime_name: Display name AWS gives the runtime model.

        Returns:
            The runtime identifier the Mantle model resolves to, if any.
        """
        from stdapi.models import EXTRA_MODELS  # noqa: PLC0415

        EXTRA_MODELS[mantle_id] = make_model_details(mantle_id, service=MANTLE_SERVICE)
        EXTRA_MODELS[runtime_id] = make_model_details(runtime_id, name=runtime_name)
        models.update_unified_models_collections()
        return models.runtime_twin(mantle_id)

    @pytest.mark.parametrize(
        ("mantle_id", "runtime_id", "runtime_name"),
        [
            ("openai.gpt-oss-120b", "openai.gpt-oss-120b-1:0", "gpt-oss-120b"),
            (
                "anthropic.claude-haiku-4-5",
                "anthropic.claude-haiku-4-5-20251001-v1:0",
                "Claude Haiku 4.5",
            ),
            (
                "qwen.qwen3-next-80b-a3b-instruct",
                "qwen.qwen3-next-80b-a3b",
                "Qwen3-Next-80B-A3B",
            ),
            ("qwen.qwen3-32b", "qwen.qwen3-32b-v1:0", "Qwen3 32B (dense)"),
            (
                "qwen.qwen3-coder-30b-a3b-instruct",
                "qwen.qwen3-coder-30b-a3b-v1:0",
                "Qwen3-Coder-30B-A3B-Instruct",
            ),
            ("deepseek.v3.1", "deepseek.v3-v1:0", "DeepSeek-V3.1"),
            (
                "moonshotai.kimi-k2-thinking",
                "moonshot.kimi-k2-thinking",
                "Kimi K2 Thinking",
            ),
        ],
    )
    def test_the_runtime_form_of_a_mantle_model_is_found(
        self, isolated_catalog: None, mantle_id: str, runtime_id: str, runtime_name: str
    ) -> None:
        """Each observed pair is derived from the catalogue, not from a list."""
        assert self._pair(mantle_id, runtime_id, runtime_name) == runtime_id

    def test_a_mantle_model_with_no_runtime_form_pairs_with_nothing(
        self, isolated_catalog: None
    ) -> None:
        """An unrelated runtime model is not mistaken for the same model.

        A pairing rule that matched too widely would submit a batch naming a
        model the caller never asked for.
        """
        assert (
            self._pair("xai.grok-4.3", "amazon.nova-micro-v1:0", "Nova Micro") is None
        )

    def test_a_runtime_model_pairs_with_nothing(self, isolated_catalog: None) -> None:
        """A model already served by the runtime endpoint needs no pairing."""
        from stdapi.models import EXTRA_MODELS  # noqa: PLC0415

        EXTRA_MODELS[_TWIN_MODEL] = make_model_details(_TWIN_MODEL)
        models.update_unified_models_collections()

        assert models.runtime_twin(_TWIN_MODEL) is None
