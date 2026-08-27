"""Ollama-compatible model discovery: /api/tags, /api/show, /api/ps, /api/version.

Driven through the official ``ollama`` client, so every shape asserted here is
one that client can actually parse. Everything these endpoints report about a
*file* on disk -- size, quantization, parameter count, modelfile, template,
license -- describes something a hosted model does not have. Ollama Cloud has
the same nothing to report for its own cloud models: empty ``details`` on
``/api/tags``, and no ``license``/``modelfile``/``template``/``parameters`` key
at all on ``/api/show``. Each is reported the way Ollama reports an unknown
value, and never filled with a plausible invention.

Ref: https://docs.ollama.com/api/tags
     https://docs.ollama.com/openapi.yaml (#/paths/~1api~1show)
     stdapi/routes/ollama_models.py
     stdapi/models/__init__.py:_lookup_with_latest_fallback
"""

from typing import TYPE_CHECKING

import ollama
import pytest

from stdapi.api_providers.ollama import OLLAMA_API_VERSION
from stdapi.models import _lookup_with_latest_fallback
from stdapi.routes.ollama_models import model_capabilities, model_digest
from tests._helpers import make_model_details

if TYPE_CHECKING:
    from collections.abc import Callable

    import httpx

    from stdapi.models import ModelDetails


#: A real Bedrock model ID: it ends in a ``:<version>`` suffix of its own.
_BEDROCK_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0"

#: An Ollama-shaped name neither target serves: absent from the Bedrock
#: catalogue, and refused by Ollama Cloud, which hosts no ``llama3.2``.
UNKNOWN_MODEL = "llama3.2:3b"

#: Version Ollama Cloud reports, which declares no version at all.
_CLOUD_VERSION = "0.0.0"


@pytest.mark.gateway("Exercises the gateway's own catalogue lookup")
class TestLatestTagResolution:
    """``:latest`` is a fallback and never a pre-strip.

    Ollama names a model ``<name>:<tag>``, but a Bedrock model ID carries a
    colon too, so stripping the suffix before the exact lookup would break every
    real identifier. The exact name is therefore tried first, always.

    Ref: stdapi/models/__init__.py:_lookup_with_latest_fallback
    """

    @staticmethod
    def _lookup(
        catalog: dict[str, ModelDetails],
    ) -> Callable[[str], ModelDetails | None]:
        """Build a catalogue lookup over *catalog*.

        Args:
            catalog: Models keyed by identifier.

        Returns:
            A callable resolving an exact identifier, or None.
        """
        return catalog.get

    def test_a_real_bedrock_id_resolves_untouched(self) -> None:
        """An ID ending in ``:0`` is found by its exact spelling."""
        catalog = {_BEDROCK_ID: make_model_details(_BEDROCK_ID)}
        model, resolved = _lookup_with_latest_fallback(
            self._lookup(catalog), _BEDROCK_ID
        )
        assert model is not None
        assert resolved == _BEDROCK_ID

    def test_the_fallback_never_fires_before_the_exact_lookup(self) -> None:
        """A model literally named ``x:latest`` wins over the untagged one."""
        catalog = {
            "x:latest": make_model_details("x:latest"),
            "x": make_model_details("x"),
        }
        model, resolved = _lookup_with_latest_fallback(
            self._lookup(catalog), "x:latest"
        )
        assert model is not None
        assert model.id == "x:latest"
        assert resolved == "x:latest"

    def test_the_tag_is_dropped_only_after_a_miss(self) -> None:
        """``<id>:latest`` resolves to ``<id>`` when the tagged name is unknown."""
        catalog = {_BEDROCK_ID: make_model_details(_BEDROCK_ID)}
        model, resolved = _lookup_with_latest_fallback(
            self._lookup(catalog), f"{_BEDROCK_ID}:latest"
        )
        assert model is not None
        assert resolved == _BEDROCK_ID

    def test_an_unknown_name_stays_unresolved(self) -> None:
        """A name nothing matches keeps the caller's spelling for the error."""
        model, resolved = _lookup_with_latest_fallback(self._lookup({}), UNKNOWN_MODEL)
        assert model is None
        assert resolved == UNKNOWN_MODEL


@pytest.mark.gateway("Exercises the gateway's own catalogue-to-Ollama mapping")
class TestDerivedMetadata:
    """The catalogue is the only source, and nothing else is invented.

    Ref: stdapi/routes/ollama_models.py:model_capabilities
    """

    def test_digest_is_stable_and_model_specific(self) -> None:
        """The digest keys a model without claiming to address its content."""
        assert model_digest("a") == model_digest("a")
        assert model_digest("a") != model_digest("b")
        assert len(model_digest("a")) == 64

    def test_capabilities_are_derived_from_the_routes_and_modalities(self) -> None:
        """Chat, tools, vision and audio follow from what the catalogue knows."""
        model = make_model_details(
            "vendor.chat-v1",
            input_modalities=["TEXT", "IMAGE", "SPEECH"],
            supported_routes=["/api/chat"],
        )
        assert model_capabilities(model) == ["completion", "tools", "vision", "audio"]

    def test_thinking_and_insert_are_never_advertised(self) -> None:
        """No per-model source exists for either, so neither is claimed.

        Ollama Cloud advertises ``thinking`` on every model it hosts; the
        Bedrock catalogue records nothing equivalent, and a client shown the
        capability would offer a control that silently does nothing.
        """
        model = make_model_details("vendor.chat-v1", supported_routes=["/api/chat"])
        assert "thinking" not in model_capabilities(model)
        assert "insert" not in model_capabilities(model)

    def test_an_embedding_model_advertises_embedding_alone(self) -> None:
        """An embedding model offers no completion."""
        model = make_model_details(
            "vendor.embed-v1",
            output_modalities=["EMBEDDING"],
            supported_routes=["/api/embed"],
        )
        assert model_capabilities(model) == ["embedding"]

    def test_a_model_serving_no_ollama_route_advertises_nothing(self) -> None:
        """Nothing usable through this dialect means nothing is listed."""
        assert model_capabilities(make_model_details("vendor.speech-v1")) == []

    @pytest.mark.parametrize(
        ("modality", "capability"), [("IMAGE", "vision"), ("SPEECH", "audio")]
    )
    def test_an_input_modality_alone_never_lists_a_model(
        self, modality: str, capability: str
    ) -> None:
        """A model that takes images or speech but serves neither route stays unlisted.

        ``/api/tags`` publishes whatever carries a capability, so a modality
        counted on its own would offer an image generator or a speech-to-text
        model in every Ollama client's model picker -- and the request that
        followed would be refused for the output modality.

        Ref: stdapi/routes/ollama_models.py:model_capabilities
             https://docs.ollama.com/api/tags
        """
        model = make_model_details(
            "vendor.generate-v1",
            input_modalities=["TEXT", modality],
            output_modalities=[modality],
        )

        assert capability not in model_capabilities(model)
        assert model_capabilities(model) == []


def test_tags_lists_models_by_their_canonical_name(
    ollama_client: ollama.Client, ollama_chat_model: str, use_official_api: bool
) -> None:
    """The listing publishes the names to send back as ``model``.

    ``details`` is where the two targets differ, and only in what each of them
    genuinely knows: Ollama Cloud publishes no GGUF metadata for a cloud model
    and leaves ``family`` empty, while the gateway has the model's provider and
    publishes that. Everything a model *file* would supply is empty on both.

    Ref: https://docs.ollama.com/api/tags
    """
    listing = ollama_client.list()
    assert listing.models
    entry = next(model for model in listing.models if model.model == ollama_chat_model)
    assert entry.digest
    assert entry.modified_at is not None
    assert entry.size is not None
    details = entry.details
    assert details is not None
    # Nothing here comes from a model file, so nothing pretends to.
    assert details.parent_model == ""
    assert details.format == ""
    assert details.parameter_size == ""
    assert details.quantization_level == ""
    if use_official_api:
        assert details.family == ""
        assert details.families is None
    else:
        assert entry.size == 0
        assert len(entry.digest) == 64
        assert details.family
        assert details.families == [details.family]


def test_a_tags_entry_repeats_the_name_beside_the_model(
    ollama_http: httpx.Client, ollama_chat_model: str
) -> None:
    """Every entry carries ``name`` as well as ``model``, with the same value.

    Raw HTTP: the official client's ``ListResponse.Model`` has no ``name``
    field and drops it, but older clients read that one instead.

    Ref: https://docs.ollama.com/api/tags
    """
    response = ollama_http.get("/api/tags")
    assert response.status_code == 200
    entry = next(
        model
        for model in response.json()["models"]
        if model["model"] == ollama_chat_model
    )
    assert entry["name"] == ollama_chat_model


def test_tags_answers_a_head_probe(ollama_http: httpx.Client) -> None:
    """`HEAD /api/tags` is used as a liveness probe.

    Raw HTTP: the official client exposes no ``HEAD`` call.

    Ref: ollama/ollama@main:server/routes.go (GenerateRoutes)
    """
    assert ollama_http.head("/api/tags").status_code == 200


def test_show_describes_a_model(
    ollama_client: ollama.Client, ollama_chat_model: str, use_official_api: bool
) -> None:
    """Show reports the capabilities it can derive and nothing it cannot.

    ``license``, ``modelfile``, ``template`` and ``parameters`` are absent from
    both targets' answers: they describe a local model file, and neither a cloud
    model nor a Bedrock one has one. ``model_info`` is GGUF metadata read out of
    that same file, which Ollama Cloud keeps a copy of and the Bedrock catalogue
    records nothing of, so the gateway answers an empty object rather than
    inventing an architecture and a context length.

    Ref: https://docs.ollama.com/openapi.yaml (#/paths/~1api~1show)
    """
    info = ollama_client.show(ollama_chat_model)
    assert info.capabilities is not None
    assert "completion" in info.capabilities
    assert info.details is not None
    assert info.modified_at is not None
    assert info.license is None
    assert info.modelfile is None
    assert info.template is None
    assert info.parameters is None
    if use_official_api:
        assert info.modelinfo
        assert "general.architecture" in info.modelinfo
    else:
        assert info.modelinfo == {}


def test_show_accepts_the_legacy_name_field(
    ollama_http: httpx.Client, ollama_chat_model: str
) -> None:
    """`name` still names the model, as the Ollama server accepts it.

    Raw HTTP: the official client's ``ShowRequest`` only has ``model``.

    Ref: ollama/ollama@main:server/routes.go (ShowHandler)
    """
    response = ollama_http.post("/api/show", json={"name": ollama_chat_model})
    assert response.status_code == 200
    assert response.json()["capabilities"]


def test_show_refuses_an_unknown_model(ollama_client: ollama.Client) -> None:
    """An unknown model answers 404 in the Ollama error envelope.

    Ref: https://docs.ollama.com/openapi.yaml (ErrorResponse)
    """
    with pytest.raises(ollama.ResponseError) as raised:
        ollama_client.show(UNKNOWN_MODEL)
    assert raised.value.status_code == 404
    assert UNKNOWN_MODEL in raised.value.error


def test_ps_reports_nothing_resident(
    ollama_client: ollama.Client, use_official_api: bool
) -> None:
    """Nothing is ever resident: models are served on demand.

    Truthful rather than a stub -- there is no load phase to report on, which is
    also why ``load_duration`` is omitted everywhere else. Ollama Cloud does not
    serve this endpoint to a cloud API key at all and answers 401, so the
    gateway is the more complete of the two here.

    Ref: https://docs.ollama.com/api/ps
    """
    if use_official_api:
        with pytest.raises(ollama.ResponseError) as raised:
            ollama_client.ps()
        assert raised.value.status_code == 401
        return
    assert ollama_client.ps().models == []


def test_version_declares_the_targeted_ollama_api(
    ollama_http: httpx.Client, use_official_api: bool
) -> None:
    """The version is a compatibility declaration, not this server's own version.

    Clients probe it to decide whether an endpoint is an Ollama server and which
    features they may send. Ollama Cloud declares none at all and answers
    ``0.0.0``; the gateway names the release whose contract it targets, which is
    what a client gating a feature on the version needs.

    Raw HTTP: the official client exposes no version call.

    Ref: https://docs.ollama.com/openapi.yaml (#/paths/~1api~1version)
    """
    response = ollama_http.get("/api/version")
    assert response.status_code == 200
    expected = _CLOUD_VERSION if use_official_api else OLLAMA_API_VERSION
    assert response.json() == {"version": expected}


def test_version_answers_a_head_probe(ollama_http: httpx.Client) -> None:
    """`HEAD` is what several clients use as a liveness probe.

    Raw HTTP: the official client exposes no ``HEAD`` call.

    Ref: ollama/ollama@main:server/routes.go (GenerateRoutes)
    """
    assert ollama_http.head("/api/version").status_code == 200
