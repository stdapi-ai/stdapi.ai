"""Cohere models published under the names Cohere's own API uses.

Every alias asserted here is the ``Model Name`` Cohere publishes next to the
``Amazon Bedrock Model ID`` it is derived from, so an application already
calling Cohere reaches the same model after changing only its base URL.

Ref: https://docs.cohere.com/docs/models
     stdapi/models/_cohere.py
     stdapi/models/__init__.py:ModelBase.get_aliases
"""

from typing import TYPE_CHECKING

import pytest

from stdapi import models
from stdapi.config import SETTINGS
from stdapi.models import MODEL_ALIASES, resolve_model_alias
from stdapi.models.embedding import EmbeddingResponse, get_embedding_model
from stdapi.models.embedding.cohere_embed import EmbeddingModel as CohereEmbeddingModel
from stdapi.models.rerank import RerankedDocument, RerankResponse, get_rerank_model
from stdapi.models.rerank.bedrock_rerank import RerankModel
from stdapi.routes import cohere_embed, cohere_rerank
from tests._helpers import make_model_details

if TYPE_CHECKING:
    from types import ModuleType

    from starlette.testclient import TestClient

    from stdapi.input_file import InputFileUrl
    from stdapi.models import ModelDetails
    from stdapi.types import JsonMapping


#: These tests exercise the in-process registry and routes only.
pytestmark = pytest.mark.local

#: Cohere model name published for each Bedrock model ID, per Cohere's platform tables.
UPSTREAM_NAMES = {
    "cohere.embed-english-v3": "embed-english-v3.0",
    "cohere.embed-multilingual-v3": "embed-multilingual-v3.0",
    "cohere.embed-v4:0": "embed-v4.0",
    "cohere.rerank-v3-5:0": "rerank-v3.5",
}

#: Bedrock IDs that must not produce a Cohere-shaped alias.
UNALIASED_IDS = (
    # Provisioned-throughput variants: their alias would collide with the base ID's.
    "cohere.embed-english-v3:0:512",
    "cohere.embed-multilingual-v3:0:512",
    # Amazon's own rerank model has no upstream Cohere name.
    "amazon.rerank-v1:0",
)


def _catalog(*model_ids: str) -> dict[str, ModelDetails]:
    """Build a stub catalog holding *model_ids*.

    Args:
        model_ids: Bedrock model IDs to include.

    Returns:
        Model details keyed by model ID.
    """
    return {model_id: make_model_details(model_id) for model_id in model_ids}


def _derived_aliases(*model_ids: str) -> dict[str, str]:
    """Derive the aliases both Cohere model classes publish for *model_ids*.

    Args:
        model_ids: Bedrock model IDs to include in the catalog.

    Returns:
        Alias to model ID.
    """
    catalog = _catalog(*model_ids)
    return CohereEmbeddingModel.get_aliases(catalog) | RerankModel.get_aliases(catalog)


@pytest.mark.parametrize(("model_id", "upstream_name"), UPSTREAM_NAMES.items())
def test_cohere_model_is_published_under_its_upstream_name(
    model_id: str, upstream_name: str
) -> None:
    """Each Cohere Bedrock model publishes the name Cohere's own API accepts.

    Cohere spells the version with a dot and two parts (``v3.0``, ``v3.5``,
    ``v4.0``) where the Bedrock ID uses dashes and drops the minor part, so the
    alias is only useful if that suffix is rewritten.

    Ref: https://docs.cohere.com/docs/models
         stdapi/models/_cohere.py:COHERE_ALIAS_SUBSTITUTIONS
    """
    assert _derived_aliases(*UPSTREAM_NAMES)[upstream_name] == model_id


def test_no_alias_is_derived_beyond_the_upstream_names() -> None:
    """The Cohere classes publish those names and nothing else.

    Ref: stdapi/models/__init__.py:ModelBase.get_aliases
    """
    assert set(_derived_aliases(*UPSTREAM_NAMES, *UNALIASED_IDS)) == set(
        UPSTREAM_NAMES.values()
    )


@pytest.mark.parametrize("model_id", UNALIASED_IDS)
def test_unaliased_id_keeps_only_its_bedrock_name(model_id: str) -> None:
    """A provisioned variant or an Amazon rerank model publishes no Cohere alias.

    ``cohere.embed-english-v3:0:512`` is the same model as
    ``cohere.embed-english-v3`` under a provisioned-throughput ID, so an alias
    for it would silently shadow the on-demand one.

    Ref: stdapi/models/_cohere.py:COHERE_EMBED_ALIAS_MATCHER
         stdapi/models/_cohere.py:COHERE_RERANK_ALIAS_MATCHER
    """
    assert model_id not in _derived_aliases(model_id).values()


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("cohere.rerank-v4-fast:0", "rerank-v4.0-fast"),
        ("cohere.rerank-v4-pro:0", "rerank-v4.0-pro"),
        ("cohere.embed-v5:0", "embed-v5.0"),
    ],
)
def test_alias_rule_extends_to_unreleased_bedrock_ids(
    model_id: str, expected: str
) -> None:
    """The rule keeps naming models Bedrock has not published yet.

    ``rerank-v4.0-fast`` and ``rerank-v4.0-pro`` are names Cohere already
    publishes on other platforms; deriving them from the Bedrock ID shape they
    would take proves the alias is a rule rather than a hand-kept table.

    Ref: https://docs.cohere.com/docs/models
         stdapi/models/_cohere.py:COHERE_ALIAS_SUBSTITUTIONS
    """
    assert _derived_aliases(model_id)[expected] == model_id


def test_cohere_aliases_collide_with_no_other_model_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No other family claims a name the Cohere classes publish.

    ``MODEL_ALIASES`` is one flat table, so a name two classes derive resolves
    to whichever ran last; this asserts the Cohere names are unclaimed.

    Ref: stdapi/models/__init__.py:_populate_model_aliases
    """
    catalog = _catalog(
        *UPSTREAM_NAMES,
        *UNALIASED_IDS,
        "anthropic.claude-opus-5-20260115-v1:0",
        "openai.gpt-oss-120b-1:0",
        "amazon.nova-micro-v1:0",
    )
    assert models._GLOBAL_MODEL_REGISTRY  # noqa: SLF001
    others = [
        alias
        for cls in models._GLOBAL_MODEL_REGISTRY  # noqa: SLF001
        if cls not in (CohereEmbeddingModel, RerankModel)
        for alias in cls.get_aliases(catalog)
    ]
    monkeypatch.setattr(models, "MODEL_ALIASES", {})
    monkeypatch.setattr(models, "MODEL_ALIAS_OVERLAYS", {})
    models._populate_model_aliases(catalog)  # noqa: SLF001

    assert not set(UPSTREAM_NAMES.values()) & set(others)
    for model_id, upstream_name in UPSTREAM_NAMES.items():
        assert models.MODEL_ALIASES[upstream_name] == model_id
        assert upstream_name in (catalog[model_id].aliases or ())


def test_operator_alias_overrides_a_derived_cohere_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ``MODEL_ALIASES`` entry wins over the name derived from the ID.

    Ref: stdapi/models/__init__.py:_populate_model_aliases
    """
    catalog = _catalog(*UPSTREAM_NAMES)
    monkeypatch.setattr(models, "MODEL_ALIASES", {})
    monkeypatch.setattr(models, "MODEL_ALIAS_OVERLAYS", {})
    monkeypatch.setattr(
        SETTINGS, "model_aliases", {"embed-v4.0": "cohere.embed-english-v3"}
    )
    models._populate_model_aliases(catalog)  # noqa: SLF001

    assert models.MODEL_ALIASES["embed-v4.0"] == "cohere.embed-english-v3"


@pytest.mark.parametrize(("model_id", "upstream_name"), UPSTREAM_NAMES.items())
def test_alias_and_bedrock_id_resolve_to_the_same_backend(
    monkeypatch: pytest.MonkeyPatch, model_id: str, upstream_name: str
) -> None:
    """Naming the upstream name reaches the very model the Bedrock ID reaches.

    Ref: stdapi/models/__init__.py:resolve_model_alias
    """
    monkeypatch.setitem(MODEL_ALIASES, upstream_name, model_id)
    get_model = get_rerank_model if "rerank" in model_id else get_embedding_model

    assert resolve_model_alias(upstream_name) == model_id
    assert get_model(resolve_model_alias(upstream_name)) is get_model(model_id)


@pytest.mark.usefixtures("request_log")
def test_embed_route_accepts_the_upstream_cohere_name(
    monkeypatch: pytest.MonkeyPatch, app_client: TestClient
) -> None:
    """POST /cohere/v2/embed named ``embed-english-v3.0`` embeds with the Bedrock model.

    Ref: https://docs.cohere.com/reference/embed
         stdapi/routes/cohere_embed.py:embed
    """
    reached = _capture_backend(monkeypatch, cohere_embed, "get_embedding_model")
    monkeypatch.setitem(
        models._MODELS,  # noqa: SLF001
        "cohere.embed-english-v3",
        make_model_details("cohere.embed-english-v3", output_modalities=["EMBEDDING"]),
    )
    monkeypatch.setitem(MODEL_ALIASES, "embed-english-v3.0", "cohere.embed-english-v3")

    response = app_client.post(
        "/cohere/v2/embed",
        json={
            "model": "embed-english-v3.0",
            "input_type": "search_document",
            "texts": ["hello"],
        },
    )

    assert response.status_code == 200
    assert reached == ["cohere.embed-english-v3"]


@pytest.mark.usefixtures("request_log")
def test_rerank_route_accepts_the_upstream_cohere_name(
    monkeypatch: pytest.MonkeyPatch, app_client: TestClient
) -> None:
    """POST /cohere/v2/rerank named ``rerank-v3.5`` reranks with the Bedrock model.

    Ref: https://docs.cohere.com/reference/rerank
         stdapi/routes/cohere_rerank.py:rerank
    """
    reached = _capture_backend(monkeypatch, cohere_rerank, "get_rerank_model")
    monkeypatch.setitem(
        models._MODELS,  # noqa: SLF001
        "cohere.rerank-v3-5:0",
        make_model_details("cohere.rerank-v3-5:0", output_modalities=["RERANKING"]),
    )
    monkeypatch.setitem(MODEL_ALIASES, "rerank-v3.5", "cohere.rerank-v3-5:0")

    response = app_client.post(
        "/cohere/v2/rerank",
        json={"model": "rerank-v3.5", "query": "q", "documents": ["a", "b"]},
    )

    assert response.status_code == 200
    assert reached == ["cohere.rerank-v3-5:0"]


class _StubBackend:
    """Embedding and rerank backend answering without reaching Bedrock."""

    async def embed_text(
        self,
        inputs: list[InputFileUrl | str],
        dimensions: int | None,
        extra_params: JsonMapping,
    ) -> EmbeddingResponse:
        """Return one canned vector per input.

        Args:
            inputs: Texts and images to embed.
            dimensions: Ignored number of dimensions.
            extra_params: Ignored extra model parameters.

        Returns:
            The canned embedding response.
        """
        return EmbeddingResponse(embeddings=[[0.1, 0.2]] * len(inputs))

    async def rerank(
        self,
        query: str,
        documents: list[str | JsonMapping],
        *,
        top_n: int | None,
        extra_params: JsonMapping,
    ) -> RerankResponse:
        """Return the documents in their submitted order.

        Args:
            query: Ignored search query.
            documents: Documents to rank.
            top_n: Ignored result count.
            extra_params: Ignored extra model parameters.

        Returns:
            The canned rerank response.
        """
        return RerankResponse(
            results=[
                RerankedDocument(index=index, relevance_score=1.0)
                for index in range(len(documents))
            ],
            search_units=1,
        )


def _capture_backend(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, getter: str
) -> list[str]:
    """Replace *module*'s model getter with a stub recording the model ID it got.

    Args:
        monkeypatch: Patching fixture.
        module: Route module holding the getter.
        getter: Name of the model getter to replace.

    Returns:
        The list the stub appends each requested model ID to.
    """
    reached: list[str] = []

    def _get(model_id: str) -> _StubBackend:
        reached.append(model_id)
        return _StubBackend()

    monkeypatch.setattr(module, getter, _get)
    return reached
