"""Default Bedrock Mantle routing for the OpenAI GPT-6 family.

``aws_bedrock_mantle_preferred_models`` defaults to the GPT-6 family as well as
GPT-5.6, because Amazon Bedrock serves their web search and code interpreter
on Mantle alone: a request asking for one reaches Mantle with no header and no
configuration. Mantle serves the family in far fewer Regions than
bedrock-runtime (GPT-6 Luna and Sol in ``us-east-1``, Astra in ``us-west-2``,
listed live 2026-09-22), so the preference only moves a model where a
configured Mantle Region lists it, and only ever sends it to those Regions.

Ref: stdapi/config.py:DEFAULT_MANTLE_PREFERRED_MODELS
     stdapi/models/__init__.py:_merge_mantle_models
     stdapi/models/chat/__init__.py:serves_via_mantle
     stdapi/models/chat/_mantle/_default.py:ChatModel._mantle_regions
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-astra.html
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from starlette.requests import Request

from stdapi import models
from stdapi.api_errors import ApiError
from stdapi.config import DEFAULT_MANTLE_PREFERRED_MODELS, SETTINGS
from stdapi.models import (
    MANTLE_MODELS,
    MANTLE_SERVICE,
    RUNTIME_SERVICE,
    _collect_mantle_models,
    _merge_mantle_models,
    is_mantle_preferred,
)
from stdapi.models.chat import get_chat_model, serves_via_mantle
from stdapi.models.chat._mantle.openai_gpt6 import ChatModel as MantleGpt6ChatModel
from stdapi.models.chat.openai_gpt import ChatModel as RuntimeGptChatModel
from stdapi.monitoring import REQUEST
from tests._helpers import make_model_details

if TYPE_CHECKING:
    from collections.abc import Iterator

    from types_aiobotocore_bedrock.literals import RegionName

    from stdapi.models import ModelDetails

#: The catalogue and Mantle registry these tests rewrite are shared with tests/test_model_cache.py.
pytestmark = [pytest.mark.local, pytest.mark.xdist_group("model_cache")]

#: The GPT-6 models AWS lists on both endpoints.
_GPT6 = ("openai.gpt-6-astra", "openai.gpt-6-luna", "openai.gpt-6-sol")

#: Mantle Regions a deployment may configure; only the first two serve GPT-6.
_MANTLE_REGIONS: list[RegionName] = ["us-east-1", "us-west-2", "eu-west-1"]

#: Each Region's Mantle ``/v1/models`` GPT-6 listing, as observed live on 2026-09-22.
_MANTLE_LISTINGS: dict[str, list[str]] = {
    "us-east-1": ["openai.gpt-6-luna", "openai.gpt-6-sol"],
    "us-west-2": ["openai.gpt-6-astra"],
    "eu-west-1": [],
}

#: Every Region bedrock-runtime serves the GPT-6 models from in this fixture.
_RUNTIME_REGIONS: list[RegionName] = ["us-east-1", "us-west-2", "eu-west-1"]


@pytest.fixture
def mantle_catalog(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Serve the GPT-6 Mantle listings above, under the shipped default preference.

    Yields:
        Nothing; the Mantle catalogue is fetched through the real collector,
        with only the HTTP call replaced, and every registry the tests rewrite
        is restored afterwards.
    """

    async def listing(region: RegionName, method: str, path: str) -> dict[str, Any]:
        assert (method, path) == ("GET", "/v1/models")
        return {"data": [{"id": model_id} for model_id in _MANTLE_LISTINGS[region]]}

    monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_regions", _MANTLE_REGIONS)
    monkeypatch.setattr(
        SETTINGS,
        "aws_bedrock_mantle_preferred_models",
        [*DEFAULT_MANTLE_PREFERRED_MODELS],
    )
    monkeypatch.setattr(models, "mantle_request_json", listing)
    saved_mantle = dict(MANTLE_MODELS)
    saved_displaced = dict(models._DISPLACED_RUNTIME_MODELS)  # noqa: SLF001
    saved_models = dict(models._MODELS)  # noqa: SLF001
    try:
        yield
    finally:
        MANTLE_MODELS.clear()
        MANTLE_MODELS.update(saved_mantle)
        models._DISPLACED_RUNTIME_MODELS.clear()  # noqa: SLF001
        models._DISPLACED_RUNTIME_MODELS.update(saved_displaced)  # noqa: SLF001
        models._MODELS.clear()  # noqa: SLF001
        models._MODELS.update(saved_models)  # noqa: SLF001
        models.update_unified_models_collections()


def _runtime_catalogue() -> dict[str, ModelDetails]:
    """Build the bedrock-runtime entries of the three GPT-6 models.

    Returns:
        The catalogue, each model served in every fixture runtime Region.
    """
    return {
        model_id: make_model_details(
            model_id,
            provider="OpenAI",
            service=RUNTIME_SERVICE,
            regions=list(_RUNTIME_REGIONS),
        )
        for model_id in _GPT6
    }


async def _publish_catalogue() -> dict[str, ModelDetails]:
    """Build and publish the catalogue the way a sweep does, from the fixture listings.

    Returns:
        The merged catalogue.
    """
    catalogue = _runtime_catalogue()
    _merge_mantle_models(catalogue, await _collect_mantle_models({}, {}))
    models._MODELS.clear()  # noqa: SLF001
    models._MODELS.update(catalogue)  # noqa: SLF001
    models.update_unified_models_collections()
    return catalogue


class TestGpt6PrefersMantleByDefault:
    """With no configuration, GPT-6 is served where its server tools run."""

    @pytest.mark.parametrize("model_id", [*_GPT6, "openai.gpt-6.5-nova"])
    def test_the_default_preference_covers_the_family(self, model_id: str) -> None:
        """Every GPT-6 ID, current or future, matches the shipped default.

        The entries are prefixes, so a member AWS adds to the family later is
        covered without a release.

        Ref: stdapi/models/__init__.py:is_mantle_preferred
        """
        assert any(
            model_id.startswith(entry) for entry in DEFAULT_MANTLE_PREFERRED_MODELS
        )

    @pytest.mark.usefixtures("mantle_catalog")
    def test_the_setting_resolves_the_family_as_preferred(self) -> None:
        """``is_mantle_preferred`` answers True for the family under the default.

        Ref: stdapi/models/__init__.py:is_mantle_preferred
        """
        assert all(is_mantle_preferred(model_id) for model_id in _GPT6)
        assert not is_mantle_preferred("openai.gpt-5.5")

    @pytest.mark.usefixtures("mantle_catalog")
    async def test_a_web_search_request_is_served_by_mantle_without_a_header(
        self,
    ) -> None:
        """A GPT-6 request resolves to the Mantle class, sent only to Regions listing it.

        No ``x-stdapi-service`` header is bound: the default preference alone
        moves the model, so a ``web_search`` or ``code_interpreter`` tool
        reaches the endpoint that serves it. The deployment configures three
        Mantle Regions; the model's own Mantle Regions are the whole candidate
        list.

        Ref: stdapi/models/chat/__init__.py:get_chat_model
             stdapi/models/chat/_mantle/_default.py:ChatModel._mantle_regions
        """
        catalogue = await _publish_catalogue()
        assert REQUEST.get(None) is None
        for model_id, served_in in (
            ("openai.gpt-6-luna", ["us-east-1"]),
            ("openai.gpt-6-sol", ["us-east-1"]),
            ("openai.gpt-6-astra", ["us-west-2"]),
        ):
            assert catalogue[model_id].service == MANTLE_SERVICE
            assert catalogue[model_id].regions == served_in
            model = get_chat_model(model_id)
            assert type(model) is MantleGpt6ChatModel
            assert model._mantle_regions(None) == served_in  # noqa: SLF001

    @pytest.mark.usefixtures("mantle_catalog")
    async def test_where_no_mantle_region_lists_it_the_clean_400_stays(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deployment whose Mantle Regions lack GPT-6 keeps it on bedrock-runtime.

        Configuring ``eu-west-1`` alone still serves GPT-6 through
        bedrock-runtime, in every Region it has there, rather than sending it
        to a Region without the model; a server tool is then refused with a
        ``400`` naming no backend.

        Ref: stdapi/models/__init__.py:_merge_mantle_models
             stdapi/models/chat/openai_gpt.py:ChatModel.SERVER_TOOLS_UNSERVED
             stdapi/models/chat/_adapters/_common.py:NoServerTools.refuse
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_regions", ["eu-west-1"])
        catalogue = await _publish_catalogue()
        assert catalogue["openai.gpt-6-luna"].service == RUNTIME_SERVICE
        assert catalogue["openai.gpt-6-luna"].regions == _RUNTIME_REGIONS
        assert serves_via_mantle("openai.gpt-6-luna") is False
        model = get_chat_model("openai.gpt-6-luna")
        assert type(model) is RuntimeGptChatModel
        assert model.SERVER_TOOLS_UNSERVED is not None
        with pytest.raises(ApiError) as refusal:
            model.SERVER_TOOLS_UNSERVED.refuse("web_search")
        assert refusal.value.status == 400
        assert "Mantle" not in str(refusal.value)


@pytest.mark.usefixtures("mantle_catalog")
class TestGpt6ServiceHeader:
    """A deployment that cleared the preference can still route one request."""

    @pytest.mark.parametrize(
        ("model_id", "served_in"),
        [("openai.gpt-6-luna", ["us-east-1"]), ("openai.gpt-6-astra", ["us-west-2"])],
    )
    async def test_the_header_sends_the_request_to_a_serving_region(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model_id: str,
        served_in: list[RegionName],
    ) -> None:
        """With the preference emptied, the header still reaches Mantle where it lists the model.

        Ref: stdapi/models/chat/__init__.py:serves_via_mantle
             stdapi/models/chat/_mantle/_default.py:ChatModel._mantle_regions
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_preferred_models", [])
        monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_service_header", True)
        catalogue = await _publish_catalogue()
        assert catalogue[model_id].service == RUNTIME_SERVICE
        token = REQUEST.set(
            Request(
                {"type": "http", "headers": [(b"x-stdapi-service", b"bedrock-mantle")]}
            )
        )
        try:
            model = get_chat_model(model_id)
        finally:
            REQUEST.reset(token)
        assert type(model) is MantleGpt6ChatModel
        assert model._mantle_regions(None) == served_in  # noqa: SLF001
