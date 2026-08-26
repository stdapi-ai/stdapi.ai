"""Wildcard model names, resolved to the most recently released matching model.

A caller names a family (``claude-sonnet-*``, ``amazon.nova-*``) instead of a
model, and the request runs on the newest member of it the endpoint can serve.
Selection is by release date -- the same date the models endpoint publishes as
``created`` -- so it never reads a version fragment, and it refuses rather than
guesses when the date does not single one model out. Legacy, deprecated,
undated and never-subscribed models are never selected.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html
     https://stdapi.ai/operations_configuration/#alias-resolution
     stdapi/models/__init__.py:validate_model
     stdapi/models/__init__.py:_resolve_model_wildcard
     stdapi/models/__init__.py:match_model_names
"""

from datetime import UTC, datetime
from pathlib import Path
from re import compile as compile_regex
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel, ValidationError

import stdapi.main  # noqa: F401  (loads the model classes and route capabilities)
from stdapi import models
from stdapi.api_errors import AmbiguousModelError, ApiError, UnsupportedModelError
from stdapi.config import SETTINGS
from stdapi.models import ModelDetails, validate_model
from stdapi.models.capabilities import ROUTE_CAPABILITIES
from stdapi.models.deprecation import DEPRECATED_MODELS
from stdapi.monitoring import TENANT, Tenant
from stdapi.types.anthropic_messages import (
    MessageCountTokensParams,
    MessageCreateParams,
)
from stdapi.types.openai_audio import SpeechCreateParams, TranscriptionCreateParams
from stdapi.types.openai_chat_completions import CompletionCreateParams
from stdapi.types.openai_responses import ResponseCreateParams
from tests._helpers import make_model_details

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from starlette.testclient import TestClient

#: Resolution reads this server's own catalogue, so it is exercised in-process.
pytestmark = pytest.mark.local

#: Chat operation the scoped resolutions here are made against.
_CHAT_ROUTE = "openai_chat_completion"

#: Embedding operation used as the second scope of the route-scoping test.
_EMBEDDING_ROUTE = "openai_embedding"

#: Release date of the older generation of every seeded family.
_OLDER = datetime(2025, 3, 4, 17, 0, tzinfo=UTC)

#: Release date of the newer generation of every seeded family.
_NEWER = datetime(2026, 5, 6, 17, 0, tzinfo=UTC)


def _seed(**models_by_id: ModelDetails) -> dict[str, ModelDetails]:
    """Return a catalogue keyed by model ID, from keyword-named details.

    Args:
        **models_by_id: The models, named freely; each one's own ``id`` is the key.

    Returns:
        The catalogue.
    """
    return {model.id: model for model in models_by_id.values()}


def _model(
    model_id: str,
    released: datetime | None = _OLDER,
    **overrides: Any,  # noqa: ANN401
) -> ModelDetails:
    """Build a text model carrying a release date.

    Args:
        model_id: Bedrock model identifier.
        released: Value for ``start_of_life_time``; ``None`` leaves it unknown.
        **overrides: Further ``ModelDetails`` fields.

    Returns:
        The stub model.
    """
    return make_model_details(model_id, start_of_life_time=released, **overrides)


@pytest.fixture
def catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[Callable[[dict[str, ModelDetails]], None]]:
    """Serve a seeded catalogue, and put the real one back afterwards.

    The live catalogue changes under a test that asserts which model is the
    newest, so every case here seeds its own. Refreshing is disabled with it:
    a miss must answer from what was seeded rather than reach AWS. The
    non-Bedrock catalogue (``EXTRA_MODELS``, e.g. Polly/Transcribe registered
    at lifespan startup) is cleared as well, so ``_ALL_MODELS`` -- which
    ``search_models``/``/v1/models`` read -- holds exactly what was seeded,
    whether or not a lifespan ran earlier in this worker. The model-cache
    generation is bumped on install and on restore, so the ``/v1/models``
    payload the route caches for its own generation is rebuilt from what is
    actually seeded rather than served stale.

    Yields:
        Callable installing a catalogue and rebuilding every derived index.
    """
    saved = (
        dict(models._MODELS),  # noqa: SLF001
        dict(models._MODELS_OUTPUT_MODALITY),  # noqa: SLF001
        dict(models._MODELS_INPUT_MODALITY),  # noqa: SLF001
        dict(models.MODEL_ALIASES),
        dict(models.EXTRA_MODELS),
        dict(models.EXTRA_MODELS_OUTPUT_MODALITY),
        dict(models.EXTRA_MODELS_INPUT_MODALITY),
    )

    async def _no_refresh(*_args: object, **_kwargs: object) -> None:
        """Answer from the seeded catalogue instead of sweeping AWS."""

    monkeypatch.setattr(models, "refresh_stale_catalog", _no_refresh)
    monkeypatch.setattr(models, "_refresh_due", lambda: False)

    def install(catalog: dict[str, ModelDetails]) -> None:
        """Make *catalog* the whole catalogue and rebuild the indexes."""
        models._MODELS.clear()  # noqa: SLF001
        models._MODELS.update(catalog)  # noqa: SLF001
        models.EXTRA_MODELS.clear()
        models.EXTRA_MODELS_OUTPUT_MODALITY.clear()
        models.EXTRA_MODELS_INPUT_MODALITY.clear()
        for index, attribute in (
            (models._MODELS_OUTPUT_MODALITY, "output_modalities"),  # noqa: SLF001
            (models._MODELS_INPUT_MODALITY, "input_modalities"),  # noqa: SLF001
        ):
            index.clear()
            for model_id, model in catalog.items():
                for modality in getattr(model, attribute):
                    index.setdefault(modality, set()).add(model_id)
        models.MODEL_ALIASES.clear()
        models.update_unified_models_collections()
        models._CACHE["generation"] += 1  # noqa: SLF001

    yield install

    models._MODELS.clear()  # noqa: SLF001
    models._MODELS.update(saved[0])  # noqa: SLF001
    models._MODELS_OUTPUT_MODALITY.clear()  # noqa: SLF001
    models._MODELS_OUTPUT_MODALITY.update(saved[1])  # noqa: SLF001
    models._MODELS_INPUT_MODALITY.clear()  # noqa: SLF001
    models._MODELS_INPUT_MODALITY.update(saved[2])  # noqa: SLF001
    models.MODEL_ALIASES.clear()
    models.MODEL_ALIASES.update(saved[3])
    models.EXTRA_MODELS.clear()
    models.EXTRA_MODELS.update(saved[4])
    models.EXTRA_MODELS_OUTPUT_MODALITY.clear()
    models.EXTRA_MODELS_OUTPUT_MODALITY.update(saved[5])
    models.EXTRA_MODELS_INPUT_MODALITY.clear()
    models.EXTRA_MODELS_INPUT_MODALITY.update(saved[6])
    models.update_unified_models_collections()
    models._CACHE["generation"] += 1  # noqa: SLF001


class TestSelection:
    """A pattern selects the newest model it matches, on the endpoint being called.

    Both name spaces are searched: families are written as model IDs
    (``amazon.nova-*``) as often as they are written as aliases
    (``claude-sonnet-*``), and restricting to one would break half the patterns
    an operator would try.

    Ref: stdapi/models/__init__.py:_resolve_model_wildcard
         stdapi/models/__init__.py:match_model_names
    """

    async def test_alias_space_pattern_selects_the_newest(
        self,
        catalog: Callable[[dict[str, ModelDetails]], None],
        request_log: dict[str, Any],
    ) -> None:
        """``claude-sonnet-*`` matches no model ID, only the aliases, and still resolves."""
        catalog(
            _seed(
                old=_model(
                    "anthropic.claude-sonnet-9-v1:0", _OLDER, provider="Anthropic"
                ),
                new=_model(
                    "anthropic.claude-sonnet-10-v1:0", _NEWER, provider="Anthropic"
                ),
            )
        )
        resolved = await validate_model("claude-sonnet-*", route=_CHAT_ROUTE)
        assert resolved.id == "anthropic.claude-sonnet-10-v1:0"
        assert request_log["model_id"] == "anthropic.claude-sonnet-10-v1:0"

    async def test_id_space_pattern_selects_the_newest(
        self,
        catalog: Callable[[dict[str, ModelDetails]], None],
        request_log: dict[str, Any],
    ) -> None:
        """``amazon.nova-*`` matches model IDs, which carry no alias here."""
        catalog(
            _seed(
                old=_model("amazon.nova-9-lite-v1:0", _OLDER, provider="Amazon"),
                new=_model("amazon.nova-10-lite-v1:0", _NEWER, provider="Amazon"),
            )
        )
        assert (await validate_model("amazon.nova-*", route=_CHAT_ROUTE)).id == (
            "amazon.nova-10-lite-v1:0"
        )
        assert request_log["model_id"] == "amazon.nova-10-lite-v1:0"

    async def test_exact_name_is_never_overridden_by_a_pattern(
        self,
        catalog: Callable[[dict[str, ModelDetails]], None],
        request_log: dict[str, Any],
    ) -> None:
        """Naming the older model keeps it, even though a pattern would take the newer."""
        catalog(
            _seed(
                old=_model("amazon.nova-9-lite-v1:0", _OLDER, provider="Amazon"),
                new=_model("amazon.nova-10-lite-v1:0", _NEWER, provider="Amazon"),
            )
        )
        resolved = await validate_model("amazon.nova-9-lite-v1:0", route=_CHAT_ROUTE)
        assert resolved.id == "amazon.nova-9-lite-v1:0"
        assert request_log["model_id"] == "amazon.nova-9-lite-v1:0"

    async def test_a_name_carrying_wildcard_characters_is_still_looked_up_exactly(
        self,
        catalog: Callable[[dict[str, ModelDetails]], None],
        request_log: dict[str, Any],
    ) -> None:
        """An exact dict entry wins even when its own name is a valid pattern.

        The name that names *this test* asserts on -- "an exact name is never
        overridden by a pattern" -- can only be violated by a request string
        that both carries a wildcard character (so ``is_model_wildcard`` is
        True) and is an exact key in the catalogue. A real Bedrock ID never
        contains ``*``/``?``, so ``test_exact_name_is_never_overridden_by_a_pattern``
        never reaches ``_resolve_model_wildcard`` at all -- deleting it would
        not fail that test. This one does reach it: read as a pattern, the
        exact name here also matches a second, newer candidate, so an exact
        check that ran behind the wildcard check would return that newer
        model instead of the one actually named.
        """
        catalog(
            _seed(
                exact=_model("vendor.family-*-v1:0", _OLDER),
                other=_model("vendor.family-99-v1:0", _NEWER),
            )
        )
        resolved = await validate_model("vendor.family-*-v1:0", route=_CHAT_ROUTE)
        assert resolved.id == "vendor.family-*-v1:0"
        assert request_log["model_id"] == "vendor.family-*-v1:0"

    async def test_an_alias_named_like_a_pattern_wins(
        self,
        catalog: Callable[[dict[str, ModelDetails]], None],
        monkeypatch: pytest.MonkeyPatch,
        request_log: dict[str, Any],
    ) -> None:
        """An operator alias literally named ``amazon.nova-*`` is an alias, not a pattern.

        It is an explicit instruction from the operator, so it resolves before
        anything is matched.
        """
        catalog(
            _seed(
                old=_model("amazon.nova-9-lite-v1:0", _OLDER, provider="Amazon"),
                new=_model("amazon.nova-10-lite-v1:0", _NEWER, provider="Amazon"),
            )
        )
        monkeypatch.setitem(
            models.MODEL_ALIASES, "amazon.nova-*", "amazon.nova-9-lite-v1:0"
        )
        assert (await validate_model("amazon.nova-*", route=_CHAT_ROUTE)).id == (
            "amazon.nova-9-lite-v1:0"
        )
        assert request_log["model_id"] == "amazon.nova-9-lite-v1:0"

    async def test_pattern_is_scoped_to_the_route(
        self,
        catalog: Callable[[dict[str, ModelDetails]], None],
        request_log: dict[str, Any],
    ) -> None:
        """One pattern, two endpoints, two models -- and neither one is ambiguous.

        Unscoped the two are tied on their release date, which is why the scope
        is part of the selection rather than an optimisation of it.
        """
        catalog(
            _seed(
                chat=_model("vendor.dual-chat-v1:0", _NEWER),
                embed=_model(
                    "vendor.dual-embed-v1:0", _NEWER, output_modalities=["EMBEDDING"]
                ),
            )
        )
        assert (await validate_model("vendor.dual-*", route=_CHAT_ROUTE)).id == (
            "vendor.dual-chat-v1:0"
        )
        assert (
            await validate_model("vendor.dual-*", "EMBEDDING", route=_EMBEDDING_ROUTE)
        ).id == "vendor.dual-embed-v1:0"
        assert request_log["model_id"] == "vendor.dual-embed-v1:0"


class TestTenantScope:
    """A pattern resolves within the calling tenant's model scope, not around it.

    Narrowing the candidate set by tenant scope during selection, instead of
    only after a model is chosen, is what lets a pattern select the newest
    model the tenant may actually use rather than the newest model in the
    whole catalogue -- which the tenant's key may not be scoped to at all.

    Ref: stdapi/models/__init__.py:_resolve_model_wildcard
         stdapi/monitoring.py:Tenant.allows_model
    """

    async def test_pattern_selects_the_newest_model_within_scope(
        self,
        catalog: Callable[[dict[str, ModelDetails]], None],
        request_log: dict[str, Any],
    ) -> None:
        """The newest overall match is out of scope, so the pattern takes the in-scope one."""
        catalog(
            _seed(
                in_scope=_model("vendor.family-9-v1:0", _OLDER),
                out_of_scope=_model("vendor.family-10-v1:0", _NEWER),
            )
        )
        token = TENANT.set(
            Tenant(
                key_id="tk_scoped", name="scoped", models_allow=("vendor.family-9-*",)
            )
        )
        try:
            resolved = await validate_model("vendor.family-*", route=_CHAT_ROUTE)
        finally:
            TENANT.reset(token)
        assert resolved.id == "vendor.family-9-v1:0"

    async def test_pattern_matching_only_out_of_scope_models_names_the_pattern(
        self, catalog: Callable[[dict[str, ModelDetails]], None]
    ) -> None:
        """No in-scope match reads as no such model, naming the pattern the caller sent.

        Not the model the pattern would have selected outside the tenant's
        scope: that model was never a candidate, so it is never named back.
        """
        catalog(_seed(out_of_scope=_model("vendor.family-10-v1:0", _NEWER)))
        token = TENANT.set(
            Tenant(key_id="tk_scoped", name="scoped", models_allow=("vendor.other-*",))
        )
        try:
            with pytest.raises(UnsupportedModelError) as raised:
                await validate_model("vendor.family-*", route=_CHAT_ROUTE)
        finally:
            TENANT.reset(token)
        assert raised.value.status == 404
        assert raised.value.code == "model_not_found"
        assert "vendor.family-*" in str(raised.value)
        assert "vendor.family-10-v1:0" not in str(raised.value)


class TestExcludedFromSelection:
    """Four kinds of model a pattern never selects, each for its own reason.

    The legacy one is the load-bearing case: legacy models stay in the catalogue
    when ``aws_bedrock_legacy`` is set, so the selection excludes them itself
    instead of assuming the catalogue already did.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html
         stdapi/models/__init__.py:_resolve_model_wildcard
         stdapi/models/deprecation.py:DEPRECATED_MODELS
    """

    async def test_legacy_model_never_wins_with_legacy_models_enabled(
        self,
        catalog: Callable[[dict[str, ModelDetails]], None],
        monkeypatch: pytest.MonkeyPatch,
        request_log: dict[str, Any],
    ) -> None:
        """The newest match is legacy, so the pattern takes the one behind it.

        Seeding a legacy model *is* what ``aws_bedrock_legacy`` produces -- it is
        the setting that keeps one in the catalogue -- and the setting is set
        here too, because without it this case cannot happen at all and the
        exclusion would look untested rather than unreachable.
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_legacy", True)
        catalog(
            _seed(
                supported=_model("amazon.nova-9-lite-v1:0", _OLDER, provider="Amazon"),
                legacy=_model(
                    "amazon.nova-10-lite-v1:0", _NEWER, provider="Amazon", legacy=True
                ),
            )
        )
        assert (await validate_model("amazon.nova-*", route=_CHAT_ROUTE)).id == (
            "amazon.nova-9-lite-v1:0"
        )

    async def test_deprecated_model_is_skipped_rather_than_replaced(
        self,
        catalog: Callable[[dict[str, ModelDetails]], None],
        monkeypatch: pytest.MonkeyPatch,
        request_log: dict[str, Any],
    ) -> None:
        """A pattern landing on a deprecated model takes the next match, not the successor.

        Following the replacement would leave the pattern selecting a model
        outside its own match set.
        """
        catalog(
            _seed(
                supported=_model("vendor.family-9-v1:0", _OLDER),
                deprecated=_model("vendor.family-10-v1:0", _NEWER),
                successor=_model("other.successor-v1:0", _NEWER),
            )
        )
        monkeypatch.setitem(
            DEPRECATED_MODELS, "vendor.family-10-v1:0", "other.successor-v1:0"
        )
        assert (await validate_model("vendor.family-*", route=_CHAT_ROUTE)).id == (
            "vendor.family-9-v1:0"
        )

    async def test_undated_model_neither_wins_nor_creates_ambiguity(
        self,
        catalog: Callable[[dict[str, ModelDetails]], None],
        request_log: dict[str, Any],
    ) -> None:
        """A model with no known release date cannot be ordered, so it is not a candidate.

        It does not make the remaining unique match ambiguous either: it is
        absent from the comparison, not tied inside it.
        """
        catalog(
            _seed(
                dated=_model("vendor.family-9-v1:0", _OLDER),
                undated=_model("vendor.family-10-v1:0", None),
            )
        )
        assert (await validate_model("vendor.family-*", route=_CHAT_ROUTE)).id == (
            "vendor.family-9-v1:0"
        )

    async def test_model_pending_a_paid_subscription_never_wins(
        self,
        catalog: Callable[[dict[str, ModelDetails]], None],
        request_log: dict[str, Any],
    ) -> None:
        """A model whose first request would open a paid subscription must be named.

        Selecting it from a pattern would start a subscription the operator
        never asked for.
        """
        catalog(
            _seed(
                subscribed=_model("vendor.family-9-v1:0", _OLDER),
                unsubscribed=_model(
                    "vendor.family-10-v1:0", _NEWER, pending_subscription=True
                ),
            )
        )
        assert (await validate_model("vendor.family-*", route=_CHAT_ROUTE)).id == (
            "vendor.family-9-v1:0"
        )
        # Naming it explicitly still works: the operator chose it.
        assert (
            await validate_model("vendor.family-10-v1:0", route=_CHAT_ROUTE)
        ).id == "vendor.family-10-v1:0"

    async def test_every_match_excluded_reads_as_no_such_model(
        self, catalog: Callable[[dict[str, ModelDetails]], None]
    ) -> None:
        """A pattern matching only unselectable models is answered as a missing model."""
        catalog(_seed(undated=_model("vendor.family-10-v1:0", None)))
        with pytest.raises(UnsupportedModelError) as raised:
            await validate_model("vendor.family-*", route=_CHAT_ROUTE)
        assert raised.value.status == 404
        assert raised.value.code == "model_not_found"
        assert "vendor.family-*" in str(raised.value)

    async def test_a_pattern_never_resolves_on_the_speech_route(
        self, catalog: Callable[[dict[str, ModelDetails]], None]
    ) -> None:
        """A pattern is accepted but never matches on `/v1/audio/speech`.

        Every real Polly entry is registered with no release date, and Polly
        is the only class serving that route, so its whole candidate set is
        excluded from selection: `SpeechCreateParams.model` must not promise a
        pattern resolves there.

        Ref: stdapi/models/audio/amazon_polly.py:309 (Polly entries carry no
             start_of_life_time)
             stdapi/types/openai_audio.py:SpeechCreateParams.model
        """
        catalog(
            _seed(
                only=_model("amazon.polly-standard", None, output_modalities=["SPEECH"])
            )
        )
        with pytest.raises(UnsupportedModelError) as raised:
            await validate_model(
                "amazon.polly-*", "SPEECH", route="openai_audio_speech"
            )
        assert raised.value.status == 404
        assert raised.value.code == "model_not_found"


class TestRefusals:
    """A pattern that does not name one model is refused, never resolved by guessing.

    Models released together are priced differently, so breaking a tie would
    spend the operator's money on a model nobody named.

    Ref: stdapi/api_errors.py:AmbiguousModelError
         stdapi/models/__init__.py:_resolve_model_wildcard
    """

    async def test_tied_release_dates_are_refused_and_named(
        self, catalog: Callable[[dict[str, ModelDetails]], None]
    ) -> None:
        """Two matches share the newest date, so both are named back to the caller."""
        catalog(
            _seed(
                one=_model("vendor.family-10-alpha-v1:0", _NEWER),
                two=_model("vendor.family-10-beta-v1:0", _NEWER),
                old=_model("vendor.family-9-v1:0", _OLDER),
            )
        )
        with pytest.raises(AmbiguousModelError) as raised:
            await validate_model("vendor.family-*", route=_CHAT_ROUTE)
        assert raised.value.status == 400
        assert raised.value.code == "ambiguous_model"
        message = str(raised.value)
        assert "vendor.family-10-alpha-v1:0" in message
        assert "vendor.family-10-beta-v1:0" in message
        assert "vendor.family-9-v1:0" not in message

    @pytest.mark.parametrize("pattern", ["*", "?", "no*", "**"])
    async def test_pattern_naming_too_little_is_refused(
        self, catalog: Callable[[dict[str, ModelDetails]], None], pattern: str
    ) -> None:
        """A pattern with fewer than three leading literal characters selects nothing.

        ``*`` on its own would resolve to whichever model happens to be newest,
        which is a request nobody deliberately writes.
        """
        catalog(_seed(only=_model("vendor.family-9-v1:0", _OLDER)))
        with pytest.raises(ApiError) as raised:
            await validate_model(pattern, route=_CHAT_ROUTE)
        assert raised.value.status == 400
        assert "at least 3 characters" in str(raised.value)

    async def test_pattern_matching_nothing_reads_as_no_such_model(
        self, catalog: Callable[[dict[str, ModelDetails]], None]
    ) -> None:
        """An unmatched pattern is answered exactly as an unknown model name is."""
        catalog(_seed(only=_model("vendor.family-9-v1:0", _OLDER)))
        with pytest.raises(UnsupportedModelError) as raised:
            await validate_model("absent.family-*", route=_CHAT_ROUTE)
        assert raised.value.status == 404
        assert raised.value.code == "model_not_found"

    async def test_an_overlong_pattern_is_refused_before_it_is_compiled(
        self, catalog: Callable[[dict[str, ModelDetails]], None]
    ) -> None:
        """A pattern past the length limit is refused, never handed to `re.compile`.

        Regression test for a request-path DoS: an unbounded pattern used to
        reach `fnmatch.translate`/`re.compile` synchronously, on the event
        loop, while holding the model-cache lock.
        """
        catalog(_seed(only=_model("vendor.family-9-v1:0", _OLDER)))
        pattern = "abc" + "*a" * 200  # 403 characters, well past the 255 limit
        with pytest.raises(ApiError) as raised:
            await validate_model(pattern, route=_CHAT_ROUTE)
        assert raised.value.status == 400
        assert "255" in str(raised.value)

    @pytest.mark.parametrize("pattern", ["[a-z]*", "[!Z]*", "vendor.[fF]amily-*"])
    async def test_character_class_is_refused_not_silently_matched(
        self, catalog: Callable[[dict[str, ModelDetails]], None], pattern: str
    ) -> None:
        """A bracket expression is refused, since only `*` and `?` are documented.

        The minimum-literals guard only counts `*`/`?`, so an unrejected class
        such as ``[!Z]*`` would pass it (its first `*` sits past the offset the
        guard checks) and then match every model on the route -- the exact
        outcome a bare `*` is refused to prevent.
        """
        catalog(_seed(only=_model("vendor.family-9-v1:0", _OLDER)))
        with pytest.raises(ApiError) as raised:
            await validate_model(pattern, route=_CHAT_ROUTE)
        assert raised.value.status == 400
        assert "[" in str(raised.value)


class TestQualifierInTheVersionPosition:
    """A qualifier written where the version belongs is skipped, and stays visible.

    ``openai.gpt-daybreak-blue-5.6-sol`` puts the edition where every sibling
    puts the version, so ``gpt-5.6-*`` does not match it -- no matcher fixes
    that. Ordering by release date rather than by version fragment keeps the
    *selection* right, and the models search shows the whole match set so the
    skip is seen rather than inferred.

    Ref: stdapi/models/__init__.py:match_model_names
         stdapi/routes/core_models.py:search_models
    """

    @staticmethod
    def _catalog() -> dict[str, ModelDetails]:
        """Return the two GPT-5.6 spellings, the irregular one dated later.

        Returns:
            The catalogue.
        """
        return _seed(
            regular=_model("openai.gpt-5.6-sol", _OLDER, provider="OpenAI"),
            irregular=_model(
                "openai.gpt-daybreak-blue-5.6-sol", _NEWER, provider="OpenAI"
            ),
        )

    async def test_the_irregular_name_is_not_matched(
        self,
        catalog: Callable[[dict[str, ModelDetails]], None],
        request_log: dict[str, Any],
    ) -> None:
        """``gpt-5.6-*`` selects the model whose name the pattern actually describes."""
        catalog(self._catalog())
        assert (await validate_model("gpt-5.6-*", route=_CHAT_ROUTE)).id == (
            "openai.gpt-5.6-sol"
        )

    def test_the_models_search_shows_both(
        self, catalog: Callable[[dict[str, ModelDetails]], None], app_client: TestClient
    ) -> None:
        """``model=gpt-*`` returns the whole match set, newest first."""
        catalog(self._catalog())
        response = app_client.get("/search_models", params={"model": "gpt-*"})
        assert response.status_code == 200
        assert [model["id"] for model in response.json()] == [
            "openai.gpt-daybreak-blue-5.6-sol",
            "openai.gpt-5.6-sol",
        ]


class TestModelsSearch:
    """``search_models`` answers what a pattern matches before a request uses it.

    Ref: https://stdapi.ai/api_search_models/
         stdapi/routes/core_models.py:search_models
    """

    def test_undated_models_are_listed_last(
        self, catalog: Callable[[dict[str, ModelDetails]], None], app_client: TestClient
    ) -> None:
        """A model a pattern can never select is still shown, at the end of the list."""
        catalog(
            _seed(
                old=_model("vendor.family-9-v1:0", _OLDER),
                new=_model("vendor.family-10-v1:0", _NEWER),
                undated=_model("vendor.family-11-v1:0", None),
            )
        )
        response = app_client.get("/search_models", params={"model": "vendor.family-*"})
        assert response.status_code == 200
        assert [model["id"] for model in response.json()] == [
            "vendor.family-10-v1:0",
            "vendor.family-9-v1:0",
            "vendor.family-11-v1:0",
        ]

    def test_an_overlong_filter_is_refused_before_it_is_compiled(
        self, catalog: Callable[[dict[str, ModelDetails]], None], app_client: TestClient
    ) -> None:
        """The filter reaches the same compile the request path does, so it is bounded too.

        ``search_models`` hands its ``model`` query straight to
        ``match_model_names``, which translates and compiles it. Without a bound
        the cost of that compile grows with the query, on the event loop and
        under the catalogue lock, exactly as on the request path.

        Ref: https://stdapi.ai/api_search_models/
             stdapi/models/__init__.py:match_model_names
        """
        catalog(_seed(new=_model("vendor.family-10-v1:0", _NEWER)))
        overlong = "vendor." + "a*" * 200

        response = app_client.get("/search_models", params={"model": overlong})

        assert response.status_code == 400, response.text
        assert "255" in response.json()["error"]

    def test_filter_combines_with_the_route_filter(
        self, catalog: Callable[[dict[str, ModelDetails]], None], app_client: TestClient
    ) -> None:
        """The pattern filter is one more AND filter, not a search of its own."""
        catalog(
            _seed(
                chat=_model("vendor.dual-chat-v1:0", _NEWER),
                embed=_model(
                    "vendor.dual-embed-v1:0", _NEWER, output_modalities=["EMBEDDING"]
                ),
            )
        )
        response = app_client.get(
            "/search_models", params={"model": "vendor.dual-*", "route": _CHAT_ROUTE}
        )
        assert response.status_code == 200
        assert [model["id"] for model in response.json()] == ["vendor.dual-chat-v1:0"]

    def test_retrieving_a_pattern_answers_with_the_concrete_model(
        self, catalog: Callable[[dict[str, ModelDetails]], None], app_client: TestClient
    ) -> None:
        """``GET /v1/models/{pattern}`` reports the model it selected, never the pattern.

        The same answer retrieving by alias already gives.
        """
        catalog(
            _seed(
                old=_model("vendor.family-9-v1:0", _OLDER),
                new=_model("vendor.family-10-v1:0", _NEWER),
            )
        )
        response = app_client.get("/v1/models/vendor.family-*")
        assert response.status_code == 200
        assert response.json()["id"] == "vendor.family-10-v1:0"

    def test_an_ambiguous_pattern_is_answered_as_a_bad_request(
        self, catalog: Callable[[dict[str, ModelDetails]], None], app_client: TestClient
    ) -> None:
        """The refusal reaches the caller as a 400 naming both tied models."""
        catalog(
            _seed(
                one=_model("vendor.family-10-alpha-v1:0", _NEWER),
                two=_model("vendor.family-10-beta-v1:0", _NEWER),
            )
        )
        response = app_client.get("/v1/models/vendor.family-*")
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["code"] == "ambiguous_model"
        assert error["param"] == "model"
        assert "vendor.family-10-alpha-v1:0" in error["message"]
        assert "vendor.family-10-beta-v1:0" in error["message"]

    def test_a_pattern_is_not_an_entry_in_the_model_list(
        self, catalog: Callable[[dict[str, ModelDetails]], None], app_client: TestClient
    ) -> None:
        """Patterns are inputs, so the list holds concrete models only, as with aliases."""
        catalog(_seed(only=_model("vendor.family-9-v1:0", _OLDER)))
        response = app_client.get("/v1/models")
        assert response.status_code == 200
        assert [model["id"] for model in response.json()["data"]] == [
            "vendor.family-9-v1:0"
        ]

    def test_the_models_list_is_rebuilt_when_the_catalog_changes(
        self, catalog: Callable[[dict[str, ModelDetails]], None], app_client: TestClient
    ) -> None:
        """A second seeded catalogue is what `/v1/models` answers with, not a cached first one.

        `/v1/models` caches its payload per catalog generation; a fixture that
        seeds a catalogue without bumping the generation would leave a caller
        reading whatever an earlier test in the same worker cached.
        """
        catalog(_seed(first=_model("vendor.family-9-v1:0", _OLDER)))
        first = app_client.get("/v1/models")
        assert [model["id"] for model in first.json()["data"]] == [
            "vendor.family-9-v1:0"
        ]
        catalog(_seed(second=_model("vendor.family-10-v1:0", _NEWER)))
        second = app_client.get("/v1/models")
        assert [model["id"] for model in second.json()["data"]] == [
            "vendor.family-10-v1:0"
        ]

    def test_extra_models_do_not_leak_into_the_model_list(
        self, catalog: Callable[[dict[str, ModelDetails]], None], app_client: TestClient
    ) -> None:
        """A non-Bedrock model registered by an earlier lifespan is not part of a seeded catalogue.

        `EXTRA_MODELS` is module-global; a `catalog` that isolates only `_MODELS`
        would let a service registered at startup (e.g. Polly, Transcribe) leak
        into a catalogue this test seeded to hold exactly one model.
        """
        models.EXTRA_MODELS["amazon.transcribe"] = _model("amazon.transcribe", _NEWER)
        catalog(_seed(only=_model("vendor.family-9-v1:0", _OLDER)))
        response = app_client.get("/v1/models")
        assert response.status_code == 200
        assert [model["id"] for model in response.json()["data"]] == [
            "vendor.family-9-v1:0"
        ]

    def test_extra_models_do_not_leak_into_a_pattern_search(
        self, catalog: Callable[[dict[str, ModelDetails]], None], app_client: TestClient
    ) -> None:
        """A stray `EXTRA_MODELS` alias does not join a seeded catalogue's pattern match."""
        models.EXTRA_MODELS["amazon.transcribe"] = _model(
            "amazon.transcribe", _NEWER, aliases=["gpt-transcribe"]
        )
        catalog(_seed(regular=_model("openai.gpt-5.6-sol", _OLDER, provider="OpenAI")))
        response = app_client.get("/search_models", params={"model": "gpt-*"})
        assert response.status_code == 200
        assert [model["id"] for model in response.json()] == ["openai.gpt-5.6-sol"]


class TestEndpointsThatChooseTheModelInAdvance:
    """Two endpoints fix their model before there is a request to serve.

    Both refuse a pattern instead of resolving one late, which would let a
    credential or a moderation call mean a different model than the one it was
    issued against.

    Ref: stdapi/routes/openai_realtime.py:create_realtime_client_secret
         stdapi/routes/openai_moderations.py:openai_moderation
    """

    def test_client_secret_refuses_a_pattern(self, app_client: TestClient) -> None:
        """The secret outlives its request, so its model is fixed when it is minted."""
        response = app_client.post(
            "/v1/realtime/client_secrets",
            json={"session": {"type": "realtime", "model": "amazon.nova-*"}},
        )
        assert response.status_code == 400
        assert "pattern is not available" in response.json()["error"]["message"]

    def test_moderation_refuses_a_pattern(self, app_client: TestClient) -> None:
        """Moderation selects its model before it looks at the input."""
        response = app_client.post(
            "/v1/moderations", json={"input": "hello", "model": "omni-moderation-*"}
        )
        assert response.status_code == 400
        assert "pattern is not available" in response.json()["error"]["message"]


class TestModelFieldLengthBound:
    """Every ``model`` field bounds its length before it reaches a regex compile.

    Regression tests for a request-path DoS: these fields carried
    ``min_length=1`` and no upper bound, so an oversized ``model`` string
    reached ``fnmatch.translate``/``re.compile`` on the event loop before
    ``_resolve_model_wildcard``'s own guard could see it.

    Ref: stdapi/types/openai_chat_completions.py:CompletionCreateParams
         stdapi/types/anthropic_messages.py:MessageCreateParams
         stdapi/types/anthropic_messages.py:MessageCountTokensParams
         stdapi/types/openai_responses.py:ResponseCreateParams
         stdapi/types/openai_audio.py:SpeechCreateParams
         stdapi/types/openai_audio.py:TranscriptionCreateParams
    """

    @pytest.mark.parametrize(
        ("params_class", "extra"),
        [
            (CompletionCreateParams, {"messages": [{"role": "user", "content": "hi"}]}),
            (MessageCreateParams, {"messages": [{"role": "user", "content": "hi"}]}),
            (
                MessageCountTokensParams,
                {"messages": [{"role": "user", "content": "hi"}]},
            ),
            (ResponseCreateParams, {}),
            (SpeechCreateParams, {"input": "hi"}),
            (TranscriptionCreateParams, {}),
        ],
    )
    def test_an_overlong_model_field_is_rejected_by_the_schema(
        self, params_class: type[BaseModel], extra: dict[str, Any]
    ) -> None:
        """A 256-character `model` value never reaches request handling."""
        with pytest.raises(ValidationError) as raised:
            params_class.model_validate({"model": "x" * 256, **extra})
        assert raised.value.errors()[0]["type"] == "string_too_long"


class TestRouteLiteralIsRegistered:
    """Every ``route=`` a ``validate_model`` call site passes names a real operation.

    ``_resolve_model_wildcard`` narrows candidates with
    ``_ALL_MODELS_BY_ROUTE_OR_TOOL.get(route, set())``, built from
    ``ROUTE_CAPABILITIES``: an unregistered literal silently answers an empty
    candidate set, so a renamed operation ID would 404 every pattern on that
    route while exact model names kept working, with nothing failing loudly.

    Ref: stdapi/models/__init__.py:_resolve_model_wildcard
         stdapi/models/capabilities.py:ROUTE_CAPABILITIES
    """

    #: Matches the ``route="..."`` keyword every ``validate_model`` call site writes.
    _ROUTE_LITERAL = compile_regex(r'route="([a-zA-Z0-9_]+)"')

    def test_every_route_literal_names_a_registered_operation(self) -> None:
        """Every `route=` literal under `stdapi/` is a key of `ROUTE_CAPABILITIES`."""
        root = Path(models.__file__).parent.parent
        literals = {
            match.group(1)
            for path in root.rglob("*.py")
            for match in self._ROUTE_LITERAL.finditer(path.read_text())
        }
        assert len(literals) >= 20, (
            f"only {len(literals)} route= literals found; the pattern stopped matching"
        )
        unknown = sorted(literals - set(ROUTE_CAPABILITIES))
        assert not unknown, f"route= literals name no registered operation: {unknown}"
