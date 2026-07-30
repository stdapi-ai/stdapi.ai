"""Tests for the OpenAI /v1/models route.

The gateway synthesizes the catalogue from Bedrock ``ListFoundationModels``
across every configured region, plus the non-Bedrock services it fronts (Polly,
Transcribe, Comprehend).  Two fields are therefore gateway conventions rather
than upstream values: ``created`` is the Bedrock ``startOfLifeTime`` epoch and
falls back to ``0`` when unknown, and ``owned_by`` is the provider name (e.g.
``Amazon``) instead of ``openai``/``organization-owner``.

Ref: https://developers.openai.com/api/reference/resources/models/methods/list
     https://developers.openai.com/api/reference/resources/models/methods/retrieve
     stdapi/routes/openai_models.py:format_bedrock_model_to_openai
"""

import time

import pytest
from openai import NotFoundError, OpenAI

#: The four fields the OpenAI Models API documents on a Model object.
_MODEL_FIELDS = frozenset({"id", "object", "created", "owned_by"})

#: 2020-01-01T00:00:00Z — earliest plausible model launch date, in Unix seconds.
_EARLIEST_CREATED = 1577836800


class TestModels:
    """List and retrieve behavior of ``GET /v1/models`` and ``GET /v1/models/{model}``.

    Ref: https://developers.openai.com/api/reference/resources/models/methods/list
         stdapi/routes/openai_models.py:list_models
    """

    def test_list_models_basic_functionality(self, openai_client: OpenAI) -> None:
        """``GET /models`` returns a ``list`` envelope of ``model`` objects.

        Ref: stdapi/types/openai_models.py:Model
        """
        response = openai_client.models.list()

        assert response.object == "list"
        assert isinstance(response.data, list)
        assert len(response.data) > 1, "the catalogue must expose more than one model"

        for model in response.data:
            assert model.object == "model"
            assert isinstance(model.id, str)
            assert model.id
            assert isinstance(model.created, int)
            assert not isinstance(model.created, bool)
            assert model.created >= 0
            assert isinstance(model.owned_by, str)
            assert model.owned_by

    def test_list_models_response_structure_validation(
        self, openai_client: OpenAI
    ) -> None:
        """Every listed model carries all four documented fields, none of them null.

        The OpenAI SDK builds ``Model`` leniently, so a field the gateway failed
        to emit arrives as ``None`` rather than as a validation error: the
        serialized payload is checked instead of the attributes.

        Ref: https://developers.openai.com/api/reference/overview
             stdapi/types/openai_models.py:Model
        """
        response = openai_client.models.list()

        assert response.object == "list"
        assert response.data

        for model in response.data:
            dumped = model.model_dump()
            assert dumped.keys() >= _MODEL_FIELDS, (
                f"{model.id} is missing documented fields: "
                f"{sorted(_MODEL_FIELDS - dumped.keys())}"
            )
            assert all(dumped[field] is not None for field in _MODEL_FIELDS)
            assert dumped["object"] == "model"

    def test_retrieve_specific_model(self, openai_client: OpenAI) -> None:
        """``GET /models/{model}`` returns a bare Model echoing the requested id.

        Unlike the list route the payload is not wrapped in a ``list`` envelope,
        so it carries no ``data`` key.

        Ref: https://developers.openai.com/api/reference/resources/models/methods/retrieve
             stdapi/routes/openai_models.py:retrieve_model
        """
        models_response = openai_client.models.list()
        assert models_response.data

        test_model_id = models_response.data[0].id
        model = openai_client.models.retrieve(test_model_id)

        assert model.id == test_model_id
        assert model.object == "model"
        assert isinstance(model.created, int)
        assert model.created >= 0
        assert model.owned_by

        dumped = model.model_dump()
        assert dumped.keys() >= _MODEL_FIELDS
        assert "data" not in dumped, "retrieve must not return a list envelope"

    def test_model_filtering_and_availability(self, openai_client: OpenAI) -> None:
        """Listed ids are unique and every advertised id is individually retrievable.

        A model reachable in several regions is merged into a single catalogue
        entry, so a duplicate id would mean the cross-region merge regressed.
        The first and last entries are retrieved to prove the list and the
        single-model route agree on what exists.

        Ref: stdapi/models/__init__.py:get_all_models_details
        """
        response = openai_client.models.list()

        model_ids = [model.id for model in response.data]
        assert model_ids, "Should have at least one model available"
        assert len(set(model_ids)) == len(model_ids), "Model IDs should be unique"

        for model_id in model_ids:
            assert not any(char in model_id for char in [" ", "\n", "\t"])

        for model_id in (model_ids[0], model_ids[-1]):
            assert openai_client.models.retrieve(model_id).id == model_id

    def test_model_metadata_consistency(self, openai_client: OpenAI) -> None:
        """List and retrieve return byte-identical metadata for the same model.

        Both routes serialize through ``format_bedrock_model_to_openai``, so
        every field — not just the id — must match; a difference would mean one
        of the two paths reads a different cache.

        Ref: stdapi/routes/openai_models.py:format_bedrock_model_to_openai
        """
        models_response = openai_client.models.list()
        assert models_response.data

        test_models = models_response.data[: min(3, len(models_response.data))]

        for list_model in test_models:
            retrieved_model = openai_client.models.retrieve(list_model.id)

            assert type(list_model) is type(retrieved_model)
            assert retrieved_model.model_dump() == list_model.model_dump(), (
                f"list and retrieve disagree on {list_model.id}"
            )

    def test_model_creation_timestamps(self, openai_client: OpenAI) -> None:
        """``created`` is either 0 (launch date unknown) or a past Unix timestamp.

        The gateway derives it from the Bedrock model lifecycle's
        ``startOfLifeTime``, which is optional, and uses ``0`` as the sentinel —
        a launch date in the future would mean the epoch conversion is wrong.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-lifecycle.html
             stdapi/models/__init__.py:ModelDetails
        """
        response = openai_client.models.list()
        now = int(time.time())

        for model in response.data:
            assert isinstance(model.created, int)
            assert model.created >= 0
            assert model.created == 0 or _EARLIEST_CREATED < model.created <= now, (
                f"{model.id} has an implausible created timestamp: {model.created}"
            )

    def test_invalid_model_retrieval_error(self, openai_client: OpenAI) -> None:
        """An unknown model id is a 404 ``model_not_found`` naming the requested id.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
             stdapi/api_errors.py:UnsupportedModelError
        """
        with pytest.raises(NotFoundError) as exc_info:
            openai_client.models.retrieve("invalid-nonexistent-model-id")

        error = exc_info.value
        assert error.status_code == 404
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error"
        assert error_body["code"] == "model_not_found"
        assert "model" in error_body["message"].lower()
        assert "invalid-nonexistent-model-id" in error_body["message"]

    def test_empty_model_id_retrieval_error(self, openai_client: OpenAI) -> None:
        """The OpenAI SDK refuses an empty model id before any HTTP request is made.

        This is an SDK-side guard, not gateway behavior: no request reaches
        ``GET /models/{model}``, whose path parameter has its own
        ``min_length=1`` constraint.

        Ref: https://github.com/openai/openai-python
             stdapi/routes/openai_models.py:retrieve_model
        """
        with pytest.raises(ValueError, match=r".*(?:non-empty|empty).*") as exc_info:
            openai_client.models.retrieve("")

        error_message = str(exc_info.value)
        assert (
            "non-empty value" in error_message.lower()
            or "empty" in error_message.lower()
        )
        assert "model" in error_message.lower()

    def test_model_ownership_validation(self, openai_client: OpenAI) -> None:
        """``owned_by`` is a per-model provider string, not a constant or the model id.

        The gateway fills it from the Bedrock ``providerName`` (e.g. ``Amazon``,
        ``Anthropic``), so a catalogue spanning several providers must expose
        more than one distinct owner.

        Ref: stdapi/routes/openai_models.py:format_bedrock_model_to_openai
        """
        response = openai_client.models.list()

        owners = set()
        for model in response.data:
            assert isinstance(model.owned_by, str)
            assert model.owned_by
            assert not any(char in model.owned_by for char in ["\n", "\t"])
            assert model.owned_by != model.id
            owners.add(model.owned_by)

        assert len(owners) > 1, (
            f"owned_by must be derived per model, got a single value: {owners}"
        )

    def test_model_list_pagination_behavior(self, openai_client: OpenAI) -> None:
        """The catalogue is returned in a single, stably ordered page.

        ``GET /models`` takes no cursor parameters: iterating the SDK pager
        yields exactly the entries of ``data``.  The gateway serves the list from
        a sorted cache, so a second call must return the same ids in the same
        order rather than the region-merge iteration order of the moment.

        Ref: stdapi/routes/openai_models.py:list_models
        """
        response = openai_client.models.list()

        assert response.data
        assert len(response.data) < 1000  # Reasonable upper bound
        assert [model.id for model in response] == [
            model.id for model in response.data
        ], "the models list is not paginated: iterating must yield exactly one page"

        for model in response.data:
            assert model.id
            assert model.object == "model"
            assert model.created is not None
            assert model.owned_by is not None

        assert [model.id for model in openai_client.models.list().data] == [
            model.id for model in response.data
        ], "repeated calls must return the same models in the same order"

    def test_model_id_format_validation(self, openai_client: OpenAI) -> None:
        """Model ids are URL-usable path segments, including Bedrock ``:`` versions.

        Bedrock ids embed a version suffix (``…-v1:0``); the retrieve route must
        accept that colon unescaped, since clients paste the id straight from the
        list response.

        Ref: stdapi/routes/openai_models.py:retrieve_model
        """
        response = openai_client.models.list()

        for model in response.data:
            model_id = model.id

            assert isinstance(model_id, str)
            assert model_id
            assert len(model_id) < 256  # Route path parameter max_length
            assert model_id == model_id.strip()
            assert not any(char in model_id for char in ["\n", "\r", "\t"])

        versioned = next((model.id for model in response.data if ":" in model.id), None)
        if versioned is not None:
            assert openai_client.models.retrieve(versioned).id == versioned

    def test_model_capabilities_detection(
        self, openai_client: OpenAI, models: dict[str, str]
    ) -> None:
        """The catalogue advertises the models the suite actually calls.

        The chat and embedding entries come from Bedrock while the transcription
        entry is an extra, non-Bedrock service registered by the gateway, so all
        three being listed proves the catalogue merges every source rather than
        exposing Bedrock only.

        Ref: stdapi/models/__init__.py:EXTRA_MODELS
        """
        response = openai_client.models.list()
        model_ids = {model.id for model in response.data}

        for capability in ("chat", "embedding", "transcription"):
            assert models[capability] in model_ids, (
                f"{capability} model {models[capability]} is not advertised by /v1/models"
            )
