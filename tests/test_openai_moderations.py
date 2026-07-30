"""Tests for the OpenAI-compatible /v1/moderations route (unit and live).

Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     https://developers.openai.com/api/docs/guides/moderation
     https://stdapi.ai/api_openai_moderations/
     stdapi/routes/openai_moderations.py:create_moderation
"""

from base64 import b64encode
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from stdapi.config import SETTINGS
from stdapi.models.moderation import (
    ALL_CATEGORIES,
    IMAGE_CATEGORIES,
    amazon_bedrock_guardrail,
    amazon_comprehend,
)
from stdapi.routes import openai_moderations
from stdapi.types.openai_moderations import (
    ModerationCategories,
    ModerationCategoryAppliedInputTypes,
    ModerationCreateParams,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from openai import OpenAI

#: Minimal 1x1 PNG image bytes.
_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
    b"\xff?\x00\x05\xfe\x02\xfe\xa75\x81\x84\x00\x00\x00\x00IEND\xaeB`\x82"
)

#: The 1x1 PNG as the ``data:`` URI form the moderations route accepts.
_PNG_DATA_URI = f"data:image/png;base64,{b64encode(_PNG).decode()}"

#: A guardrail response with one flagged and one clean content filter.
_FLAGGED_RESPONSE: dict[str, Any] = {
    "action": "GUARDRAIL_INTERVENED",
    "assessments": [
        {
            "contentPolicy": {
                "filters": [
                    {"type": "HATE", "confidence": "HIGH", "action": "BLOCKED"},
                    {"type": "INSULTS", "confidence": "LOW", "action": "NONE"},
                ]
            }
        }
    ],
}

#: A guardrail response without any policy hit.
_CLEAN_RESPONSE: dict[str, Any] = {"action": "NONE", "assessments": []}


class _StubGuardrailClient:
    """Stub bedrock-runtime client recording apply_guardrail calls."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.requests: list[dict[str, Any]] = []
        self._response = response

    async def apply_guardrail(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        """Record the request and return the canned assessment."""
        self.requests.append(params)
        return self._response


@pytest.fixture
def configured_guardrail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure a default server guardrail."""
    monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", "gr123")
    monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", "1")


def _stub_client(
    monkeypatch: pytest.MonkeyPatch, response: dict[str, Any]
) -> tuple[_StubGuardrailClient, list[str]]:
    """Stub the bedrock-runtime app_client, recording the requested regions."""
    stub = _StubGuardrailClient(response)
    regions: list[str] = []

    def _get_client(_service: str, region: str) -> _StubGuardrailClient:
        regions.append(region)
        return stub

    monkeypatch.setattr(amazon_bedrock_guardrail, "get_client", _get_client)
    return stub, regions


@pytest.mark.local
@pytest.mark.usefixtures("configured_guardrail")
class TestModerationsRoute:
    """POST /v1/moderations: guardrail resolution and category mapping.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
         https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-content-filters.html
         stdapi/models/moderation/amazon_bedrock_guardrail.py:ModerationModel
    """

    def test_flags_mapped_categories(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guardrail content filters map to OpenAI categories and scores.

        Guardrails report confidence levels rather than probabilities, so
        scores come from a NONE/LOW/MEDIUM/HIGH to 0.0/0.25/0.5/0.75 table; a
        ``LOW`` non-blocking filter yields a non-zero score with the category
        still ``false``. All 13 OpenAI keys are always present even though only
        five guardrail filter types have a counterpart.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-content-filters.html
             stdapi/aws_bedrock.py:map_guardrail_filters
        """
        stub, _ = _stub_client(monkeypatch, _FLAGGED_RESPONSE)

        response = app_client.post("/v1/moderations", json={"input": "some text"})

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"].startswith("modr-")
        assert body["model"] == "gr123:1"
        (result,) = body["results"]
        assert result["flagged"] is True
        assert len(result["categories"]) == len(ALL_CATEGORIES)
        assert len(result["category_scores"]) == len(ALL_CATEGORIES)
        assert result["categories"]["hate"] is True
        assert result["categories"]["harassment"] is False
        assert result["categories"]["sexual"] is False
        assert result["category_scores"]["hate"] == 0.75
        assert result["category_scores"]["harassment"] == 0.25
        # No SEXUAL filter entry at all: the key still exists, scored 0.0.
        assert result["category_scores"]["sexual"] == 0.0
        applied = result["category_applied_input_types"]
        assert len(applied) == len(ALL_CATEGORIES)
        assert all(value == ["text"] for value in applied.values())
        (request,) = stub.requests
        assert request["guardrailIdentifier"] == "gr123"
        assert request["guardrailVersion"] == "1"
        assert request["source"] == "INPUT"
        assert request["content"] == [{"text": {"text": "some text"}}]

    def test_response_round_trips_through_the_openai_sdk(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A full response body validates against the installed openai SDK's response type.

        Guards the wire contract end to end: the aliased JSON keys
        (``violence/graphic`` and friends) must deserialize into the SDK's typed
        model and carry the mapped values, not just parse as JSON.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai_moderations.py:ModerationCreateResponse
        """
        from openai.types.moderation_create_response import (  # noqa: PLC0415
            ModerationCreateResponse as SdkModerationCreateResponse,
        )

        _stub_client(monkeypatch, _FLAGGED_RESPONSE)

        response = app_client.post("/v1/moderations", json={"input": "some text"})

        assert response.status_code == 200, response.text
        parsed = SdkModerationCreateResponse.model_validate(response.json())
        assert parsed.id.startswith("modr-")
        (result,) = parsed.results
        assert result.flagged is True
        assert result.categories.hate is True
        assert result.categories.harassment is False
        assert result.category_scores.hate == 0.75
        assert result.category_scores.harassment == 0.25
        assert result.category_applied_input_types.hate == ["text"]

    def test_multiple_inputs_yield_one_result_each(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each input element is classified independently.

        OpenAI returns one result per input element; the gateway issues one
        ApplyGuardrail call per element because ApplyGuardrail scores a whole
        content list as a single unit.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-independent-api.html
             stdapi/routes/openai_moderations.py:create_moderation
        """
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)

        response = app_client.post("/v1/moderations", json={"input": ["a", "b"]})

        assert response.status_code == 200, response.text
        results = response.json()["results"]
        assert len(results) == 2
        assert all(result["flagged"] is False for result in results)
        assert len(stub.requests) == 2
        assert sorted(
            block["text"]["text"]
            for request in stub.requests
            for block in request["content"]
        ) == ["a", "b"]

    def test_openai_model_name_uses_configured_guardrail(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OpenAI moderation model names resolve to the configured guardrail.

        The alias is echoed back verbatim in ``model`` while the call targets
        the server's guardrail, so clients written against OpenAI see their own
        model name.

        Ref: https://developers.openai.com/api/docs/guides/moderation
             stdapi/aws_bedrock.py:resolve_moderation_model
        """
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)

        response = app_client.post(
            "/v1/moderations", json={"input": "x", "model": "omni-moderation-latest"}
        )

        assert response.status_code == 200, response.text
        assert response.json()["model"] == "omni-moderation-latest"
        (request,) = stub.requests
        assert request["guardrailIdentifier"] == "gr123"
        assert request["guardrailVersion"] == "1"

    def test_default_guardrail_model_id(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """amazon.bedrock-runtime-guardrail selects the configured guardrail.

        This synthetic model ID exists only in this gateway; it is the explicit
        spelling of "the server's default guardrail".

        Ref: stdapi/aws_bedrock.py:GUARDRAIL_MODERATION_MODEL
             stdapi/aws_bedrock.py:resolve_moderation_model
        """
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)

        response = app_client.post(
            "/v1/moderations",
            json={"input": "x", "model": "amazon.bedrock-runtime-guardrail"},
        )

        assert response.status_code == 200, response.text
        assert response.json()["model"] == "amazon.bedrock-runtime-guardrail"
        (request,) = stub.requests
        assert request["guardrailIdentifier"] == "gr123"
        assert request["guardrailVersion"] == "1"

    def test_text_moderation_name_uses_comprehend(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """text-moderation-* aliases Comprehend even when a guardrail is set.

        OpenAI's legacy text-only moderation models map to the text-only AWS
        backend, so the configured guardrail is deliberately bypassed.

        Ref: https://docs.aws.amazon.com/comprehend/latest/APIReference/API_DetectToxicContent.html
             stdapi/aws_bedrock.py:is_comprehend_moderation_model
        """
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)
        batches = _stub_toxicity(monkeypatch, [[_CLEAN_RESULT]])

        response = app_client.post(
            "/v1/moderations", json={"input": "x", "model": "text-moderation-latest"}
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["model"] == "text-moderation-latest"
        assert body["results"][0]["flagged"] is False
        assert not stub.requests, "the guardrail must not be called"
        assert batches == [["x"]]

    def test_explicit_guardrail_requires_override(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit guardrail is rejected unless overrides are allowed.

        A client-supplied guardrail would let callers pick any guardrail in the
        account, so it is refused unless the operator opted in.

        Ref: stdapi/aws_bedrock.py:resolve_guardrail_model
             stdapi/config.py:_Settings.aws_bedrock_allow_guardrail_override
        """
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", False)

        response = app_client.post(
            "/v1/moderations", json={"input": "x", "model": "other456:2"}
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "not allowed" in error["message"]
        assert not stub.requests, "the rejected guardrail must not be called"

    def test_explicit_guardrail_with_override_allowed(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An explicit id:version guardrail is used when overrides are allowed.

        The ``<id>:<version>`` model form is split into ApplyGuardrail's two
        separate request fields, overriding the server's configured guardrail.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
             stdapi/aws_bedrock.py:resolve_guardrail_model
        """
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", True)

        response = app_client.post(
            "/v1/moderations", json={"input": "x", "model": "other456:2"}
        )

        assert response.status_code == 200, response.text
        assert response.json()["model"] == "other456:2"
        (request,) = stub.requests
        assert request["guardrailIdentifier"] == "other456"
        assert request["guardrailVersion"] == "2"

    def test_guardrail_arn_selects_its_region(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A guardrail ARN is applied in the region embedded in the ARN.

        Guardrails are regional resources, so the bedrock-runtime client must be
        created in the ARN's region rather than the primary Bedrock region. An
        ARN without an explicit version falls back to ``DRAFT``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
             stdapi/aws_bedrock.py:guardrail_region
        """
        stub, regions = _stub_client(monkeypatch, _CLEAN_RESPONSE)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", True)
        arn = "arn:aws:bedrock:eu-west-1:000000000000:guardrail/abc123"

        response = app_client.post("/v1/moderations", json={"input": "x", "model": arn})

        assert response.status_code == 200, response.text
        assert regions == ["eu-west-1"]
        (request,) = stub.requests
        assert request["guardrailIdentifier"] == arn
        assert request["guardrailVersion"] == "DRAFT"

    def test_image_input_data_uri(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A PNG data URI image becomes a guardrail image content block.

        The data URI is decoded to raw bytes for ApplyGuardrail's
        ``image.source.bytes``, and ``category_applied_input_types`` becomes
        ``["image"]`` only for the categories OpenAI marks image-capable, ``[]``
        for the text-only ones.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
             stdapi/models/moderation/amazon_bedrock_guardrail.py:_to_content_block
             stdapi/models/moderation/__init__.py:applied_input_types
        """
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)

        response = app_client.post(
            "/v1/moderations",
            json={
                "input": [{"type": "image_url", "image_url": {"url": _PNG_DATA_URI}}]
            },
        )

        assert response.status_code == 200, response.text
        (request,) = stub.requests
        (block,) = request["content"]
        assert block["image"]["format"] == "png"
        assert block["image"]["source"]["bytes"] == _PNG
        assert "text" not in block
        (result,) = response.json()["results"]
        assert result["flagged"] is False
        applied = result["category_applied_input_types"]
        assert applied["sexual"] == ["image"]
        assert applied["violence/graphic"] == ["image"]
        assert applied["hate"] == []
        assert applied["harassment"] == []
        assert {
            category for category, value in applied.items() if value == ["image"]
        } == set(IMAGE_CATEGORIES)
        assert all(
            applied[category] == []
            for category in ALL_CATEGORIES
            if category not in IMAGE_CATEGORIES
        )

    def test_jpeg_data_uri_maps_to_the_jpeg_guardrail_format(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A JPEG data URI becomes an ``image`` block whose ``format`` is ``jpeg``.

        ``_IMAGE_FORMATS`` maps the two MIME types ApplyGuardrail accepts onto
        its own format enum; ``image/jpeg`` must become ``jpeg`` and not the
        MIME string, which the API would reject.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
             stdapi/models/moderation/amazon_bedrock_guardrail.py:_IMAGE_FORMATS
        """
        from io import BytesIO  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415

        buffer = BytesIO()
        Image.new("RGB", (1, 1)).save(buffer, "JPEG")
        jpeg = buffer.getvalue()
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)

        response = app_client.post(
            "/v1/moderations",
            json={
                "input": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{b64encode(jpeg).decode()}"
                        },
                    }
                ]
            },
        )

        assert response.status_code == 200, response.text
        (request,) = stub.requests
        (block,) = request["content"]
        assert block["image"]["format"] == "jpeg"
        assert block["image"]["source"]["bytes"] == jpeg

    def test_https_image_url_is_fetched_and_forwarded_as_bytes(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ``https://`` image URL is downloaded and sent to the guardrail as raw bytes.

        ApplyGuardrail takes no URL locator, so the gateway resolves the
        content type from the response metadata and inlines the body itself.
        Only the transport is stubbed here; the sniffing and inlining under
        test are the gateway's.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
             stdapi/models/moderation/amazon_bedrock_guardrail.py:_to_content_block
             stdapi/input_file.py:_HttpSource
        """
        from stdapi import input_file  # noqa: PLC0415

        async def _resolve_metadata(source: Any) -> None:  # noqa: ANN401
            source._content_type = "image/png"  # noqa: SLF001
            source._size = len(_PNG)  # noqa: SLF001
            source._filename = "image.png"  # noqa: SLF001

        async def _read(_source: Any) -> bytes:  # noqa: ANN401
            return _PNG

        monkeypatch.setattr(
            input_file._HttpSource,  # noqa: SLF001
            "_resolve_metadata",
            _resolve_metadata,
        )
        monkeypatch.setattr(input_file._HttpSource, "_read", _read)  # noqa: SLF001
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)

        response = app_client.post(
            "/v1/moderations",
            json={
                "input": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.invalid/image.png"},
                    }
                ]
            },
        )

        assert response.status_code == 200, response.text
        (request,) = stub.requests
        (block,) = request["content"]
        assert block["image"]["format"] == "png"
        assert block["image"]["source"]["bytes"] == _PNG

    def test_whitespace_only_input_still_calls_the_guardrail(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only an exactly-empty string short-circuits; ``"   "`` is classified and billed.

        The shortcut is ``text == ""``, not ``text.strip() == ""``: a
        whitespace-only input reaches ApplyGuardrail and is metered as one
        1000-character text unit.

        Ref: https://stdapi.ai/api_openai_moderations/
             stdapi/models/moderation/amazon_bedrock_guardrail.py:ModerationModel.moderate
        """
        from stdapi import monitoring  # noqa: PLC0415

        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)
        written: list[dict[str, Any]] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)

        response = app_client.post("/v1/moderations", json={"input": "   "})

        assert response.status_code == 200, response.text
        assert stub.requests == [
            {
                "guardrailIdentifier": "gr123",
                "guardrailVersion": "1",
                "source": "INPUT",
                "content": [{"text": {"text": "   "}}],
                "outputScope": "FULL",
            }
        ]
        (request_log,) = [entry for entry in written if entry.get("type") == "request"]
        (usage,) = request_log["usage"]
        assert usage["model"] == "gr123:1"
        assert usage["text_units"] == 1

    def test_batches_preserve_input_order(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guardrail verdicts stay positionally aligned with the inputs across batches.

        Results are matched to inputs by position only, so a reordering in the
        batched ``gather`` would mis-attribute verdicts. The stub derives each
        verdict from the text it receives, making any reordering visible.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_moderations.py:_INPUT_BATCH_SIZE
        """
        count = 2 * openai_moderations._INPUT_BATCH_SIZE + 5  # noqa: SLF001
        texts = [f"text-{index}" for index in range(count)]
        flagged_indices = {0, 3, 11, count - 1}

        class _PerTextGuardrailClient:
            async def apply_guardrail(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
                (block,) = params["content"]
                index = int(block["text"]["text"].removeprefix("text-"))
                if index not in flagged_indices:
                    return _CLEAN_RESPONSE
                return _FLAGGED_RESPONSE

        monkeypatch.setattr(
            amazon_bedrock_guardrail,
            "get_client",
            lambda _service, _region: _PerTextGuardrailClient(),
        )

        response = app_client.post("/v1/moderations", json={"input": texts})

        assert response.status_code == 200, response.text
        results = response.json()["results"]
        assert len(results) == count
        for index, result in enumerate(results):
            assert result["flagged"] is (index in flagged_indices), index

    def test_inputs_exceeding_batch_size_are_all_classified(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """More inputs than the concurrency batch size still yield one result each.

        Inputs are classified in batches of ``_INPUT_BATCH_SIZE`` concurrent
        ApplyGuardrail calls; a count that is not a multiple of the batch size
        catches an off-by-one in the final partial batch.

        Ref: stdapi/routes/openai_moderations.py:_INPUT_BATCH_SIZE
        """
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)
        count = 2 * openai_moderations._INPUT_BATCH_SIZE + 5  # noqa: SLF001
        texts = [f"text-{index}" for index in range(count)]

        response = app_client.post("/v1/moderations", json={"input": texts})

        assert response.status_code == 200, response.text
        assert len(response.json()["results"]) == count
        assert len(stub.requests) == count
        assert sorted(
            block["text"]["text"]
            for request in stub.requests
            for block in request["content"]
        ) == sorted(texts), "every input must be classified exactly once"

    def test_unsupported_image_format_rejected(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non PNG/JPEG images are rejected with 400.

        ApplyGuardrail's image content block only accepts png and jpeg, so the
        gateway refuses other formats up front instead of letting AWS fail.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
             stdapi/models/moderation/amazon_bedrock_guardrail.py:_IMAGE_FORMATS
        """
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)
        webp_uri = f"data:image/webp;base64,{b64encode(_PNG).decode()}"

        response = app_client.post(
            "/v1/moderations",
            json={"input": [{"type": "image_url", "image_url": {"url": webp_uri}}]},
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "PNG or JPEG" in error["message"]
        assert not stub.requests, "the unsupported image must not reach AWS"

    def test_empty_input_list_rejected(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty `input` list is rejected with 400, not classified as zero results.

        ``input`` carries ``min_length=1`` on both list branches of the union, so
        the request fails schema validation and is reported as an
        ``invalid_request_error`` naming the offending field.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai_moderations.py:ModerationCreateParams
        """
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)

        response = app_client.post("/v1/moderations", json={"input": []})

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "input" in error["message"]
        assert not stub.requests

    def test_non_content_policy_hits_flag_without_categories(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Topic/word policy hits flag the input with all categories false.

        Only ApplyGuardrail's ``contentPolicy`` filters have OpenAI category
        counterparts; denied topics, word filters, sensitive information and
        contextual grounding can only surface through ``flagged``. ``action`` is
        ``NONE`` here to prove the per-entry ``action`` is what drives flagging.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
             stdapi/aws_bedrock.py:_GUARDRAIL_POLICY_ENTRIES
        """
        _stub_client(
            monkeypatch,
            {
                "action": "NONE",
                "assessments": [
                    {
                        "topicPolicy": {
                            "topics": [{"name": "Politics", "action": "BLOCKED"}]
                        },
                        "wordPolicy": {
                            "customWords": [{"match": "BLOCKWORD", "action": "BLOCKED"}]
                        },
                    }
                ],
            },
        )

        response = app_client.post("/v1/moderations", json={"input": "some text"})

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is True
        assert not any(result["categories"].values())
        assert all(score == 0.0 for score in result["category_scores"].values())

    def test_requests_full_output_scope(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """apply_guardrail is called with outputScope=FULL to surface non-flagged confidences.

        Without ``FULL`` AWS omits non-detected filter entries entirely, which
        would make every unflagged category score default to 0.0 instead of
        reporting its real confidence.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-independent-api.html
             stdapi/models/moderation/amazon_bedrock_guardrail.py:ModerationModel.moderate
        """
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)

        response = app_client.post("/v1/moderations", json={"input": "some text"})

        assert response.status_code == 200, response.text
        (request,) = stub.requests
        assert request["outputScope"] == "FULL"

    def test_non_detected_filter_confidence_is_surfaced(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-detected filter entry (only returned under FULL scope) reports its real confidence.

        ``detected: false`` with ``confidence: MEDIUM`` is the shape AWS returns
        under ``outputScope=FULL``: the OpenAI category stays ``false`` while its
        score carries the mapped 0.5, mirroring OpenAI's sub-threshold scores.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-independent-api.html
             stdapi/aws_bedrock.py:_GUARDRAIL_CONFIDENCE_SCORES
        """
        _stub_client(
            monkeypatch,
            {
                "action": "NONE",
                "assessments": [
                    {
                        "contentPolicy": {
                            "filters": [
                                {
                                    "type": "INSULTS",
                                    "confidence": "MEDIUM",
                                    "detected": False,
                                    "action": "NONE",
                                }
                            ]
                        }
                    }
                ],
            },
        )

        response = app_client.post("/v1/moderations", json={"input": "some text"})

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is False
        assert result["categories"]["harassment"] is False
        # Non-zero even though the category was never flagged.
        assert result["category_scores"]["harassment"] == 0.5

    def test_empty_string_input_is_clean_without_aws_call(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exactly-empty string yields an unflagged result and no AWS call.

        ApplyGuardrail rejects empty content, so the gateway short-circuits to
        preserve OpenAI parity (and avoid billing). The stub is armed with the
        flagged response: any AWS call would make the result flagged.

        Ref: https://stdapi.ai/api_openai_moderations/
             stdapi/models/moderation/amazon_bedrock_guardrail.py:ModerationModel.moderate
        """
        stub, _ = _stub_client(monkeypatch, _FLAGGED_RESPONSE)

        response = app_client.post("/v1/moderations", json={"input": ""})

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is False
        assert not any(result["categories"].values())
        assert all(score == 0.0 for score in result["category_scores"].values())
        applied = result["category_applied_input_types"]
        assert len(applied) == len(ALL_CATEGORIES)
        assert all(value == ["text"] for value in applied.values())
        assert not stub.requests

    def test_empty_string_among_inputs_only_skips_itself(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-empty siblings of an empty string are still classified.

        The empty-input short-circuit is per element, so exactly one
        ApplyGuardrail call is made and the result list still aligns
        positionally with the request.

        Ref: https://stdapi.ai/api_openai_moderations/
             stdapi/routes/openai_moderations.py:create_moderation
        """
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)

        response = app_client.post("/v1/moderations", json={"input": ["", "hello"]})

        assert response.status_code == 200, response.text
        results = response.json()["results"]
        assert len(results) == 2
        assert all(result["flagged"] is False for result in results)
        (request,) = stub.requests
        assert request["content"] == [{"text": {"text": "hello"}}]

    def test_empty_text_part_is_clean_without_aws_call(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty {"type": "text"} part short-circuits like a plain string.

        The multimodal text part and the bare string share the same empty-input
        path, so neither reaches ApplyGuardrail (armed here with the flagged
        response).

        Ref: https://stdapi.ai/api_openai_moderations/
             stdapi/models/moderation/amazon_bedrock_guardrail.py:ModerationModel.moderate
        """
        stub, _ = _stub_client(monkeypatch, _FLAGGED_RESPONSE)

        response = app_client.post(
            "/v1/moderations", json={"input": [{"type": "text", "text": ""}]}
        )

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is False
        assert not any(result["categories"].values())
        assert not stub.requests

    def test_missing_or_wrong_bearer_returns_401_envelope(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unauthenticated requests get the OpenAI 401 error envelope.

        Both a missing and a wrong bearer token must fail before any AWS call,
        with OpenAI's ``authentication_error`` type rather than a generic 401.

        Ref: https://developers.openai.com/api/docs/guides/error-codes
             stdapi/auth.py:authenticate
        """
        from asyncio import run  # noqa: PLC0415

        from pydantic import SecretStr  # noqa: PLC0415

        from stdapi import auth  # noqa: PLC0415
        from stdapi.main import app  # noqa: PLC0415

        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)
        # The lifespan-initialized handler is not set up here: install one.
        monkeypatch.setattr(SETTINGS, "api_key", SecretStr("a-real-secret"))
        monkeypatch.setattr(SETTINGS, "api_key_ssm_parameter", None)
        monkeypatch.setattr(SETTINGS, "api_key_secretsmanager_secret", None)
        handler = auth.AuthenticationHandler()
        assert run(handler.initialize()) is True
        monkeypatch.setattr(auth, "_auth_handler", handler)
        anonymous = TestClient(app)

        for headers in ({}, {"Authorization": "Bearer wrong-key"}):
            response = anonymous.post(
                "/v1/moderations", json={"input": "x"}, headers=headers
            )
            assert response.status_code == 401
            error = response.json()["error"]
            assert error["type"] == "authentication_error"
            assert error["message"]
        assert not stub.requests

    def test_usage_records_text_units_and_model(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guardrail moderation records billed text units per input, summed.

        AWS bills guardrail policies per 1,000-character text unit, rounded up
        per call, so two inputs are metered separately and then summed rather
        than rounded once over the concatenation.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
             stdapi/usage.py:record_guardrail_usage
        """
        from stdapi import monitoring  # noqa: PLC0415

        _stub_client(monkeypatch, _CLEAN_RESPONSE)
        written: list[dict[str, Any]] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)

        response = app_client.post(
            "/v1/moderations", json={"input": ["a" * 1_500, "short"]}
        )

        assert response.status_code == 200, response.text
        (request_log,) = [w for w in written if w.get("type") == "request"]
        (entry,) = request_log["usage"]
        assert entry["service"] == "bedrock-runtime"
        assert entry["model"] == "gr123:1"
        assert entry["region"] == SETTINGS.aws_bedrock_regions[0]
        # ceil(1500 / 1000) + ceil(5 / 1000) = 2 + 1 text units.
        assert entry["text_units"] == 3
        assert "input_images" not in entry

    def test_usage_records_images_per_image(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Guardrail image moderation records one input image per input.

        Image content is billed per image, not per text unit, so the usage entry
        must carry ``input_images`` and no character-derived quantity.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
             stdapi/usage.py:record_guardrail_usage
        """
        from stdapi import monitoring  # noqa: PLC0415

        _stub_client(monkeypatch, _CLEAN_RESPONSE)
        written: list[dict[str, Any]] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)

        response = app_client.post(
            "/v1/moderations",
            json={
                "input": [{"type": "image_url", "image_url": {"url": _PNG_DATA_URI}}]
            },
        )

        assert response.status_code == 200, response.text
        (request_log,) = [w for w in written if w.get("type") == "request"]
        (entry,) = request_log["usage"]
        assert entry["service"] == "bedrock-runtime"
        assert entry["model"] == "gr123:1"
        assert entry["input_images"] == 1
        assert "text_units" not in entry


@pytest.mark.local
class TestModerationsSdkParity:
    """Pins against the installed openai SDK to catch upstream category drift.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         https://developers.openai.com/api/docs/guides/moderation
         stdapi/types/openai_moderations.py:ModerationCategories
    """

    def test_category_json_keys_match_the_installed_sdk(self) -> None:
        """Category and applied-input-type JSON keys match openai's Moderation submodels.

        The gateway must always emit the full omni-moderation vocabulary (13
        keys, including the ones no AWS backend can ever set), so the key set is
        pinned to the installed SDK instead of a hand-written list.

        Ref: stdapi/models/moderation/__init__.py:ALL_CATEGORIES
        """
        from openai.types.moderation import Categories as SdkCategories  # noqa: PLC0415
        from openai.types.moderation import (  # noqa: PLC0415
            CategoryAppliedInputTypes as SdkCategoryAppliedInputTypes,
        )

        def _json_keys(fields: dict[str, Any]) -> set[str]:
            return {info.alias or name for name, info in fields.items()}

        sdk_categories = _json_keys(SdkCategories.model_fields)
        assert len(sdk_categories) == 13, sdk_categories
        assert set(ALL_CATEGORIES) == sdk_categories
        assert len(ALL_CATEGORIES) == len(sdk_categories), "no duplicate aliases"
        assert _json_keys(ModerationCategories.model_fields) == sdk_categories
        assert _json_keys(
            ModerationCategoryAppliedInputTypes.model_fields
        ) == _json_keys(SdkCategoryAppliedInputTypes.model_fields)

    def test_image_capable_categories_match_the_installed_sdk(self) -> None:
        """`IMAGE_CATEGORIES` matches the categories accepting "image" in the installed SDK.

        Only these categories may report ``["image"]`` in
        ``category_applied_input_types``; the rest are text-only per the OpenAI
        schema and must report ``[]`` on image inputs.

        Ref: stdapi/models/moderation/__init__.py:IMAGE_CATEGORIES
        """
        from typing import get_args  # noqa: PLC0415

        from openai.types.moderation import (  # noqa: PLC0415
            CategoryAppliedInputTypes as SdkCategoryAppliedInputTypes,
        )

        sdk_image_categories = {
            info.alias or name
            for name, info in SdkCategoryAppliedInputTypes.model_fields.items()
            if "image" in get_args(get_args(info.annotation)[0])
        }
        assert sdk_image_categories, "SDK annotations no longer expose image support"
        assert sdk_image_categories == IMAGE_CATEGORIES
        assert sdk_image_categories < set(ALL_CATEGORIES)


@pytest.mark.local
class TestModerationInputLimit:
    """The input array is capped to bound per-element AWS moderation calls.

    OpenAI's schema caps ``input`` at 2048 items; the gateway enforces the same
    bound because every element costs one billable AWS moderation call.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/types/openai_moderations.py:ModerationCreateParams
    """

    def test_input_at_the_limit_is_accepted(self) -> None:
        """An input array of exactly the maximum length validates, unchanged.

        The boundary is inclusive and validation is lossless: all 2048 elements
        survive, no element is dropped or coerced, and ``model`` stays unset so
        the server default applies.
        """
        params = ModerationCreateParams.model_validate({"input": ["x"] * 2048})

        assert params.input == ["x"] * 2048
        assert params.model is None

    def test_input_over_the_limit_is_rejected(self) -> None:
        """An input array beyond the maximum length is rejected.

        Every list branch of the ``input`` union must report ``too_long`` with
        ``max_length=2048``, so the rejection is the length cap and not an
        unrelated union mismatch.
        """
        with pytest.raises(ValidationError) as excinfo:
            ModerationCreateParams.model_validate({"input": ["x"] * 2049})

        errors = excinfo.value.errors()
        too_long = [error for error in errors if error["type"] == "too_long"]
        assert too_long, errors
        assert all(error["loc"][0] == "input" for error in too_long), too_long
        assert {error["ctx"]["max_length"] for error in too_long} == {2048}


#: A Comprehend toxicity result with violent-threat and profanity labels.
_TOXIC_RESULT: dict[str, Any] = {
    "Labels": [
        {"Name": "VIOLENCE_OR_THREAT", "Score": 0.91},
        {"Name": "PROFANITY", "Score": 0.10},
    ],
    "Toxicity": 0.88,
}

#: A Comprehend toxicity result without any label above the threshold.
_CLEAN_RESULT: dict[str, Any] = {
    "Labels": [{"Name": "INSULT", "Score": 0.02}],
    "Toxicity": 0.01,
}


def _stub_toxicity(
    monkeypatch: pytest.MonkeyPatch, results: list[list[dict[str, Any]]]
) -> list[list[str]]:
    """Stub the Comprehend failover call, returning one result entry per segment.

    Each call returns the canned result list for that call, padded with
    zero-score entries so that, as with the real API, every submitted segment
    gets one ``ResultList`` entry.

    Returns:
        The recorded text-segment batches, one entry per API call.
    """
    batches: list[list[str]] = []

    class _StubComprehendClient:
        async def detect_dominant_language(self, *, Text: str) -> dict[str, Any]:  # noqa: N803
            return {"Languages": [{"LanguageCode": "en", "Score": 0.99}]}

        async def detect_toxic_content(
            self,
            *,
            TextSegments: list[dict[str, str]],  # noqa: N803
            LanguageCode: str,  # noqa: N803
        ) -> dict[str, Any]:
            assert LanguageCode == "en"
            batches.append([segment["Text"] for segment in TextSegments])
            canned = results[min(len(batches), len(results)) - 1]
            padding: list[dict[str, Any]] = [{"Labels": [], "Toxicity": 0.0}] * (
                len(TextSegments) - len(canned)
            )
            return {"ResultList": canned + padding}

    async def _call(
        service: str,
        regions: list[str],
        call: Any,  # noqa: ANN401
    ) -> tuple[dict[str, Any], str]:
        assert service == "comprehend"
        return await call(_StubComprehendClient(), regions[0]), regions[0]

    monkeypatch.setattr(amazon_comprehend, "call_with_region_failover", _call)
    return batches


@pytest.mark.local
class TestComprehendModerationsRoute:
    """POST /v1/moderations: Amazon Comprehend toxicity backend.

    Ref: https://docs.aws.amazon.com/comprehend/latest/APIReference/API_DetectToxicContent.html
         https://docs.aws.amazon.com/comprehend/latest/dg/trust-safety.html
         stdapi/models/moderation/amazon_comprehend.py:ModerationModel
    """

    @pytest.fixture(autouse=True)
    def _no_ambient_guardrail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure no guardrail from the environment leaks into these tests."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", None)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", None)

    def test_default_backend_without_guardrail(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a configured guardrail, moderation runs on Comprehend.

        ``VIOLENCE_OR_THREAT`` maps to ``violence`` and carries the raw
        Comprehend score, while ``PROFANITY`` (also present in the canned
        result) has no OpenAI counterpart and is dropped.

        Ref: https://docs.aws.amazon.com/comprehend/latest/dg/trust-safety.html
             stdapi/models/moderation/amazon_comprehend.py:_TOXICITY_CATEGORIES
        """
        batches = _stub_toxicity(monkeypatch, [[_TOXIC_RESULT]])

        response = app_client.post(
            "/v1/moderations", json={"input": "threatening text"}
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"].startswith("modr-")
        assert body["model"] == "amazon.comprehend-toxicity"
        (result,) = body["results"]
        assert result["flagged"] is True
        assert len(result["categories"]) == len(ALL_CATEGORIES)
        assert result["categories"]["violence"] is True
        assert result["categories"]["harassment"] is False
        assert result["category_scores"]["violence"] == 0.91
        # PROFANITY has no OpenAI category: it must not leak into another one.
        assert {
            category
            for category, flagged in result["categories"].items()
            if flagged is True
        } == {"violence"}
        applied = result["category_applied_input_types"]
        assert len(applied) == len(ALL_CATEGORIES)
        assert all(value == ["text"] for value in applied.values())
        assert batches == [["threatening text"]]

    def test_inputs_exceeding_batch_size_are_all_classified_in_order(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """More inputs than the concurrency batch size yield one correctly-mapped result each.

        Results must stay positionally aligned with the request across batch
        boundaries: the stub derives each verdict from the segment text, so a
        misordered gather would surface as a wrong ``flagged`` index.

        Ref: stdapi/routes/openai_moderations.py:_INPUT_BATCH_SIZE
        """
        count = 2 * openai_moderations._INPUT_BATCH_SIZE + 5  # noqa: SLF001
        texts = [f"text-{i}" for i in range(count)]
        toxic_indices = {2, 7, count - 1}

        class _StubComprehendClient:
            async def detect_dominant_language(self, *, Text: str) -> dict[str, Any]:  # noqa: N803
                return {"Languages": [{"LanguageCode": "en", "Score": 0.99}]}

            async def detect_toxic_content(
                self,
                *,
                TextSegments: list[dict[str, str]],  # noqa: N803
                LanguageCode: str,  # noqa: N803
            ) -> dict[str, Any]:
                assert LanguageCode == "en"
                (segment,) = TextSegments
                index = int(segment["Text"].removeprefix("text-"))
                score = 0.9 if index in toxic_indices else 0.0
                return {"ResultList": [{"Labels": [], "Toxicity": score}]}

        async def _call(
            service: str,
            regions: list[str],
            call: Any,  # noqa: ANN401
        ) -> tuple[dict[str, Any], str]:
            assert service == "comprehend"
            return await call(_StubComprehendClient(), regions[0]), regions[0]

        monkeypatch.setattr(amazon_comprehend, "call_with_region_failover", _call)

        response = app_client.post("/v1/moderations", json={"input": texts})

        assert response.status_code == 200, response.text
        results = response.json()["results"]
        assert len(results) == count
        for index, result in enumerate(results):
            assert result["flagged"] is (index in toxic_indices), index

    def test_usage_records_comprehend_units_and_region(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Comprehend moderation records character-count units with the per-call minimum.

        Comprehend bills 100-character units with a 3-unit minimum per call, and
        toxicity detection needs a language, so one moderation produces two
        billable calls metered under distinct synthetic model IDs.

        Ref: https://docs.aws.amazon.com/comprehend/latest/dg/guidelines-and-limits.html
             stdapi/usage.py:record_comprehend_usage
        """
        from stdapi import monitoring  # noqa: PLC0415
        from stdapi.aws import service_regions  # noqa: PLC0415

        _stub_toxicity(monkeypatch, [[_CLEAN_RESULT]])
        written: list[dict[str, Any]] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)

        response = app_client.post("/v1/moderations", json={"input": "hi"})

        assert response.status_code == 200, response.text
        (request_log,) = [w for w in written if w.get("type") == "request"]
        # One entry for the language-detection call, one for the toxicity call.
        entries = {e["model"]: e for e in request_log["usage"]}
        entry = entries["amazon.comprehend-toxicity"]
        assert entry["service"] == "comprehend"
        assert entry["region"] == service_regions(SETTINGS.aws_comprehend_region)[0]
        # ceil(len("hi") / 100) = 1, below the 3-unit per-call minimum.
        assert entry["comprehend_units"] == 3
        language_entry = entries["amazon.comprehend-language-detection"]
        assert language_entry["service"] == "comprehend"
        assert language_entry["comprehend_units"] == 3

    def test_language_detection_samples_only_the_first_500_characters(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DetectDominantLanguage receives at most a 500-character sample of the input.

        The sample caps a per-request billable Comprehend call: without it, a
        long moderation input would be billed twice at full length. 500
        characters bill as ceil(500 / 100) = 5 units, above the 3-unit minimum.

        Ref: https://docs.aws.amazon.com/comprehend/latest/APIReference/API_DetectDominantLanguage.html
             stdapi/models/moderation/amazon_comprehend.py:_LANG_DETECT_SAMPLE_SIZE
             stdapi/usage.py:record_comprehend_usage
        """
        from stdapi import monitoring  # noqa: PLC0415

        text = "a" * 5_000
        detected: list[str] = []

        class _StubComprehendClient:
            async def detect_dominant_language(self, *, Text: str) -> dict[str, Any]:  # noqa: N803
                detected.append(Text)
                return {"Languages": [{"LanguageCode": "en", "Score": 0.99}]}

            async def detect_toxic_content(
                self,
                *,
                TextSegments: list[dict[str, str]],  # noqa: N803
                LanguageCode: str,  # noqa: N803
            ) -> dict[str, Any]:
                return {
                    "ResultList": [{"Labels": [], "Toxicity": 0.0}] * len(TextSegments)
                }

        async def _call(
            service: str,
            regions: list[str],
            call: Any,  # noqa: ANN401
        ) -> tuple[dict[str, Any], str]:
            assert service == "comprehend"
            return await call(_StubComprehendClient(), regions[0]), regions[0]

        monkeypatch.setattr(amazon_comprehend, "call_with_region_failover", _call)
        written: list[dict[str, Any]] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)

        response = app_client.post("/v1/moderations", json={"input": text})

        assert response.status_code == 200, response.text
        assert detected == [text[:500]]
        (request_log,) = [w for w in written if w.get("type") == "request"]
        entries = {entry["model"]: entry for entry in request_log["usage"]}
        assert entries["amazon.comprehend-language-detection"]["comprehend_units"] == 5

    def test_mixed_text_and_image_input_rejected_as_a_whole(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pinned current behavior: an image sibling fails the whole batch with 400.

        Comprehend has no partial-success mode: the image element's
        ``ApiError`` propagates out of the batch's ``gather``, so the
        request is rejected even though its text sibling is valid. The message
        points at the guardrail requirement without naming the server setting.

        Ref: https://docs.aws.amazon.com/comprehend/latest/APIReference/API_DetectToxicContent.html
             stdapi/models/moderation/amazon_comprehend.py:ModerationModel.moderate
        """
        _stub_toxicity(monkeypatch, [[_CLEAN_RESULT]])

        response = app_client.post(
            "/v1/moderations",
            json={
                "input": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": _PNG_DATA_URI}},
                ]
            },
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "guardrail" in error["message"]
        assert "aws_bedrock_guardrail_identifier" not in error["message"].lower()

    @pytest.mark.parametrize(
        "model", ["omni-moderation-latest", "text-moderation-latest"]
    )
    def test_openai_model_name_falls_back_to_comprehend(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        """OpenAI moderation names use Comprehend when no guardrail is set.

        ``omni-moderation-*`` normally targets the guardrail, but with none
        configured it degrades to the Comprehend backend instead of failing;
        either way the requested alias is echoed back.

        Ref: https://stdapi.ai/api_openai_moderations/
             stdapi/models/moderation/amazon_bedrock_guardrail.py:ModerationModel.get_aliases
        """
        batches = _stub_toxicity(monkeypatch, [[_CLEAN_RESULT]])

        response = app_client.post(
            "/v1/moderations", json={"input": "x", "model": model}
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["model"] == model
        assert body["results"][0]["flagged"] is False
        assert batches == [["x"]], "Comprehend must have classified the input"

    def test_default_guardrail_model_requires_guardrail(
        self, app_client: TestClient
    ) -> None:
        """amazon.bedrock-runtime-guardrail errors when no guardrail is set.

        Unlike the ``omni-moderation-*`` aliases, the explicit guardrail model ID
        does not silently fall back to Comprehend. The message must point the
        caller at the administrator without disclosing the setting name.

        Ref: https://stdapi.ai/api_openai_moderations/
             stdapi/aws_bedrock.py:resolve_moderation_model
        """
        response = app_client.post(
            "/v1/moderations",
            json={"input": "x", "model": "amazon.bedrock-runtime-guardrail"},
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        message = error["message"]
        assert "administrator" in message
        assert "aws_bedrock_guardrail_identifier" not in message.lower()

    def test_graphic_label_maps_to_violence_graphic(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A GRAPHIC hit serializes under the violence/graphic JSON key.

        ``violence/graphic`` has no Bedrock Guardrails counterpart and exists
        only on this backend; the slash-bearing alias must survive
        serialization, and GRAPHIC must not also set plain ``violence``.

        Ref: https://docs.aws.amazon.com/comprehend/latest/dg/trust-safety.html
             stdapi/models/moderation/amazon_comprehend.py:_TOXICITY_CATEGORIES
        """
        _stub_toxicity(
            monkeypatch,
            [[{"Labels": [{"Name": "GRAPHIC", "Score": 0.8}], "Toxicity": 0.4}]],
        )

        response = app_client.post("/v1/moderations", json={"input": "x"})

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is True
        assert result["categories"]["violence/graphic"] is True
        assert result["categories"]["violence"] is False
        assert result["category_scores"]["violence/graphic"] == 0.8

    @pytest.mark.parametrize("label", ["HARASSMENT_OR_ABUSE", "INSULT"])
    def test_harassment_labels_map_to_harassment_category(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch, label: str
    ) -> None:
        """HARASSMENT_OR_ABUSE and INSULT both map to harassment, at the threshold.

        Two Comprehend labels collapse onto one OpenAI category, and 0.5 is the
        inclusive flag threshold.

        Ref: https://docs.aws.amazon.com/comprehend/latest/dg/trust-safety.html
             stdapi/models/moderation/amazon_comprehend.py:_TOXICITY_THRESHOLD
        """
        _stub_toxicity(
            monkeypatch,
            [[{"Labels": [{"Name": label, "Score": 0.5}], "Toxicity": 0.1}]],
        )

        response = app_client.post("/v1/moderations", json={"input": "x"})

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is True
        assert result["categories"]["harassment"] is True
        assert result["category_scores"]["harassment"] == 0.5

    def test_text_object_input(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A {"type": "text"} input part is classified like a plain string.

        OpenAI's multimodal input form must reach ``DetectToxicContent`` with the
        part's ``text`` only, not a serialized wrapper object.

        Ref: https://developers.openai.com/api/docs/guides/moderation
             stdapi/types/openai_moderations.py:ModerationTextInput
        """
        batches = _stub_toxicity(monkeypatch, [[_CLEAN_RESULT]])

        response = app_client.post(
            "/v1/moderations", json={"input": [{"type": "text", "text": "hello"}]}
        )

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is False
        assert all(
            value == ["text"]
            for value in result["category_applied_input_types"].values()
        )
        assert batches == [["hello"]]

    def test_each_input_gets_own_result(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two inputs in one request yield two independent, correctly-mapped results.

        The stub keys its canned result on the segment text rather than
        call order, so the assertion holds regardless of how
        ``asyncio.gather`` schedules the underlying Comprehend calls.

        Ref: https://docs.aws.amazon.com/comprehend/latest/APIReference/API_DetectToxicContent.html
             stdapi/routes/openai_moderations.py:create_moderation
        """
        canned = {"bad": _TOXIC_RESULT, "ok": _CLEAN_RESULT}

        class _StubComprehendClient:
            async def detect_dominant_language(self, *, Text: str) -> dict[str, Any]:  # noqa: N803
                return {"Languages": [{"LanguageCode": "en", "Score": 0.99}]}

            async def detect_toxic_content(
                self,
                *,
                TextSegments: list[dict[str, str]],  # noqa: N803
                LanguageCode: str,  # noqa: N803
            ) -> dict[str, Any]:
                assert LanguageCode == "en"
                (segment,) = TextSegments
                return {"ResultList": [canned[segment["Text"]]]}

        async def _call(
            service: str,
            regions: list[str],
            call: Any,  # noqa: ANN401
        ) -> tuple[dict[str, Any], str]:
            assert service == "comprehend"
            return await call(_StubComprehendClient(), regions[0]), regions[0]

        monkeypatch.setattr(amazon_comprehend, "call_with_region_failover", _call)

        response = app_client.post("/v1/moderations", json={"input": ["bad", "ok"]})

        assert response.status_code == 200, response.text
        toxic, clean = response.json()["results"]
        assert toxic["flagged"] is True
        assert toxic["categories"]["violence"] is True
        assert toxic["category_scores"]["violence"] == 0.91
        assert clean["flagged"] is False
        assert not any(clean["categories"].values())

    def test_score_at_threshold_flags(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A label score of exactly 0.5 flags the input (threshold is inclusive).

        Comprehend returns probabilities, not verdicts, so the gateway applies
        its own ``>=`` threshold; the label score is reported verbatim.

        Ref: https://docs.aws.amazon.com/comprehend/latest/APIReference/API_DetectToxicContent.html
             stdapi/models/moderation/amazon_comprehend.py:_TOXICITY_THRESHOLD
        """
        _stub_toxicity(
            monkeypatch,
            [[{"Labels": [{"Name": "SEXUAL", "Score": 0.5}], "Toxicity": 0.1}]],
        )

        response = app_client.post("/v1/moderations", json={"input": "x"})

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is True
        assert result["categories"]["sexual"] is True
        assert result["category_scores"]["sexual"] == 0.5

    def test_explicit_comprehend_model_bypasses_guardrail(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """amazon.comprehend-toxicity is honored even with a guardrail set.

        Selecting the text-only backend explicitly overrides the server's
        configured guardrail, which is why the guardrail stub must stay unused.

        Ref: https://stdapi.ai/api_openai_moderations/
             stdapi/aws_bedrock.py:is_comprehend_moderation_model
        """
        stub, _ = _stub_client(monkeypatch, _FLAGGED_RESPONSE)
        batches = _stub_toxicity(monkeypatch, [[_CLEAN_RESULT]])

        response = app_client.post(
            "/v1/moderations",
            json={"input": "x", "model": "amazon.comprehend-toxicity"},
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["model"] == "amazon.comprehend-toxicity"
        assert body["results"][0]["flagged"] is False
        assert not stub.requests, "the configured guardrail must not be called"
        assert batches == [["x"]]

    def test_unmapped_label_contributes_to_flagged_only(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A PROFANITY hit flags the input without setting any category.

        PROFANITY is one of Comprehend's seven labels but has no OpenAI category,
        so it may only raise ``flagged``; leaking it into ``harassment`` would be
        a mis-mapping.

        Ref: https://docs.aws.amazon.com/comprehend/latest/dg/trust-safety.html
             stdapi/models/moderation/amazon_comprehend.py:_TOXICITY_CATEGORIES
        """
        _stub_toxicity(
            monkeypatch,
            [[{"Labels": [{"Name": "PROFANITY", "Score": 0.75}], "Toxicity": 0.3}]],
        )

        response = app_client.post("/v1/moderations", json={"input": "x"})

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is True
        assert not any(result["categories"].values())
        assert all(score == 0.0 for score in result["category_scores"].values())

    def test_long_text_is_chunked_and_scores_aggregated(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Long inputs split into 1 KB segments, 10 per call, keeping max scores.

        ``DetectToxicContent`` accepts at most 10 segments of 1 KB each, so a
        10,500-byte input becomes 10 + 1 segments over two calls, and the highest
        per-category score across all segments wins.

        Ref: https://docs.aws.amazon.com/comprehend/latest/APIReference/API_DetectToxicContent.html
             stdapi/models/moderation/amazon_comprehend.py:_split_toxicity_segments
        """
        hate = {"Labels": [{"Name": "HATE_SPEECH", "Score": 0.7}], "Toxicity": 0.6}
        batches = _stub_toxicity(monkeypatch, [[_CLEAN_RESULT], [hate]])
        text = "a" * 10_500

        response = app_client.post("/v1/moderations", json={"input": text})

        assert response.status_code == 200, response.text
        assert [len(batch) for batch in batches] == [10, 1]
        assert "".join(segment for batch in batches for segment in batch) == text
        assert all(
            len(segment.encode()) <= 1_000 for batch in batches for segment in batch
        )
        (result,) = response.json()["results"]
        assert result["flagged"] is True
        assert result["categories"]["hate"] is True
        assert result["category_scores"]["hate"] == 0.7

    @pytest.mark.parametrize(
        ("char", "sizes"),
        [("é", [1_000, 200]), ("€", [999, 201]), ("🎉", [1_000, 200])],
    )
    def test_multibyte_characters_stay_whole_at_segment_boundaries(
        self, char: str, sizes: list[int]
    ) -> None:
        """Splitting counts UTF-8 bytes without breaking characters apart.

        The 1 KB Comprehend limit is a byte limit, so a segment may stop short of
        1,000 bytes (999 for the 3-byte euro sign) rather than split a character
        and produce invalid UTF-8.

        Ref: https://docs.aws.amazon.com/comprehend/latest/APIReference/API_DetectToxicContent.html
             stdapi/models/moderation/amazon_comprehend.py:_TOXICITY_SEGMENT_BYTES
        """
        text = char * (1_200 // len(char.encode()))  # 1200 bytes total
        segments = amazon_comprehend._split_toxicity_segments(text)  # noqa: SLF001
        assert "".join(segments) == text
        assert [len(segment.encode()) for segment in segments] == sizes
        assert all(len(segment.encode()) <= 1_000 for segment in segments)

    def test_overall_toxicity_score_alone_flags(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A high overall toxicity flags the input even with low label scores.

        ``Toxicity`` is Comprehend's aggregate per-segment score, not a label, and
        it feeds ``flagged`` on its own; sub-threshold label scores are still
        reported.

        Ref: https://docs.aws.amazon.com/comprehend/latest/APIReference/API_DetectToxicContent.html
             stdapi/models/moderation/amazon_comprehend.py:ModerationModel.moderate
        """
        _stub_toxicity(
            monkeypatch,
            [[{"Labels": [{"Name": "INSULT", "Score": 0.2}], "Toxicity": 0.9}]],
        )

        response = app_client.post("/v1/moderations", json={"input": "x"})

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is True
        assert not any(result["categories"].values())
        assert result["category_scores"]["harassment"] == 0.2

    def test_empty_text_is_clean_without_api_call(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty input yields a clean result without calling Comprehend.

        Same OpenAI-parity shortcut as the guardrail backend: no
        ``DetectToxicContent`` call, hence no billing. The stub is armed with the
        toxic result, so any call would flip ``flagged``.

        Ref: https://stdapi.ai/api_openai_moderations/
             stdapi/models/moderation/amazon_comprehend.py:ModerationModel.moderate
        """
        batches = _stub_toxicity(monkeypatch, [[_TOXIC_RESULT]])

        response = app_client.post("/v1/moderations", json={"input": ""})

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is False
        assert not any(result["categories"].values())
        assert batches == []

    def test_detected_language_is_used_for_toxicity_detection(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-English dominant language is forwarded to DetectToxicContent.

        ``DetectToxicContent`` requires ``LanguageCode``, which OpenAI's request
        has no field for, so the gateway detects it with
        ``DetectDominantLanguage`` first. The dev guide claims English only, but
        the API accepts 12 languages and the gateway forwards them.

        Ref: https://docs.aws.amazon.com/comprehend/latest/APIReference/API_DetectToxicContent.html
             stdapi/models/moderation/amazon_comprehend.py:_detect_toxicity_language
        """
        language_codes: list[str] = []

        class _StubComprehendClient:
            async def detect_dominant_language(self, *, Text: str) -> dict[str, Any]:  # noqa: N803
                return {"Languages": [{"LanguageCode": "fr", "Score": 0.98}]}

            async def detect_toxic_content(
                self,
                *,
                TextSegments: list[dict[str, str]],  # noqa: N803
                LanguageCode: str,  # noqa: N803
            ) -> dict[str, Any]:
                language_codes.append(LanguageCode)
                return {"ResultList": [_CLEAN_RESULT]}

        async def _call(
            service: str,
            regions: list[str],
            call: Any,  # noqa: ANN401
        ) -> tuple[dict[str, Any], str]:
            assert service == "comprehend"
            return await call(_StubComprehendClient(), regions[0]), regions[0]

        monkeypatch.setattr(amazon_comprehend, "call_with_region_failover", _call)

        response = app_client.post(
            "/v1/moderations", json={"input": "un texte en français"}
        )

        assert response.status_code == 200, response.text
        assert language_codes == ["fr"]

    def test_unsupported_detected_language_falls_back_to_english(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dominant language DetectToxicContent doesn't support falls back to `en`.

        ``DetectDominantLanguage`` recognizes far more languages than
        ``DetectToxicContent`` accepts, so an unsupported detection must degrade
        to ``en`` instead of raising a ValidationException from AWS.

        Ref: https://docs.aws.amazon.com/comprehend/latest/APIReference/API_DetectToxicContent.html
             stdapi/models/moderation/amazon_comprehend.py:_TOXICITY_LANGUAGES
        """
        language_codes: list[str] = []

        class _StubComprehendClient:
            async def detect_dominant_language(self, *, Text: str) -> dict[str, Any]:  # noqa: N803
                # Dutch ("nl") isn't in DetectToxicContent's supported language set.
                return {"Languages": [{"LanguageCode": "nl", "Score": 0.95}]}

            async def detect_toxic_content(
                self,
                *,
                TextSegments: list[dict[str, str]],  # noqa: N803
                LanguageCode: str,  # noqa: N803
            ) -> dict[str, Any]:
                language_codes.append(LanguageCode)
                return {"ResultList": [_CLEAN_RESULT]}

        async def _call(
            service: str,
            regions: list[str],
            call: Any,  # noqa: ANN401
        ) -> tuple[dict[str, Any], str]:
            assert service == "comprehend"
            return await call(_StubComprehendClient(), regions[0]), regions[0]

        monkeypatch.setattr(amazon_comprehend, "call_with_region_failover", _call)

        response = app_client.post("/v1/moderations", json={"input": "een tekst"})

        assert response.status_code == 200, response.text
        assert language_codes == ["en"]

    def test_image_input_requires_guardrail(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Image moderation is rejected on the Comprehend backend.

        ``DetectToxicContent`` is text-only, so an image input can only be served
        by a guardrail; the error must say so without disclosing the server
        setting name.

        Ref: https://docs.aws.amazon.com/comprehend/latest/APIReference/API_DetectToxicContent.html
             stdapi/models/moderation/amazon_comprehend.py:ModerationModel.moderate
        """
        batches = _stub_toxicity(monkeypatch, [[_CLEAN_RESULT]])

        response = app_client.post(
            "/v1/moderations",
            json={
                "input": [{"type": "image_url", "image_url": {"url": _PNG_DATA_URI}}]
            },
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        message = error["message"]
        assert "guardrail" in message
        assert "aws_bedrock_guardrail_identifier" not in message.lower()
        assert batches == [], "no Comprehend call for an image input"


@pytest.fixture(scope="module")
def live_guardrail(use_official_api: bool) -> Iterator[str]:
    """Create a temporary guardrail, passed as the explicit moderation model.

    On the official API no guardrail exists; tests use the OpenAI moderation
    model instead. The server (local or --server-url) must allow guardrail
    overrides, which is automatic when no global guardrail is configured.
    Yields the guardrail ARN so the server resolves the guardrail's region.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-content-filters.html
         stdapi/aws_bedrock.py:guardrail_region
    """
    if use_official_api:
        yield ""
        return
    import time  # noqa: PLC0415

    import boto3  # type: ignore[import-untyped]  # noqa: PLC0415

    region = SETTINGS.aws_bedrock_regions[0]
    bedrock = boto3.client("bedrock", region_name=region)
    created = bedrock.create_guardrail(
        name=f"stdapi-tests-moderations-{uuid4().hex[:8]}",
        blockedInputMessaging="Blocked by test guardrail.",
        blockedOutputsMessaging="Blocked by test guardrail.",
        wordPolicyConfig={"wordsConfig": [{"text": "BLOCKWORDXYZ"}]},
        contentPolicyConfig={
            "filtersConfig": [
                {"type": name, "inputStrength": "HIGH", "outputStrength": "HIGH"}
                for name in ("HATE", "INSULTS", "SEXUAL", "VIOLENCE", "MISCONDUCT")
            ]
        },
    )
    guardrail_id = created["guardrailId"]
    try:
        for _ in range(30):
            status = bedrock.get_guardrail(guardrailIdentifier=guardrail_id)["status"]
            if status == "READY":
                break
            time.sleep(1)
        else:
            pytest.fail("Test guardrail never reached READY status.")
        yield created["guardrailArn"]
    finally:
        # A guardrail is a billable account resource: delete it on every path,
        # including a polling error or an interrupted session.
        bedrock.delete_guardrail(guardrailIdentifier=guardrail_id)


@pytest.mark.slow
@pytest.mark.xdist_group("moderations_guardrail")
class TestModerationsLive:
    """Live moderation against a real guardrail or the official API.

    Ref: https://developers.openai.com/api/docs/guides/moderation
         https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-independent-api.html
         stdapi/routes/openai_moderations.py:create_moderation
    """

    def test_moderations_endpoint(
        self, openai_client: OpenAI, live_guardrail: str, use_official_api: bool
    ) -> None:
        """Clean and harmful inputs classify as expected.

        Against the test guardrail the harmful input trips the custom word
        policy, which has no OpenAI category counterpart, so only ``flagged``
        rises; on the official API the phrase trips the ``violence`` category.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ApplyGuardrail.html
             stdapi/aws_bedrock.py:map_guardrail_filters
        """
        model = "omni-moderation-latest" if use_official_api else live_guardrail
        harmful = (
            "I am going to kill you." if use_official_api else "It has BLOCKWORDXYZ."
        )
        result = openai_client.moderations.create(
            model=model, input=["The weather is nice today.", harmful]
        )
        assert result.model
        assert len(result.results) == 2
        clean, flagged = result.results
        assert clean.flagged is False
        assert not any(clean.categories.model_dump().values())
        assert flagged.flagged is True
        # Every OpenAI category key is present on both backends.
        assert set(clean.categories.model_dump(by_alias=True)) >= set(ALL_CATEGORIES)
        assert clean.category_applied_input_types.violence == ["text"]
        if use_official_api:
            assert flagged.categories.violence is True
        else:
            assert result.id.startswith("modr-")
            assert result.model == live_guardrail

    def test_chat_moderation_param(
        self,
        openai_client: OpenAI,
        live_guardrail: str,
        chat_model: str,
        use_official_api: bool,
    ) -> None:
        """The moderation parameter reports guardrail results on the response.

        The gateway-only ``moderation`` request parameter attaches the guardrail
        to the Converse call with ``trace: enabled`` and maps the resulting trace
        into both an input and an output result set.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-converse-api.html
             stdapi/routes/_moderation.py:build_chat_moderation
        """
        if use_official_api:
            pytest.skip("the moderation request parameter is a stdapi extension")
        completion = openai_client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Say hello in one word."}],
            extra_body={"moderation": {"model": live_guardrail}},
        )
        assert completion.choices[0].message.content
        moderation = completion.moderation
        assert moderation is not None
        assert moderation.input.type == "moderation_results"
        assert moderation.input.model == live_guardrail
        (input_result,) = moderation.input.results
        assert input_result.flagged is False
        assert input_result.model == live_guardrail
        # The output direction is reported too, from outputAssessments.
        assert moderation.output.type == "moderation_results"
        assert moderation.output.model == live_guardrail
        (output_result,) = moderation.output.results
        assert output_result.model == live_guardrail


class TestComprehendModerationsLive:
    """Live moderation against Amazon Comprehend toxicity detection.

    Ref: https://docs.aws.amazon.com/comprehend/latest/APIReference/API_DetectToxicContent.html
         stdapi/models/moderation/amazon_comprehend.py:ModerationModel
    """

    def test_comprehend_moderation(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Clean and threatening inputs classify as expected.

        A death threat is a ``VIOLENCE_OR_THREAT`` hit, which maps to
        ``violence`` with a score at or above the gateway's 0.5 flag threshold.

        Ref: https://docs.aws.amazon.com/comprehend/latest/dg/trust-safety.html
             stdapi/models/moderation/amazon_comprehend.py:_TOXICITY_THRESHOLD
        """
        if use_official_api:
            pytest.skip("Amazon Comprehend moderation is gateway-specific")
        result = openai_client.moderations.create(
            model="amazon.comprehend-toxicity",
            input=["The weather is nice today.", "I am going to kill you."],
        )
        assert result.id.startswith("modr-")
        assert result.model == "amazon.comprehend-toxicity"
        assert len(result.results) == 2
        clean, flagged = result.results
        assert clean.flagged is False
        assert not any(clean.categories.model_dump().values())
        assert flagged.flagged is True
        assert flagged.categories.violence is True
        assert flagged.category_scores.violence > 0.5
        # Text-only backend: every category applies to "text".
        assert set(flagged.categories.model_dump(by_alias=True)) == set(ALL_CATEGORIES)
        assert flagged.category_applied_input_types.violence == ["text"]
        assert flagged.category_applied_input_types.hate == ["text"]
