"""Tests for the OpenAI-compatible /v1/moderations route (unit and live)."""

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
def client(api_key: str) -> TestClient:
    """Test client without lifespan (no AWS startup), pre-authenticated."""
    from stdapi.main import app  # noqa: PLC0415

    return TestClient(app, headers={"Authorization": f"Bearer {api_key}"})


@pytest.fixture
def configured_guardrail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure a default server guardrail."""
    monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", "gr123")
    monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", "1")


def _stub_client(
    monkeypatch: pytest.MonkeyPatch, response: dict[str, Any]
) -> tuple[_StubGuardrailClient, list[str]]:
    """Stub the bedrock-runtime client, recording the requested regions."""
    stub = _StubGuardrailClient(response)
    regions: list[str] = []

    def _get_client(_service: str, region: str) -> _StubGuardrailClient:
        regions.append(region)
        return stub

    monkeypatch.setattr(amazon_bedrock_guardrail, "get_client", _get_client)
    return stub, regions


@pytest.mark.local
class TestModerationsRoute:
    """POST /v1/moderations: guardrail resolution and category mapping."""

    def test_flags_mapped_categories(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """Guardrail content filters map to OpenAI categories and scores."""
        stub, _ = _stub_client(monkeypatch, _FLAGGED_RESPONSE)

        response = client.post("/v1/moderations", json={"input": "some text"})

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["id"].startswith("modr-")
        assert body["model"] == "gr123:1"
        (result,) = body["results"]
        assert result["flagged"] is True
        assert result["categories"]["hate"] is True
        assert result["categories"]["harassment"] is False
        assert result["categories"]["sexual"] is False
        assert result["category_scores"]["hate"] == 0.75
        assert result["category_scores"]["harassment"] == 0.25
        applied = result["category_applied_input_types"]
        assert len(applied) == 13
        assert all(value == ["text"] for value in applied.values())
        (request,) = stub.requests
        assert request["guardrailIdentifier"] == "gr123"
        assert request["guardrailVersion"] == "1"
        assert request["source"] == "INPUT"
        assert request["content"] == [{"text": {"text": "some text"}}]

    def test_response_round_trips_through_the_openai_sdk(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """A full response body validates against the installed openai SDK's response type."""
        from openai.types.moderation_create_response import (  # noqa: PLC0415
            ModerationCreateResponse as SdkModerationCreateResponse,
        )

        _stub_client(monkeypatch, _FLAGGED_RESPONSE)

        response = client.post("/v1/moderations", json={"input": "some text"})

        assert response.status_code == 200, response.text
        SdkModerationCreateResponse.model_validate(response.json())

    def test_multiple_inputs_yield_one_result_each(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """Each input element is classified independently."""
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)

        response = client.post("/v1/moderations", json={"input": ["a", "b"]})

        assert response.status_code == 200, response.text
        results = response.json()["results"]
        assert len(results) == 2
        assert all(result["flagged"] is False for result in results)
        assert len(stub.requests) == 2

    def test_openai_model_name_uses_configured_guardrail(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """OpenAI moderation model names resolve to the configured guardrail."""
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)

        response = client.post(
            "/v1/moderations", json={"input": "x", "model": "omni-moderation-latest"}
        )

        assert response.status_code == 200, response.text
        assert response.json()["model"] == "omni-moderation-latest"
        assert stub.requests[0]["guardrailIdentifier"] == "gr123"

    def test_default_guardrail_model_id(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """amazon.bedrock-runtime-guardrail selects the configured guardrail."""
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)

        response = client.post(
            "/v1/moderations",
            json={"input": "x", "model": "amazon.bedrock-runtime-guardrail"},
        )

        assert response.status_code == 200, response.text
        assert response.json()["model"] == "amazon.bedrock-runtime-guardrail"
        assert stub.requests[0]["guardrailIdentifier"] == "gr123"

    def test_text_moderation_name_uses_comprehend(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """text-moderation-* aliases Comprehend even when a guardrail is set."""
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)
        batches = _stub_toxicity(monkeypatch, [[_CLEAN_RESULT]])

        response = client.post(
            "/v1/moderations", json={"input": "x", "model": "text-moderation-latest"}
        )

        assert response.status_code == 200, response.text
        assert response.json()["model"] == "text-moderation-latest"
        assert not stub.requests
        assert batches == [["x"]]

    def test_explicit_guardrail_requires_override(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """An explicit guardrail is rejected unless overrides are allowed."""
        _stub_client(monkeypatch, _CLEAN_RESPONSE)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", False)

        response = client.post(
            "/v1/moderations", json={"input": "x", "model": "other456:2"}
        )

        assert response.status_code == 400
        assert "not allowed" in response.json()["error"]["message"]

    def test_explicit_guardrail_with_override_allowed(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """An explicit id:version guardrail is used when overrides are allowed."""
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", True)

        response = client.post(
            "/v1/moderations", json={"input": "x", "model": "other456:2"}
        )

        assert response.status_code == 200, response.text
        (request,) = stub.requests
        assert request["guardrailIdentifier"] == "other456"
        assert request["guardrailVersion"] == "2"

    def test_guardrail_arn_selects_its_region(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """A guardrail ARN is applied in the region embedded in the ARN."""
        stub, regions = _stub_client(monkeypatch, _CLEAN_RESPONSE)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_allow_guardrail_override", True)
        arn = "arn:aws:bedrock:eu-west-1:000000000000:guardrail/abc123"

        response = client.post("/v1/moderations", json={"input": "x", "model": arn})

        assert response.status_code == 200, response.text
        assert regions == ["eu-west-1"]
        (request,) = stub.requests
        assert request["guardrailIdentifier"] == arn
        assert request["guardrailVersion"] == "DRAFT"

    def test_image_input_data_uri(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """A PNG data URI image becomes a guardrail image content block."""
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)
        data_uri = f"data:image/png;base64,{b64encode(_PNG).decode()}"

        response = client.post(
            "/v1/moderations",
            json={"input": [{"type": "image_url", "image_url": {"url": data_uri}}]},
        )

        assert response.status_code == 200, response.text
        (request,) = stub.requests
        (block,) = request["content"]
        assert block["image"]["format"] == "png"
        assert block["image"]["source"]["bytes"] == _PNG
        (result,) = response.json()["results"]
        applied = result["category_applied_input_types"]
        assert applied["sexual"] == ["image"]
        assert applied["violence/graphic"] == ["image"]
        assert applied["hate"] == []
        assert applied["harassment"] == []

    def test_inputs_exceeding_batch_size_are_all_classified(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """More inputs than the concurrency batch size still yield one result each."""
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)
        count = 2 * openai_moderations._INPUT_BATCH_SIZE + 5  # noqa: SLF001

        response = client.post(
            "/v1/moderations", json={"input": [f"text-{i}" for i in range(count)]}
        )

        assert response.status_code == 200, response.text
        assert len(response.json()["results"]) == count
        assert len(stub.requests) == count

    def test_unsupported_image_format_rejected(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """Non PNG/JPEG images are rejected with 400."""
        _stub_client(monkeypatch, _CLEAN_RESPONSE)
        data_uri = f"data:image/webp;base64,{b64encode(_PNG).decode()}"

        response = client.post(
            "/v1/moderations",
            json={"input": [{"type": "image_url", "image_url": {"url": data_uri}}]},
        )

        assert response.status_code == 400
        assert "PNG or JPEG" in response.json()["error"]["message"]

    def test_empty_input_list_rejected(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """An empty `input` list is rejected with 400, not classified as zero results."""
        _stub_client(monkeypatch, _CLEAN_RESPONSE)

        response = client.post("/v1/moderations", json={"input": []})

        assert response.status_code == 400

    def test_non_content_policy_hits_flag_without_categories(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """Topic/word policy hits flag the input with all categories false."""
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

        response = client.post("/v1/moderations", json={"input": "some text"})

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is True
        assert not any(result["categories"].values())
        assert all(score == 0.0 for score in result["category_scores"].values())

    def test_requests_full_output_scope(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """apply_guardrail is called with outputScope=FULL to surface non-flagged confidences."""
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)

        client.post("/v1/moderations", json={"input": "some text"})

        (request,) = stub.requests
        assert request["outputScope"] == "FULL"

    def test_non_detected_filter_confidence_is_surfaced(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """A non-detected filter entry (only returned under FULL scope) reports its real confidence."""
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

        response = client.post("/v1/moderations", json={"input": "some text"})

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is False
        assert result["categories"]["harassment"] is False
        # Non-zero even though the category was never flagged.
        assert result["category_scores"]["harassment"] == 0.5

    def test_empty_string_input_is_clean_without_aws_call(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """An exactly-empty string yields an unflagged result and no AWS call."""
        stub, _ = _stub_client(monkeypatch, _FLAGGED_RESPONSE)

        response = client.post("/v1/moderations", json={"input": ""})

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is False
        assert not any(result["categories"].values())
        assert all(score == 0.0 for score in result["category_scores"].values())
        applied = result["category_applied_input_types"]
        assert len(applied) == 13
        assert all(value == ["text"] for value in applied.values())
        assert not stub.requests

    def test_empty_string_among_inputs_only_skips_itself(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """Non-empty siblings of an empty string are still classified."""
        stub, _ = _stub_client(monkeypatch, _CLEAN_RESPONSE)

        response = client.post("/v1/moderations", json={"input": ["", "hello"]})

        assert response.status_code == 200, response.text
        assert len(response.json()["results"]) == 2
        (request,) = stub.requests
        assert request["content"] == [{"text": {"text": "hello"}}]

    def test_empty_text_part_is_clean_without_aws_call(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """An empty {"type": "text"} part short-circuits like a plain string."""
        stub, _ = _stub_client(monkeypatch, _FLAGGED_RESPONSE)

        response = client.post(
            "/v1/moderations", json={"input": [{"type": "text", "text": ""}]}
        )

        assert response.status_code == 200, response.text
        assert response.json()["results"][0]["flagged"] is False
        assert not stub.requests

    def test_missing_or_wrong_bearer_returns_401_envelope(
        self, monkeypatch: pytest.MonkeyPatch, configured_guardrail: None
    ) -> None:
        """Unauthenticated requests get the OpenAI 401 error envelope."""
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
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """Guardrail moderation records billed text units per input, summed."""
        from stdapi import monitoring  # noqa: PLC0415

        _stub_client(monkeypatch, _CLEAN_RESPONSE)
        written: list[dict[str, Any]] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)

        response = client.post(
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
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """Guardrail image moderation records one input image per input."""
        from stdapi import monitoring  # noqa: PLC0415

        _stub_client(monkeypatch, _CLEAN_RESPONSE)
        written: list[dict[str, Any]] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)
        data_uri = f"data:image/png;base64,{b64encode(_PNG).decode()}"

        response = client.post(
            "/v1/moderations",
            json={"input": [{"type": "image_url", "image_url": {"url": data_uri}}]},
        )

        assert response.status_code == 200, response.text
        (request_log,) = [w for w in written if w.get("type") == "request"]
        (entry,) = request_log["usage"]
        assert entry["model"] == "gr123:1"
        assert entry["input_images"] == 1
        assert "text_units" not in entry


@pytest.mark.local
class TestModerationsSdkParity:
    """Pins against the installed openai SDK to catch upstream category drift."""

    def test_category_json_keys_match_the_installed_sdk(self) -> None:
        """Category and applied-input-type JSON keys match openai's Moderation submodels."""
        from openai.types.moderation import Categories as SdkCategories  # noqa: PLC0415
        from openai.types.moderation import (  # noqa: PLC0415
            CategoryAppliedInputTypes as SdkCategoryAppliedInputTypes,
        )

        def _json_keys(fields: dict[str, Any]) -> set[str]:
            return {info.alias or name for name, info in fields.items()}

        sdk_categories = _json_keys(SdkCategories.model_fields)
        assert set(ALL_CATEGORIES) == sdk_categories
        assert _json_keys(ModerationCategories.model_fields) == sdk_categories
        assert _json_keys(
            ModerationCategoryAppliedInputTypes.model_fields
        ) == _json_keys(SdkCategoryAppliedInputTypes.model_fields)

    def test_image_capable_categories_match_the_installed_sdk(self) -> None:
        """`IMAGE_CATEGORIES` matches the categories accepting "image" in the installed SDK."""
        from typing import get_args  # noqa: PLC0415

        from openai.types.moderation import (  # noqa: PLC0415
            CategoryAppliedInputTypes as SdkCategoryAppliedInputTypes,
        )

        sdk_image_categories = {
            info.alias or name
            for name, info in SdkCategoryAppliedInputTypes.model_fields.items()
            if "image" in get_args(get_args(info.annotation)[0])
        }
        assert sdk_image_categories == IMAGE_CATEGORIES


@pytest.mark.local
class TestModerationInputLimit:
    """The input array is capped to bound per-element AWS moderation calls."""

    def test_input_at_the_limit_is_accepted(self) -> None:
        """An input array of exactly the maximum length validates."""
        params = ModerationCreateParams.model_validate({"input": ["x"] * 2048})
        assert isinstance(params.input, list)
        assert len(params.input) == 2048

    def test_input_over_the_limit_is_rejected(self) -> None:
        """An input array beyond the maximum length is rejected."""
        with pytest.raises(ValidationError):
            ModerationCreateParams.model_validate({"input": ["x"] * 2049})


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
    """POST /v1/moderations: Amazon Comprehend toxicity backend."""

    @pytest.fixture(autouse=True)
    def _no_ambient_guardrail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Ensure no guardrail from the environment leaks into these tests."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", None)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", None)

    def test_default_backend_without_guardrail(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a configured guardrail, moderation runs on Comprehend."""
        batches = _stub_toxicity(monkeypatch, [[_TOXIC_RESULT]])

        response = client.post("/v1/moderations", json={"input": "threatening text"})

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["model"] == "amazon.comprehend-toxicity"
        (result,) = body["results"]
        assert result["flagged"] is True
        assert result["categories"]["violence"] is True
        assert result["categories"]["harassment"] is False
        assert result["category_scores"]["violence"] == 0.91
        applied = result["category_applied_input_types"]
        assert len(applied) == 13
        assert all(value == ["text"] for value in applied.values())
        assert batches == [["threatening text"]]

    def test_inputs_exceeding_batch_size_are_all_classified_in_order(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """More inputs than the concurrency batch size yield one correctly-mapped result each."""
        count = 2 * openai_moderations._INPUT_BATCH_SIZE + 5  # noqa: SLF001
        texts = [f"text-{i}" for i in range(count)]
        toxic_indices = {2, 7, count - 1}

        class _StubComprehendClient:
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

        response = client.post("/v1/moderations", json={"input": texts})

        assert response.status_code == 200, response.text
        results = response.json()["results"]
        assert len(results) == count
        for index, result in enumerate(results):
            assert result["flagged"] is (index in toxic_indices), index

    def test_usage_records_comprehend_units_and_region(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Comprehend moderation records character-count units with the per-call minimum."""
        from stdapi import monitoring  # noqa: PLC0415
        from stdapi.aws import service_regions  # noqa: PLC0415

        _stub_toxicity(monkeypatch, [[_CLEAN_RESULT]])
        written: list[dict[str, Any]] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)

        response = client.post("/v1/moderations", json={"input": "hi"})

        assert response.status_code == 200, response.text
        (request_log,) = [w for w in written if w.get("type") == "request"]
        (entry,) = request_log["usage"]
        assert entry["service"] == "comprehend"
        assert entry["model"] == "amazon.comprehend-toxicity"
        assert entry["region"] == service_regions(SETTINGS.aws_comprehend_region)[0]
        # ceil(len("hi") / 100) = 1, below the 3-unit per-call minimum.
        assert entry["comprehend_units"] == 3

    def test_mixed_text_and_image_input_rejected_as_a_whole(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pinned current behavior: an image sibling fails the whole batch with 400.

        Comprehend has no partial-success mode: the image element's
        ``ApiError`` propagates out of the batch's ``gather``, so the
        request is rejected even though its text sibling is valid.
        """
        _stub_toxicity(monkeypatch, [[_CLEAN_RESULT]])
        data_uri = f"data:image/png;base64,{b64encode(_PNG).decode()}"

        response = client.post(
            "/v1/moderations",
            json={
                "input": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ]
            },
        )

        assert response.status_code == 400
        assert "guardrail" in response.json()["error"]["message"]

    @pytest.mark.parametrize(
        "model", ["omni-moderation-latest", "text-moderation-latest"]
    )
    def test_openai_model_name_falls_back_to_comprehend(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, model: str
    ) -> None:
        """OpenAI moderation names use Comprehend when no guardrail is set."""
        _stub_toxicity(monkeypatch, [[_CLEAN_RESULT]])

        response = client.post("/v1/moderations", json={"input": "x", "model": model})

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["model"] == model
        assert body["results"][0]["flagged"] is False

    def test_default_guardrail_model_requires_guardrail(
        self, client: TestClient
    ) -> None:
        """amazon.bedrock-runtime-guardrail errors when no guardrail is set."""
        response = client.post(
            "/v1/moderations",
            json={"input": "x", "model": "amazon.bedrock-runtime-guardrail"},
        )

        assert response.status_code == 400
        message = response.json()["error"]["message"]
        assert "administrator" in message
        assert "aws_bedrock_guardrail_identifier" not in message.lower()

    def test_graphic_label_maps_to_violence_graphic(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A GRAPHIC hit serializes under the violence/graphic JSON key."""
        _stub_toxicity(
            monkeypatch,
            [[{"Labels": [{"Name": "GRAPHIC", "Score": 0.8}], "Toxicity": 0.4}]],
        )

        response = client.post("/v1/moderations", json={"input": "x"})

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is True
        assert result["categories"]["violence/graphic"] is True
        assert result["categories"]["violence"] is False
        assert result["category_scores"]["violence/graphic"] == 0.8

    @pytest.mark.parametrize("label", ["HARASSMENT_OR_ABUSE", "INSULT"])
    def test_harassment_labels_map_to_harassment_category(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch, label: str
    ) -> None:
        """HARASSMENT_OR_ABUSE and INSULT both map to harassment, at the threshold."""
        _stub_toxicity(
            monkeypatch,
            [[{"Labels": [{"Name": label, "Score": 0.5}], "Toxicity": 0.1}]],
        )

        response = client.post("/v1/moderations", json={"input": "x"})

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is True
        assert result["categories"]["harassment"] is True
        assert result["category_scores"]["harassment"] == 0.5

    def test_text_object_input(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A {"type": "text"} input part is classified like a plain string."""
        batches = _stub_toxicity(monkeypatch, [[_CLEAN_RESULT]])

        response = client.post(
            "/v1/moderations", json={"input": [{"type": "text", "text": "hello"}]}
        )

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is False
        assert batches == [["hello"]]

    def test_each_input_gets_own_result(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two inputs in one request yield two independent, correctly-mapped results.

        The stub keys its canned result on the segment text rather than
        call order, so the assertion holds regardless of how
        ``asyncio.gather`` schedules the underlying Comprehend calls.
        """
        canned = {"bad": _TOXIC_RESULT, "ok": _CLEAN_RESULT}

        class _StubComprehendClient:
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

        response = client.post("/v1/moderations", json={"input": ["bad", "ok"]})

        assert response.status_code == 200, response.text
        toxic, clean = response.json()["results"]
        assert toxic["flagged"] is True
        assert clean["flagged"] is False

    def test_score_at_threshold_flags(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A label score of exactly 0.5 flags the input (threshold is inclusive)."""
        _stub_toxicity(
            monkeypatch,
            [[{"Labels": [{"Name": "SEXUAL", "Score": 0.5}], "Toxicity": 0.1}]],
        )

        response = client.post("/v1/moderations", json={"input": "x"})

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is True
        assert result["categories"]["sexual"] is True
        assert result["category_scores"]["sexual"] == 0.5

    def test_explicit_comprehend_model_bypasses_guardrail(
        self,
        client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        configured_guardrail: None,
    ) -> None:
        """amazon.comprehend-toxicity is honored even with a guardrail set."""
        batches = _stub_toxicity(monkeypatch, [[_CLEAN_RESULT]])

        response = client.post(
            "/v1/moderations",
            json={"input": "x", "model": "amazon.comprehend-toxicity"},
        )

        assert response.status_code == 200, response.text
        assert response.json()["model"] == "amazon.comprehend-toxicity"
        assert batches == [["x"]]

    def test_unmapped_label_contributes_to_flagged_only(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A PROFANITY hit flags the input without setting any category."""
        _stub_toxicity(
            monkeypatch,
            [[{"Labels": [{"Name": "PROFANITY", "Score": 0.75}], "Toxicity": 0.3}]],
        )

        response = client.post("/v1/moderations", json={"input": "x"})

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is True
        assert not any(result["categories"].values())

    def test_long_text_is_chunked_and_scores_aggregated(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Long inputs split into 1 KB segments, 10 per call, keeping max scores."""
        hate = {"Labels": [{"Name": "HATE_SPEECH", "Score": 0.7}], "Toxicity": 0.6}
        batches = _stub_toxicity(monkeypatch, [[_CLEAN_RESULT], [hate]])
        text = "a" * 10_500

        response = client.post("/v1/moderations", json={"input": text})

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
        """Splitting counts UTF-8 bytes without breaking characters apart."""
        text = char * (1_200 // len(char.encode()))  # 1200 bytes total
        segments = amazon_comprehend._split_toxicity_segments(text)  # noqa: SLF001
        assert "".join(segments) == text
        assert [len(segment.encode()) for segment in segments] == sizes

    def test_overall_toxicity_score_alone_flags(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A high overall toxicity flags the input even with low label scores."""
        _stub_toxicity(
            monkeypatch,
            [[{"Labels": [{"Name": "INSULT", "Score": 0.2}], "Toxicity": 0.9}]],
        )

        response = client.post("/v1/moderations", json={"input": "x"})

        assert response.status_code == 200, response.text
        (result,) = response.json()["results"]
        assert result["flagged"] is True
        assert not any(result["categories"].values())

    def test_empty_text_is_clean_without_api_call(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty input yields a clean result without calling Comprehend."""
        batches = _stub_toxicity(monkeypatch, [[_TOXIC_RESULT]])

        response = client.post("/v1/moderations", json={"input": ""})

        assert response.status_code == 200, response.text
        assert response.json()["results"][0]["flagged"] is False
        assert batches == []

    def test_image_input_requires_guardrail(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Image moderation is rejected on the Comprehend backend."""
        _stub_toxicity(monkeypatch, [[_CLEAN_RESULT]])
        data_uri = f"data:image/png;base64,{b64encode(_PNG).decode()}"

        response = client.post(
            "/v1/moderations",
            json={"input": [{"type": "image_url", "image_url": {"url": data_uri}}]},
        )

        assert response.status_code == 400
        message = response.json()["error"]["message"]
        assert "guardrail" in message
        assert "aws_bedrock_guardrail_identifier" not in message.lower()


@pytest.fixture(scope="module")
def live_guardrail(use_official_api: bool) -> Iterator[str]:
    """Create a temporary guardrail, passed as the explicit moderation model.

    On the official API no guardrail exists; tests use the OpenAI moderation
    model instead. The server (local or --server-url) must allow guardrail
    overrides, which is automatic when no global guardrail is configured.
    Yields the guardrail ARN so the server resolves the guardrail's region.
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
    for _ in range(30):
        if bedrock.get_guardrail(guardrailIdentifier=guardrail_id)["status"] == "READY":
            break
        time.sleep(1)
    else:
        bedrock.delete_guardrail(guardrailIdentifier=guardrail_id)
        pytest.fail("Test guardrail never reached READY status.")
    yield created["guardrailArn"]
    bedrock.delete_guardrail(guardrailIdentifier=guardrail_id)


@pytest.mark.xdist_group("moderations_guardrail")
class TestModerationsLive:
    """Live moderation against a real guardrail or the official API."""

    def test_moderations_endpoint(
        self, openai_client: OpenAI, live_guardrail: str, use_official_api: bool
    ) -> None:
        """Clean and harmful inputs classify as expected."""
        model = "omni-moderation-latest" if use_official_api else live_guardrail
        harmful = (
            "I am going to kill you." if use_official_api else "It has BLOCKWORDXYZ."
        )
        result = openai_client.moderations.create(
            model=model, input=["The weather is nice today.", harmful]
        )
        assert result.model
        clean, flagged = result.results
        assert clean.flagged is False
        assert flagged.flagged is True
        if use_official_api:
            assert flagged.categories.violence is True
        else:
            assert result.model == live_guardrail

    def test_chat_moderation_param(
        self,
        openai_client: OpenAI,
        live_guardrail: str,
        chat_model: str,
        use_official_api: bool,
    ) -> None:
        """The moderation parameter reports guardrail results on the response."""
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


class TestComprehendModerationsLive:
    """Live moderation against Amazon Comprehend toxicity detection."""

    def test_comprehend_moderation(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """Clean and threatening inputs classify as expected."""
        if use_official_api:
            pytest.skip("Amazon Comprehend moderation is gateway-specific")
        result = openai_client.moderations.create(
            model="amazon.comprehend-toxicity",
            input=["The weather is nice today.", "I am going to kill you."],
        )
        assert result.model == "amazon.comprehend-toxicity"
        clean, flagged = result.results
        assert clean.flagged is False
        assert flagged.flagged is True
        assert flagged.categories.violence is True
        assert flagged.category_scores.violence > 0.5
