"""Tests for the OpenAI-compatible /v1/moderations route (unit)."""

from base64 import b64encode
from typing import TYPE_CHECKING, Any

import pytest
from starlette.testclient import TestClient

from stdapi.config import SETTINGS
from stdapi.routes import openai_moderations

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

    monkeypatch.setattr(openai_moderations, "get_client", _get_client)
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
        (request,) = stub.requests
        assert request["guardrailIdentifier"] == "gr123"
        assert request["guardrailVersion"] == "1"
        assert request["source"] == "INPUT"
        assert request["content"] == [{"text": {"text": "some text"}}]

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

    def test_no_guardrail_configured_is_rejected(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without any configured guardrail the request fails with 400."""
        _stub_client(monkeypatch, _CLEAN_RESPONSE)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", None)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", None)

        response = client.post("/v1/moderations", json={"input": "x"})

        assert response.status_code == 400
        assert "No guardrail" in response.json()["error"]["message"]

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


@pytest.fixture(scope="module")
def live_guardrail(use_official_api: bool) -> Iterator[str]:
    """Create a temporary guardrail, passed as the explicit moderation model.

    On the official API no guardrail exists; tests use the OpenAI moderation
    model instead. The server (local or --server-url) must allow guardrail
    overrides, which is automatic when no global guardrail is configured.
    """
    if use_official_api:
        yield ""
        return
    import time  # noqa: PLC0415

    import boto3  # noqa: PLC0415

    region = SETTINGS.aws_bedrock_regions[0]
    bedrock = boto3.client("bedrock", region_name=region)
    created = bedrock.create_guardrail(
        name="stdapi-tests-moderations",
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
    yield guardrail_id
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
        assert moderation.input.flagged is False
        assert moderation.input.model == live_guardrail
