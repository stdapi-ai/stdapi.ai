"""Unit tests for Mantle project-header forwarding on the stored-response proxy.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/projects.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
     stdapi/routes/openai_responses.py:_mantle_stored_response
     stdapi/aws_bedrock_mantle.py:mantle_request_headers
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from stdapi import aws_bedrock_mantle
from stdapi.api_errors import ApiError
from stdapi.aws_bedrock_mantle import (
    MANTLE_PROJECT_VAR,
    MantleError,
    cached_response_surface,
    encode_mantle_response_id,
)
from stdapi.config import SETTINGS
from stdapi.routes import openai_responses
from stdapi.routes.openai_responses import _mantle_stored_response

if TYPE_CHECKING:
    from collections.abc import Generator

    from starlette.testclient import TestClient
    from types_aiobotocore_bedrock.literals import RegionName

pytestmark = pytest.mark.local


@pytest.fixture
def mantle_project() -> Generator[str]:
    """Select a non-default Mantle project for the duration of the test."""
    project = "proj_abc123"
    token = MANTLE_PROJECT_VAR.set(project)
    yield project
    MANTLE_PROJECT_VAR.reset(token)


@pytest.fixture
def _empty_surface_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate the process-wide surface cache from other tests."""
    monkeypatch.setattr(aws_bedrock_mantle, "_SURFACE_CACHE", {})


@pytest.mark.usefixtures("_empty_surface_cache")
async def test_stored_response_proxy_forwards_project_header(
    monkeypatch: pytest.MonkeyPatch, mantle_project: str
) -> None:
    """The stored-response proxy call carries the selected ``OpenAI-Project`` header.

    Mantle stored responses are Region-local and Project-scoped, so a response
    created under one Project cannot be read from another; the proxy must
    therefore re-send the caller's Project scoping on every GET/DELETE/cancel/
    input_items call. Unknown IDs are probed on ``/openai/v1`` first.

    Ref: stdapi/aws_bedrock_mantle.py:cached_response_surface
    """
    calls: list[dict[str, Any]] = []

    async def _fake_request_json(
        region: str, method: str, path: str, *, headers: dict[str, str] | None = None
    ) -> dict[str, Any]:
        calls.append(
            {"region": region, "method": method, "path": path, "headers": headers}
        )
        return {"id": "resp-native-1"}

    monkeypatch.setattr(openai_responses, "request_json", _fake_request_json)

    payload = await _mantle_stored_response("us-east-1", "GET", "resp-native-1")

    assert payload == {"id": "resp-native-1"}
    assert len(calls) == 1, "the first surface answered, so no fallback probe is due"
    assert calls[0]["headers"] == {"OpenAI-Project": mantle_project}
    assert calls[0]["region"] == "us-east-1"
    assert calls[0]["method"] == "GET"
    assert calls[0]["path"] == "/openai/v1/responses/resp-native-1"


@pytest.mark.usefixtures("_empty_surface_cache")
async def test_stored_response_proxy_falls_back_to_the_other_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 404 on the first surface is retried on the other one, and the winner is cached.

    Mantle serves the OpenAI-shaped API under ``/v1`` or ``/openai/v1``
    depending on the model, and a stored response only exists on the surface
    that created it. Without the fallback probe, every response created on
    ``/v1`` becomes unretrievable once the surface cache is cold.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
         stdapi/routes/openai_responses.py:_mantle_stored_response
    """
    calls: list[str] = []

    async def _fake_request_json(
        _region: str,
        _method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,  # noqa: ARG001
    ) -> dict[str, Any]:
        calls.append(path)
        if path.startswith("/openai/v1"):
            msg = "not found"
            raise MantleError(msg, status=404)
        return {"id": "resp-native-1"}

    monkeypatch.setattr(openai_responses, "request_json", _fake_request_json)

    payload = await _mantle_stored_response("us-east-1", "GET", "resp-native-1")

    assert payload == {"id": "resp-native-1"}
    assert calls == [
        "/openai/v1/responses/resp-native-1",
        "/v1/responses/resp-native-1",
    ]
    assert cached_response_surface("resp-native-1") == "/v1"


@pytest.mark.usefixtures("_empty_surface_cache")
async def test_stored_response_proxy_double_404_reports_the_public_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 404 on both surfaces becomes a 404 naming the region-tagged public ID.

    The native Mantle ID is an internal detail: the error must quote the
    ``resp_`` identifier the caller holds, and the surface cache must keep no
    entry for a response that exists nowhere.

    Ref: stdapi/aws_bedrock_mantle.py:encode_mantle_response_id
         stdapi/routes/openai_responses.py:_mantle_stored_response
    """
    calls: list[str] = []

    async def _fake_request_json(
        _region: str,
        _method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,  # noqa: ARG001
    ) -> dict[str, Any]:
        calls.append(path)
        msg = "not found"
        raise MantleError(msg, status=404)

    monkeypatch.setattr(openai_responses, "request_json", _fake_request_json)

    with pytest.raises(ApiError) as excinfo:
        await _mantle_stored_response("us-east-1", "GET", "resp-native-1")

    assert excinfo.value.status == 404
    public_id = encode_mantle_response_id("us-east-1", "resp-native-1")
    assert str(excinfo.value) == f"Response '{public_id}' not found."
    assert "resp-native-1" not in str(excinfo.value)
    assert len(calls) == 2, "both surfaces must be probed before giving up"
    assert cached_response_surface("resp-native-1") is None


@pytest.mark.usefixtures("_empty_surface_cache")
async def test_stored_response_proxy_double_404_uses_the_message_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``missing_msg`` replaces the generic not-found wording on a double 404.

    The input-items listing passes its own message because Bedrock Mantle does
    not serve input-item listings at all, which is a different failure from an
    unknown response ID.

    Ref: stdapi/routes/openai_responses.py:list_response_input_items
    """

    async def _fake_request_json(
        _region: str,
        _method: str,
        _path: str,
        *,
        headers: dict[str, str] | None = None,  # noqa: ARG001
    ) -> dict[str, Any]:
        msg = "not found"
        raise MantleError(msg, status=404)

    monkeypatch.setattr(openai_responses, "request_json", _fake_request_json)

    with pytest.raises(ApiError) as excinfo:
        await _mantle_stored_response(
            "us-east-1",
            "GET",
            "resp-native-1",
            "/input_items",
            missing_msg="listings are not available",
        )

    assert excinfo.value.status == 404
    assert str(excinfo.value) == "listings are not available"


@pytest.mark.usefixtures("_empty_surface_cache")
def test_input_items_route_proxies_a_region_tagged_id(
    app_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``GET /v1/responses/{mantle_id}/input_items`` proxies to the tagged region.

    The public ID embeds the serving region, so the listing route must decode it
    and query that region's native ID rather than the local session store. The
    upstream list is returned as-is: Mantle does not paginate it, so ``limit``
    and ``order`` are not forwarded.

    Ref: stdapi/routes/openai_responses.py:list_response_input_items
    """
    region: RegionName = "us-east-1"
    monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_enabled", True)
    monkeypatch.setattr(SETTINGS, "aws_bedrock_mantle_regions", [region])
    public_id = encode_mantle_response_id(region, "resp-native-1")
    calls: list[tuple[str, str]] = []

    async def _fake_request_json(
        called_region: str,
        _method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,  # noqa: ARG001
    ) -> dict[str, Any]:
        calls.append((called_region, path))
        return {
            "object": "list",
            "data": [
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                    "status": "completed",
                }
            ],
            "first_id": "msg_1",
            "last_id": "msg_1",
            "has_more": False,
        }

    monkeypatch.setattr(openai_responses, "request_json", _fake_request_json)

    response = app_client.get(f"/v1/responses/{public_id}/input_items?limit=5")

    assert response.status_code == 200, response.text
    body = response.json()
    assert [item["id"] for item in body["data"]] == ["msg_1"]
    assert body["has_more"] is False
    assert calls == [(region, "/openai/v1/responses/resp-native-1/input_items")], (
        "the tagged region is queried and Mantle serves the whole list unpaginated"
    )


@pytest.mark.usefixtures("_empty_surface_cache")
async def test_stored_response_proxy_does_not_probe_on_a_non_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-404 upstream error propagates instead of triggering the fallback probe.

    Only a 404 means "wrong surface"; retrying a throttled or failing request on
    the other surface would double the upstream load and mask the real error.

    Ref: stdapi/routes/openai_responses.py:_mantle_stored_response
    """
    calls: list[str] = []

    async def _fake_request_json(
        _region: str,
        _method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,  # noqa: ARG001
    ) -> dict[str, Any]:
        calls.append(path)
        msg = "throttled"
        raise MantleError(msg, status=429)

    monkeypatch.setattr(openai_responses, "request_json", _fake_request_json)

    with pytest.raises(MantleError) as excinfo:
        await _mantle_stored_response("us-east-1", "GET", "resp-native-1")

    assert excinfo.value.status == 429
    assert len(calls) == 1, "a non-404 must not be retried on the other surface"
