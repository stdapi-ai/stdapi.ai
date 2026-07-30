"""Unit tests for Mantle project-header forwarding on the stored-response proxy.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/projects.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html
     stdapi/routes/openai_responses.py:_mantle_stored_response
     stdapi/aws_bedrock_mantle.py:mantle_request_headers
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from stdapi.aws_bedrock_mantle import MANTLE_PROJECT_VAR
from stdapi.routes import openai_responses
from stdapi.routes.openai_responses import _mantle_stored_response

if TYPE_CHECKING:
    from collections.abc import Generator

pytestmark = pytest.mark.local


@pytest.fixture
def mantle_project() -> Generator[str]:
    """Select a non-default Mantle project for the duration of the test."""
    project = "proj_abc123"
    token = MANTLE_PROJECT_VAR.set(project)
    yield project
    MANTLE_PROJECT_VAR.reset(token)


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
