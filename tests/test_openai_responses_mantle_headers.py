"""Unit tests for Mantle project-header forwarding on the stored-response proxy."""

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
    """GET/DELETE/cancel/input_items calls carry the selected OpenAI-Project header."""
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
    assert len(calls) == 1
    assert calls[0]["headers"] == {"OpenAI-Project": mantle_project}
