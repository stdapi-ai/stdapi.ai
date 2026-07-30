"""Correlation metadata attached to the ``InvokeModel`` APIs.

Converse takes ``requestMetadata`` as a map, but ``InvokeModel`` carries it as
the ``X-Amzn-Bedrock-Request-Metadata`` HTTP header, so the gateway serialises
the map to a JSON string. Because it becomes a header, values must survive
Bedrock's key/value charset and the header size budget without letting a
client-supplied identifier inject header syntax.

Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InvokeModel.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
     stdapi/models/__init__.py:_build_invoke_kwargs
     stdapi/monitoring.py:build_metadata
"""

from typing import TYPE_CHECKING

import pytest
from pydantic_core import from_json

import stdapi.models
from stdapi.models import _build_invoke_kwargs
from stdapi.monitoring import REQUEST_ID, REQUEST_LOG, EventLog
from stdapi.server import SERVER_NAME

if TYPE_CHECKING:
    from collections.abc import Iterator

    from types_aiobotocore_bedrock.literals import RegionName

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Maximum size AWS accepts for the InvokeModel request metadata header.
_MAX_METADATA_SIZE = 8500

#: Maximum length ``build_metadata`` keeps of a client-supplied user identifier.
_MAX_USER_ID_LENGTH = 256


@pytest.fixture
def request_context() -> Iterator[EventLog]:
    """Provide a minimal request context for metadata injection."""
    log: EventLog = {"level": "info"}  # type: ignore[typeddict-item]
    id_token = REQUEST_ID.set("req-1234")
    log_token = REQUEST_LOG.set(log)
    yield log
    REQUEST_LOG.reset(log_token)
    REQUEST_ID.reset(id_token)


@pytest.fixture(autouse=True)
def _stub_model_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve any model ID to itself, without hitting the model catalog."""

    async def resolve(model_id: str, _region: RegionName, **_kwargs: object) -> str:
        """Return the model ID unchanged."""
        return model_id

    monkeypatch.setattr(stdapi.models, "resolve_routed_model_id", resolve)


async def _invoke_metadata() -> dict[str, str]:
    """Build InvokeModel kwargs and return their parsed request metadata."""
    kwargs = await _build_invoke_kwargs(
        "amazon.nova-micro-v1:0", {"prompt": "hi"}, "us-east-1", inference_profile=True
    )
    metadata = kwargs["requestMetadata"]
    # AWS takes this field as a JSON string header, not as a map.
    assert isinstance(metadata, str)
    parsed = from_json(metadata)
    assert isinstance(parsed, dict)
    return parsed


@pytest.mark.usefixtures("request_context")
class TestInvokeRequestMetadata:
    """Correlation metadata forwarded on InvokeModel / InvokeModelWithResponseStream.

    Ref: stdapi/monitoring.py:build_metadata
    """

    async def test_carries_the_request_and_server_correlation_ids(self) -> None:
        """Every invocation is tagged with its stdapi.ai request and server IDs."""
        metadata = await _invoke_metadata()

        assert metadata["stdapi-ai.request_id"] == "req-1234"
        assert metadata["stdapi-ai.server_id"] == SERVER_NAME

    async def test_carries_the_user_id_when_the_client_supplies_one(
        self, request_context: EventLog
    ) -> None:
        """A client-supplied user identifier is attributed to the invocation."""
        request_context["request_user_id"] = "user-42"

        assert (await _invoke_metadata())["stdapi-ai.user_id"] == "user-42"

    async def test_omits_the_user_id_when_the_client_supplies_none(self) -> None:
        """No user key is sent when the request carries no user identifier."""
        metadata = await _invoke_metadata()

        assert "stdapi-ai.user_id" not in metadata
        assert "stdapi-ai.request_id" in metadata, (
            "the omission must be specific to the user key"
        )

    async def test_a_hostile_user_id_cannot_inject_into_the_metadata_header(
        self, request_context: EventLog
    ) -> None:
        """Characters Bedrock rejects are stripped and line breaks stay JSON-escaped.

        CR/LF is inside Bedrock's metadata charset, so it is kept in the map value
        and neutralised by JSON escaping rather than by stripping; only the
        out-of-charset ``<``/``>`` are removed.
        """
        request_context["request_user_id"] = "user<42>\r\nX-Injected: 1"

        kwargs = await _build_invoke_kwargs(
            "amazon.nova-micro-v1:0",
            {"prompt": "hi"},
            "us-east-1",
            inference_profile=True,
        )

        assert not {"<", ">", "\r", "\n"} & set(kwargs["requestMetadata"])
        parsed = from_json(kwargs["requestMetadata"])
        assert isinstance(parsed, dict)
        assert parsed["stdapi-ai.user_id"] == "user42\r\nX-Injected: 1"

    async def test_fits_within_the_bedrock_metadata_header_limit(
        self, request_context: EventLog
    ) -> None:
        """An oversized user identifier is truncated to keep the header within limits."""
        request_context["request_user_id"] = "u" * 10_000

        kwargs = await _build_invoke_kwargs(
            "amazon.nova-micro-v1:0",
            {"prompt": "hi"},
            "us-east-1",
            inference_profile=True,
        )

        assert len(kwargs["requestMetadata"]) <= _MAX_METADATA_SIZE
        parsed = from_json(kwargs["requestMetadata"])
        assert isinstance(parsed, dict)
        assert parsed["stdapi-ai.user_id"] == "u" * _MAX_USER_ID_LENGTH
