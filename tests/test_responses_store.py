"""Tests for the AWS Bedrock session-backed stored responses (unit)."""

from datetime import UTC, datetime
from typing import Any

import pytest
from botocore.exceptions import ClientError

from stdapi import responses_store
from stdapi.api_errors import ApiError
from stdapi.config import SETTINGS

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


#: A ResourceNotFoundException as raised by the AWS client.
_NOT_FOUND = ClientError(
    {"Error": {"Code": "ResourceNotFoundException", "Message": "missing"}}, "GetSession"
)


class _StubSessionClient:
    """Stub bedrock-agent-runtime client recording session API calls."""

    def __init__(self, *, missing: bool = False) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self._missing = missing
        self._stored: list[str] = []

    async def create_session(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.requests.append(("create_session", params))
        return {"sessionId": "sess-1"}

    async def create_invocation(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.requests.append(("create_invocation", params))
        return {"invocationId": "inv-1"}

    async def put_invocation_step(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.requests.append(("put_invocation_step", params))
        self._stored.append(params["payload"]["contentBlocks"][0]["text"])
        return {"invocationStepId": f"step-{len(self._stored)}"}

    async def list_invocations(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.requests.append(("list_invocations", params))
        if self._missing:
            raise _NOT_FOUND
        return {"invocationSummaries": [{"invocationId": "inv-1"}]}

    async def list_invocation_steps(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.requests.append(("list_invocation_steps", params))
        # Out-of-order summaries validate time-based reordering.
        summaries = [
            {
                "invocationStepId": f"step-{i + 1}",
                "invocationStepTime": datetime.fromtimestamp(i, tz=UTC),
            }
            for i in reversed(range(len(self._stored)))
        ]
        return {"invocationStepSummaries": summaries}

    async def get_invocation_step(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.requests.append(("get_invocation_step", params))
        index = int(params["invocationStepId"].removeprefix("step-")) - 1
        return {
            "invocationStep": {
                "payload": {"contentBlocks": [{"text": self._stored[index]}]}
            }
        }

    async def end_session(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.requests.append(("end_session", params))
        if self._missing:
            raise _NOT_FOUND
        return {}

    async def delete_session(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.requests.append(("delete_session", params))
        if self._missing:
            raise _NOT_FOUND
        return {}


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> _StubSessionClient:
    """Stub the AWS client and request metadata."""
    client = _StubSessionClient()
    monkeypatch.setattr(responses_store, "get_client", lambda *_: client)
    monkeypatch.setattr(
        responses_store, "build_metadata", lambda **_: {"aws-apn-id": "apn"}
    )
    return client


class TestStoredResponseSessions:
    """Session-backed persistence of stored responses."""

    async def test_create_session_with_tags_and_kms(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The session is created with metadata tags and the configured KMS key."""
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_session_encryption_key_arn", "arn:kms"
        )
        session_id = await responses_store.create_stored_response_session()
        assert session_id == "sess-1"
        (name, params) = stub.requests[0]
        assert name == "create_session"
        assert params == {"tags": {"aws-apn-id": "apn"}, "encryptionKeyArn": "arn:kms"}

    async def test_create_session_without_kms(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a configured KMS key, no encryptionKeyArn is passed."""
        monkeypatch.setattr(SETTINGS, "aws_bedrock_session_encryption_key_arn", None)
        session_id = await responses_store.create_stored_response_session()
        assert session_id == "sess-1"
        (name, params) = stub.requests[0]
        assert name == "create_session"
        assert params == {"tags": {"aws-apn-id": "apn"}}

    async def test_try_create_session_access_denied_returns_none(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An AccessDenied create is ignored with a warning, returning None."""

        async def _denied(**_params: Any) -> dict[str, Any]:  # noqa: ANN401
            raise ClientError(
                {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
                "CreateSession",
            )

        warnings: list[Any] = []
        monkeypatch.setattr(stub, "create_session", _denied)
        monkeypatch.setattr(
            responses_store, "log_error_details", lambda *a, **_k: warnings.extend(a)
        )
        session_id = await responses_store.try_create_stored_response_session(
            "response"
        )
        assert session_id is None
        assert any("session storage" in str(warning) for warning in warnings)

    async def test_try_create_session_other_errors_propagate(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-AccessDenied client errors are not swallowed."""

        async def _throttled(**_params: Any) -> dict[str, Any]:  # noqa: ANN401
            raise ClientError(
                {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
                "CreateSession",
            )

        monkeypatch.setattr(stub, "create_session", _throttled)
        with pytest.raises(ClientError, match="slow down"):
            await responses_store.try_create_stored_response_session("response")

    async def test_try_create_session_success(self, stub: _StubSessionClient) -> None:
        """On success the session ID is returned as-is."""
        session_id = await responses_store.try_create_stored_response_session(
            "response"
        )
        assert session_id == "sess-1"

    async def test_save_and_load_round_trip_with_chunking(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A document larger than the chunk size round-trips across steps."""
        monkeypatch.setattr(responses_store, "_CHUNK_SIZE", 10)
        document = {"response": {"id": "resp-sess-1"}, "input": "x" * 25}

        await responses_store.save_stored_response("resp-sess-1", document)
        steps = [name for name, _ in stub.requests if name == "put_invocation_step"]
        assert len(steps) > 1

        loaded = await responses_store.load_stored_response("resp-sess-1")
        assert loaded == document

    async def test_load_paginates_invocation_steps(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Steps from every nextToken page are loaded, in order."""
        monkeypatch.setattr(responses_store, "_CHUNK_SIZE", 10)
        document = {"response": {"id": "resp-sess-1"}, "input": "z" * 35}
        await responses_store.save_stored_response("resp-sess-1", document)

        real_list_invocation_steps = stub.list_invocation_steps

        async def _paginated(**params: Any) -> dict[str, Any]:  # noqa: ANN401
            page = await real_list_invocation_steps(**params)
            summaries = page["invocationStepSummaries"]
            if "nextToken" not in params:
                return {"invocationStepSummaries": summaries[:2], "nextToken": "page-2"}
            return {"invocationStepSummaries": summaries[2:]}

        monkeypatch.setattr(stub, "list_invocation_steps", _paginated)

        loaded = await responses_store.load_stored_response("resp-sess-1")
        assert loaded == document
        requests = [
            params for name, params in stub.requests if name == "list_invocation_steps"
        ]
        assert len(requests) == 2
        assert "nextToken" not in requests[0]
        assert requests[1]["nextToken"] == "page-2"

    async def test_load_unknown_session_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unknown session surfaces as a stored-response 404."""
        client = _StubSessionClient(missing=True)
        monkeypatch.setattr(responses_store, "get_client", lambda *_: client)
        with pytest.raises(ApiError, match="not found") as exc_info:
            await responses_store.load_stored_response("resp-zzz")
        assert exc_info.value.status == 404

    async def test_load_empty_session_is_not_found(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A session without stored document surfaces as 404."""

        async def _no_invocations(**_params: object) -> dict[str, Any]:
            return {"invocationSummaries": []}

        monkeypatch.setattr(stub, "list_invocations", _no_invocations)
        with pytest.raises(ApiError, match="not found"):
            await responses_store.load_stored_response("resp-sess-1")

    async def test_delete_ends_then_deletes_session(
        self, stub: _StubSessionClient
    ) -> None:
        """Deletion ends the session before deleting it."""
        await responses_store.delete_stored_response("resp-sess-1")
        assert [name for name, _ in stub.requests] == ["end_session", "delete_session"]
        assert stub.requests[1][1] == {"sessionIdentifier": "sess-1"}

    async def test_delete_unknown_session_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deleting an unknown session surfaces as 404."""
        client = _StubSessionClient(missing=True)
        monkeypatch.setattr(responses_store, "get_client", lambda *_: client)
        with pytest.raises(ApiError, match="not found"):
            await responses_store.delete_stored_response("resp-zzz")

    async def test_discard_suppresses_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Best-effort cleanup never raises."""
        client = _StubSessionClient(missing=True)
        monkeypatch.setattr(responses_store, "get_client", lambda *_: client)
        await responses_store.discard_stored_response_session("resp-zzz")
