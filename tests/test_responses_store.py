"""Stored responses and chat completions persisted in AWS Bedrock sessions (unit).

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/sessions.html
     stdapi/responses_store.py
"""

from datetime import UTC, datetime
from typing import Any

import pytest
from botocore.exceptions import ClientError
from pydantic_core import to_json

from stdapi import responses_store
from stdapi.api_errors import ApiError
from stdapi.config import SETTINGS

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local


#: A ResourceNotFoundException as raised by the AWS client.
_NOT_FOUND = ClientError(
    {"Error": {"Code": "ResourceNotFoundException", "Message": "missing"}}, "GetSession"
)

#: A ValidationException as raised by the AWS client for a malformed identifier.
_VALIDATION_ERROR = ClientError(
    {"Error": {"Code": "ValidationException", "Message": "invalid identifier"}},
    "GetSession",
)


class _StubSessionClient:
    """Stub bedrock-agent-runtime client recording session API calls."""

    def __init__(self, *, missing: bool = False) -> None:
        self.requests: list[tuple[str, dict[str, Any]]] = []
        self._missing = missing
        self._stored: list[str] = []
        #: Sessions listed by ``list_sessions``, in scan order.
        self.sessions: list[dict[str, Any]] = []
        #: Tags per session ARN, consumed by ``list_tags_for_resource``.
        self.tags: dict[str, dict[str, str]] = {}

    async def create_session(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.requests.append(("create_session", params))
        return {"sessionId": "sess-1"}

    async def list_sessions(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.requests.append(("list_sessions", params))
        max_results = params["maxResults"]
        start = int(params["nextToken"]) if "nextToken" in params else 0
        end = start + max_results
        page: dict[str, Any] = {"sessionSummaries": self.sessions[start:end]}
        if end < len(self.sessions):
            page["nextToken"] = str(end)
        return page

    async def list_tags_for_resource(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.requests.append(("list_tags_for_resource", params))
        if self._missing:
            raise _NOT_FOUND
        return {"tags": self.tags.get(params["resourceArn"], {})}

    async def get_session(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        self.requests.append(("get_session", params))
        if self._missing:
            raise _NOT_FOUND
        return {"sessionArn": f"arn:{params['sessionIdentifier']}"}

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
        return {
            "invocationSummaries": [
                {
                    "invocationId": "inv-1",
                    "createdAt": datetime.fromtimestamp(0, tz=UTC),
                }
            ]
        }

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
    # Isolate the kind-tag cache from other tests reusing the same session IDs.
    monkeypatch.setattr(responses_store, "_KIND_CACHE", {})
    return client


class TestStoredResponseSessions:
    """Create / save / load / delete over the Bedrock session workflow.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/sessions.html
         stdapi/responses_store.py
    """

    async def test_create_session_with_tags_and_kms(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CreateSession carries the request metadata tags, the kind tag and the KMS key.

        The kind tag is what later lets listing and deletion tell a stored
        response from a stored chat completion, and
        ``aws_bedrock_session_encryption_key_arn`` is forwarded as
        ``encryptionKeyArn`` so the session content is encrypted with a
        customer-managed key.

        Ref: stdapi/responses_store.py:create_stored_response_session
        """
        monkeypatch.setattr(
            SETTINGS, "aws_bedrock_session_encryption_key_arn", "arn:kms"
        )
        session_id = await responses_store.create_stored_response_session("response")
        assert session_id == "sess-1"
        (name, params) = stub.requests[0]
        assert name == "create_session"
        assert params == {
            "tags": {"aws-apn-id": "apn", "stdapi-ai.stored-object": "response"},
            "encryptionKeyArn": "arn:kms",
        }

    async def test_create_session_without_kms(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a configured KMS key, ``encryptionKeyArn`` is omitted from CreateSession.

        The parameter is left out of the call entirely rather than sent as
        null, and the kind tag records that this session holds a chat
        completion rather than a response.

        Ref: stdapi/responses_store.py:create_stored_response_session
        """
        monkeypatch.setattr(SETTINGS, "aws_bedrock_session_encryption_key_arn", None)
        session_id = await responses_store.create_stored_response_session(
            "chat_completion"
        )
        assert session_id == "sess-1"
        (name, params) = stub.requests[0]
        assert name == "create_session"
        assert params == {
            "tags": {"aws-apn-id": "apn", "stdapi-ai.stored-object": "chat_completion"}
        }

    async def test_try_create_session_access_denied_returns_none(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ``AccessDeniedException`` on CreateSession returns None and logs a warning.

        AccessDenied means session storage is not enabled on this server, so
        the request must proceed with ``store`` ignored instead of failing;
        the warning tells the administrator which permission is missing.

        Ref: stdapi/responses_store.py:try_create_stored_response_session
        """

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
        """A ``ThrottlingException`` on CreateSession propagates instead of returning None.

        Only AccessDenied means "storage not enabled"; every other client
        error is a real failure the caller must see.

        Ref: stdapi/responses_store.py:try_create_stored_response_session
        """

        async def _throttled(**_params: Any) -> dict[str, Any]:  # noqa: ANN401
            raise ClientError(
                {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
                "CreateSession",
            )

        monkeypatch.setattr(stub, "create_session", _throttled)
        with pytest.raises(ClientError) as exc_info:
            await responses_store.try_create_stored_response_session("response")
        assert exc_info.value.response["Error"]["Code"] == "ThrottlingException"
        assert exc_info.value.operation_name == "CreateSession"
        assert "slow down" in str(exc_info.value)

    async def test_try_create_session_success(self, stub: _StubSessionClient) -> None:
        """On success the created session ID is returned as-is, with no retry.

        Ref: stdapi/responses_store.py:try_create_stored_response_session
        """
        session_id = await responses_store.try_create_stored_response_session(
            "response"
        )
        assert session_id == "sess-1"
        assert [name for name, _ in stub.requests] == ["create_session"]

    async def test_save_and_load_round_trip_with_chunking(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A document larger than the chunk size round-trips across several steps.

        Invocation-step payloads are content blocks, so a document is split
        over one PutInvocationStep per chunk, all under the same invocation
        and with strictly increasing step times so the read can reorder them.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/sessions.html
             stdapi/responses_store.py:save_stored_response
        """
        monkeypatch.setattr(responses_store, "_CHUNK_SIZE", 10)
        document = {
            "response": {"id": "resp-sess-1", "object": "response"},
            "input": "x" * 25,
        }

        await responses_store.save_stored_response("resp-sess-1", document)
        steps = [
            params for name, params in stub.requests if name == "put_invocation_step"
        ]
        assert len(steps) > 1
        assert {step["invocationIdentifier"] for step in steps} == {"inv-1"}
        step_times = [step["invocationStepTime"] for step in steps]
        assert step_times == sorted(step_times)
        assert len(set(step_times)) == len(step_times)

        loaded = await responses_store.load_stored_response("resp-sess-1", "response")
        assert loaded == document

    async def test_save_chunks_by_utf8_bytes_not_characters(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multibyte content is chunked by UTF-8 byte count and round-trips exactly.

        The chunk limit bounds bytes, not characters, and a boundary
        backtracks off a continuation byte so no chunk splits a code point:
        concatenating the stored chunks reproduces the serialized document.

        Ref: stdapi/responses_store.py:_iter_utf8_chunks
        """
        monkeypatch.setattr(responses_store, "_CHUNK_SIZE", 10)
        document = {
            "response": {"id": "resp-sess-1", "object": "response"},
            "input": "€" * 20 + "文" * 5,  # 3-byte UTF-8 characters
        }

        await responses_store.save_stored_response("resp-sess-1", document)
        chunks = [
            params["payload"]["contentBlocks"][0]["text"]
            for name, params in stub.requests
            if name == "put_invocation_step"
        ]
        assert len(chunks) > 1
        assert all(len(chunk.encode()) <= 10 for chunk in chunks)
        assert "".join(chunks) == to_json(document).decode()

        loaded = await responses_store.load_stored_response("resp-sess-1", "response")
        assert loaded == document

    async def test_save_chunk_boundary_at_exact_multiple_of_chunk_size(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A byte length that is an exact multiple of the chunk size yields no empty chunk.

        The boundary case must produce exactly ``length / limit`` chunks and
        never a trailing empty one, since an empty text content block would be
        written as a useless invocation step.

        Ref: stdapi/responses_store.py:_iter_utf8_chunks
        """
        text = "x"
        document: dict[str, Any] = {
            "response": {"id": "resp-sess-1", "object": "response"},
            "input": text,
        }
        while len(to_json(document)) % 3:
            text += "x"
            document["input"] = text
        monkeypatch.setattr(responses_store, "_CHUNK_SIZE", len(to_json(document)) // 3)

        await responses_store.save_stored_response("resp-sess-1", document)
        chunks = [
            params["payload"]["contentBlocks"][0]["text"]
            for name, params in stub.requests
            if name == "put_invocation_step"
        ]
        assert len(chunks) == 3
        assert all(chunk for chunk in chunks)

        loaded = await responses_store.load_stored_response("resp-sess-1", "response")
        assert loaded == document

    async def test_load_paginates_invocation_steps(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Steps from every ListInvocationSteps page are loaded, in step-time order.

        The stub returns the step summaries newest-first and split over two
        pages, so a reader that stopped at the first page or kept the returned
        order would rebuild a corrupt document.

        Ref: stdapi/responses_store.py:_load_invocation_document
        """
        monkeypatch.setattr(responses_store, "_CHUNK_SIZE", 10)
        document = {
            "response": {"id": "resp-sess-1", "object": "response"},
            "input": "z" * 35,
        }
        await responses_store.save_stored_response("resp-sess-1", document)

        real_list_invocation_steps = stub.list_invocation_steps

        async def _paginated(**params: Any) -> dict[str, Any]:  # noqa: ANN401
            page = await real_list_invocation_steps(**params)
            summaries = page["invocationStepSummaries"]
            if "nextToken" not in params:
                return {"invocationStepSummaries": summaries[:2], "nextToken": "page-2"}
            return {"invocationStepSummaries": summaries[2:]}

        monkeypatch.setattr(stub, "list_invocation_steps", _paginated)

        loaded = await responses_store.load_stored_response("resp-sess-1", "response")
        assert loaded == document
        requests = [
            params for name, params in stub.requests if name == "list_invocation_steps"
        ]
        assert len(requests) == 2
        assert "nextToken" not in requests[0]
        assert requests[1]["nextToken"] == "page-2"

    async def test_load_session_vanishing_during_step_fetch_is_not_found(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A session deleted mid-read surfaces as a 404, not a raw ClientError.

        Ref: stdapi/responses_store.py:_stored_document_or_none
        """
        document = {"response": {"id": "resp-sess-1", "object": "response"}}
        await responses_store.save_stored_response("resp-sess-1", document)

        async def _vanished(**_params: Any) -> dict[str, Any]:  # noqa: ANN401
            raise _NOT_FOUND

        monkeypatch.setattr(stub, "get_invocation_step", _vanished)
        with pytest.raises(ApiError) as exc_info:
            await responses_store.load_stored_response("resp-sess-1", "response")
        assert exc_info.value.status == 404
        assert str(exc_info.value) == "Response with id 'resp-sess-1' not found."

    async def test_load_unknown_session_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unknown session surfaces as a 404 naming the requested response ID.

        Ref: stdapi/responses_store.py:load_stored_response
        """
        client = _StubSessionClient(missing=True)
        monkeypatch.setattr(responses_store, "get_client", lambda *_: client)
        with pytest.raises(ApiError) as exc_info:
            await responses_store.load_stored_response("resp-zzz", "response")
        assert exc_info.value.status == 404
        assert str(exc_info.value) == "Response with id 'resp-zzz' not found."

    async def test_load_malformed_identifier_is_not_found(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pattern-valid but AWS-invalid ID (ValidationException) 404s, not 400 or 500.

        An ID matching the route pattern can still fail Bedrock's own session
        identifier validation; such an ID can never name a stored object, so
        it is reported as not found rather than as an upstream error.

        Ref: stdapi/responses_store.py:_stored_document_or_none
        """

        async def _invalid(**_params: Any) -> dict[str, Any]:  # noqa: ANN401
            raise _VALIDATION_ERROR

        monkeypatch.setattr(stub, "list_invocations", _invalid)
        with pytest.raises(ApiError) as exc_info:
            await responses_store.load_stored_response("resp-notauuid", "response")
        assert exc_info.value.status == 404
        assert str(exc_info.value) == "Response with id 'resp-notauuid' not found."

    async def test_load_empty_session_is_not_found(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An existing session holding no invocation surfaces as a 404.

        Ref: stdapi/responses_store.py:_stored_document_or_none
        """

        async def _no_invocations(**_params: object) -> dict[str, Any]:
            return {"invocationSummaries": []}

        monkeypatch.setattr(stub, "list_invocations", _no_invocations)
        with pytest.raises(ApiError) as exc_info:
            await responses_store.load_stored_response("resp-sess-1", "response")
        assert exc_info.value.status == 404
        assert str(exc_info.value) == "Response with id 'resp-sess-1' not found."

    async def test_delete_ends_then_deletes_session(
        self, stub: _StubSessionClient
    ) -> None:
        """Deletion checks the kind tag, then ends the session before deleting it.

        A session is ended before it is deleted, and the kind check is a single
        tag lookup on the session ARN returned by GetSession.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/sessions.html
             stdapi/responses_store.py:delete_stored_response
        """
        await responses_store.delete_stored_response("resp-sess-1", "response")
        assert [name for name, _ in stub.requests] == [
            "get_session",
            "list_tags_for_resource",
            "end_session",
            "delete_session",
        ]
        assert stub.requests[-1][1] == {"sessionIdentifier": "sess-1"}

    @pytest.mark.parametrize("code", ["ConflictException", "ValidationException"])
    async def test_delete_tolerates_already_ended_session(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch, code: str
    ) -> None:
        """A state error from EndSession (e.g. already ended) still deletes the session.

        ``ConflictException`` and ``ValidationException`` on EndSession are
        tolerated: EndSession is attempted exactly once and DeleteSession then
        runs on the same session identifier, surfacing any real problem itself.

        Ref: stdapi/responses_store.py:delete_stored_response
        """
        end_attempts: list[dict[str, Any]] = []

        async def _already_ended(**params: Any) -> dict[str, Any]:  # noqa: ANN401
            end_attempts.append(params)
            raise ClientError(
                {"Error": {"Code": code, "Message": "already ended"}}, "EndSession"
            )

        monkeypatch.setattr(stub, "end_session", _already_ended)
        await responses_store.delete_stored_response("resp-sess-1", "response")
        assert end_attempts == [{"sessionIdentifier": "sess-1"}]
        # The raising stub replaces the recording one, so the calls the client
        # completed are the kind lookup then the delete that follows the failure.
        assert [name for name, _ in stub.requests] == [
            "get_session",
            "list_tags_for_resource",
            "delete_session",
        ]
        assert stub.requests[-1][1] == {"sessionIdentifier": "sess-1"}

    async def test_delete_propagates_end_session_throttling(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A ThrottlingException from EndSession propagates and skips the delete.

        Only the tolerated state errors defer to DeleteSession; a throttled
        EndSession must not be mistaken for "already ended" and must leave the
        session intact so the caller can retry.

        Ref: stdapi/responses_store.py:delete_stored_response
        """

        async def _throttled(**_params: Any) -> dict[str, Any]:  # noqa: ANN401
            raise ClientError(
                {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
                "EndSession",
            )

        monkeypatch.setattr(stub, "end_session", _throttled)
        with pytest.raises(ClientError) as exc_info:
            await responses_store.delete_stored_response("resp-sess-1", "response")
        assert exc_info.value.response["Error"]["Code"] == "ThrottlingException"
        assert "slow down" in str(exc_info.value)
        assert "delete_session" not in [name for name, _ in stub.requests]

    async def test_delete_malformed_identifier_is_not_found(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pattern-valid but AWS-invalid ID (ValidationException) 404s on delete.

        The tag lookup tolerates it (as it does a missing session), the end
        call defers it, and ``delete_session``'s own rejection maps to 404.

        Ref: stdapi/responses_store.py:_not_found_as_404
        """

        async def _invalid(**_params: Any) -> dict[str, Any]:  # noqa: ANN401
            raise _VALIDATION_ERROR

        monkeypatch.setattr(stub, "get_session", _invalid)
        monkeypatch.setattr(stub, "end_session", _invalid)
        monkeypatch.setattr(stub, "delete_session", _invalid)
        with pytest.raises(ApiError) as exc_info:
            await responses_store.delete_stored_response("resp-notauuid", "response")
        assert exc_info.value.status == 404
        assert str(exc_info.value) == "Response with id 'resp-notauuid' not found."

    async def test_delete_does_not_load_the_full_document(
        self, stub: _StubSessionClient
    ) -> None:
        """Deletion never reads invocations: the kind check uses only the tag call.

        Reading the whole document to check its kind would add latency and
        throttling exposure for no benefit, since the session tag already
        records the kind.

        Ref: stdapi/responses_store.py:_session_kind_tag_or_none
        """
        document = {"response": {"id": "resp-sess-1", "object": "response"}}
        await responses_store.save_stored_response("resp-sess-1", document)
        await responses_store.delete_stored_response("resp-sess-1", "response")
        called = {name for name, _ in stub.requests}
        assert "list_tags_for_resource" in called
        assert not called & {
            "list_invocations",
            "list_invocation_steps",
            "get_invocation_step",
        }

    async def test_delete_unknown_session_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deleting an unknown session surfaces as a 404 naming the response ID.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/delete
             stdapi/responses_store.py:delete_stored_response
        """
        client = _StubSessionClient(missing=True)
        monkeypatch.setattr(responses_store, "get_client", lambda *_: client)
        with pytest.raises(ApiError) as exc_info:
            await responses_store.delete_stored_response("resp-zzz", "response")
        assert exc_info.value.status == 404
        assert str(exc_info.value) == "Response with id 'resp-zzz' not found."

    async def test_discard_suppresses_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Best-effort cleanup attempts the delete and swallows its 404.

        Discard runs on a failed generation, where the response has already
        failed: the delete is attempted (EndSession here raises
        ``ResourceNotFoundException``) and the resulting 404 must not replace
        the original error.

        Ref: stdapi/responses_store.py:discard_stored_response_session
        """
        client = _StubSessionClient(missing=True)
        monkeypatch.setattr(responses_store, "get_client", lambda *_: client)
        await responses_store.discard_stored_response_session("resp-zzz", "response")
        assert [name for name, _ in client.requests] == ["get_session", "end_session"]

    async def test_load_reads_latest_invocation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reading picks the invocation with the latest ``createdAt``.

        Each save appends a new invocation rather than overwriting, so the
        newest one is the visible document.

        Ref: stdapi/responses_store.py:_stored_document_or_none
        """
        documents = {
            "inv-old": {"response": {"id": "stale", "object": "response"}},
            "inv-new": {"response": {"id": "fresh", "object": "response"}},
        }

        class _MultiInvocationClient:
            async def list_invocations(self, **_params: Any) -> dict[str, Any]:  # noqa: ANN401
                return {
                    "invocationSummaries": [
                        {
                            "invocationId": "inv-old",
                            "createdAt": datetime(2024, 1, 1, tzinfo=UTC),
                        },
                        {
                            "invocationId": "inv-new",
                            "createdAt": datetime(2024, 6, 1, tzinfo=UTC),
                        },
                    ]
                }

            async def list_invocation_steps(self, **_params: Any) -> dict[str, Any]:  # noqa: ANN401
                return {
                    "invocationStepSummaries": [
                        {
                            "invocationStepId": "step-1",
                            "invocationStepTime": datetime.fromtimestamp(0, tz=UTC),
                        }
                    ]
                }

            async def get_invocation_step(
                self,
                *,
                invocationIdentifier: str,  # noqa: N803
                **_params: Any,  # noqa: ANN401
            ) -> dict[str, Any]:
                text = to_json(documents[invocationIdentifier]).decode()
                return {
                    "invocationStep": {"payload": {"contentBlocks": [{"text": text}]}}
                }

        monkeypatch.setattr(
            responses_store, "get_client", lambda *_: _MultiInvocationClient()
        )
        loaded = await responses_store.load_stored_response("resp-sess-1", "response")
        assert loaded == documents["inv-new"]

    async def test_load_paginates_invocations_before_picking_latest(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A newer invocation on a later ListInvocations page beats a stale first-page one.

        Page order is not assumed to be creation order, so every page is
        collected before the latest invocation is chosen.

        Ref: stdapi/responses_store.py:_stored_document_or_none
        """
        documents = {
            "inv-old": {"response": {"id": "stale", "object": "response"}},
            "inv-new": {"response": {"id": "fresh", "object": "response"}},
        }

        class _PaginatedInvocationsClient:
            async def list_invocations(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
                if "nextToken" not in params:
                    return {
                        "invocationSummaries": [
                            {
                                "invocationId": "inv-old",
                                "createdAt": datetime(2024, 1, 1, tzinfo=UTC),
                            }
                        ],
                        "nextToken": "page-2",
                    }
                return {
                    "invocationSummaries": [
                        {
                            "invocationId": "inv-new",
                            "createdAt": datetime(2024, 6, 1, tzinfo=UTC),
                        }
                    ]
                }

            async def list_invocation_steps(self, **_params: Any) -> dict[str, Any]:  # noqa: ANN401
                return {
                    "invocationStepSummaries": [
                        {
                            "invocationStepId": "step-1",
                            "invocationStepTime": datetime.fromtimestamp(0, tz=UTC),
                        }
                    ]
                }

            async def get_invocation_step(
                self,
                *,
                invocationIdentifier: str,  # noqa: N803
                **_params: Any,  # noqa: ANN401
            ) -> dict[str, Any]:
                text = to_json(documents[invocationIdentifier]).decode()
                return {
                    "invocationStep": {"payload": {"contentBlocks": [{"text": text}]}}
                }

        monkeypatch.setattr(
            responses_store, "get_client", lambda *_: _PaginatedInvocationsClient()
        )
        loaded = await responses_store.load_stored_response("resp-sess-1", "response")
        assert loaded == documents["inv-new"]

    async def test_load_falls_back_when_latest_invocation_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stepless latest invocation falls back to the last-good invocation.

        An update interrupted between CreateInvocation and its first
        PutInvocationStep must not hide the previously stored document.

        Ref: stdapi/responses_store.py:_load_invocation_document
        """
        documents = {"inv-old": {"response": {"id": "stale", "object": "response"}}}

        class _EmptyLatestClient:
            async def list_invocations(self, **_params: Any) -> dict[str, Any]:  # noqa: ANN401
                return {
                    "invocationSummaries": [
                        {
                            "invocationId": "inv-old",
                            "createdAt": datetime(2024, 1, 1, tzinfo=UTC),
                        },
                        {
                            "invocationId": "inv-new",
                            "createdAt": datetime(2024, 6, 1, tzinfo=UTC),
                        },
                    ]
                }

            async def list_invocation_steps(
                self,
                *,
                invocationIdentifier: str,  # noqa: N803
                **_params: Any,  # noqa: ANN401
            ) -> dict[str, Any]:
                if invocationIdentifier == "inv-new":
                    return {"invocationStepSummaries": []}
                return {
                    "invocationStepSummaries": [
                        {
                            "invocationStepId": "step-1",
                            "invocationStepTime": datetime.fromtimestamp(0, tz=UTC),
                        }
                    ]
                }

            async def get_invocation_step(
                self,
                *,
                invocationIdentifier: str,  # noqa: N803
                **_params: Any,  # noqa: ANN401
            ) -> dict[str, Any]:
                text = to_json(documents[invocationIdentifier]).decode()
                return {
                    "invocationStep": {"payload": {"contentBlocks": [{"text": text}]}}
                }

        monkeypatch.setattr(
            responses_store, "get_client", lambda *_: _EmptyLatestClient()
        )
        loaded = await responses_store.load_stored_response("resp-sess-1", "response")
        assert loaded == documents["inv-old"]

    async def test_load_falls_back_when_latest_invocation_is_corrupt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A truncated, unparsable latest invocation falls back to the previous one.

        A read racing a partially written invocation reassembles invalid JSON;
        that invocation is skipped instead of failing the whole read.

        Ref: stdapi/responses_store.py:_load_invocation_document
        """
        documents = {"inv-old": {"response": {"id": "stale", "object": "response"}}}

        class _CorruptLatestClient:
            async def list_invocations(self, **_params: Any) -> dict[str, Any]:  # noqa: ANN401
                return {
                    "invocationSummaries": [
                        {
                            "invocationId": "inv-old",
                            "createdAt": datetime(2024, 1, 1, tzinfo=UTC),
                        },
                        {
                            "invocationId": "inv-new",
                            "createdAt": datetime(2024, 6, 1, tzinfo=UTC),
                        },
                    ]
                }

            async def list_invocation_steps(self, **_params: Any) -> dict[str, Any]:  # noqa: ANN401
                return {
                    "invocationStepSummaries": [
                        {
                            "invocationStepId": "step-1",
                            "invocationStepTime": datetime.fromtimestamp(0, tz=UTC),
                        }
                    ]
                }

            async def get_invocation_step(
                self,
                *,
                invocationIdentifier: str,  # noqa: N803
                **_params: Any,  # noqa: ANN401
            ) -> dict[str, Any]:
                text = (
                    '{"response": {"id": "fresh"'  # truncated: invalid JSON
                    if invocationIdentifier == "inv-new"
                    else to_json(documents[invocationIdentifier]).decode()
                )
                return {
                    "invocationStep": {"payload": {"contentBlocks": [{"text": text}]}}
                }

        monkeypatch.setattr(
            responses_store, "get_client", lambda *_: _CorruptLatestClient()
        )
        loaded = await responses_store.load_stored_response("resp-sess-1", "response")
        assert loaded == documents["inv-old"]

    async def test_load_is_not_found_when_all_invocations_are_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every invocation is stepless, the read surfaces as a 404.

        Ref: stdapi/responses_store.py:load_stored_response
        """

        class _AllEmptyClient:
            async def list_invocations(self, **_params: Any) -> dict[str, Any]:  # noqa: ANN401
                return {
                    "invocationSummaries": [
                        {
                            "invocationId": "inv-1",
                            "createdAt": datetime(2024, 1, 1, tzinfo=UTC),
                        },
                        {
                            "invocationId": "inv-2",
                            "createdAt": datetime(2024, 6, 1, tzinfo=UTC),
                        },
                    ]
                }

            async def list_invocation_steps(self, **_params: Any) -> dict[str, Any]:  # noqa: ANN401
                return {"invocationStepSummaries": []}

        monkeypatch.setattr(responses_store, "get_client", lambda *_: _AllEmptyClient())
        with pytest.raises(ApiError, match="not found") as exc_info:
            await responses_store.load_stored_response("resp-sess-1", "response")
        assert exc_info.value.status == 404

    async def test_load_null_response_field_is_not_found(
        self, stub: _StubSessionClient
    ) -> None:
        """A document with a null ``response`` field lacks the expected shape and 404s.

        Ref: stdapi/responses_store.py:_kind_mismatches
        """
        document = {"response": None, "input": "hello"}
        await responses_store.save_stored_response("resp-sess-1", document)
        with pytest.raises(ApiError, match="not found") as exc_info:
            await responses_store.load_stored_response("resp-sess-1", "response")
        assert exc_info.value.status == 404

    async def test_load_foreign_document_without_object_field_is_not_found(
        self, stub: _StubSessionClient
    ) -> None:
        """A step payload parsing as an unrelated JSON object 404s.

        A session created by another tool can hold arbitrary JSON; without a
        ``response.object`` field declaring the kind it is not a stored object.

        Ref: stdapi/responses_store.py:_document_kind
        """
        document = {"unrelated": "data"}
        await responses_store.save_stored_response("resp-sess-1", document)
        with pytest.raises(ApiError, match="not found") as exc_info:
            await responses_store.load_stored_response("resp-sess-1", "response")
        assert exc_info.value.status == 404

    async def test_load_kind_mismatch_is_not_found(
        self, stub: _StubSessionClient
    ) -> None:
        """A document stored as a chat completion 404s when loaded as a response.

        The ``response.object`` field is the document's self-declared kind, so
        a chat completion is never readable through the Responses routes even
        when its session ID is guessed.

        Ref: stdapi/responses_store.py:_kind_mismatches
        """
        document = {"response": {"id": "chatcmpl-sess-1", "object": "chat.completion"}}
        await responses_store.save_stored_response("resp-sess-1", document)
        with pytest.raises(ApiError) as exc_info:
            await responses_store.load_stored_response("resp-sess-1", "response")
        assert exc_info.value.status == 404
        assert str(exc_info.value) == "Response with id 'resp-sess-1' not found."

    async def test_load_kind_mismatch_the_other_direction_is_not_found(
        self, stub: _StubSessionClient
    ) -> None:
        """A document stored as a response 404s when loaded as a chat completion.

        The 404 message is worded for the requested kind, so a
        ``chatcmpl-`` ID is reported as a missing chat completion.

        Ref: stdapi/responses_store.py:_not_found
        """
        document = {"response": {"id": "resp-sess-1", "object": "response"}}
        await responses_store.save_stored_response("chatcmpl-sess-1", document)
        with pytest.raises(ApiError) as exc_info:
            await responses_store.load_stored_response(
                "chatcmpl-sess-1", "chat_completion"
            )
        assert exc_info.value.status == 404
        assert (
            str(exc_info.value)
            == "Chat completion with id 'chatcmpl-sess-1' not found."
        )

    async def test_delete_kind_mismatch_is_not_found_and_does_not_delete(
        self, stub: _StubSessionClient
    ) -> None:
        """A session tagged with a different kind 404s without ending or deleting it.

        Ref: stdapi/responses_store.py:delete_stored_response
        """
        stub.tags["arn:sess-1"] = {"stdapi-ai.stored-object": "chat_completion"}
        with pytest.raises(ApiError) as exc_info:
            await responses_store.delete_stored_response("resp-sess-1", "response")
        assert exc_info.value.status == 404
        assert str(exc_info.value) == "Response with id 'resp-sess-1' not found."
        called = [name for name, _ in stub.requests]
        assert "end_session" not in called
        assert "delete_session" not in called

    async def test_delete_missing_kind_tag_is_deletable(
        self, stub: _StubSessionClient
    ) -> None:
        """A session without a kind tag is still deletable.

        An untagged session predates kind tracking or was orphaned by a failed
        generation, so an absent tag must not be read as a kind mismatch.

        Ref: stdapi/responses_store.py:delete_stored_response
        """
        await responses_store.delete_stored_response("resp-sess-1", "response")
        assert [name for name, _ in stub.requests][-2:] == [
            "end_session",
            "delete_session",
        ]

    async def test_delete_tag_lookup_not_found_falls_through_to_delete(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A NotFound while reading the kind tag does not 404 by itself.

        It falls through to the delete calls, which apply the usual 404
        handling (here the session still exists, so deletion just proceeds).

        Ref: stdapi/responses_store.py:_session_kind_tag_or_none
        """

        async def _vanished(**_params: Any) -> dict[str, Any]:  # noqa: ANN401
            raise _NOT_FOUND

        monkeypatch.setattr(stub, "list_tags_for_resource", _vanished)
        await responses_store.delete_stored_response("resp-sess-1", "response")
        called = [name for name, _ in stub.requests]
        assert "end_session" in called
        assert "delete_session" in called


class TestListStoredSessions:
    """Bounded ListSessions scan filtered by the stored-object kind tag.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/sessions.html
         stdapi/responses_store.py:list_stored_sessions
    """

    async def test_filters_by_kind_tag(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only sessions tagged with the requested kind are returned.

        ListSessions returns every session of the account, including chat
        completions and sessions created by other tools, so the kind tag is the
        only filter available.

        Ref: stdapi/responses_store.py:list_stored_sessions
        """
        monkeypatch.setattr(responses_store, "build_metadata", lambda **_: {})
        stub.sessions = [
            {
                "sessionId": "resp-1",
                "sessionArn": "arn:resp-1",
                "createdAt": datetime(2024, 1, 1, tzinfo=UTC),
            },
            {
                "sessionId": "chat-1",
                "sessionArn": "arn:chat-1",
                "createdAt": datetime(2024, 1, 2, tzinfo=UTC),
            },
            {
                "sessionId": "untagged-1",
                "sessionArn": "arn:untagged-1",
                "createdAt": datetime(2024, 1, 3, tzinfo=UTC),
            },
        ]
        stub.tags = {
            "arn:resp-1": {"stdapi-ai.stored-object": "response"},
            "arn:chat-1": {"stdapi-ai.stored-object": "chat_completion"},
        }
        sessions = await responses_store.list_stored_sessions("response")
        assert sessions == [("resp-1", datetime(2024, 1, 1, tzinfo=UTC))]

    async def test_paginates_across_pages(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sessions from every ListSessions page are scanned and returned.

        Ref: stdapi/responses_store.py:list_stored_sessions
        """
        monkeypatch.setattr(responses_store, "_LIST_PAGE_SIZE", 2)
        stub.sessions = [
            {
                "sessionId": f"resp-{i}",
                "sessionArn": f"arn:resp-{i}",
                "createdAt": datetime(2024, 1, i + 1, tzinfo=UTC),
            }
            for i in range(3)
        ]
        stub.tags = {
            session["sessionArn"]: {"stdapi-ai.stored-object": "response"}
            for session in stub.sessions
        }
        sessions = await responses_store.list_stored_sessions("response")
        list_sessions_calls = [
            params for name, params in stub.requests if name == "list_sessions"
        ]
        assert len(list_sessions_calls) == 2
        assert sorted(sessions) == [
            (f"resp-{i}", datetime(2024, 1, i + 1, tzinfo=UTC)) for i in range(3)
        ]

    async def test_returns_session_id_and_created_at_pairs(
        self, stub: _StubSessionClient
    ) -> None:
        """Each result pair holds the session ID and its ListSessions creation time.

        Ref: stdapi/responses_store.py:list_stored_sessions
        """
        created_at = datetime(2024, 3, 4, tzinfo=UTC)
        stub.sessions = [
            {"sessionId": "resp-1", "sessionArn": "arn:resp-1", "createdAt": created_at}
        ]
        stub.tags = {"arn:resp-1": {"stdapi-ai.stored-object": "response"}}
        sessions = await responses_store.list_stored_sessions("response")
        assert sessions == [("resp-1", created_at)]

    async def test_second_list_call_does_not_refetch_tags(
        self, stub: _StubSessionClient
    ) -> None:
        """A cached kind tag is reused, not re-fetched, on a later list call.

        The kind tag never changes, so the second listing must classify the
        session from the cache and issue no further ListTagsForResource call.

        Ref: stdapi/responses_store.py:_cached_kind_tag
        """
        stub.sessions = [
            {
                "sessionId": "resp-1",
                "sessionArn": "arn:resp-1",
                "createdAt": datetime(2024, 1, 1, tzinfo=UTC),
            }
        ]
        stub.tags = {"arn:resp-1": {"stdapi-ai.stored-object": "response"}}
        expected = [("resp-1", datetime(2024, 1, 1, tzinfo=UTC))]
        assert await responses_store.list_stored_sessions("response") == expected
        assert await responses_store.list_stored_sessions("response") == expected
        tag_calls = [
            name for name, _ in stub.requests if name == "list_tags_for_resource"
        ]
        assert len(tag_calls) == 1

    async def test_second_list_call_does_not_refetch_tags_for_untagged_session(
        self, stub: _StubSessionClient
    ) -> None:
        """An untagged session is negatively cached and stays excluded, without re-fetching.

        A missing tag is cached as a sentinel rather than as a cache miss, so a
        foreign session does not cost one ListTagsForResource call per listing.

        Ref: stdapi/responses_store.py:_cached_kind_tag
        """
        stub.sessions = [
            {
                "sessionId": "untagged-1",
                "sessionArn": "arn:untagged-1",
                "createdAt": datetime(2024, 1, 1, tzinfo=UTC),
            }
        ]
        # No tags configured for "arn:untagged-1": list_tags_for_resource returns {}.
        assert await responses_store.list_stored_sessions("response") == []
        assert await responses_store.list_stored_sessions("response") == []
        tag_calls = [
            name for name, _ in stub.requests if name == "list_tags_for_resource"
        ]
        assert len(tag_calls) == 1

    async def test_bounded_concurrency_returns_correct_results(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bounding concurrent tag lookups drops or misattributes no result.

        Tag lookups run concurrently behind a semaphore and are zipped back
        onto their session summaries, so a page larger than the concurrency
        limit must still classify every session correctly.

        Ref: stdapi/responses_store.py:list_stored_sessions
        """
        monkeypatch.setattr(responses_store, "_TAG_FETCH_CONCURRENCY", 4)
        stub.sessions = [
            {
                "sessionId": f"sess-{i}",
                "sessionArn": f"arn:sess-{i}",
                "createdAt": datetime(2024, 1, 1, tzinfo=UTC),
            }
            for i in range(40)
        ]
        stub.tags = {
            f"arn:sess-{i}": {
                "stdapi-ai.stored-object": "response"
                if i % 2 == 0
                else "chat_completion"
            }
            for i in range(40)
        }
        sessions = await responses_store.list_stored_sessions("response")
        assert {session_id for session_id, _ in sessions} == {
            f"sess-{i}" for i in range(40) if i % 2 == 0
        }

    async def test_list_scan_limit_stops_scanning(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Scanning stops at ``_LIST_SCAN_LIMIT`` even when more sessions remain.

        The listing is a bounded scan, not true pagination: the last page's
        ``maxResults`` is shrunk to the remaining budget so the limit is never
        overshot and the extra sessions are simply never seen.

        Ref: stdapi/responses_store.py:list_stored_sessions
        """
        monkeypatch.setattr(responses_store, "_LIST_SCAN_LIMIT", 5)
        monkeypatch.setattr(responses_store, "_LIST_PAGE_SIZE", 2)
        stub.sessions = [
            {
                "sessionId": f"sess-{i}",
                "sessionArn": f"arn:sess-{i}",
                "createdAt": datetime(2024, 1, 1, tzinfo=UTC),
            }
            for i in range(10)
        ]
        stub.tags = {
            f"arn:sess-{i}": {"stdapi-ai.stored-object": "response"} for i in range(10)
        }
        sessions = await responses_store.list_stored_sessions("response")
        assert sorted(session_id for session_id, _ in sessions) == [
            f"sess-{i}" for i in range(5)
        ]
        assert [
            params["maxResults"]
            for name, params in stub.requests
            if name == "list_sessions"
        ] == [2, 2, 1]

    async def test_kind_cache_clears_when_exceeding_limit(
        self, stub: _StubSessionClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The kind-tag cache is cleared once its size exceeds ``_KIND_CACHE_LIMIT``.

        Because a session's kind never changes, the cache is bounded by a full
        clear rather than by eviction, and the freshly fetched entry survives.

        Ref: stdapi/responses_store.py:_cached_kind_tag
        """
        monkeypatch.setattr(responses_store, "_KIND_CACHE_LIMIT", 2)
        cache = responses_store._KIND_CACHE  # noqa: SLF001 (isolated per-test by the stub fixture)
        cache.update({"sess-0": "response", "sess-1": "response", "sess-2": "response"})

        class _TagOnlyClient:
            """Client stub exposing only the tag-lookup call."""

            async def list_tags_for_resource(self, **_params: Any) -> dict[str, Any]:  # noqa: ANN401
                return {"tags": {"stdapi-ai.stored-object": "response"}}

        kind = await responses_store._cached_kind_tag(  # noqa: SLF001
            _TagOnlyClient(),  # type: ignore[arg-type]
            "sess-3",
            "arn:sess-3",
        )
        assert kind == "response"
        assert set(cache) == {"sess-3"}
