"""Tests for the Anthropic-compatible /v1/messages/batches routes.

Upstream sends the requests inline, expires a batch 24 hours after creation,
does not guarantee the order of the result lines, and requires a batch to have
ended before it can be deleted. Each inline request carries its own model, so a
batch naming several models fans out — bounded, and with the same per-model
floor of 100 requests the backing jobs impose.

Everything below the live class runs against in-memory doubles for the object
store and the job service. Submitting for real is the one step those doubles
cannot stand in for, and it is what ``TestMessageBatchLive`` does.

Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
     https://stdapi.ai/api_anthropic_batches/
     stdapi/routes/anthropic_messages_batches.py:create
     stdapi/batches.py:prepare_anthropic_requests
"""

import contextlib
from typing import TYPE_CHECKING, Any

import pytest

from stdapi import batches
from tests import _batches
from tests._batches import converse_output

if TYPE_CHECKING:
    from anthropic import Anthropic
    from starlette.testclient import TestClient

#: Base path of the Message Batches routes.
_PATH = "/anthropic/v1/messages/batches"


#: Message parameters asking for a cache point in both cacheable components.
_CACHED_PARAMS: dict[str, Any] = {
    "system": [
        {"type": "text", "text": "context", "cache_control": {"type": "ephemeral"}}
    ],
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}
            ],
        }
    ],
}


def _requests(
    count: int, *, model: str = "amazon.nova-micro-v1:0", prefix: str = "req"
) -> list[dict[str, Any]]:
    """Return *count* well-formed inline Message Batch requests."""
    return [
        {
            "custom_id": f"{prefix}-{index}",
            "params": {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
            },
        }
        for index in range(count)
    ]


def _create(client: TestClient, requests: list[dict[str, Any]]) -> dict[str, Any]:
    """Submit a Message Batch and return the decoded response body."""
    response = client.post(_PATH, json={"requests": requests})
    return {"http_status": response.status_code, **response.json()}


def _end_batch(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> str:
    """Submit a Message Batch against the doubles and run it to `ended`."""
    s3, bedrock = _batches.install(monkeypatch)
    batch_id = str(_create(client, _requests(100))["id"])
    bedrock.finish(succeeded=100)
    _batches.write_job_output(
        s3,
        bedrock,
        [
            {"recordId": f"req-{index}", "modelOutput": converse_output("answer")}
            for index in range(100)
        ],
        {"inputTokenCount": 500, "outputTokenCount": 300},
    )
    return batch_id


@pytest.mark.local
class TestMessageBatchValidation:
    """POST /v1/messages/batches: what a Message Batch refuses.

    Every case is refused at submit, so the client never pays for a batch that
    could only have failed request by request.

    Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
         stdapi/batches.py:_group_by_model
    """

    def test_below_minimum_names_the_floor(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A batch under the per-model floor is refused, naming the floor.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html
             stdapi/batches.py:_group_by_model
        """
        _batches.install(monkeypatch)
        body = _create(anthropic_app_client, _requests(99))
        assert body["http_status"] == 400
        assert body["type"] == "error"
        assert str(batches.MIN_REQUESTS_PER_MODEL) in body["error"]["message"]

    def test_one_model_below_the_floor_names_only_that_model(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mixed batch over the total but under it per model names the culprit.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html
             stdapi/batches.py:_group_by_model
        """
        _batches.install(monkeypatch)
        requests = _requests(150) + _requests(
            50, model="amazon.nova-lite-v1:0", prefix="lite"
        )
        body = _create(anthropic_app_client, requests)
        assert body["http_status"] == 400
        message = body["error"]["message"]
        assert "amazon.nova-lite-v1:0 (50)" in message
        assert "amazon.nova-micro-v1:0" not in message

    def test_fan_out_cap_is_named(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Naming more models than the cap is refused, and the cap is stated.

        Ref: stdapi/batches.py:_group_by_model
        """
        _batches.install(monkeypatch)
        requests = [
            entry
            for index in range(batches.MAX_MODELS_PER_BATCH + 1)
            for entry in _requests(100, model=f"model-{index}", prefix=f"m{index}")
        ]
        body = _create(anthropic_app_client, requests)
        assert body["http_status"] == 400
        message = body["error"]["message"]
        assert str(batches.MAX_MODELS_PER_BATCH) in message
        assert str(batches.MAX_MODELS_PER_BATCH + 1) in message

    def test_duplicate_custom_id_is_refused(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A repeated custom_id is refused: it is the only join key results have.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:_validate_custom_ids
        """
        _batches.install(monkeypatch)
        requests = _requests(100)
        requests[9]["custom_id"] = "req-8"
        body = _create(anthropic_app_client, requests)
        assert body["http_status"] == 400
        assert "more than once" in body["error"]["message"]

    def test_streaming_request_is_refused(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request asking to stream is refused: a batch has nothing to stream to.

        Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
             stdapi/batches.py:_prepare_anthropic_request
        """
        _batches.install(monkeypatch)
        requests = _requests(100)
        requests[0]["params"]["stream"] = True
        body = _create(anthropic_app_client, requests)
        assert body["http_status"] == 400
        assert "Request 1: 'stream' is not available" in body["error"]["message"]

    def test_custom_id_charset_is_enforced(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A custom_id outside the documented charset is refused by the schema.

        Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
             stdapi/types/anthropic_batches.py:MessageBatchRequest
        """
        _batches.install(monkeypatch)
        requests = _requests(100)
        requests[0]["custom_id"] = "req 0!"
        body = _create(anthropic_app_client, requests)
        assert body["http_status"] == 400
        assert "custom_id" in body["error"]["message"]

    def test_disabled_without_service_role(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the service role the routes answer 529, not a backend error.

        Ref: stdapi/batches.py:require_batches_enabled
        """
        from stdapi.config import SETTINGS  # noqa: PLC0415

        _batches.install(monkeypatch)
        monkeypatch.setattr(SETTINGS, "aws_bedrock_batch_role_arn", None)
        response = anthropic_app_client.post(_PATH, json={"requests": _requests(100)})
        # The Anthropic envelope maps a 503 to its own overloaded_error / 529.
        assert response.status_code == 529
        assert (
            "not available on the current server" in response.json()["error"]["message"]
        )


@pytest.mark.local
class TestMessageBatchLifecycle:
    """POST/GET/DELETE /v1/messages/batches: states, results and deletion.

    Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
         stdapi/routes/anthropic_messages_batches.py:_status
    """

    def test_create_reports_in_progress(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A submitted batch is `in_progress`, with every request processing.

        Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
             stdapi/routes/anthropic_messages_batches.py:create
        """
        _, bedrock = _batches.install(monkeypatch)
        body = _create(anthropic_app_client, _requests(100))
        assert body["http_status"] == 200
        assert body["id"].startswith("msgbatch_")
        assert body["type"] == "message_batch"
        assert body["processing_status"] == "in_progress"
        assert body["request_counts"]["processing"] == 100
        assert body["request_counts"]["succeeded"] == 0
        assert "results_url" not in body
        assert body["created_at"].endswith("Z")
        assert len(bedrock.created) == 1

    def test_prompt_cache_hints_are_accepted_and_dropped(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`cache_control` is accepted, and no cache point is submitted with it.

        Batched requests read and write no cache, and one carrying a cache
        point fails — every request of the batch, not just the one that
        carried it. The hint is an optimization the caller can do without, so
        it is dropped and the request runs, at the batch price it asked for.

        The translated request is captured on its way in, so the assertion on
        the submitted body cannot pass for a request that never carried a
        cache point to begin with.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
             stdapi/batches.py:_to_model_input
        """
        from pydantic_core import from_json  # noqa: PLC0415

        s3, _ = _batches.install(monkeypatch, translate=True)
        translated = _batches.capture_translations(monkeypatch)
        requests = _requests(100)
        for request in requests:
            request["params"] |= _CACHED_PARAMS
        assert _create(anthropic_app_client, requests)["http_status"] == 200
        assert translated[0]["system"][-1] == {"cachePoint": {"type": "default"}}
        assert translated[0]["messages"][0]["content"][-1] == {
            "cachePoint": {"type": "default"}
        }
        written = next(
            body for (_b, key), body in s3.objects.items() if key.endswith(".jsonl")
        )
        model_input = from_json(written.splitlines()[0])["modelInput"]
        assert "cachePoint" not in str(model_input)
        assert model_input["system"] == [{"text": "context"}]
        assert model_input["messages"][0]["content"] == [{"text": "hi"}]

    def test_two_models_run_as_two_jobs(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A batch naming two models fans out, and still reports one batch.

        Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
             stdapi/batches.py:create_batch
        """
        _, bedrock = _batches.install(monkeypatch)
        requests = _requests(100) + _requests(
            100, model="amazon.nova-lite-v1:0", prefix="lite"
        )
        body = _create(anthropic_app_client, requests)
        assert body["http_status"] == 200
        assert body["request_counts"]["processing"] == 200
        assert len(bedrock.created) == 2
        assert {job["modelId"] for job in bedrock.created} == {
            "amazon.nova-micro-v1:0",
            "amazon.nova-lite-v1:0",
        }

    def test_a_rejected_constituent_stops_the_others(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When one model cannot batch, no sibling job or request is left behind.

        Ref: stdapi/batches.py:create_batch
        """
        s3, bedrock = _batches.install(monkeypatch)
        bedrock.reject_models = {"amazon.nova-lite-v1:0"}
        requests = _requests(100) + _requests(
            100, model="amazon.nova-lite-v1:0", prefix="lite"
        )
        body = _create(anthropic_app_client, requests)
        assert body["http_status"] == 400
        assert "not available for batched requests" in body["error"]["message"]
        assert len(bedrock.stopped) == len(bedrock.jobs)
        assert not [key for (_b, key) in s3.objects if key.endswith(".jsonl")]

    def test_ended_batch_publishes_its_results(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once the jobs end, the batch is `ended` and its results are readable.

        Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
             stdapi/batches.py:iter_anthropic_results
        """
        from pydantic_core import from_json  # noqa: PLC0415

        s3, bedrock = _batches.install(monkeypatch)
        batch_id = _create(anthropic_app_client, _requests(100))["id"]
        bedrock.finish(succeeded=99, errored=1)
        _batches.write_job_output(
            s3,
            bedrock,
            [
                *(
                    {
                        "recordId": f"req-{index}",
                        "modelOutput": converse_output("answer"),
                    }
                    for index in range(99)
                ),
                {
                    "recordId": "req-99",
                    "error": {
                        "errorCode": 400,
                        "errorMessage": "Remove the toolConfig field",
                        "expired": False,
                    },
                },
            ],
            {"inputTokenCount": 495, "outputTokenCount": 297},
        )
        body = anthropic_app_client.get(f"{_PATH}/{batch_id}").json()
        assert body["processing_status"] == "ended"
        assert body["request_counts"] == {
            "processing": 0,
            "succeeded": 99,
            "errored": 1,
            "canceled": 0,
            "expired": 0,
        }
        assert body["ended_at"] is not None
        assert body["results_url"].endswith(f"{_PATH}/{batch_id}/results")

        results = anthropic_app_client.get(f"{_PATH}/{batch_id}/results")
        assert results.status_code == 200
        lines = [from_json(line) for line in results.content.splitlines()]
        assert len(lines) == 100
        assert {line["custom_id"] for line in lines} == {
            f"req-{index}" for index in range(100)
        }
        succeeded = next(line for line in lines if line["custom_id"] == "req-0")
        assert succeeded["result"]["type"] == "succeeded"
        assert succeeded["result"]["message"]["content"][0]["text"] == "answer"
        assert succeeded["result"]["message"]["stop_reason"] == "end_turn"
        errored = next(line for line in lines if line["custom_id"] == "req-99")
        assert errored["result"]["type"] == "errored"
        assert errored["result"]["error"]["type"] == "error"
        assert "toolConfig" not in errored["result"]["error"]["error"]["message"]

    def test_results_are_not_available_before_the_end(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reading the results of a running batch answers 404, not empty output.

        Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
             stdapi/routes/anthropic_messages_batches.py:results
        """
        _batches.install(monkeypatch)
        batch_id = _create(anthropic_app_client, _requests(100))["id"]
        response = anthropic_app_client.get(f"{_PATH}/{batch_id}/results")
        assert response.status_code == 404
        assert "not available yet" in response.json()["error"]["message"]

    def test_cancel_ends_the_batch_and_is_idempotent(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cancelling stops every job; cancelling again changes nothing.

        Stopping is asynchronous, so the batch reports `canceling` until the
        jobs have actually stopped.

        Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
             stdapi/batches.py:cancel_batch
        """
        _, bedrock = _batches.install(monkeypatch)
        batch_id = _create(anthropic_app_client, _requests(100))["id"]
        first = anthropic_app_client.post(f"{_PATH}/{batch_id}/cancel").json()
        assert first["processing_status"] == "canceling"
        assert first["cancel_initiated_at"] is not None
        assert bedrock.stopped
        second = anthropic_app_client.post(f"{_PATH}/{batch_id}/cancel").json()
        assert second["cancel_initiated_at"] == first["cancel_initiated_at"]
        bedrock.finish(status="Stopped", succeeded=0)
        ended = anthropic_app_client.get(f"{_PATH}/{batch_id}").json()
        assert ended["processing_status"] == "ended"

    def test_cancelling_an_ended_batch_changes_nothing(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A batch that already ended keeps its outcome when a cancel arrives.

        A retrying client must not be able to turn a finished batch into a
        cancelled one: what its requests produced was paid for.

        Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
             stdapi/batches.py:cancel_batch
        """
        _, bedrock = _batches.install(monkeypatch)
        batch_id = _create(anthropic_app_client, _requests(100))["id"]
        bedrock.finish(succeeded=100)
        body = anthropic_app_client.post(f"{_PATH}/{batch_id}/cancel").json()
        assert body["processing_status"] == "ended"
        assert "cancel_initiated_at" not in body
        assert body["request_counts"] == {
            "processing": 0,
            "succeeded": 100,
            "errored": 0,
            "canceled": 0,
            "expired": 0,
        }
        assert not bedrock.stopped

    def test_cancelled_requests_are_reported_as_canceled(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cancelled batch counts its unanswered requests as canceled.

        Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
             stdapi/routes/anthropic_messages_batches.py:_to_message_batch
        """
        _, bedrock = _batches.install(monkeypatch)
        batch_id = _create(anthropic_app_client, _requests(100))["id"]
        anthropic_app_client.post(f"{_PATH}/{batch_id}/cancel")
        bedrock.finish(status="Stopped", succeeded=0)
        body = anthropic_app_client.get(f"{_PATH}/{batch_id}").json()
        assert body["request_counts"]["canceled"] == 100
        assert body["request_counts"]["succeeded"] == 0

    def test_cancel_keeps_the_messages_already_produced(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cancelling mid-run keeps every Message produced, and cancels the rest.

        The requests that answered before the stop were billed, so they stay
        readable; only the ones that never ran are reported `canceled`.

        Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
             stdapi/batches.py:iter_anthropic_results
        """
        from pydantic_core import from_json  # noqa: PLC0415

        s3, bedrock = _batches.install(monkeypatch)
        batch_id = _create(anthropic_app_client, _requests(100))["id"]
        anthropic_app_client.post(f"{_PATH}/{batch_id}/cancel")
        _batches.write_job_output(
            s3,
            bedrock,
            [
                {"recordId": f"req-{index}", "modelOutput": converse_output("answer")}
                for index in range(60)
            ],
            {"inputTokenCount": 300, "outputTokenCount": 180},
        )
        bedrock.finish(status="Stopped", succeeded=60)

        body = anthropic_app_client.get(f"{_PATH}/{batch_id}").json()
        assert body["processing_status"] == "ended"
        assert body["request_counts"]["succeeded"] == 60
        assert body["request_counts"]["canceled"] == 40

        results = anthropic_app_client.get(f"{_PATH}/{batch_id}/results")
        lines = {
            line["custom_id"]: line["result"]
            for line in (from_json(raw) for raw in results.content.splitlines())
        }
        assert len(lines) == 100
        assert lines["req-0"]["type"] == "succeeded"
        assert lines["req-0"]["message"]["content"][0]["text"] == "answer"
        assert lines["req-59"]["type"] == "succeeded"
        assert lines["req-60"] == {"type": "canceled"}
        assert lines["req-99"] == {"type": "canceled"}

    def test_delete_needs_an_ended_batch(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A running batch cannot be deleted; the message says to cancel first.

        Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
             stdapi/batches.py:delete_batch
        """
        _batches.install(monkeypatch)
        batch_id = _create(anthropic_app_client, _requests(100))["id"]
        response = anthropic_app_client.delete(f"{_PATH}/{batch_id}")
        assert response.status_code == 400
        assert "Cancel it first" in response.json()["error"]["message"]

    def test_delete_removes_the_batch_and_its_data(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An ended batch is deletable, and stops being retrievable afterwards.

        Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
             stdapi/batches.py:delete_batch
        """
        s3, bedrock = _batches.install(monkeypatch)
        batch_id = _create(anthropic_app_client, _requests(100))["id"]
        bedrock.finish()
        response = anthropic_app_client.delete(f"{_PATH}/{batch_id}")
        assert response.status_code == 200
        assert response.json() == {"id": batch_id, "type": "message_batch_deleted"}
        assert anthropic_app_client.get(f"{_PATH}/{batch_id}").status_code == 404
        assert not [key for (_b, key) in s3.objects if "/0/" in key]

    def test_listing_returns_the_batch(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A created batch shows up in the listing.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:list_batches
        """
        _batches.install(monkeypatch)
        batch_id = _create(anthropic_app_client, _requests(100))["id"]
        body = anthropic_app_client.get(f"{_PATH}?limit=10").json()
        assert [item["id"] for item in body["data"]] == [batch_id]
        assert body["has_more"] is False

    def test_cursors_page_forwards_and_backwards(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`after_id` walks towards older batches, `before_id` back one page.

        Upstream defines `before_id` as the page immediately before the named
        object, so a client walking back from the oldest batch must reach the
        middle page rather than the newest one it started from.

        Ref: https://platform.claude.com/docs/en/api/listing-message-batches
             stdapi/batches.py:list_batches
        """
        _batches.install(monkeypatch)
        oldest, middle, newest = (
            _create(anthropic_app_client, _requests(100))["id"] for _ in range(3)
        )
        first = anthropic_app_client.get(f"{_PATH}?limit=1").json()
        assert [item["id"] for item in first["data"]] == [newest]
        assert first["has_more"] is True

        forwards = anthropic_app_client.get(f"{_PATH}?limit=1&after_id={newest}").json()
        assert [item["id"] for item in forwards["data"]] == [middle]

        backwards = anthropic_app_client.get(
            f"{_PATH}?limit=1&before_id={oldest}"
        ).json()
        assert [item["id"] for item in backwards["data"]] == [middle]

    def test_unknown_batch_is_not_found(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An identifier that decodes but names nothing answers 404.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:_read_record
        """
        _batches.install(monkeypatch)
        response = anthropic_app_client.get(f"{_PATH}/msgbatch_{'a' * 32}")
        assert response.status_code == 404
        assert response.json()["error"]["type"] == "not_found_error"


@pytest.mark.local
class TestMessageBatchFanOut:
    """A batch naming two models, whose constituents advance independently.

    The fan-out is the feature's own semantic: one batch object answers for
    several backing jobs, so it must report the least-advanced of them, sum
    their counts, and carry every one of their results — including when one
    of them fails while the other has already answered.

    Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
         stdapi/batches.py:BatchState
    """

    @staticmethod
    def _create_two_models(client: TestClient) -> str:
        """Submit a batch naming two models and return its identifier."""
        requests = _requests(100) + _requests(
            100, model="amazon.nova-lite-v1:0", prefix="lite"
        )
        body = _create(client, requests)
        assert body["http_status"] == 200
        return str(body["id"])

    def test_status_is_the_least_advanced_constituent(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One finished job does not end a batch whose sibling is still running.

        Ref: stdapi/batches.py:BatchState.ended
        """
        _, bedrock = _batches.install(monkeypatch)
        batch_id = self._create_two_models(anthropic_app_client)
        first, second = list(bedrock.jobs)
        bedrock.finish(arn=first, succeeded=100)
        bedrock.start(arn=second)

        body = anthropic_app_client.get(f"{_PATH}/{batch_id}").json()
        assert body["processing_status"] == "in_progress"
        assert "ended_at" not in body
        assert "results_url" not in body
        assert (
            anthropic_app_client.get(f"{_PATH}/{batch_id}/results").status_code == 404
        )

    def test_request_counts_sum_across_the_jobs(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The tallies are the sum of both jobs', not one job's.

        Progress is visible while the batch runs, which is what a client polls
        `request_counts` for.

        Ref: stdapi/batches.py:BatchState.succeeded
        """
        _, bedrock = _batches.install(monkeypatch)
        batch_id = self._create_two_models(anthropic_app_client)
        first, second = list(bedrock.jobs)
        bedrock.finish(arn=first, succeeded=100)
        bedrock.start(arn=second)

        running = anthropic_app_client.get(f"{_PATH}/{batch_id}").json()
        assert running["request_counts"]["succeeded"] == 100
        assert running["request_counts"]["processing"] == 100

        bedrock.finish(arn=second, succeeded=60, errored=40)
        ended = anthropic_app_client.get(f"{_PATH}/{batch_id}").json()
        assert ended["processing_status"] == "ended"
        assert ended["request_counts"] == {
            "processing": 0,
            "succeeded": 160,
            "errored": 40,
            "canceled": 0,
            "expired": 0,
        }

    def test_a_failed_job_does_not_discard_the_other_output(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One constituent failing outright keeps the other's Messages readable.

        Ref: stdapi/batches.py:iter_anthropic_results
        """
        from pydantic_core import from_json  # noqa: PLC0415

        s3, bedrock = _batches.install(monkeypatch)
        batch_id = self._create_two_models(anthropic_app_client)
        first, second = list(bedrock.jobs)
        _batches.write_job_output(
            s3,
            bedrock,
            [
                {"recordId": f"req-{index}", "modelOutput": converse_output("answer")}
                for index in range(100)
            ],
            {"inputTokenCount": 500, "outputTokenCount": 300},
            arn=first,
        )
        bedrock.finish(arn=first, succeeded=100)
        bedrock.finish(arn=second, status="Failed", succeeded=0)

        body = anthropic_app_client.get(f"{_PATH}/{batch_id}").json()
        assert body["processing_status"] == "ended"
        assert body["request_counts"]["succeeded"] == 100

        results = anthropic_app_client.get(f"{_PATH}/{batch_id}/results")
        assert results.status_code == 200
        lines = [from_json(raw) for raw in results.content.splitlines()]
        assert {line["custom_id"] for line in lines} == {
            f"req-{index}" for index in range(100)
        }
        assert all(line["result"]["type"] == "succeeded" for line in lines)

    def test_results_carry_both_jobs_answers(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The results file concatenates every constituent's own output.

        Each job answers its own requests, so a results reader must find both
        models' `custom_id`s and each one exactly once.

        Ref: stdapi/batches.py:iter_anthropic_results
        """
        from pydantic_core import from_json  # noqa: PLC0415

        s3, bedrock = _batches.install(monkeypatch)
        batch_id = self._create_two_models(anthropic_app_client)
        first, second = list(bedrock.jobs)
        for arn, prefix in ((first, "req"), (second, "lite")):
            _batches.write_job_output(
                s3,
                bedrock,
                [
                    {
                        "recordId": f"{prefix}-{index}",
                        "modelOutput": converse_output(prefix),
                    }
                    for index in range(100)
                ],
                {"inputTokenCount": 500, "outputTokenCount": 300},
                arn=arn,
            )
        bedrock.finish(succeeded=100)

        results = anthropic_app_client.get(f"{_PATH}/{batch_id}/results")
        lines = [from_json(raw) for raw in results.content.splitlines()]
        assert len(lines) == 200
        by_id = {line["custom_id"]: line["result"] for line in lines}
        assert len(by_id) == 200
        assert by_id["req-0"]["message"]["content"][0]["text"] == "req"
        assert by_id["lite-99"]["message"]["content"][0]["text"] == "lite"

    def test_cancel_keeps_a_finished_constituent_finished(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cancelling a batch mid-run leaves the job that already ended alone.

        Only the job still running is asked to stop, and only its requests are
        reported `canceled`.

        Ref: stdapi/batches.py:cancel_batch
        """
        from pydantic_core import from_json  # noqa: PLC0415

        s3, bedrock = _batches.install(monkeypatch)
        batch_id = self._create_two_models(anthropic_app_client)
        first, second = list(bedrock.jobs)
        _batches.write_job_output(
            s3,
            bedrock,
            [
                {"recordId": f"req-{index}", "modelOutput": converse_output("answer")}
                for index in range(100)
            ],
            {"inputTokenCount": 500, "outputTokenCount": 300},
            arn=first,
        )
        bedrock.finish(arn=first, succeeded=100)
        bedrock.start(arn=second)

        cancelled = anthropic_app_client.post(f"{_PATH}/{batch_id}/cancel").json()
        assert cancelled["processing_status"] == "canceling"
        assert bedrock.jobs[first]["status"] == "Completed"
        assert bedrock.jobs[second]["status"] == "Stopping"

        bedrock.finish(arn=second, status="Stopped", succeeded=0)
        body = anthropic_app_client.get(f"{_PATH}/{batch_id}").json()
        assert body["request_counts"]["succeeded"] == 100
        assert body["request_counts"]["canceled"] == 100

        results = anthropic_app_client.get(f"{_PATH}/{batch_id}/results")
        by_id = {
            line["custom_id"]: line["result"]
            for line in (from_json(raw) for raw in results.content.splitlines())
        }
        assert by_id["req-0"]["type"] == "succeeded"
        assert by_id["lite-0"] == {"type": "canceled"}


@pytest.mark.local
class TestMessageBatchResultsUrl:
    """The address the SDK fetches a batch's results from.

    The Anthropic SDK retrieves the batch and then dials ``results_url`` with
    its own client, and an absolute value is dialled verbatim — no merge with
    the client's ``base_url``, no timeout and no retry ceiling on that call. It
    therefore has to name the origin the client itself reached, which is the
    request's and nothing else: a configured identity is under no obligation to
    resolve, and a deployment whose identity is not its API host would hand
    every client an address that never answers.

    Ref: anthropic/_base_client.py:_prepare_url
         anthropic/resources/messages/batches.py:results
         stdapi/routes/anthropic_messages_batches.py:_results_url
    """

    def test_follows_the_request_not_a_configured_identity(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every route serving the URL names the origin the client dialled.

        This checkout configures an OAuth resource identifier, which RFC 8707
        never requires to be dialable, so it must not decide the URL.

        Ref: https://datatracker.ietf.org/doc/html/rfc8707#section-2
             stdapi/routes/anthropic_messages_batches.py:_to_message_batch
        """
        from stdapi.config import SETTINGS  # noqa: PLC0415

        assert SETTINGS.oauth_resource_identifier
        batch_id = _end_batch(anthropic_app_client, monkeypatch)
        expected = f"http://testserver{_PATH}/{batch_id}/results"

        retrieved = anthropic_app_client.get(f"{_PATH}/{batch_id}").json()
        listed = anthropic_app_client.get(_PATH).json()["data"][0]
        cancelled = anthropic_app_client.post(f"{_PATH}/{batch_id}/cancel").json()

        assert retrieved["results_url"] == expected
        assert listed["results_url"] == expected
        assert cancelled["results_url"] == expected
        assert SETTINGS.oauth_resource_identifier not in retrieved["results_url"]

    def test_honors_the_forwarded_host_and_scheme(
        self, anthropic_app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Behind a TLS-terminating proxy the URL is the public origin.

        The client dialled the proxy, so the address handed back is the
        proxy's — read from the same forwarded headers the server already
        trusts for the client address, not from the connection it received.

        Ref: https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/X-Forwarded-Proto
             stdapi/main.py:app
        """
        batch_id = _end_batch(anthropic_app_client, monkeypatch)

        body = anthropic_app_client.get(
            f"{_PATH}/{batch_id}",
            headers={"host": "gateway.example.com", "x-forwarded-proto": "https"},
        ).json()

        assert (
            body["results_url"]
            == f"https://gateway.example.com{_PATH}/{batch_id}/results"
        )

    def test_the_sdk_dials_it_without_repeating_the_route_prefix(self) -> None:
        """The SDK's URL preparation leaves the absolute value untouched.

        The SDK's ``base_url`` already carries the route prefix, so a value it
        merged instead of dialling verbatim would double that prefix.

        Ref: anthropic/_base_client.py:_prepare_url
             stdapi/routes/anthropic_messages_batches.py:_results_url
        """
        from httpx import URL  # noqa: PLC0415
        from starlette.requests import Request  # noqa: PLC0415

        from stdapi.routes import anthropic_messages_batches as routes  # noqa: PLC0415

        url = routes._results_url(  # noqa: SLF001
            Request(
                {
                    "type": "http",
                    "scheme": "https",
                    "root_path": "",
                    "path": f"{_PATH}/msgbatch_x",
                    "query_string": b"",
                    "headers": [(b"host", b"host")],
                }
            ),
            "msgbatch_x",
        )
        assert url == f"https://host{_PATH}/msgbatch_x/results"
        # What anthropic._base_client._prepare_url makes of it.
        assert not URL(url).is_relative_url
        assert str(URL(url)) == url


@pytest.mark.slow
@pytest.mark.usefixtures("anthropic_batches_api")
class TestMessageBatchLive:
    """Submitting a real Message Batch, and cancelling it before it costs anything.

    Submission is the one step no double can stand for: it needs the operator's
    batch service role and the backing job service to accept the payload.
    Cancelling straight away keeps the whole proof to that submission — the
    requests never run.

    Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
         stdapi/batches.py:create_batch
    """

    def test_create_then_cancel(
        self,
        anthropic_client: Anthropic,
        anthropic_chat_model: str,
        use_official_api: bool,
    ) -> None:
        """A Message Batch is accepted as in progress, then cancelled unstarted.

        Cancellation is asynchronous on both targets — the backing work is asked
        to stop and the batch reports ``canceling`` until it has — so the
        terminal state is not waited for.

        Upstream imposes no per-model minimum, so the vendor lane submits a
        single request: whatever starts before the cancellation propagates is
        billed for real.

        Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
             stdapi/batches.py:cancel_batch
        """
        count = 1 if use_official_api else batches.MIN_REQUESTS_PER_MODEL
        batch = anthropic_client.messages.batches.create(
            requests=[
                {
                    "custom_id": f"req-{index}",
                    "params": {
                        "model": anthropic_chat_model,
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                }
                for index in range(count)
            ]
        )
        try:
            assert batch.type == "message_batch"
            assert batch.processing_status == "in_progress"
            assert anthropic_client.messages.batches.retrieve(batch.id).id == batch.id

            cancelled = anthropic_client.messages.batches.cancel(batch.id)
            assert cancelled.id == batch.id
            assert cancelled.cancel_initiated_at is not None
            assert cancelled.processing_status in {"canceling", "ended"}
        finally:
            with contextlib.suppress(Exception):
                anthropic_client.messages.batches.cancel(batch.id)
