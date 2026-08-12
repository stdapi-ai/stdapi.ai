"""Tests for the OpenAI-compatible /v1/batches routes.

Upstream forbids a mixed-model input file, caps a batch at 50,000 requests and
does not guarantee the order of the output lines, so results are matched by
``custom_id``. The backing inference jobs impose a floor of 100 requests per
model and can serve neither tool use nor a structured output schema, so those
are refused at submit time rather than surfacing as per-line failures nobody
paid for.

Everything below the live class runs against in-memory doubles for the object
store and the job service, so the whole surface is covered without an AWS
account. Submitting for real is the one step those doubles cannot stand in for,
and it is what ``TestOpenAIBatchLive`` does.

Ref: https://developers.openai.com/api/docs/guides/batch.md
     https://stdapi.ai/api_openai_batches/
     stdapi/routes/openai_batches.py:create
     stdapi/batches.py:create_batch
"""

import contextlib
from json import dumps
from typing import TYPE_CHECKING, Any

import pytest

from stdapi import batches
from tests import _batches
from tests._batches import chat_lines, converse_output

if TYPE_CHECKING:
    from openai import OpenAI
    from starlette.testclient import TestClient


#: Chat completion body asking for a cache point in both cacheable components.
_CACHED_COMPLETION: dict[str, Any] = {
    "model": "amazon.nova-micro-v1:0",
    "messages": [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "context",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "hi",
                    "prompt_cache_breakpoint": {"mode": "explicit"},
                }
            ],
        },
    ],
}


def _create(client: TestClient, file_id: str) -> dict[str, Any]:
    """Submit a batch for *file_id* and return the decoded response body."""
    response = client.post(
        "/v1/batches",
        json={
            "input_file_id": file_id,
            "endpoint": "/v1/chat/completions",
            "completion_window": "24h",
        },
    )
    return {"http_status": response.status_code, **response.json()}


@pytest.mark.local
class TestOpenAIBatchValidation:
    """POST /v1/batches: what a batch refuses, and with which message.

    Every case here is refused before any request runs, so the client learns of
    the problem in the create call instead of in a results file hours later.

    Ref: https://developers.openai.com/api/docs/guides/batch.md
         stdapi/batches.py:prepare_openai_requests
    """

    def test_below_minimum_names_the_floor(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A batch under the per-model floor is refused, naming the floor.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html
             stdapi/batches.py:_group_by_model
        """
        _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(99))
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        error = body["error"]
        assert isinstance(error, dict)
        assert str(batches.MIN_REQUESTS_PER_MODEL) in error["message"]
        assert "amazon.nova-micro-v1:0 (99)" in error["message"]

    def test_mixed_models_are_refused(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file naming two models is refused, as upstream forbids one.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:prepare_openai_requests
        """
        _batches.install(monkeypatch)
        lines = chat_lines(100) + chat_lines(
            100, model="amazon.nova-lite-v1:0", prefix="other"
        )
        file_id = _batches.install_input_file(monkeypatch, lines)
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        error = body["error"]
        assert isinstance(error, dict)
        assert "same model" in error["message"]

    def test_wrong_purpose_is_refused(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file not uploaded for batching is refused, naming the purpose.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:read_input_requests
        """
        _batches.install(monkeypatch)
        file_id = _batches.install_input_file(
            monkeypatch, chat_lines(100), purpose="assistants"
        )
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        error = body["error"]
        assert isinstance(error, dict)
        assert "purpose 'assistants'" in error["message"]

    def test_wrong_url_is_refused(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A line targeting another endpoint is refused, naming its position.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:prepare_openai_requests
        """
        _batches.install(monkeypatch)
        lines = chat_lines(100)
        lines[3] = lines[3].replace("/v1/chat/completions", "/v1/embeddings")
        file_id = _batches.install_input_file(monkeypatch, lines)
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        error = body["error"]
        assert isinstance(error, dict)
        assert "Line 4" in error["message"]

    def test_malformed_line_is_refused(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A line that is not a JSON object is refused, naming its position.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:read_input_requests
        """
        _batches.install(monkeypatch)
        lines = chat_lines(100)
        lines[7] = '"not an object"'
        file_id = _batches.install_input_file(monkeypatch, lines)
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        error = body["error"]
        assert isinstance(error, dict)
        assert "Line 8" in error["message"]

    def test_truncated_line_is_refused_with_its_position(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A line that is not valid JSON is a 400 naming it, never a 500.

        A file written by a crashed producer ends mid-object, and the decoder's
        own `ValueError` reaches no handler.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:read_input_requests
        """
        _batches.install(monkeypatch)
        lines = chat_lines(100)
        lines[7] = '{"custom_id": "req-7", "body": {'
        file_id = _batches.install_input_file(monkeypatch, lines)
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        error = body["error"]
        assert isinstance(error, dict)
        assert "Line 8" in error["message"]

    def test_an_over_cap_file_is_refused_while_it_is_read(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Too many requests is refused during the read, not after it.

        The cap is lowered rather than writing 50,001 lines; what is asserted
        is that the refusal happens without the whole file being decoded first,
        which is what keeps a legal 200 MB file from exhausting the server.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:read_input_requests
        """
        _batches.install(monkeypatch)
        monkeypatch.setitem(batches.MAX_REQUESTS, "openai", 5)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(20))
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        error = body["error"]
        assert isinstance(error, dict)
        assert "at most 5 requests" in error["message"]

    def test_more_than_one_completion_per_request_is_refused(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`n` above 1 is refused at submit: a batch answers once per request.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:_prepare_openai_request
        """
        _batches.install(monkeypatch)
        lines = chat_lines(100)
        lines[1] = lines[1].replace('"messages"', '"n": 5, "messages"')
        file_id = _batches.install_input_file(monkeypatch, lines)
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        error = body["error"]
        assert isinstance(error, dict)
        assert "Line 2: 'n' must be 1" in error["message"]

    def test_a_non_post_method_is_refused(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A line asking for another HTTP method is refused, naming its position.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:prepare_openai_requests
        """
        _batches.install(monkeypatch)
        lines = chat_lines(100)
        lines[4] = lines[4].replace('"method": "POST"', '"method": "GET"')
        file_id = _batches.install_input_file(monkeypatch, lines)
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        error = body["error"]
        assert isinstance(error, dict)
        assert "Line 5: 'method' must be 'POST'" in error["message"]

    def test_an_empty_file_is_refused(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file holding no request is refused, naming the file.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:read_input_requests
        """
        _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, [])
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        error = body["error"]
        assert isinstance(error, dict)
        assert "holds no request" in error["message"]

    def test_an_over_long_custom_id_is_refused(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A custom_id past the length cap is refused: it must round-trip whole.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:_validate_custom_ids
        """
        _batches.install(monkeypatch)
        lines = chat_lines(100)
        lines[6] = lines[6].replace('"req-6"', f'"{"x" * 65}"')
        file_id = _batches.install_input_file(monkeypatch, lines)
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        error = body["error"]
        assert isinstance(error, dict)
        assert "Line 7: 'custom_id' must be between 1 and 64" in error["message"]

    def test_duplicate_custom_id_is_refused(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A repeated custom_id is refused: it is the only join key results have.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:_validate_custom_ids
        """
        _batches.install(monkeypatch)
        lines = chat_lines(100)
        lines[5] = lines[5].replace('"req-5"', '"req-4"')
        file_id = _batches.install_input_file(monkeypatch, lines)
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        error = body["error"]
        assert isinstance(error, dict)
        assert "more than once" in error["message"]

    def test_tools_are_refused(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tool use is refused at submit, not left to fail per line.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html
             stdapi/batches.py:_check_batchable
        """
        _batches.install(monkeypatch)
        lines = chat_lines(100)
        lines[0] = lines[0].replace(
            '"messages"',
            '"tools": [{"type": "function", "function": {"name": "f"}}], "messages"',
        )
        file_id = _batches.install_input_file(monkeypatch, lines)
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        error = body["error"]
        assert isinstance(error, dict)
        assert "tool use is not available" in error["message"].lower()

    def test_json_schema_response_format_is_refused(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A structured output schema is refused at submit.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html
             stdapi/batches.py:_check_batchable
        """
        _batches.install(monkeypatch)
        lines = chat_lines(100)
        lines[0] = lines[0].replace(
            '"messages"',
            '"response_format": {"type": "json_schema", "json_schema": '
            '{"name": "s", "schema": {"type": "object"}}}, "messages"',
        )
        file_id = _batches.install_input_file(monkeypatch, lines)
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        error = body["error"]
        assert isinstance(error, dict)
        assert "structured output schema" in error["message"]

    def test_streaming_line_is_refused(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A line asking to stream is refused: a batch has nothing to stream to.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:_prepare_openai_request
        """
        _batches.install(monkeypatch)
        lines = chat_lines(100)
        lines[2] = lines[2].replace('"messages"', '"stream": true, "messages"')
        file_id = _batches.install_input_file(monkeypatch, lines)
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        error = body["error"]
        assert isinstance(error, dict)
        assert "Line 3: 'stream' is not available" in error["message"]

    def test_unknown_endpoint_is_refused(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An endpoint the server does not batch is refused by the schema.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/types/openai_batches.py:BatchCreateParams
        """
        _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        response = app_client.post(
            "/v1/batches",
            json={
                "input_file_id": file_id,
                "endpoint": "/v1/embeddings",
                "completion_window": "24h",
            },
        )
        assert response.status_code == 400
        assert "endpoint" in response.json()["error"]["message"]

    def test_a_guarded_request_is_refused(self) -> None:
        """A request a configured guardrail covers is refused, not run unguarded.

        The guardrail is attached during translation rather than named by the
        client, so the guard is asserted on the translated request — the only
        place it is visible.

        Ref: stdapi/batches.py:_check_batchable
        """
        from stdapi.api_errors import ApiError  # noqa: PLC0415

        with pytest.raises(ApiError) as raised:
            batches._check_batchable(  # noqa: SLF001
                {
                    "guardrailConfig": {
                        "guardrailIdentifier": "gr",
                        "guardrailVersion": "1",
                    }
                },  # type: ignore[typeddict-item]
                0,
                "Line",
            )
        assert "content guardrails are not available" in str(raised.value)

    def test_a_built_in_tool_is_refused(self) -> None:
        """A tool carried as a model-specific field is refused like any other.

        A built-in tool never reaches `tools`, so a guard reading only that
        parameter would let it through and fail per line hours later.

        Ref: stdapi/batches.py:_check_batchable
        """
        from stdapi.api_errors import ApiError  # noqa: PLC0415

        with pytest.raises(ApiError) as raised:
            batches._check_batchable(  # noqa: SLF001
                {"additionalModelRequestFields": {"tools": [{"type": "web_search"}]}},  # type: ignore[typeddict-item]
                0,
                "Line",
            )
        assert "tool use is not available" in str(raised.value).lower()

    def test_oversized_metadata_is_refused(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A metadata value past the documented cap is refused at creation.

        The metadata is stored with the batch and re-read on every retrieve and
        every listing page, so an unbounded value is an amplification vector.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/types/openai_batches.py:BatchCreateParams
        """
        _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        response = app_client.post(
            "/v1/batches",
            json={
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "metadata": {"note": "x" * 513},
            },
        )
        assert response.status_code == 400
        assert "at most 512" in response.json()["error"]["message"]

    def test_disabled_without_service_role(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the service role the routes answer 503, not a backend error.

        Ref: stdapi/batches.py:require_batches_enabled
        """
        from stdapi.config import SETTINGS  # noqa: PLC0415

        _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        monkeypatch.setattr(SETTINGS, "aws_bedrock_batch_role_arn", None)
        body = _create(app_client, file_id)
        assert body["http_status"] == 503
        error = body["error"]
        assert isinstance(error, dict)
        assert "not available on the current server" in error["message"]


@pytest.mark.local
class TestOpenAIBatchLifecycle:
    """POST/GET /v1/batches: the states a batch reports, and its results.

    The batch status is derived from the backing jobs on every read, so the
    transitions are exercised by moving the jobs rather than by any stored
    status.

    Ref: https://developers.openai.com/api/docs/guides/batch.md
         stdapi/routes/openai_batches.py:_status
    """

    def test_create_reports_validating(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A submitted batch is `validating`, with its counts and metadata.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/routes/openai_batches.py:create
        """
        _, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        response = app_client.post(
            "/v1/batches",
            json={
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "metadata": {"job": "nightly"},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["id"].startswith("batch_")
        assert body["status"] == "validating"
        assert body["object"] == "batch"
        assert body["input_file_id"] == file_id
        assert body["request_counts"] == {"total": 100, "completed": 0, "failed": 0}
        assert body["metadata"] == {"job": "nightly"}
        assert body["expires_at"] == body["created_at"] + 24 * 3600
        assert len(bedrock.created) == 1
        assert bedrock.created[0]["modelInvocationType"] == "Converse"
        assert bedrock.created[0]["roleArn"] == _batches.ROLE_ARN

    def test_submitted_requests_carry_the_custom_id(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each submitted request is keyed by its custom_id, the results' join key.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:_submit_job
        """
        from pydantic_core import from_json  # noqa: PLC0415

        s3, _ = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        assert _create(app_client, file_id)["http_status"] == 200
        written = next(
            body for (_b, key), body in s3.objects.items() if key.endswith(".jsonl")
        )
        records = [from_json(line) for line in written.splitlines()]
        assert [record["recordId"] for record in records] == [
            f"req-{index}" for index in range(100)
        ]
        assert "modelId" not in records[0]["modelInput"]

    def test_prompt_cache_hints_are_accepted_and_dropped(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cache breakpoint is accepted, and no cache point is submitted with it.

        Batched requests read and write no cache, and one carrying a cache
        point fails — every record of the batch, not just the one that carried
        it. The hint is an optimization the caller can do without, so it is
        dropped and the request runs, at the batch price it asked for.

        The translated request is captured on its way in, so the assertion on
        the submitted body cannot pass for a request that never carried a
        cache point to begin with.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/prompt-caching.html
             stdapi/batches.py:_to_model_input
        """
        from pydantic_core import from_json  # noqa: PLC0415

        s3, _ = _batches.install(monkeypatch, translate=True)
        translated = _batches.capture_translations(monkeypatch)
        file_id = _batches.install_input_file(
            monkeypatch, chat_lines(100, body=dumps(_CACHED_COMPLETION))
        )
        assert _create(app_client, file_id)["http_status"] == 200
        assert translated[0]["system"][-1] == {"cachePoint": {"type": "default"}}
        assert translated[0]["messages"][0]["content"][-1] == {
            "cachePoint": {"type": "default"}
        }
        written = next(
            body for (_b, key), body in s3.objects.items() if key.endswith(".jsonl")
        )
        records = [from_json(line) for line in written.splitlines()]
        model_input = records[0]["modelInput"]
        assert "cachePoint" not in dumps(model_input)
        assert model_input["system"] == [{"text": "context"}]
        assert model_input["messages"][0]["content"] == [{"text": "hi"}]

    def test_completed_batch_publishes_its_results(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once the jobs end, the batch is `completed` and names its results file.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:materialize_openai_results
        """
        s3, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        batch_id = _create(app_client, file_id)["id"]
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
        body = app_client.get(f"/v1/batches/{batch_id}").json()
        assert body["status"] == "completed"
        assert "error_file_id" not in body
        assert body["usage"]["input_tokens"] == 500
        assert body["usage"]["output_tokens"] == 300
        lines = _batches.read_result_file(s3, body["output_file_id"])
        assert len(lines) == 100
        assert {line["custom_id"] for line in lines} == {
            f"req-{index}" for index in range(100)
        }
        first = lines[0]
        assert first["response"]["status_code"] == 200
        assert first["error"] is None
        assert first["response"]["body"]["choices"][0]["message"]["content"] == "answer"

    def test_usage_is_billed_once_at_the_batch_rate(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A finished batch bills the job's counters once, at the batch tier.

        The batch tier is the whole point of batching — priced at roughly half
        the on-demand rate — and it is only observable in the usage log, so
        nothing else would catch a batch billed as a standard invocation.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference.html
             stdapi/batches.py:settle
        """
        from tests.conftest import logged_usage_entries  # noqa: PLC0415

        s3, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        batch_id = _create(app_client, file_id)["id"]
        bedrock.finish(succeeded=100)
        _batches.write_job_output(
            s3,
            bedrock,
            [
                {"recordId": f"req-{index}", "modelOutput": converse_output("answer")}
                for index in range(100)
            ],
            {
                "inputTokenCount": 500,
                "outputTokenCount": 300,
                "cacheReadInputTokenCount": 100,
                "cacheWriteInputTokenCount": 50,
            },
        )
        capfd.readouterr()
        assert app_client.get(f"/v1/batches/{batch_id}").status_code == 200
        (entry,) = logged_usage_entries(
            capfd.readouterr().out, service="bedrock-runtime"
        )
        assert entry["tier"] == "batch"
        assert entry["model"] == "amazon.nova-micro-v1:0"
        assert entry["region"] == _batches.REGION
        assert entry["input_tokens"] == 500
        assert entry["output_tokens"] == 300
        assert entry["cached_tokens"] == 100
        assert entry["cache_write_tokens"] == 50

        assert app_client.get(f"/v1/batches/{batch_id}").status_code == 200
        assert not logged_usage_entries(
            capfd.readouterr().out, service="bedrock-runtime"
        ), "a second read of an ended batch must not bill it again"

    def test_the_results_file_id_resolves_back_to_its_bucket(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The published `output_file_id` names the bucket the results are in.

        The identifier is derived from the batch's own and carries a
        fingerprint of the bucket, which the Files API reads back to find the
        object: a derivation off by one byte still writes the results but makes
        them undownloadable, which is the only way a client can read them.

        Ref: stdapi/batches.py:_derive_file_payload
             stdapi/files/_core.py:resolve_file_bucket
        """
        from binascii import crc32  # noqa: PLC0415

        from stdapi.files import (  # noqa: PLC0415
            _core,
            parse_file_id,
            resolve_file_bucket,
        )

        s3, bedrock = _batches.install(monkeypatch)
        monkeypatch.setitem(
            _core._BUCKET_CRC32,  # noqa: SLF001
            crc32(_batches.BUCKET.encode()),
            _batches.BUCKET,
        )
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        batch_id = _create(app_client, file_id)["id"]
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
        body = app_client.get(f"/v1/batches/{batch_id}").json()
        payload = parse_file_id(body["output_file_id"])
        assert resolve_file_bucket(payload) == _batches.BUCKET

    def test_failed_requests_go_to_the_error_file(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed request lands in `error_file_id` with a message of our own.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:iter_openai_results
        """
        s3, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        batch_id = _create(app_client, file_id)["id"]
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
                        "errorMessage": (
                            "Tool use is not supported for batch inference "
                            "Converse requests."
                        ),
                        "expired": False,
                    },
                },
            ],
            {"inputTokenCount": 495, "outputTokenCount": 297},
        )
        body = app_client.get(f"/v1/batches/{batch_id}").json()
        assert body["status"] == "completed"
        assert body["request_counts"] == {"total": 100, "completed": 99, "failed": 1}
        assert len(_batches.read_result_file(s3, body["output_file_id"])) == 99
        errors = _batches.read_result_file(s3, body["error_file_id"])
        assert len(errors) == 1
        assert errors[0]["custom_id"] == "req-99"
        assert errors[0]["response"] is None
        assert errors[0]["error"]["code"] == "invalid_request_error"
        assert "Converse" not in errors[0]["error"]["message"]

    def test_cancel_reaches_every_job_and_is_idempotent(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cancelling stops the jobs; cancelling again changes nothing.

        Stopping is asynchronous, so the batch reports `cancelling` until the
        jobs have actually stopped.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:cancel_batch
        """
        _, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        batch_id = _create(app_client, file_id)["id"]
        first = app_client.post(f"/v1/batches/{batch_id}/cancel").json()
        assert first["status"] == "cancelling"
        assert first["cancelling_at"] is not None
        assert bedrock.stopped
        second = app_client.post(f"/v1/batches/{batch_id}/cancel").json()
        assert second["status"] == "cancelling"
        assert second["cancelling_at"] == first["cancelling_at"]
        bedrock.finish(status="Stopped", succeeded=0)
        ended = app_client.get(f"/v1/batches/{batch_id}").json()
        assert ended["status"] == "cancelled"
        assert ended["cancelled_at"] is not None

    def test_cancelling_a_completed_batch_leaves_it_completed(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cancel arriving after the batch completed does not rewrite it.

        Upstream refuses to cancel a completed batch; here the call is a
        no-op, so a retry loop cannot turn a paid-for `completed` batch into a
        `cancelled` one with no completion time.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:cancel_batch
        """
        _, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        batch_id = _create(app_client, file_id)["id"]
        bedrock.finish(succeeded=100)
        body = app_client.post(f"/v1/batches/{batch_id}/cancel").json()
        assert body["status"] == "completed"
        assert body["completed_at"] is not None
        assert "cancelling_at" not in body
        assert not bedrock.stopped
        assert app_client.get(f"/v1/batches/{batch_id}").json()["status"] == "completed"

    def test_a_refused_cancel_is_not_answered_as_a_success(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cancel the backend refuses fails, instead of reporting `cancelling`.

        A deployment whose role cannot stop jobs would otherwise answer 200,
        record the cancellation, and let the batch run for a day at full cost
        while every poll claims it is stopping.

        Ref: stdapi/batches.py:_stop_job
        """
        _, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        batch_id = _create(app_client, file_id)["id"]
        bedrock.stop_error = "AccessDeniedException"
        response = app_client.post(f"/v1/batches/{batch_id}/cancel")
        assert response.status_code == 403
        after = app_client.get(f"/v1/batches/{batch_id}").json()
        assert after["status"] == "validating"
        assert "cancelling_at" not in after

    def test_a_recent_batch_is_listed_past_the_scan_window(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A batch stays listable once storage holds more objects than one scan.

        Batch payloads sort by creation time and storage walks them oldest
        first, so a scan that stopped at its budget would show only the oldest
        batches and never a recent one.

        Ref: stdapi/batches.py:_scan_bucket
        """
        from stdapi.config import SETTINGS  # noqa: PLC0415

        s3, _ = _batches.install(monkeypatch)
        for index in range(1500):
            key = f"{SETTINGS.aws_s3_batches_prefix}{index:032x}"
            s3.objects[_batches.BUCKET, key] = b"{}"
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        batch_id = _create(app_client, file_id)["id"]
        body = app_client.get("/v1/batches?limit=10").json()
        assert [item["id"] for item in body["data"]] == [batch_id]

    def test_failed_job_reports_a_failed_batch(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A job that stops without running its requests makes the batch `failed`.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/routes/openai_batches.py:_status
        """
        _, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        batch_id = _create(app_client, file_id)["id"]
        bedrock.finish(status="Failed", succeeded=0)
        body = app_client.get(f"/v1/batches/{batch_id}").json()
        assert body["status"] == "failed"
        assert body["failed_at"] is not None
        # No results were produced, so none are announced; the reason is.
        assert "output_file_id" not in body
        assert body["errors"]["data"][0]["code"] == "batch_failed"

    def test_listing_returns_the_batch(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A created batch shows up in the listing, newest first.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:list_batches
        """
        _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        batch_id = _create(app_client, file_id)["id"]
        body = app_client.get("/v1/batches?limit=10").json()
        assert body["object"] == "list"
        assert [item["id"] for item in body["data"]] == [batch_id]
        assert body["has_more"] is False

    def test_unknown_batch_is_not_found(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An identifier that decodes but names nothing answers 404.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:_read_record
        """
        _batches.install(monkeypatch)
        response = app_client.get(f"/v1/batches/batch_{'a' * 32}")
        assert response.status_code == 404
        assert "No batch found" in response.json()["error"]["message"]

    def test_a_message_batch_is_not_reachable_here(
        self,
        app_client: TestClient,
        anthropic_app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A batch created on the Messages API is not readable as an OpenAI batch.

        Ref: stdapi/batches.py:_read_record
        """
        _batches.install(monkeypatch)
        created = anthropic_app_client.post(
            "/anthropic/v1/messages/batches",
            json={
                "requests": [
                    {
                        "custom_id": f"req-{index}",
                        "params": {
                            "model": "amazon.nova-micro-v1:0",
                            "max_tokens": 16,
                            "messages": [{"role": "user", "content": "hi"}],
                        },
                    }
                    for index in range(100)
                ]
            },
        )
        assert created.status_code == 200
        payload = created.json()["id"].removeprefix("msgbatch_")
        assert app_client.get(f"/v1/batches/batch_{payload}").status_code == 404


@pytest.mark.local
class TestBatchResultTranslation:
    """The results translation, as a pure function over recorded backend output.

    Batched Converse output serialises every unused member of a union as null;
    the response adapters read a present key as a present value, so the nulls
    have to be dropped before the adapters see them.

    Ref: stdapi/batches.py:_strip_nulls
         stdapi/models/chat/_adapters/_openai_chat_completion.py:format_response
    """

    def test_nulls_are_dropped_before_the_adapter(self) -> None:
        """A null-filled union member does not reach the adapter as a key.

        Ref: stdapi/batches.py:_strip_nulls
        """
        stripped = batches._strip_nulls(converse_output("hello"))  # noqa: SLF001
        block = stripped["output"]["message"]["content"][0]
        assert block == {"text": "hello"}
        assert "cacheReadInputTokens" not in stripped["usage"]

    async def test_completion_body_carries_the_answer(
        self, request_log: dict[str, Any]
    ) -> None:
        """A successful line becomes a chat completion with the model's text.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:_to_completion
        """
        body = await batches._to_completion(  # noqa: SLF001
            {"recordId": "req-1", "modelOutput": converse_output("hello")},
            "amazon.nova-micro-v1:0",
            "req-1",
            1_760_000_000,
        )
        assert body["object"] == "chat.completion"
        assert body["model"] == "amazon.nova-micro-v1:0"
        assert body["choices"][0]["message"]["content"] == "hello"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["prompt_tokens"] == 5

    def test_backend_wording_never_reaches_the_client(self) -> None:
        """A per-request failure is reported without the backend's own wording.

        Ref: stdapi/batches.py:_record_error
        """
        code, message = batches._record_error(  # noqa: SLF001
            {
                "error": {
                    "errorCode": 400,
                    "errorMessage": "Remove the toolConfig field from your request",
                    "expired": False,
                }
            }
        ) or ("", "")
        assert code == "invalid_request_error"
        assert "toolConfig" not in message

    def test_an_expired_request_is_reported_as_expired(self) -> None:
        """A request that ran out of time is told apart from one that failed.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:_record_error
        """
        result = batches._record_error(  # noqa: SLF001
            {"error": {"errorCode": 408, "errorMessage": "gone", "expired": True}}
        )
        assert result is not None
        assert result[0] == "expired"

    def test_a_successful_line_reports_no_error(self) -> None:
        """A line carrying output is not mistaken for a failure.

        Ref: stdapi/batches.py:_record_error
        """
        assert (
            batches._record_error(  # noqa: SLF001
                {"recordId": "req-1", "modelOutput": converse_output("hi")}
            )
            is None
        )


@pytest.mark.slow
@pytest.mark.usefixtures("batches_api")
class TestOpenAIBatchLive:
    """Submitting a real batch, and cancelling it before it costs anything.

    Submission is the one step no double can stand for: it needs the operator's
    batch service role, the input file in the object store and the backing job
    service to accept the payload. Cancelling straight away keeps the whole
    proof to that submission — the requests never run.

    Ref: https://developers.openai.com/api/docs/guides/batch.md
         stdapi/batches.py:create_batch
    """

    def test_create_then_cancel(
        self, openai_client: OpenAI, chat_model: str, use_official_api: bool
    ) -> None:
        """A batch is accepted with its own metadata, then cancelled unstarted.

        The created batch is echoed with the input file, endpoint, window and
        metadata it was submitted with. Cancellation is asynchronous on both
        targets — the backing work is asked to stop and the batch reports
        ``cancelling`` until it has — so the terminal state is not waited for.

        Upstream imposes no per-model minimum, so the vendor lane submits a
        single request: whatever starts before the cancellation propagates is
        billed for real.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:cancel_batch
        """
        count = 1 if use_official_api else batches.MIN_REQUESTS_PER_MODEL
        lines = "\n".join(
            dumps(
                {
                    "custom_id": f"req-{index}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": chat_model,
                        "messages": [{"role": "user", "content": "hi"}],
                    },
                }
            )
            for index in range(count)
        )
        input_file = openai_client.files.create(
            file=("requests.jsonl", lines.encode()), purpose="batch"
        )
        batch = None
        try:
            batch = openai_client.batches.create(
                input_file_id=input_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"suite": "stdapi-live"},
            )
            assert batch.object == "batch"
            assert batch.input_file_id == input_file.id
            assert batch.endpoint == "/v1/chat/completions"
            assert batch.completion_window == "24h"
            assert batch.metadata == {"suite": "stdapi-live"}
            assert batch.status in {"validating", "in_progress"}
            assert openai_client.batches.retrieve(batch.id).id == batch.id

            cancelled = openai_client.batches.cancel(batch.id)
            assert cancelled.id == batch.id
            assert cancelled.cancelling_at is not None
            assert cancelled.status in {"cancelling", "cancelled"}
        finally:
            if batch is not None:
                with contextlib.suppress(Exception):
                    openai_client.batches.cancel(batch.id)
            with contextlib.suppress(Exception):
                openai_client.files.delete(input_file.id)
