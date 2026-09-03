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
from base64 import b32hexencode
from binascii import crc32
from datetime import UTC, datetime
from itertools import count
from json import dumps, loads
from math import sqrt
from time import monotonic, sleep, time
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

import pytest
from botocore.exceptions import ClientError

from stdapi import batches
from stdapi.files import payload_created_at
from tests import _batches
from tests._batches import chat_lines, converse_output

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types import Batch
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


#: Alias name the resolution tests write instead of a model ID.
_ALIAS = "nova-fast"

#: Model ``_ALIAS`` resolves to.
_ALIAS_TARGET = "amazon.nova-micro-v1:0"


def _resolve_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the batch's model validation resolve ``_ALIAS`` to its target.

    The batch doubles answer every name with itself, which is what a catalogue
    lookup does for a model ID and not what it does for an alias.

    Args:
        monkeypatch: The test's patcher.
    """
    from tests._helpers import make_model_details  # noqa: PLC0415

    async def _validate_model(model_id: str, *_args: object, **_kwargs: object) -> Any:  # noqa: ANN401
        return make_model_details(_ALIAS_TARGET if model_id == _ALIAS else model_id)

    monkeypatch.setattr(batches, "validate_model", _validate_model)


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

    def test_mixed_models_are_refused_before_any_line_is_translated(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mixed-model file is refused before a single line is translated.

        Translating a line can mean a remote fetch for the content it carries,
        so a file that will be refused anyway must not pay for translating any
        of its lines — the model names are compared first instead.

        Ref: stdapi/batches.py:prepare_openai_requests
        """
        _batches.install(monkeypatch)
        translated: list[Any] = []
        original = batches._prepare_all  # noqa: SLF001

        async def _tracked(items: Any, prepare: Any) -> Any:  # noqa: ANN401
            translated.extend(items)
            return await original(items, prepare)

        monkeypatch.setattr(batches, "_prepare_all", _tracked)
        lines = chat_lines(100) + chat_lines(
            100, model="amazon.nova-lite-v1:0", prefix="other"
        )
        file_id = _batches.install_input_file(monkeypatch, lines)
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        assert not translated

    def test_two_names_of_one_model_are_one_model(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file naming a model twice, once by alias, is one model and one job.

        The names are compared once resolved, so an alias -- or a wildcard
        pattern -- next to its own target is neither refused as two models nor
        submitted as two jobs consuming two of the batch's model slots.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:_group_by_model
             stdapi/batches.py:prepare_openai_requests
        """
        _, bedrock = _batches.install(monkeypatch)
        _resolve_alias(monkeypatch)
        lines = chat_lines(50) + chat_lines(50, model=_ALIAS, prefix="alias")
        file_id = _batches.install_input_file(monkeypatch, lines)
        body = _create(app_client, file_id)
        assert body["http_status"] == 200
        assert [job["modelId"] for job in bedrock.created] == [_ALIAS_TARGET]

    def test_a_model_name_is_resolved_once_for_the_whole_file(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every line naming one model resolves it once, not once per line.

        A pattern is matched against the whole catalogue on each resolution, so
        resolving per line makes one file scan it up to 50,000 times; worse,
        the catalogue is refreshed on the way, so the same pattern could pick a
        newer model half-way down the file and split one batch into two jobs.
        The name is therefore pinned to what it first resolved to.

        Ref: https://stdapi.ai/api_openai_batches/#model-support
             stdapi/batches.py:_resolve_model
        """
        from tests._helpers import make_model_details  # noqa: PLC0415

        _, bedrock = _batches.install(monkeypatch)
        pattern = "amazon.nova-mic*"
        resolutions: list[str] = []

        async def _validate_model(model_id: str, *_args: object, **_kw: object) -> Any:  # noqa: ANN401
            resolutions.append(model_id)
            return make_model_details(_ALIAS_TARGET)

        monkeypatch.setattr(batches, "validate_model", _validate_model)
        file_id = _batches.install_input_file(
            monkeypatch, chat_lines(200, model=pattern)
        )
        body = _create(app_client, file_id)
        assert body["http_status"] == 200
        assert resolutions.count(pattern) == 1
        assert [job["modelId"] for job in bedrock.created] == [_ALIAS_TARGET]

    def test_a_refused_model_costs_one_wave_of_resolutions(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file whose model is refused resolves one bounded wave, not every name.

        A file may name one model per request, and resolving an ARN calls the
        backend under a process-wide lock: starting every name at once would
        leave the ones a refused file had already scheduled running long after
        the client read its 400, with the deployment's other requests queued
        behind them. The names are resolved in the same waves the translation
        uses, and the first refusal cancels the rest of its wave.

        Ref: stdapi/batches.py:_resolve_distinct
        """
        from stdapi.api_errors import ApiError  # noqa: PLC0415
        from tests._helpers import make_model_details  # noqa: PLC0415

        _batches.install(monkeypatch)
        refused = "amazon.nova-000"
        started: list[str] = []

        async def _validate_model(model_id: str, *_args: object, **_kw: object) -> Any:  # noqa: ANN401
            started.append(model_id)
            if model_id == refused:
                msg = f"The model `{model_id}` is refused."
                raise ApiError(msg)
            return make_model_details(model_id)

        monkeypatch.setattr(batches, "validate_model", _validate_model)
        names = [f"amazon.nova-{index:03d}" for index in range(200)]
        lines = [
            line
            for index, name in enumerate(names)
            for line in chat_lines(1, model=name, prefix=f"m{index}")
        ]
        file_id = _batches.install_input_file(monkeypatch, lines)
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        assert len(started) <= batches._BUILD_CONCURRENCY < len(names)  # noqa: SLF001

    def test_batch_reports_the_model_that_runs_it(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The batch object names the resolved model, not the name the client wrote.

        A job runs for hours: the name it reports has to stay meaningful after
        the request that created it, which the caller's own spelling does not.

        Ref: https://developers.openai.com/api/docs/guides/batch.md
             stdapi/routes/openai_batches.py:_to_batch
        """
        _batches.install(monkeypatch)
        _resolve_alias(monkeypatch)
        file_id = _batches.install_input_file(
            monkeypatch, chat_lines(100, model=_ALIAS)
        )
        body = _create(app_client, file_id)
        assert body["http_status"] == 200
        assert body["model"] == _ALIAS_TARGET

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
                "endpoint": "/v1/responses",
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

    def test_a_region_without_storage_names_the_batch_api(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A model whose region has no bucket refuses as the Batch API, not another one.

        The same bucket resolution serves async invocation, the Files API and
        batches; an operator reading "async invocation" in the log for a batch
        that could not start goes looking at a feature nobody used.

        Ref: stdapi/batches.py:_submit_job
             stdapi/aws_s3.py:require_s3_bucket_for_region
        """
        from stdapi.aws_s3 import require_s3_bucket_for_region  # noqa: PLC0415
        from stdapi.config import SETTINGS  # noqa: PLC0415

        _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        monkeypatch.setattr(
            batches, "require_s3_bucket_for_region", require_s3_bucket_for_region
        )
        monkeypatch.setattr(SETTINGS, "aws_bedrock_regions", ["eu-west-1"])
        monkeypatch.setattr(SETTINGS, "aws_s3_regional_buckets", {})

        body = _create(app_client, file_id)

        assert body["http_status"] == 503
        assert body["error"]["message"].startswith("The Batch API is not available")
        logged = capfd.readouterr().out
        assert "aws_s3_regional_buckets" in logged

    def test_a_denied_submission_answers_like_a_disabled_deployment(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A missing permission is the operator's problem, not the caller's.

        ``CreateModelInvocationJob`` is denied both when the role cannot start a
        job and when it cannot pass the batch service role to Amazon Bedrock;
        the caller reads the same message as an unconfigured deployment, while
        the log names both permissions.

        Ref: stdapi/batches.py:_submit_job
             stdapi/api_errors.py:feature_unavailable_guard
        """
        _, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))

        async def _denied(**_kwargs: object) -> dict[str, Any]:
            raise ClientError(
                {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
                "CreateModelInvocationJob",
            )

        monkeypatch.setattr(bedrock, "create_model_invocation_job", _denied)
        body = _create(app_client, file_id)

        assert body["http_status"] == 503
        assert "not available on the current server" in body["error"]["message"]
        assert "PassRole" not in body["error"]["message"]
        logged = capfd.readouterr().out
        assert "bedrock:CreateModelInvocationJob" in logged
        assert "iam:PassRole" in logged
        assert "aws_bedrock_batch_role_arn" in logged

    def test_a_model_the_backend_refuses_is_named_to_the_caller(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model the backend will not batch is refused as that model's limitation.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-supported.html
             stdapi/batches.py:_refused_job
        """
        _, bedrock = _batches.install(monkeypatch)
        bedrock.reject_models = {"amazon.nova-micro-v1:0"}
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))

        body = _create(app_client, file_id)

        assert body["http_status"] == 400
        assert "not available for batched requests" in body["error"]["message"]
        assert "amazon.nova-micro-v1:0" in body["error"]["message"]

    def test_another_validation_failure_does_not_blame_the_model(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A refusal that is not about the model does not report it as one.

        The backend reports a quota, a service role or an account restriction
        as the same flat validation failure as an unbatchable model. Reading
        "the model is not available for batched requests" for a model the
        account simply has not used recently sends both the caller and the
        operator looking in the wrong place, so only the message naming the
        model keeps that answer; the rest is a deployment problem, with the
        real cause in the server log.

        Ref: stdapi/batches.py:_refused_job
             stdapi/api_errors.py:FeatureUnavailableError
        """
        _, bedrock = _batches.install(monkeypatch)
        bedrock.reject_models = {"amazon.nova-micro-v1:0"}
        bedrock.reject_message = (
            "This Model is marked by provider as Legacy and you have not been "
            "actively using the model in the last 30 days."
        )
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        capfd.readouterr()

        body = _create(app_client, file_id)

        assert body["http_status"] == 503
        assert body["error"]["code"] == "feature_unavailable"
        assert "not available for batched requests" not in body["error"]["message"]
        assert "Legacy" not in body["error"]["message"]
        logged = capfd.readouterr().out
        assert "marked by provider as Legacy" in logged
        assert "aws_bedrock_batch_role_arn" in logged

    def test_a_model_not_advertised_for_batch_is_still_submitted(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The catalogue's batch flag is a hint, never admission control.

        It is derived from the published rates, which lag a model's real
        support, so refusing on it locally would lock clients out of models
        the backend accepts. The batch below names a model the catalogue does
        not advertise and must still reach the job service.

        Ref: https://stdapi.ai/api_search_models/
             stdapi/models/__init__.py:sync_batch_support
             stdapi/batches.py:_submit_job
        """
        from stdapi import models as registry  # noqa: PLC0415
        from stdapi.models import EXTRA_MODELS  # noqa: PLC0415
        from stdapi.pricing import Dimension  # noqa: PLC0415
        from tests._helpers import make_model_details  # noqa: PLC0415
        from tests.conftest import set_test_price  # noqa: PLC0415

        model = "amazon.nova-micro-v1:0"
        saved = dict(EXTRA_MODELS)
        try:
            # Priced, but with no batch rate: the model is not advertised.
            set_test_price(
                "amazonnovamicro",
                "us-east-1",
                Dimension.INPUT_TOKENS,
                "0.000001",
                "USD",
            )
            EXTRA_MODELS[model] = make_model_details(model)
            registry.update_unified_models_collections()
            assert EXTRA_MODELS[model].batch is False

            _, bedrock = _batches.install(monkeypatch)
            file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
            body = _create(app_client, file_id)
        finally:
            EXTRA_MODELS.clear()
            EXTRA_MODELS.update(saved)
            registry.update_unified_models_collections()

        assert body["http_status"] == 200
        assert [job["modelId"] for job in bedrock.created] == [model]

    def test_a_failed_job_reports_its_reason_to_the_operator(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """The backend's failure message is logged when a job reports ``Failed``.

        A batch that fails wholesale — the service role cannot read its input,
        typically — carries the reason only in the job description, and the
        client is told nothing but a count of failures.

        Ref: stdapi/batches.py:_to_job_state
        """
        del monkeypatch
        ref = batches.BatchJobRef(
            model="stub-model",
            region=_batches.REGION,
            bucket=_batches.BUCKET,
            job_arn="arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/j1",
            job_id="j1",
            model_id="stub-model",
            requests=100,
            prefix="batches/1/0/",
        )
        response = {
            "status": "Failed",
            "message": "Access denied when calling s3:GetObject on the input.",
            "submitTime": datetime(2026, 8, 12, tzinfo=UTC),
        }

        state = batches._to_job_state(ref, response)  # type: ignore[arg-type] # noqa: SLF001

        assert state.status == "Failed"
        details = " ".join(str(detail) for detail in request_log["error_detail"])
        assert "s3:GetObject" in details
        assert "aws_bedrock_batch_role_arn" in details
        assert request_log["level"] == "warning"

    def test_a_running_job_logs_nothing(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A job still running writes no warning, whatever message it carries.

        Ref: stdapi/batches.py:_to_job_state
        """
        del monkeypatch
        ref = batches.BatchJobRef(
            model="stub-model",
            region=_batches.REGION,
            bucket=_batches.BUCKET,
            job_arn="arn:aws:bedrock:us-east-1:123456789012:model-invocation-job/j1",
            job_id="j1",
            model_id="stub-model",
            requests=100,
            prefix="batches/1/0/",
        )
        response = {
            "status": "InProgress",
            "message": "Validating input",
            "submitTime": datetime(2026, 8, 12, tzinfo=UTC),
        }

        batches._to_job_state(ref, response)  # type: ignore[arg-type] # noqa: SLF001

        assert "error_detail" not in request_log


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

    def _completed_batch(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        extra: dict[str, Any] | None = None,
    ) -> tuple[_batches.FakeS3, dict[str, Any]]:
        """Run a batch of 100 answers to completion and return the store and the batch."""
        s3, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        response = app_client.post(
            "/v1/batches",
            json={
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                **(extra or {}),
            },
        )
        assert response.status_code == 200, response.text
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
        return s3, app_client.get(f"/v1/batches/{response.json()['id']}").json()

    def test_output_expires_after_sets_the_result_file_expiry(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`output_expires_after` expires the result file instead of keeping it forever.

        Result files are the batch's own output: without the policy the caller
        asked for, they stay in storage, and are paid for, until deleted by hand.

        Ref: https://developers.openai.com/api/reference/resources/batches/methods/create
             stdapi/batches.py:materialize_openai_results
        """
        before = int(datetime.now(UTC).timestamp())
        s3, body = self._completed_batch(
            app_client,
            monkeypatch,
            {"output_expires_after": {"anchor": "created_at", "seconds": 3600}},
        )
        assert body["status"] == "completed"
        options = _batches.result_file_options(s3, body["output_file_id"])
        assert int(options["Metadata"]["expires-at"]) >= before + 3600
        assert "stdapi-ai.expires=true" in options["Tagging"]

    def test_result_files_are_kept_when_no_policy_is_asked_for(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control case: an omitted policy keeps the results until deleted.

        Ref: https://developers.openai.com/api/reference/resources/batches/methods/create
             stdapi/files/_core.py:put_file_content
        """
        s3, body = self._completed_batch(app_client, monkeypatch)
        options = _batches.result_file_options(s3, body["output_file_id"])
        assert options["Metadata"]["expires-at"] == ""
        assert "stdapi-ai.expires" not in options["Tagging"]

    @pytest.mark.parametrize(
        "policy",
        [
            {"anchor": "created_at", "seconds": 3599},
            {"anchor": "created_at", "seconds": 2592001},
            {"anchor": "completed_at", "seconds": 3600},
        ],
        ids=["too_short", "too_long", "unknown_anchor"],
    )
    def test_out_of_range_expiry_is_refused(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        policy: dict[str, Any],
    ) -> None:
        """A policy upstream does not accept is refused before the batch runs.

        Ref: https://developers.openai.com/api/reference/resources/batches/methods/create
             stdapi/types/openai_batches.py:BatchOutputExpiresAfter
        """
        _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        response = app_client.post(
            "/v1/batches",
            json={
                "input_file_id": file_id,
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
                "output_expires_after": policy,
            },
        )
        assert response.status_code == 400

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
        while every poll claims it is stopping. The refusal is that role's
        missing `bedrock:StopModelInvocationJob`, so it answers as a feature
        this deployment cannot run -- the same as a denied batch creation,
        rather than a 403 blaming the caller's own key.

        Ref: stdapi/batches.py:_stop_job
             stdapi/api_errors.py:denied_feature_unavailable
        """
        _, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        batch_id = _create(app_client, file_id)["id"]
        bedrock.stop_error = "AccessDeniedException"
        response = app_client.post(f"/v1/batches/{batch_id}/cancel")
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "feature_unavailable"
        after = app_client.get(f"/v1/batches/{batch_id}").json()
        assert after["status"] == "validating"
        assert "cancelling_at" not in after

    def test_a_batch_reports_the_time_it_was_named(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`created_at` is the instant the batch was named, not when submitting ended.

        Creating a batch names it first and then starts one job per model it
        contains, so a wall clock read once that fan-out has finished runs
        later than the identifier the batch is listed under — by the whole
        submission latency. The clock below stands far in the future to prove
        the reported time is not read from it, and the completion window runs
        from the reported time.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:create_batch
        """
        _batches.install(monkeypatch)
        monkeypatch.setattr(batches, "now_utc_timestamp", lambda: 2_000_000_000)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        body = _create(app_client, file_id)
        named = payload_created_at(body["id"].removeprefix("batch_"))
        assert body["created_at"] == named
        assert body["expires_at"] == named + 24 * 3600

    def test_a_batch_stored_with_another_time_keeps_reporting_it(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A stored `created_at` that disagrees with the identifier is reported as stored.

        Batches created by an earlier version recorded `created_at` once
        submission had ended, so it can sit later than the identifier the
        batch is listed under. Reading it back from the identifier instead
        would move both the reported time and the `expires_at` deadline the
        completion window runs from, for every batch already stored.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:_read_record
        """
        s3, _ = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        batch_id = _create(app_client, file_id)["id"]
        payload = batch_id.removeprefix("batch_")
        key = batches.batch_s3_key(payload)
        stored = loads(s3.objects[_batches.BUCKET, key])
        recorded = payload_created_at(payload) + 600
        stored["created_at"] = recorded
        s3.objects[_batches.BUCKET, key] = dumps(stored).encode()

        body = app_client.get(f"/v1/batches/{batch_id}").json()
        assert body["created_at"] == recorded
        assert body["expires_at"] == recorded + 24 * 3600

    def test_paging_a_listing_orders_by_the_times_it_reports(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Batches page back newest first, reporting times in that same order.

        A listing pages in identifier order, so the time each batch reports has
        to be the one its identifier carries: a time read from any other clock
        gives a client a sequence its own timestamps contradict. The reported
        times are therefore asserted against the identifiers rather than merely
        against each other — three batches created inside one second report the
        same value, so a descending-order check alone passes on a listing that
        publishes an order it does not use. The wall clock is moved far from
        the identifiers' instant for the whole creation: it is the negative
        control, and a reported time read from it fails the comparison instead
        of coinciding with it.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:list_batches
        """
        _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        with monkeypatch.context() as clock:
            clock.setattr(batches, "now_utc_timestamp", lambda: 2_000_000_000)
            oldest, middle, newest = (
                _create(app_client, file_id)["id"] for _ in range(3)
            )

        paged: list[dict[str, Any]] = []
        cursor: str | None = None
        for _page in range(3):
            body = app_client.get(
                f"/v1/batches?limit=1{f'&after={cursor}' if cursor else ''}"
            ).json()
            assert cursor not in [item["id"] for item in body["data"]]
            paged.extend(body["data"])
            cursor = paged[-1]["id"]
            assert body["has_more"] is (len(paged) < 3)

        assert [item["id"] for item in paged] == [newest, middle, oldest]
        times = [item["created_at"] for item in paged]
        assert times == [
            payload_created_at(item["id"].removeprefix("batch_")) for item in paged
        ], times
        assert times == sorted(times, reverse=True), times

        single = app_client.get("/v1/batches?limit=100").json()["data"]
        assert [item["id"] for item in single] == [newest, middle, oldest]
        assert [item["created_at"] for item in single] == times

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

    def test_the_newest_batch_is_listed_past_a_hundred_thousand_records(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The listing window is the newest records however many are stored.

        Storage lists keys ascending, so a scan that walks forward from the
        start of the bucket holds the newest records only while the whole
        bucket fits in the budget it walks with: past that the window is taken
        from the *oldest* records and the recent batches a client is listing
        for are the ones missing. More records than any such walk covers are
        seeded here — a year of them, each stored under the key its creation
        instant gives it, as the server stores its own — so the batch created
        last is reached only by a scan that seeks the end of the key space.

        Ref: https://stdapi.ai/api_openai_batches/#listing-order
             stdapi/batches.py:_scan_bucket
        """
        s3, _ = _batches.install(monkeypatch)
        fingerprint = crc32(_batches.BUCKET.encode()).to_bytes(4, "big")
        stored, year_ms = 101_000, 365 * 24 * 3600 * 1000
        # Stored an hour back so the batch created below is the newest record.
        newest_ms = int(time() * 1000) - 3_600_000
        for index in range(stored):
            created_ms = newest_ms - (stored - index) * (year_ms // stored)
            payload = (
                b32hexencode(
                    created_ms.to_bytes(6, "big")
                    + index.to_bytes(10, "big")
                    + fingerprint
                )
                .lower()
                .decode()
            )
            s3.objects[_batches.BUCKET, batches.batch_s3_key(payload)] = b"{}"
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        batch_id = _create(app_client, file_id)["id"]

        body = app_client.get("/v1/batches?limit=10").json()
        assert [item["id"] for item in body["data"]] == [batch_id]

    def test_the_newest_batch_is_listed_when_records_crowd_one_hour(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A scan reaches the newest batch when records crowd the instant it probes.

        The scan seeks the end of the key space by probing an instant and
        moving it, so records packed densely enough that a probe cannot walk
        past them are what makes it move the instant forward instead of back.
        Enough are seeded inside one hour for the first probed instant to leave
        more behind it than one probe walks, which is the path a bucket with a
        steady flow of batches takes on every listing.

        A year of history sits behind the crowded hour so that the seek phase
        (finding an instant complete-probeable at all) and the bisect phase
        (narrowing onto the crowded hour) both have to run: a scan that only
        pages forward from the start of the bucket, as before this feature,
        never reaches the crowded hour and cannot find the newest record
        either, so this seeds what would make that reversion fail too.

        Ref: stdapi/batches.py:_scan_bucket
        """
        s3, _ = _batches.install(monkeypatch)
        fingerprint = crc32(_batches.BUCKET.encode()).to_bytes(4, "big")
        year_stored, year_ms = 101_000, 365 * 24 * 3600 * 1000
        hour_stored, hour_ms = 5_000, 3600 * 1000
        newest_ms = int(time() * 1000) - 1_000
        # The year of history ends an hour before "now", where the crowded
        # hour begins, so the two layers do not overlap in key order.
        year_newest_ms = newest_ms - hour_ms
        for index in range(year_stored):
            created_ms = year_newest_ms - (year_stored - index) * (
                year_ms // year_stored
            )
            payload = (
                b32hexencode(
                    created_ms.to_bytes(6, "big")
                    + index.to_bytes(10, "big")
                    + fingerprint
                )
                .lower()
                .decode()
            )
            s3.objects[_batches.BUCKET, batches.batch_s3_key(payload)] = b"{}"
        for index in range(hour_stored):
            created_ms = newest_ms - (hour_stored - index) * (hour_ms // hour_stored)
            payload = (
                b32hexencode(
                    created_ms.to_bytes(6, "big")
                    + (year_stored + index).to_bytes(10, "big")
                    + fingerprint
                )
                .lower()
                .decode()
            )
            s3.objects[_batches.BUCKET, batches.batch_s3_key(payload)] = b"{}"
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        batch_id = _create(app_client, file_id)["id"]

        body = app_client.get("/v1/batches?limit=10").json()
        assert [item["id"] for item in body["data"]] == [batch_id]

    def test_a_listing_keeps_seeking_while_its_request_budget_is_unspent(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A scan widens its window while probes still complete and budget remains.

        A probe that reaches the end of the listing is already a correct tail,
        so settling for the first one that holds a few hundred records — while
        most of the 20-request budget sits unspent — shrinks the window well
        below the "up to 1,000 most recent batch records" the API documents. A
        marker is placed deep enough into a week of history that only a scan
        spending its idle budget on a wider probe reaches it.

        Ref: https://stdapi.ai/api_openai_batches/#listing-window
             stdapi/batches.py:_scan_bucket
        """
        s3, _ = _batches.install(monkeypatch)
        fingerprint = crc32(_batches.BUCKET.encode()).to_bytes(4, "big")
        stored, week_ms = 5_000, 7 * 24 * 3600 * 1000
        newest_ms = int(time() * 1000) - 1_000
        # Deep enough that a scan settling near the 200-record floor misses it,
        # but shallow enough that a scan spending its full budget reaches it.
        marker_depth = 700
        marker_ms = 0
        for index in range(stored):
            created_ms = newest_ms - (stored - index) * (week_ms // stored)
            if stored - index == marker_depth:
                marker_ms = created_ms
                continue
            payload = (
                b32hexencode(
                    created_ms.to_bytes(6, "big")
                    + index.to_bytes(10, "big")
                    + fingerprint
                )
                .lower()
                .decode()
            )
            s3.objects[_batches.BUCKET, batches.batch_s3_key(payload)] = b"{}"
        assert marker_ms

        # A real batch, created at the marker's instant, so it carries a real
        # job the listing can settle rather than a hand-written record.
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        with monkeypatch.context() as clock:
            clock.setattr(
                "stdapi.files._core.uuid7",
                lambda: UUID(bytes=marker_ms.to_bytes(6, "big") + bytes(10)),
            )
            marker_id = _create(app_client, file_id)["id"]

        body = app_client.get("/v1/batches?limit=10").json()
        assert marker_id in [item["id"] for item in body["data"]]

    def test_a_batchs_own_objects_stay_rolled_up_under_the_scans_budget(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The scan reads batch records, not every object a batch owns.

        A real batch stores its input, its job output and its manifest under
        its own key prefix, so listing without a delimiter would count every
        one of those objects against the same 1,000-key pages the scan reads —
        shrinking the window a batch's own data has nothing to do with. Ten
        objects are seeded per record here, matching a batch's real layout
        closely enough that the newest record is only reached because the
        scan rolls each batch's data up into its own listing entry.

        Ref: stdapi/batches.py:_walk_tail
        """
        s3, _ = _batches.install(monkeypatch)
        fingerprint = crc32(_batches.BUCKET.encode()).to_bytes(4, "big")
        stored, hour_ms = 5_000, 3600 * 1000
        newest_ms = int(time() * 1000) - 1_000
        for index in range(stored):
            created_ms = newest_ms - (stored - index) * (hour_ms // stored)
            payload = (
                b32hexencode(
                    created_ms.to_bytes(6, "big")
                    + index.to_bytes(10, "big")
                    + fingerprint
                )
                .lower()
                .decode()
            )
            key = batches.batch_s3_key(payload)
            s3.objects[_batches.BUCKET, key] = b"{}"
            for sub in range(10):
                s3.objects[_batches.BUCKET, f"{key}/0/file{sub}.jsonl"] = b"{}"
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        batch_id = _create(app_client, file_id)["id"]

        body = app_client.get("/v1/batches?limit=10").json()
        assert [item["id"] for item in body["data"]] == [batch_id]

    def test_the_newest_batch_is_listed_after_a_burst_in_the_real_key_layout(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A nightly burst, quiet since, still lists the batch created after it.

        An ended batch leaves three entries at the record level — the record,
        the `.billed` marker the billing claim writes, and the one common
        prefix its own objects roll up into — so a 1,000-key page carries only
        about 333 records where a bare record layout carries 1,000. A burst
        that fits in a probe's pages under the bare layout therefore does not
        under the real one, and the seek has to cross it by the records each
        probe reads rather than by halving its window: a burst sitting some
        hours in the past is otherwise never bisected into within the budget,
        and the listing answers from its oldest end.

        Ref: https://stdapi.ai/api_openai_batches/#listing-window
             stdapi/batches.py:_scan_bucket
        """
        s3, _ = _batches.install(monkeypatch)
        fingerprint = crc32(_batches.BUCKET.encode()).to_bytes(4, "big")
        stored, hour_ms = 1_500, 3600 * 1000
        # The burst spans three hours and ended 21 hours ago, with nothing
        # since: no probe of the recent past reaches it, and every bisect the
        # budget affords lands before it.
        burst_end_ms = int(time() * 1000) - 21 * hour_ms
        for index in range(stored):
            created_ms = burst_end_ms - (stored - index) * (3 * hour_ms // stored)
            payload = (
                b32hexencode(
                    created_ms.to_bytes(6, "big")
                    + index.to_bytes(10, "big")
                    + fingerprint
                )
                .lower()
                .decode()
            )
            key = batches.batch_s3_key(payload)
            s3.objects[_batches.BUCKET, key] = b"{}"
            s3.objects[_batches.BUCKET, f"{key}.billed"] = b""
            s3.objects[_batches.BUCKET, f"{key}/0/input.jsonl"] = b"{}"

        # The batch to find is the last of the burst, so reaching it is the
        # seek's job rather than a recent probe's.
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        with monkeypatch.context() as clock:
            clock.setattr(
                "stdapi.files._core.uuid7",
                lambda: UUID(bytes=burst_end_ms.to_bytes(6, "big") + bytes(10)),
            )
            batch_id = _create(app_client, file_id)["id"]

        body = app_client.get("/v1/batches?limit=10").json()
        assert batch_id in [item["id"] for item in body["data"]]

    def test_a_batch_created_during_a_density_spike_is_still_listed(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A burst too dense for one probe still leaves the newest batch listed.

        The scan spends at most 20 storage requests, and 5,000 records inside
        one minute is denser than any probe can walk past — so halving the seek
        window alone never lands inside the burst before the budget runs out.
        Each probe that runs out of pages instead carries the seek forward to
        the last record it read, which is what makes a burst crossable at the
        rate the pages read it. Retrieval by identifier, which never depends on
        the scan, is asserted alongside it.

        Ref: https://stdapi.ai/api_openai_batches/#listing-window
             stdapi/batches.py:_scan_bucket
        """
        s3, _ = _batches.install(monkeypatch)
        fingerprint = crc32(_batches.BUCKET.encode()).to_bytes(4, "big")
        stored, minute_ms = 5_000, 60 * 1000
        newest_ms = int(time() * 1000) - 1_000
        for index in range(stored):
            created_ms = newest_ms - (stored - index) * (minute_ms // stored)
            payload = (
                b32hexencode(
                    created_ms.to_bytes(6, "big")
                    + index.to_bytes(10, "big")
                    + fingerprint
                )
                .lower()
                .decode()
            )
            s3.objects[_batches.BUCKET, batches.batch_s3_key(payload)] = b"{}"
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        batch_id = _create(app_client, file_id)["id"]

        assert batch_id in [
            item["id"] for item in app_client.get("/v1/batches?limit=10").json()["data"]
        ]
        assert app_client.get(f"/v1/batches/{batch_id}").status_code == 200

    def test_batches_sharing_a_second_page_in_identifier_order(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two batches created in the same second keep one order across pages.

        `created_at` is published in whole seconds, so batches created inside
        one second report the same value and the identifier is what breaks the
        tie — the stability the API documentation promises a client paging
        with `after`.

        Ref: https://stdapi.ai/api_openai_batches/#listing-order
             stdapi/batches.py:list_batches
        """
        _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        minted = count(1)
        monkeypatch.setattr(
            "stdapi.files._core.uuid7",
            lambda: UUID(
                bytes=(1_700_000_000_123).to_bytes(6, "big")
                + next(minted).to_bytes(10, "big")
            ),
        )
        older, newer = (_create(app_client, file_id)["id"] for _ in range(2))

        body = app_client.get("/v1/batches?limit=10").json()
        assert [item["id"] for item in body["data"]] == [newer, older]
        assert {item["created_at"] for item in body["data"]} == {1_700_000_000}

        page = app_client.get("/v1/batches?limit=1").json()
        assert [item["id"] for item in page["data"]] == [newer]
        page = app_client.get(f"/v1/batches?limit=1&after={newer}").json()
        assert [item["id"] for item in page["data"]] == [older]

    def test_a_cursor_older_than_the_scan_window_pages_nothing(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An `after` cursor naming a batch past the scan window returns an empty page.

        The listing carries the most recent batches only, so a cursor naming
        a batch pushed out of that window pages nothing even though an older
        batch follows it and both are still retrievable by ID — the
        limitation the API documentation states.

        Ref: https://stdapi.ai/api_openai_batches/#listing-order
             stdapi/batches.py:list_batches
        """
        from stdapi.config import SETTINGS  # noqa: PLC0415

        s3, _ = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        older, cursor = (_create(app_client, file_id)["id"] for _ in range(2))
        # Sort after every real payload, so both batches fall out of the window.
        for index in range(1000):
            key = f"{SETTINGS.aws_s3_batches_prefix}z{index:031x}"
            s3.objects[_batches.BUCKET, key] = b"{}"

        for batch_id in (older, cursor):
            assert app_client.get(f"/v1/batches/{batch_id}").status_code == 200
        body = app_client.get(f"/v1/batches?limit=10&after={cursor}").json()
        assert body["data"] == []
        assert body["has_more"] is False

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

    def _ended_batch(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> tuple[_batches.FakeS3, _batches.FakeBedrock, str]:
        """Run a batch of 100 answers to its end without ever retrieving it."""
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
        return s3, bedrock, batch_id

    def test_listing_settles_and_publishes_an_ended_batch(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A batch first seen in the listing is settled and published there.

        A client that polls the listing never retrieves a batch on its own, so
        a listing reporting `completed` with no `output_file_id` would leave it
        with nowhere to read its results — and its usage unrecorded until
        somebody happened to retrieve it, which is a batch nobody is billed for.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/batches.py:finish_listed
        """
        from tests.conftest import logged_usage_entries  # noqa: PLC0415

        s3, _, batch_id = self._ended_batch(app_client, monkeypatch)
        capfd.readouterr()

        (listed,) = app_client.get("/v1/batches?limit=10").json()["data"]

        assert listed["id"] == batch_id
        assert listed["status"] == "completed"
        assert listed["usage"]["input_tokens"] == 500
        assert listed["usage"]["output_tokens"] == 300
        assert len(_batches.read_result_file(s3, listed["output_file_id"])) == 100
        (entry,) = logged_usage_entries(
            capfd.readouterr().out, service="bedrock-runtime"
        )
        assert entry["tier"] == "batch"
        assert entry["input_tokens"] == 500

    def test_listing_settles_only_what_changed(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A listing costs nothing for a batch that is running or already settled.

        Settling is what makes a listing expensive — every job's counters, then
        the whole results file — so a page of batches must not pay for it on
        every poll, only for the ones that ended since the last read.

        Ref: stdapi/batches.py:finish_listed
        """
        from tests.conftest import logged_usage_entries  # noqa: PLC0415

        s3, _, _ = self._ended_batch(app_client, monkeypatch)
        _create(app_client, _batches.install_input_file(monkeypatch, chat_lines(100)))
        assert len(app_client.get("/v1/batches?limit=10").json()["data"]) == 2
        capfd.readouterr()
        read: list[str] = []
        written: list[str] = []
        get_object, put_object = s3.get_object, s3.put_object

        async def _read(*, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
            read.append(Key)
            return await get_object(Bucket=Bucket, Key=Key)

        async def _write(*, Key: str, **kwargs: Any) -> dict[str, Any]:  # noqa: N803, ANN401
            written.append(Key)
            return await put_object(Key=Key, **kwargs)

        monkeypatch.setattr(s3, "get_object", _read)
        monkeypatch.setattr(s3, "put_object", _write)

        body = app_client.get("/v1/batches?limit=10").json()

        assert len(body["data"]) == 2
        # The records themselves, and not one job counter or results file.
        assert not [key for key in read if key.endswith(".out")]
        assert not written
        assert not logged_usage_entries(
            capfd.readouterr().out, service="bedrock-runtime"
        )

    def test_a_batch_that_cannot_be_settled_still_lists(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """One batch that cannot be settled does not take the whole page down.

        Nothing is claimed by a settlement that failed, so the batch is listed
        as it stands and the next read settles it; failing the listing instead
        would leave the client unable to see any of its batches.

        Ref: stdapi/batches.py:finish_listed
        """
        s3, _, batch_id = self._ended_batch(app_client, monkeypatch)
        get_object = s3.get_object

        async def _read(*, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
            if Key.endswith("manifest.json.out"):
                raise ClientError(
                    {"Error": {"Code": "InternalError", "Message": "storage failed"}},
                    "GetObject",
                )
            return await get_object(Bucket=Bucket, Key=Key)

        monkeypatch.setattr(s3, "get_object", _read)
        capfd.readouterr()

        response = app_client.get("/v1/batches?limit=10")

        assert response.status_code == 200
        (listed,) = response.json()["data"]
        assert listed["id"] == batch_id
        # Nothing was published, so the batch has not completed: it is finalizing.
        assert listed["status"] == "finalizing"
        assert "output_file_id" not in listed
        assert "usage" not in listed
        assert "could not be settled" in capfd.readouterr().out

    def test_an_ended_batch_is_finalizing_until_its_results_are_published(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A batch whose requests are done but whose results are not is `finalizing`.

        Upstream reports `finalizing` while the results of a batch whose work
        has finished are being assembled, and `completed` only once they are
        readable. The same window exists here — the result files are written
        after the last job ends — and reporting `completed` through it tells a
        client to download an `output_file_id` the batch does not have yet.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/routes/openai_batches.py:_status
        """
        from stdapi.config import SETTINGS  # noqa: PLC0415

        s3, _, batch_id = self._ended_batch(app_client, monkeypatch)
        put_object = s3.put_object
        ended_at = int(datetime(2026, 8, 12, 1, tzinfo=UTC).timestamp())

        async def _write(*, Key: str, **kwargs: Any) -> dict[str, Any]:  # noqa: N803, ANN401
            if Key.startswith(SETTINGS.aws_s3_files_prefix):
                raise ClientError(
                    {"Error": {"Code": "InternalError", "Message": "storage failed"}},
                    "PutObject",
                )
            return await put_object(Key=Key, **kwargs)

        monkeypatch.setattr(s3, "put_object", _write)

        (listed,) = app_client.get("/v1/batches?limit=10").json()["data"]

        assert listed["id"] == batch_id
        assert listed["status"] == "finalizing"
        assert listed["finalizing_at"] == ended_at
        assert "completed_at" not in listed
        assert "output_file_id" not in listed
        # Settled, and still not readable: finalizing is about the results.
        assert listed["usage"]["output_tokens"] == 300

        monkeypatch.setattr(s3, "put_object", put_object)

        published = app_client.get(f"/v1/batches/{batch_id}").json()

        assert published["status"] == "completed"
        assert published["finalizing_at"] == ended_at
        assert published["completed_at"] == ended_at
        assert len(_batches.read_result_file(s3, published["output_file_id"])) == 100

    def test_a_batch_that_cannot_be_stored_stops_its_jobs(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A batch whose record cannot be written leaves no job running.

        The record is the only thing that names the jobs: without it nothing
        can find, cancel or settle them, and they would run for their whole
        window and be billed with no trace an operator could act on.

        Ref: stdapi/batches.py:create_batch
        """
        s3, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))

        async def _unwritable(_record: batches.BatchRecord) -> None:
            raise ClientError(
                {"Error": {"Code": "InternalError", "Message": "storage failed"}},
                "PutObject",
            )

        monkeypatch.setattr(batches, "_write_record", _unwritable)
        capfd.readouterr()

        body = _create(app_client, file_id)

        assert body["http_status"] >= 500
        assert bedrock.stopped == list(bedrock.jobs)
        assert not [key for (_b, key) in s3.objects if key.endswith(".jsonl")]

    def test_a_job_that_cannot_be_stopped_is_named_to_the_operator(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """A job left running by a failed creation is reported, since nothing else can.

        Ref: stdapi/batches.py:_abandon_jobs
        """
        _, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))

        async def _unwritable(_record: batches.BatchRecord) -> None:
            raise ClientError(
                {"Error": {"Code": "InternalError", "Message": "storage failed"}},
                "PutObject",
            )

        monkeypatch.setattr(batches, "_write_record", _unwritable)
        bedrock.stop_error = "AccessDeniedException"
        capfd.readouterr()

        assert _create(app_client, file_id)["http_status"] >= 500

        logged = capfd.readouterr().out
        assert "could not be stopped" in logged
        assert "bills until its window ends" in logged

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
        choices, usage = body["choices"], body["usage"]
        assert isinstance(choices, list)
        assert isinstance(usage, dict)
        choice = choices[0]
        assert isinstance(choice, dict)
        message = choice["message"]
        assert isinstance(message, dict)
        assert message["content"] == "hello"
        assert choice["finish_reason"] == "stop"
        assert usage["prompt_tokens"] == 5

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


@pytest.mark.local
class TestOpenAIEmbeddingsBatch:
    """A batch of ``/v1/embeddings`` requests, from submission to the vectors.

    An embeddings batch runs each request as the model's own invocation rather
    than as a conversation, and the two shapes are not interchangeable: a job
    started for the wrong one is accepted, runs its whole 24-hour window and
    fails every record. The request bodies and the answers read back are the
    ones the embeddings route itself builds and parses.

    Ref: https://developers.openai.com/api/docs/guides/batch.md
         tests/probes/results/bedrock.batch-embeddings.json
         stdapi/batches.py:_prepare_embedding_request
    """

    @staticmethod
    def _create(
        client: TestClient, file_id: str, endpoint: str = "/v1/embeddings"
    ) -> dict[str, Any]:
        """Submit a batch of *file_id* against *endpoint*."""
        response = client.post(
            "/v1/batches",
            json={
                "input_file_id": file_id,
                "endpoint": endpoint,
                "completion_window": "24h",
            },
        )
        return {"http_status": response.status_code, **response.json()}

    def test_the_job_runs_the_models_own_invocation(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An embeddings job is started for the invocation the model answers.

        A conversation-typed job is accepted by the backend and then fails
        every record, so this is the assertion that keeps a whole billed
        window from producing nothing.

        Ref: tests/probes/results/bedrock.batch-embeddings.json
             stdapi/batches.py:_invocation_type
        """
        _, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(
            monkeypatch, _batches.embedding_lines(100)
        )
        assert self._create(app_client, file_id)["http_status"] == 200
        assert bedrock.created[0]["modelInvocationType"] == "InvokeModel"

    def test_a_chat_batch_still_runs_a_conversation(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The chat endpoint keeps the invocation it has always been run with."""
        _, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        assert (
            self._create(app_client, file_id, "/v1/chat/completions")["http_status"]
            == 200
        )
        assert bedrock.created[0]["modelInvocationType"] == "Converse"

    def test_each_record_carries_the_models_own_request_body(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A record holds the body the model reads, keyed by its ``custom_id``.

        Ref: tests/probes/results/bedrock.batch-embeddings.json
        """
        s3, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(
            monkeypatch, _batches.embedding_lines(100)
        )
        assert self._create(app_client, file_id)["http_status"] == 200
        uri = bedrock.created[0]["inputDataConfig"]["s3InputDataConfig"]["s3Uri"]
        written = s3.objects[_batches.BUCKET, uri.split("/", 3)[3]].splitlines()
        first = loads(written[0])
        assert first == {"recordId": "req-0", "modelInput": {"inputText": "text 0"}}
        assert len(written) == 100

    def test_the_results_carry_the_vectors(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A finished batch's results are the embeddings response of each request.

        Ref: https://developers.openai.com/api/docs/api-reference/embeddings/object
             stdapi/batches.py:_to_embeddings
        """
        s3, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(
            monkeypatch, _batches.embedding_lines(100)
        )
        created = self._create(app_client, file_id)
        assert created["http_status"] == 200
        bedrock.finish()
        _batches.write_job_output(
            s3,
            bedrock,
            [{"recordId": "req-0", "modelOutput": _batches.embedding_output(tokens=7)}],
            {"inputTokenCount": 7, "outputTokenCount": 0},
        )
        batch = app_client.get(f"/v1/batches/{created['id']}").json()
        assert batch["status"] == "completed"
        (line,) = _batches.read_result_file(s3, batch["output_file_id"])
        assert line["custom_id"] == "req-0"
        body = line["response"]["body"]
        assert body["object"] == "list"
        assert body["model"] == "amazon.titan-embed-text-v2:0"
        assert body["data"][0]["embedding"] == [0.1, 0.2, 0.3, 0.4]
        assert body["data"][0]["index"] == 0
        assert body["usage"]["prompt_tokens"] == 7

    def test_more_than_one_input_per_request_is_refused(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request embedding several inputs is refused, naming the way forward.

        One record is one invocation of the model, and the models that batch
        embed one input per invocation.

        Ref: stdapi/models/embedding/__init__.py:EmbeddingModelBase
        """
        _batches.install(monkeypatch)
        lines = _batches.embedding_lines(100)
        lines[2] = _batches.embedding_lines(
            1,
            prefix="multi",
            body='{"model": "amazon.titan-embed-text-v2:0", "input": ["a", "b"]}',
        )[0]
        file_id = _batches.install_input_file(monkeypatch, lines)
        body = self._create(app_client, file_id)
        assert body["http_status"] == 400
        assert body["error"]["message"].startswith("Line 3: ")
        assert "one input per batched request" in body["error"]["message"]

    def test_base64_vectors_are_refused(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request asking for base64 vectors is refused, naming its position.

        The results are written without the request that produced them, so a
        vector cannot be encoded the way one line asked for; refusing at submit
        beats returning numbers to a client that cannot read them.

        Ref: https://developers.openai.com/api/docs/api-reference/embeddings/create
             stdapi/batches.py:_prepare_embedding_request
        """
        _batches.install(monkeypatch)
        lines = _batches.embedding_lines(100)
        lines[1] = _batches.embedding_lines(
            1,
            prefix="b64",
            body=(
                '{"model": "amazon.titan-embed-text-v2:0", "input": "a", '
                '"encoding_format": "base64"}'
            ),
        )[0]
        file_id = _batches.install_input_file(monkeypatch, lines)
        body = self._create(app_client, file_id)
        assert body["http_status"] == 400
        assert "Line 2: 'encoding_format'" in body["error"]["message"]

    def test_the_media_a_job_consumed_is_recorded(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """Images and durations reported by a job are billed, not only tokens.

        A multimodal embedding model is priced per image and per second of
        media, so a batch of them recording tokens alone reports no cost at all.

        Ref: tests/probes/results/bedrock.batch-embeddings.json
             stdapi/batches.py:_record_media_usage
        """
        from tests.conftest import logged_usage_entries  # noqa: PLC0415

        s3, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(
            monkeypatch, _batches.embedding_lines(100)
        )
        created = self._create(app_client, file_id)
        assert created["http_status"] == 200
        bedrock.finish()
        _batches.write_job_output(
            s3,
            bedrock,
            [{"recordId": "req-0", "modelOutput": _batches.embedding_output()}],
            {
                "inputTokenCount": 12,
                "outputTokenCount": 0,
                "inputStandardImageCount": 3,
                "inputAudioSecond": 5,
            },
        )
        capfd.readouterr()
        assert app_client.get(f"/v1/batches/{created['id']}").status_code == 200
        entries = logged_usage_entries(
            capfd.readouterr().out, service="bedrock-runtime"
        )
        assert all(entry["tier"] == "batch" for entry in entries)
        assert sum(entry.get("input_images", 0) for entry in entries) == 3
        assert sum(entry.get("input_seconds", 0) for entry in entries) == 5

    def test_a_chat_job_bills_its_media_as_tokens_only(
        self,
        app_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """The same counters on a conversation are not billed a second time.

        A conversation's images are already inside its input tokens, so
        recording them again would bill every batched image twice.

        Ref: stdapi/batches.py:settle
        """
        from tests.conftest import logged_usage_entries  # noqa: PLC0415

        s3, bedrock = _batches.install(monkeypatch)
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        batch_id = _create(app_client, file_id)["id"]
        bedrock.finish()
        _batches.write_job_output(
            s3,
            bedrock,
            [{"recordId": "req-0", "modelOutput": converse_output("answer")}],
            {
                "inputTokenCount": 12,
                "outputTokenCount": 4,
                "inputStandardImageCount": 3,
                "inputAudioSecond": 5,
            },
        )
        capfd.readouterr()
        assert app_client.get(f"/v1/batches/{batch_id}").status_code == 200
        entries = logged_usage_entries(
            capfd.readouterr().out, service="bedrock-runtime"
        )
        assert sum(entry.get("input_images", 0) for entry in entries) == 0
        assert sum(entry.get("input_seconds", 0) for entry in entries) == 0
        assert sum(entry.get("input_tokens", 0) for entry in entries) == 12


@pytest.mark.local
class TestBatchModelRouting:
    """A batch names a model; the job runs the identifier that can batch it.

    Batches run on one backend only, so a model this server normally reaches
    through another one is submitted under the identifier the batching backend
    knows it by. A model that backend knows under no name keeps a refusal of
    its own, which says something different to the caller.

    Ref: audit/milestone-1-16/probe-150-mantle-batch.md
         stdapi/batches.py:_batch_model_id
    """

    @staticmethod
    def _install_mantle(
        monkeypatch: pytest.MonkeyPatch, twin: str | None
    ) -> _batches.FakeBedrock:
        """Make every model Mantle-served, resolving to *twin* when there is one."""
        _, bedrock = _batches.install(monkeypatch)
        monkeypatch.setattr(batches, "serves_via_mantle", lambda _model_id: True)
        monkeypatch.setattr(batches, "runtime_twin", lambda _model_id: twin)
        return bedrock

    def test_the_runtime_form_of_the_model_is_submitted(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model served elsewhere is batched under the identifier that batches."""
        bedrock = self._install_mantle(monkeypatch, "openai.gpt-oss-120b-1:0")
        file_id = _batches.install_input_file(
            monkeypatch, chat_lines(100, model="openai.gpt-oss-120b")
        )
        body = _create(app_client, file_id)
        assert body["http_status"] == 200
        assert bedrock.created[0]["modelId"] == "openai.gpt-oss-120b-1:0"

    def test_a_model_with_no_runtime_form_is_refused(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model no backend can batch is refused before anything is written.

        The message differs from the routing case on purpose: this one is a
        capability the model does not have, not an identifier to translate.
        """
        bedrock = self._install_mantle(monkeypatch, None)
        file_id = _batches.install_input_file(
            monkeypatch, chat_lines(100, model="xai.grok-4.3")
        )
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        assert "cannot run batched requests" in body["error"]["message"]
        assert not bedrock.created

    def test_a_refusal_naming_no_model_still_answers_the_caller(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refusal on model grounds is a 400, even when it names no model.

        Both wordings the backend answers a model it will not batch with name
        no identifier at all, so matching on the identifier alone reported the
        deployment as broken instead of the model as unsupported.

        Ref: audit/milestone-1-16/probe-150-mantle-batch.md
             stdapi/batches.py:_refused_job
        """
        _, bedrock = _batches.install(monkeypatch)
        bedrock.reject_models = {"amazon.nova-micro-v1:0"}
        bedrock.reject_message = (
            "Batch inference is not supported for the requested model"
        )
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        body = _create(app_client, file_id)
        assert body["http_status"] == 400
        assert "amazon.nova-micro-v1:0" in body["error"]["message"]

    def test_a_deployment_failure_is_still_told_apart(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refusal about the deployment is not reported as an unusable model."""
        _, bedrock = _batches.install(monkeypatch)
        bedrock.reject_models = {"amazon.nova-micro-v1:0"}
        bedrock.reject_message = "The specified bucket does not exist"
        file_id = _batches.install_input_file(monkeypatch, chat_lines(100))
        body = _create(app_client, file_id)
        assert body["http_status"] == 503


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


#: Statuses a batch has stopped changing at.
_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"completed", "failed", "expired", "cancelled"}
)

#: How long a batch of the minimum size is given to reach one of them.
_ROUND_TRIP_TIMEOUT: float = 3600.0

#: Seconds between two status reads of a running batch.
_POLL_INTERVAL: float = 20.0

#: Model served through the endpoint that runs no batch, batched by its twin.
_ELSEWHERE_SERVED_MODEL: str = "openai.gpt-oss-20b"

#: Name the endpoint that does run batches knows that same model by.
_ELSEWHERE_SERVED_MODEL_BATCHED: str = "openai.gpt-oss-20b-1:0"


def _cosine(left: list[float], right: list[float]) -> float:
    """Return the cosine similarity of two vectors of the same length."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (sqrt(sum(a * a for a in left)) * sqrt(sum(b * b for b in right)))


# A real batch is given _ROUND_TRIP_TIMEOUT to end, well past the suite default.
@pytest.mark.timeout(4200)
@pytest.mark.slow
@pytest.mark.usefixtures("batches_api")
class TestOpenAIBatchRoundTrip:
    """Running real batches to completion, and reading the answers back.

    A submission that is accepted says nothing about a batch: the requests
    still have to run, and the results still have to read back as the
    responses of the endpoint they were written for. Each test here submits
    the smallest batch the backing jobs accept, waits for it to end, and reads
    every line of the output file.

    One test per endpoint the surface serves, on a different model family
    each, so a translation bug belonging to one family cannot pass unseen.

    Ref: https://developers.openai.com/api/docs/guides/batch.md
         stdapi/batches.py:iter_openai_results
    """

    @staticmethod
    def _wait(client: OpenAI, batch_id: str) -> Batch:
        """Read *batch_id* until it stops changing, and return it.

        Args:
            client: SDK client bound to the target.
            batch_id: Identifier of the batch to wait for.

        Returns:
            The batch, in whichever terminal status it reached.
        """
        deadline = monotonic() + _ROUND_TRIP_TIMEOUT
        while (
            batch := client.batches.retrieve(batch_id)
        ).status not in _TERMINAL_STATUSES:
            assert monotonic() < deadline, (
                f"{batch_id} was still '{batch.status}' after "
                f"{_ROUND_TRIP_TIMEOUT:.0f}s ({batch.request_counts})"
            )
            sleep(_POLL_INTERVAL)
        return batch

    def _round_trip(
        self,
        client: OpenAI,
        endpoint: Literal["/v1/chat/completions", "/v1/embeddings"],
        bodies: list[dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Run *bodies* through a batch and read the results back.

        Args:
            client: SDK client bound to the target.
            endpoint: Endpoint every request of the batch targets.
            bodies: One request body per line, in input order.

        Returns:
            The result lines, indexed by the ``custom_id`` that produced them.
        """
        payload = "\n".join(
            dumps(
                {
                    "custom_id": f"req-{index}",
                    "method": "POST",
                    "url": endpoint,
                    "body": body,
                }
            )
            for index, body in enumerate(bodies)
        ).encode()
        input_file = client.files.create(
            file=("requests.jsonl", payload), purpose="batch"
        )
        batch = None
        try:
            batch = client.batches.create(
                input_file_id=input_file.id, endpoint=endpoint, completion_window="24h"
            )
            batch = self._wait(client, batch.id)
            assert batch.status == "completed", batch.errors
            assert batch.request_counts is not None
            assert batch.request_counts.total == len(bodies)
            assert batch.request_counts.completed == len(bodies)
            assert batch.request_counts.failed == 0
            assert batch.output_file_id is not None
            lines = [
                loads(line)
                for line in client.files.content(batch.output_file_id).text.splitlines()
                if line
            ]
            assert len(lines) == len(bodies)
            results = {line["custom_id"]: line for line in lines}
            assert set(results) == {f"req-{index}" for index in range(len(bodies))}
            return results
        finally:
            if batch is not None and batch.status not in _TERMINAL_STATUSES:
                with contextlib.suppress(Exception):
                    client.batches.cancel(batch.id)
            with contextlib.suppress(Exception):
                client.files.delete(input_file.id)

    @staticmethod
    def _answer(line: dict[str, Any]) -> dict[str, Any]:
        """Return the response body of one result line, asserting it succeeded."""
        assert line["error"] is None, line
        assert line["response"]["status_code"] == 200, line
        assert line["response"]["request_id"]
        body: dict[str, Any] = line["response"]["body"]
        return body

    def test_a_chat_batch_answers_every_request(
        self, openai_client: OpenAI, chat_model: str, use_official_api: bool
    ) -> None:
        """Every batched chat request comes back as its own completion.

        The whole round trip is the assertion: a batch that is merely accepted
        can still fail every one of its requests, so the answers are read from
        the output file and each one checked to be a usable completion.

        Ref: https://developers.openai.com/api/docs/api-reference/chat/object
             stdapi/batches.py:_to_completion
        """
        count = 1 if use_official_api else batches.MIN_REQUESTS_PER_MODEL
        results = self._round_trip(
            openai_client,
            "/v1/chat/completions",
            [
                {
                    "model": chat_model,
                    "messages": [{"role": "user", "content": f"Say hello, #{index}."}],
                }
                for index in range(count)
            ],
        )
        for line in results.values():
            body = self._answer(line)
            assert body["object"] == "chat.completion"
            assert body["choices"][0]["message"]["role"] == "assistant"
            assert body["choices"][0]["message"]["content"]
            assert body["choices"][0]["finish_reason"] == "stop"
            assert body["usage"]["prompt_tokens"] > 0
            assert body["usage"]["completion_tokens"] > 0

    def test_an_embeddings_batch_answers_with_the_vectors(
        self, openai_client: OpenAI, embedding_model: str, use_official_api: bool
    ) -> None:
        """Every batched embeddings request comes back as its own vector.

        The vectors are the proof that the job ran as the model's own
        invocation rather than as a conversation, which the backend accepts
        either way and only one of which produces embeddings. Two of them are
        compared against the same text embedded synchronously: results come
        back in no particular order, so a vector reaching the wrong
        ``custom_id`` is the failure this catches.

        Ref: https://developers.openai.com/api/docs/api-reference/embeddings/object
             stdapi/batches.py:_to_embeddings
        """
        count = 1 if use_official_api else batches.MIN_REQUESTS_PER_MODEL
        inputs = [f"A passage to embed, number {index}." for index in range(count)]
        results = self._round_trip(
            openai_client,
            "/v1/embeddings",
            [{"model": embedding_model, "input": text} for text in inputs],
        )
        dimensions = set()
        for line in results.values():
            body = self._answer(line)
            assert body["object"] == "list"
            (item,) = body["data"]
            assert item["object"] == "embedding"
            assert item["index"] == 0
            assert all(isinstance(value, float) for value in item["embedding"])
            assert body["usage"]["prompt_tokens"] > 0
            dimensions.add(len(item["embedding"]))
        assert len(dimensions) == 1
        assert dimensions.pop() > 1
        for index in range(min(2, count)):
            synchronous = openai_client.embeddings.create(
                model=embedding_model, input=inputs[index]
            )
            batched = results[f"req-{index}"]["response"]["body"]["data"][0]
            assert _cosine(batched["embedding"], synchronous.data[0].embedding) > 0.99

    @pytest.mark.gateway("no vendor API serves a model of this deployment's own")
    def test_a_model_served_elsewhere_batches_under_its_other_name(
        self, openai_client: OpenAI
    ) -> None:
        """A model normally served by an endpoint that runs no batch still batches.

        Its requests are submitted under the identifier the endpoint that does
        run batches knows it by, which is also the model each answer reports.
        Before that pairing existed the batch was refused outright, so this is
        the whole of the path's coverage.

        Ref: https://stdapi.ai/api_openai_batches/#models
             stdapi/batches.py:_batch_model_id
        """
        results = self._round_trip(
            openai_client,
            "/v1/chat/completions",
            [
                {
                    "model": _ELSEWHERE_SERVED_MODEL,
                    "messages": [{"role": "user", "content": f"Say hello, #{index}."}],
                }
                for index in range(batches.MIN_REQUESTS_PER_MODEL)
            ],
        )
        for line in results.values():
            body = self._answer(line)
            assert body["model"] == _ELSEWHERE_SERVED_MODEL_BATCHED
            assert body["choices"][0]["message"]["content"]
            assert body["usage"]["completion_tokens"] > 0
