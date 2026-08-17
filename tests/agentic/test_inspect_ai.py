"""inspect-ai running real evaluations through both of the gateway's Batch APIs.

``inspect-ai`` is the UK AI Safety Institute's evaluation framework, and the only
client in this lane that reaches a batch surface at all -- let alone both of
them. An evaluation is exactly the workload batching exists for: a few hundred
independent prompts, no interactivity, and a result read later at half the price.
The framework knows it, and turns a whole dataset into one provider batch when
its ``batch`` option is set.

It is here because the Batch API is the one surface no agent, no workflow runner
and no chat library in this lane will ever touch, and because it reaches it
twice: ``/v1/files`` + ``/v1/batches`` through the OpenAI provider, and
``/v1/messages/batches`` through the Anthropic one, from the same dataset, the
same committed script and the same eval loop. Only the provider prefix on the
model name differs.

Four things shape the tests below:

- **the assertion is the gateway's request log, not the eval result.** A batching
  layer that quietly fell back to one request per sample produces an identical
  eval record -- same answers, same usage, same status -- so the only thing that
  distinguishes a batch from a hundred synchronous calls is the traffic. Each
  test therefore requires the submission requests it expects to see, and requires
  the synchronous route to have stayed empty.
- **it runs in a container, not in the test process.** Unlike every other Python
  client here it cannot share the lane's overlay: it depends on ``aioboto3``,
  whose every release pins ``aiobotocore`` to a 2.x version, which pins
  ``botocore`` below this project's floor -- and the gateway under test runs on
  the interpreter that overlay is layered onto. Its own image group has no such
  neighbour (see ``Containerfile.inspect``).
- **the batch's minimum size is the client's default.** ``BatchConfig`` holds
  requests back until it has ``size`` of them and defaults to exactly 100, which
  is also this backend's minimum per model, so a hundred-sample dataset submits
  as a single batch.
- **the client owns the poll loop and imposes no deadline of its own.** It waits
  for the batch as long as the provider takes, so the bound is the caller's --
  the same hour the unit suite gives a batch of the minimum size.

Both tests are ``slow`` as well as ``agentic``: they start a real batch job and
wait for it, which is tens of minutes of wall clock, and there is no
submit-then-cancel shortcut when the client is the one waiting.

Requires --agentic --slow, podman, Bedrock credentials, and a deployment
configured for batch inference; the module skips when the gateway does not serve
the Batch API.

Ref: https://inspect.aisi.org.uk/models.html
     https://inspect.aisi.org.uk/parallelism.html
     stdapi/batches.py:MIN_REQUESTS_PER_MODEL
     stdapi/routes/openai_batches.py:create_batch
     stdapi/routes/anthropic_messages_batches.py:create_message_batch
     tests/agentic/inspect_eval.py
"""

from __future__ import annotations

from json import dumps
from typing import TYPE_CHECKING

import httpx
import pytest

from stdapi.batches import MIN_REQUESTS_PER_MODEL

from ._runner import ModelConfig, assert_result, log_metrics, run_agent
from ._tools import INSPECT_AI, AgenticTool, inspect_record

if TYPE_CHECKING:
    from pathlib import Path

    from ._server import AgenticServer

pytestmark = [
    pytest.mark.agentic,
    pytest.mark.slow,
    # Two batches of the minimum size are two real jobs on one account; running
    # them on one worker keeps a parallel session from doubling that at once.
    pytest.mark.xdist_group("inspect_ai_batches"),
]

TOOL: AgenticTool = INSPECT_AI

#: Requests per batch, and therefore samples per dataset.
#:
#: The backend refuses a batch carrying fewer than this for any model it names,
#: and the client holds a batch back until it has that many requests, so the
#: dataset size, the client's batch size and the backend's minimum are one
#: number. A dataset smaller than this would wait out the client's send delay and
#: then be refused.
_BATCH_SIZE = MIN_REQUESTS_PER_MODEL

#: Reference each request asks to have repeated, handed to the run as its prompt.
_MARKER = "ORCHID"

#: Seconds one evaluation may take, batch wait included.
#:
#: The client polls for as long as the provider takes and enforces no deadline of
#: its own, so this is the only bound. It matches the hour the unit suite gives a
#: batch of the minimum size.
_RUN_TIMEOUT = 3600

#: Share of answers that must carry the reference of the request that asked for it.
#:
#: Attribution is what this measures: results come back keyed by an identifier the
#: client generated, so a mapping that lost or shifted one drops this to near
#: zero, while a model that occasionally answers in its own words costs a few
#: points. The floor sits far above what a broken mapping can reach and below what
#: a working round trip produces.
_MIN_ECHOED_SHARE = 0.9

#: Seconds a Batch API availability probe may take.
_PROBE_TIMEOUT = 30.0

#: Status the gateway answers with when this deployment cannot run batches.
_FEATURE_UNAVAILABLE = 503

#: Cheap Bedrock-native model batched through the OpenAI surface.
#:
#: ``responses_api`` is switched off because the framework batches a model name it
#: does not recognise through ``/v1/responses``, which this gateway's Batch API
#: does not serve; the client's own option is how a caller selects the endpoint it
#: does serve.
_OPENAI_MODEL_CONFIG = pytest.param(
    ModelConfig(
        model="amazon.nova-micro-v1:0",
        timeout=_RUN_TIMEOUT,
        extra_env={
            "INSPECT_PROVIDER": "openai",
            "INSPECT_ROUTE": "/v1",
            "BATCH_SIZE": str(_BATCH_SIZE),
            "MODEL_ARGS": dumps({"responses_api": False}),
        },
    ),
    id="nova-micro",
)

#: Cheapest Claude, batched through the Anthropic surface whose own API this is.
_ANTHROPIC_MODEL_CONFIG = pytest.param(
    ModelConfig(
        model="anthropic.claude-haiku-4-5-20251001-v1:0",
        timeout=_RUN_TIMEOUT,
        extra_env={
            "INSPECT_PROVIDER": "anthropic",
            "INSPECT_ROUTE": "/anthropic",
            "BATCH_SIZE": str(_BATCH_SIZE),
        },
    ),
    id="claude-haiku-4-5",
)


@pytest.fixture(scope="module")
def batch_api(agentic_server: AgenticServer) -> None:
    """Skip the module unless this deployment serves both Batch APIs.

    Batch inference needs a role and a bucket the gateway is configured with, and
    the agentic server inherits the environment of the test process -- so a
    machine without them answers 503 rather than failing in a way that names the
    missing setting. Probing both surfaces here reports that as a skip once,
    instead of as two unexplained client errors an hour apart.

    Raises:
        httpx.HTTPStatusError: If either surface fails for any other reason.
    """
    key = agentic_server.api_key
    probes = {
        "/v1/batches?limit=1": {"Authorization": f"Bearer {key}"},
        "/anthropic/v1/messages/batches?limit=1": {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
    }
    for path, headers in probes.items():
        response = httpx.get(
            agentic_server.url(path), headers=headers, timeout=_PROBE_TIMEOUT
        )
        if response.status_code == _FEATURE_UNAVAILABLE:
            pytest.skip(f"this deployment does not serve the Batch API at {path}")
        response.raise_for_status()


def _requests(
    server: AgenticServer, log_start: int, path: str
) -> list[dict[str, object]]:
    """Return the gateway's logged POST requests to one route.

    Args:
        server: Gateway the evaluation was pointed at.
        log_start: Log index captured before the evaluation.
        path: Route path to select.

    Returns:
        The matching entries, oldest first; empty for an external server, whose
        log is not observable here.

    Ref: stdapi/monitoring.py:EventLog
    """
    if server.process is None:
        return []  # External server: its log is not observable here.
    return [
        entry
        for entry in server.log_entries(log_start)
        if entry.get("type") == "request"
        and entry.get("path") == path
        and entry.get("method") == "POST"
    ]


def _assert_every_sample_answered(record: dict[str, object], model: str) -> None:
    """Assert the batch answered every request, with the model it was given.

    Args:
        record: Record the evaluation wrote to its working directory.
        model: Bedrock model every answer must have been produced by.
    """
    assert record.get("status") == "success", record.get("error")
    samples = record.get("samples")
    assert isinstance(samples, list)
    assert len(samples) == _BATCH_SIZE, (
        f"{len(samples)} of {_BATCH_SIZE} samples came back"
    )
    answers = {int(sample["id"]): str(sample["completion"]) for sample in samples}
    empty = sorted(index for index, text in answers.items() if not text.strip())
    assert not empty, f"{len(empty)} answers were empty: {empty[:5]}"
    models = {str(sample["model"]) for sample in samples}
    assert models == {model}, f"the batch was served by {sorted(models)}"
    echoed = sum(f"{_MARKER}-{index:03d}" in text for index, text in answers.items())
    assert echoed >= _MIN_ECHOED_SHARE * _BATCH_SIZE, (
        f"only {echoed} of {_BATCH_SIZE} answers carry the reference of the "
        "request that asked for it, so the results were mapped back to the "
        "wrong requests"
    )


def _evaluate(
    *,
    agentic_server: AgenticServer,
    agentic_image: str,
    agentic_workdir: Path,
    model_config: ModelConfig,
    test_name: str,
) -> dict[str, object]:
    """Run one batched evaluation and return the record it wrote.

    Args:
        agentic_server: Gateway serving the batch.
        agentic_image: Image holding the framework.
        agentic_workdir: Per-test writable directory the record lands in.
        model_config: Model and surface under test.
        test_name: Logical test name, for the run's session identifier.

    Returns:
        The decoded record.
    """
    result = run_agent(
        tool=TOOL,
        server=agentic_server,
        image=agentic_image,
        config=model_config,
        prompt=_MARKER,
        workdir=agentic_workdir,
        test_name=test_name,
    )
    log_metrics(TOOL, result, model_config, test_name)
    assert_result(result, config=model_config, min_steps=_BATCH_SIZE, contains=_MARKER)
    return inspect_record(agentic_workdir)


@pytest.mark.usefixtures("batch_api")
@pytest.mark.parametrize("model_config", [_OPENAI_MODEL_CONFIG])
class TestOpenAIBatch:
    """A whole evaluation submitted through ``/v1/files`` and ``/v1/batches``.

    Ref: https://inspect.aisi.org.uk/models.html
         https://developers.openai.com/api/docs/guides/batch
         stdapi/routes/openai_batches.py:create_batch
    """

    def test_the_dataset_is_uploaded_as_a_file_and_run_as_one_batch(
        self,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
        request: pytest.FixtureRequest,
    ) -> None:
        """The eval completes, and the traffic that carried it is a batch.

        Both halves are required. The eval record alone cannot tell a batch from
        a hundred synchronous completions -- it holds the same answers either way
        -- so the upload of the request file and the batch created from it are
        asserted on the gateway's own log. The upload's ``purpose`` is part of
        it: it is what makes the file a batch input rather than any other upload
        the framework might have made. The synchronous route staying empty is the
        other part, and the one a silent fallback fails: a client that gave up on
        batching would upload nothing, create nothing, and leave a hundred
        completions there.

        Ref: https://developers.openai.com/api/docs/guides/batch
             stdapi/routes/openai_files.py:create_file
        """
        log_start = len(agentic_server.logs)

        record = _evaluate(
            agentic_server=agentic_server,
            agentic_image=agentic_image,
            agentic_workdir=agentic_workdir,
            model_config=model_config,
            test_name=request.node.originalname or request.node.name,
        )

        _assert_every_sample_answered(record, model_config.model)
        uploads = _requests(agentic_server, log_start, "/v1/files")
        batch_creations = _requests(agentic_server, log_start, "/v1/batches")
        if agentic_server.process is not None:
            assert uploads, "no file was uploaded, so nothing was batched"
            purposes = {
                params.get("purpose")
                for entry in uploads
                if isinstance(params := entry.get("request_params"), dict)
            }
            assert "batch" in purposes, f"no upload was a batch input: {purposes}"
            assert batch_creations, (
                "the requests were answered without a batch ever being created"
            )
        synchronous = _requests(agentic_server, log_start, "/v1/chat/completions")
        assert not synchronous, (
            f"{len(synchronous)} requests were served synchronously, so the "
            "batch was not what answered them"
        )


@pytest.mark.usefixtures("batch_api")
@pytest.mark.parametrize("model_config", [_ANTHROPIC_MODEL_CONFIG])
class TestAnthropicMessageBatch:
    """The same evaluation submitted through ``/v1/messages/batches``.

    Ref: https://inspect.aisi.org.uk/models.html
         https://platform.claude.com/docs/en/build-with-claude/batch-processing
         stdapi/routes/anthropic_messages_batches.py:create_message_batch
    """

    def test_the_dataset_is_run_as_one_message_batch(
        self,
        model_config: ModelConfig,
        agentic_server: AgenticServer,
        agentic_image: str,
        agentic_workdir: Path,
        request: pytest.FixtureRequest,
    ) -> None:
        """The eval completes, and one Message Batch is what carried it.

        The Anthropic surface takes the requests in the creation call itself, so
        there is no upload to look for and the batch creation is the whole signal
        -- together with the synchronous route staying empty, which is what a
        client falling back to one message per sample would fill.

        Ref: https://platform.claude.com/docs/en/build-with-claude/batch-processing
             stdapi/routes/anthropic_messages.py:create_message
        """
        log_start = len(agentic_server.logs)

        record = _evaluate(
            agentic_server=agentic_server,
            agentic_image=agentic_image,
            agentic_workdir=agentic_workdir,
            model_config=model_config,
            test_name=request.node.originalname or request.node.name,
        )

        _assert_every_sample_answered(record, model_config.model)
        batch_creations = _requests(
            agentic_server, log_start, "/anthropic/v1/messages/batches"
        )
        if agentic_server.process is not None:
            assert batch_creations, (
                "the requests were answered without a message batch ever being created"
            )
        synchronous = _requests(agentic_server, log_start, "/anthropic/v1/messages")
        assert not synchronous, (
            f"{len(synchronous)} messages were served synchronously, so the "
            "batch was not what answered them"
        )
