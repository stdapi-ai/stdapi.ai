# mypy: disable-error-code="import-not-found"
"""One inspect-ai evaluation, batched, driven against the gateway.

Executed by ``python /work/inspect_eval.py`` inside the ``inspect`` image group's
container and never on the host: the framework lives in that image alone, because
it cannot share an interpreter with the gateway (see ``Containerfile.inspect``).
Everything varying per run arrives through the environment, so the file is copied
in verbatim rather than rendered.

The whole dataset becomes a single provider batch: ``BatchConfig`` holds requests
back until it has ``size`` of them, and the dataset carries exactly that many, so
one submission covers the run. Which surface it is submitted to follows from the
provider prefix on the model name alone -- ``openai/`` uploads a request file and
creates a batch, ``anthropic/`` creates a Message Batch -- which is what makes one
script cover both.

Prints exactly one JSON object, and writes the same object next to itself:
``tests/agentic/_tools.py:_inspect_parse`` normalises the printed copy into the
lane's shared result, while the assertions need every sample in the record and
read the file back through ``tests/agentic/_tools.py:inspect_record``.

Ref: https://inspect.aisi.org.uk/models.html
     https://inspect.aisi.org.uk/parallelism.html
     tests/agentic/test_inspect_ai.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import anyio
from inspect_ai import Task, eval_async
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import BatchConfig, GenerateConfig
from inspect_ai.solver import generate

#: Gateway under test, without a route prefix.
BASE_URL = os.environ["STDAPI_BASE_URL"]

#: Route prefix the provider's SDK is pointed at (``/v1`` or ``/anthropic``).
BASE_ROUTE = os.environ["INSPECT_ROUTE"]

#: Provider the framework resolves the model through (``openai`` or ``anthropic``).
PROVIDER = os.environ["INSPECT_PROVIDER"]

#: Model every request of the batch names, without the provider prefix.
CHAT_MODEL = os.environ["CHAT_MODEL"]

#: Gateway API key, read by whichever vendor SDK the provider drives.
API_KEY = os.environ["STDAPI_API_KEY"]

#: Requests in the batch, and therefore samples in the dataset.
BATCH_SIZE = int(os.environ["BATCH_SIZE"])

#: Reference each request asks to have repeated back.
MARKER = os.environ["MARKER"]

#: Tokens one answer may take; every sample asks for a single short line.
MAX_TOKENS = int(os.environ["MAX_TOKENS"])

#: Record written beside this script, read back by the assertions.
RUN_OUTPUT = Path(__file__).parent / "inspect_run.json"

#: Extra provider arguments, as JSON; ``responses_api`` is selected this way.
MODEL_ARGS: dict[str, Any] = json.loads(os.environ.get("MODEL_ARGS") or "{}")


def marker(index: int) -> str:
    """Return the reference the sample at *index* asks to have repeated.

    Args:
        index: Position of the sample in the dataset.

    Returns:
        A reference belonging to that sample and to no other.
    """
    return f"{MARKER}-{index:03d}"


def build_task() -> Task:
    """Return the evaluation, configured to batch its whole dataset at once.

    Every sample carries a reference of its own, so an answer can be traced back
    to the request that asked for it once the results have been through a batch
    and come back in whatever order the backend chose.

    Returns:
        The task, with one sample per request of a full batch.
    """
    dataset = MemoryDataset(
        [
            Sample(
                id=index,
                input=(
                    "Repeat this reference exactly and write nothing else: "
                    f"{marker(index)}"
                ),
                target=marker(index),
            )
            for index in range(BATCH_SIZE)
        ]
    )
    return Task(
        dataset=dataset,
        solver=generate(),
        config=GenerateConfig(
            batch=BatchConfig(size=BATCH_SIZE), max_tokens=MAX_TOKENS
        ),
    )


async def run() -> dict[str, Any]:
    """Run the evaluation and return the record describing what came back.

    ``eval_async`` is awaited rather than ``eval`` called: the synchronous
    wrapper starts an event loop of its own. ``max_samples`` is raised to the
    batch size because it otherwise defaults to a handful, which would leave the
    batcher holding far fewer requests than the backend accepts.

    Returns:
        The record: the run's status, every sample's answer, and the usage the
        framework accounted for.
    """
    logs = await eval_async(
        build_task(),
        model=f"{PROVIDER}/{CHAT_MODEL}",
        model_base_url=f"{BASE_URL}{BASE_ROUTE}",
        model_args={"api_key": API_KEY, **MODEL_ARGS},
        log_dir=str(Path(__file__).parent / "logs"),
        max_samples=BATCH_SIZE,
    )
    log = logs[0]
    usage = list((log.stats.model_usage or {}).values())
    return {
        "provider": PROVIDER,
        "model": CHAT_MODEL,
        "status": log.status,
        "error": str(log.error) if log.error else None,
        "samples": [
            {
                "id": sample.id,
                "completion": sample.output.completion or "",
                "model": sample.output.model,
            }
            for sample in (log.samples or ())
        ],
        "usage": {
            "input_tokens": sum(entry.input_tokens for entry in usage),
            "output_tokens": sum(entry.output_tokens for entry in usage),
        },
    }


def main() -> None:
    """Run the evaluation, then print and write its record."""
    record = anyio.run(run)
    serialized = json.dumps(record)
    RUN_OUTPUT.write_text(serialized, encoding="utf-8")
    print(serialized)  # noqa: T201


if __name__ == "__main__":
    main()
