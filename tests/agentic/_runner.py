"""Shared execution, assertions and metrics for every agentic tool.

This is the half of the agentic lane that does not vary per CLI: launching a run in
the sandbox, normalising failures, checking that the traffic really reached the model
under test, and printing comparable benchmark lines.

Ref: tests/agentic/_tools.py:AgenticTool
     tests/agentic/_podman.py:run_in_container
     stdapi/models/__init__.py:validate_model
"""

from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ._podman import _redacted, run_in_container
from ._server import REPO_ROOT
from ._tools import SRC_MOUNT, AgenticResult, Invocation

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ._server import AgenticServer
    from ._tools import AgenticTool

#: Only the gateway package is mounted: the CLI never sees tests/.env, .git or
#: the virtualenv, so no credential on this machine is reachable from a run.
SOURCE_MOUNTS: Mapping[Path, str] = {REPO_ROOT / "stdapi": f"{SRC_MOUNT}/stdapi"}

#: Result text shorter than this means the CLI produced no real answer.
_MIN_RESULT_CHARS = 10

#: Model IDs a CLI may legitimately call for its own background work.
_AUXILIARY_MODEL_PREFIXES = ("anthropic.claude-opus-", "anthropic.claude-haiku-")

#: Session IDs whose CLI run completed during the current test.
_completed_runs: list[str] = []


def reset_run_tracking() -> None:
    """Forget the previous test's completed runs."""
    _completed_runs.clear()


def any_run_completed() -> bool:
    """True when at least one CLI run finished successfully in the current test."""
    return bool(_completed_runs)


@dataclass(frozen=True)
class ModelConfig:
    """One model a tool is exercised against.

    Attributes:
        model: Bedrock model ID routed by the gateway.
        extra_env: CLI environment overrides (capabilities, caching, thinking).
        supports_effort: Whether the CLI's effort levels apply to this model.
        timeout: Seconds allowed per run. A ceiling, not an expectation, but it
            has to stay below the lane's own budget: one run allowed to last
            longer than the whole pass defines the pass, and 24 workers sit idle
            behind it. The default is measured -- p95 of a run is ~250 s and the
            slowest legitimate one ~375 s -- so it leaves a wide margin and still
            fails a run that has stopped making progress.
        flaky: Whether content-quality failures downgrade to xfail. Set only for
            models known to answer inconsistently; every other failure signature
            still fails the test, so real regressions stay visible.
        extra_args: CLI arguments appended to the command line, for a test that
            needs a capability the shared configuration turns off.
    """

    model: str
    extra_env: Mapping[str, str] = field(default_factory=dict)
    supports_effort: bool = False
    timeout: int = 600
    flaky: bool = False
    extra_args: tuple[str, ...] = ()


def xfail_if_flaky(config: ModelConfig, signature: str) -> None:
    """Downgrade a known-flaky failure signature to an xfail (best effort).

    Args:
        config: Model under test.
        signature: Short label of the matched failure signature.
    """
    if config.flaky:
        pytest.xfail(f"known-flaky {signature} on {config.model} (best effort)")


def session_id(model: str, test_name: str) -> str:
    """Return a stable UUID5 for a (model, test) pair.

    Deterministic so a run's requests can be correlated with the server log
    across sessions, and so a rerun reuses the same identifier.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{model}:{test_name}"))


def run_agent(
    *,
    tool: AgenticTool,
    server: AgenticServer,
    image: str,
    config: ModelConfig,
    prompt: str,
    workdir: Path,
    test_name: str,
    effort: str | None = None,
) -> AgenticResult:
    """Run one agentic CLI in the sandbox and return its normalised result.

    Args:
        tool: CLI to drive.
        server: Gateway the CLI is pointed at.
        image: Container image holding the CLI.
        config: Model under test.
        prompt: Task handed to the agent.
        workdir: Per-test writable directory, bind-mounted into the container.
        test_name: Logical test name, used to derive the session identifier.
        effort: Optional effort level, for CLIs and models that support one.

    Returns:
        The parsed result of the run.

    Raises:
        subprocess.TimeoutExpired: If the run exceeds the model's timeout and the
            model is not marked flaky.
    """
    invocation = Invocation(
        workdir=workdir,
        port=server.forward_port or 0,
        api_key=server.api_key,
        model=config.model,
        prompt=prompt,
        session_id=session_id(config.model, test_name),
        effort=effort,
        extra_env=config.extra_env,
        extra_args=config.extra_args,
    )
    tool.prepare_workdir(invocation)
    command = tool.build(invocation)

    try:
        process = run_in_container(
            image=image,
            argv=command.argv,
            workdir=workdir,
            mounts=SOURCE_MOUNTS,
            env=command.env,
            forward_port=server.forward_port,
            timeout=config.timeout,
            stdin=command.stdin,
        )
    except subprocess.TimeoutExpired:
        xfail_if_flaky(config, "CLI timeout")
        raise

    if process.returncode != 0:
        xfail_if_flaky(config, f"CLI exit code {process.returncode}")
        # Redacted before truncation: CLI output can echo the environment, and
        # truncating first could leave a recognisable fragment of the API key.
        pytest.fail(
            f"{tool.id} exited with code {process.returncode}\n"
            f"stdout: {_redacted(process.stdout, command.env)[:1500]}\n"
            f"stderr: {_redacted(process.stderr, command.env)[:500]}"
        )
    try:
        result = tool.parse(process.stdout)
    except ValueError as exc:
        xfail_if_flaky(config, "unparsable output")
        # Parse errors embed output excerpts, so they get the same redaction.
        pytest.fail(_redacted(str(exc), command.env))
    _completed_runs.append(invocation.session_id)
    _record_sample(tool, config, prompt, process.stdout, result)
    return result


#: Environment variable naming a directory to drop one transcript per run into.
#:
#: Unset in a normal run: this exists to collect examples of what each client
#: actually sends and prints, which is otherwise only visible inside a container
#: that is deleted when the run ends.
SAMPLE_DIR_VAR = "AGENTIC_SAMPLE_DIR"

#: Characters of raw CLI output kept at each end of a clipped transcript.
_SAMPLE_EDGE_CHARS = 40_000


def _clip(text: str) -> str:
    """Return *text*, or its two ends around a marker naming what was dropped.

    Args:
        text: Raw output to record.

    Returns:
        Text short enough to read.
    """
    if len(text) <= 2 * _SAMPLE_EDGE_CHARS:
        return text
    dropped = len(text) - 2 * _SAMPLE_EDGE_CHARS
    return (
        f"{text[:_SAMPLE_EDGE_CHARS]}\n"
        f"\n[... {dropped} characters of streamed output omitted ...]\n\n"
        f"{text[-_SAMPLE_EDGE_CHARS:]}"
    )


def _record_sample(
    tool: AgenticTool,
    config: ModelConfig,
    prompt: str,
    stdout: str,
    result: AgenticResult,
) -> None:
    """Write one successful run's prompt, raw output and metrics to a file.

    A run that parsed but reported an error is skipped: parsing only proves the
    CLI printed something readable, and the point of a captured file is to show
    a client working as intended. One file per tool and model; a later run of the
    same pair overwrites it.

    Args:
        tool: CLI that produced the output.
        config: Model the run used.
        prompt: Task the agent was given.
        stdout: Everything the CLI printed.
        result: Parsed result, for the summary header.
    """
    directory = os.environ.get(SAMPLE_DIR_VAR)
    if not directory or result.is_error:
        return
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    slug = f"{tool.id}__{config.model}".replace("/", "_").replace(":", "_")
    (target / f"{slug}.txt").write_text(
        f"# tool:   {tool.id}\n"
        f"# model:  {config.model}\n"
        f"# route:  {tool.route}\n"
        f"# steps:  {result.steps}\n"
        f"# tokens: in={result.input_tokens} out={result.output_tokens}\n"
        f"\n===== PROMPT =====\n{prompt}\n"
        f"\n===== RAW CLI OUTPUT =====\n{_clip(stdout)}\n"
        f"\n===== PARSED ANSWER =====\n{_clip(result.text)}\n",
        encoding="utf-8",
    )


def assert_result(
    result: AgenticResult,
    *,
    config: ModelConfig,
    min_steps: int = 0,
    contains: str | None = None,
    any_of: Sequence[str] = (),
) -> str:
    """Assert an agentic run produced a real, codebase-derived answer.

    The step floor is what distinguishes a genuine run from an answer recited
    from the model's own knowledge: the task cannot be completed without reading
    files, so too few steps means the tool-use round trip never carried file
    contents back through the gateway. It is deliberately far below the ~10 calls
    the prompts target, because the weaker models vary in how many they need.

    Args:
        result: Normalised run result.
        config: Model under test, for flaky-signature scoping.
        min_steps: Minimum turns (Claude Code) or shell executions (Codex).
        contains: Substring that must appear in the answer, case-insensitive.
        any_of: Substrings of which at least one must appear, case-insensitive.

    Returns:
        The answer text.
    """
    if result.is_error:
        xfail_if_flaky(config, "CLI-reported error")
        pytest.fail(f"{config.model} reported an error: {result.error_detail}")
    if len(result.text) <= _MIN_RESULT_CHARS:
        xfail_if_flaky(config, "empty result")
        pytest.fail(f"result is unexpectedly short: {result.text!r}")
    lowered = result.text.lower()
    if contains and contains.lower() not in lowered:
        xfail_if_flaky(config, "content assertion")
        pytest.fail(f"Expected {contains!r} in result text:\n{result.text}")
    if any_of and not any(keyword.lower() in lowered for keyword in any_of):
        xfail_if_flaky(config, "content assertion")
        pytest.fail(f"Expected one of {list(any_of)} in result:\n{result.text}")
    if min_steps > 0 and result.steps < min_steps:
        xfail_if_flaky(config, "step-count assertion")
        pytest.fail(
            f"Expected at least {min_steps} steps (the model must explore the "
            f"source to answer), got {result.steps} — response:\n{result.text[:300]}"
        )
    return result.text


def log_metrics(
    tool: AgenticTool, result: AgenticResult, config: ModelConfig, test_name: str
) -> None:
    """Print one benchmark line per run, visible with ``pytest -s``.

    Token counts are reported rather than cost: a CLI's own cost estimate uses
    its vendor's pricing regardless of which Bedrock model was actually routed,
    which makes cross-model cost comparison meaningless.

    Collect a whole run with e.g.::

        pytest --agentic -s 2>&1 | grep CC-METRICS
    """
    cache = ""
    if result.cache_read or result.cache_created:
        cache = (
            f" [cache_read={result.cache_read} cache_created={result.cache_created}]"
        )
    print(  # noqa: T201
        f"\n{tool.metrics_prefix} | {config.model:<30} | {test_name:<40} "
        f"| steps={result.steps:>3} "
        f"| in={result.input_tokens:>6} out={result.output_tokens:>5}{cache}"
    )


def grounding_requests(server: AgenticServer, log_start: int) -> int:
    """Return the billed web-grounding calls the gateway logged since *log_start*.

    Amazon Bedrock runs the search inside the model invocation, so a grounded
    answer leaves no trace in the CLI's own transcript beyond the text. The
    gateway's usage log is where it is observable -- and where AWS bills it.

    Args:
        server: Gateway whose log is inspected.
        log_start: Log index captured before the test ran.

    Returns:
        Sum of ``grounding_requests`` over the usage entries recorded since.
    """
    total = 0
    for entry in server.log_entries(log_start):
        usages = entry.get("usage")
        if not isinstance(usages, list):
            continue
        total += sum(
            int(usage.get("grounding_requests") or 0)
            for usage in usages
            if isinstance(usage, dict)
        )
    return total


def assert_model_identity(
    *,
    tool: AgenticTool,
    server: AgenticServer,
    log_start: int,
    config: ModelConfig,
    test_name: str,
    ran: bool,
) -> None:
    """Assert every request this test produced targeted the model under test.

    Without this a test would still pass if the CLI silently fell back to its own
    default model, so the run would prove nothing about the gateway routing the
    model named in the parametrization.

    Requests are attributed to the test by session identifier when the CLI
    propagates one (``request_user_id``); otherwise only positionally, ignoring
    entries that arrive before the first expected-model entry, because a
    previously timed-out CLI's orphaned streams can still be draining.

    Args:
        tool: CLI that was driven.
        server: Gateway whose log is inspected.
        log_start: Log index captured before the test ran.
        config: Model the requests must target.
        test_name: Logical test name, to rebuild the session identifier.
        ran: Whether a CLI run completed successfully during the test.
    """
    if server.process is None:
        return  # External server: no log to inspect.

    expected = session_id(config.model, test_name)
    model_ids = [
        str(entry["model_id"])
        for entry in server.log_entries(log_start)
        if entry.get("model_id")
        and (
            not tool.attributes_sessions
            or expected in str(entry.get("request_user_id") or "")
        )
    ]

    if not model_ids:
        # A completed run with no attributable request means attribution itself
        # broke and the check silently stopped covering anything.
        assert not (ran and tool.attributes_sessions), (
            "No server-log request was attributed to this test's session ID "
            f"although {tool.id} completed successfully: attribution via "
            "request_user_id is broken."
        )
        return

    prefix = config.model.split(":", maxsplit=1)[0]
    if not tool.attributes_sessions:
        first = next(
            (i for i, mid in enumerate(model_ids) if mid.startswith(prefix)), None
        )
        if first is None:
            return  # Never reached the model; nothing this test can assert.
        model_ids = model_ids[first:]

    unexpected = [mid for mid in model_ids if not mid.startswith(prefix)]
    if unexpected and any(mid.startswith(prefix) for mid in model_ids):
        # A CLI's own background calls are tolerated once the conversation itself
        # reached the expected model. Auxiliary prefixes that the expected model's
        # own prefix matches are not dropped, so a mis-route to another build of
        # the same family is still caught.
        auxiliary = tuple(
            p for p in _AUXILIARY_MODEL_PREFIXES if not prefix.startswith(p)
        )
        unexpected = [mid for mid in unexpected if not mid.startswith(auxiliary)]
    assert not unexpected, (
        f"Model identity mismatch: expected requests to {config.model!r}, "
        f"but saw {list(dict.fromkeys(unexpected))} in server logs."
    )
