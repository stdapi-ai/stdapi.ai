"""The prober's offline plumbing: import order and outcome classification.

Nothing here calls a model. What is pinned is the machinery a live sweep relies
on: the discovery path importing the suite's conftest before starlette's test
client (the conftest asserts that order), and the classifiers that decide
whether an exception is the model's answer or the probe's own fault.

Ref: tests/probes/probe_model.py
     tests/conftest.py (httpx2 alias installed before starlette.testclient)
"""

from __future__ import annotations

import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from stdapi.aws_bedrock_mantle import MantleError
from tests.probes import probe_model
from tests.probes.probe_model import (
    _OUTPUT_BUDGET,
    MANTLE_PROBES,
    PROBES,
    Probe,
    _chat_truncated,
    _classify,
    _classify_mantle,
    _observed_result,
    _reasoning_off,
    _run_probe,
    _suite_test_client,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

pytestmark = pytest.mark.local

#: Repository root, the working directory the prober is documented to run from.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Seconds allowed for a fresh interpreter to import the suite's conftest.
_IMPORT_TIMEOUT = 180


class TestDiscoveryImportOrder:
    """``--all`` discovery survives the conftest's import-order assertion.

    Ref: tests/probes/probe_model.py:_suite_test_client
    """

    def test_suite_test_client_imports_cleanly_in_a_fresh_interpreter(self) -> None:
        """The conftest is imported before starlette's test client.

        The suite's conftest installs an ``httpx2`` alias and asserts that
        ``starlette.testclient`` was not imported before it, so the wrong order
        aborts every ``--all`` sweep with an AssertionError. Only a fresh
        interpreter can pin the order: in this process the conftest is long
        imported.

        Ref: tests/probes/probe_model.py:discover_chat_models
        """
        code = (
            "from tests.probes.probe_model import _suite_test_client\n"
            "assert _suite_test_client().__name__ == 'TestClient'\n"
        )
        process = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=_IMPORT_TIMEOUT,
            check=False,
        )
        assert process.returncode == 0, process.stderr[-2000:]

    def test_suite_test_client_returns_the_test_client_class(self) -> None:
        """The helper hands back the class discovery instantiates.

        Ref: tests/probes/probe_model.py:_suite_test_client
        """
        assert _suite_test_client().__name__ == "TestClient"


class TestClassify:
    """A Converse refusal is told apart from a fault in the probe itself.

    Ref: tests/probes/probe_model.py:_classify
    """

    def test_a_validation_exception_is_a_rejection(self) -> None:
        """Bedrock's refusal class counts as the model declining the shape.

        Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
        """
        exc = type("ValidationException", (Exception,), {})("malformed input")
        assert _classify(exc) == "rejected"

    def test_a_refusal_message_is_a_rejection_whatever_the_class(self) -> None:
        """Refusal wording counts even from another exception class.

        Prompt caching, for one, is refused with an AccessDeniedException.

        Ref: tests/probes/probe_model.py:_REFUSAL_MARKERS
        """
        assert _classify(Exception("This model does not support tools")) == "rejected"

    def test_anything_else_is_an_error(self) -> None:
        """An unexplained failure is recorded verbatim, not guessed at."""
        assert _classify(Exception("Connection reset by peer")) == "error"


class TestClassifyMantle:
    """A Mantle refusal is told apart from transport and credential faults.

    Uses the real ``MantleError`` so the ``status`` contract the classifier
    reads is the one the client actually raises.

    Ref: tests/probes/probe_model.py:_classify_mantle
         stdapi/aws_bedrock_mantle.py:MantleError
    """

    def test_a_4xx_is_a_rejection(self) -> None:
        """The endpoint refusing the probed shape is the answer sought."""
        assert _classify_mantle(MantleError("unknown field", status=400)) == "rejected"
        assert _classify_mantle(MantleError("no such model", status=404)) == "rejected"

    def test_throttling_is_an_error(self) -> None:
        """A 429 is a transient region condition, not the model's refusal."""
        exc = MantleError("Too many requests", status=429, failover=True)
        assert _classify_mantle(exc) == "error"

    def test_connection_and_credential_faults_are_errors(self) -> None:
        """Transport (503) and mapped credential (500) faults are not refusals.

        Ref: stdapi/aws_bedrock_mantle.py:_map_error
        """
        unreachable = MantleError(
            "The service is temporarily unavailable. Retry the request.",
            status=503,
            failover=True,
        )
        auth = MantleError(
            "The request could not be completed. Retry the request.", status=500
        )
        assert _classify_mantle(unreachable) == "error"
        assert _classify_mantle(auth) == "error"

    def test_a_statusless_exception_falls_back_to_refusal_markers(self) -> None:
        """A bare exception is an error unless its message is a refusal.

        Ref: tests/probes/probe_model.py:_REFUSAL_MARKERS
        """
        assert _classify_mantle(ConnectionError("reset")) == "error"
        refusal = Exception("streaming isn't supported for this model")
        assert _classify_mantle(refusal) == "rejected"


class _RecordingClient:
    """Stand-in bedrock-runtime client answering a fixed sequence of responses."""

    def __init__(self, *responses: dict[str, Any]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []

    async def converse(self, **request: Any) -> dict[str, Any]:  # noqa: ANN401
        """Record the request and answer the next canned response."""
        self.requests.append(request)
        return self.responses.pop(0)


def _response(
    *blocks: dict[str, Any], stop: str = "end_turn", **usage: int
) -> dict[str, Any]:
    """Build a Converse response carrying *blocks* and *usage*."""
    return {
        "output": {"message": {"role": "assistant", "content": list(blocks)}},
        "stopReason": stop,
        "usage": usage,
    }


#: Converse probes judged on the answer, whose budget schema 6 raised.
_CONVERSE_ANSWER_PROBES = (
    "stop_sequences",
    "tool_use",
    "tool_choice_any",
    "tool_choice_tool",
    "json_mode",
    "output_config_json_schema",
)

#: Chat Completions probes judged on the answer, whose budget schema 6 raised.
_MANTLE_ANSWER_PROBES = (
    "stop_sequences",
    "tool_use",
    "tool_choice_any",
    "tool_choice_tool",
    "parallel_tool_calls_false",
    "json_mode",
    "json_schema",
)

#: Schema version that raised the answer-judged probes' output budget.
_BUDGET_SCHEMA_VERSION = 6


def _probe_named(name: str) -> Probe:
    """Return the committed Converse probe called *name*."""
    return next(probe for probe in PROBES if probe.name == name)


class TestObservedResult:
    """A successful call is classified by the effect the probe looks for.

    Ref: tests/probes/probe_model.py:_observed_result
    """

    def test_an_answer_cut_before_the_effect_is_an_error(self) -> None:
        """A reasoning model spending the budget first must not read as ``accepted``.

        Kimi K3's tool and JSON probes read as ``accepted`` at 64 output tokens.
        """
        response = _response({"reasoningContent": {}}, stop="max_tokens")

        result = _observed_result(_probe_named("tool_use"), response, truncated=True)

        assert result.outcome == "error"
        assert "output budget" in result.detail

    def test_an_effect_outside_the_answer_is_still_judged(self) -> None:
        """A cache counter does not depend on the answer reaching its end."""
        probe = _probe_named("implicit_prompt_cache")

        result = _observed_result(probe, _response(stop="max_tokens"), truncated=True)

        assert result.outcome == "accepted"

    def test_a_probe_without_an_effect_is_accepted_whatever_the_stop(self) -> None:
        """A probe asking only for acceptance is not failed by a short budget."""
        probe = _probe_named("top_k")

        result = _observed_result(probe, _response(stop="max_tokens"), truncated=True)

        assert result.outcome == "accepted"

    def test_an_observed_effect_is_supported(self) -> None:
        """The effect wins over a truncated answer that still shows it."""
        response = _response({"toolUse": {"name": "get_weather"}}, stop="max_tokens")

        result = _observed_result(_probe_named("tool_use"), response, truncated=True)

        assert result.outcome == "supported"

    def test_output_probes_get_a_budget_a_reasoning_model_can_answer_in(self) -> None:
        """Every probe judged on its answer asks for more than the 64-token baseline.

        Ref: tests/probes/probe_model.py:_OUTPUT_BUDGET
        """
        for name in _CONVERSE_ANSWER_PROBES:
            probe = _probe_named(name)
            assert probe.overrides["inferenceConfig"]["maxTokens"] >= _OUTPUT_BUDGET
            assert probe.changed_in >= _BUDGET_SCHEMA_VERSION, name

    def test_mantle_output_probes_get_the_same_budget(self) -> None:
        """The Chat Completions probes judged on their answer get the budget too.

        Ref: tests/probes/probe_model.py:MANTLE_PROBES
        """
        for name in _MANTLE_ANSWER_PROBES:
            probe = next(p for p in MANTLE_PROBES if p.name == name)
            assert probe.overrides["max_tokens"] >= _OUTPUT_BUDGET, name
            assert probe.changed_in >= _BUDGET_SCHEMA_VERSION, name

    def test_a_chat_answer_cut_at_the_budget_is_truncated(self) -> None:
        """``finish_reason: "length"`` is what marks a Mantle answer as cut short.

        Ref: tests/probes/probe_model.py:_chat_truncated
        """
        cut = {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
        done = {"choices": [{"finish_reason": "stop", "message": {"content": "{}"}}]}

        assert _chat_truncated(cut)
        assert not _chat_truncated(done)
        assert not _chat_truncated({})
        result = _observed_result(
            next(p for p in MANTLE_PROBES if p.name == "json_mode"),
            cut,
            truncated=_chat_truncated(cut),
        )
        assert result.outcome == "error"

    async def test_the_mantle_prober_judges_a_cut_answer_as_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``probe_mantle_model`` hands a cut answer to the classifier as truncated.

        Every probe after the baseline answers ``finish_reason: "length"`` with no
        content, so a JSON probe judged on its answer must read as ``error``.

        Ref: tests/probes/probe_model.py:probe_mantle_model
        """
        from stdapi import aws_bedrock_mantle  # noqa: PLC0415

        done = {"finish_reason": "stop", "message": {"content": "OK"}}
        cut = {"finish_reason": "length", "message": {"content": ""}}
        calls: list[dict[str, Any]] = []

        async def fake_invoke(
            _region: str,
            _path: str,
            payload: dict[str, Any],
            **_: Any,  # noqa: ANN401
        ) -> dict[str, Any]:
            calls.append(payload)
            return {"choices": [done if len(calls) == 1 else cut]}

        @asynccontextmanager
        async def no_session() -> AsyncIterator[None]:
            yield

        monkeypatch.setattr(aws_bedrock_mantle, "invoke", fake_invoke)
        monkeypatch.setattr(aws_bedrock_mantle, "mantle_http_session", no_session)

        record = await probe_model.probe_mantle_model("m", "us-east-1")

        outcomes = {entry["name"]: entry["outcome"] for entry in record["probes"]}
        assert outcomes["baseline"] in {"supported", "accepted"}
        assert outcomes["json_mode"] == "error"


class TestRepeatedProbe:
    """A repeated probe observes its last response only.

    Ref: tests/probes/probe_model.py:_run_probe
    """

    async def test_implicit_cache_is_read_on_the_second_call(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same request, with no cache point, is sent twice; the read counts."""
        monkeypatch.setattr(probe_model, "_REPEAT_DELAY", 0)
        client = _RecordingClient(
            _response({"text": "OK"}, cacheWriteInputTokens=2802),
            _response({"text": "OK"}, cacheReadInputTokens=2802),
        )

        result = await _run_probe(
            client, "m", _probe_named("implicit_prompt_cache"), {"messages": []}
        )

        assert len(client.requests) == 2
        assert client.requests[0] == client.requests[1]
        assert "cachePoint" not in str(client.requests[0])
        assert result.outcome == "supported"
        assert result.detail == "cacheRead=2802 on the repeated call"

    async def test_a_write_alone_is_not_a_read(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A second call that only writes again shows no implicit caching."""
        monkeypatch.setattr(probe_model, "_REPEAT_DELAY", 0)
        client = _RecordingClient(
            _response({"text": "OK"}, cacheWriteInputTokens=2802),
            _response({"text": "OK"}, cacheWriteInputTokens=2802),
        )

        result = await _run_probe(
            client, "m", _probe_named("implicit_prompt_cache"), {"messages": []}
        )

        assert result.outcome == "accepted"


class TestReasoningOff:
    """An answer without reasoning is told apart from one that reasoned.

    Ref: tests/probes/probe_model.py:_reasoning_off
    """

    def test_a_short_answer_without_reasoning_shows_the_switch(self) -> None:
        """GPT-6 at ``none``: a text block alone, in 5 output tokens."""
        assert _reasoning_off(_response({"text": "15"}, outputTokens=5))

    def test_reasoning_text_means_the_switch_was_ignored(self) -> None:
        """Kimi K3 with ``thinking`` disabled still reasoned."""
        block = {"reasoningContent": {"reasoningText": {"text": "Let me think"}}}

        assert not _reasoning_off(_response(block, {"text": "15"}, outputTokens=20))

    def test_a_long_answer_means_hidden_reasoning(self) -> None:
        """A redacted reasoning block carries no text but still costs tokens."""
        block = {"reasoningContent": {"redactedContent": b"x"}}

        assert not _reasoning_off(_response(block, {"text": "15"}, outputTokens=59))
