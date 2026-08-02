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
from pathlib import Path

import pytest

from stdapi.aws_bedrock_mantle import MantleError
from tests.probes.probe_model import _classify, _classify_mantle, _suite_test_client

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
