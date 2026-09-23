"""The skip/fail decision a Bedrock-batch round-trip test makes at its deadline.

A batch round-trip test bounds how long it waits for a real Bedrock batch job, and
the bound can legitimately expire before the queue ever dispatches the job -- that is
queue time, not a defect, and failing it hides a real regression behind an
environmental flake every time it happens. A job that started running and still did
not finish is a different story and must still fail. This is tested offline, against
fakes, precisely because a real reproduction needs the hour the bound itself allows.

Ref: https://github.com/stdapi-ai/stdapi.ai/issues/299
     tests/_helpers.py:batch_timeout_outcome
     tests/_helpers.py:poll_batch_until_ended
"""

import pytest

from tests._helpers import batch_timeout_outcome, poll_batch_until_ended


def test_an_unprocessed_batch_skips_at_the_deadline() -> None:
    """A batch with nothing processed skips, naming the queue as the reason.

    Ref: tests/_helpers.py:batch_timeout_outcome
    """
    with pytest.raises(pytest.skip.Exception, match="queue"):
        batch_timeout_outcome(processed=False, timeout=3600.0, description="Submitted")


def test_a_processed_batch_fails_at_the_deadline() -> None:
    """A batch that started but did not finish fails rather than skipping.

    Ref: tests/_helpers.py:batch_timeout_outcome
    """
    with pytest.raises(pytest.fail.Exception):
        batch_timeout_outcome(
            processed=True, timeout=3600.0, description="in_progress, 3/100 done"
        )


def test_poll_batch_until_ended_returns_the_first_ended_state() -> None:
    """The loop returns as soon as `ended` is true, without touching the deadline.

    Ref: tests/_helpers.py:poll_batch_until_ended
    """
    states = iter([{"n": 1}, {"n": 2}, {"n": 3}])

    result = poll_batch_until_ended(
        lambda: next(states),
        ended=lambda state: state["n"] == 3,
        processed=lambda _state: True,
        describe=str,
        timeout=1000.0,
        poll_interval=0.0,
    )

    assert result == {"n": 3}


def test_poll_batch_until_ended_skips_an_unprocessed_batch_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch that never left the queue skips once the deadline is reached.

    The clock is faked rather than slept out: one read for the deadline itself,
    one for the check inside the loop, jumped forward past it -- so this proves
    the decision without spending the real timeout.

    Ref: tests/_helpers.py:poll_batch_until_ended
    """
    clock = iter([0.0, 100.0])
    monkeypatch.setattr("tests._helpers.monotonic", lambda: next(clock))
    monkeypatch.setattr("tests._helpers.sleep", lambda _seconds: None)

    with pytest.raises(pytest.skip.Exception, match="queue"):
        poll_batch_until_ended(
            lambda: "Submitted",
            ended=lambda _state: False,
            processed=lambda _state: False,
            describe=str,
            timeout=1.0,
            poll_interval=0.01,
        )


def test_poll_batch_until_ended_fails_a_processed_batch_at_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A batch that started but did not finish fails once the deadline is reached.

    Ref: tests/_helpers.py:poll_batch_until_ended
    """
    clock = iter([0.0, 100.0])
    monkeypatch.setattr("tests._helpers.monotonic", lambda: next(clock))
    monkeypatch.setattr("tests._helpers.sleep", lambda _seconds: None)

    state = ("in_progress", 3)

    with pytest.raises(pytest.fail.Exception):
        poll_batch_until_ended(
            lambda: state,
            ended=lambda _state: False,
            processed=lambda current: current[1] > 0,
            describe=str,
            timeout=1.0,
            poll_interval=0.01,
        )
