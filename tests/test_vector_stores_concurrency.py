"""Bounded fan-out shared by the vector store engine and the indexes behind it.

A vector store request is a fan-out whose width the caller chooses: attaching a
batch reads one record per file, a search reads one page of chunks per key
batch, and detaching deletes them the same way. Those calls run under a fixed
bound so one request cannot open hundreds of backend calls at once, and the
bound has to behave like a ceiling rather than a barrier -- a slow call may
delay itself, never the calls queued behind it.

The failure side matters as much: the first refusal answers the request, so the
calls beside it are cancelled instead of running on to be billed for a result
nobody reads, the calls that never started are closed rather than dropped, and
the refusal reaches the caller as itself. That last point is load-bearing --
every index call is wrapped in ``feature_unavailable_guard``, which catches by
exception type, so an exception group would slip past it and answer 500 where a
503 is owed.

Ref: stdapi/vector_stores/_concurrency.py:gather_bounded
     stdapi/vector_stores/s3_vectors.py:S3VectorsIndex.get_vectors
     stdapi/vector_stores/records.py:gather_records
"""

from __future__ import annotations

from asyncio import CancelledError, Event, create_task, sleep
from inspect import CORO_CLOSED, getcoroutinestate

import pytest
from botocore.exceptions import ClientError

from stdapi.api_errors import ApiError
from stdapi.vector_stores._concurrency import gather_bounded

pytestmark = pytest.mark.local

#: Concurrency bound under test, small enough to read in an assertion message.
_WAVE = 4

#: Calls per fan-out, several times the bound so the bound is observable.
_CALLS = _WAVE * 3

#: Event loop passes that let every runnable call reach its next await.
_SETTLE_PASSES = _CALLS * 4


async def _settle() -> None:
    """Give every task still runnable the chance to reach its next await."""
    for _ in range(_SETTLE_PASSES):
        await sleep(0)


class _Recorder:
    """Stands in for the backend calls of one fan-out.

    Records what started, what finished and the concurrency it all ran at, and
    can hold every call on a gate the test opens so what became of them is read
    after the fan-out has been answered rather than raced against it.
    """

    def __init__(self) -> None:
        """Start with nothing recorded and the gate closed."""
        self.gate = Event()
        self.started: list[int] = []
        self.finished: list[int] = []
        self.active = 0
        self.peak = 0

    async def call(self, index: int, yields: int = 1) -> int:
        """Run one call, staying in flight long enough for siblings to start.

        Args:
            index: Position of this call in the fan-out.
            yields: Event loop passes the call stays in flight for.

        Returns:
            The index it was given, identifying the call in the results.
        """
        self.started.append(index)
        self.active += 1
        self.peak = max(self.peak, self.active)
        for _ in range(yields):
            await sleep(0)
        self.active -= 1
        self.finished.append(index)
        return index

    async def held(self, index: int) -> int:
        """Run one call that only the gate can complete.

        Args:
            index: Position of this call in the fan-out.

        Returns:
            The index it was given, identifying the call in the results.
        """
        self.started.append(index)
        await self.gate.wait()
        self.finished.append(index)
        return index

    async def refuse(self, index: int) -> int:
        """Run one call the backend refuses.

        Args:
            index: Position of this call in the fan-out.

        Returns:
            Never; the call always raises.

        Raises:
            ApiError: Always, standing in for a refused backend call.
        """
        self.started.append(index)
        msg = "The backend refused this call."
        raise ApiError(msg)


class TestGatherBounded:
    """Ordering, the bound, and what a failure does to the calls beside it.

    Ref: stdapi/vector_stores/_concurrency.py:gather_bounded
    """

    async def test_no_calls_returns_no_results(self) -> None:
        """An empty fan-out is answered without touching the backend.

        Attaching a batch whose files are all already attached reaches the
        helper with nothing to do, and must not raise there.
        """
        assert await gather_bounded([], _WAVE) == []

    async def test_results_come_back_in_the_order_the_calls_were_given(self) -> None:
        """Results follow the call order, not the order the calls completed.

        Each call is given a shorter time in flight than the one before it, so
        completion order is the reverse of call order: a result list rebuilt
        from completions would come back reversed.
        """
        recorder = _Recorder()

        results = await gather_bounded(
            [recorder.call(index, yields=_CALLS - index) for index in range(_CALLS)],
            _WAVE,
        )

        assert results == list(range(_CALLS))
        assert recorder.finished != list(range(_CALLS)), (
            "the calls must have completed out of order for this to prove anything"
        )

    async def test_fan_out_never_exceeds_the_bound(self) -> None:
        """A fan-out several times the bound runs at most the bound at once.

        Without the bound every call would start before the first completed,
        so the observed peak would be the whole call count.
        """
        recorder = _Recorder()

        await gather_bounded(
            [recorder.call(index, yields=3) for index in range(_CALLS)], _WAVE
        )

        assert len(recorder.started) == _CALLS, "every call must run"
        assert recorder.peak == _WAVE, (
            f"{_CALLS} calls ran {recorder.peak} at once; the bound is {_WAVE}"
        )

    async def test_a_slow_call_does_not_hold_back_the_ones_behind_it(self) -> None:
        """The bound is a ceiling on calls in flight, not a barrier between batches.

        One call is held open while the rest complete. A batched fan-out would
        stop at the batch holding it and start nothing further; a ceiling lets
        every other call take the slots it is not using.
        """
        recorder = _Recorder()
        calls = [recorder.held(0)]
        calls += [recorder.call(index) for index in range(1, _CALLS)]

        fan_out = create_task(gather_bounded(calls, _WAVE))
        await _settle()

        assert len(recorder.started) == _CALLS, (
            f"only {len(recorder.started)} of {_CALLS} calls started while one "
            "was held open"
        )
        recorder.gate.set()
        assert await fan_out == list(range(_CALLS))

    async def test_a_refused_call_stops_the_calls_beside_it(self) -> None:
        """One refusal ends the whole fan-out instead of leaving it running.

        The request is already answered with the error, so a call still in
        flight bills the account for a result nothing will read.
        """
        recorder = _Recorder()
        calls = [recorder.held(0), recorder.refuse(1)]
        calls += [recorder.held(index) for index in range(2, _CALLS)]

        with pytest.raises(ApiError):
            await gather_bounded(calls, _WAVE)

        started_when_refused = list(recorder.started)
        recorder.gate.set()
        await _settle()

        assert recorder.finished == [], (
            f"calls {recorder.finished} ran on past the refusal that answered "
            "the request"
        )
        assert recorder.started == started_when_refused, (
            "calls started after the request was answered"
        )
        assert len(started_when_refused) < _CALLS, (
            "the bound must have kept the calls behind the refusal from starting"
        )

    async def test_a_refused_call_closes_the_calls_that_never_started(self) -> None:
        """Calls dropped by a refusal are closed, not left unawaited.

        A coroutine that is never awaited keeps its frame alive until it is
        collected and is reported then, out of the request that built it.
        """
        recorder = _Recorder()
        calls = [recorder.refuse(0)]
        calls += [recorder.held(index) for index in range(1, _CALLS)]

        with pytest.raises(ApiError):
            await gather_bounded(calls, _WAVE)

        ran = set(recorder.started)
        dropped = [call for index, call in enumerate(calls) if index not in ran]
        assert dropped, "the bound must have kept some calls from ever starting"
        assert [getcoroutinestate(call) for call in dropped] == [CORO_CLOSED] * len(
            dropped
        ), "every call dropped by the refusal must be closed"

    async def test_the_refusal_reaches_the_caller_as_itself(self) -> None:
        """The failing call's own exception is raised, not a group holding it.

        Index calls are wrapped in a guard that maps a denied backend call to
        the feature being unavailable, and it catches by exception type: a
        group would slip past it and answer a 500 where a 503 is owed.
        """
        recorder = _Recorder()
        denied = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "denied"}},
            "GetVectors",
        )

        async def refuse() -> int:
            """Refuse the way a denied backend call does.

            Returns:
                Never; the call always raises.

            Raises:
                ClientError: Always.
            """
            raise denied

        with pytest.raises(ClientError) as raised:
            await gather_bounded([recorder.call(0), refuse(), recorder.call(2)], _WAVE)

        assert raised.value is denied

    async def test_cancelling_the_fan_out_cancels_the_calls_it_started(self) -> None:
        """A cancelled fan-out takes its calls with it.

        A caller racing two reads cancels the loser, and the calls it opened
        must end with it rather than outliving the request.
        """
        recorder = _Recorder()
        calls = [recorder.held(index) for index in range(_CALLS)]

        fan_out = create_task(gather_bounded(calls, _WAVE))
        await _settle()
        fan_out.cancel()
        with pytest.raises(CancelledError):
            await fan_out

        recorder.gate.set()
        await _settle()

        assert recorder.finished == [], (
            f"calls {recorder.finished} ran on past the cancellation"
        )
        assert [getcoroutinestate(call) for call in calls[_WAVE:]] == [CORO_CLOSED] * (
            _CALLS - _WAVE
        ), "every call the bound had not started must be closed"
