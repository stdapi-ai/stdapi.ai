"""Factories, normalizers and assertions shared by more than one test module.

Anything used by a single module belongs in that module; this is for the shapes
where a drifting copy is the real risk. Fixtures live in ``tests/conftest.py``
instead -- only plain callables belong here, so a test module can import them
without pytest fixture resolution getting involved.

Ref: stdapi/models/__init__.py:ModelDetails
     stdapi/monitoring.py:EventLog
"""

from __future__ import annotations

import struct
import zlib
from datetime import UTC, datetime
from time import monotonic, sleep
from typing import TYPE_CHECKING, Any, NoReturn

import pytest
from botocore.exceptions import ClientError
from pybase64 import b64decode, b64encode

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from openai.types import CreateEmbeddingResponse

    from stdapi.models import ModelDetails
    from stdapi.monitoring import EventLog


def red_png() -> bytes:
    """Build a minimal valid 1x1 red PNG.

    Hand-built rather than encoded with Pillow so the bytes are fixed: vision
    tests assert a model reads "red" out of them, which a re-encode could change.

    Returns:
        The complete PNG file content.
    """

    def chunk(name: bytes, data: bytes) -> bytes:
        length = struct.pack(">I", len(data))
        crc = struct.pack(">I", zlib.crc32(name + data) & 0xFFFFFFFF)
        return length + name + data + crc

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )


def red_png_b64() -> str:
    """Return :func:`red_png` base64-encoded, as the image APIs carry it."""
    return b64encode(red_png()).decode()


def decoded_png(b64_json: str | None) -> bytes:
    """Decode a base64 image payload and assert it carries a PNG signature.

    Args:
        b64_json: The ``b64_json`` field of an image response.

    Returns:
        The decoded image bytes.
    """
    assert b64_json is not None, "response carries no b64_json payload"
    data = b64decode(b64_json)
    assert data.startswith(b"\x89PNG\r\n\x1a\n"), "payload is not a PNG"
    return data


def ollama_route(suffix: str) -> str:
    """Build an Ollama route path from the live ``ollama_routes_prefix`` setting.

    Args:
        suffix: The route's fixed part, e.g. ``"/api/chat"``.

    Returns:
        *suffix* prefixed with the configured Ollama routes prefix.
    """
    from stdapi.config import SETTINGS  # noqa: PLC0415

    return f"{SETTINGS.ollama_routes_prefix}{suffix}"


def strip_code_fence(text: str) -> str:
    """Strip a wrapping Markdown code fence (e.g. ` ```json `) from model output.

    Args:
        text: Raw model output, possibly fenced.

    Returns:
        ``text`` with a leading/trailing triple-backtick fence removed, or
        ``text`` stripped of surrounding whitespace when it is not fenced.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    lines = lines[1:-1] if len(lines) > 1 and lines[-1].strip() == "```" else lines[1:]
    return "\n".join(lines).strip()


def make_event_log(**overrides: Any) -> EventLog:  # noqa: ANN401
    """Build a minimal ``request`` EventLog for priming ``REQUEST_LOG``.

    Args:
        **overrides: Fields replacing the canned defaults.

    Returns:
        An EventLog carrying the five always-required fields plus *overrides*.
    """
    log: dict[str, Any] = {
        "type": "request",
        "level": "info",
        "date": datetime.now(UTC),
        "server_id": "test",
        "server_version": "0.0.0",
    }
    log.update(overrides)
    return log  # type: ignore[return-value]


def make_client_error(
    code: str,
    operation: str = "SomeOperation",
    *,
    message: str | None = None,
    status: int | None = None,
) -> ClientError:
    """Build a botocore ClientError for *code*.

    Args:
        code: The AWS error code, e.g. ``ThrottlingException``.
        operation: The API operation the error is attributed to.
        message: Error message; defaults to *code*.
        status: HTTP status to report, when the code under test reads it.

    Returns:
        The corresponding ClientError.
    """
    response: Any = {"Error": {"Code": code, "Message": message or code}}
    if status is not None:
        response["ResponseMetadata"] = {"HTTPStatusCode": status}
    return ClientError(response, operation)


def make_model_details(model_id: str, **overrides: Any) -> ModelDetails:  # noqa: ANN401
    """Build canned model details, so a test needs no live Bedrock catalog.

    Args:
        model_id: Value for both ``id`` and ``name`` unless overridden.
        **overrides: Fields replacing the canned defaults.

    Returns:
        The stub ModelDetails.
    """
    from stdapi.models import ModelDetails as _ModelDetails  # noqa: PLC0415

    fields: dict[str, Any] = {
        "id": model_id,
        "name": model_id,
        "provider": "Vendor",
        "input_modalities": ["TEXT"],
        "output_modalities": ["TEXT"],
        "regions": ["us-east-1"],
    }
    fields.update(overrides)
    return _ModelDetails(**fields)


def assert_embedding_list(
    response: CreateEmbeddingResponse,
    *,
    count: int,
    min_dimensions: int | None = None,
    dimensions: int | None = None,
    uniform_width: bool = True,
    nonzero: bool = True,
    normalized: bool = False,
) -> list[Sequence[float]]:
    """Assert the shape of a float-format embeddings response.

    Args:
        response: The embeddings response.
        count: Expected number of vectors, in request order.
        min_dimensions: Lower bound on each vector's width.
        dimensions: Exact width of each vector.
        uniform_width: Require every vector in a batch to share one width.
        nonzero: Require each vector to hold at least one non-zero component.
        normalized: Require each vector to be L2-normalized (within 5%).

    Returns:
        The vectors, in response order, for any further per-model assertions.
    """
    from math import hypot  # noqa: PLC0415

    assert response.object == "list"
    assert len(response.data) == count

    vectors: list[Sequence[float]] = []
    for index, item in enumerate(response.data):
        assert item.object == "embedding"
        assert item.index == index, "vectors are out of request order"
        assert isinstance(item.embedding, list)
        assert all(isinstance(value, float) for value in item.embedding)
        if dimensions is not None:
            assert len(item.embedding) == dimensions
        if min_dimensions is not None:
            assert len(item.embedding) >= min_dimensions
        if nonzero:
            assert any(value != 0.0 for value in item.embedding), "vector is all zeros"
        if normalized:
            assert hypot(*item.embedding) == pytest.approx(1.0, abs=0.05), (
                "vector is not L2-normalized"
            )
        vectors.append(item.embedding)

    if uniform_width and count > 1:
        assert len({len(vector) for vector in vectors}) == 1, (
            "batch returned vectors of different widths"
        )
    return vectors


class FakeCollectedItem:
    """A collected item exposing just the marker API ``pytest_collection_modifyitems`` uses."""

    def __init__(self, *markers: pytest.MarkDecorator) -> None:
        self.nodeid = "tests/test_x.py::test_y"
        self._marks = [marker.mark for marker in markers]
        self.fixturenames: tuple[str, ...] = ()
        self.added: list[Any] = []

    def get_closest_marker(self, name: str) -> Any | None:  # noqa: ANN401
        """Return the first of the item's own markers that *name* matches.

        Args:
            name: Marker name to look up.

        Returns:
            The matching mark, or None when the item carries none.
        """
        return next((mark for mark in self._marks if mark.name == name), None)

    def add_marker(self, marker: Any) -> None:  # noqa: ANN401
        """Record a marker the hook applied.

        Args:
            marker: Marker the hook added.
        """
        self.added.append(marker)


def batch_timeout_outcome(
    *, processed: bool, timeout: float, description: str
) -> NoReturn:
    """Skip a batch that never started; fail one that started but did not finish.

    The Bedrock batch queue, not the gateway, decides how long a job waits before
    it is picked up, so a bound expiring on a job that never left the queue is
    queue time, not a regression -- while one that expires after processing began
    is a real failure the timeout should still catch.

    Args:
        processed: Whether at least one request in the batch has been processed
            (succeeded, errored, canceled or expired), however the backend
            currently reports the batch's own status.
        timeout: Seconds the batch was given before this decision, for the message.
        description: The batch's current state, for the message.

    Raises:
        pytest.fail.Exception: If `processed` is True.
        pytest.skip.Exception: If `processed` is False.
    """
    if processed:
        pytest.fail(f"batch did not finish within {timeout:.0f}s: {description}")
    pytest.skip(
        f"queue: batch had not started processing after {timeout:.0f}s: {description}"
    )


def poll_batch_until_ended[T](
    retrieve: Callable[[], T],
    *,
    ended: Callable[[T], bool],
    processed: Callable[[T], bool],
    describe: Callable[[T], str],
    timeout: float,
    poll_interval: float,
) -> T:
    """Poll a batch until it ends, skipping instead of failing one that never started.

    Args:
        retrieve: Reads the batch's current state.
        ended: Whether the batch has reached a terminal state.
        processed: Whether at least one request has been processed; read only at
            the deadline, to choose between failing and skipping.
        describe: Renders the current state for a skip or failure message.
        timeout: Seconds allowed before giving up.
        poll_interval: Seconds between two reads.

    Returns:
        The batch, once ``ended`` is true of it.

    Raises:
        pytest.fail.Exception: If the deadline passes after processing started.
        pytest.skip.Exception: If the deadline passes before processing started.
    """
    deadline = monotonic() + timeout
    while not ended(current := retrieve()):
        if monotonic() >= deadline:
            batch_timeout_outcome(
                processed=processed(current),
                timeout=timeout,
                description=describe(current),
            )
        sleep(poll_interval)
    return current
