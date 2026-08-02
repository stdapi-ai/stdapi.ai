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
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import ClientError
from pybase64 import b64decode, b64encode

if TYPE_CHECKING:
    from collections.abc import Sequence

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
