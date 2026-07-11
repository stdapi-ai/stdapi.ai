"""Unit tests for input file handling (:mod:`stdapi.input_file`)."""

from __future__ import annotations

import pytest
from pybase64 import b64encode

from stdapi.api_errors import ApiError
from stdapi.config import SETTINGS
from stdapi.input_file import InputFile


async def test_to_bytes_rejects_oversized_inline_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inline base64 input larger than the limit is rejected with HTTP 413."""
    monkeypatch.setattr(SETTINGS, "max_input_file_size", 8)
    payload = b64encode(b"x" * 64).decode()
    with pytest.raises(ApiError) as exc:
        await InputFile(payload).to_bytes()
    assert exc.value.status == 413


async def test_to_bytes_allows_input_within_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An inline input within the configured limit is returned unchanged."""
    monkeypatch.setattr(SETTINGS, "max_input_file_size", 1024)
    data = b"x" * 64
    assert await InputFile(b64encode(data).decode()).to_bytes() == data


async def test_to_bytes_unlimited_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the limit disabled (0), inputs of any size are accepted."""
    monkeypatch.setattr(SETTINGS, "max_input_file_size", 0)
    data = b"y" * 4096
    assert await InputFile(b64encode(data).decode()).to_bytes() == data


def test_max_concurrent_input_downloads_default() -> None:
    """The per-request input-download concurrency limit defaults to 8."""
    assert SETTINGS.max_concurrent_input_downloads == 8


async def test_to_base64_rejects_oversized_inline_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oversized inline base64 input is rejected via the to_base64 path (413)."""
    monkeypatch.setattr(SETTINGS, "max_input_file_size", 8)
    payload = b64encode(b"x" * 64).decode()
    with pytest.raises(ApiError) as exc:
        await InputFile(payload).to_base64()
    assert exc.value.status == 413


async def test_to_data_uri_rejects_oversized_inline_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oversized inline data URI is rejected via the to_data_uri path (413)."""
    monkeypatch.setattr(SETTINGS, "max_input_file_size", 8)
    data_uri = f"data:image/png;base64,{b64encode(b'x' * 64).decode()}"
    with pytest.raises(ApiError) as exc:
        await InputFile(data_uri).to_data_uri()
    assert exc.value.status == 413
