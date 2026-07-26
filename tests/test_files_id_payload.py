"""Unit tests for Files API payload encoding (no AWS calls)."""

from base64 import b32encode
from binascii import crc32
from re import compile as re_compile
from uuid import uuid7

import pytest

from stdapi.files import _core
from stdapi.files._core import decode_id_payload, encode_id_payload
from stdapi.types import FILE_ID_PATTERN, UPLOAD_ID_PATTERN

pytestmark = pytest.mark.local

#: Bucket name the payloads under test are fingerprinted with.
_BUCKET = "test-bucket"


def _legacy_payload(bucket: str, uuid_bytes: bytes | None = None) -> str:
    """Return a payload in the pre-base32hex alphabet."""
    return (
        b32encode(
            (uuid_bytes or uuid7().bytes) + crc32(bucket.encode()).to_bytes(4, "big")
        )
        .lower()
        .decode()
    )


@pytest.fixture(autouse=True)
def _known_bucket(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the test bucket so a payload fingerprint resolves."""
    monkeypatch.setattr(_core, "_BUCKET_CRC32", {crc32(_BUCKET.encode()): _BUCKET})


def test_payloads_sort_in_creation_order() -> None:
    """Lexicographic payload order matches creation order, which listing relies on."""
    payloads = [encode_id_payload(_BUCKET) for _ in range(500)]

    assert payloads == sorted(payloads)


def test_legacy_base32_payloads_do_not_sort_in_creation_order() -> None:
    """The previous alphabet inverted the order this test suite now guards against."""
    payloads = [_legacy_payload(_BUCKET) for _ in range(500)]

    assert payloads != sorted(payloads)


def test_payload_round_trips_to_its_uuid_and_bucket_fingerprint() -> None:
    """A generated payload decodes back to its UUIDv7 and bucket CRC32."""
    decoded = decode_id_payload(encode_id_payload(_BUCKET))

    assert len(decoded) == 20
    assert int.from_bytes(decoded[16:], "big") == crc32(_BUCKET.encode())


def test_legacy_payload_still_decodes_to_its_bucket_fingerprint() -> None:
    """IDs issued before the alphabet change keep resolving to their bucket."""
    uuid_bytes = uuid7().bytes

    decoded = decode_id_payload(_legacy_payload(_BUCKET, uuid_bytes))

    assert decoded[:16] == uuid_bytes
    assert int.from_bytes(decoded[16:], "big") == crc32(_BUCKET.encode())


@pytest.mark.parametrize("pattern", [FILE_ID_PATTERN, UPLOAD_ID_PATTERN])
def test_id_patterns_accept_both_alphabets(pattern: str) -> None:
    """Both the current and the legacy payload alphabets pass ID validation."""
    match = re_compile(pattern).match
    prefix = "file-" if "file" in pattern else "upload_"

    assert match(f"{prefix}{encode_id_payload(_BUCKET)}")
    assert match(f"{prefix}{_legacy_payload(_BUCKET)}")
