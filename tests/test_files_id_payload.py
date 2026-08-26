"""Unit tests for the Files API ID payload encoding (no AWS calls).

A payload is ``base32hex(uuid7_bytes + crc32(bucket))``: the UUIDv7 prefix keeps
IDs sorted by creation time and the CRC32 suffix names the bucket, which is what
lets ``/v1/files`` page S3 keys directly and resolve a bucket without any lookup
table.  The base32hex alphabet (``0-9a-v``) is required for that ordering —
standard base32 maps the six highest values to ``2-7``, which sort below ``a-z``.

Ref: https://stdapi.ai/api_openai_files/
     stdapi/files/_core.py:encode_id_payload
     stdapi/files/_core.py:decode_id_payload
"""

from base64 import b32encode
from binascii import crc32
from re import compile as re_compile
from uuid import UUID, uuid7

import pytest

from stdapi.files import _core
from stdapi.files._core import decode_id_payload, encode_id_payload, payload_created_at
from stdapi.types import FILE_ID_PATTERN, UPLOAD_ID_PATTERN
from stdapi.utils import now_utc_timestamp

pytestmark = pytest.mark.local

#: Bucket name the payloads under test are fingerprinted with.
_BUCKET = "test-bucket"


def _legacy_payload(bucket: str, uuid_bytes: bytes | None = None) -> str:
    """Return a payload in the pre-base32hex (standard base32) alphabet.

    Ref: stdapi/files/_core.py:decode_id_payload
    """
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
    """Lexicographic payload order matches creation order, which listing relies on.

    ``list_files`` pages raw S3 keys and never sorts by timestamp, so ascending
    key order must equal ascending creation order.

    Ref: stdapi/files/_core.py:encode_id_payload
         stdapi/files/_core.py:list_files
    """
    payloads = [encode_id_payload(_BUCKET) for _ in range(500)]

    assert payloads == sorted(payloads)
    assert len(set(payloads)) == len(payloads), "payloads must be unique"
    assert all(len(payload) == 32 for payload in payloads)


def test_legacy_base32_payloads_do_not_sort_in_creation_order() -> None:
    """The previous alphabet inverted the order this test suite now guards against.

    Standard base32 encodes the six highest 5-bit values as ``2-7``, which sort
    below ``a-z``, so 500 sequential payloads are provably not sorted.

    Ref: stdapi/files/_core.py:encode_id_payload
    """
    payloads = [_legacy_payload(_BUCKET) for _ in range(500)]

    assert payloads != sorted(payloads)


def test_payload_created_at_decodes_the_second_the_payload_was_minted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A payload decodes back to the second it encodes.

    Anything listed by payload order — files, batches — reports the creation
    time read from the payload rather than from a clock consulted later, so
    the order a client pages in and the times it is given agree only while
    this decode is exact.

    Ref: stdapi/files/_core.py:payload_created_at
    """
    before = now_utc_timestamp()
    assert (
        before <= payload_created_at(encode_id_payload(_BUCKET)) <= now_utc_timestamp()
    )

    second = 1_700_000_060
    monkeypatch.setattr(
        _core,
        "uuid7",
        lambda: UUID(bytes=(second * 1000).to_bytes(6, "big") + bytes(10)),
    )
    assert payload_created_at(encode_id_payload(_BUCKET)) == second


def test_payload_round_trips_to_its_uuid_and_bucket_fingerprint() -> None:
    """A generated payload decodes back to its UUIDv7 and bucket CRC32.

    Ref: stdapi/files/_core.py:decode_id_payload
         stdapi/files/_core.py:resolve_file_bucket
    """
    payload = encode_id_payload(_BUCKET)
    decoded = decode_id_payload(payload)

    assert len(decoded) == 20
    assert int.from_bytes(decoded[16:], "big") == crc32(_BUCKET.encode())
    assert _core.resolve_file_bucket(payload) == _BUCKET


def test_legacy_payload_still_decodes_to_its_bucket_fingerprint() -> None:
    """IDs issued before the alphabet change keep resolving to their bucket.

    The two alphabets overlap, so decoding tries both and keeps the candidate
    whose trailing CRC32 names a configured bucket.

    Ref: stdapi/files/_core.py:decode_id_payload
    """
    uuid_bytes = uuid7().bytes
    payload = _legacy_payload(_BUCKET, uuid_bytes)

    decoded = decode_id_payload(payload)

    assert decoded[:16] == uuid_bytes
    assert int.from_bytes(decoded[16:], "big") == crc32(_BUCKET.encode())
    assert _core.resolve_file_bucket(payload) == _BUCKET


@pytest.mark.parametrize("pattern", [FILE_ID_PATTERN, UPLOAD_ID_PATTERN])
def test_id_patterns_accept_both_alphabets(pattern: str) -> None:
    """Both the current and the legacy payload alphabets pass ID validation.

    The route-level patterns gate every ``file-``/``upload_`` identifier, so an
    alphabet they reject would turn previously issued IDs into 400s.

    Ref: stdapi/types/__init__.py:FILE_ID_PATTERN
         stdapi/types/__init__.py:UPLOAD_ID_PATTERN
    """
    match = re_compile(pattern).match
    prefix = "file-" if "file" in pattern else "upload_"

    assert match(f"{prefix}{encode_id_payload(_BUCKET)}")
    assert match(f"{prefix}{_legacy_payload(_BUCKET)}")
    assert match(f"{prefix}{encode_id_payload(_BUCKET)[:31]}") is None, (
        "a truncated payload must not validate"
    )
