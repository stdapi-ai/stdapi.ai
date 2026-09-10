"""Vector store bookkeeping listings, against an in-memory object store (unit).

S3 lists keys in ascending order only and is free to answer with fewer keys
than ``MaxKeys`` asked for, so a newest-first listing built on one listing call
answers the oldest records once the bucket outgrows a page. The fake here
honours both of those rules, which is what makes the paging testable without a
bucket holding a thousand stores.

Ref: https://docs.aws.amazon.com/AmazonS3/latest/API/API_ListObjectsV2.html
     stdapi/vector_stores/records.py:list_stores
"""

from typing import Any

import pytest

from stdapi.vector_stores import records
from stdapi.vector_stores.models import StoreRecord

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Key prefix every store record is stored under.
_PREFIX = "vector_stores/"

#: Identifiers the fake bucket is seeded with, oldest first.
_STORE_IDS = [f"vs_{index:04d}" for index in range(25)]


class _PagingObjectStore:
    """Object store answering listings one short page at a time."""

    def __init__(self, keys: list[str], page_size: int) -> None:
        self.keys = sorted(keys)
        self.page_size = page_size
        self.pages = 0

    async def list_objects_v2(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        """List one page of common prefixes, ascending, never past *page_size*."""
        self.pages += 1
        prefix = params["Prefix"]
        delimiter = params["Delimiter"]
        start = params.get("ContinuationToken") or ""
        entries = sorted(
            {
                prefix + key[len(prefix) :].split(delimiter, 1)[0] + delimiter
                for key in self.keys
                if key.startswith(prefix) and delimiter in key[len(prefix) :]
            }
        )
        entries = [entry for entry in entries if entry > start]
        page = entries[: min(self.page_size, params["MaxKeys"])]
        return {
            "CommonPrefixes": [{"Prefix": entry} for entry in page],
            "IsTruncated": len(entries) > len(page),
            "NextContinuationToken": page[-1] if page else "",
        }

    async def get_object(self, **params: Any) -> dict[str, Any]:  # noqa: ANN401
        """Answer the store record the key names."""
        store_id = params["Key"][len(_PREFIX) :].split("/", 1)[0]
        record = StoreRecord(
            id=store_id,
            created_at=int(store_id.removeprefix("vs_")),
            last_active_at=0,
            embedding_model="amazon.titan-embed-text-v2:0",
            dimensions=256,
        )
        return {"Body": _Body(record.model_dump_json().encode()), "ETag": '"etag"'}


class _Body:
    """The streaming body an object read answers with."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self) -> bytes:
        """Return the whole object body."""
        return self._data


@pytest.fixture
def paged_store(monkeypatch: pytest.MonkeyPatch) -> _PagingObjectStore:
    """Bind the record helpers to a bucket answering ten prefixes per page."""
    store = _PagingObjectStore(
        [f"{_PREFIX}{store_id}/store.json" for store_id in _STORE_IDS], page_size=10
    )
    monkeypatch.setattr(records, "records_bucket", lambda: "bucket")
    monkeypatch.setattr(records, "records_client", lambda: store)
    monkeypatch.setattr(records.SETTINGS, "aws_s3_vector_stores_prefix", _PREFIX)
    return store


class TestListStoresOrdering:
    """A newest-first listing answers the newest stores, at any bucket size.

    Ref: https://platform.openai.com/docs/api-reference/vector-stores/list
         stdapi/vector_stores/records.py:list_stores
    """

    async def test_descending_listing_starts_at_the_newest_store(
        self, paged_store: _PagingObjectStore
    ) -> None:
        """The default page is the newest stores, not the oldest ones reversed.

        Reading a single listing page answers a bucket larger than that page
        with its oldest identifiers: reversing those still never reaches the
        stores a caller listing newest-first is asking for.
        """
        found, has_more = await records.list_stores(
            after="", before="", limit=5, order="desc"
        )

        assert [record.id for record in found] == _STORE_IDS[::-1][:5]
        assert has_more is True
        assert paged_store.pages == 3, "the listing must page to the newest key"

    async def test_descending_cursor_pages_through_the_whole_bucket(
        self, paged_store: _PagingObjectStore
    ) -> None:
        """Paging on the ``after`` cursor walks every store, ending on has_more."""
        seen: list[str] = []
        cursor = ""
        while True:
            found, has_more = await records.list_stores(
                after=cursor, before="", limit=10, order="desc"
            )
            seen.extend(record.id for record in found)
            if not has_more:
                break
            cursor = seen[-1]

        assert seen == _STORE_IDS[::-1]

    async def test_ascending_listing_still_starts_at_the_oldest_store(
        self, paged_store: _PagingObjectStore
    ) -> None:
        """Ascending order is served by the cursor listing, one page, oldest first."""
        found, has_more = await records.list_stores(
            after="", before="", limit=5, order="asc"
        )

        assert [record.id for record in found] == _STORE_IDS[:5]
        assert has_more is True
        assert paged_store.pages == 1
