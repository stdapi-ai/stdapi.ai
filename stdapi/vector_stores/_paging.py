"""Cutting a listing into pages of the quantity it reports.

A listing orders on the same value it answers with, so the order holds across
pages rather than only within one. An identifier is a different quantity: an
uploaded file's identifier is minted when the file is uploaded and not when it
is attached here, and one naming a resource held elsewhere carries no time at
all — both sort, and neither sorts by the ``created_at`` the page reports. The
one identifier that may stand in for it is a store's own, minted from the same
instant the store records as its creation.

The cursors are positions in that order rather than values compared against it:
a cursor naming a record the listing no longer holds ends the walk instead of
restarting it from the top.
"""

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


class TimedRecord(Protocol):
    """A record a listing reports an identifier and a creation time for."""

    id: str
    created_at: int


def page_identifiers(
    ordered: Sequence[str], *, after: str, before: str, limit: int, order: str
) -> tuple[list[str], bool]:
    """Return one page of *ordered*, which is already in creation order.

    Args:
        ordered: Every identifier the listing holds, oldest first.
        after: Return the identifiers following this one, or ``""``.
        before: Return the page ending immediately before this one, or ``""``.
        limit: Maximum identifiers to return.
        order: ``"asc"`` or ``"desc"``.

    Returns:
        ``(identifiers, has_more)``.
    """
    entries = list(reversed(ordered)) if order == "desc" else list(ordered)
    if after:
        index = entries.index(after) + 1 if after in entries else len(entries)
        entries = entries[index:]
    if not before:
        return entries[:limit], len(entries) > limit
    entries = entries[: entries.index(before)] if before in entries else []
    # The page ends at the cursor, so it is filled from the far end.
    return entries[-limit:], len(entries) > limit


def page_records[RecordT: TimedRecord](
    records: Iterable[RecordT], *, after: str, before: str, limit: int, order: str
) -> tuple[list[RecordT], bool]:
    """Return one page of *records*, ordered by the creation time they report.

    Args:
        records: Every record the listing holds, in any order.
        after: Return the records following this identifier, or ``""``.
        before: Return the page ending immediately before this identifier, or ``""``.
        limit: Maximum records to return.
        order: ``"asc"`` or ``"desc"``.

    Returns:
        ``(records, has_more)``, ordered by ``created_at`` and, for the records
        sharing a second, by identifier.
    """
    ordered = sorted(records, key=lambda record: (record.created_at, record.id))
    by_id = {record.id: record for record in ordered}
    page, has_more = page_identifiers(
        [record.id for record in ordered],
        after=after,
        before=before,
        limit=limit,
        order=order,
    )
    return [by_id[identifier] for identifier in page], has_more
