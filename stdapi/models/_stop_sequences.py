"""Content a Converse call returns, cut right after the first stop sequence its text holds.

Some models check stop sequences once per output chunk and cut the chunk after
its last match, returning content generated past the first one. Cutting it
again here keeps the documented contract: generation halts at the first
sequence matched, which stays at the end of the text as the backend returns it,
and nothing after it is returned. A ``tool_use`` stop reason left with no tool
call becomes ``end_turn``, so no client acts on a call the cut removed.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from types_aiobotocore_bedrock_runtime.type_defs import (
        ConverseResponseTypeDef,
        ConverseStreamOutputTypeDef,
    )


def _match_end(text: str, stop_sequences: Sequence[str]) -> int | None:
    """Return where the first stop sequence completed in *text* ends.

    Args:
        text: Generated text.
        stop_sequences: Sequences that halt generation.

    Returns:
        The index just past the match that completes first, or ``None`` when
        *text* holds none.
    """
    return min(
        (
            start + len(stop)
            for stop in stop_sequences
            if (start := text.find(stop)) >= 0
        ),
        default=None,
    )


def truncate_response_text(
    response: ConverseResponseTypeDef, stop_sequences: Sequence[str]
) -> None:
    """Cut a Converse response right after the first stop sequence its text holds, in place.

    The text block holding the match is cut after it and every later content
    block is removed.

    Args:
        response: Converse response, its union keys free of null values.
        stop_sequences: The request's ``stopSequences``.
    """
    content = response.get("output", {}).get("message", {}).get("content", [])
    for position, block in enumerate(content):
        if (
            "text" in block
            and (end := _match_end(block["text"], stop_sequences)) is not None
        ):
            block["text"] = block["text"][:end]
            del content[position + 1 :]
            if response.get("stopReason") == "tool_use" and not any(
                "toolUse" in kept for kept in content
            ):
                response["stopReason"] = "end_turn"
            return


class StreamCut:
    """Cuts a ConverseStream right after the first stop sequence its text holds.

    A match may span any number of text deltas of one content block, never two
    blocks. Once matched, the delta completing the match is cut after it, and
    every later content event is dropped except the matching block's
    ``contentBlockStop``; message and metadata events still pass.
    """

    __slots__ = (
        "_cut_block",
        "_stop_sequences",
        "_tail",
        "_tail_block",
        "_tail_length",
        "_tool_use_kept",
    )

    def __init__(self, stop_sequences: Sequence[str]) -> None:
        """Prepare the cut of one stream.

        Args:
            stop_sequences: The request's ``stopSequences``.
        """
        self._stop_sequences = stop_sequences
        # The tail holds the end of the block's text already passed, long
        # enough to complete a match whose end lands in a later delta.
        self._tail_length = max(map(len, stop_sequences)) - 1
        self._tail = ""
        self._tail_block: int | None = None
        self._cut_block: int | None = None
        self._tool_use_kept = False

    def keep(self, event: ConverseStreamOutputTypeDef) -> bool:
        """Return whether *event* reaches the client, cutting its text in place.

        Args:
            event: The next ConverseStream event.

        Returns:
            ``False`` for an event generated after the match.
        """
        if self._cut_block is not None:
            if block_stop := event.get("contentBlockStop"):
                return block_stop["contentBlockIndex"] == self._cut_block
            if "contentBlockStart" in event or "contentBlockDelta" in event:
                return False
            if (
                (message_stop := event.get("messageStop"))
                and message_stop["stopReason"] == "tool_use"
                and not self._tool_use_kept
            ):
                message_stop["stopReason"] = "end_turn"
            return True
        if (block_start := event.get("contentBlockStart")) and (
            "toolUse" in block_start["start"]
        ):
            self._tool_use_kept = True
        elif (block_delta := event.get("contentBlockDelta")) and "text" in (
            delta := block_delta["delta"]
        ):
            index = block_delta["contentBlockIndex"]
            if index != self._tail_block:
                self._tail, self._tail_block = "", index
            window = self._tail + delta["text"]
            if (end := _match_end(window, self._stop_sequences)) is not None:
                # No match ends inside the tail: it would have been cut earlier.
                delta["text"] = window[len(self._tail) : end]
                self._cut_block = index
            else:
                self._tail = window[max(len(window) - self._tail_length, 0) :]
        return True
