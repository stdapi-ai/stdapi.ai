"""MTEB coverage for embedding and reranking models, dedicated to the public domain.

MTEB publishes per-task result files, not a stored aggregate: the leaderboard
computes its means at render time from benchmark definitions that exist only as
Python source. Averaging whatever tasks a model happens to have been run on
would produce a number that looks comparable across models and is not — models
here range from twenty-eight domain tasks to the full English suite. So this
source contributes coverage and a link to the results, never a score.
"""

from __future__ import annotations

from typing import Any

from docs_gen.model_catalog.http import get_json
from docs_gen.model_catalog.sources import RawReference, SourceResult, snapshot

#: Root tree of the CC0 results repository behind the MTEB leaderboard.
#:
#: The Contents API caps at 1000 entries and truncates in silence; the Trees API
#: says when it has. A recursive listing from the root hits that cap partway
#: through the repository's test fixtures, so this is read non-recursively and
#: only the ``results`` subtree is then listed, which stays well under it.
_ROOT_TREE_URL: str = (
    "https://api.github.com/repos/embeddings-benchmark/results/git/trees/main"
)

#: One tree object by its SHA, for listing ``results`` without recursing.
_TREE_URL: str = (
    "https://api.github.com/repos/embeddings-benchmark/results/git/trees/{sha}"
)

#: Human-readable view of one model's results.
_MODEL_URL: str = (
    "https://github.com/embeddings-benchmark/results/tree/main/results/{directory}"
)

#: Directory-name fragments worth matching against this catalogue's models.
_RELEVANT: tuple[str, ...] = (
    "bedrock__",
    "cohere__",
    "amazon__",
    "titan",
    "twelvelabs__",
)


def _collect() -> list[dict[str, Any]]:
    """List the model directories the results repository publishes.

    Returns:
        The directory entries, filtered to the publishers this gateway serves.

    Raises:
        RuntimeError: GitHub returned no usable tree (e.g. it rate-limited the
            request), or truncated the ``results`` directory listing.
    """
    root = get_json(_ROOT_TREE_URL)
    if "tree" not in root:
        msg = f"GitHub returned no tree for the results repository: {root}"
        raise RuntimeError(msg)
    results_dir = next(
        (
            entry
            for entry in root["tree"]
            if entry.get("type") == "tree" and entry.get("path") == "results"
        ),
        None,
    )
    if results_dir is None:
        msg = "the results repository has no top-level 'results' directory"
        raise RuntimeError(msg)
    listing = get_json(_TREE_URL.format(sha=results_dir["sha"]))
    if listing.get("truncated"):
        msg = "the MTEB results directory listing was truncated by GitHub"
        raise RuntimeError(msg)
    return [
        {"name": f"results/{entry['path']}"}
        for entry in listing.get("tree", ())
        if entry.get("type") == "tree"
        and any(part in str(entry.get("path", "")).lower() for part in _RELEVANT)
    ]


def fetch(*, refresh: bool = False) -> SourceResult:
    """Read which models MTEB has results for.

    Args:
        refresh: Ignore any cached snapshot.

    Returns:
        One reference per evaluated model, with no score attached.
    """
    raw = snapshot("mteb", _collect, refresh=refresh)
    assert isinstance(raw, list)  # noqa: S101 -- snapshot round-trips its own JSON
    references: list[RawReference] = []
    for entry in raw:
        directory = str(entry["name"]).removeprefix("results/")
        organization, _, model = directory.partition("__")
        references.append(
            RawReference(
                source="mteb",
                name=model.replace("_", " ") or directory,
                organization=organization,
                label="MTEB results",
                detail="Per-task results; MTEB publishes no stored aggregate.",
                url=_MODEL_URL.format(directory=directory),
            )
        )
    notes = [] if references else ["no results directory matched a served publisher"]
    return SourceResult(
        key="mteb", as_of="", scores=[], references=references, notes=notes
    )
