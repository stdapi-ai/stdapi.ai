"""Command line entry point of the model catalogue generator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from docs_gen.model_catalog import build as build_module
from docs_gen.model_catalog import page
from docs_gen.model_catalog.config import DATA_DIR, DEFAULT_MATCH_MODEL, UNMATCHED_PATH
from docs_gen.model_catalog.gateway import Gateway
from docs_gen.model_catalog.matching import Matcher

if TYPE_CHECKING:
    from docs_gen.model_catalog.enrichment import Applied

#: Environment variable holding the bearer token, so it stays out of ``ps``.
_API_KEY_ENV: str = "STDAPI_API_KEY"

#: Default instance the generator reads the catalogue from.
_DEFAULT_GATEWAY: str = "http://127.0.0.1:8000"


def _parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="python -m docs_gen.model_catalog",
        description="Regenerate the data set behind the public Models page.",
    )
    parser.add_argument(
        "--gateway", default=_DEFAULT_GATEWAY, help="stdapi.ai instance to read"
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get(_API_KEY_ENV),
        help=f"bearer token for that instance; defaults to ${_API_KEY_ENV}",
    )
    parser.add_argument(
        "--match-model",
        default=DEFAULT_MATCH_MODEL,
        help="model used to resolve leaderboard names the rules cannot settle",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="rules and overrides only; leave every unresolved name unmatched",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="ignore cached leaderboard snapshots"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="replace the published data set instead of updating it, dropping "
        "retired models and any value AWS has stopped returning",
    )
    parser.add_argument(
        "--accept-retirements",
        action="store_true",
        help="publish a run that retires more models than the safety ceiling "
        "allows, keeping the history --fresh would discard",
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        choices=sorted(build_module.COLLECTORS),
        help="restrict collection to this source; repeatable",
    )
    parser.add_argument(
        "--generated",
        default=datetime.now(tz=UTC).date().isoformat(),
        help="date stamp recorded in the manifest",
    )
    return parser


def _report_enrichment(overlay: Applied) -> None:
    """Print what the hand-curated overlay contributed to this run.

    Args:
        overlay: What applying the overlay produced.
    """
    if not (overlay.filled or overlay.unknown):
        return
    print(
        f"enrichment        {sum(overlay.filled.values())} field(s) filled by hand, "
        f"{sum(overlay.refined.values())} refined, {overlay.skipped} already known"
    )
    if overlay.filled:
        print(f"  filled          {json.dumps(overlay.filled)}")
    if overlay.refined:
        print(f"  refined         {json.dumps(overlay.refined)}")
    if overlay.disputed:
        # A curator who read the vendor page and disagrees with the machine
        # source is the most valuable signal this file produces.
        print(f"  disputed        {len(overlay.disputed)} cited value(s) overruled")
        for entry in overlay.disputed:
            print(f"    {entry}", file=sys.stderr)
    if overlay.unknown:
        print(f"  unknown entries {', '.join(overlay.unknown[:8])}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    """Regenerate the published data set.

    Args:
        argv: Command line arguments; ``sys.argv`` when omitted.

    Returns:
        Process exit status.
    """
    args = _parser().parse_args(argv)
    instance = Gateway(args.gateway, args.api_key)
    matcher = Matcher(
        gateway_url=None if args.no_llm else args.gateway,
        api_key=args.api_key,
        model=args.match_model,
    )

    catalog, report, price_cards, collected, facts = build_module.build(
        instance,
        generated=args.generated,
        refresh=args.refresh,
        sources=args.sources,
        matcher=matcher,
        reuse=not args.fresh,
        accept_retirements=args.accept_retirements,
    )
    build_module.write(catalog, price_cards, facts, report)
    unmatched = build_module.write_unmatched(catalog, collected)
    page.write(catalog)

    print(f"models            {report.models}")
    print(f"detail documents  {report.price_documents}")
    print(f"scores attached   {report.matched}")
    print(f"llm calls         {report.llm_calls}")
    _report_enrichment(report.enrichment)
    print(f"cited by hand     {report.citations} published value(s)")
    merged = report.merge
    print(
        f"merge             +{len(merged.added)} new, {len(merged.retired)} newly "
        f"retired, {len(merged.returned)} returned, "
        f"{sum(merged.carried.values())} values carried forward"
    )
    if merged.added:
        print(f"  new             {', '.join(merged.added[:8])}")
    if merged.retired:
        print(f"  retired         {', '.join(merged.retired[:8])}")
    if merged.carried:
        print(f"  carried         {json.dumps(merged.carried)}")
    print(f"index (gzipped)   {report.index_bytes / 1024:.1f} KiB")
    print(f"written to        {DATA_DIR}")
    for source in catalog.manifest.sources:
        print(
            f"  {source.key:<10} {source.matched:>4} matched of {source.rows:>5} rows  {source.as_of}"
        )
    unscored = len(unmatched["models_without_a_score"])
    print(f"models without a score  {unscored}  (see {UNMATCHED_PATH})")
    for note in report.notes:
        print(f"note: {note}", file=sys.stderr)
    if catalog.manifest.unreachable_regions:
        print(
            f"note: {len(catalog.manifest.unreachable_regions)} region(s) unreachable: "
            f"{json.dumps(catalog.manifest.unreachable_regions)[:300]}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
