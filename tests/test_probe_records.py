"""The recorded model probes stay readable and stay in sync with the prober.

A probe record is evidence: it is what a model-specific branch in
``stdapi/models/chat/`` is justified by. A record written by an older probe set,
or naming an outcome the prober no longer produces, is evidence for nothing, so
the shape is asserted offline on every run.

Ref: tests/probes/README.md
     tests/probes/probe_model.py:probe_model
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, get_args

import pytest

from tests.probes.probe_model import (
    MANTLE_PROBES,
    PROBES,
    RESULTS_DIR,
    SCHEMA_VERSION,
    STREAM_PROBES,
    Outcome,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.local

#: Every probe record on disk, or an empty list before the first probe run.
_RECORDS = sorted(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.is_dir() else []

#: Keys every record carries, whatever the model.
_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "model_id",
        "invoked_id",
        "transport",
        "region",
        "probed_at",
        "probes",
    }
)


@pytest.mark.parametrize("path", _RECORDS, ids=lambda p: p.stem)
class TestProbeRecords:
    """Each committed record parses and matches the current probe set."""

    def test_record_has_the_current_shape(self, path: Path) -> None:
        """A record carries every field, at the schema version in force."""
        record = json.loads(path.read_text())

        assert record.keys() >= _REQUIRED_KEYS, (
            f"{path.name} is missing {sorted(_REQUIRED_KEYS - record.keys())}"
        )
        assert record["schema_version"] == SCHEMA_VERSION, (
            f"{path.name} was written by probe schema v{record['schema_version']}; "
            f"re-probe the model or bump nothing until it is refreshed"
        )
        assert record["probes"], f"{path.name} records no probe at all"

    def test_every_probe_names_a_known_outcome(self, path: Path) -> None:
        """No record claims an outcome the prober cannot produce."""
        record = json.loads(path.read_text())
        outcomes = get_args(Outcome)

        for probe in record["probes"]:
            assert probe["outcome"] in outcomes, (
                f"{path.name}: {probe['name']} has unknown outcome {probe['outcome']!r}"
            )
            assert probe["detail"], (
                f"{path.name}: {probe['name']} records no detail, so the outcome "
                f"cannot be checked against the implementation"
            )

    def test_a_reached_model_ran_the_whole_probe_set(self, path: Path) -> None:
        """A record whose baseline succeeded covers every probe in the set.

        A short record means the run died partway and the gaps would read as
        "not probed" rather than as "unknown".
        """
        record = json.loads(path.read_text())
        names = [probe["name"] for probe in record["probes"]]
        if names[:1] != ["baseline"] or record["probes"][0]["outcome"] not in {
            "supported",
            "accepted",
        }:
            pytest.skip(f"{path.name}: the model was never reached")

        expected = (
            [probe.name for probe in MANTLE_PROBES]
            if record["transport"] == "mantle"
            else [probe.name for probe in (*PROBES, *STREAM_PROBES)]
        )
        assert names[1:] == expected, (
            f"{path.name} does not match the current probe set; re-run "
            f"`uv run python -m tests.probes.probe_model {record['model_id']}`"
        )
