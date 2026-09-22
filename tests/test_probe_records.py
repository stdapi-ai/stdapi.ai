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
from typing import TYPE_CHECKING, Any, get_args

import pytest

from tests.probes.probe_model import (
    MANTLE_PROBES,
    PROBES,
    RESULTS_DIR,
    SCHEMA_VERSION,
    STREAM_PROBES,
    Outcome,
    Probe,
    _jsonable,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.local

#: Every probe record on disk, or an empty list before the first probe run.
_RECORDS = sorted(RESULTS_DIR.glob("*.json")) if RESULTS_DIR.is_dir() else []

#: Oldest probe schema whose records still match the prober's record shape.
_OLDEST_SCHEMA_VERSION = 5

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
        """A record carries every field, at a schema version the prober has had.

        A record older than the current version stays evidence for the probes it
        ran; the probes added since are the ones it lacks.
        """
        record = json.loads(path.read_text())

        assert record.keys() >= _REQUIRED_KEYS, (
            f"{path.name} is missing {sorted(_REQUIRED_KEYS - record.keys())}"
        )
        assert _OLDEST_SCHEMA_VERSION <= record["schema_version"] <= SCHEMA_VERSION, (
            f"{path.name} was written by probe schema v{record['schema_version']}; "
            f"re-probe the model"
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
        """A record whose baseline succeeded covers every probe of its schema version.

        A short record means the run died partway and the gaps would read as
        "not probed" rather than as "unknown". Probes added after the record's
        version are the only ones it may lack.
        """
        record = json.loads(path.read_text())
        names = [probe["name"] for probe in record["probes"]]
        if names[:1] != ["baseline"] or record["probes"][0]["outcome"] not in {
            "supported",
            "accepted",
        }:
            pytest.skip(f"{path.name}: the model was never reached")

        probes = (
            MANTLE_PROBES
            if record["transport"] == "mantle"
            else (*PROBES, *STREAM_PROBES)
        )
        expected = [
            probe.name for probe in probes if probe.added_in <= record["schema_version"]
        ]
        assert names[1:] == expected, (
            f"{path.name} does not match the current probe set; re-run "
            f"`uv run python -m tests.probes.probe_model {record['model_id']}`"
        )

    def test_a_current_result_was_measured_with_the_current_request(
        self, path: Path
    ) -> None:
        """A probe's recorded request is today's, unless its ``changed_in`` is newer.

        This is what makes ``changed_in`` track request changes: editing a
        probe's request without raising it fails every record probed since.
        """
        record = json.loads(path.read_text())
        probes = _probes_by_name(record)

        for entry in record["probes"]:
            probe = probes.get(entry["name"])
            if probe is None or probe.changed_in > record["schema_version"]:
                continue
            assert entry["request"] == _as_recorded(probe.overrides), (
                f"{path.name}: {entry['name']} was recorded with another request; "
                f"raise its changed_in to SCHEMA_VERSION and re-probe"
            )

    def test_an_outdated_request_is_not_read_as_an_inert_knob(self, path: Path) -> None:
        """An ``accepted`` measured before a probe's request changed is inconclusive.

        Before schema 6 an answer-judged probe had 64 output tokens, which a
        model reasoning by default can spend before answering; the record says
        nothing about whether the knob works until the model is re-probed.
        """
        record = json.loads(path.read_text())
        probes = _probes_by_name(record)
        outdated = [
            entry["name"]
            for entry in record["probes"]
            if (probe := probes.get(entry["name"])) is not None
            and probe.changed_in > record["schema_version"]
            and entry["outcome"] == "accepted"
        ]
        if outdated:
            pytest.skip(
                f"{path.name}: {outdated} read 'accepted' under a request changed "
                f"since schema v{record['schema_version']}; re-probe the model"
            )


def _probes_by_name(record: dict[str, Any]) -> dict[str, Probe]:
    """Return the current probes of *record*'s transport, keyed by name.

    Args:
        record: A committed probe record.

    Returns:
        The probes of the record's transport, keyed by probe name.
    """
    probes = (
        MANTLE_PROBES if record["transport"] == "mantle" else (*PROBES, *STREAM_PROBES)
    )
    return {probe.name: probe for probe in probes}


def _as_recorded(overrides: dict[str, Any]) -> object:
    """Return *overrides* as a record stores them, bytes replaced by placeholders.

    Args:
        overrides: A probe's request overrides.

    Returns:
        The overrides after the record's JSON round trip.
    """
    return json.loads(json.dumps(overrides, default=_jsonable))
