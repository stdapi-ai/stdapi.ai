"""Drift detection for the documentation pages' pinned browser assets.

``stdapi/docs_assets/__init__.py`` pins Swagger UI and ReDoc to an exact release
and records a SHA-256 per file. The pin is what keeps an operator's browser off
a floating major tag, and the digests are what keep an unverified blob out of a
redistributed image -- but a pin nobody ever bumps is the failure mode of
pinning, and nothing in a running gateway can notice that upstream moved on.

This lane reads the npm registry and says so. It never upgrades anything: a new
major of either library can change the page, so a person decides, and the pin is
the safety until they do.

Three sources of finding, and they are not interchangeable:

===========  =============================================  ================
Outcome      Meaning                                        Effect
===========  =============================================  ================
OUTDATED     npm publishes a newer release than the pin     **fails**
DIGEST       the publisher serves bytes the manifest         **fails**
             does not record, for the pinned version
LICENCE      npm declares another licence for the package   **fails**
UNREACHABLE  a source could not be fetched or parsed        reported only
CURRENT      the pin is the latest and the bytes match      none
===========  =============================================  ================

A DIGEST finding is the serious one: either the recorded hash is wrong, in which
case every image build already fails, or an immutable published file changed
under us. A LICENCE finding matters because these files are redistributed inside
a paid Marketplace image, so the notice that travels with them has to be the
right one.

The live check is opt-in (``--drift``), like the price detector it rides beside:
upstream cutting a release is not a regression in this repository and must never
turn an unrelated run red. It fails only inside its own lane, where a hard
failure is what opens the tracking issue. The classifier itself is exercised
offline, including against a deliberately stale pin, because a detector nobody
has seen fail is not known to work.

Ref: https://github.com/stdapi-ai/stdapi.ai/issues/185
     stdapi/docs_assets/__init__.py
     tests/test_pricing_drift.py
     https://github.com/swagger-api/swagger-ui/releases
     https://github.com/Redocly/redoc/releases
"""

import warnings
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Final

import httpx
import pytest

from stdapi.docs_assets import (
    BROWSER_ASSETS,
    LICENSE_ASSETS,
    REDOC_VERSION,
    SWAGGER_UI_VERSION,
    UPSTREAM_LICENSES,
    Asset,
    digest,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

#: Where npm publishes the manifest of a package's latest release.
_REGISTRY: Final[str] = "https://registry.npmjs.org/{package}/latest"

#: Seconds a source gets to answer; slower than this counts as unreachable.
_FETCH_TIMEOUT: Final[float] = 60.0

#: Identifies these requests to the registry and the CDN.
_USER_AGENT: Final[str] = "stdapi.ai-tests docs-asset drift detector"

#: The pinned release of each package, which every one of its assets must name.
PINNED_VERSIONS: Final[dict[str, str]] = {
    "swagger-ui-dist": SWAGGER_UI_VERSION,
    "redoc": REDOC_VERSION,
}


class Outcome(StrEnum):
    """What comparing one pin against its source established.

    Declared worst first: the report is grouped in this order, so what needs a
    person shows above what does not.
    """

    OUTDATED = "OUTDATED"
    DIGEST = "DIGEST"
    LICENCE = "LICENCE"
    UNREACHABLE = "UNREACHABLE"
    CURRENT = "CURRENT"


@dataclass(frozen=True, slots=True)
class Finding:
    """One outcome, for one package or file, in enough detail to act on."""

    outcome: Outcome
    subject: str
    detail: str


class AssetSourceWarning(UserWarning):
    """A source said something the run reports but must not fail on."""


def classify_release(package: str, pinned: str, published: dict[str, str]) -> Finding:
    """Compare one pin against the release npm currently calls latest.

    Args:
        package: npm package the pin names.
        pinned: Version the manifest pins.
        published: The registry's manifest of the latest release.

    Returns:
        Whether the pin is the latest release, or which one superseded it.
    """
    latest = published.get("version")
    if not latest:
        detail = f"the registry manifest states no version: {sorted(published)}"
        return Finding(Outcome.UNREACHABLE, package, detail)
    if latest != pinned:
        detail = (
            f"pinned {pinned}, npm publishes {latest}. Read that release's notes "
            f"before bumping: a page that renders differently is not a patch."
        )
        return Finding(Outcome.OUTDATED, package, detail)
    return Finding(Outcome.CURRENT, package, f"pinned {pinned}, the latest release")


def classify_licence(package: str, published: dict[str, str]) -> Finding:
    """Compare the licence recorded for a package against what npm declares.

    Args:
        package: npm package the pin names.
        published: The registry's manifest of the latest release.

    Returns:
        Whether the redistributed notice is still the right one.
    """
    recorded = UPSTREAM_LICENSES[package]
    declared = published.get("license")
    if not declared:
        return Finding(
            Outcome.UNREACHABLE, package, "the registry manifest states no licence"
        )
    if declared != recorded:
        detail = (
            f"redistributed as {recorded}, npm now declares {declared}. The image "
            f"ships this package's licence text, so the notice has to follow."
        )
        return Finding(Outcome.LICENCE, package, detail)
    return Finding(Outcome.CURRENT, package, f"still {recorded}")


def classify_digest(asset: Asset, served: bytes | None, problem: str | None) -> Finding:
    """Compare what the publisher serves for a pinned file against its digest.

    Args:
        asset: The pinned file.
        served: What the publisher answered with, or None when it did not.
        problem: Why it did not, or None when it did.

    Returns:
        Whether the published bytes are the ones every image build bakes in.
    """
    if served is None:
        return Finding(Outcome.UNREACHABLE, asset.url, str(problem))
    if (found := digest(served)) != asset.sha256:
        detail = (
            f"the manifest records {asset.sha256}, {asset.url} now serves {found}. "
            f"An exact version is meant to be immutable: establish which is right "
            f"before either digest ships."
        )
        return Finding(Outcome.DIGEST, asset.name, detail)
    return Finding(Outcome.CURRENT, asset.name, f"sha256 {found}")


def format_report(findings: Iterable[Finding]) -> str:
    """Render *findings* grouped by outcome, worst first.

    Args:
        findings: What the run established.

    Returns:
        The report, one group per outcome present.
    """
    collected = list(findings)
    lines = ["The pinned documentation assets vs. what upstream publishes:", ""]
    for outcome in Outcome:
        selected = [finding for finding in collected if finding.outcome is outcome]
        if not selected:
            continue
        lines.append(f"{outcome.value} ({len(selected)}):")
        lines.extend(f"  {finding.subject}: {finding.detail}" for finding in selected)
        lines.append("")
    return "\n".join(lines)


#: What to do about a finding, printed with the failure that reports one.
_FIX: Final[str] = (
    "FIX: bump the pinned version in stdapi/docs_assets/__init__.py, re-record "
    "every SHA-256 beside it (python -m stdapi.docs_assets prints what it "
    "verified; the digests are what the build checks), re-run the container "
    "suite so both images still serve the pages, and note the change in the "
    "release entry. Never bump without reading the upstream release notes: a "
    "new major of either library can change the page, which is what the pin is "
    "protecting against."
)


def _collect(client: httpx.Client) -> list[Finding]:
    """Read every source once and classify every pin against it."""
    findings: list[Finding] = []
    for package, pinned in PINNED_VERSIONS.items():
        url = _REGISTRY.format(package=package)
        try:
            response = client.get(url)
            response.raise_for_status()
            published = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            findings.append(
                Finding(Outcome.UNREACHABLE, url, f"{type(exc).__name__}: {exc}")
            )
            continue
        findings.append(classify_release(package, pinned, published))
        findings.append(classify_licence(package, published))

    for asset in (*BROWSER_ASSETS.values(), *LICENSE_ASSETS):
        try:
            response = client.get(asset.url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            findings.append(
                classify_digest(asset, None, f"{type(exc).__name__}: {exc}")
            )
        else:
            findings.append(classify_digest(asset, response.content, None))
    return findings


@pytest.mark.drift
def test_the_pinned_documentation_assets_are_still_current() -> None:
    """Every pinned release must still be the latest, and serve the recorded bytes.

    The pin is deliberate and is never bumped automatically, so the only thing
    keeping it from quietly rotting is this run saying that upstream moved. A
    source that could not be read is reported instead of failed -- it says
    nothing about the pin -- and a run that could compare nothing at all skips,
    so a green result never means "not checked".

    Ref: stdapi/docs_assets/__init__.py
         https://registry.npmjs.org/swagger-ui-dist/latest
         https://registry.npmjs.org/redoc/latest
    """
    with httpx.Client(
        timeout=_FETCH_TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    ) as client:
        findings = _collect(client)

    report = format_report(findings)
    print(report)  # noqa: T201 -- shown by pytest on failure, and with -s
    if all(finding.outcome is Outcome.UNREACHABLE for finding in findings):
        pytest.skip(f"No source could be read.\n{report}")
    reported = [f for f in findings if f.outcome is Outcome.UNREACHABLE]
    if reported:
        warnings.warn(format_report(reported), AssetSourceWarning, stacklevel=2)
    if any(
        finding.outcome not in {Outcome.CURRENT, Outcome.UNREACHABLE}
        for finding in findings
    ):
        pytest.fail(f"{report}\n{_FIX}")


class TestManifest:
    """The manifest itself, which the pin and the fallback URL both come from.

    Ref: stdapi/docs_assets/__init__.py
    """

    @pytest.mark.parametrize("asset", [*BROWSER_ASSETS.values(), *LICENSE_ASSETS])
    def test_every_asset_names_its_package_pinned_release(self, asset: Asset) -> None:
        """No URL may carry a floating tag, in an image or in a source checkout.

        A floating major is what FastAPI defaults to and what this replaces: the
        code a reader's browser runs would otherwise change without anything
        here changing, and the recorded digest would stop describing it.
        """
        assert asset.version == PINNED_VERSIONS[asset.package]
        for url in asset.urls:
            assert f"{asset.package}@{asset.version}/" in url
            assert url.startswith("https://")

    @pytest.mark.parametrize("asset", [*BROWSER_ASSETS.values(), *LICENSE_ASSETS])
    def test_every_asset_records_a_sha256(self, asset: Asset) -> None:
        """A build verifies each download against this digest before writing it."""
        assert len(asset.sha256) == 64
        assert asset.sha256 == asset.sha256.lower()
        bytes.fromhex(asset.sha256)

    def test_every_package_records_the_licence_it_redistributes(self) -> None:
        """Each package a file comes from ships its licence text, under that licence."""
        packaged = {asset.package for asset in LICENSE_ASSETS}

        assert {asset.package for asset in BROWSER_ASSETS.values()} <= packaged
        assert set(UPSTREAM_LICENSES) == packaged


class TestDetection:
    """The detector's directions, proved on the pins the repository ships.

    Ref: stdapi/docs_assets/__init__.py
    """

    def test_the_shipped_pin_matches_a_registry_naming_it_latest(self) -> None:
        """A registry answer naming the pinned version reports nothing to do."""
        published = {"version": SWAGGER_UI_VERSION, "license": "Apache-2.0"}

        assert (
            classify_release("swagger-ui-dist", SWAGGER_UI_VERSION, published).outcome
            is Outcome.CURRENT
        )

    def test_a_newer_release_is_reported_as_outdated(self) -> None:
        """The finding this lane exists for names both versions.

        This is the failure a pin rots into: nothing at runtime can notice it,
        and nobody bumps what nobody is told about.
        """
        finding = classify_release(
            "redoc", REDOC_VERSION, {"version": "2.9.9", "license": "MIT"}
        )

        assert finding.outcome is Outcome.OUTDATED
        assert REDOC_VERSION in finding.detail
        assert "2.9.9" in finding.detail

    def test_a_registry_answer_without_a_version_is_unreachable(self) -> None:
        """A changed registry payload must not read as a withdrawn release."""
        finding = classify_release("redoc", REDOC_VERSION, {"dist-tags": "moved"})

        assert finding.outcome is Outcome.UNREACHABLE

    def test_a_relicensed_package_is_reported(self) -> None:
        """The image redistributes these licence texts, so a change has to surface."""
        finding = classify_licence(
            "redoc", {"version": REDOC_VERSION, "license": "BSL"}
        )

        assert finding.outcome is Outcome.LICENCE
        assert "BSL" in finding.detail
        assert "MIT" in finding.detail

    def test_the_recorded_digest_matches_the_bytes_it_describes(self) -> None:
        """A publisher serving exactly the recorded bytes reports nothing to do."""
        asset = BROWSER_ASSETS["swagger-ui.css"]
        served = b"body{}"
        recorded = Asset(
            asset.package, asset.version, asset.path, digest(served), asset.media_type
        )

        assert classify_digest(recorded, served, None).outcome is Outcome.CURRENT

    def test_changed_bytes_at_a_pinned_version_are_reported(self) -> None:
        """An exact version is meant to be immutable; a build would fail on this."""
        asset = BROWSER_ASSETS["swagger-ui-bundle.js"]

        finding = classify_digest(asset, b"not what was recorded", None)

        assert finding.outcome is Outcome.DIGEST
        assert asset.sha256 in finding.detail

    def test_an_unreachable_publisher_is_never_a_digest_finding(self) -> None:
        """A file that could not be fetched says nothing about the pinned bytes."""
        asset = BROWSER_ASSETS["redoc.standalone.js"]

        finding = classify_digest(asset, None, "ConnectTimeout")

        assert finding.outcome is Outcome.UNREACHABLE
        assert "ConnectTimeout" in finding.detail


class TestReport:
    """The report groups findings so the actionable ones are read first.

    Ref: tests/test_docs_assets_drift.py:format_report
    """

    def test_every_outcome_is_grouped_and_counted(self) -> None:
        """Each outcome present gets a counted heading, worst first."""
        report = format_report(
            [
                Finding(Outcome.CURRENT, "a", "ok"),
                Finding(Outcome.OUTDATED, "b", "moved"),
                Finding(Outcome.OUTDATED, "c", "moved"),
            ]
        )

        assert "CURRENT (1):" in report
        assert "OUTDATED (2):" in report
        assert "DIGEST" not in report
        assert report.index("OUTDATED") < report.index("CURRENT")
