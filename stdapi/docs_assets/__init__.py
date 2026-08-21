"""Browser assets the interactive API documentation pages load.

FastAPI's ``/docs`` and ``/redoc`` pages load Swagger UI and ReDoc from
``cdn.jsdelivr.net``, at floating major tags. That is two defects at once: an
air-gapped or egress-restricted deployment renders a blank page, and every other
deployment runs, in an operator's browser, whatever those tags resolve to at
that moment. The container image builds fetch these files instead -- pinned to
an exact release, each verified against the digest recorded below -- and the
gateway serves them itself.

A source checkout has no fetched file, so the pages name the publisher URL
instead. That URL still carries the exact version: development and the
in-process suite keep working without ever handing a browser a floating tag.

``python -m stdapi.docs_assets`` fetches and verifies them into this package;
``tests/test_docs_assets_drift.py`` reports when a pin falls behind upstream.

Ref: https://github.com/stdapi-ai/stdapi.ai/issues/185
     https://github.com/stdapi-ai/stdapi.ai/issues/184
"""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from urllib.request import urlopen

#: Exact ``swagger-ui-dist`` release the Swagger UI page loads, under Apache-2.0.
SWAGGER_UI_VERSION = "5.32.14"

#: Exact ``redoc`` release the ReDoc page loads, under the MIT licence.
REDOC_VERSION = "2.5.3"

#: Licence each package is published under, checked against npm by the drift lane.
UPSTREAM_LICENSES = {"swagger-ui-dist": "Apache-2.0", "redoc": "MIT"}

#: Publishers of an exact npm file, tried in order; the first is what a page names.
_PUBLISHERS = (
    "https://cdn.jsdelivr.net/npm/{package}@{version}/{path}",
    "https://unpkg.com/{package}@{version}/{path}",
)

#: Address the gateway serves the fetched files under, below any root path.
ASSETS_PATH = "/docs-assets"

#: Directory a build writes the fetched files to, and the gateway reads them from.
ASSETS_DIR = Path(__file__).parent

#: Directory holding the upstream licence texts, one subdirectory per package.
LICENSES_DIR = ASSETS_DIR / "licenses"

#: Seconds one publisher gets to serve one file.
_FETCH_TIMEOUT = 60.0

#: Attempts per publisher before the next one is tried.
_ATTEMPTS = 3


@dataclass(frozen=True, slots=True)
class Asset:
    """One upstream file, pinned to an exact release and verified by digest.

    Attributes:
        package: npm package publishing the file.
        version: Exact release it is taken from, never a floating tag.
        path: Its path inside the published package.
        sha256: Hex SHA-256 the fetched bytes must have, or the build fails.
        media_type: Content type the gateway serves it as; empty for a licence
            text, which is redistributed but never served.
    """

    package: str
    version: str
    path: str
    sha256: str
    media_type: str = ""

    @property
    def name(self) -> str:
        """The file's own name, which is how it is stored and served."""
        return self.path.rpartition("/")[2]

    @property
    def urls(self) -> tuple[str, ...]:
        """Every publisher of this exact file, in the order a build tries them."""
        return tuple(
            template.format(package=self.package, version=self.version, path=self.path)
            for template in _PUBLISHERS
        )

    @property
    def url(self) -> str:
        """Where a deployment without a fetched copy points a browser instead."""
        return self.urls[0]

    @property
    def target(self) -> Path:
        """Where a fetched copy is written inside this package."""
        if self.media_type:
            return ASSETS_DIR / self.name
        return LICENSES_DIR / self.package / self.name


#: Files a documentation page loads in a browser, keyed by the name it is served as.
BROWSER_ASSETS: dict[str, Asset] = {
    asset.name: asset
    for asset in (
        Asset(
            "swagger-ui-dist",
            SWAGGER_UI_VERSION,
            "swagger-ui-bundle.js",
            "16d93d5cc19e54c98fb0b81157dbb3bd90780aa36b914e128a643b31e54a93f4",
            "text/javascript",
        ),
        Asset(
            "swagger-ui-dist",
            SWAGGER_UI_VERSION,
            "swagger-ui.css",
            "d7f39f764aa18c7b47dd05b9af5613e373e4ac0f3557c2693d52d0abc2464d76",
            "text/css",
        ),
        Asset(
            "redoc",
            REDOC_VERSION,
            "bundles/redoc.standalone.js",
            "1320f442151c57c447d3b70c7ffc6c4f86d08464020fe34c8cc5d3164e9944f0",
            "text/javascript",
        ),
    )
}

#: Upstream licence and notice texts, redistributed beside the files they cover.
LICENSE_ASSETS: tuple[Asset, ...] = (
    Asset(
        "swagger-ui-dist",
        SWAGGER_UI_VERSION,
        "LICENSE",
        "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    ),
    Asset(
        "swagger-ui-dist",
        SWAGGER_UI_VERSION,
        "NOTICE",
        "0d20d1adef18aee3f40dd258172155521ce702ac445cb5f7b7d60ed32dad2fb2",
    ),
    Asset(
        "swagger-ui-dist",
        SWAGGER_UI_VERSION,
        "swagger-ui-bundle.js.LICENSE.txt",
        "c07853f3704b510a864eb56561ca4f36e0347fdaefc5176611c57575e4b5593d",
    ),
    Asset(
        "redoc",
        REDOC_VERSION,
        "LICENSE",
        "d3026d549cf68ab7355bcfa85877bf8f845b3334a7efbfdc63936432fb34ff0e",
    ),
    Asset(
        "redoc",
        REDOC_VERSION,
        "bundles/redoc.standalone.js.LICENSE.txt",
        "469cc94b600aac09643f70e167cd1f66f24301ebb546532fad5db7c60f7b30d0",
    ),
)

#: Addresses the fetched files answer at, which the request log ignores like the icon.
ASSET_PATHS: frozenset[str] = frozenset(
    f"{ASSETS_PATH}/{name}" for name in BROWSER_ASSETS
)

#: Files this deployment has a fetched copy of; empty in a source checkout.
LOCAL_ASSETS: dict[str, Path] = {
    name: path for name in BROWSER_ASSETS if (path := ASSETS_DIR / name).is_file()
}


def digest(data: bytes) -> str:
    """Return the SHA-256 of *data*, in the form the manifest records it.

    Args:
        data: Bytes to hash.

    Returns:
        The lowercase hex digest.
    """
    return sha256(data).hexdigest()


def fetch(asset: Asset) -> bytes:
    """Download one pinned file and return it only once its digest matches.

    Every publisher is tried before giving up, but a publisher that answers with
    unexpected bytes ends the attempt immediately: baking an unverified blob
    into a redistributed image is worse than loading it from a CDN at runtime.

    Args:
        asset: The file to download.

    Returns:
        Its bytes, verified against the recorded digest.

    Raises:
        RuntimeError: If a publisher served bytes the manifest does not record,
            or if no publisher served the file at all.
    """
    problems: list[str] = []
    for url in asset.urls:
        for attempt in range(_ATTEMPTS):
            try:
                with urlopen(url, timeout=_FETCH_TIMEOUT) as response:  # noqa: S310 -- every publisher above is an https literal
                    data: bytes = response.read()
            except OSError as exc:
                problems.append(f"{url} (attempt {attempt + 1}): {exc}")
                continue
            if (found := digest(data)) != asset.sha256:
                msg = (
                    f"{url} served sha256 {found}, the manifest records "
                    f"{asset.sha256}. Refusing to ship unverified bytes."
                )
                raise RuntimeError(msg)
            return data
    msg = f"{asset.name} could not be fetched: {'; '.join(problems)}"
    raise RuntimeError(msg)


def fetch_all() -> list[Path]:
    """Fetch every pinned file and licence text into this package.

    Returns:
        The files written, in the order they were fetched.
    """
    written: list[Path] = []
    for asset in (*BROWSER_ASSETS.values(), *LICENSE_ASSETS):
        data = fetch(asset)
        asset.target.parent.mkdir(parents=True, exist_ok=True)
        asset.target.write_bytes(data)
        written.append(asset.target)
    return written
