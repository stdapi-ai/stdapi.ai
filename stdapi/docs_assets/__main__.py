"""Fetch the documentation pages' pinned browser assets into this package.

Run by both container image builds, right after the application sources are
copied in, so the built image serves ``/docs`` and ``/redoc`` without reaching
any CDN. Every file is verified against the digest the manifest records; a
mismatch or an unreachable publisher fails the build rather than shipping a page
that half-loads.
"""

from stdapi.docs_assets import ASSETS_DIR, fetch_all

for path in fetch_all():
    print(f"Fetched {path.relative_to(ASSETS_DIR)}")  # noqa: T201 -- read from the image build log
