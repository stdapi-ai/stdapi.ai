"""Fetching the documentation pages' browser assets into the image.

``python -m stdapi.docs_assets`` runs inside both container image builds, and
what it writes is redistributed in a paid product. So the one thing it must
never do is write bytes nobody reviewed: a publisher that answers with something
other than the recorded digest ends the build, rather than baking an unverified
blob into an image where it would be trusted precisely because it is local.

Nothing here reaches the network. The publishers are stubbed, because what is
under test is what the fetcher does with an answer -- a good one, a wrong one,
and none at all -- not whether a CDN was up. Whether the pins are still current,
and whether the recorded digests still describe what upstream publishes, is
``tests/test_docs_assets_drift.py``.

Ref: https://github.com/stdapi-ai/stdapi.ai/issues/185
     stdapi/docs_assets/__init__.py
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Final

import pytest

import stdapi.docs_assets
from stdapi.docs_assets import BROWSER_ASSETS, LICENSE_ASSETS, Asset, digest, fetch

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

pytestmark = pytest.mark.local

#: Bytes the stubbed publishers answer with, and the asset recording their digest.
_BODY: Final[bytes] = b"/* not the real Swagger UI */"


def _asset() -> Asset:
    """Return an asset whose recorded digest is that of the stubbed answer."""
    return Asset(
        "swagger-ui-dist", "9.9.9", "swagger-ui.css", digest(_BODY), "text/css"
    )


@pytest.fixture
def publishers(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[str]]:
    """Stub every publisher, and record the addresses the fetcher asked for.

    Returns:
        A callable taking one answer per attempt -- bytes to serve, or an
        exception to raise -- and returning the list it will record into.
    """

    def install(*answers: bytes | OSError) -> list[str]:
        asked: list[str] = []
        remaining = list(answers)

        @contextmanager
        def opener(url: str, timeout: float = 0) -> Iterator[object]:  # noqa: ARG001
            asked.append(url)
            answer = remaining.pop(0) if remaining else OSError("nothing left to serve")
            if isinstance(answer, OSError):
                raise answer
            yield type("Answer", (), {"read": lambda _self: answer})()

        monkeypatch.setattr(stdapi.docs_assets, "urlopen", opener)
        return asked

    return install


class TestFetching:
    """What the build writes, and what it refuses to write.

    Ref: stdapi/docs_assets/__init__.py:fetch
    """

    def test_a_matching_answer_is_returned(
        self, publishers: Callable[..., list[str]]
    ) -> None:
        """Bytes whose digest is the recorded one are the bytes the build writes."""
        asset = _asset()
        asked = publishers(_BODY)

        assert fetch(asset) == _BODY
        assert asked == [asset.url]

    def test_bytes_the_manifest_does_not_record_are_refused(
        self, publishers: Callable[..., list[str]]
    ) -> None:
        """A publisher answering with anything else ends the build.

        This is the whole point of recording a digest: an image ships these
        files as its own, so a browser trusts them further than it ever trusted
        the CDN. A substituted script would be trusted with them.
        """
        publishers(b"window.alert(1)")

        with pytest.raises(RuntimeError, match="Refusing to ship unverified bytes"):
            fetch(_asset())

    def test_a_wrong_answer_is_never_retried_at_the_next_publisher(
        self, publishers: Callable[..., list[str]]
    ) -> None:
        """Shopping around for a publisher that agrees is not verification."""
        asked = publishers(b"window.alert(1)", _BODY)

        with pytest.raises(RuntimeError):
            fetch(_asset())
        assert len(asked) == 1

    def test_the_second_publisher_answers_when_the_first_cannot(
        self, publishers: Callable[..., list[str]]
    ) -> None:
        """One publisher being down is not a reason to fail an image build.

        The digest is what makes a second publisher safe to accept at all: the
        file is the same file or it is not fetched.
        """
        asset = _asset()
        asked = publishers(*[OSError("connection refused")] * 3, _BODY)

        assert fetch(asset) == _BODY
        assert asked[-1] == asset.urls[-1]

    def test_no_publisher_answering_names_every_address_tried(
        self, publishers: Callable[..., list[str]]
    ) -> None:
        """A build that could not fetch says where it looked, and fails."""
        asset = _asset()
        publishers()

        with pytest.raises(RuntimeError, match="could not be fetched") as failure:
            fetch(asset)
        for url in asset.urls:
            assert url in str(failure.value)

    def test_every_file_lands_where_the_gateway_and_the_image_read_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A fetched run writes each script beside the package and each licence below it.

        The two locations are what the rest of the change depends on: the
        gateway serves from the first, and both Dockerfiles copy the second into
        ``/usr/share/licenses``.
        """
        monkeypatch.setattr(stdapi.docs_assets, "ASSETS_DIR", tmp_path)
        monkeypatch.setattr(stdapi.docs_assets, "LICENSES_DIR", tmp_path / "licenses")
        monkeypatch.setattr(
            stdapi.docs_assets, "fetch", lambda asset: asset.name.encode()
        )

        written = stdapi.docs_assets.fetch_all()

        assert {path.relative_to(tmp_path).as_posix() for path in written} == {
            *BROWSER_ASSETS,
            *(f"licenses/{a.package}/{a.name}" for a in LICENSE_ASSETS),
        }
        for path in written:
            assert path.read_bytes() == path.name.encode()
