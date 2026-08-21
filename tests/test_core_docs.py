"""The interactive documentation pages and everything a browser loads with them.

FastAPI's built-in ``/docs`` and ``/redoc`` write three third-party URLs into
every page: an icon on ``fastapi.tiangolo.com``, Swagger UI and ReDoc on
``cdn.jsdelivr.net`` at floating major tags, and a ReDoc web font on
``fonts.googleapis.com``. An air-gapped or egress-restricted deployment can
reach none of them, and no other response the gateway produces needs to. The
pages are served here instead, from an icon the package ships and from the
pinned scripts the image build fetched.

The pages are disabled in the test configuration, so the ones under test are
built from a private copy of the route module re-executed with them enabled --
never a reload, which would replace the running application's own routes. That
copy is also where the two deployments are told apart: an image, which fetched
the scripts, and a source checkout, which did not.

Ref: https://github.com/stdapi-ai/stdapi.ai/issues/184
     https://github.com/stdapi-ai/stdapi.ai/issues/185
     stdapi/routes/core_docs.py
     stdapi/docs_assets/__init__.py
"""

from __future__ import annotations

from importlib.resources import files
from importlib.util import find_spec, module_from_spec
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI

import stdapi.docs_assets
from stdapi.config import SETTINGS
from stdapi.docs_assets import (
    ASSET_PATHS,
    ASSETS_PATH,
    BROWSER_ASSETS,
    LOCAL_ASSETS,
    REDOC_VERSION,
    SWAGGER_UI_VERSION,
    digest,
)
from stdapi.monitoring import LOGGING_PATHS_IGNORE
from stdapi.routes.core_docs import FAVICON_PATH

if TYPE_CHECKING:
    from starlette.testclient import TestClient

pytestmark = pytest.mark.local

#: Host FastAPI's built-in documentation pages load their icon from.
_THIRD_PARTY_HOST = "fastapi.tiangolo.com"

#: Host FastAPI's built-in pages load Swagger UI and ReDoc from.
_CDN_HOST = "cdn.jsdelivr.net"

#: Host FastAPI's built-in ReDoc page loads a web font from.
_FONTS_HOST = "fonts.googleapis.com"

#: The floating major tags FastAPI defaults to, which no page may ever name.
_FLOATING_TAGS = ("swagger-ui-dist@5/", "redoc@2/")

#: Scripts and stylesheets each page has to load, by page path.
_PAGE_ASSETS = {
    "/docs": ("swagger-ui-bundle.js", "swagger-ui.css"),
    "/redoc": ("redoc.standalone.js",),
}

#: Title the documentation pages under test name the deployment with.
_TITLE = "Test Gateway"

#: Root path a deployment behind a proxy or a mount prefix serves the whole app below.
_ROOT_PATH = "/gateway"

#: The published documentation site's mark, which the packaged icon has to be.
_SITE_LOGO = Path(__file__).parents[1] / "docs" / "styles" / "logo.svg"


def _documentation_app(
    monkeypatch: pytest.MonkeyPatch, fetched: dict[str, Path]
) -> FastAPI:
    """Build an application serving both pages, given the files a build fetched.

    Built like the served one: the schema to document, a title to name it by,
    and FastAPI's own pages left unregistered so the replacements answer. The
    route module is re-executed so it reads *fetched* as its own, which is what
    decides whether the pages point at the gateway or at the publisher.

    Args:
        monkeypatch: Patcher undone at the end of the test.
        fetched: Files the deployment has a fetched copy of, by served name.

    Returns:
        An application holding a private copy of the routes.
    """
    monkeypatch.setattr(SETTINGS, "enable_docs", True)
    monkeypatch.setattr(SETTINGS, "enable_redoc", True)
    monkeypatch.setattr(stdapi.docs_assets, "LOCAL_ASSETS", fetched)
    spec = find_spec("stdapi.routes.core_docs")
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    app = FastAPI(
        title=_TITLE, openapi_url="/openapi.json", docs_url=None, redoc_url=None
    )
    app.include_router(module.router)
    return app


@pytest.fixture
def fetched_assets(tmp_path: Path) -> dict[str, Path]:
    """Stand-ins for the files an image build fetches into the package.

    Their bytes are not the upstream ones -- the digests of those are checked
    where they are fetched -- only distinct enough that serving the wrong file
    is visible.

    Returns:
        A file per manifest entry, by served name.
    """
    written: dict[str, Path] = {}
    for name in BROWSER_ASSETS:
        path = tmp_path / name
        path.write_bytes(f"/* {name} */".encode())
        written[name] = path
    return written


@pytest.fixture
def docs_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """An application serving both pages in a source checkout, which fetched nothing."""
    return _documentation_app(monkeypatch, {})


@pytest.fixture
def local_docs_app(
    monkeypatch: pytest.MonkeyPatch, fetched_assets: dict[str, Path]
) -> FastAPI:
    """An application serving both pages as a built image does, with its own scripts."""
    return _documentation_app(monkeypatch, fetched_assets)


@pytest.fixture
def docs_client(docs_app: FastAPI) -> TestClient:
    """Client for the application serving both documentation pages."""
    from starlette.testclient import TestClient  # noqa: PLC0415

    return TestClient(docs_app)


@pytest.fixture
def local_docs_client(local_docs_app: FastAPI) -> TestClient:
    """Client for the application serving its own copies of the scripts."""
    from starlette.testclient import TestClient  # noqa: PLC0415

    return TestClient(local_docs_app)


@pytest.fixture
def mounted_docs_client(docs_app: FastAPI) -> TestClient:
    """Client for the same application mounted under a non-empty root path."""
    from starlette.testclient import TestClient  # noqa: PLC0415

    return TestClient(docs_app, root_path=_ROOT_PATH)


@pytest.fixture
def mounted_local_docs_client(local_docs_app: FastAPI) -> TestClient:
    """Client for the image-like application mounted under a non-empty root path."""
    from starlette.testclient import TestClient  # noqa: PLC0415

    return TestClient(local_docs_app, root_path=_ROOT_PATH)


class TestFavicon:
    """The icon route the documentation pages point at.

    Ref: stdapi/routes/core_docs.py:favicon
    """

    def test_serves_the_packaged_icon(self, enforced_auth_client: TestClient) -> None:
        """The icon route returns the mark the package ships, as an SVG document.

        The pages name this address instead of a third-party URL, so a browser
        that cannot leave the deployment still renders the icon.
        """
        response = enforced_auth_client.get(FAVICON_PATH)

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/svg+xml"
        assert response.content == (files("stdapi") / "favicon.svg").read_bytes()

    def test_is_cacheable(self, enforced_auth_client: TestClient) -> None:
        """The icon is publicly cacheable: it changes only with the server version."""
        cache_control = enforced_auth_client.get(FAVICON_PATH).headers["cache-control"]

        assert "public" in cache_control
        assert "max-age=0" not in cache_control

    def test_no_auth_required(self, enforced_auth_client: TestClient) -> None:
        """The icon is served with no credentials, against an armed API key check.

        ``/docs`` and ``/redoc`` are reachable without a key, so an icon that
        answered 401 would leave both pages unbranded for exactly the readers
        they exist for.

        Ref: stdapi/auth.py:authenticate
        """
        anonymous = enforced_auth_client.get(FAVICON_PATH)
        assert anonymous.status_code == 200

        bad_key = enforced_auth_client.get(
            FAVICON_PATH, headers={"Authorization": "Bearer wrong-key"}
        )
        assert bad_key.status_code == 200
        assert bad_key.content == anonymous.content

    def test_is_exempt_from_request_logging(self) -> None:
        """The icon address is one of the paths the request log ignores.

        Every documentation page load fetches it, and a log entry per icon is
        noise no operator asked for.

        Ref: stdapi/monitoring.py:LOGGING_PATHS_IGNORE
        """
        assert FAVICON_PATH in LOGGING_PATHS_IGNORE


class TestSwaggerUi:
    """The ``/docs`` page the gateway serves in place of FastAPI's own.

    Ref: stdapi/routes/core_docs.py:swagger_ui_html
    """

    def test_page_loads_its_icon_from_the_gateway(
        self, docs_client: TestClient
    ) -> None:
        """``/docs`` names the gateway's icon address and no third-party host."""
        page = docs_client.get("/docs")

        assert page.status_code == 200
        assert f'href="{FAVICON_PATH}"' in page.text
        assert _THIRD_PARTY_HOST not in page.text

    def test_page_documents_this_deployment(self, docs_client: TestClient) -> None:
        """``/docs`` reads the application's own schema and names it by its title."""
        page = docs_client.get("/docs").text

        assert f"<title>{_TITLE} - Swagger UI</title>" in page
        assert "'/openapi.json'" in page

    def test_addresses_follow_the_root_path(
        self, mounted_docs_client: TestClient
    ) -> None:
        """Behind a mount prefix, every address the page names carries it.

        A page-relative address written without the root path points at nothing,
        which is a broken icon and an empty page rather than a visible error.
        """
        page = mounted_docs_client.get("/docs").text

        assert f'href="{_ROOT_PATH}{FAVICON_PATH}"' in page
        assert f"'{_ROOT_PATH}/openapi.json'" in page

    def test_the_oauth2_redirect_page_is_still_served(
        self, docs_client: TestClient
    ) -> None:
        """The page Swagger UI's authorization flow ends on is mounted as before.

        FastAPI registers it alongside its own ``/docs``; taking that page over
        without it would leave the flow returning to a 404.
        """
        assert docs_client.get("/docs/oauth2-redirect").status_code == 200


class TestReDoc:
    """The ``/redoc`` page the gateway serves in place of FastAPI's own.

    Ref: stdapi/routes/core_docs.py:redoc_html
    """

    def test_page_loads_its_icon_from_the_gateway(
        self, docs_client: TestClient
    ) -> None:
        """``/redoc`` names the gateway's icon address and no third-party host."""
        page = docs_client.get("/redoc")

        assert page.status_code == 200
        assert f'href="{FAVICON_PATH}"' in page.text
        assert _THIRD_PARTY_HOST not in page.text

    def test_page_documents_this_deployment(self, docs_client: TestClient) -> None:
        """``/redoc`` reads the application's own schema and names it by its title."""
        page = docs_client.get("/redoc").text

        assert f"<title>{_TITLE} - ReDoc</title>" in page
        assert 'spec-url="/openapi.json"' in page

    def test_addresses_follow_the_root_path(
        self, mounted_docs_client: TestClient
    ) -> None:
        """Behind a mount prefix, every address the page names carries it."""
        page = mounted_docs_client.get("/redoc").text

        assert f'href="{_ROOT_PATH}{FAVICON_PATH}"' in page
        assert f'spec-url="{_ROOT_PATH}/openapi.json"' in page


class TestDocumentationAssets:
    """The Swagger UI and ReDoc files the pages load, served by the gateway.

    Ref: https://github.com/stdapi-ai/stdapi.ai/issues/185
         stdapi/routes/core_docs.py:docs_asset
    """

    @pytest.mark.parametrize("page", list(_PAGE_ASSETS))
    def test_a_page_loads_every_script_from_the_gateway(
        self, local_docs_client: TestClient, page: str
    ) -> None:
        """A deployment that fetched the scripts names no third-party host at all.

        This is what an air-gapped or egress-restricted deployment needs: the
        page is useless if the browser has to leave the deployment to render it,
        and it renders nothing at all when it cannot.
        """
        rendered = local_docs_client.get(page)

        assert rendered.status_code == 200
        for name in _PAGE_ASSETS[page]:
            assert f'"{ASSETS_PATH}/{name}"' in rendered.text
        assert _CDN_HOST not in rendered.text
        assert _FONTS_HOST not in rendered.text
        assert _THIRD_PARTY_HOST not in rendered.text

    @pytest.mark.parametrize("page", list(_PAGE_ASSETS))
    def test_a_source_checkout_falls_back_to_the_pinned_publisher(
        self, docs_client: TestClient, page: str
    ) -> None:
        """Without a fetched copy the page names the publisher, at the exact version.

        Only a source checkout is in that state, and it keeps development and
        this suite working. What it must never do is hand a browser a floating
        major tag, which is what FastAPI's own default is: the version served
        then changes under every reader without anything here changing.
        """
        rendered = docs_client.get(page).text

        for name in _PAGE_ASSETS[page]:
            assert BROWSER_ASSETS[name].url in rendered
        assert SWAGGER_UI_VERSION in rendered or REDOC_VERSION in rendered
        for tag in _FLOATING_TAGS:
            assert tag not in rendered
        assert _FONTS_HOST not in rendered

    @pytest.mark.parametrize("name", list(BROWSER_ASSETS))
    def test_the_route_serves_the_fetched_file_itself(
        self, local_docs_client: TestClient, fetched_assets: dict[str, Path], name: str
    ) -> None:
        """Each asset address answers with that file's bytes and its own type."""
        response = local_docs_client.get(f"{ASSETS_PATH}/{name}")

        assert response.status_code == 200
        assert response.content == fetched_assets[name].read_bytes()
        assert response.headers["content-type"].startswith(
            BROWSER_ASSETS[name].media_type
        )

    def test_the_assets_are_cacheable(self, local_docs_client: TestClient) -> None:
        """They change only with the server version, so a browser may keep them."""
        name = next(iter(BROWSER_ASSETS))
        cache_control = local_docs_client.get(f"{ASSETS_PATH}/{name}").headers[
            "cache-control"
        ]

        assert "public" in cache_control
        assert "max-age=0" not in cache_control

    @pytest.mark.parametrize("name", ["unknown.js", "swagger-ui.css"])
    def test_a_file_this_deployment_never_fetched_is_not_served(
        self, docs_client: TestClient, name: str
    ) -> None:
        """An unknown name, and a known one nothing fetched, are both a plain 404.

        The served name is looked up in the manifest rather than joined onto a
        directory, so no name a caller invents can name a file outside it.
        """
        assert docs_client.get(f"{ASSETS_PATH}/{name}").status_code == 404

    def test_addresses_follow_the_root_path(
        self, mounted_local_docs_client: TestClient
    ) -> None:
        """Behind a mount prefix, the script addresses carry it like every other.

        Without it the page requests a script from the proxy's own root, gets
        whatever answers there, and renders blank -- the exact failure this
        replaces, with the third party swapped for the reverse proxy.
        """
        rendered = mounted_local_docs_client.get("/docs").text

        for name in _PAGE_ASSETS["/docs"]:
            assert f'"{_ROOT_PATH}{ASSETS_PATH}/{name}"' in rendered

    def test_no_auth_required(
        self,
        enforced_auth_client: TestClient,
        monkeypatch: pytest.MonkeyPatch,
        fetched_assets: dict[str, Path],
    ) -> None:
        """The scripts are served with no credentials, against an armed API key check.

        ``/docs`` and ``/redoc`` are reachable without a key, so scripts that
        answered 401 would leave both pages blank for exactly the readers they
        exist for.

        Ref: stdapi/auth.py:authenticate
        """
        name, path = next(iter(fetched_assets.items()))
        monkeypatch.setitem(LOCAL_ASSETS, name, path)

        anonymous = enforced_auth_client.get(f"{ASSETS_PATH}/{name}")
        assert anonymous.status_code == 200
        assert anonymous.content == path.read_bytes()

        bad_key = enforced_auth_client.get(
            f"{ASSETS_PATH}/{name}", headers={"Authorization": "Bearer wrong-key"}
        )
        assert bad_key.status_code == 200

    def test_they_are_exempt_from_request_logging(self) -> None:
        """Their addresses are among the paths the request log ignores.

        Every documentation page load fetches all of them, and a log entry per
        script is noise no operator asked for.

        Ref: stdapi/monitoring.py:LOGGING_PATHS_IGNORE
        """
        assert ASSET_PATHS
        assert ASSET_PATHS <= LOGGING_PATHS_IGNORE

    @pytest.mark.parametrize("name", list(BROWSER_ASSETS))
    def test_a_fetched_copy_is_the_bytes_the_manifest_records(self, name: str) -> None:
        """Whatever this deployment fetched is what the recorded digest says it is.

        The build verifies each download before writing it, so this only fails
        where that never ran or where the file changed afterwards -- and it is
        the same assertion the container suite makes inside both images.
        """
        path = LOCAL_ASSETS.get(name)
        if path is None:
            pytest.skip("this checkout fetched no documentation assets")

        assert digest(path.read_bytes()) == BROWSER_ASSETS[name].sha256


class TestPackagedIcon:
    """Where the icon lives, which decides whether a deployment has it at all.

    Ref: stdapi/routes/core_docs.py:_FAVICON_RESPONSE
    """

    def test_ships_inside_the_installed_package(self) -> None:
        """The icon resolves as package data, not from the repository checkout.

        ``docs/`` is not part of the distribution: an icon read from there works
        in a source tree and 404s in the container images and in an installed
        wheel, which is every deployment there is.
        """
        icon = files("stdapi") / "favicon.svg"
        package = files("stdapi")

        assert icon.is_file()
        assert str(icon).startswith(str(package))
        assert icon.read_bytes().startswith(b"<svg")

    def test_is_the_mark_the_documentation_site_publishes(self) -> None:
        """The packaged icon is byte-identical to the documentation site's logo.

        The two are one brand mark. When ``docs/styles/logo.svg`` is redrawn,
        copy it over ``stdapi/favicon.svg`` so the API pages do not keep the
        previous one.

        Ref: mkdocs.yml
        """
        assert (files("stdapi") / "favicon.svg").read_bytes() == _SITE_LOGO.read_bytes()


def test_the_application_registers_no_builtin_documentation_page() -> None:
    """The served application leaves FastAPI's own ``/docs`` and ``/redoc`` unmounted.

    They hardcode the third-party icon URL and offer no way to change it, so the
    replacements only reach a reader while the built-ins stay unregistered.

    Ref: stdapi/main.py:app
    """
    from stdapi.main import app  # noqa: PLC0415

    assert app.docs_url is None
    assert app.redoc_url is None
