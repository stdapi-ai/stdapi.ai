"""The interactive documentation pages and the icon the gateway brands them with.

FastAPI's built-in ``/docs`` and ``/redoc`` write a ``fastapi.tiangolo.com`` URL
into every page, so opening either one sends a browser to a third party -- which
an air-gapped or egress-restricted deployment cannot do at all, and which no
other response the gateway produces does. The pages are served here instead, from
an icon the package ships.

The pages are disabled in the test configuration, so the ones under test are
built from a private copy of the route module re-executed with them enabled --
never a reload, which would replace the running application's own routes.

Ref: https://github.com/stdapi-ai/stdapi.ai/issues/184
     stdapi/routes/core_docs.py
"""

from __future__ import annotations

from importlib.resources import files
from importlib.util import find_spec, module_from_spec
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from fastapi import FastAPI

from stdapi.config import SETTINGS
from stdapi.monitoring import LOGGING_PATHS_IGNORE
from stdapi.routes.core_docs import FAVICON_PATH

if TYPE_CHECKING:
    from starlette.testclient import TestClient

pytestmark = pytest.mark.local

#: Host FastAPI's built-in documentation pages load their icon from.
_THIRD_PARTY_HOST = "fastapi.tiangolo.com"

#: Title the documentation pages under test name the deployment with.
_TITLE = "Test Gateway"

#: Root path a deployment behind a proxy or a mount prefix serves the whole app below.
_ROOT_PATH = "/gateway"

#: The published documentation site's mark, which the packaged icon has to be.
_SITE_LOGO = Path(__file__).parents[1] / "docs" / "styles" / "logo.svg"


@pytest.fixture
def docs_app(monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    """An application serving both documentation pages, whatever the settings say.

    Built like the served one: the schema to document, a title to name it by,
    and FastAPI's own pages left unregistered so the replacements answer.

    Returns:
        An application holding a private copy of the routes.
    """
    monkeypatch.setattr(SETTINGS, "enable_docs", True)
    monkeypatch.setattr(SETTINGS, "enable_redoc", True)
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
def docs_client(docs_app: FastAPI) -> TestClient:
    """Client for the application serving both documentation pages."""
    from starlette.testclient import TestClient  # noqa: PLC0415

    return TestClient(docs_app)


@pytest.fixture
def mounted_docs_client(docs_app: FastAPI) -> TestClient:
    """Client for the same application mounted under a non-empty root path."""
    from starlette.testclient import TestClient  # noqa: PLC0415

    return TestClient(docs_app, root_path=_ROOT_PATH)


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
