"""Reads the catalogue from a running stdapi.ai instance.

The generator is a client of the product's own public API: everything the page
shows about a model comes from ``search_models`` and ``model_pricing``, so a
catalogue the page cannot render is a finding about the API, not the generator.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from docs_gen.model_catalog.http import FetchError, get_json

if TYPE_CHECKING:
    from collections.abc import Iterator

#: Recorded version when an instance serves no OpenAPI document to read one from.
UNKNOWN_VERSION: str = "unknown"

#: How long to keep retrying ``model_pricing`` while the price catalog loads.
_PRICE_CATALOG_WAIT: float = 600.0

#: Delay between attempts while the price catalog is still loading, in seconds.
_PRICE_CATALOG_POLL: float = 10.0


class Gateway:
    """A running stdapi.ai instance the generator reads the catalogue from.

    Attributes:
        base_url: Root URL of the instance, without a trailing slash.
    """

    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        """Bind to an instance.

        Args:
            base_url: Root URL of the instance.
            api_key: Bearer token, when the instance has authentication enabled.
        """
        self.base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _get(self, path: str, **params: str) -> Any:  # noqa: ANN401
        """Fetch a JSON document from the instance.

        Args:
            path: Route path, starting with a slash.
            **params: Query-string parameters.

        Returns:
            The decoded response.
        """
        query = f"?{urlencode(params)}" if params else ""
        return get_json(f"{self.base_url}{path}{query}", headers=self._headers)

    def version(self) -> str:
        """Return the instance's advertised version.

        The version is only exposed on the OpenAPI document, which an instance
        may not serve. That is a reason to record the version as unknown, never
        a reason to fail the run.

        Returns:
            The version string, or :data:`UNKNOWN_VERSION`.
        """
        try:
            info = self._get("/openapi.json")["info"]
        except FetchError, KeyError, TypeError:
            return UNKNOWN_VERSION
        return str(info.get("version") or UNKNOWN_VERSION)

    def models(self) -> list[dict[str, Any]]:
        """Return every model, active and deprecated.

        ``search_models`` excludes deprecated models unless ``legacy=true`` is
        passed, and that flag returns *only* deprecated ones, so the full
        catalogue is the union of both calls.

        Returns:
            Model records, deprecated ones carrying ``legacy: True``.
        """
        active: list[dict[str, Any]] = self._get("/search_models")
        legacy: list[dict[str, Any]] = self._get("/search_models", legacy="true")
        for model in legacy:
            model["legacy"] = True
        return sorted([*active, *legacy], key=lambda model: str(model["id"]))

    def prices(self) -> list[dict[str, Any]]:
        """Return the full published price table for every model.

        ``all_prices=true`` returns every published row rather than only the
        ones this instance's configuration would bill, so the page is not a
        picture of one deployment's settings.

        Returns:
            One price card per model.

        Raises:
            FetchError: The price catalog never finished loading.
        """
        deadline = time.monotonic() + _PRICE_CATALOG_WAIT
        while True:
            try:
                result: list[dict[str, Any]] = self._get(
                    "/model_pricing", all_prices="true"
                )
            except FetchError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(_PRICE_CATALOG_POLL)
            else:
                return result


def iter_price_rows(card: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield the price rows of one model's price card.

    Args:
        card: A ``model_pricing`` entry.

    Yields:
        Each published price row.
    """
    yield from card.get("prices", ())
