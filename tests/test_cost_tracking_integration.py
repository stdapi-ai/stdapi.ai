"""Full integration tests for cost tracking: live catalog, pricing API, usage logs.

These load the real AWS Price List catalog (and make one tiny Bedrock call),
so they run only with ``--expensive``, like the pricing coverage test.
"""

from decimal import Decimal
from typing import TYPE_CHECKING

import pytest
from aiobotocore.session import get_session

from stdapi import monitoring, pricing
from stdapi.config import SETTINGS
from stdapi.pricing import Dimension, Service, resolve_price
from stdapi.usage import format_cost

if TYPE_CHECKING:
    from starlette.testclient import TestClient

    from stdapi.monitoring import EventLog

#: Small, widely available chat model used for the one live Bedrock call.
_CHAT_MODEL = "amazon.nova-lite-v1:0"

#: One live catalog load shared by all tests (reloads get throttled by AWS).
_CATALOG_CACHE: dict[pricing.PriceKey, pricing.Price] = {}


@pytest.fixture
def client(test_client: TestClient | None) -> TestClient:
    """Return the session test client, skipping if not running locally."""
    if test_client is None:
        pytest.skip("Requires local test server")
    return test_client


@pytest.fixture
async def live_catalog(
    test_client: TestClient | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enable cost tracking and load the real price catalog from AWS.

    The app's shared pricing client is bound to the server's event loop, so
    the load runs through a fresh client on this test loop instead. Skips
    before loading anything when not running against the local test server
    (same reason as the ``client`` fixture) -- otherwise a ``--server-url
    --expensive`` run would pay for a full Price List load just to skip.
    """
    if test_client is None:
        pytest.skip("Requires local test server")
    monkeypatch.setattr(SETTINGS, "cost_tracking", True)
    if not _CATALOG_CACHE:
        client_patch = pytest.MonkeyPatch()
        async with get_session().create_client(
            "pricing", region_name=pricing.pricing_endpoint_region()
        ) as fresh_client:
            client_patch.setattr(
                pricing, "get_client", lambda _service, _region=None: fresh_client
            )
            try:
                await pricing._load_price_catalog([])  # noqa: SLF001
            finally:
                client_patch.undo()
        _CATALOG_CACHE.update(pricing._state.price_index)  # noqa: SLF001
    else:
        pricing._state.price_index = dict(_CATALOG_CACHE)  # noqa: SLF001
    assert pricing._state.price_index  # noqa: SLF001


@pytest.mark.expensive
@pytest.mark.usefixtures("live_catalog")
@pytest.mark.xdist_group("cost_tracking_integration")
class TestCostTrackingIntegration:
    """End-to-end over the real AWS Price List catalog (no fixtures/stubs)."""

    def test_search_then_price_workflow(self, client: TestClient, api_key: str) -> None:
        """The documented agent workflow: search_models, then price the shortlist."""
        headers = {"Authorization": f"Bearer {api_key}"}
        found = client.get(
            "/search_models",
            params={
                "route": "openai_chat_completion",
                "legacy": "false",
                "region": "us-east-1",
            },
            headers=headers,
        )
        assert found.status_code == 200
        shortlist = [m["id"] for m in found.json()][:3]
        assert shortlist

        response = client.get(
            "/model_pricing",
            params={
                "model": shortlist,
                "region": "us-east-1",
                "dimension": ["input_tokens", "output_tokens"],
                "variants": "false",
            },
            headers=headers,
        )
        assert response.status_code == 200
        cards = response.json()
        assert [card["id"] for card in cards] == shortlist

        priced_rows = [row for card in cards for row in card["prices"]]
        assert priced_rows  # At least one shortlisted model has published prices.
        for row in priced_rows:
            assert row["region"] == "us-east-1"
            assert row["tier"] == "standard"
            assert row["dimension"] in ("input_tokens", "output_tokens")
            assert Decimal(row["unit_price"]) > 0

    def test_api_prices_match_resolve_price(
        self, client: TestClient, api_key: str
    ) -> None:
        """API rows are byte-identical to what request billing would resolve."""
        response = client.get(
            "/model_pricing",
            params={
                "model": _CHAT_MODEL,
                "region": "us-east-1",
                "dimension": "input_tokens",
                "variants": "false",
            },
            headers={"Authorization": f"Bearer {api_key}"},
        )
        (card,) = response.json()
        (row,) = card["prices"]
        billed = resolve_price(
            Service.BEDROCK, _CHAT_MODEL, "us-east-1", Dimension.INPUT_TOKENS
        )
        assert billed is not None
        assert row["unit_price"] == format_cost(billed.amount)
        assert row["currency"] == billed.currency

    def test_chat_completion_records_usage_and_cost(
        self, client: TestClient, api_key: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real Bedrock call lands in the request log with usage and a cost."""
        written: list[EventLog] = []
        monkeypatch.setattr(monitoring, "write_log_event", written.append)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": _CHAT_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
            },
            headers={"Authorization": f"Bearer {api_key}"},
        )
        assert response.status_code == 200
        assert response.json()["usage"]["prompt_tokens"] > 0

        (request_log,) = [w for w in written if w.get("type") == "request"]
        (entry,) = request_log["usage"]
        assert entry["service"] == "bedrock-runtime"
        assert entry["input_tokens"] > 0
        assert Decimal(entry["cost"]) > 0
        assert request_log["cost"] == {entry["currency"]: entry["cost"]}
