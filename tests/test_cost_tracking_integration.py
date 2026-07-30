"""End-to-end cost tracking over the real AWS Price List catalog: pricing routes and usage logs.

The prices the ``/model_pricing`` route publishes and the prices request
billing resolves come from the same catalog, so the two must agree
byte-for-byte. These tests load that catalog for real (and make one tiny
Bedrock call), so they run only with ``--slow``.

Ref: stdapi/pricing.py:_load_price_catalog
     stdapi/routes/core_models.py:model_pricing
     stdapi/usage.py:compute_costs
"""

from decimal import Decimal
from typing import TYPE_CHECKING

import pytest
import pytest_asyncio
from aiobotocore.session import get_session

from stdapi import monitoring, pricing
from stdapi.config import SETTINGS
from stdapi.pricing import Dimension, Service, resolve_price
from stdapi.usage import format_cost

if TYPE_CHECKING:
    from starlette.testclient import TestClient

    from stdapi.monitoring import EventLog

#: Small, widely available chat model used for the one live Bedrock call.
_CHAT_MODEL = "amazon.nova-micro-v1:0"


@pytest.fixture
def client(test_client: TestClient | None) -> TestClient:
    """Return the session test client, skipping if not running locally."""
    if test_client is None:
        pytest.skip("Requires local test server")
    return test_client


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def price_catalog(
    test_client: TestClient | None,
) -> dict[pricing.PriceKey, pricing.Price]:
    """Load the real AWS Price List catalog once for the whole module.

    Reloads get throttled by AWS, so this pays for exactly one load. The app's
    shared pricing client is bound to the server's event loop, so the load runs
    through a fresh client here instead. Skips before loading anything when not
    running against the local test server -- otherwise a ``--server-url --slow``
    run would pay for a full Price List load just to skip.

    Returns:
        The loaded price index.
    """
    if test_client is None:
        pytest.skip("Requires local test server")
    with pytest.MonkeyPatch.context() as patch:
        async with get_session().create_client(
            "pricing", region_name=pricing.pricing_endpoint_region()
        ) as fresh_client:
            patch.setattr(
                pricing, "get_client", lambda _service, _region=None: fresh_client
            )
            await pricing._load_price_catalog([])  # noqa: SLF001
    catalog = dict(pricing._state.price_index)  # noqa: SLF001
    assert catalog
    return catalog


@pytest.fixture
def live_catalog(
    price_catalog: dict[pricing.PriceKey, pricing.Price],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enable cost tracking and install the loaded catalog for one test.

    Reinstalling per test is required: conftest's autouse ``_clean_price_index``
    empties the index before every test.
    """
    monkeypatch.setattr(SETTINGS, "cost_tracking", True)
    monkeypatch.setattr(pricing._state, "price_index", dict(price_catalog))  # noqa: SLF001


@pytest.mark.slow
@pytest.mark.usefixtures("live_catalog")
@pytest.mark.xdist_group("cost_tracking_integration")
class TestCostTrackingIntegration:
    """End-to-end over the real AWS Price List catalog (no seeded prices).

    Ref: stdapi/routes/core_models.py:model_pricing
         stdapi/pricing.py:resolve_price
    """

    def test_search_then_price_workflow(self, client: TestClient, api_key: str) -> None:
        """search_models then model_pricing returns the shortlist in order, priced per Region.

        The documented agent workflow: discover models for a route, then price
        the shortlist. Price cards come back one per requested model, in the
        requested order, carrying only the requested Region, standard tier and
        requested dimensions.

        Ref: stdapi/routes/core_models.py:search_models
        """
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
        """A published price row is byte-identical to the price request billing resolves.

        Both go through ``resolve_price`` and ``format_cost``, so a formatting
        or fallback divergence between the route and billing would show up as
        an unequal ``unit_price`` string.

        Ref: stdapi/usage.py:format_cost
        """
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
        assert response.status_code == 200
        (card,) = response.json()
        (row,) = card["prices"]
        assert row["dimension"] == "input_tokens"
        billed = resolve_price(
            Service.BEDROCK, _CHAT_MODEL, "us-east-1", Dimension.INPUT_TOKENS
        )
        assert billed is not None
        assert row["unit_price"] == format_cost(billed.amount)
        assert row["currency"] == billed.currency

    def test_chat_completion_records_usage_and_cost(
        self, client: TestClient, api_key: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A real Bedrock call logs one bedrock-runtime usage entry with a priced cost.

        The request-level ``cost`` rollup is a per-currency total of the
        entries, so with a single entry it must equal that entry exactly.

        Ref: stdapi/usage.py:total_costs_by_currency
             stdapi/monitoring.py:write_log_event
        """
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
        reported_usage = response.json()["usage"]
        assert reported_usage["prompt_tokens"] > 0

        (request_log,) = [w for w in written if w.get("type") == "request"]
        (entry,) = request_log["usage"]
        assert entry["service"] == "bedrock-runtime"
        assert entry["model"] == _CHAT_MODEL
        assert entry["input_tokens"] > 0
        assert Decimal(entry["cost"]) > 0
        assert request_log["cost"] == {entry["currency"]: entry["cost"]}
        # The log records Bedrock's own counters, where inputTokens excludes
        # cache reads/writes, while OpenAI's prompt_tokens includes them.
        cached = (reported_usage.get("prompt_tokens_details") or {}).get(
            "cached_tokens", 0
        )
        assert reported_usage["prompt_tokens"] == (
            entry["input_tokens"] + cached + entry.get("cache_write_tokens", 0)
        )
