"""Model facts from models.dev, an open database of AI models, under MIT.

Amazon publishes a model's context window only through ``consoleIDEMetadata``,
which it does not return to an ordinary caller, and the gateway does not track
it either. Context window is a first-order buying criterion, so it comes from
here instead — along with the knowledge cutoff, tool-calling and reasoning
support, and the output ceiling for the models AWS leaves blank.

The join is exact: models.dev keys its Amazon Bedrock entries by Bedrock model
ID, so nothing here is guessed. Entries whose ID carries a cross-region
geography prefix describe the same model, and the unprefixed entry wins when
both exist.
"""

from __future__ import annotations

from typing import Any

from docs_gen.model_catalog.http import get_json
from docs_gen.model_catalog.sources import snapshot
from docs_gen.model_catalog.tokens import format_tokens

#: The whole database, one document.
_API_URL: str = "https://models.dev/api.json"

#: Provider whose entries are keyed by Amazon Bedrock model ID.
_PROVIDER: str = "amazon-bedrock"

#: Cross-region inference geographies models.dev prefixes some IDs with.
_GEOGRAPHY_PREFIXES: frozenset[str] = frozenset(
    {"us", "eu", "apac", "jp", "au", "ca", "global", "us-gov"}
)


def _collect() -> dict[str, dict[str, Any]]:
    """Download the Amazon Bedrock section of the database.

    Returns:
        Model ID to that model's published facts.
    """
    payload = get_json(_API_URL)
    provider = payload.get(_PROVIDER) or {}
    models = provider.get("models") or {}
    return {str(key): value for key, value in models.items()}


def _base_id(model_id: str) -> str:
    """Strip a cross-region geography prefix from a Bedrock model ID.

    Args:
        model_id: An ID as models.dev spells it.

    Returns:
        The ID without its geography prefix.
    """
    head, _, tail = model_id.partition(".")
    return tail if tail and head in _GEOGRAPHY_PREFIXES else model_id


def fetch(*, refresh: bool = False) -> dict[str, dict[str, Any]]:
    """Read the facts, keyed by the Bedrock model ID they describe.

    Args:
        refresh: Ignore any cached snapshot.

    Returns:
        Bedrock model ID to the facts this source contributes. Later entries do
        not overwrite an unprefixed one, so a plain ID always wins.
    """
    raw = snapshot("models_dev", _collect, refresh=refresh)
    assert isinstance(raw, dict)  # noqa: S101 -- snapshot round-trips its own JSON
    facts: dict[str, dict[str, Any]] = {}
    for model_id, entry in sorted(raw.items(), key=lambda item: len(item[0])):
        if not isinstance(entry, dict):
            continue
        base = _base_id(model_id)
        if base in facts:
            continue
        limit = entry.get("limit") or {}
        # An output ceiling equal to the context window is this database's
        # placeholder for "unknown": no model can answer with its whole input.
        output = limit.get("output")
        if not isinstance(output, int) or output == limit.get("context"):
            output = None
        contributed = {
            "context_window": format_tokens(limit.get("context")),
            "max_output_tokens": output,
            "knowledge_cutoff": str(entry["knowledge"])
            if entry.get("knowledge")
            else None,
            "reasoning": entry.get("reasoning")
            if isinstance(entry.get("reasoning"), bool)
            else None,
            "tool_call": entry.get("tool_call")
            if isinstance(entry.get("tool_call"), bool)
            else None,
            "open_weights": entry.get("open_weights")
            if isinstance(entry.get("open_weights"), bool)
            else None,
        }
        facts[base] = {
            key: value for key, value in contributed.items() if value is not None
        }
    return facts
