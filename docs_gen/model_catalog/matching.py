"""Maps leaderboard rows onto Amazon Bedrock model IDs.

Publishing a benchmark score against the wrong model is worse than publishing
none, so nothing is guessed. Three passes decide, and only the first two can
create a match:

1. deterministic rules over normalised names, provider and aliases;
2. an LLM, asked only about what the rules could not settle, through the running
   gateway, twice from opposite directions — a disagreement means no match;
3. human overrides, which pin both matches and rejections and which neither
   pass may overrule.

Leaderboards name configurations, not only models: ``claude-opus-4-6-thinking``
and ``gpt-image-2 (medium)`` are settings we cannot assume are the default. A
plain row always wins; when only configuration rows exist the most-voted one is
published and its exact name travels with the score so the reader sees what was
measured.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from docs_gen.model_catalog.config import (
    DEFAULT_MATCH_MODEL,
    MATCH_CACHE_PATH,
    MATCH_CONFIDENCE_FLOOR,
    OVERRIDES_PATH,
)
from docs_gen.model_catalog.http import FetchError, post_json

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from docs_gen.model_catalog.sources import RawReference, RawScore

#: Trailing Amazon Bedrock version markers that carry no identity.
_VERSION_SUFFIX: re.Pattern[str] = re.compile(r"[-:]v\d+([-:]\d+)?$")

#: Eight-digit release stamps Bedrock embeds in model IDs.
_DATE_STAMP: re.Pattern[str] = re.compile(r"[-_]?20\d{6}")

#: Bracketed or parenthesised qualifiers leaderboards append to a model name.
_BRACKETED: re.Pattern[str] = re.compile(r"[([{][^)\]}]*[)\]}]")

#: Everything that is not a name character, collapsed to a single separator.
_SEPARATORS: re.Pattern[str] = re.compile(r"[^a-z0-9]+")

#: Reasoning-effort and serving qualifiers that name a configuration, not a model.
VARIANT_TOKENS: frozenset[str] = frozenset(
    {
        "high",
        "low",
        "medium",
        "max",
        "xhigh",
        "minimal",
        "thinking",
        "nonthinking",
        "reasoning",
        "preview",
        "latest",
        "search",
        "grounding",
        "web",
        "chat",
        "quality",
        "fast",
        "turbo",
        "beta",
    }
)

#: Source organisation names accepted for each catalogue provider.
PROVIDER_ALIASES: dict[str, frozenset[str]] = {
    "AI21 Labs": frozenset({"ai21", "ai21labs"}),
    "Amazon": frozenset({"amazon", "aws", "amazonagi", "bedrock"}),
    "Anthropic": frozenset({"anthropic"}),
    "Cohere": frozenset({"cohere", "coherelabs"}),
    "DeepSeek": frozenset({"deepseek"}),
    "Google": frozenset({"google", "googledeepmind"}),
    "Luma AI": frozenset({"luma", "lumaai", "lumalabs"}),
    "Meta": frozenset({"meta", "metaai"}),
    "MiniMax": frozenset({"minimax"}),
    "Mistral AI": frozenset({"mistral", "mistralai"}),
    "Moonshot AI": frozenset({"moonshot", "moonshotai"}),
    "NVIDIA": frozenset({"nvidia"}),
    "OpenAI": frozenset({"openai"}),
    "Qwen": frozenset({"qwen", "alibaba", "alibabacloud"}),
    "Stability AI": frozenset({"stability", "stabilityai"}),
    "TwelveLabs": frozenset({"twelvelabs"}),
    "Writer": frozenset({"writer"}),
    "Z.AI": frozenset({"zai", "zhipu", "zhipuai"}),
    "Zhipu AI": frozenset({"zhipu", "zhipuai", "zai"}),
    "xAI": frozenset({"xai"}),
}

#: Every vendor token a source may prefix a model name with.
VENDOR_TOKENS: frozenset[str] = frozenset(
    token for aliases in PROVIDER_ALIASES.values() for token in aliases
)

#: Organisations that name the platform a model is served on, not its vendor.
PLATFORM_ORGANIZATIONS: frozenset[str] = frozenset({"bedrock", "awsbedrock"})

#: Largest candidate shortlist handed to the LLM for one model and board.
_SHORTLIST_LIMIT: int = 40

#: Appended to every system prompt so a model without structured output complies.
_JSON_INSTRUCTION: str = (
    'Reply with JSON only, shaped {"candidate": string or null, '
    '"confidence": number between 0 and 1, "reason": short string}. '
    "No prose, no code fence."
)

#: First JSON object in a reply, for models that wrap it in prose.
_JSON_OBJECT: re.Pattern[str] = re.compile(r"\{.*\}", re.DOTALL)

#: Response shape the matching model must return.
_MATCH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidate": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["candidate", "confidence", "reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class CatalogModel:
    """The identity facts the matcher needs about one catalogue model.

    Attributes:
        id: Amazon Bedrock model ID.
        name: Human-readable model name.
        provider: Model provider as the catalogue spells it.
        aliases: Alternate names the gateway accepts.
    """

    id: str
    name: str
    provider: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Decision:
    """How one model was matched on one board.

    Attributes:
        matched_name: Source name that was matched, or ``None`` when nothing was.
        method: ``rule``, ``llm``, ``override`` or ``none``.
        reason: Why, for the unmatched report and for review.
    """

    matched_name: str | None
    method: str
    reason: str


def normalise(value: str) -> str:
    """Reduce a model name to its comparable core, keeping what identifies it.

    Strips the vendor namespace and bracketed qualifiers, then collapses every
    separator. The release stamp is deliberately kept: two Bedrock models can
    differ only by it, and dropping it makes ``claude-3-5-sonnet-20240620`` and
    ``claude-3-5-sonnet-20241022`` indistinguishable.

    Args:
        value: Raw model name or ID.

    Returns:
        The normalised form, possibly empty.
    """
    text = _BRACKETED.sub(" ", value.lower())
    text = text.split("/")[-1]
    if "." in text and not text.startswith("v"):
        head, _, tail = text.partition(".")
        if head.isalpha() and tail:
            text = tail
    return _SEPARATORS.sub("-", text).strip("-")


def strict_forms(value: str) -> set[str]:
    """Return the forms of a name that still carry its full identity.

    Args:
        value: Raw model name or ID.

    Returns:
        Normalised forms, with and without a trailing Bedrock version marker.
    """
    base = normalise(value)
    if not base:
        return set()
    return _usable({base, _VERSION_SUFFIX.sub("", base)})


def loose_forms(value: str) -> set[str]:
    """Return the forms of a name with every optional marker removed.

    Drops the release stamp, the version marker and a leading vendor token, so
    ``cohere-embed-english-v3`` and ``cohere.embed-english-v3`` meet. A loose
    form is only ever trusted when no other model in the catalogue shares it.

    Args:
        value: Raw model name or ID.

    Returns:
        The loosened forms.
    """
    forms: set[str] = set()
    for form in strict_forms(value):
        stripped = _VERSION_SUFFIX.sub("", _DATE_STAMP.sub("", form)).strip("-")
        # Keep the unstripped form when stripping leaves nothing usable, or a
        # plain row becomes unreachable and only its variants can be matched.
        forms.add(stripped if _usable({stripped}) else form)
        head, _, tail = stripped.partition("-")
        if tail and head in VENDOR_TOKENS:
            forms.add(tail)
    return _usable(forms)


def _usable(forms: set[str]) -> set[str]:
    """Drop forms too weak to identify a model.

    Stripping a trailing version marker can eat the whole name: ``deepseek-v3.2``
    reduces to ``deepseek``, which every DeepSeek model would then answer to.
    A form that is only a vendor name, or barely a name at all, is discarded.

    Args:
        forms: Candidate forms.

    Returns:
        The forms that still identify something.
    """
    return {form for form in forms if len(form) > 1 and form not in VENDOR_TOKENS}


def split_variant(name: str) -> tuple[str, tuple[str, ...]]:
    """Separate a model's identity from the configuration a row names.

    Args:
        name: Normalised source name.

    Returns:
        The identity part, and the configuration tokens that were removed.
    """
    parts = name.split("-")
    variant: list[str] = []
    while len(parts) > 1 and parts[-1] in VARIANT_TOKENS:
        variant.insert(0, parts.pop())
    return "-".join(parts), tuple(variant)


def _raw_names(model: CatalogModel) -> set[str]:
    """Return every string that names a catalogue model.

    Args:
        model: The catalogue model.

    Returns:
        The ID, the display name and every alias.
    """
    return {model.id, model.name, *model.aliases}


def strict_keys(model: CatalogModel) -> set[str]:
    """Return the identity-preserving keys a catalogue model answers to.

    Args:
        model: The catalogue model.

    Returns:
        Strict normalised keys.
    """
    return {key for value in _raw_names(model) for key in strict_forms(value)}


def loose_keys(model: CatalogModel) -> set[str]:
    """Return the loosened keys a catalogue model could answer to.

    Args:
        model: The catalogue model.

    Returns:
        Loose normalised keys.
    """
    return {key for value in _raw_names(model) for key in loose_forms(value)}


def _provider_matches(provider: str, organization: str) -> bool:
    """Report whether a source's publisher is compatible with a catalogue provider.

    An empty organisation is treated as compatible: several sources leave it
    blank, and rejecting on a missing field would drop correct matches.

    Args:
        provider: Catalogue provider name.
        organization: Organisation the source attributes the model to.

    Returns:
        ``True`` when the two can name the same vendor.
    """
    if not organization:
        return True
    normalised = _SEPARATORS.sub("", organization.lower())
    if normalised in PLATFORM_ORGANIZATIONS:
        return True
    aliases = PROVIDER_ALIASES.get(provider)
    if aliases is None:
        return normalised == _SEPARATORS.sub("", provider.lower())
    return normalised in aliases


class Matcher:
    """Decides, and remembers, how leaderboard rows map onto catalogue models.

    Attributes:
        llm_calls: Number of chat completions issued during this run.
    """

    def __init__(
        self,
        *,
        gateway_url: str | None = None,
        api_key: str | None = None,
        model: str = DEFAULT_MATCH_MODEL,
        ambiguous_keys: frozenset[str] = frozenset(),
    ) -> None:
        """Load the override file and the decision cache.

        Args:
            gateway_url: Instance to ask when the rules cannot decide; ``None``
                disables the LLM pass entirely.
            api_key: Bearer token for that instance.
            model: Model used for matching.
            ambiguous_keys: Loose keys more than one catalogue model answers to,
                which may therefore never decide a match on their own.
        """
        self._ambiguous = ambiguous_keys
        self._gateway_url = gateway_url.rstrip("/") if gateway_url else None
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._model = model
        self._overrides: dict[str, Any] = (
            json.loads(OVERRIDES_PATH.read_text()) if OVERRIDES_PATH.is_file() else {}
        )
        self._cache: dict[str, Any] = (
            json.loads(MATCH_CACHE_PATH.read_text())
            if MATCH_CACHE_PATH.is_file()
            else {}
        )
        self.llm_calls = 0
        self.llm_failures = 0
        self.last_error: str | None = None
        self._structured = True

    def set_ambiguous_keys(self, keys: frozenset[str]) -> None:
        """Declare the loose keys that may not decide a match on their own.

        Args:
            keys: Loose keys more than one catalogue model answers to.
        """
        self._ambiguous = keys

    def save_cache(self) -> None:
        """Persist the decision cache so the next run asks about new rows only."""
        MATCH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        MATCH_CACHE_PATH.write_text(
            json.dumps(self._cache, indent=1, sort_keys=True) + "\n"
        )

    def match_board(
        self,
        model: CatalogModel,
        source: str,
        board: str,
        rows: Sequence[RawScore | RawReference],
    ) -> tuple[Decision, RawScore | RawReference | None]:
        """Pick the row of one board that belongs to one model.

        Args:
            model: Catalogue model to match.
            source: Source key.
            board: Sub-leaderboard key.
            rows: Every row published on that board.

        Returns:
            The decision, and the row it selected when there is one.
        """
        pinned = self._override_for(model.id, source, board)
        if pinned is not None:
            name, reason = pinned
            if name is None:
                return Decision(None, "override", reason), None
            chosen = next((row for row in rows if row.name == name), None)
            if chosen is None:
                return Decision(
                    None, "override", f"pinned name {name!r} is no longer published"
                ), None
            return Decision(name, "override", reason), chosen

        eligible = [
            row for row in rows if _provider_matches(model.provider, row.organization)
        ]
        chosen = self._rule_match(model, eligible)
        if chosen is not None:
            return Decision(chosen.name, "rule", "normalised name match"), chosen

        cache_key = f"{source}/{board}/{model.id}"
        cached = self._cache.get(cache_key)
        if cached is not None and cached.get("candidates") == _digest(eligible):
            name = cached["match"]
            chosen = next((row for row in eligible if row.name == name), None)
            method = "llm" if name else "none"
            return Decision(name, method, cached.get("reason", "cached")), chosen

        asked = self.llm_calls
        failed = self.llm_failures
        decision, chosen = self._llm_match(model, eligible)
        answered = self.llm_calls > asked and self.llm_failures == failed
        # A gateway that was down is not a model that does not exist: caching
        # that as "no match" makes a transient failure permanent.
        if self._gateway_url is not None and answered:
            # A run with the LLM pass disabled must not record "no match" as a
            # decision: the next run would serve it from the cache and never ask.
            self._cache[cache_key] = {
                "match": decision.matched_name,
                "reason": decision.reason,
                "candidates": _digest(eligible),
            }
        return decision, chosen

    def _override_for(
        self, model_id: str, source: str, board: str
    ) -> tuple[str | None, str] | None:
        """Return the pinned decision for one model and board, when any.

        Args:
            model_id: Amazon Bedrock model ID.
            source: Source key.
            board: Sub-leaderboard key.

        Returns:
            The pinned name — ``None`` for a pinned rejection — with its reason,
            or ``None`` when nothing is pinned.
        """
        entry = self._overrides.get(model_id, {}).get(f"{source}/{board}")
        if entry is None:
            return None
        return entry.get("match"), str(entry.get("reason", "pinned by hand"))

    def _rule_match(
        self, model: CatalogModel, rows: Sequence[RawScore | RawReference]
    ) -> RawScore | RawReference | None:
        """Match by normalised name, preferring a row that names no configuration.

        Identity-preserving keys are tried first. Loosened keys — no release
        stamp, no version marker, no vendor prefix — are only consulted when no
        other model in the catalogue answers to the same loose key, so two
        Bedrock models that differ only by their release date can never take
        each other's score.

        Args:
            model: Catalogue model to match.
            rows: Rows already filtered to a compatible publisher.

        Returns:
            The selected row, or ``None`` when the rules cannot decide.
        """
        strict = strict_keys(model)
        loose = {key for key in loose_keys(model) if key not in self._ambiguous}
        for keys, forms in ((strict, strict_forms), (loose, loose_forms)):
            if not keys:
                continue
            chosen = _select(rows, keys, forms)
            if chosen is not None:
                return chosen
        return None

    def _llm_match(
        self, model: CatalogModel, rows: Sequence[RawScore | RawReference]
    ) -> tuple[Decision, RawScore | RawReference | None]:
        """Ask the gateway to resolve what the rules could not.

        The same question is put twice, from opposite directions, and both
        answers must name the same row for a match to stand.

        Args:
            model: Catalogue model to match.
            rows: Rows already filtered to a compatible publisher.

        Returns:
            The decision, and the row it selected when there is one.
        """
        shortlist = _shortlist(model, rows) if self._gateway_url and rows else []
        if not shortlist:
            return Decision(
                None, "none", "no rule match and no plausible candidate"
            ), None

        forward = self._ask(
            "You map Amazon Bedrock model IDs onto leaderboard entries.",
            f"Bedrock model ID: {model.id}\n"
            f"Display name: {model.name}\n"
            f"Provider: {model.provider}\n\n"
            "Which of these leaderboard entries is the SAME model?\n"
            + "\n".join(f"- {name}" for name in shortlist)
            + "\n\nAnswer with the entry exactly as written, or null if none is the "
            "same model. A different size, generation or vendor is not the same model.",
        )
        candidate = str((forward or {}).get("candidate") or "")
        confidence = float((forward or {}).get("confidence") or 0.0)
        if not candidate:
            return Decision(
                None, "none", "the model was not recognised on this board"
            ), None
        if confidence < MATCH_CONFIDENCE_FLOOR:
            return Decision(
                None, "none", f"confidence {confidence:.2f} below floor"
            ), None
        if not self._confirms(model, candidate):
            return Decision(None, "none", "the two directions disagreed"), None

        chosen = next((row for row in rows if row.name == candidate), None)
        if chosen is None:
            return Decision(None, "none", "the answer named no published row"), None
        return Decision(
            candidate, "llm", str((forward or {}).get("reason", ""))[:200]
        ), chosen

    def _confirms(self, model: CatalogModel, candidate: str) -> bool:
        """Ask the same question backwards and report whether the answers agree.

        Args:
            model: Catalogue model the forward pass proposed a match for.
            candidate: Leaderboard entry the forward pass named.

        Returns:
            ``True`` when the reverse question lands back on the same model.
        """
        identifiers = {model.id, *model.aliases}
        reverse = self._ask(
            "You map leaderboard entries onto Amazon Bedrock model IDs.",
            f"Leaderboard entry: {candidate}\n\n"
            "Which of these Amazon Bedrock model IDs is the SAME model?\n"
            + "\n".join(f"- {value}" for value in sorted(identifiers))
            + "\n- none of these\n\nAnswer with the ID exactly as written, or null.",
        )
        return str((reverse or {}).get("candidate") or "") in identifiers

    def _ask(self, system: str, prompt: str) -> dict[str, Any] | None:
        """Put one structured question to the matching model.

        The first call asks for a JSON-schema response. Not every model on
        Bedrock accepts one — several first-party models reject the field
        outright — so a refusal downgrades the whole run to asking for JSON in
        the prompt instead, rather than silently returning no answer.

        Args:
            system: System prompt.
            prompt: User prompt.

        Returns:
            The decoded answer, or ``None`` when the call or the parse failed.
        """
        assert self._gateway_url is not None  # noqa: S101 -- guarded by the caller
        for structured in (True, False) if self._structured else (False,):
            self.llm_calls += 1
            payload: dict[str, Any] = {
                "model": self._model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": f"{system} {_JSON_INSTRUCTION}"},
                    {"role": "user", "content": prompt},
                ],
            }
            if structured:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "match",
                        "schema": _MATCH_SCHEMA,
                        "strict": True,
                    },
                }
            try:
                response = post_json(
                    f"{self._gateway_url}/v1/chat/completions",
                    payload,
                    headers=self._headers,
                )
                content = response["choices"][0]["message"]["content"]
            except (FetchError, KeyError, IndexError, TypeError) as error:
                self.llm_failures += 1
                self.last_error = str(error)[:300]
                if structured:
                    self._structured = False
                    continue
                return None
            return _parse_answer(content)
        return None


def _digest(rows: Sequence[RawScore | RawReference]) -> str:
    """Fingerprint the candidates a decision was taken against.

    The cache is committed, so it stores a digest rather than the candidate
    names themselves: the names are already in the snapshot, and repeating a
    few hundred of them per model turns every leaderboard update into an
    unreadable diff.

    Args:
        rows: Rows the decision considered.

    Returns:
        A short hex digest of the candidate names.
    """
    joined = "\n".join(sorted(row.name for row in rows))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def _parse_answer(content: object) -> dict[str, Any] | None:
    """Decode one answer from the matching model.

    Args:
        content: Message content the model returned.

    Returns:
        The decoded object, or ``None`` when the reply was not usable.
    """
    if not isinstance(content, str):
        return None
    match = _JSON_OBJECT.search(content)
    if match is None:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _select(
    rows: Sequence[RawScore | RawReference],
    keys: set[str],
    forms: Callable[[str], set[str]],
) -> RawScore | RawReference | None:
    """Pick the row of *rows* that best answers to *keys*.

    Args:
        rows: Candidate rows.
        keys: Keys the model answers to.
        forms: Function producing the comparable forms of a row's name.

    Returns:
        The row naming no configuration when one exists, otherwise the
        most-voted configuration row, otherwise ``None``.
    """
    plain: list[RawScore | RawReference] = []
    variants: list[RawScore | RawReference] = []
    for row in rows:
        candidate = forms(row.name)
        identities = {split_variant(form)[0] for form in candidate}
        if not (identities & keys):
            continue
        (plain if candidate & keys else variants).append(row)
    if plain:
        return max(plain, key=_representativeness)
    if variants:
        return max(variants, key=_representativeness)
    return None


def plain_identities(rows: Sequence[RawScore | RawReference]) -> set[str]:
    """Return the identities a board publishes without a configuration suffix.

    Args:
        rows: A board's rows.

    Returns:
        The normalised identities that have a plain row.
    """
    identities: set[str] = set()
    for row in rows:
        for form in strict_forms(row.name):
            identity, variant = split_variant(form)
            if not variant:
                identities.add(identity)
    return identities


def _representativeness(row: RawScore | RawReference) -> tuple[int, int]:
    """Rank rows by how much evidence stands behind them.

    Args:
        row: A candidate row.

    Returns:
        A sort key preferring the most-voted row, then the best-ranked one.
    """
    samples = getattr(row, "samples", None) or 0
    rank = getattr(row, "rank", None)
    # A ranked row must beat an unranked one, so an absent rank sorts last.
    return (int(samples), -int(rank) if rank else -(10**9))


def _shortlist(
    model: CatalogModel, rows: Iterable[RawScore | RawReference]
) -> list[str]:
    """Reduce a board to the rows plausibly naming one model.

    Args:
        model: Catalogue model to match.
        rows: Rows already filtered to a compatible publisher.

    Returns:
        Candidate names, capped so the prompt stays cheap.
    """
    tokens = {
        token for key in loose_keys(model) for token in key.split("-") if len(token) > 2
    }
    scored: list[tuple[int, str]] = []
    for row in rows:
        candidate = normalise(row.name)
        overlap = len(tokens & set(candidate.split("-")))
        if overlap:
            scored.append((overlap, row.name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in scored[:_SHORTLIST_LIMIT]]
