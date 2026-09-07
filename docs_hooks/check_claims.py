"""Build-time guard against documentation claims that are stale or banned.

Scans every surface the project publishes — the Markdown, HTML overrides and SVG
assets under ``docs/``, and the repository ``README.md`` — for two classes of
drift that review keeps missing:

* **Stale phrasings**: enumerations and counts that were true at one release and
  quietly stopped being true, such as the API dialect list or a second page
  naming the latest version.
* **Banned claims**: the "never write" column of the ``stdapi-marketing`` skill,
  each of which is unsupportable rather than merely unfashionable.

Every surface is read as a reader and a crawler read it: character entities are
decoded, so one rule catches a phrasing whatever spelling the surface needs, and
on the markup surfaces tags, their attributes, style and script blocks and
template constructs are blanked out first, so an ``id``, a ``class``, a CSS
property or a URL never counts as a claim. The JSON-LD script block is kept: it
is the description a search engine and an LLM read.

Run it with ``uv run python -m docs_hooks.check_claims``. It prints
``path:line`` for every hit and exits non-zero, so CI fails on the drift instead
of a reader finding it. Claims that cannot be judged by pattern are deliberately
absent: ``optimized`` is the value of an AWS latency header, "automatic failover"
is banned only when unqualified, and "WAF included" depends on which Terraform
variables a deployment sets.

A hit that is genuinely correct is silenced by adding its ``(rule, path)`` pair
to ``_EXCEPTIONS`` with the reason, so every exception is reviewed like code.
"""

import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

#: Repository root, derived from this file's location.
_ROOT: Path = Path(__file__).resolve().parent.parent

#: Glob patterns, relative to the repository root, of the surfaces to scan.
_SURFACES: tuple[str, ...] = (
    "docs/**/*.md",
    "docs/**/*.html",
    "docs/**/*.svg",
    "README.md",
)

#: Suffixes whose text content is reached through markup rather than read as-is.
_MARKUP_SUFFIXES: tuple[str, ...] = (".html", ".svg")

#: Markup that carries no prose: comments, style and script blocks other than the
#: JSON-LD a crawler reads, tags with their attributes, and Jinja tags.
_MARKUP_RE: re.Pattern[str] = re.compile(
    r"<!--.*?-->"
    r"|<style\b[^>]*>.*?</style\s*>"
    r"|<script(?![^>]*ld\+json)\b[^>]*>.*?</script\s*>"
    r"|<[^>]*>"
    r"|\{\{.*?\}\}"
    r"|\{[%#].*?[%#]\}",
    re.DOTALL | re.IGNORECASE,
)

#: Character entities markup escapes text with, decoded before the rules run.
_ENTITIES: dict[str, str] = {
    "&amp;": "&",
    "&#38;": "&",
    "&quot;": '"',
    "&apos;": "'",
    "&lt;": "<",
    "&gt;": ">",
}

#: Alternation matching any entity of ``_ENTITIES``, decoded in a single pass.
_ENTITY_RE: re.Pattern[str] = re.compile(
    "|".join(re.escape(entity) for entity in _ENTITIES), re.IGNORECASE
)


class _Rule(NamedTuple):
    """One phrasing the documentation must not ship.

    Attributes:
        name: Short identifier printed with every hit.
        pattern: Regex whose match in a surface fails the check.
        message: What to write instead, in one line.
        exempt: Paths, relative to the repository root, the rule does not apply to.
        allowed_in: Regex which, matched on the hit's own line, clears the hit.
        stale: Predicate on a match which, when given, must hold for it to be a hit.
    """

    name: str
    pattern: re.Pattern[str]
    message: str
    exempt: tuple[str, ...] = ()
    allowed_in: re.Pattern[str] | None = None
    stale: Callable[[re.Match[str]], bool] | None = None


#: Published floor for each counted noun, from the marketing skill's canonical facts.
_FLOORS: dict[str, int] = {
    "models": 100,
    "providers": 20,
    "tools": 90,
    "endpoints": 100,
    "operations": 100,
    "tests": 8000,
    "probes": 100,
}

#: A published floor, as ``<number>+`` and the counted noun it qualifies. The
#: boundaries exclude letters and digits rather than using ``\b``, whose word
#: class contains the underscore: ``__6,000+ test cases__`` is bold in Markdown
#: and was read as one long word, so the claim escaped the check entirely.
_COUNT_RE: re.Pattern[str] = re.compile(
    r"(?<![A-Za-z0-9])(?P<value>\d[\d,]*)\+\s*(?:[\w-]+\s+){0,3}?"
    r"(?:(?P<models>models)|(?P<providers>providers)|(?P<tools>tools)"
    r"|(?P<endpoints>endpoints)|(?P<operations>operations)"
    r"|(?P<tests>tests|test\s+cases)|(?P<probes>probe\s+records))"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def _below_floor(match: re.Match[str]) -> bool:
    """Judge a published count against the floor for the noun it counts.

    Args:
        match: A ``_COUNT_RE`` match, whose set named group names the noun.

    Returns:
        ``True`` when the surface publishes a number below that floor.
    """
    noun = next(
        name for name, group in match.groupdict().items() if group and name != "value"
    )
    return int(match["value"].replace(",", "")) < _FLOORS[noun]


#: Phrasings that fail the check, with the wording that replaces each one.
_RULES: tuple[_Rule, ...] = (
    _Rule(
        "dialect-count",
        re.compile(r"\bthree\s+(?:\w+\s+){0,2}dialects\b", re.IGNORECASE),
        "the gateway serves four dialects: OpenAI, Anthropic, Cohere and Ollama",
    ),
    _Rule(
        "dialect-list",
        re.compile(r"\bOpenAI-?,\s*Anthropic-?\s*,?\s*(?:and|&)\s*Cohere\b"),
        "name Ollama as well, or link the page that lists the dialects",
        # The vendors' own APIs and SDKs really are three: that is the
        # cross-validation lane, not the list of dialects the gateway serves.
        allowed_in=re.compile(r"\breal\b|\bgenuine\b|\bofficial\b|\bSDKs\b"),
    ),
    _Rule(
        "latest-version",
        re.compile(r"^\s*\*{0,2}Latest:\s*v\d", re.MULTILINE),
        "exactly one page names the latest version, and it is docs/roadmap.md",
        exempt=("docs/roadmap.md",),
    ),
    _Rule(
        "stale-count",
        _COUNT_RE,
        "use the published floor from the marketing skill's canonical facts",
        stale=_below_floor,
    ),
    _Rule(
        "data-never-leaves",
        re.compile(r"never leaves (?:your|the) (?:AWS|account|cloud)", re.IGNORECASE),
        "S3 I/O, web grounding and Nova system tools egress: write "
        "'no third party sits between your users and your models'",
    ),
    _Rule(
        "inherited-compliance",
        re.compile(
            r"stdapi\.ai is (?:HIPAA|FedRAMP|ISO|SOC|PCI)"
            r"|inherits?\s+(?:the\s+)?(?:AWS|Amazon)[^.\n]{0,40}"
            r"(?:certification|compliance)",
            re.IGNORECASE,
        ),
        "stdapi.ai is not certified: certifications apply to the AWS services and "
        "regions you choose and are not inherited",
    ),
    _Rule(
        "legal-conclusion",
        re.compile(r"mitigat\w*[^.\n]{0,40}(?:CLOUD Act|FISA)", re.IGNORECASE),
        "describe the KMS or region control, never its legal effect",
    ),
    _Rule(
        "quota-multiplier",
        re.compile(
            r"\b\d+\s*[x\u00d7]\s*(?:tokens|quota|throughput)\b|\bn\u00d7\s*quota\b"
        ),
        "each region you enable adds its own quota; a multiple is not guaranteed",
    ),
    _Rule(
        "zero-errors",
        re.compile(r"(?:\b0|\bzero)\s+errors\b", re.IGNORECASE),
        "retry is conditional: 'eligible failures retry in another enabled region'",
    ),
    _Rule(
        "time-to-production",
        re.compile(r"minutes? to production|minutes? using Terraform", re.IGNORECASE),
        "time to production depends on domain, certificate and model access: "
        "write 'two Terraform commands'",
    ),
    _Rule(
        "zero-configuration",
        re.compile(
            r"zero[- ]configuration|0 required configuration|no configuration required",
            re.IGNORECASE,
        ),
        "write 'secure defaults' rather than a claim of no configuration",
    ),
    _Rule(
        "no-code-changes",
        re.compile(
            r"(?:zero|no)\s+code\s+changes|your code never changes|entire migration",
            re.IGNORECASE,
        ),
        "adoption is quick: point the client at your gateway, and name a different "
        "model only where the name differs",
    ),
    _Rule(
        "absolute-compatibility",
        re.compile(
            r"connects? instantly|works? instantly|\bany tool can\b"
            r"|(?:works?|integrates?|connects?|compatible)\s+with\s+any\s+tool",
            re.IGNORECASE,
        ),
        "write 'standard SDKs connect on the base URL alone'",
    ),
    _Rule(
        "cost-accuracy",
        re.compile(r"(?:exact|real-time|1:1)[- ]cost", re.IGNORECASE),
        "cost tracking is opt-in and an estimate: 'estimated from published AWS "
        "prices, not read back from your invoice'",
    ),
    _Rule(
        "empty-adjective",
        re.compile(
            r"\bseamless(?:ly)?\b|\benterprise[- ]grade\b|\bpowerful\b", re.IGNORECASE
        ),
        "state the specific fact, or nothing",
    ),
    _Rule(
        "social-proof",
        re.compile(
            r"\btestimonials?\b|\btrusted by\b|\b\d+\+?\s+(?:happy\s+)?customers\b",
            re.IGNORECASE,
        ),
        "no customer proof exists: cite the test coverage instead",
    ),
    _Rule(
        "continuity-risk",
        re.compile(
            r"one[- ]person project|solo maintainer|\bsmall team\b", re.IGNORECASE
        ),
        "give the scope and the published response time, and the AGPL edition as "
        "the escape hatch",
    ),
)

#: Hits accepted as correct, as ``(rule name, path)`` pairs, each with its reason.
_EXCEPTIONS: frozenset[tuple[str, str]] = frozenset(
    {
        # Release entries record what was true at that release and are never
        # rewritten, so both the dialect count and the wording of a past entry
        # stay as published.
        ("dialect-count", "docs/roadmap.md"),
        ("empty-adjective", "docs/roadmap.md"),
    }
)


def _blank(match: re.Match[str]) -> str:
    """Erase a markup construct, keeping the line structure around it.

    Args:
        match: The markup to erase.

    Returns:
        The match with every character but its newlines replaced by a space.
    """
    return re.sub(r"[^\n]", " ", match.group())


def _text_content(text: str, *, markup: bool) -> str:
    """Reduce a surface to the text a reader and a crawler see.

    Args:
        text: The surface's source.
        markup: Whether tags, style and script blocks and template constructs
            have to be blanked out first.

    Returns:
        The text content, character entities decoded, at unchanged line numbers.
    """
    if markup:
        text = _MARKUP_RE.sub(_blank, text)
    return _ENTITY_RE.sub(lambda hit: _ENTITIES[hit.group().lower()], text)


def _surfaces() -> list[Path]:
    """Collect the surfaces the rules apply to.

    Returns:
        Sorted absolute paths of every scanned file.
    """
    found: set[Path] = set()
    for pattern in _SURFACES:
        found.update(path for path in _ROOT.glob(pattern) if path.is_file())
    return sorted(found)


def scan(text: str, relative: str) -> list[tuple[int, str]]:
    """Match every rule that applies to one surface against its text.

    Args:
        text: The surface's source.
        relative: Its path relative to the repository root, which decides
            which rules and exceptions apply and how the source is read.

    Returns:
        One ``(line, "[rule] match — message")`` pair per hit.
    """
    hits: list[tuple[int, str]] = []
    text = _text_content(text, markup=relative.endswith(_MARKUP_SUFFIXES))
    lines = text.splitlines()
    for rule in _RULES:
        if relative in rule.exempt or (rule.name, relative) in _EXCEPTIONS:
            continue
        for match in rule.pattern.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            if rule.allowed_in and rule.allowed_in.search(lines[line - 1]):
                continue
            if rule.stale and not rule.stale(match):
                continue
            hits.append(
                (line, f"[{rule.name}] {match.group().strip()!r} — {rule.message}")
            )
    return hits


def check() -> list[str]:
    """Match every rule against every surface.

    Returns:
        One ``path:line: [rule] match — message`` string per hit, in file order.
    """
    hits: list[tuple[str, int, str]] = []
    for path in _surfaces():
        relative = path.relative_to(_ROOT).as_posix()
        hits.extend(
            (relative, line, detail)
            for line, detail in scan(path.read_text(encoding="utf-8"), relative)
        )
    return [f"{path}:{line}: {detail}" for path, line, detail in sorted(hits)]


def main() -> int:
    """Report every stale or banned claim found in the documentation.

    Returns:
        ``1`` when at least one rule matched, otherwise ``0``.
    """
    failures = check()
    for failure in failures:
        print(failure)
    files = len({failure.split(":", 1)[0] for failure in failures})
    print(
        f"\n{len(failures)} claim check failure(s) in {files} file(s); "
        "fix the wording, or add the pair to _EXCEPTIONS with its reason."
        if failures
        else f"No stale or banned claims in {len(_surfaces())} published surfaces."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
