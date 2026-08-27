"""MkDocs hook attributing the third-party trademarks each page actually uses.

Holds the single canonical mark-to-owner registry for the whole site. For every
page, it derives which entries the page really references — from the brand logo
assets it embeds and from the product names it spells out — and exposes them as
``page.meta["trademarks"]`` for ``partials/copyright.html`` to render in the
page footer.

Nothing has to be declared per page: a page that later gains a logo or a product
name gains its attribution on the next build. A brand logo asset with no registry
entry raises a build warning, which ``mkdocs build --strict`` turns into an error,
so an unattributed mark cannot ship silently.
"""

import logging
import re
from pathlib import Path
from typing import Any, NamedTuple

#: Logger under the MkDocs namespace, so ``--strict`` fails the build on warnings.
_LOG: logging.Logger = logging.getLogger("mkdocs.hooks.trademarks")


class _Entry(NamedTuple):
    """One canonical trademark registry entry.

    Attributes:
        label: Short form rendered in the page footer.
        marks: Individual marks listed on the trademarks page.
        owner: Legal owner, exactly as it must be attributed everywhere.
        pattern: Regex whose match in a page's Markdown means the page uses a mark.
        assets: Stems of ``docs/styles/logo_<stem>.svg`` files covered by this entry.
    """

    label: str
    marks: tuple[str, ...]
    owner: str
    pattern: str
    assets: tuple[str, ...] = ()


#: Canonical mark-to-owner registry: the single source of truth for the whole site.
_REGISTRY: tuple[_Entry, ...] = (
    _Entry(
        label="AWS, Amazon and Amazon product names and logos",
        marks=(
            "Amazon",
            "Amazon Web Services",
            "AWS",
            "Amazon Bedrock",
            "Amazon Nova",
            "Amazon Titan",
            "Amazon Polly",
            "Amazon Transcribe",
            "Amazon Translate",
            "Amazon Comprehend",
            "Amazon SageMaker AI",
            "Amazon S3",
            "Amazon DynamoDB",
            "Amazon CloudWatch",
            "Amazon Cognito",
            "Amazon ElastiCache",
            "Amazon OpenSearch Service",
            "Amazon Aurora",
            "Amazon ECS",
            "Amazon EFS",
            "Amazon SQS",
            "Amazon VPC",
            "AWS Fargate",
            "AWS Marketplace",
            "AWS Secrets Manager",
            "AWS Systems Manager",
            "AWS WAF",
            "AWS X-Ray",
            "AWS Global Accelerator",
            "AWS Key Management Service",
            "Elastic Load Balancing",
        ),
        owner="Amazon.com, Inc. or its affiliates",
        pattern=r"\bAmazon\b|\bAWS\b|:material-aws:",
        assets=(
            "amazon",
            "amazon_aurora",
            "amazon_bedrock",
            "amazon_cloudwatch",
            "amazon_cognito",
            "amazon_comprehend",
            "amazon_dynamodb",
            "amazon_ecs",
            "amazon_efs",
            "amazon_elasticache",
            "amazon_global_accelerator",
            "amazon_load_balancing",
            "amazon_marketplace",
            "amazon_nova",
            "amazon_opensearch",
            "amazon_polly",
            "amazon_s3",
            "amazon_sagemaker",
            "amazon_secrets_manager",
            "amazon_sqs",
            "amazon_systems_manager",
            "amazon_transcribe",
            "amazon_translate",
            "amazon_vpc",
            "amazon_waf",
            "amazon_xray",
        ),
    ),
    _Entry(
        label="AI21 Labs, Jamba",
        marks=("AI21 Labs", "Jamba"),
        owner="AI21 Labs Ltd.",
        pattern=r"\bAI21\b|\bJamba\b",
        assets=("ai21",),
    ),
    _Entry(
        label="Alibaba Cloud, Qwen",
        marks=("Alibaba Cloud", "Qwen"),
        owner="Alibaba Group Holding Limited",
        pattern=r"\bAlibaba\b|\bQwen\b",
        assets=("alibaba", "qwen"),
    ),
    _Entry(
        label="Anthropic, Claude",
        marks=("Anthropic", "Claude"),
        owner="Anthropic PBC",
        pattern=r"\bAnthropic\b|\bClaude\b",
        assets=("anthropic", "anthropic_claude"),
    ),
    _Entry(
        label="Cohere",
        marks=("Cohere",),
        owner="Cohere Inc.",
        pattern=r"\bCohere\b",
        assets=("cohere",),
    ),
    _Entry(
        label="DeepSeek",
        marks=("DeepSeek",),
        owner="DeepSeek",
        pattern=r"\bDeepSeek\b",
        assets=("deepSeek",),
    ),
    _Entry(
        label="Docker",
        marks=("Docker",),
        owner="Docker, Inc.",
        pattern=r"\bDocker\b|:material-docker:",
    ),
    _Entry(
        label="Docling",
        marks=("Docling",),
        owner="the Docling project, a project of the LF AI & Data Foundation",
        pattern=r"\bDocling\b|\bdocling\b",
        assets=("docling",),
    ),
    _Entry(
        label="GitHub", marks=("GitHub",), owner="GitHub, Inc.", pattern=r"\bGitHub\b"
    ),
    _Entry(
        label="Google, Gemini, Gemma",
        marks=("Google", "Gemini", "Gemma"),
        owner="Google LLC",
        pattern=r"\bGoogle\b|\bGemini\b|\bGemma\b",
        assets=("google",),
    ),
    _Entry(
        label="Hermes",
        marks=("Hermes",),
        owner="Nous Research",
        pattern=r"\bHermes\b",
        assets=("hermes_agent",),
    ),
    _Entry(
        label="Home Assistant, Wyoming",
        marks=("Home Assistant", "Wyoming"),
        owner="the Open Home Foundation",
        pattern=r"\bHome Assistant\b|\bWyoming\b",
        assets=("home_assistant",),
    ),
    _Entry(
        label="Kubernetes",
        marks=("Kubernetes",),
        owner="The Linux Foundation",
        pattern=r"\bKubernetes\b",
    ),
    _Entry(
        label="LobeHub, LobeChat",
        marks=("LobeHub", "LobeChat"),
        owner="LobeHub",
        pattern=r"\bLobeHub\b|\bLobeChat\b|\blobehub\b",
        assets=("lobehub",),
    ),
    _Entry(
        label="Luma AI",
        marks=("Luma AI", "Ray"),
        owner="Luma AI, Inc.",
        pattern=r"\bLuma\b",
        assets=("luma",),
    ),
    _Entry(
        label="Meta, Llama",
        marks=("Meta", "Llama"),
        owner="Meta Platforms, Inc.",
        pattern=r"\bLlama\b|\bMeta AI\b|\bMeta Platforms\b",
        assets=("meta",),
    ),
    _Entry(
        label="MiniMax",
        marks=("MiniMax",),
        owner="MiniMax",
        pattern=r"\bMiniMax\b",
        assets=("minimax",),
    ),
    _Entry(
        label="Mistral AI",
        marks=("Mistral AI", "Pixtral"),
        owner="Mistral AI SAS",
        pattern=r"\bMistral\b|\bPixtral\b",
        assets=("mistralai",),
    ),
    _Entry(
        label="Moonshot AI, Kimi",
        marks=("Moonshot AI", "Kimi"),
        owner="Moonshot AI",
        pattern=r"\bMoonshot\b|\bKimi\b",
        assets=("moonshot",),
    ),
    _Entry(
        label="n8n",
        marks=("n8n",),
        owner="n8n GmbH",
        pattern=r"\bn8n\b",
        assets=("n8n",),
    ),
    _Entry(
        label="NVIDIA, Nemotron",
        marks=("NVIDIA", "Nemotron"),
        owner="NVIDIA Corporation",
        pattern=r"\bNVIDIA\b|\bNvidia\b|\bNemotron\b",
        assets=("nvidia",),
    ),
    _Entry(
        label="Ollama",
        marks=("Ollama",),
        owner="Ollama Inc.",
        pattern=r"\bOllama\b",
        assets=("ollama",),
    ),
    _Entry(
        label="OpenAI, ChatGPT, GPT, Codex",
        marks=("OpenAI", "ChatGPT", "GPT", "Codex"),
        owner="OpenAI, Inc.",
        pattern=r"\bOpenAI\b|\bChatGPT\b|\bGPT\b|\bCodex\b",
        assets=("openai",),
    ),
    _Entry(
        label="OpenClaw",
        marks=("OpenClaw",),
        owner="the OpenClaw project",
        pattern=r"\bOpenClaw\b|\bopenclaw\b",
        assets=("openclaw",),
    ),
    _Entry(
        label="Open WebUI",
        marks=("Open WebUI",),
        owner="the Open WebUI project",
        pattern=r"\bOpen WebUI\b|\bopen-webui\b|\bopenwebui\b",
        assets=("openwebui",),
    ),
    _Entry(
        label="OpenSearch",
        marks=("OpenSearch",),
        owner="LF Projects, LLC",
        pattern=r"(?<!Amazon )\bOpenSearch\b",
    ),
    _Entry(
        label="OpenTofu",
        marks=("OpenTofu",),
        owner="The Linux Foundation",
        pattern=r"\bOpenTofu\b",
    ),
    _Entry(
        label="ParadeDB",
        marks=("ParadeDB",),
        owner="ParadeDB, Inc.",
        pattern=r"\bParadeDB\b",
    ),
    _Entry(
        label="PostgreSQL, Postgres",
        marks=("PostgreSQL", "Postgres", "the Slonik elephant logo"),
        owner="the PostgreSQL Community Association of Canada",
        pattern=r"\bPostgreSQL\b|\bPostgres\b",
    ),
    _Entry(
        label="Python",
        marks=("Python",),
        owner="the Python Software Foundation",
        pattern=r"\bPython\b",
        assets=("python",),
    ),
    _Entry(
        label="Playwright, Visual Studio Code",
        marks=("Playwright", "Visual Studio Code"),
        owner="Microsoft Corporation",
        pattern=r"\bPlaywright\b|\bVisual Studio Code\b|\bVS Code\b",
        assets=("playwright", "vscode"),
    ),
    _Entry(
        label="RAGFlow",
        marks=("RAGFlow",),
        owner="InfiniFlow",
        pattern=r"\bRAGFlow\b|\bragflow\b",
        assets=("ragflow",),
    ),
    _Entry(
        label="SearXNG",
        marks=("SearXNG",),
        owner="the SearXNG project",
        pattern=r"\bSearXNG\b|\bsearxng\b",
        assets=("searxng",),
    ),
    _Entry(
        label="Stability AI, Stable Diffusion",
        marks=("Stability AI", "Stable Diffusion"),
        owner="Stability AI Ltd.",
        pattern=r"\bStability AI\b|\bStable Diffusion\b",
        assets=("stabilityai",),
    ),
    _Entry(
        label="Terraform, HashiCorp",
        marks=("Terraform", "HashiCorp"),
        owner="HashiCorp, Inc.",
        pattern=r"\bTerraform\b|\bHashiCorp\b",
    ),
    _Entry(
        label="TwelveLabs, Marengo, Pegasus",
        marks=("TwelveLabs", "Marengo", "Pegasus"),
        owner="Twelve Labs, Inc.",
        pattern=r"\bTwelveLabs\b|\bTwelve Labs\b|\bMarengo\b|\bPegasus\b",
        assets=("twelvelabs",),
    ),
    _Entry(
        label="Valkey",
        marks=("Valkey",),
        owner="LF Projects, LLC",
        pattern=r"\bValkey\b",
    ),
    _Entry(
        label="Writer, Palmyra",
        marks=("Writer", "Palmyra"),
        owner="Writer, Inc.",
        pattern=r"\bPalmyra\b",
        assets=("writer",),
    ),
    _Entry(
        label="xAI, Grok",
        marks=("xAI", "Grok"),
        owner="xAI Corp.",
        pattern=r"\bxAI\b|\bGrok\b",
        assets=("xai",),
    ),
    _Entry(
        label="Z.ai, Zhipu AI, GLM",
        marks=("Z.ai", "Zhipu AI", "GLM"),
        owner="Z.ai",
        pattern=r"\bZ\.ai\b|\bZhipu AI\b|\bGLM\b",
        assets=("zai",),
    ),
)

#: Logo assets that are stdapi.ai's own artwork and need no third-party attribution.
_OWN_ASSETS: frozenset[str] = frozenset({"logo.svg"})

#: Page whose body carries the generated registry table instead of a footer notice.
_TRADEMARKS_PAGE: str = "trademarks.md"

#: Placeholder in the trademarks page replaced by the generated registry table.
_TABLE_MARKER: str = "<!-- trademarks-table -->"

#: Matches a reference to a brand logo asset, capturing its stem.
_ASSET_RE: re.Pattern[str] = re.compile(r"styles/logo_([A-Za-z0-9_]+)\.svg")

#: Registry entries keyed by the logo asset stems they cover.
_ENTRY_BY_ASSET: dict[str, _Entry] = {
    asset: entry for entry in _REGISTRY for asset in entry.assets
}

#: Registry entries paired with their compiled detection pattern.
_COMPILED: tuple[tuple[_Entry, re.Pattern[str]], ...] = tuple(
    (entry, re.compile(entry.pattern)) for entry in _REGISTRY
)


def _owner_groups(entries: list[_Entry]) -> list[dict[str, str]]:
    """Group entries by owner so a single owner is never attributed twice.

    Args:
        entries: Matched registry entries.

    Returns:
        List of ``{"labels": ..., "owner": ...}`` dicts sorted by first label. The
        owner is empty when it would merely repeat the mark it owns.
    """
    by_owner: dict[str, list[str]] = {}
    for entry in entries:
        by_owner.setdefault(entry.owner, []).append(entry.label)
    groups = [
        {
            "labels": (labels := ", ".join(marks)),
            "owner": "" if labels == owner else owner,
        }
        for owner, marks in by_owner.items()
    ]
    return sorted(groups, key=lambda group: group["labels"].lower())


def _render_table() -> str:
    """Render the canonical registry as a Markdown table.

    Returns:
        Markdown table of every mark and its owner, sorted by mark.
    """
    rows = sorted(
        ((mark, entry.owner) for entry in _REGISTRY for mark in entry.marks),
        key=lambda row: row[0].lower(),
    )
    lines = ["| Mark | Owner |", "| --- | --- |"]
    lines.extend(f"| {mark} | {owner} |" for mark, owner in rows)
    return "\n".join(lines)


def on_page_markdown(
    markdown: str,
    page: Any,  # noqa: ANN401
    config: Any,  # noqa: ARG001,ANN401
    files: Any,  # noqa: ARG001,ANN401
) -> str | None:
    """Derive the page's trademark attributions from what the page references.

    Args:
        markdown: The page's Markdown source.
        page: The MkDocs page being rendered.
        config: MkDocs configuration object.
        files: The MkDocs file collection.

    Returns:
        The trademarks page with its table filled in, otherwise ``None``.
    """
    if page.file.src_uri == _TRADEMARKS_PAGE:
        return markdown.replace(_TABLE_MARKER, _render_table())

    for stem in set(_ASSET_RE.findall(markdown)):
        if stem not in _ENTRY_BY_ASSET:
            _LOG.warning(
                "%s uses the brand logo 'styles/logo_%s.svg' with no trademark "
                "registry entry; add one to docs_hooks/trademarks.py",
                page.file.src_uri,
                stem,
            )

    matched = [entry for entry, pattern in _COMPILED if pattern.search(markdown)]
    if matched:
        page.meta["trademarks"] = _owner_groups(matched)
    return None


def on_post_build(config: Any) -> None:  # noqa: ANN401
    """Warn about unattributed logo assets and fill the agent-readable page copy.

    Args:
        config: MkDocs configuration object.
    """
    for asset in sorted(Path(config["docs_dir"], "styles").glob("logo*.svg")):
        stem = asset.stem.removeprefix("logo_")
        if asset.name not in _OWN_ASSETS and stem not in _ENTRY_BY_ASSET:
            _LOG.warning(
                "Brand logo asset 'styles/%s' has no trademark registry entry; "
                "add one to docs_hooks/trademarks.py",
                asset.name,
            )

    agent_copy = Path(config["site_dir"], "md", _TRADEMARKS_PAGE)
    if agent_copy.is_file():
        agent_copy.write_text(
            agent_copy.read_text(encoding="utf-8").replace(
                _TABLE_MARKER, _render_table()
            ),
            encoding="utf-8",
        )
