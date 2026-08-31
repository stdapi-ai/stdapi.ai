"""Editorial choices and fixed parameters of the model catalogue generator.

Every judgement call the page depends on — how AWS regions map to the four
geography buttons, which leaderboard is allowed to be published and under what
licence, what the artefact may weigh — is declared here rather than spread
through the collectors, the matcher and the page.
"""

from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable

#: Repository root, derived from this file's location.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Directory holding the published artefacts the page fetches at runtime.
DATA_DIR: Path = REPO_ROOT / "docs" / "models"

#: Directory holding generator state that is committed but never published.
STATE_DIR: Path = REPO_ROOT / "docs_gen" / "model_catalog" / "state"

#: Human-curated match decisions, applied after every automatic pass.
OVERRIDES_PATH: Path = STATE_DIR / "overrides.json"

#: Match decisions already taken, so a re-run asks the LLM only about new rows.
MATCH_CACHE_PATH: Path = STATE_DIR / "match_cache.json"

#: Facts looked up by hand from a vendor's own documentation, with citations.
ENRICHMENT_PATH: Path = STATE_DIR / "enrichment.json"

#: Where each hand-filled value came from, written by the generator.
PROVENANCE_PATH: Path = STATE_DIR / "provenance.json"

#: Leaderboard rows and models left without a counterpart by the last run.
UNMATCHED_PATH: Path = STATE_DIR / "unmatched.json"

#: Raw upstream snapshots, kept out of git so a re-run can reuse them locally.
#:
#: Under the repository rather than /tmp: a predictable world-writable path lets
#: anyone with a local account pre-plant a snapshot and have their numbers
#: published under a source's name and licence.
SNAPSHOT_DIR: Path = REPO_ROOT / ".cache" / "model-catalog"

#: Geography bucket for each AWS region prefix — the page's editorial mapping.
REGION_BUCKETS: dict[str, str] = {
    "us": "americas",
    "ca": "americas",
    "mx": "americas",
    "sa": "americas",
    "eu": "europe",
    "eusc": "europe",
    "ap": "asia_pacific",
    "me": "middle_east",
    "il": "middle_east",
    "af": "africa",
}

#: ISO 3166-1 country each AWS region physically sits in, for the page's flags.
#:
#: A region is not a country and AWS does not publish this mapping as data, so
#: it is an editorial table like the buckets above. A region missing from it
#: simply shows no flag.
REGION_COUNTRIES: dict[str, str] = {
    "af-south-1": "ZA",
    "ap-east-1": "HK",
    "ap-east-2": "TW",
    "ap-northeast-1": "JP",
    "ap-northeast-2": "KR",
    "ap-northeast-3": "JP",
    "ap-south-1": "IN",
    "ap-south-2": "IN",
    "ap-southeast-1": "SG",
    "ap-southeast-2": "AU",
    "ap-southeast-3": "ID",
    "ap-southeast-4": "AU",
    "ap-southeast-5": "MY",
    "ap-southeast-6": "NZ",
    "ap-southeast-7": "TH",
    "ca-central-1": "CA",
    "ca-west-1": "CA",
    "eu-central-1": "DE",
    "eu-central-2": "CH",
    "eu-north-1": "SE",
    "eu-south-1": "IT",
    "eu-south-2": "ES",
    "eu-west-1": "IE",
    "eu-west-2": "GB",
    "eu-west-3": "FR",
    "eusc-de-east-1": "DE",
    "il-central-1": "IL",
    "me-central-1": "AE",
    "me-south-1": "BH",
    "mx-central-1": "MX",
    "sa-east-1": "BR",
    "us-east-1": "US",
    "us-east-2": "US",
    "us-west-1": "US",
    "us-west-2": "US",
}

#: Brand logo asset stem for each AWS service that serves a model.
SERVICE_LOGOS: dict[str, str] = {
    "AWS Bedrock Runtime": "amazon_bedrock",
    "AWS Bedrock Mantle": "amazon_bedrock",
    "AWS Polly": "amazon_polly",
    "AWS Transcribe": "amazon_transcribe",
    "AWS Comprehend": "amazon_comprehend",
    "AWS Translate": "amazon_translate",
}

#: Bucket shown when a region prefix is not in ``REGION_BUCKETS``.
UNKNOWN_BUCKET: str = "other"

#: Cross-region inference profile prefixes that mean a model is served globally.
GLOBAL_PROFILE_PREFIXES: tuple[str, ...] = ("global.",)

#: Partition every v1 price belongs to; the schema carries it so EUSC can follow.
DEFAULT_PARTITION: str = "aws"

#: Tiers that need a different call, not just a different parameter.
#:
#: Batch is a separate job API: a caller cannot reach its price by changing a
#: request, so quoting it as this model's cheaper rate would misprice the work
#: they actually came to do. Flex and priority are request-level and stay.
TIERS_NEEDING_A_REWRITE: frozenset[str] = frozenset({"batch"})

#: Billed dimensions promoted to the table as headline prices, in display order.
HEADLINE_DIMENSIONS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_images",
    "input_seconds",
    "output_seconds",
    "input_characters",
    "input_images",
    "search_units",
    "comprehend_units",
    "text_units",
    "grounding_requests",
)

#: Region whose headline prices the table shows until the reader picks another.
REFERENCE_REGION: str = "us-east-1"

#: Ceiling on the gzipped index the page loads on first paint, in bytes.
INDEX_GZIP_BUDGET: int = 150 * 1024

#: Bedrock model used for LLM-assisted matching; first-party and credit-eligible.
DEFAULT_MATCH_MODEL: str = "amazon.nova-2-lite-v1:0"

#: Minimum self-reported confidence below which an LLM match is discarded.
MATCH_CONFIDENCE_FLOOR: float = 0.8


class SourceInfo(NamedTuple):
    """Attribution record for one published data source.

    Attributes:
        key: Stable identifier used in the artefact and in override files.
        name: Display name shown in the page's sources section.
        url: Canonical human-readable landing page.
        licence: Licence short name, exactly as the source declares it.
        licence_url: Canonical licence text.
        attribution: Sentence the page must render to satisfy the licence.
    """

    key: str
    name: str
    url: str
    licence: str
    licence_url: str
    attribution: str


#: Every leaderboard the page is allowed to publish, with its attribution.
SOURCES: tuple[SourceInfo, ...] = (
    SourceInfo(
        key="lmarena",
        name="LMArena Leaderboard",
        url="https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset",
        licence="CC BY 4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        attribution=(
            "Arena Elo ratings by LMArena (Arena Intelligence Inc.), reproduced "
            "unmodified under CC BY 4.0. The mapping to Amazon Bedrock model IDs "
            "is ours."
        ),
    ),
    SourceInfo(
        key="mteb",
        name="MTEB — Massive Text Embedding Benchmark",
        url="https://github.com/embeddings-benchmark/results",
        licence="CC0 1.0",
        licence_url="https://creativecommons.org/publicdomain/zero/1.0/",
        attribution=(
            "Benchmark results from the MTEB results repository, dedicated to the "
            "public domain under CC0 1.0. The mapping to Amazon Bedrock model IDs "
            "is ours."
        ),
    ),
    SourceInfo(
        key="epoch",
        name="Epoch AI — AI Benchmarking Hub",
        url="https://epoch.ai/benchmarks/use-this-data",
        licence="CC BY 4.0",
        licence_url="https://creativecommons.org/licenses/by/4.0/",
        attribution=(
            "Benchmark results by Epoch AI, reproduced unmodified under CC BY 4.0. "
            "Rows derived from the Aider Polyglot and Terminal-Bench leaderboards "
            "keep their Apache-2.0 licence. The mapping to Amazon Bedrock model "
            "IDs is ours."
        ),
    ),
    SourceInfo(
        key="aws_model_cards",
        name="Amazon Bedrock model cards",
        url="https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html",
        licence="AWS documentation",
        licence_url="https://aws.amazon.com/terms/",
        attribution=(
            "Context windows, output limits, knowledge cutoffs and lifecycle "
            "dates as stated on each model's Amazon Bedrock model card. Only "
            "those facts are taken; AWS's own descriptive copy is not "
            "reproduced. A lifecycle date is read from the card only where "
            "Amazon Bedrock's own API states none, and a card's launch date is "
            "often the vendor's announcement rather than the date the model "
            "reached Amazon Bedrock, so those few dates are an approximation."
        ),
    ),
    SourceInfo(
        key="models_dev",
        name="models.dev",
        url="https://models.dev/",
        licence="MIT",
        licence_url="https://github.com/anomalyco/models.dev/blob/dev/LICENSE",
        attribution=(
            "Context windows, knowledge cutoffs and capability flags from "
            "models.dev, an open database of AI models, used under the MIT "
            "licence. Its Amazon Bedrock entries are keyed by Bedrock model ID, "
            "so the join is exact."
        ),
    ),
    SourceInfo(
        key="open_asr",
        name="Open ASR Leaderboard",
        url="https://huggingface.co/spaces/hf-audio/open_asr_leaderboard",
        licence="Apache-2.0",
        licence_url="https://www.apache.org/licenses/LICENSE-2.0",
        attribution=(
            "Word error rates from the Open ASR Leaderboard, reproduced unmodified "
            "under Apache-2.0. The mapping to Amazon Bedrock model IDs is ours."
        ),
    ),
)

#: Output modality a board can possibly be describing, by ``source/board``.
#:
#: A leaderboard entry can only belong to a model that produces what the board
#: measures. Without this, a confident-sounding name match can hand an image
#: *embedding* model a score from an image *generation* arena.
BOARD_OUTPUT_MODALITIES: dict[str, frozenset[str]] = {
    "lmarena/text": frozenset({"TEXT"}),
    "lmarena/vision": frozenset({"TEXT"}),
    "lmarena/search": frozenset({"TEXT"}),
    "lmarena/text_to_image": frozenset({"IMAGE"}),
    "lmarena/image_edit": frozenset({"IMAGE"}),
    "lmarena/text_to_video": frozenset({"VIDEO"}),
    "lmarena/image_to_video": frozenset({"VIDEO"}),
    "epoch/gpqa_diamond": frozenset({"TEXT"}),
    "epoch/swe_bench_verified": frozenset({"TEXT"}),
    "epoch/frontiermath": frozenset({"TEXT"}),
    "epoch/math_level_5": frozenset({"TEXT"}),
    "epoch/simpleqa_verified": frozenset({"TEXT"}),
    "epoch/aider_polyglot_external": frozenset({"TEXT"}),
    "open_asr/english_short": frozenset({"TEXT"}),
    "mteb/reference": frozenset({"EMBEDDING", "RERANKING"}),
}

#: Input modality a board's models must accept, by ``source/board``.
BOARD_INPUT_MODALITIES: dict[str, frozenset[str]] = {
    "lmarena/vision": frozenset({"IMAGE"}),
    "lmarena/image_edit": frozenset({"IMAGE"}),
    "lmarena/image_to_video": frozenset({"IMAGE"}),
    "open_asr/english_short": frozenset({"AUDIO", "SPEECH"}),
}


def board_fits_model(
    board_key: str, input_modalities: Iterable[str], output_modalities: Iterable[str]
) -> bool:
    """Report whether a board can possibly be describing a given model.

    Args:
        board_key: ``source/board`` identifier.
        input_modalities: Input types the model accepts.
        output_modalities: Output types the model produces.

    Returns:
        ``False`` when the model cannot produce what the board measures.
    """
    wanted_out = BOARD_OUTPUT_MODALITIES.get(board_key)
    if wanted_out is not None and not wanted_out & set(output_modalities):
        return False
    wanted_in = BOARD_INPUT_MODALITIES.get(board_key)
    return wanted_in is None or bool(wanted_in & set(input_modalities))


#: Brand logo asset stem for each catalogue provider, under ``docs/styles``.
#:
#: A provider with no entry fails the run rather than rendering a nameless row:
#: an unattributed mark is exactly what ``docs_hooks/trademarks.py`` exists to
#: prevent, and a silent blank is how one ships.
PROVIDER_LOGOS: dict[str, str] = {
    "AI21 Labs": "ai21",
    "Amazon": "amazon",
    "Anthropic": "anthropic",
    "Cohere": "cohere",
    "DeepSeek": "deepSeek",
    "Google": "google",
    "Luma AI": "luma",
    "Meta": "meta",
    "MiniMax": "minimax",
    "Mistral AI": "mistralai",
    "Moonshot AI": "moonshot",
    "NVIDIA": "nvidia",
    "OpenAI": "openai",
    "Qwen": "qwen",
    "Stability AI": "stabilityai",
    "TwelveLabs": "twelvelabs",
    "Writer": "writer",
    "Z.AI": "zai",
    "Zhipu AI": "zai",
    "xAI": "xai",
}


def region_bucket(region: str) -> str:
    """Return the geography bucket of an AWS region.

    Args:
        region: AWS region name, for example ``eu-west-1``.

    Returns:
        The bucket key, or ``UNKNOWN_BUCKET`` for an unrecognised prefix.
    """
    return REGION_BUCKETS.get(region.split("-", 1)[0], UNKNOWN_BUCKET)
