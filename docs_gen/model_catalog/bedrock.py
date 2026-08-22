"""Per-region Amazon Bedrock metadata that the gateway's API does not expose.

``ListFoundationModels`` is read over plain signed HTTP rather than through a
generated client, because the wire response carries eleven fields the modelled
response shape drops on the floor — among them which inference APIs a model
answers, how many tokens it can emit, whether it can be cached, batched or
latency-optimised, and which media types it accepts.

Every field read here is optional. A region that cannot be reached, a field AWS
stops returning, or a payload that changes shape degrades the affected column
instead of failing the run. ``consoleIDEMetadata`` in particular is only
returned to some callers: it carries the model's description, context window and
use cases, and is parsed when present and ignored when not.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from botocore.session import get_session

from docs_gen.model_catalog.http import USER_AGENT, map_concurrent
from docs_gen.model_catalog.tokens import format_tokens

if TYPE_CHECKING:
    from collections.abc import Iterable

#: SSM path listing every AWS region Amazon Bedrock is available in.
_BEDROCK_REGIONS_PARAMETER: str = (
    "/aws/service/global-infrastructure/services/bedrock/regions"
)

#: Partitions this generator does not collect; EUSC and GovCloud need their own run.
_FOREIGN_REGION_PREFIXES: tuple[str, ...] = ("eusc-", "us-gov-")

#: Region used to query the global-infrastructure parameters.
_INFRASTRUCTURE_REGION: str = "us-east-1"

#: Control-plane endpoint whose raw response carries the undocumented fields.
_ENDPOINT: str = "https://bedrock.{region}.amazonaws.com/foundation-models"

#: Seconds to wait for one region's control plane before giving up on it.
_TIMEOUT: float = 20.0

#: Short-fused client settings for the regions the account has not opted into.
_CLIENT_CONFIG: Config = Config(
    connect_timeout=5, read_timeout=30, retries={"max_attempts": 2, "mode": "standard"}
)


class RegionalModels(NamedTuple):
    """What one region reported.

    Attributes:
        region: AWS region name.
        summaries: ``modelSummaries`` entries, empty when the region failed.
        error: Why the region could not be read, or ``None``.
    """

    region: str
    summaries: list[dict[str, Any]]
    error: str | None


class ModelFacts(NamedTuple):
    """The per-model capabilities the catalogue publishes.

    Attributes:
        table: Small, sortable facts that belong in the table.
        detail: Long-form facts shown only when one model is opened.
    """

    table: dict[str, Any]
    detail: dict[str, Any]


def commercial_bedrock_regions() -> list[str]:
    """Return every commercial-partition AWS region that offers Amazon Bedrock.

    Returns:
        Region names, sorted.
    """
    session = get_session()
    client = session.create_client(
        "ssm", region_name=_INFRASTRUCTURE_REGION, config=_CLIENT_CONFIG
    )
    paginator = client.get_paginator("get_parameters_by_path")
    regions: set[str] = set()
    for page in paginator.paginate(Path=_BEDROCK_REGIONS_PARAMETER):
        regions.update(str(parameter["Value"]) for parameter in page["Parameters"])
    return sorted(
        region for region in regions if not region.startswith(_FOREIGN_REGION_PREFIXES)
    )


def list_foundation_models(regions: Iterable[str]) -> list[RegionalModels]:
    """Read the raw ``ListFoundationModels`` response in every region.

    Args:
        regions: AWS regions to query.

    Returns:
        One entry per region, in the order given.
    """
    session = get_session()
    try:
        credentials = session.get_credentials().get_frozen_credentials()  # type: ignore[union-attr]
    except (ClientError, BotoCoreError, AttributeError) as error:
        return [
            RegionalModels(region, [], f"no credentials: {error}") for region in regions
        ]

    def pull(region: str) -> RegionalModels:
        url = _ENDPOINT.format(region=region)
        signed = AWSRequest(
            method="GET", url=url, headers={"Accept": "application/json"}
        )
        SigV4Auth(credentials, "bedrock", region).add_auth(signed)
        headers = {**dict(signed.headers), "User-Agent": USER_AGENT}
        request = Request(url, headers=headers)  # noqa: S310 -- fixed https endpoint
        try:
            with urlopen(request, timeout=_TIMEOUT) as response:  # noqa: S310
                payload = json.loads(response.read())
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            return RegionalModels(region, [], f"{type(error).__name__}: {error}")
        summaries = payload.get("modelSummaries")
        if not isinstance(summaries, list):
            return RegionalModels(region, [], "unexpected payload shape")
        return RegionalModels(region, summaries, None)

    return map_concurrent(pull, list(regions))


#: Nested ``inferenceAPIsSupported`` flags, and the API each one names.
_API_GROUPS: dict[str, dict[str, str]] = {
    "converse": {"sync": "Converse", "streaming": "ConverseStream"},
    "invokeModel": {
        "sync": "InvokeModel",
        "responseStreaming": "InvokeModelStream",
        "bidirectionalStreaming": "InvokeModelBidi",
    },
}

#: Top-level ``inferenceAPIsSupported`` flags, and the API each one names.
_API_FLAGS: dict[str, str] = {
    "asyncInvoke": "StartAsyncInvoke",
    "openAiChatCompletions": "OpenAI Chat Completions",
    "openAiResponses": "OpenAI Responses",
}


def _console_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    """Decode the console metadata blob, when the caller is given one.

    Args:
        summary: One ``modelSummaries`` entry.

    Returns:
        The decoded metadata, or an empty mapping.
    """
    raw = summary.get("consoleIDEMetadata")
    if not isinstance(raw, str):
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _apis(summary: dict[str, Any]) -> list[str]:
    """Return the inference APIs a model answers.

    Args:
        summary: One ``modelSummaries`` entry.

    Returns:
        Sorted API names, empty when the field is absent.
    """
    supported = summary.get("inferenceAPIsSupported")
    if not isinstance(supported, dict):
        return []
    names: set[str] = set()
    for group, members in _API_GROUPS.items():
        block = supported.get(group)
        if not isinstance(block, dict):
            continue
        names.update(label for key, label in members.items() if block.get(key))
    names.update(label for key, label in _API_FLAGS.items() if supported.get(key))
    return sorted(names)


def _flag(value: object) -> bool | None:
    """Read a capability flag that may be absent or nested.

    Args:
        value: A boolean, a ``{"isSupported": bool}`` mapping, or ``None``.

    Returns:
        The flag, or ``None`` when the field was not returned.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, dict) and isinstance(value.get("isSupported"), bool):
        return bool(value["isSupported"])
    return None


def _merge_flag(*, current: bool | None, incoming: bool | None) -> bool | None:
    """Combine one capability flag across regions.

    A capability offered in any region is offered by the model, so the union
    wins; ``None`` only survives when no region reported the field at all.

    Args:
        current: Value accumulated so far.
        incoming: Value from this region.

    Returns:
        The merged flag.
    """
    if incoming is None:
        return current
    return incoming if current is None else (current or incoming)


def index_by_model(regional: Iterable[RegionalModels]) -> dict[str, ModelFacts]:
    """Collapse per-region summaries into per-model capabilities.

    Args:
        regional: Per-region ``ListFoundationModels`` results.

    Returns:
        Model ID to the facts the catalogue publishes for it.
    """
    sets: dict[str, dict[str, set[str]]] = {}
    flags: dict[str, dict[str, bool | None]] = {}
    detail: dict[str, dict[str, Any]] = {}

    for entry in regional:
        for summary in entry.summaries:
            model_id = str(summary.get("modelId") or "")
            if not model_id:
                continue
            _absorb(model_id, summary, sets, flags, detail)

    return {
        model_id: ModelFacts(
            table={
                **{key: sorted(values) for key, values in sets[model_id].items()},
                **{
                    key: value
                    for key, value in flags[model_id].items()
                    if value is not None
                },
            },
            detail=detail.get(model_id, {}),
        )
        for model_id in sets
    }


def _absorb(
    model_id: str,
    summary: dict[str, Any],
    sets: dict[str, dict[str, set[str]]],
    flags: dict[str, dict[str, bool | None]],
    detail: dict[str, dict[str, Any]],
) -> None:
    """Fold one region's report of one model into the accumulators.

    Args:
        model_id: Amazon Bedrock model ID.
        summary: That region's ``modelSummaries`` entry.
        sets: Accumulator for list-valued facts.
        flags: Accumulator for capability flags.
        detail: Accumulator for long-form facts.
    """
    bucket = sets.setdefault(
        model_id,
        {
            "inference_types": set(),
            "customizations": set(),
            "apis": set(),
            "families": set(),
            "image_types": set(),
            "document_types": set(),
            "video_types": set(),
        },
    )
    bucket["inference_types"].update(
        map(str, summary.get("inferenceTypesSupported", ()))
    )
    bucket["customizations"].update(
        map(str, summary.get("customizationsSupported", ()))
    )
    bucket["apis"].update(_apis(summary))
    if summary.get("modelFamily"):
        bucket["families"].add(str(summary["modelFamily"]))

    converse = summary.get("converse")
    if isinstance(converse, dict):
        bucket["image_types"].update(
            map(str, converse.get("userImageTypesSupported", ()))
        )
        bucket["document_types"].update(
            map(str, converse.get("userDocumentTypesSupported", ()))
        )
        bucket["video_types"].update(
            map(str, converse.get("userVideoTypesSupported", ()))
        )
        # converse.maxTokensMaximum is the largest value the maxTokens request
        # parameter accepts, not the largest response the model produces: for
        # Gemma 3 it is the whole context window. The model cards state the real
        # ceiling, so that is where max_output_tokens comes from.

    features = summary.get("featuresSupported")
    features = features if isinstance(features, dict) else {}
    batch = summary.get("batchSupported")
    batch = batch if isinstance(batch, dict) else {}
    slot_flags = flags.setdefault(
        model_id,
        {
            "prompt_caching": None,
            "guardrails": None,
            "latency_optimized": None,
            "provisioned": None,
            "count_tokens": None,
            "prompt_routing": None,
            "batch_in_region": None,
            "batch_cross_region": None,
        },
    )
    for key, value in (
        (
            "prompt_caching",
            _flag(summary.get("explicitPromptCaching"))
            or _flag(features.get("promptCaching")),
        ),
        ("guardrails", _flag(summary.get("guardrailsSupported"))),
        ("latency_optimized", _flag(summary.get("latencyOptimizationSupported"))),
        ("provisioned", _flag(features.get("provisionedThroughput"))),
        ("count_tokens", _flag(features.get("countTokens"))),
        ("prompt_routing", _flag(summary.get("intelligentPromptRouting"))),
        ("batch_in_region", _flag(batch.get("baseModelSupported"))),
        ("batch_cross_region", _flag(batch.get("crossRegionSupported"))),
    ):
        slot_flags[key] = _merge_flag(current=slot_flags[key], incoming=value)

    _absorb_detail(model_id, summary, detail)


def _absorb_detail(
    model_id: str, summary: dict[str, Any], detail: dict[str, dict[str, Any]]
) -> None:
    """Collect the long-form facts shown when one model is opened.

    ``description`` is either a sentence or a whole structured block, depending
    on the model — the same block ``consoleIDEMetadata`` carries. Both shapes
    are read, and the structured one also yields the context window, which no
    other AWS response exposes.

    Args:
        model_id: Amazon Bedrock model ID.
        summary: That region's ``modelSummaries`` entry.
        detail: Accumulator for long-form facts.
    """
    slot = detail.setdefault(model_id, {})
    described = summary.get("description")
    if isinstance(described, str) and described.strip() and "description" not in slot:
        slot["description"] = described.strip()
    elif isinstance(described, dict):
        _absorb_described(described, slot)

    metadata = _console_metadata(summary)
    if isinstance(metadata.get("description"), dict):
        _absorb_described(metadata["description"], slot)


def _absorb_described(block: dict[str, Any], slot: dict[str, Any]) -> None:
    """Read a vendor description block into the model's detail.

    Args:
        block: The description block AWS returns.
        slot: Accumulator for that model's long-form facts.
    """
    for source_key, target in (
        ("fullDescription", "description"),
        ("shortDescription", "summary"),
        ("modelAttributes", "attributes"),
        ("supportedLanguages", "languages"),
        ("supportedUseCases", "use_cases"),
        ("policy", "policy_url"),
    ):
        value = block.get(source_key)
        if isinstance(value, str) and value.strip() and target not in slot:
            slot[target] = value.strip()
    if "context_window" not in slot:
        window = format_tokens(block.get("maxContextWindow"))
        if window:
            slot["context_window"] = window
