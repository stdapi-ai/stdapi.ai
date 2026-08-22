"""Shape of the artefacts the Models page reads.

The index (``catalog.json``) is loaded on first paint and carries everything the
table can sort or filter on. The full price matrix of a model is an order of
magnitude larger than its index row, so it lives in a per-model detail document
fetched only when the reader opens or compares that model.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Score(BaseModel):
    """One published benchmark result attributed to one model.

    Attributes:
        source: Key of the :data:`~docs_gen.model_catalog.config.SOURCES` entry.
        board: Sub-leaderboard within the source, e.g. ``text`` or ``text_to_image``.
        metric: Metric name as the source reports it, e.g. ``elo`` or ``wer``.
        label: Short display label for the column header.
        value: The score itself.
        unit: Display unit, empty when the metric is unitless.
        higher_is_better: Whether a larger value is a better result.
        rank: Rank within the sub-leaderboard, when the source publishes one.
        ci_low: Lower bound of the published confidence interval, when any.
        ci_high: Upper bound of the published confidence interval, when any.
        samples: Vote or sample count behind the score, when published.
        as_of: Date of the snapshot the score was taken from.
        matched_name: Name the source uses, so a reader can verify the mapping.
        match_method: How the mapping was decided: ``rule``, ``llm`` or ``override``.
    """

    source: str
    board: str
    metric: str
    label: str
    value: float
    unit: str = ""
    higher_is_better: bool = True
    rank: int | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    samples: int | None = None
    as_of: str
    matched_name: str
    match_method: str


class Reference(BaseModel):
    """A pointer to a third-party evaluation that publishes no comparable score.

    Attributes:
        source: Key of the :data:`~docs_gen.model_catalog.config.SOURCES` entry.
        label: Short display label for the link.
        detail: What the reader will find there, e.g. how many tasks were run.
        url: Page the link points at.
        matched_name: Name the source uses, so a reader can verify the mapping.
        match_method: How the mapping was decided: ``rule``, ``llm`` or ``override``.
    """

    source: str
    label: str
    detail: str
    url: str
    matched_name: str
    match_method: str


class PriceGroup(BaseModel):
    """Headline unit prices shared by a set of regions.

    Attributes:
        regions: Indices into ``Manifest.regions`` that share these prices.
        prices: Billed dimension to exact standard-tier unit price, as a plain
            decimal string.
        cheapest: The same, for the cheapest whole service tier AWS publishes,
            for every dimension that tier prices — including ones priced the
            same as standard. A dimension in ``prices`` but absent here is one
            that tier does not sell, and the page must show a dash for it
            rather than fall back to the standard rate.
        cheapest_tier: Which tier that was, so the page can name it.
        routing: What product the price is — ``region`` for in-region
            inference, ``geography`` for a cross-region profile, ``global``,
            or ``""`` when AWS publishes no routing for it. A region appears
            once per routing it offers, because the caller chooses one and the
            products are priced differently.
        service: Which AWS service charges this, set only on a model served
            through more than one where they do not charge the same. Empty
            means the services agree and the choice has no cost consequence.
    """

    regions: list[int]
    prices: dict[str, str]
    cheapest: dict[str, str] = Field(default_factory=dict)
    cheapest_tier: str = ""
    routing: str = ""
    service: str = ""


class ServiceVariant(BaseModel):
    """One AWS service's way of calling a model the gateway serves twice.

    Amazon serves some models through both Bedrock Runtime and Bedrock Mantle.
    They are one model with one price, reached by two different ``model``
    values, so the catalogue carries one row and names both.

    Attributes:
        id: The ``model`` value this service accepts.
        service: The AWS service serving it.
        service_logo: Stem of that service's logo, when there is one.
        regions: Indices into ``Manifest.regions`` this variant reaches.
    """

    id: str
    service: str
    service_logo: str = ""
    regions: list[int] = Field(default_factory=list)


class ModelRow(BaseModel):
    """One row of the table: everything sortable or filterable about a model.

    Attributes:
        id: Amazon Bedrock model ID, the value other endpoints accept.
        slug: Filename-safe form of ``id``, used for the detail document.
        name: Human-readable model name.
        provider: Model provider.
        service: AWS service serving the model.
        input_modalities: Accepted input types.
        output_modalities: Produced output types.
        aliases: Alternate names the ``model`` parameter accepts.
        routes: API routes the model can be used with.
        mcp_tools: MCP tool names the model supports.
        regions: Indices into ``Manifest.regions`` the model is callable from.
        buckets: Geography buckets those regions fall into.
        serving: Where inference runs: ``global``, a geography prefix, or a region.
        streaming: Whether streaming responses are supported.
        batch: Whether the model is advertised for the Batch API.
        legacy: Whether the model is deprecated.
        start_of_life: GA date, if known.
        end_of_life: Deprecation date, if known.
        legacy_time: Date the model was marked legacy, if known.
        public_extended_access: Extended public-access end date, if known.
        inference_types: Bedrock inference types, e.g. ``ON_DEMAND``.
        customizations: Bedrock customizations supported, e.g. ``FINE_TUNING``.
        logo: Stem of the provider's brand logo under ``docs/styles``.
        service_logo: Stem of the serving AWS service's logo, when there is one.
        default_routing: The routing this gateway bills by default, when that
            is not simply the region called. Empty for everything else.
        variants: Every service this model is callable through, when more
            than one, each with the ``model`` value that service accepts.
        logo_backdrop: What the provider's logo needs behind it to stay
            visible: ``light``, ``dark`` or nothing.
        family: Model family AWS groups the model under, when it publishes one.
        apis: Amazon Bedrock inference APIs the model answers.
        max_output_tokens: Largest response the model is documented to produce.
        context_window: Context window the model accepts, when published.
        knowledge_cutoff: Date the model's training data ends, when published.
        reasoning: Whether the model supports explicit reasoning.
        tool_call: Whether the model supports tool calling.
        open_weights: Whether the model's weights are publicly released.
        licence: The weights licence, when a source classifies it.
        parameters: Total parameter count, for models that publish one.
        active_parameters: Parameters active per token, for mixture-of-
            experts models, where the total overstates the work done.
        prompt_caching: Whether prompt caching is supported.
        guardrails: Whether Amazon Bedrock Guardrails apply to it.
        latency_optimized: Whether a latency-optimised variant is published.
        provisioned: Whether provisioned throughput can be bought for it.
        count_tokens: Whether the token-counting API accepts it.
        prompt_routing: Whether intelligent prompt routing accepts it.
        batch_in_region: Whether batch inference runs in a single region.
        batch_cross_region: Whether batch inference runs across regions.
        image_types: Image media types the Converse API accepts for it.
        document_types: Document media types the Converse API accepts for it.
        video_types: Video media types the Converse API accepts for it.
        currency: ISO currency code of every price in ``price_groups``.
        price_groups: Headline prices, deduplicated across regions.
        has_prices: Whether a per-model price detail document was written.
        scores: Independent leaderboard results matched to this model.
        references: Third-party evaluations that publish no comparable score.
        first_seen: Date this model first appeared in the published data set.
        last_seen: Date a collection last saw it.
        retired: Whether AWS has stopped listing it; the row is kept anyway.
    """

    #: Validates every later ``setattr``, not only construction — the merge
    #: and the enrichment overlay both assign fields after the row is built.
    model_config = ConfigDict(validate_assignment=True)

    id: str
    slug: str
    name: str
    provider: str
    service: str
    input_modalities: list[str]
    output_modalities: list[str]
    aliases: list[str] = Field(default_factory=list)
    routes: list[str] = Field(default_factory=list)
    mcp_tools: list[str] = Field(default_factory=list)
    regions: list[int] = Field(default_factory=list)
    buckets: list[str] = Field(default_factory=list)
    serving: list[str] = Field(default_factory=list)
    streaming: bool | None = None
    batch: bool | None = None
    legacy: bool = False
    start_of_life: str | None = None
    end_of_life: str | None = None
    legacy_time: str | None = None
    public_extended_access: str | None = None
    inference_types: list[str] = Field(default_factory=list)
    customizations: list[str] = Field(default_factory=list)
    logo: str = ""
    service_logo: str = ""
    default_routing: str = ""
    variants: list[ServiceVariant] = Field(default_factory=list)
    logo_backdrop: str = ""
    family: str = ""
    apis: list[str] = Field(default_factory=list)
    max_output_tokens: int | None = None
    context_window: str | None = None
    knowledge_cutoff: str | None = None
    reasoning: bool | None = None
    tool_call: bool | None = None
    open_weights: bool | None = None
    licence: str = ""
    parameters: str | None = None
    active_parameters: str | None = None
    prompt_caching: bool | None = None
    guardrails: bool | None = None
    latency_optimized: bool | None = None
    provisioned: bool | None = None
    count_tokens: bool | None = None
    prompt_routing: bool | None = None
    batch_in_region: bool | None = None
    batch_cross_region: bool | None = None
    image_types: list[str] = Field(default_factory=list)
    document_types: list[str] = Field(default_factory=list)
    video_types: list[str] = Field(default_factory=list)
    currency: str = "USD"
    price_groups: list[PriceGroup] = Field(default_factory=list)
    has_prices: bool = False
    scores: list[Score] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    first_seen: str | None = None
    last_seen: str | None = None
    retired: bool = False


class SourceReport(BaseModel):
    """A published source, its licence, and what it contributed to this build.

    Attributes:
        key: Stable source identifier.
        name: Display name.
        url: Canonical landing page.
        licence: Licence short name.
        licence_url: Canonical licence text.
        attribution: Sentence the page renders to satisfy the licence.
        as_of: Date of the snapshot used.
        rows: Rows read from the source.
        matched: Rows mapped onto a model in this catalogue.
    """

    key: str
    name: str
    url: str
    licence: str
    licence_url: str
    attribution: str
    as_of: str
    rows: int
    matched: int


class Manifest(BaseModel):
    """Provenance of one generated data set.

    Attributes:
        generated: Date the data set was generated.
        gateway_version: Version of the stdapi.ai instance it was read from.
        partitions: AWS partitions covered.
        currencies: ISO currency codes present in the prices.
        reference_region: Region whose prices the table shows by default.
        regions: Every region referenced, in index order.
        region_buckets: Region to geography bucket, the page's editorial mapping.
        region_countries: Region to the ISO country it sits in, for the flags.
        unreachable_regions: Regions that could not be read, and why.
        sources: Published sources with their licence and contribution.
        carried_fields: Values kept from the previous data set because this run
            collected none, by field name.
        retired_models: Models the catalogue keeps that AWS no longer lists.
        headline_dimensions: Billed-dimension keys priced in every group, in
            display order, so the page can assert it agrees with the generator.
    """

    generated: str
    gateway_version: str
    partitions: list[str]
    currencies: list[str]
    reference_region: str
    regions: list[str]
    region_buckets: dict[str, str]
    region_countries: dict[str, str] = Field(default_factory=dict)
    unreachable_regions: dict[str, str] = Field(default_factory=dict)
    sources: list[SourceReport] = Field(default_factory=list)
    carried_fields: dict[str, int] = Field(default_factory=dict)
    retired_models: int = 0
    headline_dimensions: list[str] = Field(default_factory=list)


class Catalog(BaseModel):
    """The document the page fetches on first paint.

    Attributes:
        manifest: Provenance of this data set.
        models: Every model the gateway serves, sorted by ID.
    """

    manifest: Manifest
    models: list[ModelRow]


class ModelDetail(BaseModel):
    """Everything shown only once a reader opens a single model.

    Kept out of the index because the full price matrix and the vendor's own
    prose are an order of magnitude larger than the row they belong to.

    Attributes:
        id: Amazon Bedrock model ID.
        prices: The model's full published price card, exactly as the gateway
            returns it.
        description: The vendor's full description, when AWS publishes one.
        summary: One-line description, when AWS publishes one.
        attributes: Capability summary AWS publishes for the model.
        languages: Languages the vendor states the model supports.
        use_cases: Use cases the vendor states the model is built for.
        context_window: Context window as AWS advertises it.
        policy_url: Third-party model policy the vendor's terms point at.
    """

    id: str
    prices: dict[str, Any] = Field(default_factory=dict)
    description: str | None = None
    summary: str | None = None
    attributes: str | None = None
    languages: str | None = None
    use_cases: str | None = None
    context_window: str | None = None
    policy_url: str | None = None
