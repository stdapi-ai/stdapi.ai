"""Generator for the public Models page data set.

Collects the catalogue from a running stdapi.ai instance, enriches it with
per-region Amazon Bedrock metadata and with independent leaderboard scores, and
writes the committed artefacts the static page reads.

Run with ``python -m docs_gen.model_catalog``.

Check the hand-copied price tables first, with
``pytest tests/test_pricing_drift.py --drift``: the gateway serves those rates
into ``model_pricing`` and this generator publishes them.

The instance it reads decides what gets published, and a narrow one publishes a
smaller catalogue rather than failing: start it with ``AWS_BEDROCK_REGIONS`` set
to :func:`~docs_gen.model_catalog.bedrock.commercial_bedrock_regions` (that list
also drives Polly, Transcribe and Comprehend), ``AWS_BEDROCK_LEGACY=true`` so
deprecated models are listed, and ``COST_TRACKING=true`` so ``/model_pricing``
answers at all. Pricing loads in a background task after readiness, so wait for
it before starting the run. Always pass every region: an unreachable one is
reported as a warning and recorded in the manifest, while dropping it from the
configuration removes it from the page until someone notices.
"""
