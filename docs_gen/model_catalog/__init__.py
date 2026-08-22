"""Generator for the public Models page data set.

Collects the catalogue from a running stdapi.ai instance, enriches it with
per-region Amazon Bedrock metadata and with independent leaderboard scores, and
writes the committed artefacts the static page reads.

Run with ``python -m docs_gen.model_catalog``.
"""
