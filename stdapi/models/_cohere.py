"""Naming conventions shared by the Cohere models Amazon Bedrock serves.

Cohere publishes a dotted two-part version (``embed-english-v3.0``,
``embed-v4.0``, ``rerank-v3.5``) where the Bedrock model ID carries a
dash-separated one (``cohere.embed-english-v3``, ``cohere.embed-v4:0``,
``cohere.rerank-v3-5:0``), so the alias published for a Cohere model has to
rewrite that suffix to match the name Cohere's own API accepts.

Ref: https://docs.cohere.com/docs/models
"""

from re import compile as re_compile

#: Alias body of a Cohere embedding model ID; the provisioned-only ``:0:512`` variants never match.
COHERE_EMBED_ALIAS_MATCHER = re_compile(r"^cohere\.(embed-[^:]+)(?::\d+)?$")

#: Alias body of a Cohere rerank model ID; the Amazon rerank models never match.
COHERE_RERANK_ALIAS_MATCHER = re_compile(r"^cohere\.(rerank-[^:]+)(?::\d+)?$")

#: Rewrites Bedrock's version suffix (``-v3``, ``-v3-5``) into Cohere's own (``-v3.0``, ``-v3.5``).
COHERE_ALIAS_SUBSTITUTIONS = (
    (re_compile(r"-v(\d+)-(\d+)(?=-|$)"), r"-v\1.\2"),
    (re_compile(r"-v(\d+)(?=-|$)"), r"-v\1.0"),
)
