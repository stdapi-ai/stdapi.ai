"""Model-ID-to-price-key overrides for the AWS price catalog.

``stdapi.pricing`` is model-agnostic: it derives price keys automatically
(``normalize_model_key``/``normalize_usagetype_model``) and only consults the
tables below, registered by ``stdapi.models`` at import time, when AWS's
naming diverges. New models therefore work with no code change unless the
pricing coverage test reports them as unpriced.

HOW TO FIX an "unpriced model" (also printed by the coverage test on failure):
  1. Run: uv run pytest tests/test_pricing.py::test_bedrock_model_pricing_coverage --expensive
  2. Its failure output lists UNPRICED MODELS and CANDIDATE MATCHES: add one
     line below per obvious pair: "<unpriced-model-id>": "<matching-unclaimed-key>",
  3. Re-run the test. If no candidate matches, AWS hasn't published pricing
     yet -- wait, check https://aws.amazon.com/bedrock/pricing/ for a rate to
     add to DEFAULT_MODEL_PRICES, or record a confirmed upstream gap in the
     test's _KNOWN_PRICING_GAPS.
"""

from typing import Final

from stdapi.pricing import Dimension

#: Manual model-ID-to-price-key overrides where automatic normalization mismatches.
MODEL_KEY_OVERRIDES: Final[dict[str, str]] = {
    "openai.gpt-oss-120b-1:0": "gptoss120b",
    "openai.gpt-oss-20b-1:0": "gptoss20b",
    "deepseek.v3.2": "deepseekv32",
    # Bedrock's ID for DeepSeek-V3.1 -- normalize_model_key() yields "v3",
    # but the Price List `model` attribute is "DeepSeek V3.1".
    "deepseek.v3-v1:0": "deepseekv31",
    # Mantle IDs whose Price List `model` attribute is the display name
    # without the "-instruct" suffix (mantle rows keyed per invocation API).
    "deepseek.v3.1": "deepseekv31",
    "qwen.qwen3-coder-30b-a3b-instruct": "qwen3coder30ba3b",
    "qwen.qwen3-coder-480b-a35b-instruct": "qwen3coder480ba35b",
    "qwen.qwen3-next-80b-a3b-instruct": "qwen3next80ba3b",
    "qwen.qwen3-vl-235b-a22b-instruct": "qwen3vl235ba22b",
    # Dated snapshot aliases billed at the bare model's rate.
    "openai.gpt-5.4-2026-03-05": "gpt54",
    "openai.gpt-5.5-2026-04-23": "gpt55",
    "amazon.nova-2-omni-v1:0": "nova20omni",
    "amazon.nova-2-pro-v1:0": "nova20pro",
    # Marketplace-listed -- keyed from the listing name (servicename).
    "stability.sd3-large-v1:0": "stablediffusion3largev10",
    "stability.sd3-5-large-v1:0": "stablediffusion35largev10",
    "cohere.command-r-v1:0": "coherecommandr",
    "cohere.command-r-plus-v1:0": "coherecommandr+",
    "twelvelabs.pegasus-1-2-v1:0": "twelvelabspegasus12",
    "twelvelabs.marengo-embed-3-0-v1:0": "twelvelabsmarengoembed30",
    "twelvelabs.marengo-embed-2-7-v1:0": "twelvelabsmarengoembed27",
    "google.gemma-3-12b-it": "gemma312b",
    "google.gemma-3-27b-it": "gemma327b",
    "google.gemma-3-4b-it": "gemma34b",
    "mistral.mistral-large-3-675b-instruct": "mistrallarge3",
    "mistral.ministral-3-14b-instruct": "ministral14b30",
    "mistral.ministral-3-8b-instruct": "ministral8b30",
    "mistral.ministral-3-3b-instruct": "ministral3b30",
    "mistral.devstral-2-123b": "devstral",
    "mistral.magistral-small-2509": "magistralsmall12",
    "mistral.voxtral-mini-3b-2507": "voxtralmini10",
    "mistral.voxtral-small-24b-2507": "voxtralsmall10",
    "mistral.mistral-7b-instruct-v0:2": "mistral7b",
    "mistral.mixtral-8x7b-instruct-v0:1": "mixtral8x7b",
    "mistral.mistral-large-2402-v1:0": "mistrallarge",
    "mistral.mistral-small-2402-v1:0": "mistralsmall",
    "meta.llama3-8b-instruct-v1:0": "llama38b",
    "meta.llama3-70b-instruct-v1:0": "llama370b",
    "meta.llama3-1-8b-instruct-v1:0": "llama318b",
    "meta.llama3-1-70b-instruct-v1:0": "llama3170b",
    "meta.llama3-3-70b-instruct-v1:0": "llama3370b",
    "meta.llama4-scout-17b-instruct-v1:0": "llama4scout17b",
    "meta.llama4-maverick-17b-instruct-v1:0": "llama4maverick17b",
    "writer.palmyra-vision-7b": "writerpalmyravision7b",
    "nvidia.nemotron-super-3-120b": "nvidianemotron3super120ba12b",
    "nvidia.nemotron-nano-9b-v2": "nvidianemotronnano2",
    "nvidia.nemotron-nano-12b-v2": "nvidianemotronnano2vl",
    "amazon.nova-2-lite-v1:0": "nova20lite",
    "amazon.nova-2-sonic-v1:0": "novasonic20",
    # No Price List `model` attribute at all: values below are
    # normalize_usagetype_model()'s output, not normalize_model_key()'s.
    "amazon.titan-embed-text-v1": "titanembeddingsg1text",
    "amazon.titan-embed-text-v2:0": "titanembeddingv2text",
    "amazon.nova-2-multimodal-embeddings-v1:0": "novamultimodalembeddings",
    "amazon.rerank-v1:0": "amazonrerankv1searchunits",
    "amazon.titan-text-express-v1": "titantextg1express",
    "amazon.titan-text-lite-v1": "titantextg1lite",
    "amazon.titan-embed-image-v1": "titanembeddingsg1",
    # normalize_model_key() strips "V2" as a version suffix but not "G1".
    "amazon.titan-image-generator-v1": "titanimagegeneratorg1",
    # Deprecated/legacy IDs (stdapi/models/deprecation.py): still invoked as-is
    # where AWS hasn't retired them, so they need pricing coverage of their own.
    "anthropic.claude-v2": "claude20",
    "anthropic.claude-v2:1": "claude21",
    "meta.llama3-1-405b-instruct-v1:0": "llama31405b",
    "meta.llama3-2-1b-instruct-v1:0": "llama321b",
    "meta.llama3-2-3b-instruct-v1:0": "llama323b",
    "meta.llama3-2-11b-instruct-v1:0": "llama3211b",
    "meta.llama3-2-90b-instruct-v1:0": "llama3290b",
    # Marketplace-listed legacy models -- naming diverges from normalize_model_key.
    "cohere.embed-english-v3": "cohereembed3modelenglish",
    "cohere.embed-multilingual-v3": "cohereembedmodel3multilingual",
    "cohere.embed-v4:0": "cohereembed4model",
    "cohere.rerank-v3-5:0": "coherererankv35",
    "cohere.command-text-v14": "coheregeneratemodelcommand",
    "cohere.command-light-text-v14": "coheregeneratemodelcommandlight",
    "ai21.j2-mid-v1": "jurassic2mid",
    "ai21.j2-ultra-v1": "jurassic2ultra",
    "ai21.j2-grande-instruct": "jurassic2mid",
    "ai21.j2-jumbo-instruct": "jurassic2ultra",
    "ai21.jamba-instruct-v1:0": "jambainstruct",
    "meta.llama2-13b-chat-v1": "metallama2chat13b",
    "meta.llama2-70b-chat-v1": "metallama2chat70b",
    "meta.llama2-13b-v1": "metallama2chat13b",
    "meta.llama2-70b-v1": "metallama2chat70b",
}

#: Regions the AWS pricing page lists for the DEFAULT_MODEL_PRICES rates.
DEFAULT_MODEL_PRICE_REGIONS: Final[tuple[str, ...]] = (
    "us-east-1",
    "us-east-2",
    "us-west-2",
)

#: Pricing-page per-generation fallback rates, absent from the Price List API.
DEFAULT_MODEL_PRICES: Final[dict[str, dict[Dimension, str]]] = {
    "stability.stable-image-remove-background-v1:0": {Dimension.OUTPUT_IMAGES: "0.07"},
    "stability.stable-image-erase-object-v1:0": {Dimension.OUTPUT_IMAGES: "0.07"},
    "stability.stable-image-control-structure-v1:0": {Dimension.OUTPUT_IMAGES: "0.07"},
    "stability.stable-image-control-sketch-v1:0": {Dimension.OUTPUT_IMAGES: "0.07"},
    "stability.stable-image-style-guide-v1:0": {Dimension.OUTPUT_IMAGES: "0.07"},
    "stability.stable-image-search-replace-v1:0": {Dimension.OUTPUT_IMAGES: "0.07"},
    "stability.stable-image-inpaint-v1:0": {Dimension.OUTPUT_IMAGES: "0.07"},
    "stability.stable-image-search-recolor-v1:0": {Dimension.OUTPUT_IMAGES: "0.07"},
    "stability.stable-style-transfer-v1:0": {Dimension.OUTPUT_IMAGES: "0.08"},
    "stability.stable-conservative-upscale-v1:0": {Dimension.OUTPUT_IMAGES: "0.40"},
    "stability.stable-creative-upscale-v1:0": {Dimension.OUTPUT_IMAGES: "0.60"},
    "stability.stable-fast-upscale-v1:0": {Dimension.OUTPUT_IMAGES: "0.03"},
    "stability.stable-outpaint-v1:0": {Dimension.OUTPUT_IMAGES: "0.06"},
    # OpenAI Mantle models: Bedrock pricing page per-1M rates / 1e6 (retrieved
    # 2026-07-19), absent from the Price List API outside GovCloud. Dated
    # aliases share these via MODEL_KEY_OVERRIDES.
    "openai.gpt-5.4": {
        Dimension.INPUT_TOKENS: "0.00000275",
        Dimension.CACHE_READ_TOKENS: "0.000000275",
        Dimension.OUTPUT_TOKENS: "0.0000165",
    },
    "openai.gpt-5.5": {
        Dimension.INPUT_TOKENS: "0.0000055",
        Dimension.CACHE_READ_TOKENS: "0.00000055",
        Dimension.OUTPUT_TOKENS: "0.000033",
    },
    "openai.gpt-5.6-luna": {
        Dimension.INPUT_TOKENS: "0.0000011",
        Dimension.CACHE_WRITE_TOKENS: "0.00000138",
        Dimension.CACHE_READ_TOKENS: "0.00000011",
        Dimension.OUTPUT_TOKENS: "0.0000066",
    },
    "openai.gpt-5.6-sol": {
        Dimension.INPUT_TOKENS: "0.0000055",
        Dimension.CACHE_WRITE_TOKENS: "0.00000688",
        Dimension.CACHE_READ_TOKENS: "0.00000055",
        Dimension.OUTPUT_TOKENS: "0.000033",
    },
    "openai.gpt-5.6-terra": {
        Dimension.INPUT_TOKENS: "0.00000275",
        Dimension.CACHE_WRITE_TOKENS: "0.00000344",
        Dimension.CACHE_READ_TOKENS: "0.00000028",
        Dimension.OUTPUT_TOKENS: "0.0000165",
    },
}
