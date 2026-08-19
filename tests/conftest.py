"""Shared pytest configuration and fixtures for the whole suite.

The suite runs against three interchangeable targets, selected by CLI flag:
the in-process ASGI app (default), a remote deployment (``--server-url``) or the
official OpenAI/Anthropic/Cohere APIs (``--use-official-api``). Every client and
model fixture resolves per target, so the same test body exercises the gateway and
the API it re-implements. ``MODEL_MAPPINGS`` and its Anthropic/Cohere counterparts
hold that per-target model choice; each entry is pinned deliberately (cheapest
model that has the capability under test) and must not be changed casually.

Environment variables are written at import time, before any test module can
import ``stdapi.config``, because ``SETTINGS`` is a module-level singleton built
from ``os.environ`` on first import.

Ref: stdapi/config.py:_Settings
     stdapi/main.py:app
"""

from __future__ import annotations

import base64
import faulthandler
import re
import signal
import sys
from io import BytesIO
from json import JSONDecodeError, dumps, loads
from os import environ, getenv
from pathlib import Path
from secrets import token_hex
from typing import TYPE_CHECKING, NamedTuple

import cohere
import httpx
import pytest
from aiobotocore.session import get_session
from anthropic import Anthropic, AnthropicBedrock
from anthropic import APIStatusError as AnthropicAPIStatusError
from dotenv import load_dotenv
from openai import APIStatusError, AsyncOpenAI, OpenAI
from PIL import Image as PILImage
from pybase64 import b64encode

# Starlette's TestClient prefers httpx2, which every vendor SDK rejects as an
# ``http_client``; aliasing the name while starlette binds it takes its httpx
# fallback. Dropped at once so genai-prices still gets the real httpx2.
assert "starlette.testclient" not in sys.modules, (
    "starlette.testclient was imported before this alias could be installed"
)
sys.modules["httpx2"] = httpx
try:
    from starlette.testclient import TestClient
finally:
    del sys.modules["httpx2"]

# A stall in an in-process live-AWS test just stops the suite; SIGUSR1 dumps every
# thread's stack, the only way to locate it on a host that restricts ptrace.
faulthandler.register(signal.SIGUSR1, all_threads=True)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator
    from typing import Any

    from pluggy import Result as _PluggyResult

    from stdapi.pricing import (
        CacheTtlBucket,
        ContextLength,
        Dimension,
        Routing,
        Service,
    )


def logged_usage_entries(
    captured_stdout: str,
    *,
    service: str | None = None,
    operation: str | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Extract usage entries from captured JSON logs, optionally filtered.

    Usage is only observable through the structured request log, so cost/metering
    tests read it back from captured stdout rather than from a return value.

    Args:
        captured_stdout: Captured stdout with one JSON log event per line.
        service: Filter to entries with this service.
        operation: Filter to entries with this operation.
        model: Filter to entries with this model.

    Returns:
        Matching usage entry dicts, in log order.
    """
    entries: list[dict[str, Any]] = []
    for line in captured_stdout.splitlines():
        if '"usage"' not in line:
            continue
        try:
            event = loads(line)
        except JSONDecodeError:
            continue
        entries.extend(
            usage
            for usage in event.get("usage", [])
            if (service is None or usage.get("service") == service)
            and (operation is None or usage.get("operation") == operation)
            and (model is None or usage.get("model") == model)
        )
    return entries


def set_test_price(
    model: str,
    region: str,
    dimension: Dimension,
    amount: str,
    currency: str,
    *,
    tier: str = "standard",
    cache_ttl: CacheTtlBucket = "",
    routing: Routing = "",
    spec: str = "",
    context: ContextLength = "",
    service: Service | None = None,
) -> None:
    """Seed the price index with one test price.

    For tests that need a real, resolvable price rather than a mocked
    ``resolve_price``. The defaults seed the standard tier, undifferentiated
    bucket; the keyword arguments seed a non-default ``PriceKey`` bucket.

    Args:
        model: Model ID, used as-is (not normalized) -- pick a name with no
            "-"/"_" separators so ``normalize_model_key`` cannot alter it.
        region: AWS region.
        dimension: The billed dimension.
        amount: The unit price, as a decimal string.
        currency: The currency code.
        tier: Service tier (standard, flex, priority, batch).
        cache_ttl: Cache TTL bucket ("5m"/"1h"), for CACHE_WRITE_TOKENS.
        routing: Serving profile ("global", "latency" or "").
        spec: Media/image spec bucket -- see ``PriceKey.spec``.
        context: Context-length bucket ("long" or "").
        service: The priced service; defaults to ``Service.BEDROCK``.
    """
    from decimal import Decimal  # noqa: PLC0415

    from stdapi.pricing import Price, PriceKey, Service, _state  # noqa: PLC0415

    key = PriceKey(
        service or Service.BEDROCK,
        model,
        region,
        dimension,
        tier,
        cache_ttl,
        routing,
        spec,
        context,
    )
    # Swap (don't mutate): model_prices caches a per-index grouping by identity.
    _state.price_index = {**_state.price_index, key: Price(Decimal(amount), currency)}


@pytest.fixture(autouse=True)
def _clean_input_files() -> Generator[None]:
    """Reset the per-request InputFile registry around each test.

    ``_CURRENT_INPUT_FILES`` is bound per request in production but is just a
    context variable here, so a file registered by one test stays visible to the
    next. The next test's ``prefetch_all_content_types`` then resolves a stranger's
    S3-backed file and reaches for a client it never set up.
    """
    from stdapi.input_file import _CURRENT_INPUT_FILES  # noqa: PLC0415

    token = _CURRENT_INPUT_FILES.set([])
    yield
    _CURRENT_INPUT_FILES.reset(token)


@pytest.fixture(autouse=True)
def _clean_price_index() -> Generator[None]:
    """Reset the price index around each test so seeded prices cannot leak.

    ``set_test_price`` mutates process-wide pricing state; without this the first
    test to seed a price would change every later cost assertion.
    """
    from stdapi.pricing import _state  # noqa: PLC0415

    original = _state.price_index
    _state.price_index = {}
    yield
    _state.price_index = original


@pytest.fixture(autouse=True)
def _keep_bidi_client_pool() -> Generator[None]:
    """Put back the bidirectional client pool a test's teardown emptied.

    ``AWSConnectionManager`` drops that pool whenever it unwinds, so every test
    driving the startup or shutdown path empties the one the session lifespan
    built, and each later bidirectional test fails with ``KeyError``. The
    clients are only dereferenced there, never closed, so restoring them works.
    """
    from sys import modules  # noqa: PLC0415

    bidi = modules.get("stdapi.aws_bidi")
    pool: dict[str, dict[str, Any]] = getattr(bidi, "_BIDI_CLIENTS", {})
    snapshot = {service: dict(clients) for service, clients in pool.items()}
    yield
    if snapshot and not pool:
        pool.update(snapshot)


_loaded_env_file: str | None = None


def _load_env_profile() -> None:
    """Load environment variables from a profile .env file.

    Builds a candidate list of dotenv files and loads the first one that exists:

    1. ``--env-profile <name>`` (CLI or ``PYTEST_ENV_PROFILE`` env var) adds
       ``tests/.env.<name>`` as the highest-priority candidate.
    2. ``--use-official-api`` appends ``tests/.env.use-official-api``.
    3. ``--server-url`` appends ``tests/.env.server-url``.
    4. ``tests/.env`` is always appended as the default fallback.

    This runs at module import time (before pytest parses arguments) so that
    ``PYTEST_ADDOPTS`` from the ``.env`` file is available during arg parsing.
    """
    global _loaded_env_file  # noqa: PLW0603

    def _argv_value(argv: list[str], option: str) -> str:
        """Return the value of a ``--option value`` or ``--option=value`` CLI arg."""
        prefix = f"{option}="
        for i, arg in enumerate(argv):
            if arg == option and i + 1 < len(argv):
                return argv[i + 1]
            if arg.startswith(prefix):
                return arg.split("=", 1)[1]
        return ""

    argv = sys.argv
    profile = _argv_value(argv, "--env-profile") or environ.get(
        "PYTEST_ENV_PROFILE", ""
    )
    tests_dir = Path(__file__).parent
    candidates = (
        f".env.{profile}" if profile else "",
        ".env.use-official-api" if "--use-official-api" in argv else "",
        ".env.server-url" if _argv_value(argv, "--server-url") else "",
        ".env",
    )
    for name in candidates:
        if name and (env_file := tests_dir / name).is_file():
            _loaded_env_file = str(env_file.relative_to(tests_dir.parent))
            load_dotenv(env_file, override=True)
            return


_load_env_profile()
del _load_env_profile

# ---------------------------------------------------------------------------
# Early environment setup
#
# Set at module level: ``stdapi.config.SETTINGS`` is a singleton built from
# ``os.environ`` by the first import of ``stdapi.config``, which any test module
# importing stdapi at collection time would trigger with the defaults instead.
# ---------------------------------------------------------------------------

#: Fixed API key shared by the test server and all test clients.
_TEST_API_KEY: str = token_hex()

environ.update(
    {
        # Authentication — must match the key used by the test clients.
        "api_key": _TEST_API_KEY,
        # Ensure all optional features are enabled so their tests can run.
        "aws_bedrock_allow_application_inference_profile_arn": "true",
        "aws_bedrock_allow_prompt_router_arn": "true",
        # OAuth 2.0 discovery: publishes the metadata document named in every 401.
        "oauth_resource_identifier": "https://gateway.tests.stdapi.ai",
        "oauth_authorization_servers": (
            "https://cognito-idp.eu-west-3.amazonaws.com/eu-west-3_tEsTpOoL1"
        ),
        "oauth_scopes_supported": "stdapi/invoke",
        # Ensure invalid inputs in tests are detected.
        "strict_input_validation": "true",
        # Avoid "too many reqsuests" error with Pytest xdist and many CPU
        "aws_adaptive_retry": "true",
        # Enable all optional middlewares so their behaviour is tested.
        "cors_allow_origins": '["*"]',
        "enable_gzip": "true",
        "enable_mcp_streamable_http": "true",
        "enable_proxy_headers": "true",
        "log_client_ip": "true",
        "log_request_params": "true",
        "model_cache_seconds": "10",
        "otel_enabled": "true",
        # Instrument every request but sample no span: no collector listens on the
        # exporter endpoint, and its retries drown the captured server output.
        "otel_sample_rate": "0.0",
        "cloudwatch_metrics": "true",
        "trusted_hosts": '["*"]',
        # Model-specific extra configuration
        "aws_bedrock_legacy": "true",
        "aws_bedrock_model_region_restrict": dumps(
            {
                # Required for system tools (Like "nova_grounding")
                "amazon.nova-2-lite-v1:0": ["us-east-1"],
                # kimi-k2.5 us-east-1 deployment has intermittent 60 s per-turn latency
                # spikes that cause T-CO test timeouts; us-west-2 is consistently fast.
                "moonshotai.kimi-k2.5": ["us-west-2"],
            }
        ),
        # Default service tier per model (used in tests for service tier defaulting)
        "default_model_service_tiers": dumps(
            {
                # amazon.nova-micro-v1:0 is a lighter model used in many tests
                "amazon.nova-micro-v1:0": "default"
            }
        ),
    }
)
# Disable cost tracking unless a profile enables it (avoids AWS Pricing API calls).
environ.setdefault("cost_tracking", "false")
# Usage-log tests require info-level request logs to capture JSON usage events.
environ.setdefault("log_level", "info")
# Pin Bedrock Mantle to one region: deterministic surfaces and stored-response locality.
environ.setdefault("aws_bedrock_mantle_regions", "us-east-1")
# Serve the dual-homed Gemma 3 and Luna test models via Mantle, not bedrock-runtime.
environ.setdefault(
    "aws_bedrock_mantle_preferred_models", "google.gemma-3-4b-it,openai.gpt-5.6-luna"
)

# Model mappings for different test contexts
MODEL_MAPPINGS = {
    "local": {
        "transcription": "amazon.transcribe",
        "transcription_stream": "amazon.transcribe",
        "transcription_diarize": "amazon.transcribe",
        "speech_standard": "amazon.polly-standard",
        "speech_generative": "amazon.polly-generative",
        "chat": "amazon.nova-micro-v1:0",
        "completion": "amazon.nova-micro-v1:0",
        "chat_vision": "amazon.nova-lite-v1:0",
        # Claude Haiku 4.5 judges image outputs: stronger VLM than the cheap vision model.
        "chat_vision_judge": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "chat_legacy": "amazon.nova-micro-v1:0",
        "chat_reasoning": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "chat_audio": "mistral.voxtral-mini-3b-2507",
        "embedding": "amazon.titan-embed-text-v2:0",
        "responses": "amazon.nova-micro-v1:0",
        # Bedrock ``outputConfig`` is rejected by Nova: keep the cheapest Claude.
        "responses_json_output": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "responses_web_search": "amazon.nova-2-lite-v1:0",
        "responses_code_interpreter": "amazon.nova-2-lite-v1:0",
        # Bedrock CountTokens only supports Anthropic models; the call is unbilled.
        "input_tokens": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "image_generation": "stability.stable-image-core-v1:1",
        # Nova Canvas is the only model mapping OpenAI ``quality``/``style`` to Bedrock
        # params, and legacy on purpose: they are reachable nowhere else (#93).
        "image_generation_hd": "amazon.nova-canvas-v1:0",
        "image_generation_stream": "stability.stable-image-core-v1:1",
        # Luma is the only non-legacy video model (Nova Reel is LEGACY on AWS).
        "video_generation": "luma.ray-v2:0",
        "realtime": "amazon.nova-2-sonic-v1:0",
    },
    "openai": {
        "transcription": "whisper-1",
        "transcription_stream": "gpt-4o-mini-transcribe",
        "transcription_diarize": "gpt-4o-transcribe-diarize",
        "speech_standard": "tts-1",
        "speech_generative": "tts-1",
        "chat": "gpt-5-nano",
        "completion": "gpt-3.5-turbo-instruct",
        "chat_vision": "gpt-5-nano",
        "chat_vision_judge": "gpt-5-nano",
        "chat_legacy": "gpt-4o-mini",
        "chat_reasoning": "gpt-5-nano",
        "chat_audio": "gpt-audio",
        "embedding": "text-embedding-3-small",
        "responses": "gpt-5-nano",
        "responses_json_output": "gpt-5-nano",
        "responses_web_search": "gpt-5-nano",
        "responses_code_interpreter": "gpt-5-nano",
        "input_tokens": "gpt-4o-mini",
        # gpt-image-1 is the only image model OpenAI still serves (the DALL-E family
        # is refused with "The model ... does not exist"), and it has no ``style``.
        "image_generation": "gpt-image-1",
        "image_generation_hd": "gpt-image-1",
        "image_generation_stream": "gpt-image-1",
        "video_generation": "sora-2",  # Cheapest video model
        "realtime": "gpt-realtime-mini",  # Cheapest realtime model
    },
}

#: finish_reason values the OpenAI Chat Completions reference defines.
FINISH_REASONS = frozenset({"stop", "length", "content_filter", "tool_calls"})

#: Amazon Nova Canvas, the only Bedrock image model mapping OpenAI quality/style.
NOVA_CANVAS_V1 = "amazon.nova-canvas-v1:0"

#: Nova Canvas roster for the sweeps that must cover every Nova Canvas image model.
NOVA_CANVAS_ALL = (NOVA_CANVAS_V1,)

#: Nova Canvas roster for the cases where one representative model is enough.
NOVA_CANVAS_SAMPLE = (NOVA_CANVAS_V1,)

#: Amazon Titan Image Generator V2, the image backend the Titan tests exercise.
TITAN_V2 = "amazon.titan-image-generator-v2:0"

#: Titan roster for the sweeps that must cover every Titan image model.
TITAN_ALL = (TITAN_V2,)

#: Titan roster for the cases where one representative model is enough.
TITAN_SAMPLE = (TITAN_V2,)

#: TwelveLabs Pegasus, the Bedrock-only video understanding model.
PEGASUS_MODEL = "twelvelabs.pegasus-1-2-v1:0"

#: Smallest image size each model accepts, within its cheapest billing tier.
#:
#: Bedrock image models bill per resolution tier -- Titan's low tier is anything up
#: to 512px, Nova Canvas tiers at 1024/2048/4096 -- so the smallest size a model
#: supports is the cheapest way to exercise it.
#:
#: Ref: stdapi/models/image/amazon_titan_image_generator.py:image_spec
IMAGE_MODEL_SIZES: dict[str, str] = {
    # Sizes are converted to the nearest supported aspect ratio and billed flat.
    "stability.stable-image-core-v1:1": "1024x1024",
    "stability.stable-image-inpaint-v1:0": "1024x1024",
    # 320-4096 divisible by 16; 512 is well inside the cheapest (<=1024) tier.
    "amazon.nova-canvas-v1:0": "512x512",
    # 512 is both the default and the top of the low pricing tier.
    "amazon.titan-image-generator-v1": "512x512",
    "amazon.titan-image-generator-v2:0": "512x512",
    # The only sizes gpt-image-1 accepts are 1024x1024, 1024x1536 and 1536x1024.
    "gpt-image-1": "1024x1024",
}

#: Size used for a model absent from :data:`IMAGE_MODEL_SIZES`.
_DEFAULT_IMAGE_SIZE = "1024x1024"

#: Image models that always answer with base64 and reject ``response_format``.
_B64_ONLY_IMAGE_MODELS = frozenset({"gpt-image-1"})

#: Sizes accepted by models that only take a fixed set. A model absent from this
#: table accepts arbitrary ``WIDTHxHEIGHT`` values -- the Bedrock backends map them
#: onto their own resolutions or aspect ratios.
IMAGE_MODEL_ACCEPTED_SIZES: dict[str, frozenset[str]] = {
    "gpt-image-1": frozenset({"1024x1024", "1024x1536", "1536x1024"})
}


def image_size_supported(model: str, size: str) -> bool:
    """True when *model* accepts *size*.

    Args:
        model: Image model ID, as resolved for the current target.
        size: A ``"<width>x<height>"`` size.

    Returns:
        True unless the model enumerates its sizes and *size* is not among them.
    """
    accepted = IMAGE_MODEL_ACCEPTED_SIZES.get(model)
    return accepted is None or size in accepted


def image_returns_base64_only(model: str) -> bool:
    """True when *model* always answers with base64 and never a URL.

    ``gpt-image-1`` has no ``response_format`` parameter and only returns
    ``b64_json``, so a test asserting a URL has to branch on this.

    Args:
        model: Image model ID, as resolved for the current target.

    Returns:
        Whether the model's responses only ever carry base64 data.
    """
    return model in _B64_ONLY_IMAGE_MODELS


def smallest_image_size(model: str) -> str:
    """Return the cheapest image size to request from *model*.

    Args:
        model: Image model ID, as resolved for the current target.

    Returns:
        A ``"<width>x<height>"`` size, defaulting to 1024x1024 for an unlisted
        model since every backend in use accepts it.
    """
    return IMAGE_MODEL_SIZES.get(model, _DEFAULT_IMAGE_SIZE)


_CACHE_DIR = Path(__file__).parent / ".cache"
#: Runs of characters replaced by a dash in a model ID used inside a cache filename.
_UNSAFE_CACHE_NAME_CHARS = re.compile(r"[^A-Za-z0-9]+")
#: Repository checkout root, for tests that must reference real source paths.
REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = Path(__file__).parent / "samples"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
_OPENAI_ORGANIZATION = "tests_stdapi.ai"
#: Markers whose tests are collected only when the matching ``--<marker>`` flag is passed.
_OPT_IN_MARKERS = ("expensive", "agentic", "slow", "video", "container", "drift")
#: Fallback skip reason for a ``gateway`` marker that names none of its own.
_GATEWAY_SKIP_REASON = "Exercises a gateway-only capability (official API selected)"
#: Message every route of an optional API answers when the operator left it unconfigured.
_FEATURE_DISABLED = "not available on the current server"
#: Seconds the loopback server backing the WebSocket lane gets to bind and boot.
_LIVE_SERVER_BOOT_TIMEOUT = 120.0
#: Seconds that server gets to stop before the session moves on without it.
_LIVE_SERVER_STOP_TIMEOUT = 30.0
#: Extra attempts a ``retry`` test gets when it names no count of its own.
_DEFAULT_RERUNS = 2
#: Seconds between two attempts, long enough for a written cache entry to be visible.
_DEFAULT_RERUN_DELAY = 3.0
#: Root fixtures reaching a live service; a test whose closure holds one cannot run offline.
_LIVE_FIXTURES = frozenset(
    {
        "anthropic_client",
        "async_openai_client",
        "aws_session_info",
        "bedrock_user_role_arn",
        "cognito_sandbox_pool",
        "cohere_client",
        "indexing_job_queue",
        "live_guardrail",
        "live_server",
        "openai_client",
        "test_client",
    }
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the target-selection and opt-in-marker command line options."""
    parser.addoption(
        "--env-profile",
        action="store",
        default="",
        help="Name of env profile to load (e.g. 'remote' loads tests/.env.remote). "
        "Default loads tests/.env if it exists. "
        "Can also be set via the PYTEST_ENV_PROFILE environment variable.",
    )
    parser.addoption(
        "--server-url",
        action="store",
        default=None,
        help="URL of the server to test against instead of using test client",
    )
    parser.addoption(
        "--use-official-api",
        action="store_true",
        default=False,
        help="Run tests against an official API (OpenAI, Anthropic, etc.) instead of local implementation",
    )
    parser.addoption(
        "--expensive",
        action="store_true",
        default=False,
        help="Run compute/cost expensive tests",
    )
    parser.addoption(
        "--agentic", action="store_true", default=False, help="Run agentic tests"
    )
    parser.addoption(
        "--agentic-rebuild",
        action="store_true",
        default=False,
        help="Rebuild the agentic CLI container image, re-resolving the tools' "
        "'@latest' versions instead of reusing the cached image",
    )
    parser.addoption(
        "--slow", action="store_true", default=False, help="Run slow tests"
    )
    parser.addoption(
        "--video", action="store_true", default=False, help="Run video generation tests"
    )
    parser.addoption(
        "--container",
        action="store_true",
        default=False,
        help="Build the container images and run the tests against them",
    )
    parser.addoption(
        "--drift",
        action="store_true",
        default=False,
        help="Compare hardcoded vendor facts against their published source "
        "(needs public internet, no AWS credentials and no vendor key)",
    )
    parser.addoption(
        "--offline",
        action="store_true",
        default=False,
        help="Run only tests needing no AWS credentials and no vendor API key, for CI",
    )


def pytest_report_header() -> str | None:
    """Show which env file was loaded in the pytest session header."""
    if _loaded_env_file:
        return f"envfile: {_loaded_env_file}"
    return None


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip opt-in tests, and tests their target cannot serve, at collection time.

    ``local``-marked tests are skipped when ``--server-url`` or
    ``--use-official-api`` selects a remote target; ``gateway``-marked tests are
    skipped only for ``--use-official-api``, since a deployed gateway can serve
    them but the upstream vendors cannot; the ``_OPT_IN_MARKERS`` tests are
    skipped unless their matching flag is passed.

    ``gateway`` takes an optional reason -- ``@pytest.mark.gateway("Amazon Polly
    is not available on the official OpenAI API")`` -- so a whole module or class
    can declare once why it is skipped.

    ``--offline`` keeps only the tests that need neither AWS credentials nor a
    vendor API key, by skipping every test whose fixture closure contains one of
    ``_LIVE_FIXTURES``. Deriving the set from the closure rather than a marker
    means a new test is classified by the fixtures it requests, with nothing to
    remember to annotate.

    ``retry`` is translated into the rerun plugin's own marker here rather than
    used directly, so that every retried test has to state why it is allowed to
    fail once. A retry hides a real regression if it is ever added without one.

    Raises:
        pytest.UsageError: If a ``retry`` marker states no reason.
    """

    def skip(reason: str, matches: Callable[[pytest.Item], bool]) -> None:
        """Add a skip marker carrying *reason* to every item *matches* selects."""
        marker = pytest.mark.skip(reason=reason)
        for item in items:
            if matches(item):
                item.add_marker(marker)

    def marked(name: str) -> Callable[[pytest.Item], bool]:
        """Return a predicate selecting the items carrying the *name* marker."""
        return lambda item: item.get_closest_marker(name) is not None

    def needs_live_service(item: pytest.Item) -> bool:
        """Whether *item* pulls a live AWS/vendor/gateway fixture into its closure."""
        return bool(_LIVE_FIXTURES.intersection(getattr(item, "fixturenames", ())))

    def skip_gateway_only() -> None:
        """Skip ``gateway`` items, preferring the reason the marker carries."""
        for item in items:
            marker = item.get_closest_marker("gateway")
            if marker is not None:
                reason = str(marker.args[0]) if marker.args else _GATEWAY_SKIP_REASON
                item.add_marker(pytest.mark.skip(reason=reason))

    def apply_retries() -> None:
        """Rewrite every ``retry`` marker as the rerun plugin's ``flaky``."""
        for item in items:
            marker = item.get_closest_marker("retry")
            if marker is None:
                continue
            if not marker.args or not str(marker.args[0]).strip():
                msg = f"{item.nodeid}: @pytest.mark.retry must state a reason"
                raise pytest.UsageError(msg)
            item.add_marker(
                pytest.mark.flaky(
                    reruns=marker.kwargs.get("reruns", _DEFAULT_RERUNS),
                    reruns_delay=marker.kwargs.get("delay", _DEFAULT_RERUN_DELAY),
                )
            )

    if config.getoption("--server-url") or config.getoption("--use-official-api"):
        skip("Tests the local implementation (remote target selected)", marked("local"))
    if config.getoption("--use-official-api"):
        skip_gateway_only()
    if config.getoption("--offline"):
        skip("Needs a live service (--offline selected)", needs_live_service)
    for opt_in in _OPT_IN_MARKERS:
        if not config.getoption(f"--{opt_in}"):
            skip(f"Need --{opt_in} option to run this test", marked(opt_in))
    apply_retries()


@pytest.fixture(scope="session")
def use_official_api(request: pytest.FixtureRequest) -> bool:
    """True when the suite targets an official API instead of the gateway."""
    return request.config.getoption("--use-official-api")  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def server_url(request: pytest.FixtureRequest) -> str | None:
    """Base URL of the remote gateway under test, or None when none was selected.

    Normalised once here (trailing slash stripped) so every client fixture can
    append its own route prefix.
    """
    url: str | None = request.config.getoption("--server-url")
    return url.rstrip("/") if url else None


@pytest.fixture(scope="session")
def models(use_official_api: bool) -> dict[str, str]:
    """Per-capability model IDs for the selected target (Bedrock IDs, or OpenAI IDs)."""
    return MODEL_MAPPINGS["openai" if use_official_api else "local"]


@pytest.fixture(scope="session")
def transcription_model(models: dict[str, str]) -> str:
    """Model for batch transcription (Amazon Transcribe locally)."""
    return models["transcription"]


@pytest.fixture(scope="session")
def transcription_stream_model(models: dict[str, str]) -> str:
    """Model for streaming transcription."""
    return models["transcription_stream"]


@pytest.fixture(scope="session")
def transcription_diarize_model(models: dict[str, str]) -> str:
    """Model for transcription with speaker partitioning (diarization)."""
    return models["transcription_diarize"]


@pytest.fixture(scope="session")
def speech_standard_model(models: dict[str, str]) -> str:
    """Model for text-to-speech using the standard (non-neural) engine."""
    return models["speech_standard"]


@pytest.fixture(scope="session")
def speech_generative_model(models: dict[str, str]) -> str:
    """Model for text-to-speech with the voices that stream long inputs."""
    return models["speech_generative"]


@pytest.fixture(scope="session")
def chat_model(models: dict[str, str]) -> str:
    """Cheapest text-only chat model."""
    return models["chat"]


@pytest.fixture(scope="session")
def completion_model(models: dict[str, str]) -> str:
    """Model for the legacy /v1/completions route."""
    return models["completion"]


@pytest.fixture(scope="session")
def chat_vision_model(models: dict[str, str]) -> str:
    """Cheapest chat model accepting IMAGE input."""
    return models["chat_vision"]


@pytest.fixture(scope="session")
def chat_vision_judge_model(models: dict[str, str]) -> str:
    """Stronger vision model used to grade generated images in image tests."""
    return models["chat_vision_judge"]


@pytest.fixture(scope="session")
def chat_reasoning_model(models: dict[str, str]) -> str:
    """Chat model exposing reasoning/extended thinking."""
    return models["chat_reasoning"]


@pytest.fixture(scope="session")
def chat_legacy_model(models: dict[str, str]) -> str:
    """Chat model accepting the deprecated ``functions``/``function_call`` input."""
    return models["chat_legacy"]


@pytest.fixture(scope="session")
def chat_audio_model(models: dict[str, str]) -> str:
    """Chat model accepting audio input and producing audio output."""
    return models["chat_audio"]


@pytest.fixture(scope="session")
def embedding_model(models: dict[str, str]) -> str:
    """Default text embeddings model."""
    return models["embedding"]


@pytest.fixture(scope="session")
def responses_model(models: dict[str, str]) -> str:
    """Cheapest model for the Responses API."""
    return models["responses"]


@pytest.fixture(scope="session")
def responses_json_output_model(models: dict[str, str]) -> str:
    """Provide a model for Responses API JSON output (Bedrock ``outputConfig``).

    Local: Claude Haiku 4.5 (supports Bedrock Converse ``outputConfig``).
    Official API: ``gpt-5-nano``.
    """
    return models["responses_json_output"]


@pytest.fixture(scope="session")
def responses_web_search_model(models: dict[str, str]) -> str:
    """Responses-API model supporting the hosted web-search tool."""
    return models["responses_web_search"]


@pytest.fixture(scope="session")
def responses_code_interpreter_model(models: dict[str, str]) -> str:
    """Provide the appropriate model for Responses API code interpreter (executes code autonomously).

    Local: Amazon Nova 2 Lite (``nova_code_interpreter`` system tool).
    Official API: ``gpt-5-nano`` (native Python execution via OpenAI code interpreter).
    """
    return models["responses_code_interpreter"]


@pytest.fixture(scope="session")
def responses_input_tokens_model(models: dict[str, str]) -> str:
    """Model for /responses/input_tokens (Bedrock CountTokens is Anthropic-only)."""
    return models["input_tokens"]


@pytest.fixture(scope="session")
def image_generation_model(models: dict[str, str]) -> str:
    """Cheapest image generation model, used as the default."""
    return models["image_generation"]


@pytest.fixture(scope="session")
def image_generation_hd_model(models: dict[str, str]) -> str:
    """Image model mapping OpenAI ``quality``/``style`` onto backend parameters."""
    return models["image_generation_hd"]


@pytest.fixture(scope="session")
def image_generation_stream_model(models: dict[str, str]) -> str:
    """Image model used for streaming partial images."""
    return models["image_generation_stream"]


@pytest.fixture(scope="session")
def image_generation_size(image_generation_model: str) -> str:
    """Cheapest size accepted by ``image_generation_model``."""
    return smallest_image_size(image_generation_model)


@pytest.fixture(scope="session")
def image_generation_hd_size(image_generation_hd_model: str) -> str:
    """Cheapest size accepted by ``image_generation_hd_model``."""
    return smallest_image_size(image_generation_hd_model)


@pytest.fixture(scope="session")
def image_generation_stream_size(image_generation_stream_model: str) -> str:
    """Cheapest size accepted by ``image_generation_stream_model``."""
    return smallest_image_size(image_generation_stream_model)


@pytest.fixture(scope="session")
def video_generation_model(models: dict[str, str]) -> str:
    """Cheapest non-legacy video generation model."""
    return models["video_generation"]


@pytest.fixture(scope="session")
def realtime_model(models: dict[str, str]) -> str:
    """Cheapest speech-to-speech model served over the Realtime WebSocket."""
    return models["realtime"]


@pytest.fixture(scope="session")
def api_key() -> str:
    """API key shared by the test server and every local client."""
    return _TEST_API_KEY


class CognitoSandboxPool(NamedTuple):
    """The real Amazon Cognito user pool the token tests obtain a token from.

    ``foreign_client_id``/``foreign_client_secret`` name a second application of
    the same pool, allowed the same scope: a token minted by it differs from an
    accepted one only in the application it was issued to.
    """

    user_pool_id: str
    token_url: str
    scope: str
    client_id: str
    client_secret: str
    foreign_client_id: str
    foreign_client_secret: str


@pytest.fixture(scope="session")
def cognito_sandbox_pool() -> CognitoSandboxPool:
    """Coordinates of the user pool provisioned for this checkout, or skip.

    The pool is created by the ``terraform-sandbox`` stack and wired into
    ``tests/.env``; a checkout without one skips rather than fails, since no
    test can create a pool for itself.

    Returns:
        The pool, its token endpoint, and the two applications registered in it.
    """
    values = {
        name: getenv(f"TEST_COGNITO_{name.upper()}", "")
        for name in CognitoSandboxPool._fields
    }
    if missing := sorted(name for name, value in values.items() if not value):
        pytest.skip(
            "No Amazon Cognito user pool is configured for this checkout "
            f"(tests/.env sets no TEST_COGNITO_{missing[0].upper()})"
        )
    return CognitoSandboxPool(**values)


@pytest.fixture(scope="session")
def test_client(
    use_official_api: bool, server_url: str | None
) -> Generator[TestClient | None]:
    """In-process ASGI test client, or None when a remote target was selected.

    Yielding None (rather than skipping) lets every dependent fixture and test
    decide for itself whether it can run against a remote target. Entering the
    ``TestClient`` context runs the app's lifespan, so startup work such as
    authentication initialisation and MCP mounting happens exactly once per session.
    """
    if not use_official_api and not server_url:
        from stdapi.main import app  # noqa: PLC0415

        with TestClient(app) as test_client:
            yield test_client
    else:
        yield None


@pytest.fixture(scope="session")
def local_test_client(test_client: TestClient | None) -> TestClient:
    """``test_client``, skipping the test when a remote target was selected.

    For tests that reach a real backend yet still need the in-process app -- to
    read the gateway's own usage log, or to patch its process state. The
    ``local`` marker cannot express that: it also exempts a test from the
    unavailable-model xfail mask, which only holds for tests that call nothing.

    Returns:
        The in-process ASGI test client.
    """
    if test_client is None:
        pytest.skip("Requires the in-process app (remote target selected)")
    return test_client


@pytest.fixture
def app_client(api_key: str) -> TestClient:
    """Pre-authenticated ASGI client that does **not** run the app lifespan.

    Unit tests that only exercise routing, validation and error shaping use this
    instead of ``test_client``: skipping the lifespan skips the AWS startup work,
    so they stay in-process and free. Use ``test_client`` when the app's startup
    state (authentication init, MCP mounting) is part of what is under test.
    """
    from stdapi.main import app  # noqa: PLC0415

    return TestClient(app, headers={"Authorization": f"Bearer {api_key}"})


@pytest.fixture
def anthropic_app_client(api_key: str) -> TestClient:
    """``app_client`` with the Anthropic route's own auth and version headers."""
    from stdapi.main import app  # noqa: PLC0415

    return TestClient(
        app, headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    )


@pytest.fixture
def request_log() -> Generator[dict[str, Any]]:
    """Bind a request-log context for code that writes into it outside a request.

    ``set_effective_region`` and the usage/metering helpers append to the
    ``REQUEST_LOG`` context variable, which only exists inside a real request.
    Calling them directly raises ``LookupError`` without this.

    Yields:
        The mutable log dict, so a test can assert what the code under test wrote.
    """
    from stdapi.monitoring import REQUEST_LOG  # noqa: PLC0415

    log: dict[str, Any] = {"level": "info"}
    token = REQUEST_LOG.set(log)  # type: ignore[arg-type]
    yield log
    REQUEST_LOG.reset(token)


@pytest.fixture
def usage_scope() -> Generator[None]:
    """Install fresh per-request usage, model-state and image-spec scopes.

    Metering accumulates into context variables that only exist inside a real
    request; calling the recorders directly raises ``LookupError`` without this.
    """
    from stdapi import usage  # noqa: PLC0415

    usage_token = usage.init_usage()
    state_token = usage.init_model_state()
    image_spec_token = usage.IMAGE_SPEC.set("")
    yield
    usage.USAGE.reset(usage_token)
    usage.MODEL_STATE.reset(state_token)
    usage.IMAGE_SPEC.reset(image_spec_token)


@pytest.fixture
def configured_guardrail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configure a default server guardrail, as ``moderation: true`` requires."""
    from stdapi.config import SETTINGS  # noqa: PLC0415

    monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_identifier", "gr123")
    monkeypatch.setattr(SETTINGS, "aws_bedrock_guardrail_version", "1")


@pytest.fixture(scope="session")
def live_guardrail(use_official_api: bool) -> Iterator[str]:
    """Create one temporary guardrail for the whole session.

    Blocks the word ``BLOCKWORDXYZ`` and every content filter category, so a
    test can trip it deterministically. On the official API no guardrail
    exists; tests use the OpenAI moderation model instead. The server (local or
    --server-url) must allow guardrail overrides, which is automatic when no
    global guardrail is configured. Yields the guardrail ARN so the server
    resolves the guardrail's region.

    Creating one is slow and billable, hence the session scope: pin every
    consumer to the ``moderations_guardrail`` xdist group so a single worker
    creates it.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-content-filters.html
         stdapi/aws_bedrock.py:guardrail_region
    """
    if use_official_api:
        yield ""
        return
    import time  # noqa: PLC0415
    from uuid import uuid4  # noqa: PLC0415

    import boto3  # type: ignore[import-untyped]  # noqa: PLC0415

    from stdapi.config import SETTINGS  # noqa: PLC0415

    region = SETTINGS.aws_bedrock_regions[0]
    bedrock = boto3.client("bedrock", region_name=region)
    created = bedrock.create_guardrail(
        name=f"stdapi-tests-moderations-{uuid4().hex[:8]}",
        blockedInputMessaging="Blocked by test guardrail.",
        blockedOutputsMessaging="Blocked by test guardrail.",
        wordPolicyConfig={"wordsConfig": [{"text": "BLOCKWORDXYZ"}]},
        contentPolicyConfig={
            "filtersConfig": [
                {"type": name, "inputStrength": "HIGH", "outputStrength": "HIGH"}
                for name in ("HATE", "INSULTS", "SEXUAL", "VIOLENCE", "MISCONDUCT")
            ]
        },
    )
    guardrail_id = created["guardrailId"]
    try:
        for _ in range(30):
            status = bedrock.get_guardrail(guardrailIdentifier=guardrail_id)["status"]
            if status == "READY":
                break
            time.sleep(1)
        else:
            pytest.fail("Test guardrail never reached READY status.")
        yield created["guardrailArn"]
    finally:
        # A guardrail is billable: delete it on every path.
        bedrock.delete_guardrail(guardrailIdentifier=guardrail_id)


@pytest.fixture(scope="session")
def openai_client(
    use_official_api: bool,
    server_url: str | None,
    test_client: TestClient | None,
    api_key: str,
) -> OpenAI:
    """OpenAI SDK client bound to the selected target.

    Against the in-process app the SDK is given the ``TestClient`` as its HTTP
    transport and ``max_retries=0``, so a failure surfaces as the gateway's own
    status code instead of being retried away.
    """
    # Local test
    if test_client:
        return OpenAI(
            base_url="http://testserver/v1",
            api_key=api_key,
            max_retries=0,
            organization=_OPENAI_ORGANIZATION,
            # The agentic overlay pins an older `openai` whose client is typed
            # against httpx 1.x, where the suite runs on httpx 2.
            http_client=test_client,  # type: ignore[arg-type]
        )

    # Official API test
    if use_official_api:
        return OpenAI(max_retries=5)

    # Remote server test
    return OpenAI(
        base_url=f"{server_url}/v1", max_retries=0, organization=_OPENAI_ORGANIZATION
    )


@pytest.fixture(scope="session")
def live_server(use_official_api: bool, server_url: str | None) -> Iterator[str | None]:
    """Base URL of a target reachable over a real socket, or None for the vendor.

    A WebSocket route cannot be exercised through ``TestClient`` as an SDK
    transport: the official client dials the URL itself, with its own WebSocket
    stack. The in-process lane therefore serves the same ASGI app from a real
    uvicorn on a loopback port, running a second lifespan of it in the same
    process. That lifespan drops the shared AWS client pool when it unwinds, so
    the pool the ``test_client`` lifespan built is put back afterwards: any
    session fixture finalised later still answers through it.

    Yields:
        The HTTP base URL to dial, or None when the vendor's own endpoint is the
        target.
    """
    if use_official_api:
        yield None
        return
    if server_url:
        yield server_url
        return

    import socket  # noqa: PLC0415
    import threading  # noqa: PLC0415
    import time  # noqa: PLC0415

    import uvicorn  # noqa: PLC0415

    from stdapi.aws import _CLIENTS  # noqa: PLC0415
    from stdapi.main import app  # noqa: PLC0415

    pool = {service: dict(clients) for service, clients in _CLIENTS.items()}
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = int(listener.getsockname()[1])
    server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()
    try:
        deadline = time.monotonic() + _LIVE_SERVER_BOOT_TIMEOUT
        while not server.started:
            if time.monotonic() > deadline or not thread.is_alive():
                pytest.fail("the in-process WebSocket server did not start")
            time.sleep(0.05)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=_LIVE_SERVER_STOP_TIMEOUT)
        for service, clients in pool.items():
            _CLIENTS.setdefault(service, {}).update(clients)


@pytest.fixture(scope="session")
def async_openai_client(live_server: str | None, api_key: str) -> AsyncOpenAI:
    """Async OpenAI SDK client bound to a socket-reachable target.

    The WebSocket routes need this rather than ``openai_client``, whose in-process
    transport cannot upgrade a connection.
    """
    if live_server is None:
        return AsyncOpenAI(max_retries=5)
    return AsyncOpenAI(
        base_url=f"{live_server}/v1",
        api_key=api_key,
        max_retries=0,
        organization=_OPENAI_ORGANIZATION,
    )


@pytest.fixture(scope="session")
def sample_audio_pcm24(sample_audio_file: bytes) -> bytes:
    """The spoken WAV sample as 24 kHz mono 16-bit PCM, the Realtime input format."""
    import shutil  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg, "ffmpeg is required to condition the Realtime audio sample"
    converted = subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-v",
            "quiet",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            "24000",
            "pipe:1",
        ],
        input=sample_audio_file,
        capture_output=True,
        check=False,
    )
    assert converted.returncode == 0, converted.stderr.decode(errors="replace")
    assert converted.stdout, "ffmpeg produced no PCM for the speech sample"
    return converted.stdout


def _skip_when_feature_disabled(
    error: APIStatusError | AnthropicAPIStatusError, feature: str, setting: str
) -> None:
    """Skip the test when *error* is the answer of an API the target does not serve.

    An optional API answers the same "not available on the current server"
    envelope on every one of its routes when the operator configured none of
    the resources it needs. That is a deployment choice, not a failure, so the
    tests exercising it must skip rather than fail everywhere it is off.

    Args:
        error: Status error the probe request raised.
        feature: Name of the API, for the skip reason.
        setting: Setting the operator sets to enable it.
    """
    if error.status_code >= 500 and _FEATURE_DISABLED in str(error):
        pytest.skip(f"The target serves no {feature} (set {setting} to enable it)")


@pytest.fixture(scope="session")
def vector_stores_api(openai_client: OpenAI) -> None:
    """Skip the test unless the target actually serves the Vector Stores API.

    The target is asked rather than this process's settings read, so the answer
    is also true for a deployed gateway and for the official API.
    """
    try:
        openai_client.vector_stores.list(limit=1)
    except APIStatusError as error:
        _skip_when_feature_disabled(error, "Vector Stores API", "aws_s3_vectors_bucket")
        raise


@pytest.fixture(scope="session")
def batches_api(openai_client: OpenAI) -> None:
    """Skip the test unless the target actually serves the Batch API."""
    try:
        openai_client.batches.list(limit=1)
    except APIStatusError as error:
        _skip_when_feature_disabled(error, "Batch API", "aws_bedrock_batch_role_arn")
        raise


@pytest.fixture(scope="session")
def anthropic_batches_api(anthropic_client: Anthropic, is_bedrock_direct: bool) -> None:
    """Skip the test unless the target actually serves the Message Batches API."""
    if is_bedrock_direct:
        pytest.skip("Message Batches API not available on Bedrock")
    try:
        anthropic_client.messages.batches.list(limit=1)
    except AnthropicAPIStatusError as error:
        _skip_when_feature_disabled(
            error, "Message Batches API", "aws_bedrock_batch_role_arn"
        )
        raise


def _sample_cache_file(name: str, model: str, suffix: str) -> Path:
    """Return the on-disk cache path of a sample produced by *model*.

    The producing model differs per target, so it is part of the file name: an
    artefact synthesized by one target is never reused by another.

    Args:
        name: Sample name, e.g. ``"audio"``.
        model: ID of the model producing the sample.
        suffix: File extension, without the leading dot.

    Returns:
        Path under ``tests/.cache``.
    """
    return _CACHE_DIR / f"{name}-{_UNSAFE_CACHE_NAME_CHARS.sub('-', model)}.{suffix}"


@pytest.fixture(scope="session")
def sample_audio_file(openai_client: OpenAI, speech_standard_model: str) -> bytes:
    """Short WAV snippet produced once by the speech endpoint and cached on disk.

    Cached under ``tests/.cache`` per producing model so the whole suite costs a
    single synthesis call, and survives across sessions.
    """
    audio_file = _sample_cache_file("audio", speech_standard_model, "wav")
    if audio_file.exists():
        with audio_file.open("rb") as file:
            return file.read()
    content = openai_client.audio.speech.create(
        model=speech_standard_model,
        voice="alloy",
        input="This is a test.",
        response_format="wav",
    ).content
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with audio_file.open("wb") as file:
        file.write(content)
    return content


@pytest.fixture(scope="session")
def sample_audio_file_base64(sample_audio_file: bytes) -> str:
    """The WAV sample as a ``data:audio/wav;base64,`` URL."""
    return f"data:audio/wav;base64,{b64encode(sample_audio_file).decode('utf-8')}"


@pytest.fixture(scope="session")
def sample_audio_mp3_file(openai_client: OpenAI, speech_standard_model: str) -> bytes:
    """Short MP3 snippet produced once by the speech endpoint and cached on disk."""
    audio_file = _sample_cache_file("audio", speech_standard_model, "mp3")
    if audio_file.exists():
        with audio_file.open("rb") as file:
            return file.read()
    content = openai_client.audio.speech.create(
        model=speech_standard_model,
        voice="alloy",
        input="This is a test.",
        response_format="mp3",
    ).content
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with audio_file.open("wb") as file:
        file.write(content)
    return content


@pytest.fixture(scope="session")
def sample_audio_mp3_file_base64(sample_audio_mp3_file: bytes) -> str:
    """The MP3 sample as a ``data:audio/mp3;base64,`` URL."""
    return f"data:audio/mp3;base64,{b64encode(sample_audio_mp3_file).decode('utf-8')}"


@pytest.fixture(scope="session")
def sample_image_file(
    openai_client: OpenAI, image_generation_model: str, image_generation_size: str
) -> bytes:
    """PNG produced once by the Images API at the model's cheapest size, cached on disk.

    Base64 is requested rather than a URL so the fixture never has to fetch a
    presigned link. ``response_format`` is only sent to models that accept it:
    ``gpt-image-1`` always answers with base64 and rejects the parameter outright.
    """
    image_file = _sample_cache_file("image", image_generation_model, "png")
    if image_file.exists():
        with image_file.open("rb") as file:
            return file.read()

    if image_generation_model in _B64_ONLY_IMAGE_MODELS:
        response = openai_client.images.generate(
            prompt="A rainbow llama",
            model=image_generation_model,
            n=1,
            size=image_generation_size,
        )
    else:
        response = openai_client.images.generate(
            prompt="A rainbow llama",
            model=image_generation_model,
            n=1,
            size=image_generation_size,
            response_format="b64_json",
        )
    # Extract and decode base64 image
    data_list = response.data or []
    assert len(data_list) >= 1
    b64_data = data_list[0].b64_json
    assert b64_data is not None
    assert isinstance(b64_data, str)
    image_bytes = base64.b64decode(b64_data)

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with image_file.open("wb") as file:
        file.write(image_bytes)
    return image_bytes


@pytest.fixture(scope="session")
def sample_image_file_base64(sample_image_file: bytes) -> str:
    """The PNG sample as a ``data:image/png;base64,`` URL."""
    return f"data:image/png;base64,{b64encode(sample_image_file).decode('utf-8')}"


@pytest.fixture(scope="session")
def sample_mask_file(sample_image_file: bytes) -> bytes:
    """Bedrock-style RGB mask matching the sample image, white in the centre.

    White marks the region to edit for Titan/Nova inpainting.

    Returns:
        PNG mask bytes.
    """
    width, height = PILImage.open(BytesIO(sample_image_file)).size

    mask = PILImage.new("RGB", (width, height), color=(0, 0, 0))
    for x in range(width // 4, 3 * width // 4):
        for y in range(height // 4, 3 * height // 4):
            mask.putpixel((x, y), (255, 255, 255))

    buffer = BytesIO()
    mask.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def sample_alpha_mask_file(sample_image_file: bytes) -> bytes:
    """OpenAI-style RGBA mask matching the sample image, transparent in the centre.

    Transparency marks the region to edit, the polarity ``alpha_mask_to_bw``
    converts for Bedrock.

    Returns:
        PNG mask bytes with an alpha channel.
    """
    width, height = PILImage.open(BytesIO(sample_image_file)).size

    mask = PILImage.new("RGBA", (width, height), color=(0, 0, 0, 255))
    for x in range(width // 4, 3 * width // 4):
        for y in range(height // 4, 3 * height // 4):
            mask.putpixel((x, y), (0, 0, 0, 0))

    buffer = BytesIO()
    mask.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def sample_video_file() -> bytes:
    """Locally provided mp4 sample, skipping the test when it is absent.

    The clip cannot be synthesized cheaply, so it is supplied by hand; skipping
    here propagates to every dependent test instead of each one re-checking.
    """
    video_file = _CACHE_DIR / "video.mp4"
    if not video_file.exists():
        pytest.skip("Missing sample video: add an MP4 at tests/.cache/video.mp4")
    with video_file.open("rb") as file:
        return file.read()


@pytest.fixture(scope="session")
def sample_video_file_base64(sample_video_file: bytes) -> str:
    """The mp4 sample as a ``data:video/mp4;base64,`` URL."""
    return f"data:video/mp4;base64,{b64encode(sample_video_file).decode('utf-8')}"


@pytest.fixture(scope="session")
def sample_pdf_file() -> bytes:
    """Minimal valid single-page PDF containing the text "Hello World".

    Built inline rather than generated so document-input tests have byte-stable
    content with no extra dependency.

    Returns:
        PDF file bytes.
    """
    # Minimal PDF with "Hello World" text
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /Resources 4 0 R "
        b"/MediaBox [0 0 612 792] /Contents 5 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Font << /F1 << /Type /Font /Subtype /Type1 "
        b"/BaseFont /Helvetica >> >> >>\nendobj\n"
        b"5 0 obj\n<< /Length 44 >>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET\nendstream\nendobj\n"
        b"xref\n0 6\n0000000000 65535 f \n0000000009 00000 n \n"
        b"0000000058 00000 n \n0000000115 00000 n \n0000000228 00000 n \n"
        b"0000000327 00000 n \ntrailer\n<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n420\n%%EOF\n"
    )


@pytest.fixture(scope="session")
def sample_pdf_file_data_uri(sample_pdf_file: bytes) -> str:
    """The PDF sample as a ``data:application/pdf;base64,`` URI."""
    return f"data:application/pdf;base64,{b64encode(sample_pdf_file).decode('utf-8')}"


@pytest.fixture(scope="session")
async def aws_session_info() -> tuple[str, str]:
    """The session's AWS region and account ID, from one ``sts:GetCallerIdentity`` call.

    Session-scoped and shared so the suite makes a single STS call.

    Returns:
        ``(region, account_id)``.
    """
    session = get_session()
    async with session.create_client("sts") as sts:
        region = session.get_config_variable("region")
        return region, (await sts.get_caller_identity())["Account"]


@pytest.fixture(scope="session")
async def aws_region(aws_session_info: tuple[str, str]) -> str:
    """The AWS region the test session is configured for."""
    return aws_session_info[0]


@pytest.fixture(scope="session")
async def aws_account_id(aws_session_info: tuple[str, str]) -> str:
    """The AWS account ID the test session's credentials belong to."""
    return aws_session_info[1]


@pytest.fixture(scope="session")
def bedrock_user_role_arn() -> str:
    """ARN of the role end user sessions are opened on, or skip.

    The role is an IAM resource of the account under test, created outside this
    repository, and named by ``TEST_BEDROCK_USER_ROLE_ARN`` in ``tests/.env``;
    a checkout without one skips rather than fails, since no test can create
    the trust policy the feature needs.

    Returns:
        The role ARN per-end-user cost attribution assumes.
    """
    arn = getenv("TEST_BEDROCK_USER_ROLE_ARN", "")
    if not arn:
        pytest.skip(
            "Per-end-user cost attribution needs a role to assume "
            "(tests/.env sets no TEST_BEDROCK_USER_ROLE_ARN)"
        )
    return arn


# ---------------------------------------------------------------------------
# Anthropic API fixtures (shared across Anthropic test modules)
# ---------------------------------------------------------------------------

#: Model mappings for Anthropic /v1/messages tests.
ANTHROPIC_MODEL_MAPPINGS: dict[str, dict[str, str]] = {
    "local": {
        "chat": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "chat_basic": "amazon.nova-micro-v1:0",
        "chat_vision": "amazon.nova-lite-v1:0",
        "chat_reasoning": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "chat_system_as_messages": "anthropic.claude-opus-5",
        "count_tokens": "anthropic.claude-haiku-4-5-20251001-v1:0",
    },
    "anthropic": {
        "chat": "claude-haiku-4-5-20251001",
        "chat_basic": "claude-haiku-4-5-20251001",
        "chat_vision": "claude-haiku-4-5-20251001",
        "chat_reasoning": "claude-haiku-4-5-20251001",
        "chat_system_as_messages": "claude-opus-5",
        "count_tokens": "claude-haiku-4-5-20251001",
    },
}
ANTHROPIC_MODEL_MAPPINGS["bedrock"] = {
    key: f"global.{value}" for key, value in ANTHROPIC_MODEL_MAPPINGS["local"].items()
}
# Amazon models have no ``global.`` profile and reject the Anthropic-native payload:
# keep the official-API/Bedrock parity lane on Claude.
ANTHROPIC_MODEL_MAPPINGS["bedrock"]["chat_basic"] = (
    "global.anthropic.claude-haiku-4-5-20251001-v1:0"
)
ANTHROPIC_MODEL_MAPPINGS["bedrock"]["chat_vision"] = (
    "global.anthropic.claude-haiku-4-5-20251001-v1:0"
)
# CountTokens accepts the base model ID directly, no inference profile needed.
ANTHROPIC_MODEL_MAPPINGS["bedrock"]["count_tokens"] = (
    "anthropic.claude-haiku-4-5-20251001-v1:0"
)


@pytest.fixture(scope="session")
def anthropic_models(use_official_api: bool) -> dict[str, str]:
    """Per-capability Anthropic model IDs for the selected target."""
    return (
        (
            ANTHROPIC_MODEL_MAPPINGS["anthropic"]
            if getenv("ANTHROPIC_API_KEY")
            else ANTHROPIC_MODEL_MAPPINGS["bedrock"]
        )
        if use_official_api
        else ANTHROPIC_MODEL_MAPPINGS["local"]
    )


@pytest.fixture(scope="session")
def anthropic_chat_model(anthropic_models: dict[str, str]) -> str:
    """Default model for the Anthropic Messages route."""
    return anthropic_models["chat"]


@pytest.fixture(scope="session")
def anthropic_chat_basic_model(anthropic_models: dict[str, str]) -> str:
    """Cheapest Anthropic-route chat model, for API plumbing tests."""
    return anthropic_models["chat_basic"]


@pytest.fixture(scope="session")
def anthropic_chat_vision_model(anthropic_models: dict[str, str]) -> str:
    """Vision-capable model for the Anthropic Messages route."""
    return anthropic_models["chat_vision"]


@pytest.fixture(scope="session")
def anthropic_chat_reasoning_model(anthropic_models: dict[str, str]) -> str:
    """Extended-thinking-capable model for the Anthropic Messages route."""
    return anthropic_models["chat_reasoning"]


@pytest.fixture(scope="session")
def anthropic_system_as_messages_model(anthropic_models: dict[str, str]) -> str:
    """Provide a model forwarding system-role messages natively (Claude Opus 4.8+, Fable, Mythos)."""
    return anthropic_models["chat_system_as_messages"]


@pytest.fixture(scope="session")
def anthropic_count_tokens_model(anthropic_models: dict[str, str]) -> str:
    """Model for the Anthropic count_tokens route (Bedrock CountTokens is Anthropic-only)."""
    return anthropic_models["count_tokens"]


@pytest.fixture(scope="session")
def anthropic_client(
    use_official_api: bool,
    server_url: str | None,
    test_client: TestClient | None,
    api_key: str,
) -> Anthropic | AnthropicBedrock:
    """Anthropic SDK client bound to the selected target.

    With ``--use-official-api`` the client is ``Anthropic`` when an
    ``ANTHROPIC_API_KEY`` is present and ``AnthropicBedrock`` otherwise, which is
    why model IDs differ per lane (see ``ANTHROPIC_MODEL_MAPPINGS``).
    """
    if test_client:
        return Anthropic(
            base_url="http://testserver/anthropic/",
            api_key=api_key,
            max_retries=0,
            # Starlette types TestClient against httpx2; the alias fixes runtime only.
            http_client=test_client,  # type: ignore[arg-type]
        )
    if use_official_api:
        if getenv("ANTHROPIC_API_KEY"):
            # AWS-hosted Anthropic endpoints scope requests to a workspace.
            workspace_id = getenv("ANTHROPIC_WORKSPACE_ID")
            return Anthropic(
                max_retries=5,
                default_headers=(
                    {"anthropic-workspace-id": workspace_id} if workspace_id else None
                ),
            )
        return AnthropicBedrock(max_retries=5)
    return Anthropic(
        base_url=f"{server_url}/anthropic/",
        max_retries=0,
        api_key=getenv("OPENAI_API_KEY"),
    )


@pytest.fixture(scope="session")
def is_bedrock_direct(anthropic_client: Anthropic) -> bool:
    """True when ``anthropic_client`` talks directly to AWS Bedrock (AnthropicBedrock).

    Bedrock returns ``tool_use`` blocks (not ``server_tool_use``) for system tools
    and does not support ``code_execution`` or ``web_fetch`` tool types.
    """
    return isinstance(anthropic_client, AnthropicBedrock)


# ---------------------------------------------------------------------------
# Cohere API fixtures (shared across Cohere test modules)
# ---------------------------------------------------------------------------

#: Model mappings for Cohere-compatible route tests.
COHERE_MODEL_MAPPINGS: dict[str, dict[str, str]] = {
    "local": {
        "embed_multilingual": "cohere.embed-multilingual-v3",
        "embed_v4": "cohere.embed-v4:0",
        "rerank": "cohere.rerank-v3-5:0",
    },
    "cohere": {
        "embed_multilingual": "embed-multilingual-v3.0",
        "embed_v4": "embed-v4.0",
        "rerank": "rerank-v3.5",
    },
}


@pytest.fixture(scope="session")
def cohere_models(use_official_api: bool) -> dict[str, str]:
    """Per-capability Cohere model IDs for the selected target."""
    return COHERE_MODEL_MAPPINGS["cohere" if use_official_api else "local"]


@pytest.fixture(scope="session")
def cohere_embed_multilingual_model(cohere_models: dict[str, str]) -> str:
    """Cohere Embed v3 multilingual model."""
    return cohere_models["embed_multilingual"]


@pytest.fixture(scope="session")
def cohere_embed_v4_model(cohere_models: dict[str, str]) -> str:
    """Cohere Embed v4 model (image inputs and ``output_dimension``)."""
    return cohere_models["embed_v4"]


@pytest.fixture(scope="session")
def cohere_rerank_model(cohere_models: dict[str, str]) -> str:
    """Cohere Rerank model."""
    return cohere_models["rerank"]


def _build_cohere_client[ClientT: cohere.Client](
    client_class: type[ClientT],
    use_official_api: bool,
    server_url: str | None,
    test_client: TestClient | None,
    api_key: str,
) -> ClientT:
    """Build a Cohere SDK client of *client_class* bound to the selected target.

    Args:
        client_class: Cohere SDK client class (``Client`` for v1, ``ClientV2`` for v2).
        use_official_api: Whether the official Cohere API is the selected target.
        server_url: Base URL of the remote gateway, when one is selected.
        test_client: In-process ASGI client, when the gateway runs in-process.
        api_key: API key shared by the test server and its clients.

    Returns:
        Client pointed at the in-process app, the remote gateway or Cohere itself.
    """
    if test_client:
        return client_class(
            api_key=api_key,
            base_url="http://testserver/cohere",
            # Starlette types TestClient against httpx2; the alias fixes runtime only.
            httpx_client=test_client,  # type: ignore[arg-type]
        )
    if use_official_api:
        if not getenv("CO_API_KEY"):
            pytest.skip("CO_API_KEY is required to test the official Cohere API")
        return client_class()
    return client_class(
        api_key=getenv("OPENAI_API_KEY", ""), base_url=f"{server_url}/cohere"
    )


@pytest.fixture(scope="session")
def cohere_client(
    use_official_api: bool,
    server_url: str | None,
    test_client: TestClient | None,
    api_key: str,
) -> cohere.ClientV2:
    """Cohere SDK client bound to the selected target."""
    return _build_cohere_client(
        cohere.ClientV2, use_official_api, server_url, test_client, api_key
    )


@pytest.fixture(scope="session")
def cohere_client_v1(
    use_official_api: bool,
    server_url: str | None,
    test_client: TestClient | None,
    api_key: str,
) -> cohere.Client:
    """Create a Cohere v1 client for either local or official API testing."""
    return _build_cohere_client(
        cohere.Client, use_official_api, server_url, test_client, api_key
    )


#: Failure-output substrings marking a model unavailable on the live backend.
_UNAVAILABLE_MODEL_MARKERS = (
    "marked by provider as Legacy",  # AWS deprecation / 30-day inactivity lock-out
    "model identifier is invalid",  # absent from the cross-region inference catalog
    "is not allowed from unsupported countries",  # geo-restricted provider (Meta Llama)
)
#: ``wasxfail`` reason marking a report masked because the model was unavailable.
_UNAVAILABLE_XFAIL_REASON = "model not available on this backend"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, _PluggyResult[pytest.TestReport]]:
    """Report failures caused by an unavailable model as xfail.

    A model the backend will not serve is an environmental limitation, not a
    defect, so any live test whose failure output carries one of
    ``_UNAVAILABLE_MODEL_MARKERS`` -- including one surfacing only in the server
    logs behind a bare status assertion -- is converted to ``xfail``.
    """
    outcome: _PluggyResult[pytest.TestReport] = yield
    if call.when != "call":
        return
    report = outcome.get_result()
    if not report.failed:
        return
    # Local (in-process) tests never reach a real backend, so a matching marker
    # in their output is a genuine assertion and must not be masked.
    if item.get_closest_marker("local") is not None:
        return
    haystack = f"{report.longreprtext}{report.capstdout}{report.capstderr}"
    if not any(marker in haystack for marker in _UNAVAILABLE_MODEL_MARKERS):
        return
    report.outcome = "xfailed"  # type: ignore[assignment]
    report.wasxfail = _UNAVAILABLE_XFAIL_REASON
    report.longrepr = None


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """List the tests masked as xfail by an unavailable model.

    Without this, a test pinned to a single retired model looks green forever: the
    mask is only visible under ``-rx``. Surfacing it every run keeps a silent
    no-op from passing for coverage. Reads the collected reports rather than a
    module-level list so it also works under ``xdist``, where the mask is applied
    on a worker and only the serialized report reaches this hook.
    """
    masked = [
        report.nodeid
        for report in terminalreporter.stats.get("xfailed", ())
        if getattr(report, "wasxfail", None) == _UNAVAILABLE_XFAIL_REASON
    ]
    if not masked:
        return
    terminalreporter.section("Tests masked by an unavailable model", yellow=True)
    for nodeid in masked:
        terminalreporter.line(f"  {nodeid}")
    terminalreporter.line(
        f"{len(masked)} test(s) did not exercise their model. "
        "Re-pin them to a live model, or drop them if the capability is gone."
    )
