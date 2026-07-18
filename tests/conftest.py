"""Pytest configuration and fixtures."""

from __future__ import annotations

import base64
import sys
from io import BytesIO
from json import JSONDecodeError, dumps, loads
from os import environ, getenv
from pathlib import Path
from secrets import token_hex
from typing import TYPE_CHECKING

import cohere
import pytest
from aiobotocore.session import get_session
from anthropic import Anthropic, AnthropicBedrock, BadRequestError, NotFoundError
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image as PILImage
from pybase64 import b64encode
from starlette.testclient import TestClient

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any

    from pluggy import Result as _PluggyResult

    from stdapi.pricing import CacheTtlBucket, ContextLength, Dimension, Routing


def logged_usage_entries(
    captured_stdout: str,
    *,
    service: str | None = None,
    operation: str | None = None,
    model: str | None = None,
) -> list[dict[str, Any]]:
    """Extract usage entries from captured JSON logs, optionally filtered.

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
) -> None:
    """Seed the price index with one test price.

    Shared by pricing/usage/monitoring tests that need a real, resolvable
    price rather than mocking ``resolve_price`` itself. *model* is used
    as-is, not normalized -- pick a name with no "-"/"_" separators so
    normalize_model_key() doesn't alter it. Defaults seed the standard tier,
    undifferentiated bucket; pass ``tier``/``cache_ttl``/``routing``/``spec``/
    ``context`` to seed a non-default ``PriceKey`` bucket (e.g. a routed,
    cache-TTL, media-spec, or long-context price) without building
    ``PriceKey``/``Price`` by hand.

    Args:
        model: Model ID, used as-is (see above).
        region: AWS region.
        dimension: The billed dimension.
        amount: The unit price, as a decimal string.
        currency: The currency code.
        tier: Service tier (standard, flex, priority, batch).
        cache_ttl: Cache TTL bucket ("5m"/"1h"), for CACHE_WRITE_TOKENS.
        routing: Serving profile ("global", "latency" or "").
        spec: Media/image spec bucket -- see ``PriceKey.spec``.
        context: Context-length bucket ("long" or "").
    """
    from decimal import Decimal  # noqa: PLC0415

    from stdapi.pricing import Price, PriceKey, Service, _state  # noqa: PLC0415

    key = PriceKey(
        Service.BEDROCK,
        model,
        region,
        dimension,
        tier,
        cache_ttl,
        routing,
        spec,
        context,
    )
    _state.price_index[key] = Price(Decimal(amount), currency)


@pytest.fixture(autouse=True)
def _clean_price_index() -> Generator[None]:
    """Reset the price index before each test to prevent leakage."""
    from stdapi.pricing import _state  # noqa: PLC0415

    original = _state.price_index
    _state.price_index = {}
    yield
    _state.price_index = original


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
# These vars must be set at module level — before pytest imports any test file
# that may import stdapi modules at the top level (e.g. for unit tests).
# stdapi.config.SETTINGS is a module-level singleton: the first import of
# stdapi.config runs ``SETTINGS = _Settings()`` from whatever os.environ
# values are present at that moment.  If we only set these inside the
# test_client fixture, any test file that imports from stdapi at collection
# time would see the defaults (api_key=None, allow_prompt_router=False).
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
# Serve the cheap dual-homed Gemma 3 test model via Mantle instead of bedrock-runtime.
environ.setdefault("aws_bedrock_mantle_preferred_models", "google.gemma-3-4b-it")

# Model mappings for different test contexts
MODEL_MAPPINGS = {
    "local": {
        "transcription": "amazon.transcribe",
        "transcription_stream": "amazon.transcribe",
        "transcription_diarize": "amazon.transcribe",
        "speech_standard": "amazon.polly-standard",
        "chat": "amazon.nova-micro-v1:0",
        "completion": "amazon.nova-micro-v1:0",
        "chat_vision": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "chat_legacy": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "chat_reasoning": "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "chat_audio": "mistral.voxtral-mini-3b-2507",
        "embedding": "amazon.titan-embed-text-v2:0",
        "responses": "amazon.nova-micro-v1:0",
        "responses_json_output": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "responses_web_search": "amazon.nova-2-lite-v1:0",
        "responses_code_interpreter": "amazon.nova-2-lite-v1:0",
        "input_tokens": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        "image_generation": "amazon.nova-canvas-v1:0",
        "image_generation_hd": "amazon.nova-canvas-v1:0",
        "image_generation_stream": "amazon.nova-canvas-v1:0",
        # Luma is the only non-legacy video model (Nova Reel is LEGACY on AWS).
        "video_generation": "luma.ray-v2:0",
    },
    "openai": {
        "transcription": "whisper-1",
        "transcription_stream": "gpt-4o-mini-transcribe",
        "transcription_diarize": "gpt-4o-transcribe-diarize",
        "speech_standard": "tts-1",
        "chat": "gpt-5-nano",
        "completion": "gpt-3.5-turbo-instruct",
        "chat_vision": "gpt-5-nano",
        "chat_legacy": "gpt-4o-mini",
        "chat_reasoning": "gpt-5-nano",
        "chat_audio": "gpt-audio",
        "embedding": "text-embedding-3-small",
        "responses": "gpt-5-nano",
        "responses_json_output": "gpt-5-nano",
        "responses_web_search": "gpt-5-nano",
        "responses_code_interpreter": "gpt-5-nano",
        "input_tokens": "gpt-4o-mini",
        "image_generation": "dall-e-2",  # Cheapest/default model
        "image_generation_hd": "dall-e-3",  # For HD & style quality features
        "image_generation_stream": "gpt-image-1",  # For streaming features
        "video_generation": "sora-2",  # Cheapest video model
    },
}
_CACHE_DIR = Path(__file__).parent / ".cache"
SAMPLES_DIR = Path(__file__).parent / "samples"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
_OPENAI_ORGANIZATION = "tests_stdapi.ai"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom pytest command line options."""
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
        help="Run compute/cost/time expensive tests",
    )
    parser.addoption(
        "--agentic", action="store_true", default=False, help="Run agentic tests"
    )


def pytest_report_header() -> str | None:
    """Show which env file was loaded in the pytest session header."""
    if _loaded_env_file:
        return f"envfile: {_loaded_env_file}"
    return None


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip expensive/agentic tests, and local tests against a remote target, at collection time.

    ``local``-marked tests are skipped when ``--server-url`` or
    ``--use-official-api`` selects a remote target; ``expensive``/``agentic``
    tests are skipped unless their matching flag is passed.
    """
    if config.getoption("--server-url") or config.getoption("--use-official-api"):
        skip_marker = pytest.mark.skip(
            reason="Tests the local implementation (remote target selected)"
        )
        for item in items:
            if item.get_closest_marker("local"):
                item.add_marker(skip_marker)
    if not config.getoption("--expensive"):
        skip_marker = pytest.mark.skip(
            reason="Need --expensive option to run this test"
        )
        for item in items:
            if item.get_closest_marker("expensive"):
                item.add_marker(skip_marker)
    if not config.getoption("--agentic"):
        skip_marker = pytest.mark.skip(reason="Need --agentic option to run this test")
        for item in items:
            if item.get_closest_marker("agentic"):
                item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def use_official_api(request: pytest.FixtureRequest) -> bool:
    """Determine if we should use an official API (OpenAI, Anthropic, etc.)."""
    return request.config.getoption("--use-official-api")  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def models(use_official_api: bool) -> dict[str, str]:
    """Provide appropriate models based on test context."""
    return MODEL_MAPPINGS["openai" if use_official_api else "local"].copy()


@pytest.fixture(scope="session")
def transcription_model(models: dict[str, str]) -> str:
    """Provide the appropriate transcription model."""
    return models["transcription"]


@pytest.fixture(scope="session")
def transcription_stream_model(models: dict[str, str]) -> str:
    """Provide the appropriate transcription model."""
    return models["transcription_stream"]


@pytest.fixture(scope="session")
def transcription_diarize_model(models: dict[str, str]) -> str:
    """Provide the appropriate transcription model with diarization support."""
    return models["transcription_diarize"]


@pytest.fixture(scope="session")
def speech_standard_model(models: dict[str, str]) -> str:
    """Provide the appropriate standard speech model."""
    return models["speech_standard"]


@pytest.fixture(scope="session")
def chat_model(models: dict[str, str]) -> str:
    """Provide the appropriate chat model."""
    return models["chat"]


@pytest.fixture(scope="session")
def completion_model(models: dict[str, str]) -> str:
    """Provide the appropriate chat model."""
    return models["completion"]


@pytest.fixture(scope="session")
def chat_vision_model(models: dict[str, str]) -> str:
    """Provide a chat model that supports IMAGE input."""
    return models["chat_vision"]


@pytest.fixture(scope="session")
def chat_reasoning_model(models: dict[str, str]) -> str:
    """Provide a chat model that supports reasoning."""
    return models["chat_reasoning"]


@pytest.fixture(scope="session")
def chat_legacy_model(models: dict[str, str]) -> str:
    """Provide a chat model that supports legacy function input."""
    return models["chat_legacy"]


@pytest.fixture(scope="session")
def chat_audio_model(models: dict[str, str]) -> str:
    """Provide a chat model that supports audio output."""
    return models["chat_audio"]


@pytest.fixture(scope="session")
def embedding_model(models: dict[str, str]) -> str:
    """Provide the appropriate embeddings model."""
    return models["embedding"]


@pytest.fixture(scope="session")
def responses_model(models: dict[str, str]) -> str:
    """Provide the appropriate model for the Responses API."""
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
    """Provide the appropriate model for the Responses API with web search support."""
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
    """Provide the appropriate model for Responses API input token counting."""
    return models["input_tokens"]


@pytest.fixture(scope="session")
def image_generation_model(models: dict[str, str]) -> str:
    """Provide the appropriate default image generation model (cheapest)."""
    return models["image_generation"]


@pytest.fixture(scope="session")
def image_generation_hd_model(models: dict[str, str]) -> str:
    """Provide the appropriate HD image generation model."""
    return models["image_generation_hd"]


@pytest.fixture(scope="session")
def image_generation_stream_model(models: dict[str, str]) -> str:
    """Provide the appropriate advanced image generation model."""
    return models["image_generation_stream"]


@pytest.fixture(scope="session")
def video_generation_model(models: dict[str, str]) -> str:
    """Provide the appropriate default video generation model."""
    return models["video_generation"]


@pytest.fixture(scope="session")
def api_key() -> str:
    """Returns the API key used for the test session with local clients."""
    return _TEST_API_KEY


@pytest.fixture(scope="session")
def test_client(request: pytest.FixtureRequest) -> Generator[TestClient | None]:
    """Create a Starlette test client for local API testing."""
    if not request.config.getoption(
        "--use-official-api"
    ) and not request.config.getoption("--server-url"):
        from stdapi.main import app  # noqa: PLC0415

        with TestClient(app) as test_client:
            yield test_client
    else:
        yield None


@pytest.fixture(scope="session")
def openai_client(
    request: pytest.FixtureRequest, test_client: TestClient | None, api_key: str
) -> OpenAI:
    """Create an OpenAI client for either local or official API testing."""
    # Local test
    if test_client:
        return OpenAI(
            base_url="http://testserver/v1",
            api_key=api_key,
            max_retries=0,
            organization=_OPENAI_ORGANIZATION,
            http_client=test_client,
        )

    # Official API test
    if request.config.getoption("--use-official-api"):
        return OpenAI(max_retries=5)

    # Remote server test
    return OpenAI(
        base_url=f"{request.config.getoption('--server-url').rstrip('/')}/v1",
        max_retries=0,
        organization=_OPENAI_ORGANIZATION,
    )


@pytest.fixture(scope="session")
def sample_audio_file(openai_client: OpenAI, speech_standard_model: str) -> bytes:
    """Create a sample audio file for testing using the speech endpoint.

    This fixture generates a short WAV audio snippet using the TTS endpoint once,
    caches it under tests/.cache/audio.wav, and returns its bytes for reuse by
    tests (both local server and --use-official-api modes).
    """
    audio_file = _CACHE_DIR / "audio.wav"
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
    """Generates a WAV data URL containing a base64-encoded audio.

    Returns:
        str: A string representing the data URL of a WAV audio in base64 encoding.
    """
    return f"data:audio/wav;base64,{b64encode(sample_audio_file).decode('utf-8')}"


@pytest.fixture(scope="session")
def sample_audio_mp3_file(openai_client: OpenAI, speech_standard_model: str) -> bytes:
    """Create a sample audio file for testing using the speech endpoint.

    This fixture generates a short MP3 audio snippet using the TTS endpoint once,
    caches it under tests/.cache/audio.mp3, and returns its bytes for reuse by
    tests (both local server and --use-official-api modes).
    """
    audio_file = _CACHE_DIR / "audio.mp3"
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
    """Generates a MP3 data URL containing a base64-encoded audio.

    Returns:
        str: A string representing the data URL of a MP3 audio in base64 encoding.
    """
    return f"data:audio/mp3;base64,{b64encode(sample_audio_mp3_file).decode('utf-8')}"


@pytest.fixture(scope="session")
def sample_image_file(openai_client: OpenAI, image_generation_model: str) -> bytes:
    """Create a sample PNG image for testing using the Images API.

    The fixture prefers the b64_json response format to avoid external downloads.
    It generates a small 512x512 image once, caches it under tests/.cache/image.png,
    and returns its bytes for reuse across tests and sessions.
    """
    image_file = _CACHE_DIR / "image.png"
    if image_file.exists():
        with image_file.open("rb") as file:
            return file.read()

    response = openai_client.images.generate(
        prompt="A rainbow llama",
        model=image_generation_model,
        n=1,
        size="512x512",
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
    """Generates a PNG data URL containing a base64-encoded image.

    Returns:
        str: A string representing the data URL of a PNG image in base64 encoding.
    """
    return f"data:image/png;base64,{b64encode(sample_image_file).decode('utf-8')}"


@pytest.fixture(scope="session")
def sample_mask_file(sample_image_file: bytes) -> bytes:
    """Create a test mask image with white area in center (masked zone).

    Args:
        sample_image_file: Sample image file to get dimensions from

    Returns:
        Mask image bytes
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
def sample_video_file() -> bytes:
    """Return a local mp4 video file as sample for testing."""
    video_file = _CACHE_DIR / "video.mp4"
    if not video_file.exists():
        return b""
    with video_file.open("rb") as file:
        return file.read()


@pytest.fixture(scope="session")
def sample_video_file_base64(sample_video_file: bytes) -> str:
    """Generates a mp4 data URL containing a base64-encoded video.

    Returns:
        str: A string representing the data URL of a mp4 video in base64 encoding.
    """
    if sample_video_file:
        return f"data:video/mp4;base64,{b64encode(sample_video_file).decode('utf-8')}"
    return ""


@pytest.fixture(scope="session")
def sample_pdf_file() -> bytes:
    """Create a minimal PDF file for testing.

    Returns:
        bytes: A minimal valid PDF file containing "Hello World" text.
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
    """Generates a PDF data URI containing a base64-encoded PDF.

    Returns:
        str: A string representing the data URI of a PDF file in base64 encoding.
    """
    return f"data:application/pdf;base64,{b64encode(sample_pdf_file).decode('utf-8')}"


@pytest.fixture(scope="session")
async def aws_session_info() -> tuple[str, str]:
    """Get AWS region and account ID from STS client in a single call.

    Returns:
        tuple[str, str]: A tuple containing (region, account_id).
    """
    session = get_session()
    async with session.create_client("sts") as sts:
        region = session.get_config_variable("region")
        return region, (await sts.get_caller_identity())["Account"]


@pytest.fixture(scope="session")
async def aws_region(aws_session_info: tuple[str, str]) -> str:
    """Get AWS region from session info.

    Returns:
        str: AWS region name.
    """
    return aws_session_info[0]


@pytest.fixture(scope="session")
async def aws_account_id(aws_session_info: tuple[str, str]) -> str:
    """Get AWS account ID from session info.

    Returns:
        str: AWS account ID.
    """
    return aws_session_info[1]


# ---------------------------------------------------------------------------
# Anthropic API fixtures (shared across Anthropic test modules)
# ---------------------------------------------------------------------------

#: Model mappings for Anthropic /v1/messages tests.
ANTHROPIC_MODEL_MAPPINGS: dict[str, dict[str, str]] = {
    "local": {
        "chat": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "chat_vision": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "chat_reasoning": "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "chat_system_as_messages": "anthropic.claude-sonnet-5",
        "count_tokens": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    },
    "anthropic": {
        "chat": "claude-haiku-4-5-20251001",
        "chat_vision": "claude-haiku-4-5-20251001",
        "chat_reasoning": "claude-sonnet-4-5-20250929",
        "chat_system_as_messages": "claude-sonnet-5",
        "count_tokens": "anthropic.claude-3-5-sonnet-20240620-v1:0",
    },
}
ANTHROPIC_MODEL_MAPPINGS["bedrock"] = {
    key: f"global.{value}" for key, value in ANTHROPIC_MODEL_MAPPINGS["local"].items()
}
ANTHROPIC_MODEL_MAPPINGS["bedrock"]["count_tokens"] = (
    "anthropic.claude-3-5-sonnet-20240620-v1:0"
)


@pytest.fixture(scope="session")
def use_anthropic_api(request: pytest.FixtureRequest) -> bool:
    """Determine if we should use the official Anthropic API."""
    return request.config.getoption("--use-official-api")  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def anthropic_models(use_anthropic_api: bool) -> dict[str, str]:
    """Get Anthropic model mappings based on test target."""
    return (
        (
            ANTHROPIC_MODEL_MAPPINGS["anthropic"]
            if getenv("ANTHROPIC_API_KEY")
            else ANTHROPIC_MODEL_MAPPINGS["bedrock"]
        )
        if use_anthropic_api
        else ANTHROPIC_MODEL_MAPPINGS["local"]
    )


@pytest.fixture(scope="session")
def anthropic_chat_model(anthropic_models: dict[str, str]) -> str:
    """Provide the appropriate Anthropic chat model."""
    return anthropic_models["chat"]


@pytest.fixture(scope="session")
def anthropic_chat_vision_model(anthropic_models: dict[str, str]) -> str:
    """Provide the appropriate vision-capable Anthropic chat model."""
    return anthropic_models["chat_vision"]


@pytest.fixture(scope="session")
def anthropic_chat_reasoning_model(anthropic_models: dict[str, str]) -> str:
    """Provide the appropriate reasoning-capable Anthropic chat model."""
    return anthropic_models["chat_reasoning"]


@pytest.fixture(scope="session")
def anthropic_system_as_messages_model(anthropic_models: dict[str, str]) -> str:
    """Provide a model where system-role messages are forwarded as messages (Claude Opus 4.8+)."""
    return anthropic_models["chat_system_as_messages"]


@pytest.fixture(scope="session")
def anthropic_count_tokens_model(anthropic_models: dict[str, str]) -> str:
    """Provide the appropriate Anthropic model for count tokens testing."""
    return anthropic_models["count_tokens"]


@pytest.fixture(scope="session")
def anthropic_client(
    request: pytest.FixtureRequest, test_client: TestClient | None, api_key: str
) -> Anthropic | AnthropicBedrock:
    """Create an Anthropic client for either local or official API testing."""
    if test_client:
        return Anthropic(
            base_url="http://testserver/anthropic/",
            api_key=api_key,
            max_retries=0,
            http_client=test_client,
        )
    if request.config.getoption("--use-official-api"):
        if getenv("ANTHROPIC_API_KEY"):
            return Anthropic(max_retries=5)
        return AnthropicBedrock(max_retries=5)
    return Anthropic(
        base_url=f"{request.config.getoption('--server-url').rstrip('/')}/anthropic/",
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
    """Get Cohere model mappings based on test target."""
    return COHERE_MODEL_MAPPINGS["cohere" if use_official_api else "local"]


@pytest.fixture(scope="session")
def cohere_embed_multilingual_model(cohere_models: dict[str, str]) -> str:
    """Provide the appropriate multilingual Cohere embed model."""
    return cohere_models["embed_multilingual"]


@pytest.fixture(scope="session")
def cohere_embed_v4_model(cohere_models: dict[str, str]) -> str:
    """Provide the appropriate Cohere embed v4 model (images, output_dimension)."""
    return cohere_models["embed_v4"]


@pytest.fixture(scope="session")
def cohere_rerank_model(cohere_models: dict[str, str]) -> str:
    """Provide the appropriate Cohere rerank model."""
    return cohere_models["rerank"]


@pytest.fixture(scope="session")
def cohere_client(
    request: pytest.FixtureRequest, test_client: TestClient | None, api_key: str
) -> cohere.ClientV2:
    """Create a Cohere client for either local or official API testing."""
    if test_client:
        return cohere.ClientV2(
            api_key=api_key,
            base_url="http://testserver/cohere",
            httpx_client=test_client,
        )
    if request.config.getoption("--use-official-api"):
        if not getenv("CO_API_KEY"):
            pytest.skip("CO_API_KEY is required to test the official Cohere API")
        return cohere.ClientV2()
    return cohere.ClientV2(
        api_key=getenv("OPENAI_API_KEY", ""),
        base_url=f"{request.config.getoption('--server-url').rstrip('/')}/cohere",
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, _PluggyResult[pytest.TestReport]]:
    """Convert 'invalid model identifier' errors to xfail for cross-model tests.

    When tests are parametrized over all Claude models, some Bedrock model IDs may
    not exist in the cross-region inference catalog.  Rather than failing hard, those
    tests are reported as ``xfail`` so the run stays informative without blocking CI.
    """
    outcome: _PluggyResult[pytest.TestReport] = yield
    if call.when != "call":
        return
    report = outcome.get_result()
    if not report.failed or call.excinfo is None:
        return
    exc = call.excinfo.value
    if (
        isinstance(exc, BadRequestError) and "model identifier is invalid" in str(exc)
    ) or (isinstance(exc, NotFoundError) and "Legacy" in str(exc)):
        pass
    else:
        return
    if "test_anthropic_messages_anthropic_claude" not in getattr(
        getattr(item, "module", None), "__name__", ""
    ):
        return
    report.outcome = "xfailed"  # type: ignore[assignment]
    report.wasxfail = f"model not available on this backend: {call.excinfo.value}"
    report.longrepr = None
