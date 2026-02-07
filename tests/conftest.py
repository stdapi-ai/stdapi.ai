"""Pytest configuration and fixtures."""

import base64
from io import BytesIO
from os import environ
from pathlib import Path
from secrets import token_hex
from typing import TYPE_CHECKING

import pytest
from aiobotocore.session import get_session
from openai import OpenAI
from PIL import Image as PILImage
from pybase64 import b64encode
from starlette.testclient import TestClient

if TYPE_CHECKING:
    from collections.abc import Generator

# Model mappings for different test contexts
MODEL_MAPPINGS = {
    "local": {
        "transcription": "amazon.transcribe",
        "transcription_stream": "amazon.transcribe",
        "speech_standard": "amazon.polly-standard",
        "chat": "amazon.nova-micro-v1:0",
        "chat_vision": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "chat_legacy": "anthropic.claude-haiku-4-5-20251001-v1:0",
        "chat_reasoning": "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "chat_audio": "mistral.voxtral-mini-3b-2507",
        "embedding": "amazon.titan-embed-text-v2:0",
        "responses": "amazon.nova-micro-v1:0",
        "image_generation": "amazon.titan-image-generator-v2:0",
        "image_generation_hd": "amazon.nova-canvas-v1:0",
        "image_generation_stream": "amazon.titan-image-generator-v2:0",
    },
    "openai": {
        "transcription": "whisper-1",
        "transcription_stream": "gpt-4o-mini-transcribe",
        "speech_standard": "tts-1",
        "chat": "gpt-5-nano",
        "chat_vision": "gpt-5-nano",
        "chat_legacy": "gpt-4o-mini",
        "chat_reasoning": "gpt-5-nano",
        "chat_audio": "gpt-audio",
        "embedding": "text-embedding-3-small",
        "responses": "gpt-5-nano",
        "image_generation": "dall-e-2",  # Cheapest/default model
        "image_generation_hd": "dall-e-3",  # For HD & style quality features
        "image_generation_stream": "gpt-image-1",  # For streaming features
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
        "--server-url",
        action="store",
        default=None,
        help="URL of the server to test against instead of using test client",
    )
    parser.addoption(
        "--use-openai-api",
        action="store_true",
        default=False,
        help="Run tests against the official OpenAI API instead of local implementation",
    )
    parser.addoption(
        "--expensive",
        action="store_true",
        default=False,
        help="Run compute/cost/time expensive tests",
    )
    parser.addoption(
        "--info",
        action="store_true",
        default=False,
        help="Run the test server with 'info' log level. "
        "Only if --server-url and --use-openai-api are not specified.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip expensive tests at collection time unless explicitly requested."""
    if not config.getoption("--expensive"):
        skip_marker = pytest.mark.skip(
            reason="Need --expensive option to run this test"
        )
        for item in items:
            if item.get_closest_marker("expensive"):
                item.add_marker(skip_marker)


@pytest.fixture(scope="session")
def use_openai_api(request: pytest.FixtureRequest) -> bool:
    """Determine if we should use the official OpenAI API."""
    return request.config.getoption("--use-openai-api")  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def models(use_openai_api: bool) -> dict[str, str]:
    """Provide appropriate models based on test context."""
    return MODEL_MAPPINGS["openai" if use_openai_api else "local"].copy()


@pytest.fixture(scope="session")
def transcription_model(models: dict[str, str]) -> str:
    """Provide the appropriate transcription model."""
    return models["transcription"]


@pytest.fixture(scope="session")
def transcription_stream_model(models: dict[str, str]) -> str:
    """Provide the appropriate transcription model."""
    return models["transcription_stream"]


@pytest.fixture(scope="session")
def speech_standard_model(models: dict[str, str]) -> str:
    """Provide the appropriate standard speech model."""
    return models["speech_standard"]


@pytest.fixture(scope="session")
def chat_model(models: dict[str, str]) -> str:
    """Provide the appropriate chat model."""
    return models["chat"]


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
def openai_client(request: pytest.FixtureRequest) -> Generator[OpenAI]:
    """Create an OpenAI client for either local or official API testing."""
    server_url = request.config.getoption("--server-url")
    if request.config.getoption("--use-openai-api"):
        # Use official OpenAI API
        yield OpenAI(max_retries=5)

    elif server_url:
        # Use specified server URL (for local testing against external server)
        yield OpenAI(
            base_url=f"{server_url.rstrip('/')}/v1",
            max_retries=5,
            organization=_OPENAI_ORGANIZATION,
        )

    else:
        api_key = token_hex()
        environ.update(
            {
                # Use FastAPI TestClient for local testing
                "log_level": "info" if request.config.getoption("--info") else "error",
                # Ensure invalid inputs in tests are detected
                "strict_input_validation": "true",
                # Ensure all optional middlewares and features are enabled
                "api_key": api_key,
                "cors_allow_origins": '["*"]',
                "enable_gzip": "true",
                "enable_proxy_headers": "true",
                "log_client_ip": "true",
                "log_request_params": "true",
                "model_cache_seconds": "10",
                "otel_enabled": "true",
                "tokens_estimation": "true",
                "trusted_hosts": '["*"]',
                "aws_bedrock_allow_application_inference_profile_arn": "true",
                "aws_bedrock_allow_prompt_router_arn": "true",
            }
        )
        from stdapi.main import app  # noqa: PLC0415

        with TestClient(app) as test_client:
            yield OpenAI(
                base_url="http://testserver/v1",
                api_key=api_key,
                max_retries=5,
                organization=_OPENAI_ORGANIZATION,
                http_client=test_client,
            )


@pytest.fixture(scope="session")
def sample_audio_file(openai_client: OpenAI, speech_standard_model: str) -> bytes:
    """Create a sample audio file for testing using the speech endpoint.

    This fixture generates a short WAV audio snippet using the TTS endpoint once,
    caches it under tests/.cache/audio.wav, and returns its bytes for reuse by
    tests (both local server and --use-openai-api modes).
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
    tests (both local server and --use-openai-api modes).
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
