"""Tests for OpenAI Images API endpoints.

This module contains comprehensive tests for the /v1/images/generations endpoint,
validating functionality, error handling, and compliance with OpenAI API specification.
"""

import base64
import re
import time
from collections.abc import Iterable

import pytest
from openai import BadRequestError, OpenAI
from openai.types.image_gen_completed_event import ImageGenCompletedEvent
from openai.types.image_gen_partial_image_event import ImageGenPartialImageEvent


def validate_base64_image(b64_data: str) -> str:
    """Validate base64 encoded image and return detected format.

    Args:
        b64_data: Base64 encoded image data

    Returns:
        Detected image format (png, jpeg, webp)

    Raises:
        AssertionError: If image data is invalid or format unsupported
    """
    try:
        image_bytes = base64.b64decode(b64_data)
    except ValueError as e:
        pytest.fail(f"Invalid base64 format: {e}")

    # Check image format by header bytes
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:12]:
        return "webp"
    pytest.fail("Unsupported image format detected in base64 data")


def validate_url_format(url: str) -> None:
    """Validate that URL follows expected format.

    Args:
        url: URL string to validate

    Raises:
        AssertionError: If URL format is invalid
    """
    url_pattern = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)
    assert url_pattern.match(url), f"Invalid URL format: {url}"


def validate_timestamp(timestamp: int) -> None:
    """Validate timestamp is reasonable (within last hour to future minute).

    Args:
        timestamp: Unix timestamp to validate

    Raises:
        AssertionError: If timestamp is invalid
    """
    current_time = int(time.time())
    # Allow for some time skew - timestamp should be within last hour to next minute
    assert (current_time - 3600) <= timestamp <= (current_time + 60), (
        f"Timestamp {timestamp} is outside reasonable range around {current_time}"
    )


def validate_error_response(
    error: BadRequestError,
    expected_type: str | None = None,
    expected_code: str | None = None,
    expected_param: str | None = None,
) -> None:
    """Validate error response matches OpenAI API specification.

    Args:
        error: BadRequestError instance
        expected_type: Expected error type (e.g., "invalid_request_error")
        expected_code: Expected error code (e.g., "invalid_parameter")
        expected_param: Expected parameter name that caused the error

    Raises:
        AssertionError: If error format doesn't match OpenAI specification
    """
    assert error.status_code == 400, (
        f"Expected status code 400, got {error.status_code}"
    )

    # Check if error has body with proper structure
    if (
        hasattr(error, "body")
        and isinstance(error.body, dict)
        and "error" in error.body
    ):
        error_details = error.body["error"]

        if expected_type:
            assert error_details.get("type") == expected_type, (
                f"Expected error type '{expected_type}', got '{error_details.get('type')}'"
            )

        if expected_code:
            assert error_details.get("code") == expected_code, (
                f"Expected error code '{expected_code}', got '{error_details.get('code')}'"
            )

        if expected_param:
            assert error_details.get("param") == expected_param, (
                f"Expected error param '{expected_param}', got '{error_details.get('param')}'"
            )

        # Ensure message exists
        assert "message" in error_details, "Error response missing 'message' field"
        assert error_details["message"], "Error message should not be empty"


def validate_streaming_image_response(
    response: Iterable[ImageGenCompletedEvent | ImageGenPartialImageEvent],
) -> None:
    """Validate a streaming image generation response according to OpenAI API specification.

    This function validates the complete structure of streaming events from the OpenAI Images API,
    based on the official streaming specification:
    https://platform.openai.com/docs/api-reference/images_streaming/image_generation

    Args:
        response: The streaming response object from OpenAI images.generate()

    Raises:
        AssertionError: If the response doesn't match the expected streaming format
    """
    # Response should be iterable for streaming
    assert hasattr(response, "__iter__"), "Streaming response must be iterable"

    # Collect events from the stream
    events = []
    event_types_seen = set()

    for event in response:
        events.append(event)

        # Each event should have basic attributes
        assert hasattr(event, "type"), f"Event missing 'type' attribute: {event}"
        assert hasattr(event, "created_at"), (
            f"Event missing 'created_at' attribute: {event}"
        )

        event_types_seen.add(event.type)

        # Validate event type-specific attributes
        if event.type == "image_generation.completed":
            # Final completion event should have all metadata
            assert hasattr(event, "output_format"), (
                f"Completion event missing 'output_format': {event}"
            )
            assert hasattr(event, "size"), f"Completion event missing 'size': {event}"
            assert hasattr(event, "usage"), f"Completion event missing 'usage': {event}"

            # Validate format values
            assert event.output_format in ["png", "jpeg", "webp"], (
                f"Invalid output_format: {event.output_format}"
            )
            assert "x" in event.size, f"Invalid size format: {event.size}"

            # Usage should have token information
            assert hasattr(event.usage, "total_tokens"), (
                f"Usage missing total_tokens: {event.usage}"
            )

        elif event.type in ["image_generation.partial", "image_generation.progress"]:
            # Partial/progress events should have basic metadata
            assert hasattr(event, "output_format"), (
                f"Partial event missing 'output_format': {event}"
            )

        # Validate created_at is a reasonable timestamp
        validate_timestamp(event.created_at)

        # Don't process too many events to avoid hanging
        if len(events) > 20:
            break

    # Should have at least one event
    assert len(events) > 0, "Streaming response should contain at least one event"

    # Should have a completion event for successful generation
    assert "image_generation.completed" in event_types_seen, (
        f"Missing completion event. Event types seen: {event_types_seen}"
    )


class TestImageGeneration:
    """Test cases for the /v1/images/generations endpoint.

    These tests validate the OpenAI Images API implementation against the official specification,
    ensuring proper functionality, error handling, and response format compliance.
    """

    @pytest.mark.expensive
    @pytest.mark.parametrize(
        ("prompt", "description"),
        [("A beautiful sunset over mountains", "basic prompt")],
    )
    def test_image_generation_with_various_prompts(
        self,
        openai_client: OpenAI,
        image_generation_model: str,
        prompt: str,
        description: str,
    ) -> None:
        """Test image generation with various prompt types and lengths.

        This comprehensive test covers basic functionality, different prompt types,
        and validates proper response structure according to OpenAI API specification.

        Args:
            openai_client: OpenAI client instance for API calls
            image_generation_model: Image generation model identifier
            prompt: The prompt text to test
            description: Description of the test case type
        """
        response = openai_client.images.generate(
            prompt=prompt, model=image_generation_model, n=1, size="512x512"
        )

        # Validate response structure
        assert response.created is not None, (
            f"Missing 'created' field for {description}"
        )
        validate_timestamp(response.created)

        assert response.data is not None, f"Missing 'data' field for {description}"
        assert len(response.data) == 1, (
            f"Expected 1 image, got {len(response.data)} for {description}"
        )

        # Validate image data
        image = response.data[0]
        assert image.url is not None, f"Missing URL for {description}"
        validate_url_format(image.url)
        assert image.b64_json is None, (
            f"Unexpected b64_json in URL response for {description}"
        )

    @pytest.mark.expensive
    def test_image_generation_with_user_parameter(
        self, openai_client: OpenAI, image_generation_model: str
    ) -> None:
        """Test image generation with user parameter for tracking."""
        response = openai_client.images.generate(
            prompt="A beautiful landscape",
            model=image_generation_model,
            n=1,
            size="512x512",
            user="test-user-123",
        )

        assert response.created is not None
        validate_timestamp(response.created)
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].url is not None
        validate_url_format(response.data[0].url)

    @pytest.mark.expensive
    def test_multiple_images_generation(
        self, openai_client: OpenAI, image_generation_model: str
    ) -> None:
        """Test generation of multiple images in a single request."""
        response = openai_client.images.generate(
            prompt="A cute cat playing",
            model=image_generation_model,
            n=2,
            size="512x512",
        )

        assert response.created is not None
        validate_timestamp(response.created)
        assert response.data is not None
        assert len(response.data) == 2, f"Expected 2 images, got {len(response.data)}"

        for i, image in enumerate(response.data):
            assert image.url is not None, f"Missing URL for image {i}"
            validate_url_format(image.url)
            assert image.b64_json is None, f"Unexpected b64_json for image {i}"

    @pytest.mark.expensive
    def test_response_format_url(
        self, openai_client: OpenAI, image_generation_model: str
    ) -> None:
        """Test image generation with explicit URL response format."""
        response = openai_client.images.generate(
            prompt="A landscape painting",
            model=image_generation_model,
            n=1,
            size="512x512",
            response_format="url",
        )

        assert response.created is not None
        validate_timestamp(response.created)
        assert response.data is not None
        assert len(response.data) == 1

        image = response.data[0]
        assert image.url is not None, "Missing URL in URL format response"
        validate_url_format(image.url)
        assert image.b64_json is None, "Unexpected b64_json in URL format response"

    @pytest.mark.expensive
    def test_response_format_b64_json(
        self, openai_client: OpenAI, image_generation_model: str
    ) -> None:
        """Test image generation with base64 JSON response format."""
        response = openai_client.images.generate(
            prompt="A simple drawing",
            model=image_generation_model,
            n=1,
            size="512x512",
            response_format="b64_json",
        )

        assert response.created is not None
        validate_timestamp(response.created)
        assert response.data is not None
        assert len(response.data) == 1

        image = response.data[0]
        assert image.b64_json is not None, "Missing b64_json in base64 format response"
        assert image.url is None, "Unexpected URL in base64 format response"

        # Validate base64 format and image content
        image_format = validate_base64_image(image.b64_json)
        assert image_format in ["png", "jpeg", "webp"], (
            f"Unsupported image format: {image_format}"
        )

    @pytest.mark.expensive
    @pytest.mark.parametrize("size", ["512x512", "1024x1024"])
    def test_size_parameter_functionality(
        self, openai_client: OpenAI, image_generation_model: str, size: str
    ) -> None:
        """Test various size parameters work correctly.

        Args:
            openai_client: OpenAI client instance for API calls
            image_generation_model: Image generation model identifier
            size: Image size parameter to test
        """
        response = openai_client.images.generate(
            prompt="A geometric shape", model=image_generation_model, n=1, size=size
        )  # type: ignore[call-overload]

        assert response.created is not None
        validate_timestamp(response.created)
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].url is not None
        validate_url_format(response.data[0].url)

    @pytest.mark.expensive
    def test_quality_parameter(
        self,
        openai_client: OpenAI,
        image_generation_hd_model: str,
        use_openai_api: bool,
    ) -> None:
        """Test quality parameter functionality with HD-capable models."""
        response = openai_client.images.generate(
            prompt="A detailed portrait",
            model=image_generation_hd_model,
            n=1,
            size="1024x1024",
            quality="hd" if use_openai_api else "premium",  # type: ignore[call-overload]
        )

        assert response.created is not None
        validate_timestamp(response.created)
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].url is not None
        validate_url_format(response.data[0].url)

    @pytest.mark.expensive
    def test_style_parameter(
        self,
        openai_client: OpenAI,
        image_generation_hd_model: str,
        use_openai_api: bool,
    ) -> None:
        """Test style parameter functionality with compatible models."""
        response = openai_client.images.generate(
            prompt="An abstract artwork",
            model=image_generation_hd_model,
            n=1,
            size="1024x1024",
            style="vivid" if use_openai_api else "PHOTOREALISM",  # type: ignore[call-overload]
        )

        assert response.created is not None
        validate_timestamp(response.created)
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].url is not None
        validate_url_format(response.data[0].url)

    @pytest.mark.expensive
    def test_stream_parameter(
        self, openai_client: OpenAI, image_generation_stream_model: str
    ) -> None:
        """Test streaming functionality with advanced models."""
        response = openai_client.images.generate(
            prompt="A test image",
            model=image_generation_stream_model,
            n=1,
            size="1024x1024",
            stream=True,
        )

        # Validate the streaming response structure
        assert response is not None
        validate_streaming_image_response(response)

    @pytest.mark.expensive
    @pytest.mark.parametrize("partial_images_value", [0, 2, 3])
    def test_stream_with_partial_images(
        self,
        openai_client: OpenAI,
        image_generation_stream_model: str,
        partial_images_value: int,
    ) -> None:
        """Test streaming with various partial_images values.

        This test validates that all valid partial_images values (0, 2, 3) work
        correctly when stream=True is enabled with advanced models.
        """
        response = openai_client.images.generate(
            prompt=f"A test image with partial_images={partial_images_value}",
            model=image_generation_stream_model,
            n=1,
            size="1024x1024",
            stream=True,
            partial_images=partial_images_value,
        )

        assert response is not None
        validate_streaming_image_response(response)

    def test_empty_prompt_error(
        self, openai_client: OpenAI, image_generation_model: str
    ) -> None:
        """Test error handling for empty prompt parameter."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(
                prompt="", model=image_generation_model, n=1, size="512x512"
            )

        validate_error_response(
            exc_info.value,
            expected_type="invalid_request_error",
            expected_param="prompt",
        )

    def test_invalid_model_error(self, openai_client: OpenAI) -> None:
        """Test error handling for invalid model parameter."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(
                prompt="A test image", model="invalid-model-name", n=1, size="512x512"
            )

        validate_error_response(
            exc_info.value,
            expected_type="invalid_request_error",
            expected_param="model",
        )

    @pytest.mark.parametrize("invalid_n", [0, -1, 11])
    def test_invalid_n_parameter_error(
        self, openai_client: OpenAI, image_generation_model: str, invalid_n: int
    ) -> None:
        """Test error handling for invalid n parameter values."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(
                prompt="A test image",
                model=image_generation_model,
                n=invalid_n,
                size="512x512",
            )

        validate_error_response(
            exc_info.value, expected_type="invalid_request_error", expected_param="n"
        )

    def test_invalid_size_error(
        self, openai_client: OpenAI, image_generation_model: str
    ) -> None:
        """Test error handling for invalid size parameter."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(  # type: ignore[call-overload]
                prompt="A test image",
                model=image_generation_model,
                n=1,
                size="invalid-size",
            )

        validate_error_response(
            exc_info.value, expected_type="invalid_request_error", expected_param="size"
        )

    def test_invalid_response_format_error(
        self, openai_client: OpenAI, image_generation_model: str
    ) -> None:
        """Test error handling for invalid response_format parameter."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(  # type: ignore[call-overload]
                prompt="A test image",
                model=image_generation_model,
                n=1,
                size="512x512",
                response_format="invalid-format",
            )

        validate_error_response(
            exc_info.value,
            expected_type="invalid_request_error",
            expected_param="response_format",
        )

    def test_invalid_quality_error(
        self, openai_client: OpenAI, image_generation_model: str
    ) -> None:
        """Test error handling for invalid quality parameter."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(  # type: ignore[call-overload]
                prompt="A test image",
                model=image_generation_model,
                n=1,
                size="512x512",
                quality="invalid-quality",
            )

        validate_error_response(
            exc_info.value,
            expected_type="invalid_request_error",
            expected_param="quality",
        )

    def test_invalid_style_error(
        self, openai_client: OpenAI, image_generation_model: str
    ) -> None:
        """Test error handling for invalid style parameter."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(  # type: ignore[call-overload]
                prompt="A test image",
                model=image_generation_model,
                n=1,
                size="512x512",
                style="invalid-style",
            )

        validate_error_response(
            exc_info.value,
            expected_type="invalid_request_error",
            expected_param="style",
        )

    def test_stream_parameter_error_with_unsupported_models(
        self, openai_client: OpenAI, image_generation_model: str, use_openai_api: bool
    ) -> None:
        """Test stream parameter error with models that don't support streaming.

        The stream parameter should only work with advanced models like gpt-image-1
        and return 'unknown_parameter' error with other models like dall-e-2, dall-e-3.
        """
        if not use_openai_api:
            pytest.skip(
                "Streaming supported on all Bedrock models in this implementation"
            )

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(
                prompt="A test image",
                model=image_generation_model,
                n=1,
                size="512x512",
                stream=True,
            )

        validate_error_response(
            exc_info.value,
            expected_type="invalid_request_error",
            expected_code="unknown_parameter",
            expected_param="stream",
        )

    def test_partial_images_without_stream_error(
        self, openai_client: OpenAI, image_generation_model: str
    ) -> None:
        """Test partial_images parameter error when used without streaming."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(
                prompt="A test image",
                model=image_generation_model,
                n=1,
                size="512x512",
                partial_images=1,
            )

        validate_error_response(
            exc_info.value,
            expected_type="invalid_request_error",
            expected_code="unknown_parameter",
            expected_param="partial_images",
        )
