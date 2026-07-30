"""OpenAI Images Generations API (``/v1/images/generations``) compatibility tests.

Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     https://stdapi.ai/api_openai_images_generations/
     stdapi/routes/openai_images_generations.py:create_images
"""

import base64
import json
import re
import time
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast

import pytest
from openai import BadRequestError, OpenAI
from pydantic import ValidationError

from stdapi.models.image import ImageGenerationJobBase, ImageGenerationResponse
from stdapi.monitoring import REQUEST_LOG, REQUEST_TIME, EventLog
from stdapi.routes._images_common import build_images_response
from stdapi.routes.openai_images_generations import stream_generator
from stdapi.types.openai_images import ImageGenerateParams
from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Iterable

    from openai.types import ImageEditCompletedEvent, ImageEditPartialImageEvent
    from openai.types.image_gen_completed_event import ImageGenCompletedEvent
    from openai.types.image_gen_partial_image_event import ImageGenPartialImageEvent
    from starlette.testclient import TestClient as TestClientType

    type ImageStreamEvent = (
        ImageGenCompletedEvent
        | ImageGenPartialImageEvent
        | ImageEditPartialImageEvent
        | ImageEditCompletedEvent
    )


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
    expected_type: str = "invalid_request_error",
    expected_code: str | None = None,
    expected_param: str | None = None,
    expected_status: int = 400,
) -> None:
    """Validate an error against the OpenAI error envelope.

    ``APIStatusError.body`` is already the *inner* ``error`` object: the OpenAI
    client stores ``body.get("error", body)``. It is therefore indexed directly --
    an ``error.body["error"]`` lookup would always miss.

    The gateway leaves ``param`` null for Pydantic validation failures and names
    the offending field in the message instead, so *expected_param* is satisfied
    by either the ``param`` field or the message.

    Args:
        error: The raised API error.
        expected_type: Expected error type (e.g. "invalid_request_error").
        expected_code: Expected error code, when the envelope carries one.
        expected_param: Name of the parameter the error must identify.
        expected_status: Expected HTTP status code.

    Raises:
        AssertionError: If the error does not match the OpenAI envelope.
    """
    assert error.status_code == expected_status, (
        f"Expected status code {expected_status}, got {error.status_code}"
    )

    body = error.body
    assert isinstance(body, dict), f"Expected an error object body, got {body!r}"
    assert body.get("type") == expected_type, (
        f"Expected error type '{expected_type}', got '{body.get('type')}'"
    )

    message = body.get("message")
    assert isinstance(message, str), (
        f"Error 'message' must be a string, got {message!r}"
    )
    assert message, "Error message should not be empty"

    if expected_code:
        assert body.get("code") == expected_code, (
            f"Expected error code '{expected_code}', got '{body.get('code')}'"
        )

    if expected_param:
        identified = body.get("param") == expected_param or re.search(
            rf"\b{re.escape(expected_param)}\b", message
        )
        assert identified, (
            f"Error does not identify parameter '{expected_param}': "
            f"param={body.get('param')!r}, message={message!r}"
        )


def validate_streaming_image_response(
    response: Iterable[ImageStreamEvent],
) -> list[ImageStreamEvent]:
    """Validate a streaming image generation response and return its events.

    Every event carries a type and a creation timestamp; a completed event
    carries the final image, its format/size and a self-consistent usage report.

    Args:
        response: The streaming response returned by ``images.generate``.

    Returns:
        The events collected from the stream.

    Raises:
        AssertionError: If the stream does not match the OpenAI streaming shape.
    """
    # Response should be iterable for streaming
    assert hasattr(response, "__iter__"), "Streaming response must be iterable"

    # Collect events from the stream
    events: list[ImageStreamEvent] = []
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
            assert re.fullmatch(r"\d+x\d+", event.size), (
                f"Size must be WIDTHxHEIGHT: {event.size}"
            )

            # The completed event carries the final image in the announced format
            assert validate_base64_image(event.b64_json) == event.output_format, (
                "Completed event payload does not match the announced output_format"
            )

            # Usage should have token information
            assert event.usage, f"Usage missing total_tokens: {event.usage}"
            assert (
                event.usage.total_tokens
                == event.usage.input_tokens + event.usage.output_tokens
            ), f"Inconsistent usage totals: {event.usage}"
            assert (
                event.usage.input_tokens_details.image_tokens
                + event.usage.input_tokens_details.text_tokens
                == event.usage.input_tokens
            ), f"Input token details do not sum to input_tokens: {event.usage}"

        elif event.type == "image_generation.partial_image":
            # Partial events carry a 0-based index and the announced format
            assert hasattr(event, "output_format"), (
                f"Partial event missing 'output_format': {event}"
            )
            assert event.output_format in ["png", "jpeg", "webp"], (
                f"Invalid output_format: {event.output_format}"
            )
            assert event.partial_image_index >= 0, (
                f"Partial index must be 0-based: {event.partial_image_index}"
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
    return events


class TestImageGeneration:
    """Request/response contract of ``POST /v1/images/generations``.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/types/openai_images.py:ImageGenerateParams
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
        """A text prompt returns a single image referenced by URL.

        ``response_format`` is left unset, and its default is ``url``, so the
        image is uploaded and referenced instead of being inlined as base64.

        Ref: stdapi/routes/_images_common.py:build_images_response
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
        """``user`` is accepted for attribution and leaves the response unchanged.

        ``user`` is recorded on the request log for monitoring and abuse
        detection; it is never echoed back, so an unchanged image response is the
        whole observable contract.

        Ref: stdapi/types/openai_images.py:_ImageBaseParams
             stdapi/monitoring.py:log_request_params
        """
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
        assert response.data[0].b64_json is None

    @pytest.mark.expensive
    def test_multiple_images_generation(
        self, openai_client: OpenAI, image_generation_model: str
    ) -> None:
        """``n=2`` returns two separately addressable images.

        Each image gets its own indexed object, so the two entries never share a
        URL even when the backend needs one invocation per image.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._get_image_url
        """
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

        assert response.data[0].url != response.data[1].url, (
            "each generated image must be returned under its own URL"
        )

    @pytest.mark.expensive
    def test_response_format_url(
        self, openai_client: OpenAI, image_generation_model: str
    ) -> None:
        """``response_format="url"`` returns a URL and no inline base64 payload.

        The gateway supports ``url`` for every image model, unlike OpenAI's GPT
        image models which always return base64.

        Ref: https://stdapi.ai/api_openai_images_generations/
             stdapi/models/image/__init__.py:ImageGenerationJobBase._get_image_url
        """
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
        """``response_format="b64_json"`` inlines decodable image bytes and no URL.

        Ref: stdapi/routes/_images_common.py:build_images_response
        """
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
        if response.output_format is not None:
            assert response.output_format == image_format, (
                "Reported output_format must match the returned bytes"
            )

    @pytest.mark.expensive
    @pytest.mark.parametrize("size", ["512x512", "1024x1024"])
    def test_size_parameter_functionality(
        self, openai_client: OpenAI, image_generation_model: str, size: str
    ) -> None:
        """Both square sizes are accepted and produce an image.

        The response's ``size`` is the size the model actually produced, not the
        requested one: backends such as Stability translate ``WIDTHxHEIGHT`` into
        an aspect ratio and pick their own resolution, so it is deliberately not
        asserted to equal the request.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase.width
             stdapi/models/image/_stability.py:_get_aspect_ratio
        """
        response = openai_client.images.generate(
            prompt="A geometric shape", model=image_generation_model, n=1, size=size
        )

        assert response.created is not None
        validate_timestamp(response.created)
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].url is not None
        validate_url_format(response.data[0].url)
        if response.size is not None:
            assert re.fullmatch(r"\d+x\d+", response.size), (
                f"Reported size must be WIDTHxHEIGHT, got {response.size}"
            )

    @pytest.mark.expensive
    def test_quality_parameter(
        self,
        openai_client: OpenAI,
        image_generation_hd_model: str,
        use_official_api: bool,
    ) -> None:
        """A top-tier ``quality`` request is echoed back as ``high``.

        The gateway normalizes OpenAI's quality vocabulary onto the backend's own
        levels and then reports the level actually used, so the Bedrock
        ``premium`` tier surfaces as ``high``.

        Ref: stdapi/routes/openai_images_generations.py:_OPENAI_QUALITY_LEVELS
             stdapi/models/image/amazon_titan_image_generator.py:AMZ_QUALITY_MAP
        """
        response = openai_client.images.generate(
            prompt="A detailed portrait",
            model=image_generation_hd_model,
            n=1,
            size="1024x1024",
            quality="hd" if use_official_api else "premium",  # type: ignore[call-overload]
        )

        assert response.created is not None
        validate_timestamp(response.created)
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].url is not None
        validate_url_format(response.data[0].url)
        if not use_official_api:
            assert response.quality == "high", (
                "the Bedrock premium tier must be reported as OpenAI quality 'high'"
            )

    @pytest.mark.expensive
    def test_style_parameter(
        self,
        openai_client: OpenAI,
        image_generation_hd_model: str,
        use_official_api: bool,
    ) -> None:
        """A model-specific ``style`` is forwarded and accepted by the backend.

        ``style`` values are model-dependent and are not validated by the
        gateway, which upper-cases them into the provider payload. The response
        carries no style echo, so a generated image is the proof the value was
        accepted -- an unknown style comes back as a 400 instead.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
             stdapi/types/openai_images.py:ImageGenerateParams
        """
        response = openai_client.images.generate(
            prompt="An abstract artwork",
            model=image_generation_hd_model,
            n=1,
            size="1024x1024",
            style="vivid" if use_official_api else "PHOTOREALISM",  # type: ignore[call-overload]
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
        """``stream=True`` yields SSE events ending in one completed event per image.

        Ref: https://developers.openai.com/api/docs/guides/image-generation
             stdapi/routes/openai_images_generations.py:stream_generator
        """
        response = openai_client.images.generate(
            prompt="A test image",
            model=image_generation_stream_model,
            n=1,
            size="1024x1024",
            stream=True,
        )

        # Validate the streaming response structure
        assert response is not None
        events = validate_streaming_image_response(response)
        completed = [e for e in events if e.type == "image_generation.completed"]
        assert len(completed) == 1, (
            f"Expected one completed event for n=1, got {len(completed)}"
        )

    @pytest.mark.expensive
    @pytest.mark.parametrize("partial_images_value", [0, 2, 3])
    def test_stream_with_partial_images(
        self,
        openai_client: OpenAI,
        image_generation_stream_model: str,
        partial_images_value: int,
    ) -> None:
        """Every legal ``partial_images`` value still ends the stream with one completed event.

        ``partial_images`` is accepted for compatibility for all of 0-3; no
        Bedrock image model reports intermediate images, so the value cannot
        change the number of completed events.

        Ref: stdapi/types/openai_images.py:ImageGenerateParams
             stdapi/routes/openai_images_generations.py:stream_generator
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
        events = validate_streaming_image_response(response)
        completed = [e for e in events if e.type == "image_generation.completed"]
        assert len(completed) == 1, (
            f"Expected one completed event for n=1, got {len(completed)}"
        )

    def test_empty_prompt_error(
        self, openai_client: OpenAI, image_generation_model: str
    ) -> None:
        """An empty ``prompt`` is rejected as an ``invalid_request_error``.

        ``prompt`` has ``min_length=1``, so the request never reaches Bedrock.

        Ref: stdapi/types/openai_images.py:ImageGenerateParams
             stdapi/main.py:handle_validation_exception
        """
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
        """An unknown ``model`` is rejected with ``model_not_found`` as a 400.

        The images route passes ``error_status=400`` to model validation, so an
        unknown model is reported as a bad request rather than the 404 the
        underlying error class defaults to.

        Ref: stdapi/routes/openai_images_generations.py:create_images
             stdapi/api_errors.py:UnsupportedModelError
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(
                prompt="A test image", model="invalid-model-name", n=1, size="512x512"
            )

        validate_error_response(
            exc_info.value,
            expected_type="invalid_request_error",
            expected_code="model_not_found",
            expected_param="model",
        )

    @pytest.mark.parametrize("invalid_n", [0, -1, 11])
    def test_invalid_n_parameter_error(
        self, openai_client: OpenAI, image_generation_model: str, invalid_n: int
    ) -> None:
        """``n`` outside 1-10 is rejected as an ``invalid_request_error``.

        Ref: stdapi/types/openai_images.py:_ImageBaseParams
             stdapi/main.py:handle_validation_exception
        """
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
        """A ``size`` that is neither ``auto`` nor ``WIDTHxHEIGHT`` is rejected.

        Ref: stdapi/types/openai_images.py:_ImageBaseParams
             stdapi/main.py:handle_validation_exception
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(
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
        """``response_format`` outside ``url``/``b64_json`` is rejected.

        Ref: stdapi/types/openai_images.py:_ImageBaseParams
             stdapi/main.py:handle_validation_exception
        """
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
        """A ``quality`` the model cannot honor is rejected as a 400 naming ``quality``.

        ``quality`` is a free-form string on the gateway because the accepted
        values are model-dependent, so the rejection comes from the backend job
        rather than from schema validation.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._validate_no_quality
             stdapi/api_providers/openai.py:_format_error
        """
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
        """A ``style`` the model cannot honor is rejected as a 400 naming ``style``.

        Like ``quality``, ``style`` is model-dependent and free-form on the
        gateway, so the rejection comes from the backend job.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._validate_no_style
             stdapi/api_providers/openai.py:_format_error
        """
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
        self, openai_client: OpenAI, image_generation_model: str, use_official_api: bool
    ) -> None:
        """Upstream rejects ``stream`` on the DALL-E models as ``unknown_parameter``.

        Only OpenAI's GPT image models stream. The gateway streams every Bedrock
        image model instead, so this is exercised against the official API only.

        Ref: https://developers.openai.com/api/docs/guides/image-generation
             stdapi/routes/openai_images_generations.py:stream_generator
        """
        if not use_official_api:
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
        """``partial_images`` without ``stream=True`` is rejected as a 400.

        The incompatibility is caught by a model-level validator, so the gateway
        envelope names ``partial_images`` in the message and carries no ``param``.

        Ref: stdapi/types/openai_images.py:ImageGenerateParams._unsupported
             stdapi/main.py:handle_validation_exception
        """
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
            expected_param="partial_images",
        )

    def test_model_not_supporting_text_to_image(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """An edit-only model is rejected on the text-to-image route.

        Upscale models expose no text-to-image job, so the shared base
        implementation raises instead of building a Bedrock request.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._generate_images_from_text
        """
        if use_official_api:
            pytest.skip(
                "Bedrock model catalog (variations-only models) is gateway-specific; "
                "no equivalent model exists on the official API"
            )
        # Use a model that only supports variations (not text-to-image)
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(
                prompt="A test image",
                model="stability.stable-fast-upscale-v1:0",  # Upscale model doesn't support text-to-image
                n=1,
                size="512x512",
            )

        validate_error_response(exc_info.value, expected_type="invalid_request_error")
        error_msg = str(exc_info.value).lower()
        assert (
            "not supported" in error_msg
            or "text-to-image" in error_msg
            or "invalid value" in error_msg
        )


class TestImageGenerationUsage:
    """Usage logging emitted by ``POST /v1/images/generations``.

    Ref: stdapi/models/image/__init__.py:ImageModelBase._record_invoke_usage
    """

    @pytest.mark.expensive
    def test_image_generation_usage_logged(
        self,
        test_client: TestClientType | None,
        image_generation_model: str,
        api_key: str,
        capfd: pytest.CaptureFixture[str],
    ) -> None:
        """Image generation logs one ``bedrock-runtime`` entry with ``output_images``.

        Image models are billed per generated image rather than per token, so
        ``output_images`` -- not the token counters -- is the billable quantity
        recorded for the request.

        Ref: stdapi/usage.py:IMAGE_SPEC
             stdapi/routes/_images_common.py:build_images_response
        """
        if test_client is None:
            pytest.skip("Requires local test server")
        capfd.readouterr()
        response = test_client.post(
            "/v1/images/generations",
            json={
                "model": image_generation_model,
                "prompt": "A red circle on a white background",
                "n": 1,
                "size": "512x512",
            },
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if response.status_code != 200 and (
            "have access to the model" in (detail := response.text.lower())
            or "does not exist" in detail
        ):
            pytest.skip(
                f"Image model {image_generation_model} is not accessible in this environment"
            )
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["data"]) == 1
        usage = payload["usage"]
        assert usage["total_tokens"] == usage["input_tokens"] + usage["output_tokens"]
        entries = logged_usage_entries(
            capfd.readouterr().out,
            service="bedrock-runtime",
            operation="/v1/images/generations",
            model=image_generation_model,
        )
        assert entries, "Expected a bedrock image usage log entry"
        assert entries[0]["output_images"] == 1


class TestStreamGeneratorUsageMatchesNonStream:
    """stream_generator: the final SSE usage matches the non-streaming path's.

    Ref: stdapi/routes/openai_images_generations.py:stream_generator
         stdapi/routes/_images_common.py:build_images_response
    """

    @pytest.mark.local
    async def test_final_stream_event_usage_matches_non_streaming(self) -> None:
        """The last completed SSE event reports the job's real billed tokens, like non-streaming.

        A backend that fans out one invocation per image (Stability) only knows
        its final token counts once the last image is done, so each completed
        event must be built from the counts billed so far rather than from a
        snapshot taken when the stream opened.
        """
        REQUEST_TIME.set(datetime.now(UTC))
        log_token = REQUEST_LOG.set(cast("EventLog", {"level": "info"}))
        try:
            job = object.__new__(ImageGenerationJobBase)
            job._count = 2  # noqa: SLF001
            job._response_width = 512  # noqa: SLF001
            job._response_height = 512  # noqa: SLF001
            job._response_quality = "medium"  # noqa: SLF001
            job._output_format = "png"  # noqa: SLF001
            job._response_output_format = "png"  # noqa: SLF001
            job._input_tokens = None  # noqa: SLF001
            job._output_tokens = None  # noqa: SLF001

            async def image_stream() -> AsyncGenerator[ImageGenerationResponse]:
                """Simulate progressive per-image token accumulation (e.g. Stability fan-out)."""
                job._input_tokens = 5  # noqa: SLF001
                job._output_tokens = 1  # noqa: SLF001
                yield ImageGenerationResponse(image="aaa", index=0)
                job._input_tokens = 12  # noqa: SLF001
                job._output_tokens = 2  # noqa: SLF001
                yield ImageGenerationResponse(image="bbb", index=1)

            events = []
            async for event in stream_generator(image_stream(), job, created=0):
                assert event.data is not None
                events.append(json.loads(event.data))
            completed = [e for e in events if e["type"] == "image_generation.completed"]

            assert len(completed) == 2
            assert [e["b64_json"] for e in completed] == ["aaa", "bbb"]
            # Job metadata is echoed on every completed event.
            assert completed[0]["created_at"] == 0
            assert completed[0]["size"] == "512x512"
            assert completed[0]["output_format"] == "png"
            assert completed[0]["quality"] == "medium"
            assert completed[0]["background"] == "opaque"
            # The first completed event reflects tokens billed so far, not a stale zero.
            assert completed[0]["usage"]["input_tokens"] == 5
            assert completed[0]["usage"]["output_tokens"] == 1
            # The final event reports the job's fully accumulated total.
            assert completed[-1]["usage"]["input_tokens"] == 12
            assert completed[-1]["usage"]["output_tokens"] == 2
            assert completed[-1]["usage"]["total_tokens"] == 14
            # Text-to-image has no input images, so every input token is text.
            assert completed[-1]["usage"]["input_tokens_details"] == {
                "image_tokens": 0,
                "text_tokens": 12,
            }

            non_stream = await build_images_response(
                job=job,
                results=[ImageGenerationResponse(image="bbb", index=1)],
                response_format="b64_json",
                output_image_count=2,
            )
            assert non_stream.usage is not None
            assert (
                completed[-1]["usage"]["input_tokens"] == non_stream.usage.input_tokens
            )
            assert (
                completed[-1]["usage"]["output_tokens"]
                == non_stream.usage.output_tokens
            )
        finally:
            REQUEST_LOG.reset(log_token)


class TestSizeAutoAccepted:
    """ImageGenerateParams.size: the OpenAI literal `auto` is accepted, not rejected.

    Ref: stdapi/types/openai_images.py:_ImageBaseParams._resolve_auto_size
    """

    pytestmark = pytest.mark.local

    def test_auto_size_resolves_to_default(self) -> None:
        """`size="auto"` is resolved to the gateway's default 1024x1024.

        OpenAI lets the model pick a size; the gateway must send a concrete
        ``WIDTHxHEIGHT`` to Bedrock, so it substitutes its default instead.
        """
        params = ImageGenerateParams(model="m", prompt="p", size="auto")
        assert params.size == "1024x1024"

    def test_explicit_size_is_unchanged(self) -> None:
        """An explicit `WIDTHxHEIGHT` size passes through unmodified."""
        params = ImageGenerateParams(model="m", prompt="p", size="512x512")
        assert params.size == "512x512"

    def test_invalid_size_still_rejected(self) -> None:
        """A non-`auto`, non-`WIDTHxHEIGHT` size fails the ``size`` pattern."""
        with pytest.raises(ValidationError, match="size") as exc_info:
            ImageGenerateParams(model="m", prompt="p", size="invalid-size")

        errors = exc_info.value.errors()
        assert any(
            error["loc"] == ("size",) and error["type"] == "string_pattern_mismatch"
            for error in errors
        ), errors


class TestPartialImagesAccepted:
    """ImageGenerateParams.partial_images: accepted (0-3) even though no model emits partials.

    Ref: stdapi/types/openai_images.py:ImageGenerateParams._unsupported
    """

    pytestmark = pytest.mark.local

    @pytest.mark.parametrize("value", [0, 1, 2, 3])
    def test_value_is_accepted(self, value: int) -> None:
        """`partial_images` in 0-3 is accepted; no current model emits partial events."""
        params = ImageGenerateParams(
            model="m", prompt="p", stream=True, partial_images=value
        )
        assert params.partial_images == value
        assert params.stream is True

    def test_omitted_is_accepted(self) -> None:
        """Omitting `partial_images` (None) is accepted."""
        params = ImageGenerateParams(model="m", prompt="p", stream=True)
        assert params.partial_images is None

    def test_requires_stream(self) -> None:
        """`partial_images` without `stream=True` is rejected by the model validator."""
        with pytest.raises(ValidationError, match="partial_images") as exc_info:
            ImageGenerateParams(model="m", prompt="p", partial_images=0)

        errors = exc_info.value.errors()
        assert any(
            error["type"] == "value_error"
            and "partial_images requires streaming mode" in error["msg"]
            for error in errors
        ), errors
