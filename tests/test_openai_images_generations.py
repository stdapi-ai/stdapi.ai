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
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest
from openai import BadRequestError, OpenAI
from pydantic import ValidationError

from stdapi.api_errors import FeatureUnavailableError, UnsupportedModelError
from stdapi.config import SETTINGS
from stdapi.models.image import ImageGenerationJobBase, ImageGenerationResponse
from stdapi.monitoring import REQUEST_LOG, REQUEST_TIME, EventLog
from stdapi.routes import openai_images_generations
from stdapi.routes._images_common import build_images_response
from stdapi.routes.openai_images_generations import stream_generator
from stdapi.types.openai_images import ImageEditParams, ImageGenerateParams
from tests.conftest import (
    image_returns_base64_only,
    image_size_supported,
    logged_usage_entries,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable, Iterable

    from openai.types import ImageEditCompletedEvent, ImageEditPartialImageEvent
    from openai.types.image_gen_completed_event import ImageGenCompletedEvent
    from openai.types.image_gen_partial_image_event import ImageGenPartialImageEvent
    from openai.types.images_response import Usage
    from starlette.testclient import TestClient as TestClientType

    type ImageStreamEvent = (
        ImageGenCompletedEvent
        | ImageGenPartialImageEvent
        | ImageEditPartialImageEvent
        | ImageEditCompletedEvent
    )

    type ImageParams = ImageGenerateParams | ImageEditParams


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


#: OpenAI image models that have been retired upstream. A request naming one is
#: refused before any other parameter is looked at, so nothing else is observable.
_RETIRED_OFFICIAL_IMAGE_MODELS = frozenset({"dall-e-2", "dall-e-3"})

#: OpenAI image models that reject ``stream``; every other one streams.
_NON_STREAMING_OFFICIAL_IMAGE_MODELS = frozenset({"dall-e-2", "dall-e-3"})


def _skip_retired_official_image_model(
    use_official_api: bool, model: str, parameter: str
) -> None:
    """Skip the official lane when the mapped OpenAI image model has been retired.

    OpenAI has removed the DALL-E models: ``GET /v1/models/dall-e-2`` answers
    ``model_not_found``, and every ``/v1/images/*`` request naming one is
    refused with ``The model 'dall-e-2' does not exist`` before any other
    parameter is looked at. The upstream rejection of *parameter* is therefore
    unobservable until ``MODEL_MAPPINGS["openai"]`` is repointed at a live image
    model; the gateway lane keeps asserting it.

    The guard tests the model, not just the lane, so repointing the mapping at a
    live model restores this coverage automatically instead of leaving a skip that
    claims a live model was retired.

    Args:
        use_official_api: True when the suite targets the official OpenAI API.
        model: Model id the test sends.
        parameter: Request parameter whose rejection the test asserts.

    Ref: https://developers.openai.com/api/docs/guides/image-generation
         tests/conftest.py:MODEL_MAPPINGS
    """
    if use_official_api and model in _RETIRED_OFFICIAL_IMAGE_MODELS:
        pytest.skip(
            f"OpenAI retired '{model}': the request is rejected as "
            f"\"The model '{model}' does not exist\" before '{parameter}' is "
            f"validated"
        )


def validate_streaming_image_response(
    response: Iterable[ImageStreamEvent], prefix: str = "image_generation"
) -> list[ImageStreamEvent]:
    """Validate a streaming image generation response and return its events.

    Every event carries a type and a creation timestamp; a completed event
    carries the final image, its format/size and a self-consistent usage report.

    Args:
        response: The streaming response returned by ``images.generate``.
        prefix: Event-name family the endpoint emits — ``image_generation`` for
            generations, ``image_edit`` for edits.

    Returns:
        The events collected from the stream.

    Raises:
        AssertionError: If the stream does not match the OpenAI streaming shape.
    """
    assert hasattr(response, "__iter__"), "Streaming response must be iterable"

    events: list[ImageStreamEvent] = []
    event_types_seen = set()

    for event in response:
        events.append(event)

        assert hasattr(event, "type"), f"Event missing 'type' attribute: {event}"
        assert hasattr(event, "created_at"), (
            f"Event missing 'created_at' attribute: {event}"
        )

        event_types_seen.add(event.type)

        if event.type == f"{prefix}.completed":
            assert hasattr(event, "output_format"), (
                f"Completion event missing 'output_format': {event}"
            )
            assert hasattr(event, "size"), f"Completion event missing 'size': {event}"
            assert hasattr(event, "usage"), f"Completion event missing 'usage': {event}"

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

        elif event.type == f"{prefix}.partial_image":
            # Partial events carry a 0-based index and the announced format
            assert hasattr(event, "output_format"), (
                f"Partial event missing 'output_format': {event}"
            )
            assert event.output_format in ["png", "jpeg", "webp"], (
                f"Invalid output_format: {event.output_format}"
            )
            partial = cast(
                "ImageGenPartialImageEvent | ImageEditPartialImageEvent", event
            )
            assert partial.partial_image_index >= 0, (
                f"Partial index must be 0-based: {partial.partial_image_index}"
            )

        validate_timestamp(event.created_at)

        # Don't process too many events to avoid hanging
        if len(events) > 20:
            break

    assert len(events) > 0, "Streaming response should contain at least one event"

    assert f"{prefix}.completed" in event_types_seen, (
        f"Missing completion event. Event types seen: {event_types_seen}"
    )
    return events


def validate_image_usage(usage: Usage | None) -> None:
    """Assert an image usage report's counters are internally consistent.

    Requires ``input_tokens`` and ``total_tokens`` positive, ``total_tokens``
    equal to ``input_tokens + output_tokens``, and the input token detail
    breakdown (image + text) summing back to ``input_tokens``.

    Args:
        usage: The response's ``usage`` field.

    Raises:
        AssertionError: If any counter is missing or the totals don't add up.
    """
    assert usage is not None
    assert usage.input_tokens > 0
    assert usage.input_tokens_details.text_tokens >= 0
    assert usage.input_tokens_details.image_tokens > 0
    assert usage.total_tokens > 0
    assert usage.total_tokens == usage.input_tokens + usage.output_tokens
    assert (
        usage.input_tokens_details.image_tokens + usage.input_tokens_details.text_tokens
        == usage.input_tokens
    )


class TestImageGeneration:
    """Request/response contract of ``POST /v1/images/generations``.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/types/openai_images.py:ImageGenerateParams
    """

    @pytest.mark.expensive
    def test_image_generation_default_response_format_is_url(
        self,
        openai_client: OpenAI,
        image_generation_model: str,
        image_generation_size: str,
    ) -> None:
        """A text prompt returns a single image referenced by URL.

        ``response_format`` is left unset, and its default is ``url``, so the
        image is uploaded and referenced instead of being inlined as base64;
        passing ``url`` explicitly takes the same branch. ``user`` rides along
        because it is recorded on the request log and never echoed back, so an
        unchanged response is the whole observable contract -- and a second
        generation to prove it would buy nothing.

        Ref: stdapi/routes/_images_common.py:build_images_response
             stdapi/types/openai_images.py:_ImageBaseParams
        """
        response = openai_client.images.generate(
            prompt="A beautiful sunset over mountains",
            model=image_generation_model,
            n=1,
            size=image_generation_size,
            user="test-user-123",
        )

        assert response.created is not None
        validate_timestamp(response.created)
        assert response.data is not None
        assert len(response.data) == 1

        image = response.data[0]
        assert image.url is not None
        validate_url_format(image.url)
        assert image.b64_json is None, "the url default must not inline base64 data"

    @pytest.mark.expensive
    def test_multiple_images_generation(
        self,
        openai_client: OpenAI,
        image_generation_model: str,
        image_generation_size: str,
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
            size=image_generation_size,
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
    def test_response_format_b64_json(
        self,
        openai_client: OpenAI,
        image_generation_model: str,
        image_generation_size: str,
    ) -> None:
        """``response_format="b64_json"`` inlines decodable image bytes and no URL.

        Ref: stdapi/routes/_images_common.py:build_images_response
        """
        response = openai_client.images.generate(
            prompt="A simple drawing",
            model=image_generation_model,
            n=1,
            size=image_generation_size,
            response_format="b64_json",
        )

        assert response.created is not None
        validate_timestamp(response.created)
        assert response.data is not None
        assert len(response.data) == 1

        image = response.data[0]
        assert image.b64_json is not None, "Missing b64_json in base64 format response"
        assert image.url is None, "Unexpected URL in base64 format response"

        image_format = validate_base64_image(image.b64_json)
        assert image_format in ["png", "jpeg", "webp"], (
            f"Unsupported image format: {image_format}"
        )
        if response.output_format is not None:
            assert response.output_format == image_format, (
                "Reported output_format must match the returned bytes"
            )

    @pytest.mark.expensive
    # One size only: every other live test here already runs at the fixture's
    # default, so a second value would re-prove the same acceptance branch.
    @pytest.mark.parametrize("size", ["512x512"])
    def test_size_parameter_functionality(
        self, openai_client: OpenAI, image_generation_model: str, size: str
    ) -> None:
        """A requested square size is accepted and produces an image.

        The response's ``size`` is the size the model actually produced, not the
        requested one: backends such as Stability translate ``WIDTHxHEIGHT`` into
        an aspect ratio and pick their own resolution, so it is deliberately not
        asserted to equal the request. A model that enumerates its sizes rather
        than mapping arbitrary ones -- ``gpt-image-1`` takes only 1024x1024,
        1024x1536 and 1536x1024 -- skips the sizes it cannot accept.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase.width
             stdapi/models/image/_stability.py:_get_aspect_ratio
             tests/conftest.py:IMAGE_MODEL_ACCEPTED_SIZES
        """
        if not image_size_supported(image_generation_model, size):
            pytest.skip(f"{image_generation_model} does not accept size {size}")
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
        image_generation_hd_size: str,
        use_official_api: bool,
    ) -> None:
        """A top-tier ``quality`` request is echoed back as ``high``.

        The gateway normalizes OpenAI's quality vocabulary onto the backend's own
        levels and then reports the level actually used, so the Bedrock
        ``premium`` tier surfaces as ``high``.

        Only Amazon's image models map ``quality`` and they are all legacy (every
        Stability model rejects it), so pinning a legacy model here is
        deliberate (#93).

        Ref: stdapi/routes/openai_images_generations.py:_OPENAI_QUALITY_LEVELS
             stdapi/models/image/amazon_titan_image_generator.py:AMZ_QUALITY_MAP
        """
        response = openai_client.images.generate(
            prompt="A detailed portrait",
            model=image_generation_hd_model,
            n=1,
            size=image_generation_hd_size,
            # "hd" was dall-e-3's vocabulary; gpt-image-1 takes auto/low/medium/high.
            quality="high" if use_official_api else "premium",  # type: ignore[call-overload]
        )

        assert response.created is not None
        validate_timestamp(response.created)
        assert response.data is not None
        assert len(response.data) == 1
        if image_returns_base64_only(image_generation_hd_model):
            # gpt-image-1 has no response_format and never returns a URL.
            assert response.data[0].b64_json is not None
            validate_base64_image(response.data[0].b64_json)
        else:
            assert response.data[0].url is not None
            validate_url_format(response.data[0].url)
        if not use_official_api:
            assert response.quality == "high", (
                "the Bedrock premium tier must be reported as OpenAI quality 'high'"
            )

    @pytest.mark.expensive
    @pytest.mark.gateway
    def test_style_parameter(
        self,
        openai_client: OpenAI,
        image_generation_hd_model: str,
        image_generation_hd_size: str,
    ) -> None:
        """A model-specific ``style`` is forwarded and accepted by the backend.

        ``style`` values are model-dependent and are not validated by the
        gateway, which upper-cases them into the provider payload. The response
        carries no style echo, so a generated image is the proof the value was
        accepted -- an unknown style comes back as a 400 instead.

        Gateway-only: ``dall-e-3`` was the sole OpenAI model with a ``style``
        parameter and it has been retired, so ``gpt-image-1`` answers 400
        ``unknown_parameter`` for ``style`` whatever the value.

        Only Amazon's image models accept ``style`` and they are all legacy
        (every Stability model rejects it), so pinning a legacy model here is
        deliberate (#93).

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
             stdapi/types/openai_images.py:ImageGenerateParams
        """
        response = openai_client.images.generate(
            prompt="An abstract artwork",
            model=image_generation_hd_model,
            n=1,
            size=image_generation_hd_size,
            style="PHOTOREALISM",  # type: ignore[call-overload]
        )

        assert response.created is not None
        validate_timestamp(response.created)
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].url is not None
        validate_url_format(response.data[0].url)

    @pytest.mark.expensive
    def test_stream_parameter(
        self,
        openai_client: OpenAI,
        image_generation_stream_model: str,
        image_generation_stream_size: str,
    ) -> None:
        """``stream=True`` yields SSE events ending in one completed event per image.

        ``partial_images`` is accepted for compatibility over its whole 0-3
        range, and no Bedrock image model reports intermediate images, so the
        value cannot change what the stream carries: one legal value rides on
        this request rather than paying for a generation per value.

        Ref: https://developers.openai.com/api/docs/guides/image-generation
             stdapi/types/openai_images.py:ImageGenerateParams
             stdapi/routes/openai_images_generations.py:stream_generator
        """
        response = openai_client.images.generate(
            prompt="A test image",
            model=image_generation_stream_model,
            n=1,
            size=image_generation_stream_size,
            stream=True,
            partial_images=2,
        )

        assert response is not None
        events = validate_streaming_image_response(response)
        completed = [e for e in events if e.type == "image_generation.completed"]
        assert len(completed) == 1, (
            f"Expected one completed event for n=1, got {len(completed)}"
        )

    def test_empty_prompt_error(
        self,
        openai_client: OpenAI,
        image_generation_model: str,
        image_generation_size: str,
        use_official_api: bool,
    ) -> None:
        """An empty ``prompt`` is rejected as an ``invalid_request_error``.

        ``prompt`` has ``min_length=1``, so the request never reaches Bedrock.
        OpenAI reports the same envelope (``invalid_request_error`` on
        ``prompt``, code ``empty_string``) on a live image model.

        Ref: stdapi/types/openai_images.py:ImageGenerateParams
             stdapi/main.py:handle_validation_exception
        """
        _skip_retired_official_image_model(
            use_official_api, image_generation_model, "prompt"
        )

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(
                prompt="", model=image_generation_model, n=1, size=image_generation_size
            )

        validate_error_response(
            exc_info.value,
            expected_type="invalid_request_error",
            expected_param="prompt",
        )

    def test_invalid_model_error(
        self, openai_client: OpenAI, image_generation_size: str, use_official_api: bool
    ) -> None:
        """An unknown ``model`` is rejected as a 400 naming ``model``.

        The images route passes ``error_status=400`` to model validation, so an
        unknown model is reported as a bad request rather than the 404 the
        underlying error class defaults to.

        The two targets label that 400 differently: OpenAI answers
        ``image_generation_user_error``/``invalid_value`` on its image
        endpoints, whereas the gateway answers the generic
        ``invalid_request_error``/``model_not_found`` it uses everywhere else.

        Ref: stdapi/routes/openai_images_generations.py:create_images
             stdapi/api_errors.py:UnsupportedModelError
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(
                prompt="A test image",
                model="invalid-model-name",
                n=1,
                size=image_generation_size,
            )

        validate_error_response(
            exc_info.value,
            expected_type=(
                "image_generation_user_error"
                if use_official_api
                else "invalid_request_error"
            ),
            expected_code="invalid_value" if use_official_api else "model_not_found",
            expected_param="model",
        )

    @pytest.mark.parametrize("invalid_n", [0, -1, 11])
    def test_invalid_n_parameter_error(
        self,
        openai_client: OpenAI,
        image_generation_model: str,
        image_generation_size: str,
        invalid_n: int,
        use_official_api: bool,
    ) -> None:
        """``n`` outside 1-10 is rejected as an ``invalid_request_error``.

        OpenAI enforces the same 1-10 bound with the same envelope
        (``invalid_request_error`` on ``n``) on a live image model.

        Ref: stdapi/types/openai_images.py:_ImageBaseParams
             stdapi/main.py:handle_validation_exception
        """
        _skip_retired_official_image_model(
            use_official_api, image_generation_model, "n"
        )

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(
                prompt="A test image",
                model=image_generation_model,
                n=invalid_n,
                size=image_generation_size,
            )

        validate_error_response(
            exc_info.value, expected_type="invalid_request_error", expected_param="n"
        )

    def test_invalid_size_error(
        self, openai_client: OpenAI, image_generation_model: str, use_official_api: bool
    ) -> None:
        """A ``size`` that is neither ``auto`` nor ``WIDTHxHEIGHT`` is rejected.

        The two targets label that 400 differently, so the expected envelope
        follows the selected lane: the gateway rejects it in schema validation
        as an ``invalid_request_error``, while OpenAI rejects it after model
        resolution as an ``image_generation_user_error``/``invalid_value``.
        Both must name ``size``.

        Ref: stdapi/types/openai_images.py:_ImageBaseParams
             stdapi/main.py:handle_validation_exception
        """
        _skip_retired_official_image_model(
            use_official_api, image_generation_model, "size"
        )

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(
                prompt="A test image",
                model=image_generation_model,
                n=1,
                size="invalid-size",
            )

        validate_error_response(
            exc_info.value,
            expected_type=(
                "image_generation_user_error"
                if use_official_api
                else "invalid_request_error"
            ),
            expected_code="invalid_value" if use_official_api else None,
            expected_param="size",
        )

    def test_invalid_response_format_error(
        self,
        openai_client: OpenAI,
        image_generation_model: str,
        image_generation_size: str,
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
                size=image_generation_size,
                response_format="invalid-format",
            )

        validate_error_response(
            exc_info.value,
            expected_type="invalid_request_error",
            expected_param="response_format",
        )

    @pytest.mark.expensive
    def test_invalid_quality_error(
        self,
        openai_client: OpenAI,
        image_generation_model: str,
        image_generation_size: str,
        use_official_api: bool,
    ) -> None:
        """A ``quality`` a model has no control for is ignored, not refused.

        Quality steers the backend; it never decides whether an image can be
        produced. Refusing over it would break every client that sends OpenAI's
        default on a model that has no such control, so the gateway drops it
        with a warning and reports the quality actually produced.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._drop_unsupported_quality
        """
        _skip_retired_official_image_model(
            use_official_api, image_generation_model, "quality"
        )

        response = openai_client.images.generate(  # type: ignore[call-overload]
            prompt="A test image",
            model=image_generation_model,
            n=1,
            size=image_generation_size,
            quality="invalid-quality",
        )

        assert response.data
        assert len(response.data) == 1
        assert response.data[0].b64_json or response.data[0].url, (
            "the image is produced even though the requested quality was dropped"
        )
        assert getattr(response, "quality", None) in ("low", "medium", "high"), (
            "the response reports the quality actually produced, never the "
            "refused, invalid value"
        )

    @pytest.mark.expensive
    def test_invalid_style_error(
        self,
        openai_client: OpenAI,
        image_generation_model: str,
        image_generation_size: str,
    ) -> None:
        """A ``style`` a model has no control for is ignored, not refused.

        Same rationale as ``quality``: a style the backend cannot honour changes
        the look of the result, never whether there is one.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._drop_unsupported_style
        """
        response = openai_client.images.generate(  # type: ignore[call-overload]
            prompt="A test image",
            model=image_generation_model,
            n=1,
            size=image_generation_size,
            style="invalid-style",
        )

        assert response.data
        assert len(response.data) == 1
        assert response.data[0].b64_json or response.data[0].url, (
            "the image is produced even though the requested style was dropped"
        )

    def test_stream_parameter_error_with_unsupported_models(
        self,
        openai_client: OpenAI,
        image_generation_model: str,
        image_generation_size: str,
        use_official_api: bool,
    ) -> None:
        """Upstream rejects ``stream`` on the DALL-E models as ``unknown_parameter``.

        Only OpenAI's GPT image models stream. The gateway streams every Bedrock
        image model instead, so this is exercised against the official API only.

        The claim is currently unobservable on both targets: OpenAI has retired
        the DALL-E family, and every remaining official image model streams. The
        skip is model-conditional either way, so mapping a non-streaming image
        model back in restores the coverage instead of silently billing a
        successful generation.

        Ref: https://developers.openai.com/api/docs/guides/image-generation
             stdapi/routes/openai_images_generations.py:stream_generator
        """
        if not use_official_api:
            pytest.skip(
                "Streaming supported on all Bedrock models in this implementation"
            )
        if image_generation_model not in _NON_STREAMING_OFFICIAL_IMAGE_MODELS:
            pytest.skip(
                f"'{image_generation_model}' supports streaming: only the "
                f"DALL-E models rejected 'stream'"
            )
        _skip_retired_official_image_model(
            use_official_api, image_generation_model, "stream"
        )

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(
                prompt="A test image",
                model=image_generation_model,
                n=1,
                size=image_generation_size,
                stream=True,
            )

        validate_error_response(
            exc_info.value,
            expected_type="invalid_request_error",
            expected_code="unknown_parameter",
            expected_param="stream",
        )

    def test_partial_images_without_stream_error(
        self,
        openai_client: OpenAI,
        image_generation_model: str,
        image_generation_size: str,
        use_official_api: bool,
    ) -> None:
        """``partial_images`` without ``stream=True`` is rejected as a 400.

        The expected envelope follows the selected lane. The incompatibility is
        caught by a model-level validator on the gateway, whose envelope is an
        ``invalid_request_error`` naming ``partial_images`` in the message and
        carrying no ``param``. OpenAI answers an ``image_generation_user_error``
        (code ``unsupported_parameter``) that blames the whole ``input``, so the
        field name never appears in its envelope.

        Ref: stdapi/types/openai_images.py:ImageGenerateParams._unsupported
             stdapi/main.py:handle_validation_exception
        """
        _skip_retired_official_image_model(
            use_official_api, image_generation_model, "partial_images"
        )

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.generate(
                prompt="A test image",
                model=image_generation_model,
                n=1,
                size=image_generation_size,
                partial_images=1,
            )

        validate_error_response(
            exc_info.value,
            expected_type=(
                "image_generation_user_error"
                if use_official_api
                else "invalid_request_error"
            ),
            expected_code="unsupported_parameter" if use_official_api else None,
            expected_param="input" if use_official_api else "partial_images",
        )

    @pytest.mark.gateway(
        "Bedrock model catalog (variations-only models) is gateway-specific; "
        "no equivalent model exists on the official API"
    )
    def test_model_not_supporting_text_to_image(self, openai_client: OpenAI) -> None:
        """An edit-only model is rejected on the text-to-image route.

        Upscale models expose no text-to-image job, so the shared base
        implementation raises instead of building a Bedrock request.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._generate_images_from_text
        """
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
        local_test_client: TestClientType,
        image_generation_model: str,
        image_generation_size: str,
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
        capfd.readouterr()
        response = local_test_client.post(
            "/v1/images/generations",
            json={
                "model": image_generation_model,
                "prompt": "A red circle on a white background",
                "n": 1,
                "size": image_generation_size,
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


class TestBuildImagesResponseUrlFormat:
    """``response_format="url"`` returns links, ``b64_json`` returns payloads.

    The job has already turned each image into a presigned link by then, so the
    only difference here is the field it lands in — and putting a URL in
    ``b64_json`` (or the reverse) breaks every client that reads one of them.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/routes/_images_common.py:build_images_response
    """

    @staticmethod
    def _job(input_tokens: int, output_tokens: int) -> ImageGenerationJobBase[Any]:
        """Build a finished 512x512 PNG job billed *input_tokens*/*output_tokens*."""
        job = ImageGenerationJobBase(
            model=cast("Any", None),
            prompt="a cat",
            count=1,
            width=512,
            height=512,
            quality="medium",
            style=None,
            output_format="png",
            output_compression=0,
            extra_params={},
        )
        # Response-side counters a real invocation fills in from the model.
        job._response_width = 512  # noqa: SLF001
        job._response_height = 512  # noqa: SLF001
        job._input_tokens = input_tokens  # noqa: SLF001
        job._output_tokens = output_tokens  # noqa: SLF001
        return job

    @pytest.mark.local
    @pytest.mark.usefixtures("request_log")
    async def test_url_format_fills_url_and_leaves_b64_json_unset(self) -> None:
        """Each result lands in ``url`` only, with the usage split still reported."""
        REQUEST_TIME.set(datetime.now(UTC))

        response = await build_images_response(
            job=self._job(5, 1),
            results=[
                ImageGenerationResponse(image="https://example.invalid/a", index=0)
            ],
            response_format="url",
            output_image_count=1,
            input_image_count=2,
        )

        assert response.data is not None
        assert response.data[0].url == "https://example.invalid/a"
        assert response.data[0].b64_json is None
        assert response.usage is not None
        assert response.usage.input_tokens_details is not None
        # Two input images out of five input tokens leaves three text tokens.
        assert response.usage.input_tokens_details.image_tokens == 2
        assert response.usage.input_tokens_details.text_tokens == 3

    @pytest.mark.local
    @pytest.mark.usefixtures("request_log")
    async def test_b64_json_format_fills_b64_json_and_leaves_url_unset(self) -> None:
        """The same results in ``b64_json`` mode land in the payload field instead."""
        REQUEST_TIME.set(datetime.now(UTC))

        response = await build_images_response(
            job=self._job(0, 0),
            results=[ImageGenerationResponse(image="aaa", index=0)],
            response_format="b64_json",
            output_image_count=1,
        )

        assert response.data is not None
        assert response.data[0].b64_json == "aaa"
        assert response.data[0].url is None
        assert response.usage is not None
        # A reported zero is a real count, not a missing one to substitute.
        assert response.usage.input_tokens == 0
        assert response.usage.output_tokens == 0


class TestStreamGeneratorEventNames:
    """stream_generator: each endpoint emits its own SSE event family.

    OpenAI gives the edits endpoint ``image_edit.*`` event names and the
    generations endpoint ``image_generation.*`` ones, and the client
    discriminates each stream union on them: an event named for the wrong
    endpoint parses into the wrong model, leaving ``usage`` an unconverted dict.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/routes/openai_images_generations.py:stream_generator
    """

    @staticmethod
    def _job() -> ImageGenerationJobBase[Any]:
        """Build a minimal job whose counters are already final."""
        job = object.__new__(ImageGenerationJobBase)
        job._count = 1  # noqa: SLF001
        job._response_width = 512  # noqa: SLF001
        job._response_height = 512  # noqa: SLF001
        job._response_quality = "medium"  # noqa: SLF001
        job._output_format = "png"  # noqa: SLF001
        job._response_output_format = "png"  # noqa: SLF001
        job._input_tokens = 3  # noqa: SLF001
        job._output_tokens = 1  # noqa: SLF001
        return job

    @pytest.mark.local
    @pytest.mark.parametrize(
        ("edit", "prefix"), [(False, "image_generation"), (True, "image_edit")]
    )
    async def test_event_names_follow_the_endpoint(
        self, edit: bool, prefix: str
    ) -> None:
        """A partial and a completed event both carry the endpoint's own prefix."""
        REQUEST_TIME.set(datetime.now(UTC))
        log_token = REQUEST_LOG.set(cast("EventLog", {"level": "info"}))
        try:
            job = self._job()

            async def image_stream() -> AsyncGenerator[ImageGenerationResponse]:
                """Yield one preview frame and then the final image."""
                yield ImageGenerationResponse(image="aaa", index=0, partial=True)
                yield ImageGenerationResponse(image="bbb", index=0)

            events = []
            async for event in stream_generator(
                image_stream(), job, created=0, input_image_count=1, edit=edit
            ):
                assert event.data is not None
                events.append(json.loads(event.data))

            assert [e["type"] for e in events] == [
                f"{prefix}.partial_image",
                f"{prefix}.completed",
            ]
        finally:
            REQUEST_LOG.reset(log_token)

    @pytest.mark.local
    async def test_input_images_are_counted_in_stream_usage(self) -> None:
        """Streamed usage attributes the input images the edit consumed.

        The non-streaming edits path reports the source image and mask as input
        image tokens; a stream that leaves them out under-reports the same edit.
        """
        REQUEST_TIME.set(datetime.now(UTC))
        log_token = REQUEST_LOG.set(cast("EventLog", {"level": "info"}))
        try:
            job = self._job()

            async def image_stream() -> AsyncGenerator[ImageGenerationResponse]:
                """Yield the single edited image."""
                yield ImageGenerationResponse(image="bbb", index=0)

            events = []
            async for event in stream_generator(
                image_stream(), job, created=0, input_image_count=2, edit=True
            ):
                assert event.data is not None
                events.append(json.loads(event.data))

            usage = events[-1]["usage"]
            assert usage["input_tokens_details"] == {
                "image_tokens": 2,
                "text_tokens": 1,
            }
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


def _generation_params(**kwargs: object) -> ImageGenerateParams:
    """Build generation request params, filling in the required prompt.

    Args:
        **kwargs: Fields under test, merged over the required ones.

    Returns:
        The validated generation request params.
    """
    return ImageGenerateParams.model_validate({"model": "m", "prompt": "p", **kwargs})


def _edit_params(**kwargs: object) -> ImageEditParams:
    """Build multipart edit request params (their prompt defaults to empty).

    Args:
        **kwargs: Fields under test, merged over the required ones.

    Returns:
        The validated edit request params.
    """
    return ImageEditParams.model_validate({"model": "m", **kwargs})


#: Both request models declaring ``partial_images``, each with its own validator.
_PARAMS_FACTORIES = pytest.mark.parametrize(
    "make_params", [_generation_params, _edit_params], ids=["generations", "edits"]
)


class TestPartialImagesAccepted:
    """``partial_images``: accepted (0-3) with ``stream``, rejected without it.

    The generation and edit request models validate the field with two separate
    validators, so both are exercised here rather than in each endpoint's module.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/types/openai_images.py:ImageGenerateParams._unsupported
         stdapi/types/openai_images.py:_ImageEditCommonParams._unsupported
    """

    pytestmark = pytest.mark.local

    @_PARAMS_FACTORIES
    @pytest.mark.parametrize("value", [0, 1, 2, 3])
    def test_value_is_accepted(
        self, make_params: Callable[..., ImageParams], value: int
    ) -> None:
        """`partial_images` in 0-3 is accepted; no current model emits partial events."""
        params = make_params(stream=True, partial_images=value)
        assert params.partial_images == value
        assert params.stream is True

    @_PARAMS_FACTORIES
    def test_omitted_is_accepted(self, make_params: Callable[..., ImageParams]) -> None:
        """Omitting `partial_images` leaves it unset rather than defaulting to 0."""
        params = make_params(stream=True)
        assert params.partial_images is None

    @_PARAMS_FACTORIES
    def test_requires_stream(self, make_params: Callable[..., ImageParams]) -> None:
        """`partial_images` without `stream=True` is a single value error on the model."""
        with pytest.raises(ValidationError, match="partial_images") as exc_info:
            make_params(partial_images=0)

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "value_error"
        assert "partial_images requires streaming mode." in errors[0]["msg"]


@pytest.mark.local
class TestImageGenerationUnsupportedOptions:
    """Generation options the backend cannot honour are rejected, never ignored.

    Every response is built with ``background="opaque"`` and no Bedrock image
    model exposes a moderation level, so accepting these values would let a
    caller believe the request changed the output when it did not.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/types/openai_images.py:ImageGenerateParams._unsupported
    """

    def test_transparent_background_rejected(self) -> None:
        """``background="transparent"`` is a single value error on the request model.

        Ref: stdapi/routes/_images_common.py:build_images_response
        """
        with pytest.raises(ValidationError) as exc_info:
            _generation_params(background="transparent")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "value_error"
        assert "Background transparency is not supported" in errors[0]["msg"]

    @pytest.mark.parametrize("value", ["auto", "opaque"])
    def test_non_transparent_background_accepted(self, value: str) -> None:
        """``auto`` and ``opaque`` backgrounds are accepted and kept verbatim."""
        params = ImageGenerateParams.model_validate(
            {"model": "m", "prompt": "p", "background": value}
        )
        assert params.background == value

    def test_moderation_low_rejected(self) -> None:
        """``moderation="low"`` is rejected instead of silently keeping ``auto``.

        Loosening content filtering is not something the gateway can pass on to
        Bedrock, so accepting the value would misrepresent the safety level
        actually applied.
        """
        with pytest.raises(ValidationError) as exc_info:
            _generation_params(moderation="low")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "value_error"
        assert "'moderation' parameter is not supported" in errors[0]["msg"]

    def test_moderation_auto_is_the_default_and_accepted(self) -> None:
        """``auto`` is the default and the only accepted moderation level."""
        assert _generation_params().moderation == "auto"
        assert _generation_params(moderation="auto").moderation == "auto"


@pytest.mark.local
class TestResponseFormatUrlRequiresBucket:
    """``response_format="url"`` is refused when the server has no S3 bucket.

    ``url`` is the default for all three image endpoints, so without this guard
    a bucket-less deployment would generate the images, bill them, and only
    then fail on the upload that cannot happen. The refusal is the same one the
    upload would have raised -- a feature this deployment cannot run, answered
    like every other: 503 to the caller, the missing setting to the operator.

    Ref: https://stdapi.ai/api_openai_images_generations/
         stdapi/types/openai_images.py:_ImageBaseParams._validate_response_format
         stdapi/aws_s3.py:require_url_response_bucket
    """

    def test_url_rejected_without_a_bucket(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """Validation refuses the format as unavailable, and logs why for the operator."""
        monkeypatch.setattr(SETTINGS, "aws_s3_bucket", None)

        with pytest.raises(FeatureUnavailableError) as exc_info:
            _generation_params(response_format="url")

        assert exc_info.value.status == 503
        assert str(exc_info.value) == (
            "The 'url' response format is not available on the current server. "
            "Please contact the administrator to enable it."
        )
        assert "aws_s3_bucket" not in str(exc_info.value)
        assert any(
            "aws_s3_bucket" in str(detail) for detail in request_log["error_detail"]
        ), request_log
        assert request_log["level"] == "warning", (
            "an unconfigured bucket is an operator warning, not a critical"
        )

    def test_the_route_answers_the_refusal_as_a_503(
        self, app_client: TestClientType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The error raised during body validation still reaches the API envelope.

        Raised from a field validator, it travels out of FastAPI's own request
        parsing rather than out of the route body, which is the only thing that
        decides whether the caller reads the shared sentence or a 422.
        """
        monkeypatch.setattr(SETTINGS, "aws_s3_bucket", None)

        response = app_client.post(
            "/v1/images/generations",
            json={"model": "m", "prompt": "p", "response_format": "url"},
        )

        assert response.status_code == 503
        error = response.json()["error"]
        assert error["code"] == "feature_unavailable"
        assert "not available on the current server" in error["message"]

    def test_b64_json_still_accepted_without_a_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``b64_json`` needs no bucket and stays available on such a deployment."""
        monkeypatch.setattr(SETTINGS, "aws_s3_bucket", None)

        assert _generation_params(response_format="b64_json").response_format == (
            "b64_json"
        )


@pytest.mark.local
class TestImageGenerationJobParameters:
    """The route translates request options into image generation job parameters.

    ``quality`` is remapped through the OpenAI compatibility table and
    ``output_compression`` is forwarded as-is; both are only observable at the
    job boundary, so the job is stubbed and the request stops there.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/routes/openai_images_generations.py:create_images
    """

    @pytest.fixture
    def job_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        """Stub model resolution and the image model, recording the job parameters.

        The stub fails the request with a 400 once the parameters are recorded,
        so nothing reaches Bedrock.

        Returns:
            The dict the stub records the job keyword arguments into.
        """
        captured: dict[str, Any] = {}

        class _StubModel:
            """Image model recording the requested generation job parameters."""

            def get_image_generation_job(self, **kwargs: object) -> object:
                """Record the job parameters, then fail the request with a 400."""
                captured.update(kwargs)
                model_id = "stub-model"
                raise UnsupportedModelError(model_id, status=400)

        async def _validate_model(
            model_id: str, *_args: object, **_kwargs: object
        ) -> object:
            """Accept any model ID without calling AWS."""
            return SimpleNamespace(id=model_id)

        monkeypatch.setattr(
            openai_images_generations, "validate_model", _validate_model
        )
        monkeypatch.setattr(
            openai_images_generations, "get_image_model", lambda _model_id: _StubModel()
        )
        return captured

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [
            ("standard", "medium"),
            ("hd", "high"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("auto", None),
            ("premium", "premium"),
        ],
    )
    def test_quality_is_mapped_to_the_backend_levels(
        self,
        app_client: TestClientType,
        job_kwargs: dict[str, Any],
        requested: str,
        expected: str | None,
    ) -> None:
        """OpenAI quality names collapse to the three backend levels; others pass through.

        ``auto`` becomes ``None`` so the backend keeps its own default, and a
        backend-native value such as Nova Canvas' ``premium`` is forwarded
        unchanged.

        Ref: stdapi/routes/openai_images_generations.py:_OPENAI_QUALITY_LEVELS
        """
        response = app_client.post(
            "/v1/images/generations",
            json={
                "model": "stub-model",
                "prompt": "a cat",
                "quality": requested,
                "response_format": "b64_json",
            },
        )

        assert response.status_code == 400
        assert job_kwargs["quality"] == expected

    def test_output_compression_reaches_the_job(
        self, app_client: TestClientType, job_kwargs: dict[str, Any]
    ) -> None:
        """``output_compression`` is forwarded to the job as requested.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._encode_image
        """
        response = app_client.post(
            "/v1/images/generations",
            json={
                "model": "stub-model",
                "prompt": "a cat",
                "output_format": "jpeg",
                "output_compression": 42,
                "response_format": "b64_json",
            },
        )

        assert response.status_code == 400
        assert job_kwargs["output_compression"] == 42
        assert job_kwargs["output_format"] == "jpeg"

    def test_output_compression_below_the_minimum_is_rejected(
        self, app_client: TestClientType
    ) -> None:
        """``output_compression=0`` is outside the documented 1-100 range."""
        response = app_client.post(
            "/v1/images/generations",
            json={
                "model": "stub-model",
                "prompt": "a cat",
                "output_format": "jpeg",
                "output_compression": 0,
            },
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "output_compression" in error["message"]


@pytest.mark.local
class TestStreamGeneratorPartialImages:
    """stream_generator: partial images are numbered per source image, from 0.

    No Bedrock backend sets ``partial=True`` today, so the branch is only
    reachable from a stubbed stream — which is exactly why the numbering needs
    pinning: the first backend that streams previews would otherwise ship an
    off-by-one index to every OpenAI SDK client.

    Ref: stdapi/routes/openai_images_generations.py:stream_generator
         stdapi/types/openai_images.py:ImageGenPartialImageEvent
    """

    async def test_partial_events_carry_job_metadata_and_a_per_image_counter(
        self,
    ) -> None:
        """Each source image gets its own counter, and partials carry no usage."""
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
            job._input_tokens = 7  # noqa: SLF001
            job._output_tokens = 2  # noqa: SLF001

            async def image_stream() -> AsyncGenerator[ImageGenerationResponse]:
                """Emit two previews for image 0 and one for image 1, then finals."""
                yield ImageGenerationResponse(image="p0a", index=0, partial=True)
                yield ImageGenerationResponse(image="p0b", index=0, partial=True)
                yield ImageGenerationResponse(image="p1a", index=1, partial=True)
                yield ImageGenerationResponse(image="final0", index=0)
                yield ImageGenerationResponse(image="final1", index=1)

            events = [
                json.loads(event.data)
                async for event in stream_generator(image_stream(), job, created=11)
                if event.data is not None
            ]
        finally:
            REQUEST_LOG.reset(log_token)

        partials = [
            event
            for event in events
            if event["type"] == "image_generation.partial_image"
        ]
        assert [event["b64_json"] for event in partials] == ["p0a", "p0b", "p1a"]
        # The counter restarts for every source image index.
        assert [event["partial_image_index"] for event in partials] == [0, 1, 0]
        assert all(event["created_at"] == 11 for event in partials)
        assert all(event["size"] == "512x512" for event in partials)
        assert all(event["output_format"] == "png" for event in partials)
        assert all(event["quality"] == "medium" for event in partials)
        assert all(event["background"] == "opaque" for event in partials)
        assert all("usage" not in event for event in partials), (
            "usage is only reported on completed events"
        )
        assert [
            event["b64_json"]
            for event in events
            if event["type"] == "image_generation.completed"
        ] == ["final0", "final1"]
