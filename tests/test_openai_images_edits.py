"""Coverage of the OpenAI-compatible /v1/images/edits endpoint.

Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     https://stdapi.ai/api_openai_images_edits/
     stdapi/routes/openai_images_edits.py:edit_images
"""

import re
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError, OpenAI
from pydantic import ValidationError

from stdapi.api_errors import UnsupportedModelError
from stdapi.routes import openai_images_edits
from stdapi.types.openai_images import ImageEditParams

# Import validation helpers from generations tests
from .test_openai_images_generations import (
    validate_base64_image,
    validate_error_response,
    validate_image_usage,
    validate_streaming_image_response,
    validate_timestamp,
    validate_url_format,
)

if TYPE_CHECKING:
    from starlette.testclient import TestClient

#: Shape of the ``size`` field built by ``build_images_response`` ("WIDTHxHEIGHT")
_SIZE_PATTERN = re.compile(r"^\d+x\d+$")


class TestImagesEditsBasic:
    """Editing-specific behavior of /v1/images/edits: masks, image fields, streaming.

    Ref: https://stdapi.ai/api_openai_images_edits/
         stdapi/routes/openai_images_edits.py:edit_images
    """

    @pytest.mark.expensive
    def test_edit_image_with_mask(
        self, openai_client: OpenAI, sample_image_file: bytes, sample_mask_file: bytes
    ) -> None:
        """An explicit mask is accepted and the edited image is returned as a URL.

        ``sample_mask_file`` is an RGB PNG with no alpha channel, so the gateway
        forwards it unchanged; Stability inpaint reads white pixels as maximum
        inpaint strength, i.e. its white centre square is the regenerated area.
        The mask counts as an input image in the usage breakdown, and ``size``
        is measured on the produced image rather than echoing the requested
        ``512x512``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/stable-image-services.html
             stdapi/routes/_images_common.py:build_images_response
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_mask_file,
            prompt="A blue circle in the masked area",
            model="stability.stable-image-inpaint-v1:0",
            size="512x512",
            n=1,
        )

        assert response.created is not None
        validate_timestamp(response.created)
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].url is not None
        validate_url_format(response.data[0].url)
        assert response.data[0].b64_json is None

        # Response metadata built by build_images_response
        assert response.background == "opaque"
        assert response.output_format == "png"
        assert response.size is not None
        assert _SIZE_PATTERN.match(response.size), response.size

        # Validate usage metadata
        assert response.usage is not None
        assert response.usage.output_tokens > 0
        validate_image_usage(response.usage)

    @pytest.mark.expensive
    def test_edit_image_b64_json_with_mask(
        self, openai_client: OpenAI, sample_image_file: bytes, sample_mask_file: bytes
    ) -> None:
        """`response_format=b64_json` returns inline image data and no URL.

        ``output_format`` is reported from the format the backend actually
        produced, so it must match the magic bytes of the returned payload.

        Ref: stdapi/routes/_images_common.py:build_images_response
             stdapi/models/image/__init__.py:ImageGenerationJobBase.output_format
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_mask_file,
            prompt="A colorful pattern",
            model="stability.stable-image-inpaint-v1:0",
            size="512x512",
            n=1,
            response_format="b64_json",
        )

        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.data[0].url is None

        image_format = validate_base64_image(response.data[0].b64_json)
        assert image_format == "png"
        assert response.output_format == image_format

    def test_invalid_model(
        self,
        openai_client: OpenAI,
        sample_image_file: bytes,
        image_generation_size: str,
        use_official_api: bool,
    ) -> None:
        """An unknown model id is rejected with a 400 naming the requested model.

        The two targets label that 400 differently: OpenAI answers
        ``image_generation_user_error``/``invalid_value`` on its image
        endpoints, whereas the gateway answers the generic
        ``invalid_request_error``/``model_not_found``.

        Ref: stdapi/models/__init__.py:validate_model
             stdapi/api_errors.py:UnsupportedModelError
        """
        expected_type = (
            "image_generation_user_error"
            if use_official_api
            else "invalid_request_error"
        )
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                prompt="A test image",
                model="invalid-model-name",
                size=image_generation_size,
            )
        validate_error_response(
            exc_info.value,
            expected_type=expected_type,
            expected_code="invalid_value" if use_official_api else "model_not_found",
            expected_param="model",
        )
        assert exc_info.value.type == expected_type
        assert "invalid-model-name" in str(exc_info.value)

    @pytest.mark.gateway
    def test_invalid_mask_format(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """A mask that is not a decodable image fails the request with a 400.

        The gateway never decodes the mask: the bytes are base64-encoded and
        forwarded, and the backend's ``ValidationException`` is mapped to a 400
        ``invalid_request_error``. The message text comes from AWS, so only the
        status and the envelope are asserted. The mapping of a Bedrock
        ``ValidationException`` onto the OpenAI envelope exists only in the
        gateway, and the inpaint model it needs is Bedrock-only.

        Ref: stdapi/aws_bedrock.py:handle_bedrock_client_error
             stdapi/routes/openai_images_edits.py:edit_images
        """
        invalid_mask = b"not a valid image"

        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                mask=invalid_mask,
                prompt="A test image",
                model="stability.stable-image-inpaint-v1:0",
                size="512x512",
            )
        validate_error_response(exc_info.value)
        assert exc_info.value.type == "invalid_request_error"
        body = exc_info.value.body
        assert isinstance(body, dict)
        assert body["message"], "error envelope must carry a message"

    def test_image_array_notation_accepted(
        self,
        openai_client: OpenAI,
        sample_image_file: bytes,
        image_generation_size: str,
        use_official_api: bool,
    ) -> None:
        """The multipart ``image[]`` field name is merged into the image list.

        Regression: ``image[]`` was silently ignored after the JSON body support
        was added, causing a 400 "at least 1 image" error instead of reaching
        the model. Failing on the unknown model instead proves the upload was
        parsed.

        OpenAI labels that unknown-model 400 ``image_generation_user_error`` on
        its image endpoints, where the gateway uses ``invalid_request_error``.

        Ref: stdapi/routes/openai_images_edits.py:_merge_image_parameters
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}images/edits",
            files={"image[]": ("image.png", sample_image_file, "image/png")},
            data={
                "prompt": "test",
                "model": "invalid-model-name",
                "size": image_generation_size,
            },
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        # image[] was parsed → error is about the model, not missing image
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == (
            "image_generation_user_error"
            if use_official_api
            else "invalid_request_error"
        )
        assert "model" in error["message"].lower()
        assert "invalid-model-name" in error["message"]

    def test_image_array_notation_invalid_type(
        self, openai_client: OpenAI, image_generation_size: str, use_official_api: bool
    ) -> None:
        """A non-file value under ``image[]`` is a validation error on ``body.image[]``.

        Sending only form fields makes httpx encode the body as
        ``application/x-www-form-urlencoded``. The gateway parses that encoding
        like multipart and reports the offending field, while OpenAI accepts
        only ``multipart/form-data`` or ``application/json`` here and rejects
        the request on its content type before looking at any field. Both
        targets answer 400 ``invalid_request_error``.

        Ref: stdapi/routes/openai_images_edits.py:_merge_image_parameters
             stdapi/main.py:handle_validation_exception
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}images/edits",
            data={
                "prompt": "test",
                "model": "stability.stable-image-inpaint-v1:0",
                "size": image_generation_size,
                "image[]": "not_a_file",
            },
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        expected = (
            "application/x-www-form-urlencoded" if use_official_api else "body.image[]"
        )
        assert expected in error["message"], error["message"]

    @pytest.mark.skip("Currently no models to use for this test case")
    def test_model_not_supporting_image_editing(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """A model without edit support reports that editing is unsupported.

        Skipped: every catalogued image model that accepts an image input also
        implements ``_edit_image``, so no model id reaches the base-class
        fallback.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._edit_image
        """
        # Use a model that only supports text-to-image generation (not editing)
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                prompt="Edit this image",
                model="",  # Only supports generation and variations
                size="512x512",
            )

        error_msg = str(exc_info.value).lower()
        assert (
            "not supported" in error_msg
            or "editing" in error_msg
            or "edit" in error_msg
        )

    @pytest.mark.gateway
    def test_multiple_images_error(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """Two images sent to a single-image edit model are rejected.

        The OpenAI schema allows up to 16 images for GPT image models; the
        Bedrock backends consume exactly one, so the extra image is rejected by
        the job rather than by request validation. Both the single-image limit
        and the model enforcing it are Bedrock-specific.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._get_one_image_from_list
        """
        # Pass a sequence of images to trigger the "exactly one image" validation
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=[sample_image_file, sample_image_file],
                prompt="Edit this image",
                model="stability.stable-image-inpaint-v1:0",
                size="512x512",
            )

        assert exc_info.value.type == "invalid_request_error"
        assert "Exactly one image must be provided." in str(exc_info.value)

    @pytest.mark.gateway(
        "Amazon Nova Canvas is not available on the official OpenAI API"
    )
    def test_mask_not_supported_error(
        self, openai_client: OpenAI, sample_image_file: bytes, sample_mask_file: bytes
    ) -> None:
        """A mask sent to a model with no mask input is rejected as unsupported.

        Background removal derives its own matte, so any supplied mask is
        refused instead of being silently dropped.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._validate_no_mask
             stdapi/models/image/stability_stable_image_remove_background.py:_RemoveBackgroundJob
        """
        # Use a model that doesn't support masks (like background removal or upscale)
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                mask=sample_mask_file,
                prompt="Remove background",
                model="stability.stable-image-remove-background-v1:0",
                size="512x512",
            )

        assert exc_info.value.type == "invalid_request_error"
        assert '"mask" parameter is not supported' in str(exc_info.value)

    @pytest.mark.gateway(
        "Amazon Nova Canvas is not available on the official OpenAI API"
    )
    def test_mask_required_error(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """VIRTUAL_TRY_ON without a mask is rejected because the mask is the reference image.

        Nova Canvas' VIRTUAL_TRY_ON task maps the OpenAI ``mask`` upload onto
        its ``referenceImage``, so the parameter is mandatory for that task type
        even though it is optional for the endpoint.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
             stdapi/models/image/amazon_nova_canvas.py:_get_request_virtual_try_on
        """
        # Use VIRTUAL_TRY_ON taskType which requires a mask
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.edit(
                image=sample_image_file,
                prompt="Try on garment",
                model="amazon.nova-canvas-v1:0",
                size="1024x1024",
                extra_body={"taskType": "VIRTUAL_TRY_ON"},
            )

        assert exc_info.value.type == "invalid_request_error"
        assert '"mask" parameter is required with VIRTUAL_TRY_ON taskType' in str(
            exc_info.value
        )

    @pytest.mark.expensive
    def test_edit_with_streaming(
        self, openai_client: OpenAI, sample_image_file: bytes, sample_mask_file: bytes
    ) -> None:
        """Streaming an edit emits a single ``image_generation.completed`` SSE event.

        The edits route reuses the generations stream serializer, so it emits
        OpenAI's ``image_generation.*`` event names instead of the
        ``image_edit.*`` names the OpenAI schema documents for this endpoint,
        and Stability backends never produce preview frames.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/routes/openai_images_generations.py:stream_generator
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            mask=sample_mask_file,
            prompt="Add colorful elements",
            model="stability.stable-image-inpaint-v1:0",
            size="512x512",
            n=1,
            stream=True,
        )

        events = list(response)
        # Validate the streaming response structure
        validate_streaming_image_response(events)
        assert [str(event.type) for event in events] == ["image_generation.completed"]
        assert validate_base64_image(events[-1].b64_json) == "png"

    @pytest.mark.expensive
    @pytest.mark.parametrize("partial_images_value", [0, 2, 3])
    def test_stream_with_partial_images(
        self, openai_client: OpenAI, sample_image_file: bytes, partial_images_value: int
    ) -> None:
        """Any accepted `partial_images` value still yields only the final image event.

        ``partial_images`` (0-3, streaming only) is validated and forwarded, but
        no available backend emits preview frames, so the stream carries exactly
        one ``image_generation.completed`` event and no ``partial_image`` event
        whatever the requested preview count.

        Ref: stdapi/types/openai_images.py:_ImageEditCommonParams
             stdapi/routes/openai_images_generations.py:stream_generator
        """
        response = openai_client.images.edit(
            image=sample_image_file,
            prompt="A test image",
            model="stability.stable-image-inpaint-v1:0",
            size="512x512",
            stream=True,
            partial_images=partial_images_value,
        )

        events = list(response)
        validate_streaming_image_response(events)
        assert [str(event.type) for event in events] == ["image_generation.completed"]

    @pytest.mark.expensive
    def test_image_parameter_aliases(
        self, openai_client: OpenAI, sample_image_file: bytes, sample_mask_file: bytes
    ) -> None:
        """`image` and `image[]` are interchangeable multipart field names.

        Also covers the two failure modes of the same merge step: no image field
        at all (min-length error on ``body.image``) and a non-file value under
        ``body.image[]``.

        Ref: stdapi/routes/openai_images_edits.py:_merge_image_parameters
        """
        # Derive HTTP client from OpenAI client
        http_client = openai_client._client  # noqa: SLF001

        # Prepare common headers with authentication
        headers = {
            "Authorization": f"Bearer {openai_client.api_key}",
            "OpenAI-Organization": openai_client.organization or "",
        }

        # Test with 'image' parameter name
        files_image = {
            "image": ("image.png", sample_image_file, "image/png"),
            "mask": ("mask.png", sample_mask_file, "image/png"),
        }
        data_image = {
            "prompt": "A red square",
            "model": "stability.stable-image-inpaint-v1:0",
            "size": "512x512",
            "n": "1",
        }

        response_image = http_client.post(
            f"{openai_client.base_url}images/edits",
            files=files_image,
            data=data_image,
            headers=headers,
        )
        assert response_image.status_code == 200, (
            f"Expected 200, got {response_image.status_code}: {response_image.text}"
        )
        json_response_image = response_image.json()
        assert json_response_image.get("created") is not None
        validate_timestamp(json_response_image["created"])
        assert json_response_image.get("data") is not None
        assert len(json_response_image["data"]) == 1
        assert json_response_image["data"][0].get("url") is not None
        validate_url_format(json_response_image["data"][0]["url"])

        # Test with 'image[]' parameter name (array notation)
        files_image_array = {
            "image[]": ("image.png", sample_image_file, "image/png"),
            "mask": ("mask.png", sample_mask_file, "image/png"),
        }
        data_image_array = {
            "prompt": "A blue circle",
            "model": "stability.stable-image-inpaint-v1:0",
            "size": "512x512",
            "n": "1",
        }

        response_image_array = http_client.post(
            f"{openai_client.base_url}images/edits",
            files=files_image_array,
            data=data_image_array,
            headers=headers,
        )
        assert response_image_array.status_code == 200, (
            f"Expected 200, got {response_image_array.status_code}: {response_image_array.text}"
        )
        json_response_array = response_image_array.json()
        assert json_response_array.get("created") is not None
        validate_timestamp(json_response_array["created"])
        assert json_response_array.get("data") is not None
        assert len(json_response_array["data"]) == 1
        assert json_response_array["data"][0].get("url") is not None
        validate_url_format(json_response_array["data"][0]["url"])

        # Test error case: no image provided
        data_no_image = {
            "prompt": "A test image",
            "model": "stability.stable-image-inpaint-v1:0",
            "size": "512x512",
            "n": "1",
        }

        response_no_image = http_client.post(
            f"{openai_client.base_url}images/edits", data=data_no_image, headers=headers
        )
        assert response_no_image.status_code == 400
        error_response = response_no_image.json()
        assert "error" in error_response
        assert error_response["error"]["type"] == "invalid_request_error"

        # Verify the error details match Pydantic's min_length validation format
        assert "message" in error_response["error"]
        error_message = error_response["error"]["message"]
        assert "body.image" in error_message, error_message
        assert (
            "validation" in error_message.lower() or "at least" in error_message.lower()
        )

        # Test error case: invalid type for image[] (string instead of file)
        data_invalid_type = {
            "prompt": "A test image",
            "model": "stability.stable-image-inpaint-v1:0",
            "size": "512x512",
            "n": "1",
            "image[]": "not_a_file",  # String instead of file upload
        }

        response_invalid_type = http_client.post(
            f"{openai_client.base_url}images/edits",
            data=data_invalid_type,
            headers=headers,
        )
        assert response_invalid_type.status_code == 400
        error_response_invalid = response_invalid_type.json()
        assert "error" in error_response_invalid
        assert error_response_invalid["error"]["type"] == "invalid_request_error"

        # Verify the error message indicates type mismatch
        assert "message" in error_response_invalid["error"]
        error_message_invalid = error_response_invalid["error"]["message"]
        assert "body.image[]" in error_message_invalid, error_message_invalid


class TestImagesEditsJsonBody:
    """/v1/images/edits with an ``application/json`` body instead of multipart.

    Images are referenced by URL/data URL or Files API identifier rather than
    uploaded as binary parts.

    Ref: https://stdapi.ai/api_openai_images_edits/
         stdapi/types/openai_images.py:ImageEditJsonBody
    """

    pytestmark = pytest.mark.gateway(
        "stability.stable-image-inpaint-v1:0 not available on the official OpenAI API"
    )

    @pytest.mark.expensive
    def test_edit_with_image_url(
        self, openai_client: OpenAI, sample_image_file_base64: str
    ) -> None:
        """A ``data:`` URL in the JSON ``images`` array is accepted as the source image.

        Ref: stdapi/types/openai_images.py:ImageInputReferenceParam
             stdapi/input_file.py:InputFile
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}images/edits",
            json={
                "model": "stability.stable-image-inpaint-v1:0",
                "prompt": "Make it look like a painting",
                "images": [{"image_url": sample_image_file_base64}],
                "response_format": "b64_json",
                "size": "512x512",
                "n": 1,
            },
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        body = response.json()
        assert body.get("created") is not None
        assert body.get("data") is not None
        assert len(body["data"]) == 1
        assert body["data"][0].get("b64_json") is not None
        assert body["data"][0].get("url") is None
        assert body["output_format"] == validate_base64_image(
            body["data"][0]["b64_json"]
        )

    @pytest.mark.expensive
    def test_edit_with_file_id(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """A Files API ``file_id`` reference in the JSON body resolves to the stored bytes.

        The Files API is S3-backed here and the identifier it returns carries
        the ``file-`` prefix that the edit body accepts.

        Ref: https://stdapi.ai/api_openai_files/
             stdapi/input_file.py:FileIdInputFile
        """
        uploaded = openai_client.files.create(
            file=("image.png", sample_image_file, "image/png"), purpose="assistants"
        )
        try:
            assert uploaded.id.startswith("file-"), uploaded.id
            http_client = openai_client._client  # noqa: SLF001
            response = http_client.post(
                f"{openai_client.base_url}images/edits",
                json={
                    "model": "stability.stable-image-inpaint-v1:0",
                    "prompt": "Make it darker",
                    "images": [{"file_id": uploaded.id}],
                    "response_format": "b64_json",
                    "size": "512x512",
                    "n": 1,
                },
                headers={"Authorization": f"Bearer {openai_client.api_key}"},
            )
            assert response.status_code == 200, (
                f"Expected 200, got {response.status_code}: {response.text}"
            )
            body = response.json()
            assert body.get("data") is not None
            assert len(body["data"]) == 1
            assert body["data"][0].get("b64_json") is not None
            assert body["output_format"] == validate_base64_image(
                body["data"][0]["b64_json"]
            )
        finally:
            openai_client.files.delete(uploaded.id)

    def test_missing_images_returns_400(self, openai_client: OpenAI) -> None:
        """A JSON body without an ``images`` array is a validation error naming the field.

        Ref: stdapi/types/openai_images.py:ImageEditJsonBody
             stdapi/main.py:handle_validation_exception
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}images/edits",
            json={
                "model": "stability.stable-image-inpaint-v1:0",
                "prompt": "Make it darker",
            },
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 400
        error = response.json().get("error", {})
        assert error.get("type") == "invalid_request_error"
        assert error["message"].startswith("Validation error"), error["message"]
        assert "images" in error["message"], error["message"]

    def test_empty_image_ref_returns_400(self, openai_client: OpenAI) -> None:
        """An ``images`` entry with neither ``file_id`` nor ``image_url`` is rejected.

        The failing element is reported positionally (``images.0``) by the
        model validator that requires exactly one image source.

        Ref: stdapi/types/openai_images.py:ImageInputReferenceParam
             stdapi/main.py:handle_validation_exception
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}images/edits",
            json={
                "model": "stability.stable-image-inpaint-v1:0",
                "prompt": "Make it darker",
                "images": [{}],
            },
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 400
        error = response.json().get("error", {})
        assert error.get("type") == "invalid_request_error"
        assert "images.0" in error["message"], error["message"]
        assert "file_id" in error["message"], error["message"]


@pytest.mark.local
class TestImagesEditsModelField:
    """The ``model`` field is read from either request encoding (no AWS calls).

    Ref: stdapi/routes/openai_images_edits.py:edit_images
    """

    @pytest.fixture
    def probed_model_ids(self, monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Stub validate_model to record the requested model id and fail fast."""
        calls: list[str] = []

        async def _validate_model(
            model_id: str, *_args: object, **_kwargs: object
        ) -> None:
            calls.append(model_id)
            raise UnsupportedModelError(model_id, status=400)

        monkeypatch.setattr(openai_images_edits, "validate_model", _validate_model)
        return calls

    def test_json_body_model_field_reaches_model_resolution(
        self, app_client: TestClient, probed_model_ids: list[str]
    ) -> None:
        """A model supplied only in the JSON body reaches model resolution.

        Regression: the unused, form-only 'model' Form parameter carried
        'min_length=1' with an empty-string default, which rejected every
        JSON-body request with a 422 before the JSON 'model' field was read.

        Ref: stdapi/types/openai_images.py:ImageEditJsonBody
        """
        response = app_client.post(
            "/v1/images/edits",
            json={
                "model": "probe-model-id",
                "prompt": "Make it darker",
                "images": [{"image_url": "data:image/png;base64,AA=="}],
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"
        assert probed_model_ids == ["probe-model-id"]

    def test_multipart_form_model_field_still_works(
        self, app_client: TestClient, probed_model_ids: list[str]
    ) -> None:
        """A model supplied via multipart form data reaches model resolution unchanged.

        Ref: stdapi/types/openai_images.py:ImageEditParams
        """
        response = app_client.post(
            "/v1/images/edits",
            data={"model": "probe-model-id", "prompt": "test"},
            files={"image": ("image.png", b"fake-bytes", "image/png")},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"
        assert probed_model_ids == ["probe-model-id"]


@pytest.mark.local
class TestImagesEditsUnsupportedOptions:
    """Edit options the backend cannot honour are rejected, never silently ignored.

    Every edit response is built with ``background="opaque"``, and no Bedrock
    backend exposes an input-fidelity control, so accepting these values would
    return an image that contradicts the request without any signal.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/types/openai_images.py:_ImageEditCommonParams._unsupported
    """

    def test_transparent_background_rejected(self) -> None:
        """``background="transparent"`` is a single value error on the request model.

        Ref: stdapi/routes/_images_common.py:build_images_response
        """
        with pytest.raises(ValidationError) as exc_info:
            ImageEditParams(model="m", background="transparent")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "value_error"
        assert "Background transparency is not supported" in errors[0]["msg"]

    @pytest.mark.parametrize("value", ["auto", "opaque"])
    def test_non_transparent_background_accepted(self, value: str) -> None:
        """``auto`` and ``opaque`` backgrounds are accepted and kept verbatim."""
        params = ImageEditParams.model_validate({"model": "m", "background": value})
        assert params.background == value

    def test_input_fidelity_high_rejected(self) -> None:
        """``input_fidelity="high"`` is rejected rather than accepted and ignored.

        The gateway has no way to bias a Bedrock edit towards the input image,
        so the documented "high" effort level fails validation instead of
        returning an image that ignored the request.
        """
        with pytest.raises(ValidationError) as exc_info:
            ImageEditParams(model="m", input_fidelity="high")

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert errors[0]["type"] == "value_error"
        assert "'input_fidelity' parameter is not supported" in errors[0]["msg"]

    def test_input_fidelity_low_is_the_default_and_accepted(self) -> None:
        """``low`` is the default and the only accepted input fidelity."""
        assert ImageEditParams(model="m").input_fidelity == "low"
        assert ImageEditParams(model="m", input_fidelity="low").input_fidelity == "low"

    def test_multipart_form_transparent_background_returns_400(
        self, app_client: TestClient
    ) -> None:
        """The multipart form path rejects ``transparent`` before resolving the model.

        The form field is validated by the same request model, so no Bedrock
        call is made and the failure carries the OpenAI error envelope.

        Ref: stdapi/routes/openai_images_edits.py:edit_images
        """
        response = app_client.post(
            "/v1/images/edits",
            data={
                "model": "probe-model-id",
                "prompt": "test",
                "background": "transparent",
            },
            files={"image": ("image.png", b"fake-bytes", "image/png")},
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "Background transparency is not supported" in error["message"]

    def test_multipart_form_input_fidelity_high_returns_400(
        self, app_client: TestClient
    ) -> None:
        """The multipart form path rejects ``input_fidelity=high`` with a 400.

        Ref: stdapi/routes/openai_images_edits.py:edit_images
        """
        response = app_client.post(
            "/v1/images/edits",
            data={
                "model": "probe-model-id",
                "prompt": "test",
                "input_fidelity": "high",
            },
            files={"image": ("image.png", b"fake-bytes", "image/png")},
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "'input_fidelity' parameter is not supported" in error["message"]


@pytest.mark.local
class TestImagesEditsSizeAuto:
    """Multipart form `size`: the OpenAI literal `auto` is accepted, not rejected.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/types/openai_images.py:_ImageBaseParams._resolve_auto_size
    """

    def test_multipart_form_auto_size_reaches_model_resolution(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`size=auto` in multipart form data is resolved instead of rejected with 422.

        Regression: the Form `size` parameter's pattern only matched
        `WIDTHxHEIGHT`, rejecting the OpenAI `auto` literal already accepted
        by the JSON body path.

        Ref: stdapi/routes/openai_images_edits.py:edit_images
        """

        async def _validate_model(
            model_id: str, *_args: object, **_kwargs: object
        ) -> None:
            raise UnsupportedModelError(model_id, status=400)

        monkeypatch.setattr(openai_images_edits, "validate_model", _validate_model)
        response = app_client.post(
            "/v1/images/edits",
            data={
                "model": "probe-model-id",
                "prompt": "test",
                "size": "auto",
                "response_format": "b64_json",
            },
            files={"image": ("image.png", b"fake-bytes", "image/png")},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"


@pytest.mark.local
class TestImagesEditsJsonBodyMask:
    """The JSON body ``mask`` is resolved and handed to the job as the mask, not an image.

    OpenAI's edits API is multipart-only, so the JSON body is a gateway extra;
    inpainting is its main use case, and the mask must stay separate from the
    ``images`` array because it also feeds the alpha-to-black/white conversion
    and the billed input image count.

    Ref: https://stdapi.ai/api_openai_images_edits/
         stdapi/routes/openai_images_edits.py:edit_images
    """

    @pytest.fixture
    def edit_job_calls(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
        """Stub model resolution and the edit job, recording its call arguments.

        The stub job fails with a 400 once it has recorded the arguments, so the
        request never reaches Bedrock and the response body is deterministic.

        Returns:
            The dict the stub job records ``images``/``mask`` into.
        """
        calls: dict[str, object] = {}

        class _StubJob:
            """Edit job recording the resolved images and mask."""

            async def edit_images(
                self, images: list[str], mask: str | None
            ) -> list[object]:
                """Record the arguments, then fail the request with a 400."""
                calls["images"] = images
                calls["mask"] = mask
                model_id = "stub-model"
                raise UnsupportedModelError(model_id, status=400)

        class _StubModel:
            """Image model returning the recording edit job."""

            def get_image_edit_job(self, **_kwargs: object) -> _StubJob:
                """Return the recording job, ignoring the job parameters."""
                return _StubJob()

        async def _validate_model(
            model_id: str, *_args: object, **_kwargs: object
        ) -> object:
            """Accept any model ID without calling AWS."""
            return SimpleNamespace(id=model_id)

        monkeypatch.setattr(openai_images_edits, "validate_model", _validate_model)
        monkeypatch.setattr(
            openai_images_edits, "get_image_model", lambda _model_id: _StubModel()
        )
        return calls

    def test_mask_reference_is_passed_as_the_mask(
        self, app_client: TestClient, edit_job_calls: dict[str, object]
    ) -> None:
        """A ``mask`` data URL is decoded and passed separately from ``images``."""
        response = app_client.post(
            "/v1/images/edits",
            json={
                "model": "stub-model",
                "prompt": "Make it darker",
                "images": [{"image_url": "data:image/png;base64,aW1hZ2U="}],
                "mask": {"image_url": "data:image/png;base64,bWFzaw=="},
                "response_format": "b64_json",
            },
        )

        assert response.status_code == 400
        assert edit_job_calls["images"] == ["aW1hZ2U="]
        assert edit_job_calls["mask"] == "bWFzaw=="

    def test_omitted_mask_leaves_the_job_mask_unset(
        self, app_client: TestClient, edit_job_calls: dict[str, object]
    ) -> None:
        """Without a ``mask`` field the job receives ``None``, not an extra image."""
        response = app_client.post(
            "/v1/images/edits",
            json={
                "model": "stub-model",
                "prompt": "Make it darker",
                "images": [{"image_url": "data:image/png;base64,aW1hZ2U="}],
                "response_format": "b64_json",
            },
        )

        assert response.status_code == 400
        assert edit_job_calls["images"] == ["aW1hZ2U="]
        assert edit_job_calls["mask"] is None
