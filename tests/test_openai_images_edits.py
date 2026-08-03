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
from stdapi.models.image import ImageGenerationResponse
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
    from fastapi import UploadFile
    from starlette.datastructures import FormData
    from starlette.testclient import TestClient

#: Shape of the ``size`` field built by ``build_images_response`` ("WIDTHxHEIGHT")
_SIZE_PATTERN = re.compile(r"^\d+x\d+$")

#: Model id used by tests that must fail at model resolution, before any AWS call.
_PROBE_MODEL_ID = "probe-model-id"


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

        An ignored ``image[]`` field fails with a 400 "at least 1 image" before
        model resolution, so failing on the unknown model is what proves the
        upload was parsed.

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

        Nova Canvas is the only backend offering VIRTUAL_TRY_ON and it is legacy,
        so pinning a legacy model here is deliberate (#93).

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
             stdapi/models/image/amazon_nova_canvas.py:_get_request_virtual_try_on
        """
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
        """Streaming an edit emits a single ``image_edit.completed`` SSE event.

        The edits endpoint has its own event names, and the OpenAI client
        discriminates its edit stream union on them: an ``image_generation.*``
        name here leaves ``usage`` an unparsed dict on the client side. Stability
        backends never produce preview frames, so one event is the whole stream
        whatever ``partial_images`` asks for -- which is why a legal value rides
        on this request instead of paying for an edit per value.

        Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
             stdapi/types/openai_images.py:_ImageEditCommonParams
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
            partial_images=2,
        )

        events = list(response)
        validate_streaming_image_response(events, prefix="image_edit")
        assert [str(event.type) for event in events] == ["image_edit.completed"]
        assert validate_base64_image(events[-1].b64_json) == "png"

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

        The form-only ``model`` Form parameter must not constrain this path: a
        ``min_length`` on it rejects every JSON-body request with a 422 before
        the JSON ``model`` field is read.

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
class TestImagesEditsImageFieldBinding:
    """``image`` and ``image[]`` are merged by hand, so both are validated by hand.

    FastAPI resolves no ``validation_alias`` for multipart ``File`` parameters,
    so the route reads ``image[]`` off the raw form; the errors that binding
    would normally raise have to be produced explicitly and stay OpenAI-shaped.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/routes/openai_images_edits.py:_merge_image_parameters
    """

    def test_multipart_without_any_image_is_a_field_error_on_image(
        self, app_client: TestClient
    ) -> None:
        """A form carrying neither ``image`` nor ``image[]`` fails on ``image``.

        Model resolution must not run first: the client's mistake is the
        missing file, not an unknown model.
        """
        response = app_client.post(
            "/v1/images/edits",
            data={"model": "probe-model-id", "prompt": "Make it darker"},
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "image" in error["message"]
        assert error["code"] != "model_not_found"

    def test_a_non_file_value_under_image_bracket_is_rejected(
        self, app_client: TestClient
    ) -> None:
        """A plain text value posted as ``image[]`` cannot be read as an upload."""
        response = app_client.post(
            "/v1/images/edits",
            data={
                "model": "probe-model-id",
                "prompt": "Make it darker",
                "image[]": "not-a-file",
            },
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "image[]" in error["message"]

    def test_bracket_suffixed_uploads_are_merged_with_the_bare_field(
        self, app_client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Files sent as ``image`` and as ``image[]`` are collected into one list.

        The official SDK posts repeated files under ``image[]``; a client
        mixing both encodings must not lose either upload.
        """
        counts: list[int] = []

        def _capture(
            form_data: FormData, image_param: list[UploadFile] | None
        ) -> list[UploadFile]:
            """Count the merged uploads, then stop the request before any backend call."""
            counts.append(len(original(form_data, image_param)))
            raise UnsupportedModelError(_PROBE_MODEL_ID, status=400)

        original = openai_images_edits._merge_image_parameters  # noqa: SLF001
        monkeypatch.setattr(openai_images_edits, "_merge_image_parameters", _capture)

        response = app_client.post(
            "/v1/images/edits",
            data={"model": "probe-model-id", "prompt": "Make it darker"},
            files=[
                ("image", ("a.png", b"fake-a", "image/png")),
                ("image[]", ("b.png", b"fake-b", "image/png")),
                ("image[]", ("c.png", b"fake-c", "image/png")),
            ],
        )

        assert response.status_code == 400
        assert counts == [3]


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

        The Form ``size`` pattern must accept the OpenAI ``auto`` literal and
        not only ``WIDTHxHEIGHT``, as the JSON body path already does.

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


@pytest.mark.local
@pytest.mark.usefixtures("_stub_edit_job")
class TestImagesEditsMaskUsageAccounting:
    """A mask is billed as an input image, not as prompt text.

    The response usage splits the job's ``input_tokens`` into image tokens --
    capped at the number of inputs the request carried -- and the remaining
    text tokens. The mask is one of those inputs, so leaving it out of the count
    would silently reattribute it to the prompt.

    Ref: https://stdapi.ai/api_openai_images_edits/
         stdapi/routes/openai_images_edits.py:edit_images
         stdapi/routes/_images_common.py:build_images_response
    """

    @pytest.fixture
    def _stub_edit_job(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stub model resolution and the edit job so nothing reaches Bedrock.

        The job reports fixed token counts, leaving the image/text split as the
        only variable in the response usage.
        """

        class _StubJob:
            """Edit job returning one image with fixed billed token counts."""

            input_tokens = 6
            output_tokens = 1
            output_format = "png"
            width = 512
            height = 512
            quality = "medium"

            async def edit_images(
                self, images: list[str], mask: str | None
            ) -> list[ImageGenerationResponse]:
                """Return a single edited image, whatever the inputs were."""
                return [ImageGenerationResponse(image="ZWRpdA==", index=0)]

        class _StubModel:
            """Image model returning the fixed-usage edit job."""

            def get_image_edit_job(self, **_kwargs: object) -> _StubJob:
                """Return the stub job, ignoring the job parameters."""
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

    def test_mask_is_billed_as_a_second_input_image(
        self, app_client: TestClient
    ) -> None:
        """An edit with a mask reports two image tokens and the rest as text."""
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

        assert response.status_code == 200
        usage = response.json()["usage"]
        assert usage["input_tokens"] == 6
        assert usage["input_tokens_details"] == {"image_tokens": 2, "text_tokens": 4}

    def test_without_a_mask_only_the_source_image_is_billed(
        self, app_client: TestClient
    ) -> None:
        """The same edit without a mask attributes one token less to the images."""
        response = app_client.post(
            "/v1/images/edits",
            json={
                "model": "stub-model",
                "prompt": "Make it darker",
                "images": [{"image_url": "data:image/png;base64,aW1hZ2U="}],
                "response_format": "b64_json",
            },
        )

        assert response.status_code == 200
        usage = response.json()["usage"]
        assert usage["input_tokens"] == 6
        assert usage["input_tokens_details"] == {"image_tokens": 1, "text_tokens": 5}


@pytest.mark.local
class TestImagesEditsOutputEncodingParameters:
    """``output_format`` and ``output_compression`` reach the edit job.

    Both are documented as supported on this route, and both are only
    observable at the job boundary: the job is stubbed and the request stops
    there, since the produced bytes depend on the backend image.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/routes/openai_images_edits.py:edit_images
    """

    @pytest.fixture
    def job_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
        """Stub model resolution and the image model, recording the job parameters.

        Returns:
            The dict the stub records the edit job keyword arguments into.
        """
        captured: dict[str, object] = {}

        class _StubModel:
            """Image model recording the requested edit job parameters."""

            def get_image_edit_job(self, **kwargs: object) -> object:
                """Record the job parameters, then fail the request with a 400."""
                captured.update(kwargs)
                model_id = "stub-model"
                raise UnsupportedModelError(model_id, status=400)

        async def _validate_model(
            model_id: str, *_args: object, **_kwargs: object
        ) -> object:
            """Accept any model ID without calling AWS."""
            return SimpleNamespace(id=model_id)

        monkeypatch.setattr(openai_images_edits, "validate_model", _validate_model)
        monkeypatch.setattr(
            openai_images_edits, "get_image_model", lambda _model_id: _StubModel()
        )
        return captured

    def test_output_encoding_reaches_the_job(
        self, app_client: TestClient, job_kwargs: dict[str, object]
    ) -> None:
        """A jpeg request at 42% compression is forwarded verbatim to the job."""
        response = app_client.post(
            "/v1/images/edits",
            json={
                "model": "stub-model",
                "prompt": "Make it darker",
                "images": [{"image_url": "data:image/png;base64,aW1hZ2U="}],
                "output_format": "jpeg",
                "output_compression": 42,
                "response_format": "b64_json",
            },
        )

        assert response.status_code == 400
        assert job_kwargs["output_format"] == "jpeg"
        assert job_kwargs["output_compression"] == 42

    def test_output_compression_below_the_minimum_is_rejected(
        self, app_client: TestClient
    ) -> None:
        """``output_compression=0`` is outside the documented 1-100 range."""
        response = app_client.post(
            "/v1/images/edits",
            json={
                "model": "stub-model",
                "prompt": "Make it darker",
                "images": [{"image_url": "data:image/png;base64,aW1hZ2U="}],
                "output_format": "jpeg",
                "output_compression": 0,
            },
        )

        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "output_compression" in error["message"]
