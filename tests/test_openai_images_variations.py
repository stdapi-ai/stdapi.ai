"""Coverage of the OpenAI-compatible ``/v1/images/variations`` endpoint.

The gateway accepts both ``multipart/form-data`` (binary upload) and
``application/json`` (``image`` referenced by Files API id or URL); the OpenAI
schema only defines the multipart form.

Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     https://stdapi.ai/api_openai_images_variations/
     stdapi/routes/openai_images_variations.py:create_image_variations
"""

from typing import TYPE_CHECKING

import pytest
from openai import BadRequestError, NotFoundError, OpenAI

from stdapi.api_errors import UnsupportedModelError
from stdapi.routes import openai_images_variations
from tests import conftest

# Import validation helpers from generations tests
from .test_openai_images_generations import (
    validate_base64_image,
    validate_error_response,
    validate_timestamp,
    validate_url_format,
)

if TYPE_CHECKING:
    from starlette.testclient import TestClient

    from tests.conftest import VariationSize

#: Image model implementing editing but no variation operation.
_NO_VARIATION_MODEL = "stability.stable-fast-upscale-v1:0"


class TestImagesVariationsBasic:
    """Variation requests that carry an image and no prompt.

    Ref: stdapi/routes/_images_common.py:build_images_response
    """

    pytestmark = pytest.mark.gateway(
        "dall-e-2 was the only OpenAI model with a variations endpoint and it "
        "has been retired: POST /v1/images/variations now answers a Cloudflare "
        "502 HTML page instead of reaching the API at all"
    )

    @pytest.mark.expensive
    def test_create_variation_basic(
        self,
        openai_client: OpenAI,
        sample_image_file: bytes,
        image_variation_model: str,
        image_variation_size: VariationSize,
    ) -> None:
        """A prompt-less variation returns one presigned URL and image-only usage.

        ``response_format="url"`` uploads each result to S3 and returns a
        presigned URL, so ``b64_json`` stays unset. A variation carries no
        requested resolution to the backend, which renders at its own; the
        echoed ``size`` is therefore the produced image's, and only the square
        ratio of the square source image is guaranteed.

        Ref: https://docs.aws.amazon.com/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html
             stdapi/models/image/stability_stable_diffusion.py:TextToImageJob
        """
        response = openai_client.images.create_variation(
            image=sample_image_file,
            model=image_variation_model,
            size=image_variation_size,
            n=1,
        )

        assert response.created is not None
        validate_timestamp(response.created)
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].url is not None
        validate_url_format(response.data[0].url)
        assert response.data[0].b64_json is None, (
            "response_format=url must not also inline the image data"
        )
        # Stability generates at its own resolution but keeps the 1:1 ratio asked for.
        assert response.size is not None
        width, height = (int(value) for value in response.size.split("x"))
        assert width == height

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.input_tokens_details.image_tokens > 0
        assert response.usage.output_tokens == 1
        assert response.usage.total_tokens > 0
        assert (
            response.usage.total_tokens
            == response.usage.input_tokens + response.usage.output_tokens
        )

    @pytest.mark.expensive
    def test_create_variations_multiple(
        self,
        openai_client: OpenAI,
        sample_image_file: bytes,
        image_variation_model: str,
        image_variation_size: VariationSize,
    ) -> None:
        """``n=2`` returns two separately stored images.

        Stability models invoke Bedrock once per requested image and store each
        result under an index-suffixed S3 key, so the two presigned URLs differ.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._get_image_url
        """
        response = openai_client.images.create_variation(
            image=sample_image_file,
            model=image_variation_model,
            size=image_variation_size,
            n=2,
        )

        assert response.data is not None
        assert len(response.data) == 2
        for img_data in response.data:
            assert img_data.url is not None
            validate_url_format(img_data.url)
        assert len({img_data.url for img_data in response.data}) == 2, (
            "each variation must be stored under its own key"
        )

        assert response.usage is not None
        assert response.usage.output_tokens == 2

    @pytest.mark.expensive
    def test_create_variation_b64_json(
        self,
        openai_client: OpenAI,
        sample_image_file: bytes,
        image_variation_model: str,
        image_variation_size: VariationSize,
    ) -> None:
        """``response_format="b64_json"`` returns inline PNG data and no URL.

        The route requests no specific output format, so the Stability backend
        falls back to PNG and echoes it in ``output_format``.

        Ref: stdapi/models/image/_stability.py:StabilityImageGenerationJobBase._finalize_request
        """
        response = openai_client.images.create_variation(
            image=sample_image_file,
            model=image_variation_model,
            size=image_variation_size,
            n=1,
            response_format="b64_json",
        )

        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.data[0].url is None
        assert validate_base64_image(response.data[0].b64_json) == "png"
        assert response.output_format == "png"


class TestImagesVariationsErrors:
    """Rejections specific to variations: unknown, non-image and non-variation models."""

    @pytest.mark.gateway(
        "dall-e-2 was the only OpenAI model with a variations endpoint and it has "
        "been retired, so /v1/images/variations answers an HTML 5xx upstream "
        "instead of resolving a model at all"
    )
    def test_invalid_model(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """An unknown model id is rejected as ``model_not_found``.

        The route resolves the model before any Bedrock call and passes
        ``error_status=400``, so the gateway answers 400 where OpenAI answers
        404.

        Ref: stdapi/api_errors.py:UnsupportedModelError
             stdapi/models/__init__.py:validate_model
        """
        with pytest.raises((BadRequestError, NotFoundError)) as exc_info:
            openai_client.images.create_variation(
                image=sample_image_file, model="invalid-model"
            )

        error = exc_info.value
        if isinstance(error, BadRequestError):
            validate_error_response(error)
            # On the OpenAI client, ``body`` is already the inner error envelope.
            body = error.body
            assert isinstance(body, dict)
            assert body["type"] == "invalid_request_error"
            assert body["code"] == "model_not_found"
            assert "invalid-model" in body["message"]
        else:
            assert error.status_code == 404

    @pytest.mark.gateway(
        "Bedrock model catalog (generation/editing-only models) is "
        "gateway-specific; no equivalent model exists on the official API"
    )
    def test_unsupported_model_for_variations(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """A model whose job class implements no variation operation is rejected.

        The upscale model passes the route's IMAGE→IMAGE modality check, so the
        rejection comes from the job base class, which raises for every
        operation the concrete job does not override.

        Ref: stdapi/models/image/__init__.py:ImageGenerationJobBase._create_image_variations
             stdapi/models/image/stability_stable_fast_upscale.py:_FastUpscaleJob
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.create_variation(
                image=sample_image_file, model=_NO_VARIATION_MODEL
            )

        body = exc_info.value.body
        assert isinstance(body, dict)
        assert exc_info.value.status_code == 400
        assert body["type"] == "invalid_request_error"
        assert "image variations are not supported by" in body["message"].lower()
        assert _NO_VARIATION_MODEL in body["message"]

    @pytest.mark.gateway(
        "Bedrock chat model catalog is gateway-specific; no equivalent "
        "model exists on the official API"
    )
    def test_non_image_model_for_variations(
        self, openai_client: OpenAI, sample_image_file: bytes
    ) -> None:
        """A text-output chat model is rejected on the IMAGE output modality check.

        Claude 3.5 Sonnet accepts image input but only produces text, and
        ``validate_model`` checks the output modality first.

        Ref: stdapi/models/__init__.py:validate_model
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.create_variation(
                image=sample_image_file,
                model="anthropic.claude-3-5-sonnet-20241022-v2:0",  # Chat model, not image model
            )

        body = exc_info.value.body
        assert isinstance(body, dict)
        assert exc_info.value.status_code == 400
        assert body["type"] == "invalid_request_error"
        assert "does not support image output modality" in body["message"]


class TestImagesVariationsProviderParams:
    """Provider-specific extra parameters forwarded to the Bedrock request body.

    Ref: stdapi/aws_bedrock.py:get_extra_model_parameters
         stdapi/routes/openai_images_variations.py:_KNOWN_PARAMS
    """

    pytestmark = pytest.mark.gateway("Unittest only for local tests.")

    @pytest.mark.expensive
    def test_variation_with_strength(
        self,
        openai_client: OpenAI,
        sample_image_file_base64: str,
        image_variation_model: str,
        image_variation_size: VariationSize,
    ) -> None:
        """Stability's numeric ``strength`` extra parameter is accepted in a JSON body.

        Sent as a JSON body: over ``multipart/form-data`` every extra parameter
        reaches the model as a string, which Bedrock rejects for numeric fields.
        Extras are merged into the request before the gateway's
        ``setdefault("strength", 0.35)``, so the sent value wins.

        Ref: https://github.com/stdapi-ai/stdapi.ai/issues/91
             stdapi/models/image/stability_stable_diffusion.py:TextToImageJob._create_image_variations
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}images/variations",
            json={
                "model": image_variation_model,
                "image": {"image_url": sample_image_file_base64},
                "size": image_variation_size,
                "n": 1,
                "strength": 0.7,
            },
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )

        assert response.status_code == 200, (
            f"Expected 200, got {response.status_code}: {response.text}"
        )
        body = response.json()
        assert len(body["data"]) == 1
        assert body["data"][0]["url"]
        assert body["usage"]["output_tokens"] == 1

    @pytest.mark.expensive
    def test_variation_with_negative_prompt(
        self,
        openai_client: OpenAI,
        sample_image_file: bytes,
        image_variation_model: str,
        image_variation_size: VariationSize,
    ) -> None:
        """Stability's string ``negative_prompt`` extra parameter is accepted over multipart.

        Form fields outside ``_KNOWN_PARAMS`` are collected as model extras, so a
        string-valued provider parameter is forwarded instead of triggering a
        validation error (numeric ones cannot take this path, see issue 91).

        Ref: https://github.com/stdapi-ai/stdapi.ai/issues/91
             stdapi/models/image/_stability.py:TextToImageRequest
        """
        response = openai_client.images.create_variation(
            image=sample_image_file,
            model=image_variation_model,
            size=image_variation_size,
            n=1,
            extra_body={"negative_prompt": "blurry, low quality"},
        )

        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].url is not None
        validate_url_format(response.data[0].url)
        assert response.usage is not None
        assert response.usage.output_tokens == 1


class TestImagesVariationsJsonBody:
    """Variation requests using the gateway's ``application/json`` body form.

    Ref: stdapi/types/openai_images.py:ImageVariationJsonBody
    """

    pytestmark = pytest.mark.gateway(
        "the official OpenAI API removed the variations endpoint, and the "
        "Bedrock model this route's variations run on has no counterpart there"
    )

    @pytest.mark.expensive
    def test_variation_with_file_id(
        self,
        openai_client: OpenAI,
        sample_image_file: bytes,
        image_variation_model: str,
        image_variation_size: VariationSize,
    ) -> None:
        """A Files API ``file_id`` reference in a JSON body produces a variation.

        The gateway's Files API is S3-backed and its identifiers are prefixed
        ``file-``; the route reads the object back and re-encodes it as the model
        input.

        Ref: stdapi/files/_core.py:parse_file_id
             stdapi/input_file.py:InputFile
        """
        uploaded = openai_client.files.create(
            file=("image.png", sample_image_file, "image/png"), purpose="assistants"
        )
        try:
            assert uploaded.id.startswith("file-")
            http_client = openai_client._client  # noqa: SLF001
            response = http_client.post(
                f"{openai_client.base_url}images/variations",
                json={
                    "model": image_variation_model,
                    "image": {"file_id": uploaded.id},
                    "response_format": "b64_json",
                    "size": image_variation_size,
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
            assert validate_base64_image(body["data"][0]["b64_json"]) == "png"
        finally:
            openai_client.files.delete(uploaded.id)

    def test_missing_image_returns_400(
        self, openai_client: OpenAI, image_variation_model: str
    ) -> None:
        """A JSON body without ``image`` is rejected, naming that field.

        ``image`` is the only required field the JSON body adds over the
        multipart form, and the gateway reports the Pydantic location in the
        message.

        Ref: stdapi/main.py:handle_validation_exception
        """
        http_client = openai_client._client  # noqa: SLF001
        response = http_client.post(
            f"{openai_client.base_url}images/variations",
            json={"model": image_variation_model},
            headers={"Authorization": f"Bearer {openai_client.api_key}"},
        )
        assert response.status_code == 400
        error = response.json().get("error", {})
        assert error.get("type") == "invalid_request_error"
        assert error.get("message", "").startswith("Validation error at image:")


@pytest.fixture
def probed_model_ids(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub validate_model to record the requested model id and fail fast.

    Returns:
        The list the stub appends every requested model id to.
    """
    calls: list[str] = []

    async def _validate_model(model_id: str, *_args: object, **_kwargs: object) -> None:
        calls.append(model_id)
        raise UnsupportedModelError(model_id, status=400)

    monkeypatch.setattr(openai_images_variations, "validate_model", _validate_model)
    return calls


@pytest.mark.local
class TestImagesVariationsModelField:
    """Model-field plumbing for both request encodings, with model resolution stubbed.

    Ref: stdapi/routes/openai_images_variations.py:create_image_variations
    """

    def test_json_body_model_field_reaches_model_resolution(
        self, app_client: TestClient, probed_model_ids: list[str]
    ) -> None:
        """A model supplied only in the JSON body reaches model resolution.

        Regression: the unused, form-only 'model' Form parameter carried
        'min_length=1' with an empty-string default, which rejected every
        JSON-body request with a 422 before the JSON 'model' field was read.
        """
        response = app_client.post(
            "/v1/images/variations",
            json={
                "model": "probe-model-id",
                "image": {"image_url": "data:image/png;base64,AA=="},
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"
        assert probed_model_ids == ["probe-model-id"]

    def test_multipart_form_model_field_still_works(
        self, app_client: TestClient, probed_model_ids: list[str]
    ) -> None:
        """A model supplied via multipart form data still reaches model resolution unchanged."""
        response = app_client.post(
            "/v1/images/variations",
            data={"model": "probe-model-id", "response_format": "b64_json"},
            files={"image": ("image.png", b"fake-bytes", "image/png")},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"
        assert probed_model_ids == ["probe-model-id"]

    def test_multipart_missing_image_returns_400(
        self, app_client: TestClient, image_variation_model: str
    ) -> None:
        """A multipart request without an 'image' file returns 400, not an unhandled error.

        Regression: the ValidationError for the missing 'image' file was raised
        outside validation_error_handler(), bypassing FastAPI's RequestValidationError
        conversion and producing an unhandled 500 instead of a JSON 400 envelope.

        Ref: stdapi/utils.py:validation_error_handler
             stdapi/main.py:handle_validation_exception
        """
        response = app_client.post(
            "/v1/images/variations", data={"model": image_variation_model}
        )
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"
        assert response.json()["error"]["message"].startswith(
            "Validation error at body.image:"
        )


@pytest.mark.local
class TestImagesVariationsSizeAuto:
    """``size="auto"`` is accepted on both the JSON body and the multipart form.

    ``_ImageBaseParams`` resolves ``auto`` to the default size for both
    encodings; the multipart ``size`` form field pattern matches ``auto`` too,
    consistent with generations and edits.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/routes/openai_images_variations.py:create_image_variations
         stdapi/types/openai_images.py:_ImageBaseParams._resolve_auto_size
    """

    def test_json_body_auto_size_reaches_model_resolution(
        self, app_client: TestClient, probed_model_ids: list[str]
    ) -> None:
        """A JSON body ``size="auto"`` is resolved and the request reaches the model."""
        response = app_client.post(
            "/v1/images/variations",
            json={
                "model": "probe-model-id",
                "image": {"image_url": "data:image/png;base64,AA=="},
                "size": "auto",
                "response_format": "b64_json",
            },
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"
        assert probed_model_ids == ["probe-model-id"]

    def test_multipart_form_auto_size_is_accepted(
        self, app_client: TestClient, probed_model_ids: list[str]
    ) -> None:
        """A multipart ``size="auto"`` passes the form pattern and reaches model resolution.

        Regression: the form ``size`` pattern only matched ``WIDTHxHEIGHT``,
        rejecting ``auto`` with a 422 before the edits-route twin's pattern fix
        was mirrored here.
        """
        response = app_client.post(
            "/v1/images/variations",
            data={
                "model": "probe-model-id",
                "size": "auto",
                "response_format": "b64_json",
            },
            files={"image": ("image.png", b"fake-bytes", "image/png")},
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"
        assert probed_model_ids == ["probe-model-id"]

    def test_multipart_form_explicit_size_is_accepted(
        self, app_client: TestClient, probed_model_ids: list[str]
    ) -> None:
        """An explicit ``WIDTHxHEIGHT`` multipart size passes the pattern."""
        response = app_client.post(
            "/v1/images/variations",
            data={
                "model": "probe-model-id",
                "size": "512x512",
                "response_format": "b64_json",
            },
            files={"image": ("image.png", b"fake-bytes", "image/png")},
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "model_not_found"
        assert probed_model_ids == ["probe-model-id"]


@pytest.mark.local
class TestImageVariationSizeValidation:
    """``conftest.variation_size`` rejects a mapped size the SDK does not type.

    Repointing ``MODEL_MAPPINGS["local"]["image_variation"]`` at a model whose
    cheapest size (:data:`conftest.IMAGE_MODEL_SIZES`) is not one of the three
    sizes ``images.create_variation`` accepts must fail loudly here, not as a
    live 400 from an ``--expensive`` run.

    Ref: tests/conftest.py:variation_size
    """

    def test_a_size_outside_the_sdk_literal_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A model mapped to an untyped size raises instead of being cast blind."""
        monkeypatch.setitem(conftest.IMAGE_MODEL_SIZES, "probe-model-id", "1536x1024")

        with pytest.raises(AssertionError, match="does not accept"):
            conftest.variation_size("probe-model-id")

    def test_a_size_inside_the_sdk_literal_is_accepted(self) -> None:
        """A model mapped to one of the three typed sizes returns it unchanged."""
        assert conftest.variation_size("amazon.titan-image-generator-v2:0") == "512x512"
