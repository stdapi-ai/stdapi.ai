"""Amazon Nova Canvas backend of the OpenAI-compatible ``/v1/images/variations`` endpoint.

The whole module is skipped: Nova Canvas is deprecated (see the module-level
``pytest.skip`` below).

Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
     stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob._create_image_variations
"""

from typing import TYPE_CHECKING, cast

import pytest
from openai import BadRequestError

from tests.conftest import (
    NOVA_CANVAS_ALL,
    NOVA_CANVAS_SAMPLE,
    NOVA_CANVAS_V1,
    smallest_image_size,
)

#: Every test in this module is reported as skipped: the model is deprecated.
pytestmark = pytest.mark.skip(reason="Amazon Nova Canvas is deprecated")

if TYPE_CHECKING:
    from typing import Literal

    from openai import OpenAI

#: Cheapest size accepted by Nova Canvas, typed as the variations client expects it.
NOVA_CANVAS_SIZE = cast(
    'Literal["256x256", "512x512", "1024x1024"]', smallest_image_size(NOVA_CANVAS_V1)
)


@pytest.fixture(autouse=True)
def _skip_on_official_api(use_official_api: bool) -> None:
    """Skip every test here: Nova Canvas has no official OpenAI equivalent."""
    if use_official_api:
        pytest.skip("Amazon Nova Canvas is not available on the official OpenAI API")


class TestAmazonNovaCanvasVariations:
    """Nova Canvas ``taskType`` selection on the variations route.

    The route defaults to ``IMAGE_VARIATION`` and additionally accepts
    ``TEXT_IMAGE`` and ``COLOR_GUIDED_GENERATION``, which take the uploaded image
    as a conditioning/reference image instead.
    """

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_ALL)
    def test_variation_b64_single(
        self, openai_client: OpenAI, sample_image_file: bytes, model_id: str
    ) -> None:
        """The default ``IMAGE_VARIATION`` task returns one inline image and image-only usage.

        Nova Canvas honours the requested pixel size, and the variation route
        sends no prompt, so all reported input tokens are image tokens.

        Ref: stdapi/models/image/amazon_nova_canvas.py:_get_request_image_variation
             stdapi/routes/_images_common.py:build_images_response
        """
        response = openai_client.images.create_variation(
            image=sample_image_file,
            model=model_id,
            size=NOVA_CANVAS_SIZE,
            response_format="b64_json",
        )

        assert response.created > 0
        assert response.size is not None
        assert response.data is not None
        assert len(response.data) == 1
        img = response.data[0]
        assert img.b64_json is not None
        assert img.url is None

        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.input_tokens_details.image_tokens > 0
        assert response.usage.input_tokens_details.text_tokens == 0
        assert response.usage.output_tokens == 1
        assert (
            response.usage.total_tokens
            == response.usage.input_tokens + response.usage.output_tokens
        )

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_variation_with_text_image_task_type(
        self, openai_client: OpenAI, sample_image_file: bytes, model_id: str
    ) -> None:
        """``taskType=TEXT_IMAGE`` uses the upload as a conditioning image.

        Ref: stdapi/models/image/amazon_nova_canvas.py:_get_request_text_image
        """
        response = openai_client.images.create_variation(
            image=sample_image_file,
            model=model_id,
            size=NOVA_CANVAS_SIZE,
            n=1,
            response_format="b64_json",
            extra_body={"taskType": "TEXT_IMAGE"},
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.size == NOVA_CANVAS_SIZE

    @pytest.mark.expensive
    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_variation_with_color_guided_task_type(
        self, openai_client: OpenAI, sample_image_file: bytes, model_id: str
    ) -> None:
        """``taskType=COLOR_GUIDED_GENERATION`` uses the upload as the reference image.

        The requested colors are forwarded as
        ``colorGuidedGenerationParams.colors`` and the missing prompt is replaced
        by the gateway's default variation prompt.

        Ref: stdapi/models/image/amazon_nova_canvas.py:_get_request_color_guided_generation
        """
        response = openai_client.images.create_variation(
            image=sample_image_file,
            model=model_id,
            size=NOVA_CANVAS_SIZE,
            n=1,
            response_format="b64_json",
            extra_body={
                "taskType": "COLOR_GUIDED_GENERATION",
                "colorGuidedGenerationParams": {
                    "colors": ["#FF5733", "#33FF57", "#3357FF"]
                },
            },
        )

        assert response.created > 0
        assert response.data is not None
        assert len(response.data) == 1
        assert response.data[0].b64_json is not None
        assert response.size == NOVA_CANVAS_SIZE

    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_variation_with_invalid_task_type(
        self, openai_client: OpenAI, sample_image_file: bytes, model_id: str
    ) -> None:
        """A ``taskType`` outside the route's allow-list is a 400 listing the legal values.

        Ref: stdapi/models/image/amazon_nova_canvas.py:_ImageGenerationJob._create_image_variations
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.create_variation(
                image=sample_image_file,
                model=model_id,
                size=NOVA_CANVAS_SIZE,
                extra_body={"taskType": "INVALID_TASK_TYPE"},
            )

        body = exc_info.value.body
        assert isinstance(body, dict)
        assert exc_info.value.status_code == 400
        assert body["type"] == "invalid_request_error"
        assert body["message"] == (
            '"taskType" value must be "IMAGE_VARIATION", "TEXT_IMAGE" '
            'or "COLOR_GUIDED_GENERATION".'
        )

    @pytest.mark.parametrize("model_id", NOVA_CANVAS_SAMPLE)
    def test_variation_color_guided_missing_colors(
        self, openai_client: OpenAI, sample_image_file: bytes, model_id: str
    ) -> None:
        """``COLOR_GUIDED_GENERATION`` without ``colors`` is a 400 naming the missing key.

        Nova Canvas requires the color list; the gateway reports it before
        invoking the model.

        Ref: stdapi/models/image/amazon_nova_canvas.py:_get_request_color_guided_generation
        """
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.images.create_variation(
                image=sample_image_file,
                model=model_id,
                size=NOVA_CANVAS_SIZE,
                extra_body={
                    "taskType": "COLOR_GUIDED_GENERATION",
                    "colorGuidedGenerationParams": {},
                },
            )

        body = exc_info.value.body
        assert isinstance(body, dict)
        assert exc_info.value.status_code == 400
        assert "colorGuidedGenerationParams.colors" in body["message"]
