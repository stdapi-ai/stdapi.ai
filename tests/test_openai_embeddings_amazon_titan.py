"""Tests for Amazon Titan embedding models on the OpenAI-compatible route.

Titan embeds one input per InvokeModel call and names its vector-width parameter
differently per family: ``dimensions`` for Titan Text V2, ``embeddingConfig``
for Titan Multimodal.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-mm.html
     stdapi/models/embedding/amazon_titan_embed.py:EmbeddingModel
"""

from math import hypot
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openai import OpenAI

TITAN_V1_TEXT = "amazon.titan-embed-text-v1"
TITAN_V1_IMAGE = "amazon.titan-embed-image-v1"
TITAN_V2_TEXT = "amazon.titan-embed-text-v2:0"

TITAN_TEXT_ALL = (TITAN_V1_TEXT, TITAN_V2_TEXT)
TITAN_TEXT_SAMPLE = (TITAN_V2_TEXT,)
TITAN_IMAGE_SAMPLE = (TITAN_V1_IMAGE,)

#: Default vector width of Titan Text Embeddings V2 and Titan Multimodal Embeddings G1.
_DEFAULT_DIMENSIONS = 1024

#: Fixed vector width of each Titan text embedding model with no ``dimensions`` set.
_TEXT_DIMENSIONS = {TITAN_V1_TEXT: 1536, TITAN_V2_TEXT: _DEFAULT_DIMENSIONS}


class TestAmazonTitanEmbeddings:
    """Text and image behavior of the Titan embedding families.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
         stdapi/models/embedding/amazon_titan_embed.py:EmbeddingModel.embed_text
    """

    @pytest.fixture(autouse=True)
    def _skip_on_official_api(self, use_official_api: bool) -> None:
        """Skip the whole class when the target is the official OpenAI API."""
        if use_official_api:
            pytest.skip(
                "Amazon Titan models are not available on the official OpenAI API"
            )

    @pytest.mark.parametrize("model_id", TITAN_TEXT_SAMPLE)
    def test_text_extra_params_normalize(
        self, openai_client: OpenAI, model_id: str
    ) -> None:
        """The Titan-only ``normalize`` body field is accepted and yields a unit vector.

        ``normalize`` has no OpenAI equivalent: the gateway merges unknown body
        fields into the native Titan request. With ``normalize=true`` (also Titan's
        default) the returned vector has L2 norm 1.

        Ref: https://stdapi.ai/api_openai_embeddings/
             stdapi/aws_bedrock.py:get_extra_model_parameters
        """
        response = openai_client.embeddings.create(
            model=model_id,
            input="Hello from Titan text embeddings.",
            extra_body={"normalize": True},
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert item.index == 0
        assert isinstance(item.embedding, list)
        assert len(item.embedding) == _DEFAULT_DIMENSIONS
        assert hypot(*item.embedding) == pytest.approx(1.0, abs=0.05), (
            "normalize=true did not return an L2-normalized vector"
        )

    @pytest.mark.parametrize("model_id", TITAN_TEXT_ALL)
    def test_text_single(self, openai_client: OpenAI, model_id: str) -> None:
        """Titan text models return their native vector width and bill input tokens.

        The width is per family and not negotiable without ``dimensions``: Titan Text
        Embeddings G1 always returns 1,536 values while V2 defaults to 1,024.  Both
        report ``inputTextTokenCount``, which the route surfaces as ``prompt_tokens``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
             stdapi/models/embedding/amazon_titan_embed.py:EmbeddingModel.embed_text
        """
        response = openai_client.embeddings.create(
            model=model_id, input="Hello from Titan text embeddings."
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert item.index == 0
        assert isinstance(item.embedding, list)
        assert len(item.embedding) == _TEXT_DIMENSIONS[model_id]
        assert any(x != 0.0 for x in item.embedding), "vector is all zeros"
        assert response.usage.prompt_tokens > 0, "no input tokens billed for text input"

    @pytest.mark.parametrize("model_id", TITAN_TEXT_SAMPLE)
    def test_text_dimensions(self, openai_client: OpenAI, model_id: str) -> None:
        """``dimensions=256`` is forwarded to Titan Text V2 and shortens the vector.

        Titan Text Embeddings V2 accepts only 1024 (default), 512 and 256; the
        gateway passes the OpenAI ``dimensions`` value straight through as the
        native ``dimensions`` field.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
             stdapi/models/embedding/amazon_titan_embed.py:EmbeddingModel.embed_text
        """
        dimensions = 256
        response = openai_client.embeddings.create(
            model=model_id, input="Dimensions parameter test.", dimensions=dimensions
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) == dimensions
        assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.parametrize("model_id", TITAN_IMAGE_SAMPLE)
    def test_image_single(
        self, openai_client: OpenAI, sample_image_file_base64: str, model_id: str
    ) -> None:
        """A PNG data URI embeds on Titan Multimodal G1 as a 1024-dimension vector.

        A data URI is not a valid OpenAI ``input``: the gateway detects the media
        type and sends it as the native ``inputImage`` field instead of
        ``inputText``. 1,024 is the model's default output vector size.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/titan-multiemb-models.html
             stdapi/models/embedding/amazon_titan_embed.py:EmbeddingModel._invoke
        """
        response = openai_client.embeddings.create(
            model=model_id, input=sample_image_file_base64
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert item.index == 0
        assert isinstance(item.embedding, list)
        assert len(item.embedding) == _DEFAULT_DIMENSIONS
        assert any(x != 0.0 for x in item.embedding), "vector is all zeros"
