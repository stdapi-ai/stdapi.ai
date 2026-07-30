"""Tests for Cohere Embed models on the OpenAI-compatible embeddings route.

Cohere requires an ``input_type`` that OpenAI has no equivalent for, so the
gateway always supplies ``search_document`` unless the caller overrides it, and it
picks between the ``texts``, ``images`` and (v4-only) fused ``inputs`` request
shapes from the kinds of input received.

Ref: https://docs.cohere.com/reference/embed
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v4.html
     stdapi/models/embedding/cohere_embed.py:EmbeddingModel
"""

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from stdapi.api_errors import ApiError
from stdapi.input_file import InputFileUrl
from stdapi.models import InvokeResult
from stdapi.models.embedding.cohere_embed import EmbeddingModel
from tests._helpers import assert_embedding_list

if TYPE_CHECKING:
    from openai import OpenAI

COHERE_V3 = "cohere.embed-english-v3"
COHERE_V4 = "cohere.embed-v4:0"

COHERE_ALL = (COHERE_V3, COHERE_V4)
COHERE_SAMPLE = (COHERE_V4,)

#: Minimal data URI, sufficient to exercise the image branch without a real image.
_SAMPLE_IMAGE_DATA_URI = "data:image/png;base64,AAAA"

#: Vector widths each model family can return (v3 is fixed, v4 is selectable).
_SUPPORTED_DIMENSIONS = {
    COHERE_V3: frozenset({1024}),
    COHERE_V4: frozenset({256, 512, 1024, 1536}),
}


@pytest.mark.gateway("Cohere models are not available on the official OpenAI API")
class TestCohereEmbeddings:
    """Live behavior of the Cohere Embed v3 and v4 families.

    Ref: https://docs.cohere.com/reference/embed
         stdapi/models/embedding/cohere_embed.py:EmbeddingModel.embed_text
    """

    @pytest.mark.parametrize("model_id", COHERE_SAMPLE)
    def test_text_extra_params_truncate(
        self, openai_client: OpenAI, model_id: str
    ) -> None:
        """The Cohere-only ``truncate`` body field is accepted alongside OpenAI fields.

        ``truncate`` has no OpenAI equivalent and its vocabulary changed between
        families: v3 takes ``START``/``END``, v4 takes ``LEFT``/``RIGHT``. The value
        only matters for over-long inputs, so a short input must simply succeed.

        Ref: https://stdapi.ai/api_openai_embeddings/
             stdapi/models/embedding/cohere_embed.py:_Request
        """
        truncate_value = "START" if model_id.endswith("v3") else "LEFT"
        response = openai_client.embeddings.create(
            model=model_id,
            input="The quick brown fox.",
            extra_body={"truncate": truncate_value},
        )
        (vector,) = assert_embedding_list(response, count=1)
        assert len(vector) in _SUPPORTED_DIMENSIONS[model_id]

    @pytest.mark.parametrize("model_id", COHERE_ALL)
    def test_text_single(self, openai_client: OpenAI, model_id: str) -> None:
        """A text input is sent as ``texts`` and returns one vector of the model's width.

        Ref: https://docs.cohere.com/reference/embed
             stdapi/models/embedding/cohere_embed.py:EmbeddingModel.embed_text
        """
        response = openai_client.embeddings.create(
            model=model_id, input="The quick brown fox jumps over the lazy dog."
        )
        (vector,) = assert_embedding_list(response, count=1)
        assert len(vector) in _SUPPORTED_DIMENSIONS[model_id]

    @pytest.mark.parametrize("model_id", COHERE_ALL)
    def test_image_single(
        self, openai_client: OpenAI, sample_image_file_base64: str, model_id: str
    ) -> None:
        """A PNG data URI is embedded through the ``images`` request field.

        Bedrock accepts a single image per Cohere Embed call, passed as a data URI;
        on v3 the gateway also has to switch ``input_type`` to ``image``, since v3
        rejects images under a text input type.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html
             stdapi/models/embedding/cohere_embed.py:EmbeddingModel.embed_text
        """
        response = openai_client.embeddings.create(
            model=model_id, input=sample_image_file_base64
        )
        (vector,) = assert_embedding_list(response, count=1)
        assert len(vector) in _SUPPORTED_DIMENSIONS[model_id]

    @pytest.mark.parametrize("model_id", COHERE_ALL)
    def test_text_batch(self, openai_client: OpenAI, model_id: str) -> None:
        """A text batch is one Cohere call returning one vector per input, in order.

        Unlike Titan, Cohere embeds the whole ``texts`` array in a single
        InvokeModel call, so the ordering guarantee comes from the provider.

        Ref: https://docs.cohere.com/reference/embed
             stdapi/models/embedding/cohere_embed.py:EmbeddingModel.embed_text
        """
        inputs = [
            "First text input for embedding.",
            "Second different sentence.",
            "Third entry to complete batch.",
        ]
        response = openai_client.embeddings.create(model=model_id, input=inputs)
        vectors = assert_embedding_list(
            response, count=len(inputs), nonzero=False, uniform_width=False
        )
        assert all(len(vector) in _SUPPORTED_DIMENSIONS[model_id] for vector in vectors)
        assert len({tuple(vector) for vector in vectors}) == len(inputs), (
            "distinct inputs produced identical vectors"
        )

    @pytest.mark.parametrize("model_id", COHERE_SAMPLE)
    def test_image_batch(
        self, openai_client: OpenAI, sample_image_file_base64: str, model_id: str
    ) -> None:
        """Several images in one call return one vector per image on Embed v4.

        Multi-image calls are v4-only: Bedrock limits Cohere Embed v3 to a single
        image per request.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v4.html
             stdapi/models/embedding/cohere_embed.py:EmbeddingModel.embed_text
        """
        inputs = [
            sample_image_file_base64,
            sample_image_file_base64,
            sample_image_file_base64,
        ]
        response = openai_client.embeddings.create(model=model_id, input=inputs)
        vectors = assert_embedding_list(
            response, count=len(inputs), uniform_width=False
        )
        assert all(len(vector) in _SUPPORTED_DIMENSIONS[model_id] for vector in vectors)

    @pytest.mark.parametrize("model_id", COHERE_SAMPLE)
    def test_mixed_text_image_batch(
        self, openai_client: OpenAI, sample_image_file_base64: str, model_id: str
    ) -> None:
        """A text+image batch on v4 is fused into ``inputs`` and stays one vector per item.

        Mixing modalities in one call is a v4-only capability; the gateway rewrites
        the batch into Cohere's ``inputs`` content-part shape rather than the
        mutually exclusive ``texts``/``images`` fields.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v4.html
             stdapi/models/embedding/cohere_embed.py:EmbeddingModel.embed_text
        """
        inputs = ["A sample image.", sample_image_file_base64]
        response = openai_client.embeddings.create(model=model_id, input=inputs)
        vectors = assert_embedding_list(
            response, count=len(inputs), uniform_width=False
        )
        assert all(len(vector) in _SUPPORTED_DIMENSIONS[model_id] for vector in vectors)

        assert vectors[0] != vectors[1], (
            "text and image inputs returned the same vector"
        )

    @pytest.mark.parametrize("model_id", COHERE_SAMPLE)
    def test_dimensions_supported_when_valid(
        self, openai_client: OpenAI, model_id: str
    ) -> None:
        """``dimensions`` maps to Cohere's ``output_dimension`` and is honored exactly.

        Embed v4 accepts 256, 512, 1024 and 1536, so a request for 512 must come
        back with exactly 512 components rather than the model's default width.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v4.html
             stdapi/models/embedding/cohere_embed.py:EmbeddingModel.embed_text
        """
        dimensions = 512
        response = openai_client.embeddings.create(
            model=model_id,
            input="Test sentence for dimensions parameter.",
            dimensions=dimensions,
        )
        assert_embedding_list(response, count=1, dimensions=dimensions)


@pytest.mark.local
class TestCohereEmbedFusedInputsGuard:
    """Offline checks for mixed text+image handling in the Cohere model layer.

    Ref: https://docs.cohere.com/reference/embed
         stdapi/models/embedding/cohere_embed.py:EmbeddingModel.embed_text
    """

    async def test_mixed_input_on_v3_model_raises_clear_error(self) -> None:
        """A fused text+image request on a V3 model raises a 400 ApiError.

        Only Embed v4 has the ``inputs`` content-part shape, so the gateway refuses
        the mixed batch up front instead of letting Bedrock reject it.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html
             stdapi/models/embedding/cohere_embed.py:EmbeddingModel.embed_text
        """
        model = EmbeddingModel(COHERE_V3)
        model.invoke = AsyncMock()  # type: ignore[method-assign]
        with pytest.raises(ApiError, match="Cohere Embed v4") as exc_info:
            await model.embed_text(
                ["A sample text.", InputFileUrl(_SAMPLE_IMAGE_DATA_URI)],
                dimensions=None,
                extra_params={},
            )
        assert exc_info.value.status == 400
        model.invoke.assert_not_called()

    async def test_mixed_input_on_v4_model_builds_fused_body(self) -> None:
        """A fused text+image request on the V4 model builds the ``inputs`` body.

        Each input becomes its own ``content`` list entry, keeping request order, and
        the gateway supplies the ``input_type`` Cohere requires but OpenAI does not
        have. The mutually exclusive ``texts``/``images`` fields must stay absent.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v4.html
             stdapi/models/embedding/cohere_embed.py:EmbeddingModel._to_input_content
        """
        model = EmbeddingModel(COHERE_V4)
        model.invoke = AsyncMock(  # type: ignore[method-assign]
            return_value=InvokeResult(response={"embeddings": [[0.1]]})
        )
        response = await model.embed_text(
            ["A sample text.", InputFileUrl(_SAMPLE_IMAGE_DATA_URI)],
            dimensions=None,
            extra_params={},
        )
        request = model.invoke.call_args.args[0]
        assert request["inputs"] == [
            {"content": [{"type": "text", "text": "A sample text."}]},
            {
                "content": [
                    {"type": "image_url", "image_url": {"url": _SAMPLE_IMAGE_DATA_URI}}
                ]
            },
        ]
        assert request["input_type"] == "search_document"
        assert "texts" not in request
        assert "images" not in request
        assert response.embeddings == [[0.1]]
