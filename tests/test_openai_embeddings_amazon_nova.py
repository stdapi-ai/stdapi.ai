"""Tests for Amazon Nova multimodal embeddings on the OpenAI-compatible route.

Nova embeds text, images, audio and video into one shared vector space. The
gateway sends one ``SINGLE_EMBEDDING`` InvokeModel call per input, switching to
the asynchronous ``SEGMENTED_EMBEDDING`` job only for inputs above the per-media
sync cutoffs, and forwards the Nova-only ``embeddingPurpose`` from the body.

Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
     https://docs.aws.amazon.com/nova/latest/nova2-userguide/embeddings.html
     stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel
"""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from openai import OpenAI

NOVA_V1 = "amazon.nova-2-multimodal-embeddings-v1:0"

NOVA_ALL = (NOVA_V1,)
NOVA_SAMPLE = (NOVA_V1,)

#: The four ``embeddingDimension`` values Nova multimodal embeddings can return.
_SUPPORTED_DIMENSIONS = frozenset({256, 384, 1024, 3072})


class TestAmazonNovaEmbeddings:
    """Live behavior of Amazon Nova multimodal embeddings.

    Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
         stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel.embed_text
    """

    @pytest.mark.parametrize("model_id", NOVA_ALL)
    def test_text_single(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """A text input returns one vector of a documented Nova width.

        Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/embeddings.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel._embed_single
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )
        response = openai_client.embeddings.create(
            model=model_id, input="Hello from Amazon Nova multimodal embeddings."
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert item.index == 0
        assert isinstance(item.embedding, list)
        assert len(item.embedding) in _SUPPORTED_DIMENSIONS
        assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.parametrize("model_id", NOVA_ALL)
    def test_text_batch(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """A text batch returns one distinct vector per item, in request order.

        Nova has no batch request shape, so the gateway fans the array out into one
        InvokeModel call per input and re-assembles the results in input order.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel.embed_text
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        inputs = [
            "First text input for embedding.",
            "Second different sentence.",
            "Third entry to complete batch.",
        ]
        response = openai_client.embeddings.create(model=model_id, input=inputs)
        assert response.object == "list"
        assert len(response.data) == len(inputs)
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) in _SUPPORTED_DIMENSIONS

        vectors = [item.embedding for item in response.data]
        assert len({len(vector) for vector in vectors}) == 1, (
            "batch returned vectors of different widths"
        )
        assert len({tuple(vector) for vector in vectors}) == len(inputs), (
            "distinct inputs produced identical vectors"
        )

    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_dimensions(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """``dimensions`` becomes ``embeddingDimension`` and is honored exactly.

        Nova accepts 256, 384, 1024 and 3072; the OpenAI ``dimensions`` value is
        forwarded verbatim, so 256 must yield exactly 256 components.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel.embed_text
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        dimensions = 256
        response = openai_client.embeddings.create(
            model=model_id,
            input="Test sentence for dimensions parameter.",
            dimensions=dimensions,
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) == dimensions
        assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_extra_params_embedding_purpose(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """The Nova-only ``embeddingPurpose`` body field is accepted for a text input.

        ``embeddingPurpose`` has no OpenAI equivalent; the gateway pops it from the
        body and sends it as a top-level embedding parameter, defaulting to
        ``GENERIC_INDEX``. Nova defines nine purposes, of which ``CLASSIFICATION``
        is one — an out-of-enum value would be rejected by Bedrock.

        Ref: https://stdapi.ai/api_openai_embeddings/
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel.embed_text
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        response = openai_client.embeddings.create(
            model=model_id,
            input="Classification test sentence.",
            extra_body={"embeddingPurpose": "CLASSIFICATION"},
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert item.index == 0
        assert isinstance(item.embedding, list)
        assert len(item.embedding) in _SUPPORTED_DIMENSIONS
        assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.parametrize("model_id", NOVA_ALL)
    def test_image_single(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """A PNG data URI is embedded as an ``image`` input, not as text.

        A data URI is not a valid OpenAI embeddings input: the gateway sniffs the
        media type and builds Nova's ``image`` block with the detected format.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel._embed_single
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )
        response = openai_client.embeddings.create(
            model=model_id, input=sample_image_file_base64
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert item.index == 0
        assert isinstance(item.embedding, list)
        assert len(item.embedding) in _SUPPORTED_DIMENSIONS
        assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_image_batch(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """An image batch returns one same-width vector per image, in request order.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel.embed_text
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        inputs = [sample_image_file_base64, sample_image_file_base64]
        response = openai_client.embeddings.create(model=model_id, input=inputs)
        assert response.object == "list"
        assert len(response.data) == len(inputs)
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) in _SUPPORTED_DIMENSIONS
            assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

        assert len({len(item.embedding) for item in response.data}) == 1, (
            "batch returned vectors of different widths"
        )

    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_mixed_text_image_batch(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """A text+image batch returns one vector per input in the same vector space.

        Nova embeds every modality into one shared space, so mixed batches are not
        fused: each input keeps its own index and all vectors share a width.

        Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/embeddings.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel.embed_text
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        inputs = [
            "A sample text description.",
            sample_image_file_base64,
            "Another text input.",
        ]
        response = openai_client.embeddings.create(model=model_id, input=inputs)
        assert response.object == "list"
        assert len(response.data) == len(inputs)
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) in _SUPPORTED_DIMENSIONS
            assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

        assert len({len(item.embedding) for item in response.data}) == 1, (
            "text and image vectors are not in the same space"
        )
        assert len({tuple(item.embedding) for item in response.data}) == len(inputs), (
            "distinct inputs produced identical vectors"
        )

    @pytest.mark.parametrize("model_id", NOVA_ALL)
    def test_audio_single(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_audio_mp3_file_base64: str,
        model_id: str,
    ) -> None:
        """An MP3 data URI is embedded as an ``audio`` input.

        Nova covers up to 30 s of audio per synchronous embedding; the sample is
        inlined as base64 because it is below the Bedrock payload limit.

        Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/embeddings.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel._embed_single
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )
        response = openai_client.embeddings.create(
            model=model_id, input=sample_audio_mp3_file_base64
        )
        assert response.object == "list"
        assert len(response.data) >= 1  # May return multiple embeddings for segments
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) in _SUPPORTED_DIMENSIONS
            assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.slow
    @pytest.mark.parametrize("model_id", NOVA_ALL)
    def test_video_single(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_video_file_base64: str,
        model_id: str,
    ) -> None:
        """An MP4 data URI is embedded as a ``video`` input.

        Nova covers up to 30 s of video per synchronous embedding and defaults to
        ``AUDIO_VIDEO_COMBINED``, so a short clip yields a single combined vector
        rather than one per track.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:_DEFAULT_VIDEO_EMBEDDING_MODE
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )
        if not sample_video_file_base64:
            pytest.skip(
                "Missing video sample file. Skipping test. Add a MP4 file to 'tests/.cache/video.mp4'."
            )
        response = openai_client.embeddings.create(
            model=model_id, input=sample_video_file_base64
        )
        assert response.object == "list"
        assert len(response.data) >= 1  # May return multiple embeddings for segments
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) in _SUPPORTED_DIMENSIONS
            assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_multimodal_batch(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file_base64: str,
        sample_audio_mp3_file_base64: str,
        model_id: str,
    ) -> None:
        """A text+image+audio batch returns same-width vectors for every modality.

        The shared semantic space is what makes cross-modal retrieval possible, so
        every returned vector must have the same width regardless of input modality.

        Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/embeddings.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel.embed_text
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        inputs = [
            "Text description of the content.",
            sample_image_file_base64,
            sample_audio_mp3_file_base64,
        ]
        response = openai_client.embeddings.create(model=model_id, input=inputs)
        assert response.object == "list"
        # Audio may return multiple segments, so total may be >= len(inputs)
        assert len(response.data) >= len(inputs)
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) in _SUPPORTED_DIMENSIONS
            assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

        assert len({len(item.embedding) for item in response.data}) == 1, (
            "modalities returned vectors of different widths"
        )

    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_image_with_dimensions(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """``dimensions`` applies to image inputs exactly as it does to text.

        ``embeddingDimension`` is a top-level embedding parameter, not part of the
        per-media block, so it survives the media-type dispatch.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel._embed_single
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        dimensions = 256
        response = openai_client.embeddings.create(
            model=model_id, input=sample_image_file_base64, dimensions=dimensions
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert isinstance(item.embedding, list)
        assert len(item.embedding) == dimensions
        assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_clustering_purpose(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """``embeddingPurpose=CLUSTERING`` is accepted for a whole batch.

        The purpose is a request-level parameter, so it is repeated on each of the
        per-input InvokeModel calls the batch fans out into.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel.embed_text
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        inputs = [
            "First document for clustering.",
            "Second document for clustering.",
            "Third document for clustering.",
        ]
        response = openai_client.embeddings.create(
            model=model_id, input=inputs, extra_body={"embeddingPurpose": "CLUSTERING"}
        )
        assert response.object == "list"
        assert len(response.data) == len(inputs)
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) in _SUPPORTED_DIMENSIONS

        assert len({tuple(item.embedding) for item in response.data}) == len(inputs), (
            "distinct inputs produced identical vectors"
        )

    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_force_s3_data_with_small_image(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """``force_s3_data`` stages a small image on S3 instead of inlining it.

        The gateway normally inlines anything below the Bedrock payload limit;
        ``force_s3_data`` makes it upload the input and reference it through
        ``source.s3Location``. The call stays a synchronous ``SINGLE_EMBEDDING``
        because the image is far below the 50 MB image sync cutoff.

        Ref: https://stdapi.ai/api_openai_embeddings/
             stdapi/models/embedding/amazon_nova_embed.py:_SYNC_LIMIT_SIZES
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        response = openai_client.embeddings.create(
            model=model_id,
            input=sample_image_file_base64,
            extra_body={"force_s3_data": True},
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert item.index == 0
        assert isinstance(item.embedding, list)
        assert len(item.embedding) in _SUPPORTED_DIMENSIONS
        assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_force_s3_data_with_audio(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_audio_mp3_file_base64: str,
        model_id: str,
    ) -> None:
        """``force_s3_data`` stages audio on S3 and still embeds it synchronously.

        Only inputs above the 100 MB audio sync cutoff switch to the asynchronous
        segmented job, so a short sample uploaded to S3 comes back inline.

        Ref: https://stdapi.ai/api_openai_embeddings/
             stdapi/models/embedding/amazon_nova_embed.py:_SYNC_LIMIT_SIZES
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        response = openai_client.embeddings.create(
            model=model_id,
            input=sample_audio_mp3_file_base64,
            extra_body={"force_s3_data": True},
        )
        assert response.object == "list"
        assert len(response.data) >= 1
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) in _SUPPORTED_DIMENSIONS
            assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.slow
    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_force_s3_data_with_video(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_video_file_base64: str,
        model_id: str,
    ) -> None:
        """``force_s3_data`` stages video on S3 and still embeds it synchronously.

        Ref: https://stdapi.ai/api_openai_embeddings/
             stdapi/models/embedding/amazon_nova_embed.py:_SYNC_LIMIT_SIZES
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )
        if not sample_video_file_base64:
            pytest.skip(
                "Missing video sample file. Skipping test. Add a MP4 file to 'tests/.cache/video.mp4'."
            )

        response = openai_client.embeddings.create(
            model=model_id,
            input=sample_video_file_base64,
            extra_body={"force_s3_data": True},
        )
        assert response.object == "list"
        assert len(response.data) >= 1
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) in _SUPPORTED_DIMENSIONS
            assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.parametrize("model_id", NOVA_SAMPLE)
    def test_force_s3_data_with_multimodal_batch(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file_base64: str,
        sample_audio_mp3_file_base64: str,
        model_id: str,
    ) -> None:
        """``force_s3_data`` applies to every item of a mixed batch, text included.

        ``force_s3_data`` is popped once from the body and passed to each per-input
        call, so the plain string is uploaded as a text object too and referenced
        through ``text.source.s3Location``.

        Ref: https://stdapi.ai/api_openai_embeddings/
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel._embed
        """
        if use_official_api:
            pytest.skip(
                "Amazon Nova models are not available on the official OpenAI API"
            )

        inputs = [
            "Text description of the content.",
            sample_image_file_base64,
            sample_audio_mp3_file_base64,
        ]
        response = openai_client.embeddings.create(
            model=model_id, input=inputs, extra_body={"force_s3_data": True}
        )
        assert response.object == "list"
        # Audio may return multiple segments, so total may be >= len(inputs)
        assert len(response.data) >= len(inputs)
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) in _SUPPORTED_DIMENSIONS
            assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

        assert len({len(item.embedding) for item in response.data}) == 1, (
            "modalities returned vectors of different widths"
        )
