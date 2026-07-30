"""Tests for TwelveLabs Marengo embeddings on the OpenAI-compatible route.

Marengo keys its native payload on ``inputType`` and nests the media object under
a matching key. Text and image go through InvokeModel (which requires an
inference-profile ID), while video and audio can only be embedded through
StartAsyncInvoke, whatever their size.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html
     stdapi/models/embedding/twelvelabs_marengo_embed.py:EmbeddingModel
"""

import pytest
from openai import BadRequestError, OpenAI

MARANGO_V2 = None  # "twelvelabs.marengo-embed-2-7-v1:0" # No more available
MARANGO_V3 = "twelvelabs.marengo-embed-3-0-v1:0"

MARANGO_ALL = (MARANGO_V3,)
MARANGO_SAMPLE = (MARANGO_V3,)

#: Lower bound on the Marengo vector width; the model has no ``dimensions`` option.
_MIN_DIMENSIONS = 256


class TestTwelveLabsMarengoEmbeddings:
    """Live behavior of the TwelveLabs Marengo embedding families.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
         stdapi/models/embedding/twelvelabs_marengo_embed.py:EmbeddingModel.embed_text
    """

    @pytest.mark.parametrize("model_id", MARANGO_ALL)
    def test_text_single(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """A text input returns a single vector through the synchronous path.

        Text is one of the two modalities Marengo exposes on InvokeModel, and the
        gateway resolves the model to its inference profile to make that call.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_build_request
        """
        if use_official_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
            )
        response = openai_client.embeddings.create(
            model=model_id, input="Hello from TwelveLabs Marengo embeddings."
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert item.index == 0
        assert isinstance(item.embedding, list)
        assert len(item.embedding) >= _MIN_DIMENSIONS
        assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.parametrize("model_id", MARANGO_ALL)
    def test_image_single(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """A PNG data URI is embedded as an ``image`` input, inlined as base64.

        The sample is below the Bedrock payload limit, so no S3 staging happens and
        the call stays synchronous.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_media_source
        """
        if use_official_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
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
        assert len(item.embedding) >= _MIN_DIMENSIONS
        assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.slow
    @pytest.mark.parametrize("model_id", MARANGO_ALL)
    def test_video_single(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_video_file_base64: str,
        model_id: str,
    ) -> None:
        """A video input is embedded through the asynchronous path.

        Marengo rejects video on InvokeModel regardless of size, so the gateway
        always routes video through StartAsyncInvoke; the clip may come back as
        several clip-scoped vectors.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_ASYNC_MEDIA_TYPES
        """
        if use_official_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
            )
        if not sample_video_file_base64:
            pytest.skip(
                "Missing video sample file. Skipping test. Add a MP4 file to 'tests/.cache/video.mp4'."
            )
        response = openai_client.embeddings.create(
            model=model_id, input=sample_video_file_base64
        )
        assert response.object == "list"
        assert len(response.data) >= 1
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) >= _MIN_DIMENSIONS
            assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.parametrize("model_id", [MARANGO_V2])
    def test_text_extra_params_text_truncate(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """The Marengo 2.7-only ``textTruncate`` body field is accepted for text input.

        ``textTruncate`` is a top-level field of the legacy 2.7 text request shape
        (3.0 nests text parameters under ``text``) and has no OpenAI equivalent, so
        the gateway forwards it as an extra body field.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_build_v2_request
        """
        if use_official_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
            )
        elif model_id is None:
            pytest.skip("Required TwelveLabs model is not available")
        response = openai_client.embeddings.create(
            model=model_id,
            input="Hello from TwelveLabs Marengo embeddings.",
            extra_body={"textTruncate": "end"},
        )
        assert response.object == "list"
        assert len(response.data) == 1
        item = response.data[0]
        assert item.object == "embedding"
        assert item.index == 0
        assert isinstance(item.embedding, list)
        assert len(item.embedding) >= _MIN_DIMENSIONS
        assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.parametrize("model_id", MARANGO_SAMPLE)
    def test_dimensions_unsupported_error(
        self, openai_client: OpenAI, use_official_api: bool, model_id: str
    ) -> None:
        """``dimensions`` is rejected as a 400 naming the unsupported option.

        Marengo has a fixed vector width and no dimension parameter in any of its
        request shapes, so the gateway refuses the request in its model layer before
        calling Bedrock rather than silently ignoring the value.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
             stdapi/models/embedding/twelvelabs_marengo_embed.py:EmbeddingModel.embed_text
        """
        if use_official_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
            )
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.embeddings.create(
                model=model_id, input="Dims not supported.", dimensions=128
            )

        error = exc_info.value
        assert error.status_code == 400
        error_body = error.body
        assert isinstance(error_body, dict)
        assert error_body["type"] == "invalid_request_error", error_body
        assert "dimensions" in error_body["message"], error_body
        assert "not supported" in error_body["message"], error_body

    @pytest.mark.slow
    @pytest.mark.parametrize("model_id", MARANGO_ALL)
    def test_force_s3_data_with_small_image(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """``force_s3_data`` stages a small image on S3, which forces the async path.

        Any S3-referenced input goes through StartAsyncInvoke, so this exercises the
        asynchronous branch with an input that would otherwise be inlined and
        embedded synchronously.

        Ref: https://stdapi.ai/api_openai_embeddings/
             stdapi/models/embedding/twelvelabs_marengo_embed.py:EmbeddingModel._embed
        """
        if use_official_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
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
        assert len(item.embedding) >= _MIN_DIMENSIONS
        assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.slow
    @pytest.mark.parametrize("model_id", MARANGO_ALL)
    def test_force_s3_data_with_video(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_video_file_base64: str,
        model_id: str,
    ) -> None:
        """``force_s3_data`` stages video on S3 and references it by URI.

        Video is asynchronous either way; the difference is that the payload is
        passed as ``mediaSource.s3Location`` instead of an inline base64 string.

        Ref: https://stdapi.ai/api_openai_embeddings/
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_media_source
        """
        if use_official_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
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
            assert len(item.embedding) >= _MIN_DIMENSIONS
            assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.slow
    @pytest.mark.parametrize("model_id", MARANGO_ALL)
    def test_force_s3_data_with_audio(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_audio_mp3_file_base64: str,
        model_id: str,
    ) -> None:
        """``force_s3_data`` stages audio on S3 and references it by URI.

        Ref: https://stdapi.ai/api_openai_embeddings/
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_media_source
        """
        if use_official_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
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
            assert len(item.embedding) >= _MIN_DIMENSIONS
            assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.slow
    @pytest.mark.parametrize("model_id", MARANGO_ALL)
    def test_force_s3_data_with_mixed_batch(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file_base64: str,
        sample_video_file_base64: str,
        model_id: str,
    ) -> None:
        """A three-item mixed batch keeps one call per input and is not fused.

        The text+image fusion only triggers on a two-item batch, so a text+image+video
        batch stays three independent embeddings (with the video possibly segmented),
        each staged on S3 by ``force_s3_data``.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_get_text_image_input
        """
        if use_official_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
            )
        if not sample_video_file_base64:
            pytest.skip(
                "Missing video sample file. Skipping test. Add a MP4 file to 'tests/.cache/video.mp4'."
            )

        inputs = [
            "Text description.",
            sample_image_file_base64,
            sample_video_file_base64,
        ]
        response = openai_client.embeddings.create(
            model=model_id, input=inputs, extra_body={"force_s3_data": True}
        )
        assert response.object == "list"
        # Video may return multiple segments
        assert len(response.data) >= len(inputs)
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) >= _MIN_DIMENSIONS
            assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.parametrize("model_id", [MARANGO_V3])
    def test_text_image_pair_v3(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """A text+image pair collapses into one ``text_image`` embedding on v3.

        Marengo 3.0 adds the ``text_image`` input type, so the gateway detects a
        two-item batch of exactly one text and one image and fuses it into a single
        vector instead of embedding the items separately.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_get_text_image_input
        """
        if use_official_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
            )

        # Test with text first, then image
        inputs = ["A beautiful sunset over the ocean.", sample_image_file_base64]
        response = openai_client.embeddings.create(model=model_id, input=inputs)

        assert response.object == "list"
        assert len(response.data) == 1  # Combined into single text_image embedding
        item = response.data[0]
        assert item.object == "embedding"
        assert item.index == 0
        assert isinstance(item.embedding, list)
        assert len(item.embedding) >= _MIN_DIMENSIONS
        assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.parametrize("model_id", [MARANGO_V2])
    def test_text_image_pair_not_combined_v2(
        self,
        openai_client: OpenAI,
        use_official_api: bool,
        sample_image_file_base64: str,
        model_id: str,
    ) -> None:
        """A text+image pair stays two embeddings on Marengo 2.7.

        2.7 has no ``text_image`` input type, so the fusion is skipped for ``-2-``
        model IDs and the two inputs are embedded independently.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_is_v2
        """
        if use_official_api:
            pytest.skip(
                "TwelveLabs models are not available on the official OpenAI API"
            )
        elif model_id is None:
            pytest.skip("Required TwelveLabs model is not available")

        inputs = ["A beautiful sunset over the ocean.", sample_image_file_base64]
        response = openai_client.embeddings.create(model=model_id, input=inputs)

        assert response.object == "list"
        assert len(response.data) == 2  # NOT combined in v2, returns 2 embeddings
        for i, item in enumerate(response.data):
            assert item.index == i
            assert item.object == "embedding"
            assert isinstance(item.embedding, list)
            assert len(item.embedding) >= _MIN_DIMENSIONS
            assert any(x != 0.0 for x in item.embedding), "vector is all zeros"
