"""Tests for TwelveLabs Marengo embeddings on the OpenAI-compatible route.

Marengo keys its native payload on ``inputType`` and nests the media object under
a matching key. Text and image go through InvokeModel (which requires an
inference-profile ID), while video and audio can only be embedded through
StartAsyncInvoke, whatever their size.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html
     stdapi/models/embedding/twelvelabs_marengo_embed.py:EmbeddingModel
"""

from typing import Any

import pytest
from openai import BadRequestError, OpenAI

from tests._helpers import assert_embedding_list

#: Only Marengo generation currently reachable on Bedrock.
MARANGO_V3 = "twelvelabs.marengo-embed-3-0-v1:0"

#: Legacy 2.7 generation: no longer served, but its request shape is still built.
MARANGO_V2 = "twelvelabs.marengo-embed-2-7-v1:0"

#: Marengo model IDs the live tests run against.
MARANGO_MODELS = [MARANGO_V3]

#: Lower bound on the Marengo vector width; the model has no ``dimensions`` option.
_MIN_DIMENSIONS = 256


@pytest.mark.gateway("TwelveLabs models are not available on the official OpenAI API")
class TestTwelveLabsMarengoEmbeddings:
    """Live behavior of the TwelveLabs Marengo embedding families.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
         stdapi/models/embedding/twelvelabs_marengo_embed.py:EmbeddingModel.embed_text
    """

    @pytest.mark.parametrize("model_id", MARANGO_MODELS)
    def test_text_single(self, openai_client: OpenAI, model_id: str) -> None:
        """A text input returns a single vector through the synchronous path.

        Text is one of the two modalities Marengo exposes on InvokeModel, and the
        gateway resolves the model to its inference profile to make that call.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_build_request
        """
        response = openai_client.embeddings.create(
            model=model_id, input="Hello from TwelveLabs Marengo embeddings."
        )
        assert_embedding_list(response, count=1, min_dimensions=_MIN_DIMENSIONS)

    @pytest.mark.parametrize("model_id", MARANGO_MODELS)
    def test_image_single(
        self, openai_client: OpenAI, sample_image_file_base64: str, model_id: str
    ) -> None:
        """A PNG data URI is embedded as an ``image`` input, inlined as base64.

        The sample is below the Bedrock payload limit, so no S3 staging happens and
        the call stays synchronous.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_media_source
        """
        response = openai_client.embeddings.create(
            model=model_id, input=sample_image_file_base64
        )
        assert_embedding_list(response, count=1, min_dimensions=_MIN_DIMENSIONS)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_id", MARANGO_MODELS)
    def test_video_single(
        self, openai_client: OpenAI, sample_video_file_base64: str, model_id: str
    ) -> None:
        """A video input is embedded through the asynchronous path.

        Marengo rejects video on InvokeModel regardless of size, so the gateway
        always routes video through StartAsyncInvoke; the clip may come back as
        several clip-scoped vectors.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_ASYNC_MEDIA_TYPES
        """
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

    @pytest.mark.parametrize("model_id", MARANGO_MODELS)
    def test_dimensions_unsupported_error(
        self, openai_client: OpenAI, model_id: str
    ) -> None:
        """``dimensions`` is rejected as a 400 naming the unsupported option.

        Marengo has a fixed vector width and no dimension parameter in any of its
        request shapes, so the gateway refuses the request in its model layer before
        calling Bedrock rather than silently ignoring the value.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
             stdapi/models/embedding/twelvelabs_marengo_embed.py:EmbeddingModel.embed_text
        """
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
    @pytest.mark.parametrize("model_id", MARANGO_MODELS)
    def test_force_s3_data_with_small_image(
        self, openai_client: OpenAI, sample_image_file_base64: str, model_id: str
    ) -> None:
        """``force_s3_data`` stages a small image on S3, which forces the async path.

        Any S3-referenced input goes through StartAsyncInvoke, so this exercises the
        asynchronous branch with an input that would otherwise be inlined and
        embedded synchronously.

        Ref: https://stdapi.ai/api_openai_embeddings/
             stdapi/models/embedding/twelvelabs_marengo_embed.py:EmbeddingModel._embed
        """
        response = openai_client.embeddings.create(
            model=model_id,
            input=sample_image_file_base64,
            extra_body={"force_s3_data": True},
        )
        assert_embedding_list(response, count=1, min_dimensions=_MIN_DIMENSIONS)

    @pytest.mark.slow
    @pytest.mark.parametrize("model_id", MARANGO_MODELS)
    def test_force_s3_data_with_video(
        self, openai_client: OpenAI, sample_video_file_base64: str, model_id: str
    ) -> None:
        """``force_s3_data`` stages video on S3 and references it by URI.

        Video is asynchronous either way; the difference is that the payload is
        passed as ``mediaSource.s3Location`` instead of an inline base64 string.

        Ref: https://stdapi.ai/api_openai_embeddings/
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_media_source
        """
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
    @pytest.mark.parametrize("model_id", MARANGO_MODELS)
    def test_force_s3_data_with_audio(
        self, openai_client: OpenAI, sample_audio_mp3_file_base64: str, model_id: str
    ) -> None:
        """``force_s3_data`` stages audio on S3 and references it by URI.

        Ref: https://stdapi.ai/api_openai_embeddings/
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_media_source
        """
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
    @pytest.mark.parametrize("model_id", MARANGO_MODELS)
    def test_force_s3_data_with_mixed_batch(
        self,
        openai_client: OpenAI,
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
        self, openai_client: OpenAI, sample_image_file_base64: str, model_id: str
    ) -> None:
        """A text+image pair collapses into one ``text_image`` embedding on v3.

        Marengo 3.0 adds the ``text_image`` input type, so the gateway detects a
        two-item batch of exactly one text and one image and fuses it into a single
        vector instead of embedding the items separately.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_get_text_image_input
        """
        # Test with text first, then image
        inputs = ["A beautiful sunset over the ocean.", sample_image_file_base64]
        response = openai_client.embeddings.create(model=model_id, input=inputs)

        # Combined into a single text_image embedding.
        assert_embedding_list(response, count=1, min_dimensions=_MIN_DIMENSIONS)


@pytest.mark.local
class TestMarengoRequestShapes:
    """Offline unit tests: the native request bodies the gateway builds.

    ``_build_request`` / ``_build_v2_request`` are pure, so they are called
    directly instead of through Bedrock. The 2.7 shape is still built for any
    model ID containing ``-2-`` even though no such model is currently served.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html
         https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
         stdapi/models/embedding/twelvelabs_marengo_embed.py:EmbeddingModel
    """

    @staticmethod
    def _model(model_id: str) -> Any:  # noqa: ANN401
        """Build a model instance without touching the AWS-backed model registry."""
        from stdapi.models.embedding.twelvelabs_marengo_embed import (  # noqa: PLC0415
            EmbeddingModel,
        )

        return EmbeddingModel(model_id)

    def test_v2_text_request_keeps_input_text_at_the_top_level(self) -> None:
        """The 2.7 text body carries ``inputText`` at the top level, not nested.

        3.0 moved text parameters under a ``text`` object; the legacy shape must
        stay flat or Bedrock rejects the body.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_build_v2_request
        """
        request = self._model(MARANGO_V2)._build_v2_request("text", "hello", {})  # noqa: SLF001

        assert request == {"inputType": "text", "inputText": "hello"}

    def test_v2_extra_params_are_merged_at_the_top_level(self) -> None:
        """A 2.7-only body field such as ``textTruncate`` is merged next to ``inputType``.

        The field has no OpenAI equivalent, so it can only reach the model as a
        forwarded extra body parameter.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_build_v2_request
        """
        request = self._model(MARANGO_V2)._build_v2_request(  # noqa: SLF001
            "text", "hello", {"textTruncate": "end"}
        )

        assert request == {
            "inputType": "text",
            "inputText": "hello",
            "textTruncate": "end",
        }

    def test_v3_text_request_nests_input_text_under_text(self) -> None:
        """The 3.0 text body nests ``inputText`` under the ``text`` payload key.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_build_request
        """
        request = self._model(MARANGO_V3)._build_request("text", "hello", {})  # noqa: SLF001

        assert request == {"inputType": "text", "text": {"inputText": "hello"}}

    def test_reserved_media_params_cannot_be_overridden(self) -> None:
        """Client extras never rewrite ``inputType``, ``inputText`` or ``mediaSource``.

        Extras are forwarded verbatim to Bedrock, so the reserved set is the only
        thing stopping a caller from repointing ``mediaSource`` at an arbitrary
        S3 object the gateway's task role can read, or from substituting the
        text that actually gets embedded.

        Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo-3.html
             stdapi/models/embedding/twelvelabs_marengo_embed.py:_RESERVED_MEDIA_PARAMS
        """
        evil = {
            "inputType": "video",
            "inputText": "attacker text",
            "mediaSource": {"s3Location": {"uri": "s3://other-bucket/secret"}},
            "harmless": 1,
        }

        v3_request = self._model(MARANGO_V3)._build_request("text", "hello", evil)  # noqa: SLF001
        assert v3_request == {
            "inputType": "text",
            "text": {"inputText": "hello", "harmless": 1},
        }

        v2_request = self._model(MARANGO_V2)._build_v2_request("text", "hello", evil)  # noqa: SLF001
        assert v2_request == {"inputType": "text", "inputText": "hello", "harmless": 1}
