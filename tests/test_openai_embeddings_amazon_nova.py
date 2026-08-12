"""Tests for Amazon Nova multimodal embeddings on the OpenAI-compatible route.

Nova embeds text, images, audio and video into one shared vector space. The
gateway sends one ``SINGLE_EMBEDDING`` InvokeModel call per input, switching to
the asynchronous ``SEGMENTED_EMBEDDING`` job only for inputs above the per-media
sync cutoffs, and forwards the Nova-only ``embeddingPurpose`` from the body.

Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
     https://docs.aws.amazon.com/nova/latest/nova2-userguide/embeddings.html
     stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel
"""

from typing import TYPE_CHECKING, Any

import pytest

from tests._helpers import assert_embedding_list

if TYPE_CHECKING:
    from openai import OpenAI

#: Only Nova multimodal embedding model currently reachable on Bedrock.
NOVA_V1 = "amazon.nova-2-multimodal-embeddings-v1:0"

#: Nova model IDs the live tests run against.
NOVA_MODELS = [NOVA_V1]

#: The four ``embeddingDimension`` values Nova multimodal embeddings can return.
_SUPPORTED_DIMENSIONS = frozenset({256, 384, 1024, 3072})


@pytest.mark.gateway("Amazon Nova models are not available on the official OpenAI API")
class TestAmazonNovaEmbeddings:
    """Live behavior of Amazon Nova multimodal embeddings.

    Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
         stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel.embed_text
    """

    @pytest.mark.parametrize("model_id", NOVA_MODELS)
    def test_text_single(self, openai_client: OpenAI, model_id: str) -> None:
        """A text input returns one vector of a documented Nova width.

        Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/embeddings.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel._embed_single
        """
        response = openai_client.embeddings.create(
            model=model_id, input="Hello from Amazon Nova multimodal embeddings."
        )
        (vector,) = assert_embedding_list(response, count=1)
        assert len(vector) in _SUPPORTED_DIMENSIONS

    @pytest.mark.parametrize("model_id", NOVA_MODELS)
    def test_text_batch(self, openai_client: OpenAI, model_id: str) -> None:
        """A text batch returns one distinct vector per item, in request order.

        Nova has no batch request shape, so the gateway fans the array out into one
        InvokeModel call per input and re-assembles the results in input order.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel.embed_text
        """
        inputs = [
            "First text input for embedding.",
            "Second different sentence.",
            "Third entry to complete batch.",
        ]
        response = openai_client.embeddings.create(model=model_id, input=inputs)
        vectors = assert_embedding_list(response, count=len(inputs), nonzero=False)
        assert all(len(vector) in _SUPPORTED_DIMENSIONS for vector in vectors)
        assert len({tuple(vector) for vector in vectors}) == len(inputs), (
            "distinct inputs produced identical vectors"
        )

    @pytest.mark.parametrize("model_id", NOVA_MODELS)
    def test_dimensions(self, openai_client: OpenAI, model_id: str) -> None:
        """``dimensions`` becomes ``embeddingDimension`` and is honored exactly.

        Nova accepts 256, 384, 1024 and 3072; the OpenAI ``dimensions`` value is
        forwarded verbatim, so 256 must yield exactly 256 components.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel.embed_text
        """
        dimensions = 256
        response = openai_client.embeddings.create(
            model=model_id,
            input="Test sentence for dimensions parameter.",
            dimensions=dimensions,
        )
        assert_embedding_list(response, count=1, dimensions=dimensions)

    @pytest.mark.parametrize("model_id", NOVA_MODELS)
    def test_extra_params_embedding_purpose(
        self, openai_client: OpenAI, model_id: str
    ) -> None:
        """The Nova-only ``embeddingPurpose`` body field is accepted for a text input.

        ``embeddingPurpose`` has no OpenAI equivalent; the gateway pops it from the
        body and sends it as a top-level embedding parameter, defaulting to
        ``GENERIC_INDEX``. Nova defines nine purposes, of which ``CLASSIFICATION``
        is one — an out-of-enum value would be rejected by Bedrock.

        Ref: https://stdapi.ai/api_openai_embeddings/
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel.embed_text
        """
        response = openai_client.embeddings.create(
            model=model_id,
            input="Classification test sentence.",
            extra_body={"embeddingPurpose": "CLASSIFICATION"},
        )
        (vector,) = assert_embedding_list(response, count=1)
        assert len(vector) in _SUPPORTED_DIMENSIONS

    @pytest.mark.parametrize("model_id", NOVA_MODELS)
    def test_image_single(
        self, openai_client: OpenAI, sample_image_file_base64: str, model_id: str
    ) -> None:
        """A PNG data URI is embedded as an ``image`` input, not as text.

        A data URI is not a valid OpenAI embeddings input: the gateway sniffs the
        media type and builds Nova's ``image`` block with the detected format.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel._embed_single
        """
        response = openai_client.embeddings.create(
            model=model_id, input=sample_image_file_base64
        )
        (vector,) = assert_embedding_list(response, count=1)
        assert len(vector) in _SUPPORTED_DIMENSIONS

    @pytest.mark.parametrize("model_id", NOVA_MODELS)
    def test_image_batch(
        self, openai_client: OpenAI, sample_image_file_base64: str, model_id: str
    ) -> None:
        """An image batch returns one same-width vector per image, in request order.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel.embed_text
        """
        inputs = [sample_image_file_base64, sample_image_file_base64]
        response = openai_client.embeddings.create(model=model_id, input=inputs)
        vectors = assert_embedding_list(response, count=len(inputs))
        assert all(len(vector) in _SUPPORTED_DIMENSIONS for vector in vectors)

    @pytest.mark.parametrize("model_id", NOVA_MODELS)
    def test_mixed_text_image_batch(
        self, openai_client: OpenAI, sample_image_file_base64: str, model_id: str
    ) -> None:
        """A text+image batch returns one vector per input in the same vector space.

        Nova embeds every modality into one shared space, so mixed batches are not
        fused: each input keeps its own index and all vectors share a width.

        Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/embeddings.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel.embed_text
        """
        inputs = [
            "A sample text description.",
            sample_image_file_base64,
            "Another text input.",
        ]
        response = openai_client.embeddings.create(model=model_id, input=inputs)
        vectors = assert_embedding_list(response, count=len(inputs))
        assert all(len(vector) in _SUPPORTED_DIMENSIONS for vector in vectors)
        assert len({tuple(vector) for vector in vectors}) == len(inputs), (
            "distinct inputs produced identical vectors"
        )

    @pytest.mark.parametrize("model_id", NOVA_MODELS)
    def test_audio_single(
        self, openai_client: OpenAI, sample_audio_mp3_file_base64: str, model_id: str
    ) -> None:
        """An MP3 data URI is embedded as an ``audio`` input.

        Nova covers up to 30 s of audio per synchronous embedding; the sample is
        inlined as base64 because it is below the Bedrock payload limit.

        Ref: https://docs.aws.amazon.com/nova/latest/nova2-userguide/embeddings.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel._embed_single
        """
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
    @pytest.mark.parametrize("model_id", NOVA_MODELS)
    def test_video_single(
        self, openai_client: OpenAI, sample_video_file_base64: str, model_id: str
    ) -> None:
        """An MP4 data URI is embedded as a ``video`` input.

        Nova covers up to 30 s of video per synchronous embedding and defaults to
        ``AUDIO_VIDEO_COMBINED``, so a short clip yields a single combined vector
        rather than one per track.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:_DEFAULT_VIDEO_EMBEDDING_MODE
        """
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

    @pytest.mark.parametrize("model_id", NOVA_MODELS)
    def test_multimodal_batch(
        self,
        openai_client: OpenAI,
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

    @pytest.mark.parametrize("model_id", NOVA_MODELS)
    def test_image_with_dimensions(
        self, openai_client: OpenAI, sample_image_file_base64: str, model_id: str
    ) -> None:
        """``dimensions`` applies to image inputs exactly as it does to text.

        ``embeddingDimension`` is a top-level embedding parameter, not part of the
        per-media block, so it survives the media-type dispatch.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel._embed_single
        """
        dimensions = 256
        response = openai_client.embeddings.create(
            model=model_id, input=sample_image_file_base64, dimensions=dimensions
        )
        assert_embedding_list(response, count=1, dimensions=dimensions)

    @pytest.mark.parametrize("model_id", NOVA_MODELS)
    def test_clustering_purpose(self, openai_client: OpenAI, model_id: str) -> None:
        """``embeddingPurpose=CLUSTERING`` is accepted for a whole batch.

        The purpose is a request-level parameter, so it is repeated on each of the
        per-input InvokeModel calls the batch fans out into.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel.embed_text
        """
        inputs = [
            "First document for clustering.",
            "Second document for clustering.",
            "Third document for clustering.",
        ]
        response = openai_client.embeddings.create(
            model=model_id, input=inputs, extra_body={"embeddingPurpose": "CLUSTERING"}
        )
        vectors = assert_embedding_list(
            response, count=len(inputs), nonzero=False, uniform_width=False
        )
        assert all(len(vector) in _SUPPORTED_DIMENSIONS for vector in vectors)
        assert len({tuple(vector) for vector in vectors}) == len(inputs), (
            "distinct inputs produced identical vectors"
        )

    @pytest.mark.parametrize("model_id", NOVA_MODELS)
    def test_force_s3_data_with_small_image(
        self, openai_client: OpenAI, sample_image_file_base64: str, model_id: str
    ) -> None:
        """``force_s3_data`` stages a small image on S3 instead of inlining it.

        The gateway normally inlines anything below the Bedrock payload limit;
        ``force_s3_data`` makes it upload the input and reference it through
        ``source.s3Location``. The call stays a synchronous ``SINGLE_EMBEDDING``
        because the image is far below the 50 MB image sync cutoff.

        Ref: https://stdapi.ai/api_openai_embeddings/
             stdapi/models/embedding/amazon_nova_embed.py:_SYNC_LIMIT_SIZES
        """
        response = openai_client.embeddings.create(
            model=model_id,
            input=sample_image_file_base64,
            extra_body={"force_s3_data": True},
        )
        (vector,) = assert_embedding_list(response, count=1)
        assert len(vector) in _SUPPORTED_DIMENSIONS

    @pytest.mark.parametrize("model_id", NOVA_MODELS)
    def test_force_s3_data_with_audio(
        self, openai_client: OpenAI, sample_audio_mp3_file_base64: str, model_id: str
    ) -> None:
        """``force_s3_data`` stages audio on S3 and still embeds it synchronously.

        Only inputs above the 100 MB audio sync cutoff switch to the asynchronous
        segmented job, so a short sample uploaded to S3 comes back inline.

        Ref: https://stdapi.ai/api_openai_embeddings/
             stdapi/models/embedding/amazon_nova_embed.py:_SYNC_LIMIT_SIZES
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
            assert len(item.embedding) in _SUPPORTED_DIMENSIONS
            assert any(x != 0.0 for x in item.embedding), "vector is all zeros"

    @pytest.mark.slow
    @pytest.mark.parametrize("model_id", NOVA_MODELS)
    def test_force_s3_data_with_video(
        self, openai_client: OpenAI, sample_video_file_base64: str, model_id: str
    ) -> None:
        """``force_s3_data`` stages video on S3 and still embeds it synchronously.

        Ref: https://stdapi.ai/api_openai_embeddings/
             stdapi/models/embedding/amazon_nova_embed.py:_SYNC_LIMIT_SIZES
        """
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

    @pytest.mark.parametrize("model_id", NOVA_MODELS)
    def test_force_s3_data_with_multimodal_batch(
        self,
        openai_client: OpenAI,
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


@pytest.mark.local
class TestAmazonNovaSegmentedEmbedding:
    """Offline unit tests: the asynchronous ``SEGMENTED_EMBEDDING`` path.

    ``_embed`` routes here whenever an S3 input exceeds the per-media sync
    cutoff, so this is the whole large-media embedding path. Bedrock and S3 are
    stubbed; the request shaping, JSONL parsing and failure aggregation under
    test are the gateway's own.

    Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
         stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel._embed_segmented
    """

    @staticmethod
    def _stub_backend(
        monkeypatch: pytest.MonkeyPatch,
        embedding_results: list[dict[str, Any]],
        objects: dict[str, bytes],
    ) -> list[dict[str, Any]]:
        """Stub ``invoke_async``, the S3 fetch and the temporary-object tracker.

        Returns:
            The list the stub appends each ``invoke_async`` request body to.
        """
        from stdapi.models import InvokeResult  # noqa: PLC0415
        from stdapi.models.embedding import amazon_nova_embed  # noqa: PLC0415

        requests: list[dict[str, Any]] = []

        async def _invoke_async(
            _self: Any,  # noqa: ANN401
            request: dict[str, Any],
            **_kwargs: Any,  # noqa: ANN401
        ) -> InvokeResult[Any]:
            requests.append(request)
            return InvokeResult(
                response={"embeddingResults": embedding_results}, region="us-east-1"
            )

        class _StubBody:
            def __init__(self, data: bytes) -> None:
                self._data = data

            async def read(self) -> bytes:
                return self._data

        class _StubS3Client:
            async def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:  # noqa: N803
                return {"Body": _StubBody(objects[f"s3://{Bucket}/{Key}"])}

        monkeypatch.setattr(
            amazon_nova_embed.EmbeddingModel, "invoke_async", _invoke_async
        )
        monkeypatch.setattr(
            amazon_nova_embed, "get_client", lambda _service, _region: _StubS3Client()
        )
        monkeypatch.setattr(
            amazon_nova_embed,
            "track_temporary_s3_objects",
            lambda *_args, **_kwargs: None,
        )
        return requests

    async def test_segmented_text_job_parses_jsonl_in_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A segmented job asks for ``SEGMENTED_EMBEDDING`` and keeps the JSONL line order.

        Nova writes one JSONL line per segment into
        ``segmented-embedding-result.json``; the gateway concatenates the files
        result-by-result and the lines within each file, so vector *n* stays the
        embedding of segment *n*.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:_fetch_and_parse_embedding_jsonl
        """
        from stdapi.aws_s3 import S3Object  # noqa: PLC0415
        from stdapi.models.embedding.amazon_nova_embed import (  # noqa: PLC0415
            EmbeddingModel,
        )

        requests = self._stub_backend(
            monkeypatch,
            [
                {"status": "SUCCESS", "outputFileUri": "s3://out/a.jsonl"},
                {"status": "SUCCESS", "outputFileUri": "s3://out/b.jsonl"},
            ],
            {
                "s3://out/a.jsonl": (
                    b'{"embedding": [1.0, 0.0], "segmentMetadata": {}}\n'
                    b'{"embedding": [2.0, 0.0], "segmentMetadata": {}}\n'
                ),
                "s3://out/b.jsonl": b'{"embedding": [3.0, 0.0], "segmentMetadata": {}}',
            },
        )

        result = await EmbeddingModel(NOVA_V1)._embed_segmented(  # noqa: SLF001
            S3Object(bucket="in", key="big.txt"),
            "text",
            "txt",
            {"embeddingPurpose": "GENERIC_INDEX"},
            {},
        )

        assert [entry["embedding"] for entry in result.response["embeddings"]] == [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
        ]
        (request,) = requests
        assert request["taskType"] == "SEGMENTED_EMBEDDING"
        text_params = request["segmentedEmbeddingParams"]["text"]
        assert text_params["source"] == {"s3Location": {"uri": "s3://in/big.txt"}}
        assert "segmentationConfig" in text_params

    async def test_failed_segment_result_raises_a_400_naming_the_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-``SUCCESS`` result aborts the request with its ``failureReason``.

        Partially-failed jobs must not return silently truncated vectors: the
        failure is surfaced as a client error carrying Nova's own reason and
        message rather than a 500 or a short result set.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel._embed_segmented
        """
        from stdapi.api_errors import ApiError  # noqa: PLC0415
        from stdapi.aws_s3 import S3Object  # noqa: PLC0415
        from stdapi.models.embedding.amazon_nova_embed import (  # noqa: PLC0415
            EmbeddingModel,
        )

        self._stub_backend(
            monkeypatch,
            [
                {"status": "SUCCESS", "outputFileUri": "s3://out/a.jsonl"},
                {
                    "status": "FAILED",
                    "failureReason": "INVALID_CONTENT",
                    "message": "unreadable segment",
                },
            ],
            {"s3://out/a.jsonl": b'{"embedding": [1.0], "segmentMetadata": {}}'},
        )

        with pytest.raises(ApiError) as exc_info:
            await EmbeddingModel(NOVA_V1)._embed_segmented(  # noqa: SLF001
                S3Object(bucket="in", key="big.txt"),
                "text",
                "txt",
                {"embeddingPurpose": "GENERIC_INDEX"},
                {},
            )

        assert "INVALID_CONTENT" in str(exc_info.value)
        assert "unreadable segment" in str(exc_info.value)

    def test_reserved_media_params_cannot_be_overridden(self) -> None:
        """Client extras never rewrite ``source``, ``format`` or ``value``.

        Extras are forwarded verbatim into the media object, so the reserved set
        is the only thing stopping a caller from repointing ``source`` at an
        arbitrary S3 URI the gateway's task role can read, or from swapping the
        text that actually gets embedded.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
             stdapi/models/embedding/amazon_nova_embed.py:_RESERVED_MEDIA_PARAMS
        """
        from stdapi.models.embedding.amazon_nova_embed import (  # noqa: PLC0415
            EmbeddingModel,
        )

        params: dict[str, Any] = {
            "image": {"source": {"s3Location": {"uri": "s3://mine/ok.png"}}}
        }
        EmbeddingModel._add_extra_params(  # noqa: SLF001
            {
                "image": {
                    "source": {"s3Location": {"uri": "s3://other-bucket/secret"}},
                    "format": "gif",
                    "value": "attacker text",
                    "detailLevel": "DOCUMENT_IMAGE",
                }
            },
            "image",
            params,  # type: ignore[arg-type]
        )

        assert params == {
            "image": {
                "source": {"s3Location": {"uri": "s3://mine/ok.png"}},
                "detailLevel": "DOCUMENT_IMAGE",
            }
        }


@pytest.mark.local
class TestAmazonNovaStorageThreshold:
    """Media too big for the request body is embedded from storage instead.

    The body limit applies to the request Bedrock receives, and media travels in
    it base64-encoded — a third larger than the file itself. Weighing the file
    instead of its encoded form leaves a whole band of inputs inlined into a
    body the backend refuses.

    Ref: https://docs.aws.amazon.com/nova/latest/userguide/embeddings-schema.html
         stdapi/models/embedding/amazon_nova_embed.py:EmbeddingModel._embed
    """

    @staticmethod
    def _remote(declared_size: int) -> Any:  # noqa: ANN401
        """Build a remote input whose origin declares *declared_size* bytes.

        Returns:
            An ``InputFile`` that resolves its metadata without a request.
        """
        from stdapi.input_file import InputFile  # noqa: PLC0415

        value = InputFile("https://example.com/photo.png", content_type="image/png")
        source = value._source  # noqa: SLF001
        source._filename = None  # noqa: SLF001
        source._size = declared_size  # noqa: SLF001
        return value

    @staticmethod
    def _stub_transports(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
        """Stub the region choice, the upload and the invocation.

        Returns:
            The list the stubbed invocation appends the value it was given to.
        """
        from stdapi.aws_s3 import S3Object  # noqa: PLC0415
        from stdapi.input_file import InputFile  # noqa: PLC0415
        from stdapi.models.embedding.amazon_nova_embed import (  # noqa: PLC0415
            EmbeddingModel,
        )

        sent: list[Any] = []

        async def _select_region(_self: Any, **_kwargs: Any) -> str:  # noqa: ANN401
            return "us-east-1"

        async def _to_s3(_self: Any, **_kwargs: Any) -> S3Object:  # noqa: ANN401
            return S3Object(bucket="a-bucket", key="staged")

        async def _to_base64(_self: Any) -> str:  # noqa: ANN401
            return "AAAA"

        async def _embed_single(_self: Any, *, value: Any, **_kwargs: Any) -> Any:  # noqa: ANN401
            sent.append(value)

        monkeypatch.setattr(EmbeddingModel, "select_region", _select_region)
        monkeypatch.setattr(EmbeddingModel, "_embed_single", _embed_single)
        monkeypatch.setattr(InputFile, "to_s3", _to_s3)
        monkeypatch.setattr(InputFile, "to_base64", _to_base64)
        return sent

    async def _embed(self, monkeypatch: pytest.MonkeyPatch, declared_size: int) -> Any:  # noqa: ANN401
        """Run the transport decision over an input of *declared_size* bytes.

        Returns:
            The value handed to the invocation: a reference or inline base64.
        """
        from stdapi.models.embedding.amazon_nova_embed import (  # noqa: PLC0415
            EmbeddingModel,
        )

        sent = self._stub_transports(monkeypatch)
        await EmbeddingModel(NOVA_V1)._embed(  # noqa: SLF001
            self._remote(declared_size), {"embeddingPurpose": "GENERIC_INDEX"}, {}
        )
        (value,) = sent
        return value

    async def test_media_that_only_fits_undecoded_is_read_from_storage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file under the body limit whose base64 form is over it goes to storage."""
        from stdapi.aws_s3 import S3Object  # noqa: PLC0415

        # 20,000,000 bytes weigh 26,666,668 once base64-encoded.
        assert isinstance(await self._embed(monkeypatch, 20_000_000), S3Object)

    async def test_media_that_fits_encoded_travels_in_the_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file whose base64 form is still under the limit stays inline."""
        # 18,000,000 bytes weigh 24,000,000 once base64-encoded.
        assert await self._embed(monkeypatch, 18_000_000) == "AAAA"
