"""Bounded fan-out of the embedding models that invoke the backend once per input.

Titan, Nova multimodal and Marengo embed a single input per model call, so a
request carrying N inputs issues N invocations. That count is caller-controlled
-- the Vector Stores chunker routinely produces hundreds -- and the invocations
must therefore run under a fixed concurrency bound rather than all at once.

Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
     https://docs.aws.amazon.com/nova/latest/userguide/embeddings.html
     stdapi/models/embedding/__init__.py:EmbeddingModelBase._gather_bounded
"""

from asyncio import sleep
from typing import TYPE_CHECKING, Any

import pytest

from stdapi.models import InvokeResult
from stdapi.models.embedding import _EMBED_CONCURRENCY
from stdapi.models.embedding.amazon_nova_embed import EmbeddingModel as NovaModel
from stdapi.models.embedding.amazon_titan_embed import EmbeddingModel as TitanModel
from stdapi.models.embedding.twelvelabs_marengo_embed import (
    EmbeddingModel as MarengoModel,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from stdapi.models.embedding import EmbeddingModelBase

#: Inputs sent per request, well above the bound so a cap is observable.
_INPUT_COUNT = _EMBED_CONCURRENCY * 3

#: Event loop passes each fake invocation stays in flight; one is enough for every
#: unbounded sibling to have started, the rest are margin.
_YIELDS_PER_INVOCATION = 3


class _FanOutRecorder:
    """Stands in for one model invocation and records the concurrency it ran at."""

    def __init__(self, make_response: Callable[[list[float]], Any]) -> None:
        """Store the per-model response shape.

        Args:
            make_response: Wraps a vector into the response body of that model.
        """
        self._make_response = make_response
        self.active = 0
        self.peak = 0
        self.values: list[str] = []

    async def invoke(self, *args: object, **kwargs: object) -> InvokeResult[Any]:
        """Embed one input, staying in flight long enough for siblings to start.

        Args:
            args: Positional call arguments; the input value is the last one.
            kwargs: Keyword call arguments; the input value is ``value``.

        Returns:
            A response carrying the one vector identifying the embedded input.
        """
        value = str(kwargs["value"] if "value" in kwargs else args[-1])
        self.values.append(value)
        self.active += 1
        self.peak = max(self.peak, self.active)
        for _ in range(_YIELDS_PER_INVOCATION):
            await sleep(0)
        self.active -= 1
        return InvokeResult(
            response=self._make_response([float(value.rsplit("-", 1)[1])]),
            input_tokens=1,
            output_tokens=0,
        )


@pytest.mark.local
@pytest.mark.parametrize(
    ("model_class", "model_id", "invocation", "make_response"),
    [
        pytest.param(
            TitanModel,
            "amazon.titan-embed-text-v2:0",
            "_invoke",
            lambda vector: {"embedding": vector},
            id="amazon-titan",
        ),
        pytest.param(
            NovaModel,
            "amazon.nova-2-multimodal-embeddings-v1:0",
            "_embed",
            lambda vector: {"embeddings": [{"embedding": vector}]},
            id="amazon-nova",
        ),
        pytest.param(
            MarengoModel,
            "twelvelabs.marengo-embed-2-7-v1:0",
            "_embed",
            lambda vector: {"data": [{"embedding": vector}]},
            id="twelvelabs-marengo",
        ),
    ],
)
class TestEmbeddingFanOutBound:
    """Per-request concurrency bound of the one-input-per-call embedding models.

    Ref: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-embed-text.html
         stdapi/models/embedding/__init__.py:EmbeddingModelBase._gather_bounded
    """

    async def test_fan_out_never_exceeds_the_concurrency_bound(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model_class: type[EmbeddingModelBase[Any, Any]],
        model_id: str,
        invocation: str,
        make_response: Callable[[list[float]], Any],
    ) -> None:
        """A request carrying many inputs runs at most `_EMBED_CONCURRENCY` calls at once.

        Without the bound every input starts its own invocation before the first
        one completes, so the observed peak would be the whole input count.
        """
        recorder = _FanOutRecorder(make_response)
        monkeypatch.setattr(model_class, invocation, staticmethod(recorder.invoke))
        inputs: list[Any] = [f"input-{index}" for index in range(_INPUT_COUNT)]

        await model_class(model_id).embed_text(inputs, dimensions=None, extra_params={})

        assert len(recorder.values) == _INPUT_COUNT, "every input must be embedded"
        assert recorder.peak == _EMBED_CONCURRENCY, (
            f"{_INPUT_COUNT} inputs ran {recorder.peak} concurrent invocations; the "
            f"per-request bound is {_EMBED_CONCURRENCY}"
        )

    async def test_bounded_fan_out_keeps_results_in_input_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model_class: type[EmbeddingModelBase[Any, Any]],
        model_id: str,
        invocation: str,
        make_response: Callable[[list[float]], Any],
    ) -> None:
        """Vectors come back in input order even though calls complete in waves.

        The bound releases inputs in batches, so the response must be rebuilt
        from the awaited order rather than from the completion order.
        """
        recorder = _FanOutRecorder(make_response)
        monkeypatch.setattr(model_class, invocation, staticmethod(recorder.invoke))
        inputs: list[Any] = [f"input-{index}" for index in range(_INPUT_COUNT)]

        response = await model_class(model_id).embed_text(
            inputs, dimensions=None, extra_params={}
        )

        assert response.embeddings == [
            [float(index)] for index in range(_INPUT_COUNT)
        ], "embeddings must be ordered like the inputs that produced them"
