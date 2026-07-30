"""Model-ID rosters shared by the multi-model test modules.

A roster only belongs here when it is genuinely identical across the modules
that use it; per-surface rosters that merely look similar stay local to their
module, because merging them would silently change which models a route is
tested against.

Ref: tests/test_openai_chat_completions_multi_model.py
     tests/test_openai_responses_multi_model.py
"""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from _pytest.mark.structures import ParameterSet

#: Vision-capable models exercised on both OpenAI routes (Chat Completions and Responses).
VISION_MODELS_OPENAI = (
    "anthropic.claude-haiku-4-5-20251001-v1:0",  # Claude (reference)
    "amazon.nova-lite-v1:0",  # Amazon Nova
    "mistral.pixtral-large-2502-v1:0",  # Mistral Pixtral Large
    "qwen.qwen3-vl-235b-a22b",  # Qwen3 VL 235B
    "writer.palmyra-vision-7b",  # Writer Palmyra Vision 7B
)


def with_marks(
    models: Sequence[str], marks: Mapping[str, pytest.MarkDecorator]
) -> list[str | ParameterSet]:
    """Build parametrize params from *models*, applying per-model marks.

    Args:
        models: Plain model-id roster, in parametrize order.
        marks: Mapping of model id to the mark it must carry; models absent
            from the mapping are passed through unmarked.

    Returns:
        A list suitable as the second argument to ``pytest.mark.parametrize``.
    """
    return [
        pytest.param(model, marks=marks[model]) if model in marks else model
        for model in models
    ]
