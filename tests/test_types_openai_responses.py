"""OpenAI Responses type mirror against the OpenAI SDK types (no AWS calls).

A ``configuration_update`` item is written into conversation history by the
Responses API whenever a client changes reasoning effort mid-conversation, so it
arrives here through the ordinary stateless loop of resending the previous items
verbatim. It is one arm of the ``input`` item union, so an unmodelled item type
fails every arm and the whole request is refused instead of the item alone.

Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
     https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_configuration_update_item.py
     stdapi/types/openai_responses.py:ConfigurationUpdateInput
     stdapi/types/openai_responses.py:ResponseConfigurationUpdateItem
"""

from typing import Any, Required, get_args, get_origin, get_type_hints

import pytest
from openai.types.responses import (
    ResponseConfigurationUpdateItem as SdkResponseConfigurationUpdateItem,
)
from openai.types.responses import (
    ResponseConfigurationUpdateItemParamParam as SdkConfigurationUpdateParam,
)
from openai.types.shared.reasoning_effort import ReasoningEffort as SdkReasoningEffort
from pydantic import TypeAdapter, ValidationError

from stdapi.types.openai_responses import (
    ConfigurationUpdateInput,
    ReasoningEffort,
    ResponseConfigurationUpdateItem,
    ResponseCreateParams,
    ResponseItem,
)

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Adapter over the union backing conversation and response item listings.
RESPONSE_ITEM_ADAPTER: TypeAdapter[ResponseItem] = TypeAdapter[ResponseItem](
    ResponseItem
)

#: Every reasoning effort level the installed OpenAI SDK declares.
# The SDK alias is Optional[Literal[...]], so the levels sit one level down.
SDK_REASONING_EFFORTS: tuple[Any, ...] = tuple(
    effort for member in get_args(SdkReasoningEffort) for effort in get_args(member)
)


def _sdk_param_keys() -> dict[str, bool]:
    """Read the SDK input-item TypedDict as a field name to required-ness map.

    ``__required_keys__`` is empty for these generated TypedDicts, so the
    annotations are what says which fields upstream demands.

    Returns:
        Each declared field name mapped to whether upstream requires it.
    """
    hints = get_type_hints(SdkConfigurationUpdateParam, include_extras=True)
    return {name: get_origin(hint) is Required for name, hint in hints.items()}


class TestConfigurationUpdateInput:
    """A replayed ``configuration_update`` item on POST /v1/responses.

    Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/types/openai_responses.py:ConfigurationUpdateInput
    """

    def test_a_replayed_item_does_not_refuse_the_request(self) -> None:
        """A request whose input carries the item validates, item included.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/types/openai_responses.py:ConfigurationUpdateInput
        """
        params = ResponseCreateParams.model_validate(
            {
                "model": "m",
                "input": [
                    {"role": "user", "content": "Hello"},
                    {
                        "type": "configuration_update",
                        "id": "cfg_1",
                        "reasoning": {"effort": "low"},
                    },
                ],
            }
        )

        assert isinstance(params.input, list)
        item = params.input[1]
        assert isinstance(item, ConfigurationUpdateInput)
        assert item.id == "cfg_1"
        assert item.reasoning is not None
        assert item.reasoning.effort == "low"

    def test_the_item_never_sets_the_request_reasoning_effort(self) -> None:
        """A replayed effort does not become the effort of the request.

        The gateway is stateless per request: the request-level ``reasoning``
        parameter is the only thing that configures a turn, so a stale history
        item must not override what the caller asked for now.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/types/openai_responses.py:ConfigurationUpdateInput
        """
        params = ResponseCreateParams.model_validate(
            {
                "model": "m",
                "input": [
                    {"type": "configuration_update", "reasoning": {"effort": "high"}}
                ],
                "reasoning": {"effort": "minimal"},
            }
        )

        assert params.reasoning is not None
        assert params.reasoning.effort == "minimal"

    def test_only_the_type_is_required(self) -> None:
        """``id`` and ``reasoning`` are optional, as they are upstream.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/types/openai_responses.py:ConfigurationUpdateInput
        """
        item = ConfigurationUpdateInput.model_validate({"type": "configuration_update"})

        assert item.id is None
        assert item.reasoning is None

    def test_a_reasoning_update_without_an_effort_is_accepted(self) -> None:
        """``reasoning`` carries an optional effort and nothing else upstream.

        Ref: https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_configuration_update_item_param_param.py
             stdapi/types/openai_responses.py:ConfigurationUpdateReasoningInput
        """
        item = ConfigurationUpdateInput.model_validate(
            {"type": "configuration_update", "reasoning": {}}
        )

        assert item.reasoning is not None
        assert item.reasoning.effort is None

    def test_the_declared_fields_match_the_sdk(self) -> None:
        """The mirrored item declares the SDK's fields, with its required set.

        Ref: https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_configuration_update_item_param_param.py
             stdapi/types/openai_responses.py:ConfigurationUpdateInput
        """
        sdk_fields = _sdk_param_keys()
        assert sdk_fields, "the SDK TypedDict declares no field to compare against"

        assert set(ConfigurationUpdateInput.model_fields) == set(sdk_fields)
        assert {
            name
            for name, field in ConfigurationUpdateInput.model_fields.items()
            if field.is_required()
        } == {name for name, required in sdk_fields.items() if required}

    def test_the_effort_levels_match_the_sdk(self) -> None:
        """The declared levels are exactly the SDK's, and there is a level to check.

        Kept out of the parametrized test below: an empty derivation would make
        pytest skip that one rather than fail it, so the comparison has to live
        in a body that always runs.

        Ref: https://github.com/openai/openai-python/blob/main/src/openai/types/shared/reasoning_effort.py
             stdapi/types/openai_responses.py:ReasoningEffort
        """
        assert SDK_REASONING_EFFORTS, "the SDK declares no effort level to compare"
        assert set(get_args(ReasoningEffort)) == set(SDK_REASONING_EFFORTS)

    @pytest.mark.parametrize("effort", SDK_REASONING_EFFORTS)
    def test_every_sdk_effort_level_is_accepted(self, effort: str) -> None:
        """Each effort the SDK can write into history validates here.

        Ref: https://github.com/openai/openai-python/blob/main/src/openai/types/shared/reasoning_effort.py
             stdapi/types/openai_responses.py:ReasoningEffort
        """
        item = ConfigurationUpdateInput.model_validate(
            {"type": "configuration_update", "reasoning": {"effort": effort}}
        )

        assert item.reasoning is not None
        assert item.reasoning.effort == effort


class TestResponseConfigurationUpdateItem:
    """A ``configuration_update`` item returned by an items listing.

    Ref: https://developers.openai.com/api/reference/resources/conversations/subresources/items/methods/list
         stdapi/types/openai_responses.py:ResponseConfigurationUpdateItem
    """

    def test_a_listed_item_validates_against_the_item_union(self) -> None:
        """The listing union accepts the item instead of dropping it.

        Ref: https://developers.openai.com/api/reference/resources/conversations/subresources/items/methods/list
             stdapi/types/openai_responses.py:ResponseItem
        """
        item = RESPONSE_ITEM_ADAPTER.validate_python(
            {
                "type": "configuration_update",
                "id": "cfg_1",
                "reasoning": {"effort": "medium"},
            }
        )

        assert isinstance(item, ResponseConfigurationUpdateItem)
        assert item.id == "cfg_1"
        assert item.reasoning is not None
        assert item.reasoning.effort == "medium"

    def test_the_listed_item_requires_an_id(self) -> None:
        """A listed item carries the identifier the conversation minted for it.

        Ref: https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_configuration_update_item.py
             stdapi/types/openai_responses.py:ResponseConfigurationUpdateItem
        """
        with pytest.raises(ValidationError):
            ResponseConfigurationUpdateItem.model_validate(
                {"type": "configuration_update"}
            )

    def test_the_declared_fields_match_the_sdk(self) -> None:
        """The mirrored item declares the SDK's fields, with its required set.

        Ref: https://github.com/openai/openai-python/blob/main/src/openai/types/responses/response_configuration_update_item.py
             stdapi/types/openai_responses.py:ResponseConfigurationUpdateItem
        """
        sdk_fields = SdkResponseConfigurationUpdateItem.model_fields
        assert sdk_fields

        assert set(ResponseConfigurationUpdateItem.model_fields) == set(sdk_fields)
        assert {
            name
            for name, field in ResponseConfigurationUpdateItem.model_fields.items()
            if field.is_required()
        } == {name for name, field in sdk_fields.items() if field.is_required()}
