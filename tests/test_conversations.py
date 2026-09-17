"""Conversation store helpers exercised in-process (no AWS calls).

The listing filter decides which stored items a conversation listing and a
response input-item listing can answer with. The store is append-only, so an
item it drops is unreachable through the API for good: a drop the listing was
built for stays silent, and a drop that means the server can no longer read
back something it accepted is reported to the operator.

The two item unions therefore have to agree. Whatever the input union accepts is
written verbatim, so the item union a listing answers with must be able to
express every one of those shapes, and that agreement is asserted over shapes
derived from the input union rather than over a hand-written sample.

Ref: stdapi/conversations.py:listable_items
     stdapi/types/openai_responses.py:ResponseInputItem
     stdapi/types/openai_responses.py:ResponseItem
     https://developers.openai.com/api/reference/resources/conversations/subresources/items/methods/list
"""

from types import UnionType
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import pytest
from pydantic import BaseModel, TypeAdapter

from stdapi.conversations import listable_items, stored_item
from stdapi.types.openai_responses import ResponseInputItem, ResponseItem

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: A stored item the listing can express and read back unchanged.
_READABLE_ITEM: dict[str, Any] = {
    "id": "msg_2",
    "type": "message",
    "role": "user",
    "status": "completed",
    "content": [{"type": "input_text", "text": "hello"}],
}

#: Body of the unreadable item below, which no report may ever quote.
_UNREADABLE_BODY: str = "conversation content that must stay out of the logs"

#: A stored item of a listed type carrying a content part no schema here declares.
# Minted through the writer so it is a real stored shape, then given a part type
# the read side cannot express: that is what schema drift leaves behind.
_UNREADABLE_ITEM: dict[str, Any] = stored_item(
    {
        "type": "message",
        "role": "user",
        "content": [{"type": "drifted_text", "text": _UNREADABLE_BODY}],
    }
)

#: Adapter over the union every item of a request's input is validated against.
_INPUT_ADAPTER: TypeAdapter[Any] = TypeAdapter[Any](ResponseInputItem)

#: Placeholder filled into a synthesized string field.
_PLACEHOLDER: str = "x"

#: Sample values per scalar annotation, the unconstrained ones covering both shapes.
_SCALAR_SAMPLES: dict[Any, list[Any]] = {
    bool: [True],
    int: [1],
    float: [1.0],
    str: [_PLACEHOLDER],
    object: [{"key": _PLACEHOLDER}, _PLACEHOLDER],
    Any: [{"key": _PLACEHOLDER}, _PLACEHOLDER],
}


def _type_literals(annotation: object) -> list[str]:
    """Collect the ``type`` discriminator values an annotation allows.

    Args:
        annotation: Annotation of a model's ``type`` field.

    Returns:
        Every string literal it accepts, which is empty when it has none.
    """
    literals = []
    for arg in get_args(annotation):
        if isinstance(arg, str):
            literals.append(arg)
        else:
            literals.extend(_type_literals(arg))
    return literals


def _field_samples(annotation: object, seen: frozenset[type]) -> list[Any]:
    """Build one value per distinct shape an annotation accepts.

    Args:
        annotation: The field annotation to synthesize values for.
        seen: Models already being synthesized, breaking recursive references.

    Returns:
        A value per union arm, ``None`` last so a filled shape comes first.
    """
    origin = get_origin(annotation)
    if origin is Annotated:
        return _field_samples(get_args(annotation)[0], seen)
    if origin is Literal:
        return [get_args(annotation)[0]]
    if origin in (Union, UnionType):
        arms = [arg for arg in get_args(annotation) if arg is not type(None)]
        samples = [sample for arm in arms for sample in _field_samples(arm, seen)]
        return samples if len(arms) == len(get_args(annotation)) else [*samples, None]
    if origin in (list, dict):
        args = get_args(annotation)
        inner = _field_samples(args[-1], seen) if args else [_PLACEHOLDER]
        return (
            [[sample] for sample in inner]
            if origin is list
            else [{"key": sample} for sample in inner]
        )
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return (
            [{}]
            if annotation in seen
            else list(_model_samples(annotation, seen | {annotation}))
        )
    return _SCALAR_SAMPLES.get(annotation, [_PLACEHOLDER])


def _model_samples(
    model: type[BaseModel], seen: frozenset[type] = frozenset()
) -> list[dict[str, Any]]:
    """Build the payloads covering every shape a model accepts.

    One payload fills every field with its first sample, then one further
    payload per remaining sample varies a single field, so the count stays
    linear in the number of fields instead of combinatorial.

    Args:
        model: The model to synthesize payloads for.
        seen: Models already being synthesized, breaking recursive references.

    Returns:
        The payloads, the fully filled one first.
    """
    by_field = {
        name: _field_samples(field.annotation, seen)
        for name, field in model.model_fields.items()
    }
    base = {
        name: samples[0]
        for name, samples in by_field.items()
        if samples and samples[0] is not None
    }
    return [
        base,
        *(
            {**base, name: sample}
            for name, samples in by_field.items()
            for sample in samples[1:]
            if sample is not None
        ),
    ]


def _input_union_cases() -> list[tuple[str, dict[str, Any]]]:
    """Derive from the input union every shape a listing has to read back.

    An input item type no listed item type expresses -- an ``item_reference``,
    a ``compaction_trigger`` -- is meant to be absent from a listing, so it is
    left out rather than asserted on.

    Returns:
        Each case as the declaring member's name and the payload to send.
    """
    listed = {
        literal
        for member in get_args(ResponseItem)
        for literal in _type_literals(member.model_fields["type"].annotation)
    }
    cases: list[tuple[str, dict[str, Any]]] = []
    for member in get_args(ResponseInputItem):
        field = member.model_fields.get("type")
        literals = _type_literals(field.annotation) if field is not None else []
        if not any(literal in listed for literal in literals):
            continue
        cases.extend((member.__name__, payload) for payload in _model_samples(member))
    return cases


class TestItemUnionsAgree:
    """Every shape the input union accepts survives a store round trip.

    Ref: stdapi/types/openai_responses.py:ResponseInputItem
         stdapi/types/openai_responses.py:ResponseItem
         stdapi/conversations.py:listable_items
    """

    @pytest.mark.parametrize(
        ("member", "payload"),
        _input_union_cases(),
        ids=lambda value: value if isinstance(value, str) else "",
    )
    def test_an_accepted_input_shape_is_read_back(
        self, member: str, payload: dict[str, Any], request_log: dict[str, Any]
    ) -> None:
        """A shape the writer stores is one the listing can answer with.

        The store is append-only, so a shape the input union accepts but the
        item union cannot express is written once and then dropped from every
        listing for good. Deriving the shapes from the input union makes a twin
        widened there without a matching read shape fail here.

        Ref: stdapi/conversations.py:listable_items
        """
        item = stored_item(
            _INPUT_ADAPTER.validate_python(payload).model_dump(
                mode="json", by_alias=True, exclude_none=True
            )
        )

        assert [listed["id"] for listed in listable_items([item])] == [item["id"]], (
            f"{member} is stored but no listing can read it back"
        )
        assert "error_detail" not in request_log


class TestListableItems:
    """The read-time filter over stored conversation items.

    Ref: stdapi/conversations.py:listable_items
    """

    def test_an_unreadable_item_of_a_listed_type_is_reported(
        self, request_log: dict[str, Any]
    ) -> None:
        """A drop the listing was not designed for reaches the request log.

        ``message`` is a type the listing expresses, so a stored ``message``
        failing validation is schema drift: the item was accepted on write and
        is now unreadable, which no operator can see unless it is logged.

        Ref: stdapi/conversations.py:listable_items
        """
        listable = listable_items([_UNREADABLE_ITEM, _READABLE_ITEM])

        assert [item["id"] for item in listable] == ["msg_2"]
        assert request_log["level"] == "warning"
        details = request_log["error_detail"]
        assert len(details) == 1
        assert _UNREADABLE_ITEM["id"] in details[0]
        assert "message" in details[0]

    def test_the_report_carries_no_conversation_content(
        self, request_log: dict[str, Any]
    ) -> None:
        """The dropped item is named by ID and type, never by its body.

        Ref: stdapi/conversations.py:listable_items
        """
        listable_items([_UNREADABLE_ITEM])

        assert _UNREADABLE_BODY not in request_log["error_detail"][0]

    def test_an_item_type_no_listing_expresses_is_dropped_quietly(
        self, request_log: dict[str, Any]
    ) -> None:
        """A ``compaction_trigger`` is meant to be absent, so it raises nothing.

        It configures a request instead of recording history: every listing of a
        conversation holding one would warn forever over working behaviour.

        Ref: https://developers.openai.com/api/reference/resources/responses/methods/create
             stdapi/conversations.py:listable_items
        """
        listable = listable_items(
            [{"id": "cmp_1", "type": "compaction_trigger"}, _READABLE_ITEM]
        )

        assert [item["id"] for item in listable] == ["msg_2"]
        assert request_log["level"] == "info"
        assert "error_detail" not in request_log

    def test_a_backfilled_item_is_kept_without_a_report(
        self, request_log: dict[str, Any]
    ) -> None:
        """A canonical shape missing a required field is listed, not reported.

        Ref: stdapi/conversations.py:listable_items
        """
        listable = listable_items(
            [
                {
                    "id": "fco_1",
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "42",
                }
            ]
        )

        assert listable == [
            {
                "id": "fco_1",
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "42",
                "status": "completed",
            }
        ]
        assert "error_detail" not in request_log

    def test_a_tool_search_call_sent_without_its_execution_is_kept(
        self, request_log: dict[str, Any]
    ) -> None:
        """Upstream makes ``execution`` optional to send and required to read back.

        A client may legitimately post a ``tool_search_call`` without it, so
        without a backfill the item is stored and then unreadable for the life of
        the conversation.

        Ref: openai.types.responses.response_tool_search_call.ResponseToolSearchCall
             openai.types.responses.response_tool_search_output_item_param.ResponseToolSearchOutputItemParam
             stdapi/conversations.py:listable_items
        """
        listable = listable_items(
            [{"id": "tsc_1", "type": "tool_search_call", "arguments": {"query": "c"}}]
        )

        assert [item["execution"] for item in listable] == ["client"]
        assert "error_detail" not in request_log
