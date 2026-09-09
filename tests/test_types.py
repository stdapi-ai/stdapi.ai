"""Form bracket-notation parsing in ``BaseModelRequestWithFormExtra``.

``multipart/form-data`` has no nested types, so the Images routes accept
PHP/HTML-style bracket keys (``params[key][]``) and the model's ``before``
validator rebuilds the nested Bedrock payload from them. A string value is
JSON-decoded whenever its key is not a field declared on the model — whether
that key is plain or bracketed — so a multipart or JSON extra parameter reaches
``model_extra`` with the same type either way; a value under a declared field
name is always left untouched.

Ref: stdapi/types/__init__.py:BaseModelRequestWithFormExtra
     https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
"""

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from stdapi.types import (
    _MAX_FORM_BRACKET_DEPTH,
    _MAX_FORM_LIST_LENGTH,
    BaseModelRequestWithFormExtra,
)

if TYPE_CHECKING:
    from starlette.testclient import TestClient

pytestmark = pytest.mark.local


def _form_extra(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse a bracket-notation form payload and return the rebuilt ``model_extra``.

    Returns:
        The extra fields the ``before`` validator reconstructed.
    """
    model = BaseModelRequestWithFormExtra(**payload)
    assert model.model_extra is not None
    return model.model_extra


class TestBaseModelRequestWithFormExtra:
    """Bracket keys are expanded into nested dicts and lists under ``model_extra``.

    Ref: stdapi/types/__init__.py:BaseModelRequestWithFormExtra._deserialize_forms
    """

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            pytest.param({"key": "value"}, {"key": "value"}, id="plain-key"),
            pytest.param(
                {"param[key]": "value"}, {"param": {"key": "value"}}, id="nested-dict"
            ),
            pytest.param(
                {"param[]": "value1"}, {"param": ["value1"]}, id="array-append"
            ),
            pytest.param(
                {"param[0]": "value"}, {"param": ["value"]}, id="array-explicit-index"
            ),
            pytest.param(
                {"param[key][]": "value"},
                {"param": {"key": ["value"]}},
                id="array-inside-dict",
            ),
            pytest.param(
                {"param[a][b][c]": "value"},
                {"param": {"a": {"b": {"c": "value"}}}},
                id="deep-nesting",
            ),
            pytest.param(
                {"colorGuidedGenerationParams[colors][]": "#3357FF"},
                {"colorGuidedGenerationParams": {"colors": ["#3357FF"]}},
                id="array-append-at-nested-leaf",
            ),
            pytest.param(
                {"simple": "value1", "nested[key]": "value2", "array[]": "value3"},
                {"simple": "value1", "nested": {"key": "value2"}, "array": ["value3"]},
                id="mixed-plain-dict-and-list",
            ),
            pytest.param(
                {"items[0]": "first", "items[1]": "second", "items[2]": "third"},
                {"items": ["first", "second", "third"]},
                id="consecutive-numeric-indices",
            ),
            pytest.param(
                {"config[settings][0]": "option1", "config[settings][1]": "option2"},
                {"config": {"settings": ["option1", "option2"]}},
                id="indexed-leaves-under-dict-key",
            ),
            pytest.param(
                {
                    "root[level1][level2][level3]": "deep",
                    "root[level1][sibling]": "value",
                },
                {
                    "root": {
                        "level1": {"level2": {"level3": "deep"}, "sibling": "value"}
                    }
                },
                id="deep-branch-merges-with-shallow-sibling",
            ),
            pytest.param(
                {"a[b][c][d]": "value"},
                {"a": {"b": {"c": {"d": "value"}}}},
                id="three-bracket-levels",
            ),
            pytest.param(
                {"mixed[0][name]": "first", "mixed[1][name]": "second"},
                {"mixed": [{"name": "first"}, {"name": "second"}]},
                id="numeric-then-named-segment",
            ),
            pytest.param(
                {"param": "not-json-value"},
                {"param": "not-json-value"},
                id="plain-key-non-json-value-kept-as-is",
            ),
            pytest.param(
                {"count": "42"}, {"count": 42}, id="plain-key-json-value-decoded"
            ),
            pytest.param({}, {}, id="empty-payload"),
        ],
    )
    def test_bracket_key_expansion(
        self, payload: dict[str, Any], expected: dict[str, Any]
    ) -> None:
        """A bracket path is expanded into the nested dicts and lists it describes.

        A numeric or empty final segment yields a list; every other segment yields a
        dict. Keys carrying no bracket go through the same JSON-decode attempt as a
        bracket leaf, so a plain-key value that parses as JSON (e.g. ``"42"``) comes
        out decoded, and one that does not (e.g. ``"not-json-value"``) is kept as the
        original string.
        """
        assert _form_extra(payload) == expected

    def test_consecutive_brackets_nested_dict(self) -> None:
        """Two keys sharing a prefix merge into one dict instead of overwriting it.

        The Nova Canvas virtual-try-on payload sends ``maskType`` and
        ``imageBasedMask[maskImage]`` under the same parent.
        """
        extra = _form_extra(
            {
                "virtualTryOnParams[maskType]": "IMAGE",
                "virtualTryOnParams[imageBasedMask][maskImage]": "/9j/4A",
            }
        )
        assert extra == {
            "virtualTryOnParams": {
                "maskType": "IMAGE",
                "imageBasedMask": {"maskImage": "/9j/4A"},
            }
        }

    def test_multiple_array_appends(self) -> None:
        """A single ``items[]`` field still yields a list, not a scalar.

        Repeated form fields with the same name cannot be expressed as Python
        keyword arguments, so only the one-value case is reachable here; the
        multi-value case is handled by the web framework before this validator.
        """
        extra = _form_extra({"items[]": "value2"})
        assert extra == {"items": ["value2"]}

    def test_sparse_array(self) -> None:
        """A gap between explicit indices is padded with ``None`` up to the highest one.

        The index is honoured as a position rather than an append order, so
        ``items[5]`` lands at index 5 and the list keeps a stable length of 6.
        """
        extra = _form_extra({"items[0]": "first", "items[5]": "sixth"})
        assert extra == {"items": ["first", None, None, None, None, "sixth"]}

    def test_json_string_deserialization(self) -> None:
        """Bracket values are JSON-decoded, so numbers, booleans and null keep their type.

        Form fields are always strings on the wire, but Bedrock model parameters
        are typed, so ``"123"`` must reach the backend as the integer 123.
        """
        extra = _form_extra(
            {
                "param[number]": "123",
                "param[boolean]": "true",
                "param[null]": "null",
                "param[string]": '"text"',
                "param[array]": "[1,2,3]",
            }
        )
        assert extra == {
            "param": {
                "number": 123,
                "boolean": True,
                "null": None,
                "string": "text",
                "array": [1, 2, 3],
            }
        }

    @pytest.mark.xfail(
        reason="Pattern outer[][inner] cannot be handled: [] means append but we need to "
        "navigate into the appended item. This would require lookahead to determine if "
        "an appended item needs to be a dict or list, significantly complicating the "
        "implementation for a rare edge case.",
        strict=True,
    )
    def test_empty_bracket_in_middle_position(self) -> None:
        """An empty bracket in a non-final position is not supported (documented xfail).

        ``outer[][inner]`` would have to append a dict and then descend into it,
        which needs lookahead the parser deliberately does not implement; no
        Bedrock image parameter uses that shape.

        Ref: stdapi/types/__init__.py:_navigate_bracket_part
        """
        extra = _form_extra({"outer[][inner]": "value"})
        assert extra == {"outer": [{"inner": "value"}]}

    def test_real_world_virtual_tryon_params(self) -> None:
        """A full VIRTUAL_TRY_ON form payload rebuilds its nested Nova Canvas params.

        The declared OpenAI image fields travel alongside the bracket keys, so
        this checks the nested reconstruction without disturbing the flat ones.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
        """
        extra = _form_extra(
            {
                "prompt": "ignored",
                "model": "amazon.nova-canvas-v1:0",
                "response_format": "b64_json",
                "n": 1,
                "size": "1024x1024",
                "user": None,
                "background": "auto",
                "input_fidelity": "low",
                "output_compression": 100,
                "output_format": None,
                "partial_images": None,
                "quality": "auto",
                "stream": False,
                "taskType": "VIRTUAL_TRY_ON",
                "virtualTryOnParams[maskType]": "IMAGE",
                "virtualTryOnParams[imageBasedMask][maskImage]": "/9j/4A",
            }
        )
        assert extra["virtualTryOnParams"] == {
            "maskType": "IMAGE",
            "imageBasedMask": {"maskImage": "/9j/4A"},
        }
        # Bracket-free fields keep their original name, value and Python type.
        assert extra["taskType"] == "VIRTUAL_TRY_ON"
        assert extra["size"] == "1024x1024"
        assert extra["n"] == 1
        assert extra["user"] is None

    def test_real_world_color_guided_generation(self) -> None:
        """A full COLOR_GUIDED_GENERATION form payload rebuilds its nested colour list.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
        """
        extra = _form_extra(
            {
                "model": "amazon.titan-image-generator-v2:0",
                "response_format": "b64_json",
                "n": 1,
                "size": "512x512",
                "user": None,
                "taskType": "COLOR_GUIDED_GENERATION",
                "colorGuidedGenerationParams[colors][]": "#3357FF",
            }
        )
        assert extra["colorGuidedGenerationParams"] == {"colors": ["#3357FF"]}
        assert extra["taskType"] == "COLOR_GUIDED_GENERATION"

    def test_array_of_objects(self) -> None:
        """Indexed object keys build a list of dicts, with each value JSON-decoded.

        ``items[0][value]="1"`` yields the integer 1: JSON decoding applies to
        every bracket leaf, including those inside an array of objects.
        """
        extra = _form_extra(
            {
                "items[0][name]": "first",
                "items[0][value]": "1",
                "items[1][name]": "second",
                "items[1][value]": "2",
            }
        )
        assert extra == {
            "items": [{"name": "first", "value": 1}, {"name": "second", "value": 2}]
        }


class TestFormBracketNameBounds:
    """A bracket key may only describe a structure of a bounded size.

    The field name is what a caller controls, and it is tiny next to the
    structure it asks for: an index positions a value, so an unbounded one turns
    a handful of characters into an arbitrarily long padded list, and an
    unbounded nesting depth multiplies that by one padded list per level. Both
    are bounded, and a name past either bound is refused rather than truncated —
    a silently shortened list would send the backend parameters the caller never
    asked for.

    Ref: stdapi/types/__init__.py:_bracket_index
    """

    @staticmethod
    def _key(depth: int) -> str:
        """Build a bracket key nesting *depth* segments in total.

        Returns:
            A key such as ``a[b][b]`` for a depth of three.
        """
        return "a" + "[b]" * (depth - 1)

    def test_last_supported_index_is_accepted(self) -> None:
        """The highest in-range index still positions its value.

        The list it produces is exactly the supported length, which pins the
        bound to a value rather than to "large".
        """
        extra = _form_extra({f"param[{_MAX_FORM_LIST_LENGTH - 1}]": "last"})
        assert len(extra["param"]) == _MAX_FORM_LIST_LENGTH
        assert extra["param"][-1] == "last"
        assert extra["param"][0] is None

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param(f"param[{_MAX_FORM_LIST_LENGTH}]", id="leaf-one-past-the-cap"),
            pytest.param("param[999999999]", id="leaf-far-past-the-cap"),
            pytest.param("param[999999999][name]", id="intermediate-container"),
            pytest.param("param[items][999999999]", id="leaf-under-a-dict-key"),
            pytest.param(f"param[{'9' * 5000}]", id="index-of-5000-digits"),
            pytest.param("param[²]", id="index-that-is-not-a-decimal-number"),
        ],
    )
    def test_index_past_the_supported_length_is_rejected(self, key: str) -> None:
        """An unusable index fails validation instead of padding a list to it.

        Both bracket paths are covered: the leaf that assigns the value, and the
        intermediate segment that creates the container below it. The last two
        cases are ones Python's own integer conversion refuses — a number too
        long to convert, and a digit that is not a decimal one — so the message
        the caller reads has to come from the bound rather than from the
        conversion.
        """
        with pytest.raises(ValidationError) as raised:
            _form_extra({key: "value"})
        message = str(raised.value)
        assert "bracket index" in message
        assert str(_MAX_FORM_LIST_LENGTH) in message
        # Python's own integer-conversion limit must never be what refuses this.
        assert "digits" not in message

    def test_deepest_supported_nesting_is_accepted(self) -> None:
        """A key nesting the maximum number of segments still builds its structure."""
        extra = _form_extra({self._key(_MAX_FORM_BRACKET_DEPTH): "deep"})
        current: Any = extra
        for segment in ("a", *["b"] * (_MAX_FORM_BRACKET_DEPTH - 1)):
            current = current[segment]
        assert current == "deep"

    def test_nesting_past_the_supported_depth_is_rejected(self) -> None:
        """One segment too many fails validation, whatever the segments contain.

        Depth is the second multiplier — a numeric segment pads a list of its
        own at every level, so bounding one index still leaves a long name
        asking for one padded list per level. The bound is on the name, so it
        holds for the plain segments used here too.
        """
        with pytest.raises(ValidationError) as raised:
            _form_extra({self._key(_MAX_FORM_BRACKET_DEPTH + 1): "value"})
        message = str(raised.value)
        assert "nest" in message
        assert str(_MAX_FORM_BRACKET_DEPTH) in message

    def test_rejected_bracket_key_answers_400(self, app_client: TestClient) -> None:
        """A route taking extra form parameters refuses the name with a client error.

        The refusal has to reach the caller as an invalid request, not as a
        server error: the request is malformed, and the caller can fix it.

        Ref: stdapi/routes/openai_videos.py:create_video
        """
        response = app_client.post(
            "/v1/videos",
            data={"model": "any-model", "prompt": "a", "param[999999999]": "1"},
        )
        assert response.status_code == 400
        error = response.json()["error"]
        assert error["type"] == "invalid_request_error"
        assert "bracket index" in error["message"]
