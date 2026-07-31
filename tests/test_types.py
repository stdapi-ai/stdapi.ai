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

from typing import Any

import pytest

from stdapi.types import BaseModelRequestWithFormExtra

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
