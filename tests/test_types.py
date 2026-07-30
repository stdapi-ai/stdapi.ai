"""Form bracket-notation parsing in ``BaseModelRequestWithFormExtra``.

``multipart/form-data`` has no nested types, so the Images routes accept
PHP/HTML-style bracket keys (``params[key][]``) and the model's ``before``
validator rebuilds the nested Bedrock payload from them. Only values that arrived
under a bracket key are JSON-decoded; a plain key is copied through verbatim.

Ref: stdapi/types/__init__.py:BaseModelRequestWithFormExtra
     https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
"""

from typing import Any

import pytest


@pytest.fixture
def base_model_form_extra() -> type[Any]:
    """Import and return BaseModelRequestWithFormExtra only when tests run."""
    from stdapi.types import BaseModelRequestWithFormExtra  # noqa: PLC0415

    return BaseModelRequestWithFormExtra


class TestBaseModelRequestWithFormExtra:
    """Bracket keys are expanded into nested dicts and lists under ``model_extra``.

    Ref: stdapi/types/__init__.py:BaseModelRequestWithFormExtra._deserialize_forms
    """

    def test_simple_key(self, base_model_form_extra: type[Any]) -> None:
        """A key with no brackets is stored under its own name, unchanged."""
        model = base_model_form_extra(key="value")
        assert model.model_extra == {"key": "value"}

    def test_nested_dict(self, base_model_form_extra: type[Any]) -> None:
        """``param[key]`` becomes a one-entry dict under ``param``."""
        model = base_model_form_extra(**{"param[key]": "value"})
        assert model.model_extra == {"param": {"key": "value"}}

    def test_array_append(self, base_model_form_extra: type[Any]) -> None:
        """``param[]`` becomes a list, so a single value yields a one-element list."""
        model = base_model_form_extra(**{"param[]": "value1"})
        assert model.model_extra == {"param": ["value1"]}

    def test_array_explicit_index(self, base_model_form_extra: type[Any]) -> None:
        """``param[0]`` becomes a list, not a dict keyed by the string "0"."""
        model = base_model_form_extra(**{"param[0]": "value"})
        assert model.model_extra == {"param": ["value"]}

    def test_nested_dict_with_array(self, base_model_form_extra: type[Any]) -> None:
        """``param[key][]`` nests a list inside a dict."""
        model = base_model_form_extra(**{"param[key][]": "value"})
        assert model.model_extra == {"param": {"key": ["value"]}}

    def test_deep_nesting(self, base_model_form_extra: type[Any]) -> None:
        """Every intermediate bracket segment creates one more dict level."""
        model = base_model_form_extra(**{"param[a][b][c]": "value"})
        assert model.model_extra == {"param": {"a": {"b": {"c": "value"}}}}

    def test_consecutive_brackets_nested_dict(
        self, base_model_form_extra: type[Any]
    ) -> None:
        """Two keys sharing a prefix merge into one dict instead of overwriting it.

        The Nova Canvas virtual-try-on payload sends ``maskType`` and
        ``imageBasedMask[maskImage]`` under the same parent, which the original
        implementation flattened into a single overwritten branch.
        """
        model = base_model_form_extra(
            **{
                "virtualTryOnParams[maskType]": "IMAGE",
                "virtualTryOnParams[imageBasedMask][maskImage]": "/9j/4A",
            }
        )
        assert model.model_extra == {
            "virtualTryOnParams": {
                "maskType": "IMAGE",
                "imageBasedMask": {"maskImage": "/9j/4A"},
            }
        }

    def test_array_append_with_nested_key(
        self, base_model_form_extra: type[Any]
    ) -> None:
        """A trailing ``[]`` inside a nested path appends to a list at that leaf."""
        model = base_model_form_extra(
            **{"colorGuidedGenerationParams[colors][]": "#3357FF"}
        )
        assert model.model_extra == {
            "colorGuidedGenerationParams": {"colors": ["#3357FF"]}
        }

    def test_multiple_array_appends(self, base_model_form_extra: type[Any]) -> None:
        """A single ``items[]`` field still yields a list, not a scalar.

        Repeated form fields with the same name cannot be expressed as Python
        keyword arguments, so only the one-value case is reachable here; the
        multi-value case is handled by the web framework before this validator.
        """
        model = base_model_form_extra(**{"items[]": "value2"})
        assert model.model_extra == {"items": ["value2"]}

    def test_mixed_simple_and_nested(self, base_model_form_extra: type[Any]) -> None:
        """Plain, dict and list keys coexist in one payload without interfering."""
        model = base_model_form_extra(
            **{"simple": "value1", "nested[key]": "value2", "array[]": "value3"}
        )
        assert model.model_extra == {
            "simple": "value1",
            "nested": {"key": "value2"},
            "array": ["value3"],
        }

    def test_numeric_array_indices(self, base_model_form_extra: type[Any]) -> None:
        """Consecutive explicit indices produce one list in index order."""
        model = base_model_form_extra(
            **{"items[0]": "first", "items[1]": "second", "items[2]": "third"}
        )
        assert model.model_extra == {"items": ["first", "second", "third"]}

    def test_sparse_array(self, base_model_form_extra: type[Any]) -> None:
        """A gap between explicit indices is padded with ``None`` up to the highest one.

        The index is honoured as a position rather than an append order, so
        ``items[5]`` lands at index 5 and the list keeps a stable length of 6.
        """
        model = base_model_form_extra(**{"items[0]": "first", "items[5]": "sixth"})
        assert model.model_extra == {
            "items": ["first", None, None, None, None, "sixth"]
        }

    def test_nested_array_in_dict(self, base_model_form_extra: type[Any]) -> None:
        """Indexed leaves under a dict key build a list, not an integer-keyed dict."""
        model = base_model_form_extra(
            **{"config[settings][0]": "option1", "config[settings][1]": "option2"}
        )
        assert model.model_extra == {"config": {"settings": ["option1", "option2"]}}

    def test_complex_nested_structure(self, base_model_form_extra: type[Any]) -> None:
        """A deep branch and a shallower sibling merge into the same parent dict."""
        model = base_model_form_extra(
            **{"root[level1][level2][level3]": "deep", "root[level1][sibling]": "value"}
        )
        assert model.model_extra == {
            "root": {"level1": {"level2": {"level3": "deep"}, "sibling": "value"}}
        }

    def test_json_string_deserialization(
        self, base_model_form_extra: type[Any]
    ) -> None:
        """Bracket values are JSON-decoded, so numbers, booleans and null keep their type.

        Form fields are always strings on the wire, but Bedrock model parameters
        are typed, so ``"123"`` must reach the backend as the integer 123.
        """
        model = base_model_form_extra(
            **{
                "param[number]": "123",
                "param[boolean]": "true",
                "param[null]": "null",
                "param[string]": '"text"',
                "param[array]": "[1,2,3]",
            }
        )
        assert model.model_extra == {
            "param": {
                "number": 123,
                "boolean": True,
                "null": None,
                "string": "text",
                "array": [1, 2, 3],
            }
        }

    def test_invalid_json_string_kept_as_string(
        self, base_model_form_extra: type[Any]
    ) -> None:
        """A value under a bracket-free key is never JSON-decoded, so it stays a string."""
        model = base_model_form_extra(param="not-json-value")
        assert model.model_extra == {"param": "not-json-value"}

    @pytest.mark.xfail(
        reason="Pattern outer[][inner] cannot be handled: [] means append but we need to "
        "navigate into the appended item. This would require lookahead to determine if "
        "an appended item needs to be a dict or list, significantly complicating the "
        "implementation for a rare edge case.",
        strict=True,
    )
    def test_empty_bracket_in_middle_position(
        self, base_model_form_extra: type[Any]
    ) -> None:
        """An empty bracket in a non-final position is not supported (documented xfail).

        ``outer[][inner]`` would have to append a dict and then descend into it,
        which needs lookahead the parser deliberately does not implement; no
        Bedrock image parameter uses that shape.

        Ref: stdapi/types/__init__.py:_navigate_bracket_part
        """
        model = base_model_form_extra(**{"outer[][inner]": "value"})
        # Expected: outer should be an array containing one dict with key 'inner'
        assert model.model_extra == {"outer": [{"inner": "value"}]}

    def test_empty_dict(self, base_model_form_extra: type[Any]) -> None:
        """A payload with no fields yields an empty ``model_extra``, not None."""
        model = base_model_form_extra()
        assert model.model_extra == {}

    def test_real_world_virtual_tryon_params(
        self, base_model_form_extra: type[Any]
    ) -> None:
        """A full VIRTUAL_TRY_ON form payload rebuilds its nested Nova Canvas params.

        The declared OpenAI image fields travel alongside the bracket keys, so
        this checks the nested reconstruction without disturbing the flat ones.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
        """
        model = base_model_form_extra(
            **{
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
        assert model.model_extra["virtualTryOnParams"] == {
            "maskType": "IMAGE",
            "imageBasedMask": {"maskImage": "/9j/4A"},
        }
        # Bracket-free fields keep their original name, value and Python type.
        assert model.model_extra["taskType"] == "VIRTUAL_TRY_ON"
        assert model.model_extra["size"] == "1024x1024"
        assert model.model_extra["n"] == 1
        assert model.model_extra["user"] is None

    def test_real_world_color_guided_generation(
        self, base_model_form_extra: type[Any]
    ) -> None:
        """A full COLOR_GUIDED_GENERATION form payload rebuilds its nested colour list.

        Ref: https://docs.aws.amazon.com/nova/latest/userguide/image-gen-req-resp-structure.html
        """
        model = base_model_form_extra(
            **{
                "model": "amazon.titan-image-generator-v2:0",
                "response_format": "b64_json",
                "n": 1,
                "size": "512x512",
                "user": None,
                "taskType": "COLOR_GUIDED_GENERATION",
                "colorGuidedGenerationParams[colors][]": "#3357FF",
            }
        )
        assert model.model_extra["colorGuidedGenerationParams"] == {
            "colors": ["#3357FF"]
        }
        assert model.model_extra["taskType"] == "COLOR_GUIDED_GENERATION"

    def test_array_of_objects(self, base_model_form_extra: type[Any]) -> None:
        """Indexed object keys build a list of dicts, with each value JSON-decoded.

        ``items[0][value]="1"`` yields the integer 1: JSON decoding applies to
        every bracket leaf, including those inside an array of objects.
        """
        model = base_model_form_extra(
            **{
                "items[0][name]": "first",
                "items[0][value]": "1",
                "items[1][name]": "second",
                "items[1][value]": "2",
            }
        )
        assert model.model_extra == {
            "items": [{"name": "first", "value": 1}, {"name": "second", "value": 2}]
        }

    def test_consecutive_brackets_three_levels(
        self, base_model_form_extra: type[Any]
    ) -> None:
        """Three consecutive brackets nest three dict levels under the root key."""
        model = base_model_form_extra(**{"a[b][c][d]": "value"})
        assert model.model_extra == {"a": {"b": {"c": {"d": "value"}}}}

    def test_brackets_with_numbers_and_names(
        self, base_model_form_extra: type[Any]
    ) -> None:
        """A numeric segment followed by a named one builds a list of dicts."""
        model = base_model_form_extra(
            **{"mixed[0][name]": "first", "mixed[1][name]": "second"}
        )
        assert model.model_extra == {"mixed": [{"name": "first"}, {"name": "second"}]}
