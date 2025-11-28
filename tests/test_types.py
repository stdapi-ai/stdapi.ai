"""Tests for stdapi.types module.

Comprehensive test suite for form bracket notation parsing in BaseModelRequestWithFormExtra.
"""

from typing import Any

import pytest


@pytest.fixture
def base_model_form_extra() -> type[Any]:
    """Import and return BaseModelRequestWithFormExtra only when tests run."""
    from stdapi.types import BaseModelRequestWithFormExtra  # noqa: PLC0415

    return BaseModelRequestWithFormExtra


class TestBaseModelRequestWithFormExtra:
    """Test form bracket notation parsing in BaseModelRequestWithFormExtra."""

    def test_simple_key(self, base_model_form_extra: type[Any]) -> None:
        """Test simple key without brackets."""
        model = base_model_form_extra(key="value")
        assert model.model_extra == {"key": "value"}

    def test_nested_dict(self, base_model_form_extra: type[Any]) -> None:
        """Test nested dictionary notation."""
        model = base_model_form_extra(**{"param[key]": "value"})
        assert model.model_extra == {"param": {"key": "value"}}

    def test_array_append(self, base_model_form_extra: type[Any]) -> None:
        """Test array append notation with []."""
        model = base_model_form_extra(**{"param[]": "value1"})
        assert model.model_extra == {"param": ["value1"]}

    def test_array_explicit_index(self, base_model_form_extra: type[Any]) -> None:
        """Test explicit array index."""
        model = base_model_form_extra(**{"param[0]": "value"})
        assert model.model_extra == {"param": ["value"]}

    def test_nested_dict_with_array(self, base_model_form_extra: type[Any]) -> None:
        """Test nested dict containing array."""
        model = base_model_form_extra(**{"param[key][]": "value"})
        assert model.model_extra == {"param": {"key": ["value"]}}

    def test_deep_nesting(self, base_model_form_extra: type[Any]) -> None:
        """Test deep nested structure."""
        model = base_model_form_extra(**{"param[a][b][c]": "value"})
        assert model.model_extra == {"param": {"a": {"b": {"c": "value"}}}}

    def test_consecutive_brackets_nested_dict(
        self, base_model_form_extra: type[Any]
    ) -> None:
        """Test consecutive brackets creating nested dicts (original bug)."""
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
        """Test array append in nested structure."""
        model = base_model_form_extra(
            **{"colorGuidedGenerationParams[colors][]": "#3357FF"}
        )
        assert model.model_extra == {
            "colorGuidedGenerationParams": {"colors": ["#3357FF"]}
        }

    def test_multiple_array_appends(self, base_model_form_extra: type[Any]) -> None:
        """Test multiple values appended to same array.

        Note: This is a Python dict limitation - duplicate keys result in last value.
        In real multipart/form-data, multiple values with same key would be handled
        differently by the web framework.
        """
        # Only the last value survives due to dict key uniqueness
        model = base_model_form_extra(**{"items[]": "value2"})
        assert model.model_extra == {"items": ["value2"]}

    def test_mixed_simple_and_nested(self, base_model_form_extra: type[Any]) -> None:
        """Test mixing simple keys with nested notation."""
        model = base_model_form_extra(
            **{"simple": "value1", "nested[key]": "value2", "array[]": "value3"}
        )
        assert model.model_extra == {
            "simple": "value1",
            "nested": {"key": "value2"},
            "array": ["value3"],
        }

    def test_numeric_array_indices(self, base_model_form_extra: type[Any]) -> None:
        """Test explicit numeric array indices."""
        model = base_model_form_extra(
            **{"items[0]": "first", "items[1]": "second", "items[2]": "third"}
        )
        assert model.model_extra == {"items": ["first", "second", "third"]}

    def test_sparse_array(self, base_model_form_extra: type[Any]) -> None:
        """Test sparse array with gaps filled by None."""
        model = base_model_form_extra(**{"items[0]": "first", "items[5]": "sixth"})
        assert model.model_extra == {
            "items": ["first", None, None, None, None, "sixth"]
        }

    def test_nested_array_in_dict(self, base_model_form_extra: type[Any]) -> None:
        """Test array nested within dictionary."""
        model = base_model_form_extra(
            **{"config[settings][0]": "option1", "config[settings][1]": "option2"}
        )
        assert model.model_extra == {"config": {"settings": ["option1", "option2"]}}

    def test_complex_nested_structure(self, base_model_form_extra: type[Any]) -> None:
        """Test complex multi-level nested structure."""
        model = base_model_form_extra(
            **{"root[level1][level2][level3]": "deep", "root[level1][sibling]": "value"}
        )
        assert model.model_extra == {
            "root": {"level1": {"level2": {"level3": "deep"}, "sibling": "value"}}
        }

    def test_json_string_deserialization(
        self, base_model_form_extra: type[Any]
    ) -> None:
        """Test automatic JSON deserialization in bracket notation values."""
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
        """Test that invalid JSON strings remain as strings."""
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
        """Test empty bracket [] in middle of path should create array with nested dict."""
        model = base_model_form_extra(**{"outer[][inner]": "value"})
        # Expected: outer should be an array containing one dict with key 'inner'
        assert model.model_extra == {"outer": [{"inner": "value"}]}

    def test_empty_dict(self, base_model_form_extra: type[Any]) -> None:
        """Test empty dictionary input."""
        model = base_model_form_extra()
        assert model.model_extra == {}

    def test_real_world_virtual_tryon_params(
        self, base_model_form_extra: type[Any]
    ) -> None:
        """Test real-world virtual try-on parameters."""
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

    def test_real_world_color_guided_generation(
        self, base_model_form_extra: type[Any]
    ) -> None:
        """Test real-world color guided generation parameters."""
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

    def test_array_of_objects(self, base_model_form_extra: type[Any]) -> None:
        """Test array containing objects (values auto-deserialize from JSON)."""
        model = base_model_form_extra(
            **{
                "items[0][name]": "first",
                "items[0][value]": "1",
                "items[1][name]": "second",
                "items[1][value]": "2",
            }
        )
        # Note: "1" and "2" are auto-deserialized to integers 1 and 2
        assert model.model_extra == {
            "items": [{"name": "first", "value": 1}, {"name": "second", "value": 2}]
        }

    def test_consecutive_brackets_three_levels(
        self, base_model_form_extra: type[Any]
    ) -> None:
        """Test three consecutive brackets creating three-level nesting."""
        model = base_model_form_extra(**{"a[b][c][d]": "value"})
        assert model.model_extra == {"a": {"b": {"c": {"d": "value"}}}}

    def test_brackets_with_numbers_and_names(
        self, base_model_form_extra: type[Any]
    ) -> None:
        """Test mixing numeric and named keys."""
        model = base_model_form_extra(
            **{"mixed[0][name]": "first", "mixed[1][name]": "second"}
        )
        assert model.model_extra == {"mixed": [{"name": "first"}, {"name": "second"}]}
