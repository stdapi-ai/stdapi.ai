"""Local types for the stdapi.ai project.

This package hosts OpenAI-compatible type definitions used by the routes to
avoid hard runtime dependencies on the official openai.types package.
"""

from contextlib import suppress
from re import compile as regex_compile
from typing import Any

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator
from pydantic_core import from_json

from stdapi.config import SETTINGS

#: Regex pattern for parsing form bracket notation
_BRACKET_PARSE_PATTERN = regex_compile(r"([^\[\]]+)|\[\]")


class BaseModelRequest(BaseModel):
    """Pydantic Basemodel request."""

    model_config = ConfigDict(
        extra="forbid" if SETTINGS.strict_input_validation else "allow", frozen=True
    )


class BaseModelRequestWithExtra(BaseModel):
    """Pydantic Basemodel request storing extra JSON fields."""

    model_config = ConfigDict(extra="allow", frozen=True)
    __pydantic_extra__: dict[str, JsonValue] = {}


def _ensure_list_size(lst: list[JsonValue], idx: int) -> None:
    """Extend list to accommodate a specific index.

    Pads the list with None values up to the required index if needed.
    This allows setting values at arbitrary indices without IndexError.

    Args:
        lst: The list to extend (modified in-place).
        idx: The target index that must be accessible.
    """
    lst.extend([None] * max(0, idx + 1 - len(lst)))


def _navigate_bracket_part(
    current: dict[str, JsonValue] | list[JsonValue], part: str, next_key: str
) -> dict[str, JsonValue] | list[JsonValue]:
    """Navigate one level deeper in form bracket notation structure.

    Processes a single bracket notation segment and returns the next level
    in the nested structure, creating intermediate dicts/lists as needed.

    The type of structure created (dict vs list) is determined by examining
    the next key: empty string or numeric indicates a list should be created,
    otherwise a dict is created.

    Args:
        current: Current position in the nested structure.
        part: Current bracket notation segment (e.g., "key", "0", "").
        next_key: Next bracket notation segment to determine structure type.

    Returns:
        The next level in the nested structure.
    """
    is_array = not next_key or next_key.isdigit()

    if part.isdigit() and isinstance(current, list):
        idx = int(part)
        _ensure_list_size(current, idx)
        if current[idx] is None or not isinstance(current[idx], (dict, list)):
            current[idx] = [] if is_array else {}
        return current[idx]  # type: ignore[return-value]

    if isinstance(current, dict):
        if part not in current or not isinstance(current[part], (dict, list)):
            current[part] = [] if is_array else {}
        return current[part]  # type: ignore[return-value]

    return current


def _set_bracket_leaf(
    current: dict[str, JsonValue] | list[JsonValue], leaf: str, value: JsonValue
) -> None:
    """Set the final value at the leaf position of bracket notation.

    Handles three types of leaf assignments:
    - Empty string: Append to array (e.g., "items[]")
    - Numeric string: Set array index (e.g., "items[0]")
    - Named string: Set dict key (e.g., "items[name]")

    Args:
        current: The container (dict or list) to set the value in.
        leaf: The final bracket notation segment.
        value: The value to set.
    """
    if not leaf and isinstance(current, list):
        current.append(value)
    elif leaf.isdigit() and isinstance(current, list):
        idx = int(leaf)
        _ensure_list_size(current, idx)
        current[idx] = value
    elif isinstance(current, dict):
        current[leaf] = value


class BaseModelRequestWithFormExtra(BaseModelRequestWithExtra):
    """Pydantic Basemodel request storing extra from form.

    Automatically deserializes from in extra fields to their Python equivalents.
    """

    @model_validator(mode="before")
    @classmethod
    def _deserialize_forms(cls, data: Any) -> Any:  # noqa: ANN401
        """Deserialize PHP/HTML-style form bracket notation into nested structures.

        Parses multipart/form-data fields with bracket notation into proper
        nested dictionaries and lists, matching the behavior of PHP's $_POST
        array parsing. This enables complex nested data structures to be sent
        via HTML forms or multipart requests.

        String values are automatically deserialized from JSON when possible,
        converting types like "123" to 123, "true" to True, etc.

        Bracket Notation Patterns:
            - 'param[key]' -> {'param': {'key': value}}
              Nested dictionary access

            - 'param[]' -> {'param': [value1, value2, ...]}
              Array append (multiple fields with same name)

            - 'param[0]' -> {'param': [value]}
              Explicit array index

            - 'param[key][]' -> {'param': {'key': [value1, value2, ...]}}
              Nested dict containing array

            - 'param[a][b][c]' -> {'param': {'a': {'b': {'c': value}}}}
              Deep nesting

        Args:
            data: Raw form data dictionary with bracket notation keys.

        Returns:
            Transformed dictionary with nested structures, or original data
            if not a dict.
        """
        if not isinstance(data, dict):
            return data

        result: dict[str, JsonValue] = {}
        for key, value in data.items():
            if "[" not in key:
                result[key] = value
                continue

            parts = [p or "" for p in _BRACKET_PARSE_PATTERN.findall(key)]
            if not parts:
                continue

            json_value: JsonValue = value
            if isinstance(value, str):
                with suppress(ValueError):
                    json_value = from_json(value)

            *path, leaf = parts
            current: dict[str, JsonValue] | list[JsonValue] = result
            for part, next_key in zip(path, [*path[1:], leaf], strict=False):
                current = _navigate_bracket_part(current, part, next_key)

            _set_bracket_leaf(current, leaf, json_value)

        return result


class BaseModelResponse(BaseModel):
    """Pydantic Basemodel response."""

    model_config = ConfigDict(extra="forbid")
