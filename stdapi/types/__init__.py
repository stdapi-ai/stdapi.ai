"""Local types, avoiding a runtime dependency on the official provider packages."""

from contextlib import suppress
from re import compile as regex_compile
from typing import Any

from pydantic import BaseModel, ConfigDict, JsonValue, model_validator
from pydantic_core import from_json

from stdapi.config import SETTINGS

#: Regex pattern for parsing form bracket notation
_BRACKET_PARSE_PATTERN = regex_compile(r"([^\[\]]+)|\[\]")

#: Regex pattern that a valid Files API file ID must match on input (both prefixes accepted).
FILE_ID_PATTERN: str = r"^file[-_][a-z0-9]{32}$"

#: Regex pattern that a valid Uploads API upload ID must match.
UPLOAD_ID_PATTERN: str = r"^upload_[a-z0-9]{32}$"

#: Regex pattern that a valid Uploads API part ID must match.
PART_ID_PATTERN: str = r"^part_[0-9a-f]{32}$"

#: JSON mapping type for structured data
JsonMapping = dict[str, JsonValue]

#: JSON list type for structured data
JsonList = list[JsonValue]

#: JSON list or mapping
JsonMappingOrList = JsonMapping | JsonList


class BaseModelRequest(BaseModel):
    """Pydantic Basemodel request."""

    model_config = ConfigDict(
        extra="forbid" if SETTINGS.strict_input_validation else "ignore", frozen=True
    )


class BaseModelRequestWithExtra(BaseModel):
    """Pydantic Basemodel request storing extra JSON fields."""

    model_config = ConfigDict(extra="allow", frozen=True)
    __pydantic_extra__: JsonMapping = {}


def _ensure_list_size(lst: JsonList, idx: int) -> None:
    """Pad the list with ``None`` so *idx* can be assigned without IndexError.

    Args:
        lst: The list to extend (modified in-place).
        idx: The target index that must be accessible.
    """
    lst.extend([None] * max(0, idx + 1 - len(lst)))


def _navigate_bracket_part(
    current: JsonMappingOrList, part: str, next_key: str
) -> JsonMappingOrList:
    """Navigate one level deeper in form bracket notation structure.

    The missing intermediate container is created as a list when *next_key* is
    empty or numeric, as a dict otherwise.

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


def _set_bracket_leaf(current: JsonMappingOrList, leaf: str, value: JsonValue) -> None:
    """Set the final value at the leaf position of bracket notation.

    Appends for ``items[]``, sets the index for ``items[0]``, the key otherwise.

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
    """Pydantic Basemodel request storing form extra fields, JSON-deserialized."""

    @model_validator(mode="before")
    @classmethod
    def _deserialize_forms(cls, data: Any) -> Any:  # noqa: ANN401
        """Deserialize PHP/HTML-style form bracket notation into nested structures.

        String values of *extra* fields (those not declared on the model, e.g.
        provider-specific parameters) are deserialized from JSON when possible,
        mirroring how a JSON request body would already carry them. Declared
        fields are left untouched here (a string that merely looks like JSON,
        e.g. a prompt of "null", must reach the field as-is).

        Bracket Notation Patterns:
            - 'param[key]' -> {'param': {'key': value}}
            - 'param[]' -> {'param': [value1, value2, ...]}
            - 'param[0]' -> {'param': [value]}
            - 'param[key][]' -> {'param': {'key': [value1, value2, ...]}}
            - 'param[a][b][c]' -> {'param': {'a': {'b': {'c': value}}}}

        Args:
            data: Raw form data dictionary with bracket notation keys.

        Returns:
            Transformed dictionary with nested structures, or original data
            if not a dict.
        """
        if not isinstance(data, dict):
            return data

        result: JsonMapping = {}
        for key, value in data.items():
            json_value: JsonValue = value
            if key.split("[", 1)[0] not in cls.model_fields and isinstance(value, str):
                with suppress(ValueError):
                    json_value = from_json(value)

            if "[" not in key:
                result[key] = json_value
                continue

            parts = [p or "" for p in _BRACKET_PARSE_PATTERN.findall(key)]
            if not parts:
                continue

            *path, leaf = parts
            current: JsonMappingOrList = result
            for part, next_key in zip(path, [*path[1:], leaf], strict=False):
                current = _navigate_bracket_part(current, part, next_key)

            _set_bracket_leaf(current, leaf, json_value)

        return result


class BaseModelResponse(BaseModel):
    """Pydantic Basemodel response."""

    model_config = ConfigDict(extra="forbid")
