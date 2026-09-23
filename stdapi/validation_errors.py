"""Word request-validation faults as the API a route mirrors words the same fault.

Every wording here was observed on the vendor's own API with raw requests: the
OpenAI routes do not share one — Chat Completions, Responses, Images, Batches,
Vector Stores, Conversations and Videos name the parameter and the failure class
in ``param`` and ``code``, while Moderations and Audio relay their validator's
error list, Embeddings writes a sentence per field, and Files and Uploads use
JSON-schema wording, all with ``param`` and ``code`` null save for the few fields
upstream names (``_FIELD_REFUSALS``, ``_JSON_SCHEMA_NAMED_FIELDS``).
"""

from collections.abc import Mapping
from re import compile as compile_regex
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

#: A route, as its method and path template.
type Route = tuple[str, str]

#: A refusal: its message, then OpenAI's ``param`` and ``code``.
type Refusal = tuple[str, str | None, str | None]

#: Pydantic's own container and union tags in an error location, e.g. ``list[union[A,B]]``.
_PYDANTIC_TYPE_TAG = compile_regex(r"[a-z-]+\[.+\]")

#: Request parts FastAPI names ahead of the field path of a validation error.
_REQUEST_PARTS: Final = frozenset({"body", "query", "path", "header", "cookie"})

#: Location parts shaped like a Pydantic union branch: a container or a scalar type.
_UNION_BRANCH_TAG = compile_regex(r"[a-z-]+\[.+\]|str|int|float|bool|none")

#: Fields a union member is told apart by: a failure on one rules the member out.
_DISCRIMINATING_FIELDS: Final = frozenset({"type", "role"})

#: What each Pydantic type error expected, in OpenAI's words and as a JSON-schema type.
_EXPECTED_TYPES: Final = {
    "string_type": ("a string", "string"),
    "int_type": ("an integer", "integer"),
    "int_parsing": ("an integer", "integer"),
    "int_from_float": ("an integer", "integer"),
    "float_type": ("a decimal", "number"),
    "float_parsing": ("a decimal", "number"),
    "bool_type": ("a boolean", "boolean"),
    "bool_parsing": ("a boolean", "boolean"),
    "list_type": ("an array", "array"),
    "tuple_type": ("an array", "array"),
    "set_type": ("an array", "array"),
    "dict_type": ("an object", "object"),
    "model_type": ("an object", "object"),
    "model_attributes_type": ("an object", "object"),
    "dataclass_type": ("an object", "object"),
}

#: Range errors: the bound's context key, its operator, and the side it limits in words and in the code.
_RANGE_ERRORS: Final = {
    "greater_than": ("gt", ">", "below minimum", "below_min"),
    "greater_than_equal": ("ge", ">=", "below minimum", "below_min"),
    "less_than": ("lt", "<", "above maximum", "above_max"),
    "less_than_equal": ("le", "<=", "above maximum", "above_max"),
}

#: Errors naming a list of accepted values.
_ENUM_ERRORS: Final = frozenset({"literal_error", "enum", "union_tag_invalid"})

#: OpenAI's answer to a body that is not a JSON object.
_UNPARSABLE_BODY: Final = "We could not parse the JSON body of your request."

#: OpenAI's answer to a request naming no model, on the routes that check it first.
_NO_MODEL: Final = "you must provide a model parameter"

#: How many faults an error-list answer relays before counting the rest.
_MAX_LISTED_ERRORS: Final = 20

#: Separators Pydantic lists the accepted values of a literal or a union tag with.
_LISTED_VALUES_SEPARATOR = compile_regex(r", | or ")

#: Routes that read ``model`` before validating the body.
_MODEL_FIRST_ROUTES: Final = frozenset(
    ("POST", path)
    for path in (
        "/v1/chat/completions",
        "/v1/completions",
        "/v1/embeddings",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
    )
)

#: Routes that relay their validator's error list, with ``param`` and ``code`` null.
_ERROR_LIST_ROUTES: Final = frozenset(
    ("POST", path)
    for path in (
        "/v1/moderations",
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/audio/translations",
    )
)

#: The route that words each field's refusal in a sentence of its own, ``param`` and ``code`` null.
_EMBEDDINGS_ROUTE: Final = ("POST", "/v1/embeddings")

#: Routes refusing in JSON-schema wording, ``code`` null.
_JSON_SCHEMA_ROUTES: Final = frozenset(
    {("POST", "/v1/uploads"), ("POST", "/v1/files"), ("GET", "/v1/files")}
)

#: Routes whose ``image`` takes one file or several.
_FILE_OR_FILES_ROUTES: Final = frozenset({("POST", "/v1/images/edits")})

#: Fields a JSON-schema route names in ``param`` when it refuses their value.
_JSON_SCHEMA_NAMED_FIELDS: Final = frozenset({"purpose"})

#: Fields refused ahead of the route's own wording: a fixed refusal, or None for the Chat Completions wording.
_FIELD_REFUSALS: Final[dict[Route, dict[str, Refusal | None]]] = {
    ("POST", "/v1/audio/speech"): {
        "response_format": (
            "Invalid response_format.",
            "response_format",
            "unsupported_value",
        ),
        "instructions": None,
    }
}


def validation_error_path(loc: Iterable[Any]) -> str:
    """Join a Pydantic error location into a field path a client can act on.

    Drops the union and list wrappers Pydantic descended through, which bury the
    failing field: only those carry a parameterized type name, while a field the
    client sent, such as the multipart ``image[]``, keeps its empty brackets.

    Args:
        loc: Location parts of one Pydantic error.

    Returns:
        The dotted field path, empty when nothing addressable remains.
    """
    return ".".join(
        part for part in map(str, loc) if not _PYDANTIC_TYPE_TAG.fullmatch(part)
    )


def _is_branch_tag(part: Any) -> bool:  # noqa: ANN401
    """Whether a location part names a union member: a class, a container or a scalar type.

    Args:
        part: One part of an error location.

    Returns:
        True for a branch tag.
    """
    return isinstance(part, str) and (
        part[:1].isupper() or bool(_UNION_BRANCH_TAG.fullmatch(part))
    )


def reported_error(errors: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Pick the fault to report: the deepest, in a union member the value could be.

    A union-typed field reports one error per member, and the shallowest blames
    the whole field instead of the item inside it that failed. A member whose
    ``type`` or ``role`` the value does not match is ruled out first, so an item
    is judged by the member it names rather than the one with the longest path.

    Args:
        errors: Every fault of the request.

    Returns:
        The fault to report, None when there is none.
    """
    ruled_out = {
        tuple(loc[:-1])
        for error in errors
        if len(loc := error.get("loc", ())) > 1
        and loc[-1] in _DISCRIMINATING_FIELDS
        and _is_branch_tag(loc[-2])
        and error.get("type") in {"literal_error", "missing"}
    }
    candidates = [
        error
        for error in errors
        if not any(
            tuple(error.get("loc", ())[:end]) in ruled_out
            for end in range(1, len(error.get("loc", ())) + 1)
        )
    ] or list(errors)
    return max(candidates, key=lambda error: len(error.get("loc", ())), default=None)


def _field_location(loc: Sequence[Any]) -> Sequence[Any]:
    """Drop the request part FastAPI names ahead of an error location.

    A body the route parses itself reports the field path alone, which a
    single-part location such as a ``query`` field keeps whole.

    Args:
        loc: Location parts of one Pydantic error.

    Returns:
        The location of the field within its request part.
    """
    if loc and loc[0] in _REQUEST_PARTS and (len(loc) > 1 or loc[0] == "body"):
        return loc[1:]
    return loc


def _sent_field_path(loc: Sequence[Any], body: Any) -> list[str | int]:  # noqa: ANN401
    """Return the parts of an error location that name what the client sent.

    Pydantic also names the union branches it descended through. With the
    payload at hand, a part is kept when the payload holds it -- unless it is
    the object's own ``type`` or ``role`` value followed by one of its fields,
    which is a tagged-union branch -- and the last one also when it names a
    field missing from an object rather than a branch. Without it, the parts
    shaped like a container or scalar branch are dropped instead, since the
    models a route validates itself may name their fields in ``CamelCase``.

    Args:
        loc: Location parts of one Pydantic error.
        body: The request payload FastAPI validated, None when the route parsed it.

    Returns:
        The field names and item indexes leading to the failing field.
    """
    parts = _field_location(loc)
    if not isinstance(body, Mapping | list):
        return [
            part
            for part in parts
            if isinstance(part, int) or not _UNION_BRANCH_TAG.fullmatch(str(part))
        ]
    path: list[str | int] = []
    node = body
    last = len(parts) - 1
    for index, part in enumerate(parts):
        if isinstance(node, Mapping) and isinstance(part, str) and part in node:
            if (
                index < last
                and any(node.get(field) == part for field in _DISCRIMINATING_FIELDS)
                and parts[index + 1] in node
            ):
                continue
            node = node[part]
        elif isinstance(node, list) and isinstance(part, int) and part < len(node):
            node = node[part]
        elif index != last or not isinstance(node, Mapping) or _is_branch_tag(part):
            continue
        path.append(part)
    return path


def _openai_param(path: Iterable[str | int]) -> str | None:
    """Write a field path in OpenAI's notation, e.g. ``messages[0].content``.

    Args:
        path: Field names and item indexes.

    Returns:
        The parameter, None for the whole body.
    """
    return (
        "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}" for part in path
        ).removeprefix(".")
        or None
    )


def _listed(items: Sequence[str], conjunction: str) -> str:
    """Join values as OpenAI lists them: ``'a' and 'b'``, ``'a', 'b', and 'c'``.

    Args:
        items: Values, already quoted.
        conjunction: Word ahead of the last value.

    Returns:
        The enumeration.
    """
    if len(items) < 3:
        return f" {conjunction} ".join(items)
    return f"{', '.join(items[:-1])}, {conjunction} {items[-1]}"


def _accepted_values(ctx: Mapping[str, Any]) -> list[str]:
    """Return the accepted values a literal or union-tag error lists, each still quoted.

    Args:
        ctx: The error's context.

    Returns:
        The values, as Pydantic quotes them.
    """
    listed = str(ctx.get("expected") or ctx.get("expected_tags") or "")
    return _LISTED_VALUES_SEPARATOR.split(listed)


def _json_type(value: Any) -> str | None:  # noqa: ANN401, PLR0911 - one arm per JSON type
    """Name the JSON type of a value as OpenAI names what it received.

    Args:
        value: The value that failed to validate.

    Returns:
        The type, None when the value is not a JSON value.
    """
    match value:
        case None:
            return "null"
        case bool():
            return "a boolean"
        case int():
            return "an integer"
        case float():
            return "a decimal number"
        case str():
            return "a string"
        case list() | tuple():
            return "an array"
        case Mapping():
            return "an object"
        case _:
            return None


def _invalid_type(
    param: str,
    expected: Sequence[str],
    value: Any,  # noqa: ANN401
    *,
    unparsed: bool = False,
) -> Refusal:
    """Build OpenAI's refusal of a value of the wrong type.

    Args:
        param: The parameter, in OpenAI's notation.
        expected: Every type the parameter accepts.
        value: The value received.
        unparsed: Whether the value is form or query text no integer could be read from.

    Returns:
        The refusal.
    """
    wanted = expected[0] if len(expected) == 1 else f"one of {_listed(expected, 'or')}"
    if unparsed:
        got = ", but got a string value that could not be converted into an integer"
    elif received := _json_type(value):
        got = f", but got {received} instead"
    else:
        got = ""
    return f"Invalid type for '{param}': expected {wanted}{got}.", param, "invalid_type"


def _is_file_error(kind: str, ctx: Mapping[str, Any], msg: str) -> bool:
    """Whether an error refuses a value that is not an uploaded file.

    Args:
        kind: The Pydantic error type.
        ctx: The error's context.
        msg: The error's message.

    Returns:
        True when a file was expected.
    """
    return (kind == "is_instance_of" and ctx.get("class") == "UploadFile") or (
        kind == "value_error" and "Expected UploadFile" in msg
    )


def _chat_completions_wording(  # noqa: C901, PLR0911, PLR0912 - one arm per failure class
    error: Mapping[str, Any],
    errors: Sequence[Mapping[str, Any]],
    body: Any,  # noqa: ANN401
    *,
    form: bool,
) -> Refusal:
    """Word a fault as OpenAI's Chat Completions and Responses routes word it.

    Args:
        error: The fault reported to the client.
        errors: Every fault of the request, for the other branches of a union.
        body: The request payload FastAPI validated, None when the route parsed it.
        form: Whether the body is a form, whose values arrive as text.

    Returns:
        The refusal.
    """
    kind = str(error.get("type", ""))
    ctx: Mapping[str, Any] = error.get("ctx") or {}
    value = error.get("input")
    loc = error.get("loc", ())
    path = _sent_field_path(loc, body)
    param = _openai_param(path)
    msg = str(error.get("msg", ""))
    reason = msg.removeprefix("Value error, ").removeprefix("Assertion failed, ")
    if kind in {"union_tag_invalid", "union_tag_not_found"}:
        # A field discriminator is quoted; a callable one names no field.
        field = str(ctx.get("discriminator", ""))
        if field.startswith("'") and (field := field.strip("'")).isidentifier():
            param = _openai_param([*path, field])
    # A body that is absent, or is no object, is one OpenAI cannot parse.
    if kind == "json_invalid" or (
        param is None
        and (kind == "missing" or _EXPECTED_TYPES.get(kind, ("",))[0] == "an object")
    ):
        return _UNPARSABLE_BODY, None, None
    if param is None:
        return reason, None, None
    if _is_file_error(kind, ctx, msg):
        return _invalid_type(param, ["a file"], value)
    match kind:
        case "extra_forbidden":
            return f"Unknown parameter: '{param}'.", param, "unknown_parameter"
        case "missing" | "union_tag_not_found":
            return (
                f"Missing required parameter: '{param}'.",
                param,
                "missing_required_parameter",
            )
        case kind if kind in _ENUM_ERRORS:
            accepted = _accepted_values(ctx)
            if kind == "union_tag_invalid":
                value = ctx.get("tag")
            if not isinstance(value, str):
                return _invalid_type(
                    param, [f"one of {_listed(accepted, 'or')}"], value
                )
            listing = _listed(accepted, "and")
            return (
                f"Invalid value: '{value}'. Supported values are: {listing}.",
                param,
                "invalid_value",
            )
        case "value_error" | "assertion_error":
            return reason, param, None
        case kind if kind in _EXPECTED_TYPES:
            # A union reports one type error per branch, all on the same field.
            expected = list(
                dict.fromkeys(
                    _EXPECTED_TYPES[other["type"]][0]
                    for other in errors
                    if other.get("type") in _EXPECTED_TYPES
                    and _sent_field_path(other.get("loc", ()), body) == path
                )
            )
            unparsed = (
                kind == "int_parsing"
                and isinstance(value, str)
                and (form or loc[:1] == ("query",))
            )
            return _invalid_type(param, expected, value, unparsed=unparsed)
        case kind if kind in _RANGE_ERRORS:
            key, operator, side, side_code = _RANGE_ERRORS[kind]
            bound = ctx.get(key)
            # The bound carries the field's type: a float field bounds with a float.
            number = (
                "integer"
                if isinstance(bound, int) and not isinstance(bound, bool)
                else "decimal"
            )
            message = (
                f"Invalid '{param}': {number} {side} value. Expected a value "
                f"{operator} {bound}, but got {value} instead."
            )
            return message, param, f"{number}_{side_code}_value"
        case "too_short" if value == []:
            message = (
                f"Invalid '{param}': empty array. Expected an array with minimum "
                f"length {ctx.get('min_length')}, but got an empty array instead."
            )
            return message, param, "empty_array"
        case "too_long" if isinstance(value, list):
            message = (
                f"Invalid '{param}': array too long. Expected an array with maximum "
                f"length {ctx.get('max_length')}, but got an array with length "
                f"{len(value)} instead."
            )
            return message, param, "array_above_max_length"
        case "string_too_short" if value == "":
            message = (
                f"Invalid '{param}': empty string. Expected a string with minimum "
                f"length {ctx.get('min_length')}, but got an empty string instead."
            )
            return message, param, "empty_string"
        case "string_too_long" if isinstance(value, str):
            message = (
                f"Invalid '{param}': string too long. Expected a string with maximum "
                f"length {ctx.get('max_length')}, but got a string with length "
                f"{len(value)} instead."
            )
            return message, param, "string_above_max_length"
    return f"Invalid value for '{param}': {msg}.", param, "invalid_value"


def _error_list_wording(errors: Sequence[Mapping[str, Any]]) -> str:
    """List the faults as the Moderations and Audio routes do: Pydantic's own errors.

    The value that failed is left out, as upstream leaves it out, except where
    the error carries none at all or the body did not parse. A validator's
    exception is quoted by its message alone, and the list is capped where
    upstream lists every fault, so the item count of a request cannot drive the
    size of its answer.

    Args:
        errors: Every fault of the request.

    Returns:
        The message.
    """
    listed: list[object] = [
        {
            key: (
                {
                    name: str(item) if isinstance(item, BaseException) else item
                    for name, item in value.items()
                }
                if key == "ctx" and isinstance(value, Mapping)
                else value
            )
            for key, value in error.items()
            if key != "url"
            and (key != "input" or value is None or error.get("type") == "json_invalid")
        }
        for error in errors[:_MAX_LISTED_ERRORS]
    ]
    if (dropped := len(errors) - _MAX_LISTED_ERRORS) > 0:
        listed.append(f"and {dropped} more validation errors")
    return str(listed)


def _embeddings_wording(error: Mapping[str, Any], field: str) -> str | None:
    """Word a fault as the Embeddings route does: one sentence per field.

    Args:
        error: The fault reported to the client.
        field: The parameter, in OpenAI's notation.

    Returns:
        The message, None for a fault upstream has no sentence for.
    """
    kind = str(error.get("type", ""))
    ctx: Mapping[str, Any] = error.get("ctx") or {}
    value = error.get("input")
    if field == "input" or field.startswith(("input.", "input[")):
        if kind == "missing":
            return "Please submit an `input`."
        return "Invalid 'input': expected a string or token array."
    if kind in _ENUM_ERRORS:
        accepted = [value.strip("'") for value in _accepted_values(ctx)]
        return f"Invalid value for '{field}' = {value}. Supported values: {accepted!r}."
    if kind in _RANGE_ERRORS:
        key = _RANGE_ERRORS[kind][0]
        bound = ctx.get(key)
        rule = {
            "gt": f"greater than {bound}",
            # An integer bound reads as the exclusive one upstream states.
            "ge": (
                f"greater than {bound - 1}"
                if isinstance(bound, int)
                else f"greater than or equal to {bound}"
            ),
            "lt": f"less than {bound}",
            "le": f"less than or equal to {bound}",
        }[key]
        return f"Invalid value for '{field}' = {value}. Must be {rule}."
    if kind in _EXPECTED_TYPES:
        return f"{value!r} is not of type '{_EXPECTED_TYPES[kind][1]}' - '{field}'"
    return None


def _json_schema_wording(
    error: Mapping[str, Any], field: str, *, query: bool
) -> str | None:
    """Word a fault as the Files and Uploads routes do, in JSON-schema terms.

    Args:
        error: The fault reported to the client.
        field: The parameter, in OpenAI's notation.
        query: Whether the field is a query parameter.

    Returns:
        The message, None for a fault upstream has no sentence for.
    """
    kind = str(error.get("type", ""))
    ctx: Mapping[str, Any] = error.get("ctx") or {}
    value = error.get("input")
    where = "" if query else f" - '{field}'"
    if kind == "missing":
        return f"'{field}' is a required property"
    if kind in _ENUM_ERRORS:
        accepted = [value.strip("'") for value in _accepted_values(ctx)]
        return f"{value!r} is not one of {accepted!r}{where}"
    if kind in _EXPECTED_TYPES:
        schema_type = _EXPECTED_TYPES[kind][1]
        if query:
            return f"Wrong type, expected '{schema_type}' for query parameter '{field}'"
        return f"{value!r} is not of type '{schema_type}'{where}"
    if kind in _RANGE_ERRORS:
        key = _RANGE_ERRORS[kind][0]
        side = (
            "less than the minimum"
            if key in {"gt", "ge"}
            else "greater than the maximum"
        )
        return f"{value} is {side} of {ctx.get(key)}{where}"
    return None


def openai_validation_error(  # noqa: C901, PLR0911, PLR0912 - one arm per route wording
    route: Route,
    errors: Sequence[Mapping[str, Any]],
    body: Any,  # noqa: ANN401
    *,
    form: bool,
) -> Refusal:
    """Word a request's validation faults as the OpenAI route it called words them.

    Args:
        route: The method and path template of the route.
        errors: Every fault of the request.
        body: The request payload FastAPI validated, None when the route parsed it.
        form: Whether the body is a form, whose values arrive as text.

    Returns:
        The refusal.
    """
    error = reported_error(errors)
    model_first = route in _MODEL_FIRST_ROUTES
    if model_first:
        for model_error in errors:
            if _field_location(model_error.get("loc", ())) == ("model",):
                kind = model_error.get("type")
                if kind in {"missing", "string_too_short"} or (
                    model_error.get("input") is None
                ):
                    return _NO_MODEL, None, None
                if kind == "string_type":
                    return _UNPARSABLE_BODY, None, None
                # Any other refusal of the model is the one reported, as upstream reads it first.
                error = model_error
                break
    if error is None:
        return _UNPARSABLE_BODY, None, None
    wording = _chat_completions_wording(error, errors, body, form=form)
    message, param, _code = wording
    if param is None:
        # Upstream lists a whole-body fault too, where no model is read first.
        if route in _ERROR_LIST_ROUTES and not model_first:
            return _error_list_wording(errors), None, None
        return wording
    if (
        route in _FILE_OR_FILES_ROUTES
        and param == "image"
        and _is_file_error(
            str(error.get("type")), error.get("ctx") or {}, str(error.get("msg", ""))
        )
    ):
        return _invalid_type(
            param, ["one of an array of files or file"], error.get("input")
        )
    field = param.split(".", 1)[0].split("[", 1)[0]
    fields = _FIELD_REFUSALS.get(route, {})
    if field in fields:
        return fields[field] or wording
    if route in _ERROR_LIST_ROUTES:
        return _error_list_wording(errors), None, None
    if route == _EMBEDDINGS_ROUTE:
        return _embeddings_wording(error, param) or message, None, None
    if route in _JSON_SCHEMA_ROUTES:
        query = error.get("loc", ())[:1] == ("query",)
        schema_message = _json_schema_wording(error, param, query=query)
        named = (
            param
            if param in _JSON_SCHEMA_NAMED_FIELDS
            and str(error.get("type")) in _ENUM_ERRORS
            else None
        )
        return schema_message or message, named, None
    return wording


def anthropic_validation_message(errors: Sequence[Mapping[str, Any]]) -> str:
    """Word a request's validation faults as the Anthropic API does: ``<loc>: <msg>``.

    Args:
        errors: Every fault of the request.

    Returns:
        The message.
    """
    if (error := reported_error(errors)) is None:
        return "Validation error"
    if error.get("type") == "json_invalid":
        return f"The request body is not valid JSON: {(error.get('ctx') or {}).get('error')}"
    msg = str(error.get("msg", ""))
    if path := validation_error_path(_field_location(error.get("loc", ()))):
        return f"{path}: {msg}"
    kind = str(error.get("type", ""))
    if kind == "missing" or _EXPECTED_TYPES.get(kind, ("",))[0] == "an object":
        # An absent body reaches validation as None, like a JSON null one.
        got = type(error.get("input")).__name__
        return f"The request body must be a JSON object, got {got}."
    return msg
