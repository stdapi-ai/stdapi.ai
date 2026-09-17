"""Unit tests for stdapi.types.openai_images local OpenAI-compatible image types.

The OpenAI image schemas mark most of their optional fields ``nullable: true``,
so an explicit JSON ``null`` is a legal way to leave one unset. Clients that
serialise a struct of optional fields -- the default in several JSON libraries
-- send ``null`` instead of omitting the key, so every one of those fields has
to validate exactly as its omission does.

Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
     https://stdapi.ai/api_openai_images_generations/
     stdapi/types/openai_images.py
"""

from typing import TYPE_CHECKING, Any

import pytest
from pydantic import ValidationError

from stdapi.types.openai_images import (
    ImageEditJsonBody,
    ImageEditParams,
    ImageGenerateParams,
    ImageVariationJsonBody,
    ImageVariationParams,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Smallest image reference the JSON request bodies accept.
_IMAGE_DATA_URI = "data:image/png;base64,AA=="

#: Fields `CreateImageRequest` marks nullable, `model` excluded (required here).
_GENERATION_NULLABLE = (
    "background",
    "moderation",
    "n",
    "output_compression",
    "output_format",
    "partial_images",
    "quality",
    "response_format",
    "size",
    "stream",
    "style",
)

#: Fields `CreateImageEditRequest` marks nullable, `model` excluded (required here).
_EDIT_NULLABLE = (
    "background",
    "input_fidelity",
    "n",
    "output_compression",
    "output_format",
    "partial_images",
    "quality",
    "response_format",
    "size",
    "stream",
)

#: Fields `CreateImageVariationRequest` marks nullable, `model` excluded (required here).
_VARIATION_NULLABLE = ("n", "response_format", "size")

#: Each images request model, a minimal valid body for it, and its nullable fields.
_REQUESTS: tuple[tuple[str, type[BaseModel], dict[str, Any], tuple[str, ...]], ...] = (
    (
        "generations",
        ImageGenerateParams,
        {"model": "m", "prompt": "p"},
        _GENERATION_NULLABLE,
    ),
    ("edits-form", ImageEditParams, {"model": "m"}, _EDIT_NULLABLE),
    (
        "edits-json",
        ImageEditJsonBody,
        {"model": "m", "images": [_IMAGE_DATA_URI]},
        _EDIT_NULLABLE,
    ),
    ("variations-form", ImageVariationParams, {"model": "m"}, _VARIATION_NULLABLE),
    (
        "variations-json",
        ImageVariationJsonBody,
        {"model": "m", "image": _IMAGE_DATA_URI},
        _VARIATION_NULLABLE,
    ),
)

#: One case per (request model, nullable field) pair.
_NULLABLE_FIELDS = pytest.mark.parametrize(
    ("params_type", "body", "field"),
    [
        (params_type, body, field)
        for _, params_type, body, fields in _REQUESTS
        for field in fields
    ],
    ids=[f"{name}-{field}" for name, _, _, fields in _REQUESTS for field in fields],
)

#: Every images request model, with a minimal valid body for it.
_EVERY_REQUEST = pytest.mark.parametrize(
    ("params_type", "body"),
    [(params_type, body) for _, params_type, body, _ in _REQUESTS],
    ids=[name for name, _, _, _ in _REQUESTS],
)


def _comparable(params: BaseModel) -> dict[str, Any]:
    """Return the request's fields, without the image references.

    A reference holds a parsed input file, which compares by identity, so two
    validations of the same body are never equal on those fields.

    Args:
        params: A validated images request.

    Returns:
        The request's fields, image references excluded.
    """
    return params.model_dump(exclude={"image", "images", "mask"})


class TestNullableFieldsReadAsUnset:
    """An explicit ``null`` leaves a nullable field unset, as it does upstream.

    Refusing it makes the gateway stricter than the API it mirrors and turns
    an ordinary request from a struct-serialising client into an error.

    Ref: https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml
         stdapi/types/openai_images.py:_ImageBaseParams
    """

    @_NULLABLE_FIELDS
    def test_a_null_validates_exactly_like_an_omitted_field(
        self, params_type: type[BaseModel], body: dict[str, Any], field: str
    ) -> None:
        """The request carrying the null is indistinguishable from one without the key.

        Comparing the two validated bodies asserts the whole request, so a null
        that reached the model as a value -- or as an extra parameter forwarded
        to the backend -- fails here rather than downstream.
        """
        assert _comparable(params_type.model_validate({**body, field: None})) == (
            _comparable(params_type.model_validate(body))
        )

    @_NULLABLE_FIELDS
    def test_a_null_keeps_the_field_unset(
        self, params_type: type[BaseModel], body: dict[str, Any], field: str
    ) -> None:
        """The field reads as never sent, which is what ``null`` asks for."""
        params = params_type.model_validate({**body, field: None})

        assert field not in params.model_fields_set

    @_EVERY_REQUEST
    def test_a_null_model_is_reported_as_the_missing_field(
        self, params_type: type[BaseModel], body: dict[str, Any]
    ) -> None:
        """``model`` is required here, so its null is the error an omission gives.

        Upstream falls back to a default model; this API has none to fall back
        to, and the answer names the field the caller has to send rather than
        the type it got wrong.
        """
        with pytest.raises(ValidationError) as exc_info:
            params_type.model_validate({**body, "model": None})

        errors = exc_info.value.errors()
        assert [(error["loc"], error["type"]) for error in errors] == [
            (("model",), "missing")
        ]

    @pytest.mark.parametrize(
        ("params_type", "body", "field"),
        [
            (ImageGenerateParams, {"model": "m", "prompt": "p"}, "prompt"),
            (ImageEditJsonBody, {"model": "m", "images": [_IMAGE_DATA_URI]}, "images"),
            (ImageVariationJsonBody, {"model": "m", "image": _IMAGE_DATA_URI}, "image"),
        ],
        ids=["prompt", "images", "image"],
    )
    def test_a_field_that_is_not_nullable_still_refuses_a_null(
        self, params_type: type[BaseModel], body: dict[str, Any], field: str
    ) -> None:
        """A required field sent as null is still an error, as it is upstream.

        The schema marks none of these nullable, so reading their null as an
        omission would turn a malformed request into a silent default.
        """
        with pytest.raises(ValidationError) as exc_info:
            params_type.model_validate({**body, field: None})

        assert [error["loc"][0] for error in exc_info.value.errors()] == [field]

    def test_an_unsupported_value_is_still_refused_when_a_neighbour_is_null(
        self,
    ) -> None:
        """Reading nulls as unset does not stop the unsupported-option checks.

        ``background="transparent"`` is refused whatever else the body carries,
        so the null handling cannot be a way around it.
        """
        with pytest.raises(ValidationError) as exc_info:
            ImageGenerateParams.model_validate(
                {"model": "m", "prompt": "p", "background": "transparent", "n": None}
            )

        errors = exc_info.value.errors()
        assert len(errors) == 1
        assert "Background transparency is not supported" in errors[0]["msg"]

    def test_a_null_extra_parameter_is_left_untouched(self) -> None:
        """A backend-specific parameter is passed on as sent, null included.

        Only the fields the OpenAI schema declares are read as unset; an extra
        one belongs to the model it is forwarded to, which is what decides
        what its null means.
        """
        params = ImageGenerateParams.model_validate(
            {"model": "m", "prompt": "p", "seed": None}
        )

        assert params.model_extra == {"seed": None}
