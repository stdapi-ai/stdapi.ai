"""Stability AI "Stable Image" models.

Supported Models:
- stability.stable-image-core-v1
- stability.stable-image-ultra-v1
"""

from re import compile as compile_regex
from typing import ClassVar

from stdapi.models.image._stability import StabilityImageModelBase
from stdapi.models.image.stability_stable_diffusion import TextToImageJob


class StabilityCoreTextToImageJob(TextToImageJob):
    """Job for text-to-image models.

    Generation, edition and variation are inherited unchanged; these models
    only accept a narrower set of output formats.
    """

    __slots__ = ()

    #: Supported output formats, narrower than the Stability default.
    _OUTPUT_FORMATS: ClassVar[frozenset[str]] = frozenset({"png", "jpeg"})


class ImageModel(StabilityImageModelBase):
    """Stability AI image models."""

    __slots__ = ()

    MATCHER = compile_regex(r"^stability\.stable-image-(?:core|ultra)-v\d+:\d+$")
    IMAGE_GENERATION_JOB_CLASS = StabilityCoreTextToImageJob
