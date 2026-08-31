"""MkDocs hooks for documentation generation.

This module provides hooks that run during MkDocs build process to generate
dynamic content like the OpenAPI specification.
"""

from pathlib import Path
from typing import Any
from warnings import catch_warnings, simplefilter

from yaml import dump


def on_pre_build(config: Any) -> None:  # noqa: ARG001,ANN401
    """Generate OpenAPI specification before building documentation.

    This hook is called before MkDocs starts processing documentation files.
    It generates the openapi.yml file from the FastAPI application.

    Args:
        config: MkDocs configuration object.
    """
    with catch_warnings():
        simplefilter("ignore")
        from stdapi.main import app  # noqa: PLC0415

        openapi_schema = app.openapi()
    # The tagged contact URL is meant for the gateway's own Swagger/ReDoc
    # pages; left in, the docs site's API Reference page (also rendered with
    # ReDoc) would link to itself with the tag, overwriting a real visitor's
    # traffic source with our own.
    contact_url = openapi_schema.get("info", {}).get("contact", {}).get("url")
    if contact_url:
        openapi_schema["info"]["contact"]["url"] = contact_url.split("?", 1)[0]
    with Path("docs/openapi.yml").open("w") as f:
        content = dump(openapi_schema, stream=None, sort_keys=False, allow_unicode=True)
        content = content.replace(" (Community Edition)", "")
        f.write(content)
