"""Root endpoint for the API."""

from fastapi import APIRouter

router = APIRouter()

_WELCOME = {
    "message": "Welcome to the stdapi.ai API! Documentation is available at https://stdapi.ai/api_reference/"
}


@router.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Return a welcome message for the API root endpoint.

    Returns:
        Dictionary containing a welcome message with link to documentation.
    """
    return _WELCOME
