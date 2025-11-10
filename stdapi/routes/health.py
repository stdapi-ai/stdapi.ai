"""Health check endpoint for container orchestration and monitoring.

Provides a simple health check endpoint that can be used by Docker, Kubernetes,
load balancers, and monitoring systems to verify the application is running and
responsive.
"""

from fastapi import APIRouter
from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response for the /health endpoint.

    Indicates the application is running and ready to accept requests.
    """

    status: str = "ok"


router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Health Check",
    description="Simple health check endpoint to verify the service is running",
    response_description="Health status of the service",
    responses={
        200: {
            "description": "Service is healthy and operational",
            "content": {
                "application/json": {
                    "examples": {
                        "healthy": {
                            "summary": "Healthy response",
                            "value": {"status": "ok"},
                        }
                    }
                }
            },
        }
    },
)
async def health_check() -> HealthResponse:
    """Check if the service is healthy and operational.

    This endpoint does not require authentication and returns a simple status
    response. It is designed to be used by container orchestration systems
    (Docker, Kubernetes), load balancers, and monitoring tools to verify
    the application is running.

    Returns:
        HealthResponse with status "ok" when the service is operational
    """
    return HealthResponse()
