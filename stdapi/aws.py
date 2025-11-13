"""AWS client management and connection pooling."""

import os
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, TypeVar

from aioboto3 import Session
from aiobotocore.config import AioConfig
from aiohttp import ClientError, ClientSession, ClientTimeout

from stdapi.config import SETTINGS
from stdapi.server import USER_AGENT

if TYPE_CHECKING:
    from types import TracebackType


#: Current detected region
REGION: str = Session().region_name or SETTINGS.aws_bedrock_regions[0]

#: Session with the default region
SESSION = Session(region_name=SETTINGS.aws_bedrock_regions[0])

#: AWS account information (populated during startup)
AWS_ACCOUNT_INFO: dict[str, str] = {}

_CLIENTS: dict[str, dict[str, Any]] = {}

_RETRIES = {"max_attempts": 10, "mode": "adaptive"}
_MAX_POOL_CONNECTIONS = 50

#: Default configuration
CONFIG = AioConfig(
    user_agent=USER_AGENT,
    retries=_RETRIES,
    max_pool_connections=_MAX_POOL_CONNECTIONS,
    parameter_validation=False,
)


class AWSConnectionManager:
    """Manages persistent AWS client connections."""

    __slots__ = ("_client_specs", "_exit_stack")

    def __init__(self, *clients: tuple[str, str | None]) -> None:
        """Initialize AWS connection manager with client specifications.

        Args:
            *clients: Variable number of tuples containing service name and optional region.
                Each tuple contains (service_name, region_name or None).
        """
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._client_specs = clients

    async def __aenter__(self) -> "AWSConnectionManager":
        """Initialize AWS clients.

        Returns:
            AWSConnectionManager: The initialized connection manager.
        """
        await self._exit_stack.__aenter__()
        for service, region in {
            (service, region or SESSION.region_name)
            for service, region in self._client_specs
        }:
            config = (
                AioConfig(
                    user_agent=USER_AGENT,
                    retries=_RETRIES,
                    max_pool_connections=_MAX_POOL_CONNECTIONS,
                    parameter_validation=False,
                    s3={"use_accelerate_endpoint": SETTINGS.aws_s3_accelerate},
                )
                if service == "s3.accelerate"
                else CONFIG
            )
            _CLIENTS.setdefault(service, {})[
                region
            ] = await self._exit_stack.enter_async_context(
                SESSION.client(
                    service.split(".", 1)[0], region_name=region, config=config
                )  # type: ignore[call-overload]
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: "TracebackType | None",
    ) -> None:
        """Cleanup all AWS clients.

        Args:
            exc_type: Exception type if an error occurred within the context.
            exc_val: Exception instance if an error occurred within the context.
            exc_tb: Traceback object if an error occurred within the context.
        """
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        _CLIENTS.clear()


ClientT = TypeVar("ClientT")


def get_client(service: str, region_name: str | None = None) -> Any:  # noqa:ANN401
    """Get AWS client.

    Args:
        service: AWS service name.
        region_name: Optional specific region,
            use default region if not specified.

    Returns:
        AWS client instance.

    Raises:
        KeyError: If multiple regional clients exist and the requested region
            is not available in the pool.
    """
    clients = _CLIENTS[service]
    try:
        return clients[region_name or SESSION.region_name]
    except KeyError:
        if len(clients) == 1:
            return next(iter(clients.values()))
        raise


async def initialize_aws_account_info() -> None:
    """Initialize AWS account information at server startup.

    Retrieves AWS account ID from ECS container metadata (if available)
    or falls back to STS API. Also extracts ECS task ID if running in ECS.
    Stores results in AWS_ACCOUNT_INFO dict.
    """
    try:
        metadata_path = os.environ["ECS_CONTAINER_METADATA_URI_V4"]
    except KeyError:
        # Not running in ECS
        pass
    else:
        try:
            async with (
                ClientSession(timeout=ClientTimeout(total=2, connect=1)) as session,
                session.get(f"http://169.254.170.2{metadata_path}/task") as resp,
            ):
                resp.raise_for_status()
                parts = (await resp.json())["TaskARN"].split(":")
                AWS_ACCOUNT_INFO["account_id"] = parts[4]
                AWS_ACCOUNT_INFO["task_id"] = parts[5].split("/")[-1]
                return
        except (OSError, ClientError):
            pass

    async with SESSION.client("sts", config=CONFIG, region_name=REGION) as sts_client:
        AWS_ACCOUNT_INFO["account_id"] = (await sts_client.get_caller_identity())[
            "Account"
        ]
