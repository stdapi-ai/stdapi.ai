"""AWS client management and connection pooling."""

from contextlib import AsyncExitStack
from logging import getLogger
from os import environ
from typing import TYPE_CHECKING, Any, NotRequired, Self, TypedDict, TypeVar

from aiobotocore.config import AioConfig
from aiohttp import ClientSession, ClientTimeout

from stdapi import server
from stdapi.config import AWS_REGION, AWS_SESSION, SETTINGS

if TYPE_CHECKING:
    from types import TracebackType

    from types_aiobotocore_bedrock.literals import RegionName

    class AwsEnvironment(TypedDict):
        """AWS environment."""

        account_id: NotRequired[str]


#: AWS environment information (populated during startup)
AWS_ENVIRONMENT: AwsEnvironment = {}

#: Cached AWS service clients keyed by (service, region)
_CLIENTS: dict[str, dict[RegionName, Any]] = {}

#: Default retry configuration (configurable attempts and retry mode)
_RETRIES = {
    "max_attempts": SETTINGS.aws_bedrock_max_retries + 1,
    "mode": "adaptive" if SETTINGS.aws_adaptive_retry else "standard",
}

#: Default configuration — used by all services including bedrock-runtime
CONFIG = AioConfig(
    user_agent=server.USER_AGENT,
    retries=_RETRIES,
    max_pool_connections=SETTINGS.aws_max_pool_connections,
    parameter_validation=False,
    connect_timeout=SETTINGS.aws_connect_timeout,
    read_timeout=SETTINGS.ai_response_timeout,
)

#: No-retry configuration — used when the application retry loop manages failover across regions
CONFIG_NO_RETRY = AioConfig(
    user_agent=server.USER_AGENT,
    retries={
        "max_attempts": 1,
        "mode": "adaptive" if SETTINGS.aws_adaptive_retry else "standard",
    },
    max_pool_connections=SETTINGS.aws_max_pool_connections,
    parameter_validation=False,
    connect_timeout=SETTINGS.aws_connect_timeout,
    read_timeout=SETTINGS.ai_response_timeout,
)

getLogger("aiobotocore").setLevel("CRITICAL")


class AWSConnectionManager:
    """Manages persistent AWS client connections."""

    __slots__ = ("_client_specs", "_exit_stack")

    def __init__(self, *clients: tuple[str, RegionName | None]) -> None:
        """Initialize AWS connection manager with client specifications.

        Args:
            *clients: Variable number of tuples containing service name and optional region.
                Each tuple contains (service_name, region_name or None).
        """
        self._exit_stack: AsyncExitStack = AsyncExitStack()
        self._client_specs = clients

    async def __aenter__(self) -> Self:
        """Initialize AWS clients.

        Returns:
            AWSConnectionManager: The initialized connection manager.
        """
        await self._exit_stack.__aenter__()
        for service, region in {
            (service, region or SETTINGS.aws_bedrock_regions[0])
            for service, region in self._client_specs
        }:
            if service == "s3.accelerate":
                config = AioConfig(
                    user_agent=server.USER_AGENT,
                    retries=_RETRIES,
                    max_pool_connections=SETTINGS.aws_max_pool_connections,
                    parameter_validation=False,
                    connect_timeout=SETTINGS.aws_connect_timeout,
                    s3={"use_accelerate_endpoint": SETTINGS.aws_s3_accelerate},
                )
            else:
                config = CONFIG
            _CLIENTS.setdefault(service, {})[
                region
            ] = await self._exit_stack.enter_async_context(
                AWS_SESSION.create_client(  # type: ignore[call-overload]
                    service.split(".", 1)[0], region_name=region, config=config
                )
            )

        if (
            SETTINGS.aws_bedrock_region_routing != "disabled"
            and "bedrock-runtime" in _CLIENTS
        ):
            for region in _CLIENTS["bedrock-runtime"]:
                _CLIENTS.setdefault("bedrock-runtime.no-retry", {})[
                    region
                ] = await self._exit_stack.enter_async_context(
                    AWS_SESSION.create_client(
                        "bedrock-runtime", region_name=region, config=CONFIG_NO_RETRY
                    )
                )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
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


def get_client(service: str, region_name: RegionName | None = None) -> Any:  # noqa:ANN401
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
        return clients[region_name or SETTINGS.aws_bedrock_regions[0]]
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
        metadata_path = environ["ECS_CONTAINER_METADATA_URI_V4"]
    except KeyError:
        async with AWS_SESSION.create_client(
            "sts",
            config=AioConfig(user_agent=server.USER_AGENT, parameter_validation=False),
            region_name=AWS_REGION,
        ) as sts_client:
            AWS_ENVIRONMENT["account_id"] = (await sts_client.get_caller_identity())[
                "Account"
            ]
    else:
        async with ClientSession(
            headers=server.HTTP_CLIENT_HEADERS,
            timeout=ClientTimeout(total=2, connect=1),
        ) as session:
            async with session.get(metadata_path) as resp:
                resp.raise_for_status()
                container_name = (await resp.json())["Name"]
            async with session.get(f"{metadata_path}/task") as resp:
                resp.raise_for_status()
                parts = (await resp.json())["TaskARN"].split(":")
                AWS_ENVIRONMENT["account_id"] = parts[4]
                task_id = parts[5].split("/")[-1]
        server.SERVER_NAME = f"{task_id}-{container_name}-{server.SERVER_ID}"
