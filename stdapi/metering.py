"""AWS Marketplace Metering."""

from typing import TYPE_CHECKING

from botocore.exceptions import ClientError as _ClientError

from stdapi.aws import CONFIG
from stdapi.config import AWS_REGION, AWS_SESSION
from stdapi.exceptions import (
    InvalidProductError,
    NotEntitledError,
    UnsupportedPlatformError,
)
from stdapi.server import SERVER_ID, SERVER_VERSION

if TYPE_CHECKING:
    from stdapi.monitoring import EventLog

#: AWS Marketplace product code; empty for the community (unmetered) build.
PRODUCT_CODE = ""
#: Campaign parameters marking a click that started on the gateway's own docs surface.
DOCS_UTM = "utm_source=api-docs&utm_medium=product&utm_campaign=owned-surfaces"


def licence_info(product_code: str) -> dict[str, str]:
    """Build the OpenAPI ``license`` object for one product code.

    A function rather than a module-level constant alone: ``PRODUCT_CODE`` is
    fixed at import time, so this is what lets a test build the commercial
    branch without a second build of the server.

    Args:
        product_code: AWS Marketplace product code; empty for the community build.

    Returns:
        The ``license`` object reported in the OpenAPI spec.
    """
    return (
        {"name": "Commercial License", "url": "https://stdapi.ai/operations_licensing/"}
        if product_code
        else {
            "name": "GNU Affero General Public License v3.0 or later (Commercial license available)",
            "identifier": "AGPL-3.0-or-later",
        }
    )


#: License metadata reported in the OpenAPI spec.
LICENCE_INFO = licence_info(PRODUCT_CODE)
#: Server edition title reported in the OpenAPI spec.
EDITION_TITLE = f"stdapi.ai ({'Enterprise' if PRODUCT_CODE else 'Community'} Edition)"
#: Full server version string, suffixed with the edition marker ('e'/'c').
SERVER_FULL_VERSION = f"{SERVER_VERSION}+{'e' if PRODUCT_CODE else 'c'}"


async def register(start_event: EventLog) -> None:
    """Register this host with AWS Marketplace Metering.

    Applicable to ECS, EKS, and Fargate hosts running hourly-billed products.

    Args:
        start_event: Startup event log populated with ``register_usage_response``
            when a product code is configured.

    Raises:
        NotEntitledError: If the account has no entitlement for the product.
        UnsupportedPlatformError: If the platform or region does not support metering.
        InvalidProductError: If the product code or public key version is invalid.
    """
    if PRODUCT_CODE:
        product_public_key_version = 1
        async with AWS_SESSION.create_client(
            "meteringmarketplace", config=CONFIG, region_name=AWS_REGION
        ) as metering:
            try:
                start_event["register_usage_response"] = await metering.register_usage(
                    ProductCode=PRODUCT_CODE,
                    PublicKeyVersion=product_public_key_version,
                    Nonce=SERVER_ID,
                )
            except _ClientError as error:
                exc_type, exc_msg = {
                    "CustomerNotEntitledException": (
                        NotEntitledError,
                        (
                            f"No entitlement found for the '{PRODUCT_CODE}' AWS "
                            "Marketplace product. Subscribe to it with the account "
                            "running this server, then start it again."
                        ),
                    ),
                    "PlatformNotSupportedException": (
                        UnsupportedPlatformError,
                        (
                            "The AWS Marketplace product is only supported on Amazon ECS, "
                            "Amazon EKS, and AWS Fargate."
                        ),
                    ),
                    "DisabledApiException": (
                        UnsupportedPlatformError,
                        "The AWS Marketplace metering API is not available in this region.",
                    ),
                    "InvalidProductCodeException": (
                        InvalidProductError,
                        (
                            "Invalid AWS Marketplace product: "
                            f"{PRODUCT_CODE} {product_public_key_version}"
                        ),
                    ),
                    "InvalidPublicKeyVersionException": (
                        InvalidProductError,
                        (
                            "Invalid AWS Marketplace product: "
                            f"{PRODUCT_CODE} {product_public_key_version}"
                        ),
                    ),
                    "ParamValidationError": (
                        InvalidProductError,
                        f"Invalid AWS Marketplace product: {error}",
                    ),
                }.get(error.response["Error"]["Code"], (None, ""))
                if exc_type:
                    raise exc_type(exc_msg) from None
                raise  # pragma: no cover
