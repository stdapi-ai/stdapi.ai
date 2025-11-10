"""AWS Marketplace Metering."""

from botocore.exceptions import ClientError as _ClientError

from stdapi.aws import CONFIG, REGION, SESSION
from stdapi.exceptions import (
    InvalidProductError,
    NotEntitledError,
    UnsupportedPlatformError,
)
from stdapi.server import SERVER_ID, SERVER_VERSION

PRODUCT_CODE = ""
LICENCE_INFO = (
    {"name": "Commercial License", "url": "https://stdapi.ai/commercial_license"}
    if PRODUCT_CODE
    else {
        "name": "GNU Affero General Public License v3.0 or later (Commercial license available)",
        "identifier": "AGPL-3.0-or-later",
    }
)
EDITION_TITLE = f"stdapi.ai ({'Professional' if PRODUCT_CODE else 'Community'} Edition)"
SERVER_FULL_VERSION = f"{SERVER_VERSION}+{'p' if PRODUCT_CODE else 'c'}"


async def register() -> None:
    """Register AWS Marketplace for the current host.

    For ECS, EKS & Fargate hosts running hourly billed products.

    Args:
        config: Application configuration.
    """
    if PRODUCT_CODE:
        product_public_key_version = 1
        product_url = ""
        async with SESSION.client(
            "meteringmarketplace", config=CONFIG, region_name=REGION
        ) as metering:
            try:
                await metering.register_usage(
                    ProductCode=PRODUCT_CODE,
                    PublicKeyVersion=product_public_key_version,
                    Nonce=SERVER_ID,
                )
            except _ClientError as error:
                exc_type, exc_msg = {
                    "CustomerNotEntitledException": (
                        NotEntitledError,
                        (
                            "No entitlement found for this AWS Marketplace product."
                            f" Please subscribe to the product: {product_url}"
                            if product_url
                            else ""
                        ),
                    ),
                    "PlatformNotSupportedException": (
                        UnsupportedPlatformError,
                        "The AWS Marketplace product is only supported on Amazon ECS, "
                        "Amazon EKS, and AWS Fargate.",
                    ),
                    "DisabledApiException": (
                        UnsupportedPlatformError,
                        "The AWS Marketplace metering API is not available in this region.",
                    ),
                    "InvalidProductCodeException": (
                        InvalidProductError,
                        "Invalid AWS Marketplace product: "
                        f"{PRODUCT_CODE} {product_public_key_version}",
                    ),
                    "InvalidPublicKeyVersionException": (
                        InvalidProductError,
                        "Invalid AWS Marketplace product: "
                        f"{PRODUCT_CODE} {product_public_key_version}",
                    ),
                    "ParamValidationError": (
                        InvalidProductError,
                        f"Invalid AWS Marketplace product: {error}",
                    ),
                }.get(error.response["Error"]["Code"], (None, ""))
                if exc_type:
                    raise exc_type(exc_msg) from None
                raise  # pragma: no cover
