"""Server exceptions."""


class ServerError(Exception):
    """Exception occurring when running the server."""


class UnsupportedPlatformError(ServerError):
    """Exception occurring when the platform is not supported."""


class NotEntitledError(ServerError):
    """Exception occurring when the user is not entitled to the product."""


class InvalidProductError(ServerError):
    """Exception occurring when the product is invalid."""
