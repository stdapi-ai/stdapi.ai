"""Server information."""

from os import getpid
from socket import gethostname

from stdapi.utils import webuuid

#: Unique server ID
SERVER_ID = webuuid()

#: Unique server full name
SERVER_NAME = f"{gethostname()}-{getpid()}-{SERVER_ID}"

#: Server version
SERVER_VERSION = "1.0.2"

#: Server user agent
USER_AGENT = f"stdapi.ai/{SERVER_VERSION}/{SERVER_ID}"

#: Default headers for HTTP clients used in the server
HTTP_CLIENT_HEADERS = {"User-Agent": USER_AGENT}
