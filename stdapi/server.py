"""Server information."""

from os import getpid
from socket import gethostname

from stdapi.utils import webuuid

#: Unique server ID
SERVER_ID = webuuid()

#: Unique server full name (replaced on ECS at startup)
SERVER_NAME = f"{gethostname()[:128]}-{getpid()}-{SERVER_ID}"

#: Server version
SERVER_VERSION = "1.20.1"

#: Server title reported in the OpenAPI spec and to MCP clients
SERVER_TITLE = "stdapi.ai"

#: License metadata reported in the OpenAPI spec
LICENSE_INFO = {
    "name": "GNU Affero General Public License v3.0 or later",
    "identifier": "AGPL-3.0-or-later",
}

#: Campaign parameters marking a click that started on the gateway's own docs surface
DOCS_UTM = "utm_source=api-docs&utm_medium=product&utm_campaign=owned-surfaces"

#: Server user agent
USER_AGENT = f"stdapi.ai/{SERVER_VERSION}/{SERVER_ID}"

#: User agent used by the MCP internal HTTP client
MCP_USER_AGENT = f"stdapi.ai/MCP/{SERVER_ID}"

#: Default headers for HTTP clients used in the server
HTTP_CLIENT_HEADERS = frozendict({"User-Agent": USER_AGENT})

#: Internal header name for request ID propagation; randomised at startup to prevent spoofing.
INTERNAL_REQUEST_ID_HEADER = f"x-stdapi-{webuuid()}"
