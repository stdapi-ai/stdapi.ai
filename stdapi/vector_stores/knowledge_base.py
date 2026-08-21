"""The Amazon Bedrock Knowledge Bases implementation of the vector index contract.

A knowledge base is never this server's: it addresses one the operator
allowlisted, and neither creates nor deletes it. What it does own is the
document plane — ingesting, listing and deleting the documents of one data
source of it — and the retrieval the store answers searches with. Reading is
wider than writing: a search spans every data source of the knowledge base, so
a passage of the corpus behind the store reads back where it lives.

Everything specific to that service lives here and nowhere else: the store
identifier, the allowlist, the data source resolution, the document identifier
encoding, the filter dialect, the two generations' differences, and the IAM
actions each call needs.
"""

from base64 import urlsafe_b64decode, urlsafe_b64encode
from binascii import Error as BinasciiError
from hashlib import blake2b
from re import compile as re_compile
from typing import TYPE_CHECKING, Any, Final, NoReturn

from botocore.exceptions import BotoCoreError, ClientError

from stdapi.api_errors import (
    ApiError,
    FeatureUnavailableError,
    feature_unavailable_guard,
)
from stdapi.aws import get_client
from stdapi.config import SETTINGS
from stdapi.files import get_file, get_file_content, parse_file_id
from stdapi.monitoring import add_server_warning, log_error_details
from stdapi.types.openai_vector_stores import (
    Attributes,
    AttributeValue,
    CompoundFilter,
    SearchFilter,
)
from stdapi.usage import record_knowledge_base_usage
from stdapi.utils import now_utc_timestamp, to_json_bytes
from stdapi.vector_stores._concurrency import gather_bounded
from stdapi.vector_stores._paging import page_records
from stdapi.vector_stores.backend import (
    FEATURE,
    IndexCapabilities,
    IndexVector,
    VectorMatch,
    parse_filter,
    raise_not_found,
    unsupported_file_message,
)
from stdapi.vector_stores.models import (
    FileErrorRecord,
    FileRecord,
    PendingFile,
    StoreRecord,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterable, Sequence
    from contextlib import AbstractContextManager

    from stdapi.monitoring import EventLog
    from stdapi.types import JsonMapping

#: Identifier prefix addressing a store served by a knowledge base.
STORE_ID_PREFIX: Final[str] = "vs_kb_"

#: Regex fragment a knowledge base identifier matches.
KNOWLEDGE_BASE_ID_PATTERN: Final[str] = r"[0-9A-Za-z]{10}"

#: Matcher for a data source identifier, which is shaped as a knowledge base's.
_DATA_SOURCE_ID_RE = re_compile(rf"^{KNOWLEDGE_BASE_ID_PATTERN}$").match

#: Identifier prefix of a document this server did not attach itself.
_DOCUMENT_ID_PREFIX: Final[str] = "kbdoc_"

#: Regex fragment a document identifier matches on input.
DOCUMENT_ID_PATTERN: Final[str] = rf"{_DOCUMENT_ID_PREFIX}[A-Za-z0-9_-]{{1,3000}}"

#: Compiled matcher for the document identifiers this server encodes.
_DOCUMENT_ID_RE = re_compile(rf"^{DOCUMENT_ID_PATTERN}$").match

#: Metadata key holding the name of the file a document was attached from.
#: Never leading-underscored: a managed knowledge base reserves that prefix and
#: fails the document rather than ingesting it.
_FILENAME_KEY: Final[str] = "stdapi-filename"

#: Metadata key prefixes the service reserves, one per generation.
_SERVICE_METADATA_PREFIXES: Final[tuple[str, ...]] = ("x-amz-bedrock-kb-", "_")

#: Metadata keys the service reports the document's own location under.
_SERVICE_URI_KEYS: Final[tuple[str, ...]] = (
    "x-amz-bedrock-kb-source-uri",
    "_source_uri",
)

#: Metadata keys the service reports a passage's own data source under.
_SERVICE_DATA_SOURCE_KEYS: Final[tuple[str, ...]] = (
    "x-amz-bedrock-kb-data-source-id",
    "_data_source_id",
)

#: Scheme a document location carries when the document is an object of a bucket.
_OBJECT_URI_SCHEME: Final[str] = "s3://"

#: Documents one ingest, read or delete call carries.
_DOCUMENTS_BATCH: Final[int] = 10

#: Documents one listing call reads: the managed generation refuses more.
_LIST_PAGE_MAX: Final[int] = 100

#: Documents a listing reads, over as many calls as it takes, before paging in memory.
_LIST_SCAN_MAX: Final[int] = 1000

#: Bytes of one document the inline ingestion path accepts.
_MAX_DOCUMENT_BYTES: Final[int] = 5 * 1024 * 1024

#: Bytes of attributes one document carries, as searchable metadata.
_MAX_ATTRIBUTE_BYTES: Final[int] = 2048

#: Queries and document calls issued concurrently by one request.
_CALL_WAVE: Final[int] = 8

#: Knowledge base kinds this backend serves, out of the four the service models.
#: A ``SQL`` one answers rows of a structured store and a ``KENDRA`` one an index
#: of another service: neither holds the passages a vector store answers with.
SERVED_KINDS: Final[tuple[str, ...]] = ("MANAGED", "VECTOR")

#: Characters a search query may hold, per knowledge base generation.
_QUERY_CHARACTERS_MAX: Final[dict[str, int]] = {"VECTOR": 1000, "MANAGED": 10000}

#: Largest ``numberOfResults`` a retrieval accepts.
_RESULTS_MAX: Final[int] = 100

#: Service error code meaning the knowledge base does not exist.
_MISSING_CODE: Final[str] = "ResourceNotFoundException"

#: Service error code covering every call the service validates and rejects.
_INVALID_CODE: Final[str] = "ValidationException"

#: Fragment of that error naming a document addressed as a kind its data source
#: is not, which is the one thing its message is matched on.
_WRONG_KIND_MARKER: Final[str] = "dataSourceType"

#: Document status meaning the data source no longer holds it.
_ABSENT_STATUS: Final[str] = "NOT_FOUND"

#: OpenAI comparison operator -> retrieval filter operator.
_FILTER_OPERATORS: Final[dict[str, str]] = {
    "eq": "equals",
    "ne": "notEquals",
    "gt": "greaterThan",
    "gte": "greaterThanOrEquals",
    "lt": "lessThan",
    "lte": "lessThanOrEquals",
    "in": "in",
    "nin": "notIn",
}

#: OpenAI combinator -> retrieval filter combinator.
_FILTER_COMBINATORS: Final[dict[str, str]] = {"and": "andAll", "or": "orAll"}

#: Document formats every knowledge base indexes as they stand.
_DOCUMENT_MEDIA_TYPES: Final[frozenset[str]] = frozenset(
    {
        "application/msword",
        "application/pdf",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/csv",
        "text/html",
        "text/markdown",
        "text/plain",
    }
)

#: Formats a managed knowledge base additionally indexes, with media extraction.
#: Exactly the three extraction lists the service documents, no wider and no
#: narrower: a format outside them is left for the knowledge base to refuse.
MANAGED_MEDIA_TYPES: Final[frozenset[str]] = frozenset(
    {
        # Presentations, whose visuals are extracted alongside their text.
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        # .mp3, .wav, .m4a, .flac, .ogg
        "audio/flac",
        "audio/mp4",
        "audio/mpeg",
        "audio/ogg",
        "audio/wav",
        # .png, .jpg, .jpeg, .jpe, .tif, .tiff, .gif, .bmp, .webp, .svg, .jp2, .heic
        "image/bmp",
        "image/gif",
        "image/heic",
        "image/jp2",
        "image/jpeg",
        "image/png",
        "image/svg+xml",
        "image/tiff",
        "image/webp",
        # .mp4, .mov, .m4v
        "video/mp4",
        "video/quicktime",
        "video/x-m4v",
    }
)

#: Media types ingested as the text they already are, rather than as a document.
_PLAIN_TEXT_MEDIA_TYPES: Final[frozenset[str]] = frozenset(
    {"text/markdown", "text/plain"}
)

#: Media types whose bytes hold no indexable document, refused before they are read.
_REFUSED_MEDIA_TYPES: Final[frozenset[str]] = frozenset(
    {"application/gzip", "application/octet-stream", "application/zip"}
)

#: What this backend can express, as the engine reads it.
CAPABILITIES: Final = IndexCapabilities(
    filter_operators=frozenset(_FILTER_OPERATORS),
    filter_combinators=frozenset(_FILTER_COMBINATORS),
    ingested_media_types=_DOCUMENT_MEDIA_TYPES,
    refused_media_types=_REFUSED_MEDIA_TYPES,
    ingests_decodable_text=True,
    chunks_on_ingestion=True,
    # A retrieval score is a bare relevance value with no stated range: it is
    # reported as it stands and never rescaled into a similarity.
    normalised_score=False,
    max_chunk_bytes=0,
)

#: Document status reported by the service -> the status the API answers with.
_DOCUMENT_STATUS: Final[dict[str, str]] = {
    "INDEXED": "completed",
    "PARTIALLY_INDEXED": "completed",
    "METADATA_PARTIALLY_INDEXED": "completed",
    "PENDING": "in_progress",
    "STARTING": "in_progress",
    "IN_PROGRESS": "in_progress",
    "DELETING": "in_progress",
    "DELETE_IN_PROGRESS": "in_progress",
    "FAILED": "failed",
    "METADATA_UPDATE_FAILED": "failed",
    "IGNORED": "failed",
}


def is_knowledge_base_store(store_id: str) -> bool:
    """Whether *store_id* addresses a store served by a knowledge base."""
    return store_id.startswith(STORE_ID_PREFIX)


def store_id_of(knowledge_base_id: str) -> str:
    """Return the vector store identifier addressing *knowledge_base_id*."""
    return f"{STORE_ID_PREFIX}{knowledge_base_id}"


def knowledge_base_id_of(store_id: str) -> str:
    """Return the knowledge base *store_id* addresses."""
    return store_id.removeprefix(STORE_ID_PREFIX)


def allowlist() -> dict[str, str]:
    """Return the addressable knowledge bases and the data source each ingests into.

    Returns:
        Knowledge base identifier to data source identifier, the latter empty
        when the deployment did not name one.
    """
    entries: dict[str, str] = {}
    for entry in SETTINGS.aws_bedrock_knowledge_base_ids:
        knowledge_base_id, _, data_source_id = entry.partition("/")
        entries[knowledge_base_id] = data_source_id
    return entries


def check_allowlisted(store_id: str) -> str:
    """Return the knowledge base *store_id* addresses, if the deployment may.

    A knowledge base that was not allowlisted is answered exactly as one that
    does not exist: the wording, the status and the body are the same, so the
    allowlist cannot be probed through the API.

    Args:
        store_id: A vector store identifier.

    Returns:
        The knowledge base identifier.

    Raises:
        ApiError: When the knowledge base is not addressable (404).
    """
    knowledge_base_id = knowledge_base_id_of(store_id)
    if knowledge_base_id not in allowlist():
        raise_not_found("vector store", store_id)
    return knowledge_base_id


def document_file_id(identifier: JsonMapping, data_source_id: str = "") -> str:
    """Return the file identifier addressing one document.

    A document this server attached carries the uploaded file's own identifier,
    so it addresses unchanged. Any other one — a document of the customer's own
    corpus, located by a URI — is encoded into an opaque identifier the routes
    can carry and this module decodes back. The data source travels with it: a
    knowledge base holds several, and a document is only addressable in the one
    holding it.

    Args:
        identifier: The document identifier as the service reports it.
        data_source_id: The data source holding the document, when it is known
            to be one other than the store's own.

    Returns:
        The file identifier.
    """
    source_type = identifier.get("dataSourceType")
    if source_type == "CUSTOM":
        custom: JsonMapping = identifier.get("custom") or {}  # type: ignore[assignment]
        document_id = str(custom.get("id", ""))
        if _is_file_id(document_id):
            return document_id
        return _encode_document_id("c", data_source_id, document_id)
    s3: JsonMapping = identifier.get("s3") or {}  # type: ignore[assignment]
    return _encode_document_id("s", data_source_id, str(s3.get("uri", "")))


def document_target(file_id: str) -> tuple[str, dict[str, Any]]:
    """Return where a file identifier addresses its document, and as what.

    Args:
        file_id: The file identifier the request carried.

    Returns:
        ``(data_source_id, identifier)``, the former empty when the document
        belongs to the store's own data source.

    Raises:
        ApiError: When the identifier does not decode, or names something that
            is not a data source (404, as an unknown file).
    """
    if not _DOCUMENT_ID_RE(file_id):
        return "", {"dataSourceType": "CUSTOM", "custom": {"id": file_id}}
    encoded = file_id.removeprefix(_DOCUMENT_ID_PREFIX)
    try:
        decoded = urlsafe_b64decode(f"{encoded}{'=' * (-len(encoded) % 4)}").decode()
    except BinasciiError, UnicodeDecodeError:
        raise_not_found("file", file_id)
    kind, _, rest = decoded.partition(":")
    data_source_id, _, value = rest.partition(":")
    # The identifier is the caller's to forge, and what it names reaches the
    # service: anything but a data source addresses no document of this store.
    if data_source_id and not _DATA_SOURCE_ID_RE(data_source_id):
        raise_not_found("file", file_id)
    if kind == "s":
        return data_source_id, {"dataSourceType": "S3", "s3": {"uri": value}}
    return data_source_id, {"dataSourceType": "CUSTOM", "custom": {"id": value}}


def _encode_document_id(kind: str, data_source_id: str, value: str) -> str:
    """Return the opaque file identifier carrying one document's location.

    Args:
        kind: ``"s"`` for an object of a bucket, ``"c"`` for anything else.
        data_source_id: The data source holding it, or ``""`` for the store's own.
        value: The location itself.

    Returns:
        The file identifier.
    """
    encoded = urlsafe_b64encode(f"{kind}:{data_source_id}:{value}".encode())
    return f"{_DOCUMENT_ID_PREFIX}{encoded.decode().rstrip('=')}"


def _is_file_id(value: str) -> bool:
    """Whether *value* is an uploaded file's own identifier."""
    return value.startswith(("file-", "file_"))


def translate_filter(search_filter: SearchFilter | JsonMapping) -> dict[str, Any]:
    """Translate an upstream search filter into a retrieval filter.

    Args:
        search_filter: The filter as the API received it. A filter nested more
            than one level deep arrives as a plain mapping and is validated here.

    Returns:
        The equivalent retrieval filter.

    Raises:
        RequestValidationError: When a nested filter is not a valid filter.
    """
    node = parse_filter(search_filter)
    if isinstance(node, CompoundFilter):
        members = [translate_filter(inner) for inner in node.filters]
        # A combination needs at least two members, so a single one is itself.
        if len(members) == 1:
            return members[0]
        return {_FILTER_COMBINATORS[node.type]: members}
    return {_FILTER_OPERATORS[node.type]: {"key": node.key, "value": node.value}}


def agent_client() -> Any:  # noqa: ANN401 - the document operations are untyped
    """Return the client managing the knowledge base's documents.

    The knowledge bases are addressed in one region: they are regional
    resources, and a knowledge base absent from a region is not the same
    knowledge base served from another.
    """
    return get_client("bedrock-agent", SETTINGS.aws_bedrock_regions[0])


def runtime_client() -> Any:  # noqa: ANN401 - kept symmetrical with the above
    """Return the client the knowledge base is queried through."""
    return get_client("bedrock-agent-runtime", SETTINGS.aws_bedrock_regions[0])


def _guard(*actions: str) -> AbstractContextManager[None]:
    """Answer a denied knowledge base call as the Vector Stores API being unavailable.

    Args:
        *actions: The ``bedrock`` actions the guarded call needs.

    Returns:
        The guard wrapping the call.
    """
    permissions = ", ".join(f"bedrock:{action}" for action in actions)
    return feature_unavailable_guard(
        FEATURE,
        missing=(
            f"{permissions} on the knowledge bases set in "
            "'aws_bedrock_knowledge_base_ids'"
        ),
    )


def _attribute_value(value: JsonMapping) -> AttributeValue | None:
    """Return the attribute one document metadata entry holds, when it holds one."""
    match value.get("type"):
        case "STRING":
            return str(value.get("stringValue", ""))
        case "NUMBER":
            return float(value.get("numberValue", 0))  # type: ignore[arg-type]
        case "BOOLEAN":
            return bool(value.get("booleanValue", False))
        case _:
            return None


def _to_attribute(value: AttributeValue) -> dict[str, Any]:
    """Return the document metadata entry carrying one attribute."""
    if isinstance(value, bool):
        return {"type": "BOOLEAN", "booleanValue": value}
    if isinstance(value, str):
        return {"type": "STRING", "stringValue": value}
    return {"type": "NUMBER", "numberValue": float(value)}


def _caller_attributes(metadata: JsonMapping) -> Attributes:
    """Return the caller attributes a retrieved passage's metadata carries.

    The two generations name their own metadata differently, and neither is the
    caller's: what the service wrote is left out whichever prefix it used.
    """
    attributes: Attributes = {}
    for key, value in metadata.items():
        if key == _FILENAME_KEY or key.startswith(_SERVICE_METADATA_PREFIXES):
            continue
        if isinstance(value, bool | float | int | str):
            attributes[key] = value if isinstance(value, bool | str) else float(value)
    return attributes


def _filename_of(result: JsonMapping, metadata: JsonMapping, file_id: str) -> str:
    """Return the name reported for the file a passage comes from.

    Args:
        result: The retrieval result.
        metadata: Its metadata.
        file_id: The file identifier the passage is reported under.

    Returns:
        The file name, falling back to the identifier when the document carries
        none.
    """
    for key in (_FILENAME_KEY, *_SERVICE_URI_KEYS):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value.rsplit("/", 1)[-1]
    location: JsonMapping = result.get("location") or {}  # type: ignore[assignment]
    s3: JsonMapping = location.get("s3Location") or {}  # type: ignore[assignment]
    uri = s3.get("uri")
    if isinstance(uri, str) and uri:
        return uri.rsplit("/", 1)[-1]
    return file_id


def _located_as(document_id: str) -> dict[str, Any]:
    """Return the document identifier addressing one reported location.

    A location holding an object URI is an object of a bucket, whatever the
    result reported its kind as: the managed generation reports a document
    synced from a bucket under the custom location, while the data source
    holding it addresses that same document by its URI. Taking the reported
    kind at face value builds an identifier the read path cannot resolve.

    Args:
        document_id: The location as the retrieval result reports it.

    Returns:
        The document identifier, as the service takes it.
    """
    if document_id.startswith(_OBJECT_URI_SCHEME):
        return {"dataSourceType": "S3", "s3": {"uri": document_id}}
    return {"dataSourceType": "CUSTOM", "custom": {"id": document_id}}


def _location_identifier(result: JsonMapping) -> dict[str, Any]:
    """Return the document identifier a retrieval result points at."""
    # The managed generation reports a synced object's location as a browser
    # URL, which addresses nothing; only the document identifier carries the
    # object URI its data source holds it under.
    document_id = str(result.get("documentId") or "")
    if document_id.startswith(_OBJECT_URI_SCHEME):
        return _located_as(document_id)
    location: JsonMapping = result.get("location") or {}  # type: ignore[assignment]
    custom: JsonMapping = location.get("customDocumentLocation") or {}  # type: ignore[assignment]
    if custom.get("id"):
        return _located_as(str(custom["id"]))
    s3: JsonMapping = location.get("s3Location") or {}  # type: ignore[assignment]
    if s3.get("uri"):
        return {"dataSourceType": "S3", "s3": {"uri": str(s3["uri"])}}
    return _located_as(document_id)


def _result_data_source(metadata: JsonMapping) -> str:
    """Return the data source a retrieved passage's document belongs to.

    The two generations name the key differently, and a store's search spans
    every data source of the knowledge base, so which one answered is the only
    way back to the document itself.

    Args:
        metadata: The retrieval result's metadata.

    Returns:
        The data source identifier, or ``""`` when the result names none.
    """
    for key in _SERVICE_DATA_SOURCE_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def _addressed_the_wrong_kind(error: ClientError) -> bool:
    """Whether the service refused a document call for the kind it addressed.

    A document is addressed as an object of a bucket, as a document handed over
    directly, or as one of the other kinds a data source holds, and the service
    refuses the whole call when that is not the kind of the data source it names.

    Args:
        error: The error the document call raised.

    Returns:
        Whether that is what happened.
    """
    reported = error.response["Error"]
    return (
        reported["Code"] == _INVALID_CODE and _WRONG_KIND_MARKER in reported["Message"]
    )


def _report_wrong_kind(knowledge_base_id: str, data_source_id: str) -> None:
    """Tell the operator which data source holds no document of this API.

    The caller can change neither the knowledge base nor the data source the
    deployment allowlisted, so the cause is written where the person who can is
    the one reading it.

    Args:
        knowledge_base_id: The knowledge base the store addresses.
        data_source_id: The data source its documents were looked for in.
    """
    log_error_details(
        f"Data source '{data_source_id}' of knowledge base '{knowledge_base_id}' "
        "is not a custom one, so the Vector Stores API can neither attach "
        "documents to it nor address them one by one. Point the entry in "
        "'aws_bedrock_knowledge_base_ids' at a custom data source, as "
        f"'{knowledge_base_id}/<dataSourceId>', or serve that store for search "
        "only.",
        level="warning",
    )


def _passage_text(content: JsonMapping) -> str:
    """Return the text of one retrieved passage, or ``""`` when it carries none."""
    for key in ("text", "byteContent"):
        value = content.get(key)
        if isinstance(value, str) and value:
            return value
    for key, field in (("audio", "transcription"), ("video", "summary")):
        media: JsonMapping = content.get(key) or {}  # type: ignore[assignment]
        value = media.get(field)
        if isinstance(value, str) and value:
            return value
    return ""


def _document_record(details: JsonMapping) -> FileRecord:
    """Return the file record one document's details answer with.

    ``usage_bytes`` and the chunking are left unset rather than invented: the
    knowledge base reports neither, and it cuts the passages itself.

    ``created_at`` is the ``updatedAt`` the service reports, which is the only
    time it gives for a document: the instant it last ingested it, and its
    creation for a document ingested once.  It is also what the listing is
    ordered on, so a document never reports a time contradicting its own row —
    and it answers the same question as a store of our own, where re-attaching a
    file likewise resets the moment it reports.

    Args:
        details: The document details as the service reports them.

    Returns:
        The file record.
    """
    identifier: JsonMapping = details.get("identifier") or {}  # type: ignore[assignment]
    reported = str(details.get("status", ""))
    status = _DOCUMENT_STATUS.get(reported, "in_progress")
    updated = details.get("updatedAt")
    last_error = None
    if status == "failed":
        # ``IGNORED`` is a status the service models but does not reach here: a
        # format this store does not index is refused before it is ever sent,
        # so a document that got far enough to fail is one the store said it
        # takes, and its failure is the store's rather than the file's type.
        code = "unsupported_file" if reported == "IGNORED" else "server_error"
        last_error = FileErrorRecord(
            code=code,  # type: ignore[arg-type]
            message=str(
                details.get("statusReason") or "The file could not be indexed."
            ),
        )
    return FileRecord(
        id=document_file_id(identifier, str(details.get("dataSourceId", ""))),
        created_at=int(updated.timestamp()) if updated is not None else 0,  # type: ignore[union-attr]
        status=status,  # type: ignore[arg-type]
        last_error=last_error,
        max_chunk_size_tokens=0,
    )


class KnowledgeBaseIndex:
    """A vector store served by a knowledge base the deployment was given."""

    __slots__ = ()

    @property
    def capabilities(self) -> IndexCapabilities:
        """What this backend can express.

        The declaration is the floor every knowledge base meets; a managed one
        additionally takes :data:`MANAGED_MEDIA_TYPES`, which the ingestion path
        allows against the store it is serving.
        """
        return CAPABILITIES

    def check_configured(self) -> None:
        """Raise when no knowledge base is addressable.

        Raises:
            FeatureUnavailableError: When the allowlist is empty (503).
        """
        if not SETTINGS.aws_bedrock_knowledge_base_ids:
            raise FeatureUnavailableError(
                FEATURE,
                "No knowledge base is allowlisted "
                "(aws_bedrock_knowledge_base_ids): no knowledge base is "
                "addressable as a vector store.",
            )

    def check_attributes(self, attributes: Attributes) -> None:
        """Reject attributes the store cannot keep searchable.

        Args:
            attributes: The caller-supplied attributes.

        Raises:
            ApiError: When they do not fit the per-file budget, or use the key
                the file name is stored under (400).
        """
        if not attributes:
            return
        if _FILENAME_KEY in attributes:
            msg = (
                f"'{_FILENAME_KEY}' is reserved on this vector store and cannot "
                "be used as an attribute key. Rename it."
            )
            raise ApiError(msg)
        size = len(to_json_bytes(dict(attributes)))
        if size > _MAX_ATTRIBUTE_BYTES:
            msg = (
                f"The 'attributes' of this file take {size} bytes, above the "
                f"{_MAX_ATTRIBUTE_BYTES}-byte limit for searchable attributes. "
                "Use fewer keys, or shorter values."
            )
            raise ApiError(msg)

    def refuse(self, action: str) -> NoReturn:
        """Raise the 400 answering what an externally-owned store cannot do.

        Args:
            action: What the request asked for, as the caller phrased it.

        Raises:
            ApiError: Always (400).
        """
        msg = f"This vector store is managed outside this server, so {action}"
        raise ApiError(msg)

    async def read_store(self, store_id: str) -> StoreRecord:
        """Return what the knowledge base reports about itself.

        Args:
            store_id: A validated vector store identifier.

        Returns:
            The store as the API answers with it.

        Raises:
            ApiError: When the knowledge base does not exist (404).
        """
        return _store_record(store_id, await _describe(check_allowlisted(store_id)))

    async def list_stores(self) -> list[StoreRecord]:
        """Return every allowlisted knowledge base, skipping the unreadable ones.

        A knowledge base that was deleted, or that this deployment may not read,
        is left out rather than failing the listing of the others.

        Returns:
            The store records, oldest first.
        """
        described = await gather_bounded(
            [_describe(entry, missing_is_none=True) for entry in allowlist()],
            _CALL_WAVE,
        )
        records = [
            _store_record(store_id_of(str(entry["knowledgeBaseId"])), entry)
            for entry in described
            if entry is not None
        ]
        records.sort(key=lambda record: (record.created_at, record.id))
        return records

    async def attach_documents(
        self, store_id: str, pending: Sequence[PendingFile]
    ) -> list[FileRecord]:
        """Ingest *pending* files into the knowledge base as documents.

        Args:
            store_id: A validated vector store identifier.
            pending: The files to ingest.

        Returns:
            The file records, in the order of *pending*.

        Raises:
            ApiError: When a file does not exist (404), the knowledge base does
                not index it as it stands, or the store keeps a corpus it takes
                no document into (400).
        """
        knowledge_base_id = check_allowlisted(store_id)
        described = await _describe(knowledge_base_id)
        data_source_id = await _data_source_id(knowledge_base_id)
        accepted = _ingested_media_types(described)
        documents = [await _document(entry, accepted) for entry in pending]
        details: list[JsonMapping] = []
        for start in range(0, len(documents), _DOCUMENTS_BATCH):
            try:
                with _guard("IngestKnowledgeBaseDocuments"):
                    response = await agent_client().ingest_knowledge_base_documents(
                        knowledgeBaseId=knowledge_base_id,
                        dataSourceId=data_source_id,
                        documents=documents[start : start + _DOCUMENTS_BATCH],
                    )
            except ClientError as error:
                if not _addressed_the_wrong_kind(error):
                    raise
                _report_wrong_kind(knowledge_base_id, data_source_id)
                self.refuse(
                    "files cannot be attached to it: its corpus is kept up to "
                    "date where the store is. Attach the file to a vector "
                    "store this server owns, or ask the administrator for one "
                    "that accepts uploads."
                )
            details.extend(response.get("documentDetails", ()))
        # Answered in the order the files were given, whatever order they settle in.
        records = {record.id: record for record in map(_document_record, details)}
        return [
            records.get(entry.file_id)
            or FileRecord(
                id=entry.file_id,
                created_at=now_utc_timestamp(),
                max_chunk_size_tokens=0,
            )
            for entry in pending
        ]

    async def list_documents(
        self,
        store_id: str,
        *,
        after: str,
        before: str,
        limit: int,
        order: str,
        status: str,
    ) -> tuple[list[FileRecord], bool]:
        """List the documents of the knowledge base's data source, newest first.

        The service lists documents in its own order, and its own cursor does
        not survive the reordering, so the listing is read whole — a hundred
        documents per call, which is all the managed generation accepts — and
        the page is cut here, on the time each document reports.

        Args:
            store_id: A validated vector store identifier.
            after: Return the documents following this identifier.
            before: Return the page ending immediately before this identifier.
            limit: Maximum records to return.
            order: ``"asc"`` or ``"desc"``.
            status: Keep only documents with this status, or ``""`` for all.

        Returns:
            ``(records, has_more)``.
        """
        knowledge_base_id = check_allowlisted(store_id)
        data_source_id = await _data_source_id(knowledge_base_id)
        client = agent_client()
        details: list[JsonMapping] = []
        token = ""
        with _guard("ListKnowledgeBaseDocuments"):
            while len(details) < _LIST_SCAN_MAX:
                response = await client.list_knowledge_base_documents(
                    knowledgeBaseId=knowledge_base_id,
                    dataSourceId=data_source_id,
                    maxResults=_LIST_PAGE_MAX,
                    **({"nextToken": token} if token else {}),
                )
                details.extend(response.get("documentDetails", ()))
                token = response.get("nextToken", "")
                if not token:
                    break
        # A deleted document is kept in the listing as a tombstone: it is gone,
        # and reporting it as a file still settling would never resolve.
        records = [
            _document_record(entry)
            for entry in details
            if entry.get("status") != _ABSENT_STATUS
        ]
        if status:
            records = [record for record in records if record.status == status]
        return page_records(
            records, after=after, before=before, limit=limit, order=order
        )

    async def read_document(self, store_id: str, file_id: str) -> FileRecord:
        """Return one document of the knowledge base.

        A search spans every data source of the knowledge base, so a passage may
        come from the corpus behind the store rather than from the documents
        attached here; its identifier carries where it lives, and it is read
        there so that searching and reading round trip.

        Args:
            store_id: A validated vector store identifier.
            file_id: The document to read.

        Returns:
            The file record.

        Raises:
            ApiError: When the knowledge base holds no such document (404).
        """
        knowledge_base_id = check_allowlisted(store_id)
        data_source_id, identifier = document_target(file_id)
        data_source_id = data_source_id or await _data_source_id(knowledge_base_id)
        try:
            with _guard("GetKnowledgeBaseDocuments"):
                response = await agent_client().get_knowledge_base_documents(
                    knowledgeBaseId=knowledge_base_id,
                    dataSourceId=data_source_id,
                    documentIdentifiers=[identifier],
                )
        except ClientError as error:
            if _addressed_the_wrong_kind(error):
                _report_wrong_kind(knowledge_base_id, data_source_id)
            elif error.response["Error"]["Code"] != _MISSING_CODE:
                raise
            # Neither the kind nor the data source addresses a document of this
            # store, and an identifier that addresses none names no file of it.
            raise_not_found("file", file_id)
        found = [
            entry
            for entry in response.get("documentDetails", ())
            if entry.get("status") != _ABSENT_STATUS
        ]
        if not found:
            raise_not_found("file", file_id)
        return _document_record(found[0])

    async def delete_document(self, store_id: str, file_id: str) -> None:
        """Remove one document from the knowledge base.

        Only the documents this store attaches are removable. One of the corpus
        behind it is readable and never deletable: it was put there by something
        that maintains it, and removing it here would take it out of a corpus
        this server does not own.

        Args:
            store_id: A validated vector store identifier.
            file_id: The document to remove.

        Raises:
            ApiError: When the knowledge base holds no such document (404), or
                the document belongs to the corpus behind the store (400).
        """
        knowledge_base_id = check_allowlisted(store_id)
        await self.read_document(store_id, file_id)
        holding, identifier = document_target(file_id)
        data_source_id = await _data_source_id(knowledge_base_id)
        if holding and holding != data_source_id:
            self.refuse(
                "a document of the corpus behind it cannot be removed: it is "
                "maintained where that corpus comes from."
            )
        with _guard("DeleteKnowledgeBaseDocuments"):
            await agent_client().delete_knowledge_base_documents(
                knowledgeBaseId=knowledge_base_id,
                dataSourceId=data_source_id,
                documentIdentifiers=[identifier],
            )

    async def query_text(
        self,
        store_id: str,
        queries: Sequence[str],
        *,
        max_results: int,
        search_filter: SearchFilter | None,
    ) -> list[VectorMatch]:
        """Return the passages of the knowledge base closest to *queries*.

        Args:
            store_id: A validated vector store identifier.
            queries: One or more query texts.
            max_results: Maximum matches to return per query.
            search_filter: Restriction over the documents' metadata, if any.

        Returns:
            The matches of every query, in no particular order.

        Raises:
            ApiError: When a query is longer than the store accepts (400).
        """
        knowledge_base_id = check_allowlisted(store_id)
        described = await _describe(knowledge_base_id)
        kind = _generation(described)
        _check_queries(queries, kind)
        configuration: dict[str, Any] = {
            "numberOfResults": min(max_results, _RESULTS_MAX)
        }
        if search_filter is not None:
            configuration["filter"] = translate_filter(search_filter)
        key = (
            "managedSearchConfiguration"
            if kind == "MANAGED"
            else "vectorSearchConfiguration"
        )
        client = runtime_client()
        with _guard("Retrieve"):
            responses = await gather_bounded(
                [
                    client.retrieve(
                        knowledgeBaseId=knowledge_base_id,
                        retrievalQuery={"text": query},
                        retrievalConfiguration={key: configuration},
                    )
                    for query in queries
                ],
                _CALL_WAVE,
            )
        if kind == "MANAGED":
            # The only generation AWS bills the retrieval itself for.
            record_knowledge_base_usage(
                len(queries), region=SETTINGS.aws_bedrock_regions[0]
            )
        return [
            match
            for response in responses
            for match in map(_to_match, response.get("retrievalResults", ()))
            if match is not None
        ]

    async def create_index(self, store_id: str, *, dimensions: int) -> None:
        """Refuse to create a knowledge base.

        Args:
            store_id: A validated vector store identifier.
            dimensions: Length of the vectors it would hold.

        Raises:
            ApiError: Always (400).
        """
        del store_id, dimensions
        self.refuse("it cannot be created here.")

    async def delete_index(self, store_id: str) -> None:
        """Refuse to delete a knowledge base.

        Args:
            store_id: A validated vector store identifier.

        Raises:
            ApiError: Always (400).
        """
        del store_id
        self.refuse("it cannot be deleted here.")

    async def put_vectors(
        self, store_id: str, vectors: AsyncIterable[IndexVector]
    ) -> None:
        """Refuse a write of passages this store cuts itself.

        Args:
            store_id: A validated vector store identifier.
            vectors: The passages that would be written.

        Raises:
            ApiError: Always (400).
        """
        del store_id, vectors
        self.refuse("its passages cannot be written directly.")

    async def get_vectors(
        self, store_id: str, keys: Sequence[str], *, with_embeddings: bool
    ) -> list[IndexVector]:
        """Refuse a read of passages this store does not address individually.

        Args:
            store_id: A validated vector store identifier.
            keys: The passage keys that would be read.
            with_embeddings: Whether the vectors themselves are needed.

        Raises:
            ApiError: Always (400).
        """
        del store_id, keys, with_embeddings
        self.refuse("its passages cannot be listed.")

    async def delete_vectors(self, store_id: str, keys: Sequence[str]) -> None:
        """Refuse a delete of passages this store does not address individually.

        Args:
            store_id: A validated vector store identifier.
            keys: The passage keys that would be removed.

        Raises:
            ApiError: Always (400).
        """
        del store_id, keys
        self.refuse("its passages cannot be removed one by one.")

    async def query(
        self,
        store_id: str,
        embeddings: Sequence[Sequence[float]],
        *,
        max_results: int,
        search_filter: SearchFilter | None,
    ) -> list[VectorMatch]:
        """Refuse a search by vector: this store embeds the query itself.

        Args:
            store_id: A validated vector store identifier.
            embeddings: One vector per query text.
            max_results: Maximum matches to return per query.
            search_filter: Restriction over the documents' metadata, if any.

        Raises:
            ApiError: Always (400).
        """
        del store_id, embeddings, max_results, search_filter
        self.refuse("it cannot be searched by vector.")


def _generation(described: JsonMapping) -> str:
    """Return the kind of a described knowledge base, or ``""`` when it states none.

    Never defaulted to a kind: which one it is decides how it is searched, and
    guessing is what serves a knowledge base this backend does not handle.
    """
    configuration: JsonMapping = described.get("knowledgeBaseConfiguration") or {}  # type: ignore[assignment]
    return str(configuration.get("type", ""))


def _unserved_kind_detail(knowledge_base_id: str, kind: str) -> str:
    """Return what the operator reads about a knowledge base of an unserved kind.

    Args:
        knowledge_base_id: The knowledge base the store addresses.
        kind: The kind the service reports it as, or ``""`` when it states none.

    Returns:
        The detail, naming both the kind found and the kinds served.
    """
    served = " and ".join(SERVED_KINDS)
    return (
        f"Knowledge base '{knowledge_base_id}' is of kind '{kind or 'unknown'}', "
        f"which the Vector Stores API does not serve: only {served} knowledge "
        "bases hold the passages a vector store answers with, and a retrieval "
        "against any other kind comes back with nothing this server can read. "
        f"Point that entry of 'aws_bedrock_knowledge_base_ids' at a {served} "
        "knowledge base, or remove it."
    )


def _check_served_kind(knowledge_base_id: str, described: JsonMapping) -> None:
    """Refuse a knowledge base of a kind this backend does not serve.

    Refused where the kind is known rather than at the retrieval: the service
    takes a vector search against a structured knowledge base without
    complaining and answers it with rows, which carry no passage text, so the
    store would answer every search ``200`` with nothing in it, forever.

    Args:
        knowledge_base_id: The knowledge base the store addresses.
        described: The knowledge base as the service describes it.

    Raises:
        FeatureUnavailableError: When it is not one of :data:`SERVED_KINDS` (503).
    """
    kind = _generation(described)
    if kind not in SERVED_KINDS:
        raise FeatureUnavailableError(
            FEATURE, _unserved_kind_detail(knowledge_base_id, kind)
        )


async def verify_knowledge_bases(start_event: EventLog) -> None:
    """Check every allowlisted knowledge base at startup, so its kind is known there.

    Reported and never fatal, for the two deployments that would otherwise stop
    booting on it: one whose role predates ``bedrock:GetKnowledgeBase``, and one
    whose knowledge base is a moment away from existing. Each store checks
    itself again on its own first request, so nothing is served on the strength
    of this having passed.

    Args:
        start_event: Startup log event the findings are reported on.
    """
    if not (entries := list(allowlist())):
        return
    for warning in await gather_bounded(
        [_verify_knowledge_base(entry) for entry in entries], _CALL_WAVE
    ):
        if warning:
            add_server_warning(start_event, warning)


async def _verify_knowledge_base(knowledge_base_id: str) -> str:
    """Return what the operator must be told about one allowlisted entry.

    Args:
        knowledge_base_id: The allowlisted knowledge base.

    Returns:
        The warning, or ``""`` when it is a knowledge base this server serves.
    """
    try:
        described = await _get_knowledge_base(knowledge_base_id)
    except (ApiError, BotoCoreError, ClientError) as exception:
        return (
            f"Knowledge base '{knowledge_base_id}' could not be read at startup "
            f"({type(exception).__name__}), so the kind it is stays unchecked: "
            "grant bedrock:GetKnowledgeBase on it to have that checked here "
            "rather than on the first request against the store."
        )
    kind = _generation(described)
    if kind in SERVED_KINDS:
        return ""
    return _unserved_kind_detail(knowledge_base_id, kind)


async def _get_knowledge_base(knowledge_base_id: str) -> Any:  # noqa: ANN401 - the description is an untyped service document
    """Return what the service reports about one knowledge base, unchecked.

    Args:
        knowledge_base_id: The knowledge base to describe.

    Returns:
        The knowledge base description.
    """
    with _guard("GetKnowledgeBase"):
        response = await agent_client().get_knowledge_base(
            knowledgeBaseId=knowledge_base_id
        )
    return response["knowledgeBase"]


def _ingested_media_types(described: JsonMapping) -> frozenset[str]:
    """Return the media types the described knowledge base indexes as they stand."""
    if _generation(described) == "MANAGED":
        return _DOCUMENT_MEDIA_TYPES | MANAGED_MEDIA_TYPES
    return _DOCUMENT_MEDIA_TYPES


def _embedding_model(described: JsonMapping) -> str:
    """Return the embedding model of a described knowledge base, when it names one."""
    configuration: JsonMapping = described.get("knowledgeBaseConfiguration") or {}  # type: ignore[assignment]
    for key in (
        "vectorKnowledgeBaseConfiguration",
        "managedKnowledgeBaseConfiguration",
    ):
        inner: JsonMapping = configuration.get(key) or {}  # type: ignore[assignment]
        arn = inner.get("embeddingModelArn")
        if isinstance(arn, str) and arn:
            return arn.rsplit("/", 1)[-1]
    return ""


def _store_record(store_id: str, described: JsonMapping) -> StoreRecord:
    """Return the store record a described knowledge base answers with.

    The file counts and ``usage_bytes`` stay at zero: the corpus is the
    customer's, and counting it would mean scanning it on every read.

    Args:
        store_id: The vector store identifier addressing it.
        described: The knowledge base as the service describes it.

    Returns:
        The store record.
    """
    created = described.get("createdAt")
    updated = described.get("updatedAt")
    return StoreRecord(
        id=store_id,
        created_at=int(created.timestamp()) if created is not None else 0,  # type: ignore[union-attr]
        last_active_at=(
            int(updated.timestamp()) if updated is not None else now_utc_timestamp()  # type: ignore[union-attr]
        ),
        name=str(described.get("name", "")),
        description=str(described.get("description", "")),
        embedding_model=_embedding_model(described),
        dimensions=0,
        external_status=(
            "completed" if described.get("status") == "ACTIVE" else "in_progress"
        ),
    )


async def _describe(knowledge_base_id: str, *, missing_is_none: bool = False) -> Any:  # noqa: ANN401 - the description is an untyped service document
    """Return what the service reports about one knowledge base.

    Args:
        knowledge_base_id: The knowledge base to describe.
        missing_is_none: Whether a knowledge base that cannot be read, or that
            is not one this server serves, is reported as ``None`` rather than
            raising — which is how a listing leaves it out instead of failing.

    Returns:
        The knowledge base description, or ``None``.

    Raises:
        ApiError: When the knowledge base does not exist (404, worded exactly
            as an identifier this deployment was never given is), or is of a
            kind this backend does not serve (503).
    """
    try:
        described = await _get_knowledge_base(knowledge_base_id)
        _check_served_kind(knowledge_base_id, described)
    except ClientError as exc:
        if missing_is_none:
            return None
        if exc.response["Error"]["Code"] != _MISSING_CODE:
            raise
        # Never the raw error: a knowledge base that is gone and one this
        # deployment was not given answer with the very same 404.
        raise_not_found("vector store", store_id_of(knowledge_base_id))
    except ApiError:
        if missing_is_none:
            return None
        raise
    return described


async def _data_source_id(knowledge_base_id: str) -> str:
    """Return the data source the knowledge base's documents are held in.

    Args:
        knowledge_base_id: The knowledge base the store addresses.

    Returns:
        The data source identifier.

    Raises:
        FeatureUnavailableError: When the knowledge base has no single data
            source and the deployment did not name one (503).
    """
    configured = allowlist().get(knowledge_base_id, "")
    if configured:
        return configured
    with _guard("ListDataSources"):
        response = await agent_client().list_data_sources(
            knowledgeBaseId=knowledge_base_id, maxResults=_DOCUMENTS_BATCH
        )
    summaries = list(response.get("dataSourceSummaries", ()))
    if len(summaries) != 1:
        raise FeatureUnavailableError(
            FEATURE,
            f"Knowledge base '{knowledge_base_id}' has {len(summaries)} data "
            "sources, so the one its documents belong to is ambiguous: write it "
            "in aws_bedrock_knowledge_base_ids as "
            f"'{knowledge_base_id}/<dataSourceId>'.",
        )
    return str(summaries[0]["dataSourceId"])


def _check_queries(queries: Sequence[str], kind: str) -> None:
    """Refuse a query longer than the store accepts, rather than truncating it.

    Args:
        queries: The query texts.
        kind: The knowledge base generation.

    Raises:
        ApiError: When one query is over the limit (400).
    """
    limit = _QUERY_CHARACTERS_MAX.get(kind, _QUERY_CHARACTERS_MAX["VECTOR"])
    for query in queries:
        if len(query) > limit:
            msg = (
                f"The search query is {len(query)} characters long, above the "
                f"{limit} characters this vector store accepts. Shorten it."
            )
            raise ApiError(msg)


async def _document(pending: PendingFile, accepted: frozenset[str]) -> dict[str, Any]:
    """Return the document one pending file is ingested as.

    Args:
        pending: The file to ingest.
        accepted: The media types the knowledge base indexes as they stand.

    Returns:
        The document, as the service takes it.

    Raises:
        ApiError: When the file does not exist (404), or the knowledge base
            does not index it as it stands (400).
    """
    payload = parse_file_id(pending.file_id)
    source = await get_file(payload)
    stream, content_type = await get_file_content(payload)
    media_type = content_type.split(";", 1)[0].strip()
    body = bytearray()
    async for part in stream:
        body.extend(part)
        if len(body) > _MAX_DOCUMENT_BYTES:
            msg = (
                "The file is larger than this vector store accepts for "
                f"indexing ({_MAX_DOCUMENT_BYTES} bytes)."
            )
            raise ApiError(msg)
    content = _inline_content(bytes(body), media_type, accepted)
    attributes = [
        {"key": key, "value": _to_attribute(value)}
        for key, value in pending.attributes.items()
    ]
    attributes.append(
        {"key": _FILENAME_KEY, "value": _to_attribute(source.filename or payload)}
    )
    return {
        "content": {
            "dataSourceType": "CUSTOM",
            "custom": {
                "customDocumentIdentifier": {"id": pending.file_id},
                "sourceType": "IN_LINE",
                "inlineContent": content,
            },
        },
        "metadata": {"type": "IN_LINE_ATTRIBUTE", "inlineAttributes": attributes},
    }


def _inline_content(
    body: bytes, media_type: str, accepted: frozenset[str]
) -> dict[str, Any]:
    """Return the inline content one file is ingested as.

    Plain text goes in as text. A document format the store parses is sent as
    it stands, so its structure survives. Anything else is indexable only when
    its bytes turn out to be text.

    Args:
        body: The file content.
        media_type: The media type it was uploaded with.
        accepted: The media types the knowledge base indexes as they stand.

    Returns:
        The inline content, as the service takes it.

    Raises:
        ApiError: When the file holds nothing this store indexes (400).
    """
    if media_type not in CAPABILITIES.refused_media_types and (
        media_type in _PLAIN_TEXT_MEDIA_TYPES or media_type not in accepted
    ):
        try:
            text = body.decode()
        except UnicodeDecodeError:
            text = ""
        if text and "\x00" not in text:
            return {"type": "TEXT", "textContent": {"data": text}}
    if media_type in accepted:
        return {"type": "BYTE", "byteContent": {"mimeType": media_type, "data": body}}
    raise ApiError(unsupported_file_message(CAPABILITIES, ingested=accepted))


def _to_match(result: JsonMapping) -> VectorMatch | None:
    """Return the match one retrieval result carries, when it carries text.

    The score is reported exactly as the store measured it: it has no stated
    range, so rescaling it into a similarity would be inventing a number.

    Args:
        result: One retrieval result.

    Returns:
        The match, or ``None`` when the passage holds no text.
    """
    content: JsonMapping = result.get("content") or {}  # type: ignore[assignment]
    text = _passage_text(content)
    if not text:
        return None
    metadata: JsonMapping = result.get("metadata") or {}  # type: ignore[assignment]
    file_id = document_file_id(
        _location_identifier(result), _result_data_source(metadata)
    )
    digest = blake2b(text.encode(), digest_size=8).hexdigest()
    return VectorMatch(
        key=f"{file_id}#{digest}",
        score=float(result.get("score", 0.0)),  # type: ignore[arg-type]
        file_id=file_id,
        filename=_filename_of(result, metadata, file_id),
        text=text,
        attributes=_caller_attributes(metadata),
    )
