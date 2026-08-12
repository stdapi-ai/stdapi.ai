"""Shared media handling of the Amazon Nova chat model generations."""

from stdapi.input_file import BedrockMediaType, InlineMediaLimits

#: Media kinds Amazon Nova reads from storage instead of the request body.
NOVA_S3_LOCATION_MEDIA_TYPES: frozenset[BedrockMediaType] = frozenset(
    {"image", "document", "video"}
)

#: Amazon Nova refuses a single media block past 25,000,000 base64 bytes.
NOVA_INLINE_MEDIA_LIMITS = InlineMediaLimits(max_file_base64_size=25_000_000)
