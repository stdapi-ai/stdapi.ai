"""Amazon Transcribe Medical model implementation.

Medical transcription runs on the standard Transcribe plumbing -- the same
bucket, region failover, polling, cleanup and transcript parsing -- through
its own operations. It transcribes US English only, writes no subtitles, and
takes a specialty other than primary care on a live session only.
"""

from typing import TYPE_CHECKING, Any, Final, Literal

from aws_sdk_transcribe_streaming.models import (
    StartMedicalStreamTranscriptionInput,
    StartStreamTranscriptionInput,
)

from stdapi.api_errors import ApiError, InvalidLanguageFormatError
from stdapi.config import SETTINGS
from stdapi.models.audio import AudioModelBase, unsupported_response_format
from stdapi.models.audio.amazon_transcribe import (
    _STREAM_SAMPLE_RATE,
    AWS_TRANSCRIBE_MEDICAL_MODEL_ID,
    _job_settings,
    _JobOperations,
    _TranscribeSettingsParams,
)
from stdapi.models.audio.amazon_transcribe import AudioModel as TranscribeAudioModel
from stdapi.monitoring import build_metadata
from stdapi.types.openai_audio import SUBTITLE_FORMATS
from stdapi.utils import format_language_code, validation_error_handler

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping

    from types_aiobotocore_transcribe.client import TranscribeServiceClient
    from types_aiobotocore_transcribe.type_defs import (
        StartMedicalTranscriptionJobRequestTypeDef,
    )

    from stdapi.input_file import InputFile
    from stdapi.models import ModelDetails
    from stdapi.models.audio.amazon_transcribe import TranscribeJobData
    from stdapi.types import JsonMapping
    from stdapi.types.openai_audio import (
        AudioResponseFormat,
        AudioTimestampGranularities,
    )

#: The one language medical transcription transcribes.
_MEDICAL_LANGUAGE_CODE: Final = "en-US"

#: Specialty a transcription job accepts, and a live session defaults to.
_PRIMARY_CARE: Final = "PRIMARYCARE"

#: Audio type when the request names none: the broader of the two.
_DEFAULT_TYPE: Final = "CONVERSATION"

#: Extra parameters a live session honours; any other one needs a job.
_STREAM_FIELDS: Final = frozenset({"Specialty", "Type", "VocabularyName"})


class _TranscribeMedicalExtraParams(_TranscribeSettingsParams):
    """Supported extra parameters for medical transcription.

    ``Settings``' sub-fields are flattened to the top level, as for the
    standard model; ``Type`` and ``Specialty`` keep AWS's own names.
    """

    Specialty: (
        Literal[
            "PRIMARYCARE", "CARDIOLOGY", "NEUROLOGY", "ONCOLOGY", "RADIOLOGY", "UROLOGY"
        ]
        | None
    ) = None
    Type: Literal["CONVERSATION", "DICTATION"] | None = None


def _parse_extra_params(
    extra_params: JsonMapping | None,
) -> _TranscribeMedicalExtraParams | None:
    """Validate a request's extra parameters.

    Args:
        extra_params: Raw extra parameters from the request.

    Returns:
        The validated parameters, or None when there are none.

    Raises:
        RequestValidationError: A parameter is unknown or invalid.
    """
    if not extra_params:
        return None
    with validation_error_handler():
        return _TranscribeMedicalExtraParams(**extra_params)  # type: ignore[arg-type]


def _validate_language(language: str | None, languages: list[str] | None) -> None:
    """Refuse any expected language other than US English.

    Args:
        language: Optional language code.
        languages: Optional expected input language codes.

    Raises:
        InvalidLanguageFormatError: A language other than US English is named.
    """
    for param, codes in (
        ("language", (language,) if language else ()),
        ("languages", languages or ()),
    ):
        for code in codes:
            if format_language_code(code) != _MEDICAL_LANGUAGE_CODE:
                error = InvalidLanguageFormatError(
                    f"Language '{code}' is not supported by the model. Medical "
                    "transcription is available in US English ('en') only."
                )
                error.param = param
                raise error


def _build_medical_job_params(
    job_id: str,
    s3_bucket: str,
    _language: str | None,
    response_format: str,
    extra: _TranscribeMedicalExtraParams | None,
    _languages: list[str] | None,
) -> StartMedicalTranscriptionJobRequestTypeDef:
    """Build a medical transcription job's parameters.

    Args:
        job_id: Unique job identifier.
        s3_bucket: S3 bucket name, for both the audio and the transcript.
        _language: Unused: the language is always US English.
        response_format: Response format for transcription.
        extra: Optional extra parameters.
        _languages: Unused: the language is always US English.

    Returns:
        Job parameters for AWS Transcribe Medical.
    """
    s3_prefix = SETTINGS.aws_s3_tmp_prefix
    job_params: StartMedicalTranscriptionJobRequestTypeDef = {
        "MedicalTranscriptionJobName": job_id,
        "LanguageCode": _MEDICAL_LANGUAGE_CODE,
        "Media": {"MediaFileUri": f"s3://{s3_bucket}/{s3_prefix}{job_id}/input"},
        "OutputBucketName": s3_bucket,
        "OutputKey": f"{s3_prefix}{job_id}/output.json",
        "Specialty": _PRIMARY_CARE,
        "Type": (extra.Type if extra is not None else None) or _DEFAULT_TYPE,
        "Tags": [{"Key": k, "Value": v} for k, v in build_metadata(apn=True).items()],
    }
    if key_arn := SETTINGS.aws_transcribe_output_encryption_key_arn:
        job_params["OutputEncryptionKMSKeyId"] = key_arn
        job_params["KMSEncryptionContext"] = build_metadata()
    if settings := _job_settings(response_format, extra):
        job_params["Settings"] = settings  # type: ignore[typeddict-item]
    return job_params


async def _describe_medical_job(
    transcribe: TranscribeServiceClient, job_name: str
) -> Mapping[str, Any]:
    """Return a medical transcription job's description.

    Args:
        transcribe: Transcribe client.
        job_name: The job's name.

    Returns:
        The job's status, failure reason and output locations.
    """
    return (
        await transcribe.get_medical_transcription_job(
            MedicalTranscriptionJobName=job_name
        )
    )["MedicalTranscriptionJob"]


#: The calls a medical transcription job goes through.
_MEDICAL_TRANSCRIPTION_JOB: Final = _JobOperations(
    feature="Medical transcription",
    start_action="transcribe:StartMedicalTranscriptionJob",
    build=_build_medical_job_params,
    start=lambda transcribe, params: transcribe.start_medical_transcription_job(
        **params
    ),
    describe=_describe_medical_job,
    delete=lambda transcribe, name: transcribe.delete_medical_transcription_job(
        MedicalTranscriptionJobName=name
    ),
)


class AudioModel(TranscribeAudioModel):
    """Amazon Transcribe Medical audio model implementation (transcription only)."""

    __slots__ = ()

    MATCHER = AWS_TRANSCRIBE_MEDICAL_MODEL_ID

    SUPPORTED_RESPONSES_FORMATS = frozenset(
        {"json", "text", "verbose_json", "diarized_json"}
    )

    USAGE_MODEL_ID = AWS_TRANSCRIBE_MEDICAL_MODEL_ID
    EXTRA_PARAMS = _TranscribeMedicalExtraParams
    JOB_OPERATIONS = _MEDICAL_TRANSCRIPTION_JOB

    # The transcript is US English already: there is nothing to translate.
    stt_translate = AudioModelBase.stt_translate

    @classmethod
    def get_aliases(
        cls,
        all_models: dict[str, ModelDetails],  # noqa: ARG003
    ) -> dict[str, str]:
        """Return no alias: no upstream model transcribes medical audio.

        Args:
            all_models: All available models keyed by model ID.

        Returns:
            An empty dict.
        """
        return {}

    @classmethod
    def _validate_response_formats(
        cls,
        value: AudioResponseFormat,
        timestamp_granularities: list[AudioTimestampGranularities] | None = None,
    ) -> None:
        """Validate the response format, pointing subtitles at timed segments.

        Args:
            value: The requested response format.
            timestamp_granularities: Timestamp granularities.

        Raises:
            ApiError: The format is not supported by this model.
        """
        if value in SUBTITLE_FORMATS:
            msg = (
                f"Response format '{value}' is not supported by this model. "
                "Request 'verbose_json' for timed segments."
            )
            raise unsupported_response_format(msg)
        super()._validate_response_formats(value, timestamp_granularities)

    @classmethod
    def _live_request(
        cls,
        language: str | None,
        languages: list[str] | None,
        extra_params: JsonMapping | None,
        *,
        diarize: bool,
    ) -> StartMedicalStreamTranscriptionInput | None:
        """Build the request a live medical session serves this transcription with.

        Args:
            language: Optional language code.
            languages: Optional expected input language codes.
            extra_params: Optional extra parameters.
            diarize: Whether each word must be attributed to a speaker.

        Returns:
            The session request, or None when a parameter only a transcription
            job honours was set.
        """
        _validate_language(language, languages)
        extra = _parse_extra_params(extra_params)
        if extra is not None and set(extra.model_dump(exclude_none=True)).difference(
            _STREAM_FIELDS
        ):
            return None
        return StartMedicalStreamTranscriptionInput(
            language_code=_MEDICAL_LANGUAGE_CODE,
            media_encoding="pcm",
            media_sample_rate_hertz=_STREAM_SAMPLE_RATE,
            specialty=(extra.Specialty if extra is not None else None) or _PRIMARY_CARE,
            type=(extra.Type if extra is not None else None) or _DEFAULT_TYPE,
            show_speaker_label=diarize,
            vocabulary_name=extra.VocabularyName if extra is not None else None,
        )

    @classmethod
    def _job_serves(cls, extra_params: JsonMapping | None) -> bool:
        """Whether a medical job can serve the request: primary care only.

        Args:
            extra_params: Optional extra parameters.

        Returns:
            False when another specialty was asked for, which only a live
            session transcribes.
        """
        extra = _parse_extra_params(extra_params)
        return extra is None or extra.Specialty in {None, _PRIMARY_CARE}

    @staticmethod
    def _open_live_session(
        client: Any,  # noqa: ANN401
        request: StartStreamTranscriptionInput | StartMedicalStreamTranscriptionInput,
    ) -> Awaitable[Any]:
        """Open a live medical session.

        Args:
            client: The region's bidirectional Transcribe client.
            request: The session request, as :meth:`_live_request` built it.

        Returns:
            The SDK's pending duplex stream.
        """
        return client.start_medical_stream_transcription(request)  # type: ignore[no-any-return]

    async def _transcribe(
        self,
        audio_content: InputFile,
        response_format: AudioResponseFormat,
        language: str | None = None,
        prompt: str | None = None,
        temperature: float | None = None,
        extra_params: JsonMapping | None = None,
        languages: list[str] | None = None,
        *,
        logprobs: bool = False,
    ) -> TranscribeJobData:
        """Transcribe the audio as a medical transcription job.

        Args:
            audio_content: Audio file content file.
            response_format: Format for the output response.
            language: Optional language code; US English only.
            prompt: Optional prompt for transcription.
            temperature: Optional temperature for transcription.
            extra_params: Optional extra parameters.
            languages: Optional expected input language codes; US English only.
            logprobs: If true, return log probabilities.

        Returns:
            Raw transcription response, with its language set: a medical
            transcript names none.

        Raises:
            ApiError: A specialty other than primary care was asked for, which
                only a live session transcribes.
            InvalidLanguageFormatError: A language other than US English is named.
        """
        _validate_language(language, languages)
        extra = _parse_extra_params(extra_params)
        if extra is not None and extra.Specialty not in {None, _PRIMARY_CARE}:
            error = ApiError(
                f"Specialty '{extra.Specialty}' needs stream=true, without the "
                "ShowSpeakerLabels, MaxSpeakerLabels, ChannelIdentification, "
                "ShowAlternatives or MaxAlternatives extra parameters. Omit "
                "Specialty to transcribe as primary care."
            )
            error.param = "Specialty"
            raise error
        transcript_data = await super()._transcribe(
            audio_content,
            response_format,
            language,
            prompt,
            temperature,
            extra_params,
            languages,
            logprobs=logprobs,
        )
        transcript_data.setdefault("language_code", _MEDICAL_LANGUAGE_CODE)
        return transcript_data
