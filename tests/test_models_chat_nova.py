"""Converse request building and code-interpreter mapping for Amazon Nova 2 (no AWS calls).

Nova 2 rejects a Converse request that combines ``inferenceConfig.maxTokens``
with ``reasoningConfig.maxReasoningEffort: high``, so the model class drops the
token limit and records a warning in the request log instead of letting Bedrock
fail the call.  It also serves Anthropic's ``code_execution`` server tool through
its own ``nova_code_interpreter`` system tool, which needs a translation in both
directions.

Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_Converse.html
     https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
     stdapi/models/chat/amazon_nova_2.py:ChatModel._prepare_converse_request
"""

from asyncio import gather
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest

from stdapi.api_errors import ApiError
from stdapi.aws_bedrock import BEDROCK_PROMPT_VAR, BedrockPrompt
from stdapi.aws_s3 import S3Object
from stdapi.config import SETTINGS
from stdapi.input_file import (
    InlineMediaLimits,
    InputFile,
    plan_bedrock_media_transport,
    resolve_all_bedrock_content_blocks,
)
from stdapi.models import ModelDetails
from stdapi.models.chat.amazon_nova_2 import ChatModel
from stdapi.types.anthropic_messages import (
    CodeExecutionResultBlock,
    CodeExecutionResultBlockParam,
    CodeExecutionToolResultBlock,
    CodeExecutionToolResultBlockParam,
    CodeExecutionToolResultErrorParam,
    TextBlockParam,
)
from tests._helpers import red_png_b64
from tests.test_input_file import input_files  # noqa: F401

if TYPE_CHECKING:
    from openai import OpenAI
    from types_aiobotocore_bedrock.literals import RegionName
    from types_aiobotocore_bedrock_runtime.type_defs import (
        ContentBlockTypeDef,
        InferenceConfigurationTypeDef,
    )

    from stdapi.monitoring import EventLog

pytestmark = [pytest.mark.local, pytest.mark.usefixtures("request_log")]

#: Model this module drives; the cheapest Nova 2 generation.
_MODEL_ID = "amazon.nova-2-lite-v1:0"

#: Model instance used across tests; construction is side-effect free.
_MODEL = ChatModel(_MODEL_ID)


async def _stub_upload(*_args: object, **_kwargs: object) -> S3Object:
    """Stand in for staging an attachment, without touching AWS.

    Returns:
        A fixed object reference.
    """
    return S3Object(bucket="a-bucket", key="staged")


def _model_details(regions: list[str]) -> Any:  # noqa: ANN401
    """Build the minimal model details the region selection reads.

    Returns:
        Model details available in *regions*.
    """
    return ModelDetails(
        id=_MODEL_ID,
        name=_MODEL_ID,
        provider="Amazon",
        input_modalities=["TEXT", "IMAGE"],
        output_modalities=["TEXT"],
        regions=regions,  # type: ignore[arg-type]
    )


class TestPrepareConverseRequestReasoningMaxTokens:
    """``_prepare_converse_request`` drops ``maxTokens`` only for high-effort reasoning.

    Ref: stdapi/models/chat/amazon_nova_2.py:ChatModel._prepare_converse_request
    """

    async def test_high_effort_reasoning_drops_max_tokens(
        self, request_log: EventLog
    ) -> None:
        """High reasoning effort strips 'maxTokens' from the inference config and warns.

        Only the token limit is removed: the ``reasoningConfig`` that triggered
        the conflict is still forwarded in ``additionalModelRequestFields``.
        """
        inference_cfg: InferenceConfigurationTypeDef = {"maxTokens": 1000}
        request = await _MODEL._prepare_converse_request(  # noqa: SLF001
            bedrock_messages=[],
            inference_cfg=inference_cfg,
            system_blocks=None,
            tool_config=None,
            additional_request_fields={
                "reasoningConfig": {"type": "enabled", "maxReasoningEffort": "high"}
            },
            service_tier=None,
        )
        assert "maxTokens" not in request["inferenceConfig"]
        assert request["additionalModelRequestFields"]["reasoningConfig"] == {
            "type": "enabled",
            "maxReasoningEffort": "high",
        }
        assert request_log["level"] == "warning"
        assert any(
            "max_tokens" in str(detail)
            for detail in request_log.get("error_detail", [])
        ), "the dropped token limit must be reported in the request log"

    async def test_medium_effort_reasoning_preserves_max_tokens(
        self, request_log: EventLog
    ) -> None:
        """Medium reasoning effort leaves 'maxTokens' untouched and does not warn."""
        inference_cfg: InferenceConfigurationTypeDef = {"maxTokens": 1000}
        request = await _MODEL._prepare_converse_request(  # noqa: SLF001
            bedrock_messages=[],
            inference_cfg=inference_cfg,
            system_blocks=None,
            tool_config=None,
            additional_request_fields={
                "reasoningConfig": {"type": "enabled", "maxReasoningEffort": "medium"}
            },
            service_tier=None,
        )
        assert request["inferenceConfig"]["maxTokens"] == 1000
        assert request_log["level"] == "info"
        assert "error_detail" not in request_log

    async def test_no_reasoning_preserves_max_tokens(
        self, request_log: EventLog
    ) -> None:
        """No reasoning config leaves 'maxTokens' untouched and does not warn."""
        inference_cfg: InferenceConfigurationTypeDef = {"maxTokens": 1000}
        request = await _MODEL._prepare_converse_request(  # noqa: SLF001
            bedrock_messages=[],
            inference_cfg=inference_cfg,
            system_blocks=None,
            tool_config=None,
            additional_request_fields={},
            service_tier=None,
        )
        assert request["inferenceConfig"]["maxTokens"] == 1000
        assert request_log["level"] == "info"
        assert "error_detail" not in request_log

    async def test_high_effort_reasoning_without_max_tokens_does_not_warn(
        self, request_log: EventLog
    ) -> None:
        """High reasoning effort with no 'maxTokens' set logs no warning."""
        inference_cfg: InferenceConfigurationTypeDef = {}
        request = await _MODEL._prepare_converse_request(  # noqa: SLF001
            bedrock_messages=[],
            inference_cfg=inference_cfg,
            system_blocks=None,
            tool_config=None,
            additional_request_fields={
                "reasoningConfig": {"type": "enabled", "maxReasoningEffort": "high"}
            },
            service_tier=None,
        )
        assert "maxTokens" not in request["inferenceConfig"]
        assert request_log["level"] == "info"
        assert "error_detail" not in request_log


class TestCodeInterpreterRoundTrip:
    """Anthropic ``code_execution`` blocks translate to Nova's ``nova_code_interpreter``.

    Nova exposes code execution as a Bedrock ``systemTool`` returning a
    ``toolResult`` with a JSON payload, while Anthropic models it as a
    ``code_execution_tool_result`` block wrapping stdout/stderr/return code.  The
    two shapes must survive a full turn: the response mapping is what the client
    reads, and the request mapping is what lets it replay the block on the next
    turn.  The ``srvtoolu_``/``tooluse_`` prefix swap keeps the two sides
    correlated.

    Ref: https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultBlock.html
         https://platform.claude.com/docs/en/agents-and-tools/tool-use/code-execution-tool
         stdapi/models/chat/amazon_nova_2.py:ChatModel._resp_map_tool_result
    """

    def test_response_tool_result_becomes_a_code_execution_block(self) -> None:
        """A successful ``nova_code_interpreter`` result maps to return code zero.

        Bedrock reports success through ``isError``, not through ``exitCode``, so a
        payload flagged as successful must yield ``return_code == 0`` whatever the
        reported exit code is.
        """
        blocks = _MODEL._resp_map_tool_result(  # noqa: SLF001
            "tooluse_abc",
            "nova_code_interpreter",
            [
                {
                    "json": {
                        "stdOut": "42\n",
                        "stdErr": "",
                        "exitCode": 0,
                        "isError": False,
                    }
                }
            ],
        )
        assert blocks is not None
        (block,) = blocks
        assert isinstance(block, CodeExecutionToolResultBlock)
        assert block.tool_use_id == "srvtoolu_abc"
        result = block.content
        assert isinstance(result, CodeExecutionResultBlock)
        assert result.stdout == "42\n"
        assert result.return_code == 0

    def test_failed_code_execution_keeps_its_exit_code(self) -> None:
        """A failing result carries stderr and the reported exit code.

        ``exitCode`` is only trusted when ``isError`` is set; a missing or zero
        exit code on a failed run degrades to ``1`` so the client never reads a
        failure as a success.
        """
        blocks = _MODEL._resp_map_tool_result(  # noqa: SLF001
            "tooluse_abc",
            "nova_code_interpreter",
            [{"json": {"stdOut": "", "stdErr": "boom", "isError": True}}],
        )
        assert blocks is not None
        (block,) = blocks
        assert isinstance(block, CodeExecutionToolResultBlock)
        result = block.content
        assert isinstance(result, CodeExecutionResultBlock)
        assert result.stderr == "boom"
        assert result.return_code == 1

    def test_unknown_tool_name_falls_back_to_the_base_mapping(self) -> None:
        """A ``toolResult`` from any other tool is not claimed by the Nova mapping.

        Returning ``None`` is what hands the block back to the generic adapter, so
        a future Nova system tool is not silently rendered as a code-execution
        result.
        """
        assert (
            _MODEL._resp_map_tool_result(  # noqa: SLF001
                "tooluse_abc", "nova_grounding", [{"json": {}}]
            )
            is None
        )

    def test_stream_tool_result_is_mapped_by_result_type(self) -> None:
        """The streamed ``nova_code_interpreter_result`` type yields the same block.

        The streaming path identifies the result by the ``type`` on the
        ``contentBlockStart`` rather than by a tool name, and must agree with the
        non-streaming mapping.
        """
        block = _MODEL._resp_stream_map_tool_result(  # noqa: SLF001
            "tooluse_abc",
            "nova_code_interpreter_result",
            [{"json": {"stdOut": "hi", "stdErr": "", "exitCode": 0, "isError": False}}],
        )
        assert block is not None
        assert isinstance(block, CodeExecutionToolResultBlock)
        assert block.tool_use_id == "srvtoolu_abc"
        result = block.content
        assert isinstance(result, CodeExecutionResultBlock)
        assert result.stdout == "hi"
        assert (
            _MODEL._resp_stream_map_tool_result(  # noqa: SLF001
                "tooluse_abc", "nova_grounding_result", []
            )
            is None
        )

    def test_replayed_result_block_becomes_a_bedrock_tool_result(self) -> None:
        """A replayed ``code_execution_tool_result`` maps back to a ``toolResult``.

        The Anthropic ``srvtoolu_`` id is restored to the Bedrock ``tooluse_``
        form it came from, and a non-zero return code is reported through both
        ``isError`` and the block ``status`` Bedrock uses.
        """
        block = _MODEL._req_map_content_block(  # noqa: SLF001
            CodeExecutionToolResultBlockParam(
                type="code_execution_tool_result",
                tool_use_id="srvtoolu_abc",
                content=CodeExecutionResultBlockParam(
                    type="code_execution_result",
                    content=[],
                    return_code=2,
                    stderr="boom",
                    stdout="",
                ),
            )
        )
        assert block == {
            "toolResult": {
                "toolUseId": "tooluse_abc",
                "content": [
                    {
                        "json": {
                            "stdOut": "",
                            "stdErr": "boom",
                            "exitCode": 2,
                            "isError": True,
                        }
                    }
                ],
                "status": "error",
            }
        }

    def test_replayed_error_result_becomes_an_empty_error_tool_result(self) -> None:
        """An error-variant result replays as an empty payload flagged as an error.

        ``code_execution_tool_result_error`` has no stdout/stderr to forward, so
        the Bedrock block carries an empty JSON object; dropping the ``status``
        would replay the failure to Nova as a success.
        """
        block = _MODEL._req_map_content_block(  # noqa: SLF001
            CodeExecutionToolResultBlockParam(
                type="code_execution_tool_result",
                tool_use_id="srvtoolu_abc",
                content=CodeExecutionToolResultErrorParam(
                    type="code_execution_tool_result_error",
                    error_code="execution_time_exceeded",
                ),
            )
        )
        assert block == {
            "toolResult": {
                "toolUseId": "tooluse_abc",
                "content": [{"json": {}}],
                "status": "error",
            }
        }

    def test_other_block_types_fall_through_to_the_base_mapping(self) -> None:
        """A block Nova does not special-case is left to the generic adapter.

        ``None`` means "not mine", which is what keeps text, image and tool_use
        blocks on the shared mapping path.
        """
        assert (
            _MODEL._req_map_content_block(  # noqa: SLF001
                TextBlockParam(type="text", text="hi")
            )
            is None
        )


@pytest.mark.usefixtures("input_files")
class TestInlineMediaTransport:
    """Amazon Nova reads an oversized attachment from storage instead of the request.

    Nova refuses a single media block past 25,000,000 base64 bytes while accepting
    the same content by reference, so the gateway stages what is too large and
    pins the request to a region that can serve it — a reference is only readable
    from the region it was written to.

    Ref: https://docs.aws.amazon.com/nova/latest/userguide/modalities-document.html
         stdapi/models/chat/_amazon_nova.py:NOVA_INLINE_MEDIA_LIMITS
         stdapi/models/__init__.py:ModelBase._converse_candidate_regions
    """

    @staticmethod
    async def _attach(base64_length: int) -> ContentBlockTypeDef:
        """Register an image attachment of *base64_length* base64 bytes.

        Returns:
            Its pending Bedrock content block.
        """
        file = InputFile(f"data:image/png;base64,{'A' * base64_length}")
        return await file.to_bedrock_content_block()

    async def test_an_attachment_at_the_limit_travels_in_the_request(self) -> None:
        """25,000,000 base64 bytes is the largest single block Nova accepts inline."""
        await self._attach(25_000_000)

        assert not await plan_bedrock_media_transport(
            _MODEL.INLINE_MEDIA_LIMITS,
            s3_location_media_types=_MODEL.S3_LOCATION_MEDIA_TYPES,
        )

    async def test_an_attachment_past_the_limit_is_read_from_storage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One base64 quad past the limit is enough to route the attachment to storage."""
        monkeypatch.setattr(InputFile, "to_s3", _stub_upload)
        block = await self._attach(25_000_004)

        assert await plan_bedrock_media_transport(
            _MODEL.INLINE_MEDIA_LIMITS,
            s3_location_media_types=_MODEL.S3_LOCATION_MEDIA_TYPES,
        )

        await resolve_all_bedrock_content_blocks(
            "us-east-1", s3_location_media_types=_MODEL.S3_LOCATION_MEDIA_TYPES
        )
        assert "s3Location" in block["image"]["source"]

    async def test_a_document_past_the_limit_is_read_from_storage(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An oversized PDF is staged too, not only an oversized image.

        Nova reads documents from storage as it reads images, and a document is
        the attachment most likely to be over the limit. Were that kind dropped
        from what the model declares, this attachment would be refused with a
        413 instead of being staged.

        Ref: stdapi/models/chat/_amazon_nova.py:NOVA_S3_LOCATION_MEDIA_TYPES
        """
        monkeypatch.setattr(InputFile, "to_s3", _stub_upload)
        file = InputFile(f"data:application/pdf;base64,{'A' * 25_000_004}")
        block = await file.to_bedrock_content_block(filename="report.pdf")

        assert await plan_bedrock_media_transport(
            _MODEL.INLINE_MEDIA_LIMITS,
            s3_location_media_types=_MODEL.S3_LOCATION_MEDIA_TYPES,
        )

        await resolve_all_bedrock_content_blocks(
            "us-east-1", s3_location_media_types=_MODEL.S3_LOCATION_MEDIA_TYPES
        )
        assert "s3Location" in block["document"]["source"]

    async def test_a_request_with_staged_media_is_pinned_to_one_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Only the region holding the staged attachment can serve the request.

        The block is resolved once, so a second candidate region would be handed a
        reference written elsewhere, which the model cannot read.
        """
        monkeypatch.setattr(InputFile, "to_s3", _stub_upload)
        await self._attach(25_000_004)

        with (
            patch(
                "stdapi.models.get_model_details",
                new=AsyncMock(return_value=_model_details(["us-east-1", "us-west-2"])),
            ),
            patch(
                "stdapi.models.get_s3_bucket_for_region",
                side_effect=lambda region: (
                    "a-bucket" if region == "us-west-2" else None
                ),
            ),
        ):
            assert await _MODEL._converse_candidate_regions() == ["us-west-2"]  # noqa: SLF001

    async def test_every_choice_of_a_request_targets_the_pinned_region(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A request asking for several choices sends them all to one region.

        Each choice is a Converse call selecting a region of its own, and region
        routing hands successive calls different ones — but the attachment is
        staged in only one of them, so the later calls must follow the first.
        """
        monkeypatch.setattr(InputFile, "to_s3", _stub_upload)
        await self._attach(25_000_004)
        rotation = iter(["us-east-1", "us-west-2"])

        with (
            patch(
                "stdapi.models.get_model_details",
                new=AsyncMock(return_value=_model_details(["us-east-1", "us-west-2"])),
            ),
            patch("stdapi.models.get_s3_bucket_for_region", return_value="a-bucket"),
            patch.object(
                ChatModel,
                "select_region",
                new=AsyncMock(side_effect=lambda **_kwargs: next(rotation)),
            ),
        ):
            candidates = await gather(
                _MODEL._converse_candidate_regions(),  # noqa: SLF001
                _MODEL._converse_candidate_regions(),  # noqa: SLF001
            )

        assert list(candidates) == [["us-east-1"], ["us-east-1"]], (
            "the second choice must not be sent to the region the first did not stage in"
        )

    @pytest.mark.parametrize("from_a_prompt", [False, True])
    async def test_media_no_region_can_store_is_refused_with_the_size_it_accepts(
        self, from_a_prompt: bool
    ) -> None:
        """With nowhere to stage it, the caller is told the size that does fit.

        The message must state what the server accepts rather than reporting an
        unrelated feature as unavailable, which is what the caller can act on.
        A request served by a stored prompt is bound to that prompt's region, so
        it is refused on the same terms when that one region cannot store either.

        Ref: stdapi/models/__init__.py:ModelBase._converse_candidate_regions
        """
        await self._attach(25_000_004)
        token = (
            BEDROCK_PROMPT_VAR.set(
                BedrockPrompt(
                    arn="arn:aws:bedrock:us-east-1:111111111111:prompt/PROMPT",
                    region="us-east-1",
                    model_id=_MODEL_ID,
                )
            )
            if from_a_prompt
            else None
        )

        try:
            with (
                patch(
                    "stdapi.models.get_model_details",
                    new=AsyncMock(return_value=_model_details(["us-east-1"])),
                ),
                patch("stdapi.models.get_s3_bucket_for_region", return_value=None),
                pytest.raises(ApiError) as exc,
            ):
                await _MODEL._converse_candidate_regions()  # noqa: SLF001
        finally:
            if token is not None:
                BEDROCK_PROMPT_VAR.reset(token)

        assert exc.value.status == 413
        message = str(exc.value)
        assert "18750000 bytes" in message
        assert "Async invocation" not in message


@pytest.mark.xdist_group("nova_2_inline_media_limits")
class TestStagedAttachmentRoundTrip:
    """An attachment too large to travel inline still reaches the model intact.

    The gateway's own tests can only show which transport was chosen; whether the
    model then reads the attachment is a property of the backend, and a reference
    it cannot read is refused outright rather than answered.  Billing the same
    number of prompt tokens as the inline request is what shows the same picture
    arrived: the model's limit is lowered here so a one-pixel image is enough to
    trigger the staging, and the answer's wording is deliberately not asserted.

    Ref: https://docs.aws.amazon.com/nova/latest/userguide/modalities-image.html
         stdapi/input_file.py:plan_bedrock_media_transport
    """

    @staticmethod
    def _describe(client: OpenAI) -> Any:  # noqa: ANN401
        """Ask the model about a one-pixel image.

        Returns:
            The chat completion.
        """
        return client.chat.completions.create(
            model=_MODEL_ID,
            max_tokens=16,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{red_png_b64()}"
                            },
                        },
                        {"type": "text", "text": "What is this? Reply in one word."},
                    ],
                }
            ],
        )

    def test_a_staged_image_reaches_the_model_whole(
        self, monkeypatch: pytest.MonkeyPatch, openai_client: OpenAI
    ) -> None:
        """The staged request is answered, and billed exactly like the inline one."""
        if not SETTINGS.aws_s3_bucket:
            pytest.skip("aws_s3_bucket not configured — an attachment cannot be staged")
        inline = self._describe(openai_client)
        assert inline.usage is not None

        staged: list[str] = []
        inline_to_s3 = InputFile.to_s3

        async def _spy(file: InputFile, region: RegionName) -> S3Object:
            """Record the staged object, then stage it for real.

            Returns:
                The staged object reference.
            """
            staged.append((obj := await inline_to_s3(file, region)).uri)
            return obj

        monkeypatch.setattr(
            ChatModel,
            "INLINE_MEDIA_LIMITS",
            InlineMediaLimits(max_file_base64_size=8, max_total_base64_size=8),
        )
        monkeypatch.setattr(InputFile, "to_s3", _spy)

        completion = self._describe(openai_client)

        assert staged, "the over-limit attachment must be staged, not sent inline"
        assert completion.choices[0].message.content
        assert completion.usage is not None
        assert completion.usage.prompt_tokens == inline.usage.prompt_tokens, (
            "the staged attachment must reach the model as the inline one does"
        )
