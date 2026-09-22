"""Reasoning effort on the OpenAI GPT and gpt-oss models, on both serving paths.

Probed on Converse (2026-09-22): GPT-5.6 and GPT-6 honour
``additionalModelRequestFields.reasoning.effort`` (``none`` through ``max``,
``low`` through ``max`` on GPT-6 Astra), reject ``minimal`` on every model and
reject the flat ``reasoning_effort`` and ``thinking``. The gpt-oss models are
the opposite: they silently ignore the effort object and honour the flat
``reasoning_effort`` (``low``, ``medium``, ``high``), rejecting ``none`` and
``minimal``. Probed on Mantle the same day: ``minimal`` is rejected by GPT-5.6
Luna and GPT-6 Luna on both APIs, and ``none`` by GPT-6 Astra.

Ref: https://developers.openai.com/api/docs/guides/reasoning
     https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-astra.html
     tests/probes/results/openai.gpt-6-luna.json
     stdapi/models/chat/openai_gpt.py:ChatModel
     stdapi/models/chat/openai_gpt_oss.py:ChatModel
     stdapi/models/chat/_mantle/_openai_gpt.py:OpenAIGptChatModel
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from openai import BadRequestError

from stdapi.models import _find_model_class
from stdapi.models.chat._mantle import get_mantle_chat_model
from stdapi.models.chat._mantle._default import ChatModel as MantleChatModel
from stdapi.models.chat._mantle._openai_gpt import OpenAIGptChatModel
from stdapi.models.chat._reasoning_effort import ReasoningEffortChatModel
from stdapi.models.chat.openai_gpt import ChatModel as GptChatModel
from stdapi.models.chat.openai_gpt_oss import ChatModel as GptOssChatModel
from stdapi.types.anthropic_messages import MessageCreateParams
from stdapi.types.openai_chat_completions import CompletionCreateParams

if TYPE_CHECKING:
    from openai import OpenAI

    from stdapi.aws_bedrock_mantle import MantleApi
    from stdapi.models.chat import Effort
    from stdapi.types import JsonMapping
    from stdapi.types.openai_chat_completions import ReasoningEffort

#: GPT-6 Luna, which the test deployment prefers on Mantle (tests/conftest.py).
_GPT6_LUNA = "openai.gpt-6-luna"

#: GPT-6 Sol, which the test deployment does not prefer on Mantle: served by Converse.
_GPT6_SOL = "openai.gpt-6-sol"

#: GPT-6 Astra, the one GPT model that cannot stop reasoning.
_GPT6_ASTRA = "openai.gpt-6-astra"

#: A Daybreak edition of Astra, which shares its effort range (assumed, unprobed).
_DAYBREAK_ASTRA = "openai.gpt-daybreak-blue-6-astra"

#: Output tokens under which an answer to the reasoning prompt carries no reasoning.
_UNREASONED_OUTPUT_TOKENS = 32

#: The cheapest gpt-oss model served by Converse.
_GPT_OSS_20B = "openai.gpt-oss-20b-1:0"

#: The official-API model the vendor lane runs the same test body against.
_OFFICIAL_REASONING_MODEL = "gpt-5.1"

#: Prompt that needs several steps, so a reasoning model has a reason to reason.
_REASONING_PROMPT = (
    "How many positive integers below 200 are divisible by 3 or 5 but not by 7? "
    "Answer with the number only."
)


def _completion(model_id: str, **overrides: object) -> CompletionCreateParams:
    """Build a validated chat completion request for *model_id*."""
    body: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        **overrides,
    }
    return CompletionCreateParams.model_validate(body)


@pytest.mark.local
class TestMatcher:
    """GPT models resolve to the effort-object class, gpt-oss to the flat-field one.

    Ref: stdapi/models/__init__.py:_find_model_class
    """

    @pytest.mark.parametrize(
        "model_id",
        [
            "openai.gpt-5.6-luna",
            "openai.gpt-5.6-sol",
            "openai.gpt-5.6-terra",
            "openai.gpt-6-astra",
            _GPT6_LUNA,
            "openai.gpt-6-sol",
            "openai.gpt-7",
        ],
    )
    def test_gpt_models_take_the_effort_object(self, model_id: str) -> None:
        """Every numbered GPT generation, including later ones, reaches the GPT class."""
        model_class = _find_model_class(model_id)

        assert model_class is GptChatModel
        assert issubclass(model_class, ReasoningEffortChatModel)

    @pytest.mark.parametrize(
        "model_id",
        [
            "openai.gpt-oss-20b-1:0",
            "openai.gpt-oss-120b-1:0",
            "openai.gpt-oss-safeguard-20b",
            "openai.gpt-oss-safeguard-120b",
        ],
    )
    def test_gpt_oss_models_take_the_flat_field(self, model_id: str) -> None:
        """Every gpt-oss model reaches its own class, not the GPT one."""
        assert _find_model_class(model_id) is GptOssChatModel


@pytest.mark.local
class TestGptReasoningFields:
    """The GPT models get the effort object from every dialect.

    Ref: stdapi/models/chat/_reasoning_effort.py:ReasoningEffortChatModel._req_configure_reasoning
    """

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [("none", "none"), ("minimal", "low"), ("medium", "medium"), ("max", "max")],
    )
    async def test_chat_completions_reasoning_effort(
        self, requested: str, expected: str
    ) -> None:
        """``reasoning_effort`` reaches the model; ``minimal``, which it rejects, is ``low``.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
        """
        request = _completion(_GPT6_LUNA, reasoning_effort=requested)

        payload, _, _ = await GptChatModel(_GPT6_LUNA).build_completion_request(request)

        assert payload["additionalModelRequestFields"] == {
            "reasoning": {"effort": expected}
        }

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [("none", None), ("minimal", {"effort": "low"}), ("max", {"effort": "max"})],
    )
    async def test_astra_is_never_sent_none(
        self,
        requested: str,
        expected: dict[str, str] | None,
        request_log: dict[str, Any],
    ) -> None:
        """GPT-6 Astra rejects ``none``, so disabling reasoning keeps its default.

        Probed: ``Unsupported value: 'none' is not supported with the
        'us.openai.gpt-6-astra' model. Supported values are: 'low', 'medium',
        'high', 'xhigh', and 'max'.``

        Ref: tests/probes/results/openai.gpt-6-astra.json
             stdapi/models/chat/openai_gpt.py:ChatModel.REASONING_DISABLE_SUPPORTED
        """
        request = _completion(_GPT6_ASTRA, reasoning_effort=requested)

        payload, _, _ = await GptChatModel(_GPT6_ASTRA).build_completion_request(
            request
        )

        assert payload.get("additionalModelRequestFields", {}).get("reasoning") == (
            expected
        )
        assert ("error_detail" in request_log) is (expected is None)

    @pytest.mark.parametrize(
        "model_id",
        ["openai.gpt-5.6-luna", _GPT6_SOL, "openai.gpt-daybreak-blue-5.6-sol"],
    )
    def test_other_gpt_models_can_disable_reasoning(self, model_id: str) -> None:
        """Only Astra withholds ``none``; the other GPT models all accept it.

        Ref: tests/probes/results/openai.gpt-6-sol.json
        """
        assert GptChatModel(model_id).REASONING_DISABLE_SUPPORTED

    @pytest.mark.parametrize("model_id", [_GPT6_ASTRA, _DAYBREAK_ASTRA])
    def test_every_astra_edition_withholds_none(self, model_id: str) -> None:
        """The Daybreak qualifier does not hide an Astra from the gate.

        Ref: stdapi/models/chat/openai_gpt.py:ALWAYS_REASONING_MATCHER
        """
        assert not GptChatModel(model_id).REASONING_DISABLE_SUPPORTED

    async def test_anthropic_thinking_disabled(self) -> None:
        """Disabling thinking on the Messages route sends ``none``.

        Ref: https://docs.claude.com/en/api/messages
        """
        request = MessageCreateParams.model_validate(
            {
                "model": _GPT6_LUNA,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "thinking": {"type": "disabled"},
            }
        )

        payload, _ = await GptChatModel(_GPT6_LUNA).build_message_request(request)

        assert payload["additionalModelRequestFields"] == {
            "reasoning": {"effort": "none"}
        }


@pytest.mark.local
class TestGptOssReasoningFields:
    """The gpt-oss models get the flat ``reasoning_effort`` on their own three-level scale.

    Ref: stdapi/models/chat/openai_gpt_oss.py:ChatModel._req_configure_reasoning
    """

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [
            ("minimal", "low"),
            ("low", "low"),
            ("medium", "medium"),
            ("high", "high"),
            ("xhigh", "high"),
            ("max", "high"),
        ],
    )
    def test_effort_maps_onto_three_levels(self, requested: str, expected: str) -> None:
        """Every level is sent flat, capped at ``high`` and lifted from ``minimal``."""
        fields: JsonMapping = {}

        GptOssChatModel(_GPT_OSS_20B)._req_configure_reasoning(  # noqa: SLF001
            fields, enabled=True, reasoning_effort=cast("Effort", requested)
        )

        assert fields == {"reasoning_effort": expected}

    @pytest.mark.parametrize("requested", [None, "none"])
    def test_disabled_sends_nothing(
        self, requested: str | None, request_log: dict[str, Any]
    ) -> None:
        """Reasoning cannot be turned off: ``none`` is rejected, so nothing is sent.

        Probed: ``Harmony does not support reasoning_effort='none'``.

        Ref: tests/probes/results/openai.gpt-oss-20b-1_0.json
        """
        fields: JsonMapping = {}

        GptOssChatModel(_GPT_OSS_20B)._req_configure_reasoning(  # noqa: SLF001
            fields, enabled=False, reasoning_effort=cast("Effort | None", requested)
        )

        assert fields == {}
        assert "cannot be disabled" in str(request_log["error_detail"])

    def test_enabled_without_a_level_sends_nothing(self) -> None:
        """Asking for reasoning without a level leaves the model default."""
        fields: JsonMapping = {}

        GptOssChatModel(_GPT_OSS_20B)._req_configure_reasoning(  # noqa: SLF001
            fields, enabled=True, budget_tokens=2048
        )

        assert fields == {}


def _completion_tokens(client: OpenAI, model: str, effort: ReasoningEffort) -> int:
    """Ask *model* the reasoning prompt at *effort* and return its completion tokens.

    Args:
        client: Client bound to the target under test.
        model: Model to call.
        effort: Reasoning effort to request.

    Returns:
        The completion token count, which includes the reasoning tokens. An
        answer that spent the whole budget reasoning (gpt-oss at ``high`` does
        so now and then) still counts, since only the count is compared.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": _REASONING_PROMPT}],
        reasoning_effort=effort,
        max_completion_tokens=4096,
    )
    assert response.usage is not None
    return response.usage.completion_tokens


class TestReasoningEffortReachesTheModel:
    """A requested effort changes how much the model reasons.

    Ref: https://developers.openai.com/api/docs/guides/reasoning
    """

    def test_reasoning_effort_none_stops_reasoning(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """``none`` answers in a few tokens, fewer than ``high``.

        Completion tokens include the reasoning tokens, so a dropped effort
        leaves both requests reasoning at the model default, well above the
        bound. Probed on GPT-6 Sol: ``none`` answers in 5 tokens. GPT-6 Sol is
        the model the test deployment serves through Converse.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             stdapi/models/chat/openai_gpt.py:ChatModel
        """
        model = _OFFICIAL_REASONING_MODEL if use_official_api else _GPT6_SOL

        without = _completion_tokens(openai_client, model, "none")
        reasoned = _completion_tokens(openai_client, model, "high")

        assert without < _UNREASONED_OUTPUT_TOKENS
        assert without < reasoned

    @pytest.mark.gateway(
        "OpenAI refuses 'minimal' on GPT-5.1 and later with a 400 "
        "unsupported_value; the gateway serves it as 'low' by design"
    )
    @pytest.mark.parametrize("model", [_GPT6_LUNA, _GPT6_SOL])
    def test_minimal_is_served_on_every_path(
        self, openai_client: OpenAI, model: str
    ) -> None:
        """``minimal``, which every GPT-5.6 and GPT-6 model rejects, is still served.

        GPT-6 Luna is served by Mantle in the test deployment and GPT-6 Sol by
        Converse, so each serving path answers a ``minimal`` request once.

        Ref: stdapi/models/chat/_mantle/_openai_gpt.py:OpenAIGptChatModel._fit_effort
             stdapi/models/chat/_reasoning_effort.py:ReasoningEffortChatModel
        """
        response = openai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            reasoning_effort="minimal",
            max_completion_tokens=256,
        )

        assert response.choices[0].finish_reason in {"stop", "length"}

    @pytest.mark.gateway("gpt-oss is not served by the official OpenAI API")
    def test_gpt_oss_reasoning_effort_sizes_the_reasoning(
        self, openai_client: OpenAI
    ) -> None:
        """``low`` reasons for less than half the tokens of ``high`` on gpt-oss.

        Probed on gpt-oss-20b: the flat field gave 261 to 296 completion tokens
        at ``low`` and 893 or more at ``high``, while an ignored effort left
        both near the 785-to-909 default, a spread two samples of the same
        default cannot double.

        Ref: tests/probes/results/openai.gpt-oss-20b-1_0.json
             stdapi/models/chat/openai_gpt_oss.py:ChatModel
        """
        light = _completion_tokens(openai_client, _GPT_OSS_20B, "low")
        deep = _completion_tokens(openai_client, _GPT_OSS_20B, "high")

        assert deep > 2 * light

    def test_reasoning_tokens_are_reported(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """``usage.completion_tokens_details.reasoning_tokens`` counts the reasoning.

        OpenAI reports the count on every completion, within
        ``completion_tokens``. GPT-6 Luna is served by Mantle in the test
        deployment, which reports the count; Converse does not split it out, so
        a Converse-served model omits ``completion_tokens_details``.

        Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
             stdapi/models/chat/_mantle/_openai_gpt.py:OpenAIGptChatModel
        """
        response = openai_client.chat.completions.create(
            model=_OFFICIAL_REASONING_MODEL if use_official_api else _GPT6_LUNA,
            messages=[{"role": "user", "content": _REASONING_PROMPT}],
            reasoning_effort="medium",
            max_completion_tokens=4096,
        )

        usage = response.usage
        assert usage is not None
        assert usage.completion_tokens_details is not None
        reasoning = usage.completion_tokens_details.reasoning_tokens
        assert reasoning is not None
        assert 0 < reasoning <= usage.completion_tokens


class TestReasoningEffortValidation:
    """A reasoning effort outside the accepted values is refused with a 400.

    OpenAI names the field in ``param`` and sets ``code`` (``unsupported_value``
    on Chat Completions, listing the model's own levels, ``invalid_value`` on
    Responses). The gateway validates the field before choosing a model and
    answers with ``param`` and ``code`` unset, naming the field and every
    accepted level in the message instead.

    Ref: https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
         https://developers.openai.com/api/reference/resources/responses/methods/create
         stdapi/main.py:handle_validation_exception
    """

    @staticmethod
    def _assert_refused(
        error: BadRequestError, param: str, code: str, *, use_official_api: bool
    ) -> None:
        """Assert *error* is the 400 refusing an unknown effort level in *param*."""
        assert error.status_code == 400
        body = error.body
        assert isinstance(body, dict)
        assert body["type"] == "invalid_request_error"
        assert "'low'" in body["message"], "the message lists the accepted levels"
        if use_official_api:
            assert body["param"] == param
            assert body["code"] == code
        else:
            assert body["param"] is None
            assert body["code"] is None
            assert param in body["message"]

    def test_chat_completions_refuses_an_unknown_level(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """``reasoning_effort: "bogus"`` is refused naming ``reasoning_effort``."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.chat.completions.create(  # type: ignore[call-overload]
                model=_OFFICIAL_REASONING_MODEL if use_official_api else _GPT6_LUNA,
                messages=[{"role": "user", "content": "Reply with OK."}],
                reasoning_effort="bogus",
            )

        self._assert_refused(
            exc_info.value,
            "reasoning_effort",
            "unsupported_value",
            use_official_api=use_official_api,
        )

    def test_responses_refuses_an_unknown_level(
        self, openai_client: OpenAI, use_official_api: bool
    ) -> None:
        """``reasoning: {"effort": "bogus"}`` is refused naming ``reasoning.effort``."""
        with pytest.raises(BadRequestError) as exc_info:
            openai_client.responses.create(  # type: ignore[call-overload]
                model=_OFFICIAL_REASONING_MODEL if use_official_api else _GPT6_LUNA,
                input="Reply with OK.",
                reasoning={"effort": "bogus"},
            )

        self._assert_refused(
            exc_info.value,
            "reasoning.effort",
            "invalid_value",
            use_official_api=use_official_api,
        )


@pytest.mark.local
class TestMantleEffort:
    """On Mantle, the effort is forwarded as sent except where the model rejects it.

    Probed on Mantle (2026-09-22): ``Unsupported value: 'minimal' is not
    supported with the 'openai.gpt-6-luna' model``, the same on GPT-5.6 Luna and
    GPT-6 Astra, and ``'none' is not supported with the 'openai.gpt-6-astra'
    model`` on both Chat Completions and Responses.

    Ref: stdapi/models/chat/_mantle/_openai_gpt.py:OpenAIGptChatModel
    """

    @staticmethod
    async def _served_payload(
        monkeypatch: pytest.MonkeyPatch,
        model_id: str,
        inbound: MantleApi,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Serve *payload* with the upstream call stubbed, and return what reached it."""
        seen: list[dict[str, Any]] = []

        async def _upstream(
            _self: object, _inbound: MantleApi, body: dict[str, Any], **_: object
        ) -> tuple[MantleApi, str, dict[str, Any]]:
            seen.append(dict(body))
            return _inbound, "us-east-1", {}

        monkeypatch.setattr(MantleChatModel, "_serve", _upstream)
        model = get_mantle_chat_model(model_id)
        assert isinstance(model, OpenAIGptChatModel)
        await model._serve(inbound, payload, stream=False)  # noqa: SLF001
        return seen[0]

    @pytest.mark.parametrize(
        "model_id",
        ["openai.gpt-5.6-luna", _GPT6_LUNA, "openai.gpt-daybreak-blue-5.6-sol"],
    )
    async def test_chat_minimal_is_sent_as_low(
        self, monkeypatch: pytest.MonkeyPatch, model_id: str
    ) -> None:
        """``minimal`` becomes ``low`` on every GPT-5 and GPT-6 Mantle model."""
        sent = await self._served_payload(
            monkeypatch, model_id, "chat_completions", {"reasoning_effort": "minimal"}
        )

        assert sent["reasoning_effort"] == "low"

    async def test_responses_minimal_is_sent_as_low(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``reasoning.effort: "minimal"`` becomes ``low`` on the Responses API."""
        sent = await self._served_payload(
            monkeypatch,
            _GPT6_LUNA,
            "responses",
            {"reasoning": {"effort": "minimal", "summary": "auto"}},
        )

        assert sent["reasoning"] == {"effort": "low", "summary": "auto"}

    @pytest.mark.parametrize("model_id", [_GPT6_ASTRA, _DAYBREAK_ASTRA])
    async def test_astra_none_keeps_the_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        model_id: str,
        request_log: dict[str, Any],
    ) -> None:
        """Astra is never sent ``none``: the model's default level is used, with a warning."""
        sent = await self._served_payload(
            monkeypatch, model_id, "chat_completions", {"reasoning_effort": "none"}
        )

        assert "reasoning_effort" not in sent
        assert "cannot be disabled" in str(request_log["error_detail"])

    @pytest.mark.parametrize(
        ("model_id", "effort"),
        [(_GPT6_LUNA, "none"), (_GPT6_ASTRA, "max"), (_GPT6_LUNA, "xhigh")],
    )
    async def test_accepted_levels_are_forwarded_as_sent(
        self, monkeypatch: pytest.MonkeyPatch, model_id: str, effort: str
    ) -> None:
        """Every level the model accepts reaches it unchanged."""
        sent = await self._served_payload(
            monkeypatch, model_id, "chat_completions", {"reasoning_effort": effort}
        )

        assert sent["reasoning_effort"] == effort
