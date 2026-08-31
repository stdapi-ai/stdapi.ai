"""The shared answers a route gives when the request cannot be served.

A missing IAM permission, an unconfigured bucket or an unreachable endpoint are
none of the caller's doing: they read one generic 503, identical whatever is
absent, while the operator reads what is missing by name in the server log. One
helper owns both halves so the two messages cannot drift apart per feature.

A model this server does not serve is the caller's to act on instead, so its
404 has to say where the models it does serve are listed.

Ref: https://developers.openai.com/api/docs/guides/error-codes.md
     stdapi/api_errors.py:FeatureUnavailableError
     stdapi/api_errors.py:feature_unavailable_guard
     stdapi/api_errors.py:UnsupportedModelError
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    EndpointResolutionError,
)

from stdapi import models as models_module
from stdapi.api_errors import (
    DENIED_CALL_KEY,
    ApiError,
    FeatureUnavailableError,
    UnsupportedModelError,
    denied_feature_unavailable,
    feature_unavailable_guard,
    iam_action,
    iam_denial_detail,
)
from stdapi.aws import _record_after_call
from stdapi.config import AWS_SESSION

if TYPE_CHECKING:
    from botocore.model import OperationModel

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: How S3 spells a denial, where Bedrock spells it ``AccessDeniedException``.
_S3_DENIED_CODE = "AccessDenied"

#: An AWS error the guard must leave to its own handler.
_THROTTLED_CODE = "ThrottlingException"

#: What the caller reads, whichever permission, setting or resource is missing.
_CLIENT_MESSAGE = (
    "The Batch API is not available on the current server. "
    "Please contact the administrator to enable it."
)


#: What a model-not-found answer may not exceed: one sentence plus its pointer.
_MODEL_NOT_FOUND_MAX_CHARS = 300

#: A catalogue large enough that enumerating it would blow that budget apart.
_CATALOGUE = tuple(f"vendor.model-{index}-v1:0" for index in range(90))


def _denied(code: str = "AccessDeniedException") -> ClientError:
    """Return the error AWS raises when the server's own role lacks a permission."""
    return ClientError(
        {"Error": {"Code": code, "Message": "User is not authorized to perform"}},
        "CreateModelInvocationJob",
    )


class TestFeatureUnavailableError:
    """The 503 both audiences get, and what each of them is told.

    Ref: stdapi/api_errors.py:FeatureUnavailableError
    """

    def test_the_caller_learns_nothing_about_the_backend(
        self, request_log: dict[str, Any]
    ) -> None:
        """The message names the API and the administrator, never what is missing."""
        error = FeatureUnavailableError(
            "The Batch API", "iam:PassRole on arn:aws:iam::1234:role/batch is missing."
        )

        assert error.status == 503
        assert error.code == "feature_unavailable"
        assert str(error) == _CLIENT_MESSAGE
        assert request_log["error_detail"] == [
            "iam:PassRole on arn:aws:iam::1234:role/batch is missing."
        ]

    def test_a_real_denial_leaks_none_of_its_words_to_the_caller(
        self, request_log: dict[str, Any]
    ) -> None:
        """What AWS said about the refused call reaches the log, never the answer.

        ``denied_feature_unavailable`` builds its operator detail from the
        operation name and AWS's own message, so the log carries them for real
        before the client message is checked to have dropped them.
        """
        error = denied_feature_unavailable(_denied())
        assert error is not None

        detail = str(request_log["error_detail"])
        message = str(error)
        for leaked in ("CreateModelInvocationJob", "not authorized"):
            assert leaked in detail
            assert leaked not in message

    def test_the_operator_detail_is_a_warning_not_a_crash(
        self, request_log: dict[str, Any]
    ) -> None:
        """The entry is raised to ``warning``: a misconfiguration, not a failure.

        Logged with no level at all it would resolve to ``critical`` and page
        whoever watches the log for real incidents.
        """
        FeatureUnavailableError("The Files API", "aws_s3_bucket is not set.")

        assert request_log["level"] == "warning"

    @pytest.mark.usefixtures("request_log")
    def test_two_features_answer_the_same_shape(self) -> None:
        """A permission and a setting are indistinguishable to the caller.

        The difference between "no permission" and "not configured" is a map of
        the backend, and the caller can act on neither.
        """
        denied = FeatureUnavailableError("The Vector Stores API", "s3vectors:Query")
        unset = FeatureUnavailableError("The Vector Stores API", "bucket not set")

        assert str(denied) == str(unset)
        assert denied.status == unset.status == 503


class TestFeatureUnavailableGuard:
    """Which backend failures become the unavailable answer, and which do not.

    Ref: stdapi/api_errors.py:feature_unavailable_guard
    """

    def test_a_denied_call_names_its_permission_to_the_operator(
        self, request_log: dict[str, Any]
    ) -> None:
        """AccessDenied answers 503 and logs the permission the role needs."""
        with (
            pytest.raises(FeatureUnavailableError) as raised,
            feature_unavailable_guard(
                "The Batch API", missing="bedrock:CreateModelInvocationJob"
            ),
        ):
            raise _denied()

        assert raised.value.status == 503
        assert str(raised.value) == _CLIENT_MESSAGE
        assert any(
            "bedrock:CreateModelInvocationJob" in str(detail)
            for detail in request_log["error_detail"]
        )

    @pytest.mark.usefixtures("request_log")
    def test_the_s3_spelling_of_a_denial_counts_too(self) -> None:
        """S3 answers ``AccessDenied`` where Bedrock answers ``AccessDeniedException``."""
        with (
            pytest.raises(FeatureUnavailableError),
            feature_unavailable_guard("The Files API", missing="s3:GetObject"),
        ):
            raise _denied(_S3_DENIED_CODE)

    @pytest.mark.usefixtures("request_log")
    def test_another_aws_error_is_left_alone(self) -> None:
        """A throttle or a validation error still reaches its own handler.

        Answering every backend failure as "not available on this deployment"
        would hide the transient ones a client can retry.
        """
        with (
            pytest.raises(ClientError),
            feature_unavailable_guard("The Batch API", missing="bedrock:Create"),
        ):
            raise _denied(_THROTTLED_CODE)

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(
                EndpointConnectionError(endpoint_url="https://x.invalid"),
                id="unreachable",
            ),
            pytest.param(EndpointResolutionError(msg="no endpoint"), id="no-endpoint"),
            pytest.param(
                ConnectTimeoutError(endpoint_url="https://x.invalid"), id="timeout"
            ),
        ],
    )
    def test_an_endpoint_the_region_does_not_serve_is_unavailable_too(
        self, request_log: dict[str, Any], error: Exception
    ) -> None:
        """A region that does not offer the service is a deployment choice.

        These services cover fewer regions than model inference, so the answer
        is the same 503 rather than a 500 nobody can act on.
        """
        with (
            pytest.raises(FeatureUnavailableError),
            feature_unavailable_guard(
                "The Batch API",
                missing="bedrock:CreateModelInvocationJob",
                unreachable="Set a region that serves batch inference.",
            ),
        ):
            raise error

        assert request_log["error_detail"] == [
            "Set a region that serves batch inference."
        ]

    @pytest.mark.usefixtures("request_log")
    def test_an_unreachable_endpoint_propagates_when_unclaimed(self) -> None:
        """A guard naming no regional cause leaves the connection error alone."""
        with (
            pytest.raises(EndpointConnectionError),
            feature_unavailable_guard("The Batch API", missing="bedrock:Create"),
        ):
            raise EndpointConnectionError(endpoint_url="https://x.invalid")

    def test_a_clean_call_is_untouched(self, request_log: dict[str, Any]) -> None:
        """The guard writes nothing when the call succeeds."""
        with feature_unavailable_guard("The Batch API", missing="bedrock:Create"):
            pass

        assert "error_detail" not in request_log

    def test_the_error_is_the_api_error_every_route_renders(self) -> None:
        """It is an ``ApiError``, so the envelope is the calling API's own."""
        assert issubclass(FeatureUnavailableError, ApiError)


class TestUnsupportedModelError:
    """The 404 for a name this server does not serve, and what it teaches.

    Adopting the gateway is two actions — the base URL, then the model — so the
    answer to an unknown name has to point at the second one instead of
    enumerating every model behind it.

    Ref: https://developers.openai.com/api/docs/guides/error-codes.md
         stdapi/api_errors.py:UnsupportedModelError
    """

    def test_the_answer_names_the_model_and_where_the_served_ones_are_listed(
        self,
    ) -> None:
        """The requested name is quoted back, and the caller is sent to the catalogue."""
        error = UnsupportedModelError("gpt-4o")

        message = str(error)
        assert error.status == 404
        assert error.code == "model_not_found"
        assert "`gpt-4o`" in message
        assert "does not exist or you do not have access to it" in message
        assert "models endpoint" in message

    def test_the_answer_is_short_enough_to_read(self) -> None:
        """One sentence and its pointer, not a payload to scroll through."""
        assert len(str(UnsupportedModelError("gpt-4o"))) <= _MODEL_NOT_FOUND_MAX_CHARS

    def test_a_deprecated_model_still_says_what_replaced_it(self) -> None:
        """The extra context a deprecation carries is kept, and stays bounded."""
        error = UnsupportedModelError(
            "old-model", detail="Please use 'new-model' instead."
        )

        message = str(error)
        assert "Please use 'new-model' instead." in message
        assert len(message) <= _MODEL_NOT_FOUND_MAX_CHARS

    def test_the_status_override_still_applies(self) -> None:
        """Routes whose upstream answers 400 for an unknown model keep doing so."""
        assert UnsupportedModelError("gpt-4o", status=400).status == 400

    async def test_a_loaded_catalogue_never_reaches_the_caller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A miss against a full catalogue answers with that same short body.

        The message used to join every served model ID into itself, so a client
        exception carried the whole catalogue rather than the way to search it.

        Ref: stdapi/models/__init__.py:validate_model
        """

        async def _already_loaded(_start_event: object = None) -> bool:
            """Stand in for the refresh a cache miss triggers."""
            return False

        catalogue = dict.fromkeys(_CATALOGUE)
        monkeypatch.setattr(models_module, "_MODELS", catalogue)
        monkeypatch.setattr(models_module, "_refresh_bedrock_models", _already_loaded)

        with pytest.raises(UnsupportedModelError) as raised:
            await models_module.validate_model("gpt-4o")

        message = str(raised.value)
        assert len(message) <= _MODEL_NOT_FOUND_MAX_CHARS
        assert not any(model_id in message for model_id in _CATALOGUE)


#: An AWS denial as worded by the services that name the action they refused.
_DENIAL_MESSAGE = (
    "User: arn:aws:sts::123456789012:assumed-role/stdapi/task is not authorized "
    "to perform: bedrock:ListProvisionedModelThroughputs on resource: "
    "arn:aws:bedrock:eu-west-3:123456789012:provisioned-model/* because no "
    "identity-based policy allows the action"
)

#: The resource ARN that denial was attempted on.
_DENIAL_RESOURCE = "arn:aws:bedrock:eu-west-3:123456789012:provisioned-model/*"


def _operation_model(service: str, operation: str) -> OperationModel:
    """Return the botocore operation model of a real AWS API operation."""
    from botocore.session import get_session  # noqa: PLC0415

    model: OperationModel = (
        get_session().get_service_model(service).operation_model(operation)
    )
    return model


def _denied_call(message: str, operation: str, **recorded: str) -> ClientError:
    """Return a denial as botocore raises it, optionally hook-enriched."""
    response: Any = {"Error": {"Code": "AccessDeniedException", "Message": message}}
    if recorded:
        response[DENIED_CALL_KEY] = recorded
    return ClientError(response, operation)


class TestIamActionDerivation:
    """The IAM action a failed AWS call needed, derived rather than tabulated.

    An operator reading "access denied" learns nothing actionable, and the
    action name is the whole answer. botocore already carries both halves of
    it — the service's IAM prefix in ``signingName`` and the operation name in
    the operation model — so deriving it beats a table that silently rots as
    services are added.

    Ref: https://docs.aws.amazon.com/service-authorization/latest/reference/reference_policies_actions-resources-contextkeys.html
         stdapi/api_errors.py:iam_action
    """

    @pytest.mark.parametrize(
        ("service", "operation", "expected"),
        [
            pytest.param(
                "bedrock",
                "ListProvisionedModelThroughputs",
                "bedrock:ListProvisionedModelThroughputs",
                id="prefix-equals-endpoint",
            ),
            pytest.param(
                "bedrock-runtime",
                "Converse",
                "bedrock:Converse",
                id="signing-name-wins",
            ),
            pytest.param(
                "pricing",
                "GetProducts",
                "pricing:GetProducts",
                id="api-endpoint-prefix",
            ),
            pytest.param("sts", "AssumeRole", "sts:AssumeRole", id="no-signing-name"),
            pytest.param(
                "meteringmarketplace",
                "BatchMeterUsage",
                "aws-marketplace:BatchMeterUsage",
                id="renamed-service",
            ),
        ],
    )
    def test_the_action_comes_from_the_service_model(
        self, service: str, operation: str, expected: str
    ) -> None:
        """The prefix is the signing name, falling back to the endpoint prefix.

        ``bedrock-runtime``, ``api.pricing`` and ``metering.marketplace`` are
        the three shapes proving the endpoint prefix alone would be wrong.
        """
        assert iam_action(_operation_model(service, operation)) == expected

    def test_s3_operations_map_to_the_action_that_authorizes_them(self) -> None:
        """S3 is the one service this server calls that renames its actions.

        It also answers a bare "Access Denied" naming nothing, so the derived
        name is the only one an operator ever gets for it.
        """
        assert iam_action(_operation_model("s3", "ListObjectsV2")) == "s3:ListBucket"
        assert iam_action(_operation_model("s3", "HeadObject")) == "s3:GetObject"
        assert iam_action(_operation_model("s3", "GetObject")) == "s3:GetObject"


class TestIamDenialDetail:
    """What an operator is told when AWS refuses one of this server's own calls.

    Ref: stdapi/api_errors.py:iam_denial_detail
         stdapi/aws.py:_record_after_call
    """

    def test_the_action_aws_named_wins_over_the_derived_one(self) -> None:
        """AWS names the authoritative action, so its message is parsed first.

        An operation can be authorized by more than one action, and only AWS
        knows which of them was evaluated.
        """
        detail = iam_denial_detail(
            _denied_call(_DENIAL_MESSAGE, "ListProvisionedModelThroughputs")
        )

        assert detail is not None
        assert "bedrock:ListProvisionedModelThroughputs" in detail
        assert _DENIAL_RESOURCE in detail, (
            "the resource the action was attempted on belongs in the detail"
        )
        assert "assumed-role/stdapi/task" not in detail, (
            "the principal is the server's own role and adds nothing"
        )

    def test_a_bare_denial_falls_back_to_the_recorded_action(self) -> None:
        """S3 names no action, so the one the after-call hook recorded is used."""
        response: Any = {
            "Error": {"Code": "AccessDenied", "Message": "Access Denied"},
            DENIED_CALL_KEY: {"action": "s3:ListBucket", "region": "eu-west-3"},
        }

        detail = iam_denial_detail(ClientError(response, "ListObjectsV2"))

        assert detail is not None
        assert "s3:ListBucket" in detail
        assert "eu-west-3" in detail, "the region the call was made in belongs in it"
        assert "is missing the IAM permission" not in detail, (
            "only IAM's own grammar proves a policy gap; this one is unattributed"
        )

    def test_a_service_refusing_the_account_is_not_called_a_policy_gap(self) -> None:
        """Bedrock refuses the account where provisioned throughput is not offered.

        Verbatim from ``bedrock:ListProvisionedModelThroughputs`` in af-south-1
        on 2026-08-31: an ``AccessDeniedException`` that names no principal and
        no action, which no IAM policy can fix. Saying "the server role is
        missing" it would send an operator editing a policy for nothing.
        """
        response: Any = {
            "Error": {
                "Code": "AccessDeniedException",
                "Message": "Your account is not authorized to invoke this API operation.",
            },
            DENIED_CALL_KEY: {
                "action": "bedrock:ListProvisionedModelThroughputs",
                "region": "af-south-1",
            },
        }

        detail = iam_denial_detail(
            ClientError(response, "ListProvisionedModelThroughputs")
        )

        assert detail is not None
        assert "bedrock:ListProvisionedModelThroughputs" in detail
        assert "af-south-1" in detail
        assert "is missing the IAM permission" not in detail

    def test_a_failure_that_is_not_a_denial_is_not_described_as_one(self) -> None:
        """A throttle and a transport error must not read as a missing permission."""
        assert iam_denial_detail(_denied(_THROTTLED_CODE)) is None
        assert (
            iam_denial_detail(EndpointConnectionError(endpoint_url="https://x.invalid"))
            is None
        )

    def test_the_after_call_hook_records_the_action_and_the_region(self) -> None:
        """Botocore raises from the very dict the hook is handed, so it can enrich it.

        The hook is the only place where the operation model and the client's
        region are both still known: a ``ClientError`` carries neither.
        """
        parsed: dict[str, Any] = {
            "Error": {"Code": "AccessDeniedException", "Message": "Access Denied"},
            "ResponseMetadata": {"RequestId": "req-1"},
        }

        _record_after_call(
            parsed,
            _operation_model("bedrock", "ListProvisionedModelThroughputs"),
            {"client_region": "eu-west-3"},
        )

        assert parsed[DENIED_CALL_KEY] == {
            "action": "bedrock:ListProvisionedModelThroughputs",
            "region": "eu-west-3",
        }

    async def test_a_real_client_raises_the_enriched_error(self) -> None:
        """End to end: the error botocore raises carries what the hook recorded.

        The design rests on botocore raising ``ClientError`` from the very dict
        it passed to ``after-call``; only the transport is faked here, so a
        botocore release that stopped doing that fails this test.
        """
        parsed: Any = {
            "Error": {"Code": "AccessDeniedException", "Message": "Access Denied"},
            "ResponseMetadata": {"RequestId": "req-3", "HTTPStatusCode": 403},
        }

        async def _make_request(
            *_args: object, **_kwargs: object
        ) -> tuple[SimpleNamespace, Any]:
            return SimpleNamespace(status_code=403), parsed

        async with AWS_SESSION.create_client(
            "bedrock",
            region_name="eu-west-3",
            aws_access_key_id="x",
            aws_secret_access_key="y",  # noqa: S106
        ) as client:
            client._endpoint.make_request = _make_request  # type: ignore[attr-defined] # noqa: SLF001
            with pytest.raises(ClientError) as raised:
                await client.list_provisioned_model_throughputs()

        detail = iam_denial_detail(raised.value)
        assert detail is not None
        assert "bedrock:ListProvisionedModelThroughputs" in detail
        assert "eu-west-3" in detail

    def test_a_successful_call_is_left_untouched(self) -> None:
        """Nothing denied, nothing recorded: the happy path pays nothing."""
        parsed: dict[str, Any] = {"ResponseMetadata": {"RequestId": "req-2"}}

        _record_after_call(
            parsed, _operation_model("bedrock", "ListFoundationModels"), {}
        )

        assert DENIED_CALL_KEY not in parsed

    def test_the_generic_denial_answer_names_the_action(
        self, request_log: dict[str, Any]
    ) -> None:
        """The last-resort net names the permission, not only the operation."""
        assert (
            denied_feature_unavailable(
                _denied_call(_DENIAL_MESSAGE, "ListProvisionedModelThroughputs")
            )
            is not None
        )

        assert any(
            "bedrock:ListProvisionedModelThroughputs" in str(detail)
            for detail in request_log["error_detail"]
        )
