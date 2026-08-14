"""The shared answer to a feature this deployment cannot run.

A missing IAM permission, an unconfigured bucket or an unreachable endpoint are
none of the caller's doing: they read one generic 503, identical whatever is
absent, while the operator reads what is missing by name in the server log. One
helper owns both halves so the two messages cannot drift apart per feature.

Ref: https://developers.openai.com/api/docs/guides/error-codes.md
     stdapi/api_errors.py:FeatureUnavailableError
     stdapi/api_errors.py:feature_unavailable_guard
"""

from typing import Any

import pytest
from botocore.exceptions import (
    ClientError,
    ConnectTimeoutError,
    EndpointConnectionError,
    EndpointResolutionError,
)

from stdapi.api_errors import (
    ApiError,
    FeatureUnavailableError,
    denied_feature_unavailable,
    feature_unavailable_guard,
)

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
