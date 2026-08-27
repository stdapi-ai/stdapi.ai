"""Transport toward an Amazon SageMaker AI OpenAI-compatible endpoint.

Everything here is measurable without an endpoint: how the bearer token is
built, what URL a request goes to, how the containers' error bodies map to our
envelope, and how a scale-from-zero is absorbed. The cold start itself, the
container's real wire shapes and the usage it reports need a live endpoint and
live at ``tests/test_chat_sagemaker_endpoint.py``.

Ref: https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-openai-compatible.html
     https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-zero-instances.html
     stdapi/aws_sagemaker.py
     stdapi/aws_http.py:presigned_bearer_token
"""

from __future__ import annotations

from asyncio import Event, gather, sleep
from base64 import b64decode
from gc import collect as gc_collect
from json import dumps, loads
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlsplit

import pytest
from aiohttp import ClientError as AiohttpClientError

from stdapi import aws_sagemaker
from stdapi.aws_sagemaker import SageMakerError
from stdapi.config import AWS_SESSION, SETTINGS

if TYPE_CHECKING:
    from typing import Self

    from types_aiobotocore_bedrock.literals import RegionName

#: All tests in this module exercise the local implementation in-process.
pytestmark = pytest.mark.local

#: Endpoint name the fake transport answers for.
ENDPOINT = "stdapi-test-endpoint"

#: Inference component name the fake transport answers for.
COMPONENT = "stdapi-test-component"

#: Region the fake endpoint lives in.
REGION: RegionName = "us-east-1"

#: Console URL the front door appends to every error it wrote about the container.
CONSOLE_URL = (
    "https://us-west-2.console.aws.amazon.com/cloudwatch/home?region=us-west-2"
    "#logEventViewer:group=/aws/sagemaker/Endpoints/stdapi-test-endpoint"
)

#: How the front door quotes the body the container answered with, verbatim.
VLLM_ERROR_BODY = (
    '{"error":{"message":"Received client error (400) from Kl6NPmsGIHdsibbMO7hO '
    'with message \\"{\\"error\\":{\\"message\\":\\"max_tokens=10000000 cannot be '
    "greater than max_model_len=max_total_tokens=40960. Please request fewer "
    'output tokens. (parameter=max_tokens, value=10000000)\\",\\"type\\":'
    '\\"BadRequestError\\",\\"param\\":\\"max_tokens\\",\\"code\\":400}}\\". See '
    f'{CONSOLE_URL} in account 123456789012 for more information.",'
    '"type":"server_error","code":"model_error"}}'
)

#: The same wrapper when the front door could not read what the container answered.
UNREADABLE_CONTAINER_BODY = (
    '{"error":{"message":"Received client error (400) from Kl6NPmsGIHdsibbMO7hO '
    "and could not load the entire response body. See "
    f'{CONSOLE_URL} in account 123456789012 for more information.",'
    '"type":"server_error","code":"model_error"}}'
)

#: The body a request to an endpoint with no live copy is rejected with, verbatim.
NO_CAPACITY_BODY = (
    '{"error":{"message":"Inference Component has no capacity to process this '
    "request. ApplicationAutoScaling may be in-progress (if configured) or try "
    "to increase the capacity by invoking UpdateInferenceComponentRuntimeConfig "
    'API.","type":"invalid_request_error","code":"validation_error"}}'
)

#: The body the front door refuses an endpoint name it does not know with, verbatim.
UNKNOWN_ENDPOINT_BODY = (
    '{"error":{"message":"Endpoint stdapi-test-endpoint of account 123456789012 '
    'not found.","type":"invalid_request_error","code":"validation_error"}}'
)


def _quoting(container_body: str) -> str:
    """Wrap *container_body* the way the front door quotes what the model answered.

    Args:
        container_body: The body the container itself would have answered with.

    Returns:
        The response body the front door sends on, quoting it whole.
    """
    return dumps(
        {
            "error": {
                "message": (
                    f"Received client error (400) from Kl6NPmsGIHdsibbMO7hO with "
                    f'message "{container_body}". See {CONSOLE_URL} in account '
                    "123456789012 for more information."
                ),
                "type": "server_error",
                "code": "model_error",
            }
        }
    )


#: The message vLLM rejects a malformed request with, verbatim, traceback and all.
CONTAINER_TRACEBACK_MESSAGE = (
    "1 validation error:\n"
    "  {'type': 'value_error', 'loc': ('body',), 'msg': 'Value error, When "
    "using `tool_choice`, `tools` must be set.', 'input': {'model': '', "
    "'messages': [{'role': 'user', 'content': 'Hi'}], 'max_tokens': 4, "
    "'tool_choice': {'type': 'function', 'function': {'name': 'nope'}}}, "
    "'ctx': {'error': ValueError('When using `tool_choice`, `tools` must be "
    "set.')}}\n\n"
    '  File "/usr/local/lib/python3.12/site-packages/vllm/entrypoints/utils.py",'
    " line 40, in create_chat_completion\n"
    "    POST /v1/chat/completions [{'type': 'value_error', 'loc': ('body',), "
    "'msg': 'Value error, When using `tool_choice`, `tools` must be set.', "
    "'input': {'model': '', 'messages': [{'role': 'user', 'content': 'Hi'}], "
    "'max_tokens': 4, 'tool_choice': {'type': 'function', 'function': "
    "{'name': 'nope'}}}, 'ctx': {'error': ValueError('When using "
    "`tool_choice`, `tools` must be set.')}}]"
)

#: The same message as the endpoint sends it on, quoted whole inside its own.
CONTAINER_TRACEBACK_BODY = _quoting(
    dumps(
        {
            "error": {
                "message": CONTAINER_TRACEBACK_MESSAGE,
                "type": "Bad Request",
                "param": None,
                "code": 400,
            }
        }
    )
)


class _FakeFrozenCredentials:
    """Frozen credentials stand-in for the token presigner."""

    def __init__(self) -> None:
        self.access_key = "AKIAFAKEACCESSKEY"
        self.secret_key = "fakesecretkey"  # noqa: S105
        self.token = "faketoken"  # noqa: S105


class _FakeCredentials:
    """Credential object exposing frozen credentials."""

    async def get_frozen_credentials(self) -> _FakeFrozenCredentials:
        """Answer the frozen credentials the presigner signs with."""
        return _FakeFrozenCredentials()


class _FakeResponse:
    """Minimal ``ClientResponse`` stand-in."""

    def __init__(self, status: int = 200, body: str = "{}") -> None:
        self.status = status
        self._body = body
        self.released = False
        self.closed = False

    async def text(self) -> str:
        """Answer the raw body."""
        return self._body

    async def json(self, **kwargs: object) -> Any:  # noqa: ANN401
        """Answer the parsed body."""
        return loads(self._body)

    def release(self) -> None:
        """Record that the caller released the connection."""
        self.released = True

    def close(self) -> None:
        """Close the response, as the streaming finalizer would."""
        self.closed = True
        self.released = True

    async def __aenter__(self) -> Self:
        """Enter the response context, as aiohttp's own does."""
        return self

    async def __aexit__(self, *args: object) -> None:
        """Release the connection on exit."""
        self.release()


class _FakeContent:
    """Response body read as the line iterator the SSE reader consumes."""

    def __init__(self, lines: list[bytes | Exception]) -> None:
        self._lines = iter(lines)

    def __aiter__(self) -> Self:
        """Answer the iterator itself."""
        return self

    async def __anext__(self) -> bytes:
        """Answer the next line, or raise the scripted transport failure."""
        try:
            line = next(self._lines)
        except StopIteration:
            raise StopAsyncIteration from None
        if isinstance(line, Exception):
            raise line
        return line


class _FakeStreamResponse(_FakeResponse):
    """Streaming ``ClientResponse`` stand-in answering scripted SSE lines."""

    def __init__(self, lines: list[bytes | Exception]) -> None:
        super().__init__(200, "")
        self.content = _FakeContent(lines)


class _FakeSession:
    """Session answering a scripted sequence of responses, recording its calls."""

    def __init__(
        self, *responses: _FakeResponse | Exception, repeat: _FakeResponse | None = None
    ) -> None:
        self._responses = list(responses)
        self._repeat = repeat
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.bodies: list[bytes] = []

    async def post(
        self, url: str, *, data: bytes, headers: dict[str, str]
    ) -> _FakeResponse:
        """Answer the next scripted response for one POST."""
        self.urls.append(url)
        self.headers.append(headers)
        self.bodies.append(data)
        if self._responses:
            answer = self._responses.pop(0)
        else:
            answer = self._repeat or _FakeResponse()
        if isinstance(answer, Exception):
            raise answer
        return answer


class _Clock:
    """A monotonic clock that only advances when the code under test sleeps.

    The warm-up loop is bounded by wall-clock time, so its behaviour is only
    testable with the clock under the test's control: a real one would make
    the assertions depend on how fast the machine runs them.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        """Answer the current time."""
        return self.now

    async def sleep(self, delay: float) -> None:
        """Advance the clock by *delay* and yield to the event loop."""
        self.slept.append(delay)
        self.now += delay
        await sleep(0)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _Clock:
    """Drive the warm-up loop on a stubbed clock."""
    stub = _Clock()
    monkeypatch.setattr(aws_sagemaker, "monotonic", stub)
    monkeypatch.setattr(aws_sagemaker, "sleep", stub.sleep)
    return stub


@pytest.fixture
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Presign every token from fixed credentials, with an empty cache."""

    async def fake_get_credentials() -> _FakeCredentials:
        return _FakeCredentials()

    monkeypatch.setattr(AWS_SESSION, "get_credentials", fake_get_credentials)
    monkeypatch.setattr(aws_sagemaker, "_TOKENS", {})


class TestBearerToken:
    """The bearer token is a presigned ``CallWithBearerToken`` request.

    Plain SigV4 on the invocation route is answered with 403: the endpoint takes
    an API key whose payload is a presigned URL against the *control-plane*
    host, signed for the ``sagemaker`` service.

    Ref: https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-openai-compatible.html
         stdapi/aws_sagemaker.py:bearer_token
    """

    async def test_token_carries_the_sagemaker_prefix_and_presigned_url(
        self, credentials: None
    ) -> None:
        """The prefix, host, action, signing service and version are all fixed."""
        del credentials

        token = await aws_sagemaker.bearer_token(REGION)

        assert token.startswith("sagemaker-api-key-")
        decoded = b64decode(token.removeprefix("sagemaker-api-key-")).decode()
        assert decoded.startswith("sagemaker.amazonaws.com/?")
        assert decoded.endswith("&Version=1")
        query = parse_qs(urlsplit(f"https://{decoded}").query)
        assert query["Action"] == ["CallWithBearerToken"]
        assert query["X-Amz-Algorithm"] == ["AWS4-HMAC-SHA256"]
        assert f"/{REGION}/sagemaker/aws4_request" in query["X-Amz-Credential"][0]
        assert query["X-Amz-Security-Token"] == ["faketoken"]

    async def test_token_differs_from_the_bedrock_mantle_one(
        self, credentials: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two transports share a scheme, not a token.

        A token signed for ``bedrock`` is rejected by SageMaker AI and the
        other way round, so the shared presigner must be given both constants.
        """
        del credentials
        # Imported here: only this test compares the two transports.
        from stdapi import aws_bedrock_mantle  # noqa: PLC0415

        monkeypatched = await aws_sagemaker.bearer_token(REGION)
        # Replaced, never cleared: minting from these fake credentials into the
        # other module's live cache would poison every later Mantle call on
        # this worker for the whole token TTL.
        monkeypatch.setattr(aws_bedrock_mantle, "_TOKENS", {})
        mantle = await aws_bedrock_mantle.bearer_token(REGION)

        assert not mantle.startswith("sagemaker-api-key-")
        assert monkeypatched != mantle

    async def test_token_is_cached_per_region(self, credentials: None) -> None:
        """One token is minted per region and reused until its TTL expires."""
        del credentials

        first = await aws_sagemaker.bearer_token(REGION)
        again = await aws_sagemaker.bearer_token(REGION)
        west = await aws_sagemaker.bearer_token("us-west-2")

        assert first == again
        assert first != west
        assert (
            "us-west-2" in b64decode(west.removeprefix("sagemaker-api-key-")).decode()
        )


class TestRequestUrl:
    """The endpoint and the inference component are named in the URL path.

    ``X-Amzn-SageMaker-Inference-Component`` is rejected on this route whatever
    its value, so the component can only reach the endpoint through the path.

    Ref: stdapi/aws_sagemaker.py:invocation_path
    """

    def test_path_names_the_endpoint_and_the_component(self) -> None:
        """Both names sit in the path, ahead of the OpenAI route."""
        assert aws_sagemaker.invocation_path(ENDPOINT, COMPONENT) == (
            f"/endpoints/{ENDPOINT}/inference-components/{COMPONENT}"
            "/openai/v1/chat/completions"
        )

    def test_path_without_a_component_addresses_the_endpoint(self) -> None:
        """An endpoint hosting a model directly has no component segment."""
        assert aws_sagemaker.invocation_path(ENDPOINT) == (
            f"/endpoints/{ENDPOINT}/openai/v1/chat/completions"
        )

    def test_names_are_escaped_into_the_path(self) -> None:
        """A name is percent-encoded rather than interpolated raw.

        The names come from operator configuration, so this is defence in
        depth: a value carrying ``/`` or ``?`` must not re-target the request.
        """
        path = aws_sagemaker.invocation_path("a/b", "c?d")

        assert "/endpoints/a%2Fb/inference-components/c%3Fd/" in path

    def test_endpoint_url_is_resolved_per_partition(self) -> None:
        """The runtime host follows the region's partition, not a fixed suffix.

        A SageMaker AI endpoint resolves in every partition, which is the
        reason this backend reaches deployments Bedrock Mantle cannot. The
        host is the one the service's endpoint rules state, not the legacy
        resolver's ``sagemaker-runtime.<region>.<suffix>``, which does not
        exist.

        Ref: botocore/data/sagemaker-runtime/2017-05-13/endpoint-rule-set-1.json
        """
        assert aws_sagemaker.endpoint_url(REGION) == (
            "https://runtime.sagemaker.us-east-1.amazonaws.com"
        )
        # Regions of the other partitions, which the Bedrock region literal
        # this transport shares does not enumerate.
        for region, expected in (
            ("cn-north-1", "https://runtime.sagemaker.cn-north-1.amazonaws.com.cn"),
            ("eusc-de-east-1", "https://runtime.sagemaker.eusc-de-east-1.amazonaws.eu"),
            ("us-gov-west-1", "https://runtime.sagemaker.us-gov-west-1.amazonaws.com"),
        ):
            assert aws_sagemaker.endpoint_url(region) == expected  # type: ignore[arg-type]

    def test_configured_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A VPC-endpoint template replaces the resolved host, per region."""
        monkeypatch.setattr(
            SETTINGS, "aws_sagemaker_endpoint_url", "https://vpce.{region}.example/"
        )

        assert aws_sagemaker.endpoint_url(REGION) == "https://vpce.us-east-1.example"

    async def test_request_targets_the_resolved_url_with_a_bearer_token(
        self, credentials: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The composed request carries the token and the JSON content type."""
        del credentials
        session = _FakeSession(_FakeResponse())
        monkeypatch.setattr(aws_sagemaker, "_SESSION", session)

        await aws_sagemaker.invoke(REGION, ENDPOINT, COMPONENT, {"model": ""})

        expected = (
            "https://runtime.sagemaker.us-east-1.amazonaws.com"
            f"/endpoints/{ENDPOINT}/inference-components/{COMPONENT}"
            "/openai/v1/chat/completions"
        )
        assert session.urls == [expected]
        assert session.headers[0]["Authorization"].startswith(
            "Bearer sagemaker-api-key-"
        )
        assert session.headers[0]["Content-Type"] == "application/json"


class TestErrorMapping:
    """Container and front-door error bodies map to the API envelope.

    Two writers answer on this route and only the container describes the
    caller's own request; the front door describes the operator's account. Both
    use OpenAI's envelope, so the one thing that tells them apart is that the
    front door quotes the container's whole body inside its own message.

    Ref: stdapi/aws_sagemaker.py:_map_error
    """

    def test_the_quoted_container_body_is_what_the_caller_reads(self) -> None:
        """The container's rejection is unwrapped from the front door's quote.

        The body is what a real endpoint answered a ``max_tokens`` above its
        context window with, and everything the front door added around it --
        the variant identifier, the CloudWatch console URL, the account ID --
        belongs to the operator, not to the caller who sent a bad parameter.
        """
        error = aws_sagemaker._map_error(400, VLLM_ERROR_BODY, REGION)  # noqa: SLF001

        assert isinstance(error, SageMakerError)
        assert error.status == 400
        assert "max_tokens=10000000 cannot be greater than" in str(error)
        assert error.param == "max_tokens"
        assert error.no_capacity is False
        assert "123456789012" not in str(error)
        assert "console.aws.amazon.com" not in str(error)
        assert "Received client error" not in str(error)

    def test_an_unquoted_openai_envelope_is_the_front_door_s(
        self, request_log: dict[str, Any]
    ) -> None:
        """The shape is no evidence: only a quoted body is the container's.

        Measured against a real endpoint, the front door answers its own
        refusals in OpenAI's ``{"error": ...}`` envelope too, so an envelope
        that quotes nothing is the front door describing the operator's
        infrastructure and is never forwarded.
        """
        error = aws_sagemaker._map_error(400, UNKNOWN_ENDPOINT_BODY, REGION)  # noqa: SLF001

        assert error.status == 400
        assert "123456789012" not in str(error)
        assert ENDPOINT not in str(error)
        assert "123456789012" in str(request_log["error_detail"])

    def test_an_unreadable_container_body_leaks_nothing_either(
        self, request_log: dict[str, Any]
    ) -> None:
        """The front door quotes nothing when it could not read the answer.

        Its message then carries only its own text -- the account ID and a
        console URL for the operator's logs -- so there is nothing in it the
        caller may read.
        """
        error = aws_sagemaker._map_error(400, UNREADABLE_CONTAINER_BODY, REGION)  # noqa: SLF001

        assert error.status == 400
        assert "123456789012" not in str(error)
        assert "console.aws.amazon.com" not in str(error)
        assert ENDPOINT not in str(error)
        assert "console.aws.amazon.com" in str(request_log["error_detail"])

    def test_front_door_message_reaches_the_operator_only(
        self, request_log: dict[str, Any]
    ) -> None:
        """The runtime's not-found body names the endpoint and the account ID.

        Neither is the caller's to read (AGENTS.md, *Never Leak Internals*),
        and neither is anything they could act on; the operator needs both, so
        the upstream text goes to the server log instead.

        Ref: botocore/data/sagemaker-runtime/2017-05-13/service-2.json
        """
        body = (
            '{"Message": "Endpoint stdapi-test-endpoint of account '
            '123456789012 not found."}'
        )

        error = aws_sagemaker._map_error(400, body, REGION)  # noqa: SLF001

        assert error.status == 400
        assert "123456789012" not in str(error)
        assert ENDPOINT not in str(error)
        assert "123456789012" in str(request_log["error_detail"])

    def test_unparseable_body_is_not_forwarded_either(
        self, request_log: dict[str, Any]
    ) -> None:
        """An HTML or plain-text body is an intermediary's, not the container's."""
        error = aws_sagemaker._map_error(502, "Bad Gateway", REGION)  # noqa: SLF001

        assert error.status == 502
        assert "Bad Gateway" not in str(error)
        assert "temporarily unavailable" in str(error)
        assert "Bad Gateway" in str(request_log["error_detail"])

    def test_denial_is_generic_to_the_caller_and_evicts_the_token(
        self, monkeypatch: pytest.MonkeyPatch, request_log: dict[str, Any]
    ) -> None:
        """A refused role is the operator's problem, never the caller's.

        The message names no permission and no ARN; the warning log carries
        both, and the cached token is dropped so a rotated credential heals.
        """
        monkeypatch.setitem(aws_sagemaker._TOKENS, REGION, ("stale", 1e18))  # noqa: SLF001

        error = aws_sagemaker._map_error(403, '{"message": "role X denied"}', REGION)  # noqa: SLF001

        assert error.status == 503
        assert error.code == "feature_unavailable"
        assert "denied" not in str(error)
        assert "sagemaker:CallWithBearerToken" in str(request_log["error_detail"])
        assert REGION not in aws_sagemaker._TOKENS  # noqa: SLF001


class TestContainerMessageSanitizing:
    """A container's message reaches the caller without its traceback.

    What the container says about the request is the only account of why it was
    refused, so it is forwarded; the traceback it may append is not, because the
    frames name the runtime version and the library layout of a server nobody
    published, to anyone able to send a malformed request.

    The rules are structural -- a traceback frame, a rooted path -- so they hold
    for a deployment serving something other than vLLM.

    Ref: stdapi/aws_sagemaker.py:_sanitize_container_message
    """

    def test_the_traceback_goes_and_the_complaint_stays(self) -> None:
        """The measured body: a pydantic complaint with a frame stapled to it.

        The caller keeps every part it can act on -- which parameter, why, and
        the value it sent -- and reads nothing of where the server keeps its
        files.
        """
        error = aws_sagemaker._map_error(400, CONTAINER_TRACEBACK_BODY, REGION)  # noqa: SLF001

        assert "When using `tool_choice`, `tools` must be set." in str(error)
        assert "1 validation error:" in str(error)
        assert 'File "' not in str(error)
        assert "/usr/local/lib" not in str(error)
        assert "python3.12" not in str(error)
        assert "create_chat_completion" not in str(error)

    def test_a_message_with_no_traceback_is_untouched(self) -> None:
        """The measured clean body reaches the caller exactly as written."""
        message = (
            "max_tokens=10000000 cannot be greater than "
            "max_model_len=max_total_tokens=40960. Please request fewer output "
            "tokens. (parameter=max_tokens, value=10000000)"
        )

        assert aws_sagemaker._sanitize_container_message(message) == message  # noqa: SLF001

    def test_a_clean_multi_line_message_passes_through_byte_for_byte(self) -> None:
        """Nothing is reflowed, re-indented or trimmed out of a clean message."""
        message = (
            "The request could not be served:\n"
            "  - `top_p` must be in (0, 1]\n"
            "  - `n` must be 1 for this route\n"
            "See /v1/chat/completions for the accepted parameters."
        )

        assert aws_sagemaker._sanitize_container_message(message) == message  # noqa: SLF001

    def test_a_message_that_is_only_a_traceback_leaves_a_message_anyway(self) -> None:
        """Stripping everything must still answer something a caller can read."""
        message = (
            "Traceback (most recent call last):\n"
            '  File "/opt/serve/app/handler.py", line 118, in generate\n'
            "    raise RuntimeError(weights)\n"
            '  File "/opt/serve/app/engine.py", line 41, in load\n'
            "    return self._weights\n"
        )

        sanitized = aws_sagemaker._sanitize_container_message(message)  # noqa: SLF001

        assert sanitized
        assert 'File "' not in sanitized
        assert "/opt/serve" not in sanitized
        assert "Traceback" not in sanitized
        assert "rejected the request" in sanitized


class TestNoCapacityClassifier:
    """Only a real no-capacity rejection may hold a caller for minutes.

    A classifier too permissive would turn an ordinary outage into a
    five-minute hang, which is the worst failure this feature can produce.

    Ref: stdapi/aws_sagemaker.py:_map_error
    """

    def test_no_capacity_400_is_recognised(self) -> None:
        """The measured rejection: HTTP 400, in about a second, at zero copies.

        The body is what a real endpoint at zero copies answered on
        2026-08-27, and it is OpenAI's own ``error`` envelope: the front door
        writes the same shape the container does on this route, so the shape
        can never be part of this classification.
        """
        error = aws_sagemaker._map_error(400, NO_CAPACITY_BODY, REGION)  # noqa: SLF001

        assert isinstance(error, SageMakerError)
        assert error.no_capacity is True
        # The front door's own wording names the component and the API that
        # would resize it, neither of which is the caller's to read.
        assert "Inference Component" not in str(error)
        assert "UpdateInferenceComponentRuntimeConfig" not in str(error)

    @pytest.mark.parametrize(
        ("status", "body"),
        [
            (400, VLLM_ERROR_BODY),
            # The front door's other 400s share the envelope and the error code.
            (400, UNKNOWN_ENDPOINT_BODY),
            (400, '{"message": "Endpoint not found"}'),
            (503, NO_CAPACITY_BODY),
            (500, NO_CAPACITY_BODY),
            (429, '{"message": "Too many requests"}'),
            # The front door names its subject; a bare phrase is not its wording.
            (400, '{"message": "no capacity"}'),
        ],
    )
    def test_everything_else_is_not_a_cold_start(self, status: int, body: str) -> None:
        """A different status, or a different message, fails fast instead."""
        error = aws_sagemaker._map_error(status, body, REGION)  # noqa: SLF001

        assert isinstance(error, SageMakerError)
        assert error.no_capacity is False

    @pytest.mark.parametrize(
        "body",
        [
            # vLLM echoes the offending value into its validation message.
            _quoting(
                '{"message": "Named tool \'the endpoint has no capacity\' not '
                'found in tool list", "type": "BadRequestError", "code": 400}'
            ),
            _quoting(
                '{"error": {"message": "Input should be a valid number '
                "[input_value='the endpoint has no capacity']\", "
                '"code": "invalid_value"}}'
            ),
            _quoting(
                '{"error": {"message": "\'Endpoint has no capacity\' is not a '
                'valid tool name", "code": "invalid_value"}}'
            ),
        ],
    )
    def test_client_text_echoed_by_the_container_is_never_a_cold_start(
        self, body: str
    ) -> None:
        """A caller must not be able to put itself into a ten-minute warm-up.

        The container quotes the request back in its validation errors, so a
        client sending the front door's own phrasing as a parameter value could
        otherwise have its own 400 read as a scale-from-zero: the connection
        would be held for the whole warm-up budget, probes fired at a healthy
        endpoint, and the real 400 masked by a 503 -- repeatable at will.

        The envelope cannot say who wrote the body here, so the guard is that
        the front door's wording opens its message and echoed text never does.
        """
        error = aws_sagemaker._map_error(400, body, REGION)  # noqa: SLF001

        assert isinstance(error, SageMakerError)
        assert error.no_capacity is False
        # A container error does describe the caller's own request, so it shows.
        assert "no capacity" in str(error)


class TestColdStartWait:
    """A scale-from-zero is absorbed before the response object exists.

    The request that is refused for want of capacity is itself what makes
    SageMaker AI provision an instance again, so the transport's whole job is
    to wait and re-send.

    Ref: https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-zero-instances.html
         stdapi/aws_sagemaker.py:_request_with_warmup
    """

    async def test_a_cold_endpoint_is_waited_for_and_retried(
        self, credentials: None, clock: _Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The caller sees a slow success, never the cold-start rejection.

        Both the refused attempt and the successful probe give their pooled
        connection back at once: under a cold-start storm every request is one
        of the two, which is exactly when connections are scarcest.
        """
        del credentials
        rejected = _FakeResponse(400, NO_CAPACITY_BODY)  # the caller's first attempt
        probe_cold = _FakeResponse(400, NO_CAPACITY_BODY)  # first probe: still cold
        probe_warm = _FakeResponse(200, "{}")  # second probe: capacity is back
        session = _FakeSession(
            rejected,
            probe_cold,
            probe_warm,
            _FakeResponse(200, '{"id": "chatcmpl-1"}'),  # the caller's retry
        )
        monkeypatch.setattr(aws_sagemaker, "_SESSION", session)
        monkeypatch.setattr(aws_sagemaker, "_WARMING", {})
        monkeypatch.setattr(SETTINGS, "aws_sagemaker_warmup_timeout", 600)

        result = await aws_sagemaker.invoke(REGION, ENDPOINT, COMPONENT, {"model": ""})

        assert result == {"id": "chatcmpl-1"}
        assert len(session.urls) == 4
        assert clock.now < 600
        assert rejected.released is True
        assert probe_cold.released is True
        assert probe_warm.released is True

    async def test_a_warm_probe_and_a_refused_request_stop_looping(
        self, credentials: None, clock: _Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A probe reporting capacity while the request keeps failing is bounded.

        That pair is either a race for the single copy that just came up, or a
        rejection this only looks like. A few rounds cover the race; without a
        bound the second holds the caller for the entire budget and answers a
        503 that masks the real error.
        """
        del credentials

        class _WarmProbeSession:
            def __init__(self) -> None:
                self.requests = 0

            async def post(
                self, url: str, *, data: bytes, headers: dict[str, str]
            ) -> _FakeResponse:
                del url, headers
                if b'"max_tokens":1' in data.replace(b" ", b""):
                    return _FakeResponse(200, "{}")
                self.requests += 1
                return _FakeResponse(400, NO_CAPACITY_BODY)

        session = _WarmProbeSession()
        monkeypatch.setattr(aws_sagemaker, "_SESSION", session)
        monkeypatch.setattr(aws_sagemaker, "_WARMING", {})
        monkeypatch.setattr(SETTINGS, "aws_sagemaker_warmup_timeout", 600)

        with pytest.raises(SageMakerError) as exc_info:
            await aws_sagemaker.invoke(REGION, ENDPOINT, COMPONENT, {"model": ""})

        assert exc_info.value.no_capacity is True
        # Surfaced without the budget running out, it is still the retriable
        # status: the caller learns nothing from the front door's own 400.
        assert exc_info.value.status == 503
        assert session.requests == aws_sagemaker._MAX_WARMUP_RETRIES + 1  # noqa: SLF001
        assert clock.now < 600

    async def test_exhausted_budget_answers_503_without_leaking_the_cause(
        self, credentials: None, clock: _Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A budget that runs out is a real HTTP status, not a hung request.

        The message tells the caller to retry and names nothing of the backend;
        the endpoint, the elapsed budget and the missing-alarm diagnosis go to
        the operator's log instead. The loop stops at the budget rather than
        one probe past it.
        """
        del credentials
        # Every attempt stays cold, so only the budget can end the loop.
        monkeypatch.setattr(
            aws_sagemaker,
            "_SESSION",
            _FakeSession(repeat=_FakeResponse(400, NO_CAPACITY_BODY)),
        )
        monkeypatch.setattr(aws_sagemaker, "_WARMING", {})
        monkeypatch.setattr(SETTINGS, "aws_sagemaker_warmup_timeout", 30)

        with pytest.raises(SageMakerError) as exc_info:
            await aws_sagemaker.invoke(REGION, ENDPOINT, COMPONENT, {"model": ""})

        assert exc_info.value.status == 503
        assert "starting up" in str(exc_info.value)
        assert ENDPOINT not in str(exc_info.value)
        assert clock.now == 30

    async def test_zero_budget_disables_the_wait(
        self, credentials: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``0`` restores fail-fast for an operator who prefers an error.

        The error is a `503`, not the front door's `400`: a cold endpoint is a
        transient condition, and an SDK retries a `503` where it never retries
        a `400`.
        """
        del credentials
        session = _FakeSession(_FakeResponse(400, NO_CAPACITY_BODY))
        monkeypatch.setattr(aws_sagemaker, "_SESSION", session)
        monkeypatch.setattr(SETTINGS, "aws_sagemaker_warmup_timeout", 0)

        with pytest.raises(SageMakerError) as exc_info:
            await aws_sagemaker.invoke(REGION, ENDPOINT, COMPONENT, {"model": ""})

        assert exc_info.value.status == 503
        assert len(session.urls) == 1

    async def test_an_ordinary_error_is_not_waited_on(
        self, credentials: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A validation error is answered immediately, budget or no budget."""
        del credentials
        session = _FakeSession(_FakeResponse(400, VLLM_ERROR_BODY))
        monkeypatch.setattr(aws_sagemaker, "_SESSION", session)
        monkeypatch.setattr(SETTINGS, "aws_sagemaker_warmup_timeout", 600)

        with pytest.raises(SageMakerError) as exc_info:
            await aws_sagemaker.invoke(REGION, ENDPOINT, COMPONENT, {"model": ""})

        assert "max_tokens=10000000" in str(exc_info.value)
        assert len(session.urls) == 1


class TestColdStartCoalescing:
    """Concurrent callers of one cold endpoint share a single warm-up.

    Ten requests must not become ten independent five-minute loops, and the
    probe belongs to none of them: a client hanging up must not strand the
    others.

    Ref: stdapi/aws_sagemaker.py:_wait_for_capacity
    """

    async def test_concurrent_requests_wait_on_one_probe(
        self, credentials: None, clock: _Clock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Three callers produce three rejections, one probe and three retries.

        Without coalescing this would be three independent probe loops, each
        adding load to an endpoint that by definition has none to give.
        """
        del credentials, clock
        warm = Event()
        calls: list[str] = []

        class _CountingSession:
            async def post(
                self, url: str, *, data: bytes, headers: dict[str, str]
            ) -> _FakeResponse:
                probe = b'"max_tokens":1' in data.replace(b" ", b"")
                calls.append("probe" if probe else "request")
                if probe:
                    warm.set()
                if warm.is_set():
                    return _FakeResponse(200, '{"id": "chatcmpl-1"}')
                return _FakeResponse(400, NO_CAPACITY_BODY)

        monkeypatch.setattr(aws_sagemaker, "_SESSION", _CountingSession())
        monkeypatch.setattr(aws_sagemaker, "_WARMING", {})
        monkeypatch.setattr(SETTINGS, "aws_sagemaker_warmup_timeout", 600)

        results = await gather(
            *(
                aws_sagemaker.invoke(REGION, ENDPOINT, COMPONENT, {"model": ""})
                for _ in range(3)
            )
        )

        assert all(result == {"id": "chatcmpl-1"} for result in results)
        assert calls.count("probe") == 1
        assert calls.count("request") == 6

    async def test_a_disconnecting_client_does_not_cancel_the_shared_probe(
        self, credentials: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The probe outlives the request that started it.

        Awaiting a task propagates cancellation into it, which would let the
        first caller to hang up abort the wait every other caller is on.
        """
        del credentials
        started = Event()

        async def never_ready(*args: object, **kwargs: object) -> bool:  # noqa: ARG001
            started.set()
            await Event().wait()
            return False

        monkeypatch.setattr(aws_sagemaker, "_WARMING", {})
        monkeypatch.setattr(aws_sagemaker, "_watch_warm_up", never_ready)
        waiter = aws_sagemaker._wait_for_capacity(REGION, ENDPOINT, COMPONENT, 1e18)  # noqa: SLF001
        task = _spawn(waiter)
        await started.wait()
        probe = aws_sagemaker._WARMING[(REGION, ENDPOINT, COMPONENT)]  # noqa: SLF001

        task.cancel()
        await sleep(0)

        assert task.cancelled()
        assert not probe.cancelled()
        probe.cancel()


def _spawn(coroutine: Any) -> Any:  # noqa: ANN401
    """Schedule *coroutine* as a task on the running loop."""
    # Imported here: only this helper needs the loop accessor.
    from asyncio import ensure_future  # noqa: PLC0415

    return ensure_future(coroutine)


class TestConnectionFailure:
    """A transport failure never reaches the caller as an aiohttp error.

    Ref: stdapi/aws_sagemaker.py:_request
    """

    async def test_connection_error_maps_to_503(
        self,
        credentials: None,
        monkeypatch: pytest.MonkeyPatch,
        request_log: dict[str, Any],
    ) -> None:
        """An unreachable endpoint answers the caller a generic 503."""
        del credentials, request_log
        monkeypatch.setattr(
            aws_sagemaker, "_SESSION", _FakeSession(AiohttpClientError())
        )

        with pytest.raises(SageMakerError) as exc_info:
            await aws_sagemaker.invoke(REGION, ENDPOINT, COMPONENT, {"model": ""})

        assert exc_info.value.status == 503
        assert exc_info.value.no_capacity is False

    async def test_an_error_response_releases_its_connection(
        self, credentials: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A refused request gives its pooled connection back immediately."""
        del credentials
        rejected = _FakeResponse(400, VLLM_ERROR_BODY)
        monkeypatch.setattr(aws_sagemaker, "_SESSION", _FakeSession(rejected))
        monkeypatch.setattr(SETTINGS, "aws_sagemaker_warmup_timeout", 0)

        with pytest.raises(SageMakerError):
            await aws_sagemaker.invoke(REGION, ENDPOINT, COMPONENT, {"model": ""})

        assert rejected.released is True


class TestStreaming:
    """A streamed invocation owns its upstream connection to the end.

    Ref: stdapi/aws_sagemaker.py:invoke_stream
         stdapi/aws_http.py:iter_sse
    """

    async def test_an_abandoned_generator_closes_the_response(
        self, credentials: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A generator dropped before its first iteration still closes.

        The ``async with`` guarding the response only runs once the generator
        body starts, so a client that opens a stream and disconnects before the
        first chunk would otherwise leak one pooled connection per request.
        """
        del credentials
        response = _FakeStreamResponse([b'data: {"id": "1"}\n', b"\n"])
        monkeypatch.setattr(aws_sagemaker, "_SESSION", _FakeSession(response))

        generator = await aws_sagemaker.invoke_stream(
            REGION, ENDPOINT, COMPONENT, {"model": ""}
        )
        assert response.closed is False

        del generator
        gc_collect()

        assert response.closed is True

    async def test_a_dropped_stream_is_this_transport_s_own_error(
        self, credentials: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mid-stream drop maps to a 502 raised as this backend's error.

        The SSE reader is shared with Amazon Bedrock Mantle and is told which
        class to raise; passing the wrong one would brand a SageMaker AI
        failure as a Mantle one everywhere the two are told apart.
        """
        del credentials
        response = _FakeStreamResponse(
            [b'data: {"id": "1"}\n', b"\n", AiohttpClientError()]
        )
        monkeypatch.setattr(aws_sagemaker, "_SESSION", _FakeSession(response))

        generator = await aws_sagemaker.invoke_stream(
            REGION, ENDPOINT, COMPONENT, {"model": ""}
        )
        events: list[tuple[str | None, str]] = []

        async def drain() -> None:
            async for event in generator:
                events.append(event)  # noqa: PERF401 - the raise is the subject

        with pytest.raises(SageMakerError) as exc_info:
            await drain()

        assert events == [(None, '{"id": "1"}')]
        assert exc_info.value.status == 502
        assert "stream was interrupted" in str(exc_info.value)


class TestSessionLifespan:
    """The shared session and everything it owns end with the server's lifespan.

    Ref: stdapi/aws_sagemaker.py:sagemaker_http_session
    """

    async def test_shutdown_cancels_a_warm_up_probe_in_flight(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A probe still sleeping at shutdown is cancelled, never orphaned.

        Dropping the dict entry alone leaves the task scheduled: it wakes up to
        one probe interval later and, in a process that reopens the lifespan,
        sends a request through a session it does not own.
        """
        started = Event()

        async def never_ready(*args: object, **kwargs: object) -> bool:  # noqa: ARG001
            started.set()
            await Event().wait()
            return False

        monkeypatch.setattr(aws_sagemaker, "_SESSION", None)
        monkeypatch.setattr(aws_sagemaker, "_TOKENS", {})
        monkeypatch.setattr(aws_sagemaker, "_WARMING", {})
        monkeypatch.setattr(aws_sagemaker, "_watch_warm_up", never_ready)

        async with aws_sagemaker.sagemaker_http_session() as session:
            waiter = _spawn(
                aws_sagemaker._wait_for_capacity(REGION, ENDPOINT, COMPONENT, 1e18)  # noqa: SLF001
            )
            await started.wait()
            probe = aws_sagemaker._WARMING[(REGION, ENDPOINT, COMPONENT)]  # noqa: SLF001
        for _ in range(3):
            await sleep(0)

        assert probe.cancelled()
        assert not aws_sagemaker._WARMING  # noqa: SLF001
        assert session.closed
        assert aws_sagemaker._SESSION is None  # noqa: SLF001
        waiter.cancel()

    async def test_a_second_opener_reuses_the_server_s_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A nested opener never closes the session out from under the server."""
        monkeypatch.setattr(aws_sagemaker, "_SESSION", None)
        monkeypatch.setattr(aws_sagemaker, "_TOKENS", {})
        monkeypatch.setattr(aws_sagemaker, "_WARMING", {})

        async with aws_sagemaker.sagemaker_http_session() as outer:
            async with aws_sagemaker.sagemaker_http_session() as inner:
                assert inner is outer
            assert not outer.closed

        assert outer.closed
