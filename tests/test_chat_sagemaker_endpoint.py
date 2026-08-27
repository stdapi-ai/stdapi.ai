"""Serving a really deployed Amazon SageMaker AI endpoint.

**This lane provisions nothing.** It runs against the scale-to-zero endpoint
defined in ``terraform-sandbox`` and named by ``TEST_SAGEMAKER_ENDPOINT`` in
``tests/.env``; without that key every test here skips, so a normal run never
touches paid infrastructure and a torn-down sandbox reads as a skip rather than
twenty failures.

What only a real endpoint can prove, and what the whole lane exists for:

- the **cold start is absorbed** -- an endpoint at zero copies rejects a request
  in about a second, that rejection is what makes AWS provision an instance, and
  the caller must see a slow success rather than the rejection;
- **concurrent callers share one wait** rather than each starting their own;
- the container's real **SSE shape**, **finish reasons**, **usage block** and
  **error JSON**, none of which are OpenAI's own and none of which a stub shaped
  like OpenAI would reproduce.

**Budget for a run that ends dirtier than it started.** The teardown asks for
zero copies, and Application Auto Scaling then puts one back: the cold-start
test's own probes leave the ``NoCapacityInvocationFailures`` alarm in ALARM, so
the step scaling policy re-scales the component before the alarm clears.
Measured on 2026-08-27: teardown at 18:57, a copy live again at 19:10, back to
zero at 19:13. Expect **10 to 15 instance-minutes of GPU time after the run
reports green**, on top of the cold start the run itself pays for. Nothing here
should try to prevent it -- writing the copy count against the policy that owns
it only makes the fight longer.

Nothing here asserts ``reasoning_content``: the container sets a tool-call
parser and no reasoning parser, so a reasoning model's ``<think>`` block comes
back inline in ``content`` with ``reasoning`` null. That is the container's
configuration, not the gateway's behaviour.

Ref: https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-openai-compatible.html
     https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-zero-instances.html
     stdapi/aws_sagemaker.py
     stdapi/models/sagemaker_endpoints.py
"""

from asyncio import gather, sleep, to_thread
from json import loads
from re import compile as compile_regex
from time import monotonic
from typing import TYPE_CHECKING, Any

import pytest
from aiobotocore.session import get_session
from botocore.exceptions import ClientError

from stdapi import aws_sagemaker
from stdapi.config import SETTINGS
from stdapi.models import (
    SAGEMAKER_ENDPOINT_MODELS,
    SAGEMAKER_SERVICE,
    initialize_bedrock_models,
)
from tests.conftest import logged_usage_entries

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from starlette.testclient import TestClient as TestClientType
    from types_aiobotocore_bedrock.literals import RegionName

    from tests.conftest import SageMakerSandboxEndpoint

#: Real money on a GPU instance, and a cold start measured in minutes.
pytestmark = [
    pytest.mark.expensive,
    pytest.mark.slow,
    pytest.mark.gateway("A SageMaker AI endpoint is an AWS-only backend"),
    # One endpoint, one worker: these tests share a single paid resource, and
    # one of them deliberately takes its capacity away.
    pytest.mark.xdist_group("sagemaker_endpoint"),
]

#: How long to wait for the inference component to release its last copy.
_SCALE_IN_TIMEOUT = 900.0

#: Seconds between two readiness reads while scaling in.
_SCALE_IN_POLL = 15.0

#: Component statuses meaning a runtime-config update has not settled yet.
_UNSETTLED_STATUSES = frozenset({"Creating", "Updating", "Deleting"})

#: How long to wait for a runtime-config update already in flight to settle.
_SETTLE_TIMEOUT = 600.0

#: Seconds between two status reads while an update settles.
_SETTLE_POLL = 10.0

#: Times a copy-count write is re-attempted after losing a race with auto scaling.
_UPDATE_ATTEMPTS = 3

#: What SageMaker AI refuses a copy-count write already in flight with.
_UPDATE_IN_PROGRESS = "UPDATE_RC_IN_PROGRESS"

#: A rooted filesystem path naming a file, which no answer to a caller may carry.
_ABSOLUTE_PATH_RE = compile_regex(
    r"(?:[A-Za-z]:)?[/\\](?:[\w.+-]+[/\\]){2,}[\w.+-]*\.\w+"
)

#: Concurrent callers fired at the cold endpoint.
_CONCURRENT_CALLERS = 3

#: A warm answer is seconds; a cold start is minutes. Anything above this many
#: seconds could not have been served without a wait.
_COLD_START_FLOOR = 30.0


async def _sagemaker_client(region: str) -> AsyncIterator[Any]:
    """Open a SageMaker AI control-plane client for the test's own resource.

    The gateway never calls a mutating SageMaker AI API -- driving an operator's
    inference fleet is not its business, and it would fight Application Auto
    Scaling over the same value. A fixture acting on the sandbox's own endpoint
    is a different actor, and this client is its own.

    Args:
        region: Region the endpoint lives in.

    Yields:
        The control-plane client.
    """
    async with get_session().create_client("sagemaker", region_name=region) as client:
        yield client


async def _runtime_config(endpoint: SageMakerSandboxEndpoint) -> tuple[str, int, int]:
    """Return the component's status and its desired and live copy counts.

    Args:
        endpoint: The endpoint under test.

    Returns:
        Tuple of (``InferenceComponentStatus``, ``DesiredCopyCount``,
        ``CurrentCopyCount``); a current count of ``0`` means scaled to zero.
    """
    async for client in _sagemaker_client(endpoint.region):
        detail = await client.describe_inference_component(
            InferenceComponentName=endpoint.inference_component
        )
        runtime = detail["RuntimeConfig"]
        return (
            detail["InferenceComponentStatus"],
            int(runtime.get("DesiredCopyCount", 0)),
            int(runtime.get("CurrentCopyCount", 0)),
        )
    raise AssertionError  # pragma: no cover - the generator always yields once


async def _copy_count(endpoint: SageMakerSandboxEndpoint) -> int:
    """Return how many copies of the inference component are live.

    Args:
        endpoint: The endpoint under test.

    Returns:
        ``CurrentCopyCount``; ``0`` means scaled to zero.
    """
    return (await _runtime_config(endpoint))[2]


async def _settled_state(endpoint: SageMakerSandboxEndpoint) -> tuple[str, int]:
    """Wait until no runtime-config update is settling, and report the state.

    Application Auto Scaling drives the same ``DesiredCopyCount`` this fixture
    does, and SageMaker AI refuses a second write while the first is in
    progress ("Cannot update inference component ... while it is in state
    UPDATE_RC_IN_PROGRESS"), which surfaces as ``Updating``.

    Args:
        endpoint: The endpoint under test.

    Returns:
        Tuple of (status, ``DesiredCopyCount``) once the status is terminal, or
        the last one read when the wait ran out.
    """
    deadline = monotonic() + _SETTLE_TIMEOUT
    while True:
        status, desired, _ = await _runtime_config(endpoint)
        if status not in _UNSETTLED_STATUSES or monotonic() >= deadline:
            return status, desired
        await sleep(_SETTLE_POLL)


async def _set_copy_count(endpoint: SageMakerSandboxEndpoint, count: int) -> None:
    """Ask for *count* live copies of the inference component.

    The write waits for any update already settling, and stands down when that
    update is the one it wanted: asking for a count the component is already
    heading to would only be refused. Auto scaling can still start one in the
    moment between the read and the write, so that single refusal is waited out
    and retried; any other refusal is a real failure and is raised.

    Args:
        endpoint: The endpoint under test.
        count: Desired copy count.
    """
    async for client in _sagemaker_client(endpoint.region):
        for _ in range(_UPDATE_ATTEMPTS):
            if (await _settled_state(endpoint))[1] == count:
                return
            try:
                await client.update_inference_component_runtime_config(
                    InferenceComponentName=endpoint.inference_component,
                    DesiredRuntimeConfig={"CopyCount": count},
                )
            except ClientError as error:
                if _UPDATE_IN_PROGRESS not in str(error):
                    raise
                continue
            return


@pytest.fixture(scope="module")
async def sagemaker_model(sagemaker_sandbox_endpoint: SageMakerSandboxEndpoint) -> str:
    """Return the model ID the declared endpoint is published under.

    The lane runs against the server's real configuration rather than patching
    it: ``AWS_SAGEMAKER_ENDPOINTS`` belongs in ``tests/.env`` beside the
    ``TEST_SAGEMAKER_*`` keys, because settings are immutable after startup and
    the catalogue is built once for the whole session.

    Args:
        sagemaker_sandbox_endpoint: The endpoint under test, from ``tests/.env``.

    Returns:
        The published model ID clients ask for.
    """
    declared = [
        model_id
        for model_id, entry in SETTINGS.aws_sagemaker_endpoints.items()
        if entry.endpoint == sagemaker_sandbox_endpoint.endpoint
    ]
    if not declared:
        pytest.skip(
            "Declare the endpoint in AWS_SAGEMAKER_ENDPOINTS in tests/.env to serve it"
        )
    await initialize_bedrock_models()
    assert declared[0] in SAGEMAKER_ENDPOINT_MODELS, (
        f"{declared[0]} is declared but was not published in the catalogue"
    )
    return declared[0]


@pytest.fixture(scope="module")
async def cold_endpoint(
    sagemaker_sandbox_endpoint: SageMakerSandboxEndpoint, sagemaker_model: str
) -> AsyncIterator[SageMakerSandboxEndpoint]:
    """Take the endpoint's capacity away, and give it back cheaply afterwards.

    Scale-in is also the teardown: AWS holds the instance for a further quarter
    of an hour after the last request otherwise, which is several times the cost
    of the run itself.

    Args:
        sagemaker_sandbox_endpoint: The endpoint under test.
        sagemaker_model: The published model ID, ordered before this fixture so
            a misconfigured lane skips before it scales anything.

    Yields:
        The endpoint, at zero copies.
    """
    del sagemaker_model
    if not sagemaker_sandbox_endpoint.inference_component:
        pytest.skip(
            "The cold-start path needs a component-hosted endpoint "
            "(tests/.env sets no TEST_SAGEMAKER_INFERENCE_COMPONENT)"
        )
    await _set_copy_count(sagemaker_sandbox_endpoint, 0)
    deadline = monotonic() + _SCALE_IN_TIMEOUT
    while await _copy_count(sagemaker_sandbox_endpoint):
        if monotonic() >= deadline:
            pytest.skip("The inference component did not release its copies in time")
        await sleep(_SCALE_IN_POLL)
    try:
        yield sagemaker_sandbox_endpoint
    finally:
        await _set_copy_count(sagemaker_sandbox_endpoint, 0)


def test_endpoint_is_published_in_the_catalogue(
    local_test_client: TestClientType, sagemaker_model: str, api_key: str
) -> None:
    """The endpoint appears in ``/search_models`` as a text chat model.

    Its endpoint and inference component names must not appear anywhere in the
    response: they are the operator's own infrastructure, which the catalogue
    never publishes.

    Ref: stdapi/models/sagemaker_endpoints.py:_model_from_endpoint
    """
    response = local_test_client.get(
        "/search_models?route=/v1/chat/completions&output_modalities=TEXT",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 200
    entries = [m for m in response.json() if m["id"] == sagemaker_model]
    assert entries, f"{sagemaker_model} is missing from the catalogue"
    assert entries[0]["output_modalities"] == ["TEXT"]
    assert "/v1/chat/completions" in entries[0]["supported_routes"]
    assert "/v1/responses" in entries[0]["supported_routes"]
    assert "/anthropic/v1/messages" in entries[0]["supported_routes"]
    assert "/v1/responses/input_tokens" not in entries[0]["supported_routes"]
    assert SAGEMAKER_ENDPOINT_MODELS[sagemaker_model].service == SAGEMAKER_SERVICE
    endpoint = SAGEMAKER_ENDPOINT_MODELS[sagemaker_model].sagemaker_endpoint
    assert endpoint is not None
    assert endpoint.endpoint not in response.text


def test_chat_completion_reports_stop_and_real_usage(
    local_test_client: TestClientType, sagemaker_model: str, api_key: str
) -> None:
    """A non-streamed completion comes back with a finish reason and token counts.

    The endpoint names its own model, so the request body carries an empty
    ``model``: a mismatch here surfaces as an upstream 400 rather than as a
    wrong answer.

    Ref: stdapi/models/chat/_sagemaker.py:SageMakerChatModel._invoke_api
    """
    response = local_test_client.post(
        "/v1/chat/completions",
        json={
            "model": sagemaker_model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_completion_tokens": 64,
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["choices"][0]["message"]["content"]
    assert body["choices"][0]["finish_reason"] in ("stop", "length")
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["completion_tokens"] > 0


def test_max_tokens_reports_length(
    local_test_client: TestClientType, sagemaker_model: str, api_key: str
) -> None:
    """A truncated answer reports ``length``, not ``stop``.

    Ref: https://platform.openai.com/docs/api-reference/chat/object
    """
    response = local_test_client.post(
        "/v1/chat/completions",
        json={
            "model": sagemaker_model,
            "messages": [{"role": "user", "content": "Count from 1 to 200."}],
            "max_completion_tokens": 4,
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["finish_reason"] == "length"


def test_streamed_completion_has_the_openai_chunk_shape(
    local_test_client: TestClientType, sagemaker_model: str, api_key: str
) -> None:
    """The container's real SSE stream terminates and carries usage on request.

    ``stream_options.include_usage`` is the only way a Chat Completions stream
    reports what it consumed, and whether the container honours it is exactly
    the kind of fact a stub shaped like OpenAI cannot establish.

    Ref: https://platform.openai.com/docs/api-reference/chat/streaming
    """
    with local_test_client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": sagemaker_model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_completion_tokens": 64,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        headers={"Authorization": f"Bearer {api_key}"},
    ) as response:
        assert response.status_code == 200
        chunks = [line for line in response.iter_lines() if line.startswith("data: ")]

    assert len(chunks) > 1, "A stream must deliver more than its terminator"
    assert chunks[-1] == "data: [DONE]"
    payloads = [loads(chunk.removeprefix("data: ")) for chunk in chunks[:-1]]
    assert all(payload["object"] == "chat.completion.chunk" for payload in payloads)
    usage_chunks = [payload for payload in payloads if payload.get("usage")]
    assert usage_chunks, "include_usage asked for a usage chunk and got none"
    assert usage_chunks[-1]["usage"]["completion_tokens"] > 0


def test_usage_is_recorded_with_no_price_and_no_warning(
    local_test_client: TestClientType,
    sagemaker_model: str,
    api_key: str,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """Token quantities are recorded apart from Bedrock, and priced at nothing.

    AWS bills the endpoint by the instance-hour and publishes no per-token rate,
    so the record must carry the counts, resolve no cost, and raise no
    pricing-miss warning -- that signal is reserved for a real catalogue gap.

    Ref: stdapi/usage.py:UNPRICED_SERVICES
         stdapi/models/chat/_sagemaker.py:SageMakerChatModel._record_usage
    """
    capfd.readouterr()
    response = local_test_client.post(
        "/v1/chat/completions",
        json={
            "model": sagemaker_model,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_completion_tokens": 64,
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )
    assert response.status_code == 200
    captured = capfd.readouterr().out

    entries = logged_usage_entries(
        captured,
        service="sagemaker",
        operation="/v1/chat/completions",
        model=sagemaker_model,
    )
    assert entries, "Expected a SageMaker AI endpoint usage entry"
    assert entries[0]["input_tokens"] > 0
    assert entries[0]["output_tokens"] > 0
    assert "cost" not in entries[0]
    assert not logged_usage_entries(
        captured, service="bedrock-runtime", model=sagemaker_model
    )
    assert "No price found" not in captured


def test_model_pricing_reports_the_endpoint_service_and_no_rate(
    local_test_client: TestClientType, sagemaker_model: str, api_key: str
) -> None:
    """``/model_pricing`` names the serving endpoint and publishes no unit price.

    Ref: stdapi/routes/core_models.py:_preferred_service
    """
    response = local_test_client.get(
        f"/model_pricing?model={sagemaker_model}",
        headers={"Authorization": f"Bearer {api_key}"},
    )

    if response.status_code == 503:
        pytest.skip("Cost tracking is disabled on this server")
    assert response.status_code == 200
    card = response.json()[0]
    assert card["service"] == "sagemaker"
    assert card["prices"] == []


def test_container_error_json_becomes_our_envelope(
    local_test_client: TestClientType, sagemaker_model: str, api_key: str
) -> None:
    """The container's error body is not OpenAI's, and must not reach the client.

    The container answers its own error object and the endpoint quotes it whole
    inside a message of its own, adding the endpoint name, the account ID and a
    console URL; the caller must receive the envelope its SDK parses, carrying
    what the container said about the request and nothing the endpoint added.

    ``max_tokens`` above the served context window is the rejection to ask for:
    a value outside OpenAI's own range is not, since the container accepts
    ``temperature`` far beyond it and answers 200.

    Ref: stdapi/aws_sagemaker.py:_map_error
    """
    response = local_test_client.post(
        "/v1/chat/completions",
        json={
            "model": sagemaker_model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_completion_tokens": 10_000_000,
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 400, response.text
    error = response.json()["error"]
    assert "10000000" in error["message"], (
        "the gateway rejected this itself: the container never saw the request"
    )
    assert "sagemaker" not in response.text.lower()
    assert "amazonaws.com" not in response.text
    assert "Received client error" not in response.text
    assert 'File "' not in error["message"]
    assert not _ABSOLUTE_PATH_RE.search(error["message"])


def test_a_container_traceback_never_reaches_the_client(
    local_test_client: TestClientType, sagemaker_model: str, api_key: str
) -> None:
    """A rejection the container answers with a traceback is forwarded without it.

    ``tool_choice`` with no ``tools`` is the rejection to ask for: the container
    validates the body with pydantic and staples the frame that raised to the
    complaint, so the answer carries the runtime version and the library layout
    of a server nobody published. The caller must keep the complaint -- which
    parameter, and why -- and read none of the rest.

    Ref: stdapi/aws_sagemaker.py:_sanitize_container_message
    """
    response = local_test_client.post(
        "/v1/chat/completions",
        json={
            "model": sagemaker_model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_completion_tokens": 4,
            "tool_choice": {"type": "function", "function": {"name": "nope"}},
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 400, response.text
    message = response.json()["error"]["message"]
    assert "tool_choice" in message, (
        "the gateway rejected this itself: the container never saw the request"
    )
    assert 'File "' not in message
    assert "Traceback" not in message
    assert not _ABSOLUTE_PATH_RE.search(message), message


@pytest.mark.parametrize(
    ("route", "payload", "answer"),
    [
        (
            "/v1/responses",
            {"input": "Reply with OK.", "max_output_tokens": 64},
            "output",
        ),
        (
            "/anthropic/v1/messages",
            {
                "messages": [{"role": "user", "content": "Reply with OK."}],
                "max_tokens": 64,
            },
            "content",
        ),
    ],
)
def test_every_dialect_reaches_the_same_endpoint(
    local_test_client: TestClientType,
    sagemaker_model: str,
    api_key: str,
    route: str,
    payload: dict[str, Any],
    answer: str,
) -> None:
    """Responses and Anthropic Messages are served by converting to Chat Completions.

    The container serves one API, so the other two dialects ride the shared
    conversion; a regression there would otherwise surface only in production.

    Ref: stdapi/models/chat/_mantle/_convert.py:convert_payload
    """
    response = local_test_client.post(
        route,
        json={"model": sagemaker_model, **payload},
        headers={"Authorization": f"Bearer {api_key}"},
    )

    assert response.status_code == 200, response.text
    assert response.json()[answer]


async def test_concurrent_callers_absorb_one_cold_start(
    cold_endpoint: SageMakerSandboxEndpoint,
    sagemaker_model: str,
    local_test_client: TestClientType,
    api_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The single most valuable test here: a scaled-to-zero endpoint answers 200.

    An endpoint at zero copies rejects a request in about a second, and that
    rejection is what makes AWS provision an instance again. Every caller must
    see a slow success instead, and the concurrent ones must wait on the same
    warm-up rather than each starting one of their own -- which is why all three
    are fired at once, all three have to succeed, and only one probe may have
    been started for them.

    The callers are dispatched to threads rather than to an async SDK client:
    the transport's HTTP session belongs to the event loop of the lifespan that
    opened it, and reaching it from a second one is a different bug than the
    one under test.

    Ref: https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-auto-scaling-zero-instances.html
         stdapi/aws_sagemaker.py:_wait_for_capacity
    """
    assert await _copy_count(cold_endpoint) == 0, "the endpoint was not cold"
    live = peak = 0
    watch_warm_up = aws_sagemaker._watch_warm_up  # noqa: SLF001

    async def counted(
        region: RegionName, endpoint: str, inference_component: str, deadline: float
    ) -> bool:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        try:
            return await watch_warm_up(region, endpoint, inference_component, deadline)
        finally:
            live -= 1

    monkeypatch.setattr(aws_sagemaker, "_watch_warm_up", counted)
    started = monotonic()

    responses = await gather(
        *(
            to_thread(
                local_test_client.post,
                "/v1/chat/completions",
                json={
                    "model": sagemaker_model,
                    "messages": [{"role": "user", "content": "Reply with OK."}],
                    "max_completion_tokens": 64,
                },
                headers={"Authorization": f"Bearer {api_key}"},
            )
            for _ in range(_CONCURRENT_CALLERS)
        )
    )
    elapsed = monotonic() - started

    assert len(responses) == _CONCURRENT_CALLERS
    assert all(response.status_code == 200 for response in responses), [
        response.text for response in responses if response.status_code != 200
    ]
    assert all(
        response.json()["choices"][0]["message"]["content"] for response in responses
    )
    assert elapsed > _COLD_START_FLOOR, (
        "the endpoint answered too fast to have been cold: the test proved nothing"
    )
    assert peak == 1, f"{_CONCURRENT_CALLERS} callers ran {peak} warm-ups at once"
    assert await _copy_count(cold_endpoint) >= 1
