"""Home Assistant using the gateway as its Assist conversation agent.

``docs/use_cases_home_assistant.md`` tells a household to point Home Assistant's
**built-in Ollama integration** at the gateway's Ollama-compatible routes and to
select the result as the conversation agent of an Assist pipeline. Nothing else
in this repository boots Home Assistant: ``test_wyoming_audio.py`` covers the two
speech halves of that pipeline through a proxy, and the conversation half -- the
only half that reaches a model -- was untested until this module.

What makes this client different from every other one here is that **the gateway
is validated by a config flow before a single chat request is sent**. Home
Assistant refuses to create the integration at all unless ``GET /api/tags``
answers, through the real ``ollama`` client, in the shape it expects. So the
first thing proven here is not an answer but an *acceptance*, and the second is
its mirror image: the same flow against the same server with the API key left
out has to be refused, which is the ``401`` the documentation warns about.

The requests it then makes are its own, not this test's: ``stream=True`` NDJSON
on ``/ollama/api/chat``, ``keep_alive`` as the integer ``-1``, ``options.num_ctx``,
``think``, and -- whenever ``llm_hass_api`` is configured -- Assist's whole tool
schema in Ollama's ``{"type": "function", "function": {...}}`` spelling.

Five traps, all paid for once:

- **the image's entry point is an s6 overlay.** It wants a writable root and root
  privileges, which the service sandbox denies, so Home Assistant is started
  directly with ``entrypoint="python3"`` and ``-m homeassistant`` -- the same
  command the overlay's own service script runs.
- **``default_config:`` is not used.** Home Assistant writes it into a config
  directory it finds empty, and it pulls in cloud, discovery and hardware
  integrations this sandbox has nothing for. A seeded ``configuration.yaml``
  naming only ``http`` and ``conversation`` is what keeps the boot to one job.
- **a cold start takes minutes.** The container health probe answers on
  ``/manifest.json`` long before the REST API is usable, so it is followed by a
  protocol-aware probe on ``GET /api/onboarding``, as ``test_wyoming_audio.py``
  follows its TCP probe with a Wyoming ``describe``.
- **the config flow gives the server five seconds.** ``ollama.const.DEFAULT_TIMEOUT``
  bounds both the flow's ``list()`` and the entry's own setup, and the gateway
  builds its Ollama catalogue on the first call, so the tags route is warmed from
  the host first.
- **a second flow for the same URL aborts before it validates anything.**
  ``_async_abort_entries_match`` runs ahead of the connection check, so the
  no-API-key case is submitted with a trailing slash: ``ollama._parse_host``
  strips it, leaving the request identical and the stored URL different.

Requires ``--agentic``, podman, and Bedrock credentials.

Ref: https://www.home-assistant.io/integrations/ollama
     https://developers.home-assistant.io/docs/auth_api/
     https://developers.home-assistant.io/docs/intent_conversation_api/
     https://github.com/home-assistant/core/blob/2026.7.4/homeassistant/components/ollama/config_flow.py
     https://github.com/home-assistant/core/blob/2026.7.4/homeassistant/components/conversation/http.py
     docs/use_cases_home_assistant.md
     stdapi/routes/ollama_chat.py:chat
     stdapi/routes/ollama_models.py
     tests/agentic/_podman.py:start_service_container
"""

from __future__ import annotations

from secrets import token_hex
from time import monotonic, sleep
from typing import TYPE_CHECKING, Any

import httpx
import pytest

from stdapi.config import SETTINGS

from ._podman import start_service_container, stop_service_container
from ._server import find_free_port

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from ._podman import ServiceContainer
    from ._server import AgenticServer

pytestmark = pytest.mark.agentic

#: Image driven here, pinned to the release the deployment sample pins.
#:
#: Unlike the moving tags elsewhere in the lane, this one is fixed: the config
#: flow, the subentry schema and the REST shapes asserted below are read from
#: this exact tag of home-assistant/core, and Home Assistant renames flow steps
#: between monthly releases.
_IMAGE = "ghcr.io/home-assistant/home-assistant:2026.7.4"

#: Executable replacing the image's own entry point.
#:
#: That entry point is ``/init``, an s6 overlay that needs a writable root and
#: root privileges -- both denied by the service sandbox. This is the command its
#: own service script ends up running.
_ENTRYPOINT = "python3"

#: Configuration directory, under the only writable mount.
_CONFIG_DIR = "/work/config"

#: Arguments to :data:`_ENTRYPOINT`.
#:
#: ``--skip-pip`` keeps the boot off PyPI: the official image already installs
#: every integration requirement, so the only thing a resolver could do here is
#: stall on a network this container has no business using.
_ARGV = ("-m", "homeassistant", "--config", _CONFIG_DIR, "--skip-pip")

#: Seconds allowed for the container to answer ``/manifest.json`` after launch.
_STARTUP_TIMEOUT = 600

#: Seconds allowed, after that, for the REST API to become usable.
_READY_TIMEOUT = 600.0

#: Seconds between two readiness probes of a booting Home Assistant.
_READY_POLL_INTERVAL = 5.0

#: Seconds one request to Home Assistant may take; a Bedrock call runs behind each.
_REQUEST_TIMEOUT = 600.0

#: Seconds allowed for a subentry's conversation entity to reach the state machine.
_ENTITY_TIMEOUT = 180.0

#: Seconds between two polls of the state machine.
_ENTITY_POLL_INTERVAL = 2.0

#: Model the conversation agents run on.
#:
#: The same Claude the Ollama-dialect reference client drives its tool round trip
#: on (``tests/agentic/test_ollama_sdk.py``): Assist attaches a dozen tools to
#: every turn once ``llm_hass_api`` is set, which is more than the cheapest chat
#: models in this lane answer reliably.
_MODEL = "anthropic.claude-haiku-4-5-20251001-v1:0"

#: Assist's own LLM API, the value ``llm_hass_api`` takes.
_ASSIST_API = "assist"

#: Tool Assist appends to every tool list, whatever is exposed to it.
#:
#: The others are filtered against the exposed entity domains; this one is not,
#: which makes it the only name that can be asserted on unconditionally.
_UNCONDITIONAL_TOOL = "GetDateTime"

#: Title of the conversation agent configured without Assist's tools.
_PLAIN_AGENT_TITLE = "Gateway Plain"

#: Title of the conversation agent configured with them.
_ASSIST_AGENT_TITLE = "Gateway Assist"

#: Question whose answer is one proper noun no model has to be prompted for.
_PLAIN_QUESTION = "What is the capital of France? Answer with the city name only."

#: Word that answer has to contain.
_PLAIN_ANSWER = "paris"

#: Question that can only be answered by calling Assist's date and time tool.
_TOOL_QUESTION = "What is the current date and time? Use your tools to find out."

#: Name of the owner account onboarding creates; only ever reachable on loopback.
_OWNER_USERNAME = "stdapi-agentic"

#: Display name of that account.
_OWNER_NAME = "stdapi.ai agentic lane"

#: Language every request declares, so intent matching never guesses.
_LANGUAGE = "en"

#: Path of the Ollama chat route on the gateway, as the request log spells it.
_CHAT_PATH = f"{SETTINGS.ollama_routes_prefix}/api/chat"

#: Path of the Ollama model listing route, likewise.
_TAGS_PATH = f"{SETTINGS.ollama_routes_prefix}/api/tags"

#: Fields the conversation subentry form has to offer.
#:
#: ``model`` is what the doc's second dialog asks a user to pick and ``llm_hass_api`` is
#: what turns the agent into a device controller, so a release that renamed
#: either would leave the documentation wrong.
_EXPECTED_SUBENTRY_FIELDS = frozenset({"name", "model", "prompt", "llm_hass_api"})

#: Contents seeded into the configuration directory before the first boot.
#:
#: Home Assistant writes a ``default_config:`` file of its own into an empty
#: directory, which sets up cloud, discovery, hardware and recorder integrations
#: that have nothing to do with this test and cost minutes of boot. ``http``
#: carries the port, and ``conversation`` brings both ``POST
#: /api/conversation/process`` and Assist's LLM API up before the agent exists.
_CONFIGURATION_YAML = """\
# Written by tests/agentic/test_home_assistant.py -- not a sample configuration.
http:
  server_port: {port}
conversation:
"""


def _gateway_url(server: AgenticServer) -> str:
    """Return the gateway's base URL as seen from inside the container.

    Args:
        server: Gateway under test.

    Returns:
        The loopback URL pasta forwards, or the external deployment's own URL
        when ``--server-url`` selected one.
    """
    if server.forward_port is None:
        return server.base_url
    return f"http://127.0.0.1:{server.forward_port}"


def _ollama_url(server: AgenticServer) -> str:
    """Return the URL the documentation tells a user to type into the config flow.

    The gateway's base URL plus its Ollama routes prefix, and nothing more: the
    ``ollama`` client keeps that path and httpx merges ``api/tags`` and
    ``api/chat`` onto it, which is the whole reason the prefix has to survive the
    round trip.

    Args:
        server: Gateway under test.

    Returns:
        The URL to submit as the flow's ``url`` field.
    """
    return f"{_gateway_url(server)}{SETTINGS.ollama_routes_prefix}"


def _environment() -> Mapping[str, str]:
    """Return the container environment.

    Home Assistant is configured by its ``configuration.yaml`` and by its own
    APIs, so nothing here carries a setting under test -- and no secret: the
    gateway's API key travels in the config flow, over loopback, never in an
    environment a container image could print.

    Returns:
        The environment to start the container with.
    """
    return {
        # The root filesystem is read-only: everything Home Assistant writes has
        # to be under /work.
        "HOME": "/work/home",
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def _seed_configuration(workdir: Path, port: int) -> None:
    """Write the configuration file Home Assistant boots from.

    Args:
        workdir: Host directory mounted at ``/work``.
        port: Port Home Assistant must listen on, published on the same host port.
    """
    config = workdir / "config"
    config.mkdir(parents=True, exist_ok=True)
    (config / "configuration.yaml").write_text(
        _CONFIGURATION_YAML.format(port=port), "utf-8"
    )


def _await_ready(service: ServiceContainer) -> None:
    """Wait until Home Assistant's REST API answers.

    The container's own health poll cannot tell: ``/manifest.json`` is served by
    the frontend integration, and any answer below 500 -- including the 404 the
    HTTP layer returns while the rest of the bootstrap is still running --
    satisfies it. ``GET /api/onboarding`` is the meaningful probe, because
    ``onboarding`` is set up behind ``frontend``, which is behind ``http``: an
    answered list proves the whole chain the config-entries API sits on.

    Args:
        service: Container started for this module.

    Raises:
        AssertionError: If the API never answers within the deadline.
    """
    deadline = monotonic() + _READY_TIMEOUT
    last = "no answer"
    while monotonic() < deadline:
        try:
            response = httpx.get(
                f"{service.base_url}/api/onboarding", timeout=_READY_POLL_INTERVAL
            )
            if response.status_code == 200 and isinstance(response.json(), list):
                return
            last = f"HTTP {response.status_code}"
        except (httpx.HTTPError, ValueError) as error:  # ValueError: not JSON yet.
            last = str(error)
        sleep(_READY_POLL_INTERVAL)
    pytest.fail(
        f"Home Assistant did not serve its REST API on port {service.port} within "
        f"{_READY_TIMEOUT}s ({last}).\nLast output:\n{service.logs()[-3000:]}"
    )


def _authenticate(client: httpx.Client, base_url: str) -> str:
    """Onboard Home Assistant and return an owner bearer token.

    The sequence the frontend's own onboarding wizard posts, minus the browser:
    ``POST /api/onboarding/users`` is unauthenticated and answers with an
    authorization code, ``POST /auth/token`` trades it for a bearer token, and
    the three remaining steps are marked done with that token -- Home Assistant
    keeps redirecting to the wizard while any of them is outstanding.

    ``client_id`` is Home Assistant's own base URL and ``redirect_uri`` a path on
    it, so indieauth's same-origin rule is satisfied without it fetching the
    client identifier and parsing a document that does not exist. A loopback
    address is the one IP literal it accepts as a client identifier.

    Args:
        client: Client bound to Home Assistant's base URL.
        base_url: That base URL, needed verbatim for the indieauth fields.

    Returns:
        The owner's access token.
    """
    client_id = f"{base_url}/"
    created = client.post(
        "/api/onboarding/users",
        json={
            "name": _OWNER_NAME,
            "username": _OWNER_USERNAME,
            # Generated per run and never leaves this process: the account is
            # only reachable on the host loopback the container publishes to.
            "password": token_hex(16),
            "client_id": client_id,
            "language": _LANGUAGE,
        },
    )
    assert created.status_code == 200, f"onboarding failed: {created.text[:500]}"

    exchanged = client.post(
        "/auth/token",
        # The one form-encoded request in the whole sequence.
        data={
            "client_id": client_id,
            "grant_type": "authorization_code",
            "code": created.json()["auth_code"],
        },
    )
    assert exchanged.status_code == 200, (
        f"token exchange failed: {exchanged.text[:500]}"
    )
    token = exchanged.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    for step, payload in (
        ("core_config", None),
        ("analytics", None),
        (
            "integration",
            {"client_id": client_id, "redirect_uri": f"{base_url}/agentic-callback"},
        ),
    ):
        finished = client.post(f"/api/onboarding/{step}", json=payload, headers=headers)
        assert finished.status_code == 200, (
            f"onboarding step {step} failed: {finished.text[:500]}"
        )
    return str(token)


def _warm_catalogue(server: AgenticServer) -> None:
    """Make the gateway build its Ollama catalogue before Home Assistant asks for it.

    Both the config flow and the entry's own setup wrap ``list()`` in
    ``asyncio.timeout(ollama.const.DEFAULT_TIMEOUT)`` -- five seconds -- and the
    gateway builds and caches its ``/api/tags`` response on the first call. A
    cold catalogue would therefore surface as ``cannot_connect``, blaming the
    connection for a latency the client simply refused to wait for.

    Args:
        server: Gateway under test.
    """
    response = httpx.get(
        f"{server.base_url}{_TAGS_PATH}",
        headers={"Authorization": f"Bearer {server.api_key}"},
        timeout=_REQUEST_TIMEOUT,
    )
    assert response.status_code == 200, response.text[:500]
    served = {model["model"] for model in response.json()["models"]}
    assert _MODEL in served, (
        f"the gateway does not serve {_MODEL} on {_TAGS_PATH}; Home Assistant "
        "would treat it as a model to download"
    )


def _start_flow(client: httpx.Client, path: str, handler: object) -> dict[str, Any]:
    """Start a config-entry or subentry flow and return its first step.

    Args:
        client: Authenticated Home Assistant client.
        path: Flow index route, config-entry or subentry.
        handler: ``"ollama"`` for a config entry, ``[entry_id, type]`` for a
            subentry -- the two-element list Home Assistant coerces to a tuple.

    Returns:
        The decoded flow result.
    """
    response = client.post(path, json={"handler": handler})
    assert response.status_code == 200, f"{path} refused: {response.text[:500]}"
    result: dict[str, Any] = response.json()
    return result


def _advance_flow(
    client: httpx.Client, path: str, flow_id: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Submit one step of a running flow and return its result.

    Args:
        client: Authenticated Home Assistant client.
        path: Flow index route the flow was started on.
        flow_id: Identifier the first step returned.
        payload: Values for the step's schema.

    Returns:
        The decoded flow result.
    """
    response = client.post(f"{path}/{flow_id}", json=dict(payload))
    assert response.status_code == 200, f"{path} step refused: {response.text[:500]}"
    result: dict[str, Any] = response.json()
    return result


def _schema_fields(result: Mapping[str, Any]) -> set[str]:
    """Return the field names a form step advertises.

    Args:
        result: A flow result of type ``form``.

    Returns:
        The names in its ``data_schema``.
    """
    return {str(field["name"]) for field in result.get("data_schema") or ()}


def _add_conversation_agent(
    client: httpx.Client, entry_id: str, title: str, *, assist: bool
) -> dict[str, Any]:
    """Create one conversation subentry and return the flow's final result.

    Every optional field is sent explicitly rather than left to the form's
    suggested values, so the request Home Assistant then puts on the wire --
    ``options.num_ctx``, ``keep_alive``, ``think`` -- is the same on every run.

    Args:
        client: Authenticated Home Assistant client.
        entry_id: Parent Ollama config entry.
        title: Name the subentry, its device and its entity take.
        assist: Whether to hand the agent Assist's LLM API, which is what makes
            Home Assistant attach its tool schema to every turn.

    Returns:
        The decoded flow result, expected to be ``create_entry``.
    """
    path = "/api/config/config_entries/subentries/flow"
    form = _start_flow(client, path, [entry_id, "conversation"])
    assert form["type"] == "form", form
    assert form["step_id"] == "set_options", form
    assert _schema_fields(form) >= _EXPECTED_SUBENTRY_FIELDS, (
        f"the conversation subentry form offers {sorted(_schema_fields(form))}"
    )

    payload: dict[str, Any] = {
        "name": title,
        "model": _MODEL,
        "num_ctx": 8192,
        "max_history": 20,
        "keep_alive": -1,
        "think": False,
    }
    if assist:
        payload["llm_hass_api"] = [_ASSIST_API]
    result = _advance_flow(client, path, form["flow_id"], payload)
    assert result["type"] == "create_entry", (
        f"the conversation subentry was not created: {result}. A "
        "'show_progress' step here means the gateway did not list "
        f"{_MODEL} on {_TAGS_PATH} and Home Assistant tried to download it"
    )
    return result


def _conversation_entity(client: httpx.Client, title: str) -> str:
    """Return the entity id the subentry named *title* registered.

    The subentry flow answers before its platform has finished loading, and its
    own result carries no identifier -- Home Assistant strips ``result`` from a
    subentry ``create_entry`` -- so the state machine is polled instead. The
    entity takes the subentry's own title as its friendly name, because the
    Ollama entity declares ``_attr_has_entity_name`` with no name of its own.

    Args:
        client: Authenticated Home Assistant client.
        title: Name the subentry was created with.

    Returns:
        The ``conversation.*`` entity id.

    Raises:
        AssertionError: If no such entity appears within the deadline.
    """
    deadline = monotonic() + _ENTITY_TIMEOUT
    seen: list[str] = []
    while monotonic() < deadline:
        response = client.get("/api/states")
        assert response.status_code == 200, response.text[:500]
        agents = [
            state
            for state in response.json()
            if str(state["entity_id"]).startswith("conversation.")
        ]
        seen = [str(state["entity_id"]) for state in agents]
        for state in agents:
            if state["attributes"].get("friendly_name") == title:
                return str(state["entity_id"])
        sleep(_ENTITY_POLL_INTERVAL)
    pytest.fail(
        f"no conversation entity named {title!r} appeared within "
        f"{_ENTITY_TIMEOUT}s; conversation entities present: {seen}"
    )


def _process(client: httpx.Client, agent_id: str, text: str) -> dict[str, Any]:
    """Run one Assist conversation turn and return its intent response.

    Args:
        client: Authenticated Home Assistant client.
        agent_id: Conversation entity to route the turn to.
        text: What the user said.

    Returns:
        The ``response`` object of the reply.
    """
    reply = client.post(
        "/api/conversation/process",
        json={"text": text, "agent_id": agent_id, "language": _LANGUAGE},
    )
    assert reply.status_code == 200, f"the turn was refused: {reply.text[:500]}"
    response: dict[str, Any] = reply.json()["response"]
    return response


def _spoken(response: Mapping[str, Any]) -> str:
    """Return the sentence Assist would hand to text to speech.

    Args:
        response: Intent response of a turn.

    Returns:
        The plain spoken text, empty when the agent produced none.
    """
    speech = response.get("speech") or {}
    plain = speech.get("plain") or {}
    return str(plain.get("speech") or "")


def _assert_answered(response: Mapping[str, Any]) -> str:
    """Assert a turn produced a real spoken answer, and return it.

    Home Assistant reports an agent failure as a 200 carrying an ``error``
    response type, so the status code alone proves nothing.

    Args:
        response: Intent response of a turn.

    Returns:
        The spoken text.
    """
    spoken = _spoken(response)
    assert response.get("response_type") != "error", (
        f"the conversation agent failed: {spoken or response}"
    )
    assert spoken.strip(), f"the agent spoke nothing: {response}"
    return spoken


def _chat_requests(server: AgenticServer, log_start: int) -> list[Mapping[str, object]]:
    """Return the Ollama chat requests the gateway logged since *log_start*.

    Args:
        server: Gateway Home Assistant was pointed at.
        log_start: Log index captured before the turn.

    Returns:
        One entry per request, in order. Empty when the gateway's log is not
        observable, which is the case for the deployment ``--server-url``
        selects, so a caller may treat an empty list as "skip the log assertion".
    """
    if server.process is None:
        return []
    return [
        entry
        for entry in server.log_entries(log_start)
        if entry.get("type") == "request" and entry.get("path") == _CHAT_PATH
    ]


def _request_params(entry: Mapping[str, object]) -> dict[str, Any]:
    """Return the request body one logged entry recorded.

    Args:
        entry: A ``request`` log entry.

    Returns:
        The decoded body, empty when the gateway logged none.
    """
    params = entry.get("request_params")
    return params if isinstance(params, dict) else {}


def _tool_names(entry: Mapping[str, object]) -> list[str]:
    """Return the tool names one logged chat request carried.

    Args:
        entry: A ``request`` log entry for the chat route.

    Returns:
        The declared function names, empty when the request declared no tool.

    Raises:
        AssertionError: If a declared tool is not in Ollama's tool shape.
    """
    names = []
    for tool in _request_params(entry).get("tools") or ():
        assert tool.get("type") == "function", f"unexpected tool shape: {tool}"
        function = tool.get("function") or {}
        assert isinstance(function.get("parameters"), dict), (
            f"tool {function.get('name')!r} carried no JSON Schema: {tool}"
        )
        names.append(str(function["name"]))
    return names


def _assert_model(server: AgenticServer, entries: list[Mapping[str, object]]) -> None:
    """Assert every logged chat request resolved the model the subentry names.

    Home Assistant is not a registered CLI, so the lane's autouse identity check
    has no tool to attribute requests to; without this, an agent silently falling
    back to a default model would still answer and still pass.

    Args:
        server: Gateway Home Assistant was pointed at.
        entries: Chat requests collected for the turn.

    Ref: stdapi/monitoring.py:EventLog
    """
    if server.process is None:
        return  # External server: its log is not observable here.
    assert entries, f"no request reached {_CHAT_PATH}"
    resolved = {str(entry.get("model_id") or "") for entry in entries}
    assert resolved == {_MODEL}, f"{_CHAT_PATH} resolved {sorted(resolved)}"


@pytest.fixture(scope="module")
def home_assistant(
    agentic_server: AgenticServer, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[httpx.Client]:
    """One onboarded Home Assistant, authenticated as its owner.

    Module-scoped because a boot costs minutes and because Home Assistant takes
    an exclusive lock on ``<config>/.ha_run.lock``: a second instance against the
    same directory reports "Another Home Assistant instance is already running!"
    and exits. Each module run therefore gets its own configuration directory,
    onboarded from scratch.

    The container runs as the owner of the working directory. Its image declares
    no ``USER``, and container root under ``--userns=keep-id`` is a subordinate
    host UID that cannot write into that directory.

    Yields:
        A client bound to Home Assistant, carrying the owner's bearer token.
    """
    workdir = tmp_path_factory.mktemp("home-assistant")
    port = find_free_port()
    _seed_configuration(workdir, port)
    container = start_service_container(
        image=_IMAGE,
        port=port,
        workdir=workdir,
        env=_environment(),
        forward_port=agentic_server.forward_port,
        argv=_ARGV,
        entrypoint=_ENTRYPOINT,
        data_dirs=("config", "home"),
        health_path="/manifest.json",
        startup_timeout=_STARTUP_TIMEOUT,
        user=f"{workdir.stat().st_uid}:{workdir.stat().st_gid}",
        # No refresh, unlike the moving tags elsewhere: this one is a release
        # tag, so --agentic-rebuild would re-pull a gigabyte to get the same
        # layers back.
    )
    try:
        _await_ready(container)
        with httpx.Client(
            base_url=container.base_url, timeout=_REQUEST_TIMEOUT
        ) as client:
            token = _authenticate(client, container.base_url)
            client.headers["Authorization"] = f"Bearer {token}"
            yield client
    finally:
        stop_service_container(container)


@pytest.fixture(scope="module")
def ollama_entry(
    home_assistant: httpx.Client, agentic_server: AgenticServer
) -> dict[str, Any]:
    """The Ollama config entry, created exactly as the documentation says.

    Module-scoped: the flow validates the gateway once, and every conversation
    agent below is a subentry of the entry it produces.

    Returns:
        The flow's final result, expected to be ``create_entry``.
    """
    _warm_catalogue(agentic_server)
    path = "/api/config/config_entries/flow"
    form = _start_flow(home_assistant, path, "ollama")
    assert form["type"] == "form", form
    assert form["step_id"] == "user", form
    return _advance_flow(
        home_assistant,
        path,
        form["flow_id"],
        {"url": _ollama_url(agentic_server), "api_key": agentic_server.api_key},
    )


@pytest.fixture(scope="module")
def plain_agent(home_assistant: httpx.Client, ollama_entry: dict[str, Any]) -> str:
    """A conversation agent with no Assist API, the control for the tool tests.

    Returns:
        Its ``conversation.*`` entity id.
    """
    _add_conversation_agent(
        home_assistant,
        ollama_entry["result"]["entry_id"],
        _PLAIN_AGENT_TITLE,
        assist=False,
    )
    return _conversation_entity(home_assistant, _PLAIN_AGENT_TITLE)


@pytest.fixture(scope="module")
def assist_agent(home_assistant: httpx.Client, ollama_entry: dict[str, Any]) -> str:
    """A conversation agent holding Assist's LLM API, so every turn carries tools.

    Returns:
        Its ``conversation.*`` entity id.
    """
    _add_conversation_agent(
        home_assistant,
        ollama_entry["result"]["entry_id"],
        _ASSIST_AGENT_TITLE,
        assist=True,
    )
    return _conversation_entity(home_assistant, _ASSIST_AGENT_TITLE)


class TestHomeAssistantConfigFlow:
    """Both dialogs of the documentation's Conversation Agent section.

    Ref: https://www.home-assistant.io/integrations/ollama
         docs/use_cases_home_assistant.md
         stdapi/routes/ollama_models.py
    """

    def test_the_flow_accepts_the_gateway_as_an_ollama_server(
        self, ollama_entry: dict[str, Any], agentic_server: AgenticServer
    ) -> None:
        """Submitting the gateway's URL and key reaches ``create_entry``.

        Home Assistant validates the ``user`` step by calling ``GET /api/tags``
        through the real ``ollama`` client, so an entry it agreed to create is
        proof of three separate things at once: the routes prefix survived the
        client's own host parsing, the key was accepted as
        ``Authorization: Bearer``, and the listing deserialised into the shape
        the client's types demand. None of them is observable from the gateway
        side alone.

        Ref: https://github.com/home-assistant/core/blob/2026.7.4/homeassistant/components/ollama/config_flow.py
        """
        assert ollama_entry["type"] == "create_entry", (
            f"Home Assistant refused the gateway: {ollama_entry}"
        )
        entry = ollama_entry["result"]
        assert entry["domain"] == "ollama", entry
        assert entry["entry_id"], entry
        assert entry["title"] == _ollama_url(agentic_server), entry

    def test_the_flow_is_refused_when_the_api_key_is_omitted(
        self,
        home_assistant: httpx.Client,
        agentic_server: AgenticServer,
        ollama_entry: dict[str, Any],
    ) -> None:
        """The same flow with no ``api_key`` fails on the gateway's ``401``.

        The documentation warns that a local Ollama needs no credentials and that
        skipping the field here refuses every request; this is that warning,
        checked. The gateway answers ``401``, the ``ollama`` client raises a
        ``ResponseError`` carrying the status, and the flow re-shows its own form
        with ``invalid_auth`` -- which is a *silent* outcome for anyone who only
        looks at the HTTP status, since the flow route itself answers 200.

        The URL carries a trailing slash the successful entry does not: the flow
        matches existing entries on the literal string and would otherwise abort
        as ``already_configured`` before validating anything, while
        ``ollama._parse_host`` strips it so the request is byte-identical. The
        accepted entry is therefore a dependency rather than an accident of test
        order, and its survival is the last assertion here.

        Ref: https://github.com/home-assistant/core/blob/2026.7.4/homeassistant/components/ollama/config_flow.py
             stdapi/auth.py:authenticate
        """
        path = "/api/config/config_entries/flow"
        form = _start_flow(home_assistant, path, "ollama")
        result = _advance_flow(
            home_assistant,
            path,
            form["flow_id"],
            {"url": f"{_ollama_url(agentic_server)}/"},
        )
        assert result["type"] == "form", (
            f"an unauthenticated flow was not refused: {result}"
        )
        assert result["step_id"] == "user", result
        assert result.get("errors") == {"base": "invalid_auth"}, result

        listed = home_assistant.get(
            "/api/config/config_entries/entry", params={"domain": "ollama"}
        )
        assert listed.status_code == 200, listed.text[:500]
        assert [entry["entry_id"] for entry in listed.json()] == [
            ollama_entry["result"]["entry_id"]
        ], f"the refused flow left an entry behind: {listed.json()}"


class TestHomeAssistantConversationAgent:
    """Step 4 and the pipeline selection that follows it.

    Ref: https://developers.home-assistant.io/docs/intent_conversation_api/
         docs/use_cases_home_assistant.md
         stdapi/routes/ollama_chat.py:chat
    """

    def test_the_subentry_registers_a_conversation_entity(
        self, plain_agent: str, home_assistant: httpx.Client
    ) -> None:
        """A conversation subentry becomes an entity an Assist pipeline can select.

        The agent is a *subentry* of the Ollama entry, not the entry itself, and
        only the entity it registers can be named as a pipeline's
        ``conversation_engine`` -- so this is what "select the integration as the
        conversation agent" actually resolves to.

        Ref: https://github.com/home-assistant/core/blob/2026.7.4/homeassistant/components/ollama/conversation.py
        """
        assert plain_agent.startswith("conversation."), plain_agent
        state = home_assistant.get(f"/api/states/{plain_agent}")
        assert state.status_code == 200, state.text[:500]
        assert state.json()["attributes"]["friendly_name"] == _PLAIN_AGENT_TITLE

    def test_a_turn_is_answered_through_the_gateway(
        self,
        plain_agent: str,
        home_assistant: httpx.Client,
        agentic_server: AgenticServer,
    ) -> None:
        """User text comes back as a spoken answer generated by the model.

        The whole conversation stage in one call: Home Assistant builds the
        prompt, streams ``/ollama/api/chat`` as NDJSON, reassembles the deltas
        into an intent response, and hands back the sentence Assist would speak.
        Without ``llm_hass_api`` the request declares no tool at all, which is
        the control for the tool assertions below.

        Ref: https://developers.home-assistant.io/docs/intent_conversation_api/
        """
        log_start = len(agentic_server.logs)
        spoken = _assert_answered(
            _process(home_assistant, plain_agent, _PLAIN_QUESTION)
        )
        assert _PLAIN_ANSWER in spoken.lower(), spoken[:500]

        entries = _chat_requests(agentic_server, log_start)
        _assert_model(agentic_server, entries)
        assert all(not _tool_names(entry) for entry in entries), (
            "an agent configured without llm_hass_api still declared tools"
        )


class TestHomeAssistantAssistTools:
    """The tool schema Assist attaches once ``llm_hass_api`` is configured.

    The documentation ends by telling a household to "choose a model that
    supports tool calling if you want the agent to control devices". What that
    costs the gateway is Home Assistant's own tool payload -- a dozen intent
    handlers converted from voluptuous to JSON Schema -- on every turn.

    No device action is asserted, and deliberately: the sandbox exposes no
    controllable entity, and seeding one would mean carrying a template light
    platform in ``configuration.yaml`` whose schema moves between Home Assistant
    releases for no gain here. What is proven instead is everything up to the
    device: the schema reaches the model in Ollama's tool shape, and a tool the
    model chose is executed and its result replayed back through the gateway.

    Ref: https://github.com/home-assistant/core/blob/2026.7.4/homeassistant/components/ollama/entity.py
         https://github.com/home-assistant/core/blob/2026.7.4/homeassistant/helpers/llm.py
    """

    def test_the_assist_tool_schema_reaches_the_model(
        self,
        assist_agent: str,
        home_assistant: httpx.Client,
        agentic_server: AgenticServer,
    ) -> None:
        """A turn on the Assist agent carries Home Assistant's tools and is answered.

        Every tool is asserted structurally rather than by name, because Assist
        filters most of them against the entity domains exposed to it; only
        ``GetDateTime`` is appended unconditionally, which makes it the one name
        that means the same thing on an empty installation as on a full one.

        Ref: https://github.com/home-assistant/core/blob/2026.7.4/homeassistant/helpers/llm.py
        """
        log_start = len(agentic_server.logs)
        _assert_answered(_process(home_assistant, assist_agent, _TOOL_QUESTION))

        entries = _chat_requests(agentic_server, log_start)
        _assert_model(agentic_server, entries)
        if not entries:
            return  # External server: no log to inspect.
        declared = _tool_names(entries[0])
        assert declared, "the Assist agent declared no tool to the gateway"
        assert _UNCONDITIONAL_TOOL in declared, declared

    @pytest.mark.retry(
        "whether the model calls GetDateTime rather than answering from its own "
        "notion of the date is a model decision, not a gateway behaviour"
    )
    def test_a_tool_result_is_replayed_back_through_the_gateway(
        self,
        assist_agent: str,
        home_assistant: httpx.Client,
        agentic_server: AgenticServer,
    ) -> None:
        """The agent runs a full tool loop: call, execute, replay, answer.

        Home Assistant re-sends the whole history on each iteration, so a second
        request whose messages carry a ``tool`` role is the proof that the
        gateway parsed a tool call out of its NDJSON stream, that Home Assistant
        executed it, and that the gateway accepted the result on the way back.
        A single request would mean the model answered without ever calling.

        Ref: https://github.com/home-assistant/core/blob/2026.7.4/homeassistant/components/ollama/entity.py
        """
        log_start = len(agentic_server.logs)
        _assert_answered(_process(home_assistant, assist_agent, _TOOL_QUESTION))

        entries = _chat_requests(agentic_server, log_start)
        _assert_model(agentic_server, entries)
        if not entries:
            return  # External server: no log to inspect.
        assert len(entries) > 1, (
            f"the Assist agent never called a tool: {len(entries)} chat request(s)"
        )
        roles = [
            message.get("role")
            for entry in entries
            for message in _request_params(entry).get("messages") or ()
        ]
        assert "tool" in roles, f"no tool result was replayed; roles seen: {roles}"
