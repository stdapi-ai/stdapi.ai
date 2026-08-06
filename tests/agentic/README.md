# Agentic test lane

Real third-party clients — Claude Code, Codex, pi, OpenClaw, Hermes, Qwen Code,
n8n, Haystack, Open WebUI, wyoming-openai, LangChain and pydantic-ai —
driven end to end against a live stdapi.ai server. They are the only tests that
exercise a client's own wire behaviour: its system prompt, tool definitions,
multi-turn tool-call replays, SSE streaming and multipart uploads, all translated
by the gateway for a model that is usually not the vendor's own.

| Client | Gateway route | Why it is here |
|---|---|---|
| Claude Code | `/anthropic` | Anthropic Messages, with a large tool set and session attribution |
| Codex | `/v1/responses` | Responses, with a ~7600-token `instructions` field |
| pi | all three | The same run over Chat Completions, Responses **and** Anthropic Messages |
| OpenClaw | all three | The wire format as a single flag: `--custom-compatibility openai \| openai-responses \| anthropic` |
| Hermes | all three | Three `transport` values, and the only client choosing its own Anthropic cache TTL tier |
| Qwen Code | `/v1/chat/completions` | The only agent binary replaying `reasoning_content` back, and the only client sending the `reasoning` object |
| n8n | twelve routes | All three chat dialects, embeddings, audio ×3, files, images ×2, videos, legacy completions |
| Haystack | `/cohere/v2/rerank` | The only client that reranks, chained onto embeddings and chat |
| Open WebUI | chat, audio ×2, images, embeddings, rerank | The documented integration's own environment block, as a service |
| wyoming-openai | audio ×2 | The only client streaming `/v1/audio/speech` and the only one calling it concurrently |
| langchain-openai / langchain-anthropic | `/v1` chat + embeddings, `/anthropic` | Streaming, `bind_tools`, `with_structured_output`, and the embeddings token-array trap, in plain Python |
| pydantic-ai | `/v1/chat/completions` | Proves a real multi-turn tool loop survives Claude's silent `reasoning_content` replay drop |

pi is parametrized over its three providers, so one failure isolates to one
adapter: the binary, prompt, model and assertions are identical across the three,
and only the wire format differs. It is also the only tool driving
`/v1/chat/completions`, which no other agentic test reaches. Its provider is
declared the way `docs/use_cases_coding_assistants.md` documents it — a
`~/.pi/agent/models.json` naming `baseUrl`, `api` and one model `id` — so the lane
tests the mechanism the documentation promises rather than a private extension API.

OpenClaw and Hermes repeat that three-route sweep with two things pi does not
have. OpenClaw is the only client where the wire format is *one flag*:
`--custom-compatibility openai | openai-responses | anthropic` on the onboarding
command, with the provider declaration otherwise byte-identical, so a
single-value failure cannot be blamed on the configuration. Hermes selects the
same three routes with `transport` on its provider entry, and is the only client
in the lane that puts explicit Anthropic prompt-caching breakpoints on the wire:
`prompt_caching.cache_ttl` marks the system prompt and the last three messages
with `cache_control`, spelling out `ttl: "1h"` for the long tier and leaving it
implicit for the 5-minute default — the branch `_build_cache_point` exists for.
Both assert on the gateway's request log, not on the CLI's exit code: the route
each value reached, and — for Hermes — the marker shape the gateway received.

Both matrices also pair each wire format with a **Bedrock Mantle** model
alongside the Converse ones, because Mantle serves a given model on some routes
only (`openai.gpt-5.6-luna` and `google.gemma-4-31b` reject a Chat Completions
request outright; Qwen 3 32B is the one with observed tool calls there). n8n pairs
the same three models onto its own three conversational-route tests.
`ChatOpenAI` in `test_langchain.py` and pydantic-ai reach only Chat Completions, so
each pairs Qwen 3 32B onto that route alone; pydantic-ai's is the one that drives a
real multi-turn tool loop with reasoning enabled — the client-level proof that a
Chat Completions client reading reasoning text back under `reasoning`, where
Converse emits it under `reasoning_content`, survives more than a single response.

Qwen Code drives the same route pi does, for a behaviour pi does not have: it
keeps an assistant message's `reasoning_content` in its own history and writes it
back onto that message when it replays the conversation, so a multi-turn tool loop
sends the gateway's own reasoning text back to it. It is the only agent *binary*
that does — pydantic-ai does the same in-process — and the only client that asks
for an effort level as the `reasoning` **object** rather than the flat
`reasoning_effort`.

n8n is the lane's only non-agent. It is a workflow runner whose OpenAI node
implements the whole OpenAI REST surface and whose credential carries a base URL,
so one npm package covers every route the coding agents never call — and the
node/route pairs it drives are exactly the ones `docs/use_cases_n8n.md` promises.

Haystack is a library rather than a CLI, so its "client" is one committed script,
`rag_pipeline.py`, run to completion by the same runner. It is the only route to
`/cohere/v2/rerank`: n8n's Cohere Reranker node takes no base URL, and no coding
agent reranks anything.

Open WebUI reaches no route the other five miss. It is here because it is a
*shipped, documented* integration: its container environment is copied from
`docs/use_cases_openwebui.md` section by section, so a setting the documentation
promises and the gateway stops honouring fails at the layer users configure. It is
also the lane's first service-shaped client, and the reason the service-container
primitive exists.

wyoming-openai reaches no new route either, but it is the only client that reads
`/v1/audio/speech` the way a voice assistant does: it asks for
`response_format=wav`, parses the RIFF header out of the body **as it arrives**,
and announces the rate it found to its own client — and it keeps three synthesis
requests in flight at once. Every other client here reads that route in one
piece, one call at a time.

langchain-openai and pydantic-ai reach no new route either, but they are the
lane's only pure Python HTTP client libraries: no npm package, no container, no
podman round trip, just the library talking straight to `agentic_server` in the
test process. langchain-openai is the closest thing here to a second independent
implementation of Chat Completions and Embeddings, and its `OpenAIEmbeddings`
class is the vehicle for the token-array trap (see `test_langchain.py`).
langchain-anthropic repeats the basic chat round trip over `/anthropic` through
the same `base_url` mechanism pi uses for `PI_MESSAGES`. pydantic-ai exists for
one reason: its `reasoning_content` replay is genuine (see `test_pydantic_ai.py`
and the "pure-Python HTTP clients" section below), so it is the only tool in the
lane that can prove Claude's dropped-reasoning-block behavior does not break a
real multi-turn tool loop — and the same replay path, driven against Qwen 3 32B
over Mantle instead, proves the gateway's `reasoning`/`reasoning_content` rename
survives one too.

```bash
uv run pytest tests/agentic --agentic -s          # whole lane, with metric lines
uv run pytest tests/agentic --agentic --agentic-rebuild   # refresh the CLIs first
uv run pytest tests/agentic/test_codex.py --agentic -k nova-2-lite
uv run pytest tests/agentic/test_pi.py --agentic -k chat-completions   # one route
uv run pytest tests/agentic/test_openclaw.py --agentic -k anthropic   # one wire format
uv run pytest tests/agentic/test_hermes.py --agentic -k "1h or 5m"   # the cache TTL tiers
uv run pytest tests/agentic/test_qwen_code.py --agentic   # the reasoning round trip
uv run pytest tests/agentic/test_n8n.py --agentic --expensive   # + the image tests
uv run pytest tests/agentic/test_rag_haystack.py --agentic   # the rerank route
uv run pytest tests/agentic/test_open_webui.py --agentic   # the documented env block
uv run pytest tests/agentic/test_wyoming_audio.py --agentic   # streamed TTS
uv run pytest tests/agentic/test_langchain.py --agentic      # both langchain routes
uv run pytest tests/agentic/test_pydantic_ai.py --agentic    # the reasoning-replay proof
```

The last three need the client overlay described under
[In-process clients](#in-process-clients); without it they are dropped from
collection and the run says so.

`-s` surfaces one benchmark line per run:

```
CC-METRICS | amazon.nova-2-lite-v1:0 | test_trace_request_pipeline | steps=  7 | in= 41230 out= 1180
```

## Requirements

- `--agentic` (the lane is opt-in; it makes billable Bedrock calls)
- podman on `PATH`. Inside a toolbox container the local engine cannot create the
  user namespace a rootless container needs, so `--remote` is used automatically
  against the host's podman socket
- Bedrock credentials, as for the rest of the suite
- the client overlay, for the three modules listed below

The CLIs are **never executed on the host**. If podman is unavailable the lane
skips rather than falling back to a host binary, so a run always reports the tool
version it actually tested.

## In-process clients

Three clients are Python libraries rather than binaries, so they run in the test
process: `test_langchain.py`, `test_pydantic_ai.py` and `test_wyoming_audio.py`. Their packages are listed in `requirements.txt` and layered
over the project environment at run time:

```bash
uv run --with-requirements tests/agentic/requirements.txt pytest tests/agentic --agentic
uv run --refresh --with-requirements tests/agentic/requirements.txt pytest tests/agentic --agentic
```

`--refresh` re-resolves them, and is to these clients what `--agentic-rebuild` is
to the images.

They are deliberately **not** in `pyproject.toml`. `uv.lock` is a single universal
resolution covering every dependency group at once, so a client declared there
would be pinned to whatever version also satisfies the gateway's own constraints —
and would be installed by every CI job, none of which run this lane. Keeping them
in a separate requirements file is what lets them float, exactly like the unpinned
packages the container images install, so an upstream release that breaks the
gateway surfaces here. The versions a session actually resolved are printed in its
header:

```
agentic clients: langchain-anthropic==1.5.3, langchain-openai==1.4.1, pydantic-ai-slim==2.23.0, wyoming==1.10.0
```

Without the overlay, those modules are dropped from collection and the header
names them rather than letting the run look complete. `mypy` needs the overlay too,
which is why CI's type-checking step uses it.

## Isolation

Each run is a throwaway container:

| Control | Effect |
|---|---|
| `--network=pasta:-T,<port>` | Own network namespace; only the test server's port is forwarded, from the container's loopback to the host's. The server itself never binds beyond `127.0.0.1`. |
| `--read-only` + `--tmpfs /tmp` | No writes to the image. |
| `--cap-drop=ALL`, `--security-opt=no-new-privileges` | No capabilities, no privilege escalation. |
| `--userns=keep-id` | Runs as the host user; files the agent writes stay owned by the test runner. |
| `--memory`, `--pids-limit` | A runaway agent cannot exhaust the machine. |
| mounts | `stdapi/` read-only at `/src/stdapi`; one writable per-test directory at `/work`. |

Only the gateway package is mounted. The agent cannot read `tests/.env`, `.git`,
`~/.aws` or the virtualenv, so no credential on the machine is reachable from a run.

The repository is **copied** into a staging directory the test process owns rather
than bind-mounted. Under SELinux a tree carries whatever label it was last given,
and one stamped with a container's private MCS categories is unreadable from every
other container — the CLI then reports an empty source tree and the run fails for a
reason that looks nothing like a labelling problem. The copy is relabelled instead,
so a test run never changes a label on the checkout.

The CLIs' own sandboxes are off inside the container. Codex's is built on Landlock,
which the runtime's seccomp profile denies, and it refuses to run any shell command
at all rather than continue unsandboxed. The container above is the boundary.

## Adding a tool

1. Append an `AgenticTool` to `AGENTIC_TOOLS` in `_tools.py`, supplying the npm
   package (or `None`, plus an `image_group`, for a client npm cannot install —
   see below), the gateway route it speaks, a `build` that turns an `Invocation` into
   a command line, a `parse` that normalises its output into an `AgenticResult`,
   and a `prepare_workdir` that seeds any config it needs.
2. Add a `test_<tool>.py` in this directory that sets a module-level `TOOL` and
   parametrizes on `model_config`. A module driving one CLI over several routes
   parametrizes an `agentic_tool` fixture instead of setting `TOOL`; the
   identity check picks up either (see `test_pi.py`).

Nothing else is needed. The image picks the package up automatically, because its
tag is a digest of the package list and the `Containerfile` — changing either
rebuilds on the next run. The server, sandboxing, assertions, metrics and the
autouse model-identity check are all shared.

A client that is a Python library takes a third step instead of the first one's
npm package: add it to `requirements.txt`, and register the module against its
import name in `_HOST_CLIENTS` in `conftest.py` so a run without the overlay drops
it from collection rather than failing to import. See
[In-process clients](#in-process-clients).

## Two primitives for clients that do not fit

### Image groups — a client the shared image should not carry

Every tool lands in the shared Node.js image by default. A tool naming an
`image_group` gets its own image instead, built from that group's `containerfile`
and from the npm packages of the tools in that group alone:

```python
IMAGE_GROUPS = {
    "default": ImageGroup(name="default"),
    "rag": ImageGroup(name="rag", containerfile="Containerfile.rag"),
    "hermes": ImageGroup(name="hermes", containerfile="Containerfile.hermes"),
}

AgenticTool(id="haystack", npm_package=None, binary="python", image_group="rag")
```

Reach for a group when the client is **not an npm package** (`npm_package=None`
plus a Containerfile of its own), or when its install tree is large enough that
baking it into the shared image would make every other tool's `--agentic-rebuild`
pay for it — n8n's `"n8n"` group is the second reason: same `Containerfile`, own
package list, and a 2.3 GB image nobody else waits for. `agentic_image` resolves
the group of the tool under test and builds each group at most once per session,
so a run that touches one group never builds another.

`Containerfile.hermes` is the first reason again: `hermes-agent` is a PyPI package,
so its group starts from a Python base rather than the Node.js one. That base names
a minor version — `hermes-agent` declares `requires-python >=3.11,<3.14` and refuses
to install on 3.14 — while the package itself and the patch level still float, and
the build records what it installed for the session header, exactly as the `rag`
group does.

### Service containers — a client shaped as a server

`run_in_container` runs one command to completion, which is wrong for a client
that *is* a server. `_podman.py:start_service_container` starts a **pulled** image
detached, publishes its port back to the host's loopback, and polls until it
answers:

```python
service = start_service_container(
    image="ghcr.io/example/thing:main",
    port=find_free_port(),  # same port inside the container and on the host
    workdir=agentic_workdir,
    env={"DATA_DIR": "/work/data", "HOME": "/work/home"},
    forward_port=agentic_server.forward_port,  # how it reaches the gateway
    data_dirs=("data", "home"),  # created before the start, owned by the runner
    health_path="/health",  # None probes the TCP port, for a non-HTTP service
    startup_timeout=120,
    user=f"{workdir.stat().st_uid}:{workdir.stat().st_gid}",  # image runs as root
)
# ... the tests talk to service.base_url ...
stop_service_container(service)
```

The sandbox is the one-shot runner's: no capabilities, no new privileges, a
read-only root, `/work` the only writable mount. Five things bite:

- the service must listen on **all interfaces** inside the container — pasta
  forwards an inbound connection to the container's own address, so a server bound
  to the container's loopback is unreachable and the health poll times out;
- the host port and the container port are the same number, since a pasta port
  spec cannot be remapped inside podman's comma-separated option string;
- state the service writes must live under `/work` (`data_dirs`), because the root
  filesystem is read-only. `read_only=False` exists for an image that writes
  elsewhere — use it only with evidence, and say why at the call site;
- an image whose `USER` is root needs `user=`. Under `--userns=keep-id` container
  root is a subordinate host UID, so with no capabilities it cannot write into
  `/work` at all, and anything it did write would be undeletable by the test
  runner. Open WebUI boots fine read-only once it runs as the workdir's owner.
- `health_path=None` proves only that the **port was published**, not that the
  service bound it: pasta accepts a connection on the host side as soon as the
  container is created and resets it until something inside is listening. A
  caller polling the TCP port has to follow up with a handshake of its own
  protocol — `test_wyoming_audio.py:_await_ready` is what that looks like.

Containers still running when the interpreter exits are force-removed, so a
crashed test leaks nothing.

## The Qwen Code reasoning round trip

`test_qwen_code.py` runs the lane's standard agent task over three models, then
asks a reasoning model the same kind of question with `reasoning.effort` set and
asserts on the **gateway's request log**, not on the answer: a client that read
the reasoning text and dropped it would still answer correctly, and only the
logged body shows the text arriving back on an assistant message. Three things
bite:

- **`--output-format json` prints the whole transcript in one write**, and Node's
  write to a pipe is asynchronous while the process exits as soon as it has
  printed — the tail is lost and the JSON ends mid-string. The run redirects to
  `/work/run.json` and `cat`s it back, exactly as the n8n runs do. `"$@"` carries
  the argument vector through `sh -c`, so the prompt is never re-quoted.
- **`--safe-mode` ignores `--core-tools`** (it says so on stderr, then registers
  every tool including `write_file` and `edit`), so the read-only posture comes
  from `--exclude-tools`, which safe mode does honour.
- **the effort level travels through `~/.qwen/settings.json`**, not a flag:
  `model.reasoningEffort` becomes `reasoning: {effort: …}` on every request. It is
  only written when the test asks for an effort, because a model with no reasoning
  knob would reject the object.

DeepSeek occasionally ends a turn by writing its own DSML tool-call markup as
answer text instead of making a tool call, which leaves no prose to assert on;
that test carries `@pytest.mark.retry` for it.

## n8n workflows

`workflows/*.json` holds one committed n8n workflow per gateway surface. The port
is allocated per session and the model per test, so a literal workflow cannot
work: `_tools.py:_n8n_prepare` renders the template, replacing `__PORT__`,
`__MODEL__`, `__PROMPT__` and `__FILES__`. n8n takes no arguments from a test, so
`ModelConfig.extra_env` doubles as the substitution table — every entry is
exported to the container *and* available to the template as `__KEY__` (that is
how `__SIZE__`, `__SECONDS__` and `__PURPOSE__` arrive).

Each run is `import:credentials` → `import:workflow` → `execute --id --rawOutput`
against a database created from scratch inside `/work`. Four things bite:

- **`execute` truncates a large stdout.** Node writes to a pipe asynchronously and
  the command exits as soon as it has printed, which cut a 400 kB embeddings
  record mid-number. The run redirects to `/work/run.json` — a synchronous write —
  and `cat`s it back, which is also what lets a test re-read every node's output
  rather than just the terminal one's (`n8n_execution_record`, `n8n_node_items`).
- **file nodes are restricted to `~/.n8n-files` by default** and answer "Access to
  the file is not allowed" anywhere else, so `N8N_RESTRICT_FILE_ACCESS_TO` names
  `/work/files` — which also keeps the run's own database and credential file out
  of reach of the workflow under test.
- **a resource-locator model id must be written as an expression** (`"=__MODEL__"`)
  wherever the node gates another parameter on the model name. The Edit an Image
  node only shows its binary input when the model id contains `gpt-image` or
  `dall-e`, so a Bedrock id makes `getNodeParameter("binaryPropertyName")` throw;
  a value starting with `=` short-circuits every `displayOptions` rule keyed on it.
- **some operations hard-code their model.** Transcribe and Translate always send
  `whisper-1`, and Classify always sends `omni-moderation-latest`, so those tests
  pin the gateway's *aliases* rather than a model parameter.

## The Haystack RAG pipeline

`rag_pipeline.py` is committed as real Python — ruff and mypy see it, the `rag`
image runs it — and is copied verbatim into `/work` per run; the gateway, the
three models and the question all arrive as environment variables, so nothing is
rendered. It prints one JSON record and writes the same record to
`/work/rag_run.json`, which is what `_tools.py:haystack_record` reads back for the
assertions. Two things bite:

- **the assertion is the *movement*, not the status code.** The corpus is built so
  the two rankings disagree: every decoy repeats the question's wording while the
  planted document answers it in a different register, so vector retrieval leaves
  the planted document mid-table and only a rerank reply the client could actually
  consume puts it at rank 0. Both rankings are deterministic, so the margins
  (~0.20 on retrieval, ~0.12 on rerank) are stable rather than lucky.
- **the model under test has to be the pipeline's last call.** The shared identity
  check attributes positionally and ignores everything before the first request to
  the model it was given, so the chat model is the parametrized one and the
  embedding and reranking models are pinned by an explicit per-route assertion.
- **`CohereRanker` appends `v2/rerank` to its base URL**, so `api_base_url` is the
  gateway's `/cohere` prefix and not the full route.

## The Open WebUI service

`test_open_webui.py` boots one container per module and talks to its REST API with
plain `httpx`; there is no `AgenticTool` entry and no browser. Four things bite:

- **its settings are `PersistentConfig`.** Open WebUI reads them from the
  environment on the *first* boot only and stores them in SQLite, so `DATA_DIR`
  must be a directory no earlier run touched — hence the module-scoped
  `tmp_path_factory` workdir. There is also no fallback from `RAG_OPENAI_*`,
  `IMAGES_OPENAI_*` or `AUDIO_*_OPENAI_*` to the core `OPENAI_API_*` pair: a
  missing pair disables that feature instead of inheriting the connection.
- **the admin account is minted by `POST /api/v1/auths/signin`.** With
  `WEBUI_AUTH=False` that handler creates `admin@localhost` on first use and
  returns its JWT, which is what replaces the browser-only setup form.
- **the image's `CMD` is relative** (`bash start.sh` from `/app/backend`), and the
  sandbox sets the working directory to `/work`, so the entry point is passed
  absolutely. `PORT` and `WEBUI_SECRET_KEY` are set too: the default port is fixed
  at 8080, and without a key `start.sh` writes one next to its read-only self.
- **the rerank assertion is the movement**, as in the Haystack pipeline. The same
  question is asked twice — `hybrid: false` for the pure vector order, `hybrid:
  true` for the configured hybrid-plus-rerank order — and the paragraph that
  answers it is last on vectors and first after the rerank. `CHUNK_SIZE` is
  lowered so each paragraph of the uploaded document is its own chunk; the default
  would fold the file into one and leave the reranker nothing to order.

## The wyoming-openai proxy

`test_wyoming_audio.py` boots the proxy as a service and drives it with a plain
asyncio client from the `wyoming` package, so nothing parses audio inside the
container. One Polly model is named in both `TTS_MODELS` and
`TTS_STREAMING_MODELS`, which is what puts its voice in the proxy's streaming
program and makes both synthesis paths reachable from one boot. Four things bite:

- **the readiness probe has to speak Wyoming.** See the service-container caveat
  above: a TCP connect proves nothing here, and every test failed with a
  connection reset before `_await_ready` exchanged a real `describe`.
- **the announced sample rate is the assertion.** The proxy buffers the response
  body only until `wave` parses a RIFF header, then strips it and forwards the
  frames; when no header ever parses it falls back to a fixed 24 kHz. Reading
  back the gateway's 16 kHz is what proves the header arrived early enough to be
  parsed *mid-stream*, and that the payload no longer carries it.
- **concurrency is read off the server log.** Each entry carries the instant the
  gateway started serving the request and how long it took, so overlapping
  windows are a one-way proof: a client that serialises its calls can never
  produce one. Four sentences in a single `synthesize-chunk` give three
  concurrent calls plus a final one, because the proxy holds back only the
  trailing sentence until `synthesize-stop`.
- **the models are pinned per route.** The proxy drives no registered CLI, so the
  lane's autouse identity check has no tool to attribute requests to and the
  explicit route assertion is the only thing tying `amazon.polly-neural` and
  `amazon.transcribe` to the traffic.

## The pure-Python HTTP clients

`test_langchain.py` and `test_pydantic_ai.py` drive their
libraries in-process against `agentic_server`, with no container at all: unlike
every other module here, they are plain HTTP client libraries, not third-party
binaries, so there is nothing for podman to sandbox. Each still declares a real
`AgenticTool` (or an `agentic_tool` fixture, for `test_langchain.py`'s two routes)
purely so the autouse model-identity check runs for free; its
`build`/`parse`/`prepare_workdir` are never called, since none of them ever
requests `agentic_image` or calls `run_agent`. Their packages come from the
overlay, not from `uv.lock` — see [In-process clients](#in-process-clients).
Two things bite:

- **the shared podman skip still applies.** `_model_identity_check` checks for
  podman before resolving `agentic_server`, for every module in this directory —
  including these two. A machine without podman skips them exactly like every
  container-driven tool, even though neither module would otherwise need it. This
  is an accepted limitation of the shared fixture, not something either module
  works around.
- **pydantic-ai's `OpenAIChatModel` reads `reasoning_content` back with no
  signature.** It falls back through `reasoning` and `reasoning_content` when no
  provider profile names a custom field, and replays whichever field it read from
  under the same name — the OpenAI Chat Completions wire format has nowhere to
  carry a signature at all. Claude models on this gateway reject exactly that
  kind of replay and drop the block instead of the request (see
  `docs/api_openai_chat_completions.md`, "Replaying Reasoning in a Multi-Turn
  Conversation"); `test_pydantic_ai.py` is the empirical proof that a real
  multi-turn tool loop still completes. The same fallback also reads Qwen 3 32B's
  reasoning text back from Mantle's `reasoning` field: `TestMantleReasoningReplaySurvivesToolLoop`
  is the client-level guard for that gateway-side rename, which only emits its text
  at `reasoning_effort="high"` — `"low"` has no observable effect on this model.

## What the tests actually assert

- the client exits 0 and emits parsable output;
- for the agents, the answer contains vocabulary that only appears in files the
  agent had to open, so an answer recited from the model's own knowledge fails;
  for n8n and Haystack, the run's own output is asserted stage by stage (the
  planted document ranks first, the transcript carries the synthesised words, the
  uploaded file id appears in the listing, the image decodes);
- a **step floor** — turns for Claude Code, completed shell calls for Codex,
  completed tool calls for pi and OpenClaw, model API calls for Hermes, emitted
  tool-use blocks for Qwen Code, executed nodes for n8n, components that produced
  output for Haystack — proving the round trip carried real data back through the
  gateway;
- **model identity** (autouse): every request the gateway logged for this test targeted
  the parametrized Bedrock model. Without it, a CLI silently falling back to its own
  default model would still pass and the test would prove nothing. The n8n and
  Haystack tests add an explicit route assertion on the server log, because several
  of their surfaces resolve a model the identity check cannot see; Open WebUI and
  wyoming-openai drive no registered CLI, so that per-route assertion is the only
  thing pinning their models.

Models marked `flaky=True` in their `ModelConfig` downgrade *content-quality*
failures to `xfail`; every other failure signature still fails, so real regressions
stay visible.
