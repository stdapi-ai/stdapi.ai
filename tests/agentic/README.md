# Agentic test lane

Real coding-agent CLIs — Claude Code and Codex — driven end to end against a live
stdapi.ai server. They are the only tests that exercise a full agentic loop: a
third-party client's own system prompt, tool definitions, multi-turn tool-call
replays and SSE streaming, all translated by the gateway for a model that is
usually not the vendor's own.

```bash
uv run pytest tests/agentic --agentic -s          # whole lane, with metric lines
uv run pytest tests/agentic --agentic --agentic-rebuild   # refresh the CLIs first
uv run pytest tests/agentic/test_codex.py --agentic -k nova-2-lite
```

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

The CLIs are **never executed on the host**. If podman is unavailable the lane
skips rather than falling back to a host binary, so a run always reports the tool
version it actually tested.

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

## Adding a tool

1. Append an `AgenticTool` to `AGENTIC_TOOLS` in `_tools.py`, supplying the npm
   package, the gateway route it speaks, a `build` that turns an `Invocation` into
   a command line, a `parse` that normalises its output into an `AgenticResult`,
   and a `prepare_workdir` that seeds any config it needs.
2. Add a `test_<tool>.py` in this directory that sets a module-level `TOOL` and
   parametrizes on `model_config`.

Nothing else is needed. The image picks the package up automatically, because its
tag is a digest of the package list and the `Containerfile` — changing either
rebuilds on the next run. The server, sandboxing, assertions, metrics and the
autouse model-identity check are all shared.

## What the tests actually assert

- the CLI exits 0 and emits parsable output;
- the answer is non-trivial and contains vocabulary that only appears in files the
  agent had to open, so an answer recited from the model's own knowledge fails;
- a **step floor** — turns for Claude Code, completed shell calls for Codex — proving
  the tool-use round trip carried file contents back through the gateway;
- **model identity** (autouse): every request the gateway logged for this test targeted
  the parametrized Bedrock model. Without it, a CLI silently falling back to its own
  default model would still pass and the test would prove nothing.

Models marked `flaky=True` in their `ModelConfig` downgrade *content-quality*
failures to `xfail`; every other failure signature still fails, so real regressions
stay visible.
