# Model capability probes

Model cards and service models declare *shapes*. They do not tell you whether a
model honours a parameter, silently ignores it, or refuses it — and that gap is
where this gateway's model-specific bugs live. Three found in one week:

- Kimi K2.5 accepted `thinking={"type":"enabled"}` and returned no reasoning at
  all; only `reasoning_effort="high"` turns it on.
- Claude Sonnet/Haiku 4.5+ reject `temperature` and `topP` together, where every
  earlier generation accepted both.
- Every Stability image model refuses `quality`, which the models the gateway
  offers it to happily volunteer.

None of the three is documented. All three are one probe away.

## Probing a model

```bash
uv run python -m tests.probes.probe_model <model-id> [<model-id> ...] [--region us-east-1]
```

Each probe sends one request that differs from a known-good baseline in exactly
one way and classifies what came back:

| Outcome | Meaning |
| --- | --- |
| `supported` | The call succeeded **and** the feature's own effect appeared — a reasoning block, a tool call, a cache write. |
| `accepted` | The call succeeded and nothing observable changed. The knob is tolerated and inert: the most dangerous result, because it looks like success. |
| `rejected` | The backend refused it. The recorded message is what a caller would see. |
| `error` | Anything else, recorded verbatim — including an answer that stopped at the output budget before the effect it was looked for in could appear. |
| `skipped` | The probe does not apply to this model. |

Results are written to `results/<model-id>.json` and are meant to be committed:
they are the evidence behind every model-specific branch in
`stdapi/models/chat/`. A cross-region-only model is retried through its regional
inference profile automatically, and the id actually invoked is recorded.

## Reading the result

Work through the record against the model's class in `stdapi/models/chat/`:

- **`accepted` on a knob the gateway forwards** — the gateway is promising
  something the model does not do. Either send the field that *does* work
  (Kimi's `reasoning_effort`), or stop advertising the capability.
- **`rejected` on a knob the gateway forwards** — a request that should work
  returns a 400. Decide per the *Unsupported features* rule in `AGENTS.md`:
  drop the incidental knob, refuse the explicit ask.
- **`supported` on a knob the gateway does not forward** — a capability the
  gateway is leaving on the table.
- **A difference between two models in the same family** — the family matcher is
  too wide, or the class needs a per-generation branch.

Then write the test. The probe proves the behaviour once; the test keeps it
proven.

## Adding a probe

Append a `Probe` to `PROBES` in `probe_model.py`. Give it an `observe` predicate
whenever the feature has an observable effect — a probe without one can only ever
report `accepted`, which distinguishes nothing. Bump `SCHEMA_VERSION` and give
the new probe `added_in=<that version>`: every record names the version it was
probed under, so an older record stays valid without the probes added since,
and `tests/test_probe_records.py` still catches a run that died partway. Re-probe
a model to give its record the newer probes.

**Changing an existing probe's request** (its fields, its budget) also bumps
`SCHEMA_VERSION`, and sets the probe's `changed_in=<that version>`: the records
test compares every recorded request with the current one and fails when they
differ under an unchanged `changed_in`. An older record's result for that probe
is kept but reported as a skip when it reads `accepted`, until the model is
re-probed. When a change makes every older record wrong rather than merely
dated, raise `_OLDEST_SCHEMA_VERSION` in `tests/test_probe_records.py` to the new
version and re-probe the affected models.

Two traps shaped the probe set:

- **A model that always reasons spends a small output budget before answering.**
  With 64 tokens, Kimi K3 never reached its tool call or its JSON, and both read
  as `accepted`. A probe whose effect is in the answer gets a larger budget, and
  an answer that still stops at it is recorded as `error`. Those probes carry
  `changed_in=6`: in an older record, their `accepted` is inconclusive.
- **Implicit prompt caching needs the same prefix twice.** One call can only show
  a cache write; `implicit_prompt_cache` sends its request twice (`calls=2`) and
  looks for the read on the second.

A probe of a reasoning *off* switch (`reasoning_object_effort_none`,
`reasoning_disabled`) proves something only on a model that reasons by default:
read it against `thinking_enabled` and the effort probes of the same record.

The probes cost a handful of tokens each on the smallest prompt that exercises
the path; a full run on a small model is well under a cent.
