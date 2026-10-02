# context-harness

One core, two domains. The core (`harness/`, stdlib-only) is a small harness around a language-model call.
It assembles context under a hard token budget, resolves stale and contradictory records before the prompt is
built, gates sensitive facts by relevance while always shipping their constraints, routes across model tiers,
lets the model recall what was dropped, checks the answer deterministically, writes memory back, and traces
every decision. It knows no subject area and no vendor: a `Domain` supplies the subject, and a `sender`
supplies the model call.

Not medical advice and not a medical device. It has only been run on mock data, and the live model path has never made a real call.

```
harness/            the core: context, resolve, safety, router, memory, tools, trace, turn, model (stub + sender seam)
providers/
  anthropic.py      the only module that imports `anthropic`: build_request and send (tool loop, refusals, errors)
domains/
  lab/              a research assistant for one scientist; its live path uses providers.anthropic.send directly
  health/           wearables plus a local-only EHR; reaches the model only through model_boundary.send_remote
  health/mock/      the invented-data generator
run.py              lab CLI          health_cli.py   health CLI
evals/              lab_cases.py, health_cases.py, judge.py, run.py (--domain lab|health|all)
tests/              test_lab_*, test_health_*, test_core*
```

Python 3.10+. Everything except live model calls and the health vault is stdlib. Extras: `pip install .[live]`
(anthropic) and `.[health]` (cryptography), or `pip install -r requirements.txt` for both.

How a turn runs: safety pre-check (emergency patterns end the turn with a fixed reply and no model call);
resolve (latest record wins, a touched plan becomes a tension to surface, domain rules add notes and values);
memory read; assemble tiered candidates under the budget (required ones degrade but never drop, optional ones
drop with a recorded reason); route small, mid or large; call (live with a three-round `recall` tool loop, or the
stub); post-check (no endorsement of a risky plan, every tension addressed, no forbidden term recommended, every
number present in the prompt, length cap; one failure escalates a tier, a second ships a templated fallback);
write memory. Every stage lands in the trace. A budget below the context floor exits with status 2.

## Domains

### Lab

A postdoc is purifying a protein for crystallography. The turn is hard because of plan drift (the Sep 19 plan,
37°C induction, and the Sep 26 revision, 18°C, are both active and today's message goes back to 37°C), a
preference reversal, a same-day conflict ("run looks clean" then "huge void-volume peak"), a notebook that says
about 2 mg/mL where the instrument says 0.6, and a sensitive pinned record whose name is gated but whose
constraint always ships.

```
python3 run.py --stub                                   # default turn, budget 1050; writes runs/
python3 run.py --stub --budget 600                      # tight: more degrading and dropping
python3 run.py --stub --trigger morning_brief --memory runs/memory_after.json --ts 2026-09-30T08:30
python3 run.py --stub --trigger temperature_excursion   # signal trigger
python3 run.py --stub --out examples                    # regenerate the committed example
python3 run.py --live                                   # needs the anthropic package and credentials
python3 -m evals.run --domain lab                       # add --live --judge for the LLM judge
```

Without `--live` or `--stub`, the mode is live if `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` is set. The first
output line says which mode ran and why.

### Health

A personal health harness for one person's own data: an Epic EHR (through a patient portal), a Garmin
watch and a WHOOP strap. It reads their exports, keeps the sources apart (each against its own baseline, precedence
set by you, disagreement reported and never averaged), and assembles a traced, budgeted context. All data in this
repo is mock.

**The rule: EHR data never leaves the machine.** Enforced in code and tests, not by convention:

- **Origins and modes.** Every record and context candidate has an `origin` (`ehr`, `wearable`, `user`,
  `derived_local`, `system`). `local` assembly (stub or an on-device model) may render everything; `remote` assembly
  (any model API) removes `ehr` candidates before assembly, never truncating them into the prompt. A live call needs `remote`.
- **Derived constraints are the only bridge.** `domains/health/ehr/derive.py` maps EHR content, with a local table, to a
  fixed allowlist of template sentences (no names, drugs, conditions, codes, values, or dates finer than a month), e.g.
  "A clinical factor lowers resting heart rate ... from February 2026". They are `derived_local` and pinned.
- **Scrub, then guard.** Outgoing prompts first have EHR medications, conditions, labs, values, identifiers and codes
  (plus lay terms and brands) replaced by placeholders like "[a medication on file]"; `send_remote` then scans every
  request against a denylist built from your EHR plus Safe Harbor patterns. A hit blocks the call and reports
  categories, never values; an unreadable store blocks everything. Only `providers/anthropic.py` imports `anthropic`,
  and only `domains/health/model_boundary.py` calls the provider; tests enforce both.
- **Local answers.** "What meds am I on", "my last A1c", "my allergies", and questions about your own lab results
  ("is my A1c bad?") are answered by deterministic code from the store, in both modes; no model call is made.
- **At rest.** EHR content lives in one Fernet file, key from `HEALTH_HARNESS_PASSPHRASE` via scrypt with a random salt.
  Every read appends a line to `access.log` (time, purpose, resource types; no values). Traces, history and new memory
  are redacted against the denylist before they are written, and before any of their text is clipped to a length limit, so a
  cut cannot leave half a denylisted word behind. The data dir (`~/.health-harness`) is 0700.

HIPAA, briefly: an individual processing their own records in their own tool is neither a covered entity nor a business
associate, so HIPAA does not govern their copy (HHS OCR's health-app guidance), though the tool is built to the shape of the
Security Rule (encryption at rest, access logging, minimum-necessary egress, no PHI in logs) so that offering it to others
later needs no redesign. Sending PHI to any model vendor would require a HIPAA-eligible service and a business associate
agreement, which is why EHR content stays local; this is not legal advice.

Mock quick start:

```
pip install -r requirements.txt          # cryptography; anthropic only for --live
export HEALTH_HARNESS_PASSPHRASE='choose-one'  HEALTH_HARNESS_DATA=/tmp/hh-demo
python3 health_cli.py mock --out mock_exports
python3 health_cli.py import whoop  mock_exports/whoop
python3 health_cli.py import garmin mock_exports/garmin
python3 health_cli.py import epic   mock_exports/epic        # directory of FHIR bundles; also derives constraints
python3 health_cli.py import memory mock_exports/memory_seed.json
python3 health_cli.py status             # sources, coverage, overlap window, gaps, imports
python3 health_cli.py audit              # signal x source x granularity x coverage x overlap, 14-day disagreement
python3 health_cli.py turn "can I take melatonin tonight?" --stub --mode remote --show-prompt
python3 health_cli.py brief --ts 2026-09-30T08:30
python3 health_cli.py egress-check       # renders the remote prompt, runs the guard: PASS or blocked categories
python3 -m evals.run --domain health
```

`turn` defaults to `--stub` (local assembly). `--live` forces remote assembly and the guard. `--mode remote` with the
stub shows exactly what a model would be sent. `sync whoop|garmin|epic` prints what each would need (nothing is
implemented). `precedence.json` in the data dir sets which source wins per signal; sleep differing by more than 45 min
or resting HR by more than 5 bpm becomes a note the model sees. Mark travel or illness with
`health_cli.py regime add START END KIND LABEL`.

Real exports:

- **WHOOP:** app > More > App Settings > Data Export > Create Export; the emailed zip of CSVs: `import whoop export.zip`.
- **Garmin:** garmin.com account > Data Management > Export Your Data; the emailed zip (48 h to 30 days):
  `import garmin export.zip`. The Health API is for approved businesses, not individuals.
- **Epic:** health systems on Epic expose FHIR R4 patient access; find your organisation's base URL in Epic's public
  endpoint directory (https://open.epic.com/Endpoints/R4). It uses SMART standalone launch with PKCE S256 and needs a
  registered patient-facing app. Some portals offer only a password-protected PDF download, others offer the XDM zip.
  Save each search response as a JSON file and `import epic <dir>`. If your organisation offers MyChart's XDM zip,
  `import epic <zip>`.

A file inside a zip larger than 512 MB uncompressed, or compressed more than 200 to 1, is refused before it is read.
So is a whole archive whose members declare more than 2 GiB uncompressed in total, that has more than 20,000 members, or
that has a member name which is absolute or contains `..`; these are checked from the zip's directory before anything is read.

## Tests

```
python3 -m unittest discover tests      # lab, health and core suites
python3 -m evals.run                    # every eval case, both domains, stub mode; writes runs/evals.json
```

`--only NAME` runs one eval case. Repository rules are tests (`tests/test_core_repo.py`): the core never imports from
`domains/` or `providers/`, `anthropic` is imported in one file, the health domain reaches the provider only through
`send_remote`, no mock EHR value appears outside `domains/health/mock/` and the tests.

## How to add a domain

Read `harness/domain.py`; it defines the `Domain` dataclass and everything the core asks of one. `domains/lab/` is a
small complete example: scenario data, a memory seed, resolution rules, extra context candidates, safety patterns
(risk rules take `(text, pinned, scenario)`), triggers and a synonym table. `domains/health/` shows the optional hooks:
local responders (a flag, or `"*"` for every turn, routes to deterministic code instead of the model), a domain stub,
post-check requirements, and a `Boundary(mode, sender, preflight, scrub, scrub_detail)` for a domain that needs an
outbound guard. For a live call, pass `providers.anthropic.send` as the sender (the lab does), or wrap it, passing a
`check(request)`, if the domain must guard what leaves (the health domain does). Add a cases module under `evals/` and
register it in `evals/run.py`. The core never imports from `domains/`.

## Status

v0.2. The lab path is wired live but the committed example trace in `examples/` is from the stub. Retrieval is keyword
overlap and will not scale past a few hundred records. Model-extracted memories are not implemented; the seam is
commented in `harness/turn.py`. Health is mock-only: importers were built against documented and community-reverse-
engineered formats (WHOOP CSV, Garmin Connect export, Epic FHIR) and need checking against a real export; places that
guess are marked `format: unconfirmed`. No sync is implemented (the WHOOP API mapping is tested on mock JSON; Garmin and
Epic are skeletons). The live health path is untested with real credentials. The C-CDA parser covers four sections only
(allergies, medications, problems, results). The denylist matches the exact strings in your own record, which is why
EHR content reaches a model only as templates.

Built with Claude Code, which wrote most first drafts.
