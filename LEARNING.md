# LEARNING.md — real bugs, real decisions, in the order they happened

This documents what actually happened building TicketPulse, not a cleaned-up
retelling. See [DESIGN.md](DESIGN.md) for the architecture these decisions
produced.

## 1. `AmazonScience/massive` doesn't load with a current `datasets` install

The build spec suggested `amazon_reviews_multi`-style datasets or MASSIVE
directly. `load_dataset("AmazonScience/massive", "en-US", ...)` fails:

```
RuntimeError: Dataset scripts are no longer supported, but found massive.py
```

Recent `datasets` versions dropped support for the old dataset-loading-script
format, and `AmazonScience/massive` on the Hub still ships one. A public
Parquet mirror of the same corpus's scenario-classification split,
`mteb/amazon_massive_scenario`, has the identical utterances/labels in a
format the current library actually loads — confirmed by loading all 8
locales this project samples from (`en`, `es`, `fr`, `de`, `hi`, `ar`,
`zh-CN`, `pt`) and checking the label set matches MASSIVE's documented 18
scenarios exactly. Switched to that mirror; see DESIGN.md.

## 2. `pytest` (the console script) can't find `src`, `python -m pytest` can

First real test run:

```
ModuleNotFoundError: No module named 'src'
```

...despite `pip install -e .` reporting success and every test importing
`from src.common import config`. The editable install's `.pth` file adds
`src/` itself to `sys.path` (setuptools' src-layout auto-discovery unpacks
`src/agents`, `src/common`, `src/orchestrator` as top-level packages, not
`src` as a package) — so `import common.config` would work, but
`import src.common.config` (what every file in this project actually
writes) would not, *unless* the project root itself is also on `sys.path`
so `src/__init__.py` is reachable as a regular package.

The actual fix wasn't a config change: `python -m pytest` adds the current
working directory to `sys.path[0]`; the bare `pytest` console-script entry
point does not (its `sys.path[0]` is `.venv/bin`, not the cwd). Every
invocation in this project's Makefile and CI-equivalent commands uses
`.venv/bin/python -m pytest`, never a bare `pytest`, for exactly this
reason.

## 3. Running every agent as a genuinely separate process is a real RAM cost

Mid-build, a smoke test with all 4 model-loaded/queue-consuming agents
running as separate long-lived processes pushed the host into **18.3GB of
19.4GB swap used**, with CPU pinned in `sys` time and almost no forward
progress (1 of 140 tickets processed in 9 minutes). This wasn't a bug in
the pipeline logic — it was the direct, structural cost of "genuinely
separate agent processes" the build spec asks for: a monolithic script
could load the translation model, use it, unload it, then load the
classification model; five independent long-lived services each hold
their own model resident in memory for the entire run, so peak RAM is the
*sum* of every agent's model, not the max. On a host already running
Docker Desktop's own VM (~8GB resident) and other applications, that sum
didn't fit.

Fix wasn't code — it was recognizing the constraint and adapting the run
strategy: killed the workers, ran the actual evaluated pipeline as local
OS processes against a native Homebrew Redis (`brew install redis`)
instead of a full `docker compose up` stack, avoiding Docker Desktop's own
VM overhead on top of the model memory. The Dockerfile/docker-compose.yml
are still real, tested deliverables (`docker compose config` validates;
see README) — they just weren't the execution path for the run this
project's numbers came from, and that's stated plainly rather than implied.

## 4. Back-translation on every ticket was needless 2x cost

Initial `translate_with_consistency_check` always did a forward translate
*and* a back-translate-and-compare on every non-English ticket, to compute
the consistency score. Once real inference started, this was clearly the
translation stage's dominant cost — and unnecessary: the eval spec asks
for this metric "across a held-out sample," not on every ticket. Added
`config.BACK_TRANSLATION_SAMPLE_RATE` (default 0.20) so most non-English
tickets get only the one forward translate the pipeline actually needs;
the round-trip only runs on a sampled subset, which is all the quality
metric ever needed. Combined with switching `generate()` from m2m100's
default beam search to greedy decoding (`num_beams=1` — a live streaming
agent trading a little quality for CPU throughput on short utterances),
this cut translation-stage wall-clock by roughly 7x in local timing
(~7s/ticket to ~1s/ticket on this machine).

## 5. `dict.setdefault(k, RunState(k))` silently truncated every output file on every message

The most serious bug found this session, and the reason it's worth a full
retrospective: it produced no exception, no error log, no visibly wrong
behavior while the pipeline ran — it just quietly corrupted the eval data.

`anomaly_agent/main.py`'s loop looked up (or created) the per-run state with:

```python
state = states.setdefault(msg.run_id, RunState(msg.run_id))
```

`dict.setdefault(key, default)` evaluates `default` **eagerly, every single
call**, whether or not `key` is already present — only the *storage* is
conditional, not the construction. `RunState.__init__` truncates
`day_counts.csv` and `tickets_scored.jsonl` (`open(path, "w")`, to start
each file fresh for a new run). So on every ticket and every boundary
message — not just the first one for a run — a throwaway `RunState` got
constructed, truncated both files to empty, and was then discarded because
the dict already had a real entry for that `run_id`. The real state's
`record_ticket`/`close_day` calls right after that always ran against an
already-truncated file, so both files only ever ended up holding whatever
was written *after the last message's throwaway truncation* — in practice,
the final day's rows and nothing else.

Caught by noticing `reports/smoke2/day_counts.csv` only contained day 1's
rows after a clean 2-day smoke test, despite `summary.jsonl` (built from the
same in-memory `RunState.history`, never touching that file) correctly
showing day 0's counts. That mismatch — one output correct, a sibling
output derived from the same data wrong — was the tell that this was a
file-write bug, not a counting bug.

Fixed by replacing the one-liner with an explicit check
(`get_or_create_state()`), and added a regression test
(`test_get_or_create_state_does_not_retruncate_files_on_repeat_calls`) that
calls it twice for the same `run_id` across two days and asserts both
days' rows survive in the CSV — the exact shape of bug this class of test
exists to catch, and the kind `dict.setdefault`'s eager-evaluation gotcha
will keep producing if it's ever reintroduced elsewhere in this codebase.

**This means the earlier smoke1/smoke2 numbers referenced in section 3-4
above were themselves collected before this fix landed** — this project's
actual reported eval numbers (README.md) are from a run made after this fix.

## 6. Aborted runs leave orphaned messages in the shared queue

Queues are shared across `run_id`s by design (so two runs could stream
concurrently — every message and every `RunState` is keyed by `run_id`).
That has a real downside: a run killed mid-stream (as in #3) leaves its
unconsumed messages sitting in `tickets:raw`/`tickets:translated`/etc.
forever. The *next* run's `bash scripts/run_local_demo.sh` then silently
drains those leftovers first, alongside its own fresh messages — harmless
to correctness (every message still carries its own `run_id`, so results
land in the right place), but it wastes real model-inference time
reprocessing a run nobody cares about anymore, and inflates queue-depth
numbers in a confusing way while debugging. `make flush` (or
`redis-cli flushall` directly against the local instance) is the
mitigation; a real system would want a cancel-run message or a per-run
queue namespace instead.

---

*(Real end-to-end numbers — translation quality, classification F1 on both
taxonomies, anomaly detection precision/recall/F1 with the threshold sweep,
and hard-example accuracy — are in README.md, from the run described
there.)*
