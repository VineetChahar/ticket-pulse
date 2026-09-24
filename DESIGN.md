# TicketPulse — design

## Goal

A multi-agent pipeline that ingests multilingual customer support tickets,
translates and classifies them, detects temporal anomalies in per-category
ticket volume, and synthesizes a grounded daily triage report — entirely
with local, no-login models and public data, with an honest evaluation
harness behind every number.

## Data

No real support-ticket dataset is public and no-login. Per the build spec's
own recommendation, this project generates a **synthetic ticket stream**:
real multilingual utterances sampled from a public dataset, laid out on a
simulated calendar, with deliberately injected volume anomalies whose
(category, day-range) ground truth is known exactly — which is the only way
to report real anomaly-detection precision/recall rather than "it looks
right."

**Source corpus**: `mteb/amazon_massive_scenario`, a public, no-login
Parquet mirror of Amazon's MASSIVE corpus's scenario-classification split.
(`AmazonScience/massive` itself ships as a dataset-scripts loader that
current `datasets` versions refuse to run — this mirror has the same
utterances and labels.) Each row is a short utterance (~7 words), a
`label` from 18 scenario categories (alarm, calendar, email, iot, weather,
...), and a language. Rows are id-aligned across per-language configs, so
the same "ticket" exists in all 8 sampled locales (`en`, `es`, `fr`, `de`,
`hi`, `ar`, `zh-CN`, `pt`).

`scripts/generate_tickets.py` turns this into `data/synthetic/tickets.jsonl`:
for each of 35 simulated days and 18 categories, it draws a Poisson-distributed
ticket count around a per-category baseline rate, multiplies that rate inside
3 deliberately injected anomaly windows (see README for the exact windows and
why they're each a different difficulty), and for each ticket samples a
locale and pulls a real utterance from that (locale, category) pool. It also
writes `ground_truth_anomalies.json` (the injected windows) and
`hard_examples.jsonl` (a ~40-example set sampled from scenario groups this
project's own taxonomy treats as confusable, e.g. datetime/calendar/alarm).

## Architecture

Five agents, each a separate OS process, communicating only through Redis
lists (`src/common/broker.py`) — never a shared in-memory object:

```
ingestion_agent  →  translation_agent  →  classification_agent  →  anomaly_agent  →  synthesis_agent
  (one-shot,          (long-lived            (long-lived              (long-lived         (long-lived
   the "run"           service)               service)                 service)            service)
   trigger)
```

Each hop is a Redis list (`tickets:raw` → `tickets:translated` →
`tickets:classified`), with the anomaly agent fanning out a `DaySummary`
per completed day onto `days:summary` for the synthesis agent. Redis
**lists** with `BRPOP`/`LPUSH`, not `PUBLISH`/`SUBSCRIBE`: pub/sub has no
persistence, so a message published before a worker subscribes is dropped
forever — a real race on process startup. A list holds a message until a
worker is ready to pop it (at-least-once delivery).

### The day-boundary barrier

Tickets stream in per simulated day (`ingestion_agent` reads
`tickets.jsonl` grouped by `day_index` and sleeps between days). Every
ticket for day N is pushed, then a `boundary` message for day N is pushed
*after* them. Every downstream agent must forward that boundary immediately
— before doing any of its own end-of-day work — so the barrier propagates
through translation and classification untouched, and only the anomaly
agent (the first stage that needs "day N is complete" as a real signal,
since anomaly detection is inherently day-batched) acts on it: it closes
day N's per-category counts, runs both detectors over the series-so-far,
and emits the `DaySummary`.

This is the only synchronization primitive in the whole system. No agent
needs to know how many tickets are in a day, only that a message with
`kind="boundary"` means "everything before this, for this day, has already
passed through me."

### One evolving message schema, not one per stage

`TicketMessage` (`src/common/schemas.py`) carries a ticket through all
three per-ticket hops. Each stage only ever *adds* optional fields — never
reshapes or removes. This is a deliberate departure from repo-health-radar's
convention of a distinct pydantic Task/Result pair per agent call: that
project's agents each handle one discrete request-response, so a stronger
"this message shape is only valid at this stage" guarantee earns its
ceremony. Here, a message's shape doesn't change hop to hop — it's one
record accumulating fields as it goes — so three separate classes would
just be indirection with no payoff.

### Why zero-shot classification runs twice per ticket

`classification_agent` classifies every ticket against **two** label sets
(`taxonomy.py`): MASSIVE's own 18 scenario labels (scored against real
dataset ground truth), and a 7-category support-desk taxonomy this project
invented that the model has never seen a training example for. There's no
dataset-provided ground truth for the second one — `SCENARIO_TO_CUSTOM` is
a deterministic, human-authored mapping used as a silver standard, so eval
can report a real accuracy number for "does zero-shot generalize to an
unseen taxonomy" instead of a qualitative shrug.

### Anomaly detection: two independent statistical methods

`detector.py` implements both as pure functions of a count array (no I/O,
no agent state), so `eval/run_eval.py` can re-sweep their thresholds
offline against the known injected windows without touching Redis or a
model:

- **CUSUM** (two-sided mean-shift detector): cheap, explainable, catches a
  step-change. Adapted from the same algorithm the repo-health-radar
  project uses for CI-duration anomalies, generalized here to run
  independently per ticket category with a fixed baseline window.
- **Poisson-rate test**: ticket counts are a counting process, so "is
  today's count surprising given the baseline rate" is naturally a Poisson
  tail test, not a Gaussian one — meaningful even at small per-category
  daily counts where a std-dev-based test is noisy. Used instead of
  STL/seasonal-decomposition (the spec's other suggested method):
  `statsmodels.seasonal_decompose` needs >= 2 full periods, and a 35-day
  run has no reliable weekly-seasonality signal to decompose against.

### Synthesis: grounded, not free-generated

`narrative.py` builds a prompt that hands the local Ollama model
(`qwen2.5:3b-instruct`) the day's exact category counts and exact anomaly
flags, and explicitly instructs it not to invent numbers or categories not
listed. If Ollama is unreachable, a template report (no LLM call) is used
instead, and which path ran is recorded (`narrative_source`) so eval and
the report itself can be honest about which reports are template
fallbacks — the same pattern repo-health-radar uses for its synthesis
agent.

## Performance: sampling the back-translation check

Back-translation consistency (translate to English, then back to the
source language, and score similarity against the original) is a real
translation-quality signal, but it's a *second* `generate()` call per
ticket, doubling translation cost. The eval spec asks for this metric
"across a held-out sample," not on every ticket — so
`config.BACK_TRANSLATION_SAMPLE_RATE` (default 0.20) decides per-ticket
whether the round-trip runs at all; every non-English ticket still gets
its one forward translation into English (needed for classification)
regardless. Combined with switching `generate()` to greedy decoding
(`num_beams=1` — a live streaming agent should trade a little translation
quality for CPU throughput on short utterances), this cut the pipeline's
translation-stage cost by roughly 7x. See LEARNING.md for the numbers.

## What's out of scope

- **Run cancellation / queue isolation.** Messages are tagged with
  `run_id` (so results always land in the right run's `reports/<run_id>/`),
  but the queues themselves are shared across runs. A run killed mid-stream
  leaves its unconsumed messages sitting in the queue, silently drained
  alongside whatever the next run pushes — harmless to correctness, wasteful
  of compute. `make flush` clears this by hand; a real system would want an
  explicit per-run queue or a cancel message.
- **Horizontal scaling.** One process per agent. The architecture doesn't
  prevent running N translation-agent replicas against the same queue (Redis
  lists are safe for multiple consumers), but nothing here exercises that.
