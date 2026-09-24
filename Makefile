.PHONY: venv fixture infra-up infra-down flush demo eval test lint

venv:
	python3 -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q -e ".[dev]"

# Synthetic ticket stream + known injected anomaly windows + hard-examples
# set, all from the public no-login mteb/amazon_massive_scenario mirror of
# MASSIVE. No token needed.
fixture:
	.venv/bin/python scripts/generate_tickets.py

infra-up:
	docker compose up -d redis ollama translation_agent classification_agent anomaly_agent synthesis_agent

infra-down:
	docker compose down

# Queues are shared across run_ids by design (so two runs COULD stream
# concurrently, each tracked by its own RunState) -- but that means a run
# killed mid-stream (Ctrl-C, a crash) leaves its unconsumed messages sitting
# in the queue forever, to be silently drained alongside whatever the NEXT
# run pushes. Harmless to correctness (every message is run_id-tagged, so
# results still land in the right run's reports/), but it wastes real
# model-inference time reprocessing a run nobody cares about anymore. Run
# this before a fresh `make demo` if a previous one didn't finish cleanly.
flush:
	docker compose exec redis redis-cli flushall

# End-to-end demo against the synthetic fixture, fully containerized over
# the real Redis broker and the real local models. First run pays the
# model-download cost (~3.5GB HF models into the hf_cache volume, ~1.9GB
# Ollama model into ollama_data) inline.
demo: infra-up
	RUN_ID=demo-$$(date +%s); \
	DATA_FILE=tickets.jsonl RUN_ID=$$RUN_ID docker compose --profile run run --rm ingestion_agent; \
	echo "--- waiting for the pipeline to drain ---"; \
	for i in $$(seq 1 120); do \
		grep -q '"is_final": true' reports/$$RUN_ID/summary.jsonl 2>/dev/null && break; \
		sleep 5; \
	done; \
	echo "--- reports/$$RUN_ID/ ---"; ls reports/$$RUN_ID; \
	.venv/bin/python eval/run_eval.py --run-id $$RUN_ID

eval:
	.venv/bin/python eval/run_eval.py --run-id $(RUN_ID)

test:
	.venv/bin/python -m pytest -q

lint:
	.venv/bin/ruff check src scripts tests eval
