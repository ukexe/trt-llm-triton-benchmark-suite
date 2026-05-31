# Convenience targets for the trt-triton-llm-bench suite.
# Windows users without `make` can run the underlying commands directly.

PYTHON ?= python
CONFIG ?= benchmarks/configs/single_gpu_baseline.yaml
OUT    ?= results

.PHONY: help setup dev lint format test bench-mock cost compose-config clean

help:
	@echo "Targets:"
	@echo "  setup          Install runtime dependencies"
	@echo "  dev            Install dev + analysis extras (editable)"
	@echo "  lint           Run ruff check"
	@echo "  format         Run ruff format"
	@echo "  test           Run pytest"
	@echo "  bench-mock     Run a mock benchmark (no GPU/server needed)"
	@echo "  cost           Cost-analyze the mock benchmark summary"
	@echo "  compose-config Validate infra/docker-compose.yml"

setup:
	$(PYTHON) -m pip install -r requirements.txt

dev:
	$(PYTHON) -m pip install -e ".[dev,analysis]"

lint:
	ruff check .

format:
	ruff format .

test:
	$(PYTHON) -m pytest

bench-mock:
	$(PYTHON) -m benchmarks.runner.client --config $(CONFIG) --backend mock --out $(OUT)

cost:
	$(PYTHON) -m cost.cost_analysis --summary $(OUT)/single_gpu_baseline/summary.json --gpu A100-80GB --provider runpod

compose-config:
	docker compose -f infra/docker-compose.yml config

clean:
	rm -rf results .pytest_cache .ruff_cache **/__pycache__
