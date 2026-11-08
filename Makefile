# FloodWatch KP local targets
PY ?= python

.PHONY: help install check lint test golden contract fusion-smoke app-imports ownership validate-a validate-b demo clean

help:
	@echo "install        install dev dependencies (requirements-dev.txt)"
	@echo "check          ruff + offline test suite"
	@echo "golden         replay the committed golden fixtures"
	@echo "validate-a     validate artifacts/a/predictions_a.parquet"
	@echo "validate-b     validate artifacts/b/predictions_b.parquet"
	@echo "demo           rebuild the committed offline demo run"

install:
	$(PY) -m pip install -r requirements-dev.txt
	$(PY) -m pip install -e .

lint:
	ruff check .
	ruff format --check .

test:
	pytest -q -m "not network"

check: lint test

golden:
	pytest -q tests/test_golden_replay.py

contract:
	pytest -q tests/test_schema.py tests/test_tiers.py tests/test_model_output_files.py

fusion-smoke:
	pytest -q tests/test_fusion.py

app-imports:
	pytest -q tests/test_app_imports.py

ownership:
	$(PY) scripts/check_ownership.py

validate-a:
	$(PY) -c "import pandas as pd; from pathlib import Path; from floodwatch.schema import validate_model_file; g=pd.read_parquet('data/processed/grid_index.parquet'); print(validate_model_file('artifacts/a/predictions_a.parquet','A_cloudburst_cnn_lstm',g).shape)"

validate-b:
	$(PY) -c "import pandas as pd; from pathlib import Path; from floodwatch.schema import validate_model_file; g=pd.read_parquet('data/processed/grid_index.parquet'); print(validate_model_file('artifacts/b/predictions_b.parquet','B_snowmelt_lstm',g).shape)"

demo:
	$(PY) scripts/00_build_demo.py --force
	$(PY) scripts/04_train_b.py --quick
	$(PY) scripts/05_fit_thresholds.py
	$(PY) scripts/06_run.py --run-id demo --offline

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
