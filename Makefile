.PHONY: help install data run quick diagrams test lint clean

PYTHON ?= python

help:
	@echo "make install    install the package and dev tools"
	@echo "make data       download the MegaFon dataset to data/ (~270 MB)"
	@echo "make run        full pipeline: figures/ and results/ (~15 min on 2 CPU cores)"
	@echo "make quick      smoke run on 60,000 rows with a smaller grid (~1 min)"
	@echo "make diagrams   redraw figures/diagrams/"
	@echo "make test       run the unit tests (synthetic data, no download)"
	@echo "make lint       ruff checks"

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) scripts/download_data.py

run:
	$(PYTHON) -m uplift_telecom.pipeline

quick:
	$(PYTHON) -m uplift_telecom.pipeline --sample 60000 --fast --root build/quick

diagrams:
	$(PYTHON) scripts/make_diagrams.py

test:
	$(PYTHON) -m pytest

lint:
	ruff check src tests scripts

clean:
	rm -rf build .pytest_cache .ruff_cache src/*.egg-info
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
