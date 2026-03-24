.PHONY: all install run debug clean lint

all: install run

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "call_me_maybe.egg-info -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache .venv
	rm -rf data/output

lint:
	uv run flake8 src/
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs