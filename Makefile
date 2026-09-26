.PHONY: setup lint format test data train causal auction report all

setup:
	uv sync --all-groups

lint:
	uv run ruff check src tests
	uv run black --check src tests

format:
	uv run black src tests
	uv run ruff check --fix src tests

test:
	uv run pytest

data:
	uv run python -m mua.cli generate
	uv run python -m mua.cli features

train:
	uv run python -m mua.cli train

causal:
	uv run python -m mua.cli causal

auction:
	uv run python -m mua.cli auction

report:
	uv run python -m mua.cli report

all: setup lint test data train causal auction report
