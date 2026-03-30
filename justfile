default:
    @just --list

install:
    uv sync

lint:
    uv run ruff check predictor/ tests/ API/
    uv run ruff format --check predictor/ tests/ API/

format:
    uv run ruff check --fix predictor/ tests/ API/
    uv run ruff format predictor/ tests/ API/

test:
    uv run pytest

typecheck:
    uv run ty check predictor/

check: lint typecheck test
