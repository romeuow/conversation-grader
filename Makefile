.PHONY: install dev test lint format up down run eval serve

install:
	uv sync --no-dev

dev:
	uv sync

test:
	uv run pytest -q --cov=src --cov-report=term-missing

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

run:
	uv run grader run examples/conversations.jsonl --rubric sales_v1 --fake

eval:
	uv run grader eval --fake

serve:
	uv run grader serve --port 8000

up:
	docker compose up --build -d

down:
	docker compose down
