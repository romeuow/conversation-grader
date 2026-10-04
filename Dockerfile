# syntax=docker/dockerfile:1.7
# ---- build stage: resolve and install dependencies with uv ---------------------------
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ---- runtime stage: no uv, non-root user, only the virtualenv --------------------------
FROM python:3.12-slim AS runtime

RUN groupadd --system app && useradd --system --gid app --home /app app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app rubrics ./rubrics
COPY --chown=app:app examples ./examples
COPY --chown=app:app evals ./evals

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    RUBRICS_DIR=/app/rubrics

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status == 200 else 1)"

CMD ["grader", "serve", "--host", "0.0.0.0", "--port", "8000"]
