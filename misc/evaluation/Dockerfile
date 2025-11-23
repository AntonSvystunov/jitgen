FROM python:3.13-slim AS builder

ENV PYTHONUNBUFFERED=1
RUN pip install uv

WORKDIR /app

# Copy jitgen libraries
COPY libs/ ./libs/

# Copy evaluation package sources and pyproject.toml
COPY misc/evaluation/ ./misc/evaluation/

# Sync evaluation package to create .venv
WORKDIR /app/misc/evaluation
RUN uv sync


FROM python:3.13-slim AS runtime

ENV VIRTUAL_ENV=/app/misc/evaluation/.venv \
    PATH="/app/misc/evaluation/.venv/bin:$PATH"

# Copy venv from builder
COPY --from=builder /app/misc/evaluation/.venv /app/misc/evaluation/.venv

WORKDIR /app/misc/evaluation

# Run jit-eval
CMD ["uv", "run", "jit-eval"]