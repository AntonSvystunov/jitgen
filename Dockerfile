FROM python:3.13-slim AS builder

ENV PYTHONUNBUFFERED=1
RUN pip install uv

WORKDIR /app
COPY ./pyproject.toml ./uv.lock ./


RUN uv sync --no-dev --no-editable --no-python-downloads --group evaluation


FROM python:3.13-slim-bookworm AS runtime

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app"

COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

WORKDIR /app

COPY ./jitgen ./jitgen/
COPY ./evaluation ./evaluation/

CMD ["python", "evaluation/run.py"]