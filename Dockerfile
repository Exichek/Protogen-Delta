FROM python:3.14-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=2.4.1

WORKDIR /build

RUN pip install "poetry==${POETRY_VERSION}"

RUN python -m venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml poetry.lock ./

RUN poetry install --only main --no-root --no-interaction --no-ansi

FROM builder AS package

COPY README.md ./
COPY src ./src

RUN poetry build --format wheel


FROM python:3.14-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-rus \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd --system protogen \
    && useradd --system \
        --gid protogen \
        --create-home \
        --home-dir /home/protogen \
        protogen \
    && mkdir -p /app/data /home/protogen/.cache/huggingface \
    && chown -R protogen:protogen /app /home/protogen/.cache

COPY --from=builder /opt/venv /opt/venv
COPY --from=package /build/dist/*.whl /tmp/delta-wheel/
RUN pip install --no-deps /tmp/delta-wheel/*.whl \
    && rm -rf /tmp/delta-wheel

USER protogen

CMD ["python", "-m", "protogen_delta.main"]
