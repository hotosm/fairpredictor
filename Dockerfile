FROM python:3.12-slim-bookworm AS builder

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get --no-install-recommends -y install \
    build-essential libgdal-dev libboost-numpy-dev \
    libagg-dev libpotrace-dev pkg-config potrace

WORKDIR /build

COPY pyproject.toml poetry.lock* ./

RUN pip install --upgrade pip \
    && pip install poetry \
    && poetry config virtualenvs.create false

FROM python:3.12-slim-bookworm

RUN apt-get update \
    && apt-get -y upgrade \
    && apt-get --no-install-recommends -y install \
    libgdal30 libpotrace0 potrace \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/pyproject.toml ./
COPY --from=builder /build/poetry.lock* ./

RUN pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install --no-dev

COPY predictor ./predictor
COPY README.md ./
COPY API/main.py ./

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]