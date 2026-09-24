# Single shared image for all five agents -- they need the identical
# dependency set and the full src/ tree, differing only in which module
# docker-compose's `command:` runs. Five near-identical Dockerfiles would
# just be the same layers duplicated five times.
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

COPY src/ src/

ENV PYTHONUNBUFFERED=1
