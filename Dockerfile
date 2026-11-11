FROM python:3.11-slim

ARG EXTRAS=api
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 FINRAG_CONFIG=configs/docker.yaml

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN pip install ".[${EXTRAS}]"

# data/ (index and metadata built with `python -m finrag ingest`) is mounted at run time
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["uvicorn", "finrag.api:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8080"]
