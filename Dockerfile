# syntax=docker/dockerfile:1
# Model-serving image. Build AFTER training (needs models/model.joblib):  make train && make docker-build
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HEARTRISK_HOME=/app \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/model.joblib \
    LOG_LEVEL=INFO

WORKDIR /app

# Dependencies first -> cached unless requirements-serve.txt changes.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

COPY src ./src
COPY models/model.joblib models/metadata.json ./models/

# Run as an unprivileged user.
RUN useradd --system --uid 10001 --no-create-home appuser
USER 10001

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "heartrisk.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
