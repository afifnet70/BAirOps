# ---- stage 1: build wheels (build tools stay out of the final image) ----------------------------
FROM python:3.13-slim AS builder
WORKDIR /build
COPY requirements-api.txt .
RUN pip install --upgrade pip && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements-api.txt

# ---- stage 2: slim runtime --------------------------------------------------------------------
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user
COPY --from=builder /wheels /wheels
COPY requirements-api.txt /tmp/requirements-api.txt
RUN pip install --no-index --find-links=/wheels -r /tmp/requirements-api.txt && rm -rf /wheels
USER user
ENV HOME=/home/user PYTHONPATH=/home/user/app:/home/user/app/src
WORKDIR /home/user/app
COPY --chown=user params.yaml ./
COPY --chown=user src ./src
COPY --chown=user app ./app
EXPOSE 7860
HEALTHCHECK --interval=60s --timeout=10s --start-period=180s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health', timeout=8)" || exit 1
# The production model is downloaded from the MLflow registry at start-up (see app/service.py)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
