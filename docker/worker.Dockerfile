# syntax=docker/dockerfile:1.7
# Worker: scheduled scraper (Playwright/Chromium) + CPU inference.
# The playwright pip version in requirements/scraper.txt MUST match this tag.
FROM mcr.microsoft.com/playwright/python:v1.56.0-noble@sha256:a7f6cf3ae520c9d670ad956572c13747ed5abdbba5123a01526f873ed1662528

ARG WITH_TORCH=1
ARG TORCH_VERSION=2.14.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HOME=/tmp

WORKDIR /app
COPY requirements/ requirements/
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    pip install -r requirements/web.txt -r requirements/scraper.txt \
 && if [ "$WITH_TORCH" = "1" ]; then \
      pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" && pip install -r requirements/ml.txt; \
    fi

COPY pyproject.toml README.md ./
COPY tl ./tl
RUN pip install --no-deps . && rm -rf /app/tl /app/build \
 && mkdir -p /data /models /secrets && chown pwuser:pwuser /data /models /secrets && chmod 700 /secrets

# pwuser is uid 1001 in the base image (same uid as the web/train "app" user).
USER pwuser
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 CMD ["tl", "worker-health"]
CMD ["tl", "worker"]
