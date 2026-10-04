# syntax=docker/dockerfile:1.7
# Opt-in training image (`docker compose --profile train ...`). CPU by default; for GPU build with
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124  (see docker-compose.gpu.yml)
FROM python:3.11.13-slim-bookworm@sha256:86adf8dbadc3d6e82ee5dd2c74bec2e1c2467cdad47886280501df722372d2e1

ARG TORCH_VERSION=2.14.1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/hf \
    HOME=/tmp

RUN groupadd -g 1001 app && useradd -u 1001 -g app -M -d /nonexistent -s /usr/sbin/nologin app \
 && mkdir -p /data /models /hf && chown app:app /data /models /hf

WORKDIR /app
COPY requirements/ requirements/
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}" \
 && pip install -r requirements/web.txt -r requirements/ml.txt

COPY pyproject.toml README.md ./
COPY tl ./tl
RUN pip install --no-deps . && rm -rf /app/tl /app/build

USER app
ENTRYPOINT ["tl"]
CMD ["train"]
