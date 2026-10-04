# syntax=docker/dockerfile:1.7
# Web UI: FastAPI + templates. No browser, no torch, no secrets volume.
FROM python:3.11.13-slim-bookworm@sha256:86adf8dbadc3d6e82ee5dd2c74bec2e1c2467cdad47886280501df722372d2e1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# uid/gid 1001 is shared by web, worker (pwuser) and train so they can all write the data volume.
RUN groupadd -g 1001 app && useradd -u 1001 -g app -M -d /nonexistent -s /usr/sbin/nologin app \
 && mkdir -p /data /models /secrets && chown app:app /data /models /secrets && chmod 700 /secrets

WORKDIR /app
COPY requirements/web.txt requirements/web.txt
# Optional extra CA (only for building behind a TLS-intercepting proxy): --secret id=extra_ca,src=ca.crt
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then export PIP_CERT=/run/secrets/extra_ca; fi; \
    pip install -r requirements/web.txt

COPY pyproject.toml README.md ./
COPY tl ./tl
RUN pip install --no-deps . && rm -rf /app/tl /app/build

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"]
CMD ["tl", "serve", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
