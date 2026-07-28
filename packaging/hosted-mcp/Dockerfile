# syntax=docker/dockerfile:1

FROM python:3.12-slim

ARG OPENHCS_VERSION

LABEL org.opencontainers.image.title="OpenHCS hosted MCP"
LABEL org.opencontainers.image.description="Read-only Streamable HTTP MCP service for the OpenHCS ChatGPT plugin"
LABEL org.opencontainers.image.source="https://github.com/OpenHCSDev/OpenHCS"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OPENHCS_CPU_ONLY=true
ENV OPENHCS_MCP_HTTP_AUTH_MODE=public_read_only
ENV OPENHCS_MCP_HTTP_BIND_HOST=0.0.0.0
ENV OPENHCS_MCP_HTTP_BIND_PORT=8000
ENV OPENHCS_AGENT_READ_ROOTS=/usr/local/lib/python3.12/site-packages/openhcs:/opt/openhcs-data
ENV OPENHCS_AGENT_WRITE_ROOTS=/opt/openhcs-data

RUN test -n "${OPENHCS_VERSION}" \
    && python -m pip install --no-cache-dir \
        "openhcs[mcp-http]==${OPENHCS_VERSION}" \
    && useradd --create-home --uid 10001 openhcs \
    && install -d -o openhcs -g openhcs /opt/openhcs-data

USER openhcs
WORKDIR /opt/openhcs-data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).read()"]

ENTRYPOINT ["openhcs-mcp-http"]
