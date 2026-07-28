# OpenHCS hosted MCP container

This image runs the registry-derived, read-only OpenHCS MCP surface for the
ChatGPT web plugin. It installs an exact released OpenHCS version from PyPI and
does not copy a development checkout into the image.

Build it after the corresponding OpenHCS release is public:

```bash
docker build \
  --build-arg OPENHCS_VERSION=0.6.5 \
  --tag openhcs-mcp:0.6.5 \
  packaging/hosted-mcp
```

The production service must terminate TLS in front of the container and set:

```text
OPENHCS_MCP_HTTP_PUBLIC_URL=https://mcp.example.org/mcp
OPENHCS_MCP_HTTP_ALLOWED_HOSTS=mcp.example.org
```

During OpenAI domain verification, also set the exact token supplied by the
plugin submission portal:

```text
OPENHCS_MCP_HTTP_OPENAI_DOMAIN_CHALLENGE_TOKEN=<portal-token>
```

The token is exposed verbatim at
`/.well-known/openai-apps-challenge`. Remove or rotate it when the portal
instructs you to do so.

The default container mode is `public_read_only`. At server construction,
OpenHCS rejects either authentication mode if the hosted capability registry
contains a mutating tool. Authentication changes who may read the surface, not
its mutation boundary. The service exposes packaged documentation, architecture
projection, function discovery, and configuration-schema reflection. It cannot
access a visitor's computer, local OpenHCS installation, microscopy images,
GUI, viewers, or execution processes.

For a private deployment, set
`OPENHCS_MCP_HTTP_AUTH_MODE=oauth_introspection` and supply the OAuth
environment contract documented in the OpenHCS MCP release guide. The
authenticated mode remains subject-isolated and is not the universal public
plugin configuration.

The unauthenticated `/healthz` route contains only service status, transport,
and authentication mode. The MCP endpoint remains at the path in
`OPENHCS_MCP_HTTP_PUBLIC_URL`, normally `/mcp`.
