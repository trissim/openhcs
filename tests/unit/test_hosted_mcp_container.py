"""Static acceptance gates for the public hosted MCP container."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "packaging" / "hosted-mcp" / "Dockerfile"
README = REPO_ROOT / "packaging" / "hosted-mcp" / "README.md"


def test_hosted_container_pins_release_and_runs_unprivileged():
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG OPENHCS_VERSION" in dockerfile
    assert '"openhcs[mcp-http]==${OPENHCS_VERSION}"' in dockerfile
    assert "USER openhcs" in dockerfile
    assert "OPENHCS_MCP_HTTP_AUTH_MODE=public_read_only" in dockerfile
    assert 'ENTRYPOINT ["openhcs-mcp-http"]' in dockerfile


def test_hosted_container_documents_external_security_boundary():
    readme = README.read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())

    assert "terminate TLS" in readme
    assert "OPENHCS_MCP_HTTP_PUBLIC_URL" in readme
    assert "OPENHCS_MCP_HTTP_ALLOWED_HOSTS" in readme
    assert "OPENHCS_MCP_HTTP_OPENAI_DOMAIN_CHALLENGE_TOKEN" in readme
    assert "cannot access a visitor's computer" in normalized_readme
