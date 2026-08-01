# Security Policy

## Supported versions

Security fixes are applied to the current `0.7.x` beta line. Older beta lines
may be asked to upgrade before a fix can be evaluated.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Email
[tristan.simas@mail.mcgill.ca](mailto:tristan.simas@mail.mcgill.ca) with the
affected version, impact, minimal reproduction, and any suggested mitigation.
Please avoid sending real patient data, credentials, access tokens, or
proprietary microscopy datasets. You should receive an acknowledgment within
five business days.

## MCP and desktop trust boundary

The supported MCP server is local and uses explicit read and write roots. Its
tools can inspect data, author pipelines, start local execution, and write
outputs within the configured capabilities. Review a compiled plan before
execution, grant the smallest useful roots, and apply the privacy and retention
policy of the agent client and model provider you choose. OpenHCS does not run a
public hosted MCP endpoint.

The current Windows and macOS beta installers are not code-signed or notarized.
Download them only from the official GitHub Release links and verify that the
resolved repository owner is `OpenHCSDev`.
