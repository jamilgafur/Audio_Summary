# Security Policy

## Supported versions

The latest release tag. Older tags are supported on a best-effort basis while
the default branch remains buildable.

## Reporting a vulnerability

**Do not open a public issue for security problems.**

- Open a private security advisory on this repository: GitHub → *Security* →
  *Triaging a new vulnerability report* → *Report a vulnerability* (requires
  the maintainer to have enabled security advisories — if that button is
  missing, email the maintainer directly instead).
- Include: what you found, how you found it, a minimal reproduction if you
  have one, and any impact on stored data.

Response targets: acknowledgement within 7 days, a fix or a documented
mitigation within 30 days for confirmed high-severity issues.

## Security posture — read this before exposing the app

Paper Reader is a **local/LAN tool by design**. There is no authentication,
and it is expected to run on a machine or network you trust. Two deliberate
boundaries:

1. **`POST /api/upload-url` fetches arbitrary URLs** (arXiv, PDF links) so the
   UI can import papers by link. This is an SSRF surface: anyone with network
   access to the app can make it fetch internal URLs. If you expose the app
   beyond a trusted LAN, put IP allowlisting (or a proxy with auth) in front
   of it first.
2. **The web UI trusts the browser.** Anyone who can reach the port can
   upload, delete, and render papers and read their contents. Keep the port
   private; the compose file binds it for local use.

Other notes:

- User content (papers, scripts, renders) lives only in your mounted
  `input/`, `output/`, and `models/` directories.
- Dependency vulnerabilities: CI runs `pip-audit` on every push/PR and
  Dependabot checks `requirements*.txt` and the Actions workflows weekly.
- The Docker build compiles `llama-cpp-python` from a version range
  (`>=0.2.90`) pinned in the Dockerfile rather than `requirements.txt`; audit
  that range when reviewing dependency updates.
