# Security Policy

## Supported Versions

AAanalysis follows semantic versioning. Security fixes are applied to the
latest `1.1.x` release line only; please upgrade to the newest `1.1.x` before
reporting.

| Version | Supported          |
| ------- | ------------------ |
| 1.1.x   | :white_check_mark: |
| < 1.1   | :x:                |

## Reporting a Vulnerability

Please report suspected vulnerabilities **privately** — do not open a public
issue or pull request for a security problem.

- Preferred: open a private report through GitHub Security Advisories at
  <https://github.com/breimanntools/aaanalysis/security/advisories/new>.
- Alternatively, email the maintainer at <stephanbreimann@gmail.com> with
  "AAanalysis security" in the subject.

Please include the affected version, a description of the issue, and (if
possible) a minimal reproduction. We aim to acknowledge a report within a few
working days and will coordinate a fix and disclosure timeline with you.

## Releases

AAanalysis publishes to PyPI through a single canonical, automated release
workflow that uses PyPI OIDC trusted publishing (no long-lived API tokens).
Install only from the official [PyPI project](https://pypi.org/project/aaanalysis/).
