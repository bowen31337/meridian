#!/usr/bin/env bash
# Runs ruff format check and lint. On failure:
#   1. Surfaces a structured error message to stderr.
#   2. Appends an NDJSON audit entry to ${AUDIT_LOG_PATH:-ci-audit.ndjson}.
set -uo pipefail

_AUDIT_LOG="${AUDIT_LOG_PATH:-ci-audit.ndjson}"
_TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
_FAILED=0

uv run ruff format --check . || _FAILED=1
uv run ruff check . || _FAILED=1

if [ "$_FAILED" -eq 0 ]; then
    exit 0
fi

echo "" >&2
echo "ERROR: ruff check failed." >&2
echo "Run 'ruff format . && ruff check --fix .' to auto-fix, then commit." >&2
echo "" >&2

printf '%s\n' \
    "{\"level\":\"error\",\"event\":\"ci.ruff.failed\",\"timestamp\":\"${_TIMESTAMP}\",\"detail\":{\"message\":\"ruff format or lint check failed; see CI logs for per-file errors\"}}" \
    >> "${_AUDIT_LOG}"

exit 1
