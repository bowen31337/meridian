#!/usr/bin/env bash
# Runs Pyright in strict mode. On failure:
#   1. Surfaces a structured error message to stderr.
#   2. Appends an NDJSON audit entry to ${AUDIT_LOG_PATH:-ci-audit.ndjson}.
set -uo pipefail

_AUDIT_LOG="${AUDIT_LOG_PATH:-ci-audit.ndjson}"
_TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

if pyright; then
    exit 0
fi

echo "" >&2
echo "ERROR: Pyright strict mode check failed." >&2
echo "Every public symbol must have a complete type annotation." >&2
echo "Fix the errors above and re-run: pyright" >&2
echo "" >&2

printf '%s\n' \
    "{\"level\":\"error\",\"event\":\"ci.pyright.failed\",\"timestamp\":\"${_TIMESTAMP}\",\"detail\":{\"message\":\"Pyright strict mode check failed; see CI logs for per-file errors\"}}" \
    >> "${_AUDIT_LOG}"

exit 1
