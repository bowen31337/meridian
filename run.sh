#!/usr/bin/env bash
# run.sh — meridian dev helper
# All commands are injected with secrets from 1Password at runtime.
# Usage: ./run.sh <command> [args...]
#
#   ./run.sh forge   [args]     claw-forge with any args (default: no args)
#   ./run.sh state              start claw-forge state service (port 8420)
#   ./run.sh agent  "desc"      run claw-forge with a feature description
#   ./run.sh shell              drop into a shell with secrets injected

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_TPL="${SCRIPT_DIR}/.env.tpl"

# ── Guards ────────────────────────────────────────────────────────────────────

if ! command -v op &>/dev/null; then
  echo "error: 1Password CLI (op) not found. Install via: brew install 1password-cli" >&2
  exit 1
fi

if [[ ! -f "${ENV_TPL}" ]]; then
  echo "error: ${ENV_TPL} not found." >&2
  exit 1
fi

if [[ -f "${SCRIPT_DIR}/.env" ]]; then
  echo "warning: plaintext .env file detected — remove it: rm ${SCRIPT_DIR}/.env" >&2
fi

# ── Wrapper ───────────────────────────────────────────────────────────────────

OP="op run --env-file=${ENV_TPL} --"

cmd="${1:-help}"
shift || true

case "${cmd}" in

  forge)
    exec ${OP} claw-forge "$@"
    ;;

  state)
    echo "→ Starting claw-forge state service on port 8420..."
    exec ${OP} claw-forge state "$@"
    ;;

  agent)
    if [[ $# -eq 0 ]]; then
      echo "usage: ./run.sh agent \"feature description\"" >&2
      exit 1
    fi
    echo "→ Running agent: $*"
    exec ${OP} claw-forge run "$@"
    ;;

  shell)
    echo "→ Dropping into shell with secrets injected (type 'exit' to leave)..."
    exec ${OP} bash
    ;;

  help|--help|-h)
    grep '^#' "${BASH_SOURCE[0]}" | grep -E '^\s*#\s+\.' | sed 's/^#//'
    ;;

  *)
    echo "error: unknown command '${cmd}'. Run ./run.sh help for usage." >&2
    exit 1
    ;;

esac
