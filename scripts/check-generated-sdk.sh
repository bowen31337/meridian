#!/usr/bin/env bash
# L6 — Reject manual edits to generated SDK files. These files are owned by the
# codegen pipeline — edit the templates/spec instead, not the output.
set -euo pipefail

if [ "$#" -eq 0 ]; then
    exit 0
fi

echo "ERROR [L6]: Manual edits to generated SDK files are not allowed." >&2
echo "" >&2
echo "The following staged files are owned by the codegen pipeline" >&2
echo "and must not be edited directly:" >&2
for file in "$@"; do
    echo "  - $file" >&2
done
echo "" >&2
echo "To update the generated SDKs, trigger the codegen pipeline instead:" >&2
echo "  make codegen           # regenerate locally" >&2
echo "  # or push to trigger the 'codegen' CI job in GitHub Actions" >&2
echo "" >&2
echo "See docs/codegen.md for the codegen pipeline documentation." >&2
echo "" >&2
echo "Rule L6: packages/sdk-ts/src/** and packages/sdk-py/src/** are codegen-owned." >&2

exit 1
