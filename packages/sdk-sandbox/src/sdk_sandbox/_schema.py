from __future__ import annotations

from typing import Any

import jsonschema
import jsonschema.exceptions


class OutputSchemaError(Exception):
    """Raised when post-dispatch output JSON Schema validation fails."""

    def __init__(self, message: str, errors: list[str]) -> None:
        super().__init__(message)
        self.errors: list[str] = errors


def _fmt_path(absolute_path: Any) -> str:
    """Format a jsonschema absolute_path deque as a JSON Path string (e.g. $.field[0].key)."""
    parts = ["$"]
    for key in absolute_path:
        if isinstance(key, int):
            parts.append(f"[{key}]")
        else:
            parts.append(f".{key}")
    return "".join(parts)


def validate_output(schema: dict[str, Any], data: Any) -> None:
    """Validate *data* against *schema* after the dispatcher returns.

    Raises OutputSchemaError with per-error strings formatted as
    "<json-path>: <message>" so callers can surface the offending field path.
    Never raises other exceptions — jsonschema failures are always wrapped.
    """
    validator = jsonschema.Draft7Validator(schema)
    errors: list[str] = []
    for e in sorted(validator.iter_errors(data), key=str):
        path = _fmt_path(e.absolute_path)
        errors.append(f"{path}: {e.message}")
    if errors:
        raise OutputSchemaError(
            f"Output validation failed ({len(errors)} error(s)): {errors[0]}",
            errors=errors,
        )
