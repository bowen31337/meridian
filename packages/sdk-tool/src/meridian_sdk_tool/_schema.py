from __future__ import annotations

from typing import Any

import jsonschema
import jsonschema.exceptions


class SchemaValidationError(Exception):
    """Raised when JSON Schema validation fails; carries the raw validation errors."""

    def __init__(self, message: str, errors: list[str] | None = None) -> None:
        super().__init__(message)
        self.errors: list[str] = errors or []


def _fmt_path(absolute_path: Any) -> str:
    """Format a jsonschema absolute_path deque as a JSON Path string (e.g. $.field[0].key)."""
    parts = ["$"]
    for key in absolute_path:
        if isinstance(key, int):
            parts.append(f"[{key}]")
        else:
            parts.append(f".{key}")
    return "".join(parts)


def _collect_errors(schema: dict[str, Any], data: Any) -> list[str]:
    validator = jsonschema.Draft7Validator(schema)
    errors = []
    for e in sorted(validator.iter_errors(data), key=str):
        path = _fmt_path(e.absolute_path)
        errors.append(f"{path}: {e.message}")
    return errors


def validate_input(schema: dict[str, Any], data: Any) -> None:
    """Validate *data* against *schema* before dispatching the tool.

    Raises SchemaValidationError so the caller can surface is_error=true to
    the model (Architecture §11.4 — schema failure never crashes the harness).
    """
    errors = _collect_errors(schema, data)
    if errors:
        raise SchemaValidationError(
            f"Input validation failed ({len(errors)} error(s)): {errors[0]}",
            errors=errors,
        )


def validate_output(schema: dict[str, Any], data: Any) -> None:
    """Validate *data* against *schema* after the tool returns.

    Same surface behaviour as validate_input — is_error=true to the model,
    never an orchestrator crash.
    """
    errors = _collect_errors(schema, data)
    if errors:
        raise SchemaValidationError(
            f"Output validation failed ({len(errors)} error(s)): {errors[0]}",
            errors=errors,
        )
