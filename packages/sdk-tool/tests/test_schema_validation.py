"""Unit tests for JSON Schema validation (Architecture §11, PRD F-SB-3)."""

from __future__ import annotations

import pytest

from meridian_sdk_tool._schema import SchemaValidationError, validate_input, validate_output

_STR_SCHEMA: dict = {"type": "string"}
_OBJ_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "count": {"type": "integer", "minimum": 0},
    },
    "required": ["name"],
}


class TestValidateInput:
    def test_valid_string(self) -> None:
        validate_input(_STR_SCHEMA, "hello")  # should not raise

    def test_valid_object(self) -> None:
        validate_input(_OBJ_SCHEMA, {"name": "foo", "count": 3})

    def test_valid_object_optional_field_absent(self) -> None:
        validate_input(_OBJ_SCHEMA, {"name": "bar"})

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_input(_STR_SCHEMA, 42)
        assert "Input validation failed" in str(exc_info.value)
        assert exc_info.value.errors

    def test_missing_required_field_raises(self) -> None:
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_input(_OBJ_SCHEMA, {"count": 5})
        assert exc_info.value.errors

    def test_minimum_violation_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            validate_input(_OBJ_SCHEMA, {"name": "x", "count": -1})

    def test_empty_schema_accepts_anything(self) -> None:
        validate_input({}, {"whatever": True})

    def test_multiple_errors_collected(self) -> None:
        schema: dict = {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        }
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_input(schema, {})
        # Both 'a' and 'b' are missing
        assert len(exc_info.value.errors) >= 2


class TestValidateOutput:
    def test_valid_output(self) -> None:
        validate_output(_OBJ_SCHEMA, {"name": "result"})

    def test_invalid_output_raises(self) -> None:
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_output(_OBJ_SCHEMA, "not an object")
        assert "Output validation failed" in str(exc_info.value)


class TestFieldPaths:
    """Errors must be prefixed with their JSON Path ($.field[0].key)."""

    def test_root_level_error_prefixed_with_dollar(self) -> None:
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_input(_STR_SCHEMA, 42)
        assert exc_info.value.errors[0].startswith("$:")

    def test_nested_field_path_included(self) -> None:
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_input(_OBJ_SCHEMA, {"name": "x", "count": -1})
        paths = [e.split(":")[0] for e in exc_info.value.errors]
        assert "$.count" in paths

    def test_missing_required_field_at_root(self) -> None:
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_input(_OBJ_SCHEMA, {})
        # Missing required shows at root ($)
        paths = [e.split(":")[0] for e in exc_info.value.errors]
        assert any(p == "$" for p in paths)

    def test_array_index_in_path(self) -> None:
        schema: dict = {
            "type": "array",
            "items": {"type": "integer"},
        }
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_input(schema, [1, "two", 3])
        paths = [e.split(":")[0] for e in exc_info.value.errors]
        assert "$[1]" in paths

    def test_output_errors_also_include_paths(self) -> None:
        with pytest.raises(SchemaValidationError) as exc_info:
            validate_output(_OBJ_SCHEMA, {"name": "x", "count": -1})
        paths = [e.split(":")[0] for e in exc_info.value.errors]
        assert "$.count" in paths
