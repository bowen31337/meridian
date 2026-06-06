"""Unit coverage for the entry-point discovery resolution branches and the
plugins.yml source-level failure path that the loader-level tests stub out."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

from core_errors import NoopAuditLog
from meridian_plugin_loader import PluginLoader
from meridian_plugin_loader._discovery import discover_from_entry_points


class _EP:
    def __init__(self, obj: Any) -> None:
        self._obj = obj

    def load(self) -> Any:
        return self._obj


class _ModelLike:
    def model_dump(self) -> dict[str, Any]:
        return {"name": "from-model", "kind": "tool"}


def test_discover_entry_points_resolves_all_shapes() -> None:
    eps = [
        _EP({"name": "plain-dict", "kind": "tool"}),
        _EP(_ModelLike()),
        _EP(lambda: {"name": "from-callable", "kind": "tool"}),
        _EP(lambda: _ModelLike()),
        _EP("unsupported-string"),  # skipped: not dict, no model_dump, not callable
    ]
    with patch("meridian_plugin_loader._discovery.entry_points", return_value=eps):
        results = discover_from_entry_points()

    names = {r["name"] for r in results}
    assert names == {"plain-dict", "from-model", "from-callable"}
    assert len(results) == 4  # two from-model entries collapse by value but stay distinct items


def test_yml_discovery_failure_surfaces_error() -> None:
    loader = PluginLoader(audit_log=NoopAuditLog())
    with (
        patch("meridian_plugin_loader._loader.discover_from_entry_points", return_value=[]),
        patch(
            "meridian_plugin_loader._loader.discover_from_yml",
            side_effect=RuntimeError("yml boom"),
        ),
    ):
        result = loader.load_all()

    assert any(e.code == "yml_discovery_failed" for e in result.errors)
    assert any("yml boom" in e.message for e in result.errors)
