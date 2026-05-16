from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import yaml

ENTRY_POINT_GROUP = "meridian.plugins"
DEFAULT_PLUGINS_YML = Path.home() / ".meridian" / "plugins.yml"


def discover_from_entry_points() -> list[dict[str, Any]]:
    """Load raw plugin dicts from all installed meridian.plugins entry points.

    Each entry point value must resolve to one of:
    - a ``dict`` representing a PluginManifest
    - a ``PluginManifest`` instance (has ``model_dump()``)
    - a callable returning either of the above

    Entry points that resolve to an unsupported type are silently skipped;
    the caller is responsible for validating the returned dicts.
    """
    eps = entry_points(group=ENTRY_POINT_GROUP)
    results: list[dict[str, Any]] = []
    for ep in eps:
        obj = ep.load()
        if callable(obj) and not hasattr(obj, "model_dump"):
            obj = obj()
        if isinstance(obj, dict):
            results.append(obj)
        elif hasattr(obj, "model_dump"):
            results.append(obj.model_dump())
    return results


def discover_from_yml(path: Path) -> list[dict[str, Any]]:
    """Load raw plugin dicts from a plugins.yml file.

    The file must be a YAML mapping with a top-level ``plugins`` list, e.g.::

        plugins:
          - name: my-tool
            kind: tool
            sandbox_mode: out_of_process
            entry_point: my_pkg:my_tool

    Returns an empty list when the file does not exist.
    """
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        return []
    plugins = raw.get("plugins", [])
    if not isinstance(plugins, list):
        return []
    return [p for p in plugins if isinstance(p, dict)]
