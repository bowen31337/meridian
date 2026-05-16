"""Shared fixtures for the meridiand conformance suite."""
from __future__ import annotations

import pytest


@pytest.fixture()
def storage_root(tmp_path):
    root = tmp_path / "storage"
    root.mkdir()
    return root
