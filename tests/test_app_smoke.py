from __future__ import annotations

import importlib.util


def test_app_file_exists() -> None:
    spec = importlib.util.spec_from_file_location("app", "app.py")
    assert spec is not None
