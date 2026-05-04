from __future__ import annotations

from pathlib import Path

from ticket_clustering.cache import ResultStore
from ticket_clustering.models import MethodResult


def test_result_store_round_trip(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    result = MethodResult(
        method_id="A",
        display_name="Online Clustering",
        status="computed",
        metrics={"cluster_count": 2},
    )
    store.save("abc123", {"A": result})
    loaded = store.load("abc123")

    assert loaded["A"].method_id == "A"
    assert loaded["A"].metrics["cluster_count"] == 2
