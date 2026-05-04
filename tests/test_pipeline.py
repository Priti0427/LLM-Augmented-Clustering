from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ticket_clustering.cache import ResultStore
from ticket_clustering.data import build_dataset
from ticket_clustering.pipeline import DenseClusterOutput, PipelineRunner


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tiny_zendesk.json"


class StubOpenAIService:
    def embed_texts(self, texts, namespace):
        return [[float(index + 1), float(len(text.split()))] for index, text in enumerate(texts)]

    def classify_issue(self, text, namespace):
        return {"is_issue": True, "reason": "stubbed"}

    def extract_issue(self, text, namespace):
        words = text.split()[:5]
        return {"issue_statement": " ".join(words), "confidence": 0.9}

    def name_cluster(self, issue_statements, namespace):
        return {"label": "Stubbed cluster", "summary": "stub summary"}


def stub_dense_cluster(features):
    labels = [0 if row[0] <= 2 else 1 for row in features]
    projection = [[row[0], row[1]] for row in features]
    return DenseClusterOutput(
        labels=np.array(labels),
        reduced_features=np.array(projection),
        projection_2d=np.array(projection),
    )


def load_fixture_dataset():
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return build_dataset(payload, source_name="tiny_zendesk.json")


def test_online_clustering_assigns_clusters(tmp_path: Path) -> None:
    dataset = load_fixture_dataset()
    runner = PipelineRunner(dataset, result_store=ResultStore(tmp_path), openai_service=StubOpenAIService())
    result = runner.run_method("A")

    assert result.method_id == "A"
    assert result.metrics["cluster_count"] >= 1
    assert len(result.assignments) == 3


def test_pipeline_load_or_run_returns_all_methods(tmp_path: Path) -> None:
    dataset = load_fixture_dataset()
    runner = PipelineRunner(
        dataset,
        result_store=ResultStore(tmp_path),
        openai_service=StubOpenAIService(),
        dense_cluster_fn=stub_dense_cluster,
    )
    results = runner.load_or_run(force=True)

    assert set(results.keys()) == {"A", "B", "C", "D"}
    assert results["C"].status == "computed"
    assert results["D"].ticket_artifacts["t1"]["issue_statement"]
