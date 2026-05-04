from __future__ import annotations

from ticket_clustering.data import load_dataset_file
from ticket_clustering.pipeline import PipelineRunner


def main() -> None:
    dataset = load_dataset_file("test_dataset_500_zendesk.json")
    runner = PipelineRunner(dataset)
    results = runner.load_or_run(force=True)
    print(f"Cached {len(results)} methods for {dataset.source_name} ({dataset.dataset_hash})")
    for method_id, result in results.items():
        print(f"- {method_id}: {result.status} ({result.artifact_origin})")


if __name__ == "__main__":
    main()
