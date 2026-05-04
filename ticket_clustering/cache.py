from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import OPENAI_CACHE_DIR, RESULTS_DIR
from .models import MethodResult


class ResultStore:
    def __init__(self, results_dir: Path | None = None):
        self.results_dir = results_dir or RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, dataset_hash: str) -> Path:
        return self.results_dir / f"{dataset_hash}.json"

    def load(self, dataset_hash: str) -> dict[str, MethodResult]:
        path = self.path_for(dataset_hash)
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {
            method_id: MethodResult.from_dict(method_payload)
            for method_id, method_payload in payload.get("methods", {}).items()
        }

    def save(self, dataset_hash: str, methods: dict[str, MethodResult], metadata: dict[str, Any] | None = None) -> Path:
        path = self.path_for(dataset_hash)
        payload = {
            "dataset_hash": dataset_hash,
            "metadata": metadata or {},
            "methods": {
                method_id: method_result.to_dict()
                for method_id, method_result in methods.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return path


class OpenAIStageCache:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or OPENAI_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, stage: str) -> Path:
        return self.cache_dir / f"{stage}.json"

    def get(self, stage: str, key: str) -> Any | None:
        path = self._path_for(stage)
        if not path.exists():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload.get(key)

    def set(self, stage: str, key: str, value: Any) -> None:
        path = self._path_for(stage)
        payload: dict[str, Any] = {}
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        payload[key] = value
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
