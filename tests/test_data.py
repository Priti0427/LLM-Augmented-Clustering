from __future__ import annotations

import json
from pathlib import Path

import pytest

from ticket_clustering.data import DatasetValidationError, build_analysis_text, build_dataset, compute_dataset_hash


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "tiny_zendesk.json"


def test_build_dataset_parses_fixture() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    dataset = build_dataset(payload, source_name="tiny.json")

    assert dataset.stats["ticket_count"] == 3
    assert dataset.tickets[0].ticket_id == "t1"
    assert "wrong size shoes" in dataset.tickets[0].analysis_text.lower()


def test_build_analysis_text_uses_customer_messages_only() -> None:
    ticket = {
        "subject": "Login problem",
        "description": "Cannot sign in",
        "messages": [
            {"role": "agent", "content": "Hello there"},
            {"role": "customer", "content": "I still cannot access my account"},
        ],
    }
    text = build_analysis_text(ticket)
    assert "Hello there" not in text
    assert "access my account" in text


def test_invalid_dataset_raises_validation_error() -> None:
    payload = {"tickets": [{"language": "en"}]}
    with pytest.raises(DatasetValidationError):
        build_dataset(payload)


def test_dataset_hash_is_stable() -> None:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert compute_dataset_hash(payload) == compute_dataset_hash(payload)
