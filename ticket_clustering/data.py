from __future__ import annotations

import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from statistics import mean
from typing import Any

from .models import DatasetBundle, TicketMessage, TicketRecord


class DatasetValidationError(ValueError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


def load_dataset_file(path: str | Path) -> DatasetBundle:
    dataset_path = Path(path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    return build_dataset(payload, source_name=dataset_path.name)


def build_dataset(payload: dict[str, Any], source_name: str = "uploaded.json") -> DatasetBundle:
    errors = validate_dataset_payload(payload)
    if errors:
        raise DatasetValidationError(errors)

    tickets = [normalize_ticket(ticket_payload) for ticket_payload in payload["tickets"]]
    dataset_hash = compute_dataset_hash(payload)
    stats = build_dataset_stats(tickets, payload.get("metadata", {}))
    return DatasetBundle(
        source_name=source_name,
        dataset_hash=dataset_hash,
        raw_metadata=payload.get("metadata", {}),
        tickets=tickets,
        stats=stats,
    )


def validate_dataset_payload(payload: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["Top-level JSON must be an object."]

    tickets = payload.get("tickets")
    if not isinstance(tickets, list) or not tickets:
        errors.append("`tickets` must be a non-empty list.")
        return errors

    for index, ticket in enumerate(tickets):
        prefix = f"tickets[{index}]"
        if not isinstance(ticket, dict):
            errors.append(f"{prefix} must be an object.")
            continue

        if not ticket.get("id"):
            errors.append(f"{prefix}.id is required.")
        if not ticket.get("language"):
            errors.append(f"{prefix}.language is required.")
        subject = (ticket.get("subject") or "").strip()
        description = (ticket.get("description") or "").strip()
        if not subject and not description:
            errors.append(f"{prefix} must include at least one of `subject` or `description`.")

        messages = ticket.get("messages", [])
        if messages is not None and not isinstance(messages, list):
            errors.append(f"{prefix}.messages must be a list when present.")
            continue
        for message_index, message in enumerate(messages or []):
            if not isinstance(message, dict):
                errors.append(f"{prefix}.messages[{message_index}] must be an object.")
                continue
            if not message.get("content"):
                errors.append(f"{prefix}.messages[{message_index}].content is required.")
            if not message.get("role"):
                errors.append(f"{prefix}.messages[{message_index}].role is required.")

    return errors


def normalize_ticket(ticket_payload: dict[str, Any]) -> TicketRecord:
    messages = [
        TicketMessage(
            role=str(message.get("role", "")).strip(),
            content=str(message.get("content", "")).strip(),
            timestamp=message.get("timestamp"),
        )
        for message in (ticket_payload.get("messages") or [])
        if str(message.get("content", "")).strip()
    ]
    analysis_text = build_analysis_text(ticket_payload, messages)
    customer = ticket_payload.get("customer") or {}
    return TicketRecord(
        ticket_id=str(ticket_payload["id"]),
        language=str(ticket_payload["language"]),
        subject=str(ticket_payload.get("subject", "")).strip(),
        description=str(ticket_payload.get("description", "")).strip(),
        created_at=ticket_payload.get("created_at"),
        updated_at=ticket_payload.get("updated_at"),
        status=ticket_payload.get("status"),
        priority=ticket_payload.get("priority"),
        customer_name=customer.get("name"),
        customer_email=customer.get("email"),
        messages=messages,
        analysis_text=analysis_text,
    )


def build_analysis_text(ticket_payload: dict[str, Any], messages: list[TicketMessage] | None = None) -> str:
    message_list = messages if messages is not None else [
        TicketMessage(
            role=str(message.get("role", "")).strip(),
            content=str(message.get("content", "")).strip(),
            timestamp=message.get("timestamp"),
        )
        for message in (ticket_payload.get("messages") or [])
    ]
    parts = [
        str(ticket_payload.get("subject", "")).strip(),
        str(ticket_payload.get("description", "")).strip(),
    ]
    parts.extend(message.content for message in message_list if message.role.lower() == "customer")
    return " ".join(part for part in parts if part).strip()


def compute_dataset_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_dataset_stats(tickets: list[TicketRecord], metadata: dict[str, Any]) -> dict[str, Any]:
    languages = Counter(ticket.language for ticket in tickets)
    statuses = Counter(ticket.status or "unknown" for ticket in tickets)
    priorities = Counter(ticket.priority or "unknown" for ticket in tickets)
    message_counts = [len(ticket.messages) for ticket in tickets]
    message_word_counts: list[int] = []
    customer_word_counts: list[int] = []

    for ticket in tickets:
        for message in ticket.messages:
            count = len(message.content.split())
            message_word_counts.append(count)
            if message.role.lower() == "customer":
                customer_word_counts.append(count)

    return {
        "ticket_count": len(tickets),
        "languages": dict(sorted(languages.items())),
        "statuses": dict(sorted(statuses.items())),
        "priorities": dict(sorted(priorities.items())),
        "messages_total": sum(message_counts),
        "messages_avg": round(mean(message_counts), 2),
        "messages_min": min(message_counts),
        "messages_max": max(message_counts),
        "words_per_message_avg": round(mean(message_word_counts), 2) if message_word_counts else 0.0,
        "customer_words_avg": round(mean(customer_word_counts), 2) if customer_word_counts else 0.0,
        "date_range": metadata.get("date_range"),
        "source": metadata.get("source"),
    }
