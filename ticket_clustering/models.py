from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TicketMessage:
    role: str
    content: str
    timestamp: str | None = None


@dataclass
class TicketRecord:
    ticket_id: str
    language: str
    subject: str
    description: str
    created_at: str | None
    updated_at: str | None
    status: str | None
    priority: str | None
    customer_name: str | None
    customer_email: str | None
    messages: list[TicketMessage]
    analysis_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetBundle:
    source_name: str
    dataset_hash: str
    raw_metadata: dict[str, Any]
    tickets: list[TicketRecord]
    stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "dataset_hash": self.dataset_hash,
            "raw_metadata": self.raw_metadata,
            "tickets": [ticket.to_dict() for ticket in self.tickets],
            "stats": self.stats,
        }


@dataclass
class ClusterRecord:
    cluster_id: int
    label: str
    size: int
    representative_ticket_ids: list[str]
    representative_issues: list[str]
    top_terms: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectionPoint:
    ticket_id: str
    cluster_id: int
    x: float
    y: float
    label: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MethodResult:
    method_id: str
    display_name: str
    status: str
    metrics: dict[str, Any]
    clusters: list[ClusterRecord] = field(default_factory=list)
    assignments: dict[str, int] = field(default_factory=dict)
    projection: list[ProjectionPoint] = field(default_factory=list)
    ticket_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    artifact_origin: str = "computed"

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "display_name": self.display_name,
            "status": self.status,
            "metrics": self.metrics,
            "clusters": [cluster.to_dict() for cluster in self.clusters],
            "assignments": self.assignments,
            "projection": [point.to_dict() for point in self.projection],
            "ticket_artifacts": self.ticket_artifacts,
            "warnings": self.warnings,
            "notes": self.notes,
            "artifact_origin": self.artifact_origin,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MethodResult":
        return cls(
            method_id=payload["method_id"],
            display_name=payload["display_name"],
            status=payload["status"],
            metrics=payload.get("metrics", {}),
            clusters=[ClusterRecord(**cluster) for cluster in payload.get("clusters", [])],
            assignments=payload.get("assignments", {}),
            projection=[ProjectionPoint(**point) for point in payload.get("projection", [])],
            ticket_artifacts=payload.get("ticket_artifacts", {}),
            warnings=payload.get("warnings", []),
            notes=payload.get("notes", []),
            artifact_origin=payload.get("artifact_origin", "computed"),
        )
