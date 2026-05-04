from __future__ import annotations

from .config import BUNDLED_REFERENCE_METRICS, METHOD_DEFINITIONS
from .models import MethodResult


def build_reference_method_result(method_id: str, reason: str) -> MethodResult:
    return MethodResult(
        method_id=method_id,
        display_name=METHOD_DEFINITIONS[method_id]["name"],
        status="reference_only",
        metrics=dict(BUNDLED_REFERENCE_METRICS[method_id]),
        warnings=[reason],
        notes=[
            "Reference metrics come from the poster/abstract and are shown until cached OpenAI artifacts are generated.",
        ],
        artifact_origin="poster_reference",
    )
