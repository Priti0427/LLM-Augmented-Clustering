from __future__ import annotations

import json
import os
import re
import time
from hashlib import sha256
from typing import Any, Iterable

from openai import OpenAI

from .cache import OpenAIStageCache


_FENCE_RE = re.compile(r"^```(?:json|JSON)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _parse_json_text(text: str) -> Any:
    """Parse JSON from a model response, tolerating markdown code fences.

    GPT-4o-mini occasionally wraps JSON in ```json ... ``` even when asked
    for compact JSON, so strip the fence before json.loads.
    """
    stripped = text.strip()
    fence_match = _FENCE_RE.match(stripped)
    if fence_match:
        stripped = fence_match.group(1).strip()
    return json.loads(stripped)


def _coerce_object(parsed: Any) -> dict[str, Any]:
    """Coerce a model JSON response to a single object.

    Models occasionally return a single-element or multi-element list when we
    asked for one object. We unwrap to the first dict so downstream callers
    that do `obj.get("label")` keep working.
    """
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                return item
    raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}: {parsed!r}")


class OpenAIUnavailableError(RuntimeError):
    pass


class OpenAIService:
    def __init__(
        self,
        cache: OpenAIStageCache | None = None,
        api_key: str | None = None,
        embedding_model: str = "text-embedding-3-large",
        llm_model: str = "gpt-4o-mini",
        max_retries: int = 3,
    ):
        self.cache = cache or OpenAIStageCache()
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.max_retries = max_retries
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None

    @property
    def is_configured(self) -> bool:
        return self.client is not None

    def _require_client(self) -> OpenAI:
        if self.client is None:
            raise OpenAIUnavailableError(
                "OPENAI_API_KEY is not configured. Add the key to compute methods C and D."
            )
        return self.client

    def _cache_key(self, namespace: str, payload: Any) -> str:
        digest = sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
        return f"{namespace}:{digest}"

    def _with_retry(self, func):
        attempt = 0
        while True:
            try:
                return func()
            except Exception:
                attempt += 1
                if attempt >= self.max_retries:
                    raise
                time.sleep(1.5 * attempt)

    def embed_texts(self, texts: Iterable[str], namespace: str) -> list[list[float]]:
        client = self._require_client()
        outputs: list[list[float]] = []
        missing_texts: list[str] = []
        missing_keys: list[str] = []

        for text in texts:
            key = self._cache_key(namespace, {"text": text, "model": self.embedding_model})
            cached = self.cache.get("embed", key)
            if cached is None:
                missing_keys.append(key)
                missing_texts.append(text)
            else:
                outputs.append(cached)

        if missing_texts:
            batch_size = 64
            fresh_vectors: dict[str, list[float]] = {}
            for start in range(0, len(missing_texts), batch_size):
                chunk = missing_texts[start : start + batch_size]
                key_chunk = missing_keys[start : start + batch_size]
                response = self._with_retry(
                    lambda: client.embeddings.create(model=self.embedding_model, input=chunk)
                )
                for key, item in zip(key_chunk, response.data, strict=True):
                    fresh_vectors[key] = item.embedding
                    self.cache.set("embed", key, item.embedding)

        ordered_vectors: list[list[float]] = []
        for text in texts:
            key = self._cache_key(namespace, {"text": text, "model": self.embedding_model})
            vector = self.cache.get("embed", key)
            if vector is None:
                raise RuntimeError(f"Embedding cache miss after API call for key {key}.")
            ordered_vectors.append(vector)
        return ordered_vectors

    def classify_issue(self, text: str, namespace: str) -> dict[str, Any]:
        return self._json_response(
            stage="filter",
            namespace=namespace,
            payload={"text": text, "task": "issue_classification"},
            system_prompt=(
                "You classify customer support tickets. Reply with compact JSON containing "
                "`is_issue` (boolean) and `reason` (string). An issue is a problem, defect, "
                "billing/shipping/account problem, or service failure. General information "
                "requests with no problem are not issues."
            ),
            user_prompt=text,
        )

    def extract_issue(self, text: str, namespace: str) -> dict[str, Any]:
        return self._json_response(
            stage="extract",
            namespace=namespace,
            payload={"text": text, "task": "issue_extraction"},
            system_prompt=(
                "You normalize customer support tickets into a concise issue statement. "
                "Reply with compact JSON containing `issue_statement` and `confidence`. "
                "The issue statement must be 3 to 15 words, imperative-free, and capture "
                "the underlying problem rather than conversational filler."
            ),
            user_prompt=text,
        )

    def name_cluster(self, issue_statements: list[str], namespace: str) -> dict[str, Any]:
        joined = "\n".join(f"- {issue}" for issue in issue_statements)
        return self._json_response(
            stage="name",
            namespace=namespace,
            payload={"issues": issue_statements, "task": "cluster_naming"},
            system_prompt=(
                "You name a single customer-support ticket cluster. Reply with one "
                "compact JSON object (not a list) with exactly two string keys: "
                "`label` (2 to 6 words, actionable, name the underlying problem, "
                "no quotes) and `summary` (one short sentence). Do not wrap the "
                "object in an array."
            ),
            user_prompt=joined,
        )

    def _json_response(
        self,
        stage: str,
        namespace: str,
        payload: dict[str, Any],
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        client = self._require_client()
        key = self._cache_key(namespace, {"stage": stage, "model": self.llm_model, "payload": payload})
        cached = self.cache.get(stage, key)
        if cached is not None:
            # Heal historical cache entries that were stored as a list before
            # _coerce_object was added.
            try:
                return _coerce_object(cached)
            except ValueError:
                # Drop the malformed entry and fall through to a fresh API call.
                pass

        response = self._with_retry(
            lambda: client.responses.create(
                model=self.llm_model,
                instructions=system_prompt,
                input=user_prompt,
                max_output_tokens=250,
            )
        )
        raw_text = getattr(response, "output_text", "").strip()
        if not raw_text:
            raise RuntimeError(f"OpenAI returned an empty response for stage `{stage}`.")
        try:
            parsed = _parse_json_text(raw_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"OpenAI returned non-JSON output for stage `{stage}`: {raw_text}") from exc
        try:
            parsed = _coerce_object(parsed)
        except ValueError as exc:
            raise RuntimeError(f"OpenAI returned unexpected JSON shape for stage `{stage}`: {raw_text}") from exc
        self.cache.set(stage, key, parsed)
        return parsed
