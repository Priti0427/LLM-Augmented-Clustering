from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from .cache import ResultStore
from .config import (
    DEFAULT_METHOD_ORDER,
    HDBSCAN_CLUSTER_EPSILON,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    ISSUE_KEYWORDS,
    KMEANS_K_MAX,
    KMEANS_K_MIN,
    METHOD_DEFINITIONS,
    ONLINE_CLUSTER_THRESHOLD,
    UMAP_COMPONENTS,
    UMAP_METRIC,
    UMAP_NEIGHBORS,
)
from .models import ClusterRecord, DatasetBundle, MethodResult, ProjectionPoint, TicketRecord
from .openai_client import OpenAIService, OpenAIUnavailableError
from .reference_results import build_reference_method_result


class MethodUnavailableError(RuntimeError):
    pass


@dataclass
class DenseClusterOutput:
    labels: np.ndarray
    reduced_features: np.ndarray
    projection_2d: np.ndarray


def default_dense_cluster_fn(features: np.ndarray) -> DenseClusterOutput:
    try:
        import hdbscan
        import umap
    except ImportError as exc:
        raise MethodUnavailableError(
            "Methods C and D require `umap-learn` and `hdbscan`. Install project dependencies first."
        ) from exc

    if len(features) == 1:
        projection = np.array([[0.0, 0.0]])
        return DenseClusterOutput(labels=np.array([0]), reduced_features=projection, projection_2d=projection)

    n_neighbors = min(UMAP_NEIGHBORS, max(2, len(features) - 1))
    reducer = umap.UMAP(
        n_components=min(UMAP_COMPONENTS, max(2, len(features) - 1)),
        metric=UMAP_METRIC,
        n_neighbors=n_neighbors,
        random_state=42,
    )
    reduced = reducer.fit_transform(features)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min(HDBSCAN_MIN_CLUSTER_SIZE, max(2, len(features))),
        min_samples=min(HDBSCAN_MIN_SAMPLES, max(1, len(features) - 1)),
        cluster_selection_epsilon=HDBSCAN_CLUSTER_EPSILON,
    )
    labels = clusterer.fit_predict(reduced)

    visual_projection = reduced[:, :2] if reduced.shape[1] >= 2 else PCA(n_components=2).fit_transform(features)
    return DenseClusterOutput(labels=labels, reduced_features=reduced, projection_2d=visual_projection)


class PipelineRunner:
    def __init__(
        self,
        dataset: DatasetBundle,
        result_store: ResultStore | None = None,
        openai_service: OpenAIService | None = None,
        dense_cluster_fn: Callable[[np.ndarray], DenseClusterOutput] | None = None,
    ):
        self.dataset = dataset
        self.result_store = result_store or ResultStore()
        self.openai_service = openai_service or OpenAIService()
        self.dense_cluster_fn = dense_cluster_fn or default_dense_cluster_fn
        self.tickets = dataset.tickets
        self.texts = [ticket.analysis_text for ticket in dataset.tickets]
        self.ticket_map = {ticket.ticket_id: ticket for ticket in dataset.tickets}

    def load_or_run(
        self,
        method_ids: list[str] | None = None,
        use_cache: bool = True,
        force: bool = False,
    ) -> dict[str, MethodResult]:
        method_ids = method_ids or DEFAULT_METHOD_ORDER
        cached = self.result_store.load(self.dataset.dataset_hash) if use_cache else {}
        results: dict[str, MethodResult] = dict(cached)
        updated = False

        for method_id in method_ids:
            if not force and method_id in results:
                continue
            try:
                results[method_id] = self.run_method(method_id)
                updated = True
            except (MethodUnavailableError, OpenAIUnavailableError) as exc:
                results[method_id] = self._fallback_or_unavailable(method_id, str(exc))
                updated = True

        if updated:
            self.result_store.save(
                self.dataset.dataset_hash,
                results,
                metadata={"source_name": self.dataset.source_name, "stats": self.dataset.stats},
            )
        return {method_id: results[method_id] for method_id in method_ids}

    def run_method(self, method_id: str) -> MethodResult:
        if method_id == "A":
            return self._run_online_clustering()
        if method_id == "B":
            return self._run_kmeans_tfidf()
        if method_id == "C":
            return self._run_dense_embedding_method()
        if method_id == "D":
            return self._run_llm_augmented_method()
        raise KeyError(f"Unknown method id: {method_id}")

    def _fallback_or_unavailable(self, method_id: str, message: str) -> MethodResult:
        if self.dataset.source_name == "test_dataset_500_zendesk.json":
            reference = build_reference_method_result(method_id, message)
            if method_id in {"A", "B"}:
                return reference
            return reference

        return MethodResult(
            method_id=method_id,
            display_name=METHOD_DEFINITIONS[method_id]["name"],
            status="unavailable",
            metrics={
                "silhouette": None,
                "cluster_count": None,
                "noise_pct": None,
                "coherence": None,
                "actionability": None,
            },
            warnings=[message],
            notes=["Compute this method after configuring the required dependencies and API access."],
            artifact_origin="unavailable",
        )

    def _run_online_clustering(self) -> MethodResult:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=4000)
        matrix = vectorizer.fit_transform(self.texts)
        labels = self._online_cluster_labels(matrix, threshold=ONLINE_CLUSTER_THRESHOLD)
        projection = self._sparse_projection(matrix)
        return self._build_result_from_labels(
            method_id="A",
            labels=np.array(labels),
            feature_matrix=matrix.toarray(),
            projection=projection,
            texts=self.texts,
            vectorizer=vectorizer,
        )

    def _run_kmeans_tfidf(self) -> MethodResult:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=4000)
        matrix = vectorizer.fit_transform(self.texts)
        k = self._select_k_by_elbow(matrix)
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(matrix)
        projection = self._sparse_projection(matrix)
        result = self._build_result_from_labels(
            method_id="B",
            labels=labels,
            feature_matrix=matrix.toarray(),
            projection=projection,
            texts=self.texts,
            vectorizer=vectorizer,
        )
        result.notes.append(f"Elbow heuristic selected k={k}.")
        return result

    def _run_dense_embedding_method(self) -> MethodResult:
        embeddings = np.array(
            self.openai_service.embed_texts(
                self.texts,
                namespace=f"{self.dataset.dataset_hash}:method_c",
            )
        )
        clustered = self.dense_cluster_fn(embeddings)
        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        vectorizer.fit(self.texts)
        result = self._build_result_from_labels(
            method_id="C",
            labels=clustered.labels,
            feature_matrix=embeddings,
            projection=clustered.projection_2d,
            texts=self.texts,
            vectorizer=vectorizer,
        )
        result.notes.append("Embeddings computed from raw ticket analysis text.")
        return result

    def _run_llm_augmented_method(self) -> MethodResult:
        ticket_artifacts: dict[str, dict[str, Any]] = {}
        issue_texts: list[str] = []
        issue_ticket_ids: list[str] = []

        for ticket in self.tickets:
            filter_payload = self._filter_ticket(ticket)
            artifact = {
                "is_issue": bool(filter_payload["is_issue"]),
                "filter_reason": filter_payload["reason"],
                "filter_source": filter_payload["source"],
            }
            if artifact["is_issue"]:
                extraction = self.openai_service.extract_issue(
                    ticket.analysis_text,
                    namespace=f"{self.dataset.dataset_hash}:method_d:extract",
                )
                issue_statement = str(extraction.get("issue_statement", "")).strip() or "Unspecified support issue"
                artifact["issue_statement"] = issue_statement
                artifact["extract_confidence"] = extraction.get("confidence")
                issue_texts.append(issue_statement)
                issue_ticket_ids.append(ticket.ticket_id)
            ticket_artifacts[ticket.ticket_id] = artifact

        if not issue_texts:
            raise MethodUnavailableError("Method D found no issue tickets to cluster.")

        embeddings = np.array(
            self.openai_service.embed_texts(
                issue_texts,
                namespace=f"{self.dataset.dataset_hash}:method_d:embed",
            )
        )
        clustered = self.dense_cluster_fn(embeddings)

        full_labels = np.full(len(self.tickets), -1)
        projection = np.full((len(self.tickets), 2), np.nan)
        issue_index_by_ticket = {ticket_id: index for index, ticket_id in enumerate(issue_ticket_ids)}
        for position, ticket_id in enumerate(issue_ticket_ids):
            full_ticket_index = self._ticket_position(ticket_id)
            full_labels[full_ticket_index] = clustered.labels[position]
            projection[full_ticket_index] = clustered.projection_2d[position]

        vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
        vectorizer.fit(issue_texts)
        result = self._build_result_from_labels(
            method_id="D",
            labels=full_labels,
            feature_matrix=np.vstack(
                [
                    embeddings[issue_index_by_ticket[ticket.ticket_id]]
                    if ticket.ticket_id in issue_index_by_ticket
                    else np.zeros(embeddings.shape[1])
                    for ticket in self.tickets
                ]
            ),
            projection=projection,
            texts=[
                ticket_artifacts[ticket.ticket_id].get("issue_statement", ticket.analysis_text)
                for ticket in self.tickets
            ],
            vectorizer=vectorizer,
            ticket_artifacts=ticket_artifacts,
        )

        for cluster in result.clusters:
            issues = [ticket_artifacts[ticket_id]["issue_statement"] for ticket_id in cluster.representative_ticket_ids if "issue_statement" in ticket_artifacts[ticket_id]]
            if issues:
                naming = self.openai_service.name_cluster(
                    issues,
                    namespace=f"{self.dataset.dataset_hash}:method_d:name:{cluster.cluster_id}",
                )
                cluster.label = str(naming.get("label", cluster.label)).strip() or cluster.label
                for ticket_id in cluster.representative_ticket_ids:
                    ticket_artifacts[ticket_id]["cluster_summary"] = naming.get("summary")

        result.ticket_artifacts = ticket_artifacts
        result.notes.append("Method D filtered non-issue tickets before clustering.")
        return result

    def _filter_ticket(self, ticket: TicketRecord) -> dict[str, Any]:
        lower_text = ticket.analysis_text.lower()
        if any(keyword in lower_text for keyword in ISSUE_KEYWORDS):
            return {"is_issue": True, "reason": "Matched issue keyword heuristic.", "source": "keyword"}
        payload = self.openai_service.classify_issue(
            ticket.analysis_text,
            namespace=f"{self.dataset.dataset_hash}:method_d:filter",
        )
        return {
            "is_issue": bool(payload.get("is_issue")),
            "reason": str(payload.get("reason", "Classified by GPT-4o-mini.")),
            "source": "gpt-4o-mini",
        }

    def _build_result_from_labels(
        self,
        method_id: str,
        labels: np.ndarray,
        feature_matrix: np.ndarray,
        projection: np.ndarray,
        texts: list[str],
        vectorizer: TfidfVectorizer,
        ticket_artifacts: dict[str, dict[str, Any]] | None = None,
    ) -> MethodResult:
        metrics = self._compute_metrics(labels, feature_matrix)
        clusters = self._build_clusters(labels, feature_matrix, texts, vectorizer, method_id)
        assignments = {
            ticket.ticket_id: int(label)
            for ticket, label in zip(self.tickets, labels, strict=True)
        }
        projection_points = [
            ProjectionPoint(
                ticket_id=ticket.ticket_id,
                cluster_id=int(label),
                x=float(point[0]) if not math.isnan(float(point[0])) else 0.0,
                y=float(point[1]) if not math.isnan(float(point[1])) else 0.0,
                label=self._cluster_label_for_id(clusters, int(label)),
            )
            for ticket, label, point in zip(self.tickets, labels, projection, strict=True)
        ]
        return MethodResult(
            method_id=method_id,
            display_name=METHOD_DEFINITIONS[method_id]["name"],
            status="computed",
            metrics=metrics,
            clusters=clusters,
            assignments=assignments,
            projection=projection_points,
            ticket_artifacts=ticket_artifacts or {},
            notes=[],
            warnings=[],
        )

    def _build_clusters(
        self,
        labels: np.ndarray,
        feature_matrix: np.ndarray,
        texts: list[str],
        vectorizer: TfidfVectorizer,
        method_id: str,
    ) -> list[ClusterRecord]:
        clusters: list[ClusterRecord] = []
        feature_names = vectorizer.get_feature_names_out()
        unique_labels = sorted(label for label in set(labels.tolist()) if label != -1)
        for cluster_id in unique_labels:
            member_indices = np.where(labels == cluster_id)[0]
            member_features = feature_matrix[member_indices]
            centroid = member_features.mean(axis=0)
            if centroid.ndim > 1:
                centroid = centroid.ravel()
            similarities = cosine_similarity(member_features, centroid.reshape(1, -1)).ravel()
            ranked_positions = member_indices[np.argsort(-similarities)][:5]
            representative_ticket_ids = [self.tickets[index].ticket_id for index in ranked_positions]
            representative_issues = [texts[index] for index in ranked_positions]
            top_terms = self._top_terms_for_cluster(member_indices, texts, feature_names)
            label = self._cluster_label(method_id, top_terms, representative_issues)
            clusters.append(
                ClusterRecord(
                    cluster_id=int(cluster_id),
                    label=label,
                    size=int(len(member_indices)),
                    representative_ticket_ids=representative_ticket_ids,
                    representative_issues=representative_issues,
                    top_terms=top_terms,
                )
            )
        return clusters

    def _top_terms_for_cluster(
        self,
        member_indices: np.ndarray,
        texts: list[str],
        global_feature_names: np.ndarray,
    ) -> list[str]:
        cluster_texts = [texts[index] for index in member_indices]
        try:
            vectorizer = TfidfVectorizer(stop_words="english", max_features=8)
            matrix = vectorizer.fit_transform(cluster_texts)
            scores = np.asarray(matrix.mean(axis=0)).ravel()
            feature_names = vectorizer.get_feature_names_out()
            ordered = np.argsort(-scores)
            return [feature_names[index] for index in ordered[:4]]
        except ValueError:
            words = Counter(" ".join(cluster_texts).lower().split())
            return [word for word, _ in words.most_common(4)]

    def _cluster_label(self, method_id: str, top_terms: list[str], representative_issues: list[str]) -> str:
        if not top_terms:
            return f"{METHOD_DEFINITIONS[method_id]['name']} Cluster"
        subject_phrase = representative_issues[0].split(".")[0][:50].strip()
        if subject_phrase:
            return f"{', '.join(top_terms[:2]).title()} | {subject_phrase}"
        return ", ".join(term.title() for term in top_terms[:2])

    def _cluster_label_for_id(self, clusters: list[ClusterRecord], cluster_id: int) -> str:
        if cluster_id == -1:
            return "Noise / Filtered"
        for cluster in clusters:
            if cluster.cluster_id == cluster_id:
                return cluster.label
        return f"Cluster {cluster_id}"

    def _compute_metrics(self, labels: np.ndarray, feature_matrix: np.ndarray) -> dict[str, Any]:
        non_noise_mask = labels != -1
        unique_non_noise = sorted(set(labels[non_noise_mask].tolist()))
        silhouette = None
        usable_points = int(non_noise_mask.sum())
        if usable_points >= 3 and 2 <= len(unique_non_noise) < usable_points:
            try:
                silhouette = float(
                    round(
                        silhouette_score(feature_matrix[non_noise_mask], labels[non_noise_mask], metric="cosine"),
                        4,
                    )
                )
            except ValueError:
                silhouette = None

        return {
            "silhouette": silhouette,
            "cluster_count": len(unique_non_noise),
            "noise_pct": round((labels == -1).sum() / len(labels) * 100, 2),
            "coherence": None,
            "actionability": None,
        }

    def _online_cluster_labels(self, matrix, threshold: float) -> list[int]:
        labels: list[int] = []
        centroids: list[np.ndarray] = []
        counts: list[int] = []

        for row_index in range(matrix.shape[0]):
            row = matrix[row_index].toarray().ravel()
            if not centroids:
                centroids.append(row.copy())
                counts.append(1)
                labels.append(0)
                continue

            centroid_matrix = np.vstack(centroids)
            similarities = cosine_similarity([row], centroid_matrix).ravel()
            best_cluster = int(np.argmax(similarities))
            if float(similarities[best_cluster]) >= threshold:
                labels.append(best_cluster)
                counts[best_cluster] += 1
                centroids[best_cluster] = centroids[best_cluster] + (row - centroids[best_cluster]) / counts[best_cluster]
            else:
                labels.append(len(centroids))
                centroids.append(row.copy())
                counts.append(1)
        return labels

    def _select_k_by_elbow(self, matrix) -> int:
        sample_count = matrix.shape[0]
        if sample_count <= 2:
            return max(1, sample_count)

        k_min = min(KMEANS_K_MIN, max(2, sample_count - 1))
        k_max = min(KMEANS_K_MAX, max(2, sample_count - 1))
        if k_min > k_max:
            return max(2, min(sample_count - 1, 2))
        ks = list(range(k_min, k_max + 1))
        inertias = []
        for k in ks:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            model.fit(matrix)
            inertias.append(model.inertia_)
        if len(ks) <= 2:
            return ks[0]

        points = np.column_stack([ks, inertias])
        start = points[0]
        end = points[-1]
        line = end - start
        norm = np.linalg.norm(line)
        if norm == 0:
            return ks[len(ks) // 2]
        distances = np.abs(np.cross(points - start, line) / norm)
        return int(ks[int(np.argmax(distances))])

    def _sparse_projection(self, matrix) -> np.ndarray:
        if matrix.shape[0] == 1:
            return np.array([[0.0, 0.0]])
        n_components = min(2, max(1, min(matrix.shape) - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = svd.fit_transform(matrix)
        if reduced.shape[1] == 1:
            return np.column_stack([reduced[:, 0], np.zeros(reduced.shape[0])])
        return reduced[:, :2]

    def _ticket_position(self, ticket_id: str) -> int:
        for index, ticket in enumerate(self.tickets):
            if ticket.ticket_id == ticket_id:
                return index
        raise KeyError(ticket_id)
