from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_TITLE = "Ticket Clustering Comparison Studio"
BUNDLED_DATASET_PATH = ROOT_DIR / "test_dataset_500_zendesk.json"
CACHE_DIR = ROOT_DIR / "data_cache"
RESULTS_DIR = CACHE_DIR / "results"
OPENAI_CACHE_DIR = CACHE_DIR / "openai_cache"

DEFAULT_METHOD_ORDER = ["A", "B", "C", "D"]

METHOD_DEFINITIONS = {
    "A": {
        "name": "Online Clustering",
        "description": "Incremental centroid assignment on TF-IDF vectors.",
    },
    "B": {
        "name": "K-Means + TF-IDF",
        "description": "Elbow-selected K-Means over sparse lexical features.",
    },
    "C": {
        "name": "OpenAI Embeddings + UMAP + HDBSCAN",
        "description": "Dense semantic clustering on raw ticket text.",
    },
    "D": {
        "name": "LLM + OpenAI Embeddings + UMAP + HDBSCAN",
        "description": "Issue filtering and extraction before dense clustering.",
    },
}

ONLINE_CLUSTER_THRESHOLD = 0.35
KMEANS_K_MIN = 5
KMEANS_K_MAX = 60
UMAP_COMPONENTS = 10
UMAP_NEIGHBORS = 15
UMAP_METRIC = "cosine"
HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_MIN_SAMPLES = 2
HDBSCAN_CLUSTER_EPSILON = 0.3

BUNDLED_REFERENCE_METRICS = {
    "A": {
        "silhouette": None,
        "cluster_count": 200,
        "noise_pct": 0.0,
        "coherence": 0.21,
        "actionability": 0.15,
    },
    "B": {
        "silhouette": 0.08,
        "cluster_count": 55,
        "noise_pct": 0.0,
        "coherence": 0.35,
        "actionability": 0.28,
    },
    "C": {
        "silhouette": 0.34,
        "cluster_count": 38,
        "noise_pct": 18.0,
        "coherence": 0.61,
        "actionability": 0.55,
    },
    "D": {
        "silhouette": 0.52,
        "cluster_count": 42,
        "noise_pct": 7.0,
        "coherence": 0.78,
        "actionability": 0.82,
    },
}

ISSUE_KEYWORDS = [
    "wrong",
    "refund",
    "return",
    "size",
    "item",
    "shipping",
    "delivery",
    "delivered",
    "tracking",
    "account",
    "login",
    "password",
    "subscription",
    "plan",
    "billing",
    "charge",
    "payment",
    "invoice",
    "cancel",
    "damaged",
    "missing",
]
