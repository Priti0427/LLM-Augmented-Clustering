from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ticket_clustering.cache import ResultStore
from ticket_clustering.config import APP_TITLE, BUNDLED_DATASET_PATH, DEFAULT_METHOD_ORDER, METHOD_DEFINITIONS
from ticket_clustering.data import DatasetValidationError, build_dataset, load_dataset_file
from ticket_clustering.openai_client import OpenAIService
from ticket_clustering.pipeline import PipelineRunner


st.set_page_config(page_title=APP_TITLE, page_icon=":bar_chart:", layout="wide")


def load_active_dataset():
    uploaded = st.sidebar.file_uploader("Upload Zendesk-style JSON", type=["json"])
    if uploaded is not None:
        try:
            payload = json.loads(uploaded.getvalue().decode("utf-8"))
            dataset = build_dataset(payload, source_name=uploaded.name)
            return dataset, False
        except json.JSONDecodeError as exc:
            st.sidebar.error(f"Invalid JSON: {exc}")
            return None, False
        except DatasetValidationError as exc:
            st.sidebar.error("Dataset validation failed:")
            for error in exc.errors:
                st.sidebar.write(f"- {error}")
            return None, False
    return load_dataset_file(BUNDLED_DATASET_PATH), True


def load_results(dataset, is_bundled: bool, force: bool = False):
    runner = PipelineRunner(
        dataset=dataset,
        result_store=ResultStore(),
        openai_service=OpenAIService(),
    )
    return runner.load_or_run(DEFAULT_METHOD_ORDER, use_cache=True, force=force)


def build_metric_table(results: dict):
    rows = []
    for method_id in DEFAULT_METHOD_ORDER:
        result = results[method_id]
        row = {
            "Method": result.display_name,
            "Status": result.status,
            "Silhouette": result.metrics.get("silhouette"),
            "Clusters": result.metrics.get("cluster_count"),
            "Noise %": result.metrics.get("noise_pct"),
            "Coherence": result.metrics.get("coherence"),
            "Actionability": result.metrics.get("actionability"),
            "Origin": result.artifact_origin,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_ticket_lookup(dataset):
    return {ticket.ticket_id: ticket for ticket in dataset.tickets}


def build_projection_frame(result):
    return pd.DataFrame(
        [
            {
                "ticket_id": point.ticket_id,
                "cluster_id": point.cluster_id,
                "x": point.x,
                "y": point.y,
                "label": point.label,
            }
            for point in result.projection
        ]
    )


def render_dataset_view(dataset):
    st.subheader("Dataset Summary")
    stats = dataset.stats
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Tickets", stats["ticket_count"])
    col2.metric("Messages", stats["messages_total"])
    col3.metric("Avg messages/ticket", stats["messages_avg"])
    col4.metric("Avg customer words", stats["customer_words_avg"])

    language_df = pd.DataFrame(
        {"language": list(stats["languages"].keys()), "count": list(stats["languages"].values())}
    )
    status_df = pd.DataFrame(
        {"status": list(stats["statuses"].keys()), "count": list(stats["statuses"].values())}
    )
    priority_df = pd.DataFrame(
        {"priority": list(stats["priorities"].keys()), "count": list(stats["priorities"].values())}
    )

    chart_col1, chart_col2, chart_col3 = st.columns(3)
    chart_col1.plotly_chart(px.bar(language_df, x="language", y="count", title="Language Mix"), use_container_width=True)
    chart_col2.plotly_chart(px.bar(status_df, x="status", y="count", title="Status Mix"), use_container_width=True)
    chart_col3.plotly_chart(px.bar(priority_df, x="priority", y="count", title="Priority Mix"), use_container_width=True)

    st.markdown("### Ticket Samples")
    sample_df = pd.DataFrame(
        [
            {
                "ticket_id": ticket.ticket_id,
                "language": ticket.language,
                "status": ticket.status,
                "priority": ticket.priority,
                "subject": ticket.subject,
                "analysis_text": ticket.analysis_text[:180],
            }
            for ticket in dataset.tickets[:10]
        ]
    )
    st.dataframe(sample_df, use_container_width=True, hide_index=True)


def render_method_comparison(results, is_bundled: bool):
    st.subheader("Method Comparison")
    metrics_df = build_metric_table(results)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    chart_df = metrics_df.melt(
        id_vars=["Method", "Status", "Origin"],
        value_vars=["Silhouette", "Clusters", "Noise %", "Coherence", "Actionability"],
        var_name="Metric",
        value_name="Value",
    )
    st.plotly_chart(
        px.bar(chart_df, x="Method", y="Value", color="Metric", barmode="group", title="Metric Comparison"),
        use_container_width=True,
    )

    if is_bundled:
        st.info(
            "Bundled dataset shows poster reference metrics for any method that has not been recomputed into the cache yet."
        )

    for method_id in DEFAULT_METHOD_ORDER:
        result = results[method_id]
        if result.warnings:
            with st.expander(f"{result.display_name} notes"):
                for warning in result.warnings:
                    st.write(f"- {warning}")
                for note in result.notes:
                    st.write(f"- {note}")


def render_cluster_explorer(dataset, results):
    st.subheader("Cluster Explorer")
    method_id = st.selectbox(
        "Method",
        DEFAULT_METHOD_ORDER,
        format_func=lambda value: METHOD_DEFINITIONS[value]["name"],
    )
    result = results[method_id]
    ticket_lookup = build_ticket_lookup(dataset)

    if not result.clusters:
        st.warning("No cluster artifacts are available for this method yet.")
        if result.warnings:
            for warning in result.warnings:
                st.write(f"- {warning}")
        return

    projection_df = build_projection_frame(result)
    st.plotly_chart(
        px.scatter(
            projection_df,
            x="x",
            y="y",
            color="label",
            hover_data=["ticket_id", "cluster_id"],
            title="2D Cluster Projection",
        ),
        use_container_width=True,
    )

    cluster_options = {f"{cluster.label} ({cluster.size} tickets)": cluster for cluster in result.clusters}
    selected_cluster = st.selectbox("Cluster", list(cluster_options.keys()))
    cluster = cluster_options[selected_cluster]

    st.markdown(f"### {cluster.label}")
    st.write(f"Cluster size: {cluster.size}")
    st.write(f"Top terms: {', '.join(cluster.top_terms) if cluster.top_terms else 'n/a'}")

    rows = []
    for ticket_id in cluster.representative_ticket_ids:
        ticket = ticket_lookup[ticket_id]
        rows.append(
            {
                "ticket_id": ticket.ticket_id,
                "subject": ticket.subject,
                "language": ticket.language,
                "status": ticket.status,
                "analysis_text": ticket.analysis_text[:220],
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("### Per-ticket comparison")
    cluster_ticket_id = st.selectbox("Ticket", cluster.representative_ticket_ids)
    comparison_rows = []
    for compare_method_id in DEFAULT_METHOD_ORDER:
        compare_result = results[compare_method_id]
        assigned_cluster = compare_result.assignments.get(cluster_ticket_id)
        label = next(
            (item.label for item in compare_result.clusters if item.cluster_id == assigned_cluster),
            "Noise / Filtered" if assigned_cluster == -1 else "Unavailable",
        )
        comparison_rows.append(
            {
                "method": compare_result.display_name,
                "status": compare_result.status,
                "cluster_id": assigned_cluster,
                "cluster_label": label,
            }
        )
    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    ticket = ticket_lookup[cluster_ticket_id]
    with st.expander("Conversation detail"):
        for message in ticket.messages:
            st.markdown(f"**{message.role.title()}**: {message.content}")


def render_llm_trace(dataset, results):
    st.subheader("LLM Pipeline Trace")
    result = results["D"]
    if not result.ticket_artifacts:
        st.warning("Method D has no cached issue-extraction artifacts yet.")
        for warning in result.warnings:
            st.write(f"- {warning}")
        return

    ticket_lookup = build_ticket_lookup(dataset)
    available_ids = [
        ticket_id
        for ticket_id, artifact in result.ticket_artifacts.items()
        if artifact.get("issue_statement")
    ]
    if not available_ids:
        st.warning("No extracted issue statements are available yet.")
        return

    ticket_id = st.selectbox("Ticket for trace", available_ids)
    ticket = ticket_lookup[ticket_id]
    artifact = result.ticket_artifacts[ticket_id]
    assigned_cluster = result.assignments.get(ticket_id, -1)
    cluster_label = next(
        (cluster.label for cluster in result.clusters if cluster.cluster_id == assigned_cluster),
        "Noise / Filtered",
    )

    col1, col2, col3 = st.columns(3)
    col1.markdown("#### Raw Ticket")
    col1.write(ticket.analysis_text)
    col2.markdown("#### Filter + Extract")
    col2.write(f"Issue? `{artifact.get('is_issue')}`")
    col2.write(f"Filter source: `{artifact.get('filter_source')}`")
    col2.write(f"Reason: {artifact.get('filter_reason')}")
    col2.write(f"Extracted issue: {artifact.get('issue_statement', 'n/a')}")
    col3.markdown("#### Cluster Outcome")
    col3.write(f"Cluster id: `{assigned_cluster}`")
    col3.write(f"Cluster label: {cluster_label}")
    col3.write(f"Cluster summary: {artifact.get('cluster_summary', 'n/a')}")


def render_sidebar(dataset, results):
    st.sidebar.markdown("### Dataset")
    st.sidebar.write(f"Source: `{dataset.source_name}`")
    st.sidebar.write(f"Hash: `{dataset.dataset_hash}`")
    st.sidebar.write(f"Tickets: `{dataset.stats['ticket_count']}`")

    st.sidebar.markdown("### Compute")
    if st.sidebar.button("Recompute all methods"):
        st.session_state["force_recompute"] = True

    st.sidebar.markdown("### Method status")
    for method_id in DEFAULT_METHOD_ORDER:
        result = results[method_id]
        st.sidebar.write(f"`{method_id}` {result.status}")


def main():
    st.title(APP_TITLE)
    st.caption("Compare four clustering approaches for Zendesk-style support tickets.")

    dataset, is_bundled = load_active_dataset()
    if dataset is None:
        st.stop()

    force_recompute = bool(st.session_state.pop("force_recompute", False))
    results = load_results(dataset, is_bundled=is_bundled, force=force_recompute)
    render_sidebar(dataset, results)

    view = st.radio(
        "View",
        ["Dataset", "Method Comparison", "Cluster Explorer", "LLM Pipeline Trace"],
        horizontal=True,
    )

    if view == "Dataset":
        render_dataset_view(dataset)
    elif view == "Method Comparison":
        render_method_comparison(results, is_bundled=is_bundled)
    elif view == "Cluster Explorer":
        render_cluster_explorer(dataset, results)
    else:
        render_llm_trace(dataset, results)


if __name__ == "__main__":
    main()
