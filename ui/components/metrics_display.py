"""
Metrics display components for the Insurance LLM Framework.

This module provides UI components for displaying evaluation metrics and benchmark results.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, Any, Optional, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def metrics_selector(available_metrics: List[Dict[str, Any]]) -> List[str]:
    """
    Display a selector for choosing which metrics to display.

    Args:
        available_metrics: List of available metrics

    Returns:
        List of selected metric names
    """

    metric_options = {f"{m['name']} - {m['description']}": m['name']
                      for m in available_metrics}

    selected_options = st.multiselect(
        "Select Metrics",
        options=list(metric_options.keys()),
        default=list(metric_options.keys())[:min(3, len(metric_options))],
        help="Choose which metrics to display"
    )

    return [metric_options[option] for option in selected_options]

def metrics_weights_editor(metrics: List[str],
                           default_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Display sliders for adjusting the weights of different metrics.

    Args:
        metrics: List of metric names
        default_weights: Dictionary mapping metric names to default weights

    Returns:
        Dictionary mapping metric names to weights
    """
    st.subheader("Metric Weights")
    st.caption("Adjust the importance of each metric in the overall score")

    weights = {}

    for metric in metrics:
        default_weight = default_weights.get(
            metric, 1.0) if default_weights else 1.0
        weights[metric] = st.slider(
            f"{metric} Weight",
            min_value=0.0,
            max_value=2.0,
            value=default_weight,
            step=0.1,
            help=f"Weight for the {metric} metric"
        )

    total_weight = sum(weights.values())
    if total_weight > 0:
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
    else:
        normalized_weights = {k: 0.0 for k in weights.keys()}

    with st.expander("Normalized Weights", expanded=False):
        weights_df = pd.DataFrame({
            "Metric": list(normalized_weights.keys()),
            "Raw Weight": list(weights.values()),
            "Normalized Weight": list(normalized_weights.values())
        })
        st.dataframe(weights_df, hide_index=True)

    return weights

def display_model_comparison(model_results: Dict[str, Dict[str, Any]],
                             metrics: Optional[List[str]] = None):
    """
    Display a comparison of results from multiple models.

    Args:
        model_results: Dictionary mapping model names to their results
        metrics: Optional list of metrics to include in the comparison
    """
    if not model_results:
        st.warning("No model results to compare")
        return

    st.subheader("Model Comparison")

    if not metrics:

        all_metrics = set()
        for model_name, results in model_results.items():
            if "metrics" in results:
                all_metrics.update(results["metrics"].keys())
        metrics = list(all_metrics)

    comparison_data = []
    for model_name, results in model_results.items():
        model_data = {"Model": model_name}

        if "metrics" in results:
            for metric in metrics:
                if metric in results["metrics"]:
                    result = results["metrics"][metric]

                    if hasattr(result, 'as_dict'):
                        result_dict = result.as_dict()
                    else:
                        result_dict = result

                    normalized_score = result_dict.get("normalized_score", 0)
                    model_data[metric] = normalized_score
                else:
                    model_data[metric] = None

        if "overall_score" in results:
            model_data["Overall"] = results["overall_score"]
        elif "metrics" in results and len(results["metrics"]) > 0:

            valid_scores = [v for k, v in model_data.items(
            ) if k != "Model" and v is not None]
            if valid_scores:
                model_data["Overall"] = sum(valid_scores) / len(valid_scores)

        comparison_data.append(model_data)

    df = pd.DataFrame(comparison_data)

    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True
    )

    if len(df) > 0 and len(metrics) > 0:

        plot_df = df.melt(
            id_vars=["Model"],
            value_vars=metrics +
            (["Overall"] if "Overall" in df.columns else []),
            var_name="Metric",
            value_name="Score"
        )

        fig = px.bar(
            plot_df,
            x="Model",
            y="Score",
            color="Metric",
            barmode="group",
            range_y=[0, 1],
            labels={"Score": "Normalized Score"},
            title="Model Comparison by Metric"
        )

        fig.update_layout(
            xaxis_title=None,
            legend_title="Metric",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        if len(metrics) >= 3:

            categories = metrics

            fig = go.Figure()

            for model_name in df["Model"]:
                model_row = df[df["Model"] == model_name].iloc[0]
                values = [model_row.get(metric, 0) for metric in categories]

                values.append(values[0])
                categories_closed = categories + [categories[0]]

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories_closed,
                    fill='toself',
                    name=model_name
                ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Model Comparison (Radar Chart)",
                height=500
            )

            st.plotly_chart(fig, use_container_width=True)

def display_benchmark_results(benchmark_result: Dict[str, Any]):
    """
    Display the results of a benchmark evaluation.

    Args:
        benchmark_result: Dictionary containing benchmark results
    """
    if not benchmark_result:
        st.warning("No benchmark results to display")
        return

    st.subheader(
        f"Benchmark Results: {benchmark_result.get('benchmark_name', 'Unknown')}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"**Model:** {benchmark_result.get('model_id', 'Unknown')}")
        st.markdown(
            f"**Task Type:** {benchmark_result.get('task_type', 'Unknown')}")

    with col2:
        if "timestamp" in benchmark_result:
            st.markdown(f"**Date:** {benchmark_result['timestamp']}")
        if "total_examples" in benchmark_result:
            st.markdown(f"**Examples:** {benchmark_result['total_examples']}")

    if "aggregate_scores" in benchmark_result:
        st.subheader("Aggregate Scores")

        scores_data = [{"Metric": metric, "Score": score}
                       for metric, score in benchmark_result["aggregate_scores"].items()]

        scores_df = pd.DataFrame(scores_data)

        overall_score = scores_df["Score"].mean()

        st.metric(
            "Overall Score",
            f"{overall_score:.3f}",
            help="Average of all metric scores"
        )

        fig = px.bar(
            scores_df,
            x="Metric",
            y="Score",
            range_y=[0, 1],
            labels={"Score": "Score (normalized)"},
            title="Benchmark Metrics"
        )

        fig.add_hline(
            y=overall_score,
            line_dash="dash",
            line_color="green",
            annotation_text=f"Avg: {overall_score:.3f}"
        )

        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Score",
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)

    if "results" in benchmark_result and benchmark_result["results"]:
        st.subheader("Example Results")

        results_df = pd.DataFrame(benchmark_result["results"])

        if not results_df.empty:

            metric_cols = [
                col for col in results_df.columns if col.startswith("metric_")]

            display_cols = {"example_id": "ID"}
            display_cols.update({col: col.replace("metric_", "")
                                for col in metric_cols})

            display_df = results_df[["example_id"] +
                                    metric_cols].rename(columns=display_cols)

            st.dataframe(display_df, hide_index=True, use_container_width=True)

        with st.expander("Explore Individual Examples", expanded=False):

            example_ids = results_df["example_id"].tolist()
            selected_id = st.selectbox("Select Example", options=example_ids)

            if selected_id:

                example = results_df[results_df["example_id"]
                                     == selected_id].iloc[0]

                st.markdown("### Example Details")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Input:**")
                    st.text_area("Input Text", value=example.get(
                        "input_text", ""), height=200, disabled=True)

                with col2:
                    st.markdown("**Generated Output:**")
                    st.text_area("Generated Text", value=example.get(
                        "generated_text", ""), height=200, disabled=True)

                if "reference_text" in example and example["reference_text"]:
                    st.markdown("**Reference Output:**")
                    st.text_area(
                        "Reference Text", value=example["reference_text"], height=150, disabled=True)

                st.markdown("**Metric Scores:**")
                metric_scores = {k.replace(
                    "metric_", ""): v for k, v in example.items() if k.startswith("metric_")}

                metrics_df = pd.DataFrame({
                    "Metric": list(metric_scores.keys()),
                    "Score": list(metric_scores.values())
                })

                if not metrics_df.empty:
                    fig = px.bar(
                        metrics_df,
                        y="Metric",
                        x="Score",
                        orientation='h',
                        range_x=[0, 1],
                        labels={"Score": "Score (normalized)"},
                        height=300
                    )

                    st.plotly_chart(fig, use_container_width=True)
