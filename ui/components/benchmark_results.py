"""
Benchmark results components for the Insurance LLM Framework.

This module provides UI components for displaying and comparing benchmark results.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, List
import logging
import json
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def benchmark_selector(benchmarks: List[Dict[str, Any]]) -> Optional[str]:
    """
    Display a selector for choosing a benchmark.

    Args:
        benchmarks: List of available benchmarks

    Returns:
        The selected benchmark name or None
    """
    # Sort benchmarks by task type
    sorted_benchmarks = sorted(
        benchmarks, key=lambda x: x.get("task_type", ""))

    # Group benchmarks by task type for easier selection
    task_types = {}
    for benchmark in sorted_benchmarks:
        task_type = benchmark.get("task_type", "Other")
        if task_type not in task_types:
            task_types[task_type] = []
        task_types[task_type].append(benchmark["name"])

    # Create a hierarchical selector
    selected_task_type = st.selectbox(
        "Select Task Type",
        options=list(task_types.keys()),
        help="Choose the type of task to benchmark"
    )

    if selected_task_type and selected_task_type in task_types:
        selected_benchmark = st.selectbox(
            "Select Benchmark",
            options=task_types[selected_task_type],
            help="Choose the specific benchmark to run"
        )

        if selected_benchmark:
            # Show benchmark details
            benchmark_info = next(
                (b for b in benchmarks if b["name"] == selected_benchmark), None)
            if benchmark_info:
                st.info(benchmark_info.get(
                    "description", "No description available"))
                st.markdown(
                    f"**Examples:** {len(benchmark_info.get('examples', []))}")

                # Show metrics used
                if "metrics" in benchmark_info:
                    st.markdown("**Metrics:**")
                    for metric in benchmark_info["metrics"]:
                        st.markdown(f"- {metric}")

            return selected_benchmark

    return None


def display_benchmark_summary(benchmark_results: List[Dict[str, Any]]):
    """
    Display a summary of all benchmark results.

    Args:
        benchmark_results: List of benchmark result dictionaries
    """
    if not benchmark_results:
        st.warning("No benchmark results available")
        return

    st.subheader("Benchmark Results Summary")

    # Extract summary data for each benchmark result
    summary_data = []
    for result in benchmark_results:
        # Basic information
        summary_info = {
            "Benchmark": result.get("benchmark_name", "Unknown"),
            "Model": result.get("model_id", "Unknown"),
            "Task Type": result.get("task_type", "Unknown"),
            "Date": result.get("timestamp", "Unknown")
        }

        # Add aggregate scores
        if "aggregate_scores" in result:
            for metric, score in result["aggregate_scores"].items():
                summary_info[f"metric_{metric}"] = score

            # Calculate overall score
            scores = list(result["aggregate_scores"].values())
            if scores:
                summary_info["Overall Score"] = sum(scores) / len(scores)

        summary_data.append(summary_info)

    # Create DataFrame
    df = pd.DataFrame(summary_data)

    # Extract metric columns
    metric_cols = [col for col in df.columns if col.startswith("metric_")]
    display_cols = ["Benchmark", "Model", "Task Type",
                    "Date"] + metric_cols + ["Overall Score"]

    # Rename metric columns for display
    rename_map = {col: col.replace("metric_", "") for col in metric_cols}
    df_display = df[display_cols].rename(columns=rename_map)

    # Display table
    st.dataframe(df_display, hide_index=True, use_container_width=True)

    # Create visualization if we have data
    if len(df) > 0 and "Overall Score" in df.columns:
        # Create bar chart of overall scores by model and benchmark
        pivot_df = df.pivot_table(
            index="Benchmark",
            columns="Model",
            values="Overall Score",
            aggfunc="mean"
        ).reset_index()

        # Melt for plotting
        plot_df = pd.melt(
            pivot_df,
            id_vars=["Benchmark"],
            var_name="Model",
            value_name="Score"
        )

        # Create plot
        fig = px.bar(
            plot_df,
            x="Benchmark",
            y="Score",
            color="Model",
            barmode="group",
            range_y=[0, 1],
            labels={"Score": "Overall Score"},
            title="Benchmark Comparison by Model"
        )

        # Update layout
        fig.update_layout(
            xaxis_title=None,
            legend_title="Model",
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)


def display_benchmark_comparison(benchmark_results: List[Dict[str, Any]]):
    """
    Display a detailed comparison of benchmark results for different models.

    Args:
        benchmark_results: List of benchmark result dictionaries
    """
    if not benchmark_results:
        st.warning("No benchmark results available")
        return

    # Group by benchmark
    benchmarks = {}
    for result in benchmark_results:
        benchmark_name = result.get("benchmark_name", "Unknown")
        if benchmark_name not in benchmarks:
            benchmarks[benchmark_name] = []
        benchmarks[benchmark_name].append(result)

    # Select benchmark to compare
    selected_benchmark = st.selectbox(
        "Select Benchmark for Comparison",
        options=list(benchmarks.keys()),
        help="Choose which benchmark to compare across models"
    )

    if selected_benchmark and selected_benchmark in benchmarks:
        results = benchmarks[selected_benchmark]

        # Extract available metrics from first result
        all_metrics = set()
        for result in results:
            if "aggregate_scores" in result:
                all_metrics.update(result["aggregate_scores"].keys())

        # Select metrics to compare
        if all_metrics:
            selected_metrics = st.multiselect(
                "Select Metrics",
                options=list(all_metrics),
                default=list(all_metrics),
                help="Choose which metrics to include in the comparison"
            )
        else:
            selected_metrics = []

        # Prepare data for visualization
        comparison_data = []
        for result in results:
            model_id = result.get("model_id", "Unknown")

            for metric in selected_metrics:
                if "aggregate_scores" in result and metric in result["aggregate_scores"]:
                    comparison_data.append({
                        "Model": model_id,
                        "Metric": metric,
                        "Score": result["aggregate_scores"][metric]
                    })

        # Create DataFrame
        df = pd.DataFrame(comparison_data)

        if not df.empty:
            # Display comparison chart
            fig = px.bar(
                df,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                range_y=[0, 1],
                labels={"Score": "Normalized Score"},
                title=f"Model Comparison: {selected_benchmark}"
            )

            # Update layout
            fig.update_layout(
                xaxis_title=None,
                legend_title="Metric",
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Calculate and show overall scores
            overall_scores = df.groupby("Model")["Score"].mean().reset_index()
            overall_scores = overall_scores.sort_values(
                "Score", ascending=False)

            # Create overall score bar chart
            fig = px.bar(
                overall_scores,
                x="Model",
                y="Score",
                range_y=[0, 1],
                labels={"Score": "Overall Score"},
                title="Overall Benchmark Performance"
            )

            # Update layout
            fig.update_layout(
                xaxis_title=None,
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

            # Detail table
            st.markdown("### Detailed Scores")

            # Pivot the data for a cleaner display
            pivot_df = df.pivot(index="Model", columns="Metric",
                                values="Score").reset_index()

            # Add overall column
            pivot_df["Overall"] = pivot_df.mean(axis=1)

            # Format numbers
            for col in pivot_df.columns:
                if col != "Model":
                    pivot_df[col] = pivot_df[col].apply(lambda x: f"{x:.3f}")

            # Display table
            st.dataframe(pivot_df, hide_index=True, use_container_width=True)
        else:
            st.warning("No metric data available for the selected benchmark")
    else:
        st.info("Select a benchmark to compare results")


def export_benchmark_results(benchmark_results: List[Dict[str, Any]], format: str = "csv"):
    """
    Export benchmark results to a file.

    Args:
        benchmark_results: List of benchmark result dictionaries
        format: Export format ("csv" or "json")

    Returns:
        Tuple of (filename, file_data) for download
    """
    if not benchmark_results:
        return None, None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if format == "csv":
        # Extract summary data
        rows = []
        for result in benchmark_results:
            row = {
                "benchmark": result.get("benchmark_name", "Unknown"),
                "model": result.get("model_id", "Unknown"),
                "task_type": result.get("task_type", "Unknown"),
                "date": result.get("timestamp", "Unknown")
            }

            # Add aggregate scores
            if "aggregate_scores" in result:
                for metric, score in result["aggregate_scores"].items():
                    row[f"metric_{metric}"] = score

            rows.append(row)

        # Create DataFrame and export to CSV
        df = pd.DataFrame(rows)
        csv_data = df.to_csv(index=False)
        filename = f"benchmark_results_{timestamp}.csv"

        return filename, csv_data

    elif format == "json":
        # Export full results as JSON
        json_data = json.dumps(benchmark_results, indent=2)
        filename = f"benchmark_results_{timestamp}.json"

        return filename, json_data

    return None, None
