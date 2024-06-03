"""
Benchmarks Page for the Insurance LLM Framework.

This module provides the interface for running standardized benchmarks on models.
"""

import streamlit as st
import logging
import os
import json
import pandas as pd
import time
from datetime import datetime
from typing import Dict, Any, Optional, List

# Import UI components
from ui.components.benchmark_results import benchmark_selector, display_benchmark_summary, display_benchmark_comparison

logger = logging.getLogger(__name__)


def render():
    """Render the benchmarks page."""
    st.title("📊 Benchmarks")

    with st.expander("ℹ️ About Benchmarks", expanded=False):
        st.markdown("""
        This page allows you to run standardized benchmarks to evaluate model performance on insurance tasks.
        
        ### Benchmark Types
        - **Task-specific benchmarks**: Evaluate models on specific insurance tasks
        - **Industry benchmarks**: Compare against established insurance industry standards
        - **Custom benchmarks**: Create and run your own benchmarks
        
        ### Benchmarking Process
        1. Select a benchmark suite or create a custom benchmark
        2. Choose models to evaluate
        3. Run the benchmark
        4. Review results and export findings
        """)

    # Check if model is loaded
    if "active_model" not in st.session_state or not st.session_state.active_model:
        st.warning(
            "⚠️ No model is currently loaded. Please select a model in the Model Selection page.")
        if st.button("Go to Model Selection"):
            # This would navigate to model selection page in a real app
            st.info("In a real app, this would navigate to the Model Selection page.")
        return

    # Create tabs for different benchmark functions
    tabs = st.tabs(["Run Benchmarks", "Benchmark Results", "Create Benchmark"])

    # Run Benchmarks Tab
    with tabs[0]:
        st.subheader("Run Benchmarks")

        # Benchmark selection
        selected_benchmark = benchmark_selector()

        if selected_benchmark:
            st.markdown(f"### {selected_benchmark['name']}")
            st.markdown(selected_benchmark["description"])

            # Display benchmark details
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown(
                    f"**Task Type:** {selected_benchmark['task_type'].capitalize()}")
                st.markdown(
                    f"**Number of Examples:** {len(selected_benchmark.get('examples', []))}")

            with col2:
                st.markdown(
                    f"**Domain:** {selected_benchmark['domain'].capitalize()}")
                metrics_list = ", ".join(
                    [m.capitalize() for m in selected_benchmark.get("metrics", [])])
                st.markdown(f"**Metrics:** {metrics_list}")

            # Sample of benchmark examples
            with st.expander("View Example Items", expanded=False):
                examples = selected_benchmark.get("examples", [])
                if examples:
                    # Show first 3 examples
                    for i, example in enumerate(examples[:3]):
                        st.markdown(f"#### Example {i+1}")
                        st.markdown(f"**Input:**")
                        st.markdown(
                            f"```\n{example.get('input', 'No input')}\n```")

                        if "reference" in example:
                            st.markdown(f"**Reference Output:**")
                            st.markdown(f"```\n{example['reference']}\n```")

                if len(examples) > 3:
                    st.markdown(f"*...and {len(examples) - 3} more examples*")

            # Model selection
            st.markdown("### Select Models")

            # Current model is always selected
            current_model = st.session_state.active_model
            st.markdown(f"Current model: **{current_model}**")

            # Option to compare with other models
            compare_with_others = st.checkbox("Compare with other models")

            other_models = []
            if compare_with_others:
                # In a real app, this would fetch available models
                # For demo, we'll use a sample list
                available_models = ["llama2-7b", "llama2-13b",
                                    "mistral-7b-instruct", "falcon-7b"]
                available_models = [
                    m for m in available_models if m != current_model]

                if available_models:
                    other_models = st.multiselect(
                        "Select models to compare with",
                        options=available_models
                    )
                else:
                    st.info("No other models available for comparison.")

            # Benchmark configuration
            st.markdown("### Benchmark Configuration")

            col1, col2 = st.columns([1, 1])

            with col1:
                # Allow configuring temperature and max tokens
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=0.3,  # Lower default for benchmarks
                    step=0.1,
                    help="Lower values recommended for benchmarks"
                )

            with col2:
                max_tokens = st.slider(
                    "Max Tokens",
                    min_value=50,
                    max_value=4000,
                    value=1000,
                    step=50,
                    help="Maximum tokens for generation"
                )

            # Advanced options
            with st.expander("Advanced Options", expanded=False):
                # Sampling options
                sampling_method = st.selectbox(
                    "Sampling Method",
                    options=["greedy", "top_p", "top_k", "beam"],
                    index=0,
                    help="Greedy decoding recommended for benchmarks"
                )

                # Number of runs option
                num_runs = st.slider(
                    "Number of Runs",
                    min_value=1,
                    max_value=5,
                    value=1,
                    step=1,
                    help="Run benchmark multiple times for variance analysis"
                )

                # Seed option
                seed = st.number_input(
                    "Random Seed",
                    min_value=0,
                    max_value=10000,
                    value=42,
                    help="Set seed for reproducibility"
                )

            # Run benchmark button
            all_models = [current_model] + other_models
            if st.button("Run Benchmark", type="primary", use_container_width=True):
                # Simulate running benchmark
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Store benchmark results
                benchmark_results = {
                    "benchmark_id": selected_benchmark["id"],
                    "benchmark_name": selected_benchmark["name"],
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "configuration": {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "sampling_method": sampling_method,
                        "num_runs": num_runs,
                        "seed": seed
                    },
                    "models": {},
                    "overall_results": {}
                }

                for i, model in enumerate(all_models):
                    progress_value = i / len(all_models)
                    progress_bar.progress(progress_value)
                    status_text.text(f"Running benchmark on model: {model}")

                    # Simulate processing time based on model size
                    if "13b" in model:
                        delay = 3
                    elif "7b" in model:
                        delay = 2
                    else:
                        delay = 1

                    time.sleep(delay)

                    # Generate simulated results
                    examples = selected_benchmark.get("examples", [])
                    metrics = selected_benchmark.get("metrics", [])

                    model_results = {
                        "example_results": [],
                        "aggregate_metrics": {}
                    }

                    # Simulate example results
                    for j, example in enumerate(examples):
                        # Update progress for each example
                        example_progress = progress_value + \
                            (1 / len(all_models)) * (j / len(examples))
                        progress_bar.progress(example_progress)
                        status_text.text(
                            f"Model: {model} - Processing example {j+1}/{len(examples)}")

                        # Simulate example result
                        example_result = {
                            "input": example.get("input", ""),
                            "output": f"Simulated output for {model} on example {j+1}",
                            "metrics": {}
                        }

                        # Add reference if available
                        if "reference" in example:
                            example_result["reference"] = example["reference"]

                        # Simulate metric scores
                        import random
                        for metric in metrics:
                            # Larger models tend to get better scores in simulation
                            model_bonus = 0.1 if "13b" in model else 0.05 if "7b" in model else 0.0
                            base_score = 0.65 + model_bonus

                            # Different metrics for different tasks
                            if metric == "accuracy":
                                example_result["metrics"][metric] = min(
                                    0.95, base_score + random.uniform(0, 0.2))
                            elif metric == "clarity":
                                example_result["metrics"][metric] = min(
                                    0.9, base_score + random.uniform(0, 0.15))
                            elif metric == "completeness":
                                example_result["metrics"][metric] = min(
                                    0.9, base_score + random.uniform(0, 0.2))
                            elif metric == "relevance":
                                example_result["metrics"][metric] = min(
                                    0.92, base_score + random.uniform(0, 0.1))
                            elif metric == "bleu":
                                example_result["metrics"][metric] = min(
                                    0.8, base_score + random.uniform(0, 0.1) - 0.1)
                            elif metric == "rouge":
                                example_result["metrics"][metric] = min(
                                    0.85, base_score + random.uniform(0, 0.15))
                            else:
                                example_result["metrics"][metric] = min(
                                    0.9, base_score + random.uniform(0, 0.2))

                        model_results["example_results"].append(example_result)

                    # Calculate aggregate metrics
                    for metric in metrics:
                        scores = [ex["metrics"].get(
                            metric, 0) for ex in model_results["example_results"]]
                        if scores:
                            model_results["aggregate_metrics"][metric] = {
                                "mean": sum(scores) / len(scores),
                                "min": min(scores),
                                "max": max(scores)
                            }

                    # Calculate overall score
                    if model_results["aggregate_metrics"]:
                        mean_scores = [
                            metrics_data["mean"] for metrics_data in model_results["aggregate_metrics"].values()]
                        model_results["overall_score"] = sum(
                            mean_scores) / len(mean_scores)

                    # Add to benchmark results
                    benchmark_results["models"][model] = model_results

                # Complete progress bar
                progress_bar.progress(1.0)
                status_text.text("Benchmark completed!")

                # Calculate overall ranking
                model_scores = {model: results.get(
                    "overall_score", 0) for model, results in benchmark_results["models"].items()}
                sorted_models = sorted(
                    model_scores.items(), key=lambda x: x[1], reverse=True)
                benchmark_results["overall_results"]["ranking"] = [
                    {"model": model, "score": score} for model, score in sorted_models]

                # Average scores across all models for each metric
                all_metrics = selected_benchmark.get("metrics", [])
                benchmark_results["overall_results"]["metric_averages"] = {}

                for metric in all_metrics:
                    scores = []
                    for model, results in benchmark_results["models"].items():
                        if metric in results.get("aggregate_metrics", {}):
                            scores.append(
                                results["aggregate_metrics"][metric]["mean"])

                    if scores:
                        benchmark_results["overall_results"]["metric_averages"][metric] = sum(
                            scores) / len(scores)

                # Store benchmark results in session state
                if "benchmark_results" not in st.session_state:
                    st.session_state.benchmark_results = []

                st.session_state.benchmark_results.append(benchmark_results)
                st.session_state.current_benchmark_results = benchmark_results

                # Show success message
                st.success("Benchmark completed successfully!")

                # Display summary results
                display_benchmark_summary(benchmark_results)
        else:
            st.info("Select a benchmark to run.")

    # Benchmark Results Tab
    with tabs[1]:
        st.subheader("Benchmark Results")

        # Check if we have benchmark results
        if "benchmark_results" not in st.session_state or not st.session_state.benchmark_results:
            st.info("No benchmark results available. Run a benchmark first.")
        else:
            # Display list of benchmark runs
            benchmark_runs = st.session_state.benchmark_results

            # Create a table of benchmark runs
            run_data = [
                {
                    "Timestamp": run["timestamp"],
                    "Benchmark": run["benchmark_name"],
                    "Models": ", ".join(list(run["models"].keys())),
                    "Overall Score": f"{run['models'][list(run['models'].keys())[0]].get('overall_score', 0):.2f}" if run["models"] else "N/A"
                }
                for run in benchmark_runs
            ]

            run_df = pd.DataFrame(run_data)
            st.dataframe(run_df, use_container_width=True, hide_index=True)

            # Allow selecting a benchmark run
            run_timestamps = [run["timestamp"] for run in benchmark_runs]
            selected_timestamp = st.selectbox(
                "Select Benchmark Run",
                options=run_timestamps,
                format_func=lambda x: f"{x} - {next((run['benchmark_name'] for run in benchmark_runs if run['timestamp'] == x), 'Unknown')}"
            )

            selected_run = next(
                (run for run in benchmark_runs if run["timestamp"] == selected_timestamp), None)

            if selected_run:
                # Display benchmark results
                st.markdown(
                    f"### Results for {selected_run['benchmark_name']}")
                st.markdown(f"*Run at: {selected_run['timestamp']}*")

                # Configuration info
                with st.expander("Benchmark Configuration", expanded=False):
                    config = selected_run.get("configuration", {})
                    st.markdown(
                        f"**Temperature:** {config.get('temperature', 'N/A')}")
                    st.markdown(
                        f"**Max Tokens:** {config.get('max_tokens', 'N/A')}")
                    st.markdown(
                        f"**Sampling Method:** {config.get('sampling_method', 'N/A')}")
                    st.markdown(
                        f"**Number of Runs:** {config.get('num_runs', 'N/A')}")
                    st.markdown(
                        f"**Random Seed:** {config.get('seed', 'N/A')}")

                # Display benchmark results
                display_benchmark_comparison(selected_run)

                # Export options
                export_format = st.selectbox(
                    "Export Format",
                    options=["JSON", "CSV", "PDF"]
                )

                if st.button("Export Results"):
                    if export_format == "JSON":
                        # Export as JSON
                        st.download_button(
                            label="Download JSON",
                            data=json.dumps(selected_run, indent=2),
                            file_name=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    elif export_format == "CSV":
                        # Export as CSV (flattened structure)
                        csv_data = []

                        for model, results in selected_run["models"].items():
                            model_row = {
                                "benchmark": selected_run["benchmark_name"],
                                "timestamp": selected_run["timestamp"],
                                "model": model,
                                "overall_score": results.get("overall_score", "N/A")
                            }

                            # Add aggregate metrics
                            for metric, values in results.get("aggregate_metrics", {}).items():
                                for key, value in values.items():
                                    model_row[f"{metric}_{key}"] = value

                            csv_data.append(model_row)

                        csv_df = pd.DataFrame(csv_data)
                        csv_string = csv_df.to_csv(index=False)

                        st.download_button(
                            label="Download CSV",
                            data=csv_string,
                            file_name=f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        # PDF would be implemented in a real app
                        st.info(
                            "PDF export would be implemented in a production version.")

    # Create Benchmark Tab
    with tabs[2]:
        st.subheader("Create Custom Benchmark")

        st.markdown("""
        Create a custom benchmark to evaluate models on specific insurance tasks.
        Custom benchmarks can be tailored to your specific needs and use cases.
        """)

        # Form for creating a benchmark
        with st.form("create_benchmark_form"):
            # Basic information
            benchmark_name = st.text_input(
                "Benchmark Name", placeholder="e.g., Policy Summarization Benchmark")
            benchmark_description = st.text_area(
                "Description", placeholder="Describe the purpose and focus of this benchmark")

            col1, col2 = st.columns([1, 1])

            with col1:
                task_options = ["summarization", "generation",
                                "analysis", "classification", "extraction", "other"]
                task_type = st.selectbox("Task Type", options=task_options)

            with col2:
                domain_options = ["general", "policy", "claims",
                                  "underwriting", "customer service", "compliance", "other"]
                domain = st.selectbox("Insurance Domain",
                                      options=domain_options)

            # Evaluation metrics
            st.markdown("### Evaluation Metrics")
            metrics_options = ["accuracy", "clarity", "completeness",
                               "relevance", "bleu", "rouge", "semantic_similarity"]
            selected_metrics = st.multiselect(
                "Select Metrics", options=metrics_options)

            # Benchmark examples
            st.markdown("### Benchmark Examples")
            st.info(
                "Add examples for your benchmark. Each example should include an input and reference output.")

            num_examples = st.slider(
                "Number of Examples", min_value=1, max_value=20, value=3)

            examples = []
            for i in range(num_examples):
                with st.expander(f"Example {i+1}", expanded=i == 0):
                    example_input = st.text_area(
                        f"Input {i+1}", key=f"input_{i}", height=100)
                    example_reference = st.text_area(
                        f"Reference Output {i+1}", key=f"reference_{i}", height=100)

                    if example_input:
                        examples.append({
                            "input": example_input,
                            "reference": example_reference
                        })

            # Submit button
            submitted = st.form_submit_button(
                "Create Benchmark", use_container_width=True)

            if submitted:
                if not benchmark_name:
                    st.error("Benchmark name is required")
                elif not selected_metrics:
                    st.error("Please select at least one evaluation metric")
                elif not examples:
                    st.error("Please add at least one example")
                else:
                    # Create benchmark dictionary
                    benchmark = {
                        "id": f"custom_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                        "name": benchmark_name,
                        "description": benchmark_description,
                        "task_type": task_type,
                        "domain": domain,
                        "metrics": selected_metrics,
                        "examples": examples,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

                    # Store in session state
                    if "custom_benchmarks" not in st.session_state:
                        st.session_state.custom_benchmarks = []

                    st.session_state.custom_benchmarks.append(benchmark)

                    # Success message
                    st.success(
                        f"Benchmark '{benchmark_name}' created successfully!")

                    # Option to run the benchmark immediately
                    if st.button("Run This Benchmark Now"):
                        # In a real app, this would navigate to the run benchmark tab with this benchmark selected
                        st.info(
                            "In a real app, this would navigate to the Run Benchmark tab with this benchmark selected.")


if __name__ == "__main__":
    # For testing the page in isolation
    st.set_page_config(
        page_title="Benchmarks - Insurance LLM Framework",
        page_icon="📊",
        layout="wide"
    )

    # Initialize session state for testing
    if "active_model" not in st.session_state:
        st.session_state.active_model = "llama2-7b"

    if "benchmark_results" not in st.session_state:
        # Sample benchmark results for testing
        st.session_state.benchmark_results = [{
            "benchmark_id": "policy_summarization",
            "benchmark_name": "Policy Summarization Benchmark",
            "timestamp": "2023-10-25 14:30:45",
            "configuration": {
                "temperature": 0.3,
                "max_tokens": 1000,
                "sampling_method": "greedy",
                "num_runs": 1,
                "seed": 42
            },
            "models": {
                "llama2-7b": {
                    "overall_score": 0.78,
                    "aggregate_metrics": {
                        "clarity": {
                            "mean": 0.82,
                            "min": 0.75,
                            "max": 0.88
                        },
                        "completeness": {
                            "mean": 0.75,
                            "min": 0.68,
                            "max": 0.85
                        },
                        "accuracy": {
                            "mean": 0.77,
                            "min": 0.7,
                            "max": 0.83
                        }
                    }
                }
            },
            "overall_results": {
                "ranking": [
                    {"model": "llama2-7b", "score": 0.78}
                ],
                "metric_averages": {
                    "clarity": 0.82,
                    "completeness": 0.75,
                    "accuracy": 0.77
                }
            }
        }]

    render()
