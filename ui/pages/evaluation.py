"""
Evaluation Page for the Insurance LLM Framework.

This module provides the interface for evaluating generated text using different metrics.
"""

import streamlit as st
import logging
import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List

from ui.components.metrics_display import display_metrics
from ui.components.output_display import display_evaluation_results

logger = logging.getLogger(__name__)

def render():
    """Render the evaluation page."""
    st.title("🔍 Evaluation")

    with st.expander("ℹ️ About Evaluation", expanded=False):
        st.markdown("""
        This page allows you to evaluate the performance of language models on insurance tasks.

        - **Automated Metrics**: Quantitative measures of output quality
        - **Human Evaluation**: Qualitative assessment by human evaluators
        - **Insurance-Specific Metrics**: Domain-specific measures for insurance text

        1. Select outputs to evaluate
        2. Choose evaluation metrics
        3. Run evaluation
        4. Review and export results
        """)

    tabs = st.tabs(["Automatic Evaluation", "Human Evaluation",
                   "Comparative Evaluation"])

    with tabs[0]:
        st.subheader("Automatic Evaluation")

        st.markdown("""
        Automatic evaluation uses predefined metrics to assess the quality of generated text.
        These metrics provide objective measurements across different dimensions.
        """)

        if "generation_history" not in st.session_state or not st.session_state.generation_history:
            st.warning(
                "No generations available for evaluation. Generate text first.")
            return

        selected_indices = []
        generations = st.session_state.generation_history

        generation_data = [
            {
                "Timestamp": g["timestamp"],
                "Model": g["model"],
                "Template": g["template_name"],
                "Task": g["task_type"].capitalize() if "task_type" in g else "Unknown",
            }
            for g in generations
        ]

        generation_df = pd.DataFrame(generation_data)
        st.dataframe(generation_df, use_container_width=True, hide_index=True)

        generation_timestamps = [g["timestamp"] for g in generations]
        selected_timestamp = st.selectbox(
            "Select Generation to Evaluate",
            options=generation_timestamps,
            format_func=lambda x: f"{x} - {next((g['template_name'] for g in generations if g['timestamp'] == x), 'Unknown')}"
        )

        selected_generation = next(
            (g for g in generations if g["timestamp"] == selected_timestamp), None)

        if selected_generation:
            st.markdown(
                f"### Evaluating: {selected_generation['template_name']}")

            st.markdown("### Select Metrics")

            col1, col2 = st.columns(2)

            with col1:
                basic_metrics = st.multiselect(
                    "Basic Metrics",
                    options=["Length", "Readability",
                             "Coherence", "Grammatical Correctness"],
                    default=["Readability", "Coherence"]
                )

            with col2:
                insurance_metrics = st.multiselect(
                    "Insurance-Specific Metrics",
                    options=["Compliance", "Term Accuracy",
                             "Clarity", "Completeness"],
                    default=["Clarity", "Completeness"]
                )

            st.markdown("### Reference Text (Optional)")
            st.markdown(
                "If you have a reference or 'gold standard' text, you can use it for reference-based metrics.")

            reference_text = st.text_area(
                "Reference Text",
                height=150,
                placeholder="Enter reference text for comparison-based metrics (optional)"
            )

            if reference_text:
                with col1:
                    reference_metrics = st.multiselect(
                        "Reference-Based Metrics",
                        options=["BLEU", "ROUGE", "BERTScore",
                                 "Semantic Similarity"],
                        default=["ROUGE", "Semantic Similarity"]
                    )

            if st.button("Run Evaluation", type="primary", use_container_width=True):

                with st.spinner("Calculating metrics..."):

                    eval_results = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "model": selected_generation["model"],
                        "task_type": selected_generation.get("task_type", "Unknown"),
                        "template": selected_generation["template_name"],
                        "metrics": {}
                    }

                    if "Length" in basic_metrics:
                        eval_results["metrics"]["Length"] = {
                            "chars": len(selected_generation["generated_text"]),
                            "words": len(selected_generation["generated_text"].split()),
                            "sentences": selected_generation["generated_text"].count(".") + selected_generation["generated_text"].count("!") + selected_generation["generated_text"].count("?")
                        }

                    if "Readability" in basic_metrics:
                        eval_results["metrics"]["Readability"] = {
                            "flesch_reading_ease": 65.2,  # Simulated
                            "flesch_kincaid_grade": 8.7,  # Simulated
                            "dale_chall_readability": 7.2  # Simulated
                        }

                    if "Coherence" in basic_metrics:
                        eval_results["metrics"]["Coherence"] = {
                            "score": 0.78,  # Simulated
                            "explanation": "Text demonstrates strong logical flow and consistent topic maintenance."
                        }

                    if "Grammatical Correctness" in basic_metrics:
                        eval_results["metrics"]["Grammatical Correctness"] = {
                            "score": 0.92,  # Simulated
                            "errors": 2,  # Simulated
                            "explanation": "Few minor errors detected, generally well-formed."
                        }

                    if "Compliance" in insurance_metrics:
                        eval_results["metrics"]["Compliance"] = {
                            "score": 0.85,  # Simulated
                            "issues": 0,  # Simulated
                            "explanation": "Text adheres to regulatory guidelines."
                        }

                    if "Term Accuracy" in insurance_metrics:
                        eval_results["metrics"]["Term Accuracy"] = {
                            "score": 0.91,  # Simulated
                            "explanation": "Insurance terms used correctly and appropriately."
                        }

                    if "Clarity" in insurance_metrics:
                        eval_results["metrics"]["Clarity"] = {
                            "score": 0.83,  # Simulated
                            "explanation": "Insurance concepts explained clearly for general audience."
                        }

                    if "Completeness" in insurance_metrics:
                        eval_results["metrics"]["Completeness"] = {
                            "score": 0.78,  # Simulated
                            "explanation": "Addresses most key aspects of the task."
                        }

                    if reference_text:
                        if "BLEU" in reference_metrics:
                            eval_results["metrics"]["BLEU"] = {
                                "score": 0.65,  # Simulated
                                "explanation": "Moderate n-gram overlap with reference."
                            }

                        if "ROUGE" in reference_metrics:
                            eval_results["metrics"]["ROUGE"] = {
                                "rouge-1": 0.72,  # Simulated
                                "rouge-2": 0.48,  # Simulated
                                "rouge-l": 0.67,  # Simulated
                                "explanation": "Good recall of key phrases from reference."
                            }

                        if "BERTScore" in reference_metrics:
                            eval_results["metrics"]["BERTScore"] = {
                                "precision": 0.87,  # Simulated
                                "recall": 0.82,  # Simulated
                                "f1": 0.84,  # Simulated
                                "explanation": "Strong semantic similarity to reference."
                            }

                        if "Semantic Similarity" in reference_metrics:
                            eval_results["metrics"]["Semantic Similarity"] = {
                                "score": 0.81,  # Simulated
                                "explanation": "Strong semantic alignment with reference content."
                            }

                    overall_scores = []
                    for metric_name, metric_data in eval_results["metrics"].items():
                        if isinstance(metric_data, dict) and "score" in metric_data:
                            overall_scores.append(metric_data["score"])
                        elif isinstance(metric_data, dict) and "f1" in metric_data:
                            overall_scores.append(metric_data["f1"])
                        elif metric_name == "ROUGE" and isinstance(metric_data, dict):
                            overall_scores.append(metric_data["rouge-l"])

                    if overall_scores:
                        eval_results["overall_score"] = sum(
                            overall_scores) / len(overall_scores)

                    if "automatic_evaluations" not in st.session_state:
                        st.session_state.automatic_evaluations = []

                    st.session_state.automatic_evaluations.append(eval_results)

                    selected_generation["automatic_evaluation"] = eval_results

                st.success("Evaluation complete!")

                display_metrics(eval_results)

                export_formats = ["JSON", "CSV", "PDF"]
                export_format = st.selectbox("Export Format", export_formats)

                if st.button("Export Results"):
                    if export_format == "JSON":
                        st.download_button(
                            label="Download JSON",
                            data=json.dumps(eval_results, indent=2),
                            file_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    elif export_format == "CSV":

                        flat_metrics = {}
                        for metric_name, metric_data in eval_results["metrics"].items():
                            if isinstance(metric_data, dict):
                                for k, v in metric_data.items():
                                    if isinstance(v, (int, float, str)):
                                        flat_metrics[f"{metric_name}_{k}"] = v

                        flat_data = {
                            "timestamp": eval_results["timestamp"],
                            "model": eval_results["model"],
                            "task_type": eval_results["task_type"],
                            "template": eval_results["template"],
                            "overall_score": eval_results.get("overall_score", "N/A"),
                            **flat_metrics
                        }

                        csv_df = pd.DataFrame([flat_data])
                        csv_string = csv_df.to_csv(index=False)

                        st.download_button(
                            label="Download CSV",
                            data=csv_string,
                            file_name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.info(
                            "PDF export would be implemented in a production version.")

    with tabs[1]:
        st.subheader("Human Evaluation")

        st.markdown("""
        Human evaluation involves manual assessment of text quality across various dimensions.
        This provides qualitative insights that automated metrics may miss.
        """)

        if "evaluations" not in st.session_state or not st.session_state.evaluations:
            st.info("No human evaluations have been submitted yet.")

            st.markdown("### Create New Evaluation")

            if "generation_history" not in st.session_state or not st.session_state.generation_history:
                st.warning(
                    "No generations available to evaluate. Generate text first.")
            else:

                generation_options = [
                    f"{g['timestamp']} - {g['template_name']}"
                    for g in st.session_state.generation_history
                ]

                selected_generation_option = st.selectbox(
                    "Select Generation to Evaluate",
                    options=generation_options
                )

                if selected_generation_option:
                    selected_timestamp = selected_generation_option.split(
                        " - ")[0]
                    selected_generation = next(
                        (g for g in st.session_state.generation_history if g["timestamp"] == selected_timestamp),
                        None
                    )

                    if selected_generation:
                        st.markdown("### Generation to Evaluate")
                        st.markdown(
                            f"**Template:** {selected_generation['template_name']}")
                        st.markdown(
                            f"**Model:** {selected_generation['model']}")

                        st.markdown("### Generated Text")
                        st.text_area(
                            "Text to Evaluate",
                            value=selected_generation["generated_text"],
                            height=200,
                            disabled=True
                        )

                        st.info(
                            "Use the Human Evaluation tab in the Text Generation page to evaluate this text.")

                        if st.button("Go to Evaluation Form"):

                            st.session_state.current_generation = selected_generation
                            st.info(
                                "In a real app, this would navigate to the evaluation form page.")
        else:

            st.markdown("### Submitted Evaluations")

            eval_data = [
                {
                    "Timestamp": e["timestamp"],
                    "Model": e["model"],
                    "Task Type": e["task_type"].capitalize(),
                    "Overall Score": f"{e['overall_score']:.2f}/5.0",
                }
                for e in st.session_state.evaluations
            ]

            eval_df = pd.DataFrame(eval_data)
            st.dataframe(eval_df, use_container_width=True, hide_index=True)

            eval_timestamps = [e["timestamp"]
                               for e in st.session_state.evaluations]
            selected_eval_timestamp = st.selectbox(
                "Select Evaluation to View",
                options=eval_timestamps,
                format_func=lambda x: f"{x} - {next((e['model'] for e in st.session_state.evaluations if e['timestamp'] == x), 'Unknown')}"
            )

            selected_eval = next(
                (e for e in st.session_state.evaluations if e["timestamp"] == selected_eval_timestamp), None)

            if selected_eval:

                display_evaluation_results(selected_eval)

                if "generation_id" in selected_eval and selected_eval["generation_id"]:
                    associated_generation = next(
                        (g for g in st.session_state.generation_history if g["timestamp"] == selected_eval["generation_id"]),
                        None
                    )

                    if associated_generation:
                        with st.expander("View Generated Text", expanded=False):
                            st.markdown("### Generated Text")
                            st.text_area(
                                "Text",
                                value=associated_generation["generated_text"],
                                height=200,
                                disabled=True
                            )

    with tabs[2]:
        st.subheader("Comparative Evaluation")

        st.markdown("""
        Comparative evaluation allows you to compare the performance of different models or prompts on the same task.
        This helps identify strengths and weaknesses across different approaches.
        """)

        if (
            "automatic_evaluations" not in st.session_state or
            not st.session_state.automatic_evaluations or
            len(st.session_state.automatic_evaluations) < 2
        ):
            st.warning(
                "Need at least two automatic evaluations to compare. Run more evaluations first.")
            return

        st.markdown("### Select Evaluations to Compare")

        evaluations = st.session_state.automatic_evaluations

        eval_data = [
            {
                "Timestamp": e["timestamp"],
                "Model": e["model"],
                "Task": e["task_type"].capitalize() if "task_type" in e else "Unknown",
                "Template": e["template"],
                "Overall Score": f"{e.get('overall_score', 0):.2f}",
            }
            for e in evaluations
        ]

        eval_df = pd.DataFrame(eval_data)
        st.dataframe(eval_df, use_container_width=True, hide_index=True)

        eval_timestamps = [e["timestamp"] for e in evaluations]
        selected_evals = st.multiselect(
            "Select Evaluations to Compare",
            options=eval_timestamps,
            format_func=lambda x: f"{x} - {next((e['model'] for e in evaluations if e['timestamp'] == x), 'Unknown')}"
        )

        if len(selected_evals) < 2:
            st.info("Select at least two evaluations to compare.")
        else:

            selected_evaluation_objects = [
                e for e in evaluations if e["timestamp"] in selected_evals
            ]

            st.markdown("### Comparison Results")

            all_metrics = set()
            for eval_obj in selected_evaluation_objects:
                all_metrics.update(eval_obj["metrics"].keys())

            common_metrics = list(all_metrics)

            comparison_data = {}

            for metric in common_metrics:
                metric_values = {}

                for eval_obj in selected_evaluation_objects:
                    eval_name = f"{eval_obj['model']} - {eval_obj['template']}"

                    if metric in eval_obj["metrics"]:
                        metric_data = eval_obj["metrics"][metric]

                        if isinstance(metric_data, dict) and "score" in metric_data:
                            metric_values[eval_name] = metric_data["score"]
                        elif isinstance(metric_data, dict) and "f1" in metric_data:
                            metric_values[eval_name] = metric_data["f1"]
                        elif metric == "ROUGE" and isinstance(metric_data, dict) and "rouge-l" in metric_data:
                            metric_values[eval_name] = metric_data["rouge-l"]

                        elif metric == "Length" and isinstance(metric_data, dict) and "words" in metric_data:
                            metric_values[eval_name] = metric_data["words"]

                        elif metric == "Readability" and isinstance(metric_data, dict) and "flesch_reading_ease" in metric_data:
                            metric_values[eval_name] = metric_data["flesch_reading_ease"]

                if metric_values:
                    comparison_data[metric] = metric_values

            for metric, values in comparison_data.items():
                st.markdown(f"#### {metric}")

                df = pd.DataFrame(
                    {"Model-Template": list(values.keys()), "Score": list(values.values())})

                st.bar_chart(df.set_index("Model-Template"))

            st.markdown("#### Overall Score Comparison")

            overall_scores = {}
            for eval_obj in selected_evaluation_objects:
                eval_name = f"{eval_obj['model']} - {eval_obj['template']}"
                if "overall_score" in eval_obj:
                    overall_scores[eval_name] = eval_obj["overall_score"]

            if overall_scores:
                overall_df = pd.DataFrame(
                    {"Model-Template": list(overall_scores.keys()), "Score": list(overall_scores.values())})
                st.bar_chart(overall_df.set_index("Model-Template"))

            st.markdown("### Summary and Recommendations")

            best_for_metrics = {}
            for metric, values in comparison_data.items():
                best_model = max(values.items(), key=lambda x: x[1])[0]
                best_for_metrics[metric] = best_model

            st.markdown("#### Best Model-Template by Metric")

            best_df = pd.DataFrame({"Metric": list(best_for_metrics.keys(
            )), "Best Model-Template": list(best_for_metrics.values())})
            st.dataframe(best_df, use_container_width=True, hide_index=True)

            if overall_scores:
                best_overall = max(overall_scores.items(),
                                   key=lambda x: x[1])[0]
                st.markdown(f"#### Overall Recommendation")
                st.info(
                    f"The best overall performance was achieved by: **{best_overall}**")

if __name__ == "__main__":

    st.set_page_config(
        page_title="Evaluation - Insurance LLM Framework",
        page_icon="🔍",
        layout="wide"
    )

    if "generation_history" not in st.session_state:

        st.session_state.generation_history = [
            {
                "model": "llama2-7b",
                "prompt": "Summarize the following insurance policy...",
                "generated_text": "This insurance policy provides coverage for property damage and liability...",
                "parameters": {"temperature": 0.7, "max_tokens": 1024},
                "template_id": "policy_summary",
                "template_name": "Policy Summarization",
                "timestamp": "2023-10-20 14:30:45",
                "task_type": "summarization",
            },
            {
                "model": "mistral-7b-instruct",
                "prompt": "Analyze the risk factors in the following scenario...",
                "generated_text": "Risk Assessment Analysis:\n\n1. Primary Risk Factors:\n   - Commercial building age...",
                "parameters": {"temperature": 0.3, "max_tokens": 1536},
                "template_id": "risk_analysis",
                "template_name": "Risk Assessment",
                "timestamp": "2023-10-21 09:15:22",
                "task_type": "analysis",
            }
        ]

    if "evaluations" not in st.session_state:

        st.session_state.evaluations = [
            {
                "timestamp": "2023-10-20 15:45:30",
                "model": "llama2-7b",
                "task_type": "summarization",
                "ratings": {
                    "relevance": 4,
                    "accuracy": 4,
                    "coherence": 5,
                    "completeness": 3,
                    "brevity": 4,
                    "key_points": 4,
                    "simplification": 5
                },
                "overall_score": 4.14,
                "comments": "Good summary that captures the essential information in simple language.",
                "generation_id": "2023-10-20 14:30:45"
            }
        ]

    if "automatic_evaluations" not in st.session_state:

        st.session_state.automatic_evaluations = [
            {
                "timestamp": "2023-10-20 16:00:12",
                "model": "llama2-7b",
                "task_type": "summarization",
                "template": "Policy Summarization",
                "metrics": {
                    "Readability": {
                        "flesch_reading_ease": 68.5,
                        "flesch_kincaid_grade": 7.8,
                        "dale_chall_readability": 6.9,
                        "score": 0.78
                    },
                    "Coherence": {
                        "score": 0.82,
                        "explanation": "Text demonstrates strong logical flow and consistent topic maintenance."
                    },
                    "Clarity": {
                        "score": 0.85,
                        "explanation": "Insurance concepts explained clearly for general audience."
                    },
                    "Completeness": {
                        "score": 0.75,
                        "explanation": "Addresses most key aspects of the policy."
                    }
                },
                "overall_score": 0.8
            },
            {
                "timestamp": "2023-10-21 10:30:45",
                "model": "mistral-7b-instruct",
                "task_type": "analysis",
                "template": "Risk Assessment",
                "metrics": {
                    "Readability": {
                        "flesch_reading_ease": 62.3,
                        "flesch_kincaid_grade": 9.2,
                        "dale_chall_readability": 7.5,
                        "score": 0.72
                    },
                    "Coherence": {
                        "score": 0.88,
                        "explanation": "Excellent logical flow with strong argumentation."
                    },
                    "Depth": {
                        "score": 0.91,
                        "explanation": "Thorough analysis of all relevant factors."
                    },
                    "Term Accuracy": {
                        "score": 0.94,
                        "explanation": "Insurance terms used correctly and appropriately."
                    }
                },
                "overall_score": 0.86
            }
        ]

    render()
