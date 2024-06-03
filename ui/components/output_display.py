"""
Output Display Component for the Insurance LLM Framework.

This module provides UI components for displaying and evaluating generated text.
"""

import streamlit as st
import pandas as pd
import os
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def display_generated_text(
    prompt: str,
    generated_text: str,
    model_name: str,
    parameters: Dict[str, Any] = None
) -> None:
    """
    Display generated text with options for evaluation and export.

    Args:
        prompt: The input prompt
        generated_text: The generated text
        model_name: Name of the model used
        parameters: Generation parameters
    """
    # Display prompt and generated text in tabs
    input_output_tabs = st.tabs(
        ["Generated Output", "Input Prompt", "Human Evaluation"])

    with input_output_tabs[0]:
        st.markdown("### Generated Output")
        st.markdown(f"*Generated using {model_name}*")

        # Display the generated text
        st.text_area(
            "Output",
            value=generated_text,
            height=300,
            disabled=True
        )

        # Export options
        export_col1, export_col2, export_col3 = st.columns([1, 1, 1])

        with export_col1:
            # Copy to clipboard button
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                try:
                    st.write("Text copied to clipboard!")
                    st.toast("Text copied to clipboard!")
                except:
                    st.error("Failed to copy to clipboard")

        with export_col2:
            # Save as text file
            if st.button("💾 Save as Text File", use_container_width=True):
                # In a real app, this would actually save the file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"generated_text_{timestamp}.txt"

                # For demo, we'll just show what would happen
                st.success(f"Text would be saved as {filename}")
                st.download_button(
                    label="Download Text",
                    data=generated_text,
                    file_name=filename,
                    mime="text/plain"
                )

        with export_col3:
            # Save to document library
            if st.button("📚 Save to Library", use_container_width=True):
                # In a real app, this would save to a document library
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                task_type = "output"

                if "current_generation" in st.session_state:
                    task_type = st.session_state.current_generation.get(
                        "task_type", "output")

                filename = f"{task_type}_{timestamp}.txt"

                # Create directory if it doesn't exist
                save_dir = "data/saved_documents"
                os.makedirs(save_dir, exist_ok=True)

                # Save the file
                try:
                    with open(os.path.join(save_dir, filename), "w") as f:
                        f.write(generated_text)

                    st.success(f"Saved to document library as {filename}")
                    logger.info(f"Document saved to library: {filename}")
                except Exception as e:
                    st.error(f"Failed to save document: {str(e)}")
                    logger.error(f"Failed to save document: {str(e)}")

    with input_output_tabs[1]:
        st.markdown("### Input Prompt")

        # Display the prompt
        st.text_area(
            "Prompt",
            value=prompt,
            height=300,
            disabled=True
        )

        # Display parameters if available
        if parameters:
            st.markdown("### Generation Parameters")

            # Create a more readable display of parameters
            param_df = pd.DataFrame(
                [{k: str(v) for k, v in parameters.items()}])
            st.dataframe(param_df, use_container_width=True, hide_index=True)

    with input_output_tabs[2]:
        st.markdown("### Human Evaluation")

        # Create a form for human evaluation
        with st.form("human_evaluation_form"):
            st.markdown("""
            Please evaluate the quality of the generated text for the given task.
            Rate each aspect on a scale of 1-5, where 1 is poor and 5 is excellent.
            """)

            # Determine criteria based on task type
            task_type = "general"
            if "current_generation" in st.session_state:
                task_type = st.session_state.current_generation.get(
                    "task_type", "general")

            # Default evaluation criteria
            criteria = {
                "relevance": "Relevance to the prompt",
                "accuracy": "Factual accuracy",
                "coherence": "Coherence and readability",
                "completeness": "Completeness of response"
            }

            # Task-specific criteria
            if task_type == "summarization":
                criteria.update({
                    "brevity": "Appropriate brevity",
                    "key_points": "Captures key points",
                    "simplification": "Simplifies complex concepts"
                })
            elif task_type == "analysis":
                criteria.update({
                    "depth": "Depth of analysis",
                    "reasoning": "Quality of reasoning",
                    "insights": "Valuable insights"
                })
            elif "claim" in task_type:
                criteria.update({
                    "clarity": "Clarity of decision",
                    "empathy": "Empathetic tone",
                    "actionable": "Provides clear next steps"
                })

            # Display sliders for each criterion
            ratings = {}
            for criterion, description in criteria.items():
                ratings[criterion] = st.slider(
                    f"{description}",
                    min_value=1,
                    max_value=5,
                    value=3,
                    help=f"Rate the {criterion} from 1 (poor) to 5 (excellent)"
                )

            # Additional comments
            comments = st.text_area(
                "Additional Comments",
                placeholder="Enter any additional feedback or comments..."
            )

            # Submit button
            submitted = st.form_submit_button(
                "Submit Evaluation", use_container_width=True)

            if submitted:
                # Calculate overall score
                overall_score = sum(ratings.values()) / len(ratings)

                # Store evaluation in session state
                if "evaluations" not in st.session_state:
                    st.session_state.evaluations = []

                evaluation_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "model": model_name,
                    "task_type": task_type,
                    "ratings": ratings,
                    "overall_score": overall_score,
                    "comments": comments,
                    "generation_id": st.session_state.current_generation["timestamp"] if "current_generation" in st.session_state else None
                }

                st.session_state.evaluations.append(evaluation_data)

                # Show success message with score
                st.success(
                    f"Evaluation submitted. Overall score: {overall_score:.2f}/5.0")
                logger.info(
                    f"Human evaluation submitted with score {overall_score:.2f}")

                # Update the current generation with evaluation data
                if "current_generation" in st.session_state:
                    st.session_state.current_generation["human_evaluation"] = evaluation_data

                # Show detailed results
                st.markdown("### Evaluation Results")

                # Create a radar chart or bar chart of ratings
                ratings_df = pd.DataFrame({
                    "Criterion": list(ratings.keys()),
                    "Rating": list(ratings.values())
                })

                st.bar_chart(ratings_df.set_index("Criterion"))

        # Display previous evaluations if available
        if "evaluations" in st.session_state and st.session_state.evaluations:
            st.markdown("### Previous Evaluations")

            # Get evaluations for the current generation
            current_gen_id = st.session_state.current_generation[
                "timestamp"] if "current_generation" in st.session_state else None

            relevant_evals = [
                e for e in st.session_state.evaluations
                if e.get("generation_id") == current_gen_id
            ]

            if relevant_evals:
                # Display as a table
                eval_data = [
                    {
                        "Timestamp": e["timestamp"],
                        "Overall Score": f"{e['overall_score']:.2f}/5.0",
                        "Comments": e["comments"][:50] + "..." if len(e["comments"]) > 50 else e["comments"]
                    }
                    for e in relevant_evals
                ]

                eval_df = pd.DataFrame(eval_data)
                st.dataframe(eval_df, use_container_width=True,
                             hide_index=True)
            else:
                st.info("No previous evaluations for this generation.")

    # Automatic evaluation section
    with st.expander("Automatic Evaluation", expanded=False):
        st.markdown("### Automatic Metrics")
        st.info(
            "In a production system, this would display automatic evaluation metrics for the generated text.")

        # Create tabs for different automatic evaluation methods
        auto_eval_tabs = st.tabs(
            ["Basic Metrics", "Insurance-Specific", "Reference-Based"])

        with auto_eval_tabs[0]:
            # Basic metrics like length, readability
            st.markdown("#### Basic Text Metrics")

            # Sample metrics (would be calculated in a real system)
            basic_metrics = {
                "Length (chars)": len(generated_text),
                "Length (words)": len(generated_text.split()),
                "Avg Word Length": sum(len(w) for w in generated_text.split()) / max(len(generated_text.split()), 1),
                "Sentences": generated_text.count(".") + generated_text.count("!") + generated_text.count("?"),
            }

            # Display as metrics
            metrics_cols = st.columns(len(basic_metrics))
            for i, (metric, value) in enumerate(basic_metrics.items()):
                with metrics_cols[i]:
                    st.metric(metric, value)

            # Readability score (simulated)
            st.markdown("#### Readability")
            readability_score = 65  # Sample score, would be calculated in production
            st.progress(readability_score/100)
            st.caption(
                f"Flesch Reading Ease: {readability_score}/100 (Higher is more readable)")

        with auto_eval_tabs[1]:
            # Insurance-specific metrics
            st.markdown("#### Insurance Domain Metrics")

            # Sample domain metrics (would be calculated in a real system)
            domain_metrics = {
                "Insurance Term Usage": "Appropriate",
                "Regulatory Compliance": "High",
                "Customer Friendliness": "Medium-High",
                "Technical Accuracy": "High"
            }

            # Display as a table
            domain_df = pd.DataFrame(
                {"Metric": domain_metrics.keys(), "Rating": domain_metrics.values()})
            st.dataframe(domain_df, use_container_width=True, hide_index=True)

            # Domain-specific warnings or issues
            st.markdown("#### Domain Issues")
            # Sample, would be detected in production
            domain_issues = ["None detected"]

            for issue in domain_issues:
                st.markdown(f"- {issue}")

        with auto_eval_tabs[2]:
            # Reference-based metrics (if reference is available)
            st.markdown("#### Reference-Based Metrics")
            st.info(
                "To use reference-based metrics, upload a reference document to compare against.")

            # Reference text input
            reference_text = st.text_area(
                "Reference Text (optional)",
                placeholder="Enter a reference or 'gold standard' text to compare against...",
                height=150
            )

            if reference_text:
                # Sample reference-based metrics (would be calculated in a real system)
                ref_metrics = {
                    "BLEU Score": "0.75",
                    "ROUGE-L": "0.68",
                    "BERTScore": "0.82",
                    "Semantic Similarity": "0.79"
                }

                # Display as a table
                ref_df = pd.DataFrame(
                    {"Metric": ref_metrics.keys(), "Score": ref_metrics.values()})
                st.dataframe(ref_df, use_container_width=True, hide_index=True)
            else:
                st.warning(
                    "Enter reference text to calculate similarity metrics.")

        # Run evaluation button
        if st.button("Run Automatic Evaluation", use_container_width=True):
            st.info(
                "In a production system, this would calculate and display real metrics.")
            st.success("Automatic evaluation complete.")


def display_evaluation_results(evaluation_data: Dict[str, Any]) -> None:
    """
    Display detailed evaluation results.

    Args:
        evaluation_data: Dictionary containing evaluation results
    """
    st.markdown(
        f"### Evaluation for {evaluation_data.get('model', 'Unknown Model')}")

    # Display overall score
    st.metric("Overall Score",
              f"{evaluation_data.get('overall_score', 0):.2f}/5.0")

    # Display ratings by category
    if "ratings" in evaluation_data:
        ratings = evaluation_data["ratings"]

        # Create a DataFrame for the ratings
        ratings_df = pd.DataFrame({
            "Criterion": list(ratings.keys()),
            "Rating": list(ratings.values())
        })

        # Display as a bar chart
        st.bar_chart(ratings_df.set_index("Criterion"))

    # Display comments
    if "comments" in evaluation_data and evaluation_data["comments"]:
        st.markdown("### Comments")
        st.markdown(evaluation_data["comments"])

    # Display timestamp
    if "timestamp" in evaluation_data:
        st.caption(f"Evaluation performed at {evaluation_data['timestamp']}")


if __name__ == "__main__":
    # For testing the component in isolation
    st.set_page_config(
        page_title="Output Display - Insurance LLM Framework",
        page_icon="📊",
        layout="wide"
    )

    st.title("Output Display Component Test")

    # Sample prompt and output for testing
    sample_prompt = "Summarize the following insurance policy document:\n\nThis homeowner's insurance policy provides coverage for your dwelling, personal property, and liability..."

    sample_output = """This homeowner's insurance policy offers three main types of coverage:

1. Dwelling Protection: Coverage up to $300,000 for the physical structure of your home against covered perils such as fire, wind, and vandalism.

2. Personal Property Coverage: Protection up to $150,000 for your belongings inside the home, including furniture, appliances, and clothing.

3. Liability Coverage: Coverage up to $500,000 for legal and medical expenses if someone is injured on your property.

Key exclusions include flood damage (requires separate policy), intentional damage, normal wear and tear, and earth movement. A $1,000 deductible applies to most claims."""

    # Test the component
    display_generated_text(
        prompt=sample_prompt,
        generated_text=sample_output,
        model_name="llama2-7b",
        parameters={"temperature": 0.7, "max_tokens": 1024}
    )
