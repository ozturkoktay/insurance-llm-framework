"""
Model Selector Component for the Insurance LLM Framework.

This module provides UI components for selecting and displaying information about language models.
"""

import streamlit as st
import pandas as pd
import os
import logging
import json
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)


def get_available_models() -> List[Dict[str, Any]]:
    """
    Get a list of available models.

    In a production app, this would query the model_loader module.
    For this demo, we'll use a hard-coded list.

    Returns:
        List[Dict[str, Any]]: List of model information dictionaries
    """
    # Models available for selection
    models = [
        {
            "name": "llama2-7b",
            "family": "llama2",
            "size": "7B",
            "type": "local",
            "quantization": "4-bit",
            "context_length": 4096,
            "strengths": ["General text generation", "Instruction following"],
            "recommended_tasks": ["Policy summarization", "Claim categorization"],
            "description": "A 7 billion parameter model from Meta, fine-tuned for instruction following.",
        },
        {
            "name": "llama2-13b",
            "family": "llama2",
            "size": "13B",
            "type": "local",
            "quantization": "4-bit",
            "context_length": 4096,
            "strengths": ["Enhanced reasoning", "Better instruction following"],
            "recommended_tasks": ["Risk assessment", "Policy analysis"],
            "description": "A 13 billion parameter model from Meta with improved reasoning capabilities.",
        },
        {
            "name": "mistral-7b-instruct",
            "family": "mistral",
            "size": "7B",
            "type": "local",
            "quantization": "4-bit",
            "context_length": 8192,
            "strengths": ["Strong instruction following", "Long context understanding"],
            "recommended_tasks": ["Customer communication", "Claim response drafting"],
            "description": "A 7 billion parameter model from Mistral AI with excellent instruction following and context handling.",
        },
        {
            "name": "falcon-7b",
            "family": "falcon",
            "size": "7B",
            "type": "local",
            "quantization": "4-bit",
            "context_length": 2048,
            "strengths": ["Factual knowledge", "Efficient inference"],
            "recommended_tasks": ["Regulatory compliance", "FAQ generation"],
            "description": "A 7 billion parameter model from Technology Innovation Institute with strong factual knowledge.",
        },
        {
            "name": "gpt-4-turbo",
            "family": "gpt",
            "size": "Unknown",
            "type": "api",
            "quantization": "N/A",
            "context_length": 128000,
            "strengths": ["Advanced reasoning", "Complex instruction following", "Multimodal capabilities"],
            "recommended_tasks": ["Complex policy analysis", "Risk modeling", "Claim fraud detection"],
            "description": "OpenAI's advanced LLM with very long context window and strong reasoning.",
        },
        {
            "name": "claude-3-sonnet",
            "family": "claude",
            "size": "Unknown",
            "type": "api",
            "quantization": "N/A",
            "context_length": 180000,
            "strengths": ["Nuanced understanding", "Document analysis", "Long context reasoning"],
            "recommended_tasks": ["Document analysis", "Compliance checking", "Complex case evaluation"],
            "description": "Anthropic's powerful assistant model with extremely long context window.",
        },
    ]

    return models


def model_selector() -> Optional[str]:
    """
    Display a UI for selecting a language model.

    Returns:
        Optional[str]: Name of the selected model, or None if no model is selected
    """
    # Get available models
    models = get_available_models()

    # Create tabs for local and API models
    model_types = ["Local Models", "API Models", "Custom Models"]
    tabs = st.tabs(model_types)

    selected_model = None

    # Local Models Tab
    with tabs[0]:
        local_models = [m for m in models if m["type"] == "local"]

        if not local_models:
            st.info(
                "No local models available. You can add models by placing them in the models directory.")
        else:
            # Filter options
            col1, col2 = st.columns([1, 1])
            with col1:
                families = list(set(m["family"] for m in local_models))
                selected_family = st.multiselect(
                    "Filter by Model Family",
                    options=families,
                    default=None,
                    help="Select model families to filter the list"
                )

            with col2:
                sizes = list(set(m["size"] for m in local_models))
                selected_size = st.multiselect(
                    "Filter by Model Size",
                    options=sizes,
                    default=None,
                    help="Select model sizes to filter the list"
                )

            # Apply filters
            filtered_models = local_models
            if selected_family:
                filtered_models = [
                    m for m in filtered_models if m["family"] in selected_family]
            if selected_size:
                filtered_models = [
                    m for m in filtered_models if m["size"] in selected_size]

            if not filtered_models:
                st.warning("No models match the selected filters.")
            else:
                # Create a table of models
                model_df = pd.DataFrame([
                    {
                        "Model": str(m["name"]),
                        "Family": str(m["family"].capitalize()),
                        "Size": str(m["size"]),
                        "Context Length": f"{m['context_length']} tokens",
                        "Quantization": str(m["quantization"]),
                    }
                    for m in filtered_models
                ])

                st.dataframe(model_df, use_container_width=True,
                             hide_index=True)

                # Model selection dropdown
                local_model_names = [m["name"] for m in filtered_models]
                selected_model_name = st.selectbox(
                    "Select a Model",
                    options=local_model_names,
                    help="Choose a model to use for your tasks"
                )

                # If a model is selected, update the selected_model variable
                if selected_model_name:
                    selected_model = selected_model_name

                    # Display the selected model info
                    selected_model_info = next(
                        (m for m in models if m["name"] == selected_model_name), None)
                    if selected_model_info:
                        st.markdown(f"### {selected_model_info['name']}")
                        st.markdown(selected_model_info["description"])

                        st.markdown("#### Recommended for:")
                        for task in selected_model_info["recommended_tasks"]:
                            st.markdown(f"- {task}")

    # API Models Tab
    with tabs[1]:
        api_models = [m for m in models if m["type"] == "api"]

        if not api_models:
            st.info(
                "No API models configured. You need to add API keys in the settings.")
        else:
            # Create a table of models
            model_df = pd.DataFrame([
                {
                    "Model": m["name"],
                    "Provider": m["family"].capitalize(),
                    "Context Length": f"{m['context_length']} tokens",
                    "API Key Configured": "✅" if st.session_state.get(f"{m['family']}_api_key") else "❌",
                }
                for m in api_models
            ])

            st.dataframe(model_df, use_container_width=True, hide_index=True)

            # Check if API keys are configured
            api_keys_configured = any(st.session_state.get(
                f"{m['family']}_api_key") for m in api_models)

            if not api_keys_configured:
                st.warning(
                    "No API keys configured. Please add them in the Settings page.")
                if st.button("Go to Settings"):
                    # This would navigate to the settings page in a real app
                    st.info(
                        "In a real app, this would navigate to the Settings page.")

            # Model selection dropdown
            api_model_names = [m["name"] for m in api_models]
            selected_api_model = st.selectbox(
                "Select an API Model",
                options=api_model_names,
                help="Choose an API-based model to use for your tasks"
            )

            # If a model is selected, update the selected_model variable
            if selected_api_model:
                selected_model = selected_api_model

                # Display the selected model info
                selected_model_info = next(
                    (m for m in models if m["name"] == selected_api_model), None)
                if selected_model_info:
                    st.markdown(f"### {selected_model_info['name']}")
                    st.markdown(selected_model_info["description"])

                    st.markdown("#### Recommended for:")
                    for task in selected_model_info["recommended_tasks"]:
                        st.markdown(f"- {task}")

    # Custom Models Tab
    with tabs[2]:
        st.markdown("### Custom Model Configuration")
        st.markdown("""
        You can configure a custom model by providing its details and connection information.
        This is useful for:
        - Self-hosted models not in the standard list
        - Models served via custom APIs
        - Experimental or fine-tuned models
        """)

        # Custom model form
        with st.form("custom_model_form"):
            custom_name = st.text_input(
                "Model Name", placeholder="my-custom-model")

            col1, col2 = st.columns(2)
            with col1:
                custom_family = st.text_input(
                    "Model Family/Type", placeholder="llama2")
                custom_size = st.text_input("Parameter Size", placeholder="7B")

            with col2:
                custom_type = st.selectbox("Deployment Type", options=[
                                           "local", "api", "vllm", "text-generation-webui"])
                custom_context = st.number_input(
                    "Context Length (tokens)", min_value=512, max_value=1000000, value=4096)

            custom_endpoint = st.text_input(
                "API Endpoint (if applicable)", placeholder="http://localhost:8000/v1")

            custom_description = st.text_area(
                "Model Description", placeholder="Description of the model's capabilities...")

            submitted = st.form_submit_button("Add Custom Model")

            if submitted:
                if custom_name and custom_family and custom_type:
                    # In a real app, would save this to a config file or database
                    st.success(
                        f"Custom model '{custom_name}' added successfully!")
                    logger.info(f"Custom model '{custom_name}' added")

                    # Create a custom model dict
                    custom_model = {
                        "name": custom_name,
                        "family": custom_family,
                        "size": custom_size,
                        "type": custom_type,
                        "quantization": "custom",
                        "context_length": custom_context,
                        "strengths": [],
                        "recommended_tasks": [],
                        "description": custom_description,
                        "endpoint": custom_endpoint
                    }

                    # In a production app, would save this and update the model list
                    # For demo, we'll just select this model
                    selected_model = custom_name
                else:
                    st.error(
                        "Please fill in all required fields (Name, Family, Type).")

    return selected_model


def display_model_info(model_name: str) -> None:
    """
    Display detailed information about a model.

    Args:
        model_name: Name of the model to display information for
    """
    # Get model information
    models = get_available_models()
    model_info = next((m for m in models if m["name"] == model_name), None)

    if not model_info:
        st.warning(f"Model information for '{model_name}' not found.")
        return

    # Display model information
    st.markdown(f"## {model_info['name']}")

    # Basic info card
    st.markdown("### Model Information")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.metric("Model Family", model_info["family"].capitalize())
        st.metric("Parameter Size", model_info["size"])

    with col2:
        st.metric("Deployment Type", model_info["type"].capitalize())
        st.metric("Quantization", model_info["quantization"])

    with col3:
        st.metric("Context Length", f"{model_info['context_length']} tokens")

    # Model description
    st.markdown("### Description")
    st.markdown(model_info["description"])

    # Model strengths and recommended tasks
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Strengths")
        for strength in model_info["strengths"]:
            st.markdown(f"- {strength}")

    with col2:
        st.markdown("### Recommended for")
        for task in model_info["recommended_tasks"]:
            st.markdown(f"- {task}")

    # Usage examples card
    with st.expander("Usage Examples", expanded=False):
        st.markdown("#### Example Prompts")

        example_prompts = [
            {
                "title": "Policy Summarization",
                "prompt": f"Summarize the following insurance policy in simple terms, highlighting the key coverage, exclusions, and limitations:\n\n[POLICY TEXT]",
                "suitable_for": ["llama2-13b", "mistral-7b-instruct", "gpt-4-turbo"]
            },
            {
                "title": "Claim Response",
                "prompt": f"Draft a response to the following insurance claim, explaining the coverage decision and next steps:\n\n[CLAIM DETAILS]",
                "suitable_for": ["mistral-7b-instruct", "gpt-4-turbo", "claude-3-sonnet"]
            },
            {
                "title": "Risk Assessment",
                "prompt": f"Analyze the following property details and identify potential risk factors that should be considered for insurance coverage:\n\n[PROPERTY DETAILS]",
                "suitable_for": ["llama2-13b", "falcon-7b", "gpt-4-turbo"]
            }
        ]

        # Filter examples suitable for this model
        suitable_examples = [
            ex for ex in example_prompts if model_name in ex["suitable_for"]]

        if suitable_examples:
            for ex in suitable_examples:
                st.markdown(f"##### {ex['title']}")
                st.code(ex["prompt"], language="text")
        else:
            # Show generic examples
            for ex in example_prompts[:1]:
                st.markdown(f"##### {ex['title']}")
                st.code(ex["prompt"], language="text")

    # Model configuration recommendations
    with st.expander("Recommended Configuration", expanded=False):
        st.markdown("### Recommended Parameters for Different Tasks")

        tasks = [
            {
                "task": "Policy Summarization",
                "temperature": 0.3,
                "max_tokens": 1024,
                "top_p": 0.9,
                "description": "Lower temperature for more factual and concise summaries."
            },
            {
                "task": "Customer Communications",
                "temperature": 0.7,
                "max_tokens": 512,
                "top_p": 0.95,
                "description": "Moderate temperature for natural-sounding but professional responses."
            },
            {
                "task": "Risk Assessment",
                "temperature": 0.2,
                "max_tokens": 1536,
                "top_p": 0.8,
                "description": "Low temperature for factual, detailed analysis."
            },
            {
                "task": "Creative Content",
                "temperature": 0.9,
                "max_tokens": 2048,
                "top_p": 1.0,
                "description": "Higher temperature for more varied and creative outputs."
            }
        ]

        # Create a table of recommendations
        task_df = pd.DataFrame(tasks)
        st.dataframe(task_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    # For testing the component in isolation
    st.set_page_config(
        page_title="Model Selector - Insurance LLM Framework",
        page_icon="🤖",
        layout="wide"
    )

    st.title("Model Selector Component Test")

    # Initialize session state for testing
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = None
    if "anthropic_api_key" not in st.session_state:
        st.session_state.anthropic_api_key = None

    # Test the component
    selected_model = model_selector()

    if selected_model:
        st.success(f"Selected model: {selected_model}")

        # Test displaying model info
        display_model_info(selected_model)
