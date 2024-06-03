"""
Sidebar navigation component for the Insurance LLM Framework.

This module provides the sidebar navigation used throughout the application.
"""

import streamlit as st
from typing import Callable


def create_sidebar():
    """
    Create the application sidebar with navigation options.

    Returns:
        The selected navigation option
    """
    with st.sidebar:
        st.title("Insurance LLM Framework")

        st.markdown("---")

        # Navigation
        selected = st.radio(
            "Navigate:",
            options=[
                "Home",
                "Model Selection",
                "Prompt Engineering",
                "Text Generation",
                "Evaluation",
                "Benchmarks",
                "Model Comparison",
                "Settings"
            ],
        )

        st.markdown("---")

        # Show model status if a model is loaded
        if "current_model" in st.session_state and st.session_state.current_model:
            model_info = st.session_state.current_model
            st.success(f"Model loaded: {model_info['name']}")
            st.info(f"Type: {model_info['type']}")
            if "quantization" in model_info:
                st.info(f"Quantization: {model_info['quantization']}")
        else:
            st.warning("No model loaded")

        # Show current prompt template if selected
        if "current_template" in st.session_state and st.session_state.current_template:
            template = st.session_state.current_template
            st.success(f"Template: {template['name']}")
            st.info(f"Task: {template['task_type']}")

        st.markdown("---")

        # App info
        st.markdown("### About")
        st.info(
            "An open-source prompt engineering and evaluation framework "
            "for insurance domain applications."
        )

        # GitHub link
        st.markdown(
            "[GitHub Repository](https://github.com/yourusername/insurance-llm-framework)")

    return selected


def display_logo():
    """Display the application logo in the sidebar."""
    with st.sidebar:
        # You can add a logo image here if available
        st.markdown("# 🏥 Insurance LLM Framework")


def set_page_config():
    """Set the page configuration for the Streamlit app."""
    st.set_page_config(
        page_title="Insurance LLM Framework",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
