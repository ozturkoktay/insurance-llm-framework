"""
Home page for the Insurance LLM Framework.

This module provides the home/welcome page for the application.
"""

import streamlit as st
import pandas as pd
import os
from typing import Dict, Any, Optional, List


def render():
    """Render the home page."""
    st.title("🏥 Insurance LLM Framework")
    st.markdown(
        "### An Open-Source Prompt Engineering and Evaluation Framework for Insurance Domain Applications")

    # Introduction
    st.markdown("""
    Welcome to the Insurance LLM Framework, a specialized toolkit designed to help insurance professionals leverage 
    the power of Large Language Models (LLMs) for various insurance tasks.
    
    This framework provides tools for designing effective prompts, evaluating model outputs, and benchmarking 
    different models specifically for insurance applications.
    """)

    # Main features display
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        st.markdown("### 🔍 Prompt Engineering")
        st.markdown("""
        - Design domain-specific prompts
        - Choose from various prompt strategies
        - Access a library of insurance templates
        - Customize prompts for your needs
        """)

    with col2:
        st.markdown("### 🧪 Model Evaluation")
        st.markdown("""
        - Evaluate outputs with specialized metrics
        - Conduct human evaluations with rubrics
        - Run benchmarks for standardized testing
        - Compare different models head-to-head
        """)

    with col3:
        st.markdown("### 📄 Insurance Tasks")
        st.markdown("""
        - Policy summarization
        - Claim response drafting
        - Risk assessment reporting
        - Customer communications
        - Compliance checking
        """)

    # Getting started section
    st.markdown("## Getting Started")
    st.markdown("""
    To get started with the Insurance LLM Framework:
    
    1. Go to the **Model Selection** page to load a language model
    2. Visit the **Prompt Engineering** page to select or create a prompt template
    3. Use the **Text Generation** page to generate insurance content
    4. Evaluate the results using the **Evaluation** page
    
    For more detailed information, check the documentation pages:
    """)

    # Documentation links
    doc_col1, doc_col2, doc_col3 = st.columns([1, 1, 1])

    with doc_col1:
        with st.expander("📋 Use Cases", expanded=False):
            st.markdown("""
            Learn about the specific insurance use cases supported by the framework:
            - Policy summarization
            - Claim response generation
            - Customer inquiry handling
            - Risk assessment
            - And more...
            """)
            if os.path.exists("docs/use_cases.md"):
                st.markdown("[View Use Cases Documentation]()")

    with doc_col2:
        with st.expander("✏️ Prompt Engineering Guide", expanded=False):
            st.markdown("""
            Discover how to design effective prompts for insurance tasks:
            - Zero-shot, few-shot, and chain-of-thought strategies
            - Best practices for insurance-specific prompts
            - Template customization guides
            - Advanced techniques
            """)
            if os.path.exists("docs/prompt_engineering.md"):
                st.markdown("[View Prompt Engineering Guide]()")

    with doc_col3:
        with st.expander("📊 Evaluation Guide", expanded=False):
            st.markdown("""
            Learn how to evaluate LLM outputs for insurance content:
            - Automated metrics explanation
            - Human evaluation framework
            - Benchmark creation
            - Results interpretation
            """)
            if os.path.exists("docs/evaluation.md"):
                st.markdown("[View Evaluation Guide]()")

    # Display supported models
    st.markdown("## Supported Models")

    # Check if model_loader is available to get supported models
    try:
        from models.model_loader import get_supported_models
        supported_models = get_supported_models()

        # Group models by type/family
        model_families = {}
        for model in supported_models:
            family = model.split("-")[0]
            if family not in model_families:
                model_families[family] = []
            model_families[family].append(model)

        # Display grouped models
        for family, models in model_families.items():
            with st.expander(f"{family.capitalize()} Models", expanded=False):
                for model in sorted(models):
                    st.markdown(f"- `{model}`")
    except:
        # Fallback if we can't import the module
        st.markdown("""
        The framework supports a variety of open-source models including:
        - LLaMA-2 (7B, 13B)
        - Mistral (7B)
        - Falcon (7B, 40B)
        - And more...
        """)

    # Footer
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    The Insurance LLM Framework is an open-source project designed to make LLMs more accessible
    and effective for insurance industry applications. It focuses on domain-specific prompt engineering,
    evaluation metrics tailored to insurance needs, and benchmarks that reflect real insurance tasks.
    """)

    # Version info
    st.caption("Version 0.1.0 • MIT License")


if __name__ == "__main__":
    # For testing the page in isolation
    st.set_page_config(
        page_title="Insurance LLM Framework",
        page_icon="🏥",
        layout="wide"
    )
    render()
