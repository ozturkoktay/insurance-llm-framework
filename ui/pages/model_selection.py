"""
Model Selection Page for the Insurance LLM Framework.

This module provides the interface for selecting and configuring language models.
"""

import streamlit as st
import logging
import os
from typing import Dict, Any, Optional, List
import time

# Import UI components
from ui.components.model_selector import model_selector, display_model_info
from ui.components.metrics_display import display_model_statistics

logger = logging.getLogger(__name__)


def render():
    """Render the model selection page."""
    st.title("🤖 Model Selection")

    with st.expander("ℹ️ About Model Selection", expanded=False):
        st.markdown("""
        This page allows you to select and configure language models for your insurance tasks.
        
        ### Features:
        - Browse and select from available local and remote models
        - Configure model parameters like temperature and max tokens
        - View model capabilities and recommendations for insurance tasks
        - Load multiple models for comparison
        """)

    # Sidebar metrics (if a model is loaded)
    if "active_model" in st.session_state and st.session_state.active_model:
        with st.sidebar:
            st.markdown("### Active Model")
            st.info(f"Currently using: **{st.session_state.active_model}**")

            if "model_config" in st.session_state:
                st.caption(
                    f"Temperature: {st.session_state.model_config.get('temperature', 'N/A')}")
                st.caption(
                    f"Max Tokens: {st.session_state.model_config.get('max_tokens', 'N/A')}")

    # Main content
    tabs = st.tabs(
        ["Model Selection", "Model Configuration", "Model Information"])

    with tabs[0]:
        # Model selection area
        selected_model = model_selector()

        # Load model button
        if selected_model:
            if st.button("Load Selected Model", use_container_width=True):
                with st.spinner(f"Loading {selected_model}..."):
                    try:
                        # Simulate loading for demonstration (in a real app, we'd load the actual model)
                        # In a production app, you'd call a function like:
                        # from models.model_loader import load_model
                        # model = load_model(selected_model, **model_config)

                        # For demo, we'll just set session state and wait
                        time.sleep(2)  # Simulate loading time

                        st.session_state.active_model = selected_model
                        st.session_state.model_config = {
                            "temperature": 0.7,
                            "max_tokens": 1024,
                            "top_p": 1.0,
                        }

                        st.success(
                            f"Model {selected_model} loaded successfully!")
                        logger.info(
                            f"Model {selected_model} loaded successfully")

                        # Force a rerun to update the sidebar
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
                        logger.error(
                            f"Error loading model {selected_model}: {str(e)}")

    with tabs[1]:
        st.subheader("Model Configuration")

        # Only show if a model is loaded or being configured
        model_name = None
        if "active_model" in st.session_state:
            model_name = st.session_state.active_model
        elif selected_model:
            model_name = selected_model

        if model_name:
            st.markdown(f"Configure parameters for **{model_name}**")

            # Get current config if available
            config = {}
            if "model_config" in st.session_state:
                config = st.session_state.model_config

            # Model parameters
            col1, col2 = st.columns(2)

            with col1:
                temperature = st.slider(
                    "Temperature",
                    min_value=0.0,
                    max_value=2.0,
                    value=config.get("temperature", 0.7),
                    step=0.1,
                    help="Higher values make output more random, lower values more deterministic"
                )

                top_p = st.slider(
                    "Top P",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.get("top_p", 1.0),
                    step=0.05,
                    help="Nucleus sampling parameter. 1.0 disables it."
                )

            with col2:
                max_tokens = st.slider(
                    "Max Tokens",
                    min_value=64,
                    max_value=4096,
                    value=config.get("max_tokens", 1024),
                    step=64,
                    help="Maximum number of tokens to generate"
                )

                presence_penalty = st.slider(
                    "Presence Penalty",
                    min_value=-2.0,
                    max_value=2.0,
                    value=config.get("presence_penalty", 0.0),
                    step=0.1,
                    help="Positive values penalize tokens that have already appeared in the text"
                )

            # Advanced configuration
            with st.expander("Advanced Configuration", expanded=False):
                frequency_penalty = st.slider(
                    "Frequency Penalty",
                    min_value=-2.0,
                    max_value=2.0,
                    value=config.get("frequency_penalty", 0.0),
                    step=0.1,
                    help="Positive values penalize tokens proportionally to how often they've appeared"
                )

                sample_method = st.selectbox(
                    "Sampling Method",
                    options=["top_k", "top_p", "beam", "greedy"],
                    index=0,
                    help="Method used to sample tokens from the model's output distribution"
                )

                if sample_method == "top_k":
                    top_k = st.slider(
                        "Top K",
                        min_value=1,
                        max_value=100,
                        value=config.get("top_k", 40),
                        help="Only sample from the top K most likely tokens"
                    )
                elif sample_method == "beam":
                    num_beams = st.slider(
                        "Number of Beams",
                        min_value=1,
                        max_value=10,
                        value=config.get("num_beams", 4),
                        help="Number of beams for beam search"
                    )

            # Save configuration button
            if st.button("Save Configuration", use_container_width=True):
                # Update configuration
                new_config = {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty
                }

                # Add sampling method specifics
                if sample_method == "top_k":
                    new_config["top_k"] = top_k
                elif sample_method == "beam":
                    new_config["num_beams"] = num_beams

                new_config["sample_method"] = sample_method

                # Update session state
                st.session_state.model_config = new_config

                st.success("Configuration saved!")
                logger.info(
                    f"Configuration updated for model {model_name}: {new_config}")
        else:
            st.info("Please select a model first to configure its parameters.")

    with tabs[2]:
        # Model information
        if "active_model" in st.session_state:
            display_model_info(st.session_state.active_model)

            # If we have usage statistics
            if "model_stats" in st.session_state:
                display_model_statistics(st.session_state.model_stats)
        elif selected_model:
            display_model_info(selected_model)
        else:
            st.info("Select a model to view its information.")

    # Reset model button (only show if a model is loaded)
    if "active_model" in st.session_state:
        if st.button("Unload Model", type="secondary"):
            if "active_model" in st.session_state:
                del st.session_state.active_model
            if "model_config" in st.session_state:
                del st.session_state.model_config

            st.success("Model unloaded successfully.")
            logger.info("Model unloaded")
            st.rerun()


if __name__ == "__main__":
    # For testing the page in isolation
    st.set_page_config(
        page_title="Model Selection - Insurance LLM Framework",
        page_icon="🤖",
        layout="wide"
    )

    # Initialize session state for testing
    if "active_model" not in st.session_state:
        st.session_state.active_model = None

    render()
