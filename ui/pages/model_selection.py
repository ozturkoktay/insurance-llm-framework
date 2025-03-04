"""
Model Selection Page for the Insurance LLM Framework.

This module provides the interface for selecting and configuring language models.
"""

import streamlit as st
import logging
import os
from typing import Dict, Any, Optional, List
import time

from ui.components.model_selector import model_selector, display_model_info
from ui.components.metrics_display import display_model_statistics

logger = logging.getLogger(__name__)

def render():
    """Render the model selection page."""
    st.title("🤖 Model Selection")

    with st.expander("ℹ️ About Model Selection", expanded=False):
        st.markdown("""
        This page allows you to select and configure language models for your insurance tasks.

        - Browse and select from available local and remote models
        - Configure model parameters like temperature and max tokens
        - View model capabilities and recommendations for insurance tasks
        - Load multiple models for comparison
        """)

    if "active_model" in st.session_state and st.session_state.active_model:
        with st.sidebar:
            st.markdown("### Active Model")
            st.info(f"Currently using: **{st.session_state.active_model}**")

            if "model_config" in st.session_state:
                st.caption(
                    f"Temperature: {st.session_state.model_config.get('temperature', 'N/A')}")
                st.caption(
                    f"Max Tokens: {st.session_state.model_config.get('max_tokens', 'N/A')}")

    tabs = st.tabs(
        ["Model Selection", "Model Configuration", "Model Information"])

    with tabs[0]:

        selected_model = model_selector()

        if selected_model:
            if st.button("Load Selected Model", use_container_width=True):
                with st.spinner(f"Loading {selected_model}..."):
                    try:

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

                        st.rerun()
                    except Exception as e:
                        st.error(f"Error loading model: {str(e)}")
                        logger.error(
                            f"Error loading model {selected_model}: {str(e)}")

    with tabs[1]:
        st.subheader("Model Configuration")

        model_name = None
        if "active_model" in st.session_state:
            model_name = st.session_state.active_model
        elif selected_model:
            model_name = selected_model

        if model_name:
            st.markdown(f"Configure parameters for **{model_name}**")

            config = {}
            if "model_config" in st.session_state:
                config = st.session_state.model_config

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

            if st.button("Save Configuration", use_container_width=True):

                new_config = {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": top_p,
                    "presence_penalty": presence_penalty,
                    "frequency_penalty": frequency_penalty
                }

                if sample_method == "top_k":
                    new_config["top_k"] = top_k
                elif sample_method == "beam":
                    new_config["num_beams"] = num_beams

                new_config["sample_method"] = sample_method

                st.session_state.model_config = new_config

                st.success("Configuration saved!")
                logger.info(
                    f"Configuration updated for model {model_name}: {new_config}")
        else:
            st.info("Please select a model first to configure its parameters.")

    with tabs[2]:

        if "active_model" in st.session_state:
            display_model_info(st.session_state.active_model)

            if "model_stats" in st.session_state:
                display_model_statistics(st.session_state.model_stats)
        elif selected_model:
            display_model_info(selected_model)
        else:
            st.info("Select a model to view its information.")

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

    st.set_page_config(
        page_title="Model Selection - Insurance LLM Framework",
        page_icon="🤖",
        layout="wide"
    )

    if "active_model" not in st.session_state:
        st.session_state.active_model = None

    render()
