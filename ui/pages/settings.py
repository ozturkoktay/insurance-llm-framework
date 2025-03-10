"""
Settings Page for the Insurance LLM Framework.

This module provides the interface for configuring application settings.
"""

import streamlit as st
import logging
import os
import json
from typing import Dict, Any, Optional, List
import pandas as pd

logger = logging.getLogger(__name__)

def render():
    """Render the settings page."""
    st.title("⚙️ Settings")

    with st.expander("ℹ️ About Settings", expanded=False):
        st.markdown("""
        This page allows you to configure global settings for the Insurance LLM Framework.

        - **API Keys**: Configure API keys for commercial models
        - **Model Settings**: Set default parameters for models
        - **UI Preferences**: Customize the user interface
        - **File Paths**: Configure directories for data and saved files
        - **Cache Settings**: Manage model and tokenizer caches
        """)

    tabs = st.tabs(["API Settings", "Model Settings",
                   "UI Settings", "Advanced Settings"])

    with tabs[0]:
        st.subheader("API Settings")

        st.markdown("""
        Configure API keys for accessing commercial LLM services.
        These keys are stored securely in your session and not sent to any external servers.
        """)

        with st.form("api_keys_form"):

            openai_key = st.text_input(
                "OpenAI API Key",
                type="password",
                value=st.session_state.get("openai_api_key", ""),
                help="Required for GPT models"
            )

            anthropic_key = st.text_input(
                "Anthropic API Key",
                type="password",
                value=st.session_state.get("anthropic_api_key", ""),
                help="Required for Claude models"
            )

            cohere_key = st.text_input(
                "Cohere API Key",
                type="password",
                value=st.session_state.get("cohere_api_key", ""),
                help="Required for Cohere models"
            )

            hf_token = st.text_input(
                "HuggingFace API Token",
                type="password",
                value=st.session_state.get("huggingface_api_token", ""),
                help="Required for accessing gated HuggingFace models"
            )

            submitted = st.form_submit_button(
                "Save API Keys", use_container_width=True)

            if submitted:

                st.session_state.openai_api_key = openai_key
                st.session_state.anthropic_api_key = anthropic_key
                st.session_state.cohere_api_key = cohere_key
                st.session_state.huggingface_api_token = hf_token

                logger.info("API keys updated")
                st.success("API keys saved successfully")

        with st.expander("Custom API Endpoints", expanded=False):
            st.markdown("""
            Configure custom endpoints for API services.
            This is useful for self-hosted deployments or alternative API providers.
            """)

            with st.form("api_endpoints_form"):

                openai_endpoint = st.text_input(
                    "OpenAI API Endpoint",
                    value=st.session_state.get(
                        "openai_api_endpoint", "https://api.openai.com/v1"),
                    help="Default: https://api.openai.com/v1"
                )

                anthropic_endpoint = st.text_input(
                    "Anthropic API Endpoint",
                    value=st.session_state.get(
                        "anthropic_api_endpoint", "https://api.anthropic.com"),
                    help="Default: https://api.anthropic.com"
                )

                endpoints_submitted = st.form_submit_button(
                    "Save API Endpoints", use_container_width=True)

                if endpoints_submitted:

                    st.session_state.openai_api_endpoint = openai_endpoint
                    st.session_state.anthropic_api_endpoint = anthropic_endpoint

                    logger.info("API endpoints updated")
                    st.success("API endpoints saved successfully")

    with tabs[1]:
        st.subheader("Model Settings")

        st.markdown("""
        Configure default parameters for language models.
        These settings will be used as defaults when loading models.
        """)

        st.markdown("### Default Model Parameters")

        col1, col2 = st.columns([1, 1])

        with col1:
            default_temperature = st.slider(
                "Default Temperature",
                min_value=0.0,
                max_value=2.0,
                value=st.session_state.get("default_temperature", 0.7),
                step=0.1,
                help="Default temperature for generation"
            )

            default_top_p = st.slider(
                "Default Top P",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("default_top_p", 1.0),
                step=0.05,
                help="Default top_p for nucleus sampling"
            )

        with col2:
            default_max_tokens = st.slider(
                "Default Max Tokens",
                min_value=50,
                max_value=4000,
                value=st.session_state.get("default_max_tokens", 1000),
                step=50,
                help="Default maximum tokens for generation"
            )

            default_presence_penalty = st.slider(
                "Default Presence Penalty",
                min_value=-2.0,
                max_value=2.0,
                value=st.session_state.get("default_presence_penalty", 0.0),
                step=0.1,
                help="Default presence penalty"
            )

        if st.button("Save Default Parameters", use_container_width=True):

            st.session_state.default_temperature = default_temperature
            st.session_state.default_top_p = default_top_p
            st.session_state.default_max_tokens = default_max_tokens
            st.session_state.default_presence_penalty = default_presence_penalty

            logger.info("Default model parameters updated")
            st.success("Default parameters saved successfully")

        st.markdown("### Model Cache Settings")

        cache_dir = st.text_input(
            "Model Cache Directory",
            value=st.session_state.get(
                "cache_dir", os.path.join(os.getcwd(), "cache", "models")),
            help="Directory to store cached models"
        )

        max_cache_size = st.slider(
            "Max Cache Size (GB)",
            min_value=1,
            max_value=100,
            value=st.session_state.get("max_cache_size", 10),
            step=1,
            help="Maximum size of the model cache"
        )

        clear_cache = st.checkbox(
            "Clear Model Cache on Exit",
            value=st.session_state.get("clear_cache_on_exit", False),
            help="Clear the model cache when the application exits"
        )

        if st.button("Save Cache Settings", use_container_width=True):

            st.session_state.cache_dir = cache_dir
            st.session_state.max_cache_size = max_cache_size
            st.session_state.clear_cache_on_exit = clear_cache

            os.makedirs(cache_dir, exist_ok=True)

            logger.info(
                f"Model cache settings updated: {cache_dir}, {max_cache_size}GB")
            st.success("Cache settings saved successfully")

        if st.button("Clear Cache Now", type="secondary"):
            st.info("In a production app, this would clear the model cache.")

    with tabs[2]:
        st.subheader("UI Settings")

        st.markdown("""
        Configure user interface settings for the Insurance LLM Framework.
        These settings control how the application appears and behaves.
        """)

        st.markdown("### Theme Settings")

        with st.form("theme_settings_form"):
            theme_options = ["Light", "Dark", "Auto"]
            selected_theme = st.selectbox(
                "Theme Mode",
                options=theme_options,
                index=theme_options.index(
                    st.session_state.get("theme", "Auto"))
            )

            accent_color_options = ["Blue", "Green", "Orange", "Red", "Purple"]
            selected_accent = st.selectbox(
                "Accent Color",
                options=accent_color_options,
                index=accent_color_options.index(
                    st.session_state.get("accent_color", "Blue"))
            )

            font_size_options = ["Small", "Medium", "Large"]
            selected_font_size = st.selectbox(
                "Font Size",
                options=font_size_options,
                index=font_size_options.index(
                    st.session_state.get("font_size", "Medium"))
            )

            theme_submitted = st.form_submit_button(
                "Save Theme Settings", use_container_width=True)

            if theme_submitted:

                st.session_state.theme = selected_theme
                st.session_state.accent_color = selected_accent
                st.session_state.font_size = selected_font_size

                logger.info(
                    f"Theme settings updated: {selected_theme}, {selected_accent}, {selected_font_size}")
                st.success("Theme settings saved successfully")
                st.info(
                    "Theme changes will take effect after restarting the application.")

        st.markdown("### Layout Settings")

        with st.form("layout_settings_form"):
            layout_options = ["Wide", "Centered"]
            selected_layout = st.selectbox(
                "Layout Mode",
                options=layout_options,
                index=layout_options.index(
                    st.session_state.get("layout", "Wide"))
            )

            sidebar_options = ["Always expanded",
                               "Auto-collapse", "Collapsed by default"]
            selected_sidebar = st.selectbox(
                "Sidebar Behavior",
                options=sidebar_options,
                index=sidebar_options.index(st.session_state.get(
                    "sidebar_behavior", "Always expanded"))
            )

            st.markdown("#### Display Settings")

            show_tooltips = st.checkbox(
                "Show Tooltips",
                value=st.session_state.get("show_tooltips", True),
                help="Show helpful tooltips throughout the application"
            )

            show_animations = st.checkbox(
                "Show Animations",
                value=st.session_state.get("show_animations", True),
                help="Enable UI animations"
            )

            layout_submitted = st.form_submit_button(
                "Save Layout Settings", use_container_width=True)

            if layout_submitted:

                st.session_state.layout = selected_layout
                st.session_state.sidebar_behavior = selected_sidebar
                st.session_state.show_tooltips = show_tooltips
                st.session_state.show_animations = show_animations

                logger.info(
                    f"Layout settings updated: {selected_layout}, {selected_sidebar}")
                st.success("Layout settings saved successfully")
                st.info(
                    "Some layout changes will take effect after restarting the application.")

    with tabs[3]:
        st.subheader("Advanced Settings")

        st.markdown("""
        Configure advanced settings for the Insurance LLM Framework.
        These settings are intended for technical users and developers.
        """)

        st.markdown("### Logging Settings")

        log_level_options = ["DEBUG", "INFO", "WARNING", "ERROR"]
        selected_log_level = st.selectbox(
            "Log Level",
            options=log_level_options,
            index=log_level_options.index(
                st.session_state.get("log_level", "INFO"))
        )

        log_file = st.text_input(
            "Log File Path",
            value=st.session_state.get(
                "log_file", os.path.join(os.getcwd(), "logs", "app.log")),
            help="Path to the log file"
        )

        if st.button("Save Logging Settings", use_container_width=True):

            st.session_state.log_level = selected_log_level
            st.session_state.log_file = log_file

            log_dir = os.path.dirname(log_file)
            os.makedirs(log_dir, exist_ok=True)

            logger.info(
                f"Logging settings updated: {selected_log_level}, {log_file}")
            st.success("Logging settings saved successfully")

        st.markdown("### Export/Import Settings")

        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Export Settings", use_container_width=True):

                settings = {}

                settings["openai_api_endpoint"] = st.session_state.get(
                    "openai_api_endpoint", "https://api.openai.com/v1")
                settings["anthropic_api_endpoint"] = st.session_state.get(
                    "anthropic_api_endpoint", "https://api.anthropic.com")

                settings["default_temperature"] = st.session_state.get(
                    "default_temperature", 0.7)
                settings["default_top_p"] = st.session_state.get(
                    "default_top_p", 1.0)
                settings["default_max_tokens"] = st.session_state.get(
                    "default_max_tokens", 1000)
                settings["default_presence_penalty"] = st.session_state.get(
                    "default_presence_penalty", 0.0)
                settings["cache_dir"] = st.session_state.get(
                    "cache_dir", os.path.join(os.getcwd(), "cache", "models"))
                settings["max_cache_size"] = st.session_state.get(
                    "max_cache_size", 10)
                settings["clear_cache_on_exit"] = st.session_state.get(
                    "clear_cache_on_exit", False)

                settings["theme"] = st.session_state.get("theme", "Auto")
                settings["accent_color"] = st.session_state.get(
                    "accent_color", "Blue")
                settings["font_size"] = st.session_state.get(
                    "font_size", "Medium")
                settings["layout"] = st.session_state.get("layout", "Wide")
                settings["sidebar_behavior"] = st.session_state.get(
                    "sidebar_behavior", "Always expanded")
                settings["show_tooltips"] = st.session_state.get(
                    "show_tooltips", True)
                settings["show_animations"] = st.session_state.get(
                    "show_animations", True)

                settings["log_level"] = st.session_state.get(
                    "log_level", "INFO")
                settings["log_file"] = st.session_state.get(
                    "log_file", os.path.join(os.getcwd(), "logs", "app.log"))

                settings_json = json.dumps(settings, indent=2)

                st.download_button(
                    label="Download Settings",
                    data=settings_json,
                    file_name="insurance_llm_framework_settings.json",
                    mime="application/json"
                )

        with col2:

            uploaded_file = st.file_uploader(
                "Import Settings",
                type=["json"],
                help="Upload a settings file"
            )

            if uploaded_file is not None:
                try:

                    settings = json.load(uploaded_file)

                    for key, value in settings.items():
                        st.session_state[key] = value

                    logger.info("Settings imported from file")
                    st.success("Settings imported successfully")
                    st.info(
                        "Some settings may require restarting the application to take effect.")
                except Exception as e:
                    st.error(f"Error importing settings: {str(e)}")

        st.markdown("### Danger Zone")

        with st.expander("Reset All Settings", expanded=False):
            st.warning(
                "This will reset all settings to their default values. This action cannot be undone.")

            if st.button("Reset All Settings", type="primary"):

                settings_to_reset = [

                    "openai_api_key", "anthropic_api_key", "cohere_api_key", "huggingface_api_token",
                    "openai_api_endpoint", "anthropic_api_endpoint",

                    "default_temperature", "default_top_p", "default_max_tokens", "default_presence_penalty",
                    "cache_dir", "max_cache_size", "clear_cache_on_exit",

                    "theme", "accent_color", "font_size", "layout", "sidebar_behavior",
                    "show_tooltips", "show_animations",

                    "log_level", "log_file"
                ]

                for key in settings_to_reset:
                    if key in st.session_state:
                        del st.session_state[key]

                logger.info("All settings reset to defaults")
                st.success(
                    "All settings have been reset to their default values")
                st.info(
                    "Some changes may require restarting the application to take effect.")

if __name__ == "__main__":

    st.set_page_config(
        page_title="Settings - Insurance LLM Framework",
        page_icon="⚙️",
        layout="wide"
    )

    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = ""

    render()
