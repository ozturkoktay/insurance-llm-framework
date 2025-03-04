"""
Prompt Engineering Page for the Insurance LLM Framework.

This module provides the interface for creating, editing, and managing prompt templates.
"""

import streamlit as st
import logging
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

from ui.components.prompt_templates import (
    template_selector,
    display_template_info,
    template_editor,
    template_preview
)

logger = logging.getLogger(__name__)

def render():
    """Render the prompt engineering page."""
    st.title("✏️ Prompt Engineering")

    with st.expander("ℹ️ About Prompt Engineering", expanded=False):
        st.markdown("""
        This page allows you to create, edit, and manage prompt templates for insurance tasks.

        Crafting effective prompts for insurance tasks involves:
        - **Being specific** about the task and desired output format
        - **Providing context** relevant to insurance scenarios
        - **Using examples** of good responses (few-shot learning)
        - **Specifying constraints** like regulatory requirements

        - **Zero-shot**: Direct instructions without examples
        - **Few-shot**: Including examples of desired inputs and outputs
        - **Chain-of-thought**: Guiding the model through reasoning steps
        - **Self-consistency**: Having the model generate multiple solutions
        """)

    tabs = st.tabs(
        ["Browse Templates", "Create/Edit Template", "Template Information"])

    if "active_model" not in st.session_state or not st.session_state.active_model:
        st.warning(
            "⚠️ No model is currently loaded. Templates will only be saved but not tested.")

    with tabs[0]:
        st.subheader("Browse Prompt Templates")

        selected_template = template_selector()

        if selected_template:

            template_preview(selected_template)

            col1, col2, col3 = st.columns([1, 1, 1])

            with col1:
                if st.button("📝 Edit Template", use_container_width=True):
                    st.session_state.editing_template = selected_template
                    st.rerun()

            with col2:
                if st.button("📋 Use Template", use_container_width=True):
                    st.session_state.active_template = selected_template
                    st.success(
                        f"Template '{selected_template['name']}' set as active")
                    logger.info(
                        f"Template '{selected_template['name']}' set as active")

            with col3:

                if st.button("🔄 Clone Template", use_container_width=True):

                    new_template = selected_template.copy()
                    new_template["name"] = f"{selected_template['name']} (Copy)"
                    new_template["id"] = f"{selected_template['id']}_copy_{datetime.now().strftime('%Y%m%d%H%M%S')}"

                    st.session_state.editing_template = new_template
                    st.rerun()

    with tabs[1]:

        editing = False
        template_to_edit = None

        if "editing_template" in st.session_state and st.session_state.editing_template:
            editing = True
            template_to_edit = st.session_state.editing_template
            st.subheader(f"Edit Template: {template_to_edit['name']}")
        else:
            st.subheader("Create New Template")

        template_editor(template_to_edit)

    with tabs[2]:
        st.subheader("Template Information")

        selected_template_info = None

        if "active_template" in st.session_state and st.session_state.active_template:
            selected_template_info = st.session_state.active_template

        elif selected_template:
            selected_template_info = selected_template

        if selected_template_info:
            display_template_info(selected_template_info)
        else:
            st.info(
                "Select a template to view its information or set an active template.")

    with st.sidebar:
        if "active_template" in st.session_state and st.session_state.active_template:
            st.markdown("### Active Template")
            active_template = st.session_state.active_template
            st.info(f"Currently using: **{active_template['name']}**")

            st.caption(f"Format: {active_template['format']}")
            st.caption(f"Task: {active_template['task_type']}")

            if st.button("Clear Active Template"):
                st.session_state.active_template = None
                st.success("Active template cleared")
                st.rerun()

if __name__ == "__main__":

    st.set_page_config(
        page_title="Prompt Engineering - Insurance LLM Framework",
        page_icon="✏️",
        layout="wide"
    )

    if "active_model" not in st.session_state:
        st.session_state.active_model = "llama2-7b"

    if "active_template" not in st.session_state:
        st.session_state.active_template = None

    render()
